################################################################################
#
# Copyright (c) 2009 The MadGraph7 Development team and Contributors
#
# This file is a part of the MadGraph7 project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph7 license which should accompany this
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""Systematic MG7 cross-section regression tests.

For every process listed in ``check_xsec_processes_reference.json`` this
module generates a ``test_<section>_<id>_mg7`` method on a single
``unittest.TestCase`` (so the whole suite is discoverable and launchable
through ``tests/test_manager.py``, exactly like the other mg7 acceptance
tests). Each test:

  1. builds an MG5 interface and ``generate``s the process (the ``_quark`` /
     ``_anti_quark`` merged-flavor particles are used directly from the sm
     model),
  2. ``output mg7``s it,
  3. edits ``Cards/run_card.toml`` -- it pins the configuration the references
     were generated with: the fixed scale (mu = 91.188 GeV, e_cm = 13000 GeV),
     since the template now defaults to the dynamical HT/2 scale, and the PDF
     set (``NNPDF23_lo_as_0130_qed``), so the references stay valid no matter
     which set the template defaults to; it also sets the event count and, for
     the hadronic tt~ decays, neutralises the jet cuts (see CLAUDE.md). An
     entry may add cuts (``"cuts": {"<cut>.<bound>": value}``), set
     ``[phasespace] cut_decays`` (``"cut_decays": true``) and switch MadNIS
     off (``"madnis": false``, see below),
  4. runs ``bin/generate_events -f`` and reads the cross-section from the
     madspace ``Events/*/info.json`` (``process.mean`` / ``process.error``),
  5. asserts the relative difference to the reference stays within the
     tolerance plus ``MG7_XSEC_NSIGMA`` times the combined MC error,
  6. asserts the overweight tail of the unweighted sample (the fraction of
     |sigma| carried by events with |w| > <|w|>, read from
     ``Events/*/events.lhe.gz``) stays below ``MG7_XSEC_OVERWEIGHT_FACTOR``
     times the run card's ``max_overweight_truncation``. This piggybacks on
     the events the cross-section check generates anyway, so the regression
     fixed in PR #191 (MadNIS runs carrying ~3-5x the intended tail) is caught
     at no extra CI cost.

The source of truth is ``check_xsec_processes_reference.json`` (mirrors the
table in CLAUDE.md, produced with fixed scale and 1M events).

MadNIS training (``[madnis] enable = "auto"``: always on for four or more
final-state particles, on for three when the survey spread is large) is ~95%
of the run time of the processes that use it, independently of the event
count. Half of those entries therefore carry ``"madnis": false`` and run on
the VEGAS multichannel sampling alone, including the inclusive ``p p > ...``
ones (whose reference is the sum of the sub-channel references); the other
half keeps the default, so both samplers stay covered.

Four knobs are read from the environment so the CI can dial them without
touching the code:

  * ``MG7_XSEC_TOLERANCE`` -- allowed relative difference on top of the MC
                             error (default 0.01, 1%)
  * ``MG7_XSEC_NSIGMA``    -- how many combined MC errors (this run and the
                             reference, in quadrature) are allowed on top of
                             the tolerance (default 3)
  * ``MG7_XSEC_EVENTS``    -- events per run (default 10000; the reference
                             used 1M, reduced here to keep the CI affordable:
                             the MC error is then ~0.25%, hence the
                             MG7_XSEC_NSIGMA allowance: some processes sit
                             up to ~0.7% off their reference even at 100k)
  * ``MG7_XSEC_OVERWEIGHT_FACTOR`` -- allowed overweight tail, in units of the
                             run card's ``max_overweight_truncation``
                             (default 2.5; the tail is a sum over a handful
                             of events at 10k, so it fluctuates)

Run everything locally with e.g.::

    ./tests/test_manager.py test_.*_mg7 -pA -t0 -l INFO

or a single section::

    ./tests/test_manager.py test_ttx_decay_.*_mg7 -pA -t0

Each test self-skips (rather than failing) when the mg7 runtime stack
(madspace + LHAPDF + the NNPDF23 grid) is not available.
"""

from __future__ import absolute_import
from __future__ import division

import glob
import gzip
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
import unittest

import madgraph.interface.master_interface as MGCmd
import madgraph.various.misc as misc

pjoin = os.path.join
_HERE = os.path.dirname(os.path.abspath(__file__))
_REFERENCE = pjoin(_HERE, 'check_xsec_processes_reference.json')

# PDF set the reference cross-sections were generated with. The test pins it
# into every run_card.toml (see _edit_run_card) instead of relying on the
# template default, so that changing the default PDF of the mg7 run_card does
# not silently invalidate all ~40 reference values. It is also the set
# _require_mg7_runtime requires to be installed.
_REFERENCE_PDF = 'NNPDF23_lo_as_0130_qed'

# Environment-tunable knobs (see module docstring). Kept as module globals so
# the dynamically generated test methods pick up the CI-provided values.
_TOLERANCE = float(os.environ.get('MG7_XSEC_TOLERANCE', 0.01))
_EVENTS = int(os.environ.get('MG7_XSEC_EVENTS', 10000))
_NSIGMA = float(os.environ.get('MG7_XSEC_NSIGMA', 3))
_OVERWEIGHT_FACTOR = float(os.environ.get('MG7_XSEC_OVERWEIGHT_FACTOR', 2.5))

# Optional: when set (by the CI workflow), one JSON result record per process
# is written here so a later job can build a GitHub Actions job summary out of
# them. The per-process run directory (and its log) is torn down at the end of
# each test, so anything the summary needs -- including the tail of the log on
# failure -- must be captured into this record before tearDown() runs.
_RESULTS_DIR = os.environ.get('MG7_XSEC_RESULTS_DIR')


# madspace renders a live-updating terminal UI (cursor moves, line clears,
# colors) into the log; strip that so the tail is readable once scraped into
# a plain-text summary.
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]|\x1b[()][A-Za-z0-9]|\x1b[78]')


def _tail(path, n=60):
    """Return the last *n* (ANSI-stripped) lines of *path*, best effort."""
    try:
        with open(path) as f:
            lines = f.readlines()
        return _ANSI_RE.sub('', ''.join(lines[-n:])).strip()
    except Exception:
        return ''


def _require_mg7_runtime(test):
    """``skipTest`` (on *test*) when the mg7 runtime stack -- madspace, LHAPDF
    and the PDF set the references were generated with -- is unavailable.

    Only a skip gate: the run itself must locate LHAPDF from the configuration
    the way a user's ``bin/generate_events`` does, with no environment help.
    """
    try:
        import madspace
        has_mg7 = hasattr(madspace, 'ChannelEventGenerator')
    except ImportError:
        has_mg7 = False
    if not has_mg7:
        test.skipTest('mg7 runtime stack (madspace) unavailable')

    if not misc.resolve_lhapdf().find_set(_REFERENCE_PDF):
        test.skipTest('%s LHAPDF data not found (set the lhapdf option or '
                      '$LHAPDF_DATA_PATH)' % _REFERENCE_PDF)


def _edit_run_card(toml_path, events, disable_jet_cuts, cuts=None,
                   cut_decays=None, madnis=None):
    """Set the event count, (optionally) neutralise the jet cuts, add the
    entry's extra cuts and set its cut_decays and MadNIS switch.

    The reference cross-sections were produced with FIXED scales
    (mu = 91.188 GeV) and with the NNPDF23_lo_as_0130_qed PDF set. The
    run_card.toml template now defaults to the dynamical HT/2 scale instead,
    and its default PDF set is free to change, so these replacements are what
    pins the configuration back to the one the references were generated with.
    They are load-bearing: drop them and every reference value below goes
    stale. The PDF pin in particular decouples the references from the
    template default -- ``_require_mg7_runtime`` already guarantees the pinned
    set is the one present on disk."""
    t = open(toml_path).read()
    t = t.replace('fixed_ren_scale = false', 'fixed_ren_scale = true')
    t = t.replace('fixed_fact_scale = false', 'fixed_fact_scale = true')
    # one set per beam (pdf1/pdf2); a card from before the split has pdf
    t, pinned = re.subn(r'(?m)^(pdf[12]?) = ".*"$',
                        r'\1 = "%s"' % _REFERENCE_PDF, t)
    if not pinned:
        raise RuntimeError('no [beam] pdf entry to pin in %s' % toml_path)
    t = re.sub(r'events = \d+', 'events = %d' % events, t)
    if disable_jet_cuts:
        # jet cuts must be disabled for the hadronic tt~ decay processes to
        # match the reference (see CLAUDE.md).
        t = re.sub(r'jet-pt\.min\s*=.*', 'jet-pt.min = 0.0', t)
        t = re.sub(r'jet-eta_abs\.max\s*=.*', 'jet-eta_abs.max = 100.0', t)
        t = re.sub(r'jet-delta_r\.min\s*=.*', 'jet-delta_r.min = 0.0', t)
        t = re.sub(r'jet-lepton-delta_r\.min\s*=.*', 'jet-lepton-delta_r.min = 0.0', t)
    if cuts:
        # right under the [cuts] header ([histograms] has keys of the same
        # form, e.g. its own sqrt_s.min)
        lines = ''.join('%s = %r\n' % (key, float(value))
                        for key, value in cuts.items())
        t, count = re.subn(r'(?m)^\[cuts\]\n', lambda m: m.group(0) + lines, t)
        assert count == 1, 'no [cuts] section in %s' % toml_path
    if cut_decays is not None:
        t, count = re.subn(r'(?m)^cut_decays = \w+',
                           'cut_decays = %s' % str(bool(cut_decays)).lower(), t)
        assert count == 1, 'no cut_decays in %s' % toml_path
    if madnis is not None:
        # the first `enable` after the [madnis] header ([vegas] and
        # [systematics] have one too)
        t, count = re.subn(r'(?m)(^\[madnis\]\n(?:(?!^\[).*\n)*?)enable = .*$',
                           lambda m: m.group(1) + 'enable = %s'
                           % str(bool(madnis)).lower(), t)
        assert count == 1, 'no [madnis] enable in %s' % toml_path
    open(toml_path, 'w').write(t)


def _max_overweight_truncation(toml_path):
    """The ``max_overweight_truncation`` the run was made with."""
    m = re.search(r'(?m)^max_overweight_truncation\s*=\s*([^\s#]+)',
                  open(toml_path).read())
    return float(m.group(1))


def _overweight_fraction(lhe_path):
    """Fraction of |sigma| carried by the events with |w| > <|w|>.

    An unweighted event has the weight of its channel's cap, unless its raw
    weight was above that cap: it is then kept with the weight ratio. So this
    is the quantity ``max_overweight_truncation`` bounds (per channel, hence
    also summed)."""
    weights = []
    with gzip.open(lhe_path, 'rt') as f:
        in_event = False
        for line in f:
            if in_event:
                # first line of the event block: NUP IDPRUP XWGTUP ...
                weights.append(abs(float(line.split()[2])))
                in_event = False
            elif line.startswith('<event'):
                in_event = True
    total = sum(weights)
    if not total:
        return 0.
    mean = total / len(weights)
    # the regular events all carry the same weight, slightly below the mean;
    # the relative margin only guards against the rounding of the LHE output
    return sum(w for w in weights if w > mean * (1 + 1e-7)) / total


with open(_REFERENCE) as _f:
    _REF = json.load(_f)


class CheckXsecProcessesMG7Test(unittest.TestCase):
    """Regression test: MG7 (madspace) reproduces the recorded reference
    cross-sections within ``MG7_XSEC_TOLERANCE``. One ``test_*`` method per
    process is attached below from ``check_xsec_processes_reference.json``."""

    def setUp(self):
        self.path = tempfile.mkdtemp(prefix='check_xsec_mg7_')

    def tearDown(self):
        shutil.rmtree(self.path, ignore_errors=True)

    def _record_result(self, entry, section, status, got=None, err=None,
                        message=None, overweight=None, overweight_max=None):
        """Persist a machine-readable record of this process' outcome (used
        by the CI workflow to build a job summary). No-op unless
        ``MG7_XSEC_RESULTS_DIR`` is set."""
        if not _RESULTS_DIR:
            return
        try:
            os.makedirs(_RESULTS_DIR, exist_ok=True)
            record = {
                'section': section,
                'id': entry['id'],
                'process': entry['process'],
                'status': status,
                'cross': got,
                'error': err,
                'ref_cross': entry['cross'],
                'ref_error': entry.get('error'),
                'message': message,
                'overweight': overweight,
                'overweight_max': overweight_max,
            }
            out = pjoin(_RESULTS_DIR, '%s_%s.json' % (section, entry['id']))
            with open(out, 'w') as f:
                json.dump(record, f)
        except Exception:
            # Recording the result must never mask the actual test outcome.
            pass

    def _check_process(self, entry, defines, section):
        """Generate + integrate a single process and assert its cross-section
        matches the reference within tolerance."""
        try:
            self._run_and_check(entry, defines, section)
        except unittest.SkipTest:
            raise
        except AssertionError:
            # Already recorded (with a precise message) by _run_and_check.
            raise
        except Exception:
            self._record_result(entry, section, 'fail',
                                 message=traceback.format_exc())
            raise

    def _run_and_check(self, entry, defines, section):
        _require_mg7_runtime(self)

        run_dir = pjoin(self.path, entry['id'])
        mg = MGCmd.MasterCmd()
        mg.no_notification()
        mg.exec_cmd('set automatic_html_opening False --no_save')
        for d in defines:
            mg.exec_cmd(d)
        mg.exec_cmd('generate %s' % entry['process'])
        mg.exec_cmd('output mg7 %s' % run_dir)

        toml = pjoin(run_dir, 'Cards', 'run_card.toml')
        _edit_run_card(toml, _EVENTS, entry.get('disable_jet_cuts', False),
                       entry.get('cuts'), entry.get('cut_decays'),
                       entry.get('madnis'))

        log = pjoin(run_dir, 'mg7_gen.log')
        with open(log, 'w') as logfh:
            ret = subprocess.call(
                [sys.executable, pjoin(run_dir, 'bin', 'generate_events'), '-f'],
                cwd=run_dir, stdout=logfh, stderr=subprocess.STDOUT)
        if ret != 0:
            message = ('mg7 generate_events failed (exit %d)\n\n%s'
                       % (ret, _tail(log)))
            self._record_result(entry, section, 'fail', message=message)
            self.fail('mg7 generate_events failed (see %s)' % log)

        infos = sorted(glob.glob(pjoin(run_dir, 'Events', '*', 'info.json')))
        if not infos:
            message = 'no mg7 info.json produced\n\n%s' % _tail(log)
            self._record_result(entry, section, 'fail', message=message)
            self.fail('no mg7 info.json under %s (see %s)' % (run_dir, log))
        with open(infos[-1]) as infofh:
            info = json.load(infofh)['process']
        got = float(info['mean'])
        err = float(info.get('error') or 0.0)

        # overweight tail of the unweighted sample (only the LHE output is
        # read; the npy formats are not what the CI runs)
        overweight = overweight_max = None
        lhe = pjoin(os.path.dirname(infos[-1]), 'events.lhe.gz')
        if os.path.exists(lhe):
            overweight = _overweight_fraction(lhe)
            overweight_max = _OVERWEIGHT_FACTOR * _max_overweight_truncation(toml)

        ref_x = entry['cross']
        # abs(ref_x): an interference has a negative cross section, and a
        # negative relative difference would always pass
        reldiff = abs(got - ref_x) / abs(ref_x) if ref_x else float('inf')
        # the tolerance covers genuine differences; the MC error of the run
        # (and of the reference) comes on top of it, so that fewer events do
        # not turn statistical fluctuations into failures
        sigma = math.sqrt(err ** 2 + (entry.get('error') or 0.0) ** 2)
        allowed = _TOLERANCE + (_NSIGMA * sigma / abs(ref_x) if ref_x else 0.0)
        xsec_ok = reldiff <= allowed
        overweight_ok = overweight is None or overweight <= overweight_max
        problems = []
        if not xsec_ok:
            # A cross-section was successfully obtained here, just outside
            # tolerance -- no need for the (noisy) log tail, the deviation
            # itself is the useful diagnostic.
            problems.append(
                '%s (%s): mg7 xsec %.6g +- %.3g pb differs from reference '
                '%.6g pb by %.3f%% (> %.3f%% = %.3f%% tolerance + %g sigma)'
                % (entry['id'], entry['process'], got, err, ref_x,
                   100 * reldiff, 100 * allowed, 100 * _TOLERANCE, _NSIGMA))
        if not overweight_ok:
            problems.append(
                '%s (%s): %.3f%% of the cross section is carried by '
                'overweight events (|w| > <|w|>), above the %.3f%% allowed '
                '(%g x max_overweight_truncation)'
                % (entry['id'], entry['process'], 100 * overweight,
                   100 * overweight_max, _OVERWEIGHT_FACTOR))
        message = '\n'.join(problems) or None
        self._record_result(entry, section, 'fail' if problems else 'pass',
                             got=got, err=err, message=message,
                             overweight=overweight,
                             overweight_max=overweight_max)
        if problems:
            self.fail(message)


def _make_test(entry, defines, section):
    """Build a bound-method-shaped closure for a single reference entry."""
    def test(self):
        self._check_process(entry, defines, section)
    test.__doc__ = '%s: %s (ref %.6g pb)' % (
        entry['id'], entry['process'], entry['cross'])
    return test


# Attach one test_<section>_<id>_mg7 method per process. The section is part of
# the method name so the CI (and a developer) can select a single section with
# a regexp, e.g. test_manager.py test_ttx_decay_.*_mg7 -pA
for _section, _entries in _REF['sections'].items():
    for _entry in _entries:
        _name = 'test_%s_%s_mg7' % (_section, _entry['id'])
        setattr(CheckXsecProcessesMG7Test, _name,
                _make_test(_entry, _REF['defines'], _section))

if __name__ == '__main__':
    unittest.main()
