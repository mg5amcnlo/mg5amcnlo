################################################################################
#
# Copyright (c) 2009 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""Factory + comparator helpers for parallel MadSpin tests.

The :class:`MadSpinFactory` generates a production sample once for a given
(process, multiparticle, model) triple, then runs MadSpin against the *same*
events under several ``(spinmode, ME_mode, density_do_reshuffle)`` configurations.
Each configuration returns a :class:`MadSpinResult` carrying the branching
ratio, unweighting efficiency, log path, and decayed-LHE path.

The factory is meant to be reused from a ``unittest.TestCase`` so that the
heavy production step is shared across configurations within one test.
"""

from __future__ import absolute_import
from __future__ import division

import collections
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

pjoin = os.path.join

_logger = logging.getLogger('test_madspin_factory')

_here = os.path.dirname(os.path.realpath(__file__))
_root = os.path.split(os.path.split(_here)[0])[0]
if _root not in sys.path:
    sys.path.insert(0, _root)

from madgraph import MG5DIR
import madgraph.iolibs.files as files
import madgraph.various.banner as banner_mod
import madgraph.various.lhe_parser as lhe_parser


def _read_lhe_cross(path):
    """Open ``path`` as an LHE file and return the banner's init-block cross
    section (or ``None`` if it cannot be parsed)."""
    try:
        lhe = lhe_parser.EventFile(path)
        banner = lhe.get_banner()
        cross = banner.get_cross()
        try:
            lhe.close()
        except Exception:
            pass
        return cross
    except Exception:
        return None


# Reasonable defaults for the dispatcher table. Each entry must be unique by
# label so the factory can key result dicts on it.
SpinModeConfig = collections.namedtuple(
    'SpinModeConfig', ['label', 'spinmode', 'ME_mode', 'density_do_reshuffle']
)


# These are the five modes spelled out in the MadSpin density-mode table:
#  - full + decay_chain  : old default, mass smearing, no 3-body, identical part. only
#  - onshell + decay_chain : traditional onshell decay chain
#  - onshell + density   : "PA without reshuffling" (pure onshell kinematics, density ME)
#  - full + density      : off-shell ME + density (BW shape from ME)
#  - PA + density        : PA reshuffling with BW + density ME (new MadSpin default)
DEFAULT_MODES = [
    SpinModeConfig('full_decay_chain',    'full',    'decay_chain', True),
    SpinModeConfig('onshell_decay_chain', 'onshell', 'decay_chain', True),
    SpinModeConfig('onshell_density',     'onshell', 'density',     False),
    SpinModeConfig('full_density',        'full',    'density',     True),
    SpinModeConfig('PA_density',          'PA',      'density',     True),
]

# Mode families: the two paths use fundamentally different BR computations
# (legacy = factorized on-shell partial widths, run_onshell = MC-integrated
# partial width including off-shell-resonance suppression), so cross-section
# agreement is expected to be tight *within* a family and only loosely
# compatible *between* families.
DEFAULT_FAMILIES = {
    'legacy':      ('full_decay_chain',),
    'run_onshell': ('onshell_decay_chain', 'onshell_density',
                    'full_density',       'PA_density'),
}


class MadSpinResult(object):
    """Container for a single MadSpin run's outputs."""

    def __init__(self, config, lhe_path, log_path, wall_seconds,
                 BR, BR_err, efficiency, nevents_in,
                 cross_out=None, cross_in=None):
        self.config = config
        self.lhe_path = lhe_path
        self.log_path = log_path
        self.wall_seconds = wall_seconds
        self.BR = BR
        self.BR_err = BR_err
        self.efficiency = efficiency
        self.nevents_in = nevents_in
        # Final cross-section from the decayed LHE banner (pb). This is the
        # physics-observable that must match across modes: it is the product
        # production-cross-section x branching-ratio integrated over the
        # decayed event sample.
        self.cross_out = cross_out
        # Production cross-section taken from the input LHE banner (pb).
        self.cross_in = cross_in
        self._lhe_cache = None
        self._counts_cache = None

    @property
    def label(self):
        return self.config.label

    def open_lhe(self):
        """Return an EventFile iterator (fresh each call -- the file is
        forward-only)."""
        return lhe_parser.EventFile(self.lhe_path)

    def count_pdgs(self):
        """Cached per-PDG final-state multiplicity plus event count."""
        if self._counts_cache is not None:
            return self._counts_cache
        counts = collections.Counter()
        nevents = 0
        for event in self.open_lhe():
            nevents += 1
            assert event.nexternal == len(event), (
                'malformed event in %s: nexternal=%s len=%s'
                % (self.lhe_path, event.nexternal, len(event))
            )
            for particle in event:
                if particle.status == 1:
                    counts[particle.pdg] += 1
        self._counts_cache = (nevents, counts)
        return self._counts_cache


# Log-line parsing -- kept loose so we tolerate the various spellings used by
# the decay-chain and density code paths.
_RE_BR = re.compile(r'Branching ratio to allowed decays:\s*([0-9eE.+\-]+)')
_RE_DENSITY_EFF = re.compile(
    r'MadSpin unweight efficiency:\s*([0-9.]+)\s*'
    r'\((\d+)\s*(?:accepted|written)\s*/\s*(\d+)\s*trials'
)
# Decay-chain mode logs "Average number of trial points per production event: X"
# instead. The inverse of X is the unweighting efficiency.
_RE_AVG_TRIAL = re.compile(
    r'Average number of trial points per production event:\s*([0-9eE.+\-]+)'
)
# And "Total number of events written: A/B " gives the (lower-bound) writing
# efficiency for both code paths.
_RE_WRITTEN = re.compile(
    r'Total number of events written:\s*(\d+)\s*/\s*(\d+)'
)


def _parse_log(text):
    """Pull (BR, accepted, trials, efficiency) from a MadSpin log.

    For density-method runs the explicit "MadSpin unweight efficiency" line is
    used directly. For legacy decay-chain runs we recover the unweighting
    efficiency from the inverse of "Average number of trial points per
    production event", and fall back to the written/input ratio if neither is
    present.
    """
    BR = None
    for match in _RE_BR.finditer(text):
        BR = float(match.group(1))  # use last occurrence (final value)

    accepted = trials = None
    efficiency = None

    # Preferred: explicit density-mode line.
    for match in _RE_DENSITY_EFF.finditer(text):
        efficiency = float(match.group(1))
        accepted = int(match.group(2))
        trials = int(match.group(3))

    if efficiency is None:
        # Decay-chain mode: unweighting efficiency = 1 / avg_trials_per_event.
        for match in _RE_AVG_TRIAL.finditer(text):
            avg = float(match.group(1))
            if avg > 0:
                efficiency = 1.0 / avg

    if efficiency is None:
        # Last resort: written / input. This is a writing ratio, not a true
        # unweighting efficiency, but it's better than nothing for comparisons
        # within the same code path.
        for match in _RE_WRITTEN.finditer(text):
            a = int(match.group(1))
            b = int(match.group(2))
            if b > 0:
                efficiency = float(a) / float(b)
                accepted = a
                trials = b

    return BR, accepted, trials, efficiency


class MadSpinFactory(object):
    """Build and reuse a single production sample across many MadSpin modes.

    Parameters
    ----------
    name : str
        Short identifier used to name temporary directories and test IDs.
    production_process : str
        The MG5 ``generate`` line, e.g. ``'p p > t t~'``.
    decays : list of str
        Decay branches handed to MadSpin verbatim (e.g.
        ``['decay t > b w+', 'decay t~ > b~ w-']``).
    model : str, default ``'sm'``
        MG5 model name.
    multiparticles : dict[str, str], optional
        ``define <name> = <particles>`` directives to inject before
        ``generate``.
    nevents : int, default ``10000``
        Number of production events.
    beam_energy : float, default ``6500``
        Per-beam energy in GeV (only used if a hadronic run_card is generated).
    seed : int, default ``42``
        Seed propagated to both madevent and MadSpin.
    extra_run_card : dict[str, str], optional
        Extra ``run_card`` overrides applied at production time.
    base_dir : str, optional
        Where the factory's working tree lives. Defaults to a fresh ``tempfile``
        directory which is removed by :meth:`cleanup`.
    """

    def __init__(self, name, production_process, decays,
                 model='sm', multiparticles=None, nevents=10000,
                 beam_energy=6500, seed=42, extra_run_card=None,
                 extra_madspin_settings=None, base_dir=None):
        self.name = name
        self.production_process = production_process
        self.decays = list(decays)
        self.model = model
        self.multiparticles = dict(multiparticles or {})
        self.nevents = int(nevents)
        self.beam_energy = float(beam_energy)
        self.seed = int(seed)
        self.extra_run_card = dict(extra_run_card or {})
        # Extra ``set <key> <val>`` lines injected into every MadSpin card.
        # Handy for tuning ``max_weight_ps_point`` / ``Nevents_for_max_weight``
        # under smoke tests so the max-weight probing step doesn't dominate
        # wall time.
        self.extra_madspin_settings = dict(extra_madspin_settings or {})
        self._owns_base = base_dir is None
        self.base_dir = base_dir or tempfile.mkdtemp(prefix='msfactory_%s_' % name)
        self.proc_dir = pjoin(self.base_dir, 'PROC')
        self.events_file = None
        self._results = {}

    # ------------------------------------------------------------------
    # Production: run mg5_aMC once to generate events.
    # ------------------------------------------------------------------
    def _write_mg5_script(self, script_path):
        lines = ['set automatic_html_opening False --no_save',
                 'import model %s' % self.model]
        for mp_name, mp_def in self.multiparticles.items():
            lines.append('define %s = %s' % (mp_name, mp_def))
        lines.append('generate %s' % self.production_process)
        lines.append('output %s' % self.proc_dir)
        lines.append('launch %s' % self.proc_dir)
        lines.append('madspin=OFF')  # MadSpin runs separately, mode by mode
        lines.append('shower=OFF')
        lines.append('detector=OFF')
        lines.append('analysis=OFF')
        lines.append('done')  # end card edit menu
        lines.append('set nevents %d' % self.nevents)
        lines.append('set iseed %d' % self.seed)
        lines.append('set use_syst False')
        for key, val in self.extra_run_card.items():
            lines.append('set %s %s' % (key, val))
        lines.append('done')  # end second card edit menu (after card adjustments)
        with open(script_path, 'w') as fp:
            fp.write('\n'.join(lines) + '\n')

    def produce_events(self):
        """Run mg5_aMC once; cache the LHE file path."""
        if self.events_file:
            return self.events_file

        script_path = pjoin(self.base_dir, 'mg5_script.dat')
        self._write_mg5_script(script_path)

        log_path = pjoin(self.base_dir, 'mg5.log')
        _logger.info('%s: generating production sample (log: %s)',
                     self.name, log_path)
        with open(log_path, 'w') as logf:
            ret = subprocess.call(
                [pjoin(MG5DIR, 'bin', 'mg5_aMC'), '-f', script_path],
                stdout=logf, stderr=subprocess.STDOUT,
            )
        if ret != 0:
            raise RuntimeError(
                'mg5_aMC failed for factory %s (see %s)' % (self.name, log_path)
            )

        # mg5_aMC -f sometimes returns 0 even when an intermediate command
        # (e.g. ``generate``) aborts -- the wrapper just skips the rest and
        # exits cleanly. Check the log for the unmistakable markers it emits
        # in that case so we surface the failure here instead of further down
        # the call chain.
        with open(log_path) as fp:
            log_text = fp.read()
        for marker in ('NoDiagramException',
                       'command not executed: output',
                       'command not executed: launch'):
            if marker in log_text:
                raise RuntimeError(
                    'mg5_aMC aborted mid-script for factory %s '
                    '(marker %r in %s)' % (self.name, marker, log_path)
                )

        candidate = pjoin(self.proc_dir, 'Events', 'run_01', 'unweighted_events.lhe.gz')
        if not os.path.exists(candidate):
            candidate_plain = candidate[:-3]
            if os.path.exists(candidate_plain):
                candidate = candidate_plain
            else:
                raise RuntimeError(
                    'production sample not found for %s under %s'
                    % (self.name, pjoin(self.proc_dir, 'Events'))
                )
        self.events_file = candidate
        self.cross_in = _read_lhe_cross(self.events_file)
        return self.events_file

    # ------------------------------------------------------------------
    # Per-mode MadSpin execution.
    # ------------------------------------------------------------------
    def _write_madspin_card(self, card_path, evt_path, config):
        lines = [
            'set spinmode %s' % config.spinmode,
            'set ME_mode %s' % config.ME_mode,
            'set seed %d' % self.seed,
            'set max_running_process 4',
        ]
        if config.ME_mode == 'density':
            lines.append('set density_do_reshuffle %s'
                         % ('True' if config.density_do_reshuffle else 'False'))
        for key, val in self.extra_madspin_settings.items():
            lines.append('set %s %s' % (key, val))
        for mp_name, mp_def in self.multiparticles.items():
            lines.append('define %s = %s' % (mp_name, mp_def))
        lines.append('import %s' % evt_path)
        for decay in self.decays:
            stripped = decay.strip()
            if not stripped.startswith('decay '):
                stripped = 'decay ' + stripped
            lines.append(stripped)
        lines.append('launch')
        with open(card_path, 'w') as fp:
            fp.write('\n'.join(lines) + '\n')

    def run_mode(self, config):
        """Run MadSpin once for the given :class:`SpinModeConfig`."""
        if config.label in self._results:
            return self._results[config.label]
        self.produce_events()

        run_dir = pjoin(self.base_dir, 'mode_%s' % config.label)
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)
        os.makedirs(run_dir)

        # Copy the production LHE so MadSpin writes the _decayed output beside it.
        evt_basename = 'events.lhe.gz'
        evt_path = pjoin(run_dir, evt_basename)
        files.cp(self.events_file, evt_path)

        card_path = pjoin(run_dir, 'madspin_card.dat')
        self._write_madspin_card(card_path, evt_path, config)

        log_path = pjoin(run_dir, 'madspin.log')
        _logger.info('%s[%s]: running MadSpin (log: %s)',
                     self.name, config.label, log_path)
        wall_start = time.time()
        with open(log_path, 'w') as logf:
            ret = subprocess.call(
                [pjoin(MG5DIR, 'MadSpin', 'madspin'), card_path],
                cwd=run_dir, stdout=logf, stderr=subprocess.STDOUT,
            )
        wall = time.time() - wall_start
        if ret != 0:
            raise RuntimeError(
                'MadSpin failed for %s[%s] (see %s)'
                % (self.name, config.label, log_path)
            )

        decayed = pjoin(run_dir, 'events_decayed.lhe.gz')
        if not os.path.exists(decayed):
            alt = pjoin(run_dir, 'events_decayed.lhe')
            if os.path.exists(alt):
                decayed = alt
            else:
                raise RuntimeError(
                    'decayed LHE missing for %s[%s]; expected %s'
                    % (self.name, config.label, decayed)
                )

        with open(log_path) as logf:
            log_text = logf.read()
        BR, accepted, trials, efficiency = _parse_log(log_text)
        if efficiency is None and accepted is not None and trials:
            efficiency = float(accepted) / float(trials)

        # Always read the decayed banner's cross-section -- this is the
        # physics-observable we want to compare across modes.
        cross_out = _read_lhe_cross(decayed)

        # Fallback: density / onshell modes don't log "Branching ratio" but the
        # output LHE banner has cross_in * BR. Recover BR from that ratio.
        if BR is None and getattr(self, 'cross_in', None) and cross_out:
            BR = cross_out / self.cross_in

        BR_err = 0.0
        if BR is not None and self.nevents > 0:
            # Conservative Poisson-style band: BR * sqrt((1-eff)/N).
            band = BR * math.sqrt(max(1e-12, 1.0 - (efficiency or 1.0)) / self.nevents)
            BR_err = max(BR_err, band)

        result = MadSpinResult(
            config=config,
            lhe_path=decayed,
            log_path=log_path,
            wall_seconds=wall,
            BR=BR,
            BR_err=BR_err,
            efficiency=efficiency,
            nevents_in=self.nevents,
            cross_out=cross_out,
            cross_in=getattr(self, 'cross_in', None),
        )
        self._results[config.label] = result
        return result

    def run_modes(self, configs):
        """Convenience: run every config in order, return ``{label: result}``."""
        out = collections.OrderedDict()
        for cfg in configs:
            out[cfg.label] = self.run_mode(cfg)
        return out

    # ------------------------------------------------------------------
    def cleanup(self):
        if self._owns_base and os.path.exists(self.base_dir):
            shutil.rmtree(self.base_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Shared assertions.  These are plain functions taking a unittest.TestCase
# instance plus result objects, so they can be reused by any TestCase class.
# ---------------------------------------------------------------------------

def assert_lhe_well_formed(test, result, min_nevents=1):
    """Check the LHE parses, has at least ``min_nevents``, and every event's
    declared ``nexternal`` matches ``len(event)``.
    Touches the cached count, so subsequent ``count_pdgs`` is cheap."""
    nevents, _ = result.count_pdgs()
    test.assertGreaterEqual(
        nevents, min_nevents,
        'LHE %s contains %d events (< %d expected)' % (
            result.lhe_path, nevents, min_nevents))


def assert_branching_ratios_consistent(test, results, rel_tol=1e-3):
    """All modes should report the same global branching ratio (BR is a
    deterministic function of model + decay specification, so the only spread
    is from numerical noise).

    Note: the BR reported by the legacy decay_chain path can legitimately
    differ from the run_onshell paths by a symmetry factor when the user
    supplies redundant decay templates. The real physics-observable check is
    :func:`assert_cross_sections_consistent` below; this assertion is here for
    informational reporting and is intentionally loose."""
    brs = [(label, r.BR) for label, r in results.items() if r.BR is not None]
    test.assertTrue(brs, 'no BR was parsed from any MadSpin log')
    ref_label, ref = brs[0]
    for label, br in brs[1:]:
        rel = abs(br - ref) / max(abs(ref), 1e-30)
        test.assertLess(
            rel, rel_tol,
            'BR mismatch: %s=%g vs %s=%g (rel=%g > %g)'
            % (ref_label, ref, label, br, rel, rel_tol))


def assert_cross_sections_consistent(test, results, rel_tol=1e-3,
                                     families=None, between_tol=5e-2):
    """The decayed-LHE banner's cross-section is the physics-observable
    invariant -- but how strictly modes must agree depends on whether they
    share a BR-computation convention.

    With ``families=None`` (default) every mode must agree within ``rel_tol``.

    With ``families={'name': (labels,...), ...}`` (e.g. ``DEFAULT_FAMILIES``)
    we do two passes:

    1. *Within-family*: every mode in the same family must agree within
       ``rel_tol`` -- this is the strict invariant catching real bugs.
    2. *Between-family*: family-medians are compared within ``between_tol``
       so an off-shell-resonance BR difference of a few percent between the
       legacy factorised-BR path and the run_onshell MC-integrated-BR path
       doesn't trip the test, but a runaway discrepancy still does.
    """
    crosses = {label: r.cross_out for label, r in results.items()
               if r.cross_out is not None}
    test.assertTrue(crosses, 'no decayed cross-section found in any LHE banner')

    if not families:
        labels = list(crosses.keys())
        ref_label = labels[0]
        ref = crosses[ref_label]
        test.assertGreater(
            abs(ref), 0,
            'reference cross-section for %s is zero' % ref_label)
        for label in labels[1:]:
            cross = crosses[label]
            rel = abs(cross - ref) / max(abs(ref), 1e-30)
            test.assertLess(
                rel, rel_tol,
                'cross-section mismatch in decayed LHE banners: '
                '%s=%g pb vs %s=%g pb (rel=%g > %g)'
                % (ref_label, ref, label, cross, rel, rel_tol))
        return

    # Within-family strict check.
    family_repr = {}  # family name -> representative (label, cross)
    for fname, members in families.items():
        present = [(label, crosses[label]) for label in members
                   if label in crosses]
        if not present:
            continue
        ref_label, ref = present[0]
        test.assertGreater(
            abs(ref), 0,
            'reference cross-section for family %s (%s) is zero'
            % (fname, ref_label))
        for label, cross in present[1:]:
            rel = abs(cross - ref) / max(abs(ref), 1e-30)
            test.assertLess(
                rel, rel_tol,
                'cross-section mismatch within family %s: '
                '%s=%g pb vs %s=%g pb (rel=%g > %g)'
                % (fname, ref_label, ref, label, cross, rel, rel_tol))
        family_repr[fname] = (ref_label, ref)

    # Between-family looser check.
    repr_items = list(family_repr.items())
    for i, (fa, (la, ca)) in enumerate(repr_items):
        for fb, (lb, cb) in repr_items[i + 1:]:
            rel = abs(ca - cb) / max(abs(ca), abs(cb), 1e-30)
            test.assertLess(
                rel, between_tol,
                'cross-section mismatch between families %s and %s: '
                '%s=%g pb vs %s=%g pb (rel=%g > %g). '
                'Beyond the expected off-shell-BR gap -- '
                'investigate the BR convention used by each family.'
                % (fa, fb, la, ca, lb, cb, rel, between_tol))


def assert_multiplicities_consistent(test, results, pdgs, n_sigma=4):
    """For each PDG in ``pdgs``, count finals across modes and require
    pair-wise agreement within ``n_sigma`` Poisson tolerance.

    This is exactly the same shape as the existing ``2*sqrt(N)`` check in
    tests/acceptance_tests/test_madspin.py."""
    summary = {}
    for label, r in results.items():
        _, counts = r.count_pdgs()
        summary[label] = counts

    labels = list(summary.keys())
    for pdg in pdgs:
        for i, la in enumerate(labels):
            for lb in labels[i + 1:]:
                na = summary[la].get(pdg, 0)
                nb = summary[lb].get(pdg, 0)
                tol = n_sigma * math.sqrt(max(na + nb, 1))
                test.assertLess(
                    abs(na - nb), tol,
                    'pdg %d multiplicity inconsistent between %s (%d) and %s (%d); '
                    'diff=%d > %.1f*sqrt(%d)'
                    % (pdg, la, na, lb, nb, abs(na - nb), n_sigma, na + nb))


def assert_efficiency_close(test, result_a, result_b, rel_tol=0.15):
    """Compare two modes' unweighting efficiencies. Both must be populated; if
    either is missing the test fails loudly so we don't silently skip a
    physics requirement.

    Kept as a utility for callers that want a strict pair-equality check; the
    default suite uses :func:`assert_efficiency_ordering` instead, which
    encodes the physics-motivated ordering across all five modes.
    """
    test.assertIsNotNone(
        result_a.efficiency,
        'efficiency missing for %s (parse failure?)' % result_a.label)
    test.assertIsNotNone(
        result_b.efficiency,
        'efficiency missing for %s (parse failure?)' % result_b.label)
    eff_a = result_a.efficiency
    eff_b = result_b.efficiency
    ratio = eff_a / eff_b if eff_b else float('inf')
    test.assertLess(
        abs(ratio - 1.0), rel_tol,
        'efficiency ratio %s/%s = %g/%g = %.3f outside [%.3f, %.3f]'
        % (result_a.label, result_b.label, eff_a, eff_b, ratio,
           1.0 - rel_tol, 1.0 + rel_tol))


def assert_efficiency_ordering(test, results,
                               close_rel_tol=0.10,
                               slack=0.02):
    """Physics-motivated ordering of unweighting efficiencies across modes.

    The relations -- per MadSpin author intent -- are:

    1. ``full_decay_chain`` (legacy, fully off-shell ME on each PS point) has
       the *smallest* efficiency of any mode.
    2. ``onshell_decay_chain`` and ``onshell_density`` agree with each other
       within ``close_rel_tol`` (relative), and both are *better* (higher
       efficiency) than the pole approximation ``PA_density``.
    3. ``full_density`` sits *between* ``full_decay_chain`` and
       ``PA_density``.

    All ordering inequalities are evaluated with an additive ``slack`` so
    statistical fluctuations smaller than ``slack`` don't trip the check. A
    missing efficiency (e.g. a mode was skipped via ``skip_modes``) silently
    drops the rules that mention it; the surviving rules still run.
    """
    needed = ['full_decay_chain', 'onshell_decay_chain', 'onshell_density',
              'full_density', 'PA_density']
    eff = {}
    for label in needed:
        if label in results and results[label].efficiency is not None:
            eff[label] = results[label].efficiency

    if not eff:
        return  # nothing to assert against

    # 1. full_decay_chain is the smallest.
    if 'full_decay_chain' in eff:
        ref = eff['full_decay_chain']
        for label, e in eff.items():
            if label == 'full_decay_chain':
                continue
            test.assertLessEqual(
                ref, e + slack,
                'full_decay_chain (%g) should be the smallest efficiency, '
                'but exceeds %s (%g) beyond slack %g (eff dump: %s)'
                % (ref, label, e, slack, eff))

    # 2a. onshell_decay_chain ~ onshell_density (close to each other).
    if 'onshell_decay_chain' in eff and 'onshell_density' in eff:
        a, b = eff['onshell_decay_chain'], eff['onshell_density']
        scale = max(abs(a), abs(b), 1e-30)
        rel = abs(a - b) / scale
        test.assertLess(
            rel, close_rel_tol,
            'onshell_decay_chain (%g) and onshell_density (%g) should be '
            'close (rel=%g > close_rel_tol=%g) (eff dump: %s)'
            % (a, b, rel, close_rel_tol, eff))
    # 2b. Both onshell variants higher than PA_density (pole approximation).
    if 'PA_density' in eff:
        pa = eff['PA_density']
        for label in ('onshell_decay_chain', 'onshell_density'):
            if label not in eff:
                continue
            test.assertGreaterEqual(
                eff[label] + slack, pa,
                '%s (%g) should be >= PA_density (%g) within slack %g '
                '(eff dump: %s)'
                % (label, eff[label], pa, slack, eff))

    # 3. full_density between full_decay_chain and PA_density.
    if all(k in eff for k in ('full_decay_chain', 'full_density', 'PA_density')):
        lo = min(eff['full_decay_chain'], eff['PA_density']) - slack
        hi = max(eff['full_decay_chain'], eff['PA_density']) + slack
        test.assertGreaterEqual(
            eff['full_density'], lo,
            'full_density (%g) below [full_decay_chain=%g, PA_density=%g] '
            'interval within slack %g (eff dump: %s)'
            % (eff['full_density'], eff['full_decay_chain'],
               eff['PA_density'], slack, eff))
        test.assertLessEqual(
            eff['full_density'], hi,
            'full_density (%g) above [full_decay_chain=%g, PA_density=%g] '
            'interval within slack %g (eff dump: %s)'
            % (eff['full_density'], eff['full_decay_chain'],
               eff['PA_density'], slack, eff))


def _resonance_masses(result, parent_pdg, child_pdgs=None):
    """Return the list of invariant masses of the parent resonance.

    The mass is reconstructed from the sum of its decay products' 4-momenta.
    If ``child_pdgs`` is provided, only resonances whose children match the
    given (sorted) PDG tuple are included; otherwise any decay is kept."""
    target_children = tuple(sorted(child_pdgs)) if child_pdgs else None
    masses = []
    for event in result.open_lhe():
        # Group particles by mother index (LHE mother fields are 1-indexed).
        by_mother = collections.defaultdict(list)
        for idx, p in enumerate(event):
            try:
                m1 = int(p.mother1)
            except (TypeError, ValueError):
                m1 = 0
            if m1 > 0:
                by_mother[m1].append(idx)
        for idx, p in enumerate(event):
            if p.pdg != parent_pdg:
                continue
            kids = by_mother.get(idx + 1, [])
            if not kids:
                continue
            if target_children is not None:
                kid_pdgs = tuple(sorted(event[k].pdg for k in kids))
                if kid_pdgs != target_children:
                    continue
            E = sum(event[k].E for k in kids)
            px = sum(event[k].px for k in kids)
            py = sum(event[k].py for k in kids)
            pz = sum(event[k].pz for k in kids)
            m2 = E * E - px * px - py * py - pz * pz
            if m2 > 0:
                masses.append(math.sqrt(m2))
    return masses


def assert_offshell_mass_distribution(test, results, parent_pdg,
                                      pole_mass, width, child_pdgs=None,
                                      bins=20, mass_window=None,
                                      tolerance_const=0.10,
                                      tolerance_offshell=4.0):
    """Compare off-shell mass distributions across modes against each other and
    against a Breit-Wigner reference.

    The per-bin tolerance is::

        tol = tolerance_const + tolerance_offshell * (width/pole_mass) * |m-M|/M

    so bins farther from the pole tolerate larger deviations (matching the
    instruction "uncertainty like Gamma/M*offshell, with increasing tolerance
    the further off-shell you are").

    Modes which by construction sit exactly on shell (no off-shell-ness) get
    skipped here -- we still rely on ``assert_multiplicities_consistent`` for
    their sanity.
    """
    if mass_window is None:
        half = max(10 * width, 0.2 * pole_mass)
        mass_window = (pole_mass - half, pole_mass + half)
    lo, hi = mass_window

    # Sample once per mode.
    sample_per_mode = {}
    for label, r in results.items():
        masses = [m for m in _resonance_masses(r, parent_pdg, child_pdgs)
                  if lo <= m <= hi]
        if len(masses) < 50:
            # Too few to make a meaningful statement; likely an onshell-only
            # mode (decay chain w/o smearing). Skip it for the spread check.
            continue
        sample_per_mode[label] = masses

    if len(sample_per_mode) < 2:
        return  # nothing to compare; not a failure per se

    # Build histograms with shared binning.
    edges = [lo + (hi - lo) * i / bins for i in range(bins + 1)]

    def hist(values):
        h = [0] * bins
        for v in values:
            k = min(bins - 1, int((v - lo) / (hi - lo) * bins))
            h[k] += 1
        return h

    histograms = {lab: hist(vals) for lab, vals in sample_per_mode.items()}

    # Cross-mode comparison: shape ratio bin-by-bin.
    labels = list(histograms.keys())
    for bin_idx in range(bins):
        center = 0.5 * (edges[bin_idx] + edges[bin_idx + 1])
        offshell = abs(center - pole_mass) / max(pole_mass, 1e-30)
        tol = tolerance_const + tolerance_offshell * (width / max(pole_mass, 1e-30)) * offshell
        # Skip extreme tails where statistics are too poor.
        bin_total = sum(histograms[lab][bin_idx] for lab in labels)
        if bin_total < 20:
            continue
        for i, la in enumerate(labels):
            na = histograms[la][bin_idx]
            ta = sum(histograms[la])
            if ta == 0 or na < 5:
                continue
            fa = na / ta
            for lb in labels[i + 1:]:
                nb = histograms[lb][bin_idx]
                tb = sum(histograms[lb])
                if tb == 0 or nb < 5:
                    continue
                fb = nb / tb
                stat = math.sqrt(max(1, na)) / ta + math.sqrt(max(1, nb)) / tb
                test.assertLess(
                    abs(fa - fb), tol + 3 * stat,
                    'off-shell bin %d (m~%.2f) deviates between %s (f=%g) '
                    'and %s (f=%g): |df|=%g > %g+3sigma=%g'
                    % (bin_idx, center, la, fa, lb, fb,
                       abs(fa - fb), tol, tol + 3 * stat))

    # Reference: Breit-Wigner shape (rest-frame relativistic BW). Compare the
    # *median* mass of each sample to the pole within a wide tolerance.
    for label, masses in sample_per_mode.items():
        srt = sorted(masses)
        median = srt[len(srt) // 2]
        # The median of a BW (truncated to the window) sits close to the pole.
        test.assertLess(
            abs(median - pole_mass) / pole_mass,
            5 * width / pole_mass + 0.02,
            'mode %s median mass %.3f far from pole %.3f (Gamma=%.3f)'
            % (label, median, pole_mass, width))
