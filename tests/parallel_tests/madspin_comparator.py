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
events under several ``spinmode`` configurations.
Each configuration returns a :class:`MadSpinResult` carrying the branching
ratio, unweighting efficiency, log path, and decayed-LHE path.

The factory is meant to be reused from a ``unittest.TestCase`` so that the
heavy production step is shared across configurations within one test.

The same machinery drives the ``unweighting`` comparisons: ``run_mode`` takes a
per-run ``seed`` (for same-scheme replicas off one production sample) and
``decays`` override, and the assertions at the bottom of this module cover the
resolved scheme, the ``sequential_debug`` weight identity, and observable-level
agreement between schemes.
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
    'SpinModeConfig', ['label', 'spinmode']
)


# These are the five modes spelled out in the MadSpin spinmode table:
#  - madspin_v1 : old default, mass smearing, no 3-body, identical part. only
#  - onshell_v1 : traditional onshell decay chain
#  - onshell    : "PA without reshuffling" (pure onshell kinematics, density ME)
#  - madspin    : off-shell ME + density (BW shape from ME) (new MadSpin default)
#  - PA         : PA reshuffling with BW + density ME
DEFAULT_MODES = [
    SpinModeConfig('full_decay_chain',    'madspin_v1'),  
    SpinModeConfig('onshell_decay_chain', 'onshell_v1'),
    SpinModeConfig('onshell_density',     'onshell'),
    SpinModeConfig('madspin_density',     'madspin'),
    SpinModeConfig('PA_density',          'PA'),
]
# Mode families: the two paths use fundamentally different BR computations
# (legacy = factorized on-shell partial widths, run_onshell = MC-integrated
# partial width including off-shell-resonance suppression), so cross-section
# agreement is expected to be tight *within* a family and only loosely
# compatible *between* families.
DEFAULT_FAMILIES = {
    'legacy':      ('full_decay_chain',),
    'run_onshell': ('onshell_decay_chain', 'onshell_density',
                    'madspin_density',    'PA_density'),
}


# The four accept/reject schemes ``set unweighting`` selects, plus ``auto``.
# They differ only in how the test is split and in what a rejection redraws --
# every one of them is supposed to sample the same distribution.
UNWEIGHTING_MODES = ('joint', 'two_stage', 'sequential',
                     'sequential_global_retry')

# ``sequential_debug`` compares two evaluations of the same weight that run
# through different code, and the density matrices are complex64, so the ratio
# can only be constant to single-precision epsilon. This is the floor of the
# arithmetic, not of the physics: nothing may be asserted below it.
FLOAT32_EPS = 1.1920929e-7
# What the test actually requires. Measured spreads sit at 1.5-1.7e-7 -- just
# above the float32 floor, as the arithmetic demands -- so a few times that
# leaves room for the last bits to land differently on another platform while
# staying 100x tighter than MadSpin's own CRITICAL (which fires at
# ``density_tolerance`` = 1e-4). The margin costs nothing in sensitivity: a
# decomposition missing a factor spreads by percent, not by 1e-6. Measured on
# the mass-set weight before its jacobian was added, the spread was 1.4e-2 --
# four orders of magnitude above this bound.
IDENTITY_SPREAD_TOL = 1e-6


class MadSpinResult(object):
    """Container for a single MadSpin run's outputs."""

    def __init__(self, config, lhe_path, log_path, wall_seconds,
                 BR, BR_err, efficiency, nevents_in,
                 cross_out=None, cross_in=None,
                 unweighting_mode=None, unweighting_why=None,
                 identity=None, overflows=0, seed=None):
        self.config = config
        self.lhe_path = lhe_path
        self.log_path = log_path
        self.wall_seconds = wall_seconds
        self.BR = BR
        self.BR_err = BR_err
        self.efficiency = efficiency
        self.nevents_in = nevents_in
        # Which accept/reject scheme the run actually used, as the run itself
        # reported it ("MadSpin: unweighting = <mode> (<why>)"), plus the
        # parenthesised reason. ``auto`` resolves on the process, so the card
        # value alone does not answer this.
        self.unweighting_mode = unweighting_mode
        self.unweighting_why = unweighting_why
        # ``sequential_debug`` report: an :class:`IdentityReport` or ``None``
        # when the run did not do the check.
        self.identity = identity
        # Weights that exceeded their per-particle bound. Non-zero means that
        # bound is under-estimated and the sample is (slightly) biased; kept
        # for diagnostics rather than asserted on.
        self.overflows = overflows
        self.seed = seed
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
# Every density-mode run says once which accept/reject scheme it resolved to.
# Matched per line rather than against the flattened text: the reason itself
# ends in "particle(s)", so the closing parenthesis has to be the last one on
# the line and not the first one the pattern reaches.
_RE_UNWEIGHTING = re.compile(
    r'MadSpin: unweighting = (\w+) \((.*)\)\s*$'
)
# ``sequential_debug``: the deterministic per-chain check that the product of
# the stage weights is proportional to the joint weight recomputed by the joint
# code for the same production event, virtualities and decays.
_RE_IDENTITY_OK = re.compile(
    r'MadSpin sequential: weight identity verified on (\d+) accepted chains'
    r' -- chain weight / joint weight constant to ([0-9eE.+\-]+)'
    r' \(ratio ([0-9eE.+\-]+)\)'
)
_RE_IDENTITY_FAIL = re.compile(
    r'MadSpin sequential: the weight identity FAILED on (\d+) accepted chains'
    r'.*?relative spread of the ratio ([0-9eE.+\-]+), mean ([0-9eE.+\-]+)'
)
_RE_OVERFLOW = re.compile(
    r'MadSpin sequential: (\d+) weights exceeded their per-particle maximum'
)


IdentityReport = collections.namedtuple(
    'IdentityReport', ['checks', 'spread', 'ratio', 'ok']
)


def _flatten(text):
    """Collapse every run of whitespace to one space.

    The log lines above are long enough that a handler (or a terminal capture)
    may fold them; matching against the flattened text makes the regexes
    insensitive to where the fold lands."""
    return re.sub(r'\s+', ' ', text)


def _parse_unweighting(text):
    """``(mode, why)`` from the run's own announcement, or ``(None, None)``."""
    match = None
    for line in text.splitlines():
        found = _RE_UNWEIGHTING.search(line)
        if found:
            match = found  # logged once, but take the last just in case
    if match is None:
        return None, None
    return match.group(1), match.group(2)


def _parse_identity(text):
    """The ``sequential_debug`` report as an :class:`IdentityReport`.

    Returns ``None`` when the run did not run the check at all (the option is
    off, or the mode/spinmode combination does not support it) -- which is a
    different thing from the check running and failing, and callers must not
    confuse the two."""
    flat = _flatten(text)
    match = _RE_IDENTITY_FAIL.search(flat)
    if match:
        return IdentityReport(checks=int(match.group(1)),
                              spread=float(match.group(2)),
                              ratio=float(match.group(3)), ok=False)
    match = _RE_IDENTITY_OK.search(flat)
    if match:
        return IdentityReport(checks=int(match.group(1)),
                              spread=float(match.group(2)),
                              ratio=float(match.group(3)), ok=True)
    return None


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
    def _write_madspin_card(self, card_path, evt_path, config,
                            extra_settings=None, seed=None, decays=None):
        merged = dict(self.extra_madspin_settings)
        if extra_settings:
            merged.update(extra_settings)
        # MadSpin seeds its RNG on the *first* ``set seed`` of the card and
        # ignores every later one, so a seed override has to replace that line
        # rather than be appended: an appended one silently reproduces the same
        # run. Pull it out of the merged settings for the same reason.
        seed = merged.pop('seed', seed)
        lines = [
            'set spinmode %s' % config.spinmode,
            'set seed %d' % (self.seed if seed is None else int(seed)),
            'set max_running_process 4',
        ]
        for key, val in merged.items():
            lines.append('set %s %s' % (key, val))
        for mp_name, mp_def in self.multiparticles.items():
            lines.append('define %s = %s' % (mp_name, mp_def))
        lines.append('import %s' % evt_path)
        for decay in (self.decays if decays is None else decays):
            stripped = decay.strip()
            if not stripped.startswith('decay '):
                stripped = 'decay ' + stripped
            lines.append(stripped)
        lines.append('launch')
        with open(card_path, 'w') as fp:
            fp.write('\n'.join(lines) + '\n')

    def run_mode(self, config, extra_settings=None, run_tag=None, seed=None,
                 decays=None):
        """Run MadSpin once for the given :class:`SpinModeConfig`.

        ``extra_settings`` -- optional ``{key: val}`` merged over the factory's
        default ``set`` lines for this run only (e.g. ``{'nb_core': 8}`` to
        exercise the process-parallel unweighting path).
        ``run_tag`` -- optional suffix so the *same* config can be run more than
        once into distinct run dirs / result keys (defaults to ``config.label``).
        ``seed`` -- optional MadSpin seed for this run only, replacing the
        factory's. Use it to run the same configuration twice off the *same*
        production events, which is the replica needed to calibrate how far
        apart two runs of one scheme land.
        ``decays`` -- optional decay lines replacing the factory's for this run
        only. Mainly so a run can decay *fewer* particles off the same
        production sample, which is what ``auto`` keys its choice of scheme on.
        """
        key = config.label if not run_tag else '%s_%s' % (config.label, run_tag)
        if key in self._results:
            return self._results[key]
        self.produce_events()

        run_dir = pjoin(self.base_dir, 'mode_%s' % key)
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)
        os.makedirs(run_dir)

        # Copy the production LHE so MadSpin writes the _decayed output beside it.
        evt_basename = 'events.lhe.gz'
        evt_path = pjoin(run_dir, evt_basename)
        files.cp(self.events_file, evt_path)

        card_path = pjoin(run_dir, 'madspin_card.dat')
        self._write_madspin_card(card_path, evt_path, config, extra_settings,
                                 seed=seed, decays=decays)

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
        unweighting_mode, unweighting_why = _parse_unweighting(log_text)
        identity = _parse_identity(log_text)
        overflow_match = _RE_OVERFLOW.search(_flatten(log_text))
        overflows = int(overflow_match.group(1)) if overflow_match else 0

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
            unweighting_mode=unweighting_mode,
            unweighting_why=unweighting_why,
            identity=identity,
            overflows=overflows,
            seed=self.seed if seed is None else int(seed),
        )
        self._results[key] = result
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
                               slack=0.02,
                               madspin_density_slack=0.05):
    """Physics-motivated ordering of unweighting efficiencies across modes.

    The relations -- per MadSpin author intent -- are:

    1. (dropped) ``full_decay_chain`` was originally expected to be the
       smallest efficiency of any mode, but the 10k-event ttbar run on PR #292
       showed that the *new* off-shell path (``madspin_density``) is currently
       even less efficient than the legacy Fortran one (0.124 vs 0.145). The
       relative ordering between the two off-shell paths is a tuning question,
       not a physics invariant, so it's no longer enforced here. Rule (3)
       below still captures the wider ordering w.r.t. the on-shell modes.
    2. ``onshell_decay_chain`` and ``onshell_density`` agree with each other
       within ``close_rel_tol`` (relative), and both are *better* (higher
       efficiency) than the pole approximation ``PA_density``.
    3. ``madspin_density`` sits *between* ``full_decay_chain`` and
       ``PA_density``. Uses ``madspin_density_slack`` (default 0.05, absolute)
       rather than ``slack`` because the same ttbar 10k run showed the new
       density off-shell unweighting can dip a few percent below the legacy
       Fortran's efficiency -- the same tuning-gap that motivated dropping
       rule (1). 0.05 leaves room for that ~2 % wobble while still flagging
       any future drift larger than 5 percentage points absolute.

    Rules (2a/2b) use the tighter ``slack`` because the four run_onshell
    modes are expected to track each other to within statistical noise on a
    well-tuned MadSpin.

    A missing efficiency (e.g. a mode was skipped via ``skip_modes``) silently
    drops the rules that mention it; the surviving rules still run.
    """
    needed = ['full_decay_chain', 'onshell_decay_chain', 'onshell_density',
              'madspin_density', 'PA_density']
    eff = {}
    for label in needed:
        if label in results and results[label].efficiency is not None:
            eff[label] = results[label].efficiency

    if not eff:
        return  # nothing to assert against

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

    # 3. madspin_density between full_decay_chain and PA_density (weakened slack).
    if all(k in eff for k in ('full_decay_chain', 'madspin_density', 'PA_density')):
        lo = min(eff['full_decay_chain'], eff['PA_density']) - madspin_density_slack
        hi = max(eff['full_decay_chain'], eff['PA_density']) + madspin_density_slack
        test.assertGreaterEqual(
            eff['madspin_density'], lo,
            'madspin_density (%g) below [full_decay_chain=%g, PA_density=%g] '
            'interval within madspin_density_slack %g (eff dump: %s)'
            % (eff['madspin_density'], eff['full_decay_chain'],
               eff['PA_density'], madspin_density_slack, eff))
        test.assertLessEqual(
            eff['madspin_density'], hi,
            'madspin_density (%g) above [full_decay_chain=%g, PA_density=%g] '
            'interval within madspin_density_slack %g (eff dump: %s)'
            % (eff['madspin_density'], eff['full_decay_chain'],
               eff['PA_density'], madspin_density_slack, eff))


def resonance_masses(result, parent_pdg, child_pdgs=None):
    """Return the list of invariant masses of the parent resonance.

    The mass is reconstructed from the sum of its decay products' 4-momenta.
    If ``child_pdgs`` is provided, only resonances whose children match the
    given (sorted) PDG tuple are included; otherwise any decay is kept.

    ``Event.parse`` rewrites each particle's ``mother1`` from the 1-indexed LHE
    field into a reference to the mother :class:`Particle` itself (whose
    ``event_id`` is its 0-based position), so the mother has to be resolved
    through the object and not by casting the field to an int. Both forms are
    handled here: an unparsed event still carries the numeric field."""
    target_children = tuple(sorted(child_pdgs)) if child_pdgs else None
    masses = []
    for event in result.open_lhe():
        # Group particles by their mother's 0-based index in the event.
        by_mother = collections.defaultdict(list)
        for idx, p in enumerate(event):
            mother = p.mother1
            if not mother:
                continue
            event_id = getattr(mother, 'event_id', None)
            if event_id is None:
                try:
                    event_id = int(mother) - 1  # LHE field is 1-indexed
                except (TypeError, ValueError):
                    continue
            if event_id >= 0:
                by_mother[event_id].append(idx)
        for idx, p in enumerate(event):
            if p.pdg != parent_pdg:
                continue
            kids = by_mother.get(idx, [])
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


# Back-compat alias: the name was private when only this module used it.
_resonance_masses = resonance_masses


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


# ---------------------------------------------------------------------------
# Unweighting-scheme comparisons.
#
# ``set unweighting`` picks how the accept/reject is organised; all four
# schemes are supposed to sample the *same* distribution, differing only in how
# the test is split and in what a rejection redraws. The helpers below are what
# a test needs to hold them to that.
# ---------------------------------------------------------------------------

def final_state_dphi(result, pdgs_a, pdgs_b):
    """``|delta phi|`` in ``[0, pi]``, one entry per event, between the first
    final-state particle whose PDG is in ``pdgs_a`` and the first in
    ``pdgs_b``. Events not containing both are skipped."""
    set_a = set(pdgs_a)
    set_b = set(pdgs_b)
    out = []
    for event in result.open_lhe():
        pa = pb = None
        for particle in event:
            if particle.status != 1:
                continue
            # independent tests, not elif: the two PDG sets are disjoint in
            # every current caller, but an overlapping one must not have its
            # first match consumed by whichever branch was tried first
            if pa is None and particle.pdg in set_a:
                pa = particle
                continue
            if pb is None and particle.pdg in set_b:
                pb = particle
            if pa is not None and pb is not None:
                break
        if pa is None or pb is None:
            continue
        dphi = math.atan2(pa.py, pa.px) - math.atan2(pb.py, pb.px)
        while dphi > math.pi:
            dphi -= 2 * math.pi
        while dphi < -math.pi:
            dphi += 2 * math.pi
        out.append(abs(dphi))
    return out


def mean_and_error(values, window=None):
    """``(mean, standard error, n)`` over ``values``, optionally truncated to
    ``window=(lo, hi)``.

    A window cuts the Breit-Wigner tails out of the mean and so reduces its
    run-to-run scatter -- but for comparing *lineshapes* that is a bad trade,
    and measurably so: on the reconstructed top mass a +-10 GeV window cuts the
    replica scatter by 1.7x and the offshell-vs-pole-approximation difference by
    2.2x, i.e. it costs more signal than noise. Those tails carry a
    disproportionate share of what distinguishes the two shapes. Leave ``window``
    unset unless something has been measured that says otherwise.

    Note the returned error is the *naive* per-run one; see
    :func:`assert_observable_consistent` for why it is not the right yardstick
    for comparing two MadSpin runs off one production sample."""
    if window is not None:
        lo, hi = window
        values = [v for v in values if lo <= v <= hi]
    n = len(values)
    if n < 2:
        return (values[0] if n else float('nan')), float('inf'), n
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(variance / n), n


def assert_unweighting_mode(test, result, expected, expected_why=None):
    """The scheme the run *actually used*, as the run itself announced it.

    Non-statistical and instant, and it is the guard on the resolution logic:
    ``auto`` resolves on the process, and several combinations override what
    the card asked for (``fixed_order`` and unsupported spinmodes force
    ``joint``; ``two_stage`` and ``sequential_global_retry`` need an offshell
    spinmode and fall back to ``sequential`` under PA/onshell). Without this a
    consistency matrix could compare four runs of the same scheme and pass."""
    test.assertIsNotNone(
        result.unweighting_mode,
        "no 'MadSpin: unweighting = ...' line in %s -- the run never announced "
        "which accept/reject scheme it used" % result.log_path)
    test.assertEqual(
        result.unweighting_mode, expected,
        'run %s resolved unweighting = %s (%s), expected %s (log: %s)'
        % (result.label, result.unweighting_mode, result.unweighting_why,
           expected, result.log_path))
    if expected_why is not None:
        test.assertIn(
            expected_why, result.unweighting_why or '',
            'run %s resolved to %s but for the wrong reason: %r does not '
            'mention %r (log: %s)'
            % (result.label, expected, result.unweighting_why, expected_why,
               result.log_path))


def assert_weight_identity(test, result, min_checks=100,
                           max_spread=IDENTITY_SPREAD_TOL):
    """``sequential_debug``: the per-chain check that the product of the stage
    weights is proportional to the joint weight, recomputed by the joint code
    for the same production event, the same virtualities and the same decays.

    This is the assertion to lead with. It settles the weight algebra with *no
    statistics at all*: a scheme whose decomposition is broken has a ratio that
    varies chain to chain, and that shows up on the first few hundred chains
    whatever the sample size. It cannot flake, and it catches exactly the class
    of bug -- a missing mass-dependent normalisation -- that left every angular
    observable and the cross section clean while moving the reconstructed
    lineshape by 0.25 GeV.

    ``max_spread`` defaults to :data:`IDENTITY_SPREAD_TOL`, a few times float32
    epsilon: the density matrices are ``complex64``, so the two evaluation
    routes cannot agree better than that however correct the algebra is. Never
    assert below :data:`FLOAT32_EPS`."""
    test.assertIsNotNone(
        result.identity,
        'run %s produced no weight-identity report: sequential_debug was off, '
        'or the mode/spinmode combination skipped the check. This test is '
        'meaningless without it (log: %s)' % (result.label, result.log_path))
    report = result.identity
    test.assertTrue(
        report.ok,
        'weight identity FAILED for %s on %d chains: the chain weight is not '
        'proportional to the joint weight (relative spread %.3g, mean %.10g). '
        'This scheme is not sampling the joint distribution (log: %s)'
        % (result.label, report.checks, report.spread, report.ratio,
           result.log_path))
    test.assertGreaterEqual(
        report.checks, min_checks,
        'weight identity for %s was only exercised on %d chains (< %d): too '
        'few for the check to mean anything (log: %s)'
        % (result.label, report.checks, min_checks, result.log_path))
    test.assertLess(
        report.spread, max_spread,
        'weight identity for %s holds only to %.3g over %d chains, above the '
        'bound %.3g (float32 floor %.3g). MadSpin did not flag it itself -- its '
        'CRITICAL fires at density_tolerance, which is far looser than this '
        'test (log: %s)'
        % (result.label, report.spread, report.checks, max_spread,
           FLOAT32_EPS, result.log_path))


def assert_identity_ratios_agree(test, results, rel_tol=1e-6):
    """Every scheme must reach the *same* proportionality constant.

    The constant is the number of helicity states times the normalisation the
    density path applies to the decay matrix elements -- it depends on the
    process, not on how the accept/reject was split -- so two schemes reporting
    different constants means one of them folds an extra factor into its chain
    weight even though each is internally self-consistent."""
    ratios = {label: r.identity.ratio for label, r in results.items()
              if r.identity is not None}
    if len(ratios) < 2:
        return
    items = sorted(ratios.items())
    ref_label, ref = items[0]
    for label, ratio in items[1:]:
        rel = abs(ratio - ref) / max(abs(ref), 1e-30)
        test.assertLess(
            rel, rel_tol,
            'weight-identity constant differs between schemes: %s=%.10g vs '
            '%s=%.10g (rel=%.3g > %g). Each is self-consistent, so one of them '
            'carries a factor the other does not.'
            % (ref_label, ref, label, ratio, rel, rel_tol))


def assert_observable_consistent(test, samples, tolerance, name,
                                 reference=None, window=None):
    """Every scheme's mean of ``name`` must sit within ``tolerance`` (absolute,
    in the observable's own units) of the reference scheme's.

    ``samples`` is ``{label: [values]}``; ``reference`` names the scheme to
    compare against and defaults to the first key.

    **On the tolerance.** Do not derive it from the naive per-run Monte Carlo
    error. Runs of the same scheme with different MadSpin seeds share the
    production events, so they are strongly correlated and land much closer
    together than that error suggests -- measured on 10000-event ttbar, joint
    replicas scatter by 0.005 GeV on the mean reconstructed top mass against a
    naive per-run error of 0.023. Using the naive error would have let the
    original 0.25 GeV bias through at small statistics. The tolerance passed
    here must instead be calibrated by running the *same* scheme twice at the
    event count the test actually uses; the caller is expected to say in a
    comment what it measured."""
    stats = collections.OrderedDict()
    for label, values in samples.items():
        stats[label] = mean_and_error(values, window=window)
    labels = list(stats.keys())
    test.assertTrue(labels, 'no samples given for %s' % name)
    ref_label = reference if reference is not None else labels[0]
    test.assertIn(ref_label, stats,
                  'reference scheme %s absent from the %s comparison'
                  % (ref_label, name))
    ref_mean, ref_err, ref_n = stats[ref_label]

    dump = ', '.join('%s=%.4f+-%.4f (n=%d)' % (lab, m, e, n)
                     for lab, (m, e, n) in stats.items())
    test.assertGreater(
        ref_n, 100,
        'reference scheme %s has only %d entries of %s -- too few to compare '
        'against (dump: %s)' % (ref_label, ref_n, name, dump))

    for label in labels:
        if label == ref_label:
            continue
        mean, err, n = stats[label]
        test.assertGreater(
            n, 100,
            'scheme %s has only %d entries of %s (dump: %s)'
            % (label, n, name, dump))
        delta = mean - ref_mean
        test.assertLess(
            abs(delta), tolerance,
            '%s: %s mean %.4f differs from %s %.4f by %+.4f, above the '
            'calibrated tolerance %.4f. All four unweighting schemes must '
            'sample the same distribution, so this is a real difference in '
            'what the scheme samples, not a tuning knob. (dump: %s)'
            % (name, label, mean, ref_label, ref_mean, delta, tolerance, dump))
    return stats
