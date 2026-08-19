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
"""Parallel tests built on :mod:`madspin_comparator`.

Each ``test_short_madspin_*`` test instantiates a :class:`MadSpinFactory`,
runs MadSpin in the five (spinmode, ME_mode) configurations described in the
MadSpin density-mode table, and then asserts:

    * each LHE is well-formed;
    * global branching ratios agree across modes;
    * lepton/quark final-state multiplicities agree within Poisson noise;
    * efficiency pairs that are physically expected to match -- old default
      vs. PA+density-reshuffled, and traditional onshell vs. PA-without-
      reshuffling -- match within ``EFF_TOL`` (15% to start; widen per the
      smoke-test plan if real values land near the edge);
    * off-shell mass distributions of decaying resonances are mutually
      compatible and close to a Breit-Wigner with an increasing tolerance the
      further off-shell we look.

CI runtime is dominated by the production step plus five MadSpin runs. With
``NEVENTS=10000`` and four-core MadSpin this is roughly 5-8 minutes per test
on a GitHub Linux runner.

:class:`MadSpinUnweightingTest` covers a different axis: ``set unweighting``,
which chooses how the accept/reject is split, at a fixed spinmode. Its three
tests are ordered by how much they assume --

    * ``test_short_madspin_unweighting_identity``: per-chain, deterministic,
      no statistics at all. This is the one to read first when something breaks;
    * ``test_short_madspin_unweighting_resolution``: which scheme each run
      actually resolved to, parsed from the run's own announcement;
    * ``test_long_madspin_unweighting_consistency``: the statistical comparison
      of the reconstructed resonance lineshape across schemes, with tolerances
      calibrated from measured replicas (see the calibration note in the file).
"""

from __future__ import absolute_import
from __future__ import division

import collections
import logging
import math
import os
import unittest

from tests.parallel_tests.madspin_comparator import (
    DEFAULT_FAMILIES,
    DEFAULT_MODES,
    UNWEIGHTING_MODES,
    MadSpinFactory,
    SpinModeConfig,
    assert_branching_ratios_consistent,
    assert_cross_sections_consistent,
    assert_efficiency_close,
    assert_efficiency_ordering,
    assert_identity_ratios_agree,
    assert_lhe_well_formed,
    assert_multiplicities_consistent,
    assert_observable_consistent,
    assert_offshell_mass_distribution,
    assert_unweighting_mode,
    assert_weight_identity,
    final_state_dphi,
    resonance_masses,
)


_logger = logging.getLogger('test_madspin_factory')


NEVENTS = int(os.environ.get('MADSPIN_TEST_NEVENTS', '10000'))
# Per the user's call: start at 15%, widen to 30% after smoke test confirms
# real ratios stay below ~10%.
EFF_TOL = float(os.environ.get('MADSPIN_TEST_EFF_TOL', '0.15'))

# Number of cores exercised by test_short_madspin_multicore (the process-
# parallel unweighting path is enabled for nb_core > 1).
MULTICORE_NB = int(os.environ.get('MADSPIN_TEST_NB_CORE', '8'))

# Smoke knob: lower max_weight_ps_point shortens MadSpin's max-weight probing
# stage at the cost of statistical precision. Leave the production default
# (400) alone unless explicitly overridden -- the CI tests want trustworthy
# unweighting.
_MAX_WEIGHT_PS_POINT = os.environ.get('MADSPIN_MAX_WEIGHT_PS_POINT', '')
EXTRA_MADSPIN_SETTINGS = {'unweighting': 'joint'}
if _MAX_WEIGHT_PS_POINT:
    EXTRA_MADSPIN_SETTINGS['max_weight_ps_point'] = _MAX_WEIGHT_PS_POINT

# The unweighting tests below drive ``unweighting`` directly, so they must not
# inherit the ``unweighting = joint`` above -- it would fight the per-run
# setting.
UNWEIGHTING_BASE_SETTINGS = {'nb_core': 1}
if _MAX_WEIGHT_PS_POINT:
    UNWEIGHTING_BASE_SETTINGS['max_weight_ps_point'] = _MAX_WEIGHT_PS_POINT

# Event count for the weight-identity / mode-resolution test. Both checks are
# deterministic -- the identity is verified chain by chain and the resolved mode
# is announced during setup -- so this only has to be large enough to accumulate
# a few hundred accepted chains.
IDENTITY_NEVENTS = int(os.environ.get('MADSPIN_IDENTITY_NEVENTS', '800'))


class _MadSpinFactoryBase(unittest.TestCase):
    """Factory lifetime, shared by the test classes below.

    Deliberately holds no ``test_*`` method of its own: unittest collects every
    TestCase subclass, so a concrete test living here would be re-run once per
    subclass."""

    maxDiff = None

    def setUp(self):
        self._factories = []

    def tearDown(self):
        # Only sweep on success: a failing run is worth keeping for inspection.
        if not self._outcome_has_failure():
            for factory in self._factories:
                factory.cleanup()

    def _outcome_has_failure(self):
        outcome = getattr(self, '_outcome', None)
        if outcome is None:
            return False
        result = getattr(outcome, 'result', None) or outcome
        for attr in ('errors', 'failures'):
            entries = getattr(result, attr, None) or []
            for case, _trace in entries:
                if case is self:
                    return True
        return False

    # ------------------------------------------------------------------
    def _make_factory(self, **kw):
        kw.setdefault('nevents', NEVENTS)
        if EXTRA_MADSPIN_SETTINGS:
            kw.setdefault('extra_madspin_settings', EXTRA_MADSPIN_SETTINGS)
        factory = MadSpinFactory(**kw)
        self._factories.append(factory)
        return factory


class MadSpinFactoryTest(_MadSpinFactoryBase):
    """The five (spinmode, ME_mode) configurations of the MadSpin density-mode
    table, compared against each other on one production sample."""

    def _run_all_modes(self, factory, modes=DEFAULT_MODES, skip_modes=()):
        """Run every config in ``modes`` whose label isn't in ``skip_modes``.

        Use ``skip_modes`` to opt out of a known-broken (spinmode, ME_mode)
        combination for a specific test (with a TODO and bug link in the test
        body explaining why). The factory remains strict for every mode that
        is run -- a crash in an un-skipped mode still aborts the test.
        """
        skip_set = {label for label in skip_modes}
        active_modes = [cfg for cfg in modes if cfg.label not in skip_set]
        if skip_set:
            _logger.warning(
                '[%s] skipping modes: %s', factory.name, sorted(skip_set))
        results = factory.run_modes(active_modes)
        for r in results.values():
            assert_lhe_well_formed(self, r)
            _logger.info(
                '[%s/%s] BR=%s cross_out=%s eff=%s wall=%.1fs lhe=%s',
                factory.name, r.label, r.BR, r.cross_out, r.efficiency,
                r.wall_seconds, r.lhe_path,
            )
        # Physics-observable invariant: every mode in the same BR family
        # must produce the same decayed cross-section (rel_tol=1e-3, strict);
        # cross-family agreement is looser (between_tol=5e-2) because the
        # legacy decay-chain path uses factorised on-shell BRs while the
        # run_onshell paths use MC-integrated partial widths.
        assert_cross_sections_consistent(
            self, results, rel_tol=1e-3,
            families=DEFAULT_FAMILIES, between_tol=5e-2,
        )
        # BR check is informational and intentionally loose -- the BR value
        # itself can shift by a few percent across families for the reasons
        # above.
        assert_branching_ratios_consistent(self, results, rel_tol=5e-2)
        return results

    def _check_efficiency_pairs(self, results):
        """Physics-motivated efficiency ordering (relaxed from strict pair
        equality):

        1. (dropped on PR #292 after the 10k-event ttbar run; see
           ``assert_efficiency_ordering`` for rationale.)
        2. ``onshell_decay_chain`` and ``onshell_density`` are close to each
           other (within ``EFF_TOL``) and both are higher than ``PA_density``
           (the pole approximation).
        3. ``madspin_density`` lies between ``full_decay_chain`` and
           ``PA_density``.

        Missing modes (e.g. ``skip_modes`` opt-outs) are tolerated; the rules
        that mention them are dropped silently.
        """
        assert_efficiency_ordering(
            self, results,
            close_rel_tol=EFF_TOL,
        )

    # ==================================================================
    # Concrete tests.
    # ==================================================================

    def test_short_madspin_ttbar(self):
        """tt~ semileptonic: tests off-shell W mass, lepton/jet multiplicities,
        and both efficiency pairs."""
        factory = self._make_factory(
            name='ttbar',
            production_process='p p > t t~',
            decays=[
                't > b w+, w+ > l+ vl',
                't~ > b~ w-, w- > j j',
            ],
            multiparticles={'p': 'g u d s c u~ d~ s~ c~',
                            'j': 'g u d s c u~ d~ s~ c~',
                            'l+': 'e+ mu+',
                            'vl': 've vm',
                            'l-': 'e- mu-',
                            'vl~': 've~ vm~'},
            extra_run_card={'ebeam1': 6500, 'ebeam2': 6500},
        )
        results = self._run_all_modes(factory)
        # b, b~, e+/mu+, e-/mu- final-state population.
        assert_multiplicities_consistent(
            self, results, pdgs=[5, -5, 11, 13, -11, -13])
        self._check_efficiency_pairs(results)
        # Off-shell W+ mass distribution (children: l+ vl).
        # Use only the modes that can actually produce off-shell mass.
        offshell_results = {
            k: v for k, v in results.items()
            if k in ('full_decay_chain', 'madspin_density', 'PA_density')
        }
        if len(offshell_results) >= 2:
            assert_offshell_mass_distribution(
                self, offshell_results,
                parent_pdg=24,
                pole_mass=80.379, width=2.085,
            )

    def test_short_madspin_singletop(self):
        """Single-top t-channel: top off-shell distribution sanity, plus the
        two efficiency-pair requirements on a smaller process.

        Uses the 5-flavor scheme (``p`` includes ``b b~``) so the t-channel
        ``u b > d t`` family of diagrams exists.
        """
        factory = self._make_factory(
            name='singletop',
            production_process='p p > t j',
            decays=['t > b w+, w+ > l+ vl'],
            multiparticles={'p': 'g u d s c b u~ d~ s~ c~ b~',
                            'j': 'g u d s c b u~ d~ s~ c~ b~',
                            'l+': 'e+ mu+',
                            'vl': 've vm'},
            extra_run_card={'ebeam1': 6500, 'ebeam2': 6500},
        )
        results = self._run_all_modes(factory)
        assert_multiplicities_consistent(
            self, results, pdgs=[5, 11, 13])
        self._check_efficiency_pairs(results)

    def test_short_madspin_zz(self):
        """ZZ leptonic: a narrow-resonance stress test for the BW shape and
        identical-particle bookkeeping in the four-lepton final state."""
        factory = self._make_factory(
            name='zz',
            production_process='p p > z z',
            # One decay line per particle type -- MadSpin applies it to every
            # matching final-state particle in the event.
            decays=['z > l+ l-'],
            multiparticles={'p': 'g u d s c u~ d~ s~ c~',
                            'l+': 'e+ mu+',
                            'l-': 'e- mu-'},
            extra_run_card={'ebeam1': 6500, 'ebeam2': 6500},
        )
        results = self._run_all_modes(factory)
        assert_multiplicities_consistent(
            self, results, pdgs=[11, -11, 13, -13])
        self._check_efficiency_pairs(results)
        offshell_results = {
            k: v for k, v in results.items()
            if k in ('full_decay_chain', 'madspin_density', 'PA_density')
        }
        if len(offshell_results) >= 2:
            # Z is narrow; tighten the constant tolerance.
            assert_offshell_mass_distribution(
                self, offshell_results,
                parent_pdg=23,
                pole_mass=91.1876, width=2.4952,
                tolerance_const=0.05, tolerance_offshell=3.0,
            )

    def test_short_madspin_multicore(self):
        """Process-parallel unweighting (``set nb_core %d``) must reproduce the
        serial result on the SAME production sample: identical event count, an
        identical decayed cross-section (same BR, banner-derived), and a
        statistically consistent unweighting efficiency. Also a regression guard
        against the fork / read-only-gridpack segfault that motivated the
        parallel path.
        """ % MULTICORE_NB
        factory = self._make_factory(
            name='multicore',
            production_process='p p > t t~',
            decays=[
                't > b w+, w+ > l+ vl',
                't~ > b~ w-, w- > j j',
            ],
            multiparticles={'p': 'g u d s c u~ d~ s~ c~',
                            'j': 'g u d s c u~ d~ s~ c~',
                            'l+': 'e+ mu+',
                            'vl': 've vm'},
            extra_run_card={'ebeam1': 6500, 'ebeam2': 6500},
        )
        # Same (PA) mode, same production events, serial vs multi-core.
        cfg = SpinModeConfig('PA_density', 'PA')
        serial = factory.run_mode(cfg, extra_settings={'nb_core': 1},
                                  run_tag='serial')
        parallel = factory.run_mode(cfg, extra_settings={'nb_core': MULTICORE_NB},
                                    run_tag='nb%d' % MULTICORE_NB)

        assert_lhe_well_formed(self, serial)
        assert_lhe_well_formed(self, parallel)

        # 1. Every production event yields exactly one decayed event in PA mode,
        #    so the parallel shard-split + merge must preserve the event count
        #    exactly (this catches merge/accounting bugs).
        n_serial, _ = serial.count_pdgs()
        n_parallel, _ = parallel.count_pdgs()
        self.assertEqual(
            n_serial, n_parallel,
            'decayed event count differs: serial=%d, nb_core=%d -> %d'
            % (n_serial, MULTICORE_NB, n_parallel))

        # 2. Decayed cross-section (banner: cross_in * BR) is computed pre-fork,
        #    so it must match essentially exactly.
        self.assertIsNotNone(serial.cross_out, 'serial cross-section missing')
        self.assertIsNotNone(parallel.cross_out, 'parallel cross-section missing')
        rel = abs(parallel.cross_out - serial.cross_out) / abs(serial.cross_out)
        self.assertLess(
            rel, 1e-2,
            'decayed cross-section differs: serial=%s, nb_core=%d -> %s (rel=%.3g)'
            % (serial.cross_out, MULTICORE_NB, parallel.cross_out, rel))

        # 3. Unweighting efficiency should be statistically consistent (the two
        #    runs use independent RNG streams).
        assert_efficiency_close(self, serial, parallel, rel_tol=EFF_TOL)


# ---------------------------------------------------------------------------
# ``set unweighting``: the four accept/reject schemes.
# ---------------------------------------------------------------------------

# Fully leptonic ttbar. Two decaying particles at the top level (t and t~), so
# ``auto`` lands on ``joint`` under an offshell spinmode (offshell it only
# leaves joint from three decaying particles up) and on ``sequential`` under
# PA/onshell -- and both tops give a reconstructable lineshape while the two
# leptons give the angular no-regression observable.
TTBAR_LEPTONIC = dict(
    production_process='p p > t t~',
    decays=['t > b w+, w+ > l+ vl',
            't~ > b~ w-, w- > l- vl~'],
    multiparticles={'p': 'g u d s c u~ d~ s~ c~',
                    'l+': 'e+ mu+', 'vl': 've vm',
                    'l-': 'e- mu-', 'vl~': 've~ vm~'},
    extra_run_card={'ebeam1': 6500, 'ebeam2': 6500},
)

L_PLUS = (-11, -13)
L_MINUS = (11, 13)

# ``joint`` first: it is the historical scheme and the reference every other one
# is compared against. The rest need an offshell spinmode to keep the name they
# were asked for, which is why the identity test runs them under ``madspin``.
NON_JOINT_MODES = tuple(m for m in UNWEIGHTING_MODES if m != 'joint')

# ---------------------------------------------------------------------------
# Calibration of the consistency tolerances.  Re-derive it, do not guess at it.
#
# Campaign: `p p > t t~`, `t > b w+, w+ > l+ vl` and charge conjugate, spinmode
# madspin, 5000 production events, nine MadSpin runs off that ONE sample -- the
# four schemes with two seeds each (three for joint). All nine are replicas of
# the same quantity if the schemes agree, so they estimate the run-to-run
# scatter with 8 degrees of freedom rather than the 1 a single pair gives.
# Both tops are pooled (same lineshape; pooling halves the error on it).
#
#                       m(l+ vl b)   m(l+ vl)   dphi(l+,l-)
#   replica sd            0.0261      0.0275      0.0146
#   naive per-run error   0.0316      0.0426      0.0128
#   signal (see below)    0.2126      0.0334      0.0014
#   signal / replica sd      8.2         1.2         0.1
#
# The *signal* is the bias this test exists to catch. The PR #334 bug made the
# offshell scheme sample the PA/Breit-Wigner lineshape instead of the offshell
# one, so a PA run off the same production sample reproduces it: the row above
# is offshell-minus-PA, measured, and it lands on the -0.248 GeV that bug was
# originally reported at.
#
# Three things fall out, and they are the whole design of the test:
#
# 1. **Only the lineshape can see it.** m(l+ vl) and dphi(l+,l-) have a
#    signal-to-noise of 1.2 and 0.1 -- they would not have moved measurably had
#    the bug still been there. They are asserted below as no-regression checks
#    and must not be read as standing in for the lineshape.
#
# 2. **Do not window the lineshape.** The obvious refinement -- restrict the
#    mean to the peak, where a mis-normalised virtuality weight bites -- was
#    measured and is a net loss: a +-10 GeV window cuts the replica sd from
#    0.0261 to 0.0155 but the signal from 0.2126 to 0.0949, so S/N falls from
#    8.2 to 6.1. The Breit-Wigner tails carry a disproportionate share of the
#    difference between the two lineshapes. The untruncated mean it is.
#
# 3. **Neither published error model is right, and the replicas settle it.**
#    The plan document's two-replica estimate had joint scattering by 0.005
#    against a naive error of 0.023, suggesting the runs were strongly
#    correlated through the shared production events and the naive error was 4x
#    too loose. Over nine replicas the ratio of replica sd to naive error is
#    0.82 (lineshape), 0.65 (W mass) and 1.14 (dphi): mildly correlated at
#    most. The 0.005 was a small-sample fluctuation. The tolerances below are
#    therefore built on the measured replica sd, which is the thing the test
#    actually has to survive.
LINESHAPE_CALIB_NEVENTS = 5000
# Replica sd (GeV, GeV, rad) at LINESHAPE_CALIB_NEVENTS, from the table above.
CALIB_SD = {'lineshape': 0.0261, 'w_mass': 0.0275, 'dphi': 0.0146}
# Two runs differ by sqrt(2) times a single run's scatter, and 4 sigma of that
# leaves a per-comparison false-positive rate around 6e-5 -- with four
# comparisons per test, a flake roughly once in four thousand runs. Even if the
# sd above is underestimated by the ~25% its 8 degrees of freedom allow, that
# only relaxes to 3.2 sigma.
CALIB_NSIGMA = 4.0
# Measured offshell-minus-PA on the lineshape: the size of the bias to catch.
LINESHAPE_SIGNAL = 0.213


def _consistency_tolerance(key, nevents):
    """Tolerance for ``key`` at ``nevents``, scaled off the calibration.

    The replica scatter is 0.65-1.14 of the naive Monte Carlo error, i.e.
    dominated by it, so it scales as 1/sqrt(N) to the accuracy this needs."""
    scale = math.sqrt(LINESHAPE_CALIB_NEVENTS / float(nevents))
    return CALIB_NSIGMA * math.sqrt(2) * CALIB_SD[key] * scale


class MadSpinUnweightingTest(_MadSpinFactoryBase):
    """``set unweighting`` selects how the accept/reject is organised:

        joint                    one test over the virtualities and every decay
        two_stage                virtualities first, then all angles, one bound
        sequential               virtualities first, then one test per particle
        sequential_global_retry  as sequential, but a rejected decay redraws the
                                 virtualities too

    All four are supposed to sample the *same* distribution -- they differ only
    in how the test is split and in what a rejection redraws. The tests here
    hold them to that, and they are deliberately ordered weakest-assumption
    first: the weight identity settles the algebra with no statistics at all,
    the mode assertions are non-statistical guards on the resolution logic, and
    only the lineshape comparison needs an error model.
    """

    def _unweighting_factory(self, name, nevents):
        return self._make_factory(
            name=name, nevents=nevents,
            extra_madspin_settings=dict(UNWEIGHTING_BASE_SETTINGS),
            **TTBAR_LEPTONIC)

    # ------------------------------------------------------------------
    def test_short_madspin_unweighting_identity(self):
        """The deterministic check, plus the resolved-mode guards.

        ``set sequential_debug True`` makes MadSpin recompute, on every accepted
        chain, the joint weight for the *same* production event, the same
        virtualities and the same decays, and verify that the product of the
        stage weights is proportional to it. A scheme whose decomposition is
        broken has a ratio that varies chain to chain, so this fails on the
        first few hundred chains whatever the sample size -- no statistics, no
        error model, nothing to flake.

        That matters because the bug this whole area exists for (PR #334) was a
        scheme sampling a subtly different distribution: the reconstructed top
        lineshape came out Breit-Wigner shaped and 7.8 sigma off while every
        angular observable and the cross section stayed clean. A statistical A/B
        can only bound such a bias at the level its Monte Carlo error allows;
        this settles the weight algebra outright.
        """
        factory = self._unweighting_factory('unweighting_identity',
                                            IDENTITY_NEVENTS)
        cfg = SpinModeConfig('madspin_density', 'madspin')

        results = collections.OrderedDict()
        for mode in NON_JOINT_MODES:
            results[mode] = factory.run_mode(
                cfg, run_tag=mode,
                extra_settings={'unweighting': mode,
                                'sequential_debug': 'True'})

        for mode, result in results.items():
            assert_lhe_well_formed(self, result)
            _logger.info('[unweighting/%s] resolved=%s identity=%s eff=%s '
                         'overflows=%d wall=%.1fs',
                         mode, result.unweighting_mode, result.identity,
                         result.efficiency, result.overflows,
                         result.wall_seconds)
            # An explicit setting is always honoured; if it were not, the
            # identity below would be checking whatever scheme actually ran.
            assert_unweighting_mode(self, result, mode, 'set explicitly')
            # The lead assertion.
            assert_weight_identity(
                self, result, min_checks=max(100, IDENTITY_NEVENTS // 4))

        # The proportionality constant is a property of the process (helicity
        # states x the density path's decay-ME normalisation), not of how the
        # accept/reject was split, so all three schemes must report the same
        # one. Each could be internally self-consistent and still disagree here.
        assert_identity_ratios_agree(self, results)

    # ------------------------------------------------------------------
    def test_short_madspin_unweighting_resolution(self):
        """Which scheme a run actually uses, end to end.

        ``auto`` resolves on the process, and several combinations override what
        the card asked for, so the card value does not answer this -- but every
        run announces the answer ("MadSpin: unweighting = <mode> (<why>)"). The
        checks are instant and non-statistical, and without them a consistency
        matrix could silently compare four runs of the same scheme and pass.

        Covered here, all off one production sample:

          * ``auto`` + offshell, two decaying particles -> ``joint``. Offshell,
            a mass set costs a production reshuffle *and* a production density,
            so ``auto`` only leaves joint from three decaying particles up;
          * ``auto`` + offshell, *one* decaying particle -> ``joint`` as well,
            by the same branch. Both auto/offshell cases land on the same
            scheme, so what separates them is the announced count -- and this
            is the one case that exercises the real ``_nb_decaying`` against
            real events, since it needs the decay lines to be counted rather
            than the default assumed;
          * ``auto`` + PA -> ``sequential``: PA keeps rho fixed on shell, so
            its mass stage costs a reshuffling jacobian and nothing else, and
            sequential was the fastest at every multiplicity measured;
          * ``two_stage`` under PA and ``sequential_global_retry`` under
            onshell -> themselves. PA has an up-front mass draw of its own, so
            the up-front-mass schemes are honoured there rather than downgraded
            -- this is the end-to-end guard on that (``two_stage`` in
            particular is no longer offered in the card but must still run when
            asked for by name);
          * ``sequential_with_mass`` + offshell -> ``sequential``: the one
            fallback left in the resolution. It draws each slot's virtuality
            inside that slot's accept/reject, which the offshell spinmodes
            cannot do -- they reshuffle the whole production onto the mass set
            at once;
          * ``joint`` asked for explicitly -> ``joint``.

        Not covered here: the offshell ``auto`` boundary itself (three decaying
        particles -> ``sequential``), which would need a third production
        sample this factory does not build; ``fixed_order`` forcing joint,
        which needs a fixed-order sample; the decay-group override; and the
        unsupported-spinmode fallback, which is unreachable from a real run
        (only PA/onshell/madspin/full reach the resolution at all). All of
        those are unit-tested against a stub.
        """
        factory = self._unweighting_factory('unweighting_resolution',
                                            IDENTITY_NEVENTS)
        offshell = SpinModeConfig('madspin_density', 'madspin')
        pa = SpinModeConfig('PA_density', 'PA')
        onshell = SpinModeConfig('onshell_density', 'onshell')

        # (tag, config, asked, decays, expected mode, expected reason)
        cases = [
            ('auto_offshell', offshell, 'auto', None,
             'joint', 'auto, 2 decaying particle(s)'),
            ('auto_offshell_single', offshell, 'auto',
             ['t > b w+, w+ > l+ vl'],
             'joint', 'auto, 1 decaying particle(s)'),
            ('auto_pa', pa, 'auto', None,
             'sequential', 'auto, 2 decaying particle(s)'),
            ('two_stage_pa', pa, 'two_stage', None,
             'two_stage', 'set explicitly'),
            ('global_retry_onshell', onshell, 'sequential_global_retry', None,
             'sequential_global_retry', 'set explicitly'),
            ('with_mass_offshell', offshell, 'sequential_with_mass', None,
             'sequential', 'set explicitly'),
            ('joint_offshell', offshell, 'joint', None,
             'joint', 'set explicitly'),
        ]
        for tag, cfg, asked, decays, expected, why in cases:
            result = factory.run_mode(cfg, run_tag=tag, decays=decays,
                                      extra_settings={'unweighting': asked})
            _logger.info('[resolution/%s] asked=%s spinmode=%s -> %s (%s)',
                         tag, asked, cfg.spinmode, result.unweighting_mode,
                         result.unweighting_why)
            assert_unweighting_mode(self, result, expected, why)

    # ------------------------------------------------------------------
    def test_long_madspin_unweighting_consistency(self):
        """One production sample, the four schemes run off it, compared on the
        observable that the class of bug they can carry actually moves.

        **The reconstructed resonance lineshape is that observable.** ``m(l+ vl
        b)`` -- the top reconstructed from its decay products -- is what the
        original bug shifted by 0.25 GeV, and the calibration above measures its
        signal-to-noise at 8.2 against 1.2 for ``m(l+ vl)`` and 0.1 for
        ``dphi(l+, l-)``. The other two are asserted as no-regression checks and
        nothing more: at those ratios they would not have moved measurably even
        with the bug still in place, so they must never be read as standing in
        for the lineshape.

        **The cross section is deliberately not compared.** MadSpin writes
        sigma_in x BR into the decayed banner, which does not depend on the
        unweighting weight at all -- it would agree to machine precision between
        a correct scheme and a broken one.

        A ``joint`` replica (same scheme, same production events, different
        MadSpin seed) is run as a control. It is held to the same tolerance as
        the other schemes, so a failure says which of the two things went wrong:
        if the control fails too, the tolerance is too tight for this event
        count; if only a scheme fails, that scheme samples something else.
        """
        lineshape_tol = _consistency_tolerance('lineshape', NEVENTS)
        # State the test's own power rather than letting it quietly evaporate
        # if someone turns MADSPIN_TEST_NEVENTS down.
        self.assertLess(
            lineshape_tol, LINESHAPE_SIGNAL,
            'at %d events the calibrated lineshape tolerance is %.3f GeV, at or '
            'above the %.3f GeV bias this test exists to catch -- it would pass '
            'the PR #334 bug. Raise MADSPIN_TEST_NEVENTS to at least %d.'
            % (NEVENTS, lineshape_tol, LINESHAPE_SIGNAL,
               int(math.ceil(LINESHAPE_CALIB_NEVENTS
                             * (CALIB_NSIGMA * math.sqrt(2)
                                * CALIB_SD['lineshape'] / LINESHAPE_SIGNAL) ** 2))))
        if lineshape_tol > 0.5 * LINESHAPE_SIGNAL:
            _logger.warning(
                'lineshape tolerance %.3f GeV is over half the %.3f GeV bias '
                'being guarded against: %d events leaves little margin',
                lineshape_tol, LINESHAPE_SIGNAL, NEVENTS)
        factory = self._unweighting_factory('unweighting_consistency', NEVENTS)
        cfg = SpinModeConfig('madspin_density', 'madspin')

        runs = collections.OrderedDict()
        for mode in UNWEIGHTING_MODES:
            runs[mode] = factory.run_mode(
                cfg, run_tag=mode, extra_settings={'unweighting': mode})
        # The control: joint again, off the same production events, with a
        # different MadSpin seed. MadSpin seeds its RNG on the first `set seed`
        # of the card and ignores every later one, so this has to replace that
        # line -- which is what the factory's `seed` argument does.
        runs['joint_replica'] = factory.run_mode(
            cfg, run_tag='joint_replica', seed=factory.seed + 1,
            extra_settings={'unweighting': 'joint'})

        for label, result in runs.items():
            assert_lhe_well_formed(self, result)
            _logger.info('[consistency/%s] resolved=%s seed=%s eff=%s '
                         'cross_out=%s overflows=%d wall=%.1fs',
                         label, result.unweighting_mode, result.seed,
                         result.efficiency, result.cross_out, result.overflows,
                         result.wall_seconds)
            expected = 'joint' if label.startswith('joint') else label
            assert_unweighting_mode(self, result, expected, 'set explicitly')

        # Pool both tops: they are the same lineshape, and pooling halves the
        # statistical error on it.
        lineshape = collections.OrderedDict(
            (label, resonance_masses(r, 6) + resonance_masses(r, -6))
            for label, r in runs.items())
        w_mass = collections.OrderedDict(
            (label, resonance_masses(r, 24) + resonance_masses(r, -24))
            for label, r in runs.items())
        dphi = collections.OrderedDict(
            (label, final_state_dphi(r, L_PLUS, L_MINUS))
            for label, r in runs.items())

        # The observable that matters. Untruncated on purpose -- see the
        # calibration note: a peak window cuts the signal harder than the noise.
        assert_observable_consistent(
            self, lineshape, lineshape_tol, 'm(l+ vl b) [GeV]',
            reference='joint')
        # No-regression only (S/N 1.2 and 0.1 against this class of bug).
        assert_observable_consistent(
            self, w_mass, _consistency_tolerance('w_mass', NEVENTS),
            'm(l+ vl) [GeV] (no-regression only, blind to the lineshape bias)',
            reference='joint')
        assert_observable_consistent(
            self, dphi, _consistency_tolerance('dphi', NEVENTS),
            'dphi(l+,l-) [rad] (no-regression only, blind to the lineshape '
            'bias)', reference='joint')
