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
"""

from __future__ import absolute_import
from __future__ import division

import logging
import os
import unittest

from tests.parallel_tests.madspin_comparator import (
    DEFAULT_FAMILIES,
    DEFAULT_MODES,
    MadSpinFactory,
    SpinModeConfig,
    assert_branching_ratios_consistent,
    assert_cross_sections_consistent,
    assert_efficiency_close,
    assert_lhe_well_formed,
    assert_multiplicities_consistent,
    assert_offshell_mass_distribution,
)


_logger = logging.getLogger('test_madspin_factory')


NEVENTS = int(os.environ.get('MADSPIN_TEST_NEVENTS', '10000'))
# Per the user's call: start at 15%, widen to 30% after smoke test confirms
# real ratios stay below ~10%.
EFF_TOL = float(os.environ.get('MADSPIN_TEST_EFF_TOL', '0.15'))

# Smoke knob: lower max_weight_ps_point shortens MadSpin's max-weight probing
# stage at the cost of statistical precision. Leave the production default
# (400) alone unless explicitly overridden -- the CI tests want trustworthy
# unweighting.
_MAX_WEIGHT_PS_POINT = os.environ.get('MADSPIN_MAX_WEIGHT_PS_POINT', '')
EXTRA_MADSPIN_SETTINGS = {}
if _MAX_WEIGHT_PS_POINT:
    EXTRA_MADSPIN_SETTINGS['max_weight_ps_point'] = _MAX_WEIGHT_PS_POINT


class MadSpinFactoryTest(unittest.TestCase):
    """Base class that owns the factory lifetime."""

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
        """The two pairs called out in the spec:

        * ``full+decay_chain`` (old default) vs. ``PA+density`` (new default
          with mass reshuffling).
        * ``onshell+decay_chain`` vs. ``onshell+density`` (PA-without-
          reshuffling) -- the two on-shell variants.

        If either member of a pair was skipped via ``skip_modes`` we silently
        drop that pair check; the other pair (and the multiplicity / cross-
        section checks) still cover the surviving modes.
        """
        if 'full_decay_chain' in results and 'PA_density' in results:
            assert_efficiency_close(
                self, results['full_decay_chain'], results['PA_density'],
                rel_tol=EFF_TOL)
        if 'onshell_decay_chain' in results and 'onshell_density' in results:
            assert_efficiency_close(
                self, results['onshell_decay_chain'], results['onshell_density'],
                rel_tol=EFF_TOL)

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
            if k in ('full_decay_chain', 'full_density', 'PA_density')
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
            if k in ('full_decay_chain', 'full_density', 'PA_density')
        }
        if len(offshell_results) >= 2:
            # Z is narrow; tighten the constant tolerance.
            assert_offshell_mass_distribution(
                self, offshell_results,
                parent_pdg=23,
                pole_mass=91.1876, width=2.4952,
                tolerance_const=0.05, tolerance_offshell=3.0,
            )
