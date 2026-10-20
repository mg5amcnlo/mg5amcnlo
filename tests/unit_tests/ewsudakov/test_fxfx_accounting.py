"""Per-event accounting across clustering, scalar fallback and xi evaluations."""

import unittest
from unittest import mock

from madgraph.various import fxfx_ewsudakov as fx
from tests.unit_tests.ewsudakov import test_sudakov_factor_limits as limits


class TestFxFxAccounting(unittest.TestCase):
    def setUp(self):
        fx.fks_reset_accounting()
        self.addCleanup(fx.fks_reset_accounting)
        event_id = mock.patch.object(fx, 'CURRENT_EVENT_ID', 0)
        event_id.start()
        self.addCleanup(event_id.stop)
        self.fixture = limits.TestSudakovFactorLimits()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.event = self.fixture.event
        self.driver = limits.FactorDriver()

    def test_density_no_cluster_gives_unit_weights_for_each_policy(self):
        self.event = self.driver._cluster_fxfx_event(self.event)[0]
        self.fixture.event = self.event
        self.event.tag = ''
        for policy in ('accept', 'unit_weight'):
            with self.subTest(policy=policy), mock.patch.object(fx, 'FKS_FALLBACK_POLICY', policy):
                fx.fks_reset_accounting()
                self.driver._compute_ewsudakov_reweight = mock.Mock(return_value={'orig': -2.})
                for event_id, expected in ((0, 0), (1, 1), (1, 1), (2, 2), (2, 2), (0, 2)):
                    fx.CURRENT_EVENT_ID = event_id
                    result = self.driver._compute_density_ewsudakov_reweight(self.event, None)
                    self.assertEqual(fx.FKS_QUALITY_COUNTS['events_clustering_failed'], expected)
                    self.assertEqual(fx.FKS_QUALITY_COUNTS['events_unit_weight_by_policy'], 0)
                    self.fixture.assert_factors(self.driver, result, [1., 1., 1.])
                self.driver._compute_ewsudakov_reweight.assert_not_called()

    def test_density_dimension_limit_counts_scalar_neutralization(self):
        module = self.fixture.scalar_module(self.driver, [201., 41., 31.])
        with mock.patch.object(fx, 'density_dimensions_supported', return_value=False), \
                mock.patch.object(fx, '_DIM_OVERFLOW_WARNED', True):
            for event_id, expected in ((0, 0), (1, 1), (1, 1), (2, 2), (2, 2), (0, 2)):
                fx.CURRENT_EVENT_ID = event_id
                result = self.driver._compute_density_ewsudakov_reweight(self.event, module)
                self.assertEqual(fx.FKS_QUALITY_COUNTS['events_density_dim_overflow'], expected)
                self.assertEqual(fx.FKS_QUALITY_COUNTS['events_runaway_damped_scalar'], expected)
                self.assertEqual(sum(fx.FKS_METHOD_COUNTS.values()), 2 * expected)
                self.fixture.assert_factors(self.driver, result, [1., 1., 1.])

    def test_density_and_scalar_fsr_outcomes_count_once_across_xis(self):
        module = self.fixture.scalar_module(self.driver, [1.2, 1.1, 1.05])
        cluster = self.driver._cluster_fxfx_event_internal

        def with_fsr_outcome(*args, **kwargs):
            result = cluster(*args, **kwargs)
            result[0].fks_methods = ['fks_fsr_unreachable_target']
            return result

        with mock.patch.object(self.driver, '_cluster_fxfx_event_internal', side_effect=with_fsr_outcome), \
                mock.patch.object(fx.ResonanceIdentifier, 'find_resonances', return_value=[]), \
                mock.patch.object(fx, 'FKS_FALLBACK_POLICY', 'accept'):
            for event_id in (1, 2):
                fx.CURRENT_EVENT_ID = event_id
                for xi in (.5, 1.5, 3.):
                    self.driver.mW2_cluster_scale = xi
                    result = self.driver._compute_density_ewsudakov_reweight(self.event, module)
                    self.assertEqual(fx.FKS_QUALITY_COUNTS['events_with_fsr_fallback_density'], event_id)
                    self.assertEqual(fx.FKS_QUALITY_COUNTS['events_with_fsr_fallback'], event_id)
                    self.assertEqual(sum(fx.FKS_METHOD_COUNTS.values()), 2 * event_id)
                    self.fixture.assert_factors(self.driver, result, [1.2, 1.1, 1.05])

    def test_unit_weight_policy_in_each_driver(self):
        cluster = self.driver._cluster_fxfx_event_internal

        def with_fsr_outcome(*args, **kwargs):
            result = cluster(*args, **kwargs)
            result[0].fks_methods = ['fks_fsr_unreachable_target']
            return result

        with mock.patch.object(self.driver, '_cluster_fxfx_event_internal', side_effect=with_fsr_outcome), \
                mock.patch.object(fx, 'FKS_FALLBACK_POLICY', 'unit_weight'):
            fx.CURRENT_EVENT_ID = 1
            for driver in (self.driver._compute_density_ewsudakov_reweight,
                           self.driver._compute_ewsudakov_fxfx_reweight):
                for _ in range(2):
                    result = driver(self.event, None)
                    self.fixture.assert_factors(self.driver, result, [1., 1., 1.])
            self.assertEqual(fx.FKS_QUALITY_COUNTS['events_unit_weight_by_policy'], 1)
            self.assertEqual(fx.FKS_QUALITY_COUNTS['events_with_fsr_fallback_density'], 1)
            self.assertEqual(fx.FKS_QUALITY_COUNTS['events_with_fsr_fallback'], 1)

    def test_launch_reset_allows_event_ids_to_restart(self):
        self.event = self.driver._cluster_fxfx_event(self.event)[0]
        self.fixture.event = self.event
        self.event.tag = ''
        with mock.patch.object(fx, 'FKS_FALLBACK_POLICY', 'unit_weight'):
            for _ in range(2):
                fx.fks_reset_accounting()
                self.assertEqual(fx._FKS_QUALITY_EVENT_IDS, {})
                fx.CURRENT_EVENT_ID = 1
                self.driver._compute_density_ewsudakov_reweight(self.event, None)
                self.assertEqual(fx.FKS_QUALITY_COUNTS['events_clustering_failed'], 1)


if __name__ == '__main__':
    unittest.main()
