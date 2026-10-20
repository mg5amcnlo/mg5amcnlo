"""Central-factor limits across scalar and density Sudakov evaluation."""

from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from madgraph.interface.reweight_interface import ReweightInterface
from madgraph.various import fxfx_ewsudakov as fx
from tests.unit_tests.ewsudakov.test_fxfx_decay_transport import make_event, ww_event


class FactorDriver(fx.FxFxEWSudakovMixin):
    def __init__(self):
        self.model = None
        self.mW2_cluster_scale = 1.5
        self.fxfx_clustered_final_states = set()
        self._density_logger = mock.Mock()

    def _init_pole_masses_from_banner(self):
        pass


class TestSudakovFactorLimits(unittest.TestCase):
    cases = [
        ([1001., 41., 31.], [1., 1., 1.]),
        ([-201., 41., 31.], [1., 1., 1.]),
        ([200., 300., -301.], [200., 300., -301.]),
        ([-200., -300., 301.], [-200., -300., 301.]),
        ([1.1, 300., -301.], [1.1, 300., -301.]),
        ([1., 300., -301.], [1., 300., -301.]),
    ]

    def setUp(self):
        fx.fks_reset_accounting()
        self.addCleanup(fx.fks_reset_accounting)
        self.event = ww_event([])
        self.event.wgt = -2.
        self.event.parse_reweight()
        self.event.reweight_data = {'1001': -2., '1101': 3.}
        self.event.reweight_order = ['1001', '1101']

    def scalar_module(self, driver, factors):
        clustered, tag = driver._cluster_fxfx_event(self.event, record_tag=True)
        return SimpleNamespace(
            original_pdg_list_dict={tag: tag},
            ewsudakov=mock.Mock(return_value=np.array([1., factors[2]-1., factors[0]-1.,
                                                      factors[1]-1., factors[1]-1., factors[0]-1.])))

    def assert_factors(self, driver, result, factors, event=None):
        event = event if event is not None else self.event
        self.assertEqual(set(result), set(driver._sudakov_weight_metadata) | {'orig'})
        for source, weight in event.parse_reweight().items():
            for v, factor in enumerate(factors):
                key = driver._sudakov_weight_map[0, v, source]
                self.assertAlmostEqual(result[key], weight * factor, places=9)

    def test_scalar_fxfx_central_factor_controls_all_variants(self):
        for factors, expected in self.cases:
            with self.subTest(factors=factors):
                driver = FactorDriver()
                module = self.scalar_module(driver, factors)
                result = driver._compute_ewsudakov_fxfx_reweight(self.event, module)
                self.assertTrue(module.ewsudakov.called)
                self.assert_factors(driver, result, expected)

    def test_density_scalar_fallback_uses_central_factor(self):
        for factors, expected in self.cases:
            with self.subTest(factors=factors):
                driver = FactorDriver()
                module = self.scalar_module(driver, factors)
                result = driver._compute_density_ewsudakov_reweight(self.event, module)
                self.assertEqual(driver._density_logger.log_event.call_args.kwargs['mode'], 'scalar')
                self.assert_factors(driver, result, expected)

    def test_density_contraction_controls_all_variants(self):
        for factors, expected in self.cases:
            with self.subTest(factors=factors):
                driver = FactorDriver()
                module = self.scalar_module(driver, [1.1, 0.9, factors[2]])
                module.density_sudakov = mock.Mock(return_value={
                    'density_delta': np.full(9, (factors[0] - 1.) / 2),
                    'density_delta_var': np.full(9, (factors[1] - 1.) / 2),
                    'born_diag': np.ones(9), 'total_dim': 9, 'nres': 2, 'results': [9.]})
                driver.density_prod_id_to_path = {'process': True}
                driver.density_prod_f2pylib = {'module': True}
                driver._get_decay_density_matrix = mock.Mock(return_value=
                    fx.DensityMatrix.from_full_matrix(np.eye(9), [24, -24]))
                with mock.patch.object(fx, 'DensityHelper') as helper:
                    helper.return_value.compute_density_matrix.return_value = np.eye(9)
                    result = driver._compute_density_ewsudakov_reweight(self.event, module)
                self.assertEqual(driver._density_logger.log_event.call_args.kwargs['mode'], 'density')
                self.assert_factors(driver, result, expected)

    def test_lo_five_variant_set_uses_central_factor(self):
        event = make_event([
            (2, -1, np.array([500., 0, 0, 500.])),
            (-2, -1, np.array([500., 0, 0, -500.])),
            (-13, 1, np.array([500., 300., 0, 400.])),
            (13, 1, np.array([500., -300., 0, -400.]))])
        event.wgt = 2.
        event.parse_reweight()
        event.reweight_data = {'1001': 2., '1101': 3.}
        event.reweight_order = ['1001', '1101']
        tag, order = event.get_tag_and_order()
        tag = (tuple(order[0]), tuple(sorted(order[1])))
        for factors, expected in self.cases:
            with self.subTest(factors=factors):
                driver = FactorDriver()
                driver.banner = SimpleNamespace(get_detail=lambda *args: 'u u~ > mu+ mu-')
                driver.mg5cmd = None
                driver._configure_sudakov_weight_ids(event.parse_reweight(), [1.5], driver.SUDAKOV_LO_VARIANT_NAMES)
                module = SimpleNamespace(original_pdg_list_dict={tag: order}, ewsudakov=mock.Mock(
                    return_value=[1., factors[2]-1., factors[0]-1., factors[1]-1., 21., 32.]))
                result = ReweightInterface._compute_ewsudakov_reweight(driver, event, module)
                extra = [1., 1.] if abs(factors[0]) > 200 else [22., 33.]
                self.assert_factors(driver, result, expected + extra, event)

    def test_accounting_counts_central_neutralization(self):
        with mock.patch.object(fx, 'CURRENT_EVENT_ID', 1):
            fx.limit_sudakov_factors([201., 2., 3.], 'events_runaway_damped_density')
            fx.limit_sudakov_factors([-201., 2., 3.], 'events_runaway_damped_scalar')
            fx.limit_sudakov_factors([1., 300., 400.], 'events_runaway_damped_density')
        self.assertEqual(fx.FKS_QUALITY_COUNTS,
                         {'events_runaway_damped_density': 1, 'events_runaway_damped_scalar': 1})


if __name__ == '__main__':
    unittest.main()
