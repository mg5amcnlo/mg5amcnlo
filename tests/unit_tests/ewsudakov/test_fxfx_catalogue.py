"""Process coverage for complete FxFx inputs and xi scans."""

import math
import shlex
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from madgraph.various import fxfx_ewsudakov as fx, lhe_parser
from tests.unit_tests.ewsudakov.test_fxfx_decay_transport import ww_event


class CatalogueDriver(fx.FxFxEWSudakovMixin):
    def __init__(self, path):
        self.inc_sudakov = True
        self.banner = SimpleNamespace(get=lambda *args: 3)
        self.lhe_input = lhe_parser.EventFile(str(path))
        self.model = None
        self.fxfx_clustered_final_states = set()
        self.fxfx_clustered_decay_processes = set()
        self._fxfx_cluster_catalog_ready = False
        self.sudakov_xi_scan = [.01, 1.5]
        self.mW2_cluster_scale = 1.5

    def _init_pole_masses_from_banner(self):
        pass


class TestFxFxCatalogue(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.first = ww_event([np.array([50., 20, 10, math.sqrt(2000)])])
        self.second = ww_event([np.array([50., 20, 10, math.sqrt(2000)])])
        self.second[0].pid, self.second[1].pid = 1, -1
        self.second.tag = '<clustering step_1="(1,5);mom=1"></clustering>'
        self.full = self.write_events('complete.lhe', [self.first, self.second])
        self.chunk = self.write_events('chunk.lhe', [self.first])
        self.driver = CatalogueDriver(self.chunk)
        self.addCleanup(self.driver.lhe_input.close)

    def write_events(self, name, events):
        path = Path(self.temp.name) / name
        path.write_text('<LesHouchesEvents version="3.0">\n' +
                        ''.join(str(event) for event in events) + '</LesHouchesEvents>\n')
        return path

    def test_complete_input_covers_all_channels_and_xis(self):
        self.driver.fxfx_catalogue_source = str(self.full)
        self.driver._ensure_fxfx_cluster_catalogue()
        expected = {(initial, final) for initial in ((2, -2), (1, -1))
                    for final in ((-24, 21, 24), (-24, 24))}
        self.assertEqual(self.driver.fxfx_clustered_final_states, expected)
        self.assertEqual(self.driver.fxfx_clustered_decay_processes,
                         {(24, (-13, 14)), (-24, (-12, 11))})
        self.assertEqual(self.driver.mW2_cluster_scale, 1.5)

    def test_scan_keeps_worker_stream_position(self):
        first = next(self.driver.lhe_input)
        self.driver.fxfx_catalogue_source = str(self.full)
        self.driver._ensure_fxfx_cluster_catalogue()
        self.assertFalse(self.driver.lhe_input.closed)
        self.assertEqual(first[0].pid, 2)
        self.assertRaises(StopIteration, next, self.driver.lhe_input)

    def test_single_input_and_catalogue_reuse(self):
        self.driver._ensure_fxfx_cluster_catalogue()
        self.assertEqual(len(self.driver.fxfx_clustered_final_states), 2)
        with mock.patch.object(self.driver, '_cluster_fxfx_event', side_effect=AssertionError('Unexpected scan')):
            self.driver._ensure_fxfx_cluster_catalogue()

    def test_xi_change_collects_required_processes(self):
        self.driver.sudakov_xi_scan = [.01]
        self.driver._ensure_fxfx_cluster_catalogue()
        self.assertEqual(self.driver.fxfx_clustered_final_states, {((2, -2), (-24, 21, 24))})
        self.driver.sudakov_xi_scan = [1.5]
        self.driver._ensure_fxfx_cluster_catalogue()
        self.assertEqual(self.driver.fxfx_clustered_final_states, {((2, -2), (-24, 24))})

    def test_source_change_collects_required_processes(self):
        self.driver._ensure_fxfx_cluster_catalogue()
        self.driver.fxfx_catalogue_source = str(self.full)
        self.driver._ensure_fxfx_cluster_catalogue()
        self.assertEqual(len(self.driver.fxfx_clustered_final_states), 4)

    def test_compiled_library_requires_complete_coverage(self):
        self.driver.fxfx_catalogue_source = str(self.full)
        self.driver._ensure_fxfx_cluster_catalogue()
        module = SimpleNamespace(original_pdg_list_dict={((2, -2), (-24, 24)): None})
        with self.assertRaisesRegex(fx.MissingSudakovProcessError, 'complete event input'):
            self.driver._validate_fxfx_sudakov_catalogue(module)
        module.original_pdg_list_dict = dict.fromkeys(self.driver.fxfx_clustered_final_states)
        self.driver._validate_fxfx_sudakov_catalogue(module)

    def test_single_particle_passthrough_requires_no_library_entry(self):
        self.driver.fxfx_clustered_final_states = {((2, -2), (23,))}
        self.driver._validate_fxfx_sudakov_catalogue(SimpleNamespace(original_pdg_list_dict={}))

    def test_both_drivers_report_missing_process(self):
        module = SimpleNamespace(original_pdg_list_dict={})
        for name in ('_compute_ewsudakov_fxfx_reweight', '_compute_density_ewsudakov_reweight'):
            with self.subTest(driver=name):
                with self.assertRaisesRegex(fx.MissingSudakovProcessError, 'Sudakov library has no process'):
                    getattr(self.driver, name)(self.first, module)

    def test_scan_restores_xi_on_error(self):
        with mock.patch.object(self.driver, '_cluster_fxfx_event', side_effect=ValueError('Invalid event')):
            with self.assertRaisesRegex(ValueError, 'Invalid event'):
                self.driver._ensure_fxfx_cluster_catalogue()
        self.assertEqual(self.driver.mW2_cluster_scale, 1.5)
        self.assertFalse(self.driver._fxfx_cluster_catalog_ready)

    def test_multicore_creator_scans_unsplit_input(self):
        from madgraph.interface import common_run_interface as common
        from madgraph.interface import reweight_interface as reweight

        root = Path(self.temp.name)
        (root / 'Cards').mkdir()
        (root / 'Cards/reweight_card.dat').write_text('change include_sudakov True\nlaunch\n')
        workers = []
        scheduler = mock.Mock(spec=common.cluster.MultiCore)
        card = {'nevents': 1000, 'nevt_job': 1, 'ickkw': 3}

        def command(force):
            return SimpleNamespace(
                me_dir=str(root), options={'nb_core': 2, 'run_mode': 2},
                force_run=force, run_name='run', run_tag='tag', run_card=card,
                results=mock.Mock(current={'cross': 1}), cluster=scheduler,
                split_arg=shlex.split, to_store=[], update_status=mock.Mock(),
                check_decay_events=lambda args: args.__setitem__(slice(None), [args[0]]))

        def interface(path, **kwargs):
            worker = CatalogueDriver(path)
            self.addCleanup(worker.lhe_input.close)
            worker.get_weight_names = lambda: ['']
            worker.has_nlo = False
            worker.import_command_file = lambda path: worker._ensure_fxfx_cluster_catalogue()
            worker.do_quit = lambda line: None
            workers.append(worker)
            return worker

        def run_workers(*args):
            for submitted in scheduler.submit.call_args_list:
                arguments = submitted.kwargs['argument']
                arguments = arguments[arguments.index('reweight') + 1:]
                common.CommonRunCmd.do_reweight(command(True), shlex.join(arguments))

        scheduler.wait.side_effect = run_workers
        combined = mock.Mock()
        combined.write.return_value = (2, {})
        with mock.patch.object(reweight, 'ReweightInterface', side_effect=interface), \
                mock.patch.object(common.banner_mod, 'RunCard', return_value=card), \
                mock.patch.object(lhe_parser.EventFile, 'get_banner', return_value={'mgruncard': ''}), \
                mock.patch.object(lhe_parser, 'MultiEventFile', return_value=combined):
            common.CommonRunCmd.do_reweight(command(False), '%s -from_cards' % self.full)
        self.assertEqual(len(workers), 2)
        self.assertEqual(workers[0].fxfx_catalogue_source, str(self.full))
        self.assertIsNone(workers[1].fxfx_catalogue_source)
        self.assertEqual(len(workers[0].fxfx_clustered_final_states), 4)
        self.assertEqual(len(workers[1].fxfx_clustered_final_states), 2)
        library = SimpleNamespace(original_pdg_list_dict=dict.fromkeys(workers[0].fxfx_clustered_final_states))
        workers[1]._validate_fxfx_sudakov_catalogue(library)


if __name__ == '__main__':
    unittest.main()
