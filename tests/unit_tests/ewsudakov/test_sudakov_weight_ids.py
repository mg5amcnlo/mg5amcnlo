"""Unique Sudakov weight IDs and their LHE descriptions."""

import contextlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
from xml.etree import ElementTree as ET

from madgraph.interface.reweight_interface import ReweightInterface
from madgraph.various import banner, fxfx_ewsudakov as fx, lhe_parser
from models import check_param_card
from tests.unit_tests.ewsudakov.test_fxfx_decay_transport import ww_event


class TestSudakovWeightIds(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def event(self, sources):
        event = ww_event([])
        event.wgt = 2.0
        event.parse_reweight()
        event.reweight_data = dict(sources)
        event.reweight_order = list(sources)
        return event

    def driver(self, events, header='', mode=3):
        path = self.root / 'events.lhe'
        path.write_text('<LesHouchesEvents version="3.0">\n' +
                        ''.join(map(str, events)) + '</LesHouchesEvents>\n')
        driver = ReweightInterface.__new__(ReweightInterface)
        driver.exitted = True
        driver.inc_sudakov = True
        driver.banner = banner.Banner()
        driver.banner.lhe_version = 3.0
        driver.banner.run_card = {'ickkw': mode}
        driver.banner['initrwgt'] = header
        driver.banner['slha'] = 'BLOCK MASS\n 24 80.379\n'
        driver.banner['init'] = '2212 2212 6500 6500 0 0 0 0 -4 1\n1 0 1 1\n'
        driver.lhe_input = lhe_parser.EventFile(str(path))
        self.addCleanup(driver.lhe_input.close)
        driver.fxfx_catalogue_source = None
        driver.sudakov_xi_scan = [0.5, 1.5]
        driver.mW2_cluster_scale = 1.5
        driver._sudakov_source_ids = set().union(*(set(e.parse_reweight() or {'1001': e.wgt}) for e in events))
        driver._collect_sudakov_weight_sources()
        driver._prepare_sudakov_weight_ids()
        driver.rwgt_dir = str(self.root)
        driver.me_dir = str(self.root)
        driver.second_model = driver.second_process = driver.dedicated_path = None
        driver.nb_library = 1
        driver.options = {'rwgt_name': None, 'allow_missing_finalstate': False,
                          'identical_particle_in_prod_and_decay': False}
        driver.flag_density_matrix = False
        driver.new_param_card = check_param_card.ParamCard(driver.banner['slha'].splitlines())
        driver.has_nlo = False
        driver.rwgt_mode = 'LO'
        driver.output_type = 'default'
        driver.f2pylib = {}
        driver.mother = None
        driver.systematics = False
        driver.multicore = False
        driver.id_to_path = {}
        driver.processes = []
        driver.all_cross_section = {}
        driver.terminate_fortran_executables = mock.Mock()
        driver._validate_fxfx_sudakov_catalogue = mock.Mock()
        (self.root / 'rw_me').mkdir(exist_ok=True)
        return driver

    def test_source_suffixes_and_large_numeric_ids(self):
        sources = {str(i): float(i) for i in range(1001, 2202)}
        event = self.event(sources)
        driver = self.driver([event])
        outputs = set(driver._sudakov_weight_metadata)
        self.assertEqual(len(outputs), 1201 * 6)
        self.assertFalse(outputs & set(sources))
        self.assertTrue(all(key.isdigit() for key in outputs))
        for k in range(2):
            driver._current_xi_idx = k
            result = driver._build_sudakov_rwgt_dict(event, [1.1, 0.9, 1.05])
            for source, value in sources.items():
                for v, factor in enumerate([1.1, 0.9, 1.05]):
                    self.assertEqual(result[driver._sudakov_weight_map[k, v, source]], value * factor)

    def test_numeric_allocation_is_deterministic(self):
        sources = ['1001', '1101', '10000', 'pdf_member', '2001', '010001']
        a, b = fx.FxFxEWSudakovMixin(), fx.FxFxEWSudakovMixin()
        a._configure_sudakov_weight_ids(sources, [0.5, 1.5])
        b._configure_sudakov_weight_ids(reversed(sources), [0.5, 1.5])
        self.assertEqual(a._sudakov_weight_map, b._sudakov_weight_map)
        self.assertNotIn('10001', a._sudakov_weight_metadata)

    def test_scale_weight_ids(self):
        driver = self.driver([self.event({str(i): i for i in range(1001, 1046)})])
        for k in range(2):
            for v in range(3):
                for source in range(1001, 1046):
                    self.assertEqual(driver._sudakov_weight_map[k, v, str(source)],
                                     str((20 + 5*k + v)*100 + source - 1000))

    def test_header_contains_full_source_and_variant_metadata(self):
        header = '<weightgroup name="PDF"><weight id="1101">PDF member</weight></weightgroup>'
        driver = self.driver([self.event({'1001': 2., '1101': 3., 'pdf_A': 4.})], header)
        ids = list(driver._sudakov_weight_metadata)
        driver.handle_param_card('', ['--keep_card'], ids)
        tree = ET.fromstring('<initrwgt>' + driver.banner['initrwgt'] + '</initrwgt>')
        entries = tree.findall("weightgroup[@name='ew_sudakov']/weight")
        self.assertEqual([e.get('id') for e in entries], ids)
        self.assertEqual(len(entries), 18)
        all_ids = [e.get('id') for e in tree.iter('weight')]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        for entry in entries:
            source, k, xi, variant = driver._sudakov_weight_metadata[entry.get('id')]
            self.assertEqual(entry.get('source_id'), source)
            self.assertEqual(int(entry.get('xi_index')), k)
            self.assertEqual(float(entry.get('xi')), xi)
            self.assertEqual(entry.get('variant'), variant)
        self.assertEqual(tree.find("weightgroup[@name='PDF']/weight").text, 'PDF member')

    def test_header_identifies_sudakov_inputs(self):
        for header in ['<weightgroup name="ew_sudakov"><weight id="10000">source=1001</weight></weightgroup>',
                       '<weightgroup name="mg_reweighting"><weight id="2001">scale_1001_sud xi=1.5</weight></weightgroup>']:
            with self.subTest(header=header), self.assertRaisesRegex(ValueError, 'Input declares EW Sudakov'):
                self.driver([self.event({'1001': 1., '2001': 2.})], header)

    def test_lo_path_emits_five_declared_variants(self):
        driver = self.driver([self.event({'1001': 2., '1101': 3.})], mode=0)
        self.assertEqual(len(driver._sudakov_weight_metadata), 10)
        result = driver._build_sudakov_rwgt_dict(self.event({'1001': 2., '1101': 3.}), [1, 2, 3, 4, 5])
        self.assertEqual(set(result) - {'orig'}, set(driver._sudakov_weight_metadata))

    def test_shared_layout_contains_sources_from_all_chunks(self):
        creator = self.driver([self.event({'1001': 2., '1101': 3.})])
        creator.save_to_pickle()
        waiter = self.driver([self.event({'1101': 3.})])
        waiter.multicore = 'wait'
        waiter.load_from_pickle()
        waiter._prepare_sudakov_weight_ids()
        self.assertEqual(waiter._sudakov_weight_map, creator._sudakov_weight_map)

    def test_source_scan_keeps_input_position(self):
        driver = self.driver([self.event({'1001': 1.}), self.event({'1101': 2.})], mode=0)
        next(driver.lhe_input)
        driver._collect_sudakov_weight_sources()
        self.assertEqual(driver._sudakov_source_ids, {'1001', '1101'})
        self.assertEqual(set(next(driver.lhe_input).parse_reweight()), {'1101'})

    def test_empty_source_uses_nominal_weight(self):
        event = self.event({})
        driver = self.driver([event])
        result = driver._build_sudakov_rwgt_dict(event, [1.1, 0.9, 1.05])
        self.assertEqual(result['2001'], event.wgt * 1.1)

    def test_event_sources_require_declared_layout(self):
        driver = self.driver([self.event({'1001': 2.})])
        with self.assertRaisesRegex(ValueError, 'does not contain source/xi/variant'):
            driver._build_sudakov_rwgt_dict(self.event({'1101': 3.}), [1., 1., 1.])

    def test_launch_prepares_ids_before_header_and_events(self):
        driver = self.driver([self.event({'1001': 2., '1101': 3.})])
        driver.split_arg = lambda line: ['--keep_card']
        driver.check_launch = lambda args: {'rwgt_name': None, 'rwgt_info': None}
        driver.mg5cmd = None
        driver.banner.proc_card = SimpleNamespace(get=lambda key: 'sm')
        driver.load_model = mock.Mock()
        driver._ensure_fxfx_cluster_catalogue = mock.Mock()
        driver.has_standalone_dir = True
        driver.get_weight_names = lambda: []
        driver.launch_actual_reweighting = mock.Mock()
        driver.do_launch('--keep_card')
        args = driver.launch_actual_reweighting.call_args.args
        self.assertEqual(args[2], list(driver._sudakov_weight_metadata))
        tree = ET.fromstring('<initrwgt>' + driver.banner['initrwgt'] + '</initrwgt>')
        self.assertEqual([e.get('id') for e in tree.findall("weightgroup[@name='ew_sudakov']/weight")], args[2])
        self.assertEqual(len(args[2]), 12)

    def test_extended_xi_scan_reserves_all_candidate_ids(self):
        driver = fx.FxFxEWSudakovMixin()
        driver._configure_sudakov_weight_ids(['1001', '1101'], list(range(20)))
        self.assertEqual(len(driver._sudakov_weight_metadata), 120)
        self.assertEqual(driver._sudakov_weight_map[16, 0, '1001'], '10001')
        self.assertNotEqual(driver._sudakov_weight_map[0, 0, '1101'], '10001')

    def test_lhe_writers_follow_complete_weight_layout(self):
        for output_type in ['default', '2.0']:
            with self.subTest(output_type=output_type):
                event = self.event({'1001': 2., '1101': 3., '2001': 4.})
                driver = self.driver([event])
                driver.output_type = output_type
                ids = list(driver._sudakov_weight_metadata)
                driver.handle_param_card('', ['--keep_card'], ids)
                driver._compute_density_ewsudakov_reweight = lambda e, module: driver._build_sudakov_rwgt_dict(e, [1.1, 0.9, 1.05])
                with mock.patch('importlib.import_module', return_value=SimpleNamespace()):
                    driver.launch_actual_reweighting([], '', ids, str(self.root))
                if output_type == 'default':
                    paths = [self.root / 'events_out.lhe']
                else:
                    paths = list(self.root.glob('rwgt_events*.lhe.gz'))
                    self.assertEqual(len(paths), 18)
                for path in paths:
                    with contextlib.closing(lhe_parser.EventFile(str(path))) as stream:
                        written = list(stream)
                    self.assertEqual(len(written), 1)
                    if output_type == 'default':
                        data = written[0].parse_reweight()
                        self.assertEqual(set(data), set(ids) | {'1001', '1101', '2001'})
                        for source, value in event.parse_reweight().items():
                            self.assertEqual(data[source], value)
                            for k in range(2):
                                for v, factor in enumerate([1.1, 0.9, 1.05]):
                                    self.assertAlmostEqual(data[driver._sudakov_weight_map[k, v, source]], value * factor)
                    else:
                        key = path.name[len('rwgt_events'):].split('_')[0]
                        source, k, xi, variant = driver._sudakov_weight_metadata[key]
                        factor = dict(zip(driver.SUDAKOV_VARIANT_NAMES, [1.1, 0.9, 1.05]))[variant]
                        self.assertAlmostEqual(written[0].wgt, event.parse_reweight()[source] * factor)


if __name__ == '__main__':
    unittest.main()
