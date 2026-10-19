"""Checks for benchmark isolation and stable archived-result extraction."""
import gzip
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.input_files.benchmark_virtual_quality import (
    clone_export, compare, observables, run_one, summarize)


HAS_NUMPY = importlib.util.find_spec('numpy') is not None


class TestVirtualQualityBenchmark(unittest.TestCase):
    @unittest.skipUnless(HAS_NUMPY, 'requires numpy')
    def test_lhe_event_attributes_are_read_without_matching_event_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'events.lhe.gz'
            particles = ('1 1 0 0 501 0 3 4 0 5 0 0 9\n'
                         '-1 1 0 0 0 501 -3 -4 0 5 0 0 9\n')
            with gzip.open(path, 'wt') as out:
                out.write('<eventgroup nreal="1" ncounter="0">\n'
                          "<event npLO=' -1 ' npNLO=' 1 '>\n"
                          '2 1 2.0 91.0 0.007 0.118\n' + particles +
                          '</event>\n</eventgroup>\n<event>\n'
                          '2 1 -3.0 91.0 0.007 0.118\n' + particles +
                          '</event>\n')
            values, weights = observables(path)
            self.assertEqual(weights.tolist(), [2.0, -3.0])
            self.assertEqual(values.tolist(), [[10.0, 5.0, 0.0, 10.0]] * 2)

    @unittest.skipUnless(HAS_NUMPY, 'requires numpy')
    def test_comparison_handles_empty_bins_and_zero_reference_calls(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / 'run.json'
            report.write_text(json.dumps(dict(
                process_dir=str(root), run_name='test',
                stages={'2': {'virtual_points': 0}})))
            values = np.array([[0.] * 4, [10.] * 4])
            weights = np.ones(2)
            with mock.patch('tests.input_files.benchmark_virtual_quality.observables',
                            return_value=(values, weights)):
                result = compare(report, report)
            self.assertIsNone(result['generation_call_ratio'])
            self.assertIsNone(result['total_call_ratio'])
            self.assertTrue(result['distributions'])
            for distribution in result['distributions'].values():
                self.assertIn(0., distribution['baseline_variance'])
                self.assertEqual(distribution['pulls'], [0.] * distribution['bins'])
            json.dumps(result, allow_nan=False)

    def test_successful_report_is_reused_only_with_matching_run_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / 'pool_1731.json'
            settings = {'ebeam1': 6500, 'ebeam2': 6500}
            report.write_text(json.dumps(dict(
                completed=True, events_requested=10000, seed=1731,
                req_acc=.02, mode='pool', run_settings=settings, returncode=0)))
            self.assertEqual(run_one(root, 'pool', 10000, 1731, .02, 2, settings), report)
            with self.assertRaisesRegex(RuntimeError, 'different configuration'):
                run_one(root, 'pool', 10000, 1731, .02, 2,
                        dict(settings, ebeam1=7000))
            foreign = json.loads(report.read_text())
            foreign['process_dir'] = str(root/'another_export')
            report.write_text(json.dumps(foreign))
            with self.assertRaisesRegex(RuntimeError, 'different configuration'):
                run_one(root, 'pool', 10000, 1731, .02, 2, settings)
            failed = json.loads(report.read_text())
            failed['returncode'] = 1
            report.write_text(json.dumps(failed))
            with self.assertRaisesRegex(RuntimeError, 'different configuration'):
                run_one(root, 'pool', 10000, 1731, .02, 2, settings)

    def test_clone_preserves_relative_link_contents_without_source_aliases(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / 'source'
            (source / 'Source').mkdir(parents=True)
            (source / 'SubProcesses/P0_test/GF1.0').mkdir(parents=True)
            include = source / 'SubProcesses/timing_variables.inc'
            include.write_text('original include\n')
            (source / 'Source/timing_variables.inc').symlink_to(
                '../SubProcesses/timing_variables.inc')
            (source / 'Source/absolute.inc').symlink_to(include)
            external = base / 'external.inc'
            external.write_text('external original\n')
            (source / 'Source/external.inc').symlink_to(external)
            (source / 'Source/missing.inc').symlink_to('../missing.inc')
            (source / 'SubProcesses/job_status.pkl').write_text('old jobs')
            (source/'pool_3731.json').write_text('old benchmark report')
            destination = clone_export(source, base / 'clone')
            copied = destination / 'Source/timing_variables.inc'
            self.assertTrue(copied.is_file())
            self.assertEqual(copied.read_text(), 'original include\n')
            self.assertTrue(copied.is_symlink())
            self.assertEqual(copied.resolve(), destination / 'SubProcesses/timing_variables.inc')
            copied.write_text('modified include\n')
            self.assertEqual(include.read_text(), 'original include\n')
            self.assertEqual((destination / 'Source/absolute.inc').read_text(), 'modified include\n')
            external_copy = destination / 'Source/external.inc'
            self.assertFalse(external_copy.is_symlink())
            external_copy.write_text('external clone\n')
            self.assertEqual(external.read_text(), 'external original\n')
            self.assertFalse((destination / 'Source/missing.inc').exists())
            self.assertFalse((destination / 'SubProcesses/job_status.pkl').exists())
            self.assertFalse((destination/'pool_3731.json').exists())
            self.assertFalse((destination / 'SubProcesses/P0_test/GF1.0').exists())

    def test_clone_keeps_worker_configuration_and_seed_linked_to_clone(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / 'source'
            (source / 'Cards').mkdir(parents=True)
            process = source / 'SubProcesses/P0_test'
            process.mkdir(parents=True)
            (source / 'Cards/FKS_params.dat').write_text('pooled true\n')
            (source / 'SubProcesses/FKS_params.dat').symlink_to('../Cards/FKS_params.dat')
            (process / 'FKS_params.dat').symlink_to('../FKS_params.dat')
            (source / 'SubProcesses/randinit').write_text('r=32\n')
            (process / 'randinit').symlink_to('../randinit')
            clone = clone_export(source, base / 'clone')
            (clone / 'Cards/FKS_params.dat').write_text('pooled false\n')
            (clone / 'SubProcesses/randinit').write_text('r=1731\n')
            worker = clone / 'SubProcesses/P0_test'
            self.assertEqual((worker / 'FKS_params.dat').read_text(), 'pooled false\n')
            self.assertEqual((worker / 'randinit').read_text(), 'r=1731\n')
            self.assertEqual((process / 'FKS_params.dat').read_text(), 'pooled true\n')
            self.assertEqual((process / 'randinit').read_text(), 'r=32\n')

    def test_fresh_clone_retains_links_to_includes_generated_at_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base/'source'
            (source/'SubProcesses/P0_test').mkdir(parents=True)
            (source/'Source').mkdir()
            (source/'SubProcesses/P0_test/run_card.inc').symlink_to('../run_card.inc')
            (source/'Source/run_card.inc').symlink_to('../SubProcesses/run_card.inc')
            destination = clone_export(source, base/'clone')
            (destination/'SubProcesses/run_card.inc').write_text('generated in clone\n')
            self.assertEqual((destination/'SubProcesses/P0_test/run_card.inc').read_text(), 'generated in clone\n')
            self.assertEqual((destination/'Source/run_card.inc').read_text(), 'generated in clone\n')
            self.assertFalse((source/'SubProcesses/run_card.inc').exists())

    def test_summary_uses_archived_channels_without_counting_split_copies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archived = root / 'Events/test/native_snapshot/P0_test'
            for name in ('GF1.0', 'GF1.0_1', 'GF1.0_2'):
                channel = archived / name
                channel.mkdir(parents=True)
                (channel / 'mint_grids').write_text(
                    ' MAX 100 200\n MAX 40\n'
                    ' SUM 10 9 -1 0 2 0\n'
                    ' QSM 0.4 0.3 0 0 0.2 0\n'
                    ' MAX 0.15\nVEP 3\n')
            current = root / 'SubProcesses/P0_test/GF1.0'
            current.mkdir(parents=True)
            (current / 'mint_grids').write_text(' SUM 100 90 -10 0 20 0\n')
            result = summarize(root, 'test')
            self.assertEqual(result['cross_section_pb'], 9)
            self.assertEqual(result['cross_section_error_pb'], 0.3)
            self.assertEqual(result['residual_absolute_integral_pb'], 2)
            self.assertEqual(result['native_channel_grids'][
                'SubProcesses/P0_test/GF1.0']['virtual_maxima'], [40])
            self.assertEqual(result['native_channel_grids'][
                'SubProcesses/P0_test/GF1.0']['virtual_born_carrier_bounds'], [0.15])


    def test_calibration_calls_remain_in_total_virtual_cost(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            eventdir = root / 'Events/test'
            eventdir.mkdir(parents=True)
            (root / 'SubProcesses').mkdir()
            (eventdir / 'alllogs_2.html').write_text(
                '<PRE>\n'
                'Virtual calibration phase-space attempts / evaluated residuals: 140 128\n'
                'Virtual calibration overflows / largest ratio / last excess: 1 1.4 0.064\n'
                'Virtual calibration validated / scalar fallback: F T\n'
                'events generated, virt: 12\n'
                'Virtual Born carrier ratio bound: 0.15\n'
                'Virtual Born carrier attempts / vetoes: 44 8\n'
                'Virtual proposal attempts / evaluated residuals: 40 36\n'
                'Virtual proposal base norm / estimated effective norm: 0.2 0.18\n'
                'Virtual envelope overflows / excess fraction: 1 0.037\n'
                'Total points tried: 164\n</PRE>\n')
            result = summarize(root, 'test')['stages']['2']
            self.assertEqual(result['virtual_points'], 164)
            self.assertEqual(result['virtual_calibration_residual_evaluations'], 128)
            self.assertEqual(result['virtual_proposal_evaluated_residuals'], 36)
            self.assertEqual(result['virtual_calibration_validated_channels'], [False])
            self.assertEqual(result['virtual_born_carrier_attempts'], 44)
            self.assertEqual(result['virtual_born_carrier_vetoes'], 8)
            self.assertEqual(result['virtual_born_carrier_ratio_bounds_by_channel'], [0.15])
            self.assertEqual(result['virtual_envelope_overflows_by_channel'],
                             [{'count': 1, 'excess_fraction': 0.037}])

    def test_vep4_keeps_plain_and_velocity_bounds_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            eventdir = root/'Events/test'
            grid = eventdir/'native_snapshot/P0_test/GF1.0'
            grid.mkdir(parents=True)
            (root/'SubProcesses').mkdir()
            (grid/'mint_grids').write_text(
                ' MAX 100 200\n MAX 40\n SUM 10 9 -1 0 2 0\n'
                ' QSM .4 .3 0 0 .2 0\n MAX .15 .025\nVEP 4\n')
            (eventdir/'alllogs_2.html').write_text(
                '<PRE>\nVirtual Born carrier ratio bound: .15\n'
                'Virtual velocity carrier ratio bound: .025\n</PRE>\n')
            result = summarize(root, 'test')
            channel = result['native_channel_grids']['SubProcesses/P0_test/GF1.0']
            self.assertEqual(channel['virtual_maxima'], [40])
            self.assertEqual(channel['virtual_born_carrier_bounds'], [.15])
            self.assertEqual(channel['virtual_velocity_carrier_bounds'], [.025])
            self.assertEqual(result['stages']['2'][
                'virtual_velocity_carrier_ratio_bounds_by_channel'], [.025])

    def test_missing_overflow_diagnostics_are_not_reported_as_zero_excess(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            eventdir = root / 'Events/test'
            grid = eventdir / 'native_snapshot/P0_test/GF1.0'
            grid.mkdir(parents=True)
            (grid / 'mint_grids').write_text(' SUM 10 9 -1 0 2 0\n')
            log = eventdir / 'alllogs_2.html'
            log.write_text('LOG file for integration channel P0_test/GF1.0,\n'
                           '<PRE>\nTotal points tried: 12\n</PRE>\n')
            self.assertNotIn('estimated_envelope_excess_pb', summarize(root, 'test'))
            log.write_text(log.read_text().replace('</PRE>',
                           'Virtual envelope overflows / excess fraction: 1 0.05\n</PRE>'))
            result = summarize(root, 'test')
            self.assertAlmostEqual(result['estimated_envelope_excess_pb'], 0.1)
            self.assertAlmostEqual(
                result['estimated_envelope_excess_fraction_of_absolute_total'], 0.1 / 12)
            log.write_text('LOG file for integration channel P0_test/GF1.0,\n'
                           '<PRE>\nTotal points tried: 128\n'
                           'Generation efficiencies: 0.21 -1.0\n</PRE>\n')
            result = summarize(root, 'test')
            self.assertEqual(result['estimated_envelope_excess_pb'], 0)
            self.assertTrue(result['stages']['2']['channels'][0][
                'residual_generation_not_sampled'])


if __name__ == '__main__':
    unittest.main()
