"""Pooled fits synchronize once and travel with independent integration jobs."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from madgraph.interface.amcatnlo_run_interface import aMCatNLOCmd, aMCatNLOError


class TestVirtualPoolController(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='mg5_virtual_pool_')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'Cards').mkdir()
        (self.root / 'Cards' / 'FKS_params.dat').write_text('#UsePooledVirtual\nT\n')
        for name in ('P0_gg', 'P1_qq'):
            (self.root / 'SubProcesses' / name).mkdir(parents=True)
        self.command = object.__new__(aMCatNLOCmd)
        self.command.stop_for_runweb = True
        self.command.me_dir = str(self.root)
        self.command.run_card = {}
        self.command.options = {}

    def configure(self, mode='aMC@NLO', resume=False):
        self.command.configure_virtual_pool({'only_generation': resume},
                                             ['P0_gg', 'P1_qq'], mode)

    def job(self, channel='1', split=0, p_dir='P0_gg', fixed=False):
        name = ('all_G' if fixed else 'GF') + channel
        if split:
            name += '_%d' % split
        directory = self.root / 'SubProcesses' / p_dir / name
        directory.mkdir(exist_ok=True)
        return dict(dirname=str(directory), p_dir=p_dir, channel=channel,
                    split=split, run_mode='all')

    @staticmethod
    def fit(job_dirs, output_model, report_path, job_groups=None):
        Path(output_model).write_text('frozen model for %s\n' % job_dirs)
        Path(report_path).write_text('{}\n')
        return dict(status='trained', model_written=True)

    def test_global_barrier_groups_by_subprocess_and_deduplicates(self):
        self.configure()
        first = self.job(split=1)
        second = self.job(split=2)
        third = self.job(channel='2')
        other = self.job(p_dir='P1_qq')
        with mock.patch('madgraph.various.virtual_pool.train_pool', side_effect=self.fit) as fit:
            self.command.collect_virtual_pool([first, second, third, other, first], 0, False)
            self.assertEqual(fit.call_count, 2)
            args, kwargs = fit.call_args_list[0]
            self.assertEqual(len(args[0]), 3)
            self.assertEqual(kwargs['job_groups'][first['dirname']], '1')
            self.assertEqual(kwargs['job_groups'][second['dirname']], '1')
            self.assertEqual(kwargs['job_groups'][third['dirname']], '2')
            self.command.collect_virtual_pool([first, second], 0, False)
            self.command.collect_virtual_pool([first, second], 1, False)
            self.assertEqual(fit.call_count, 2)

    def test_fixed_order_collects_only_step_one(self):
        self.configure(mode='NLO')
        job = self.job(fixed=True)
        directory = Path(job['dirname'])
        with mock.patch('madgraph.various.virtual_pool.train_pool', side_effect=self.fit) as fit:
            self.command.stage_virtual_pool(job, 0, True)
            self.assertEqual((directory / 'virtual_pool_mode.dat').read_text(), '0\n')
            self.command.collect_virtual_pool([job], 0, True)
            fit.assert_not_called()
            self.command.stage_virtual_pool(job, 1, True)
            self.assertEqual((directory / 'virtual_pool_mode.dat').read_text(), '1\n')
            self.command.collect_virtual_pool([job], 1, True)
            fit.assert_called_once()
            self.command.stage_virtual_pool(job, 2, True)
            self.assertEqual((directory / 'virtual_pool_mode.dat').read_text(), '2\n')
            self.command.collect_virtual_pool([job], 2, True)
            fit.assert_called_once()

    def test_collects_before_result_combination_or_next_job_preparation(self):
        self.configure()
        job = self.job()
        calls = []
        self.command.collect_virtual_pool = lambda *args: calls.append('pool')

        def stop_at_results(*args):
            calls.append('results')
            raise RuntimeError('stop')

        self.command.append_the_results = stop_at_results
        with self.assertRaisesRegex(RuntimeError, 'stop'):
            self.command.collect_the_results({}, .1, [job], [job], 0,
                                             'aMC@NLO', 'all', fixed_order=False)
        self.assertEqual(calls, ['pool', 'results'])

    def test_launch_prepares_capture_and_barrier_waits_for_all_jobs(self):
        self.configure()
        first, second = self.job(), self.job(channel='2')
        self.command.cluster_mode = 0
        self.command.update_status = mock.Mock()
        calls = []

        def run(executable, arguments, run_type, cwd):
            calls.append(arguments[0])
            for job in (first, second):
                self.assertEqual((Path(job['dirname']) / 'virtual_pool_mode.dat').read_text(), '1\n')

        self.command.run_exe = run
        self.command.wait_for_complete = lambda run_type: calls.append('wait')
        self.command.run_all_jobs([first, second], 0, fixed_order=False)
        self.assertEqual(calls, ['1', '2', 'wait'])

    def test_frozen_model_copied_to_new_split_in_prepare_directories(self):
        self.configure()
        job = self.job()
        with mock.patch('madgraph.various.virtual_pool.train_pool', side_effect=self.fit):
            self.command.collect_virtual_pool([job], 0, False)
        for name in ('grid.MC_integer', 'mint_grids', 'res_1'):
            (Path(job['dirname']) / name).write_text('grid\n')
        split = dict(job, split=3)
        self.command.write_input_file = mock.Mock()
        self.command.prepare_directories([split], 'aMC@NLO', fixed_order=False)
        destination = Path(split['dirname'])
        source = self.root / 'SubProcesses' / 'P0_gg' / 'virtual_pool_model.dat'
        self.assertEqual(destination.name, 'GF1_3')
        self.assertEqual((destination / 'virtual_pool_model.dat').read_bytes(), source.read_bytes())
        self.assertFalse((destination / 'virtual_pool_model.dat').is_symlink())
        self.assertEqual((destination / 'virtual_pool_mode.dat').read_text(), '2\n')

    def test_resume_reuses_model_and_does_not_fit(self):
        self.configure()
        job = self.job()
        with mock.patch('madgraph.various.virtual_pool.train_pool', side_effect=self.fit) as fit:
            self.command.collect_virtual_pool([job], 0, False)
            # Workers evaluate frozen models in Fortran; losing NumPy on
            # the controller must not change an existing event envelope.
            with mock.patch.dict(sys.modules, {'numpy': None}):
                self.configure(resume=True)
                self.command.collect_virtual_pool([job], 1, False)
                self.command.stage_virtual_pool(job, 2, False)
            fit.assert_called_once()
        self.assertEqual((Path(job['dirname']) / 'virtual_pool_mode.dat').read_text(), '2\n')

    def test_fixed_order_split_gets_frozen_model_and_grid_copies(self):
        self.configure(mode='NLO')
        job = self.job(fixed=True)
        with mock.patch('madgraph.various.virtual_pool.train_pool', side_effect=self.fit):
            self.command.collect_virtual_pool([job], 1, True)
        for name in ('grid.MC_integer', 'mint_grids'):
            (Path(job['dirname']) / name).write_text('grid\n')
        split = dict(job, split=2)
        self.command.write_input_file = mock.Mock()
        self.command.prepare_directories([split], 'NLO', fixed_order=True)
        destination = Path(split['dirname'])
        self.assertEqual(destination.name, 'all_G1_2')
        self.assertEqual((destination / 'virtual_pool_mode.dat').read_text(), '2\n')
        self.assertTrue((destination / 'virtual_pool_model.dat').is_file())
        self.assertEqual((destination / 'mint_grids').read_text(), 'grid\n')

    def test_resume_requires_manifest_and_original_model(self):
        self.configure(resume=True)
        job = self.job()
        with self.assertRaisesRegex(aMCatNLOError, 'Cannot resume pooled virtuals'):
            self.command.stage_virtual_pool(job, 2, False)
        self.configure()
        with mock.patch('madgraph.various.virtual_pool.train_pool', side_effect=self.fit):
            self.command.collect_virtual_pool([job], 0, False)
        model = self.root / 'SubProcesses' / 'P0_gg' / 'virtual_pool_model.dat'
        model.write_text('changed since envelope construction\n')
        self.configure(resume=True)
        with self.assertRaisesRegex(aMCatNLOError, 'checksum'):
            self.command.stage_virtual_pool(job, 2, False)
        model.unlink()
        with self.assertRaisesRegex(aMCatNLOError, 'Cannot resume pooled virtuals'):
            self.command.stage_virtual_pool(job, 2, False)

    def test_empty_capture_has_explicit_persistent_fallback(self):
        self.configure()
        job = self.job()
        with mock.patch('madgraph.various.virtual_pool.train_pool',
                        return_value=dict(status='skipped', model_written=False)):
            self.command.collect_virtual_pool([job], 0, False)
        manifest = self.root / 'SubProcesses' / 'P0_gg' / 'virtual_pool_manifest.json'
        self.assertEqual(json.loads(manifest.read_text())['status'], 'fallback')
        self.configure(resume=True)
        self.command.stage_virtual_pool(job, 2, False)
        self.assertEqual((Path(job['dirname']) / 'virtual_pool_mode.dat').read_text(), '0\n')

    def test_zero_blend_model_has_fallback_status_and_frozen_checksum(self):
        self.configure()
        job = self.job()

        def fallback(*args, **kwargs):
            self.fit(*args, **kwargs)
            return dict(status='fallback', model_written=True)

        with mock.patch('madgraph.various.virtual_pool.train_pool', side_effect=fallback):
            self.command.collect_virtual_pool([job], 0, False)
        manifest = self.root / 'SubProcesses' / 'P0_gg' / 'virtual_pool_manifest.json'
        state = json.loads(manifest.read_text())
        self.assertEqual(state['status'], 'fallback')
        self.assertTrue(state['model_sha256'])
        self.configure(resume=True)
        self.command.stage_virtual_pool(job, 2, False)
        self.assertEqual((Path(job['dirname']) / 'virtual_pool_mode.dat').read_text(), '2\n')

    def test_missing_numpy_keeps_mint_through_barrier_and_resume(self):
        for fixed in (False, True):
            with self.subTest(fixed_order=fixed):
                mode = 'NLO' if fixed else 'aMC@NLO'
                stage = 1 if fixed else 0
                self.configure(mode=mode)
                job = self.job(fixed=fixed)
                directory = Path(job['dirname'])
                parent = directory.parent
                self.command.stage_virtual_pool(job, stage, fixed)
                header = ('VIRTUAL_POOL_SAMPLES 2\n5 8 1 4 2\n'
                          '21 21 6 -6\n0 0 173 173\n1 1 0 0\n')
                rows = ['%d 1 1 1 8 .5 1 2 1 0 0 0 0 0 0 0 0 .3 .7\n' % i
                        for i in range(1, 61)]
                (directory / 'virtual_pool_samples.dat').write_text(header + ''.join(rows))
                (directory / 'virtual_pool_iterations.dat').write_text('1 1 100\n')
                # An old file must not turn a missing-NumPy fallback into apply mode.
                (parent / 'virtual_pool_model.dat').write_text('stale\n')
                (directory / 'virtual_pool_model.dat').write_text('stale\n')
                with mock.patch.dict(sys.modules, {'numpy': None}):
                    self.command.collect_virtual_pool([job], stage, fixed)
                    self.command.stage_virtual_pool(job, stage + 1, fixed)
                    self.assertEqual((directory / 'virtual_pool_mode.dat').read_text(), '0\n')
                    self.assertFalse((directory / 'virtual_pool_model.dat').exists())
                    self.assertFalse((parent / 'virtual_pool_model.dat').exists())
                    state = json.loads((parent / 'virtual_pool_manifest.json').read_text())
                    self.assertEqual(state['status'], 'fallback')
                    self.assertIsNone(state['model_sha256'])
                    report = json.loads((parent / 'virtual_pool_report.json').read_text())
                    self.assertEqual(report['accepted_rows'], 60)
                    self.assertEqual(report['reason'], 'numpy_unavailable')
                    self.configure(mode=mode, resume=True)
                    split = self.job(split=2, fixed=fixed)
                    with mock.patch('madgraph.various.virtual_pool.train_pool') as fit:
                        self.command.collect_virtual_pool([split], stage + 1, fixed)
                        self.command.stage_virtual_pool(split, stage + 2, fixed)
                        fit.assert_not_called()
                    self.assertEqual((Path(split['dirname']) / 'virtual_pool_mode.dat').read_text(), '0\n')

    def test_fresh_run_discards_stale_fits_and_capture_files(self):
        parent = self.root / 'SubProcesses' / 'P0_gg'
        for name in ('virtual_pool_model.dat', 'virtual_pool_report.json', 'virtual_pool_manifest.json'):
            (parent / name).write_text('old run\n')
        self.configure()
        self.assertFalse((parent / 'virtual_pool_model.dat').exists())
        self.assertFalse((parent / 'virtual_pool_manifest.json').exists())
        job = self.job()
        for name in ('virtual_pool_model.dat', 'virtual_pool_samples.dat', 'virtual_pool_iterations.dat'):
            (Path(job['dirname']) / name).write_text('old run\n')
        self.command.stage_virtual_pool(job, 0, False, clear_records=True)
        for name in ('virtual_pool_model.dat', 'virtual_pool_samples.dat', 'virtual_pool_iterations.dat'):
            self.assertFalse((Path(job['dirname']) / name).exists())

    def test_grid_io_transfers_full_directory_including_model_and_control(self):
        self.configure()
        job = self.job(split=3)
        with mock.patch('madgraph.various.virtual_pool.train_pool', side_effect=self.fit):
            self.command.collect_virtual_pool([job], 0, False)
        self.command.stage_virtual_pool(job, 2, False)
        self.command.get_pdf_input_filename = lambda: str(self.root / 'missing_pdf')
        inputs, outputs, required, args = self.command.getIO_ajob(
            'ajob1', str(Path(job['dirname']).parent), ['1', 'F', '3', '2'])
        self.assertIn(job['dirname'], inputs)
        self.assertIn('GF1_3', outputs)
        self.assertIn('GF1_3/virtual_pool_mode.dat', required)
        self.assertIn('GF1_3/virtual_pool_model.dat', required)
        self.assertNotIn('GF1_3/virtual_pool_samples.dat', required)

    def test_disabled_does_not_change_job_files_or_import_fitter(self):
        (self.root / 'Cards' / 'FKS_params.dat').write_text('#UsePooledVirtual\nF\n')
        parent = self.root / 'SubProcesses' / 'P0_gg'
        (parent / 'virtual_pool_model.dat').write_text('existing\n')
        self.configure()
        job = self.job()
        with mock.patch('madgraph.various.virtual_pool.train_pool') as fit:
            self.command.stage_virtual_pool(job, 0, False, clear_records=True)
            self.command.collect_virtual_pool([job], 0, False)
            fit.assert_not_called()
        self.assertEqual(list(Path(job['dirname']).iterdir()), [])
        self.assertEqual((parent / 'virtual_pool_model.dat').read_text(), 'existing\n')

    def test_card_parser_matches_fortran_case_and_last_value(self):
        card = self.root / 'Cards' / 'FKS_params.dat'
        card.write_text('#UsePooledVirtual\nT\n#UsePooledVirtual\nF\n')
        self.configure()
        self.assertFalse(self.command._virtual_pool_enabled)
        card.write_text('#usepooledvirtual\nT\n')
        self.configure()
        self.assertFalse(self.command._virtual_pool_enabled)


if __name__ == '__main__':
    unittest.main()
