"""Exercise the rerun selection, physics cards and stale-result protection.

The export command is replaced with a small directory fixture; no integration
or shower is launched. Card preparation uses the real production inputs.
"""

import ast
import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]


def load_script(name):
    spec = importlib.util.spec_from_file_location(
        'single_top_' + name, ROOT / 'paper' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = load_script('run_g_variations')


def rows():
    with (ROOT / 'paper/proposed_runs.csv').open() as stream:
        return {row['run_id']: row for row in csv.DictReader(stream)}


def run_setting(path, name):
    match = re.search(r'^\s*([^!\n=]+?)\s*=\s*' + re.escape(name)
                      + r'\b', path.read_text(), re.MULTILINE)
    if match is None:
        raise AssertionError('Missing run-card setting: ' + name)
    return match[1].strip()


def fks_setting(path, name):
    match = re.search(r'^#' + re.escape(name) + r'\s*\n\s*(\S+)',
                      path.read_text(), re.MULTILINE)
    if match is None:
        raise AssertionError('Missing FKS setting: ' + name)
    return match[1]


class TestSingleTopCampaignPreparation(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory(prefix='mg5_tj_reruns_')
        self.addCleanup(self.work.cleanup)
        self.work_path = Path(self.work.name)

    def fake_export(self, command, **kwargs):
        self.assertEqual(Path(command[1]), ROOT / 'bin/mg5_aMC')
        commands = Path(command[2]).read_text()
        destination = Path(re.search(r'^output\s+(.+)$', commands,
                                     re.MULTILINE)[1])
        for name in ['Cards', 'SubProcesses', 'MCatNLO/PY8Analyzer',
                     'MCatNLO/include']:
            (destination / name).mkdir(parents=True, exist_ok=True)
        for name in RUNNER.MATCHING_SOURCES:
            target = destination / Path(name).relative_to('Template/NLO')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        return subprocess.CompletedProcess(command, 0)

    def prepare(self, run_id):
        row = rows()[run_id]
        directory = self.work_path / run_id
        with mock.patch.object(RUNNER, 'preflight_pythia'), \
                mock.patch.object(RUNNER.subprocess, 'run',
                                  side_effect=self.fake_export):
            RUNNER.prepare(row, directory, 64, io.StringIO())
        return row, directory

    def campaign(self, *arguments):
        with contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(RUNNER, 'preflight_pythia'):
            RUNNER.run_campaign({'G': 4, 'P': 2}, 'g_runs_fixed', 'test',
                                argv=list(arguments))

    def test_central_and_variation_cards_preserve_physics_and_shower(self):
        for run_id in ['C_GLB', 'C_LOC', 'C_Q', 'C_QT', 'G_LOC_H005',
                       'G_Q_H020', 'P_QT_COMMON_W', 'S_Q_Q2']:
            with self.subTest(run_id=run_id):
                row, directory = self.prepare(run_id)
                run = directory / 'Cards/run_card.dat'
                fks = directory / 'Cards/FKS_params.dat'
                shower = (directory / 'Cards/shower_card.dat').read_text()
                for name in ['nevents', 'iseed', 'reweight_scale',
                             'store_rwgt_info', 'shower_scale_factor']:
                    self.assertEqual(run_setting(run, name), row[name])
                self.assertEqual(run_setting(run, 'parton_shower'), 'PYTHIA8')
                self.assertEqual(run_setting(run, 'ickkw'), '0')
                self.assertEqual(run_setting(run, 'MCatNLO_DELTA'), 'False')
                self.assertEqual(run_setting(run, 'pdlabel').strip("'\""),
                                 'nn23nlo')
                self.assertEqual(float(run_setting(run, 'ebeam1')), 6500.)
                self.assertEqual(float(run_setting(run, 'ebeam2')), 6500.)
                self.assertEqual(fks_setting(fks, 'FKSPythiaLocalRecoil'),
                                 row['FKSPythiaLocalRecoil'])
                self.assertEqual(fks_setting(fks, 'MCExplicitKLSum'), '.true.')
                self.assertEqual(fks_setting(fks, 'MCSubtractionAtFixedFlow'),
                                 '.true.')
                self.assertEqual(fks_setting(fks, 'FKSExplicitSum'),
                                 '.true.' if row['FKSPythiaLocalRecoil'] == '1'
                                 else '.false.')
                for name in ['FKSISRMapping', 'FKSFinalRecoil']:
                    self.assertEqual(fks_setting(fks, name), '0')
                for name in ['PythiaLocalCollinearPatch', 'PythiaLocalSoftPatch']:
                    self.assertEqual(fks_setting(fks, name),
                                     '.' + row[name].lower() + '.')
                self.assertEqual(fks_setting(fks, 'PythiaLocalGProfile'),
                                 '1' if row['profile'] == 'w_forced' else '0')
                if row['PythiaLocalGWidth']:
                    self.assertEqual(float(fks_setting(fks, 'PythiaLocalGWidth')
                                           .lower().replace('d', 'e')),
                                     float(row['PythiaLocalGWidth']))
                options = ast.literal_eval(re.search(
                    r'^pythia8_options\s*=\s*(\{.*\})$', shower,
                    re.MULTILINE)[1])
                for name in ['PartonLevel:ISR', 'PartonLevel:FSR']:
                    self.assertEqual(options[name], 'on')
                for name in ['6:mayDecay', 'PartonLevel:MPI',
                             'PartonLevel:Remnants', 'HadronLevel:all',
                             'ColourReconnection:reconnect',
                             'BeamRemnants:primordialKT',
                             'TimeShower:QEDshowerByQ',
                             'SpaceShower:QEDshowerByQ']:
                    self.assertEqual(options[name], 'off')
                local = row['FKSPythiaLocalRecoil'] == '1'
                self.assertEqual(options['SpaceShower:dipoleRecoil'],
                                 'on' if local else 'off')
                self.assertEqual(options['TimeShower:globalRecoil'],
                                 'off' if local else 'on')
                self.assertRegex(shower, r'(?m)^rnd_seed\s*=\s*'
                                 + row['rnd_seed'] + r'\s*$')
                self.assertRegex(shower, r'(?m)^nevents\s*=\s*-1\s*$')
                self.assertRegex(shower, r'(?m)^nsplit_jobs\s*=\s*64\s*$')
                self.assertRegex(shower, r'(?m)^combine_td\s*=\s*T\s*$')
                config = (directory / 'Cards/amcatnlo_configuration.txt').read_text()
                prefix = re.search(r'^pythia8_path\s*=\s*(\S+)',
                                   config, re.MULTILINE)[1]
                self.assertIn('xmlPath = ' + prefix + '/share/Pythia8/xmldoc/',
                              shower)
                self.assertRegex(config, r'(?m)^nb_core\s*=\s*64$')
                self.assertEqual(json.loads((directory / 'run_spec.json').read_text()), row)
                self.assertEqual((directory / 'Cards/param_card.dat').read_bytes(),
                                 (ROOT / row['source_process'] / 'Cards/param_card.dat').read_bytes())
                if row['reweight_scale'] == 'True':
                    for name in ['rw_rscale', 'rw_fscale']:
                        self.assertEqual([float(v) for v in run_setting(run, name).split(',')],
                                         [float(v) for v in row[name].split(';')])

    def test_missing_fix_is_rejected_before_reusing_export(self):
        source = self.work_path / 'cluster.f'
        source.write_text('      p_inter(:,1,0)=pcl(:,1)\n')
        with self.assertRaisesRegex(RuntimeError, 'normalization fix'):
            RUNNER.check_fixed_clustering(source)
        source.write_text('      p_inter(:,0,0)=pcl(:,3)\n')
        RUNNER.check_fixed_clustering(source)

    def test_shower_workers_divide_sample_without_exceeding_core_budget(self):
        for nevents, cores, expected in [(1000000, 64, 64),
                                         (1000000, 3, 2),
                                         (1000000, 1, 1),
                                         (12, 7, 6),
                                         (16, 64, 16),
                                         (101, 128, 1)]:
            with self.subTest(nevents=nevents, cores=cores):
                jobs = RUNNER.shower_jobs(nevents, cores)
                self.assertEqual(jobs, expected)
                self.assertEqual(nevents % jobs, 0)
                self.assertLessEqual(jobs, min(cores, 99))

    def test_generation_provenance_covers_fix_and_exact_run_settings(self):
        row = rows()['G_LOC_H005']
        inputs = RUNNER.generation_inputs(row)
        self.assertEqual(inputs['schema'], 3)
        self.assertIn('Template/NLO/SubProcesses/cluster.f', inputs['source_sha256'])
        self.assertIn('numerical_fix', inputs)
        for name in ['genps_fks_helpers.f', 'fks_Sij.f', 'initial_recoil.f',
                     'genps_fks_radiation.f']:
            self.assertIn('Template/NLO/SubProcesses/' + name,
                          inputs['source_sha256'])
        self.assertEqual(inputs['run_spec'], row)
        changed = dict(row, rnd_seed=str(int(row['rnd_seed']) + 1))
        self.assertNotEqual(inputs, RUNNER.generation_inputs(changed))

    def test_legacy_done_file_cannot_skip_biased_sample(self):
        row = rows()['G_LOC_H005']
        events = self.work_path / row['run_id'] / 'Events' / row['run_id']
        events.mkdir(parents=True)
        (events / 'campaign_done.json').write_text(json.dumps(row))
        with self.assertRaisesRegex(SystemExit, 'Unverified|changed inputs'):
            self.campaign('--output-dir', str(self.work_path), '--only', row['run_id'])

    def test_preparation_can_resume_but_modified_cards_cannot(self):
        run_id = 'G_LOC_H005'
        argv = ['--output-dir', str(self.work_path), '--only', run_id,
                '--prepare-only', '--cores', '64']
        with mock.patch.object(RUNNER.subprocess, 'run',
                               side_effect=self.fake_export) as export:
            self.campaign(*argv)
            self.assertEqual(export.call_count, 1)
            self.campaign(*argv[:-1], '8')
            self.assertEqual(export.call_count, 1)
            shower = (self.work_path / run_id / 'Cards/shower_card.dat').read_text()
            self.assertRegex(shower, r'(?m)^nsplit_jobs\s*=\s*8\s*$')
            self.campaign(*argv[:-1], '8')
            self.assertEqual(export.call_count, 1)
            card = self.work_path / run_id / 'Cards/FKS_params.dat'
            RUNNER.set_fks_card(card, 'PythiaLocalGWidth', '0.02d0')
            with self.assertRaisesRegex(SystemExit, 'Prepared inputs changed'):
                self.campaign(*argv)

    def test_changed_shower_library_invalidates_prepared_export(self):
        run_id = 'G_Q_H020'
        argv = ['--output-dir', str(self.work_path), '--only', run_id,
                '--prepare-only']
        with mock.patch.object(RUNNER.subprocess, 'run',
                               side_effect=self.fake_export):
            self.campaign(*argv)
        config = self.work_path / run_id / 'Cards/amcatnlo_configuration.txt'
        config.write_text(re.sub(r'^pythia8_path\s*=.*$',
                                 'pythia8_path = /different/library',
                                 config.read_text(), flags=re.MULTILINE))
        with self.assertRaisesRegex(SystemExit, 'Prepared inputs changed'):
            self.campaign(*argv)

    def test_changed_geometry_source_invalidates_prepared_export(self):
        run_id = 'G_LOC_H005'
        argv = ['--output-dir', str(self.work_path), '--only', run_id,
                '--prepare-only']
        with mock.patch.object(RUNNER.subprocess, 'run',
                               side_effect=self.fake_export):
            self.campaign(*argv)
        helper = self.work_path / run_id / 'SubProcesses/genps_fks_helpers.f'
        with helper.open('a') as stream:
            stream.write('\nc changed geometry\n')
        with self.assertRaisesRegex(SystemExit, 'Prepared inputs changed'):
            self.campaign(*argv)

    def interrupted_shower_fixture(self):
        run_id = 'G_LOC_H005'
        argv = ['--output-dir', str(self.work_path), '--only', run_id,
                '--cores', '64']
        with mock.patch.object(RUNNER.subprocess, 'run', side_effect=self.fake_export):
            self.campaign(*argv, '--prepare-only')
        row = rows()[run_id]
        directory = self.work_path / run_id
        events = directory / 'Events' / run_id
        events.mkdir(parents=True)
        lhe = events / 'events.lhe'
        lhe.write_text('verified completed generation\n')
        summary = events / 'summary.txt'
        summary.write_text('Number of events generated: ' + row['nevents'] + '\n')
        marker = events / 'campaign_shower_pending.json'
        marker.write_text(json.dumps({
            'schema': RUNNER.SCHEMA, 'inputs': RUNNER.generation_inputs(row),
            'output_sha256': {path.name: RUNNER.sha256(path)
                              for path in (lhe, summary)}}))
        log = directory / RUNNER.SHOWER_LOG
        log.parent.mkdir(parents=True)
        log.write_text('interrupted serial shower log\n')
        (log.parent / 'events.lhe').symlink_to(
            Path('../../Events') / run_id / 'events.lhe')
        plot = events / 'plot_PYTHIA8_1_0.HwU'
        plot.write_text('interrupted shower histogram\n')
        return row, directory, events, argv, marker, log, plot

    def test_verified_interrupted_shower_resumes_without_regenerating_events(self):
        row, directory, events, argv, _, log, plot = self.interrupted_shower_fixture()
        original_lhe = (events / 'events.lhe').read_bytes()
        with mock.patch.object(RUNNER.subprocess, 'run') as execute, \
                mock.patch.object(RUNNER, 'check_output', return_value=events), \
                mock.patch.object(RUNNER, 'completion_record', return_value={'completed': True}):
            self.campaign(*argv)
        execute.assert_called_once()
        self.assertEqual(execute.call_args.args[0],
                         [str(directory / 'bin/shower'), row['run_id'], '-f'])
        self.assertEqual((events / 'events.lhe').read_bytes(), original_lhe)
        self.assertEqual(json.loads((events / 'campaign_done.json').read_text()),
                         {'completed': True})
        self.assertFalse(log.exists())
        self.assertFalse(plot.exists())

    def test_changed_lhe_cannot_resume_trusted_interrupted_shower(self):
        _, directory, events, argv, _, _, _ = self.interrupted_shower_fixture()
        (events / 'events.lhe').write_text('changed generation\n')
        with mock.patch.object(RUNNER.subprocess, 'run') as execute:
            with self.assertRaises((RuntimeError, SystemExit)):
                self.campaign(*argv)
        execute.assert_not_called()
        self.assertFalse(list((directory / 'MCatNLO').glob('INTERRUPTED_RUN_*')))

    def test_unverified_incomplete_run_cannot_resume(self):
        _, _, _, argv, marker, _, _ = self.interrupted_shower_fixture()
        marker.unlink()
        with mock.patch.object(RUNNER.subprocess, 'run') as execute:
            with self.assertRaisesRegex(SystemExit, 'Incomplete run'):
                self.campaign(*argv)
        execute.assert_not_called()

    def test_interrupted_shower_marker_must_match_current_physics_inputs(self):
        _, _, _, argv, marker, _, _ = self.interrupted_shower_fixture()
        stored = json.loads(marker.read_text())
        stored['inputs']['normalization_fix'] = 'obsolete-matching-source'
        marker.write_text(json.dumps(stored))
        with mock.patch.object(RUNNER.subprocess, 'run') as execute:
            with self.assertRaises((RuntimeError, SystemExit)):
                self.campaign(*argv)
        execute.assert_not_called()

    def interrupted_generation_fixture(self, partial_events=False):
        run_id = 'G_LOC_H005'
        argv = ['--output-dir', str(self.work_path), '--only', run_id,
                '--cores', '64']
        with mock.patch.object(RUNNER.subprocess, 'run', side_effect=self.fake_export):
            self.campaign(*argv, '--prepare-only')
        row = rows()[run_id]
        directory = self.work_path / run_id
        processes = ['P0_bu_td', 'P0_ub_td', 'P0_bdx_tux', 'P0_dxb_tux']
        jobs, retained = [], {}
        for process in processes:
            grid = directory / 'SubProcesses' / process / 'GF1.0'
            grid.mkdir(parents=True)
            for name in ['mint_grids', 'grid.MC_integer', 'res_1.dat', 'res_1']:
                output = grid / name
                output.write_text('verified upper-envelope input: ' + name + '\n')
                retained[str(output.relative_to(directory))] = RUNNER.sha256(output)
            jobs.append({'p_dir': process, 'split': 0, 'mint_mode': 2,
                         'channel': '1.0', 'run_mode': 'all',
                         'dirname': str(grid), 'nevents': int(row['nevents']) // 4})
        status = directory / 'SubProcesses/job_status.pkl'
        status.write_bytes(pickle.dumps(jobs))
        retained[str(status.relative_to(directory))] = RUNNER.sha256(status)
        marker = directory / 'campaign_generation_pending.json'
        marker.write_text(json.dumps({
            'schema': RUNNER.SCHEMA, 'inputs': RUNNER.generation_inputs(row),
            'retained_sha256': retained}))
        events = directory / 'Events' / run_id
        chunk = directory / 'SubProcesses' / processes[0] / 'GF1.0_1'
        if partial_events:
            events.mkdir(parents=True)
            (events / 'alllogs_1.html').write_text('completed integration logs\n')
            chunk.mkdir()
            (chunk / 'events.lhe').write_text('truncated generation\n')
            (chunk / 'mint_grids').symlink_to('../GF1.0/mint_grids')
        return row, directory, events, argv, marker, chunk

    def test_verified_generation_resumes_from_upper_grids_without_event_directory(self):
        row, directory, events, argv, marker, _ = self.interrupted_generation_fixture()

        def generate(command, **kwargs):
            self.assertEqual(command, [str(directory / 'bin/generate_events'),
                                      'aMC@NLO', '-f', '--name=' + row['run_id'],
                                      '--only_generation'])
            self.assertEqual(kwargs['cwd'], directory)
            events.mkdir(parents=True)

        retained = json.loads(marker.read_text())['retained_sha256']
        with mock.patch.object(RUNNER.subprocess, 'run', side_effect=generate) as execute, \
                mock.patch.object(RUNNER, 'check_output', return_value=events), \
                mock.patch.object(RUNNER, 'completion_record', return_value={'completed': True}):
            self.campaign(*argv)
        execute.assert_called_once()
        self.assertEqual(json.loads((events / 'campaign_done.json').read_text()),
                         {'completed': True})
        self.assertTrue(marker.is_file())
        for name, digest in retained.items():
            self.assertEqual(RUNNER.sha256(directory / name), digest)

    def test_generation_resume_preserves_partial_events_and_grid_links(self):
        row, directory, events, argv, _, chunk = self.interrupted_generation_fixture(True)

        def generate(command, **kwargs):
            self.assertIn('--only_generation', command)
            self.assertFalse(events.exists())
            self.assertFalse(chunk.exists())
            archived_chunk = chunk.with_name('INTERRUPTED_GF1.0_1_attempt_1')
            self.assertEqual((archived_chunk / 'events.lhe').read_text(),
                             'truncated generation\n')
            self.assertEqual((archived_chunk / 'mint_grids').resolve(),
                             chunk.parent / 'GF1.0/mint_grids')
            archived_events = events.with_name('INTERRUPTED_' + row['run_id'] + '_attempt_1')
            self.assertEqual((archived_events / 'alllogs_1.html').read_text(),
                             'completed integration logs\n')
            events.mkdir()

        with mock.patch.object(RUNNER.subprocess, 'run', side_effect=generate) as execute, \
                mock.patch.object(RUNNER, 'check_output', return_value=events), \
                mock.patch.object(RUNNER, 'completion_record', return_value={'completed': True}):
            self.campaign(*argv)
        execute.assert_called_once()

    def test_changed_retained_grid_prevents_generation_retry_and_archival(self):
        _, directory, events, argv, _, chunk = self.interrupted_generation_fixture(True)
        grid = chunk.parent / 'GF1.0/mint_grids'
        grid.write_text('changed integration grid\n')
        with mock.patch.object(RUNNER.subprocess, 'run') as execute:
            with self.assertRaisesRegex(RuntimeError, 'Changed retained generation input'):
                self.campaign(*argv)
        execute.assert_not_called()
        self.assertTrue(events.is_dir())
        self.assertTrue(chunk.is_dir())

    def test_pending_generation_rejects_changed_inputs_or_missing_grid_record(self):
        row, directory, _, _, marker, _ = self.interrupted_generation_fixture()
        original = json.loads(marker.read_text())
        for corruption in ['inputs', 'missing_grid']:
            with self.subTest(corruption=corruption):
                record = json.loads(json.dumps(original))
                if corruption == 'inputs':
                    record['inputs']['numerical_fix'] = 'unverified source'
                else:
                    name = next(name for name in record['retained_sha256']
                                if name.endswith('/mint_grids'))
                    del record['retained_sha256'][name]
                marker.write_text(json.dumps(record))
                with self.assertRaisesRegex(RuntimeError, 'Unverified|Wrong retained'):
                    RUNNER.pending_generation(row, directory, RUNNER.generation_inputs(row))

    def test_pending_generation_rejects_wrong_job_event_count(self):
        row, directory, _, _, marker, _ = self.interrupted_generation_fixture()
        record = json.loads(marker.read_text())
        status = directory / 'SubProcesses/job_status.pkl'
        jobs = pickle.loads(status.read_bytes())
        jobs[0]['nevents'] -= 1
        status.write_bytes(pickle.dumps(jobs))
        record['retained_sha256']['SubProcesses/job_status.pkl'] = RUNNER.sha256(status)
        marker.write_text(json.dumps(record))
        with self.assertRaisesRegex(RuntimeError, 'Wrong retained generation count'):
            RUNNER.pending_generation(row, directory, RUNNER.generation_inputs(row))

    def test_prepare_only_preserves_pending_generation_outputs(self):
        _, directory, events, argv, _, chunk = self.interrupted_generation_fixture(True)
        with mock.patch.object(RUNNER.subprocess, 'run') as execute:
            self.campaign(*argv, '--prepare-only')
        execute.assert_not_called()
        self.assertTrue(events.is_dir())
        self.assertTrue(chunk.is_dir())
        self.assertFalse(list(directory.glob('SubProcesses/P*/INTERRUPTED_*')))

    def test_prepare_only_preserves_interrupted_shower_files(self):
        _, directory, _, argv, _, log, plot = self.interrupted_shower_fixture()
        originals = {path: path.read_bytes() for path in (log, plot)}
        with mock.patch.object(RUNNER.subprocess, 'run') as execute:
            self.campaign(*argv, '--prepare-only')
        execute.assert_not_called()
        self.assertFalse(list((directory / 'MCatNLO').glob('INTERRUPTED_RUN_*')))
        self.assertFalse((directory / 'Events' / 'G_LOC_H005' / 'interrupted_showers').exists())
        for path, original in originals.items():
            self.assertEqual(path.read_bytes(), original)

    def test_archived_interrupted_shower_keeps_relative_paths_and_generation(self):
        row, directory, events, _, _, log, plot = self.interrupted_shower_fixture()
        original_log, original_plot = log.read_bytes(), plot.read_bytes()
        RUNNER.archive_showers(row, directory)
        archived_runs = list((directory / 'MCatNLO').glob('INTERRUPTED_RUN_PYTHIA8_1_attempt_*'))
        archived_plots = list((events / 'interrupted_showers').glob('attempt_*'))
        self.assertEqual(len(archived_runs), 1)
        self.assertEqual(len(archived_plots), 1)
        self.assertEqual((archived_runs[0] / log.name).read_bytes(), original_log)
        self.assertEqual((archived_plots[0] / plot.name).read_bytes(), original_plot)
        self.assertFalse(log.exists())
        self.assertFalse(plot.exists())
        archived_lhe_link = archived_runs[0] / 'events.lhe'
        self.assertTrue(archived_lhe_link.is_symlink())
        self.assertEqual(archived_lhe_link.resolve(), events / 'events.lhe')
        self.assertTrue((events / 'events.lhe').is_file())
        self.assertTrue((events / 'summary.txt').is_file())

    def test_truncated_generation_summary_is_not_marked_complete(self):
        row, directory = self.prepare('G_LOC_H005')
        events = directory / 'Events' / row['run_id']
        events.mkdir(parents=True)
        for name in RUNNER.OUTPUT_NAMES:
            (events / name).write_text('not a completed sample\n')
        (events / 'summary.txt').write_text('Number of events generated: 1000\n')
        with self.assertRaisesRegex(RuntimeError, 'event|Event'):
            RUNNER.check_output(row, directory)

    def test_invalidated_output_is_rejected_even_when_files_exist(self):
        row, directory = self.prepare('G_LOC_H005')
        (directory / 'NORMALIZATION_FIX.json').write_text(json.dumps(
            {'invalid_runs': [row['run_id']]}))
        with self.assertRaisesRegex(RuntimeError, 'Superseded'):
            RUNNER.check_output(row, directory)

    def completed_fixture(self, run_id):
        row, directory = self.prepare(run_id)
        events = directory / 'Events' / row['run_id']
        events.mkdir(parents=True)
        for name in RUNNER.OUTPUT_NAMES:
            (events / name).write_text('fixture\n')
        (events / 'summary.txt').write_text(
            'Number of events generated: ' + row['nevents'] + '\n')
        self.write_shower_logs(row, directory)
        weights = {'central': 1., 'stat_error': .01}
        if row['reweight_scale'] == 'True':
            for r in map(float, row['rw_rscale'].split(';')):
                for f in map(float, row['rw_fscale'].split(';')):
                    weights['scale', r, f] = 1.
        histograms = [SimpleNamespace(
            title='observable' + str(i % 25),
            type=['ALL', 'TAG', 'GAP'][i // 25],
            bins=[SimpleNamespace(wgts=dict(weights))]) for i in range(75)]
        histograms.append(SimpleNamespace(
            title='auxiliary diagnostic', type='AUX', bins=[]))
        return row, directory, histograms

    def write_shower_logs(self, row, directory):
        shower = (directory / 'Cards/shower_card.dat').read_text()
        jobs = int(re.search(r'^nsplit_jobs\s*=\s*(\d+)', shower,
                             re.MULTILINE)[1])
        run = (directory / RUNNER.SHOWER_LOG).parent
        run.mkdir(parents=True, exist_ok=True)
        paths = ([run / 'mcatnlo_run.log'] if jobs == 1 else
                 [run / ('mcatnlo_run_' + str(i) + '.log')
                  for i in range(1, jobs + 1)])
        count = str(int(row['nevents']) // jobs)
        for path in paths:
            path.write_text('| sum | ' + ' '.join([count] * 3) + ' |\n')
        return paths

    def test_shower_must_accept_entire_sample_and_auxiliary_histograms_are_allowed(self):
        row, directory, histograms = self.completed_fixture('G_LOC_H005')
        with mock.patch('madgraph.various.histograms.HwUList', return_value=histograms):
            RUNNER.check_output(row, directory)
        split_log = (directory / RUNNER.SHOWER_LOG).with_name('mcatnlo_run_64.log')
        split_log.write_text('| sum | 15625 15625 15624 |\n')
        with self.assertRaisesRegex(RuntimeError, 'accept all'):
            RUNNER.check_output(row, directory)

    def test_parallel_shower_requires_every_worker_log(self):
        row, directory, histograms = self.completed_fixture('G_LOC_H005')
        (directory / RUNNER.SHOWER_LOG).with_name('mcatnlo_run_64.log').unlink()
        with self.assertRaisesRegex(RuntimeError, 'Missing.*shower log'):
            RUNNER.check_output(row, directory)

    def test_wrong_worker_event_counts_cannot_cancel_between_splits(self):
        row, directory, _ = self.completed_fixture('G_LOC_H005')
        run = (directory / RUNNER.SHOWER_LOG).parent
        (run / 'mcatnlo_run_1.log').write_text('| sum | 15625 15625 15624 |\n')
        (run / 'mcatnlo_run_2.log').write_text('| sum | 15626 15626 15626 |\n')
        with self.assertRaisesRegex(RuntimeError, 'accept all'):
            RUNNER.check_output(row, directory)

    def test_serial_shower_completion_is_still_supported(self):
        row, directory, histograms = self.completed_fixture('G_LOC_H005')
        card = directory / 'Cards/shower_card.dat'
        RUNNER.replace(card, r'^(nsplit_jobs\s*=\s*)[^!\n]*(.*)$', 1)
        for path in (directory / RUNNER.SHOWER_LOG).parent.glob('mcatnlo_run_*.log'):
            path.unlink()
        self.write_shower_logs(row, directory)
        with mock.patch('madgraph.various.histograms.HwUList', return_value=histograms):
            RUNNER.check_output(row, directory)

    def test_completion_fingerprint_covers_every_parallel_worker_log(self):
        row, directory, _ = self.completed_fixture('G_LOC_H005')
        before = RUNNER.completion_record(row, directory, RUNNER.generation_inputs(row))
        logs = {name for name in before['output_sha256']
                if re.search(r'/mcatnlo_run_\d+\.log$', name)}
        self.assertEqual(len(logs), 64)
        log = (directory / RUNNER.SHOWER_LOG).with_name('mcatnlo_run_64.log')
        with log.open('a') as stream:
            stream.write('changed worker output\n')
        after = RUNNER.completion_record(row, directory, RUNNER.generation_inputs(row))
        self.assertNotEqual(before, after)

    def compatible_global_source_fixture(self):
        row, directory, _ = self.completed_fixture('C_GLB')
        source = directory / 'SubProcesses/resonance_recoil.f'
        text = source.read_text().replace(
            'c Invariants here may be arbitrarily small near a massless FI emitter.\n'
            'c The legacy external dot rounds products below 1d-6 GeV^2 to zero;\n'
            'c use the unrounded product for these dimensionless scale ratios.\n'
            '      use fks_phase_space_helpers, only: dot\n', '')
        text = text.replace(
            '     $     s,mass2,kq,kt,jq,jt,diff(0:3),delta\n'
            '      logical pass\n',
            '     $     s,mass2,kq,kt,jq,jt,diff(0:3),delta,dot\n'
            '      logical pass\n'
            '      external dot\n')
        source.write_text(text)
        before = 'cfb85cf75959fee528c89c13a349bbd1ad97e47bbda7e188b0800951f4df05b8'
        self.assertEqual(RUNNER.sha256(source), before)
        inputs = RUNNER.generation_inputs(row)
        original = json.loads(json.dumps(inputs))
        original['source_sha256']['Template/NLO/SubProcesses/resonance_recoil.f'] = before
        marker = directory / 'campaign_source_migration.json'
        marker.write_text(json.dumps({'schema': RUNNER.SCHEMA, 'run_id': 'C_GLB',
                                     'original_generation_inputs': original}))
        return row, directory, inputs, original, marker

    def test_global_source_migration_preserves_actual_generation_provenance(self):
        row, directory, inputs, original, marker = self.compatible_global_source_fixture()
        actual_cards = RUNNER.prepared_files(directory)
        record = RUNNER.completion_record(row, directory, inputs)
        self.assertEqual(record['inputs'], inputs)
        self.assertEqual(record['generation_inputs'], original)
        self.assertEqual(record['source_migration_sha256'], RUNNER.sha256(marker))
        self.assertEqual(record['cards_sha256'], actual_cards)
        self.assertNotEqual(record['generation_inputs']['source_sha256'],
                            record['inputs']['source_sha256'])
        with marker.open('a') as stream:
            stream.write('\n')
        changed = RUNNER.completion_record(row, directory, inputs)
        self.assertNotEqual(record, changed)

    def test_source_migration_rejects_unrelated_source_change_and_wrong_sample(self):
        row, directory, inputs, _, marker = self.compatible_global_source_fixture()
        stored = json.loads(marker.read_text())
        changed = json.loads(json.dumps(stored))
        changed['original_generation_inputs']['source_sha256'][
            'Template/NLO/SubProcesses/cluster.f'] = 'unverified older clustering'
        marker.write_text(json.dumps(changed))
        with self.assertRaisesRegex(RuntimeError, 'Unverified compatible source migration'):
            RUNNER.source_migration_provenance(row, directory, inputs)
        marker.write_text(json.dumps(stored))
        for changed_row in [dict(row, run_id='C_LOC'),
                            dict(row, FKSPythiaLocalRecoil='1')]:
            with self.assertRaisesRegex(RuntimeError, 'Unverified compatible source migration'):
                RUNNER.source_migration_provenance(changed_row, directory, inputs)
        RUNNER.set_fks_card(directory / 'Cards/FKS_params.dat', 'FKSFinalRecoil', '1')
        with self.assertRaisesRegex(RuntimeError, 'Unverified compatible source migration'):
            RUNNER.source_migration_provenance(row, directory, inputs)

    def test_source_migration_requires_original_export_source(self):
        row, directory, inputs, _, _ = self.compatible_global_source_fixture()
        shutil.copy2(ROOT / 'Template/NLO/SubProcesses/resonance_recoil.f',
                     directory / 'SubProcesses/resonance_recoil.f')
        with self.assertRaisesRegex(RuntimeError, 'Unverified compatible source migration'):
            RUNNER.source_migration_provenance(row, directory, inputs)

    def test_analysis_requires_matching_source_migration_record(self):
        from Analyses.matched_run_inputs import validate_completion
        row, directory, inputs, original, marker = self.compatible_global_source_fixture()
        manifest = {'inputs': inputs, 'prepared_sha256': RUNNER.prepared_files(directory)}
        (directory / 'campaign_manifest.json').write_text(json.dumps(manifest))
        done = directory / 'Events/C_GLB/campaign_done.json'
        done.write_text(json.dumps(RUNNER.completion_record(row, directory, inputs)))
        self.assertEqual(validate_completion(directory, row['run_id'])['generation_inputs'], original)
        marker.unlink()
        with self.assertRaisesRegex(ValueError, 'Changed or unverified'):
            validate_completion(directory, row['run_id'])

    def test_analysis_report_attributes_global_sample_to_actual_generation_source(self):
        analysis = load_script('analyse_matched_runs')
        row, directory, inputs, original, _ = self.compatible_global_source_fixture()
        events = directory / 'Events/C_GLB'
        (events / 'summary.txt').write_text(
            'Number of events generated: ' + row['nevents'] + '\n'
            'Total cross section: 132.0 +- 0.3 pb\n')
        (events / 'campaign_done.json').write_text(json.dumps(
            RUNNER.completion_record(row, directory, inputs)))
        sample = SimpleNamespace(path=events / 'plot_PYTHIA8_1_0.HwU',
                                 events=events, directory=directory,
                                 metrics=lambda: {'ALL': {'value': 132., 'error': .3}})
        report = analysis.report({'default': sample}, False)['samples']['default']
        self.assertEqual(report['matching_source_sha256'], original['source_sha256'])
        self.assertEqual(report['compatible_current_source_sha256'], inputs['source_sha256'])
        self.assertIn('source_migration_sha256', report)

    def test_nonfinite_bins_and_missing_scale_weights_are_rejected(self):
        row, directory, histograms = self.completed_fixture('S_Q_Q2')
        with mock.patch('madgraph.various.histograms.HwUList', return_value=histograms):
            RUNNER.check_output(row, directory)
            histograms[0].bins[0].wgts['central'] = float('nan')
            with self.assertRaisesRegex(RuntimeError, 'Non-finite'):
                RUNNER.check_output(row, directory)
            histograms[0].bins[0].wgts['central'] = 1.
            del histograms[0].bins[0].wgts['scale', 2., .5]
            with self.assertRaisesRegex(RuntimeError, 'Missing hard-scale'):
                RUNNER.check_output(row, directory)

    def test_pythia_preflight_rejects_missing_patches_then_accepts_paired_build(self):
        row = rows()['C_QT']
        prefix = self.work_path / 'pythia8'
        config = self.work_path / row['source_process'] / 'Cards/amcatnlo_configuration.txt'
        config.parent.mkdir(parents=True)
        config.write_text('pythia8_path = ' + str(prefix) + '\n')
        tool = prefix / 'bin/pythia8-config'
        tool.parent.mkdir(parents=True)
        tool.touch()
        library = prefix / 'lib/libpythia8.so'
        library.parent.mkdir(parents=True)
        library.touch()
        xml = prefix / 'share/Pythia8/xmldoc'
        xml.mkdir(parents=True)
        settings = xml / 'SpacelikeShowers.xml'
        settings.write_text('<settings/>\n')
        result = subprocess.CompletedProcess([str(tool), '--xmldoc'], 0,
                                             stdout=str(xml) + '\n')
        with mock.patch.object(RUNNER, 'ROOT', self.work_path), \
                mock.patch.object(RUNNER.subprocess, 'run', return_value=result):
            with self.assertRaisesRegex(RuntimeError, 'IFquarkCollinearCorrection'):
                RUNNER.preflight_pythia(row)
            settings.write_text('<flag name="SpaceShower:IFquarkCollinearCorrection"/>\n')
            with self.assertRaisesRegex(RuntimeError, 'IFtopSoftCorrection'):
                RUNNER.preflight_pythia(row)
            with settings.open('a') as stream:
                stream.write('<flag name="SpaceShower:IFtopSoftCorrection"/>\n')
            self.assertEqual(RUNNER.preflight_pythia(row), prefix)


class TestSingleTopCampaignOrchestration(unittest.TestCase):
    def test_dry_run_lists_exactly_22_affected_runs_and_writes_nothing(self):
        with tempfile.TemporaryDirectory(prefix='mg5_tj_dry_') as work:
            output = Path(work) / 'campaign'
            result = subprocess.run(
                [sys.executable, str(ROOT / 'paper/run_all.py'),
                 '--dry-run', '--output-root', str(output)], cwd=ROOT,
                capture_output=True, text=True, check=True, timeout=30)
            identifiers = re.findall(r'^(\w+): pp_tj_', result.stdout, re.MULTILINE)
            expected = {name for name, row in rows().items()
                        if row['group'] in {'C', 'G', 'P', 'S'}}
            self.assertEqual(len(expected), 22)
            self.assertEqual(set(identifiers), expected)
            self.assertEqual(len(identifiers), 22)
            self.assertFalse(output.exists())

    def test_master_orders_campaigns_before_complete_analysis_and_uses_64_cores(self):
        master = load_script('run_all')
        with tempfile.TemporaryDirectory(prefix='mg5_tj_master_') as work:
            with mock.patch.object(master.subprocess, 'run') as execute:
                with contextlib.redirect_stdout(io.StringIO()):
                    master.main(['--output-root', work])
            commands = [call.args[0] for call in execute.call_args_list]
        self.assertEqual([Path(command[1]).name for command in commands[:3]],
                         ['run_central.py', 'run_g_variations.py', 'run_shower_scales.py'])
        for command, output in zip(commands[:3],
                                   ['central_runs_fixed', 'g_runs_fixed', 'scale_runs_fixed']):
            self.assertEqual(command[command.index('--cores') + 1], '64')
            self.assertEqual(Path(command[command.index('--output-dir') + 1]).name,
                             output)
        analysis = commands[3]
        self.assertEqual(Path(analysis[1]).name, 'analyse_matched_runs.py')
        self.assertIn('--require-complete', analysis)

    def test_prepare_only_never_starts_postprocessing(self):
        master = load_script('run_all')
        with tempfile.TemporaryDirectory(prefix='mg5_tj_master_') as work:
            with mock.patch.object(master.subprocess, 'run') as execute:
                with contextlib.redirect_stdout(io.StringIO()):
                    master.main(['--prepare-only', '--output-root', work, '--cores', '8'])
        self.assertEqual(execute.call_count, 3)
        for call in execute.call_args_list:
            command = call.args[0]
            self.assertIn('--prepare-only', command)
            self.assertEqual(command[command.index('--cores') + 1], '8')


if __name__ == '__main__':
    unittest.main()
