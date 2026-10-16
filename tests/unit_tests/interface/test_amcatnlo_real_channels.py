"""Fixed-order real-channel scheduling, input and restart compatibility."""

import copy
import io
import os
import pickle
import tempfile
import unittest
from unittest import mock

from madgraph.interface import amcatnlo_run_interface as amcatnlo
from madgraph.various import banner


class TestRealChannelJobs(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.command = object.__new__(amcatnlo.aMCatNLOCmd)
        self.command.stop_for_runweb = True
        self.command.me_dir = self.directory.name
        self.command.run_card = banner.RunCardNLO()
        self.command.run_card['real_channels_x0'] = 0.1
        self.subprocess = os.path.join(self.directory.name, 'SubProcesses', 'P0_test')
        os.makedirs(self.subprocess)

    def write_channels(self, born, hard=None):
        with open(os.path.join(self.subprocess, 'channels.txt'), 'w') as stream:
            stream.write(' '.join(born))
        if hard is not None:
            with open(os.path.join(self.subprocess, 'channels_hard.txt'), 'w') as stream:
                stream.write(' '.join(hard))

    def create_jobs(self, run_mode='all', mode='NLO', fixed_order=True):
        return self.command.create_jobs_to_run({'only_generation': False}, ['P0_test'],
                                              -1, run_mode, -1, mode, fixed_order)[0]

    def write_compiled_setting(self, value, p_dir='P0_test', exe='madevent_mintFO'):
        for filename in ('real_channels_x0.dat', exe + '.real_channels_x0.dat'):
            with open(os.path.join(self.directory.name, 'SubProcesses', p_dir, filename),
                      'w') as stream:
                stream.write('%s\n' % value)

    def test_run_card_default_and_fortran_value(self):
        card = banner.RunCardNLO()
        self.assertEqual(card['real_channels_x0'], -1.0)
        self.assertIn('real_channels_x0', card.hidden_param)
        self.assertNotIn('real_channels', card)
        card.read('0.15 = real_channels_x0\n', consistency=False)
        output = io.StringIO()
        card.write_include_file(None, output_file=output)
        assignment = next(line for line in output.getvalue().splitlines()
                          if 'real_channels_x0 =' in line)
        self.assertAlmostEqual(float(assignment.split('=')[1].replace('d', 'e')), 0.15)

    def test_partition_balanced_channel_jobs(self):
        born = ['%d.1' % i for i in range(1, 24)]
        hard = [str(i) for i in range(1, 42)]
        self.write_channels(born, hard)
        jobs = self.create_jobs()
        self.assertEqual([job['channel'] for job in jobs], ['1', '2', '3', '4', '5'])
        self.assertEqual([job['nchans'] for job in jobs], [12, 11, 14, 14, 13])
        for kind, expected in ((0, born), (1, hard)):
            actual = [config for job in jobs if job['channel_type'] == kind
                      for config in job['configs'].split()]
            self.assertEqual(actual, expected)
        self.assertTrue(all(job['real_channels_x0'] == 0.1 for job in jobs))

    def test_empty_channel_pool(self):
        self.write_channels([], ['1', '2'])
        jobs = self.create_jobs()
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]['channel_type'], 1)
        self.assertEqual(jobs[0]['configs'], '1 2')

    def test_disabled_keeps_legacy_grouping_and_ignores_hard_file(self):
        born = ['%d.0' % i for i in range(1, 42)]
        self.write_channels(born, ['1', '2'])
        for value in (-1.0, 0.0, 1.0, 2.0):
            self.command.run_card['real_channels_x0'] = value
            jobs = self.create_jobs()
            self.assertEqual([job['nchans'] for job in jobs], [14, 14, 13])
            self.assertEqual(' '.join(job['configs'] for job in jobs), ' '.join(born))
            self.assertTrue(all(job['channel_type'] == 0 for job in jobs))
            self.assertTrue(all(job['real_channels_x0'] == -1 for job in jobs))

    def test_contribution_modes_without_real_do_not_need_hard_list(self):
        self.write_channels(['1.0'])
        for mode in ('born', 'bovi', 'bosk', 'virt', 'viSA'):
            jobs = self.create_jobs(run_mode=mode)
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0]['channel_type'], 0)
        with self.assertRaisesRegex(amcatnlo.aMCatNLOError, 'Missing hard real-emission'):
            self.create_jobs()

    def test_enabled_rejects_matching_lo_and_merging(self):
        for mode in ('LO', 'aMC@LO', 'noshowerLO', 'aMC@NLO', 'noshower'):
            with self.assertRaisesRegex(self.command.InvalidCmd, 'fixed-order NLO'):
                self.command.check_real_channels_mode(mode)
        for ickkw in (-1, 3, 4):
            self.command.run_card['ickkw'] = ickkw
            with self.assertRaisesRegex(self.command.InvalidCmd, 'ickkw=0'):
                self.command.check_real_channels_mode('NLO')
        self.command.run_card['real_channels_x0'] = 1.0
        self.assertEqual(self.command.check_real_channels_mode('aMC@NLO'), -1.0)

    def test_fixed_order_input_declares_kind_before_channels(self):
        self.write_channels(['1.0'], ['1'])
        jobs = self.create_jobs()
        for job in jobs:
            job['dirname'] = self.subprocess
            self.command.write_input_file(job, fixed_order=True)
            with open(os.path.join(self.subprocess, 'input_app.txt')) as stream:
                text = stream.read()
            if job['channel_type'] == 1:
                self.assertIn('CHANNEL_TYPE = 1\n', text)
                self.assertLess(text.index('\nCHANNEL_TYPE'), text.index('\nCHANNEL ='))
            else:
                self.assertNotIn('CHANNEL_TYPE', text)
        del jobs[0]['channel_type']
        self.command.write_input_file(jobs[0], fixed_order=True)
        with open(os.path.join(self.subprocess, 'input_app.txt')) as stream:
            self.assertNotIn('CHANNEL_TYPE', stream.read())

    def test_nocompile_rejects_setting_changes(self):
        for previous, requested in ((-1., .1), (.1, -1.), (.1, .2), (.2, .1)):
            self.write_compiled_setting('%sd0' % previous)
            self.command.run_card['real_channels_x0'] = requested
            with self.assertRaisesRegex(self.command.InvalidCmd, 'without --nocompile'):
                self.command.check_real_channels_nocompile(['P0_test'])

    def test_nocompile_accepts_unchanged_or_equivalent_disabled_settings(self):
        for previous, requested in (('1.0D-1', .1), ('2d-1', .2),
                                    ('-1d0', 0.), ('0d0', 1.), ('1d0', 2.)):
            self.write_compiled_setting(previous)
            self.command.run_card['real_channels_x0'] = requested
            self.command.check_real_channels_nocompile(['P0_test'])

    def test_nocompile_legacy_process_is_disabled(self):
        self.command.run_card['real_channels_x0'] = -1.
        self.command.check_real_channels_nocompile(['P0_test'])
        self.command.run_card['real_channels_x0'] = .1
        with self.assertRaisesRegex(self.command.InvalidCmd, 'without --nocompile'):
            self.command.check_real_channels_nocompile(['P0_test'])

    def test_nocompile_uses_each_compiled_marker_not_rewritten_include(self):
        source = os.path.join(self.directory.name, 'Source')
        os.makedirs(source)
        with open(os.path.join(source, 'run_card.inc'), 'w') as stream:
            stream.write('      real_channels_x0 = 0.1d0\n')
        self.command.run_card['real_channels_x0'] = .1
        self.write_compiled_setting('1.00000000000000006E-001')
        self.command.check_real_channels_nocompile(['P0_test'])
        second_subprocess = os.path.join(self.directory.name, 'SubProcesses', 'P1_test')
        os.makedirs(second_subprocess)
        self.write_compiled_setting('2.00000000000000011E-001', p_dir='P1_test')
        with self.assertRaisesRegex(self.command.InvalidCmd, 'P1_test'):
            self.command.check_real_channels_nocompile(['P0_test', 'P1_test'])
        with open(os.path.join(self.subprocess, 'real_channels_x0.dat'), 'w') as stream:
            stream.write('compilation incomplete\n')
        with self.assertRaisesRegex(self.command.InvalidCmd, 'without --nocompile'):
            self.command.check_real_channels_nocompile(['P0_test'])

    def test_nocompile_checks_selected_executable_after_switching_modes(self):
        self.write_compiled_setting('0.1', exe='madevent_mintFO')
        self.write_compiled_setting('-1', exe='madevent_mintMC')
        self.command.run_card['real_channels_x0'] = -1.
        self.command.check_real_channels_nocompile(['P0_test'], 'madevent_mintMC')
        with self.assertRaisesRegex(self.command.InvalidCmd, 'without --nocompile'):
            self.command.check_real_channels_nocompile(['P0_test'], 'madevent_mintFO')
        self.command.run_card['real_channels_x0'] = .1
        with self.assertRaisesRegex(self.command.InvalidCmd, 'without --nocompile'):
            self.command.check_real_channels_nocompile(['P0_test'], 'madevent_mintFO')

    def test_nocompile_rejects_unreadable_or_nonfinite_markers(self):
        self.command.run_card['real_channels_x0'] = -1.
        for value in ('NaN', 'Inf', '-Inf', 'compilation incomplete'):
            self.write_compiled_setting(value)
            with self.assertRaisesRegex(self.command.InvalidCmd, 'Cannot determine'):
                self.command.check_real_channels_nocompile(['P0_test'])
        marker = os.path.join(self.subprocess, 'real_channels_x0.dat')
        os.remove(marker)
        os.makedirs(marker)
        with self.assertRaisesRegex(self.command.InvalidCmd, 'Cannot determine'):
            self.command.check_real_channels_nocompile(['P0_test'])

    def test_compile_finishes_executable_before_publishing_marker(self):
        marker = os.path.join(self.subprocess, 'real_channels_x0.dat')
        with open(os.path.join(self.subprocess, 'real_channels.inc'), 'w'):
            pass
        completed = []

        def compile(targets, **kwargs):
            completed.extend(targets)
            with open(marker) as stream:
                self.assertEqual(stream.read(), 'compilation incomplete\n')

        def call(args, **kwargs):
            self.assertEqual(args, ['./gensym', 'NLO'])
            self.assertEqual(completed, ['gensym', 'madevent_mintFO'])
            with open(marker, 'w') as stream:
                stream.write('0.1\n')
            return 0

        with mock.patch.object(amcatnlo.misc, 'compile', side_effect=compile), \
                mock.patch.object(amcatnlo.misc, 'call', side_effect=call):
            amcatnlo.compile_dir(self.directory.name, 'P0_test', 'NLO',
                                {'reweightonly': False}, [], 'madevent_mintFO', 0)
        self.command.check_real_channels_nocompile(['P0_test'])

    def test_failed_rebuild_invalidates_markers(self):
        self.write_compiled_setting('0.1')
        with open(os.path.join(self.subprocess, 'real_channels.inc'), 'w'):
            pass
        with mock.patch.object(amcatnlo.misc, 'compile',
                               side_effect=amcatnlo.MadGraph5Error('build failed')), \
                mock.patch.object(amcatnlo.misc, 'call') as call:
            with self.assertRaisesRegex(amcatnlo.aMCatNLOError, 'build failed'):
                amcatnlo.compile_dir(self.directory.name, 'P0_test', 'NLO',
                                    {'reweightonly': False}, [], 'madevent_mintFO', 0)
            call.assert_not_called()
        with self.assertRaisesRegex(self.command.InvalidCmd, 'Cannot determine'):
            self.command.check_real_channels_nocompile(['P0_test'])

    def test_reweight_only_preserves_compiled_markers(self):
        self.write_compiled_setting('0.1')
        with mock.patch.object(amcatnlo.misc, 'compile') as compile, \
                mock.patch.object(amcatnlo.misc, 'call') as call:
            amcatnlo.compile_dir(self.directory.name, 'P0_test', 'NLO',
                                {'reweightonly': True}, [], 'madevent_mintFO', 0)
            compile.assert_not_called()
            call.assert_not_called()
        self.command.check_real_channels_nocompile(['P0_test'])

    def test_legacy_rebuild_does_not_create_incomplete_markers(self):
        with mock.patch.object(amcatnlo.misc, 'compile'), \
                mock.patch.object(amcatnlo.misc, 'call'):
            amcatnlo.compile_dir(self.directory.name, 'P0_test', 'NLO',
                                {'reweightonly': False}, [], 'madevent_mintFO', 0)
        self.command.run_card['real_channels_x0'] = -1.
        self.command.check_real_channels_nocompile(['P0_test'])

    def test_disabled_mc_jobs_use_legacy_input(self):
        self.command.run_card['real_channels_x0'] = -1.0
        self.write_channels(['1.0', '2.0'], ['1'])
        jobs = self.create_jobs(mode='aMC@NLO', fixed_order=False)
        self.assertEqual([job['channel'] for job in jobs], ['1.0', '2.0'])
        jobs[0]['dirname'] = self.subprocess
        self.command.write_input_file(jobs[0], fixed_order=False)
        with open(os.path.join(self.subprocess, 'input_app.txt')) as stream:
            self.assertNotIn('CHANNEL_TYPE', stream.read())

    def test_restart_accepts_equivalent_disabled_settings(self):
        for old in (None, -1.0, 0.0, 1.0, 2.0):
            job = {} if old is None else {'real_channels_x0': old}
            self.command.check_real_channels_restart([job], -1.0)
            self.assertEqual(job['real_channels_x0'], -1.0)
            self.assertEqual(job['channel_type'], 0)
        for saved, current in (({}, 0.1), ({'real_channels_x0': 0.1}, -1.0),
                               ({'real_channels_x0': 0.1}, 0.2)):
            with self.assertRaisesRegex(amcatnlo.aMCatNLOError, 'Cannot restart'):
                self.command.check_real_channels_restart([saved], current)
        with self.assertRaisesRegex(amcatnlo.aMCatNLOError, 'metadata'):
            self.command.check_real_channels_restart([{'channel_type': 1}], -1.0)

    def test_restart_validates_before_collecting_or_combining(self):
        job = {'dirname': '/old/location/SubProcesses/P0_test/all_G1',
               'channel_type': 1, 'real_channels_x0': 0.2}
        status = os.path.join(self.directory.name, 'SubProcesses', 'job_status.pkl')
        with open(status, 'wb') as stream:
            pickle.dump([job], stream)
        with mock.patch.object(self.command, 'collect_the_results') as collect:
            with self.assertRaisesRegex(amcatnlo.aMCatNLOError, 'Cannot restart'):
                self.command.create_jobs_to_run({'only_generation': True}, ['P0_test'],
                                                -1, 'all', -1, 'NLO')
            collect.assert_not_called()
        self.command.run_card['real_channels_x0'] = 0.2
        with mock.patch.object(self.command, 'collect_the_results',
                               side_effect=lambda options, acc, run, collect, *args:
                               (run, collect)) as collect:
            run, collected, step = self.command.create_jobs_to_run(
                {'only_generation': True}, ['P0_test'], -1, 'all', -1, 'NLO')
            collect.assert_called_once()
        self.assertEqual(run[0]['dirname'], os.path.join(self.subprocess, 'all_G1'))
        self.assertEqual(run[0]['channel_type'], 1)
        self.assertEqual(collected[0]['real_channels_x0'], 0.2)

    def test_split_and_recombine_preserve_channel_kind(self):
        self.write_channels(['1.0'], ['1'])
        jobs = self.create_jobs()
        self.command.options = {'run_mode': 2, 'nb_core': 2}
        for job in jobs:
            job.update(dirname=os.path.join(self.subprocess, 'all_G' + job['channel']),
                       result=1.0, resultABS=1.0, error=0.1, errorABS=0.1,
                       niters=10, npoints=10000, niters_done=1, npoints_done=1000,
                       time_spend=10, combined=1)
        split, collected = self.command.split_jobs_fixed_order(jobs, copy.copy(jobs))
        self.assertEqual(len(split), 4)
        self.assertEqual(len(collected), 4)
        self.assertEqual([job['channel_type'] for job in split], [0, 0, 1, 1])
        with mock.patch.object(self.command, 'combine_split_order_grids') as grids:
            merged = self.command.combine_split_order_run(split)
        self.assertEqual(grids.call_count, 2)
        self.assertEqual([job['channel_type'] for job in merged], [0, 1])
        self.assertTrue(all(job['real_channels_x0'] == 0.1 for job in merged))
        self.assertTrue(all(job['split'] == 0 for job in merged))

    def test_mixed_grid_group_rejected_before_files_are_read(self):
        common = dict(p_dir='P0_test', channel='1', configs='1', real_channels_x0=0.1)
        jobs = [dict(common, channel_type=0), dict(common, channel_type=1)]
        with self.assertRaisesRegex(amcatnlo.aMCatNLOError, 'Cannot combine'):
            self.command.combine_split_order_grids(jobs)
        with self.assertRaisesRegex(amcatnlo.aMCatNLOError, 'Cannot combine'):
            self.command.combine_split_order_jobs(jobs)
