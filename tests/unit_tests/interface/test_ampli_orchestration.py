"""Job coordination contracts shared by the MINT and AmpliCol backends."""

import math
import json
import os
import pickle
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest import mock

from madgraph.interface.amcatnlo_run_interface import aMCatNLOCmd, aMCatNLOError
from madgraph.various.banner import RunCardNLO
from madgraph.various import ampli_pool
from tests.unit_tests.fks.test_ampli_lhe import candidate_file
from tests.unit_tests.various.test_ampli_pool import native_pool_file


class TestAmpliOrchestration(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cmd = object.__new__(aMCatNLOCmd)
        self.cmd.stop_for_runweb = True
        self.cmd.me_dir = self.directory.name
        self.cmd.options = {}
        self.cmd.run_card = {
            'folding': [1, 1, 1], 'born_spreading': False,
            'npoints_FO_grid': 100, 'niters_FO_grid': 2,
            'nevents': 10, 'nevt_job': 3,
        }
        self.cmd.cross_sect_dict = {
            'xseca': 10., 'erra': .1, 'xsect': 2., 'errt': .1,
            'nevents': 10, 'p_labels': [],
        }
        self.process = os.path.join(self.directory.name, 'SubProcesses', 'P0_test')
        os.makedirs(self.process)
        os.makedirs(os.path.join(self.directory.name, 'Cards'))
        self.card = os.path.join(self.directory.name, 'Cards', 'FKS_params.dat')
        with open(os.path.join(self.process, 'channels.txt'), 'w') as stream:
            stream.write('1 2\n')
        self.cmd.get_randinit_seed = mock.Mock(return_value=123)
        seed_marker = mock.patch(
            'madgraph.interface.amcatnlo_run_interface.random.mg_seedset',
            123, create=True)
        seed_marker.start()
        self.addCleanup(seed_marker.stop)
        self.cmd.get_pdf_input_filename = mock.Mock(
            return_value=os.path.join(self.directory.name, 'absent_pdf_input'))

    def write_card(self, text):
        with open(self.card, 'w') as stream:
            stream.write(text)

    def create_jobs(self, restart=False, fixed_order=False):
        return self.cmd.create_jobs_to_run(
            {'only_generation': restart}, ['P0_test'], -.1, 'all', 1,
            'noshower', fixed_order=fixed_order)

    def save_jobs(self, jobs):
        self.cmd.prepare_directories(jobs, 'noshower', fixed_order=False)
        for job in jobs:
            with open(os.path.join(job['dirname'], 'res_1.dat'), 'w') as stream:
                stream.write('5 .1 1 .1 4 100 .5\n')
        with open(os.path.join(self.directory.name, 'SubProcesses', 'job_status.pkl'),
                  'wb') as stream:
            pickle.dump(jobs, stream)

    def test_backend_default_selection_and_validation(self):
        self.assertEqual(self.cmd.get_nlops_integrator(), 0)
        self.write_card('#IRPoleCheckThreshold\n1d-5\n')
        self.assertEqual(self.cmd.get_nlops_integrator(), 0)
        for value in (0, 1):
            self.write_card('! backend\n#NLOPSIntegrator\n%d ! selected\n' % value)
            self.assertEqual(self.cmd.get_nlops_integrator(), value)
        for text in ('#NLOPSIntegrator\n2\n', '#NLOPSIntegrator\n-1\n',
                     '#NLOPSIntegrator\n1.0\n', '#NLOPSIntegrator\n',
                     '#NLOPSIntegrator\n#Other\n0\n',
                     '#NLOPSIntegrator\n1\n#NLOPSIntegrator\n0\n'):
            self.write_card(text)
            with self.assertRaises(aMCatNLOError):
                self.cmd.get_nlops_integrator()

    def test_separate_channels_and_fixed_order_ignores_backend(self):
        self.write_card('#NLOPSIntegrator\n1\n')
        jobs, _, _ = self.create_jobs()
        self.assertEqual([job['channel'] for job in jobs], ['1', '2'])
        self.assertTrue(all(job['nlops_integrator'] == 1 for job in jobs))
        self.assertTrue(all(job['mint_mode'] == 1 for job in jobs))
        self.assertTrue(all(job['accuracy'] == .03 for job in jobs))
        self.write_card('#NLOPSIntegrator\ninvalid\n')
        jobs, _, _ = self.cmd.create_jobs_to_run(
            {'only_generation': False}, ['P0_test'], -1, 'all', 0,
            'NLO', fixed_order=True)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]['nchans'], 2)
        self.assertNotIn('nlops_integrator', jobs[0])

    def check_run_stages(self, backend, restart=False):
        """Exercise real scheduling, result parsing, quotas and checkpoint links."""
        self.write_card('#NLOPSIntegrator\n%d\n' % backend)
        self.cmd.run_card.update(parton_shower='PYTHIA8', event_norm='average',
                                 req_acc=-1, folding=[2, 1, 1])
        self.cmd.ninitial = 2
        self.cmd.cluster_mode = 0
        self.cmd.run_name = 'test_run'
        self.cmd.results = mock.Mock()
        for method in ('get_characteristics', 'setup_cluster_or_multicore',
                       'update_random_seed', 'clean_previous_results',
                       'update_status', 'collect_log_files', 'print_summary',
                       'check_event_files'):
            setattr(self.cmd, method, mock.Mock())
        self.cmd.make_make_all_html_results = mock.Mock(return_value=(2., .1))
        self.cmd.reweight_and_collect_events = mock.Mock(return_value='events.lhe')
        self.cmd.finalize_ampli_pools = mock.Mock(side_effect=lambda jobs, mode: ([], jobs))
        for filename, content in (('subproc.mg', 'P0_test\n'),
                                  ('orderstags_glob.dat', '1\n0\n')):
            with open(os.path.join(self.directory.name, 'SubProcesses', filename), 'w') as stream:
                stream.write(content)
        launches = []
        def execute(jobs, step, fixed_order=False):
            self.assertFalse(fixed_order)
            launches.append(step)
            for job in jobs:
                self.assertEqual(job['mint_mode'], step)
                with open(os.path.join(job['dirname'], 'input_app.txt')) as stream:
                    rows = stream.readlines()
                self.assertEqual(int(rows[7].split()[0]), step)
                self.assertEqual([int(value) for value in rows[8].split('!')[0].split()],
                                 [1, 1, 1] if step == 0 else [2, 1, 1])
                state = 'ampli_grids' if backend else 'mint_grids'
                if backend and step == 1:
                    self.assertEqual(job['accuracy'], .03)
                    self.assertFalse(os.path.exists(os.path.join(job['dirname'], state)))
                    self.assertFalse(os.path.exists(os.path.join(job['dirname'], 'res_0.dat')))
                if step == 2:
                    self.assertTrue(os.path.isfile(os.path.join(job['dirname'], state)))
                    self.assertTrue(os.path.isfile(os.path.join(job['dirname'], 'res_1')))
                else:
                    for filename in (state, 'grid.MC_integer', 'res_1'):
                        with open(os.path.join(job['dirname'], filename), 'w') as stream:
                            stream.write('survey state\n')
                with open(os.path.join(job['dirname'], 'res_%d.dat' % step), 'w') as stream:
                    stream.write('5 .1 1 .1 4 100 .5\n')
        self.cmd.run_all_jobs = mock.Mock(side_effect=execute)
        options = dict(reweightonly=False, only_generation=False)
        self.assertEqual(self.cmd.run('noshower', options), 'events.lhe')
        self.assertEqual(launches, [1, 2] if backend else [0, 1, 2])
        if restart:
            launches[:] = []
            options['only_generation'] = True
            # Older accurate checkpoints with too few survey iterations must
            # not silently bypass the new minimum when restarting production.
            survey_result = os.path.join(self.process, 'GF1', 'res_1.dat')
            with open(survey_result, 'w') as stream:
                stream.write('5 .1 1 .1 3 100 .5\n')
            with self.assertRaisesRegex(aMCatNLOError, 'at least 4 iterations'):
                self.cmd.run('noshower', options)
            self.assertEqual(launches, [])
            with open(survey_result, 'w') as stream:
                stream.write('5 .1 1 .1 4 100 .5\n')
            self.assertEqual(self.cmd.run('noshower', options), 'events.lhe')
            self.assertEqual(launches, [2])

    def test_ampli_runs_survey_then_generation_and_restarts_without_resurvey(self):
        self.check_run_stages(1, restart=True)

    def test_mint_keeps_three_integration_stages(self):
        self.check_run_stages(0)

    def test_job_creation_with_real_nlo_run_card(self):
        self.cmd.run_card = RunCardNLO()
        self.write_card('#NLOPSIntegrator\n1\n')
        jobs, _, _ = self.create_jobs()
        self.assertEqual(len(jobs), 2)
        self.assertEqual(jobs[0]['integrator_config']['run_settings']['born_spreading'],
                         self.cmd.run_card['born_spreading'])
        self.save_jobs(jobs)
        restarted, _, _ = self.create_jobs(restart=True)
        self.assertEqual(restarted[0]['nlops_integrator'], 1)

    def test_ampli_restart_checks_physics_and_allows_generation_controls(self):
        self.cmd.run_card = RunCardNLO()
        self.write_card('#NLOPSIntegrator\n1\n')
        param_path = os.path.join(self.directory.name, 'Cards', 'param_card.dat')
        with open(param_path, 'w') as stream:
            stream.write('BLOCK MASS\n 6 172.5\n')
        jobs, _, _ = self.create_jobs()
        self.save_jobs(jobs)
        for key, value in (('nevents', 101), ('iseed', 1234), ('nevt_job', 10),
                           ('event_norm', 'sum'), ('req_acc', .1),
                           ('store_rwgt_info', True), ('reweight_scale', [False]),
                           ('rw_rscale', [1., 4.]), ('lhe_version', 2),
                           ('lhaid', [244600, 244601]),
                           ('dynamical_scale_choice', [-1, 1])):
            self.cmd.run_card[key] = value
        self.create_jobs(restart=True)
        for key, value in (('ebeam1', 7000.), ('ptj', 20.),
                           ('lhaid', [244601]), ('dynamical_scale_choice', [1]),
                           ('parton_shower', 'PYTHIA8'), ('event_norm', 'bias')):
            previous = self.cmd.run_card[key]
            self.cmd.run_card[key] = value
            with self.assertRaisesRegex(aMCatNLOError, 'physics settings'):
                self.create_jobs(restart=True)
            self.cmd.run_card[key] = previous
        with open(param_path, 'w') as stream:
            stream.write('BLOCK MASS\n 6 173.0\n')
        with self.assertRaisesRegex(aMCatNLOError, 'param_card.dat'):
            self.create_jobs(restart=True)

    def test_mint_restart_keeps_existing_configuration_policy(self):
        self.cmd.run_card = RunCardNLO()
        jobs, _, _ = self.create_jobs()
        self.save_jobs(jobs)
        self.cmd.run_card['ebeam1'] = 7000.
        self.cmd.run_card['folding'] = [2, 1, 1]
        self.write_card('#NLOPSIntegrator\n0\n#IRPoleCheckThreshold\n1d-4\n')
        restarted, _, _ = self.create_jobs(restart=True)
        self.assertEqual(restarted[0]['nlops_integrator'], 0)

    def test_restart_checks_backend_and_grid_configuration(self):
        self.write_card('#NLOPSIntegrator\n1\n#IRPoleCheckThreshold\n1d-5\n')
        jobs, _, _ = self.create_jobs()
        self.save_jobs(jobs)
        self.cmd.run_card['nevents'] = 100
        restarted, _, _ = self.create_jobs(restart=True)
        self.assertEqual(restarted[0]['nlops_integrator'], 1)
        self.assertEqual(restarted[0]['resultABS'], 5.)
        # Comments do not affect checkpoint compatibility.
        self.write_card('#NLOPSIntegrator\n1 ! unchanged\n'
                        '#IRPoleCheckThreshold\n1d-5\n! new comment\n')
        self.create_jobs(restart=True)
        self.write_card('#NLOPSIntegrator\n0\n#IRPoleCheckThreshold\n1d-5\n')
        with self.assertRaisesRegex(aMCatNLOError, 'integrator differs'):
            self.create_jobs(restart=True)
        self.write_card('#NLOPSIntegrator\n1\n#IRPoleCheckThreshold\n1d-4\n')
        with self.assertRaisesRegex(aMCatNLOError, 'settings'):
            self.create_jobs(restart=True)
        self.write_card('#NLOPSIntegrator\n1\n#IRPoleCheckThreshold\n1d-5\n')
        self.cmd.run_card['folding'] = [2, 1, 1]
        with self.assertRaisesRegex(aMCatNLOError, 'folding'):
            self.create_jobs(restart=True)
        self.cmd.run_card['folding'] = [1, 1, 1]
        self.cmd.run_card['born_spreading'] = True
        with self.assertRaisesRegex(aMCatNLOError, 'Born spreading'):
            self.create_jobs(restart=True)

    def test_legacy_saved_jobs_are_mint(self):
        jobs, _, _ = self.create_jobs()
        for job in jobs:
            del job['nlops_integrator']
            del job['integrator_config']
        self.save_jobs(jobs)
        restarted, _, _ = self.create_jobs(restart=True)
        self.assertEqual(restarted[0]['nlops_integrator'], 0)
        self.write_card('#NLOPSIntegrator\n1\n')
        with self.assertRaisesRegex(aMCatNLOError, 'integrator differs'):
            self.create_jobs(restart=True)

    def test_generation_split_preserves_quota_and_backend_state(self):
        for backend, state in ((0, 'mint_grids'), (1, 'ampli_grids')):
            self.write_card('#NLOPSIntegrator\n%d\n' % backend)
            self.cmd.run_card['born_spreading'] = True
            jobs, _, _ = self.create_jobs()
            self.cmd.prepare_directories(jobs, 'noshower', fixed_order=False)
            for job in jobs:
                job['mint_mode'] = 2
                job['nevents'] = 10 if job['channel'] == '1' else 0
                for name in (state, 'grid.MC_integer', 'res_1', 'born_spreading.dat'):
                    with open(os.path.join(job['dirname'], name), 'w') as stream:
                        stream.write('trained state\n')
            split, collected = self.cmd.check_the_need_to_split(jobs, jobs)
            self.assertEqual([job['nevents'] for job in split], [3, 3, 2, 2])
            self.assertEqual(sum(job['nevents'] for job in collected), 10)
            self.assertAlmostEqual(sum(job['wgt_frac'] for job in split), 1.)
            self.cmd.prepare_directories(split, 'noshower', fixed_order=False)
            for job in split:
                self.assertEqual(job['nlops_integrator'], backend)
                for name in (state, 'grid.MC_integer', 'born_spreading.dat'):
                    target = os.path.join(job['dirname'], name)
                    self.assertTrue(os.path.islink(target))
                    self.assertEqual(os.path.realpath(target),
                                     os.path.join(jobs[0]['dirname'], name))
                with open(os.path.join(job['dirname'], 'input_app.txt')) as stream:
                    text = stream.read()
                self.assertIn('0 %d ! process label' % job['nevents'], text)

    def test_quota_allocation_excludes_zero_channels_and_is_exact(self):
        jobs = [{'resultABS': value} for value in (0., 3., 7., 0.)]
        # The exact zero RNG endpoint must select the first positive channel.
        with mock.patch('madgraph.interface.amcatnlo_run_interface.random.random',
                        side_effect=[0., .9] * 5):
            updated = self.cmd.update_jobs_to_run(-1, 1, jobs, fixed_order=False)
        self.assertEqual([job['nevents'] for job in updated], [0, 5, 5, 0])
        self.assertEqual(sum(job['nevents'] for job in updated), 10)
        self.assertTrue(all(job['mint_mode'] == 2 for job in updated))

    def test_zero_total_can_integrate_but_cannot_generate(self):
        self.cmd.cross_sect_dict['xseca'] = 0.
        self.cmd.cross_sect_dict['xsect'] = 0.
        jobs = [{'resultABS': 0.}, {'resultABS': 0.}]
        updated = self.cmd.update_jobs_to_run(-1, 0, jobs, fixed_order=False)
        self.assertTrue(all(job['accuracy'] == .2 for job in updated))
        with self.assertRaisesRegex(aMCatNLOError, 'absolute cross section is zero'):
            self.cmd.update_jobs_to_run(-1, 1, jobs, fixed_order=False)
        self.cmd.run_card['nevents'] = 0
        updated = self.cmd.update_jobs_to_run(.1, 1, jobs, fixed_order=False)
        self.assertEqual([job['nevents'] for job in updated], [0, 0])

    def test_ampli_survey_is_three_percent_for_every_event_channel(self):
        jobs = [dict(resultABS=rate, nlops_integrator=1) for rate in (9.999, .001, 0.)]
        for requested_accuracy in (.001, .2, -1):
            self.cmd.update_jobs_to_run(requested_accuracy, 0, jobs, fixed_order=False)
            self.assertEqual([job['accuracy'] for job in jobs], [.03] * 3)
        self.cmd.run_card['nevents'] = 0
        jobs = [{'resultABS': 10., 'nlops_integrator': 1}]
        self.cmd.update_jobs_to_run(.001, 0, jobs, fixed_order=False)
        self.assertEqual(jobs[0]['accuracy'], .03)
        self.cmd.run_card['nevents'] = 10
        jobs[0]['nlops_integrator'] = 0
        self.cmd.update_jobs_to_run(.001, 0, jobs, fixed_order=False)
        self.assertEqual(jobs[0]['accuracy'], .001)

    def test_ampli_quotas_are_exact_reproducible_and_have_ten_percent_reserve(self):
        self.cmd.run_card['nevents'] = 100
        jobs = [dict(resultABS=rate, errorABS=.03 * rate, nlops_integrator=1, niters_done=4)
                for rate in (6., 4., 0.)]
        self.cmd.update_jobs_to_run(.003, 1, jobs, fixed_order=False)
        quotas = [job['ampli_final_quota'] for job in jobs]
        self.assertEqual(sum(quotas), 100)
        self.assertEqual(quotas[2], 0)
        self.assertTrue(all(0 < quota < 100 for quota in quotas[:2]))
        self.assertEqual([job['ampli_generated_target'] for job in jobs],
                         [(11 * quota + 9) // 10 for quota in quotas])
        self.assertTrue(all('ampli_budget' not in job for job in jobs))
        self.cmd.cross_sect_dict['xsect'] = 0.
        self.cmd.update_jobs_to_run(.1, 1, jobs, fixed_order=False)
        self.assertEqual([job['ampli_final_quota'] for job in jobs], quotas)
        self.assertTrue(all(job['mint_mode'] == 2 for job in jobs))

    def test_ampli_refuses_an_inaccurate_or_invalid_survey(self):
        for rate, error in ((10., .30001), (0., .1), (-1., 0.),
                            (10., -1.), (float('nan'), .1), (10., float('inf'))):
            with self.subTest(rate=rate, error=error):
                jobs = [dict(resultABS=rate, errorABS=error, nlops_integrator=1, niters_done=4)]
                with self.assertRaisesRegex(aMCatNLOError, 'survey did not reach 3%'):
                    self.cmd.update_jobs_to_run(.1, 1, jobs, fixed_order=False)
        # A tiny positive channel has no minimum 1000-event pool.
        jobs = [dict(resultABS=10., errorABS=.1, nlops_integrator=1, niters_done=4)]
        self.cmd.update_jobs_to_run(.1, 1, jobs, fixed_order=False)
        self.assertEqual(jobs[0]['ampli_final_quota'], 10)
        self.assertEqual(jobs[0]['ampli_generated_target'], 11)

    def test_ampli_refuses_accurate_survey_with_fewer_than_four_iterations(self):
        for iterations in (0, 1, 2, 3):
            jobs = [dict(resultABS=10., errorABS=0., nlops_integrator=1,
                         niters_done=iterations)]
            with self.subTest(iterations=iterations):
                with self.assertRaisesRegex(aMCatNLOError, 'at least 4 iterations'):
                    self.cmd.update_jobs_to_run(-1, 1, jobs, fixed_order=False)

    def test_ampli_split_event_protocol_and_unique_restart_streams(self):
        self.write_card('#NLOPSIntegrator\n1\n')
        channels, _, _ = self.create_jobs()
        self.cmd.prepare_directories(channels, 'noshower', fixed_order=False)
        for channel in channels:
            channel.update(resultABS=5., errorABS=.1, error=.1, npoints_done=200,
                           niters_done=4)
            for name in ('ampli_grids', 'grid.MC_integer', 'res_1'):
                with open(os.path.join(channel['dirname'], name), 'w') as stream:
                    stream.write('trained state\n')
        self.cmd.run_card['nevt_job'] = 3
        self.cmd.plan_ampli_production(channels, .1)
        jobs = self.cmd.split_ampli_batch(channels)
        self.cmd.prepare_directories(jobs, 'noshower', fixed_order=False)
        self.assertEqual(sum(job['ampli_final_quota'] for job in jobs), 10)
        for channel in channels:
            parts = [job for job in jobs if job['channel'] == channel['channel']]
            self.assertEqual(sum(job['ampli_final_quota'] for job in parts), channel['nevents'])
            self.assertEqual(sum(job['nevents'] for job in parts), channel['ampli_generated_target'])
            self.assertAlmostEqual(sum(job['wgt_frac'] for job in parts), 1.)
        for job in jobs:
            self.assertLessEqual(job['nevents'], 3)
            self.assertEqual(job['nevents'], (11 * job['ampli_final_quota'] + 9) // 10)
            with open(os.path.join(job['dirname'], 'ampli_job.dat')) as stream:
                self.assertEqual(stream.read(), 'MG5_AMPLI_JOB 2\n%d %d\n' %
                                 (job['nevents'], job['ampli_final_quota']))
            self.assertEqual(os.path.realpath(os.path.join(job['dirname'], 'ampli_grids')),
                             os.path.join(job['ampli_parent'], 'ampli_grids'))
        restarted = self.cmd.split_ampli_batch(channels)
        self.assertFalse({job['dirname'] for job in jobs} & {job['dirname'] for job in restarted})
        self.assertEqual(sum(job['ampli_final_quota'] for job in restarted), 10)
        self.cmd.run_card['nevt_job'] = 1
        with self.assertRaisesRegex(aMCatNLOError, 'nevt_job must be at least 2'):
            self.cmd.split_ampli_batch(channels)

    def test_collection_preserves_survey_rates_and_collects_legacy_worker_quotas(self):
        for event_norm, normalization in (('average', 10.), ('sum', 1.), ('unity', 1.)):
            with self.subTest(event_norm=event_norm):
                self.check_survey_collection(event_norm, normalization)

    def check_survey_collection(self, event_norm, normalization):
        self.write_card('#NLOPSIntegrator\n1\n')
        self.cmd.run_card.update(event_norm=event_norm, nevt_job=-1)
        channels, _, _ = self.create_jobs()
        self.cmd.prepare_directories(channels, 'noshower', fixed_order=False)
        for index, channel in enumerate(channels):
            channel.update(resultABS=5., result=float(index + 1), errorABS=.1, error=.1,
                           npoints_done=100, niters_done=4, time_spend=.5,
                           err_percABS=2., err_perc=10., nevents=6 if index == 0 else 4,
                           ampli_final_quota=6 if index == 0 else 4, mint_mode=2)
        batches = self.cmd.split_ampli_batch(channels)
        self.cmd._ampli_channels = channels
        self.cmd.run_name = 'test_run'
        self.cmd.results = mock.Mock()
        self.cmd.make_make_all_html_results = mock.Mock(return_value=(3., .2))
        self.cmd.run_all_jobs = mock.Mock()
        pool_data = {}
        for index, job in enumerate(batches):
            pool_data[job['dirname']] = dict(trials=4000 + index,
                available=job['nevents'], generated_target=job['nevents'],
                final_quota=job['ampli_final_quota'],
                # Deliberately incompatible production diagnostics must never
                # replace published survey rates or change final quotas.
                absolute=600., signed=200., error_abs=20., error_signed=10.)
        def append_worker_results(jobs, step):
            self.assertEqual(step, 2)
            for job in jobs:
                job.update(resultABS=600., result=200., errorABS=20., error=10.,
                           npoints_done=4000, time_spend=2.)
        self.cmd.append_the_results = mock.Mock(side_effect=append_worker_results)
        with mock.patch('madgraph.interface.amcatnlo_run_interface.ampli_pool.read_pool',
                        side_effect=lambda path: pool_data[path]), \
             mock.patch('madgraph.interface.amcatnlo_run_interface.ampli_pool.combine_moments',
                        side_effect=lambda pools: dict(trials=sum(pool['trials'] for pool in pools),
                                                       absolute=600., signed=200.)), \
             mock.patch('madgraph.interface.amcatnlo_run_interface.ampli_pool.available_events',
                        side_effect=lambda pools: sum(pool['available'] for pool in pools)), \
             mock.patch('madgraph.interface.amcatnlo_run_interface.ampli_pool.finalize_channel',
                        return_value={}) as finalize:
            remaining, finalized = self.cmd.finalize_ampli_pools(batches, 'noshower')
        self.assertEqual(remaining, [])
        self.cmd.run_all_jobs.assert_not_called()
        self.assertEqual(self.cmd.cross_sect_dict['xseca'], 10.)
        self.assertEqual(self.cmd.cross_sect_dict['xsect'], 3.)
        self.assertAlmostEqual(self.cmd.cross_sect_dict['errt'], math.sqrt(.02))
        self.assertEqual([job['nevents'] for job in finalized], [6, 4])
        self.assertEqual(sum(job['nevents'] for job in finalized), 10)
        self.assertEqual([call.args[1] for call in finalize.call_args_list], [6, 4])
        for call in finalize.call_args_list:
            self.assertEqual(call.args[3], normalization)
            row = call.kwargs['cross_sections'][0]
            self.assertEqual(row[:2], (0, 3.))
            self.assertAlmostEqual(row[2], math.sqrt(.02))
        with open(os.path.join(self.directory.name, 'SubProcesses', 'ampli_production.json')) as stream:
            manifest = json.load(stream)
        self.assertEqual(manifest['version'], 2)
        self.assertEqual(manifest['rate_source'], 'survey')
        self.assertEqual(manifest['sampling'], 'frozen')
        self.assertEqual(manifest['weight_convention'], 'draw_time_importance')
        self.assertEqual(manifest['survey_relative_accuracy'], .03)
        self.assertEqual(manifest['requested_events'], 10)
        self.assertEqual(manifest['allowed_overweight_factor'], .01)
        self.assertEqual(sum(channel['quota'] for channel in manifest['channels']), 10)
        self.assertEqual(manifest['generation_trials'], 8001)
        self.assertEqual(manifest['generation_cpu_seconds'], 4.)
        self.assertTrue(all(channel['survey_points'] == 100 for channel in manifest['channels']))
        self.assertTrue(all(channel['production_moments_diagnostic']['absolute'] == 600.
                            for channel in manifest['channels']))
        self.assertTrue(all(worker['adaptation'] == {'mode': 'frozen'}
                            for channel in manifest['channels'] for worker in channel['batches']))

    def test_collection_rejects_wrong_worker_quota_before_writing(self):
        self.cmd._ampli_channels = []
        self.cmd.append_the_results = mock.Mock()
        jobs = [dict(dirname=self.process, nevents=11, ampli_final_quota=10)]
        with mock.patch('madgraph.interface.amcatnlo_run_interface.ampli_pool.read_pool',
                        return_value=dict(generated_target=11, final_quota=9)):
            with self.assertRaisesRegex(aMCatNLOError, 'quotas disagree'):
                self.cmd.finalize_ampli_pools(jobs, 'noshower')

    def test_live_collection_merges_split_workers_and_keeps_survey_normalization(self):
        self.write_card('#NLOPSIntegrator\n1\n')
        self.cmd.run_card.update(event_norm='average', nevt_job=3)
        channels, _, _ = self.create_jobs()
        self.cmd.prepare_directories(channels, 'noshower', fixed_order=False)
        for index, channel in enumerate(channels):
            channel.update(p_label='42', resultABS=5., result=float(index + 1),
                           errorABS=.1, error=.1, npoints_done=100, niters_done=4,
                           time_spend=.5, err_percABS=2., err_perc=10., mint_mode=2,
                           nevents=6 if index == 0 else 4,
                           ampli_final_quota=6 if index == 0 else 4)
        workers = self.cmd.split_ampli_batch(channels)
        self.assertEqual(len(workers), 5)
        for worker_index, job in enumerate(workers):
            os.makedirs(job['dirname'])
            target, quota = job['nevents'], job['ampli_final_quota']
            # Constant trial weights give exactly unit corrections. The last
            # candidate lies on the excluded threshold; all others are kept.
            lines = ['MG5_AMPLI_POOL 3', '%d %d 100' % (target + 1, target + 1),
                     '100 100 0 0 0', '%d %d 100 0 0 0' % (target, quota),
                     '3 1 2 4 %d' % (target - 1), '1 0 1']
            lines += ['100 %.16e 1 0 1' % math.log(200. + index)
                      for index in range(target)]
            lines += ['100 %.16e 0 0 1' % math.log(100.)]
            with open(os.path.join(job['dirname'], 'ampli_pool.dat'), 'w') as stream:
                stream.write('\n'.join(lines) + '\n')
            content = candidate_file([1.] * (target + 1))
            content = content.replace('# candidate ', '# worker %d candidate ' % worker_index)
            with open(os.path.join(job['dirname'], 'ampli_candidates.lhe'), 'w') as stream:
                stream.write(content)
            with open(os.path.join(job['dirname'], 'res_2.dat'), 'w') as stream:
                stream.write('100 0 100 0 1 %d 2\n' % (target + 1))
        self.cmd._ampli_channels = channels
        self.cmd.run_name = 'test_run'
        self.cmd.results = mock.Mock()
        self.cmd.make_make_all_html_results = mock.Mock(return_value=(3., .2))
        remaining, finalized = self.cmd.finalize_ampli_pools(workers, 'noshower')
        self.assertEqual(remaining, [])
        self.assertEqual(self.cmd.cross_sect_dict['xsect'], 3.)
        self.assertEqual(self.cmd.cross_sect_dict['xseca'], 10.)
        self.assertAlmostEqual(self.cmd.cross_sect_dict['errt'], math.sqrt(.02))
        identities = []
        for channel in finalized:
            with open(os.path.join(channel['dirname'], 'events.lhe')) as stream:
                content = stream.read()
            tree = ET.fromstring(content)
            events = tree.findall('event')
            self.assertEqual(len(events), channel['ampli_final_quota'])
            for event in events:
                self.assertEqual(float(event.text.split()[2]), 10.)
                identities.extend(line.strip() for line in event.text.splitlines()
                                  if '# worker' in line)
            init = tree.find('init').text.splitlines()
            self.assertEqual(init[1].split()[-2], '-4')
            rate, error, unused, label = map(float, init[2].split())
            self.assertEqual((rate, label), (3., 42.))
            self.assertAlmostEqual(error, math.sqrt(.02))
        self.assertEqual(len(identities), 10)
        self.assertEqual(len(set(identities)), 10)
        with open(os.path.join(self.directory.name, 'SubProcesses', 'ampli_production.json')) as stream:
            manifest = json.load(stream)
        self.assertEqual(manifest['generation_trials'], 20)
        self.assertEqual(manifest['sampling'], 'adaptive_unfolded')
        self.assertEqual(manifest['weight_convention'], 'draw_time_importance')
        self.assertEqual(manifest['generation_cpu_seconds'], 10.)
        self.assertEqual(sum(row['generated_target'] for row in manifest['channels']), 15)
        for row in manifest['channels']:
            self.assertEqual(row['available_candidates'], row['generated_target'])
            for worker in row['batches']:
                self.assertEqual(worker['pool_protocol'], 3)
                self.assertEqual(worker['adaptation'], dict(
                    mode='adaptive_unfolded', ndim=3, mask=[1, 0, 1],
                    updates=1, interval=2, batch=4, points=2, completed_batches=1))
            self.assertEqual(row['production_moments_diagnostic']['absolute'], 100.)
            self.assertEqual(row['selection']['selected_tail'], 0.)
            self.assertEqual(row['selection']['selected'], row['quota'])

    def test_native_collection_updates_rates_reallocates_and_tops_up_short_channel(self):
        self._native_collection_updates_rates_reallocates_and_tops_up_short_channel(False)

    def test_native_proposal_history_collection_accepts_legacy_workers_and_new_topups(self):
        self._native_collection_updates_rates_reallocates_and_tops_up_short_channel(True)

    def _native_collection_updates_rates_reallocates_and_tops_up_short_channel(self, proposal_history):
        self.write_card('#NLOPSIntegrator\n1\n')
        self.cmd.run_card.update(event_norm='sum', nevt_job=-1)
        channels, _, _ = self.create_jobs()
        self.cmd.prepare_directories(channels, 'noshower', fixed_order=False)
        for index, channel in enumerate(channels):
            channel.update(p_label='42', resultABS=5., result=float(index+1),
                           errorABS=.1, error=.1, npoints_done=100, niters_done=4,
                           time_spend=.5, err_percABS=2., err_perc=10., mint_mode=2)
        self.cmd.plan_ampli_production(channels, .01)
        initial_quotas = [channel['ampli_final_quota'] for channel in channels]
        self.assertEqual(initial_quotas, [4, 6])
        workers = self.cmd.split_ampli_batch(channels)
        self.cmd._ampli_channels = channels
        self.cmd.run_name = 'native_run'
        self.cmd.results = mock.Mock()
        self.cmd.make_make_all_html_results = mock.Mock(return_value=(1., .1))
        self.cmd.collect_log_files = mock.Mock()
        generated = []
        def write_workers(jobs, step, fixed_order=False):
            self.assertEqual(step, 2)
            self.assertFalse(fixed_order)
            for job in jobs:
                value = 1000. if job['channel'] == '1' else .1
                native_pool_file(job['dirname'], [[value]*job['nevents']], job['ampli_final_quota'],
                                 proposal_ids=[1] if proposal_history and generated else None)
                with open(os.path.join(job['dirname'], 'res_2.dat'), 'w') as stream:
                    stream.write('%g 0 %g 0 1 %d 2\n' % (value, value, job['nevents']))
                generated.append(job)
        write_workers(workers, 2)
        self.cmd.run_all_jobs = mock.Mock(side_effect=write_workers)
        remaining, finalized = self.cmd.finalize_ampli_pools(workers, 'noshower')
        self.assertEqual(remaining, [])
        self.cmd.run_all_jobs.assert_called_once()
        topups = self.cmd.run_all_jobs.call_args.args[0]
        self.assertEqual(len(topups), 1)
        self.assertEqual(topups[0]['channel'], '1')
        self.assertEqual(topups[0]['ampli_final_quota'], 5)
        self.assertEqual(topups[0]['nevents'], 6)
        self.assertEqual([channel['ampli_final_quota'] for channel in channels], [10, 0])
        self.assertEqual(len(finalized), 1)
        self.assertEqual(sum(channel['nevents'] for channel in finalized), 10)
        # Survey 100 points enters once, including across the second worker.
        expected_absolute = (500.+11000.)/111. + (500.+.7)/107.
        expected_signed = (100.+11000.)/111. + (200.+.7)/107.
        self.assertAlmostEqual(self.cmd.cross_sect_dict['xseca'], expected_absolute)
        self.assertAlmostEqual(self.cmd.cross_sect_dict['xsect'], expected_signed)
        with open(os.path.join(finalized[0]['dirname'], 'events.lhe')) as stream:
            tree = ET.fromstring(stream.read())
        self.assertEqual(len(tree.findall('event')), 10)
        for event in tree.findall('event'):
            self.assertAlmostEqual(float(event.text.split()[2]), expected_absolute/10.)
        init = tree.find('init').text.splitlines()
        self.assertEqual(init[1].split()[-2], '-4')
        self.assertAlmostEqual(float(init[2].split()[0]), expected_signed)
        with open(os.path.join(self.directory.name, 'SubProcesses', 'ampli_production.json')) as stream:
            manifest = json.load(stream)
        self.assertEqual(manifest['version'], 4)
        self.assertEqual(manifest['rate_source'], 'survey_plus_production')
        self.assertEqual(manifest['integration_stages'], ['survey', 'generation'])
        self.assertEqual(manifest['survey_min_iterations'], 4)
        self.assertTrue(all(row['survey_iterations'] == 4 for row in manifest['channels']))
        self.assertEqual(len(manifest['rounds']), 2)
        self.assertEqual(manifest['generation_trials'], 18)
        self.assertEqual(manifest['generation_cpu_seconds'], 6.)
        self.assertEqual([row['initial_quota'] for row in manifest['channels']], [4, 6])
        self.assertEqual([row['quota'] for row in manifest['channels']], [10, 0])
        self.assertEqual([row['survey_points'] for row in manifest['channels']], [100, 100])
        self.assertEqual(manifest['channels'][0]['updated_rates']['trials'], 111)
        self.assertEqual(manifest['channels'][1]['updated_rates']['trials'], 107)
        self.assertGreater(manifest['channels'][0]['error_abs'], .1)
        if proposal_history:
            self.assertEqual(manifest['channels'][0]['batches'][-1]['adaptation']['proposals'], 1)
            self.assertEqual(manifest['channels'][0]['batches'][-1]['adaptation']['history'],
                             'last_eight_event_proposals')

    def test_updated_quota_draws_keep_channel_order_when_reporting_sorts_errors(self):
        channels = [dict(dirname=self.process, p_dir='P0_test', channel='1', p_label='42',
                         resultABS=5., result=1., errorABS=.1, error=.1,
                         niters_done=1, npoints_done=100, time_spend=0., err_perc=10., err_percABS=2., wgt_frac=1.),
                    dict(dirname=self.process, p_dir='P0_test', channel='2', p_label='42',
                         resultABS=5., result=1., errorABS=10., error=.1,
                         niters_done=1, npoints_done=100, time_spend=0., err_perc=10., err_percABS=200., wgt_frac=1.)]
        self.cmd._allocate_updated_ampli_quotas(channels)
        expected = {row['channel']: row['ampli_final_quota'] for row in channels}
        self.cmd.write_res_txt_file(list(channels), 2)
        self.cmd._allocate_updated_ampli_quotas(channels)
        self.assertEqual({row['channel']: row['ampli_final_quota'] for row in channels}, expected)
        self.assertEqual([row['channel'] for row in channels], ['1', '2'])

    def test_updated_allocation_preserves_initial_cdf_despite_separate_report_list_order(self):
        jobs = [dict(resultABS=5., errorABS=.1, niters_done=4, p_dir='P0_test', channel='1'),
                dict(resultABS=5., errorABS=.1, niters_done=4, p_dir='P0_test', channel='2')]
        self.cmd.plan_ampli_production(jobs, .01)
        expected = {job['channel']: job['ampli_final_quota'] for job in jobs}
        self.assertEqual(expected, {'1': 4, '2': 6})
        channels = [dict(job) for job in reversed(jobs)]
        for channel in channels:
            channel['ampli_final_quota'] = -1
        self.cmd._allocate_updated_ampli_quotas(channels)
        self.assertEqual({job['channel']: job['ampli_final_quota'] for job in channels}, expected)
        self.assertEqual([job['ampli_allocation_order'] for job in channels], [1, 0])

    def test_native_tail_rethresholding_triggers_targeted_supply_topups_before_collection(self):
        self.write_card('#NLOPSIntegrator\n1\n')
        self.cmd.run_card.update(event_norm='average', nevt_job=-1, nevents=400)
        self.cmd.cross_sect_dict['nevents'] = 400
        channels, _, _ = self.create_jobs()
        self.cmd.prepare_directories(channels, 'noshower', fixed_order=False)
        for index, channel in enumerate(channels):
            channel.update(p_label='42', resultABS=5., result=float(index+1),
                           errorABS=.1, error=.1, npoints_done=100, niters_done=4,
                           time_spend=.5, err_percABS=2., err_perc=10., mint_mode=2)
        self.cmd.plan_ampli_production(channels, .01)
        self.assertEqual([row['ampli_final_quota'] for row in channels], [207, 193])
        workers = self.cmd.split_ampli_batch(channels)
        self.cmd._ampli_channels = channels
        self.cmd.run_name = 'native_tail_run'
        self.cmd.results = mock.Mock()
        self.cmd.make_make_all_html_results = mock.Mock(return_value=(1., .1))
        self.cmd.collect_log_files = mock.Mock()
        generated = []
        def write_workers(jobs, step, fixed_order=False):
            for job in jobs:
                value = 1. if job['channel'] == '1' else 10.
                if not generated and job['channel'] == '1':
                    points = [2.]+[1.]*(job['nevents']-1)
                    candidates = [(1, 2., math.log(4.), 1.)] + [
                        (1, 1., math.log(1.5), 1.)]*(job['nevents']-1)
                    native_pool_file(job['dirname'], [points], job['ampli_final_quota'],
                                     envelopes=[1.], candidates=candidates, proposal_ids=[1])
                else:
                    native_pool_file(job['dirname'], [[value]*job['nevents']], job['ampli_final_quota'],
                                     proposal_ids=[1])
                with open(os.path.join(job['dirname'], 'res_2.dat'), 'w') as stream:
                    stream.write('%g 0 %g 0 1 %d 2\n' % (value, value, job['nevents']))
                generated.append(job)
        write_workers(workers, 2)
        path = os.path.join(workers[0]['dirname'], 'ampli_pool.dat')
        with open(path) as stream:
            immutable_original = stream.read()
        first_pool = ampli_pool.read_pool(path)
        self.assertLess(ampli_pool.pool_status([first_pool], 207)['overweight'], .01)
        self.cmd.run_all_jobs = mock.Mock(side_effect=write_workers)
        original_select = ampli_pool.select_candidates
        def select_after_topups(*args, **kwargs):
            self.assertTrue(self.cmd.run_all_jobs.called)
            return original_select(*args, **kwargs)
        with mock.patch.object(ampli_pool, 'select_candidates', side_effect=select_after_topups) as select:
            remaining, finalized = self.cmd.finalize_ampli_pools(workers, 'noshower')
        self.assertEqual(remaining, [])
        self.assertEqual(select.call_count, len(finalized))
        self.assertEqual(sum(row['nevents'] for row in finalized), 400)
        with open(path) as stream:
            self.assertEqual(stream.read(), immutable_original)
        with open(os.path.join(self.directory.name, 'SubProcesses', 'ampli_production.json')) as stream:
            manifest = json.load(stream)
        first = manifest['channels'][0]
        self.assertLess(first['quota'], first['initial_quota'])
        self.assertGreater(first['batches'][0]['collection_log_z'], first['batches'][0]['native_log_z'])
        self.assertEqual(first['batches'][0]['available'], 1)
        self.assertEqual(first['selection']['selected_tail'], 0.)
        self.assertGreater(len(manifest['rounds']), 1)
        self.assertEqual(first['updated_rates']['survey_trials'], 100)
        self.assertEqual(manifest['generation_trials'], sum(job['nevents'] for job in generated))

    def test_signed_cancellation_does_not_break_result_parsing(self):
        with open(os.path.join(self.process, 'res_1.dat'), 'w') as stream:
            stream.write('5 .1 0 .1 2 100 .5\n')
        job = {'dirname': self.process}
        self.cmd.append_the_results([job], 1)
        self.assertEqual(job['result'], 0.)
        self.assertTrue(math.isinf(job['err_perc']))
        self.assertEqual(job['err_percABS'], 2.)

    def test_cluster_requires_new_checkpoint_only_for_ampli_integration(self):
        self.write_card('#NLOPSIntegrator\n1\n')
        self.cmd.run_card['born_spreading'] = True
        for step in ('0', '1'):
            _, _, required, _ = self.cmd.getIO_ajob(
                'ajob1', self.process, ['1', 'F', '0', step])
            self.assertIn('GF1/ampli_grids', required)
            self.assertIn('GF1/grid.MC_integer', required)
            self.assertIn('GF1/res_%s.dat' % step, required)
            self.assertNotIn('GF1/mint_grids', required)
            self.assertEqual('GF1/born_spreading.dat' in required, step == '1')
        _, _, required, _ = self.cmd.getIO_ajob(
            'ajob1', self.process, ['1', 'F', '1', '2'])
        self.assertNotIn('GF1_1/ampli_grids', required)
        self.assertIn('GF1_1/ampli_pool.dat', required)
        self.assertIn('GF1_1/ampli_candidates.lhe', required)
        self.assertIn('GF1_1/res_2.dat', required)
        _, _, required, _ = self.cmd.getIO_ajob(
            'ajob1', self.process, ['1', 'all', '0', '0'])
        self.assertIn('all_G1/mint_grids', required)
        self.write_card('#NLOPSIntegrator\n0\n')
        _, _, required, _ = self.cmd.getIO_ajob(
            'ajob1', self.process, ['1', 'F', '0', '0'])
        self.assertNotIn('GF1/ampli_grids', required)
        self.assertIn('GF1/born_spreading.dat', required)


if __name__ == '__main__':
    unittest.main()
