"""Job coordination contracts shared by the MINT and AmpliCol backends."""

import math
import os
import pickle
import tempfile
import unittest
from unittest import mock

from madgraph.interface.amcatnlo_run_interface import aMCatNLOCmd, aMCatNLOError
from madgraph.various.banner import RunCardNLO


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
                stream.write('5 .1 1 .1 2 100 .5\n')
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
        self.write_card('#NLOPSIntegrator\ninvalid\n')
        jobs, _, _ = self.cmd.create_jobs_to_run(
            {'only_generation': False}, ['P0_test'], -1, 'all', 0,
            'NLO', fixed_order=True)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]['nchans'], 2)
        self.assertNotIn('nlops_integrator', jobs[0])

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
        for step in ('0', '1'):
            _, _, required, _ = self.cmd.getIO_ajob(
                'ajob1', self.process, ['1', 'F', '0', step])
            self.assertIn('GF1/ampli_grids', required)
            self.assertIn('GF1/grid.MC_integer', required)
            self.assertIn('GF1/res_%s.dat' % step, required)
            self.assertNotIn('GF1/mint_grids', required)
        _, _, required, _ = self.cmd.getIO_ajob(
            'ajob1', self.process, ['1', 'F', '1', '2'])
        self.assertNotIn('GF1_1/ampli_grids', required)
        _, _, required, _ = self.cmd.getIO_ajob(
            'ajob1', self.process, ['1', 'all', '0', '0'])
        self.assertIn('all_G1/mint_grids', required)
        self.write_card('#NLOPSIntegrator\n0\n')
        _, _, required, _ = self.cmd.getIO_ajob(
            'ajob1', self.process, ['1', 'F', '0', '0'])
        self.assertNotIn('GF1/ampli_grids', required)


if __name__ == '__main__':
    unittest.main()
