"""Reject unsupported shower radiation before compilation or job submission."""

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from madgraph.fks import fks_base, fks_helas_objects
from madgraph.interface import amcatnlo_run_interface as run_interface
from madgraph.interface import master_interface


class TestShowerPreflight(unittest.TestCase):
    def command(self, shower='PYTHIA6PT', fsr=True):
        command = object.__new__(run_interface.aMCatNLOCmd)
        command.stop_for_runweb = True
        command.run_card = {'parton_shower': shower}
        command.proc_characteristics = {'has_fsr': fsr}
        command.get_characteristics = mock.Mock()
        return command

    def test_unsupported_fsr_stops_before_compile_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            command = self.command()
            command.me_dir = directory
            command.update_status = mock.Mock()
            with self.assertRaisesRegex(run_interface.aMCatNLOError,
                                        'PYTHIA6PT does not support processes with FSR'):
                command.compile('aMC@NLO', {})
            self.assertEqual(list(Path(directory).iterdir()), [])
            command.update_status.assert_not_called()

    def test_direct_run_rejects_before_job_or_seed_setup(self):
        command = self.command()
        command.me_dir = '/unused'
        command.setup_cluster_or_multicore = mock.Mock()
        command.update_random_seed = mock.Mock()
        with self.assertRaisesRegex(run_interface.aMCatNLOError, 'FSR'):
            command.run('noshower', {})
        command.setup_cluster_or_multicore.assert_not_called()
        command.update_random_seed.assert_not_called()

    def test_supported_showers_and_fixed_order_remain_available(self):
        for mode, shower, fsr in [
            ('aMC@NLO', 'PYTHIA6PT', False),
            ('aMC@NLO', 'PYTHIA6Q', True),
            ('aMC@NLO', 'PYTHIA8', True),
            ('aMC@NLO', 'HERWIG6', True),
            ('NLO', 'PYTHIA6PT', True),
            ('LO', 'PYTHIA6PT', True),
        ]:
            with self.subTest(mode=mode, shower=shower, fsr=fsr):
                self.command(shower, fsr).check_shower_compatibility(mode)

    def test_matched_modes_share_the_fsr_guard(self):
        for mode in ('aMC@NLO', 'aMC@LO', 'noshower', 'noshowerLO',
                     'aMC@NLO+extra'):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(run_interface.aMCatNLOError, 'FSR'):
                    self.command('pythia6pt').check_shower_compatibility(mode)

    def test_existing_event_reweighting_does_not_require_shower_matching(self):
        self.command().check_shower_compatibility('aMC@NLO', {'reweightonly': True})


class TestRadiationCharacteristics(unittest.TestCase):
    def test_multicore_matrix_elements_keep_radiation_characteristics(self):
        for process, isr, fsr in [
            ('e+ e- > t t~ [real=QCD]', False, True),
            ('u d~ > e+ ve [real=QCD]', True, False),
        ]:
            with self.subTest(process=process):
                command = master_interface.MasterCmd()
                command.no_notification()
                for line in (
                        'set low_mem_multicore_nlo_generation True --no_save',
                        'set nb_core 2 --no_save', 'import model sm',
                        'set include_lepton_initiated_processes True --no_save',
                        'generate ' + process):
                    command.exec_cmd(line, errorhandling=False, printcmd=False,
                                     precmd=True, postcmd=True)
                multiprocess = command._fks_multi_proc
                matrix_elements = fks_helas_objects.FKSHelasMultiProcess(multiprocess)
                for filename in matrix_elements['matrix_elements']:
                    self.addCleanup(Path(filename).unlink, missing_ok=True)
                self.assertEqual(matrix_elements['has_isr'], isr)
                self.assertEqual(matrix_elements['has_fsr'], fsr)
                self.assertEqual(multiprocess['has_isr'], isr)
                self.assertEqual(multiprocess['has_fsr'], fsr)

    def test_flags_follow_generated_radiation_and_restrictions(self):
        for process, isr, fsr in [
            ('e+ e- > t t~ [real=QCD]', False, True),
            ('u d~ > e+ ve [real=QCD]', True, False),
            # Coloured final particles alone do not establish real radiation.
            ('e+ e- > t t~ [LOonly=QCD]', False, False),
        ]:
            with self.subTest(process=process):
                command = master_interface.MasterCmd()
                command.no_notification()
                for line in (
                        'set low_mem_multicore_nlo_generation False --no_save',
                        'import model sm',
                        'set include_lepton_initiated_processes True --no_save',
                        'generate ' + process):
                    command.exec_cmd(line, errorhandling=False, printcmd=False,
                                     precmd=True, postcmd=True)
                multiprocess = command._fks_multi_proc
                self.assertEqual(multiprocess['has_isr'], isr)
                self.assertEqual(multiprocess['has_fsr'], fsr)
                for born in multiprocess['born_processes']:
                    self.assertEqual(born.isr, isr)
                    self.assertEqual(born.fsr, fsr)

    def test_discarded_real_amplitudes_do_not_supply_radiation_flags(self):
        born = object.__new__(fks_base.FKSProcess)
        born.nincoming = 2
        initial = mock.Mock(fks_infos=[{'j': 1}])
        final = mock.Mock(fks_infos=[{'j': 4}])
        born.real_amps = [initial, final]
        born.update_radiation_flags()
        self.assertTrue(born.isr)
        self.assertTrue(born.fsr)
        # The multicore path passes only real amplitudes retained by workers.
        born.update_radiation_flags([initial])
        self.assertTrue(born.isr)
        self.assertFalse(born.fsr)
        born.update_radiation_flags([])
        self.assertFalse(born.isr)
        self.assertFalse(born.fsr)
