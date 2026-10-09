"""Reject invalid colours/kinematics without retaining part of a folded point."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import ROOT, TEMPLATE, fortran_routine


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestBornFlowRejection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='mg5_born_flow_rejection_')
        cls.addClassCleanup(cls.tmp.cleanup)
        work = Path(cls.tmp.name)
        includes = {
            'nexternal.inc': 'integer nexternal,nincoming\nparameter(nexternal=5,nincoming=2)',
            'nFKSconfigs.inc': 'integer fks_configs\nparameter(fks_configs=1)',
            'genps.inc': 'integer ngraphs,ncolor\nparameter(ngraphs=1,ncolor=3)',
            'born_nhel.inc': 'integer max_bcol\nparameter(max_bcol=3)',
            'orders.inc': 'integer amp_split_size,nsplitorders\n'
                          'parameter(amp_split_size=1,nsplitorders=1)',
            'run.inc': 'integer ickkw\nlogical mcatnlo_delta\n'
                       'common/test_run/ickkw,mcatnlo_delta',
            'fks_info.inc': '',
            'mc_histories.inc': 'integer MC_HIST_COUNT\nparameter(MC_HIST_COUNT=1)',
        }
        for name, source in includes.items():
            (work / name).write_text(''.join('      '+line+'\n' for line in source.splitlines()))
        (work / 'routines.f').write_text(
            fortran_routine(TEMPLATE / 'fks_singular.f', 'get_born_flow') +
            fortran_routine(TEMPLATE / 'fks_singular.f', 'get_born_flow_weights') +
            fortran_routine(TEMPLATE / 'driver_mintMC.f', 'sigintF'))
        # These operations do not participate in the rejection protocol. The
        # fixture below supplies the Born proposals and contribution records.
        noops = ('find_iproc_map', 'setup_event_attributes', 'update_fks_dir',
                 'init_process_module_nbody_wrapper',
                 'set_FxFx_scale', 'set_cms_stuff', 'set_born_spread_point',
                 'set_alphaS', 'include_multichannel_enhance', 'compute_ewsudakov',
                 'compute_shower_scale_nbody', 'save_shower_scale_nbody',
                 'Bornonly_shower_scale', 'compute_shower_scale_n1body',
                 'compute_prefactors_nbody', 'compute_prefactors_n1body',
                 'special_check_SoftSing', 'mc_set_history',
                 'include_bias_wgt', 'sum_identical_contributions',
                 'apply_born_spread_weight')
        (work / 'noops.f90').write_text('\n'.join(
            'subroutine '+name+'\nend subroutine' for name in noops))
        cls.executable = work / 'check_rejection'
        result = subprocess.run([
            shutil.which('gfortran'), '-O2', '-std=legacy', '-fcheck=all',
            '-ffixed-line-length-none', '-I', str(work),
            str(TEMPLATE / 'FKSParams.f90'),
            str(TEMPLATE / 'weight_lines.f'),
            str(ROOT / 'tests/input_files/born_flow_rejection_context.f90'),
            str(work / 'routines.f'), str(work / 'noops.f90'),
            str(ROOT / 'tests/input_files/check_born_flow_rejection.f90'),
            '-o', str(cls.executable)], cwd=work, capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def check_mode(self, mode):
        result = subprocess.run([str(self.executable), mode], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('PASS '+mode, result.stdout)

    def test_nonfinite_and_zero_colour_weights(self):
        self.check_mode('selection')

    def test_outer_rejection_discards_all_contributions(self):
        self.check_mode('outer')

    def test_later_native_failure_discards_earlier_folds(self):
        self.check_mode('native')

    def test_later_kinematic_failure_discards_earlier_folds(self):
        self.check_mode('kinematics')

    def test_kinematic_failure_with_explicit_outer_sum(self):
        self.check_mode('kinematics_exp')

    def test_born_only_rejection(self):
        self.check_mode('born')

    def test_virtual_only_rejection(self):
        self.check_mode('virtual')

    def test_rejection_below_born_cuts(self):
        self.check_mode('uncut')

    def test_rejection_with_explicit_outer_fks_sum(self):
        self.check_mode('explicit')

    def test_outer_rejection_with_native_sector_matching(self):
        self.check_mode('legacy')

    def test_nonfinite_weights_discard_all_folds_and_recover(self):
        for mode in ('weight_nan', 'weight_inf', 'weight_neginf', 'weight_scaled',
                     'weight_pdf', 'weight_parton', 'weight_sum'):
            with self.subTest(mode=mode):
                self.check_mode(mode)
