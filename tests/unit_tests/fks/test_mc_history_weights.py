"""Compare specialized history weights with the complete native evaluator."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import ROOT, TEMPLATE, fortran_routine


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestMCHistoryWeights(unittest.TestCase):
    def test_h_and_s_paths_preserve_native_weights(self):
        with tempfile.TemporaryDirectory(prefix='mg5_mc_history_weights_') as tmp:
            work = Path(tmp)
            includes = {
                'nexternal.inc': 'integer nexternal,nincoming\n'
                                 'parameter(nexternal=5,nincoming=2)',
                'genps.inc': 'integer maxproc\nparameter(maxproc=1)',
                'coupl.inc': 'double precision g\ncommon/test_couplings/g',
                'orders.inc': 'integer nsplitorders,amp_split_size,qcd_pos,qed_pos,cpower_pos\n'
                              'parameter(nsplitorders=2,amp_split_size=3)\n'
                              'parameter(qcd_pos=1,qed_pos=2,cpower_pos=0)\n'
                              'double precision amp_split(3),test_mass\n'
                              'common/test_amplitudes/amp_split\ncommon/test_mass/test_mass',
                'pmass.inc': 'pmass=0d0\npmass(3)=test_mass',
                'run.inc': 'integer ickkw,lpp(2)\n'
                           'double precision scale,q2fact(2),xbk(2)\n'
                           'common/test_run/scale,q2fact,xbk,ickkw,lpp',
                'q_es.inc': 'double precision QES2\ncommon/test_qes/QES2',
                'fks_info.inc': 'integer fks_i_d(1)\nparameter(fks_i_d=[5])',
            }
            for name, source in includes.items():
                (work / name).write_text(''.join('      '+s+'\n' for s in source.splitlines()))
            for name in ('timing_variables.inc', 'fks_powers.inc'):
                shutil.copyfile(TEMPLATE / name, work / name)
            (work / 'contexts.f90').write_text(
                'module extra_weights\n'
                'integer :: QCD_power,orders_tag,amp_pos\n'
                'double precision :: wgtcpower\nend module\n'
                'module mc_native_context\n'
                'integer :: active_history=0,native_provider_ids(1)=1,native_context_ids(1)=1\n'
                'end module\n'
                'module FKSParams\n'
                'integer :: VetoedContributionTypes(0:1)=0,SelectedContributionTypes(0:1)=0\n'
                'integer :: SelectedCouplingOrders(2,0:1)=0\n'
                'integer :: QCD_squared_selected=-1,QED_squared_selected=-1\nend module\n'
                'module mint_module\nlogical :: pass_cuts_check=.false.\nend module\n'
                'module kinematics_module\n'
                'double precision :: gfactsf,gfactcl,gfactazi\nend module\n')
            routines = ('compute_native_NLOPS_weights', 'compute_real_emission',
                        'compute_soft_counter_term', 'compute_collinear_counter_term',
                        'compute_soft_collinear_counter_term', 'add_wgt')
            (work / 'weights.f').write_text('\n'.join(
                fortran_routine(TEMPLATE / 'fks_singular.f', name) for name in routines))
            executable = work / 'check_history_weights'
            result = subprocess.run([
                shutil.which('gfortran'), '-O2', '-std=legacy', '-fcheck=all',
                '-ffixed-line-length-none', '-I', str(work),
                str(work / 'contexts.f90'), str(TEMPLATE / 'weight_lines.f'),
                str(work / 'weights.f'),
                str(ROOT / 'tests/input_files/check_mc_history_weights.f90'),
                '-o', str(executable)], cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(executable)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('PASS native H and S weights', result.stdout)
