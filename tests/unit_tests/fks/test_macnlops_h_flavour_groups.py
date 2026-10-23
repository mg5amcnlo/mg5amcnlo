"""Complete H grouping must retain each member's native flavour row."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import ROOT, TEMPLATE, fortran_routine


@unittest.skipUnless(shutil.which('gfortran'), 'gfortran is required')
class TestMacNLOPSHFlavourGroups(unittest.TestCase):
    def test_reordered_rows_group_before_sign_and_preserve_reweighting(self):
        with tempfile.TemporaryDirectory(prefix='macnlops-h-flavours-') as temporary:
            work = Path(temporary)
            includes = {
                'nexternal.inc': 'integer nexternal,nincoming\nparameter(nexternal=5,nincoming=2)',
                'genps.inc': 'integer maxproc\nparameter(maxproc=3)',
                'nFKSconfigs.inc': 'integer fks_configs\nparameter(fks_configs=2)',
                'fks_info.inc': 'integer fks_i_d(2),fks_j_d(2),pdg_type_d(2,5)\n'
                                'parameter(fks_i_d=[5,5],fks_j_d=[1,2])\n'
                                'parameter(pdg_type_d=21)',
                'run.inc': 'logical macnlops\ninteger Flavour_Bias(2)\n'
                           'common/test_mac_mode/macnlops,Flavour_Bias',
            }
            for name, text in includes.items():
                (work / name).write_text(''.join('      '+line+'\n' for line in text.splitlines()))
            (work / 'contexts.f90').write_text('''module mint_module
  integer :: born_spread_phase=0,ndim=1,ifold(1)=1,imode=1
  logical :: only_virt=.false.
end module
module process_module
  integer,parameter :: ndelH=5
end module
module scale_module
  double precision :: emsca_H(2,1,5,5)=0d0
  integer :: event_colour_H(2,5,2,1)=0
end module
module mc_native_context
  integer :: history_permutations(5,2)
end module
module extra_weights
  integer :: n_ctr_found=0,n_mom_conf=0,max_mom_str=1,max_mext=5,max_n_ctr=1
  double precision :: wgtref
  double precision,allocatable :: momenta_str(:,:,:)
  character(len=1024),allocatable :: n_ctr_str(:)
end module
''')
            routines = ('sum_identical_contributions', 'include_inverse_bias_wgt', 'fill_rwgt_lines')
            (work / 'groups.f').write_text('\n'.join(
                fortran_routine(TEMPLATE / 'fks_singular.f', name) for name in routines))
            executable = work / 'check'
            compiled = subprocess.run(['gfortran', '-O2', '-fcheck=all',
                '-ffixed-line-length-none', '-I'+str(work),
                str(work / 'contexts.f90'), str(TEMPLATE / 'weight_lines.f'),
                str(work / 'groups.f'), str(ROOT / 'tests/input_files/check_macnlops_h_flavour_groups.f90'),
                '-o', str(executable)], cwd=work, capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stdout+compiled.stderr)
            result = subprocess.run([str(executable)], cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertIn('PASS native H flavour groups, bias and reweight rows', result.stdout)
            rows = [line.split()[1:] for line in result.stdout.splitlines() if line.startswith('RWGT ')]
            self.assertEqual(len(rows), 2)
            self.assertEqual([list(map(int, row[6:11])) for row in rows], [[2,21,6,-6,2]]*2)
            self.assertEqual([float(row[-6].replace('D','E')) for row in rows], [-7.,10.])
            self.assertEqual([list(map(int,row[-4:])) for row in rows], [[7,8,1,1],[9,10,2,1]])
