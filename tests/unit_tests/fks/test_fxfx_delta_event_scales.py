"""Exercise production FxFx event metadata with directed Delta scales."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import fortran_routine


ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = ROOT / 'Template/NLO/SubProcesses'


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestFxFxDeltaEventScales(unittest.TestCase):
    def test_scalar_marker_and_directed_scales_survive_resonances(self):
        with tempfile.TemporaryDirectory(prefix='mg5_fxfx_delta_event_') as tmp:
            work = Path(tmp)
            includes = {
                'nexternal.inc': 'integer nexternal,nincoming\n'
                                 'parameter(nexternal=5,nincoming=2)',
                'nFKSconfigs.inc': 'integer fks_configs\nparameter(fks_configs=1)',
                'coupl.inc': 'double precision g,gal(2),mu_r\n'
                             'common/test_couplings/g,gal,mu_r',
            }
            for name, contents in includes.items():
                (work / name).write_text(''.join('      ' + line + '\n'
                                                for line in contents.splitlines()))
            # Run the production resonance remapping and FxFx marker code.
            # Earlier add_write_info work constructs the process/decay tree;
            # this fixture supplies that tree directly, with one resonance.
            source = (TEMPLATE / 'add_write_info.f').read_text()
            tail = source[source.index('c     Remove non-resonant mothers,'):
                          source.index('      subroutine set_itree(')]
            wrapper = '''      subroutine finish_event_metadata(nexpart,nres,ns,
     $     jpart,icolalt,pb,shower_scale_a,npart)
      use process_module
      use scale_module
      implicit none
      include 'nexternal.inc'
      integer nexpart,nres,ns,npart,i,j,ires,ickkw
      common/test_ickkw/ickkw
      integer jpart(7,-2:7),icolalt(2,-2:7),ito(-2:5)
      double precision pb(0:4,-2:7),shower_scale_a(-2:7,-2:7)
      integer need_matching(5)
      common/c_need_matching_to_write/need_matching
      integer iSorH_lhe,ifks_lhe(1),jfks_lhe(1),fksfather_lhe(1),
     $     ipartner_lhe(1)
      common/cto_LHE1/iSorH_lhe,ifks_lhe,jfks_lhe,
     $     fksfather_lhe,ipartner_lhe
      double precision ptclus,matching_scale
      character temp*600,temp0*7,float*18,integfour*4
      character*1000 ptclusstring
      common/c_ptclusstring/ptclusstring
'''
            (work / 'event_metadata.f').write_text(wrapper + tail + '\n' +
                fortran_routine(TEMPLATE / 'write_event.f', 'write_events_lhe'))
            (work / 'extra_weights.f90').write_text('''module extra_weights
  logical :: doreweight=.false.
  integer :: iwgtinfo=0,iwgtnumpartn=0
end module
''')
            executable = work / 'check_event_scales'
            result = subprocess.run([
                shutil.which('gfortran'), '-O2', '-std=legacy', '-fcheck=all',
                '-ffixed-line-length-none', '-ffunction-sections', '-fdata-sections',
                '-Wl,-dead_strip' if sys.platform == 'darwin' else '-Wl,--gc-sections',
                '-I', str(work), str(TEMPLATE / 'qed_shower_support.f90'),
                str(TEMPLATE / 'process_module.f90'),
                str(TEMPLATE / 'fks_phase_space_data.f'),
                str(TEMPLATE / 'genps_fks_helpers.f'),
                str(TEMPLATE / 'mcatnlo_delta_scales.f90'),
                str(TEMPLATE / 'herwig7_scales.f90'),
                str(TEMPLATE / 'scale_module.f90'), str(work / 'extra_weights.f90'),
                str(work / 'event_metadata.f'),
                str(ROOT / 'tests/input_files/check_fxfx_delta_event_scales.f90'),
                '-o', str(executable)], cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for mode in ('delta', 'ordinary'):
                result = subprocess.run([str(executable), mode], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('PASS ' + mode, result.stdout)


if __name__ == '__main__':
    unittest.main()
