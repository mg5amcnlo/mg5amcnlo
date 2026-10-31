"""Production FKS card policy for the common-scale Delta source."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks.test_momentum_maps import TEMPLATE


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestDeltaCommonScaleConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='delta-common-config-')
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.work = Path(cls.tmp.name)
        (cls.work / 'orders.inc').write_text('integer nsplitorders\nparameter(nsplitorders=1)\n')
        (cls.work / 'native.dat').write_text('#DeltaCommonScale\n.true.\n')
        (cls.work / 'old.dat').write_text('#DeltaBackwardISR\n.false.\n')
        (cls.work / 'check.f90').write_text('''program check
use FKSParams
implicit none
character(64) arg,shower
integer merging,incoming
logical delta
if(delta_common_scale)error stop 'declaration default'
call FKSParamReader('native.dat',.false.,.true.)
if(.not.delta_common_scale)error stop 'read native-history policy'
if(command_argument_count()==0)then
  call validate_delta_common_scale(.true.,'PYTHIA8',0,2)
  call FKSParamReader('old.dat',.false.,.true.)
  if(delta_common_scale)error stop 'old card reset'
  delta_common_scale=.true.
  call DefaultFKSParam()
  if(delta_common_scale)error stop 'default reset'
  print *, 'PASS common-scale policy'
else
  call get_command_argument(1,arg)
  read(arg,*)delta
  call get_command_argument(2,shower)
  call get_command_argument(3,arg)
  read(arg,*)merging
  call get_command_argument(4,arg)
  read(arg,*)incoming
  call validate_delta_common_scale(delta,shower,merging,incoming)
endif
end program
''')
        cls.exe = cls.work / 'check'
        result = subprocess.run([shutil.which('gfortran'), '-O0', '-fcheck=all',
            '-I', str(cls.work), str(TEMPLATE / 'FKSParams.f90'),
            str(cls.work / 'check.f90'), '-o', str(cls.exe)], cwd=cls.work,
            capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def test_default_override_old_card_reset_and_valid_policy(self):
        result = subprocess.run([str(self.exe)], cwd=self.work, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('PASS common-scale policy', result.stdout)

    def test_capi_identifies_native_ir_as_a_distinct_source_mode(self):
        source = (TEMPLATE / 'macnlops_capi.f').read_text()
        start = source.index('      subroutine macnlops_c_source_mode(mode)')
        end = source.index('\n      end', start) + len('\n      end')
        (self.work / 'mode.f').write_text(source[start:end] + '\n')
        (self.work / 'run.inc').write_text(
            '      logical mcatnlo_delta\n      common/test_delta_mode/mcatnlo_delta\n')
        (self.work / 'mode_check.f90').write_text('''program check
use FKSParams
use, intrinsic :: iso_c_binding, only: c_int
implicit none
include 'run.inc'
integer(c_int) mode
interface
  subroutine source_mode(mode) bind(C,name='macnlops_c_source_mode')
    import c_int
    integer(c_int) mode
  end subroutine
end interface
mcatnlo_delta=.false.
delta_common_scale=.false.
call source_mode(mode)
if(mode/=0)error stop 'ordinary mode'
mcatnlo_delta=.true.
call source_mode(mode)
if(mode/=1)error stop 'per-dipole Delta mode'
delta_common_scale=.true.
call source_mode(mode)
if(mode/=3)error stop 'native/global-IR must not reuse winner mode 2'
end program
''')
        result = subprocess.run([shutil.which('gfortran'), '-O0',
            '-ffixed-line-length-none', '-I', str(self.work),
            str(TEMPLATE / 'FKSParams.f90'), str(self.work / 'mode.f'),
            str(self.work / 'mode_check.f90'), '-o', str(self.work / 'mode_check')],
            cwd=self.work, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        result = subprocess.run([str(self.work / 'mode_check')], cwd=self.work,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_incompatible_source_policies_fail_before_integration(self):
        for args in [('false', 'PYTHIA8', '0', '2'), ('true', 'HERWIG7', '0', '2'),
                     ('true', 'PYTHIA8', '3', '2'), ('true', 'PYTHIA8', '0', '1')]:
            with self.subTest(args=args):
                result = subprocess.run([str(self.exe), *args], cwd=self.work,
                                        capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('DeltaCommonScale requires', result.stdout + result.stderr)
