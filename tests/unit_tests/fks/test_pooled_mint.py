"""Check replica normalization at the production MINT observation boundary."""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestPooledMint(unittest.TestCase):
    def test_channel_loop_budget_adapts_to_the_frozen_residual(self):
        source = (ROOT / 'Template/NLO/SubProcesses/mint_module.f90').read_text()
        routine = re.search(r'^  subroutine update_virtual_fraction\b.*?'
                            r'^  end subroutine update_virtual_fraction', source,
                            re.M | re.S).group(0)
        with tempfile.TemporaryDirectory(prefix='pooled_fraction_') as tmp:
            work = Path(tmp)
            (work/'driver.f90').write_text('''module fixture
 implicit none
 integer,parameter :: nchans=2,n_ord_virt=2
 logical :: use_pooled_virtual=.true.,pooled_virtual_applying=.false.
 real(8) :: etot(10,2)=0d0,virtual_fraction(2)=.4d0,min_virt_fraction_mint=.005d0
contains
'''+routine+'''
end module
program check
 use fixture
 implicit none
 etot(1,:)=1d0
 etot(7,:)=[.01d0,2d0]
 call update_virtual_fraction
 if(any(abs(virtual_fraction-[.1d0,.8d0])>1d-14))error stop 'collection budget did not adapt'
 pooled_virtual_applying=.true.
 call update_virtual_fraction
 if(any(abs(virtual_fraction-[.025d0,1d0])>1d-14))error stop 'frozen fit did not save loops'
 use_pooled_virtual=.false.
 call update_virtual_fraction
 if(any(abs(virtual_fraction-[.00625d0,1d0])>1d-14))error stop 'legacy budget did not adapt'
end program
''')
            compiled = subprocess.run([
                shutil.which('gfortran'), '-O2', '-fcheck=all',
                '-ffpe-trap=invalid,zero,overflow', str(work/'driver.f90'),
                '-o', str(work/'check')], cwd=work, capture_output=True,
                text=True, timeout=60)
            self.assertEqual(compiled.returncode, 0, compiled.stdout+compiled.stderr)
            checked = subprocess.run([str(work/'check')], cwd=work,
                                     capture_output=True, text=True, timeout=30)
            self.assertEqual(checked.returncode, 0, checked.stdout+checked.stderr)

    def test_capture_removes_only_replica_multiplier(self):
        source = (ROOT / 'Template/NLO/SubProcesses/mint_module.f90').read_text()
        routine = re.search(r'^  subroutine accumulate_the_point\(x\).*?'
                            r'^  end subroutine accumulate_the_point', source,
                            re.M | re.S).group(0)
        with tempfile.TemporaryDirectory(prefix='pooled_mint_') as tmp:
            work = Path(tmp)
            (work / 'routine.inc').write_text(routine + '\n')
            (work / 'driver.f90').write_text('''
module fixture
 use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
 implicit none
 integer,parameter :: ndimmax=4,ndim=4,nintegrals=10,n_ord_virt=2,nint_used_virt=8
 integer :: imode=0,ichan=1,iconfigs(1)=[7],nit=3,calls=0
 logical :: use_pooled_virtual=.true.,pooled_virtual_collecting=.true.,pass_cuts_check=.false.
 real(8) :: f(10)=0d0,wgt_mult=1d0,vtot(10,1)=0d0,etot(10,1)=0d0
 real(8) :: captured_born(2),captured_residual(2)
 integer :: non_zero_point(10)=0,pass_cuts_point=0
contains
 include 'routine.inc'
 subroutine pooled_virtual_observe(born,residual,channel,iteration,x,bins)
  real(8),intent(in) :: born(:),residual(:),x(:)
  integer,intent(in) :: channel,iteration,bins
  if(channel/=7.or.iteration/=3)error stop 'observation routing'
  if(size(x)/=1.or.any(x/=.5d0).or.bins/=8)error stop 'MINT coordinates missing'
  calls=calls+1;captured_born=born;captured_residual=residual
 end subroutine
 subroutine add_point_to_grids(x)
  real(8) :: x(:)
 end subroutine
 subroutine add_point_to_bounding_envelope
 end subroutine
end module
program check
 use fixture
 implicit none
 real(8) :: x(4),original(10)
 integer :: replicas
 x=.5d0
 original=0d0;original([8,10])=[6d0,-10d0];original([7,9])=[0d0,240d0]
 do replicas=1,4
  wgt_mult=1d0/replicas;f=original*wgt_mult;vtot=0d0;etot=0d0
  call accumulate_the_point(x)
  if(any(abs(captured_born-original([8,10]))>1d-12))error stop 'Born replica normalization'
  if(any(abs(captured_residual-original([7,9]))>1d-12))error stop 'residual replica normalization'
  if(any(f/=original*wgt_mult))error stop 'modified physical integrand'
  if(any(vtot(:,1)/=f))error stop 'physical accumulation changed'
 enddo
 if(calls/=4)error stop 'lost collection'
 pooled_virtual_collecting=.false.
 call accumulate_the_point(x)
 if(calls/=4)error stop 'applied model collected more training'
 print *,'PASS replica-independent virtual capture'
end program
subroutine HwU_add_points
end subroutine
''')
            compiled = subprocess.run([
                shutil.which('gfortran'), '-O2', '-fcheck=all',
                '-ffpe-trap=invalid,zero,overflow', '-I', str(work),
                str(work / 'driver.f90'), '-o', str(work / 'check')],
                cwd=work, capture_output=True, text=True, timeout=60)
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            checked = subprocess.run([str(work / 'check')], cwd=work,
                                     capture_output=True, text=True, timeout=30)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
            self.assertIn('PASS replica-independent', checked.stdout)
