"""Native virtual fits preserve weighted targets, fallback and saved grids."""

from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = ROOT / 'Template/NLO/SubProcesses'


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestVirtualSurrogate(unittest.TestCase):
    def compile_and_run(self, source, extra_sources=()):
        with tempfile.TemporaryDirectory(prefix='mg5_native_virtual_') as tmp:
            work = Path(tmp)
            paths = []
            for name, content in extra_sources:
                path = work / name
                path.write_text(content)
                paths.append(str(path))
            (work / 'check.f90').write_text(source)
            result = subprocess.run([
                shutil.which('gfortran'), '-O2', '-fcheck=all',
                '-ffpe-trap=invalid,zero,overflow', '-ffixed-line-length-none',
                str(TEMPLATE / 'virtual_surrogate.f90'), *paths,
                str(work / 'check.f90'), '-o', str(work / 'check')],
                cwd=work, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(work / 'check')], cwd=work,
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return result.stdout

    def test_features_map_colour_links_and_reset_between_points(self):
        self.compile_and_run('''program check
use virt_surrogate
implicit none
real(8) p(0:3,4),l
p(:,1)=[50d0,0d0,0d0,50d0]
p(:,2)=[50d0,0d0,0d0,-50d0]
p(:,3)=[50d0,50d0,0d0,0d0]
p(:,4)=[50d0,-50d0,0d0,0d0]
call vs_reset
! Real legs 4 and 5 become Born legs 3 and 4 when leg 3 is unresolved.
call vs_add_link(p,100d0,3,4,5,-2d0)
call vs_add_link(p,100d0,3,4,4,7d0)
call vs_features(p,3d0,100d0,2)
l=log(100d0)
if(.not.vs_featok.or.vs_nfeat/=16)error stop 'missing mode-2 features'
if(abs(vs_feat(2)+2d0*l)>1d-12)error stop 'incorrect colour logarithm'
if(abs(vs_feat(3)+2d0*l*l)>1d-12)error stop 'incorrect squared logarithm'
if(vs_feat(1)/=3d0.or.vs_feat(4)/=7d0)error stop 'incorrect Born features'
if(abs(vs_feat(5)-3d0*l)>1d-12)error stop 'missing all-pair logarithms'
call vs_features(p,3d0,100d0,1)
if(vs_nfeat/=4)error stop 'mode 1 retained all-pair features'
vs_vcomp=.true.;vs_ratio_set=.true.;vs_virt=9d0
call vs_begin_point
call vs_features(p,3d0,100d0,2)
if(vs_featok.or.vs_vcomp.or.vs_ratio_set.or.vs_virt/=0d0)error stop 'stale point state'
end program
''')

    def test_weighted_observations_fit_and_native_fallback_selection(self):
        modules = '''module weight_lines
integer :: icontr=2,itype(2)=[14,15],amppos(2)=[1,1]
real(8) :: wgts(1,2)=0d0
end module
module mint_module
integer :: imode=0,ichan=1,VirtTrickSurrogate=2
logical :: use_poly_virtual=.false.
real(8) :: polyfit(0:1)=0d0,average_virtual(0:1,1)=0d0
end module
'''
        self.compile_and_run('''program check
use virt_surrogate
use weight_lines
use mint_module
implicit none
integer i,j
real(8) value,c
call vs_reset
! Train on orthogonal features with unequal PDF/bias weights. The
! production helper must undo sampling before forming the fit weight.
do i=1,80
  j=1+mod(i-1,4)
  vs_feat=0d0;vs_feat(j)=1d0
  value=real(j,8);c=real(1+mod(i,3),8)
  vs_vcomp=.true.;vs_featok=.true.;vs_virt=value;vs_born=1d0;vs_fvirt=.25d0
  wgts(1,1)=c*value/vs_fvirt;wgts(1,2)=0d0
  call virt_surrogate_fit_point
enddo
call vs_solve(1,.true.)
vs_feat=0d0;vs_feat(1:4)=1d0
call vs_select(1,1d0)
if(.not.vs_ratio_set.or.abs(vs_ratio-10d0)>1d-6)error stop 'weighted fit failed'
! A perfect fallback has no negative residual. Later validation must
! reject a frozen fit that overshoots, even though it was initially kept.
do i=1,80
  vs_feat=0d0;vs_feat(1)=1d0
  call vs_accumulate(1,0d0,0d0,1d0)
enddo
call vs_solve(1,.true.)
call vs_select(1,1d0)
if(vs_ratio_set)error stop 'sign-aware validation failed to retain fallback'
end program
''', [('context.f90', modules),
      ('virtual_approximation.f', (TEMPLATE / 'virtual_approximation.f').read_text())])

    def test_separate_virtual_and_mixed_weights_fit_identical_targets(self):
        modules = '''module weight_lines
integer :: icontr=4,itype(4)=[14,14,14,15],amppos(4)=[1,2,2,1]
real(8) :: wgts(1,4)=0d0
end module
module mint_module
integer :: imode=0,ichan=1,VirtTrickSurrogate=2
logical :: use_poly_virtual=.false.
real(8) :: polyfit(0:1)=0d0,average_virtual(0:1,2)=0d0
end module
'''
        self.compile_and_run('''program check
use virt_surrogate
use weight_lines
use mint_module
implicit none
integer i,j
real(8) value,c,weight(4),target(4),expected(4),mixed,separate
call vs_reset
weight=0d0;target=0d0
! Labels vary within each feature group, so wrong physical weights
! change the fit coefficients. Some labels have cancelling residual
! orders: (V-A)_1=-4 and (V-A)_2=4, with V=A=8.
do i=1,80
  j=1+mod(i-1,4)
  value=8d0+real(mod(i/4,3)**2,8)
  c=real(1+mod(i/4,3),8)
  weight(j)=weight(j)+c*c
  target(j)=target(j)+c*c*value
  vs_feat=0d0;vs_feat(j)=1d0
  vs_vcomp=.true.;vs_featok=.true.;vs_virt=value;vs_born=1d0
  ! Mixed events contain the approximation and sampled residual.
  ichan=1;vs_fvirt=.25d0;vs_residual_order=0
  wgts(1,:)=[c*(value-12d0)/vs_fvirt,c/ vs_fvirt, &
             3d0*c/vs_fvirt,c*8d0]
  call virt_surrogate_fit_point
  ! Separate events contain only the residual. Two flavour/fold
  ! records for its saved order must both enter the normalization.
  ichan=2;vs_fvirt=1d0;vs_residual_order=2;vs_residual_reference=4d0
  wgts(1,:)=[c*(value-12d0),c,3d0*c,0d0]
  call virt_surrogate_fit_point
enddo
expected=target/weight
call vs_solve(1,.false.)
call vs_solve(2,.false.)
do j=1,4
  vs_feat=0d0;vs_feat(j)=1d0
  mixed=vs_value(1);separate=vs_value(2)
  if(abs(mixed-expected(j))>1d-6)error stop 'mixed physical fit weight'
  if(abs(separate-expected(j))>1d-6)error stop 'separate physical fit weight'
enddo
! When every residual order vanishes there is no normalization record.
! Such a point must be ignored, rather than dividing by zero.
vs_residual_reference=0d0
call virt_surrogate_fit_point
end program
''', [('context.f90', modules),
      ('virtual_approximation.f', (TEMPLATE / 'virtual_approximation.f').read_text())])

    def test_grid_roundtrip_disabled_mode_and_legacy_fallback(self):
        source = (TEMPLATE / 'mint_module.f90').read_text()
        routines = '\n'.join(re.search(
            r'^  subroutine ' + name + r'\b.*?^  end subroutine ' + name + r'\b',
            source, re.M | re.S).group(0)
            for name in ('write_grids_to_file', 'read_grids_from_file'))
        self.compile_and_run('''module fixture
use virt_surrogate
implicit none
integer :: VirtTrickSurrogate=2
integer,parameter :: nintervals=4,nintervals_virt=2,ndim=4,n_ord_virt=2,maxchannels=2,nintegrals=10
integer :: nchans=2,imode=0,ncalls0=100,itmax=7,nhits_in_grids(2)=[20,30]
real(8) :: xgrid(0:4,4,2),ymax(4,4,2),ave_virt(2,1,0:2,2)
real(8) :: ymax_virt(0:2),ans(10,0:2),unc(10,0:2),virtual_fraction(2),average_virtual(0:2,2)
logical :: use_poly_virtual=.false.
contains
''' + routines + '''
subroutine close_run_zero_res
error stop 'unexpected zero result'
end subroutine
end module
program check
use fixture
implicit none
integer i
xgrid=.3d0;ymax=3d0;ave_virt=4d0;ymax_virt=5d0;ans=6d0;unc=7d0
virtual_fraction=.125d0;average_virtual=9d0
open(unit=15,status='scratch',form='formatted')
write(15,*)'VSU',1,1,16,[(real(i,8)/3d0,i=1,16)]
write(15,*)'VSU',0,1,16,[(-real(i,8)/3d0,i=1,16)]
rewind(15);call vs_read_grids(15,2,.true.);close(15)
call write_grids_to_file
call vs_reset
imode=1
call read_grids_from_file
if(vs_nfeat/=16)error stop 'lost feature count'
vs_featok=.true.;vs_feat=0d0;vs_feat(1)=1d0
call vs_select(1,1d0)
if(.not.vs_ratio_set)error stop 'lost active surrogate'
call vs_select(2,1d0)
if(vs_ratio_set)error stop 'lost fallback selection'
do i=1,16
  vs_feat=0d0;vs_feat(i)=1d0
  if(abs(vs_value(1)-real(i,8)/3d0)>1d-14.or. &
     abs(vs_value(2)+real(i,8)/3d0)>1d-14)error stop 'lost coefficients'
enddo
if(any(ans(:,1:2)/=6d0).or.any(virtual_fraction/=.125d0))error stop 'native grid damaged'
! Disabled fits are consumed without being applied.
VirtTrickSurrogate=0
call read_grids_from_file
vs_featok=.true.;call vs_select(1,1d0)
if(vs_ratio_set)error stop 'disabled fit became active'
imode=0;call write_grids_to_file
! Older files have no saved fit and remain usable with the new default.
VirtTrickSurrogate=2;imode=1
call read_grids_from_file
vs_featok=.true.;vs_feat(1)=1d0;call vs_select(1,1d0)
if(vs_ratio_set.or.vs_value(1)/=0d0)error stop 'legacy grid acquired a fit'
end program
subroutine init_polyfit(a,b,c,d)
integer a,b,c,d
error stop 'unexpected polynomial initialization'
end subroutine
subroutine restore_polyfit(a)
integer a
end subroutine
subroutine save_polyfit(a)
integer a
end subroutine
subroutine do_polyfit
end subroutine
subroutine initplot
end subroutine
''')
