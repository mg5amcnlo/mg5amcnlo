"""Scalar virtual fits apply only to one matching Born normalization."""

from pathlib import Path
import re
import shutil
import unittest

from tests.unit_tests.fks import test_virtual_surrogate as support


ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = ROOT / 'Template/NLO/SubProcesses'


def production_fixture():
    """Compile the actual FKS choice/subtraction blocks with small input stubs."""
    source = (TEMPLATE / 'fks_singular.f').read_text()
    choice = source.split('c The scalar fit is valid only for one virtual Born order normalized by', 1)[1]
    choice = choice[:choice.index('      if (fold.eq.0) then\n         vs_vcomp=.false.')]
    choice = 'c The scalar fit is valid only for one virtual Born order normalized by'+choice
    ratio = re.search(r'^            virtual_over_born=0d0\n.*?(?=^            virt_wgt = 0d0)', source, re.M | re.S).group()
    capture = re.search(r'^            if \(use_surrogate\) then\n.*?^            endif\n(?=            virtual_over_born)', source, re.M | re.S).group()
    residual = re.search(r'^            virt_wgt = 0d0\n.*?(?=^c Separate virtual events)', source, re.M | re.S).group()
    fraction = re.search(r"^            if \(abrv.ne.'virt'\) then\n.*?^            endif", source, re.M | re.S).group()
    approximation = re.search(r"^      if \(abrv\(1:4\).ne.'virt'\) then\n.*?(?=^c eq.)", source, re.M | re.S).group()
    return '''      subroutine choose_fit
      use point_fixture
      implicit none
      use_surrogate=VirtTrickSurrogate.ge.1
''' + choice + '''      end
      subroutine evaluate_virtual
      use point_fixture
      implicit none
      integer iamp
      vs_vcomp=.false.
      vs_born=born_wgt
      vs_fvirt=virtual_fraction(ichan)
      virt_wgt=sum(amp_split_finite_ML)
      amp_split_virt=amp_split_finite_ML
      amp_split_avv=0d0
''' + capture + ratio + residual + fraction + '\n' + approximation + '''      end
'''


CONTEXT = '''module weight_lines
integer :: icontr=4,itype(4)=[14,14,15,15],amppos(4)=[1,2,1,2]
real(8) :: wgts(1,4)=0d0
end module
module mint_module
integer :: imode=0,ichan=1,VirtTrickSurrogate=2
logical :: use_poly_virtual=.false.
real(8) :: polyfit(0:2)=0d0,average_virtual(0:2,1)=0d0,virtual_fraction(1)=[.25d0]
end module
module point_fixture
use virt_surrogate
use mint_module
implicit none
integer,parameter :: amp_split_size=2
integer :: fold=0
logical :: use_surrogate=.true.
character(len=4) :: abrv='all '
real(8) :: p_born(0:3,4),QES2=100d0,born_wgt=2d0,virt_wgt,virtual_over_born,avv_wgt
real(8) :: amp_split_born_for_virt(2),amp_split_finite_ML(2)=[5d0,3d0]
real(8) :: amp_split_virt(2),amp_split_avv(2),ave_ratio(0:2)
contains
subroutine load_fit
call vs_reset
open(15,status='scratch')
write(15,*)'VSU',1,1,4,[3d0,0d0,0d0,0d0]
rewind(15);call vs_read_grids(15,1,.true.);close(15)
p_born(:,1)=[50d0,0d0,0d0,50d0]
p_born(:,2)=[50d0,0d0,0d0,-50d0]
p_born(:,3)=[50d0,50d0,0d0,0d0]
p_born(:,4)=[50d0,-50d0,0d0,0d0]
end subroutine
subroutine fresh_links
call vs_add_link(p_born,QES2,3,4,4,1d0)
end subroutine
subroutine save_weights
use weight_lines
wgts(1,1:2)=2d0*amp_split_virt
wgts(1,3:4)=2d0*amp_split_avv
end subroutine
end module
'''


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestMixedVirtualSurrogate(unittest.TestCase):
    def compile_and_run(self, program, fixture=False):
        extra = [('context.f90', CONTEXT)]
        if fixture:
            extra.append(('fks_selection.f', production_fixture()))
        extra.append(('virtual_approximation.f', (TEMPLATE / 'virtual_approximation.f').read_text()))
        return support.TestVirtualSurrogate.compile_and_run(self, program, extra)

    def test_single_matching_order_keeps_features_and_trains_a_usable_fit(self):
        self.compile_and_run('''program check
use point_fixture
use weight_lines
implicit none
integer i,j
real(8) value,c
call vs_reset
amp_split_born_for_virt=[1d0,0d0]
do i=1,80
 j=1+mod(i-1,4)
 vs_feat=0d0;vs_feat(j)=1d0
 value=real(j,8);c=real(1+mod(i,3),8)
 vs_vcomp=.true.;vs_featok=.true.;vs_virt=value;vs_born=1d0;vs_fvirt=.25d0
 if(.not.vs_single_born_order(amp_split_born_for_virt,1d0))error stop 'single order rejected'
 if(.not.vs_featok.or..not.vs_vcomp)error stop 'eligible point cleared'
 wgts(1,:)=[c*value/vs_fvirt,0d0,0d0,0d0]
 call virt_surrogate_fit_point
enddo
call vs_solve(1,.false.)
vs_feat=0d0;vs_feat(1:4)=1d0
call vs_select(1,1d0)
if(.not.vs_ratio_set.or.abs(vs_ratio-10d0)>1d-6)error stop 'single-order fit lost'
if(.not.vs_single_born_order([0d0,-2d0],-2d0))error stop 'signed matching order rejected'
end program
''')

    def test_mixed_zero_and_mismatched_orders_clear_point_but_keep_snapshot(self):
        self.compile_and_run('''program check
use point_fixture
implicit none
integer i,j
real(8) cases(2,4)
cases(:,1)=[2d0,2d0]
cases(:,2)=[0d0,0d0]
cases(:,3)=[1d0,0d0]
cases(:,4)=[2d0,1d-100]
call load_fit
average_virtual(:,1)=[2d0,1d0,2d0]
do j=1,4
 amp_split_born_for_virt=cases(:,j)
 do i=1,80
  fold=mod(i,2)
  call fresh_links
  call vs_features(p_born,born_wgt,QES2,2)
  vs_vcomp=.true.;vs_virt=99d0;vs_residual_order=2;vs_residual_reference=7d0
  vs_ratio_set=.true.;vs_ratio=123d0
  call choose_fit
  if(use_surrogate.or.vs_ratio_set.or.vs_vcomp.or.vs_featok)error stop 'ineligible snapshot applied'
  if(any(ave_ratio(1:2)/=[1d0,2d0]))error stop 'fold lost distinct fallback orders'
  if(vs_residual_order/=0.or.vs_residual_reference/=0d0)error stop 'stale separate-event state'
  call evaluate_virtual
  call save_weights
  call virt_surrogate_fit_point
 enddo
 call vs_solve(1,.false.)
 vs_feat=0d0;vs_feat(1)=1d0
 if(abs(vs_value(1)-3d0)>1d-14)error stop 'mixed observations changed saved fit'
 ! No old colour links can be reused by a later eligible point.
 call vs_features(p_born,born_wgt,QES2,2)
 if(vs_featok)error stop 'ineligible point kept colour links'
enddo
amp_split_born_for_virt=[2d0,0d0]
call fresh_links
call choose_fit
if(.not.use_surrogate.or..not.vs_ratio_set)error stop 'saved fit disabled globally'
if(abs(vs_ratio-3d0)>1d-14)error stop 'saved coefficients lost'
end program
''', fixture=True)

    def test_mixed_orders_use_native_grid_or_polynomial_and_unbiased_weights(self):
        self.compile_and_run('''program check
use point_fixture
implicit none
integer mode,i
real(8) scalar_prediction,old_ratio,old_residual,approximation(2),exact(2),expected(2)
call load_fit
amp_split_born_for_virt=[2d0,2d0]
exact=amp_split_finite_ML
call fresh_links
call vs_features(p_born,born_wgt,QES2,2)
call vs_select(1,born_wgt)
scalar_prediction=vs_value(1)
old_ratio=vs_ratio
old_residual=sum(exact-old_ratio*amp_split_born_for_virt)
! Validation compared V-A=2, but broadcasting the scalar ratio gives
! V-2A=-4. This loaded snapshot must therefore not be applied.
if(abs(sum(exact)-scalar_prediction-2d0)>1d-14)error stop 'scalar criterion setup'
if(abs(old_residual+4d0)>1d-14)error stop 'broadcast mismatch not reproduced'
do mode=1,2
 use_poly_virtual=mode.eq.2
 average_virtual(:,1)=[2d0,1d0,2d0]
 polyfit=[-3d0,4d0,-1d0]
 call fresh_links
 call choose_fit
 if(use_surrogate.or.vs_ratio_set)error stop 'mixed scalar ratio selected'
 expected=average_virtual(1:2,1)
 if(use_poly_virtual)expected=polyfit(1:2)
 if(any(ave_ratio(1:2)/=expected))error stop 'native per-order fallback lost'
 call evaluate_virtual
 approximation=expected*amp_split_born_for_virt
 if(any(abs(amp_split_avv-approximation)>1d-14))error stop 'type15 approximation changed'
 if(any(abs(amp_split_virt-(exact-approximation)/virtual_fraction(1))>1d-14)) &
      error stop 'type14 residual/fraction changed'
 ! Mix the sampled virtual event with the nonvirtual event: the type14
 ! residual and type15 approximation cancel order by order in expectation.
 do i=1,2
  if(abs(virtual_fraction(1)*(amp_split_virt(i)+amp_split_avv(i)) &
      +(1d0-virtual_fraction(1))*amp_split_avv(i)-exact(i))>1d-14) &
      error stop 'virtual sampling became biased'
 enddo
enddo
end program
''', fixture=True)

    def test_zero_born_fallback_ratio_is_safe_with_floating_point_traps(self):
        self.compile_and_run('''program check
use point_fixture
implicit none
call load_fit
born_wgt=0d0
amp_split_born_for_virt=0d0
call fresh_links
call choose_fit
call evaluate_virtual
if(use_surrogate.or.vs_ratio_set.or.vs_vcomp)error stop 'zero Born retained fit'
if(virtual_over_born/=0d0)error stop 'zero Born ratio not guarded'
if(any(amp_split_avv/=0d0))error stop 'zero Born approximation nonzero'
if(any(abs(amp_split_virt-amp_split_finite_ML/virtual_fraction(1))>1d-14)) &
     error stop 'zero Born exact virtual lost'
end program
''', fixture=True)
