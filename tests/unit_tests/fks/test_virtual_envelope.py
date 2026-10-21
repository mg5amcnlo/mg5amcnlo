"""Residual proposals preserve the signed distribution and restart state."""

from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestVirtualEnvelope(unittest.TestCase):
    def run_fixture(self, routines, fixture):
        source = (ROOT / 'Template/NLO/SubProcesses/mint_module.f90').read_text()
        extracted = []
        for name in routines:
            extracted.append(re.search(
                r'^  subroutine ' + name + r'\b.*?^  end subroutine ' + name + r'\b',
                source, re.M | re.S).group(0))
        with tempfile.TemporaryDirectory(prefix='virtual_envelope_') as tmp:
            work = Path(tmp)
            (work / 'routines.inc').write_text('\n'.join(extracted) + '\n')
            (work / 'driver.f90').write_text(fixture)
            compiled = subprocess.run([
                shutil.which('gfortran'), '-O2', '-fcheck=all',
                '-ffpe-trap=invalid,zero,overflow', '-I', str(work),
                str(ROOT / 'Template/NLO/SubProcesses/virtual_surrogate.f90'),
                str(work / 'driver.f90'), '-o', str(work / 'check')],
                cwd=work, capture_output=True, text=True, timeout=60)
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            checked = subprocess.run([str(work / 'check')], cwd=work,
                                     capture_output=True, text=True, timeout=60)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
            return checked.stdout

    def test_signed_peaked_distribution_and_overflow_accounting(self):
        output = self.run_fixture([
            'update_virtual_envelopes', 'update_residual_envelope',
            'prepare_residual_envelope', 'residual_cell_bound',
            'get_residual_weighted_cell', 'get_virtual_candidate', 'check_upper_bound'], r'''
module fixture
 use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
 implicit none
 integer,parameter :: nintervals=32,ndimmax=2,ndim=2,maxchannels=1,nchans=1
 integer :: ichan=1,ifold(2)=1,ncell(2)=1,gen_counters(13)=0,target_case=1
 integer(kind=8) :: residual_proposal_attempts=0
 real(8) :: residual_max(32,2,1)=1d0,residual_cdf(32,2,1),rand(2)
 real(8) :: ymax_virt(0:1)=10251d0,residual_envelope_norm(1),upper_bound,f(5)=0d0
 real(8) :: residual_gen_weight=0d0,residual_gen_excess=0d0
 logical :: residual_envelope_available=.true.,residual_envelope_selected(1)=.false.
contains
 include 'routines.inc'
 real(8) function ran3(even)
  logical,intent(in) :: even
  call random_number(ran3)
 end function
 subroutine get_channel
  ichan=1
 end subroutine
 subroutine get_random_x_next_fold(x,vol,kfold)
  real(8) :: x(ndimmax),vol
  integer :: kfold(ndimmax)
  x=(ncell-1+rand)/nintervals
  vol=1d0
 end subroutine
 subroutine get_random_cell_flat(x,vol)
  real(8) :: x(ndimmax),vol
  integer :: d,kfold(ndimmax)
  do d=1,ndim
   ncell(d)=1+int(ran3(.false.)*(nintervals/ifold(d)))
   rand(d)=ran3(.false.)
  enddo
  kfold=1;call get_random_x_next_fold(x,vol,kfold)
  upper_bound=ymax_virt(ichan)
 end subroutine
 real(8) function target(cell)
  integer,intent(in) :: cell(2)
  target=1d0
  if(target_case==1)then
   if(cell(1)==1)target=target*201d0
   if(cell(2)==1)target=target*51d0
  else
   if(cell(1)<=3.and.cell(2)<=5)target=target+60d0
   if(cell(1)>=29.and.cell(2)>=27)target=target+35d0
   if(abs(cell(1)-cell(2))<=1.and.cell(1)>=10.and.cell(1)<=18)target=target+40d0
  endif
 end function
end module
program check
 use fixture
 implicit none
 integer :: i,j,k,accepted,first,second,seed_size
 integer,allocatable :: seeds(:)
 real(8) :: weight,bound,x(2),vol,signed_sum,relative_excess
 real(8) :: true_norm,total_weight,expected_sign,expected_first,expected_second,estimated_norm
 real(8) :: legacy_x(2),legacy_next
 logical :: found
 call random_seed(size=seed_size)
 allocate(seeds(seed_size));seeds=137;call random_seed(put=seeds)
 ! Every cell is covered, including the rare peak at their intersection.
 do j=nintervals,1,-1
  do i=nintervals,1,-1
   ncell=[i,j];call update_virtual_envelopes(target(ncell))
  enddo
 enddo
 if(ymax_virt(1)/=20502d0)error stop 'scalar comparator lost the observed peak'
 do j=1,nintervals
  do i=1,nintervals
   ncell=[i,j]
   bound=residual_max(i,1,1)*residual_max(j,2,1)
   if(bound<2d0*target(ncell)*(1d0-1d-12))error stop 'lost an observed bound'
  enddo
 enddo
 call prepare_residual_envelope
 if(.not.residual_envelope_selected(1))error stop 'failed to choose improved proposal'
 if(residual_envelope_norm(1)>ymax_virt(1)/20d0)error stop 'peaked proposal remains inefficient'
 accepted=0;first=0;second=0;signed_sum=0d0
 do k=1,250000
  call get_virtual_candidate(x,vol)
  if(any(x<0d0).or.any(x>1d0).or.vol/=1d0)error stop 'incorrect proposal coordinates'
  f(1)=target(ncell)
  call check_upper_bound(1,found)
  if(.not.found)cycle
  accepted=accepted+1
  if(ncell(1)==1)first=first+1
  if(ncell(2)==1)second=second+1
  if(x(1)<.5d0)then
   signed_sum=signed_sum+1d0
  else
   signed_sum=signed_sum-1d0
  endif
 enddo
 if(accepted<10000)error stop 'too few accepted virtual events'
 if(abs(dble(first)/accepted-201d0/232d0)>.006d0)error stop 'first marginal is biased'
 if(abs(dble(second)/accepted-51d0/82d0)>.006d0)error stop 'second marginal is biased'
 if(abs(signed_sum/accepted-200d0/232d0)>.006d0)error stop 'signed distribution is biased'
 if(gen_counters(8)/=0.or.residual_gen_excess/=0d0)error stop 'speed came from bound violations'
 ! Diagonal and separated rectangular peaks cannot factorize. Check the
 ! pointwise-minimum proposal against exact enumeration, including signs.
 target_case=2;residual_max=1d0;ymax_virt=0d0
 do j=nintervals,1,-1
  do i=nintervals,1,-1
   ncell=[i,j];call update_virtual_envelopes(target(ncell))
  enddo
 enddo
 call prepare_residual_envelope
 true_norm=0d0;total_weight=0d0;expected_sign=0d0;expected_first=0d0;expected_second=0d0
 do j=1,nintervals
  do i=1,nintervals
   ncell=[i,j];weight=target(ncell);call residual_cell_bound(bound)
   true_norm=true_norm+min(bound,ymax_virt(1))/nintervals**2
   total_weight=total_weight+weight
   expected_sign=expected_sign+merge(weight,-weight,i>=j)
   if(i<=3)expected_first=expected_first+weight
   if(j>=27)expected_second=expected_second+weight
  enddo
 enddo
 residual_proposal_attempts=0;residual_gen_weight=0d0;residual_gen_excess=0d0;gen_counters=0
 accepted=0;first=0;second=0;signed_sum=0d0
 do k=1,250000
  call get_virtual_candidate(x,vol)
  f(1)=target(ncell);call check_upper_bound(1,found)
  if(.not.found)cycle
  accepted=accepted+1
  if(ncell(1)<=3)first=first+1
  if(ncell(2)>=27)second=second+1
  signed_sum=signed_sum+merge(1d0,-1d0,ncell(1)>=ncell(2))
 enddo
 estimated_norm=residual_envelope_norm(1)*250000d0/residual_proposal_attempts
 if(residual_proposal_attempts<=250000)error stop 'minimum envelope never applied cheap rejection'
 if(abs(estimated_norm/true_norm-1d0)>.01d0)error stop 'incorrect cheap rejection normalization'
 if(abs(dble(first)/accepted-expected_first/total_weight)>.009d0)error stop 'correlated first marginal'
 if(abs(dble(second)/accepted-expected_second/total_weight)>.009d0)error stop 'correlated second marginal'
 if(abs(signed_sum/accepted-expected_sign/total_weight)>.009d0)error stop 'correlated signed distribution'
 if(gen_counters(8)/=0.or.residual_gen_excess/=0d0)error stop 'correlated target exceeds bounds'
 ! Diagnose clipped acceptance explicitly, weighted by the actual proposal.
 gen_counters=0;residual_gen_weight=0d0;residual_gen_excess=0d0
 upper_bound=2d0;f(1)=6d0;call check_upper_bound(1,found)
 relative_excess=residual_gen_excess/residual_gen_weight
 if(gen_counters(8)/=1.or.abs(relative_excess-2d0/3d0)>1d-14)error stop 'overflow accounting'
 residual_max=1d6
 call prepare_residual_envelope
 if(residual_envelope_selected(1))error stop 'failed to retain cheaper scalar proposal'
 ! Folded dimensions have fewer proposal cells. Inactive serialized padding
 ! must neither enter the normalization nor acquire sampling probability.
 ifold=[2,4];residual_max=0d0
 residual_max(1:16,1,1)=2d0;residual_max(1:8,2,1)=3d0
 call prepare_residual_envelope
 if(abs(residual_envelope_norm(1)-6d0)>1d-14)error stop 'folded normalization'
 do k=1,1000
  call get_residual_weighted_cell(x,vol)
  if(ncell(1)>16.or.ncell(2)>8.or.abs(upper_bound-6d0)>1d-14)error stop 'sampled inactive folded cell'
 enddo
 ! The product can be tiny, or have cancelling extreme factors, without
 ! a zero proposal bound, an invalid CDF, or an endless pre-veto loop.
 ifold=1;residual_max=1d-250;ymax_virt=2d0*tiny(1d0)
 call prepare_residual_envelope
 call get_virtual_candidate(x,vol)
 if(.not.ieee_is_finite(upper_bound).or.upper_bound<=0d0)error stop 'tiny envelope lost support'
 residual_max(:,1,:)=1d-250;residual_max(:,2,:)=1d250;ymax_virt=2d0
 call prepare_residual_envelope
 call get_virtual_candidate(x,vol)
 if(abs(upper_bound-1d0)>1d-12)error stop 'intermediate product underflowed'
 residual_envelope_available=.false.;ymax_virt=3d0
 call update_virtual_envelopes(100d0)
 if(ymax_virt(1)/=6d0)error stop 'changed legacy scalar-envelope policy'
 ! With pooling disabled, wrapping the native flat selector consumes exactly
 ! the same random draws and returns the same cells, coordinates and bound.
 call random_seed(put=seeds)
 call get_random_cell_flat(legacy_x,vol);legacy_next=ran3(.false.)
 call random_seed(put=seeds)
 call prepare_residual_envelope
 call get_virtual_candidate(x,vol)
 if(any(x/=legacy_x).or.upper_bound/=6d0)error stop 'changed legacy flat proposal'
 if(ran3(.false.)/=legacy_next)error stop 'changed legacy random-number sequence'
 print *,'PASS signed residual distribution, efficiency and overflow accounting',accepted
end program
''')
        self.assertIn('PASS signed residual', output)

    def test_restart_roundtrip_and_legacy_scalar_fallback(self):
        output = self.run_fixture(['write_grids_to_file', 'read_grids_from_file'], r'''
module fixture
 use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
 use virt_surrogate
 implicit none
 integer :: VirtTrickSurrogate=0
 integer,parameter :: nintervals=4,nintervals_virt=2,ndim=4,n_ord_virt=2,maxchannels=2,nintegrals=10
 integer :: nchans=2,imode=1,ncalls0=100,itmax=7,nhits_in_grids(2)=[20,30]
 real(8) :: xgrid(0:4,4,2),ymax(4,4,2),ave_virt(2,1,0:2,2)
 real(8) :: ymax_virt(0:2),ans(10,0:2),unc(10,0:2),virtual_fraction(2),average_virtual(0:2,2)
 real(8) :: residual_max(4,4,2)
 real(8) :: residual_kappa(2)=0d0
 real(8) :: residual_velocity_kappa(2)=0d0
 logical :: use_poly_virtual=.false.,use_pooled_virtual=.true.,residual_envelope_available=.false.
contains
 include 'routines.inc'
 subroutine pooled_virtual_init(orders,coordinates)
  integer :: orders,coordinates
  if(orders/=2.or.coordinates/=1)error stop 'wrong model initialization'
 end subroutine
 subroutine close_run_zero_res
  error stop 'unexpected zero result'
 end subroutine
end module
program check
 use fixture
 implicit none
 integer :: i,j,k,count,ios
 real(8) :: saved(4,4,2)
 character(len=4096) :: lines(100)
 xgrid=.3d0;ymax=3d0;ave_virt=4d0;ymax_virt=5d0;ans=6d0;unc=7d0
 virtual_fraction=.125d0;average_virtual=9d0
 ! Older grids have no VEP marker and retain the scalar algorithm.
 call write_grids_to_file
 imode=2;residual_envelope_available=.true.;residual_max=-1d0
 call read_grids_from_file
 if(residual_envelope_available)error stop 'legacy grid acquired a proposal'
 if(any(ans(:,1:2)/=6d0).or.any(ymax_virt(1:2)/=5d0))error stop 'legacy grid damaged'
 do k=1,2
  do j=1,4
   do i=1,4
    residual_max(i,j,k)=100*k+10*j+i
   enddo
  enddo
 enddo
 saved=residual_max
 residual_kappa=[.3d0,.7d0]
 residual_velocity_kappa=[.1d0,.2d0]
 residual_envelope_available=.true.;imode=1
 call write_grids_to_file
 imode=2;residual_max=0d0;residual_envelope_available=.false.
 call read_grids_from_file
 if(.not.residual_envelope_available.or.any(residual_max/=saved))error stop 'lost residual envelope'
 if(any(residual_kappa/=[.3d0,.7d0]))error stop 'lost Born carrier bounds'
 if(any(residual_velocity_kappa/=[.1d0,.2d0]))error stop 'lost velocity carrier bounds'
 if(any(ans(:,1:2)/=6d0).or.any(virtual_fraction/=.125d0))error stop 'new grid damaged native fields'
 ! Version 2 keeps its product proposal but has no evidence for a Born veto.
 open(unit=15,file='mint_grids',status='old')
 count=0
 do
  read(15,'(a)',iostat=ios)lines(count+1)
  if(ios/=0)exit
  count=count+1
 enddo
 close(15)
 ! Version 3 retains its original plain-Born denominator. It must not gain
 ! a velocity bound merely because the current writer uses version 4.
 open(unit=15,file='mint_grids',status='replace')
 do i=1,count-4
  write(15,'(a)')trim(lines(i))
 enddo
 do i=1,2
  write(15,*)'MAX',residual_kappa(i)
 enddo
 write(15,'(a)')'PVP 1','VEP 3'
 close(15)
 call read_grids_from_file
 if(any(residual_kappa/=[.3d0,.7d0]))error stop 'version-3 Born bound changed'
 if(any(residual_velocity_kappa/=0d0))error stop 'version-3 denominator silently changed'
 open(unit=15,file='mint_grids',status='replace')
 do i=1,count-4
  write(15,'(a)')trim(lines(i))
 enddo
 write(15,'(a)')'PVP 1','VEP 2'
 close(15)
 call read_grids_from_file
 if(any(residual_kappa/=0d0).or.any(residual_max/=saved))error stop 'version-2 fallback changed'
 ! Nonpooled grids store the per-channel fit after their fallback grids.
 use_pooled_virtual=.false.;VirtTrickSurrogate=2;imode=0
 residual_envelope_available=.false.
 open(unit=15,status='scratch',form='formatted')
 write(15,*)'VSU',1,1,16,[(dble(i)/3d0,i=1,16)]
 write(15,*)'VSU',0,1,16,[(-dble(i)/3d0,i=1,16)]
 rewind(15);call vs_read_grids(15,2,.true.);close(15)
 call write_grids_to_file
 call vs_reset
 imode=1
 call read_grids_from_file
 if(vs_nfeat/=16)error stop 'lost fitted feature count'
 vs_featok=.true.;vs_feat=0d0;vs_feat(1)=1d0
 call vs_select(1,1d0)
 if(.not.vs_ratio_set)error stop 'lost active surrogate'
 call vs_select(2,1d0)
 if(vs_ratio_set)error stop 'lost fallback selection'
 do i=1,16
  vs_feat=0d0;vs_feat(i)=1d0
  if(abs(vs_value(1)-dble(i)/3d0)>1d-14.or. &
       abs(vs_value(2)+dble(i)/3d0)>1d-14)error stop 'lost fitted coefficients'
 enddo
 ! An explicit mode 0 consumes saved fit rows but uses the fallback.
 VirtTrickSurrogate=0
 call read_grids_from_file
 vs_featok=.true.;call vs_select(1,1d0)
 if(vs_ratio_set)error stop 'disabled surrogate restored active fit'
 imode=0;call write_grids_to_file
 ! Old files without VSU rows remain readable with the new default.
 VirtTrickSurrogate=2;imode=1
 call read_grids_from_file
 vs_featok=.true.;vs_feat(1)=1d0;call vs_select(1,1d0)
 if(vs_ratio_set.or.vs_value(1)/=0d0)error stop 'old grid acquired a fitted surrogate'
 print *,'PASS residual grid roundtrip and legacy fallback'
end program
subroutine init_polyfit(a,b,c,d)
 integer :: a,b,c,d
 error stop 'unexpected polynomial initialization'
end subroutine
subroutine restore_polyfit(a)
 integer :: a
end subroutine
subroutine save_polyfit(a)
 integer :: a
end subroutine
subroutine do_polyfit
end subroutine
subroutine initplot
end subroutine
''')
        self.assertIn('PASS residual grid roundtrip', output)

    def test_velocity_screen_handles_threshold_peak_and_signed_distribution(self):
        output = self.run_fixture(['observe_virtual_ratio', 'screen_virtual_carrier',
                                   'check_upper_bound'], r'''
module fixture
 use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
 implicit none
 integer :: imode=1,ichan=1,gen_counters(13)=0
 integer(kind=8) :: virtual_carrier_attempts=0,virtual_carrier_vetoes=0
 real(8) :: residual_kappa(1)=0d0,residual_velocity_kappa(1)=0d0
 real(8) :: pooled_virtual_velocity_factor=1d0,upper_bound,f(5)=0d0
 real(8) :: residual_gen_weight=0d0,residual_gen_excess=0d0
 logical :: residual_envelope_available=.true.,pooled_virtual_velocity_valid=.true.
contains
 include 'routines.inc'
 real(8) function ran3(even)
  logical,intent(in) :: even
  call random_number(ran3)
 end function
end module
program check
 use fixture
 implicit none
 integer :: k,loops,accepted,low,seed_size
 integer,allocatable :: seeds(:),before(:),after(:)
 real(8) :: x,born,signed_sum
 logical :: pass,found
 call random_seed(size=seed_size);allocate(seeds(seed_size),before(seed_size),after(seed_size))
 seeds=19437;call random_seed(put=seeds)
 ! Born vanishes like velocity while the finite virtual stays nonzero.
 ! Learn both carriers from the same exact observations, without RNG.
 call random_seed(get=before)
 do k=0,30
  x=10d0**(-6d0+k/5d0);born=sqrt(x)
  pooled_virtual_velocity_factor=1d0+1d0/born
  call observe_virtual_ratio([1d0],[born])
 enddo
 call random_seed(get=after)
 if(any(before/=after))error stop 'velocity learning consumed random draws'
 if(abs(residual_kappa(1)-2000d0)>1d-10)error stop 'plain threshold bound changed'
 if(abs(residual_velocity_kappa(1)-2d0/1.001d0)>1d-13)error stop 'velocity bound normalization'
 imode=2;loops=0;accepted=0;low=0;signed_sum=0d0
 do k=1,500000
  x=max(ran3(.false.),tiny(1d0));born=sqrt(x)
  pooled_virtual_velocity_factor=1d0+1d0/born;upper_bound=100d0
  call screen_virtual_carrier(born,0d0,pass)
  if(.not.pass)cycle
  loops=loops+1;f(1)=1d0
  call check_upper_bound(1,found)
  if(.not.found)cycle
  accepted=accepted+1
  if(x<.2d0)low=low+1
  signed_sum=signed_sum+merge(1d0,-1d0,x<.5d0)
 enddo
 if(loops>19000.or.accepted<4300)error stop 'velocity carrier did not reduce threshold cost'
 if(abs(dble(low)/accepted-.2d0)>.025d0)error stop 'threshold distribution biased'
 if(abs(signed_sum/accepted)>.05d0)error stop 'threshold signs biased'
 if(gen_counters(8)/=0.or.residual_gen_excess/=0d0)error stop 'velocity speed came from lost support'
 ! Undefined geometry requests the exact residual against the outer bound,
 ! with no carrier rejection or extra random draw.
 pooled_virtual_velocity_valid=.false.;upper_bound=100d0
 call random_seed(get=before)
 call screen_virtual_carrier(.001d0,0d0,pass)
 call random_seed(get=after)
 if(.not.pass.or.any(before/=after).or.upper_bound/=100d0)error stop 'invalid velocity fallback'
 print *,'PASS threshold velocity carrier and signed target',loops,accepted
end program
''')
        self.assertIn('PASS threshold velocity carrier', output)

    def test_born_screen_preserves_signed_target_and_learns_without_physics(self):
        output = self.run_fixture(['observe_virtual_ratio', 'screen_virtual_carrier',
                                   'check_upper_bound'], r'''
module fixture
 use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
 implicit none
 integer :: imode=1,ichan=1,gen_counters(13)=0
 integer(kind=8) :: virtual_carrier_attempts=0,virtual_carrier_vetoes=0
 real(8) :: residual_kappa(1)=0d0,upper_bound,f(5)=0d0
 real(8) :: residual_velocity_kappa(1)=0d0,pooled_virtual_velocity_factor=1d0
 real(8) :: residual_gen_weight=0d0,residual_gen_excess=0d0
 logical :: residual_envelope_available=.true.
 logical :: pooled_virtual_velocity_valid=.true.
contains
 include 'routines.inc'
 real(8) function ran3(even)
  logical,intent(in) :: even
  call random_number(ran3)
 end function
end module
program check
 use fixture
 implicit none
 integer :: k,loops,accepted,peak,seed_size
 integer,allocatable :: seeds(:),before(:),after(:)
 real(8) :: x,carrier,signed_sum,expected_peak,expected_sign
 logical :: pass,found
 call random_seed(size=seed_size);allocate(seeds(seed_size),before(seed_size),after(seed_size))
 seeds=937;call random_seed(put=seeds)
 call random_seed(get=before)
 call observe_virtual_ratio([2d0,-3d0],[1d0,-2d0])
 call random_seed(get=after)
 if(any(before/=after))error stop 'learning consumed random draws'
 if(abs(residual_kappa(1)-2d0)>1d-14)error stop 'signed Born cancellation entered carrier'
 imode=0;call observe_virtual_ratio([200d0,-300d0],[1d0,-2d0])
 imode=2;call observe_virtual_ratio([200d0,-300d0],[1d0,-2d0])
 if(abs(residual_kappa(1)-2d0)>1d-14)error stop 'ratio learned outside frozen envelope stage'
 imode=1;residual_envelope_available=.false.
 call observe_virtual_ratio([200d0,-300d0],[1d0,-2d0])
 if(abs(residual_kappa(1)-2d0)>1d-14)error stop 'disabled feature learned ratio'
 residual_envelope_available=.true.
 call observe_virtual_ratio([2d0,-3d0],[0d0,0d0])
 if(abs(residual_kappa(1)-2d0)>1d-14)error stop 'zero carrier generated invalid ratio'
 ! Uniform outer envelope. A narrow Born peak is detected before any exact
 ! target evaluation. The accepted distribution includes its signed weight.
 loops=0;accepted=0;peak=0;signed_sum=0d0
 do k=1,500000
  x=ran3(.false.);carrier=1d0
  if(x<.01d0)carrier=100d0
  upper_bound=100d0
  call screen_virtual_carrier(carrier,0d0,pass)
  if(.not.pass)cycle
  loops=loops+1
  f(1)=carrier*(.2d0+.7d0*x)
  call check_upper_bound(1,found)
  if(.not.found)cycle
  accepted=accepted+1
  if(x<.01d0)peak=peak+1
  signed_sum=signed_sum+merge(1d0,-1d0,x<.5d0)
 enddo
 expected_peak=.2035d0/.751465d0
 expected_sign=(.388965d0-.3625d0)/.751465d0
 if(loops>17500.or.accepted<3200)error stop 'Born screening did not save exact evaluations'
 if(abs(dble(peak)/accepted-expected_peak)>.025d0)error stop 'Born peak distribution biased'
 if(abs(signed_sum/accepted-expected_sign)>.045d0)error stop 'signed screened distribution biased'
 if(gen_counters(8)/=0.or.residual_gen_excess/=0d0)error stop 'speed came from clipping target'
 ! Cancelling extreme factors must neither overflow nor erase support.
 residual_velocity_kappa=0d0
 residual_kappa=1d-250;upper_bound=2d0
 call screen_virtual_carrier(1d250,0d0,pass)
 if(pass.and.abs(upper_bound-1d0)>1d-12)error stop 'tiny kappa screening bound'
 residual_kappa=1d250;upper_bound=2d0
 call screen_virtual_carrier(1d-250,0d0,pass)
 if(pass.and.abs(upper_bound-1d0)>1d-12)error stop 'large kappa screening bound'
 residual_kappa=0d0;upper_bound=2d0
 call random_seed(get=before)
 call screen_virtual_carrier(1d0,0d0,pass)
 call random_seed(get=after)
 if(.not.pass.or.any(before/=after).or.upper_bound/=2d0)error stop 'no-evidence fallback changed sampling'
 print *,'PASS signed Born screen, bounded cost and stage-only learning',loops,accepted
end program
''')
        self.assertIn('PASS signed Born screen', output)
