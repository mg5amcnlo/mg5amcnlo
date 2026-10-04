"""Exercise the AmpliCol MC@NLO stage protocol with analytic stub physics."""

import pathlib
import shutil
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'Template' / 'NLO' / 'SubProcesses'


STUB = r'''
module mint_module
  implicit none
  integer,parameter :: ndimmax=4,nintegrals=6
  integer :: ndim=2,ifold(ndimmax)=1,imode=0,itmax=4,ncalls0=4000,iconfig=7,born_spread_phase=0
  logical :: new_point=.false.,only_virt=.false.,born_spread_active=.false.,born_spread_ready=.false.
  double precision :: virtual_fraction(1)=0.5d0,accuracy=0.03d0
  double precision :: ans(nintegrals,0:1)=0d0,unc(nintegrals,0:1)=0d0
  double precision :: written_absolute_error=-1d0
contains
  subroutine nlops_init_auxiliary(reset)
    logical,intent(in) :: reset
  end subroutine
  subroutine nlops_prepare_point(x)
    double precision,intent(in) :: x(ndimmax)
  end subroutine
  subroutine nlops_next_fold(folds,iret)
    integer,intent(inout) :: folds(ndimmax)
    integer,intent(out) :: iret
    integer :: i
    iret=0
    do i=ndim,1,-1
       if (folds(i).lt.ifold(i)) then
          folds(i)=folds(i)+1
          return
       endif
       folds(i)=1
    enddo
    iret=1
  end subroutine
  subroutine nlops_train_virtual(x,values)
    double precision,intent(in) :: x(ndimmax),values(nintegrals)
  end subroutine
  subroutine nlops_update_auxiliary(errors)
    double precision,intent(in) :: errors(nintegrals)
    integer :: unit
    open(newunit=unit,file='grid.MC_integer',status='replace')
    write(unit,'(a)') 'STUB_INTEGER_STATE'
    close(unit)
  end subroutine
  subroutine calibrate_born_spreading(fun,sample)
    double precision,external :: fun
    external :: sample
    error stop 'unexpected stub Born calibration'
  end subroutine
  subroutine born_spread_write_table
  end subroutine
  subroutine reset_MC_grid
  end subroutine
  subroutine nlops_write_results(absolute_error)
    double precision,intent(in) :: absolute_error
    written_absolute_error=absolute_error
  end subroutine
  subroutine nlops_save_auxiliary(unit)
    integer,intent(in) :: unit
    write(unit,*) 'STUB_AUX ',virtual_fraction
  end subroutine
  subroutine nlops_load_auxiliary(unit)
    integer,intent(in) :: unit
    character(len=16) :: marker
    read(unit,*) marker,virtual_fraction
    if (marker.ne.'STUB_AUX') error stop 'auxiliary checkpoint misaligned'
  end subroutine
end module mint_module

module stub_state
  implicit none
  integer(kind=8) :: rng_state=1234567_8,ncalls=0_8,nopen=0_8,nfold=0_8,nclose=0_8
  double precision :: nv_sum=0d0,v_sum=0d0,last_sign=1d0
  logical :: evaluate_virtual=.false.,zero_target=.false.
  integer :: survey_points=0
  double precision :: survey_sum(4)=0d0,survey_square(4)=0d0
end module stub_state

double precision function ran2()
  use stub_state,only: rng_state
  implicit none
  rng_state=mod(48271_8*rng_state,2147483647_8)
  ran2=dble(rng_state)/2147483647d0
end function ran2

double precision function analytic_sigint(x,vol,ifl,values)
  use mint_module
  use stub_state
  implicit none
  double precision,intent(in) :: x(ndimmax),vol
  integer,intent(in) :: ifl
  double precision,intent(out) :: values(nintegrals)
  double precision :: base,ran2,survey_target
  integer :: survey_iteration
  character(len=4) :: abrv
  common /to_abrv/ abrv
  external :: ran2
  ncalls=ncalls+1_8
  values=0d0
  analytic_sigint=0d0
  if (ifl.eq.0) then
     nopen=nopen+1_8
     nv_sum=0d0
     v_sum=0d0
     evaluate_virtual=abrv.eq.'virt'
     if (abrv.eq.'all') evaluate_virtual=ran2().lt.virtual_fraction(1)
  elseif (ifl.eq.1) then
     nfold=nfold+1_8
  elseif (ifl.eq.2) then
     nclose=nclose+1_8
     values(2)=nv_sum+v_sum
     values(3)=v_sum
     values(5)=abs(v_sum)
     if (imode.eq.1.and..not.only_virt) then
        values(1)=abs(nv_sum)
     else
        values(1)=abs(nv_sum+v_sum)
     endif
     if (imode.eq.1) then
        survey_points=survey_points+1
        survey_iteration=(survey_points-1)/4000+1
        survey_target=values(1)
        if (.not.only_virt) survey_target=survey_target+values(5)
        survey_sum(survey_iteration)=survey_sum(survey_iteration)+survey_target
        survey_square(survey_iteration)=survey_square(survey_iteration)+survey_target**2
     endif
     last_sign=sign(1d0,values(2))
     return
  else
     error stop 'unexpected callback ifl'
  endif
  if (zero_target) return
  base=(0.1d0+exp(-10d0*x(1)))*(1d0+x(2))*vol
  if (abrv.ne.'virt') nv_sum=nv_sum+2d0*base
  if (evaluate_virtual) then
     if (abrv.eq.'virt') then
        v_sum=v_sum-0.5d0*base
     else
        v_sum=v_sum-0.5d0*base/virtual_fraction(1)
     endif
  endif
end function analytic_sigint
'''


DRIVER = r'''
program test_adapter
  use mint_module
  use ampli_mint_adapter
  use stub_state
  implicit none
  double precision,external :: analytic_sigint
  double precision :: expected,base,weight,scale,aqed,aqcd,signsum,saved_absolute_error
  double precision,allocatable :: weights(:)
  integer :: ini_fin_fks,unit,iu,i,candidates,quota,nup,idprup,kept,ios,negatives
  integer(kind=8) :: opens_before,folds_before,closes_before
  logical :: to_write,done
  character(len=4) :: abrv
  character(len=40) :: task
  character(len=300) :: line
  common /fks_channels/ ini_fin_fks
  common /to_abrv/ abrv
  call get_command_argument(1,task)
  ini_fin_fks=3
  abrv='all'
  if (task.eq.'missing') then
     call nlops_update_auxiliary(unc(:,1))
     call ampli_load_production
     error stop 'missing checkpoint unexpectedly accepted'
  endif
  if (task.eq.'only_virt') then
     only_virt=.true.
     abrv='virt'
  elseif (task.eq.'born') then
     abrv='born'
  elseif (task.eq.'zero') then
     zero_target=.true.
  elseif (task.eq.'covariance') then
     virtual_fraction=1d0
  endif
  imode=0
  ifold=1
  call ampli_integrate(analytic_sigint)
  if (task.eq.'wrong_channel') then
     iconfig=iconfig+1
  endif
  imode=1
  ncalls0=4000
  itmax=4
  ifold(1:ndim)=[2,4]
  opens_before=nopen
  folds_before=nfold
  closes_before=nclose
  call ampli_integrate(analytic_sigint)
  call check(nopen-opens_before.eq.16000_8,'survey point count')
  call check(nclose-closes_before.eq.16000_8,'one closing callback per point')
  call check(nfold-folds_before.eq.7_8*16000_8,'all eight folds evaluated')
  call check(written_absolute_error.eq.ampli_absolute_uncertainty,'absolute uncertainty passed to results writer')
  expected=sqrt(sum((survey_square-survey_sum**2/4000d0)*4000d0/3999d0))/16000d0
  call check(abs(ampli_absolute_uncertainty-expected).lt.max(1d-12,expected*1d-9), &
       'absolute uncertainty matches direct target second moments')
  if (task.eq.'covariance') then
     call check(abs(ampli_absolute_uncertainty-(unc(1,1)+unc(5,1))).lt.1d-12, &
          'perfect positive covariance retained')
     call check(ampli_absolute_uncertainty.gt.1.2d0*sqrt(unc(1,1)**2+unc(5,1)**2), &
          'covariance test distinguishes separate-stream quadrature')
  endif
  base=1.5d0*(0.1d0+(1d0-exp(-10d0))/10d0)
  if (task.eq.'zero') then
     call check(all(ans.eq.0d0).and.all(unc.eq.0d0),'zero survey results')
     imode=2
     call ampli_load_production
     call ampli_start_generation(0)
     call ampli_final_weights(weights)
     call check(size(weights).eq.0,'zero-quota complete workflow')
     print *, 'PASS ',trim(task)
     stop
  elseif (only_virt) then
     call check(abs(ans(1,1)-0.5d0*base).lt.max(6d0*unc(1,1),1d-3),'only-virtual absolute rate')
     call check(abs(ans(2,1)+0.5d0*base).lt.max(6d0*unc(2,1),1d-3),'only-virtual signed rate')
     call check(ans(5,1).eq.0d0,'only-virtual residual not double counted')
  else
     call check(abs(ans(1,1)-2d0*base).lt.max(6d0*unc(1,1),1d-3),'nonvirtual absolute rate')
     if (task.ne.'born') then
        call check(abs(ans(5,1)-0.5d0*base).lt.max(6d0*unc(5,1),1d-3),'residual absolute rate')
        call check(abs(ans(2,1)-1.5d0*base).lt.max(6d0*unc(2,1),1d-3),'full signed rate')
     else
        call check(ans(5,1).eq.0d0,'Born-only has no residual stream')
     endif
  endif
  imode=2
  saved_absolute_error=ampli_absolute_uncertainty
  ampli_absolute_uncertainty=-1d0
  if (task.eq.'wrong_folding') ifold(1)=4
  if (task.eq.'missing_MC_state') then
     open(newunit=unit,file='grid.MC_integer',status='old')
     close(unit,status='delete')
  endif
  if (task.eq.'wrong_stage') then
     ! A second survey cannot consume a production-stage checkpoint.
     imode=1
     call ampli_integrate(analytic_sigint)
     error stop 'wrong checkpoint stage unexpectedly accepted'
  endif
  call ampli_load_production
  call check(ampli_absolute_uncertainty.eq.saved_absolute_error,'absolute uncertainty checkpoint round trip')
  quota=4000
  call ampli_start_generation(quota)
  open(newunit=unit,file='candidates.lhe',status='replace')
  write(unit,'(a)') '<LesHouchesEvents version="3.0">'
  write(unit,'(a)') '<header>sample header</header>'
  candidates=0
  do
     call ampli_next_candidate(analytic_sigint,to_write,done)
     if (to_write) then
        candidates=candidates+1
        write(unit,'(a)') '<event test="preserve">'
        write(unit,*) 1,1,7d0*last_sign,91d0,0.007d0,0.12d0
        write(unit,'(a)') ' 11 1 0 0 0 0 0 0 0 1 0 0 9'
        write(unit,'(a)') '<mgrwgt>reference data must survive</mgrwgt>'
        write(unit,'(a)') '</event>'
     endif
     if (done) exit
  enddo
  write(unit,'(a)') '</LesHouchesEvents>'
  close(unit)
  call ampli_final_weights(weights)
  call check(size(weights).eq.candidates,'candidate payload ordering')
  call check(count(weights.gt.0d0).eq.quota,'exact driver quota')
  call check(all(pack(weights,weights.gt.0d0).eq.1d0),'strict unweighting retained unit factors')
  call ampli_rewrite_events('candidates.lhe','events.lhe',weights,quota)
  open(newunit=iu,file='events.lhe',status='old')
  kept=0
  negatives=0
  do
     read(iu,'(a)',iostat=ios) line
     if (ios.lt.0) exit
     call check(ios.eq.0,'read final events')
     if (index(line,'<event test=').eq.1) then
        kept=kept+1
        read(iu,*) nup,idprup,weight,scale,aqed,aqcd
        call check(abs(weight).eq.7d0,'external MG5 event normalization preserved')
        call check(scale.eq.91d0,'shower scale preserved')
        if (weight.lt.0d0) negatives=negatives+1
        read(iu,'(a)') line
        read(iu,'(a)') line
        call check(index(line,'reference data must survive').gt.0,'reweighting payload preserved')
     endif
  enddo
  close(iu)
  call check(kept.eq.quota,'LHE rewrite exact quota')
  if (only_virt) then
     call check(negatives.eq.quota,'only-virtual event signs')
  elseif (task.eq.'born') then
     call check(negatives.eq.0,'Born-only event signs')
  else
     call check(abs(dble(negatives)/dble(quota)-0.2d0).lt.0.035d0,'virtual split signed event normalization')
  endif
  print *, 'PASS ',trim(task)
contains
  subroutine check(condition,message)
    logical,intent(in) :: condition
    character(len=*),intent(in) :: message
    if (.not.condition) then
       print *, 'FAIL: ',message
       error stop 1
    endif
  end subroutine check
end program test_adapter
'''


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestAmpliAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory(prefix='mg5_ampli_adapter_')
        cls.work = pathlib.Path(cls.tempdir.name)
        (cls.work / 'stub.f90').write_text(STUB)
        (cls.work / 'driver.f90').write_text(DRIVER)
        cls.executable = cls.work / 'test_adapter'
        command = [shutil.which('gfortran'), '-O1', '-g', '-fcheck=all,no-recursion',
                   '-ffpe-trap=invalid,zero,overflow', '-fbacktrace',
                   str(SOURCE / 'integrator_helpers.f90'),
                   str(SOURCE / 'simple_integrator.f90'), str(cls.work / 'stub.f90'),
                   str(SOURCE / 'ampli_mint_adapter.f90'), str(cls.work / 'driver.f90'),
                   '-o', str(cls.executable)]
        result = subprocess.run(command, cwd=cls.work, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode:
            cls.tempdir.cleanup()
            raise RuntimeError(result.stdout)

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    def run_case(self, name, expected_error=None):
        with tempfile.TemporaryDirectory(dir=self.work, prefix=name + '_') as run:
            result = subprocess.run([str(self.executable), name], cwd=run,
                                    text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, timeout=45)
            if expected_error is None:
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertIn('PASS', result.stdout)
            else:
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn(expected_error, result.stdout)

    def test_complete_signed_workflow(self):
        self.run_case('signed')

    def test_only_virtual_workflow(self):
        self.run_case('only_virt')

    def test_born_only_workflow(self):
        self.run_case('born')

    def test_zero_target_and_quota(self):
        self.run_case('zero')

    def test_correlated_absolute_rate_uncertainty(self):
        self.run_case('covariance')

    def test_missing_checkpoint(self):
        self.run_case('missing', 'Missing AmpliCol checkpoint')

    def test_missing_integer_sampling_checkpoint(self):
        self.run_case('missing_MC_state', 'Missing MC_integer state')

    def test_wrong_checkpoint_channel(self):
        self.run_case('wrong_channel', 'checkpoint belongs to a different channel')

    def test_wrong_checkpoint_folding(self):
        self.run_case('wrong_folding', 'checkpoint folding differs')

    def test_wrong_checkpoint_stage(self):
        self.run_case('wrong_stage', 'checkpoint version or stage')
