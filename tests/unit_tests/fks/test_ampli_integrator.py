"""Compile and exercise the standalone AmpliCol adapter without MG5 physics."""

import pathlib
import shutil
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'Template' / 'NLO' / 'SubProcesses'


PROGRAM = r'''
module test_rng
  implicit none
  integer(kind=8) :: rng_state=1234567_8
  real(kind=8) :: forced_random=-1d0
end module test_rng

double precision function ran2()
  use test_rng
  implicit none
  if (forced_random.ge.0d0) then
     ran2=forced_random
  else
     rng_state=mod(48271_8*rng_state,2147483647_8)
     ran2=dble(rng_state)/2147483647d0
  endif
end function ran2

program test_integrator
  use simple_integrator_mod,only: staged_integrator
  use test_rng
  implicit none
  type(staged_integrator) :: integ,restored
  real(kind=8) :: x(2),y(2),u(2),w,v,values(2),res(2),unc(2),r2(2),e2(2),target
  real(kind=8) :: actual,expected,weight,constant_res(2),constant_unc(2)
  real(kind=8),allocatable :: weights(:),weights2(:),signs(:)
  integer :: i,j,k,iteration,unit,nwritten,case_seed,quota_value,envelope_choice,nlarge,rep
  integer(kind=8) :: saved_rng
  logical :: to_write,done
  character(len=40) :: task
  call get_command_argument(1,task)
  call integ%init(2,2)
  select case(trim(task))
  case('analytic')
     do iteration=1,4
        call integ%begin_iteration(iteration.lt.4)
        do i=1,25000
           call integ%sample(x,w)
           values(1)=exp(-50d0*x(1))*(1d0+x(2))*w
           values(2)=values(1)*(2d0*x(2)-1d0)
           call integ%observe(x,values,values(1))
        enddo
        call integ%finish_iteration(res,unc)
     enddo
     call check(abs(res(1)-0.03d0).lt.max(5d0*unc(1),1d-5),'absolute integral')
     call check(abs(res(2)-1d0/300d0).lt.max(5d0*unc(2),1d-5),'signed integral')
     call check(integ%total_points.eq.100000_8,'count all integration trials')
     call check(all(unc.gt.0d0),'nonzero analytic uncertainties')
  case('folding')
     call integ%begin_iteration(.false.)
     do i=1,40000
        call integ%sample(x,w,u)
        values=0d0
        do j=1,4
           do k=1,2
              call integ%map_fold(u,[j,k],[4,2],y,v)
              call check(y(1).ge.dble(j-1)/4d0.and.y(1).le.dble(j)/4d0,'folded x interval')
              call check(y(2).ge.dble(k-1)/2d0.and.y(2).le.dble(k)/2d0,'folded y interval')
              values(1)=values(1)+v
              values(2)=values(2)+v*(y(1)**2-y(2)**3)
           enddo
        enddo
        call check(abs(values(1)-1d0).lt.1d-13,'fold Jacobian normalizes')
        call integ%observe(x,values,abs(values(2)))
     enddo
     call integ%finish_iteration(res,unc)
     call check(abs(res(2)-1d0/12d0).lt.max(5d0*unc(2),1d-5),'folded signed integral')
     call check(abs(res(1)-1d0).lt.1d-13,'folded constant integral')
     ! An adapted grid uses quantile folds, not equal physical intervals.
     call train(integ,3000)
     call integ%begin_iteration(.false.)
     do i=1,40000
        call integ%sample(x,w,u)
        values=0d0
        do j=1,4
           do k=1,2
              call integ%map_fold(u,[j,k],[4,2],y,v)
              values(1)=values(1)+v
              values(2)=values(2)+v*(y(1)**2-y(2)**3)
           enddo
        enddo
        call integ%observe(x,values,abs(values(2)))
     enddo
     call integ%finish_iteration(res,unc)
     call check(abs(res(1)-1d0).lt.max(5d0*unc(1),1d-4),'adapted folded constant')
     call check(abs(res(2)-1d0/12d0).lt.max(5d0*unc(2),1d-4),'adapted folded signed integral')
  case('checkpoint')
     call integ%begin_iteration(.true.)
     call accumulate(integ,2000)
     open(newunit=unit,status='scratch',form='formatted')
     call integ%save(unit)
     rewind(unit)
     call restored%load(unit,2,2)
     close(unit)
     call check(restored%npoints.eq.2000_8,'checkpoint point count')
     saved_rng=rng_state
     call accumulate(integ,3000)
     rng_state=saved_rng
     call accumulate(restored,3000)
     call integ%finish_iteration(res,unc)
     call restored%finish_iteration(r2,e2)
     call check(all(res.eq.r2).and.all(unc.eq.e2),'exact accumulator round trip')
     forced_random=0.312345678901234d0
     call integ%sample(x,w)
     call restored%sample(y,v)
     call check(all(x.eq.y).and.w.eq.v,'exact adapted map round trip')
     forced_random=-1d0
     call integ%start_production(40,5d0,10000_8)
     do i=1,50
        call integ%consider(1d0,to_write,done)
        call check(.not.done,'checkpoint during unfinished production')
     enddo
     open(newunit=unit,status='scratch',form='formatted')
     call integ%save(unit)
     rewind(unit)
     call restored%load(unit,2,2)
     close(unit)
     saved_rng=rng_state
     do
        call integ%consider(1d0,to_write,done)
        if (done) exit
     enddo
     rng_state=saved_rng
     do
        call restored%consider(1d0,to_write,done)
        if (done) exit
     enddo
     call integ%final_weights(weights)
     call restored%final_weights(weights2)
     call check(integ%ntrials.eq.restored%ntrials,'production RNG continuation')
     call check(size(weights).eq.size(weights2),'production candidate round trip')
     call check(all(weights.eq.weights2),'production selection round trip')
  case('zero_sparse')
     call integ%begin_iteration(.true.)
     do i=1,1000
        call integ%sample(x,w)
        call integ%observe(x,[0d0,0d0],0d0)
     enddo
     call integ%finish_iteration(res,unc)
     call check(all(res.eq.0d0).and.all(unc.eq.0d0),'finite zero integrals')
     call check(integ%npoints.eq.1000_8,'zeros count toward integration budget')
     call integ%sample(x,w)
     call check(abs(w-1d0).lt.1d-13,'zero target preserves grid')
     call integ%begin_iteration(.true.)
     do i=1,1000
        call integ%sample(x,w)
        target=0d0
        if (i.eq.500) target=1000d0
        call integ%observe(x,[target,-target],target)
     enddo
     call integ%finish_iteration(res,unc)
     call check(abs(res(1)-1d0).lt.1d-12.and.abs(res(2)+1d0).lt.1d-12,'sparse signed integral')
     call check(abs(unc(1)-1d0).lt.1d-12,'sparse trial uncertainty')
     call integ%start_production(2,1d0,25_8)
     do i=1,25
        call integ%consider(0d0,to_write,done)
        call check(.not.to_write,'zero target cannot write event')
        call check(done.eqv.(i.eq.25),'zero production bounded by total trials')
     enddo
     call check(integ%exhausted.and..not.integ%quota_complete,'zero production exhaustion')
     call integ%start_production(0,0d0,0_8)
     call integ%final_weights(weights)
     call check(size(weights).eq.0.and.integ%quota_complete,'zero event quota')
  case('quotas')
     call integ%start_production(101,2d0,10000_8)
     nwritten=0
     do
        call integ%consider(1d0,to_write,done)
        if (to_write) nwritten=nwritten+1
        if (done) exit
     enddo
     call integ%final_weights(weights)
     call check(size(weights).eq.nwritten,'candidate labels match temporary events')
     call check(count(weights.gt.0d0).eq.101,'exact event quota')
     call check(all(pack(weights,weights.gt.0d0).eq.1d0),'unit weights under valid envelope')
     call check(integ%overweight.eq.0d0.and..not.integ%exhausted,'ordinary unweighting succeeds')
     call check(abs(sum(weights)-101d0).lt.1d-12,'nominal normalization unchanged')
  case('overweight')
     ! Deliberately underestimate the envelope, and exhaust the budget before
     ! the raw overweight criterion is met. The tail must retain its correction.
     call integ%start_production(4,0d0,6_8)
     forced_random=0.5d0
     do i=1,6
        weight=1d0
        if (i.eq.1) weight=100d0
        call integ%consider(weight,to_write,done)
        call check(to_write,'zero envelope retains every positive candidate')
     enddo
     forced_random=-1d0
     call check(done.and.integ%exhausted.and.integ%quota_complete,'overweight quota at budget limit')
     call integ%final_weights(weights)
     call check(count(weights.gt.0d0).eq.4,'overweight selection exact quota')
     call check(integ%overweight.gt.1d0,'raw overweight explicitly reported')
     call check(weights(1).gt.3d0,'large tail has nonunit corrected weight')
     call check(abs(sum(weights)-4d0).lt.1d-12,'corrected weights preserve nominal normalization')
     ! Many independent signed heavy-tail samples: the correction restores the
     ! known rate while uncorrected signs would give a strongly different value.
     actual=0d0
     allocate(signs(100000))
     do case_seed=1,20
        call integ%start_production(2000,1d0,30000_8)
        nwritten=0
        do
           call random_weight(weight,target)
           call integ%consider(weight,to_write,done)
           if (to_write) then
              nwritten=nwritten+1
              signs(nwritten)=target
           endif
           if (done) exit
        enddo
        call check(integ%quota_complete,'heavy-tail quota reached')
        call integ%final_weights(weights)
        actual=actual+sum(weights*signs(1:nwritten))/2000d0
     enddo
     actual=actual/20d0
     ! q=0.02 has negative weight 100, q=0.98 positive weight 1.
     expected=(0.98d0-2d0)/(0.98d0+2d0)
     call check(abs(actual-expected).lt.0.035d0,'corrected signed heavy-tail rate')
  case('strict_rejection')
     ! At tiny quotas a rank-selected pool is detectably biased. Ordinary
     ! rejection must give p(weight=2)=2/3 for either valid frozen envelope.
     do quota_value=1,2
        do envelope_choice=1,2
           nlarge=0
           do rep=1,20000
              call integ%start_production(quota_value,2d0*envelope_choice,1000_8,overweight_tolerance=0d0)
              do
                 call random_two_weights(weight)
                 call integ%consider(weight,to_write,done)
                 if (to_write.and.weight.eq.2d0) nlarge=nlarge+1
                 if (done) exit
              enddo
              call check(integ%quota_complete.and..not.integ%envelope_exceeded,'valid strict bound completes')
              call integ%final_weights(weights)
              call check(size(weights).eq.quota_value.and.all(weights.eq.1d0),'strict quota and unit weights')
           enddo
           actual=dble(nlarge)/dble(20000*quota_value)
           call check(abs(actual-2d0/3d0).lt.0.012d0,'strict small-quota event density')
           call check(abs((1d0-2d0*actual)+1d0/3d0).lt.0.024d0,'strict small-quota signed rate')
        enddo
     enddo
     call integ%start_production(2,1d0,1000_8,overweight_tolerance=0d0)
     call integ%consider(2d0,to_write,done)
     call check(done.and..not.to_write.and.integ%envelope_exceeded,'observed underestimated bound invalidates attempt')
     call check(.not.integ%quota_complete,'invalid attempt cannot claim quota completion')
  case('strict_bad_bound')
     call integ%start_production(1,1d0,1000_8,overweight_tolerance=0d0)
     call integ%consider(2d0,to_write,done)
     call integ%final_weights(weights)
     error stop 'violated strict envelope unexpectedly accepted'
  case('bad_version')
     open(newunit=unit,status='scratch',form='formatted')
     write(unit,'(a)') 'MG5_SIMPLE_INTEGRATOR 999'
     rewind(unit)
     call restored%load(unit,2,2)
     error stop 'bad version unexpectedly accepted'
  case('bad_dimensions')
     open(newunit=unit,status='scratch',form='formatted')
     call integ%save(unit)
     rewind(unit)
     call restored%load(unit,3,2)
     error stop 'bad dimensions unexpectedly accepted'
  case('truncated')
     open(newunit=unit,status='scratch',form='formatted')
     write(unit,'(a)') 'MG5_SIMPLE_INTEGRATOR 1'
     write(unit,*) 2,2
     rewind(unit)
     call restored%load(unit,2,2)
     error stop 'truncated checkpoint unexpectedly accepted'
  case('incomplete_quota')
     call integ%start_production(4,1d0,2_8)
     call integ%consider(0d0,to_write,done)
     call integ%consider(0d0,to_write,done)
     call integ%final_weights(weights)
     error stop 'incomplete quota unexpectedly accepted'
  case default
     error stop 'unknown test case'
  end select
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

  subroutine accumulate(sampler,n)
    type(staged_integrator),intent(inout) :: sampler
    integer,intent(in) :: n
    integer :: point
    real(kind=8) :: xx(2),ww,ff(2)
    do point=1,n
       call sampler%sample(xx,ww)
       ff(1)=(0.1d0+exp(-10d0*xx(1)))*(1d0+xx(2))*ww
       ff(2)=ff(1)*(2d0*xx(2)-1d0)
       call sampler%observe(xx,ff,ff(1))
    enddo
  end subroutine accumulate

  subroutine train(sampler,n)
    type(staged_integrator),intent(inout) :: sampler
    integer,intent(in) :: n
    real(kind=8) :: rr(2),ee(2)
    call sampler%begin_iteration(.true.)
    call accumulate(sampler,n)
    call sampler%finish_iteration(rr,ee)
  end subroutine train

  subroutine random_weight(ww,sgn)
    real(kind=8),intent(out) :: ww,sgn
    real(kind=8),external :: ran2
    ww=1d0
    sgn=1d0
    if (ran2().lt.0.02d0) then
       ww=100d0
       sgn=-1d0
    endif
  end subroutine random_weight

  subroutine random_two_weights(ww)
    real(kind=8),intent(out) :: ww
    real(kind=8),external :: ran2
    ww=1d0
    if (ran2().lt.0.5d0) ww=2d0
  end subroutine random_two_weights
end program test_integrator
'''


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestAmpliIntegrator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory(prefix='mg5_ampli_integrator_')
        cls.work = pathlib.Path(cls.tempdir.name)
        cls.program = cls.work / 'test_integrator.f90'
        cls.program.write_text(PROGRAM)
        cls.executable = cls.work / 'test_integrator'
        # GCC 12 can emit an undefined is_recursive symbol for the upstream
        # polymorphic grid_update with optimized recursion instrumentation.
        command = [shutil.which('gfortran'), '-O1', '-g', '-fcheck=all,no-recursion',
                   '-ffpe-trap=invalid,zero,overflow', '-fbacktrace',
                   str(SOURCE / 'integrator_helpers.f90'),
                   str(SOURCE / 'simple_integrator.f90'), str(cls.program),
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
        result = subprocess.run([str(self.executable), name], cwd=self.work,
                                text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=45)
        if expected_error is None:
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn('PASS', result.stdout)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn(expected_error, result.stdout)

    def test_signed_analytic_integrals(self):
        self.run_case('analytic')

    def test_folded_integrals(self):
        self.run_case('folding')

    def test_checkpoint_continuation(self):
        self.run_case('checkpoint')

    def test_zero_and_sparse_targets(self):
        self.run_case('zero_sparse')

    def test_exact_event_quotas(self):
        self.run_case('quotas')

    def test_overweight_corrections_and_signed_tail(self):
        self.run_case('overweight')

    def test_strict_rejection_tiny_quota_densities(self):
        self.run_case('strict_rejection')

    def test_strict_rejection_refuses_underestimated_envelope(self):
        self.run_case('strict_bad_bound', 'strict production envelope exceeded')

    def test_incompatible_checkpoint_version(self):
        self.run_case('bad_version', 'incompatible checkpoint version')

    def test_incompatible_checkpoint_dimensions(self):
        self.run_case('bad_dimensions', 'incompatible checkpoint dimensions')

    def test_truncated_checkpoint(self):
        self.run_case('truncated', 'invalid checkpoint counters')

    def test_insufficient_event_quota_fails(self):
        self.run_case('incomplete_quota', 'event quota not reached')
