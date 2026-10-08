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
  use,intrinsic :: ieee_arithmetic,only: ieee_value,ieee_quiet_nan,ieee_positive_inf,ieee_is_finite
  implicit none
  type(staged_integrator) :: integ,restored
  real(kind=8) :: x(2),y(2),u(2),w,v,values(2),res(2),unc(2),r2(2),e2(2),target
  real(kind=8) :: actual,expected,weight,constant_res(2),constant_unc(2),fold_points(2,8)
  real(kind=8),allocatable :: weights(:),weights2(:),signs(:),priorities(:),priorities2(:)
  real(kind=8) :: expected_weights(100000),expected_priorities(100000),observables(100000),event_signs(100000)
  real(kind=8) :: before_map(2,19),after_map(2,19),before_jac(19),sums(2),squares(2),errors(2),sgn
  real(kind=8),allocatable :: candidate_factors(:),candidate_factors2(:),observed_factors(:)
  integer,allocatable :: candidate_tail(:),candidate_tail2(:)
  integer :: fill_before(2),fill_after(2)
  integer :: native_epoch_count,native_stored,native_mask(2),native_updates,epoch_id,eligible,last_active,proposal_id
  integer :: row_epoch(100000),row_tail(100000),generated_target,final_target
  integer(kind=8) :: native_trials,row_n,row_nz,row_target
  real(kind=8) :: native_moments(5),epoch_moments(5),row_c,row_m,row_t,native_logz,native_tails(3)
  real(kind=8) :: candidate_x(2,100000),candidate_jac(100000),row_correction(100000),row_factor,lo,hi,mid,newjac
  real(kind=8) :: expected_envelope,last_envelope,oldest_mean,signed_mean,first_map(2)
  integer :: i,j,k,iteration,unit,unit2,ios,nwritten,case_seed,quota_value,envelope_choice,nlarge,rep
  integer(kind=8) :: saved_rng
  logical :: to_write,done
  character(len=40) :: task
  character(len=100000) :: line
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
  case('folded_adaptation')
     call integ%begin_iteration(.true.)
     do i=1,4000
        call integ%sample(x,w,u)
        values=0d0
        do j=1,4
           do k=1,2
              call integ%map_fold(u,[j,k],[4,2],y,v)
              fold_points(:,2*(j-1)+k)=y
              values(1)=values(1)+v
           enddo
        enddo
        call integ%observe(fold_points(:,1),values,values(1),fold_points)
     enddo
     call integ%finish_iteration(res,unc)
     call check(integ%npoints.eq.4000_8,'folds remain one statistical observation')
     call check(abs(res(1)-1d0).lt.1d-13,'folded constant normalization')
     do i=1,19
        forced_random=dble(i)/20d0
        call integ%sample(x,w)
        call check(all(abs(x-forced_random).lt.0.03d0),'folded adaptation covers every image of constant target')
     enddo
     forced_random=-1d0
  case('checkpoint')
     call train(integ,2048)
     call checkpoint_fill_sizes(integ,fill_before)
     call check(all(fill_before.eq.16),'checkpoint starts from first refined survey grid')
     call integ%begin_iteration(.true.)
     call accumulate(integ,2000)
     open(newunit=unit,status='scratch',form='formatted')
     call integ%save(unit)
     rewind(unit)
     call restored%load(unit,2,2)
     close(unit)
     call check(restored%npoints.eq.2000_8,'checkpoint point count')
     call checkpoint_fill_sizes(restored,fill_after)
     call check(all(fill_before.eq.fill_after),'checkpoint preserves refined survey resolution')
     saved_rng=rng_state
     call accumulate(integ,3000)
     rng_state=saved_rng
     call accumulate(restored,3000)
     call integ%finish_iteration(res,unc)
     call restored%finish_iteration(r2,e2)
     call checkpoint_fill_sizes(integ,fill_before)
     call checkpoint_fill_sizes(restored,fill_after)
     call check(all(fill_before.eq.32).and.all(fill_after.eq.fill_before),'checkpoint resumes survey bin doubling')
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
  case('survey_resolution')
     call checkpoint_fill_sizes(integ,fill_before)
     call check(all(fill_before.eq.8),'survey begins with eight fill bins')
     do iteration=1,4
        call train(integ,2048)
        call checkpoint_fill_sizes(integ,fill_after)
        call check(all(fill_after.eq.min(8*2**iteration,64)),'early survey doubles fill bins through 64')
     enddo
     ! The early doubling floor is not a cap on statistics-driven refinement.
     call train(integ,640000)
     call checkpoint_fill_sizes(integ,fill_before)
     call check(all(fill_before.gt.64),'well-populated survey can refine above early doubling floor')
     call train(integ,2048)
     call checkpoint_fill_sizes(integ,fill_after)
     call check(all(fill_after.eq.fill_before),'small later survey batch cannot shrink the grid')
  case('survey_no_adaptation')
     call checkpoint_fill_sizes(integ,fill_before)
     do i=1,19
        forced_random=dble(i)/20d0
        call integ%sample(before_map(:,i),before_jac(i))
     enddo
     forced_random=-1d0
     do iteration=1,4
        call integ%begin_iteration(iteration.ne.1)
        if (iteration.le.2) then
           call accumulate(integ,2048)
        elseif (iteration.eq.3) then
           do i=1,2048
              call integ%sample(x,w)
              call integ%observe(x,[0d0,0d0],0d0)
           enddo
        endif
        ! Cover disabled collection, explicit finish override, zero and empty.
        if (iteration.eq.2) then
           call integ%finish_iteration(res,unc,adapt=.false.)
        else
           call integ%finish_iteration(res,unc)
        endif
        call checkpoint_fill_sizes(integ,fill_after)
        call check(all(fill_after.eq.fill_before),'non-adapting survey preserves fill resolution')
        do i=1,19
           forced_random=dble(i)/20d0
           call integ%sample(x,w)
           call check(all(x.eq.before_map(:,i)).and.w.eq.before_jac(i),'non-adapting survey preserves map bitwise')
        enddo
        forced_random=-1d0
     enddo
     call train(integ,2048)
     call checkpoint_fill_sizes(integ,fill_after)
     call check(all(fill_after.eq.16),'first useful adaptation still performs first doubling')
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
     call check(integ%overweight.gt.0.9d0,'full tail mass explicitly reported')
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
  case('tail_mass')
     ! A barely-overweight event contributes its WHOLE weight to the tail.
     ! Mean excess would accept this sample, but the 1% tail test must fail.
     call integ%start_production(100,1d0,10000_8,final_quota=100)
     forced_random=0.5d0
     do i=1,150
        weight=1d0
        if (i.eq.1) weight=2.001d0
        call integ%consider(weight,to_write,done)
     enddo
     call check(.not.done,'full tail rejects tiny-excess sample')
     call check(integ%reserve_tail.gt.0.01d0,'full selected tail rather than excess')
     call check(abs(integ%full_trial_tail-2.001d0/151.001d0).lt.1d-12,'all-trial tail includes every trial')
     ! More candidates raise the threshold and reevaluate the stored tail.
     forced_random=0.1d0
     do i=1,500
        call integ%consider(1d0,to_write,done)
        if (done) exit
     enddo
     call check(done.and.integ%overweight.lt.0.01d0,'generation continues until tail is controlled')
     call integ%final_weights(weights)
     call check(count(weights.gt.0d0).eq.100,'tail reevaluation preserves quota')
  case('tail_worst_subset')
     call integ%start_production(110,1d0,10000_8,final_quota=100)
     forced_random=0.9d0
     do i=1,1000
        call integ%consider(0.5d0,to_write,done)
        call check(.not.to_write,'rejected observations contribute to tail denominator')
     enddo
     forced_random=0.5d0
     do i=1,165
        weight=1d0
        if (i.eq.1) weight=2.1d0
        call integ%consider(weight,to_write,done)
     enddo
     call check(integ%full_trial_tail.lt.0.01d0.and.integ%reserve_tail.lt.0.01d0,'trial and reserve pass')
     call check(integ%worst_subset_tail.gt.0.01d0.and..not.done,'thinned sample worst case prevents completion')
     call check(abs(integ%worst_subset_tail-1.05d0/100.05d0).lt.1d-12,'analytic worst-subset fraction')
  case('tail_factor')
     call integ%start_production(110,1d0,10000_8,final_quota=110)
     forced_random=0.9d0
     do i=1,1000
        call integ%consider(0.5d0,to_write,done)
     enddo
     forced_random=0.5d0
     do i=1,165
        weight=1d0
        if (i.eq.165) weight=2.01d0
        call integ%consider(weight,to_write,done)
     enddo
     call check(done,'provisional unit LHE factors pass')
     call integ%record_candidate_factor(100d0,done)
     call check(.not.done.and.integ%reserve_tail.gt.0.4d0,'actual bias factor revokes provisional completion')
     open(newunit=unit,status='scratch',form='formatted')
     call integ%save(unit)
     rewind(unit)
     call restored%load(unit,2,2)
     close(unit)
     call check(restored%total_abs.eq.integ%total_abs.and.restored%threshold.eq.integ%threshold, &
          'checkpoint preserves trial mass and threshold')
     call check(restored%reserve_tail.eq.integ%reserve_tail.and. &
          restored%full_trial_tail.eq.integ%full_trial_tail.and. &
          restored%worst_subset_tail.eq.integ%worst_subset_tail,'checkpoint preserves tail diagnostics')
     ! Rechecking a different last factor after restoration tests saved factor
     ! storage and active production state, not just the cached diagnostics.
     call restored%record_candidate_factor(100d0,done)
     call check(.not.done,'checkpoint preserves revoked completion')
     forced_random=0.1d0
     do i=1,500
        call integ%consider(1d0,to_write,done)
        if (to_write) call integ%record_candidate_factor(1d0,done)
        if (done) exit
     enddo
     call check(done.and.integ%overweight.lt.0.01d0,'factor-corrected generator resumes')
  case('tail_extreme_priority')
     call integ%start_production(1,1d0,100_8)
     forced_random=0d0
     call integ%consider(huge(1d0)/4d0,to_write,done)
     call check(.not.done,'first extreme candidate cannot finish priority selection')
     call integ%consider(huge(1d0)/4d0,to_write,done)
     call check(done.and.integ%threshold.eq.huge(1d0),'unrepresentable priority threshold saturates safely')
     call check(integ%overweight.eq.0d0,'finite raw weights below saturated threshold have no tail')
     call integ%final_weights(weights)
     call check(count(weights.gt.0d0).eq.1.and.sum(weights).eq.1d0,'extreme priority preserves quota')
  case('tail_equality')
     call integ%start_production(199,0d0,10000_8)
     forced_random=0.5d0
     do i=1,299
        weight=0.5d0
        if (i.eq.1) weight=2d0
        call integ%consider(weight,to_write,done)
     enddo
     call check(integ%reserve_tail.eq.0.01d0,'constructed one-percent reserve tail')
     call check(.not.done,'one percent equality is rejected')
  case('production_adapt_maps')
     call train(integ,3000)
     do i=1,19
        forced_random=dble(i)/20d0
        call integ%sample(before_map(:,i),before_jac(i))
     enddo
     forced_random=-1d0
     call integ%start_production(10000,1d9,96_8,ifold=[1,4],adaptation_interval=16_8)
     do i=1,96
        call integ%sample(x,w)
        weight=0d0
        if (i.gt.16) weight=exp(-20d0*x(1))*w
        call integ%consider(weight,to_write,done,x=x)
        call check(.not.to_write,'rejected and zero trials train production maps')
        if (i.eq.16) then
           call check(integ%adaptation_updates.eq.0,'all-zero adaptation batch leaves map fixed')
           call check(integ%adaptation_batch.eq.32_8.and.integ%adaptation_points.eq.0_8,'zero batch advances schedule')
        endif
     enddo
     call check(integ%adaptation_updates.eq.1,'nonzero rejected batch updates grid')
     call check(integ%adaptation_points.eq.48_8.and.integ%adaptation_batch.eq.64_8,'all trials counted toward next batch')
     do i=1,19
        forced_random=dble(i)/20d0
        call integ%sample(after_map(:,i),v)
     enddo
     call check(all(before_map(2,:).eq.after_map(2,:)),'folded nonuniform coordinate map is bitwise frozen')
     call check(maxval(abs(before_map(1,:)-after_map(1,:))).gt.1d-3,'unfolded proposal map changes')
     call check(all(after_map(:,2:19).gt.after_map(:,1:18)),'updated sampled map monotone')
     call integ%start_pool(0_8,1d0)
     call check(.not.any(integ%adaptation_mask).and.integ%adaptation_updates.eq.0,'legacy pool clears adaptation')
     call check(integ%adaptation_interval.eq.0_8.and.integ%adaptation_points.eq.0_8,'legacy pool clears schedule')
     call integ%init(2,2)
     call check(.not.any(integ%adaptation_mask),'reinitialization clears adaptation mask')
  case('production_adapt_resolution')
     call train(integ,64000)
     call checkpoint_fill_sizes(integ,fill_before)
     call check(all(fill_before.gt.8),'survey has finer adaptation resolution')
     call integ%start_production(1000,1d9,16_8,ifold=[1,2],adaptation_interval=16_8)
     do i=1,16
        call integ%sample(x,w)
        call integ%consider((1d0+9d0*x(1)**2)*w,to_write,done,x=x)
     enddo
     call checkpoint_fill_sizes(integ,fill_after)
     call check(fill_after(1).eq.fill_before(1),'production preserves finer survey fill resolution')
     call check(fill_after(2).eq.fill_before(2),'folded fill resolution unchanged')
  case('production_no_bin_growth')
     call train(integ,2048)
     call checkpoint_fill_sizes(integ,fill_before)
     call check(all(fill_before.eq.16),'production inherits early survey resolution')
     call integ%start_production(100000,1d12,65536_8,ifold=[1,2],adaptation_interval=65536_8)
     do i=1,65536
        call integ%sample(x,w)
        call integ%consider((1d0+9d0*x(1)**2)*w,to_write,done,x=x)
     enddo
     call check(integ%adaptation_updates.eq.1,'large legacy production batch adapts grid positions')
     call checkpoint_fill_sizes(integ,fill_after)
     call check(all(fill_after.eq.fill_before),'large production batch cannot increase fill resolution')
  case('production_allfolded')
     call train(integ,3000)
     do i=1,19
        forced_random=dble(i)/20d0
        call integ%sample(before_map(:,i),before_jac(i))
     enddo
     call integ%start_production(1000,1d0,200_8,ifold=[2,4],adaptation_interval=16_8)
     do i=1,200
        call integ%consider(1d0,to_write,done)
     enddo
     call check(.not.any(integ%adaptation_mask),'all-folded generation disables adaptation')
     call check(integ%adaptation_updates.eq.0.and.integ%adaptation_interval.eq.0_8.and. &
          integ%adaptation_batch.eq.0_8.and.integ%adaptation_points.eq.0_8,'disabled metadata has zero counters')
     do i=1,19
        forced_random=dble(i)/20d0
        call integ%sample(x,w)
        call check(all(x.eq.before_map(:,i)).and.w.eq.before_jac(i),'all-folded grids remain bitwise fixed')
     enddo
  case('production_adapt_analytic')
     ! Fixed number of draws: test importance estimates while proposals change.
     call integ%start_production(50000,4d0,20000_8,ifold=[1,2],adaptation_interval=64_8)
     sums=0d0
     squares=0d0
     do i=1,20000
        call integ%sample(x,w)
        weight=(1d0+9d0*x(1)**2)*w
        sgn=1d0
        if (x(2).lt.0.5d0) sgn=-1d0
        values=[weight,weight*sgn]
        sums=sums+values
        squares=squares+values**2
        call integ%consider(weight,to_write,done,x=x)
     enddo
     res=sums/20000d0
     errors=sqrt(max(squares-sums*sums/20000d0,0d0)/20000d0/19999d0)
     call check(abs(res(1)-4d0).lt.6d0*errors(1),'adaptive absolute integral agrees with analytic value')
     call check(abs(res(2)).lt.6d0*errors(2),'adaptive signed integral agrees with analytic value')
     call check(integ%adaptation_updates.ge.5,'analytic integration crosses multiple proposal updates')
     ! Independently check retained event density, including pre-update points.
     call integ%start_production(10000,4d0,100000_8,ifold=[1,2],adaptation_interval=64_8)
     nwritten=0
     do i=1,100000
        call integ%sample(x,w)
        weight=(1d0+9d0*x(1)**2)*w
        forced_random=0.001d0+0.998d0*dble(mod(29*i,997))/997d0
        target=log(weight)-log(forced_random)
        call integ%consider(weight,to_write,done,x=x)
        forced_random=-1d0
        if (to_write) then
           nwritten=nwritten+1
           expected_weights(nwritten)=weight
           expected_priorities(nwritten)=target
           observables(nwritten)=0d0
           if (x(1).gt.0.5d0) observables(nwritten)=1d0
           event_signs(nwritten)=1d0
           if (x(2).lt.0.5d0) event_signs(nwritten)=-1d0
        endif
        if (done) exit
     enddo
     call check(integ%quota_complete.and..not.integ%exhausted,'adaptive event generation reaches quota and tail bound')
     call integ%production_candidates(weights2,priorities,weights,candidate_tail,candidate_factors)
     call check(all(weights2.eq.expected_weights(1:nwritten)),'draw-time weights immutable across map updates')
     call check(all(priorities.eq.expected_priorities(1:nwritten)),'draw-time random priorities immutable')
     call check(abs(sum(weights*observables(1:nwritten))/10000d0-0.78125d0).lt.0.02d0, &
          'adaptive generated event density agrees with analytic integral')
     call check(abs(sum(weights*event_signs(1:nwritten))/10000d0).lt.0.04d0,'adaptive signed event sample agrees')
  case('production_adapt_checkpoint')
     call integ%start_production(500,1d0,100000_8,ifold=[1,3],adaptation_interval=32_8)
     do i=1,125
        call generation_point(integ,done)
     enddo
     call check(integ%adaptation_updates.eq.2.and.integ%adaptation_points.eq.29_8,'checkpoint in partially trained batch')
     open(newunit=unit,status='scratch',form='formatted')
     call integ%save(unit)
     rewind(unit)
     call restored%load(unit,2,2)
     close(unit)
     saved_rng=rng_state
     do
        call generation_point(integ,done)
        if (done) exit
     enddo
     rng_state=saved_rng
     do
        call generation_point(restored,done)
        if (done) exit
     enddo
     call check(integ%ntrials.eq.restored%ntrials.and.integ%total_abs.eq.restored%total_abs, &
          'adaptive checkpoint reproduces exact trial stream')
     call check(integ%adaptation_updates.eq.restored%adaptation_updates.and. &
          integ%adaptation_points.eq.restored%adaptation_points.and. &
          integ%adaptation_batch.eq.restored%adaptation_batch,'adaptive checkpoint reproduces update schedule')
     call integ%production_candidates(weights,priorities,weights2,candidate_tail,candidate_factors)
     call restored%production_candidates(signs,priorities2,candidate_factors2,candidate_tail2,observed_factors)
     call check(all(weights.eq.signs).and.all(priorities.eq.priorities2),'adaptive checkpoint preserves candidate history')
     call check(all(weights2.eq.candidate_factors2).and.all(candidate_tail.eq.candidate_tail2).and. &
          all(candidate_factors.eq.observed_factors),'adaptive checkpoint preserves final corrections and LHE factors')
     forced_random=0.314159d0
     call integ%sample(x,w)
     call restored%sample(y,v)
     call check(all(x.eq.y).and.w.eq.v,'adaptive checkpoint reproduces final maps')
     ! A completed production object can start a new independent survey.
     call integ%begin_iteration(.false.)
     call check(.not.any(integ%adaptation_mask),'new survey clears production adaptation state')
     open(newunit=unit,status='scratch',form='formatted')
     call integ%save(unit)
     rewind(unit)
     call restored%load(unit,2,2)
     close(unit)
  case('adapt_bad_folds')
     call integ%start_production(10,1d0,1000_8,ifold=[1,0])
  case('adapt_bad_dimensions')
     call integ%start_production(10,1d0,1000_8,ifold=[1])
  case('adapt_bad_interval')
     call integ%start_production(10,1d0,1000_8,ifold=[1,2],adaptation_interval=0_8)
  case('adapt_large_interval')
     call integ%start_production(10,1d0,1000_8,ifold=[1,2],adaptation_interval=65537_8)
  case('adapt_interval_without_folds')
     call integ%start_production(10,1d0,1000_8,adaptation_interval=16_8)
  case('adapt_missing_x')
     call integ%start_production(10,1d0,1000_8,ifold=[1,2])
     call integ%consider(1d0,to_write,done)
  case('adapt_bad_x_size')
     call integ%start_production(10,1d0,1000_8,ifold=[1,2])
     call integ%consider(1d0,to_write,done,x=[0.2d0])
  case('adapt_bad_x')
     call integ%start_production(10,1d0,1000_8,ifold=[1,2])
     call integ%consider(1d0,to_write,done,x=[0.2d0,1.1d0])
  case('adapt_nan_x')
     call integ%start_production(10,1d0,1000_8,ifold=[1,2])
     call integ%consider(1d0,to_write,done,x=[0.2d0,ieee_value(1d0,ieee_quiet_nan)])
  case('native_stable_training')
     call restored%init(2,2)
     call integ%start_native_production(100000,90000,[1,2],1d0,1d0,1000000_8,initial_nonzero=1000_8)
     call restored%start_native_production(100000,90000,[1,2],1d0,1d0,1000000_8,initial_nonzero=1000_8)
     call native_training_draws(integ,1,1000,.false.)
     call integ%finish_native_iteration(done)
     call check(.not.done.and.integ%native_training_points.eq.0_8.and.integ%adaptation_updates.eq.1, &
          'first substantial training batch is consumed exactly once')
     call native_training_draws(restored,1,1000,.false.)
     call restored%finish_native_iteration(done)
     do i=1,19
        forced_random=dble(i)/20d0
        call integ%sample(before_map(:,i),before_jac(i))
     enddo
     ! Control only the public batch budgets to exercise the exact gate.
     ! These unfinished fixtures are never exported as production pools.
     integ%native_target_nonzero=100_8
     call native_training_draws(integ,1,125,.true.)
     call check(integ%native_training_points.eq.125_8,'zero trials also count towards pending grid training')
     call integ%finish_native_iteration(done)
     call check(integ%native_training_points.eq.125_8.and.integ%adaptation_updates.eq.1, &
          'small completed batch retains its pending histogram')
     call check(integ%native_proposal_id.eq.2,'skipping adaptation preserves proposal identity')
     integ%native_target_nonzero=100_8
     call native_training_draws(integ,126,250,.true.)
     call integ%finish_native_iteration(done)
     call check(integ%native_training_points.eq.250_8.and.integ%ntrials.eq.1250_8, &
          'pending training counts every attempted draw across skipped batches')
     call check(integ%adaptation_updates.eq.1.and.integ%native_proposal_id.eq.2, &
          'exactly twenty percent new statistics does not adapt')
     do i=1,19
        forced_random=dble(i)/20d0
        call integ%sample(x,w)
        call check(all(x.eq.before_map(:,i)).and.w.eq.before_jac(i),'skipped updates leave the proposal bitwise fixed')
     enddo
     integ%native_target_nonzero=1_8
     call native_training_draws(integ,251,251,.true.)
     call integ%finish_native_iteration(done)
     call check(integ%native_training_points.eq.0_8.and.integ%adaptation_updates.eq.2, &
          'accumulated training above twenty percent updates and resets the histogram')
     call check(integ%native_proposal_id.eq.3,'a changed proposal advances identity once')
     restored%native_target_nonzero=201_8
     call native_training_draws(restored,1,251,.true.)
     call restored%finish_native_iteration(done)
     call check(restored%adaptation_updates.eq.2,'combined reference uses the same number of grid updates')
     do i=1,19
        forced_random=dble(i)/20d0
        call integ%sample(x,w)
        call restored%sample(y,v)
        call check(all(x.eq.y).and.w.eq.v,'skipped-batch accumulators reproduce one combined training batch exactly')
        call check(x(2).eq.before_map(2,i),'folded coordinate remains fixed across skipped and actual updates')
     enddo
     call integ%native_rates(res,unc,native_moments)
     call restored%native_rates(r2,e2,epoch_moments)
     call check(all(res.eq.r2).and.all(unc.eq.e2).and.all(native_moments.eq.epoch_moments), &
          'batch grouping preserves all signed rates and central moments')
     forced_random=-1d0
  case('native_proposal_group_expiry')
     call integ%start_native_production(2000,1800,[1,2],1d0,1d0,100000_8,initial_nonzero=128_8)
     forced_random=0d0
     do i=1,128
        call integ%native_consider([1d0,1d0],[0.2d0,0.3d0],to_write,done)
     enddo
     call integ%finish_native_iteration(done)
     call check(.not.done.and.integ%native_proposal_id.eq.2,'first substantial batch changes the proposal')
     ! Public batch budgets provide an exact gate-boundary fixture. The
     ! resulting planned-target metadata is not used to validate scheduling.
     integ%native_target_nonzero=32_8
     do i=1,32
        call integ%native_consider([1d0,1d0],[0.2d0,0.3d0],to_write,done)
     enddo
     call integ%finish_native_iteration(done)
     call check(.not.done.and.integ%native_proposal_id.eq.2,'a small batch keeps proposal two')
     integ%native_target_nonzero=1_8
     call integ%native_consider([1d0,1d0],[0.2d0,0.3d0],to_write,done)
     call integ%finish_native_iteration(done)
     call check(.not.done.and.integ%native_proposal_id.eq.3,'pooled batches train the next proposal')
     do iteration=3,9
        integ%native_target_nonzero=integ%ntrials/4_8+1_8
        do i=1,int(integ%native_target_nonzero)
           call integ%native_consider([1d0,1d0],[0.2d0,0.3d0],to_write,done)
        enddo
        call integ%finish_native_iteration(done)
        call check(.not.done.and.integ%native_proposal_id.eq.iteration+1,'substantial batches change proposal identity')
     enddo
     call check(integ%native_proposal_id.eq.10,'the active window will displace the multi-batch second proposal')
     call check(integ%native_effective_generated.eq.dble(integ%ntrials-128_8-33_8), &
          'expiry forecast removes every survivor from both batches of the oldest proposal')
     do
        do i=1,int(integ%native_target_nonzero)
           call integ%native_consider([1d0,1d0],[0.2d0,0.3d0],to_write,done)
        enddo
        call integ%finish_native_iteration(done)
        if (done) exit
     enddo
     call check(integ%overweight.lt.0.01d0,'group-history reserve passes all existing tail checks')
     call integ%native_rates(res,unc,native_moments)
     call check(all(res.eq.1d0).and.all(unc.eq.0d0),'expired proposal groups remain in the integral estimates')
     open(newunit=unit,status='scratch')
     call integ%write_native_pool(unit)
     rewind(unit)
     read(unit,'(a)') line
     read(unit,*) native_trials,native_stored,native_epoch_count
     do i=1,4
        read(unit,'(a)') line
     enddo
     j=0
     last_active=0
     do i=1,native_epoch_count
        read(unit,*) epoch_id,row_n,row_nz,row_target,epoch_moments,row_c,row_m,row_t,eligible,proposal_id
        if (proposal_id.eq.2) then
           j=j+1
           call check(eligible.eq.0.and.row_t.eq.0d0,'both batches of the displaced proposal are ineligible')
        endif
        if (eligible.eq.1) last_active=last_active+1
     enddo
     call check(j.eq.2.and.last_active.eq.8,'eight distinct recent proposals remain after grouped expiry')
     close(unit)
     forced_random=-1d0
  case('native_nonzero')
     call integ%start_native_production(200,180,[2,4],0.5d0,1d0,100000_8)
     call check(integ%native_target_nonzero.eq.1024_8,'native initial nonzero request uses modest minimum')
     do i=1,2048
        weight=0d0
        if (mod(i,2).eq.0) weight=1d0
        call integ%native_consider([weight,weight],[0.2d0,0.3d0],to_write,done)
        if (to_write) call integ%record_candidate_factor(1d0,to_write)
        call check(done.eqv.(i.eq.2048),'complete nonzero target, no accepted-event early stop')
     enddo
     call integ%native_rates(res,unc,native_moments)
     call check(all(res.eq.0.5d0),'all zero and rejected draws included in native rate')
     call check(maxval(abs(unc-sqrt(512d0)/2048d0)).lt.1d-14,'native population errors')
     call check(maxval(abs(native_moments-[0.5d0,0.5d0,512d0,512d0,512d0])).lt.1d-10, &
          'native aggregate central moments include zeros')
     call integ%finish_native_iteration(done)
     call check(done.and.integ%native_iteration.eq.1,'constant target finishes in one native iteration')
     open(newunit=unit,file='native_nonzero_pool.dat',status='replace')
     call integ%write_native_pool(unit)
     close(unit)
  case('native_early_completion')
     call integ%start_native_production(200,180,[1,2],1d0,1d0,10000_8,initial_nonzero=4096_8)
     forced_random=0.5d0
     call integ%sample(first_map,w)
     do i=1,1024
        sgn=1d0
        if (mod(i,2).eq.0) sgn=-1d0
        call integ%native_consider([1d0,sgn],[0.2d0,0.3d0],to_write,done)
        if (to_write) call integ%record_candidate_factor(1d0,to_write)
        call check(.not.done,'completion gate does not change the whole-batch completion flag')
        call check(integ%native_completion_due.eqv.(i.eq.1024),'first completion gate uses nonzero points')
     enddo
     call integ%check_native_completion(done)
     call check(done.and.integ%production_done.and.integ%native_iteration_done,'partial final epoch completes production')
     call check(.not.integ%native_completion_due,'successful completion consumes pending gate')
     call check(integ%native_target_nonzero.eq.4096_8.and.integ%native_nonzero.eq.1024_8, &
          'early completion preserves planned and observed nonzero counts separately')
     call check(integ%native_iteration.eq.1.and.integ%adaptation_updates.eq.0,'completion never creates a grid update')
     call integ%sample(x,v)
     call check(all(x.eq.first_map).and.v.eq.w,'both folded and unfolded maps stay fixed within the epoch')
     call integ%native_rates(res,unc,native_moments)
     call check(maxval(abs(res-[1d0,0d0])).lt.1d-14.and.maxval(abs(unc-[0d0,1d0/32d0])).lt.1d-14, &
          'early rate uses all actual signed observations')
     call check(maxval(abs(native_moments-[1d0,0d0,0d0,1024d0,0d0])).lt.1d-10, &
          'early central moments use the actual point count')
     open(newunit=unit,file='native_early_pool.dat',status='replace')
     call integ%write_native_pool(unit)
     close(unit)
     open(newunit=unit,file='native_early_pool.dat',status='old')
     read(unit,'(a)') line
     call check(trim(line).eq.'MG5_AMPLI_POOL 5','early completion retains the POOL5 format')
     read(unit,*) native_trials,native_stored,native_epoch_count
     call check(native_trials.eq.1024_8.and.native_stored.eq.1024.and.native_epoch_count.eq.1,'early pool actual counts')
     read(unit,*) native_moments
     read(unit,*) generated_target,final_target,native_logz,native_tails
     call check(generated_target.eq.200.and.final_target.eq.180,'early completion preserves reserve and final quotas')
     call check(all(native_tails.eq.0d0),'all existing final tail checks pass')
     read(unit,*) k,native_updates
     read(unit,*) native_mask
     read(unit,*) epoch_id,row_n,row_nz,row_target,epoch_moments,row_c,row_m,row_t,eligible,proposal_id
     call check(row_n.eq.1024_8.and.row_nz.eq.1024_8.and.row_target.eq.4096_8,'POOL5 describes a short final epoch')
     nwritten=0
     do i=1,native_stored
        read(unit,*) epoch_id,weight,target,actual,j,row_factor
        if (actual.gt.0d0) nwritten=nwritten+1
        call check(actual.eq.0d0.or.actual.eq.1d0,'early constant target needs no weight correction')
     enddo
     close(unit)
     call check(nwritten.eq.200,'early finalization retains the exact reserve')
  case('native_failed_probe_neutral')
     call train(integ,2048)
     open(newunit=unit,status='scratch')
     call integ%save(unit)
     rewind(unit)
     call restored%load(unit,2,2)
     close(unit)
     do i=1,19
        forced_random=dble(i)/20d0
        call integ%sample(before_map(:,i),before_jac(i))
     enddo
     forced_random=-1d0
     saved_rng=rng_state
     call integ%start_native_production(200,180,[1,2],1d0,1d0,100000_8,initial_nonzero=4096_8)
     call failed_probe_batch(integ,.true.)
     call integ%native_rates(res,unc,native_moments)
     native_trials=integ%ntrials
     rng_state=saved_rng
     call restored%start_native_production(200,180,[1,2],1d0,1d0,100000_8,initial_nonzero=4096_8)
     call failed_probe_batch(restored,.false.)
     call restored%native_rates(r2,e2,epoch_moments)
     call check(all(res.eq.r2).and.all(unc.eq.e2).and.all(native_moments.eq.epoch_moments), &
          'failed probes preserve the rates and central moments of an unprobed replay')
     call check(integ%ntrials.eq.restored%ntrials.and.integ%ncandidates.eq.restored%ncandidates, &
          'failed probes preserve the trial and candidate histories')
     call check(integ%native_target_nonzero.eq.restored%native_target_nonzero, &
          'failed probes preserve the next whole-batch forecast')
     call check(integ%native_iteration.eq.2.and.restored%native_iteration.eq.2.and. &
          integ%adaptation_updates.eq.1.and.restored%adaptation_updates.eq.1,'only a whole batch advances the epoch')
     do i=1,19
        forced_random=dble(i)/20d0
        call integ%sample(x,w)
        call restored%sample(y,v)
        call check(all(x.eq.y).and.w.eq.v,'failed probes preserve next adaptation and all accumulated training data')
        call check(x(2).eq.before_map(2,i),'failed probes preserve folded coordinate maps')
     enddo
     forced_random=-1d0
  case('native_probe_latest_factor')
     call integ%start_native_production(200,180,[2,4],1d0,1d0,10000_8,initial_nonzero=4096_8)
     call restored%init(2,2)
     call restored%start_native_production(200,180,[2,4],1d0,1d0,10000_8,initial_nonzero=4096_8)
     forced_random=0.9999d0
     do i=1,1024
        weight=1d0
        if (i.eq.1024) weight=1.001d0
        call integ%native_consider([weight,weight],[0.2d0,0.3d0],to_write,done)
        call check(to_write,'latest-factor comparison retains every observation')
        actual=1d0
        if (i.eq.1024) actual=100d0
        call integ%record_candidate_factor(actual,to_write)
        call restored%native_consider([weight,weight],[0.2d0,0.3d0],to_write,done)
        call restored%record_candidate_factor(1d0,to_write)
     enddo
     call check(integ%native_completion_due.and.restored%native_completion_due,'latest LHE factor precedes pending check')
     call integ%check_native_completion(done)
     call check(.not.done.and..not.integ%production_done,'latest actual LHE factor prevents premature completion')
     call check(integ%full_trial_tail.lt.0.01d0.and.integ%reserve_tail.gt.0.01d0.and. &
          integ%worst_subset_tail.gt.0.01d0,'collection tail checks remain authoritative after a probe')
     call restored%check_native_completion(done)
     call check(done.and.restored%overweight.lt.0.01d0,'the same raw pool passes with unit LHE factors')
  case('native_probe_history')
     call integ%start_native_production(50000,45000,[2,4],1d0,1d0,1000000_8,initial_nonzero=128_8)
     forced_random=0.5d0
     do iteration=1,7
        do i=1,int(integ%native_target_nonzero)
           call integ%native_consider([1d0,1d0],[0.2d0,0.3d0],to_write,done)
        enddo
        call integ%finish_native_iteration(done)
        call check(.not.done,'history test fills seven batches of one frozen proposal')
     enddo
     call check(integ%native_iteration.eq.8,'history probe starts at eighth epoch')
     do i=1,2048
        call integ%native_consider([1d0,1d0],[0.2d0,0.3d0],to_write,done)
        if (mod(i,1024).eq.0) then
           saved_rng=rng_state
           call integ%check_native_completion(done)
           call check(.not.done.and.integ%native_iteration.eq.8.and.integ%adaptation_updates.eq.0, &
                'repeated probes retain the current frozen proposal epoch')
           call check(rng_state.eq.saved_rng,'history probes cannot consume randomness')
        endif
     enddo
     do i=2049,int(integ%native_target_nonzero)
        call integ%native_consider([1d0,1d0],[0.2d0,0.3d0],to_write,done)
     enddo
     call integ%finish_native_iteration(done)
     call check(integ%native_effective_generated.eq.32640d0.and.integ%native_expected_remaining.eq.17360d0, &
          'same-proposal probes preserve every candidate when forecasting the ninth batch')
     do i=1,int(integ%native_target_nonzero)
        call integ%native_consider([1d0,1d0],[0.2d0,0.3d0],to_write,done)
     enddo
     call integ%finish_native_iteration(done)
     call check(done,'history probe replay retains enough events to finish normally')
     open(newunit=unit,status='scratch')
     call integ%write_native_pool(unit)
     rewind(unit)
     do i=1,6
        read(unit,'(a)') line
     enddo
     last_active=0
     do i=1,9
        read(unit,*) epoch_id,row_n,row_nz,row_target,epoch_moments,row_c,row_m,row_t,eligible,proposal_id
        call check(eligible.eq.1.and.proposal_id.eq.1,'all frozen-proposal epochs remain eligible')
        last_active=last_active+eligible
     enddo
     close(unit)
     call check(last_active.eq.9,'completion probes retain more than eight batches of one proposal')
  case('native_probe_analytic_seeds')
     ! Independent seeds compare completion probes with the original whole-
     ! batch API on a signed density with a narrow high-weight region. This
     ! is a modest regression diagnostic, not a physics coverage benchmark.
     probe_statistics: block
       integer :: mode,seed
       real(kind=8) :: group_rates(2,2),group_variances(2,2),group_shapes(2,2)
       real(kind=8) :: rates_one(2),errors_one(2),shapes_one(2),truth(2),shape_truth(2),base
       integer(kind=8) :: group_trials(2),trials_one
       group_rates=0d0
       group_variances=0d0
       group_shapes=0d0
       group_trials=0_8
       base=0.1d0+(1d0-exp(-30d0))/30d0
       truth=[1.5d0*base,0.25d0*base]
       shape_truth=[(0.01d0+(1d0-exp(-3d0))/30d0)/base,1d0/6d0]
       do mode=1,2
          do seed=1,16
             rng_state=1000003_8+7919_8*seed+104729_8*mode
             call native_analytic_probe_run(mode.eq.2,rates_one,errors_one,shapes_one,trials_one)
             group_rates(:,mode)=group_rates(:,mode)+rates_one
             group_variances(:,mode)=group_variances(:,mode)+errors_one**2
             group_shapes(:,mode)=group_shapes(:,mode)+shapes_one
             group_trials(mode)=group_trials(mode)+trials_one
          enddo
          call check(all(abs(group_rates(:,mode)-16d0*truth).lt.6d0*sqrt(group_variances(:,mode))), &
               'independent-seed means agree with both analytic signed and absolute integrals')
          call check(all(abs(group_shapes(:,mode)/16d0-shape_truth).lt.0.035d0), &
               'independent-seed corrected events reproduce the narrow-region and signed event densities')
       enddo
       call check(all(abs(group_rates(:,1)-group_rates(:,2)).lt. &
            6d0*sqrt(group_variances(:,1)+group_variances(:,2))), &
            'probe and whole-batch rates agree within their independent-seed uncertainties')
       call check(all(abs(group_shapes(:,1)-group_shapes(:,2))/16d0.lt.0.04d0), &
            'probe and whole-batch signed event densities agree')
       call check(group_trials(2).lt.group_trials(1),'analytic probe regression exercises earlier final completion')
     end block probe_statistics
  case('native_zero_last_success')
     call integ%start_native_production(10,9,[2,4],1d0,1d0,2000_8,initial_nonzero=1024_8)
     forced_random=0.5d0
     do i=1,2000
        weight=0d0
        if (i.ge.1990) weight=1d0
        call integ%native_consider([weight,weight],[0.2d0,0.3d0],to_write,done)
        if (to_write) call integ%record_candidate_factor(1d0,to_write)
        call check(.not.done,'zero-heavy last trial does not finish the full nonzero budget')
        call check(integ%native_completion_due.eqv.(i.eq.2000),'trial ceiling schedules a terminal completion check')
     enddo
     call integ%check_native_completion(done)
     call check(done.and.integ%ntrials.eq.2000_8.and.integ%native_nonzero.eq.11_8, &
          'last permitted trial can complete a short final epoch')
     call integ%native_rates(res,unc,native_moments)
     expected=11d0/2000d0
     call check(maxval(abs(res-expected)).lt.1d-15,'terminal completion includes every zero trial in the rate')
     call check(maxval(abs(unc-sqrt(11d0*(1d0-expected))/2000d0)).lt.1d-15, &
          'terminal completion uses actual zero-heavy central moments')
     open(newunit=unit,status='scratch')
     call integ%write_native_pool(unit)
     close(unit)
  case('native_zero_last_failure')
     call integ%start_native_production(10,9,[2,4],1d0,1d0,2000_8,initial_nonzero=1024_8)
     forced_random=0.5d0
     do i=1,2000
        weight=0d0
        if (i.ge.1991) weight=1d0
        call integ%native_consider([weight,weight],[0.2d0,0.3d0],to_write,done)
        if (to_write) call integ%record_candidate_factor(1d0,to_write)
     enddo
     call check(integ%native_completion_due,'terminal failure still waits for the actual candidate factor')
     call integ%check_native_completion(done)
  case('native_trial_overrun')
     call integ%start_native_production(10,9,[2,4],1d0,1d0,10_8,initial_nonzero=2_8)
     do i=1,11
        call integ%native_consider([0d0,0d0],[0.2d0,0.3d0],to_write,done)
     enddo
  case('native_no_bin_growth')
     call train(integ,2048)
     call checkpoint_fill_sizes(integ,fill_before)
     call check(all(fill_before.eq.16),'native production inherits early survey resolution')
     do i=1,19
        forced_random=dble(i)/20d0
        call integ%sample(before_map(:,i),before_jac(i))
     enddo
     forced_random=-1d0
     call integ%start_native_production(100000,90000,[1,2],1d12,1d15,200000_8,initial_nonzero=65536_8)
     do i=1,65536
        call integ%sample(x,w)
        weight=(1d0+9d0*x(1)**2)*w
        call integ%native_consider([weight,weight],x,to_write,done)
     enddo
     call integ%finish_native_iteration(done)
     call check(.not.done.and.integ%adaptation_updates.eq.1,'large native batch adapts before the next epoch')
     do i=1,19
        forced_random=dble(i)/20d0
        call integ%sample(after_map(:,i),before_jac(i))
     enddo
     call check(all(after_map(2,:).eq.before_map(2,:)),'native folded coordinate remains bitwise fixed')
     call check(maxval(abs(after_map(1,:)-before_map(1,:))).gt.1d-3,'native unfolded coordinate still adapts')
     ! Native checkpoints are intentionally unsupported. Reset only production
     ! metadata so the existing checkpoint reader can inspect preserved maps.
     call integ%start_pool(0_8,1d0)
     do i=1,19
        forced_random=dble(i)/20d0
        call integ%sample(x,w)
        call check(all(x.eq.after_map(:,i)).and.w.eq.before_jac(i),'native metadata reset preserves maps')
     enddo
     forced_random=-1d0
     call checkpoint_fill_sizes(integ,fill_after)
     call check(all(fill_after.eq.fill_before),'large native batch cannot increase fill resolution')
  case('native_huge_survey_maximum')
     call integ%start_native_production(2500,2250,[2,4],1d0,1d100,1000000_8)
     call check(integ%native_target_nonzero.eq.2500_8,'huge survey maximum cannot inflate the first batch')
     call check(integ%envelope.eq.1d100,'saved maximum remains the initial envelope estimate')
     forced_random=0.5d0
     do i=1,2500
        call integ%native_consider([1d0,1d0],[0.2d0,0.3d0],to_write,done)
        call check(to_write,'storage cutoff retains candidates despite enormous surveyed maximum')
     enddo
     call integ%finish_native_iteration(done)
     call check(.not.done,'exact quota still requires an excluded rank for finalization')
     call check(integ%native_target_nonzero.eq.1024_8,'near-quota forecast shrinks rather than forcing doubling')
     do
        do
           call integ%native_consider([1d0,1d0],[0.2d0,0.3d0],to_write,done)
           if (done) exit
        enddo
        call integ%finish_native_iteration(done)
        if (done) exit
     enddo
     call check(integ%ntrials.lt.10000_8,'huge surveyed maximum does not delay production adaptation')
     call check(integ%overweight.eq.0d0.and.integ%quota_complete,'modest iterations preserve exact quota and tail checks')
     call integ%native_rates(res,unc,native_moments)
     call check(all(res.eq.1d0).and.all(unc.eq.0d0),'all exploratory production observations remain in rates')
     open(newunit=unit,status='scratch')
     call integ%write_native_pool(unit)
     rewind(unit)
     do i=1,6
        read(unit,'(a)') line
     enddo
     read(unit,*) epoch_id,row_n,row_nz,row_target,epoch_moments,row_c,row_m,row_t,eligible,proposal_id
     call check(row_m.eq.1d0.and.row_c.eq.1d0,'mature first proposal uses the candidate quantile independently of survey maximum')
     read(unit,*) epoch_id,row_n,row_nz,row_target,epoch_moments,row_c,row_m,row_t,eligible,proposal_id
     call check(row_m.eq.1d0.and.proposal_id.eq.1, &
          'later identical proposals reuse the same mature relative quantile')
     close(unit)
  case('native_relative_sparse_boundary')
     forced_random=0.5d0
     do nlarge=200,201
        call integ%start_native_production(20,18,[2,4],1d0,1000d0,10000_8,initial_nonzero=4096_8)
        do i=1,nlarge
           call integ%native_consider([1d0,1d0],[0.2d0,0.3d0],to_write,done)
           call check(to_write,'bootstrap boundary retains all prescribed candidates')
        enddo
        saved_rng=rng_state
        call integ%check_native_completion(done)
        call check(done.and.rng_state.eq.saved_rng,'bootstrap-boundary completion uses no RNG')
        open(newunit=unit,status='scratch')
        call integ%write_native_pool(unit)
        rewind(unit)
        do i=1,6
           read(unit,'(a)') line
        enddo
        read(unit,*) epoch_id,row_n,row_nz,row_target,epoch_moments,row_c,row_m,row_t,eligible,proposal_id
        close(unit)
        if (nlarge.eq.200) then
           call check(row_m.eq.1000d0,'sparse relative envelope preserves survey maximum')
        else
           call check(row_m.eq.1d0,'mature relative envelope releases initial survey floor')
        endif
        call check(abs(row_t-2d0).lt.1d-12,'common scale cancellation preserves actual threshold')
        call check(integ%full_trial_tail.eq.0d0.and.integ%reserve_tail.eq.0d0, &
             'bootstrap scale transition leaves a constant target unweighted')
     enddo
  case('native_relative_retained_outlier')
     forced_random=0.9d0
     do quota_value=100,900,800
        call integ%start_native_production(quota_value,quota_value*9/10,[2,4],1d0,1000d0,10000_8, &
             initial_nonzero=4096_8)
        do i=1,1024
           weight=1d0
           sgn=1d0
           if (i.eq.1024) then
              weight=5d0
              sgn=-1d0
           endif
           call integ%native_consider([weight,sgn*weight],[0.2d0,0.3d0],to_write,done)
           call check(to_write,'relative-quantile fixture keeps every candidate including outlier')
        enddo
        call integ%native_rates(res,unc,native_moments)
        saved_rng=rng_state
        call integ%check_native_completion(done)
        call check(rng_state.eq.saved_rng,'quantile calculation leaves RNG unchanged')
        call integ%native_rates(r2,e2,epoch_moments)
        call check(all(res.eq.r2).and.all(unc.eq.e2).and.all(native_moments.eq.epoch_moments), &
             'relative scale recomputation leaves every rate moment unchanged')
        call check(abs(integ%full_trial_tail-5d0/1028d0).lt.1d-14, &
             'outlier contributes its whole observed weight above the quantile')
        if (quota_value.eq.100) then
           call check(.not.done.and.integ%reserve_tail.gt.0.01d0.and.integ%worst_subset_tail.gt.0.01d0, &
                'robust relative scale cannot hide an outlier that fails final-sample tail checks')
        else
           call check(done.and.integ%overweight.lt.0.01d0,'sufficient reserve retains outlier correction safely')
           open(newunit=unit,status='scratch')
           call integ%write_native_pool(unit)
           rewind(unit)
           do i=1,6
              read(unit,'(a)') line
           enddo
           read(unit,*) epoch_id,row_n,row_nz,row_target,epoch_moments,row_c,row_m,row_t,eligible,proposal_id
           call check(row_m.eq.1d0,'a single large outlier cannot control the mature relative quantile')
           actual=0d0
           do i=1,1024
              read(unit,*) epoch_id,weight,target,row_correction(i),j,row_factor
              if (i.lt.1024.and.row_correction(i).gt.0d0) actual=row_correction(i)
              if (i.eq.1024) then
                 call check(weight.eq.5d0.and.j.eq.1.and.row_correction(i).gt.0d0, &
                      'serialized outlier remains selected and tagged with original weight')
                 call check(abs(row_correction(i)/actual-4.5d0).lt.1d-12, &
                      'outlier native correction includes its full weight, not only excess')
              endif
           enddo
           close(unit)
        endif
     enddo
  case('native_small_trial_budget')
     call integ%start_native_production(10,9,[2,4],1d0,1d100,40_8)
     call check(integ%native_target_nonzero.eq.40_8,'default initial request respects a smaller trial ceiling')
     forced_random=0.5d0
     do i=1,40
        call integ%native_consider([1d0,1d0],[0.2d0,0.3d0],to_write,done)
     enddo
     call integ%finish_native_iteration(done)
     call check(done.and.integ%ntrials.eq.40_8,'small permitted run completes without expanding safety budget')
  case('native_sparse_cutoff')
     call integ%start_native_production(20,18,[2,4],1d12,1d15,100000_8)
     forced_random=0.5d0
     do i=1,1024
        call integ%native_consider([1d0,1d0],[0.2d0,0.3d0],to_write,done)
        call check(.not.to_write,'overestimated survey produces no retained pilot candidates')
     enddo
     call integ%finish_native_iteration(done)
     call check(.not.done.and.integ%native_target_nonzero.eq.2048_8,'zero efficiency uses bounded growth')
     forced_random=0.25d0
     do i=1,2048
        call integ%native_consider([1d0,1d0],[0.2d0,0.3d0],to_write,done)
        call check(to_write,'all-trial statistics lower the sparse-pool storage cutoff')
     enddo
     call integ%finish_native_iteration(done)
     call check(done.and.integ%native_iteration.eq.2,'cutoff recovery needs no minimum candidate count')
     call check(integ%overweight.eq.0d0,'sparse-pool recovery preserves tail protection')
  case('native_growth_ceiling')
     call integ%start_native_production(5000,4500,[2,4],1d0,1d0,10000000_8,initial_nonzero=128_8)
     forced_random=0.5d0
     do i=1,128
        weight=0.01d0
        if (i.eq.1) weight=1d0
        call integ%native_consider([weight,weight],[0.2d0,0.3d0],to_write,done)
     enddo
     call integ%finish_native_iteration(done)
     call check(.not.done.and.integ%native_generation_efficiency.gt.0d0,'small measured nonzero acceptance')
     call check(integ%native_expected_remaining/integ%native_generation_efficiency.gt.500000d0, &
          'unbounded efficiency forecast would postpone the next update')
     call check(integ%native_target_nonzero.eq.256_8,'efficiency forecast cannot exceed doubling ceiling')
  case('native_expiring_budget')
     call integ%start_native_production(50000,45000,[2,4],1d0,1d0,1000000_8)
     call check(integ%native_target_nonzero.eq.8192_8,'large quota still starts with a modest bounded batch')
     call integ%start_native_production(50000,45000,[2,4],1d0,1d0,1000000_8,initial_nonzero=128_8)
     forced_random=0.5d0
     do iteration=1,8
        call check(integ%native_iteration.eq.iteration,'large sample advances through adaptation epochs')
        saved_rng=integ%native_target_nonzero
        do i=1,int(saved_rng)
           call integ%native_consider([1d0,1d0],[0.2d0,0.3d0],to_write,done)
        enddo
        call integ%finish_native_iteration(done)
        call check(.not.done,'large sample still needs events before ninth epoch')
        call check(integ%native_target_nonzero.le.2_8*saved_rng,'every forecast obeys growth ceiling')
     enddo
     call check(integ%native_effective_generated.eq.32640d0,'unchanged proposal keeps the oldest 128 events')
     call check(integ%native_expected_remaining.eq.17360d0,'unchanged proposal requires no replacement events')
     call check(integ%native_target_nonzero.gt.8192_8,'no fixed batch cap prevents large-quota completion')
     saved_rng=integ%native_target_nonzero
     do i=1,int(saved_rng)
        call integ%native_consider([1d0,1d0],[0.2d0,0.3d0],to_write,done)
     enddo
     call integ%finish_native_iteration(done)
     call check(done.and.integ%native_iteration.eq.9,'large quota completes with all batches of its frozen proposal')
     call integ%native_rates(res,unc,native_moments)
     call check(all(res.eq.1d0).and.all(unc.eq.0d0),'all retained proposal batches enter integration estimates')
  case('native_tail_reevaluation')
     call integ%start_native_production(100,90,[2,4],1d0,1d0,10000_8,initial_nonzero=150_8)
     forced_random=0.5d0
     do i=1,150
        weight=1d0
        if (i.eq.1) weight=2.001d0
        call integ%native_consider([weight,weight],[0.2d0,0.3d0],to_write,done)
     enddo
     call check(done,'first native nonzero iteration complete')
     call integ%finish_native_iteration(done)
     call check(.not.done.and.integ%overweight.gt.0.01d0,'native full tail rejects negligible excess')
     ! No threshold below 2.001 can satisfy the observed full-tail bound.
     ! At that threshold only the outlier survives: ordinary priorities are 2.
     call check(integ%native_effective_generated.eq.1d0,'forecast counts actual tail-safe priority survivors')
     call check(integ%native_expected_remaining.eq.99d0,'tail-safe capacity determines remaining event demand')
     call check(abs(integ%native_generation_efficiency-1d0/150d0).lt.1d-14, &
          'acceptance uses the forecast threshold and current nonzero trials')
     call check(integ%ncandidates.eq.150,'forecast preserves all stored candidates')
     call check(abs(integ%full_trial_tail-2.001d0/151.001d0).lt.1d-12, &
          'forecast leaves final full-trial diagnostic at the actual rank threshold')
     call check(abs(integ%reserve_tail-1.0005d0/100.0005d0).lt.1d-12.and. &
          abs(integ%worst_subset_tail-1.0005d0/90.0005d0).lt.1d-12, &
          'forecast leaves final reserve and collection diagnostics unchanged')
     call check(integ%native_target_nonzero.eq.300_8,'small explicit initial request grows within its doubling ceiling')
     forced_random=0.1d0
     do i=1,300
        call integ%native_consider([1d0,1d0],[0.2d0,0.3d0],to_write,done)
     enddo
     call integ%finish_native_iteration(done)
     call check(done.and.integ%overweight.eq.0d0,'later iteration rethresholds earlier overweight')
     call integ%native_rates(res,unc,native_moments)
     call check(abs(res(1)-(450d0+1.001d0)/450d0).lt.1d-13,'evolving rate includes both iterations')
  case('native_forecast_below_quota')
     call integ%start_native_production(100,90,[2,4],1d0,1d0,10000_8,initial_nonzero=80_8)
     forced_random=0.5d0
     do i=1,80
        weight=1d0
        if (i.eq.1) weight=2.001d0
        call integ%native_consider([weight,weight],[0.2d0,0.3d0],to_write,done)
     enddo
     call integ%finish_native_iteration(done)
     call check(.not.done.and..not.integ%quota_complete,'incomplete reserve still needs final rank selection')
     call check(integ%native_effective_generated.eq.1d0.and.integ%native_expected_remaining.eq.99d0, &
          'tail-safe forecasting applies before the raw candidate count reaches quota')
     call check(abs(integ%native_generation_efficiency-1d0/80d0).lt.1d-14, &
          'below-quota forecast excludes ordinary candidates lost at safe threshold')
     call check(integ%full_trial_tail.eq.1d0.and.integ%reserve_tail.eq.1d0.and. &
          integ%worst_subset_tail.eq.1d0.and.integ%overweight.eq.huge(1d0), &
          'forecast cannot replace incomplete final-check sentinels')
  case('native_forecast_all_trials')
     do rep=0,1
        call integ%start_native_production(1000,900,[2,4],1d0,1d0,10000_8, &
             initial_nonzero=int(101+1000*rep,8))
        forced_random=0.9d0
        do i=1,1000*rep
           call integ%native_consider([0.5d0,0.5d0],[0.2d0,0.3d0],to_write,done)
           call check(.not.to_write,'rejected nonzero trials remain absent from the event pool')
        enddo
        do i=1,500*rep
           call integ%native_consider([0d0,0d0],[0.2d0,0.3d0],to_write,done)
           call check(.not.to_write.and..not.done,'zero trials do not finish the nonzero budget')
        enddo
        forced_random=0.5d0
        do i=1,101
           weight=1d0
           if (i.eq.1) weight=2.1d0
           call integ%native_consider([weight,weight],[0.2d0,0.3d0],to_write,done)
           if (i.eq.1) call integ%record_candidate_factor(0.1d0,to_write)
        enddo
        call integ%finish_native_iteration(done)
        ! The small tail LHE factor makes the collection bound harmless.
        ! Rejected mass lowers 2.1/102.1 to 2.1/602.1, allowing the floor
        ! threshold (1) instead of 2.1. This changes safe capacity 1 -> 101.
        expected=dble(1+100*rep)
        call check(.not.done.and.integ%ncandidates.eq.101,'forecast cannot manufacture retained candidates')
        call check(integ%native_effective_generated.eq.expected,'full-tail forecast includes rejected trial mass')
        call check(abs(integ%native_generation_efficiency-expected/dble(101+1000*rep)).lt.1d-14, &
             'current acceptance counts nonzero trials including storage rejections')
        call check(integ%ntrials.eq.int(101+1500*rep,8),'zero observations remain in all-trial statistics')
        call integ%native_rates(res,unc,native_moments)
        call check(abs(res(1)-dble(102.1d0+500*rep)/dble(101+1500*rep)).lt.1d-13, &
             'forecast preserves rates from rejected and zero observations')
     enddo
  case('native_forecast_factors')
     do rep=1,4
        call integ%start_native_production(250,200,[2,4],1d0,1d0,10000_8,initial_nonzero=1201_8)
        target=1d0
        if (rep.eq.3) target=1d-250
        if (rep.eq.4) target=1d250
        forced_random=0.9d0
        do i=1,1000
           call integ%native_consider([0.5d0,0.5d0],[0.2d0,0.3d0],to_write,done)
        enddo
        do i=1,201
           forced_random=0.5d0
           weight=1d0
           if (i.eq.1) weight=2.1d0
           if (i.eq.2) forced_random=1d0/2.095d0
           call integ%native_consider([weight,weight],[0.2d0,0.3d0],to_write,done)
           row_factor=target
           if (i.eq.1.and.mod(rep,2).eq.0) row_factor=2d0*target
           call integ%record_candidate_factor(row_factor,to_write)
        enddo
        call integ%finish_native_iteration(done)
        ! Full-trial tail is safe already at threshold 1. For factor 2,
        ! collection needs T > 4.2*99/199 = 2.08945..., so just the outlier
        ! and the candidate of priority 2.095 survive. Raising T all the way
        ! to the maximum raw weight 2.1 would incorrectly discard the latter.
        ! Common factor rescalings must leave both capacities unchanged.
        expected=201d0
        if (mod(rep,2).eq.0) expected=2d0
        call check(.not.done.and..not.integ%quota_complete,'factor-aware forecast never finalizes incomplete reserves')
        call check(integ%native_effective_generated.eq.expected,'collection forecast uses actual LHE factors')
        call check(abs(integ%native_generation_efficiency-expected/1201d0).lt.1d-14, &
             'tail-safe acceptance retains allowed overweight corrections')
     enddo
  case('native_forecast_ties')
     call integ%start_native_production(100,90,[2,4],1d0,1d0,10000_8,initial_nonzero=150_8)
     forced_random=0.5d0
     do i=1,150
        weight=1d0
        if (i.le.3) weight=3d0
        call integ%native_consider([weight,weight],[0.2d0,0.3d0],to_write,done)
     enddo
     call integ%finish_native_iteration(done)
     call check(.not.done.and.integ%native_effective_generated.eq.3d0, &
          'equal raw tail weights leave the forecast tail together at their common threshold')
     call check(abs(integ%full_trial_tail-9d0/156d0).lt.1d-12, &
          'tied-weight forecast preserves the failed final full-tail diagnostic')
  case('native_recent_history')
     call integ%start_native_production(100,90,[2,4],1d0,1d0,100000_8,initial_nonzero=2_8)
     forced_random=0.5d0
     do iteration=1,9
        do i=1,int(integ%native_target_nonzero)
           weight=1d0
           if (integ%ntrials.eq.0_8) weight=100d0
           call integ%native_consider([weight,weight],[0.2d0,0.3d0],to_write,done)
        enddo
        call integ%finish_native_iteration(done)
        call check(.not.done,'a known overweight cannot disappear after eight frozen-proposal batches')
        call check(integ%native_proposal_id.eq.1.and.integ%adaptation_updates.eq.0, &
             'all-folded batches retain their original proposal identity')
     enddo
     call check(integ%full_trial_tail.gt.0.01d0.and.integ%worst_subset_tail.gt.0.01d0, &
          'ninth-batch tail checks still contain the original overweight event')
     ! Genuine new rejection uniforms raise the rank threshold enough to
     ! finish; the old outlier is still checked rather than silently expired.
     forced_random=0.001d0
     do i=1,int(integ%native_target_nonzero)
        call integ%native_consider([1d0,1d0],[0.2d0,0.3d0],to_write,done)
     enddo
     call integ%finish_native_iteration(done)
     call check(done.and.integ%native_iteration.eq.10,'new draws complete the unchanged-proposal reserve')
     call check(integ%overweight.eq.0d0,'all existing tail checks pass after genuine additional sampling')
     call integ%native_rates(res,unc,native_moments)
     call check(abs(res(1)-(1d0+99d0/dble(integ%ntrials))).lt.1d-12,'every batch remains in the rate estimate')
     open(newunit=unit,file='native_recent_pool.dat',status='replace')
     call integ%write_native_pool(unit)
     close(unit)
     open(newunit=unit,file='native_recent_pool.dat',status='old')
     do i=1,6
        read(unit,'(a)') line
     enddo
     last_active=0
     do i=1,10
        read(unit,*) epoch_id,row_n,row_nz,row_target,epoch_moments,row_c,row_m,row_t,eligible,proposal_id
        call check(eligible.eq.1.and.proposal_id.eq.1,'every event-bearing batch of the same proposal remains useful')
        if (i.eq.1) call check(row_t.ge.100d0,'final threshold explicitly covers the original overweight')
        last_active=last_active+eligible
     enddo
     call check(last_active.eq.10,'ten frozen batches fit in one proposal-history slot')
     read(unit,*) epoch_id,weight,target,actual,j,row_factor
     call check(epoch_id.eq.1.and.weight.eq.100d0.and.j.eq.0,'the original candidate remains in exported history')
     close(unit)
  case('native_analytic_history')
     ! Nontrivial signed density and frozen second-coordinate proposal.
     forced_random=0.271828d0
     call integ%sample(first_map,w)
     forced_random=-1d0
     base_native: block
       real(kind=8) :: integral_base
       integral_base=0.1d0+(1d0-exp(-30d0))/30d0
       call integ%start_native_production(4000,3600,[1,2],1.5d0*integral_base, &
            1.5d0*integral_base,1000000_8,initial_nonzero=1024_8)
       nwritten=0
       do
          do
             call integ%sample(x,w)
             weight=(0.1d0+exp(-30d0*x(1)))*(1d0+x(2))*w
             sgn=1d0
             if (x(2).lt.0.5d0) sgn=-1d0
             call integ%native_consider([weight,weight*sgn],x,to_write,done)
             if (to_write) then
                nwritten=nwritten+1
                call check(nwritten.le.100000,'native analytic candidate buffer')
                expected_weights(nwritten)=weight
                expected_priorities(nwritten)=0d0
                candidate_x(:,nwritten)=x
                candidate_jac(nwritten)=w
                row_epoch(nwritten)=integ%native_iteration
                observables(nwritten)=0d0
                if (x(1).lt.0.1d0) observables(nwritten)=1d0
                event_signs(nwritten)=sgn
                call integ%record_candidate_factor(1d0,to_write)
             endif
             if (done) exit
          enddo
          call integ%finish_native_iteration(done)
          if (done) exit
       enddo
       call check(integ%native_iteration.gt.1.and.integ%adaptation_updates.gt.0,'native analytic test adapts between iterations')
       call integ%native_rates(res,unc,native_moments)
       call check(abs(res(1)-1.5d0*integral_base).lt.6d0*unc(1),'native evolving absolute integral')
       call check(abs(res(2)-0.25d0*integral_base).lt.6d0*unc(2),'native evolving signed integral')
       forced_random=0.271828d0
       call integ%sample(x,w)
       call check(x(2).eq.first_map(2),'native folded proposal map remains bitwise frozen')
       forced_random=-1d0
       open(newunit=unit,file='native_analytic_pool.dat',status='replace')
       call integ%write_native_pool(unit)
       close(unit)
       open(newunit=unit,file='native_analytic_pool.dat',status='old')
       read(unit,'(a)') line
       call check(trim(line).eq.'MG5_AMPLI_POOL 5','native metadata version')
       read(unit,*) native_trials,native_stored,native_epoch_count
       call check(native_stored.eq.nwritten,'native LHE candidate labels stable')
       read(unit,*) native_moments
       read(unit,*) generated_target,final_target,native_logz,native_tails
       call check(all(native_tails.lt.0.01d0),'all native full-tail criteria satisfied')
       read(unit,*) k,native_updates
       read(unit,*) native_mask
       call check(all(native_mask.eq.[1,0]),'native adaptation mask exported')
       do i=1,native_epoch_count
          read(unit,*) epoch_id,row_n,row_nz,row_target,epoch_moments,row_c,row_m,row_t,eligible,proposal_id
          call check(row_nz.eq.row_target,'every native epoch completes requested nonzero budget')
          if (eligible.eq.1) call check(row_t.ge.row_c,'native thresholds never reopen rejected candidates')
          last_envelope=row_m
       enddo
       do i=1,nwritten
          read(unit,*) epoch_id,weight,target,row_correction(i),j,row_factor
          call check(epoch_id.eq.row_epoch(i).and.weight.eq.expected_weights(i),'native birth epoch and weight immutable')
       enddo
       close(unit)
       call check(count(row_correction(1:nwritten).gt.0d0).eq.4000,'native exact event reserve')
       expected=(0.01d0+(1d0-exp(-3d0))/30d0)/integral_base
       actual=sum(row_correction(1:nwritten)*observables(1:nwritten))/4000d0
       call check(abs(actual-expected).lt.0.04d0,'native final event density agrees with analytic density')
       signed_mean=sum(row_correction(1:nwritten)*event_signs(1:nwritten))/4000d0
       call check(abs(signed_mean-1d0/6d0).lt.0.05d0,'native signed event density')
       ! Independently reconstruct the final epoch envelope at every saved
       ! physical coordinate by inverting the current 1D map. The frozen
       ! second map has unit Jacobian, so the returned volume is J_unfolded.
       expected_envelope=0d0
       do i=1,nwritten
          lo=0d0
          hi=1d0
          do j=1,55
             mid=(lo+hi)/2d0
             forced_random=mid
             call integ%sample(y,newjac)
             if (y(1).lt.candidate_x(1,i)) then
                lo=mid
             else
                hi=mid
             endif
          enddo
          expected_priorities(i)=expected_weights(i)*newjac/candidate_jac(i)
       enddo
       ! Independent repeated-maximum reference, not the integrator heap code.
       do j=1,max(int(0.05d0*nwritten),1)
          k=maxloc(expected_priorities(1:nwritten),dim=1)
          expected_envelope=expected_priorities(k)
          expected_priorities(k)=-1d0
       enddo
       call check(abs(last_envelope-expected_envelope).lt.1d-8*expected_envelope, &
            'historical relative quantile uses candidate coordinates and birth Jacobians')
     end block base_native
  case('native_zero')
     call integ%start_native_production(0,0,[1,2],0d0,1d0,0_8)
     call check(integ%production_done.and.integ%quota_complete.and.integ%native_iteration.eq.0,'empty native quota')
     call integ%native_rates(res,unc,native_moments)
     call check(all(native_moments.eq.0d0),'empty native rate moments')
     open(newunit=unit,status='scratch')
     call integ%write_native_pool(unit)
     close(unit)
  case('native_incomplete_finish')
     call integ%start_native_production(10,9,[1,2],1d0,1d0,10000_8)
     call integ%finish_native_iteration(done)
  case('native_zero_safety')
     call integ%start_native_production(10,9,[1,2],1d0,1d0,10_8,initial_nonzero=2_8)
     do i=1,10
        call integ%native_consider([0d0,0d0],[0.2d0,0.3d0],to_write,done)
     enddo
     call integ%check_native_completion(done)
  case('native_iteration_safety')
     call integ%start_native_production(100,90,[1,2],1d0,1d0,10000_8,max_iterations=1,initial_nonzero=2_8)
     call integ%native_consider([1d0,1d0],[0.2d0,0.3d0],to_write,done)
     call integ%native_consider([1d0,1d0],[0.2d0,0.3d0],to_write,done)
     call integ%finish_native_iteration(done)
  case('native_checkpoint_rejected')
     call integ%start_native_production(10,9,[1,2],1d0,1d0,10000_8)
     open(newunit=unit,status='scratch')
     call integ%save(unit)
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
  case('pool_budget')
     call integ%start_pool(100_8,1d0)
     forced_random=0.5d0
     nwritten=0
     do i=1,100
        weight=100d0
        if (mod(i,3).eq.0) weight=0d0
        call integ%consider_pool(weight,to_write,done)
        if (to_write) nwritten=nwritten+1
        call check(to_write.eqv.(weight.gt.0d0),'large candidates retained and zeros excluded')
        call check(done.eqv.(i.eq.100),'pool completes only at fixed trial budget')
     enddo
     call check(integ%ntrials.eq.100_8.and.integ%ncandidates.eq.nwritten,'pool counters include zeros')
     call check(.not.integ%envelope_exceeded,'underestimated storage cutoff is not fatal')
     call integ%pool_candidates(weights,priorities)
     call check(size(weights).eq.nwritten.and.all(weights.eq.100d0),'pool weights exported without correction')
     call check(all(priorities.eq.log(100d0)-log(0.5d0)),'pool priorities retain original variates')
     call integ%start_pool(0_8,1d0)
     call integ%pool_candidates(weights,priorities)
     call check(integ%production_done.and.size(weights).eq.0,'empty fixed budget is complete')
  case('pool_retention')
     call integ%start_pool(1000_8,2d0)
     nwritten=0
     do i=1,1000
        forced_random=dble(mod(29*i,997)+1)/998d0
        weight=dble(mod(i,17))/5d0
        if (i.eq.1000) then
           weight=huge(1d0)
           forced_random=0d0
        endif
        if (weight.gt.0d0) then
           target=log(weight)-log(max(forced_random,tiny(1d0)))
           if (target.gt.log(2d0)) then
              nwritten=nwritten+1
              expected_weights(nwritten)=weight
              expected_priorities(nwritten)=target
           endif
        endif
        call integ%consider_pool(weight,to_write,done)
     enddo
     call integ%pool_candidates(weights,priorities)
     call check(size(weights).eq.nwritten,'storage cutoff preserves every eligible candidate')
     call check(all(weights.eq.expected_weights(1:nwritten)),'candidate order preserves event indices')
     call check(all(priorities.eq.expected_priorities(1:nwritten)),'candidate priorities unchanged')
     call check(all(ieee_is_finite(priorities)),'extreme weights and zero variate remain finite')
     do i=1,nwritten
        ! Any later threshold at least as large as the storage cutoff can be
        ! applied to the retained metadata without losing an eligible event.
        if (expected_priorities(i).gt.log(20d0)) &
             call check(priorities(i).gt.log(20d0),'raised threshold retains all eligible events')
     enddo
     open(newunit=unit,status='scratch',form='formatted')
     call integ%export_pool(unit)
     rewind(unit)
     read(unit,'(a)') line
     call check(trim(line).eq.'MG5_SIMPLE_POOL 1','versioned pool export')
     read(unit,*) saved_rng,j
     call check(saved_rng.eq.1000_8.and.j.eq.nwritten,'pool export trial and candidate counts')
     read(unit,*) target
     call check(target.eq.2d0,'pool export cutoff')
     do i=1,nwritten
        read(unit,*) j,weight,target
        call check(j.eq.i.and.weight.eq.weights(i).and.target.eq.priorities(i),'stable export indices')
     enddo
     read(unit,'(a)') line
     call check(trim(line).eq.'END_MG5_SIMPLE_POOL','complete pool export')
     close(unit)
  case('pool_checkpoint')
     call train(integ,3000)
     forced_random=0.3141592653589d0
     call integ%sample(x,w)
     forced_random=-1d0
     call integ%start_pool(1000_8,2d0)
     do i=1,50
        call random_two_weights(weight)
        call integ%consider_pool(weight,to_write,done)
     enddo
     open(newunit=unit,status='scratch',form='formatted')
     call integ%save(unit)
     rewind(unit)
     call restored%load(unit,2,2)
     close(unit)
     saved_rng=rng_state
     do i=51,1000
        call random_two_weights(weight)
        call integ%consider_pool(weight,to_write,done)
     enddo
     rng_state=saved_rng
     do i=51,1000
        call random_two_weights(weight)
        call restored%consider_pool(weight,to_write,done)
     enddo
     call integ%pool_candidates(weights,priorities)
     call restored%pool_candidates(weights2,priorities2)
     call check(integ%ntrials.eq.restored%ntrials,'pool checkpoint trial counts')
     call check(size(weights).eq.size(weights2),'pool checkpoint candidate count')
     call check(all(weights.eq.weights2).and.all(priorities.eq.priorities2),'exact pool checkpoint continuation')
     forced_random=0.3141592653589d0
     call restored%sample(y,v)
     call check(all(x.eq.y).and.w.eq.v,'sampling map frozen through production pool')
  case('legacy_checkpoint')
     call train(integ,1000)
     open(newunit=unit,status='scratch',form='formatted')
     open(newunit=unit2,status='scratch',form='formatted')
     call integ%save(unit)
     rewind(unit)
     i=0
     do
        read(unit,'(a)',iostat=ios) line
        if (ios.ne.0) exit
        i=i+1
        if (i.eq.1) line='MG5_SIMPLE_INTEGRATOR 1'
        ! Version 3 adds tail metadata before the envelope line.
        if (i.eq.5.or.i.eq.8.or.i.eq.9) cycle
        ! Version 1 ends the flag line before the final pool-mode flag.
        if (i.eq.7) line=line(1:len_trim(line)-1)
        write(unit2,'(a)') trim(line)
     enddo
     rewind(unit2)
     call restored%load(unit2,2,2)
     close(unit)
     close(unit2)
     call check(all(integ%res.eq.restored%res),'version 1 survey checkpoints remain readable')
  case('pool_bad_budget')
     call integ%start_pool(-1_8,1d0)
  case('pool_bad_cutoff')
     call integ%start_pool(10_8,0d0)
  case('pool_nan_cutoff')
     call integ%start_pool(10_8,ieee_value(1d0,ieee_quiet_nan))
  case('pool_infinite_cutoff')
     call integ%start_pool(10_8,ieee_value(1d0,ieee_positive_inf))
  case('pool_nan_weight')
     call integ%start_pool(10_8,1d0)
     call integ%consider_pool(ieee_value(1d0,ieee_quiet_nan),to_write,done)
  case('pool_negative_weight')
     call integ%start_pool(10_8,1d0)
     call integ%consider_pool(-1d0,to_write,done)
  case('pool_premature_export')
     call integ%start_pool(10_8,1d0)
     call integ%pool_candidates(weights,priorities)
  case('pool_no_local_selection')
     call integ%start_pool(0_8,1d0)
     call integ%final_weights(weights)
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
  subroutine native_training_draws(sampler,first,last,include_zeros)
    type(staged_integrator),intent(inout) :: sampler
    integer,intent(in) :: first,last
    logical,intent(in) :: include_zeros
    integer :: point
    real(kind=8) :: xx(2),jac,ff(2)
    logical :: stored,batch_done
    do point=first,last
       forced_random=dble(mod(17*point,97)+1)/98d0
       call sampler%sample(xx,jac)
       ff(1)=(0.1d0+4d0*xx(1)**2)*jac
       if (include_zeros.and.point.le.250.and.mod(point,5).eq.1) ff(1)=0d0
       ff(2)=ff(1)
       if (mod(point,2).eq.0) ff(2)=-ff(1)
       call sampler%native_consider(ff,xx,stored,batch_done)
    enddo
    call check(batch_done,'training fixture fills exactly its nonzero batch')
    forced_random=-1d0
  end subroutine native_training_draws

  subroutine checkpoint_fill_sizes(sampler,fill_sizes)
    type(staged_integrator),intent(in) :: sampler
    integer,intent(out) :: fill_sizes(2)
    integer :: stream,j,k,size_map
    character(len=100000) :: record
    open(newunit=stream,status='scratch',form='formatted')
    call sampler%save(stream)
    rewind(stream)
    ! The v4 header includes grid-independent state through four moments.
    do j=1,13
       read(stream,'(a)') record
    enddo
    do j=1,2
       read(stream,*) size_map,fill_sizes(j)
       do k=1,4
          read(stream,'(a)') record
       enddo
    enddo
    close(stream)
  end subroutine checkpoint_fill_sizes

  subroutine native_analytic_probe_run(with_probes,rates,errors,shapes,trials)
    logical,intent(in) :: with_probes
    real(kind=8),intent(out) :: rates(2),errors(2),shapes(2)
    integer(kind=8),intent(out) :: trials
    type(staged_integrator) :: sampler
    real(kind=8) :: xx(2),ww,ff(2),mm(5),base,correction,factor,raw,priority
    integer :: point,epoch,flag,stream,nepochs,nstored,header
    logical :: stored,batch_done,completed
    base=0.1d0+(1d0-exp(-30d0))/30d0
    call sampler%init(2,2)
    call sampler%start_native_production(400,360,[1,2],1.5d0*base,1.5d0*base,1000000_8,initial_nonzero=4096_8)
    nstored=0
    do
       call sampler%sample(xx,ww)
       ff(1)=(0.1d0+exp(-30d0*xx(1)))*(1d0+xx(2))*ww
       ff(2)=ff(1)
       if (xx(2).lt.0.5d0) ff(2)=-ff(2)
       call sampler%native_consider(ff,xx,stored,batch_done)
       if (stored) then
          nstored=nstored+1
          call check(nstored.le.100000,'analytic probe candidate buffer')
          observables(nstored)=merge(1d0,0d0,xx(1).lt.0.1d0)
          event_signs(nstored)=sign(1d0,ff(2))
          call sampler%record_candidate_factor(1d0,completed)
       endif
       completed=.false.
       if (with_probes.and.sampler%native_completion_due) call sampler%check_native_completion(completed)
       if (completed) exit
       if (batch_done) call sampler%finish_native_iteration(completed)
       if (completed) exit
    enddo
    call sampler%native_rates(rates,errors,mm)
    trials=sampler%ntrials
    call check(sampler%quota_complete.and.sampler%overweight.lt.0.01d0,'analytic probe run passes every final tail criterion')
    open(newunit=stream,status='scratch')
    call sampler%write_native_pool(stream)
    rewind(stream)
    read(stream,'(a)') line
    read(stream,*) trials,point,nepochs
    call check(point.eq.nstored,'analytic probe stable candidate labels')
    do header=1,4+nepochs
       read(stream,'(a)') line
    enddo
    shapes=0d0
    do point=1,nstored
       read(stream,*) epoch,raw,priority,correction,flag,factor
       shapes(1)=shapes(1)+correction*observables(point)
       shapes(2)=shapes(2)+correction*event_signs(point)
    enddo
    close(stream)
    shapes=shapes/400d0
  end subroutine native_analytic_probe_run

  subroutine failed_probe_batch(sampler,with_probes)
    type(staged_integrator),intent(inout) :: sampler
    logical,intent(in) :: with_probes
    real(kind=8) :: xx(2),ww,ff(2),rr(2),ee(2),mm(5),rr2(2),ee2(2),mm2(5)
    integer :: point,stored_before,probes
    integer(kind=8) :: rng_before,trials_before,nonzero_before,target_before
    logical :: stored,batch_done,completed
    point=0
    probes=0
    do
       point=point+1
       call sampler%sample(xx,ww)
       if (with_probes) then
          candidate_x(:,point)=xx
          candidate_jac(point)=ww
       else
          call check(all(xx.eq.candidate_x(:,point)).and.ww.eq.candidate_jac(point), &
               'failed probe leaves all subsequent random draws and maps unchanged')
       endif
       ff(1)=(1d0+9d0*xx(1)**2)*ww
       if (point.eq.1) ff(1)=1d6
       if (mod(point,5).eq.0) ff(1)=0d0
       ff(2)=ff(1)
       if (xx(2).lt.0.5d0) ff(2)=-ff(2)
       call sampler%native_consider(ff,xx,stored,batch_done)
       if (stored) call sampler%record_candidate_factor(1d0,completed)
       if (with_probes.and.sampler%native_completion_due.and..not.batch_done) then
          call sampler%native_rates(rr,ee,mm)
          rng_before=rng_state
          trials_before=sampler%ntrials
          stored_before=sampler%ncandidates
          nonzero_before=sampler%native_nonzero
          target_before=sampler%native_target_nonzero
          call sampler%check_native_completion(completed)
          call check(.not.completed,'constructed tail prevents early completion')
          call check(rng_state.eq.rng_before,'failed completion probe is RNG-neutral')
          call check(sampler%ntrials.eq.trials_before.and.sampler%ncandidates.eq.stored_before.and. &
               sampler%native_nonzero.eq.nonzero_before.and.sampler%native_target_nonzero.eq.target_before, &
               'failed completion probe preserves counts and the original adaptation target')
          call check(sampler%native_iteration.eq.1.and.sampler%adaptation_updates.eq.0, &
               'failed completion probe cannot change the proposal epoch')
          call sampler%native_rates(rr2,ee2,mm2)
          call check(all(rr.eq.rr2).and.all(ee.eq.ee2).and.all(mm.eq.mm2),'failed probe preserves all rate accumulators')
          probes=probes+1
       endif
       if (batch_done) exit
    enddo
    if (with_probes) call check(probes.gt.0,'neutrality replay exercised an unfinished-batch probe')
    call sampler%finish_native_iteration(completed)
    call check(.not.completed,'tail test advances only at the original whole-batch boundary')
  end subroutine failed_probe_batch

  subroutine generation_point(sampler,is_done)
    type(staged_integrator),intent(inout) :: sampler
    logical,intent(out) :: is_done
    real(kind=8) :: xx(2),ww,target_value
    logical :: stored
    call sampler%sample(xx,ww)
    target_value=(0.1d0+exp(-10d0*xx(1)))*(1d0+xx(2))*ww
    call sampler%consider(target_value,stored,is_done,x=xx)
    if (stored) call sampler%record_candidate_factor(1d0+xx(1),is_done)
  end subroutine generation_point

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

    def test_folded_survey_adapts_using_all_fold_images(self):
        self.run_case('folded_adaptation')

    def test_folded_integrals(self):
        self.run_case('folding')

    def test_checkpoint_continuation(self):
        self.run_case('checkpoint')

    def test_survey_early_bin_doubling_and_statistics_driven_refinement(self):
        self.run_case('survey_resolution')

    def test_survey_disabled_zero_and_empty_adaptation_preserve_maps(self):
        self.run_case('survey_no_adaptation')

    def test_zero_and_sparse_targets(self):
        self.run_case('zero_sparse')

    def test_exact_event_quotas(self):
        self.run_case('quotas')

    def test_full_tail_mass_and_native_reevaluation(self):
        self.run_case('tail_mass')

    def test_worst_subset_protects_collection(self):
        self.run_case('tail_worst_subset')

    def test_actual_lhe_bias_factor_revokes_completion(self):
        self.run_case('tail_factor')

    def test_extreme_priority_threshold_is_finite(self):
        self.run_case('tail_extreme_priority')

    def test_one_percent_equality_fails(self):
        self.run_case('tail_equality')

    def test_overweight_corrections_and_signed_tail(self):
        self.run_case('overweight')

    def test_production_adapts_only_unfolded_maps_from_all_draws(self):
        self.run_case('production_adapt_maps')

    def test_production_preserves_survey_grid_resolution(self):
        self.run_case('production_adapt_resolution')

    def test_large_production_batches_preserve_survey_bin_count(self):
        self.run_case('production_no_bin_growth')

    def test_all_folded_production_is_bitwise_frozen(self):
        self.run_case('production_allfolded')

    def test_adaptive_production_integrals_and_event_density(self):
        self.run_case('production_adapt_analytic')

    def test_adaptive_production_exact_checkpoint_continuation(self):
        self.run_case('production_adapt_checkpoint')

    def test_invalid_production_adaptation_requests(self):
        for name, error in (
                ('adapt_bad_folds', 'invalid production folding factor'),
                ('adapt_bad_dimensions', 'adaptation folding dimension mismatch'),
                ('adapt_bad_interval', 'invalid production adaptation interval'),
                ('adapt_large_interval', 'invalid production adaptation interval'),
                ('adapt_interval_without_folds', 'adaptation interval requires folding factors'),
                ('adapt_missing_x', 'production adaptation requires coordinates'),
                ('adapt_bad_x_size', 'production coordinate dimension mismatch'),
                ('adapt_bad_x', 'invalid production adaptation coordinates'),
                ('adapt_nan_x', 'invalid production adaptation coordinates')):
            with self.subTest(name=name):
                self.run_case(name, error)

    def test_native_nonzero_iterations_and_all_trial_rates(self):
        self.run_case('native_nonzero')

    def test_native_skipped_adaptation_accumulates_all_trials_and_preserves_histograms(self):
        self.run_case('native_stable_training')

    def test_native_expiry_and_forecast_displace_every_batch_of_oldest_proposal(self):
        self.run_case('native_proposal_group_expiry')

    def test_native_completion_before_adaptation_preserves_rates_and_pool_metadata(self):
        self.run_case('native_early_completion')

    def test_native_failed_completion_probes_are_rng_and_adaptation_neutral(self):
        self.run_case('native_failed_probe_neutral')

    def test_native_completion_uses_the_latest_actual_lhe_factor(self):
        self.run_case('native_probe_latest_factor')

    def test_native_completion_probes_do_not_expire_event_history(self):
        self.run_case('native_probe_history')

    def test_native_completion_independent_seed_signed_rates_and_event_density(self):
        self.run_case('native_probe_analytic_seeds')

    def test_native_last_trial_can_complete_a_zero_heavy_epoch(self):
        self.run_case('native_zero_last_success')

    def test_native_last_trial_failure_and_draw_overrun_are_rejected(self):
        self.run_case('native_zero_last_failure', 'trial safety limit before event/tail completion')
        self.run_case('native_trial_overrun', 'trial safety limit before event/tail completion')

    def test_large_native_batches_preserve_survey_bin_count_and_folding(self):
        self.run_case('native_no_bin_growth')

    def test_native_tail_rethresholding_and_evolving_rates(self):
        self.run_case('native_tail_reevaluation')

    def test_native_tail_forecast_before_quota_preserves_final_checks(self):
        self.run_case('native_forecast_below_quota')

    def test_native_tail_forecast_includes_rejected_and_zero_trial_statistics(self):
        self.run_case('native_forecast_all_trials')

    def test_native_tail_forecast_accounts_for_collection_factors(self):
        self.run_case('native_forecast_factors')

    def test_native_tail_forecast_handles_tied_raw_weights(self):
        self.run_case('native_forecast_ties')

    def test_native_same_proposal_history_retains_known_tails_beyond_eight_batches(self):
        self.run_case('native_recent_history')

    def test_native_huge_survey_maximum_uses_modest_batches_and_storage(self):
        self.run_case('native_huge_survey_maximum')

    def test_native_initial_batch_respects_small_trial_ceiling(self):
        self.run_case('native_small_trial_budget')

    def test_native_relative_scale_sparse_bootstrap_boundary(self):
        self.run_case('native_relative_sparse_boundary')

    def test_native_relative_scale_retains_full_outlier_and_final_tail_checks(self):
        self.run_case('native_relative_retained_outlier')

    def test_native_sparse_pool_recovers_storage_cutoff(self):
        self.run_case('native_sparse_cutoff')

    def test_native_efficiency_forecast_has_a_growth_ceiling(self):
        self.run_case('native_growth_ceiling')

    def test_native_frozen_proposal_forecast_does_not_expire_useful_events(self):
        self.run_case('native_expiring_budget')

    def test_native_signed_density_and_historical_jacobians(self):
        self.run_case('native_analytic_history')

    def test_native_empty_quota(self):
        self.run_case('native_zero')

    def test_native_stage_and_safety_guards(self):
        for name, error in (
                ('native_incomplete_finish', 'incomplete or already finalized iteration'),
                ('native_zero_safety', 'trial safety limit before event/tail completion'),
                ('native_iteration_safety', 'iteration limit before event/tail completion'),
                ('native_checkpoint_rejected', 'active checkpoint unsupported')):
            with self.subTest(name=name):
                self.run_case(name, error)

    def test_strict_rejection_tiny_quota_densities(self):
        self.run_case('strict_rejection')

    def test_strict_rejection_refuses_underestimated_envelope(self):
        self.run_case('strict_bad_bound', 'strict production envelope exceeded')

    def test_fixed_pool_budget_including_zeros_and_overweights(self):
        self.run_case('pool_budget')

    def test_pool_retention_and_stable_export_indices(self):
        self.run_case('pool_retention')

    def test_pool_checkpoint_continuation(self):
        self.run_case('pool_checkpoint')

    def test_existing_survey_checkpoint_still_readable(self):
        self.run_case('legacy_checkpoint')

    def test_pool_input_guards(self):
        for case, error in (
                ('pool_bad_budget', 'invalid pool budget'),
                ('pool_bad_cutoff', 'invalid pool cutoff'),
                ('pool_nan_cutoff', 'invalid pool cutoff'),
                ('pool_infinite_cutoff', 'invalid pool cutoff'),
                ('pool_nan_weight', 'invalid pool target'),
                ('pool_negative_weight', 'invalid pool target'),
                ('pool_premature_export', 'pool is not finished'),
                ('pool_no_local_selection', 'pool requires coordinator finalization')):
            with self.subTest(case=case):
                self.run_case(case, error)

    def test_incompatible_checkpoint_version(self):
        self.run_case('bad_version', 'incompatible checkpoint version')

    def test_incompatible_checkpoint_dimensions(self):
        self.run_case('bad_dimensions', 'incompatible checkpoint dimensions')

    def test_truncated_checkpoint(self):
        self.run_case('truncated', 'invalid checkpoint counters')

    def test_insufficient_event_quota_fails(self):
        self.run_case('incomplete_quota', 'event quota not reached')
