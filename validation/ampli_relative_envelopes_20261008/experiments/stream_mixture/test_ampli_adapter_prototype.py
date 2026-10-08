"""Exercise the AmpliCol MC@NLO stage protocol with analytic stub physics."""

import math
import pathlib
import shutil
import subprocess
import tempfile
import unittest
from madgraph.various import ampli_pool


ROOT = pathlib.Path(__file__).resolve().parents[4]
SOURCE = ROOT / 'Template' / 'NLO' / 'SubProcesses'
ADAPTER_SOURCE = pathlib.Path(__file__).with_name('ampli_mint_adapter.f90')


STUB = r'''
module mint_module
  implicit none
  integer,parameter :: ndimmax=4,nintegrals=6
  integer :: ndim=2,ifold(ndimmax)=1,imode=0,itmax=4,ncalls0=4000,iconfig=7,born_spread_phase=0
  logical :: new_point=.false.,only_virt=.false.,born_spread_active=.false.,born_spread_ready=.false.
  double precision :: virtual_fraction(1)=0.5d0,accuracy=0.03d0,aux_shift=0d0
  logical :: changing_auxiliary=.false.
  double precision :: ans(nintegrals,0:1)=0d0,unc(nintegrals,0:1)=0d0
  double precision :: written_absolute_error=-1d0
  integer(kind=8) :: virtual_training_calls=0_8,auxiliary_updates=0_8
  integer :: calibration_calls=0,table_writes=0,integer_resets=0
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
    virtual_training_calls=virtual_training_calls+1_8
  end subroutine
  subroutine nlops_update_auxiliary(errors)
    double precision,intent(in) :: errors(nintegrals)
    integer :: unit
    auxiliary_updates=auxiliary_updates+1_8
    if (changing_auxiliary) aux_shift=aux_shift-0.1d0
    open(newunit=unit,file='grid.MC_integer',status='replace')
    write(unit,'(a)') 'STUB_INTEGER_STATE'
    close(unit)
  end subroutine
  subroutine calibrate_born_spreading(fun,sample)
    double precision,external :: fun
    external :: sample
    double precision :: x(ndimmax),vol
    integer :: folds(ndimmax)
    if (.not.born_spread_active.or.born_spread_ready) error stop 'unexpected stub Born calibration'
    if (imode.ne.0.or.any(ifold(1:ndim).ne.1)) error stop 'Born calibration must use unfolded callback'
    call sample(x,vol,folds)
    if (any(folds.ne.1).or.vol.le.0d0) error stop 'invalid Born calibration point'
    calibration_calls=calibration_calls+1
    ! Represent a fitted decomposition with a different absolute target and
    ! the same exact signed integral. Pre-calibration estimates must be dropped.
    aux_shift=0.25d0
  end subroutine
  subroutine born_spread_write_table
    table_writes=table_writes+1
  end subroutine
  subroutine reset_MC_grid
    integer_resets=integer_resets+1
  end subroutine
  subroutine nlops_write_results(absolute_error)
    double precision,intent(in) :: absolute_error
    written_absolute_error=absolute_error
  end subroutine
  subroutine nlops_save_auxiliary(unit)
    integer,intent(in) :: unit
    write(unit,*) 'STUB_AUX ',virtual_fraction,aux_shift
  end subroutine
  subroutine nlops_load_auxiliary(unit)
    integer,intent(in) :: unit
    character(len=16) :: marker
    read(unit,*) marker,virtual_fraction,aux_shift
    if (marker.ne.'STUB_AUX') error stop 'auxiliary checkpoint misaligned'
  end subroutine
end module mint_module

module stub_state
  implicit none
  integer(kind=8) :: rng_state=1234567_8,ncalls=0_8,nopen=0_8,nfold=0_8,nclose=0_8
  double precision :: nv_sum=0d0,v_sum=0d0,last_sign=1d0
  logical :: evaluate_virtual=.false.,zero_target=.false.,production_sparse=.false.,zero_point=.false.
  logical :: sharp_production=.false.,virtual_peak=.false.
  double precision :: virtual_coefficient=0.5d0
  integer :: survey_points=0
  double precision :: survey_sum(8)=0d0,survey_square(8)=0d0,survey_max(2,8)=0d0
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
  double precision :: base,ran2,survey_target,residual_base
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
     zero_point=zero_target
     if (imode.eq.2.and.production_sparse.and.mod(nopen,4_8).eq.0_8) zero_point=.true.
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
        survey_max(1,survey_iteration)=max(survey_max(1,survey_iteration),values(1))
        survey_max(2,survey_iteration)=max(survey_max(2,survey_iteration),values(5)*virtual_fraction(1))
     endif
     last_sign=sign(1d0,values(2))
     return
  else
     error stop 'unexpected callback ifl'
  endif
  if (zero_point) return
  base=(0.1d0+exp(-10d0*x(1)))*(1d0+x(2))*vol
  if (imode.eq.2.and.sharp_production) &
       base=(0.1d0+5d0*(1d0-exp(-10d0))/(1d0-exp(-50d0))*exp(-50d0*x(1)))*(1d0+x(2))*vol
  if (abrv.ne.'virt') nv_sum=nv_sum+(2d0+aux_shift)*base
  if (evaluate_virtual) then
     residual_base=base
     if (virtual_peak) residual_base=base*merge(5d0,0.5d0,x(1).lt.0.1d0)
     if (abrv.eq.'virt') then
        v_sum=v_sum-(virtual_coefficient+aux_shift)*residual_base
     else
        v_sum=v_sum-(virtual_coefficient+aux_shift)*residual_base/virtual_fraction(1)
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
  double precision :: expected,base,weight,saved_absolute_error,virtual_prob,probability
  double precision :: moments(5),exported(5),delta_abs,delta_signed,abs_target,signed_target,priority,cutoff
  double precision,allocatable :: candidate_weights(:),candidate_factors(:)
  double precision :: survey_ans(nintegrals),survey_unc(nintegrals),threshold,tail_fractions(3),correction,factor
  double precision :: saved_virtual_fraction,expected_envelope,residual_base,probability_inputs(4)
  integer :: final_quota,export_quota,export_final,tail_flag,selected,nepochs,birth,epoch_id,eligible
  integer :: export_ndim,export_updates,export_mask(ndimmax),expected_updates,checkpoint_size
  integer :: proposal_ids(100),eligible_epochs(100),event_counts(100),newest_proposal,k
  integer :: ini_fin_fks,unit,iu,i,candidates,quota,ios,negatives,export_candidates,last_survey
  integer(kind=8) :: opens_before,folds_before,closes_before,budget,trials,export_trials
  integer(kind=8) :: training_before,auxiliary_before,export_interval,export_batch,export_pending
  integer(kind=8) :: expected_batch,expected_pending,epoch_trials,epoch_nonzero,epoch_target,total_nonzero
  double precision :: epoch_cutoffs(100),epoch_thresholds(100),epoch_envelopes(100),epoch_envelope,epoch_moments(5)
  logical :: to_write,done
  character(len=:),allocatable :: saved_checkpoint,current_checkpoint
  character(len=4) :: abrv
  character(len=40) :: task
  character(len=300) :: line
  common /fks_channels/ ini_fin_fks
  common /to_abrv/ abrv
  call get_command_argument(1,task)
  if (task.eq.'stream_probability') then
     read(*,*) probability_inputs
     virtual_prob=ampli_stream_probability(probability_inputs(1),probability_inputs(2),probability_inputs(3:4))
     write(*,'(a,es25.16)') 'PROBABILITY ',virtual_prob
     stop
  endif
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
  elseif (task.eq.'born_spreading') then
     born_spread_active=.true.
  elseif (task.eq.'changing_auxiliary') then
     changing_auxiliary=.true.
  elseif (task.eq.'covariance') then
     virtual_fraction=1d0
  elseif (task.eq.'sparse_production'.or.task.eq.'adaptive_sparse') then
     production_sparse=.true.
  endif
  if (task.eq.'stream_mixture'.or.task.eq.'stream_mixture_adaptive') then
     virtual_coefficient=0.02d0
     virtual_peak=.true.
  endif
  imode=1
  if (task.eq.'wrong_stage') then
     imode=0
     call ampli_integrate(analytic_sigint)
     error stop 'obsolete stage zero unexpectedly accepted'
  endif
  ncalls0=4000
  itmax=2 ! The adapter must enforce at least four survey iterations.
  ifold(1:ndim)=[2,4]
  if (task.eq.'adaptive'.or.task.eq.'adaptive_sparse'.or.task.eq.'completion_factors') then
     ifold(1:ndim)=[1,4]
     ! Sharpen the production density while preserving its exact integral,
     ! so the saved maximum requires updates and more than one epoch.
     sharp_production=.true.
  endif
  if (task.eq.'stream_mixture_adaptive') ifold(1:ndim)=[1,4]
  opens_before=nopen
  folds_before=nfold
  closes_before=nclose
  if (task.eq.'failed_survey') accuracy=1d-12
  call ampli_integrate(analytic_sigint)
  last_survey=4
  if (born_spread_active) last_survey=8
  call check(nopen-opens_before.eq.int(last_survey,8)*4000_8,'survey point count')
  call check(nclose-closes_before.eq.int(last_survey,8)*4000_8,'one closing callback per point')
  call check(nfold-folds_before.eq.int(product(ifold(1:ndim))-1,8)*int(last_survey,8)*4000_8, &
       'all survey folds evaluated')
  call check(written_absolute_error.eq.ampli_absolute_uncertainty,'absolute uncertainty passed to results writer')
  call check(ncalls0.eq.4000.and.itmax.eq.4,'final iteration statistical points and minimum four iterations')
  call check(auxiliary_updates.eq.int(last_survey-1,8),'only unsuccessful survey iterations update auxiliary physics')
  if (born_spread_active) then
     call check(calibration_calls.eq.1.and.table_writes.eq.1.and.integer_resets.eq.1, &
          'single in-survey Born calibration installs table and resets integer grid')
     call check(born_spread_ready.and.born_spread_phase.eq.3.and.imode.eq.1, &
          'survey restores production Born state and external survey stage')
     call check(all(ifold(1:ndim).eq.[2,4]),'requested folds restored after calibration')
     call check(abs(survey_sum(4)-survey_sum(8)).gt.100d0,'calibration fixture changes absolute target')
  endif
  call check(abs(ans(1,1)+merge(0d0,ans(5,1),only_virt)-survey_sum(last_survey)/4000d0).lt.1d-12, &
       'only the final survey target enters reported absolute estimate')
  expected=sqrt((survey_square(last_survey)-survey_sum(last_survey)**2/4000d0)/4000d0/3999d0)
  call check(abs(ampli_absolute_uncertainty-expected).lt.max(1d-12,expected*1d-9), &
       'absolute uncertainty matches direct target second moments')
  if (task.eq.'covariance') then
     call check(abs(ampli_absolute_uncertainty-(unc(1,1)+unc(5,1))).lt.1d-12, &
          'perfect positive covariance retained')
     call check(ampli_absolute_uncertainty.gt.1.2d0*sqrt(unc(1,1)**2+unc(5,1)**2), &
          'covariance test distinguishes separate-stream quadrature')
  endif
  base=1.5d0*(0.1d0+(1d0-exp(-10d0))/10d0)
  residual_base=base
  if (virtual_peak) residual_base=0.5d0*base+4.5d0*1.5d0*(0.01d0+(1d0-exp(-1d0))/10d0)
  if (task.eq.'zero') then
     call check(all(ans.eq.0d0).and.all(unc.eq.0d0),'zero survey results')
     imode=2
     call ampli_load_production
     call ampli_start_generation(0)
     call ampli_finish_pool
     open(newunit=iu,file='ampli_pool.dat',status='old')
     read(iu,'(a)') line
     call check(trim(line).eq.'MG5_AMPLI_POOL 5','zero-pool metadata version')
     read(iu,*) export_trials,export_candidates,nepochs
     read(iu,*) exported
     read(iu,*) export_quota,export_final,threshold,tail_fractions
     read(iu,*) export_ndim,export_updates
     read(iu,*) export_mask(1:ndim)
     close(iu)
     call check(export_trials.eq.0_8.and.export_candidates.eq.0,'zero-quota complete workflow')
     call check(all(exported.eq.0d0),'empty-pool moments')
     call check(all(ans.eq.0d0).and.all(unc.eq.0d0),'zero production results')
     call check(export_ndim.eq.ndim.and.export_updates.eq.0,'zero-quota adaptation metadata')
     call check(all(export_mask(1:ndim).eq.0).and.nepochs.eq.0,'all-folded zero-quota grids stay fixed')
     print *, 'PASS ',trim(task)
     stop
  elseif (only_virt) then
     call check(abs(ans(1,1)-0.5d0*base).lt.max(6d0*unc(1,1),1d-3),'only-virtual absolute rate')
     call check(abs(ans(2,1)+0.5d0*base).lt.max(6d0*unc(2,1),1d-3),'only-virtual signed rate')
     call check(ans(5,1).eq.0d0,'only-virtual residual not double counted')
  else
     call check(abs(ans(1,1)-(2d0+aux_shift)*base).lt.max(6d0*unc(1,1),1d-3),'nonvirtual absolute rate')
     if (task.ne.'born') then
        call check(abs(ans(5,1)-(virtual_coefficient+aux_shift)*residual_base).lt.max(6d0*unc(5,1),1d-3), &
             'residual absolute rate')
        call check(abs(ans(2,1)-((2d0+aux_shift)*base-(virtual_coefficient+aux_shift)*residual_base)) &
             .lt.max(6d0*unc(2,1),1d-3),'full signed rate')
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
  if (task.eq.'wrong_channel') iconfig=iconfig+1
  if (task.eq.'old_checkpoint') then
     inquire(file='ampli_grids',size=checkpoint_size)
     allocate(character(len=checkpoint_size) :: saved_checkpoint)
     open(newunit=iu,file='ampli_grids',status='old',access='stream',form='unformatted')
     read(iu) saved_checkpoint
     close(iu)
     i=index(saved_checkpoint,achar(10))
     open(newunit=iu,file='ampli_grids',status='replace',access='stream',form='unformatted')
     write(iu) 'MG5_AMPLICOL 2 1'//achar(10)//saved_checkpoint(i+1:)
     close(iu)
     deallocate(saved_checkpoint)
  endif
  call ampli_load_production
  call check(ampli_absolute_uncertainty.eq.saved_absolute_error,'absolute uncertainty checkpoint round trip')
  virtual_prob=0d0
  if (.not.only_virt.and.task.ne.'born') then
     probability=max(1d-3,min(1d0-1d-3,ans(5,1)/(ans(1,1)+ans(5,1))))
     expected=survey_max(2,last_survey)/sum(survey_max(:,last_survey))
     virtual_prob=max(probability,min(expected,4d0*probability,0.1d0))
     if (virtual_peak) call check(virtual_prob.gt.1.2d0*probability, &
          'tail fixture increases the fixed virtual draw probability')
  endif
  expected_envelope=survey_max(1,last_survey)
  if (virtual_prob.gt.0d0) expected_envelope= &
       max(expected_envelope/(1d0-virtual_prob),survey_max(2,last_survey)/virtual_prob)
  quota=4000
  final_quota=3636
  if (task.eq.'tiny_quota') then
     quota=2
     final_quota=1
  endif
  if (task.eq.'custom_quota') then
     quota=1237
     final_quota=1000
  endif
  open(newunit=unit,file='ampli_job.dat',status='replace')
  write(unit,'(a)') 'MG5_AMPLI_JOB 2'
  write(unit,*) quota,final_quota
  close(unit)
  survey_ans=ans(:,1)
  survey_unc=unc(:,1)
  training_before=virtual_training_calls
  auxiliary_before=auxiliary_updates
  saved_virtual_fraction=virtual_fraction(1)
  inquire(file='ampli_grids',size=checkpoint_size)
  allocate(character(len=checkpoint_size) :: saved_checkpoint,current_checkpoint)
  open(newunit=iu,file='ampli_grids',status='old',access='stream',form='unformatted')
  read(iu) saved_checkpoint
  close(iu)
  call ampli_start_generation(quota)
  allocate(candidate_weights(1000000),candidate_factors(1000000))
  candidates=0
  negatives=0
  trials=0_8
  moments=0d0
  opens_before=nopen
  folds_before=nfold
  closes_before=nclose
  do
     call ampli_next_candidate(analytic_sigint,to_write,done)
     if (done.and..not.to_write) exit
     trials=trials+1_8
     call check(trials.lt.1000000_8,'analytic generator terminates within safety limit')
     probability=1d0
     if (virtual_prob.gt.0d0) then
        if (abrv.eq.'virt') then
           probability=virtual_prob
        else
           probability=1d0-virtual_prob
        endif
     endif
     signed_target=(nv_sum+v_sum)/probability
     abs_target=abs(signed_target)
     delta_abs=abs_target-moments(1)
     delta_signed=signed_target-moments(2)
     moments(1)=moments(1)+delta_abs/dble(trials)
     moments(2)=moments(2)+delta_signed/dble(trials)
     moments(3)=moments(3)+delta_abs*(abs_target-moments(1))
     moments(4)=moments(4)+delta_signed*(signed_target-moments(2))
     moments(5)=moments(5)+delta_abs*(signed_target-moments(2))
     if (to_write) then
        candidates=candidates+1
        factor=1d0
        if (task.eq.'completion_factors') factor=0.5d0+dble(mod(candidates,7))/7d0
        call ampli_record_candidate_factor(factor,done)
        candidate_weights(candidates)=abs_target
        candidate_factors(candidates)=factor
        call check(last_sign.eq.sign(1d0,signed_target),'candidate sign follows sampled stream')
        if (last_sign.lt.0d0) negatives=negatives+1
     endif
     if (done) exit
  enddo
  budget=trials
  call check(nopen-opens_before.eq.budget,'production point count')
  call check(nclose-closes_before.eq.budget,'one production closing callback per point')
  call check(nfold-folds_before.eq.int(product(ifold(1:ndim))-1,8)*budget,'all production folds evaluated')
  call ampli_finish_pool
  call check(ncalls0.eq.budget.and.itmax.ge.1,'results account for every production trial')
  call check(abs(ans(1,1)-moments(1)).lt.1d-10,'published worker absolute rate evolves during production')
  call check(abs(ans(2,1)-moments(2)).lt.1d-10,'published worker signed rate evolves during production')
  call check(abs(unc(1,1)-sqrt(moments(3))/dble(trials)).lt.1d-10,'native absolute population uncertainty')
  call check(abs(unc(2,1)-sqrt(moments(4))/dble(trials)).lt.1d-10,'native signed population uncertainty')
  call check(ampli_absolute_uncertainty.eq.unc(1,1),'production absolute uncertainty published')
  call check(ans(5,1).eq.0d0,'production residual is included exactly once')
  call check(virtual_training_calls.eq.training_before.and.auxiliary_updates.eq.auxiliary_before, &
       'virtual approximation and integer sampling are not retrained during production')
  call check(virtual_fraction(1).eq.saved_virtual_fraction,'virtual sampling fraction remains frozen')
  open(newunit=iu,file='ampli_grids',status='old',access='stream',form='unformatted')
  read(iu) current_checkpoint
  close(iu)
  call check(current_checkpoint.eq.saved_checkpoint,'survey grid checkpoint is unchanged by production')
  open(newunit=iu,file='ampli_pool.dat',status='old')
  read(iu,'(a)') line
  call check(trim(line).eq.'MG5_AMPLI_POOL 5','pool metadata version')
  read(iu,*) export_trials,export_candidates,nepochs
  call check(export_trials.eq.budget.and.export_candidates.eq.candidates,'metadata counts match callbacks')
  call check(nepochs.eq.itmax.and.nepochs.gt.0,'all completed iterations exported')
  read(iu,*) exported
  call check(all(abs(exported-moments).le.1d-10*max(1d0,abs(moments))),'exported means variances and covariance')
  read(iu,*) export_quota,export_final,threshold,tail_fractions
  call check(export_quota.eq.quota.and.export_final.eq.final_quota,'generated and final quotas exported')
  call check(all(tail_fractions.lt.0.01d0),'all three tail criteria below one percent')
  read(iu,*) export_ndim,export_updates
  read(iu,*) export_mask(1:ndim)
  call check(export_ndim.eq.ndim,'adaptation dimension metadata')
  call check(all(export_mask(1:ndim).eq.merge(1,0,ifold(1:ndim).eq.1)),'only unfolded coordinates adapt')
  if (any(ifold(1:ndim).eq.1)) then
     call check(export_updates.gt.0,'mixed-fold production updates the grid')
  else
     call check(export_updates.eq.0,'all-folded production disables grid adaptation')
  endif
  expected_pending=0_8
  total_nonzero=0_8
  do i=1,nepochs
     read(iu,*) epoch_id,epoch_trials,epoch_nonzero,epoch_target,epoch_moments, &
          epoch_cutoffs(i),epoch_envelope,epoch_thresholds(i),eligible,proposal_ids(i)
     call check(epoch_id.eq.i,'ordered native epochs')
     if (i.eq.1) then
        call check(abs(epoch_cutoffs(i)-min(expected_envelope,survey_ans(1)+survey_ans(5))).lt.1d-12, &
             'first storage cutoff retains candidates below a conservative survey maximum')
        call check(epoch_target.eq.max(1024_8,min(8192_8,int(quota,8))), &
             'first event-producing iteration is bounded independently of the survey maximum')
        if (candidates.le.200) call check(epoch_envelope.ge.expected_envelope*(1d0-1d-13), &
             'sparse relative scale retains the survey maximum')
     endif
     if (i.lt.nepochs) then
        call check(epoch_nonzero.eq.epoch_target,'nonfinal iteration completes its nonzero point budget')
     else
        call check(epoch_nonzero.gt.0_8.and.epoch_nonzero.le.epoch_target, &
             'final iteration may stop within its nonzero point budget')
     endif
     call check(epoch_trials.ge.epoch_nonzero,'zero points included in attempted counts')
     if (eligible.eq.1) call check(epoch_thresholds(i).ge.epoch_cutoffs(i),'birth threshold covers discarded candidates')
     eligible_epochs(i)=eligible
     if (i.eq.1) then
        call check(proposal_ids(i).eq.1,'first survey proposal has stable identity')
     else
        call check(proposal_ids(i).eq.proposal_ids(i-1).or.proposal_ids(i).eq.proposal_ids(i-1)+1, &
             'proposal identity advances only when the sampling map changes')
     endif
     epoch_envelopes(i)=epoch_envelope
     call check(epoch_envelope.gt.0d0,'every event-producing proposal has a positive relative scale')
     if (proposal_ids(i).eq.1.and.candidates.le.200) &
          call check(epoch_envelope.ge.expected_envelope*(1d0-1d-13),'sparse first proposal retains survey maximum')
     if (.not.any(ifold(1:ndim).eq.1).and.candidates.gt.200) then
        k=max(int(0.05d0*dble(candidates)),1)
        call check(count(candidate_weights(1:candidates).gt.epoch_envelope).lt.k.and. &
             count(candidate_weights(1:candidates).ge.epoch_envelope).ge.k, &
             'frozen-grid mature relative scale is the candidate upper-five-percent order statistic')
     endif
     expected_pending=expected_pending+epoch_trials
     total_nonzero=total_nonzero+epoch_nonzero
  enddo
  call check(expected_pending.eq.trials,'all epoch trials enter evolving rates')
  if (production_sparse) then
     call check(total_nonzero.eq.trials-trials/4_8,'nonzero stopping distinguishes zero trials')
  endif
  call check(proposal_ids(nepochs).eq.export_updates+1,'proposal identities match actual map updates')
  event_counts=0
  selected=0
  do i=1,candidates
     read(iu,*,iostat=ios) birth,weight,priority,correction,tail_flag,factor
     event_counts(birth)=event_counts(birth)+1
     if (correction.gt.0d0) selected=selected+1
     if (eligible_epochs(birth).eq.1) then
        call check((tail_flag.eq.1).eqv.(weight.gt.epoch_thresholds(birth)),'birth epoch full tail flag')
     else
        call check(correction.eq.0d0.and.tail_flag.eq.0,'expired proposal contributes no selected event')
     endif
     call check(factor.eq.candidate_factors(i),'nominal LHE factor preserved')
     call check(ios.eq.0,'each retained event has metadata')
     call check(abs(weight-candidate_weights(i)).lt.1d-12,'candidate draw-time weight and ordering stay unchanged')
     call check(priority.ge.log(weight).and.priority.gt.log(epoch_cutoffs(birth)),'stored candidate has valid retained priority')
  enddo
  close(iu)
  newest_proposal=0
  do i=nepochs,1,-1
     if (event_counts(i).eq.0) cycle
     if (newest_proposal.eq.0) newest_proposal=proposal_ids(i)
     k=0
     do birth=i+1,nepochs
        if (event_counts(birth).eq.0.or.proposal_ids(birth).le.proposal_ids(i)) cycle
        if (birth.gt.i+1) then
           if (any(proposal_ids(i+1:birth-1).eq.proposal_ids(birth).and.event_counts(i+1:birth-1).gt.0)) cycle
        endif
        k=k+1
     enddo
     call check(eligible_epochs(i).eq.merge(1,0,k.lt.8),'history keeps the last eight event-bearing proposals')
  enddo
  call check(selected.eq.quota,'exact generated reserve size')
  call check(candidates.gt.0,'nonzero process retains candidates')
  if (quota.gt.10) call check(candidates.lt.budget,'rejected points still contribute to production moments')
  if (production_sparse) then
     call check(candidates.le.3_8*budget/4_8,'zero production points cannot produce candidates')
     base=0.75d0*base
     residual_base=0.75d0*residual_base
  endif
  if (only_virt) then
     call check(negatives.eq.candidates,'only-virtual candidate signs')
  elseif (task.eq.'born') then
     call check(negatives.eq.0,'Born-only candidate signs')
  elseif (quota.gt.10) then
     expected=(virtual_coefficient+aux_shift)*residual_base/ &
          ((2d0+aux_shift)*base+(virtual_coefficient+aux_shift)*residual_base)
     call check(abs(dble(negatives)/dble(candidates)-expected).lt.0.05d0, &
          'virtual split signed candidate density')
     if (virtual_peak) then
        expected=(2d0+aux_shift)*base+(virtual_coefficient+aux_shift)*residual_base
        call check(abs(ans(1,1)-expected).lt.max(6d0*unc(1,1),1d-3),'changed-mixture absolute integral')
        expected=(2d0+aux_shift)*base-(virtual_coefficient+aux_shift)*residual_base
        call check(abs(ans(2,1)-expected).lt.max(6d0*unc(2,1),1d-3),'changed-mixture signed integral')
     endif
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
                   str(ADAPTER_SOURCE), str(cls.work / 'driver.f90'),
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
                self.assertIn('AmpliCol production adaptation dimensions, updates:', result.stdout)
                if name != 'zero':
                    self.assertIn('AmpliCol native production iteration, trials, ABS, signed, errors:', result.stdout)
                pool = ampli_pool.read_pool(run)
                self.assertEqual(pool['version'], 5)
                status = ampli_pool.pool_status([pool], pool['final_quota'])
                self.assertLess(status['overweight'], .01)
                return pool
            else:
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn(expected_error, result.stdout)

    def stream_probability(self, values, expected_error=None):
        result = subprocess.run([str(self.executable), 'stream_probability'],
                                input=' '.join(map(str, values)) + '\n',
                                cwd=self.work, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=10)
        if expected_error:
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn(expected_error, result.stdout)
            return
        self.assertEqual(result.returncode, 0, result.stdout)
        return float(result.stdout.split('PROBABILITY ', 1)[1].split()[0])

    def test_fixed_stream_probability_bounds_and_mixture(self):
        cases = [
            ((99, 1, 99, 3), 3/102),  # equalize the conditional maxima
            ((99, 1, 1, 99), .04),  # no more than four times rate probability
            ((95, 5, 1, 99), .1),  # cap upward correction at ten percent
            ((8, 2, 1, 99), .2),  # preserve already larger rate probabilities
            ((99, 1, 999, 1), .01),  # do not decrease virtual sampling
            ((99, 1, 0, 0), .01),  # no maximum information
            ((1, 0, 1, 0), .001),  # retain lower probability clamp
            ((0, 1, 0, 1), .999),  # retain upper probability clamp
            ((0, 0, 0, 0), 0),
            ((1.e308, 1.e308, 1.e308, 1.e308), .5),
            ((99, 1, 1.e308, 1.e308), .04),
            ((1.e-300, 1.e-300, 1.e-300, 1.e-300), .5),
        ]
        for inputs, expected in cases:
            with self.subTest(inputs=inputs):
                result = self.stream_probability(inputs)
                self.assertTrue(math.isfinite(result))
                self.assertAlmostEqual(result, expected, places=14)

    def test_fixed_stream_probability_rejects_invalid_survey_values(self):
        for inputs in [(-1, 1, 1, 1), (1, -1, 1, 1),
                       ('NaN', 1, 1, 1), (1, 'Infinity', 1, 1)]:
            with self.subTest(inputs=inputs):
                self.stream_probability(inputs, 'Invalid AmpliCol survey stream rates')
        for inputs in [(1, 1, -1, 1), (1, 1, 1, -1),
                       (1, 1, 'NaN', 1), (1, 1, 1, 'Infinity')]:
            with self.subTest(inputs=inputs):
                self.stream_probability(inputs, 'Invalid AmpliCol survey stream maxima')

    def test_changed_stream_probability_preserves_signed_moments_with_folding(self):
        self.run_case('stream_mixture')

    def test_changed_stream_probability_stays_fixed_while_unfolded_grids_adapt(self):
        self.run_case('stream_mixture_adaptive')

    def test_complete_signed_workflow(self):
        self.run_case('signed')

    def test_mixed_fold_generation_adapts_only_unfolded_dimensions(self):
        pool = self.run_case('adaptive')
        self.assertLess(pool['epochs'][-1]['nonzero'],
                        pool['epochs'][-1]['target_nonzero'])
        self.assertEqual(pool['adaptation']['updates'],
                         pool['epochs'][-1]['proposal_id']-1)
        self.assertLessEqual(pool['adaptation']['updates'], len(pool['epochs'])-1)

    def test_early_completion_retains_lhe_factors_in_all_tail_checks(self):
        pool = self.run_case('completion_factors')
        self.assertLess(pool['epochs'][-1]['nonzero'],
                        pool['epochs'][-1]['target_nonzero'])
        self.assertGreater(len({row[4] for row in pool['candidates']}), 1)
        self.assertEqual(pool['adaptation']['updates'],
                         pool['epochs'][-1]['proposal_id']-1)
        self.assertLessEqual(pool['adaptation']['updates'], len(pool['epochs'])-1)

    def test_adaptive_generation_counts_zero_and_rejected_trials(self):
        self.run_case('adaptive_sparse')

    def test_only_virtual_workflow(self):
        self.run_case('only_virt')

    def test_born_only_workflow(self):
        self.run_case('born')

    def test_zero_target_and_quota(self):
        self.run_case('zero')

    def test_correlated_absolute_rate_uncertainty(self):
        self.run_case('covariance')

    def test_born_spreading_calibrates_inside_survey_then_measures_four_iterations(self):
        self.run_case('born_spreading')

    def test_old_three_stage_checkpoint_is_rejected(self):
        self.run_case('old_checkpoint', 'checkpoint version or stage')

    def test_changing_auxiliary_retains_only_final_absolute_target(self):
        self.run_case('changing_auxiliary')

    def test_custom_channel_event_quotas(self):
        self.run_case('custom_quota')

    def test_small_quota_finishes_with_tail_bound(self):
        self.run_case('tiny_quota')

    def test_production_moments_include_zero_and_rejected_points(self):
        self.run_case('sparse_production')

    def test_survey_must_reach_requested_accuracy(self):
        self.run_case('failed_survey', 'survey failed to reach requested channel accuracy')

    def test_missing_checkpoint(self):
        self.run_case('missing', 'Missing AmpliCol checkpoint')

    def test_missing_integer_sampling_checkpoint(self):
        self.run_case('missing_MC_state', 'Missing MC_integer state')

    def test_wrong_checkpoint_channel(self):
        self.run_case('wrong_channel', 'checkpoint belongs to a different channel')

    def test_wrong_checkpoint_folding(self):
        self.run_case('wrong_folding', 'checkpoint folding differs')

    def test_wrong_checkpoint_stage(self):
        self.run_case('wrong_stage', 'AmpliCol requires survey stage 1 or generation stage 2')
