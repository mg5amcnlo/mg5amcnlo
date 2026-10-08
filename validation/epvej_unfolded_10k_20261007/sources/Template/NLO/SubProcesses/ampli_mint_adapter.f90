! AmpliCol numerical backend for the existing one-channel MC@NLO job protocol.
! The run manager owns channel quotas and normalization. Physics stays in
! sigintF and the shared MC@NLO services of mint_module.
module ampli_mint_adapter
  use simple_integrator_mod, only: staged_integrator
  use mint_module
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: ampli_integrate,ampli_load_production,ampli_start_generation
  public :: ampli_next_candidate,ampli_finish_pool,ampli_rewrite_events,ampli_record_candidate_factor
  double precision,public,save :: ampli_absolute_uncertainty=0d0
  type(staged_integrator),save :: sampler
  double precision,save :: stream_max(2)=0d0,virtual_probability=0d0
  integer,save :: checkpoint_stage=-1
  integer,parameter :: checkpoint_version=2
  character(len=4),save :: generation_mode
  integer(kind=8),save :: production_points=0_8
contains
  subroutine ampli_evaluate(fun,values,xfirst)
    double precision,external :: fun
    double precision,intent(out) :: values(nintegrals),xfirst(ndimmax)
    double precision :: base(ndim),x(ndimmax),vol,dummy,tmp(ndim),unused
    integer :: folds(ndimmax),iret,ifl
    call sampler%sample(tmp,unused,base)
    folds=1
    ifl=0
    x=0d0
    new_point=.true.
    do
       call sampler%map_fold(base,folds(1:ndim),ifold(1:ndim),x(1:ndim),vol)
       call nlops_prepare_point(x)
       if (ifl.eq.0) xfirst=x
       values=0d0
       dummy=fun(x,vol,ifl,values)
       ifl=1
       call nlops_next_fold(folds,iret)
       if (iret.ne.0) exit
    enddo
    values=0d0
    dummy=fun(x,vol,2,values)
  end subroutine ampli_evaluate

  subroutine ampli_calibration_point(x,vol,folds)
    double precision :: x(ndimmax),vol
    integer :: folds(ndimmax)
    x=0d0
    folds=1
    call sampler%sample(x(1:ndim),vol)
    call nlops_prepare_point(x)
  end subroutine ampli_calibration_point

  subroutine ampli_integrate(fun)
    double precision,external :: fun
    double precision :: values(nintegrals),x(ndimmax),r(nintegrals),e(nintegrals)
    double precision :: sums(nintegrals),variance(nintegrals),target,relerr
    double precision :: target_mean,target_m2,delta,target_variance
    integer :: iteration,ipoint,points,limit,total,phase
    logical :: adapt,converged
    if (imode.ne.0.and.imode.ne.1) error stop 'Invalid AmpliCol integration stage'
    if (imode.eq.0) then
       if (any(ifold(1:ndim).ne.1)) error stop 'AmpliCol adaptation requires unfolded stage 0'
       call sampler%init(ndim,nintegrals)
       call nlops_init_auxiliary(.true.)
    else
       call ampli_read_checkpoint(0)
       call nlops_init_auxiliary(.false.)
    endif
    adapt=imode.eq.0
    limit=max(itmax,4)
    points=1024
    if (ncalls0.gt.0) points=ncalls0
    ! Born-spreading calibration changes the target. Discard the previous
    ! estimates and repeat adaptation after the normalized table is fitted.
    do phase=1,2
       sums=0d0
       variance=0d0
       target_variance=0d0
       total=0
       stream_max=0d0
       converged=.false.
       do iteration=1,limit
          call sampler%begin_iteration(adapt)
          target_mean=0d0
          target_m2=0d0
          do ipoint=1,points
             call ampli_evaluate(fun,values,x)
             if (adapt) call nlops_train_virtual(x,values)
             target=values(1)
             if (.not.adapt.and..not.only_virt) target=target+values(5)
             delta=target-target_mean
             target_mean=target_mean+delta/dble(ipoint)
             target_m2=target_m2+delta*(target-target_mean)
             call sampler%observe(x(1:ndim),values,target)
             stream_max(1)=max(stream_max(1),values(1))
             stream_max(2)=max(stream_max(2),values(5)*virtual_fraction(1))
          enddo
          call sampler%finish_iteration(r,e)
          sums=sums+r*dble(points)
          variance=variance+(e*dble(points))**2
          if (points.gt.1) target_variance=target_variance+ &
               max(target_m2,0d0)*dble(points)/dble(points-1)
          total=total+points
          ans(:,1)=sums/dble(total)
          unc(:,1)=sqrt(variance)/dble(total)
          ampli_absolute_uncertainty=sqrt(target_variance)/dble(total)
          ans(:,0)=ans(:,1)
          unc(:,0)=unc(:,1)
          if (adapt) call nlops_update_auxiliary(e)
          target=ans(1,1)
          relerr=ampli_absolute_uncertainty
          if (.not.adapt.and..not.only_virt) then
             target=target+ans(5,1)
          endif
          if (target.gt.0d0) relerr=relerr/target
          write(*,*) 'AmpliCol iteration, trials, ABS, signed, relative error:', &
               iteration,total,target,ans(2,1),relerr
          converged=iteration.ge.4.and.(target.eq.0d0.or.(accuracy.gt.0d0.and.relerr.lt.accuracy))
          if (converged) exit
          if (ncalls0.le.0) points=min(2*points,1048576)
       enddo
       if (imode.ne.0.or..not.born_spread_active.or.born_spread_ready) exit
       call calibrate_born_spreading(fun,ampli_calibration_point)
       call born_spread_write_table
       born_spread_ready=.true.
       born_spread_phase=3
       call reset_MC_grid
       points=1024
    enddo
    if (.not.converged) then
       if (imode.eq.1) error stop 'AmpliCol survey failed to reach requested channel accuracy'
       write(*,*) 'AmpliCol adaptation iteration limit reached; continuing to survey'
    endif
    if (only_virt.and.imode.eq.1) then
       ans(3,1)=0d0
       ans(5,1)=0d0
       ans(:,0)=ans(:,1)
    endif
    ncalls0=total
    itmax=min(iteration,limit)
    checkpoint_stage=imode
    call ampli_write_checkpoint
    call nlops_write_results(ampli_absolute_uncertainty)
  end subroutine ampli_integrate

  subroutine ampli_write_checkpoint
    integer :: iu,ios,ini_fin_fks
    common /fks_channels/ ini_fin_fks
    open(newunit=iu,file='ampli_grids.tmp',status='replace',action='write')
    write(iu,*) 'MG5_AMPLICOL ',checkpoint_version,checkpoint_stage
    write(iu,*) ndim,iconfig,ini_fin_fks,nintegrals
    write(iu,*) ifold(1:ndim)
    write(iu,*) ans(:,1),unc(:,1),stream_max
    write(iu,*) ampli_absolute_uncertainty
    call sampler%save(iu)
    call nlops_save_auxiliary(iu)
    close(iu)
    call rename('ampli_grids.tmp','ampli_grids',ios)
    if (ios.ne.0) error stop 'Cannot install AmpliCol checkpoint'
  end subroutine ampli_write_checkpoint

  subroutine ampli_read_checkpoint(required_stage)
    integer,intent(in) :: required_stage
    integer :: iu,ios,version,dimension,configuration,sector,nval,ini_fin_fks
    integer :: folds(ndim)
    character(len=32) :: tag
    logical :: exists
    common /fks_channels/ ini_fin_fks
    inquire(file='grid.MC_integer',exist=exists)
    if (.not.exists) error stop 'Missing MC_integer state for AmpliCol checkpoint'
    open(newunit=iu,file='ampli_grids',status='old',action='read',iostat=ios)
    if (ios.ne.0) error stop 'Missing AmpliCol checkpoint; run integration first'
    read(iu,*,iostat=ios) tag,version,checkpoint_stage
    if (ios.ne.0) error stop 'Invalid AmpliCol checkpoint header'
    if (tag.ne.'MG5_AMPLICOL'.or.version.ne.checkpoint_version.or.checkpoint_stage.ne.required_stage) &
         error stop 'Incompatible AmpliCol checkpoint version or stage'
    read(iu,*) dimension,configuration,sector,nval
    if (dimension.ne.ndim.or.configuration.ne.iconfig.or.sector.ne.ini_fin_fks.or.nval.ne.nintegrals) &
         error stop 'AmpliCol checkpoint belongs to a different channel'
    read(iu,*) folds
    if (required_stage.eq.1.and.any(folds.ne.ifold(1:ndim))) &
         error stop 'AmpliCol checkpoint folding differs from requested folding'
    read(iu,*) ans(:,1),unc(:,1),stream_max
    read(iu,*) ampli_absolute_uncertainty
    ans(:,0)=ans(:,1)
    unc(:,0)=unc(:,1)
    call sampler%load(iu,ndim,nintegrals)
    call nlops_load_auxiliary(iu)
    close(iu)
  end subroutine ampli_read_checkpoint

  subroutine ampli_load_production
    call ampli_read_checkpoint(1)
    call nlops_init_auxiliary(.false.)
  end subroutine ampli_load_production

  subroutine ampli_start_generation(quota)
    integer,intent(in) :: quota
    character(len=4) :: abrv
    character(len=32) :: tag
    double precision :: total
    integer :: iu,ios,version,generated_target,final_quota
    integer(kind=8) :: safety_limit
    logical :: exists
    common /to_abrv/ abrv
    generation_mode=abrv
    total=ans(1,1)+ans(5,1)
    if (total.le.0d0.and.quota.gt.0) error stop 'Cannot generate events from a zero AmpliCol rate'
    virtual_probability=0d0
    if (generation_mode.ne.'born'.and..not.only_virt.and.total.gt.0d0) &
         virtual_probability=max(1d-3,min(1d0-1d-3,ans(5,1)/total))
    generated_target=quota
    final_quota=quota
    inquire(file='ampli_job.dat',exist=exists)
    if (exists) then
       open(newunit=iu,file='ampli_job.dat',status='old',action='read')
       read(iu,*,iostat=ios) tag,version
       if (ios.ne.0) error stop 'Invalid AmpliCol production job header'
       if (tag.ne.'MG5_AMPLI_JOB'.or.version.ne.2) error stop 'Incompatible AmpliCol production job'
       read(iu,*,iostat=ios) generated_target,final_quota
       if (ios.ne.0.or.generated_target.ne.quota.or.final_quota.lt.0.or. &
            final_quota.gt.generated_target.or.(quota.gt.0.and.final_quota.eq.0)) &
            error stop 'Invalid AmpliCol production event quotas'
       close(iu)
    elseif (quota.gt.0) then
       error stop 'Missing AmpliCol production event quotas'
    endif
    production_points=0_8
    ! This is an emergency ceiling, never a point-budget stopping target.
    ! Completion requires both the event quota and all three tail-mass checks.
    safety_limit=max(10000000_8,100000_8*int(quota,8))
    if (quota.eq.0) safety_limit=0_8
    ! A complete folded evaluation is one observation. Only coordinates that
    ! are not folded may adapt; all MC@NLO auxiliary physics stays surveyed.
    call sampler%start_native_production(quota,final_quota,ifold(1:ndim),total, &
         max(total,tiny(1d0)),safety_limit)
  end subroutine ampli_start_generation

  subroutine ampli_next_candidate(fun,to_write,done)
    double precision,external :: fun
    logical,intent(out) :: to_write,done
    double precision :: values(nintegrals),x(ndimmax),probability,ran2,point(2),rates(2),errors(2),moments(5)
    logical :: iteration_done
    character(len=4) :: abrv
    common /to_abrv/ abrv
    external ran2
    ! The last candidate's LHE factor must be recorded before an iteration
    ! can finalize its acceptance thresholds and tail bounds.
    if (sampler%native_iteration_done.and..not.sampler%production_done) then
       call sampler%finish_native_iteration(done)
       call sampler%native_rates(rates,errors,moments)
       write(*,*) 'AmpliCol native production iteration, trials, ABS, signed, errors:', &
            sampler%native_iteration,production_points,rates,errors
       write(*,*) 'AmpliCol native grid updates, next nonzero target:', &
            sampler%adaptation_updates,sampler%native_target_nonzero
    endif
    if (sampler%production_done) then
       to_write=.false.
       done=.true.
       return
    endif
    probability=1d0
    abrv=generation_mode
    if (generation_mode.ne.'born'.and..not.only_virt) then
       if (ran2().lt.virtual_probability) then
          abrv='virt'
          probability=virtual_probability
       else
          abrv='novi'
          probability=1d0-virtual_probability
       endif
    endif
    call ampli_evaluate(fun,values,x)
    point=[values(1),values(2)]/probability
    if (.not.all(ieee_is_finite(point))) error stop 'Nonfinite AmpliCol production contribution'
    production_points=production_points+1_8
    call sampler%native_consider(point,x(1:ndim),to_write,iteration_done)
    done=.false.
  end subroutine ampli_next_candidate

  subroutine ampli_record_candidate_factor(factor,done)
    double precision,intent(in) :: factor
    logical,intent(out) :: done
    call sampler%record_candidate_factor(factor,done)
  end subroutine ampli_record_candidate_factor

  subroutine ampli_finish_pool
    double precision :: rates(2),errors(2),moments(5)
    integer :: iu,ios
    if (.not.sampler%quota_complete) error stop 'AmpliCol generation safety limit reached before event quota'
    if (sampler%overweight.ge.1d-2) error stop 'AmpliCol generation safety limit reached before one-percent tail bound'
    if (production_points.ne.sampler%ntrials) error stop 'AmpliCol production moments are incomplete'
    open(newunit=iu,file='ampli_pool.dat.tmp',status='replace',action='write')
    call sampler%write_native_pool(iu)
    close(iu)
    call rename('ampli_pool.dat.tmp','ampli_pool.dat',ios)
    if (ios.ne.0) error stop 'Cannot install AmpliCol candidate metadata'
    ! Report production alone for this worker. Python combines all completed
    ! iterations with the parent survey exactly once, even for split jobs.
    call sampler%native_rates(rates,errors,moments)
    ans(1,1)=rates(1)
    ans(2,1)=rates(2)
    ans(5,1)=0d0
    unc(1,1)=errors(1)
    unc(2,1)=errors(2)
    unc(5,1)=0d0
    ans(:,0)=ans(:,1)
    unc(:,0)=unc(:,1)
    ampli_absolute_uncertainty=errors(1)
    ncalls0=int(min(production_points,int(huge(ncalls0),8)))
    itmax=sampler%native_iteration
    write(*,*) 'AmpliCol generation trials, candidates, requested events:', &
         production_points,sampler%ncandidates,sampler%quota
    write(*,*) 'AmpliCol full-trial, reserve, worst-collected tail fractions:', &
         sampler%full_trial_tail,sampler%reserve_tail,sampler%worst_subset_tail
    write(*,*) 'AmpliCol production adaptation dimensions, updates:', &
         count(sampler%adaptation_mask),sampler%adaptation_updates
  end subroutine ampli_finish_pool

  ! Only XWGTUP changes. The internal mgrwgt coefficients and reference weight
  ! stay together: the normal MG5 reweighter subsequently multiplies their
  ! ratio by this corrected XWGTUP. Shower scales and event attributes survive.
  subroutine ampli_rewrite_events(source,destination,weights,expected)
    character(len=*),intent(in) :: source,destination
    double precision,intent(in) :: weights(:)
    integer,intent(in) :: expected
    character(len=65536) :: line
    integer :: iu,ou,ios,candidate,kept,nup,idprup
    double precision :: xwgt,scalup,aqed,aqcd
    logical :: inside,keep
    open(newunit=iu,file=source,status='old',action='read')
    open(newunit=ou,file=destination,status='replace',action='write')
    candidate=0
    kept=0
    inside=.false.
    keep=.true.
    do
       read(iu,'(a)',iostat=ios) line
       if (ios.lt.0) exit
       if (ios.ne.0) error stop 'Cannot read AmpliCol candidate events'
       if (index(adjustl(line),'<event>').eq.1.or.index(adjustl(line),'<event ').eq.1) then
          candidate=candidate+1
          if (candidate.gt.size(weights)) error stop 'Too many AmpliCol candidate events'
          inside=.true.
          keep=weights(candidate).gt.0d0
          if (keep) write(ou,'(a)') trim(line)
          read(iu,*,iostat=ios) nup,idprup,xwgt,scalup,aqed,aqcd
          if (ios.ne.0) error stop 'Invalid AmpliCol candidate event header'
          if (keep) then
             kept=kept+1
             write(ou,'(1x,i4,1x,i6,4(1x,es24.16))') nup,idprup,xwgt*weights(candidate),scalup,aqed,aqcd
          endif
       else
          if (.not.inside.or.keep) write(ou,'(a)') trim(line)
          if (index(line,'</event>').ne.0) inside=.false.
       endif
    enddo
    close(iu)
    close(ou)
    if (inside.or.candidate.ne.size(weights).or.kept.ne.expected) &
         error stop 'Incomplete AmpliCol candidate selection'
  end subroutine ampli_rewrite_events
end module ampli_mint_adapter
