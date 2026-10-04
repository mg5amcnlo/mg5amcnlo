! AmpliCol numerical backend for the existing one-channel MC@NLO job protocol.
! The run manager owns channel quotas and normalization. Physics stays in
! sigintF and the shared MC@NLO services of mint_module.
module ampli_mint_adapter
  use simple_integrator_mod, only: staged_integrator
  use mint_module
  implicit none
  private
  public :: ampli_integrate,ampli_load_production,ampli_start_generation
  public :: ampli_next_candidate,ampli_final_weights,ampli_rewrite_events
  double precision,public,save :: ampli_absolute_uncertainty=0d0
  type(staged_integrator),save :: sampler
  double precision,save :: stream_max(2)=0d0,virtual_probability=0d0
  integer,save :: checkpoint_stage=-1
  integer,parameter :: checkpoint_version=2
  character(len=4),save :: generation_mode
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
    if (.not.converged) write(*,*) 'AmpliCol integration iteration limit reached; reported uncertainty applies'
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
    double precision :: envelope,total
    common /to_abrv/ abrv
    generation_mode=abrv
    total=ans(1,1)+ans(5,1)
    if (total.le.0d0.and.quota.gt.0) error stop 'Cannot generate events from a zero AmpliCol rate'
    virtual_probability=0d0
    if (generation_mode.ne.'born'.and..not.only_virt.and.total.gt.0d0) &
         virtual_probability=ans(5,1)/total
    envelope=0d0
    if (virtual_probability.lt.1d0) envelope=stream_max(1)/(1d0-virtual_probability)
    if (virtual_probability.gt.0d0) envelope=max(envelope,stream_max(2)/virtual_probability)
    ! A fixed survey envelope makes even one-event channel quotas ordinary
    ! rejection samples. Observed violations must invalidate production;
    ! changing the envelope conditional on accepted events would bias them.
    call sampler%start_production(quota,2d0*max(envelope,total,tiny(1d0)), &
         max(1000000_8,100000_8*int(quota,8)),overweight_tolerance=0d0)
  end subroutine ampli_start_generation

  subroutine ampli_next_candidate(fun,to_write,done)
    double precision,external :: fun
    logical,intent(out) :: to_write,done
    double precision :: values(nintegrals),x(ndimmax),probability,ran2
    character(len=4) :: abrv
    common /to_abrv/ abrv
    external ran2
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
    call sampler%consider(values(1)/probability,to_write,done)
    if (sampler%envelope_exceeded) &
         error stop 'AmpliCol production envelope exceeded; repeat integration with more statistics'
    if (done.and..not.sampler%quota_complete) error stop 'AmpliCol production budget exhausted before quota'
  end subroutine ampli_next_candidate

  subroutine ampli_final_weights(weights)
    double precision,allocatable,intent(out) :: weights(:)
    call sampler%final_weights(weights)
    ! MG5's sum/unity conventions advertise unweighted events (IDWTUP=-3).
    ! Strict frozen-envelope rejection must return unit factors throughout.
    if (sampler%overweight.gt.0d0) error stop 'AmpliCol production budget exhausted with overweight events'
    write(*,*) 'AmpliCol production trials, candidates, quota, overweight:', &
         sampler%ntrials,sampler%ncandidates,sampler%quota,sampler%overweight
  end subroutine ampli_final_weights

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
