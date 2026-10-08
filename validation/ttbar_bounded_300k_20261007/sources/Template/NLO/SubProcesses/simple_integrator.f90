! Vendored from AmpliCol SimpleIntegrator/simple_integrator.f03, commit
! 61b9cd52f44b4cdfc4773daa197e94eeb482b560 (2026-10-04).
! The original integrator and helper module names are retained. MG5 additions
! are the staged_integrator type and staged_* procedures below. They reuse the
! original adaptive grids; MC@NLO production adapts only unfolded dimensions
! and uses quota-driven priority selection with repeated threshold evaluation and
! a full overweight-tail cross-section bound. Process allocation, normalization,
! physics and RNG ownership stay with MG5. No second RNG is supplied.
!===============================================================================
! SimpleIntegrator
!===============================================================================
!
! Purpose
! -------
! This module provides a compact adaptive Monte Carlo integrator with optional
! unweighted event generation.  It is intended for matrix-element, phase-space,
! or similar event-generation programs where the caller owns the physics
! function and the integrator owns:
!
!   * adaptive one-dimensional grids for every integration variable,
!   * distribution of requested points across channels and integrals,
!   * accumulated signed and absolute integral estimates,
!   * candidate-event storage and final unweighted-event weights.
!
! Public module and dependencies
! ------------------------------
! Use the public type through
!
!     use simple_integrator_mod
!     type(integrator) :: integ
!
! The module expects the helper modules in helper_modules.f03 and the random
! number function ran2() from ranmar.f (or a caller-provided compatible
! function returning a double precision number in (0,1)).  Several progress
! messages are written to stdout and to Fortran unit 99.  Standalone programs
! should open unit 99 before using the integrator if they want to control the
! log destination; otherwise many compilers create a default file for that unit.
!
! Concepts
! --------
! A "channel" is a separately adapted sampling map, usually a phase-space
! channel.  Each channel can contain one or more "integrals", for example
! subprocesses sharing the same channel.  The public channel and integral labels
! returned by get_points are one-based indices.
!
! Each point has ndim adapted coordinates and ndim_extra flat random
! coordinates.  Only the first ndim coordinates are included in adaptive grid
! weights and in compute_wgt_from_x.  The caller may use ndim_extra for random
! choices that should not affect grid adaptation.
!
! Public workflow
! ---------------
! 1. Initialise once:
!
!        call integ%init(nchannel, ndim, ndim_extra, nintegral, &
!             nevts_unw_req, niters)
!
!    ndim, ndim_extra, and nintegral are arrays of length nchannel.
!    nevts_unw_req is the requested number of unweighted events.  niters is the
!    maximum number of adaptation/generation iterations.
!
! 2. Repeatedly request points, evaluate the caller's integrand, and return the
!    values:
!
!        done = .false.
!        do while (.not. done)
!           call integ%get_points(npoints, ichan, iint)
!           do ip = 1, npoints
!              ! integ%x(:,ip) contains the random point.
!              ! integ%wgt(ip) is the grid Jacobian/volume factor.
!              ! Compute f_abs(ip) >= 0 and f(ip), including integ%wgt(ip).
!           end do
!           call integ%fill_points(npoints, f_abs, f, to_write, done)
!           ! If to_write(ip) is true, write/store the corresponding event now.
!        end do
!
!    fill_points may be called with fewer points than were returned by
!    get_points; unused trailing points are discarded.  A new get_points call
!    must not be made until the previous batch has been returned with
!    fill_points.
!
! 3. After done is true, retrieve final event weights:
!
!        call integ%assign_evnt_wgts(wgts)
!
!    wgts has shape (3, number_of_written_candidate_events).  Column i contains
!    the nominal event weight, the adjusted event weight including any
!    overweight correction, and the overweight excess.  Events rejected by the
!    final unweighting have zero weights.
!
! Integrand convention
! --------------------
! f_abs is the non-negative target used for grid adaptation, maximum-weight
! estimates, and unweighting.  f is the signed contribution to the physical
! integral.  In many event generators f_abs=abs(f), but callers may choose a
! different positive envelope if needed.
!
! compute_wgt_from_x can be used when an external multichannel combination needs
! the current adaptive-grid weight for a point already known in a specific
! channel.
!
! Tunable internal parameters
! ---------------------------
! The parameters below the type declarations control minimum statistics, grid
! sizes, event-generation startup, and allowed overweight fraction.  They are
! compile-time parameters in this standalone version.
!
! Limitations
! -----------
! This implementation is serial and keeps candidate events in memory until final
! weighting.  It does not currently read or write grids; read_all_grids and
! write_all_grids are placeholders.
!
module simple_integrator_mod
  implicit none
  private
  ! One adaptive sampling channel.  A channel owns one grid per adapted
  ! dimension and one or more integral estimates sharing those grids.
  type :: channel
     integer :: ndim,nintegral,current_integral,current_iter&
          &,number,max_iters,nevts_unw_req,ndim_extra
     integer(kind=8) :: npoints,npoints_iter
     real(kind=8),dimension(2) :: res,unc,res_iter,res2_iter,unc_iter
     real(kind=8) :: overweight
     logical :: done,evgen_done
     type(grid),allocatable,dimension(:,:) :: grids
     type(integral),allocatable,dimension(:) :: integrals
   contains
     procedure,private :: init => channel_init
     procedure,private :: add_point => channel_add_point
     procedure,private :: get_point => channel_get_point
     procedure,private :: update_result_iter => channel_update_result_iter
     procedure,private :: combine_iters => channel_combine_iters
     procedure,private :: print_result_iter => channel_print_result_iter
     procedure,private :: print_combined_result => channel_print_combined_result
     procedure,private :: init_next_iter => channel_init_next_iter
     procedure,private :: check_gen_evnts => channel_check_gen_evnts
     procedure,private :: update_grids => channel_update_grids
     procedure,private :: update_nevts_unw_req => channel_update_nevts_unw_req
     procedure,private :: recompute_wgt_from_x
  end type channel
  ! One physical integral inside a channel.  It accumulates iteration estimates
  ! and stores candidate events until the final unweighting decision.
  type :: integral
     real(kind=8) :: max_value,overweight
     real(kind=8),dimension(:),allocatable :: f_max
     real(kind=8),dimension(2) :: res,unc,res_iter,res2_iter,accum&
          &,accum2,unc_iter
     integer :: ichan,nevts_unw_gen,evnt,nevnt_in_list,ndim &
          &,current_iter,max_iters,nevts_unw_req
     integer(kind=8) :: npoints_iter,npoints,npoints_requested&
          &,npoints_nonzero,npoints_nonzero_total
     logical :: done,evgen_done
     type(evnt),dimension(:),allocatable :: evnt_list
   contains
     procedure,private :: init => integral_init
     procedure,private :: add_point => integral_add_point
     procedure,private :: update_result_iter => integral_update_result_iter
     procedure,private :: combine_iters => integral_combine_iters
     procedure,private :: init_next_iter => integral_init_next_iter
     procedure,private :: compute_fmax => integral_compute_fmax
     procedure,private :: compute_fmax_next_iter => integral_compute_fmax_next_iter
     procedure,private :: unwgt => integral_unwgt
     procedure,private :: update_max_value,check_write_evnt,increase_size_evnt_list,compute_wgts,check_overweight
  end type integral
  ! One monotone one-dimensional adaptive grid.  current maps uniform random
  ! cells to physical integration coordinates; accum stores the adaptation data.
  type :: grid
     integer :: size,size_fill
     real(kind=8),allocatable,dimension(:) :: current,accum,current_for_fillcell
     integer,allocatable,dimension(:) :: nhits
   contains
     procedure,private :: init => grid_init
     procedure,private :: add_point => grid_add_point
     procedure,private :: get_x,get_wgt,massage_accum,find_cell&
          &,interpolate_current,find_cell_to_fill
     procedure,private :: update => grid_update
  end type grid
  ! One completed native production iteration. Rates always use the original
  ! proposal weights, including zero/rejected draws. Historical maps are used
  ! only to reassess candidate envelopes across iterations.
  type :: native_production_epoch
     type(grid),allocatable :: maps(:)
     integer(kind=8) :: trials=0_8,nonzero=0_8,target_nonzero=0_8
     real(kind=8) :: mean(2)=0d0,m2(2)=0d0,covariance=0d0
     real(kind=8) :: cutoff=0d0,envelope=0d0,threshold=0d0
     logical :: eligible=.false.
  end type native_production_epoch
  ! Candidate event metadata saved during event generation.
  type :: evnt
     real(kind=8),allocatable,dimension(:) :: x,f_abs
     real(kind=8) :: wgt,rnd,overwgt
     integer :: iter,label
     logical :: unwgt
  end type evnt
  ! Public driver object.  Users call init, then alternate get_points and
  ! fill_points until done, then call assign_evnt_wgts if events were written.
  type,public :: integrator
     integer :: nchannel,current_channel,nevts_unw_req,npoints_gen
     integer(kind=8) :: npoints_requested
     real(kind=8),dimension(2) :: res,unc
     real(kind=8),allocatable,dimension(:,:),public :: x
     real(kind=8),allocatable,dimension(:),public :: wgt
     type(channel),allocatable,dimension(:) :: channels
     ! x and wgt are allocated by get_points and released by fill_points.
   contains
     procedure,public :: init,get_points,fill_points,compute_wgt_from_x,assign_evnt_wgts
     procedure,private :: read_all_grids,write_all_grids&
          &,get_channel_and_integral,update_points_requested&
          &,print_results,compute_total_rate,update_nevts_unw_req&
          &,count_unweighted_evnts,init_next_iter&
          &,get_npoints_nonzero_iter,finalise_iter,update_grids
  end type integrator
  ! Single-channel interface for MG5's separately scheduled integration stages.
  ! Values passed to observe already include the sampling Jacobian. Its first
  ! component need not be the signed rate; all components have independent
  ! first/second moments. Grid adaptation uses the separate nonnegative target.
  !
  ! sample optionally returns the underlying uniform variates. map_fold maps
  ! these through the grid at (u+k-1)/F, including 1/product(F) in the Jacobian.
  ! Thus it implements MINT-style grid-quantile folding with frozen grids.
  !
  ! save/load preserve numerical state, including active-iteration accumulators
  ! and candidate metadata. The caller must checkpoint its own RNG and LHE
  ! payload alongside this state for exact interrupted-job continuation.
  !
  ! Production with overweight_tolerance=0 uses ordinary rejection against the
  ! supplied, frozen envelope. It is exact for a valid bound, including quotas
  ! of one event. An observed violation invalidates the production attempt;
  ! finite pilot statistics alone cannot certify absence of unseen tails.
  ! Positive tolerance selects the upstream-style finite-pool rank scheme with
  ! normalized overweight corrections. This mode is approximate at finite
  ! quota and must not be used to promise strict unweighted event densities.
  !
  ! Legacy start_pool/consider_pool (unused by current MC@NLO) collect an immutable-budget production
  ! batch. The positive cutoff only controls storage: every point with priority
  ! weight/uniform above it is retained, including arbitrarily large weights.
  ! A coordinator may raise that cutoff and combine independent batches before
  ! final selection. No event quota or final selection is imposed here.
  type,public :: staged_integrator
     private
     integer,public :: ndim=0,nvalues=0,ncandidates=0,quota=0,final_quota=0
     integer(kind=8),public :: npoints=0_8,total_points=0_8,ntrials=0_8
     real(kind=8),public :: max_weight=0d0,envelope=0d0,overweight=0d0
     real(kind=8),public :: total_abs=0d0,threshold=0d0,full_trial_tail=0d0,reserve_tail=0d0,worst_subset_tail=0d0
     real(kind=8),allocatable,public :: res(:),unc(:)
     logical,public :: quota_complete=.false.,exhausted=.false.,production_done=.false.
     logical,public :: envelope_exceeded=.false.
     integer,public :: adaptation_updates=0
     logical,allocatable,public :: adaptation_mask(:)
     integer(kind=8),public :: adaptation_interval=0_8,adaptation_batch=0_8,adaptation_points=0_8
     type(grid),allocatable :: maps(:)
     real(kind=8),allocatable :: mean(:),m2(:),candidate_weight(:),candidate_priority(:),candidate_factor(:)
     integer(kind=8) :: max_trials=0_8,next_pool_check=0_8
     real(kind=8) :: overweight_tolerance=1d-2
     logical :: iteration_active=.false.,adapt_iteration=.false.,production_active=.false.
     logical :: pool_mode=.false.
     logical,public :: native_mode=.false.,native_iteration_done=.false.
     integer,public :: native_iteration=0
     integer(kind=8),public :: native_target_nonzero=0_8,native_nonzero=0_8
     real(kind=8),public :: native_effective_generated=0d0,native_expected_remaining=0d0,native_generation_efficiency=0d0
     integer :: native_max_iterations=0
     type(native_production_epoch),allocatable :: native_epochs(:)
     real(kind=8),allocatable :: native_candidate_x(:,:),native_birth_logjac(:),native_correction(:)
     integer,allocatable :: native_birth_epoch(:),native_tail(:)
     real(kind=8) :: native_mean(2)=0d0,native_m2(2)=0d0,native_covariance=0d0,native_log_z=0d0
   contains
     procedure,public :: init => staged_init
     procedure,public :: begin_iteration => staged_begin_iteration
     procedure,public :: sample => staged_sample
     procedure,public :: map_fold => staged_map_fold
     procedure,public :: observe => staged_observe
     procedure,public :: finish_iteration => staged_finish_iteration
     procedure,public :: save => staged_save
     procedure,public :: load => staged_load
     procedure,public :: start_production => staged_start_production
     procedure,public :: consider => staged_consider
     procedure,public :: final_weights => staged_final_weights
     procedure,public :: record_candidate_factor => staged_record_candidate_factor
     procedure,public :: production_candidates => staged_production_candidates
     procedure,public :: start_pool => staged_start_pool
     procedure,public :: consider_pool => staged_consider_pool
     procedure,public :: pool_candidates => staged_pool_candidates
     procedure,public :: export_pool => staged_export_pool
     procedure,private :: select_candidates => staged_select_candidates
     procedure,private :: grow_candidates => staged_grow_candidates
     procedure,private :: reset_production_adaptation => staged_reset_production_adaptation
     procedure,private :: train_production => staged_train_production
     procedure,public :: start_native_production
     procedure,public :: native_consider,finish_native_iteration,native_rates,write_native_pool
     procedure,private :: reset_native_production,native_log_jacobian,native_reweighted_value
     procedure,private :: native_update_envelopes,native_select,native_update_maps,native_next_cutoff
  end type staged_integrator
  double precision, external :: ran2
  integer,save :: iters_without_evnts,evnt_label=0
  integer,parameter :: importance_sampling_strategy=3
  ! Fraction of largest candidate weights ignored when estimating f_max.
  real(kind=8),parameter :: write_evnt_fraction=0.05d0
  integer,parameter :: min_points_per_channel=1024
  integer,parameter :: min_points_per_integral=128
  logical,parameter :: turn_off_evnt_generation=.false.
  real(kind=8),parameter :: required_accuracy_factor=10d0
  integer,parameter :: min_grid_size=8
  integer,parameter :: max_grid_size=2048
  ! Keep the MC@NLO pool coordinator's ALLOWED_OVERWEIGHT_FACTOR in sync.
  real(kind=8),parameter :: allowed_overweight_factor=0.01d0
  integer,parameter :: final_n_iters_for_evnt_gen=8
contains

  ! Initialise the public integrator object and all channel/integral state.
  subroutine init(this,nchannel,ndim,ndim_extra,nintegral,nevts_unw_req,niters)
    implicit none
    class(integrator),intent(inout) :: this
    integer,intent(in) :: nchannel,nevts_unw_req,niters
    integer,dimension(nchannel),intent(in) :: ndim,nintegral,ndim_extra
    integer :: i
    if (nchannel.lt.1) then
       write (*,*) 'ERROR: nchannel must be at least 1'
       stop 1
    endif
    if (niters.lt.1) then
       write (*,*) 'ERROR: niters must be at least 1'
       stop 1
    endif
    if (nevts_unw_req.lt.1) then
       write (*,*) 'ERROR: nevts_unw_req must be at least 1'
       stop 1
    endif
    if (any(ndim.lt.1)) then
       write (*,*) 'ERROR: all channels must have at least one adapted dimension'
       stop 1
    endif
    if (any(ndim_extra.lt.0)) then
       write (*,*) 'ERROR: ndim_extra cannot be negative'
       stop 1
    endif
    if (any(nintegral.lt.1)) then
       write (*,*) 'ERROR: all channels must have at least one integral'
       stop 1
    endif
    if (allocated(this%channels)) deallocate(this%channels)
    if (allocated(this%x)) deallocate(this%x)
    if (allocated(this%wgt)) deallocate(this%wgt)
    evnt_label=0
    this%nchannel=nchannel
    this%nevts_unw_req=nevts_unw_req
    ! if we assume 1% unweighting efficiency, we expect ~10% time
    ! spend in iterations that do not produce events:
    iters_without_evnts=5
    this%npoints_requested=int(nevts_unw_req/(0.1d0*2**iters_without_evnts),kind=8)
    do while (this%npoints_requested/this%nchannel.lt.max(min_points_per_channel,min_points_per_integral*maxval(nintegral)) &
         .and. iters_without_evnts.gt.3)
       iters_without_evnts=iters_without_evnts-1
       this%npoints_requested=int(nevts_unw_req/(0.03*2**iters_without_evnts),kind=8)
    enddo
    this%npoints_requested=max(this%npoints_requested,min_points_per_channel*this%nchannel,&
         min_points_per_integral*maxval(nintegral)*this%nchannel)
    allocate(this%channels(this%nchannel))
    do i=1,this%nchannel
       call this%channels(i)%init(ndim(i),ndim_extra(i),nintegral(i),this%npoints_requested/nchannel,niters,i)
    enddo
    this%current_channel=0
    this%npoints_gen=0
    this%res=0d0
    this%unc=0d0
  end subroutine init

  ! Initialise one channel, including its first set of grids and integrals.
  subroutine channel_init(this,ndim,ndim_extra,nintegral,npoints,niters,ichan)
    implicit none
    class(channel),intent(inout) :: this
    integer,intent(in) :: ndim,nintegral,niters,ichan,ndim_extra
    integer(kind=8) :: npoints
    integer :: i
    this%ndim=ndim
    this%ndim_extra=ndim_extra
    this%max_iters=niters
    this%nintegral=nintegral
    this%number=ichan
    this%nevts_unw_req=0
    allocate(this%grids(1:this%ndim,1:this%max_iters+1))
    allocate(this%integrals(1:this%nintegral))
    do i=1,this%ndim
       call this%grids(i,1)%init(npoints)
    enddo
    do i=1,this%nintegral
       call this%integrals(i)%init(ndim,npoints/this%nintegral,this%number,this%max_iters)
    enddo
    this%current_integral=0
    this%current_iter=0
    this%npoints=0_8
    this%res=0d0
    this%unc=0d0
    this%overweight=0d0
    call this%init_next_iter()
    this%evgen_done=.false.
  end subroutine channel_init

  ! Reset channel accumulators and advance to the next iteration.
  subroutine channel_init_next_iter(this)
    implicit none
    class(channel),intent(inout) :: this
    integer :: i
    this%res_iter=0d0
    this%res2_iter=0d0
    this%unc_iter=0d0
    this%npoints_iter=0_8
    this%done=.false.
    if (all(this%integrals%evgen_done)) then
       this%evgen_done=.true.
       this%done=.true.
    else
       this%evgen_done=.false.
    endif
    do i=1,this%nintegral
       call this%integrals(i)%init_next_iter(this)
    enddo
    this%current_iter=this%current_iter+1
  end subroutine channel_init_next_iter

  ! Reset one integral for a new iteration and choose the active f_max estimate.
  subroutine integral_init_next_iter(this,thischan)
    implicit none
    class(integral),intent(inout) :: this
    class(channel),intent(inout) :: thischan
    this%current_iter=this%current_iter+1
    this%res_iter=0d0
    this%res2_iter=0d0
    this%accum=0d0
    this%accum2=0d0
    this%unc_iter=0d0
    this%npoints_iter=0_8
    this%npoints_nonzero=0_8
    this%evnt=0
    if (.not. this%evgen_done) this%done=.false.
    if (this%current_iter.eq.1) then
       this%f_max(this%current_iter)=-1d0
    elseif (this%current_iter.le.iters_without_evnts+1) then
       this%f_max(this%current_iter)=this%max_value
    else
       call this%compute_fmax_next_iter(thischan)
    endif
    this%max_value=0d0
  end subroutine integral_init_next_iter

  ! Initialise a grid either uniformly or by interpolating a previous grid.
  subroutine grid_init(this,npoints,current)
    implicit none
    class(grid),intent(inout) :: this
    integer(kind=8),intent(in) :: npoints
    real(kind=8),dimension(:),intent(in),optional :: current
    integer :: i,isize
    this%size=max_grid_size
    call determine_sizefill(npoints,this%size_fill)
    allocate(this%current(0:this%size))
    allocate(this%current_for_fillcell(0:this%size_fill))
    if (present(current)) then
       isize=size(current)-1
       if (isize.ne.this%size_fill) then
          call this%interpolate_current(isize,this%size_fill,current,this%current_for_fillcell)
       else
          this%current_for_fillcell=current
       endif
       call this%interpolate_current(isize,this%size,current,this%current)
    else
       do i=0,this%size_fill
          this%current_for_fillcell(i)=dble(i)/this%size_fill
       enddo
       do i=0,this%size
          this%current(i)=dble(i)/this%size
       enddo
    endif
    allocate(this%accum(0:this%size_fill))
    allocate(this%nhits(this%size_fill))
    this%accum=0d0
    this%nhits=0
  end subroutine grid_init

  ! Choose the number of adaptation fill cells from the requested statistics.
  subroutine determine_sizefill(npoints,isize)
    implicit none
    integer(kind=8),intent(in) :: npoints
    integer,intent(out) :: isize
    isize=int(sqrt(dble(npoints))/10)
    isize=max(isize,min_grid_size)
    isize=min(isize,max_grid_size)
  end subroutine determine_sizefill

  ! Initialise one integral estimate and its candidate-event buffer.
  subroutine integral_init(this,ndim,npoints,ichan,niters)
    implicit none
    class(integral),intent(inout) :: this
    integer,intent(in) :: ndim,ichan,niters
    integer(kind=8) :: npoints
    this%ndim=ndim
    this%ichan=ichan
    this%npoints=0_8
    this%npoints_iter=0_8
    this%npoints_nonzero=0_8
    this%npoints_requested=npoints
    this%max_iters=niters
    this%nevts_unw_req=0
    this%nevts_unw_gen=0
    this%evnt=0
    allocate(this%f_max(this%max_iters))
    this%f_max=-1d0
    allocate(this%evnt_list(npoints))
    this%nevnt_in_list=0
    this%evgen_done=.false.
    this%done=.false.
    this%current_iter=0
    this%npoints_nonzero_total=0_8
    this%overweight=0d0
    this%max_value=0d0
    this%res=0d0
    this%unc=0d0
    this%res_iter=0d0
    this%res2_iter=0d0
    this%unc_iter=0d0
    this%accum=0d0
    this%accum2=0d0
  end subroutine integral_init

  ! Select an active channel/integral and generate a batch of random points.
  subroutine get_points(this,npoints,ichan,iint)
    implicit none
    class(integrator),intent(inout) :: this
    integer,intent(in) :: npoints
    integer,intent(out) :: ichan,iint
    integer :: i,ntot
    real(kind=8) :: wgt_chan
    if (npoints.lt.1) then
       write (*,*) 'ERROR: get_points requires at least one point'
       stop 1
    endif
    if (this%npoints_gen.ne.0) then
       write (*,*) 'ERROR: previous points must be returned with fill_points before get_points is called again'
       stop 1
    endif
    if (all(this%channels%done .or. this%channels%evgen_done)) then
       write (*,*) 'ERROR: get_points called after integration is done'
       stop 1
    endif

    call this%get_channel_and_integral(ichan,iint,wgt_chan)
    this%current_channel=ichan

    ntot=this%channels(this%current_channel)%ndim+this%channels(this%current_channel)%ndim_extra
    allocate(this%x(1:ntot,1:npoints))
    allocate(this%wgt(1:npoints))

    do i=1,npoints
       call this%channels(this%current_channel)%get_point(this%x(1,i),this%wgt(i))
    enddo
    this%wgt=this%wgt*wgt_chan
    this%npoints_gen=npoints

  end subroutine get_points

  ! Return evaluated values for the most recent batch and trigger iteration
  ! finalisation when all active channels/integrals have enough statistics.
  subroutine fill_points(this,npoints,f_abs,f,to_write,done)
    implicit none
    class(integrator),intent(inout) :: this
    integer,intent(in) :: npoints
    real(kind=8),dimension(npoints),intent(in) :: f,f_abs
    logical,dimension(npoints),intent(out) :: to_write
    logical,intent(out) :: done
    integer :: i
    done=.false.
    if (this%npoints_gen.eq.0) then
       write (*,*) 'ERROR: fill_points called before get_points'
       stop 1
    endif
    if (npoints.lt.1) then
       write (*,*) 'ERROR: fill_points requires at least one point'
       stop 1
    endif
    if (npoints.gt.this%npoints_gen) then
       write (*,*) 'ERROR: too many points returned'
       stop 1
    endif
    do i=1,npoints
       call this%channels(this%current_channel)%add_point(this%x(1,i),this%wgt(i),f_abs(i),f(i),to_write(i))
    enddo
    this%npoints_gen=0
    if (all(this%channels%done)) then
       call this%finalise_iter(done)
    endif
    deallocate(this%x)
    deallocate(this%wgt)
  end subroutine fill_points

  ! Finish one global iteration: combine rates, unweight candidates, update
  ! grids, and decide whether the requested event sample is complete.
  subroutine finalise_iter(this,done)
    implicit none
    class(integrator),intent(inout) :: this
    logical,intent(out) :: done
    character(len=8) :: date
    character(len=10) :: time
    character(len=5) :: zone
    character(len=19) :: formatted
    integer(kind=8) :: npoints_nonzero
    call date_and_time(date, time, zone)
    write(formatted, '(A4,"-",A2,"-",A2," ",A2,":",A2,":",A2)') &
         date(1:4),date(5:6),date(7:8),time(1:2),time(3:4),time(5:6)
    call this%get_npoints_nonzero_iter(npoints_nonzero)
    write (*,*) ''
    write (*,'(a,x,i4,x,a,x,i10,x,a)') &
         'iteration',this%channels(1)%current_iter,'(',npoints_nonzero, &
         'points) '//trim(formatted)//' :'
    write (99,*) ''
    write (99,'(a,x,i4,x,a,x,i10,x,a)') &
         'iteration',this%channels(1)%current_iter,'(',npoints_nonzero, &
         'points) '//trim(formatted)//' :'
    call this%compute_total_rate()
    call this%count_unweighted_evnts()
    call this%print_results()
    call this%update_grids()
    call this%init_next_iter()
    if (all(this%channels%evgen_done)) done=.true.
    if (turn_off_evnt_generation .and. this%res(1).gt.0d0 .and. &
         this%unc(1)/this%res(1).lt.1d0/(sqrt(dble(this%nevts_unw_req))*required_accuracy_factor)) done=.true.
    if (all(this%channels%done)) done=.true.
    call flush(99)
  end subroutine finalise_iter

  ! Update all active channel grids after an iteration has been finalised.
  subroutine update_grids(this)
    implicit none
    class(integrator),intent(inout) :: this
    integer :: i
    do i=1,this%nchannel
       call this%channels(i)%update_grids()
    enddo
  end subroutine update_grids

  ! Count non-zero points from the current iteration across all integrals.
  subroutine get_npoints_nonzero_iter(this,npoints_nonzero)
    implicit none
    class(integrator),intent(inout) :: this
    integer :: i
    integer(kind=8) :: npoints_nonzero
    npoints_nonzero=0_8
    do i=1,this%nchannel
       npoints_nonzero=npoints_nonzero+sum(this%channels(i)%integrals(1:this%channels(i)%nintegral)%npoints_nonzero)
    enddo
  end subroutine get_npoints_nonzero_iter

  ! Start the next iteration for channels that have not reached max_iters.
  subroutine init_next_iter(this)
    implicit none
    class(integrator),intent(inout) :: this
    integer :: i
    do i=1,this%nchannel
       if (this%channels(i)%current_iter.lt.this%channels(i)%max_iters) then
          call this%channels(i)%init_next_iter()
       endif
    enddo
    call this%update_points_requested()
  end subroutine init_next_iter

  ! Distribute requested events, test stored candidates, and mark completed
  ! channel/integral event-generation tasks.
  subroutine count_unweighted_evnts(this)
    implicit none
    class(integrator),intent(inout) :: this
    integer :: i
    call this%update_nevts_unw_req
    do i=1,this%nchannel
       call this%channels(i)%check_gen_evnts()
    enddo
  end subroutine count_unweighted_evnts

  ! Split the total requested event count over channels in proportion to their
  ! current absolute integral estimates.
  subroutine update_nevts_unw_req(this)
    use sort_array_mod
    implicit none
    class(integrator),intent(inout) :: this
    real(kind=8),dimension(this%nchannel) :: res
    integer :: nevts_to_distribute,i
    integer,dimension(this%nchannel) :: idx
    real(kind=8) :: total
    res=this%channels%res(1)
    call sort_indices_by_values(res,idx)
    nevts_to_distribute=this%nevts_unw_req
    total=this%res(1)
    if (total.le.0d0) then
       do i=1,this%nchannel
          this%channels(i)%nevts_unw_req=0
          call this%channels(i)%update_nevts_unw_req()
       enddo
       return
    endif
    do i=1,this%nchannel-1
       this%channels(idx(i))%nevts_unw_req=int(nevts_to_distribute*this%channels(idx(i))%res(1)/total)
       if (ran2().lt.nevts_to_distribute*this%channels(idx(i))%res(1)/this%res(1)-&
                     this%channels(idx(i))%nevts_unw_req) then
          this%channels(idx(i))%nevts_unw_req=this%channels(idx(i))%nevts_unw_req+1
       endif
       nevts_to_distribute=nevts_to_distribute-this%channels(idx(i))%nevts_unw_req
       total=total-this%channels(idx(i))%res(1)
    enddo
    this%channels(idx(this%nchannel))%nevts_unw_req=nevts_to_distribute
    do i=1,this%nchannel
       call this%channels(i)%update_nevts_unw_req()
    enddo
  end subroutine update_nevts_unw_req

  ! Split one channel's requested event count across its integrals.
  subroutine channel_update_nevts_unw_req(this)
    use sort_array_mod
    implicit none
    class(channel),intent(inout) :: this
    real(kind=8),dimension(this%nintegral) :: res
    integer :: nevts_to_distribute,i
    integer,dimension(this%nintegral) :: idx
    real(kind=8) :: total
    res=this%integrals%res(1)
    call sort_indices_by_values(res,idx)
    nevts_to_distribute=this%nevts_unw_req
    total=this%res(1)
    if (total.le.0d0) then
       this%integrals%nevts_unw_req=0
       return
    endif
    do i=1,this%nintegral-1
       this%integrals(idx(i))%nevts_unw_req=int(nevts_to_distribute*this%integrals(idx(i))%res(1)/total)
       if (ran2().lt.nevts_to_distribute*this%integrals(idx(i))%res(1)/this%res(1)-&
                     this%integrals(idx(i))%nevts_unw_req) then
          this%integrals(idx(i))%nevts_unw_req=this%integrals(idx(i))%nevts_unw_req+1
       endif
       nevts_to_distribute=nevts_to_distribute-this%integrals(idx(i))%nevts_unw_req
       total=total-this%integrals(idx(i))%res(1)
    enddo
    this%integrals(idx(this%nintegral))%nevts_unw_req=nevts_to_distribute
  end subroutine channel_update_nevts_unw_req

  ! Combine active channel estimates into the public total result.
  subroutine compute_total_rate(this)
    implicit none
    class(integrator),intent(inout) :: this
    integer :: i
    do i=1,this%nchannel
       if (.not.this%channels(i)%evgen_done) then
          call this%channels(i)%update_result_iter()
          call this%channels(i)%combine_iters()
       endif
    enddo
    do i=1,2
       this%res(i)=sum(this%channels(1:this%nchannel)%res(i))
       this%unc(i)=sqrt(sum(this%channels(1:this%nchannel)%unc(i)**2))
    enddo
  end subroutine compute_total_rate

  ! Print per-channel and total integration progress to stdout and unit 99.
  subroutine print_results(this)
    implicit none
    class(integrator),intent(inout) :: this
    integer :: i
    real(kind=8) :: rel_unc
    do i=1,this%nchannel
       if (.not. this%channels(i)%evgen_done) call this%channels(i)%print_result_iter()
       call this%channels(i)%print_combined_result()
    enddo
    if (this%res(1).gt.0d0) then
       rel_unc=this%unc(1)/this%res(1)*100d0
    else
       rel_unc=0d0
    endif
    write(*,'(4x,a,1x,e12.6,1x,a,1x,e10.4,1x,a,f8.4,1x,a)') &
         'Integral ABS (accum):',this%res(1),'+/-',this%unc(1),'(',rel_unc,'%)'
    write(*,'(4x,a,1x,e12.6,1x,a,1x,e10.4,1x,a,f8.4,1x,a)') &
         'Integral     (accum):',this%res(2),'+/-',this%unc(2),'(',rel_unc,'%)'
    write(99,'(4x,a,1x,e12.6,1x,a,1x,e10.4,1x,a,f8.4,1x,a)') &
         'Integral ABS (accum):',this%res(1),'+/-',this%unc(1),'(',rel_unc,'%)'
    write(99,'(4x,a,1x,e12.6,1x,a,1x,e10.4,1x,a,f8.4,1x,a)') &
         'Integral     (accum):',this%res(2),'+/-',this%unc(2),'(',rel_unc,'%)'
    call flush()
  end subroutine print_results

  ! Choose how many non-zero points to request in the next iteration.
  subroutine update_points_requested(this)
    implicit none
    class(integrator),intent(inout) :: this
    real(kind=8) :: total,total_channel
    integer :: i,j
    integer(kind=8) :: npoints,npoints_channel
    this%npoints_requested=this%npoints_requested*2
    npoints=0_8
    total=sum(this%channels%res(1),mask=.not.this%channels%evgen_done)
    if (total.le.0d0) then
       do i=1,this%nchannel
          if (this%channels(i)%evgen_done) cycle
          do j=1,this%channels(i)%nintegral
             if (this%channels(i)%integrals(j)%evgen_done) cycle
             this%channels(i)%integrals(j)%npoints_requested=min_points_per_integral
             npoints=npoints+this%channels(i)%integrals(j)%npoints_requested
          enddo
       enddo
       this%npoints_requested=npoints
       return
    endif
    do i=1,this%nchannel
       if (this%channels(i)%evgen_done) cycle
       npoints_channel=max(int(this%channels(i)%res(1)/total*dble(this%npoints_requested),kind=8),&
            min_points_per_channel)
       total_channel=sum(this%channels(i)%integrals%res(1),mask=.not.this%channels(i)%integrals%evgen_done)
       do j=1,this%channels(i)%nintegral
          if (this%channels(i)%integrals(j)%evgen_done) cycle
          if (total_channel.gt.0d0) then
             this%channels(i)%integrals(j)%npoints_requested=&
                  max(int(this%channels(i)%integrals(j)%res(1)/total_channel*dble(npoints_channel),kind=8),&
                  min_points_per_integral)
          else
             this%channels(i)%integrals(j)%npoints_requested=min_points_per_integral
          endif
          npoints=npoints+this%channels(i)%integrals(j)%npoints_requested
       enddo
    enddo
    this%npoints_requested=npoints
  end subroutine update_points_requested

  ! Build the next iteration's adaptive grids for one channel.
  subroutine channel_update_grids(this)
    implicit none
    class(channel),intent(inout) :: this
    type(grid) :: new_grid
    integer :: i
    logical update_grids
    if (this%res(1).gt.0d0) then
       update_grids=(((.not.this%evgen_done) .and. &
            this%unc(1)/this%res(1).gt.1d0/(sqrt(dble(max(this%nevts_unw_req,1)))*required_accuracy_factor)) .or. &
            this%current_iter.le.iters_without_evnts) .and. &
            this%npoints_iter.gt.int(this%npoints*0.2d0)
    else
       update_grids=(this%current_iter.le.iters_without_evnts)
    endif
    if (.not.update_grids) then
       write (99,*) 'keeping grids fixed for channel',this%number
    endif
    do i=1,this%ndim
       if (update_grids) then
          call this%grids(i,this%current_iter)%update(this%npoints_iter,new_grid)
          if (this%current_iter .lt. this%max_iters) then
             this%grids(i,this%current_iter+1)=new_grid
          endif
       else
          this%grids(i,this%current_iter+1)=this%grids(i,this%current_iter)
       endif
    enddo
  end subroutine channel_update_grids

  ! Recompute f_max values, unweight stored candidates, and decide whether one
  ! channel has produced enough acceptable events.
  subroutine channel_check_gen_evnts(this)
    implicit none
    class(channel),intent(inout) :: this
    integer :: i,j
    logical :: done
    do i=1,this%nintegral
       call this%integrals(i)%compute_fmax(this)
       if (this%integrals(i)%nevts_unw_req.eq.0) then
          this%integrals(i)%evgen_done=.true.
          this%integrals(i)%nevts_unw_gen=0
          this%integrals(i)%overweight=0d0
          do j=1,this%integrals(i)%nevnt_in_list
             this%integrals(i)%evnt_list(j)%unwgt=.false.
          enddo
          cycle
       endif
       call this%integrals(i)%unwgt()
       if (this%integrals(i)%nevts_unw_gen.gt.this%integrals(i)%nevts_unw_req) then
          call this%integrals(i)%check_overweight(done)
          if (done) then
             this%integrals(i)%evgen_done=.true.
          else
             this%integrals(i)%evgen_done=.false.
             this%integrals(i)%nevts_unw_gen=min(int(this%integrals(i)%nevts_unw_req*0.8d0),&
                  int(this%integrals(i)%nevts_unw_req*allowed_overweight_factor/this%integrals(i)%overweight))
          endif
       else
          this%integrals(i)%evgen_done=.false.
       endif
    enddo
    this%overweight=sum(this%integrals%overweight)
  end subroutine channel_check_gen_evnts


  ! Accept/reject the best candidate events and measure the overweight excess.
  subroutine check_overweight(this,done)
    use topk_heap_mod
    implicit none
    class(integral),intent(inout) :: this
    logical,intent(out) :: done
    integer :: j,k
    real(kind=8),dimension(this%current_iter) :: fmax
    real(kind=8),dimension(this%nevnt_in_list) :: fabs
    real(kind=8),dimension(this%nevts_unw_req) :: fabs_top
    integer,dimension(this%nevts_unw_req) :: top_idx
    real(kind=8) :: fmax_req,tmp,tail_mass,total_mass
    logical,dimension(this%current_iter) :: to_include
    ! check which iterations to include (only the final
    ! 'final_n_iters_for_evnt_gen' that generated events for this
    ! integral will be included)
    to_include=.false.
    k=0
    do j=this%nevnt_in_list,1,-1
       if (.not.to_include(this%evnt_list(j)%iter)) then
          k=k+1
          to_include(this%evnt_list(j)%iter)=.true.
       endif
       if (k.eq.final_n_iters_for_evnt_gen) exit
    enddo
    ! rescale all f_abs such that they are equivalent for all iterations
    fmax=0d0
    do j=1,this%nevnt_in_list
       this%evnt_list(j)%unwgt=.false.
       do k=iters_without_evnts+1,this%current_iter
          fmax(k)=max(fmax(k),this%evnt_list(j)%f_abs(k))
       enddo
    enddo
    ! rescale
    do j=1,this%nevnt_in_list
       if (to_include(this%evnt_list(j)%iter)) then
          fabs(j)=(this%evnt_list(j)%f_abs(this%evnt_list(j)%iter)/this%evnt_list(j)%rnd)/fmax(this%evnt_list(j)%iter)
       else
          fabs(j)=0d0
       endif
    enddo
    ! Take the nevts_unw_req largest
    k=this%nevts_unw_req
    call topk_largest(fabs,k,fabs_top,top_idx)
    ! find the fmax such that all remain
    fmax_req=fabs_top(k)
    ! check the overweight fraction
    this%overweight=0d0
    tail_mass=0d0
    total_mass=0d0
    do j=1,this%nevts_unw_req
       this%evnt_list(top_idx(j))%unwgt=.true.
       tmp=this%evnt_list(top_idx(j))%f_abs(this%evnt_list(top_idx(j))%iter)/fmax(this%evnt_list(top_idx(j))%iter)
       this%evnt_list(top_idx(j))%overwgt=tmp/fmax_req
       total_mass=total_mass+max(1d0,tmp/fmax_req)
       if (tmp.gt.fmax_req) tail_mass=tail_mass+tmp/fmax_req
    enddo
    if (total_mass.gt.0d0) this%overweight=tail_mass/total_mass
    if (this%overweight.lt.allowed_overweight_factor) then
       done=.true.
    else
       done=.false.
    endif
  end subroutine check_overweight

  ! Count events that pass the current iteration-by-iteration f_max thresholds.
  subroutine integral_unwgt(this)
    implicit none
    class(integral),intent(inout) :: this
    integer :: j,iter
    this%nevts_unw_gen=0
    do j=1,this%nevnt_in_list
       iter=this%evnt_list(j)%iter
       if (this%evnt_list(j)%f_abs(iter).gt.this%f_max(iter)*this%evnt_list(j)%rnd) &
            this%nevts_unw_gen=this%nevts_unw_gen+1
    enddo
  end subroutine integral_unwgt

  ! Recompute candidate-event envelopes for all active iterations and set f_max.
  subroutine integral_compute_fmax(this,thischan)
    use topk_heap_mod
    implicit none
    class(integral),intent(inout) :: this
    class(channel),intent(inout) :: thischan
    real(kind=8),dimension(this%ndim) :: x
    real(kind=8) :: wgt,wgt_new
    integer :: j,k,nevnt,iter
    integer,allocatable,dimension(:) :: index_fmax_top
    real(kind=8),allocatable,dimension(:) :: fmax_top
    real(kind=8),dimension(this%nevnt_in_list,this%current_iter) :: fabs
    nevnt=this%nevnt_in_list
    if (nevnt.eq.0) return
    do j=1,nevnt
       iter=this%evnt_list(j)%iter
       x=this%evnt_list(j)%x
       wgt=this%evnt_list(j)%wgt
       do k=iters_without_evnts+1,this%current_iter
          if ( (k.eq.iter .and. iter.ne.this%current_iter) .or. &
               (k.ne.iter .and. iter.eq.this%current_iter) ) then
             call thischan%recompute_wgt_from_x(k,x,wgt_new)
             this%evnt_list(j)%f_abs(k)=this%evnt_list(j)%f_abs(iter)*wgt_new/wgt
          endif
          fabs(j,k)=this%evnt_list(j)%f_abs(k)
       enddo
    enddo
    nevnt=max(int(write_evnt_fraction*nevnt),1)
    allocate(fmax_top(nevnt))
    allocate(index_fmax_top(nevnt))
    do k=iters_without_evnts+1,this%current_iter
       call topk_largest(fabs(:,k),nevnt,fmax_top,index_fmax_top)
       this%f_max(k)=fmax_top(nevnt)
    enddo
    deallocate(fmax_top)
    deallocate(index_fmax_top)
  end subroutine integral_compute_fmax

  ! Predict the next iteration's f_max from stored candidate events.
  subroutine integral_compute_fmax_next_iter(this,thischan)
    use topk_heap_mod
    implicit none
    class(integral),intent(inout) :: this
    class(channel),intent(inout) :: thischan
    real(kind=8),dimension(this%ndim) :: x
    real(kind=8) :: wgt,wgt_new
    integer :: j,nevnt,iter,next_iter
    integer,allocatable,dimension(:) :: index_fmax_top
    real(kind=8),allocatable,dimension(:) :: fmax_top,fabs
    next_iter=this%current_iter
    nevnt=this%nevnt_in_list
    if (nevnt.le.200) then
       this%f_max(next_iter)=this%f_max(next_iter-1)
       return
    endif
    allocate(fabs(nevnt))
    do j=1,nevnt
       iter=this%evnt_list(j)%iter
       x=this%evnt_list(j)%x
       wgt=this%evnt_list(j)%wgt
       call thischan%recompute_wgt_from_x(next_iter,x,wgt_new)
       this%evnt_list(j)%f_abs(next_iter)=this%evnt_list(j)%f_abs(iter)*wgt_new/wgt
       fabs(j)=this%evnt_list(j)%f_abs(next_iter)
    enddo
    nevnt=max(int(write_evnt_fraction*nevnt),1)
    nevnt=max(int(nevnt*dble(thischan%max_iters-this%current_iter)/dble(thischan%max_iters)),1)
    allocate(fmax_top(nevnt))
    allocate(index_fmax_top(nevnt))
    call topk_largest(fabs,nevnt,fmax_top,index_fmax_top)
    this%f_max(next_iter)=fmax_top(nevnt)
    deallocate(fabs)
    deallocate(fmax_top)
    deallocate(index_fmax_top)
  end subroutine integral_compute_fmax_next_iter

  ! Re-evaluate the adaptive-grid weight for an existing point in a given
  ! channel iteration.
  subroutine recompute_wgt_from_x(this,iter,x,wgt)
    implicit none
    class(channel),intent(inout) :: this
    integer,intent(in) :: iter
    real(kind=8),dimension(this%ndim),intent(in) :: x
    real(kind=8),intent(out) :: wgt
    integer :: i
    wgt=1d0
    do i=1,this%ndim
       call this%grids(i,iter)%get_wgt(x(i),wgt)
    enddo
  end subroutine recompute_wgt_from_x

  ! Multiply wgt by the Jacobian contribution for x in this grid.
  subroutine get_wgt(this,x,wgt)
    implicit none
    class(grid),intent(inout) :: this
    real(kind=8),intent(in) :: x
    real(kind=8),intent(inout) :: wgt
    real(kind=8) :: dx
    integer :: cell
    call this%find_cell(x,cell)
    dx=this%current(cell)-this%current(cell-1)
    wgt=wgt*dx*this%size
  end subroutine get_wgt

  ! Locate the interpolation cell containing x in the full grid.
  subroutine find_cell(this,x,cell)
    implicit none
    class(grid),intent(inout) :: this
    real(kind=8),intent(in) :: x
    integer,intent(out) :: cell
    integer :: lo,hi,mid
    lo=0
    hi=this%size
    do
       mid=(lo+hi)/2
       if (x.lt.this%current(mid)) then
          hi=mid
       else
          lo=mid+1
       end if
       if (lo.ge.hi) exit
    enddo
    cell=min(max(lo,1),this%size)
  end subroutine find_cell

  ! Locate the adaptation fill cell containing x.
  subroutine find_cell_to_fill(this,x,cell)
    implicit none
    class(grid),intent(inout) :: this
    real(kind=8),intent(in) :: x
    integer,intent(out) :: cell
    integer :: lo,hi,mid
    lo=0
    hi=this%size_fill
    do
       mid=(lo+hi)/2
       if (x.lt.this%current_for_fillcell(mid)) then
          hi=mid
       else
          lo=mid+1
       end if
       if (lo.ge.hi) exit
    enddo
    cell=min(max(lo,1),this%size_fill)
  end subroutine find_cell_to_fill

  ! Print accumulated channel and integral results to the log unit.
  subroutine channel_print_combined_result(this)
    implicit none
    class(channel),intent(inout) :: this
    integer :: i
    real(kind=8) :: rel_unc
    if (this%res(1).gt.0d0) then
       rel_unc=this%unc(1)/this%res(1)*100d0
    else
       rel_unc=0d0
    endif
    write(99,'(4x,i4,1x,a,1x,e10.4,1x,a,1x,e10.4,1x,a,f7.3,1x,a)') &
         this%number,'channel ABS (accum):',this%res(1),'+/-',this%unc(1),'(',rel_unc,'%)'
    write(99,'(4x,i4,1x,a,1x,e10.4,1x,a,1x,e10.4,1x,a,f7.3,1x,a)') &
         this%number,'channel     (accum):',this%res(2),'+/-',this%unc(2),'(',rel_unc,'%)'
    do i=1,this%nintegral
       this%integrals(i)%npoints_nonzero_total=this%integrals(i)%npoints_nonzero_total+this%integrals(i)%npoints_nonzero
       if (this%integrals(i)%nevts_unw_gen.ge.this%integrals(i)%nevts_unw_req) then
          write(99,'(23x,i4,1x,a,1x,e10.4,1x,a,1x,e10.4,1x,a,1x,i10,1x,a,1x,i10,1x,a,1x,f8.6,1x,a,1x,i10,1x,a)') &
               i,':',this%integrals(i)%res(2),'+/-',this%integrals(i)%unc(2),&
               '--',this%integrals(i)%npoints_nonzero_total,'--',this%integrals(i)%nevnt_in_list,&
               '--',this%integrals(i)%overweight,'--',this%integrals(i)%nevts_unw_req,'-- DONE'
       else
          if (this%integrals(i)%nevnt_in_list.lt.this%integrals(i)%nevts_unw_req) then
             write(99,'(23x,i4,1x,a,1x,e10.4,1x,a,1x,e10.4,1x,a,1x,i10,1x,a,1x,i10,1x,a,1x,a,1x,a,1x,i10)') &
                  i,':',this%integrals(i)%res(2),'+/-',this%integrals(i)%unc(2),&
                  '--',this%integrals(i)%npoints_nonzero_total,'--',this%integrals(i)%nevnt_in_list,&
                  '--','    N/A ','--',this%integrals(i)%nevts_unw_req

          else
             write(99,'(23x,i4,1x,a,1x,e10.4,1x,a,1x,e10.4,1x,a,1x,i10,1x,a,1x,i10,1x,a,1x,f8.6,1x,a,1x,i10)') &
                  i,':',this%integrals(i)%res(2),'+/-',this%integrals(i)%unc(2),&
                  '--',this%integrals(i)%npoints_nonzero_total,'--',this%integrals(i)%nevnt_in_list,&
                  '--',this%integrals(i)%overweight,'--',this%integrals(i)%nevts_unw_req
          endif
       endif
    enddo
  end subroutine channel_print_combined_result

  ! Print the current iteration-only result for one active channel.
  subroutine channel_print_result_iter(this)
    implicit none
    class(channel),intent(inout) :: this
    real(kind=8) :: rel_unc
    if (this%res_iter(1).gt.0d0) then
       rel_unc=this%unc_iter(1)/this%res_iter(1)*100d0
    else
       rel_unc=0d0
    endif
    write(99,'(4x,i4,1x,a,1x,e10.4,1x,a,1x,e10.4,1x,a,f7.3,1x,a)') &
         this%number,'channel ABS:',this%res_iter(1),'+/-',this%unc_iter(1),'(',rel_unc,'%)'
    write(99,'(4x,i4,1x,a,1x,e10.4,1x,a,1x,e10.4,1x,a,f7.3,1x,a)') &
         this%number,'channel    :',this%res_iter(2),'+/-',this%unc_iter(2),'(',rel_unc,'%)'
  end subroutine channel_print_result_iter

  ! Combine all integral estimates inside one channel.
  subroutine channel_combine_iters(this)
    implicit none
    class(channel),intent(inout) :: this
    integer :: i
    do i=1,this%nintegral
       if (.not. this%integrals(i)%evgen_done) &
            call this%integrals(i)%combine_iters(this%current_iter)
    enddo
    do i=1,2
       this%res(i)=sum(this%integrals(1:this%nintegral)%res(i))
       this%unc(i)=sqrt(sum(this%integrals(1:this%nintegral)%unc(i)**2))
    enddo
    if (this%current_iter.eq.1) then
       this%npoints=this%npoints_iter
    else
       this%npoints=this%npoints+this%npoints_iter
    endif
  end subroutine channel_combine_iters

  ! Combine a new iteration estimate into one integral's accumulated estimate.
  subroutine integral_combine_iters(this,iter)
    implicit none
    class(integral),intent(inout) :: this
    integer,intent(in) :: iter
    integer :: i
    if (iter.eq.1) then
       this%res=this%res_iter
       this%unc=this%unc_iter
       this%npoints=this%npoints_iter
    else
       do i=1,2
          call update_res_and_unc(this%res(i),this%unc(i),this%npoints,this%res_iter(i),this%unc_iter(i),this%npoints_iter)
       enddo
       this%npoints=this%npoints+this%npoints_iter
    endif
  end subroutine integral_combine_iters

  ! Combine two independent sample means and their standard errors.
  subroutine update_res_and_unc(res,unc,npoints,res_iter,unc_iter,npoints_iter)
    implicit none
    real(kind=8),intent(inout) :: res,unc
    real(kind=8),intent(in) :: res_iter,unc_iter
    integer(kind=8),intent(inout) :: npoints,npoints_iter
    integer(kind=8) :: np
    np=npoints+npoints_iter
    unc=sqrt((unc**2*dble(npoints)**2+unc_iter**2*dble(npoints_iter)**2)&
         &/dble(np)**2+npoints*(res-res_iter)**2*dble(npoints_iter)&
         &**2/(dble(npoints_iter)*dble(np)**3))
    res=(npoints*res+npoints_iter*res_iter)/dble(np)
  end subroutine update_res_and_unc

  ! Compute the current iteration estimate for every integral in a channel.
  subroutine channel_update_result_iter(this)
    implicit none
    class(channel),intent(inout) :: this
    integer :: i
    do i=1,this%nintegral
       call this%integrals(i)%update_result_iter()
    enddo
    do i=1,2
       this%res_iter(i)=sum(this%integrals(1:this%nintegral)%res_iter(i))
       this%unc_iter(i)=sqrt(sum(this%integrals(1:this%nintegral)%unc_iter(i)**2))
    enddo
  end subroutine channel_update_result_iter

  ! Compute the standard error of the mean from first and second moments.
  subroutine compute_uncertainty(acc,acc2,np,unc)
    implicit none
    real(kind=8),intent(in) :: acc,acc2
    integer(kind=8),intent(in) :: np
    real(kind=8),intent(out) :: unc
    unc=sqrt(abs(acc2-acc**2)/dble(np))
  end subroutine compute_uncertainty

  ! Convert one integral's accumulated sums into an iteration estimate.
  subroutine integral_update_result_iter(this)
    implicit none
    class(integral),intent(inout) :: this
    integer :: i
    if (this%npoints_iter.ne.0_8) then
       this%res_iter=this%accum/dble(this%npoints_iter)
       this%res2_iter=this%accum2/dble(this%npoints_iter)
       do i=1,2
          call compute_uncertainty(this%res_iter(i),this%res2_iter(i),this%npoints_iter,this%unc_iter(i))
       enddo
    endif
  end subroutine integral_update_result_iter

  ! Add one evaluated point to a channel grid and to its active integral.
  subroutine channel_add_point(this,x,wgt,f_abs,f,to_write)
    implicit none
    class(channel),intent(inout) :: this
    real(kind=8),dimension(this%ndim),intent(in) :: x
    real(kind=8),intent(in) :: f_abs,f,wgt
    logical,intent(out) :: to_write
    integer :: i
    this%npoints_iter=this%npoints_iter+1
    do i=1,this%ndim
       call this%grids(i,this%current_iter)%add_point(x(i),f_abs)
    enddo
    call this%integrals(this%current_integral)%add_point(x,wgt,f_abs,f,to_write)
    if (all(this%integrals%done)) this%done=.true.
  end subroutine channel_add_point

  ! Accumulate one point in an integral and optionally save it as a candidate
  ! event for later final unweighting.
  subroutine integral_add_point(this,x,wgt,f_abs,f,to_write)
    implicit none
    class(integral),intent(inout) :: this
    real(kind=8),intent(in) :: f_abs,f,wgt
    real(kind=8),dimension(this%ndim),intent(in) :: x
    logical,intent(out) :: to_write
    logical :: enough
    this%npoints_iter=this%npoints_iter+1
    if (f_abs.gt.0d0) this%npoints_nonzero=this%npoints_nonzero+1
    this%accum(1)=this%accum(1)+f_abs
    this%accum(2)=this%accum(2)+f
    this%accum2(1)=this%accum2(1)+f_abs**2
    this%accum2(2)=this%accum2(2)+f**2
    if (this%current_iter.le.iters_without_evnts) call this%update_max_value(f_abs)
    call this%check_write_evnt(x,wgt,f_abs,to_write,enough)
    if (this%npoints_nonzero.ge.this%npoints_requested .or. enough) this%done=.true.
  end subroutine integral_add_point

  ! Track the largest absolute integrand value seen before event generation.
  subroutine update_max_value(this,f_abs)
    implicit none
    class(integral),intent(inout) :: this
    real(kind=8),intent(in) :: f_abs
    this%max_value=max(this%max_value,f_abs)
  end subroutine update_max_value

  ! Decide whether a point should be written as a candidate event.
  subroutine check_write_evnt(this,x,wgt,f_abs,to_write,enough)
    implicit none
    class(integral),intent(inout) :: this
    real(kind=8),intent(in) :: f_abs,wgt
    real(kind=8),dimension(this%ndim),intent(in) :: x
    logical,intent(out) :: to_write,enough
    real(kind=8) :: rnd
    to_write=.false.
    enough=.false.
    if (turn_off_evnt_generation) return
    if (this%current_iter.le.iters_without_evnts) return
    rnd=ran2()
    if (f_abs.gt.this%f_max(this%current_iter)*rnd) then
       to_write=.true.
       evnt_label=evnt_label+1
       this%evnt=this%evnt+1
       this%nevnt_in_list=this%nevnt_in_list+1
       if (this%nevnt_in_list.gt.size(this%evnt_list)) call this%increase_size_evnt_list()
       allocate(this%evnt_list(this%nevnt_in_list)%f_abs(this%max_iters))
       allocate(this%evnt_list(this%nevnt_in_list)%x(this%ndim))
       this%evnt_list(this%nevnt_in_list)%f_abs=0d0
       this%evnt_list(this%nevnt_in_list)%f_abs(this%current_iter)=f_abs
       this%evnt_list(this%nevnt_in_list)%rnd=rnd
       this%evnt_list(this%nevnt_in_list)%wgt=wgt
       this%evnt_list(this%nevnt_in_list)%x=x
       this%evnt_list(this%nevnt_in_list)%iter=this%current_iter
       this%evnt_list(this%nevnt_in_list)%label=evnt_label
       this%nevts_unw_gen=this%nevts_unw_gen+1
       if (this%nevts_unw_gen.gt.1.5d0*this%nevts_unw_req) enough=.true.
    endif
  end subroutine check_write_evnt

  ! Return final per-event weights for all candidate events written by the
  ! caller during the get_points/fill_points loop.
  subroutine assign_evnt_wgts(this,wgts)
    implicit none
    class(integrator) :: this
    real(kind=8),allocatable,dimension(:,:),intent(out) :: wgts
    integer :: i,j
    real(kind=8) :: nominal_wgt
    allocate(wgts(3,evnt_label))
    nominal_wgt=this%res(1)
    do i=1,this%nchannel
       do j=1,this%channels(i)%nintegral
          call this%channels(i)%integrals(j)%compute_wgts(nominal_wgt,wgts)
       enddo
    enddo
  end subroutine assign_evnt_wgts

  ! Fill the final weight columns for candidate events belonging to one integral.
  subroutine compute_wgts(this,nominal_wgt,wgts)
    implicit none
    class(integral) :: this
    real(kind=8),dimension(3,evnt_label),intent(inout) :: wgts
    real(kind=8),intent(in) :: nominal_wgt
    real(kind=8) :: number_of_evnts,number_of_wgts
    integer :: i
    number_of_evnts=0d0
    number_of_wgts=0d0
    do i=1,this%nevnt_in_list
       if (this%evnt_list(i)%unwgt) then
          number_of_evnts=number_of_evnts+1d0
          number_of_wgts=number_of_wgts+max(1d0,this%evnt_list(i)%overwgt)
       endif
    enddo
    do i=1,this%nevnt_in_list
       if (this%evnt_list(i)%unwgt) then
          wgts(1,this%evnt_list(i)%label)=nominal_wgt
          wgts(2,this%evnt_list(i)%label)=nominal_wgt*max(1d0,this%evnt_list(i)%overwgt) &
               *number_of_evnts/number_of_wgts
          wgts(3,this%evnt_list(i)%label)=max(0d0,this%evnt_list(i)%overwgt-1d0)
       else
          wgts(1:3,this%evnt_list(i)%label)=0d0
       endif
    enddo
  end subroutine compute_wgts

  ! Double the candidate-event buffer while preserving existing events.
  subroutine increase_size_evnt_list(this)
    implicit none
    class(integral),intent(inout) :: this
    type(evnt),allocatable,dimension(:) :: tmp_list
    integer :: isize
    isize=size(this%evnt_list)
    allocate(tmp_list(2*isize))
    tmp_list(1:isize)=this%evnt_list(1:isize)
    deallocate(this%evnt_list)
    this%evnt_list=tmp_list
  end subroutine increase_size_evnt_list

  ! Accumulate one sampled point into the grid-adaptation histogram.
  subroutine grid_add_point(this,x,f_abs)
    class(grid),intent(inout) :: this
    real(kind=8),intent(in) :: x,f_abs
    integer :: cell
    call this%find_cell_to_fill(x,cell)
    if (importance_sampling_strategy.eq.1) then
       this%accum(cell)=this%accum(cell)+f_abs
    elseif (importance_sampling_strategy.eq.2) then
       this%accum(cell)=max(this%accum(cell),f_abs)
    elseif (importance_sampling_strategy.eq.3) then
       if (f_abs.gt.this%accum(cell)) then
          this%accum(cell)=this%accum(cell)+(f_abs-this%accum(cell))*0.1d0
       endif
    elseif (importance_sampling_strategy.eq.4) then
       if (this%accum(cell).le.0d0) then
          this%accum(cell)=f_abs*1d-4
       elseif (f_abs.gt.this%accum(cell)) then
          this%accum(cell)=this%accum(cell)*1.1d0
       endif
    endif
    this%nhits(cell)=this%nhits(cell)+1
  end subroutine grid_add_point

  ! Convert accumulated grid information into a new monotone grid.
  subroutine grid_update(this,npoints,new_grid)
    implicit none
    class(grid),intent(inout) :: this
    integer(kind=8),intent(in) :: npoints
    class(grid),intent(out) :: new_grid
    real(kind=8),dimension(0:this%size_fill) :: current
    integer :: i,j
    real(kind=8) :: r
    call this%massage_accum()
    current(0)=0d0
    do i=1,this%size_fill
       r=dble(i)/dble(this%size_fill)
       do j=1,this%size_fill
          if (r.lt.this%accum(j)) then
             current(i)=this%current_for_fillcell(j-1)+(r-this%accum(j-1))/ &
                  (this%accum(j)-this%accum(j-1))*(this%current_for_fillcell(j)-this%current_for_fillcell(j-1))
             exit
          endif
       enddo
    enddo
    deallocate(this%accum)
    deallocate(this%nhits)
    current(this%size_fill)=1d0
    call new_grid%init(npoints,current)
  end subroutine grid_update

  ! Resize a monotone grid with shape-preserving interpolation.
  subroutine interpolate_current(this,size_in,size_out,current_in,current_out)
    use pchip_uniform_strict
    implicit none
    class(grid),intent(inout) :: this
    integer,intent(in) :: size_in,size_out
    real(kind=8),dimension(0:size_in),intent(in) :: current_in
    real(kind=8),dimension(0:size_out),intent(out) :: current_out
    call resize_arr_pchip_strict(current_in,size_out,current_out)
  end subroutine interpolate_current

  ! Smooth and normalise the adaptation histogram into a cumulative map.
  subroutine massage_accum(this)
    implicit none
    class(grid),intent(inout) :: this
    integer :: i
    real(kind=8) :: total
    real(kind=8), parameter :: tiny=1d-8
    do i=1,this%size_fill
       if (this%nhits(i).eq.0) cycle
       if (importance_sampling_strategy.eq.1) then
          this%accum(i)=this%accum(i)/this%nhits(i)
       else
          this%accum(i)=this%accum(i)
       endif
    enddo
    total=sum(this%accum)
    do i=1,this%size_fill
       if (this%accum(i).lt.1d-12*total) then
          this%accum(i)=0d0
       elseif (this%accum(i).lt.(1d0-1d-12)*total) then
          this%accum(i)=((this%accum(i)/total-1d0)/log(this%accum(i)/total))**1.5
       else
          this%accum(i)=1d0
       endif
       this%accum(i)=this%accum(i-1)+max(this%accum(i),0d0)
    enddo
    this%accum=this%accum/this%accum(this%size_fill)
    ! make sure the elements are at least 'tiny' apart
    do i=1,this%size_fill
       if (this%accum(i).lt.this%accum(i-1)+tiny) then
          this%accum(i)=this%accum(i-1)+tiny
       endif
    enddo
    this%accum=this%accum/this%accum(this%size_fill)
  end subroutine massage_accum

  ! Generate one full point for a channel, including flat extra coordinates.
  subroutine channel_get_point(this,x,wgt)
    implicit none
    class(channel),intent(inout) :: this
    real(kind=8),dimension(this%ndim+this%ndim_extra),intent(out) :: x
    real(kind=8),intent(out) :: wgt
    integer :: i
    wgt=1d0
    do i=1,this%ndim
       call this%grids(i,this%current_iter)%get_x(x(i),wgt)
    enddo
    do i=this%ndim+1,this%ndim+this%ndim_extra
       x(i)=ran2()
    enddo
  end subroutine channel_get_point

  ! Generate one adapted coordinate and multiply by its Jacobian.
  subroutine get_x(this,x,wgt)
    implicit none
    class(grid),intent(inout) :: this
    real(kind=8),intent(out) :: x
    integer :: cell
    real(kind=8),intent(inout) :: wgt
    real(kind=8) :: rnd,dx
    rnd=this%size*ran2()
    cell=int(rnd)+1
    rnd=rnd-dble(cell-1)
    dx=this%current(cell)-this%current(cell-1)
    x=this%current(cell-1)+rnd*dx
    wgt=wgt*dx*this%size
  end subroutine get_x

  ! Randomly select an unfinished channel/integral pair for the next batch.
  subroutine get_channel_and_integral(this,ichan,iint,wgt_chan)
    implicit none
    class(integrator),intent(inout) :: this
    integer,intent(out) :: ichan,iint
    real(kind=8),intent(out) :: wgt_chan
    logical :: done
    do
       ichan=int(ran2()*this%nchannel)+1
       if (.not.(this%channels(ichan)%done.or.this%channels(ichan)%evgen_done)) exit
    enddo
    wgt_chan=1d0!dble(this%nchannel)
    done=.false.
    do
       iint=int(ran2()*this%channels(ichan)%nintegral)+1
       if (.not.(this%channels(ichan)%integrals(iint)%done.or.this%channels(ichan)%integrals(iint)%evgen_done)) exit
    enddo
    wgt_chan=wgt_chan*1d0!dble(this%channels(ichan)%nintegral)
    this%channels(ichan)%current_integral=iint
  end subroutine get_channel_and_integral

  ! Public helper: compute the current adaptive-grid weight for an existing
  ! point in channel ichan.
  subroutine compute_wgt_from_x(this,ichan,x,wgt)
    implicit none
    class(integrator),intent(inout) :: this
    integer,intent(in) :: ichan
    real(kind=8),dimension(this%channels(ichan)%ndim),intent(in) :: x
    real(kind=8),intent(out) :: wgt
    call this%channels(ichan)%recompute_wgt_from_x(this%channels(ichan)%current_iter,x,wgt)
  end subroutine compute_wgt_from_x

  ! Placeholder for future grid restart support.
  subroutine read_all_grids(this)
    implicit none
    class(integrator),intent(inout) :: this
  end subroutine read_all_grids

  ! Placeholder for future grid checkpoint support.
  subroutine write_all_grids(this)
    implicit none
    class(integrator),intent(inout) :: this
  end subroutine write_all_grids

  subroutine staged_init(this,ndim,nvalues)
    class(staged_integrator),intent(inout) :: this
    integer,intent(in) :: ndim,nvalues
    integer :: i
    if (ndim.lt.1.or.nvalues.lt.1) error stop 'staged integrator: invalid dimensions'
    if (allocated(this%maps)) deallocate(this%maps,this%mean,this%m2,this%res,this%unc)
    if (allocated(this%adaptation_mask)) deallocate(this%adaptation_mask)
    if (allocated(this%candidate_weight)) deallocate(this%candidate_weight,this%candidate_priority,this%candidate_factor)
    call this%reset_native_production()
    this%ndim=ndim
    this%nvalues=nvalues
    allocate(this%maps(ndim),this%mean(nvalues),this%m2(nvalues),this%res(nvalues),this%unc(nvalues))
    allocate(this%adaptation_mask(ndim))
    call this%reset_production_adaptation()
    do i=1,ndim
       call this%maps(i)%init(int(min_points_per_channel,kind=8))
    enddo
    this%mean=0d0
    this%m2=0d0
    this%res=0d0
    this%unc=0d0
    this%npoints=0_8
    this%total_points=0_8
    this%max_weight=0d0
    this%iteration_active=.false.
    this%adapt_iteration=.false.
    this%production_active=.false.
    this%pool_mode=.false.
    this%production_done=.false.
    this%quota_complete=.false.
    this%exhausted=.false.
    this%envelope_exceeded=.false.
    this%ntrials=0_8
    this%max_trials=0_8
    this%next_pool_check=0_8
    this%ncandidates=0
    this%quota=0
    this%final_quota=0
    this%total_abs=0d0
    this%threshold=0d0
    this%full_trial_tail=0d0
    this%reserve_tail=0d0
    this%worst_subset_tail=0d0
    this%envelope=0d0
    this%overweight=0d0
    this%overweight_tolerance=allowed_overweight_factor
  end subroutine staged_init

  subroutine staged_begin_iteration(this,adapt)
    class(staged_integrator),intent(inout) :: this
    logical,intent(in) :: adapt
    integer :: i
    if (.not.allocated(this%maps)) error stop 'staged integrator: begin before init'
    if (this%iteration_active) error stop 'staged integrator: iteration already active'
    if (this%production_active) error stop 'staged integrator: iteration during production'
    call this%reset_production_adaptation()
    this%npoints=0_8
    this%mean=0d0
    this%m2=0d0
    this%max_weight=0d0
    this%adapt_iteration=adapt
    this%iteration_active=.true.
    do i=1,this%ndim
       this%maps(i)%accum=0d0
       this%maps(i)%nhits=0
    enddo
  end subroutine staged_begin_iteration

  subroutine staged_sample(this,x,wgt,base_random)
    use,intrinsic :: ieee_arithmetic,only: ieee_is_finite
    class(staged_integrator),intent(in) :: this
    real(kind=8),intent(out) :: x(:),wgt
    real(kind=8),intent(out),optional :: base_random(:)
    real(kind=8) :: u,r,dx
    integer :: i,cell
    if (.not.allocated(this%maps)) error stop 'staged integrator: sample before init'
    if (size(x).ne.this%ndim) error stop 'staged integrator: sample dimension mismatch'
    if (present(base_random)) then
       if (size(base_random).ne.this%ndim) error stop 'staged integrator: random dimension mismatch'
    endif
    wgt=1d0
    do i=1,this%ndim
       u=ran2()
       if (.not.ieee_is_finite(u).or.u.lt.0d0.or.u.ge.1d0) error stop 'staged integrator: invalid RNG value'
       if (present(base_random)) base_random(i)=u
       r=u*this%maps(i)%size
       cell=min(int(r)+1,this%maps(i)%size)
       dx=this%maps(i)%current(cell)-this%maps(i)%current(cell-1)
       x(i)=this%maps(i)%current(cell-1)+(r-dble(cell-1))*dx
       wgt=wgt*dx*this%maps(i)%size
    enddo
  end subroutine staged_sample

  subroutine staged_map_fold(this,base_random,kfold,ifold,x,wgt)
    use,intrinsic :: ieee_arithmetic,only: ieee_is_finite
    class(staged_integrator),intent(in) :: this
    real(kind=8),intent(in) :: base_random(:)
    integer,intent(in) :: kfold(:),ifold(:)
    real(kind=8),intent(out) :: x(:),wgt
    real(kind=8) :: u,r,dx
    integer :: i,cell
    if (.not.allocated(this%maps)) error stop 'staged integrator: folding before init'
    if (size(base_random).ne.this%ndim.or.size(kfold).ne.this%ndim.or.&
         size(ifold).ne.this%ndim.or.size(x).ne.this%ndim) &
         error stop 'staged integrator: fold dimension mismatch'
    if (any(ifold.lt.1).or.any(kfold.lt.1).or.any(kfold.gt.ifold)) &
         error stop 'staged integrator: invalid fold index'
    if (any(.not.ieee_is_finite(base_random)).or.any(base_random.lt.0d0).or.any(base_random.ge.1d0)) &
         error stop 'staged integrator: invalid fold variate'
    wgt=1d0
    do i=1,this%ndim
       u=(base_random(i)+dble(kfold(i)-1))/dble(ifold(i))
       r=u*this%maps(i)%size
       cell=min(int(r)+1,this%maps(i)%size)
       dx=this%maps(i)%current(cell)-this%maps(i)%current(cell-1)
       x(i)=this%maps(i)%current(cell-1)+(r-dble(cell-1))*dx
       wgt=wgt*dx*this%maps(i)%size/dble(ifold(i))
    enddo
  end subroutine staged_map_fold

  subroutine staged_observe(this,x,values,adapt_target,fold_points)
    use,intrinsic :: ieee_arithmetic,only: ieee_is_finite
    class(staged_integrator),intent(inout) :: this
    real(kind=8),intent(in) :: x(:),values(:)
    real(kind=8),intent(in),optional :: adapt_target,fold_points(:,:)
    real(kind=8) :: target,delta(this%nvalues)
    integer :: i,j
    if (.not.this%iteration_active) error stop 'staged integrator: observe outside iteration'
    if (size(x).ne.this%ndim.or.size(values).ne.this%nvalues) &
         error stop 'staged integrator: observation dimension mismatch'
    if (any(.not.ieee_is_finite(x)).or.any(x.lt.0d0).or.any(x.gt.1d0).or.&
         any(.not.ieee_is_finite(values))) error stop 'staged integrator: invalid observation'
    target=abs(values(1))
    if (present(adapt_target)) target=adapt_target
    if (.not.ieee_is_finite(target).or.target.lt.0d0) error stop 'staged integrator: invalid adaptation target'
    this%npoints=this%npoints+1_8
    this%total_points=this%total_points+1_8
    delta=values-this%mean
    this%mean=this%mean+delta/dble(this%npoints)
    this%m2=this%m2+delta*(values-this%mean)
    this%max_weight=max(this%max_weight,target)
    if (this%adapt_iteration) then
       if (present(fold_points)) then
          if (size(fold_points,1).ne.this%ndim.or.size(fold_points,2).lt.1.or. &
               any(.not.ieee_is_finite(fold_points)).or.any(fold_points.lt.0d0).or.any(fold_points.gt.1d0)) &
               error stop 'staged integrator: invalid folded adaptation coordinates'
          ! A folded observation is one statistical sample. Deposit its full
          ! target at every fold image so folded coordinates cover their whole
          ! domain, not just the first image used to identify the observation.
          do i=1,this%ndim
             do j=1,size(fold_points,2)
                call this%maps(i)%add_point(fold_points(i,j),target)
             enddo
          enddo
       else
          do i=1,this%ndim
             call this%maps(i)%add_point(x(i),target)
          enddo
       endif
    endif
  end subroutine staged_observe

  subroutine staged_finish_iteration(this,res,unc,adapt)
    class(staged_integrator),intent(inout) :: this
    real(kind=8),intent(out) :: res(:),unc(:)
    logical,intent(in),optional :: adapt
    logical :: update
    integer :: i
    type(grid) :: next_grid
    if (.not.this%iteration_active) error stop 'staged integrator: finish outside iteration'
    if (size(res).ne.this%nvalues.or.size(unc).ne.this%nvalues) &
         error stop 'staged integrator: result dimension mismatch'
    update=this%adapt_iteration
    if (present(adapt)) then
       if (adapt.and..not.this%adapt_iteration) error stop 'staged integrator: adaptation data not collected'
       update=adapt
    endif
    this%res=this%mean
    this%unc=0d0
    if (this%npoints.gt.1_8) then
       this%unc=sqrt(max(this%m2,0d0)/dble(this%npoints)/dble(this%npoints-1_8))
    elseif (this%npoints.eq.1_8) then
       this%unc=huge(1d0)
    endif
    res=this%res
    unc=this%unc
    ! Empty/all-zero channels retain their maps; the original grid smoother
    ! assumes a positive adaptation histogram and would otherwise divide by 0.
    if (update.and.this%max_weight.gt.0d0) then
       do i=1,this%ndim
          call this%maps(i)%update(this%npoints,next_grid)
          this%maps(i)=next_grid
       enddo
    endif
    this%iteration_active=.false.
  end subroutine staged_finish_iteration

  subroutine staged_start_production(this,quota,envelope,max_trials,overweight_tolerance,final_quota, &
       ifold,adaptation_interval)
    use,intrinsic :: ieee_arithmetic,only: ieee_is_finite
    class(staged_integrator),intent(inout) :: this
    integer,intent(in) :: quota
    real(kind=8),intent(in) :: envelope
    integer(kind=8),intent(in) :: max_trials
    real(kind=8),intent(in),optional :: overweight_tolerance
    integer,intent(in),optional :: final_quota,ifold(:)
    integer(kind=8),intent(in),optional :: adaptation_interval
    integer :: i
    if (.not.allocated(this%maps)) error stop 'staged integrator: production before init'
    call this%reset_native_production()
    if (this%iteration_active) error stop 'staged integrator: production during iteration'
    if (quota.lt.0.or.max_trials.lt.0_8.or.(quota.gt.0.and.max_trials.eq.0_8)) &
         error stop 'staged integrator: invalid production budget'
    if (.not.ieee_is_finite(envelope).or.envelope.lt.0d0) error stop 'staged integrator: invalid envelope'
    this%overweight_tolerance=allowed_overweight_factor
    if (present(overweight_tolerance)) this%overweight_tolerance=overweight_tolerance
    if (.not.ieee_is_finite(this%overweight_tolerance).or.this%overweight_tolerance.lt.0d0) &
         error stop 'staged integrator: invalid overweight tolerance'
    if (quota.gt.0.and.envelope.eq.0d0.and.this%overweight_tolerance.eq.0d0) &
         error stop 'staged integrator: strict production requires a positive envelope'
    if (allocated(this%candidate_weight)) deallocate(this%candidate_weight,this%candidate_priority,this%candidate_factor)
    allocate(this%candidate_weight(16),this%candidate_priority(16),this%candidate_factor(16))
    this%candidate_factor=1d0
    this%quota=quota
    this%final_quota=quota
    if (present(final_quota)) this%final_quota=final_quota
    if (this%final_quota.lt.0.or.this%final_quota.gt.quota) error stop 'staged integrator: invalid final quota'
    this%total_abs=0d0
    this%threshold=envelope
    this%full_trial_tail=0d0
    this%reserve_tail=0d0
    this%worst_subset_tail=0d0
    this%envelope=envelope
    this%max_trials=max_trials
    this%ntrials=0_8
    this%ncandidates=0
    this%next_pool_check=max(int(quota,8)+1_8,(3_8*int(quota,8)+1_8)/2_8)
    this%overweight=0d0
    this%exhausted=.false.
    this%envelope_exceeded=.false.
    this%quota_complete=quota.eq.0
    this%production_done=quota.eq.0
    this%production_active=.true.
    this%pool_mode=.false.
    call this%reset_production_adaptation()
    if (present(adaptation_interval)) then
       if (.not.present(ifold)) error stop 'staged integrator: adaptation interval requires folding factors'
       if (adaptation_interval.lt.1_8.or.adaptation_interval.gt.65536_8) &
            error stop 'staged integrator: invalid production adaptation interval'
    endif
    if (present(ifold)) then
       if (size(ifold).ne.this%ndim) error stop 'staged integrator: adaptation folding dimension mismatch'
       if (any(ifold.lt.1)) error stop 'staged integrator: invalid production folding factor'
       this%adaptation_mask=ifold.eq.1
       if (any(this%adaptation_mask)) then
          this%adaptation_interval=1024_8
          if (present(adaptation_interval)) this%adaptation_interval=adaptation_interval
          this%adaptation_batch=this%adaptation_interval
          do i=1,this%ndim
             if (.not.this%adaptation_mask(i)) cycle
             this%maps(i)%accum=0d0
             this%maps(i)%nhits=0
          enddo
       endif
    endif
  end subroutine staged_start_production

  subroutine staged_reset_production_adaptation(this)
    class(staged_integrator),intent(inout) :: this
    this%adaptation_mask=.false.
    this%adaptation_updates=0
    this%adaptation_interval=0_8
    this%adaptation_batch=0_8
    this%adaptation_points=0_8
  end subroutine staged_reset_production_adaptation

  subroutine staged_train_production(this,weight,x)
    use,intrinsic :: ieee_arithmetic,only: ieee_is_finite
    class(staged_integrator),intent(inout) :: this
    real(kind=8),intent(in) :: weight
    real(kind=8),intent(in),optional :: x(:)
    type(grid) :: next_grid
    integer :: i,n,nfill
    logical :: updated
    if (.not.any(this%adaptation_mask)) return
    if (.not.present(x)) error stop 'staged integrator: production adaptation requires coordinates'
    if (size(x).ne.this%ndim) error stop 'staged integrator: production coordinate dimension mismatch'
    if (any(.not.ieee_is_finite(x))) error stop 'staged integrator: invalid production adaptation coordinates'
    if (any(x.lt.0d0).or.any(x.gt.1d0)) error stop 'staged integrator: invalid production adaptation coordinates'
    ! weight is f/q_draw for this particular proposal epoch. Train all draws,
    ! including rejected and zero points, without changing any stored weight
    ! or random priority from earlier proposal epochs.
    do i=1,this%ndim
       if (this%adaptation_mask(i)) call this%maps(i)%add_point(x(i),weight)
    enddo
    this%adaptation_points=this%adaptation_points+1_8
    if (this%adaptation_points.ne.this%adaptation_batch) return
    updated=.false.
    do i=1,this%ndim
       if (.not.this%adaptation_mask(i)) cycle
       if (maxval(this%maps(i)%accum).gt.0d0) then
          ! update() uses this argument only to select the new fill-cell
          ! resolution. Preserve the finer survey/current grid when the
          ! production batch is small; all trial counts remain unchanged.
          call this%maps(i)%update(max(this%adaptation_points, &
               100_8*int(this%maps(i)%size_fill,8)**2),next_grid)
          n=next_grid%size
          nfill=next_grid%size_fill
          if (any(.not.ieee_is_finite(next_grid%current)).or. &
               any(.not.ieee_is_finite(next_grid%current_for_fillcell))) &
               error stop 'staged integrator: invalid adapted production grid'
          if (any(next_grid%current(1:n).le.next_grid%current(0:n-1)).or. &
               next_grid%current(0).ne.0d0.or.next_grid%current(n).ne.1d0.or. &
               any(next_grid%current_for_fillcell(1:nfill).le.next_grid%current_for_fillcell(0:nfill-1))) &
               error stop 'staged integrator: invalid adapted production grid'
          this%maps(i)=next_grid
          updated=.true.
       endif
       this%maps(i)%accum=0d0
       this%maps(i)%nhits=0
    enddo
    if (updated) this%adaptation_updates=this%adaptation_updates+1
    this%adaptation_points=0_8
    this%adaptation_batch=min(2_8*this%adaptation_batch,65536_8)
  end subroutine staged_train_production

  subroutine staged_start_pool(this,budget,cutoff)
    use,intrinsic :: ieee_arithmetic,only: ieee_is_finite
    class(staged_integrator),intent(inout) :: this
    integer(kind=8),intent(in) :: budget
    real(kind=8),intent(in) :: cutoff
    if (.not.allocated(this%maps)) error stop 'staged integrator: pool before init'
    call this%reset_native_production()
    if (this%iteration_active) error stop 'staged integrator: pool during iteration'
    if (budget.lt.0_8) error stop 'staged integrator: invalid pool budget'
    if (.not.ieee_is_finite(cutoff)) error stop 'staged integrator: invalid pool cutoff'
    if (cutoff.le.0d0) error stop 'staged integrator: invalid pool cutoff'
    if (allocated(this%candidate_weight)) deallocate(this%candidate_weight,this%candidate_priority,this%candidate_factor)
    allocate(this%candidate_weight(16),this%candidate_priority(16),this%candidate_factor(16))
    this%candidate_factor=1d0
    this%quota=0
    this%final_quota=0
    this%total_abs=0d0
    this%threshold=cutoff
    this%full_trial_tail=0d0
    this%reserve_tail=0d0
    this%worst_subset_tail=0d0
    this%envelope=cutoff
    this%max_trials=budget
    this%ntrials=0_8
    this%ncandidates=0
    this%next_pool_check=0_8
    this%max_weight=0d0
    this%overweight=0d0
    this%overweight_tolerance=0d0
    this%exhausted=budget.eq.0_8
    this%envelope_exceeded=.false.
    this%quota_complete=.false.
    this%production_done=budget.eq.0_8
    this%production_active=.true.
    this%pool_mode=.true.
    call this%reset_production_adaptation()
  end subroutine staged_start_pool

  subroutine staged_consider_pool(this,weight,to_write,done)
    use,intrinsic :: ieee_arithmetic,only: ieee_is_finite
    class(staged_integrator),intent(inout) :: this
    real(kind=8),intent(in) :: weight
    logical,intent(out) :: to_write,done
    real(kind=8) :: rnd,priority
    if (.not.this%pool_mode.or..not.this%production_active.or.this%production_done) &
         error stop 'staged integrator: candidate outside active pool'
    if (.not.ieee_is_finite(weight)) error stop 'staged integrator: invalid pool target'
    if (weight.lt.0d0) error stop 'staged integrator: invalid pool target'
    this%ntrials=this%ntrials+1_8
    this%max_weight=max(this%max_weight,weight)
    to_write=.false.
    if (weight.gt.0d0) then
       rnd=ran2()
       if (.not.ieee_is_finite(rnd)) error stop 'staged integrator: invalid RNG value'
       if (rnd.lt.0d0.or.rnd.ge.1d0) error stop 'staged integrator: invalid RNG value'
       ! Keeping priorities logarithmic avoids both overflow for large weights
       ! and underflow in cutoff*rnd. The same variate must be kept for any later
       ! threshold revision; drawing it again would distort the retained pool.
       priority=log(weight)-log(max(rnd,tiny(1d0)))
       if (priority.gt.log(this%envelope)) then
          if (this%ncandidates.eq.huge(this%ncandidates)) error stop 'staged integrator: too many candidate events'
          this%ncandidates=this%ncandidates+1
          if (this%ncandidates.gt.size(this%candidate_weight)) call this%grow_candidates()
          this%candidate_weight(this%ncandidates)=weight
          this%candidate_priority(this%ncandidates)=priority
          to_write=.true.
       endif
    endif
    this%exhausted=this%ntrials.eq.this%max_trials
    this%production_done=this%exhausted
    done=this%production_done
  end subroutine staged_consider_pool

  subroutine staged_pool_candidates(this,weights,priorities)
    class(staged_integrator),intent(inout) :: this
    real(kind=8),allocatable,intent(out) :: weights(:),priorities(:)
    if (.not.this%pool_mode.or..not.this%production_done) &
         error stop 'staged integrator: pool is not finished'
    allocate(weights(this%ncandidates),priorities(this%ncandidates))
    weights=this%candidate_weight(1:this%ncandidates)
    priorities=this%candidate_priority(1:this%ncandidates)
    this%production_active=.false.
  end subroutine staged_pool_candidates

  subroutine staged_export_pool(this,unit)
    class(staged_integrator),intent(inout) :: this
    integer,intent(in) :: unit
    real(kind=8),allocatable :: weights(:),priorities(:)
    integer :: i
    call this%pool_candidates(weights,priorities)
    write(unit,'(a)') 'MG5_SIMPLE_POOL 1'
    write(unit,*) this%ntrials,this%ncandidates
    write(unit,'(es25.17)') this%envelope
    do i=1,this%ncandidates
       write(unit,'(i0,1x,2(es25.17,1x))') i,weights(i),priorities(i)
    enddo
    write(unit,'(a)') 'END_MG5_SIMPLE_POOL'
  end subroutine staged_export_pool

  subroutine staged_consider(this,weight,to_write,done,x)
    use,intrinsic :: ieee_arithmetic,only: ieee_is_finite
    class(staged_integrator),intent(inout) :: this
    real(kind=8),intent(in) :: weight
    real(kind=8),intent(in),optional :: x(:)
    logical,intent(out) :: to_write,done
    real(kind=8) :: rnd
    real(kind=8),allocatable :: weights(:)
    if (this%pool_mode.or..not.this%production_active.or.this%production_done) &
         error stop 'staged integrator: candidate outside active production'
    if (.not.ieee_is_finite(weight).or.weight.lt.0d0) error stop 'staged integrator: invalid event target'
    this%ntrials=this%ntrials+1_8
    call this%train_production(weight,x)
    this%total_abs=this%total_abs+weight
    if (.not.ieee_is_finite(this%total_abs)) error stop 'staged integrator: absolute trial sum overflow'
    to_write=.false.
    ! Do not clip an overweight or adapt the envelope while retaining events.
    ! In particular, adaptive rank cutoffs bias single-event channel quotas.
    if (this%overweight_tolerance.eq.0d0.and.weight.gt.this%envelope) then
       this%envelope_exceeded=.true.
       this%production_done=.true.
       this%quota_complete=.false.
       done=.true.
       return
    endif
    if (weight.gt.0d0) then
       rnd=ran2()
       if (.not.ieee_is_finite(rnd).or.rnd.lt.0d0.or.rnd.ge.1d0) &
            error stop 'staged integrator: invalid RNG value'
       rnd=max(rnd,tiny(1d0))
       if (weight.gt.this%envelope*rnd) then
          to_write=.true.
          this%ncandidates=this%ncandidates+1
          if (this%ncandidates.gt.size(this%candidate_weight)) call this%grow_candidates()
          this%candidate_weight(this%ncandidates)=weight
          this%candidate_factor(this%ncandidates)=1d0
          ! Log priorities preserve ordering without overflowing for tiny rnd.
          this%candidate_priority(this%ncandidates)=log(weight)-log(rnd)
       endif
    endif
    this%quota_complete=this%ncandidates.ge.this%quota
    this%exhausted=this%ntrials.ge.this%max_trials
    if (this%overweight_tolerance.eq.0d0) then
       this%production_done=this%quota_complete.or.this%exhausted
       done=this%production_done
       return
    endif
    if (this%quota_complete.and.(int(this%ncandidates,8).ge.this%next_pool_check.or.this%exhausted)) then
       call this%select_candidates(weights)
       this%production_done=this%overweight.lt.this%overweight_tolerance.or.this%exhausted
       this%next_pool_check=int(this%ncandidates,8)+max(1_8,(int(this%quota,8)+1_8)/2_8)
    endif
    if (this%exhausted) this%production_done=.true.
    done=this%production_done
  end subroutine staged_consider

  subroutine staged_grow_candidates(this)
    class(staged_integrator),intent(inout) :: this
    real(kind=8),allocatable :: weights(:),priorities(:),factors(:),coords(:,:),logjac(:)
    integer,allocatable :: birth_epoch(:)
    integer :: n
    n=size(this%candidate_weight)
    if (n.gt.huge(n)/2) error stop 'staged integrator: too many candidate events'
    allocate(weights(2*n),priorities(2*n),factors(2*n))
    factors=1d0
    weights(1:n)=this%candidate_weight
    priorities(1:n)=this%candidate_priority
    factors(1:n)=this%candidate_factor
    call move_alloc(factors,this%candidate_factor)
    call move_alloc(weights,this%candidate_weight)
    call move_alloc(priorities,this%candidate_priority)
    if (this%native_mode) then
       allocate(coords(this%ndim,2*n),logjac(2*n),birth_epoch(2*n))
       coords(:,1:n)=this%native_candidate_x
       logjac(1:n)=this%native_birth_logjac
       birth_epoch(1:n)=this%native_birth_epoch
       call move_alloc(coords,this%native_candidate_x)
       call move_alloc(logjac,this%native_birth_logjac)
       call move_alloc(birth_epoch,this%native_birth_epoch)
    endif
  end subroutine staged_grow_candidates

  subroutine staged_select_candidates(this,weights)
    use topk_heap_mod,only: topk_largest
    class(staged_integrator),intent(inout) :: this
    real(kind=8),allocatable,intent(out) :: weights(:)
    real(kind=8),allocatable :: top(:),correction(:),mass(:),ordinary(:),smallest(:)
    integer,allocatable :: indices(:),ordinary_indices(:)
    logical,allocatable :: tail(:)
    real(kind=8) :: log_threshold,correction_sum,max_log,tail_mass,denominator
    integer :: j,ntop,ntail,nordinary,nkeep
    if (this%envelope_exceeded) error stop 'staged integrator: strict production envelope exceeded'
    if (.not.this%quota_complete) error stop 'staged integrator: event quota not reached'
    allocate(weights(this%ncandidates))
    weights=0d0
    this%overweight=0d0
    this%full_trial_tail=0d0
    this%reserve_tail=0d0
    this%worst_subset_tail=0d0
    if (this%quota.eq.0) return
    if (this%overweight_tolerance.eq.0d0) then
       if (this%ncandidates.ne.this%quota) error stop 'staged integrator: strict event count mismatch'
       weights=1d0
       this%threshold=this%envelope
       return
    endif
    ntop=min(this%quota+1,this%ncandidates)
    allocate(top(ntop),indices(ntop),correction(this%quota),mass(this%quota),tail(this%quota))
    call topk_largest(this%candidate_priority(1:this%ncandidates),ntop,top,indices)
    if (ntop.gt.this%quota) then
       log_threshold=top(ntop)
    elseif (this%envelope.gt.0d0) then
       log_threshold=log(this%envelope)
    else
       log_threshold=minval(log(this%candidate_weight(1:this%ncandidates)))
    endif
    if (log_threshold.ge.log(huge(1d0))) then
       this%threshold=huge(1d0)
    else
       this%threshold=max(this%envelope,exp(log_threshold))
    endif
    ! Every point above this threshold passed the initial retention cutoff;
    ! its raw contribution is therefore available in the candidate metadata.
    if (this%total_abs.gt.0d0) this%full_trial_tail= &
         sum(this%candidate_weight(1:this%ncandidates), &
             mask=this%candidate_weight(1:this%ncandidates).gt.this%threshold)/this%total_abs
    do j=1,this%quota
       correction(j)=max(0d0,log(this%candidate_weight(indices(j)))-log_threshold)
       tail(j)=this%candidate_weight(indices(j)).gt.this%threshold
    enddo
    max_log=maxval(correction)
    correction=exp(correction-max_log)
    correction_sum=sum(correction)
    do j=1,this%quota
       weights(indices(j))=correction(j)*dble(this%quota)/correction_sum
       if (weights(indices(j)).eq.0d0) error stop 'staged integrator: correction exceeds floating-point dynamic range'
       mass(j)=log(correction(j))+log(this%candidate_factor(indices(j)))
    enddo
    ! Common rescaling avoids overflow and cancels in all tail fractions.
    mass=exp(mass-maxval(mass))
    tail_mass=sum(mass,mask=tail)
    denominator=sum(mass)
    this%reserve_tail=tail_mass/denominator
    ntail=count(tail)
    if (ntail.gt.0.and.this%final_quota.gt.0) then
       if (ntail.ge.this%final_quota) then
          this%worst_subset_tail=1d0
       else
          nkeep=this%final_quota-ntail
          nordinary=this%quota-ntail
          allocate(ordinary(nordinary),smallest(nkeep),ordinary_indices(nkeep))
          ordinary=-pack(mass,.not.tail)
          call topk_largest(ordinary,nkeep,smallest,ordinary_indices)
          this%worst_subset_tail=tail_mass/(tail_mass-sum(smallest))
       endif
    endif
    this%overweight=max(this%full_trial_tail,this%reserve_tail,this%worst_subset_tail)
  end subroutine staged_select_candidates

  subroutine staged_record_candidate_factor(this,factor,done)
    use,intrinsic :: ieee_arithmetic,only: ieee_is_finite
    class(staged_integrator),intent(inout) :: this
    real(kind=8),intent(in) :: factor
    logical,intent(out) :: done
    real(kind=8),allocatable :: weights(:)
    if (this%ncandidates.le.0.or..not.this%production_active.or.this%pool_mode) &
         error stop 'staged integrator: no production candidate for LHE factor'
    if (.not.ieee_is_finite(factor).or.factor.le.0d0) error stop 'staged integrator: invalid LHE candidate factor'
    this%candidate_factor(this%ncandidates)=factor
    if (this%native_mode) then
       if (this%production_done) error stop 'native integrator: LHE factor after iteration finalization'
       done=.false.
       return
    endif
    ! consider() sees a provisional factor of one. Its final decision must be
    ! repeated after the actual selected contribution and inverse bias exist.
    if (this%production_done.and.this%quota_complete.and.this%overweight_tolerance.gt.0d0) then
       call this%select_candidates(weights)
       this%production_done=this%overweight.lt.this%overweight_tolerance.or.this%exhausted
    endif
    done=this%production_done
  end subroutine staged_record_candidate_factor

  subroutine staged_production_candidates(this,raw,priorities,weights,tail,factors)
    class(staged_integrator),intent(inout) :: this
    real(kind=8),allocatable,intent(out) :: raw(:),priorities(:),weights(:),factors(:)
    integer,allocatable,intent(out) :: tail(:)
    call this%final_weights(weights)
    raw=this%candidate_weight(1:this%ncandidates)
    priorities=this%candidate_priority(1:this%ncandidates)
    factors=this%candidate_factor(1:this%ncandidates)
    allocate(tail(this%ncandidates))
    tail=0
    where(raw.gt.this%threshold) tail=1
  end subroutine staged_production_candidates

  subroutine staged_final_weights(this,weights)
    class(staged_integrator),intent(inout) :: this
    real(kind=8),allocatable,intent(out) :: weights(:)
    if (this%pool_mode) error stop 'staged integrator: pool requires coordinator finalization'
    if (.not.this%production_active.or..not.this%production_done) &
         error stop 'staged integrator: production is not finished'
    call this%select_candidates(weights)
    this%production_active=.false.
  end subroutine staged_final_weights

  subroutine reset_native_production(this)
    class(staged_integrator),intent(inout) :: this
    if (allocated(this%native_epochs)) deallocate(this%native_epochs)
    if (allocated(this%native_candidate_x)) &
         deallocate(this%native_candidate_x,this%native_birth_logjac,this%native_birth_epoch)
    if (allocated(this%native_correction)) deallocate(this%native_correction,this%native_tail)
    this%native_mode=.false.
    this%native_iteration_done=.false.
    this%native_iteration=0
    this%native_max_iterations=0
    this%native_target_nonzero=0_8
    this%native_nonzero=0_8
    this%native_effective_generated=0d0
    this%native_expected_remaining=0d0
    this%native_generation_efficiency=0d0
    this%native_mean=0d0
    this%native_m2=0d0
    this%native_covariance=0d0
    this%native_log_z=0d0
  end subroutine reset_native_production

  subroutine start_native_production(this,quota,final_quota,ifold,survey_absolute,initial_envelope, &
       max_trials,max_iterations,initial_nonzero)
    use,intrinsic :: ieee_arithmetic,only: ieee_is_finite
    class(staged_integrator),intent(inout) :: this
    integer,intent(in) :: quota,final_quota,ifold(:)
    real(kind=8),intent(in) :: survey_absolute,initial_envelope
    integer(kind=8),intent(in) :: max_trials
    integer,intent(in),optional :: max_iterations
    integer(kind=8),intent(in),optional :: initial_nonzero
    real(kind=8) :: storage_cutoff
    integer :: i
    if (.not.ieee_is_finite(survey_absolute).or..not.ieee_is_finite(initial_envelope)) &
         error stop 'native integrator: invalid survey estimate or envelope'
    if (survey_absolute.lt.0d0.or.initial_envelope.le.0d0.or.quota.lt.0.or.final_quota.lt.0.or. &
         (quota.gt.0.and.(survey_absolute.eq.0d0.or.final_quota.eq.0)).or.(quota.eq.0.and.final_quota.ne.0)) &
         error stop 'native integrator: positive survey rate and event quotas required'
    call this%start_production(quota,initial_envelope,max_trials,1d-2,final_quota)
    if (size(ifold).ne.this%ndim) error stop 'native integrator: folding dimension mismatch'
    if (any(ifold.lt.1)) error stop 'native integrator: invalid folding factor'
    this%native_mode=.true.
    this%native_max_iterations=32
    if (present(max_iterations)) this%native_max_iterations=max_iterations
    if (this%native_max_iterations.lt.1) error stop 'native integrator: invalid iteration limit'
    allocate(this%native_epochs(this%native_max_iterations))
    allocate(this%native_candidate_x(this%ndim,size(this%candidate_weight)), &
         this%native_birth_logjac(size(this%candidate_weight)),this%native_birth_epoch(size(this%candidate_weight)))
    this%adaptation_mask=ifold.eq.1
    if (quota.eq.0) return
    this%native_iteration=1
    ! Start with a modest batch so a rare survey maximum cannot postpone
    ! grid and envelope updates for the entire event sample. Subsequent
    ! nonzero budgets use the observed generation efficiency.
    this%native_target_nonzero=min(max_trials,max(1024_8,min(8192_8,int(quota,8))))
    if (present(initial_nonzero)) then
       if (initial_nonzero.lt.1_8) error stop 'native integrator: invalid nonzero point target'
       this%native_target_nonzero=initial_nonzero
    endif
    if (this%native_target_nonzero.gt.max_trials) error stop 'native integrator: point request exceeds safety limit'
    this%native_epochs(1)%target_nonzero=this%native_target_nonzero
    ! The surveyed maximum seeds the first envelope estimate. Candidate
    ! storage deliberately uses a lower cutoff: it may safely retain extra
    ! observations, while an excessive storage floor would constrain all
    ! historical unweighting thresholds until this epoch expired.
    storage_cutoff=max(min(initial_envelope,survey_absolute),tiny(1d0))
    this%native_epochs(1)%envelope=initial_envelope
    this%native_epochs(1)%cutoff=storage_cutoff
    allocate(this%native_epochs(1)%maps(this%ndim))
    this%native_epochs(1)%maps=this%maps
    do i=1,this%ndim
       this%maps(i)%accum=0d0
       this%maps(i)%nhits=0
    enddo
    write(*,*) 'AmpliCol native surveyed maximum, initial storage cutoff:',initial_envelope,storage_cutoff
    write(*,*) 'AmpliCol native iteration, nonzero target, storage envelope:', &
         this%native_iteration,this%native_target_nonzero,storage_cutoff
  end subroutine start_native_production

  function native_log_jacobian(this,x,epoch) result(logjac)
    use,intrinsic :: ieee_arithmetic,only: ieee_is_finite
    class(staged_integrator),intent(inout) :: this
    real(kind=8),intent(in) :: x(:)
    integer,intent(in) :: epoch
    real(kind=8) :: logjac,jac
    integer :: i
    logjac=0d0
    do i=1,this%ndim
       if (.not.this%adaptation_mask(i)) cycle
       jac=1d0
       if (epoch.eq.0) then
          call this%maps(i)%get_wgt(x(i),jac)
       else
          call this%native_epochs(epoch)%maps(i)%get_wgt(x(i),jac)
       endif
       if (.not.ieee_is_finite(jac)) error stop 'native integrator: nonfinite grid Jacobian'
       if (jac.le.0d0) error stop 'native integrator: nonpositive grid Jacobian'
       logjac=logjac+log(jac)
    enddo
  end function native_log_jacobian

  function native_reweighted_value(this,candidate,epoch) result(value)
    class(staged_integrator),intent(inout) :: this
    integer,intent(in) :: candidate,epoch
    real(kind=8) :: value,logvalue
    if (epoch.eq.this%native_birth_epoch(candidate)) then
       value=this%candidate_weight(candidate)
       return
    endif
    logvalue=log(this%candidate_weight(candidate))+ &
         this%native_log_jacobian(this%native_candidate_x(:,candidate),epoch)-this%native_birth_logjac(candidate)
    if (logvalue.ge.log(huge(1d0))) error stop 'native integrator: historical envelope overflow'
    value=exp(logvalue)
  end function native_reweighted_value

  subroutine native_consider(this,values,x,to_write,iteration_done)
    use,intrinsic :: ieee_arithmetic,only: ieee_is_finite
    class(staged_integrator),intent(inout) :: this
    real(kind=8),intent(in) :: values(2),x(:)
    logical,intent(out) :: to_write,iteration_done
    real(kind=8) :: delta(2),rnd,priority
    integer :: i,k,j
    if (.not.this%native_mode.or..not.this%production_active.or.this%production_done.or. &
         this%native_iteration_done) error stop 'native integrator: draw outside active iteration'
    if (any(.not.ieee_is_finite(values))) error stop 'native integrator: nonfinite contribution'
    if (values(1).lt.0d0) error stop 'native integrator: negative absolute contribution'
    if (size(x).ne.this%ndim) error stop 'native integrator: coordinate dimension mismatch'
    if (any(.not.ieee_is_finite(x))) error stop 'native integrator: invalid coordinates'
    if (any(x.lt.0d0).or.any(x.gt.1d0)) error stop 'native integrator: invalid coordinates'
    k=this%native_iteration
    this%ntrials=this%ntrials+1_8
    this%native_epochs(k)%trials=this%native_epochs(k)%trials+1_8
    delta=values-this%native_mean
    this%native_mean=this%native_mean+delta/dble(this%ntrials)
    this%native_m2=this%native_m2+delta*(values-this%native_mean)
    this%native_covariance=this%native_covariance+delta(1)*(values(2)-this%native_mean(2))
    delta=values-this%native_epochs(k)%mean
    this%native_epochs(k)%mean=this%native_epochs(k)%mean+delta/dble(this%native_epochs(k)%trials)
    this%native_epochs(k)%m2=this%native_epochs(k)%m2+delta*(values-this%native_epochs(k)%mean)
    this%native_epochs(k)%covariance=this%native_epochs(k)%covariance+ &
         delta(1)*(values(2)-this%native_epochs(k)%mean(2))
    if (.not.all(ieee_is_finite(this%native_m2)).or..not.ieee_is_finite(this%native_covariance)) &
         error stop 'native integrator: production moments overflow'
    this%total_abs=this%total_abs+values(1)
    if (.not.ieee_is_finite(this%total_abs)) error stop 'native integrator: absolute trial sum overflow'
    do i=1,this%ndim
       if (this%adaptation_mask(i)) call this%maps(i)%add_point(x(i),values(1))
    enddo
    to_write=.false.
    if (values(1).gt.0d0) then
       this%native_epochs(k)%nonzero=this%native_epochs(k)%nonzero+1_8
       rnd=ran2()
       if (.not.ieee_is_finite(rnd)) error stop 'native integrator: invalid random value'
       if (rnd.lt.0d0.or.rnd.ge.1d0) error stop 'native integrator: invalid random value'
       priority=log(values(1))-log(max(rnd,tiny(1d0)))
       if (priority.gt.log(this%native_epochs(k)%cutoff)) then
          to_write=.true.
          if (this%ncandidates.eq.huge(this%ncandidates)) error stop 'native integrator: too many candidates'
          this%ncandidates=this%ncandidates+1
          if (this%ncandidates.gt.size(this%candidate_weight)) call this%grow_candidates()
          j=this%ncandidates
          this%candidate_weight(j)=values(1)
          this%candidate_priority(j)=priority
          this%candidate_factor(j)=1d0
          this%native_candidate_x(:,j)=x
          this%native_birth_epoch(j)=k
          this%native_birth_logjac(j)=this%native_log_jacobian(x,0)
       endif
    endif
    this%native_nonzero=this%native_epochs(k)%nonzero
    this%native_iteration_done=this%native_nonzero.ge.this%native_target_nonzero
    if (this%ntrials.ge.this%max_trials.and..not.this%native_iteration_done) &
         error stop 'native integrator: trial safety limit before nonzero iteration target'
    iteration_done=this%native_iteration_done
  end subroutine native_consider

  subroutine native_update_envelopes(this)
    class(staged_integrator),intent(inout) :: this
    integer :: i,k,remaining
    real(kind=8) :: value
    remaining=final_n_iters_for_evnt_gen
    do k=this%native_iteration,1,-1
       this%native_epochs(k)%eligible=.false.
       this%native_epochs(k)%threshold=0d0
       if (remaining.gt.0.and.any(this%native_birth_epoch(1:this%ncandidates).eq.k)) then
          this%native_epochs(k)%eligible=.true.
          remaining=remaining-1
       endif
       this%native_epochs(k)%envelope=0d0
       ! The first proposal is exactly the surveyed grid. Its observed survey
       ! maximum remains valid information for this epoch even if production
       ! has not sampled that point again. Later proposals get fresh estimates.
       if (k.eq.1) this%native_epochs(k)%envelope=this%envelope
       do i=1,this%ncandidates
          value=this%native_reweighted_value(i,k)
          this%native_epochs(k)%envelope=max(this%native_epochs(k)%envelope,value)
       enddo
    enddo
  end subroutine native_update_envelopes

  subroutine native_select(this)
    use topk_heap_mod,only: topk_largest
    class(staged_integrator),intent(inout) :: this
    real(kind=8),allocatable :: priorities(:),top(:),logs(:),mass(:),ordinary(:),smallest(:)
    integer,allocatable :: labels(:),indices(:),smallest_indices(:)
    logical,allocatable :: tail(:)
    real(kind=8) :: floor_z,log_threshold,tail_sum,total_sum,max_log,correction_sum,tail_mass
    integer :: i,j,k,nactive,ntail,nkeep
    if (allocated(this%native_correction)) deallocate(this%native_correction,this%native_tail)
    allocate(this%native_correction(this%ncandidates),this%native_tail(this%ncandidates))
    this%native_correction=0d0
    this%native_tail=0
    this%quota_complete=.false.
    this%overweight=huge(1d0)
    this%full_trial_tail=1d0
    this%reserve_tail=1d0
    this%worst_subset_tail=1d0
    nactive=0
    floor_z=-huge(1d0)
    do k=1,this%native_iteration
       if (.not.this%native_epochs(k)%eligible) cycle
       nactive=nactive+count(this%native_birth_epoch(1:this%ncandidates).eq.k)
       floor_z=max(floor_z,log(this%native_epochs(k)%cutoff)-log(this%native_epochs(k)%envelope))
    enddo
    this%native_log_z=floor_z
    if (nactive.le.this%quota) return
    allocate(priorities(nactive),labels(nactive),top(this%quota+1),indices(this%quota+1))
    j=0
    do i=1,this%ncandidates
       k=this%native_birth_epoch(i)
       if (.not.this%native_epochs(k)%eligible) cycle
       j=j+1
       labels(j)=i
       priorities(j)=this%candidate_priority(i)-log(this%native_epochs(k)%envelope)
    enddo
    call topk_largest(priorities,this%quota+1,top,indices)
    ! Native normalization compares different proposal epochs. The first
    ! excluded normalized priority sets the common rank threshold. Never
    ! lower an epoch threshold below the cutoff used to store its candidates.
    this%native_log_z=max(top(this%quota+1),floor_z)
    if (floor_z.gt.top(this%quota)) return
    this%quota_complete=.true.
    tail_sum=0d0
    total_sum=0d0
    do k=1,this%native_iteration
       if (.not.this%native_epochs(k)%eligible) cycle
       log_threshold=log(this%native_epochs(k)%envelope)+this%native_log_z
       if (log_threshold.ge.log(huge(1d0))) then
          this%native_epochs(k)%threshold=huge(1d0)
       else
          this%native_epochs(k)%threshold=max(this%native_epochs(k)%cutoff,exp(log_threshold))
       endif
       total_sum=total_sum+dble(this%native_epochs(k)%trials)*this%native_epochs(k)%mean(1)
    enddo
    do i=1,this%ncandidates
       k=this%native_birth_epoch(i)
       if (.not.this%native_epochs(k)%eligible) cycle
       if (this%candidate_weight(i).gt.this%native_epochs(k)%threshold) then
          this%native_tail(i)=1
          tail_sum=tail_sum+this%candidate_weight(i)
       endif
    enddo
    this%full_trial_tail=tail_sum/total_sum
    allocate(logs(this%quota),mass(this%quota),tail(this%quota))
    do j=1,this%quota
       i=labels(indices(j))
       k=this%native_birth_epoch(i)
       logs(j)=max(0d0,log(this%candidate_weight(i))-log(this%native_epochs(k)%threshold))
       tail(j)=this%native_tail(i).eq.1
    enddo
    max_log=maxval(logs)
    correction_sum=sum(exp(logs-max_log))
    do j=1,this%quota
       i=labels(indices(j))
       this%native_correction(i)=exp(logs(j)-max_log)*dble(this%quota)/correction_sum
       if (this%native_correction(i).eq.0d0) error stop 'native integrator: correction underflow'
       mass(j)=logs(j)+log(this%candidate_factor(i))
    enddo
    mass=exp(mass-maxval(mass))
    tail_mass=sum(mass,mask=tail)
    this%reserve_tail=tail_mass/sum(mass)
    this%worst_subset_tail=0d0
    ntail=count(tail)
    if (ntail.gt.0) then
       if (ntail.ge.this%final_quota) then
          this%worst_subset_tail=1d0
       else
          nkeep=this%final_quota-ntail
          allocate(ordinary(this%quota-ntail),smallest(nkeep),smallest_indices(nkeep))
          ordinary=-pack(mass,.not.tail)
          call topk_largest(ordinary,nkeep,smallest,smallest_indices)
          this%worst_subset_tail=tail_mass/(tail_mass-sum(smallest))
       endif
    endif
    this%overweight=max(this%full_trial_tail,this%reserve_tail,this%worst_subset_tail)
  end subroutine native_select

  subroutine native_update_maps(this)
    use,intrinsic :: ieee_arithmetic,only: ieee_is_finite
    class(staged_integrator),intent(inout) :: this
    type(grid) :: next_grid
    integer :: i,n
    logical :: updated
    updated=.false.
    do i=1,this%ndim
       if (.not.this%adaptation_mask(i)) cycle
       if (maxval(this%maps(i)%accum).gt.0d0) then
          call this%maps(i)%update(max(this%native_epochs(this%native_iteration)%trials, &
               100_8*int(this%maps(i)%size_fill,8)**2),next_grid)
          if (any(.not.ieee_is_finite(next_grid%current))) error stop 'native integrator: nonfinite adapted grid'
          n=next_grid%size
          if (any(next_grid%current(1:n).le.next_grid%current(0:n-1)).or. &
               next_grid%current(0).ne.0d0.or.next_grid%current(n).ne.1d0) &
               error stop 'native integrator: non-monotone adapted grid'
          this%maps(i)=next_grid
          updated=.true.
       endif
       this%maps(i)%accum=0d0
       this%maps(i)%nhits=0
    enddo
    if (updated) this%adaptation_updates=this%adaptation_updates+1
  end subroutine native_update_maps

  function native_next_cutoff(this) result(cutoff)
    use topk_heap_mod,only: topk_largest
    class(staged_integrator),intent(inout) :: this
    real(kind=8) :: cutoff,nonzero_mean
    real(kind=8),allocatable :: values(:),top(:)
    integer,allocatable :: indices(:)
    integer :: i,k
    cutoff=this%native_epochs(this%native_iteration)%cutoff
    if (this%ncandidates.le.200) then
       ! A sparse retained pool is precisely when an excessive storage cutoff
       ! needs to recover. Use all trials, including rejected observations,
       ! rather than keeping the same cutoff until 200 candidates exist.
       k=this%native_iteration
       nonzero_mean=this%native_epochs(k)%mean(1)* &
            (dble(this%native_epochs(k)%trials)/dble(max(1_8,this%native_epochs(k)%nonzero)))
       if (nonzero_mean.gt.0d0) cutoff=max(min(cutoff,2d0*nonzero_mean),tiny(1d0))
       return
    endif
    allocate(values(this%ncandidates))
    do i=1,this%ncandidates
       values(i)=this%native_reweighted_value(i,0)
    enddo
    k=max(int(write_evnt_fraction*this%ncandidates),1)
    k=max(int(dble(k)*dble(this%native_max_iterations-this%native_iteration)/dble(this%native_max_iterations)),1)
    allocate(top(k),indices(k))
    call topk_largest(values,k,top,indices)
    cutoff=max(top(k),tiny(1d0))
  end function native_next_cutoff

  subroutine finish_native_iteration(this,done)
    class(staged_integrator),intent(inout) :: this
    logical,intent(out) :: done
    integer(kind=8) :: next_target,remaining_trials,growth_limit
    real(kind=8) :: cutoff,estimate,priority
    integer :: i,k,birth,available,current_available,expiring,expiring_available
    if (.not.this%native_mode.or..not.this%native_iteration_done.or.this%production_done) &
         error stop 'native integrator: incomplete or already finalized iteration'
    k=this%native_iteration
    call this%native_update_envelopes()
    call this%native_select()
    done=this%quota_complete.and.this%overweight.lt.this%overweight_tolerance
    write(*,*) 'AmpliCol native completed iteration, trials, nonzero, ABS, signed:', &
         k,this%native_epochs(k)%trials,this%native_epochs(k)%nonzero,this%native_mean
    write(*,*) 'AmpliCol native full-trial, reserve, worst-collected tails:', &
         this%full_trial_tail,this%reserve_tail,this%worst_subset_tail
    if (done) then
       this%production_done=.true.
       return
    endif
    if (k.ge.this%native_max_iterations) error stop 'native integrator: iteration limit before event/tail completion'
    remaining_trials=this%max_trials-this%ntrials
    if (remaining_trials.le.0_8) error stop 'native integrator: trial safety limit before event/tail completion'
    ! Forecast from events that survive the current native epoch envelopes,
    ! not the number of raw stored candidates. Expired epochs never count.
    ! When the tail fails, use AmpliCol's reduction of effective completion.
    available=0
    current_available=0
    expiring=0
    expiring_available=0
    ! The next event-producing epoch displaces the oldest one once the
    ! eight-epoch history is full. Budget its replacement now, otherwise
    ! near-quota forecasts can repeatedly lose as many events as they add.
    if (count(this%native_epochs(1:k)%eligible).ge.final_n_iters_for_evnt_gen) then
       do i=1,k
          if (this%native_epochs(i)%eligible) then
             expiring=i
             exit
          endif
       enddo
    endif
    do i=1,this%ncandidates
       birth=this%native_birth_epoch(i)
       if (.not.this%native_epochs(birth)%eligible) cycle
       priority=this%candidate_priority(i)-log(this%native_epochs(birth)%envelope)
       if (priority.lt.this%native_log_z) cycle
       available=available+1
       if (birth.eq.k) current_available=current_available+1
       if (birth.eq.expiring) expiring_available=expiring_available+1
    enddo
    this%native_effective_generated=dble(min(this%quota,available-expiring_available))
    if (this%quota_complete) this%native_effective_generated=min(this%native_effective_generated, &
         0.8d0*dble(this%quota),dble(this%quota)*this%overweight_tolerance/this%overweight)
    this%native_expected_remaining=max(1d0,dble(this%quota)-this%native_effective_generated)
    this%native_generation_efficiency=dble(current_available)/dble(max(1_8,this%native_epochs(k)%nonzero))
    ! Doubling is a growth ceiling, not a compulsory minimum. There is no
    ! fixed upper batch cap: large quotas must still fit in the finite event
    ! history. The small floor prevents noisy tiny adaptation iterations.
    growth_limit=min(this%native_target_nonzero,remaining_trials)
    growth_limit=growth_limit+min(growth_limit,remaining_trials-growth_limit)
    next_target=growth_limit
    if (this%native_generation_efficiency.gt.0d0) then
       estimate=1.1d0*this%native_expected_remaining/this%native_generation_efficiency
       if (estimate.lt.dble(growth_limit)) &
            next_target=max(min(1024_8,growth_limit),ceiling(estimate,kind=8))
    endif
    write(*,*) 'AmpliCol native effective events, remaining, nonzero acceptance:', &
         this%native_effective_generated,this%native_expected_remaining,this%native_generation_efficiency
    next_target=max(1_8,min(next_target,remaining_trials))
    call this%native_update_maps()
    cutoff=this%native_next_cutoff()
    this%native_iteration=k+1
    this%native_target_nonzero=next_target
    this%native_nonzero=0_8
    this%native_iteration_done=.false.
    this%native_epochs(k+1)%cutoff=cutoff
    this%native_epochs(k+1)%target_nonzero=next_target
    allocate(this%native_epochs(k+1)%maps(this%ndim))
    this%native_epochs(k+1)%maps=this%maps
    do i=1,this%ndim
       this%maps(i)%accum=0d0
       this%maps(i)%nhits=0
    enddo
    write(*,*) 'AmpliCol native iteration, nonzero target, storage envelope:', &
         this%native_iteration,next_target,cutoff
  end subroutine finish_native_iteration

  subroutine native_rates(this,mean,error,moments)
    class(staged_integrator),intent(in) :: this
    real(kind=8),intent(out) :: mean(2),error(2),moments(5)
    if (.not.this%native_mode) error stop 'native integrator: rates outside native production'
    mean=this%native_mean
    error=0d0
    if (this%ntrials.gt.0_8) error=sqrt(max(this%native_m2,0d0))/dble(this%ntrials)
    moments=[this%native_mean,this%native_m2,this%native_covariance]
  end subroutine native_rates

  subroutine write_native_pool(this,unit)
    class(staged_integrator),intent(inout) :: this
    integer,intent(in) :: unit
    integer :: i,k
    if (.not.this%native_mode.or..not.this%production_done.or..not.this%quota_complete) &
         error stop 'native integrator: incomplete event reserve'
    if (this%overweight.ge.this%overweight_tolerance) error stop 'native integrator: full-tail bound not reached'
    write(unit,'(a)') 'MG5_AMPLI_POOL 4'
    write(unit,*) this%ntrials,this%ncandidates,this%native_iteration
    write(unit,*) this%native_mean,this%native_m2,this%native_covariance
    write(unit,*) this%quota,this%final_quota,this%native_log_z, &
         this%full_trial_tail,this%reserve_tail,this%worst_subset_tail
    write(unit,*) this%ndim,this%adaptation_updates
    write(unit,*) (merge(1,0,this%adaptation_mask(i)),i=1,this%ndim)
    do k=1,this%native_iteration
       write(unit,*) k,this%native_epochs(k)%trials,this%native_epochs(k)%nonzero, &
            this%native_epochs(k)%target_nonzero,this%native_epochs(k)%mean,this%native_epochs(k)%m2, &
            this%native_epochs(k)%covariance,this%native_epochs(k)%cutoff,this%native_epochs(k)%envelope, &
            this%native_epochs(k)%threshold,merge(1,0,this%native_epochs(k)%eligible)
    enddo
    do i=1,this%ncandidates
       write(unit,*) this%native_birth_epoch(i),this%candidate_weight(i),this%candidate_priority(i), &
            this%native_correction(i),this%native_tail(i),this%candidate_factor(i)
    enddo
  end subroutine write_native_pool

  subroutine staged_save(this,unit)
    class(staged_integrator),intent(in) :: this
    integer,intent(in) :: unit
    integer :: i
    if (.not.allocated(this%maps)) error stop 'staged integrator: save before init'
    if (this%native_mode) error stop 'native integrator: restart production from survey; active checkpoint unsupported'
    write(unit,'(a)') 'MG5_SIMPLE_INTEGRATOR 4'
    write(unit,*) this%ndim,this%nvalues
    write(unit,*) this%npoints,this%total_points,this%ntrials,this%max_trials,this%next_pool_check
    write(unit,*) this%quota,this%ncandidates
    write(unit,*) this%final_quota,this%total_abs,this%threshold,this%full_trial_tail,this%reserve_tail,this%worst_subset_tail
    write(unit,'(4(es25.17,1x))') this%max_weight,this%envelope,this%overweight,this%overweight_tolerance
    write(unit,*) this%iteration_active,this%adapt_iteration,this%production_active,&
         this%production_done,this%quota_complete,this%exhausted,this%envelope_exceeded,this%pool_mode
    write(unit,*) this%adaptation_updates,this%adaptation_interval,this%adaptation_batch,this%adaptation_points
    write(unit,*) this%adaptation_mask
    write(unit,'(*(es25.17,1x))') this%mean
    write(unit,'(*(es25.17,1x))') this%m2
    write(unit,'(*(es25.17,1x))') this%res
    write(unit,'(*(es25.17,1x))') this%unc
    do i=1,this%ndim
       write(unit,*) this%maps(i)%size,this%maps(i)%size_fill
       write(unit,'(*(es25.17,1x))') this%maps(i)%current
       write(unit,'(*(es25.17,1x))') this%maps(i)%current_for_fillcell
       write(unit,'(*(es25.17,1x))') this%maps(i)%accum
       write(unit,*) this%maps(i)%nhits
    enddo
    do i=1,this%ncandidates
       write(unit,'(3(es25.17,1x))') this%candidate_weight(i),this%candidate_priority(i),this%candidate_factor(i)
    enddo
    write(unit,'(a)') 'END_MG5_SIMPLE_INTEGRATOR'
  end subroutine staged_save

  subroutine staged_load(this,unit,ndim,nvalues)
    use,intrinsic :: ieee_arithmetic,only: ieee_is_finite
    class(staged_integrator),intent(inout) :: this
    integer,intent(in) :: unit,ndim,nvalues
    integer :: saved_ndim,saved_nvalues,ios,i,n,nfill,version
    integer(kind=8) :: expected_batch,remaining_points,completed_batches
    character(len=80) :: marker
    read(unit,'(a)',iostat=ios) marker
    if (ios.ne.0) error stop 'staged integrator: missing checkpoint header'
    select case(trim(marker))
    case('MG5_SIMPLE_INTEGRATOR 1')
       version=1
    case('MG5_SIMPLE_INTEGRATOR 2')
       version=2
    case('MG5_SIMPLE_INTEGRATOR 3')
       version=3
    case('MG5_SIMPLE_INTEGRATOR 4')
       version=4
    case default
       error stop 'staged integrator: incompatible checkpoint version'
    end select
    read(unit,*,iostat=ios) saved_ndim,saved_nvalues
    if (ios.ne.0) error stop 'staged integrator: invalid checkpoint dimensions'
    if (saved_ndim.ne.ndim.or.saved_nvalues.ne.nvalues) error stop 'staged integrator: incompatible checkpoint dimensions'
    call this%init(ndim,nvalues)
    read(unit,*,iostat=ios) this%npoints,this%total_points,this%ntrials,this%max_trials,this%next_pool_check
    if (ios.ne.0) error stop 'staged integrator: invalid checkpoint counters'
    read(unit,*,iostat=ios) this%quota,this%ncandidates
    if (ios.ne.0) error stop 'staged integrator: invalid checkpoint event counters'
    this%final_quota=this%quota
    if (version.ge.3) then
       read(unit,*,iostat=ios) this%final_quota,this%total_abs,this%threshold, &
            this%full_trial_tail,this%reserve_tail,this%worst_subset_tail
       if (ios.ne.0.or.this%final_quota.lt.0.or.this%final_quota.gt.this%quota.or. &
            .not.all(ieee_is_finite([this%total_abs,this%threshold,this%full_trial_tail, &
            this%reserve_tail,this%worst_subset_tail]))) error stop 'staged integrator: invalid checkpoint tail state'
    elseif (this%ntrials.gt.0_8) then
       error stop 'staged integrator: old production checkpoint lacks tail statistics'
    endif
    if (this%npoints.lt.0_8.or.this%total_points.lt.this%npoints.or.this%ntrials.lt.0_8.or.&
         this%max_trials.lt.this%ntrials.or.this%next_pool_check.lt.0_8.or.this%quota.lt.0.or.this%ncandidates.lt.0) &
         error stop 'staged integrator: corrupt checkpoint counters'
    read(unit,*,iostat=ios) this%max_weight,this%envelope,this%overweight,this%overweight_tolerance
    if (ios.ne.0) error stop 'staged integrator: invalid checkpoint envelopes'
    if (.not.all(ieee_is_finite([this%max_weight,this%envelope,this%overweight,this%overweight_tolerance])).or.&
         min(this%max_weight,this%envelope,this%overweight,this%overweight_tolerance).lt.0d0) &
         error stop 'staged integrator: corrupt checkpoint envelopes'
    if (version.eq.1) then
       read(unit,*,iostat=ios) this%iteration_active,this%adapt_iteration,this%production_active,&
            this%production_done,this%quota_complete,this%exhausted,this%envelope_exceeded
    else
       read(unit,*,iostat=ios) this%iteration_active,this%adapt_iteration,this%production_active,&
            this%production_done,this%quota_complete,this%exhausted,this%envelope_exceeded,this%pool_mode
    endif
    if (ios.ne.0) error stop 'staged integrator: invalid checkpoint flags'
    if (version.ge.4) then
       read(unit,*,iostat=ios) this%adaptation_updates,this%adaptation_interval,this%adaptation_batch,this%adaptation_points
       if (ios.ne.0) error stop 'staged integrator: invalid checkpoint adaptation counters'
       read(unit,*,iostat=ios) this%adaptation_mask
       if (ios.ne.0) error stop 'staged integrator: invalid checkpoint adaptation mask'
       if (any(this%adaptation_mask)) then
          if (this%adaptation_interval.lt.1_8.or.this%adaptation_interval.gt.65536_8.or. &
               this%adaptation_updates.lt.0.or.this%pool_mode.or.this%iteration_active) &
               error stop 'staged integrator: invalid checkpoint production adaptation'
          expected_batch=this%adaptation_interval
          remaining_points=this%ntrials
          completed_batches=0_8
          do while (expected_batch.lt.65536_8.and.remaining_points.ge.expected_batch)
             remaining_points=remaining_points-expected_batch
             completed_batches=completed_batches+1_8
             expected_batch=min(2_8*expected_batch,65536_8)
          enddo
          completed_batches=completed_batches+remaining_points/expected_batch
          remaining_points=mod(remaining_points,expected_batch)
          if (this%adaptation_batch.ne.expected_batch.or.this%adaptation_points.ne.remaining_points.or. &
               int(this%adaptation_updates,8).gt.completed_batches) &
               error stop 'staged integrator: inconsistent checkpoint adaptation schedule'
       elseif (this%adaptation_updates.ne.0.or.this%adaptation_interval.ne.0_8.or. &
            this%adaptation_batch.ne.0_8.or.this%adaptation_points.ne.0_8) then
          error stop 'staged integrator: disabled checkpoint adaptation has counters'
       endif
    endif
    if (int(this%ncandidates,8).gt.this%ntrials) error stop 'staged integrator: corrupt checkpoint event counters'
    if (this%pool_mode) then
       if (this%quota.ne.0.or.this%envelope.le.0d0.or.this%iteration_active.or.&
            this%envelope_exceeded.or.this%quota_complete.or.&
            (this%production_done.neqv.(this%ntrials.eq.this%max_trials))) &
            error stop 'staged integrator: corrupt checkpoint pool state'
    endif
    read(unit,*,iostat=ios) this%mean
    if (ios.ne.0) error stop 'staged integrator: invalid checkpoint mean'
    read(unit,*,iostat=ios) this%m2
    if (ios.ne.0) error stop 'staged integrator: invalid checkpoint variance'
    read(unit,*,iostat=ios) this%res
    if (ios.ne.0) error stop 'staged integrator: invalid checkpoint result'
    read(unit,*,iostat=ios) this%unc
    if (ios.ne.0) error stop 'staged integrator: invalid checkpoint uncertainty'
    if (any(.not.ieee_is_finite(this%mean)).or.any(.not.ieee_is_finite(this%m2)).or.&
         any(.not.ieee_is_finite(this%res)).or.any(.not.ieee_is_finite(this%unc)).or.any(this%unc.lt.0d0)) &
         error stop 'staged integrator: corrupt checkpoint statistics'
    do i=1,this%ndim
       read(unit,*,iostat=ios) n,nfill
       if (ios.ne.0) error stop 'staged integrator: invalid checkpoint grid size'
       if (n.ne.max_grid_size.or.nfill.lt.min_grid_size.or.nfill.gt.max_grid_size) &
            error stop 'staged integrator: incompatible checkpoint grid size'
       deallocate(this%maps(i)%current,this%maps(i)%current_for_fillcell,this%maps(i)%accum,this%maps(i)%nhits)
       this%maps(i)%size=n
       this%maps(i)%size_fill=nfill
       allocate(this%maps(i)%current(0:n),this%maps(i)%current_for_fillcell(0:nfill),&
            this%maps(i)%accum(0:nfill),this%maps(i)%nhits(nfill))
       read(unit,*,iostat=ios) this%maps(i)%current
       if (ios.ne.0) error stop 'staged integrator: invalid checkpoint grid'
       read(unit,*,iostat=ios) this%maps(i)%current_for_fillcell
       if (ios.ne.0) error stop 'staged integrator: invalid checkpoint fill grid'
       read(unit,*,iostat=ios) this%maps(i)%accum
       if (ios.ne.0) error stop 'staged integrator: invalid checkpoint grid accumulator'
       read(unit,*,iostat=ios) this%maps(i)%nhits
       if (ios.ne.0) error stop 'staged integrator: invalid checkpoint grid hits'
       if (any(.not.ieee_is_finite(this%maps(i)%current)).or.&
            any(this%maps(i)%current(1:n).le.this%maps(i)%current(0:n-1)).or.&
            this%maps(i)%current(0).ne.0d0.or.this%maps(i)%current(n).ne.1d0) &
            error stop 'staged integrator: non-monotone checkpoint grid'
       if (any(.not.ieee_is_finite(this%maps(i)%current_for_fillcell)).or.&
            any(this%maps(i)%current_for_fillcell(1:nfill).le.this%maps(i)%current_for_fillcell(0:nfill-1)).or.&
            this%maps(i)%current_for_fillcell(0).ne.0d0.or.this%maps(i)%current_for_fillcell(nfill).ne.1d0.or.&
            any(.not.ieee_is_finite(this%maps(i)%accum)).or.any(this%maps(i)%nhits.lt.0)) &
            error stop 'staged integrator: corrupt checkpoint adaptation grid'
    enddo
    allocate(this%candidate_weight(max(16,this%ncandidates)),this%candidate_priority(max(16,this%ncandidates)), &
         this%candidate_factor(max(16,this%ncandidates)))
    this%candidate_factor=1d0
    do i=1,this%ncandidates
       if (version.ge.3) then
          read(unit,*,iostat=ios) this%candidate_weight(i),this%candidate_priority(i),this%candidate_factor(i)
       else
          read(unit,*,iostat=ios) this%candidate_weight(i),this%candidate_priority(i)
       endif
       if (ios.ne.0) error stop 'staged integrator: invalid checkpoint candidate'
       if (.not.ieee_is_finite(this%candidate_weight(i)).or.this%candidate_weight(i).le.0d0.or.&
            .not.ieee_is_finite(this%candidate_priority(i)).or..not.ieee_is_finite(this%candidate_factor(i)).or. &
            this%candidate_factor(i).le.0d0) error stop 'staged integrator: corrupt checkpoint candidate'
    enddo
    read(unit,'(a)',iostat=ios) marker
    if (ios.ne.0) error stop 'staged integrator: truncated checkpoint'
    if (trim(marker).ne.'END_MG5_SIMPLE_INTEGRATOR') error stop 'staged integrator: invalid checkpoint trailer'
  end subroutine staged_load
end module simple_integrator_mod
