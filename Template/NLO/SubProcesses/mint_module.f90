!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! MINT Integrator Package
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! Original version by Paolo Nason (for POWHEG (BOX))
! Modified by Rikkert Frederix (for MadGraph5_aMC@NLO)
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!      subroutine mint(fun,ndim,ncalls0,itmax,imode,
! ndim=number of dimensions
! ncalls0=# of calls per iteration
! itmax =# of iterations
! fun(xx,www,ifirst): returns the function to be integrated multiplied by www;
!                     xx(1:ndim) are the variables of integration
!                     ifirst=0: normal behaviour
! imode: integer flag
!
! imode=-1:
! same as imode=0 as far as this routine is concerned, except for the
! fact that a grid is read at the beginning (rather than initialized).
! The return value of imode will be zero.
!
! imode=0:
! When called with imode=0 the routine integrates the absolute value of
! the function and sets up a grid xgrid(0:50,ndim) such that in each
! ndim-1 dimensional slice (i.e. xgrid(m-1,n)<xx(n)<xgrid(m,n)) the
! contribution of the integral is the same the array xgrid is setup at
! this stage; ans and err are the integral and its error
!
! imode=1 (in fact #0)
! When called with imode=1, the routine performs the integral of the
! function fun using the grid xgrid. If some number in the array ifold,
! (say, ifold(n)) is different from 1, it must be a divisor of 50, and
! the 50 intervals xgrid(0:50,n) are grouped into ifold(n) groups, each
! group containing 50/ifold(n) nearby intervals. For example, if
! ifold(1)=5, the 50 intervals for the first dimension are divided in 5
! groups of 10. The integral is then performed by folding on top of each
! other these 5 groups. Suppose, for example, that we choose a random
! point in xx(1) = xgrid(2,1)+x*(xgrid(3,1)-xgrid(2,1)), in the group of
! the first 5 interval.  we sum the contribution of this point to the
! contributions of points
! xgrid(2+m*10,1)+x*(xgrid(3+m*10,1)-xgrid(2+m*10,1)), with m=1,...,4.
! In the sequence of calls to the function fun, the call for the first
! point is performed with ifirst=0, and that for all subsequent points
! with ifirst=1, so that the function can avoid to compute quantities
! that only depend upon dimensions that have ifold=1, and do not change
! in each group of folded call. The values returned by fun in a sequence
! of folded calls with ifirst=0 and ifirst=1 are not used. The function
! itself must accumulate the values, and must return them when called
! with ifirst=2.
! 
! Added the posibility to keep track of more than one integral:
!
! nintegrals=1 : the function that is used to update the grids. This is
! the ABS cross section. If imode.eq.1, this does not contain the
! virtual corrections because for them a separate maximum is kept using (5).
! nintegrals=2 : the actual cross section. This includes virtual corrections.
! nintegrals=3 : the cross section from the M_Virt/M_Born ratio alone:
! this defines the average virtual that is added to each phase-space
! point
! nintegrals=4 : the cross section of the actual virtual minus the
! average virtual. This is used to determine the fraction of phase-space
! points for which we include the virtual.
! nintegrals=5 : abs of 3
! nintegrals=6 : born
! nintegrals>6 : virtual and born order by order
!

module mint_module
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use FKSParams ! contains use_poly_virtual
  implicit none
  integer, parameter, private :: nintervals=32    ! max number of intervals in the integration grids
  integer, parameter, public  :: ndimmax=60       ! max number of dimensions of the integral
  integer, parameter, public  :: n_ave_virt=10    ! max number of grids to set up to approx virtual
  integer, parameter, public  :: nintegrals=26    ! number of integrals to keep track of
  integer, parameter, private :: nintervals_virt=8! max number of intervals in the grids for the approx virtual
  integer, parameter, private :: min_inter=4      ! minimal number of intervals
  integer, parameter, private :: min_it0=4        ! minimal number of iterations in the mint step 0 phase
  integer, parameter, private :: min_it1=5        ! minimal number of iterations in the mint step 1 phase
  integer, parameter, public :: max_fold=512     ! 8*8*8 is max folding for the three variables
  integer, parameter, private :: max_points=100000! maximum number of points to trow per iteration if not enough non-zero points can be found.
  integer, parameter, public  :: maxchannels=20 ! set as least as large as in amcatnlo_run_interface
  integer, parameter, public :: born_spread_nxi=40,born_spread_ny=40
  integer, parameter, private :: born_spread_nbins=born_spread_nxi*born_spread_ny
  integer, parameter, private :: born_spread_train_points=800000
  integer, parameter, private :: born_spread_validation_points=200000
  ! Note that the number of intervals in the integration grids, 'nintervals', cannot be arbitrarily large.
  ! It should be equal to
  !     nintervals = min_inter * 2^n,
  ! where 'n' is an integer smaller than or equal to min(min_it0,min_it1).
  !
  ! The number of intergrals should be equal to
  !     nintegrals=6+2*n_ave_virt
  !

! public variables 
  integer, public :: ncalls0,ndim,itmax,imode,n_ord_virt,nchans,iconfig,ichan,ifold_energy,ifold_yij,ifold_phi
  integer, dimension(ndimmax), public :: ifold
  integer, dimension(maxchannels), public :: iconfigs
  double precision, public :: accuracy,min_virt_fraction_mint,wgt_mult
  double precision, dimension(0:n_ave_virt,maxchannels), public :: average_virtual
  double precision, dimension(0:n_ave_virt), public :: virt_wgt_mint,born_wgt_mint,polyfit
  double precision, dimension(maxchannels), public :: virtual_fraction
  double precision, dimension(nintegrals,0:maxchannels), public :: ans,unc
  logical, public :: born_spread_active=.false.,born_spread_ready=.false.
  logical, private :: born_spread_calibrating=.false.
  integer, public :: born_spread_phase=0,born_spread_current_bin=1
  integer, private :: born_spread_nexternal=0,born_spread_nincoming=0
  integer, private :: born_spread_nfks=0,born_spread_ndim=0
  integer, private :: born_spread_restart_ncalls=0
  integer, private :: born_spread_training_count=0
  integer, private :: born_spread_validation_count=0
  double precision, public :: born_spread_x=0d0,born_spread_y=0d0
  double precision, public :: born_spread_factor(born_spread_nxi,born_spread_ny)
  double precision, private :: born_spread_mean_delta=0d0
  double precision, private :: born_spread_m2_delta=0d0
  double precision, private :: born_spread_base_total=0d0
  double precision, private :: born_spread_spread_total=0d0
  double precision, private :: born_spread_point_base=0d0
  double precision, private :: born_spread_point_spread=0d0
  double precision, dimension(born_spread_nbins), private :: born_spread_deriv0=0d0
  integer, dimension(:), allocatable, private :: born_spread_break_bin
  double precision, dimension(:), allocatable, private :: born_spread_break
  double precision, dimension(:), allocatable, private :: born_spread_jump
  integer, private :: born_spread_nbreak=0,born_spread_capacity=0
  logical :: only_virt,new_point,pass_cuts_check

! private variables
  character(len=13), parameter, dimension(nintegrals), private :: title=(/ &
                                                   'ABS integral ', & !  1
                                                   'Integral     ', & !  2
                                                   'Virtual      ', & !  3
                                                   'Virtual ratio', & !  4
                                                   'ABS virtual  ', & !  5
                                                   'Born         ', & !  6
                                                   'V  1         ', & !  7
                                                   'B  1         ', & !  8
                                                   'V  2         ', & !  9
                                                   'B  2         ', & ! 10
                                                   'V  3         ', & ! 11
                                                   'B  3         ', & ! 12
                                                   'V  4         ', & ! 13
                                                   'B  4         ', & ! 14
                                                   'V  5         ', & ! 15
                                                   'B  5         ', & ! 16
                                                   'V  6         ', & ! 17
                                                   'B  6         ', & ! 18
                                                   'V  7         ', & ! 19
                                                   'B  7         ', & ! 20
                                                   'V  8         ', & ! 21
                                                   'B  8         ', & ! 22
                                                   'V  9         ', & ! 23
                                                   'B  9         ', & ! 24
                                                   'V 10         ', & ! 25
                                                   'B 10         '/)  ! 26


  integer, private :: nit,nit_included,kpoint_iter,nint_used,nint_used_virt,min_it,ncalls,pass_cuts_point,ng,npg,k
  integer, dimension(ndimmax), private :: icell,ncell
  integer, dimension(nintegrals), private :: non_zero_point,ntotcalls
  integer, dimension(nintervals,ndimmax,maxchannels), private :: nhits
  integer, dimension(maxchannels), private :: nhits_in_grids
  integer, dimension(nintervals_virt,ndimmax,0:n_ave_virt,maxchannels), private :: nvirt,nvirt_acc
  integer, dimension(13), private :: gen_counters
  logical, private :: double_events,reset,even_rn,firsttime
  logical, dimension(maxchannels), private :: regridded
  double precision, dimension(0:nintervals,ndimmax,maxchannels), private :: xgrid,xacc
  double precision, dimension(nintervals,ndimmax,maxchannels), private :: ymax,xmmm
  double precision, dimension(nintegrals,0:maxchannels), private :: vtot,etot,chi2
  double precision, dimension(nintegrals,3), private :: ans3,unc3
  double precision, dimension(nintegrals), private :: ans_l3,unc_l3,chi2_l3,f
  double precision, dimension(0:maxchannels), private :: ymax_virt,ans_chan
  double precision, dimension(2), private :: HwU_values
  double precision, dimension(nintervals_virt,ndimmax,0:n_ave_virt,maxchannels), private :: ave_virt,ave_virt_acc,ave_born_acc
  double precision, private :: upper_bound,vol_chan
  double precision, dimension(ndimmax), private :: rand
  double precision, dimension(0:nintervals,ndimmax) :: xgrid_new

! Common blocks used elsewhere in the code
  integer                                   npoints
  double precision            cross_section
  common /for_FixedOrder_lhe/ cross_section,npoints
  logical              fixed_order,nlo_ps
  common /c_fnlo_nlops/fixed_order,nlo_ps

! functions and subroutines:
  public :: mint,gen,read_grids_from_file,get_mint_wgt
  public :: born_spread_configure,born_spread_set_point
  public :: born_spread_observe_sample,born_spread_get_factor
  public :: born_spread_load_table,born_spread_write_table
  private :: initialise_mint,setup_basic_mint &
       &,update_accumulated_results,prepare_next_iteration &
       &,check_desired_accuracy,update_integration_grids &
       &,combine_final_three_iterations &
       &,print_results_accumulated_three_iterations &
       &,update_virtual_fraction,combine_iterations &
       &,print_results_accumulated,check_fractional_uncertainty &
       &,print_results_current_iteration &
       &,compute_fractional_uncertainty,combine_results_channels &
       &,check_for_special_channels_loop &
       &,combine_results_channels_special_loop,get_amount_of_points &
       &,add_point_to_grids,add_point_to_bounding_envelope &
       &,accumulate_the_point,compute_integrand,get_random_x &
       &,start_iteration,reset_accumulated_grids_for_updating &
       &,check_evenly_random_numbers,finalise_mint,write_results &
       &,write_channel_info,setup_imode_1 &
       &,reset_upper_bounding_envelope,setup_imode_m1,setup_imode_0 &
       &,reset_mint_grids,setup_common,write_grids_to_file &
       &,double_grid,regrid,smooth_xacc,nextlexi ,init_ave_virt&
       &,get_ave_virt,fill_ave_virt,regrid_ave_virt ,double_ave_virt&
       &,get_channel,close_run_zero_res,ran3 &
       &,initialize_even_random_numbers,get_ran &
       &,increase_gen_counters_middle,increase_gen_counters_before &
       &,increase_gen_counters_end,check_upper_bound &
       &,get_random_cell_flat,get_weighted_cell,initialise_mint_gen &
       &,print_gen_counters
contains

  subroutine mint(fun)
    implicit none
    integer kpoint
    double precision :: vol
    double precision, dimension(ndimmax) :: x
    integer, dimension(ndimmax) :: kfold
    double precision, external :: fun
    logical :: enough_points,channel_loop_done
    call initialise_mint
10  continue
    do while (nit.lt.itmax)
       call start_iteration
2      kpoint_iter=kpoint_iter+1
       do kpoint=1,ncalls
          new_point=.true.
          call get_random_x(x,vol,kfold)
          call compute_integrand(fun,x,vol)
          call accumulate_the_point(x)
       enddo
       call get_amount_of_points(enough_points)
       if (.not.enough_points) goto 2
       ! The special first-iteration loop requires the unit channel markers
       ! from reset_mint_grids. After calibration we retain the adapted
       ! channel weights, so use normal averaging on the restart instead.
       if (imode.eq.0 .and. nit.eq.1 .and. double_events .and. &
            .not.born_spread_ready) then
          call check_for_special_channels_loop(channel_loop_done)
          if (.not.channel_loop_done) goto 2
          call combine_results_channels_special_loop
       else
          call combine_results_channels
       endif
       call update_accumulated_results
    enddo
    if (imode.eq.0 .and. born_spread_active .and. &
         .not.born_spread_ready) then
       call calibrate_born_spreading(fun)
       call born_spread_write_table
       born_spread_ready=.true.
       born_spread_phase=3
       if (double_events) born_spread_restart_ncalls= &
            80*ndim*(nchans/3+1)
       ncalls0=born_spread_restart_ncalls
       call setup_common
       call reset_MC_grid
       call reset_accumulated_grids_for_updating
       if (even_rn) call initialize_even_random_numbers
       write(*,*) 'Restarting MINT with the normalized born-spreading table'
       goto 10
    endif
    call finalise_mint
  end subroutine mint

  subroutine calibrate_born_spreading(fun)
    implicit none
    double precision, external :: fun
    double precision :: x(ndimmax),vol,dummy
    integer :: kfold(ndimmax),ipoint
    logical :: old_even_rn
    born_spread_phase=1
    born_spread_calibrating=.true.
    born_spread_training_count=0
    born_spread_validation_count=0
    born_spread_mean_delta=0d0
    born_spread_m2_delta=0d0
    born_spread_base_total=0d0
    born_spread_spread_total=0d0
    born_spread_point_base=0d0
    born_spread_point_spread=0d0
    born_spread_deriv0=0d0
    born_spread_nbreak=0
    old_even_rn=even_rn
    even_rn=.false.
    write(*,*) 'Training born-spreading table with ', &
         born_spread_train_points,' independent phase-space points'
    do ipoint=1,born_spread_train_points
       new_point=.true.
       call get_random_x(x,vol,kfold)
       call compute_integrand(fun,x,vol)
    enddo
    call solve_born_spreading_table
    born_spread_phase=2
    write(*,*) 'Validating born-spreading table with ', &
         born_spread_validation_points,' independent phase-space points'
    do ipoint=1,born_spread_validation_points
       new_point=.true.
       call get_random_x(x,vol,kfold)
       call compute_integrand(fun,x,vol)
    enddo
    born_spread_calibrating=.false.
    even_rn=old_even_rn
    call finish_born_spreading_validation
    call deallocate_born_spread_lines
    if (allocated(born_spread_break_bin)) deallocate(born_spread_break_bin)
    if (allocated(born_spread_break)) deallocate(born_spread_break)
    if (allocated(born_spread_jump)) deallocate(born_spread_jump)
    born_spread_nbreak=0
    born_spread_capacity=0
  end subroutine calibrate_born_spreading

  subroutine solve_born_spreading_table
    implicit none
    integer, allocatable :: seg_bin(:)
    double precision, allocatable :: seg_slope(:),seg_length(:)
    double precision, dimension(born_spread_nbins) :: caps,bin_area
    double precision :: deriv,prev,turn,jump_total,length,remaining
    double precision :: group_slope,cap_total,low,high,mid,trial,total_add
    double precision :: max_factor,weighted_norm
    integer :: ibin,k,j,nseg,iseg,first,last,pass
    if (born_spread_nbreak.gt.1) &
         call born_spread_sort_breaks(1,born_spread_nbreak)
    allocate(seg_bin(born_spread_nbreak+born_spread_nbins))
    allocate(seg_slope(born_spread_nbreak+born_spread_nbins))
    allocate(seg_length(born_spread_nbreak+born_spread_nbins))
    nseg=0
    k=1
    do ibin=1,born_spread_nbins
       bin_area(ibin)=born_spread_bin_area(ibin)
       max_factor=1d0/bin_area(ibin)
       deriv=born_spread_deriv0(ibin)
       prev=0d0
       do while (k.le.born_spread_nbreak)
          if (born_spread_break_bin(k).ne.ibin) exit
          turn=born_spread_break(k)
          jump_total=0d0
          j=k
          do while (j.le.born_spread_nbreak)
             if (born_spread_break_bin(j).ne.ibin) exit
             if (.not.born_spread_same_break(turn, &
                  born_spread_break(j))) exit
             jump_total=jump_total+born_spread_jump(j)
             j=j+1
          enddo
          if (turn.gt.prev.and.prev.lt.max_factor) then
             length=min(turn-prev,max_factor-prev)
             if (length.gt.0d0) then
                nseg=nseg+1
                seg_bin(nseg)=ibin
                seg_slope(nseg)=deriv/bin_area(ibin)
                seg_length(nseg)=length
             endif
          endif
          deriv=deriv+jump_total
          prev=max(prev,turn)
          k=j
       enddo
       if (prev.lt.max_factor) then
          nseg=nseg+1
          seg_bin(nseg)=ibin
          seg_slope(nseg)=deriv/bin_area(ibin)
          seg_length(nseg)=max_factor-prev
       endif
    enddo
    if (nseg.le.0) then
       born_spread_factor=1d0
       deallocate(seg_bin,seg_slope,seg_length)
       return
    endif
    if (nseg.gt.1) call born_spread_sort_segments( &
         seg_slope,seg_bin,seg_length,1,nseg)
    born_spread_factor=0d0
    remaining=1d0
    iseg=1
    do while (iseg.le.nseg.and.remaining.gt.1d-12)
       first=iseg
       group_slope=seg_slope(iseg)
       do while (iseg.le.nseg)
          if (.not.born_spread_same_slope(group_slope, &
               seg_slope(iseg))) exit
          iseg=iseg+1
       enddo
       last=iseg-1
       caps=0d0
       do j=first,last
          caps(seg_bin(j))=caps(seg_bin(j))+seg_length(j)
       enddo
       cap_total=sum(caps*bin_area)
       if (remaining.ge.cap_total-1d-12) then
          do ibin=1,born_spread_nbins
             if (caps(ibin).gt.0d0) then
                j=(ibin-1)/born_spread_nxi+1
                k=ibin-(j-1)*born_spread_nxi
                born_spread_factor(k,j)=born_spread_factor(k,j)+caps(ibin)
             endif
          enddo
          remaining=max(0d0,remaining-cap_total)
       else
          low=-maxval(1d0/bin_area)
          high=maxval(1d0/bin_area)
          do pass=1,100
             mid=(low+high)/2d0
             total_add=0d0
             do ibin=1,born_spread_nbins
                if (caps(ibin).le.0d0) cycle
                j=(ibin-1)/born_spread_nxi+1
                k=ibin-(j-1)*born_spread_nxi
                trial=max(born_spread_factor(k,j), &
                     min(born_spread_factor(k,j)+caps(ibin),1d0+mid))
                total_add=total_add+bin_area(ibin)* &
                     (trial-born_spread_factor(k,j))
             enddo
             if (total_add.lt.remaining) then
                low=mid
             else
                high=mid
             endif
          enddo
          mid=(low+high)/2d0
          do ibin=1,born_spread_nbins
             if (caps(ibin).le.0d0) cycle
             j=(ibin-1)/born_spread_nxi+1
             k=ibin-(j-1)*born_spread_nxi
             trial=max(born_spread_factor(k,j), &
                  min(born_spread_factor(k,j)+caps(ibin),1d0+mid))
             born_spread_factor(k,j)=trial
          enddo
          remaining=0d0
       endif
    enddo
    if (remaining.gt.1d-7) then
       write(*,*) 'ERROR: born-spreading optimizer could not satisfy', &
            ' its normalization constraint',remaining
       stop 1
    endif
    weighted_norm=0d0
    do ibin=1,born_spread_nbins
       j=(ibin-1)/born_spread_nxi+1
       k=ibin-(j-1)*born_spread_nxi
       weighted_norm=weighted_norm+bin_area(ibin)* &
            born_spread_factor(k,j)
    enddo
    if (weighted_norm.le.0d0.or.abs(weighted_norm-1d0).gt.1d-7) then
       write(*,*) 'ERROR: born-spreading optimizer produced invalid ', &
            'normalization',weighted_norm
       stop 1
    endif
    born_spread_factor=born_spread_factor/weighted_norm
    deallocate(seg_bin,seg_slope,seg_length)
  end subroutine solve_born_spreading_table

  double precision function born_spread_bin_area(ibin)
    implicit none
    integer, intent(in) :: ibin
    integer :: ix,iy
    double precision :: xlow,xhigh,ylow,yhigh
    iy=(ibin-1)/born_spread_nxi+1
    ix=ibin-(iy-1)*born_spread_nxi
    xlow=sqrt(dble(ix-1)/born_spread_nxi)
    xhigh=sqrt(dble(ix)/born_spread_nxi)
    ylow=sqrt(dble(iy-1)/born_spread_ny)
    yhigh=sqrt(dble(iy)/born_spread_ny)
    born_spread_bin_area=(xhigh-xlow)*(yhigh-ylow)
  end function born_spread_bin_area

  subroutine born_spread_solver_self_test
    implicit none
    integer :: ibin,ix,iy
    double precision :: expected,bval(1),cval(1),area
    born_spread_deriv0=0d0
    born_spread_nbreak=0
    born_spread_phase=1
    do ibin=1,born_spread_nbins
       born_spread_current_bin=ibin
       area=born_spread_bin_area(ibin)
       ix=mod(ibin-1,born_spread_nxi)+1
       bval(1)=area
       cval(1)=-2d0*area
       if (ix.gt.born_spread_nxi/4) then
          bval(1)=-area
          cval(1)=-area
       endif
       call born_spread_observe_sample(bval,cval,1)
    enddo
    born_spread_phase=0
    call solve_born_spreading_table
    do ibin=1,born_spread_nbins
       iy=(ibin-1)/born_spread_nxi+1
       ix=ibin-(iy-1)*born_spread_nxi
       expected=0d0
       if (ix.le.born_spread_nxi/4) expected=2d0
       if (abs(born_spread_factor(ix,iy)-expected).gt.1d-10) then
          write(*,*) 'ERROR: born-spreading optimizer self-test failed'
          stop 1
       endif
    enddo
    born_spread_deriv0=0d0
    born_spread_nbreak=0
    born_spread_training_count=0
    call solve_born_spreading_table
    if (maxval(abs(born_spread_factor-1d0)).gt.1d-10) then
       write(*,*) 'ERROR: born-spreading tie-break self-test failed'
       stop 1
    endif
    born_spread_factor=1d0
    born_spread_deriv0=0d0
    born_spread_current_bin=1
    born_spread_nbreak=0
    born_spread_capacity=0
    if (allocated(born_spread_break_bin)) deallocate(born_spread_break_bin)
    if (allocated(born_spread_break)) deallocate(born_spread_break)
    if (allocated(born_spread_jump)) deallocate(born_spread_jump)
  end subroutine born_spread_solver_self_test

  subroutine finish_born_spreading_validation
    implicit none
    double precision :: validation_error,improvement,required
    validation_error=0d0
    if (born_spread_validation_count.gt.1) &
         validation_error=sqrt(max(0d0,born_spread_m2_delta/ &
         (born_spread_validation_count-1)/born_spread_validation_count))
    improvement=born_spread_base_total-born_spread_spread_total
    required=2d0*validation_error*born_spread_validation_count
    if (born_spread_validation_count.lt.2.or. &
         born_spread_base_total.le.0d0.or.improvement.le.required) then
       born_spread_factor=1d0
       write(*,*) 'Born spreading validation found no significant reduction; ', &
            'using the normalized unit table'
    else
       write(*,*) 'Born spreading validation reduced sampled negative mass by ', &
            improvement,' +/- ',validation_error*born_spread_validation_count
    endif
  end subroutine finish_born_spreading_validation

  recursive subroutine born_spread_sort_breaks(left,right)
    implicit none
    integer, intent(in) :: left,right
    integer :: i,j,pivot_bin,temp_bin
    double precision :: pivot_turn,temp_turn,temp_jump
    i=left
    j=right
    pivot_bin=born_spread_break_bin((left+right)/2)
    pivot_turn=born_spread_break((left+right)/2)
    do
       do while (born_spread_break_bin(i).lt.pivot_bin.or. &
            (born_spread_break_bin(i).eq.pivot_bin.and. &
            born_spread_break(i).lt.pivot_turn))
          i=i+1
       enddo
       do while (born_spread_break_bin(j).gt.pivot_bin.or. &
            (born_spread_break_bin(j).eq.pivot_bin.and. &
            born_spread_break(j).gt.pivot_turn))
          j=j-1
       enddo
       if (i.le.j) then
          temp_bin=born_spread_break_bin(i)
          born_spread_break_bin(i)=born_spread_break_bin(j)
          born_spread_break_bin(j)=temp_bin
          temp_turn=born_spread_break(i)
          born_spread_break(i)=born_spread_break(j)
          born_spread_break(j)=temp_turn
          temp_jump=born_spread_jump(i)
          born_spread_jump(i)=born_spread_jump(j)
          born_spread_jump(j)=temp_jump
          i=i+1
          j=j-1
       endif
       if (i.gt.j) exit
    enddo
    if (left.lt.j) call born_spread_sort_breaks(left,j)
    if (i.lt.right) call born_spread_sort_breaks(i,right)
  end subroutine born_spread_sort_breaks

  recursive subroutine born_spread_sort_segments(slopes,bins,lengths,left,right)
    implicit none
    integer, intent(in) :: left,right
    integer, intent(inout) :: bins(:)
    double precision, intent(inout) :: slopes(:),lengths(:)
    integer :: i,j,pivot_bin,temp_bin
    double precision :: pivot,temp_slope,temp_length
    i=left
    j=right
    pivot=slopes((left+right)/2)
    do
       do while (slopes(i).lt.pivot)
          i=i+1
       enddo
       do while (slopes(j).gt.pivot)
          j=j-1
       enddo
       if (i.le.j) then
          temp_slope=slopes(i)
          slopes(i)=slopes(j)
          slopes(j)=temp_slope
          temp_length=lengths(i)
          lengths(i)=lengths(j)
          lengths(j)=temp_length
          temp_bin=bins(i)
          bins(i)=bins(j)
          bins(j)=temp_bin
          i=i+1
          j=j-1
       endif
       if (i.gt.j) exit
    enddo
    if (left.lt.j) call born_spread_sort_segments( &
         slopes,bins,lengths,left,j)
    if (i.lt.right) call born_spread_sort_segments( &
         slopes,bins,lengths,i,right)
  end subroutine born_spread_sort_segments

  logical function born_spread_same_break(a,b)
    implicit none
    double precision, intent(in) :: a,b
    born_spread_same_break=(a.eq.b)
  end function born_spread_same_break

  logical function born_spread_same_slope(a,b)
    implicit none
    double precision, intent(in) :: a,b
    double precision :: scale
    scale=max(abs(a),abs(b),1d-300)
    born_spread_same_slope=abs(a-b).le.1d-13*scale
  end function born_spread_same_slope

  subroutine born_spread_configure(enabled,nexternal,nincoming,nfks,ndim_in)
    implicit none
    logical, intent(in) :: enabled
    integer, intent(in) :: nexternal,nincoming,nfks,ndim_in
    born_spread_active=enabled
    born_spread_ready=.false.
    born_spread_calibrating=.false.
    born_spread_phase=0
    born_spread_factor=1d0
    born_spread_x=0d0
    born_spread_y=0d0
    born_spread_current_bin=1
    born_spread_nexternal=nexternal
    born_spread_nincoming=nincoming
    born_spread_nfks=nfks
    born_spread_ndim=ndim_in
    born_spread_restart_ncalls=0
    born_spread_training_count=0
    born_spread_validation_count=0
    born_spread_mean_delta=0d0
    born_spread_m2_delta=0d0
    born_spread_base_total=0d0
    born_spread_spread_total=0d0
    born_spread_point_base=0d0
    born_spread_point_spread=0d0
    born_spread_deriv0=0d0
    born_spread_nbreak=0
    born_spread_capacity=0
    if (allocated(born_spread_break_bin)) deallocate(born_spread_break_bin)
    if (allocated(born_spread_break)) deallocate(born_spread_break)
    if (allocated(born_spread_jump)) deallocate(born_spread_jump)
    if (enabled) call born_spread_solver_self_test
  end subroutine born_spread_configure

  subroutine born_spread_set_point(x,y)
    implicit none
    double precision, intent(in) :: x,y
    integer :: ix,iy
    born_spread_x=max(0d0,min(x,1d0))
    born_spread_y=max(0d0,min(y,1d0))
    ix=min(int(born_spread_x*born_spread_nxi)+1,born_spread_nxi)
    iy=min(int(born_spread_y*born_spread_ny)+1,born_spread_ny)
    born_spread_current_bin=(iy-1)*born_spread_nxi+ix
  end subroutine born_spread_set_point

  double precision function born_spread_get_factor()
    implicit none
    integer :: ix,iy
    iy=(born_spread_current_bin-1)/born_spread_nxi+1
    ix=born_spread_current_bin-(iy-1)*born_spread_nxi
    born_spread_get_factor=born_spread_factor(ix,iy)
  end function born_spread_get_factor

  subroutine born_spread_observe_sample(bvals,cvals,nvals,point_complete)
    implicit none
    integer, intent(in) :: nvals
    double precision, intent(in) :: bvals(*),cvals(*)
    logical, intent(in), optional :: point_complete
    integer :: j,ibin
    double precision :: b,c,turn,base_neg,spread_neg,delta,old_mean
    if (born_spread_phase.eq.1) then
       if (.not.present(point_complete)) then
          born_spread_training_count=born_spread_training_count+1
       elseif (point_complete) then
          born_spread_training_count=born_spread_training_count+1
       endif
       ibin=born_spread_current_bin
       do j=1,nvals
          b=bvals(j)
          c=cvals(j)
          if (.not.ieee_is_finite(b) .or. .not.ieee_is_finite(c)) cycle
          if (b.eq.0d0) cycle
          if (c.lt.0d0.or.(c.eq.0d0.and.b.lt.0d0)) &
               born_spread_deriv0(ibin)=born_spread_deriv0(ibin)-b
          turn=-c/b
          if (.not.ieee_is_finite(turn)) cycle
          if (turn.eq.0d0.and.c.lt.0d0.and.b.gt.0d0) &
               born_spread_deriv0(ibin)=born_spread_deriv0(ibin)+b
          if (turn.gt.0d0.and.turn.lt. &
               1d0/born_spread_bin_area(ibin)) &
               call born_spread_append(ibin,turn,abs(b))
       enddo
    elseif (born_spread_phase.eq.2) then
       base_neg=0d0
       spread_neg=0d0
       do j=1,nvals
          b=bvals(j)
          c=cvals(j)
          if (.not.ieee_is_finite(b) .or. .not.ieee_is_finite(c)) cycle
          base_neg=base_neg+max(0d0,-(b+c))
          spread_neg=spread_neg+max(0d0,-(b*born_spread_get_factor()+c))
       enddo
       born_spread_point_base=born_spread_point_base+base_neg
       born_spread_point_spread=born_spread_point_spread+spread_neg
       if (present(point_complete)) then
          if (point_complete) then
             born_spread_validation_count=born_spread_validation_count+1
             delta=born_spread_point_spread-born_spread_point_base
             old_mean=born_spread_mean_delta
             born_spread_mean_delta=old_mean+ &
                  (delta-old_mean)/born_spread_validation_count
             born_spread_m2_delta=born_spread_m2_delta+ &
                  (delta-old_mean)*(delta-born_spread_mean_delta)
             born_spread_base_total=born_spread_base_total+ &
                  born_spread_point_base
             born_spread_spread_total=born_spread_spread_total+ &
                  born_spread_point_spread
             born_spread_point_base=0d0
             born_spread_point_spread=0d0
          endif
       else
! Preserve the standalone observer contract for single-fold callers.
          born_spread_validation_count=born_spread_validation_count+1
          delta=born_spread_point_spread-born_spread_point_base
          old_mean=born_spread_mean_delta
          born_spread_mean_delta=old_mean+ &
               (delta-old_mean)/born_spread_validation_count
          born_spread_m2_delta=born_spread_m2_delta+ &
               (delta-old_mean)*(delta-born_spread_mean_delta)
          born_spread_base_total=born_spread_base_total+ &
               born_spread_point_base
          born_spread_spread_total=born_spread_spread_total+ &
               born_spread_point_spread
          born_spread_point_base=0d0
          born_spread_point_spread=0d0
       endif
    endif
  end subroutine born_spread_observe_sample

  subroutine born_spread_append(ibin,turn,jump)
    implicit none
    integer, intent(in) :: ibin
    double precision, intent(in) :: turn,jump
    integer, allocatable :: itemp(:)
    double precision, allocatable :: dtemp(:)
    integer :: new_capacity
    if (born_spread_nbreak.eq.born_spread_capacity) then
       new_capacity=max(1024,2*born_spread_capacity)
       allocate(itemp(new_capacity))
       if (born_spread_nbreak.gt.0) &
            itemp(1:born_spread_nbreak)=born_spread_break_bin
       call move_alloc(itemp,born_spread_break_bin)
       allocate(dtemp(new_capacity))
       if (born_spread_nbreak.gt.0) &
            dtemp(1:born_spread_nbreak)=born_spread_break
       call move_alloc(dtemp,born_spread_break)
       allocate(dtemp(new_capacity))
       if (born_spread_nbreak.gt.0) &
            dtemp(1:born_spread_nbreak)=born_spread_jump
       call move_alloc(dtemp,born_spread_jump)
       born_spread_capacity=new_capacity
    endif
    born_spread_nbreak=born_spread_nbreak+1
    born_spread_break_bin(born_spread_nbreak)=ibin
    born_spread_break(born_spread_nbreak)=turn
    born_spread_jump(born_spread_nbreak)=jump
  end subroutine born_spread_append

  subroutine born_spread_load_table
    implicit none
    integer :: ios,version,nxi,ny,nexternal,nincoming,nfks,ndim_in
    integer :: i,j
    character(len=32) :: tag
    integer, parameter :: lun=71
    open(unit=lun,file='born_spreading.dat',status='old',iostat=ios)
    if (ios.ne.0) call born_spread_table_error( &
         'Cannot open born_spreading.dat; rerun integration step 0 or set born_spreading = False')
    read(lun,*,iostat=ios) tag,version
    if (ios.ne.0) &
         call born_spread_table_error('Unsupported born_spreading.dat format')
    if (trim(tag).ne.'BORN_SPREAD'.or.version.ne.1) &
         call born_spread_table_error('Unsupported born_spreading.dat format')
    read(lun,*,iostat=ios) nxi,ny,nexternal,nincoming,nfks,ndim_in
    if (ios.ne.0) call born_spread_table_error('Corrupt born_spreading.dat signature')
    if (nxi.ne.born_spread_nxi.or.ny.ne.born_spread_ny.or. &
         nexternal.ne.born_spread_nexternal.or. &
         nincoming.ne.born_spread_nincoming.or. &
         nfks.ne.born_spread_nfks.or.ndim_in.ne.born_spread_ndim) &
         call born_spread_table_error( &
         'born_spreading.dat belongs to an incompatible process; rerun step 0')
    read(lun,*,iostat=ios) born_spread_training_count, &
         born_spread_validation_count
    if (ios.ne.0) call born_spread_table_error('Corrupt born_spreading.dat statistics')
    read(lun,*,iostat=ios) born_spread_base_total, &
         born_spread_spread_total,born_spread_mean_delta,born_spread_m2_delta
    if (ios.ne.0) call born_spread_table_error('Corrupt born_spreading.dat statistics')
    do j=1,born_spread_ny
       read(lun,*,iostat=ios) (born_spread_factor(i,j),i=1,born_spread_nxi)
       if (ios.ne.0) call born_spread_table_error('Corrupt born_spreading.dat table')
    enddo
    close(lun)
    if (any(.not.ieee_is_finite(born_spread_factor)).or. &
         any(born_spread_factor.lt.0d0)) &
         call born_spread_table_error('born_spreading.dat contains invalid factors')
    if (abs(born_spread_normalization()-1d0).gt.1d-7) &
         call born_spread_table_error('born_spreading.dat violates the normalization constraint')
    born_spread_ready=.true.
    born_spread_phase=3
    write(*,*) 'Loaded normalized born-spreading table from born_spreading.dat'
  end subroutine born_spread_load_table

  double precision function born_spread_normalization()
    implicit none
    integer :: ix,iy,ibin
    born_spread_normalization=0d0
    do iy=1,born_spread_ny
       do ix=1,born_spread_nxi
          ibin=(iy-1)*born_spread_nxi+ix
          born_spread_normalization=born_spread_normalization+ &
               born_spread_bin_area(ibin)*born_spread_factor(ix,iy)
       enddo
    enddo
  end function born_spread_normalization

  subroutine born_spread_write_table
    implicit none
    integer :: ios,i,j
    integer, parameter :: lun=72
    open(unit=lun,file='born_spreading.dat',status='replace',iostat=ios)
    if (ios.ne.0) then
       write(*,*) 'ERROR: cannot write born_spreading.dat'
       stop 1
    endif
    write(lun,*) 'BORN_SPREAD',1
    write(lun,*) born_spread_nxi,born_spread_ny,born_spread_nexternal, &
         born_spread_nincoming,born_spread_nfks,born_spread_ndim
    write(lun,*) born_spread_training_count,born_spread_validation_count
    write(lun,'(4(1x,es24.16))') born_spread_base_total, &
         born_spread_spread_total,born_spread_mean_delta,born_spread_m2_delta
    do j=1,born_spread_ny
       write(lun,'(40(1x,es24.16))') &
            (born_spread_factor(i,j),i=1,born_spread_nxi)
    enddo
    close(lun)
  end subroutine born_spread_write_table

  subroutine born_spread_table_error(message)
    implicit none
    character(len=*), intent(in) :: message
    write(*,*) 'ERROR: ',trim(message)
    stop 1
  end subroutine born_spread_table_error

  subroutine initialise_mint
    implicit none
    if (imode.ne.0) call read_grids_from_file
    call setup_basic_mint
    if (imode.eq.0) then
       call setup_imode_0
    elseif (imode.eq.-1) then
       call setup_imode_m1
    elseif (imode.eq.1) then
       call setup_imode_1
    endif
    cross_section=ans_chan(0) * wgt_mult
    call setup_common
  end subroutine initialise_mint

  subroutine setup_basic_mint
    implicit none
    ! if ncalls0 is greater than 0, use the default running, i.e. do not
    ! double the events after each iteration as well as use a fixed number
    ! of intervals in the grids.
    if (ncalls0.gt.0) then
       double_events=.false.
       nint_used=nintervals
       nint_used_virt=nintervals_virt
    else
       ! if ncalls0.le.0, reset it and double the events per iteration
       ncalls0=80*ndim*(nchans/3+1)
       double_events=.true.
       if (imode.eq.1 .or. imode.eq.-1) then
          nint_used=nintervals
          nint_used_virt=nintervals_virt
       else
          nint_used=min_inter
          nint_used_virt=min_inter
       endif
    endif
    reset=.false.
    ncalls=0  ! # PS points (updated below)
    if (imode.eq.0.and.born_spread_active.and. &
         born_spread_restart_ncalls.eq.0) &
         born_spread_restart_ncalls=ncalls0
  end subroutine setup_basic_mint

  subroutine update_accumulated_results
    implicit none
    double precision, dimension(nintegrals) :: efrac
    logical :: iterations_done
    call compute_fractional_uncertainty(efrac)
    call print_results_current_iteration(efrac)
    call check_fractional_uncertainty(efrac)
    if (reset) return ! iteration was not accurate enough: do not include it
    call combine_iterations
    call combine_final_three_iterations
    if (fixed_order) then
       call accum(.true.)
       call HwU_accum_iter(.true.,ntotcalls(1),HwU_values)
    endif
    if (imode.eq.0) then
       call update_virtual_fraction
       call update_integration_grids
    endif
    call check_desired_accuracy(iterations_done)
    if (.not. iterations_done) then
       call prepare_next_iteration
    else
       nit=itmax
    endif
  end subroutine update_accumulated_results

  subroutine prepare_next_iteration
    implicit none
    integer :: kchan,kdim,k_ord_virt
    if (double_events) then
! Double the number of intervals in the grids if not yet reach the maximum
       if (2*nint_used.le.nintervals) then
          do kchan=1,nchans
             do kdim=1,ndim
                call double_grid(kdim,kchan)
             enddo
          enddo
          nint_used=2*nint_used
       endif
       if (2*nint_used_virt.le.nintervals_virt) then
          do k_ord_virt=0,n_ord_virt
             call double_ave_virt(k_ord_virt)
          enddo
          nint_used_virt=2*nint_used_virt
       endif
! double the number of points for the next iteration
       ncalls0=ncalls0*2
    endif
  end subroutine prepare_next_iteration
  
  subroutine check_desired_accuracy(iterations_done)
    implicit none
    logical :: iterations_done
    integer :: i
! Quit if the desired accuracy has been reached
    iterations_done=.false.
    if (nit_included.ge.min_it .and. accuracy.gt.0d0) then
       if (unc(1,0)/ans(1,0)*max(1d0,chi2(1,0)/dble(nit_included-1)).lt.accuracy) then
          write (*,*) 'Found desired accuracy'
          iterations_done=.true.
       elseif(unc_l3(1)/ans_l3(1)*max(1d0,chi2_l3(1)).lt.accuracy) then
          write (*,*) 'Found desired accuracy in last 3 iterations'
          iterations_done=.true.
          ! overwrite results with the results from the last three iterations
          do i=1,nintegrals
             ans(i,0)=ans_l3(i)
             unc(i,0)=unc_l3(i)
             chi2(i,0)=chi2_l3(i)*dble(nit_included-1)
          enddo
       endif
    endif
  end subroutine check_desired_accuracy
  
  subroutine update_integration_grids
    implicit none
    integer :: kchan,kdim,k_ord_virt
    do kchan=1,nchans
       do kdim=1,ndim
          call regrid(kdim,kchan)
       enddo
       ! overwrite xgrid with the new xgrid
       if (regridded(kchan)) xgrid(1:nint_used,1:ndim,kchan)=xgrid_new(1:nint_used,1:ndim)
    enddo
    if (use_poly_virtual) then
       call do_polyfit()
    else
       do k_ord_virt=0,n_ord_virt
          call regrid_ave_virt(k_ord_virt)
       enddo
    endif
! Regrid the MC over integers (used for the MC over FKS dirs)
    call regrid_MC_integer
  end subroutine update_integration_grids

  
  subroutine combine_final_three_iterations
    implicit none
    integer :: i,j
! Update the results of the last tree iterations
    do j=1,2
       ans3(1:nintegrals,j)=ans3(1:nintegrals,j+1)
       unc3(1:nintegrals,j)=unc3(1:nintegrals,j+1)
    enddo
    ans3(1:nintegrals,3)=vtot(1:nintegrals,0)
    unc3(1:nintegrals,3)=etot(1:nintegrals,0)
! Compute the results of the last three iterations
    if (nit_included.ge.4) then
       do i=1,nintegrals
          ans_l3(i)=0d0
          unc_l3(i)=ans3(i,1)*1d99
          chi2_l3(i)=0d0
          do j=1,3 ! the three final iterations
             if (i.ne.1 .and. (unc_l3(i).eq.0d0 .or. unc3(i,j).eq.0d0)) then
                continue ! do not do anything
             else
                ans_l3(i)=(ans_l3(i)/unc_l3(i)+ans3(i,j)/unc3(i,j))/(1d0/unc_l3(i)+1d0/unc3(i,j))
                unc_l3(i)=1d0/sqrt(1d0/unc_l3(i)**2+1/unc3(i,j)**2)
                chi2_l3(i)=chi2_l3(i)+(ans3(i,j)-ans_l3(i))**2/unc3(i,j)**2
             endif
          enddo
          chi2_l3(i)=chi2_l3(i)/2d0 ! three iterations, so 2 degrees of freedom
       enddo
       call print_results_accumulated_three_iterations
    endif
  end subroutine combine_final_three_iterations


  subroutine print_results_accumulated_three_iterations
    implicit none
    integer :: i
    double precision, dimension(nintegrals) :: efrac
    do i=1,2
       if (ans_l3(i).ne.0d0) then
          efrac(i)=abs(unc_l3(i)/ans_l3(i))
       else
          efrac(i)=0d0
       endif
       if (ans_l3(i).ne.0d0 .and. unc_l3(i).ne.0d0) then
          write(*,'(a,1x,e10.4,1x,a,1x,e10.4,1x,a,1x,f7.3,1x,a)')  &
               'accumulated results last 3 iterations '//title(i)//' =' , &
               ans_l3(i),' +/- ',unc_l3(i) ,' (',efrac(i)*100d0 ,'%)'
       endif
    enddo
    write(*,'(a,1x,e10.4)') 'accumulated result last 3 iterrations Chi^2 per DoF =' &
         ,chi2_l3(1)
  end subroutine print_results_accumulated_three_iterations


  
  subroutine update_virtual_fraction
! Update the fraction of the events for which we include the virtual corrections
! in the calculation
    implicit none
    integer kchan,k_ord_virt
    double precision :: error_virt
    do kchan=1,nchans
       error_virt=0d0
       do k_ord_virt=1,n_ord_virt
          error_virt=error_virt+etot(2*k_ord_virt+5,kchan)**2
       enddo
       error_virt=sqrt(error_virt)
       virtual_fraction(kchan)=max(min(virtual_fraction(kchan) &
            *max(min(2d0*error_virt/etot(1,kchan),2d0),0.25d0),1d0) &
            ,Min_virt_fraction_mint)
    enddo
  end subroutine update_virtual_fraction

  
  subroutine combine_iterations
    implicit none
    integer i,kchan
    HwU_values(1)=etot(1,0)
    HwU_values(2)=unc(1,0)
    if(nit.eq.1) then ! first iteration
       ans(1:nintegrals,0:nchans)=vtot(1:nintegrals,0:nchans)
       unc(1:nintegrals,0:nchans)=etot(1:nintegrals,0:nchans)
       ans_chan(0:nchans)=ans(1,0:nchans)
       write (*,'(a,1x,e10.4)') 'Chi^2 per d.o.f.',0d0
    else
       do kchan=nchans,0,-1 ! go backwards so that kchan=0 goes last
                            ! (this makes sure central value is correctly updated).
          do i=1,nintegrals
             if (i.ne.1 .and. (etot(i,0).eq.0d0 .or. unc(i,0).eq.0d0)) then
                continue ! do not do anything
             else
                ans(i,kchan)=(ans(i,kchan)/unc(i,0)+vtot(i,kchan)/etot(i,0))/(1d0/unc(i,0)+1d0/etot(i,0))
                unc(i,kchan)=1d0/sqrt(1d0/unc(i,kchan)**2+1d0/etot(i,kchan)**2)
                chi2(i,kchan)=chi2(i,kchan)+(vtot(i,kchan)-ans(i,kchan))**2/etot(i,kchan)**2
             endif
          enddo
          ans_chan(kchan)=ans(1,kchan)
       enddo
       write (*,'(a,1x,e10.4)') 'Chi^2=',(vtot(1,0)-ans(1,0))**2/etot(1,0)**2
    endif
    nit_included=nit_included+1
    call print_results_accumulated
    cross_section=ans(1,0)
  end subroutine combine_iterations

  subroutine print_results_accumulated
    implicit none
    integer i
    double precision, dimension(nintegrals) :: efrac
    do i=1,nintegrals
       if (ans(i,0).ne.0d0) then
          efrac(i)=abs(unc(i,0)/ans(i,0))
       else
          efrac(i)=0d0
       endif
       if (ans(i,0).ne.0d0 .and. unc(i,0).ne.0d0) then
          write(*,'(a,1x,e10.4,1x,a,1x,e10.4,1x,a,1x,f7.3,1x,a)')  &
               'accumulated results '//title(i)//' =',ans(i,0),' +/- ',unc(i,0) ,' (',efrac(i)*100d0,'%)'
       endif
    enddo
    if (nit_included.le.1) then
       write (*,'(a,1x,e10.4)') 'accumulated result Chi^2 per DoF =',0d0
    else
       write (*,'(a,1x,e10.4)') 'accumulated result Chi^2 per DoF =',chi2(1,0)/dble(nit_included-1)
    endif
  end subroutine print_results_accumulated

 
  subroutine check_fractional_uncertainty(efrac)
    implicit none
    double precision, dimension(nintegrals) :: efrac
    logical, save :: bad_iteration=.false.
    integer iappl
    common /for_applgrid/ iappl
! If there was a large fluctation in this iteration, be careful with
! including it in the accumalated results and plots.
    if (efrac(1).gt.0.3d0 .and. iappl.eq.0 .and. nit.gt.3) then
! Do not include the results in the plots
       if (fixed_order) call accum(.false.)
       if (fixed_order) call HwU_accum_iter(.false.,ntotcalls(1),HwU_values)
! Do not include the results in the updating of the grids.
       write (*,*) 'Large fluctuation ( >30 % ). Not including iteration in results.'
! empty the accumulated results in the MC over integers
       call empty_MC_integer
! empty the accumulated results for the MINT grids (Cannot really
! skip the increase of the upper bounding envelope. So, simply
! continue here. Note that no matter how large the integrand for the
! PS point, the upper bounding envelope is at most increased by a
! factor 2, so this should be fine).
       reset=.true.
! double the number of points for the next iteration
       if (double_events) ncalls0=ncalls0*2
       if (bad_iteration .and. imode.eq.0 .and. double_events) then
! 2nd bad iteration is a row. Reset grids
          write (*,*)'2nd bad iteration in a row. Resetting grids and starting from scratch...'
          if (double_events) then
             if (imode.eq.0) nint_used=min_inter ! reset number of intervals
             ncalls0=ncalls0/8   ! Start with larger number
          endif
          call reset_mint_grids
          call reset_MC_grid  ! reset the grid for the integers
          if (fixed_order) call initplot  ! Also reset all the plots
          call setup_common
          bad_iteration=.false.
       else
          bad_iteration=.true.
       endif
    else
       bad_iteration=.false.
    endif
  end subroutine check_fractional_uncertainty

  

  subroutine print_results_current_iteration(efrac)
    implicit none
    integer :: i
    double precision, dimension(nintegrals) :: efrac
    do i=1,nintegrals
       if (vtot(i,0).ne.0d0 .and. etot(i,0).ne.0d0) then
          write(*,'(a,1x,e10.4,1x,a,1x,e10.4,1x,a,1x,f7.3,1x,a)') &
               title(i)//' =',vtot(i,0),' +/- ',etot(i,0),' (',efrac(i)*100d0 ,'%)'
       endif
    enddo
  end subroutine print_results_current_iteration

  subroutine compute_fractional_uncertainty(efrac)
    implicit none
    integer :: i
    double precision, dimension(nintegrals) :: efrac
    do i=1,nintegrals
       if (vtot(i,0).ne.0d0) then
          efrac(i)=abs(etot(i,0)/vtot(i,0))
       else
          efrac(i)=0d0
       endif
    enddo
  end subroutine compute_fractional_uncertainty
  
  subroutine combine_results_channels
    implicit none
    integer :: kchan
    vtot(1:nintegrals,0)=sum(vtot(1:nintegrals,1:nchans),dim=2)
    etot(1:nintegrals,0)=sum(etot(1:nintegrals,1:nchans),dim=2)
    do kchan=0,nchans
       vtot(1:nintegrals,kchan)=vtot(1:nintegrals,kchan)/dble(ntotcalls(1:nintegrals))
       etot(1:nintegrals,kchan)=etot(1:nintegrals,kchan)/dble(ntotcalls(1:nintegrals))
       etot(1:nintegrals,kchan)=sqrt(abs(etot(1:nintegrals,kchan)-vtot(1:nintegrals,kchan)**2)  &
                                /dble(ntotcalls(1:nintegrals)))
    enddo
  end subroutine combine_results_channels
  
  subroutine check_for_special_channels_loop(channel_loop_done)
    implicit none
    logical :: channel_loop_done
    integer :: kchan,i
    do kchan=nchans,1,-1
       if (ans_chan(kchan).eq.1d0) then
! results of the current channel
          vtot(1:nintegrals,kchan)=vtot(1:nintegrals,kchan)/dble(ntotcalls(1:nintegrals))
          etot(1:nintegrals,kchan)=etot(1:nintegrals,kchan)/dble(ntotcalls(1:nintegrals))
          etot(1:nintegrals,kchan)=sqrt(abs(etot(1:nintegrals,kchan)-vtot(1:nintegrals,kchan)**2)  &
                                   /dble(ntotcalls(1:nintegrals)))
          if (kchan.eq.nchans) then
! done all channels
             channel_loop_done=.true.
             return
          endif
! prepare for the next channel
          ans_chan(kchan)=0d0
          ans_chan(kchan+1)=1d0
          ntotcalls(1:nintegrals)=0
          non_zero_point(1:nintegrals)=0
          pass_cuts_point=0
          kpoint_iter=0
          channel_loop_done=.false.
          return
       endif
    enddo
  end subroutine check_for_special_channels_loop

  subroutine combine_results_channels_special_loop
    implicit none
! set the total result for the first iteration to the sum over all the channels
    vtot(1:nintegrals,0)=sum(vtot(1:nintegrals,1:nchans),dim=2)
    etot(1:nintegrals,0)=sum(etot(1:nintegrals,1:nchans)**2,dim=2)
    etot(1:nintegrals,0)=sqrt(etot(1:nintegrals,0))
    ncalls0=ncalls0*nchans
  end subroutine combine_results_channels_special_loop
  

  subroutine get_amount_of_points(enough_points)
    ! fill the ntotcalls() array with the total number of calls used
    ! and check if this is enough for this iteration.
    implicit none
    logical :: enough_points
    integer :: i
    do i=1,nintegrals
! Number of phase-space points used
       ntotcalls(i)=ncalls*kpoint_iter
! Special for the computation of the 'computed virtual'
       if (i.eq.4 .and. non_zero_point(i).ne.0 ) &
            ntotcalls(i) = non_zero_point(i)
    enddo
    
    if (.not.double_events) then
! If not doubling the number of events for each iteration, nothing
! needs to be done here.
       enough_points=.true.
       return
    endif
    if (pass_cuts_point.lt.25) then
! Not enough points have passed to cuts to get a reliable estimate
       if (ntotcalls(1).gt.max_points) then
! tried many points already. Need to crash. 
          write (*,*) 'ERROR: NOT ENOUGH POINTS PASS THE CUTS. ' // &
               'RESULTS CANNOT BE TRUSTED. ' // &
               'LOOSEN THE GENERATION CUTS, OR ADAPT SET_TAU_MIN()' // &
               ' IN SETCUTS.F ACCORDINGLY.'
          stop 1
       else
          enough_points=.false.
          return
       endif
    endif
    if (non_zero_point(1).lt.int(0.99*ncalls)) then
! Not enough (non-zero) points have been generated
       if ( pass_cuts_point.gt.ncalls .and. &
            non_zero_point(1).lt.2) then
! Many points passed the cuts, but less than 2 non-zero integrand
! values: must be that the PDFs or the matrix elements (e.g. coupling
! constants) are numerically zero. End the run gracefully
          if (nit.gt.1 .or. imode.ne.0) then
             write (*,*) 'THE INTEGRAL APPEARS TO BE ZERO: END THE RUN GRACEFULLY.'
             write (*,*) 'TRIED',ntotcalls(1),'PS POINTS AND ONLY '  &
                  ,non_zero_point(1),' GAVE A NON-ZERO INTEGRAND.'
             call close_run_zero_res
             stop 0
          else
! This is for the special channels loop. Simply assume that the result
! for this channel is zero, and go to the next channel. If all
! channels give a zero result, end the run gracefully.
             vtot(1,ichan)=0d0
             if(ichan.eq.nchans .and. all(vtot(1,1:nchans).eq.0d0) ) then
                write (*,*) 'THE INTEGRAL APPEARS TO BE ZERO: END THE RUN GRACEFULLY.'
                write (*,*) 'TRIED',ntotcalls(1),'PS POINTS AND ONLY '  &
                     ,non_zero_point(1),' GAVE A NON-ZERO INTEGRAND.'
                call close_run_zero_res
                stop 0
             endif
             enough_points=.true.
             return
          endif
       else
          if (ntotcalls(1).lt.max_points) then
             enough_points=.false.
             return
          endif
       endif
    endif
    enough_points=.true.
  end subroutine get_amount_of_points
  
  

  subroutine add_point_to_grids(x)
    implicit none
    integer :: kdim,k_ord_virt,ithree,isix
    double precision, dimension(ndimmax) :: x
    double precision :: virtual,born
! accumulate the function in xacc(icell(kdim),kdim) to adjust the grid later
    do kdim=1,ndim
       xacc(icell(kdim),kdim,ichan) = xacc(icell(kdim),kdim,ichan) + f(1)
    enddo
! Set the Born contribution (to compute the average_virtual) to zero if
! the virtual was not computed for this phase-space point. Compensate by
! including the virtual_fraction.
    do k_ord_virt=0,n_ord_virt
       if (k_ord_virt.eq.0) then
          ithree=3
          isix=6
       else
          ithree=2*k_ord_virt+5
          isix=2*k_ord_virt+6
       endif
       if (f(ithree).ne.0d0) then
          born=f(isix)
          ! virt_wgt_mint=(virtual-average_virtual*born)/virtual_fraction. Compensate:
          if (use_poly_virtual) then
             virtual=f(ithree)*virtual_fraction(ichan)+ &
                  polyfit(k_ord_virt)*f(isix)
             call add_point_polyfit(ichan,k_ord_virt,x(1:ndim-3), &
                  virtual/born,born/wgt_mult)
          else
             virtual=f(ithree)*virtual_fraction(ichan)+ &
                  average_virtual(k_ord_virt,ichan)*f(isix)
             call fill_ave_virt(x,k_ord_virt,virtual,born)
          endif
       else
          f(isix)=0d0
       endif
    enddo
  end subroutine add_point_to_grids

  subroutine add_point_to_bounding_envelope
    implicit none
    integer :: kdim,k_ord_virt,ithree,isix
    double precision :: prod
! update the upper bounding envelope total rate
    prod=1d0
    do kdim=1,ndim
       prod=prod*ymax(ncell(kdim),kdim,ichan)
    enddo
    prod=(f(1)/prod)
    if (prod.gt.1d0) then
! Weight for this PS point is larger than current upper bound. Increase
! the bound so that it is equal to the current max weight.  If the new
! point is more than twice as large as current upper bound, increase
! bound by factor 2 only to prevent a single unstable points to
! completely screw up the efficiency
       prod=min(2d0,prod)
       prod=prod**(1d0/dble(ndim))
       do kdim=1,ndim
          ymax(ncell(kdim),kdim,ichan)=ymax(ncell(kdim),kdim,ichan)*prod
       enddo
    endif
! Update the upper bounding envelope virtual. Do not include the
! enhancement due to the virtual_fraction. (And again limit by factor 2
! at most).
    if (f(5)*virtual_fraction(ichan).gt.ymax_virt(ichan)) &
         ymax_virt(ichan) = min(f(5)*virtual_fraction(ichan),ymax_virt(ichan)*2d0)
! for consistent printing in the log files (in particular when doing LO
! runs), set also f(6) to zero when imode.eq.1 and the virtuals are not
! included.
    do k_ord_virt=0,n_ord_virt
       if (k_ord_virt.eq.0) then
          ithree=3
          isix=6
       else
          ithree=2*k_ord_virt+5
          isix=2*k_ord_virt+6
       endif
       if (f(ithree).eq.0) f(isix)=0d0
    enddo
  end subroutine add_point_to_bounding_envelope
     
  subroutine accumulate_the_point(x)
    implicit none
    integer :: i
    double precision, dimension(ndimmax) :: x
    if(imode.eq.0) then
       call add_point_to_grids(x)
    else
       call add_point_to_bounding_envelope
    endif
    do i=1,nintegrals
       if (f(i).ne.0d0) non_zero_point(i)=non_zero_point(i)+1
    enddo
    if (pass_cuts_check) pass_cuts_point=pass_cuts_point+1
! Add the PS point to the result of this iteration
    vtot(1:nintegrals,ichan)=vtot(1:nintegrals,ichan)+f(1:nintegrals)
    etot(1:nintegrals,ichan)=etot(1:nintegrals,ichan)+f(1:nintegrals)**2
! Accumulate the points in the HwU histograms    
    if (f(1).ne.0d0) call HwU_add_points
  end subroutine accumulate_the_point

  
  subroutine compute_integrand(fun,x,vol)
    implicit none
    integer :: ifirst,iret
    integer, dimension(ndimmax) :: kfold
    double precision :: dummy,vol
    double precision, dimension(nintegrals) :: f1
    double precision, dimension(ndimmax) :: x
    double precision, external :: fun
    ! contribution to integral
    ifirst=0
    if(imode.eq.0) then
       dummy=fun(x,vol,ifirst,f1)
       if (.not. fixed_order) dummy=fun(x,vol,2,f1)
       f(1:nintegrals)=f1(1:nintegrals)
    else
       f(1:nintegrals)=0d0
       kfold(1:ndim)=1
1      continue
       ! this accumulated value will not be used
       dummy=fun(x,vol,ifirst,f1)
       ifirst=1
       call nextlexi(ifold,kfold,iret)
       if(iret.eq.0) then
          call get_random_x_next_fold(x,vol,kfold)
          goto 1
       endif
       !closing call: accumulated value with correct sign
       ifirst=2
       dummy=fun(x,vol,ifirst,f1)
       f(1:nintegrals)=f1(1:nintegrals)
    endif
  end subroutine compute_integrand

  subroutine get_mint_wgt(x,vol)
    ! Given a random point x, compute the corresponding wgt from the
    ! importance sampling accompanying that point.
    implicit none
    double precision, dimension(ndimmax),intent(in) :: x
    double precision,intent(out) :: vol
    integer :: kdim,icell
    double precision :: dx
    ! This assumes we are in channel 'ichan'
    vol=1d0/vol_chan * wgt_mult
    do kdim=1,ndim
       ! determine the cell
       icell=1
       do while (xgrid(icell,kdim,ichan).lt.x(kdim))
          icell=icell+1
       enddo
       ! compute wgt
       dx=xgrid(icell,kdim,ichan)-xgrid(icell-1,kdim,ichan)
       vol=vol*dx*nint_used/ifold(kdim)
    enddo
  end subroutine get_mint_wgt
  
  subroutine get_random_x(x,vol,kfold)
    implicit none
    integer :: kdim,k_ord_virt,nintcurr
    integer, dimension(ndimmax) :: kfold
    double precision :: vol,dx
    double precision, dimension(ndimmax) :: x
    call get_channel
! find random x, and its random cell
    do kdim=1,ndim
! if(even_rn), we should compute the ncell and the rand from the ran3()
       if (even_rn) then
          rand(kdim)=ran3(even_rn)
          ncell(kdim)= min(int(rand(kdim)*nint_used)+1,nint_used)
          rand(kdim)=rand(kdim)*nint_used-(ncell(kdim)-1)
       else
          ncell(kdim)=min(int(nint_used/ifold(kdim)*ran3(even_rn))+1,nint_used/ifold(kdim))
          rand(kdim)=ran3(even_rn)
       endif
    enddo
    kfold(1:ndim)=1
    entry get_random_x_next_fold(x,vol,kfold)
    vol=1d0/vol_chan * wgt_mult
! convert 'flat x' ('rand') to 'vegas x' ('x') and include jacobian ('vol')
    do kdim=1,ndim
       nintcurr=nint_used/ifold(kdim)
       icell(kdim)=ncell(kdim)+(kfold(kdim)-1)*nintcurr
       dx=xgrid(icell(kdim),kdim,ichan)-xgrid(icell(kdim)-1,kdim,ichan)
       vol=vol*dx*nintcurr
       x(kdim)=xgrid(icell(kdim)-1,kdim,ichan)+rand(kdim)*dx
       if(imode.eq.0.and..not.born_spread_calibrating) &
            nhits(icell(kdim),kdim,ichan)= &
            nhits(icell(kdim),kdim,ichan)+1
    enddo
    do k_ord_virt=0,n_ord_virt
       if (use_poly_virtual) then
          call get_polyfit(ichan,k_ord_virt,x(1:ndim-3),polyfit(k_ord_virt))
       else
          call get_ave_virt(x,k_ord_virt)
       endif
    enddo
  end subroutine get_random_x
  

  subroutine start_iteration
    implicit none
    call write_channel_info
    nit=nit+1
    write (*,*) '------- iteration',nit
    call check_evenly_random_numbers
    if (imode.eq.0) then
       call reset_accumulated_grids_for_updating
    endif
    vtot(1:nintegrals,0:nchans)=0d0
    etot(1:nintegrals,0:nchans)=0d0
    kpoint_iter=0
    non_zero_point(1:nintegrals)=0
    pass_cuts_point=0
  end subroutine start_iteration

  subroutine reset_accumulated_grids_for_updating
    implicit none
    integer :: kchan
    do kchan=1,nchans
       ! only reset if grids were updated (or there is a forced reset)
       if (regridded(kchan).or.reset) then
          if (regridded(kchan) .and. .not. reset) then
             ! set nhits_in_grids equal to the number of points used for the last update
             nhits_in_grids(kchan)=sum(nhits(1:nint_used,1,kchan),dim=1)
          elseif (regridded(kchan) .and. reset) then
             nhits_in_grids(kchan)=0
          endif
          xacc(0:nint_used,1:ndim,kchan)=0d0
          nhits(1:nint_used,1:ndim,kchan)=0
       endif
    enddo
    reset=.false.
  end subroutine reset_accumulated_grids_for_updating

  subroutine check_evenly_random_numbers
    implicit none
    if (even_rn .and. ncalls.ne.ncalls0) then
       ! Uses more evenly distributed random numbers. This overwrites
       ! the number of calls
       call initialize_even_random_numbers
       write (*,*) 'Update # PS points (even_rn): ',ncalls0,' --> ',ncalls
    elseif (ncalls0.ne.ncalls) then
       ncalls=ncalls0
       write (*,*) 'Update # PS points: ',ncalls0,' --> ',ncalls
    endif
    npoints=ncalls
  end subroutine check_evenly_random_numbers

  subroutine finalise_mint
    implicit none
    integer :: kchan
    call write_channel_info
    if (nit_included.ge.2) then
       chi2(1,0:nchans)=chi2(1,0:nchans)/dble(nit_included-1)
    else
       chi2(1,0:nchans)=0d0
    endif
    write (*,*) '-------'
    ncalls0=ncalls*kpoint_iter ! return number of points used
    if (double_events) then
       itmax=2
    else
       itmax=nit_included
    endif
    cross_section=ans(2,0)
    do kchan=1,nchans
       if (regridded(kchan)) then
       ! set equal to number of points used for the last update
          nhits_in_grids(kchan)=sum(nhits(1:nint_used,1,kchan),dim=1) 
       endif
    enddo
    call write_grids_to_file
    call write_results
  end subroutine finalise_mint

  subroutine write_results
    implicit none
    integer :: kchan
    if (fixed_order) then
       write(*,*)'Final result [ABS]:',ans(1,0),' +/-',unc(1,0)
       write(*,*)'Final result:',ans(2,0),' +/-',unc(2,0)
       write(*,*)'chi**2 per D.o.F.:',chi2(1,0)
       open(unit=58,file='results.dat',status='unknown')
       do kchan=0,nchans
          write(58,*) ans(1,kchan),unc(2,kchan),0d0,0,0,0,0,0d0,0d0,ans(2,kchan)
       enddo
       close(58)
    else
       if (imode.eq.0) then
          open(unit=58,file='res_0',status='unknown')
          write(58,*)'Final result [ABS]:',ans(1,1),' +/-',unc(1,1)
          write(58,*)'Final result:',ans(2,1),' +/-',unc(2,1)
          close(58)
          write(*,*)'Final result [ABS]:',ans(1,1),' +/-',unc(1,1)
          write(*,*)'Final result:',ans(2,1),' +/-',unc(2,1)
          write(*,*)'chi**2 per D.o.F.:',chi2(1,1)
       elseif (imode.eq.1) then
! If integrating the virtuals alone, we include the virtuals in
! ans(1). Therefore, no need to have them in ans(5) and we have to set
! them to zero.
          if (only_virt) then
             ans(3,1)=0d0 ! virtual Xsec
             ans(5,1)=0d0 ! ABS virtual Xsec
          endif
          open(unit=58,file='res_1',status='unknown')
          write(58,*)'Final result [ABS]:',ans(1,1)+ans(5,1),' +/-',sqrt(unc(1,1)**2+unc(5,1)**2)
          write(58,*)'Final result:',ans(2,1),' +/-',unc(2,1)
          close(58)
          write(*,*)'Final result [ABS]:',ans(1,1)+ans(5,1),' +/-',sqrt(unc(1,1)**2+unc(5,1)**2)
          write(*,*)'Final result:',ans(2,1),' +/-',unc(2,1)
          write(*,*)'chi**2 per D.o.F.:',chi2(1,1)
          open(unit=58,file='results.dat',status='unknown')
          write(58,*)ans(1,1)+ans(5,1),unc(2,1),0d0,0,0,0,0,0d0,0d0,ans(2,1) 
          close(58)
       else
          continue
       endif
    endif
  end subroutine write_results
  
  subroutine write_channel_info
    implicit none
    integer :: kchan,np
    do kchan=1,nchans
       np=sum(nhits(1:nint_used,1,kchan))
       write (*,250) 'channel',kchan,':',iconfigs(kchan) &
            ,regridded(kchan),np,nhits_in_grids(kchan)   &
            ,ans_chan(kchan),ans(2,kchan),virtual_fraction(kchan)
    enddo
    call flush(6)
    return
250 format(a7,i5,1x,a1,1x,i5,1x,l,1x,i8,1x,i8,2x,e10.4,2x,e10.4,2x,e10.4)
  end subroutine write_channel_info
  

  subroutine setup_imode_1
    implicit none
    even_rn=.false.
    min_it=min_it1
    call reset_upper_bounding_envelope
    ans_chan(1:nchans)=ans(1,1:nchans)
    ans_chan(0)=sum(ans(1,1:nchans))
  end subroutine setup_imode_1

  subroutine reset_upper_bounding_envelope
    implicit none
    integer :: kdim,kint,nintcurr,nintcurr_virt
    do kdim=1,ndim
       nintcurr=nint_used/ifold(kdim)
       nintcurr_virt=nint_used_virt/ifold(kdim)
       if(nintcurr*ifold(kdim).ne.nint_used .or. &
            nintcurr_virt*ifold(kdim).ne.nint_used_virt) then
          write(*,*) 'mint: the values in the ifold array shoud be divisors of', &
               nint_used,'and',nint_used_virt
          stop 1
       endif
       do kint=1,nintcurr
          ymax(kint,kdim,1:nchans)=ans(1,1:nchans)**(1d0/ndim)
       enddo
    enddo
    ymax_virt(1:nchans)=ans(5,1:nchans)
  end subroutine reset_upper_bounding_envelope
  
  subroutine setup_imode_m1
    implicit none
    even_rn=.true.
    imode=0
    min_it=min_it0
    ans_chan(1:nchans)=ans(1,1:nchans)
    ans_chan(0)=sum(ans(1,1:nchans))
  end subroutine setup_imode_m1
  
  subroutine setup_imode_0
    implicit none
    even_rn=.true.
    min_it=min_it0
    call reset_mint_grids
  end subroutine setup_imode_0

  subroutine reset_mint_grids
    implicit none
    integer :: kdim,kint
    do kint=0,nint_used
       xgrid(kint,1:ndim,1:nchans)=dble(kint)/nint_used
    enddo
    nhits(1:nint_used,1:ndim,1:nchans)=0
    regridded(1:nchans)=.true.
    nhits_in_grids(1:nchans)=0
    if (use_poly_virtual) then
       call init_polyfit(ndim-3,nchans,n_ord_virt,1000)
    else
       call init_ave_virt
    endif
    virtual_fraction(1:nchans)=max(virt_fraction,min_virt_fraction)
    average_virtual(0:n_ave_virt,1:nchans)=0d0
    ans_chan(0:nchans)=0d0
    if (double_events) then
       ! when double events, start with the very first channel only. For the
       ! first iteration, we compute each channel separately.
       ans_chan(0)=1d0
       ans_chan(1)=1d0
       ncalls0=ncalls0/nchans
    endif
  end subroutine reset_mint_grids
  
  subroutine setup_common
    implicit none
    nit=0
    nit_included=0
    ans(1:nintegrals,0:nchans)=0d0
    unc(1:nintegrals,0:nchans)=0d0
    chi2(1:nintegrals,0:nchans)=0d0
    ans3(1:nintegrals,1:3)=0d0
    unc3(1:nintegrals,1:3)=0d0
    ans_l3(1:nintegrals)=0d0
    unc_l3(1:nintegrals)=0d0
    chi2_l3(1:nintegrals)=0d0
    vtot(1:nintegrals,0:nchans)=0d0
    etot(1:nintegrals,0:nchans)=0d0
    ntotcalls(1:nintegrals)=0
    non_zero_point(1:nintegrals)=0
    pass_cuts_point=0
    HwU_values(1:2)=0d0
  end subroutine setup_common
  
  
  subroutine write_grids_to_file
! Write the MINT integration grids to file
    implicit none
    integer :: i,j,k,kchan
    open (unit=12,file='mint_grids',status='unknown')
    do kchan=1,nchans
       do j=0,nintervals
          write (12,*) 'AVE',(xgrid(j,i,kchan),i=1,ndim)
       enddo
       if (imode.ge.1) then
          do j=1,nintervals
             write (12,*) 'MAX',(ymax(j,i,kchan),i=1,ndim)
          enddo
       endif
       if (.not.use_poly_virtual) then
          do j=1,nintervals_virt
             do k=0,n_ord_virt
                write (12,*) 'AVE',(ave_virt(j,i,k,kchan),i=1,ndim-3)
             enddo
          enddo
       endif
       if (imode.ge.1) then
          write (12,*) 'MAX',ymax_virt(kchan)
       endif
       write (12,*) 'SUM',(ans(i,kchan),i=1,nintegrals)
       write (12,*) 'QSM',(unc(i,kchan),i=1,nintegrals)
       write (12,*) 'SPE',ncalls0,itmax,nhits_in_grids(kchan)
       write (12,*) 'AVE',virtual_fraction(kchan),average_virtual(0,kchan)
    enddo
    if (use_poly_virtual) call save_polyfit(12)
    close (12)
  end subroutine write_grids_to_file
  
  subroutine read_grids_from_file
! Read the MINT integration grids from file
    implicit none
    integer :: i,j,k,kchan,idum
    integer,dimension(maxchannels) :: points
    character(len=3) :: dummy
    open (unit=12, file='mint_grids',status='old')
    ans(1,0)=0d0
    unc(1,0)=0d0
    do kchan=1,nchans
       do j=0,nintervals
          read (12,*) dummy,(xgrid(j,i,kchan),i=1,ndim)
       enddo
       if (imode.ge.2) then
          do j=1,nintervals
             read (12,*) dummy,(ymax(j,i,kchan),i=1,ndim)
          enddo
       endif
       if (.not.use_poly_virtual) then
          do j=1,nintervals_virt
             do k=0,n_ord_virt
                read (12,*) dummy,(ave_virt(j,i,k,kchan),i=1,ndim-3)
             enddo
          enddo
       endif
       if (imode.ge.2) then
          read (12,*) dummy,ymax_virt(kchan)
       endif
       read (12,*) dummy,(ans(i,kchan),i=1,nintegrals)
       read (12,*) dummy,(unc(i,kchan),i=1,nintegrals)
       read (12,*) dummy,idum,idum,nhits_in_grids(kchan)
       read (12,*) dummy,virtual_fraction(kchan),average_virtual(0,kchan)
       ans(1,0)=ans(1,0)+ans(1,kchan)
       unc(1,0)=unc(1,0)+unc(1,kchan)**2
    enddo
    unc(1,0)=sqrt(unc(1,0))
    ! polyfit stuff:
    if (use_poly_virtual) then
       do kchan=1,nchans
          read (12,*) dummy,points(kchan)
       enddo
       do kchan=1,nchans
          backspace(12)
       enddo
       call init_polyfit(ndim-3,nchans,n_ord_virt,maxval(points(1:nchans)))
       call restore_polyfit(12)
       call do_polyfit()
    endif
    close (12)
! check for zero cross-section: if restoring grids corresponding to
! sigma=0, just terminate the run
    if (imode.ne.0.and.ans(1,0).eq.0d0.and.unc(1,0).eq.0d0) then
       call initplot()
       call close_run_zero_res
       stop 0
    endif
  end subroutine read_grids_from_file

  subroutine double_grid(kdim,kchan)
    implicit none
    integer :: kchan,kdim,i
    do i=nint_used,1,-1
       xgrid(i*2,kdim,kchan)=xgrid(i,kdim,kchan)
       xgrid(i*2-1,kdim,kchan)=(xgrid(i,kdim,kchan)+xgrid(i-1,kdim,kchan))/2d0
       if ((.not.regridded(kchan)) .and. (.not.reset)) then
          nhits(i*2,kdim,kchan)=nhits(i,kdim,kchan)/2
          nhits(i*2-1,kdim,kchan)=nhits(i,kdim,kchan)-nhits(i*2,kdim,kchan)
          if (nhits(i,kdim,kchan).ne.0) then
             xacc(i*2,kdim,kchan)=xacc(i,kdim,kchan)*nhits(i*2,kdim,kchan)/dble(nhits(i,kdim,kchan))
             xacc(i*2-1,kdim,kchan)=xacc(i,kdim,kchan)*nhits(i*2-1,kdim,kchan)/dble(nhits(i,kdim,kchan))
          else
             xacc(i*2,kdim,kchan)=0d0
             xacc(i*2-1,kdim,kchan)=0d0
          endif
       endif
    enddo
  end subroutine double_grid

  subroutine regrid(kdim,kchan)
    implicit none
    integer :: kdim,kchan,kint,jint
    double precision :: r,total
    double precision, parameter :: tiny=1d-8
! compute total number of points and update grids if large
    regridded(kchan)=.false.
    if (sum(nhits(1:nint_used,kdim,kchan),dim=1).lt.nint(0.9*nhits_in_grids(kchan))) return
    regridded(kchan)=.true.
! Use the same smoothing as in VEGAS uses for the grids, i.e. use the
! average of the central and the two neighbouring grid points: (Only do
! this if we are already at the maximum intervals, because the doubling
! of the grids also includes a smoothing).
    if (nint_used.eq.nintervals) then
       call smooth_xacc(kdim,kchan)
    endif
    do kint=1,nint_used
       if (nhits(kint,kdim,kchan).ne.0) then
          xacc(kint,kdim,kchan)=abs(xacc(kint,kdim,kchan))/nhits(kint,kdim,kchan)
       else
          xacc(kint,kdim,kchan)=0d0
       endif
    enddo
! Overwrite xacc so that it accumulates the cross section with each
! successive interval.  It already contains a factor equal to the
! interval size. Thus the integral of rho is performed by summing up
    total=sum(xacc(1:nint_used,kdim,kchan),dim=1)
    do kint=1,nint_used
       if(nhits(kint,kdim,kchan).ne.0) then
!     take logarithm to help convergence (taken from LO dsample.f)
          if (xacc(kint,kdim,kchan).ne.total) then
             xacc(kint,kdim,kchan)=((xacc(kint,kdim,kchan)/total-1d0)/log(xacc(kint,kdim,kchan)/total))**1.5
          else
             xacc(kint,kdim,kchan)=1d0
          endif
          xacc(kint,kdim,kchan)= xacc(kint-1,kdim,kchan) + abs(xacc(kint,kdim,kchan))
       else
          xacc(kint,kdim,kchan)=xacc(kint-1,kdim,kchan)
       endif
    enddo
! No valid points. Simply return
    if (xacc(nint_used,kdim,kchan).eq.0d0) return
! normalise xacc so that it goes from 0 to 1.
    xacc(1:nint_used,kdim,kchan)=xacc(1:nint_used,kdim,kchan)/xacc(nint_used,kdim,kchan)
! Check that we have a reasonable result and update the accumulated results if need be
    do kint=1,nint_used
       if (xacc(kint,kdim,kchan).lt.(xacc(kint-1,kdim,kchan)+tiny)) then
          xacc(kint,kdim,kchan)=xacc(kint-1,kdim,kchan)+tiny
       endif
    enddo
! it could happen that the change above yielded xacc() values greater than 1: one more update needed
    xacc(nint_used,kdim,kchan)=1d0
    do kint=1,nint_used
       if (xacc(nint_used-kint,kdim,kchan).gt.(xacc(nint_used-kint+1,kdim,kchan)-tiny)) then
          xacc(nint_used-kint,kdim,kchan)=1d0-dble(kint)*tiny
       else
          exit
       endif
    enddo
! adjust 'xgrid_new' (temporary grid) so that each element contains identical cross section    
    xgrid_new(0,kdim)=0d0
    do kint=1,nint_used
       r=dble(kint)/dble(nint_used)
       do jint=1,nint_used
          if(r.lt.xacc(jint,kdim,kchan)) then
             xgrid_new(kint,kdim)=xgrid(jint-1,kdim,kchan)+(r-xacc(jint-1,kdim,kchan))/ &
                  (xacc(jint,kdim,kchan)-xacc(jint-1,kdim,kchan))* &
                  (xgrid(jint,kdim,kchan)-xgrid(jint-1,kdim,kchan))
             goto 11
          endif
       enddo
       if(jint.ne.nint_used+1.and.kint.ne.nint_used) then
          write(*,*) 'ERROR',jint,nint_used
          stop 1
       endif
11     continue
    enddo
    xgrid_new(nint_used,kdim)=1d0
  end subroutine regrid

  subroutine smooth_xacc(kdim,kchan)
    implicit none
    integer :: kdim,kchan,kint,kk,itot,kkint
    double precision :: tot
    integer, parameter :: isize=1
    integer, dimension(1:nintervals) :: local_nhits
    double precision, dimension(1:nintervals) :: local_xacc
    do kint=1,nint_used
       tot=0d0
       itot=0
       do kk=-isize,isize
          kkint=kint+kk
          if (kkint.le. 0) kkint=1
          if (kkint.ge.nint_used+1) kkint=nint_used
          tot=tot+xacc(kkint,kdim,kchan)
          itot=itot+nhits(kkint,kdim,kchan)
       enddo
       local_xacc(kint)=tot/dble(2*isize+1)
       local_nhits(kint)=nint(itot/dble(2*isize+1))
    enddo
    xacc(1:nint_used,kdim,kchan)=local_xacc(1:nint_used)
    nhits(1:nint_used,kdim,kchan)=local_nhits(1:nint_used)
  end subroutine smooth_xacc
    
  subroutine nextlexi(iii,kkk,iret)
! kkk: array of integers 1 <= kkk(j) <= iii(j), j=1,ndim
! at each call iii is increased lexicographycally.
! for example, starting from ndim=3, kkk=(1,1,1), iii=(2,3,2)
! subsequent calls to nextlexi return
!         kkk(1)      kkk(2)      kkk(3)    iret
! 0 calls   1           1           1       0
! 1         1           1           2       0    
! 2         1           2           1       0
! 3         1           2           2       0
! 4         1           3           1       0
! 5         1           3           2       0
! 6         2           1           1       0
! 7         2           1           2       0
! 8         2           2           1       0
! 9         2           2           2       0
! 10        2           3           1       0
! 11        2           3           2       0
! 12        2           3           2       1
    implicit none
    integer :: k,iret
    integer,dimension(ndimmax) :: kkk,iii
    k=ndim
1   continue
    if(kkk(k).lt.iii(k)) then
       kkk(k)=kkk(k)+1
       iret=0
       return
    else
       kkk(k)=1
       k=k-1
       if(k.eq.0) then
          iret=1
          return
       endif
       goto 1
    endif
  end subroutine nextlexi

  subroutine init_ave_virt
    implicit none
    if (n_ord_virt.gt.n_ave_virt) then
       write (*,*) 'Too many grids to keep track off',n_ord_virt,n_ave_virt
       stop 1
    endif
    nvirt(1:nint_used_virt,1:ndim-3,0:n_ord_virt,1:nchans)=0
    ave_virt(1:nint_used_virt,1:ndim-3,0:n_ord_virt,1:nchans)=0d0
    nvirt_acc(1:nint_used_virt,1:ndim-3,0:n_ord_virt,1:nchans)=0
    ave_virt_acc(1:nint_used_virt,1:ndim-3,0:n_ord_virt,1:nchans)=0d0
    ave_born_acc(1:nint_used_virt,1:ndim-3,0:n_ord_virt,1:nchans)=0d0
  end subroutine init_ave_virt

  subroutine get_ave_virt(x,k_ord_virt)
    implicit none
    integer :: kdim,ncell,k_ord_virt
    double precision, dimension(ndimmax) :: x
    average_virtual(k_ord_virt,ichan)=0d0
    do kdim=1,ndim-3
       ncell=min(int(x(kdim)*nint_used_virt)+1,nint_used_virt)
       average_virtual(k_ord_virt,ichan)=average_virtual(k_ord_virt,ichan) &
                                        +ave_virt(ncell,kdim,k_ord_virt,ichan)
    enddo
    average_virtual(k_ord_virt,ichan)=average_virtual(k_ord_virt,ichan)/(ndim-3)
  end subroutine get_ave_virt

  subroutine fill_ave_virt(x,k_ord_virt,virtual,born)
    implicit none
    integer :: kdim,ncell,k_ord_virt
    double precision,dimension(ndimmax) :: x(ndimmax)
    double precision :: virtual,born
    do kdim=1,ndim-3
       ncell=min(int(x(kdim)*nint_used_virt)+1,nint_used_virt)
       nvirt_acc(ncell,kdim,k_ord_virt,ichan)=nvirt_acc(ncell,kdim,k_ord_virt,ichan)+1
       ave_virt_acc(ncell,kdim,k_ord_virt,ichan)=ave_virt_acc(ncell,kdim,k_ord_virt,ichan)+virtual
       ave_born_acc(ncell,kdim,k_ord_virt,ichan)=ave_born_acc(ncell,kdim,k_ord_virt,ichan)+born
    enddo
  end subroutine fill_ave_virt

  subroutine regrid_ave_virt(k_ord_virt)
    implicit none
    integer kchan,kdim,i,k_ord_virt
! need to solve for k_new = (virt+k_old*born)/born = virt/born + k_old
    do kchan=1,nchans
       do kdim=1,ndim-3
          do i=1,nint_used_virt
             if (ave_born_acc(i,kdim,k_ord_virt,kchan).eq.0d0) cycle
             if (ave_virt(i,kdim,k_ord_virt,kchan).eq.0d0) then ! i.e. first iteration
                ave_virt(i,kdim,k_ord_virt,kchan)= ave_virt_acc(i,kdim,k_ord_virt,kchan) &
                     /ave_born_acc(i,kdim,k_ord_virt,kchan)+ave_virt(i,kdim,k_ord_virt,kchan)
             else  ! give some importance to the iterations already done
                ave_virt(i,kdim,k_ord_virt,kchan)=(nvirt_acc(i,kdim,k_ord_virt,kchan)* &
                     ave_virt_acc(i,kdim,k_ord_virt,kchan)/ave_born_acc(i,kdim,k_ord_virt,kchan)+ &
                     nvirt(i,kdim,k_ord_virt,kchan)*ave_virt(i,kdim,k_ord_virt,kchan))/ &
                     dble(nvirt_acc(i,kdim,k_ord_virt,kchan)+nvirt(i,kdim,k_ord_virt,kchan))
             endif
          enddo
       enddo
    enddo
! reset the acc values
    nvirt(1:nint_used_virt,1:ndim-3,k_ord_virt,1:nchans)= &
                 nvirt(1:nint_used_virt,1:ndim-3,k_ord_virt,1:nchans)  &
                 + nvirt_acc(1:nint_used_virt,1:ndim-3,k_ord_virt,1:nchans)
    nvirt_acc(1:nint_used_virt,1:ndim-3,k_ord_virt,1:nchans)=0
    ave_born_acc(1:nint_used_virt,1:ndim-3,k_ord_virt,1:nchans)=0d0
    ave_virt_acc(1:nint_used_virt,1:ndim-3,k_ord_virt,1:nchans)=0d0
  end subroutine regrid_ave_virt


  subroutine double_ave_virt(k_ord_virt)
    implicit none
    integer :: kdim,i,k_ord_virt,kchan
    do kchan=1,nchans
       do kdim=1,ndim-3
          do i=nint_used_virt,1,-1
             ave_virt(i*2,kdim,k_ord_virt,kchan)=ave_virt(i,kdim,k_ord_virt,kchan)
             if (nvirt(i,kdim,k_ord_virt,kchan).ne.0) then
                nvirt(i*2,kdim,k_ord_virt,kchan)=max(nvirt(i,kdim,k_ord_virt,kchan)/2,1)
             else
                nvirt(i*2,kdim,k_ord_virt,kchan)=0
             endif
             if (i.ne.1) then
                ave_virt(i*2-1,kdim,k_ord_virt,kchan)=(ave_virt(i,kdim,k_ord_virt,kchan)  &
                                                      +ave_virt(i-1,kdim,k_ord_virt,kchan))/2d0
                if (nvirt(i,kdim,k_ord_virt,kchan)+nvirt(i-1,kdim,k_ord_virt,kchan).ne.0) then
                   nvirt(i*2-1,kdim,k_ord_virt,kchan)= &
                        max((nvirt(i,kdim,k_ord_virt,kchan)+nvirt(i-1,kdim,k_ord_virt,kchan))/4,1)
                else
                   nvirt(i*2-1,kdim,k_ord_virt,kchan)=0
                endif
             else
                if (nvirt(1,kdim,k_ord_virt,kchan).ne.0) then
                   nvirt(1,kdim,k_ord_virt,kchan)=max(nvirt(1,kdim,k_ord_virt,kchan)/2,1)
                else
                   nvirt(1,kdim,k_ord_virt,kchan)=0
                endif
             endif
          enddo
       enddo
    enddo
  end subroutine double_ave_virt


  subroutine get_channel
! Picks one random 'ichan' among the 'nchans' integration channels and
! fills the channels common block in mint.inc.
    implicit none
    double precision :: trgt,total
    if (nchans.eq.1) then
       ichan=1
       iconfig=iconfigs(ichan)
       vol_chan=1d0
    elseif (nchans.gt.1) then
       if (ans_chan(0).le.0d0) then
!     pick one at random (flat)
          ichan=int(ran3(.false.)*nchans)+1
          iconfig=iconfigs(ichan)
          vol_chan=1d0/dble(nchans)
       else
!     pick one at random (weighted by cross section)
          total=sum(ans_chan(1:nchans))
          if (abs(total-ans_chan(0))/(total+ans_chan(0)).gt.1d-8) then
             write (*,*) 'ERROR: total should be equal to ans',total,ans_chan(0)
             stop 1
          endif
          trgt=ans_chan(0)*ran3(.false.)
          total=0d0
          ichan=0
          do while (total.lt.trgt)
             ichan=ichan+1
             total=total+ans_chan(ichan)
          enddo
          if (ichan.eq.0 .or. ichan.gt.nchans) then
             write (*,*) 'ERROR: ichan cannot be zero or larger than nchans',ichan,nchans
             stop 1
          endif
          iconfig=iconfigs(ichan)
          vol_chan=ans_chan(ichan)/ans_chan(0)
       endif
    endif
  end subroutine get_channel


  subroutine close_run_zero_res
    implicit none
    integer :: kchan
    xgrid(0:nintervals,1:ndim,1:nchans)=0d0
    ymax(1:nintervals,1:ndim,1:nchans)=0d0
    ave_virt(1:nintervals_virt,1:ndim-3,0:n_ord_virt,1:nchans)=0d0
    ymax_virt(1:nchans)=0d0
    ans(1:nintegrals,1:nchans)=0d0
    unc(1:nintegrals,1:nchans)=0d0
    nhits_in_grids(1:nchans)=0
    virtual_fraction(1:nchans)=1d0
    average_virtual(0,1:nchans)=0d0
    call write_grids_to_file
    call write_results
    call regrid_MC_integer
    open (unit=12, file='res.dat',status='unknown')
    do kchan=0,nchans
       write (12,*)ans(1,kchan),unc(1,kchan),ans(2,kchan),unc(2,kchan) &
            ,itmax,ncalls0,0d0
    enddo
    close(12)
  end subroutine close_run_zero_res

  function ran3(even)
    implicit none
    double precision :: ran3
    logical :: even
    double precision, external :: ran2
    if (even) then
       ran3=get_ran()
    else
       ran3=ran2()
    endif
  end function ran3

  subroutine initialize_even_random_numbers
! Recompute the number of calls. Uses the algorithm from VEGAS
    implicit none
    integer :: i
! Make sure that hypercubes are newly initialized
    firsttime=.true.
! Number of elements in which we can split one dimension
    ng=(ncalls0/2.)**(1./ndim)
! Total number of hypercubes
    k=ng**ndim
! Number of PS points in each hypercube (at least 2)
    npg=max(ncalls0/k,2)
! Number of PS points for this iteration
    ncalls=npg*k
  end subroutine initialize_even_random_numbers

  function get_ran()
    implicit none
    double precision :: get_ran
    double precision, save :: dng
    double precision, external ::  ran2
    integer, dimension(ndimmax), save ::  iii,kkk
    integer, save :: current_dim
    integer :: i,iret
    if (firsttime) then
! initialise the hypercubes
       dng=1d0/dble(ng)
       current_dim=0
       do i=1,ndim
          iii(i)=ng
          kkk(i)=1
       enddo
       firsttime=.false.
    endif
    current_dim=mod(current_dim,ndim)+1
! This is the random number in the hypercube 'k' for current_dim
    get_ran=dng*(ran2()+dble(kkk(current_dim)-1))
! Got random numbers for all dimensions, update kkk() for the next call
    if (current_dim.eq.ndim) then
       call nextlexi(iii,kkk,iret)
       if (iret.eq.1) then
          call nextlexi(iii,kkk,iret)
       endif
    endif
  end function get_ran





  subroutine gen(fun,gen_mode,vn,x)
    implicit none
    integer :: vn,gen_mode
    logical :: found_point
    double precision, external :: fun
    double precision, dimension(ndimmax) :: x
    double precision :: vol
    if (gen_mode.eq.0) then
       call initialise_mint_gen
    elseif(gen_mode.eq.3) then
       call print_gen_counters
    elseif(gen_mode.eq.1) then
       call increase_gen_counters_before(vn)
10     continue
       new_point=.true.
       if (vn.eq.1) then
          call get_random_cell_flat(x,vol)
       else
          call get_weighted_cell(x,vol)
       endif
       call compute_integrand(fun,x,vol)
       call increase_gen_counters_middle(vn)
       call check_upper_bound(vn,found_point)
       if (.not.found_point) goto 10
       call increase_gen_counters_end(vn)
    else
       write (*,*) "Unknown gen_mode in gen (from mint_module)",gen_mode
       stop 1
    endif
  end subroutine gen

  subroutine increase_gen_counters_middle(vn)
    implicit none
    integer :: vn
    gen_counters(3)=gen_counters(3)+1
    if (vn.eq.1) then
       gen_counters(5)=gen_counters(5)+1
    else
       gen_counters(6)=gen_counters(6)+1
    endif
    if (f(1).eq.0d0) then
       gen_counters(4)=gen_counters(4)+1
    endif
  end subroutine increase_gen_counters_middle
  
  subroutine increase_gen_counters_before(vn)
    implicit none
    integer :: vn
    if (vn.eq.1) then
       gen_counters(1)=gen_counters(1)+1
    else
       gen_counters(2)=gen_counters(2)+1
    endif
  end subroutine increase_gen_counters_before

  subroutine increase_gen_counters_end(vn)
    implicit none
    integer :: vn
    if (vn.eq.2) then
       gen_counters(11)=gen_counters(11)+1
    elseif (vn.eq.1) then
       gen_counters(12)=gen_counters(12)+1
    elseif (vn.eq.3) then
       gen_counters(13)=gen_counters(13)+1
    endif
  end subroutine increase_gen_counters_end

  subroutine check_upper_bound(vn,found_point)
    implicit none
    logical :: found_point
    integer :: vn
    if (f(1).gt.upper_bound) then
       if (vn.eq.2) then
          gen_counters(7)=gen_counters(7)+1
       elseif (vn.eq.1) then
          gen_counters(8)=gen_counters(8)+1
       elseif(vn.eq.3) then
          gen_counters(9)=gen_counters(9)+1
       endif
    endif
    upper_bound=upper_bound*ran3(.false.)
    if (upper_bound.gt.f(1)) then
       gen_counters(10)=gen_counters(10)+1
       found_point=.false.
    else
       found_point=.true.
    endif
  end subroutine check_upper_bound
  
  subroutine get_random_cell_flat(x,vol)
    implicit none
    double precision :: vol
    double precision, dimension(ndimmax) :: x
    integer, dimension(ndimmax) :: kfold
    call get_random_x(x,vol,kfold)
    upper_bound=ymax_virt(ichan)
  end subroutine get_random_cell_flat

  subroutine get_weighted_cell(x,vol)
    implicit none
    integer :: kdim,nintcurr,kint
    integer, dimension(ndimmax) :: kfold
    double precision :: vol,r
    double precision, dimension(ndimmax) :: x
    call get_channel
    do kdim=1,ndim
       nintcurr=nintervals/ifold(kdim)
       r=ran3(.false.)
       do kint=1,nintcurr
          if(r.lt.xmmm(kint,kdim,ichan)) then
             ncell(kdim)=kint
             exit
          endif
       enddo
       rand(kdim)=ran3(.false.)
    enddo
    kfold(1:ndim)=1
    call get_random_x_next_fold(x,vol,kfold)
    upper_bound=1d0
    do kdim=1,ndim
       upper_bound=upper_bound*ymax(ncell(kdim),kdim,ichan)
    enddo
  end subroutine get_weighted_cell
  

  subroutine initialise_mint_gen
    implicit none
    integer :: kdim,kint
    integer :: nintcurr
    if (nchans.ne.1) then
       write (*,*) 'ERROR in mint_module: for event generation, can do only 1 channel at a time',nchans
       stop 1
    endif
    even_rn=.false.
    nint_used=nintervals
    nint_used_virt=nintervals_virt
    do kdim=1,ndim
       nintcurr=nintervals/ifold(kdim)
       xmmm(1,kdim,1:nchans)=ymax(1,kdim,1:nchans)
       do kint=2,nintcurr
          xmmm(kint,kdim,1:nchans)=xmmm(kint-1,kdim,1:nchans)+ymax(kint,kdim,1:nchans)
       enddo
       do kint=1,nintcurr
          xmmm(kint,kdim,1:nchans)=xmmm(kint,kdim,1:nchans)/xmmm(nintcurr,kdim,1:nchans)
       enddo
    enddo
    gen_counters(1:13)=0
  end subroutine initialise_mint_gen

  subroutine print_gen_counters
    implicit none
    double precision :: unwgt_eff, unwgt_eff_virt
    if (gen_counters( 3).ne.0) write (*,*) 'another call to the function:',gen_counters(3)
    if (gen_counters(11).ne.0) write (*,*) 'events generated, novi:',gen_counters(11)
    if (gen_counters(12).ne.0) write (*,*) 'events generated, virt:',gen_counters(12)
    if (gen_counters(13).ne.0) write (*,*) 'events generated, born:',gen_counters(13)
    if (gen_counters( 7).ne.0) write (*,*) 'upper bound failure, novi:',gen_counters(7)
    if (gen_counters( 8).ne.0) write (*,*) 'upper bound failure, virt:',gen_counters(8)
    if (gen_counters( 9).ne.0) write (*,*) 'upper bound failure, born:',gen_counters(9)
    if (gen_counters(10).ne.0) write (*,*) 'vetoed calls in inclusive cross section:',gen_counters(10)
    if (gen_counters( 4).ne.0) write (*,*) 'failed generation cuts:',gen_counters(4)
    if(gen_counters(6).ne.0) then
       unwgt_eff=dble(gen_counters(2))/dble(gen_counters(6))
    else
       unwgt_eff=-1d0
    endif
    if(gen_counters(5).ne.0) then
       unwgt_eff_virt=dble(gen_counters(1))/dble(gen_counters(5))
    else
       unwgt_eff_virt=-1d0
    endif
    write (*,*) 'Generation efficiencies:',unwgt_eff,unwgt_eff_virt
  end subroutine print_gen_counters

  
end module mint_module
  
! Dummy subroutine (normally used with vegas when resuming plots)
subroutine resume()
end subroutine resume
