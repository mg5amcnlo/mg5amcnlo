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
  integer, parameter, private :: max_points=100000! maximum number of points to trow per iteration if not enough non-zero points can be found.
  integer, parameter, public  :: maxchannels=20 ! set as least as large as in amcatnlo_run_interface
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
  public :: mint,gen,read_grids_from_file
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
       if (imode.eq.0 .and. nit.eq.1 .and. double_events) then
          call check_for_special_channels_loop(channel_loop_done)
          if (.not.channel_loop_done) goto 2
          call combine_results_channels_special_loop
       else
          call combine_results_channels
       endif
       call update_accumulated_results
    enddo
    call finalise_mint
  end subroutine mint

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
       if(imode.eq.0) nhits(icell(kdim),kdim,ichan)=nhits(icell(kdim),kdim,ichan)+1
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
    ans3(1:nintegrals,1:3)=0d0
    unc3(1:nintegrals,1:3)=0d0
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
                write (12,*) 'AVE',(ave_virt(j,i,k,kchan),i=1,ndim)
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
    write (12,*) 'IDE',(ifold(i),i=1,ndim)
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
                read (12,*) dummy,(ave_virt(j,i,k,kchan),i=1,ndim)
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
    read (12,*) dummy,(ifold(i),i=1,ndim)
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
    nvirt(1:nint_used_virt,1:ndim,0:n_ord_virt,1:nchans)=0
    ave_virt(1:nint_used_virt,1:ndim,0:n_ord_virt,1:nchans)=0d0
    nvirt_acc(1:nint_used_virt,1:ndim,0:n_ord_virt,1:nchans)=0
    ave_virt_acc(1:nint_used_virt,1:ndim,0:n_ord_virt,1:nchans)=0d0
    ave_born_acc(1:nint_used_virt,1:ndim,0:n_ord_virt,1:nchans)=0d0
  end subroutine init_ave_virt

  subroutine get_ave_virt(x,k_ord_virt)
    implicit none
    integer :: kdim,ncell,k_ord_virt
    double precision, dimension(ndimmax) :: x
    average_virtual(k_ord_virt,ichan)=0d0
    do kdim=1,ndim
       ncell=min(int(x(kdim)*nint_used_virt)+1,nint_used_virt)
       average_virtual(k_ord_virt,ichan)=average_virtual(k_ord_virt,ichan) &
                                        +ave_virt(ncell,kdim,k_ord_virt,ichan)
    enddo
    average_virtual(k_ord_virt,ichan)=average_virtual(k_ord_virt,ichan)/ndim
  end subroutine get_ave_virt

  subroutine fill_ave_virt(x,k_ord_virt,virtual,born)
    implicit none
    integer :: kdim,ncell,k_ord_virt
    double precision,dimension(ndimmax) :: x(ndimmax)
    double precision :: virtual,born
    do kdim=1,ndim
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
       do kdim=1,ndim
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
    nvirt(1:nint_used_virt,1:ndim,k_ord_virt,1:nchans)= &
                 nvirt(1:nint_used_virt,1:ndim,k_ord_virt,1:nchans)  &
                 + nvirt_acc(1:nint_used_virt,1:ndim,k_ord_virt,1:nchans)
    nvirt_acc(1:nint_used_virt,1:ndim,k_ord_virt,1:nchans)=0
    ave_born_acc(1:nint_used_virt,1:ndim,k_ord_virt,1:nchans)=0d0
    ave_virt_acc(1:nint_used_virt,1:ndim,k_ord_virt,1:nchans)=0d0
  end subroutine regrid_ave_virt


  subroutine double_ave_virt(k_ord_virt)
    implicit none
    integer :: kdim,i,k_ord_virt,kchan
    do kchan=1,nchans
       do kdim=1,ndim
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
    ave_virt(1:nintervals_virt,1:ndim,0:n_ord_virt,1:nchans)=0d0
    ymax_virt(1:nchans)=0d0
    ans(1:nintegrals,1:nchans)=0d0
    unc(1:nintegrals,1:nchans)=0d0
    nhits_in_grids(1:nchans)=0
    virtual_fraction(1:nchans)=1d0
    average_virtual(0,1:nchans)=0d0
    call write_grids_to_file
    call write_results
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
