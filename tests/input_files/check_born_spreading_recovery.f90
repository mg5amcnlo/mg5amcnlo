module recovery_observations
  implicit none
  integer :: bad_iterations=0,previous_bad_iteration=-1
  integer :: recovered_channel_calls(3)=0
  logical :: recovered=.false.
end module recovery_observations

program check_recovery
  use mint_module
  use recovery_observations
  implicit none
  double precision, external :: integrand
  double precision, parameter :: expected=2.5d0*(1d0+2d0+4d0)

  nchans=3
  iconfigs(1:3)=[1,2,3]
  ndim=4
  ncalls0=-1
  itmax=6
  imode=0
  n_ord_virt=0
  ifold=1
  accuracy=0.03d0
  min_virt_fraction_mint=0.01d0
  wgt_mult=1d0
  fixed_order=.false.
  nlo_ps=.true.
  only_virt=.false.
  call born_spread_configure(.true.,5,2,1,ndim)
  call mint(integrand)
  if (.not.born_spread_ready) error stop 'calibration did not run'
  if (bad_iterations.ne.2.or..not.recovered) error stop 'grid recovery did not run'
  if (any(recovered_channel_calls.eq.0)) error stop 'recovery lost an integration channel'
  if (abs(ans(2,0)-expected).gt.max(5d0*unc(2,0),0.03d0*expected)) &
       error stop 'recovery lost part of the integral'
  if (unc(2,0).le.0d0.or.unc(2,0).gt.0.03d0*expected) &
       error stop 'unexpected recovered integration error'
  write(*,*) 'PASS MINT calibrated-table recovery',ans(2,0),unc(2,0)
  write(*,*) 'recovered channel calls',recovered_channel_calls
end program check_recovery

double precision function integrand(x,vol,ifl,weights)
  use mint_module, only: ndimmax,nintegrals,pass_cuts_check,ichan,nit,itmax,born_spread_ready
  use recovery_observations
  implicit none
  double precision :: x(ndimmax),vol,weights(nintegrals),value
  double precision, parameter :: channel_value(3)=[1d0,2d0,4d0]
  integer :: ifl

  value=channel_value(ichan)*(2d0+x(1))
  if (born_spread_ready.and.ifl.eq.2) then
     if (bad_iterations.eq.2.and.nit.eq.1) then
        recovered=.true.
        ! Inspect the recovered pilot immediately. Later adaptive iterations
        ! must not be able to conceal a missing channel in that first scan.
        itmax=1
     endif
     if (recovered) then
        recovered_channel_calls(ichan)=recovered_channel_calls(ichan)+1
     elseif (nit.ge.4.and.nit.ne.previous_bad_iteration) then
        ! One large outlier makes each iteration fail the ordinary 30% error
        ! criterion. Iterations 4 and 5 exercise its real consecutive-failure
        ! recovery, rather than calling the reset helper directly.
        value=1d12
        previous_bad_iteration=nit
        bad_iterations=bad_iterations+1
     endif
  endif
  weights=0d0
  weights(1)=value*vol
  weights(2)=weights(1)
  pass_cuts_check=.true.
  integrand=weights(1)
end function integrand

double precision function ran2()
  implicit none
  integer, parameter :: wide=selected_int_kind(18)
  integer(kind=wide), save :: seed=12345
  seed=mod(16807_wide*seed,2147483647_wide)
  ran2=dble(seed)/2147483647d0
end function ran2
