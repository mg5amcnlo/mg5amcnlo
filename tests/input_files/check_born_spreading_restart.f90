program check_restart
  use mint_module
  implicit none
  character(len=16) :: arg
  integer :: enabled,i
  double precision, external :: integrand
  double precision :: expected

  call get_command_argument(1,arg)
  read(arg,*) nchans
  call get_command_argument(2,arg)
  read(arg,*) enabled
  ndim=4
  ncalls0=-1
  ! End after one iteration so that later iterations cannot hide a bad restart.
  itmax=1
  imode=0
  n_ord_virt=0
  ifold=1
  do i=1,nchans
     iconfigs(i)=i
  enddo
  accuracy=0.03d0
  min_virt_fraction_mint=0.01d0
  wgt_mult=1d0
  fixed_order=.false.
  nlo_ps=.true.
  only_virt=.false.
  call born_spread_configure(enabled.eq.1,5,2,1,ndim)
  ! No negative weights are supplied to calibration, giving a unit table.
  ! The production calibration and restart run without any source changes.
  call mint(integrand)
  expected=2.5d0*nchans
  if (abs(ans(2,0)-expected).gt.0.1d0*expected) stop 1
  if (unc(2,0).le.0d0.or.unc(2,0).gt.0.1d0*expected) stop 2
  if (born_spread_ready.neqv.(enabled.eq.1)) stop 3
  write(*,*) 'PASS MINT restart normalization',ans(2,0),unc(2,0)
end program check_restart

double precision function integrand(x,vol,ifl,weights)
  use mint_module, only: ndimmax,nintegrals,pass_cuts_check
  implicit none
  double precision :: x(ndimmax),vol,weights(nintegrals)
  integer :: ifl
  weights=0d0
  weights(1)=(2d0+x(1))*vol
  weights(2)=weights(1)
  pass_cuts_check=.true.
  integrand=weights(1)
end function integrand

double precision function ran2()
  ! Deterministic Park-Miller generator; use wide integers for the product.
  implicit none
  integer, parameter :: wide=selected_int_kind(18)
  integer(kind=wide), save :: seed=12345
  seed=mod(16807_wide*seed,2147483647_wide)
  ran2=dble(seed)/2147483647d0
end function ran2
