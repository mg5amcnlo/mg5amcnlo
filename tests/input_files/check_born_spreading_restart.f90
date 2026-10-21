program check_restart
  use mint_module
  implicit none
  character(len=16) :: arg
  integer :: enabled,pooled,i,unit
  double precision, external :: integrand
  double precision :: expected

  call get_command_argument(1,arg)
  read(arg,*) nchans
  call get_command_argument(2,arg)
  read(arg,*) enabled
  pooled=0
  if(command_argument_count().gt.2)then
     call get_command_argument(3,arg)
     read(arg,*) pooled
  endif
  use_pooled_virtual=pooled.eq.1
  if(use_pooled_virtual)then
     open(newunit=unit,file='virtual_pool_mode.dat',status='replace')
     write(unit,*) 1
     close(unit)
  endif
  ndim=4
  ncalls0=-1
  ! End after one iteration so that later iterations cannot hide a bad restart.
  itmax=1
  imode=0
  n_ord_virt=pooled
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
  use mint_module, only: ndimmax,nintegrals,pass_cuts_check,use_pooled_virtual, &
       born_spread_phase,average_virtual,ichan,pooled_virtual_prepare, &
       pooled_virtual_mark_sampled,pooled_virtual_ratio
  use virtual_pooled, only: pooled_virtual_set_helicity_weights
  implicit none
  double precision :: x(ndimmax),vol,weights(nintegrals)
  double precision :: p(0:3,4),ratio(1),dipoles(4,4),helicity_weights(1,1)
  integer :: ifl
  weights=0d0
  weights(1)=(2d0+x(1))*vol
  weights(2)=weights(1)
  if(use_pooled_virtual.and.born_spread_phase.ne.1.and.born_spread_phase.ne.2)then
     if(ifl.eq.0)then
        p(:,1)=[250d0,0d0,0d0,250d0]
        p(:,2)=[250d0,0d0,0d0,-250d0]
        p(:,3)=[250d0,250d0,0d0,0d0]
        p(:,4)=[250d0,-250d0,0d0,0d0]
        ratio=average_virtual(1:1,ichan)
        call pooled_virtual_prepare(p,[0d0,0d0,0d0,0d0],[.true.,.true.,.false.,.false.], &
             [21,21,21,21],10000d0,ratio,10000d0,.true.,[1d0],[8,8,8,8], &
             surrogate=[1d0,0d0,0d0,0d0],supported=.true.,channel=ichan)
        dipoles=0d0;helicity_weights=1d0
        call pooled_virtual_set_helicity_weights(helicity_weights,dipoles)
        call pooled_virtual_mark_sampled(1d0)
     endif
     weights(8)=(1d0+x(1))*vol
     weights(7)=(3d0-pooled_virtual_ratio(1))*weights(8)
     weights(3)=weights(7)
     weights(5)=abs(weights(3))
     weights(6)=weights(8)
  endif
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
