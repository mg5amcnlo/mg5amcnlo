! Controlled matrix elements for the actual limit-test routines. Inactive
! coupling orders must stay zero; each active order has a distinct amplitude.
module limit_fixture
  implicit none
  integer, parameter :: nexternal=5, nstep=10, namp=3
  real(8) :: mc_amplitudes(namp), amp_split(namp)
  common /test_amp_split/ amp_split
  logical :: softtest, colltest
  common /sctests/ softtest, colltest
  integer :: i_fks, j_fks, nerr(0:namp)
  common /fks_indices/ i_fks, j_fks
  common /c_nerr/ nerr
  real(8) :: p1_cnt(0:3,nexternal,-2:2), wgt_cnt(-2:2), &
       pswgt_cnt(-2:2), jac_cnt(-2:2)
  common /counterevnts/ p1_cnt, wgt_cnt, pswgt_cnt, jac_cnt
  real(8) :: xi_ev, y_ev, p_i_ev(0:3), p_i_cnt(0:3,-2:2)
  common /fksvariables/ xi_ev, y_ev, p_i_ev, p_i_cnt
  real(8) :: xi_cnt(-2:2)
  common /cxiifkscnt/ xi_cnt
end module limit_fixture

program check_soft_col_limits
  use limit_fixture
  implicit none
  character(32) :: name
  real(8) :: amp(namp,nstep), weights(nstep), xp(0:3,nexternal+1,nstep), &
       limit(namp), wlimit, plimit(0:3,nexternal+1), x(99), values(nstep), &
       target, scales(4)
  integer :: i, j, iflag, iret

  call get_command_argument(1,name)
  x=0.5d0
  xp=0d0
  plimit=0d0
  weights=1d0
  wlimit=1d0
  i_fks=5
  j_fks=1
  scales=[1d0,1d-20,1d20,-13d0]
  do iflag=0,1
    softtest=iflag.eq.0
    colltest=iflag.eq.1
    select case(trim(name))
    case('roundoff')
      do j=1,size(scales)
        target=scales(j)
        do i=1,nstep
          values(i)=target*(1d0+(-1d0)**i*8d0*epsilon(1d0))
        enddo
        call expect_convergence(0)
      enddo
    case('convergence')
      do j=1,size(scales)
        target=scales(j)
        values=target*(1d0+1d-10)
        call expect_convergence(1)
        values=target*1.01d0
        call expect_convergence(1)
        values=0d0
        call expect_convergence(1)
        ! The usual linear-soft and square-root-collinear convergence.
        do i=1,nstep
          values(i)=target*(1d0+10d0**(-dble(i)/(iflag+1)))
        enddo
        call expect_convergence(0)
        ! Approaching the target too slowly must still fail.
        do i=1,nstep
          values(i)=target*(1d0+0.01d0*0.9d0**i)
        enddo
        call expect_convergence(1)
      enddo
      target=0d0
      values=0d0
      call expect_convergence(0)
      do i=1,nstep
        values(i)=10d0**(-dble(i)/(iflag+1))
      enddo
      call expect_convergence(0)
      values=1d-20
      call expect_convergence(1)
    case('missing_mc','native_sector')
      ! The exact sector limit is (1/4)*[0,5,7] times its PS weight 3.
      ! A real-point PS weight of 2 needs MC amplitudes [0,15/8,21/8].
      mc_amplitudes=[0d0,15d0/8d0,21d0/8d0]
      if (trim(name).eq.'missing_mc') mc_amplitudes=0d0
      do i=1,nstep
        call compute_towards_limit(1,x,amp(:,i),weights(i),xp(:,:,i))
      enddo
      call compute_in_the_limit(1,x,limit,wlimit,plimit)
      if (any(limit.ne.[0d0,15d0/4d0,21d0/4d0])) stop 2
      if (any(amp(1,:).ne.0d0)) stop 3
      nerr=0
      call check_limits(nstep,amp,weights,xp,limit,wlimit,plimit,1)
      if (trim(name).eq.'missing_mc') then
        if (any(amp.ne.0d0)) stop 4
        if (any(nerr.ne.[1,0,1,1])) stop 5
      else
        if (any(nerr.ne.0)) stop 6
      endif
    case('fixed_order')
      call compute_towards_limit(2,x,amp(:,1),weights(1),xp(:,:,1))
      call compute_in_the_limit(2,x,limit,wlimit,plimit)
      if (any(amp(:,1).ne.[0d0,10d0,14d0])) stop 7
      if (any(limit.ne.[0d0,15d0,21d0])) stop 8
    case default
      stop 9
    end select
  enddo
  print *, 'PASS '//trim(name)
contains
  subroutine expect_convergence(expected)
    integer, intent(in) :: expected
    call checkres(values,target,weights,wlimit,xp,plimit, &
         iflag,nstep,1,nexternal,i_fks,j_fks,iret)
    if (iret.ne.expected) then
      print *, trim(name), iflag, target, 'return code', iret, 'expected', expected
      print *, values
      stop 10
    endif
  end subroutine expect_convergence
end program check_soft_col_limits

subroutine generate_momenta(ndim,iconfig,wgt,x,p,p_lab,p_cms)
  use limit_fixture
  implicit none
  integer :: ndim, iconfig
  real(8) :: wgt, x(99), p(0:3,nexternal), p_lab(0:3,nexternal), p_cms(0:3,nexternal)
  wgt=2d0
  jac_cnt=3d0
  wgt_cnt=1d0
  pswgt_cnt=1d0
  p=0d0
  p_lab=0d0
  p_cms=0d0
  p1_cnt=0d0
  p_i_ev=0d0
  p_i_cnt=0d0
  xi_ev=0.2d0
  xi_cnt=0.2d0
  y_ev=0.4d0
end subroutine generate_momenta

subroutine sreal(p,xi,y,fx)
  use limit_fixture
  implicit none
  real(8) :: p(0:3,nexternal), xi, y, fx
  amp_split=[0d0,5d0,7d0]
  fx=sum(amp_split)
end subroutine sreal

subroutine compute_MC_subt_term_test(p,p_cms,wgt)
  use limit_fixture
  implicit none
  real(8) :: p(0:3,nexternal), p_cms(0:3,nexternal), wgt
  amp_split=mc_amplitudes
end subroutine compute_MC_subt_term_test

double precision function fks_Sij(p,i,j,xi,y)
  use limit_fixture
  implicit none
  real(8) :: p(0:3,nexternal), xi, y
  integer :: i, j
  fks_Sij=0.25d0
end function fks_Sij

subroutine set_cms_stuff(icnt)
  implicit none
  integer :: icnt
end subroutine set_cms_stuff

subroutine fks_inc_chooser()
end subroutine fks_inc_chooser

subroutine update_coltype_and_charge(nfks,i,j)
  implicit none
  integer :: nfks, i, j
end subroutine update_coltype_and_charge
