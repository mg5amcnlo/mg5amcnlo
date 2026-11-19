! Regression for the physical massless-FI point that stopped C_LOC/GF1.0_13.
! Link the real legacy external dot as well: its 1e-6 GeV^2 clipping must
! not enter the dimensionless local subtraction-scale ratios.
program check_resonance_subtraction_scales
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  integer, parameter :: qp=selected_real_kind(30)
  double precision :: total(0:3),q(0:3),pj(0:3),phat(0:3),scales(4),factor
  real(qp) :: tq(0:3),qq(0:3),jq(0:3),kq(0:3),reference(4),s,m2,kdot,jdot
  double precision, external :: dot
  logical :: pass
  integer :: power
  character(len=32) :: mode
  call get_command_argument(1,mode)
  total=[173.01498736478135d0,0d0,0d0,0d0]
  q=[.11225686582671314d0,1.8238986729647572d-4, &
     -2.8876376934121680d-4,.11225297352198853d0]
  pj=[1.4986715643670656d-2,1.8238986729647572d-4, &
      -2.8876376934121680d-4,1.4982823338946041d-2]
  phat=[9.7270149750516927d-2,-2.6429281950793353d-12, &
        4.1607640815430868d-12,9.7270149750516927d-2]
  ! This is the production helper's precise failure trigger, not a zero
  ! invariant or an invalid momentum. Confirm the legacy dot is linked.
  if(dot(q,q).ne.0d0.or.dot(phat,q).ne.0d0.or.dot(pj,q).ne.0d0) &
       error stop 'fixture must exercise legacy dot clipping'
  tq=real(total,qp); qq=real(q,qp); jq=real(pj,qp); kq=real(phat,qp)
  s=mdot(tq,tq); m2=mdot(qq,qq); kdot=mdot(kq,qq); jdot=mdot(jq,qq)
  if(min(s,m2,kdot,jdot).le.0)error stop 'invalid reference point'
  reference(1)=(s/m2)*kdot/mdot(kq,tq)
  reference(2)=(s/m2)*jdot/mdot(jq,tq)
  reference(3)=(s/m2)/reference(2)**2
  reference(4)=log(reference(2)/reference(1))
  if(mode.eq.'point')then
    call check(1d0)
  elseif(mode.eq.'units')then
    do power=-9,9,3
      factor=10d0**power
      call check(factor)
    enddo
  elseif(mode.eq.'invalid')then
    q=0d0
    call resonance_subtraction_scales(total,q,pj,phat, &
         scales(1),scales(2),scales(3),scales(4),pass)
    if(pass)error stop 'zero recoil invariant accepted'
  else
    error stop 'unknown test mode'
  endif
  print *, 'PASS '//trim(mode)
contains
  real(qp) function mdot(a,b)
    real(qp),intent(in) :: a(0:3),b(0:3)
    mdot=a(0)*b(0)-sum(a(1:3)*b(1:3))
  end function
  subroutine check(factor)
    double precision,intent(in) :: factor
    call resonance_subtraction_scales(factor*total,factor*q, &
         factor*pj,factor*phat,scales(1),scales(2),scales(3),scales(4),pass)
    if(.not.pass.or..not.all(ieee_is_finite(scales))) &
         error stop 'valid small-invariant local scales rejected'
    if(maxval(abs(real(scales,qp)/reference-1)).gt.1e-9_qp)then
      print *, 'scales',scales,'reference',reference,'factor',factor
      error stop 'incorrect local subtraction-scale ratios'
    endif
  end subroutine
end program
