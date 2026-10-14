! Kinematic carrier for a residual envelope near massive-pair thresholds.
! This module evaluates no matrix elements and consumes no random numbers.
module virtual_envelope_carrier
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  integer, parameter :: dp=kind(1d0)
  public :: virtual_envelope_carrier_factor
contains
  pure recursive subroutine virtual_envelope_carrier_factor(p,mass,incoming,factor,valid)
    real(dp), intent(in) :: p(0:,:),mass(:)
    logical, intent(in) :: incoming(:)
    real(dp), intent(out) :: factor
    logical, intent(out) :: valid
    real(dp) :: scale,a(0:3),b(0:3),mi,mj,dot,mm,delta,error
    real(dp) :: spatial,mass2,vrel,inverse_v,proposed
    integer :: i,j,n

    ! A false status requires the caller's exact fallback. In particular, do
    ! not impose a velocity floor: that would change the meaning of a bound.
    factor=1d0
    valid=.false.
    n=size(mass)
    if(size(p,1)/=4.or.size(p,2)/=n.or.size(incoming)/=n) return
    if(count(incoming)<1.or.count(incoming)>2) return
    if(.not.all(ieee_is_finite(p)).or..not.all(ieee_is_finite(mass))) return
    if(any(mass<0d0)) return

    ! Reject inconsistent massive momenta, while allowing roundoff in their
    ! on-shell relation. Scaling first also permits extreme common unit changes.
    do i=1,n
       if(incoming(i).or.mass(i)==0d0) cycle
       if(p(0,i)<=0d0) return
       scale=max(maxval(abs(p(:,i))),mass(i))
       a=p(:,i)/scale
       mi=mass(i)/scale
       spatial=sqrt(sum(a(1:3)**2))
       mass2=(a(0)-spatial)*(a(0)+spatial)
       error=64d0*epsilon(1d0)*(a(0)**2+spatial**2+mi**2)
       if(abs(mass2-mi**2)>error) return
    enddo

    proposed=1d0
    do i=1,n-1
       if(incoming(i).or.mass(i)==0d0) cycle
       do j=i+1,n
          if(incoming(j).or.mass(j)==0d0) cycle
          scale=max(maxval(abs(p(:,i))),maxval(abs(p(:,j))),mass(i),mass(j))
          a=p(:,i)/scale
          b=p(:,j)/scale
          mi=mass(i)/scale
          mj=mass(j)/scale
          dot=a(0)*b(0)-sum(a(1:3)*b(1:3))
          mm=mi*mj
          delta=dot-mm
          error=64d0*epsilon(1d0)*(abs(a(0)*b(0))+sum(abs(a(1:3)*b(1:3)))+mm)
          ! Co-moving pairs and velocities below floating-point resolution
          ! have no reliably finite carrier. Small negative roundoff receives
          ! the same fallback as exact threshold, never a clipped finite value.
          if(dot<=0d0.or.delta<=error) return
          vrel=sqrt((delta/dot)*((dot+mm)/dot))
          if(.not.ieee_is_finite(vrel).or.vrel<=0d0) return
          inverse_v=1d0/vrel
          if(.not.ieee_is_finite(inverse_v)) return
          if(inverse_v>huge(proposed)-proposed) return
          proposed=proposed+inverse_v
       enddo
    enddo
    factor=proposed
    valid=.true.
  end subroutine virtual_envelope_carrier_factor
end module virtual_envelope_carrier
