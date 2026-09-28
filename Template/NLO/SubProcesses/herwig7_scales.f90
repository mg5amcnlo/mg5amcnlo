! Herwig++/Herwig7 angular-shower starting scales for ordinary MC@NLO.
! Independent implementation of the kinematics in Gieseke, Stephens and
! Webber, JHEP 12 (2003) 045, hep-ph/0310083. See herwig7_scales.README.
! No Herwig library or random-number generator is required.
module herwig7_scales
  implicit none
  private
  integer, parameter :: dp = selected_real_kind(15,307)
  integer, parameter, public :: hw7_ok=0, hw7_bad_input=1, hw7_numerical=2
  public :: herwig7_starting_scales

contains

  ! Born ordering, with two massless incoming particles at indices 1,2.
  ! All momenta have positive energy and components (E,px,py,pz).
  ! The first matrix index is the emitter. Unconnected entries are -1.
  !
  ! SCALUP is a pT veto, not the angular evolution variable qtilde.
  ! Return min(hard_scale, kinematic pT ceiling) for each connection.
  ! The optional angular_scales returns the UNCAPPED qtilde starting
  ! scales. The full shower support also depends on z; these pT ceilings
  ! must never replace the angular dead-zone test in the subtraction.
  ! Masses are on-shell model masses, without shower infrared cutoffs.
  subroutine herwig7_starting_scales(nborn,p,mass,connected,hard_scale, &
                                     scales,status,angular_scales)
    integer, intent(in) :: nborn
    real(dp), intent(in) :: p(0:3,nborn),mass(nborn),hard_scale
    logical, intent(in) :: connected(nborn,nborn)
    real(dp), intent(out) :: scales(nborn,nborn)
    integer, intent(out) :: status
    real(dp), intent(out), optional :: angular_scales(nborn,nborn)
    real(dp) :: dotij,q2,a,mi2,mj2,lambda,threshold,tolerance
    real(dp) :: ratio,root,z,one_minus_z,pt_bound,total(0:3)
    integer :: i,j

    scales=-1._dp
    if (present(angular_scales)) angular_scales=-1._dp
    status=hw7_bad_input
    if (nborn < 3) return
    if (.not.(hard_scale > 0._dp .and. hard_scale <= huge(1._dp))) return
    if (.not.all(abs(p) <= huge(1._dp)) .or. any(p(0,:) <= 0._dp)) return
    if (.not.all(mass >= 0._dp .and. mass <= sqrt(huge(1._dp)))) return
    if (any(mass(1:2) /= 0._dp)) return
    do i=1,nborn
      if (connected(i,i)) return
    end do

    status=hw7_numerical
    do i=1,nborn
      mi2=mass(i)**2
      do j=1,nborn
        if (.not.connected(i,j)) cycle
        mj2=mass(j)**2
        dotij=p(0,i)*p(0,j)-sum(p(1:3,i)*p(1:3,j))
        tolerance=64._dp*epsilon(1._dp)*max(p(0,i)*p(0,j),mi2,mj2)
        if (.not.(dotij >= -tolerance .and. dotij <= huge(1._dp))) return

        ! qtilde^2: II and IF use 2 p_i.p_j; FI adds the emitter mass.
        ! FF uses the symmetric partition of angular phase space.
        a=2._dp*max(0._dp,dotij)
        if (i > 2) then
          a=a+mi2
          if (j > 2) then
            total=p(:,i)+p(:,j)
            q2=total(0)**2-sum(total(1:3)**2)
            threshold=(mass(i)+mass(j))**2
            tolerance=64._dp*epsilon(1._dp)*max(total(0)**2,threshold)
            if (.not.(q2 >= threshold-tolerance .and. q2 <= huge(1._dp))) return
            q2=max(q2,threshold)
            lambda=sqrt(max(0._dp,q2-threshold))* &
                   sqrt(max(0._dp,q2-(mass(i)-mass(j))**2))
            a=0.5_dp*(q2+mi2-mj2+lambda)
          end if
        end if
        if (.not.(a >= 0._dp .and. a <= huge(1._dp))) return
        if (present(angular_scales)) angular_scales(i,j)=sqrt(a)

        ! ISR: pT=(1-z)*qtilde, so qtilde is a conservative envelope;
        ! Bjorken-x limits and finite shower masses can reduce it further.
        pt_bound=sqrt(a)
        if (i > 2) then
          ! FSR: pT^2=(1-z)^2*(z^2*qtilde^2-m_i^2).
          ! Its maximum is at z=(1+sqrt(1+8*m_i^2/qtilde^2))/4.
          ! At that stationary point pT^2=qtilde^2*z*(1-z)^3.
          ! Rationalize 1-z to retain accuracy near qtilde=m_i.
          pt_bound=0._dp
          if (a > mi2) then
            ratio=mi2/a
            root=sqrt(1._dp+8._dp*ratio)
            z=(1._dp+root)/4._dp
            one_minus_z=2._dp*((a-mi2)/a)/(3._dp+root)
            pt_bound=sqrt(a*z*one_minus_z**3)
          end if
        end if
        scales(i,j)=min(hard_scale,pt_bound)
      end do
    end do
    status=hw7_ok
  end subroutine herwig7_starting_scales
end module herwig7_scales
