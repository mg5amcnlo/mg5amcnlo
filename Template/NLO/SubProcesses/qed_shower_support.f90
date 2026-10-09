! QED reference dipoles of the PYTHIA Simple showers. These are directed
! radiator/recoiler pairs, not electric-charge correlations in the soft ME.
!
! Scope: a single hard system with two incoming legs, no rescattering or
! resonance decays. All momenta have positive energies, incoming legs come
! first, and PDG codes/charges are physical (not all-outgoing conventions).
! The reference partner is required even with global recoil: PYTHIA uses
! its invariant mass for TimeShower:limitPTmaxGlobal. This module does not
! decide whether a particular event is eligible for global recoil.
module qed_shower_support
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: build_pythia8_qed_dipoles
  public :: pythia8_qed_radiator
contains

  pure recursive logical function pythia8_qed_radiator(pdg, charge, is_initial) result(active)
    integer, intent(in) :: pdg
    double precision, intent(in) :: charge
    logical, intent(in), optional :: is_initial
    integer :: id

    ! SimpleTimeShower also radiates photons from charged resonances via
    ! QEDshowerByOther. Include W bosons without assuming that an arbitrary
    ! charged BSM particle has PYTHIA's isResonance flag. SimpleSpaceShower
    ! does not provide the corresponding incoming-W evolution.
    id=abs(pdg)
    active=pdg.eq.22.or.(charge.ne.0d0.and. &
         ((id.ge.1.and.id.le.6).or.id.eq.11.or.id.eq.13.or.id.eq.15.or.id.eq.24))
    if (present(is_initial)) then
      if (is_initial.and.id.eq.24) active=.false.
    endif
  end function pythia8_qed_radiator

  pure recursive subroutine build_pythia8_qed_dipoles(p,pdg,charge,mass,nincoming, &
       dipoles,ierr,allow_beam_recoil)
    double precision, intent(in) :: p(0:,:),charge(:),mass(:)
    integer, intent(in) :: pdg(:),nincoming
    logical, intent(out) :: dipoles(:,:)
    integer, intent(out) :: ierr
    logical, intent(in), optional :: allow_beam_recoil
    integer :: n,rad,rec,j,first,stage
    double precision :: distance,best
    logical :: beam_recoil,candidate

    ! ierr=1: unsupported incoming multiplicity or inconsistent shapes.
    ! ierr=2: nonfinite inputs or negative masses/energies.
    ! ierr=3: an active final radiator has no available recoil partner.
    ! On error the complete output is cleared, to avoid partial use.
    dipoles=.false.
    ierr=1
    n=size(pdg)
    if (nincoming.ne.2.or.n.le.nincoming) return
    if (size(p,1).ne.4.or.size(p,2).ne.n.or.size(charge).ne.n.or. &
         size(mass).ne.n.or.size(dipoles,1).ne.n.or. &
         size(dipoles,2).ne.n) return
    ierr=2
    if (.not.all(ieee_is_finite(p)).or. &
         .not.all(ieee_is_finite(charge)).or. &
         .not.all(ieee_is_finite(mass))) return
    if (any(mass.lt.0d0).or.any(p(0,:).lt.0d0)) return

    beam_recoil=.true.
    if (present(allow_beam_recoil)) beam_recoil=allow_beam_recoil
    first=1
    if (.not.beam_recoil) first=nincoming+1

    do rad=1,n
      if (.not.pythia8_qed_radiator(pdg(rad),charge(rad),rad.le.nincoming)) cycle
      if (rad.le.nincoming) then
        ! SimpleSpaceShower::prepare uses the other incoming leg for
        ! QED evolution, independently of that leg's electric charge.
        dipoles(rad,3-rad)=.true.
        cycle
      endif

      rec=0
      ! SimpleTimeShower::setupQEDdip searches successive preference
      ! classes. Minimize p_rad.p_rec - m_rad*m_rec within each class;
      ! the second search weights it by the recoiler's charge squared.
      ! A factor of nine from PYTHIA's integer chargeType cancels.
      do stage=1,3
        best=huge(1d0)
        do j=first,n
          if (j.eq.rad) cycle
          select case(stage)
          case(1)
            candidate=(j.le.nincoming.and.pdg(j).eq.pdg(rad)).or. &
                 (j.gt.nincoming.and.pdg(j).eq.-pdg(rad))
          case(2)
            candidate=charge(j).ne.0d0
          case(3)
            candidate=j.gt.nincoming
          end select
          if (.not.candidate) cycle
          distance=p(0,rad)*p(0,j)-sum(p(1:3,rad)*p(1:3,j)) &
               -mass(rad)*mass(j)
          if (stage.eq.2) distance=distance/(charge(j)*charge(j))
          ! Strict comparison retains the first partner at equal distance,
          ! as in PYTHIA's event-order traversal.
          if (distance.lt.best) then
            best=distance
            rec=j
          endif
        enddo
        if (rec.ne.0) exit
      enddo
      if (rec.eq.0) then
        dipoles=.false.
        ierr=3
        return
      endif
      dipoles(rad,rec)=.true.
    enddo
    ierr=0
  end subroutine build_pythia8_qed_dipoles
end module qed_shower_support
