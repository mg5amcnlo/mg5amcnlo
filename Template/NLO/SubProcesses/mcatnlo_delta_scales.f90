! Deterministic scale reconstruction for the default MC@NLO-Delta interface.
! Formulae ported from PYTHIA 8.313, History::pTLund and
! Merging::clusterAndStore/getDipoles (Stefan Prestel, PYTHIA authors).
! SPDX-License-Identifier: GPL-2.0-or-later
!
! This module needs neither PYTHIA nor a random number generator.
! Momenta are (E,px,py,pz), with positive-energy incoming particles.
! See mcatnlo_delta_scales.README for scope and provenance, and
! mcatnlo_delta_scales.COPYING for the license.
module mcatnlo_delta_scales
  implicit none
  private
  integer, parameter, public :: dp = selected_real_kind(15,307)
  integer, parameter, public :: delta_ok = 0, delta_bad_input = 1, delta_numerical = 2
  public :: delta_scale_matrices, pythia_pt_lund

contains

  real(dp) function dot4(a,b)
    real(dp), intent(in) :: a(0:3), b(0:3)
    ! Preserve the subtraction order in PYTHIA's Vec4.
    dot4 = a(0)*b(0) - a(1)*b(1) - a(2)*b(2) - a(3)*b(3)
  end function dot4

  logical function qcd_parton(id)
    integer, intent(in) :: id
    qcd_parton = (abs(id) >= 1 .and. abs(id) <= 6) .or. id == 21
  end function qcd_parton

  ! A single History::pTLund evaluation in runtimeAMCATNLOInterface mode.
  ! t is pT in GeV, NOT pT**2. Keep PYTHIA's raw sentinel values here.
  ! charm_mass/bottom_mass are particleData.m0, not event invariant masses.
  ! w_parent_mass is only relevant when abs(id_emt)==24; the QCD matrix
  ! routine never uses that branch. It corresponds to m0(idRadBef).
  subroutine pythia_pt_lund(pr, pe, pk, id_rad, id_emt, rad_final, rec_final, &
                            charm_mass, bottom_mass, t, status, w_parent_mass)
    real(dp), intent(in) :: pr(0:3), pe(0:3), pk(0:3)
    integer, intent(in) :: id_rad, id_emt
    logical, intent(in) :: rad_final, rec_final
    real(dp), intent(in) :: charm_mass, bottom_mass
    real(dp), intent(out) :: t
    integer, intent(out) :: status
    real(dp), intent(in), optional :: w_parent_mass
    real(dp) :: q(0:3), r(0:3), total(0:3), qsq, mr2, me2, mb2
    real(dp) :: mfinal2, mar2, ratio, mdip2, x1, x2, qbr2, qar2
    real(dp) :: discriminant, lambda13, k1, k3, z, pt2, mh2

    status = delta_ok
    t = 0._dp
    if (.not.all(abs(pr) <= huge(1._dp)) .or. &
        .not.all(abs(pe) <= huge(1._dp)) .or. &
        .not.all(abs(pk) <= huge(1._dp))) then
      status = delta_bad_input
      return
    end if
    if (.not.(charm_mass >= 0._dp .and. bottom_mass >= 0._dp)) then
      status = delta_bad_input
      return
    end if

    if (rad_final) then
      q = pr + pe
      qsq = dot4(q,q)
    else
      q = pr - pe
      qsq = -dot4(q,q)
    end if
    if (abs(qsq) < 1.e-6_dp) then
      t = 1.e-6_dp
      return
    end if

    mr2 = dot4(pr,pr)
    me2 = dot4(pe,pe)
    mb2 = 0._dp
    if (abs(id_rad) /= 21 .and. abs(id_rad) /= 22 .and. &
        abs(id_emt) /= 24 .and. abs(id_rad) /= abs(id_emt)) then
      mb2 = mr2
    else if (abs(id_emt) == 24) then
      if (present(w_parent_mass)) mb2 = w_parent_mass**2
    else if (.not.rad_final) then
      if (abs(id_rad) == 21 .and. abs(id_emt) /= 21) mb2 = me2
    end if

    if (rad_final) then
      r = pk
      if (.not.rec_final) then
        total = pr + pk + pe
        mfinal2 = dot4(total,total)
        mar2 = mfinal2 - 2._dp*qsq + 2._dp*mb2
        ! The original returns its sentinel after rescaling; returning first
        ! avoids an unused division by zero on an already rejected point.
        if (qsq > mar2) then
          t = 1.e10_dp
          return
        end if
        if (mar2 - mb2 == 0._dp) then
          status = delta_numerical
          return
        end if
        ratio = (qsq - mb2)/(mar2 - mb2)
        if (1._dp + ratio == 0._dp) then
          status = delta_numerical
          return
        end if
        r = r*((1._dp - ratio)/(1._dp + ratio))
      end if
      total = pr + r + pe
      mdip2 = dot4(total,total)
      discriminant = (qsq - mr2 - me2)**2 - 4._dp*mr2*me2
      if (mdip2 == 0._dp .or. discriminant <= 0._dp) then
        status = delta_numerical
        return
      end if
      x1 = 2._dp*dot4(total,pr)/mdip2
      x2 = 2._dp*dot4(total,r)/mdip2
      lambda13 = sqrt(discriminant)
      k1 = (qsq - lambda13 + (me2 - mr2))/(2._dp*qsq)
      k3 = (qsq - lambda13 - (me2 - mr2))/(2._dp*qsq)
      if (1._dp-k1-k3 == 0._dp .or. 2._dp-x2 == 0._dp) then
        status = delta_numerical
        return
      end if
      z = 1._dp/(1._dp-k1-k3)*(x1/(2._dp-x2)-k3)
      pt2 = z*(1._dp-z)*(qsq-mb2)
    else
      ! The FSR-only x1, x2 and lambda13 expressions in the C++ routine
      ! are unused in this branch and need not be evaluated.
      q = pr - pe + pk
      qbr2 = dot4(q,q)
      q = pr + pk
      qar2 = dot4(q,q)
      if (qbr2 < 0._dp) then
        t = 1.e-5_dp
        return
      end if
      if (qar2 == 0._dp) then
        status = delta_numerical
        return
      end if
      z = qbr2/qar2
      pt2 = (1._dp-z)*qsq
      mh2 = 0._dp
      if ((abs(id_rad) == 4 .or. abs(id_emt) == 4) .and. &
          abs(id_rad) /= abs(id_emt)) then
        mh2 = charm_mass**2
      else if ((abs(id_rad) == 5 .or. abs(id_emt) == 5) .and. &
               abs(id_rad) /= abs(id_emt)) then
        mh2 = bottom_mass**2
      end if
      if (pt2 < 2._dp*mh2) pt2 = (qsq+mh2)*(1._dp-qbr2/qar2)
    end if

    if (pt2 < 0._dp) then
      t = 1.e-6_dp
    else if (pt2 <= huge(1._dp)) then
      t = sqrt(pt2)
    else
      status = delta_numerical
    end if
  end subroutine pythia_pt_lund

  ! Born colour tags are in Born order (real order with emitted removed).
  ! Return arrays use MG5's real labels: 1..nreal inside zero-based buffers.
  ! Row/column 0, the emitted leg, diagonal, and absent dipoles stay -1/true.
  ! This assumes the supplied Born flow is the valid FKS underlying flow;
  ! it does not search for histories or reconstruct Born momenta.
  subroutine delta_scale_matrices(nreal, emitted, p, id_real, born_col, &
                                 charm_mass, bottom_mass, scales, masses, dead, status)
    integer, intent(in) :: nreal, emitted
    real(dp), intent(in) :: p(0:3,nreal), charm_mass, bottom_mass
    integer, intent(in) :: id_real(nreal), born_col(2,nreal-1)
    real(dp), intent(out) :: scales(0:,0:), masses(0:,0:)
    logical, intent(out) :: dead(0:,0:)
    integer, intent(out) :: status
    integer :: i, j, ir, jr, kr, c, other, count, tag, local_status
    real(dp) :: raw, qdip(0:3)
    logical :: connected(nreal-1,nreal-1)

    scales = -1._dp
    masses = -1._dp
    dead = .true.
    status = delta_bad_input
    if (nreal < 4 .or. emitted <= 2 .or. emitted > nreal) return
    if (minval(shape(scales)) <= nreal .or. minval(shape(masses)) <= nreal .or. &
        minval(shape(dead)) <= nreal) return
    if (.not.qcd_parton(id_real(emitted))) return
    if (any(born_col < 0)) return
    if (.not.all(abs(p) <= huge(1._dp))) return
    if (.not.(charm_mass >= 0._dp .and. bottom_mass >= 0._dp)) return

    ! Same-side colour lines join colour to anticolour; across initial/final
    ! states they join the same slot. Validate each nonzero colour endpoint.
    connected = .false.
    do i=1,nreal-1
      ir = i
      if (ir >= emitted) ir = ir+1
      if (any(born_col(:,i) /= 0) .and. .not.qcd_parton(id_real(ir))) return
      do c=1,2
        tag = born_col(c,i)
        if (tag == 0) cycle
        count = 0
        do j=1,nreal-1
          if (i == j) cycle
          other = c
          if ((i <= 2) .eqv. (j <= 2)) other = 3-c
          if (born_col(other,j) /= tag) cycle
          connected(i,j) = .true.
          count = count+1
        end do
        if (count /= 1) return
      end do
    end do

    status = delta_ok
    do i=1,nreal-1
      ir = i
      if (ir >= emitted) ir = ir+1
      do j=1,nreal-1
        if (.not.connected(i,j)) cycle
        jr = j
        if (jr >= emitted) jr = jr+1
        kr = jr
        ! Default aMC@NLO:debugScales=off: ISR always uses the other beam,
        ! even when its colour partner is a final-state particle.
        if (ir <= 2) kr = 3-ir
        call pythia_pt_lund(p(:,ir), p(:,emitted), p(:,kr), &
                           id_real(ir), id_real(emitted), ir > 2, kr > 2, &
                           charm_mass, bottom_mass, raw, local_status)
        if (local_status /= delta_ok) status = local_status
        ! Sum in the order used by Merging::clusterAndStore.
        qdip = p(:,ir)
        if (ir <= 2) qdip = -qdip
        if (kr > 2) then
          qdip = qdip + p(:,kr)
        else
          qdip = qdip - p(:,kr)
        end if
        qdip = qdip + p(:,emitted)
        masses(ir,jr) = sqrt(abs(dot4(qdip,qdip)))
        scales(ir,jr) = raw
        if (raw == 1.e10_dp) scales(ir,jr) = 0._dp
        ! Deliberately match the C++ sentinel tests: 1e-6 is NOT dead.
        dead(ir,jr) = raw <= 0._dp .or. raw == 1.e10_dp .or. raw == 1.e-5_dp &
                      .or. local_status /= delta_ok
      end do
    end do
  end subroutine delta_scale_matrices

end module mcatnlo_delta_scales
