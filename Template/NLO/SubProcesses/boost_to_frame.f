c**************************************************************************
c     Rest frame in which the matrix-elements are evaluated.
c
c     Polarisation of a massive particle is not boost invariant, so for a
c     polarised process the matrix-elements must be evaluated in a definite
c     frame, chosen by the user through me_frame in the run_card. This file
c     provides the kinematics; the selection of which legs define the frame
c     is resolved by the caller and passed in as the mask ids().
c
c     IMPORTANT (see docs/nlo_polarisation_boost_plan.md, D3): the frame must
c     be recomputed from the momenta of *each* kinematic configuration (Born,
c     real, and every counter-event separately). Computing it once from the
c     real and reusing it for the reduced Born breaks the cancellation of the
c     collinear pole.
c**************************************************************************


      subroutine snap_mother_at_rest(ids, npart, imother, pm)
c**************************************************************************
c     When me_frame selects exactly one leg and that leg is the FKS mother,
c     the mother is analytically at rest in this frame. boostx only reaches
c     that to a few 1d-14, and everything downstream tests for rest with an
c     exact .eq.0d0 -- azifact_from_kperp divides by |pmother| and guards it
c     with rn.eq.0d0. Left as noise the guard never fires and the azimuth is
c     reconstructed from a direction that is pure rounding error; imposed,
c     the caller gets that routine's clear 'mother at rest' error instead of
c     a silently wrong phase. boost_to_me_frame imposes the same property on
c     the full momentum set, and for the same reason.
c**************************************************************************
      implicit none
      integer npart, imother
      integer ids(npart)
      double precision pm(0:3)
      integer i, nsel, isel

      nsel=0
      isel=0
      do i=1,npart
         if (ids(i).eq.1) then
            nsel=nsel+1
            isel=i
         endif
      enddo
      if (nsel.eq.1 .and. isel.eq.imother) then
         pm(1)=0d0
         pm(2)=0d0
         pm(3)=0d0
      endif

      return
      end


      subroutine azifact_me_frame(p_born_in, imother, p_i, p_j,
     &                            y_ij_fks, azifact, boosted)
c**************************************************************************
c     Collinear azimuthal factor -exp(2 i psi), evaluated in the me_frame.
c
c     The frame is rebuilt from the very momenta the matching Born call is
c     given, so the spin-correlated Born and the phase multiplying it are
c     guaranteed to live in the same frame. That is the whole point: the
c     Born is boosted inside sborn_frame, and if the phase were left in the
c     partonic c.m. the Q term would be wrong.
c
c     The mother and the stored emission direction are boosted together.
c     xij_kperp is orthogonal to the mother (k.p = 0 in the frame it was
c     built in), and orthogonality is preserved by the boost, so projecting
c     the boosted vector onto the helicity basis of the boosted mother
c     recovers the azimuth correctly.
c
c     boosted=.false. means no frame was requested (or the selection is one
c     of the partonic-c.m. spellings). azifact is then left untouched and
c     the caller must keep its legacy value, so that runs without a frame
c     stay bit-identical.
c
c     input:  p_born_in(0:3,nexternal-1)  Born momenta of this configuration
c             imother                     position of the mother in them
c     output: azifact, boosted
c**************************************************************************
      implicit none
      include 'nexternal.inc'
      double precision p_born_in(0:3,nexternal-1)
      integer imother
      double precision p_i(0:3), p_j(0:3), y_ij_fks
      double complex azifact
      logical boosted
      double precision pboost(0:3,2), pm(0:3), kp(0:3)
      double precision pib(0:3), pjb(0:3)
      double precision cphi_m, sphi_m, vtiny
      parameter (vtiny=1d-8)
      integer ids(nexternal-1)
      logical trivial
      double precision xij_kperp(0:3)
      common/cxij_kperp/xij_kperp

      call get_frame_mask_born(ids)
      call get_me_frame_boost(p_born_in, nexternal-1, ids, pboost,
     &                        trivial)
      if (trivial) then
         boosted=.false.
         return
      endif

      call me_frame_apply(p_born_in(0,imother), pboost, pm)
      call snap_mother_at_rest(ids, nexternal-1, imother, pm)

      if (1d0-y_ij_fks.lt.vtiny) then
c        Collinear: <ij>/[ij] is 0/0, so rebuild -exp(2 i psi) from the
c        stored emission direction. Same threshold as the legacy branch
c        in sborncol_fsr/isr, so boosted and unboosted switch together.
         call me_frame_apply(xij_kperp, pboost, kp)
         call azifact_from_kperp(pm, kp, azifact)
      else
c        Away from the collinear point the spinors are exact, while the
c        kperp reconstruction carries an O((kt/E)/betaT) error -- 136%
c        at betaT=1e-5 when this was measured for azifact_mc_frame,
c        which is why that routine branches the same way. sreal enters
c        here whenever 1-y_ij_fks < tiny (1e-6 in production), not only
c        at y=1, so this branch is reached with genuinely off-collinear
c        kinematics. Convert <ij>/[ij] to the -exp(2 i psi) this routine
c        contracts to return, using <ij>/[ij] = -exp(2 i psi) e^(2 i phi_m).
         call me_frame_apply(p_i, pboost, pib)
         call me_frame_apply(p_j, pboost, pjb)
         call azifact_from_spinors(pib, pjb, azifact)
         call getaziangles(pm, cphi_m, sphi_m)
         azifact=azifact*dcmplx(cphi_m,-sphi_m)**2
      endif
      boosted=.true.

      return
      end


      subroutine me_frame_born_boost(p_born_in, pboost, boosted)
c**************************************************************************
c     The boosts sborn_frame() would apply to these Born momenta, exposed
c     so the MC counterterms can put their azimuthal ingredients in the
c     same frame (apply them with me_frame_apply). boosted=.false. means no
c     frame was requested.
c**************************************************************************
      implicit none
      include 'nexternal.inc'
      double precision p_born_in(0:3,nexternal-1), pboost(0:3,2)
      logical boosted
      integer ids(nexternal-1)
      logical trivial

      call get_frame_mask_born(ids)
      call get_me_frame_boost(p_born_in, nexternal-1, ids, pboost,
     &                        trivial)
      boosted=.not.trivial

      return
      end


      subroutine azifact_from_spinors(pi, pj, azifact)
c**************************************************************************
c     <ij>/[ij] from two massless momenta -- the expression the legacy
c     branches build inline. Those keep their own copies on purpose:
c     rewriting them would break the byte-identity of the no-frame path.
c**************************************************************************
      USE ALOHA_OBJECT
      implicit none
      double precision pi(0:3), pj(0:3)
      double complex azifact
      double precision zero
      parameter (zero=0d0)
      TYPE(ALOHA) W1,W2,W3,W4
      double complex Wij_angle,Wij_recta
      integer i

      CALL IXXXXX(pi ,ZERO ,+1,+1,1,W1)
      CALL OXXXXX(pj ,ZERO ,-1,+1,1,W2)
      CALL IXXXXX(pi ,ZERO ,-1,+1,1,W3)
      CALL OXXXXX(pj ,ZERO ,+1,+1,1,W4)
      Wij_angle=(0d0,0d0)
      Wij_recta=(0d0,0d0)
      do i=1,4
         Wij_angle = Wij_angle + W1%W(i)*W2%W(i)
         Wij_recta = Wij_recta + W3%W(i)*W4%W(i)
      enddo
      azifact=Wij_angle/Wij_recta

      return
      end


      subroutine azifact_mc_frame(p_born_in, imother, p_i, p_j,
     &                            y_ij_fks, pboost, azifact,
     &                            cphi_mother, sphi_mother)
c**************************************************************************
c     Azimuthal ingredients of the MC counterterms, in the me_frame, scaled
c     so that the caller's existing multiply comes out right:
c         ISR   -(cphi_mother + i sphi_mother)**2 * conjg(azifact)
c         FSR   -(cphi_mother - i sphi_mother)**2 *       azifact
c     both = +-exp(-+2 i psi), psi the emission azimuth about the mother.
c     The boosts (pboost, from me_frame_born_boost) are the Born's own, since
c     the phase multiplies borntilde. azifact is <ij>/[ij] only in the
c     generic branch; the collinear one pre-cancels the caller's mother
c     factor instead, so do not assume it if you change the caller.
c**************************************************************************
      implicit none
      include 'nexternal.inc'
      double precision p_born_in(0:3,nexternal-1)
      integer imother
      double precision p_i(0:3), p_j(0:3), y_ij_fks, pboost(0:3,2)
      double complex azifact
      double precision cphi_mother, sphi_mother
      double precision pm(0:3), kp(0:3), pib(0:3), pjb(0:3)
      integer ids(nexternal-1)
      double precision xij_kperp(0:3)
      common/cxij_kperp/xij_kperp

      call me_frame_apply(p_born_in(0,imother), pboost, pm)
      call get_frame_mask_born(ids)
      call snap_mother_at_rest(ids, nexternal-1, imother, pm)
      call getaziangles(pm, cphi_mother, sphi_mother)

      if (1d0-y_ij_fks.le.0d0) then
c        Exactly collinear: <ij>/[ij] is 0/0 in every frame, so rebuild it
c        from the stored emission direction, using
c            lim_{kt->0} <ij>/[ij] = -exp(2 i psi) * exp(2 i phi_m),
c        the identity that makes this branch agree with the generic one;
c        azifact_from_kperp returns only the first factor.
c
c        Taken at kt exactly zero and nowhere else, on purpose: the spinors
c        are exact at any kt>0, while this reconstruction carries an
c        O((kt/E)/betaT) error, betaT the transverse velocity of the boost.
         call me_frame_apply(xij_kperp, pboost, kp)
         call azifact_from_kperp(pm, kp, azifact)
         azifact=azifact*dcmplx(cphi_mother,sphi_mother)**2
      else
         call me_frame_apply(p_i, pboost, pib)
         call me_frame_apply(p_j, pboost, pjb)
         call azifact_from_spinors(pib, pjb, azifact)
      endif

      return
      end


      subroutine azifact_from_kperp(pmother, kperp, azifact)
c**************************************************************************
c     Rebuild the collinear limit of <ij>/[ij] from the emission direction.
c
c     The counterterms need -exp(2 i psi), with psi the azimuth of the
c     emission about the mother, measured in the HELAS transverse basis of
c     the mother direction. The generator stores that limit as xij_aor, but
c     xij_aor is a 0/0 limit of <ij>/[ij]: a boost maps exactly-parallel
c     null vectors onto exactly-parallel null vectors, so the degeneracy
c     survives in every frame and xij_aor can never be recomputed after a
c     boost. The azimuthal information has to be carried covariantly
c     instead, which is what xij_kperp is for.
c
c     Basis, for a mother direction n = (sin(th)cos(ph), sin(th)sin(ph),
c     cos(th)):
c         e1 = ( cos(th)cos(ph), cos(th)sin(ph), -sin(th) )
c         e2 = (        -sin(ph),      cos(ph),        0  )
c     which is the standard helicity basis; for n = +z it is (x,y) and for
c     n = -z it is (-x,y).
c
c     This reproduces both incoming legs with no idir bookkeeping:
c       j_fks=1, n=+z : psi = phi_i        -> -exp( 2 i phi_i)
c       j_fks=2, n=-z : psi = pi - phi_i   -> -exp(-2 i phi_i)
c     the doubled angle turning the basis flip into a harmless 2 pi. The
c     R_y(pi) rotation the old ISR code applied for j_fks=2 is therefore
c     not needed here -- and it must not be reinstated, since after a boost
c     it no longer maps the mother onto +z.
c
c     input:  pmother(0:3)  mother 4-momentum (only its direction is used)
c             kperp(0:3)    emission direction transverse to the mother
c     output: azifact       -exp(2 i psi)
c**************************************************************************
      implicit none
      double precision pmother(0:3), kperp(0:3)
      double complex azifact
      double precision n(3), e1(3), e2(3)
      double precision rn, cth, sth, cph, sph, a, b, norm2
      double complex ximag
      parameter (ximag=(0d0,1d0))
      integer i

      rn=sqrt(pmother(1)**2+pmother(2)**2+pmother(3)**2)
      if (rn.eq.0d0) then
         write (*,*) 'ERROR in azifact_from_kperp: mother at rest'
         stop 1
      endif
      do i=1,3
         n(i)=pmother(i)/rn
      enddo

      cth=n(3)
      sth=sqrt(max(1d0-cth**2,0d0))
      if (sth.ne.0d0) then
         cph=n(1)/sth
         sph=n(2)/sth
      else
c        mother on the beam axis: its azimuth is undefined, take phi=0.
c        Same convention as getaziangles().
         cph=1d0
         sph=0d0
      endif

      e1(1)= cth*cph
      e1(2)= cth*sph
      e1(3)=-sth
      e2(1)=-sph
      e2(2)= cph
      e2(3)= 0d0

      a=0d0
      b=0d0
      do i=1,3
         a=a+kperp(i)*e1(i)
         b=b+kperp(i)*e2(i)
      enddo

c     -exp(2 i psi) with psi=atan2(b,a), written without trigonometry so
c     that only the component of kperp transverse to the mother enters.
      norm2=a**2+b**2
      if (norm2.eq.0d0) then
         write (*,*) 'ERROR in azifact_from_kperp: the emission'
         write (*,*) 'direction is parallel to the mother, so its'
         write (*,*) 'azimuth is undefined.'
         stop 1
      endif
      azifact=-(dcmplx(a,b)**2)/norm2

      return
      end


      subroutine sborn_frame(p_in, ans_summed)
c**************************************************************************
c     Evaluate the Born in the frame selected by me_frame.
c
c     The frame is rebuilt from the momenta actually passed in, so each
c     kinematic configuration (the Born of the n-body contribution, and in
c     later milestones each counter-event separately) gets its own boost.
c     Reusing one frame across configurations breaks the cancellation of the
c     collinear pole -- see docs/nlo_polarisation_boost_plan.md, D3.
c
c     The boost is applied to a local copy: /pborn/ is read by the event
c     record, the cuts and the scales, none of which want the ME frame.
c
c     Every caller of SBORN within one event must agree on whether it passes
c     boosted or unboosted momenta: SBORN caches its amplitudes against
c     (E, p_z) and that cache is shared with sborn_sf, born_hel and
c     extra_cnt. Mixing the two silently returns amplitudes from the wrong
c     frame.
c**************************************************************************
      implicit none
      include 'nexternal.inc'
      double precision p_in(0:3,nexternal-1)
      double precision ans_summed
      double precision p_f(0:3,nexternal-1)
      integer ids(nexternal-1)

      call get_frame_mask_born(ids)
      call boost_to_me_frame(p_in, nexternal-1, ids, p_f)
      call sborn(p_f, ans_summed)

      return
      end


      subroutine sborn_onehel_frame(p_in, hel1, chosen_hel, ans)
c**************************************************************************
c     One-helicity Born in the frame selected by me_frame.
c
c     Needed wherever a caller already goes through sborn_frame: SBORN
c     caches its amplitudes against (E, p_z) and SBORN_ONEHEL shares that
c     cache, so passing boosted momenta to one and unboosted to the other
c     silently returns amplitudes from the wrong frame -- see the note on
c     sborn_frame. check_sudakov does exactly that mix.
c**************************************************************************
      implicit none
      include 'nexternal.inc'
      double precision p_in(0:3,nexternal-1)
      integer hel1, chosen_hel
      double precision ans
      double precision p_f(0:3,nexternal-1)
      integer ids(nexternal-1)

      call get_frame_mask_born(ids)
      call boost_to_me_frame(p_in, nexternal-1, ids, p_f)
      call sborn_onehel(p_f, hel1, chosen_hel, ans)

      return
      end


      subroutine binothlha_frame(p_in, born_wgt, virt_wgt)
c**************************************************************************
c     Virtual in the me_frame rest frame.
c
c     BinothLHA hands its argument straight to sloopmatrix_thres, and takes
c     born_wgt from the caller rather than recomputing it, so boosting the
c     momenta here is enough to put the virtual in the same frame as the
c     Born that bornsoftvirtual already boosted.
c
c     Two MadLoop settings interact with an imposed frame, both via the
c     HELAS branch at exactly zero momentum that a one-leg me_frame sits on
c     (see boost_to_me_frame):
c
c       - NRotations_DP / NRotations_QP re-evaluate the loop at a rotated
c         phase-space point and expect |M|^2 back unchanged. That holds for
c         fixed helicities in general, but NOT for a particle at rest,
c         whose polarisation axis is the frame z axis rather than its own
c         momentum, so a rotation of the momenta does not carry the axis
c         with it. Both default to 0, so the rotation test is off unless a
c         user turns it on; if they do, a polarised run will see spurious
c         instability. Keep them at 0.
c       - ImprovePSPoint (default 2) shifts momenta to restore exact
c         onshellness, which can move the deliberately-zeroed leg off zero.
c         The shift is along z, so the momentum direction stays on the axis
c         and eps_L comes back along +-z, differing at most by a sign that
c         cancels in |M|^2. Checked to be harmless by check_poles rather
c         than assumed.
c**************************************************************************
      implicit none
      include 'nexternal.inc'
      double precision p_in(0:3,nexternal-1)
      double precision born_wgt, virt_wgt
      double precision p_f(0:3,nexternal-1)
      integer ids(nexternal-1)

      call get_frame_mask_born(ids)
      call boost_to_me_frame(p_in, nexternal-1, ids, p_f)
      call BinothLHA(p_f, born_wgt, virt_wgt)

      return
      end


      subroutine sborn_sf_frame(p_in, m, n, wgt)
c**************************************************************************
c     Colour-linked Born in the me_frame rest frame.
c
c     SBORN_SF reads the amplitudes SBORN left in the shared cache, so it
c     must be handed exactly the momenta the matching sborn_frame call was
c     handed, or it silently returns amplitudes from the other frame.
c**************************************************************************
      implicit none
      include 'nexternal.inc'
      double precision p_in(0:3,nexternal-1)
      integer m, n
      double precision wgt
      double precision p_f(0:3,nexternal-1)
      integer ids(nexternal-1)

      call get_frame_mask_born(ids)
      call boost_to_me_frame(p_in, nexternal-1, ids, p_f)
      call sborn_sf(p_f, m, n, wgt)

      return
      end


      subroutine extra_cnt_frame(p_in, icnt, cnts)
c**************************************************************************
c     Extra g/a -> q qbar counterterm Born in the me_frame rest frame.
c     Same cache as SBORN, so the same rule applies.
c**************************************************************************
      implicit none
      include 'nexternal.inc'
      include 'orders.inc'
      double precision p_in(0:3,nexternal-1)
      integer icnt
      double complex cnts(2,nsplitorders)
      double precision p_f(0:3,nexternal-1)
      integer ids(nexternal-1)

      call get_frame_mask_born(ids)
      call boost_to_me_frame(p_in, nexternal-1, ids, p_f)
      call extra_cnt(p_f, icnt, cnts)

      return
      end


      subroutine smatrix_real_frame(p_in, wgt)
c**************************************************************************
c     Real-emission matrix element in the me_frame rest frame.
c
c     nexternal legs, so the mask is the real one; it is derived from the
c     Born mask and i_fks of the current FKS configuration.
c**************************************************************************
      implicit none
      include 'nexternal.inc'
      double precision p_in(0:3,nexternal)
      double precision wgt
      double precision p_f(0:3,nexternal)
      integer ids(nexternal)
      integer nFKSprocess
      common/c_nFKSprocess/nFKSprocess

      call get_frame_mask_real(nFKSprocess, ids)
      call boost_to_me_frame(p_in, nexternal, ids, p_f)
      call smatrix_real(p_f, wgt)

      return
      end


      subroutine get_frame_mask_born(ids)
c**************************************************************************
c     Mask over the nexternal-1 Born legs selected by me_frame.
c
c     me_frame is given in the numbering of the process as the user wrote
c     it, which the FKS sort does not preserve; frame_map_born (written at
c     generation time into frame_info.inc) maps Born position -> user number.
c
c     output: ids(nexternal-1)
c**************************************************************************
      implicit none
      include 'nexternal.inc'
      include 'frame_info.inc'
      include 'run.inc'
      integer ids(nexternal-1)
      integer i
c     The answer depends only on frame_id, fixed for the run from the
c     run_card, and on frame_map_born, a compile-time DATA table -- but
c     this is called from every _frame wrapper on every phase-space point.
c     Cache it, keyed on frame_id so the cache cannot go stale if that
c     ever becomes settable mid-run. No OpenMP in these sources; MadEvent
c     parallelises by forking, so each process gets its own copy.
      integer cached_ids(nexternal-1)
      integer cached_id
      save cached_ids, cached_id
      data cached_id /-1/

      if (frame_id.ne.cached_id) then
         do i=1,nexternal-1
            cached_ids(i)=0
            if (btest(frame_id, frame_map_born(i))) then
               cached_ids(i)=1
            endif
         enddo
         cached_id=frame_id
      endif

      do i=1,nexternal-1
         ids(i)=cached_ids(i)
      enddo

      return
      end


      subroutine get_frame_mask_real(iFKS, ids)
c**************************************************************************
c     Mask over the nexternal legs of the real emission selected by
c     me_frame, for FKS configuration iFKS.
c
c     The FKS convention fixes the real -> underlying-Born correspondence in
c     terms of i_fks alone (see set_pdg in chooser_functions.f):
c         real r  ->  Born r      for r <  i_fks
c         real r  ->  Born r-1    for r >  i_fks
c         real i_fks has no Born counterpart (it is the extra parton)
c     so no separate table is needed for the reals.
c
c     Note on r = j_fks: positionally the Born leg is the *mother* ij of
c     i_fks and j_fks. That is still the right leg to inherit the mask from,
c     and NOT because the polarised legs are spectators -- they are not: a
c     polarised massive quark IS j_fks in some configurations, and the real
c     carries its polarisation (see fks_common.carry_polarization).
c     It is right because in that case the real's j_fks is the SAME particle
c     as the Born's ij, so "the leg the user selected" means the same thing on
c     both sides, and the two matrix elements are boosted into frames that
c     coincide in the singular limit.
c
c     That equality is what makes the whole polarised subtraction well
c     defined, and it fails in two places, both refused at generation time by
c     madgraph_interface.check_process_format. Do not read this routine as a
c     licence to relax either refusal:
c       - a massless coloured emitter in the final state: g -> q q~ has no
c         leg with the mother's identity to correspond to.
c       - a coloured particle in the INITIAL state, whatever its mass. The
c         initial-state splitting runs backwards, g -> q(-> Born) q~, so the
c         polarised Born leg is an INTERNAL line of the real: there is no
c         real-side external leg for this mask to select at all, and the
c         positional r <-> b table above has nothing to say about it.
c
c     output: ids(nexternal)
c**************************************************************************
      implicit none
      include 'nexternal.inc'
      include 'fks_info.inc'
      include 'nFKSconfigs.inc'
      include 'run.inc'
      integer iFKS
      integer ids(nexternal)
      integer ids_born(nexternal-1)
      integer r, b
c     Constant for a given iFKS: the Born mask is run-constant and the
c     r <-> b correspondence is fixed by fks_i_D(iFKS), a DATA table. So
c     cache one mask per FKS configuration rather than rebuilding both
c     this and the Born mask on every real-emission evaluation. Keyed on
c     frame_id for the same reason as get_frame_mask_born.
      integer cached_real(nexternal, fks_configs)
      logical cached_ok(fks_configs)
      integer cached_id_r
      save cached_real, cached_ok, cached_id_r
      data cached_ok /fks_configs*.false./
      data cached_id_r /-1/

      if (frame_id.ne.cached_id_r) then
         do r=1,fks_configs
            cached_ok(r)=.false.
         enddo
         cached_id_r=frame_id
      endif
      if (iFKS.ge.1 .and. iFKS.le.fks_configs) then
         if (cached_ok(iFKS)) then
            do r=1,nexternal
               ids(r)=cached_real(r,iFKS)
            enddo
            return
         endif
      endif

      call get_frame_mask_born(ids_born)

      do r=1,nexternal
         ids(r)=0
         if (r.eq.fks_i_D(iFKS)) then
c           the extra parton: no Born counterpart, never part of the frame
            cycle
         elseif (r.lt.fks_i_D(iFKS)) then
            b=r
         else
            b=r-1
         endif
         if (b.ge.1 .and. b.le.nexternal-1) then
            ids(r)=ids_born(b)
         endif
      enddo

      if (iFKS.ge.1 .and. iFKS.le.fks_configs) then
         do r=1,nexternal
            cached_real(r,iFKS)=ids(r)
         enddo
         cached_ok(iFKS)=.true.
      endif

      return
      end


      subroutine get_me_frame_boost(p, npart, ids, pboost, trivial)
c**************************************************************************
c     Build the boosts that take the selected system to rest.
c
c     pboost(:,1) and pboost(:,2) are the longitudinal and the transverse
c     step of get_me_frame_boosts (Template/Common/Source), to be applied
c     in that order -- me_frame_apply does both. Not the single boost along
c     the momentum of the system: that one depends on the frame the
c     momenta come in, and FKS hands the real emission its momenta in the
c     c.m. frame of the Born of whichever FKS sector generated it. See
c     get_me_frame_boosts for what that did to a single leg at rest.
c
c     trivial is returned .true. when the boost should be skipped rather
c     than applied. Applying an identity boost is not free: it is a
c     multiply-and-add through boostx that perturbs the momenta in the last
c     bits, which is enough to move an adaptive integration. The LO code
c     avoids exactly this, in two places -- the call site skips frame_id=6
c     (auto_dsig_v4.inc) and boost_to_frame() skips the all-final-state
c     selection (genps.f) -- and this routine folds both in.
c
c     Four cases:
c       - nothing selected;
c       - the selection is exactly the initial state;
c       - the selection is exactly the whole final state;
c       - the selected system already has exactly zero 3-momentum.
c
c     The middle two are both "the partonic c.m.", and skipping them means
c     "the frame the momenta come in". For the Born, the virtual and every
c     counter-event that is the Born c.m., so the whole-final-state skip is
c     exact there. For the real emission it is the FKS 'tilde' frame: the
c     real's own c.m. in a final-state sector, but the c.m. of the
c     underlying Born in an initial-state one, a boost along z away. So
c     with initial-state QCD radiation, skipping the initial-state
c     selection made the real depend on the FKS sector, and the polarised
c     result on the subtraction partition (+9.5% on p p > w+{0} j [QCD]
c     when only fks_b changes). Honouring it literally is no better: the
c     real's own c.m. jumps by a finite boost across the initial-state
c     collinear limit, so it is not infrared safe. The run_card therefore
c     never sends the initial state here for such a process: its default
c     is the massive final state, and RunCardNLO.get_me_frame_legs
c     replaces an explicit [1,2] by the Born final state (refusing it when
c     that contains a jet). What still reaches this skip
c     is a process without initial-state QCD splittings (e+ e-), where the
c     tilde frame is the partonic c.m. in every sector.
c
c     input:  p(0:3,npart)  momenta of this configuration
c             ids(npart)    0/1 mask, bit n of frame_id = leg n+1
c     output: pboost(0:3,2) the two boosts, as boostx arguments
c             trivial       .true. if the boost may be skipped
c**************************************************************************
      implicit none
      integer npart
      double precision p(0:3,npart), pboost(0:3,2)
      integer ids(npart)
      logical trivial
      double precision psum(0:3)
      integer i, j
      integer nsel, nini, nfin
      double precision m2, pvec2
      double precision m2i, betamin, mtiny
      parameter (betamin=1d-4, mtiny=1d-12)
      logical massive_ini
      include 'nexternal.inc'

      psum(0:3)=0d0
      nsel=0
      nini=0
      nfin=0
      do i=1,npart
         if (ids(i).eq.1) then
            nsel=nsel+1
            if (i.le.nincoming) then
               nini=nini+1
            else
               nfin=nfin+1
            endif
            do j=0,3
               psum(j)=psum(j)+p(j,i)
            enddo
         endif
      enddo

c     Nothing selected: nothing to do. Do not treat this as an error, it is
c     how frame_id=0 (or a mask that misses this configuration) shows up.
      if (nsel.eq.0) then
         trivial=.true.
         return
      endif

c     Exactly the initial state, or exactly the whole final state: both name
c     the partonic c.m.. Skip rather than boost -- see the header.
c
c     Note the second of these is multiplicity-dependent and therefore does
c     not mean the same thing for the Born and for the real: me_frame=[3,4]
c     on p p > z z is the whole Born final state, but in the real the same
c     two legs are only two of three. It was checked that this asymmetry is
c     *not* what makes that configuration fail -- removing the skip leaves
c     the failure fractions bit-identical, because p_born is handed to us in
c     the Born partonic c.m. and so the zero-3-momentum test below catches
c     the same case anyway. The skip is kept for parity with LO.
      if (nfin.eq.0 .and. nini.eq.nincoming) then
         trivial=.true.
         return
      endif
      if (nini.eq.0 .and. nfin.eq.npart-nincoming) then
         trivial=.true.
         return
      endif

      pvec2=psum(1)**2+psum(2)**2+psum(3)**2

c     Already at rest: skip, so the default costs nothing and changes nothing.
      if (pvec2.eq.0d0) then
         trivial=.true.
         return
      endif

c     A near-identity boost tilts the on-axis incoming partons just
c     enough that p(0)+p(3) of the backward one becomes E*beta**2/2,
c     computed as a difference of two numbers of size E: relative error
c     2*eps/beta**2, worst at small beta. A betamin=1d-4 skip used to
c     hide that (blocker B9), and was removed because PR #111 takes the
c     component from p+ p- = pT**2 instead, leaving no cancellation.
c
c     That holds for a MASSLESS incoming parton. PR #111 rewrote only
c     the fmass.eq.rZero branch of ixxxxx/oxxxxx; the massive branch
c     still forms pp3 = max(pp+p(3),rZero) and feeds it to
c     sqrt(pp3/(2*pp)), which is the same cancelling difference. So the
c     skip is kept exactly where it is still needed -- a massive
c     initial-state parton, e.g. b in a 4-flavour scheme -- and nowhere
c     else. The deeper fix is to give the massive branch the same
c     treatment as the massless one, across the ALOHA templates, the
c     legacy HELAS sources and the C++ backends; until that lands this
c     guard is what stops a massive beam leg losing its precision here.
      massive_ini=.false.
      do i=1,min(nincoming,npart)
         m2i=p(0,i)**2-p(1,i)**2-p(2,i)**2-p(3,i)**2
         if (m2i.gt.mtiny*p(0,i)**2) massive_ini=.true.
      enddo
      if (massive_ini .and. pvec2.lt.(betamin*psum(0))**2) then
         trivial=.true.
         return
      endif

c     The selected system must be timelike for its rest frame to exist. A
c     lightlike or spacelike sum means the run_card selection is nonsense
c     (e.g. a single massless leg); the LO boost_to_frame has no such guard.
      m2=psum(0)**2-pvec2
      if (m2.le.0d0) then
         write (*,*) 'ERROR in get_me_frame_boost: the system'
         write (*,*) 'selected by me_frame is not timelike, so its'
         write (*,*) 'rest frame does not exist. m^2 = ', m2
         write (*,*) 'Selected legs:', (ids(i),i=1,npart)
         stop 1
      endif

      call get_me_frame_boosts(ids, npart, nincoming, p, pboost(0,1),
     &                         pboost(0,2))
      trivial=.false.

      return
      end


      subroutine me_frame_apply(v, pboost, vout)
c**************************************************************************
c     Express v in the me_frame, given the boosts of get_me_frame_boost.
c     v and vout must not be the same array.
c**************************************************************************
      implicit none
      double precision v(0:3), pboost(0:3,2), vout(0:3)
      double precision vz(0:3)

      call boostx(v, pboost(0,1), vz)
      call boostx(vz, pboost(0,2), vout)

      return
      end


      subroutine boost_to_me_frame(p, npart, ids, p_out)
c**************************************************************************
c     Express all npart momenta in the rest frame of the selected system,
c     reached by a boost along z and then a transverse one, so that the
c     frame does not depend on which boost along z the momenta come in
c     (get_me_frame_boosts). For two or more selected legs the orientation
c     would not matter: for fixed external helicities |M|^2 is rotation
c     invariant (HELAS builds the polarisation vectors in each particle's
c     own momentum-direction basis, so a rotation only produces
c     per-particle phases which cancel in the modulus). It matters for a
c     single leg, held at rest with its spin along the frame z axis, and
c     for the collinear counterterm azimuthal phases, which the callers
c     build with the same boosts.
c
c     p and p_out may be the same array.
c**************************************************************************
      implicit none
      integer npart
      double precision p(0:3,npart), p_out(0:3,npart)
      integer ids(npart)
      double precision pboost(0:3,2)
      logical trivial
      integer i

      call get_me_frame_boost(p, npart, ids, pboost, trivial)

      if (trivial) then
         do i=1,npart
            p_out(0:3,i)=p(0:3,i)
         enddo
         return
      endif

      do i=1,npart
         call me_frame_apply(p(0,i), pboost, p_out(0,i))
      enddo

c     A selection made of a single leg puts that leg at rest, and there
c     the boost has to be *exactly* right, not right to rounding --
c     see Template/Common/Source/impose_frame_rest.f for why HELAS
c     cares.
c
c     What is specific to NLO is that whether the residual rounds to
c     zero is decided independently for the real and for the reduced
c     Born, so the two are then evaluated with different quantisation
c     axes and the subtraction stops cancelling. Measured on
c     p p > z{0} z{0} [real=QCD] with me_frame=[3]: the soft/collinear
c     ratio is 1 to 1d-7 whenever the residual is exactly zero and
c     2.2d-3 whenever it is 3d-14, with no intermediate values.
      call impose_frame_rest(ids, npart, p_out)

      return
      end
