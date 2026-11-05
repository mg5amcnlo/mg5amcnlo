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
      use fks_phase_space_data, only: xij_kperp
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
      use fks_phase_space_data, only: xij_kperp
      implicit none
      include 'nexternal.inc'
      double precision p_born_in(0:3,nexternal-1)
      integer imother
      double precision p_i(0:3), p_j(0:3), y_ij_fks, pboost(0:3,2)
      double complex azifact
      double precision cphi_mother, sphi_mother
      double precision pm(0:3), kp(0:3), pib(0:3), pjb(0:3)
      integer ids(nexternal-1)

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


      SUBROUTINE SBORN_FRAME(P_IN, ANS_SUMMED)
      USE MC_BORN_TYPES,ONLY:BORN_ACTIVE_HELICITY_FRAME
      USE MC_NATIVE_CONTEXT,ONLY:ENSURE_NATIVE_CONTEXT,ACTIVE_CONTEXT
      IMPLICIT NONE
      INTEGER SAVED_HELICITY_FRAME(2),FRAME_LEG
      INCLUDE 'nexternal.inc'
      DOUBLE PRECISION P_IN(0:3,NEXTERNAL-1)
      DOUBLE PRECISION ANS_SUMMED
      DOUBLE PRECISION P_F(0:3,NEXTERNAL-1)
      INTEGER IDS(NEXTERNAL-1)
      CALL ENSURE_NATIVE_CONTEXT()
      CALL GET_FRAME_MASK_BORN(IDS)
      CALL BOOST_TO_ME_FRAME(P_IN, NEXTERNAL-1, IDS, P_F)
      SAVED_HELICITY_FRAME=BORN_ACTIVE_HELICITY_FRAME
      IF(ANY(IDS.NE.0).AND..NOT.(ALL(IDS(1:NINCOMING).EQ.1)
     $ .AND.ALL(IDS(NINCOMING+1:NEXTERNAL-1).EQ.0))
     $ .AND..NOT.(ALL(IDS(1:NINCOMING).EQ.0).AND.ALL(IDS(NINCOMING
     $ +1:NEXTERNAL-1).EQ.1)))THEN
        BORN_ACTIVE_HELICITY_FRAME=[ACTIVE_CONTEXT,0]
        DO FRAME_LEG=1,NEXTERNAL-1
          IF(IDS(FRAME_LEG).NE.0)BORN_ACTIVE_HELICITY_FRAME(2)
     $     =IBSET(BORN_ACTIVE_HELICITY_FRAME(2),FRAME_LEG)
        ENDDO
      ENDIF
      CALL SBORN_NATIVE(P_F, ANS_SUMMED)
      BORN_ACTIVE_HELICITY_FRAME=SAVED_HELICITY_FRAME
      RETURN
      END
      SUBROUTINE SBORN_ONEHEL_FRAME(P_IN, HEL1, CHOSEN_HEL, ANS)
      USE MC_BORN_TYPES,ONLY:BORN_ACTIVE_HELICITY_FRAME
      USE MC_NATIVE_CONTEXT,ONLY:ENSURE_NATIVE_CONTEXT,ACTIVE_CONTEXT
      IMPLICIT NONE
      INTEGER SAVED_HELICITY_FRAME(2),FRAME_LEG
      INCLUDE 'nexternal.inc'
      DOUBLE PRECISION P_IN(0:3,NEXTERNAL-1)
      INTEGER HEL1, CHOSEN_HEL
      DOUBLE PRECISION ANS
      DOUBLE PRECISION P_F(0:3,NEXTERNAL-1)
      INTEGER IDS(NEXTERNAL-1)
      CALL ENSURE_NATIVE_CONTEXT()
      CALL GET_FRAME_MASK_BORN(IDS)
      CALL BOOST_TO_ME_FRAME(P_IN, NEXTERNAL-1, IDS, P_F)
      SAVED_HELICITY_FRAME=BORN_ACTIVE_HELICITY_FRAME
      IF(ANY(IDS.NE.0).AND..NOT.(ALL(IDS(1:NINCOMING).EQ.1)
     $ .AND.ALL(IDS(NINCOMING+1:NEXTERNAL-1).EQ.0))
     $ .AND..NOT.(ALL(IDS(1:NINCOMING).EQ.0).AND.ALL(IDS(NINCOMING
     $ +1:NEXTERNAL-1).EQ.1)))THEN
        BORN_ACTIVE_HELICITY_FRAME=[ACTIVE_CONTEXT,0]
        DO FRAME_LEG=1,NEXTERNAL-1
          IF(IDS(FRAME_LEG).NE.0)BORN_ACTIVE_HELICITY_FRAME(2)
     $     =IBSET(BORN_ACTIVE_HELICITY_FRAME(2),FRAME_LEG)
        ENDDO
      ENDIF
      CALL SBORN_ONEHEL(P_F, HEL1, CHOSEN_HEL, ANS)
      BORN_ACTIVE_HELICITY_FRAME=SAVED_HELICITY_FRAME
      RETURN
      END
      SUBROUTINE BINOTHLHA_FRAME(P_IN, BORN_WGT, VIRT_WGT)
      USE MC_BORN_TYPES,ONLY:BORN_ACTIVE_HELICITY_FRAME
      USE MC_NATIVE_CONTEXT,ONLY:ENSURE_NATIVE_CONTEXT,ACTIVE_CONTEXT
      IMPLICIT NONE
      INTEGER SAVED_HELICITY_FRAME(2),FRAME_LEG
      INCLUDE 'nexternal.inc'
      DOUBLE PRECISION P_IN(0:3,NEXTERNAL-1)
      DOUBLE PRECISION BORN_WGT, VIRT_WGT
      DOUBLE PRECISION P_F(0:3,NEXTERNAL-1)
      INTEGER IDS(NEXTERNAL-1)
      CALL ENSURE_NATIVE_CONTEXT()
      CALL GET_FRAME_MASK_BORN(IDS)
      CALL BOOST_TO_ME_FRAME(P_IN, NEXTERNAL-1, IDS, P_F)
      SAVED_HELICITY_FRAME=BORN_ACTIVE_HELICITY_FRAME
      IF(ANY(IDS.NE.0).AND..NOT.(ALL(IDS(1:NINCOMING).EQ.1)
     $ .AND.ALL(IDS(NINCOMING+1:NEXTERNAL-1).EQ.0))
     $ .AND..NOT.(ALL(IDS(1:NINCOMING).EQ.0).AND.ALL(IDS(NINCOMING
     $ +1:NEXTERNAL-1).EQ.1)))THEN
        BORN_ACTIVE_HELICITY_FRAME=[ACTIVE_CONTEXT,0]
        DO FRAME_LEG=1,NEXTERNAL-1
          IF(IDS(FRAME_LEG).NE.0)BORN_ACTIVE_HELICITY_FRAME(2)
     $     =IBSET(BORN_ACTIVE_HELICITY_FRAME(2),FRAME_LEG)
        ENDDO
      ENDIF
      CALL BINOTHLHA(P_F, BORN_WGT, VIRT_WGT)
      BORN_ACTIVE_HELICITY_FRAME=SAVED_HELICITY_FRAME
      RETURN
      END
      SUBROUTINE SBORN_SF_FRAME(P_IN, M, N, WGT)
      USE MC_BORN_TYPES,ONLY:BORN_ACTIVE_HELICITY_FRAME
      USE MC_NATIVE_CONTEXT,ONLY:ENSURE_NATIVE_CONTEXT,ACTIVE_CONTEXT
      IMPLICIT NONE
      INTEGER SAVED_HELICITY_FRAME(2),FRAME_LEG
      INCLUDE 'nexternal.inc'
      DOUBLE PRECISION P_IN(0:3,NEXTERNAL-1)
      INTEGER M, N
      DOUBLE PRECISION WGT
      DOUBLE PRECISION P_F(0:3,NEXTERNAL-1)
      INTEGER IDS(NEXTERNAL-1)
      CALL ENSURE_NATIVE_CONTEXT()
      CALL GET_FRAME_MASK_BORN(IDS)
      CALL BOOST_TO_ME_FRAME(P_IN, NEXTERNAL-1, IDS, P_F)
      SAVED_HELICITY_FRAME=BORN_ACTIVE_HELICITY_FRAME
      IF(ANY(IDS.NE.0).AND..NOT.(ALL(IDS(1:NINCOMING).EQ.1)
     $ .AND.ALL(IDS(NINCOMING+1:NEXTERNAL-1).EQ.0))
     $ .AND..NOT.(ALL(IDS(1:NINCOMING).EQ.0).AND.ALL(IDS(NINCOMING
     $ +1:NEXTERNAL-1).EQ.1)))THEN
        BORN_ACTIVE_HELICITY_FRAME=[ACTIVE_CONTEXT,0]
        DO FRAME_LEG=1,NEXTERNAL-1
          IF(IDS(FRAME_LEG).NE.0)BORN_ACTIVE_HELICITY_FRAME(2)
     $     =IBSET(BORN_ACTIVE_HELICITY_FRAME(2),FRAME_LEG)
        ENDDO
      ENDIF
      CALL SBORN_SF_NATIVE(P_F, M, N, WGT)
      BORN_ACTIVE_HELICITY_FRAME=SAVED_HELICITY_FRAME
      RETURN
      END
      SUBROUTINE EXTRA_CNT_FRAME(P_IN, ICNT, CNTS)
      USE MC_BORN_TYPES,ONLY:BORN_ACTIVE_HELICITY_FRAME
      USE MC_NATIVE_CONTEXT,ONLY:ENSURE_NATIVE_CONTEXT,ACTIVE_CONTEXT
      IMPLICIT NONE
      INTEGER SAVED_HELICITY_FRAME(2),FRAME_LEG
      INCLUDE 'nexternal.inc'
      INCLUDE 'orders.inc'
      DOUBLE PRECISION P_IN(0:3,NEXTERNAL-1)
      INTEGER ICNT
      DOUBLE COMPLEX CNTS(2,NSPLITORDERS)
      DOUBLE PRECISION P_F(0:3,NEXTERNAL-1)
      INTEGER IDS(NEXTERNAL-1)
      CALL ENSURE_NATIVE_CONTEXT()
      CALL GET_FRAME_MASK_BORN(IDS)
      CALL BOOST_TO_ME_FRAME(P_IN, NEXTERNAL-1, IDS, P_F)
      SAVED_HELICITY_FRAME=BORN_ACTIVE_HELICITY_FRAME
      IF(ANY(IDS.NE.0).AND..NOT.(ALL(IDS(1:NINCOMING).EQ.1)
     $ .AND.ALL(IDS(NINCOMING+1:NEXTERNAL-1).EQ.0))
     $ .AND..NOT.(ALL(IDS(1:NINCOMING).EQ.0).AND.ALL(IDS(NINCOMING
     $ +1:NEXTERNAL-1).EQ.1)))THEN
        BORN_ACTIVE_HELICITY_FRAME=[ACTIVE_CONTEXT,0]
        DO FRAME_LEG=1,NEXTERNAL-1
          IF(IDS(FRAME_LEG).NE.0)BORN_ACTIVE_HELICITY_FRAME(2)
     $     =IBSET(BORN_ACTIVE_HELICITY_FRAME(2),FRAME_LEG)
        ENDDO
      ENDIF
      CALL EXTRA_CNT_NATIVE(P_F, ICNT, CNTS)
      BORN_ACTIVE_HELICITY_FRAME=SAVED_HELICITY_FRAME
      RETURN
      END
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


      SUBROUTINE GET_FRAME_MASK_BORN(IDS)
      USE MC_NATIVE_CONTEXT,ONLY:NATIVE_EPOCH,NATIVE_METADATA
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'frame_info.inc'
      INCLUDE 'run.inc'
      INTEGER IDS(NEXTERNAL-1)
      INTEGER I
      INTEGER CACHED_IDS(NEXTERNAL-1)
      INTEGER CACHED_EPOCH
      SAVE CACHED_EPOCH
      DATA CACHED_EPOCH/-1/
      INTEGER FRAME_MAP_ACTIVE(NEXTERNAL-1)
      INTEGER CACHED_ID
      SAVE CACHED_IDS, CACHED_ID
      DATA CACHED_ID /-1/
      IF (FRAME_ID.NE.CACHED_ID.OR.NATIVE_EPOCH.NE.CACHED_EPOCH) THEN
        FRAME_MAP_ACTIVE=FRAME_MAP_BORN
        IF(ASSOCIATED(NATIVE_METADATA))THEN
          IF(ALLOCATED(NATIVE_METADATA%FRAME_MAP))
     $     FRAME_MAP_ACTIVE=NATIVE_METADATA%FRAME_MAP
        ENDIF
        DO I=1,NEXTERNAL-1
          CACHED_IDS(I)=0
          IF (BTEST(FRAME_ID, FRAME_MAP_ACTIVE(I))) THEN
            CACHED_IDS(I)=1
          ENDIF
        ENDDO
        CACHED_ID=FRAME_ID
        CACHED_EPOCH=NATIVE_EPOCH
      ENDIF
      DO I=1,NEXTERNAL-1
        IDS(I)=CACHED_IDS(I)
      ENDDO
      RETURN
      END
      SUBROUTINE GET_FRAME_MASK_REAL(IFKS, IDS)
      USE MC_NATIVE_CONTEXT,ONLY:NATIVE_EPOCH
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'fks_info.inc'
      INCLUDE 'nFKSconfigs.inc'
      INCLUDE 'run.inc'
      INTEGER IFKS
      INTEGER IDS(NEXTERNAL)
      INTEGER IDS_BORN(NEXTERNAL-1)
      INTEGER R, B
      INTEGER CACHED_REAL(NEXTERNAL, FKS_CONFIGS)
      LOGICAL CACHED_OK(FKS_CONFIGS)
      INTEGER CACHED_EPOCH
      SAVE CACHED_EPOCH
      DATA CACHED_EPOCH/-1/
      INTEGER CACHED_ID_R
      SAVE CACHED_REAL, CACHED_OK, CACHED_ID_R
      DATA CACHED_OK /FKS_CONFIGS*.FALSE./
      DATA CACHED_ID_R /-1/
      IF (FRAME_ID.NE.CACHED_ID_R.OR.NATIVE_EPOCH.NE.CACHED_EPOCH) THEN
        DO R=1,FKS_CONFIGS
          CACHED_OK(R)=.FALSE.
        ENDDO
        CACHED_ID_R=FRAME_ID
        CACHED_EPOCH=NATIVE_EPOCH
      ENDIF
      IF (IFKS.GE.1 .AND. IFKS.LE.FKS_CONFIGS) THEN
        IF (CACHED_OK(IFKS)) THEN
          DO R=1,NEXTERNAL
            IDS(R)=CACHED_REAL(R,IFKS)
          ENDDO
          RETURN
        ENDIF
      ENDIF
      CALL GET_FRAME_MASK_BORN(IDS_BORN)
      DO R=1,NEXTERNAL
        IDS(R)=0
        IF (R.EQ.FKS_I_D(IFKS)) THEN
          CYCLE
        ELSEIF (R.LT.FKS_I_D(IFKS)) THEN
          B=R
        ELSE
          B=R-1
        ENDIF
        IF (B.GE.1 .AND. B.LE.NEXTERNAL-1) THEN
          IDS(R)=IDS_BORN(B)
        ENDIF
      ENDDO
      IF (IFKS.GE.1 .AND. IFKS.LE.FKS_CONFIGS) THEN
        DO R=1,NEXTERNAL
          CACHED_REAL(R,IFKS)=IDS(R)
        ENDDO
        CACHED_OK(IFKS)=.TRUE.
      ENDIF
      RETURN
      END
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
