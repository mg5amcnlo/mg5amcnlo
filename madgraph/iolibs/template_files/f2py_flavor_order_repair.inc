      SUBROUTINE FLAVOR_ORDER_REPAIR(GETIDX, GETFLV, NEXT, NINI, NFLAV,
     &   NFROZEN, FROZEN, DOREPAIR, FLAVOR, PDGS, P, PFIX)
C     Re-order the external legs so that GETIDX resolves their flavor.
C
C     The generated FLAV_TABLE tabulates ONE leg ordering per flavor
C     combination: get_external_flavors keeps a single representative per
C     class of orderings that differ only by permuting legs inside the
C     initial or inside the final state, so for `q q' > z q q'` the column
C     (1,2,1,1,2) is there and (2,1,1,2,1) is not.  GETIDX answers the 0
C     sentinel for every other ordering and the matrix element is then 0.
C     That is by design -- absorbing the ordering is the caller's job -- and
C     this is where it is done, once, for every python caller of the
C     all_matrix wrapper (MadSpin's density, reweighting's smatrixhel).
C
C     Moving two legs of the same merged pdg inside the same block -- taking
C     their momentum and their flavor together -- is a pure renaming of
C     external lines, so |M|^2 and every entry of the density are unchanged
C     (the fermionic sign of the exchange is a phase on M and cancels in
C     M M*).  Any ordering that resolves therefore gives the same answer and
C     the first one found is taken.
C
C     Arguments:
C       GETIDX  : the ME's GET_FLAVOR_INDEX (FLAVOR -> 1..NFLAV, 0 if absent)
C       GETFLV  : the ME's GET_FLAVOR (index -> the tabulated FLAVOR column)
C       NEXT    : number of external legs of this ME
C       NINI    : number of initial legs (legs never move across that split)
C       NFLAV   : number of columns of the ME's flavor table
C       NFROZEN, FROZEN : legs that must NOT move.  For the density these are
C                 POS, the legs whose helicity is an open index of rho:
C                 permuting one of those would permute the rows/columns of
C                 rho, which is a genuine ambiguity and not ours to settle.
C       DOREPAIR: .FALSE. leaves the ordering untouched.  SMATRIXHEL passes
C                 NHEL .LT. 1 here: a caller asking for one specific helicity
C                 row names it by leg position, and re-ordering the legs would
C                 answer for a different helicity assignment.  Rather than
C                 guess, such a call keeps the 0 it returned before.
C       FLAVOR  : per-leg group positions, repaired in place
C       PDGS    : per-leg merged pdg codes (read only; two legs may only be
C                 exchanged when they carry the same merged code)
C       P       : input momenta
C       PFIX    : momenta to hand to the matrix element.  Always written --
C                 a copy of P when nothing had to move -- so that the caller
C                 has a single array to pass on and P is never modified.
      IMPLICIT NONE
      INTEGER MAXLEG
      PARAMETER (MAXLEG=%(maxpart)i)
      INTEGER GETIDX
      EXTERNAL GETIDX
      EXTERNAL GETFLV
      INTEGER NEXT, NINI, NFLAV, NFROZEN
      LOGICAL DOREPAIR
      INTEGER FROZEN(*), FLAVOR(*), PDGS(*)
      DOUBLE PRECISION P(0:3,*), PFIX(0:3,*)

      INTEGER FR_I, FR_J, FR_K, FR_LO, FR_HI, FR_IFLAV
      INTEGER CAND(MAXLEG), PERM(MAXLEG), TAKEN(MAXLEG), FTMP(MAXLEG)
      LOGICAL FR_OK, FR_KEEP(MAXLEG)

      DO FR_J = 1, NEXT
        DO FR_K = 0, 3
          PFIX(FR_K, FR_J) = P(FR_K, FR_J)
        ENDDO
      ENDDO

      IF (.NOT. DOREPAIR) RETURN
      IF (NEXT .GT. MAXLEG) RETURN
C     the ordering we were given is tabulated: nothing to repair
      IF (GETIDX(FLAVOR) .NE. 0) RETURN

      DO FR_J = 1, NEXT
        FR_KEEP(FR_J) = .FALSE.
      ENDDO
      DO FR_I = 1, NFROZEN
        IF (FROZEN(FR_I) .GE. 1 .AND. FROZEN(FR_I) .LE. NEXT) THEN
          FR_KEEP(FROZEN(FR_I)) = .TRUE.
        ENDIF
      ENDDO

      DO FR_IFLAV = 1, NFLAV
        CALL GETFLV(FR_IFLAV, CAND)
        DO FR_J = 1, NEXT
          PERM(FR_J) = 0
          TAKEN(FR_J) = 0
        ENDDO
C       Match FLAVOR onto this column leg by leg, taking the fixed points
C       first so that an ordering only moves the legs it has to.  Matching
C       is on equal values inside one block, so the greedy choice is never
C       wrong: if any within-block permutation maps FLAVOR onto CAND, this
C       finds one.
        DO FR_J = 1, NEXT
          IF (FLAVOR(FR_J) .EQ. CAND(FR_J)) THEN
            PERM(FR_J) = FR_J
            TAKEN(FR_J) = 1
          ENDIF
        ENDDO
        FR_OK = .TRUE.
        DO FR_J = 1, NEXT
          IF (PERM(FR_J) .NE. 0) CYCLE
C         a frozen leg that is not already a fixed point cannot be helped
          IF (FR_KEEP(FR_J)) THEN
            FR_OK = .FALSE.
            EXIT
          ENDIF
          IF (FR_J .LE. NINI) THEN
            FR_LO = 1
            FR_HI = NINI
          ELSE
            FR_LO = NINI + 1
            FR_HI = NEXT
          ENDIF
          DO FR_I = FR_LO, FR_HI
            IF (TAKEN(FR_I) .EQ. 0 .AND. .NOT. FR_KEEP(FR_I)
     &         .AND. FLAVOR(FR_I) .EQ. CAND(FR_J)
     &         .AND. PDGS(FR_I) .EQ. PDGS(FR_J)) THEN
              PERM(FR_J) = FR_I
              TAKEN(FR_I) = 1
              EXIT
            ENDIF
          ENDDO
          IF (PERM(FR_J) .EQ. 0) THEN
            FR_OK = .FALSE.
            EXIT
          ENDIF
        ENDDO
        IF (.NOT. FR_OK) CYCLE
C       slot FR_J takes the leg that currently sits in PERM(FR_J)
        DO FR_J = 1, NEXT
          FTMP(FR_J) = FLAVOR(PERM(FR_J))
          DO FR_K = 0, 3
            PFIX(FR_K, FR_J) = P(FR_K, PERM(FR_J))
          ENDDO
        ENDDO
        DO FR_J = 1, NEXT
          FLAVOR(FR_J) = FTMP(FR_J)
        ENDDO
        RETURN
      ENDDO
C     no tabulated ordering matches: leave everything as it was, the matrix
C     element returns 0 exactly as it did before.
      RETURN
      END
