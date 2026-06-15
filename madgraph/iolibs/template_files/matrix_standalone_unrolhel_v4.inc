      SUBROUTINE %(proc_prefix)sSMATRIX(P, FLAVOR, ANS)
C
%(info_lines)s
C
C MadGraph5_aMC@NLO StandAlone Version - UNROLLED HELICITY
C
C Returns amplitude squared summed/avg over colors and helicities
c for the point in phase space P(0:3,NEXTERNAL).
C
C In this (--unrolhel) variant the helicity loop is unrolled: GET_AMP
C computes the amplitude for ALL helicity combinations at once into the
C 2-D array AMP(NCOMB,NGRAPHS) using the cartesian-product (H) ALOHA
C routines, GET_JAMP turns it into JAMP(NCOLOR,NCOMB), and the squared
C matrix element is summed over the NCOMB helicity combinations.
C
%(process_lines)s
C
      use aloha_object
      IMPLICIT NONE
C
C CONSTANTS
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER                 NCOMB
      PARAMETER (             NCOMB=%(ncomb)d)
      INTEGER    NGRAPHS, NCOLOR
      PARAMETER (NGRAPHS=%(ngraphs)d, NCOLOR=%(ncolor)d)
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL),ANS
      INTEGER FLAVOR(NEXTERNAL)
CF2PY INTENT(OUT) :: ANS
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
CF2PY INTENT(IN) :: FLAVOR(NEXTERNAL)
C
C LOCAL VARIABLES
C
      INTEGER IHEL, IDEN, I
      INTEGER NWARMUP
      PARAMETER (NWARMUP=2)
      REAL*8 T
C     Broken-symmetry (flavor) factor, same as the standard standalone.
      INTEGER %(proc_prefix)sBROKEN_SYM
      COMPLEX*16 AMP(NCOMB,NGRAPHS), JAMP(NCOLOR,NCOMB)
      INTEGER %(proc_prefix)sCF(NCOLOR*(NCOLOR+1)/2)
      INTEGER %(proc_prefix)sDENOM
      common/%(proc_prefix)scolor_matrix/%(proc_prefix)sCF,%(proc_prefix)sDENOM
C     Persistent good-helicity state (shared with GET_AMP).  During the first
C     NWARMUP evaluations every helicity combination is computed and GOODHEL is
C     filled; afterwards the dead (always-zero) combinations are skipped here
C     and their per-call masks let GET_AMP skip the corresponding components.
      LOGICAL GOODHEL(NCOMB), MASKS_READY
      INTEGER NTRY
      common/%(proc_prefix)sunrolhel_state/GOODHEL, NTRY, MASKS_READY
      DATA NTRY/0/
      DATA MASKS_READY/.FALSE./
      DATA GOODHEL/NCOMB*.FALSE./
C     The good-helicity set (and the per-amplitude masks derived from it) are
C     flavor dependent, so restart the warmup whenever the flavor changes.
      INTEGER PREV_FLAVOR(NEXTERNAL)
      LOGICAL FLAVOR_CHANGED
      SAVE PREV_FLAVOR
      DATA PREV_FLAVOR/NEXTERNAL*0/
C
C COLOR DATA
C     (SMATRIX needs no couplings; the amplitude/coupling work is in GET_AMP)
C
%(color_data_lines)s
%(den_factor_line)s
C ----------
C BEGIN CODE
C ----------
      FLAVOR_CHANGED = .FALSE.
      DO I = 1, NEXTERNAL
         IF (FLAVOR(I) .NE. PREV_FLAVOR(I)) FLAVOR_CHANGED = .TRUE.
         PREV_FLAVOR(I) = FLAVOR(I)
      ENDDO
      IF (FLAVOR_CHANGED) THEN
         NTRY = 0
         MASKS_READY = .FALSE.
         DO I = 1, NCOMB
            GOODHEL(I) = .FALSE.
         ENDDO
      ENDIF
      NTRY = NTRY + 1
      CALL %(proc_prefix)sGET_AMP(P, FLAVOR, AMP)
      CALL %(proc_prefix)sGET_JAMP(AMP, JAMP)
      ANS = 0D0
      DO IHEL = 1, NCOMB
         IF (NTRY .GT. NWARMUP .AND. .NOT. GOODHEL(IHEL)) CYCLE
         CALL %(proc_prefix)sGET_MATRIX(JAMP(1,IHEL), T)
         IF (NTRY .LE. NWARMUP .AND. T .NE. 0D0) GOODHEL(IHEL) = .TRUE.
         ANS = ANS + T
      ENDDO
      ANS = ANS / DBLE(IDEN) * %(proc_prefix)sBROKEN_SYM(FLAVOR)
      END


      SUBROUTINE %(proc_prefix)sGET_AMP(P, FLAVOR, AMP)
      use model_object
      use aloha_object
C
%(process_lines)s
C
      IMPLICIT NONE
C
C CONSTANTS
C
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=%(ngraphs)d)
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER    NWAVEFUNCS, NCOLOR
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d, NCOLOR=%(ncolor)d)
      INTEGER                 NCOMB
      PARAMETER (             NCOMB=%(ncomb)d)
      INTEGER    MAXWF
      PARAMETER (MAXWF=%(maxwf)d)
      INTEGER    NWARMUP
      PARAMETER (NWARMUP=2)
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER FLAVOR(NEXTERNAL)
      COMPLEX*16 AMP(NCOMB,NGRAPHS)
CF2PY INTENT(OUT) :: AMP
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
CF2PY INTENT(IN) :: FLAVOR(NEXTERNAL)
C
C LOCAL VARIABLES
C
      INTEGER I, IQH, ICH
C     Wavefunction workspace: type(aloha_H)%%W is a fixed static array
C     (MAXWFSIZE_H,MAXNCOMB_H) so there is no allocation at all; SAVE keeps it
C     off the stack (it is large) and the routine is already non-reentrant.
      type(aloha_H) W(NWAVEFUNCS)
      SAVE W
      COMPLEX*16 SCRATCH(NCOMB)
      LOGICAL MASK_ALL(NCOMB)
C     Per-diagram helicity-index permutation: maps each diagram's cartesian
C     combination order to the common canonical order before the color sum.
      INTEGER HELPERM(NCOMB,NGRAPHS)
%(helperm_data)s
C     Per-amplitude good-helicity masks (computed once after the warmup).
C     MASK_AMP lets an amplitude skip the helicity combinations where IT
C     vanishes; GOODHEL_AMP records, during the warmup, whether each amplitude
C     is ever non-zero for each (canonical) helicity combination.
      LOGICAL MASK_AMP(NCOMB,NGRAPHS), GOODHEL_AMP(NCOMB,NGRAPHS)
      SAVE MASK_AMP, GOODHEL_AMP
      COMPLEX*16 DUM0,DUM1
      DATA DUM0, DUM1/(0d0, 0d0), (1d0, 0d0)/
      double precision bwcutoff
%(flavor_mask_decl)s
C
C GLOBAL VARIABLES
C
      LOGICAL GOODHEL(NCOMB), MASKS_READY
      INTEGER NTRY
      common/%(proc_prefix)sunrolhel_state/GOODHEL, NTRY, MASKS_READY
      include 'coupl.inc'
C ----------
C BEGIN CODE
C ----------
      bwcutoff=15 ! use if $ syntax is defined in the process
C     Wavefunctions are always built in full (all-true mask); the amplitude
C     evaluation, the JAMP construction and the color sum all skip the dead
C     (always-zero) helicity combinations discovered during the warmup.
      MASK_ALL = .TRUE.
      IF (NTRY .EQ. 1) THEN
         DO I=1,NGRAPHS
            DO ICH=1,NCOMB
               GOODHEL_AMP(ICH,I) = .FALSE.
            ENDDO
         ENDDO
      ENDIF
      IF (NTRY .LE. NWARMUP) THEN
         DO I=1,NGRAPHS
            DO ICH=1,NCOMB
               MASK_AMP(ICH,I) = .TRUE.
            ENDDO
         ENDDO
      ELSE IF (.NOT. MASKS_READY) THEN
C        Compute amplitude I at combination ICH only if it feeds a globally
C        good helicity AND that amplitude is itself non-zero there (skips
C        amplitudes that vanish by chirality, etc., even where the total does
C        not).
         DO I=1,NGRAPHS
            DO ICH=1,NCOMB
               MASK_AMP(ICH,I) = GOODHEL(HELPERM(ICH,I))
     &              .AND. GOODHEL_AMP(HELPERM(ICH,I),I)
            ENDDO
         ENDDO
         MASKS_READY = .TRUE.
      ENDIF
C     W is statically allocated (no allocate); the cartesian-product routines
C     fill only the unmasked components of each W(I)%%W.
C     Zero AMP so masked (unfilled) combinations contribute nothing to JAMP.
      DO I=1,NGRAPHS
         DO ICH=1,NCOMB
            AMP(ICH,I) = (0D0,0D0)
         ENDDO
      ENDDO
%(flavor_mask_setup)s
%(helas_calls)s
C     During the warmup, record for every amplitude which helicity
C     combinations make it non-zero (used to build MASK_AMP above).
      IF (NTRY .LE. NWARMUP) THEN
         DO I=1,NGRAPHS
            DO ICH=1,NCOMB
               IF (AMP(ICH,I) .NE. (0D0,0D0)) GOODHEL_AMP(ICH,I) = .TRUE.
            ENDDO
         ENDDO
      ENDIF
C     W is kept allocated (SAVE) for reuse on the next call.
      END


      SUBROUTINE %(proc_prefix)sGET_JAMP(AMP,JAMP)
C
%(process_lines)s
C
      IMPLICIT NONE
C
C CONSTANTS
C
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=%(ngraphs)d)
      INTEGER    NCOLOR
      PARAMETER ( NCOLOR=%(ncolor)d)
      INTEGER                 NCOMB
      PARAMETER (             NCOMB=%(ncomb)d)
      COMPLEX*16 IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
      COMPLEX*16 AMP(NCOMB,NGRAPHS), JAMP(NCOLOR,NCOMB), TMP_JAMP(%(nb_temp_jamp)i)
CF2PY INTENT(OUT) :: JAMP
CF2PY INTENT(IN) :: AMP
C
C LOCAL VARIABLES
C
      INTEGER IH
      INTEGER NWARMUP
      PARAMETER (NWARMUP=2)
C     Shared good-helicity state: after the warmup, bypass the JAMP
C     construction entirely for the dead helicity combinations.
      LOGICAL GOODHEL(NCOMB), MASKS_READY
      INTEGER NTRY
      common/%(proc_prefix)sunrolhel_state/GOODHEL, NTRY, MASKS_READY
C ----------
C BEGIN CODE
C ----------
      DO IH = 1, NCOMB
         IF (NTRY .GT. NWARMUP .AND. .NOT. GOODHEL(IH)) CYCLE
%(jamp_lines)s
      ENDDO
      END


      SUBROUTINE %(proc_prefix)sGET_MATRIX(JAMP,MATRIX)
C
%(process_lines)s
C
      IMPLICIT NONE
C
C CONSTANTS
C
CF2PY INTENT(OUT) :: MATRIX
CF2PY INTENT(IN) :: JAMP
      INTEGER    NCOLOR
      PARAMETER (NCOLOR=%(ncolor)d)
      REAL*8     ZERO,MATRIX
      PARAMETER (ZERO=0D0)
C
C LOCAL VARIABLES
C
      INTEGER I,J
      COMPLEX*16 ZTEMP
      INTEGER CF_INDEX
      INTEGER %(proc_prefix)sCF(NCOLOR*(NCOLOR+1)/2)
      INTEGER %(proc_prefix)sDENOM
      common/%(proc_prefix)scolor_matrix/%(proc_prefix)sCF,%(proc_prefix)sDENOM
      COMPLEX*16 JAMP(NCOLOR)
C ----------
C BEGIN CODE
C ----------
      MATRIX = 0.D0
      CF_INDEX = 0
      DO I = 1, NCOLOR
        ZTEMP = (0.D0,0.D0)
        DO J = I, NCOLOR
          CF_INDEX = CF_INDEX + 1
          ZTEMP = ZTEMP + %(proc_prefix)sCF(CF_INDEX)*JAMP(J)
        ENDDO
        MATRIX = MATRIX+ZTEMP*DCONJG(JAMP(I))/%(proc_prefix)sDENOM
      ENDDO
      END


C     Density-matrix helper required by the standalone driver. Not yet
C     supported in the unrolled-helicity output; returns zeros so the
C     driver links and the (main) SMATRIX path works.
      SUBROUTINE %(proc_prefix)sGET_DENSITY(P, POS, N_CHANGING,
     $  ALLOW_HEL, N_COMB, FLAVOR, ALPHAS, INTER)
      IMPLICIT NONE
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER N_CHANGING, N_COMB
      REAL*8 P(0:3,NEXTERNAL), ALPHAS
      INTEGER POS(N_CHANGING), ALLOW_HEL(N_CHANGING*N_COMB)
      INTEGER FLAVOR(NEXTERNAL)
      DOUBLE COMPLEX INTER((N_COMB*(N_COMB+1))/2)
CF2PY INTENT(OUT) :: INTER((N_COMB*(N_COMB+1))/2)
      INTEGER I
      DO I = 1, (N_COMB*(N_COMB+1))/2
         INTER(I) = (0D0,0D0)
      ENDDO
      END


C     Broken-symmetry (flavor) factor function, identical to the standard
C     standalone output.
%(broken_sym_function)s
