      SUBROUTINE SMATRIX_PROD(P, FLAV_IDX, ANS)
C
%(info_lines)s
C
C MadGraph5_aMC@NLO StandAlone Version
C
C Returns amplitude squared summed/avg over colors
c and helicities
c for the point in phase space P(0:3,NEXTERNAL)
C
%(process_lines)s
C
      IMPLICIT NONE
C
C CONSTANTS
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER    NINCOMING
      PARAMETER (NINCOMING=%(nincoming)d)
      INTEGER                 NCOMB
      PARAMETER (             NCOMB=%(ncomb)d)
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL),ANS
C     FLAV_IDX (the dummy argument) is declared with NFLAV below.
C
C LOCAL VARIABLES
C
      INTEGER FLAVOR(NEXTERNAL)
      INTEGER NHEL(NEXTERNAL,NCOMB)
      REAL*8 T
      REAL*8 MATRIX_PROD
      INTEGER IHEL,IDEN,I,JJ
      INTEGER JC(NEXTERNAL)
      INTEGER NFLAV
      PARAMETER (NFLAV=%(nflav)d)
      INTEGER NNTRY_FLAV, NGOODHEL_FLAV
      PARAMETER (NNTRY_FLAV=NFLAV)
      PARAMETER (NGOODHEL_FLAV=NCOMB*NFLAV)
      INTEGER FLAV_IDX
      INTEGER GET_FLAVOR_INDEX_PROD
      INTEGER NTRY(NFLAV)
      LOGICAL GOODHEL(NCOMB,NFLAV)
      DATA NTRY/NNTRY_FLAV*0/
      DATA GOODHEL/NGOODHEL_FLAV*.FALSE./
      double precision beampol(2)
      common /to_beampol/ beampol
C
C FUNCTIONS
C
      INTEGER BROKEN_SYM_PROD
%(helicity_lines)s
%(den_factor_line)s
C ----------
C BEGIN CODE
C ----------
C     FLAV_IDX=0 (or out of range): GET_FLAVOR_INDEX could not resolve the
C     requested flavor -- not an allowed combination, matrix element is zero.
      IF (FLAV_IDX.LT.1 .OR. FLAV_IDX.GT.NFLAV) THEN
        ANS = 0D0
        RETURN
      ENDIF
      CALL GET_FLAVOR_PROD(FLAV_IDX, FLAVOR)
      NTRY(FLAV_IDX)=NTRY(FLAV_IDX)+1
      ANS = 0D0
          DO IHEL=1,NCOMB
             IF (GOODHEL(IHEL,FLAV_IDX) .OR. NTRY(FLAV_IDX) .LT. 100) THEN
                 T=MATRIX_PROD(P ,NHEL(1,IHEL),FLAV_IDX)
C Support for polarised beam
                 DO JJ=1,NINCOMING
                   IF(BEAMPOL(JJ).NE.1D0.AND.NHEL(JJ,IHEL).EQ.INT(SIGN(1D0,BEAMPOL(JJ)))) THEN
                     T=T*ABS(BEAMPOL(JJ))
                   ELSE IF(BEAMPOL(JJ).NE.1D0)THEN
                     T=T*(2D0-ABS(BEAMPOL(JJ)))
                   ENDIF
                 ENDDO

               ANS=ANS+T
               IF (T .NE. 0D0) THEN
                   GOODHEL(IHEL,FLAV_IDX)=.TRUE.
               ENDIF
             ENDIF
          ENDDO
      ANS=ANS/DBLE(IDEN)*BROKEN_SYM_PROD(FLAVOR)
      END


      REAL*8 FUNCTION MATRIX_PROD(P,NHEL,FLAV_IDX)
      use model_object
C
%(info_lines)s
C
C Returns amplitude squared summed/avg over colors
c for the point with external lines W(0:6,NEXTERNAL)
C
%(process_lines)s
C
      use aloha_object
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
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
      COMPLEX*16 IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL)
      INTEGER FLAV_IDX
C
C LOCAL VARIABLES
C
      INTEGER FLAVOR(NEXTERNAL)
      INTEGER I,J
      COMPLEX*16 ZTEMP
      INTEGER CF(NCOLOR*(NCOLOR+1)/2)
      INTEGER DENOM,CF_INDEX
      COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR)
      COMPLEX*16 TMP_JAMP(%(nb_temp_jamp)i)
      type(aloha) W(NWAVEFUNCS)
      COMPLEX*16 DUM0,DUM1
      DATA DUM0, DUM1/(0d0, 0d0), (1d0, 0d0)/
%(flavor_mask_decl)s
C
C GLOBAL VARIABLES
C
      include 'coupl.inc'
%(global_variable)s
C
C COLOR DATA
C
%(color_data_lines)s
C ----------
C BEGIN CODE
C ----------
%(flavor_mask_setup)s
%(helas_calls)s
%(jamp_lines)s

      MATRIX_PROD = 0.D0
      CF_INDEX = 0
      DO I = 1, NCOLOR
          ZTEMP = (0.D0,0.D0)
          DO J = I, NCOLOR
              CF_INDEX = CF_INDEX +1
              ZTEMP = ZTEMP + CF(CF_INDEX)*JAMP(J)
          ENDDO
          MATRIX_PROD = MATRIX_PROD+ZTEMP*DCONJG(JAMP(I))
      ENDDO
      MATRIX_PROD = MATRIX_PROD/DENOM
%(amp2_lines)s
      END


%(broken_sym_function)s


%(flavor_index_function)s


%(flavor_array_function)s
