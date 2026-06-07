      SUBROUTINE SMATRIX(P, FLAVOR, ANS)
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
      INTEGER FLAVOR(NEXTERNAL)
C
C     GLOBAL
C
      include 'helamp.inc'
C
C LOCAL VARIABLES
C
      INTEGER NHEL(NEXTERNAL,NCOMB),NTRY
      REAL*8 T
      REAL*8 MATRIX
      INTEGER IHEL,IDEN,I,JJ
      INTEGER JC(NEXTERNAL)
      LOGICAL GOODHEL(NCOMB)
      DATA NTRY/0/
      DATA GOODHEL/NCOMB*.FALSE./
      double precision beampol(2)
      common /to_beampol/ beampol
C
C FUNCTIONS
C
      INTEGER BROKEN_SYM
%(helicity_lines)s
%(den_factor_line)s
C ----------
C BEGIN CODE
C ----------
      NTRY=NTRY+1
      ANS = 0D0
          DO IHEL=1,NCOMB
             IF (GOODHEL(IHEL) .OR. NTRY .LT. 100) THEN
                 T=MATRIX(P ,NHEL(1,IHEL),FLAVOR(1))
               ANS=ANS+T
C HANDLE POLARISED BEAM               
              DO JJ=1,NINCOMING
                IF(BEAMPOL(JJ).NE.1D0.AND.NHEL(JJ,IHEL).EQ.INT(SIGN(1D0,BEAMPOL(JJ)))) THEN
                  T=T*ABS(BEAMPOL(JJ))
                ELSE IF(BEAMPOL(JJ).NE.1D0)THEN
                  T=T*(2D0-ABS(BEAMPOL(JJ)))
                ENDIF
              ENDDO

                 HELAMP(IHEL)=T
               IF (T .NE. 0D0) THEN
                   GOODHEL(IHEL)=.TRUE.
               ENDIF
             ENDIF
          ENDDO
      ANS=ANS/DBLE(IDEN)*BROKEN_SYM(FLAVOR)
      END


      REAL*8 FUNCTION MATRIX(P,NHEL,FLAVOR)
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
      INTEGER FLAVOR(NEXTERNAL)
C
C LOCAL VARIABLES
C
      INTEGER I,J
      COMPLEX*16 ZTEMP
      INTEGER CF(NCOLOR*(NCOLOR+1)/2)
      INTEGER CF_INDEX, DENOM
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

      MATRIX = 0.D0
      CF_INDEX = 0
      DO I = 1, NCOLOR
          ZTEMP = (0.D0,0.D0)
          DO J = I, NCOLOR
              CF_INDEX = CF_INDEX +1
              ZTEMP = ZTEMP + CF(CF_INDEX)*JAMP(J)
          ENDDO
          MATRIX = MATRIX+ZTEMP*DCONJG(JAMP(I))
      ENDDO
      MATRIX = MATRIX /DENOM

      END


      SUBROUTINE GET_HELICITIES(ID,HELSET)
      IMPLICIT NONE
C
C CONSTANTS
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER                 NCOMB
      PARAMETER (             NCOMB=%(ncomb)d)
C
C ARGUMENTS
C
      INTEGER ID, HELSET(NEXTERNAL)
C
C LOCAL VARIABLES
C
      INTEGER NHEL(NEXTERNAL,NCOMB),NTRY
      INTEGER IHEL,IDEN, I
%(helicity_lines)s
%(den_factor_line)s


      DO I=1,NEXTERNAL
        HELSET(I)=NHEL(I,ID)
      ENDDO

      return
      end

%(broken_sym_function)s
