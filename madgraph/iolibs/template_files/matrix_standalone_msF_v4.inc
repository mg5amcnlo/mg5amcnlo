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
      INTEGER IHEL,IDEN, I
      LOGICAL GOODHEL(NCOMB)
      DATA NTRY/0/
      DATA GOODHEL/NCOMB*.FALSE./
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

      INTEGER FUNCTION BROKEN_SYM(FLAV)

      include 'nexternal.inc'
      INTEGER FLAV(NEXTERNAL)
      INTEGER I,J,K,ICOMP
      INTEGER N_TOT, OLD_FACTOR, TOTAL_FACTOR
      INTEGER NCOMP, NENTRIES
      PARAMETER (NCOMP=%(broken_sym_ncomponents)d)
      PARAMETER (NENTRIES=%(broken_sym_nentries)d)
      INTEGER COMP_BEG(NCOMP), COMP_END(NCOMP), COMP_OLD(NCOMP)
      INTEGER PID_LIST(NENTRIES), PID_WORK(NENTRIES)
      INTEGER BLOCK_START(NENTRIES), BLOCK_LEN(NENTRIES)
      LOGICAL SAME_BLOCK
      DATA COMP_BEG /%(broken_sym_component_starts)s/
      DATA COMP_END /%(broken_sym_component_ends)s/
      DATA COMP_OLD /%(broken_sym_component_old_factors)s/
      DATA PID_LIST /%(broken_sym_pid_list)s/
      DATA BLOCK_START /%(broken_sym_block_starts)s/
      DATA BLOCK_LEN /%(broken_sym_block_lengths)s/

      PID_WORK = PID_LIST
      TOTAL_FACTOR = 1
      DO ICOMP=1,NCOMP
         OLD_FACTOR = COMP_OLD(ICOMP)
         IF (COMP_OLD(ICOMP).GT.1) THEN
         DO I=COMP_BEG(ICOMP),COMP_END(ICOMP)
            IF (PID_WORK(I).EQ.0) CYCLE
            N_TOT = 1
            DO J=I+1,COMP_END(ICOMP)
               IF (PID_WORK(I).EQ.PID_WORK(J)) THEN
                  SAME_BLOCK = .TRUE.
                  IF (BLOCK_LEN(I).NE.BLOCK_LEN(J)) SAME_BLOCK = .FALSE.
                  DO K=1,BLOCK_LEN(I)
                     IF (FLAV(BLOCK_START(I)+K-1).NE.FLAV(BLOCK_START(J)+K-1)) THEN
                        SAME_BLOCK = .FALSE.
                     ENDIF
                  ENDDO
                  IF (SAME_BLOCK) THEN
                     PID_WORK(J) = 0
                     N_TOT = N_TOT + 1
                     OLD_FACTOR = OLD_FACTOR/N_TOT
                  ENDIF
               ENDIF
            ENDDO
         ENDDO
         ENDIF
         TOTAL_FACTOR = TOTAL_FACTOR*OLD_FACTOR
      ENDDO
      BROKEN_SYM = TOTAL_FACTOR
   return
   end
