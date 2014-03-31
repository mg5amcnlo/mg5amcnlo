      SUBROUTINE SMATRIX(P,ANS)
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
      INTEGER JC(NEXTERNAL)
      LOGICAL GOODHEL(NCOMB)
      DATA NTRY/0/
      DATA GOODHEL/NCOMB*.FALSE./
%(helicity_lines)s
%(den_factor_line)s
C ----------
C BEGIN CODE
C ----------
      NTRY=NTRY+1
      DO IHEL=1,NEXTERNAL
         JC(IHEL) = +1
      ENDDO
      ANS = 0D0
          DO IHEL=1,NCOMB
             IF (GOODHEL(IHEL) .OR. NTRY .LT. 100) THEN
                 T=MATRIX(P ,NHEL(1,IHEL),JC(1))            
               ANS=ANS+T
                 HELAMP(IHEL)=T
               IF (T .NE. 0D0) THEN
                   GOODHEL(IHEL)=.TRUE.
               ENDIF
             ENDIF
          ENDDO
      ANS=ANS/DBLE(IDEN)
      END
       
       
      REAL*8 FUNCTION MATRIX(P,NHEL,IC)
C  
%(info_lines)s
C
C Returns amplitude squared summed/avg over colors
c for the point with external lines W(0:6,NEXTERNAL)
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
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
      COMPLEX*16 IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
C  
C LOCAL VARIABLES 
C  
      INTEGER I,J
      COMPLEX*16 ZTEMP
      REAL*8 DENOM(NCOLOR), CF(NCOLOR,NCOLOR)
      COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR)
      COMPLEX*16 W(18,NWAVEFUNCS)
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
      DO I = 1, NCOLOR
          ZTEMP = (0.D0,0.D0)
          DO J = 1, NCOLOR
              ZTEMP = ZTEMP + CF(J,I)*JAMP(J)
          ENDDO
          MATRIX = MATRIX+ZTEMP*DCONJG(JAMP(I))/DENOM(I)   
      ENDDO

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

