      SUBROUTINE SLOOPMATRIX(P,ANS)
C  
%(info_lines)s
C 
C MadGraph StandAlone Version
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
      INTEGER    NCOMB         
      PARAMETER (NCOMB=%(ncomb)d)
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL),ANS(3)
C
C GLOBAL VARIABLES
C
	  REAL*8 LSCALE
      common/CT/LSCALE
C  
C LOCAL VARIABLES 
C  
      INTEGER NHEL(NEXTERNAL,NCOMB),NTRY
      REAL*8 T(3)
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

      LSCALE=DSQRT((P(0,1)+P(0,2))**2-(P(1,1)+P(1,2))**2-(P(2,1)+P(2,2))**2-(P(3,1)+P(3,2))**2)

      NTRY=NTRY+1
      DO IHEL=1,NEXTERNAL
         JC(IHEL) = +1
      ENDDO
      ANS(1) = 0.D0
      ANS(2) = 0.D0
      ANS(3) = 0.D0
          DO IHEL=1,NCOMB
             IF (GOODHEL(IHEL) .OR. NTRY .LT. 2) THEN
               CALL LOOPMATRIX(P ,NHEL(1,IHEL),JC(1),T(1))
               DO I=1,3
                 ANS(I)=ANS(I)+T(I)
               ENDDO
               IF (T(1) .NE. 0D0 .AND. .NOT.    GOODHEL(IHEL)) THEN
                   GOODHEL(IHEL)=.TRUE.
               ENDIF
             ENDIF
          ENDDO
      DO I=1,3          
        ANS(I)=ANS(I)/DBLE(IDEN)
      ENDDO
      END
       
       
      SUBROUTINE LOOPMATRIX(P,NHEL,IC,RES)
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
      INTEGER    NLOOPAMPS, NBORNAMPS
      PARAMETER (NLOOPAMPS=%(nloopamps)d, NBORNAMPS=%(nbornamps)d) 
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d)
      INTEGER    NCOLORLOOP, NCOLORBORN      
      PARAMETER (NCOLORLOOP=%(ncolorloop)d, NCOLORBORN=%(ncolorborn)d) 
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
      COMPLEX*16 IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
      REAL*8 RES(3)
C  
C LOCAL VARIABLES 
C  
      INTEGER I,J,K
      COMPLEX*16 ZTEMP(3)
      REAL*8 DENOM(NCOLORLOOP), CF(NCOLORBORN,NCOLORLOOP)
      COMPLEX*16 AMPL(3,NLOOPAMPS), AMP(NBORNAMPS)
      COMPLEX*16 JAMPL(3,NCOLORLOOP), JAMPB(NCOLORBORN)
      COMPLEX*16 W(20,NWAVEFUNCS)	  
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
DO K=1, 3
  DO I=1, NLOOPAMPS
    AMPL(K,I)=(0.D0,0.D0)
  ENDDO
ENDDO

%(helas_calls)s

C BORN JAMPS
%(born_jamp_lines)s

C LOOP JAMPS
DO K=1,3
%(loop_jamp_lines)s
ENDDO

      RES(1) = 0.D0
      RES(2) = 0.D0
      RES(3) = 0.D0
      DO K = 1, 3
          DO I = 1, NCOLORLOOP
              ZTEMP(1) = (0.D0,0.D0)
              ZTEMP(2) = (0.D0,0.D0)
              ZTEMP(3) = (0.D0,0.D0)
              DO J = 1, NCOLORBORN
                  ZTEMP(K) = ZTEMP(K) + CF(J,I)*JAMPB(J)
              ENDDO
              RES(K) = RES(K)+ZTEMP(K)*DCONJG(JAMPL(K,I))/DENOM(I) 
          ENDDO
      ENDDO
      END
