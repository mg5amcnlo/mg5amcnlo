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
	  INTEGER NTRY
      common/INIT/NTRY
C  
C LOCAL VARIABLES 
C  
      INTEGER NHEL(NEXTERNAL,NCOMB)
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
               CALL LOOPMATRIX(P ,NHEL(1,IHEL),IHEL,JC(1),T(1))
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
       
       
      SUBROUTINE LOOPMATRIX(P,NHEL,H,IC,RES)
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
	  INTEGER    NCOMB
      PARAMETER (NCOMB=%(ncomb)d)
	  INTEGER    NCOMBAMP
      PARAMETER (NCOMBAMP=NCOMB*NLOOPAMPS)
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
      COMPLEX*16 IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
	  INTEGER H
      REAL*8 RES(3)
C  
C LOCAL VARIABLES 
C  
      INTEGER I,J,K
      COMPLEX*16 ZTEMP
      INTEGER DENOM(NCOLORLOOP), CF(NCOLORBORN,NCOLORLOOP)
      COMPLEX*16 JAMPL(3,NCOLORLOOP), JAMPB(NCOLORBORN)	  
C  
C GLOBAL VARIABLES
C  
      include 'coupl.inc'
	  INTEGER NTRY
      common/INIT/NTRY
      COMPLEX*16 AMPL(3,NLOOPAMPS), AMP(NBORNAMPS)
      COMPLEX*16 W(20,NWAVEFUNCS)
	  LOGICAL GOODAMP(NLOOPAMPS,NCOMB)	
	  DATA ((GOODAMP(I,J),I=1,NLOOPAMPS),J=1,NCOMB) / NCOMBAMP*.TRUE. /
	  common/SPLIT/W,AMPL,AMP,GOODAMP
C  
C COLOR DATA
C  
%(born_jamps_coefs)s	  
%(loop_jamps_coefs)s
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
DO I=1,NCOLORBORN
  JAMPB(I)=(0.D0,0.D0)
  DO J=1,NCONTRIBAMPSB(I)
	IF (AMPNUMBERSB(I,J)>0) THEN  
      JAMPB(I)=JAMPB(I)+AMP(ABS(AMPNUMBERSB(I,J)))*NUMCCOEFB(I,J)
	ELSE
      JAMPB(I)=JAMPB(I)+IMAG1*AMP(ABS(AMPNUMBERSB(I,J)))*NUMCCOEFB(I,J)
	ENDIF
  ENDDO
  JAMPB(I)=JAMPB(I)/(1.D0*DENOMCCOEFB(I))
ENDDO

C LOOP JAMPS
DO K=1,3
  DO I=1,NCOLORLOOP
    JAMPL(K,I)=(0.D0,0.D0)
    DO J=1,NCONTRIBAMPSL(I)
	  IF (AMPNUMBERSL(I,J)>0) THEN
        JAMPL(K,I)=JAMPL(K,I)+AMPL(K,ABS(AMPNUMBERSL(I,J)))*NUMCCOEFL(I,J)
      ELSE
        JAMPL(K,I)=JAMPL(K,I)+IMAG1*AMPL(K,ABS(AMPNUMBERSL(I,J)))*NUMCCOEFL(I,J)
	  ENDIF
    ENDDO
	JAMPL(K,I)=JAMPL(K,I)/(1.D0*DENOMCCOEFL(I))
  ENDDO
ENDDO

      RES(1) = 0.D0
      RES(2) = 0.D0
      RES(3) = 0.D0
      DO K = 1, 3
          DO I = 1, NCOLORLOOP
              ZTEMP = (0.D0,0.D0)		  
              DO J = 1, NCOLORBORN
                  ZTEMP = ZTEMP + CF(J,I)*JAMPB(J)
              ENDDO
              RES(K) = RES(K)+DBLE(ZTEMP*DCONJG(JAMPL(K,I))/DENOM(I)) 
          ENDDO
		  DO J = 1, NCOLORBORN
              ZTEMP = (0.D0,0.D0)
              DO I = 1, NCOLORLOOP
                  ZTEMP = ZTEMP + CF(J,I)*JAMPL(K,I)/DENOM(I)
              ENDDO
              RES(K) = RES(K)+DBLE(ZTEMP*DCONJG(JAMPB(J)))
          ENDDO
      ENDDO

	  IF (NTRY.LT.2) THEN
	  DO I=1, NLOOPAMPS
          IF (SQRT(ABS(RES(1)+RES(2)+RES(3))).GT.0.D0.and.(ABS(AMPL(1,I)+AMPL(2,I)+AMPL(3,I))/SQRT(ABS(RES(1)+RES(2)+RES(3)))).LT.1.0D-10) THEN
		      GOODAMP(I,H)=.FALSE.
		  ENDIF
      ENDDO
	  ENDIF

      END
