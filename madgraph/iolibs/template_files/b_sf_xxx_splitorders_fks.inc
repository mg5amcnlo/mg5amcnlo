      SUBROUTINE SB%(iborn)d_SF_%(ilink)3.3d(P1,ANS_SUMMED)
C
C     Simple routine wrapper to provide the same interface for
C     backward compatibility for usage without split orders.
C
C  
C CONSTANTS
C 
      include 'nexternal.inc'
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
C  
C ARGUMENTS 
C 
      REAL*8 P(0:3,NEXTERNAL), ANS_SUMMED
C
C VARIABLES
C
      INTEGER I
	  REAL*8 ANS(0:NSQAMPSO)
C
C BEGIN CODE
C
	  CALL SB%(iborn)d_SF_%(ilink)3.3d_SPLITORDERS(P,ANS)
	  ANS_SUMMED=ANS(0)

END


    SUBROUTINE SB%(iborn)d_SF_%(ilink)3.3d_SPLITORDERS(P1,ANS)
C  
%(info_lines)s
C
C RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C AND HELICITIES
C FOR THE POINT IN PHASE SPACE P(0:3,NEXTERNAL-1)
C  
%(process_lines)s
C
      IMPLICIT NONE
C  
C CONSTANTS
C  
      Include "nexternal.inc"
      INTEGER     NCOMB
      PARAMETER ( NCOMB=  %(ncomb)d )
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
      INTEGER    THEL
      PARAMETER (THEL=NCOMB*%(nconfs)d)
      INTEGER NGRAPHS
      PARAMETER (NGRAPHS=   %(ngraphs)d)
C  
C ARGUMENTS 
C  
      REAL*8 P1(0:3,NEXTERNAL-1),ANS(0:NSQAMPSO)
C  
C LOCAL VARIABLES 
C  
      INTEGER IHEL,IDEN,i,j
      double precision T(0:NSQAMPSO)
%(den_factor_lines)s
C  
C GLOBAL VARIABLES
C  
      LOGICAL GOODHEL(NCOMB,%(nconfs)d)
      common /c_goodhel/ goodhel
      double precision savemom(nexternal-1,2)
      common/to_savemom/savemom
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      integer nfksprocess
      common/c_nfksprocess/nfksprocess
C ----------
C BEGIN CODE
C ----------
      iden=iden_values(nfksprocess)
      if (calculatedBorn) then
         do j=1,nexternal-1
            if (savemom(j,1).ne.p1(0,j) .or. savemom(j,2).ne.p1(3,j)) then
               calculatedBorn=.false.
               write(*,*) 'Error in sb_sf: momenta not the same in the born'
               stop
            endif
         enddo
      else
         write(*,*) 'Error in sb_sf: color_linked borns should be called only with calculatedborn = true'
         stop
      endif
      DO I=0,NSQAMPSO
          ANS(I) = 0D0
      ENDDO
      DO IHEL=1,NCOMB
         IF (GOODHEL(IHEL,nFKSprocess)) THEN
           CALL B%(iborn)d_SF_%(ilink)3.3d(P1,IHEL,T)
           DO I=0,NSQAMPSO
             ANS(I)=ANS(I)+T(I)
           ENDDO
         ENDIF
      ENDDO
      DO I=0,NSQAMPSO
        ANS(I)=ANS(I)/DBLE(IDEN)
      ENDDO
      END
       
       
      SUBROUTINE B%(iborn)d_SF_%(ilink)3.3d(P,HELL,ANS)
C  
%(info_lines)s
C RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C FOR THE POINT WITH EXTERNAL LINES W(0:6,NEXTERNAL-1)

%(process_lines)s
C  
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
      INTEGER     NGRAPHS
      PARAMETER ( NGRAPHS = %(ngraphs)d ) 
      INTEGER NCOLOR1, NCOLOR2
      PARAMETER (NCOLOR1=%(ncolor1)d, NCOLOR2=%(ncolor2)d) 
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
      complex*16 imag1
      parameter (imag1 = (0d0,1d0))
      include "nexternal.inc"
      include "born_nhel.inc"
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL-1)
      INTEGER HELL
      REAL*8 ANS(0:NSQAMPSO)
C  
C LOCAL VARIABLES 
C  
      INTEGER I,J,M,N
      REAL*8 DENOM(NCOLOR1), CF(NCOLOR2,NCOLOR1)
      COMPLEX*16 ZTEMP, AMP(NGRAPHS), JAMP1(NCOLOR1,NAMPSO), JAMP2(NCOLOR2,NAMPSO)
C  
C GLOBAL VARIABLES
C  
      double complex saveamp(ngraphs,max_bhel)
      common/to_saveamp/saveamp
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn
C
C FUNCTION
C
      INTEGER SQSOINDEXB%(iborn)s
C  
C COLOR DATA
C  
%(color_data_lines)s
C ----------
C BEGIN CODE
C ----------
      if (.not. calculatedBorn) then
        write(*,*) 'Error in b_sf: color_linked borns should be called only with calculatedborn = true'
        stop
      elseif (calculatedBorn) then
      do i=1,ngraphs
         amp(i)=saveamp(i,hell)
      enddo
      endif
%(jamp1_lines)s
%(jamp2_lines)s
    DO I = 0, NSQAMPSO
    ANS(I) = 0d0
    ENDDO
    DO M = 1, NAMPSO
      DO I = 1, NCOLOR1
          ZTEMP = (0.D0,0.D0)
          DO J = 1, NCOLOR2
              ZTEMP = ZTEMP + CF(j,i)*JAMP2(J,M)
          ENDDO
	  DO N = 1, NAMPSO
              ANS(SQSOINDEXB%(iborn)d(M,N))=ANS(SQSOINDEXB%(iborn)d(M,N))+ZTEMP*DCONJG(JAMP1(I,N))/DENOM(I)
          ENDDO
      ENDDO
    ENDDO
    do i = 1, nsqampso
    ans(0) = ans(0) + ans(i)
    enddo
    END
       
       
