      SUBROUTINE SBORN_HEL(P1,ANS)
C  
%(info_lines)s
C
C RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C AND HELICITIES
C FOR THE POINT IN PHASE SPACE P1(0:3,NEXTERNAL-1)
C  
%(process_lines)s
C
      IMPLICIT NONE
C  
C CONSTANTS
C  
      include "nexternal.inc"
      include "born_nhel.inc"
      INTEGER     NCOMB
      PARAMETER ( NCOMB=  %(ncomb)d )
      INTEGER    THEL
      PARAMETER (THEL=NCOMB*%(nconfs)d)
      INTEGER NGRAPHS
      PARAMETER (NGRAPHS = %(ngraphs)d)
C  
C ARGUMENTS 
C  
      REAL*8 P1(0:3,NEXTERNAL-1),ANS
C  
C LOCAL VARIABLES 
C  
      INTEGER IHEL,IDEN,J
      REAL*8 BORN_HEL
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
      double precision wgt_hel(max_bhel)
      common/c_born_hel/wgt_hel

C ----------
C BEGIN CODE
C ----------
      iden=iden_values(nfksprocess)
      if (calculatedBorn) then
         do j=1,nexternal-1
            if (savemom(j,1).ne.p1(0,j) .or. savemom(j,2).ne.p1(3,j)) then
               calculatedBorn=.false.
               write (*,*) "momenta not the same in Born_hel"
	       stop
            endif
         enddo
      else
         write(*,*) 'Error in born_hel: should be called only with calculatedborn = true'
	 stop
      endif
      ANS = 0D0
      DO IHEL=1,NCOMB
         wgt_hel(ihel)=0d0
         IF (GOODHEL(IHEL,nFKSprocess)) THEN
           wgt_hel(ihel)=BORN_HEL(P1,IHEL)/DBLE(IDEN)
           ANS=ANS+wgt_hel(ihel)
         ENDIF
      ENDDO
      END
       
       
      REAL*8 FUNCTION born_hel(P,HELL)
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
      INTEGER     NGRAPHS
      PARAMETER ( NGRAPHS = %(ngraphs)d ) 
      INTEGER    NCOLOR
      PARAMETER (NCOLOR=%(ncolor)d) 
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
C  
C LOCAL VARIABLES 
C  
      INTEGER I,J
      INTEGER CF(NCOLOR*(NCOLOR+1)/2)
      INTEGER DENOM, CF_INDEX
      COMPLEX*16 ZTEMP, AMP(NGRAPHS), JAMP(NCOLOR)
      COMPLEX*16 TMP_JAMP(%(nb_temp_jamp)i)
C  
C GLOBAL VARIABLES
C  
      double complex saveamp(ngraphs,max_bhel)
      common/to_saveamp/saveamp
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn
C  
C COLOR DATA
C  
%(color_data_lines)s
C ----------
C BEGIN CODE
C ----------
      if (.not. calculatedBorn) then
        write(*,*) 'Error in born_hel.f: this should be called only with calculatedborn = true'
        stop
      elseif (calculatedBorn) then
         do i=1,ngraphs
            amp(i)=saveamp(i,hell)
         enddo
      endif
  %(jamp_lines)s
      born_hel = 0.D0 
      CF_INDEX=0
      DO I = 1, NCOLOR
          ZTEMP = (0.D0,0.D0)
          DO J = I, NCOLOR
              CF_INDEX = CF_INDEX + 1
              ZTEMP = ZTEMP + CF(CF_INDEX)*JAMP(J)
          ENDDO
          born_hel =born_hel+ZTEMP*DCONJG(JAMP(I))
      ENDDO
      born_hel = born_hel/DENOM
      END
