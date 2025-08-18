      SUBROUTINE SBORN_HEL(P,ANS_SUMMED)
C
C     Return the sum of the split orders which are required in orders.inc (BORN_ORDERS)
C     Also the values needed for the counterterms are stored in the C_BORN_CNT common block
C
C  
C CONSTANTS
C 
implicit none
      include 'nexternal.inc'
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
C  
C ARGUMENTS 
C 
      REAL*8 P(0:3,NEXTERNAL-1), ANS_SUMMED
C
C VARIABLES
C
      INTEGER I, J
      include 'orders.inc'
      REAL*8 ANS(0:NSQAMPSO)
      INCLUDE 'born_nhel.inc'
      DOUBLE PRECISION WGT_HEL(NSQAMPSO, MAX_BHEL)
      COMMON/C_BORN_HEL_SPLIT/WGT_HEL
      DOUBLE PRECISION WGT_HEL_SUMMED(MAX_BHEL)
      COMMON/C_BORN_HEL/WGT_HEL_SUMMED
C
C     FUNCTIONS
C
      integer GETORDPOWFROMINDEX_B
C
C BEGIN CODE
C
C Store all the orders that come from the diagrams, regardless
C of the fact that they satisfy or not the squared-orders constraints


C look for orders which match the born order constraint 
	  CALL SBORN_HEL_SPLITORDERS(P,ANS)
ans_summed = 0d0
  DO J = 1, MAX_BHEL
    WGT_HEL_SUMMED(J) = 0d0
  ENDDO
  DO I = 1, NSQAMPSO
    ANS_SUMMED = ANS_SUMMED + ANS(I)
    DO J = 1, MAX_BHEL
        WGT_HEL_SUMMED(J) = WGT_HEL_SUMMED(J) + WGT_HEL(I,J)
    ENDDO
  ENDDO

return
END


    SUBROUTINE SBORN_HEL_SPLITORDERS(P1,ANS)
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
      Include "nexternal.inc"
      include "born_nhel.inc"
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
      double precision T(NSQAMPSO)
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
      double precision wgt_hel(NSQAMPSO, max_bhel)
      COMMON/C_BORN_HEL_SPLIT/WGT_HEL
C ----------
C BEGIN CODE
C ----------
      iden=iden_values(nfksprocess)
      if (calculatedBorn) then
         do j=1,nexternal-1
            if (savemom(j,1).ne.p1(0,j) .or. savemom(j,2).ne.p1(3,j)) then
               calculatedBorn=.false.
               write(*,*) 'Error in sborn_hel_splitorders: momenta not the same in the born'
               stop
            endif
         enddo
      else
         write(*,*) 'Error in sborn_hel_splitorders: this should be called only with calculatedborn = true'
         stop
      endif
      DO I=0,NSQAMPSO
          ANS(I) = 0D0
      ENDDO
      DO IHEL=1,NCOMB
         IF (GOODHEL(IHEL,nFKSprocess)) THEN
           CALL BORN_HEL_SPLITORDERS(P1,IHEL,T)
           DO I=1,NSQAMPSO
             wgt_hel(i, ihel) = T(I) / DBLE(IDEN)
             ANS(I)=ANS(I)+T(I)
           ENDDO
         ENDIF
      ENDDO
      DO I=1,NSQAMPSO
        ANS(I)=ANS(I)/DBLE(IDEN)
        ANS(0)=ANS(0)+ANS(I)
      ENDDO
      END
       
       
      SUBROUTINE BORN_HEL_SPLITORDERS(P,HELL,ANS)
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
      INTEGER NCOLOR
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
      REAL*8 ANS(NSQAMPSO)
C  
C LOCAL VARIABLES 
C  
      INTEGER I,J,M,N
      INTEGER CF(NCOLOR*(NCOLOR+1)/2)
      INTEGER DENOM, CF_INDEX
      COMPLEX*16 ZTEMP, AMP(NGRAPHS), JAMP(NCOLOR,NAMPSO)
      COMPLEX*16 TMP_JAMP(%(nb_temp_jamp)i)
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
      INTEGER SQSOINDEXB
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
%(jamp_lines)s
    DO I = 1, NSQAMPSO
    ANS(I) = 0d0
    ENDDO
    DO M = 1, NAMPSO
      CF_INDEX = 0 
      DO I = 1, NCOLOR
          ZTEMP = (0.D0,0.D0)
          DO J = I, NCOLOR
              CF_INDEX = CF_INDEX +1
              ZTEMP = ZTEMP + CF(CF_INDEX)*JAMP(J,M)
          ENDDO
          ANS(SQSOINDEXB(M,M))=ANS(SQSOINDEXB(M,M))+ZTEMP*DCONJG(JAMP(I,M))
      ENDDO
          ANS(SQSOINDEXB(M,M))=ANS(SQSOINDEXB(M,M))/DENOM
    ENDDO
    END
       
       


subroutine PickHelicityMC(p,goodhel,hel,ihel_out,vol)
implicit none
include 'nexternal.inc'
INCLUDE 'born_nhel.inc'
double precision p(0:3, nexternal-1)
integer goodhel(max_bhel),hel(0:max_bhel)
integer ihel_out
double precision vol

INTEGER NSQAMPSO
PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
DOUBLE PRECISION WGT_HEL(NSQAMPSO, MAX_BHEL)
COMMON/C_BORN_HEL_SPLIT/WGT_HEL
double precision sum_hel(nsqampso)
integer i, ihel

integer n_nonzero_ord
double precision sum_all
double precision accum, target 
double precision born_wgt_recomp_direct

double precision ran2

call sborn_hel(p,born_wgt_recomp_direct)

C Loop over the various orders of squared Feynman diagrams and compute for each order the sum
n_nonzero_ord = 0
sum_all = 0d0
do i = 1, nsqampso
sum_hel(i) = 0d0
  do ihel = 1, hel(0)
  if (wgt_hel(i, hel(ihel)).lt.0d0) then
    write(*,*) 'Helicities from squared diagrams must be > 0!'
    stop 1
  endif
    sum_hel(i)=sum_hel(i) + wgt_hel(i, hel(ihel))*dble(goodhel(ihel))
  enddo
  if (sum_hel(i).gt.0d0) then
    n_nonzero_ord = n_nonzero_ord + 1
    sum_all = sum_all + sum_hel(i)
  endif
enddo


target=ran2()
ihel=1
accum=0d0

do i = 1, nsqampso
if (sum_hel(i).eq.0d0) cycle
accum=accum+wgt_hel(i,hel(ihel))/sum_hel(i)*dble(goodhel(ihel))/n_nonzero_ord
enddo

do while (accum.lt.target) 
ihel=ihel+1
do i = 1, nsqampso
if (sum_hel(i).eq.0d0) cycle
accum=accum+wgt_hel(i,hel(ihel))/sum_hel(i)*dble(goodhel(ihel))/n_nonzero_ord
enddo
enddo

vol=0d0
do i = 1, nsqampso
if (sum_hel(i).eq.0d0) cycle
vol=vol+wgt_hel(i,hel(ihel))/sum_hel(i)*dble(goodhel(ihel))/n_nonzero_ord
enddo


ihel_out=ihel

return
end


