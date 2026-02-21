      SUBROUTINE SB_SF_%(ilink)3.3d(P,ANS_SUMMED)
C  
%(info_lines)s
C
C
C     Return the sum of the split orders which are required in orders.inc (BORN_ORDERS)
C     Also the values needed for the counterterms are stored in the C_BORN_CNT common block
C
C  
%(process_lines)s
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
      REAL*8 P(0:3,NEXTERNAL), ANS_SUMMED
C
C VARIABLES
C
      INTEGER I,j
      include 'orders.inc'
      REAL*8 ANS(0:NSQAMPSO)
      LOGICAL KEEP_ORDER_CNT(NSPLITORDERS, NSQAMPSO)
      common /c_keep_order_cnt/ keep_order_cnt 
      integer amp_orders(nsplitorders)
      double precision tiny
      parameter (tiny = 1d-12)
      double precision max_val
C
C     FUNCTIONS
C
      integer GETORDPOWFROMINDEX_B
      integer orders_to_amp_split_pos
C
C BEGIN CODE
C
	  CALL SB_SF_%(ilink)3.3d_SPLITORDERS(P,ANS)
C color-linked borns are called for QCD-type emissions
      ans_summed = 0d0
      max_val = 0d0

C reset the amp_split_cnt array
amp_split_cnt(1:amp_split_size,1:2,1:nsplitorders) = dcmplx(0d0,0d0)


do i = 1, nsqampso
 max_val = max(max_val, abs(ans(I)))
enddo

      do i = 1, nsqampso
        if (keep_order_cnt(qcd_pos, i)) then
         ANS_SUMMED = ans_summed + ANS(I)
         do j = 1, nsplitorders
          amp_orders(j) = GETORDPOWFROMINDEX_B(j, i)
C take into account the fact that this is for QCD
          if (j.eq.qcd_pos) amp_orders(j) = amp_orders(j) + 2
         enddo
         !amp_split_cnt(orders_to_amp_split_pos(amp_orders),1,qcd_pos) = ans(I)
         if(abs(ans(i)).gt.max_val*tiny) amp_split_cnt(orders_to_amp_split_pos(amp_orders),1,qcd_pos) = ans(i)
        endif
      enddo

C this is to avoid fake non-zero contributions 
if (abs(ans_summed).lt.max_val*tiny) ans_summed=0d0

return
END


    SUBROUTINE SB_SF_%(ilink)3.3d_SPLITORDERS(P1,ANS)
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
           CALL B_SF_%(ilink)3.3d(IHEL,T)
           DO I=1,NSQAMPSO
             ANS(I)=ANS(I)+T(I)
           ENDDO
         ENDIF
      ENDDO
      DO I=1,NSQAMPSO
        ANS(I)=ANS(I)/DBLE(IDEN)
        ANS(0)=ANS(0)+ANS(I)
      ENDDO
      END
       
       
      SUBROUTINE B_SF_%(ilink)3.3d(HELL,ANS)
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
      INTEGER HELL
      REAL*8 ANS(NSQAMPSO)
C  
C LOCAL VARIABLES 
C  
      INTEGER I,J,M,N
      INTEGER CF(NCOLOR2,NCOLOR1),DENOM
      COMPLEX*16 ZTEMP, AMP(NGRAPHS), JAMP1(NCOLOR1,NAMPSO), JAMP2(NCOLOR2,NAMPSO)
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
%(jamp1_lines)s
%(jamp2_lines)s
    DO I = 1, NSQAMPSO
    ANS(I) = 0d0
    ENDDO
    DO M = 1, NAMPSO
      DO I = 1, NCOLOR1
          ZTEMP = (0.D0,0.D0)
          DO J = 1, NCOLOR2
              ZTEMP = ZTEMP + CF(j,i)*JAMP2(J,M)
          ENDDO
	  DO N = 1, NAMPSO
              ANS(SQSOINDEXB(M,N))=ANS(SQSOINDEXB(M,N))+ZTEMP*DCONJG(JAMP1(I,N))/DENOM
          ENDDO
      ENDDO
    ENDDO
    END
       
       
