      SUBROUTINE EWSDK_GOLD_ME_%(ime)d(P,NHEL,ANS_SUMMED)
C  
%(info_lines)s
C
C
C     Return the sum of the split orders which are required in orders.inc 
C     With respect to those at the born, the QED power is increased by 2
C     Note that helicities must be passed as an input argument.
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
      REAL*8 P(0:3,NEXTERNAL-1), ANS_SUMMED
      integer nhel(nexternal-1)
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
      double complex amp_split_ewsud(amp_split_size)
      common /to_amp_split_ewsud/ amp_split_ewsud
      integer iden
      parameter (iden = %(den_factor)d)
C
C     FUNCTIONS
C
      integer GETORDPOWFROMINDEX_B
      integer orders_to_amp_split_pos
C
C BEGIN CODE
C

      CALL EWSDK_GOLD_ME_%(ime)d_SPLITORDERS(P,NHEL,ANS)
      ans_summed = 0d0
      max_val = 0d0

C reset the amp_split_ewsud array
amp_split_ewsud(1:amp_split_size) = dcmplx(0d0,0d0)


do i = 1, nsqampso
 max_val = max(max_val, abs(ans(I)))
enddo

      do i = 1, nsqampso
        if (keep_order_cnt(qed_pos, i)) then
         ANS_SUMMED = ans_summed + ANS(I)
         do j = 1, nsplitorders
          amp_orders(j) = GETORDPOWFROMINDEX_B(j, i)
C take into account the fact that this is for EW corrections (QED)
          if (j.eq.qed_pos) amp_orders(j) = amp_orders(j) + 2
         enddo
         !amp_split_ewsud(orders_to_amp_split_pos(amp_orders)) = ans(I)
         if(abs(ans(i)).gt.max_val*tiny) amp_split_ewsud(orders_to_amp_split_pos(amp_orders)) = ans(i) / iden
        endif
      enddo

C this is to avoid fake non-zero contributions 
if (abs(ans_summed).lt.max_val*tiny) ans_summed=0d0

return
END


       
       
      SUBROUTINE EWSDK_GOLD_ME_%(ime)d_SPLITORDERS(p,NHEL,ANS)
C  
%(info_lines)s
C RETURNS THE INTERFERENCE OF THE BORN DIAGRAMS WITH THOSE
C NEEDED FOR THE EVALUATION OF THE EW CORRECTIONS IN THE
C SUDAKOV APPROCIMATION,
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
      PARAMETER ( NGRAPHS=%(ngraphs)d) 
      INTEGER    NWAVEFUNCS, NCOLOR
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d, NCOLOR=%(ncolor)d) 
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
      complex*16 imag1
      parameter (imag1 = (0d0,1d0))
      include "nexternal.inc"
      include "born_nhel.inc"
      INCLUDE "coupl.inc"
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL-1)
      INTEGER NHEL(NEXTERNAL-1)
      REAL*8 ANS(NSQAMPSO)
C  
C LOCAL VARIABLES 
C  
      INTEGER I,J,M,N
      INTEGER IC(NEXTERNAL-1),NMO
      PARAMETER (NMO=NEXTERNAL-1)
      DATA IC /NMO*1/
      REAL*8 DENOM(NCOLOR), CF(NCOLOR,NCOLOR)
      COMPLEX*16 ZTEMP, AMP(NGRAPHS), JAMP(NCOLOR,NAMPSO), W(%(wavefunctionsize)d,NWAVEFUNCS)
C  
C GLOBAL VARIABLES
C  
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
%(helas_calls)s
%(jamp_lines)s

    DO I = 1, NSQAMPSO
    ANS(I) = 0d0
    ENDDO
    DO M = 1, NAMPSO
      DO I = 1, NCOLOR
          ZTEMP = (0.D0,0.D0)
          DO J = 1, NCOLOR
              ZTEMP = ZTEMP + CF(j,i)*JAMP(J,M)
          ENDDO
	  DO N = 1, NAMPSO
              ANS(SQSOINDEXB(M,N))=ANS(SQSOINDEXB(M,N))+ZTEMP*DCONJG(JAMP(I,N))/DENOM(I)
          ENDDO
      ENDDO
    ENDDO
    END
       
       
