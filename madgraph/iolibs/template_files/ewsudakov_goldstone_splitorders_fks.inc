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
      REAL*8 ANS(NSQAMPSO)
      LOGICAL KEEP_ORDER(NSQAMPSO)
      common /c_keep_order_sdkg%(ime)d/ keep_order
      LOGICAL KEEP_ORDER_CNT(NSPLITORDERS, NSQAMPSO)
      common /c_keep_order_cnt/ keep_order_cnt 
      integer amp_orders(nsplitorders)
      logical firsttime
      data firsttime / .TRUE. /
      double precision tiny
      parameter (tiny = 1d-12)
      double precision max_val
      include 'ewsudakov_haslo.inc'
      double complex amp_split_ewsud(amp_split_size)
      common /to_amp_split_ewsud/ amp_split_ewsud
      double complex amp_split_ewsud_LO2(amp_split_size)
      common /to_amp_split_ewsud_LO2/ amp_split_ewsud_LO2
      integer iden
      parameter (iden = %(den_factor)d)
C
C     FUNCTIONS
C
      integer GETORDPOWFROMINDEX_SDKG%(ime)d
      integer orders_to_amp_split_pos
C
C     stuff for the LO2
      integer lo2_orders(nsplitorders)
      logical orders_equal
      external orders_equal
C
C BEGIN CODE
C

if (firsttime) then
 do i = 1, nsqampso
  keep_order(i) = .true.
C this is for the orders of the born to integrate
  do j = 1, nsplitorders
   if(GETORDPOWFROMINDEX_SDKG%(ime)d(j, i) .gt. born_orders(j)) then
    keep_order(i) = .false.
    EXIT 
   endif
  enddo

  if (keep_order(i)) then
   write(*,*) 'SDK_GOLDSTONE%(ime)d: keeping split order ', i
  else
   write(*,*) 'SDK_GOLDSTONE%(ime)d: not keeping split order ', i
  endif
 enddo

 firsttime = .false.
endif

      CALL EWSDK_GOLD_ME_%(ime)d_SPLITORDERS(P,NHEL,ANS)
      ans_summed = 0d0
      max_val = 0d0

C reset the amp_split_ewsud array
amp_split_ewsud(1:amp_split_size) = dcmplx(0d0,0d0)
amp_split_ewsud_lo2(1:amp_split_size) = dcmplx(0d0,0d0)


do i = 1, nsqampso
 max_val = max(max_val, abs(ans(I)))
enddo

call get_lo2_orders(lo2_orders)

do i = 1, nsqampso
        if (keep_order(i)) then
         ANS_SUMMED = ans_summed + ANS(I)
         do j = 1, nsplitorders
          amp_orders(j) = GETORDPOWFROMINDEX_SDKG%(ime)d(j, i)
         enddo
         !amp_split_ewsud(orders_to_amp_split_pos(amp_orders)) = ans(I)
         if(abs(ans(i)).gt.max_val*tiny) amp_split_ewsud(orders_to_amp_split_pos(amp_orders)) = ans(i) / iden
        endif

C the following is for the LO2
    if (has_lo2) then
     do j = 1, nsplitorders
      amp_orders(j) = GETORDPOWFROMINDEX_SDKG%(ime)d(j, i)
     enddo
     if (orders_equal(amp_orders, lo2_orders)) then
      if(abs(ans(i)).gt.max_val*tiny) amp_split_ewsud_lo2(orders_to_amp_split_pos(amp_orders)) = ans(i) / iden
     endif
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
      REAL*8 CF(NCOLOR,NCOLOR)
      COMPLEX*16 ZTEMP, AMP(NGRAPHS), JAMP(NCOLOR,NAMPSO), W(%(wavefunctionsize)d,NWAVEFUNCS)
      COMPLEX*16 TMP_JAMP(%(nb_temp_jamp)i)
C  
C GLOBAL VARIABLES
C  
C
C FUNCTION
C
      INTEGER SQSOINDEXSDKG%(ime)d
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
              ANS(SQSOINDEXSDKG%(ime)d(M,N))=ANS(SQSOINDEXSDKG%(ime)d(M,N))+ZTEMP*DCONJG(JAMP(I,N))
          ENDDO
      ENDDO
    ENDDO
    END
       
       

C
C Helper functions to deal with the split orders.
C

      INTEGER FUNCTION SQSOINDEXSDKG%(ime)d(AMPORDERA,AMPORDERB)
C
C This functions plays the role of the interference matrix. It can be hardcoded or 
C made more elegant using hashtables if its execution speed ever becomes a relevant
C factor. From two split order indices of the jamps, it return the corresponding
C index in the squared order canonical ordering.
C
C CONSTANTS
C      
      implicit none
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
	  INTEGER NSPLITORDERS
	  PARAMETER (NSPLITORDERS=%(nSplitOrders)d)
C
C ARGUMENTS
C
	  INTEGER AMPORDERA, AMPORDERB
C
C LOCAL VARIABLES
C
      INTEGER I, SQORDERS(NSPLITORDERS)
      INTEGER AMPSPLITORDERS(NAMPSO,NSPLITORDERS)
	  %(ampsplitorders)s
C
C FUNCTION
C
      INTEGER SQSOINDEXSDKG%(ime)d_FROM_ORDERS
C
C BEGIN CODE
C
      DO I=1,NSPLITORDERS
	    SQORDERS(I)=AMPSPLITORDERS(AMPORDERA,I)+AMPSPLITORDERS(AMPORDERB,I)
	  ENDDO
	  SQSOINDEXSDKG%(ime)d=SQSOINDEXSDKG%(ime)d_FROM_ORDERS(SQORDERS)
	  END



      INTEGER FUNCTION SQSOINDEXSDKG%(ime)d_FROM_ORDERS(ORDERS)
C
C From a list of values for the split orders, this function returns the
c corresponding index in the squared orders canonical ordering.
C
      implicit none
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
	  INTEGER NSPLITORDERS
	  PARAMETER (NSPLITORDERS=%(nSplitOrders)d)
C
C ARGUMENTS
C
	  INTEGER ORDERS(NSPLITORDERS)
C
C LOCAL VARIABLES
C
      INTEGER I,J
      INTEGER SQSPLITORDERS(NSQAMPSO,NSPLITORDERS)
%(sqsplitorders)s
C
C BEGIN CODE
C
      DO I=1,NSQAMPSO
	    DO J=1,NSPLITORDERS
		  IF (ORDERS(J).NE.SQSPLITORDERS(I,J)) GOTO 1009
		ENDDO
		SQSOINDEXSDKG%(ime)d_FROM_ORDERS = I
		RETURN
1009    CONTINUE
	  ENDDO

	  WRITE(*,*) 'ERROR:: Stopping function sqsoindex_from_orders'
	  WRITE(*,*) 'Could not find squared orders ',(ORDERS(I),I=1,NSPLITORDERS)
	  STOP

	  END



      INTEGER FUNCTION GETORDPOWFROMINDEX_SDKG%(ime)d(IORDER, INDX)
C
C Return the power of the IORDER-th order appearing at position INDX
C in the split-orders output
C
      implicit none
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
	  INTEGER NSPLITORDERS
	  PARAMETER (NSPLITORDERS=%(nSplitOrders)d)
C
C ARGUMENTS
C
	  INTEGER IORDER, INDX
C
C LOCAL VARIABLES
C
      integer i
      INTEGER SQSPLITORDERS(NSQAMPSO,NSPLITORDERS)
%(sqsplitorders)s
C
C BEGIN CODE
C
      IF (IORDER.GT.NSPLITORDERS.OR.IORDER.LT.1) THEN
      WRITE(*,*) "INVALID IORDER B", IORDER
      WRITE(*,*) "SHOULD BE BETWEEN 1 AND ", NSPLITORDERS
      STOP
      ENDIF

      IF (INDX.GT.NSQAMPSO.OR.INDX.LT.1) THEN
      WRITE(*,*) "INVALID INDX B", INDX
      WRITE(*,*) "SHOULD BE BETWEEN 1 AND ", NSQAMPSO
      STOP
      ENDIF

      GETORDPOWFROMINDEX_SDKG%(ime)d=SQSPLITORDERS(INDX, IORDER)
      END


      SUBROUTINE GET_NSQSO_SDKG%(ime)d(NSQSO)
C
C     Simple subroutine returning the number of squared split order
C     contributions returned in ANS when calling SMATRIX_SPLITORDERS
C
      implicit none
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
	  INTEGER NSQSO

	  NSQSO=NSQAMPSO

      END


