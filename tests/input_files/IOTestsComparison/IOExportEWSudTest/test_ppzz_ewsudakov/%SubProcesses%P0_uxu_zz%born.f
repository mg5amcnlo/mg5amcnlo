      subroutine sborn(p,ans)
      use mc_born_types
      implicit none
      include 'nexternal.inc'
      include 'orders.inc'
      double precision p(0:3,nexternal-1),ans
      type(BornRequest) request
      type(BornResult) result
      double precision amp2(2),jamp2(0:1)
      common/to_amps/amp2,jamp2
      double complex ans_cnt(2,nsplitorders)
      common/c_born_cnt/ans_cnt
      double precision wgt_ME_born,wgt_ME_real
      common/c_wgt_ME_tree/wgt_ME_born,wgt_ME_real
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      call mc_born_local(p,request,result)
      ans=result%%born
      wgt_ME_born=ans
      amp_split=result%%amplitudes
      amp_split_cnt=result%%split_counterterms
      ans_cnt=result%%counterterms
      amp2=result%%diagrams
      jamp2(0)=1d0
      jamp2(1:1)=result%%flows
      calculatedBorn=.true.
      end
      subroutine sborn_onehel(p,nhel,hell,ans)
      use mc_born_types
      implicit none
      include 'nexternal.inc'
      include 'orders.inc'
      double precision p(0:3,nexternal-1),ans
      type(BornRequest) request
      type(BornResult) result
      integer nhel(nexternal-1),hell
      double complex amp_split_ewsud(amp_split_size)
      common/to_amp_split_ewsud/amp_split_ewsud
      double complex amp_split_ewsud_lo2(amp_split_size)
      common/to_amp_split_ewsud_lo2/amp_split_ewsud_lo2
      request%%single_helicity=hell
      request%%helicity=nhel
      call mc_born_local(p,request,result)
      ans=result%%single_helicity
      amp_split_ewsud=result%%ewsudakov
      amp_split_ewsud_lo2=result%%ewsudakov_lo2
      end
      INTEGER FUNCTION SQSOINDEXB(AMPORDERA,AMPORDERB)
      IMPLICIT NONE
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=1, NSQAMPSO=1)
      INTEGER NSPLITORDERS
      PARAMETER (NSPLITORDERS=2)
      INTEGER AMPORDERA, AMPORDERB
      INTEGER I, SQORDERS(NSPLITORDERS)
      INTEGER AMPSPLITORDERS(NAMPSO,NSPLITORDERS)
      DATA (AMPSPLITORDERS(  1,I),I=  1,  2) /    0,    2/
      INTEGER SQSOINDEXB_FROM_ORDERS
      DO I=1,NSPLITORDERS
        SQORDERS(I)=AMPSPLITORDERS(AMPORDERA,I)    +AMPSPLITORDERS(AMPORDERB,I)
      ENDDO
      SQSOINDEXB=SQSOINDEXB_FROM_ORDERS(SQORDERS)
      END
      INTEGER FUNCTION SQSOINDEXB_FROM_ORDERS(ORDERS)
      IMPLICIT NONE
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=1)
      INTEGER NSPLITORDERS
      PARAMETER (NSPLITORDERS=2)
      INTEGER ORDERS(NSPLITORDERS)
      INTEGER I,J
      INTEGER SQSPLITORDERS(NSQAMPSO,NSPLITORDERS)
      DATA (SQSPLITORDERS(  1,I),I=  1,  2) /    0,    4/
      DO I=1,NSQAMPSO
        DO J=1,NSPLITORDERS
          IF (ORDERS(J).NE.SQSPLITORDERS(I,J)) GOTO 1009
        ENDDO
        SQSOINDEXB_FROM_ORDERS = I
        RETURN
 1009 CONTINUE
      ENDDO
      WRITE(*,*) 'ERROR:: Stopping function sqsoindex_from_orders'
      WRITE(*,*) 'Could not find squared orders ',(ORDERS(I),I=1  ,NSPLITORDERS)
      STOP
      END
      INTEGER FUNCTION GETORDPOWFROMINDEX_B(IORDER, INDX)
      IMPLICIT NONE
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=1)
      INTEGER NSPLITORDERS
      PARAMETER (NSPLITORDERS=2)
      INTEGER IORDER, INDX
      INTEGER I
      INTEGER SQSPLITORDERS(NSQAMPSO,NSPLITORDERS)
      DATA (SQSPLITORDERS(  1,I),I=  1,  2) /    0,    4/
      IF (IORDER.GT.NSPLITORDERS.OR.IORDER.LT.1) THEN
        WRITE(*,*) 'INVALID IORDER B', IORDER
        WRITE(*,*) 'SHOULD BE BETWEEN 1 AND ', NSPLITORDERS
        STOP
      ENDIF
      IF (INDX.GT.NSQAMPSO.OR.INDX.LT.1) THEN
        WRITE(*,*) 'INVALID INDX B', INDX
        WRITE(*,*) 'SHOULD BE BETWEEN 1 AND ', NSQAMPSO
        STOP
      ENDIF
      GETORDPOWFROMINDEX_B=SQSPLITORDERS(INDX, IORDER)
      END
      SUBROUTINE GET_NSQSO_B(NSQSO)
      IMPLICIT NONE
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=1)
      INTEGER NSQSO
      NSQSO=NSQAMPSO
      END
