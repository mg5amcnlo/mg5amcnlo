      subroutine extra_cnt(p,icnt,cnts)
      use mc_born_types
      implicit none
      include 'nexternal.inc'
      include 'orders.inc'
      double precision p(0:3,nexternal-1),ans
      type(BornRequest) request
      type(BornResult) result
      integer icnt
      double complex cnts(2,nsplitorders)
      cnts=(0d0,0d0)
      if(icnt.le.0)return
      request%%extra=icnt
      call mc_born_local(p,request,result)
      cnts=result%%extra
      amp_split_cnt=result%%split_counterterms
      end
      INTEGER FUNCTION GET_EXTRA_CNT_COLOR(ICNT,IPART)
      INTEGER ICNT IPART
      INCLUDE 'nexternal.inc'
      INTEGER NEXTERNALB
      PARAMETER(NEXTERNALB=NEXTERNAL-1)
      INTEGER NCNT,I
      PARAMETER (NCNT=1)
      INTEGER CNT_COLOR(NCNT,NEXTERNALB)
      DATA (CNT_COLOR(1,I), I=1,NEXTERNALB) / NEXTERNALB * 1 /
      IF (ICNT.GT.NCNT.OR.ICNT.LE.0) THEN
        WRITE(*,*) 'ERROR#1 in get_extra_cnt_color', ICNT
        STOP
      ENDIF
      IF (IPART.GE.NEXTERNAL.OR.IPART.LE.0) THEN
        WRITE(*,*) 'ERROR#2 in get_extra_cnt_color', IPART
        STOP
      ENDIF
      GET_EXTRA_CNT_COLOR=CNT_COLOR(ICNT,IPART)
      RETURN
      END
      INTEGER FUNCTION GET_EXTRA_CNT_PDG(ICNT,IPART)
      INTEGER ICNT IPART
      INCLUDE 'nexternal.inc'
      INTEGER NEXTERNALB
      PARAMETER(NEXTERNALB=NEXTERNAL-1)
      INTEGER NCNT,I
      PARAMETER (NCNT=1)
      INTEGER CNT_PDG(NCNT,NEXTERNALB)
      DATA (CNT_PDG(1,I), I=1,NEXTERNALB) / NEXTERNALB * 0 /
      IF (ICNT.GT.NCNT.OR.ICNT.LE.0) THEN
        WRITE(*,*) 'ERROR#1 in get_extra_cnt_pdg', ICNT
        STOP
      ENDIF
      IF (IPART.GE.NEXTERNAL.OR.IPART.LE.0) THEN
        WRITE(*,*) 'ERROR#2 in get_extra_cnt_pdg', IPART
        STOP
      ENDIF
      GET_EXTRA_CNT_PDG=CNT_PDG(ICNT,IPART)
      RETURN
      END
      DOUBLE PRECISION FUNCTION GET_EXTRA_CNT_CHARGE(ICNT,IPART)
      INTEGER ICNT IPART
      INCLUDE 'nexternal.inc'
      INTEGER NEXTERNALB
      PARAMETER(NEXTERNALB=NEXTERNAL-1)
      INTEGER NCNT,I
      PARAMETER (NCNT=1)
      DOUBLE PRECISION CNT_CHARGE(NCNT,NEXTERNALB)
      DATA (CNT_CHARGE(1,I), I=1,NEXTERNALB) / NEXTERNALB * 0D0 /
      IF (ICNT.GT.NCNT.OR.ICNT.LE.0) THEN
        WRITE(*,*) 'ERROR#1 in get_extra_cnt_charge', ICNT
        STOP
      ENDIF
      IF (IPART.GE.NEXTERNAL.OR.IPART.LE.0) THEN
        WRITE(*,*) 'ERROR#2 in get_extra_cnt_charge', IPART
        STOP
      ENDIF
      GET_EXTRA_CNT_CHARGE=CNT_CHARGE(ICNT,IPART)
      RETURN
      END
