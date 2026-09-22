      SUBROUTINE EXTRA_CNT(P, ICNT, CNTS)
C     call the extra cnt corresponding to icnt
C     may be a dummy function, depending on the process
      IMPLICIT NONE
      INTEGER ICNT, I
      INCLUDE 'nexternal.inc'
      DOUBLE PRECISION P(0:3, NEXTERNAL-1)
      INCLUDE 'orders.inc'
      DOUBLE COMPLEX CNTS(2,NSPLITORDERS)

      DO I = 1, NSPLITORDERS
        CNTS(1,I) = DCMPLX(0D0,0D0)
        CNTS(2,I) = DCMPLX(0D0,0D0)
      ENDDO

      RETURN
      END


      INTEGER FUNCTION GET_EXTRA_CNT_COLOR(ICNT,IPART)
C     return the color of the ipart-th particle of 
C     counterterm icnt
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
C     return the pdg id of the ipart-th particle of 
C     counterterm icnt
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
C     return the charge id of the ipart-th particle of 
C     counterterm icnt
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

