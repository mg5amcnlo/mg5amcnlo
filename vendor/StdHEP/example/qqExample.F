      PROGRAM QQTST

C  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
C    Initialize QQ stuff
C  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

      IMPLICIT NONE

#include "qqinfo.inc"
#include "qqluns.inc"
#include "stdlun.inc"

C-- Local variables
      LOGICAL LERROR
      INTEGER IRUN, IEVT, ISTAT
C  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

C     Initialize variables
      CALL QBINIT

C     Initialize unit numbers
      CALL QQLUNINI
      LNHOUT=LOUTQQ
      LTTOQQ=LNHOUT

      OPEN(UNIT=LOUTQQ,FILE='qqExample.lpt',
     1      STATUS='NEW',FORM='FORMATTED')

C     Date and time from Driver
      CALL DATIMH(DATEQQ, TIMEQQ)
      WRITE(LOUTQQ,1001) DATEQQ,TIMEQQ
1001  FORMAT(' QQ date and time: ',A8,2x,A8)

C     Initialize QQ
      CALL QQINIT(LERROR)
      IF(LERROR) STOP
C
      CALL HPTRLSTQ
C
      STOP
      END
