
      subroutine STDQQUSET

C  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
C    Initialize QQ stuff
C  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

      IMPLICIT NONE

#include "stdlun.inc"
#include "qqinfo.inc"
#include "qqluns.inc"
#include "qqmisc.inc"

      logical LERROR,lfile
C  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

C     Initialize variables 
      call QBINIT

C     Initialize unit numbers
      LTTOQQ = lnhout
      LTTIQQ = 5
      LOUTQQ = lnhout
      call QQGLUN(LTMPQQ)
      call QQGLUN(LTINQQ)
      call QQGLUN(LTOUQQ)
C     set the QQ user decay file if appropriate
      if(qqufile .ne. ' ')then
        inquire(file=qqufile,exist=lfile)
        if(.not.lfile)then
          write(LNHOUT,1002) qqufile
        else
          newdec = qqufile
        endif
      endif

C     Date and time from Driver
      call DATIMH(DATEQQ, TIMEQQ)
      write(LTTOQQ,1001) DATEQQ,TIMEQQ

C     Initialize QQ
      call QQINIT(LERROR)
      if(LERROR) STOP

      return
1001  format(//' QQ date and time: ',A8,2x,A8)
1002  format(' STDQQUSET: WARNING file ',A80,' does not exist'/
     1       '            The QQ user decay file is not defined')
      end
