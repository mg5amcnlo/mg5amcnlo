      subroutine stdversn
C
C...print STDHEP version number
C
#include "stdver.inc"
#include "stdlun.inc"
      logical lfirst
      data lfirst/.TRUE./
      save lfirst

      if(lfirst)then
        lfirst = .FALSE.
        stdhep_ver  = '5.06.01'
        stdhep_date = 'November 20, 2007'
c$$$ THIS HAS BEEN CHANGED BY PT
c$$$        write(lnhout,1001) stdhep_ver,stdhep_date
        write(99,1001) stdhep_ver,stdhep_date
      endif
1001  format(//
     1 10X,'********************************************************'/
     2 10X,'*       STDHEP version ',a7,' -  ',a20,' *'/
     3 10X,'********************************************************'//)
      return
      end
