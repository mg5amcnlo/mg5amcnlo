
      subroutine STDPARENT(IP,IPC,LYN)

C...Is IPC a parent of IP
C
C       IP and IPC are indices to the HEPEVT common block
C
C	LYN = true/false

      integer IP,IPC
      logical LYN
      integer NPR,LSIZE
      PARAMETER (LSIZE = 1000)
      integer LPR(LSIZE)

#include "stdlun.inc"

      LYN = .FALSE.
      call STDPARENTLST(IP,NPR,LSIZE,LPR)
      if(NPR.GT.0)then
        NL = MIN(NPR,LSIZE)
        do J = 1,NL
          if(LPR(J).EQ.IPC) LYN = .TRUE.
        enddo
      endif
      if(NPR.GT.LSIZE) write(lnhout,1001)
      return
1001  format(' STDPARENT: could not search entire parent list',
     1        ' (BUG in stdhep)')
      end
