
      subroutine STDDAUTR(IP,IDC,LYN)

C...Is IDC a daughter of IP
C
C       IP and IDC are indices to the HEPEVT common block
C
C	LYN = true/false

      integer IP,IDC
      logical LYN
      integer ND,LSZ,J
      PARAMETER (LSZ = 1000)
      integer LD(LSZ)

#include "stdlun.inc"

      LYN = .FALSE.
      call STDDSCNDLST(IP,ND,LSZ,LD)
      if(ND.GT.0)then
        NL = MIN(ND,LSZ)
        do J = 1,NL
          if(LD(J).EQ.IDC) LYN = .TRUE.
        enddo
      endif
      if(ND.GT.LSZ) write(lnhout,1001)
      return
1001  format(' STDDAUTR: could not search entire daughter list',
     1        ' (BUG in stdhep)')
      end
