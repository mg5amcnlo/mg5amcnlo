
      subroutine STDDSCNDLST(IP,ND2,LSZ2,LD2)

C...Search the HEPEVT common block for all the descendants of IP
C
C     IP = index to HEPEVT common block
C     ND2 = number of descendants found
C     LD2(LSZ2) = list of descendants of IP

      integer IP,ND2,LSZ2,I
      integer LD2(LSZ2)
      logical LPAR

#include "stdhep.inc"
#include "stdlun.inc"

      ND2 = 0
C... use brute force:
C... loop over all particles and see if this is a parent
      do I=1,NHEP
        call STDPARENT(I,IP,LPAR)
        if(LPAR)then
          ND2 = ND2 + 1
          if(ND2.LE.LSZ2) LD2(ND2) = I
        endif
      enddo
      if(ND2.GT.LSZ2) write(lnhout,1001) ND2,LSZ2
      return
1001  format(' STDDSCNDLST: found ',I4,
     1        ' daughters, but the list only allows ',I4)
      end
