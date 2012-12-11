
      subroutine STDPARENTLST(IP,NPRNT,LSIZE,LPRNT)

C...Search the HEPEVT common block for all ancestors of IP
C
C     IP = index to HEPEVT common block
C     NPRNT = number of ancestors found
C     LPRNT(LSIZE) = list of ancestors of IP

      integer IP,NPRNT,LSIZE,ITHIS
      integer LPRNT(LSIZE)

#include "stdhep.inc"
#include "stdlun.inc"

      NPRNT = 0
      ITHIS = IP
      do WHILE(JMOHEP(1,ITHIS).GT.0)
        NPRNT = NPRNT + 1
        if(NPRNT.LE.LSIZE) LPRNT(NPRNT) = JMOHEP(1,ITHIS)
        ITHIS = JMOHEP(1,ITHIS)
      enddo
      if(NPRNT.GT.LSIZE) write(lnhout,1001) NPRNT,LSIZE
      return
1001  format(' STDPARENTLST: found ',I4,
     1        ' parents, but the list only allows ',I4)
      end
