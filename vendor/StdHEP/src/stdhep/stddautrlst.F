
      subroutine STDDAUTRLST(IP,NDAU,LSIZE,LDTR)

C...Search the HEPEVT common block for the immediate daughters of IP
C
C     IP = index to HEPEVT common block
C     NDAU = number of daughters found
C     LDTR(LSIZE) = list of daughters of IP

      integer IP,NDAU,LSIZE
      integer LDTR(LSIZE)

#include "stdhep.inc"
#include "stdlun.inc"

      NDAU = 0
      JD1 = JDAHEP(1,IP)
      JD2 = JDAHEP(2,IP)
      if(JD1.LE.0)then
C... no daughters
      elseif(JD1.GT.JD2)then
C... just search the entire list
        do K=1,NHEP
          if(JMOHEP(1,K).EQ.IP) then
            NDAU = NDAU + 1
            if(NDAU.LE.LSIZE) LDTR(NDAU) = K
          endif
        enddo
      else
C... double check the immediate mother
        do K=JD1,JD2
          if(JMOHEP(1,K).EQ.IP) then
            NDAU = NDAU + 1
            if(NDAU.LE.LSIZE) LDTR(NDAU) = K
          endif
        enddo
      endif
      if(NDAU.GT.LSIZE) write(lnhout,1001) NDAU,LSIZE
      return
1001  format(' STDDAUTRLST: found ',I4,
     1        ' daughters, but the list only allows ',I4)
      end
