
      subroutine HEPTREE

C...Purpose: to write the tree structure of an event 
C...         from the standard common block.
C

#include "stdhep.inc"
#include "stdlun.inc"

      integer LSZ
      PARAMETER (LSZ = 70)
      integer LMX,I,ND,NL,LDT(LSZ)

      character*20 CHAD(LSZ),CHAP
      character*20 stdexpname
      external stdexpname

C...Write header
      LMX=18
      write(lnhout,1100) NEVHEP

      do I=1,NHEP
C...Get the immediate daughters of this particle
        call STDDAUTRLST(I,ND,LSZ,LDT)
        if(ND.GT.0)then
C...Get particle name, pad it and check it is not too long.
          CHAP = stdexpname(I,LMX)
          if(ND.GT.LSZ) write(lnhout,1001) ND,LSZ
          NL = MIN(ND,LSZ)
          do J=1,NL
C...Get particle name, pad it and check it is not too long.
            CHAD(J) = stdexpname(LDT(J),LMX)
          enddo
C...Write data for particle/jet.
          write(lnhout,1200) I,CHAP,(CHAD(K),K=1,NL)
        endif
      enddo
      return
C...Format statements for output on unit lnhout:
 1001 format(' HEPTREE: the following particle has ',I3,' daughters ',
     1       '- ',I2,' will be printed')
 1100 format(///18X,'Event tree (HEP format)',12X,'Event: ',I8)
 1200 format(1X,I4,2X,A18,' -> ',5(A18,2X)/29X,5(A18,2X)
     1       /29X,5(A18,2X)/29X,5(A18,2X)/29X,5(A18,2X)/29X,5(A18,2X)
     2       /29X,5(A18,2X)/29X,5(A18,2X)/29X,5(A18,2X)/29X,5(A18,2X)
     3       /29X,5(A18,2X)/29X,5(A18,2X)/29X,5(A18,2X)/29X,5(A18,2X))

      end
