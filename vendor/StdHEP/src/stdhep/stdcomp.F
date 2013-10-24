      subroutine stdcomp(icomp)
C
C    compare particle masses
C
C    icomp = 1	qq versus jetset/pythia
C            2	qq versus isajet
C            3	qq versus herwig
C
#include "stdlun.inc"

      integer icomp

      if (icomp.eq.1)then
          call stdmsjet
      elseif(icomp.eq.2)then
          call stdmsisa
      elseif(icomp.eq.2)then
          call stdmshwg
      else
          write(lnhout,1001)
      endif
      return
1001  format(' STDCOMP: icomp must be 1, 2, or 3'/
     1       '          icomp = 1 for qq versus jetset/pythia'/
     2       '          icomp = 2 for qq versus isajet'/
     3       '          icomp = 3 for qq versus herwig')
      end
