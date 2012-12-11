      subroutine stdprthd
C
C...print  mcfio initial information as read by stdhep
C
#include "stdhd.inc"
#include "stdlun.inc"
      logical lfirst
      data lfirst/.TRUE./
      save lfirst
c
C...print version number if this is the first call
      if(lfirst)then
        call stdversn
        lfirst=.FALSE.
      endif
      write(lnhout,1001) title(1:tlen),date(1:dlen),
     1                   comment(1:clen),numblocks


1001  format(/' StdHep MCFio header information: '/
     1       10x,'title: ',a60/
     2       10x,'date:  ',a60/
     3       10x,a70/
     5       20x,i10,' blocks per event'/)
      return
      end
