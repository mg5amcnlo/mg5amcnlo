      subroutine stdxropen(filename,ntries,istream,lok)
c
c  initialize xdr reading
c
      implicit none
#include "mcfio.inc"
#include "stdlun.inc"
#include "stdhd.inc"
      integer istream,lok,ntries
      character*(*) filename

      logical lfirst
      data lfirst/.TRUE./
      save lfirst
c
c      Initialization phase.
c
C...print version number if this is the first call
      if(lfirst)then
        call stdversn
        lfirst=.FALSE.
      endif
      lok = 0
      istream = mcfio_OpenReadDirect(filename) 
      if (istream .eq. -1) go to 900
      call mcfio_InfoStreamChar(istream,MCFIO_CREATIONDATE,date,dlen)
      call mcfio_InfoStreamChar(istream,MCFIO_TITLE,title,tlen)
      call mcfio_InfoStreamChar(istream,MCFIO_COMMENT,comment,clen)
      call mcfio_InfoStreamInt(istream,MCFIO_NUMEVTS,ntries)
      call mcfio_InfoStreamInt(istream,MCFIO_NUMBLOCKS,numblocks)
      call mcfio_InfoStreamInt(istream,MCFIO_BLOCKIDS,blkids)
      write(lnhout,1001) istream,title(1:tlen),date(1:dlen),
     1                   comment(1:clen),ntries,numblocks
      return
        
 900  continue
      write(lnhout,1002)
      lok = -1
      stop
        
1001  format(/' STDXROPEN: successfully opened input stream ',i5/
     1       10x,'title: ',a60/
     2       10x,'date:  ',a60/
     3       10x,a70/
     4       20x,i10,' events'/
     5       20x,i10,' blocks per event'/)
1002  format(' STDXROPEN: Cannot open input file, give up ')
      end
