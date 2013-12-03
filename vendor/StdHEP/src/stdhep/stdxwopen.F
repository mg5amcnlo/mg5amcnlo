      subroutine stdxwopen(filename,gtitle,ntries,istream,lok)
c
c  initialize xdr tape writing
c  WARNING:  this routine cannot be used if you want to write anything
c            besides stdhep records
c
      implicit none
#include "mcfio.inc"
#include "stdlun.inc"
#include "stdhd.inc"
      integer istream,lok,ntries
      character *(*) filename
      character *(*) gtitle

      logical lfirst
      data lfirst/.TRUE./
      save lfirst

C...print version number if this is the first call
      if(lfirst)then
        call stdversn
        lfirst=.FALSE.
      endif
c
c      Initialization phase.
c
      lok = 0
      title = gtitle
      numblocks = 8
      blkids(1) = MCFIO_STDHEP
      blkids(2) = MCFIO_STDHEPM
      blkids(3) = MCFIO_STDHEPBEG
      blkids(4) = MCFIO_STDHEPEND
      blkids(5) = MCFIO_STDHEP4
      blkids(6) = MCFIO_STDHEP4M
      blkids(7) = MCFIO_HEPEUP
      blkids(8) = MCFIO_HEPRUP
      
      istream = mcfio_OpenWriteDirect(filename, title, %val(0), 
     &             ntries, blkids, numblocks)
      if (istream .eq. -1) then 
           write(lnhout,1002)
           lok = -1
           stop
      end if
c$$$ THIS HAS BEEN CHANGED BY PT
c$$$      write(lnhout,1001)
      write(99,1001)        
      return
1001  format(' STDXWOPEN WARNING: I/O is initialized for stdhep only')
1002  format(' STDXWOPEN: Cannot open output file, give up ')
      end
