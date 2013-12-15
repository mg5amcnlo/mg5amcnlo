      subroutine stdxwinit(filename,title,ntries,istream,lok)
c
c  initialize xdr tape writing
c  WARNING:  this routine cannot be used if you want to write anything
c            besides stdhep records
c
      implicit none
#include "mcfio.inc"
#include "stdlun.inc"
      integer istream,lok,ntries
      character *(*) filename
      character *(*) title

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
      call mcfio_init()
      call stdxwopen(filename,title,ntries,istream,lok)
        
      return
      end
