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
        write(*,*) "try to call version number" 
        call stdversn
        write(*,*) "done to call version number" 
        lfirst=.FALSE.
      endif
c
c      Initialization phase.
c
      write(*,*) 'mcfio starting to init'
      call mcfio_init()
      write(*,*) 'start to open the file'
      call stdxwopen(filename,title,ntries,istream,lok)
      write(*,*) "done"
      return
      end
