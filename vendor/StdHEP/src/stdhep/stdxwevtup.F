      subroutine stdxwevtup(ilbl,istream,lok)

C...Purpose: to write an event from the standard common block.
C
C	   ilbl = 11	write HEPEUP common block
C	   ilbl = 12	write HEPRUP common block
C	   lok  = 0     if no problems were encountered

#include "hepeup.inc"
#include "heprup.inc"
#include "stdcnt.inc"
#include "stdlun.inc"
#include "mcfio.inc"

      integer ilbl,lok,istream
      integer xdr_hepeup, xdr_heprup
      external xdr_hepeup, xdr_heprup

      lok=0

      if(ilbl.eq.11)then
        if(nup.le.0)then
            write(lnhout,101) nup
	else
            if(mcfio_block(istream, MCFIO_HEPEUP, xdr_hepeup)
     1                   .eq. -1) go to 700
            if(mcfio_NextEvent(istream) .eq. -1) go to 900
            nlhwrt = nlhwrt + 1
	  endif
      elseif(ilbl.eq.12)then
        if(mcfio_block(istream, MCFIO_HEPRUP, xdr_heprup)
     1      .eq. -1) go to 800
        if(mcfio_NextEvent(istream) .eq. -1) go to 900
        nlhwrt = nlhwrt + 1
      endif

      return

  700 write (lnhout,701) 
      lok=2
      stop
  800 write (lnhout,801) 
      lok=2
      stop
  900 write (lnhout,901) 
      lok=1
      stop
  101 format(/5X,'stdxwevtup: no particles to write for HEPEUP block')
  701 format(/5X,'stdxwevtup: error filling Les Houches HEPEUP block ')
  801 format(/5X,'stdxwevtup: error filling Les Houches HEPRUP block ')
  901 format(/5X,'stdxwevtup: error writing Les Houches event ')
      end
