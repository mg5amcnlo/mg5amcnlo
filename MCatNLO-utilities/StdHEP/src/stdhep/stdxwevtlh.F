      subroutine stdxwevtlh(ilbl,istream,lok)

C...Purpose: to write an event from the standard common block.
C
C	   ilbl = 4	write HEPEVT and HEPEV4 common blocks
C	   ilbl = 5	write HEPEVT, HEPEV2, HEPEV3, and HEPEV4 common blocks
C	   lok  = 0     if no problems were encountered

#include "stdhep.inc"
#include "stdcnt.inc"
#include "stdlun.inc"
#include "mcfio.inc"

      integer ilbl,lok,istream
      integer xdr_stdhep_4, xdr_stdhep_4_multi
      external xdr_stdhep_4, xdr_stdhep_4_multi

      lok=0
      if(NHEP.LE.0)then
        write(lnhout,101) NEVHEP
      elseif(ilbl.eq.4)then
        call stdtcopy(1,istream,lok)
        if(lok.ne.0) go to 700
        if(mcfio_block(istream, MCFIO_STDHEP4, xdr_stdhep_4)
     1      .eq. -1) go to 800
        call mcfio_SetEventInfo(istream, MCFIO_STORENUMBER, nevhep)
        if(mcfio_NextEvent(istream) .eq. -1) go to 900
        nstdwrt = nstdwrt + 1
      elseif(ilbl.eq.5)then
        if(mcfio_block(istream, MCFIO_STDHEP4M, xdr_stdhep_4_multi)
     1      .eq. -1) go to 800
        call mcfio_SetEventInfo(istream, MCFIO_STORENUMBER, nevhep)
        if(mcfio_NextEvent(istream) .eq. -1) go to 900
        nstdwrt = nstdwrt + 1
      endif
      return
  700 write (lnhout,701) NEVHEP
      lok = 4
      stop
  800 write (lnhout,801) NEVHEP
      lok=2
      stop
  900 write (lnhout,901) NEVHEP
      lok=1
      stop
  101 format(/5X,'STDXWEVTLH: no particles to write in event',I8)
  701 format(/5X,'STDXWEVTLH: error copying stdhep block for event ',I8)
  801 format(/5X,'STDXWEVTLH: error filling stdhep block for event ',I8)
  901 format(/5X,'STDXWEVTLH: error writing event ',I8)
      end
