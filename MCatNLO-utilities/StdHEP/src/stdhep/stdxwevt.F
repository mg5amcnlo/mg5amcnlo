      subroutine stdxwevt(ilbl,istream,lok)

C...Purpose: to write an event from the standard common block.
C
C	lok = 0 if no problems were encountered

#include "stdhep.inc"
#include "stdcnt.inc"
#include "stdlun.inc"
#include "mcfio.inc"

      integer ilbl,lok,istream
      integer xdr_stdhep, xdr_stdhep_multi
      external xdr_stdhep, xdr_stdhep_multi

c$$$ THIS HAS BEEN ADDED BY PT
      logical error
      common/cerror/error
c$$$

      lok=0
      if(NHEP.LE.0)then
        write(lnhout,101) NEVHEP
      elseif(ilbl.eq.1)then
        call stdtcopy(1,istream,lok)
c$$$ THIS HAS BEEN ADDED BY PT
        if(error)goto 1111
c$$$
        if(lok.ne.0) go to 700
        if(mcfio_block(istream, MCFIO_STDHEP, xdr_stdhep)
     1      .eq. -1) go to 800
        call mcfio_SetEventInfo(istream, MCFIO_STORENUMBER, nevhep)
        if(mcfio_NextEvent(istream) .eq. -1) go to 900
c$$$ THIS HAS BEEN CHANGED BY PT
c$$$        nstdwrt = nstdwrt + 1
 1111   nstdwrt = nstdwrt + 1
c$$$
      elseif(ilbl.eq.2)then
        if(mcfio_block(istream, MCFIO_STDHEPM, xdr_stdhep_multi)
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
  101 format(/5X,'STDXWEVT: no particles to write in event',I8)
  701 format(/5X,'STDXWEVT: error copying stdhep block for event ',I8)
  801 format(/5X,'STDXWEVT: error filling stdhep block for event ',I8)
  901 format(/5X,'STDXWEVT: error writing event ',I8)
      end
