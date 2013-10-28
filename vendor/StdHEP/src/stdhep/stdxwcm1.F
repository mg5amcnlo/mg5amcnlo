      subroutine stdxwcm1(ilbl,istream,lok)

C...Purpose: to write begin/end run information in a standard format
C
C	if ilbl = 100   write STDHEP begin run record
C	if ilbl = 200   write STDHEP end run record
C
C	lok = 0 if no problems were encountered

#include "stdcm1.inc"
#include "stdcnt.inc"
#include "stdlun.inc"
#include "mcfio.inc"

      integer ilbl,lok
      integer xdr_stdhep_cm1
      external xdr_stdhep_cm1

      lok=0
      if(nevtlh.eq.0) nevtlh = nlhwrt
      if(ilbl.eq.100)then
        if(mcfio_block(istream, MCFIO_STDHEPBEG, xdr_stdhep_cm1)
     1      .eq. -1) go to 800
        if(mcfio_NextEvent(istream) .eq. -1) go to 900
      elseif(ilbl.eq.200)then
        if(mcfio_block(istream, MCFIO_STDHEPEND, xdr_stdhep_cm1)
     1      .eq. -1) go to 800
        if(mcfio_NextEvent(istream) .eq. -1) go to 900
      else
        lok = 3
        write (lnhout,701) ilbl
      endif

      return
  800 write (lnhout,801) ilbl
      lok=2
      stop
  900 write (lnhout,901)
      lok=1
      stop
  701 format(/5X,'STDXWCM1: called with improper label ',i4)
  801 format(/5X,'STDXWCM1: error filling stdhep cm1 block for label '
     1       ,i4)
  901 format(/5X,'STDXWCM1: error writing stdhep cm1 block ')
      end

