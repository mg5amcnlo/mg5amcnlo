      subroutine stdtcopy(idir,istr,lok)

C...Purpose: to copy an event to/from the standard common block.
C
      implicit none

#include "stdhep.inc"
#include "hepev4.inc"
#include "stdtmp.inc"
#include "stdlun.inc"
c$$$ THIS HAS BEEN ADDED BY PT
#include "stdcnt.inc"
c$$$

      integer idir,lok,i,k,istr
c$$$ THIS HAS BEEN ADDED BY PT
      DOUBLE PRECISION EVWGT_LH
      COMMON/CEVWGT_LH/EVWGT_LH
      integer idiff,offset
      logical error
      common/cerror/error
      save offset
c$$$
      lok=0
      if(idir.eq.1)then
c... copy from hepevt to stdtmp
        nhept = nhep
c$$$ THIS HAS BEEN ADDED BY PT
        error=.false.
        idiff=nevhep-nstdwrt+offset-1
        if (idiff.ne.0)then
           offset=offset+1
           error=.true.
           return
        endif
        eventweightlh = evwgt_lh
c$$$
        nevhept = nevhep
	idrupt = idruplh
	eventweightt = eventweightlh
	alphaqedt = alphaqedlh
	alphaqcdt = alphaqcdlh
        do i=1,10
	    scalet(i) = scalelh(i)
	enddo
        do i=1,nhep
          isthept(i) = isthep(i)
          idhept(i) = idhep(i)
          do k=1,2
            jmohept(k,i) = jmohep(k,i)
            jdahept(k,i) = jdahep(k,i)
	    icolorflowt(k,i) = icolorflowlh(k,i)
          enddo
          do k=1,5
            phept(k,i) = phep(k,i)
          enddo
          do k=1,4
            vhept(k,i) = vhep(k,i)
          enddo
          do k=1,3
            spint(k,i) = spinlh(k,i)
          enddo
        enddo
      elseif(idir.eq.2)then
c... copy from stdtmp to hepevt
c... allow for multiple interactions
        if((nhep+nhept) .gt. NMXHEP) go to 900
        nevhep = nevhept
c... no multiple interaction option for hepev4 information
	idruplh = idrupt
	eventweightlh = eventweightt
	alphaqedlh = alphaqedt
	alphaqcdlh = alphaqcdt
        do i=1,10
	    scalelh(i) = scalet(i)
	enddo
        do i=1,nhept
          isthep(i+nhep) = isthept(i)
          idhep(i+nhep) = idhept(i)
          do k=1,2
            jmohep(k,i+nhep) = jmohept(k,i)
            jdahep(k,i+nhep) = jdahept(k,i)
	    icolorflowlh(k,i+nhep) = icolorflowt(k,i)
          enddo
          do k=1,5
            phep(k,i+nhep) = phept(k,i)
          enddo
          do k=1,4
            vhep(k,i+nhep) = vhept(k,i)
          enddo
          do k=1,3
            spinlh(k,i+nhep) = spint(k,i)
          enddo
        enddo
        nmulti = nmulti + 1
	if(nmulti.le.NMXMLT) then
	  nevmulti(nmulti) = nevhept
          itrkmulti(nmulti) = nhep + 1
	  mltstr(nmulti) = istr
	  idrupmulti(nmulti) = idrupt
	  eventweightmulti(nmulti) = eventweightt
	  alphaqedmulti(nmulti) = alphaqedt
	  alphaqcdmulti(nmulti) = alphaqcdt
	  do i=1,10
	      scalemulti(i,nmulti) = scalet(i)
	  enddo
	else
	   write(lnhout,902) nmulti,NMXMLT
	endif
C... adjust pointers for "multiple interaction" events
        do i=1,nhept
          jmulti(nhep+i) = nmulti
          do k=1,2
c... make sure 0 pointers remain 0
            if(jmohep(k,i+nhep).ne.0) 
     1	           jmohep(k,i+nhep) = jmohep(k,i+nhep) + nhep
            if(jdahep(k,i+nhep).ne.0)
     1	           jdahep(k,i+nhep) = jdahep(k,i+nhep) + nhep
            if(icolorflowlh(k,i+nhep).ne.0)
     1	           icolorflowlh(k,i+nhep) = icolorflowlh(k,i+nhep) + nhep
          enddo
        enddo
        nhep = nhep + nhept
      else
        write (lnhout,801)
      endif
      return
  900 continue
      write (lnhout,901) nevhept
      lok = 5 
      return
  801 format(/5X,'STDTCOPY: improper calling flag')
  901 format(/5X,'STDTCOPY: event would overflow HEPEVT array size'/
     1        5X,'STDTCOPY: event ',i8,' has been lost')
  902 format(/5X,'STDTCOPY: ',i2,' multiple interactions in this event'/
     1      5X,'STDTCOPY: only ',i2,'multiple interactions are allowed')
      end
