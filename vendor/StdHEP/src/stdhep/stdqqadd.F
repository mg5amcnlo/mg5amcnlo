
      subroutine STDQQADD(IHEP)

C...add QQ tracks to HEPEVT common
C...the first QQ particle corresponds to HEPEVT particle IHEP

      IMPLICIT NONE

#include "stdhep.inc"
#include "stdlun.inc"
#include "qqpars.inc"
#include "qqprop.inc"
#include "qqtrak.inc"
#include "qqevnt.inc"
#include "qqvrtx.inc"
#include "mcgen.inc"

      double precision PMTMP,cfactor
      integer I,J,L,NEW,NADD,IHEP,IOFF
      integer qqtran

      logical lfirst
      data lfirst/.TRUE./

c... convert picoseconds to mm for decay time
      data cfactor /0.299792458/

      save lfirst, cfactor

C...print version number if this is the first call
      if(lfirst)then
        call stdversn
        lfirst=.FALSE.
      endif
C...figure out the offsets
      if(IHEP.GT.0) then
        IOFF = NHEP - 1
        NEW = NTRKQQ + NHEP - 1
      else
        IOFF = NHEP
        NEW = NTRKQQ + NHEP
      endif
C...check array limits
      if(NEW.GT.NMXHEP)then
        write(lnhout,1001) NEW,NHEP
        NADD = NMXHEP - NHEP
        NHEP = NMXHEP
      else
        NADD = NTRKQQ
        NHEP = NEW
      endif
C...add QQ particles onto HEPEVT list
      do L=1,NADD
        if(L.EQ.1 .AND. IHEP.GT.0)then
C...keep existing information
          I = IHEP
        else
C...add a new particle
          I = IOFF + L
          IDHEP(I) = qqtran(ITYPEV(L,1),1)
          JMOHEP(1,I) = IPRNTV(L) + IOFF
          if(IPRNTV(L).EQ.1 .AND. IHEP.GT.0) JMOHEP(1,I) = IHEP
          JMOHEP(2,I) = 0
          do J=1,4
            PHEP(J,I) = P4QQ(J,L)
          enddo
          PMTMP = PHEP(4,I)**2-PHEP(1,I)**2-PHEP(2,I)**2-PHEP(3,I)**2
          if(PMTMP.LE.0.) then
            PHEP(5,I) = 0.
          else
            PHEP(5,I) = SQRT(PMTMP)
          endif
          do J=1,3 
            VHEP(J,I) = XVTX(IVPROD(L),J)*1000.
          enddo
          VHEP(4,I) = TVTX(IVPROD(L))*cfactor
        endif
C...set stable code
        ISTHEP(I) = 1
        if(ISTBMC(L).EQ.0) ISTHEP(I) = 2
C...get daughter information
        if(NDAUTV(L).GT.0) then
          JDAHEP(1,I) = IDAUTV(L) + IOFF
          JDAHEP(2,I) = IDAUTV(L) + NDAUTV(L) - 1 + IOFF
        else
          JDAHEP(1,I) = 0
          JDAHEP(2,I) = 0
        endif
      enddo
      return
1001  format(' STDQQADD:  want to add ',I5,
     1     ' particles, but NHEP is already',I5)
      end
