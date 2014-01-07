
      subroutine STDQQDCY(IT)

C... decay particle IT via QQ and add the results to the HEPEVT common block

C       IT = index to HEPEVT common block

      IMPLICIT NONE

#include "stdhep.inc"
#include "stdlun.inc"
#include "qqpars.inc"
#include "qqbrat.inc"
#include "qqtrak.inc"
#include "qqevnt.inc"
#include "qqvrtx.inc"
#include "mcgen.inc"

      double precision cfactor
      integer IT,J,IQQTYP
      integer qqtran
      external qqtran

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

      IQQTYP = qqtran(IDHEP(IT),2)
C...check for quarks and undefined particles
      if(IQQTYP.LE.0) GO TO 900
C...will QQ decay this particle?
      if(IPLIST(1,IQQTYP).GT.0)then
C...Fill QQ common block /JET/ with information needed by DECADD
        IEVTQQ = NEVHEP
        call STDZEROQQ
        NVRTX = 1
        NC = 0
        N =  1
        K(N,1) = 0
        K(N,2) = IQQTYP
        do J=1,5
          P(N,J) = PHEP(J,IT)
        enddo
        do J=1,3
          XVTX(NVRTX,J) = VHEP(J,IT)/1000.
        enddo
        TVTX(NVRTX) = VHEP(4,IT)/cfactor
        RVTX(NVRTX) = SQRT(XVTX(NVRTX,1)**2 + XVTX(NVRTX,2)**2)
        ITRKIN(NVRTX) = 0
        NTRKOU(NVRTX) = 0
        ITRKOU(NVRTX) = 0
        IVKODE(NVRTX) = 1
C...now decay this track
        call DECADD
C...add this information to the list
        call STDQQADD(IT)
      endif
      return
  900 CONTINUE
      write(lnhout,1001) IDHEP(IT)
 1001 format(' STDQQDCY: cannot decay particle ',I7,' in QQ')
      return
      end
