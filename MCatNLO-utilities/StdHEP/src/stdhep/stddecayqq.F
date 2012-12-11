      subroutine STDDECAYQQ

C...search HEPEVT list for particles which QQ can decay
C...one by one, decay these particles and add the results to the HEPEVT common

      IMPLICIT NONE

#include "stdhep.inc"
#include "qqpars.inc"
#include "qqbrat.inc"

      integer I,NCHK,IQQTYP
      integer qqtran
      external qqtran

      logical lfirst
      data lfirst/.TRUE./
      save lfirst

C...print version number if this is the first call
      if(lfirst)then
        call stdversn
        lfirst=.FALSE.
      endif
      call stdfixmass
C...find particles which QQ will decay
      NCHK = NHEP
      do I=1,NCHK
C...check "stable" particles only
        if(ISTHEP(I).EQ.1)then
C...check for generator-specific "particles"
          if(IDHEP(I).GE.81 .AND. IDHEP(I).LE.99) GO TO 101
          IQQTYP = qqtran(IDHEP(I),2)
C...check for quarks and undefined particles
          if(IQQTYP.LE.0) GO TO 101
C...will QQ decay this particle?
          if(IPLIST(1,IQQTYP).GT.0)then
C...allow this particle to decay
            call STDQQDCY(I)
          endif
 101      CONTINUE
        endif
      enddo
      return
      end
