
      subroutine QQHEP(mconv)

C...Convert QQ event record to and from the HEPEVT common block
C...convert (mconv=1) from QQ numbering scheme to PDG numbering scheme
C...     or (mconv=2) from PDG numbering scheme to QQ numbering scheme
C...     or (mconv=3) from PDG numbering scheme to QQ numbering scheme
C...                  but allow "stable" particles to decay

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
      integer I,J,mconv
      integer qqtran
      logical NOMATCH

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
      if(mconv.EQ.1) then
C...convert from QQ to HEPEVT
        NEVHEP = IEVTQQ
        if(NTRKQQ.GT.NMXHEP)
     1         write(lnhout,*) ' QQHEP:  NTRKQQ exceeds NMXHEP!'
        NHEP = MIN(NMXHEP,NTRKQQ)
        do I=1,NHEP
          ISTHEP(I) = 1
          if(ISTBMC(I).EQ.0) ISTHEP(I) = 2
          IDHEP(I) = qqtran(ITYPEV(I,1),mconv)
          JMOHEP(1,I) = IPRNTV(I)
          JMOHEP(2,I) = 0
          if(NDAUTV(I).GT.0) then
            JDAHEP(1,I) = IDAUTV(I)
            JDAHEP(2,I) = IDAUTV(I) + NDAUTV(I) -1
          else
            JDAHEP(1,I) = 0
            JDAHEP(2,I) = 0
          endif
          do J=1,4
            PHEP(J,I) = P4QQ(J,I)
          enddo
          PMTMP = PHEP(4,I)**2-PHEP(1,I)**2-PHEP(2,I)**2-PHEP(3,I)**2
          if(PMTMP.LE.0.) then
            PHEP(5,I) = 0.
          else
            PHEP(5,I) = SQRT(PMTMP)
          endif
          do J=1,3 
            VHEP(J,I) = XVTX(IVPROD(I),J)*1000.
          enddo
          VHEP(4,I) = TVTX(IVPROD(I))*cfactor
        enddo
      elseif(mconv.EQ.2 .OR. mconv.EQ.3) then
C...first, sort HEPEVT by daughter list
        call STDSORT
C...convert from HEPEVT to QQ
        IEVTQQ = NEVHEP
        if(NHEP.GT.MCTRK)
     1         write(lnhout,*) ' QQHEP: NHEP exceeds MCTRK!'
        NTRKQQ = MIN(NHEP,MCTRK)
        NSTBQQ = 0
        NCHGQQ = 0
        NVRTX = 0
        N = 0
        do I=1,NTRKQQ
          IPRNTV(I) = JMOHEP(I,1)
          ITYPEV(I,1) = qqtran(IDHEP(I),2)
          IVPROD(I) = 0
          IDECSV(I) = 0
          IVDECA(I) = 0
          IDAUTV(I) = 0
          NDAUTV(I) = 0
          ITYPEV(I,2) = 0
          ISTBMC(I) = 0
          do J=1,4
            P4QQ(J,I) = PHEP(J,I)
            PSAV(I,J) = PHEP(J,I)
          enddo
          HELCQQ(I) = -100.
          if(ISTHEP(I).EQ.1)then
C...stable
            if(mconv.EQ.3 .AND. IDMC(ITYPEV(I,1)).LT.0)then
C...allow this particle to decay
C...Fill QQ common block /JET/ with information needed by DECADD
              N = N + 1
              if(N.LE.250)then
                K(N,1) = 0
                K(N,2) = ITYPEV(I,1)
                do J=1,5
                  P(N,J) = PHEP(J,I)
                enddo
              endif
            else
C...fill in stable particle information
              NSTBQQ = NSTBQQ + 1
              ITYPEV(I,2) = IDMC(ITYPEV(I,1))
              ISTBMC(I) = NSTBQQ
              IDSTBL(NSTBQQ) = I
              if(CHARGE(ITYPEV(I,1)).NE.0.) NCHGQQ = NCHGQQ + 1
            endif
          else
C...intermediate
C...fragmentation does not always provide daughter information
            if(JDAHEP(1,I).GT.0)then
              IDAUTV(I) = JDAHEP(1,I)
              NDAUTV(I) = JDAHEP(2,I) - JDAHEP(1,I) + 1
            endif
          endif
C...fill production vertex list
          if(I.EQ.1)then
C...by definition, this is the first vertex
            NVRTX = NVRTX + 1
            IVPROD(I) = NVRTX
            do J=1,3
              XVTX(NVRTX,J) = VHEP(J,I)/1000.
            enddo
            TVTX(NVRTX) = VHEP(4,I)/cfactor
            RVTX(NVRTX) = SQRT(XVTX(NVRTX,1)**2 + XVTX(NVRTX,2)**2)
            ITRKIN(NVRTX) = JMOHEP(1,I)
            NTRKOU(NVRTX) = 1
            ITRKOU(NVRTX) = I
            IVKODE(NVRTX) = 1
          else
C...does this match an existing vertex?
            NOMATCH = .TRUE.
            do J=1,NVRTX
              if(JMOHEP(1,I).EQ.ITRKIN(J)) then
                NOMATCH = .FALSE.
                NTRKOU(J) = NTRKOU(J) + 1
              endif
            enddo
            if(NOMATCH)then
C...new vertex
              NVRTX = NVRTX + 1
              IVPROD(I) = NVRTX
              IVDECA(IPRNTV(I)) = NVRTX
              do J=1,3
                XVTX(NVRTX,J) = VHEP(J,I)/1000.
              enddo
              TVTX(NVRTX) = VHEP(4,I)/cfactor
              RVTX(NVRTX) = SQRT(XVTX(NVRTX,1)**2 + XVTX(NVRTX,2)**2)
              ITRKIN(NVRTX) = I
              NTRKOU(NVRTX) = 1
              ITRKOU(NVRTX) = I
              IVKODE(NVRTX) = 1
            endif
          endif
        enddo
        NTRKMC = NTRKQQ
        NSTBMC = NSTBQQ
        NCHGMC = NCHGQQ
C...clean up N after telling us how bad it is...
        if(N.GT.250)then
          write(lnhout,1001) N
          N = 250
        endif
      else
C...unsupported option
        write(lnhout,*) ' QQHEP: unallowed conversion option'
      endif
      return
1001  format(' QQHEP: too many particles for QQ to decay'
     1        /10X,'N was ',I5,' and is reset to 250')
      end
