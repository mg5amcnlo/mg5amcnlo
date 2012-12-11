
      subroutine stdquarklst(iq,nqrk,lsize,lqrk)

C...Search the HEPEVT common block for all mesons and baryons with
C...quark content IQ
C
C     IQ  = quark type
C     NQRK = number of particles with quark type IQ
C     LQRK(LSIZE) = list of particles with quark type IQ

      implicit none

      integer iq,nqrk,lsize
      integer lqrk(lsize)

#include "stdhep.inc"
#include "stdlun.inc"

      integer i,kq,kqa,kq1,kq2,kq3,kqx,kqj,kql,kqr

      nqrk = 0
      do i=1,nhep
        kq=idhep(i)
        kqa=iabs(kq)

        if(kqa.gt.99.and.kqa.lt.10000000) then
C...Subdivide standard ID code into constituent pieces.
          call stdquarks(i,kq1,kq2,kq3,kql,kqj,kqr,kqx)
          if(kq1.eq.iq .or. kq2.eq.iq .or. kq3.eq.iq)then
            nqrk = nqrk + 1
            if(nqrk.le.lsize) lqrk(nqrk) = i
          endif
        endif
      enddo
      if(nqrk.gt.lsize) write(lnhout,1001) nqrk,lsize
      return
1001  format(' stdquarklst: found ',i4,
     1        ' particles, but the list only allows ',i4)
      end
