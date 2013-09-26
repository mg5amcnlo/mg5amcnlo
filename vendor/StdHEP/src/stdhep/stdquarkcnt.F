
      subroutine stdquarkcnt(it,iq,iret,ipar)

C...Find out if particle IT or any of it's parents contain quark IQ
C...    IT = index to HEPEVT common block
C...    IQ = quark type 
C
C...	IRET = 0	no match
C...	IRET = 1	this particle contains quark IQ
C...	IRET = 2	parent IPAR contains quark IQ
C...	IPAR = 0 unless IRET = 2

      implicit none

      integer iq,iret,ipar,npr,lsz,i
      integer it,kqx,kq3,kq2,kq1,kqj,kql,kqr
      
      parameter (lsz=500)
      integer lpr(lsz)

#include "stdhep.inc"
#include "stdlun.inc"

      iret = 0
      ipar = 0
C...Subdivide standard ID code into constituent pieces.
      call stdquarks(it,kq1,kq2,kq3,kql,kqj,kqr,kqx)
      if(kq1.eq.iq .or. kq2.eq.iq .or. kq3.eq.iq) then
        iret = 1
      else
C... search the parent list
        call stdparentlst(it,npr,lsz,lpr)
        do i=1,npr
          call stdquarks(lpr(i),kq1,kq2,kq3,kql,kqj,kqr,kqx)
          if(kq1.eq.iq .or. kq2.eq.iq .or. kq3.eq.iq) then
            iret = 2
            ipar = lpr(i)
          endif
        enddo
      endif
      return
      end
