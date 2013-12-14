      subroutine std3to4(mconv)

C...Purpose: to convert StdHep 3.04 event record contents to or from
C...the new numbering scheme.
C...convert (mconv=1) from StdHep 3.04 numbering scheme to StdHep 4.06 numbering scheme
C...     or (mconv=2) from StdHep 4.06 numbering scheme to StdHep 3.04 numbering scheme

C...Only IDHEP needs to be changed

#include "stdhep.inc"

      integer itmp,std3to4tran,mconv,i,j
      external std3to4tran

      logical lfirst
      data lfirst/.true./
      save lfirst

C...print version number if this is the first call
      if(lfirst)then
        call stdversn
        lfirst=.false.
      endif
C...Conversion from old to new numbering scheme
      if(mconv.EQ.1) then
        do i=1,nhep
            itmp=idhep(i)
            idhep(i) = std3to4tran(itmp,1)
	enddo
C...Conversion from new to old numbering scheme
      elseif(mconv.eq.2)then
        do i=1,nhep
            itmp=idhep(i)
            idhep(i) = std3to4tran(itmp,2)
	enddo
      endif
      return
      end
