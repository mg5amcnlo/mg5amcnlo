       subroutine std3to4prt(id1)
c
c this is the guts of std3to4lst
c
#include "stdlun.inc"
      integer id1,id2,itmp1,itmp2,it1,itt1,it2,itt2
      character*20 hnam1,hnam2,cnam1,cnam2
c
      integer hepcmp,hepcmp_old,std3to4tran
      external hepcmp,hepcmp_old,std3to4tran
c
      itmp1=hepcmp_old(id1)
      if(itmp1.gt.0)then
	it1=std3to4tran(id1,1)
	itt1=std3to4tran(it1,2)
	call hepnam_old(id1,cnam1)
	call hepnam(it1,hnam1)
	id2=-id1
	itmp2=hepcmp_old(id2)
	it2=std3to4tran(id2,1)
	if(itmp2.eq.0)then
          write(lnhout,1101) id1,it1,hnam1,itt1,cnam1
	else
          call hepnam_old(id2,cnam2)
          itt2=std3to4tran(it2,2)
	  call hepnam(it2,hnam2)
          write(lnhout,1102) id1,it1,hnam1,itt1,cnam1,
     1               id2,it2,hnam2,itt2,cnam2
        endif
      endif
      return
 1101 format(1X,I10,1X,I10,2X,a16,1X,I10,2X,a16)
 1102 format(1X,I10,1X,I10,2X,a16,1X,I10,2X,a16,
     1          I10,1X,I10,1X,a16,1X,I10,1X,a16)
      end
