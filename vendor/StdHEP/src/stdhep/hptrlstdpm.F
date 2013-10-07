
      subroutine hptrlstdpm
C
C    compare particle ID for herwig and standard
C
#include "stdlun.inc"
#include "dpm_dpar.inc"

      integer i,id1,it1,itt1
      integer dpmtran
      external dpmtran
      character*20 cnam1
      character*8 blnk

      blnk = ' '
      write(lnhout,1001)
C... 
      do 100 i=1,210
        id1 = mpdgha(i)
	if(id1.eq.99999 .or. id1.eq.0)then
c... dpmjet doesn't translate this particle
          cnam1 = ' '
	  it1 = 0
	  itt1 = 0
	else
	  it1 = dpmtran(id1,1)
	  call hepnam(it1,cnam1)
	  itt1 = dpmtran(it1,2)
	endif
        write(lnhout,1101) i,id1,aname(i),it1,cnam1,itt1
 100  continue
c... remaining unused(?) portion of list
      do i=211,410
        id1 = mpdgha(i)
	if(id1.eq.99999 .or. id1.eq.0)then
          cnam1 = ' '
	  it1 = 0
	  itt1 = 0
	else
	  it1 = dpmtran(id1,1)
	  call hepnam(it1,cnam1)
	  itt1 = dpmtran(it1,2)
	endif
        write(lnhout,1101) i,id1,blnk,it1,cnam1,itt1
      enddo
      return
 1001 format(//15X,'DPMJET Translation'/
     1 6X,'I',5X,'IDPM',2X,'DNAME',8x,'ISTD  STDNAM',14X,'IDPM')
 1101 format(1X,I7,1X,I7,2X,A8,1X,I8,2X,a16,1X,I7)
      end
