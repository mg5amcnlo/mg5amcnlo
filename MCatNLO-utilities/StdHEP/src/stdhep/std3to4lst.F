      subroutine std3to4lst
C
C    compare new and old numbering scheme
C
#include "stdlun.inc"
      integer id1,id2,itmp1,itmp2,it1,itt1,it2,itt2
      character*20 hnam1,hnam2,cnam1,cnam2

      integer hepcmp
      external hepcmp

C...special cases
      write(lnhout,1001)
      do 100 i=1,100
        id1=i
	call std3to4prt(id1)
 100  continue
C...miscellaneous generator specific particles
      do l=11,999
        id1=l*10
	call std3to4prt(id1)
      enddo
      do l=9991,9999
        id1=l
	call std3to4prt(id1)
      enddo
C...diquarks
      write(lnhout,1002)
      do 200 i=11,99
        do 190 j=1,9
          id1=100*i+j
	  call std3to4prt(id1)
 190    continue
 200  continue
C...mesons
      write(lnhout,1003)
      do i=1,9
        do j=1,9
          do k=1,9
            do l=0,9
              id1=10000*l+100*i+10*j+k
	      call std3to4prt(id1)
            enddo
          enddo
        enddo
      enddo
C...baryons
      write(lnhout,1004)
      do 400 i=1,9
        do 390 j=1,9
          do 380 k=1,9
            do 370 l=1,9
              id1=1000*i+100*j+10*k+l
	      call std3to4prt(id1)
 370        continue
 380      continue
 390    continue
 400  continue
C...ions
      write(lnhout,1005)
      do 500 j=1,2
        do 490 i=1,4
          do 480 k=1,2
            id1 = 1000000000 + i*1000000 + j*1000 + k
	    call std3to4prt(id1)
 480      continue
 490    continue
 500  continue

      return
 1001 format(//5X,'Special Cases'/7X,'IOLD',7X,'ISTD  HNAM',19x,
     1     'IOLD  ONAME',17X,'IOLD',7X,'ISTD  HNAM',18x,'IOLD  ONAME')
 1002 format(//5X,'Diquarks'/7X,'IOLD',7X,'ISTD  HNAM',19x,
     1     'IOLD  ONAME',17X,'IOLD',7X,'ISTD  HNAM',18x,'IOLD  ONAME')
 1003 format(//5X,'Mesons'/7X,'IOLD',7X,'ISTD  HNAM',19x,
     1     'IOLD  ONAME',17X,'IOLD',7X,'ISTD  HNAM',18x,'IOLD  ONAME')
 1004 format(//5X,'Baryons'/7X,'IOLD',7X,'ISTD  HNAM',19x,
     1     'IOLD  ONAME',17X,'IOLD',7X,'ISTD  HNAM',18x,'IOLD  ONAME')
 1005 format(//5X,'Ions'/7X,'IOLD',7X,'ISTD  HNAM',19x,
     1     'IOLD  ONAME',17X,'IOLD',7X,'ISTD  HNAM',18x,'IOLD  ONAME')
      END
