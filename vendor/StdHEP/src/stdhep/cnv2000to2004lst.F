      subroutine cnv2000to2004lst
C
C    compare 2000 to 2004 numbers - don't bother with charge conjugates
C
#include "stdlun.inc"
      integer id1,id2,id3,itmp1
      integer i,j,k,l,m,h
      character*20 hnam1

      integer hepcmp_2004, cnv2000to2004
      external hepcmp_2004, cnv2000to2004

C...special cases
      write(lnhout,1001)
      write(lnhout,1005)
      do 100 i=1,100
        itmp1=hepcmp_2004(i)
        if(itmp1.gt.0)then
          id1=i
          call hepnam(id1,hnam1)
	  id2 = cnv2000to2004(id1,2)
	  id3 = cnv2000to2004(id2,1)
          write(lnhout,1101) id1,hnam1,id2,id3
        endif
 100  continue
      do 150 j=11,999
        i=j*10
        itmp1=hepcmp_2004(i)
        if(itmp1.gt.0)then
          id1=i
	  id2 = cnv2000to2004(id1,2)
	  id3 = cnv2000to2004(id2,1)
          call hepnam(id1,hnam1)
          write(lnhout,1101) id1,hnam1,id2,id3
        endif
 150  continue
C...SUSY, technicolor, etc.
      do m=1,5
        do i=0,9
          do j=0,9
            do k=0,9
              do l=0,9
        	 do h=0,6
        	  id1=1000000*m+100000*l+10000*h+100*i+10*j+k
        	  itmp1=hepcmp_2004(id1)
        	  if(itmp1.gt.0)then
                    call hepnam(id1,hnam1)
		    id2 = cnv2000to2004(id1,2)
		    id3 = cnv2000to2004(id2,1)
                    write(lnhout,1101) id1,hnam1,id2,id3
                  endif
        	enddo
              enddo
            enddo
          enddo
        enddo
      enddo
C...miscellaneous generator specific particles
      do l=1,2
        do h=0,9
	  do i=0,9
            do j=1,9
              do k=1,9
        	  id1=9900000+10000*l+1000*h+100*i+10*j+k
        	  itmp1=hepcmp_2004(id1)
        	  if(itmp1.gt.0)then
                    call hepnam(id1,hnam1)
		    id2 = cnv2000to2004(id1,2)
		    id3 = cnv2000to2004(id2,1)
                    write(lnhout,1101) id1,hnam1,id2,id3
                  endif
              enddo
            enddo
          enddo
	enddo
      enddo
C...diquarks
      write(lnhout,1002)
      write(lnhout,1005)
      do 200 i=11,99
        do 190 j=1,10
          id1=100*i+j-1
          itmp1=hepcmp_2004(id1)
          if(itmp1.gt.0)then
	    id2 = cnv2000to2004(id1,2)
	    id3 = cnv2000to2004(id2,1)
            call hepnam(id1,hnam1)
            write(lnhout,1101) id1,hnam1,id2,id3
          endif
 190    continue
 200  continue
C...mesons
      write(lnhout,1003)
      write(lnhout,1005)
      do i=1,9
        do j=1,9
          do k=1,9
            do l=0,9
	      do m=0,8
        	id1=100000*m+10000*l+100*i+10*j+k
        	itmp1=hepcmp_2004(id1)
        	if(itmp1.gt.0)then
		  id2 = cnv2000to2004(id1,2)
		  id3 = cnv2000to2004(id2,1)
		  id4 = cnv2000to2004(id1,1)
                  call hepnam(id1,hnam1)
                  write(lnhout,1102) id1,hnam1,id2,id3,id4
                endif
        	id1=9000000+100000*m+10000*(l-1)+100*i+10*j+k
        	itmp1=hepcmp_2004(id1)
        	if(itmp1.gt.0)then
                  call hepnam(id1,hnam1)
		  id2 = cnv2000to2004(id1,2)
		  id3 = cnv2000to2004(id2,1)
		  id4 = cnv2000to2004(id1,1)
                  write(lnhout,1102) id1,hnam1,id2,id3,id4
                endif
              enddo
            enddo
          enddo
        enddo
      enddo
C...baryons
      write(lnhout,1004)
      write(lnhout,1005)
      do 400 i=1,9
        do 390 j=1,9
          do 380 k=1,9
            do 370 l=1,9
              id1=1000*i+100*j+10*k+l
              itmp1=hepcmp_2004(id1)
              if(itmp1.gt.0)then
                call hepnam(id1,hnam1)
	        id2 = cnv2000to2004(id1,2)
		id3 = cnv2000to2004(id2,1)
                write(lnhout,1101) id1,hnam1,id2,id3
              endif
 370        continue
 380      continue
 390    continue
 400  continue
      return
 1001 format(//10X,'Compare 2000 to 2004 Numbers'
     1     /5X,'Special Cases')
 1002 format(//5X,'Diquarks')
 1003 format(//5X,'Mesons')
 1004 format(//5X,'Baryons')
 1005 format(5X,'ISTD  name',11X,'ISTD->2000',2x,'2000->2004',
     2      2x,'ISTD->2004')
 1101 format(1x,i8,2x,a16,1x,i8,4x,i8)
 1102 format(1x,i8,2x,a16,1x,i8,4x,i8,4x,i8)
      END
