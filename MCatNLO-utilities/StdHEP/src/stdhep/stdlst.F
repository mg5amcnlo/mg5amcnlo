      subroutine stdlst
C
C    list particle ID for stdhep 
C
#include "stdlun.inc"
      integer id1,id2,itmp1,itmp2
      integer h,i,j,k,l,m
      character*20 hnam1,hnam2

      integer hepcmp
      external hepcmp

C...special cases
      write(lnhout,1001)
      do m=0,2
	do 100 i=1,100
          id1 = m*10000+i
          itmp1=hepcmp(id1)
          if(itmp1.gt.0)then
            call hepnam(id1,hnam1)
            id2=-id1
            itmp2=hepcmp(id2)
            if(itmp2.eq.0)then
              write(lnhout,1101) id1,hnam1,itmp1
            else
              call hepnam(id2,hnam2)
              write(lnhout,1102) id1,hnam1,itmp1,id2,hnam2,itmp2
            endif
          endif
 100    continue
      enddo
      do 150 j=11,999
        i=j*10
        itmp1=hepcmp(i)
        if(itmp1.gt.0)then
          id1=i
          call hepnam(id1,hnam1)
          id2=-i
          itmp2=hepcmp(id2)
          if(itmp2.eq.0)then
            write(lnhout,1101) id1,hnam1,itmp1
          else
            call hepnam(id2,hnam2)
            write(lnhout,1102) id1,hnam1,itmp1,id2,hnam2,itmp2
          endif
        endif
 150  continue
C...SUSY
      do m=1,2
        do j=0,9
          do k=0,9
             id1=1000000*m+10*j+k
             itmp1=hepcmp(id1)
             if(itmp1.gt.0)then
               id2=-id1
               call hepnam(id1,hnam1)
               itmp2=hepcmp(id2)
               if(itmp2.eq.0)then
                 write(lnhout,1101) id1,hnam1,itmp1
               else
                 call hepnam(id2,hnam2)
                 write(lnhout,1102) 
     1                     id1,hnam1,itmp1,id2,hnam2,itmp2
               endif
             endif
          enddo
        enddo
      enddo
C...R-hadrons [m=1]
        do k=1,4
          do j=1,9
            do i=1,9
              do h=0,9
c.. l=0        	 
                 if(h.eq.0 .or. h.eq.6 .or. h.eq.9 )then
        	   id1=1000000+1000*h+100*i+10*j+k
        	   itmp1=hepcmp(id1)
        	   if(itmp1.gt.0)then
                     id2=-id1
                     call hepnam(id1,hnam1)
                     itmp2=hepcmp(id2)
                     if(itmp2.eq.0)then
                       write(lnhout,1101) id1,hnam1,itmp1
                     else
                       call hepnam(id2,hnam2)
                       write(lnhout,1102) 
     1                     id1,hnam1,itmp1,id2,hnam2,itmp2
                     endif
                   endif
                 endif
c.. l=9        	 
        	 id1=1000000+90000+1000*h+100*i+10*j+k
        	 itmp1=hepcmp(id1)
        	 if(itmp1.gt.0)then
                   id2=-id1
                   call hepnam(id1,hnam1)
                   itmp2=hepcmp(id2)
                   if(itmp2.eq.0)then
                     write(lnhout,1101) id1,hnam1,itmp1
                   else
                     call hepnam(id2,hnam2)
                     write(lnhout,1102) 
     1                     id1,hnam1,itmp1,id2,hnam2,itmp2
                   endif
                 endif
               enddo
            enddo
          enddo
        enddo
C...technicolor, etc.
      do m=3,4
        do i=0,9
          do j=0,9
            do k=0,9
              do l=0,9
        	 do h=0,6
        	   id1=1000000*m+100000*l+10000*h+100*i+10*j+k
        	   itmp1=hepcmp(id1)
        	   if(itmp1.gt.0)then
                     id2=-id1
                     call hepnam(id1,hnam1)
                     itmp2=hepcmp(id2)
                     if(itmp2.eq.0)then
                       write(lnhout,1101) id1,hnam1,itmp1
                     else
                       call hepnam(id2,hnam2)
                       write(lnhout,1102) 
     1                     id1,hnam1,itmp1,id2,hnam2,itmp2
                     endif
                   endif
        	 enddo
               enddo
            enddo
          enddo
        enddo
      enddo
C...miscellaneous generator specific particles
      do l=0,2
        do h=0,9
	  do i=0,9
            do j=1,9
              do k=1,9
        	  id1=9900000+10000*l+1000*h+100*i+10*j+k
        	  itmp1=hepcmp(id1)
        	  if(itmp1.gt.0)then
                    id2=-id1
                    call hepnam(id1,hnam1)
                    itmp2=hepcmp(id2)
                    if(itmp2.eq.0)then
                      write(lnhout,1101) id1,hnam1,itmp1
                    else
                      call hepnam(id2,hnam2)
                      write(lnhout,1102) id1,hnam1,itmp1,id2,hnam2,itmp2
                    endif
                  endif
              enddo
            enddo
          enddo
	enddo
      enddo
C...pentaquarks
      do m=1,8
	do l=1,9
          do h=1,9
	    do i=1,9
              do j=1,9
        	do k=1,9
        	    id1=9000000+100000*m+10000*l+1000*h+100*i+10*j+k
        	    itmp1=hepcmp(id1)
        	    if(itmp1.gt.0)then
                      id2=-id1
                      call hepnam(id1,hnam1)
                      itmp2=hepcmp(id2)
                      if(itmp2.eq.0)then
                	write(lnhout,1101) id1,hnam1,itmp1
                      else
                	call hepnam(id2,hnam2)
                write(lnhout,1102) id1,hnam1,itmp1,id2,hnam2,itmp2
                      endif
                    endif
        	enddo
              enddo
            enddo
	  enddo
	enddo
      enddo
C...diquarks
      write(lnhout,1002)
      do 200 i=11,99
        do 190 j=1,10
          id1=100*i+j-1
          itmp1=hepcmp(id1)
          if(itmp1.gt.0)then
            id2=-id1
            call hepnam(id1,hnam1)
            itmp2=hepcmp(id2)
            if(itmp2.eq.0)then
              write(lnhout,1101) id1,hnam1,itmp1
            else
              call hepnam(id2,hnam2)
              write(lnhout,1102) id1,hnam1,itmp1,id2,hnam2,itmp2
            endif
          endif
 190    continue
 200  continue
C...mesons
      write(lnhout,1003)
      do i=1,9
        do j=1,9
          do k=1,9
            do m=0,8
	      do l=0,9
        	id1=100000*m+10000*l+100*i+10*j+k
        	itmp1=hepcmp(id1)
        	if(itmp1.gt.0)then
                  id2=-id1
                  call hepnam(id1,hnam1)
                  itmp2=hepcmp(id2)
                  if(itmp2.eq.0)then
                    write(lnhout,1101) id1,hnam1,itmp1
                  else
                    call hepnam(id2,hnam2)
                    write(lnhout,1102) id1,hnam1,itmp1,id2,hnam2,itmp2
                  endif
                endif
        	id1=9000000+100000*m+10000*l+100*i+10*j+k
        	itmp1=hepcmp(id1)
        	if(itmp1.gt.0)then
                  id2=-id1
                  call hepnam(id1,hnam1)
                  itmp2=hepcmp(id2)
                  if(itmp2.eq.0)then
                    write(lnhout,1101) id1,hnam1,itmp1
                  else
                    call hepnam(id2,hnam2)
                    write(lnhout,1102) id1,hnam1,itmp1,id2,hnam2,itmp2
                  endif
                endif
              enddo
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
              itmp1=hepcmp(id1)
              if(itmp1.gt.0)then
                id2=-id1
                call hepnam(id1,hnam1)
                itmp2=hepcmp(id2)
                if(itmp2.eq.0)then
                  write(lnhout,1101) id1,hnam1,itmp1
                else
                  call hepnam(id2,hnam2)
                  write(lnhout,1102) id1,hnam1,itmp1,id2,hnam2,itmp2
                endif
              endif
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
            itmp1=hepcmp(id1)
            if(itmp1.gt.0)then
              id2=-id1
              call hepnam(id1,hnam1)
              itmp2=hepcmp(id2)
              if(itmp2.eq.0)then
                write(lnhout,1103) id1,hnam1,itmp1
              else
                call hepnam(id2,hnam2)
                write(lnhout,1104) id1,hnam1,itmp1,id2,hnam2,itmp2
              endif
            endif
 480      continue
 490    continue
 500  continue

      return
 1001 format(//5X,'Special Cases'
     1     /5X,'ISTD  HNAM',14X,'ICOMP',8X,'ISTD  HNAM',14X,'ICOMP')
 1002 format(//5X,'Diquarks'
     1     /5X,'ISTD  HNAM',14X,'ICOMP',8X,'ISTD  HNAM',14X,'ICOMP')
 1003 format(//5X,'Mesons'
     1     /5X,'ISTD  HNAM',14X,'ICOMP',8X,'ISTD  HNAM',14X,'ICOMP')
 1004 format(//5X,'Baryons'
     1     /5X,'ISTD  HNAM',14X,'ICOMP',8X,'ISTD  HNAM',14X,'ICOMP')
 1005 format(//5X,'Ions'
     1     /4X,'ISTD',5X,'HNAM',14X,'ICOMP',11X,'ISTD  HNAM',
     2     14X,'ICOMP')
 1101 format(1x,i8,2x,a16,1x,i7)
 1102 format(1x,i8,2x,a16,1x,i7,5x,i8,2x,a16,1x,i7)
 1103 format(1x,i10,2x,a16,1x,i7)
 1104 format(1x,i10,2x,a16,1x,i7,5x,i10,2x,a16,1x,i7)
      END
