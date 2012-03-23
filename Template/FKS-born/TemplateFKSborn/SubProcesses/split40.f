      PROGRAM SPLIT
      IMPLICIT NONE
      character*80 buff
      integer number,i,j

      open (unit=1, file="dir", status="old")

      number=0
      do 
         read(1,*,end=99,err=99) buff
         number=number+1
      enddo
 99   continue
      rewind(1)
      i=1
      do while (number.gt.40)
         open(unit=2,file='pdir'//char(48+i),status='unknown')
         write (2,'(i3)') 40
         do j=1,40
            read(1,'(a)',end=98,err=98) buff
            write (2,'(a)') buff
         enddo
c         do j=1,9
c            write (2,'(i1)') 1
c         enddo
         close(2)
         i=i+1
         number=number-40
      enddo
      if (number.gt.0) then
         open(unit=2,file='pdir'//char(48+i),status='unknown')
         if (number.gt.99) then
            write (2,'(i3)') number
         elseif(number.gt.9)then
            write (2,'(i2)') number
         else
            write (2,'(i1)') number
         endif
         do j=1,number
            read(1,'(a)',end=98,err=98) buff
            write (2,'(a)') buff
         enddo
c         do j=1,number
c            write (2,'(i1)') 1
c         enddo
         close(2)
      endif


 98   return
      end
