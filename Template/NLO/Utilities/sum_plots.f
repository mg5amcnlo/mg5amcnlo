      program sum_topdrawer
C
C Sum top drawer files
C
      implicit none
      integer i
      character*200 buff
      integer nmaxfiles
      parameter (nmaxfiles=1000)
      integer nif,idata(nmaxfiles)
      character*200 filename(0:nmaxfiles),titleTB(nmaxfiles),ylimits(nmaxfiles),xlimits(nmaxfiles),
     &     orders(nmaxfiles),yscale(nmaxfiles),title(nmaxfiles),history(nmaxfiles)
      double precision mfactors(nmaxfiles),x(1000,nmaxfiles),y(1000,nmaxfiles),dy(1000,nmaxfiles)
      common/c_all/titleTB,ylimits,xlimits,orders,yscale,title,history,
     &     mfactors,x,y,dy,nif,idata
      logical pass(8)
      logical debug
      parameter (debug=.false.)

      ! set every string to empty
      do i = 1, nmaxfiles
        titleTB(i) = ''
        ylimits(i) = ''
        xlimits(i) = ''
        orders(i) = ''
        yscale(i) = ''
        title(i) = ''
        history(i) = ''
      enddo

      write (*,*) "Give the number of input files (max ",nmaxfiles," )"
      read (*,*) nif
      if(nif.gt.nmaxfiles)then
         write(*,*)'Too many files!!', nif
         stop
      endif

      write (*,*) "Give their file names"
      do i=1,nif
         read(*,'(a)') filename(i)
         open (unit=10+i,file=filename(i),status='old')
      enddo

      write (*,*) "Enter rescaling factors"
      do i=1,nif
         read (*,*) mfactors(i)
      enddo
      filename(0)='sum.top'
      open (unit=10,file=filename(0),status='unknown')

      do i=1,nif
         idata(i)=0
      enddo

c---  Nothing found yet
      do i=1,8
         pass(i)=.false.
      enddo

c---  Loop over all the lines of the first file
      do 
         read(11,'(a)',end=99) buff
         if ( index(buff,'TITLE').ne.0 .and.(
     &        index(buff,'BOTTOM').ne.0 .or.
     &        index(buff,'TOP').ne.0 )) then
            if (debug) write (*,*) 'title found'
c---  Title found
            titleTB(1)=buff
            pass(1)=.true.
         elseif (index(buff,'SET SCALE Y').ne.0
     &           .and. pass(1)) then
            if (debug) write (*,*) 'y-scale found'
c---  y-scale found
            yscale(1)=buff
            pass(2)=.true.
         elseif (index(buff,'SET LIMITS X').ne.0
     &           .and. pass(2)) then
            if (debug) write (*,*) 'x-limits found'
c---  x-limits found
            xlimits(1)=buff
            pass(3)=.true.
            pass(4)=.true.
         elseif (index(buff,'SET LIMITS Y').ne.0
     &           .and. pass(3)) then
            if (debug) write (*,*) 'y-limits found'
c---  y-limits found
            ylimits(1)=buff
            pass(4)=.true.
         elseif (index(buff,'SET ORDER X Y DY').ne.0
     &           .and. pass(4)) then
            if (debug) write (*,*) 'orders found'
c---  orders found
            orders(1)=buff
            pass(5)=.true.
         elseif(buff(1:4).eq.' (  '
     &          .and. pass(5)) then
            if (debug) write (*,*) 'title-tag found'
c--- proper title-tag found
            title(1)=buff
            pass(6)=.true.
         elseif(pass(6)) then
            if (debug) write (*,*) 'historical line found'
c--- line that is there for historical reasons
            history(1)=buff
            pass(6)=.false.
            pass(7)=.true.
         elseif(pass(7) .and. index(buff,'HIST').eq.0) then
            if (debug) write (*,*) 'data is here'
c--- Here is the data
            idata(1)=idata(1)+1
            read(buff,*) x(idata(1),1),y(idata(1),1),dy(idata(1),1)
            pass(8)=.true.
         elseif(index(buff,'HIST').ne.0) then
            if (debug) write (*,*) 'end of the data'
c--- all data found, read the other files
            do i=2,nif
               call get_other_data(i)
            enddo
c--- order all the data
            call order_data

c--- dump the data to a new file
            call write_data2

            do i=1,8
               pass(i)=.false.
            enddo
            do i=1,nif
               idata(i)=0
            enddo
c            exit
         endif
      enddo

 99   continue
      write(*,*)" "
      write(*,*) "All done"
      do i=0,nif
         close (i+10)
      enddo
      return
      end

      subroutine get_other_data(ifile)
      implicit none
      integer nmaxfiles
      parameter (nmaxfiles=1000)
      integer unit,ifile,i,iline(nmaxfiles),iline_last_title(nmaxfiles)
      data iline /nmaxfiles*0/
      save iline_last_title
      character*200 buff
      integer nif,idata(nmaxfiles)
      character*200 filename(0:nmaxfiles),titleTB(nmaxfiles),ylimits(nmaxfiles),xlimits(nmaxfiles),
     &     orders(nmaxfiles),yscale(nmaxfiles),title(nmaxfiles),history(nmaxfiles)
      double precision mfactors(nmaxfiles),x(1000,nmaxfiles),y(1000,nmaxfiles),dy(1000,nmaxfiles)
      common/c_all/titleTB,ylimits,xlimits,orders,yscale,title,history,
     &     mfactors,x,y,dy,nif,idata
      logical pass(8)
      logical debug
      parameter (debug=.false.)

      unit=ifile+10
      idata(ifile)=0

c---  Nothing found yet
      do i=1,8
         pass(i)=.false.
      enddo
c---  Loop over all the lines of the first file
      do 
         read(unit,'(a)',end=99) buff
         iline(ifile)=iline(ifile)+1
         if ( index(buff,'TITLE').ne.0 .and.(
     &        index(buff,'BOTTOM').ne.0 .or.
     &        index(buff,'TOP').ne.0 )) then
            if (debug) write (*,*) 'title found'
c---  Title found
            titleTB(ifile)=buff
            pass(1)=.true.
         elseif (index(buff,'SET SCALE Y').ne.0
     &           .and. pass(1)) then
            if (debug) write (*,*) 'y-scale found'
c---  y-scale found
            yscale(ifile)=buff
            pass(2)=.true.
         elseif (index(buff,'SET LIMITS X').ne.0
     &           .and. pass(2)) then
            if (debug) write (*,*) 'x-limits found'
c---  x-limits found
            xlimits(ifile)=buff
            pass(3)=.true.
            pass(4)=.true.
         elseif (index(buff,'SET LIMITS Y').ne.0
     &           .and. pass(3)) then
            if (debug) write (*,*) 'y-limits found'
c---  y-limits found
            ylimits(ifile)=buff
            pass(4)=.true.
         elseif (index(buff,'SET ORDER X Y DY').ne.0
     &           .and. pass(4)) then
            if (debug) write (*,*) 'orders found'
c---  orders found
            orders(ifile)=buff
            pass(5)=.true.
c
c CHECK IF TITLE IS EQUAL TO SAVED TITLE
c
         elseif(buff.eq.title(1)
     &           .and. pass(5)) then
            if (debug) write (*,*) 'title-tag found'
c--- proper title-tag found
            iline_last_title(ifile)=iline(ifile)
            title(ifile)=buff
            pass(6)=.true.
         elseif(pass(6)) then
            if (debug) write (*,*) 'historical line found'
c--- line that is there for historical reasons
            history(ifile)=buff
            pass(6)=.false.
            pass(7)=.true.
         elseif(pass(7) .and. index(buff,'HIST').eq.0) then
            if (debug) write (*,*) 'data is here'
c--- Here is the data
            idata(ifile)=idata(ifile)+1
            read (buff,*) x(idata(ifile),ifile),
     &           y(idata(ifile),ifile),dy(idata(ifile),ifile)
            pass(8)=.true.
         elseif(index(buff,'HIST').ne.0.and.pass(8)) then
            if (debug) write (*,*) 'end of the data'
c all the data found, exit the loop normally
            exit
         endif
      enddo
      return
c End of file: data not found
 99   continue
      title(ifile)='ERROR ERROR ERROR DATA NOT FOUND'
c Rewind the file back to the last title that we found
      rewind(unit)
      do i=1,iline_last_title(ifile)
         read(unit,'(a)',end=99) buff
      enddo
      return
      end



      subroutine order_data
      implicit none
      integer i,j,k
      integer nmaxfiles
      parameter (nmaxfiles=1000)
      double precision minx,maxx,step(nmaxfiles),
     &     xt(1000,nmaxfiles),yt(1000,nmaxfiles),dyt(1000,nmaxfiles)
      integer nif,idata(nmaxfiles)
      character*200 filename(0:nmaxfiles),titleTB(nmaxfiles),ylimits(nmaxfiles),xlimits(nmaxfiles),
     &     orders(nmaxfiles),yscale(nmaxfiles),title(nmaxfiles),history(nmaxfiles)
      double precision mfactors(nmaxfiles),x(1000,nmaxfiles),y(1000,nmaxfiles),dy(1000,nmaxfiles)
      common/c_all/titleTB,ylimits,xlimits,orders,yscale,title,history,
     &     mfactors,x,y,dy,nif,idata
      double precision tiny
      parameter (tiny=1d-2) ! 1% is small enough

      do i=1,nif
         step(i)=9d99
      enddo
      
      minx=9d99
      maxx=-9d99
      do i=1,nif
         minx=min(x(1,i),minx)
         maxx=max(x(idata(i),i),maxx)
         do j=1,idata(i)-1
            step(i)=min(abs(x(j,i)-x(j+1,i)),step(i))
         enddo
         if (step(i).eq.0d0) then
            write (*,*) 'Step size too small',i
            stop
         endif
      enddo

      do i=2,nif
         if (abs(step(1)-step(i))/step(1).gt.tiny) then
            write (*,*) 'Step size not consistent'
            write (*,*) step(1),i,step(i)
            stop
         endif
      enddo
         
      do i=1,nif
         idata(i)=nint((maxx-minx)/step(i))+1
         if (idata(i).ne.idata(1)) then
            write (*,*) 'Data size not consistent'
            stop
         endif
      enddo

      do i=1,nif
         k=1
         do j=1,idata(i)
            if (j.eq.1) then
               xt(j,i)=minx
            else
               xt(j,i)=xt(j-1,i)+step(i)
            endif
            if (abs(xt(j,i)-x(k,i))/step(i).lt.tiny) then
               xt(j,i)=x(k,i)
               yt(j,i)=y(k,i)*mfactors(i)
               dyt(j,i)=dy(k,i)*mfactors(i)
               k=k+1
            else
               yt(j,i)=0d0
               dyt(j,i)=0d0
            endif
         enddo
      enddo

      do i=1,nif
         do j=1,idata(i)
            x(j,i)=xt(j,i)
            y(j,i)=yt(j,i)
            dy(j,i)=dyt(j,i)
         enddo
      enddo

      return
      end




      subroutine write_data2
      implicit none
      integer i,j
      integer nmaxfiles
      parameter (nmaxfiles=1000)
      integer nif,idata(nmaxfiles)
      character*200 filename(0:nmaxfiles),titleTB(nmaxfiles),ylimits(nmaxfiles),xlimits(nmaxfiles),
     &     orders(nmaxfiles),yscale(nmaxfiles),title(nmaxfiles),history(nmaxfiles)
      double precision mfactors(nmaxfiles),x(1000,nmaxfiles),y(1000,nmaxfiles),dy(1000,nmaxfiles)
      common/c_all/titleTB,ylimits,xlimits,orders,yscale,title,history,
     &     mfactors,x,y,dy,nif,idata
      double precision sumy(1000),sumdy(1000)
      logical firsttime
      data firsttime/.true./
      integer istrl
      external istrl

      if (firsttime) then
         write (10,'(a)') 'SET DEVICE POSTSCRIPT ORIENTATION=3'
         write (10,'(a)') 'set font duplex'
         write (10,'(a)') ''
         firsttime=.false.
      endif
      do i=1,1000
         sumy(i)=0d0
         sumdy(i)=0d0
      enddo

      write (10,'(a)') 'NEW PLOT'
      write (10,'(a)') 'set intensity 4'
      write (10,'(a)') 'SET WINDOW X 2 12'
      write (10,'(a)') 'SET WINDOW Y 4 9'
      write (10,'(a)') 'SET LABELS BOTTOM OFF'
      if (index(titleTB(1),' TOP ').ne.0) then
         write (10,'(a)') titleTB(1)(1:istrl(titleTB(1)))
      else
         write (10,'(a)')
     &        'TITLE BOTTOM "'//title(1)(1:istrl(title(1)))//'"'
      endif
      write (10,'(a)') yscale(1)(1:istrl(yscale(1)))
      write (10,'(a)') ylimits(1)(1:istrl(ylimits(1)))
      write (10,'(a)') xlimits(1)(1:istrl(xlimits(1)))
      write (10,'(a)') orders(1)(1:istrl(orders(1)))
      write (10,'(a)') title(1)(1:istrl(title(1)))
      write (10,'(a)') history(1)(1:istrl(history(1)))

      do j=1,idata(1)
         do i=1,nif
            sumy(j)=sumy(j)+y(j,i)
            sumdy(j)=sumdy(j)+dy(j,i)
         enddo
         write (10,'(3(3X,e12.5))') x(j,1),sumy(j),sumdy(j)
      enddo
      write (10,'(a)') '  HIST SOLID'
      write (10,'(a)') ''

      return
      end



      function istrl(string)
c returns the position of the last non-blank character in string
      character * (*) string
      i = len(string)
      dowhile(i.gt.0.and.string(i:i).eq.' ')
         i=i-1
      enddo
      istrl = i
      end
