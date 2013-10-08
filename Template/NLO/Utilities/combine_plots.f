      program combine_topdrawer
C
C Combine top drawer files and take their ratio
C By Rikkert Frederix, 2012
C
      implicit none
      integer i
      character*200 buff
      integer nif,idata(6)
      character*200 filename(0:6),titleTB(6),ylimits(6),xlimits(6),
     &     orders(6),yscale(6),title(6),history(6)
      double precision mfactors(6),x(1000,6),y(1000,6),dy(1000,6)
      common/c_all/titleTB,ylimits,xlimits,orders,yscale,title,history,
     &     mfactors,x,y,dy,nif,idata
      logical pass(8)
      logical debug
      parameter (debug=.false.)

      ! set every string to empty
      do i = 1, 6
        titleTB(i) = ''
        ylimits(i) = ''
        xlimits(i) = ''
        orders(i) = ''
        yscale(i) = ''
        title(i) = ''
        history(i) = ''
      enddo

      write (*,*) "Give the number of input files (max 6)"
      read (*,*) nif

      write (*,*) "Give their file names"
      do i=1,nif
         read(*,'(a)') filename(i)
         open (unit=10+i,file=filename(i),status='old')
      enddo

      write (*,*) "Enter rescaling factors"
      do i=1,nif
         read (*,*) mfactors(i)
      enddo
      filename(0)='combine.top'
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
     &           .and. pass(5)) then
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
            call write_data

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
      integer unit,ifile,i,iline(6),iline_last_title(6)
      data iline /6*0/
      save iline_last_title
      character*200 buff
      integer nif,idata(6)
      character*200 filename(0:6),titleTB(6),ylimits(6),xlimits(6),
     &     orders(6),yscale(6),title(6),history(6)
      double precision mfactors(6),x(1000,6),y(1000,6),dy(1000,6)
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
      double precision minx,maxx,step(6),
     &     xt(1000,6),yt(1000,6),dyt(1000,6)
      integer nif,idata(6)
      character*200 filename(0:6),titleTB(6),ylimits(6),xlimits(6),
     &     orders(6),yscale(6),title(6),history(6)
      double precision mfactors(6),x(1000,6),y(1000,6),dy(1000,6)
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




      subroutine write_data
      implicit none
      integer i,j
      integer nif,idata(6)
      character*200 filename(0:6),titleTB(6),ylimits(6),xlimits(6),
     &     orders(6),yscale(6),title(6),history(6)
      double precision mfactors(6),x(1000,6),y(1000,6),dy(1000,6)
      common/c_all/titleTB,ylimits,xlimits,orders,yscale,title,history,
     &     mfactors,x,y,dy,nif,idata
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


      write (10,'(a)') 'NEW PLOT'
      write (10,'(a)') 'set intensity 4'
      write (10,'(a)') 'SET WINDOW X 2 12'
      write (10,'(a)') 'SET WINDOW Y 4 9'
      write (10,'(a)') 'SET LABELS BOTTOM OFF'
      if (index(titleTB(1),' TOP ').ne.0) then
         write (10,'(a)') titleTB(1)(1:istrl(titleTB(1)))
      else
         write (10,'(a)')
     &        'TITLE TOP "'//title(1)(1:istrl(title(1)))//'"'
      endif
      write (10,'(a)') yscale(1)(1:istrl(yscale(1)))
      write (10,'(a)') ylimits(1)(1:istrl(ylimits(1)))
      write (10,'(a)') xlimits(1)(1:istrl(xlimits(1)))
      write (10,'(a)') orders(1)(1:istrl(orders(1)))
      write (10,'(a)') title(1)(1:istrl(title(1)))
      write (10,'(a)') history(1)(1:istrl(history(1)))

      do i=1,nif
         do j=1,idata(i)
            write (10,'(3(3X,e12.5))') x(j,i),y(j,i),dy(j,i)
         enddo
         if (i.eq.1) then
            write (10,'(a)') '  HIST SOLID'
         elseif (i.eq.2) then
            write (10,'(a)') '  HIST SOLID RED'
         elseif (i.eq.3) then
            write (10,'(a)') '  HIST SOLID BLUE'
         elseif (i.eq.4) then
            write (10,'(a)') '  HIST SOLID MAGENTA'
         elseif (i.eq.5) then
            write (10,'(a)') '  HIST SOLID GREEN'
         elseif (i.eq.6) then
            write (10,'(a)') '  HIST SOLID CYAN'
         endif
         write (10,'(a)') ''
      enddo
      write (10,'(a)') 'SET WINDOW Y 1 4'
      write (10,'(a)') 'SET LABELS BOTTOM ON'
      if (index(titleTB(1),' BOTTOM ').ne.0) then
         write (10,'(a)') titleTB(1)(1:istrl(titleTB(1)))
      else
         write (10,'(a)') 
     &        'TITLE BOTTOM "'//title(1)(1:istrl(title(1)))//'"'
      endif
      write (10,'(a)') xlimits(1)(1:istrl(xlimits(1)))
      write (10,'(a)') '  SET LIMITS Y 0.5 1.5'
      write (10,'(a)') '  SET SCALE Y LIN'
      do i=1,nif
         do j=1,idata(i)
            if (y(j,i).ne.y(j,1) .and. y(j,1).ne.0d0) then
               write (10,'(3(3X,e12.5))') x(j,i),y(j,i)/y(j,1),0d0
            else
               write (10,'(3(3X,e12.5))') x(j,i),1d0,0d0
            endif
         enddo
         if (i.eq.1) then
            write (10,'(a)') '  HIST SOLID'
         elseif (i.eq.2) then
            write (10,'(a)') '  HIST SOLID RED'
         elseif (i.eq.3) then
            write (10,'(a)') '  HIST SOLID BLUE'
         elseif (i.eq.4) then
            write (10,'(a)') '  HIST SOLID MAGENTA'
         elseif (i.eq.5) then
            write (10,'(a)') '  HIST SOLID GREEN'
         elseif (i.eq.6) then
            write (10,'(a)') '  HIST SOLID CYAN'
         endif
      enddo

      write (10,'(a)') ''
      write (10,'(a)') ''

      return
      end



      subroutine write_data2
      implicit none
      integer i,j
      integer nif,idata(6)
      character*200 filename(0:6),titleTB(6),ylimits(6),xlimits(6),
     &     orders(6),yscale(6),title(6),history(6)
      double precision mfactors(6),x(1000,6),y(1000,6),dy(1000,6)
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
     &        'TITLE TOP "'//title(1)(1:istrl(title(1)))//'"'
      endif
      write (10,'(a)') yscale(1)(1:istrl(yscale(1)))
      write (10,'(a)') ylimits(1)(1:istrl(ylimits(1)))
      write (10,'(a)') xlimits(1)(1:istrl(xlimits(1)))
      write (10,'(a)') orders(1)(1:istrl(orders(1)))
      write (10,'(a)') title(1)(1:istrl(title(1)))
      write (10,'(a)') history(1)(1:istrl(history(1)))

      do j=1,idata(1)
         do i=1,nif
            sumy(j)=sumy(j)+y(j,i)*mfactors(i)
            sumdy(j)=sqrt(sumdy(j)**2+(dy(j,i)*mfactors(i))**2)
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
