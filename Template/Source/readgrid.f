C...READGRID reads the IS Sudakov grid for all flavors from a file
      subroutine readgrid(lun)
      implicit none

c...global variables
      include 'sudgrid.inc'
      include 'pdf.inc'
      include 'maxparticles.inc'
      include 'run.inc'

c...arguments
      integer lun

c...local variables
      integer i,j,ipt2,ix1,ix2,kfl,ipoints
      data kfl,ipoints/-1,1/
      logical opened

c   integer nbins(3)
c   data nbins/npt2,nx1,nx2/
      character*100 buf
      character*7 pdgrid
      double precision ebeam1,ebeam2

c   Check that the file lun is opened
      inquire(unit=lun,opened=opened)
      if(.not.opened)then
        write(*,*) 'readgrid: Error, unit ',lun,' not opened'
        stop
      endif
      
      ebeam1=0
      ebeam2=0
      
c...Check that the grid is correct
      read(lun,'(a)',ERR=999,END=999) buf
      do while(buf(1:1).eq.'#'.or.buf.eq.'')
        if(index(buf,'pdlabel').ne.0)then
          call getfirst(pdgrid,buf(2:))
          if(pdgrid .ne. pdlabel)then
            write(*,*)'Error: ',
     $         'Different pdf labels in Sudakov grid and run_card.'
            write(*,*)'Please regenerate grid file issudgrid.dat ',
     $         'or use pdlabel ',pdgrid
            stop
          endif
        endif
        if(index(buf,'ebeam1').ne.0)then
          read(buf(2:index(buf,'=')-1),*)ebeam1
        endif
        if(index(buf,'ebeam2').ne.0)then
          read(buf(2:index(buf,'=')-1),*)ebeam2
        endif
        if(ebeam1.ne.0.and.ebeam2.ne.0)then
          if(abs(ebeam1-ebeam(1))/ebeam(1).gt.1d-3.or.
     $       abs(ebeam2-ebeam(2))/ebeam(2).gt.1d-3)then
            write(*,*)'Fatal error: ',
     $         'Different beam energies in Sudakov grid and run_card.'
            write(*,*)'Please regenerate grid file issudgrid.dat ',
     $         'or use beam energies'
            write(*,*) ebeam1,ebeam2
            stop
          endif
        endif
        read(lun,'(a)',ERR=999,END=999) buf
      enddo
      rewind(lun)

c...read grid points
      do i=-2,5
      read(lun,'(a)',ERR=999,END=999) buf
      do while(buf(1:1).eq.'#'.or.buf.eq.'')
        if(index(buf,'kfl').ne.0)then
          read(buf(2:index(buf,'=')),*) kfl
          if(kfl.eq.21) kfl=0
          if(i.ne.kfl)
     $       write(*,'(''#'',a,i3)')
     $       'Warning! Expecting flavor ',i,' but read ',kfl
          if(kfl.lt.-2.or.kfl.gt.5)then
            write(*,*) 'Error! Only partons between -2 and 5 allowed'
            write(*,*) ' (gluon is 0 or 21)'
            stop
          endif
          if(iabs(kfl).eq.5) then 
            ipoints=2
          else
            ipoints=1
          endif
        endif
        read(lun,'(a)',ERR=999,END=999) buf
      enddo
      do ix2=1,nx2
        do ix1=1,nx1
          do ipt2=1,npt2
            read(buf,*,ERR=900,END=900)
     $         points(ix2,ipoints),points(nx2+ix1,ipoints),
     $         points(nx2+nx1+ipt2,ipoints),sudgrid(ix2,ix1,ipt2,kfl)
            points(ix2,ipoints)=log(points(ix2,ipoints))
            points(nx2+nx1+ipt2,ipoints)=
     $         2*log(points(nx2+nx1+ipt2,ipoints))
            if(ix2.lt.nx2.or.ix1.lt.nx1.or.ipt2.lt.npt2)
     $         read(lun,'(a)',ERR=900,END=900) buf
          enddo
        enddo
      enddo
      enddo

      write(*,'(''#'',a)') 'Done reading IS Sudakov grid'
      return

 900  write(*,*) 'Error reading IS Sudakov grid!'
      write(*,*) 'kfl=',kfl,' ix2=',ix2,' ix1=',ix1,'ipt2=',ipt2
      stop

 999  write(*,'(''#'',a,a,i2,a)') 'Warning: Failed to read IS ',
     $   'Sudakov grid for flavor ',i,' and up'
      return

      end


      subroutine getfirst(first,string)

      implicit none
      character*(*) string
      character*20 first
      character*20 temp

      temp=string
      do while(temp(1:1) .eq. ' '.or.temp(1:1).eq.'''') 
	temp=temp(2:len(temp))
      end do
      first=temp(1:index(temp,' ')-1)
      if(index(first,'''').gt.0) first=first(1:index(first,'''')-1)

      end
