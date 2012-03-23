      program sum
c********************************************************************************
c     Program to combine results from all of the different sub amplitudes 
c     and given total
c     cross section and error.
c*****************************************************************************
      implicit none
c
c     Constants
c
      character*(*) rfile
      parameter (rfile='result')
      character*(*) symfile
      parameter (symfile='symfact.dat')
      integer   max_amps
      parameter (max_amps=999)
c
c     local
c
      double precision xsec(max_amps), xerr(max_amps)
      character*80 fname
      integer i,j
      double precision xtot,errtot
c-----
c  Begin Code
c-----
      xtot=0d0
      errtot=0d0
      open(unit=15,file=symfile,status='old',err=999)
      do while (.true.)
         read(15,*,err=99) i,j
         if (j .gt. 0) then
            if (i .lt. 10) then
               write(fname,'(a,i1,a,a)') 'G',i,'/',rfile
c               write(*,*) fname
            else if (i .lt. 100) then
               write(fname,'(a,i2,a,a)') 'G',i,'/',rfile
c               write(*,*) fname
            else if (i .lt. 1000) then
               write(fname,'(a,i3,a,a)') 'G',i,'/',rfile
c               write(*,*) fname
            endif
            open(unit=25,file=fname,status='old',err=95)
            read(25,*) xsec(i), xerr(i)
            xtot = xtot+xsec(i)*j
            errtot=errtot+j*xerr(i)**2
            write(*,'(2i4,2e12.4)') i,j, xsec(i),xerr(i)
 95         close(25)
         endif
      enddo
 99   write(*,*) 'done',xtot,sqrt(errtot)
      close(15)
      stop
 999  write(*,*) 'error'
      end
