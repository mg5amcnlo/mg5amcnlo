      program scale_events
c********************************************************************
c     Takes events from events.dat and scales them according to 
c     the correct cross section for each diagram
c********************************************************************
      implicit none
c
c     Constants
c
      character*(*) symfile
      parameter (symfile='symfact.dat')
      character*(*) scaled_file
      parameter (scaled_file='scaled.dat')
c
c     local
c
      character*30 dirname
      double precision xi
      integer j,k,ip
c-----
c  Begin Code
c-----
      open(unit=16,file=scaled_file,status='unknown',err=999)
      open(unit=35,file=symfile,status='old',err=59)
      do while (.true.)
         read(35,*,err=99,end=99) xi,j
         if (j .gt. 0) then
            if ( (xi-int(xi+.01)) .lt. 1d-5) then
               k = int(xi+.01)
               if (k .lt. 10) then
                  write(dirname,'(a,i1,a)') 'G',k,'/'
               else if (k .lt. 100) then
                  write(dirname,'(a,i2,a)') 'G',k,'/'
               else if (k .lt. 1000) then
                  write(dirname,'(a,i3,a)') 'G',k,'/'
               else if (k .lt. 10000) then
                  write(dirname,'(a,i4,a)') 'G',k,'/'
               endif
            else               !Handle B.W.
               if (xi .lt. 10) then
                  write(dirname,'(a,f5.3,a,a)') 'G',xi,'/'
               else if (xi .lt. 100) then
                  write(dirname,'(a,f6.3,a,a)') 'G',xi,'/'
               else if (xi .lt. 1000) then
                  write(dirname,'(a,f7.3,a,a)') 'G',xi,'/'
               else if (xi .lt. 10000) then
                  write(dirname,'(a,3f8.3,a,a)') 'G',xi,'/'
               endif
            endif
            ip = index(dirname,'/')
            write(*,*) 'Scaling ',dirname(1:ip)
            call scale_dir(dirname,j,ip)
         endif
      enddo
 99   close(35)
      close(16)
      stop
c
c     Come here if there isn't a symfact file. Means we will work on
c     this file alone
c
 59   dirname="./"
      j = 1
      ip = 2
      write(*,*) 'Scaling ',dirname(1:ip)
      call scale_dir(dirname,j,ip)
      close(16)
      stop   
 999  write(*,*) 'Error opening file ',scaled_file
      close(16)
      end

      subroutine scale_dir(dirname,mfact, ip)
c********************************************************************
c     Takes events from events.dat and scales them according to 
c     the correct cross section. 
c********************************************************************
      implicit none
c
c     parameters
c     
      include 'nexternal.inc'
      character*(*) event_file,           xsec_file
      parameter (event_file='events.lhe', xsec_file='results.dat')
      integer maxexternal
      parameter (maxexternal=15)

c
c     Arguments
c
      integer mfact, ip
      character*(30) dirname
c
c     Local
c
      double precision xsec, sum, wgt, mxwgt
      double precision x1,x2,p(0:4,nexternal)
      integer i,j,k, kevent,m, ic(7,maxexternal),n
      double precision scale,aqed,aqcd
      integer ievent
      character*79 buff
      logical done
c-----
c  Begin Code
c-----     
      open(unit=15,file=dirname(1:ip) // xsec_file,status='old',err=998)
      read(15,*) xsec
      xsec = xsec * mfact 
      close(15)
      sum=0d0
      mxwgt=-1d0
      kevent = 0
      open(unit=15,file=dirname(1:ip)//event_file,status='old',err=999)
      done = .false.
      do while (.not. done)
         call read_event(15,p,wgt,n,ic,ievent,scale,aqcd,aqed,done)
         if (.not. done) then
            sum=sum+wgt
            mxwgt = max(wgt,mxwgt)
            kevent = kevent+1
         endif
      enddo      
 99   close(15)
      write(*,*) 'Found ',kevent,' events'
      write(*,*) 'total weight',sum
      write(*,*) 'Integrated weight',xsec
c      stop
c
c     Now write out scaled events
c
      call write_comments(16)
      open(unit=15,file=dirname(1:ip) //event_file,status='old',err=999)
      done=.false.
      m = 0
      do while (.not. done)
         m=m+1
         call read_event(15,p,wgt,n,ic,ievent,scale,aqcd,aqed,done)
c         call read_event(15,p,wgt,n,ic,done)
         if (.not. done) then
            call write_event(16,p,wgt*xsec/sum,n,ic,m,scale,aqcd,aqed)
         endif
      enddo 
 900  close(15)
      return
 55   format(i3,4e19.12)         
 998  write(*,*) 'Error opening file ',dirname(1:ip) // xsec_file
      return
 999  write(*,*) 'Error opening file ',dirname(1:ip) //event_file
      return
      end
