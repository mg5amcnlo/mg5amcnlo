      program test
c*****************************************************************
c     Program to combine results from channels that have been
c     split into multiple jobs. Multi-job channels are identified
c     by local file mjobs.dat in the channel directory.
c****************************************************************
      implicit none
c
c     Constants
c
      include 'run_config.inc'
      integer    maxsubprocesses
      parameter (maxsubprocesses=999)
c     
c     Local
c
      character*300 procname(maxsubprocesses)
      character*300 channame(maxsubprocesses)
      character*300 pathname
      integer iproc,nproc
      integer ichan,nchan
      integer jc

c-----
c  Begin Code
c-----

c
c     Get list of subprocesses
c
      call get_subprocess(procname,nproc)

      do iproc = 1, nproc   !Loop over each subprocess
         jc = index(procname(iproc),' ')
         pathname = procname(iproc)(1:jc)
c
c        get list of integration channels for this subprocess
c
         call get_channels(pathname,channame,nchan)
         do ichan = 1, nchan
            call sum_multichannel(channame(ichan))
         enddo
      enddo

      end

      subroutine get_subprocess(subname,ns)
c*****************************************************************
c     Opens file subproc.mg to determine all subprocesses
c****************************************************************
      implicit none
c
c     Constants
c
      character*(*) plist
      parameter    (plist='subproc.mg')
      integer    maxsubprocesses
      parameter (maxsubprocesses=999)
c
c     Arguments
c
      character*(*) subname(maxsubprocesses)
      integer ns
c-----
c  Begin Code
c-----
      ns = 1
      open(unit=15, file=plist,status='old',err=99)
      do while (.true.)
         read(15,*,err=999,end=999) subname(ns)
         ns=ns+1
      enddo
 99   subname(ns) = './'
      write(*,*) "Did not find ", plist
      return
 999  ns = ns-1
      write(*,*) "Found ", ns," subprocesses"
      close(15)
      end

      subroutine get_channels(pathname,channame,nchan)
c*****************************************************************
c     Opens file symfact.dat to determine all channels
c****************************************************************
      implicit none
c
c     Constants
c
      character*(*) symfile
      parameter    (symfile='symfact.dat')
      include 'maxparticles.inc'
      integer    maxsubprocesses
      parameter (maxsubprocesses=999)
c
c     Arguments
c
      character*(*) channame(maxsubprocesses)
      character*(*) pathname
      integer nchan
c
c     Local
c
      character*300 fname
      character*50 dirname
      integer jc,ip
      double precision xi
      integer j,k
      integer ncode,npos
      character*20 formstr
c-----
c  Begin Code
c-----
      jc = index(pathname," ")
c     ncode is number of digits needed for the bw coding
      ncode=int(dlog10(3d0)*(max_particles-3))+1
      fname = pathname(1:jc-1) // "/" // symfile
      nchan = 0
      open(unit=35, file=fname,status='old',err=99)
      do while (.true.)
         read(35,*,err=99,end=99) xi,j
         if (j .gt. 0) then
            k = int(xi*(1+10**-ncode))
            npos=int(dlog10(dble(k)))+1
            if ( (xi-k) .eq. 0) then
c              Write with correct number of digits
               write(formstr,'(a,i1,a)') '(a,i',npos,',a)'
               write(dirname, formstr) 'G',k,'/'
            else               !Handle B.W.
c              Write with correct number of digits
               write(formstr,'(a,i1,a,i1,a)') '(a,f',npos+ncode+1,
     $                 '.',ncode,',a)'
               write(dirname,formstr)  'G',xi,'/'
            endif     
            ip = index(dirname,'/')
            nchan=nchan+1            
            channame(nchan) = fname(1:jc-1)// "/" //dirname(1:ip)
         endif
 98   enddo
 99   close(35)
      write(*,*) "Found ", nchan, " channels."
      end

      subroutine sum_multichannel(channame)
c*****************************************************************
c     Looks in channel to see if there are multiple runs that
c     need to be combined. If so combines them into single run
c****************************************************************
      implicit none
c
c     Constants
c
      character*(*) mfile
      parameter    (mfile='multijob.dat')
      integer       maxjobs
      parameter    (maxjobs=26)
c
c     Arguments
c
      character*(*) channame
c
c     Local
c
      character*300 fname
      character*300 pathname
      character*26 alphabet
      integer jc,i
      integer njobs
      double precision xsec(maxjobs),xerr(maxjobs)
      double precision xsec_it(9,maxjobs),xerr_it(9,maxjobs)
      double precision eff(9,maxjobs),wmax(9,maxjobs)
      integer nevents(maxjobs),ntry(maxjobs)
      double precision xsec_tot, xerr_tot
      integer tot_nevents, tot_ntry
      integer lunw,nw
      double precision wgt
c-----
c  Begin Code
c-----
      alphabet="abcdefghijklmnopqrstuvwxyz"
      jc = index(channame," ")
      fname = channame(1:jc-1) // mfile
      njobs = 0
      open(unit=35, file=fname,status='old',err=99)
      read(35,*,err=99,end=99) njobs
 99   close(35)
      if(njobs .gt. 0) then
         write(*,*) "Found ", njobs, " jobs in ",fname
c
c     Read in results from each of the jobs
c
      do i = 1, njobs
         pathname = channame(1:jc-2) // alphabet(i:i)
         call get_results(pathname,xsec(i),xerr(i),ntry(i),nevents(i),
     $        xsec_it(1,i),xerr_it(1,i),eff(1,i),wmax(1,i))
      enddo
c
c     Process results from each job to get final statistics
c
      xsec_tot = 0d0
      xerr_tot = 0d0
      tot_ntry = 0
      tot_nevents = 0
      do i = 1, njobs
         xsec_tot = xsec_tot+xsec(i)
         xerr_tot = xerr_tot+xerr(i)**2
         tot_ntry = tot_ntry+ntry(i)
         tot_nevents = tot_nevents+nevents(i)
      enddo
      xsec_tot = xsec_tot / njobs
      xerr_tot = sqrt(xerr_tot)/njobs

      pathname = channame(1:jc-2)
      i=1
      call put_results(pathname,xsec_tot,xerr_tot,tot_ntry,tot_nevents,
     $        xsec_it(1,i),xerr_it(1,i),eff(1,i),wmax(1,i))
      call write_logfile(pathname,xsec,xerr,ntry,nevents,
     $        xsec_it,xerr_it,eff,wmax,njobs)
c
c     Now read in all of the events and write them
c     back out with the appropriate scaled weight
c      
      lunw=15
      jc = index(pathname," ")
      open(unit=lunw,file=pathname(1:jc-1)//"/"//"events.lhe",
     $     status="unknown",err=999)
      write(*,*) "Placing combined events in ",
     $     pathname(1:jc-1)//"/"//"events.lhe"
      wgt = xsec_tot / tot_nevents
      tot_nevents=0
      do i = 1, njobs
         jc = index(channame," ")
         pathname = channame(1:jc-2) // alphabet(i:i)
         call copy_events(pathname,lunw,wgt,nw)
         tot_nevents=tot_nevents+nw
         write(*,*) "Added ",nw," events from run ", alphabet(i:i)
         if (nw .ne. nevents(i)) then
            write(*,*) "Error writing events ",i,nw,nevents(i),
     $           pathname
         else
         endif
      enddo
      write(*,*) "Combined ",tot_nevents," to ",channame(1:jc-2)
      endif
      close(lunw)
      return
 999  close(lunw)
      write(*,*) "Error, unable to open events.lhe file for output"
     $     ,pathname
      end

      subroutine copy_events(pathname,lunw,wgt,nw)
c*********************************************************************
c     Copy events from separate runs into one file w/ appropriate wgts
c*********************************************************************
      implicit none
c     
c     Constants
c
      character*(*) eventfname
      parameter    (eventfname="events.lhe")
      include 'maxparticles.inc'
      integer    maxexternal     !Max number external momenta
      parameter (maxexternal=2*max_particles-3)
c     
c     Arguments
c
      character*(*) pathname
      double precision wgt
      integer lunw,nw
c     
c     Local
c
      character*300 fname
      integer jc
      double precision P(0:4,maxexternal),xwgt
      integer n,ic(7,maxexternal),ievent
      double precision scale,aqcd,aqed
      character*140 buff
      logical done
c-----
c  Begin Code
c-----
      jc = index(pathname," ")
      fname = pathname(1:jc-1) // "/" // eventfname
      nw = 0
      write(*,*) "Copy from ",fname(1:jc+10)
      open(unit=35, file=fname,status='old',err=999)
      done = .false.
      do while (.not. done .and. nw < 999999)
         done = .true.
         call read_event(35,P,xwgt,n,ic,ievent,scale,
     $        aqcd,aqed,buff,done)
         if (.not. done) then
            call write_event(lunw,P,wgt,n,ic,ievent,scale,
     $           aqcd,aqed,buff)
            nw=nw+1
         endif
      enddo
 99   close(35)
      return
 999  close(35)
      write(*,*) "Error unable to open ",fname
      end

      subroutine get_results(pathname,xsec,xerr,ntry,nevents,
     $     xsec_it,xerr_it,eff,wmax)
c*********************************************************************
c     Read run results from file results.dat
c*********************************************************************
      implicit none
c     
c     Constants
c
      character*(*) resultfname
      parameter    (resultfname="results.dat")
c     
c     Arguments
c
      character*(*) pathname
      double precision xsec,xerr
      double precision xsec_it(9),xerr_it(9)
      double precision eff(9),wmax(9)
      integer nevents,ntry
c     
c     Local
c
      character*300 fname
      integer jc,i,j
      double precision x1,x2,x3,x4
c-----
c  Begin Code
c-----
      jc = index(pathname," ")
      fname = pathname(1:jc-1) // "/" // resultfname
      nevents = 0
      i=1
      open(unit=35, file=fname,status='old',err=99)
      read(35,*,err=99,end=99) xsec,xerr, x1, ntry,x3,x4,nevents
      do while (.true.)
         read(35,*,end=99,err=99) x1,xsec_it(i),xerr_it(i),
     $        eff(i),wmax(i)
         i=i+1
      enddo
 99   close(35)
      i=i-1
      xsec_it(i+1) = -1d0
      if (nevents .gt. 0) then
         write(*,*) "Found ",nevents, " events in ", pathname(1:jc-1)
         i=1
         do while (xsec_it(i) .gt. 0d0)
            write(*,*) i,xsec_it(i),xerr_it(i),eff(i),wmax(i)
            i=i+1
         enddo
      endif      
      end

      subroutine write_logfile(pathname,xsec,xerr,ntry,nevents,
     $     xsec_it,xerr_it,eff,wmax,njobs)
c*********************************************************************
c     Read run results from file results.dat
c*********************************************************************
      implicit none
c     
c     Constants
c
      integer       maxjobs
      parameter    (maxjobs=26)
      character*(*) logfname
      parameter    (logfname="log.txt")
c     
c     Arguments
c
      character*(*) pathname
      double precision xsec(maxjobs),xerr(maxjobs)
      double precision xsec_it(9,maxjobs),xerr_it(9,maxjobs)
      double precision eff(9,maxjobs),wmax(9,maxjobs)
      integer nevents(maxjobs),ntry(maxjobs)
      double precision xsec_tot, xerr_tot
      integer tot_nevents, tot_ntry
      integer njobs
c     
c     Local
c
      character*300 fname
      character*26 alphabet
      integer jc,i,j
c-----
c  Begin Code
c-----
      alphabet="abcdefghijklmnopqrstuvwxyz"
      jc = index(pathname," ")
      fname = pathname(1:jc-1) // "/" // logfname
      open(unit=35,file=fname,status="old",access="append",err=999)

      write(35,*) '--------------------- Multi run with ',njobs,' jobs. ',
     $            '---------------------'
      do i=1,njobs
         write(35,*) "job ",alphabet(i:i),":",xsec(i),nevents(i)
      enddo
      close(35)
      return
 999  write(*,*) "Error opening ",fname(1:jc+10), "for append"
      close(35)
      end

      subroutine put_results(pathname,xsec,xerr,ntry,nevents,
     $     xsec_it,xerr_it,eff,wmax)
c*********************************************************************
c     Read run results from file results.dat
c*********************************************************************
      implicit none
c     
c     Constants
c
      character*(*) resultfname
      parameter    (resultfname="results.dat")
c     
c     Arguments
c
      character*(*) pathname
      double precision xsec,xerr
      double precision xsec_it(9),xerr_it(9)
      double precision eff(9),wmax(9)
      integer nevents,ntry
c     
c     Local
c
      character*300 fname
      integer jc,i,j
      double precision x1,x2
      integer i3,i4
c-----
c  Begin Code
c-----
      jc = index(pathname," ")
      fname = pathname(1:jc-1) // "/" // resultfname
      x1 = 0d0
      x2 = 0d0
      i3 = 0
      i4 = 0
      open(unit=35, file=fname,status='unknown',err=99)
      write(35,'(3e12.5,2i9,i5,i9,e10.3)') xsec,xerr, x1, ntry,i3,i4
     $     ,nevents,(1.00*nevents)/xsec
      i=1
      do while (xsec_it(i) .gt. 0 .and. i .lt. 10)
         write(35,'(i4,4e15.5)') i,xsec_it(i),xerr_it(i),
     $        eff(i),wmax(i)
         i=i+1
      enddo
 99   close(35)
      end
