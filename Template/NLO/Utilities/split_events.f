      program split_events
c Split events files into event files with lower number of events
c gfortran -I../SubProcesses/P0_<anydir> -o split_events split_events.f
c handling_lhe_events.f fill_MC_mshell.f
      implicit none
      integer maxevt,ifile,ofile,i,j,npart,mgfile,ifile2,ione
      parameter(ione=1)
      integer IDBMUP(2),PDFGUP(2),PDFSUP(2),IDWTUP,NPRUP
      double precision EBMUP(2),XSECUP,XERRUP,XMAXUP,LPRUP
      INTEGER MAXNUP
      PARAMETER (MAXNUP=500)
      INTEGER NUP,IDPRUP,IDUP(MAXNUP),ISTUP(MAXNUP),
     # MOTHUP(2,MAXNUP),ICOLUP(2,MAXNUP)
      DOUBLE PRECISION XWGTUP,SCALUP,AQEDUP,AQCDUP,
     # PUP(5,MAXNUP),VTIMUP(MAXNUP),SPINUP(MAXNUP)
      character*80 event_file,fname1,executable,inputfile
      character*140 buff
      character*10 MonteCarlo
      character*3 str
      logical back
      integer evts,leftover,loc,loc1,loc2,isc,ipdf,jmax
      integer numscales,numPDFpairs,numPDFs
      common/cwgxsec1/numscales,numPDFpairs,numPDFs
c
      back=.true.
      write (*,*) 'Give the name of the original event file'
      read (*,*) event_file
      ifile=34
      open(unit=ifile,file=event_file,status='old')
      call read_lhef_header_full(ifile,maxevt,isc,ipdf,MonteCarlo)
      numPDFpairs=ipdf/2
      numscales=int(sqrt(dble(isc)))
      call read_lhef_init(ifile,
     &     IDBMUP,EBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,
     &     XSECUP,XERRUP,XMAXUP,LPRUP)
      close(34)
      write (*,*) 'File contains ',maxevt,' events'
      write (*,*) 'Give the number of splitted files you want (< 999)'
      read (*,*) npart
      if (npart.gt.999) then
         write (*,*) 'Too many event files (999 is max)', npart
         stop
      endif
      evts=int(dble(maxevt)/dble(npart))
      leftover=maxevt-evts*npart
      write (*,*) npart,' files will have',evts,' events each'
      write (*,*) ione,' file will have',leftover,' events'
      write (*,*) ' '

c$$$      mgfile=36
c$$$      open (unit=mgfile,file='mcatnlo.cmd',status='unknown')
c$$$      call open_cmd(mgfile)
c$$$      write (*,*) ''
c$$$      write (*,*) 'To write condor cmd file we need some extra info'
c$$$      write (*,*) 'Give the name for the MCatNLO executable'
c$$$      read (*,*) executable
c$$$      write (*,*) 'Give the name for the MCatNLO input file'
c$$$      read (*,*) inputfile
      executable=' '
      inputfile=' '

      loc=index(event_file,'.',back)
      loc1=index(executable,' ')
      loc2=index(inputfile,' ')
         
      ifile=34
      ifile2=44
      open(unit=ifile,file=event_file,status='old')
      open(unit=ifile2,file='headfile',status='unknown')
      call copy_header(ifile,ifile2,evts)
      close(ifile2)
      call read_lhef_init(ifile,
     &     IDBMUP,EBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,
     &     XSECUP,XERRUP,XMAXUP,LPRUP)

      do i=1,npart+1
         jmax=evts
         if(leftover.eq.0.and.i.eq.npart+1)cycle
         if(leftover.ne.0.and.i.eq.npart+1)jmax=leftover
         str='000'
         if (i.le.9) write (str(3:3),'(i1)') i
         if (i.gt.9.and.i.le.99) write (str(2:3),'(i2)') i
         if (i.gt.99.and.i.le.999) write (str(1:3),'(i3)') i
         fname1=event_file(1:loc-1)//'_'//str//'.lhe'
         write (*,*) 'writing event file ',fname1(1:40)
         ofile=35
         ifile2=44
         open(unit=ofile,file=fname1,status='unknown')
         open(unit=ifile2,file='headfile',status='old')
         call copy_header(ifile2,ofile,jmax)
         call write_lhef_init(ofile,
     &        IDBMUP,EBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,
     &        XSECUP,XERRUP,XMAXUP,LPRUP)
         do j=1,jmax
            call read_lhef_event(ifile,
     &           NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,
     &           IDUP,ISTUP,MOTHUP,ICOLUP,PUP,VTIMUP,SPINUP,buff)

            call write_lhef_event(ofile,
     &           NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,
     &           IDUP,ISTUP,MOTHUP,ICOLUP,PUP,VTIMUP,SPINUP,buff)
         enddo
         write(ofile,*)'</LesHouchesEvents>'
         close(ifile2)
         close(ofile)
c$$$         write (mgfile,'(a)') 'Arguments = '//executable(1:loc1)//
c$$$     &        inputfile(1:loc2)//event_file(1:loc)//str
c$$$         write (mgfile,'(a)') 'queue'
      enddo 
      close(ifile)
c$$$      close(mgfile)
      end


      subroutine open_cmd(mgfile)
      implicit none
      integer mgfile
      write (mgfile,'(a)') 'universe = vanilla'
      write (mgfile,'(a)') 'executable = ajob'
      write (mgfile,'(a)') 'output = /dev/null'
      write (mgfile,'(a)') 'error = /dev/null'
      write (mgfile,'(a)') 'requirements = (MADGRAPH == True)'
      write (mgfile,'(a)') 'log = /dev/null'
      return
      end

