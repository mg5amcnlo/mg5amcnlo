      program split_events
c Split events files into event files with lower number of events
c gfortran -o split_events split_events.f handling_lhe_events.f
      implicit none
      integer maxevt,ifile,ofile,i,j,npart,mgfile
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
      character*2 str
      integer evts,leftover,loc,loc1,loc2
c
      write (*,*) 'give the name of the original event file'
      read (*,*) event_file
      ifile=34
      open(unit=ifile,file=event_file,status='old')
      call read_lhef_header(ifile,maxevt,MonteCarlo)
      write (*,*) 'file contains ',maxevt,' events'
      write (*,*) 'Give the number of splitted files you want'
      read (*,*) npart
      if (npart.gt.99) then
         write (*,*) 'too many event files (99 is max)', npart
         stop
      endif
      evts=int(dble(maxevt)/dble(npart))
      leftover=maxevt-evts*npart
      write (*,*) 'events per file:', evts
      write (*,*) 'left-over events:', leftover
      call read_lhef_init(ifile,
     &     IDBMUP,EBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,
     &     XSECUP,XERRUP,XMAXUP,LPRUP)

      mgfile=36
      open (unit=mgfile,file='mcatnlo.cmd',status='unknown')
      call open_cmd(mgfile)
      write (*,*) ''
      write (*,*) 'To write condor cmd file we need some extra info'
      write (*,*) 'Give the name for the MCatNLO executable'
      read (*,*) executable
      write (*,*) 'Give the name for the MCatNLO input file'
      read (*,*) inputfile

      loc=index(event_file,' ')
      loc1=index(executable,' ')
      loc2=index(inputfile,' ')
         
      do i=1,npart
         str='00'
         if (i.le.9) write (str(2:2),'(i1)') i
         if (i.gt.9.and.i.le.99) write (str(1:2),'(i2)') i
         fname1=event_file(1:loc-1)//'.'//str
         write (*,*) 'writing event file ',fname1
         ofile=35
         open(unit=ofile,file=fname1,status='unknown')
         call write_lhef_header(ofile,evts,MonteCarlo)
         call write_lhef_init(ofile,
     &        IDBMUP,EBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,
     &        XSECUP,XERRUP,XMAXUP,LPRUP)
         do j=1,evts
            call read_lhef_event(ifile,
     &           NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,
     &           IDUP,ISTUP,MOTHUP,ICOLUP,PUP,VTIMUP,SPINUP,buff)

            call write_lhef_event(ofile,
     &           NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,
     &           IDUP,ISTUP,MOTHUP,ICOLUP,PUP,VTIMUP,SPINUP,buff)
         enddo
         write(ofile,*)'</LesHouchesEvents>'
         close(ofile)
         write (mgfile,'(a)') 'Arguments = '//executable(1:loc1)//
     &        inputfile(1:loc2)//event_file(1:loc)//str
         write (mgfile,'(a)') 'queue'
      enddo
      close(ifile)
      close(mgfile)
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

