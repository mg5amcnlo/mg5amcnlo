      program merge_events
c Merge many event files into a unique file
c gfortran -I../SubProcesses/P0_<anydir> -o merge_events
c merge_events.f handling_lhe_events.f fill_MC_mshell.f
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
      character*80 event_file,fname1,executable,inputfile,pref
      character*140 buff,tmpstr
      character*10 MonteCarlo,mc
      character*3 str
      integer evts,leftover,loc,loc1,loc2,num_file,sumevt
      integer numscales,numPDFpairs,isc,ipdf
      common/cwgxsec1/numscales,numPDFpairs
c
      write(*,*)'pref of the files (in the form pref.001, ...) '
      read(*,*)pref
      write(*,*)'number of files to merge'
      read(*,*)num_file


      ifile=34
      sumevt=0
      loc=index(pref,' ')
      do i=1,num_file
         str='000'
         if (i.le.9) write (str(3:3),'(i1)') i
         if (i.gt.9.and.i.le.99) write (str(2:3),'(i2)') i
         if (i.gt.99.and.i.le.999) write (str(1:3),'(i3)') i
         fname1=pref(1:loc-1)//'.'//str
         open(unit=ifile,file=fname1(1:loc+3),status='unknown')
         call read_lhef_header_full(ifile,evts,isc,ipdf,MonteCarlo)
         numscales=int(sqrt(dble(isc)))
         numPDFpairs=ipdf/2
         if(i.eq.1)then
            mc=MonteCarlo
            call read_lhef_init(ifile,
     &           IDBMUP,EBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,
     &           XSECUP,XERRUP,XMAXUP,LPRUP)
         endif
         if(MonteCarlo.ne.mc)then
            write(*,*)'incompatible files',mc,MonteCarlo,' file ',i
            stop
         endif
         sumevt=sumevt+evts
      enddo

      ofile=34
      ifile=35         
      open(unit=ofile,file=pref(1:loc-1)//'.complete',status='unknown')
      open(unit=ifile,file=pref(1:loc-1)//'.001',status='old')
      call copy_header(ifile,ofile,sumevt)
      close(ifile)
      call write_lhef_init(ofile,
     &        IDBMUP,EBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,
     &        XSECUP,XERRUP,XMAXUP,LPRUP)
      do i=1,num_file
         str='00'
         if (i.le.9) write (str(3:3),'(i1)') i
         if (i.gt.9.and.i.le.99) write (str(2:3),'(i2)') i
         if (i.gt.99.and.i.le.999) write (str(1:3),'(i3)') i
         fname1=pref(1:loc-1)//'.'//str
         write(*,*)'merging file ',fname1(1:loc+3)
         open(unit=ifile,file=fname1(1:loc+3),status='unknown')
         call read_lhef_header(ifile,evts,MonteCarlo)
         call read_lhef_init(ifile,
     &        IDBMUP,EBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,
     &        XSECUP,XERRUP,XMAXUP,LPRUP)
         do j=1,evts
            if(j.eq.evts)then
               call read_lhef_event(ifile,
     &              NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,
     &              IDUP,ISTUP,MOTHUP,ICOLUP,PUP,VTIMUP,SPINUP,buff)
               read(ifile,*)tmpstr
               goto 999
            endif
            call read_lhef_event(ifile,
     &           NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,
     &           IDUP,ISTUP,MOTHUP,ICOLUP,PUP,VTIMUP,SPINUP,buff)
 999        continue
            call write_lhef_event(ofile,
     &           NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,
     &           IDUP,ISTUP,MOTHUP,ICOLUP,PUP,VTIMUP,SPINUP,buff)
         enddo
         close(ifile)
      enddo
      write(ofile,*)'</LesHouchesEvents>'
      close(ofile)

      stop
      end
