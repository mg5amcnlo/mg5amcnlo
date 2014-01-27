      program combine_jobs
C combines event files generated with the split_jobs.sh script
C It basically concatenates the files (no randomization in the events)
C updating the weight of each event, reflecting the relative number of
C events requested in the splits.
      implicit none
      include 'max_split.inc'
      character*140 buff
      character*10 filename
      character*14 filename2
      integer i,j,tot_evts,nevts(max_split),idummy,njobs
      double precision Xsect,wgt_fact(max_split)
      character*10 MonteCarlo
      integer IDBMUP(2),PDFGUP(2),PDFSUP(2),IDWTUP,NPRUP,LPRUP
      double precision EBMUP(2),XSECUP,XERRUP,XMAXUP
      integer IDBMUP1(2),PDFGUP1(2),PDFSUP1(2),IDWTUP1,NPRUP1,LPRUP1
      double precision EBMUP1(2),XSECUP1,XERRUP1,XMAXUP1
      INTEGER MAXNUP
      PARAMETER (MAXNUP=500)
      INTEGER NUP,IDPRUP,IDUP(MAXNUP),ISTUP(MAXNUP),
     # MOTHUP(2,MAXNUP),ICOLUP(2,MAXNUP)
      DOUBLE PRECISION XWGTUP,SCALUP,AQEDUP,AQCDUP,
     # PUP(5,MAXNUP),VTIMUP(MAXNUP),SPINUP(MAXNUP)
      logical done
      data done/.false./

c$$$      open (unit=1,file='res_1',status='old')
c$$$      read (1,'(a)') buff
c$$$      read (1,'(a)') buff
c$$$      read (buff(15:140),*) Xsect
c$$$      close(1)

      i=0
      tot_evts=0
      do while (i.le.99)
         i=i+1
         filename(1:7)='nevts__'
         if (i.le.9) then
            write (filename(8:10),'(i1,a)') i,'  '
         elseif (i.le.99) then
            write (filename(8:10),'(i2,a)') i,' '
         elseif (i.le.999) then
            write (filename(8:10),'(i3)') i
         else
c cycle when i is 100 here (could also change the do-loop to i.lt.99,
c but then the 'njobs=i-1' below is not correct).
            cycle
         endif
         open (unit=1,file=filename,status='old',err=99)
         read (1,*) nevts(i)
         tot_evts=tot_evts+nevts(i)
      enddo
 99   continue
      njobs=i-1
      do i=1,njobs
         wgt_fact(i)=dble(nevts(i))/dble(tot_evts)
      enddo

      open (unit=99,file='events.lhe',status='unknown')

      do i=1,njobs
         filename2(1:7)='events_'
         if (i.le.9) then
            write (filename2(8:14),'(i1,a)') i,'.lhe  '
         elseif (i.le.99) then
            write (filename2(8:14),'(i2,a)') i,'.lhe '
         elseif (i.le.999) then
            write (filename2(8:14),'(i3,a)') i,'.lhe'
         endif
         open (unit=2,file=filename2,status='old')
         call read_lhef_header(2,idummy,MonteCarlo)
         call read_lhef_init(2,
     &        IDBMUP,EBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,
     &        XSECUP,XERRUP,XMAXUP,LPRUP)
         if (i.eq.1) then
            call write_lhef_header(99,tot_evts,MonteCarlo)
            call write_lhef_init(99,
     &           IDBMUP,EBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,
     &           XSECUP,XERRUP,XMAXUP,LPRUP)
         endif
         do j=1,nevts(i)
            call read_lhef_event(2,
     &           NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,
     &           IDUP,ISTUP,MOTHUP,ICOLUP,PUP,VTIMUP,SPINUP,buff)
            XWGTUP=XWGTUP*wgt_fact(i)
            call write_lhef_event(99,
     &           NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,
     &           IDUP,ISTUP,MOTHUP,ICOLUP,PUP,VTIMUP,SPINUP,buff)
         enddo
         close(2,status='delete')
      enddo
      write(99,'(a)')'</LesHouchesEvents>'
      close(99)
      end
