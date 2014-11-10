c Utility routines for LHEF. Originally taken from collect_events.f
c
c Note: the routines read_lhef_event and write_lhef_event use the common
c blocks in reweight0.inc, relevant to reweight information. This is
c independent of the process, and in particular of process-related
c parameters such as nexternal, which is replaced here by (its supposed)
c upper bound maxparticles. The arrays which have one dimension defined
c by maxparticles may have a correspondence with process-specific ones,
c and the dimensions of the latter are typically defined by nexternal.
c Hence, one may need an explicit copy of one onto the other
c

      block data
      integer event_id
      common /c_event_id/ event_id
      data event_id /-99/
      logical rwgt_skip
      common /crwgt_skip/ rwgt_skip
      data rwgt_skip /.false./
      integer nattr,npNLO,npLO
      common/event_attributes/nattr,npNLO,npLO
      data nattr,npNLO,npLO /0,-1,-1/
      end

      subroutine write_lhef_header(ifile,nevents,MonteCarlo)
      implicit none 
      integer ifile,nevents
      character*10 MonteCarlo
c Scales
      character*80 muR_id_str,muF1_id_str,muF2_id_str,QES_id_str
      common/cscales_id_string/muR_id_str,muF1_id_str,
     #                         muF2_id_str,QES_id_str
c
      write(ifile,'(a)')
     #     '<LesHouchesEvents version="3.0">'
      write(ifile,'(a)')
     #     '  <!--'
      write(ifile,'(a)')'  <scalesfunctionalform>'
      write(ifile,'(a)')muR_id_str(1:len_trim(muR_id_str))
      write(ifile,'(a)')muF1_id_str(1:len_trim(muF1_id_str))
      write(ifile,'(a)')muF2_id_str(1:len_trim(muF2_id_str))
      write(ifile,'(a)')QES_id_str(1:len_trim(QES_id_str))
      write(ifile,'(a)')'  </scalesfunctionalform>'
      write(ifile,'(a)')
     #     MonteCarlo
      write(ifile,'(a)')
     #     '  -->'
      write(ifile,'(a)')
     #     '  <header>'
      write(ifile,250) nevents
      write(ifile,'(a)')
     #     '  </header>'
 250  format(1x,i8)
      return
      end


      subroutine write_lhef_header_banner(ifile,nevents,MonteCarlo,path)
      implicit none 
      integer ifile,nevents,iseed,i,pdf_set_min,pdf_set_max,idwgt
      double precision mcmass(-16:21),rw_Rscale_down,rw_Rscale_up
     $     ,rw_Fscale_down,rw_Fscale_up
c Scales
      character*80 muR_id_str,muF1_id_str,muF2_id_str,QES_id_str
      common/cscales_id_string/muR_id_str,muF1_id_str,
     #                         muF2_id_str,QES_id_str
      character*10 MonteCarlo
      character*100 path
      character*72 buffer,buffer_lc,buffer2
      logical rwgt_skip
      common /crwgt_skip/ rwgt_skip
      logical rwgt_skip_pdf,rwgt_skip_scales
      integer event_id
      common /c_event_id/ event_id
      include 'reweight_all.inc'
c     Set the event_id to 0. If 0 or positive, this value will be update
c     in write_lhe_event. It is set to -99 through a block data
c     statement.
      event_id=0
c
      write(ifile,'(a)') '<LesHouchesEvents version="3.0">'
      write(ifile,'(a)') '  <header>'
      write(ifile,'(a)') '  <MG5ProcCard>'
      open (unit=92,file=path(1:index(path," ")-1)//'proc_card_mg5.dat'
     &     ,err=99)
      do
         read(92,'(a)',err=89,end=89) buffer
         write(ifile,'(a)') buffer
      enddo
 89   close(92)
      write(ifile,'(a)') '  </MG5ProcCard>'
      write(ifile,'(a)') '  <slha>'
      open (unit=92,file=path(1:index(path," ")-1)//'param_card.dat'
     &     ,err=98)
      do
         read(92,'(a)',err=88,end=88) buffer
         write(ifile,'(a)') buffer
      enddo
 88   close(92)
      write(ifile,'(a)') '  </slha>'
      write(ifile,'(a)') '  <MGRunCard>'
      open (unit=92,file=path(1:index(path," ")-1)//'run_card.dat'
     &     ,err=97)
      rwgt_skip_pdf=.true.
      rwgt_skip_scales=.true.
      rwgt_skip=.true.
      pdf_set_min=-1
      pdf_set_max=-1
      numscales=0
      do
         read(92,'(a)',err=87,end=87) buffer
         buffer_lc=buffer
         call case_trap3(72,buffer_lc)
c Replace the random number seed with the one used...
         if(index(buffer_lc,'iseed').ne.0 .and. buffer(1:1).ne.'#')then
            open (unit=93,file="randinit",status="old",err=96)
            read(93,'(a)') buffer2
            if (index(buffer2,'=').eq.0) goto 96
            buffer2=buffer2(index(buffer2,'=')+1:)
            read(buffer2,*) iseed
            close(93)
            write(buffer,'(i11,a)')iseed,' =  iseed'
c Update the number of events
         elseif (index(buffer_lc,'nevents').ne.0 .and.
     &           buffer(1:1).ne.'#' .and.
     &           ( index(buffer_lc,'!').eq.0 .or.
     &             index(buffer_lc,'!').gt.index(buffer_lc,'nevents')
     &           )) then
            write(buffer,'(i11,a)')nevents,' = nevents'
         elseif (index(buffer_lc,'reweight_pdf').ne.0 .and.
     $           index(buffer_lc,'.true.').ne.0 .and.
     $           buffer(1:1).ne.'#') then
            rwgt_skip=.false.
            rwgt_skip_pdf=.false.
         elseif (index(buffer_lc,'pdf_set_min').ne.0 .and.
     &           buffer(1:1).ne.'#') then
            read(buffer(1:index(buffer,'=')-1),*) pdf_set_min
         elseif (index(buffer_lc,'pdf_set_max').ne.0 .and.
     &           buffer(1:1).ne.'#') then
            read(buffer(1:index(buffer,'=')-1),*) pdf_set_max
         elseif (index(buffer_lc,'reweight_scale').ne.0 .and.
     $           index(buffer_lc,'.true.').ne.0 .and.
     $           buffer(1:1).ne.'#') then
            rwgt_skip=.false.
            rwgt_skip_scales=.false.
            numscales=3
         elseif (index(buffer_lc,'rw_rscale_down').ne.0 .and.
     &           buffer(1:1).ne.'#') then
            read(buffer(1:index(buffer,'=')-1),*) rw_Rscale_down
         elseif (index(buffer_lc,'rw_rscale_up').ne.0 .and.
     &           buffer(1:1).ne.'#') then
            read(buffer(1:index(buffer,'=')-1),*) rw_Rscale_up
         elseif (index(buffer_lc,'rw_fscale_down').ne.0 .and.
     &           buffer(1:1).ne.'#') then
            read(buffer(1:index(buffer,'=')-1),*) rw_Fscale_down
         elseif (index(buffer_lc,'rw_fscale_up').ne.0 .and.
     &           buffer(1:1).ne.'#') then
            read(buffer(1:index(buffer,'=')-1),*) rw_Fscale_up
         endif
         goto 95
 96      write (*,*) '"randinit" file not found in write_lhef_header_'/
     &        /'banner: not overwriting iseed in event file header.'
 95      write(ifile,'(a)') buffer
      enddo
 87   close(92)
      if ( (pdf_set_min.ne.-1 .and. pdf_set_max.eq.-1) .or.
     &     (pdf_set_min.eq.-1 .and. pdf_set_max.ne.-1)) then
         write (*,*) 'Not consistent PDF reweigthing parameters'/
     $        /' found in the run_card.',pdf_set_min,pdf_set_max
         stop
      endif
      if (.not.rwgt_skip_pdf) numPDFpairs=(pdf_set_max-pdf_set_min+1)/2
      write(ifile,'(a)') '  </MGRunCard>'
c Functional form of the scales
      write(ifile,'(a)') '  <scalesfunctionalform>'
      write(ifile,'(a)')muR_id_str(1:len_trim(muR_id_str))
      write(ifile,'(a)')muF1_id_str(1:len_trim(muF1_id_str))
      write(ifile,'(a)')muF2_id_str(1:len_trim(muF2_id_str))
      write(ifile,'(a)')QES_id_str(1:len_trim(QES_id_str))
      write(ifile,'(a)') '  </scalesfunctionalform>'
c MonteCarlo Masses
      write(ifile,'(a)') '  <MonteCarloMasses>'
      call fill_MC_mshell_wrap(MonteCarlo,mcmass)
      do i=1,5
         write (ifile,'(2x,i6,3x,e12.6)')i,mcmass(i)
      enddo
      write (ifile,'(2x,i6,3x,e12.6)')11,mcmass(11)
      write (ifile,'(2x,i6,3x,e12.6)')13,mcmass(13)
      write (ifile,'(2x,i6,3x,e12.6)')15,mcmass(15)
      write (ifile,'(2x,i6,3x,e12.6)')21,mcmass(21)
      write(ifile,'(a)') '  </MonteCarloMasses>'
c Write here the reweight information if need be
      if (.not.rwgt_skip) then
         write(ifile,'(a)') '  <initrwgt>'
         if (numscales.ne.0) then
            if (numscales.ne.3) then
               write (*,*) 'Error in handling_lhe_events.f:'
               write (*,*) 'number of scale not equal to three'
     $              ,numscales
               stop
            endif
            write(ifile,'(a)') "    <weightgroup "/
     &           /"type='scale_variation' combine='envelope'>"
            write(ifile,602) "      <weight id='1001'>"/
     $           /" muR=",1d0," muF=",1d0," </weight>"
            write(ifile,602) "      <weight id='1002'>"/
     $           /" muR=",1d0," muF=",rw_Fscale_up," </weight>"
            write(ifile,602) "      <weight id='1003'>"/
     $           /" muR=",1d0," muF=",rw_Fscale_down," </weight>"
            write(ifile,602) "      <weight id='1004'>"/
     $           /" muR=",rw_Rscale_up," muF=",1d0," </weight>"
            write(ifile,602) "      <weight id='1005'> muR="
     $           ,rw_Rscale_up," muF=",rw_Fscale_up," </weight>"
            write(ifile,602) "      <weight id='1006'> muR="
     $           ,rw_Rscale_up," muF=",rw_Fscale_down," </weight>"
            write(ifile,602) "      <weight id='1007'> muR="
     $           ,rw_Rscale_down," muF=",1d0," </weight>"
            write(ifile,602) "      <weight id='1008'> muR="
     $           ,rw_Rscale_down," muF=",rw_Fscale_up," </weight>"
            write(ifile,602) "      <weight id='1009'> muR="
     $           ,rw_Rscale_down," muF=",rw_Fscale_down," </weight>"
            write(ifile,'(a)') "    </weightgroup>"
         endif
         if (numPDFpairs.ne.0) then
            if (pdf_set_min.lt.90000) then    ! MSTW & CTEQ
               write(ifile,'(a)') "    <weightgroup "/
     &              /"type='PDF_variation' combine='hessian'>"
            else                              ! NNPDF
               write(ifile,'(a)') "    <weightgroup "/
     &              /"type='PDF_variation' combine='gaussian'>"
            endif
            do i=1,numPDFpairs*2
               idwgt=2000+i
               write(ifile,'(a,i4,a,i6,a)') "      <weight id='",idwgt,
     $              "'> pdfset=",pdf_set_min+(i-1)," </weight>"
            enddo
            write(ifile,'(a)') "    </weightgroup>"
         endif
         write(ifile,'(a)') '  </initrwgt>'
      endif
      write(ifile,'(a)') '  </header>'
 250  format(1x,i8)
      return
 99   write (*,*) 'ERROR in write_lhef_header_banner: '/
     &     /' proc_card_mg5.dat not found   :',path(1:index(path," ")-1)
     &     //'proc_card_mg5.dat'
      stop
 98   write (*,*) 'ERROR in write_lhef_header_banner: '/
     &     /' param_card.dat not found   :',path(1:index(path," ")-1)
     &     //'param_card.dat'
      stop
 97   write (*,*) 'ERROR in write_lhef_header_banner: '/
     &     /' run_card.dat not found   :',path(1:index(path," ")-1)
     &     //'run_card.dat'
      stop
 602  format(a,e11.5,a,e11.5,a)
      end


      subroutine read_lhef_header(ifile,nevents,MonteCarlo)
      implicit none 
      integer ifile,nevents,i,ii,ii2,iistr
      character*10 MonteCarlo
      character*80 string,string0
      character*3 event_norm
      common/cevtnorm/event_norm
      character*80 muR_id_str,muF1_id_str,muF2_id_str,QES_id_str
      common/cscales_id_string/muR_id_str,muF1_id_str,
     #                         muF2_id_str,QES_id_str
      nevents = -1
      MonteCarlo = ''
c
      string='  '
      dowhile(string.ne.'  -->')
        string0=string
        if (index(string,'</header>').ne.0) return
        read(ifile,'(a)')string
        if(index(string,'= nevents').ne.0) read(string,*)nevents,string0
        if(index(string,'parton_shower').ne.0)then
           ii=iistr(string)
           ii2=min(index(string,'=')-1,ii+9)
           MonteCarlo=string(ii:ii2)
           call case_trap4(ii2-ii+1,MonteCarlo)
        endif
        if(index(string,'event_norm').ne.0)then
           ii=iistr(string)
           event_norm=string(ii:ii+2)
        endif
        if(index(string,'<scalesfunctionalform>').ne.0) then
           read(ifile,'(a)') muR_id_str
           read(ifile,'(a)') muF1_id_str
           read(ifile,'(a)') muF2_id_str
           read(ifile,'(a)') QES_id_str
        endif
      enddo
c Works only if the name of the MC is the last line of the comments
      MonteCarlo=string0(1:10)
      call case_trap4(10,MonteCarlo)
c Here we are at the end of (user-defined) comments. Now go to end
c of headers
      dowhile(index(string,'</header>').eq.0)
        string0=string
        read(ifile,'(a)')string
      enddo
c if the file is a partial file the header is non-standard   
      if (MonteCarlo .ne. '')read(string0,250) nevents
 250  format(1x,i8)
      return
      end


c Same as read_lhef_header, except that more parameters are read.
c Avoid overloading read_lhef_header, meant to be used in utilities
      subroutine read_lhef_header_full(ifile,nevents,itempsc,itempPDF,
     #                                 MonteCarlo)
      implicit none 
      integer ifile,nevents,i,ii,ii2,iistr,ipart,itempsc,itempPDF
      character*10 MonteCarlo
      character*80 string,string0
      character*3 event_norm
      common/cevtnorm/event_norm
      double precision temp,remcmass(-16:21)
      common/cremcmass/remcmass
c Scales
      character*80 muR_id_str,muF1_id_str,muF2_id_str,QES_id_str
      common/cscales_id_string/muR_id_str,muF1_id_str,
     #                         muF2_id_str,QES_id_str
      ipart=-1000000
      nevents = -1
      MonteCarlo = ''
      itempsc=0
      itempPDF=0
c
      string='  '
      dowhile(string.ne.'  -->')
        string0=string
        if (index(string,'</header>').ne.0) return
        read(ifile,'(a)')string
        if(index(string,'= nevents').ne.0)
     #    read(string,*)nevents,string0
        if(index(string,'parton_shower').ne.0)then
           ii=iistr(string)
           ii2=min(index(string,'=')-1,ii+9)
           MonteCarlo=string(ii:ii2)
           call case_trap4(ii2-ii+1,MonteCarlo)
        endif
        if(index(string,'event_norm').ne.0)then
           ii=iistr(string)
           event_norm=string(ii:ii+2)
        endif
        if( index(string,'<montecarlomasses>').ne.0 .or.
     #      index(string,'<MonteCarloMasses>').ne.0 )then
          read(ifile,'(a)')string
          dowhile( index(string,'</montecarlomasses>').eq.0 .and.
     #             index(string,'</MonteCarloMasses>').eq.0 )
            read(string,*)ipart,temp
            if(ipart.lt.-16.or.ipart.gt.21)then
              write(*,*)'Error in read_lhef_header:'
              write(*,*)' incomprehensible list of parton masses',ipart
              stop
            endif
            remcmass(ipart)=temp
            read(ifile,'(a)')string
          enddo
        endif
        if( index(string,'scale_variation').ne.0 )then
          read(ifile,'(a)')string
          itempsc=1
          dowhile( index(string,'</weightgroup>').eq.0 )
            read(ifile,'(a)')string
            itempsc=itempsc+1
          enddo
          itempsc=itempsc-1
        endif
        if( index(string,'PDF_variation').ne.0 )then
          read(ifile,'(a)')string
          itempPDF=1
          dowhile( index(string,'</weightgroup>').eq.0 )
            read(ifile,'(a)')string
            itempPDF=itempPDF+1
          enddo
          itempPDF=itempPDF-1
        endif
        if(index(string,'<scalesfunctionalform>').ne.0) then
           read(ifile,'(a)') muR_id_str
           read(ifile,'(a)') muF1_id_str
           read(ifile,'(a)') muF2_id_str
           read(ifile,'(a)') QES_id_str
        endif
      enddo
c Works only if the name of the MC is the last line of the comments
      MonteCarlo=string0(1:10)
      call case_trap4(10,MonteCarlo)
c Here we are at the end of (user-defined) comments. Now go to end
c of headers
      dowhile(index(string,'</header>').eq.0)
        string0=string
        read(ifile,'(a)')string
      enddo
c if the file is a partial file the header is non-standard   
      if (MonteCarlo .ne. '') read(string0,250) nevents
 250  format(1x,i8)
      return
      end


      subroutine write_lhef_init(ifile,
     #  IDBMUP,EBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,
     #  XSECUP,XERRUP,XMAXUP,LPRUP)
      implicit none
      integer ifile,i,IDBMUP(2),PDFGUP(2),PDFSUP(2),IDWTUP,NPRUP,LPRUP
      double precision EBMUP(2),XSECUP,XERRUP,XMAXUP
      double precision XSECUP2(100),XERRUP2(100),XMAXUP2(100)
      integer LPRUP2(100)
      common /lhef_init/XSECUP2,XERRUP2,XMAXUP2,LPRUP2
c
      write(ifile,'(a)')
     # '  <init>'
      write(ifile,501)IDBMUP(1),IDBMUP(2),EBMUP(1),EBMUP(2),
     #                PDFGUP(1),PDFGUP(2),PDFSUP(1),PDFSUP(2),
     #                IDWTUP,NPRUP
      write(ifile,502)XSECUP,XERRUP,XMAXUP,LPRUP
      if (NPRUP.gt.1) then
         do i=2,NPRUP
            write(ifile,502)XSECUP2(i),XERRUP2(i),XMAXUP2(i),LPRUP2(i)
         enddo
      endif
      write(ifile,'(a)')
     # '  </init>'
 501  format(2(1x,i6),2(1x,e14.8),2(1x,i2),2(1x,i6),1x,i2,1x,i3)
 502  format(3(1x,e14.8),1x,i6)
c
      return
      end


      subroutine read_lhef_init(ifile,
     #  IDBMUP,EBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,
     #  XSECUP,XERRUP,XMAXUP,LPRUP)
      implicit none
      integer ifile,i,IDBMUP(2),PDFGUP(2),PDFSUP(2),IDWTUP,NPRUP,LPRUP
      double precision EBMUP(2),XSECUP,XERRUP,XMAXUP
      double precision XSECUP2(100),XERRUP2(100),XMAXUP2(100)
      integer LPRUP2(100)
      common /lhef_init/XSECUP2,XERRUP2,XMAXUP2,LPRUP2
      character*80 string
c
      read(ifile,'(a)')string
      read(ifile,*)IDBMUP(1),IDBMUP(2),EBMUP(1),EBMUP(2),
     #                PDFGUP(1),PDFGUP(2),PDFSUP(1),PDFSUP(2),
     #                IDWTUP,NPRUP
      read(ifile,*)XSECUP,XERRUP,XMAXUP,LPRUP
      XSECUP2(1)=XSECUP
      XERRUP2(1)=XERRUP
      XMAXUP2(1)=XMAXUP
      LPRUP2(1)=LPRUP
      if (NPRUP.gt.1) then
         do i=2,NPRUP
            read(ifile,*)XSECUP2(i),XERRUP2(i),XMAXUP2(i),LPRUP2(i)
         enddo
      endif
      read(ifile,'(a)')string
c
      return
      end

      subroutine write_lhef_event(ifile,
     # NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,
     # IDUP,ISTUP,MOTHUP,ICOLUP,PUP,VTIMUP,SPINUP,buff)
      implicit none
      INTEGER NUP,IDPRUP,IDUP(*),ISTUP(*),MOTHUP(2,*),ICOLUP(2,*)
      DOUBLE PRECISION XWGTUP,SCALUP,AQEDUP,AQCDUP,
     # PUP(5,*),VTIMUP(*),SPINUP(*)
      character*140 buff
      integer ifile,i
      character*9 ch1
      integer isorh_lhe,ifks_lhe,jfks_lhe,fksfather_lhe,ipartner_lhe
      double precision scale1_lhe,scale2_lhe
      integer ii,j,nps,nng,iFKS,idwgt
      double precision wgtcentral,wgtmumin,wgtmumax,wgtpdfmin,wgtpdfmax
      logical rwgt_skip
      common /crwgt_skip/ rwgt_skip
      integer event_id
      common /c_event_id/ event_id
      integer i_process
      common/c_addwrite/i_process
      integer nattr,npNLO,npLO
      common/event_attributes/nattr,npNLO,npLO
      include 'reweight_all.inc'
      include 'unlops.inc'
c     if event_id is zero or positive (that means that there was a call
c     to write_lhef_header_banner) update it and write it
c RF: don't use the event_id:
      event_id = -99
c
      if (event_id.ge.0) then
         event_id=event_id+1
         if (event_id.le.9) then
            write(ifile,'(a,i1,a)') "  <event id='",event_id,"'>"
         elseif(event_id.le.99) then
            write(ifile,'(a,i2,a)') "  <event id='",event_id,"'>"
         elseif(event_id.le.999) then
            write(ifile,'(a,i3,a)') "  <event id='",event_id,"'>"
         elseif(event_id.le.9999) then
            write(ifile,'(a,i4,a)') "  <event id='",event_id,"'>"
         elseif(event_id.le.99999) then
            write(ifile,'(a,i5,a)') "  <event id='",event_id,"'>"
         elseif(event_id.le.999999) then
            write(ifile,'(a,i6,a)') "  <event id='",event_id,"'>"
         elseif(event_id.le.9999999) then
            write(ifile,'(a,i7,a)') "  <event id='",event_id,"'>"
         elseif(event_id.le.99999999) then
            write(ifile,'(a,i8,a)') "  <event id='",event_id,"'>"
         elseif(event_id.le.999999999) then
            write(ifile,'(a,i9,a)') "  <event id='",event_id,"'>"
         else
            write (ifile,*) "ERROR: EVENT ID TOO LARGE",event_id
            write (*,*) "ERROR: EVENT ID TOO LARGE",event_id
            stop
         endif
      elseif(nattr.eq.2) then
         if ( (npLO.ge.10.or.npLO.lt.0) .and.
     &        (npNLO.ge.10.or.npNLO.lt.0)) then
            write(ifile,'(a,i2,a,i2,a)') "  <event npLO=' ",npLO
     $           ," ' npNLO=' ",npNLO," '>"
         elseif( (npLO.lt.10.or.npLO.ge.0) .and.
     &        (npNLO.ge.10.or.npNLO.lt.0)) then
            write(ifile,'(a,i1,a,i2,a)') "  <event npLO=' ",npLO
     $           ," ' npNLO=' ",npNLO," '>"
         elseif( (npLO.ge.10.or.npLO.lt.0) .and.
     &        (npNLO.lt.10.or.npNLO.ge.0)) then
            write(ifile,'(a,i2,a,i1,a)') "  <event npLO=' ",npLO
     $           ," ' npNLO=' ",npNLO," '>"
         elseif( (npLO.lt.10.or.npLO.ge.0) .and.
     &        (npNLO.lt.10.or.npNLO.ge.0)) then
            write(ifile,'(a,i1,a,i1,a)') "  <event npLO=' ",npLO
     $           ," ' npNLO=' ",npNLO," '>"
         endif
      else
         write(ifile,'(a)') '  <event>'
      endif
      write(ifile,503)NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP
      do i=1,nup
        write(ifile,504)IDUP(I),ISTUP(I),MOTHUP(1,I),MOTHUP(2,I),
     #                  ICOLUP(1,I),ICOLUP(2,I),
     #                  PUP(1,I),PUP(2,I),PUP(3,I),PUP(4,I),PUP(5,I),
     #                  VTIMUP(I),SPINUP(I)
      enddo
      if(buff(1:1).eq.'#' .and. .not.rwgt_skip) then
        write(ifile,'(a)') buff(1:len_trim(buff))
        read(buff,*)ch1,iSorH_lhe,ifks_lhe,jfks_lhe,
     #                    fksfather_lhe,ipartner_lhe,
     #                    scale1_lhe,scale2_lhe,
     #                    jwgtinfo,mexternal,iwgtnumpartn,
     #         wgtcentral,wgtmumin,wgtmumax,wgtpdfmin,wgtpdfmax
        if(jwgtinfo.ge.1.and.jwgtinfo.le.4)then
           write(ifile,'(a)') '  <rwgt>'
          write(ifile,401)wgtref,wgtqes2(2)
          write(ifile,402)wgtxbj(1,1),wgtxbj(2,1),
     #                    wgtxbj(1,2),wgtxbj(2,2),
     #                    wgtxbj(1,3),wgtxbj(2,3),
     #                    wgtxbj(1,4),wgtxbj(2,4)
          if(jwgtinfo.eq.1)then
            write(ifile,403)wgtmuR2(1),wgtmuF12(1),wgtmuF22(1),
     #                      wgtmuR2(2),wgtmuF12(2),wgtmuF22(2)
          elseif(jwgtinfo.eq.2)then
            ii=iSorH_lhe+1
            if(ii.eq.3)ii=1
            write(ifile,404)wgtmuR2(ii),wgtmuF12(ii),wgtmuF22(ii)
            do i=1,mexternal
              write(ifile,405)(wgtkinE(j,i,iSorH_lhe),j=0,3)
            enddo
          elseif(jwgtinfo.eq.3 .or. jwgtinfo.eq.4)then
            do i=1,mexternal
              write(ifile,405)(wgtkinE(j,i,1),j=0,3)
            enddo
            do i=1,mexternal
              write(ifile,405)(wgtkinE(j,i,2),j=0,3)
            enddo
          endif
          write(ifile,441)wgtwreal(1),wgtwreal(2),
     #                    wgtwreal(3),wgtwreal(4)
          write(ifile,441)wgtwdeg(3),wgtwdeg(4),
     #                    wgtwdegmuf(3),wgtwdegmuf(4)
          write(ifile,405)wgtwborn(2),wgtwns(2),
     #                    wgtwnsmuf(2),wgtwnsmur(2)
          do i=1,iwgtnumpartn
            write(ifile,442)wgtwmcxsecE(i),
     #                      wgtmcxbjE(1,i),wgtmcxbjE(2,i)
          enddo
          if(jwgtinfo.eq.4) write(ifile,
     f         '(1x,e14.8,1x,e14.8,1x,i4,1x,i4)')
     &       wgtbpower,wgtcpower,nFKSprocess_used,nFKSprocess_used_born
          write(ifile,'(a)') '  </rwgt>'
         elseif(jwgtinfo.eq.5) then
           write(ifile,'(a)')'  <rwgt>'
           if (iSorH_lhe.eq.1) then ! S-event
              write(ifile,'(1x,e14.8,1x,e14.8,i4,i4)') 
     f             wgtbpower,wgtcpower,nScontributions,i_process
              write(ifile,'(1x,i4,1x,e14.8)') nFKSprocess_used_born
     &             ,wgtref_nbody_all(i_process)
              do i=1,mexternal
                 write(ifile,405)(wgtkin_all(j,i,2,0),j=0,3)
              enddo
              write(ifile,402) wgtxbj_all(1,2,0),wgtxbj_all(2,2,0)
              write(ifile,'(1x,e14.8)') wgtqes2_all(2,0)
              write(ifile,405)wgtwborn_all,wgtwns_all,
     &             wgtwnsmuf_all,wgtwnsmur_all
              write(ifile,404) wgtmuR2_all(2,0),wgtmuF12_all(2,0)
     $             ,wgtmuF22_all(2,0)
              do ii=1,nScontributions
                 write(ifile,'(1x,i4)') nFKSprocess_reweight(ii)
                 iFKS=nFKSprocess_reweight(ii)*2-1
                 write(ifile,'(1x,e14.8,1x,i4)')
     &                wgtref_all(iFKS,i_process),iwgtnumpartn_all(iFKS)
                 do i=1,mexternal
                    write(ifile,405)(wgtkin_all(j,i,1,iFKS),j=0,3)
                 enddo
c$$$                 do i=1,mexternal
c$$$                    write(ifile,405)(wgtkin_all(j,i,2,iFKS),j=0,3)
c$$$                 enddo
                 write(ifile,402)
     &                wgtxbj_all(1,1,iFKS),wgtxbj_all(2,1,iFKS),
     &                wgtxbj_all(1,2,iFKS),wgtxbj_all(2,2,iFKS),
     &                wgtxbj_all(1,3,iFKS),wgtxbj_all(2,3,iFKS),
     &                wgtxbj_all(1,4,iFKS),wgtxbj_all(2,4,iFKS)
                 write(ifile,'(1x,e14.8)') wgtqes2_all(2,iFKS)
                 write(ifile,441)wgtwreal_all(1,iFKS),wgtwreal_all(2
     &                ,iFKS),wgtwreal_all(3,iFKS),wgtwreal_all(4,iFKS)
                 write(ifile,441)wgtwdeg_all(3,iFKS),wgtwdeg_all(4,iFKS)
     &                ,wgtwdegmuf_all(3,iFKS),wgtwdegmuf_all(4,iFKS)
                 do i=1,iwgtnumpartn_all(iFKS)
                    write(ifile,442)wgtwmcxsec_all(i,iFKS),
     &                   wgtmcxbj_all(1,i,iFKS),wgtmcxbj_all(2,i,iFKS)
                 enddo
                 write(ifile,404) wgtmuR2_all(1,iFKS),wgtmuF12_all(1
     $                ,iFKS),wgtmuF22_all(1,iFKS)
                 write(ifile,404) wgtmuR2_all(2,iFKS),wgtmuF12_all(2
     $                ,iFKS),wgtmuF22_all(2,iFKS)
              enddo
           elseif (iSorH_lhe.eq.2) then ! H-event
              write(ifile,'(1x,e14.8,1x,e14.8,i4)')
     f             wgtbpower,wgtcpower,i_process
              iFKS=nFKSprocess_used*2
              write(ifile,'(1x,i4)') nFKSprocess_used
              write(ifile,'(1x,e14.8,1x,i4)') wgtref_all(iFKS,i_process)
     &             ,iwgtnumpartn_all(iFKS)
              do i=1,mexternal
                 write(ifile,405)(wgtkin_all(j,i,1,iFKS),j=0,3)
              enddo
              do i=1,mexternal
                 write(ifile,405)(wgtkin_all(j,i,2,iFKS),j=0,3)
              enddo
              write(ifile,402)
     &                wgtxbj_all(1,1,iFKS),wgtxbj_all(2,1,iFKS),
     &                wgtxbj_all(1,2,iFKS),wgtxbj_all(2,2,iFKS),
     &                wgtxbj_all(1,3,iFKS),wgtxbj_all(2,3,iFKS),
     &                wgtxbj_all(1,4,iFKS),wgtxbj_all(2,4,iFKS)
              write(ifile,441)wgtwreal_all(1,iFKS),wgtwreal_all(2
     &             ,iFKS),wgtwreal_all(3,iFKS),wgtwreal_all(4,iFKS)
              do i=1,iwgtnumpartn_all(iFKS)
                 write(ifile,442)wgtwmcxsec_all(i,iFKS),
     &                wgtmcxbj_all(1,i,iFKS),wgtmcxbj_all(2,i,iFKS)
              enddo
              write(ifile,404) wgtmuR2_all(1,iFKS),wgtmuF12_all(1,iFKS)
     $             ,wgtmuF22_all(1,iFKS)
              write(ifile,404) wgtmuR2_all(2,iFKS),wgtmuF12_all(2,iFKS)
     $             ,wgtmuF22_all(2,iFKS)
           else
              write (*,*) 'Not an S- or H-event in write_lhef_event'
              stop
           endif
           write(ifile,'(a)')'  </rwgt>'
         elseif(jwgtinfo.eq.15) then
           write(ifile,'(a)')'  <unlops>'
           write(ifile,*)NUP_H
           do i=1,NUP_H
              write(ifile,504)IDUP_H(I),ISTUP_H(I),MOTHUP_H(1,I)
     $             ,MOTHUP_H(2,I),ICOLUP_H(1,I),ICOLUP_H(2,I),PUP_H(1
     $             ,I),PUP_H(2,I),PUP_H(3,I),PUP_H(4,I),PUP_H(5,I),
     $             VTIMUP_H(I),SPINUP_H(I)
           enddo
           write(ifile,'(a)')'  </unlops>'
        elseif(jwgtinfo.eq.8)then
           write(ifile,'(a)') '  <rwgt>'
          write(ifile,406)wgtref,wgtxsecmu(1,1),numscales,numPDFpairs
          do i=1,numscales
            write(ifile,404)(wgtxsecmu(i,j),j=1,numscales)
          enddo
          do i=1,numPDFpairs
            nps=2*i-1
            nng=2*i
            write(ifile,404)wgtxsecPDF(nps),wgtxsecPDF(nng)
          enddo
          write(ifile,'(a)') '  </rwgt>'

        elseif(jwgtinfo.eq.9)then
           write(ifile,'(a)') '  <rwgt>'
           do i=1,numscales
              do j=1,numscales
                 idwgt=1000+(i-1)*numscales+j
                 write(ifile,601) "   <wgt id='",idwgt,"'>",wgtxsecmu(i
     $                ,j)," </wgt>"
              enddo
           enddo
           do i=1,2*numPDFpairs
              idwgt=2000+i
              write(ifile,601) "   <wgt id='",idwgt,"'>",wgtxsecPDF(i)
     $             ," </wgt>"
           enddo
           write(ifile,'(a)') '  </rwgt>'
        endif
      endif
      write(ifile,'(a)') '  </event>'
 401  format(2(1x,e14.8))
 402  format(8(1x,e14.8))
 403  format(6(1x,e14.8))
 404  format(3(1x,e14.8))
 405  format(4(1x,e14.8))
 406  format(2(1x,e14.8),2(1x,i3))
 441  format(4(1x,e16.10))
 442  format(1x,e16.10,2(1x,e14.8))
 503  format(1x,i2,1x,i6,4(1x,e14.8))
 504  format(1x,i8,1x,i2,4(1x,i4),5(1x,e14.8),2(1x,e10.4))
 601  format(a12,i4,a2,1x,e11.5,a7)
c
      return
      end


      subroutine read_lhef_event(ifile,
     # NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,
     # IDUP,ISTUP,MOTHUP,ICOLUP,PUP,VTIMUP,SPINUP,buff)
      implicit none
      INTEGER NUP,IDPRUP,IDUP(*),ISTUP(*),MOTHUP(2,*),ICOLUP(2,*)
      DOUBLE PRECISION XWGTUP,SCALUP,AQEDUP,AQCDUP,
     # PUP(5,*),VTIMUP(*),SPINUP(*)
      integer ifile,i
      character*140 buff
      character*80 string
      character*12 dummy12
      character*2 dummy2
      character*9 ch1
      integer isorh_lhe,ifks_lhe,jfks_lhe,fksfather_lhe,ipartner_lhe
      double precision scale1_lhe,scale2_lhe
      integer ii,j,nps,nng,iFKS,idwgt
      double precision wgtcentral,wgtmumin,wgtmumax,wgtpdfmin,wgtpdfmax
      integer i_process
      common/c_addwrite/i_process
      integer nattr,npNLO,npLO
      common/event_attributes/nattr,npNLO,npLO
      include 'reweight_all.inc'
      include 'unlops.inc'
c
      read(ifile,'(a)')string
      nattr=0
      npNLO=-1
      npLO=-1
      if (index(string,'npLO').ne.0) then
         nattr=2
         read(string(index(string,'npLO')+6:),*) npLO
      endif
      if (index(string,'npNLO').ne.0) then
         nattr=2
         read(string(index(string,'npNLO')+7:),*) npNLO
      endif
      read(ifile,*)NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP
      do i=1,nup
        read(ifile,*)IDUP(I),ISTUP(I),MOTHUP(1,I),MOTHUP(2,I),
     #                  ICOLUP(1,I),ICOLUP(2,I),
     #                  PUP(1,I),PUP(2,I),PUP(3,I),PUP(4,I),PUP(5,I),
     #                  VTIMUP(I),SPINUP(I)
      enddo
      read(ifile,'(a)')buff
      if(buff(1:1).eq.'#')then
        read(buff,*)ch1,iSorH_lhe,ifks_lhe,jfks_lhe,
     #                    fksfather_lhe,ipartner_lhe,
     #                    scale1_lhe,scale2_lhe,
     #                    jwgtinfo,mexternal,iwgtnumpartn,
     #         wgtcentral,wgtmumin,wgtmumax,wgtpdfmin,wgtpdfmax
        if(jwgtinfo.ge.1.and.jwgtinfo.le.4)then
          read(ifile,'(a)')string
          read(ifile,401)wgtref,wgtqes2(2)
          read(ifile,402)wgtxbj(1,1),wgtxbj(2,1),
     #                   wgtxbj(1,2),wgtxbj(2,2),
     #                   wgtxbj(1,3),wgtxbj(2,3),
     #                   wgtxbj(1,4),wgtxbj(2,4)
          if(jwgtinfo.eq.1)then
            read(ifile,403)wgtmuR2(1),wgtmuF12(1),wgtmuF22(1),
     #                     wgtmuR2(2),wgtmuF12(2),wgtmuF22(2)
          elseif(jwgtinfo.eq.2)then
            ii=iSorH_lhe+1
            if(ii.eq.3)ii=1
            read(ifile,404)wgtmuR2(ii),wgtmuF12(ii),wgtmuF22(ii)
            do i=1,mexternal
              read(ifile,405)(wgtkinE(j,i,iSorH_lhe),j=0,3)
            enddo
          elseif(jwgtinfo.eq.3 .or. jwgtinfo.eq.4)then
            do i=1,mexternal
              read(ifile,405)(wgtkinE(j,i,1),j=0,3)
            enddo
            do i=1,mexternal
              read(ifile,405)(wgtkinE(j,i,2),j=0,3)
            enddo
          endif
          read(ifile,441)wgtwreal(1),wgtwreal(2),
     #                   wgtwreal(3),wgtwreal(4)
          read(ifile,441)wgtwdeg(3),wgtwdeg(4),
     #                   wgtwdegmuf(3),wgtwdegmuf(4)
          read(ifile,405)wgtwborn(2),wgtwns(2),
     #                    wgtwnsmuf(2),wgtwnsmur(2)
          do i=1,iwgtnumpartn
            read(ifile,442)wgtwmcxsecE(i),
     #                     wgtmcxbjE(1,i),wgtmcxbjE(2,i)
          enddo
          if(jwgtinfo.eq.4) read(ifile,
     f     '(1x,e14.8,1x,e14.8,1x,i4,1x,i4)')
     &     wgtbpower,wgtcpower,nFKSprocess_used,nFKSprocess_used_born
          read(ifile,'(a)')string
        elseif(jwgtinfo.eq.5) then
           read(ifile,'(a)')string
           if (iSorH_lhe.eq.1) then ! S-event
              read(ifile,'(1x,e14.8,1x,e14.8,i4,i4)')
     f             wgtbpower,wgtcpower,nScontributions
     $             ,i_process
              read(ifile,'(1x,i4,1x,e14.8)') nFKSprocess_used_born
     &             ,wgtref_nbody_all(i_process)
              do i=1,mexternal
                 read(ifile,405)(wgtkin_all(j,i,2,0),j=0,3)
              enddo
              read(ifile,402) wgtxbj_all(1,2,0),wgtxbj_all(2,2,0)
              read(ifile,'(1x,e14.8)') wgtqes2_all(2,0)
              read(ifile,405)wgtwborn_all,wgtwns_all,
     &             wgtwnsmuf_all,wgtwnsmur_all
              read(ifile,404) wgtmuR2_all(2,0),wgtmuF12_all(2,0)
     $             ,wgtmuF22_all(2,0)
              do ii=1,nScontributions
                 read(ifile,'(1x,i4)') nFKSprocess_reweight(ii)
                 iFKS=nFKSprocess_reweight(ii)*2-1
                 read(ifile,'(1x,e14.8,1x,i4)') wgtref_all(iFKS
     $                ,i_process),iwgtnumpartn_all(iFKS)
                 do i=1,mexternal
                    read(ifile,405)(wgtkin_all(j,i,1,iFKS),j=0,3)
                 enddo
                 do i=1,mexternal
                    do j=0,3
                       wgtkin_all(j,i,2,iFKS)=wgtkin_all(j,i,2,0)
                    enddo
                 enddo
                 read(ifile,402)
     &                wgtxbj_all(1,1,iFKS),wgtxbj_all(2,1,iFKS),
     &                wgtxbj_all(1,2,iFKS),wgtxbj_all(2,2,iFKS),
     &                wgtxbj_all(1,3,iFKS),wgtxbj_all(2,3,iFKS),
     &                wgtxbj_all(1,4,iFKS),wgtxbj_all(2,4,iFKS)
                 read(ifile,'(1x,e14.8)') wgtqes2_all(2,iFKS)
                 read(ifile,441)wgtwreal_all(1,iFKS),wgtwreal_all(2
     &                ,iFKS),wgtwreal_all(3,iFKS),wgtwreal_all(4,iFKS)
                 read(ifile,441)wgtwdeg_all(3,iFKS),wgtwdeg_all(4,iFKS)
     &                ,wgtwdegmuf_all(3,iFKS),wgtwdegmuf_all(4,iFKS)
                 do i=1,iwgtnumpartn_all(iFKS)
                    read(ifile,442)wgtwmcxsec_all(i,iFKS),
     &                   wgtmcxbj_all(1,i,iFKS),wgtmcxbj_all(2,i,iFKS)
                 enddo
                 read(ifile,404) wgtmuR2_all(1,iFKS),wgtmuF12_all(1
     $                ,iFKS),wgtmuF22_all(1,iFKS)
                 read(ifile,404) wgtmuR2_all(2,iFKS),wgtmuF12_all(2
     $                ,iFKS),wgtmuF22_all(2,iFKS)
              enddo
           elseif (iSorH_lhe.eq.2) then ! H-event
              read(ifile,'(1x,e14.8,1x,e14.8,i4)')
     f             wgtbpower,wgtcpower,i_process
              read(ifile,'(1x,i4)') nFKSprocess_used
              iFKS=nFKSprocess_used*2
              read(ifile,'(1x,e14.8,1x,i4)') wgtref_all(iFKS,i_process)
     $             ,iwgtnumpartn_all(iFKS)
              do i=1,mexternal
                 read(ifile,405)(wgtkin_all(j,i,1,iFKS),j=0,3)
              enddo
              do i=1,mexternal
                 read(ifile,405)(wgtkin_all(j,i,2,iFKS),j=0,3)
              enddo
              read(ifile,402)
     &                wgtxbj_all(1,1,iFKS),wgtxbj_all(2,1,iFKS),
     &                wgtxbj_all(1,2,iFKS),wgtxbj_all(2,2,iFKS),
     &                wgtxbj_all(1,3,iFKS),wgtxbj_all(2,3,iFKS),
     &                wgtxbj_all(1,4,iFKS),wgtxbj_all(2,4,iFKS)
              read(ifile,441)wgtwreal_all(1,iFKS),wgtwreal_all(2
     &             ,iFKS),wgtwreal_all(3,iFKS),wgtwreal_all(4,iFKS)
              do i=1,iwgtnumpartn_all(iFKS)
                 read(ifile,442)wgtwmcxsec_all(i,iFKS),
     &                wgtmcxbj_all(1,i,iFKS),wgtmcxbj_all(2,i,iFKS)
              enddo
              read(ifile,404) wgtmuR2_all(1,iFKS),wgtmuF12_all(1,iFKS)
     $             ,wgtmuF22_all(1,iFKS)
              read(ifile,404) wgtmuR2_all(2,iFKS),wgtmuF12_all(2,iFKS)
     $             ,wgtmuF22_all(2,iFKS)
           else
              write (*,*) 'Not an S- or H-event in write_lhef_event'
              stop
           endif
           read(ifile,'(a)')string
         elseif(jwgtinfo.eq.15) then
           read(ifile,'(a)') string
           read(ifile,*)NUP_H
           do i=1,NUP_H
              read(ifile,*) IDUP_H(I),ISTUP_H(I),MOTHUP_H(1,I)
     $             ,MOTHUP_H(2,I),ICOLUP_H(1,I),ICOLUP_H(2,I),PUP_H(1
     $             ,I),PUP_H(2,I),PUP_H(3,I),PUP_H(4,I),PUP_H(5,I),
     $             VTIMUP_H(I),SPINUP_H(I)
           enddo
           read(ifile,'(a)') string
        elseif(jwgtinfo.eq.8)then
          read(ifile,'(a)')string
          read(ifile,406)wgtref,wgtxsecmu(1,1),numscales,numPDFpairs
          do i=1,numscales
            read(ifile,404)(wgtxsecmu(i,j),j=1,numscales)
          enddo
          do i=1,numPDFpairs
            nps=2*i-1
            nng=2*i
            read(ifile,404)wgtxsecPDF(nps),wgtxsecPDF(nng)
          enddo
          read(ifile,'(a)')string
        elseif(jwgtinfo.eq.9)then
           read(ifile,'(a)')string
           wgtref=XWGTUP
           do i=1,numscales
              do j=1,numscales
                 call read_rwgt_line(ifile,idwgt,wgtxsecmu(i,j))
              enddo
           enddo
           do i=1,2*numPDFpairs
              call read_rwgt_line(ifile,idwgt,wgtxsecPDF(i))
           enddo
           if (numscales.eq.0 .and. numPDFpairs.ne.0) then
              wgtxsecmu(1,1)=XWGTUP
           endif
           read(ifile,'(a)')string
        endif
        read(ifile,'(a)')string
      else
        string=buff(1:len_trim(buff))
        buff=' '
      endif
 401  format(2(1x,e14.8))
 402  format(8(1x,e14.8))
 403  format(6(1x,e14.8))
 404  format(3(1x,e14.8))
 405  format(4(1x,e14.8))
 406  format(2(1x,e14.8),2(1x,i3))
 441  format(4(1x,e16.10))
 442  format(1x,e16.10,2(1x,e14.8))
 503  format(1x,i2,1x,i6,4(1x,e14.8))
 504  format(1x,i8,1x,i2,4(1x,i4),5(1x,e14.8),2(1x,e10.4))
c
      return
      end


c Same as read_lhef_event, except for the end-of-file catch
      subroutine read_lhef_event_catch(ifile,
     # NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,
     # IDUP,ISTUP,MOTHUP,ICOLUP,PUP,VTIMUP,SPINUP,buff)
      implicit none
      INTEGER NUP,IDPRUP,IDUP(*),ISTUP(*),MOTHUP(2,*),ICOLUP(2,*)
      DOUBLE PRECISION XWGTUP,SCALUP,AQEDUP,AQCDUP,
     # PUP(5,*),VTIMUP(*),SPINUP(*)
      integer ifile,i
      character*140 buff
      character*80 string
      character*12 dummy12
      character*2 dummy2
      character*9 ch1
      integer isorh_lhe,ifks_lhe,jfks_lhe,fksfather_lhe,ipartner_lhe
      double precision scale1_lhe,scale2_lhe
      integer ii,j,nps,nng,iFKS,idwgt
      double precision wgtcentral,wgtmumin,wgtmumax,wgtpdfmin,wgtpdfmax
      integer i_process
      common/c_addwrite/i_process
      integer nattr,npNLO,npLO
      common/event_attributes/nattr,npNLO,npLO
      include 'reweight_all.inc'
      include 'unlops.inc'
c
      read(ifile,'(a)')string
      if(index(string,'<event').eq.0)then
        if(index(string,'</LesHouchesEvents>').ne.0)then
          buff='endoffile'
          return
        else
          write(*,*)'Unknown structure in read_lhef_event_catch:'
          write(*,*)string(1:len_trim(string))
          stop
        endif
      endif
      nattr=0
      npNLO=-1
      npLO=-1
      if (index(string,'npLO').ne.0) then
         nattr=2
         read(string(index(string,'npLO')+6:),*) npLO
      endif
      if (index(string,'npNLO').ne.0) then
         nattr=2
         read(string(index(string,'npNLO')+7:),*) npNLO
      endif
      read(ifile,*)NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP
      do i=1,nup
        read(ifile,*)IDUP(I),ISTUP(I),MOTHUP(1,I),MOTHUP(2,I),
     #                  ICOLUP(1,I),ICOLUP(2,I),
     #                  PUP(1,I),PUP(2,I),PUP(3,I),PUP(4,I),PUP(5,I),
     #                  VTIMUP(I),SPINUP(I)
      enddo
      read(ifile,'(a)')buff
      if(buff(1:1).eq.'#')then
        read(buff,*)ch1,iSorH_lhe,ifks_lhe,jfks_lhe,
     #                    fksfather_lhe,ipartner_lhe,
     #                    scale1_lhe,scale2_lhe,
     #                    jwgtinfo,mexternal,iwgtnumpartn,
     #         wgtcentral,wgtmumin,wgtmumax,wgtpdfmin,wgtpdfmax
        if(jwgtinfo.ge.1.and.jwgtinfo.le.4)then
          read(ifile,'(a)')string
          read(ifile,401)wgtref,wgtqes2(2)
          read(ifile,402)wgtxbj(1,1),wgtxbj(2,1),
     #                   wgtxbj(1,2),wgtxbj(2,2),
     #                   wgtxbj(1,3),wgtxbj(2,3),
     #                   wgtxbj(1,4),wgtxbj(2,4)
          if(jwgtinfo.eq.1)then
            read(ifile,403)wgtmuR2(1),wgtmuF12(1),wgtmuF22(1),
     #                     wgtmuR2(2),wgtmuF12(2),wgtmuF22(2)
          elseif(jwgtinfo.eq.2)then
            ii=iSorH_lhe+1
            if(ii.eq.3)ii=1
            read(ifile,404)wgtmuR2(ii),wgtmuF12(ii),wgtmuF22(ii)
            do i=1,mexternal
              read(ifile,405)(wgtkinE(j,i,iSorH_lhe),j=0,3)
            enddo
          elseif(jwgtinfo.eq.3 .or. jwgtinfo.eq.4)then
            do i=1,mexternal
              read(ifile,405)(wgtkinE(j,i,1),j=0,3)
            enddo
            do i=1,mexternal
              read(ifile,405)(wgtkinE(j,i,2),j=0,3)
            enddo
          endif
          read(ifile,441)wgtwreal(1),wgtwreal(2),
     #                   wgtwreal(3),wgtwreal(4)
          read(ifile,441)wgtwdeg(3),wgtwdeg(4),
     #                   wgtwdegmuf(3),wgtwdegmuf(4)
          read(ifile,405)wgtwborn(2),wgtwns(2),
     #                    wgtwnsmuf(2),wgtwnsmur(2)
          do i=1,iwgtnumpartn
            read(ifile,442)wgtwmcxsecE(i),
     #                     wgtmcxbjE(1,i),wgtmcxbjE(2,i)
          enddo
          if(jwgtinfo.eq.4) read(ifile,
     f      '(1x,e14.8,1x,e14.8,1x,i4,1x,i4)')
     &      wgtbpower,wgtcpower,nFKSprocess_used,nFKSprocess_used_born
          read(ifile,'(a)')string
        elseif(jwgtinfo.eq.5) then
           read(ifile,'(a)')string
           if (iSorH_lhe.eq.1) then ! S-event
              read(ifile,'(1x,e14.8,1x,e14.8,i4,i4)') 
     f             wgtbpower,wgtcpower,nScontributions
     $             ,i_process
              read(ifile,'(1x,i4,1x,e14.8)') nFKSprocess_used_born
     &             ,wgtref_nbody_all(i_process)
              do i=1,mexternal
                 read(ifile,405)(wgtkin_all(j,i,2,0),j=0,3)
              enddo
              read(ifile,402) wgtxbj_all(1,2,0),wgtxbj_all(2,2,0)
              read(ifile,'(1x,e14.8)') wgtqes2_all(2,0)
              read(ifile,405)wgtwborn_all,wgtwns_all,
     &             wgtwnsmuf_all,wgtwnsmur_all
              read(ifile,404) wgtmuR2_all(2,0),wgtmuF12_all(2,0)
     $             ,wgtmuF22_all(2,0)
              do ii=1,nScontributions
                 read(ifile,'(1x,i4)') nFKSprocess_reweight(ii)
                 iFKS=nFKSprocess_reweight(ii)*2-1
                 read(ifile,'(1x,e14.8,1x,i4)') wgtref_all(iFKS
     $                ,i_process),iwgtnumpartn_all(iFKS)
                 do i=1,mexternal
                    read(ifile,405)(wgtkin_all(j,i,1,iFKS),j=0,3)
                 enddo
                 do i=1,mexternal
                    do j=0,3
                       wgtkin_all(j,i,2,iFKS)=wgtkin_all(j,i,2,0)
                    enddo
                 enddo
                 read(ifile,402)
     &                wgtxbj_all(1,1,iFKS),wgtxbj_all(2,1,iFKS),
     &                wgtxbj_all(1,2,iFKS),wgtxbj_all(2,2,iFKS),
     &                wgtxbj_all(1,3,iFKS),wgtxbj_all(2,3,iFKS),
     &                wgtxbj_all(1,4,iFKS),wgtxbj_all(2,4,iFKS)
                 read(ifile,'(1x,e14.8)') wgtqes2_all(2,iFKS)
                 read(ifile,441)wgtwreal_all(1,iFKS),wgtwreal_all(2
     &                ,iFKS),wgtwreal_all(3,iFKS),wgtwreal_all(4,iFKS)
                 read(ifile,441)wgtwdeg_all(3,iFKS),wgtwdeg_all(4,iFKS)
     &                ,wgtwdegmuf_all(3,iFKS),wgtwdegmuf_all(4,iFKS)
                 do i=1,iwgtnumpartn_all(iFKS)
                    read(ifile,442)wgtwmcxsec_all(i,iFKS),
     &                   wgtmcxbj_all(1,i,iFKS),wgtmcxbj_all(2,i,iFKS)
                 enddo
                 read(ifile,404) wgtmuR2_all(1,iFKS),wgtmuF12_all(1
     $                ,iFKS),wgtmuF22_all(1,iFKS)
                 read(ifile,404) wgtmuR2_all(2,iFKS),wgtmuF12_all(2
     $                ,iFKS),wgtmuF22_all(2,iFKS)
              enddo
           elseif (iSorH_lhe.eq.2) then ! H-event
              read(ifile,'(1x,e14.8,1x,e14.8,i4)') 
     f             wgtbpower,wgtcpower,i_process
              read(ifile,'(1x,i4)') nFKSprocess_used
              iFKS=nFKSprocess_used*2
              read(ifile,'(1x,e14.8,1x,i4)') wgtref_all(iFKS,i_process)
     &             ,iwgtnumpartn_all(iFKS)
              do i=1,mexternal
                 read(ifile,405)(wgtkin_all(j,i,1,iFKS),j=0,3)
              enddo
              do i=1,mexternal
                 read(ifile,405)(wgtkin_all(j,i,2,iFKS),j=0,3)
              enddo
              read(ifile,402)
     &             wgtxbj_all(1,1,iFKS),wgtxbj_all(2,1,iFKS),
     &             wgtxbj_all(1,2,iFKS),wgtxbj_all(2,2,iFKS),
     &             wgtxbj_all(1,3,iFKS),wgtxbj_all(2,3,iFKS),
     &             wgtxbj_all(1,4,iFKS),wgtxbj_all(2,4,iFKS)
              read(ifile,441)wgtwreal_all(1,iFKS),wgtwreal_all(2
     &             ,iFKS),wgtwreal_all(3,iFKS),wgtwreal_all(4,iFKS)
              do i=1,iwgtnumpartn_all(iFKS)
                 read(ifile,442)wgtwmcxsec_all(i,iFKS),
     &                wgtmcxbj_all(1,i,iFKS),wgtmcxbj_all(2,i,iFKS)
              enddo
              read(ifile,404) wgtmuR2_all(1,iFKS),wgtmuF12_all(1,iFKS)
     $             ,wgtmuF22_all(1,iFKS)
              read(ifile,404) wgtmuR2_all(2,iFKS),wgtmuF12_all(2,iFKS)
     $             ,wgtmuF22_all(2,iFKS)
           else
              write (*,*) 'Not an S- or H-event in write_lhef_event'
              stop
           endif
           read(ifile,'(a)')string
         elseif(jwgtinfo.eq.15) then
           read(ifile,'(a)') string
           read(ifile,*)NUP_H
           do i=1,NUP_H
              read(ifile,*) IDUP_H(I),ISTUP_H(I),MOTHUP_H(1,I)
     $             ,MOTHUP_H(2,I),ICOLUP_H(1,I),ICOLUP_H(2,I),PUP_H(1
     $             ,I),PUP_H(2,I),PUP_H(3,I),PUP_H(4,I),PUP_H(5,I),
     $             VTIMUP_H(I),SPINUP_H(I)
           enddo
           read(ifile,'(a)') string
        elseif(jwgtinfo.eq.8)then
          read(ifile,'(a)')string
          read(ifile,406)wgtref,wgtxsecmu(1,1),numscales,numPDFpairs
          do i=1,numscales
            read(ifile,404)(wgtxsecmu(i,j),j=1,numscales)
          enddo
          do i=1,numPDFpairs
            nps=2*i-1
            nng=2*i
            read(ifile,404)wgtxsecPDF(nps),wgtxsecPDF(nng)
          enddo
          read(ifile,'(a)')string
        elseif(jwgtinfo.eq.9)then
           read(ifile,'(a)')string
           wgtref=XWGTUP
           do i=1,numscales
              do j=1,numscales
                call read_rwgt_line(ifile,idwgt,wgtxsecmu(i,j))
              enddo
           enddo
           do i=1,2*numPDFpairs
             call read_rwgt_line(ifile,idwgt,wgtxsecPDF(i))
           enddo
           if (numscales.eq.0 .and. numPDFpairs.ne.0) then
              wgtxsecmu(1,1)=XWGTUP
           endif
           read(ifile,'(a)')string
        endif
        read(ifile,'(a)')string
      else
        string=buff(1:len_trim(buff))
        buff=' '
      endif
 401  format(2(1x,e14.8))
 402  format(8(1x,e14.8))
 403  format(6(1x,e14.8))
 404  format(3(1x,e14.8))
 405  format(4(1x,e14.8))
 406  format(2(1x,e14.8),2(1x,i3))
 441  format(4(1x,e16.10))
 442  format(1x,e16.10,2(1x,e14.8))
 503  format(1x,i2,1x,i6,4(1x,e14.8))
 504  format(1x,i8,1x,i2,4(1x,i4),5(1x,e14.8),2(1x,e10.4))
c
      return
      end



      subroutine copy_header(infile,outfile,nevts)
      implicit none
      character*74 buff2
      integer nevts,infile,outfile
c
      buff2=' '
      do while(.true.)
         read(infile,'(a)')buff2
         if(index(buff2,'= nevents').eq.0)write(outfile,'(a)')buff2
         if(index(buff2,'= nevents').ne.0)exit
      enddo
      write(outfile,*)
     &nevts,' = nevents    ! Number of unweighted events requested'
      do while(index(buff2,'</header>').eq.0)
         read(infile,'(a)')buff2
         write(outfile,'(a)')buff2
      enddo
c
      return
      end


      subroutine fill_MC_mshell_wrap(MC,masses)
      double precision mcmass(-16:21),masses(-16:21)
      common/cmcmass/mcmass
      character*10 MonteCarlo,MC
      common/cMonteCarloType/MonteCarlo
      MonteCarlo=MC
      call case_trap4(10,MonteCarlo)
      call fill_MC_mshell()
      do i=-16,21
         masses(i)=mcmass(i)
      enddo
      return
      end


      function iistr(string)
c returns the position of the first non-blank character in string
c 
      implicit none
      logical is_i
      character*(*) string
      integer i,iistr
c
      is_i=.false.
      iistr=0
      do i=1,len(string)
         if(string(i:i).ne.' '.and..not.is_i)then
            is_i=.true.
            iistr=i
         endif
      enddo

      return
      end


      subroutine case_trap3(ilength,name)
c**********************************************************    
c change the string to lowercase if the input is not
c**********************************************************
      implicit none
c
c     ARGUMENT
c      
      character*(*) name
c
c     LOCAL
c
      integer i,k,ilength

      do i=1,ilength
         k=ichar(name(i:i))
         if(k.ge.65.and.k.le.90) then  !upper case A-Z
            k=ichar(name(i:i))+32   
            name(i:i)=char(k)        
         endif
      enddo

      return
      end


      subroutine case_trap4(ilength,name)
c**********************************************************    
c change the string to uppercase if the input is not
c**********************************************************
      implicit none
c
c     ARGUMENT
c      
      character*(*) name
c
c     LOCAL
c
      integer i,k,ilength

      do i=1,ilength
         k=ichar(name(i:i))
         if(k.ge.97.and.k.le.122) then  !lower case A-Z
            k=ichar(name(i:i))-32   
            name(i:i)=char(k)        
         endif
      enddo

      return
      end


      subroutine read_rwgt_line(unit,id,wgt)
c read a line in the <rwgt> tag. The syntax should be
c  <wgt id='1001'> 0.1234567e+01 </wgt>
c The id should be exactly 4 digits long.
      implicit none
      integer unit,id,wgt_start,id_start
      double precision wgt
      character*100 buff
      read (unit,'(a)') buff
c Use char() to make sure that the non-standard characters are compiler
c independent (char(62)=">", char(61)="=", char(39)="'")
      wgt_start=index(buff,CHAR(39)//CHAR(62))+2
      id_start=index(buff,'id'//CHAR(61)//CHAR(39))+4
      read (buff(id_start:100),'(i4)') id
      read (buff(wgt_start:100),*) wgt
      return
      end

