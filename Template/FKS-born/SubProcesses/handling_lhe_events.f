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

      subroutine write_lhef_header(ifile,nevents,MonteCarlo)
      implicit none 
      integer ifile,nevents
      character*10 MonteCarlo
c
      write(ifile,'(a)')
     #     '<LesHouchesEvents version="1.0">'
      write(ifile,'(a)')
     #     '  <!--'
      write(ifile,'(a)')
     #     MonteCarlo
      write(ifile,'(a)')
     #     '  -->'
      write(ifile,'(a)')
     #     '  <header>'
      write(ifile,250)nevents
      write(ifile,'(a)')
     #     '  </header>'
 250  format(1x,i8)
      return
      end



      subroutine write_lhef_header_string(ifile,string,MonteCarlo)
      implicit none 
      integer ifile
      character*10 MonteCarlo,string
c
      write(ifile,'(a)')
     #     '<LesHouchesEvents version="1.0">'
      write(ifile,'(a)')
     #     '  <!--'
      write(ifile,'(a)')
     #     MonteCarlo
      write(ifile,'(a)')
     #     '  -->'
      write(ifile,'(a)')
     #     '  <header>'
      write(ifile,'(a)')string
      write(ifile,'(a)')
     #     '  </header>'
      return
      end



      subroutine read_lhef_header(ifile,nevents,MonteCarlo)
      implicit none 
      integer ifile,nevents
      character*10 MonteCarlo
      character*80 string,string0
c
      string='  '
      dowhile(string.ne.'  -->')
        string0=string
        read(ifile,'(a)')string
      enddo
c Works only if the name of the MC is the last line of the comments
      MonteCarlo=string0(1:10)
c Here we are at the end of (user-defined) comments. Now go to end
c of headers
      dowhile(string.ne.'  </header>')
        string0=string
        read(ifile,'(a)')string
      enddo
      read(string0,250)nevents
 250  format(1x,i8)
      return
      end


      subroutine write_lhef_init(ifile,
     #  IDBMUP,EBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,
     #  XSECUP,XERRUP,XMAXUP,LPRUP)
      implicit none
      integer ifile,IDBMUP(2),PDFGUP(2),PDFSUP(2),IDWTUP,NPRUP,LPRUP
      double precision EBMUP(2),XSECUP,XERRUP,XMAXUP
c
      write(ifile,'(a)')
     # '  <init>'
      write(ifile,501)IDBMUP(1),IDBMUP(2),EBMUP(1),EBMUP(2),
     #                PDFGUP(1),PDFGUP(2),PDFSUP(1),PDFSUP(2),
     #                IDWTUP,NPRUP
      write(ifile,502)XSECUP,XERRUP,XMAXUP,LPRUP
      write(ifile,'(a)')
     # '  </init>'
 501  format(2(1x,i6),2(1x,d14.8),2(1x,i2),2(1x,i6),1x,i2,1x,i3)
 502  format(3(1x,d14.8),1x,i6)
c
      return
      end


      subroutine read_lhef_init(ifile,
     #  IDBMUP,EBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,
     #  XSECUP,XERRUP,XMAXUP,LPRUP)
      implicit none
      integer ifile,IDBMUP(2),PDFGUP(2),PDFSUP(2),IDWTUP,NPRUP,LPRUP
      double precision EBMUP(2),XSECUP,XERRUP,XMAXUP
      character*80 string
c
      read(ifile,'(a)')string
      read(ifile,501)IDBMUP(1),IDBMUP(2),EBMUP(1),EBMUP(2),
     #                PDFGUP(1),PDFGUP(2),PDFSUP(1),PDFSUP(2),
     #                IDWTUP,NPRUP
      read(ifile,502)XSECUP,XERRUP,XMAXUP,LPRUP
      read(ifile,'(a)')string
 501  format(2(1x,i6),2(1x,d14.8),2(1x,i2),2(1x,i6),1x,i2,1x,i3)
 502  format(3(1x,d14.8),1x,i6)
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
      character*1 ch1
      integer isorh_lhe,ifks_lhe,jfks_lhe,fksfather_lhe,ipartner_lhe
      double precision scale1_lhe,scale2_lhe
      integer ii,j,nps,nng
      double precision wgtcentral,wgtmumin,wgtmumax,wgtpdfmin,wgtpdfmax
      include 'reweight0.inc'
c
      write(ifile,'(a)')
     # '  <event>'
      write(ifile,503)NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP
      do i=1,nup
        write(ifile,504)IDUP(I),ISTUP(I),MOTHUP(1,I),MOTHUP(2,I),
     #                  ICOLUP(1,I),ICOLUP(2,I),
     #                  PUP(1,I),PUP(2,I),PUP(3,I),PUP(4,I),PUP(5,I),
     #                  VTIMUP(I),SPINUP(I)
      enddo
      if(buff(1:1).eq.'#') then
        write(ifile,'(a)') buff(1:len_trim(buff))
        read(buff,200)ch1,iSorH_lhe,ifks_lhe,jfks_lhe,
     #                    fksfather_lhe,ipartner_lhe,
     #                    scale1_lhe,scale2_lhe,
     #                    jwgtinfo,mexternal,iwgtnumpartn,
     #         wgtcentral,wgtmumin,wgtmumax,wgtpdfmin,wgtpdfmax
        if(jwgtinfo.ge.1.and.jwgtinfo.le.4)then
          write(ifile,'(a)')
     # '  <rwgt>'
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
          if(jwgtinfo.eq.4) write(ifile,'(1x,d14.8,1x,i4,1x,i4)')
     &         wgtbpower,nFKSprocess_used,nFKSprocess_used_born
          write(ifile,'(a)')
     # '  </rwgt>'
        elseif(jwgtinfo.eq.8)then
          write(ifile,'(a)')
     # '  <rwgt>'
          write(ifile,406)wgtref,wgtxsecmu(1,1),numscales,numPDFpairs
          do i=1,numscales
            write(ifile,404)(wgtxsecmu(i,j),j=1,numscales)
          enddo
          do i=1,numPDFpairs
            nps=2*i-1
            nng=2*i
            write(ifile,404)wgtxsecPDF(nps),wgtxsecPDF(nng)
          enddo
          write(ifile,'(a)')
     # '  </rwgt>'
        endif
      endif
      write(ifile,'(a)')
     # '  </event>'
 200  format(1a,1x,i1,4(1x,i2),2(1x,d14.8),1x,i1,2(1x,i2),5(1x,d14.8))
 401  format(2(1x,d14.8))
 402  format(8(1x,d14.8))
 403  format(6(1x,d14.8))
 404  format(3(1x,d14.8))
 405  format(4(1x,d14.8))
 406  format(2(1x,d14.8),2(1x,i3))
 441  format(4(1x,d16.10))
 442  format(1x,d16.10,2(1x,d14.8))
 503  format(1x,i2,1x,i6,4(1x,d14.8))
 504  format(1x,i8,1x,i2,4(1x,i4),5(1x,d14.8),2(1x,d10.4))
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
      character*1 ch1
      integer isorh_lhe,ifks_lhe,jfks_lhe,fksfather_lhe,ipartner_lhe
      double precision scale1_lhe,scale2_lhe
      integer ii,j,nps,nng
      double precision wgtcentral,wgtmumin,wgtmumax,wgtpdfmin,wgtpdfmax
      include 'reweight0.inc'
c
      read(ifile,'(a)')string
      read(ifile,*)NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP
      do i=1,nup
        read(ifile,*)IDUP(I),ISTUP(I),MOTHUP(1,I),MOTHUP(2,I),
     #                  ICOLUP(1,I),ICOLUP(2,I),
     #                  PUP(1,I),PUP(2,I),PUP(3,I),PUP(4,I),PUP(5,I),
     #                  VTIMUP(I),SPINUP(I)
      enddo
      read(ifile,'(a)')buff
      if(buff(1:1).eq.'#')then
        read(buff,200)ch1,iSorH_lhe,ifks_lhe,jfks_lhe,
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
          if(jwgtinfo.eq.4) read(ifile,'(1x,d14.8,1x,i4,1x,i4)')
     &         wgtbpower,nFKSprocess_used,nFKSprocess_used_born
          read(ifile,'(a)')string
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
        endif
        read(ifile,'(a)')string
      else
        string=buff(1:len_trim(buff))
        buff=' '
      endif
 200  format(1a,1x,i1,4(1x,i2),2(1x,d14.8),1x,i1,2(1x,i2),5(1x,d14.8))
 401  format(2(1x,d14.8))
 402  format(8(1x,d14.8))
 403  format(6(1x,d14.8))
 404  format(3(1x,d14.8))
 405  format(4(1x,d14.8))
 406  format(2(1x,d14.8),2(1x,i3))
 441  format(4(1x,d16.10))
 442  format(1x,d16.10,2(1x,d14.8))
 503  format(1x,i2,1x,i6,4(1x,d14.8))
 504  format(1x,i8,1x,i2,4(1x,i4),5(1x,d14.8),2(1x,d10.4))
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
      character*1 ch1
      integer isorh_lhe,ifks_lhe,jfks_lhe,fksfather_lhe,ipartner_lhe
      double precision scale1_lhe,scale2_lhe
      integer ii,j,nps,nng
      double precision wgtcentral,wgtmumin,wgtmumax,wgtpdfmin,wgtpdfmax
      include 'reweight0.inc'
c
      read(ifile,'(a)')string
      if(index(string,'<event>').eq.0)then
        if(index(string,'</LesHouchesEvents>').ne.0)then
          buff='endoffile'
          return
        else
          write(*,*)'Unknown structure in read_lhef_event_catch:'
          write(*,*)string(1:len_trim(string))
          stop
        endif
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
        read(buff,200)ch1,iSorH_lhe,ifks_lhe,jfks_lhe,
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
          if(jwgtinfo.eq.4) read(ifile,'(1x,d14.8,1x,i4,1x,i4)')
     &         wgtbpower,nFKSprocess_used,nFKSprocess_used_born
          read(ifile,'(a)')string
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
        endif
        read(ifile,'(a)')string
      else
        string=buff(1:len_trim(buff))
        buff=' '
      endif
 200  format(1a,1x,i1,4(1x,i2),2(1x,d14.8),1x,i1,2(1x,i2),5(1x,d14.8))
 401  format(2(1x,d14.8))
 402  format(8(1x,d14.8))
 403  format(6(1x,d14.8))
 404  format(3(1x,d14.8))
 405  format(4(1x,d14.8))
 406  format(2(1x,d14.8),2(1x,i3))
 441  format(4(1x,d16.10))
 442  format(1x,d16.10,2(1x,d14.8))
 503  format(1x,i2,1x,i6,4(1x,d14.8))
 504  format(1x,i8,1x,i2,4(1x,i4),5(1x,d14.8),2(1x,d10.4))
c
      return
      end
