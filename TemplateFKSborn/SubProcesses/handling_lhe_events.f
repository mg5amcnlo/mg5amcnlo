c Utility routines for LHEF. Originally taken from collect_events.f
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
      integer ifile,IDBMUP(2),PDFGUP(2),PDFSUP(2),IDWTUP,NPRUP
      double precision EBMUP(2),XSECUP,XERRUP,XMAXUP,LPRUP
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
      integer ifile,IDBMUP(2),PDFGUP(2),PDFSUP(2),IDWTUP,NPRUP
      double precision EBMUP(2),XSECUP,XERRUP,XMAXUP,LPRUP
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
      if(buff(1:1).eq.'#') write(ifile,'(a)') buff(1:len_trim(buff))
      write(ifile,'(a)')
     # '  </event>'
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
        read(ifile,'(a)')string
      else
        string=buff(1:len_trim(buff))
        buff=' '
      endif
 503  format(1x,i2,1x,i6,4(1x,d14.8))
 504  format(1x,i8,1x,i2,4(1x,i4),5(1x,d14.8),2(1x,d10.4))
c
      return
      end
