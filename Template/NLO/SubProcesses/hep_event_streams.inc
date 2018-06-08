C     Incoming event streams
C     ------------------------------
C     First the header HEPRUP
C     ------------------------------
      integer maxpup_in
      parameter(maxpup_in=100)
      integer idbmup_in,pdfgup_in,pdfsup_in,idwtup_in,nprup_in,lprup_in
      double precision ebmup_in,xsecup_in,xerrup_in,xmaxup_in
      common /heprup_in/ idbmup_in(2),ebmup_in(2),pdfgup_in(2),pdfsup_in(2),
     &     idwtup_in,nprup_in,xsecup_in(maxpup_in),xerrup_in(maxpup_in),
     &     xmaxup_in(maxpup_in),lprup_in(maxpup_in)
C     ------------------------------
C     Then an event itself in HEPEUP
C     ------------------------------
      INTEGER MAXNUP_IN
      PARAMETER (MAXNUP_IN=500)
      INTEGER NUP_IN,IDPRUP_IN,IDUP_IN(MAXNUP_IN),ISTUP_IN(MAXNUP_IN),
     &        MOTHUP_IN(2,MAXNUP_IN),ICOLUP_IN(2,MAXNUP_IN)
      DOUBLE PRECISION XWGTUP_IN,AQEDUP_IN,AQCDUP_IN, SCALUP_IN,
     &       PUP_IN(5,MAXNUP_IN),VTIMUP_IN(MAXNUP_IN),SPINUP_IN(MAXNUP_IN),
     &       SCALES_IN(2,MAXNUP_IN)

      common /hepeup_in/ nup_in, idprup_in, xwgtup_in, scalup_in,
     &                   aqedup_in, aqcdup_in, idup_in, istup_in,
     &                   mothup_in, icolup_in, pup_in, vtimup_in, spinup_in,
     &                   scales_in	 


C     Ougoing event streams
C     -----------------------
C     First the header HEPRUP
C     -----------------------
      integer maxpup_out
      parameter(maxpup_out=100)
      integer idbmup_out,pdfgup_out,pdfsup_out,idwtup_out,nprup_out,lprup_out
      double precision ebmup_out,xsecup_out,xerrup_out,xmaxup_out
      common /heprup_out/ idbmup_out(2),ebmup_out(2),pdfgup_out(2),pdfsup_out(2),
     &     idwtup_out,nprup_out,xsecup_out(maxpup_out),xerrup_out(maxpup_out),
     &     xmaxup_out(maxpup_out),lprup_out(maxpup_out)
C     ------------------------------
C     Then an event itself in HEPEUP
C     ------------------------------
      INTEGER MAXNUP_out
      PARAMETER (MAXNUP_out=500)
      INTEGER NUP_out,IDPRUP_out,IDUP_out(MAXNUP_out),ISTUP_out(MAXNUP_out),
     &        MOTHUP_out(2,MAXNUP_out),ICOLUP_out(2,MAXNUP_out)
      DOUBLE PRECISION XWGTUP_out,AQEDUP_out,AQCDUP_out, SCALUP_out,
     &       PUP_out(5,MAXNUP_out),VTIMUP_out(MAXNUP_out),SPINUP_out(MAXNUP_out),
     &       SCALES_out(2,MAXNUP_OUT),SCALUP_out_a(MAXNUP_OUT,MAXNUP_OUT)

      common /hepeup_out/ nup_out, idprup_out, xwgtup_out, scalup_out,
     &                   aqedup_out, aqcdup_out, idup_out, istup_out,
     &                   mothup_out, icolup_out, pup_out, vtimup_out, spinup_out,
     &                   scales_out, scalup_out_a
