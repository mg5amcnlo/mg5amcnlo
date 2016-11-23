c
c Example analysis for "p p > V [QCD]" process.
c
c It features the HwU format for histogram booking and output.
c The details of how to process/manipulate the resulting .HwU file,
c in particular how to plot it using gnuplot, I refer the reader to this
c FAQ:
c
c      https://answers.launchpad.net/mg5amcnlo/+faq/2671
c
c It mostly relies on using the following madgraph5 module in standalone
c
c  <MG5_aMC_install_dir>/madgraph/various/histograms.py
c
c You can learn about how to run it and what options are available with
c
c  python <MG5_aMC_install_dir>/madgraph/various/histograms.py --help
c
C----------------------------------------------------------------------
      SUBROUTINE RCLOS()
C     DUMMY IF HBOOK IS USED
C----------------------------------------------------------------------
      END


C----------------------------------------------------------------------
      SUBROUTINE HWABEG
C     USER''S ROUTINE FOR INITIALIZATION
C----------------------------------------------------------------------
      INCLUDE 'HERWIG65.INC'
      include 'reweight0.inc'
      real * 8 xm0,gam,xmlow,xmupp,bin
      real * 8 xmi,xms,pi
      PARAMETER (PI=3.14159265358979312D0)
c
c     The type suffix of the histogram title, with syntax 
c     |T@<type_name> is semantic in the HwU format. It allows for
c     various filtering when using the histogram.py module
c     (see comment at the beginning of this file).
c     It is in general a good idea to keep the same title for the
c     same observable (if they use the same range) and differentiate
c     them only using the type suffix.
c
      character*8 HwUtype(2)
      data HwUtype/'|T@NOCUT','|T@CUT  '/
      integer j,kk,l,jpr,i
      character*5 cc(2)
      data cc/'     ','Born '/
      integer nwgt,max_weight,nwgt_analysis
      common/cnwgt/nwgt
      common/c_analysis/nwgt_analysis
      character*50 weights_info(max_weight_shower)
      common/cwgtsinfo/weights_info
c Initialize histograms
      call HwU_inithist(nwgt,weights_info)
c Set method for error estimation to '0', i.e., use Poisson statistics
c for the uncertainty estimate
      call set_error_estimation(0)
      nwgt_analysis=nwgt
      do i=1,2
       l=(i-1)*5
       call HwU_book(l+1,'V pt            '//HwUtype(i),100,0.d0,200.d0)
       call HwU_book(l+2,'V log[pt]       '//HwUtype(i),98,0.1d0,5.d0)
       call HwU_book(l+3,'V y             '//HwUtype(i),40,-9.d0,9.d0)
       call HwU_book(l+4,'V eta           '//HwUtype(i),40,-9.d0,9.d0)
       call HwU_book(l+5,'mV              '//HwUtype(i),80,40.d0,120.d0)
      enddo

      END

C----------------------------------------------------------------------
      SUBROUTINE HWAEND
C     USER''S ROUTINE FOR TERMINAL CALCULATIONS, HISTOGRAM OUTPUT, ETC
C----------------------------------------------------------------------
      INCLUDE 'HERWIG65.INC'
      REAL*8 XNORM
      INTEGER I,J,KK,l,nwgt_analysis
      integer NPL
      parameter(NPL=15000)
      common/c_analysis/nwgt_analysis
c Convert from nb to pb using xnorm. This assumes that event weights
c *average* to the total cross section, so no extra weight needed
      xnorm=1d3
c Collect accumulated results
      call finalize_histograms(nevhep)
c Write the histograms to disk. 
      open (unit=99,file='MADatNLO.HwU',status='unknown')
      call HwU_output(99,xnorm)
      close (99)
      return
      END

C----------------------------------------------------------------------
      SUBROUTINE HWANAL
C     USER''S ROUTINE TO ANALYSE DATA FROM EVENT
C----------------------------------------------------------------------
      INCLUDE 'HERWIG65.INC'
      include 'reweight0.inc'
      DOUBLE PRECISION HWVDOT,PSUM(4),XME,PPV(5),PPE(5),PPNU(5),
     # PPDCE(5),PPDCNU(5),WT,ETAEMIN(2),ETAEMAX(2),PTEMIN(2),
     # XMV,PTV,YV,GETRAPIDITY,PTE,THE,ETAE,PTNU,THNU,ETANU,
     # PTDCE,THDCE,ETADCE,PTDCNU,THDCNU,ETADCNU,ETAV,GETPSEUDORAP
      INTEGER ICHSUM,ICHINI,IHEP,JPR,IDENT,IFV,IST,ID,ID1,IHRD,IV,
     # IJ,IE,INU,J
      LOGICAL DIDSOF,TEST1,TEST2,FLAG
      REAL*8 PI
      PARAMETER (PI=3.14159265358979312D0)
      REAL*8 TINY
      INTEGER KK,i,l
      DATA TINY/.1D-5/
      DATA XME/5.11D-4/
      integer nwgt_analysis,max_weight
      common/c_analysis/nwgt_analysis
      integer maxRWGT
      parameter (maxRWGT=100)
      double precision wgtxsecRWGT(maxRWGT)
      parameter (max_weight=maxscales*maxscales+maxpdfs+maxRWGT+1)
      double precision ww(max_weight),www(max_weight)
      common/cww/ww
c
      IF (IERROR.NE.0) RETURN
      IF (WW(1).EQ.0D0) THEN
         WRITE(*,*)'WW(1) = 0. Stopping'
         STOP
      ENDIF
c
c CHOOSE IDENT=24 FOR W+, IDENT=-24 FOR W-, IDENT=23 FOR Z0
      IDENT=23
C INCOMING PARTONS MAY TRAVEL IN THE SAME DIRECTION: IT''S A POWER-SUPPRESSED
C EFFECT, SO THROW THE EVENT AWAY
      IF(SIGN(1.D0,PHEP(3,4)).EQ.SIGN(1.D0,PHEP(3,5)))THEN
        CALL HWWARN('HWANAL',111)
        GOTO 999
      ENDIF
      DO I=1,nwgt_analysis
         WWW(I)=EVWGT*ww(i)/ww(1)
      ENDDO
      CALL HWVSUM(4,PHEP(1,1),PHEP(1,2),PSUM)
      CALL HWVSCA(4,-1D0,PSUM,PSUM)
      ICHSUM=0
      ICHINI=ICHRG(IDHW(1))+ICHRG(IDHW(2))
      DIDSOF=.FALSE.
      IFV=0
      DO 100 IHEP=1,NHEP
        IF (IDHW(IHEP).EQ.16) DIDSOF=.TRUE.
        IF (ISTHEP(IHEP).EQ.1) THEN
          CALL HWVSUM(4,PHEP(1,IHEP),PSUM,PSUM)
          ICHSUM=ICHSUM+ICHRG(IDHW(IHEP))
        ENDIF
        IST=ISTHEP(IHEP)      
        ID=IDHW(IHEP)
        ID1=IDHEP(IHEP)
C Herwig relabels the vector boson V in Drell-Yan; this doesn''t happen with
C MC@NLO; in S events, V appears as HARD, in H as V, but with status 155
C rather than 195. We add here 195 for future compatibility
        IF(IPROC.LT.0)THEN
          TEST1=IST.EQ.155.OR.IST.EQ.195
          TEST2=ID1.EQ.IDENT
          IF(IST.EQ.120.AND.ID1.EQ.0)IHRD=IHEP
        ELSE
          TEST1=IST.EQ.120.OR.IST.EQ.195
          TEST2=ABS(ID1).EQ.IDENT
        ENDIF
        IF(TEST1.AND.TEST2)THEN
          IV=IHEP
          IFV=IFV+1
        ENDIF
  100 CONTINUE
      IF(IPROC.LT.0.AND.IFV.EQ.0)THEN
        IV=IHRD
        IFV=1
      ENDIF
      DO IJ=1,5
        PPV(IJ)=PHEP(IJ,IV)
      ENDDO
      IF(IFV.EQ.0.AND.IERROR.EQ.0) THEN
         CALL HWUEPR
         CALL HWWARN('HWANAL',503)
      ENDIF
C CHECK MOMENTUM AND CHARGE CONSERVATION
      IF (HWVDOT(3,PSUM,PSUM).GT.1.E-4*PHEP(4,1)**2) THEN
         CALL HWUEPR
         CALL HWWARN('HWANAL',112)
         GOTO 999
      ENDIF
      IF (ICHSUM.NE.ICHINI) THEN
         CALL HWUEPR
         CALL HWWARN('HWANAL',113)
         GOTO 999
      ENDIF
      IF(IFV.GT.1.AND.IERROR.EQ.0) THEN
         CALL HWUEPR
         CALL HWWARN('HWANAL',55)
         GOTO 999
      ENDIF
C FILL THE HISTOS
C Variables of the vector boson
      xmv=ppv(5)
      ptv=sqrt(ppv(1)**2+ppv(2)**2)
      yv=getrapidity(ppv(4),ppv(3))
      etav=getpseudorap(ppv(4),ppv(1),ppv(2),ppv(3))
C
      do i=1,2
      l=(i-1)*5
      call HwU_fill(l+1,ptv,WWW)
      if(ptv.gt.0) call HwU_fill(l+2,log10(ptv),WWW)
      call HwU_fill(l+3,yv,WWW)
      call HwU_fill(l+4,etav,WWW)
      call HwU_fill(l+5,xmv,WWW)
      enddo
      call HwU_add_points
C
 999  END


      function getrapidity(en,pl)
      implicit none
      real*8 getrapidity,en,pl,tiny,xplus,xminus,y
      parameter (tiny=1.d-8)
      xplus=en+pl
      xminus=en-pl
      if(xplus.gt.tiny.and.xminus.gt.tiny)then
         if( (xplus/xminus).gt.tiny.and.(xminus/xplus).gt.tiny)then
            y=0.5d0*log( xplus/xminus  )
         else
            y=sign(1.d0,pl)*1.d8
         endif
      else 
         y=sign(1.d0,pl)*1.d8
      endif
      getrapidity=y
      return
      end


      function getpseudorap(en,ptx,pty,pl)
      implicit none
      real*8 getpseudorap,en,ptx,pty,pl,tiny,pt,eta,th
      parameter (tiny=1.d-5)
c
      pt=sqrt(ptx**2+pty**2)
      if(pt.lt.tiny.and.abs(pl).lt.tiny)then
        eta=sign(1.d0,pl)*1.d8
      else
        th=atan2(pt,pl)
        eta=-log(tan(th/2.d0))
      endif
      getpseudorap=eta
      return
      end
