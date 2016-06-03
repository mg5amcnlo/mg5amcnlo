c
c Example analysis for "p p > t t~ [QCD]" process.
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
      SUBROUTINE PYABEG(nnn,wwwi)
C     USER''S ROUTINE FOR INITIALIZATION
C----------------------------------------------------------------------
      INCLUDE 'HEPMC.INC'
      include 'reweight0.inc'
      REAL*8 pi
      PARAMETER (PI=3.14159265358979312D0)
      integer j,kk,l,i,nnn
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
      integer nwgt_analysis
      common/c_analysis/nwgt_analysis
      character*50 weights_info(max_weight_shower)
     $     ,wwwi(max_weight_shower)
c
      do i=1,nnn
         weights_info(i)=wwwi(i)
      enddo
      nwgt=nnn
c Initialize histograms
      call HwU_inithist(nwgt,weights_info)
c Set method for error estimation to '0', i.e., use Poisson statistics
c for the uncertainty estimate
      call set_error_estimation(0)
      nwgt_analysis=nwgt
      do i=1,2
       l=(i-1)*20
       call HwU_book(l+ 1,'tt pt            '//HwUtype(i),
     &                                                 50,0.d0,100.d0)
       call HwU_book(l+ 2,'tt log[pt]       '//HwUtype(i),98,0.1d0,5.d0)
       call HwU_book(l+ 3,'tt inv m         '//HwUtype(i),60,300.d0,1000.d0)
       call HwU_book(l+ 4,'tt azimt         '//HwUtype(i),20,0.d0,pi)
       call HwU_book(l+ 5,'tt del R         '//HwUtype(i),60,0.d0,3*pi)
       call HwU_book(l+ 6,'tb pt            '//HwUtype(i),
     &                                                 100,0.d0,500.d0)
       call HwU_book(l+ 7,'tb log[pt]       '//HwUtype(i),98,0.1d0,5.d0)
       call HwU_book(l+ 8,'t pt             '//HwUtype(i),
     &                                                 100,0.d0,500.d0)
       call HwU_book(l+ 9,'t log[pt]        '//HwUtype(i),98,0.1d0,5.d0)
       call HwU_book(l+10,'tt delta eta     '//HwUtype(i),40,-4.d0,4.d0)
       call HwU_book(l+11,'y_tt             '//HwUtype(i),80,-4.d0,4.d0)
       call HwU_book(l+12,'delta y          '//HwUtype(i),40,-4.d0,4.d0)
       call HwU_book(l+13,'tt azimt zoomin  '//HwUtype(i),20,2*pi/3,pi)
       call HwU_book(l+14,'tt del R zoomin  '//HwUtype(i),
     &                                                 40,2*pi/3,4*pi/3)
       call HwU_book(l+15,'y_tb             '//HwUtype(i),80,-4.d0,4.d0)
       call HwU_book(l+16,'y_t              '//HwUtype(i),80,-4.d0,4.d0)
       call HwU_book(l+17,'tt log[pi-azimt] '//HwUtype(i),
     &                                                   82,-4.d0,0.1d0)
       call HwU_book(l+18,'tt pt zoomout    '//HwUtype(i),96,80.d0,2000.d0)
       call HwU_book(l+19,'tb pt zoomout    '//HwUtype(i),100,400.d0,2400.d0)
       call HwU_book(l+20,'t pt  zoomout    '//HwUtype(i),100,400.d0,2400.d0)
      enddo
 999  END

C----------------------------------------------------------------------
      SUBROUTINE PYAEND(IEVTTOT)
C     USER''S ROUTINE FOR TERMINAL CALCULATIONS, HISTOGRAM OUTPUT, ETC
C----------------------------------------------------------------------
      INCLUDE 'HEPMC.INC'
      REAL*8 XNORM,IEVTTOT
      INTEGER I,J,KK,l,nwgt_analysis
      integer NPL
      parameter(NPL=15000)
      common/c_analysis/nwgt_analysis
c Collect accumulated results. IEVTTOT is such that we need to multiply
c the results by this factor
      xnorm=ievttot
      call finalize_histograms(nevhep)
c Write the histograms to disk. 
      open (unit=99,file='MADatNLO.HwU',status='unknown')
      call HwU_output(99,xnorm)
      close (99)
      END

C----------------------------------------------------------------------
      SUBROUTINE PYANAL(nnn,xww)
C     USER''S ROUTINE TO ANALYSE DATA FROM EVENT
C----------------------------------------------------------------------
      INCLUDE 'HEPMC.INC'
      include 'reweight0.inc'
      DOUBLE PRECISION HWVDOT,PSUM(4)
      INTEGER ICHSUM,ICHINI,IHEP
      LOGICAL DIDSOF,flcuts,siq1flag,siq2flag,ddflag
      INTEGER ID,ID1,IST,IQ1,IQ2,IT1,IT2,ILP,INU,IBQ,ILM,INB,IBB,IJ
      DOUBLE PRECISION YCUT,PTCUT,ptlp,ylp,getrapidity,ptnu,ynu,
     # ptbq,ybq,ptlm,ylm,ptnb,ynb,ptbb,ybb,ptbqbb,dphibqbb,
     # getdelphi,xmbqbb,getinvm,ptlplm,dphilplm,xmlplm,ptbqlm,
     # dphibqlm,xmbqlm,ptbblp,dphibblp,xmbblp,ptbqnb,dphibqnb,
     # xmbqnb,ptbbnu,dphibbnu,xmbbnu,ptq1,ptq2,ptg,yq1,yq2,
     # etaq1,getpseudorap,etaq2,azi,azinorm,qqm,dr,yqq
      DOUBLE PRECISION XPTQ(5),XPTB(5),XPLP(5),XPNU(5),XPBQ(5),XPLM(5),
     # XPNB(5),XPBB(5)
      DOUBLE PRECISION YPBQBB(4),YPLPLM(4),YPBQLM(4),YPBBLP(4),
     # YPBQNB(4),YPBBNU(4),YPTQTB(4)
      REAL*8 PI
      PARAMETER (PI=3.14159265358979312D0)
      REAL*8 WWW0
      INTEGER KK,IVLEP1,IVLEP2,i,l
      COMMON/VVLIN/IVLEP1,IVLEP2
      integer nwgt_analysis,max_weight
      common/c_analysis/nwgt_analysis
      integer maxRWGT
      parameter (maxRWGT=100)
      double precision wgtxsecRWGT(maxRWGT)
      parameter (max_weight=maxscales*maxscales+maxpdfs+maxRWGT+1)
      double precision ww(max_weight),www(max_weight),xww(max_weight)
      common/cww/ww
c
      if(nnn.eq.0)ww(1)=1d0
      do i=1,nnn
         ww(i)=xww(i)
      enddo
c
      IF (WW(1).EQ.0D0) THEN
         WRITE(*,*)'WW(1) = 0. Stopping'
         STOP
      ENDIF
C INCOMING PARTONS MAY TRAVEL IN THE SAME DIRECTION: IT''S A POWER-SUPPRESSED
C EFFECT, SO THROW THE EVENT AWAY
      IF(SIGN(1.D0,PHEP(3,1)).EQ.SIGN(1.D0,PHEP(3,2)))THEN
         WRITE(*,*)'WARNING 111 IN PYANAL'
         GOTO 999
      ENDIF
      DO I=1,nwgt_analysis
         WWW(I)=EVWGT*ww(i)/ww(1)
      ENDDO
      ICHSUM=0
      DIDSOF=.FALSE.
      IQ1=0
      IQ2=0
      DO 100 IHEP=1,NHEP
C UNCOMMENT THE FOLLOWING WHEN REMOVING THE CHECK ON MOMENTUM 
C        IF(IQ1*IQ2.EQ.1) GOTO 11
        IST=ISTHEP(IHEP)      
        ID1=IDHEP(IHEP)
        IF(ID1.EQ.6)THEN
C FOUND A TOP; KEEP ONLY THE FIRST ON RECORD
          IQ1=IQ1+1
          IT1=IHEP
        ELSEIF(ID1.EQ.-6)THEN
C FOUND AN ANTITOP; KEEP ONLY THE FIRST ON RECORD
          IQ2=IQ2+1
          IT2=IHEP
        ENDIF
  100 CONTINUE
      IF(IQ1*IQ2.EQ.0)THEN
        WRITE(*,*)'ERROR 501 IN PYANAL'
      ENDIF
      DO IJ=1,5
         XPTQ(IJ)=PHEP(IJ,IT1)
         XPTB(IJ)=PHEP(IJ,IT2)
      ENDDO
      DO IJ=1,4
         YPTQTB(IJ)=XPTQ(IJ)+XPTB(IJ)
      ENDDO
C FILL THE HISTOS
      YCUT=2.5D0
      PTCUT=30.D0
C
      ptq1 = dsqrt(xptq(1)**2+xptq(2)**2)
      ptq2 = dsqrt(xptb(1)**2+xptb(2)**2)
      ptg = dsqrt(yptqtb(1)**2+yptqtb(2)**2)
      yq1=getrapidity(xptq(4),xptq(3))
      yq2=getrapidity(xptb(4),xptb(3))
      etaq1=getpseudorap(xptq(4),xptq(1),xptq(2),xptq(3))
      etaq2=getpseudorap(xptb(4),xptb(1),xptb(2),xptb(3))
      azi=getdelphi(xptq(1),xptq(2),xptb(1),xptb(2))
      azinorm = (pi-azi)/pi
      qqm=getinvm(yptqtb(4),yptqtb(1),yptqtb(2),yptqtb(3))
      dr  = dsqrt(azi**2+(etaq1-etaq2)**2)
      yqq=getrapidity(yptqtb(4),yptqtb(3))
c-------------------------------------------------------------
      siq1flag=ptq1.gt.ptcut.and.abs(yq1).lt.ycut
      siq2flag=ptq2.gt.ptcut.and.abs(yq2).lt.ycut
      ddflag=siq1flag.and.siq2flag
c-------------------------------------------------------------
      l=0
      call HwU_fill(l+1,ptg,WWW)
      call HwU_fill(l+18,ptg,WWW)
      if(ptg.gt.0) call HwU_fill(l+2,log10(ptg),WWW)
      call HwU_fill(l+3,qqm,WWW)
      call HwU_fill(l+4,azi,WWW)
      call HwU_fill(l+13,azi,WWW)
      if(azinorm.gt.0)call HwU_fill(l+17,log10(azinorm),WWW)
      call HwU_fill(l+5,dr,WWW)
      call HwU_fill(l+14,dr,WWW)
      call HwU_fill(l+10,etaq1-etaq2,WWW)
      call HwU_fill(l+11,yqq,WWW)
      call HwU_fill(l+12,yq1-yq2,WWW)
      call HwU_fill(l+6,ptq2,WWW)
      call HwU_fill(l+19,ptq2,WWW)
      if(ptq2.gt.0) call HwU_fill(l+7,log10(ptq2),WWW)
      call HwU_fill(l+15,yq2,WWW)
      call HwU_fill(l+8,ptq1,WWW)
      call HwU_fill(l+20,ptq1,WWW)
      if(ptq1.gt.0) call HwU_fill(l+9,log10(ptq1),WWW)
      call HwU_fill(l+16,yq1,WWW)
c
c***************************************************** with cuts
c
      l=l+20
c
      if(ddflag)then
         call HwU_fill(l+1,ptg,WWW)
         call HwU_fill(l+18,ptg,WWW)
         if(ptg.gt.0) call HwU_fill(l+2,log10(ptg),WWW)
         call HwU_fill(l+3,qqm,WWW)
         call HwU_fill(l+4,azi,WWW)
         call HwU_fill(l+13,azi,WWW)
         if(azinorm.gt.0) call HwU_fill(l+17,log10(azinorm),WWW)
         call HwU_fill(l+5,dr,WWW)
         call HwU_fill(l+14,dr,WWW)
         call HwU_fill(l+10,etaq1-etaq2,WWW)
         call HwU_fill(l+11,yqq,WWW)
         call HwU_fill(l+12,yq1-yq2,WWW)
      endif
      if(abs(yq2).lt.ycut)then
         call HwU_fill(l+6,ptq2,WWW)
         call HwU_fill(l+19,ptq2,WWW)
         if(ptq2.gt.0) call HwU_fill(l+7,log10(ptq2),WWW)
      endif
      if(ptq2.gt.ptcut)call HwU_fill(l+15,yq2,WWW)
      if(abs(yq1).lt.ycut)then
         call HwU_fill(l+8,ptq1,WWW)
         call HwU_fill(l+20,ptq1,WWW)
         if(ptq1.gt.0) call HwU_fill(l+9,log10(ptq1),WWW)
      endif
      if(ptq1.gt.ptcut)call HwU_fill(l+16,yq1,WWW)
      call HwU_add_points
 999  return
      end


      function getrapidity(en,pl)
      implicit none
      real*8 getrapidity,en,pl,tiny,xplus,xminus,y
      parameter (tiny=1.d-8)
c
      xplus=en+pl
      xminus=en-pl
      if(xplus.gt.tiny.and.xminus.gt.tiny)then
        if( (xplus/xminus).gt.tiny )then
          y=0.5d0*log( xplus/xminus )
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


      function getinvm(en,ptx,pty,pl)
      implicit none
      real*8 getinvm,en,ptx,pty,pl,tiny,tmp
      parameter (tiny=1.d-5)
c
      tmp=en**2-ptx**2-pty**2-pl**2
      if(tmp.gt.0.d0)then
        tmp=sqrt(tmp)
      elseif(tmp.gt.-tiny)then
        tmp=0.d0
      else
        write(*,*)'Attempt to compute a negative mass'
        stop
      endif
      getinvm=tmp
      return
      end


      function getdelphi(ptx1,pty1,ptx2,pty2)
      implicit none
      real*8 getdelphi,ptx1,pty1,ptx2,pty2,tiny,pt1,pt2,tmp
      parameter (tiny=1.d-5)
c
      pt1=sqrt(ptx1**2+pty1**2)
      pt2=sqrt(ptx2**2+pty2**2)
      if(pt1.ne.0.d0.and.pt2.ne.0.d0)then
        tmp=ptx1*ptx2+pty1*pty2
        tmp=tmp/(pt1*pt2)
        if(abs(tmp).gt.1.d0+tiny)then
          write(*,*)'Cosine larger than 1'
          stop
        elseif(abs(tmp).ge.1.d0)then
          tmp=sign(1.d0,tmp)
        endif
        tmp=acos(tmp)
      else
        tmp=1.d8
      endif
      getdelphi=tmp
      return
      end



