c
c Example analysis for "p p > l+ l- [QCD]" process.
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
      real * 8 bin,xmi,xms,pi
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
      integer j,kk,l,jpr,i,nnn
      character*5 cc(2)
      data cc/'     ','Born '/
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
       l=(i-1)*21
       call HwU_book(l+1,'V pt            '//HwUtype(i),100,0.d0,200.d0)
       call HwU_book(l+2,'V pt zoomout    '//HwUtype(i),100,0.d0,1000.d0)
       call HwU_book(l+3,'V log[pt]       '//HwUtype(i),98,0.1d0,5.d0)
       call HwU_book(l+4,'V y             '//HwUtype(i),40,-9.d0,9.d0)
       call HwU_book(l+5,'V eta           '//HwUtype(i),40,-9.d0,9.d0)
       call HwU_book(l+6,'mV              '//HwUtype(i),80,40.d0,120.d0)
       call HwU_book(l+ 7,'lm pt          '//HwUtype(i),100,0.d0,200.d0)
       call HwU_book(l+ 8,'lm pt 1        '//HwUtype(i),100,0.d0,1000.d0)
       call HwU_book(l+ 9,'lm log[pt]     '//HwUtype(i),98,0.1d0,5.d0)
       call HwU_book(l+10,'lm eta         '//HwUtype(i),40,-9.d0,9.d0)
       call HwU_book(l+11,'lp pt          '//HwUtype(i),100,0.d0,200.d0)
       call HwU_book(l+12,'lp pt 1        '//HwUtype(i),100,0.d0,1000.d0)
       call HwU_book(l+13,'lp log[pt]     '//HwUtype(i),98,0.1d0,5.d0)
       call HwU_book(l+14,'lp eta         '//HwUtype(i),40,-9.d0,9.d0)
       call HwU_book(l+15,'lmlp delta eta '//HwUtype(i),40,-9.d0,9.d0)
       call HwU_book(l+16,'lmlp azimt     '//HwUtype(i),20,0.d0,pi)
       call HwU_book(l+17,'lmlp log[pi-azimt] '//HwUtype(i),100,-4.d0,0.1d0)
       call HwU_book(l+18,'lmlp inv m     '//HwUtype(i),80,40.d0,120.d0)
       call HwU_book(l+19,'lmlp pt        '//HwUtype(i),100,0.d0,200.d0)
       call HwU_book(l+20,'lmlp log[pt]   '//HwUtype(i),98,0.1d0,5.d0)
       call HwU_book(l+21,'total          '//HwUtype(i),2,-1.d0,1.d0)
      enddo

      END

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
      DOUBLE PRECISION HWVDOT,PSUM(4),PPV(5),YCUT,XMV,PTV,YV,THV,ETAV,
     #  PPL(5),PPLB(5),PTL,YL,THL,ETAL,PLL,ENL,PTLB,YLB,THLB,ETALB,
     #  PLLB,ENLB,PTPAIR,DLL,CLL,AZI,AZINORM,XMLL,DETALLB
      INTEGER ICHSUM,ICHINI,IHEP,IV,IFV,IST,ID,IJ,ID1,JPR,IDENT,
     #  ILL,ILLB,IHRD,ILL0,ILLB0,NLP,NLM
      LOGICAL DIDSOF,flag,ISLP,ISLM,FOUNDP,FOUNDM
      REAL*8 PI,wmass,wgamma,bwcutoff,getinvm,getdelphi,getrapidity,
     &getpseudorap
      PARAMETER (PI=3.14159265358979312D0)
      REAL*8 TINY
      INTEGER KK,i,l
      DATA TINY/.1D-5/
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
C CHOOSE IDENT = 11 FOR ELECTRON PAIRS
C        IDENT = 13 FOR MUON PAIRS
C        IDENT = 15 FOR TAU PAIRS
      IDENT=13
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
      FOUNDP=.FALSE.
      FOUNDM=.FALSE.
      NLP=0
      NLM=0
      DO 100 IHEP=1,NHEP
        IST=ISTHEP(IHEP)      
        ID1=IDHEP(IHEP)
        ISLP=ID1.EQ.IDENT
        ISLM=ID1.EQ.-IDENT
         IF(NLM.EQ.0.AND.ISLM)THEN
            NLM=NLM+1
            ILL=IHEP
            FOUNDM=.TRUE.
         ENDIF
         IF(NLP.EQ.0.AND.ISLP)THEN
            NLP=NLP+1
            ILLB=IHEP
            FOUNDP=.TRUE.
         ENDIF
  100 CONTINUE
      IF(.NOT.FOUNDP.OR..NOT.FOUNDM)THEN
         WRITE(*,*)'NO LEPTONS FOUND.'
         WRITE(*,*)'CURRENTLY THIS ANALYSIS LOOKS FOR'
         IF(IDENT.EQ.11)WRITE(*,*)'ELECTRON PAIRS'
         IF(IDENT.EQ.13)WRITE(*,*)'MUON PAIRS'
         IF(IDENT.EQ.15)WRITE(*,*)'TAU PAIRS'
         WRITE(*,*)'IF THIS IS NOT MEANT, PLEASE EDIT IT.'
         STOP
      ENDIF
      DO IJ=1,5
        PPL(IJ)=PHEP(IJ,ILL)
        PPLB(IJ)=PHEP(IJ,ILLB)
        PPV(IJ)=PPL(IJ)+PPLB(IJ)
      ENDDO
C FILL THE HISTOS
      YCUT=2.5D0
C Variables of the vector boson
      xmv=getinvm(ppv(4),ppv(1),ppv(2),ppv(3))
      ptv=sqrt(ppv(1)**2+ppv(2)**2)
      yv=getrapidity(ppv(4),ppv(3))
      etav=getpseudorap(ppv(4),ppv(1),ppv(2),ppv(3))
C Variables of the leptons
      ptl=sqrt(ppl(1)**2+ppl(2)**2)
      yl=getrapidity(ppl(4),ppl(3))
      etal=getpseudorap(ppl(4),ppl(1),ppl(2),ppl(3))
c
      ptlb=sqrt(pplb(1)**2+pplb(2)**2)
      ylb=getrapidity(pplb(4),pplb(3))
      etalb=getpseudorap(pplb(4),pplb(1),pplb(2),pplb(3))
c
      ptpair=ptv
      azi=getdelphi(ppl(1),ppl(2),pplb(1),pplb(2))
      azinorm=(pi-azi)/pi
      xmll=xmv
      detallb=etal-etalb
c
      l=0
      call HwU_fill(l+1,(ptv),(WWW))
      call HwU_fill(l+2,(ptv),(WWW))
      if(ptv.gt.0.d0)call HwU_fill(l+3,(log10(ptv)),(WWW))
      call HwU_fill(l+4,(yv),(WWW))
      call HwU_fill(l+5,(etav),(WWW))
      call HwU_fill(l+6,(xmv),(WWW))
c
      call HwU_fill(l+7,(ptl),(WWW))
      call HwU_fill(l+8,(ptl),(WWW))
      if(ptl.gt.0.d0)call HwU_fill(l+9,(log10(ptl)),(WWW))
      call HwU_fill(l+10,(etal),(WWW))
      call HwU_fill(l+11,(ptlb),(WWW))
      call HwU_fill(l+12,(ptlb),(WWW))
      if(ptlb.gt.0.d0)call HwU_fill(l+13,(log10(ptlb)),(WWW))
      call HwU_fill(l+14,(etalb),(WWW))
c
      call HwU_fill(l+15,(detallb),(WWW))
      call HwU_fill(l+16,(azi),(WWW))
      if(azinorm.gt.0.d0)
     #  call HwU_fill(l+17,(log10(azinorm)),(WWW))
      call HwU_fill(l+18,(xmll),(WWW))
      call HwU_fill(l+19,(ptpair),(WWW))
      if(ptpair.gt.0)call HwU_fill(l+20,(log10(ptpair)),(WWW))
      call HwU_fill(l+21,(0d0),(WWW))
c
      l=l+21
      if(abs(etav).lt.ycut)then
        call HwU_fill(l+1,(ptv),(WWW))
        call HwU_fill(l+2,(ptv),(WWW))
        if(ptv.gt.0.d0)call HwU_fill(l+3,(log10(ptv)),(WWW))
      endif
      if(ptv.gt.20.d0)then
        call HwU_fill(l+4,(yv),(WWW))
        call HwU_fill(l+5,(etav),(WWW))
      endif
      if(abs(etav).lt.ycut.and.ptv.gt.20.d0)then
         call HwU_fill(l+6,(xmv),(WWW))
         call HwU_fill(l+21,(0d0),(WWW))
      endif
c
      if(abs(etal).lt.ycut)then
        call HwU_fill(l+7,(ptl),(WWW))
        call HwU_fill(l+8,(ptl),(WWW))
        if(ptl.gt.0.d0)call HwU_fill(l+9,(log10(ptl)),(WWW))
      endif
      if(ptl.gt.20.d0)call HwU_fill(l+10,(etal),(WWW))
      if(abs(etalb).lt.ycut)then
        call HwU_fill(l+11,(ptlb),(WWW))
        call HwU_fill(l+12,(ptlb),(WWW))
        if(ptlb.gt.0.d0)call HwU_fill(l+13,(log10(ptlb)),(WWW))
      endif
      if(ptlb.gt.20.d0)call HwU_fill(l+14,(etalb),(WWW))
c
      if( abs(etal).lt.ycut.and.abs(etalb).lt.ycut .and.
     #    ptl.gt.20.d0.and.ptlb.gt.20.d0)then
        call HwU_fill(l+15,(detallb),(WWW))
        call HwU_fill(l+16,(azi),(WWW))
        if(azinorm.gt.0.d0)
     #    call HwU_fill(l+17,(log10(azinorm)),(WWW))
        call HwU_fill(l+18,(xmll),(WWW))
        call HwU_fill(l+19,(ptpair),(WWW))
        if(ptpair.gt.0) 
     #    call HwU_fill(l+20,(log10(ptpair)),(WWW))
      endif
      call HwU_add_points
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
