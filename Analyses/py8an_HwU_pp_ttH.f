c
c Example analysis for "p p > t t~ H [QCD]" process.
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
      use HwU_wgts_info_len
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
      character*(wgts_info_len) weights_info(max_weight_shower)
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
      do i=1,1
        l=(i-1)*19
        call HwU_book(l+ 1,'total rate    '//HwUtype(i),1,0.5d0,5.5d0)
        call HwU_book(l+ 2,'t rap         '//HwUtype(i),50,-5d0,5d0)
        call HwU_book(l+ 3,'tx rap        '//HwUtype(i),50,-5d0,5d0)
        call HwU_book(l+ 4,'t-tx pair rap '//HwUtype(i),60,-3d0,3d0)
        call HwU_book(l+ 5,'m t-tx        '//HwUtype(i),20,100d0,2000d0)
        call HwU_book(l+ 6,'pt t          '//HwUtype(i),20,0d0,1000d0)
        call HwU_book(l+ 7,'pt tx         '//HwUtype(i),20,0d0,1000d0)
        call HwU_book(l+ 8,'pt t-tx       '//HwUtype(i),20,0d0,1000d0)
        call HwU_book(l+ 9,'pt H   '//HwUtype(i),20,0d0,1000d0)
        call HwU_book(l+ 10,'y H        '//HwUtype(i),20,-5d0,5d0)
        call HwU_book(l+ 11,'pt(tH)        '//HwUtype(i),20,0d0,1500d0)
        call HwU_book(l+ 12,'m(tH) '//HwUtype(i),20,0d0,2000d0)
        call HwU_book(l+ 13,'pt(ttH)        '//HwUtype(i),20,0d0,1000d0)
        call HwU_book(l+ 14,'m(ttH)         '//HwUtype(i),20,400d0,2000d0)
        call HwU_book(l+ 15,'pt(j)         '//HwUtype(i),20,0d0,100d0)
        call HwU_book(l+ 16,'pt(tj)       '//HwUtype(i),20,0d0,1000d0)
        call HwU_book(l+ 17,'pt(Hj)       '//HwUtype(i),20,0d0,1000d0)
        call HwU_book(l+ 18,'dr(tj)       '//HwUtype(i),20,0d0,10d0)
        call HwU_book(l+ 19,'dr(Hj)       '//HwUtype(i),20,0d0,10d0)
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
     # XPNB(5),XPBB(5),p_t(4),p_tx(4),pttx(4),
     # mtt,pt_t,pt_tx,pt_ttx,yt,ytx,yttx,var 
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

      PARAMETER (NMAX=2000)
      INTEGER NJET,JET(NMAX),IPOS(NMAX),njet_central,NN
      DOUBLE PRECISION PALG,RFJ,SYCUT,PP(4,NMAX),PJET(4,NMAX),
     # PTJET(NMAX),ETAJET(NMAX),YJET(NMAX),pjet_new(4,nmax),
     # njdble,njcdble,y_central

      double precision pt_hj,pt_h,y_h,pt_th,m_th,pt_tth,m_tth,p_h(4)
      double precision pt_tj,dr_tj,dr_hj
      double precision p_th(4),p_tth(4),p_tj(4),p_hj(4)
      real*8 getdr

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
      NN=0
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
        IF(ID1.EQ.25)THEN
          IH=IHEP
        ENDIF

        IF(ABS(ID1).GT.100.AND.IST.EQ.1) THEN
           NN=NN+1
           IF (NN.GT.NMAX) STOP 'Too many particles [hadrons]!'
           DO I=1,4
              PP(I,NN)=PHEP(I,IHEP)
           ENDDO
        ENDIF
  100 CONTINUE
      IF(IQ1*IQ2.EQ.0)THEN
         WRITE(*,*)'ERROR 501 IN PYANAL'
      ENDIF
      DO IJ=1,4
         p_t(IJ)=PHEP(IJ,IT1)
         p_tx(IJ)=PHEP(IJ,IT2)
         p_h(IJ)=PHEP(IJ,IH)
         pttx(IJ)=PHEP(IJ,IT1)+PHEP(IJ,IT2)
      ENDDO

C---CLUSTER THE EVENT
      palg = 1d0
      rfj  = 0.4d0
      sycut= 10d0
      do i=1,nmax
        do j=1,4
          pjet(j,i)=0d0
        enddo
        ptjet(i)=0d0
        jet(i)=0
      enddo
      njet=0
      call fastjetppgenkt(pp,nn,rfj,sycut,palg,pjet,njet,jet)
      do i=1,njet
         ptjet(i)=sqrt(pjet(1,i)**2+pjet(2,i)**2)
         if(i.gt.1)then
            if (ptjet(i).gt.ptjet(i-1)) then
               write (*,*) "Error 1: jets should be ordered in pt"
               WRITE(*,*)'ERROR 501 IN PYANAL'
               STOP
            endif
         endif
      enddo


      mtt    = getinvm(pttx(4),pttx(1),pttx(2),pttx(3))
      pt_t   = dsqrt(p_t(1)**2 + p_t(2)**2)
      pt_tx  = dsqrt(p_tx(1)**2 + p_tx(2)**2)
      pt_ttx = dsqrt(pttx(1)**2 + pttx(2)**2)
      yt  = getrapidity(p_t(4), p_t(3))
      ytx = getrapidity(p_tx(4), p_tx(3))
      yttx= getrapidity(pttx(4), pttx(3))

      pt_h=dsqrt(p_h(1)**2 + p_h(2)**2)
      y_h  = getrapidity(p_h(4), p_h(3))
      do i=1,4
        p_th(i)=p_t(i)+p_h(i)
        p_tth(i)=p_t(i)+p_tx(i)+p_h(i)
        p_tj(i)=p_t(i)+pjet(i,1)
        p_hj(i)=p_h(i)+pjet(i,1)
      enddo
      pt_th=dsqrt(p_th(1)**2 + p_th(2)**2)
      m_th=getinvm(p_th(4),p_th(1),p_th(2),p_th(3))
      pt_tth=dsqrt(p_tth(1)**2 + p_tth(2)**2)
      m_tth=getinvm(p_tth(4),p_tth(1),p_tth(2),p_tth(3))
      pt_tj=dsqrt(p_tj(1)**2 + p_tj(2)**2)
      pt_hj=dsqrt(p_hj(1)**2 + p_hj(2)**2)
      dr_tj=getdr(p_t(4),p_t(1),p_t(2),p_t(3),pjet(4,1),pjet(1,1),pjet(2,1),pjet(3,1))
      dr_hj=getdr(p_h(4),p_h(1),p_h(2),p_h(3),pjet(4,1),pjet(1,1),pjet(2,1),pjet(3,1))

     
      var=1.d0
      do i=1,1
         l=(i-1)*19
         call HwU_fill(l+1,var,WWW)
         call HwU_fill(l+2,yt,WWW)
         call HwU_fill(l+3,ytx,WWW)
         call HwU_fill(l+4,yttx,WWW)
         call HwU_fill(l+5,mtt,WWW)
         call HwU_fill(l+6,pt_t,WWW)
         call HwU_fill(l+7,pt_tx,WWW)
         call HwU_fill(l+8,pt_ttx,WWW)
         call HwU_fill(l+9,pt_h,WWW)
         call HwU_fill(l+10,y_h,WWW)
         call HwU_fill(l+11,pt_th,WWW)
         call HwU_fill(l+12,m_th,WWW)
         call HwU_fill(l+13,pt_tth,WWW)
         call HwU_fill(l+14,m_tth,WWW)
         call HwU_fill(l+15,pt_j,WWW)
         call HwU_fill(l+16,pt_tj,WWW)
         call HwU_fill(l+17,pt_hj,WWW)
         call HwU_fill(l+18,dr_tj,WWW)
         call HwU_fill(l+19,dr_hj,WWW)
      enddo
      call HwU_add_points
c
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

      function getdr(en1,ptx1,pty1,pl1,en2,ptx2,pty2,pl2)
      implicit none
      real*8 getdr,en1,ptx1,pty1,pl1,en2,ptx2,pty2,pl2,deta,dphi,
     # getpseudorap,getdelphi
c
      deta=getpseudorap(en1,ptx1,pty1,pl1)-
     #     getpseudorap(en2,ptx2,pty2,pl2)
      dphi=getdelphi(ptx1,pty1,ptx2,pty2)
      getdr=sqrt(dphi**2+deta**2)
      return
      end
