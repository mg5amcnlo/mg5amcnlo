c
c Example analysis for "p p > h j j [QCD]" process.
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
      integer nwgt,max_weight,nwgt_analysis
      common/cnwgt/nwgt
      common/c_analysis/nwgt_analysis
      integer maxRWGT
      parameter (maxRWGT=100)
      double precision wgtxsecRWGT(maxRWGT)
      parameter (max_weight=maxscales*maxscales+maxpdfs+maxRWGT+1)
      character*15 weights_info(max_weight)
      common/cwgtsinfo/weights_info
      character*8 cc(2)
      data cc/'        ','vbfcuts '/
      real*8 vetomin, vetomax
      integer nbinveto
      common /to_veto_hist/vetomin,vetomax,nbinveto
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c To be changed !!
      nwgt=1
      weights_info(nwgt)="central value  "
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      vetomin = 0d0
      vetomax = 100d0
      nbinveto = 50
c
c Initialize histograms
      call HwU_inithist(nwgt,weights_info)
c Set method for error estimation to '0', i.e., use Poisson statistics
c for the uncertainty estimate
      call set_error_estimation(0)
      nwgt_analysis=nwgt
      do i=1,2
        l=(i-1)*54
         call HwU_book(l+  1,'total '//HwUtype(i),5,0.5d0,5.5d0)

         call HwU_book(l+  2,'Higgs pT '//HwUtype(i),50,0.d0,400.d0)
         call HwU_book(l+  3,'Higgs pT 1 '//HwUtype(i),50,0.d0,800.d0)
         call HwU_book(l+  4,'Higgs logpT '//HwUtype(i),50,0.d0,4.d0)
         call HwU_book(l+  5,'Higgs eta '//HwUtype(i),50,-6.d0,6.d0)
         call HwU_book(l+  6,'Higgs y '//HwUtype(i),50,-6.d0,6.d0)

         call HwU_book(l+  7,'j1 pT '//HwUtype(i),50,0.d0,400.d0)
         call HwU_book(l+  8,'j1 pT 1 '//HwUtype(i),50,0.d0,800.d0)
         call HwU_book(l+  9,'j1 logpT '//HwUtype(i),50,0.d0,4.d0)
         call HwU_book(l+ 10,'j1 eta '//HwUtype(i),50,-6.d0,6.d0)
         call HwU_book(l+ 11,'j1 y '//HwUtype(i),50,-6.d0,6.d0)

         call HwU_book(l+ 12,'j2 pT '//HwUtype(i),50,0.d0,400.d0)
         call HwU_book(l+ 13,'j2 pT 1 '//HwUtype(i),50,0.d0,800.d0)
         call HwU_book(l+ 14,'j2 logpT '//HwUtype(i),50,0.d0,4.d0)
         call HwU_book(l+ 15,'j2 eta '//HwUtype(i),50,-6.d0,6.d0)
         call HwU_book(l+ 16,'j2 y '//HwUtype(i),50,-6.d0,6.d0)

         call HwU_book(l+ 17,'j3 pT '//HwUtype(i),50,0.d0,400.d0)
         call HwU_book(l+ 18,'j3 pT 1 '//HwUtype(i),50,0.d0,800.d0)
         call HwU_book(l+ 19,'j3 logpT '//HwUtype(i),50,0.d0,4.d0)
         call HwU_book(l+ 20,'j3 eta '//HwUtype(i),50,-6.d0,6.d0)
         call HwU_book(l+ 21,'j3 y '//HwUtype(i),50,-6.d0,6.d0)

         call HwU_book(l+ 22,'H+j1 pT '//HwUtype(i),50,0.d0,400.d0)
         call HwU_book(l+ 23,'H+j1 pT 1 '//HwUtype(i),50,0.d0,800.d0)
         call HwU_book(l+ 24,'H+j1 logpT '//HwUtype(i),50,0.d0,4.d0)
         call HwU_book(l+ 25,'H+j1 eta '//HwUtype(i),50,-6.d0,6.d0)
         call HwU_book(l+ 26,'H+j1 y '//HwUtype(i),50,-6.d0,6.d0)

         call HwU_book(l+ 27,'j1+j2 pT '//HwUtype(i),50,0.d0,400.d0)
         call HwU_book(l+ 28,'j1+j2 pT 1 '//HwUtype(i),50,0.d0,800.d0)
         call HwU_book(l+ 29,'j1+j2 logpT '//HwUtype(i),50,0.d0,4.d0)
         call HwU_book(l+ 30,'j1+j2 eta '//HwUtype(i),50,-6.d0,6.d0)
         call HwU_book(l+ 31,'j1+j2 y '//HwUtype(i),50,-6.d0,6.d0)

         call HwU_book(l+ 32,'syst pT '//HwUtype(i),50,0.d0,400.d0)
         call HwU_book(l+ 33,'syst pT 1 '//HwUtype(i),50,0.d0,800.d0)
         call HwU_book(l+ 34,'syst logpT '//HwUtype(i),50,0.d0,4.d0)
         call HwU_book(l+ 35,'syst eta '//HwUtype(i),50,-10.d0,10.d0)
         call HwU_book(l+ 36,'syst y '//HwUtype(i),50,-6.d0,6.d0)

         call HwU_book(l+ 37,'Dphi H-j1 '//HwUtype(i),50,0d0,pi)
         call HwU_book(l+ 38,'Dphi H-j2 '//HwUtype(i),50,0d0,pi)
         call HwU_book(l+ 39,'Dphi j1-j2 '//HwUtype(i),50,0d0,pi)

         call HwU_book(l+ 40,'DR H-j1 '//HwUtype(i),50,0d0,10.d0)
         call HwU_book(l+ 41,'DR H-j2 '//HwUtype(i),50,0d0,10.d0)
         call HwU_book(l+ 42,'DR j1-j2 '//HwUtype(i),50,0d0,10.d0)
         call HwU_book(l+ 43,'mj1j2 '//HwUtype(i),50,0d0,3000.d0)

c Nason-Oleari plots (hep-ph/0911.5299)
         call HwU_book(l+ 44,'yj1-yj2 '//HwUtype(i),50,0.d0,10.d0)
         call HwU_book(l+ 45,'yj3_rel '//HwUtype(i),50,-6.d0,6.d0)
         call HwU_book(l+ 46,'njets '//HwUtype(i),10,-0.5d0,9.5d0)
         call HwU_book(l+ 47,'ptrel_j1 '//HwUtype(i),50,0.d0,200.d0)
         call HwU_book(l+ 48,'ptrel_j2 '//HwUtype(i),50,0.d0,200.d0)
         call HwU_book(l+ 49,'P-veto '//HwUtype(i),50,vetomin,vetomax)
         call HwU_book(l+ 50,'jveto pT '//HwUtype(i),50,vetomin,vetomax)
         call HwU_book(l+ 51,'jveto pT 1 '//HwUtype(i),50,vetomin,2d0*vetomax)
         call HwU_book(l+ 52,'jveto logpT '//HwUtype(i),50,0.d0,4.d0)
         call HwU_book(l+ 53,'jveto eta '//HwUtype(i),50,-6.d0,6.d0)
         call HwU_book(l+ 54,'jveto y '//HwUtype(i),50,-6.d0,6.d0)
      enddo
 999  END

C----------------------------------------------------------------------
      SUBROUTINE HWAEND
C     USER''S ROUTINE FOR TERMINAL CALCULATIONS, HISTOGRAM OUTPUT, ETC
C----------------------------------------------------------------------
      INCLUDE 'HEPMC.INC'
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
      END

C----------------------------------------------------------------------
      SUBROUTINE HWANAL
C     USER''S ROUTINE TO ANALYSE DATA FROM EVENT
C----------------------------------------------------------------------
      INCLUDE 'HEPMC.INC'
      include 'reweight0.inc'
      DOUBLE PRECISION HWVDOT,PSUM(4),PJJ(4)
      INTEGER ICHSUM,ICHINI,IHEP,IST,ID,IJ,J,NN,I,jj,l
      LOGICAL DIDSOF
      double precision getpt,getpseudorap
c jet stuff
      INTEGER NMAX
      PARAMETER (NMAX=2000)
      INTEGER NJET,JET(NMAX),IPOS(NMAX)
      DOUBLE PRECISION PALG,RFJ,SYCUT,PP(4,NMAX),PJET(4,NMAX),
     # PTJET(NMAX),ETAJET(NMAX),RAPJET(NMAX),pjet_new(4,nmax)
      REAL*8 PI
      PARAMETER (PI=3.14159265358979312D0)
      REAL*8 WWW0
      INTEGER KK,J1,J2,nh,ih,nj
      double precision getrapidityv4,getptv4,getinvmv4,getdelphiv4,
     &getdrv4,getpseudorapv4
      double precision yjet(nmax),pH(4),pj1(4),pj2(4),pj3(4),pHj(4),
     &psyst(4),ptH,etaH,yH,ptj1,etaj1,yj1,ptj2,etaj2,yj2,ptj3,etaj3,yj3,
     &ptHj,etaHj,yHj,ptjj,etajj,yjj,ptsyst,etasyst,ysyst,
     &DphiHj1,DphiHj2,Dphij1j2,DRHj1,DRHj2,DRj1j2,mj1j2
      double precision ptj_cut, yj_cut, ptj_tag,deltay12,mj1j2min
      logical pass_tag_cuts,flag,accepted
      double precision njdble,Dyj1j2,yj3rel,ptrel_j1,ptrel_j2,pj1new(4),
     &pj2new(4),prel_j1(4),prel_j2(4),pj1boost(4),pj2boost(4)
      real*8 xd(1:3)
      data (xd(i),i=1,3)/0,0,1/
      double precision chy1,chy2,shy1,shy2,chy1mo,chy2mo,
     &ppboost(4,nmax),getmod,getcosv4
      double precision temp_scalup
      common/myscalup/temp_scalup
      integer ij1y, ij2y, ij3y
      integer ij1, ij2, ij3
      integer ijveto, ijvetoy
      integer njety
      real*8 vetomin, vetomax
      integer nbinveto
      common /to_veto_hist/vetomin,vetomax,nbinveto
      double precision pt_veto, pjveto(4), xsecup2
      integer nwgt_analysis,max_weight
      common/c_analysis/nwgt_analysis
      integer maxRWGT
      parameter (maxRWGT=100)
      double precision wgtxsecRWGT(maxRWGT)
      parameter (max_weight=maxscales*maxscales+maxpdfs+maxRWGT+1)
      double precision ww(max_weight),www(max_weight)
      common/cww/ww
c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c To be changed !!
      ww(1)=1d0
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      IF (WW(1).EQ.0D0) THEN
         WRITE(*,*)'WW(1) = 0. Stopping'
         STOP
      ENDIF
C INITIALISE
      do i=1,nmax
        do j=1,4
          pp(j,i)=0d0
        enddo
      enddo
      xsecup2=1d0
C INCOMING PARTONS MAY TRAVEL IN THE SAME DIRECTION: IT''S A POWER-SUPPRESSED
C EFFECT, SO THROW THE EVENT AWAY
      IF(SIGN(1.D0,PHEP(3,1)).EQ.SIGN(1.D0,PHEP(3,2)))THEN
         WRITE(*,*)'WARNING 502 IN HWANAL'
         GOTO 999
      ENDIF
      DO I=1,nwgt_analysis
         WWW(I)=EVWGT*ww(i)/ww(1)
      ENDDO
      ICHSUM=0
      NN=0
      NH=0
      DO 100 IHEP=1,NHEP
        IST=ISTHEP(IHEP)
        ID=IDHEP(IHEP)
        IF(ABS(ID).GT.100.AND.IST.EQ.1) THEN
           NN=NN+1
           IF (NN.GT.NMAX) STOP 'Too many particles [hadrons]!'
           DO I=1,4
              PP(I,NN)=PHEP(I,IHEP)
           ENDDO
        ENDIF
C FIND THE HIGGS
        IF(ID.EQ.25)THEN
           NH=1
           IH=IHEP
        ENDIF
  100 CONTINUE
C CHECK THAT JUST ONE HIGGS HAS BEEN FOUND
      IF(NH.EQ.0)THEN
         WRITE(*,*)'NO HIGGS FOUND!'
         STOP
      ENDIF
      IF(NH.GT.1)THEN
         WRITE(*,*)'MORE THAN ONE HIGGS! ',NH
         STOP
      ENDIF
C CHECK SOME TRACKS HAVE BEEN FOUND
      IF(NN.EQ.0)THEN
        WRITE(*,*)'NO TRACKS FOUND'
        STOP
      ENDIF

C---CLUSTER THE EVENT
      palg =-1.d0
      rfj  =0.4d0
      sycut=20d0
      yjmax=4.5d0
      do i=1,nmax
        do j=1,4
          pjet(j,i)=0d0
        enddo
        ptjet(i)=0d0
        yjet(i)=0d0
        etajet(i)=0d0
        jet(i)=0
      enddo
      ij1y=0
      ij2y=0
      ij3y=0
      njet=0
      njety=0
      ijveto = 0
      ijvetoy = 0
      call fastjetppgenkt(pp,nn,rfj,sycut,palg,pjet,njet,jet)
      do i=1,njet
         ptjet(i)=getptv4(pjet(1,i))
         if(i.gt.1)then
            if (ptjet(i).gt.ptjet(i-1)) then
               write (*,*) "Error 1: jets should be ordered in pt"
               WRITE(*,*)'ERROR 501 IN HWANAL'
               STOP
            endif
         endif
         yjet(i)=getrapidityv4(pjet(1,i))
         etajet(i)=getpseudorapv4(pjet(1,i))
c look for veto jet without y cuts
         if (i.gt.2.and.yjet(i).gt.min(yjet(1),yjet(2)).and.
     &      yjet(i).lt.max(yjet(1),yjet(2)).and.ijveto.eq.0) ijveto = i

C now look for jets within the rapidity cuts
         if (dabs(yjet(i)).lt.yjmax) then
             njety=njety+1
             if (ij1y.eq.0) then
                 ij1y = i
             else if (ij2y.eq.0) then
                 ij2y = i
             else if (ij3y.eq.0) then
                 ij3y = i
             endif
c look for veto jet with y cuts
             if (ij3y.gt.0.and.
     &           yjet(i).gt.min(yjet(ij1y),yjet(ij2y)).and.
     &           yjet(i).lt.max(yjet(ij1y),yjet(ij2y)).and.ijvetoy.eq.0) 
     &           ijvetoy = i
         endif
      enddo

c Nason-Oleari cuts (hep-ph/0911.5299)
      ptj_tag  = 20d0
      deltay12 = 4.d0
      mj1j2min = 600d0

c this is the loop for w-o / w vbf cuts
      do i=1,2
      if(i.eq.1) then 
         ij1 = 1
         ij2 = 2
         ij3 = 3
      endif
      if(i.eq.2) then
         njet = njety
         ijveto = ijvetoy
         ij1 = ij1y
         ij2 = ij2y
         ij3 = ij3y
      endif

c Load momenta
         do jj=1,4
             pH(jj)   =phep(jj,ih)
             pj1(jj)  =pjet(jj,ij1)
             pj2(jj)  =pjet(jj,ij2)
             pj3(jj)  =pjet(jj,ij3)
             pjj(jj)  =pjet(jj,ij1)+pjet(jj,ij2)
             pHj(jj)  =pjet(jj,ij1)+pH(jj)
             psyst(jj)=pjet(jj,ij1)+pjet(jj,ij2)+pH(jj)
             pjveto(jj)=pjet(jj,ijveto)
         enddo

c Define observables
c Higgs
         ptH     = getptv4(pH)
         etaH    = getpseudorapv4(pH)
         yH      = getrapidityv4(pH)
         njdble  = dble(njet)
c At least one jet
      if(njet.ge.1)then
        ptj1    = getptv4(pj1)
        etaj1   = getpseudorapv4(pj1)
        yj1     = getrapidityv4(pj1)
        ptHj    = getptv4(pHj)
        etaHj   = getpseudorapv4(pHj)
        yHj     = getrapidityv4(pHj)
        DphiHj1 = getdelphiv4(pH,pj1)
        DRHj1   = getdrv4(pH,pj1)
      endif
c At least two jets
      if(njet.ge.2)then
        ptj2    = getptv4(pj2)
        etaj2   = getpseudorapv4(pj2)
        yj2     = getrapidityv4(pj2)
        ptjj    = getptv4(pjj)
        etajj   = getpseudorapv4(pjj)
        yjj     = getrapidityv4(pjj)
        ptsyst  = getptv4(psyst)
        etasyst = getpseudorapv4(psyst)
        ysyst   = getrapidityv4(psyst)
        DphiHj2 = getdelphiv4(pH,pj2)
        Dphij1j2= getdelphiv4(pj1,pj2)
        DRHj2   = getdrv4(pH,pj2)
        DRj1j2  = getdrv4(pj1,pj2)
        mj1j2   = getinvmv4(pjj)
        Dyj1j2  = abs(yj1-yj2)
      endif
c At least three jets
      if(njet.ge.3)then
        ptj3    = getptv4(pj3)
        etaj3   = getpseudorapv4(pj3)
        yj3     = getrapidityv4(pj3)
        yj3rel  = yj3-(yj1+yj2)/2d0
      endif
c
      chy1=cosh(yj1)
      shy1=sinh(yj1)
      chy1mo=chy1-1.d0
      chy2=cosh(yj2)
      shy2=sinh(yj2)
      chy2mo=chy2-1.d0
c boostwdir3 is the same as boostwdir2, but with
c components from 1 to 4, rather than from 0 to 3
      call boostwdir3(chy1,shy1,chy1mo,xd,pj1,pj1boost)
      call boostwdir3(chy2,shy2,chy2mo,xd,pj2,pj2boost)
      ptrel_j1=0d0
      ptrel_j2=0d0

      pass_tag_cuts = njety.ge.2 .and.
     &                ptj1.ge.ptj_tag .and.
     &                ptj2.ge.ptj_tag .and.
     &                abs(yj1-yj2).ge.deltay12 .and.
     &                yj1*yj2.le.0d0 .and.
     &                mj1j2.ge.mj1j2min 

      if(i.eq.1) then 
         flag=.true.
      endif

      if(i.eq.2) then
         flag=pass_tag_cuts
      endif

      do j=1,nn
         if(njet.ge.1.and.jet(j).eq.1)then
           call boostwdir3(chy1,shy1,chy1mo,xd,pp(1,j),ppboost(1,j))
           call getwedge(ppboost(1,j),pj1boost,prel_j1)
           ptrel_j1=ptrel_j1+getmod(prel_j1)/getmod(pj1boost)
         elseif(njet.ge.2.and.jet(j).eq.2)then
           call boostwdir3(chy2,shy2,chy2mo,xd,pp(1,j),ppboost(1,j))
           call getwedge(ppboost(1,j),pj2boost,prel_j2)
           ptrel_j2=ptrel_j2+getmod(prel_j2)/getmod(pj2boost)
         endif
      enddo

      l=(i-1)*54
      if(flag)then
         call HwU_fill(l+  1,1d0,WWW)
         call HwU_fill(l+  2,ptH,WWW)
         call HwU_fill(l+  3,ptH,WWW)
         if(ptH.gt.0d0) call HwU_fill(l+  4,log10(ptH),WWW)
         call HwU_fill(l+  5,etaH,WWW)
         call HwU_fill(l+  6,yH,WWW)
         call HwU_fill(l+ 46,njdble,WWW)

         if(njet.ge.1)then
            call HwU_fill(l+  7,ptj1,WWW)
            call HwU_fill(l+  8,ptj1,WWW)
            if (ptj1.gt.0d0) call HwU_fill(l+  9,log10(ptj1),WWW)
            call HwU_fill(l+ 10,etaj1,WWW)
            call HwU_fill(l+ 11,yj1,WWW)
            call HwU_fill(l+ 22,ptHj,WWW)
            call HwU_fill(l+ 23,ptHj,WWW)
            if(ptHj.gt.0d0) call HwU_fill(l+ 24,log10(ptHj),WWW)
            call HwU_fill(l+ 25,etaHj,WWW)
            call HwU_fill(l+ 26,yHj,WWW)
            call HwU_fill(l+ 37,DphiHj1,WWW)
            call HwU_fill(l+ 40,DRHj1,WWW)
            call HwU_fill(l+ 47,ptrel_j1,WWW)
         endif 
         
         if(njet.ge.2)then
            call HwU_fill(l+ 12,ptj2,WWW)
            call HwU_fill(l+ 13,ptj2,WWW)
            if (ptj2.gt.0d0) call HwU_fill(l+ 14,log10(ptj2),WWW)
            call HwU_fill(l+ 15,etaj2,WWW)
            call HwU_fill(l+ 16,yj2,WWW)
            call HwU_fill(l+ 27,ptjj,WWW)
            call HwU_fill(l+ 28,ptjj,WWW)
            if(ptjj.gt.0d0) call HwU_fill(l+ 29,log10(ptjj),WWW)
            call HwU_fill(l+ 30,etajj,WWW)
            call HwU_fill(l+ 31,yjj,WWW)
            call HwU_fill(l+ 32,ptsyst,WWW)
            call HwU_fill(l+ 33,ptsyst,WWW)
            if(ptsyst.gt.0d0) call HwU_fill(l+ 34,log10(ptsyst),WWW)
            call HwU_fill(l+ 35,etasyst,WWW)
            call HwU_fill(l+ 36,ysyst,WWW)
            call HwU_fill(l+ 38,DphiHj2,WWW)
            call HwU_fill(l+ 39,Dphij1j2,WWW)
            call HwU_fill(l+ 41,DRHj2,WWW)
            call HwU_fill(l+ 42,DRj1j2,WWW)
            call HwU_fill(l+ 43,mj1j2,WWW)
            call HwU_fill(l+ 44,Dyj1j2,WWW)
            call HwU_fill(l+ 48,ptrel_j2,WWW)
         endif
         
         if(njet.ge.3)then
            call HwU_fill(l+ 17,ptj3,WWW)
            call HwU_fill(l+ 18,ptj3,WWW)
            if(ptj3.gt.0d0) call HwU_fill(l+ 19,log10(ptj3),WWW)
            call HwU_fill(l+ 20,etaj3,WWW)
            call HwU_fill(l+ 21,yj3,WWW)
            call HwU_fill(l+ 45,yj3rel,WWW)
         endif
         if (ijveto.gt.0) then
            pt_veto = getptv4(pjveto)
            do jj=1,nbinveto
               if (pt_veto.gt. (vetomin+(vetomax-vetomin)*dble(jj-1)
     $              /dble(nbinveto))) then
                  call HwU_fill(l+49, (vetomax-vetomin)* dble(jj)
     $                 /dble(nbinveto)*0.99, WWW/xsecup2)
               endif
            enddo
            call HwU_fill(l+50,pt_veto,WWW)
            call HwU_fill(l+51,pt_veto,WWW)
            if (pt_veto.gt.0d0) call HwU_fill(l+52,dlog10(pt_veto),WWW)
            call HwU_fill(l+53,getpseudorapv4(pjveto),WWW)
            call HwU_fill(l+54,getrapidityv4(pjveto),WWW)
         endif
      endif
      enddo
      call HwU_add_points
C
 999  END


      FUNCTION RANDA(SEED)
*     -----------------
* Ref.: K. Park and K.W. Miller, Comm. of the ACM 31 (1988) p.1192
* Use seed = 1 as first value.
*
      IMPLICIT INTEGER(A-Z)
      DOUBLE PRECISION MINV,RANDA
      SAVE
      PARAMETER(M=2147483647,A=16807,Q=127773,R=2836)
      PARAMETER(MINV=0.46566128752458d-09)
      HI = SEED/Q
      LO = MOD(SEED,Q)
      SEED = A*LO - R*HI
      IF(SEED.LE.0) SEED = SEED + M
      RANDA = SEED*MINV
      END




      function getrapidity(en,pl)
      implicit none
      real*8 getrapidity,en,pl,tiny,xplus,xminus,y
      parameter (tiny=1.d-8)
c
      xplus=en+pl
      xminus=en-pl
      if(xplus.gt.tiny.and.xminus.gt.tiny)then
        if( (xplus/xminus).gt.tiny.and.(xminus/xplus).gt.tiny )then
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


      function getdry(en1,ptx1,pty1,pl1,en2,ptx2,pty2,pl2)
      implicit none
      real*8 getdry,en1,ptx1,pty1,pl1,en2,ptx2,pty2,pl2,deta,dphi,
     # getrapidity,getdelphi
c
      deta=getrapidity(en1,pl1)-
     #     getrapidity(en2,pl2)
      dphi=getdelphi(ptx1,pty1,ptx2,pty2)
      getdry=sqrt(dphi**2+deta**2)
      return
      end


      function getptv(p)
      implicit none
      real*8 getptv,p(5)
c
      getptv=sqrt(p(1)**2+p(2)**2)
      return
      end


      function getpseudorapv(p)
      implicit none
      real*8 getpseudorapv,p(5)
      real*8 getpseudorap
c
      getpseudorapv=getpseudorap(p(4),p(1),p(2),p(3))
      return
      end


      function getrapidityv(p)
      implicit none
      real*8 getrapidityv,p(5)
      real*8 getrapidity
c
      getrapidityv=getrapidity(p(4),p(3))
      return
      end


      function getdrv(p1,p2)
      implicit none
      real*8 getdrv,p1(5),p2(5)
      real*8 getdr
c
      getdrv=getdr(p1(4),p1(1),p1(2),p1(3),
     #             p2(4),p2(1),p2(2),p2(3))
      return
      end


      function getinvmv(p)
      implicit none
      real*8 getinvmv,p(5)
      real*8 getinvm
c
      getinvmv=getinvm(p(4),p(1),p(2),p(3))
      return
      end


      function getdelphiv(p1,p2)
      implicit none
      real*8 getdelphiv,p1(5),p2(5)
      real*8 getdelphi
c
      getdelphiv=getdelphi(p1(1),p1(2),
     #                     p2(1),p2(2))
      return
      end


      function getptv4(p)
      implicit none
      real*8 getptv4,p(4)
c
      getptv4=sqrt(p(1)**2+p(2)**2)
      return
      end


      function getpseudorapv4(p)
      implicit none
      real*8 getpseudorapv4,p(4)
      real*8 getpseudorap
c
      getpseudorapv4=getpseudorap(p(4),p(1),p(2),p(3))
      return
      end


      function getrapidityv4(p)
      implicit none
      real*8 getrapidityv4,p(4)
      real*8 getrapidity
c
      getrapidityv4=getrapidity(p(4),p(3))
      return
      end


      function getdrv4(p1,p2)
      implicit none
      real*8 getdrv4,p1(4),p2(4)
      real*8 getdr
c
      getdrv4=getdr(p1(4),p1(1),p1(2),p1(3),
     #              p2(4),p2(1),p2(2),p2(3))
      return
      end


      function getinvmv4(p)
      implicit none
      real*8 getinvmv4,p(4)
      real*8 getinvm
c
      getinvmv4=getinvm(p(4),p(1),p(2),p(3))
      return
      end


      function getdelphiv4(p1,p2)
      implicit none
      real*8 getdelphiv4,p1(4),p2(4)
      real*8 getdelphi
c
      getdelphiv4=getdelphi(p1(1),p1(2),
     #                      p2(1),p2(2))
      return
      end


      function getcosv4(q1,q2)
      implicit none
      real*8 getcosv4,q1(4),q2(4)
      real*8 xnorm1,xnorm2,tmp
c
      if(q1(4).lt.0.d0.or.q2(4).lt.0.d0)then
        getcosv4=-1.d10
        return
      endif
      xnorm1=sqrt(q1(1)**2+q1(2)**2+q1(3)**2)
      xnorm2=sqrt(q2(1)**2+q2(2)**2+q2(3)**2)
      if(xnorm1.lt.1.d-6.or.xnorm2.lt.1.d-6)then
        tmp=-1.d10
      else
        tmp=q1(1)*q2(1)+q1(2)*q2(2)+q1(3)*q2(3)
        tmp=tmp/(xnorm1*xnorm2)
        if(abs(tmp).gt.1.d0.and.abs(tmp).le.1.001d0)then
          tmp=sign(1.d0,tmp)
        elseif(abs(tmp).gt.1.001d0)then
          write(*,*)'Error in getcosv4',tmp
          stop
        endif
      endif
      getcosv4=tmp
      return
      end



      function getmod(p)
      implicit none
      double precision p(4),getmod

      getmod=sqrt(p(1)**2+p(2)**2+p(3)**2)

      return
      end



      subroutine getperpenv4(q1,q2,qperp)
c Normal to the plane defined by \vec{q1},\vec{q2}
      implicit none
      real*8 q1(4),q2(4),qperp(4)
      real*8 xnorm1,xnorm2
      integer i
c
      xnorm1=sqrt(q1(1)**2+q1(2)**2+q1(3)**2)
      xnorm2=sqrt(q2(1)**2+q2(2)**2+q2(3)**2)
      if(xnorm1.lt.1.d-6.or.xnorm2.lt.1.d-6)then
        do i=1,4
          qperp(i)=-1.d10
        enddo
      else
        qperp(1)=q1(2)*q2(3)-q1(3)*q2(2)
        qperp(2)=q1(3)*q2(1)-q1(1)*q2(3)
        qperp(3)=q1(1)*q2(2)-q1(2)*q2(1)
        do i=1,3
          qperp(i)=qperp(i)/(xnorm1*xnorm2)
        enddo
        qperp(4)=1.d0
      endif
      return
      end





      subroutine boostwdir2(chybst,shybst,chybstmo,xd,xin,xout)
c chybstmo = chybst-1; if it can be computed analytically it improves
c the numerical accuracy
      implicit none
      real*8 chybst,shybst,chybstmo,xd(1:3),xin(0:3),xout(0:3)
      real*8 tmp,en,pz
      integer i
c
      if(abs(xd(1)**2+xd(2)**2+xd(3)**2-1).gt.1.d-6)then
        write(*,*)'Error #1 in boostwdir2',xd
        stop
      endif
c
      en=xin(0)
      pz=xin(1)*xd(1)+xin(2)*xd(2)+xin(3)*xd(3)
      xout(0)=en*chybst-pz*shybst
      do i=1,3
        xout(i)=xin(i)+xd(i)*(pz*chybstmo-en*shybst)
      enddo
c
      return
      end




      subroutine boostwdir3(chybst,shybst,chybstmo,xd,xxin,xxout)
      implicit none
      real*8 chybst,shybst,chybstmo,xd(1:3),xxin(4),xxout(4)
      real*8 xin(0:3),xout(0:3)
      integer i
c
      do i=1,4
         xin(mod(i,4))=xxin(i)
      enddo
      call boostwdir2(chybst,shybst,chybstmo,xd,xin,xout)
      do i=1,4
         xxout(i)=xout(mod(i,4))
      enddo
c
      return
      end




      subroutine getwedge(p1,p2,pout)
      implicit none
      real*8 p1(4),p2(4),pout(4)

      pout(1)=p1(2)*p2(3)-p1(3)*p2(2)
      pout(2)=p1(3)*p2(1)-p1(1)*p2(3)
      pout(3)=p1(1)*p2(2)-p1(2)*p2(1)
      pout(4)=0d0

      return
      end

