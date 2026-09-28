c
c Example analysis for "p p > h [QCD]" process.
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
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c To be changed !!
      nwgt=1
      weights_info(nwgt)="central value  "
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c Initialize histograms
      call HwU_inithist(nwgt,weights_info)
c Set method for error estimation to '0', i.e., use Poisson statistics
c for the uncertainty estimate
      call set_error_estimation(0)
      nwgt_analysis=nwgt
      do i=1,1
      l=(i-1)*40
      call HwU_book(l+1,'Higgs pT '//HwUtype(i),100,0.d0,200.d0)
      call HwU_book(l+2,'Higgs pT 1 '//HwUtype(i),100,0.d0,500.d0)
      call HwU_book(l+3,'Higgs log[pT] '//HwUtype(i),98,0.1d0,5.d0)
      call HwU_book(l+4,'Higgs pT, y_H<2 '//HwUtype(i),100,0.d0,200.d0)
      call HwU_book(l+5,'Higgs pT 1, y_H<2 '//HwUtype(i),100,0.d0,500.d0)
      call HwU_book(l+6,'Higgs log[pT], y_H<2 '//HwUtype(i),98,0.1d0,5.d0)

      call HwU_book(l+7,'j1 pT '//HwUtype(i),100,0.d0,200.d0)
      call HwU_book(l+8,'j1 pT 1 '//HwUtype(i),100,0.d0,500.d0)
      call HwU_book(l+9,'j1 log[pT] '//HwUtype(i),98,0.1d0,5.d0)
      call HwU_book(l+10,'j1 pT, y_j1<2 '//HwUtype(i),100,0.d0,200.d0)
      call HwU_book(l+11,'j1 pT 1, y_j1<2 '//HwUtype(i),100,0.d0,500.d0)
      call HwU_book(l+12,'j1 log[pT], y_j1<2 '//HwUtype(i),98,0.1d0,5.d0)

      call HwU_book(l+13,'Inc j pT '//HwUtype(i),100,0.d0,200.d0)
      call HwU_book(l+14,'Inc j pT 1 '//HwUtype(i),100,0.d0,500.d0)
      call HwU_book(l+15,'Inc j log[pT] '//HwUtype(i),98,0.1d0,5.d0)
      call HwU_book(l+16,'Inc j pT, y_Ij<2 '//HwUtype(i),100,0.d0,2.d2)
      call HwU_book(l+17,'Inc j pT 1, y_Ij<2 '//HwUtype(i),100,0.d0,5.d2)
      call HwU_book(l+18,'Inc j log[pT], y_Ij<2'//HwUtype(i),98,0.1d0,5.d0)

      call HwU_book(l+19,'Higgs y '//HwUtype(i),60,-6.d0,6.d0)
      call HwU_book(l+20,'Higgs y,pT_H>10GeV '//HwUtype(i),100,-6.d0,6.d0)
      call HwU_book(l+21,'Higgs y,pT_H>30GeV '//HwUtype(i),100,-6.d0,6.d0)
      call HwU_book(l+22,'Higgs y,pT_H>50GeV '//HwUtype(i),100,-6.d0,6.d0)
      call HwU_book(l+23,'Higgs y,pT_H>70GeV '//HwUtype(i),100,-6.d0,6.d0)
      call HwU_book(l+24,'Higgs y,pt_H>90GeV '//HwUtype(i),100,-6.d0,6.d0)

      call HwU_book(l+25,'j1 y '//HwUtype(i),60,-6.d0,6.d0)
      call HwU_book(l+26,'j1 y,pT_j1>10GeV '//HwUtype(i),60,-6.d0,6.d0)
      call HwU_book(l+27,'j1 y,pT_j1>30GeV '//HwUtype(i),60,-6.d0,6.d0)
      call HwU_book(l+28,'j1 y,pT_j1>50GeV '//HwUtype(i),60,-6.d0,6.d0)
      call HwU_book(l+29,'j1 y,pT_j1>70GeV '//HwUtype(i),60,-6.d0,6.d0)
      call HwU_book(l+30,'j1 y,pT_j1>90GeV '//HwUtype(i),60,-6.d0,6.d0)

      call HwU_book(l+31,'H-j1 y '//HwUtype(i),60,-6.d0,6.d0)
      call HwU_book(l+32,'H-j1 y,pT_j1>10GeV '//HwUtype(i),60,-6.d0,6.d0)
      call HwU_book(l+33,'H-j1 y,pT_j1>30GeV '//HwUtype(i),60,-6.d0,6.d0)
      call HwU_book(l+34,'H-j1 y,pT_j1>50GeV '//HwUtype(i),60,-6.d0,6.d0)
      call HwU_book(l+35,'H-j1 y,pT_j1>70GeV '//HwUtype(i),60,-6.d0,6.d0)
      call HwU_book(l+36,'H-j1 y,pT_j1>90GeV '//HwUtype(i),60,-6.d0,6.d0)
      
      call HwU_book(l+37,'njets '//HwUtype(i),11,-0.5d0,10.5d0)
      call HwU_book(l+38,'njets, y_j<2.5 '//HwUtype(i),11,-0.5d0,10.5d0)
      call HwU_book(l+39,'xsec '//HwUtype(i),3,-0.5d0,2.5d0)
      enddo

      END


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
      DOUBLE PRECISION HWVDOT,PSUM(4),PPH(5),XMH,PTH,YH,PTP,YP,
     &getrapidity,getpseudorap,etah,ECUT,PTJ1,PTJ,YJ,
     &PSUB,MJ1,Y,YCUT,YJ1
      INTEGER ICHSUM,ICHINI,IHEP,IFH,IST,ID,IJ,ID1
      LOGICAL DIDSOF
      REAL*8 WWW0
      INTEGER KK
      INTEGER NN,NMAX,I,J,NJ
      PARAMETER (NMAX=2000)
      INTEGER NJET,JET(NMAX),IPOS(NMAX),njet_central
      DOUBLE PRECISION PALG,RFJ,SYCUT,PP(4,NMAX),PJET(4,NMAX),
     # PTJET(NMAX),ETAJET(NMAX),YJET(NMAX),pjet_new(4,nmax),
     # njdble,njcdble,y_central
      integer nwgt_analysis,max_weight,l
      common/c_analysis/nwgt_analysis
      integer maxRWGT
      parameter (maxRWGT=100)
      double precision wgtxsecRWGT(maxRWGT)
      parameter (max_weight=maxscales*maxscales+maxpdfs+maxRWGT+1)
      double precision ww(max_weight),www(max_weight)
      common/cww/ww
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c To be changed !!
      ww(1)=1d0
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
      IF (WW(1).EQ.0D0) THEN
         WRITE(*,*)'WW(1) = 0. Stopping'
         STOP
      ENDIF
C INITIALISE
      DO I=1,NMAX
        DO J=1,4
          PP(J,I)=0D0
        ENDDO
      ENDDO
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
      DIDSOF=.FALSE.
      IFH=0
      NN=0
      DO 100 IHEP=1,NHEP
        IST=ISTHEP(IHEP)      
        ID1=IDHEP(IHEP)
        IF(ID1.EQ.25)THEN
          IFH=1
          DO IJ=1,5
            PPH(IJ)=PHEP(IJ,IHEP)
          ENDDO
        ENDIF
C---FIND FINAL STATE HADRONS
        IF (IST.EQ.1 .AND. ABS(ID1).GT.100) THEN
          NN=NN+1
          IF (NN.GT.NMAX)THEN
            WRITE(*,*)'TOO MANY PARTICLES!'
            STOP
          ENDIF
          DO I=1,4
             PP(I,NN)=PHEP(I,IHEP)
          ENDDO
        ENDIF
 100  CONTINUE
      IF(IFH.NE.1)THEN
         WRITE(*,*)'WARNING 503 IN HWANAL'
         GOTO 999
      ENDIF
C---CLUSTER THE EVENT
      palg =1.d0
      rfj  =0.7d0
      sycut=10d0
      do i=1,nmax
        do j=1,4
          pjet(j,i)=0d0
        enddo
        ptjet(i)=0d0
        yjet(i)=0d0
        jet(i)=0
      enddo
      njet=-1
      njet_central=0
      y_central=2.5d0
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
         yjet(i)=getrapidity(pjet(4,i),pjet(3,i))
         if(abs(yjet(i)).le.y_central)njet_central=njet_central+1
      enddo

C FILL THE HISTOS
c Higgs variables
      pth=sqrt(pph(1)**2+pph(2)**2)
      yh=getrapidity(pph(4),pph(3))
c hardest jet variables
      ptj1=ptjet(1)
      yj1=yjet(1)
c
      njdble=dble(njet)
      njcdble=dble(njet_central)
C
      do i=1,1
      l=(i-1)*40
      call HwU_fill(l+1,pth,WWW)
      call HwU_fill(l+2,pth,WWW)
      if(pth.gt.0.d0)call HwU_fill(l+3,log10(pth),WWW)
      if(abs(yh).le.2.d0)then
         call HwU_fill(l+4,pth,WWW)
         call HwU_fill(l+5,pth,WWW)
         if(pth.gt.0.d0)call HwU_fill(l+6,log10(pth),WWW)
      endif
c
      if(njet.ge.1)then
         call HwU_fill(l+7,ptj1,WWW)
         call HwU_fill(l+8,ptj1,WWW)
         if(ptj1.gt.0.d0)call HwU_fill(l+9,log10(ptj1),WWW)
         if(abs(yj1).le.2.d0)then
            call HwU_fill(l+10,ptj1,WWW)
            call HwU_fill(l+11,ptj1,WWW)
            if(ptj1.gt.0.d0)call HwU_fill(l+12,log10(ptj1),WWW)
         endif
c
         do nj=1,njet
            call HwU_fill(l+13,ptj1,WWW)
            call HwU_fill(l+14,ptj1,WWW)
            if(ptj1.gt.0.d0)call HwU_fill(l+15,log10(ptj1),WWW)
            if(abs(yj1).le.2.d0)then
               call HwU_fill(l+16,ptj1,WWW)
               call HwU_fill(l+17,ptj1,WWW)
               if(ptj1.gt.0d0)call HwU_fill(l+18,log10(ptj1),WWW)
            endif
         enddo
      endif
c
      call HwU_fill(l+19,yh,WWW)
      if(pth.ge.10.d0) call HwU_fill(l+20,yh,WWW)
      if(pth.ge.30.d0) call HwU_fill(l+21,yh,WWW)
      if(pth.ge.50.d0) call HwU_fill(l+22,yh,WWW)
      if(pth.ge.70.d0) call HwU_fill(l+23,yh,WWW)
      if(pth.ge.90.d0) call HwU_fill(l+24,yh,WWW)  
c     
      if(njet.ge.1)then
         call HwU_fill(l+25,yj1,WWW)
         if(ptj1.ge.10.d0) call HwU_fill(l+26,yj1,WWW)
         if(ptj1.ge.30.d0) call HwU_fill(l+27,yj1,WWW)
         if(ptj1.ge.50.d0) call HwU_fill(l+28,yj1,WWW)
         if(ptj1.ge.70.d0) call HwU_fill(l+29,yj1,WWW)
         if(ptj1.ge.90.d0) call HwU_fill(l+30,yj1,WWW)
c     
         call HwU_fill(l+31,yh-yj1,WWW)
         if(ptj1.ge.10.d0) call HwU_fill(l+32,yh-yj1,WWW)
         if(ptj1.ge.30.d0) call HwU_fill(l+33,yh-yj1,WWW)
         if(ptj1.ge.50.d0) call HwU_fill(l+34,yh-yj1,WWW)
         if(ptj1.ge.70.d0) call HwU_fill(l+35,yh-yj1,WWW)
         if(ptj1.ge.90.d0) call HwU_fill(l+36,yh-yj1,WWW)
      endif
c
      call HwU_fill(l+37,njdble,WWW)
      call HwU_fill(l+38,njcdble,WWW)
      call HwU_fill(l+39,1d0,WWW)
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
