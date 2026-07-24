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
      SUBROUTINE PYABEG(nnn,wwwi)
C     USER''S ROUTINE FOR INITIALIZATION
C----------------------------------------------------------------------
      use HwU_wgts_info_len
      INCLUDE 'HEPMC.INC'
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
      integer j,kk,l,jpr,i,nnn
      character*5 cc(2)
      data cc/'     ','Born '/
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
       l=(i-1)*16
       call HwU_book(l+1,'total rate      '//HwUtype(i),1,0.5d0,1.5d0)
       call HwU_book(l+2,'pt Z            '//HwUtype(i),20,10d0,1500d0)
       call HwU_book(l+3,'log[pt] Z'//HwUtype(i),20,log10(10d0),log10(1500d0))
       call HwU_book(l+4,'y Z             '//HwUtype(i),20,-9.d0,9.d0)
       call HwU_book(l+5,'eta Z           '//HwUtype(i),20,-9.d0,9.d0)
       call HwU_book(l+6,'m Z-j1',      20,0d0,2500d0)
       call HwU_book(l+7,'m Z-j2',      20,0d0,2000d0)
       call HwU_book(l+8,'pT Z-j1',      20,0d0,1500d0)
       call HwU_book(l+9,'pT Z-j2',      20,0d0,1500d0)
       call HwU_book(l+10,'pTZ/(pTZ+pTj1+pTj2)',      20,0d0,1d0)
       call HwU_book(l+11,'dr(Z,j1)',    20,0d0,15d0)
       call HwU_book(l+12,'dr(Z,j2)',    20,0d0,15d0)
       call HwU_book(l+13,'pt(j1) ',    20,log10(200d0),log10(3000d0))
       call HwU_book(l+14,'pt(j2) ',    20,log10(200d0),log10(3000d0))
       call HwU_book(l+15,'m j1-j2',      20,0d0,2500d0)
       call HwU_book(l+16,'dr(j1,j2)',    20,0d0,10d0)
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
      DOUBLE PRECISION HWVDOT,PSUM(4),XME,PPV(5),PPE(5),PPNU(5),
     # PPDCE(5),PPDCNU(5),WT,ETAEMIN(2),ETAEMAX(2),PTEMIN(2),
     # XMV,PTV,YV,GETRAPIDITY,PTE,THE,ETAE,PTNU,THNU,ETANU,
     # PTDCE,THDCE,ETADCE,PTDCNU,THDCNU,ETADCNU,ETAV,GETPSEUDORAP
      INTEGER ICHSUM,ICHINI,IHEP,JPR,IDENT,IFV,IST,ID,ID1,IHRD,IV,
     # IJ,IE,INU,J
      LOGICAL DIDSOF,FLAG
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
      double precision ww(max_weight),www(max_weight),xww(max_weight)
      common/cww/ww
c
      integer NMAX
      PARAMETER (NMAX=2000)
      INTEGER NJET,JET(NMAX),IPOS(NMAX),njet_central,NN
      DOUBLE PRECISION PALG,RFJ,SYCUT,PP(4,NMAX),PJET(4,NMAX),
     # PTJET(NMAX),ETAJET(NMAX),YJET(NMAX),pjet_new(4,nmax),
     # njdble,njcdble,y_central

      double precision m_vj1,m_vj2, dr_vj1,dr_vj2, dr_j1j2,m_j1j2,getdr
      double precision p_vj1(1:4),p_vj2(1:4)
      double precision p_j1j2(1:4), pt_ratio
      double precision var

      logical pass_jets

      if(nnn.eq.0)ww(1)=1d0
      do i=1,nnn
         ww(i)=xww(i)
      enddo

c
      IF (WW(1).EQ.0D0) THEN
         WRITE(*,*)'WW(1) = 0. Stopping'
!         STOP
      ENDIF

c
c CHOOSE IDENT=24 FOR W+, IDENT=-24 FOR W-, IDENT=23 FOR Z0
      IDENT=23
C INCOMING PARTONS MAY TRAVEL IN THE SAME DIRECTION: IT''S A POWER-SUPPRESSED
C EFFECT, SO THROW THE EVENT AWAY
      IF(SIGN(1.D0,PHEP(3,1)).EQ.SIGN(1.D0,PHEP(3,2)))THEN
         WRITE(*,*)'WARNING 502 IN PYANAL'
         GOTO 999
      ENDIF
      DO I=1,nwgt_analysis
         WWW(I)=EVWGT*ww(i)/ww(1)
      ENDDO
      ICHSUM=0
      DIDSOF=.FALSE.
      IFV=0
      NN=0
      DO 100 IHEP=1,NHEP
        IST=ISTHEP(IHEP)      
        ID1=IDHEP(IHEP)
        IF(ID1.EQ.IDENT)THEN
          IV=IHEP
          IFV=1
          DO IJ=1,5
             PPV(IJ)=PHEP(IJ,IHEP)
          ENDDO
        ENDIF
        IF(ABS(ID1).GT.100.AND.IST.EQ.1) THEN
           NN=NN+1
           IF (NN.GT.NMAX) STOP 'Too many particles [hadrons]!'
           DO I=1,4
              PP(I,NN)=PHEP(I,IHEP)
           ENDDO
        ENDIF
  100 CONTINUE
      IF(IFV.NE.1) THEN
         WRITE(*,*)'WARNING 503 IN PYANAL'
         GOTO 999
      ENDIF

C---CLUSTER THE EVENT
      palg = 1d0
      rfj  = 0.4d0
      sycut= 30d0
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
      enddo
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

      pass_jets=.false.
      if ((njet .gt. 0) .and. (ptjet(1) .ge. 200d0)) then
        pass_jets = .true.
      endif

      DO IJ=1,4
         if(njet.gt.0) p_vj1(IJ)=PPV(IJ)+pjet(IJ,1)
         if(njet.gt.1) p_vj2(IJ)=PPV(IJ)+pjet(IJ,2)
         if(njet.gt.1) p_j1j2(IJ)=pjet(IJ,1)+pjet(IJ,2)
      ENDDO

      if (njet.gt.0) then
        m_vj1 = getinvm(p_vj1(4),p_vj1(1),p_vj1(2),p_vj1(3))
      endif

      if (njet.gt.1) then
        m_vj2 = getinvm(p_vj2(4),p_vj2(1),p_vj2(2),p_vj2(3))
      endif

      dr_vj1=0d0
      dr_vj2=0d0
      if (njet.gt.0) then
      dr_vj1 = getdr(PPV(4),PPV(1),PPV(2),PPV(3)
     $             ,pjet(4,1),pjet(1,1),pjet(2,1),pjet(3,1))
      pt_vj1 = sqrt(p_vj1(1)**2+p_vj1(2)**2)
      endif
      if (njet.gt.1) then
      dr_vj2 = getdr(PPV(4),PPV(1),PPV(2),PPV(3)
     $             ,pjet(4,2),pjet(1,2),pjet(2,2),pjet(3,2))
      pt_vj2 = sqrt(p_vj2(1)**2+p_vj2(2)**2)
      endif

      if (njet.gt.1) then
      dr_j1j2 = getdr(pjet(4,1),pjet(1,1),pjet(2,1),pjet(3,1)
     $             ,pjet(4,2),pjet(1,2),pjet(2,2),pjet(3,2))
      m_j1j2 = getinvm(p_j1j2(4),p_j1j2(1),p_j1j2(2),p_j1j2(3))
      endif

      if (njet.gt.1) pt_ratio = ptV/(pTV + ptjet(1) + ptjet(2))

C FILL THE HISTOS
C Variables of the vector boson
      xmv=ppv(5)
      ptv=sqrt(ppv(1)**2+ppv(2)**2)
      yv=getrapidity(ppv(4),ppv(3))
      etav=getpseudorap(ppv(4),ppv(1),ppv(2),ppv(3))
C
      var =1.0d0
      if (pass_jets) then
       do i=1,1
        l=(i-1)*16
        call HwU_fill(l+1,var,WWW)
        call HwU_fill(l+2,ptv,WWW)
        if(ptv.gt.0) call HwU_fill(l+3,log10(ptv),WWW)
        call HwU_fill(l+4,yv,WWW)
        call HwU_fill(l+5,etav,WWW)
        
        call HwU_fill(l+6,m_vj1,WWW)
        if(njet.gt.1) call HwU_fill(l+7,m_vj2,WWW)
        call HwU_fill(l+8,pt_vj1,WWW)
        if(njet.gt.1) call HwU_fill(l+9,pt_vj2,WWW)
        if(njet.gt.1) call HwU_fill(l+10,pt_ratio,WWW)
        call HwU_fill(l+11,dr_vj1,WWW)
        if(njet.gt.1) call HwU_fill(l+12,dr_vj2,WWW)
        call HwU_fill(l+13,dlog10(ptjet(1)),WWW)
        if(njet.gt.1) call HwU_fill(l+14,dlog10(ptjet(2)),WWW)
        if(njet.gt.1) call HwU_fill(l+15,m_j1j2,WWW)
        if(njet.gt.1) call HwU_fill(l+16,dr_j1j2,WWW)
      enddo
      call HwU_add_points
      endif



C
 999  END

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
