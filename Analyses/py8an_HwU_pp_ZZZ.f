c
c Example analysis for "p p > z z z [QCD]" process.
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
      real*8 xmi,xms
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

      xmi=40.d0
      xms=120.d0

      do i=1,1
       l=(i-1)*26
        call HwU_book(l+ 1,'total rate    ',1,0.5d0,1.5d0)
        call HwU_book(l+ 2,'z1 pt     ',20,0.d0,1000.d0)
        call HwU_book(l+ 3,'z1 log pt',20,log10(0.1d0),log10(1000d0))
        call HwU_book(l+ 4,'z1 y      ', 20,-9.d0,9.d0)
        call HwU_book(l+ 5,'z1 eta    ', 20,-9.d0,9.d0)
        call HwU_book(l+ 6,'mz1       ', 80,xmi,xms)
        call HwU_book(l+ 7,'z2 pt     ',20,0.d0,1000d0)
        call HwU_book(l+ 8,'z2 log pt',20,log10(0.1d0),log10(1000d0))
        call HwU_book(l+ 9,'z2 y      ', 20,-9.d0,9.d0)
        call HwU_book(l+ 10,'z2 eta    ', 20,-9.d0,9.d0)
        call HwU_book(l+ 11,'mWz2       ', 80,xmi,xms)
        call HwU_book(l+ 12,'z3 pt     ',20,0.d0,1000.d0)
        call HwU_book(l+ 13,'z3 log pt',20,log10(0.1d0),log10(1000d0))
        call HwU_book(l+ 14,'z3 y      ', 20,-9.d0,9.d0)
        call HwU_book(l+ 15,'z3 eta    ', 20,-9.d0,9.d0)
        call HwU_book(l+ 16,'mz3       ', 80,xmi,xms)
        call HwU_book(l+ 17,'inv12       ', 20,0d0,1000d0)
        call HwU_book(l+ 18,'log inv12 ', 20,log10(0.1d0),log10(1000d0))
        call HwU_book(l+ 19,'inv23      ', 20,0d0,1000d0)
        call HwU_book(l+ 20,'log inv23 ',20,log10(0.1d0),log10(1000d0))
        call HwU_book(l+ 21,'inv13       ', 20,0d0,1000d0)
        call HwU_book(l+ 22,'log inv13 ', 20,log10(0.1d0),log10(1000d0))
        call HwU_book(l+ 23,'pt jet       ', 20,0d0,1000d0)
        call HwU_book(l+ 24,'rap jet       ', 20,-9d0,9d0)
        call HwU_book(l+ 25,'pt z1+j       ', 20,0d0,1000d0)
        call HwU_book(l+ 26,'inv z1+j       ', 20,0d0,1000d0)

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
     # XPNB(5),XPBB(5),p_t(0:3),p_tx(0:3),pttx(0:3),
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

      integer i_max1,i_max2,i_max3
      double precision pz1(0:4),xmz1,ptz1,yz1,etaz1,pT,pT_max
      double precision pz2(0:4),xmz2,ptz2,yz2,etaz2
      double precision pz3(0:4),xmz3,ptz3,yz3,etaz3
      double precision pz12(0:4),pz23(0:4),pz13(0:4),pz123(0:4)
      double precision inv12,inv13,inv23,inv123
      integer :: IST1

      PARAMETER (NMAX=2000)
      INTEGER NJET,JET(NMAX),IPOS(NMAX),njet_central,NN
      DOUBLE PRECISION PALG,RFJ,SYCUT,PP(4,NMAX),PJET(4,NMAX),
     # PTJET(NMAX),ETAJET(NMAX),YJET(NMAX),pjet_new(4,nmax),
     # njdble,njcdble,y_central

      double precision pt_j,pt_zj,m_zj,y_j

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
      pT_max = 0D0
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

      DO IHEP=1,NHEP
C UNCOMMENT THE FOLLOWING WHEN REMOVING THE CHECK ON MOMENTUM 
C        IF(IQ1*IQ2.EQ.1) GOTO 11
        IST=ISTHEP(IHEP)      
        ID1=IDHEP(IHEP)
        IF((ID1.EQ.23) .and. (IST.eq.1)) THEN
           pT=phep(1,IHEP)**2 + phep(2,IHEP)**2
           if (pT .ge. pT_max) then
             pT_max = pT
             i_max1 = IHEP
             IST1=IST
           endif
        ENDIF
        IF(ABS(ID1).GT.100.AND.IST.EQ.1) THEN
           NN=NN+1
           IF (NN.GT.NMAX) STOP 'Too many particles [hadrons]!'
           DO I=1,4
              PP(I,NN)=PHEP(I,IHEP)
           ENDDO
        ENDIF

      ENDDO

      do mu=1,4
        pz1(mu) = phep(mu,i_max1)
      enddo
      pT_max = 0D0
      do IHEP=1,NHEP
      IST=ISTHEP(IHEP)
      ID1=IDHEP(IHEP)
      if ((ID1.eq.23) .and. (IHEP .ne. i_max1) .and. (IST.eq.1)) then
           pT=phep(1,IHEP)**2 + phep(2,IHEP)**2
           if (pT .ge. pT_max) then
             pT_max = pT
             i_max2 = IHEP
             IST1=IST
           endif
      endif
      enddo
      do mu=1,4
        pz2(mu) = phep(mu,i_max2)
      enddo

      do i=1,NHEP
      IST=ISTHEP(i)
      ID1=IDHEP(i)
      if ((ID1.eq.23) .and. 
     $    (i .ne. i_max1) .and. (i .ne. i_max2).and. (IST.eq.1)) then
             i_max3 = i
             IST1=IST
      endif
      enddo

      do mu=1,4
        pz3(mu) = phep(mu,i_max3)
      enddo

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

      xmz1=sqrt(max(pz1(4)**2-pz1(1)**2-pz1(2)**2-pz1(3)**2,0d0))
      ptz1=sqrt(max(pz1(1)**2+pz1(2)**2,0d0))
      yz1=getrapidity(pz1(4),pz1(3))
      etaz1=getpseudorap(pz1(4),pz1(1),pz1(2),pz1(3))

      xmz2=sqrt(max(pz2(4)**2-pz2(1)**2-pz2(2)**2-pz2(3)**2,0d0))
      ptz2=sqrt(max(pz2(1)**2+pz2(2)**2,0d0))
      yz2=getrapidity(pz2(4),pz2(3))
      etaz2=getpseudorap(pz2(4),pz2(1),pz2(2),pz2(3))

      xmz3=sqrt(max(pz3(4)**2-pz3(1)**2-pz3(2)**2-pz3(3)**2,0d0))
      ptz3=sqrt(max(pz3(1)**2+pz3(2)**2,0d0))
      yz3=getrapidity(pz3(4),pz3(3))
      etaz3=getpseudorap(pz3(4),pz3(1),pz3(2),pz3(3))

      pz12=pz1+pz2
      pz23=pz2+pz3
      pz13=pz1+pz3
      pz123=pz1+pz2+pz3
      inv12=sqrt(max(pz12(4)**2-pz12(1)**2-pz12(2)**2-pz12(3)**2,0d0))
      inv23=sqrt(max(pz23(4)**2-pz23(1)**2-pz23(2)**2-pz23(3)**2,0d0))
      inv13=sqrt(max(pz13(4)**2-pz13(1)**2-pz13(2)**2-pz13(3)**2,0d0))
      inv123=sqrt(max(pz123(4)**2-pz123(1)**2-pz123(2)**2-pz123(3)**2,0d0))

      pt_j=sqrt(pjet(1,1)**2+pjet(2,1)**2)
      pt_zj=sqrt((pjet(1,1)+pz1(1))**2+(pjet(2,1)+pz1(1))**2)
      y_j=getrapidity(pjet(4,1),pjet(3,1))
      m_zj=getinvm(pjet(4,1)+pz1(4),pjet(1,1)+pz1(1),pjet(2,1)+pz1(2),pjet(3,1)+pz1(3))

      l=0
      var=1.0
         call HwU_fill(l+1,var,WWW)
         call HwU_fill(l+2,ptz1,WWW)
         if(ptz1.gt.0) call HwU_fill(l+3,log10(ptz1),WWW)
         call HwU_fill(l+4,yz1,WWW)
         call HwU_fill(l+5,etaz1,WWW)
         call HwU_fill(l+6,xmz1,WWW)

         call HwU_fill(l+7,ptz2,WWW)
         if(ptz2.gt.0) call HwU_fill(l+8,log10(ptz2),WWW)
         call HwU_fill(l+9,yz2,WWW)
         call HwU_fill(l+10,etaz2,WWW)
         call HwU_fill(l+11,xmz2,WWW)

         call HwU_fill(l+12,ptz3,WWW)
         if(ptz3.gt.0) call HwU_fill(l+13,log10(ptz3),WWW)
         call HwU_fill(l+14,yz3,WWW)
         call HwU_fill(l+15,etaz3,WWW)
         call HwU_fill(l+16,xmz3,WWW)

         call HwU_fill(l+17,inv12,WWW)
         if(inv12.gt.0) call HwU_fill(l+18,log10(inv12),WWW)
         call HwU_fill(l+19,inv23,WWW)
         if(inv23.gt.0) call HwU_fill(l+20,log10(inv23),WWW)
         call HwU_fill(l+21,inv13,WWW)
         if(inv13.gt.0) call HwU_fill(l+22,log10(inv13),WWW)

         call HwU_fill(l+23,pt_j,WWW)
         call HwU_fill(l+24,y_j,WWW)
         call HwU_fill(l+25,pt_zj,WWW)
         call HwU_fill(l+26,m_zj,WWW)

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

