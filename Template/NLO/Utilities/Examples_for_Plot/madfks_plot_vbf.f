c
c
c Plotting routines
c
c
      subroutine initplot
      implicit none
c Book histograms in this routine. Use bookup or bookup. The entries
c of these routines are real*8
      include 'run.inc'
      real*8 pi
      PARAMETER (PI=3.14159265358979312D0)
      integer i,kk
      character*7 cc(2)
      real*8 vetomin, vetomax
      integer nbinveto
      common /to_veto_hist/vetomin,vetomax,nbinveto
      
      cc(1)='       '
      cc(2)='vbfcuts'
      call inihist
      vetomin = 0d0
      vetomax = 100d0
      nbinveto = 50

      do i=1,2
         kk=(i-1)*100
         call bookup(kk+  1,'total rate    '//cc(i),1.0d0,0.5d0,5.5d0)

         call bookup(kk+  2,'Higgs pT      '//cc(i),8.0d0,0.d0,400.d0)
         call bookup(kk+  3,'Higgs pT      '//cc(i),16.0d0,0.d0,800.d0)
         call bookup(kk+  4,'Higgs log pT  '//cc(i),0.08d0,0.d0,4.d0)
         call bookup(kk+  5,'Higgs eta     '//cc(i),0.24d0,-6.d0,6.d0)
         call bookup(kk+  6,'Higgs y       '//cc(i),0.24d0,-6.d0,6.d0)

         call bookup(kk+  7,'j1 pT         '//cc(i),8.0d0,0.d0,400.d0)
         call bookup(kk+  8,'j1 pT         '//cc(i),16.0d0,0.d0,800.d0)
         call bookup(kk+  9,'j1 log pT     '//cc(i),0.08d0,0.d0,4.d0)
         call bookup(kk+ 10,'j1 eta        '//cc(i),0.24d0,-6.d0,6.d0)
         call bookup(kk+ 11,'j1 y          '//cc(i),0.24d0,-6.d0,6.d0)

         call bookup(kk+ 12,'j2 pT         '//cc(i),8.0d0,0.d0,400.d0)
         call bookup(kk+ 13,'j2 pT         '//cc(i),16.0d0,0.d0,800.d0)
         call bookup(kk+ 14,'j2 log pT     '//cc(i),0.08d0,0.d0,4.d0)
         call bookup(kk+ 15,'j2 eta        '//cc(i),0.24d0,-6.d0,6.d0)
         call bookup(kk+ 16,'j2 y          '//cc(i),0.24d0,-6.d0,6.d0)

         call bookup(kk+ 17,'j3 pT         '//cc(i),8.0d0,0.d0,400.d0)
         call bookup(kk+ 18,'j3 pT         '//cc(i),16.0d0,0.d0,800.d0)
         call bookup(kk+ 19,'j3 log pT     '//cc(i),0.08d0,0.d0,4.d0)
         call bookup(kk+ 20,'j3 eta        '//cc(i),0.24d0,-6.d0,6.d0)
         call bookup(kk+ 21,'j3 y          '//cc(i),0.24d0,-6.d0,6.d0)

         call bookup(kk+ 22,'H+j1 pT       '//cc(i),8.0d0,0.d0,400.d0)
         call bookup(kk+ 23,'H+j1 pT       '//cc(i),16.0d0,0.d0,800.d0)
         call bookup(kk+ 24,'H+j1 log pT   '//cc(i),0.08d0,0.d0,4.d0)
         call bookup(kk+ 25,'H+j1 eta      '//cc(i),0.24d0,-6.d0,6.d0)
         call bookup(kk+ 26,'H+j1 y        '//cc(i),0.24d0,-6.d0,6.d0)

         call bookup(kk+ 27,'j1+j2 pT      '//cc(i),8.0d0,0.d0,400.d0)
         call bookup(kk+ 28,'j1+j2 pT      '//cc(i),16.0d0,0.d0,800.d0)
         call bookup(kk+ 29,'j1+j2 log pT  '//cc(i),0.08d0,0.d0,4.d0)
         call bookup(kk+ 30,'j1+j2 eta     '//cc(i),0.24d0,-6.d0,6.d0)
         call bookup(kk+ 31,'j1+j2 y       '//cc(i),0.24d0,-6.d0,6.d0)

         call bookup(kk+ 32,'syst pT       '//cc(i),8.0d0,0.d0,400.d0)
         call bookup(kk+ 33,'syst pT       '//cc(i),16.0d0,0.d0,800.d0)
         call bookup(kk+ 34,'syst log pT   '//cc(i),0.08d0,0.d0,4.d0)
         call bookup(kk+ 35,'syst eta      '//cc(i),0.4d0,-10.d0,10.d0)
         call bookup(kk+ 36,'syst y        '//cc(i),0.24d0,-6.d0,6.d0)

         call bookup(kk+ 37,'Dphi H-j1     '//cc(i),pi/50.d0,0d0,pi)
         call bookup(kk+ 38,'Dphi H-j2     '//cc(i),pi/50.d0,0d0,pi)
         call bookup(kk+ 39,'Dphi j1-j2    '//cc(i),pi/50.d0,0d0,pi)

         call bookup(kk+ 40,'DR H-j1       '//cc(i),0.2d0,0d0,10.d0)
         call bookup(kk+ 41,'DR H-j2       '//cc(i),0.2d0,0d0,10.d0)
         call bookup(kk+ 42,'DR j1-j2      '//cc(i),0.2d0,0d0,10.d0)

         call bookup(kk+ 43,'mj1j2         '//cc(i),60.0d0,0d0,3000.d0)

c Nason-Oleari plots (hep-ph/0911.5299)
         call bookup(kk+ 44,'|yj1-yj2|     '//cc(i),0.4d0,0.d0,10.d0)
         call bookup(kk+ 45,'yj3_rel       '//cc(i),0.24d0,-6.d0,6.d0)
         call bookup(kk+ 46,'njets         '//cc(i),1.d0,-0.5d0,9.5d0)
         call bookup(kk+ 47,'ptrel_j1      '//cc(i),4.d0,0.d0,200.d0)
         call bookup(kk+ 48,'ptrel_j2      '//cc(i),4.d0,0.d0,200.d0)
         call bookup(kk+ 49,'P-veto        '//cc(i),
     1    ((vetomax-vetomin)/dble(nbinveto)),vetomin,vetomax)
         call bookup(kk+ 50,'jveto pT    '//cc(i),
     1    ((vetomax-vetomin)/dble(nbinveto)),vetomin,vetomax)
         call bookup(kk+ 51,'jveto pT    '//cc(i),
     1    ((2d0*vetomax-vetomin)/dble(nbinveto)),
     1    vetomin,2d0*vetomax)
         call bookup(kk+ 52,'jveto log pT'//cc(i),0.08d0,0.d0,4.d0)
         call bookup(kk+ 53,'jveto eta   '//cc(i),0.24d0,-6.d0,6.d0)
         call bookup(kk+ 54,'jveto y     '//cc(i),0.24d0,-6.d0,6.d0)

      enddo
      return
      end


      subroutine topout
      implicit none
      character*14 ytit
      logical usexinteg,mint
      common/cusexinteg/usexinteg,mint
      integer itmax,ncall
      common/citmax/itmax,ncall
      real*8 xnorm1,xnorm2
      logical unwgt
      double precision evtsgn
      common /c_unwgt/evtsgn,unwgt
      integer i,kk
      include 'dbook.inc'
c
      if (unwgt) then
         ytit='events per bin'
      else
         ytit='sigma per bin '
      endif
      xnorm1=1.d0/float(itmax)
      xnorm2=1.d0/float(ncall*itmax)
      do i=1,NPLOTS
        if(usexinteg.and..not.mint) then
           call mopera(i,'+',i,i,xnorm1,0.d0)
        elseif(mint) then
           call mopera(i,'+',i,i,xnorm2,0.d0)
        endif
        call mfinal(i)
      enddo
      do i=1,2
        kk=(i-1)*100
         call multitop(kk+  1,3,2,'total rate       ',' ','LOG')

         call multitop(kk+  2,3,2,'Higgs pT         ',' ','LOG')
         call multitop(kk+  3,3,2,'Higgs pT         ',' ','LOG')
         call multitop(kk+  4,3,2,'Higgs log pT     ',' ','LOG')
         call multitop(kk+  5,3,2,'Higgs eta        ',' ','LOG')
         call multitop(kk+  6,3,2,'Higgs y          ',' ','LOG')

         call multitop(kk+  7,3,2,'j1 pT            ',' ','LOG')
         call multitop(kk+  8,3,2,'j1 pT            ',' ','LOG')
         call multitop(kk+  9,3,2,'j1 log pT        ',' ','LOG')
         call multitop(kk+ 10,3,2,'j1 eta           ',' ','LOG')
         call multitop(kk+ 11,3,2,'j1 y             ',' ','LOG')

         call multitop(kk+ 12,3,2,'j2 pT            ',' ','LOG')
         call multitop(kk+ 13,3,2,'j2 pT            ',' ','LOG')
         call multitop(kk+ 14,3,2,'j2 log pT        ',' ','LOG')
         call multitop(kk+ 15,3,2,'j2 eta           ',' ','LOG')
         call multitop(kk+ 16,3,2,'j2 y             ',' ','LOG')

         call multitop(kk+ 17,3,2,'j3 pT            ',' ','LOG')
         call multitop(kk+ 18,3,2,'j3 pT            ',' ','LOG')
         call multitop(kk+ 19,3,2,'j3 log pT        ',' ','LOG')
         call multitop(kk+ 20,3,2,'j3 eta           ',' ','LOG')
         call multitop(kk+ 21,3,2,'j3 y             ',' ','LOG')

         call multitop(kk+ 22,3,2,'H+j1 pT          ',' ','LOG')
         call multitop(kk+ 23,3,2,'H+j1 pT          ',' ','LOG')
         call multitop(kk+ 24,3,2,'H+j1 log pT      ',' ','LOG')
         call multitop(kk+ 25,3,2,'H+j1 eta         ',' ','LOG')
         call multitop(kk+ 26,3,2,'H+j1 y           ',' ','LOG')

         call multitop(kk+ 27,3,2,'j1+j2 pT         ',' ','LOG')
         call multitop(kk+ 28,3,2,'j1+j2 pT         ',' ','LOG')
         call multitop(kk+ 29,3,2,'j1+j2 log pT     ',' ','LOG')
         call multitop(kk+ 30,3,2,'j1+j2 eta        ',' ','LOG')
         call multitop(kk+ 31,3,2,'j1+j2 y          ',' ','LOG')

         call multitop(kk+ 32,3,2,'syst pT          ',' ','LOG')
         call multitop(kk+ 33,3,2,'syst pT          ',' ','LOG')
         call multitop(kk+ 34,3,2,'syst log pT      ',' ','LOG')
         call multitop(kk+ 35,3,2,'syst eta         ',' ','LOG')
         call multitop(kk+ 36,3,2,'syst y           ',' ','LOG')

         call multitop(kk+ 37,3,2,'Dphi H-j1        ',' ','LOG')
         call multitop(kk+ 38,3,2,'Dphi H-j2        ',' ','LOG')
         call multitop(kk+ 39,3,2,'Dphi j1-j2       ',' ','LOG')

         call multitop(kk+ 40,3,2,'DR H-j1          ',' ','LOG')
         call multitop(kk+ 41,3,2,'DR H-j2          ',' ','LOG')
         call multitop(kk+ 42,3,2,'DR j1-j2         ',' ','LOG')

         call multitop(kk+ 43,3,2,'mj1j2            ',' ','LOG')

         call multitop(kk+ 44,3,2,'|yj1-yj2|        ',' ','LOG')
         call multitop(kk+ 45,3,2,'yj3_rel          ',' ','LOG')
         call multitop(kk+ 46,3,2,'njets            ',' ','LOG')
         call multitop(kk+ 47,3,2,'ptrel_j1         ',' ','LOG')
         call multitop(kk+ 48,3,2,'ptrel_j2         ',' ','LOG')
         call multitop(kk+ 49,3,2,'P-veto           ',' ','LOG')
         call multitop(kk+ 50,3,2,'jv pT            ',' ','LOG')
         call multitop(kk+ 51,3,2,'jv pT            ',' ','LOG')
         call multitop(kk+ 52,3,2,'jv log pT        ',' ','LOG')
         call multitop(kk+ 53,3,2,'jv eta           ',' ','LOG')
         call multitop(kk+ 54,3,2,'jv y             ',' ','LOG')
      enddo
      return                
      end


      subroutine outfun(pp,ybst_til_tolab,www,itype)
C
C *WARNING**WARNING**WARNING**WARNING**WARNING**WARNING**WARNING**WARNING*
C
C In MadFKS, the momenta PP given in input to this function are in the
C reduced parton c.m. frame. If need be, boost them to the lab frame.
C The rapidity of this boost is
C
C       YBST_TIL_TOLAB
C
C also given in input
C
C This is the rapidity that enters in the arguments of the sinh() and
C cosh() of the boost, in such a way that
C       ylab = ycm - ybst_til_tolab
C where ylab is the rapidity in the lab frame and ycm the rapidity
C in the center-of-momentum frame.
C
C *WARNING**WARNING**WARNING**WARNING**WARNING**WARNING**WARNING**WARNING*
      implicit none
      include 'nexternal.inc'
      real*8 pp(0:3,nexternal),ybst_til_tolab,www
      integer itype
      real*8 var
      real*8 ppevs(0:3,nexternal)

      real*8 getrapidity,getpseudorap,chybst,shybst,chybstmo
      real*8 xd(1:3)
      data (xd(i),i=1,3)/0,0,1/
      real*8 pplab(0:3,nexternal)
      double precision ppcl(4,nexternal),y(nexternal)
      double precision pjet(0:3,nexternal)
      double precision cthjet(nexternal)
      integer nn,njet,nsub,jet(nexternal)
      real*8 emax,getcth,cpar,dpar,thrust,dot,shat
      double precision pt, eta, rap2, delta_phi, r2
      integer i,j,kk,imax,k

      LOGICAL  IS_A_J(NEXTERNAL),IS_A_LP(NEXTERNAL),IS_A_LM(NEXTERNAL)
      COMMON /TO_SPECISA/IS_A_J,IS_A_LP,IS_A_LM
c masses
      double precision pmass(nexternal)
      double precision  pt1, eta1, y1, pt2, eta2, y2, pt3, eta3, y3, ht

      common/to_mass/pmass
      double precision ppkt(0:3,nexternal),ecut,djet
      double precision ycut, palg
      integer ij1y, ij2y, ij3y
      integer ipartjet(nexternal)
      double precision ph1(0:3), p12(0:3), psys(0:3)

      real*8 vetomin, vetomax
      integer nbinveto
      common /to_veto_hist/vetomin,vetomax,nbinveto
      include 'cuts.inc'
      double precision rfj,sycut
      double precision pqcd(0:3, nexternal)
      integer nqcd, njet_eta
c
      double precision chy1,chy1mo,chy2,chy2mo,deltay12,DphiHj1,DphiHj2,
     &Dphij1j2,DRHj1,DRHj2,DRj1j2,Dyj1j2,etaH,etaHj,etaj1,etaj2,etaj3,
     &etajj,etasyst,mj1j2,mj1j2min,njdble,ptH,ptHj,pH(0:3),pHj(0:3),
     &pj1(0:3),pj1boost(0:3),pj2(0:3),pj2boost(0:3),pj3(0:3),ptj3,
     &pjj(0:3),ptjj,pjveto(0:3),pt_veto,prel_j1(0:3),prel_j2(0:3),
     &psyst(0:3),ptj1,ptj2,ptj_tag,ptrel_j1,ptrel_j2,ptsyst,shy1,shy2,
     &xsecup2,yH,yHj,yj1,yj2,yj3,yj3rel,yjj,yjmax,ysyst,ptjet(nexternal),
     &yjet(nexternal),getptv4,getrapidityv4,getpseudorapv4,getdelphiv4,
     &getdrv4,getinvmv4,ppboost(0:3,nexternal),getmod,etajet(nexternal)
      logical flag,pass_tag_cuts
      integer ij1,ij2,ij3,ijveto,ijvetoy,njety,nmax
c
      kk=0
      if(itype.eq.11.or.itype.eq.12)then
        continue
      elseif(itype.eq.20)then
        return
      else
        write(*,*)'Error in outfun: unknown itype',itype
        stop
      endif

      chybst=cosh(ybst_til_tolab)
      shybst=sinh(ybst_til_tolab)
      chybstmo=chybst-1.d0
      do i=1,nexternal
        call boostwdir2(chybst,shybst,chybstmo,xd,
     #                  pp(0,i),pplab(0,i))
      enddo

c Put all (light) QCD partons in momentum array for jet clustering.
c From the run_card.dat, maxjetflavor defines if b quark should be
c considered here (via the logical variable 'is_a_jet').  nQCD becomes
c the number of (light) QCD partons at the real-emission level (i.e. one
c more than the Born).
        do j = 0, nexternal
           do i = 0, 3
             pQCD(i,j)=0d0
           enddo
        enddo
       
         nQCD=0
         do j=nincoming+1,nexternal
            if (is_a_j(j)) then
               nQCD=nQCD+1
               do i=0,3
                  pQCD(i,nQCD)=pplab(i,j) 
               enddo
            endif
         enddo

C---CLUSTER THE EVENT
      palg =-1.d0
      rfj  =0.4d0 
      sycut=20d0
      yjmax=4.5d0
      do i=1,nexternal
         do j=0,3
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


c******************************************************************************
c     call FASTJET to get all the jets
c
c     INPUT:
c     input momenta:               pQCD(0:3,nexternal), energy is 0th component
c     number of input momenta:     nQCD
c     radius parameter:            rfj
c     minumum jet pt:              sycut
c     jet algorithm:               palg, 1.0=kt, 0.0=C/A, -1.0 = anti-kt
c
c     OUTPUT:
c     jet momenta:                             pjet(0:3,nexternal), E is 0th cmpnt
c     the number of jets (with pt > SYCUT):    njet
c     the jet for a given particle 'i':        jet(i),   note that this is
c     the particle in pQCD, which doesn't necessarily correspond to the particle
c     label in the process
c
         call fastjetppgenkt(pQCD,nQCD,rfj,sycut,palg,pjet,njet,jet)
c
c******************************************************************************
      do i=1,njet
         ptjet(i)=getptv4(pjet(0,i))
         if(i.gt.1)then
            if (ptjet(i).gt.ptjet(i-1)) then
               write (*,*) "Error 1: jets should be ordered in pt"
               stop
            endif
         endif
         yjet(i)=getrapidityv4(pjet(0,i))
         etajet(i)=getpseudorapv4(pjet(0,i))
c look for veto jet without y cuts
         if (i.gt.2.and.yjet(i).gt.min(yjet(1),yjet(2)).and.
     &      yjet(i).lt.max(yjet(1),yjet(2)).and.ijveto.eq.0) ijveto = i
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
      kk=(i-1)*100

      if(i.eq.1) then 
         ij1 = 1
         ij2 = 2
         ij3 = 3
      endif

      if(i.eq.2) then
         njet = njety
         ijveto = ijvetoy
      endif

c Load momenta
         xsecup2=1d0
         do k=0,3
             pH(k)   =pplab(k,3)
             pj1(k)  =pjet(k,ij1)
             pj2(k)  =pjet(k,ij2)
             pj3(k)  =pjet(k,ij3)
             pjj(k)  =pjet(k,ij1)+pjet(k,ij2)
             pHj(k)  =pjet(k,ij1)+pH(k)
             psyst(k)=pjet(k,ij1)+pjet(k,ij2)+pH(k)
             pjveto(k)=pjet(k,ijveto)
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


      do j=1,nQCD
         if(njet.ge.1.and.jet(j).eq.1)then
           call boostwdir3(chy1,shy1,chy1mo,xd,pQCD(0,j),ppboost(0,j))
           call getwedge(ppboost(0,j),pj1boost,prel_j1)
           ptrel_j1=ptrel_j1+getmod(prel_j1)/getmod(pj1boost)
         elseif(njet.ge.2.and.jet(j).eq.2)then
           call boostwdir3(chy2,shy2,chy2mo,xd,pQCD(0,j),ppboost(0,j))
           call getwedge(ppboost(0,j),pj2boost,prel_j2)
           ptrel_j2=ptrel_j2+getmod(prel_j2)/getmod(pj2boost)
         endif
      enddo

         if(flag)then
            call mfill(kk+  1,(1d0),(www))
            call mfill(kk+  2,(ptH),(www))
            call mfill(kk+  3,(ptH),(www))
            if(ptH.gt.0d0)
     &      call mfill(kk+  4,(log10(ptH)),(www))
            call mfill(kk+  5,(etaH),(www))
            call mfill(kk+  6,(yH),(www))
            call mfill(kk+ 46,(njdble),(www))
cc            call mfill(kk+ 48,(temp_scalup),(www))
            
            if(njet.ge.1)then
               call mfill(kk+  7,(ptj1),(www))
               call mfill(kk+  8,(ptj1),(www))
               call mfill(kk+  9,(log10(ptj1)),(www))
               call mfill(kk+ 10,(etaj1),(www))
               call mfill(kk+ 11,(yj1),(www))
               call mfill(kk+ 22,(ptHj),(www))
               call mfill(kk+ 23,(ptHj),(www))
               if(ptHj.gt.0d0)
     &         call mfill(kk+ 24,(log10(ptHj)),(www))
               call mfill(kk+ 25,(etaHj),(www))
               call mfill(kk+ 26,(yHj),(www))
               call mfill(kk+ 37,(DphiHj1),(www))
               call mfill(kk+ 40,(DRHj1),(www))
               call mfill(kk+ 47,(ptrel_j1),(www))
            endif 

            if(njet.ge.2)then
               call mfill(kk+ 12,(ptj2),(www))
               call mfill(kk+ 13,(ptj2),(www))
               call mfill(kk+ 14,(log10(ptj2)),(www))
               call mfill(kk+ 15,(etaj2),(www))
               call mfill(kk+ 16,(yj2),(www))
               call mfill(kk+ 27,(ptjj),(www))
               call mfill(kk+ 28,(ptjj),(www))
               if(ptjj.gt.0d0)
     &         call mfill(kk+ 29,(log10(ptjj)),(www))
               call mfill(kk+ 30,(etajj),(www))
               call mfill(kk+ 31,(yjj),(www))
               call mfill(kk+ 32,(ptsyst),(www))
               call mfill(kk+ 33,(ptsyst),(www))
               if(ptsyst.gt.0d0)
     &         call mfill(kk+ 34,(log10(ptsyst)),(www))
               call mfill(kk+ 35,(etasyst),(www))
               call mfill(kk+ 36,(ysyst),(www))
               call mfill(kk+ 38,(DphiHj2),(www))
               call mfill(kk+ 39,(Dphij1j2),(www))
               call mfill(kk+ 41,(DRHj2),(www))
               call mfill(kk+ 42,(DRj1j2),(www))
               call mfill(kk+ 43,(mj1j2),(www))
               call mfill(kk+ 44,(Dyj1j2),(www))
               call mfill(kk+ 48,(ptrel_j2),(www))
            endif

            if(njet.ge.3)then
               call mfill(kk+ 17,(ptj3),(www))
               call mfill(kk+ 18,(ptj3),(www))
               call mfill(kk+ 19,(log10(ptj3)),(www))
               call mfill(kk+ 20,(etaj3),(www))
               call mfill(kk+ 21,(yj3),(www))
               call mfill(kk+ 45,(yj3rel),(www))
            endif
            if (ijveto.gt.0) then
                pt_veto = getptv4(pjveto)
                do k=1,nbinveto
                   if (pt_veto.gt.
     &          (vetomin+(vetomax-vetomin)*dble(k-1)/dble(nbinveto))) 
     &               then
                       call mfill(kk+49, 
     &                  ((vetomax-vetomin)*
     &                     dble(k)/dble(nbinveto)*0.99),
     &                     (www/xsecup2))
                   endif
                enddo
                call mfill(kk+50,(pt_veto),(www))
                call mfill(kk+51,(pt_veto),(www))
                if (pt_veto.gt.0d0)
     &           call mfill(kk+52,(dlog10(pt_veto)),(www))
             call mfill(kk+53,(getpseudorapv4(pjveto)),(www))
             call mfill(kk+54,(getrapidityv4(pjveto)),(www))
            endif

         endif
      enddo

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
      real*8 getptv4,p(0:3)
c
      getptv4=sqrt(p(1)**2+p(2)**2)
      return
      end


      function getpseudorapv4(p)
      implicit none
      real*8 getpseudorapv4,p(0:3)
      real*8 getpseudorap
c
      getpseudorapv4=getpseudorap(p(0),p(1),p(2),p(3))
      return
      end


      function getrapidityv4(p)
      implicit none
      real*8 getrapidityv4,p(0:3)
      real*8 getrapidity
c
      getrapidityv4=getrapidity(p(0),p(3))
      return
      end


      function getdrv4(p1,p2)
      implicit none
      real*8 getdrv4,p1(0:3),p2(0:3)
      real*8 getdr
c
      getdrv4=getdr(p1(0),p1(1),p1(2),p1(3),
     #              p2(0),p2(1),p2(2),p2(3))
      return
      end


      function getinvmv4(p)
      implicit none
      real*8 getinvmv4,p(0:3)
      real*8 getinvm
c
      getinvmv4=getinvm(p(0),p(1),p(2),p(3))
      return
      end


      function getdelphiv4(p1,p2)
      implicit none
      real*8 getdelphiv4,p1(0:3),p2(0:3)
      real*8 getdelphi
c
      getdelphiv4=getdelphi(p1(1),p1(2),
     #                      p2(1),p2(2))
      return
      end


      function getcosv4(q1,q2)
      implicit none
      real*8 getcosv4,q1(0:3),q2(0:3)
      real*8 xnorm1,xnorm2,tmp
c
      if(q1(0).lt.0.d0.or.q2(0).lt.0.d0)then
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
      double precision p(0:3),getmod

      getmod=sqrt(p(1)**2+p(2)**2+p(3)**2)

      return
      end



      subroutine getperpenv4(q1,q2,qperp)
c Normal to the plane defined by \vec{q1},\vec{q2}
      implicit none
      real*8 q1(0:3),q2(0:3),qperp(0:3)
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
        qperp(0)=1.d0
      endif
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
      real*8 p1(0:3),p2(0:3),pout(0:3)

      pout(1)=p1(2)*p2(3)-p1(3)*p2(2)
      pout(2)=p1(3)*p2(1)-p1(1)*p2(3)
      pout(3)=p1(1)*p2(2)-p1(2)*p2(1)
      pout(0)=0d0

      return
      end

