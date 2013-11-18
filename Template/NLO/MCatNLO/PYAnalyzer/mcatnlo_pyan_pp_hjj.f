C----------------------------------------------------------------------
      SUBROUTINE RCLOS()
C     DUMMY IF HBOOK IS USED
C----------------------------------------------------------------------
      END


C----------------------------------------------------------------------
      SUBROUTINE PYABEG
C     USER''S ROUTINE FOR INITIALIZATION
C----------------------------------------------------------------------
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
         call mbook(kk+  1,'total rate    '//cc(i),1.0d0,0.5d0,5.5d0)

         call mbook(kk+  2,'Higgs pT      '//cc(i),8.0d0,0.d0,400.d0)
         call mbook(kk+  3,'Higgs pT      '//cc(i),16.0d0,0.d0,800.d0)
         call mbook(kk+  4,'Higgs log pT  '//cc(i),0.08d0,0.d0,4.d0)
         call mbook(kk+  5,'Higgs eta     '//cc(i),0.24d0,-6.d0,6.d0)
         call mbook(kk+  6,'Higgs y       '//cc(i),0.24d0,-6.d0,6.d0)

         call mbook(kk+  7,'j1 pT         '//cc(i),8.0d0,0.d0,400.d0)
         call mbook(kk+  8,'j1 pT         '//cc(i),16.0d0,0.d0,800.d0)
         call mbook(kk+  9,'j1 log pT     '//cc(i),0.08d0,0.d0,4.d0)
         call mbook(kk+ 10,'j1 eta        '//cc(i),0.24d0,-6.d0,6.d0)
         call mbook(kk+ 11,'j1 y          '//cc(i),0.24d0,-6.d0,6.d0)

         call mbook(kk+ 12,'j2 pT         '//cc(i),8.0d0,0.d0,400.d0)
         call mbook(kk+ 13,'j2 pT         '//cc(i),16.0d0,0.d0,800.d0)
         call mbook(kk+ 14,'j2 log pT     '//cc(i),0.08d0,0.d0,4.d0)
         call mbook(kk+ 15,'j2 eta        '//cc(i),0.24d0,-6.d0,6.d0)
         call mbook(kk+ 16,'j2 y          '//cc(i),0.24d0,-6.d0,6.d0)

         call mbook(kk+ 17,'j3 pT         '//cc(i),8.0d0,0.d0,400.d0)
         call mbook(kk+ 18,'j3 pT         '//cc(i),16.0d0,0.d0,800.d0)
         call mbook(kk+ 19,'j3 log pT     '//cc(i),0.08d0,0.d0,4.d0)
         call mbook(kk+ 20,'j3 eta        '//cc(i),0.24d0,-6.d0,6.d0)
         call mbook(kk+ 21,'j3 y          '//cc(i),0.24d0,-6.d0,6.d0)

         call mbook(kk+ 22,'H+j1 pT       '//cc(i),8.0d0,0.d0,400.d0)
         call mbook(kk+ 23,'H+j1 pT       '//cc(i),16.0d0,0.d0,800.d0)
         call mbook(kk+ 24,'H+j1 log pT   '//cc(i),0.08d0,0.d0,4.d0)
         call mbook(kk+ 25,'H+j1 eta      '//cc(i),0.24d0,-6.d0,6.d0)
         call mbook(kk+ 26,'H+j1 y        '//cc(i),0.24d0,-6.d0,6.d0)

         call mbook(kk+ 27,'j1+j2 pT      '//cc(i),8.0d0,0.d0,400.d0)
         call mbook(kk+ 28,'j1+j2 pT      '//cc(i),16.0d0,0.d0,800.d0)
         call mbook(kk+ 29,'j1+j2 log pT  '//cc(i),0.08d0,0.d0,4.d0)
         call mbook(kk+ 30,'j1+j2 eta     '//cc(i),0.24d0,-6.d0,6.d0)
         call mbook(kk+ 31,'j1+j2 y       '//cc(i),0.24d0,-6.d0,6.d0)

         call mbook(kk+ 32,'syst pT       '//cc(i),8.0d0,0.d0,400.d0)
         call mbook(kk+ 33,'syst pT       '//cc(i),16.0d0,0.d0,800.d0)
         call mbook(kk+ 34,'syst log pT   '//cc(i),0.08d0,0.d0,4.d0)
         call mbook(kk+ 35,'syst eta      '//cc(i),0.4d0,-10.d0,10.d0)
         call mbook(kk+ 36,'syst y        '//cc(i),0.24d0,-6.d0,6.d0)

         call mbook(kk+ 37,'Dphi H-j1     '//cc(i),pi/50.d0,0d0,pi)
         call mbook(kk+ 38,'Dphi H-j2     '//cc(i),pi/50.d0,0d0,pi)
         call mbook(kk+ 39,'Dphi j1-j2    '//cc(i),pi/50.d0,0d0,pi)

         call mbook(kk+ 40,'DR H-j1       '//cc(i),0.2d0,0d0,10.d0)
         call mbook(kk+ 41,'DR H-j2       '//cc(i),0.2d0,0d0,10.d0)
         call mbook(kk+ 42,'DR j1-j2      '//cc(i),0.2d0,0d0,10.d0)

         call mbook(kk+ 43,'mj1j2         '//cc(i),60.0d0,0d0,3000.d0)

c Nason-Oleari plots (hep-ph/0911.5299)
         call mbook(kk+ 44,'|yj1-yj2|     '//cc(i),0.4d0,0.d0,10.d0)
         call mbook(kk+ 45,'yj3_rel       '//cc(i),0.24d0,-6.d0,6.d0)
         call mbook(kk+ 46,'njets         '//cc(i),1.d0,-0.5d0,9.5d0)
         call mbook(kk+ 47,'ptrel_j1      '//cc(i),4.d0,0.d0,200.d0)
         call mbook(kk+ 48,'ptrel_j2      '//cc(i),4.d0,0.d0,200.d0)
         call mbook(kk+ 49,'P-veto        '//cc(i),
     1    ((vetomax-vetomin)/dble(nbinveto)),vetomin,vetomax)
         call mbook(kk+ 50,'jveto pT    '//cc(i),
     1    ((vetomax-vetomin)/dble(nbinveto)),vetomin,vetomax)
         call mbook(kk+ 51,'jveto pT    '//cc(i),
     1    ((2d0*vetomax-vetomin)/dble(nbinveto)),
     1    vetomin,2d0*vetomax)
         call mbook(kk+ 52,'jveto log pT'//cc(i),0.08d0,0.d0,4.d0)
         call mbook(kk+ 53,'jveto eta   '//cc(i),0.24d0,-6.d0,6.d0)
         call mbook(kk+ 54,'jveto y     '//cc(i),0.24d0,-6.d0,6.d0)

      enddo

 999  END

C----------------------------------------------------------------------
      SUBROUTINE PYAEND(IEVT)
C     USER''S ROUTINE FOR TERMINAL CALCULATIONS, HISTOGRAM OUTPUT, ETC
C----------------------------------------------------------------------
      REAL*8 XNORM
      INTEGER I,J,K,kk
      OPEN(UNIT=99,FILE='PYTVBF.top',STATUS='UNKNOWN')
      XNORM=1.D0/IEVT
      DO I=1,200              
 	CALL MFINAL3(I)             
        CALL MCOPY(I,I+200)
        CALL MOPERA(I+200,'F',I+200,I+200,(XNORM),0.D0)
 	CALL MFINAL3(I+200)             
      ENDDO                          
C
      do i=1,2
         kk=(i-1)*100
         call multitop(200+kk+  1,99,3,2,'total rate       ',' ','LOG')

         call multitop(200+kk+  2,99,3,2,'Higgs pT         ',' ','LOG')
         call multitop(200+kk+  3,99,3,2,'Higgs pT         ',' ','LOG')
         call multitop(200+kk+  4,99,3,2,'Higgs log pT     ',' ','LOG')
         call multitop(200+kk+  5,99,3,2,'Higgs eta        ',' ','LOG')
         call multitop(200+kk+  6,99,3,2,'Higgs y          ',' ','LOG')

         call multitop(200+kk+  7,99,3,2,'j1 pT            ',' ','LOG')
         call multitop(200+kk+  8,99,3,2,'j1 pT            ',' ','LOG')
         call multitop(200+kk+  9,99,3,2,'j1 log pT        ',' ','LOG')
         call multitop(200+kk+ 10,99,3,2,'j1 eta           ',' ','LOG')
         call multitop(200+kk+ 11,99,3,2,'j1 y             ',' ','LOG')

         call multitop(200+kk+ 12,99,3,2,'j2 pT            ',' ','LOG')
         call multitop(200+kk+ 13,99,3,2,'j2 pT            ',' ','LOG')
         call multitop(200+kk+ 14,99,3,2,'j2 log pT        ',' ','LOG')
         call multitop(200+kk+ 15,99,3,2,'j2 eta           ',' ','LOG')
         call multitop(200+kk+ 16,99,3,2,'j2 y             ',' ','LOG')

         call multitop(200+kk+ 17,99,3,2,'j3 pT            ',' ','LOG')
         call multitop(200+kk+ 18,99,3,2,'j3 pT            ',' ','LOG')
         call multitop(200+kk+ 19,99,3,2,'j3 log pT        ',' ','LOG')
         call multitop(200+kk+ 20,99,3,2,'j3 eta           ',' ','LOG')
         call multitop(200+kk+ 21,99,3,2,'j3 y             ',' ','LOG')

         call multitop(200+kk+ 22,99,3,2,'H+j1 pT          ',' ','LOG')
         call multitop(200+kk+ 23,99,3,2,'H+j1 pT          ',' ','LOG')
         call multitop(200+kk+ 24,99,3,2,'H+j1 log pT      ',' ','LOG')
         call multitop(200+kk+ 25,99,3,2,'H+j1 eta         ',' ','LOG')
         call multitop(200+kk+ 26,99,3,2,'H+j1 y           ',' ','LOG')

         call multitop(200+kk+ 27,99,3,2,'j1+j2 pT         ',' ','LOG')
         call multitop(200+kk+ 28,99,3,2,'j1+j2 pT         ',' ','LOG')
         call multitop(200+kk+ 29,99,3,2,'j1+j2 log pT     ',' ','LOG')
         call multitop(200+kk+ 30,99,3,2,'j1+j2 eta        ',' ','LOG')
         call multitop(200+kk+ 31,99,3,2,'j1+j2 y          ',' ','LOG')

         call multitop(200+kk+ 32,99,3,2,'syst pT          ',' ','LOG')
         call multitop(200+kk+ 33,99,3,2,'syst pT          ',' ','LOG')
         call multitop(200+kk+ 34,99,3,2,'syst log pT      ',' ','LOG')
         call multitop(200+kk+ 35,99,3,2,'syst eta         ',' ','LOG')
         call multitop(200+kk+ 36,99,3,2,'syst y           ',' ','LOG')

         call multitop(200+kk+ 37,99,3,2,'Dphi H-j1        ',' ','LOG')
         call multitop(200+kk+ 38,99,3,2,'Dphi H-j2        ',' ','LOG')
         call multitop(200+kk+ 39,99,3,2,'Dphi j1-j2       ',' ','LOG')

         call multitop(200+kk+ 40,99,3,2,'DR H-j1          ',' ','LOG')
         call multitop(200+kk+ 41,99,3,2,'DR H-j2          ',' ','LOG')
         call multitop(200+kk+ 42,99,3,2,'DR j1-j2         ',' ','LOG')

         call multitop(200+kk+ 43,99,3,2,'mj1j2            ',' ','LOG')

         call multitop(200+kk+ 44,99,3,2,'|yj1-yj2|        ',' ','LOG')
         call multitop(200+kk+ 45,99,3,2,'yj3_rel          ',' ','LOG')
         call multitop(200+kk+ 46,99,3,2,'njets            ',' ','LOG')
         call multitop(200+kk+ 47,99,3,2,'ptrel_j1         ',' ','LOG')
         call multitop(200+kk+ 48,99,3,2,'ptrel_j2         ',' ','LOG')
         call multitop(200+kk+ 49,99,3,2,'P-veto           ',' ','LOG')
         call multitop(200+kk+ 50,99,3,2,'jv pT            ',' ','LOG')
         call multitop(200+kk+ 51,99,3,2,'jv pT            ',' ','LOG')
         call multitop(200+kk+ 52,99,3,2,'jv log pT        ',' ','LOG')
         call multitop(200+kk+ 53,99,3,2,'jv eta           ',' ','LOG')
         call multitop(200+kk+ 54,99,3,2,'jv y             ',' ','LOG')
      enddo

      CLOSE(99)
      END

C----------------------------------------------------------------------
      SUBROUTINE PYANAL
C     USER''S ROUTINE TO ANALYSE DATA FROM EVENT
C----------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
      INTEGER PYCHGE
      DOUBLE PRECISION HWVDOT,PSUM(4),PJJ(4),PIHEP(5)

      common/pyjets/n,npad,k(4000,5),p(4000,5),v(4000,5)
      common/pydat1/mstu(200),paru(200),mstj(200),parj(200)
      common/pydat2/kchg(500,4),pmas(500,4),parf(2000),vckm(4,4)
      common/pydat3/mdcy(500,3),mdme(8000,2),brat(8000),kfdp(8000,5)
      common/pysubs/msel,mselpd,msub(500),kfin(2,-40:40),ckin(200)
      common/pypars/mstp(200),parp(200),msti(200),pari(200)
      double precision evweight
      common/cevweight/evweight

      INTEGER ICHSUM,ICHINI,IHEP,IST,ID,IJ,J,NN,I,L
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
     &ptHj,etaHj,yHj,ptjj,etajj,yjj,ptsyst,etasyst,ysyst,p1(5),p2(5),yjmax,
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
c
C INITIALISE
      do i=1,nmax
        do j=1,4
          pp(j,i)=0d0
        enddo
      enddo
      xsecup2=1d0
      IF(IFAIL.EQ.1)RETURN
C INCOMING PARTONS MAY TRAVEL IN THE SAME DIRECTION: IT''S A POWER-SUPPRESSED
C EFFECT, SO THROW THE EVENT AWAY
      IF(SIGN(1.D0,P(3,3)).EQ.SIGN(1.D0,P(4,3)))THEN
         WRITE(*,*)'WARNING 111 IN PYANAL'
        GOTO 999
      ENDIF
      WWW0=EVWEIGHT
      do j=1,4
        p1(j)=p(1,j)
        p2(j)=p(2,j)
      enddo
      CALL VVSUM(4,P1,P2,PSUM)
      CALL VSCA(4,-1D0,PSUM,PSUM)
      ICHSUM=0
      ICHINI=PYCHGE(K(1,2))+PYCHGE(K(2,2))

      NN=0
      NH=0
      DO 100 IHEP=1,N
        IST=K(IHEP,1)      
        ID=K(IHEP,2)
        DO J=1,5
          PIHEP(J)=P(IHEP,J)
        ENDDO
        IF(IST.LE.10)THEN
           CALL VVSUM(4,PIHEP,PSUM,PSUM)
           ICHSUM=ICHSUM+PYCHGE(ID)
        ENDIF
        IF(ABS(ID).GT.100.AND.IST.LT.10) THEN
           NN=NN+1
           IF (NN.GT.NMAX) STOP 'Too many particles [hadrons]!'
           DO I=1,4
              PP(I,NN)=PIHEP(I)
           ENDDO
        ENDIF
C FIND THE HIGGS
        IF(ID.EQ.25.AND.IST.EQ.1)THEN
           NH=NH+1
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
C CHECK MOMENTUM AND CHARGE CONSERVATION
      IF (VDOT(3,PSUM,PSUM).GT.1.E-4*P(1,4)**2) THEN
         WRITE(*,*)'ERROR 112 IN PYANAL'
         GOTO 999
      ENDIF
      IF (ICHSUM.NE.ICHINI) THEN
         WRITE(*,*)'ERROR 113 IN PYANAL'
         GOTO 999
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
               WRITE(*,*)'ERROR 501 IN PYANAL'
               STOP
            endif
         endif
         yjet(i)=getrapidityv4(pjet(1,i))
         etajet(i)=getpseudorapv4(pjet(1,i))
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
         do l=1,4
             pH(l)   =p(ih,l)
             pj1(l)  =pjet(l,ij1)
             pj2(l)  =pjet(l,ij2)
             pj3(l)  =pjet(l,ij3)
             pjj(l)  =pjet(l,ij1)+pjet(l,ij2)
             pHj(l)  =pjet(l,ij1)+pH(l)
             psyst(l)=pjet(l,ij1)+pjet(l,ij2)+pH(l)
             pjveto(l)=pjet(l,ijveto)
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

         if(flag)then
            call mfill(kk+  1,(1d0),(www0))
            call mfill(kk+  2,(ptH),(www0))
            call mfill(kk+  3,(ptH),(www0))
            if(ptH.gt.0d0)
     &      call mfill(kk+  4,(log10(ptH)),(www0))
            call mfill(kk+  5,(etaH),(www0))
            call mfill(kk+  6,(yH),(www0))
            call mfill(kk+ 46,(njdble),(www0))
cc            call mfill(kk+ 48,(temp_scalup),(www0))
            
            if(njet.ge.1)then
               call mfill(kk+  7,(ptj1),(www0))
               call mfill(kk+  8,(ptj1),(www0))
               call mfill(kk+  9,(log10(ptj1)),(www0))
               call mfill(kk+ 10,(etaj1),(www0))
               call mfill(kk+ 11,(yj1),(www0))
               call mfill(kk+ 22,(ptHj),(www0))
               call mfill(kk+ 23,(ptHj),(www0))
               if(ptHj.gt.0d0)
     &         call mfill(kk+ 24,(log10(ptHj)),(www0))
               call mfill(kk+ 25,(etaHj),(www0))
               call mfill(kk+ 26,(yHj),(www0))
               call mfill(kk+ 37,(DphiHj1),(www0))
               call mfill(kk+ 40,(DRHj1),(www0))
               call mfill(kk+ 47,(ptrel_j1),(www0))
            endif 

            if(njet.ge.2)then
               call mfill(kk+ 12,(ptj2),(www0))
               call mfill(kk+ 13,(ptj2),(www0))
               call mfill(kk+ 14,(log10(ptj2)),(www0))
               call mfill(kk+ 15,(etaj2),(www0))
               call mfill(kk+ 16,(yj2),(www0))
               call mfill(kk+ 27,(ptjj),(www0))
               call mfill(kk+ 28,(ptjj),(www0))
               if(ptjj.gt.0d0)
     &         call mfill(kk+ 29,(log10(ptjj)),(www0))
               call mfill(kk+ 30,(etajj),(www0))
               call mfill(kk+ 31,(yjj),(www0))
               call mfill(kk+ 32,(ptsyst),(www0))
               call mfill(kk+ 33,(ptsyst),(www0))
               if(ptsyst.gt.0d0)
     &         call mfill(kk+ 34,(log10(ptsyst)),(www0))
               call mfill(kk+ 35,(etasyst),(www0))
               call mfill(kk+ 36,(ysyst),(www0))
               call mfill(kk+ 38,(DphiHj2),(www0))
               call mfill(kk+ 39,(Dphij1j2),(www0))
               call mfill(kk+ 41,(DRHj2),(www0))
               call mfill(kk+ 42,(DRj1j2),(www0))
               call mfill(kk+ 43,(mj1j2),(www0))
               call mfill(kk+ 44,(Dyj1j2),(www0))
               call mfill(kk+ 48,(ptrel_j2),(www0))
            endif

            if(njet.ge.3)then
               call mfill(kk+ 17,(ptj3),(www0))
               call mfill(kk+ 18,(ptj3),(www0))
               call mfill(kk+ 19,(log10(ptj3)),(www0))
               call mfill(kk+ 20,(etaj3),(www0))
               call mfill(kk+ 21,(yj3),(www0))
               call mfill(kk+ 45,(yj3rel),(www0))
            endif
            if (ijveto.gt.0) then
                pt_veto = getptv4(pjveto)
                do l=1,nbinveto
                   if (pt_veto.gt.
     &          (vetomin+(vetomax-vetomin)*dble(l-1)/dble(nbinveto))) 
     &               then
                       call mfill(kk+49, 
     &                  ((vetomax-vetomin)*
     &                     dble(l)/dble(nbinveto)*0.99),
     &                     (www0/xsecup2))
                   endif
                enddo
                call mfill(kk+50,(pt_veto),(www0))
                call mfill(kk+51,(pt_veto),(www0))
                if (pt_veto.gt.0d0)
     &           call mfill(kk+52,(dlog10(pt_veto)),(www0))
             call mfill(kk+53,(getpseudorapv4(pjveto)),(www0))
             call mfill(kk+54,(getrapidityv4(pjveto)),(www0))
            endif

         endif
      enddo

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

c-----------------------------------------------------------------------
      subroutine vvsum(n,p,q,r)
c-----------------------------------------------------------------------
c    vector sum
c-----------------------------------------------------------------------
      implicit none
      integer n,i
      double precision p(n),q(n),r(n)
      do 10 i=1,n
   10 r(i)=p(i)+q(i)
      end



c-----------------------------------------------------------------------
      subroutine vsca(n,c,p,q)
c-----------------------------------------------------------------------
c     vector times scalar
c-----------------------------------------------------------------------
      implicit none
      integer n,i
      double precision c,p(n),q(n)
      do 10 i=1,n
   10 q(i)=c*p(i)
      end



c-----------------------------------------------------------------------
      function vdot(n,p,q)
c-----------------------------------------------------------------------
c     vector dot product
c-----------------------------------------------------------------------
      implicit none
      integer n,i
      double precision vdot,pq,p(n),q(n)
      pq=0.
      do 10 i=1,n
   10 pq=pq+p(i)*q(i)
      vdot=pq
      end

