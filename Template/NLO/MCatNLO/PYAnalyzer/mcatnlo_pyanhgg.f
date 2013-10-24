C----------------------------------------------------------------------
      SUBROUTINE RCLOS()
C     DUMMY IF HBOOK IS USED
C----------------------------------------------------------------------
      END


C----------------------------------------------------------------------
      SUBROUTINE PYABEG
C     USER'S ROUTINE FOR INITIALIZATION
C----------------------------------------------------------------------
      implicit none
      real * 8 xmh0
      real * 8 xmhi,xmhs
      integer j,kk
      character*5 cc(2)
      data cc/'     ',' cuts'/
c
      xmh0=125d0
      xmhi=(xmh0-25d0)
      xmhs=(xmh0+25d0)
      call inihist
      do j=1,1
      kk=(j-1)*50
      call mbook(kk+1,'Higgs pT'//cc(j),2.d0,0.d0,200.d0)
      call mbook(kk+2,'Higgs pT'//cc(j),5.d0,0.d0,500.d0)
      call mbook(kk+3,'Higgs log(pT)'//cc(j),0.05d0,0.1d0,5.d0)
      call mbook(kk+4,'Higgs pT,  |y_H| < 2'//cc(j),2.d0,0.d0,200.d0)
      call mbook(kk+5,'Higgs pT,  |y_H| < 2'//cc(j),5.d0,0.d0,500.d0)
      call mbook(kk+6,'Higgs log(pT),  |y_H| < 2'//cc(j),
     #                                           0.05d0,0.1d0,5.d0)

      call mbook(kk+7,'H jet pT'//cc(j),2.d0,0.d0,200.d0)
      call mbook(kk+8,'H jet pT'//cc(j),5.d0,0.d0,500.d0)
      call mbook(kk+9,'H jet log(pT)'//cc(j),0.05d0,0.1d0,5.d0)
      call mbook(kk+10,'H jet pT,  |y_Hj| < 2'//cc(j),2.d0,0.d0,200.d0)
      call mbook(kk+11,'H jet pT,  |y_Hj| < 2'//cc(j),5.d0,0.d0,500.d0)
      call mbook(kk+12,'H jet log(pT),  |y_Hj| < 2'//cc(j),
     #                                             0.05d0,0.1d0,5.d0)

      call mbook(kk+13,'Inc jet pT'//cc(j),2.d0,0.d0,200.d0)
      call mbook(kk+14,'Inc jet pT'//cc(j),5.d0,0.d0,500.d0)
      call mbook(kk+15,'Inc jet log(pT)'//cc(j),0.05d0,0.1d0,5.d0)
      call mbook(kk+16,'Inc jet pT,  |y_Ij| < 2'//cc(j),2.d0,0.d0,2.d2)
      call mbook(kk+17,'Inc jet pT,  |y_Ij| < 2'//cc(j),5.d0,0.d0,5.d2)
      call mbook(kk+18,'Inc jet log(pT),  |y_Ij| < 2'//cc(j),
     #                                               0.05d0,0.1d0,5.d0)

      call mbook(kk+19,'Higgs y',0.2d0,-6.d0,6.d0)
      call mbook(kk+20,'Higgs y,  pT_H > 10 GeV',0.12d0,-6.d0,6.d0)
      call mbook(kk+21,'Higgs y,  pT_H > 30 GeV',0.12d0,-6.d0,6.d0)
      call mbook(kk+22,'Higgs y,  pT_H > 50 GeV',0.12d0,-6.d0,6.d0)
      call mbook(kk+23,'Higgs y,  pT_H > 70 GeV',0.12d0,-6.d0,6.d0)
      call mbook(kk+24,'Higgs y,  pt_H > 90 GeV',0.12d0,-6.d0,6.d0)

      call mbook(kk+25,'H jet y',0.2d0,-6.d0,6.d0)
      call mbook(kk+26,'H jet y,  pT_Hj > 10 GeV',0.2d0,-6.d0,6.d0)
      call mbook(kk+27,'H jet y,  pT_Hj > 30 GeV',0.2d0,-6.d0,6.d0)
      call mbook(kk+28,'H jet y,  pT_Hj > 50 GeV',0.2d0,-6.d0,6.d0)
      call mbook(kk+29,'H jet y,  pT_Hj > 70 GeV',0.2d0,-6.d0,6.d0)
      call mbook(kk+30,'H jet y,  pT_Hj > 90 GeV',0.2d0,-6.d0,6.d0)

      call mbook(kk+31,'H-Hj y',0.2d0,-6.d0,6.d0)
      call mbook(kk+32,'H-Hj y,  pT_Hj > 10 GeV',0.2d0,-6.d0,6.d0)
      call mbook(kk+33,'H-Hj y,  pT_Hj > 30 GeV',0.2d0,-6.d0,6.d0)
      call mbook(kk+34,'H-Hj y,  pT_Hj > 50 GeV',0.2d0,-6.d0,6.d0)
      call mbook(kk+35,'H-Hj y,  pT_Hj > 70 GeV',0.2d0,-6.d0,6.d0)
      call mbook(kk+36,'H-Hj y,  pT_Hj > 90 GeV',0.2d0,-6.d0,6.d0)

      call mbook(kk+37,'njets',1.d0,-0.5d0,10.5d0)
      call mbook(kk+38,'njets, |y_j| < 2.5 GeV',1.d0,-0.5d0,10.5d0)

      enddo
      END


C----------------------------------------------------------------------
      SUBROUTINE PYAEND(IEVT)
C     USER'S ROUTINE FOR TERMINAL CALCULATIONS, HISTOGRAM OUTPUT, ETC
C----------------------------------------------------------------------
      implicit none
      REAL*8 XNORM
      INTEGER I,J,KK,IEVT

      OPEN(UNIT=99,FILE='PYTHG.TOP',STATUS='UNKNOWN')
      XNORM=1.D0/IEVT
      DO I=1,100              
 	CALL MFINAL3(I)             
        CALL MCOPY(I,I+100)
        CALL MOPERA(I+100,'F',I+100,I+100,(XNORM),0.D0)
 	CALL MFINAL3(I+100)
      ENDDO
C
      do j=1,1
      kk=(j-1)*50
      call multitop(100+kk+1,99,3,2,'Higgs pT (GeV)',' ','LOG')
      call multitop(100+kk+2,99,3,2,'Higgs pT (GeV)',' ','LOG')
      call multitop(100+kk+3,99,3,2,'Higgs log(pT/GeV)',' ','LOG')
      call multitop(100+kk+4,99,3,2,'Higgs pT (GeV)',' ','LOG')
      call multitop(100+kk+5,99,3,2,'Higgs pT (GeV)',' ','LOG')
      call multitop(100+kk+6,99,3,2,'Higgs log(pT/GeV)',' ','LOG')
c
      call multitop(100+kk+7,99,3,2,'H jet pT (GeV)',' ','LOG')
      call multitop(100+kk+8,99,3,2,'H jet pT (GeV)',' ','LOG')
      call multitop(100+kk+9,99,3,2,'H jet log(pT/GeV)',' ','LOG')
      call multitop(100+kk+10,99,3,2,'H jet pT (GeV)',' ','LOG')
      call multitop(100+kk+11,99,3,2,'H jet pT (GeV)',' ','LOG')
      call multitop(100+kk+12,99,3,2,'H jet log(pT/GeV)',' ','LOG')
c
      call multitop(100+kk+13,99,3,2,'Inc jet pT (GeV)',' ','LOG')
      call multitop(100+kk+14,99,3,2,'Inc jet pT (GeV)',' ','LOG')
      call multitop(100+kk+15,99,3,2,'Inc jet log(pT/GeV)',' ','LOG')
      call multitop(100+kk+16,99,3,2,'Inc jet pT (GeV)',' ','LOG')
      call multitop(100+kk+17,99,3,2,'Inc jet pT (GeV)',' ','LOG')
      call multitop(100+kk+18,99,3,2,'Inc jet log(pT/GeV)',' ','LOG')
c
      call multitop(100+kk+19,99,3,2,'Higgs y',' ','LOG')
      call multitop(100+kk+20,99,3,2,'Higgs y',' ','LOG')
      call multitop(100+kk+21,99,3,2,'Higgs y',' ','LOG')
      call multitop(100+kk+22,99,3,2,'Higgs y',' ','LOG')
      call multitop(100+kk+23,99,3,2,'Higgs y',' ','LOG')
      call multitop(100+kk+24,99,3,2,'Higgs y',' ','LOG')
c     
      call multitop(100+kk+25,99,3,2,'H jet y',' ','LOG')
      call multitop(100+kk+26,99,3,2,'H jet y',' ','LOG')
      call multitop(100+kk+27,99,3,2,'H jet y',' ','LOG')
      call multitop(100+kk+28,99,3,2,'H jet y',' ','LOG')
      call multitop(100+kk+29,99,3,2,'H jet y',' ','LOG')
      call multitop(100+kk+30,99,3,2,'H jet y',' ','LOG')
c
      call multitop(100+kk+31,99,3,2,'H-Hj y',' ','LOG')
      call multitop(100+kk+32,99,3,2,'H-Hj y',' ','LOG')
      call multitop(100+kk+33,99,3,2,'H-Hj y',' ','LOG')
      call multitop(100+kk+34,99,3,2,'H-Hj y',' ','LOG')
      call multitop(100+kk+35,99,3,2,'H-Hj y',' ','LOG')
      call multitop(100+kk+36,99,3,2,'H-Hj y',' ','LOG')

      call multitop(100+kk+37,99,3,2,'njets',' ','LOG')
      call multitop(100+kk+38,99,3,2,'njets',' ','LOG')

      enddo
c
      CLOSE(99)
      END

C----------------------------------------------------------------------
      SUBROUTINE PYANAL
C     USER'S ROUTINE TO ANALYSE DATA FROM EVENT
C----------------------------------------------------------------------
      implicit none
      DOUBLE PRECISION PSUM(4),PPH(5),XMH,PTH,YH,P,V,PTP,YP,
     &VDOT,getrapidity,getpseudorap,etah,ECUT,PTJ1,PTJ,YJ,
     &PSUB,MJ1,Y,YCUT,YJ1
      INTEGER ICHSUM,ICHINI,IHEP,JPR,IFH,IST,ID1,IH,
     #IJ,J,KK,NMAX,N,NPAD,K,I,IORI,NJ,NSUB
      integer pychge
      external pydata
      common/pyjets/n,npad,k(4000,5),p(4000,5),v(4000,5)
      LOGICAL TEST1,TEST2,TEST3,TESTQG
      REAL*8 WWW0,TINY,p1(4),p2(4),pihep(4),MH
      INTEGER NN
      PARAMETER (NMAX=2000)
      INTEGER NJET,JET(NMAX),IPOS(NMAX),njet_central
      DOUBLE PRECISION PALG,RFJ,SYCUT,PP(4,NMAX),PJET(4,NMAX),
     # PTJET(NMAX),ETAJET(NMAX),YJET(NMAX),pjet_new(4,nmax),
     # njdble,njcdble,y_central

      DOUBLE PRECISION EVWEIGHT
      COMMON/CEVWEIGHT/EVWEIGHT
      DATA TINY/.1D-5/
      INTEGER IFAIL
      COMMON/CIFAIL/IFAIL
c
      IF(IFAIL.EQ.1)RETURN
C INITIALISE
      DO I=1,NMAX
        DO J=1,4
          PP(J,I)=0D0
        ENDDO
      ENDDO
C INCOMING PARTONS MAY TRAVEL IN THE SAME DIRECTION: IT'S A POWER-SUPPRESSED
C EFFECT, SO THROW THE EVENT AWAY
      IF(SIGN(1.D0,P(3,3)).EQ.SIGN(1.D0,P(4,3)))THEN
         WRITE(*,*)'WARNING 111 IN PYANAL'
         GOTO 999
      ENDIF
      WWW0=EVWEIGHT
      DO I=1,4
         P1(I)=P(1,I)
         P2(I)=P(2,I)
      ENDDO
      CALL VVSUM(4,P1,P2,PSUM)
      CALL VSCA(4,-1D0,PSUM,PSUM)
      ICHSUM=0
      ICHINI=PYCHGE(K(1,2))+PYCHGE(K(2,2))
      IFH=0
      NN=0
      DO 100 IHEP=1,N
        DO J=1,4
          PIHEP(J)=P(IHEP,J)
        ENDDO
        IST =K(IHEP,1)
        ID1 =K(IHEP,2)
        IORI=K(IHEP,3)
        IF (IST.LE.10) THEN
          CALL VVSUM(4,PIHEP,PSUM,PSUM)
          ICHSUM=ICHSUM+PYCHGE(ID1)
          IF(ID1.EQ.25)THEN
            IFH=IFH+1
            DO IJ=1,5
              PPH(IJ)=P(IHEP,IJ)
            ENDDO
          ENDIF
        ENDIF
C---FIND FINAL STATE HADRONS
        IF (IST.LE.10 .AND. ABS(ID1).GT.100) THEN
          NN=NN+1
          IF (NN.GT.NMAX)THEN
            WRITE(*,*)'TOO MANY PARTICLES!'
            STOP
          ENDIF
          DO I=1,4
             PP(I,NN)=P(IHEP,I)
          ENDDO
        ENDIF
 100  CONTINUE
      IF(IFH.NE.1)THEN
         WRITE(*,*)'WARNING 501 IN PYANAL'
         STOP
      ENDIF
C CHECK MOMENTUM AND CHARGE CONSERVATION
      IF (VDOT(3,PSUM,PSUM).GT.1.E-4*P(1,4)**2) THEN
         WRITE(*,*)'WARNING 112 IN PYANAL'
         GOTO 999
      ENDIF
      IF (ICHSUM.NE.ICHINI) THEN
         WRITE(*,*)'WARNING 113 IN PYANAL'
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
      njet=0
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
      kk=0
      call mfill(kk+1,(pth),(WWW0))
      call mfill(kk+2,(pth),(WWW0))
      if(pth.gt.0.d0)call mfill(kk+3,(log10(pth)),(WWW0))
      if(abs(yh).le.2.d0)then
         call mfill(kk+4,(pth),(WWW0))
         call mfill(kk+5,(pth),(WWW0))
         if(pth.gt.0.d0)call mfill(kk+6,(log10(pth)),(WWW0))
      endif
c
      if(njet.ge.1)then
      call mfill(kk+7,(ptj1),(WWW0))
      call mfill(kk+8,(ptj1),(WWW0))
      if(ptj1.gt.0.d0)call mfill(kk+9,(log10(ptj1)),(WWW0))
      if(abs(yj1).le.2.d0)then
         call mfill(kk+10,(ptj1),(WWW0))
         call mfill(kk+11,(ptj1),(WWW0))
         if(ptj1.gt.0.d0)call mfill(kk+12,(log10(ptj1)),
     &                                          (WWW0))
      endif
c
      do nj=1,njet
         call mfill(kk+13,(ptjet(nj)),(WWW0))
         call mfill(kk+14,(ptjet(nj)),(WWW0))
         if(ptjet(nj).gt.0.d0)call mfill(kk+15,(log10(ptjet(nj))),
     &                                                (WWW0))
         if(abs(yjet(nj)).le.2.d0)then
            call mfill(kk+16,(ptjet(nj)),(WWW0))
            call mfill(kk+17,(ptjet(nj)),(WWW0))
            if(ptjet(nj).gt.0d0)call mfill(kk+18,(log10(ptjet(nj))),
     &                                                   (WWW0))
         endif
      enddo
      endif
c
      call mfill(kk+19,(yh),(WWW0))
      if(pth.ge.10.d0) call mfill(kk+20,(yh),(WWW0))
      if(pth.ge.30.d0) call mfill(kk+21,(yh),(WWW0))
      if(pth.ge.50.d0) call mfill(kk+22,(yh),(WWW0))
      if(pth.ge.70.d0) call mfill(kk+23,(yh),(WWW0))
      if(pth.ge.90.d0) call mfill(kk+24,(yh),(WWW0))  
c     
      if(njet.ge.1)then
      call mfill(kk+25,(yj1),(WWW0))
      if(ptj1.ge.10.d0) call mfill(kk+26,(yj1),(WWW0))
      if(ptj1.ge.30.d0) call mfill(kk+27,(yj1),(WWW0))
      if(ptj1.ge.50.d0) call mfill(kk+28,(yj1),(WWW0))
      if(ptj1.ge.70.d0) call mfill(kk+29,(yj1),(WWW0))
      if(ptj1.ge.90.d0) call mfill(kk+30,(yj1),(WWW0))
c
      call mfill(kk+31,(yh-yj1),(WWW0))
      if(ptj1.ge.10.d0) call mfill(kk+32,(yh-yj1),(WWW0))
      if(ptj1.ge.30.d0) call mfill(kk+33,(yh-yj1),(WWW0))
      if(ptj1.ge.50.d0) call mfill(kk+34,(yh-yj1),(WWW0))
      if(ptj1.ge.70.d0) call mfill(kk+35,(yh-yj1),(WWW0))
      if(ptj1.ge.90.d0) call mfill(kk+36,(yh-yj1),(WWW0))
      endif
c
      call mfill(kk+37,(njdble),(WWW0))
      call mfill(kk+38,(njcdble),(WWW0))



 999  END


      function getrapidity(en,pl)
      implicit none
      real*8 getrapidity,en,pl,tiny,xplus,xminus,y
      parameter (tiny=1.d-5)
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

C-----------------------------------------------------------------------
      SUBROUTINE VVSUM(N,P,Q,R)
C-----------------------------------------------------------------------
C    VECTOR SUM
C-----------------------------------------------------------------------
      IMPLICIT NONE
      INTEGER N,I
      DOUBLE PRECISION P(N),Q(N),R(N)
      DO 10 I=1,N
   10 R(I)=P(I)+Q(I)
      END



C-----------------------------------------------------------------------
      SUBROUTINE VSCA(N,C,P,Q)
C-----------------------------------------------------------------------
C     VECTOR TIMES SCALAR
C-----------------------------------------------------------------------
      IMPLICIT NONE
      INTEGER N,I
      DOUBLE PRECISION C,P(N),Q(N)
      DO 10 I=1,N
   10 Q(I)=C*P(I)
      END



C-----------------------------------------------------------------------
      FUNCTION VDOT(N,P,Q)
C-----------------------------------------------------------------------
C     VECTOR DOT PRODUCT
C-----------------------------------------------------------------------
      IMPLICIT NONE
      INTEGER N,I
      DOUBLE PRECISION VDOT,PQ,P(N),Q(N)
      PQ=0.
      DO 10 I=1,N
   10 PQ=PQ+P(I)*Q(I)
      VDOT=PQ
      END
