c
c Example analysis for "p p > h [QCD]" process.
c
C----------------------------------------------------------------------
      SUBROUTINE RCLOS()
C     DUMMY IF HBOOK IS USED
C----------------------------------------------------------------------
      END


C----------------------------------------------------------------------
      SUBROUTINE PYABEG
C     USER''S ROUTINE FOR INITIALIZATION
C----------------------------------------------------------------------
      implicit none
      include 'reweight0.inc'
      integer nwgt,max_weight,nwgt_analysis,kk,l
      common/cnwgt/nwgt
      common/c_analysis/nwgt_analysis
      character*50 weights_info(max_weight_shower)
      common/cwgtsinfo/weights_info
      call inihist
      nwgt_analysis=nwgt
      do kk=1,nwgt_analysis
      l=(kk-1)*40
      call mbook(l+1,'Higgs pT '//weights_info(kk)
     $     ,2.d0,0.d0,200.d0)
      call mbook(l+2,'Higgs pT 1 '//weights_info(kk)
     $     ,5.d0,0.d0,500.d0)
      call mbook(l+3,'Higgs log[pT] '//weights_info(kk)
     $     ,0.05d0,0.1d0,5.d0)
      call mbook(l+4,'Higgs pT,|y_H|<2 '//weights_info(kk)
     $     ,2.d0,0.d0,200.d0)
      call mbook(l+5,'Higgs pT 1,|y_H|<2 '//weights_info(kk)
     $     ,5.d0,0.d0,500.d0)
      call mbook(l+6,'Higgs log[pT],|y_H|<2 '//weights_info(kk)
     $     ,0.05d0,0.1d0,5.d0)

      call mbook(l+7,'j1 pT '//weights_info(kk)
     $     ,2.d0,0.d0,200.d0)
      call mbook(l+8,'j1 pT 1 '//weights_info(kk)
     $     ,5.d0,0.d0,500.d0)
      call mbook(l+9,'j1 log[pT] '//weights_info(kk)
     $     ,0.05d0,0.1d0,5.d0)
      call mbook(l+10,'j1 pT,|y_j1|<2 '//weights_info(kk)
     $     ,2.d0,0.d0,200.d0)
      call mbook(l+11,'j1 pT 1,|y_j1|<2 '//weights_info(kk)
     $     ,5.d0,0.d0,500.d0)
      call mbook(l+12,'j1 log[pT],|y_j1|<2 '//weights_info(kk)
     $     ,0.05d0,0.1d0,5.d0)

      call mbook(l+13,'Inc j pT '//weights_info(kk)
     $     ,2.d0,0.d0,200.d0)
      call mbook(l+14,'Inc j pT 1 '//weights_info(kk)
     $     ,5.d0,0.d0,500.d0)
      call mbook(l+15,'Inc j log[pT] '//weights_info(kk)
     $     ,0.05d0,0.1d0,5.d0)
      call mbook(l+16,'Inc j pT,|y_Ij|<2 '//weights_info(kk)
     $     ,2.d0,0.d0,2.d2)
      call mbook(l+17,'Inc j pT 1,|y_Ij|<2 '//weights_info(kk)
     $     ,5.d0,0.d0,5.d2)
      call mbook(l+18,'Inc j log[pT],|y_Ij|<2'//weights_info(kk)
     $     ,0.05d0,0.1d0,5.d0)

      call mbook(l+19,'Higgs y '//weights_info(kk)
     $     ,0.2d0,-6.d0,6.d0)
      call mbook(l+20,'Higgs y,pT_H>10GeV '//weights_info(kk)
     $     ,0.12d0,-6.d0,6.d0)
      call mbook(l+21,'Higgs y,pT_H>30GeV '//weights_info(kk)
     $     ,0.12d0,-6.d0,6.d0)
      call mbook(l+22,'Higgs y,pT_H>50GeV '//weights_info(kk)
     $     ,0.12d0,-6.d0,6.d0)
      call mbook(l+23,'Higgs y,pT_H>70GeV '//weights_info(kk)
     $     ,0.12d0,-6.d0,6.d0)
      call mbook(l+24,'Higgs y,pt_H>90GeV '//weights_info(kk)
     $     ,0.12d0,-6.d0,6.d0)

      call mbook(l+25,'j1 y '//weights_info(kk)
     $     ,0.2d0,-6.d0,6.d0)
      call mbook(l+26,'j1 y,pT_j1>10GeV '//weights_info(kk)
     $     ,0.2d0,-6.d0,6.d0)
      call mbook(l+27,'j1 y,pT_j1>30GeV '//weights_info(kk)
     $     ,0.2d0,-6.d0,6.d0)
      call mbook(l+28,'j1 y,pT_j1>50GeV '//weights_info(kk)
     $     ,0.2d0,-6.d0,6.d0)
      call mbook(l+29,'j1 y,pT_j1>70GeV '//weights_info(kk)
     $     ,0.2d0,-6.d0,6.d0)
      call mbook(l+30,'j1 y,pT_j1>90GeV '//weights_info(kk)
     $     ,0.2d0,-6.d0,6.d0)

      call mbook(l+31,'H-j1 y '//weights_info(kk)
     $     ,0.2d0,-6.d0,6.d0)
      call mbook(l+32,'H-j1 y,pT_j1>10GeV '//weights_info(kk)
     $     ,0.2d0,-6.d0,6.d0)
      call mbook(l+33,'H-j1 y,pT_j1>30GeV '//weights_info(kk)
     $     ,0.2d0,-6.d0,6.d0)
      call mbook(l+34,'H-j1 y,pT_j1>50GeV '//weights_info(kk)
     $     ,0.2d0,-6.d0,6.d0)
      call mbook(l+35,'H-j1 y,pT_j1>70GeV '//weights_info(kk)
     $     ,0.2d0,-6.d0,6.d0)
      call mbook(l+36,'H-j1 y,pT_j1>90GeV '//weights_info(kk)
     $     ,0.2d0,-6.d0,6.d0)
      
      call mbook(l+37,'njets '//weights_info(kk)
     $     ,1.d0,-0.5d0,10.5d0)
      call mbook(l+38,'njets,|y_j|<2.5 '//weights_info(kk)
     $     ,1.d0,-0.5d0,10.5d0)
      call mbook(l+39,'xsec '//weights_info(kk)
     $     ,1.d0,-0.5d0,2.5d0)

      enddo
      END

C----------------------------------------------------------------------
      SUBROUTINE PYAEND(IEVT)
C     USER''S ROUTINE FOR TERMINAL CALCULATIONS, HISTOGRAM OUTPUT, ETC
C----------------------------------------------------------------------
      implicit none
      REAL*8 XNORM
      INTEGER I,J,KK,IEVT,l,nwgt_analysis
      integer NPL
      parameter(NPL=15000)
      common/c_analysis/nwgt_analysis
      OPEN(UNIT=99,FILE='PYTHG.TOP',STATUS='UNKNOWN')
      XNORM=1.D0/IEVT
      DO I=1,NPL
        CALL MFINAL3(I)
        CALL MCOPY(I,I+NPL)
        CALL MOPERA(I+NPL,'F',I+NPL,I+NPL,(XNORM),0.D0)
        CALL MFINAL3(I+NPL)
      ENDDO
C
      do kk=1,nwgt_analysis
         l=(kk-1)*40
         call multitop(NPL+l+1,NPL-1,3,2,'Higgs pT (GeV)',' ','LOG')
         call multitop(NPL+l+2,NPL-1,3,2,'Higgs pT (GeV)',' ','LOG')
         call multitop(NPL+l+3,NPL-1,3,2,'Higgs log(pT/GeV)',' ','LOG')
         call multitop(NPL+l+4,NPL-1,3,2,'Higgs pT (GeV)',' ','LOG')
         call multitop(NPL+l+5,NPL-1,3,2,'Higgs pT (GeV)',' ','LOG')
         call multitop(NPL+l+6,NPL-1,3,2,'Higgs log(pT/GeV)',' ','LOG')
c     
         call multitop(NPL+l+7,NPL-1,3,2,'j1 pT (GeV)',' ','LOG')
         call multitop(NPL+l+8,NPL-1,3,2,'j1 pT (GeV)',' ','LOG')
         call multitop(NPL+l+9,NPL-1,3,2,'j1 log(pT/GeV)',' ','LOG')
         call multitop(NPL+l+10,NPL-1,3,2,'j1 pT (GeV)',' ','LOG')
         call multitop(NPL+l+11,NPL-1,3,2,'j1 pT (GeV)',' ','LOG')
         call multitop(NPL+l+12,NPL-1,3,2,'j1 log(pT/GeV)',' ','LOG')
c
         call multitop(NPL+l+13,NPL-1,3,2,'Inc j pT (GeV)',' ','LOG')
         call multitop(NPL+l+14,NPL-1,3,2,'Inc j pT (GeV)',' ','LOG')
         call multitop(NPL+l+15,NPL-1,3,2,'Inc j log(pT/GeV)',' ','LOG')
         call multitop(NPL+l+16,NPL-1,3,2,'Inc j pT (GeV)',' ','LOG')
         call multitop(NPL+l+17,NPL-1,3,2,'Inc j pT (GeV)',' ','LOG')
         call multitop(NPL+l+18,NPL-1,3,2,'Inc j log(pT/GeV)',' ','LOG')
c
         call multitop(NPL+l+19,NPL-1,3,2,'Higgs y',' ','LOG')
         call multitop(NPL+l+20,NPL-1,3,2,'Higgs y',' ','LOG')
         call multitop(NPL+l+21,NPL-1,3,2,'Higgs y',' ','LOG')
         call multitop(NPL+l+22,NPL-1,3,2,'Higgs y',' ','LOG')
         call multitop(NPL+l+23,NPL-1,3,2,'Higgs y',' ','LOG')
         call multitop(NPL+l+24,NPL-1,3,2,'Higgs y',' ','LOG')
c     
         call multitop(NPL+l+25,NPL-1,3,2,'j1 y',' ','LOG')
         call multitop(NPL+l+26,NPL-1,3,2,'j1 y',' ','LOG')
         call multitop(NPL+l+27,NPL-1,3,2,'j1 y',' ','LOG')
         call multitop(NPL+l+28,NPL-1,3,2,'j1 y',' ','LOG')
         call multitop(NPL+l+29,NPL-1,3,2,'j1 y',' ','LOG')
         call multitop(NPL+l+30,NPL-1,3,2,'j1 y',' ','LOG')
c
         call multitop(NPL+l+31,NPL-1,3,2,'H-j1 y',' ','LOG')
         call multitop(NPL+l+32,NPL-1,3,2,'H-j1 y',' ','LOG')
         call multitop(NPL+l+33,NPL-1,3,2,'H-j1 y',' ','LOG')
         call multitop(NPL+l+34,NPL-1,3,2,'H-j1 y',' ','LOG')
         call multitop(NPL+l+35,NPL-1,3,2,'H-j1 y',' ','LOG')
         call multitop(NPL+l+36,NPL-1,3,2,'H-j1 y',' ','LOG')

         call multitop(NPL+l+37,NPL-1,3,2,'njets',' ','LOG')
         call multitop(NPL+l+38,NPL-1,3,2,'njets',' ','LOG')
         call multitop(NPL+l+39,NPL-1,3,2,'xsec',' ','LOG')
      enddo
c
      CLOSE(99)
      END

C----------------------------------------------------------------------
      SUBROUTINE PYANAL
C     USER''S ROUTINE TO ANALYSE DATA FROM EVENT
C----------------------------------------------------------------------
      implicit none
      include 'reweight0.inc'
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
      integer nwgt_analysis,max_weight,l
      common/c_analysis/nwgt_analysis
      parameter (max_weight=maxscales*maxscales+maxpdfs+1)
      double precision ww(max_weight),www(max_weight)
      common/cww/ww
      DATA TINY/.1D-5/
      DOUBLE PRECISION EVWEIGHT
      COMMON/CEVWEIGHT/EVWEIGHT
      INTEGER IFAIL
      COMMON/CIFAIL/IFAIL
      IF (WW(1).EQ.0D0) THEN
         WRITE(*,*)'WW(1) = 0. Stopping'
         STOP
      ENDIF
c
      IF(IFAIL.EQ.1)RETURN
C INITIALISE
      DO I=1,NMAX
        DO J=1,4
          PP(J,I)=0D0
        ENDDO
      ENDDO
C INCOMING PARTONS MAY TRAVEL IN THE SAME DIRECTION: IT''S A POWER-SUPPRESSED
C EFFECT, SO THROW THE EVENT AWAY
      IF(SIGN(1.D0,P(3,3)).EQ.SIGN(1.D0,P(4,3)))THEN
         WRITE(*,*)'WARNING 111 IN PYANAL'
         GOTO 999
      ENDIF
      DO I=1,nwgt_analysis
         WWW(I)=EVWEIGHT*ww(i)/ww(1)
      ENDDO
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
        IF(ID1.EQ.25)THEN
          IFH=1
          DO IJ=1,5
            PPH(IJ)=P(IHEP,IJ)
          ENDDO
        ENDIF
        IF (IST.LE.10) THEN
          CALL VVSUM(4,PIHEP,PSUM,PSUM)
          ICHSUM=ICHSUM+PYCHGE(ID1)
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
         GOTO 999
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
C
      do kk=1,nwgt_analysis
      l=(kk-1)*40
      call mfill(l+1,pth,WWW(kk))
      call mfill(l+2,pth,WWW(kk))
      if(pth.gt.0.d0)call mfill(l+3,log10(pth),WWW(kk))
      if(abs(yh).le.2.d0)then
         call mfill(l+4,pth,WWW(kk))
         call mfill(l+5,pth,WWW(kk))
         if(pth.gt.0.d0)call mfill(l+6,log10(pth),WWW(kk))
      endif
c
      if(njet.ge.1)then
         call mfill(l+7,ptj1,WWW(kk))
         call mfill(l+8,ptj1,WWW(kk))
         if(ptj1.gt.0.d0)call mfill(l+9,log10(ptj1),WWW(kk))
         if(abs(yj1).le.2.d0)then
            call mfill(l+10,ptj1,WWW(kk))
            call mfill(l+11,ptj1,WWW(kk))
            if(ptj1.gt.0.d0)call mfill(l+12,log10(ptj1),WWW(kk))
         endif
c
         do nj=1,njet
            call mfill(l+13,ptj1,WWW(kk))
            call mfill(l+14,ptj1,WWW(kk))
            if(ptj1.gt.0.d0)call mfill(l+15,log10(ptj1),WWW(kk))
            if(abs(yj1).le.2.d0)then
               call mfill(l+16,ptj1,WWW(kk))
               call mfill(l+17,ptj1,WWW(kk))
               if(ptj1.gt.0d0)call mfill(l+18,log10(ptj1),WWW(kk))
            endif
         enddo
      endif
c
      call mfill(l+19,yh,WWW(kk))
      if(pth.ge.10.d0) call mfill(l+20,yh,WWW(kk))
      if(pth.ge.30.d0) call mfill(l+21,yh,WWW(kk))
      if(pth.ge.50.d0) call mfill(l+22,yh,WWW(kk))
      if(pth.ge.70.d0) call mfill(l+23,yh,WWW(kk))
      if(pth.ge.90.d0) call mfill(l+24,yh,WWW(kk))  
c     
      if(njet.ge.1)then
         call mfill(l+25,yj1,WWW(kk))
         if(ptj1.ge.10.d0) call mfill(l+26,yj1,WWW(kk))
         if(ptj1.ge.30.d0) call mfill(l+27,yj1,WWW(kk))
         if(ptj1.ge.50.d0) call mfill(l+28,yj1,WWW(kk))
         if(ptj1.ge.70.d0) call mfill(l+29,yj1,WWW(kk))
         if(ptj1.ge.90.d0) call mfill(l+30,yj1,WWW(kk))
c     
         call mfill(l+31,yh-yj1,WWW(kk))
         if(ptj1.ge.10.d0) call mfill(l+32,yh-yj1,WWW(kk))
         if(ptj1.ge.30.d0) call mfill(l+33,yh-yj1,WWW(kk))
         if(ptj1.ge.50.d0) call mfill(l+34,yh-yj1,WWW(kk))
         if(ptj1.ge.70.d0) call mfill(l+35,yh-yj1,WWW(kk))
         if(ptj1.ge.90.d0) call mfill(l+36,yh-yj1,WWW(kk))
      endif
c
      call mfill(l+37,njdble,WWW(kk))
      call mfill(l+38,njcdble,WWW(kk))
      call mfill(l+39,1d0,WWW(kk))
c      
      enddo
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
