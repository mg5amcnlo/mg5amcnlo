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
      REAL*4 pi
      parameter (pi=3.14160E0)
      integer j,k
c

      call inihist
      k=0
      call mbook(k+ 1,'t pt',4.e0,0.e0,400.e0)
      call mbook(k+ 2,'t eta',0.2e0,-9.e0,9.e0)
      call mbook(k+ 3,'t log[pt]',0.05e0,0.1e0,5.e0)
      call mbook(k+ 4,'t y',0.2e0,-9.e0,9.e0)

      END


C----------------------------------------------------------------------
      SUBROUTINE PYAEND(IEVT)
C     USER'S ROUTINE FOR TERMINAL CALCULATIONS, HISTOGRAM OUTPUT, ETC
C----------------------------------------------------------------------
      REAL*8 XNORM
      INTEGER I,J,K
      OPEN(UNIT=99,FILE='PYTST.TOP',STATUS='UNKNOWN')
      XNORM=1.D0/IEVT
      DO I=1,100              
 	CALL MFINAL3(I)             
        CALL MCOPY(I,I+100)
        CALL MOPERA(I+100,'F',I+100,I+100,SNGL(XNORM),0.E0)
 	CALL MFINAL3(I+100)             
      ENDDO                          
C
      k=0
      call multitop(100+k+ 1,99,3,2,'t pt',' ','LOG')
      call multitop(100+k+ 2,99,3,2,'t eta',' ','LOG')
      call multitop(100+k+ 3,99,3,2,'t log[pt]',' ','LOG')
      call multitop(100+k+ 4,99,3,2,'t y',' ','LOG')
c
      CLOSE(99)
      END


C----------------------------------------------------------------------
      SUBROUTINE PYANAL
C     USER'S ROUTINE TO ANALYSE DATA FROM EVENT
C----------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
      DOUBLE PRECISION VDOT
      INTEGER ICHSUM,ICHINI,IHEP
      INTEGER ID,ID1,IST,IQ,IT1,ILP,INU,IBQ,IJ
      DOUBLE PRECISION YCUT,PTCUT,pt1,eta1,getpseudorap,yt1,
     # getrapidity,ptlp,ylp,ptnu,ynu,ptbq,ybq,xmw1,getinvm
      DOUBLE PRECISION XPTQ(5),XPLP(5),XPNU(5),XPBQ(5),YPW1(5)
      REAL*8 PI
      PARAMETER (PI=3.14159265358979312D0)
      REAL*8 WWW0
      INTEGER KK,JJ

      integer pychge
      external pydata

      double precision evweight
      common/cevweight/evweight
      double precision p1(5),p2(5),psum(5),pihep(5)

      common/pyjets/n,npad,k(4000,5),p(4000,5),v(4000,5)
      common/pydat1/mstu(200),paru(200),mstj(200),parj(200)
      common/pydat2/kchg(500,4),pmas(500,4),parf(2000),vckm(4,4)
      common/pydat3/mdcy(500,3),mdme(8000,2),brat(8000),kfdp(8000,5)
      common/pysubs/msel,mselpd,msub(500),kfin(2,-40:40),ckin(200)
      common/pypars/mstp(200),parp(200),msti(200),pari(200)
      INTEGER IFAIL
      COMMON/CIFAIL/IFAIL
c
      IF(IFAIL.EQ.1)RETURN
c
C INCOMING PARTONS MAY TRAVEL IN THE SAME DIRECTION: IT'S A POWER-SUPPRESSED
C EFFECT, SO THROW THE EVENT AWAY
      IF(SIGN(1.D0,P(3,3)).EQ.SIGN(1.D0,P(4,3)))THEN
         WRITE(*,*)'WARNING 111 IN PYANAL'
        GOTO 999
      ENDIF
      WWW0=EVWEIGHT
      DO J=1,5
         IF(J.LE.4)THEN
            P1(J)=P(1,J)
            P2(J)=P(2,J)
         ENDIF
         PSUM(J)=0D0
      ENDDO
      CALL VVSUM(4,P1,P2,PSUM)
      CALL VSCA(4,-1D0,PSUM,PSUM)
      ICHSUM=0
      KF1=K(1,2)
      KF2=K(2,2)
      ICHINI=PYCHGE(KF1)+PYCHGE(KF2)
      IQ=0
      DO 100 IHEP=1,N
        DO J=1,5
           PIHEP(J)=P(IHEP,J)
        ENDDO
        IF(K(IHEP,1).LE.10)THEN
           CALL VVSUM(4,PIHEP,PSUM,PSUM)
           ICHSUM=ICHSUM+PYCHGE(K(IHEP,2))
        ENDIF
        IST=K(IHEP,1)
        ID=K(IHEP,2)
        IF(ID.EQ.6) THEN
C FOUND A TOP
          IQ=IQ+1
          IT1=IHEP
        ELSEIF(ID.EQ.-6)THEN
C FOUND AN ANTITOP
          IQ=IQ+1
          IT1=IHEP
        ENDIF
  100 CONTINUE
      IF(IQ.EQ.0)THEN
        WRITE(*,*)"NO TOP NOR ANTITOP FOUND"
        STOP
      ENDIF  
C CHECK MOMENTUM AND CHARGE CONSERVATION
      IF (VDOT(3,PSUM,PSUM).GT.1.E-4*P(1,4)**2) THEN
         WRITE(*,*)'WARNING 112 IN PYANAL',
     &         VDOT(3,PSUM,PSUM),1.E-4*P(1,4)**2
         GOTO 999
      ENDIF
      IF (ICHSUM.NE.ICHINI) THEN
         WRITE(*,*)'ERROR 113 IN PYANAL',ICHSUM,ICHINI
         GOTO 999
      ENDIF
C FILL THE FOUR-MOMENTA
      DO IJ=1,5
        XPTQ(IJ)=P(IT1,IJ)
      ENDDO
C FILL THE HISTOS
      YCUT=2.5D0
c
      pt1=sqrt(xptq(1)**2+xptq(2)**2)
      eta1=getpseudorap(xptq(4),xptq(1),xptq(2),xptq(3))
      yt1=getrapidity(xptq(4),xptq(3))
C
      kk=0
      call mfill(kk+1,sngl(pt1),sngl(WWW0))
      call mfill(kk+2,sngl(eta1),sngl(WWW0))
      if(pt1.gt.0.d0)call mfill(kk+3,sngl(log10(pt1)),sngl(WWW0))
      call mfill(kk+4,sngl(yt1),sngl(WWW0))
 999  RETURN
      END


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
