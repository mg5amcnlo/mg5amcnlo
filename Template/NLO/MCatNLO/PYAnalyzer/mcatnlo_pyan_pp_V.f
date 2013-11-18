C----------------------------------------------------------------------
      SUBROUTINE RCLOS()
C     DUMMY IF HBOOK IS USED
C----------------------------------------------------------------------
      END


C----------------------------------------------------------------------
      SUBROUTINE PYABEG
C     USER'S ROUTINE FOR INITIALIZATION
C----------------------------------------------------------------------
      implicit double precision(a-h, o-z)
      implicit integer(i-n)
      common/pydat2/kchg(500,4),pmas(500,4),parf(2000),vckm(4,4)
      real * 8 xm0,gam,xmlow,xmupp,bin
      real * 8 xmi,xms,pi
      PARAMETER (PI=3.14159265358979312D0)
      integer j,k,jpr
      character*5 cc(5)
      data cc/'     ',' cut1',' cut2',' cut3',' cut4'/

      xmi=40.d0
      xms=120.d0
      bin=1.0d0

      call inihist

      do j=1,1
      k=30+(j-1)*5

      call mbook(k+ 1,'W pt'//cc(j),2.d0,0.d0,200.d0)
      call mbook(k+ 2,'W log pt'//cc(j),0.05d0,0.d0,5.d0)
      call mbook(k+ 3,'W y'//cc(j),0.25d0,-9.d0,9.d0)
      call mbook(k+ 4,'W eta'//cc(j),0.25d0,-9.d0,9.d0)
      call mbook(k+ 5,'mW'//cc(j),(bin),xmi,xms)

      enddo
 999  END
C----------------------------------------------------------------------
      SUBROUTINE PYAEND(IEVT)
C     USER'S ROUTINE FOR TERMINAL CALCULATIONS, HISTOGRAM OUTPUT, ETC
C----------------------------------------------------------------------
      REAL*8 XNORM
      INTEGER I,J,K
      OPEN(UNIT=99,FILE='PYTSVB.TOP',STATUS='UNKNOWN')
      XNORM=1.D0/IEVT
      DO I=1,100              
 	CALL MFINAL3(I)             
        CALL MCOPY(I,I+100)
        CALL MOPERA(I+100,'F',I+100,I+100,(XNORM),0.D0)
 	CALL MFINAL3(I+100)             
      ENDDO                          
C
      do j=1,1
      k=30+(j-1)*5
      call multitop(100+k+ 1,99,3,2,'W pt',' ','LOG')
      call multitop(100+k+ 2,99,3,2,'W log pt',' ','LOG')
      call multitop(100+k+ 3,99,3,2,'W y',' ','LOG')
      call multitop(100+k+ 4,99,3,2,'W eta',' ','LOG')
      call multitop(100+k+ 5,99,3,2,'mW',' ','LOG')
      enddo
      CLOSE(99)
      END

C----------------------------------------------------------------------
      SUBROUTINE PYANAL
C     USER'S ROUTINE TO ANALYSE DATA FROM EVENT
C----------------------------------------------------------------------
      implicit double precision(a-h, o-z)
      implicit integer(i-n)
      DOUBLE PRECISION PSUM(4),XME,PPV(5),PPE(5),PPNU(5),
     # PPDCE(5),PPDCNU(5),WT,ETAEMIN(2),ETAEMAX(2),PTEMIN(2),
     # XMV,PTV,YV,GETRAPIDITY,PTE,THE,ETAE,PTNU,THNU,ETANU,
     # PTDCE,THDCE,ETADCE,PTDCNU,THDCNU,ETADCNU,ETAV,GETPSEUDORAP
      INTEGER ICHSUM,ICHINI,IHEP,JPR,IDENT,IFV,IST,ID,ID1,IHRD,IV,
     # IJ,IE,INU,J
      integer pychge
      external pydata
      common/pyjets/n,npad,k(4000,5),p(4000,5),v(4000,5)
      common/pydat1/mstu(200),paru(200),mstj(200),parj(200)
      common/pydat2/kchg(500,4),pmas(500,4),parf(2000),vckm(4,4)
      common/pydat3/mdcy(500,3),mdme(8000,2),brat(8000),kfdp(8000,5)
      common/pysubs/msel,mselpd,msub(500),kfin(2,-40:40),ckin(200)
      common/pypars/mstp(200),parp(200),msti(200),pari(200)
      LOGICAL DIDSOF,TEST1,TEST2,FLAG
      REAL*8 PI
      PARAMETER (PI=3.14159265358979312D0)
      REAL*8 WWW0,TINY,p1(4),p2(4),pihep(4)
      INTEGER KK
      DATA TINY/.1D-5/
      DATA XME/5.11D-4/
      DOUBLE PRECISION EVWEIGHT
      COMMON/CEVWEIGHT/EVWEIGHT
      INTEGER IFAIL
      COMMON/CIFAIL/IFAIL
      SAVE INOBOSON
c
      IF(IFAIL.EQ.1)RETURN
c
      IDENT=24
C INCOMING PARTONS MAY TRAVEL IN THE SAME DIRECTION: IT'S A POWER-SUPPRESSED
C EFFECT, SO THROW THE EVENT AWAY
      IF(SIGN(1.D0,P(3,3)).EQ.SIGN(1.D0,P(4,3)))THEN
         WRITE(*,*)'WARNING 502 IN PYANAL'
         GOTO 999
      ENDIF
      WWW0=EVWEIGHT
      do i=1,4
         p1(i)=p(1,i)
         p2(i)=p(2,i)
      enddo
      CALL VVSUM(4,P1,P2,PSUM)
      CALL VSCA(4,-1D0,PSUM,PSUM)
      ICHSUM=0
      kf1=k(1,2)
      kf2=k(2,2)
      ICHINI=pychge(kf1)+pychge(kf2)
      IFV=0
      DO 100 IHEP=1,N
        do j=1,4
          pihep(j)=p(ihep,j)
        enddo
        IST=K(IHEP,1)      
        ID1=K(IHEP,2)
        IORI=K(IHEP,3)
        IF (IST.LE.10) THEN
          CALL VVSUM(4,PIHEP,PSUM,PSUM)
          ICHSUM=ICHSUM+pychge(ID1)
        ENDIF
        TEST1=IORI.EQ.0
        TEST2=ID1.EQ.IDENT
        IF(TEST1.AND.TEST2)IV0=IHEP
        TEST1=IORI.EQ.IV0
        IF(TEST1.AND.TEST2)THEN
          IV=IHEP
          IFV=IFV+1
          DO IJ=1,5
             PPV(IJ)=P(IHEP,ij)
          ENDDO
        ENDIF
  100 CONTINUE
      IF(IFV.EQ.0) THEN
         INOBOSON=INOBOSON+1
         WRITE(*,*)'WARNING 503 IN PYANAL: NO WEAK BOSON ',INOBOSON
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
      IF(IFV.GT.1) THEN
         WRITE(*,*)'WARNING 55 IN PYANAL'
         GOTO 999
      ENDIF
C FILL THE HISTOS
C Variables of the vector boson
      xmv=ppv(5)
      ptv=sqrt(ppv(1)**2+ppv(2)**2)
      yv=getrapidity(ppv(4),ppv(3))
      etav=getpseudorap(ppv(4),ppv(1),ppv(2),ppv(3))
C
      do j=1,1
        kk=30+(j-1)*5
          call mfill(kk+1,(ptv),(WWW0))
          if(ptv.gt.0)call mfill(kk+2,(log10(ptv)),(WWW0))
          call mfill(kk+3,(yv),(WWW0))
          call mfill(kk+4,(etav),(WWW0))
          call mfill(kk+5,(xmv),(WWW0))
      enddo
C
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

