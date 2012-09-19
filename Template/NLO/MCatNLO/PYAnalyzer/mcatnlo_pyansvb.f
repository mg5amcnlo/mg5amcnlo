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
      real * 4 xmi,xms,pi
      parameter (pi=3.14160E0)
      integer j,k,jpr
      character*5 cc(5)
      data cc/'     ',' cut1',' cut2',' cut3',' cut4'/
c
c$$$      jpr=mod(abs(iproc),10000)/10
c$$$      if(emmin.eq.0.d0.and.emmax.eq.0.d0)then
c$$$        if(jpr.lt.140)then
c$$$          xm0=pmas(23,1)
c$$$          gam=gamz
c$$$        else
c$$$          xm0=pmas(24,1)
c$$$          gam=gamw
c$$$        endif
c$$$        xmupp=xm0+gammax*gam
c$$$        xmlow=xm0-gammax*gam
c$$$      else
c$$$        xm0=(emmin+emmax)/2.d0
c$$$        xmupp=emmax
c$$$        xmlow=emmin
c$$$      endif
c$$$      if(abs(xmlow-xmupp).lt.1.d-3)then
c$$$        bin=0.5d0
c$$$        xmi=sngl(xm0-24.75d0)
c$$$        xms=sngl(xm0+25.25d0)
c$$$      else
c$$$        bin=(xmupp-xmlow)/100.d0
c$$$        xmi=sngl(xm0-(49*bin+bin/2))
c$$$        xms=sngl(xm0+(50*bin+bin/2))
c$$$      endif
      call inihist
c
      do j=1,3
      k=(j-1)*10
c
      call mbook(k+ 1,'e pt'//cc(j),2.e0,0.e0,200.e0)
      call mbook(k+ 2,'e eta'//cc(j),0.25e0,-9.e0,9.e0)
      call mbook(k+ 3,'nu pt'//cc(j),2.e0,0.e0,200.e0)
      call mbook(k+ 4,'nu eta'//cc(j),0.25e0,-9.e0,9.e0)
c
      call mbook(k+ 5,'DCe pt'//cc(j),2.e0,0.e0,200.e0)
      call mbook(k+ 6,'DCe eta'//cc(j),0.25e0,-9.e0,9.e0)
      call mbook(k+ 7,'DCnu pt'//cc(j),2.e0,0.e0,200.e0)
      call mbook(k+ 8,'DCnu eta'//cc(j),0.25e0,-9.e0,9.e0)

      enddo

      do j=1,5
      k=30+(j-1)*5

      bin=1.0d0
      xmi=40.d0
      xms=140.d0
      call mbook(k+ 1,'W pt'//cc(j),2.e0,0.e0,200.e0)
      call mbook(k+ 2,'W y'//cc(j),0.25e0,-9.e0,9.e0)
      call mbook(k+ 3,'mW'//cc(j),sngl(bin),xmi,xms)

      enddo
 999  END
C----------------------------------------------------------------------
      SUBROUTINE PYAEND(IEVT)
C     USER'S ROUTINE FOR TERMINAL CALCULATIONS, HISTOGRAM OUTPUT, ETC
C----------------------------------------------------------------------
      REAL*8 XNORM
      INTEGER I,J,Ke
      OPEN(UNIT=99,NAME='PYTSB.TOP',STATUS='UNKNOWN')
      XNORM=1.D0/IEVT
      DO I=1,100              
 	CALL MFINAL3(I)             
        CALL MCOPY(I,I+100)
        CALL MOPERA(I+100,'F',I+100,I+100,SNGL(XNORM),0.E0)
 	CALL MFINAL3(I+100)
      ENDDO
C
      do j=1,5
      k=30+(j-1)*5
      call multitop(100+k+ 1,99,3,2,'W pt',' ','LOG')
      call multitop(100+k+ 2,99,3,2,'W y',' ','LOG')
      call multitop(100+k+ 3,99,3,2,'mW',' ','LOG')
      enddo
c
      do j=1,3
      k=(j-1)*10
c
      call multitop(100+k+ 1,99,3,2,'e pt',' ','LOG')
      call multitop(100+k+ 2,99,3,2,'e eta',' ','LOG')
      call multitop(100+k+ 3,99,3,2,'nu pt',' ','LOG')
      call multitop(100+k+ 4,99,3,2,'nu eta',' ','LOG')
c
      call multitop(100+k+ 5,99,3,2,'DCe pt',' ','LOG')
      call multitop(100+k+ 6,99,3,2,'DCe eta',' ','LOG')
      call multitop(100+k+ 7,99,3,2,'DCnu pt',' ','LOG')
      call multitop(100+k+ 8,99,3,2,'DCnu eta',' ','LOG')
c
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
     # PTDCE,THDCE,ETADCE,PTDCNU,THDCNU,ETADCNU
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
      LOGICAL DIDSOF,TEST1,TEST2,TEST3,TEST4,TEST1E,TEST1NU,FLAG
      REAL*8 PI
      PARAMETER (PI=3.14159265358979312D0)
      REAL*8 WWW0,TINY,p1(4),p2(4),pihep(4)
      INTEGER KK

      DOUBLE PRECISION EVWEIGHT
      COMMON/CEVWEIGHT/EVWEIGHT

      DATA TINY/.1D-5/
      DATA XME/5.11D-4/
      SAVE INOBOSON
c
C--DECIDE IDENTITY OF THE VECTOR BOSON, ACCORDING TO THE PDG CODE
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
        TEST3=ABS(ID1).EQ.11
        TEST4=ABS(ID1).EQ.12
        IF(TEST1.AND.TEST2)IV0=IHEP
        TEST1=IORI.EQ.IV0
        IF(TEST1.AND.TEST2)THEN
          IV1=IHEP
          IFV=IFV+1
          DO IJ=1,5
             PPV(IJ)=P(IHEP,ij)
          ENDDO
        ENDIF
        IF(TEST1.AND.TEST3)IE0=IHEP
        IF(TEST1.AND.TEST4)INU0=IHEP
        TEST1E=IORI.EQ.IE0
        TEST1NU=IORI.EQ.INU0
        IF(TEST1E.AND.TEST3)IE=IHEP
        IF(TEST1NU.AND.TEST4)INU=IHEP
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
      DO IJ=1,5
        PPE(IJ)=P(IE,IJ)
        PPNU(IJ)=P(INU,IJ)
      ENDDO
C CHECK THAT THE LEPTONS ARE FINAL-STATE LEPTONS
      IF( ABS(K(IE,2)).LT.11 .OR. ABS(K(IE,2)).GT.16 .OR.
     #    ABS(K(INU,2)).LT.11 .OR. ABS(K(INU,2)).GT.16 )THEN
         WRITE(*,*)'WARNING 505 IN PYANAL'
         GOTO 999
      ENDIF

      IF( (K(IE,1).NE.1).OR.K(INU,1).NE.1 ) THEN
         WRITE(*,*)'WARNING 505 IN PYANAL'
         GOTO 999
      ENDIF
C INCLUDE LEPTONS RESULTING FROM ISOTROPIC W DECAY
      PPDCE(5)=XME
      PPDCNU(5)=0.D0
      CALL PDECAY(PPV,PPDCE,PPDCNU,WT)
C FILL THE HISTOS
      IF(PBEAM1.LT.5000)THEN
        ETAEMIN(1)=0.D0
        ETAEMAX(1)=1.D0
        PTEMIN(1)=20.D0
        ETAEMIN(2)=1.D0
        ETAEMAX(2)=2.5D0
        PTEMIN(2)=20.D0
      ELSE
        ETAEMIN(1)=0.D0
        ETAEMAX(1)=2.5D0
        PTEMIN(1)=20.D0
        ETAEMIN(2)=0.D0
        ETAEMAX(2)=2.5D0
        PTEMIN(2)=40.D0
      ENDIF
C Variables of the vector boson
      xmv=ppv(5)
      ptv=sqrt(ppv(1)**2+ppv(2)**2)
      yv=getrapidity(ppv(4),ppv(3))
C Variables of the leptons
      pte=sqrt(ppe(1)**2+ppe(2)**2)
      the = atan2(pte+tiny,ppe(3))
      etae= -log(tan(the/2))
      ptnu=sqrt(ppnu(1)**2+ppnu(2)**2)
      thnu = atan2(ptnu+tiny,ppnu(3))
      etanu= -log(tan(thnu/2))
C Variables of the leptons coming from W isotropic decay
      ptdce=sqrt(ppdce(1)**2+ppdce(2)**2)
      thdce = atan2(ptdce+tiny,ppdce(3))
      etadce= -log(tan(thdce/2))
      ptdcnu=sqrt(ppdcnu(1)**2+ppdcnu(2)**2)
      thdcnu = atan2(ptdcnu+tiny,ppdcnu(3))
      etadcnu= -log(tan(thdcnu/2))
C
      do j=1,5
        kk=30+(j-1)*5
        flag=.false.
        if(j.eq.1)then
          flag=.true.
        elseif(j.le.3)then
          if( ptnu.ge.20.d0 .and.
     #        pte.ge.ptemin(j-1) .and.
     #        abs(etae).ge.etaemin(j-1) .and.
     #        abs(etae).le.etaemax(j-1) )flag=.true.
        elseif(j.le.5)then
          if( ptdcnu.ge.20.d0 .and.
     #        ptdce.ge.ptemin(j-3) .and.
     #        abs(etadce).ge.etaemin(j-3) .and.
     #        abs(etadce).le.etaemax(j-3) )flag=.true.
        endif
        if(flag)then
          call mfill(kk+1,sngl(ptv),sngl(WWW0))
          call mfill(kk+2,sngl(yv),sngl(WWW0))
          call mfill(kk+3,sngl(xmv),sngl(WWW0))
        endif
      enddo
C
      kk=0
      call mfill(kk+1,sngl(pte),sngl(WWW0))
      call mfill(kk+2,sngl(etae),sngl(WWW0))
      call mfill(kk+3,sngl(ptnu),sngl(WWW0))
      call mfill(kk+4,sngl(etanu),sngl(WWW0))
      call mfill(kk+5,sngl(ptdce),sngl(WWW0))
      call mfill(kk+6,sngl(etadce),sngl(WWW0))
      call mfill(kk+7,sngl(ptdcnu),sngl(WWW0))
      call mfill(kk+8,sngl(etadcnu),sngl(WWW0))
C
      do j=2,3
        kk=(j-1)*10
        if( ptnu.ge.20.d0 .and.
     #      abs(etae).ge.etaemin(j-1) .and.
     #      abs(etae).le.etaemax(j-1) )
     #        call mfill(kk+1,sngl(pte),sngl(WWW0))
        if( ptnu.ge.20.d0 .and.
     #      pte.ge.ptemin(j-1) )
     #        call mfill(kk+2,sngl(etae),sngl(WWW0))
        if( pte.ge.ptemin(j-1) .and.
     #      abs(etae).ge.etaemin(j-1) .and.
     #      abs(etae).le.etaemax(j-1) )then
              call mfill(kk+3,sngl(ptnu),sngl(WWW0))
              if(ptnu.ge.20.d0)
     #          call mfill(kk+4,sngl(etanu),sngl(WWW0))
        endif
C
        if( ptdcnu.ge.20.d0 .and.
     #      abs(etadce).ge.etaemin(j-1) .and.
     #      abs(etadce).le.etaemax(j-1) )
     #        call mfill(kk+5,sngl(ptdce),sngl(WWW0))
        if( ptdcnu.ge.20.d0 .and.
     #      ptdce.ge.ptemin(j-1) )
     #        call mfill(kk+6,sngl(etadce),sngl(WWW0))
        if( ptdce.ge.ptemin(j-1) .and.
     #      abs(etadce).ge.etaemin(j-1) .and.
     #      abs(etadce).le.etaemax(j-1) )then
              call mfill(kk+7,sngl(ptdcnu),sngl(WWW0))
              if(ptdcnu.ge.20.d0)
     #          call mfill(kk+8,sngl(etadcnu),sngl(WWW0))
        endif
      enddo
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


      SUBROUTINE PDECAY(P,Q1,Q2,WT)
C--- Decays a particle with momentum P into two particles with momenta
C--- Q(1) and Q(2). WT is the phase space density beta_cm
C--- The decay is spherically symmetric in the decay C_of_M frame
C--- Written by MLM, modified by SF
      implicit none
      double precision pi,twopi
      PARAMETER(PI=3.14159,TWOPI=2.*PI)
      double precision q2e,qp,ctheta,stheta,phi,qplab,qplon,qptr,pmod,
     $     ptr,wt,bet,gam,randa
      double precision P(5),Q1(5),Q2(5),V(3),U(3),pm,q1m,q2m
      double precision x1,x2
      integer i,iseed
      data iseed/1/
c
      PM=P(5)
      Q1M=Q1(5)
      Q2M=Q2(5)
      X1=RANDA(ISEED)
      X2=RANDA(ISEED)
      Q2E=(PM**2-Q1M**2+Q2M**2)/(2.*PM)
      QP=SQRT(MAX(Q2E**2-Q2M**2,0.d0))
      CTHETA=2.*real(x1)-1.
      STHETA=SQRT(1.-CTHETA**2)
      PHI=TWOPI*real(x2)
      QPLON=QP*CTHETA
      QPTR=QP*STHETA
      PMOD=SQRT(P(1)**2+P(2)**2+P(3)**2)
      PTR=SQRT(P(2)**2+P(3)**2)                              

C--- if the decaying particle moves along the X axis:
      IF(PTR.LT.1.E-4) THEN
        V(1)=0.
        V(2)=1.
        V(3)=0.
        U(1)=0.
        U(2)=0.
        U(3)=1.
      ELSE
C--- 
        V(1)=0.
        V(2)=P(3)/PTR
        V(3)=-P(2)/PTR
        U(1)=PTR/PMOD
        U(2)=-P(1)*P(2)/PTR/PMOD
        U(3)=-P(1)*P(3)/PTR/PMOD
      ENDIF
      GAM=P(4)/PM
      BET=PMOD/P(4)
      QPLAB=GAM*(QPLON+BET*Q2E)
      DO I=1,3
      Q2(I)=QPLAB*P(I)/PMOD+QPTR*(V(I)*SIN(PHI)+U(I)*COS(PHI))
      Q1(I)=P(I)-Q2(I)
      END DO
      Q2(4)=GAM*(Q2E+BET*QPLON)
      Q1(4)=P(4)-Q2(4)
      WT=2.*QP/PM
      END            


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

