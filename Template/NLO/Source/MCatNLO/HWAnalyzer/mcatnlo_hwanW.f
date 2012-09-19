C----------------------------------------------------------------------
      SUBROUTINE RCLOS()
C     DUMMY IF HBOOK IS USED
C----------------------------------------------------------------------
      END


C----------------------------------------------------------------------
      SUBROUTINE HWABEG
C     USER'S ROUTINE FOR INITIALIZATION
C----------------------------------------------------------------------
      INCLUDE 'HERWIG65.INC'
      real * 8 xm0,gam,xmlow,xmupp,bin
      real * 4 xmi,xms,pi
      parameter (pi=3.14160E0)
      integer j,k,jpr
      character*5 cc(5)
      data cc/'     ',' cut1',' cut2',' cut3',' cut4'/
c
c$$$      jpr=mod(abs(iproc),10000)/10
      jpr=146
      gammaX=30d0
      emmin=0.d0
      emmax=0.d0
      if(emmin.eq.0.d0.and.emmax.eq.0.d0)then
        if(jpr.lt.140)then
          xm0=rmass(200)
          gam=gamz
        else
          xm0=rmass(198)
          gam=gamw
        endif
        xmupp=xm0+gammax*gam
        xmlow=xm0-gammax*gam
      else
        xm0=(emmin+emmax)/2.d0
        xmupp=emmax
        xmlow=emmin
      endif
      if(abs(xmlow-xmupp).lt.1.d-3)then
        bin=0.5d0
        xmi=sngl(xm0-24.75d0)
        xms=sngl(xm0+25.25d0)
      else
        bin=(xmupp-xmlow)/100.d0
        xmi=sngl(xm0-(49*bin+bin/2))
        xms=sngl(xm0+(50*bin+bin/2))
      endif
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
c
      enddo
c
      do j=1,5
      k=30+(j-1)*5
c
      call mbook(k+ 1,'W pt'//cc(j),2.e0,0.e0,200.e0)
      call mbook(k+ 2,'W y'//cc(j),0.25e0,-9.e0,9.e0)
      call mbook(k+ 3,'mW'//cc(j),sngl(bin),xmi,xms)
c
      enddo
 999  END
C----------------------------------------------------------------------
      SUBROUTINE HWAEND
C     USER'S ROUTINE FOR TERMINAL CALCULATIONS, HISTOGRAM OUTPUT, ETC
C----------------------------------------------------------------------
      INCLUDE 'HERWIG65.INC'
      REAL*8 XNORM
      INTEGER I,J,K
      OPEN(UNIT=99,NAME='HERW.TOP',STATUS='UNKNOWN')
C XNORM IS SUCH THAT THE CROSS SECTION PER BIN IS IN PB, SINCE THE HERWIG 
C WEIGHT IS IN NB, AND CORRESPONDS TO THE AVERAGE CROSS SECTION
      XNORM=1.D3/DFLOAT(NEVHEP)
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
      SUBROUTINE HWANAL
C     USER'S ROUTINE TO ANALYSE DATA FROM EVENT
C----------------------------------------------------------------------
      INCLUDE 'HERWIG65.INC'
      DOUBLE PRECISION HWVDOT,PSUM(4),XME,PPV(5),PPE(5),PPNU(5),
     # PPDCE(5),PPDCNU(5),WT,ETAEMIN(2),ETAEMAX(2),PTEMIN(2),
     # XMV,PTV,YV,GETRAPIDITY,PTE,THE,ETAE,PTNU,THNU,ETANU,
     # PTDCE,THDCE,ETADCE,PTDCNU,THDCNU,ETADCNU
      INTEGER ICHSUM,ICHINI,IHEP,JPR,IDENT,IFV,IST,ID,ID1,IHRD,IV,
     # IJ,IE,INU,J
      LOGICAL DIDSOF,TEST1,TEST2,FLAG
      REAL*8 PI
      PARAMETER (PI=3.14159265358979312D0)
      REAL*8 WWW0,TINY
      INTEGER KK
      DATA TINY/.1D-5/
      DATA XME/5.11D-4/
c
      IF (IERROR.NE.0) RETURN
c$$$c
c$$$      JPR=MOD(ABS(IPROC),10000)/10
      JPR=146
      IF(JPR.EQ.135.OR.JPR.EQ.136.OR.JPR.EQ.137) THEN
        IDENT=23
      ELSEIF(JPR.EQ.145) THEN
        IDENT=24
      ELSEIF(JPR.EQ.146) THEN
        IDENT=24
      ELSEIF(JPR.EQ.147) THEN
        IDENT=-24
      ELSE
        CALL HWWARN('HWANAL',502,*999)
      ENDIF
      IDENT=24
C INCOMING PARTONS MAY TRAVEL IN THE SAME DIRECTION: IT'S A POWER-SUPPRESSED
C EFFECT, SO THROW THE EVENT AWAY
      IF(SIGN(1.D0,PHEP(3,4)).EQ.SIGN(1.D0,PHEP(3,5)))
     #  CALL HWWARN('HWANAL',111,*999)
      WWW0=EVWGT
      CALL HWVSUM(4,PHEP(1,1),PHEP(1,2),PSUM)
      CALL HWVSCA(4,-1D0,PSUM,PSUM)
      ICHSUM=0
      ICHINI=ICHRG(IDHW(1))+ICHRG(IDHW(2))
      DIDSOF=.FALSE.
      IFV=0
      DO 100 IHEP=1,NHEP
        IF (IDHW(IHEP).EQ.16) DIDSOF=.TRUE.
        IF (ISTHEP(IHEP).EQ.1) THEN
          CALL HWVSUM(4,PHEP(1,IHEP),PSUM,PSUM)
          ICHSUM=ICHSUM+ICHRG(IDHW(IHEP))
        ENDIF
        IST=ISTHEP(IHEP)      
        ID=IDHW(IHEP)
        ID1=IDHEP(IHEP)
C Herwig relabels the vector boson V in Drell-Yan; this doesn't happen with
C MC@NLO; in S events, V appears as HARD, in H as V, but with status 155
C rather than 195. We add here 195 for future compatibility
        IF(IPROC.LT.0)THEN
          TEST1=IST.EQ.155.OR.IST.EQ.195
          TEST2=ID1.EQ.IDENT
          IF(IST.EQ.120.AND.ID1.EQ.0)IHRD=IHEP
        ELSE
          TEST1=IST.EQ.120.OR.IST.EQ.195
          TEST2=ABS(ID1).EQ.IDENT
        ENDIF
        IF(TEST1.AND.TEST2)THEN
          IV=IHEP
          IFV=IFV+1
        ENDIF
  100 CONTINUE
      IF(IPROC.LT.0.AND.IFV.EQ.0)THEN
        IV=IHRD
        IFV=1
      ENDIF
      DO IJ=1,5
        PPV(IJ)=PHEP(IJ,IV)
      ENDDO
      IF(IFV.EQ.0.AND.IERROR.EQ.0) THEN
         CALL HWUEPR
         CALL HWWARN('HWANAL',503,*999)
      ENDIF
C CHECK MOMENTUM AND CHARGE CONSERVATION
      IF (HWVDOT(3,PSUM,PSUM).GT.1.E-4*PHEP(4,1)**2) THEN
         CALL HWUEPR
         CALL HWWARN('HWANAL',112,*999)
      ENDIF
      IF (ICHSUM.NE.ICHINI) THEN
         CALL HWUEPR
         CALL HWWARN('HWANAL',113,*999)
      ENDIF
      IF(IFV.GT.1.AND.IERROR.EQ.0) THEN
         CALL HWUEPR
         CALL HWWARN('HWANAL',55,*999)
      ENDIF
C FIND E and NU_E
      IF( ABS(IDHEP(JDAHEP(1,JDAHEP(1,IV)))).EQ.11 .AND.
     #    ABS(IDHEP(JDAHEP(1,JDAHEP(2,IV)))).EQ.12 )THEN
        IE=JDAHEP(1,JDAHEP(1,IV))
        INU=JDAHEP(1,JDAHEP(2,IV))
      ELSEIF( ABS(IDHEP(JDAHEP(1,JDAHEP(2,IV)))).EQ.11 .AND.
     #        ABS(IDHEP(JDAHEP(1,JDAHEP(1,IV)))).EQ.12 )THEN
        IE=JDAHEP(1,JDAHEP(2,IV))
        INU=JDAHEP(1,JDAHEP(1,IV))
      ELSE
        CALL HWUEPR
        WRITE(*,*)IV,JDAHEP(1,JDAHEP(2,IV)),JDAHEP(1,JDAHEP(1,IV))
        CALL HWWARN('HWANAL',504,*999)
      ENDIF
      DO IJ=1,5
        PPE(IJ)=PHEP(IJ,IE)
        PPNU(IJ)=PHEP(IJ,INU)
      ENDDO
C CHECK THAT THE LEPTONS ARE FINAL-STATE LEPTONS
      IF( ABS(IDHEP(IE)).LT.11 .OR. ABS(IDHEP(IE)).GT.16 .OR.
     #    ABS(IDHEP(INU)).LT.11 .OR. ABS(IDHEP(INU)).GT.16 )THEN
        CALL HWUEPR
        CALL HWWARN('HWANAL',505,*999)
      ENDIF
      IF( (ISTHEP(IE).NE.1.AND.ISTHEP(IE).NE.195) .OR.
     #    (ISTHEP(INU).NE.1.AND.ISTHEP(INU).NE.195) )THEN
        CALL HWUEPR
        CALL HWWARN('HWANAL',506,*999)
      ENDIF
C INCLUDE LEPTONS RESULTING FROM ISOTROPIC W DECAY
      PPDCE(5)=XME
      PPDCNU(5)=0.D0
      CALL PDECAY(PPV,PPDCE,PPDCNU,WT)
C FILL THE HISTOS
      IF(PBEAM1.LT.2000)THEN
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
C--- Writte by MLM, modified by SF
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

