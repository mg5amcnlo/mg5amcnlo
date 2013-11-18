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
      SUBROUTINE HWAEND
C     USER'S ROUTINE FOR TERMINAL CALCULATIONS, HISTOGRAM OUTPUT, ETC
C----------------------------------------------------------------------
      INCLUDE 'HERWIG65.INC'
      REAL*8 XNORM
      INTEGER I,J,K
      OPEN(UNIT=99,FILE='HERSVB.TOP',STATUS='UNKNOWN')
C XNORM IS SUCH THAT THE CROSS SECTION PER BIN IS IN PB, SINCE THE HERWIG 
C WEIGHT IS IN NB, AND CORRESPONDS TO THE AVERAGE CROSS SECTION
      XNORM=1.D3/DFLOAT(NEVHEP)
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
      SUBROUTINE HWANAL
C     USER'S ROUTINE TO ANALYSE DATA FROM EVENT
C----------------------------------------------------------------------
      INCLUDE 'HERWIG65.INC'
      DOUBLE PRECISION HWVDOT,PSUM(4),XME,PPV(5),PPE(5),PPNU(5),
     # PPDCE(5),PPDCNU(5),WT,ETAEMIN(2),ETAEMAX(2),PTEMIN(2),
     # XMV,PTV,YV,GETRAPIDITY,PTE,THE,ETAE,PTNU,THNU,ETANU,
     # PTDCE,THDCE,ETADCE,PTDCNU,THDCNU,ETADCNU,ETAV,GETPSEUDORAP
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
c
      IDENT=24
C INCOMING PARTONS MAY TRAVEL IN THE SAME DIRECTION: IT'S A POWER-SUPPRESSED
C EFFECT, SO THROW THE EVENT AWAY
      IF(SIGN(1.D0,PHEP(3,4)).EQ.SIGN(1.D0,PHEP(3,5)))THEN
        CALL HWWARN('HWANAL',111)
        GOTO 999
      ENDIF
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
         CALL HWWARN('HWANAL',503)
      ENDIF
C CHECK MOMENTUM AND CHARGE CONSERVATION
      IF (HWVDOT(3,PSUM,PSUM).GT.1.E-4*PHEP(4,1)**2) THEN
         CALL HWUEPR
         CALL HWWARN('HWANAL',112)
         GOTO 999
      ENDIF
      IF (ICHSUM.NE.ICHINI) THEN
         CALL HWUEPR
         CALL HWWARN('HWANAL',113)
         GOTO 999
      ENDIF
      IF(IFV.GT.1.AND.IERROR.EQ.0) THEN
         CALL HWUEPR
         CALL HWWARN('HWANAL',55)
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
