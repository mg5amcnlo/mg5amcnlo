C----------------------------------------------------------------------
      SUBROUTINE RCLOS()
C     DUMMY IF HBOOK IS USED
C----------------------------------------------------------------------
      END


C----------------------------------------------------------------------
      SUBROUTINE HWABEG
C     USER'S ROUTINE FOR INITIALIZATION
C----------------------------------------------------------------------
      INCLUDE 'HEPMC.INC'
      real * 8 xm0,gam,xmlow,xmupp,bin
      real * 4 xmi,xms,pi
      parameter (pi=3.14160E0)
      integer j,k,jpr
      character*5 cc(2)
      data cc/'     ',' cuts'/
c
      call inihist
      do j=1,2
      k=(j-1)*50
c
      xmi=50.d0
      xms=130.d0
      bin=0.8d0
      call mbook(k+ 1,'V pt'//cc(j),2.e0,0.e0,200.e0)
      call mbook(k+ 2,'V pt'//cc(j),10.e0,0.e0,1000.e0)
      call mbook(k+ 3,'V log[pt]'//cc(j),0.05e0,0.1e0,5.e0)
      call mbook(k+ 4,'V y'//cc(j),0.2e0,-9.e0,9.e0)
      call mbook(k+ 5,'V eta'//cc(j),0.2e0,-9.e0,9.e0)
      call mbook(k+ 6,'mV'//cc(j),sngl(bin),xmi,xms)
c
      call mbook(k+ 7,'l pt'//cc(j),2.e0,0.e0,200.e0)
      call mbook(k+ 8,'l pt'//cc(j),10.e0,0.e0,1000.e0)
      call mbook(k+ 9,'l log[pt]'//cc(j),0.05e0,0.1e0,5.e0)
      call mbook(k+10,'l eta'//cc(j),0.2e0,-9.e0,9.e0)
      call mbook(k+11,'lb pt'//cc(j),2.e0,0.e0,200.e0)
      call mbook(k+12,'lb pt'//cc(j),10.e0,0.e0,1000.e0)
      call mbook(k+13,'lb log[pt]'//cc(j),0.05e0,0.1e0,5.e0)
      call mbook(k+14,'lb eta'//cc(j),0.2e0,-9.e0,9.e0)
c
      call mbook(k+15,'llb delta eta'//cc(j),0.2e0,-9.e0,9.e0)
      call mbook(k+16,'llb azimt'//cc(j),pi/20.e0,0.e0,pi)
      call mbook(k+17,'llb log[pi-azimt]'//cc(j),0.05e0,-4.e0,0.1e0)
      call mbook(k+18,'llb inv m'//cc(j),sngl(bin),xmi,xms)
      call mbook(k+19,'llb pt'//cc(j),2.e0,0.e0,200.e0)
      call mbook(k+20,'llb log[pt]'//cc(j),0.05e0,0.1e0,5.e0)
c
      call mbook(k+21,'total'//cc(j),1.e0,-1.e0,1.e0)
      enddo
 999  END


C----------------------------------------------------------------------
      SUBROUTINE HWAEND
C     USER'S ROUTINE FOR TERMINAL CALCULATIONS, HISTOGRAM OUTPUT, ETC
C----------------------------------------------------------------------
      INCLUDE 'HEPMC.INC'
      REAL*8 XNORM
      INTEGER I,J,K
      OPEN(UNIT=99,FILE='HERLL.TOP',STATUS='UNKNOWN')
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
      do j=1,2
      k=(j-1)*50
      call multitop(100+k+ 1,99,3,2,'V pt',' ','LOG')
      call multitop(100+k+ 2,99,3,2,'V pt',' ','LOG')
      call multitop(100+k+ 3,99,3,2,'V log[pt]',' ','LOG')
      call multitop(100+k+ 4,99,3,2,'V y',' ','LOG')
      call multitop(100+k+ 5,99,3,2,'V eta',' ','LOG')
      call multitop(100+k+ 6,99,3,2,'mV',' ','LOG')
      enddo
c
      do j=1,2
      k=(j-1)*50
      call multitop(100+k+ 7,99,3,2,'l pt',' ','LOG')
      call multitop(100+k+ 8,99,3,2,'l pt',' ','LOG')
      call multitop(100+k+ 9,99,3,2,'l log[pt]',' ','LOG')
      call multitop(100+k+10,99,3,2,'l eta',' ','LOG')
      call multitop(100+k+11,99,3,2,'l pt',' ','LOG')
      call multitop(100+k+12,99,3,2,'l pt',' ','LOG')
      call multitop(100+k+13,99,3,2,'l log[pt]',' ','LOG')
      call multitop(100+k+14,99,3,2,'l eta',' ','LOG')
c
      call multitop(100+k+15,99,3,2,'llb deta',' ','LOG')
      call multitop(100+k+16,99,3,2,'llb azi',' ','LOG')
      call multitop(100+k+17,99,3,2,'llb azi',' ','LOG')
      call multitop(100+k+18,99,3,2,'llb inv m',' ','LOG')
      call multitop(100+k+19,99,3,2,'llb pt',' ','LOG')
      call multitop(100+k+20,99,3,2,'llb pt',' ','LOG')
c
      call multitop(100+k+21,99,3,2,'total',' ','LOG')
      enddo
c
      CLOSE(99)
      END

C----------------------------------------------------------------------
      SUBROUTINE HWANAL
C     USER'S ROUTINE TO ANALYSE DATA FROM EVENT
C----------------------------------------------------------------------
      INCLUDE 'HEPMC.INC'
      DOUBLE PRECISION HWVDOT,PSUM(4),PPV(5),YCUT,XMV,PTV,YV,THV,ETAV,
     #  PPL(5),PPLB(5),PTL,YL,THL,ETAL,PLL,ENL,PTLB,YLB,THLB,ETALB,
     #  PLLB,ENLB,PTPAIR,DLL,CLL,AZI,AZINORM,XMLL,DETALLB
      INTEGER ICHSUM,ICHINI,IHEP,IV,IFV,IST,ID,IJ,ID1,JPR,IDENT,
     #  ILL,ILLB,IHRD
      LOGICAL DIDSOF,TEST1,TEST2,flag
      REAL*8 PI,wmass,wgamma,bwcutoff
      PARAMETER (PI=3.14159265358979312D0)
      REAL*8 WWW0,TINY
      INTEGER KK
      DATA TINY/.1D-5/
c
      IF(MOD(NEVHEP,10000).EQ.0)CALL HWAEND
c
      IDENT=24
C INCOMING PARTONS MAY TRAVEL IN THE SAME DIRECTION: IT'S A POWER-SUPPRESSED
C EFFECT, SO THROW THE EVENT AWAY
      IF(SIGN(1.D0,PHEP(3,1)).EQ.SIGN(1.D0,PHEP(3,2)))THEN
        CALL HWWARN('HWANAL',111)
        GOTO 999
      ENDIF
      WWW0=EVWGT

      ICHSUM=0
      DIDSOF=.FALSE.
      ILL=0
      ILLB=0
      DO 100 IHEP=1,NHEP
        IST=ISTHEP(IHEP)      
        ID1=IDHEP(IHEP)
        IF(IST.EQ.1.AND.ID1.EQ.-11)ILLB=IHEP
        IF(IST.EQ.1.AND.ID1.EQ.12)ILL=IHEP
  100 CONTINUE
      IF(ILL.EQ.0.OR.ILLB.EQ.0) THEN
         CALL HWUEPR
         CALL HWWARN('HWANAL',503)
      ENDIF
      DO IJ=1,5
        PPL(IJ)=PHEP(IJ,ILL)
        PPLB(IJ)=PHEP(IJ,ILLB)
        IF(IJ.LT.5)PPV(IJ)=PPL(IJ)+PPLB(IJ)
      ENDDO
      PPV(5)=SQRT(PPV(4)**2-PPV(1)**2-PPV(2)**2-PPV(3)**2)
C FILL THE HISTOS
        YCUT=2.5D0
C Variables of the vector boson
      xmv=ppv(5)
      ptv=sqrt(ppv(1)**2+ppv(2)**2)
      if(abs(ppv(4)-abs(ppv(3))).gt.tiny)then
        yv=0.5d0*log( (ppv(4)+ppv(3))/
     #                (ppv(4)-ppv(3)) )
      else
        yv=sign(1.d0,ppv(3))*1.d8
      endif
      thv = atan2(ptv+tiny,ppv(3))
      etav= -log(tan(thv/2))
C Variables of the leptons
      ptl=sqrt(ppl(1)**2+ppl(2)**2)
      if(abs(ppl(4)-abs(ppl(3))).gt.tiny)then
        yl=0.5d0*log( (ppl(4)+ppl(3))/
     #                (ppl(4)-ppl(3)) )
      else
        yl=sign(1.d0,ppl(3))*1.d8
      endif
      thl = atan2(ptl+tiny,ppl(3))
      etal= -log(tan(thl/2))
      pll = ppl(3)
      enl = ppl(4)
c
      ptlb=sqrt(pplb(1)**2+pplb(2)**2)
      if(abs(pplb(4)-abs(pplb(3))).gt.tiny)then
        ylb=0.5d0*log( (pplb(4)+pplb(3))/
     #                 (pplb(4)-pplb(3)) )
      else
        ylb=sign(1.d0,pplb(3))*1.d8
      endif
      thlb = atan2(ptlb+tiny,pplb(3))
      etalb= -log(tan(thlb/2))
      pllb = pplb(3)
      enlb = pplb(4)
c
      ptpair = dsqrt((ppl(1)+pplb(1))**2+(ppl(2)+pplb(2))**2)
      dll = ppl(1)*pplb(1)+ppl(2)*pplb(2)
      cll=0
      if(ptl.ne.0.and.ptlb.ne.0) cll = dll * (1-tiny)/(ptl*ptlb)
      if(abs(cll).gt.1) then
	write(*,*) ' cosine = ',cll ,dll,ptl,ptlb
	cll = - 1
      endif
      azi = (1-tiny)*acos(cll)
      azinorm = (pi-azi)/pi
      xmll = dsqrt( ppl(5)**2 + ppl(5)**2 + 
     #              2*(enl*enlb - pll*pllb - dll) )
      detallb = etal-etalb
c
      kk=0
      wmass=80.419d0
      wgamma=2.046d0
      bwcutoff=15.d0
      flag=(xmv.ge.wmass-wgamma*bwcutoff.and.
     &      xmv.le.wmass+wgamma*bwcutoff)
      if(flag)then
      call mfill(kk+1,sngl(ptv),sngl(WWW0))
      call mfill(kk+2,sngl(ptv),sngl(WWW0))
      if(ptv.gt.0.d0)call mfill(kk+3,sngl(log10(ptv)),sngl(WWW0))
      call mfill(kk+4,sngl(yv),sngl(WWW0))
      call mfill(kk+5,sngl(etav),sngl(WWW0))
      call mfill(kk+6,sngl(xmv),sngl(WWW0))
c
      call mfill(kk+7,sngl(ptl),sngl(WWW0))
      call mfill(kk+8,sngl(ptl),sngl(WWW0))
      if(ptl.gt.0.d0)call mfill(kk+9,sngl(log10(ptl)),sngl(WWW0))
      call mfill(kk+10,sngl(etal),sngl(WWW0))
      call mfill(kk+11,sngl(ptlb),sngl(WWW0))
      call mfill(kk+12,sngl(ptlb),sngl(WWW0))
      if(ptlb.gt.0.d0)call mfill(kk+13,sngl(log10(ptlb)),sngl(WWW0))
      call mfill(kk+14,sngl(etalb),sngl(WWW0))
c
      call mfill(kk+15,sngl(detallb),sngl(WWW0))
      call mfill(kk+16,sngl(azi),sngl(WWW0))
      if(azinorm.gt.0.d0)
     #  call mfill(kk+17,sngl(log10(azinorm)),sngl(WWW0))
      call mfill(kk+18,sngl(xmll),sngl(WWW0))
      call mfill(kk+19,sngl(ptpair),sngl(WWW0))
      if(ptpair.gt.0)call mfill(kk+20,sngl(log10(ptpair)),sngl(WWW0))
      call mfill(kk+21,sngl(0d0),sngl(WWW0))
c
      kk=50
      if(abs(etav).lt.ycut)then
        call mfill(kk+1,sngl(ptv),sngl(WWW0))
        call mfill(kk+2,sngl(ptv),sngl(WWW0))
        if(ptv.gt.0.d0)call mfill(kk+3,sngl(log10(ptv)),sngl(WWW0))
      endif
      if(ptv.gt.20.d0)then
        call mfill(kk+4,sngl(yv),sngl(WWW0))
        call mfill(kk+5,sngl(etav),sngl(WWW0))
      endif
      if(abs(etav).lt.ycut.and.ptv.gt.20.d0)then
         call mfill(kk+6,sngl(xmv),sngl(WWW0))
         call mfill(kk+21,sngl(0d0),sngl(WWW0))
      endif
c
      if(abs(etal).lt.ycut)then
        call mfill(kk+7,sngl(ptl),sngl(WWW0))
        call mfill(kk+8,sngl(ptl),sngl(WWW0))
        if(ptl.gt.0.d0)call mfill(kk+9,sngl(log10(ptl)),sngl(WWW0))
      endif
      if(ptl.gt.20.d0)call mfill(kk+10,sngl(etal),sngl(WWW0))
      if(abs(etalb).lt.ycut)then
        call mfill(kk+11,sngl(ptlb),sngl(WWW0))
        call mfill(kk+12,sngl(ptlb),sngl(WWW0))
        if(ptlb.gt.0.d0)call mfill(kk+13,sngl(log10(ptlb)),sngl(WWW0))
      endif
      if(ptlb.gt.20.d0)call mfill(kk+14,sngl(etalb),sngl(WWW0))
c
      if( abs(etal).lt.ycut.and.abs(etalb).lt.ycut .and.
     #    ptl.gt.20.d0.and.ptlb.gt.20.d0)then
        call mfill(kk+15,sngl(detallb),sngl(WWW0))
        call mfill(kk+16,sngl(azi),sngl(WWW0))
        if(azinorm.gt.0.d0)
     #    call mfill(kk+17,sngl(log10(azinorm)),sngl(WWW0))
        call mfill(kk+18,sngl(xmll),sngl(WWW0))
        call mfill(kk+19,sngl(ptpair),sngl(WWW0))
        if(ptpair.gt.0) 
     #    call mfill(kk+20,sngl(log10(ptpair)),sngl(WWW0))
      endif
      endif
 999  END


C-----------------------------------------------------------------------
      SUBROUTINE HWWARN(SUBRTN,ICODE)
C-----------------------------------------------------------------------
C     DEALS WITH ERRORS DURING EXECUTION
C     SUBRTN = NAME OF CALLING SUBROUTINE
C     ICODE  = ERROR CODE:    - -1 NONFATAL, KILL EVENT & PRINT NOTHING
C                            0- 49 NONFATAL, PRINT WARNING & CONTINUE
C                           50- 99 NONFATAL, PRINT WARNING & JUMP
C                          100-199 NONFATAL, DUMP & KILL EVENT
C                          200-299    FATAL, TERMINATE RUN
C                          300-399    FATAL, DUMP EVENT & TERMINATE RUN
C                          400-499    FATAL, DUMP EVENT & STOP DEAD
C                          500-       FATAL, STOP DEAD WITH NO DUMP
C-----------------------------------------------------------------------
      INCLUDE 'HEPMC.INC'
      INTEGER ICODE,NRN,IERROR
      CHARACTER*6 SUBRTN
      IF (ICODE.GE.0) WRITE (6,10) SUBRTN,ICODE
   10 FORMAT(/' HWWARN CALLED FROM SUBPROGRAM ',A6,': CODE =',I4)
      IF (ICODE.LT.0) THEN
         IERROR=ICODE
         RETURN
      ELSEIF (ICODE.LT.100) THEN
         WRITE (6,20) NEVHEP,NRN,EVWGT
   20    FORMAT(' EVENT',I8,':   SEEDS =',I11,' &',I11,
     &'  WEIGHT =',E11.4/' EVENT SURVIVES. EXECUTION CONTINUES')
         IF (ICODE.GT.49) RETURN
      ELSEIF (ICODE.LT.200) THEN
         WRITE (6,30) NEVHEP,NRN,EVWGT
   30    FORMAT(' EVENT',I8,':   SEEDS =',I11,' &',I11,
     &'  WEIGHT =',E11.4/' EVENT KILLED.   EXECUTION CONTINUES')
         IERROR=ICODE
         RETURN
      ELSEIF (ICODE.LT.300) THEN
         WRITE (6,40)
   40    FORMAT(' EVENT SURVIVES.  RUN ENDS GRACEFULLY')
c$$$         CALL HWEFIN
         CALL HWAEND
         STOP
      ELSEIF (ICODE.LT.400) THEN
         WRITE (6,50)
   50    FORMAT(' EVENT KILLED: DUMP FOLLOWS.  RUN ENDS GRACEFULLY')
         IERROR=ICODE
c$$$         CALL HWUEPR
c$$$         CALL HWUBPR
c$$$         CALL HWEFIN
         CALL HWAEND
         STOP
      ELSEIF (ICODE.LT.500) THEN
         WRITE (6,60)
   60    FORMAT(' EVENT KILLED: DUMP FOLLOWS.  RUN STOPS DEAD')
         IERROR=ICODE
c$$$         CALL HWUEPR
c$$$         CALL HWUBPR
         STOP
      ELSE
         WRITE (6,70)
   70    FORMAT(' RUN CANNOT CONTINUE')
         STOP
      ENDIF
      END


      subroutine HWUEPR
      INCLUDE 'HEPMC.INC'
      integer ip,i
      PRINT *,' EVENT ',NEVHEP
      DO IP=1,NHEP
         PRINT '(I4,I8,I4,4I4,1P,5D11.3)',IP,IDHEP(IP),ISTHEP(IP),
     &        JMOHEP(1,IP),JMOHEP(2,IP),JDAHEP(1,IP),JDAHEP(2,IP),
     &        (PHEP(I,IP),I=1,5)
      ENDDO
      return
      end

