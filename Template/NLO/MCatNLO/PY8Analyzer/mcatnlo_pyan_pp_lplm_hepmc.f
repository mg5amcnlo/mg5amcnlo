C----------------------------------------------------------------------
      SUBROUTINE RCLOS()
C     DUMMY IF HBOOK IS USED
C----------------------------------------------------------------------
      END


C----------------------------------------------------------------------
      SUBROUTINE PYABEG
C     USER'S ROUTINE FOR INITIALIZATION
C----------------------------------------------------------------------
      INCLUDE 'HEPMC.INC'
      real * 8 xm0,gam,xmlow,xmupp,bin
      real * 8 xmi,xms,pi
      PARAMETER (PI=3.14159265358979312D0)
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
      call mbook(k+ 1,'V pt'//cc(j),2.d0,0.d0,200.d0)
      call mbook(k+ 2,'V pt'//cc(j),10.d0,0.d0,1000.d0)
      call mbook(k+ 3,'V log[pt]'//cc(j),0.05d0,0.1d0,5.d0)
      call mbook(k+ 4,'V y'//cc(j),0.2d0,-9.d0,9.d0)
      call mbook(k+ 5,'V eta'//cc(j),0.2d0,-9.d0,9.d0)
      call mbook(k+ 6,'mV'//cc(j),(bin),xmi,xms)
c
      call mbook(k+ 7,'l pt'//cc(j),2.d0,0.d0,200.d0)
      call mbook(k+ 8,'l pt'//cc(j),10.d0,0.d0,1000.d0)
      call mbook(k+ 9,'l log[pt]'//cc(j),0.05d0,0.1d0,5.d0)
      call mbook(k+10,'l eta'//cc(j),0.2d0,-9.d0,9.d0)
      call mbook(k+11,'lb pt'//cc(j),2.d0,0.d0,200.d0)
      call mbook(k+12,'lb pt'//cc(j),10.d0,0.d0,1000.d0)
      call mbook(k+13,'lb log[pt]'//cc(j),0.05d0,0.1d0,5.d0)
      call mbook(k+14,'lb eta'//cc(j),0.2d0,-9.d0,9.d0)
c
      call mbook(k+15,'llb delta eta'//cc(j),0.2d0,-9.d0,9.d0)
      call mbook(k+16,'llb azimt'//cc(j),pi/20.d0,0.d0,pi)
      call mbook(k+17,'llb log[pi-azimt]'//cc(j),0.05d0,-4.d0,0.1d0)
      call mbook(k+18,'llb inv m'//cc(j),(bin),xmi,xms)
      call mbook(k+19,'llb pt'//cc(j),2.d0,0.d0,200.d0)
      call mbook(k+20,'llb log[pt]'//cc(j),0.05d0,0.1d0,5.d0)
c
      call mbook(k+21,'total'//cc(j),1.d0,-1.d0,1.d0)
      enddo
 999  END


C----------------------------------------------------------------------
      SUBROUTINE PYAEND(IEVTTOT)
C     USER'S ROUTINE FOR TERMINAL CALCULATIONS, HISTOGRAM OUTPUT, ETC
C----------------------------------------------------------------------
      INCLUDE 'HEPMC.INC'
      REAL*8 XNORM
      INTEGER I,J,K
      OPEN(UNIT=99,FILE='PYTLL.TOP',STATUS='UNKNOWN')
C XNORM IS SUCH THAT THE CROSS SECTION PER BIN IS IN PB, SINCE THE HERWIG 
C WEIGHT IS IN NB, AND CORRESPONDS TO THE AVERAGE CROSS SECTION
      XNORM=IEVTTOT/DFLOAT(NEVHEP)
      DO I=1,100              
 	CALL MFINAL3(I)             
        CALL MCOPY(I,I+100)
        CALL MOPERA(I+100,'F',I+100,I+100,(XNORM),0.D0)
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
      SUBROUTINE PYANAL
C     USER'S ROUTINE TO ANALYSE DATA FROM EVENT
C----------------------------------------------------------------------
      INCLUDE 'HEPMC.INC'
      DOUBLE PRECISION HWVDOT,PSUM(4),PPV(5),YCUT,XMV,PTV,YV,THV,ETAV,
     #  PPL(5),PPLB(5),PTL,YL,THL,ETAL,PLL,ENL,PTLB,YLB,THLB,ETALB,
     #  PLLB,ENLB,PTPAIR,DLL,CLL,AZI,AZINORM,XMLL,DETALLB
      INTEGER ICHSUM,ICHINI,IHEP,IV,IFV,IST,ID,IJ,ID1,JPR,IDENT,
     #  ILL,ILLB,IHRD
      LOGICAL DIDSOF,TEST1,TEST2,flag
      REAL*8 PI,wmass,wgamma,bwcutoff,getinvm,getdelphi,getrapidity,
     &getpseudorap
      PARAMETER (PI=3.14159265358979312D0)
      REAL*8 WWW0,TINY
      INTEGER KK
      DATA TINY/.1D-5/
c
      IF(MOD(NEVHEP,10000).EQ.0)RETURN
c
      IDENT=24
C INCOMING PARTONS MAY TRAVEL IN THE SAME DIRECTION: IT'S A POWER-SUPPRESSED
C EFFECT, SO THROW THE EVENT AWAY
      IF(SIGN(1.D0,PHEP(3,1)).EQ.SIGN(1.D0,PHEP(3,2)))THEN
        CALL HWWARN('PYANAL',111)
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
         CALL HWWARN('PYANAL',503)
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
      xmv=getinvm(ppv(4),ppv(1),ppv(2),ppv(3))
      ptv=sqrt(ppv(1)**2+ppv(2)**2)
      yv=getrapidity(ppv(4),ppv(3))
      etav=getpseudorap(ppv(4),ppv(1),ppv(2),ppv(3))
C Variables of the leptons
      ptl=sqrt(ppl(1)**2+ppl(2)**2)
      yl=getrapidity(ppl(4),ppl(3))
      etal=getpseudorap(ppl(4),ppl(1),ppl(2),ppl(3))
c
      ptlb=sqrt(pplb(1)**2+pplb(2)**2)
      ylb=getrapidity(pplb(4),pplb(3))
      etalb=getpseudorap(pplb(4),pplb(1),pplb(2),pplb(3))
c
      ptpair=ptv
      azi=getdelphi(ppl(1),pplb(1),ppl(2),pplb(2))
      azinorm=(pi-azi)/pi
      xmll=xmv
      detallb=etal-etalb
c
      kk=0
      wmass=80.419d0
      wgamma=2.046d0
      bwcutoff=15.d0
      flag=(xmv.ge.wmass-wgamma*bwcutoff.and.
     &      xmv.le.wmass+wgamma*bwcutoff)
      if(flag)then
      call mfill(kk+1,(ptv),(WWW0))
      call mfill(kk+2,(ptv),(WWW0))
      if(ptv.gt.0.d0)call mfill(kk+3,(log10(ptv)),(WWW0))
      call mfill(kk+4,(yv),(WWW0))
      call mfill(kk+5,(etav),(WWW0))
      call mfill(kk+6,(xmv),(WWW0))
c
      call mfill(kk+7,(ptl),(WWW0))
      call mfill(kk+8,(ptl),(WWW0))
      if(ptl.gt.0.d0)call mfill(kk+9,(log10(ptl)),(WWW0))
      call mfill(kk+10,(etal),(WWW0))
      call mfill(kk+11,(ptlb),(WWW0))
      call mfill(kk+12,(ptlb),(WWW0))
      if(ptlb.gt.0.d0)call mfill(kk+13,(log10(ptlb)),(WWW0))
      call mfill(kk+14,(etalb),(WWW0))
c
      call mfill(kk+15,(detallb),(WWW0))
      call mfill(kk+16,(azi),(WWW0))
      if(azinorm.gt.0.d0)
     #  call mfill(kk+17,(log10(azinorm)),(WWW0))
      call mfill(kk+18,(xmll),(WWW0))
      call mfill(kk+19,(ptpair),(WWW0))
      if(ptpair.gt.0)call mfill(kk+20,(log10(ptpair)),(WWW0))
      call mfill(kk+21,(0d0),(WWW0))
c
      kk=50
      if(abs(etav).lt.ycut)then
        call mfill(kk+1,(ptv),(WWW0))
        call mfill(kk+2,(ptv),(WWW0))
        if(ptv.gt.0.d0)call mfill(kk+3,(log10(ptv)),(WWW0))
      endif
      if(ptv.gt.20.d0)then
        call mfill(kk+4,(yv),(WWW0))
        call mfill(kk+5,(etav),(WWW0))
      endif
      if(abs(etav).lt.ycut.and.ptv.gt.20.d0)then
         call mfill(kk+6,(xmv),(WWW0))
         call mfill(kk+21,(0d0),(WWW0))
      endif
c
      if(abs(etal).lt.ycut)then
        call mfill(kk+7,(ptl),(WWW0))
        call mfill(kk+8,(ptl),(WWW0))
        if(ptl.gt.0.d0)call mfill(kk+9,(log10(ptl)),(WWW0))
      endif
      if(ptl.gt.20.d0)call mfill(kk+10,(etal),(WWW0))
      if(abs(etalb).lt.ycut)then
        call mfill(kk+11,(ptlb),(WWW0))
        call mfill(kk+12,(ptlb),(WWW0))
        if(ptlb.gt.0.d0)call mfill(kk+13,(log10(ptlb)),(WWW0))
      endif
      if(ptlb.gt.20.d0)call mfill(kk+14,(etalb),(WWW0))
c
      if( abs(etal).lt.ycut.and.abs(etalb).lt.ycut .and.
     #    ptl.gt.20.d0.and.ptlb.gt.20.d0)then
        call mfill(kk+15,(detallb),(WWW0))
        call mfill(kk+16,(azi),(WWW0))
        if(azinorm.gt.0.d0)
     #    call mfill(kk+17,(log10(azinorm)),(WWW0))
        call mfill(kk+18,(xmll),(WWW0))
        call mfill(kk+19,(ptpair),(WWW0))
        if(ptpair.gt.0) 
     #    call mfill(kk+20,(log10(ptpair)),(WWW0))
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
c$$$         CALL HWAEND
         STOP
      ELSEIF (ICODE.LT.400) THEN
         WRITE (6,50)
   50    FORMAT(' EVENT KILLED: DUMP FOLLOWS.  RUN ENDS GRACEFULLY')
         IERROR=ICODE
c$$$         CALL HWUEPR
c$$$         CALL HWUBPR
c$$$         CALL HWEFIN
c$$$         CALL HWAEND
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
