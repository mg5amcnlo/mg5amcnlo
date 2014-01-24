c
c Example analysis for "p p > e+ ve [QCD]" process.
c Example analysis for "p p > e- ve~ [QCD]" process.
c Example analysis for "p p > mu+ vm [QCD]" process.
c Example analysis for "p p > mu- vm~ [QCD]" process.
c Example analysis for "p p > ta+ vt [QCD]" process.
c Example analysis for "p p > ta- vt~ [QCD]" process.
c
C----------------------------------------------------------------------
      SUBROUTINE RCLOS()
C     DUMMY IF HBOOK IS USED
C----------------------------------------------------------------------
      END


C----------------------------------------------------------------------
      SUBROUTINE PYABEG(nnn,wwwi)
C     USER'S ROUTINE FOR INITIALIZATION
C----------------------------------------------------------------------
      INCLUDE 'HEPMC.INC'
      include 'reweight0.inc'
      integer j,kk,l,i,nnn
      character*5 cc(2)
      data cc/'     ','     '/
      integer nwgt,max_weight,nwgt_analysis
      common/cnwgt/nwgt
      common/c_analysis/nwgt_analysis
      parameter (max_weight=maxscales*maxscales+maxpdfs+1)
      character*15 weights_info(max_weight),wwwi(max_weight)
      common/cwgtsinfo/weights_info
c
      call inihist
      weights_info(1)="central value  "
      do i=1,nnn+1
         weights_info(i+1)=wwwi(i)
      enddo
      nwgt=nnn+1
      nwgt_analysis=nwgt
      do i=1,1
      do kk=1,nwgt_analysis
        l=(kk-1)*16+(i-1)*8
        call mbook(l+1,'total rate '//weights_info(kk)//cc(i),
     &       1.0d0,0.5d0,5.5d0)
        call mbook(l+2,'lep rapidity '//weights_info(kk)//cc(i),
     &       0.5d0,-5d0,5d0)
        call mbook(l+3,'lep pt '//weights_info(kk)//cc(i),
     &       10d0,0d0,200d0)
        call mbook(l+4,'et miss '//weights_info(kk)//cc(i),
     &       10d0,0d0,200d0)
        call mbook(l+5,'trans. mass '//weights_info(kk)//cc(i),
     &       5d0,0d0,200d0)
        call mbook(l+6,'w rapidity '//weights_info(kk)//cc(i),
     &       0.5d0,-5d0,5d0)
        call mbook(l+7,'w pt '//weights_info(kk)//cc(i),
     &       10d00,0d0,200d0)
        call mbook(l+8,'cphi[l,vl] '//weights_info(kk)//cc(i),
     &       0.05d0,-1d0,1d0)
      enddo
      enddo
 999  END
C----------------------------------------------------------------------
      SUBROUTINE PYAEND(IEVTTOT)
C     USER'S ROUTINE FOR TERMINAL CALCULATIONS, HISTOGRAM OUTPUT, ETC
C----------------------------------------------------------------------
      INCLUDE 'HEPMC.INC'
      REAL*8 XNORM
      INTEGER I,J,KK,IEVTTOT,l,nwgt_analysis
      integer NPL
      parameter(NPL=15000)
      common/c_analysis/nwgt_analysis
      OPEN(UNIT=99,FILE='PYTLL.TOP',STATUS='UNKNOWN')
C XNORM IS SUCH THAT THE CROSS SECTION PER BIN IS IN PB, SINCE THE HERWIG 
C WEIGHT IS IN NB, AND CORRESPONDS TO THE AVERAGE CROSS SECTION
      XNORM=IEVTTOT/DFLOAT(NEVHEP)
      DO I=1,NPL              
 	CALL MFINAL3(I)             
        CALL MCOPY(I,I+NPL)
        CALL MOPERA(I+NPL,'F',I+NPL,I+NPL,(XNORM),0.D0)
 	CALL MFINAL3(I+NPL)             
      ENDDO                          
C
      do i=1,1
      do kk=1,nwgt_analysis
         l=(kk-1)*16+(i-1)*8
         call multitop(NPL+l+1,NPL-1,3,2,'total rate   ',' ','LIN')
         call multitop(NPL+l+2,NPL-1,3,2,'lep rapidity ',' ','LIN')
         call multitop(NPL+l+3,NPL-1,3,2,'lep pt       ',' ','LOG')
         call multitop(NPL+l+4,NPL-1,3,2,'et miss      ',' ','LOG')
         call multitop(NPL+l+5,NPL-1,3,2,'trans. mass  ',' ','LOG')
         call multitop(NPL+l+6,NPL-1,3,2,'w rapidity   ',' ','LIN')
         call multitop(NPL+l+7,NPL-1,3,2,'w pt         ',' ','LOG')
         call multitop(NPL+l+8,NPL-1,3,2,'cphi[l,vl]   ',' ','LOG')
      enddo
      enddo
      CLOSE(99)
      END

C----------------------------------------------------------------------
      SUBROUTINE PYANAL(nnn,xww)
C     USER'S ROUTINE TO ANALYSE DATA FROM EVENT
C----------------------------------------------------------------------
      INCLUDE 'HEPMC.INC'
      include 'reweight0.inc'
      DOUBLE PRECISION HWVDOT,PSUM(4),PPV(5),PTW,YW,YE,PPL(5),PPLB(5),
     & PTE,PLL,PTLB,PLLB,var,mtr,etmiss,cphi
      INTEGER ICHSUM,ICHINI,IHEP,IV,IFV,IST,ID,IJ,ID1,JPR,IDENT,
     #  ILL,ILLB,IHRD,NL,NN
      LOGICAL DIDSOF,FOUNDL,FOUNDN,ISL,ISN
      REAL*8 PI,getrapidity
      PARAMETER (PI=3.14159265358979312D0)
      REAL*8 WWW0,TINY,SIGNL,SIGNN
      INTEGER KK,I,L,IL,IN
      DATA TINY/.1D-5/
      integer nwgt_analysis,max_weight
      common/c_analysis/nwgt_analysis
      parameter (max_weight=maxscales*maxscales+maxpdfs+1)
      double precision ww(max_weight),www(max_weight),xww(max_weight)
      common/cww/ww
c
      ww(1)=xww(2)
      if(nnn.eq.0)ww(1)=1d0
      do i=2,nnn+1
         ww(i)=xww(i)
      enddo
c
      IF (WW(1).EQ.0D0) THEN
         WRITE(*,*)'WW(1) = 0. Stopping'
         STOP
      ENDIF
C CHOOSE IDENT = 11 FOR E - NU_E
C        IDENT = 13 FOR MU - NU_MU
C        IDENT = 15 FOR TAU - NU_TAU
      IDENT=11
C INCOMING PARTONS MAY TRAVEL IN THE SAME DIRECTION: IT'S A POWER-SUPPRESSED
C EFFECT, SO THROW THE EVENT AWAY
      IF(SIGN(1.D0,PHEP(3,1)).EQ.SIGN(1.D0,PHEP(3,2)))THEN
        CALL HWWARN('PYANAL',111)
        GOTO 999
      ENDIF
      DO I=1,nwgt_analysis
         WWW(I)=EVWGT*ww(i)/ww(1)
      ENDDO
      ICHSUM=0
      DIDSOF=.FALSE.
      FOUNDL=.FALSE.
      FOUNDN=.FALSE.
      NL=0
      NN=0
      DO 100 IHEP=1,NHEP
        IST=ISTHEP(IHEP)      
        ID1=IDHEP(IHEP)
        ISL=ABS(ID1).EQ.IDENT
        ISN=ABS(ID1).EQ.IDENT+1
        IF(NL.EQ.0.AND.ISL)THEN
            NL=NL+1
            IL=IHEP
            FOUNDL=.TRUE.
            SIGNL=SIGN(1D0,DBLE(ID1))
         ENDIF
         IF(NN.EQ.0.AND.ISN)THEN
            NN=NN+1
            IN=IHEP
            FOUNDN=.TRUE.
            SIGNN=SIGN(1D0,DBLE(ID1))
         ENDIF
  100 CONTINUE
      IF(.NOT.FOUNDL.OR..NOT.FOUNDN)THEN
         WRITE(*,*)'NO LEPTONS FOUND.'
         WRITE(*,*)'CURRENTLY THIS ANALYSIS LOOKS FOR'
         IF(IDENT.EQ.11)WRITE(*,*)'E - NU_E'
         IF(IDENT.EQ.13)WRITE(*,*)'MU - NU_MU'
         IF(IDENT.EQ.15)WRITE(*,*)'TAU - NU_TAU'
         WRITE(*,*)'IF THIS IS NOT MEANT,'
         WRITE(*,*)'PLEASE CHANGE THE VALUE OF IDENT IN THIS FILE.'
         STOP
      ENDIF
      IF(SIGNN.EQ.SIGNL)THEN
         WRITE(*,*)'TWO SAME SIGN LEPTONS!'
         WRITE(*,*)IL,IN,SIGNL,SIGNN
         STOP
      ENDIF
      DO IJ=1,5
        PPL(IJ)=PHEP(IJ,IN)
        PPLB(IJ)=PHEP(IJ,IL)
        PPV(IJ)=PPL(IJ)+PPLB(IJ)
      ENDDO
      ye     = getrapidity(pplb(4), pplb(3))
      yw     = getrapidity(ppv(4), ppv(3))
      pte    = dsqrt(pplb(1)**2 + pplb(2)**2)
      ptw    = dsqrt(ppv(1)**2+ppv(2)**2)
      etmiss = dsqrt(ppl(1)**2 + ppl(2)**2)
      mtr    = dsqrt(2d0*pte*etmiss-2d0*ppl(1)*pplb(1)-2d0*ppl(2)*pplb(2))
      cphi   = (ppl(1)*pplb(1)+ppl(2)*pplb(2))/pte/etmiss
      var    = 1.d0
      do i=1,1
         do kk=1,nwgt_analysis
            l=(kk-1)*16+(i-1)*8
            call mfill(l+1,var,www(kk))
            call mfill(l+2,ye,www(kk))
            call mfill(l+3,pte,www(kk))
            call mfill(l+4,etmiss,www(kk))
            call mfill(l+5,mtr,www(kk))
            call mfill(l+6,yw,www(kk))
            call mfill(l+7,ptw,www(kk))
            call mfill(l+8,cphi,www(kk))
         enddo
      enddo
C
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
