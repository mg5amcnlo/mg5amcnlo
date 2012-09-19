C----------------------------------------------------------------------
      SUBROUTINE RCLOS()
C     DUMMY IF HBOOK IS USED
C----------------------------------------------------------------------
      END


C----------------------------------------------------------------------
      SUBROUTINE PYABEG
C     USER'S ROUTINE FOR INITIALIZATION
C----------------------------------------------------------------------
      REAL*4 pi
      parameter (pi=3.14160E0)
      integer k
c
      call inihist
      k=0
      call mbook(k+ 1,'t pt         ',4.e0,0.e0,400.e0)
      call mbook(k+ 2,'t log[pt]    ',0.05e0,0.1e0,5.e0)
      call mbook(k+ 3,'t eta        ',0.2e0,-9.e0,9.e0)
      call mbook(k+ 4,'bbar pt      ',4.e0,0.e0,400.e0)
      call mbook(k+ 5,'b log[pt]    ',0.05e0,0.1e0,5.e0)
      call mbook(k+ 6,'bbar eta     ',0.2e0,-9.e0,9.e0)
      call mbook(k+ 7,'B pt         ',4.e0,0.e0,400.e0)
      call mbook(k+ 8,'B log[pt]    ',0.05e0,0.1e0,5.e0)
      call mbook(k+ 9,'B eta        ',0.2e0,-9.e0,9.e0)
      call mbook(k+10,'j1 pt        ',4.e0,0.e0,400.e0)
      call mbook(k+11,'j1 log[pt]   ',0.05e0,0.1e0,5.e0)
      call mbook(k+12,'j1 eta       ',0.2e0,-9.e0,9.e0)
      call mbook(k+13,'j2 pt        ',4.e0,0.e0,400.e0)
      call mbook(k+14,'j2 log[pt]   ',0.05e0,0.1e0,5.e0)
      call mbook(k+15,'j2 eta       ',0.2e0,-9.e0,9.e0)
      call mbook(k+16,'t-j1 pt      ',4.e0,0.e0,400.e0)
      call mbook(k+17,'t-j1 log[pt] ',0.05e0,0.1e0,5.e0)
      call mbook(k+18,'t-j1 azimt   ',pi/20.e0,0.e0,pi)
      call mbook(k+19,'j1-j2 pt     ',4.e0,0.e0,400.e0)
      call mbook(k+20,'j1-j2 log[pt]',0.05e0,0.1e0,5.e0)
      call mbook(k+21,'j1-j2 azimt  ',pi/20.e0,0.e0,pi)
      call mbook(k+22,'B-j1 pt      ',4.e0,0.e0,400.e0)
      call mbook(k+23,'B-j1 log[pt] ',0.05e0,0.1e0,5.e0)
      call mbook(k+24,'B-j1 azimt   ',pi/20.e0,0.e0,pi)
      call mbook(k+25,'syst1 pt     ',4.e0,0.e0,400.e0)
      call mbook(k+26,'syst1 log[pt]',0.05e0,0.1e0,5.e0)
      call mbook(k+27,'syst1 eta    ',0.2e0,-9.e0,9.e0)
      call mbook(k+28,'syst2 pt     ',4.e0,0.e0,400.e0)
      call mbook(k+29,'syst2 log[pt]',0.05e0,0.1e0,5.e0)
      call mbook(k+30,'syst2 eta    ',0.2e0,-9.e0,9.e0)

      END


C----------------------------------------------------------------------
      SUBROUTINE PYAEND(SUMW)
C     USER'S ROUTINE FOR TERMINAL CALCULATIONS, HISTOGRAM OUTPUT, ETC
C----------------------------------------------------------------------
      REAL*8 XNORM
      INTEGER I,K
      OPEN(UNIT=99,NAME='PYTST.TOP',STATUS='UNKNOWN')
      IF(SUMW.NE.0)XNORM=1.D0/SUMW
      IF(SUMW.EQ.0)XNORM=1.D0
C SUMW IS N(+) - N (-) = NUMBER OF POSITIVE WEIGHTED EVENTS - NUMBER
C OF NEGATIVE WEIGHTED EVENTS: SSUMW IS COMPUTED EVENT BY EVENT; WHEN
C WWW0 = +-XSECTION, THEN THE SUM OF WWW0*XNORM=XSECTION
C MODIFY EVENTUALLY
      DO I=1,100              
 	CALL MFINAL3(I)             
        CALL MCOPY(I,I+100)
        CALL MOPERA(I+100,'F',I+100,I+100,SNGL(XNORM),0.E0)
 	CALL MFINAL3(I+100)             
      ENDDO                          
C
      k=0
      call multitop(100+k+ 1,99,3,2,'t pt         ',' ','LOG')
      call multitop(100+k+ 2,99,3,2,'t log[pt]    ',' ','LOG')
      call multitop(100+k+ 3,99,3,2,'t eta        ',' ','LOG')
      call multitop(100+k+ 4,99,3,2,'bbar pt      ',' ','LOG')
      call multitop(100+k+ 5,99,3,2,'b log[pt]    ',' ','LOG')
      call multitop(100+k+ 6,99,3,2,'bbar eta     ',' ','LOG')
      call multitop(100+k+ 7,99,3,2,'B pt         ',' ','LOG')
      call multitop(100+k+ 8,99,3,2,'B log[pt]    ',' ','LOG')
      call multitop(100+k+ 9,99,3,2,'B eta        ',' ','LOG')
      call multitop(100+k+10,99,3,2,'j1 pt        ',' ','LOG')
      call multitop(100+k+11,99,3,2,'j1 log[pt]   ',' ','LOG')
      call multitop(100+k+12,99,3,2,'j1 eta       ',' ','LOG')
      call multitop(100+k+13,99,3,2,'j2 pt        ',' ','LOG')
      call multitop(100+k+14,99,3,2,'j2 log[pt]   ',' ','LOG')
      call multitop(100+k+15,99,3,2,'j2 eta       ',' ','LOG')
      call multitop(100+k+16,99,3,2,'t-j1 pt      ',' ','LOG')
      call multitop(100+k+17,99,3,2,'t-j1 log[pt] ',' ','LOG')
      call multitop(100+k+18,99,3,2,'t-j1 azimt   ',' ','LOG')
      call multitop(100+k+19,99,3,2,'j1-j2 pt     ',' ','LOG')
      call multitop(100+k+20,99,3,2,'j1-j2 log[pt]',' ','LOG')
      call multitop(100+k+21,99,3,2,'j1-j2 azimt  ',' ','LOG')
      call multitop(100+k+22,99,3,2,'B-j1 pt      ',' ','LOG')
      call multitop(100+k+23,99,3,2,'B-j1 log[pt] ',' ','LOG')
      call multitop(100+k+24,99,3,2,'B-j1 azimt   ',' ','LOG')
      call multitop(100+k+25,99,3,2,'syst1 pt     ',' ','LOG')
      call multitop(100+k+26,99,3,2,'syst1 log[pt]',' ','LOG')
      call multitop(100+k+27,99,3,2,'syst1 eta    ',' ','LOG')
      call multitop(100+k+28,99,3,2,'syst2 pt     ',' ','LOG')
      call multitop(100+k+29,99,3,2,'syst2 log[pt]',' ','LOG')
      call multitop(100+k+30,99,3,2,'syst2 eta    ',' ','LOG')
c
      CLOSE(99)
      END


C----------------------------------------------------------------------
      SUBROUTINE PYANAL
C     USER'S ROUTINE TO ANALYSE DATA FROM EVENT
C----------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
      INTEGER PYCHGE
      EXTERNAL PYDATA
      COMMON/PYJETS/N,NPAD,K(4000,5),P(4000,5),V(4000,5)
      COMMON/PYDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
      COMMON/PYDAT2/KCHG(500,4),PMAS(500,4),PARF(2000),VCKM(4,4)
      COMMON/PYDAT3/MDCY(500,3),MDME(8000,2),BRAT(8000),KFDP(8000,5)
      COMMON/PYSUBS/MSEL,MSELPD,MSUB(500),KFIN(2,-40:40),CKIN(200)
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200)
      DOUBLE PRECISION HWVDOT,PSUM(4)
      INTEGER ICHSUM,ICHINI,IHEP
      LOGICAL DIDSOF,sicuts,flcuts
      INTEGER NMAX,JET,IQT,IQB,IB,IST,ID,ID1,NN,I,J,K,ID2,IHADR,
     # IT1,IB1,NPRIMARY,IJ,kk,njet,IT,IBQ,I1,IBH,IBH1,IBH2,IBF,
     # IMO,I2,IBMATCH,IORJET,ITMP,JETMATCH,JB(30),JB1(30),
     # JPOS(10),KMO(2)
      DOUBLE PRECISION PP,PJET,Y,PB,pt1,eta1,getpseudorap,pt2,
     # eta2,palg,rfj,SYCUT,PTU,PTCALC,PTL,ptBh,etaBh,ptj1,etaj1,
     # ptpair,dphipair,getdelphi,ptj2,etaj2,ptS1,etaS1,ptS2,
     # etaS2,XPTQ(5),XPBQ(5),PSYST1(4),PSYST2(4)
      LOGICAL FOUNDMATCH
      PARAMETER (NMAX=2000)
      DIMENSION JET(NMAX),Y(NMAX),PP(4,NMAX),PJET(4,NMAX),
     # PB(4,NMAX),IORJET(NMAX)
      REAL*8 PI
      PARAMETER (PI=3.14159265358979312D0)
      REAL*8 WWW0
      SAVE INOTOP
c
C INCOMING PARTONS MAY TRAVEL IN THE SAME DIRECTION: IT'S A POWER-SUPPRESSED
C EFFECT, SO THROW THE EVENT AWAY
      IF(SIGN(1.D0,P(3,3)).EQ.SIGN(1.D0,PHEP(4,3)))THEN
        WRITE(*,*)'WARNING 111 IN PYANAL'
        GOTO 999
      ENDIF
      WWW0=PARI(7)/ABS(PARI(7))
C EVENTUALLY NORMALITE TO THE TOTAL CROSS SECTION!
      DO I=1,4
         P1(I)=P(1,I)
         P2(I)=P(2,I)
      ENDDO
      CALL VVSUM(4,P1,P2,PSUM)
      CALL VSCA(4,-1D0,PSUM,PSUM)
      ICHSUM=0
      KF1=K(1,2)
      KF2=K(2,2)
      ICHINI=PYCHGE(KF1)+PYCHGE(KF2)
      IQT=0
      IQB=0
      IB=0
      NN=0
      NPRIMARY=0
      DO 100 IHEP=1,N
        DO J=1,4
          PIHEP(J)=P(IHEP,J)
        ENDDO
        IST=K(IHEP,1)
        ID1=K(IHEP,2)
        IORI=K(IHEP,3)
        IF (IST.LE.10) THEN
          CALL VVSUM(4,PIHEP,PSUM,PSUM)
          KFIHEP=K(IHEP,2)
          ICHSUM=ICHSUM+PYCHGE(KFIHEP)
        ENDIF
C---FIND FINAL STATE HADRONS AND PHOTONS
        IF(IST.LE.10.AND.(ABS(ID1).GT.100.OR.ID1.EQ.22))THEN
          NN=NN+1
          IF (NN.GT.NMAX) STOP 'Too many particles!'
          DO I=1,4
             PP(I,NN)=P(IHEP,I)
          ENDDO
          ID2=IHADR(ID1)
          IF(ID2.EQ.5)THEN
C FOUND A B-FLAVOURED HADRON
            IB=IB+1
            IF(IB.GT.30)THEN
              WRITE(*,*)'IB=',IB
              WRITE(*,*)'WARNING 512 IN PYANAL'
              STOP
            ENDIF
            JB(IB)=IHEP
            JB1(IB)=NN
            DO I=1,4
               PB(I,IB)=PHEP(I,IHEP)
            ENDDO
          ENDIF
        ENDIF
        IF(IST.GT.10.AND.ID1.EQ.6)THEN
C FOUND A TOP; KEEP ONLY THE FIRST ON RECORD
          IQT=IQT+1
          IF(IQT.EQ.1)IT1=IHEP
        ELSEIF(ID1.EQ.-5)THEN
C FOUND AN ANTIBOTTOM; KEEP ONLY THE FIRST ON RECORD
          IQB=IQB+1
          IF(IQB.EQ.1)IB1=IHEP
c$$$        ELSEIF(IST.GE.123.AND.IST.LE.125.AND.ABS(ID1).NE.6)THEN
c$$$          NPRIMARY=NPRIMARY+1
c$$$          IF(NPRIMARY.GT.10)THEN
c$$$            WRITE(*,*)'NPRIMARY=',NPRIMARY
c$$$            WRITE(*,*)'WARNING 512 IN PYANAL'
c$$$            STOP
c$$$          ENDIF
c$$$          JPOS(NPRIMARY)=IHEP
C
C
C FIND THE WAY TO IMPLEMENT THIS WITH PYTHIA
C FIND THE WAY TO IMPLEMENT THIS WITH PYTHIA
C FIND THE WAY TO IMPLEMENT THIS WITH PYTHIA
C FIND THE WAY TO IMPLEMENT THIS WITH PYTHIA
C
C
        ENDIF
  100 CONTINUE
      IF(IQT.EQ.0.or.IQB.EQ.0)THEN
      INOTOP=INOTOP+1
      WRITE(*,*)'WARNING 512 IN PYANAL, NO TOP OR NO ANTIBOTTOM',INOTOP
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
C FILL THE FOUR-MOMENTA
      DO IJ=1,5
        XPTQ(IJ)=P(IT1,IJ)
        XPBQ(IJ)=P(IB1,IJ)
      ENDDO
c
      pt1=sqrt(xptq(1)**2+xptq(2)**2)
      eta1=getpseudorap(xptq(4),xptq(1),xptq(2),xptq(3))
      pt2=sqrt(xpbq(1)**2+xpbq(2)**2)
      eta2=getpseudorap(xpbq(4),xpbq(1),xpbq(2),xpbq(3))
C
C WITHOUT CUTS
C
      kk=0
      call mfill(kk+1,sngl(pt1),sngl(WWW0))
      if(pt1.gt.0.d0)call mfill(kk+2,sngl(log10(pt1)),sngl(WWW0))
      call mfill(kk+3,sngl(eta1),sngl(WWW0))
      call mfill(kk+4,sngl(pt2),sngl(WWW0))
      if(pt2.gt.0.d0)call mfill(kk+5,sngl(log10(pt2)),sngl(WWW0))
      call mfill(kk+6,sngl(eta2),sngl(WWW0))
C
      palg=1.d0
      rfj=0.5d0
      SYCUT=10.d0
      call fastjetppgenkt(pp,nn,rfj,sycut,palg,pjet,njet,jet)
c
      IT=1
C FIND THE B HADRONS EMERGING FROM TOP DECAY; LOOP OVER ALL TOPS
      DO I=1,IT
c IBQ is the position in the event record of the b quark before hadronization
c$$$        IBQ=0
c$$$        IF(JDAHEP(2,IT1)-JDAHEP(1,IT1).EQ.1)THEN
c$$$          DO I1=JDAHEP(1,JDAHEP(1,JDAHEP(2,IT1))),
c$$$     #          JDAHEP(2,JDAHEP(1,JDAHEP(2,IT1)))
c$$$            IF(ISTHEP(I1).EQ.2.AND.ABS(IDHEP(I1)).EQ.5)IBQ=I1
c$$$          ENDDO
c$$$        ELSEIF(JDAHEP(2,IT1)-JDAHEP(1,IT1).EQ.2)THEN
c$$$          DO I1=JDAHEP(1,JDAHEP(1,JDAHEP(2,IT1)-1)),
c$$$     #          JDAHEP(2,JDAHEP(1,JDAHEP(2,IT1)-1))
c$$$            IF(ISTHEP(I1).EQ.2.AND.ABS(IDHEP(I1)).EQ.5)IBQ=I1
c$$$          ENDDO
c$$$        ELSE
c$$$           WRITE(*,*)'WARNING 502 IN PYANAL'
c$$$        ENDIF
c$$$        IBH=JDAHEP(1,JDAHEP(1,JDAHEP(1,IBQ)))
c$$$        IBH1=JDAHEP(2,JDAHEP(1,JDAHEP(1,IBQ)))
c$$$        ID1=IHADR(IDHEP(IBH1))
c$$$        IF(ID1.EQ.5)IBH=IBH1
c$$$        IF(ISTHEP(IBH).NE.1.AND.ISTHEP(IBH).NE.196.AND.
c$$$     #     ISTHEP(IBH).NE.197.AND.ISTHEP(IBH).NE.199)THEN
c$$$           WRITE(*,*)'WARNING 503 IN PYANAL'
c$$$        ENDIF
c$$$c IBH is the position in the event record of the b hadron emerging from top
c$$$c decay. If such hadron is unstable (status 196 or 197), find the stable
c$$$c b hadron emerging from its decay. Status 199 seems to correspond to an
c$$$c hadronically-decaying hadrons (such as eta_b).
c$$$        IF(ISTHEP(IBH).EQ.196)THEN
c$$$          IBH1=JDAHEP(1,IBH)
c$$$        ELSEIF(ISTHEP(IBH).EQ.197)THEN
c$$$          IBH1=JDAHEP(1,IBH)
c$$$          ID1=IHADR(IDHEP(IBH1))
c$$$          IF(ID1.NE.5)IBH1=JDAHEP(2,IBH)
c$$$c The following fails if there are more than two unstable b-hadrons that decay
c$$$          IF(ISTHEP(IBH1).EQ.198)THEN
c$$$            IBH2=JDAHEP(1,IBH1)
c$$$            ID1=IHADR(IDHEP(IBH2))
c$$$            IF(ID1.NE.5)IBH2=JDAHEP(2,IBH1)
c$$$            IBH1=IBH2
c$$$          ENDIF
c
c
c TO BE DONE***********************
c TO BE DONE***********************
c TO BE DONE***********************
c TO BE DONE***********************
        ELSEIF(k(IBH,1).EQ.11)THEN
c Appears to be rare, so throw the event away and continue, in order 
c not to complicate the analysis too much (for finding the bjet)
          WRITE(*,*)'WARNING 114 IN PYANAL'
          goto 999
        ELSE
          IBH1=IBH
        ENDIF
        IF(K(IBH1,1).GT.10)THEN
          WRITE(*,*)'WARNING 504 IN PYANAL'
          GOTO 999
        ENDIF
c IBH1 is the position in the event record of the stable b hadron emerging 
c from top decay. It must be one of the stable b hadrons found at the very
c beginning; set the corresponding JB to zero in such a case, or to -JB1 in
c the case of the first top in the event record, which is assumed to be
c that resulting from the hard interaction ==> positive JB's are thus 
c those associated with b hadron not emerging from top decays
        IBF=0
        DO I1=1,IB
          IF(JB(I1JB).EQ.IBH1)THEN
            IF(I.EQ.1)THEN
              JB(I1)=-JB1(I1)
            ELSE
              JB(I1)=0
            ENDIF
            IF(IBF.EQ.0)THEN
              IBF=1
            ELSE
              WRITE(*,*)'WARNING 504 IN PYANAL'
              GOTO 999
            ENDIF
          ENDIF
c Among the b hadrons not emerging from the top decay, find the one that
c comes from the hard process
          FOUNDMATCH=.FALSE.
          IF(JB(I1).GT.0.AND.(.NOT.FOUNDMATCH))THEN
            IMO=JMOHEP(1,JB(I1))
            I2=0
            DOWHILE(IDHEP(IMO).NE.91)
              IMO=JMOHEP(1,IMO)
              I2=I2+1
              IF(I2.GT.30)THEN
                CALL HWUEPR
                WRITE(*,*)'IHEP,I2=',IHEP,I2
                CALL HWWARN('HWANAL',506)
              ENDIF
            ENDDO
C Now IMO is the first cluster found going backwards from the selected 
c b hadron. Keep on going backwards scanning all clusters
            I2=0 
            DOWHILE(IDHEP(IMO).EQ.91)
              IMO=JMOHEP(1,IMO)
              I2=I2+1
              IF(I2.GT.30)THEN
                CALL HWUEPR
                WRITE(*,*)'IHEP,I2=',IHEP,I2
                CALL HWWARN('HWANAL',507)
              ENDIF
            ENDDO
            KMO(1)=JMOHEP(1,JDAHEP(1,IMO))
            KMO(2)=JMOHEP(2,JDAHEP(1,IMO))
            IF( (KMO(1).NE.IMO.AND.KMO(2).NE.IMO) .OR.
     #          ABS(IDHEP(KMO(1))).NE.5 .AND.
     #          ABS(IDHEP(KMO(2))).NE.5 )THEN
              CALL HWUEPR
              WRITE(*,*)I1,JB(I1),IMO,KMO(1),KMO(2)
              CALL HWWARN('HWANAL',508)
            ENDIF
            K=0
 123        CONTINUE
            K=K+1
            IF(K.LE.0.OR.K.GE.3)THEN
              CALL HWUEPR
              WRITE(*,*)'K=',K
              CALL HWWARN('HWANAL',513)
            ENDIF
            IF(ABS(IDHEP(KMO(K))).NE.5)GOTO 124
            IMO=KMO(K)
            I2=0 
            DOWHILE(ABS(IDHEP(IMO)).EQ.5 .or.
     &           (IDHEP(IMO).EQ.94.and.abs(IDHEP(JMOHEP(1,IMO))).eq.5))
              IMO=JMOHEP(1,IMO)
              IF(I2.GT.30)THEN
                CALL HWUEPR
                WRITE(*,*)'IHEP,I2=',IHEP,I2
                CALL HWWARN('HWANAL',509)
              ENDIF
            ENDDO
            IF(ISTHEP(IMO).EQ.120)THEN
              FOUNDMATCH=.TRUE.
              IBMATCH=I1
            ENDIF
 124        IF(K.EQ.1)GOTO 123
          ENDIF
        ENDDO
        IF(IBF.EQ.0)THEN
          CALL HWUEPR
          CALL HWWARN('HWANAL',510)
        ENDIF
      ENDDO
c
      DO I=1,NJET
        IORJET(I)=I
      ENDDO
      DO I=NJET,2,-1
        DO J=NJET,NJET+2-I,-1
          PTU=PTCALC(PJET(1,IORJET(J-1)))
          PTL=PTCALC(PJET(1,IORJET(J)))
          IF(PTL.GT.PTU)THEN
            ITMP=IORJET(J-1)
            IORJET(J-1)=IORJET(J)
            IORJET(J)=ITMP
          ENDIF
        ENDDO
      ENDDO
C Now IORJET(k) is the k^th hardest jet
c
      DO I=1,4
        PSYST1(I)=XPTQ(I)
        IF(NJET.GE.1)PSYST1(I)=PSYST1(I)+PJET(I,IORJET(1))
        IF(NJET.GE.2)PSYST1(I)=PSYST1(I)+PJET(I,IORJET(2))
      ENDDO
C Find the jet that contains the daughter of primary b
      IF(FOUNDMATCH)THEN
        IF(JB1(IBMATCH).LE.0.OR.JB1(IBMATCH).GT.NMAX)THEN
          CALL HWUEPR
          WRITE(*,*)IBMATCH,JB1(IBMATCH)
          CALL HWWARN('HWANAL',514)
        ENDIF
        JETMATCH=JET(JB1(IBMATCH))
        IF(JETMATCH.GT.NJET)THEN
          CALL HWUEPR
          WRITE(*,*)IBMATCH,JB1(IBMATCH),JETMATCH
          CALL HWWARN('HWANAL',515)
        ENDIF
        IF(JETMATCH.NE.0)THEN
          DO I=1,4
            PSYST2(I)=XPTQ(I)
            IF(NJET.GE.1)PSYST2(I)=PSYST2(I)+PJET(I,IORJET(1))
            IF(NJET.GE.2)THEN
              IF(JETMATCH.NE.IORJET(1))THEN
                PSYST2(I)=PSYST2(I)+PJET(I,JETMATCH)
              ELSE
                PSYST2(I)=PSYST2(I)+PJET(I,IORJET(2))
              ENDIF
            ENDIF
          ENDDO
        ELSE
          DO I=1,4
            PSYST2(I)=XPTQ(I)+PP(I,JB1(IBMATCH))
            IF(NJET.GE.1)PSYST2(I)=PSYST2(I)+PJET(I,IORJET(1))
          ENDDO
        ENDIF
      ENDIF
c
      ptBh=sqrt(PP(1,JB1(IBMATCH))**2+PP(2,JB1(IBMATCH))**2)
      etaBh=getpseudorap(PP(4,JB1(IBMATCH)),PP(1,JB1(IBMATCH)),
     #                   PP(2,JB1(IBMATCH)),PP(3,JB1(IBMATCH)))
      call mfill(kk+7,sngl(ptBh),sngl(WWW0))
      if(ptBh.gt.0.d0)call mfill(kk+8,sngl(log10(ptBh)),sngl(WWW0))
      call mfill(kk+9,sngl(etaBh),sngl(WWW0))
c
      if(njet.ge.1)then
        ptj1=sqrt(PJET(1,IORJET(1))**2+PJET(2,IORJET(1))**2)
        etaj1=getpseudorap(PJET(4,IORJET(1)),PJET(1,IORJET(1)),
     #                     PJET(2,IORJET(1)),PJET(3,IORJET(1)))
        call mfill(kk+10,sngl(ptj1),sngl(WWW0))
        if(ptj1.gt.0.d0)call mfill(kk+11,sngl(log10(ptj1)),sngl(WWW0))
        call mfill(kk+12,sngl(etaj1),sngl(WWW0))
c
        do i=1,4
          psum(i)=PJET(i,IORJET(1))+XPTQ(I)
        enddo
        ptpair=sqrt(psum(1)**2+psum(2)**2)
        dphipair=getdelphi(PJET(1,IORJET(1)),PJET(2,IORJET(1)),
     #                     XPTQ(1),XPTQ(2))
        call mfill(kk+16,sngl(ptpair),sngl(WWW0))
        if(ptpair.gt.0.d0)
     #    call mfill(kk+17,sngl(log10(ptpair)),sngl(WWW0))
        call mfill(kk+18,sngl(dphipair),sngl(WWW0))
c
        if(jetmatch.ne.iorjet(1))then
          do i=1,4
            psum(i)=PJET(i,IORJET(1))+PP(i,JB1(IBMATCH))
          enddo
          ptpair=sqrt(psum(1)**2+psum(2)**2)
          dphipair=getdelphi(PJET(1,IORJET(1)),PJET(2,IORJET(1)),
     #                       PP(1,JB1(IBMATCH)),PP(2,JB1(IBMATCH)))
          call mfill(kk+22,sngl(ptpair),sngl(WWW0))
          if(ptpair.gt.0.d0)
     #      call mfill(kk+23,sngl(log10(ptpair)),sngl(WWW0))
          call mfill(kk+24,sngl(dphipair),sngl(WWW0))
        endif
      endif
c
      if(njet.ge.2)then
        ptj2=sqrt(PJET(1,IORJET(2))**2+PJET(2,IORJET(2))**2)
        etaj2=getpseudorap(PJET(4,IORJET(2)),PJET(1,IORJET(2)),
     #                     PJET(2,IORJET(2)),PJET(3,IORJET(2)))
        call mfill(kk+13,sngl(ptj2),sngl(WWW0))
        if(ptj2.gt.0.d0)call mfill(kk+14,sngl(log10(ptj2)),sngl(WWW0))
        call mfill(kk+15,sngl(etaj2),sngl(WWW0))
c
        do i=1,4
          psum(i)=PJET(i,IORJET(1))+PJET(i,IORJET(2))
        enddo
        ptpair=sqrt(psum(1)**2+psum(2)**2)
        dphipair=getdelphi(PJET(1,IORJET(1)),PJET(2,IORJET(1)),
     #                     PJET(1,IORJET(2)),PJET(2,IORJET(2)))
        call mfill(kk+19,sngl(ptpair),sngl(WWW0))
        if(ptpair.gt.0.d0)
     #    call mfill(kk+20,sngl(log10(ptpair)),sngl(WWW0))
        call mfill(kk+21,sngl(dphipair),sngl(WWW0))
      endif
c
      ptS1=sqrt(PSYST1(1)**2+PSYST1(2)**2)
      etaS1=getpseudorap(PSYST1(4),PSYST1(1),
     #                   PSYST1(2),PSYST1(3))
      call mfill(kk+25,sngl(ptS1),sngl(WWW0))
      if(ptS1.gt.0.d0)call mfill(kk+26,sngl(log10(ptS1)),sngl(WWW0))
      call mfill(kk+27,sngl(etaS1),sngl(WWW0))
      ptS2=sqrt(PSYST2(1)**2+PSYST2(2)**2)
      etaS2=getpseudorap(PSYST2(4),PSYST2(1),
     #                   PSYST2(2),PSYST2(3))
      call mfill(kk+28,sngl(ptS2),sngl(WWW0))
      if(ptS2.gt.0.d0)call mfill(kk+29,sngl(log10(ptS2)),sngl(WWW0))
      call mfill(kk+30,sngl(etaS2),sngl(WWW0))
c

 999  RETURN
      END


      function getrapidity(en,pl)
      implicit none
      real*8 getrapidity,en,pl,tiny,xplus,xminus,y
      parameter (tiny=1.d-8)
c
      xplus=en+pl
      xminus=en-pl
      if(xplus.gt.tiny.and.xminus.gt.tiny)then
        if( (xplus/xminus).gt.tiny.and.(xminus/xplus).gt.tiny )then
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


      FUNCTION IHADR(ID)
c Returns the PDG code of the heavier quark in the hadron of PDG code ID
      IMPLICIT NONE
      INTEGER IHADR,ID,ID1
C
      IF(ID.NE.0)THEN
        ID1=ABS(ID)
        IF(ID1.GT.10000)ID1=ID1-1000*INT(ID1/1000)
        IHADR=ID1/(10**INT(LOG10(DFLOAT(ID1))))
      ELSE
        IHADR=0
      ENDIF
      RETURN
      END


      FUNCTION PTCALC(P)
      IMPLICIT NONE
      DOUBLE PRECISION PTCALC,P(4),PTSQ
      PTSQ=P(1)**2+P(2)**2
      IF (PTSQ.EQ.0D0) THEN
         PTCALC=0D0
      ELSE
         PTCALC=SQRT(PTSQ)
      ENDIF
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
