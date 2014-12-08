C DRIVER, TO BE MODIFIED BY THE USER (KEEPING THE READ STATEMENTS).
      PROGRAM PYDRIVER
      IMPLICIT NONE
      INTEGER J,IPROC,IPDF,IEVT,ISHW_PY,MAXER
      INTEGER NSTEP,I,JPR0,JPR,NUMDM,II,NBODIES(100),JJ,IMATCH
      INTEGER IMOTH(100),IDAUGHT(100,5),NMOTH,ARRMOTH(100),III,KK
      DOUBLE PRECISION BR(100),BRR(100,100),SUMBR(100,100)
      INTEGER IMIN(100),IMAX(100),MEDEC(100)
      CHARACTER *1 TMPCH
      CHARACTER *5 CIDAUGHT(100,5),CIMOTH(100)
      CHARACTER *8 CBR(100)
      CHARACTER *7 NORM_EVENT
      CHARACTER *25 CIDAUGHTTOT(100)
      CHARACTER *100 TMPSTR
      INTEGER N,NPAD,K,MSTU,MSTJ,KCHG,MDCY,MDME,KFDP,MSEL,MSELPD,PYCOMP,
     &MSUB,KFIN,MSTP,MSTI,MODPDF(2),MAXEV,MAXPR,ID,NGENPD,NGEN
      DOUBLE PRECISION M_Z,M_W,M_T,M_H,M_D,M_U,M_S,M_C,M_B,M_G,M_E,M_MU,M_TAU,
     &G_Z,G_W,G_T,G_H,QCDLAM,PBEAM1,PBEAM2,EMMIN,EMMAX,GAMMAX,XSEC
      DOUBLE PRECISION P,V,PARJ,PMAS,PARF,VCKM,BRAT,CKIN,PARP,PARI,PARU
      COMMON/CMASSES/M_Z,M_W,M_T,M_H,M_D,M_U,M_S,M_C,M_B,M_G,M_E,M_MU,M_TAU
      COMMON/CWIDTHS/G_Z,G_W,G_T,G_H
C QQIN IS THE EVENT FILE
      CHARACTER*50 QQIN,PART1,PART2
      CHARACTER*20 AUTPDF(2)
      COMMON/VVJIN/QQIN
      INTEGER MQQ
      COMMON/cMQQ/MQQ
      REAL*8 TMPLAM,GAMT0
      CHARACTER * 70 LHAPDF
      LOGICAL LHACRTL,OLDFORM,PI_STABLE,WP_STABLE,WM_STABLE,Z_STABLE,
     &TAUP_STABLE,TAUM_STABLE,MUP_STABLE,MUM_STABLE,H_STABLE
      PARAMETER (LHACRTL=.TRUE.)
      EXTERNAL PYDATA
      COMMON/CIEVT/IEVT
      COMMON/CQCD/QCDLAM
      COMMON/CGAMMAX/GAMMAX
      COMMON/PYJETS/N,NPAD,K(4000,5),P(4000,5),V(4000,5)
      COMMON/PYDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
      COMMON/PYDAT2/KCHG(500,4),PMAS(500,4),PARF(2000),VCKM(4,4)
      COMMON/PYDAT3/MDCY(500,3),MDME(8000,2),BRAT(8000),KFDP(8000,5)
      COMMON/PYSUBS/MSEL,MSELPD,MSUB(500),KFIN(2,-40:40),CKIN(200)
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200)
      COMMON/PYINT5/NGENPD,NGEN(0:500,3),XSEC(0:500,3)
      COMMON/CISHW/ISHW_PY
      COMMON/CIPROC/JPR
      COMMON/CPDFINFO/AUTPDF,MODPDF,IPDF
      INTEGER SUMER,SUMER_SAVE,IFAIL,ISIGN
      COMMON/CIFAIL/IFAIL
      DOUBLE PRECISION ATOTSQ,A1SQ,A2SQ,ERR_FR
      INTEGER ISTRAT,IRAD,ITAU,ICOM
      LOGICAL ENDOFRUN,IS_ST,IS_4L
      COMMON/CENDOFRUN/ENDOFRUN
      INTEGER MAXEVV
      COMMON/CMAXEVV/MAXEVV
      INTEGER RNDEVSEED
      COMMON/CRNDEVSEED/RNDEVSEED
c
      ENDOFRUN=.FALSE.
      M_Z=0D0
      M_W=0D0
      M_T=0D0
      M_H=0D0
      M_D=0D0
      M_U=0D0
      M_S=0D0
      M_C=0D0
      M_B=0D0
      M_G=0D0
      M_E=0D0
      M_MU=0D0
      M_TAU=0D0
      G_Z=0D0
      G_W=0D0
      G_T=0D0
      G_H=0D0
C
      OLDFORM=.FALSE.
      WRITE(*,*)'Enter filename for events'
      READ(*,*)QQIN
      WRITE(*,*)'Enter maximum number of events to generate'
      WRITE(*,*)'must coincide with the number of events on tape'
      READ(*,*)MAXEV
      MAXEVV=MAXEV
      MQQ=MAXEV
      WRITE(*,*)'Enter 0 to use Pythia default PDFs'
      WRITE(*,*)'      1 to use PDFs from a library'
      READ(*,*)IPDF
C OUTPUT THE RESULTS AFTER ANY NSTEP EVENTS
      NSTEP=20000
C---BEAM PARTICLES
      WRITE(*,*)'Enter colliding particles (PART1, PART2)'
      READ(*,*)PART1,PART2
C---BEAM MOMENTA
      WRITE(*,*)'Enter beam momenta (PBEAM1, PBEAM2)'
      READ(*,*)PBEAM1,PBEAM2
      IPROC=-18000
C---SHOWER TYPE
      WRITE(*,*)'Enter 0 to use Q^2-ordered Pythia'
      WRITE(*,*)'      1 to use pT-ordered Pythia'
      READ(*,*)ISHW_PY
      IF(ISHW_PY.NE.0.AND.ISHW_PY.NE.1)THEN
         WRITE(*,*)'ishw_py',ISHW_PY,' not supported'
         STOP
      ENDIF
C---GENERAL SETTINGS
      WRITE(*,*)'Enter the value of 1/alpha_em'
      READ(*,*)PARU(101)
      PARU(101)=1D0/PARU(101)
      pi_stable=.false.
      WRITE(*,*)'Do you want a stable Pi0 (.TRUE. or .FALSE)?'
      READ(*,*)PI_STABLE
      WRITE(*,*)'Do you want a stable W+ (.TRUE. or .FALSE)?'
      READ(*,*)WP_STABLE
      WRITE(*,*)'Do you want a stable W- (.TRUE. or .FALSE)?'
      READ(*,*)WM_STABLE
      WRITE(*,*)'Do you want a stable Z (.TRUE. or .FALSE)?'
      READ(*,*)Z_STABLE
      WRITE(*,*)'Do you want a stable Tau+ (.TRUE. or .FALSE)?'
      READ(*,*)TAUP_STABLE
      WRITE(*,*)'Do you want a stable Tau- (.TRUE. or .FALSE)?'
      READ(*,*)TAUM_STABLE
      WRITE(*,*)'Do you want a stable Mu+ (.TRUE. or .FALSE)?'
      READ(*,*)MUP_STABLE
      WRITE(*,*)'Do you want a stable Mu- (.TRUE. or .FALSE)?'
      READ(*,*)MUM_STABLE
      WRITE(*,*)'Do you want a stable Higgs (.TRUE. or .FALSE)?'
      READ(*,*)H_STABLE
      if(PI_STABLE)MDCY(PYCOMP(111),1)=0
      if(WP_STABLE)MDCY(PYCOMP(24),1)=0
      if(WM_STABLE)MDCY(PYCOMP(-24),1)=0
      if(Z_STABLE)MDCY(PYCOMP(23),1)=0
      if(TAUP_STABLE)MDCY(PYCOMP(-15),1)=0
      if(TAUM_STABLE)MDCY(PYCOMP(15),1)=0
      if(MUP_STABLE)MDCY(PYCOMP(-13),1)=0
      if(MUM_STABLE)MDCY(PYCOMP(13),1)=0
      if(H_STABLE)MDCY(PYCOMP(25),1)=0
      WRITE(*,*)'Enter number of events to print'
      READ(*,*)MAXPR
      if(MAXPR.gt.MAXEV)MAXPR=MAXEV
      WRITE(*,*)'Enter accepted error fraction'
      READ(*,*)ERR_FR
      if(err_fr.lt.0d0.or.err_fr.gt.1d0)then
         write(*,*)'ERR_FR should be between 0 and 1 !'
         stop
      endif
      MAXER=INT(MAXEV*ERR_FR)

      is_st=.false.
      WRITE(*,*)'Is it single-top (.TRUE. or .FALSE)?'
      READ(*,*)is_st
C POSSIBILITY TO MAKE BARIONS STABLE (AVOIDS TROUBLES FOR SINGLE TOP)
      IF(is_st)THEN
         DO ISIGN=-1,1,2
            MDCY(PYCOMP(ISIGN*511),1)=0
            MDCY(PYCOMP(ISIGN*521),1)=0
            MDCY(PYCOMP(ISIGN*531),1)=0
            MDCY(PYCOMP(ISIGN*541),1)=0
            MDCY(PYCOMP(551),1)=0
            MDCY(PYCOMP(553),1)=0
            MDCY(PYCOMP(ISIGN*5112),1)=0
            MDCY(PYCOMP(ISIGN*5122),1)=0
            MDCY(PYCOMP(ISIGN*5132),1)=0
            MDCY(PYCOMP(ISIGN*5222),1)=0
            MDCY(PYCOMP(5232),1)=0
            MDCY(PYCOMP(ISIGN*5332),1)=0
         ENDDO
      ENDIF

      CALL PYTHIA_SETTING_GENERAL
C UNDERLYING EVENT
      WRITE(*,*)'Enter .TRUE. for Underlying event, .FALSE. otherwise'
      READ(*,*)MSTP(81)
      MSTP(81)=MSTP(81)+20*ISHW_PY
      WRITE(*,*)
      IF(MOD(MSTP(81),10).eq.1)then
         WRITE(*,*)'Underlying event WILL be generated'
      ELSEIF(MOD(MSTP(81),10).eq.0)then
         WRITE(*,*)'Underlying event WILL NOT be generated'
      ELSE
         WRITE(*,*)'Unknown option for Underlying event'
         WRITE(*,*)'Turning it off'
         MSTP(81)=20*ISHW_PY
      ENDIF
      WRITE(*,*)

C HADRONIZATION
      WRITE(*,*)'Enter 1 for hadronization, 0 otherwise'
      READ(*,*)MSTP(111)
      WRITE(*,*)
      IF(MSTP(111).eq.1)WRITE(*,*)'Hadronization WILL be included'
      IF(MSTP(111).eq.0)
     &WRITE(*,*)'Hadronization WILL NOT be included'
      WRITE(*,*)

      IF(IPDF.EQ.0)THEN
         MSTP(52)=1             !INTERNAL PDF LIBRARY (250)
         MSTP(51)=7             !DEFAULT PDF SET (249)
      ELSEIF(IPDF.EQ.1)THEN
         DO I=1,2
            WRITE(*,*)'   Incoming particle # ',I
            WRITE(*,*)'Enter PDF group name (AUTPDF)'
            READ(*,*)AUTPDF(I)
            WRITE(*,*)'Enter PDF set number (MODPDF)'
            READ(*,*)MODPDF(I)
         ENDDO
C---SET LHACRTL TO FALSE IF LHAPDF DEFAULTS FOR 16, 18, AND 19 ARE OK
         IF(LHACRTL.AND.
     #      (AUTPDF(1).EQ.'LHAPDF'.OR.AUTPDF(1).EQ.'LHAEXT'))THEN
            MSTP(52)=2          !EXTERNAL PDF LIBRARY (250)
            MSTP(51)=MODPDF(1)  !PDF SET FOR EXTERNAL LIBRARY (250)
            LHAPDF='FREEZE'
            IF(AUTPDF(1).EQ.'LHAEXT')LHAPDF='EXTRAPOLATE'
            CALL SETLHACBLK(LHAPDF)
         ENDIF      
      ELSE
         WRITE(*,*)'WRONG IPDF VALUE = ',IPDF
         STOP
      ENDIF
      WRITE(*,*)'Enter Lambda_5, <0 FOR Pythia default:'
      READ(*,*)TMPLAM
C
      WRITE(*,*)'Enter Z mass, width'
      READ(*,*)M_Z,G_Z
      WRITE(*,*)'Enter W mass, width'
      READ(*,*)M_W,G_W
      WRITE(*,*)'Enter top mass, width'
      READ(*,*)M_T,G_T
      WRITE(*,*)'Enter Higgs (SM) boson mass, width'
      READ(*,*)M_H,G_H
      WRITE(*,*)'Enter quark (d,u,s,c,b) and gluon masses'
      READ(*,*)M_D,M_U,M_S,M_C,M_B,M_G
      WRITE(*,*)'Enter lepton (e,mu,tau) masses'
      READ(*,*)M_E,M_MU,M_TAU

C---OPEN LH FILE
      OPEN(UNIT=MSTP(161),FILE=QQIN)
      OPEN(UNIT=MSTP(162),FILE=QQIN)         
      OPEN(UNIT=30,FILE=QQIN)
      WRITE(*,*)' '
      WRITE(*,*)'******* '
      WRITE(*,*)'******* THE EVENT FILE IS:  ',QQIN
      WRITE(*,*)'******* '
      WRITE(*,*)' '

      WRITE(*,*)'Enter random seed (0 for default)'
      READ(*,*)RNDEVSEED

C---INITIALISE ELEMENTARY PROCESS
      CALL PYINIT('USER',' ',' ',0D0)
C---SET LAMBDA_QCD ACCORDING TO LHAPDF OR TO THE INPUT FILE
      QCDLAM=-1.D0
      IF(IPDF.EQ.1)CALL GETLAMLHA(QCDLAM,1D0)
      IF(TMPLAM.GE.0.D0)QCDLAM=TMPLAM
******************************************
C---PARAMETERS AND ROUTINE RELEVANT TO PARTON SHOWER FORM FOUR
C---FINAL STATE FERMIONS. COMMENT OUT IF THIS IS NOT THE CASE.
C---THE ROUTINES PY2FRM AND PY6FRM ARE ANALOGOUS FOR THE CASES
C---OF TWO OR SIX FERMIONS.
C---SEE MANUAL, PAGES 321 (PY2FRM), 322 (PY4FRM), 324 (PY6FRM).
      is_4l=.false.
      WRITE(*,*)'Is it 4 leptons (.TRUE. or .FALSE)?'
      READ(*,*)is_4l
      if(is_4l)then
         ATOTSQ=1d0
         A1SQ=1d0
         A2SQ=1d0
         ISTRAT=0
         IRAD=0
         ITAU=0
         ICOM=1
         CALL PY4FRM(ATOTSQ,A1SQ,A2SQ,ISTRAT,IRAD,ITAU,ICOM)
      endif
c$$$      CALL PY2FRM(IRAD,ITAU,ICOM)
c$$$******************************************

      WRITE(*,*)'How many decay modes do you want to set?'
      READ(*,*)NUMDM
      IF(NUMDM.NE.0)THEN
         NMOTH=1
         DO II=1,NUMDM
            IMIN(II)=100000
            IMAX(II)=-100000
            ARRMOTH(II)=0
            DO JJ=1,5
               IDAUGHT(II,JJ)=0
            ENDDO
            DO JJ=1,NUMDM
               BRR(II,JJ)=0D0
               SUMBR(II,JJ)=0D0
            ENDDO
            IMOTH(II)=0
            NBODIES(II)=0
            BR(II)=0D0

            WRITE(*,*)'Enter number of bodies for decay mode ',II
            READ(*,*)NBODIES(II)
            WRITE(*,*)'Enter decay mode A --> B C ... with the syntax'
            WRITE(*,*)'DM_NUM = PDG(A) > PDG(B) PDG(C) ... @ BR @ ME'
            WRITE(*,*)'where 1 <= NUM <= 99'
            READ(*,*)IMOTH(II),TMPCH,(IDAUGHT(II,JJ),JJ=1,NBODIES(II)),
     &               TMPCH,BR(II),TMPCH,MEDEC(II)
            WRITE(CIMOTH(II),'(i4)')IMOTH(II)
            WRITE(CBR(II),'(f7.5)')BR(II)
 
            CIDAUGHTTOT(II)=''
            DO JJ=NBODIES(II),1,-1
               WRITE(CIDAUGHT(II,JJ),'(i4)')ABS(IDAUGHT(II,JJ))
               CIDAUGHTTOT(II)=CIDAUGHT(II,JJ)//CIDAUGHTTOT(II)
            ENDDO
            CALL PYGIVE(CIMOTH(II)//":alloff")
         ENDDO
         DO II=1,NUMDM
            CALL PYGIVE(CIMOTH(II)//":onifmatch = "//CIDAUGHTTOT(II))
         ENDDO
C HOW TO RESET BRANCHING RATIOS FOR PYTHIA?
      ENDIF

C---EVENTS ARE NORMALIZED TO SUM OR AVERAGE TO THE TOTAL CROSS SECTION
      WRITE(*,*)'How are the events normalized ("average" or "sum")?'
      READ(*,*)NORM_EVENT
      if (NORM_EVENT.eq.'average')MQQ=1

C---USER'S INITIAL CALCULATIONS
      CALL PYABEG
      SUMER=0
      SUMER_SAVE=0
C---LOOP OVER EVENTS
      DO 100 IEVT=1,MAXEV
C---INITIALISE EVENT
         CALL PYTHIA_SETTING_EVENT
         IF(ISHW_PY.EQ.0)CALL PYEVNT
         IF(ISHW_PY.EQ.1)CALL PYEVNW
         IF(IEVT.LE.MAXPR)THEN
            CALL PYLIST(1)
            WRITE(*,*)' '
         ENDIF
         IFAIL=0
         SUMER=NGEN(0,1)-NGEN(0,3)
         IF(SUMER.NE.SUMER_SAVE)THEN
            WRITE(*,*)'EVENT',IEVT,' FAILED'
            IFAIL=1
         ENDIF
         SUMER_SAVE=SUMER
         IF(SUMER.GT.MAXER)THEN
            WRITE(*,*)'TOO MANY FAILURES, STOP RUNNING'
            STOP
         ENDIF
C---USER'S EVENT ANALYSIS
         CALL PYANAL
C---FINALIZATION EVERY NSTEP EVENTS
         IF(MOD(IEVT,NSTEP).EQ.0.OR.MAXEV-IEVT-SUMER.EQ.0) THEN
            WRITE(*,*)'# OF EVENTS PROCESSED= ',IEVT
cc            CALL PYAEND(IEVT)
            IF(MAXEV-IEVT-SUMER.EQ.0)GOTO 101
         ENDIF
 100  CONTINUE
 101  CONTINUE
      ENDOFRUN=.TRUE.
      CALL PYAEND(IEVT)
C---STATISTICS OF THE RUN
      CALL PYSTAT(1)
c      CALL PYSTAT(2)
C---CLOSE FILES
      CLOSE(MSTP(161))
      CLOSE(MSTP(162))
      CLOSE(30)

      STOP
      END



      SUBROUTINE PYTHIA_SETTING_GENERAL
C GENERAL SETTINGS FOR THE SHOWER. THE REFERENCE FOR THE MEANING OF 
C THESE QUANTITIES IS HEP-PH/0603175 AND IN PARENTHESES I INDICATE THE PAGE
      IMPLICIT NONE
      INTEGER N,NPAD,K,MSTU,MSTJ,KCHG,MDCY,MDME,KFDP,MSEL,MSELPD,MSUB,
     &     KFIN,MSTP,MSTI,ISHW_PY,IPDF,MODPDF(2),ISIGN
      DOUBLE PRECISION P,V,PARJ,PMAS,PARF,VCKM,BRAT,CKIN,PARP,PARI,
     &     GAMMAX,PARU
      LOGICAL ALL_STABLE
      CHARACTER*20 AUTPDF(2)
      COMMON/CGAMMAX/GAMMAX
      COMMON/CPDFINFO/AUTPDF,MODPDF,IPDF
      COMMON/CISHW/ISHW_PY
      COMMON/PYJETS/N,NPAD,K(4000,5),P(4000,5),V(4000,5)
      COMMON/PYDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
      COMMON/PYDAT2/KCHG(500,4),PMAS(500,4),PARF(2000),VCKM(4,4)
      COMMON/PYDAT3/MDCY(500,3),MDME(8000,2),BRAT(8000),KFDP(8000,5)
      COMMON/PYSUBS/MSEL,MSELPD,MSUB(500),KFIN(2,-40:40),CKIN(200)
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200)
      INTEGER MRPY(6)
      INTEGER RNDEVSEED
      COMMON/CRNDEVSEED/RNDEVSEED
      DATA MRPY/19780503,0,0,97,33,0/
     &               !MRPY(1) IS THE SEED FOR RANDOM GENERATION (66)
      IF(RNDEVSEED.NE.0)MRPY(1)=RNDEVSEED !CHANGE RANDOM SEED
      CKIN(1)=2D0    !MINIMUM MASS (54)
      CKIN(2)=-1D0   !MAXIMUM MASS (54)
      CKIN(3)=0.5D0  !MINIMUM PT (52)
      CKIN(4)=-1.D0  !MAXIMUM PT (52)
      MSTJ(107)=0    !INITIAL STATE PHOTON RADIATION (108)
      MSTP(41)=2     !RESONANCE DECAYS (246)
      MSTP(61)=1     !INITIAL STATE SHOWER (253)
      MSTP(71)=1     !FINAL STATE SHOWER (253)
      MSTP(125)=2    !OUTPUT MODE (256)
C IF YOU DON'T USE MSTP(125)=2 THE PARTICLES WITH STATUS OTHER THAN
C 1,11,12,21 ARE NOT VISUALIZED, SO YOU LOOSE A LOT OF INFORMATION 
      MSTU(101)=0    !FIXED ALPHA_EM (265)
      MSTJ(39)=11    !MSTJ(39)=N MEANS NO RADIATION FROM PARTICLE WITH PDG CODE = N (NOT IN THE MANUAL)
      MSTJ(39)=13    !MSTJ(39)=N MEANS NO RADIATION FROM PARTICLE WITH PDG CODE = N (NOT IN THE MANUAL)
      MSTJ(41)=1     !TYPE OF BRANCHINGS (383)
      MSTJ(42)=2     !COLOR COHERENCE FOR TIMELIKE SHOWER (383)
      MSTJ(43)=3     !Z DEFINITION FOR FSR (384)
      MSTJ(47)=0     !MATRIX ELEMENT CORRECTIONS (385)
      MSTJ(48)=0     !MAXIMUM ANGLE FOR FIRST TIMELIKE EMISSION (386)
      MSTJ(50)=2     !COLOR COHERENCE FOR THE FIRST TIMELIKE EMISSION (386)
      MSTP(62)=3     !COLOR COHERENCE FOR SPACELIKE SHOWER (388)
      MSTP(67)=2     !COLOR COHERENCE FOR THE FIRST SPACELIKE EMISSION (389)
      MSTP(68)=0     !MATRIX ELEMENT CORRECTIONS (390)
      PARP(67)=1.D0  !FACTOR MULTIPLIYNG SCALUP FOR SPACELIKE SHOWER (392) 
      PARP(71)=1.D0  !FACTOR MULTIPLIYNG SCALUP FOR TIMELIKE SHOWER (392)
      MSTP(91)=0     !PRIMORDIAL KT DISTRIBUTION (423)
      MSTJ(93)=0     !ENABLE EXTERNAL VALUES FOR QUARK MASSES (491)
C DECAY CHANNELS FOR WEAK BOSONS (UPDATED NOTES FOR PYTHIA 6424 ON THE WEB)
      WRITE(*,*)' '

      RETURN
      END





      SUBROUTINE PYTHIA_SETTING_EVENT
C SETTINGS FOR THE CURRENT EVENT. THE REFERENCE FOR THE MEANING OF 
C THESE QUANTITIES IS HEP-PH/0603175 AND IN PARENTHESES I INDICATE THE PAGE
      IMPLICIT NONE
      INTEGER ISHW_PY,JPR
      DOUBLE PRECISION M_Z,M_W,M_T,M_H,M_D,M_U,M_S,M_C,M_B,M_G,M_E,M_MU,M_TAU,
     &G_Z,G_W,G_T,G_H,QCDLAM
      COMMON/CMASSES/M_Z,M_W,M_T,M_H,M_D,M_U,M_S,M_C,M_B,M_G,M_E,M_MU,M_TAU
      COMMON/CWIDTHS/G_Z,G_W,G_T,G_H
      COMMON/CQCD/QCDLAM
      COMMON/CISHW/ISHW_PY
      COMMON/CIPROC/JPR
      INTEGER N,NPAD,K,MSTU,MSTJ,KCHG,MDCY,MDME,KFDP,MSEL,MSELPD,
     &MSUB,KFIN,MSTP,MSTI
      DOUBLE PRECISION P,V,PARJ,PMAS,PARF,VCKM,BRAT,CKIN,PARP,PARI,PARU
      COMMON/PYJETS/N,NPAD,K(4000,5),P(4000,5),V(4000,5)
      COMMON/PYDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
      COMMON/PYDAT2/KCHG(500,4),PMAS(500,4),PARF(2000),VCKM(4,4)
      COMMON/PYDAT3/MDCY(500,3),MDME(8000,2),BRAT(8000),KFDP(8000,5)
      COMMON/PYSUBS/MSEL,MSELPD,MSUB(500),KFIN(2,-40:40),CKIN(200)
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200)
C
      PMAS(23,1)=M_Z
      PMAS(24,1)=M_W
      PMAS(25,1)=M_H
      PMAS(1,1)=M_D
      PMAS(2,1)=M_U
      PMAS(3,1)=M_S
      PMAS(4,1)=M_C
      PMAS(5,1)=M_B
      PMAS(6,1)=M_T
      PMAS(11,1)=M_E
      PMAS(13,1)=M_MU
      PMAS(15,1)=M_TAU
      PMAS(21,1)=M_G
C
      PMAS(6,2)=G_T
      PMAS(23,2)=G_Z
      PMAS(24,2)=G_W
      PMAS(25,2)=G_H
C
      MSTP(2)=2      !SECOND ORDER RUNNING ALPHA(235)
      IF(QCDLAM.EQ.-1.D0)THEN
         MSTP(3)=2   !CHOICE FOR LAMBDA_QCD (235)
         MSTP(64)=3  !USE LAMBDA_MC FOR ISR
c the line above can induce some effects on the system pt at low
c pt for processes dominated by ISR
      ELSEIF(QCDLAM.GE.0.D0)THEN
c note that if MSTP(3)=2 the lambdas are interpreted as lambda_4,
c while here PARP(61) and PARP(72) are always interpreted as lambda_5,
c (MSTU(112)=Nf only affects PARP(1)): a completely fair comparison
c between internal and external pdfs can be achieved only knowing
c the lambda_4 used in the shower for internal pdfs, and inputting
c the corresponding lambda_5 value for external pdf (for example using
c equation (2.50) of Ellis-Stirling-Webber)
         MSTP(3)=1   !USER-SET LAMBDA
         MSTP(64)=3  !USE LAMBDA_MC FOR ISR
         PARP(1) =QCDLAM
         PARP(61)=QCDLAM
         PARP(72)=QCDLAM
c$$$         PARJ(81)=QCDLAM
      ELSE
         WRITE(*,*)'WRONG LAMBDA_QCD VALUE',QCDLAM
         STOP
      ENDIF

      RETURN
      END
