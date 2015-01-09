C Driver, to be modified by the user (keeping the read statements).
C Should be backward compatible with v3.1 and v3.2 (thanks to B. Kersevan)
      PROGRAM HWIGPR
      INCLUDE 'HERWIG65.INC'
      INCLUDE 'JIMMY.INC'
      INTEGER N,NSTEP,I,JPR0,JPR,NUMDM,II,NBODIES(100),JJ,IMATCH
      INTEGER IMOTH(100),IDAUGHT(100,5),NMOTH,ARRMOTH(100),III,KK
      DOUBLE PRECISION BR(100),BRR(100,100),SUMBR(100,100)
      INTEGER IMIN(100),IMAX(100),MEDEC(100)
      CHARACTER *1 TMPCH
C QQIN IS THE EVENT FILE
      CHARACTER*50 QQIN
      COMMON/VVJIN/QQIN
      CHARACTER *7 NORM_EVENT
      INTEGER MQQ
      COMMON/cMQQ/MQQ
      REAL*8 TMPLAM,GAMT0,ERR_FR
      INTEGER IPDF
      CHARACTER * 70 LHAPDF
      LOGICAL LHACRTL,OLDFORM,PI_STABLE,WP_STABLE,WM_STABLE,Z_STABLE,
     &TAUP_STABLE,TAUM_STABLE,MUP_STABLE,MUM_STABLE,H_STABLE
      PARAMETER (LHACRTL=.TRUE.)
      LOGICAL ENDOFRUN,IS_ST,IS_BB,ABORT
      COMMON/CENDOFRUN/ENDOFRUN
      INTEGER MAXEVV
      COMMON/CMAXEVV/MAXEVV
      DOUBLE PRECISION EXPONT,PTJIM0,PRRAD
C
      ENDOFRUN=.FALSE.
      OLDFORM=.FALSE.
      WRITE(*,*)'Enter filename for events'
      READ(*,*)QQIN
      WRITE(*,*)'Enter maximum number of events to generate'
      WRITE(*,*)'MUST coincide with the number of events on tape'
      READ(*,*)MAXEV
      MAXEVV=MAXEV
      MQQ=MAXEV
      WRITE(*,*)'Enter 0 to use Herwig default PDFs'
      WRITE(*,*)'      1 to use PDFs from library'
      READ(*,*)IPDF
C OUTPUT THE RESULTS AFTER ANY NSTEP EVENTS
      NSTEP=20000
C---BEAM PARTICLES
      WRITE(*,*)'Enter colliding particles (PART1, PART2)'
      READ(*,*)PART1,PART2
C---BEAM MOMENTA
      WRITE(*,*)'Enter beam momenta (PBEAM1, PBEAM2)'
      READ(*,*)PBEAM1,PBEAM2
C---PROCESS
      IPROC=-18000
C---INITIALISE OTHER COMMON BLOCKS
      CALL HWIGIN
C     JIMMY initialization
      CALL JIMMIN
C---USER CAN RESET PARAMETERS AT
C   THIS POINT, OTHERWISE DEFAULT
C   VALUES IN HWIGIN WILL BE USED.
C
C PTVETO MUST BE TRUE IN ORDER FOR SCALUP TO BE USED ALSO
C IN CASE OF RADIATION OFF RESONANCES
      PTVETO=.TRUE. 
C************************************************************************
C---UNCOMMENT THE ASSIGNMENT PRESPL=.FALSE. WHEN USING HERWIG VERSION 6.52 
C---OR HIGHER (SEE MC@NLO MANUAL, APPENDIX A.8, PAGE 25)
      PRESPL=.FALSE.
C************************************************************************
C UNDERLYING EVENT
      WRITE(*,*)'Enter .TRUE. for underlying event, .FALSE. otherwise'
      READ(*,*)LHSOFT
      WRITE(*,*)
      IF(LHSOFT)WRITE(*,*)'Underlying event WILL be generated'
      IF(.NOT.LHSOFT)WRITE(*,*)'Underlying event WILL NOT be generated'
      WRITE(*,*)
C
      IF(LHSOFT.AND.IPDF.NE.1)THEN
         WRITE(*,*)' '
         WRITE(*,*)'Underlying event requires external PDF sets'
         WRITE(*,*)'Please set PDFCODE != 0 in shower_card.dat'
         STOP
      ENDIF
      IF(IPDF.EQ.1.OR.LHSOFT)THEN
         IF(IPDF.EQ.1)THEN
            DO I=1,2
               WRITE(*,*)'   Incoming particle # ',I
               WRITE(*,*)'Enter PDF group name (AUTPDF)'
               READ(*,*)AUTPDF(I)
               WRITE(*,*)'Enter PDF set number (MODPDF)'
               READ(*,*)MODPDF(I)
            ENDDO
         ENDIF
         IF(LHSOFT)THEN
C---JIMMY UNDERLYING EVENT OVERRIDES PDF ASSIGNMENTS: MUST USE LO**
            WRITE(*,*)' '
            WRITE(*,*)'JIMMY underlying event requires PDFs=LO**'
            DO I=1,2
               IF(AUTPDF(I).NE.'LHAPDF'.AND.AUTPDF(I).NE.'LHAEXT')THEN
                  WRITE(*,*)'PDF error 1 in MG5_aMC@NLO'
                  STOP
               ENDIF
               IF(MODPDF(I).NE.20651)THEN
                  WRITE(*,*)'Forcing PDFs to be 20651'
                  MODPDF(I)=20651
               ENDIF
            ENDDO
         ENDIF
         WRITE(*,*)' '

C---SET LHACRTL TO FALSE IF LHAPDF DEFAULTS FOR 16, 18, AND 19 ARE OK
         IF(LHACRTL.AND.
     #      (AUTPDF(1).EQ.'LHAPDF'.OR.AUTPDF(1).EQ.'LHAEXT'))THEN
            LHAPDF='FREEZE'
            IF(AUTPDF(1).EQ.'LHAEXT')LHAPDF='EXTRAPOLATE'
            CALL SETLHACBLK(LHAPDF)
C---MODERN VERSIONS OF LHAPDF REQUIRE THE FOLLOWING SETTING
            DO I=1,2
               AUTPDF(I)='HWLHAPDF'
            ENDDO
         ENDIF
      ENDIF

      WRITE(*,*)'Enter Lambda_QCD, <0 for Herwig default'
      READ(*,*)TMPLAM
C
      WRITE(*,*)'Enter Z mass, width'
      READ(*,*)RMASS(200),GAMZ
      WRITE(*,*)'Enter W mass, width'
      READ(*,*)RMASS(198),GAMW
      RMASS(199)=RMASS(198)
      WRITE(*,*)'Enter top mass, width'
      READ(*,*)RMASS(6),GAMT0
      WRITE(*,*)'Enter Higgs (SM) boson mass, width'
      READ(*,*)RMASS(201),GAMH
      WRITE(*,*)'Enter quark (d,u,s,c,b) and gluon masses'
      READ(*,*)RMASS(1),RMASS(2),RMASS(3),RMASS(4),RMASS(5),RMASS(13)
      WRITE(*,*)'Enter lepton (e,mu,tau) masses'
      READ(*,*)RMASS(121),RMASS(123),RMASS(125)
      DO I=1,5
         RMASS(I+6)  =RMASS(I)
         RMASS(I+126)=RMASS(I+120)
      ENDDO

C NO SOFT AND HARD ME CORRECTIONS (ALREADY INCLUDED IN MC@NLO)
      SOFTME=.FALSE.
      HARDME=.FALSE.
      ZMXISR=0 ! No photon radiation from ISR
      NOWGT=.FALSE.
C NEGATIVE WEIGHTS ALLOWED
      NEGWTS=.TRUE.
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
      is_bb=.false.
      is_st=.false.
      WRITE(*,*)'Is it single-top (.TRUE. or .FALSE)?'
      READ(*,*)is_st
      WRITE(*,*)'Is it b-bbar (.TRUE. or .FALSE)?'
      READ(*,*)is_bb
      if(is_st.and.is_bb)then
         write(*,*)'It cannot be single top and b bbar at the same time'
         stop
      endif
      LRSUD=0
      LWSUD=77
C IN THE CASE HERWIG PDFS ARE USED, ADOPT MRST
      NSTRU=8
      PRVTX=.FALSE.
      PTMIN=0.5
      WRITE(*,*)'Enter the two random seeds (0 0 for default)'
      READ(*,*)NRN(1),NRN(2)
      if(NRN(1).eq.0)NRN(1)=1973774260
      if(NRN(2).eq.0)NRN(2)=1099242306
C THE FOLLOWING SHOULD BE USED ONLY IN WEIGHTED MODE
      IF(.NOT.NOWGT)THEN
        WGTMAX=1.000001D0
        AVABW=1.000001D0
      ENDIF
C FOR TOP PRODUCTION (HARMLESS ELSEWHERE)
      RLTIM(6)=1.D-23
      RLTIM(12)=1.D-23
C---B FRAGMENTATION PARAMETERS (FOR B PRODUCTION ONLY)
      IF(is_bb)PSPLT(2)=0.5
C---COMPUTE PARAMETER-DEPENDENT CONSTANTS
      CALL HWUINC
C---SET LAMBDA_QCD ACCORDING TO LHAPDF OR TO THE INPUT FILE
      IF(IPDF.EQ.1)CALL GETLAMLHA(QCDLAM,-1D0)
      IF(TMPLAM.GE.0.D0)QCDLAM=TMPLAM
C---CALL HWUSTA TO MAKE ANY PARTICLE STABLE
      pi_stable=.false.
      WRITE(*,*)'Do you want a stable Pi0 (.TRUE. or .FALSE)?'
      READ(*,*)PI_STABLE
      WRITE(*,*)'Do you want a stable W+ (.TRUE. or .FALSE)?'
      READ(*,*)WP_STABLE
      WRITE(*,*)'Do you want a stable W- (.TRUE. or .FALSE)?'
      READ(*,*)WM_STABLE
      WRITE(*,*)'Do you want a stable Z0 (.TRUE. or .FALSE)?'
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

      if(PI_STABLE)  CALL HWUSTA('PI0     ')
      if(WP_STABLE)  CALL HWUSTA('W+      ')
      if(WM_STABLE)  CALL HWUSTA('W-      ')
      if(Z_STABLE)   CALL HWUSTA('Z0/GAMA*')
      if(TAUP_STABLE)CALL HWUSTA('TAU+    ')
      if(TAUM_STABLE)CALL HWUSTA('TAU-    ')
      if(MUP_STABLE) CALL HWUSTA('MU+     ')
      if(MUM_STABLE) CALL HWUSTA('MU-     ')
      if(H_STABLE)   CALL HWUSTA('HIGGS   ')
C---USE THE FOLLOWING FOR SINGLE TOP -- AVOIDS TROUBLES WITH ISR
      IF(is_st)THEN
        CALL HWUSTA('B+      ')
        CALL HWUSTA('B-      ')
        CALL HWUSTA('B_D0    ')
        CALL HWUSTA('B_DBAR0 ')
        CALL HWUSTA('B_S0    ')
        CALL HWUSTA('B_SBAR0 ')
        CALL HWUSTA('SIGMA_B+')
        CALL HWUSTA('LMBDA_B0')
        CALL HWUSTA('SIGMA_B-')
        CALL HWUSTA('XI_B0   ')
        CALL HWUSTA('XI_B-   ')
        CALL HWUSTA('OMEGA_B-')
        CALL HWUSTA('B_C-    ')
        CALL HWUSTA('ETA_B   ')
        CALL HWUSTA('UPSLON1S')
        CALL HWUSTA('SGM_BBR-')
        CALL HWUSTA('LMD_BBR0')
        CALL HWUSTA('SGM_BBR+')
        CALL HWUSTA('XI_BBAR0')
        CALL HWUSTA('XI_B+   ')
        CALL HWUSTA('OMG_BBR+')
        CALL HWUSTA('B_C+    ')
      ENDIF

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

            IMATCH=0
            DO III=1,NMOTH
               IF(IMOTH(II).EQ.ARRMOTH(III))THEN
                  IMATCH=1
               ENDIF
            ENDDO
            IF(IMATCH.EQ.0)THEN
               ARRMOTH(NMOTH)=IMOTH(II)
               NMOTH=NMOTH+1
            ENDIF
         ENDDO
         NMOTH=NMOTH-1
         
         DO JJ=1,NMOTH
            DO II=1,NUMDM
               IF(IMOTH(II).EQ.ARRMOTH(JJ))THEN
                  BRR(JJ,II)=BR(II)
                  IF(II.LT.IMIN(JJ))IMIN(JJ)=II
                  IF(II.GT.IMAX(JJ))IMAX(JJ)=II
               ENDIF
            ENDDO
         ENDDO

         SYSPIN=.TRUE.
         DO JJ=1,NMOTH
            DO II=1,NUMDM
               DO KK=II+1,NUMDM
                  SUMBR(JJ,II)=SUMBR(JJ,II)+BRR(JJ,KK)
               ENDDO
               IF(NBODIES(II).EQ.3)THREEB=.TRUE.
               IF(NBODIES(II).EQ.4)FOURB=.TRUE.
               IF(MEDEC(II).EQ.0)SYSPIN=.FALSE.
               IF(ARRMOTH(JJ).EQ.IMOTH(II))THEN
                  IF(II.EQ.IMAX(JJ).OR.II.EQ.IMIN(JJ))THEN
                     CONTINUE
                  ELSE
                     BRR(JJ,II)=BRR(JJ,II)/(1-SUMBR(JJ,II))
                  ENDIF
                  CALL HWMODK(IMOTH(II),BRR(JJ,II),MEDEC(II),
     &                    IDAUGHT(II,1),IDAUGHT(II,2),IDAUGHT(II,3),
     &                    IDAUGHT(II,4),IDAUGHT(II,5))
               ENDIF
            ENDDO
         ENDDO
      ENDIF
C---EVENTS ARE NORMALIZED TO SUM OR AVERAGE TO THE TOTAL CROSS SECTION
      WRITE(*,*)'How are the events normalized ("average" or "sum")?'
      READ(*,*)NORM_EVENT
      if (NORM_EVENT.eq.'average')MQQ=1

      MSFLAG=0
      IF (LHSOFT) THEN
C---ATLAS JIMMY tune AUET2 (ATL-PHYS-PUB-2011-008, table 13)
         PRRAD=2.339D0
         PTJIM0=3.696D0
         EXPONT=0.219D0
C---AUET2 fixed parameters
         ISPAC=2
         PTRMS=1.2d0
         QSPAC=2.5d0
C---Beam 1/radii (assumed proton or antiproton)
         JMU2=PRRAD
         JMV2=PRRAD
C---Minimum PT of secondary scatters
         PTJIM=PTJIM0*(PBEAM1*PBEAM2/8.1D5)**(EXPONT/2)
C---Turn MI on(1) or off(0)
         MSFLAG=1
         CALL JMINIT
      ENDIF

C---INITIALISE ELEMENTARY PROCESS
      CALL HWEINI
C---USER'S INITIAL CALCULATIONS
      CALL HWABEG
C---LOOP OVER EVENTS
      DO 100 N=1,MAXEV
C---INITIALISE EVENT
         CALL HWUINE
C---GENERATE HARD SUBPROCESS
         CALL HWEPRO
C---GENERATE PARTON CASCADES
         CALL HWBGEN
C---  GENERATE MULTIPARTON INTERACTIONS
         IF (MSFLAG.EQ.1) THEN
            CALL HWMSCT(ABORT)
            IF (ABORT) GOTO 50
         ENDIF
C---DO HEAVY OBJECT DECAYS
         CALL HWDHOB
c         CALL HWUDPR
C---DO CLUSTER FORMATION
         CALL HWCFOR
C---DO CLUSTER DECAYS
         CALL HWCDEC
C---DO UNSTABLE PARTICLE DECAYS
         CALL HWDHAD
C---DO HEAVY FLAVOUR HADRON DECAYS
         CALL HWDHVY
C---ADD SOFT UNDERLYING EVENT IF NEEDED
         CALL HWMEVT
C---FINISH EVENT
         CALL HWUFNE
C---USER'S EVENT ANALYSIS
         CALL HWANAL
         IF(MOD(NEVHEP,NSTEP).EQ.0) THEN
            WRITE(*,*)'# of events processed=',NEVHEP
c            CALL HWAEND
         ENDIF
 50      CONTINUE
  100 CONTINUE
C---TERMINATE ELEMENTARY PROCESS
      CALL HWEFIN
C     Finish off JIMMY
      IF (LHSOFT) CALL JMEFIN
C---USER'S TERMINAL CALCULATIONS
      WRITE(*,*)'# of events processed=',NEVHEP
      ENDOFRUN=.TRUE.
      CALL HWAEND
C---CLEAN EXIT IF USING ROOT; DUMMY OTHERWISE
      CALL RCLOS()
 999  STOP
      END
