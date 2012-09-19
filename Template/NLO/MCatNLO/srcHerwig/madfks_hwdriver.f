C Driver, to be modified by the user (keeping the read statements).
C Should be backward compatible with v3.1 and v3.2 (thanks to B. Kersevan)
      PROGRAM HWIGPR
      INCLUDE 'HERWIG65.INC'
      INTEGER N,NSTEP,I,JPR0,JPR
C QQIN IS THE EVENT FILE
      CHARACTER*50 QQIN
      COMMON/VVJIN/QQIN
      REAL*8 TMPLAM,GAMT0,ERR_FR
      INTEGER IPDF
      CHARACTER * 70 LHAPDF
      LOGICAL LHACRTL,OLDFORM,PI_STABLE
      PARAMETER (LHACRTL=.TRUE.)
      LOGICAL ENDOFRUN,IS_ST,IS_BB
      COMMON/CENDOFRUN/ENDOFRUN
      INTEGER MAXEVV
      COMMON/CMAXEVV/MAXEVV
c
      ENDOFRUN=.FALSE.
      OLDFORM=.FALSE.
      WRITE(*,*)'Enter filename for events'
      READ(*,*)QQIN
      WRITE(*,*)'Enter maximum number of events to generate'
      WRITE(*,*)'MUST coincide with the number of events on tape'
      READ(*,*)MAXEV
      MAXEVV=MAXEV
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
C---USER CAN RESET PARAMETERS AT
C   THIS POINT, OTHERWISE DEFAULT
C   VALUES IN HWIGIN WILL BE USED.
C
C************************************************************************
C---UNCOMMENT THE ASSIGNMENT PRESPL=.FALSE. WHEN USING HERWIG VERSION 6.52 
C---OR HIGHER (SEE MC@NLO MANUAL, APPENDIX A.8, PAGE 25)
      PRESPL=.FALSE.
C************************************************************************
C UNDERLYING EVENT
      WRITE(*,*)'Enter .TRUE. for Underlying event, .FALSE. otherwise'
      READ(*,*)LHSOFT
      WRITE(*,*)
      IF(LHSOFT)WRITE(*,*)'Underlying event WILL be generated'
      IF(.NOT.LHSOFT)WRITE(*,*)'Underlying event WILL NOT be generated'
      WRITE(*,*)

      WRITE(*,*)'Enter decay modes for gauge bosons'
      READ(*,*)MODBOS(1),MODBOS(2)
C
      IF(IPDF.EQ.1)THEN
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
      IF(TMPLAM.GE.0.D0)QCDLAM=TMPLAM
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
      READ(*,*)RMASS(1),RMASS(2),RMASS(3),
     #         RMASS(4),RMASS(5),RMASS(13)
      DO I=1,5
         RMASS(I+6)=RMASS(I)
      ENDDO
C Set electron and muon masses equal to zero to avoid rounding problems
      RMASS(121)=0.D0
      RMASS(123)=0.D0
      RMASS(127)=0.D0
      RMASS(129)=0.D0
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
C---CALL HWUSTA TO MAKE ANY PARTICLE STABLE
      pi_stable=.false.
      WRITE(*,*)'Do you want a stable Pi0 (.TRUE. or .FALSE)?'
      READ(*,*)PI_STABLE
      if(PI_STABLE)CALL HWUSTA('PI0     ')
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
        CALL HWUSTA('UPSLON1S')
        CALL HWUSTA('SGM_BBR-')
        CALL HWUSTA('LMD_BBR0')
        CALL HWUSTA('SGM_BBR+')
        CALL HWUSTA('XI_BBAR0')
        CALL HWUSTA('XI_B+   ')
        CALL HWUSTA('OMG_BBR+')
        CALL HWUSTA('B_C+    ')
      ENDIF
C---USER'S INITIAL CALCULATIONS
      CALL HWABEG
C---INITIALISE ELEMENTARY PROCESS
      CALL HWEINI
C---LOOP OVER EVENTS
      DO 100 N=1,MAXEV
C---INITIALISE EVENT
         CALL HWUINE
C---GENERATE HARD SUBPROCESS
         CALL HWEPRO
C---GENERATE PARTON CASCADES
         CALL HWBGEN
C---DO HEAVY OBJECT DECAYS
         CALL HWDHOB
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
            CALL HWAEND
         ENDIF
  100 CONTINUE
C---TERMINATE ELEMENTARY PROCESS
      CALL HWEFIN
C---USER'S TERMINAL CALCULATIONS
      WRITE(*,*)'# of events processed=',NEVHEP
      ENDOFRUN=.TRUE.
      CALL HWAEND
C---CLEAN EXIT IF USING ROOT; DUMMY OTHERWISE
      CALL RCLOS()
 999  STOP
      END
