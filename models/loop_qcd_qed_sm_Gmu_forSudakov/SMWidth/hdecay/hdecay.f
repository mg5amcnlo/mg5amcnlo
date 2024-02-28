C         Last modification on January 5 2015 by M.S.
C ==================================================================
C ================= PROGRAM HDECAY: COMMENTS =======================
C ==================================================================
C
C                       ****************
C                       * VERSION 6.41 *
C                       ****************
C
C
C  This program calculates the total decay widths and the branching 
C  ratios of the C Standard Model Higgs boson (HSM) as well as those 
C  of the neutral (HL= the light CP-even, HH= the heavy CP-even, HA= 
C  the pseudoscalar) and the charged (HC) Higgs bosons of the Minimal
C  Supersymmetric extension of the Standard Model (MSSM) as well as a
C  general Two-Higgs-Doublet model (2HDM). It includes:
C
C - All the decay channels which are kinematically allowed and which
C   have branching ratios larger than 10**(-4). 
C
C - All QCD corrections to the fermionic and gluonic decay modes.
C   Most of these corrections are mapped into running masses in a
C   consistent way with some freedom for including high order terms. 
C
C - Below--threshold three--body decays with off--shell top quarks
C   or ONE off-shell gauge boson, as well as some decays with one
C   off-shell Higgs boson in the MSSM/2HDM. 
C
C - Double off-shell decays: HSM,HL,HH --> W*W*,Z*Z* -->4 fermions,
C   which could be important for Higgs masses close to MW or MZ.
C
C - In the MSSM, the radiative corrections with full squark mixing and 
C   uses the RG improved values of Higgs masses and couplings with the 
C   main NLO corrections implemented (based on M.Carena, M. Quiros and
C   C.E.M. Wagner, Nucl. Phys. B461 (1996) 407, hep-ph/9508343). 
C
C - In the MSSM, all the decays into CHARGINOS, NEUTRALINOS, SLEPTONS 
C   and SQUARKS (with mixing in the stop and sbottom sectors). 
C
C - Chargino, slepton and squark loops in the 2 photon decays and squark
C   loops in the gluonic decays (including QCD corrections). 
C
C  ===================================================================
C  This program has been written by A.Djouadi, J.Kalinowski, M.
C  Muehlleitner and M.Spira. For details on how to use the program see:
C  Comp. Phys. Commun. 108 (1998) 56, hep-ph/9704448.
C
C  For any question, comment, suggestion or complaint, please contact us at:
C          Abdelhak.Djouadi@th.u-psud.fr
C          kalino@fuw.edu.pl
C          maggie@particle.uni-karlsruhe.de
C          Michael.Spira@psi.ch
C
C
C ================ IT USES AS INPUT PARAMETERS:
C
C   SLHAIN: =0: READ FROM hdecay.in
C           =1: READ SUSY LES HOUCHES ACCORD INPUT (slha.in)
C
C  SLHAOUT: =0: WRITE BR TABLES
C           =1: WRITE SUSY LES HOUCHES ACCORD OUTPUT (slha.out)
C
C  COUPVAR: =0: NO VARIATION OF HIGGS COUPLINGS
C           =1: VARIATION OF HIGGS COUPLINGS       (ONLY FOR SM)
C
C    HIGGS: =0: CALCULATE BRANCHING RATIOS OF SM HIGGS BOSON
C           =1: CALCULATE BRANCHING RATIOS OF MSSM h BOSON
C           =2: CALCULATE BRANCHING RATIOS OF MSSM H BOSON
C           =3: CALCULATE BRANCHING RATIOS OF MSSM A BOSON
C           =4: CALCULATE BRANCHING RATIOS OF MSSM H+ BOSON
C           =5: CALCULATE BRANCHING RATIOS OF ALL MSSM HIGGS BOSONS
C
C      SM4: =0: CALCULATE USUAL BRANCHING RATIOS
C           =1: HIGGS WITH 4TH GENERATION (SETS HIGGS, FERMPHOB = 0)
C
C FERMPHOB: =0: CALCULATE USUAL BRANCHING RATIOS
C           =1: FERMIOPHOBIC HIGGS (SETS HIGGS = 0)
C
C     2HDM: =0: CALCULATE USUAL BRNCHING RATIOS
C           =1: 2HDM (SETS HIGGS = 5)
C
C    MODEL: USE SPECIFIC SUBROUTINE FOR MSSM HIGSS MASSES AND COUPLINGS
C           =1: CARENA ET AL., NUCL. PHYS. B461 (1996) 407 (SUBHPOLE)
C           =2: CARENA ET AL., PHYS. LETT. B355 (1995) 209 (SUBH)
C           =3: HABER ET AL.
C           =4: HEINEMEYER ET AL., HEP-PH/0002213 (FEYNHIGGSFAST1.2.2)
C           =10: hMSSM
C
C TGBET:    TAN(BETA) FOR MSSM
C MABEG:    START VALUE OF M_A FOR MSSM AND M_H FOR SM
C MAEND:    END VALUE OF M_A FOR MSSM AND M_H FOR SM
C NMA:      NUMBER OF ITERATIONS FOR M_A
C MHL:      LIGHT SCALAR HIGGS MASS FOR hMSSM (MODEL = 10)
C ALS(MZ):  VALUE FOR ALPHA_S(M_Z)
C MSBAR(2): MSBAR MASS OF STRANGE QUARK AT SCALE Q=2 GEV
C MC:       CHARM POLE MASS
C MB:       BOTTOM POLE MASS
C MT:       TOP POLE MASS
C MTAU:     TAU MASS
C MMUON:    MUON MASS
C ALPH:     INVERSE QED COUPLING
C GF:       FERMI CONSTANT
C GAMW:     W WIDTH
C GAMZ:     Z WIDTH
C MZ:       Z MASS
C MW:       W MASS
C VTB:      CKM PARAMETER |V_TB|
C VTS:      CKM PARAMETER |V_TS|
C VTD:      CKM PARAMETER |V_TD|
C VCB:      CKM PARAMETER |V_CB|
C VCS:      CKM PARAMETER |V_CS|
C VCD:      CKM PARAMETER |V_CD|
C VUB:      CKM PARAMETER |V_UB|
C VUS:      CKM PARAMETER |V_US|
C VUD:      CKM PARAMETER |V_UD|
C GG_ELW:   SCENARIO OF THE ELW. CORRECTIONS TO H -> GG (4TH GENERATION)
C MTP:      TOP' MASS    (4TH GENERATION)
C MBP:      BOTTOM' MASS (4TH GENERATION)
C MNUP:     NU' MASS     (4TH GENERATION)
C MEP:      E' MASS      (4TH GENERATION)
C TYPE:     TYPE OF 2HDM: 1 (type I), 2 (type II)
C TGBET2HDM:TAN(BETA)
C ALPHA_H:  MIXING ANGLE IN THE CP-EVEN NEUTRAL HIGGS SECTOR
C MHL:      MASS OF THE LIGHT CP-EVEN HIGGS BOSON
C MHH:      MASS OF THE HEAVY CP-EVEN HIGGS BOSON
C MHA:      MASS OF THE CP-ODD HIGGS BOSON
C MH+-:     MASS OF THE CHARGED HIGGS BOSONS
C M_12^2:   PARAMETER M12 SQUARED
C SUSYSCALE: SCALE FOR SUSY BREAKING PARAMETERS
C 1ST AND 2ND GENERATION:
C MSL1:      SUSY BREAKING MASS PARAMETERS OF LEFT HANDED SLEPTONS 
C MER1:      SUSY BREAKING MASS PARAMETERS OF RIGHT HANDED SLEPTONS 
C MQL1:      SUSY BREAKING MASS PARAMETERS OF LEFT HANDED SUPS
C MUR1:      SUSY BREAKING MASS PARAMETERS OF RIGHT HANDED SUPS
C MDR1:      SUSY BREAKING MASS PARAMETERS OF RIGHT HANDED SDOWNS 
C 3RD GENERATION:
C MSL:      SUSY BREAKING MASS PARAMETERS OF LEFT HANDED STAUS 
C MER:      SUSY BREAKING MASS PARAMETERS OF RIGHT HANDED STAUS 
C MSQ:      SUSY BREAKING MASS PARAMETERS OF LEFT HANDED STOPS
C MUR:      SUSY BREAKING MASS PARAMETERS OF RIGHT HANDED STOPS
C MDR:      SUSY BREAKING MASS PARAMETERS OF RIGHT HANDED SBOTTOMS 
C AL:       STAU TRILINEAR SOFT BREAKING TERMS 
C AU:       STOP TRILINEAR SOFT BREAKING TERMS
C AD:       SBOTTOM TRILINEAR SOFT BREAKING TERMS
C MU:       SUSY HIGGS MASS PARAMETER
C M2:       GAUGINO MASS PARAMETER
C MGLUINO:  GLUINO POLE MASS
C
C NNLO (M): =0: USE O(ALPHA_S) FORMULA FOR POLE MASS --> MSBAR MASS
C           =1: USE O(ALPHA_S**2) FORMULA FOR POLE MASS --> MSBAR MASS
C
C ON-SHELL: =0: INCLUDE OFF_SHELL DECAYS H,A --> T*T*, A --> Z*H,
C               H --> W*H+,Z*A, H+ --> W*A, W*H, T*B
C           =1: EXCLUDE THE OFF-SHELL DECAYS ABOVE
C
C ON-SH-WZ: =0: INCLUDE DOUBLE OFF-SHELL PAIR DECAYS PHI --> W*W*,Z*Z*
C           =1: INCLUDE DOUBLE OFF-SHELL PAIR DECAYS PHI --> W*W*,Z*Z*
C               BELOW THRESHOLD, BUT ON-SHELL PAIR DECAYS ABOVE
C           =-1: INCLUDE ONLY SINGLE OFF-SHELL DECAYS PHI --> W*W,Z*Z
C                BELOW THRESHOLD, BUT ON-SHELL PAIR DECAYS ABOVE
C
C IPOLE:    =0 COMPUTES RUNNING HIGGS MASSES (FASTER) 
C           =1 COMPUTES POLE HIGGS MASSES 
C
C OFF-SUSY: =0: INCLUDE DECAYS (AND LOOPS) INTO SUPERSYMMETRIC PARTICLES
C           =1: EXCLUDE DECAYS (AND LOOPS) INTO SUPERSYMMETRIC PARTICLES
C
C INDIDEC:  =0: PRINT OUT SUMS OF CHARGINO/NEUTRALINO/SFERMION DECAYS
C           =1: PRINT OUT INDIVIDUAL CHARGINO/NEUTRALINO/SFERMION DECAYS
C
C NF-GG:    NUMBER OF LIGHT FLAVORS INCLUDED IN THE GLUONIC DECAYS 
C            PHI --> GG* --> GQQ (3,4 OR 5)
C           
C IGOLD:    =0: EXCLUDE DECAYS INTO GRAVITINO + GAUGINO
C           =1: INCLUDE DECAYS INTO GRAVITINO + GAUGINO
C
C MPLANCK:  PLANCK MASS FOR DECAYS INTO GRAVITINO + GAUGINO
C MGOLD:    GRAVITINO MASS FOR DECAYS INTO GRAVITINO + GAUGINO
C
C          RESCALING OF COUPLINGS
C          ----------------------
C
C ELWK:    = 0: Include elw. corrections only for SM part
C          = 1: Include elw. corrections in all rescalings of couplings
C CW:      RESCALING FACTOR OF HWW COUPLING
C CZ:      RESCALING FACTOR OF HZZ COUPLING
C Ctau:    RESCALING FACTOR OF HTAUTAU COUPLING
C Cmu:     RESCALING FACTOR OF HMUMU COUPLING
C Ct:      RESCALING FACTOR OF HTT COUPLING
C Cb:      RESCALING FACTOR OF HBB COUPLING
C Cc:      RESCALING FACTOR OF HCC COUPLING
C Cs:      RESCALING FACTOR OF HSS COUPLING
C Cgaga:   POINT-LIKE H-GAMMA-GAMMA COUPLING
C Cgg:     POINT-LIKE HGG COUPLING
C CZga:    POINT-LIKE H-Z-GAMMA COUPLING
C
C THE POINT-LIKE COUPLINGS ARE DEFINED IN TERMS OF THE LAGRANGIAN
C
C   L = ( alpha_s/8/pi Cgg G^{a\mu\nu}G^a_{\mu\nu}
C       + alpha/8/pi Cgaga F^{\mu\nu}F_{\mu\nu}
C       + sqrt(alpha alpha_2)/4/pi CZga F^{\mu\nu}Z_{\mu\nu}) H/v
C
C WHERE G^{a\mu\nu}, F^{\mu\nu} AND Z^{\mu\nu} ARE THE FIELD STRENGTH
C TENSORS OF THE GLUON, PHOTON AND Z BOSON FIELDS. THE COUPLINGS alpha
C AND alpha_2 ARE THE ELECTROMAGNETIC (IN THE THOMPSON LIMIT) AND
C ISPOSPIN COUPLINGS (g^2 = 4 pi alpha_2), RESPECTIVELY AND v IS THE
C HIGGS VACUUM EXPECTATION VALUE.
C
C =======================================================================
C ============== BEGINNING OF THE MAIN PROGRAM ==========================
C =======================================================================
C
C      PROGRAM HDECAY
C      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C      COMMON/HMASS_HDEC/AMSM,AMA,AML,AMH,AMCH,AMAR
C      COMMON/FLAGS_HDEC/INDIDEC
C      COMMON/SLHA_vals_HDEC/islhai,islhao
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
C      COMMON/DAVID/QSUSY1,QSUSY2,LOOP
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

C      CALL READ_HDEC(TGBET,AMABEG,AMAEND,NMA)
C      if(islhao.ne.1) then
C         CALL HEAD_HDEC(TGBET,AMABEG)
C      endif

c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     write(6,*)'Loop, Factor_QCD, Factor_elw = ?'
c     read(5,*)LOOP,QSUSY1,QSUSY2
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

C      DO 9999 II=1,NMA
C       IF(NMA.NE.1)THEN
C        AMAR = AMABEG + (AMAEND-AMABEG)/(NMA-1D0)*(II-1D0)
C       ELSE
C        AMAR = AMABEG
C       ENDIF
C       AMSM = AMAR
C       AMA = AMAR
C       CALL HDEC(TGBET)
C       CALL WRITE_HDEC(TGBET)
C 9999  CONTINUE

C      CALL CLOSE_HDEC

C      STOP
C      END

      SUBROUTINE READ_HDEC(TGBET,AMABEG,AMAEND,NMA)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER(K=6,NI=87,NSA=85,NSB=86,NLA=88,NLB=89,NHA=90,NHB=91,
     .          NHC=92,NAA=93,NAB=94,NCA=95,NCB=96,NCC=50,NRA=97,NRB=98,
     .          NSUSYL=81,NSUSYA=82,NSUSYH=83,NSUSYC=84,NPAR=80,
     .          NSUSYLA=79,NSUSYLB=78,NSUSYLC=77,NSUSYLD=76,NSUSYLE=75,
     .          NSUSYLF=59,NSUSYHF=58,
     .          NSUSYHA=74,NSUSYHB=73,NSUSYHC=72,NSUSYHD=71,NSUSYHE=70,
     .          NSUSYAA=69,NSUSYAB=68,NSUSYAC=67,NSUSYAD=66,NSUSYAE=65,
     .          NSUSYCA=64,NSUSYCB=63,NSUSYCC=62,NSUSYCD=61,NSUSYCE=60,
     .          ninlha=22)
      double precision minval(1:20),smval(1:30),massval(1:50),
     .                 nmixval(4,4),umixval(2,2),vmixval(2,2),
     .                 stopmixval(2,2),sbotmixval(2,2),staumixval(2,2),
     .                 hmixval(1:10),gaugeval(1:3),msoftval(1:100),
     .                 auval(3,3),adval(3,3),aeval(3,3),yuval(3,3),
     .                 ydval(3,3),yeval(3,3),qvalue(1:20),extval(0:100),
     .                 m_softval(1:100)
      double precision slhaneut(1:4),slhaxneut(1:4),slhachar(1:2),
     .                 slhaxchar(1:2),
     .                 slhau(2,2),slhav(2,2),slhaz(4,4),
     .                 slhast(2),slhasb(2),slhasu(2),slhasd(2),
     .                 slhase(2),slhasl(2),slhasn(2),slhasnl(2),
     .                 warning(1:10)
      double precision vckmval(4)
      integer   imod(1:2)
      integer check(1:22)
      double precision mbmsbar,mbl,mbu
      character spinfo1*100,spinfo2*100,modselval*100,mincom(1:20)*20,
     .          extcom(0:100)*20,softcom(1:100)*20,hmixcom(1:10)*20,
     .          m_softcom(1:100)*20
      DIMENSION GMN(4),XMN(4),GMC(2),GMST(2),GMSB(2),GMSL(2),
     .          GMSU(2),GMSD(2),GMSE(2),GMSN(2),GMSN1(2)
      DIMENSION HLBRSC(2,2),HLBRSN(4,4),HHBRSC(2,2),HHBRSN(4,4),
     .          HABRSC(2,2),HABRSN(4,4),HCBRSU(2,4),
     .          HHBRST(2,2),HHBRSB(2,2),HCBRSTB(2,2) 
      DIMENSION AC1(2,2),AC2(2,2),AC3(2,2),
     .          AN1(4,4),AN2(4,4),AN3(4,4),
     .          ACNL(2,4),ACNR(2,4)
      DIMENSION GLTT(2,2),GLBB(2,2),GHTT(2,2),GHBB(2,2),GCTB(2,2),
     .          GLEE(2,2),GHEE(2,2),GCEN(2,2)
      DIMENSION AGDL(4),AGDA(4),AGDH(4),AGDC(2)
c -------------- common block given by read_leshouches ------------ c
      COMMON/SLHA_leshouches1_HDEC/spinfo1,spinfo2,modselval,mincom,
     .                             extcom,softcom,hmixcom
      COMMON/SLHA_leshouches2_HDEC/minval,extval,smval,massval,nmixval,
     .                      umixval,vmixval,stopmixval,sbotmixval,
     .                      staumixval,hmixval,gaugeval,msoftval,auval,
     .                      adval,aeval,yuval,ydval,yeval,alphaval,
     .                      qvalue,imod
      COMMON/SLHA_leshouches3_HDEC/vckmval
      COMMON/SD_scaleofewsb/scaleofewsb
c -------------- common blocks needed in HDECAY subroutines ---------- c
      COMMON/SLHA_vals_HDEC/islhai,islhao
      COMMON/SLHA_m1_HDEC/am1
      COMMON/SLHA_gaug_HDEC/slhaneut,slhaxneut,slhachar,slhau,slhav,
     .                      slhaz,slhaxchar
      COMMON/SLHA_sfer_HDEC/slhast,slhasb,slhasu,slhasd,slhase,slhasl,
     .                 slhasn,slhasnl,slhacot,slhasit,slhacob,slhasib,
     .                 slhacol,slhasil
      COMMON/SLHA_hmass_HDEC/slhaml,slhamh,slhamc,slha_alpha
      COMMON/SLHAVAL_HDEC/g1ew,g2ew
      COMMON/SLHA_checkval_HDEC/check
      COMMON/MASSES_HDEC/AMS,AMC,AMB,AMT
      COMMON/STRANGE_HDEC/AMSB
      COMMON/PARAM_HDEC/GF,ALPH,AMTAU,AMMUON,AMZ,AMW
      COMMON/CKMPAR_HDEC/VTB,VTS,VTD,VCB,VCS,VCD,VUB,VUS,VUD
      COMMON/HMASS_HDEC/AMSM,AMA,AML,AMH,AMCH,AMAR
      COMMON/BREAKSCALE_HDEC/SUSYSCALE
      COMMON/BREAK_HDEC/AMEL,AMER,AMSQ,AMUR,AMDR,AL,AU,AD,AMU,AM2
      COMMON/BREAKGLU_HDEC/AMGLU
      COMMON/SFER1ST_HDEC/AMQL1,AMUR1,AMDR1,AMEL1,AMER1
      COMMON/GLUINO_HDEC/AMGLUINO,XMSB1,XMSB2,STHB,CTHB,
     .              XLBB(2,2),XHBB(2,2),XABB(2,2),
     .              XMST1,XMST2,STHT,CTHT,
     .              XLTT(2,2),XHTT(2,2),XATT(2,2)
      COMMON/WZWDTH_HDEC/GAMC0,GAMT0,GAMT1,GAMW,GAMZ
      COMMON/COUP_HDEC/GAT,GAB,GLT,GLB,GHT,GHB,GZAH,GZAL,
     .            GHHH,GLLL,GHLL,GLHH,GHAA,GLAA,GLVV,GHVV,
     .            GLPM,GHPM,B,A
      COMMON/ALS_HDEC/XLAMBDA,AMC0,AMB0,AMT0,N0
      COMMON/FLAG_HDEC/IHIGGS,NNLO,IPOLE
      COMMON/SM4_HDEC/AMTP,AMBP,AMNUP,AMEP,ISM4,IGGELW
      COMMON/FERMIOPHOBIC_HDEC/IFERMPHOB
      COMMON/MODEL_HDEC/IMODEL
      COMMON/ONSHELL_HDEC/IONSH,IONWZ,IOFSUSY
      COMMON/OLDFASH_HDEC/NFGG
      COMMON/WIDTHSM_HDEC/SMBRB,SMBRL,SMBRM,SMBRS,SMBRC,SMBRT,SMBRG,
     .               SMBRGA,SMBRZGA,SMBRW,SMBRZ,SMWDTH
      COMMON/WIDTHA_HDEC/ABRB,ABRL,ABRM,ABRS,ABRC,ABRT,ABRG,ABRGA,
     .              ABRZGA,ABRZ,AWDTH
      COMMON/WIDTHHL_HDEC/HLBRB,HLBRL,HLBRM,HLBRS,HLBRC,HLBRT,HLBRG,
     .               HLBRGA,HLBRZGA,HLBRW,HLBRZ,HLBRA,HLBRAZ,HLBRHW,
     .               HLWDTH
      COMMON/WIDTHHH_HDEC/HHBRB,HHBRL,HHBRM,HHBRS,HHBRC,HHBRT,HHBRG,
     .               HHBRGA,HHBRZGA,HHBRW,HHBRZ,HHBRH,HHBRA,HHBRAZ,
     .               HHBRHW,HHWDTH
      COMMON/WIDTHHC_HDEC/HCBRB,HCBRL,HCBRM,HCBRBU,HCBRS,HCBRC,HCBRT,
     .               HCBRW,HCBRA,HCWDTH
      COMMON/WISUSY_HDEC/HLBRSC,HLBRSN,HHBRSC,HHBRSN,HABRSC,HABRSN,
     .              HCBRSU,HLBRCHT,HHBRCHT,HABRCHT,HLBRNET,HHBRNET,
     .              HABRNET,HCBRCNT,HLBRSL,HHBRSL,HCBRSL,HABRSL,HABRST,
     .              HABRSB,HHBRSQ,HHBRST,HHBRSB,HHBRSQT,HCBRSQ,HCBRSTB,
     .              HCBRSQT,HLBRSQ,HLBRSQT
      COMMON/WISFER_HDEC/BHLSLNL,BHLSLEL,BHLSLER,BHLSQUL,BHLSQUR,
     .              BHLSQDL,BHLSQDR,BHLST(2,2),BHLSB(2,2),BHLSTAU(2,2),
     .              BHHSLNL,BHHSLEL,BHHSLER,BHHSQUL,BHHSQUR,BHHSQDL,
     .              BHHSQDR,BHHST(2,2),BHHSB(2,2),BHHSTAU(2,2),
     .              BHASTAU,BHASB,BHAST,
     .              BHCSL00,BHCSL11,BHCSL21,BHCSQ,BHCSTB(2,2)
      COMMON/SMASS_HDEC/GMN,XMN,GMC,GMST,GMSB,GMSL,GMSU,GMSD,GMSE,GMSN 
     .                 ,GMSN1
      COMMON/GOLDST_HDEC/AXMPL,AXMGD,IGOLD
      COMMON/WIGOLD_HDEC/HLBRGD,HABRGD,HHBRGD,HCBRGD
      COMMON/FLAGS_HDEC/INDIDEC
      COMMON/CPSM_HDEC/CPW,CPZ,CPTAU,CPMU,CPT,CPB,CPC,CPS,
     .                 CPGAGA,CPGG,CPZGA,ICOUPELW
      COMMON/CPSM4_HDEC/CPTP,CPBP,CPNUP,CPEP
c MMM changed 21/8/13
      COMMON/THDM_HDEC/TGBET2HDM,ALPH2HDM,AMHL2HDM,AMHH2HDM,
     .     AMHA2HDM,AMHC2HDM,AM12SQ,A1LAM2HDM,A2LAM2HDM,A3LAM2HDM,
     .     A4LAM2HDM,A5LAM2HDM,ITYPE2HDM,I2HDM,IPARAM2HDM
      COMMON/WIDTH_HC_ADD/hcbrcd,hcbrts,hcbrtd
      COMMON/WIDTH_2HDM/hcbrwhh,hhbrchch,hlbrchch,abrhhaz,abrhawphm
c end MMM changed 21/8/13
      COMMON/HMSSM_HDEC/AMHL10

      unlikely = -123456789D0
c     unlikely = 0.D0

      PI = 4*DATAN(1D0)

      OPEN(NI,FILE='./hdecay/hdecay.in')
      OPEN(NPAR,FILE='./hdecay/br.input')

      read(ni,101)islhai
      read(ni,101)islhao
      READ(NI,101)ICOUPVAR
      READ(NI,101)IHIGGS
      READ(NI,101)ISM4
      READ(NI,101)IFERMPHOB
c MMM changed 21/8/13
      READ(NI,101)I2HDM
c end MMM changed 21/8/13
      READ(NI,101)IMODEL
      READ(NI,100)TGBET
      READ(NI,100)AMABEG
      READ(NI,100)AMAEND
      READ(NI,101)NMA
c-------------------------- hMSSM ------------------
      READ(NI,*)
      READ(NI,100)AMHL10
      READ(NI,*)
c---------------------------------------------------
      READ(NI,100)ALSMZ
      READ(NI,100)AMS
      READ(NI,100)AMC
      READ(NI,100)AMB
      READ(NI,100)AMT
      READ(NI,100)AMTAU
      READ(NI,100)AMMUON
      READ(NI,100)ALPH
      READ(NI,100)GF
      READ(NI,100)GAMW
      READ(NI,100)GAMZ
      READ(NI,100)AMZ
      READ(NI,100)AMW
      READ(NI,100)VTB
      READ(NI,100)VTS
      READ(NI,100)VTD
      READ(NI,100)VCB
      READ(NI,100)VCS
      READ(NI,100)VCD
      READ(NI,100)VUB
      READ(NI,100)VUS
      READ(NI,100)VUD
      READ(NI,*)
      READ(NI,*)
      READ(NI,*)
      READ(NI,*)
      READ(NI,*)
      READ(NI,101)IGGELW
      READ(NI,100)AMTP
      READ(NI,100)AMBP
      READ(NI,100)AMNUP
      READ(NI,100)AMEP
c MMM changed 21/8/13
      READ(NI,*)
      READ(NI,*)
      READ(NI,*)
      READ(NI,*)
      READ(NI,101)IPARAM2HDM 
      READ(NI,101)ITYPE2HDM 
      READ(NI,*)
      READ(NI,100)TGBET2HDM
      READ(NI,100)AM12SQ
      READ(NI,*)
      READ(NI,100)ALPH2HDM
      READ(NI,100)AMHL2HDM
      READ(NI,100)AMHH2HDM
      READ(NI,100)AMHA2HDM
      READ(NI,100)AMHC2HDM
      READ(NI,*)
      READ(NI,100)A1LAM2HDM
      READ(NI,100)A2LAM2HDM
      READ(NI,100)A3LAM2HDM
      READ(NI,100)A4LAM2HDM
      READ(NI,100)A5LAM2HDM
c end MMM changed 2178/13
      READ(NI,*)
      READ(NI,100)SUSYSCALE
      READ(NI,100)AMU
      READ(NI,100)AM2
      READ(NI,100)AMGLUINO
      READ(NI,100)AMEL1
      READ(NI,100)AMER1
      READ(NI,100)AMQL1
      READ(NI,100)AMUR1
      READ(NI,100)AMDR1
      READ(NI,100)AMEL
      READ(NI,100)AMER
      READ(NI,100)AMSQ
      READ(NI,100)AMUR
      READ(NI,100)AMDR
      READ(NI,100)AL
      READ(NI,100)AU
      READ(NI,100)AD
      READ(NI,101)NNLO
      READ(NI,101)IONSH
      READ(NI,101)IONWZ
      READ(NI,101)IPOLE
      READ(NI,101)IOFSUSY
      READ(NI,101)INDIDEC
      READ(NI,101)NFGG
      READ(NI,101)IGOLD
      READ(NI,100)AXMPL
      READ(NI,100)AXMGD
      READ(NI,*)
      READ(NI,101)ICOUPELW
      READ(NI,100)CPW
      READ(NI,100)CPZ
      READ(NI,100)CPTAU
      READ(NI,100)CPMU
      READ(NI,100)CPT
      READ(NI,100)CPB
      READ(NI,100)CPC
      READ(NI,100)CPS
      READ(NI,100)CPGAGA
      READ(NI,100)CPGG
      READ(NI,100)CPZGA
      READ(NI,*)
      READ(NI,100)CPTP
      READ(NI,100)CPBP
      READ(NI,100)CPNUP
      READ(NI,100)CPEP

c MMM changed 21/8/13
      if(ihiggs.eq.0.and.i2hdm.eq.1) then
         print*,'You have chosen ihiggs=0 and i2hdm=1. For i2hdm=1 the d
     .efault value ihiggs=5 is set. If you want to calculate the SM bran
     .ching ratios, you have to set ihiggs=0 and i2hdm=0.'
      endif
      
      if(i2hdm.ge.2) then
         print*,'You have to set i2hdm = 0 (no 2HDM) or 1 (2HDM). The de
     .fault value i2hdm=0 is set otherwise.'
         i2hdm=0
      endif

      if(i2hdm.eq.1) then 
         iofsusy = 1
         ihiggs = 5
         tgbet  = tgbet2hdm
         islhai = 0
         islhao = 0
      endif
c end MMM changed 21/8/13

c--hMSSM?
      if(imodel.eq.10) then 
         iofsusy = 1
      endif

      IF(ICOUPVAR.EQ.0)THEN
       CPW    = 1
       CPZ    = 1
       CPTAU  = 1
       CPMU   = 1
       CPT    = 1
       CPB    = 1
       CPC    = 1
       CPS    = 1
       CPGAGA = 0
       CPGG   = 0
       CPZGA  = 0
       CPTP   = 1
       CPBP   = 1
       CPNUP  = 1
       CPEP   = 1
      ENDIF

      scaleofewsb = SUSYSCALE
      do i=1,20
       qvalue(i) = SUSYSCALE
      enddo
      qvalue(2) = 0

      cw2calc = amw**2/amz**2
      sw2calc = 1-cw2calc
      cwcalc  = dsqrt(cw2calc)
      swcalc  = dsqrt(sw2calc)
      vewsb = 1.D0/dsqrt(dsqrt(2.D0)*gf)
      g2ew = 2*amw/vewsb
      g1ew = g2ew*swcalc/cwcalc

c      if(i2hdm.eq.1) then 
c         itest = 1
c         CALL SUSYCP_HDEC(TGBET)
c         itest = 0
c      endif

c -- initialization of the check array --
      do i1=1,22,1
         check(i1) = 0
      end do

      if(islhai.eq.1) then
         open(ninlha,file='slha.in')
         call SLHA_read_leshouches_HDEC(ninlha)

c -- G_F --
         if(smval(2).ne.0.D0) then
            GF = smval(2)
         endif
c -- the strong coupling constant alphas_MSbar at the scale MZ --
         if(smval(3).ne.0.D0) then
            alsmz = smval(3)
         endif
         alphasmzms = alsmz
c -- Z pole mass --
         if(smval(4).ne.0.D0) then
            AMZ = smval(4)
         endif
c -- W pole mass --
         if(massval(1).ne.0.D0) then
            AMW = massval(1)
         endif
c -- the MSbar couplings g1,g2 at the scale Q --
         if(gaugeval(1).ne.0.D0) then
            g1ew  = gaugeval(1)
         endif
         if(gaugeval(2).ne.0.D0) then
            g2ew  = gaugeval(2)*(1-gaugeval(2)**2/96/pi**2*2)
         endif
         cw2calc = amw**2/amz**2
         sw2calc = 1-cw2calc
         cwcalc  = dsqrt(cw2calc)
         swcalc  = dsqrt(sw2calc)
c -- in case the gauge couplings are not given at the scale Q --
         if(gaugeval(1).eq.0.D0.or.gaugeval(2).eq.0.D0) then
c -- v at the scale Q --         
          if(smval(2).eq.0.D0.and.hmixval(3).ne.unlikely) then
           vewsb = hmixval(3)
           gf = 1/dsqrt(2.D0)/vewsb**2
          else
           vewsb = 1.D0/dsqrt(dsqrt(2.D0)*gf)
          endif
          g2ew = 2*amw/vewsb
          g1ew = g2ew*swcalc/cwcalc
         endif
c -- the scale Q at which the couplings are given --
         if(extval(0).ne.unlikely.and.extval(0).ne.-1.D0) then
            scaleofewsb = extval(0)
            ewsbscale = scaleofewsb
         else
            qvalsum = 0.D0
            isum    = 0
            do i=1,20,1
               qvalsum = qvalsum + qvalue(i)
               if(qvalue(i).ne.0.D0) then
                  isum = isum + 1
               endif
            end do

            if(isum.ne.0) then
               scaleofewsb = qvalsum/dble(isum)
            else
               warning(1) = 1.D0
            endif
            ewsbscale   = scaleofewsb
         endif
         SUSYSCALE = scaleofewsb

c -- CKM mixing matrix --

      if(vckmval(1).ne.unlikely) then
       vus = vckmval(1)
       vcb = vckmval(1)**2*vckmval(2)
       vub = vckmval(1)**3*vckmval(2)*dsqrt(vckmval(3)**2+vckmval(4)**2)
       rvub= vub/vcb
      endif

c -- neutralino and chargino masses --      

         slhaneut(1) =dabs(massval(28))
         slhaneut(2) =dabs(massval(29))
         slhaneut(3) =dabs(massval(30))
         slhaneut(4) =dabs(massval(31))
         slhaxneut(1)=massval(28)
         slhaxneut(2)=massval(29)
         slhaxneut(3)=massval(30)
         slhaxneut(4)=massval(31)
         slhachar(1) =dabs(massval(32))
         slhachar(2) =dabs(massval(33))
         slhaxchar(1) =massval(32)
         slhaxchar(2) =massval(33)

c -- the chargino and neutralino mixing matrix elements --
      do i=1,2,1
         do j=1,2,1
            slhau(i,j)=umixval(i,j)
            slhav(i,j)=vmixval(i,j)
         end do
      end do
      do i=1,4,1
         do j=1,4,1
            slhaz(i,j)=nmixval(i,j)
         end do
      end do

c -- sfermion masses --

      slhast(1) = massval(16)
      slhast(2) = massval(17)
      slhasb(1) = massval(14)
      slhasb(2) = massval(15)

      slhasu(1) = massval(8)
      slhasu(2) = massval(9)
      slhasd(1) = massval(10)
      slhasd(2) = massval(11)

      slhase(1) = massval(18)
      slhase(2) = massval(19)
      slhasl(1) = massval(24)
      slhasl(2) = massval(25)

      slhasnl(1) = massval(20)
      slhasnl(2) = 1.D15
      slhasn(1) = massval(26)
      slhasn(2) = 1.D15

c -- the sfermion mixing angles --

      slhacot=stopmixval(1,1)
      slhasit=stopmixval(1,2)

      slhacob=sbotmixval(1,1)
      slhasib=sbotmixval(1,2)

      slhacol=staumixval(1,1)
      slhasil=staumixval(1,2)

c -- the gluino mass --

      AMGLUINO = massval(27)

c -- the Higgs masses --

      slhaml  = massval(2)
      slhamh  = massval(3)
      slhamc  = massval(5)

      if(massval(4).ne.0.D0) then
         slhama = massval(4)
      elseif(extval(26).ne.unlikely) then
         slhama = extval(26)
      elseif(extval(24).ne.unlikely) then
         slhama = dsqrt(extval(24))
      endif

      amabeg = slhama
      amaend = slhama
      nma    = 1

c -- the MSSM mixing angle alpha in the Higgs sector --
c -- Attention: It might be that alphaval is not the DRbar value at
c -- the scale Q.
      slha_alpha = alphaval

c -- the fermion pole masses --

      if(smval(6).ne.0.D0) then
         AMT = smval(6)
      endif
      if(smval(7).ne.0.D0) then
         AMTAU = smval(7)
      endif

c -- the mass mb(mb)_MSbar --
      if(smval(5).ne.0.D0) then
         mbmsbar = smval(5)
      endif

      fmt = amt
      fmtau = amtau
      fms = ams
      fmc = amc
c -- calculation of the mb pole mass from mb(mb)_MSbar --
      if(smval(5).ne.0.D0) then
       del = 1.d-10
       mbl = mbmsbar
       mbu = 2*mbmsbar
       fmb = (mbl+mbu)/2
       amsb = ams
       amc0=amc
       amt0=amt
       acc=1.d-8
       nloop=3
c      nloop=2
11     amb=fmb
       amb0=amb
       xlambda=xitla_hdec(nloop,alsmz,acc)
       n0=5
       call alsini_hdec(acc)
c      xmb = runm_hdec(fmb,5)
       xmb = runm_hdec(mbmsbar,5)
       if(xmb.eq.mbmsbar)then
        mbl = fmb
        mbu = fmb
       elseif(xmb.gt.mbmsbar)then
        mbu = fmb
       else
        mbl = fmb
       endif
       fmb = (mbl+mbu)/2
       if(dabs(xmb/mbmsbar-1).gt.del) goto 11
      endif
      amb = fmb

c -- DRbar value of tanbeta at the scale Q --

      if(hmixval(2).ne.unlikely) then
         TGBET = hmixval(2)
      endif

c -- If no DRbar value at the scale Q has been given for tanbeta --

      if(hmixval(2).eq.unlikely) then
         if(extval(25).ne.0.D0.and.extval(25).ne.unlikely) then
            TGBET = extval(25)
         elseif(minval(3).ne.0.D0.and.minval(3).ne.unlikely) then
            TGBET = minval(3)
         endif
      endif

c -- The soft SUSY breaking parameters: DRbar values at the scale Q --

      do i=1,100,1
         m_softval(i) = unlikely
      end do

      do i=1,99,1
         if(msoftval(i).ne.unlikely) then
            m_softval(i) = msoftval(i)
            m_softcom(i) = softcom(i)
         elseif(extval(i).ne.unlikely) then
            m_softval(i) = extval(i)
            m_softcom(i) = extcom(i)
         endif
      end do

      if(auval(3,3).ne.unlikely) then
       AU=auval(3,3)
      endif
      if(adval(3,3).ne.unlikely) then
       AD=adval(3,3)
      endif
      if(aeval(3,3).ne.unlikely) then
       AL=aeval(3,3)
      endif

c The mixing parameter mu in the MS_bar scheme
      if(hmixval(1).ne.unlikely) then
       amudrbar = hmixval(1)
      else
       amudrbar = extval(23)
      endif
      if(amudrbar.ne.unlikely)then
       AMU = amudrbar*(1.D0+g1ew**2/16.D0/pi**2*3.D0/5.D0+
     .                 g2ew**2/16.D0/pi**2*3.D0/4.D0)
      endif


c The soft SUSY breaking parameters M1, M2 in the MS_bar scheme 
      am1msbar = m_softval(1)*(1.D0+g1ew**2/16.D0/pi**2*0.D0)
      am2msbar = m_softval(2)*(1.D0+g2ew**2/16.D0/pi**2*2.D0)

      if(am1msbar.ne.0.d0)am1 = am1msbar
      if(am2msbar.ne.0.d0)AM2 = am2msbar

      if(m_softval(31).ne.unlikely.and.m_softval(32).ne.unlikely) then
         if(m_softval(31).ne.0.D0.and.m_softval(32).ne.0.D0) then
            AMEL1 = (m_softval(31)+m_softval(32))/2.D0
         elseif(m_softval(31).ne.0.D0) then
            AMEL1 = m_softval(31)
         elseif(m_softval(32).ne.0.D0) then
            AMEL1 = m_softval(32)
         else
            AMEL1 = 0.D0
         endif
      elseif(m_softval(31).ne.unlikely) then
         AMEL1 = m_softval(31)
      elseif(m_softval(32).ne.unlikely) then
         AMEL1 = m_softval(32)
      endif

      if(m_softval(34).ne.unlikely.and.m_softval(35).ne.unlikely) then
         if(m_softval(34).ne.0.D0.and.m_softval(35).ne.0.D0) then
            AMER1 = (m_softval(34)+m_softval(35))/2.D0
         elseif(m_softval(34).ne.0.D0) then
            AMER1 = m_softval(34)
         elseif(m_softval(35).ne.0.D0) then
            AMER1 = m_softval(35)
         else
            AMER1 = 0.D0
         endif
      elseif(m_softval(34).ne.unlikely) then
         AMER1 = m_softval(34)
      elseif(m_softval(35).ne.unlikely) then
         AMER1 = m_softval(35)
      endif

      if(m_softval(41).ne.unlikely.and.m_softval(42).ne.unlikely) then
         if(m_softval(41).ne.0.D0.and.m_softval(42).ne.0.D0) then
            AMQL1 = (m_softval(41)+m_softval(42))/2.D0
         elseif(m_softval(41).ne.0.D0) then
            AMQL1 = m_softval(41)
         elseif(m_softval(42).ne.0.D0) then
            AMQL1 = m_softval(42)
         else
            AMQL1 = 0.D0
         endif
      elseif(m_softval(41).ne.unlikely) then
         AMQL1 = m_softval(41)
      elseif(m_softval(42).ne.unlikely) then
         AMQL1 = m_softval(42)
      endif

      if(m_softval(44).ne.unlikely.and.m_softval(45).ne.unlikely) then
         if(m_softval(44).ne.0.D0.and.m_softval(45).ne.0.D0) then
            AMUR1 = (m_softval(44)+m_softval(45))/2.D0
         elseif(m_softval(44).ne.0.D0) then
            AMUR1 = m_softval(44)
         elseif(m_softval(45).ne.0.D0) then
            AMUR1 = m_softval(45)
         else
            AMUR1 = 0.D0
         endif
      elseif(m_softval(44).ne.unlikely) then
         AMUR1 = m_softval(44)
      elseif(m_softval(45).ne.unlikely) then
         AMUR1 = m_softval(45)
      endif

      if(m_softval(47).ne.unlikely.and.m_softval(48).ne.unlikely) then
         if(m_softval(47).ne.0.D0.and.m_softval(48).ne.0.D0) then
            AMDR1 = (m_softval(47)+m_softval(48))/2.D0
         elseif(m_softval(47).ne.0.D0) then
            AMDR1 = m_softval(47)
         elseif(m_softval(48).ne.0.D0) then
            AMDR1 = m_softval(48)
         else
            AMDR1 = 0.D0
         endif
      elseif(m_softval(47).ne.unlikely) then
         AMDR1 = m_softval(47)
      elseif(m_softval(48).ne.unlikely) then
         AMDR1 = m_softval(48)
      endif

      if(m_softval(33).ne.unlikely) then
         AMEL = m_softval(33)
      endif
      if(m_softval(36).ne.unlikely) then
         AMER = m_softval(36)
      endif
      if(m_softval(43).ne.unlikely) then
         AMSQ = m_softval(43)
      endif
      if(m_softval(46).ne.unlikely) then
         AMUR = m_softval(46)
      endif
      if(m_softval(49).ne.unlikely) then
         AMDR = m_softval(49)
      endif

      if(scaleofewsb.eq.0.D0)then
       scaleofewsb = (4*AMQL1+2*AMUR1+2*AMDR1+2*AMSQ+AMUR+AMDR)/12
       ewsbscale   = scaleofewsb
       SUSYSCALE = scaleofewsb
      endif

      endif

c     write(6,*)'susyscale = ',SUSYSCALE

      IF(IMODEL.EQ.3)THEN
       WRITE(6,*)'MU (UP TO THE SIGN) WILL BE IDENTIFIED WITH M_SQ...'
      ENDIF

      B = DATAN(TGBET)
      AMGLU = AMGLUINO

C     VUB=RVUB*VCB
      ALPH=1.D0/ALPH
      AMSB = AMS

      AMC0=AMC
      AMB0=AMB
      AMT0=AMT
      ACC=1.D-8
      NLOOP=3
c     NLOOP=2
      XLAMBDA=XITLA_HDEC(NLOOP,ALSMZ,ACC)
      N0=5
      CALL ALSINI_HDEC(ACC)
C--DECOUPLING THE TOP QUARK FROM ALPHAS
      AMT0=3.D8
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     QQ = 360.D0
c     X3 = ALPHAS_HDEC(AMZ,3)
c     Y3 = ALPHAS_HDEC(QQ,3)
c     write(6,*)'alpha_s: ',QQ,Y3,X3
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     QQ = 1000.D0
c     X3 = ALPHAS_HDEC(AMZ,3)
c     Y3 = ALPHAS_HDEC(QQ,3)
c     NLOOP=1
c     XLAMBDA=XITLA_HDEC(NLOOP,ALSMZ,ACC)
c     CALL ALSINI_HDEC(ACC)
c     X1 = ALPHAS_HDEC(AMZ,1)
c     Y1 = ALPHAS_HDEC(QQ,1)
c     NLOOP=2
c     XLAMBDA=XITLA_HDEC(NLOOP,ALSMZ,ACC)
c     CALL ALSINI_HDEC(ACC)
c     X2 = ALPHAS_HDEC(AMZ,2)
c     Y2 = ALPHAS_HDEC(QQ,2)
c     write(6,*)'  LO: ',X1,Y1,Y1/Y3,Y3/Y1
c     write(6,*)' NLO: ',X2,Y2,Y2/Y3,Y3/Y2
c     write(6,*)'NNLO: ',AMZ,QQ,XLAMBDA,X3,Y3
c     NLOOP=3
c     XLAMBDA=XITLA_HDEC(NLOOP,ALSMZ,ACC)
c     CALL ALSINI_HDEC(ACC)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

C--INITIALIZE COEFFICIENTS FOR POLYLOGARITHMS
      NBER = 18
      CALL BERNINI_HDEC(NBER)

C--CHECK NFGG
      IF(NFGG.GT.5.OR.NFGG.LT.3)THEN
       WRITE(6,*)'NF-GG NOT VALID. TAKING THE DEFAULT NF-GG = 3....'
       NFGG = 3
      ENDIF

C--CHECK 4TH GENERATION
      IF(ISM4.NE.0.AND.IHIGGS.NE.0)THEN
       WRITE(6,*)'4TH GENERATION. TAKING DEFAULT HIGGS = 0....'
       IHIGGS = 0
      ENDIF
      IF(ISM4.NE.0.AND.IFERMPHOB.NE.0)THEN
       WRITE(6,*)'4TH GENERATION. TAKING DEFAULT FERMPHOB = 0....'
       IFERMPHOB = 0
      ENDIF
      XXCP = 0
      IF(CPW.NE.1.D0.OR.CPZ.NE.1.D0.OR.CPTAU.NE.1.D0.OR.CPMU.NE.1.D0
     .  .OR.CPT.NE.1.D0.OR.CPB.NE.1.D0.OR.CPC.NE.1.D0.OR.CPS.NE.1.D0)
     .  XXCP = 1
c     IF(ISM4.NE.0)THEN
c      CPW   = 1
c      CPZ   = 1
c      CPTAU = 1
c      CPMU  = 1
c      CPT   = 1
c      CPB   = 1
c      CPC   = 1
c      CPS   = 1
c     ENDIF
c     IF(ISM4.NE.0.AND.XXCP.NE.0)THEN
c      WRITE(6,*)'4TH GENERATION. TAKING DEFAULT COUPINGS = 1....'
c     ENDIF

C--CHECK FERMIOPHOBIC
      IF(IFERMPHOB.NE.0.AND.IHIGGS.NE.0)THEN
       WRITE(6,*)'FERMIOPHOBIC HIGGS. TAKING DEFAULT HIGGS = 0....'
       IHIGGS = 0
      ENDIF

100   FORMAT(10X,G30.20)
101   FORMAT(10X,I30)

C--WRITE THE INPUT PARAMTERS TO A DATA-FILE

      WRITE(NPAR,8)'SLHAIN   = ',ISLHAI
      WRITE(NPAR,8)'SLHAOUT  = ',ISLHAO
      WRITE(NPAR,8)'COUPVAR  = ',ICOUPVAR
      WRITE(NPAR,8)'HIGGS    = ',IHIGGS
      WRITE(NPAR,8)'SM4      = ',ISM4
      WRITE(NPAR,8)'FERMPHOB = ',IFERMPHOB
c MMM changed 21/8/2013
      WRITE(NPAR,8)'2HDM     = ',I2HDM
c end MMM changed 21/8/2013
      WRITE(NPAR,8)'MODEL    = ',IMODEL
      WRITE(NPAR,9)'TGBET    = ',TGBET
      WRITE(NPAR,9)'MABEG    = ',AMABEG
      WRITE(NPAR,9)'MAEND    = ',AMAEND
      WRITE(NPAR,7)'NMA      = ',NMA
      WRITE(NPAR,9)'ALS(MZ)  = ',ALSMZ
      WRITE(NPAR,9)'MSBAR(1) = ',AMS
      WRITE(NPAR,9)'MC       = ',AMC
      WRITE(NPAR,9)'MB       = ',AMB
      WRITE(NPAR,9)'MT       = ',AMT
      WRITE(NPAR,9)'MTAU     = ',AMTAU
      WRITE(NPAR,9)'MMUON    = ',AMMUON
      WRITE(NPAR,9)'ALPH     = ',1.D0/ALPH
      WRITE(NPAR,9)'GF       = ',GF
      WRITE(NPAR,9)'GAMW     = ',GAMW
      WRITE(NPAR,9)'GAMZ     = ',GAMZ
      WRITE(NPAR,9)'MZ       = ',AMZ
      WRITE(NPAR,9)'MW       = ',AMW
      WRITE(NPAR,9)'VTB      = ',VTB
      WRITE(NPAR,9)'VTS      = ',VTS
      WRITE(NPAR,9)'VTD      = ',VTD
      WRITE(NPAR,9)'VCB      = ',VCB
      WRITE(NPAR,9)'VCS      = ',VCS
      WRITE(NPAR,9)'VCD      = ',VCD
      WRITE(NPAR,9)'VUB      = ',VUB
      WRITE(NPAR,9)'VUS      = ',VUS
      WRITE(NPAR,9)'VUD      = ',VUD
      WRITE(NPAR,20)'********************* 4TH GENERATION **************
     .************************'
      WRITE(NPAR,*)'  SCENARIO FOR ELW. CORRECTIONS TO H -> GG',
     .             ' (EVERYTHING IN GEV):'
      WRITE(NPAR,*)'  GG_ELW = 1: MTP = 500    MBP = 450    ',
     .             'MNUP = 375    MEP = 450'
      WRITE(NPAR,*)'  GG_ELW = 2: MBP = MNUP = MEP = 600    ',
     .             'MTP = MBP+50*(1+LOG(M_H/115)/5)'
      WRITE(NPAR,*)
      WRITE(NPAR,8)'GG_ELW   = ',IGGELW
      WRITE(NPAR,9)'MTP      = ',AMTP
      WRITE(NPAR,9)'MBP      = ',AMBP
      WRITE(NPAR,9)'MNUP     = ',AMNUP
      WRITE(NPAR,9)'MEP      = ',AMEP
c MMM changed 21/8/2013
      WRITE(NPAR,20)'************************** 2HDM *******************
     .************************'
      WRITE(NPAR,*)
     .          'TYPE: 1 (I), 2 (II), 3 (Lepton-specific), 4 (flipped)'
      WRITE(NPAR,8)'2HDM TYPE= ',ITYPE2HDM
      WRITE(NPAR,9)'TANBETA  = ',TGBET2HDM
      WRITE(NPAR,9)'ALPHA_H  = ',ALPH2HDM
      WRITE(NPAR,9)'M_h      = ',AMHL2HDM
      WRITE(NPAR,9)'M_H      = ',AMHH2HDM
      WRITE(NPAR,9)'M_A      = ',AMHA2HDM
      WRITE(NPAR,9)'M_H+     = ',AMHC2HDM
      WRITE(NPAR,9)'M_12^2   = ',AM12SQ
      WRITE(NPAR,20)'***************************************************
     .************************'
c end MMM changed 21/8/2013
      WRITE(NPAR,9)'SUSYSCALE= ',SUSYSCALE
      WRITE(NPAR,9)'MU       = ',AMU
      WRITE(NPAR,9)'M2       = ',AM2
      WRITE(NPAR,9)'MGLUINO  = ',AMGLUINO
      WRITE(NPAR,9)'MSL1     = ',AMEL1
      WRITE(NPAR,9)'MER1     = ',AMER1
      WRITE(NPAR,9)'MQL1     = ',AMQL1
      WRITE(NPAR,9)'MUR1     = ',AMUR1
      WRITE(NPAR,9)'MDR1     = ',AMDR1
      WRITE(NPAR,9)'MSL      = ',AMEL
      WRITE(NPAR,9)'MER      = ',AMER
      WRITE(NPAR,9)'MSQ      = ',AMSQ
      WRITE(NPAR,9)'MUR      = ',AMUR
      WRITE(NPAR,9)'MDR      = ',AMDR
      WRITE(NPAR,9)'AL       = ',AL
      WRITE(NPAR,9)'AU       = ',AU
      WRITE(NPAR,9)'AD       = ',AD
      WRITE(NPAR,8)'NNLO (M) = ',NNLO
      WRITE(NPAR,8)'ON-SHELL = ',IONSH
      WRITE(NPAR,8)'ON-SH-WZ = ',IONWZ
      WRITE(NPAR,8)'IPOLE    = ',IPOLE 
      WRITE(NPAR,8)'OFF-SUSY = ',IOFSUSY
      WRITE(NPAR,8)'INDIDEC  = ',INDIDEC
      WRITE(NPAR,8)'NF-GG    = ',NFGG
      WRITE(NPAR,8)'IGOLD    = ',IGOLD
      WRITE(NPAR,9)'MPLANCK  = ',AXMPL
      WRITE(NPAR,9)'MGOLD    = ',AXMGD
      WRITE(NPAR,20)'******************* VARIATION OF HIGGS COUPLINGS *
     .************************'
      WRITE(NPAR,8)'ELWK     = ',ICOUPELW
      WRITE(NPAR,9)'CW       = ',CPW
      WRITE(NPAR,9)'CZ       = ',CPZ
      WRITE(NPAR,9)'Ctau     = ',CPTAU
      WRITE(NPAR,9)'Cmu      = ',CPMU
      WRITE(NPAR,9)'Ct       = ',CPT
      WRITE(NPAR,9)'Cb       = ',CPB
      WRITE(NPAR,9)'Cc       = ',CPC
      WRITE(NPAR,9)'Cs       = ',CPS
      WRITE(NPAR,9)'Cgaga    = ',CPGAGA
      WRITE(NPAR,9)'Cgg      = ',CPGG
      WRITE(NPAR,9)'CZga     = ',CPZGA
      WRITE(NPAR,20)'********************* 4TH GENERATION **************
     .************************'
      WRITE(NPAR,9)'Ctp      = ',CPTP
      WRITE(NPAR,9)'Cbp      = ',CPBP
      WRITE(NPAR,9)'Cnup     = ',CPNUP
      WRITE(NPAR,9)'Cep      = ',CPEP

C     WRITE(NPAR,9)'LAMBDA_5 = ',XLAMBDA

      CLOSE(NPAR)

7     FORMAT(A11,I7)
8     FORMAT(A11,I4)
9     FORMAT(A11,G15.8)
10    FORMAT(A26)
20    FORMAT(A74)

      CLOSE(NI)

      RETURN
      END

      SUBROUTINE HEAD_HDEC(TGBET,AMABEG)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER(K=6,NI=87,NSA=85,NSB=86,NSC=57,NLA=88,NLB=89,NHA=90,
     .          NHB=91,NHC=92,NAA=93,NAB=94,NCA=95,NCB=96,NCC=50,NRA=97,
     .          NRB=98,
     .          NSUSYL=81,NSUSYA=82,NSUSYH=83,NSUSYC=84,NPAR=80,NTA=99,
     .          NSUSYLA=79,NSUSYLB=78,NSUSYLC=77,NSUSYLD=76,NSUSYLE=75,
     .          NSUSYLF=59,NSUSYHF=58,
     .          NSUSYHA=74,NSUSYHB=73,NSUSYHC=72,NSUSYHD=71,NSUSYHE=70,
     .          NSUSYAA=69,NSUSYAB=68,NSUSYAC=67,NSUSYAD=66,NSUSYAE=65,
     .          NSUSYCA=64,NSUSYCB=63,NSUSYCC=62,NSUSYCD=61,NSUSYCE=60)
      PARAMETER(NAC=31,NLC=32)
      DIMENSION GMN(4),XMN(4),GMC(2),GMST(2),GMSB(2),GMSL(2),
     .          GMSU(2),GMSD(2),GMSE(2),GMSN(2),GMSN1(2)
      DIMENSION HLBRSC(2,2),HLBRSN(4,4),HHBRSC(2,2),HHBRSN(4,4),
     .          HABRSC(2,2),HABRSN(4,4),HCBRSU(2,4),
     .          HHBRST(2,2),HHBRSB(2,2),HCBRSTB(2,2) 
      DIMENSION AC1(2,2),AC2(2,2),AC3(2,2),
     .          AN1(4,4),AN2(4,4),AN3(4,4),
     .          ACNL(2,4),ACNR(2,4)
      DIMENSION GLTT(2,2),GLBB(2,2),GHTT(2,2),GHBB(2,2),GCTB(2,2),
     .          GLEE(2,2),GHEE(2,2),GCEN(2,2)
      DIMENSION AGDL(4),AGDA(4),AGDH(4),AGDC(2)
      COMMON/MASSES_HDEC/AMS,AMC,AMB,AMT
      COMMON/STRANGE_HDEC/AMSB
      COMMON/PARAM_HDEC/GF,ALPH,AMTAU,AMMUON,AMZ,AMW
      COMMON/CKMPAR_HDEC/VTB,VTS,VTD,VCB,VCS,VCD,VUB,VUS,VUD
      COMMON/HMASS_HDEC/AMSM,AMA,AML,AMH,AMCH,AMAR
      COMMON/BREAK_HDEC/AMEL,AMER,AMSQ,AMUR,AMDR,AL,AU,AD,AMU,AM2
      COMMON/BREAKGLU_HDEC/AMGLU
      COMMON/SFER1ST_HDEC/AMQL1,AMUR1,AMDR1,AMEL1,AMER1
      COMMON/GLUINO_HDEC/AMGLUINO,XMSB1,XMSB2,STHB,CTHB,
     .              XLBB(2,2),XHBB(2,2),XABB(2,2),
     .              XMST1,XMST2,STHT,CTHT,
     .              XLTT(2,2),XHTT(2,2),XATT(2,2)
      COMMON/WZWDTH_HDEC/GAMC0,GAMT0,GAMT1,GAMW,GAMZ
      COMMON/COUP_HDEC/GAT,GAB,GLT,GLB,GHT,GHB,GZAH,GZAL,
     .            GHHH,GLLL,GHLL,GLHH,GHAA,GLAA,GLVV,GHVV,
     .            GLPM,GHPM,B,A
      COMMON/ALS_HDEC/XLAMBDA,AMC0,AMB0,AMT0,N0
      COMMON/FLAG_HDEC/IHIGGS,NNLO,IPOLE
      COMMON/MODEL_HDEC/IMODEL
      COMMON/ONSHELL_HDEC/IONSH,IONWZ,IOFSUSY
      COMMON/OLDFASH_HDEC/NFGG
      COMMON/SM4_HDEC/AMTP,AMBP,AMNUP,AMEP,ISM4,IGGELW
      COMMON/WIDTHSM_HDEC/SMBRB,SMBRL,SMBRM,SMBRS,SMBRC,SMBRT,SMBRG,
     .               SMBRGA,SMBRZGA,SMBRW,SMBRZ,SMWDTH
      COMMON/WIDTHSM4_HDEC/SMBRNUP,SMBREP,SMBRBP,SMBRTP
      COMMON/WIDTHA_HDEC/ABRB,ABRL,ABRM,ABRS,ABRC,ABRT,ABRG,ABRGA,
     .              ABRZGA,ABRZ,AWDTH
      COMMON/WIDTHHL_HDEC/HLBRB,HLBRL,HLBRM,HLBRS,HLBRC,HLBRT,HLBRG,
     .               HLBRGA,HLBRZGA,HLBRW,HLBRZ,HLBRA,HLBRAZ,HLBRHW,
     .               HLWDTH
      COMMON/WIDTHHH_HDEC/HHBRB,HHBRL,HHBRM,HHBRS,HHBRC,HHBRT,HHBRG,
     .               HHBRGA,HHBRZGA,HHBRW,HHBRZ,HHBRH,HHBRA,HHBRAZ,
     .               HHBRHW,HHWDTH
      COMMON/WIDTHHC_HDEC/HCBRB,HCBRL,HCBRM,HCBRBU,HCBRS,HCBRC,HCBRT,
     .               HCBRW,HCBRA,HCWDTH
      COMMON/WISUSY_HDEC/HLBRSC,HLBRSN,HHBRSC,HHBRSN,HABRSC,HABRSN,
     .              HCBRSU,HLBRCHT,HHBRCHT,HABRCHT,HLBRNET,HHBRNET,
     .              HABRNET,HCBRCNT,HLBRSL,HHBRSL,HCBRSL,HABRSL,HABRST,
     .              HABRSB,HHBRSQ,HHBRST,HHBRSB,HHBRSQT,HCBRSQ,HCBRSTB,
     .              HCBRSQT,HLBRSQ,HLBRSQT
      COMMON/WISFER_HDEC/BHLSLNL,BHLSLEL,BHLSLER,BHLSQUL,BHLSQUR,
     .              BHLSQDL,BHLSQDR,BHLST(2,2),BHLSB(2,2),BHLSTAU(2,2),
     .              BHHSLNL,BHHSLEL,BHHSLER,BHHSQUL,BHHSQUR,BHHSQDL,
     .              BHHSQDR,BHHST(2,2),BHHSB(2,2),BHHSTAU(2,2),
     .              BHASTAU,BHASB,BHAST,
     .              BHCSL00,BHCSL11,BHCSL21,BHCSQ,BHCSTB(2,2)
      COMMON/SMASS_HDEC/GMN,XMN,GMC,GMST,GMSB,GMSL,GMSU,GMSD,GMSE,GMSN
     .                 ,GMSN1
      COMMON/GOLDST_HDEC/AXMPL,AXMGD,IGOLD
      COMMON/WIGOLD_HDEC/HLBRGD,HABRGD,HHBRGD,HCBRGD
      COMMON/FLAGS_HDEC/INDIDEC
c MMM changed 21/8/13
      COMMON/THDM_HDEC/TGBET2HDM,ALPH2HDM,AMHL2HDM,AMHH2HDM,
     .     AMHA2HDM,AMHC2HDM,AM12SQ,A1LAM2HDM,A2LAM2HDM,A3LAM2HDM,
     .     A4LAM2HDM,A5LAM2HDM,ITYPE2HDM,I2HDM,IPARAM2HDM
      COMMON/THDM_TEST/itest
      COMMON/WIDTH_HC_ADD/hcbrcd,hcbrts,hcbrtd
      COMMON/WIDTH_2HDM/hcbrwhh,hhbrchch,hlbrchch,abrhhaz,abrhawphm
c end MMM changed 21/8/13

      PI = 4*DATAN(1D0)

      IF(IHIGGS.EQ.0) THEN
       OPEN(NSA,FILE='br.sm1')
       OPEN(NSB,FILE='br.sm2')
       IF(ISM4.NE.0)THEN
        OPEN(NSC,FILE='br.sm3')
       ENDIF
      ENDIF
      IF(IHIGGS.NE.0) THEN
       OPEN(NTA,FILE='br.top')
      ENDIF
      IF(IHIGGS.EQ.1.OR.IHIGGS.EQ.5) THEN
c MMM changed 21/8/2013
       if(i2hdm.eq.0) then
       OPEN(NLA,FILE='br.l1')
       OPEN(NLB,FILE='br.l2')
       elseif(i2hdm.eq.1) then
       OPEN(NLA,FILE='br.l1_2HDM')
       OPEN(NLB,FILE='br.l2_2HDM')
       OPEN(NLC,FILE='br.l3_2HDM')
       endif
c end MMM changed 21/8/2013
      IF(IOFSUSY.EQ.0)THEN 
       OPEN(NSUSYL,FILE='br.ls')
       IF(INDIDEC.NE.0)THEN 
        OPEN(NSUSYLA,FILE='br.ls1')
        OPEN(NSUSYLB,FILE='br.ls2')
        OPEN(NSUSYLC,FILE='br.ls3')
        OPEN(NSUSYLD,FILE='br.ls4')
        OPEN(NSUSYLE,FILE='br.ls5')
        OPEN(NSUSYLF,FILE='br.ls6')
       ENDIF
      ENDIF
      ENDIF
      IF(IHIGGS.EQ.2.OR.IHIGGS.EQ.5) THEN
c MMM changed 21/8/2013
       if(i2hdm.eq.0) then
       OPEN(NHA,FILE='br.h1')
       OPEN(NHB,FILE='br.h2')
       OPEN(NHC,FILE='br.h3')
      elseif(i2hdm.eq.1) then
       OPEN(NHA,FILE='br.h1_2HDM')
       OPEN(NHB,FILE='br.h2_2HDM')
       OPEN(NHC,FILE='br.h3_2HDM')
      endif
c end MMM changed 21/8/2013
      IF(IOFSUSY.EQ.0)THEN 
       OPEN(NSUSYH,FILE='br.hs')
       IF(INDIDEC.NE.0)THEN 
        OPEN(NSUSYHA,FILE='br.hs1')
        OPEN(NSUSYHB,FILE='br.hs2')
        OPEN(NSUSYHC,FILE='br.hs3')
        OPEN(NSUSYHD,FILE='br.hs4')
        OPEN(NSUSYHE,FILE='br.hs5')
        OPEN(NSUSYHF,FILE='br.hs6')
       ENDIF
      ENDIF
      ENDIF
      IF(IHIGGS.EQ.3.OR.IHIGGS.EQ.5) THEN
c MMM changed 21/8/2013
       if(i2hdm.eq.0) then
       OPEN(NAA,FILE='br.a1')
       OPEN(NAB,FILE='br.a2')
      elseif(i2hdm.eq.1) then
       OPEN(NAA,FILE='br.a1_2HDM')
       OPEN(NAB,FILE='br.a2_2HDM')
       OPEN(NAC,FILE='br.a3_2HDM')
      endif
c end MMM changed 21/8/2013
      IF(IOFSUSY.EQ.0)THEN 
       OPEN(NSUSYA,FILE='br.as')
       IF(INDIDEC.NE.0)THEN 
        OPEN(NSUSYAA,FILE='br.as1')
        OPEN(NSUSYAB,FILE='br.as2')
        OPEN(NSUSYAC,FILE='br.as3')
        OPEN(NSUSYAD,FILE='br.as4')
       ENDIF
      ENDIF
      ENDIF
      IF(IHIGGS.EQ.4.OR.IHIGGS.EQ.5) THEN
c MMM changed 21/8/2013
       if(i2hdm.eq.0) then
       OPEN(NCA,FILE='br.c1')
       OPEN(NCB,FILE='br.c2')
       OPEN(NCC,FILE='br.c3')
      elseif(i2hdm.eq.1) then
       OPEN(NCA,FILE='br.c1_2HDM')
       OPEN(NCB,FILE='br.c2_2HDM')
       OPEN(NCC,FILE='br.c3_2HDM')
      endif
c end MMM changed 21/8/2013
      IF(IOFSUSY.EQ.0)THEN 
       OPEN(NSUSYC,FILE='br.cs')
       IF(INDIDEC.NE.0)THEN 
        OPEN(NSUSYCA,FILE='br.cs1')
        OPEN(NSUSYCB,FILE='br.cs2')
        OPEN(NSUSYCC,FILE='br.cs3')
        OPEN(NSUSYCD,FILE='br.cs4')
       ENDIF
      ENDIF
      ENDIF

C--SETUP THE HEADS OF THE TABLES IN THE DATA-FILES

      IF(IHIGGS.EQ.0) THEN
      WRITE(NSA,70)'MHSM  ','BB   ','TAU TAU','MU MU ','SS ','CC ','TT '
      WRITE(NSA,69)
      WRITE(NSA,*)
      WRITE(NSB,70)'MHSM  ','GG ','GAM GAM','Z GAM ','WW ','ZZ ','WIDTH'
      WRITE(NSB,69)
      WRITE(NSB,*)
      IF(ISM4.NE.0)THEN
      WRITE(NSC,70)'MHSM  ','NUP NUP','EP EP','BP BP ','TP TP'
      WRITE(NSC,69)
      WRITE(NSC,*)
      ENDIF
      ENDIF

      IF(IHIGGS.NE.0) THEN
      WRITE(NTA,73)'MHC   ','W+- B','H+- B','WIDTH'
      WRITE(NTA,69)
      WRITE(NTA,*)
      ENDIF

      IF(IHIGGS.EQ.1.OR.IHIGGS.EQ.5) THEN
      WRITE(NLA,70)'MHL   ','BB   ','TAU TAU','MU MU ','SS ','CC ','TT '
      WRITE(NLA,69)
      WRITE(NLA,*)
c MMM changed 23/8/2013
      if(i2hdm.eq.0) then
      WRITE(NLB,70)'MHL   ','GG ','GAM GAM','Z GAM ','WW ','ZZ ','WIDTH'
      WRITE(NLB,69)
      WRITE(NLB,*)
      elseif(i2hdm.eq.1) then
      WRITE(NLB,72)'MHL   ','GG ','GAM GAM','Z GAM ','WW ','ZZ '
      WRITE(NLB,69)
      WRITE(NLB,*)
      WRITE(NLC,72)'MHL   ','AA ','Z A ','W+- H-+','H+ H- ','WIDTH '
      WRITE(NLC,69)
      WRITE(NLC,*)
      endif
c end MMM changed 23/8/2013
      ENDIF

      IF(IHIGGS.EQ.2.OR.IHIGGS.EQ.5) THEN
      WRITE(NHA,70)'MHH   ','BB   ','TAU TAU','MU MU ','SS ','CC ','TT '
      WRITE(NHA,69)
      WRITE(NHA,*)
      WRITE(NHB,72)'MHH   ','GG ','GAM GAM','Z GAM ','WW ','ZZ '
      WRITE(NHB,69)
      WRITE(NHB,*)
      WRITE(NHC,70)'MHH   ','hh ','AA ','Z A ','W+- H-+','H+ H- ','WIDTH
     . '
      WRITE(NHC,69)
      WRITE(NHC,*)
      ENDIF

      IF(IHIGGS.EQ.3.OR.IHIGGS.EQ.5) THEN
      WRITE(NAA,70)'MHA   ','BB   ','TAU TAU','MU MU ','SS ','CC ','TT '
      WRITE(NAA,69)
      WRITE(NAA,*)
c MMM changed 23/8/2013
      if(i2hdm.eq.0) then
      WRITE(NAB,72)'MHA   ','GG ','GAM GAM','Z GAM ','Z HL ','WIDTH '
      WRITE(NAB,69)
      WRITE(NAB,*)
      elseif(i2hdm.eq.1) then
      WRITE(NAB,72)'MHA   ','GG ','GAM GAM','Z GAM ','Z h ','Z H'
      WRITE(NAB,69)
      WRITE(NAB,*)
      WRITE(NAC,73)'MHA   ','W+- H-+','WIDTH '
      WRITE(NAC,69)
      WRITE(NAC,*)
      endif
c end MMM changed 23/8/2013
      ENDIF

      IF(IHIGGS.EQ.4.OR.IHIGGS.EQ.5) THEN
      WRITE(NCA,70)'MHC   ','BC   ','TAU NU ','MU NU ','SU ','CS ','TB '
      WRITE(NCA,69)
      WRITE(NCA,*)
      WRITE(NCB,70)'MHC   ','CD   ','BU   ','TS   ','TD   '
      WRITE(NCB,69)
      WRITE(NCB,*)
c MMM changed 23/8/2013
      if(i2hdm.eq.0) then
      WRITE(NCC,70)'MHC   ','hW ','AW ','WIDTH '
      elseif(i2hdm.eq.1) then
      WRITE(NCC,70)'MHC   ','hW ','HW ','AW ','WIDTH '
      endif
c end MMM changed 23/8/2013
      WRITE(NCC,69)
      WRITE(NCC,*)
      ENDIF

69    FORMAT(79('_'))
70    FORMAT(A9,6(1X,A10))
71    FORMAT(A9,4(1X,A10))
72    FORMAT(A9,5(1X,A10))
73    FORMAT(A9,3(1X,A10))

      AMAR = AMABEG
      AMSM = AMAR
      AMA = AMAR

      IF(IHIGGS.NE.0)THEN 
C *******************************  SUSY OUTPUT 

       CALL GAUGINO_HDEC(AMU,AM2,B,A,GMC,GMN,XMN,AC1,AC2,AC3,
     .              AN1,AN2,AN3,ACNL,ACNR,AGDL,AGDA,AGDH,AGDC)
      TSC = (AMSQ+AMUR+AMDR)/3
      BSC = (AMSQ+AMUR+AMDR)/3
C--DECOUPLING THE TOP QUARK FROM ALPHAS
c     AMT00 = AMT0
      AMT0 = 3.D8
      CALL SFERMION_HDEC(TSC,BSC,AMSQ,AMUR,AMDR,AMEL,AMER,AL,AU,AD,AMU,
     .               GMST,GMSB,GMSL,GMSU,GMSD,GMSE,GMSN,GMSN1,
     .               GLEE,GLTT,GLBB,GHEE,GHTT,GHBB,
     .               GAEE,GATT,GABB,GCEN,GCTB)
c     AMT0 = AMT00
      itest = 1
      CALL SUSYCP_HDEC(TGBET)
      itest = 0
c     write(6,*)'MZ,MW,SW2,alpha: ',AMZ,AMW,1-AMW**2/AMZ**2,A
c     write(6,*)'tan(beta),Ab,mu: ',TGBET,AD,AMU
c     write(6,*)'M_A, M_h, M_H, M_H+: ',AMA,AML,AMH,AMCH
c     write(6,*)'Lambda_hhh/Lambda_SM: ',GLLL/AML**2*AMZ**2/3
c     write(6,*)
c     write(96,*)ama,aml,amh,amch
c     write(97,*)glb,glt,glvv,ghb,ght,ghvv
c     write(6,*)'M_A, M_h, M_H, sin(alpha): ',AMA,AML,AMH,DSIN(A)

      IF(IOFSUSY.EQ.0)THEN
C--WRITE THE GAUGINO MASSES/ TB, MU AND M2 IN THE SUSY DATA-FILE
C--WRITE THE SFERMION MASSES/ SUSY MASSES AND COUPLINGS IN SUSY DATA-FILE
C 
       IF(IHIGGS.EQ.1.OR.IHIGGS.EQ.5) THEN
       WRITE(NSUSYL,347) TGBET,AM2,AMU,AMSQ
       WRITE(NSUSYL,348) GMC(1),GMC(2),GMN(1),GMN(2),GMN(3),GMN(4)
       WRITE(NSUSYL,349) GMST(1),GMST(2),GMSU(1),GMSU(2)
       WRITE(NSUSYL,350) GMSB(1),GMSB(2),GMSD(1),GMSD(2)
       WRITE(NSUSYL,351)GMSL(1),GMSL(2),GMSN(1),GMSE(1),GMSE(2),GMSN1(1)
       WRITE(NSUSYL,*)
       WRITE(NSUSYL,*)'   MHL        CHARGINOS  NEUTRALS   '//
     . 'SLEPTONS   SQUARKS  GRAVITINO+GAUGINO'
       WRITE(NSUSYL,69)
       WRITE(NSUSYL,*)
        IF(INDIDEC.NE.0)THEN
         WRITE(NSUSYLA,73)'MHL   ','C1 C1 ','C2 C2 ','C1 C2 '
         WRITE(NSUSYLA,69)
         WRITE(NSUSYLA,*)
         WRITE(NSUSYLB,71)'MHL   ','N1 N1 ','N2 N2 ','N3 N3 ','N4 N4 '
         WRITE(NSUSYLB,69)
         WRITE(NSUSYLB,*)
         WRITE(NSUSYLC,70)'MHL   ','N1 N2 ','N1 N3 ','N1 N4 ','N2 N3 ',
     .                    'N2 N4 ','N3 N4 '
         WRITE(NSUSYLC,69)
         WRITE(NSUSYLC,*)
         WRITE(NSUSYLD,*)'   MHL        SNL SNL    SEL SEL    '//
     .   'SER SER    STA1 STA1  STA1 STA2  STA2 STA2' 
         WRITE(NSUSYLD,69)
         WRITE(NSUSYLD,*)
         WRITE(NSUSYLE,*)'   MHL        SUL SUL    SUR SUR    '//
     .   'SDL SDL    SDR SDR'
         WRITE(NSUSYLE,69)
         WRITE(NSUSYLE,*)
         WRITE(NSUSYLF,*)'   MHL        SB1 SB1    SB1 SB2    '//
     .   'SB2 SB2    ST1 ST1    ST1 ST2    ST2 ST2'
         WRITE(NSUSYLF,69)
         WRITE(NSUSYLF,*)
        ENDIF
       ENDIF

       IF(IHIGGS.EQ.2.OR.IHIGGS.EQ.5) THEN
       WRITE(NSUSYH,347) TGBET,AM2,AMU,AMSQ
       WRITE(NSUSYH,348) GMC(1),GMC(2),GMN(1),GMN(2),GMN(3),GMN(4)
       WRITE(NSUSYH,349) GMST(1),GMST(2),GMSU(1),GMSU(2)
       WRITE(NSUSYH,350) GMSB(1),GMSB(2),GMSD(1),GMSD(2)
       WRITE(NSUSYH,351)GMSL(1),GMSL(2),GMSN(1),GMSE(1),GMSE(2),GMSN1(1)
       WRITE(NSUSYH,*)
       WRITE(NSUSYH,*)'   MHH        CHARGINOS  NEUTRALS   '//
     . 'SLEPTONS   SQUARKS  GRAVITINO+GAUGINO'
       WRITE(NSUSYH,69)
       WRITE(NSUSYH,*)
        IF(INDIDEC.NE.0)THEN
         WRITE(NSUSYHA,73)'MHH   ','C1 C1 ','C2 C2 ','C1 C2 '
         WRITE(NSUSYHA,69)
         WRITE(NSUSYHA,*)
         WRITE(NSUSYHB,71)'MHH   ','N1 N1 ','N2 N2 ','N3 N3 ','N4 N4 '
         WRITE(NSUSYHB,69)
         WRITE(NSUSYHB,*)
         WRITE(NSUSYHC,70)'MHH   ','N1 N2 ','N1 N3 ','N1 N4 ','N2 N3 ',
     .                    'N2 N4 ','N3 N4 '
         WRITE(NSUSYHC,69)
         WRITE(NSUSYHC,*)
         WRITE(NSUSYHD,*)'   MHH        SNL SNL    SEL SEL    '//
     .   'SER SER    STA1 STA1  STA1 STA2  STA2 STA2' 
         WRITE(NSUSYHD,69)
         WRITE(NSUSYHD,*)
         WRITE(NSUSYHE,*)'   MHH        SUL SUL    SUR SUR    '//
     .   'SDL SDL    SDR SDR'
         WRITE(NSUSYHE,69)
         WRITE(NSUSYHE,*)
         WRITE(NSUSYHF,*)'   MHH        SB1 SB1    SB1 SB2    '//
     .   'SB2 SB2    ST1 ST1    ST1 ST2    ST2 ST2'
         WRITE(NSUSYHF,69)
         WRITE(NSUSYHF,*)
        ENDIF
       ENDIF

       IF(IHIGGS.EQ.3.OR.IHIGGS.EQ.5) THEN
       WRITE(NSUSYA,347) TGBET,AM2,AMU,AMSQ
       WRITE(NSUSYA,348) GMC(1),GMC(2),GMN(1),GMN(2),GMN(3),GMN(4)
       WRITE(NSUSYA,349) GMST(1),GMST(2),GMSU(1),GMSU(2)
       WRITE(NSUSYA,350) GMSB(1),GMSB(2),GMSD(1),GMSD(2)
       WRITE(NSUSYA,351)GMSL(1),GMSL(2),GMSN(1),GMSE(1),GMSE(2),GMSN1(1)
       WRITE(NSUSYA,*)
       WRITE(NSUSYA,*)'   MHA        CHARGINOS  NEUTRALS   '//
     . 'SLEPTONS   SQUARKS  GRAVITINO+GAUGINO'
       WRITE(NSUSYA,69)
       WRITE(NSUSYA,*)
        IF(INDIDEC.NE.0)THEN
         WRITE(NSUSYAA,73)'MHA   ','C1 C1 ','C2 C2 ','C1 C2 '
         WRITE(NSUSYAA,69)
         WRITE(NSUSYAA,*)
         WRITE(NSUSYAB,71)'MHA   ','N1 N1 ','N2 N2 ','N3 N3 ','N4 N4 '
         WRITE(NSUSYAB,69)
         WRITE(NSUSYAB,*)
         WRITE(NSUSYAC,70)'MHA   ','N1 N2 ','N1 N3 ','N1 N4 ','N2 N3 ',
     .                    'N2 N4 ','N3 N4 '
         WRITE(NSUSYAC,69)
         WRITE(NSUSYAC,*)
         WRITE(NSUSYAD,*)
         WRITE(NSUSYAD,*)'   MHA        STA1 STA2  SB1 SB2    ST1 ST2'
         WRITE(NSUSYAD,69)
         WRITE(NSUSYAD,*)
        ENDIF
       ENDIF

       IF(IHIGGS.EQ.4.OR.IHIGGS.EQ.5) THEN
       WRITE(NSUSYC,347) TGBET,AM2,AMU,AMSQ
       WRITE(NSUSYC,348) GMC(1),GMC(2),GMN(1),GMN(2),GMN(3),GMN(4)
       WRITE(NSUSYC,349) GMST(1),GMST(2),GMSU(1),GMSU(2)
       WRITE(NSUSYC,350) GMSB(1),GMSB(2),GMSD(1),GMSD(2)
       WRITE(NSUSYC,351)GMSL(1),GMSL(2),GMSN(1),GMSE(1),GMSE(2),GMSN1(1)
       WRITE(NSUSYC,*)
       WRITE(NSUSYC,*)'   MHC        CHARG/NEU  SLEPTONS   SQUARKS',
     .                '  GRAVITINO+GAUGINO'
       WRITE(NSUSYC,69)
       WRITE(NSUSYC,*)
        IF(INDIDEC.NE.0)THEN
         WRITE(NSUSYCA,70)'MHC   ','C1 N1 ','C1 N2 ','C1 N3 ','C1 N4 '
         WRITE(NSUSYCA,69)
         WRITE(NSUSYCA,*)
         WRITE(NSUSYCB,70)'MHC   ','C2 N1 ','C2 N2 ','C2 N3 ','C2 N4 '
         WRITE(NSUSYCB,69)
         WRITE(NSUSYCB,*)
         WRITE(NSUSYCC,*)'   MHC        SEL SNL    STAU1 SNL  STAU2 SNL'
         WRITE(NSUSYCC,69)
         WRITE(NSUSYCC,*)
         WRITE(NSUSYCD,*)'   MHC        SUL SDL    ST1 SB1    '//
     .   'ST1 SB2    ST2 SB1    ST2 SB2'
         WRITE(NSUSYCD,69)
         WRITE(NSUSYCD,*)
        ENDIF
       ENDIF

347    FORMAT('TB=',G12.6,1X,'M2=',G12.6,1X,'MU=',G12.6,1X,
     .        'MSQ=',G12.6)
348    FORMAT('C1=',F7.3,1X,'C2=',F8.3,1X,'N1=',F7.3,1X,'N2=',F7.3,1X,
     .        'N3=',F8.3,1X,'N4=',F8.3)
349    FORMAT('MST1=',G12.6,1X,'MST2=',G12.6,1X,
     .        'MSUL=',G12.6,1X,'MSUR=',G12.6) 
350    FORMAT('MSB1=',G12.6,1X,'MSB2=',G12.6,1X,
     .        'MSDL=',G12.6,1X,'MSDR=',G12.6) 
351    FORMAT('TAU1=',F8.3,1X,'TAU2=',F8.3,1X,'NL=',F8.3,1X,
     .        'EL=',F8.2,1X,'ER=',F8.2,1X,'NL1=',F8.2)
C
C
C **************************************************************
      ENDIF
      ENDIF

      RETURN
      END

      SUBROUTINE WRITE_HDEC(TGBET)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER(K=6,NI=87,NSA=85,NSB=86,NSC=57,NLA=88,NLB=89,NHA=90,
     .          NHB=91,NHC=92,NAA=93,NAB=94,NCA=95,NCB=96,NCC=50,NRA=97,
     .          NRB=98,
     .          NSUSYL=81,NSUSYA=82,NSUSYH=83,NSUSYC=84,NPAR=80,NTA=99,
     .          NSUSYLA=79,NSUSYLB=78,NSUSYLC=77,NSUSYLD=76,NSUSYLE=75,
     .          NSUSYLF=59,NSUSYHF=58,
     .          NSUSYHA=74,NSUSYHB=73,NSUSYHC=72,NSUSYHD=71,NSUSYHE=70,
     .          NSUSYAA=69,NSUSYAB=68,NSUSYAC=67,NSUSYAD=66,NSUSYAE=65,
     .          NSUSYCA=64,NSUSYCB=63,NSUSYCC=62,NSUSYCD=61,NSUSYCE=60)
c MMM changed 23/8/2013
      PARAMETER (NAC=31,NLC=32)
c end MMM changed 23/8/2013
      parameter (nout=44)
      DIMENSION GMN(4),XMN(4),GMC(2),GMST(2),GMSB(2),GMSL(2),
     .          GMSU(2),GMSD(2),GMSE(2),GMSN(2),GMSN1(2)
      DIMENSION HLBRSC(2,2),HLBRSN(4,4),HHBRSC(2,2),HHBRSN(4,4),
     .          HABRSC(2,2),HABRSN(4,4),HCBRSU(2,4),
     .          HHBRST(2,2),HHBRSB(2,2),HCBRSTB(2,2) 
      DIMENSION AC1(2,2),AC2(2,2),AC3(2,2),
     .          AN1(4,4),AN2(4,4),AN3(4,4),
     .          ACNL(2,4),ACNR(2,4)
      DIMENSION GLTT(2,2),GLBB(2,2),GHTT(2,2),GHBB(2,2),GCTB(2,2),
     .          GLEE(2,2),GHEE(2,2),GCEN(2,2)
      DIMENSION AGDL(4),AGDA(4),AGDH(4),AGDC(2)
      dimension hlbrsn1(4,4),hhbrsn1(4,4),habrsn1(4,4)
      double precision minval(1:20),smval(1:30),massval(1:50),
     .                 nmixval(4,4),umixval(2,2),vmixval(2,2),
     .                 stopmixval(2,2),sbotmixval(2,2),staumixval(2,2),
     .                 hmixval(1:10),gaugeval(1:3),msoftval(1:100),
     .                 auval(3,3),adval(3,3),aeval(3,3),yuval(3,3),
     .                 ydval(3,3),yeval(3,3),qvalue(1:20),extval(0:100),
     .                 m_softval(1:100)
      double precision slhaneut(1:4),slhaxneut(1:4),slhachar(1:2),
     .                 slhau(2,2),slhav(2,2),slhaz(4,4),slhaxchar(1:2),
     .                 slhast(2),slhasb(2),slhasu(2),slhasd(2),
     .                 slhase(2),slhasl(2),slhasn(2),slhasnl(2),
     .                 warning(1:10)
      double precision vckmval(4)
      integer   imod(1:2)
      integer check(1:22)
      double precision mbmsbar,mbl,mbu
      character spinfo1*100,spinfo2*100,modselval*100,mincom(1:20)*20,
     .          extcom(0:100)*20,softcom(1:100)*20,hmixcom(1:10)*20,
     .          m_softcom(1:100)*20
      COMMON/MASSES_HDEC/AMS,AMC,AMB,AMT
      COMMON/STRANGE_HDEC/AMSB
      COMMON/PARAM_HDEC/GF,ALPH,AMTAU,AMMUON,AMZ,AMW
      COMMON/CKMPAR_HDEC/VTB,VTS,VTD,VCB,VCS,VCD,VUB,VUS,VUD
      COMMON/HMASS_HDEC/AMSM,AMA,AML,AMH,AMCH,AMAR
      COMMON/BREAK_HDEC/AMEL,AMER,AMSQ,AMUR,AMDR,AL,AU,AD,AMU,AM2
      COMMON/BREAKGLU_HDEC/AMGLU
      COMMON/SFER1ST_HDEC/AMQL1,AMUR1,AMDR1,AMEL1,AMER1
      COMMON/GLUINO_HDEC/AMGLUINO,XMSB1,XMSB2,STHB,CTHB,
     .              XLBB(2,2),XHBB(2,2),XABB(2,2),
     .              XMST1,XMST2,STHT,CTHT,
     .              XLTT(2,2),XHTT(2,2),XATT(2,2)
      COMMON/WZWDTH_HDEC/GAMC0,GAMT0,GAMT1,GAMW,GAMZ
      COMMON/COUP_HDEC/GAT,GAB,GLT,GLB,GHT,GHB,GZAH,GZAL,
     .            GHHH,GLLL,GHLL,GLHH,GHAA,GLAA,GLVV,GHVV,
     .            GLPM,GHPM,B,A
      COMMON/ALS_HDEC/XLAMBDA,AMC0,AMB0,AMT0,N0
      COMMON/FLAG_HDEC/IHIGGS,NNLO,IPOLE
      COMMON/MODEL_HDEC/IMODEL
      COMMON/ONSHELL_HDEC/IONSH,IONWZ,IOFSUSY
      COMMON/OLDFASH_HDEC/NFGG
      COMMON/SM4_HDEC/AMTP,AMBP,AMNUP,AMEP,ISM4,IGGELW
      COMMON/WIDTHSM_HDEC/SMBRB,SMBRL,SMBRM,SMBRS,SMBRC,SMBRT,SMBRG,
     .               SMBRGA,SMBRZGA,SMBRW,SMBRZ,SMWDTH
      COMMON/WIDTHSM4_HDEC/SMBRNUP,SMBREP,SMBRBP,SMBRTP
      COMMON/WIDTHA_HDEC/ABRB,ABRL,ABRM,ABRS,ABRC,ABRT,ABRG,ABRGA,
     .              ABRZGA,ABRZ,AWDTH
      COMMON/WIDTHHL_HDEC/HLBRB,HLBRL,HLBRM,HLBRS,HLBRC,HLBRT,HLBRG,
     .               HLBRGA,HLBRZGA,HLBRW,HLBRZ,HLBRA,HLBRAZ,HLBRHW,
     .               HLWDTH
      COMMON/WIDTHHH_HDEC/HHBRB,HHBRL,HHBRM,HHBRS,HHBRC,HHBRT,HHBRG,
     .               HHBRGA,HHBRZGA,HHBRW,HHBRZ,HHBRH,HHBRA,HHBRAZ,
     .               HHBRHW,HHWDTH
      COMMON/WIDTHHC_HDEC/HCBRB,HCBRL,HCBRM,HCBRBU,HCBRS,HCBRC,HCBRT,
     .               HCBRW,HCBRA,HCWDTH
      COMMON/WISUSY_HDEC/HLBRSC,HLBRSN,HHBRSC,HHBRSN,HABRSC,HABRSN,
     .              HCBRSU,HLBRCHT,HHBRCHT,HABRCHT,HLBRNET,HHBRNET,
     .              HABRNET,HCBRCNT,HLBRSL,HHBRSL,HCBRSL,HABRSL,HABRST,
     .              HABRSB,HHBRSQ,HHBRST,HHBRSB,HHBRSQT,HCBRSQ,HCBRSTB,
     .              HCBRSQT,HLBRSQ,HLBRSQT
      COMMON/WISFER_HDEC/BHLSLNL,BHLSLEL,BHLSLER,BHLSQUL,BHLSQUR,
     .              BHLSQDL,BHLSQDR,BHLST(2,2),BHLSB(2,2),BHLSTAU(2,2),
     .              BHHSLNL,BHHSLEL,BHHSLER,BHHSQUL,BHHSQUR,BHHSQDL,
     .              BHHSQDR,BHHST(2,2),BHHSB(2,2),BHHSTAU(2,2),
     .              BHASTAU,BHASB,BHAST,
     .              BHCSL00,BHCSL11,BHCSL21,BHCSQ,BHCSTB(2,2)
      COMMON/SMASS_HDEC/GMN,XMN,GMC,GMST,GMSB,GMSL,GMSU,GMSD,GMSE,GMSN 
     .                 ,GMSN1
      COMMON/GOLDST_HDEC/AXMPL,AXMGD,IGOLD
      COMMON/WIGOLD_HDEC/HLBRGD,HABRGD,HHBRGD,HCBRGD
      COMMON/FLAGS_HDEC/INDIDEC
c -------------- common block given by read_leshouches ------------ c
      COMMON/SLHA_leshouches1_HDEC/spinfo1,spinfo2,modselval,mincom,
     .                             extcom,softcom,hmixcom
      COMMON/SLHA_leshouches2_HDEC/minval,extval,smval,massval,nmixval,
     .                      umixval,vmixval,stopmixval,sbotmixval,
     .                      staumixval,hmixval,gaugeval,msoftval,auval,
     .                      adval,aeval,yuval,ydval,yeval,alphaval,
     .                      qvalue,imod
      COMMON/SLHA_leshouches3_HDEC/vckmval
      COMMON/SD_scaleofewsb/scaleofewsb
c -------------- common blocks needed in HDECAY subroutines ---------- c
      COMMON/SLHA_vals_HDEC/islhai,islhao
      COMMON/SLHA_m1_HDEC/am1
      COMMON/SLHA_gaug_HDEC/slhaneut,slhaxneut,slhachar,slhau,slhav,
     .                      slhaz,slhaxchar
      COMMON/SLHA_sfer_HDEC/slhast,slhasb,slhasu,slhasd,slhase,slhasl,
     .                 slhasn,slhasnl,slhacot,slhasit,slhacob,slhasib,
     .                 slhacol,slhasil
      COMMON/SLHA_hmass_HDEC/slhaml,slhamh,slhamc,slha_alpha
      COMMON/GAUGINOMIX_HDEC/ZZ(4,4),UU(2,2),VV(2,2)
      COMMON/TAUMIX_HDEC/CL,SL
      COMMON/SLHAVAL_HDEC/g1ew,g2ew
      COMMON/SLHA_checkval_HDEC/check
c MMM changed 21/8/13
      COMMON/THDM_HDEC/TGBET2HDM,ALPH2HDM,AMHL2HDM,AMHH2HDM,
     .     AMHA2HDM,AMHC2HDM,AM12SQ,A1LAM2HDM,A2LAM2HDM,A3LAM2HDM,
     .     A4LAM2HDM,A5LAM2HDM,ITYPE2HDM,I2HDM,IPARAM2HDM
      COMMON/WIDTH_HC_ADD/hcbrcd,hcbrts,hcbrtd
      COMMON/WIDTH_2HDM/hcbrwhh,hhbrchch,hlbrchch,abrhhaz,abrhawphm
c end MMM changed 21/8/13

      PI = 4*DATAN(1D0)

      if(islhao.eq.1) then
         open(nout,file='slha.out')

         id =1
         idb=-1
         iu =2
         iub=-2
         is =3
         isb=-3
         ic =4
         icb=-4
         ib =5
         ibb=-5
         it =6
         itb=-6
         
         ie   =11
         ine  =12
         imu  =13
         inmu =14
         itau =15
         intau=16

         ihl=25
         ihh=35
         iha=36
         ihc=37
         igl=21
         iga=22
         iz =23
         iwc=24

         isdl=1000001
         isdr=2000001
         isul=1000002
         isur=2000002
         issl=1000003
         issr=2000003
         iscl=1000004
         iscr=2000004
         isb1=1000005
         isb2=2000005
         ist1=1000006
         ist2=2000006

         iglo=1000021
         in1 =1000022
         in2 =1000023
         in3 =1000025
         in4 =1000035
         ic1 =1000024
         ic2 =1000037

         intau1=1000016 
         intau2=2000016 
         inel  =1000012
         iner  =2000012
         inmul =1000014
         inmur =2000014
      
         isell =1000011
         iselr =2000011
         ismul =1000013
         ismur =2000013
         istau1=1000015
         istau2=2000015

c ----------------------------------- c
c Information about the decay program c
c ----------------------------------- c

      write(nout,105)
      write(nout,51) 'DCINFO','Decay Program information'
      write(nout,61) 1,'HDECAY      # decay calculator'
      write(nout,61) 2,'5.10        # version number'

c ----------------------------------------------------------------- c
c The program information: Which spectrum calculator has been used. c
c ----------------------------------------------------------------- c

      if(check(22).eq.1) then
         write(nout,105)
         write(nout,51) 'SPINFO','Spectrum calculator information'
         write(nout,61) 1,spinfo1(1:50)
         write(nout,61) 2,spinfo2(1:50)
      endif

c ------------------------------------------------ c
c Information on the model which has been selected c
c ------------------------------------------------ c

      write(nout,105)
      write(nout,51) 'MODSEL','Model selection'
      write(nout,62) 1,0,'# General MSSM'

c ----------------------- c
c The SM input parameters c
c ----------------------- c

c     if(smval(1).ne.0.d0)then
c      salpha_MS = 1/smval(1)
c     else
c      salpha_MS = 1/127.934D0
c     endif
c -- calculation of mb(mb)_MSbar from the mb pole mass --
      del = 1.d-8
      rmb0 = amb
444   rmb = rmb0
      rmb0 = runm_hdec(rmb,5)
      if(dabs(rmb0/rmb-1).gt.del)goto 444
      rmb = rmb0
      alsmz = alphas_hdec(amz,3)
      write(nout,105)
      write(nout,51) 'SMINPUTS','Standard Model inputs'
c     write(nout,52) 1,1.D0/salpha_MS,'alpha_em^-1(M_Z)^MSbar'
      write(nout,52) 2,gf,'G_F [GeV^-2]'
      write(nout,52) 3,alsmz,'alpha_S(M_Z)^MSbar'
      write(nout,52) 4,amz,'M_Z pole mass'
      write(nout,52) 5,rmb,'mb(mb)^MSbar'
      write(nout,52) 6,amt,'mt pole mass'
      write(nout,52) 7,amtau,'mtau pole mass'

c ------------------------------------------------ c
c Input parameters for minimal/default SUSY models c
c ------------------------------------------------ c

      if(check(3).eq.1) then
         write(nout,105)
         write(nout,51) 'MINPAR','Input parameters - minimal models'
         unlikely = -123456789D0
         do ii=1,20,1
            if(minval(ii).ne.unlikely) then
               write(nout,52) ii,minval(ii),mincom(ii)
            endif
         end do
      endif

c ------------------------------------------------------------------- c
c Optional input parameters for non-minimal/non-universal SUSY models c
c ------------------------------------------------------------------- c

      if(check(4).eq.1) then
         write(nout,105)
         write(nout,51) 'EXTPAR','Input parameters - non-minimal models'
         unlikely = -123456789D0
         do ii=1,100,1
            if(extval(ii-1).ne.unlikely) then
               write(nout,52) ii-1,extval(ii-1),extcom(ii-1)
            endif
         end do
      endif

c ----------------- c
c The mass spectrum c
c ----------------- c

      write(nout,105)
      write(nout,51) 'MASS','Mass Spectrum'
      write(nout,50) 'PDG code           mass       particle'
      write(nout,52) iwc,amw,'W+'
      write(nout,52) ihl,aml,'h'
      write(nout,52) ihh,amh,'H'
      write(nout,52) iha,ama,'A'
      write(nout,52) ihc,amch,'H+'
      write(nout,52) ib,amb,
     .'b-quark pole mass calculated from mb(mb)_Msbar'
      write(nout,52) isdl,gmsd(1),'~d_L'
      write(nout,52) isdr,gmsd(2),'~d_R'
      write(nout,52) isul,gmsu(1),'~u_L'
      write(nout,52) isur,gmsu(2),'~u_R'
      write(nout,52) issl,gmsd(1),'~s_L'
      write(nout,52) issr,gmsd(2),'~s_R'
      write(nout,52) iscl,gmsu(1),'~c_L'
      write(nout,52) iscr,gmsu(2),'~c_R'
      write(nout,52) isb1,GMSB(1),'~b_1'
      write(nout,52) isb2,GMSB(2),'~b_2'
      write(nout,52) ist1,GMST(1),'~t_1'
      write(nout,52) ist2,GMST(2),'~t_2'
      write(nout,52) isell,GMSE(1),'~e_L'
      write(nout,52) iselr,GMSE(2),'~e_R'
      write(nout,52) inel,GMSN1(1),'~nu_eL'
      write(nout,52) ismul,GMSE(1),'~mu_L'
      write(nout,52) ismur,GMSE(2),'~mu_R'
      write(nout,52) inmul,GMSN1(1),'~nu_muL'
      write(nout,52) istau1,GMSL(1),'~tau_1'
      write(nout,52) istau2,GMSL(2),'~tau_2'
      write(nout,52) intau1,GMSN(1),'~nu_tauL'
      write(nout,52) iglo,amgluino,'~g'
      write(nout,52) in1,xmn(1),'~chi_10'
      write(nout,52) in2,xmn(2),'~chi_20'
      write(nout,52) in3,xmn(3),'~chi_30'
      write(nout,52) in4,xmn(4),'~chi_40'
      write(nout,52) ic1,gmc(1),'~chi_1+'
      write(nout,52) ic2,gmc(2),'~chi_2+'

c ------------------------------------------------------------------- c
c The neutralino mixing matrix N and the chargino mixing matrices U,V c
c ------------------------------------------------------------------- c

      write(nout,105)
      write(nout,51) 'NMIX','Neutralino Mixing Matrix'
      write(nout,53) 1,1,zz(1,1),'N_11'
      write(nout,53) 1,2,zz(1,2),'N_12'
      write(nout,53) 1,3,zz(1,3),'N_13'
      write(nout,53) 1,4,zz(1,4),'N_14'
      write(nout,53) 2,1,zz(2,1),'N_21'
      write(nout,53) 2,2,zz(2,2),'N_22'
      write(nout,53) 2,3,zz(2,3),'N_23'
      write(nout,53) 2,4,zz(2,4),'N_24'
      write(nout,53) 3,1,zz(3,1),'N_31'
      write(nout,53) 3,2,zz(3,2),'N_32'
      write(nout,53) 3,3,zz(3,3),'N_33'
      write(nout,53) 3,4,zz(3,4),'N_34'
      write(nout,53) 4,1,zz(4,1),'N_41'
      write(nout,53) 4,2,zz(4,2),'N_42'
      write(nout,53) 4,3,zz(4,3),'N_43'
      write(nout,53) 4,4,zz(4,4),'N_44'

      write(nout,105)
      write(nout,51) 'UMIX','Chargino Mixing Matrix U'
      write(nout,53) 1,1,uu(1,1),'U_11'
      write(nout,53) 1,2,uu(1,2),'U_12'
      write(nout,53) 2,1,uu(2,1),'U_21'
      write(nout,53) 2,2,uu(2,2),'U_22'

      write(nout,105)
      write(nout,51) 'VMIX','Chargino Mixing Matrix V'
      write(nout,53) 1,1,vv(1,1),'V_11'
      write(nout,53) 1,2,vv(1,2),'V_12'
      write(nout,53) 2,1,vv(2,1),'V_21'
      write(nout,53) 2,2,vv(2,2),'V_22'

c ------------------------------------------ c
c The stop, sbottom and stau mixing matrices c
c ------------------------------------------ c

      write(nout,105)
      write(nout,51) 'STOPMIX','Stop Mixing Matrix'
      write(nout,53) 1,1,ctht,'cos(theta_t)'
      write(nout,53) 1,2,stht,'sin(theta_t)'
      write(nout,53) 2,1,-stht,'-sin(theta_t)'
      write(nout,53) 2,2,ctht,'cos(theta_t)'

      write(nout,105)
      write(nout,51) 'SBOTMIX','Sbottom Mixing Matrix'
      write(nout,53) 1,1,cthb,'cos(theta_b)'
      write(nout,53) 1,2,sthb,'sin(theta_b)'
      write(nout,53) 2,1,-sthb,'-sin(theta_b)'
      write(nout,53) 2,2,cthb,'cos(theta_b)'

      write(nout,105)
      write(nout,51) 'STAUMIX','Stau Mixing Matrix'
      write(nout,53) 1,1,cl,'cos(theta_tau)'
      write(nout,53) 1,2,sl,'sin(theta_tau)'
      write(nout,53) 2,1,-sl,'-sin(theta_tau)'
      write(nout,53) 2,2,cl,'cos(theta_tau)'

c ------------------------------------------------------------------- c
c The angle alpha in the Higgs sector and the Higgs mixing parameters c
c ------------------------------------------------------------------- c

      alphaval = A
      write(nout,105)
      write(nout,51) 'ALPHA','Higgs mixing'
      write(nout,60) alphaval,
     .'Mixing angle in the neutral Higgs boson sector'

      amudrbar = AMU/(1.D0+g1ew**2/16.D0/pi**2*3.D0/5.D0+
     .                 g2ew**2/16.D0/pi**2*3.D0/4.D0)
      if(qvalue(1).ne.0.d0)then
       qq = qvalue(1)
      else
       qq = amt
      endif
      write(nout,105)
      write(nout,54) 'HMIX Q= ',qq,'DRbar Higgs Parameters'
      write(nout,52) 1,amudrbar,'mu(Q)'
      write(nout,52) 2,tgbet,'tanbeta(Q)'

c --------------------- c
c The CKM mixing matrix c
c --------------------- c

      unlikely = -123456789D0
      r0 = 0.132d0
      e0 = 0.341d0
      fac1 = r0/dsqrt(r0**2+e0**2)
      fac2 = e0/dsqrt(r0**2+e0**2)
      if(vckmval(1).eq.unlikely)vckmval(1) = vus
      if(vckmval(2).eq.unlikely)vckmval(2) = vcb/vus**2
      if(vckmval(3).eq.unlikely)vckmval(3) = fac1*vub/vus/vcb
      if(vckmval(4).eq.unlikely)vckmval(4) = fac2*vub/vus/vcb
      write(nout,105)
      write(nout,51) 'VCKMIN','CKM mixing'
      write(nout,52) 1,vckmval(1),'lambda'
      write(nout,52) 2,vckmval(2),'A'
      write(nout,52) 3,vckmval(3),'rhobar'
      write(nout,52) 4,vckmval(4),'etabar'

c ------------------- c
c The gauge couplings c
c ------------------- c
 
c     del = 1.d-8
c     g2ew0 = g2ew
c80   g2test  = g2ew/(1-g2ew0**2/96/pi**2*2)
c     g2ew1  = g2test*(1-g2test**2/96/pi**2*2)
c     write(6,*)g2ew,g2ew1,g2test
c     g2ew0 = g2test
c     if(dabs(g2ew1/g2ew-1).gt.del)goto 80
c     g2drbar = g2test

      if(qvalue(2).ne.0.d0)then
       write(nout,105)
       write(nout,54) 'GAUGE Q=',qvalue(2),'The gauge couplings'
       if(gaugeval(1).ne.0.D0) then
          write(nout,55) 1,gaugeval(1),'gprime(Q) DRbar'
       endif
       if(gaugeval(2).ne.0.D0) then
          write(nout,55) 2,gaugeval(2),'g(Q) DRbar'
       endif
      endif

c ------------------------------------- c
c The trilinear couplings Au, Ad and Ae c
c ------------------------------------- c

      qq = amt
      if(qvalue(4).ne.0.d0)qq = qvalue(4)
      write(nout,105)
      write(nout,54) 'AU Q=',qq,'The trilinear couplings'
      write(nout,53) 1,1,au,'A_u(Q) DRbar'
      write(nout,53) 2,2,au,'A_c(Q) DRbar'
      write(nout,53) 3,3,au,'A_t(Q) DRbar'

      qq = amt
      if(qvalue(5).ne.0.d0)qq = qvalue(5)
      write(nout,105)
      write(nout,54) 'AD Q=',qq,'The trilinear couplings'
      write(nout,53) 1,1,ad,'A_d(Q) DRbar'
      write(nout,53) 2,2,ad, 'A_s(Q) DRbar'
      write(nout,53) 3,3,ad,'A_b(Q) DRbar'

      qq = amt
      if(qvalue(6).ne.0.d0)qq = qvalue(6)
      write(nout,105)
      write(nout,54) 'AE Q=',qq,'The trilinear couplings'
      write(nout,53) 1,1,al,'A_e(Q) DRbar'
      write(nout,53) 2,2,al, 'A_mu(Q) DRbar'
      write(nout,53) 3,3,al,'A_tau(Q) DRbar'

c ----------------------------- c
c The soft SUSY breaking masses c
c ----------------------------- c

      if(check(15).eq.1) then
         write(nout,105)
         write(nout,54) 'MSOFT Q=',scaleofewsb,'The soft SUSY breaking m
     .asses at the scale Q'
         unlikely = -123456789D0
         do ii=1,99,1
            if(msoftval(ii).ne.unlikely) then
               if(ii.ne.11.and.ii.ne.12.and.ii.ne.13.and.ii.ne.23.and.
     .            ii.ne.24.and.ii.ne.25.and.ii.ne.26) then
                  write(nout,52) ii,msoftval(ii),softcom(ii)
               endif
            endif
         end do
      else
         write(nout,105)
         write(nout,54) 'MSOFT Q=',scaleofewsb,'The soft SUSY breaking m
     .asses at the scale Q'
         cw=amw/amz
         sw=dsqrt(1-cw**2)
         tw=sw/cw
         am1=5.D0/3.D0*tw**2*am2
         am2=am2/(1.D0+g2ew**2/16.D0/pi**2*2.D0)
         write(nout,52) 1,am1,'M_1(Q)'
         write(nout,52) 2,am2,'M_2(Q)'
         write(nout,52) 31,amel1,'AMEL1'
         write(nout,52) 33,amel,'AMEL'
         write(nout,52) 34,amer1,'AMER1'
         write(nout,52) 36,amer,'AMER'
         write(nout,52) 41,amql1,'AMQL1'
         write(nout,52) 43,amsq,'AMSQ'
         write(nout,52) 44,amur1,'AMUR1'
         write(nout,52) 46,amur,'AMUR'
         write(nout,52) 47,amdr1,'AMDR1'
         write(nout,52) 49,amdr,'AMDR'
      endif

         if(ihiggs.eq.0) then
            if(smwdth.ne.0.D0) then
               write(nout,99)
               write(nout,100) 25,smwdth,'SM Higgs decays'

               write(nout,101)
      
               if(smbrb.ne.0.D0) then
      write(nout,102) smbrb,2,ib,ibb        ,'BR(H -> b       bb     )'
               endif
               if(smbrl.ne.0.D0) then
      write(nout,102) smbrl,2,-itau,itau    ,'BR(H -> tau+    tau-   )'
               endif
               if(smbrm.ne.0.D0) then
      write(nout,102) smbrm,2,-imu,imu      ,'BR(H -> mu+     mu-    )'
               endif
               if(smbrs.ne.0.D0) then
      write(nout,102) smbrs,2,is,isb        ,'BR(H -> s       sb     )'
               endif
               if(smbrc.ne.0.D0) then
      write(nout,102) smbrc,2,ic,icb        ,'BR(H -> c       cb     )'
               endif
               if(smbrt.ne.0.D0) then
      write(nout,102) smbrt,2,it,itb        ,'BR(H -> t       tb     )' 
               endif
               if(smbrg.ne.0.D0) then
      write(nout,102) smbrg,2,igl,igl       ,'BR(H -> g       g      )' 
               endif
               if(smbrga.ne.0.D0) then
      write(nout,102) smbrga,2,iga,iga      ,'BR(H -> gam     gam    )'  
               endif
               if(smbrzga.ne.0.D0) then
      write(nout,102) smbrzga,2,iga,iz      ,'BR(H -> Z       gam    )' 
               endif
               if(smbrw.ne.0.D0) then
      write(nout,102) smbrw,2,iwc,-iwc      ,'BR(H -> W+      W-     )' 
               endif
               if(smbrz.ne.0.D0) then
      write(nout,102) smbrz,2,iz,iz         ,'BR(H -> Z       Z      )' 
               endif

            elseif(smwdth.eq.0.D0) then
               write(nout,99)
               write(nout,100) 25,0.000000000E+00,'SM Higgs decays'
               
            endif
         endif

         if(ihiggs.eq.1.or.ihiggs.eq.5) then
            write(nout,105)

      if(hlwdth.ne.0.D0) then

      write(nout,99)
      write(nout,100) 25,hlwdth,'h decays'

      write(nout,101)
      if(hlbrb.ne.0.D0) then
      write(nout,102) hlbrb,2,ib,ibb        ,'BR(h -> b       bb     )'
      endif
      if(hlbrl.ne.0.D0) then
      write(nout,102) hlbrl,2,-itau,itau    ,'BR(h -> tau+    tau-   )'
      endif
      if(hlbrm.ne.0.D0) then
      write(nout,102) hlbrm,2,-imu,imu      ,'BR(h -> mu+     mu-    )'
      endif
      if(hlbrs.ne.0.D0) then
      write(nout,102) hlbrs,2,is,isb        ,'BR(h -> s       sb     )'
      endif
      if(hlbrc.ne.0.D0) then
      write(nout,102) hlbrc,2,ic,icb        ,'BR(h -> c       cb     )'
      endif
      if(hlbrt.ne.0.D0) then
      write(nout,102) hlbrt,2,it,itb        ,'BR(h -> t       tb     )' 
      endif
      if(hlbrg.ne.0.D0) then
      write(nout,102) hlbrg,2,igl,igl       ,'BR(h -> g       g      )' 
      endif
      if(hlbrga.ne.0.D0) then
      write(nout,102) hlbrga,2,iga,iga      ,'BR(h -> gam     gam    )' 
      endif
      if(hlbrzga.ne.0.D0) then
      write(nout,102) hlbrzga,2,iga,iz      ,'BR(h -> Z       gam    )' 
      endif
      if(hlbrw.ne.0.D0) then
      write(nout,102) hlbrw,2,iwc,-iwc      ,'BR(h -> W+      W-     )' 
      endif
      if(hlbrz.ne.0.D0) then
      write(nout,102) hlbrz,2,iz,iz         ,'BR(h -> Z       Z      )' 
      endif
      if(hlbra.ne.0.D0) then
      write(nout,102) hlbra,2,iha,iha       ,'BR(h -> A       A      )' 
      endif
      if(hlbraz.ne.0.D0) then
      write(nout,102) hlbraz,2,iz,iha       ,'BR(h -> Z       A      )' 
      endif
      if(hlbrhw.ne.0.D0) then
      write(nout,102) hlbrhw/2.D0,2,iwc,-ihc,'BR(h -> W+      H-     )' 
      endif
      if(hlbrhw.ne.0.D0) then
      write(nout,102) hlbrhw/2.D0,2,-iwc,ihc,'BR(h -> W-      H+     )' 
      endif
      if(hlbrchch.ne.0.D0) then
      write(nout,102) hlbrchch,2,ihc,-ihc,   'BR(h -> H+      H-     )'
      endif
      if(hlbrsc(1,1).ne.0.D0) then
      write(nout,102) hlbrsc(1,1),2,ic1,-ic1,'BR(h -> ~chi_1+ ~chi_1-)' 
      endif
      if(hlbrsc(2,2).ne.0.D0) then
      write(nout,102) hlbrsc(2,2),2,ic2,-ic2,'BR(h -> ~chi_2+ ~chi_2-)' 
      endif
      if(hlbrsc(1,2).ne.0.D0) then
      write(nout,102) hlbrsc(1,2),2,ic1,-ic2,'BR(h -> ~chi_1+ ~chi_2-)' 
      endif
      if(hlbrsc(2,1).ne.0.D0) then
      write(nout,102) hlbrsc(2,1),2,ic2,-ic1,'BR(h -> ~chi_2+ ~chi_1-)' 
      endif
      hlbrsn1(1,2) = 2.D0*hlbrsn(1,2) 
      hlbrsn1(1,3) = 2.D0*hlbrsn(1,3) 
      hlbrsn1(1,4) = 2.D0*hlbrsn(1,4)
      hlbrsn1(2,3) = 2.D0*hlbrsn(2,3) 
      hlbrsn1(2,4) = 2.D0*hlbrsn(2,4)  
      hlbrsn1(3,4) = 2.D0*hlbrsn(3,4)  
      if(hlbrsn(1,1).ne.0.D0) then
      write(nout,102) hlbrsn(1,1),2,in1,in1 ,'BR(h -> ~chi_10 ~chi_10)' 
      endif
      if(hlbrsn(2,2).ne.0.D0) then
      write(nout,102) hlbrsn(2,2),2,in2,in2 ,'BR(h -> ~chi_20 ~chi_20)' 
      endif
      if(hlbrsn(3,3).ne.0.D0) then
      write(nout,102) hlbrsn(3,3),2,in3,in3 ,'BR(h -> ~chi_30 ~chi_30)' 
      endif
      if(hlbrsn(4,4).ne.0.D0) then
      write(nout,102) hlbrsn(4,4),2,in4,in4 ,'BR(h -> ~chi_40 ~chi_40)' 
      endif
      if(hlbrsn(1,2).ne.0.D0) then
      write(nout,102) hlbrsn1(1,2),2,in1,in2,'BR(h -> ~chi_10 ~chi_20)' 
      endif
      if(hlbrsn(1,3).ne.0.D0) then
      write(nout,102) hlbrsn1(1,3),2,in1,in3,'BR(h -> ~chi_10 ~chi_30)' 
      endif
      if(hlbrsn(1,4).ne.0.D0) then
      write(nout,102) hlbrsn1(1,4),2,in1,in4,'BR(h -> ~chi_10 ~chi_40)' 
      endif
      if(hlbrsn(2,3).ne.0.D0) then
      write(nout,102) hlbrsn1(2,3),2,in2,in3,'BR(h -> ~chi_20 ~chi_30)' 
      endif
      if(hlbrsn(2,4).ne.0.D0) then
      write(nout,102) hlbrsn1(2,4),2,in2,in4,'BR(h -> ~chi_20 ~chi_40)' 
      endif
      if(hlbrsn(3,4).ne.0.D0) then
      write(nout,102) hlbrsn1(3,4),2,in3,in4,'BR(h -> ~chi_30 ~chi_40)' 
      endif
      bhlslnl1 = bhlslnl/3.D0
      bhlslel1 = bhlslel/2.D0
      bhlsler1 = bhlsler/2.D0
      bhlsqul1 = bhlsqul/2.d0
      bhlsqur1 = bhlsqur/2.d0
      bhlsqdl1 = bhlsqdl/2.d0
      bhlsqdr1 = bhlsqdr/2.d0
      if(bhlsqul1.ne.0.D0) then
      write(nout,102) bhlsqul1,2,isul,-isul  ,'BR(h -> ~u_L    ~u_L*  )'
      endif
      if(bhlsqur1.ne.0.D0) then
      write(nout,102) bhlsqur1,2,isur,-isur  ,'BR(h -> ~u_R    ~u_R*  )'
      endif
      if(bhlsqul1.ne.0.D0) then
      write(nout,102) bhlsqul1,2,iscl,-iscl  ,'BR(h -> ~c_L    ~c_L*  )'
      endif
      if(bhlsqur1.ne.0.D0) then
      write(nout,102) bhlsqur1,2,iscr,-iscr  ,'BR(h -> ~c_R    ~c_R*  )'
      endif
      if(bhlst(1,1).ne.0.D0) then
      write(nout,102) bhlst(1,1),2,ist1,-ist1,'BR(h -> ~t_1    ~t_1*  )'
      endif
      if(bhlst(2,2).ne.0.D0) then
      write(nout,102) bhlst(2,2),2,ist2,-ist2,'BR(h -> ~t_2    ~t_2*  )'
      endif
      if(bhlst(1,2).ne.0.D0) then
      write(nout,102) bhlst(1,2),2,ist1,-ist2,'BR(h -> ~t_1    ~t_2*  )'
      endif
      if(bhlst(2,1).ne.0.D0) then
      write(nout,102) bhlst(2,1),2,ist2,-ist1,'BR(h -> ~t_2    ~t_1*  )'
      endif
      if(bhlsqdl1.ne.0.D0) then
      write(nout,102) bhlsqdl1,2,isdl,-isdl  ,'BR(h -> ~d_L    ~d_L*  )'
      endif
      if(bhlsqdr1.ne.0.D0) then
      write(nout,102) bhlsqdr1,2,isdr,-isdr  ,'BR(h -> ~d_R    ~d_R*  )'
      endif
      if(bhlsqdl1.ne.0.D0) then
      write(nout,102) bhlsqdl1,2,issl,-issl  ,'BR(h -> ~s_L    ~s_L*  )'
      endif
      if(bhlsqdr1.ne.0.D0) then
      write(nout,102) bhlsqdr1,2,issr,-issr  ,'BR(h -> ~s_R    ~s_R*  )'
      endif
      if(bhlsb(1,1).ne.0.D0) then
      write(nout,102) bhlsb(1,1),2,isb1,-isb1,'BR(h -> ~b_1    ~b_1*  )'
      endif
      if(bhlsb(2,2).ne.0.D0) then
      write(nout,102) bhlsb(2,2),2,isb2,-isb2,'BR(h -> ~b_2    ~b_2*  )'
      endif
      if(bhlsb(1,2).ne.0.D0) then
      write(nout,102) bhlsb(1,2),2,isb1,-isb2,'BR(h -> ~b_1    ~b_2*  )'
      endif
      if(bhlsb(2,1).ne.0.D0) then
      write(nout,102) bhlsb(2,1),2,isb2,-isb1,'BR(h -> ~b_2    ~b_1*  )'
      endif
      if(bhlslel1.ne.0.D0) then
      write(nout,102) bhlslel1,2,isell,-isell,'BR(h -> ~e_L-   ~e_L+  )'
      endif
      if(bhlsler1.ne.0.D0) then
      write(nout,102) bhlsler1,2,iselr,-iselr,'BR(h -> ~e_R-   ~e_R+  )'
      endif
      if(bhlslel1.ne.0.D0) then
      write(nout,102) bhlslel1,2,ismul,-ismul,'BR(h -> ~mu_L-  ~mu_L+ )'
      endif
      if(bhlsler1.ne.0.D0) then
      write(nout,102) bhlsler1,2,ismur,-ismur,'BR(h -> ~mu_R-  ~mu_R+ )'
      endif
      if(bhlstau(1,1).ne.0.D0) then
      write(nout,102) bhlstau(1,1),2,istau1,-istau1,'BR(h -> ~tau_1- ~ta
     .u_1+)'
      endif
      if(bhlstau(2,2).ne.0.D0) then
      write(nout,102) bhlstau(2,2),2,istau2,-istau2,'BR(h -> ~tau_2- ~ta
     .u_2+)'
      endif
      if(bhlstau(1,2).ne.0.D0) then
      write(nout,102) bhlstau(1,2),2,istau1,-istau2,'BR(h -> ~tau_1- ~ta
     .u_2+)'
      endif
      if(bhlstau(2,1).ne.0.D0) then
      write(nout,102) bhlstau(2,1),2,istau2,-istau1,'BR(h -> ~tau_2- ~ta
     .u_1+)'
      endif
      if(bhlslnl1.ne.0.D0) then
      write(nout,102) bhlslnl1,2,inel,-inel  ,'BR(h -> ~nu_eL  ~nu_eL*  
     . )'
      write(nout,102) bhlslnl1,2,inmul,-inmul,'BR(h -> ~nu_muL ~nu_muL* 
     . )'
      write(nout,102) bhlslnl1,2,intau1,-intau1,'BR(h -> ~nu_tauL ~nu_ta
     .uL*)'
      endif

      elseif(hlwdth.eq.0.D0) then
      write(nout,99)
      write(nout,100) 25,0.000000000E+00,'h decays'

      endif
      endif

         if(ihiggs.eq.2.or.ihiggs.eq.5) then
            write(nout,105)

      if(hhwdth.ne.0.D0) then
      write(nout,99)
      write(nout,100) 35,hhwdth,'H decays'

      write(nout,101)
      if(hhbrb.ne.0.D0) then
      write(nout,102) hhbrb,2,ib,ibb        ,'BR(H -> b       bb     )'
      endif
      if(hhbrl.ne.0.D0) then
      write(nout,102) hhbrl,2,-itau,itau    ,'BR(H -> tau+    tau-   )'
      endif
      if(hhbrm.ne.0.D0) then
      write(nout,102) hhbrm,2,-imu,imu      ,'BR(H -> mu+     mu-    )'
      endif
      if(hhbrs.ne.0.D0) then
      write(nout,102) hhbrs,2,is,isb        ,'BR(H -> s       sb     )'
      endif
      if(hhbrc.ne.0.D0) then
      write(nout,102) hhbrc,2,ic,icb        ,'BR(H -> c       cb     )'
      endif
      if(hhbrt.ne.0.D0) then
      write(nout,102) hhbrt,2,it,itb        ,'BR(H -> t       tb     )' 
      endif
      if(hhbrg.ne.0.D0) then
      write(nout,102) hhbrg,2,igl,igl       ,'BR(H -> g       g      )' 
      endif
      if(hhbrga.ne.0.D0) then
      write(nout,102) hhbrga,2,iga,iga      ,'BR(H -> gam     gam    )' 
      endif
      if(hhbrzga.ne.0.D0) then
      write(nout,102) hhbrzga,2,iz,iga      ,'BR(H -> Z       gam    )' 
      endif
      if(hhbrw.ne.0.D0) then
      write(nout,102) hhbrw,2,iwc,-iwc      ,'BR(H -> W+      W-     )' 
      endif
      if(hhbrz.ne.0.D0) then
      write(nout,102) hhbrz,2,iz,iz         ,'BR(H -> Z       Z      )' 
      endif
      if(hhbrh.ne.0.D0) then
      write(nout,102) hhbrh,2,ihl,ihl       ,'BR(H -> h       h      )' 
      endif
      if(hhbra.ne.0.D0) then
      write(nout,102) hhbra,2,iha,iha       ,'BR(H -> A       A      )' 
      endif
      if(hhbraz.ne.0.D0) then
      write(nout,102) hhbraz,2,iz,iha       ,'BR(H -> Z       A      )' 
      endif
      if(hhbrhw.ne.0.D0) then
      write(nout,102) hhbrhw/2.D0,2,iwc,-ihc,'BR(H -> W+      H-     )'
      write(nout,102) hhbrhw/2.D0,2,-iwc,ihc,'BR(H -> W-      H+     )'
      endif
      if(hhbrchch.ne.0.D0) then
      write(nout,102) hhbrchch,2,ihc,-ihc,   'BR(H -> H+      H-     )'
      endif
      if(hhbrsc(1,1).ne.0.D0) then
      write(nout,102) hhbrsc(1,1),2,ic1,-ic1,'BR(H -> ~chi_1+ ~chi_1-)' 
      endif
      if(hhbrsc(2,2).ne.0.D0) then
      write(nout,102) hhbrsc(2,2),2,ic2,-ic2,'BR(H -> ~chi_2+ ~chi_2-)' 
      endif
      if(hhbrsc(1,2).ne.0.D0) then
      write(nout,102) hhbrsc(1,2),2,ic1,-ic2,'BR(H -> ~chi_1+ ~chi_2-)' 
      endif
      if(hhbrsc(2,1).ne.0.D0) then
      write(nout,102) hhbrsc(2,1),2,ic2,-ic1,'BR(H -> ~chi_2+ ~chi_1-)' 
      endif
      if(hhbrsn(1,1).ne.0.D0) then
      write(nout,102) hhbrsn(1,1),2,in1,in1 ,'BR(H -> ~chi_10 ~chi_10)' 
      endif
      if(hhbrsn(2,2).ne.0.D0) then
      write(nout,102) hhbrsn(2,2),2,in2,in2 ,'BR(H -> ~chi_20 ~chi_20)' 
      endif
      if(hhbrsn(3,3).ne.0.D0) then
      write(nout,102) hhbrsn(3,3),2,in3,in3 ,'BR(H -> ~chi_30 ~chi_30)' 
      endif
      if(hhbrsn(4,4).ne.0.D0) then
      write(nout,102) hhbrsn(4,4),2,in4,in4 ,'BR(H -> ~chi_40 ~chi_40)' 
      endif
      hhbrsn1(1,2) = 2.D0*hhbrsn(1,2) 
      hhbrsn1(1,3) = 2.D0*hhbrsn(1,3) 
      hhbrsn1(1,4) = 2.D0*hhbrsn(1,4)
      hhbrsn1(2,3) = 2.D0*hhbrsn(2,3) 
      hhbrsn1(2,4) = 2.D0*hhbrsn(2,4)  
      hhbrsn1(3,4) = 2.D0*hhbrsn(3,4)  
      if(hhbrsn1(1,2).ne.0.D0) then
      write(nout,102) hhbrsn1(1,2),2,in1,in2,'BR(H -> ~chi_10 ~chi_20)' 
      endif
      if(hhbrsn1(1,3).ne.0.D0) then
      write(nout,102) hhbrsn1(1,3),2,in1,in3,'BR(H -> ~chi_10 ~chi_30)' 
      endif
      if(hhbrsn1(1,4).ne.0.D0) then
      write(nout,102) hhbrsn1(1,4),2,in1,in4,'BR(H -> ~chi_10 ~chi_40)' 
      endif
      if(hhbrsn1(2,3).ne.0.D0) then
      write(nout,102) hhbrsn1(2,3),2,in2,in3,'BR(H -> ~chi_20 ~chi_30)' 
      endif
      if(hhbrsn1(2,4).ne.0.D0) then
      write(nout,102) hhbrsn1(2,4),2,in2,in4,'BR(H -> ~chi_20 ~chi_40)' 
      endif
      if(hhbrsn1(3,4).ne.0.D0) then
      write(nout,102) hhbrsn1(3,4),2,in3,in4,'BR(H -> ~chi_30 ~chi_40)' 
      endif
      bhhslnl1 = bhhslnl/3.D0
      bhhslel1 = bhhslel/2.D0
      bhhsler1 = bhhsler/2.D0
      bhhsqul1 = bhhsqul/2.d0
      bhhsqur1 = bhhsqur/2.d0
      bhhsqdl1 = bhhsqdl/2.d0
      bhhsqdr1 = bhhsqdr/2.d0
      if(bhhsqul1.ne.0.D0) then
      write(nout,102) bhhsqul1,2,isul,-isul  ,'BR(H -> ~u_L    ~u_L*  )'
      endif
      if(bhhsqur1.ne.0.D0) then
      write(nout,102) bhhsqur1,2,isur,-isur  ,'BR(H -> ~u_R    ~u_R*  )'
      endif
      if(bhhsqul1.ne.0.D0) then
      write(nout,102) bhhsqul1,2,iscl,-iscl  ,'BR(H -> ~c_L    ~c_L*  )'
      endif
      if(bhhsqur1.ne.0.D0) then
      write(nout,102) bhhsqur1,2,iscr,-iscr  ,'BR(H -> ~c_R    ~c_R*  )'
      endif
      if(bhhst(1,1).ne.0.D0) then
      write(nout,102) bhhst(1,1),2,ist1,-ist1,'BR(H -> ~t_1    ~t_1*  )'
      endif
      if(bhhst(2,2).ne.0.D0) then
      write(nout,102) bhhst(2,2),2,ist2,-ist2,'BR(H -> ~t_2    ~t_2*  )'
      endif
      if(bhhst(1,2).ne.0.D0) then
      write(nout,102) bhhst(1,2),2,ist1,-ist2,'BR(H -> ~t_1    ~t_2*  )'
      endif
      if(bhhst(2,1).ne.0.D0) then
      write(nout,102) bhhst(2,1),2,ist2,-ist1,'BR(H -> ~t_2    ~t_1*  )'
      endif
      if(bhhsqdl1.ne.0.D0) then
      write(nout,102) bhhsqdl1,2,isdl,-isdl  ,'BR(H -> ~d_L    ~d_L*  )'
      endif
      if(bhhsqdr1.ne.0.D0) then
      write(nout,102) bhhsqdr1,2,isdr,-isdr  ,'BR(H -> ~d_R    ~d_R*  )'
      endif
      if(bhhsqdl1.ne.0.D0) then
      write(nout,102) bhhsqdl1,2,issl,-issl  ,'BR(H -> ~s_L    ~s_L*  )'
      endif
      if(bhhsqdr1.ne.0.D0) then
      write(nout,102) bhhsqdr1,2,issr,-issr  ,'BR(H -> ~s_R    ~s_R*  )'
      endif
      if(bhhsb(1,1).ne.0.D0) then
      write(nout,102) bhhsb(1,1),2,isb1,-isb1,'BR(H -> ~b_1    ~b_1*  )'
      endif
      if(bhhsb(2,2).ne.0.D0) then
      write(nout,102) bhhsb(2,2),2,isb2,-isb2,'BR(H -> ~b_2    ~b_2*  )'
      endif
      if(bhhsb(1,2).ne.0.D0) then
      write(nout,102) bhhsb(1,2),2,isb1,-isb2,'BR(H -> ~b_1    ~b_2*  )'
      endif
      if(bhhsb(2,1).ne.0.D0) then
      write(nout,102) bhhsb(2,1),2,isb2,-isb1,'BR(H -> ~b_2    ~b_1*  )'
      endif
      if(bhhslel1.ne.0.D0) then
      write(nout,102) bhhslel1,2,isell,-isell,'BR(H -> ~e_L-   ~e_L+  )'
      endif
      if(bhhsler1.ne.0.D0) then
      write(nout,102) bhhsler1,2,iselr,-iselr,'BR(H -> ~e_R-   ~e_R+  )'
      endif
      if(bhhslel1.ne.0.D0) then
      write(nout,102) bhhslel1,2,ismul,-ismul,'BR(H -> ~mu_L-  ~mu_L+ )'
      endif
      if(bhhsler1.ne.0.D0) then
      write(nout,102) bhhsler1,2,ismur,-ismur,'BR(H -> ~mu_R-  ~mu_R+ )'
      endif
      if(bhhstau(1,1).ne.0.D0) then
      write(nout,102) bhhstau(1,1),2,istau1,-istau1,'BR(H -> ~tau_1- ~ta
     .u_1+)'
      endif
      if(bhhstau(2,2).ne.0.D0) then
      write(nout,102) bhhstau(2,2),2,istau2,-istau2,'BR(H -> ~tau_2- ~ta
     .u_2+)'
      endif
      if(bhhstau(1,2).ne.0.D0) then
      write(nout,102) bhhstau(1,2),2,istau1,-istau2,'BR(H -> ~tau_1- ~ta
     .u_2+)'
      endif
      if(bhhstau(2,1).ne.0.D0) then
      write(nout,102) bhhstau(2,1),2,istau2,-istau1,'BR(H -> ~tau_2- ~ta
     .u_1+)'
      endif
      if(bhhslnl1.ne.0.D0) then
      write(nout,102) bhhslnl1,2,inel,-inel  ,'BR(H -> ~nu_eL  ~nu_eL*  
     . )'
      write(nout,102) bhhslnl1,2,inmul,-inmul,'BR(H -> ~nu_muL ~nu_muL* 
     . )'
      write(nout,102) bhhslnl1,2,intau1,-intau1,'BR(H -> ~nu_tauL ~nu_ta
     .uL*)'
      endif

      elseif(hhwdth.eq.0.D0) then
      write(nout,99)
      write(nout,100) 35,0.000000000E+00,'H decays'

      endif
      endif

      if(ihiggs.eq.3.or.ihiggs.eq.5) then
            write(nout,105)

      if(awdth.ne.0.D0) then
      write(nout,99)
      write(nout,100) 36,awdth,'A decays'

      write(nout,101)
      if(abrb.ne.0.D0) then
      write(nout,102) abrb,2,ib,ibb         ,'BR(A -> b       bb     )'
      endif
      if(abrl.ne.0.D0) then
      write(nout,102) abrl,2,-itau,itau     ,'BR(A -> tau+    tau-   )'
      endif
      if(abrm.ne.0.D0) then
      write(nout,102) abrm,2,-imu,imu       ,'BR(A -> mu+     mu-    )'
      endif
      if(abrs.ne.0.D0) then
      write(nout,102) abrs,2,is,isb         ,'BR(A -> s       sb     )'
      endif
      if(abrc.ne.0.D0) then
      write(nout,102) abrc,2,ic,icb         ,'BR(A -> c       cb     )'
      endif
      if(abrt.ne.0.D0) then
      write(nout,102) abrt,2,it,itb         ,'BR(A -> t       tb     )' 
      endif
      if(abrg.ne.0.D0) then
      write(nout,102) abrg,2,igl,igl        ,'BR(A -> g       g      )' 
      endif
      if(abrga.ne.0.D0) then
      write(nout,102) abrga,2,iga,iga       ,'BR(A -> gam     gam    )' 
      endif
      if(abrzga.ne.0.D0) then
      write(nout,102) abrzga,2,iz,iga       ,'BR(A -> Z       gam    )' 
      endif
      if(abrz.ne.0.D0) then
      write(nout,102) abrz,2,iz,ihl         ,'BR(A -> Z       h      )' 
      endif
      if(habrsc(1,1).ne.0.D0) then
      write(nout,102) habrsc(1,1),2,ic1,-ic1,'BR(A -> ~chi_1+ ~chi_1-)' 
      endif
      if(habrsc(2,2).ne.0.D0) then
      write(nout,102) habrsc(2,2),2,ic2,-ic2,'BR(A -> ~chi_2+ ~chi_2-)' 
      endif
      if(habrsc(1,2).ne.0.D0) then
      write(nout,102) habrsc(1,2),2,ic1,-ic2,'BR(A -> ~chi_1+ ~chi_2-)' 
      endif
      if(habrsc(2,1).ne.0.D0) then
      write(nout,102) habrsc(2,1),2,ic2,-ic1,'BR(A -> ~chi_2+ ~chi_1-)' 
      endif
      habrsn1(1,2) = 2.D0*habrsn(1,2) 
      habrsn1(1,3) = 2.D0*habrsn(1,3) 
      habrsn1(1,4) = 2.D0*habrsn(1,4)
      habrsn1(2,3) = 2.D0*habrsn(2,3) 
      habrsn1(2,4) = 2.D0*habrsn(2,4)  
      habrsn1(3,4) = 2.D0*habrsn(3,4)  
      if(habrsn(1,1).ne.0.D0) then
      write(nout,102) habrsn(1,1),2,in1,in1 ,'BR(A -> ~chi_10 ~chi_10)' 
      endif
      if(habrsn(2,2).ne.0.D0) then
      write(nout,102) habrsn(2,2),2,in2,in2 ,'BR(A -> ~chi_20 ~chi_20)' 
      endif
      if(habrsn(3,3).ne.0.D0) then
      write(nout,102) habrsn(3,3),2,in3,in3 ,'BR(A -> ~chi_30 ~chi_30)' 
      endif
      if(habrsn(4,4).ne.0.D0) then
      write(nout,102) habrsn(4,4),2,in4,in4 ,'BR(A -> ~chi_40 ~chi_40)' 
      endif
      if(habrsn1(1,2).ne.0.D0) then
      write(nout,102) habrsn1(1,2),2,in1,in2,'BR(A -> ~chi_10 ~chi_20)' 
      endif
      if(habrsn1(1,3).ne.0.D0) then
      write(nout,102) habrsn1(1,3),2,in1,in3,'BR(A -> ~chi_10 ~chi_30)' 
      endif
      if(habrsn1(1,4).ne.0.D0) then
      write(nout,102) habrsn1(1,4),2,in1,in4,'BR(A -> ~chi_10 ~chi_40)' 
      endif
      if(habrsn1(2,3).ne.0.D0) then
      write(nout,102) habrsn1(2,3),2,in2,in3,'BR(A -> ~chi_20 ~chi_30)' 
      endif
      if(habrsn1(2,4).ne.0.D0) then
      write(nout,102) habrsn1(2,4),2,in2,in4,'BR(A -> ~chi_20 ~chi_40)' 
      endif
      if(habrsn1(3,4).ne.0.D0) then
      write(nout,102) habrsn1(3,4),2,in3,in4,'BR(A -> ~chi_30 ~chi_40)' 
      endif
      if(habrst.ne.0.D0) then
      write(nout,102) habrst/2.D0,2,ist1,-ist2,'BR(A -> ~t_1    ~t_2*  )
     .'
      write(nout,102) habrst/2.D0,2,-ist1,ist2,'BR(A -> ~t_1*   ~t_2   )
     .'
      endif
      if(habrsb.ne.0.D0) then
      write(nout,102) habrsb/2.D0,2,isb1,-isb2,'BR(A -> ~b_1    ~b_2*  )
     .'
      write(nout,102) habrsb/2.D0,2,-isb1,isb2,'BR(A -> ~b_1*   ~b_2   )
     .'
      endif
      if(habrsl.ne.0.D0) then
      write(nout,102) habrsl/2.D0,2,istau1,-istau2,'BR(A -> ~tau_1- ~tau
     ._2+)'
      write(nout,102) habrsl/2.D0,2,-istau1,istau2,'BR(A -> ~tau_1+ ~tau
     ._2-)'
      endif

      elseif(awdth.eq.0.D0) then
      write(nout,99)
      write(nout,100) 36,0.000000000E+00,'A decays'

      endif
      endif

      if(ihiggs.eq.4.or.ihiggs.eq.5) then
            write(nout,105)

      if(hcwdth.ne.0.D0) then
      write(nout,99)
      write(nout,100) 37,hcwdth,'H+ decays'

      write(nout,101)
      if(hcbrb.ne.0.D0) then
      write(nout,102) hcbrb,2,ic,ibb        ,'BR(H+ -> c       bb     )'
      endif
      if(hcbrl.ne.0.D0) then
      write(nout,102) hcbrl,2,-itau,intau   ,'BR(H+ -> tau+    nu_tau )'
      endif
      if(hcbrm.ne.0.D0) then
      write(nout,102) hcbrm,2,-imu,inmu     ,'BR(H+ -> mu+     nu_mu  )'
      endif
      if(hcbrbu.ne.0.D0) then
      write(nout,102) hcbrbu,2,iu,ibb       ,'BR(H+ -> u       bb     )'
      endif
      if(hcbrs.ne.0.D0) then
      write(nout,102) hcbrs,2,iu,isb        ,'BR(H+ -> u       sb     )'
      endif
      if(hcbrcd.ne.0.D0) then
      write(nout,102) hcbrcd,2,ic,idb       ,'BR(H+ -> c       db     )'
      endif
      if(hcbrc.ne.0.D0) then
      write(nout,102) hcbrc,2,ic,isb        ,'BR(H+ -> c       sb     )'
      endif
      if(hcbrt.ne.0.D0) then
      write(nout,102) hcbrt,2,it,ibb        ,'BR(H+ -> t       bb     )'
      endif
      if(hcbrts.ne.0.D0) then
      write(nout,102) hcbrts,2,it,isb       ,'BR(H+ -> t       sb     )'
      endif
      if(hcbrtd.ne.0.D0) then
      write(nout,102) hcbrtd,2,it,idb       ,'BR(H+ -> t       db     )'
      endif
      if(hcbrw.ne.0.D0) then
      write(nout,102) hcbrw,2,iwc,ihl       ,'BR(H+ -> W+      h      )'
      endif
      if(hcbrwhh.ne.0.D0) then
      write(nout,102) hcbrwhh,2,iwc,ihh     ,'BR(H+ -> W+      H      )'
      endif
      if(hcbra.ne.0.D0) then
      write(nout,102) hcbra,2,iwc,iha       ,'BR(H+ -> W+      A      )'
      endif
      if(hcbrsu(1,1).ne.0.D0) then
      write(nout,102) hcbrsu(1,1),2,ic1,in1 ,'BR(H+ -> ~chi_1+ ~chi_10)'
      endif
      if(hcbrsu(1,2).ne.0.D0) then
      write(nout,102) hcbrsu(1,2),2,ic1,in2 ,'BR(H+ -> ~chi_1+ ~chi_20)'
      endif
      if(hcbrsu(1,3).ne.0.D0) then
      write(nout,102) hcbrsu(1,3),2,ic1,in3 ,'BR(H+ -> ~chi_1+ ~chi_30)'
      endif
      if(hcbrsu(1,4).ne.0.D0) then
      write(nout,102) hcbrsu(1,4),2,ic1,in4 ,'BR(H+ -> ~chi_1+ ~chi_40)'
      endif
      if(hcbrsu(2,1).ne.0.D0) then
      write(nout,102) hcbrsu(2,1),2,ic2,in1 ,'BR(H+ -> ~chi_2+ ~chi_10)'
      endif
      if(hcbrsu(2,2).ne.0.D0) then
      write(nout,102) hcbrsu(2,2),2,ic2,in2 ,'BR(H+ -> ~chi_2+ ~chi_20)'
      endif
      if(hcbrsu(2,3).ne.0.D0) then
      write(nout,102) hcbrsu(2,3),2,ic2,in3 ,'BR(H+ -> ~chi_2+ ~chi_30)'
      endif
      if(hcbrsu(2,4).ne.0.D0) then
      write(nout,102) hcbrsu(2,4),2,ic2,in4 ,'BR(H+ -> ~chi_2+ ~chi_40)'
      endif
      bhcsl02=bhcsl00/2.D0
      if(bhcsl02.ne.0.D0) then
      write(nout,102) bhcsl02,2,-isell,inel ,'BR(H+ -> ~e_L+   ~nu_eL )'
      write(nout,102) bhcsl02,2,-ismul,inmul,'BR(H+ -> ~mu_L+  ~nu_muL)'
      endif
      if(bhcsl11.ne.0.D0) then
      write(nout,102) bhcsl11,2,-istau1,intau1,'BR(H+ -> ~tau_1+ ~nu_tau
     .L)'
      endif
      if(bhcsl21.ne.0.D0) then
      write(nout,102) bhcsl21,2,-istau2,intau1,'BR(H+ -> ~tau_2+ ~nu_tau
     .L)'
      endif
      hcbrsq1=hcbrsq/2.D0
      if(hcbrsq1.ne.0.D0) then
      write(nout,102) hcbrsq1,2,isul,-isdl  ,'BR(H+ -> ~u_L    ~d_L*  )'
      write(nout,102) hcbrsq1,2,iscl,-issl  ,'BR(H+ -> ~c_L    ~s_L*  )'
      endif
      if(hcbrstb(1,1).ne.0.D0) then
      write(nout,102) hcbrstb(1,1),2,ist1,-isb1,'BR(H+ -> ~t_1    ~b_1* 
     . )'
      endif
      if(hcbrstb(2,2).ne.0.D0) then
      write(nout,102) hcbrstb(2,2),2,ist2,-isb2,'BR(H+ -> ~t_2    ~b_2* 
     . )'
      endif
      if(hcbrstb(1,2).ne.0.D0) then
      write(nout,102) hcbrstb(1,2),2,ist1,-isb2,'BR(H+ -> ~t_1    ~b_2* 
     . )'
      endif
      if(hcbrstb(2,1).ne.0.D0) then
      write(nout,102) hcbrstb(2,1),2,ist2,-isb1,'BR(H+ -> ~t_2    ~b_1* 
     . )'
      endif

      elseif(hcwdth.eq.0.D0) then
      write(nout,99)
      write(nout,100) 37,0.000000000E+00,'H+ decays'

      endif
      endif

      if(ihiggs.ne.0) then
            write(nout,105)

      if(hcwdth.ne.0.D0) then
      write(nout,99)
      write(nout,100) 6,gamt1,'top decays'

      write(nout,101)
      if(gamt0.ne.0.D0) then
       write(nout,102) gamt0/gamt1,2,ib,iwc,'BR(t -> b       W+    )'
      endif
      if(gamt1-gamt0.ne.0.D0) then
       write(nout,102) (gamt1-gamt0)/gamt1,2,ib,ihc,
     .                                      'BR(t -> b       H+    )'
      endif

      elseif(gamt1.eq.0.D0) then
      write(nout,99)
      write(nout,100) 6,0.000000000E+00,'top decays'

      endif
      endif

 49   format('#',1x,A,E16.8)
 50   format('#',1x,A)
 51   format('BLOCK',1x,A,2x,'#',1x,A)
 551  format(1x,A,2x,'#',1x,A)
 52   format(1x,I9,3x,1P,E16.8,0P,3x,'#',1x,A)
 552  format(2x,E16.8,0P,3x,A)
 53   format(1x,I2,1x,I2,3x,1P,E16.8,0P,3x,'#',1x,A)
 54   format('BLOCK',1x,A,1P,E16.8,2x,'#',1x,A)
 554  format(2x,A,1P,E16.8,2x,1x,A)
 55   format(1x,I5,3x,1P,E16.8,0P,3x,'#',1x,A)
 56   format(1x,I4,3x,'#',1x,A,E16.8)
 57   format(1x,I5,3x,1P,E16.8,0P,3x,'#',1x,A,E16.8)
 58   format(1x,I2,1x,I2,3x,'#',1x,A)
 59   format(1x,I2,1x,I2,3x,1P,E16.8,0P,3x,'#',1x,A,E16.8)
 60   format(9x,1P,E16.8,0P,3x,'#',1x,A)
 61   format(1x,I5,3x,A)
 661  format(2x,A)
 62   format(1x,I5,1x,I5,3x,A)
 662  format(2x,A)
 63   format(1x,I5,3x,A,1x,'#',1x,A)

 99   format('#',9x,'PDG',12x,'Width')
 100  format('DECAY',1x,I9,3x,1P,E16.8,0P,3x,'#',1x,A)
 101  format('#',10x,'BR',9x,'NDA',6x,'ID1',7x,'ID2')
 102  format(3x,1P,E16.8,0P,3x,I2,3x,(I9,1x),(I9,1x),2x,'#',1x,A)
 103  format('#',11x,'BR',9x,'NDA',6x,'ID1',7x,'ID2',7x,'ID3')
 107  format('#',11x,'BR',9x,'NDA',6x,'ID1',7x,'ID2',7x,'ID3',7x,'ID4')
 104  format(3x,1P,E16.8,0P,3x,I2,3x,(I9,1x),(I9,1x),(I9,1x),2x,'#',
     .1x,A)
 106  format(3x,1P,E16.8,0P,3x,I2,3x,(I9,1x),(I9,1x),(I9,1x),(I9,1x),
     .2x,'#',1x,A)
 105  format('#') 

       close(nout)

      else

      IF(IHIGGS.EQ.0)THEN
      WRITE(NSA,20)AMSM,SMBRB,SMBRL,SMBRM,SMBRS,SMBRC,SMBRT
      WRITE(NSB,20)AMSM,SMBRG,SMBRGA,SMBRZGA,SMBRW,SMBRZ,SMWDTH
      IF(ISM4.NE.0)THEN
       WRITE(NSC,20)AMSM,SMBRNUP,SMBREP,SMBRBP,SMBRTP
      ENDIF
      ENDIF

      IF(IHIGGS.NE.0)THEN
       WRITE(NTA,23)AMCH,GAMT0/GAMT1,(GAMT1-GAMT0)/GAMT1,GAMT1
      ENDIF

      IF(IHIGGS.EQ.1.OR.IHIGGS.EQ.5)THEN
      WRITE(NLA,20)AML,HLBRB,HLBRL,HLBRM,HLBRS,HLBRC,HLBRT
c MMM changed 23/8/2013
      if(i2hdm.eq.0) then
      WRITE(NLB,20)AML,HLBRG,HLBRGA,HLBRZGA,HLBRW,HLBRZ,HLWDTH
      elseif(i2hdm.eq.1) then
      WRITE(NLB,22)AML,HLBRG,HLBRGA,HLBRZGA,HLBRW,HLBRZ
      WRITE(NLC,20)AML,HLBRA,HLBRAZ,HLBRHW,HLBRCHCH,HLWDTH
      endif
c end MMM changed 23/8/2013
      IF(IOFSUSY.EQ.0)THEN 
       WRITE(NSUSYL,22)AML,HLBRCHT,HLBRNET,HLBRSL,HLBRSQT,HLBRGD
       IF(INDIDEC.NE.0)THEN
        WRITE(NSUSYLA,23)AML,HLBRSC(1,1),HLBRSC(2,2),
     .                   HLBRSC(1,2)+HLBRSC(2,1)
        WRITE(NSUSYLB,21)AML,HLBRSN(1,1),HLBRSN(2,2),HLBRSN(3,3),
     .                   HLBRSN(4,4)
        WRITE(NSUSYLC,20)AML,HLBRSN(1,2)+HLBRSN(2,1),
     .                   HLBRSN(1,3)+HLBRSN(3,1),
     .                   HLBRSN(1,4)+HLBRSN(4,1),
     .                   HLBRSN(2,3)+HLBRSN(3,2),
     .                   HLBRSN(2,4)+HLBRSN(4,2),
     .                   HLBRSN(3,4)+HLBRSN(4,3)
        WRITE(NSUSYLD,20)AML,BHLSLNL,BHLSLEL,BHLSLER,BHLSTAU(1,1),
     .                   BHLSTAU(1,2)+BHLSTAU(2,1),BHLSTAU(2,2)
        WRITE(NSUSYLE,21)AML,BHLSQUL,BHLSQUR,BHLSQDL,BHLSQDR
      WRITE(NSUSYLF,20)AML,BHLSB(1,1),BHLSB(1,2)+BHLSB(2,1),BHLSB(2,2),
     .                   BHLST(1,1),BHLST(1,2)+BHLST(2,1),BHLST(2,2)
       ENDIF 
      ENDIF
      ENDIF

      IF(IHIGGS.EQ.2.OR.IHIGGS.EQ.5)THEN
      WRITE(NHA,20)AMH,HHBRB,HHBRL,HHBRM,HHBRS,HHBRC,HHBRT
      WRITE(NHB,20)AMH,HHBRG,HHBRGA,HHBRZGA,HHBRW,HHBRZ
      WRITE(NHC,20)AMH,HHBRH,HHBRA,HHBRAZ,HHBRHW,HHBRCHCH,HHWDTH
      IF(IOFSUSY.EQ.0)THEN 
       WRITE(NSUSYH,22)AMH,HHBRCHT,HHBRNET,HHBRSL,HHBRSQT,HHBRGD
       IF(INDIDEC.NE.0)THEN
        WRITE(NSUSYHA,23)AMH,HHBRSC(1,1),HHBRSC(2,2),
     .                  HHBRSC(1,2)+HHBRSC(2,1)
        WRITE(NSUSYHB,21)AMH,HHBRSN(1,1),HHBRSN(2,2),HHBRSN(3,3),
     .                   HHBRSN(4,4)
        WRITE(NSUSYHC,20)AMH,HHBRSN(1,2)+HHBRSN(2,1),
     .                   HHBRSN(1,3)+HHBRSN(3,1),
     .                   HHBRSN(1,4)+HHBRSN(4,1),
     .                   HHBRSN(2,3)+HHBRSN(3,2),
     .                   HHBRSN(2,4)+HHBRSN(4,2),
     .                   HHBRSN(3,4)+HHBRSN(4,3)
        WRITE(NSUSYHD,20)AMH,BHHSLNL,BHHSLEL,BHHSLER,BHHSTAU(1,1),
     .                   BHHSTAU(1,2)+BHHSTAU(2,1),BHHSTAU(2,2)
        WRITE(NSUSYHE,21)AMH,BHHSQUL,BHHSQUR,BHHSQDL,BHHSQDR
      WRITE(NSUSYHF,20)AMH,BHHSB(1,1),BHHSB(1,2)+BHHSB(2,1),BHHSB(2,2),
     .                   BHHST(1,1),BHHST(1,2)+BHHST(2,1),BHHST(2,2)
       ENDIF
      ENDIF
      ENDIF

      IF(IHIGGS.EQ.3.OR.IHIGGS.EQ.5)THEN
      WRITE(NAA,20)AMA,ABRB,ABRL,ABRM,ABRS,ABRC,ABRT
c MMM changed 23/8/2013
      if(i2hdm.eq.0) then
      WRITE(NAB,22)AMA,ABRG,ABRGA,ABRZGA,ABRZ,AWDTH
      elseif(i2hdm.eq.1) then
      WRITE(NAB,22)AMA,ABRG,ABRGA,ABRZGA,ABRZ,ABRHHAZ
      WRITE(NAC,24)AMA,ABRHAWPHM,AWDTH
      endif
c end MMM changed 23/8/2013
      IF(IOFSUSY.EQ.0)THEN 
       WRITE(NSUSYA,22)AMA,HABRCHT,HABRNET,HABRSL,HABRST+HABRSB,HABRGD
       IF(INDIDEC.NE.0)THEN
        WRITE(NSUSYAA,23)AMA,HABRSC(1,1),HABRSC(2,2),
     .                   HABRSC(1,2)+HABRSC(2,1)
        WRITE(NSUSYAB,21)AMA,HABRSN(1,1),HABRSN(2,2),HABRSN(3,3),
     .                   HABRSN(4,4)
        WRITE(NSUSYAC,20)AMA,HABRSN(1,2)+HABRSN(2,1),
     .                   HABRSN(1,3)+HABRSN(3,1),
     .                   HABRSN(1,4)+HABRSN(4,1),
     .                   HABRSN(2,3)+HABRSN(3,2),
     .                   HABRSN(2,4)+HABRSN(4,2),
     .                   HABRSN(3,4)+HABRSN(4,3)
        WRITE(NSUSYAD,23)AMA,BHASTAU,BHASB,BHAST
       ENDIF
      ENDIF
      ENDIF

      IF(IHIGGS.EQ.4.OR.IHIGGS.EQ.5)THEN
      WRITE(NCA,20)AMCH,HCBRB,HCBRL,HCBRM,HCBRS,HCBRC,HCBRT
      WRITE(NCB,22)AMCH,HCBRCD,HCBRBU,HCBRTS,HCBRTD
c MMM changed 23/8/2013
      if(i2hdm.eq.0) then
      WRITE(NCC,22)AMCH,HCBRW,HCBRA,HCWDTH
      elseif(i2hdm.eq.1) then
      WRITE(NCC,20)AMCH,HCBRW,HCBRWHH,HCBRA,HCWDTH
      endif
c end MMM changed 23/8/2013
      IF(IOFSUSY.EQ.0)THEN 
       WRITE(NSUSYC,21)AMCH,HCBRCNT,HCBRSL,HCBRSQT,HCBRGD
       IF(INDIDEC.NE.0)THEN
        WRITE(NSUSYCA,21)AMCH,HCBRSU(1,1),HCBRSU(1,2),
     .                   HCBRSU(1,3),HCBRSU(1,4)
        WRITE(NSUSYCB,21)AMCH,HCBRSU(2,1),HCBRSU(2,2),
     .                   HCBRSU(2,3),HCBRSU(2,4)
        WRITE(NSUSYCC,23)AMCH,BHCSL00,BHCSL11,BHCSL21
        WRITE(NSUSYCD,22)AMCH,BHCSQ,BHCSTB(1,1),BHCSTB(1,2),
     .                   BHCSTB(2,1),BHCSTB(2,2)
       ENDIF
      ENDIF
      ENDIF

19    FORMAT(G12.6,7(1X,G10.4))
20    FORMAT(G12.6,6(1X,G10.4))
21    FORMAT(G12.6,4(1X,G10.4))
22    FORMAT(G12.6,5(1X,G10.4))
23    FORMAT(G12.6,3(1X,G10.4))
24    FORMAT(G12.6,2(1X,G10.4))
      endif

      RETURN
      END

      SUBROUTINE CLOSE_HDEC
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER(K=6,NI=87,NSA=85,NSB=86,NLA=88,NLB=89,NHA=90,NHB=91,
     .          NHC=92,NAA=93,NAB=94,NCA=95,NCB=96,NCC=50,NRA=97,NRB=98,
     .          NSUSYL=81,NSUSYA=82,NSUSYH=83,NSUSYC=84,NPAR=80,NTA=99,
     .          NSUSYLA=79,NSUSYLB=78,NSUSYLC=77,NSUSYLD=76,NSUSYLE=75,
     .          NSUSYLF=59,NSUSYHF=58,NSC=57,
     .          NSUSYHA=74,NSUSYHB=73,NSUSYHC=72,NSUSYHD=71,NSUSYHE=70,
     .          NSUSYAA=69,NSUSYAB=68,NSUSYAC=67,NSUSYAD=66,NSUSYAE=65,
     .          NSUSYCA=64,NSUSYCB=63,NSUSYCC=62,NSUSYCD=61,NSUSYCE=60)
      PARAMETER (NAC=31,NLC=32)
      DIMENSION GMN(4),XMN(4),GMC(2),GMST(2),GMSB(2),GMSL(2),
     .          GMSU(2),GMSD(2),GMSE(2),GMSN(2),GMSN1(2)
      DIMENSION HLBRSC(2,2),HLBRSN(4,4),HHBRSC(2,2),HHBRSN(4,4),
     .          HABRSC(2,2),HABRSN(4,4),HCBRSU(2,4),
     .          HHBRST(2,2),HHBRSB(2,2),HCBRSTB(2,2) 
      DIMENSION AC1(2,2),AC2(2,2),AC3(2,2),
     .          AN1(4,4),AN2(4,4),AN3(4,4),
     .          ACNL(2,4),ACNR(2,4)
      DIMENSION GLTT(2,2),GLBB(2,2),GHTT(2,2),GHBB(2,2),GCTB(2,2),
     .          GLEE(2,2),GHEE(2,2),GCEN(2,2)
      DIMENSION AGDL(4),AGDA(4),AGDH(4),AGDC(2)
      COMMON/MASSES_HDEC/AMS,AMC,AMB,AMT
      COMMON/STRANGE_HDEC/AMSB
      COMMON/PARAM_HDEC/GF,ALPH,AMTAU,AMMUON,AMZ,AMW
      COMMON/CKMPAR_HDEC/VTB,VTS,VTD,VCB,VCS,VCD,VUB,VUS,VUD
      COMMON/HMASS_HDEC/AMSM,AMA,AML,AMH,AMCH,AMAR
      COMMON/BREAK_HDEC/AMEL,AMER,AMSQ,AMUR,AMDR,AL,AU,AD,AMU,AM2
      COMMON/BREAKGLU_HDEC/AMGLU
      COMMON/SFER1ST_HDEC/AMQL1,AMUR1,AMDR1,AMEL1,AMER1
      COMMON/GLUINO_HDEC/AMGLUINO,XMSB1,XMSB2,STHB,CTHB,
     .              XLBB(2,2),XHBB(2,2),XABB(2,2),
     .              XMST1,XMST2,STHT,CTHT,
     .              XLTT(2,2),XHTT(2,2),XATT(2,2)
      COMMON/WZWDTH_HDEC/GAMC0,GAMT0,GAMT1,GAMW,GAMZ
      COMMON/COUP_HDEC/GAT,GAB,GLT,GLB,GHT,GHB,GZAH,GZAL,
     .            GHHH,GLLL,GHLL,GLHH,GHAA,GLAA,GLVV,GHVV,
     .            GLPM,GHPM,B,A
      COMMON/ALS_HDEC/XLAMBDA,AMC0,AMB0,AMT0,N0
      COMMON/FLAG_HDEC/IHIGGS,NNLO,IPOLE
      COMMON/MODEL_HDEC/IMODEL
      COMMON/ONSHELL_HDEC/IONSH,IONWZ,IOFSUSY
      COMMON/OLDFASH_HDEC/NFGG
      COMMON/WIDTHSM_HDEC/SMBRB,SMBRL,SMBRM,SMBRS,SMBRC,SMBRT,SMBRG,
     .               SMBRGA,SMBRZGA,SMBRW,SMBRZ,SMWDTH
      COMMON/WIDTHA_HDEC/ABRB,ABRL,ABRM,ABRS,ABRC,ABRT,ABRG,ABRGA,
     .               ABRZGA,ABRZ,AWDTH
      COMMON/WIDTHHL_HDEC/HLBRB,HLBRL,HLBRM,HLBRS,HLBRC,HLBRT,HLBRG,
     .               HLBRGA,HLBRZGA,HLBRW,HLBRZ,HLBRA,HLBRAZ,HLBRHW,
     .               HLWDTH
      COMMON/WIDTHHH_HDEC/HHBRB,HHBRL,HHBRM,HHBRS,HHBRC,HHBRT,HHBRG,
     .               HHBRGA,HHBRZGA,HHBRW,HHBRZ,HHBRH,HHBRA,HHBRAZ,
     .               HHBRHW,HHWDTH
      COMMON/WIDTHHC_HDEC/HCBRB,HCBRL,HCBRM,HCBRBU,HCBRS,HCBRC,HCBRT,
     .               HCBRW,HCBRA,HCWDTH
      COMMON/WISUSY_HDEC/HLBRSC,HLBRSN,HHBRSC,HHBRSN,HABRSC,HABRSN,
     .              HCBRSU,HLBRCHT,HHBRCHT,HABRCHT,HLBRNET,HHBRNET,
     .              HABRNET,HCBRCNT,HLBRSL,HHBRSL,HCBRSL,HABRSL,HABRST,
     .              HABRSB,HHBRSQ,HHBRST,HHBRSB,HHBRSQT,HCBRSQ,HCBRSTB,
     .              HCBRSQT,HLBRSQ,HLBRSQT
      COMMON/WISFER_HDEC/BHLSLNL,BHLSLEL,BHLSLER,BHLSQUL,BHLSQUR,
     .              BHLSQDL,BHLSQDR,BHLST(2,2),BHLSB(2,2),BHLSTAU(2,2),
     .              BHHSLNL,BHHSLEL,BHHSLER,BHHSQUL,BHHSQUR,BHHSQDL,
     .              BHHSQDR,BHHST(2,2),BHHSB(2,2),BHHSTAU(2,2),
     .              BHASTAU,BHASB,BHAST,
     .              BHCSL00,BHCSL11,BHCSL21,BHCSQ,BHCSTB(2,2)
      COMMON/SMASS_HDEC/GMN,XMN,GMC,GMST,GMSB,GMSL,GMSU,GMSD,GMSE,GMSN 
     .                 ,GMSN1
      COMMON/GOLDST_HDEC/AXMPL,AXMGD,IGOLD
      COMMON/WIGOLD_HDEC/HLBRGD,HABRGD,HHBRGD,HCBRGD
      COMMON/THDM_HDEC/TGBET2HDM,ALPH2HDM,AMHL2HDM,AMHH2HDM,
     .     AMHA2HDM,AMHC2HDM,AM12SQ,A1LAM2HDM,A2LAM2HDM,A3LAM2HDM,
     .     A4LAM2HDM,A5LAM2HDM,ITYPE2HDM,I2HDM,IPARAM2HDM

      IF(IHIGGS.EQ.0) THEN
       CLOSE(NSA)
       CLOSE(NSB)
       IF(ISM4.NE.0)THEN
        CLOSE(NSC)
       ENDIF
      ENDIF

      IF(IHIGGS.NE.0) THEN
       CLOSE(NTA)
      ENDIF

      IF(IHIGGS.EQ.1.OR.IHIGGS.EQ.5) THEN
       CLOSE(NLA)
       CLOSE(NLB) 
       IF(I2HDM.NE.0) CLOSE(NLC)
       CLOSE(NSUSYL)
      ENDIF

      IF(IHIGGS.EQ.2.OR.IHIGGS.EQ.5) THEN
       CLOSE(NHA)
       CLOSE(NHB) 
       CLOSE(NHC)
       CLOSE(NSUSYH)
      ENDIF

      IF(IHIGGS.EQ.3.OR.IHIGGS.EQ.5) THEN
       CLOSE(NAA)
       CLOSE(NAB) 
      if(i2hdm.eq.1) then
         close(nac)
      endif
       CLOSE(NSUSYA)
      ENDIF

      IF(IHIGGS.EQ.4.OR.IHIGGS.EQ.5) THEN
       CLOSE(NCA)
       CLOSE(NCB) 
       CLOSE(NCC) 
       CLOSE(NSUSYC)
      ENDIF

      RETURN
      END

C =====================================================================
C =========== BEGINNING OF THE SUBROUTINE FOR THE DECAYS ==============
C !!!!!!!!!!!!!! Any change below this line is at your own risk!!!!!!!!
C =====================================================================

      SUBROUTINE HDEC(TGBET)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DOUBLE PRECISION LAMB_HDEC
      COMPLEX*16 CFACQ_HDEC,CFACSQ_HDEC,CKOFQ_HDEC,CKOFQ,CFACQ,CFACQ0
      integer ivegas(4)
      DIMENSION XX(4),YY(4)
      DIMENSION AMCHAR(2),AMNEUT(4),XMNEUT(4),
     .          AC1(2,2),AC2(2,2),AC3(2,2),
     .          AN1(4,4),AN2(4,4),AN3(4,4),
     .          ACNL(2,4),ACNR(2,4),
     .          AMST(2),AMSB(2),AMSL(2),
     .          AMSU(2),AMSD(2),AMSE(2),AMSN(2),AMSN1(2),
     .          GLTT(2,2),GLBB(2,2),GLEE(2,2),
     .          GHTT(2,2),GHBB(2,2),GHEE(2,2),
     .          GCTB(2,2),GCEN(2,2)
      DIMENSION GMST(2),GMSB(2),GMSL(2),GMSU(2),GMSD(2),GMSE(2),
     .          GMSN(2),GMSN1(2)
      DIMENSION HLBRSC(2,2),HLBRSN(4,4),HHBRSC(2,2),
     .          HHBRSN(4,4),HABRSC(2,2),HABRSN(4,4),HCBRSU(2,4),
     .          HHBRST(2,2),HHBRSB(2,2),HCBRSTB(2,2) 
      DIMENSION WHLCH(2,2),WHLNE(4,4),WHHCH(2,2),WHHNE(4,4),
     .          WHACH(2,2),WHANE(4,4),WHCCN(2,4),
     .          WHHST(2,2),WHHSB(2,2),WHHSTAU(2,2),WHCSTB(2,2), 
     .          WHLST(2,2),WHLSB(2,2),WHLSTAU(2,2)
      DIMENSION WHHST0(2,2),WHHSB0(2,2)
      DIMENSION WHLGD(4),WHCGD(2),WHHGD(4),WHAGD(4)
      DIMENSION AGDL(4),AGDA(4),AGDH(4),AGDC(2)
      DIMENSION slhaneut(4),slhaxneut(4),slhachar(2),slhau(2,2),
     .          slhav(2,2),slhaz(4,4),xmchar(2)
      DIMENSION XGLBB(2,2),XGHBB(2,2),XGCTB(2,2)
      DIMENSION XHGG(3),XHQQ(3)
      COMPLEX*16 CF,CG,CI1,CI2,CA,CB,CTT,CTB,CTC,CTW,CLT,CLB,CLC,CLW,
     .           CAT,CAB,CAC,CAW,CAH,CTH,CLH,CX1,CX2,CAX1,CAX2,CTL,CAL,
     .           CSL,CSQ,CSB1,CSB2,CST1,CST2,CSL1,CSL2,
     .           CXL,CXQ,CXB1,CXB2,CXT1,CXT2,CXL1,CXL2
      COMPLEX*16 CSEL,CSER,CSUL,CSUR,CSDL,CSDR,
     .           CXEL,CXER,CXUL,CXUR,CXDL,CXDR
      COMPLEX*16 CAT0,CAB0,CAC0,CXUL0,CXUR0,CXDL0,CXDR0,CXB10,CXB20,
     .           CXT10,CXT20
      COMPLEX*16 CLE
      COMPLEX*16 CTTP,CTBP,CTEP,CLTP,CLBP,CLEP,CATP,CABP,CAEP
      COMPLEX*16 CATP0,CABP0
      COMPLEX*16 CAT00,CAB00,CAC00,CAL00,CAW00
      COMPLEX*16 CATP00,CABP00,CAEP00
      COMMON/HMASS_HDEC/AMSM,AMA,AML,AMH,AMCH,AMAR
      COMMON/HMASSR_HDEC/AMLR,AMHR
      COMMON/CHIMASS_HDEC/AMCHI
      COMMON/MASSES_HDEC/AMS,AMC,AMB,AMT
      COMMON/ALS_HDEC/XLAMBDA,AMC0,AMB0,AMT0,N0
      COMMON/PARAM_HDEC/GF,ALPH,AMTAU,AMMUON,AMZ,AMW
      COMMON/CKMPAR_HDEC/VTB,VTS,VTD,VCB,VCS,VCD,VUB,VUS,VUD
      COMMON/BREAK_HDEC/AMEL,AMER,AMSQ,AMUR,AMDR,AL,AU,AD,AMU,AM2
      COMMON/BREAKGLU_HDEC/AMGLU
      COMMON/WZWDTH_HDEC/GAMC0,GAMT0,GAMT1,GAMW,GAMZ
      COMMON/ONSHELL_HDEC/IONSH,IONWZ,IOFSUSY
      COMMON/OLDFASH_HDEC/NFGG
      COMMON/FLAG_HDEC/IHIGGS,NNLO,IPOLE
      COMMON/SM4_HDEC/AMTP,AMBP,AMNUP,AMEP,ISM4,IGGELW
      COMMON/FERMIOPHOBIC_HDEC/IFERMPHOB
      COMMON/WIDTHSM_HDEC/SMBRB,SMBRL,SMBRM,SMBRS,SMBRC,SMBRT,SMBRG,
     .               SMBRGA,SMBRZGA,SMBRW,SMBRZ,SMWDTH
      COMMON/WIDTHSM4_HDEC/SMBRNUP,SMBREP,SMBRBP,SMBRTP
      COMMON/WIDTHA_HDEC/ABRB,ABRL,ABRM,ABRS,ABRC,ABRT,ABRG,ABRGA,
     .              ABRZGA,ABRZ,AWDTH
      COMMON/WIDTHHL_HDEC/HLBRB,HLBRL,HLBRM,HLBRS,HLBRC,HLBRT,HLBRG,
     .               HLBRGA,HLBRZGA,HLBRW,HLBRZ,HLBRA,HLBRAZ,HLBRHW,
     .               HLWDTH
      COMMON/WIDTHHH_HDEC/HHBRB,HHBRL,HHBRM,HHBRS,HHBRC,HHBRT,HHBRG,
     .               HHBRGA,HHBRZGA,HHBRW,HHBRZ,HHBRH,HHBRA,HHBRAZ,
     .               HHBRHW,HHWDTH
      COMMON/WIDTHHC_HDEC/HCBRB,HCBRL,HCBRM,HCBRBU,HCBRS,HCBRC,HCBRT,
     .               HCBRW,HCBRA,HCWDTH
      COMMON/WISUSY_HDEC/HLBRSC,HLBRSN,HHBRSC,HHBRSN,HABRSC,HABRSN,
     .              HCBRSU,HLBRCHT,HHBRCHT,HABRCHT,HLBRNET,HHBRNET,
     .              HABRNET,HCBRCNT,HLBRSL,HHBRSL,HCBRSL,HABRSL,HABRST,
     .              HABRSB,HHBRSQ,HHBRST,HHBRSB,HHBRSQT,HCBRSQ,HCBRSTB,
     .              HCBRSQT,HLBRSQ,HLBRSQT
      COMMON/WISFER_HDEC/BHLSLNL,BHLSLEL,BHLSLER,BHLSQUL,BHLSQUR,
     .              BHLSQDL,BHLSQDR,BHLST(2,2),BHLSB(2,2),BHLSTAU(2,2),
     .              BHHSLNL,BHHSLEL,BHHSLER,BHHSQUL,BHHSQUR,BHHSQDL,
     .              BHHSQDR,BHHST(2,2),BHHSB(2,2),BHHSTAU(2,2),
     .              BHASTAU,BHASB,BHAST,
     .              BHCSL00,BHCSL11,BHCSL21,BHCSQ,BHCSTB(2,2)
      COMMON/SMASS_HDEC/AMNEUT,XMNEUT,AMCHAR,AMST,AMSB,AMSL,
     .              AMSU,AMSD,AMSE,AMSN,AMSN1
      COMMON/COUP_HDEC/GAT,GAB,GLT,GLB,GHT,GHB,GZAH,GZAL,
     .            GHHH,GLLL,GHLL,GLHH,GHAA,GLAA,GLVV,GHVV,
     .            GLPM,GHPM,B,A
      COMMON/GOLDST_HDEC/AXMPL,AXMGD,IGOLD
      COMMON/WIGOLD_HDEC/HLBRGD,HABRGD,HHBRGD,HCBRGD
      COMMON/SLHA_gaug_HDEC/slhaneut,slhaxneut,slhachar,slhau,slhav,
     .                      slhaz,xmchar
      COMMON/SLHA_vals_HDEC/islhai,islhao
      COMMON/BREAKSCALE_HDEC/SUSYSCALE
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      COMMON/DAVID/QSUSY1,QSUSY2,LOOP
      COMMON/SQNLO_HDEC/YMSB(2),STYB,CTYB,YLBB(2,2),YHBB(2,2),YABB,
     .                  YMST(2),STYT,CTYT,YLTT(2,2),YHTT(2,2),YATT
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      COMMON/CPSM_HDEC/CPW,CPZ,CPTAU,CPMU,CPT,CPB,CPC,CPS,
     .                 CPGAGA,CPGG,CPZGA,ICOUPELW
      COMMON/CPSM4_HDEC/CPTP,CPBP,CPNUP,CPEP
c MMM changed 21/8/13
      COMMON/THDM_HDEC/TGBET2HDM,ALPH2HDM,AMHL2HDM,AMHH2HDM,
     .     AMHA2HDM,AMHC2HDM,AM12SQ,A1LAM2HDM,A2LAM2HDM,A3LAM2HDM,
     .     A4LAM2HDM,A5LAM2HDM,ITYPE2HDM,I2HDM,IPARAM2HDM
      COMMON/THDM_COUP_HDEC/gllep,ghlep,galep
      COMMON/WIDTH_HC_ADD/hcbrcd,hcbrts,hcbrtd
      COMMON/WIDTH_2HDM/hcbrwhh,hhbrchch,hlbrchch,abrhhaz,abrhawphm
      common/DECPARAMETERS/amhi,amhj,amhk,gamtotj,gamtotk

      external hvhinteg
c end MMM changed 21/8/13
      HVV(X,Y)= GF/(4.D0*PI*DSQRT(2.D0))*X**3/2.D0*BETA_HDEC(Y)
     .            *(1.D0-4.D0*Y+12.D0*Y**2)
      AFF(X,Y)= GF/(4*PI*DSQRT(2.D0))*X**3*Y*(BETA_HDEC(Y))
      HFF(X,Y)= GF/(4*PI*DSQRT(2.D0))*X**3*Y*(BETA_HDEC(Y))**3
      CFF(Z,TB,X,Y)= GF/(4*PI*DSQRT(2.D0))*Z**3*LAMB_HDEC(X,Y)
     .              *((1.D0-X-Y)*(X*TB**2+Y/TB**2)-4.D0*X*Y)
      HV(V)=3.D0*(1.D0-8.D0*V+20.D0*V**2)/DSQRT((4.D0*V-1.D0))
     .      *DACOS((3.D0*V-1.D0)/2.D0/DSQRT(V**3))
     .      -(1.D0-V)*(47.D0/2.D0*V-13.D0/2.D0+1.D0/V)
     .      -3.D0/2.D0*(1.D0-6.D0*V+4.D0*V**2)*DLOG(V)
      HVH(X,Y)=0.25D0*( (1-X)*(-2+4*X-2*X**2+9*Y+9*X*Y-6*Y**2)
     .        /(3*Y)-2*(1-X-X**2+X**3-3*Y-2*X*Y-3*X**2*Y+3*Y**2
     .        +3*X*Y**2-Y**3)*(-PI/2- DATAN((1-2*X+X**2-Y-X*Y)/
     .         ((1-X)*DSQRT(-1.D0+2*X+2*Y-(X-Y)**2))))/DSQRT(-1.D0
     .         +2*X-(X-Y)**2+2*Y)-(1+X**2-2*Y-2*X*Y+Y**2)*DLOG(X))

c      HVH(X,Y)=0.25D0*( (1-X)*(5.D0*(1.D0+X)-4.D0*Y-2.D0/Y*
c     .     (-1.D0+2.D0*X+2.D0*Y-(X-Y)**2))/3
c     .        -2*(1-X-X**2+X**3-3*Y-2*X*Y-3*X**2*Y+3*Y**2
c     .        +3*X*Y**2-Y**3)*(-PI/2- DATAN((1-2*X+X**2-Y-X*Y)/
c     .         ((1-X)*DSQRT(-1.D0+2*X+2*Y-(X-Y)**2))))/DSQRT(-1.D0
c     .         +2*X-(X-Y)**2+2*Y)-(1+X**2-2*Y-2*X*Y+Y**2)*DLOG(X))

      QCD0(X) = (1+X**2)*(4*SP_HDEC((1-X)/(1+X))+2*SP_HDEC((X-1)/(X+1))
     .        - 3*DLOG((1+X)/(1-X))*DLOG(2/(1+X))
     .        - 2*DLOG((1+X)/(1-X))*DLOG(X))
     .        - 3*X*DLOG(4/(1-X**2)) - 4*X*DLOG(X)
      HQCDM(X)=QCD0(X)/X+(3+34*X**2-13*X**4)/16/X**3*DLOG((1+X)/(1-X))
     .        + 3.D0/8/X**2*(7*X**2-1)
      AQCDM(X)=QCD0(X)/X + (19+2*X**2+3*X**4)/16/X*DLOG((1+X)/(1-X))
     .        + 3.D0/8*(7-X**2)
      HQCD(X)=(4.D0/3*HQCDM(BETA_HDEC(X))
     .        +2*(4.D0/3-DLOG(X))*(1-10*X)/(1-4*X))*ASH/PI
     .       + (29.14671D0 + X*(-93.72459D0+12)
     .         +RATCOUP*(1.570D0 - 2*DLOG(HIGTOP)/3
     .                                     + DLOG(X)**2/9))*(ASH/PI)**2
     .       + (164.14D0 - 25.77D0*5 + 0.259D0*5**2)*(ASH/PI)**3
     .       +(39.34D0-220.9D0*5+9.685D0*5**2-0.0205D0*5**3)*(ASH/PI)**4
      AQCD(X)=(4.D0/3*AQCDM(BETA_HDEC(X))
     .        +2*(4.D0/3-DLOG(X))*(1-6*X)/(1-4*X))*ASH/PI
     .       + (29.14671D0 + RATCOUP*(23/6.D0 - DLOG(HIGTOP)
     .                                     + DLOG(X)**2/6))*(ASH/PI)**2
     .       + (164.14D0 - 25.77D0*5 + 0.259D0*5**2)*(ASH/PI)**3
     .       +(39.34D0-220.9D0*5+9.685D0*5**2-0.0205D0*5**3)*(ASH/PI)**4
      QCDH(X)=1.D0+HQCD(X)
      QCDA(X)=1.D0+AQCD(X)
      TQCDH(X)=1.D0+4.D0/3*HQCDM(BETA_HDEC(X))*ASH/PI
      TQCDA(X)=1.D0+4.D0/3*AQCDM(BETA_HDEC(X))*ASH/PI
      QCDC(X,Y)=1.D0+4/3.D0*ASH/PI*(9/4.D0 + (3-2*X+2*Y)/4*DLOG(X/Y)
     .         +((1.5D0-X-Y)*LAMB_HDEC(X,Y)**2+5*X*Y)/2/LAMB_HDEC(X,Y)
     .         /(1-X-Y)*DLOG(XI_HDEC(X,Y)*XI_HDEC(Y,X))
     .         + BIJ_HDEC(X,Y))
     .         + ASH/PI*(2*(4/3.D0-DLOG(X))
     .         - (X*2*(4/3.D0-DLOG(X)) + Y*2*(4/3.D0-DLOG(Y)))/(1-X-Y)
     .         - (X*2*(4/3.D0-DLOG(X))*(1-X+Y)
     .           +Y*2*(4/3.D0-DLOG(Y))*(1+X-Y))/LAMB_HDEC(X,Y)**2)
      QCDCI(X,Y)=1.D0+4/3.D0*ASH/PI*(3 + (Y-X)/2*DLOG(X/Y)
     .         +(2*(1-X-Y)+LAMB_HDEC(X,Y)**2)/2/LAMB_HDEC(X,Y)
     .         *DLOG(XI_HDEC(X,Y)*XI_HDEC(Y,X))
     .         + BIJ_HDEC(X,Y))
     .         + ASH/PI*(2*(4/3.D0-DLOG(X)) + 2*(4/3.D0-DLOG(Y))
     .         - (X*2*(4/3.D0-DLOG(X))*(1-X+Y)
     .           +Y*2*(4/3.D0-DLOG(Y))*(1+X-Y))/LAMB_HDEC(X,Y)**2)
      QCDCM(X,Y)=1.D0+4/3.D0*ASH/PI*(9/4.D0 + (3-2*X+2*Y)/4*DLOG(X/Y)
     .         +((1.5D0-X-Y)*LAMB_HDEC(X,Y)**2+5*X*Y)/2/LAMB_HDEC(X,Y)
     .         /(1-X-Y)*DLOG(4*X*Y/(1-X-Y+LAMB_HDEC(X,Y))**2)
     .         + BIJ_HDEC(X,Y))
      QCDCMI(X,Y)=1.D0+4/3.D0*ASH/PI*(3 + (Y-X)/2*DLOG(X/Y)
     .         +(2*(1-X-Y)+LAMB_HDEC(X,Y)**2)/2/LAMB_HDEC(X,Y)
     .         *DLOG(4*X*Y/(1-X-Y+LAMB_HDEC(X,Y))**2)
     .         + BIJ_HDEC(X,Y))
c------------------------------------------------------------------------
c--running x Yukawa coupling
      QCDCM1(X,Y)=1.D0+4/3.D0*ASH/PI*(9/4.D0 + (3-2*X+2*Y)/4*DLOG(X/Y)
     .         +((1.5D0-X-Y)*LAMB_HDEC(X,Y)**2+5*X*Y)/2/LAMB_HDEC(X,Y)
     .         /(1-X-Y)*DLOG(4*X*Y/(1-X-Y+LAMB_HDEC(X,Y))**2)
     .         + BIJ_HDEC(X,Y))
     .         + ASH/PI*(2*(4/3.D0-DLOG(X)))
      QCDCMI1(X,Y)=1.D0+4/3.D0*ASH/PI*(3 + (Y-X)/2*DLOG(X/Y)
     .         +(2*(1-X-Y)+LAMB_HDEC(X,Y)**2)/2/LAMB_HDEC(X,Y)
     .         *DLOG(4*X*Y/(1-X-Y+LAMB_HDEC(X,Y))**2)
     .         + BIJ_HDEC(X,Y))
     .         + ASH/PI*(2*(4/3.D0-DLOG(X)))
c------------------------------------------------------------------------
      CQCD(Z,TB,X,Y,R)= GF/(4*PI*DSQRT(2.D0))*Z**3*LAMB_HDEC(X,Y)
     .              *((1.D0-X-Y)*(X*TB**2*R**2*QCDC(X,Y)
     .                           +Y/TB**2*QCDC(Y,X))
     .               -4.D0*X*Y*R*QCDCI(X,Y))
      CQCDM(Z,TB,X,Y,R)= GF/(4*PI*DSQRT(2.D0))*Z**3*LAMB_HDEC(X,Y)
     .              *((1.D0-X-Y)*(X*TB**2*R**2*QCDCM1(X,Y)
     .                           +Y/TB**2*QCDCM(Y,X))
     .               -4.D0*X*Y*R*QCDCMI1(X,Y))
c MMM changed 21/8/13
      CQCD2HDM(Z,GCPD,GCPU,X,Y,R)= 
     .     GF/(4*PI*DSQRT(2.D0))*Z**3*LAMB_HDEC(X,Y)
     .     *((1.D0-X-Y)*(X*GCPD**2*R**2*QCDC(X,Y)
     .                  +Y*GCPU**2*QCDC(Y,X))
     .     -4.D0*GCPD*GCPU*X*Y*R*QCDCI(X,Y))
      CQCDM2HDM(Z,GCPD,GCPU,X,Y,R)= 
     .     GF/(4*PI*DSQRT(2.D0))*Z**3*LAMB_HDEC(X,Y)
     .     *((1.D0-X-Y)*(X*GCPD**2*R**2*QCDCM1(X,Y)
     .                  +Y*GCPU**2*QCDCM(Y,X))
     .     -4.D0*GCPD*GCPU*X*Y*R*QCDCMI1(X,Y))
c end MMM changed 21/8/13
      ELW(AMH,AMF,AMFP,QF,AI3F)=ELWFULL_HDEC(AMH,AMF,AMFP,QF,AI3F,AMW)
     .      + ALPH/PI*QF**2*HQCDM(BETA_HDEC(AMF**2/AMH**2))
     .      - GF*AMH**2/16.D0/PI**2/DSQRT(2.D0)*2.117203D0
      ELW0(AMH,AMF,QF,ACF)=ALPH/PI*3.D0/2*QF**2
     .                              *(3.D0/2-DLOG(AMH**2/AMF**2))
     .      +GF/8/DSQRT(2.D0)/PI**2*(ACF*AMT**2
     .        +AMW**2*(3*DLOG(CS)/SS-5)+AMZ**2*(0.5D0
     .          -3*(1-4*SS*DABS(QF))**2))
      CF(CA) = -CDLOG(-(1+CDSQRT(1-CA))/(1-CDSQRT(1-CA)))**2/4
      CG(CA) = CDSQRT(1-CA)/2*CDLOG(-(1+CDSQRT(1-CA))/(1-CDSQRT(1-CA)))
      CI1(CA,CB) = CA*CB/2/(CA-CB)
     .           + CA**2*CB**2/2/(CA-CB)**2*(CF(CA)-CF(CB))
     .           + CA**2*CB/(CA-CB)**2*(CG(CA)-CG(CB))
      CI2(CA,CB) = -CA*CB/2/(CA-CB)*(CF(CA)-CF(CB))
      HGGQCD(ASG,NF)=1.D0+ASG/PI*(95.D0/4.D0-NF*7.D0/6.D0)
      HGGQCD2(ASG,NF,AMH,AMT)=1.D0+ASG/PI*(95.D0/4.D0-NF*7.D0/6.D0)
     . +(ASG/PI)**2*(149533/288.D0-363/8.D0*ZETA2-495/8.D0*ZETA3
     .              +19/8.D0*DLOG(AMH**2/AMT**2)
     . +NF*(-4157/72.D0+11/2.D0*ZETA2+5/4.D0*ZETA3
     . +2/3.D0*DLOG(AMH**2/AMT**2))
     . +NF**2*(127/108.D0-1/6.D0*ZETA2))
     . +(ASG/PI)**3*(467.683620788D0+122.440972222D0*DLOG(AMH**2/AMT**2)
     .              +10.9409722222D0*DLOG(AMH**2/AMT**2)**2)
      PHGGQCD(ASG,NF)=1.D0+ASG/PI*(73/4.D0-7/6.D0*NF)
      PHGGQCD2(ASG,NF,AMH,AMT)=1.D0+ASG/PI*(73/4.D0-7/6.D0*NF)
     . +(ASG/PI)**2*(37631/96.D0-363/8.D0*ZETA2-495/8.D0*ZETA3
     . +NF*(-7189/144.D0+11/2.D0*ZETA2+5/4.D0*ZETA3)
     . +NF**2*(127/108.D0-1/6.D0*ZETA2))
     . +(ASG/PI)**3*(-212.447364638D0)
      DHGGQCD(ASG,NF)=1.D0+ASG/PI*(21.D0-NF*7.D0/6.D0)
      DHGGQCD2(ASG,NF,AMH,AMT)=1.D0+ASG/PI*(21.D0-NF*7.D0/6.D0)
     . +(ASG/PI)**2*(32531/72.D0-363/8.D0*ZETA2-495/8.D0*ZETA3
     .              +19/16.D0*DLOG(AMH**2/AMT**2)
     . +NF*(-15503/288.D0+11/2.D0*ZETA2+5/4.D0*ZETA3
     .     +1/3.D0*DLOG(AMH**2/AMT**2))
     . +NF**2*(127/108.D0-1/6.D0*ZETA2))
     . +(ASG/PI)**3*(63.7474683529D0+53.3715277778D0*DLOG(AMH**2/AMT**2)
     .              +5.47048611111D0*DLOG(AMH**2/AMT**2)**2)
      SGGQCD(ASG)=ASG/PI*7.D0/2.D0
      AGGQCD(ASG,NF)=1.D0+ASG/PI*(97.D0/4.D0-NF*7.D0/6.D0)
      AGGQCD2(ASG,NF,AMA,AMT)=1.D0+ASG/PI*(97.D0/4.D0-NF*7.D0/6.D0)
     . +(ASG/PI)**2*(237311/864.D0-529/24.D0*ZETA2-445/8.D0*ZETA3
     . +5*DLOG(AMA**2/AMT**2))
      HFFSELF(AMH)=1.D0+GF*AMH**2/16.D0/PI**2/DSQRT(2.D0)*2.117203D0
     .            -(GF*AMH**2/16.D0/PI**2/DSQRT(2.D0))**2*32.6567D0
      HVVSELF(AMH)=1.D0+GF*AMH**2/16.D0/PI**2/DSQRT(2.D0)*2.800952D0
     .            +(GF*AMH**2/16.D0/PI**2/DSQRT(2.D0))**2*62.0308D0
c     WTOP0(X,ASG,NF)=ASG/PI*(-2/3.D0*(PI**2+2*SP_HDEC(X)-2*SP_HDEC(1-X)
c    .               +(4*X*(1-X-2*X**2)*DLOG(X)
c    .                +2*(1-X)**2*(5+4*X)*DLOG(1-X)
c    .                -(1-X)*(5+9*X-6*X**2))
c    .               /2/(1-X)**2/(1+2*X)))
      WTOP(X,ASG,NF)=ASG/PI*(-2/3.D0*(PI**2+2*SP_HDEC(X)-2*SP_HDEC(1-X)
     .               +(4*X*(1-X-2*X**2)*DLOG(X)
     .                +2*(1-X)**2*(5+4*X)*DLOG(1-X)
     .                -(1-X)*(5+9*X-6*X**2))
     .               /2/(1-X)**2/(1+2*X)))
     .       + (ASG/2/PI)**2*(-110.4176D0+7.978D0*NF)
c     CHTOP0(X)=-4/3.D0*((5/2.D0-1/X)*DLOG(1-X)+X/(1-X)*DLOG(X)
c    .            +SP_HDEC(X)-SP_HDEC(1-X)+PI**2/2-9/4.D0)
      CHTOP(X)=8/3.D0*(SP_HDEC(1-X)-X/2/(1-X)*DLOG(X)
     .        + DLOG(X)*DLOG(1-X)/2+(1-2.5D0*X)/2/X*DLOG(1-X)-PI**2/3
     .        +9/8.D0)
      CHTOP1(X)=8/3.D0*(SP_HDEC(1-X)-X/2/(1-X)*DLOG(X)
     .        + DLOG(X)*DLOG(1-X)/2+(1-2.5D0*X)/2/X*DLOG(1-X)-PI**2/3
     .        +17/8.D0)

      PI=4D0*DATAN(1D0)
      SS=1.D0-(AMW/AMZ)**2
      CS=1.D0-SS

      ZETA2 = PI**2/6
      ZETA3 = 1.202056903159594D0
      ZETA4 = PI**4/90
      ZETA5 = 1.03692775514337D0

      IF(IHIGGS.NE.0)THEN
       TSC = (AMSQ+AMUR+AMDR)/3
       BSC = (AMSQ+AMUR+AMDR)/3
       CALL SFERMION_HDEC(TSC,BSC,AMSQ,AMUR,AMDR,AMEL,AMER,AL,AU,AD,AMU,
     .                AMST,AMSB,AMSL,AMSU,AMSD,AMSE,AMSN,AMSN1,
     .                GLEE,GLTT,GLBB,GHEE,GHTT,GHBB,
     .                GAEE,GATT,GABB,GCEN,GCTB)
       CALL SUSYCP_HDEC(TGBET)
      ENDIF

C--DECOUPLING THE TOP QUARK FROM ALPHAS
      AMT0=3.D8

C--TOP QUARK DECAY WIDTH
c     GAMT00= GF*AMT**3/8/DSQRT(2D0)/PI*(1-AMW**2/AMT**2)**2
c    .                                 *(1+2*AMW**2/AMT**2)
c    .      * (1+WTOP(AMW**2/AMT**2,ALPHAS_HDEC(AMT,3),5))
      GAMT0 = GF*AMT**3/8/DSQRT(2D0)/PI*VTB**2
     .      * LAMB_HDEC(AMB**2/AMT**2,AMW**2/AMT**2)
     .      * ((1-AMW**2/AMT**2)*(1+2*AMW**2/AMT**2)
     .         -AMB**2/AMT**2*(2-AMW**2/AMT**2-AMB**2/AMT**2))
     .      * (1+WTOP(AMW**2/AMT**2,ALPHAS_HDEC(AMT,3),5))
      IF(IHIGGS.NE.0.AND.AMT.GT.AMCH+AMB)THEN
       IF(I2HDM.EQ.0)THEN
        TSC = (AMSQ+AMUR+AMDR)/3
        BSC = (AMSQ+AMUR+AMDR)/3
       CALL SFERMION_HDEC(TSC,BSC,AMSQ,AMUR,AMDR,AMEL,AMER,AL,AU,AD,AMU,
     .                 AMST,AMSB,AMSL,AMSU,AMSD,AMSE,AMSN,AMSN1,
     .                 GLEE,GLTT,GLBB,GHEE,GHTT,GHBB,
     .                 GAEE,GATT,GABB,GCEN,GCTB)
        LOOP  = 2
        QSUSY = 1
        QSUSY1 = QSUSY
        QSUSY2 = QSUSY
        CALL BOTSUSY_HDEC(GLB,GHB,GAB,XGLB,XGHB,XGAB,QSUSY,LOOP)
       ELSE
        XGLB = GLB
        XGHB = GHB
        XGAB = GAB
       ENDIF
c      GAMT10= GF*AMT**3/8/DSQRT(2D0)/PI*VTB**2*(1-AMCH**2/AMT**2)**2
c    .        *((AMB/AMT)**2*XGAB**2 + GAT**2)
c    .      * (1+ALPHAS_HDEC(AMT,3)/PI*CHTOP0(AMCH**2/AMT**2))
       YMB = RUNM_HDEC(AMT,5)
c      GAMT10= GF*AMT**3/8/DSQRT(2D0)/PI*VTB**2
c    .      * LAMB_HDEC(AMB**2/AMT**2,AMCH**2/AMT**2)
c    .      * (((YMB/AMT)**2*XGAB**2 + GAT**2)
c    .      * (1+AMB**2/AMT**2-AMCH**2/AMT**2)+4*YMB**2/AMT**2*XGAB*GAT)
c    .      * (1+ALPHAS_HDEC(AMT,3)/PI*CHTOP0(AMCH**2/AMT**2))
       GAMT1 = GF*AMT**3/8/DSQRT(2D0)/PI*VTB**2
     .      * LAMB_HDEC(AMB**2/AMT**2,AMCH**2/AMT**2)
     .      * (GAT**2*(1+AMB**2/AMT**2-AMCH**2/AMT**2)
     .      * (1+ALPHAS_HDEC(AMT,3)/PI*CHTOP(AMCH**2/AMT**2))
     .        +(YMB/AMT)**2*XGAB**2*(1+AMB**2/AMT**2-AMCH**2/AMT**2)
     .      * (1+ALPHAS_HDEC(AMT,3)/PI*CHTOP1(AMCH**2/AMT**2))
     .      + 4*YMB**2/AMT**2*XGAB*GAT)
      ELSE
       GAMT1 = 0
      ENDIF
      GAMT1 = GAMT0+GAMT1
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     write(6,*)'topw: ',GAMT0,WTOP(AMW**2/AMT**2,ALPHAS_HDEC(AMT,3),5),
c    .                   WTOP0(AMW**2/AMT**2,ALPHAS_HDEC(AMT,3),5),
c    .                   ALPHAS_HDEC(AMT,3)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     GAMT10= GAMT00+GAMT10
c     GAMT0 = GAMT00
c     GAMT1 = GAMT10
c     write(6,*)'topw: ',GAMT0,GAMT00,GAMT00/GAMT0
c     write(6,*)'toph: ',GAMT1-GAMT0,GAMT10-GAMT00,
c    .                  (GAMT10-GAMT00)/(GAMT1-GAMT0)
c     write(6,*)'top:  ',gamt0/gamt1,(gamt1-gamt0)/gamt1,gamt1
c     write(6,*)'top0: ',gamt00/gamt10,(gamt10-gamt00)/gamt10,gamt10
c     write(6,*)'toph: ',CHTOP(AMCH**2/AMT**2),CHTOP0(AMCH**2/AMT**2),
c    .                   CHTOP(AMCH**2/AMT**2)/CHTOP0(AMCH**2/AMT**2)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     write(6,*)'CKM: ',VUS,VCB,VUB/VCB
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

      IF(IHIGGS.EQ.0)THEN

C        =========================================================
C                              SM HIGGS DECAYS
C        =========================================================
      AMXX=AMH
      AMH=AMSM
C     =============  RUNNING MASSES 
      RMS = RUNM_HDEC(AMH,3)
      RMC = RUNM_HDEC(AMH,4)
      RMB = RUNM_HDEC(AMH,5)
      RMT = RUNM_HDEC(AMH,6)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     XXX = RUNM_HDEC(AMB,5)
c     write(6,*) 'mb = ',XXX
c     XXX = RUNM_HDEC(XXX,5)
c     write(6,*) 'mb = ',XXX
c     XXX = RUNM_HDEC(XXX,5)
c     write(6,*) 'mb = ',XXX
c     XXX = RUNM_HDEC(XXX,5)
c     write(6,*) 'mb = ',XXX
c     XXX = RUNM_HDEC(XXX,5)
c     write(6,*) 'mb = ',XXX
c     XXX = RUNM_HDEC(XXX,5)
c     write(6,*) 'mb = ',XXX
c     XXX = RUNM_HDEC(XXX,5)
c     write(6,*) 'mb = ',XXX
c     XXX = RUNM_HDEC(XXX,5)
c     write(6,*) 'mb = ',XXX
c     XXX = RUNM_HDEC(XXX,5)
c     write(6,*) 'mb = ',XXX
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     write(6,"(F9.2,3(2X,G12.5))")AMH,RMC,RMB,RMT
      IF(ISM4.NE.0)THEN
       RMBP= RUNM_HDEC(AMH,7)
       RMTP= RUNM_HDEC(AMH,8)
c      write(6,*)AMTP,AMBP
c      write(6,*)RUNM_HDEC(AMTP,8),RUNM_HDEC(AMTP,7)
c      write(6,*)RMTP,RMBP,AMH
      ENDIF
      RATCOUP = 1
      HIGTOP = AMH**2/AMT**2
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     write(6,*)'strange mass: ',runm_hdec(1.d0,3),runm_hdec(2.d0,3)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     X1 = RUNM_HDEC(AMB,5)
c     X2 = RUNM_HDEC(AMH,5)
c     do i=1,100
c      x1 = RUNM_HDEC(x1,5)
c     enddo
c     write(6,*)'mb,MH      = ',AMB,AMH
c     write(6,*)'als(MZ)    = ',ALPHAS_HDEC(AMZ,3)
c     write(6,*)'als(mb,MH) = ',ALPHAS_HDEC(AMB,3),ALPHAS_HDEC(AMH,3)
c     write(6,*)'mb(mb,MH)  = ',X1,X2
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     TOPFAC = 1
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

      ASH=ALPHAS_HDEC(AMH,3)
c     write(66,('f4.0,3(g18.10)'))AMH,ASH,RMB,RUNM_HDEC(AMH/2,5)
      AMC0=1.D8
      AMB0=2.D8
      AS3=ALPHAS_HDEC(AMH,3)
      AMC0=AMC
      AS4=ALPHAS_HDEC(AMH,3)
      AMB0=AMB
C     AMT0=AMT
C     =============== PARTIAL WIDTHS 
C  H ---> G G
C
C  mass dependent NLO QCD corrections
       CALL CORRGG_HDEC(AMH,AMT,AMB,XHGG,XHQQ)
       EPS=1.D-8
       NFEXT = 3
       ASG = AS3
       CTT = 4*AMT**2/AMH**2*DCMPLX(1D0,-EPS)
       CTB = 4*AMB**2/AMH**2*DCMPLX(1D0,-EPS)
       CTC = 4*AMC**2/AMH**2*DCMPLX(1D0,-EPS)
       CAT0= 2*CTT*(1+(1-CTT)*CF(CTT))
       CAB0= 2*CTB*(1+(1-CTB)*CF(CTB))
       CAC0= 2*CTC*(1+(1-CTC)*CF(CTC))
       CAT = CAT0 * CPT
       CAB = CAB0 * CPB
       CAC = CAC0 * CPC
       CATP = 0
       CABP = 0
       IF(ISM4.NE.0)THEN
        CTTP = 4*AMTP**2/AMH**2*DCMPLX(1D0,-EPS)
        CTBP = 4*AMBP**2/AMH**2*DCMPLX(1D0,-EPS)
        CATP0= 2*CTTP*(1+(1-CTTP)*CF(CTTP))
        CABP0= 2*CTBP*(1+(1-CTBP)*CF(CTBP))
        CATP = CATP0 * CPTP
        CABP = CABP0 * CPBP
       ENDIF
       FQCD=HGGQCD(ASG,NFEXT)
       DQCD=DHGGQCD(ASG,NFEXT)
       PQCD=PHGGQCD(ASG,NFEXT)
       XFAC = CDABS(CAT+CAB+CAC+CATP+CABP)**2*FQCD
     .      + 4*DREAL(DCONJG(CAT+CAB+CAC+CATP+CABP)*CPGG)*DQCD
     .      + (2*CPGG)**2*PQCD
C  mass dependent NLO QCD corrections
       XFAC0 =(CDABS(CAT)**2*(XHGG(1)+XHQQ(1)*NFEXT)
     .       + CDABS(CAB)**2*(XHGG(2)+XHQQ(2)*NFEXT)
     .       + 2*DREAL(DCONJG(CAT)*CAB)*(XHGG(3)+XHQQ(3)*NFEXT))*ASG/PI
       XFAC = XFAC + XFAC0
       HGG=HVV(AMH,0.D0)*(ASG/PI)**2*XFAC/8

C  H ---> G G* ---> G CC   TO BE ADDED TO H ---> CC
       NFEXT = 4
       ASG = AS4
       FQCD=HGGQCD(ASG,NFEXT)
       DQCD=DHGGQCD(ASG,NFEXT)
       PQCD=PHGGQCD(ASG,NFEXT)
       XFAC = CDABS(CAT+CAB+CAC+CATP+CABP)**2*FQCD
     .      + 4*DREAL(DCONJG(CAT+CAB+CAC+CATP+CABP)*CPGG)*DQCD
     .      + (2*CPGG)**2*PQCD
C  mass dependent NLO QCD corrections
       XFAC0 =(CDABS(CAT)**2*(XHGG(1)+XHQQ(1)*NFEXT)
     .       + CDABS(CAB)**2*(XHGG(2)+XHQQ(2)*NFEXT)
     .       + 2*DREAL(DCONJG(CAT)*CAB)*(XHGG(3)+XHQQ(3)*NFEXT))*ASG/PI
       XFAC = XFAC + XFAC0
       DCC=HVV(AMH,0.D0)*(ASG/PI)**2*XFAC/8 - HGG
C  H ---> G G* ---> G BB   TO BE ADDED TO H ---> BB
       NFEXT = 5
       ASG = ASH
       FQCD=HGGQCD(ASG,NFEXT)
       DQCD=DHGGQCD(ASG,NFEXT)
       PQCD=PHGGQCD(ASG,NFEXT)
       XFAC = CDABS(CAT+CAB+CAC+CATP+CABP)**2*FQCD
     .      + 4*DREAL(DCONJG(CAT+CAB+CAC+CATP+CABP)*CPGG)*DQCD
     .      + (2*CPGG)**2*PQCD
C  mass dependent NLO QCD corrections
       XFAC0 =(CDABS(CAT)**2*(XHGG(1)+XHQQ(1)*NFEXT)
     .       + CDABS(CAB)**2*(XHGG(2)+XHQQ(2)*NFEXT)
     .       + 2*DREAL(DCONJG(CAT)*CAB)*(XHGG(3)+XHQQ(3)*NFEXT))*ASG/PI
       XFAC = XFAC + XFAC0
       DBB=HVV(AMH,0.D0)*(ASG/PI)**2*XFAC/8 - HGG - DCC

C  H ---> G G: FULL NNNLO CORRECTIONS FOR NF=5
       IF(ISM4.EQ.0)THEN
        FQCD0=HGGQCD(ASG,5)
        FQCD=HGGQCD2(ASG,5,AMH,AMT)
        DQCD=DHGGQCD2(ASG,5,AMH,AMT)
        PQCD=PHGGQCD2(ASG,5,AMH,AMT)
        XFAC = CDABS(CAT+CAB+CAC+CATP+CABP)**2*FQCD
     .      + 4*DREAL(DCONJG(CAT+CAB+CAC+CATP+CABP)*CPGG)*DQCD
     .      + (2*CPGG)**2*PQCD
C  mass dependent NLO QCD corrections
        XFAC0 =(CDABS(CAT)**2*(XHGG(1)+XHQQ(1)*NFEXT)
     .        + CDABS(CAB)**2*(XHGG(2)+XHQQ(2)*NFEXT)
     .        + 2*DREAL(DCONJG(CAT)*CAB)*(XHGG(3)+XHQQ(3)*NFEXT))*ASG/PI
        XFAC = XFAC + XFAC0
        HGG=HVV(AMH,0.D0)*(ASG/PI)**2*XFAC/8
C  electroweak corrections
        IF(ICOUPELW.EQ.0)THEN
c        XFAC00 = CDABS(CAT0+CAB0+CAC0+CATP0+CABP0)**2*FQCD
         XFAC00= DREAL(DCONJG(CAT+CAB+CAC+CATP+CABP)
     .                     * (CAT0+CAB0+CAC0+CATP0+CABP0))*FQCD
        ELSE
         XFAC00 = CDABS(CAT+CAB+CAC+CATP+CABP)**2*FQCD
     .          + 4*DREAL(DCONJG(CAT+CAB+CAC+CATP+CABP)*CPGG)*DQCD
     .          + (2*CPGG)**2*PQCD
        ENDIF
        HGG0 = HVV(AMH,0.D0)*(ASG/PI)**2*XFAC00/8 * GLGL_ELW(AMT,AMH)
        HGG=HGG+HGG0
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c       write(6,*)'Hgg: ',CAT,CAT*3
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c       write(6,*)CDABS(CAT)**2,16/9.D0,16/9.D0/CDABS(CAT)**2
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c       write(46,*)AMH,FQCD0,FQCD0+XFAC0/CDABS(CAT+CAB)**2
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c       YY0 = CDABS(CAT+CAC)**2*FQCD
c    .      + CDABS(CAT)**2*(XHGG(1)+XHQQ(1)*NFEXT)*ASG/PI
c       YTT = HVV(AMH,0.D0)*(ASG/PI)**2*YY0/8/HGG*(1+GLGL_ELW(AMT,AMH))
c       YY0 = CDABS(CAB)**2*(FQCD+(XHGG(2)+XHQQ(2)*NFEXT)*ASG/PI)
c       YBB = HVV(AMH,0.D0)*(ASG/PI)**2*YY0/8/HGG*(1+GLGL_ELW(AMT,AMH))
c       YY0 = 2*DREAL(DCONJG(CAT+CAC)*CAB)*FQCD
c    .      + 2*DREAL(DCONJG(CAT)*CAB)*(XHGG(3)+XHQQ(3)*NFEXT)*ASG/PI
c       YTB = HVV(AMH,0.D0)*(ASG/PI)**2*YY0/8/HGG*(1+GLGL_ELW(AMT,AMH))
c       write(51,('f4.0,4(g18.10)'))AMH,HGG,YTT,YBB,YTB
c       write(6,*)'gg: ',AMH,YTT+YBB+YTB
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c       XFAC1 = CDABS(CAT)**2*(XHGG(1)+XHQQ(1)*NFEXT)*ASG/PI
c       XFAC2 = CDABS(CAB)**2*(XHGG(2)+XHQQ(2)*NFEXT)*ASG/PI
c       XFAC3 = 2*DREAL(DCONJG(CAT)*CAB)*(XHGG(3)+XHQQ(3)*NFEXT)*ASG/PI
c       write(50,*)AMH,FQCD0,XFAC1/CDABS(CAT)**2/FQCD0
c       write(50,*)AMH,FQCD0,XFAC2/CDABS(CAB)**2/FQCD0
c       write(50,*)AMH,FQCD0,XFAC3/(2*DREAL(DCONJG(CAT)*CAB))/FQCD0
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c       write(50,*)AMH,FQCD0,FQCD0+XFAC1/CDABS(CAT)**2
c       write(51,*)AMH,FQCD0,FQCD0+XFAC2/CDABS(CAB)**2
c       write(52,*)AMH,FQCD0,FQCD0+XFAC3/(2*DREAL(DCONJG(CAT)*CAB))
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
       ELSE
        IM = IGGELW
        FQCD0=HGGQCD(ASG,5)
        DQCD=DHGGQCD2(ASG,5,AMH,AMT)
        PQCD=PHGGQCD2(ASG,5,AMH,AMT)
        FAC4 = -ASG**2/PI**2*(77/288.D0*2 + (2/3.D0*5+19/8.D0)
     .       * (CPBP*DLOG(AMBP**2/AMT**2)+CPTP*DLOG(AMTP**2/AMT**2))
     .         / (CPT+CPBP+CPTP))
        IF(CPBP.EQ.0.D0.AND.CPTP.EQ.0.D0.AND.CPT.EQ.0.D0)
     .                            FAC4 = -ASG**2/PI**2*(77/288.D0*2)
        FQCD=HGGQCD2(ASG,5,AMH,AMT)+FAC4
        DQCD=DQCD+FAC4/2
        XFAC = CDABS(CAT+CAB+CAC+CATP+CABP)**2*FQCD
        HGG=HVV(AMH,0.D0)*(ASG/PI)**2*XFAC/8
        IF(ICOUPELW.EQ.0)THEN
         XFAC00= DREAL(DCONJG(CAT+CAB+CAC+CATP+CABP)
     .                     * (CAT0+CAB0+CAC0+CATP0+CABP0))*FQCD
        ELSE
         XFAC00 = CDABS(CAT+CAB+CAC+CATP+CABP)**2*FQCD
     .          + 4*DREAL(DCONJG(CAT+CAB+CAC+CATP+CABP)*CPGG)*DQCD
     .          + (2*CPGG)**2*PQCD
        ENDIF
        HGG0 = HVV(AMH,0.D0)*(ASG/PI)**2*XFAC00/8 * GLGL_ELW4(IM,AMH)
        HGG=HGG+HGG0
c     write(6,*)'H -> gg: ',AMH,HGG,GLGL_ELW4(IM,AMH)*100,FQCD,
c    .                      FAC4/FQCD*100
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        IF(AMNUP.EQ.AMEP)THEN
         XXL = AMEP**2
        ELSE
         XXL =AMNUP**2*AMEP**2/(AMNUP**2-AMEP**2)*DLOG(AMNUP**2/AMEP**2)
        ENDIF
        IF(AMTP.EQ.AMBP)THEN
         XXQ = AMTP**2
        ELSE
         XXQ = AMTP**2*AMBP**2/(AMTP**2-AMBP**2)*DLOG(AMTP**2/AMBP**2)
        ENDIF
        CELW = GF/8/DSQRT(2.D0)/PI**2*(5*AMT**2/2
     .       + 7*(AMNUP**2+AMEP**2)/6 - XXL
     .       + 3*(AMTP**2+AMBP**2)/2 - 3*XXQ)
c      write(6,*)'H -> gg: ',AMH,2*CELW*100,GLGL_ELW4(IM,AMH)*100
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
       ENDIF

      IF(NFGG.EQ.3)THEN
       HGG = HGG - DBB - DCC
      ELSEIF(NFGG.EQ.4)THEN
       HGG = HGG - DBB
       DCC = 0
      ELSE
       DCC = 0
       DBB = 0
      ENDIF
      IF(IFERMPHOB.NE.0)THEN
c      write(6,*)'Fermiophobic: H -> gg ---> 0'
       HGG = 0
       DBB = 0
       DCC = 0
      ENDIF
c     write(6,*)'H -> GG: ',FQCD0,FQCD,FQCD-FQCD0
c     write(6,*)'H -> GG: ',(FQCD0-1)/ASG*PI,(FQCD-FQCD0)/ASG**2*PI**2
c     write(6,*)'H -> GG: ',GLGL_ELW(AMT,AMH)
c     write(6,*)'XFAC(',AMH,') = ',xfac
c     write(6,*)'BR(H -> gg) = ',HGG

      SM4FACF=1
      SM4FACW = 1
      SM4FACZ = 1
      IF(ISM4.NE.0)THEN
       SM4FACF=1+ELW4_HDEC(AMNUP,AMEP,AMTP,AMBP)
       SM4FACW=1+ELW4V_HDEC(1,AMNUP,AMEP,AMTP,AMBP)
       SM4FACZ=1+ELW4V_HDEC(2,AMNUP,AMEP,AMTP,AMBP)
c      write(6,*)'SM4 elw.: ff,WW,ZZ ',(SM4FACF-1)*100,
c    .                 (SM4FACW-1)*100,(SM4FACZ-1)*100
      ENDIF

C  H ---> E E
c     CPEL = CPMU
c     AMELEC = 0.510998910D-3
c     IF(AMH.LE.2*AMELEC) THEN
c      HEE = 0
c     ELSE
c     HEE=HFF(AMH,(AMELEC/AMH)**2)
c    .    *(1+ELW0(AMH,AMELEC,-1.D0,7.D0))
c    .    *(1+ELW(AMH,AMELEC,0.D0,-1.D0,-1/2.D0))
c    .    *HFFSELF(AMH)
c    .    *SM4FACF
c    .    * CPEL**2
c     IF(ICOUPELW.EQ.0)THEN
c      HEE=HFF(AMH,(AMELEC/AMH)**2) * CPMU
c    .    * ((CPEL-1)
c    .       +(1+ELW0(AMH,AMELEC,-1.D0,7.D0))
c    .       +(1+ELW(AMH,AMELEC,0.D0,-1.D0,-1/2.D0))
c    .       *HFFSELF(AMH) * SM4FACF)
c     ENDIF
c     ENDIF
C  H ---> MU MU
      IF(AMH.LE.2*AMMUON) THEN
       HMM = 0
      ELSE
      HMM=HFF(AMH,(AMMUON/AMH)**2)
c    .    *(1+ELW0(AMH,AMMUON,-1.D0,7.D0))
     .    *(1+ELW(AMH,AMMUON,0.D0,-1.D0,-1/2.D0))
     .    *HFFSELF(AMH)
     .    *SM4FACF
     .    * CPMU**2
      IF(ICOUPELW.EQ.0)THEN
       HMM=HFF(AMH,(AMMUON/AMH)**2) * CPMU
     .    * ((CPMU-1)
c    .       +(1+ELW0(AMH,AMMUON,-1.D0,7.D0))
     .       +(1+ELW(AMH,AMMUON,0.D0,-1.D0,-1/2.D0))
     .       *HFFSELF(AMH) * SM4FACF)
      ENDIF
      ENDIF
c     write(6,*)'ee/mumu: ',HEE/HMM,HEE/HMM * AMMUON**2/AMELEC**2
C  H ---> TAU TAU
      IF(AMH.LE.2*AMTAU) THEN
       HLL = 0
      ELSE
      HLL=HFF(AMH,(AMTAU/AMH)**2)
c    .    *(1+ELW0(AMH,AMTAU,-1.D0,7.D0))
     .    *(1+ELW(AMH,AMTAU,0.D0,-1.D0,-1/2.D0))
     .    *HFFSELF(AMH)
     .    *SM4FACF
     .    * CPTAU**2
      IF(ICOUPELW.EQ.0)THEN
       HLL=HFF(AMH,(AMTAU/AMH)**2) * CPTAU
     .    * ((CPTAU-1)
c    .        +(1+ELW0(AMH,AMTAU,-1.D0,7.D0))
     .        +(1+ELW(AMH,AMTAU,0.D0,-1.D0,-1/2.D0))
     .        *HFFSELF(AMH) * SM4FACF)
      ENDIF
      ENDIF

c      write(96,*)AMH,ELW0(AMH,AMTAU,-1.D0,7.D0)
c    .               -ALPH/PI*3.D0/2*(3.D0/2-DLOG(AMH**2/AMTAU**2))
c    .               ,ELW(AMH,AMTAU,0.D0,-1.D0,-1/2.D0)
c    .               -ALPH/PI*HQCDM(BETA_HDEC(AMTAU**2/AMH**2))
c    .               +GF*AMH**2/16.D0/PI**2/DSQRT(2.D0)*2.117203D0

c      write(6,*)'tau tau: ',AMH,ELW0(AMH,AMTAU,-1.D0,7.D0),HFFSELF(AMH)
C  H --> SS
      IRAT = 0
      RATCOUP = 1
      XQCD0 = QCDH(RMS**2/AMH**2)
      IF(CPT.EQ.0.D0)THEN
       RATCOUP = 0
      ELSE
       IF(CPS.EQ.0.D0)THEN
        CPS = 1.D-20
        IRAT = 1
       ENDIF
       RATCOUP = CPT/CPS
      ENDIF
      XQCD1= QCDH(RMS**2/AMH**2)
      XQCD = (XQCD0-XQCD1)/2/XQCD1
      IF(AMH.LE.2*AMS) THEN
       HSS = 0
      ELSE
       HS2=3.D0*HFF(AMH,(RMS/AMH)**2)
     .    *QCDH(RMS**2/AMH**2)
c    .    *(1+ELW0(AMH,RMS,-1.D0/3.D0,7.D0))
     .    *(1+ELW(AMH,AMS,AMC,-1/3.D0,-1/2.D0))
     .    *HFFSELF(AMH) * SM4FACF
     .    * CPS**2
      IF(ICOUPELW.EQ.0)THEN
       HS2=3.D0*HFF(AMH,(RMS/AMH)**2) * CPS
     .    *QCDH(RMS**2/AMH**2)
     .    * ((CPS-1)
c    .         +(1+ELW0(AMH,RMS,-1.D0/3.D0,7.D0)*(1+XQCD))
     .         +(1+ELW(AMH,AMS,AMC,-1/3.D0,-1/2.D0)*(1+XQCD))
     .       *HFFSELF(AMH) * SM4FACF)
      ENDIF
c      IF(HS2.LT.0.D0) HS2 = 0
       HS1=3.D0*HFF(AMH,(AMS/AMH)**2)
     .    *TQCDH(AMS**2/AMH**2)
c    .    *(1+ELW0(AMH,RMS,-1.D0/3.D0,7.D0))
     .    *(1+ELW(AMH,AMS,AMC,-1/3.D0,-1/2.D0))
     .    *HFFSELF(AMH) * SM4FACF
     .    *CPS**2
      IF(ICOUPELW.EQ.0)THEN
       HS1=3.D0*HFF(AMH,(AMS/AMH)**2) * CPS
     .    *TQCDH(AMS**2/AMH**2)
     .    * ((CPS-1)
c    .         +(1+ELW0(AMH,RMS,-1.D0/3.D0,7.D0))
     .         +(1+ELW(AMH,AMS,AMC,-1/3.D0,-1/2.D0))
     .       *HFFSELF(AMH) * SM4FACF)
      ENDIF
       RAT = 2*AMS/AMH
       HSS = QQINT_HDEC(RAT,HS1,HS2)
      ENDIF
c     HSS = HSS * SM4FACF
      IF(IRAT.EQ.1)CPS = 0
C  H --> CC
      IRAT = 0
      RATCOUP = 1
      XQCD0 = QCDH(RMC**2/AMH**2)
      IF(CPT.EQ.0.D0)THEN
       RATCOUP = 0
      ELSE
       IF(CPC.EQ.0.D0)THEN
        CPC = 1.D-20
        IRAT = 1
       ENDIF
       RATCOUP = CPT/CPC
      ENDIF
      XQCD1= QCDH(RMC**2/AMH**2)
      XQCD = (XQCD0-XQCD1)/2/XQCD1
      IF(AMH.LE.2*AMC) THEN
       HCC = 0
      ELSE
       HC2=3.D0*HFF(AMH,(RMC/AMH)**2)
     .    *QCDH(RMC**2/AMH**2)
c    .    *(1+ELW0(AMH,RMC,2.D0/3.D0,7.D0))
     .    *(1+ELW(AMH,AMC,AMS,2/3.D0,1/2.D0))
     .    *HFFSELF(AMH) * SM4FACF
     .    *CPC**2
     .   + DCC
      IF(ICOUPELW.EQ.0)THEN
       HC2=3.D0*HFF(AMH,(RMC/AMH)**2) * CPC
     .    *QCDH(RMC**2/AMH**2)
     .    * ((CPC-1)
c    .       +(1+ELW0(AMH,RMC,2.D0/3.D0,7.D0)*(1+XQCD))
     .       +(1+ELW(AMH,AMC,AMS,2/3.D0,1/2.D0)*(1+XQCD))
     .       *HFFSELF(AMH) * SM4FACF)
     .   + DCC
      ENDIF
c      IF(HC2.LT.0.D0) HC2 = 0
       HC1=3.D0*HFF(AMH,(AMC/AMH)**2)
     .    *TQCDH(AMC**2/AMH**2)
c    .    *(1+ELW0(AMH,AMC,2.D0/3.D0,7.D0))
     .    *(1+ELW(AMH,AMC,AMS,2/3.D0,1/2.D0))
     .    *HFFSELF(AMH) * SM4FACF
     .    *CPC**2
     .   + DCC
      IF(ICOUPELW.EQ.0)THEN
       HC1=3.D0*HFF(AMH,(AMC/AMH)**2) * CPC
     .    *TQCDH(AMC**2/AMH**2)
     .    * ((CPC-1)
c    .       +(1+ELW0(AMH,AMC,2.D0/3.D0,7.D0))
     .       +(1+ELW(AMH,AMC,AMS,2/3.D0,1/2.D0))
     .       *HFFSELF(AMH) * SM4FACF)
     .   + DCC
      ENDIF
       RAT = 2*AMC/AMH
       HCC = QQINT_HDEC(RAT,HC1,HC2)

c      eps00 = 1.d-15
c      xq1 = amb*(1-eps00)
c      xq2 = amb*(1+eps00)
c      xal1 = alphas_hdec(xq1,3)
c      xal2 = alphas_hdec(xq2,3)
c      xdel1= 7*alphas_hdec(xq1,3)**2/pi**2/24
c      xdel2= 7*alphas_hdec(xq2,3)**2/pi**2/24
c      write(6,*)xal1,xal2
c      write(6,*)(xal2-xal1)/xal1,(xal2-xal1)/xal2
c      write(6,*)xdel2,xdel1

c      ALS0 = ALPHAS_HDEC(AMZ,3)
c      WMC = RUNM_HDEC(AMC,4)
c      YMC = RUNM_HDEC(3.D0,4)
c      ZMC = 0.996D0-(ALS0-0.1189D0)/2.D0*9
c      do i =125,133
c       sc = i/100.D0
c       XMC = RUNM_HDEC(sc,4)
c       write(6,*)
c       write(6,*)sc,XMC,AMC,YMC,ZMC,WMC,ALS0
c      enddo
      ENDIF
c     HCC = HCC * SM4FACF
      IF(IRAT.EQ.1)CPC = 0
C  H --> BB :
      IRAT = 0
      RATCOUP = 1
      XQCD0 = QCDH(RMB**2/AMH**2)
      IF(CPT.EQ.0.D0)THEN
       RATCOUP = 0
      ELSE
       IF(CPB.EQ.0.D0)THEN
        CPB = 1.D-20
        IRAT = 1
       ENDIF
       RATCOUP = CPT/CPB
      ENDIF
      XQCD1= QCDH(RMB**2/AMH**2)
      XQCD = (XQCD0-XQCD1)/2/XQCD1
      IF(AMH.LE.2*AMB) THEN
       HBB = 0
      ELSE
       HB2=3.D0*HFF(AMH,(RMB/AMH)**2)
     .    *QCDH(RMB**2/AMH**2)
c    .    *(1+ELW0(AMH,RMB,-1.D0/3.D0,1.D0))
     .    *(1+ELW(AMH,AMB,AMT,-1/3.D0,-1/2.D0))
     .    *HFFSELF(AMH) * SM4FACF
     .    * CPB**2
     .   + DBB
       IF(ICOUPELW.EQ.0)THEN
        HB2=3.D0*HFF(AMH,(RMB/AMH)**2) * CPB
     .     *QCDH(RMB**2/AMH**2)
     .     * ((CPB-1)
c    .        +(1+ELW0(AMH,RMB,-1.D0/3.D0,1.D0)*(1+XQCD))
     .        +(1+ELW(AMH,AMB,AMT,-1/3.D0,-1/2.D0)*(1+XQCD))
     .        *HFFSELF(AMH) * SM4FACF)
     .    + DBB
       ENDIF
c      IF(HB2.LT.0.D0) HB2 = 0
       HB1=3.D0*HFF(AMH,(AMB/AMH)**2)
     .    *TQCDH(AMB**2/AMH**2)
c    .    *(1+ELW0(AMH,AMB,-1.D0/3.D0,1.D0))
     .    *(1+ELW(AMH,AMB,AMT,-1/3.D0,-1/2.D0))
     .    *HFFSELF(AMH) * SM4FACF
     .    * CPB**2
     .   + DBB
      IF(ICOUPELW.EQ.0)THEN
       HB1=3.D0*HFF(AMH,(AMB/AMH)**2) * CPB
     .    *TQCDH(AMB**2/AMH**2)
     .    * ((CPB-1)
c    .        +(1+ELW0(AMH,AMB,-1.D0/3.D0,1.D0))
     .        +(1+ELW(AMH,AMB,AMT,-1/3.D0,-1/2.D0))
     .       *HFFSELF(AMH) * SM4FACF)
     .   + DBB
      ENDIF
       RAT = 2*AMB/AMH
       HBB = QQINT_HDEC(RAT,HB1,HB2)

c      write(97,*)AMH,ELW0(AMH,AMB,-1/3.D0,1.D0)
c    .               -ALPH/PI*3.D0/2/9*(3.D0/2-DLOG(AMH**2/AMB**2))
c    .               ,ELW(AMH,AMB,AMT,-1/3.D0,-1/2.D0)
c    .               -ALPH/PI/9*HQCDM(BETA_HDEC(AMB**2/AMH**2))
c    .               +GF*AMH**2/16.D0/PI**2/DSQRT(2.D0)*2.117203D0

c      write(6,*)AMH,
c    .        ALPH/PI*AMH**2/32.D0/AMW**2/(1-AMW**2/AMZ**2)*2.117203D0
c      write(6,*)

c      write(6,*)RMC,RUNM_HDEC(AMC,4),RMC/RUNM_HDEC(AMC,4),
c    .           QCDH(RMC**2/AMH**2),
c    .           RMB,RUNM_HDEC(AMB,5),RMB/RUNM_HDEC(AMB,5),
c    .           QCDH(RMB**2/AMH**2)

c      write(6,*)'bb: ',AMH,RATCOUP*(1.570D0 - 2*DLOG(HIGTOP)/3
c    .    + DLOG(AMH**2/RMB**2)**2/9)*(ASH/PI)**2,QCDH(RMB**2/AMH**2)-1,
c    .        ELW0(AMH,AMB,-1.D0/3.D0,1.D0),HFFSELF(AMH)

c      write(6,*)AMH,HB1,HB2,HBB

c      write(61,*)AMH,1.D0+GF*AMH**2/16.D0/PI**2/DSQRT(2.D0)*2.117203D0,
c    .               HFFSELF(AMH)

c      ALS0 = ALPHAS_HDEC(AMZ,3)
c      WMB = RUNM_HDEC(AMB,5)
c      YMB = 4.163D0-(ALS0-0.1189D0)/2.D0*12
c      XMB = RUNM_HDEC(YMB,5)
c      write(6,*)
c      write(6,*)AMB,XMB,YMB,WMB,ALS0
c      do i =1,100
c       sc = i
c       XMS = RUNM_HDEC(sc,3)
c       XMC = RUNM_HDEC(sc,4)
c       XMB = RUNM_HDEC(sc,5)
c       write(66,*)sc,XMS,XMC,XMB,ALS0
c      enddo
      ENDIF
c     HBB = HBB * SM4FACF
      IF(IRAT.EQ.1)CPB = 0

c     HB1X=3.D0*HFF(AMH,(AMB/AMH)**2)
c    .    *TQCDH(AMB**2/AMH**2)
c    .    /(BETA_HDEC(AMB**2/AMH**2))**3
c     HB2X=3.D0*HFF(AMH,(RMB/AMH)**2)
c    .    *QCDH(RMB**2/AMH**2)
c    .    /(BETA_HDEC(RMB**2/AMH**2))**3
c     RATCOUP = 0
c     deltaqcd = QCDH(RMB**2/AMH**2)
c     RATCOUP = 1
c     deltat = QCDH(RMB**2/AMH**2) - deltaqcd
c     write(6,*)'SM: MH     = ',AMH
c     write(6,*)'alphas(MZ) = ',ALPHAS_HDEC(91.d0,3)
c     write(6,*)'alphas(MH) = ',ALPHAS_HDEC(AMH,3)
c     write(6,*)'alphas(mb) = ',ALPHAS_HDEC(AMB,3)
c     write(6,*)'mb,mb(mb)  = ',AMB,RUNM_HDEC(AMB,5)
c     write(6,*)'mb(MH,100) = ',RMB,RUNM_HDEC(100.D0,5)
c     write(6,*)'deltaqcd,t = ',deltaqcd,deltat
c     write(6,*)'Gamma(0)   = ',HB2X,HB1X
c     write(6,*)'Gamma(mb)  = ',HB2,HB1

c     ALS0 = ALPHAS_HDEC(AMZ,3)
c     WMT = RUNM_HDEC(AMT,6)
c     do i =165440,165450
c      sc = i/1000.D0
c      XMT = RUNM_HDEC(sc,6)
c      write(6,*)
c      write(6,*)sc,XMT,AMT,WMT,ALS0
c     enddo

C  H ---> TT
      IRAT = 0
      RATCOUP = 0
      IF(IONSH.EQ.0)THEN
       DLD=3D0
       DLU=5D0
       XM1 = 2D0*AMT-DLD
       XM2 = 2D0*AMT+DLU
       IF (AMH.LE.AMT+AMW+AMB) THEN
       HTT=0.D0
       ELSEIF (AMH.LE.XM1) THEN
        FACTT=6.D0*GF**2*AMH**3*AMT**2/2.D0/128.D0/PI**3
        CALL HTOTTS_HDEC(AMH,AMT,AMB,AMW,HTTS)
        HTT=FACTT*HTTS
       ELSEIF (AMH.LE.XM2) THEN
        XX(1) = XM1-1D0
        XX(2) = XM1
        XX(3) = XM2
        XX(4) = XM2+1D0
        FACTT=6.D0*GF**2*XX(1)**3*AMT**2/2.D0/128.D0/PI**3
        CALL HTOTTS_HDEC(XX(1),AMT,AMB,AMW,HTTS)
        YY(1)=FACTT*HTTS
        FACTT=6.D0*GF**2*XX(2)**3*AMT**2/2.D0/128.D0/PI**3
        CALL HTOTTS_HDEC(XX(2),AMT,AMB,AMW,HTTS)
        YY(2)=FACTT*HTTS
        XMT = RUNM_HDEC(XX(3),6)
        XY2=3.D0*HFF(XX(3),(XMT/XX(3))**2)
     .    *QCDH(XMT**2/XX(3)**2)
     .    *(1+ELW(XX(3),AMT,AMB,2/3.D0,1/2.D0))
     .    *HFFSELF(XX(3)) * SM4FACF
     .    * CPT**2
        IF(ICOUPELW.EQ.0)THEN
         XY2=3.D0*HFF(XX(3),(XMT/XX(3))**2) * CPT
     .      *QCDH(XMT**2/XX(3)**2)
     .      * ((CPT-1)
     .        +(1+ELW(XX(3),AMT,AMB,2/3.D0,1/2.D0))
     .         *HFFSELF(XX(3)) * SM4FACF)
        ENDIF
        IF(XY2.LT.0.D0) XY2 = 0
        XY1=3.D0*HFF(XX(3),(AMT/XX(3))**2)
     .    *TQCDH(AMT**2/XX(3)**2)
     .    *(1+ELW(XX(3),AMT,AMB,2/3.D0,1/2.D0))
     .    *HFFSELF(XX(3)) * SM4FACF
     .    * CPT**2
        IF(ICOUPELW.EQ.0)THEN
         XY1=3.D0*HFF(XX(3),(AMT/XX(3))**2) * CPT
     .     *TQCDH(AMT**2/XX(3)**2)
     .      * ((CPT-1)
     .        +(1+ELW(XX(3),AMT,AMB,2/3.D0,1/2.D0))
     .    *HFFSELF(XX(3)) * SM4FACF)
        ENDIF
        RAT = 2*AMT/XX(3)
        YY(3) = QQINT_HDEC(RAT,XY1,XY2)
        XMT = RUNM_HDEC(XX(4),6)
        XY2=3.D0*HFF(XX(4),(XMT/XX(4))**2)
     .    *QCDH(XMT**2/XX(4)**2)
     .    *(1+ELW(XX(4),AMT,AMB,2/3.D0,1/2.D0))
     .    *HFFSELF(XX(4)) * SM4FACF
     .    * CPT**2
        IF(ICOUPELW.EQ.0)THEN
         XY2=3.D0*HFF(XX(4),(XMT/XX(4))**2) * CPT
     .      *QCDH(XMT**2/XX(4)**2)
     .      * ((CPT-1)
     .         +(1+ELW(XX(4),AMT,AMB,2/3.D0,1/2.D0))
     .         *HFFSELF(XX(4)) * SM4FACF)
        ENDIF
        IF(XY2.LT.0.D0) XY2 = 0
        XY1=3.D0*HFF(XX(4),(AMT/XX(4))**2)
     .    *TQCDH(AMT**2/XX(4)**2)
     .    *(1+ELW(XX(4),AMT,AMB,2/3.D0,1/2.D0))
     .    *HFFSELF(XX(4)) * SM4FACF
     .    * CPT**2
        IF(ICOUPELW.EQ.0)THEN
         XY1=3.D0*HFF(XX(4),(AMT/XX(4))**2) * CPT
     .      *TQCDH(AMT**2/XX(4)**2)
     .      * ((CPT-1)
     .         +(1+ELW(XX(4),AMT,AMB,2/3.D0,1/2.D0))
     .         *HFFSELF(XX(4)) * SM4FACF)
        ENDIF
        RAT = 2*AMT/XX(4)
        YY(4) = QQINT_HDEC(RAT,XY1,XY2)
        HTT = FINT_HDEC(AMH,XX,YY)
       ELSE
        HT2=3.D0*HFF(AMH,(RMT/AMH)**2)
     .    *QCDH(RMT**2/AMH**2)
     .    *(1+ELW(AMH,AMT,AMB,2/3.D0,1/2.D0))
     .    *HFFSELF(AMH) * SM4FACF
     .    * CPT**2
        IF(ICOUPELW.EQ.0)THEN
         HT2=3.D0*HFF(AMH,(RMT/AMH)**2) * CPT
     .      *QCDH(RMT**2/AMH**2)
     .      * ((CPT-1)
     .         +(1+ELW(AMH,AMT,AMB,2/3.D0,1/2.D0))
     .         *HFFSELF(AMH) * SM4FACF)
        ENDIF
        IF(HT2.LT.0.D0) HT2 = 0
        HT1=3.D0*HFF(AMH,(AMT/AMH)**2)
     .    *TQCDH(AMT**2/AMH**2)
     .    *(1+ELW(AMH,AMT,AMB,2/3.D0,1/2.D0))
     .    *HFFSELF(AMH) * SM4FACF
     .    * CPT**2
        IF(ICOUPELW.EQ.0)THEN
         HT1=3.D0*HFF(AMH,(AMT/AMH)**2) * CPT
     .      *TQCDH(AMT**2/AMH**2)
     .      * ((CPT-1)
     .         +(1+ELW(AMH,AMT,AMB,2/3.D0,1/2.D0))
     .      *HFFSELF(AMH) * SM4FACF)
        ENDIF
        RAT = 2*AMT/AMH
        HTT = QQINT_HDEC(RAT,HT1,HT2)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c       HT0=HTT
c       TOPFAC = 0.5D0
c       RMT = RUNM_HDEC(TOPFAC*AMH,6)
c       HT2=3.D0*HFF(AMH,(RMT/AMH)**2)
c    .    *QCDH(RMT**2/AMH**2)
c    .    *HFFSELF(AMH)
c       HTL = QQINT_HDEC(RAT,HT1,HT2)
c       TOPFAC = 2
c       RMT = RUNM_HDEC(TOPFAC*AMH,6)
c       HT2=3.D0*HFF(AMH,(RMT/AMH)**2)
c    .    *QCDH(RMT**2/AMH**2)
c    .    *HFFSELF(AMH)
c       HTU = QQINT_HDEC(RAT,HT1,HT2)
c       write(6,*)'H -> TT: ',AMH,HT0,HTL/HT0,HTU/HT0
c       TOPFAC = 1
c       RMT = RUNM_HDEC(AMH,6)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
       ENDIF
      ELSE
       IF (AMH.LE.2.D0*AMT) THEN
        HTT=0.D0
       ELSE
        HT2=3.D0*HFF(AMH,(RMT/AMH)**2)
     .    *QCDH(RMT**2/AMH**2)
     .    *(1+ELW(AMH,AMT,AMB,2/3.D0,1/2.D0))
     .    *HFFSELF(AMH) * SM4FACF
     .    * CPT**2
        IF(ICOUPELW.EQ.0)THEN
         HT2=3.D0*HFF(AMH,(RMT/AMH)**2) * CPT
     .      *QCDH(RMT**2/AMH**2)
     .      * ((CPT-1)
     .         +(1+ELW(AMH,AMT,AMB,2/3.D0,1/2.D0))
     .         *HFFSELF(AMH) * SM4FACF)
        ENDIF
        IF(HT2.LT.0.D0) HT2 = 0
        HT1=3.D0*HFF(AMH,(AMT/AMH)**2)
     .    *TQCDH(AMT**2/AMH**2)
     .    *(1+ELW(AMH,AMT,AMB,2/3.D0,1/2.D0))
     .    *HFFSELF(AMH) * SM4FACF
     .    * CPT**2
        IF(ICOUPELW.EQ.0)THEN
         HT1=3.D0*HFF(AMH,(AMT/AMH)**2) * CPT
     .      *TQCDH(AMT**2/AMH**2)
     .      * ((CPT-1)
     .         +(1+ELW(AMH,AMT,AMB,2/3.D0,1/2.D0))
     .         *HFFSELF(AMH) * SM4FACF)
        ENDIF
        RAT = 2*AMT/AMH
        HTT = QQINT_HDEC(RAT,HT1,HT2)
       ENDIF
      ENDIF
c     HTT=HTT*SM4FACF
      IF(IFERMPHOB.NE.0)THEN
c      write(6,*)'Fermiophobic: H -> mu mu,tau tau,ss,cc,bb,tt ---> 0'
       HMM = 0
       HLL = 0
       HSS = 0
       HCC = 0
       HBB = 0
       HTT = 0
      ENDIF
C  H ---> GAMMA GAMMA
       EPS=1.D-8
       XRMC = RUNM_HDEC(AMH/2,4)*AMC/RUNM_HDEC(AMC,4)
       XRMB = RUNM_HDEC(AMH/2,5)*AMB/RUNM_HDEC(AMB,5)
       XRMT = RUNM_HDEC(AMH/2,6)*AMT/RUNM_HDEC(AMT,6)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c      XRMC = AMC
c      XRMB = AMB
c      XRMT = AMT
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
       CTT = 4*XRMT**2/AMH**2*DCMPLX(1D0,-EPS)
       CTB = 4*XRMB**2/AMH**2*DCMPLX(1D0,-EPS)
       CTC = 4*XRMC**2/AMH**2*DCMPLX(1D0,-EPS)
       CTL = 4*AMTAU**2/AMH**2*DCMPLX(1D0,-EPS)
       CTW = 4*AMW**2/AMH**2*DCMPLX(1D0,-EPS)
       CAW = -(2+3*CTW+3*CTW*(2-CTW)*CF(CTW)) * CPW
       CAW00= -(2+3*CTW+3*CTW*(2-CTW)*CF(CTW))
       CAT00= 4/3D0 * 2*CTT*(1+(1-CTT)*CF(CTT))
       CAB00= 1/3D0 * 2*CTB*(1+(1-CTB)*CF(CTB))
       CAC00= 4/3D0 * 2*CTC*(1+(1-CTC)*CF(CTC))
       CAL00 =         2*CTL*(1+(1-CTL)*CF(CTL))
       CAT0= 4/3D0 * 2*CTT*(1+(1-CTT)*CF(CTT)) * CPT
       CAB0= 1/3D0 * 2*CTB*(1+(1-CTB)*CF(CTB)) * CPB
       CAC0= 4/3D0 * 2*CTC*(1+(1-CTC)*CF(CTC)) * CPC
       CAT = 4/3D0 * 2*CTT*(1+(1-CTT)*CF(CTT)) * CPT
     .     * CFACQ_HDEC(0,AMH,XRMT)
       CAB = 1/3D0 * 2*CTB*(1+(1-CTB)*CF(CTB)) * CPB
     .     * CFACQ_HDEC(0,AMH,XRMB)
       CAC = 4/3D0 * 2*CTC*(1+(1-CTC)*CF(CTC)) * CPC
     .     * CFACQ_HDEC(0,AMH,XRMC)
       CAL =         2*CTL*(1+(1-CTL)*CF(CTL)) * CPTAU
       CATP = 0
       CABP = 0
       CAEP = 0
       XFAC0= CDABS(CAT0+CAB0+CAC0+CAL+CAW+CPGAGA)**2*GAGA_ELW(AMT,AMH)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c       DUM1 = CDABS(CAT+CAB+CAC+CAL+CAW)**2
c       CKOFQ = CKOFQ_HDEC(0,AMH**2/XRMT**2)
c       CFACQ = CKOFQ*ALPHAS_HDEC(AMH,3)/PI
c       CFACQ0 = -ALPHAS_HDEC(AMH,3)/PI
c       CATX = 4/3D0 * 2*CTT*(1+(1-CTT)*CF(CTT)) * CPT
c    .       * (1+CFACQ0)
c       DUM2 = CDABS(CATX+CAB+CAC+CAL+CAW)**2
c       write(6,*)'H -> gamma gamma: ',CKOFQ,CFACQ,DUM1
c       write(6,*)'H -> gamma gamma: ',-1.D0,CFACQ0,DUM2,DUM1/DUM2
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
       IF(ICOUPELW.EQ.0)THEN
c       XFAC0= CDABS(CAT00+CAB00+CAC00+CAL00+CAW00)**2*GAGA_ELW(AMT,AMH)
        XFAC0= DREAL(DCONJG(CAT0+CAB0+CAC0+CAL+CAW+CPGAGA)
     .       * (CAT00+CAB00+CAC00+CAL00+CAW00))*GAGA_ELW(AMT,AMH)
       ENDIF
       IF(ISM4.NE.0)THEN
        XRMBP = RUNM_HDEC(AMH/2,7)*AMBP/RUNM_HDEC(AMBP,7)
        XRMTP = RUNM_HDEC(AMH/2,8)*AMTP/RUNM_HDEC(AMTP,8)
        CTTP = 4*XRMTP**2/AMH**2*DCMPLX(1D0,-EPS)
        CTBP = 4*XRMBP**2/AMH**2*DCMPLX(1D0,-EPS)
        CTEP = 4*AMEP**2/AMH**2*DCMPLX(1D0,-EPS)
        CATP00= 4/3D0 * 2*CTTP*(1+(1-CTTP)*CF(CTTP))
        CABP00= 1/3D0 * 2*CTBP*(1+(1-CTBP)*CF(CTBP))
        CAEP00=         2*CTEP*(1+(1-CTEP)*CF(CTEP))
        CATP0= 4/3D0 * 2*CTTP*(1+(1-CTTP)*CF(CTTP)) * CPTP
        CABP0= 1/3D0 * 2*CTBP*(1+(1-CTBP)*CF(CTBP)) * CPBP
        CATP = 4/3D0 * 2*CTTP*(1+(1-CTTP)*CF(CTTP)) * CPTP
     .       * CFACQ_HDEC(0,AMH,XRMTP)
        CABP = 1/3D0 * 2*CTBP*(1+(1-CTBP)*CF(CTBP)) * CPBP
     .       * CFACQ_HDEC(0,AMH,XRMBP)
        CAEP =         2*CTEP*(1+(1-CTEP)*CF(CTEP)) * CPEP
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c       CATP = 0
c       CABP = 0
c       CAEP = 0
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        IF(AMNUP.EQ.AMEP)THEN
         XXL = AMEP**2
        ELSE
         XXL =AMNUP**2*AMEP**2/(AMNUP**2-AMEP**2)*DLOG(AMNUP**2/AMEP**2)
        ENDIF
        IF(AMTP.EQ.AMBP)THEN
         XXQ = AMTP**2
        ELSE
         XXQ = AMTP**2*AMBP**2/(AMTP**2-AMBP**2)*DLOG(AMTP**2/AMBP**2)
        ENDIF
        FFT = -6
        FFT = 1
        CELW = GF/8/DSQRT(2.D0)/PI**2*(-49*AMT**2/2*FFT
     .       + 7*AMNUP**2/6 - 65*AMEP**2/6 - XXL
     .       - 237*AMTP**2/10 - 117*AMBP**2/10 - 3*XXQ)
        CELW = CELW*(1-0.614D0*AMH**2/4/AMW**2)
        XFAC0= CDABS(CAT0+CAB0+CAC0+CAL+CAW+CATP0+CABP0+CAEP+CPGAGA)**2
     .       * ((1+CELW)**2 - 1)
       IF(ICOUPELW.EQ.0)THEN
        XFAC0= DREAL(DCONJG(CAT0+CAB0+CAC0+CAL+CAW+CATP0+CABP0+CAEP
     .                     +CPGAGA)
     .       * (CAT00+CAB00+CAC00+CAL00+CAW00+CATP00+CABP00+CAEP00))
     .       * ((1+CELW)**2 - 1)
       ENDIF
c       write(6,*)'H -> ga ga: ',AMH,2*CELW*100,((1+CELW)**2-1)*100
c       write(6,*)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c       XFAC0= 0
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
       ENDIF
       IF(IFERMPHOB.NE.0)THEN
c       write(6,*)'Fermiophobic: H -> gamma gamma ---> W loop'
        CAL = 0
        CAC = 0
        CAB = 0
        CAT = 0
        XFAC0= 0
c       write(6,*)CDABS(CAW)**2,CDABS(CAT+CAB+CAC+CAL+CAW)**2
       ENDIF
       XFAC = CDABS(CAT+CAB+CAC+CAL+CAW+CATP+CABP+CAEP+CPGAGA)**2
       HGA=HVV(AMH,0.D0)*(ALPH/PI)**2/16.D0*(XFAC+XFAC0)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c      write(6,*)'H -> gamma gamma: ',XFAC,XFAC0,HGA
c      write(6,*)'H -> gamma gamma: ',CAT0,CAB0,CAC0,CAL,CAW
c    .                               ,CATP0,CABP0,CAEP,CGAGA
c      write(6,*)'H -> gamma gamma: ',CAT00,CAB00,CAC00,CAL00,CAW00
c    .                               ,CATP00,CABP00,CAEP00
c      write(6,*)'H -> gamma gamma: ',AMH,XRMT,XRMB,XRMC
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c      write(6,*)'H -> gamma gamma: ',
c    .           dsqrt(xfac)/8
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c       write(6,*)'H gamma gamma: ',(16/9.d0-7)/8,(CAT0+CAW)/8,
c    .                              (CAT0+CAB0+CAC0+CAL+CAW)/8
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c      HGA=HVV(AMH,0.D0)*(ALPH/PI)**2/16.D0*XFAC
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c      write(6,*)'H -> gamma gamma: ',HGA,XFAC0/XFAC
c      write(6,*)'gaga: ',AMH,CAT,CAW,CAT+CAW
c      write(6,*)'gaga: ',AMH,CAT0,CAW,CAT0+CAW
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c      YY0 = CDABS(CAT+CAC)**2 + CDABS(CAT0+CAC0)**2*GAGA_ELW(AMT,AMH)
c      YTT = HVV(AMH,0.D0)*(ALPH/PI)**2/16.D0*YY0 / HGA
c      YY0 = CDABS(CAB)**2 + CDABS(CAB0)**2*GAGA_ELW(AMT,AMH)
c      YBB = HVV(AMH,0.D0)*(ALPH/PI)**2/16.D0*YY0 / HGA
c      YY0 = CDABS(CAW)**2 + CDABS(CAW)**2*GAGA_ELW(AMT,AMH)
c      YWW = HVV(AMH,0.D0)*(ALPH/PI)**2/16.D0*YY0 / HGA
c      YY0 = CDABS(CAL)**2 + CDABS(CAL)**2*GAGA_ELW(AMT,AMH)
c      YLL = HVV(AMH,0.D0)*(ALPH/PI)**2/16.D0*YY0 / HGA
c      YY0 = 2*DREAL(DCONJG(CAT+CAC)*CAB)
c    .     + 2*DREAL(DCONJG(CAT0+CAC0)*CAB0)*GAGA_ELW(AMT,AMH)
c      YTB = HVV(AMH,0.D0)*(ALPH/PI)**2/16.D0*YY0 / HGA
c      YY0 = 2*DREAL(DCONJG(CAT+CAC)*CAW)
c    .     + 2*DREAL(DCONJG(CAT0+CAC0)*CAW)*GAGA_ELW(AMT,AMH)
c      YTW = HVV(AMH,0.D0)*(ALPH/PI)**2/16.D0*YY0 / HGA
c      YY0 = 2*DREAL(DCONJG(CAB)*CAW)
c    .     + 2*DREAL(DCONJG(CAB0)*CAW)*GAGA_ELW(AMT,AMH)
c      YBW = HVV(AMH,0.D0)*(ALPH/PI)**2/16.D0*YY0 / HGA
c      YY0 = 2*DREAL(DCONJG(CAT+CAC)*CAL)
c    .     + 2*DREAL(DCONJG(CAT0+CAC0)*CAL)*GAGA_ELW(AMT,AMH)
c      YTL = HVV(AMH,0.D0)*(ALPH/PI)**2/16.D0*YY0 / HGA
c      YY0 = 2*DREAL(DCONJG(CAB)*CAL)
c    .     + 2*DREAL(DCONJG(CAB0)*CAL)*GAGA_ELW(AMT,AMH)
c      YBL = HVV(AMH,0.D0)*(ALPH/PI)**2/16.D0*YY0 / HGA
c      YY0 = 2*DREAL(DCONJG(CAL)*CAW)
c    .     + 2*DREAL(DCONJG(CAL)*CAW)*GAGA_ELW(AMT,AMH)
c      YLW = HVV(AMH,0.D0)*(ALPH/PI)**2/16.D0*YY0 / HGA
c      write(52,('f4.0,7(g13.5)'))AMH,HGA,YTT,YBB,YWW,YTB,YTW,YBW
c    .                            ,YLL,YTL,YBL,YLW
c      write(6,*)'gaga: ',AMH,YTT+YBB+YWW+YTB+YTW+YBW+YLL+YTL+YBL+YLW
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c      write(6,*)AMH,GAGA_ELW(AMT,AMH)*100
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c      CAT0 = 4/3D0 * 2*CTT*(1+(1-CTT)*CF(CTT))
c      CAB0 = 1/3D0 * 2*CTB*(1+(1-CTB)*CF(CTB))
c      CAC0 = 4/3D0 * 2*CTC*(1+(1-CTC)*CF(CTC))
c      XFACLO = CDABS(CAT0+CAB0+CAC0+CAL+CAW)**2
c      write(54,('4(1X,E12.6)'))AMH,HGA,HGA*XFACLO/XFAC,XFAC/XFACLO-1
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c      write(6,*)CDABS(CAT0+CAW)**2,(16/9.D0-7)**2,
c    .           (16/9.D0-7)**2/CDABS(CAT0+CAW)**2,16/9.D0,CAT0,-7,CAW
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
C  H ---> Z GAMMA
      XRMC = RUNM_HDEC(AMH/2,4)*AMC/RUNM_HDEC(AMC,4)
      XRMB = RUNM_HDEC(AMH/2,5)*AMB/RUNM_HDEC(AMB,5)
      XRMT = RUNM_HDEC(AMH/2,6)*AMT/RUNM_HDEC(AMT,6)
      IF(AMH.LE.AMZ)THEN
       HZGA=0
      ELSE
       EPS=1.D-8
       TS = SS/CS
       FT = -3*2D0/3*(1-4*2D0/3*SS)/DSQRT(SS*CS) * CPT
       FB = 3*1D0/3*(-1+4*1D0/3*SS)/DSQRT(SS*CS) * CPB
       FC = -3*2D0/3*(1-4*2D0/3*SS)/DSQRT(SS*CS) * CPC
       FL = (-1+4*SS)/DSQRT(SS*CS) * CPTAU
c      CTT = 4*XRMT**2/AMH**2*DCMPLX(1D0,-EPS)
c      CTB = 4*XRMB**2/AMH**2*DCMPLX(1D0,-EPS)
c      CTC = 4*XRMC**2/AMH**2*DCMPLX(1D0,-EPS)
       CTT = 4*AMT**2/AMH**2*DCMPLX(1D0,-EPS)
       CTB = 4*AMB**2/AMH**2*DCMPLX(1D0,-EPS)
       CTC = 4*AMC**2/AMH**2*DCMPLX(1D0,-EPS)
       CTL = 4*AMTAU**2/AMH**2*DCMPLX(1D0,-EPS)
       CTW = 4*AMW**2/AMH**2*DCMPLX(1D0,-EPS)
c      CLT = 4*XRMT**2/AMZ**2*DCMPLX(1D0,-EPS)
c      CLB = 4*XRMB**2/AMZ**2*DCMPLX(1D0,-EPS)
c      CLC = 4*XRMC**2/AMZ**2*DCMPLX(1D0,-EPS)
       CLT = 4*AMT**2/AMZ**2*DCMPLX(1D0,-EPS)
       CLB = 4*AMB**2/AMZ**2*DCMPLX(1D0,-EPS)
       CLC = 4*AMC**2/AMZ**2*DCMPLX(1D0,-EPS)
       CLE = 4*AMTAU**2/AMZ**2*DCMPLX(1D0,-EPS)
       CLW = 4*AMW**2/AMZ**2*DCMPLX(1D0,-EPS)
       CAT = FT*(CI1(CTT,CLT) - CI2(CTT,CLT))
       CAB = FB*(CI1(CTB,CLB) - CI2(CTB,CLB))
       CAC = FC*(CI1(CTC,CLC) - CI2(CTC,CLC))
       CAL = FL*(CI1(CTL,CLE) - CI2(CTL,CLE))
       CAW = -1/DSQRT(TS)*(4*(3-TS)*CI2(CTW,CLW)
     .     + ((1+2/CTW)*TS - (5+2/CTW))*CI1(CTW,CLW)) * CPW
       CATP = 0
       CABP = 0
       CAEP = 0
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c      write(6,*)6*CI1(CTT,CLT),2*CI2(CTT,CLT)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
       IF(ISM4.NE.0)THEN
        FTP = -3*2D0/3*(1-4*2D0/3*SS)/DSQRT(SS*CS) * CPTP
        FBP = 3*1D0/3*(-1+4*1D0/3*SS)/DSQRT(SS*CS) * CPBP
        FEP = (-1+4*SS)/DSQRT(SS*CS) * CPEP
        CTTP = 4*AMTP**2/AMH**2*DCMPLX(1D0,-EPS)
        CTBP = 4*AMBP**2/AMH**2*DCMPLX(1D0,-EPS)
        CTEP = 4*AMEP**2/AMH**2*DCMPLX(1D0,-EPS)
        CLTP = 4*AMTP**2/AMZ**2*DCMPLX(1D0,-EPS)
        CLBP = 4*AMBP**2/AMZ**2*DCMPLX(1D0,-EPS)
        CLEP = 4*AMEP**2/AMZ**2*DCMPLX(1D0,-EPS)
        CATP = FTP*(CI1(CTTP,CLTP) - CI2(CTTP,CLTP))
        CABP = FBP*(CI1(CTBP,CLBP) - CI2(CTBP,CLBP))
        CAEP = FEP*(CI1(CTEP,CLEP) - CI2(CTEP,CLEP))
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c       CATP = 0
c       CABP = 0
c       CAEP = 0
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
       ENDIF
       IF(IFERMPHOB.NE.0)THEN
c       write(6,*)'Fermiophobic: H -> Z gamma ---> W loop'
        CAB = 0
        CAT = 0
        CAL = 0
c       write(6,*)CDABS(CAW)**2,CDABS(CAT+CAB+CAW)**2
       ENDIF
       FPTLIKE = CPZGA
       XFAC = CDABS(CAT+CAB+CAC+CAL+CAW+CATP+CABP+CAEP+FPTLIKE)**2
       ACOUP = DSQRT(2D0)*GF*AMZ**2*SS*CS/PI**2
       HZGA = GF/(4.D0*PI*DSQRT(2.D0))*AMH**3*(ALPH/PI)*ACOUP/16.D0
     .        *XFAC*(1-AMZ**2/AMH**2)**3
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c      write(6,*)'H -> Z gamma: ',
c    .           1-CDABS(CAB+CAC+CAL+CAW+CATP+CABP+CAEP)**2/XFAC
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c      YY0 = CDABS(CAT+CAC)**2
c      YTT = GF/(4.D0*PI*DSQRT(2.D0))*AMH**3*(ALPH/PI)*ACOUP/16.D0
c    .     * YY0*(1-AMZ**2/AMH**2)**3 / HZGA
c      YY0 = CDABS(CAB)**2
c      YBB = GF/(4.D0*PI*DSQRT(2.D0))*AMH**3*(ALPH/PI)*ACOUP/16.D0
c    .     * YY0*(1-AMZ**2/AMH**2)**3 / HZGA
c      YY0 = CDABS(CAW)**2
c      YWW = GF/(4.D0*PI*DSQRT(2.D0))*AMH**3*(ALPH/PI)*ACOUP/16.D0
c    .     * YY0*(1-AMZ**2/AMH**2)**3 / HZGA
c      YY0 = CDABS(CAL)**2
c      YLL = GF/(4.D0*PI*DSQRT(2.D0))*AMH**3*(ALPH/PI)*ACOUP/16.D0
c    .     * YY0*(1-AMZ**2/AMH**2)**3 / HZGA
c      YY0 = 2*DREAL(DCONJG(CAT+CAC)*CAB)
c      YTB = GF/(4.D0*PI*DSQRT(2.D0))*AMH**3*(ALPH/PI)*ACOUP/16.D0
c    .     * YY0*(1-AMZ**2/AMH**2)**3 / HZGA
c      YY0 = 2*DREAL(DCONJG(CAT+CAC)*CAW)
c      YTW = GF/(4.D0*PI*DSQRT(2.D0))*AMH**3*(ALPH/PI)*ACOUP/16.D0
c    .     * YY0*(1-AMZ**2/AMH**2)**3 / HZGA
c      YY0 = 2*DREAL(DCONJG(CAB)*CAW)
c      YBW = GF/(4.D0*PI*DSQRT(2.D0))*AMH**3*(ALPH/PI)*ACOUP/16.D0
c    .     * YY0*(1-AMZ**2/AMH**2)**3 / HZGA
c      YY0 = 2*DREAL(DCONJG(CAT+CAC)*CAL)
c      YTL = GF/(4.D0*PI*DSQRT(2.D0))*AMH**3*(ALPH/PI)*ACOUP/16.D0
c    .     * YY0*(1-AMZ**2/AMH**2)**3 / HZGA
c      YY0 = 2*DREAL(DCONJG(CAB)*CAL)
c      YBL = GF/(4.D0*PI*DSQRT(2.D0))*AMH**3*(ALPH/PI)*ACOUP/16.D0
c    .     * YY0*(1-AMZ**2/AMH**2)**3 / HZGA
c      YY0 = 2*DREAL(DCONJG(CAL)*CAW)
c      YLW = GF/(4.D0*PI*DSQRT(2.D0))*AMH**3*(ALPH/PI)*ACOUP/16.D0
c    .     * YY0*(1-AMZ**2/AMH**2)**3 / HZGA
c      write(53,('f4.0,11(g13.5)'))AMH,HZGA,YTT,YBB,YWW,YTB,YTW,YBW
c    .                            ,YLL,YTL,YBL,YLW
c      write(6,*)'Zga: ',AMH,YTT+YBB+YWW+YTB+YTW+YBW+YLL+YTL+YBL+YLW
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c      XFAC0 = CDABS(CAT+CAB+CAL+CAW+CATP+CABP+CAEP)**2
c      write(6,*)'H -> Z gamma: ',XFAC/XFAC0
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      ENDIF
C  H ---> W W
      IF(IONWZ.EQ.0)THEN
       IF(IFERMPHOB.NE.0)THEN
        CALL HTOVV_HDEC(0,AMH,AMW,GAMW,HTWW)
        HTWW = HTWW*HVVSELF(AMH)
       ELSE
        CALL HTOVV_HDEC(1,AMH,AMW,GAMW,HTWW)
        CALL HTOVV_HDEC(0,AMH,AMW,GAMW,HTWW0)
        HTWW = HTWW * SM4FACW
       ENDIF
       HWW = 3D0/2D0*GF*AMW**4/DSQRT(2D0)/PI/AMH**3*HTWW * CPW**2
       IF(ICOUPELW.EQ.0)THEN
        HWW = 3D0/2D0*GF*AMW**4/DSQRT(2D0)/PI/AMH**3
     .      * CPW * (HTWW0*(CPW-1)+HTWW)
       ENDIF
      ELSEIF(IONWZ.EQ.-1)THEN
       DLD=2D0
       DLU=2D0
       XM1 = 2D0*AMW-DLD
       XM2 = 2D0*AMW+DLU
      IF (AMH.LE.AMW) THEN
       HWW=0
      ELSE IF (AMH.LE.XM1) THEN
       CWW=3.D0*GF**2*AMW**4/16.D0/PI**3
       HWW=HV(AMW**2/AMH**2)*CWW*AMH * CPW**2
     .      * SM4FACW
       IF(ICOUPELW.EQ.0)THEN
        HWW=HV(AMW**2/AMH**2)*CWW*AMH
     .      * CPW * ((CPW-1)+SM4FACW)
       ENDIF
      ELSE IF (AMH.LT.XM2) THEN
       CWW=3.D0*GF**2*AMW**4/16.D0/PI**3
       XX(1) = XM1-1D0
       XX(2) = XM1
       XX(3) = XM2
       XX(4) = XM2+1D0
       YY(1)=HV(AMW**2/XX(1)**2)*CWW*XX(1) * SM4FACW
       YY(2)=HV(AMW**2/XX(2)**2)*CWW*XX(2) * SM4FACW
       YY(3)=HVV(XX(3),AMW**2/XX(3)**2)
     .      *HVVSELF(XX(3)) * SM4FACW
       YY(4)=HVV(XX(4),AMW**2/XX(4)**2)
     .      *HVVSELF(XX(4)) * SM4FACW
       IF(ICOUPELW.EQ.0)THEN
        YY(1)=HV(AMW**2/XX(1)**2)*CWW*XX(1)
     .       * CPW * ((CPW-1)+SM4FACW)
        YY(2)=HV(AMW**2/XX(2)**2)*CWW*XX(2)
     .       * CPW * ((CPW-1)+SM4FACW)
        YY(3)=HVV(XX(3),AMW**2/XX(3)**2)
     .       * CPW * ((CPW-1)+HVVSELF(XX(3)) * SM4FACW)
        YY(4)=HVV(XX(4),AMW**2/XX(4)**2)
     .       * CPW * ((CPW-1)+HVVSELF(XX(4)) * SM4FACW)
        ENDIF
       HWW = FINT_HDEC(AMH,XX,YY)
      ELSE
       HWW=HVV(AMH,AMW**2/AMH**2)
     .     *HVVSELF(AMH) * SM4FACW * CPW**2
       IF(ICOUPELW.EQ.0)THEN
        HWW=HVV(AMH,AMW**2/AMH**2)
     .      * CPW * ((CPW-1)+HVVSELF(AMH) * SM4FACW)
       ENDIF
      ENDIF
      ELSE
       DLD=2D0
       DLU=2D0
       XM1 = 2D0*AMW-DLD
       XM2 = 2D0*AMW+DLU
       IF (AMH.LE.XM1) THEN
        CALL HTOVV_HDEC(0,AMH,AMW,GAMW,HTWW)
        HWW = 3D0/2D0*GF*AMW**4/DSQRT(2D0)/PI/AMH**3*HTWW * CPW**2
     .      * SM4FACW
        IF(ICOUPELW.EQ.0)THEN
         HWW = 3D0/2D0*GF*AMW**4/DSQRT(2D0)/PI/AMH**3*HTWW
     .       * CPW * ((CPW-1)+SM4FACW)
        ENDIF
       ELSEIF (AMH.LE.XM2) THEN
        XX(1) = XM1-1D0
        XX(2) = XM1
        XX(3) = XM2
        XX(4) = XM2+1D0
        CALL HTOVV_HDEC(0,XX(1),AMW,GAMW,HTWW)
        YY(1)=3D0/2D0*GF*AMW**4/DSQRT(2D0)/PI/XX(1)**3*HTWW * CPW**2
     .      * SM4FACW
        IF(ICOUPELW.EQ.0)THEN
         YY(1)=3D0/2D0*GF*AMW**4/DSQRT(2D0)/PI/XX(1)**3*HTWW
     .        * CPW * ((CPW-1) + SM4FACW)
        ENDIF
        CALL HTOVV_HDEC(0,XX(2),AMW,GAMW,HTWW)
        YY(2)=3D0/2D0*GF*AMW**4/DSQRT(2D0)/PI/XX(2)**3*HTWW * CPW**2
     .      * SM4FACW
        IF(ICOUPELW.EQ.0)THEN
         YY(2)=3D0/2D0*GF*AMW**4/DSQRT(2D0)/PI/XX(2)**3*HTWW * CPW**2
     .        * CPW * ((CPW-1) + SM4FACW)
        ENDIF
        YY(3)=HVV(XX(3),AMW**2/XX(3)**2)
     .       *HVVSELF(XX(3)) * SM4FACW * CPW**2
        YY(4)=HVV(XX(4),AMW**2/XX(4)**2)
     .       *HVVSELF(XX(4)) * SM4FACW * CPW**2
       IF(ICOUPELW.EQ.0)THEN
        YY(3)=HVV(XX(3),AMW**2/XX(3)**2)
     .      * CPW * ((CPW-1) + HVVSELF(XX(3)) * SM4FACW)
        YY(4)=HVV(XX(4),AMW**2/XX(4)**2)
     .      * CPW * ((CPW-1) + HVVSELF(XX(4)) * SM4FACW)
       ENDIF
        HWW = FINT_HDEC(AMH,XX,YY)
       ELSE
        HWW=HVV(AMH,AMW**2/AMH**2)
     .     *HVVSELF(AMH) * SM4FACW * CPW**2
        IF(ICOUPELW.EQ.0)THEN
         HWW=HVV(AMH,AMW**2/AMH**2)
     .      * CPW * ((CPW-1)+HVVSELF(AMH) * SM4FACW)
        ENDIF
       ENDIF
      ENDIF
c     HWW = HWW * SM4FACW
C  H ---> Z Z
      IF(IONWZ.EQ.0)THEN
       IF(IFERMPHOB.NE.0)THEN
        CALL HTOVV_HDEC(0,AMH,AMZ,GAMZ,HTZZ)
        HTZZ = HTZZ*HVVSELF(AMH)
       ELSE
        CALL HTOVV_HDEC(2,AMH,AMZ,GAMZ,HTZZ)
        CALL HTOVV_HDEC(0,AMH,AMZ,GAMZ,HTZZ0)
        HTZZ = HTZZ * SM4FACZ
       ENDIF
       HZZ = 3D0/4D0*GF*AMZ**4/DSQRT(2D0)/PI/AMH**3*HTZZ * CPZ**2
       IF(ICOUPELW.EQ.0)THEN
        HZZ = 3D0/4D0*GF*AMZ**4/DSQRT(2D0)/PI/AMH**3
     .      * CPZ * (HTZZ0*(CPZ-1)+HTZZ)
       ENDIF
      ELSEIF(IONWZ.EQ.-1)THEN
       DLD=2D0
       DLU=2D0
       XM1 = 2D0*AMZ-DLD
       XM2 = 2D0*AMZ+DLU
       IF (AMH.LE.AMZ) THEN
        HZZ=0
       ELSEIF (AMH.LE.XM1) THEN
        CZZ=3.D0*GF**2*AMZ**4/192.D0/PI**3*(7-40/3.D0*SS+160/9.D0*SS**2)
        HZZ=HV(AMZ**2/AMH**2)*CZZ*AMH * CPZ**2
     .      * SM4FACZ
        IF(ICOUPELW.EQ.0)THEN
         HZZ=HV(AMZ**2/AMH**2)*CZZ*AMH
     .      * CPZ * ((CPZ-1)+SM4FACZ)
        ENDIF
       ELSEIF (AMH.LT.XM2) THEN
        CZZ=3.D0*GF**2*AMZ**4/192.D0/PI**3*(7-40/3.D0*SS+160/9.D0*SS**2)
        XX(1) = XM1-1D0
        XX(2) = XM1
        XX(3) = XM2
        XX(4) = XM2+1D0
        YY(1)=HV(AMZ**2/XX(1)**2)*CZZ*XX(1) * CPZ**2
     .      * SM4FACZ
        YY(2)=HV(AMZ**2/XX(2)**2)*CZZ*XX(2) * CPZ**2
     .      * SM4FACZ
        YY(3)=HVV(XX(3),AMZ**2/XX(3)**2)/2
     .       *HVVSELF(XX(3)) * SM4FACZ * CPZ**2
        YY(4)=HVV(XX(4),AMZ**2/XX(4)**2)/2
     .       *HVVSELF(XX(4)) * SM4FACZ * CPZ**2
        IF(ICOUPELW.EQ.0)THEN
         YY(1)=HV(AMZ**2/XX(1)**2)*CZZ*XX(1)
     .        * CPZ * ((CPZ-1)+SM4FACZ)
         YY(2)=HV(AMZ**2/XX(2)**2)*CZZ*XX(2)
     .        * CPZ * ((CPZ-1)+SM4FACZ)
         YY(3)=HVV(XX(3),AMZ**2/XX(3)**2)/2
     .        * CPZ * ((CPZ-1)+HVVSELF(XX(3)) * SM4FACZ)
         YY(4)=HVV(XX(4),AMZ**2/XX(4)**2)/2
     .        * CPZ * ((CPZ-1)+HVVSELF(XX(4)) * SM4FACZ)
        ENDIF
        HZZ = FINT_HDEC(AMH,XX,YY)
       ELSE
        HZZ=HVV(AMH,AMZ**2/AMH**2)/2.D0
     .     *HVVSELF(AMH) * SM4FACZ * CPZ**2
        IF(ICOUPELW.EQ.0)THEN
         HZZ=HVV(AMH,AMZ**2/AMH**2)/2.D0
     .      * CPZ * ((CPZ-1)+HVVSELF(AMH) * SM4FACZ)
        ENDIF
       ENDIF
      ELSE
       DLD=2D0
       DLU=2D0
       XM1 = 2D0*AMZ-DLD
       XM2 = 2D0*AMZ+DLU
       IF (AMH.LE.XM1) THEN
        CALL HTOVV_HDEC(0,AMH,AMZ,GAMZ,HTZZ)
        HZZ = 3D0/4D0*GF*AMZ**4/DSQRT(2D0)/PI/AMH**3*HTZZ
     .      * SM4FACZ
        IF(ICOUPELW.EQ.0)THEN
         HZZ = 3D0/4D0*GF*AMZ**4/DSQRT(2D0)/PI/AMH**3*HTZZ
     .       * CPZ * ((CPZ-1)+SM4FACZ)
        ENDIF
       ELSEIF (AMH.LE.XM2) THEN
        XX(1) = XM1-1D0
        XX(2) = XM1
        XX(3) = XM2
        XX(4) = XM2+1D0
        CALL HTOVV_HDEC(0,XX(1),AMZ,GAMZ,HTZZ)
        YY(1)=3D0/4D0*GF*AMZ**4/DSQRT(2D0)/PI/XX(1)**3*HTZZ * CPZ**2
     .       * SM4FACZ
        IF(ICOUPELW.EQ.0)THEN
         YY(1)=3D0/4D0*GF*AMZ**4/DSQRT(2D0)/PI/XX(1)**3*HTZZ
     .        * CPZ * ((CPZ-1) + SM4FACZ)
        ENDIF
        CALL HTOVV_HDEC(0,XX(2),AMZ,GAMZ,HTZZ)
        YY(2)=3D0/4D0*GF*AMZ**4/DSQRT(2D0)/PI/XX(2)**3*HTZZ * CPZ**2
     .       * SM4FACZ
        IF(ICOUPELW.EQ.0)THEN
         YY(2)=3D0/4D0*GF*AMZ**4/DSQRT(2D0)/PI/XX(2)**3*HTZZ
     .        * CPZ * ((CPZ-1) + SM4FACZ)
        ENDIF
        YY(3)=HVV(XX(3),AMZ**2/XX(3)**2)/2
     .       *HVVSELF(XX(3)) * SM4FACZ * CPZ**2
        YY(4)=HVV(XX(4),AMZ**2/XX(4)**2)/2
     .       *HVVSELF(XX(4)) * SM4FACZ * CPZ**2
        IF(ICOUPELW.EQ.0)THEN
         YY(3)=HVV(XX(3),AMZ**2/XX(3)**2)/2
     .        * CPZ * ((CPZ-1)+HVVSELF(XX(3)))
         YY(4)=HVV(XX(4),AMZ**2/XX(4)**2)/2
     .        * CPZ * ((CPZ-1)+HVVSELF(XX(4)))
        ENDIF
        HZZ = FINT_HDEC(AMH,XX,YY)
       ELSE
        HZZ=HVV(AMH,AMZ**2/AMH**2)/2.D0
     .      *HVVSELF(AMH) * SM4FACZ * CPZ**2
       IF(ICOUPELW.EQ.0)THEN
        HZZ=HVV(AMH,AMZ**2/AMH**2)/2.D0
     .     * CPZ * ((CPZ-1)+HVVSELF(AMH) * SM4FACZ)
       ENDIF
       ENDIF
      ENDIF
c     HZZ = HZZ * SM4FACZ
c     write(62,*)AMH,1.D0+GF*AMH**2/16.D0/PI**2/DSQRT(2.D0)*2.800952D0,
c    .               HVVSELF(AMH)
c     write(91,*)AMH,HWW,HZZ
C
      HNUPNUP = 0
      HEPEP = 0
      HBPBP = 0
      HTPTP = 0
      IF(ISM4.NE.0)THEN
C  H ---> NUP NUP
       IF(AMH.LE.2*AMNUP) THEN
        HNUPNUP = 0
       ELSE
        HNUPNUP=HFF(AMH,(AMNUP/AMH)**2) * CPNUP**2
     .         *HFFSELF(AMH)
       ENDIF
C  H ---> EP EP
       IF(AMH.LE.2*AMEP) THEN
        HEPEP = 0
       ELSE
        HEPEP=HFF(AMH,(AMEP/AMH)**2) * CPEP**2
     .       *HFFSELF(AMH)
       ENDIF
C  H --> BP BP :
       IF(AMH.LE.2*AMBP) THEN
        HBPBP = 0
       ELSE
        HBP2=3.D0*HFF(AMH,(RMBP/AMH)**2) * CPBP**2
     .     *QCDH(RMBP**2/AMH**2)
     .     *HFFSELF(AMH)
        IF(HBP2.LT.0.D0) HBP2 = 0
        HBP1=3.D0*HFF(AMH,(AMBP/AMH)**2) * CPBP**2
     .     *TQCDH(AMBP**2/AMH**2)
     .     *HFFSELF(AMH)
        RAT = 2*AMBP/AMH
        HBPBP = QQINT_HDEC(RAT,HBP1,HBP2)
       ENDIF
C  H --> TP TP :
       IF(AMH.LE.2*AMTP) THEN
        HTPTP = 0
       ELSE
        HTP2=3.D0*HFF(AMH,(RMTP/AMH)**2) * CPTP**2
     .     *QCDH(RMTP**2/AMH**2)
     .     *HFFSELF(AMH)
        IF(HTP2.LT.0.D0) HTP2 = 0
        HTP1=3.D0*HFF(AMH,(AMTP/AMH)**2) * CPTP**2
     .     *TQCDH(AMTP**2/AMH**2)
     .     *HFFSELF(AMH)
        RAT = 2*AMTP/AMH
        HTPTP = QQINT_HDEC(RAT,HTP1,HTP2)
       ENDIF
      ENDIF
C    ==========  TOTAL WIDTH AND BRANCHING RATIOS 
C
      WTOT=HLL+HMM+HSS+HCC+HBB+HTT+HGG+HGA+HZGA+HWW+HZZ
     .    +HNUPNUP+HEPEP+HBPBP+HTPTP
c    .    +HEE
c     write(6,*)'BR(H -> ee) = ',AMH,HEE/WTOT
c     write(6,*)HLL,HMM,HSS,HCC,HBB,HTT,HGG,HGA,HZGA,HWW,HZZ
c     write(6,*)HBB,HWW,HZZ
      IF(WTOT.NE.0.D0)THEN
       SMBRT=HTT/WTOT
       SMBRB=HBB/WTOT
       SMBRL=HLL/WTOT
       SMBRM=HMM/WTOT
       SMBRC=HCC/WTOT
       SMBRS=HSS/WTOT
       SMBRG=HGG/WTOT
       SMBRGA=HGA/WTOT
       SMBRZGA=HZGA/WTOT
       SMBRW=HWW/WTOT
       SMBRZ=HZZ/WTOT
       SMBRNUP=HNUPNUP/WTOT
       SMBREP=HEPEP/WTOT
       SMBRBP=HBPBP/WTOT
       SMBRTP=HTPTP/WTOT
      ELSE
       SMBRT=0
       SMBRB=0
       SMBRL=0
       SMBRM=0
       SMBRC=0
       SMBRS=0
       SMBRG=0
       SMBRGA=0
       SMBRZGA=0
       SMBRW=0
       SMBRZ=0
       SMBRNUP=0
       SMBREP=0
       SMBRBP=0
       SMBRTP=0
      ENDIF
      SMWDTH=WTOT

c     write(6,*)HLL,HMM
c     write(6,*)HSS,HCC,HBB,HTT
c     write(6,*)HGG,HGA,HZGA,HWW,HZZ
c     write(6,*)HNUPNUP,HEPEP,HBPBP,HTPTP
c     write(6,*)WTOT
c     write(6,*)SMBRT+SMBRB+SMBRL+SMBRM+SMBRC+SMBRS+SMBRG+SMBRGA
c    .         +SMBRZGA+SMBRW+SMBRZ+SMBRNUP+SMBREP+SMBRBP+SMBRTP

      AMH=AMXX

      endif

      IF(IHIGGS.GT.0)THEN

C +++++++++++++++++++++++  SUSY HIGGSSES +++++++++++++++++++++++
C
      CALL GAUGINO_HDEC(AMU,AM2,B,A,AMCHAR,AMNEUT,XMNEUT,AC1,AC2,AC3,
     .             AN1,AN2,AN3,ACNL,ACNR,AGDL,AGDA,AGDH,AGDC)
C
      TSC = (AMSQ+AMUR+AMDR)/3
      BSC = (AMSQ+AMUR+AMDR)/3
      CALL SFERMION_HDEC(TSC,BSC,AMSQ,AMUR,AMDR,AMEL,AMER,AL,AU,AD,AMU,
     .               AMST,AMSB,AMSL,AMSU,AMSD,AMSE,AMSN,AMSN1,
     .               GLEE,GLTT,GLBB,GHEE,GHTT,GHBB,
     .               GAEE,GATT,GABB,GCEN,GCTB)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     QSUSY = 1.D0/3
c     QSUSY = 1
c     QSUSY = 3
c     LOOP = 1
c     LOOP = 2
c     write(6,*)'Loop, Factor = ?'
c     read(5,*)LOOP,QSUSY
c     QSUSY = DMIN1(AMSB(1),AMSB(2),AMGLU)*QSUSY
c     QSUSY = (AMSB(1)+AMSB(2)+AMGLU)/3*QSUSY
c     QSUSY = +0.8204315362167340D3
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     QSUSY = QSUSY1
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      QSUSY = 1
      LOOP = 2
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      QSUSY1 = QSUSY
      QSUSY2 = QSUSY
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     FACTOR = 1
c     write(6,*)'Factor?'
c     read(5,*)FACTOR
c     QSQ = FACTOR*AMST(1)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
C

c     write(6,*)'M_A, M_h, M_H, sin(alpha): ',AMA,AML,AMH,DSIN(A)

      ENDIF

c     write(6,*)'glt, glb = ',glt,glb
c     write(6,*)'ght, ghb = ',ght,ghb
c     write(6,*)'gat, gab = ',gat,gab

      IF(IHIGGS.EQ.1.OR.IHIGGS.EQ.5)THEN
C        =========================================================
C                           LIGHT CP EVEN HIGGS DECAYS
C        =========================================================
C     =============  RUNNING MASSES 
      RMS = RUNM_HDEC(AML,3)
      RMC = RUNM_HDEC(AML,4)
      RMB = RUNM_HDEC(AML,5)
      RMT = RUNM_HDEC(AML,6)
      RATCOUP = GLT/GLB
      HIGTOP = AML**2/AMT**2

      ASH=ALPHAS_HDEC(AML,3)
      AMC0=1.D8
      AMB0=2.D8
C     AMT0=3.D8
      AS3=ALPHAS_HDEC(AML,3)
      AMC0=AMC
      AS4=ALPHAS_HDEC(AML,3)
      AMB0=AMB
C     AMT0=AMT

C     =============== PARTIAL WIDTHS 
C  H ---> G G
       EPS=1.D-8
       NFEXT = 3
       ASG = AS3
       CTT = 4*AMT**2/AML**2*DCMPLX(1D0,-EPS)
       CTB = 4*AMB**2/AML**2*DCMPLX(1D0,-EPS)
       CAT = 2*CTT*(1+(1-CTT)*CF(CTT))*GLT
       CAB = 2*CTB*(1+(1-CTB)*CF(CTB))*GLB
       CTC = 4*AMC**2/AML**2*DCMPLX(1D0,-EPS)
       CAC = 2*CTC*(1+(1-CTC)*CF(CTC))*GLT
C
   
c this is for a check    
       bb = datan(tgbet2hdm)
c       print*,'1,glt/b',dcos(alph2hdm)/dsin(bb),glt,glb

c       print*,'2,glt',dcos(alph2hdm)/dsin(bb),glt
c       print*,'2,glb',-sin(alph2hdm)/dcos(bb),glb

c       print*,'v1',amt,amb,amc,cat,cab,cac
c end check

       IF(IOFSUSY.EQ.0) THEN 
       CSB1= 4*AMSB(1)**2/AML**2*DCMPLX(1D0,-EPS)
       CSB2= 4*AMSB(2)**2/AML**2*DCMPLX(1D0,-EPS)
       CST1= 4*AMST(1)**2/AML**2*DCMPLX(1D0,-EPS)
       CST2= 4*AMST(2)**2/AML**2*DCMPLX(1D0,-EPS)
       CXB1=-AMZ**2/AMSB(1)**2*CSB1*(1-CSB1*CF(CSB1))*GLBB(1,1)
       CXB2=-AMZ**2/AMSB(2)**2*CSB2*(1-CSB2*CF(CSB2))*GLBB(2,2)
       CXT1=-AMZ**2/AMST(1)**2*CST1*(1-CST1*CF(CST1))*GLTT(1,1)
       CXT2=-AMZ**2/AMST(2)**2*CST2*(1-CST2*CF(CST2))*GLTT(2,2)

       CSUL = 4*AMSU(1)**2/AML**2*DCMPLX(1D0,-EPS)
       CSUR = 4*AMSU(2)**2/AML**2*DCMPLX(1D0,-EPS)
       CSDL = 4*AMSD(1)**2/AML**2*DCMPLX(1D0,-EPS)
       CSDR = 4*AMSD(2)**2/AML**2*DCMPLX(1D0,-EPS)
       CXUL=2*(1.D0/2.D0-2.D0/3.D0*SS)*AMZ**2/AMSU(1)**2*DSIN(A+B)
     .      *CSUL*(1-CSUL*CF(CSUL))
       CXUR=2*(2.D0/3.D0*SS)*AMZ**2/AMSU(2)**2*DSIN(A+B)
     .      *CSUR*(1-CSUR*CF(CSUR))
       CXDL=2*(-1.D0/2.D0+1.D0/3.D0*SS)*AMZ**2/AMSD(1)**2*DSIN(A+B)
     .      *CSDL*(1-CSDL*CF(CSDL))
       CXDR=2*(-1.D0/3.D0*SS)*AMZ**2/AMSD(2)**2*DSIN(A+B)
     .      *CSDR*(1-CSDR*CF(CSDR))

       ELSE
       CXB1=0.D0 
       CXB2=0.D0 
       CXT1=0.D0 
       CXT2=0.D0 

       CXUL=0.D0 
       CXUR=0.D0 
       CXDL=0.D0 
       CXDR=0.D0 
       ENDIF

       FQCD=HGGQCD(ASG,NFEXT)
       SQCD=SGGQCD(ASG)
       XFAC = CDABS(CAT+CAB+CAC+CXB1+CXB2+CXT1+CXT2
     .             +CXUL+CXUR+CXDL+CXDR)**2*FQCD
     .      + DREAL(DCONJG(CAT+CAB+CAC+CXB1+CXB2+CXT1+CXT2
     .                    +CXUL+CXUR+CXDL+CXDR)
     .             *(CXB1+CXB2+CXT1+CXT2+CXUL+CXUR+CXDL+CXDR))*SQCD

c this is for a check
c       print*,'v2',xfac,CDABS(CAT+CAB+CAC)**2*FQCD
c Maggie question // mass dependent NLO QCD corrections?
c end check

       HGG=HVV(AML,0.D0)*(ASG/PI)**2*XFAC/8
c      write(6,*)'gg: ',CAT,CAB,CAC,CXB1+CXB2,CXT1+CXT2,
c    .                  CXUL+CXUR+CXDL+CXDR

c      write(6,*)'amhl, glb, glt: ',aml,glb,glt

c      print*,''
c      print*,'h decay widths'
c      print*,'hgg_NLO',hgg

C  H ---> G G* ---> G CC   TO BE ADDED TO H ---> CC
       NFEXT = 4
       ASG = AS4
       FQCD=HGGQCD(ASG,NFEXT)
       SQCD=SGGQCD(ASG)
       XFAC = CDABS(CAT+CAB+CAC+CXB1+CXB2+CXT1+CXT2
     .             +CXUL+CXUR+CXDL+CXDR)**2*FQCD
     .      + DREAL(DCONJG(CAT+CAB+CAC+CXB1+CXB2+CXT1+CXT2
     .                    +CXUL+CXUR+CXDL+CXDR)
     .             *(CXB1+CXB2+CXT1+CXT2+CXUL+CXUR+CXDL+CXDR))*SQCD
       DCC=HVV(AML,0.D0)*(ASG/PI)**2*XFAC/8 - HGG

C  H ---> G G* ---> G BB   TO BE ADDED TO H ---> BB
       NFEXT = 5
       ASG = ASH
       FQCD=HGGQCD(ASG,NFEXT)
       SQCD=SGGQCD(ASG)
       XFAC = CDABS(CAT+CAB+CAC+CXB1+CXB2+CXT1+CXT2
     .             +CXUL+CXUR+CXDL+CXDR)**2*FQCD
     .      + DREAL(DCONJG(CAT+CAB+CAC+CXB1+CXB2+CXT1+CXT2
     .                    +CXUL+CXUR+CXDL+CXDR)
     .             *(CXB1+CXB2+CXT1+CXT2+CXUL+CXUR+CXDL+CXDR))*SQCD
       DBB=HVV(AML,0.D0)*(ASG/PI)**2*XFAC/8 - HGG - DCC
       HGG=HVV(AML,0.D0)*(ASG/PI)**2*XFAC/8

C  H ---> G G: FULL NNNLO CORRECTIONS TO TOP LOOPS FOR NF=5
       FQCD0=HGGQCD(ASG,5)
       FQCD=HGGQCD2(ASG,5,AML,AMT)
       XFAC = CDABS(CAT+CAB)**2*(FQCD-FQCD0)
       HGG=HGG+HVV(AML,0.D0)*(ASG/PI)**2*XFAC/8

      IF(NFGG.EQ.3)THEN
       HGG = HGG - DBB - DCC
      ELSEIF(NFGG.EQ.4)THEN
       HGG = HGG - DBB
       DCC = 0
      ELSE
       DCC = 0
       DBB = 0
      ENDIF

c     print*,'hgg_NNLO',hgg

C  H ---> MU MU
      XGLM = GLB
      XGHM = GHB
      XGAM = GAB
      if(i2hdm.eq.1) then
         xglm = gllep
      endif
      IF(IOFSUSY.EQ.0) THEN
       CALL STAUSUSY_HDEC(GLB,GHB,GAB,XGLM,XGHM,XGAM,QSUSY,0)
      ENDIF
      IF(AML.LE.2*AMMUON) THEN
       HMM = 0
      ELSE
      HMM=HFF(AML,(AMMUON/AML)**2)*XGLM**2
      ENDIF

c      print*,'h -> mumu',hmm
C  H ---> TAU TAU
      XGLT = GLB
      XGHT = GHB
      XGAT = GAB
      if(i2hdm.eq.1) then
         xglt = gllep
      endif
      IF(IOFSUSY.EQ.0) THEN
       CALL STAUSUSY_HDEC(GLB,GHB,GAB,XGLT,XGHT,XGAT,QSUSY,1)
      ENDIF
      IF(AML.LE.2*AMTAU) THEN
       HLL = 0
      ELSE
      HLL=HFF(AML,(AMTAU/AML)**2)*XGLT**2
      ENDIF

c     write(6,*)'h: tau/mu: ',HLL/HMM*AMMUON**2/AMTAU**2,XGLT**2/XGLM**2
c      print*,'h -> tautau',hll
C  H --> SS

      XGLS = GLB
      XGHS = GHB
      XGAS = GAB
      IF(IOFSUSY.EQ.0) THEN
       CALL STRSUSY_HDEC(GLB,GHB,GAB,XGLS,XGHS,XGAS,QSUSY,LOOP)
      ENDIF
      IF(AML.LE.2*AMS) THEN
       HSS = 0
      ELSE
       HS1=3.D0*HFF(AML,(AMS/AML)**2)
     .    *XGLS**2
     .    *TQCDH(AMS**2/AML**2)
       HS2=3.D0*HFF(AML,(RMS/AML)**2)*XGLS**2
     .    *QCDH(RMS**2/AML**2)
       IF(HS2.LT.0.D0) HS2 = 0
       RAT = 2*AMS/AML
       HSS = QQINT_HDEC(RAT,HS1,HS2)
      ENDIF

c      print*,'h -> ss',hss
C  H --> CC

      RATCOUP = 1
      IF(AML.LE.2*AMC) THEN
       HCC = 0
      ELSE
       HC1=3.D0*HFF(AML,(AMC/AML)**2)
     .    *GLT**2
     .    *TQCDH(AMC**2/AML**2)
       HC2=3.D0*HFF(AML,(RMC/AML)**2)*GLT**2
     .    *QCDH(RMC**2/AML**2)
     .   + DCC
       IF(HC2.LT.0.D0) HC2 = 0
       RAT = 2*AMC/AML
       HCC = QQINT_HDEC(RAT,HC1,HC2)
      ENDIF

c      print*,'h -> cc',hcc
C  H --> BB :

      XGLB = GLB
      XGHB = GHB
      XGAB = GAB
      QQ = AMB
      SUSY = 0
      XGLB = GLB
      SSUSY = (AMSB(1)+AMSB(2)+AMGLU)/3*QSUSY
      AS0 = ALPHAS_HDEC(SSUSY,3)
      IF(IOFSUSY.EQ.0) THEN
       I0 = 1
       CALL DMBAPP_HDEC(I0,DGLB,DGHB,DGAB,QSUSY,LOOP)
       I0 = 1
       BSC = (AMSQ+AMUR+AMDR)/3
c      XMB = AMB
       DELB0 = -DGAB/(1+1/TGBET**2)
c      write(6,*)'Delta_b = ',DELB0
       XMB = RUNM_HDEC(SUSYSCALE,5)/(1+DELB0)
       SUSY = COFSUSY_HDEC(I0,AMB,XMB,QQ)*AS0/PI - 2*DGLB
c      SUSY = SUSY*ASH/AS0
       CALL BOTSUSY_HDEC(GLB,GHB,GAB,XGLB,XGHB,XGAB,QSUSY,LOOP)
      ENDIF
      RATCOUP = GLT/XGLB
      IF(AML.LE.2*AMB) THEN
       HBB = 0
      ELSE
       HB1=3.D0*HFF(AML,(AMB/AML)**2)
     .    *(XGLB**2+XGLB*GLB*SUSY)
     .    *TQCDH(AMB**2/AML**2)
       HB2=3.D0*HFF(AML,(RMB/AML)**2)
     .    *(XGLB**2+XGLB*GLB*SUSY)
     .    *QCDH(RMB**2/AML**2)
     .   + DBB
       IF(HB2.LT.0.D0) HB2 = 0
       RAT = 2*AMB/AML
       HBB = QQINT_HDEC(RAT,HB1,HB2)

c      print*,'h -> bb',hbb,rmb
c      print*
c      print*,'h -> bb:',hbb,glb,xglb,glb*susy,2*dglb,susy,2*dglb+susy

c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c--hMSSM?
c      sb = dsin(b)
c      cb0 = dcos(b)
c      c2b = cb0**2-sb**2
c      amh0 = dsqrt(((ama**2+amz**2-aml**2)*(amz**2*cb0**2+ama**2*sb**2)
c    .       -ama**2*amz**2*c2b**2)/(amz**2*cb0**2+ama**2*sb**2-aml**2))
c      amch0 = dsqrt(ama**2+amw**2)
c      a0 = -datan((amz**2+ama**2)*sb*cb0
c    .            /(amz**2*cb0**2+ama**2*sb**2-aml**2))
c      write(6,*)'0: ',AML,AMH,AMA
c      write(6,*)'1: ',AML,AMH0,AMA
c      write(6,*)'0: ',AMCH,B,A
c      write(6,*)'1: ',AMCH0,B,A0
c      write(6,*)'   ',glb,xglb
c      write(6,*)'   ',ghb,xghb
c      write(6,*)'   ',gab,xgab
c      write(6,*)((ama**2+amz**2-aml**2)*(amz**2*cb0**2+ama**2*sb**2)
c    .       -ama**2*amz**2*c2b**2),
c    .       (amz**2*cb0**2+ama**2*sb**2-aml**2)
c      write(6,*)
c      write(6,*)
c      write(6,*)'hhh: ',glll,3*aml**2/amz**2,glll/(3*aml**2/amz**2)
c      write(6,*)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c      write(6,('A3,4(1X,G15.8)'))'h: ',AMA,AML,SUSY+2*DGLB,
c    .                             SUSY/(SUSY+2*DGLB)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c      HB2=3.D0*HFF(AML,(RMB/AML)**2)
c    .    *QCDH(RMB**2/AML**2)
c    .    *(1+ELW0(AML,RMB,-1.D0/3.D0,1.D0))
c    .    *HFFSELF(AML)
c    .    + DBB
c      IF(HB2.LT.0.D0) HB2 = 0
c      HB1=3.D0*HFF(AML,(AMB/AML)**2)
c    .    *TQCDH(AMB**2/AML**2)
c    .    *HFFSELF(AML)
c      RAT = 2*AMB/AML
c      HBB0 = QQINT_HDEC(RAT,HB1,HB2)
c      write(6,*)AML,XGLB,GLB,XGLB/GLB-1,SUSY,HBB/HBB0
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c      write(6,*)AML,HB1,HB2,HBB,GLB,XGLB,SUSY+2*DGLB,2*DGLB
c    .          ,XGLB**2+XGLB*GLB*SUSY
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c      XB0=3.D0*HFF(AML,(AMB/AML)**2)
c    .    *GLB**2
c      XB1=3.D0*HFF(AML,(RMB/AML)**2)
c    .    *GLB**2
c    .    *QCDH(RMB**2/AML**2)
c    .   + DBB
c      XB2=3.D0*HFF(AML,(RMB/AML)**2)
c    .    *(XGLB**2+XGLB*GLB*SUSY)
c    .    *QCDH(RMB**2/AML**2)
c    .   + DBB
c      write(51,('5(1X,G15.8)'))AMA,AML,XB0,XB1,XB2
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c      write(51,('4(1X,G15.8)'))AMA,AML,SUSY+2*DGLB,SUSY/(SUSY+2*DGLB)
c      write(51,('4(1X,G15.8)'))AMA,AML,HBB,2*DGLB,XGLB,SUSY-1+2*DLGB,
c    .                          DSIN(A),DCOS(A)
c      write(51,*)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

c      X1 = (QCDH(RMB**2/AML**2)*HFF(AML,(RMB/AML)**2)/
c    .       HFF(AML,(AMB/AML)**2)-1)
c      X2 = (SUSY-1)

c     RATCOUP = GLT/XGLB
c      HB1X=3.D0*HFF(AML,(AMB/AML)**2)
c    .    *XGLB**2
c    .    *TQCDH(AMB**2/AML**2)
c    .    /(BETA_HDEC(AMB**2/AML**2))**3
c    .    *SUSY
c      HB2X=3.D0*HFF(AML,(RMB/AML)**2)*XGLB**2
c    .    *QCDH(RMB**2/AML**2)
c    .    /(BETA_HDEC(RMB**2/AML**2))**3
c    .    *SUSY
c      HB1X=3.D0*HFF(AML,(RMB/AML)**2)*GLB**2
c    .    *QCDH(RMB**2/AML**2)
c    .    /(BETA_HDEC(RMB**2/AML**2))**3
c    .    *(SUSY+2*DGLB)

c     RATCOUP = 0
c     deltaqcd = QCDH(RMB**2/AML**2)
c     RATCOUP = GLT/XGLB
c     deltat = QCDH(RMB**2/AML**2) - deltaqcd

c      write(6,*)
c      write(6,*)'h:'
c      write(6,*)'MB,RUNMB,alpha_s: ',AMB,RMB,ASH
c      write(6,*)'Mh =              ',AML
c      write(6,*)'MA =              ',AMA
c      write(6,*)'Delta(mb) = ',-DGAB
c      write(6,*)'QCD           SUSY          APPROX',
c    .           '        APPROX/FULL  Gbh(QCD)    Gbh(SQCD):'
c      write(6,*)X1,X2+2*DGLB,2*DGLB,2*DGLB/(X2+2*DGLB),GLB,XGLB
c      write(6,*)'Resummation: ',(XGLB/GLB)**2-1
c      write(6,*)'Rest:        ',SUSY-1
c      write(6,*)'Rest:        ',SUSY-1,dtan(a),tgbet
c      write(6,*)AMSQ,AMUR,AMDR,(SUSY-1)/(X2+2*DGLB)
c      write(6,*)'Total SUSY:  ',(XGLB/GLB)**2*SUSY-1
c      write(6,*)'deltaqcd,t = ',deltaqcd,deltat
c      write(6,*)'Gamma(0)   = ',AMA,HB2X,HB1X
c      write(6,*)'Gamma(mb)  = ',HB2,HB1
c      write(6,*)
c      write(9,*)AMA,AML,HB2X,HB2X/SUSY,GLB,XGLB
c      write(6,*)'Rest: h      ',AMA,AML,(SUSY-1)/(X2+2*DGLB)
c      write(51,*)AMA,AML,(SUSY-1)/(X2+2*DGLB)
      ENDIF
C  H ---> TT
      RATCOUP = 0
      IF (AML.LE.2*AMT) THEN
       HTT=0.D0
      ELSE
       HT1=3.D0*HFF(AML,(AMT/AML)**2)*GLT**2
     .    *TQCDH(AMT**2/AML**2)
       HT2=3.D0*HFF(AML,(RMT/AML)**2)*GLT**2
     .    *QCDH(RMT**2/AML**2)
       IF(HT2.LT.0.D0) HT2 = 0
       RAT = 2*AMT/AML
       HTT = QQINT_HDEC(RAT,HT1,HT2)
      ENDIF

c MMM changed 22/8/2013
      if(i2hdm.eq.1) then
         if(ionsh.eq.0)then
            dld=5.D0
            dlu=3.D0
            xm1 = 2d0*amt-dld
            xm2 = 2d0*amt+dlu
            if (aml.le.amt+amw+amb) then
               htt=0.d0
            elseif (aml.le.xm1) then
               factt=6.d0*gf**2*aml**3*amt**2/2.d0/128.d0/pi**3
               call HTOTT_hdec(aml,amt,amb,amw,amch,glt,glb,gat,gab,
     .              glvv,gzah,htt0)
               htt=factt*htt0
            elseif (aml.le.xm2) then
               XX(1) = XM1-1D0
               XX(2) = XM1
               XX(3) = XM2
               XX(4) = XM2+1D0

               factt=6.d0*gf**2*xx(1)**3*amt**2/2.d0/128.d0/pi**3
               call HTOTT_hdec(xx(1),amt,amb,amw,amch,glt,glb,gat,gab,
     .              glvv,gzah,htt0)
               yy(1)=factt*htt0

               factt=6.d0*gf**2*xx(2)**3*amt**2/2.d0/128.d0/pi**3
               call HTOTT_hdec(xx(2),amt,amb,amw,amch,glt,glb,gat,gab,
     .              glvv,gzah,htt0)
               yy(2)=factt*htt0

               xmt = runm_hdec(xx(3),6)
               ht1=3.d0*hff(xx(3),(amt/xx(3))**2)*glt**2
     .              *tqcdh(amt**2/xx(3)**2)
               ht2=3.d0*hff(xx(3),(xmt/xx(3))**2)*glt**2
     .              *qcdh(xmt**2/xx(3)**2)
               if(ht2.lt.0.d0) ht2 = 0
               rat = 2*amt/xx(3)
               yy(3) = qqint_hdec(rat,ht1,ht2)

               xmt = runm_hdec(xx(4),6)
               ht1=3.d0*hff(xx(4),(amt/xx(4))**2)*glt**2
     .              *tqcdh(amt**2/xx(4)**2)
               ht2=3.d0*hff(xx(4),(xmt/xx(4))**2)*glt**2
     .              *qcdh(xmt**2/xx(4)**2)
               if(ht2.lt.0.d0) ht2 = 0
               rat = 2*amt/xx(4)
               yy(4) = qqint_hdec(rat,ht1,ht2)

               htt=fint_hdec(aml,xx,yy)
            else
               ht1=3.d0*hff(aml,(amt/aml)**2)*glt**2
     .              *tqcdh(amt**2/aml**2)
               ht2=3.d0*hff(aml,(rmt/aml)**2)*glt**2
     .              *qcdh(rmt**2/aml**2)
               if(ht2.lt.0.d0) ht2 = 0
               rat = 2.D0*amt/aml
               htt = qqint_hdec(rat,ht1,ht2)
            endif
         else
            if (aml.le.2.d0*amt) then
               htt=0.d0
            else
               ht1=3.d0*hff(aml,(amt/aml)**2)*glt**2
     .              *tqcdh(amt**2/aml**2)
               ht2=3.d0*hff(aml,(rmt/aml)**2)*glt**2
     .              *qcdh(rmt**2/aml**2)
               if(ht2.lt.0.d0) ht2 = 0
               rat = 2.D0*amt/aml
               htt = qqint_hdec(rat,ht1,ht2)
            endif
         endif
      endif
c end MMM changed 22/8/2013

c      print*,'h -> tt',htt
C  H ---> GAMMA GAMMA
       EPS=1.D-8
       XRMC = RUNM_HDEC(AML/2,4)*AMC/RUNM_HDEC(AMC,4)
       XRMB = RUNM_HDEC(AML/2,5)*AMB/RUNM_HDEC(AMB,5)
       XRMT = RUNM_HDEC(AML/2,6)*AMT/RUNM_HDEC(AMT,6)

       CTT = 4*XRMT**2/AML**2*DCMPLX(1D0,-EPS)
       CTB = 4*XRMB**2/AML**2*DCMPLX(1D0,-EPS)
       CTC = 4*XRMC**2/AML**2*DCMPLX(1D0,-EPS)
       CTL = 4*AMTAU**2/AML**2*DCMPLX(1D0,-EPS)
       CTW = 4*AMW**2/AML**2*DCMPLX(1D0,-EPS)
       CTH = 4*AMCH**2/AML**2*DCMPLX(1D0,-EPS)
       CAT = 4/3D0 * 2*CTT*(1+(1-CTT)*CF(CTT))*GLT
     .     * CFACQ_HDEC(0,AML,XRMT)

c       print*,'cat_LO',4/3D0 * 2*CTT*(1+(1-CTT)*CF(CTT))*GLT
c       print*,'cat_NLO',cat
c       print*,'h-top-top coupling',glt
c       print*,'4*rmt**2/aml**2',ctt
c       print*,'rmt',rmt
c       print*,'aml',aml

       CAB = 1/3D0 * 2*CTB*(1+(1-CTB)*CF(CTB))*GLB
     .     * CFACQ_HDEC(0,AML,XRMB)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c      CALL BOTSUSY_HDEC(GLB,GHB,GAB,XGLB,XGHB,XGAB,QSUSY,LOOP)
c      CAB = 1/3D0 * 2*CTB*(1+(1-CTB)*CF(CTB))*XGLB
c    .     * CFACQ_HDEC(0,AML,XRMB)
c      write(6,*)CTB,XGLB,CFACQ_HDEC(0,AML,XRMB)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
       CAC = 4/3D0 * 2*CTC*(1+(1-CTC)*CF(CTC))*GLT
     .     * CFACQ_HDEC(0,AML,XRMC)
       CAL = 1.D0  * 2*CTL*(1+(1-CTL)*CF(CTL))*GLB
       if(i2hdm.eq.1) then
          CAL = 1.D0  * 2*CTL*(1+(1-CTL)*CF(CTL))*gllep
       endif
       CAW = -(2+3*CTW+3*CTW*(2-CTW)*CF(CTW))*GLVV
       CAH = -AMZ**2/2/AMCH**2*CTH*(1-CTH*CF(CTH))*GLPM

c       print*,'amz,amch,ghlh+h-',amz,amch,glpm,cth
c       print*,'fac1',-AMZ**2/2/AMCH**2*glpm
c       print*,'fac2',CTH*(1-CTH*CF(CTH))
c       write(6,*)'CAH,AMZ,AMCH,CTH,CF(CTH),GLPM: ',
c    .             CAH,AMZ,AMCH,CTH,CF(CTH),GLPM
       IF(IOFSUSY.EQ.0) THEN 
        RMSU1 = RUNMS_HDEC(AML/2,AMSU(1))
        RMSU2 = RUNMS_HDEC(AML/2,AMSU(2))
        RMSD1 = RUNMS_HDEC(AML/2,AMSD(1))
        RMSD2 = RUNMS_HDEC(AML/2,AMSD(2))
        RMSB1 = RUNMS_HDEC(AML/2,AMSB(1))
        RMSB2 = RUNMS_HDEC(AML/2,AMSB(2))
        RMST1 = RUNMS_HDEC(AML/2,AMST(1))
        RMST2 = RUNMS_HDEC(AML/2,AMST(2))
        CX1 = 4*AMCHAR(1)**2/AML**2*DCMPLX(1D0,-EPS)
        CX2 = 4*AMCHAR(2)**2/AML**2*DCMPLX(1D0,-EPS)
        CSB1= 4*RMSB1**2/AML**2*DCMPLX(1D0,-EPS)
        CSB2= 4*RMSB2**2/AML**2*DCMPLX(1D0,-EPS)
        CST1= 4*RMST1**2/AML**2*DCMPLX(1D0,-EPS)
        CST2= 4*RMST2**2/AML**2*DCMPLX(1D0,-EPS)
        CSL1= 4*AMSL(1)**2/AML**2*DCMPLX(1D0,-EPS)
        CSL2= 4*AMSL(2)**2/AML**2*DCMPLX(1D0,-EPS)
        CAX1= AMW/XMCHAR(1) * 2*CX1*(1+(1-CX1)*CF(CX1))*2*AC2(1,1) 
        CAX2= AMW/XMCHAR(2) * 2*CX2*(1+(1-CX2)*CF(CX2))*2*AC2(2,2) 

        CSEL = 4*AMSE(1)**2/AML**2*DCMPLX(1D0,-EPS)
        CSER = 4*AMSE(2)**2/AML**2*DCMPLX(1D0,-EPS)
        CSUL = 4*RMSU1**2/AML**2*DCMPLX(1D0,-EPS)
        CSUR = 4*RMSU2**2/AML**2*DCMPLX(1D0,-EPS)
        CSDL = 4*RMSD1**2/AML**2*DCMPLX(1D0,-EPS)
        CSDR = 4*RMSD2**2/AML**2*DCMPLX(1D0,-EPS)
        CXEL=2*(-1/2D0+SS)*AMZ**2/AMSE(1)**2*DSIN(A+B)
     .       *CSEL*(1-CSEL*CF(CSEL))
        CXER=-2*(SS)*AMZ**2/AMSE(2)**2*DSIN(A+B)
     .       *CSER*(1-CSER*CF(CSER))
        CXUL=2*4.D0/3.D0*(1.D0/2.D0-2.D0/3.D0*SS)
     .       *AMZ**2/AMSU(1)**2*DSIN(A+B)*CSUL*(1-CSUL*CF(CSUL))
     .      * CFACSQ_HDEC(AML,RMSU1)
        CXUR=2*4.D0/3.D0*(2.D0/3.D0*SS)
     .       *AMZ**2/AMSU(2)**2*DSIN(A+B)*CSUR*(1-CSUR*CF(CSUR))
     .      * CFACSQ_HDEC(AML,RMSU2)
        CXDL=2/3.D0*(-1.D0/2.D0+1.D0/3.D0*SS)
     .       *AMZ**2/AMSD(1)**2*DSIN(A+B)*CSDL*(1-CSDL*CF(CSDL))
     .      * CFACSQ_HDEC(AML,RMSD1)
        CXDR=2/3.D0*(-1.D0/3.D0*SS)
     .       *AMZ**2/AMSD(2)**2*DSIN(A+B)*CSDR*(1-CSDR*CF(CSDR))
     .      * CFACSQ_HDEC(AML,RMSD2)

        CXB1=-1/3D0*AMZ**2/AMSB(1)**2*CSB1*(1-CSB1*CF(CSB1))*GLBB(1,1)
     .      * CFACSQ_HDEC(AML,RMSB1)
        CXB2=-1/3D0*AMZ**2/AMSB(2)**2*CSB2*(1-CSB2*CF(CSB2))*GLBB(2,2)
     .      * CFACSQ_HDEC(AML,RMSB2)
        CXT1=-4/3D0*AMZ**2/AMST(1)**2*CST1*(1-CST1*CF(CST1))*GLTT(1,1)
     .      * CFACSQ_HDEC(AML,RMST1)
        CXT2=-4/3D0*AMZ**2/AMST(2)**2*CST2*(1-CST2*CF(CST2))*GLTT(2,2)
     .      * CFACSQ_HDEC(AML,RMST2)
        CSL1= 4*AMSL(1)**2/AML**2*DCMPLX(1D0,-EPS)
        CSL2= 4*AMSL(2)**2/AML**2*DCMPLX(1D0,-EPS)
        CXL1=      -AMZ**2/AMSL(1)**2*CSL1*(1-CSL1*CF(CSL1))*GLEE(1,1)
        CXL2=      -AMZ**2/AMSL(2)**2*CSL2*(1-CSL2*CF(CSL2))*GLEE(2,2)
        XFAC = CDABS(CAT+CAB+CAC+CAL+CAW+CAH+CAX1+CAX2
     .      +  CXEL+CXER+CXUL+CXUR+CXDL+CXDR
     .      +  CXB1+CXB2+CXT1+CXT2+CXL1+CXL2)**2
       ELSE 
        XFAC = CDABS(CAT+CAB+CAC+CAL+CAW+CAH)**2
       ENDIF
       HGA=HVV(AML,0.D0)*(ALPH/PI)**2/16.D0*XFAC

c       print*,'h -> gamgam',hga,aml,amch
c       print*,'cat,cab',cat,cab
c       print*,'cac,cal',cac,cal
c       print*,'caw,cah',caw,cah
c       print*,'charged Higgs loop',cah
c       print*,'charged Higgs loop',cth,glpm,cf(cth)
c      CAH = -AMZ**2/2/AMCH**2*CTH*(1-CTH*CF(CTH))*GLPM
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
       XFACQ = CDABS(CAT+CAB+CAC+CAL+CAW+CAH)**2
       XFACS = CDABS(CAT+CAB+CAC+CAL+CAW+CAH+CAX1+CAX2
     .      +  CXL1+CXL2)**2
       XFACSQ = CDABS(CAT+CAB+CAC+CAL+CAW+CAH+CAX1+CAX2
     .      +  CXB1+CXB2+CXT1+CXT2+CXL1+CXL2)**2
       HGA0 = HGA*XFACSQ/XFAC
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c      CTT = 4*AMT**2/AML**2*DCMPLX(1D0,-EPS)
c      CTB = 4*AMB**2/AML**2*DCMPLX(1D0,-EPS)
c      CTC = 4*AMC**2/AML**2*DCMPLX(1D0,-EPS)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
       CAC0 = 4/3D0 * 2*CTC*(1+(1-CTC)*CF(CTC))*GLT
       CAT0 = 4/3D0 * 2*CTT*(1+(1-CTT)*CF(CTT))*GLT
       CAB0 = 1/3D0 * 2*CTB*(1+(1-CTB)*CF(CTB))*GLB
       CXB10= -1/3D0*AMZ**2/AMSB(1)**2*CSB1*(1-CSB1*CF(CSB1))*GLBB(1,1)
       CXB20= -1/3D0*AMZ**2/AMSB(2)**2*CSB2*(1-CSB2*CF(CSB2))*GLBB(2,2)
       CXT10= -4/3D0*AMZ**2/AMST(1)**2*CST1*(1-CST1*CF(CST1))*GLTT(1,1)
       CXT20= -4/3D0*AMZ**2/AMST(2)**2*CST2*(1-CST2*CF(CST2))*GLTT(2,2)
       XFACLOQ = CDABS(CAT0+CAB0+CAC0+CAL+CAW+CAH)**2
       CXUL0=2*4.D0/3.D0*(1.D0/2.D0-2.D0/3.D0*SS)
     .      *AMZ**2/AMSU(1)**2*DSIN(A+B)*CSUL*(1-CSUL*CF(CSUL))
       CXUR0=2*4.D0/3.D0*(2.D0/3.D0*SS)
     .      *AMZ**2/AMSU(2)**2*DSIN(A+B)*CSUR*(1-CSUR*CF(CSUR))
       CXDL0=2/3.D0*(-1.D0/2.D0+1.D0/3.D0*SS)
     .      *AMZ**2/AMSD(1)**2*DSIN(A+B)*CSDL*(1-CSDL*CF(CSDL))
       CXDR0=2/3.D0*(-1.D0/3.D0*SS)
     .      *AMZ**2/AMSD(2)**2*DSIN(A+B)*CSDR*(1-CSDR*CF(CSDR))
       XFACLO = CDABS(CAT0+CAB0+CAC0+CAL+CAW+CAH+CAX1+CAX2
     .      +  CXEL+CXER+CXUL0+CXUR0+CXDL0+CXDR0
     .      +  CXB10+CXB20+CXT10+CXT20+CXL1+CXL2)**2
       CSQ = 1+3*ALPHAS_HDEC(AML,3)
       XFACSQL = CDABS(CAT+CAB+CAC+CAL+CAW+CAH+CAX1+CAX2
     .      +  CXEL+CXER+(CXUL0+CXUR0+CXDL0+CXDR0
     .      +  CXB10+CXB20+CXT10+CXT20)*CSQ+CXL1+CXL2)**2
      XFACSM = CDABS(CAT/GLT+CAB/GLB+CAC/GLT+CAL/GLB+CAW/GLVV)**2
      XFAC0  = CDABS(CAT+CAB+CAC+CAL+CAW+CXL1)**2
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
C  H ---> Z GAMMA
      XRMC = RUNM_HDEC(AML/2,4)*AMC/RUNM_HDEC(AMC,4)
      XRMB = RUNM_HDEC(AML/2,5)*AMB/RUNM_HDEC(AMB,5)
      XRMT = RUNM_HDEC(AML/2,6)*AMT/RUNM_HDEC(AMT,6)
c     print*,'xrmc,xrmb,xrmt ',xrmc,xrmb,xrmt
      IF(AML.LE.AMZ)THEN
       HZGA=0
      ELSE
       TS = SS/CS
       FT = -3*2D0/3*(1-4*2D0/3*SS)/DSQRT(SS*CS)*GLT
       FB = 3*1D0/3*(-1+4*1D0/3*SS)/DSQRT(SS*CS)*GLB
       FC = -3*2D0/3*(1-4*2D0/3*SS)/DSQRT(SS*CS)*GLT
       FL = (-1+4*SS)/DSQRT(SS*CS)*GLB
       if(i2hdm.eq.1) then
          FL = (-1+4*SS)/DSQRT(SS*CS)*gllep
       endif
       EPS=1.D-8
c      CTT = 4*XRMT**2/AML**2*DCMPLX(1D0,-EPS)
c      CTB = 4*XRMB**2/AML**2*DCMPLX(1D0,-EPS)
c      CTC = 4*XRMC**2/AML**2*DCMPLX(1D0,-EPS)
       CTT = 4*AMT**2/AML**2*DCMPLX(1D0,-EPS)
       CTB = 4*AMB**2/AML**2*DCMPLX(1D0,-EPS)
       CTC = 4*AMC**2/AML**2*DCMPLX(1D0,-EPS)
       CTL = 4*AMTAU**2/AML**2*DCMPLX(1D0,-EPS)
       CTW = 4*AMW**2/AML**2*DCMPLX(1D0,-EPS)
       CTH = 4*AMCH**2/AML**2*DCMPLX(1D0,-EPS)
c      CLT = 4*XRMT**2/AMZ**2*DCMPLX(1D0,-EPS)
c      CLB = 4*XRMB**2/AMZ**2*DCMPLX(1D0,-EPS)
c      CLC = 4*XRMC**2/AMZ**2*DCMPLX(1D0,-EPS)
       CLT = 4*AMT**2/AMZ**2*DCMPLX(1D0,-EPS)
       CLB = 4*AMB**2/AMZ**2*DCMPLX(1D0,-EPS)
       CLC = 4*AMC**2/AMZ**2*DCMPLX(1D0,-EPS)
       CLE = 4*AMTAU**2/AMZ**2*DCMPLX(1D0,-EPS)
       CLW = 4*AMW**2/AMZ**2*DCMPLX(1D0,-EPS)
       CLH = 4*AMCH**2/AMZ**2*DCMPLX(1D0,-EPS)
       CAT = FT*(CI1(CTT,CLT) - CI2(CTT,CLT))
       CAB = FB*(CI1(CTB,CLB) - CI2(CTB,CLB))
       CAC = FC*(CI1(CTC,CLC) - CI2(CTC,CLC))
       CAL = FL*(CI1(CTL,CLE) - CI2(CTL,CLE))
       CAW = -1/DSQRT(TS)*(4*(3-TS)*CI2(CTW,CLW)
     .     + ((1+2/CTW)*TS - (5+2/CTW))*CI1(CTW,CLW))*GLVV
       CAH = (1-2*SS)/DSQRT(SS*CS)*AMZ**2/2/AMCH**2*CI1(CTH,CLH)*GLPM
       XFAC = CDABS(CAT+CAB+CAC+CAL+CAW+CAH)**2
       ACOUP = DSQRT(2D0)*GF*AMZ**2*SS*CS/PI**2
       HZGA = GF/(4.D0*PI*DSQRT(2.D0))*AML**3*(ALPH/PI)*ACOUP/16.D0
     .        *XFAC*(1-AMZ**2/AML**2)**3
      ENDIF

c      print*,'h -> Zgam',hzga
C  H ---> W W
      IF(IONWZ.EQ.0)THEN
       CALL HTOVV_HDEC(0,AML,AMW,GAMW,HTWW)
       HWW = 3D0/2D0*GF*AMW**4/DSQRT(2D0)/PI/AML**3*HTWW*GLVV**2
      ELSEIF(IONWZ.EQ.-1)THEN
       DLD=2D0
       DLU=2D0
       XM1 = 2D0*AMW-DLD
       XM2 = 2D0*AMW+DLU
       IF (AML.LE.XM1) THEN
        CALL HTOVV_HDEC(0,AML,AMW,GAMW,HTWW)
        HWW = 3D0/2D0*GF*AMW**4/DSQRT(2D0)/PI/AML**3*HTWW*GLVV**2
       ELSEIF (AML.LE.XM2) THEN
        XX(1) = XM1-1D0
        XX(2) = XM1
        XX(3) = XM2
        XX(4) = XM2+1D0
        CALL HTOVV_HDEC(0,XX(1),AMW,GAMW,HTWW)
        YY(1)=3D0/2D0*GF*AMW**4/DSQRT(2D0)/PI/XX(1)**3*HTWW
        CALL HTOVV_HDEC(0,XX(2),AMW,GAMW,HTWW)
        YY(2)=3D0/2D0*GF*AMW**4/DSQRT(2D0)/PI/XX(2)**3*HTWW
        YY(3)=HVV(XX(3),AMW**2/XX(3)**2)
        YY(4)=HVV(XX(4),AMW**2/XX(4)**2)
        HWW = FINT_HDEC(AML,XX,YY)*GLVV**2
       ELSE
        HWW=HVV(AML,AMW**2/AML**2)*GLVV**2
       ENDIF
      ELSE
      DLD=2D0
      DLU=2D0
      XM1 = 2D0*AMW-DLD
      XM2 = 2D0*AMW+DLU
      IF (AML.LE.AMW) THEN
       HWW=0
      ELSE IF (AML.LE.XM1) THEN
       CWW=3.D0*GF**2*AMW**4/16.D0/PI**3
       HWW=HV(AMW**2/AML**2)*CWW*AML*GLVV**2
      ELSE IF (AML.LT.XM2) THEN
       CWW=3.D0*GF**2*AMW**4/16.D0/PI**3
       XX(1) = XM1-1D0
       XX(2) = XM1
       XX(3) = XM2
       XX(4) = XM2+1D0
       YY(1)=HV(AMW**2/XX(1)**2)*CWW*XX(1)
       YY(2)=HV(AMW**2/XX(2)**2)*CWW*XX(2)
       YY(3)=HVV(XX(3),AMW**2/XX(3)**2)
       YY(4)=HVV(XX(4),AMW**2/XX(4)**2)
       HWW = FINT_HDEC(AML,XX,YY)*GLVV**2
      ELSE
       HWW=HVV(AML,AMW**2/AML**2)*GLVV**2
      ENDIF
      ENDIF

c      print*,'h -> WW',hww
C  H ---> Z Z
      IF(IONWZ.EQ.0)THEN
       CALL HTOVV_HDEC(0,AML,AMZ,GAMZ,HTZZ)
       HZZ = 3D0/4D0*GF*AMZ**4/DSQRT(2D0)/PI/AML**3*HTZZ*GLVV**2
      ELSEIF(IONWZ.EQ.-1)THEN
       DLD=2D0
       DLU=2D0
       XM1 = 2D0*AMZ-DLD
       XM2 = 2D0*AMZ+DLU
       IF (AML.LE.XM1) THEN
        CALL HTOVV_HDEC(0,AML,AMZ,GAMZ,HTZZ)
        HZZ = 3D0/4D0*GF*AMZ**4/DSQRT(2D0)/PI/AML**3*HTZZ*GLVV**2
       ELSEIF (AML.LE.XM2) THEN
        XX(1) = XM1-1D0
        XX(2) = XM1
        XX(3) = XM2
        XX(4) = XM2+1D0
        CALL HTOVV_HDEC(0,XX(1),AMZ,GAMZ,HTZZ)
        YY(1)=3D0/4D0*GF*AMZ**4/DSQRT(2D0)/PI/XX(1)**3*HTZZ
        CALL HTOVV_HDEC(0,XX(2),AMZ,GAMZ,HTZZ)
        YY(2)=3D0/4D0*GF*AMZ**4/DSQRT(2D0)/PI/XX(2)**3*HTZZ
        YY(3)=HVV(XX(3),AMZ**2/XX(3)**2)/2
        YY(4)=HVV(XX(4),AMZ**2/XX(4)**2)/2
        HZZ = FINT_HDEC(AML,XX,YY)*GLVV**2
       ELSE
        HZZ=HVV(AML,AMZ**2/AML**2)/2.D0*GLVV**2
       ENDIF
      ELSE
      DLD=2D0
      DLU=2D0
      XM1 = 2D0*AMZ-DLD
      XM2 = 2D0*AMZ+DLU
      IF (AML.LE.AMZ) THEN
       HZZ=0
      ELSE IF (AML.LE.XM1) THEN
       CZZ=3.D0*GF**2*AMZ**4/192.D0/PI**3*(7-40/3.D0*SS+160/9.D0*SS**2)
       HZZ=HV(AMZ**2/AML**2)*CZZ*AML*GLVV**2
      ELSE IF (AML.LT.XM2) THEN
       CZZ=3.D0*GF**2*AMZ**4/192.D0/PI**3*(7-40/3.D0*SS+160/9.D0*SS**2)
       XX(1) = XM1-1D0
       XX(2) = XM1
       XX(3) = XM2
       XX(4) = XM2+1D0
       YY(1)=HV(AMZ**2/XX(1)**2)*CZZ*XX(1)
       YY(2)=HV(AMZ**2/XX(2)**2)*CZZ*XX(2)
       YY(3)=HVV(XX(3),AMZ**2/XX(3)**2)/2D0
       YY(4)=HVV(XX(4),AMZ**2/XX(4)**2)/2D0
       HZZ = FINT_HDEC(AML,XX,YY)*GLVV**2
      ELSE
       HZZ=HVV(AML,AMZ**2/AML**2)/2.D0*GLVV**2
      ENDIF
      ENDIF

c      print*,'h -> ZZ',hzz
C  H ---> A A
      IF (AML.LE.2.D0*AMA) THEN
      HAA=0
      ELSE
      HAA=GF/16.D0/DSQRT(2D0)/PI*AMZ**4/AML
     .   *BETA_HDEC(AMA**2/AML**2)*GLAA**2
      ENDIF

c MMM changed 22/8/2013
      if(i2hdm.eq.1) then
         if(ionsh.eq.0) then
            dld=0.1d0
            dlu=0.1d0
            xm1 = 2d0*ama-dld
            xm2 = 2d0*ama+dlu
            if (aml.le.ama) then
               haa = 0d0
            elseif (aml.le.xm1) then
               xa=ama**2/aml**2
               xa1=(xa-1.d0)*(2.d0-.5d0*dlog(xa))+(1.d0-5.d0*xa)
     .              *(datan((2.d0*xa-1.d0)/dsqrt(4.d0*xa-1.d0))
     .              -datan(1.d0/dsqrt(4.d0*xa-1.d0)))
     .              /dsqrt(4.d0*xa-1.d0)
               xa2=3*gf**2/32.d0/pi**3*amz**4/aml*glaa**2*gab**2*amb**2
               haa=xa1*xa2
            elseif (aml.le.xm2) then
               xx(1) = xm1-1d0
               xx(2) = xm1
               xx(3) = xm2
               xx(4) = xm2+1d0
               xa=ama**2/xx(1)**2
               xa1=(xa-1.d0)*(2.d0-.5d0*dlog(xa))+(1.d0-5.d0*xa)
     .              *(datan((2.d0*xa-1.d0)/dsqrt(4.d0*xa-1.d0))
     .              -datan(1.d0/dsqrt(4.d0*xa-1.d0)))
     .              /dsqrt(4.d0*xa-1.d0)
               xa2=3*gf**2/32.d0/pi**3*amz**4/xx(1)*gab**2*amb**2
               yy(1)=xa1*xa2
               xa=ama**2/xx(2)**2
               xa1=(xa-1.d0)*(2.d0-.5d0*dlog(xa))+(1.d0-5.d0*xa)
     .              *(datan((2.d0*xa-1.d0)/dsqrt(4.d0*xa-1.d0))
     .              -datan(1.d0/dsqrt(4.d0*xa-1.d0)))
     .              /dsqrt(4.d0*xa-1.d0)
               xa2=3*gf**2/32.d0/pi**3*amz**4/xx(2)*gab**2*amb**2
               yy(2)=xa1*xa2
               yy(3)=gf/16d0/dsqrt(2d0)/pi*amz**4/xx(3)
     .              *beta_hdec(ama**2/xx(3)**2)
               yy(4)=gf/16d0/dsqrt(2d0)/pi*amz**4/xx(4)
     .              *beta_hdec(ama**2/xx(4)**2)
               haa = fint_hdec(aml,xx,yy)*glaa**2
            else
               haa=gf/16d0/dsqrt(2d0)/pi*amz**4/aml*
     .              beta_hdec(ama**2/aml**2)*glaa**2
            endif
         else
            if (aml.le.2*ama) then
               haa=0
            else
               haa=gf/16d0/dsqrt(2d0)/pi*amz**4/aml*
     .              beta_hdec(ama**2/aml**2)*glaa**2
            endif
         endif
      endif
c end MMM changed 22/8/2013

c      print*,'h -> AA',haa

C  h ---> H+ H- 

      if(i2hdm.eq.1) then
         if (aml.le.2*amch) then
            hlchch=0.D0
         else
            hlchch=gf/8d0/dsqrt(2d0)/pi*amz**4/aml*
     .           beta_hdec(amch**2/aml**2)*glpm**2
         endif
      elseif(i2hdm.eq.0) then
         hlchch=0.D0
      endif

c     print*,'h -> H+H-',hlchch

C  H ---> A Z
      IF (AML.LE.AMZ+AMA) THEN
      HAZ=0
      ELSE
      CAZ=LAMB_HDEC(AMA**2/AML**2,AMZ**2/AML**2)
     .   *LAMB_HDEC(AML**2/AMZ**2,AMA**2/AMZ**2)**2
      HAZ=GF/8.D0/DSQRT(2D0)/PI*AMZ**4/AML*CAZ*GZAL**2
      ENDIF

c MMM changed 22/8/2013
      if(i2hdm.eq.1) then
         if(ionsh.eq.0) then
            dld=1d0
            dlu=8d0
            xm1 = ama+amz-dld
            xm2 = ama+amz+dlu
            if (aml.lt.ama) then
               haz=0
            elseif (aml.lt.xm1) then
               if(aml.le.dabs(amz-ama))then
                  haz=0
               else
                  haz=9.d0*gf**2/16.d0/pi**3*amz**4*aml*gzal**2*
     .                 (7.d0/12.d0-10.d0/9.d0*ss+40.d0/27.d0*ss**2)
     .                 *hvh((ama/aml)**2,(amz/aml)**2)
               endif
            elseif (aml.lt.xm2) then
               xx(1) = xm1-1d0
               xx(2) = xm1
               xx(3) = xm2
               xx(4) = xm2+1d0
               yy(1)=9.d0*gf**2/16.d0/pi**3*amz**4*xx(1)*
     .              (7.d0/12.d0-10.d0/9.d0*ss+40.d0/27.d0*ss**2)
     .              *hvh((ama/xx(1))**2,(amz/xx(1))**2)
               yy(2)=9.d0*gf**2/16.d0/pi**3*amz**4*xx(2)*
     .              (7.d0/12.d0-10.d0/9.d0*ss+40.d0/27.d0*ss**2)
     .              *hvh((ama/xx(2))**2,(amz/xx(2))**2)
               caz=lamb_hdec(ama**2/xx(3)**2,amz**2/xx(3)**2)
     .              *lamb_hdec(xx(3)**2/amz**2,ama**2/amz**2)**2
               yy(3)=gf/8.d0/dsqrt(2d0)/pi*amz**4/xx(3)*caz
               caz=lamb_hdec(ama**2/xx(4)**2,amz**2/xx(4)**2)
     .              *lamb_hdec(xx(4)**2/amz**2,ama**2/amz**2)**2
               yy(4)=gf/8.d0/dsqrt(2d0)/pi*amz**4/xx(4)*caz
               haz = fint_hdec(aml,xx,yy)*gzal**2
            else
               caz=lamb_hdec(ama**2/aml**2,amz**2/aml**2)
     .              *lamb_hdec(aml**2/amz**2,ama**2/amz**2)**2
               haz=gf/8.d0/dsqrt(2d0)/pi*amz**4/aml*caz*gzal**2
            endif
         else
            if (aml.lt.amz+ama) then
               haz=0
            else
               caz=lamb_hdec(ama**2/aml**2,amz**2/aml**2)
     .              *lamb_hdec(aml**2/amz**2,ama**2/amz**2)**2
               haz=gf/8.d0/dsqrt(2d0)/pi*amz**4/aml*caz*gzal**2
            endif
         endif
      endif
c end MMM changed 22/8/2013

c     print*,'h -> AZ',haz

C  H ---> H+ W+

      IF (AML.LE.AMW+AMCH) THEN
      HHW=0
      ELSE
      CHW=LAMB_HDEC(AMCH**2/AML**2,AMW**2/AML**2)
     .   *LAMB_HDEC(AML**2/AMW**2,AMCH**2/AMW**2)**2
      HHW=2*gf/8.d0/dsqrt(2d0)/pi*amw**4/aml*chw*ghvv**2
      ENDIF

c MMM changed 22/8/2013
      if(i2hdm.eq.1) then
         if(ionsh.eq.0) then
            dld=3d0
            dlu=9d0
            xm1 = amch+amw-dld
            xm2 = amch+amw+dlu
            if (aml.lt.amch) then
               hhw=0.d0
            elseif (aml.lt.xm1) then
               if(aml.le.dabs(amw-amch))then
                  hhw=0
               else
c ---- this is for a numeric check ----
                  iprint = 10

                  ivegas(1) = 30000
                  ivegas(2) = 5
                  ivegas(3) = 120000
                  ivegas(4) = 10

                  amhi = aml
                  amhj = amch
                  amhk = amw
                  gamtoti = 1.D0
                  gamtotj = 1.D0
                  gamtotk = gamw

c ---- initialization of VEGAS ----

c                  call RSTART(12,34,56,78)

c                  call INTEG(hvhinteg,2,iprint,ivegas,result,relative)

c                  print*,'result',result
c ---- end numeric check

                  hhw=9.d0*gf**2/16.d0/pi**3*amw**4*aml*ghvv**2*2
     .                 *hvh((amch/aml)**2,(amw/aml)**2)
               endif
            elseif (aml.lt.xm2) then
               xx(1) = xm1-1d0
               xx(2) = xm1
               xx(3) = xm2
               xx(4) = xm2+1d0
               yy(1)=9.d0*gf**2/16.d0/pi**3*amw**4*xx(1)*2
     .              *hvh((amch/xx(1))**2,(amw/xx(1))**2)
               yy(2)=9.d0*gf**2/16.d0/pi**3*amw**4*xx(2)*2
     .              *hvh((amch/xx(2))**2,(amw/xx(2))**2)
               chw=lamb_hdec(amch**2/xx(3)**2,amw**2/xx(3)**2)
     .              *lamb_hdec(xx(3)**2/amw**2,amch**2/amw**2)**2
               yy(3)=2*gf/8.d0/dsqrt(2d0)/pi*amw**4/xx(3)*chw
               chw=lamb_hdec(amch**2/xx(4)**2,amw**2/xx(4)**2)
     .              *lamb_hdec(xx(4)**2/amw**2,amch**2/amw**2)**2
               yy(4)=2*gf/8.d0/dsqrt(2d0)/pi*amw**4/xx(4)*chw
               hhw=fint_hdec(aml,xx,yy)*ghvv**2
            else
               chw=lamb_hdec(amch**2/aml**2,amw**2/aml**2)
     .              *lamb_hdec(aml**2/amw**2,amch**2/amw**2)**2
               hhw=2*gf/8.d0/dsqrt(2d0)/pi*amw**4/aml*chw*ghvv**2
            endif
         else
            if (aml.lt.amw+amch) then
               hhw=0.d0
            else
               chw=lamb_hdec(amch**2/aml**2,amw**2/aml**2)
     .              *lamb_hdec(aml**2/amw**2,amch**2/amw**2)**2
               hhw=2*gf/8.d0/dsqrt(2d0)/pi*amw**4/aml*chw*ghvv**2
            endif
         endif
      endif
c end MMM changed 22/8/2013

c     print*,'h -> H+W- + H-W+',hhw,hhw/2.D0

C  ============================ SUSY DECAYS 
      IF(IOFSUSY.EQ.0) THEN
C
C  HL ----> CHARGINOS
C
      DO 711 I=1,2
      DO 711 J=1,2
      IF (AML.GT.AMCHAR(I)+AMCHAR(J)) THEN
      WHLCH(I,J)=GF*AMW**2/(2*PI*DSQRT(2.D0))/AML 
     .     *LAMB_HDEC(AMCHAR(I)**2/AML**2,AMCHAR(J)**2/AML**2)
     .     *( (AC2(I,J)**2+AC2(J,I)**2)*(AML**2-AMCHAR(I)
     .         **2-AMCHAR(J)**2)-4.D0*AC2(I,J)*AC2(J,I)* 
     .         XMCHAR(I)*XMCHAR(J) ) 
      ELSE
      WHLCH(I,J)=0.D0
      ENDIF
      WHLCHT=WHLCH(1,1)+WHLCH(1,2)+WHLCH(2,1)+WHLCH(2,2)
 711  CONTINUE
C
C  HL ----> NEUTRALINOS 
C
      DO 712 I=1,4
      DO 712 J=1,4
      IF (AML.GT.AMNEUT(I)+AMNEUT(J)) THEN
      WHLNE(I,J)=GF*AMW**2/(2*PI*DSQRT(2.D0))/AML 
     .         *AN2(I,J)**2*(AML**2-(XMNEUT(I)+XMNEUT(J))**2)
     .         *LAMB_HDEC(AMNEUT(I)**2/AML**2,AMNEUT(J)**2/AML**2)
      ELSE 
      WHLNE(I,J)=0.D0
      ENDIF
 712  CONTINUE
      WHLNET= WHLNE(1,1)+WHLNE(1,2)+WHLNE(1,3)+WHLNE(1,4)
     .       +WHLNE(2,1)+WHLNE(2,2)+WHLNE(2,3)+WHLNE(2,4)
     .       +WHLNE(3,1)+WHLNE(3,2)+WHLNE(3,3)+WHLNE(3,4)
     .       +WHLNE(4,1)+WHLNE(4,2)+WHLNE(4,3)+WHLNE(4,4)
CCC
C  HL ----> SLEPTONS 
C
      IF (AML.GT.2.D0*AMSE(1)) THEN
      WHLSLEL=2*GF/2.D0/DSQRT(2D0)/PI*AMZ**4/AML*DSIN(B+A)**2
     .      *BETA_HDEC(AMSE(1)**2/AML**2)*(-0.5D0+SS)**2
      ELSE
      WHLSLEL=0.D0
      ENDIF

      IF (AML.GT.2.D0*AMSE(2)) THEN
      WHLSLER=2*GF/2.D0/DSQRT(2D0)/PI*AMZ**4/AML*DSIN(B+A)**2
     .      *BETA_HDEC(AMSE(2)**2/AML**2)*SS**2
      ELSE
      WHLSLER=0.D0
      ENDIF

      WHLSLNL=0.D0
      IF (AML.GT.2.D0*AMSN1(1)) THEN
      WHLSLNL=2*GF/2.D0/DSQRT(2D0)/PI*AMZ**4/AML*DSIN(B+A)**2
     .      *BETA_HDEC(AMSN1(1)**2/AML**2)*0.5D0**2
      ENDIF
      IF (AML.GT.2.D0*AMSN(1)) THEN
      WHLSLNL=WHLSLNL + GF/2.D0/DSQRT(2D0)/PI*AMZ**4/AML*DSIN(B+A)**2
     .      *BETA_HDEC(AMSN(1)**2/AML**2)*0.5D0**2
      ENDIF

      DO 718 I=1,2
      DO 718 J=1,2
      IF(AML.GT.AMSL(I)+AMSL(J)) THEN
      WHLSTAU(I,J)=GF*AMZ**4/2.D0/DSQRT(2.D0)/PI*GLEE(I,J)**2*
     .      LAMB_HDEC(AMSL(I)**2/AML**2,AMSL(J)**2/AML**2)/AML
      ELSE
      WHLSTAU(I,J)=0.D0
      ENDIF
 718  CONTINUE

      WHLSLT=WHLSTAU(1,1)+WHLSTAU(2,1)+WHLSTAU(1,2)+WHLSTAU(2,2) 
     .       +WHLSLEL+WHLSLER+WHLSLNL
C
C  HL ----> SQUARKS 
C
      IF (AML.GT.2.D0*AMSU(1)) THEN
      WHLSQUL=6*GF/2.D0/DSQRT(2D0)/PI*AMZ**4/AML*DSIN(B+A)**2
     .      *BETA_HDEC(AMSU(1)**2/AML**2)*(0.5D0-2.D0/3.D0*SS)**2
      ELSE
      WHLSQUL=0.D0
      ENDIF

      IF (AML.GT.2.D0*AMSU(2)) THEN
      WHLSQUR=6*GF/2.D0/DSQRT(2D0)/PI*AMZ**4/AML*DSIN(B+A)**2
     .      *BETA_HDEC(AMSU(2)**2/AML**2)*(-2.D0/3.D0*SS)**2
      ELSE
      WHLSQUR=0.D0
      ENDIF

      IF (AML.GT.2.D0*AMSD(1)) THEN
      WHLSQDL=6*GF/2.D0/DSQRT(2D0)/PI*AMZ**4/AML*DSIN(B+A)**2
     .      *BETA_HDEC(AMSD(1)**2/AML**2)*(-0.5D0+1.D0/3.D0*SS)**2
      ELSE
      WHLSQDL=0.D0
      ENDIF

      IF (AML.GT.2.D0*AMSD(2)) THEN
      WHLSQDR=6*GF/2.D0/DSQRT(2D0)/PI*AMZ**4/AML*DSIN(B+A)**2
     .      *BETA_HDEC(AMSD(2)**2/AML**2)*(+1.D0/3.D0*SS)**2
      ELSE
      WHLSQDR=0.D0
      ENDIF

      WHLSQ=WHLSQUL+WHLSQUR+WHLSQDL+WHLSQDR
      
C
C  HL ----> STOPS 
      SUSY = 1
      DO 713 I=1,2
      DO 713 J=1,2
c     QSQ = (YMST(I)+YMST(J))/2
      QSQ = AML
      SUSY = 1
      IF(AML.GT.YMST(I)+YMST(J)) THEN
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
       CALL SQMBAPP_HDEC(QSQ)
       SUSY = 1+SQSUSY_HDEC(1,1,I,J,QSQ,0,1)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
       WHLST(I,J)=3*GF*AMZ**4/2.D0/DSQRT(2.D0)/PI*YLTT(I,J)**2*
     .      LAMB_HDEC(YMST(I)**2/AML**2,YMST(J)**2/AML**2)/AML
     .          *SUSY
c      write(6,*)'h -> stop: ',I,J,AML,YMST(I),YMST(J),SUSY-1,
c    .           WHLST(I,J)/SUSY,WHLST(I,J)
c      write(6,*)'h -> stop: ',I,J,AML,YMST(I),YMST(J),SUSY-1
      ELSE
      WHLST(I,J)=0.D0
      ENDIF
 713  CONTINUE
C
C  HL ----> SBOTTOMS 
      SUSY = 1
      DO 714 I=1,2
      DO 714 J=1,2
c     QSQ = (YMSB(I)+YMSB(J))/2
      QSQ = AML
      IF(AML.GT.YMSB(I)+YMSB(J)) THEN
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
       CALL SQMBAPP_HDEC(QSQ)
       SUSY = 1+SQSUSY_HDEC(1,2,I,J,QSQ,0,1)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
       WHLSB(I,J)=3*GF*AMZ**4/2.D0/DSQRT(2.D0)/PI*YLBB(I,J)**2*
     .       LAMB_HDEC(YMSB(I)**2/AML**2,YMSB(J)**2/AML**2)/AML
     .      *SUSY
c      write(6,*)'h -> sbot: ',I,J,AML,YMSB(I),YMSB(J),SUSY-1,
c    .           WHLSB(I,J)/SUSY,WHLSB(I,J)
c      write(6,*)'h -> sbot: ',I,J,AML,YMSB(I),YMSB(J),SUSY-1
      ELSE
      WHLSB(I,J)=0.D0
      ENDIF
 714  CONTINUE
C
      WHLSTT=WHLST(1,1)+WHLST(1,2)+WHLST(2,1)+WHLST(2,2) 
      WHLSBB=WHLSB(1,1)+WHLSB(1,2)+WHLSB(2,1)+WHLSB(2,2) 
      WHLSQT=WHLSTT+WHLSBB+WHLSQ

      ELSE 
      WHLCHT=0.D0
      WHLNET=0.D0
      WHLSLT=0.D0
      WHLSQT=0.D0
C--Change thanks to Elzbieta Richter-Was
      DO I=1,2
       DO J=1,2
        WHLCH(I,J)=0.D0
        WHLST(I,J)=0.D0
        WHLSB(I,J)=0.D0
        WHLSTAU(I,J)=0.D0
       ENDDO
      ENDDO
      DO I=1,4
       DO J=1,4
        WHLNE(I,J)=0.D0
       ENDDO
      ENDDO
      ENDIF

      IF(IGOLD.NE.0)THEN
C   HL ---> GOLDSTINOS
       DO 710 I=1,4
       IF (AML.GT.AMNEUT(I)) THEN
        WHLGD(I)=AML**5/AXMPL**2/AXMGD**2/48.D0/PI*
     .           (1.D0-AMNEUT(I)**2/AML**2)**4*AGDL(I)**2
       ELSE
        WHLGD(I)=0.D0
       ENDIF
 710   CONTINUE
       WHLGDT=WHLGD(1)+WHLGD(2)+WHLGD(3)+WHLGD(4)
      ELSE
       WHLGDT=0
      ENDIF

C    ==========  TOTAL WIDTH AND BRANCHING RATIOS 
      WTOT=HLL+HMM+HSS+HCC+HBB+HTT+HGG+HGA+HZGA+HWW+HZZ+HAA+HAZ+HHW
     .    +WHLCHT+WHLNET+WHLSLT+WHLSQT + WHLGDT

      wtot = wtot + hlchch
      hlbrchch=hlchch/wtot

c     print*,'wtot',wtot

      HLBRT=HTT/WTOT
      HLBRB=HBB/WTOT
      HLBRL=HLL/WTOT
      HLBRM=HMM/WTOT
      HLBRS=HSS/WTOT
      HLBRC=HCC/WTOT
      HLBRG=HGG/WTOT
      HLBRGA=HGA/WTOT
      HLBRZGA=HZGA/WTOT
      HLBRW=HWW/WTOT
      HLBRZ=HZZ/WTOT
      HLBRA=HAA/WTOT
      HLBRAZ=HAZ/WTOT
      HLBRHW=HHW/WTOT
      DO 811 I=1,2
      DO 811 J=1,2
      HLBRSC(I,J)=WHLCH(I,J)/WTOT
811   CONTINUE
      DO 812 I=1,4
      DO 812 J=1,4
      HLBRSN(I,J)=WHLNE(I,J)/WTOT
812   CONTINUE
      HLBRCHT=WHLCHT/WTOT 
      HLBRNET=WHLNET/WTOT 
      HLBRSL=WHLSLT/WTOT 
      HLBRSQ=WHLSQ/WTOT 
      HLBRSQT=WHLSQT/WTOT 
      HLBRGD =WHLGDT/WTOT
      HLWDTH=WTOT

      BHLSLNL = WHLSLNL/WTOT
      BHLSLEL = WHLSLEL/WTOT
      BHLSLER = WHLSLER/WTOT
      BHLSQUL = WHLSQUL/WTOT
      BHLSQUR = WHLSQUR/WTOT
      BHLSQDL = WHLSQDL/WTOT
      BHLSQDR = WHLSQDR/WTOT
      DO I = 1,2
       DO J = 1,2
        BHLST(I,J) = WHLST(I,J)/WTOT
        BHLSB(I,J) = WHLSB(I,J)/WTOT
        BHLSTAU(I,J) = WHLSTAU( I,J)/WTOT
       ENDDO
      ENDDO

      ENDIF

      IF(IHIGGS.GT.1)THEN
      

C        =========================================================
C                       CHARGED HIGGS DECAYS
C        =========================================================
      TB=TGBET
C     =============  RUNNING MASSES 
      RMS = RUNM_HDEC(AMCH,3)
      RMC = RUNM_HDEC(AMCH,4)
      RMB = RUNM_HDEC(AMCH,5)
      RMT = RUNM_HDEC(AMCH,6)
      ASH=ALPHAS_HDEC(AMCH,3)
C     =============== PARTIAL WIDTHS 
C  H+ ---> MU NMU
      XGAM = GAB
      if(i2hdm.eq.1) then
         xgam = galep
      endif
      IF(IOFSUSY.EQ.0) THEN
       CALL STAUSUSY_HDEC(GLB,GHB,GAB,XGLM,XGHM,XGAM,QSUSY,0)
      ENDIF
      IF(AMCH.LE.AMMUON) THEN
       HMN = 0
      ELSE
      HMN=CFF(AMCH,XGAM,(AMMUON/AMCH)**2,0.D0)
      ENDIF

c     print*,''
c     print*,'H+ decay widths'
c     print*,'H+ -> nu mu',hmn

C  H+ ---> TAU NTAU
      XGAT = GAB
      if(i2hdm.eq.1) then
         xgat = galep
      endif
      IF(IOFSUSY.EQ.0) THEN
       CALL STAUSUSY_HDEC(GLB,GHB,GAB,XGLT,XGHT,XGAT,QSUSY,1)
      ENDIF
      IF(AMCH.LE.AMTAU) THEN
       HLN = 0
      ELSE
      HLN=CFF(AMCH,XGAT,(AMTAU/AMCH)**2,0.D0)
      ENDIF

c     print*,'H+ -> tau ntau',hln

C  H+ --> SU
      EPS = 1.D-12
      RATX = 1
      IF(IOFSUSY.EQ.0) THEN
       CALL STRSUSY_HDEC(GLB,GHB,GAB,XGLS,XGHS,XGAS,QSUSY,LOOP)
       RATX = XGAS/GAB
      ENDIF
      IF(AMCH.LE.AMS+EPS) THEN
       HSU = 0
      ELSE
       HSU1=3.D0*VUS**2*CQCDM(AMCH,TB,(AMS/AMCH)**2,EPS,RATX)
       HSU2=3.D0*VUS**2*CQCD(AMCH,TB,(RMS/AMCH)**2,EPS,RATX)
c MMM changed 21/8/13
       if(i2hdm.eq.1) then
          hsu1=3.d0*vus**2*
     .         cqcdm2hdm(amch,gab,gat,(ams/amch)**2,eps,ratx)
          hsu2=3.d0*vus**2*cqcd2hdm(amch,gab,gat,(rms/amch)**2,eps,ratx)
       endif
c end MMM changed 21/8/13
       IF(HSU2.LT.0.D0) HSU2 = 0
       RAT = AMS/AMCH
       HSU = QQINT_HDEC(RAT,HSU1,HSU2)
      ENDIF

c     print*,'H+ -> su',hsu,3.d0*vus**2*
c    .         cqcdm2hdm(amch,gab,gat,(ams/amch)**2,eps,ratx)

C  H+ --> CS
      RATX = RMS/AMS
      RATY = 1
      IF(IOFSUSY.EQ.0) THEN
       CALL STRSUSY_HDEC(GLB,GHB,GAB,XGLS,XGHS,XGAS,QSUSY,LOOP)
       RATX = RMS/AMS*XGAS/GAB
       RATY = XGAS/GAB
      ENDIF
      IF(AMCH.LE.AMS+AMC) THEN
       HSC = 0
      ELSE
       HSC1=3.D0*CQCDM(AMCH,TB,(AMS/AMCH)**2,(AMC/AMCH)**2,RATX)*VCS**2
       HSC2=3.D0*CQCD(AMCH,TB,(RMS/AMCH)**2,(RMC/AMCH)**2,RATY)*VCS**2
c MMM changed 21/8/13
       if(i2hdm.eq.1) then
          hsc1=3.d0*VCS**2*
     .         cqcdm2hdm(amch,gab,gat,(ams/amch)**2,(amc/amch)**2,ratx)
          hsc2=3.d0*VCS**2*
     .         cqcd2hdm(amch,gab,gat,(rms/amch)**2,(rmc/amch)**2,raty)
c         print*,'rms,rmc',hsc1,hsc2,rms,rmc
       endif
c end MMM changed 21/8/13
       IF(HSC2.LT.0.D0) HSC2 = 0
       RAT = (AMS+AMC)/AMCH
       HSC = QQINT_HDEC(RAT,HSC1,HSC2)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c      HSC1=3.D0*CQCDM(AMCH,TB,(AMS/AMCH)**2,(AMC/AMCH)**2,1.D0)
c      HSC2=3.D0*CQCD(AMCH,TB,(RMS/AMCH)**2,(RMC/AMCH)**2,1.D0)
c      IF(HSC2.LT.0.D0) HSC2 = 0
c      RAT = (AMS+AMC)/AMCH
c      HSC0 = QQINT_HDEC(RAT,HSC1,HSC2)
c      write(6,*)'H+- --> cs: ',AMCH,HSC,HSC0,HSC/HSC0,RATX**2
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      ENDIF

c     print*,'H+ -> cs',hsc

c Maggie added 16/10/2013
C  H+ --> CD
      RATX = 1
      EPS = 1D-12
      IF(AMCH.LE.AMC) THEN
       HCD = 0
      ELSE
       HCD1=3.D0*CQCDM(AMCH,TB,EPS,(AMC/AMCH)**2,RATX)*VCD**2
       HCD2=3.D0*CQCD(AMCH,TB,EPS,(RMC/AMCH)**2,RATX)*VCD**2
       if(i2hdm.eq.1) then
          hCD1=3.d0*vcd**2*
     .         cqcdm2hdm(amch,gab,gat,eps,(amc/amch)**2,ratx)
          hCD2=3.d0*vcd**2*
     .         cqcd2hdm(amch,gab,gat,eps,(rmc/amch)**2,ratx)
       endif
       IF(HCD2.LT.0.D0) HCD2 = 0
       RAT = (EPS+AMC)/AMCH
       HCD = QQINT_HDEC(RAT,HCD1,HCD2)
      ENDIF

c     print*,'H+ -> cd',hcd
c end Maggie added 16/10/2013

C  H+ --> CB
      RATX = 1
      QQ = AMB
      SUSY = 0
      XGAB = GAB
c     SSUSY = AMCH
      SSUSY = (AMSB(1)+AMSB(2)+AMGLU)/3*QSUSY
      AS0 = ALPHAS_HDEC(SSUSY,3)
      IF(IOFSUSY.EQ.0) THEN
       I0 = 1
c      write(6,*)
c      write(6,*)'H+ -> cb: ',amch
c      write(6,*)
       CALL DMBAPP_HDEC(I0,DGLB,DGHB,DGAB,QSUSY,LOOP)
       I0 = 1
       BSC = (AMSQ+AMUR+AMDR)/3
c      XMB = RUNM_HDEC(BSC,5)
       XMB = AMB
c      SUSY = COFSUSY_HDEC(I0,AMB,XMB,QQ)*AS0/PI - 2*DGLB
c      write(6,*)
c      write(6,*)'H+ -> cb: ',amch
c      write(6,*)
       CALL BOTSUSY_HDEC(GLB,GHB,GAB,XGLB,XGHB,XGAB,QSUSY,LOOP)
      ENDIF
      RATX = XGAB/GAB
c     write(6,*)'ratio = ',ratx
      IF(AMCH.LE.AMB+AMC) THEN
       HBC = 0
      ELSE
       HBC1=3.D0*VCB**2*CQCDM(AMCH,TB,(AMB/AMCH)**2,(AMC/AMCH)**2,RATX)
       HBC2=3.D0*VCB**2*CQCD(AMCH,TB,(RMB/AMCH)**2,(RMC/AMCH)**2,RATX)
c MMM changed 21/8/13
       if(i2hdm.eq.1) then
          hbc1=3.d0*vcb**2*
     .         cqcdm2hdm(amch,gab,gat,(amb/amch)**2,(amc/amch)**2,ratx)
          hbc2=3.d0*vcb**2*
     .         cqcd2hdm(amch,gab,gat,(rmb/amch)**2,(rmc/amch)**2,ratx)
       endif
c end MMM changed 21/8/13
       IF(HBC2.LT.0.D0) HBC2 = 0
       RAT = (AMB+AMC)/AMCH
       HBC = QQINT_HDEC(RAT,HBC1,HBC2)
      ENDIF

c     print*,'H+ -> cb',hbc

C  H+ --> BU
      EPS = 1.D-12
      IF(AMCH.LE.AMB+EPS) THEN
       HBU = 0
      ELSE
       HBU1=3.D0*VUB**2*CQCDM(AMCH,TB,(AMB/AMCH)**2,EPS,RATX)
       HBU2=3.D0*VUB**2*CQCD(AMCH,TB,(RMB/AMCH)**2,EPS,RATX)
c MMM changed 21/8/13
       if(i2hdm.eq.1) then
          hbu1=3.d0*vub**2*
     .         cqcdm2hdm(amch,gab,gat,(amb/amch)**2,eps,ratx)
          hbu2=3.d0*vub**2*
     .         cqcd2hdm(amch,gab,gat,(rmb/amch)**2,eps,ratx)
       endif
c end MMM changed 21/8/13
       IF(HBU2.LT.0.D0) HBU2 = 0
       RAT = AMB/AMCH
       HBU = QQINT_HDEC(RAT,HBU1,HBU2)
      ENDIF

c     print*,'H+ -> ub',hbu

C  H+ --> TD :
      EPS = 1.D-12
      IF(IONSH.EQ.0)THEN
       DLD=2D0
       DLU=2D0
       XM1 = AMT-DLD
       XM2 = AMT+DLU
       IF (AMCH.LE.AMW) THEN
        HDT=0.D0
       ELSEIF (AMCH.LE.XM1) THEN
        FACTB=3.D0*GF**2*AMCH*AMT**4/32.D0/PI**3*gat**2
        CALL CTOTT_HDEC(AMCH,AMT,EPS,AMW,i2hdm,gat,gab,CTT0)
        HDT=VTD**2*FACTB*CTT0
       ELSEIF (AMCH.LE.XM2) THEN
        XX(1) = XM1-1D0
        XX(2) = XM1
        XX(3) = XM2
        XX(4) = XM2+1D0
        FACTB=3.D0*GF**2*XX(1)*AMT**4/32.D0/PI**3*gat**2
        CALL CTOTT_HDEC(XX(1),AMT,EPS,AMW,i2hdm,gat,gab,CTT0)
        YY(1)=VTD**2*FACTB*CTT0
        FACTB=3.D0*GF**2*XX(2)*AMT**4/32.D0/PI**3*gat**2
        CALL CTOTT_HDEC(XX(2),AMT,EPS,AMW,i2hdm,gat,gab,CTT0)
        YY(2)=VTD**2*FACTB*CTT0
        XMB = RUNM_HDEC(XX(3),5)
        XMT = RUNM_HDEC(XX(3),6)
        XYZ2 = 3.D0*CQCD(XX(3),TB,(EPS/XX(3))**2,(XMT/XX(3))**2,RATX)
c MMM changed 21/8/13
        if(i2hdm.eq.1) then
           XYZ2 = 3.D0*
     .       CQCD2HDM(XX(3),gab,gat,(EPS/XX(3))**2,(XMT/XX(3))**2,RATX)
        endif
c end MMM changed 21/8/13
        IF(XYZ2.LT.0.D0) XYZ2 = 0
        XYZ1 = 3.D0*CQCDM(XX(3),TB,(EPS/XX(3))**2,(AMT/XX(3))**2,RATX)
c MMM changed 21/8/13
        if(i2hdm.eq.1) then
        XYZ1 = 3.D0*
     .      CQCDM2HDM(XX(3),gab,gat,(EPS/XX(3))**2,(AMT/XX(3))**2,RATX)
      endif
c end MMM changed 21/8/13
        RAT = (AMT)/XX(3)
        YY(3) = VTD**2*QQINT_HDEC(RAT,XYZ1,XYZ2)
        XMB = RUNM_HDEC(XX(4),5)
        XMT = RUNM_HDEC(XX(4),6)
        XYZ2 = 3.D0*CQCD(XX(4),TB,(EPS/XX(4))**2,(XMT/XX(4))**2,RATX)
c MMM changed 21/8/13
        if(i2hdm.eq.1) then
           XYZ2 = 3.D0*
     .       CQCD2HDM(XX(4),gab,gat,(EPS/XX(4))**2,(XMT/XX(4))**2,RATX)
        endif
c end MMM changed 21/8/13
        IF(XYZ2.LT.0.D0) XYZ2 = 0
        XYZ1 = 3.D0*CQCDM(XX(4),TB,(EPS/XX(4))**2,(AMT/XX(4))**2,RATX)
c MMM changed 21/8/13
        if(i2hdm.eq.1) then
        XYZ1 = 3.D0*
     .      CQCDM2HDM(XX(4),gab,gat,(EPS/XX(4))**2,(AMT/XX(4))**2,RATX)
      endif
c end MMM changed 21/8/13
        RAT = (AMT)/XX(4)
        YY(4) = VTD**2*QQINT_HDEC(RAT,XYZ1,XYZ2)
        HDT = FINT_HDEC(AMCH,XX,YY)
       ELSE
        HDT2=3.D0*VTD**2*CQCD(AMCH,TB,(EPS/AMCH)**2,(RMT/AMCH)**2,RATX)
c MMM changed 21/8/13
        if(i2hdm.eq.1) then
           HDT2=3.D0*VTD**2*
     .          CQCD2HDM(AMCH,gab,gat,(EPS/AMCH)**2,(RMT/AMCH)**2,RATX)
        endif
c end MMM changed 21/8/13
        IF(HDT2.LT.0.D0) HDT2 = 0
        HDT1=3.D0*VTD**2*CQCDM(AMCH,TB,(EPS/AMCH)**2,(AMT/AMCH)**2,RATX)
c MMM changed 21/8/13
        if(i2hdm.eq.1) then
           HDT1=3.D0*VTD**2*
     .          CQCDM2HDM(AMCH,gab,gat,(EPS/AMCH)**2,(AMT/AMCH)**2,RATX)
      endif
c end MMM changed 21/8/13
        RAT = (AMT)/AMCH
        HDT = QQINT_HDEC(RAT,HDT1,HDT2)
       ENDIF
      ELSE
       IF (AMCH.LE.AMT) THEN
        HDT=0.D0
       ELSE
        HDT2=3.D0*VTD**2*CQCD(AMCH,TB,EPS,(RMT/AMCH)**2,RATX)
c MMM changed 21/8/13
        if(i2hdm.eq.1) then
           HDT2=3.D0*VTD**2*
     .          CQCD2HDM(AMCH,gab,gat,EPS,(RMT/AMCH)**2,RATX)
        endif
c end MMM changed 21/8/13
        IF(HDT2.LT.0.D0) HDT2 = 0
        HDT1=3.D0*VTD**2*CQCDM(AMCH,TB,EPS,(AMT/AMCH)**2,RATX)
c MMM changed 21/8/13
        if(i2hdm.eq.1) then
           HDT1=3.D0*VTD**2*
     .          CQCDM2HDM(AMCH,gab,gat,EPS,(AMT/AMCH)**2,RATX)
      endif
c end MMM changed 21/8/13
        RAT = (AMT)/AMCH
        HDT = QQINT_HDEC(RAT,HDT1,HDT2)
       ENDIF
      ENDIF

c     print*,'H+ -> td',hdt

C  H+ --> TS :
      EPS = 1.D-12
      RATX = RMS/AMS
      RATY = 1
      XGAS = GAB
      IF(IOFSUSY.EQ.0) THEN
       CALL STRSUSY_HDEC(GLB,GHB,GAB,XGLS,XGHS,XGAS,QSUSY,LOOP)
       RATX = RMS/AMS*XGAS/GAB
       RATY = XGAS/GAB
      ENDIF
      IF(IONSH.EQ.0)THEN
       DLD=2D0
       DLU=2D0
       XM1 = AMT+AMS-DLD
       XM2 = AMT+AMS+DLU
       IF (AMCH.LE.AMW+2*AMS) THEN
        HST=0.D0
       ELSEIF (AMCH.LE.XM1) THEN
        FACTB=3.D0*GF**2*AMCH*AMT**4/32.D0/PI**3*gat**2
        CALL CTOTT_HDEC(AMCH,AMT,EPS,AMW,i2hdm,gat,xgas,CTT0)
c       CALL CTOTT_HDEC(AMCH,AMT,EPS,AMW,i2hdm,gat,gab,CTT1)
c       write(6,*)'H+ -> ts: ',amch,ctt0/ctt1
        HST=VTS**2*FACTB*CTT0
       ELSEIF (AMCH.LE.XM2) THEN
        XX(1) = XM1-1D0
        XX(2) = XM1
        XX(3) = XM2
        XX(4) = XM2+1D0
        FACTB=3.D0*GF**2*XX(1)*AMT**4/32.D0/PI**3*gat**2
        CALL CTOTT_HDEC(XX(1),AMT,EPS,AMW,i2hdm,gat,xgas,CTT0)
        YY(1)=VTS**2*FACTB*CTT0
        FACTB=3.D0*GF**2*XX(2)*AMT**4/32.D0/PI**3*gat**2
        CALL CTOTT_HDEC(XX(2),AMT,EPS,AMW,i2hdm,gat,xgas,CTT0)
        YY(2)=VTS**2*FACTB*CTT0
        XMS = RUNM_HDEC(XX(3),3)
        XMT = RUNM_HDEC(XX(3),6)
        XYZ2 = 3.D0*CQCD(XX(3),TB,(XMS/XX(3))**2,(XMT/XX(3))**2,RATY)
c MMM changed 21/8/13
        if(i2hdm.eq.1) then
           XYZ2 = 3.D0*
     .       CQCD2HDM(XX(3),gab,gat,(XMS/XX(3))**2,(XMT/XX(3))**2,RATY)
        endif
c end MMM changed 21/8/13
        IF(XYZ2.LT.0.D0) XYZ2 = 0
        XYZ1 = 3.D0*CQCDM(XX(3),TB,(AMS/XX(3))**2,(AMT/XX(3))**2,RATX)
c MMM changed 21/8/13
        if(i2hdm.eq.1) then
        XYZ1 = 3.D0*
     .      CQCDM2HDM(XX(3),gab,gat,(AMS/XX(3))**2,(AMT/XX(3))**2,RATX)
      endif
c end MMM changed 21/8/13
        RAT = (AMS+AMT)/XX(3)
        YY(3) = VTS**2*QQINT_HDEC(RAT,XYZ1,XYZ2)
        XMS = RUNM_HDEC(XX(4),3)
        XMT = RUNM_HDEC(XX(4),6)
        XYZ2 = 3.D0*CQCD(XX(4),TB,(XMS/XX(4))**2,(XMT/XX(4))**2,RATY)
c MMM changed 21/8/13
        if(i2hdm.eq.1) then
           XYZ2 = 3.D0*
     .       CQCD2HDM(XX(4),gab,gat,(XMS/XX(4))**2,(XMT/XX(4))**2,RATY)
        endif
c end MMM changed 21/8/13
        IF(XYZ2.LT.0.D0) XYZ2 = 0
        XYZ1 = 3.D0*CQCDM(XX(4),TB,(AMS/XX(4))**2,(AMT/XX(4))**2,RATX)
c MMM changed 21/8/13
        if(i2hdm.eq.1) then
        XYZ1 = 3.D0*
     .      CQCDM2HDM(XX(4),gab,gat,(AMS/XX(4))**2,(AMT/XX(4))**2,RATX)
      endif
c end MMM changed 21/8/13
        RAT = (AMS+AMT)/XX(4)
        YY(4) = VTS**2*QQINT_HDEC(RAT,XYZ1,XYZ2)
        HST = FINT_HDEC(AMCH,XX,YY)
       ELSE
        HST2=3.D0*VTS**2*CQCD(AMCH,TB,(RMS/AMCH)**2,(RMT/AMCH)**2,RATY)
c MMM changed 21/8/13
        if(i2hdm.eq.1) then
           HST2=3.D0*VTS**2*
     .          CQCD2HDM(AMCH,gab,gat,(RMS/AMCH)**2,(RMT/AMCH)**2,RATY)
c          print*,'rms,rmt',rms,rmt
        endif
c end MMM changed 21/8/13
        IF(HST2.LT.0.D0) HST2 = 0
        HST1=3.D0*VTS**2*CQCDM(AMCH,TB,(AMS/AMCH)**2,(AMT/AMCH)**2,RATX)
c MMM changed 21/8/13
        if(i2hdm.eq.1) then
           HST1=3.D0*VTS**2*
     .          CQCDM2HDM(AMCH,gab,gat,(AMS/AMCH)**2,(AMT/AMCH)**2,RATX)
      endif
c end MMM changed 21/8/13
        RAT = (AMS+AMT)/AMCH
        HST = QQINT_HDEC(RAT,HST1,HST2)
       ENDIF
      ELSE
       IF (AMCH.LE.AMT+AMS) THEN
        HST=0.D0
       ELSE
        HST2=3.D0*VTS**2*CQCD(AMCH,TB,(RMS/AMCH)**2,(RMT/AMCH)**2,RATY)
c MMM changed 21/8/13
        if(i2hdm.eq.1) then
           HST2=3.D0*VTS**2*
     .          CQCD2HDM(AMCH,gab,gat,(RMS/AMCH)**2,(RMT/AMCH)**2,RATY)
        endif
c end MMM changed 21/8/13
        IF(HST2.LT.0.D0) HST2 = 0
        HST1=3.D0*VTS**2*CQCDM(AMCH,TB,(AMS/AMCH)**2,(AMT/AMCH)**2,RATX)
c MMM changed 21/8/13
        if(i2hdm.eq.1) then
           HST1=3.D0*VTS**2*
     .          CQCDM2HDM(AMCH,gab,gat,(AMS/AMCH)**2,(AMT/AMCH)**2,RATX)
      endif
c end MMM changed 21/8/13
        RAT = (AMS+AMT)/AMCH
        HST = QQINT_HDEC(RAT,HST1,HST2)
       ENDIF
      ENDIF

c     print*,'H+ -> ts',hst

C  H+ --> TB :
      RATX = RMB/AMB
      RATY = 1
      XGAB = GAB
      IF(IOFSUSY.EQ.0) THEN
       CALL BOTSUSY_HDEC(GLB,GHB,GAB,XGLB,XGHB,XGAB,QSUSY,LOOP)
       RATX = RMB/AMB*XGAB/GAB
       RATY = XGAB/GAB
      ENDIF
      IF(IONSH.EQ.0)THEN
       DLD=2D0
       DLU=2D0
       XM1 = AMT+AMB-DLD
       XM2 = AMT+AMB+DLU
       IF (AMCH.LE.AMW+2*AMB) THEN
        HBT=0.D0
       ELSEIF (AMCH.LE.XM1) THEN
        FACTB=3.D0*GF**2*AMCH*AMT**4/32.D0/PI**3*gat**2
        CALL CTOTT_HDEC(AMCH,AMT,AMB,AMW,i2hdm,gat,xgab,CTT0)
c       CALL CTOTT_HDEC(AMCH,AMT,AMB,AMW,i2hdm,gat,gab,CTT1)
c       write(6,*)'H+ -> tb: ',amch,ctt0/ctt1
        HBT=VTB**2*FACTB*CTT0
       ELSEIF (AMCH.LE.XM2) THEN
        XX(1) = XM1-1D0
        XX(2) = XM1
        XX(3) = XM2
        XX(4) = XM2+1D0
        FACTB=3.D0*GF**2*XX(1)*AMT**4/32.D0/PI**3*gat**2
        CALL CTOTT_HDEC(XX(1),AMT,AMB,AMW,i2hdm,gat,xgab,CTT0)
        YY(1)=VTB**2*FACTB*CTT0
        FACTB=3.D0*GF**2*XX(2)*AMT**4/32.D0/PI**3*gat**2
        CALL CTOTT_HDEC(XX(2),AMT,AMB,AMW,i2hdm,gat,xgab,CTT0)
        YY(2)=VTB**2*FACTB*CTT0
        XMB = RUNM_HDEC(XX(3),5)
        XMT = RUNM_HDEC(XX(3),6)
        XYZ2 = 3.D0*CQCD(XX(3),TB,(XMB/XX(3))**2,(XMT/XX(3))**2,RATY)
c MMM changed 21/8/13
        if(i2hdm.eq.1) then
           XYZ2 = 3.D0*
     .       CQCD2HDM(XX(3),gab,gat,(XMB/XX(3))**2,(XMT/XX(3))**2,RATY)
        endif
c end MMM changed 21/8/13
        IF(XYZ2.LT.0.D0) XYZ2 = 0
        XYZ1 = 3.D0*CQCDM(XX(3),TB,(AMB/XX(3))**2,(AMT/XX(3))**2,RATX)
c MMM changed 21/8/13
        if(i2hdm.eq.1) then
        XYZ1 = 3.D0*
     .      CQCDM2HDM(XX(3),gab,gat,(AMB/XX(3))**2,(AMT/XX(3))**2,RATX)
      endif
c end MMM changed 21/8/13
        RAT = (AMB+AMT)/XX(3)
        YY(3) = VTB**2*QQINT_HDEC(RAT,XYZ1,XYZ2)
        XMB = RUNM_HDEC(XX(4),5)
        XMT = RUNM_HDEC(XX(4),6)
        XYZ2 = 3.D0*CQCD(XX(4),TB,(XMB/XX(4))**2,(XMT/XX(4))**2,RATY)
c MMM changed 21/8/13
        if(i2hdm.eq.1) then
           XYZ2 = 3.D0*
     .       CQCD2HDM(XX(4),gab,gat,(XMB/XX(4))**2,(XMT/XX(4))**2,RATY)
        endif
c end MMM changed 21/8/13
        IF(XYZ2.LT.0.D0) XYZ2 = 0
        XYZ1 = 3.D0*CQCDM(XX(4),TB,(AMB/XX(4))**2,(AMT/XX(4))**2,RATX)
c MMM changed 21/8/13
        if(i2hdm.eq.1) then
        XYZ1 = 3.D0*
     .      CQCDM2HDM(XX(4),gab,gat,(AMB/XX(4))**2,(AMT/XX(4))**2,RATX)
      endif
c end MMM changed 21/8/13
        RAT = (AMB+AMT)/XX(4)
        YY(4) = VTB**2*QQINT_HDEC(RAT,XYZ1,XYZ2)
        HBT = FINT_HDEC(AMCH,XX,YY)
       ELSE
        HBT2=3.D0*VTB**2*CQCD(AMCH,TB,(RMB/AMCH)**2,(RMT/AMCH)**2,RATY)
c MMM changed 21/8/13
        if(i2hdm.eq.1) then
           HBT2=3.D0*VTB**2*
     .          CQCD2HDM(AMCH,gab,gat,(RMB/AMCH)**2,(RMT/AMCH)**2,RATY)
c          print*,'rmb,rmt',rmb,rmt
        endif
c end MMM changed 21/8/13
        IF(HBT2.LT.0.D0) HBT2 = 0
        HBT1=3.D0*VTB**2*CQCDM(AMCH,TB,(AMB/AMCH)**2,(AMT/AMCH)**2,RATX)
c MMM changed 21/8/13
        if(i2hdm.eq.1) then
           HBT1=3.D0*VTB**2*
     .          CQCDM2HDM(AMCH,gab,gat,(AMB/AMCH)**2,(AMT/AMCH)**2,RATX)
      endif
c end MMM changed 21/8/13
        RAT = (AMB+AMT)/AMCH
        HBT = QQINT_HDEC(RAT,HBT1,HBT2)
       ENDIF
      ELSE
       IF (AMCH.LE.AMT+AMB) THEN
        HBT=0.D0
       ELSE
        HBT2=3.D0*VTB**2*CQCD(AMCH,TB,(RMB/AMCH)**2,(RMT/AMCH)**2,RATY)
c MMM changed 21/8/13
        if(i2hdm.eq.1) then
           HBT2=3.D0*VTB**2*
     .          CQCD2HDM(AMCH,gab,gat,(RMB/AMCH)**2,(RMT/AMCH)**2,RATY)
        endif
c end MMM changed 21/8/13
        IF(HBT2.LT.0.D0) HBT2 = 0
        HBT1=3.D0*VTB**2*CQCDM(AMCH,TB,(AMB/AMCH)**2,(AMT/AMCH)**2,RATX)
c MMM changed 21/8/13
        if(i2hdm.eq.1) then
           HBT1=3.D0*VTB**2*
     .          CQCDM2HDM(AMCH,gab,gat,(AMB/AMCH)**2,(AMT/AMCH)**2,RATX)
      endif
c end MMM changed 21/8/13
        RAT = (AMB+AMT)/AMCH
        HBT = QQINT_HDEC(RAT,HBT1,HBT2)
       ENDIF
      ENDIF

c     print*,'H+ -> tb',hbt

c  H+ ---> W h
      IF(IONSH.EQ.0)THEN
       DLD=3D0
       DLU=5D0
       XM1 = AMW+AML-DLD
       XM2 = AMW+AML+DLU
       IF (AMCH.LT.AML) THEN
        HWH=0
       ELSEIF (AMCH.LE.XM1) THEN
        IF(AMCH.LE.DABS(AMW-AML))THEN
         HWH=0
        ELSE
         HWH=9.D0*GF**2/16.D0/PI**3*AMW**4*AMCH*GHVV**2
     .      *HVH((AML/AMCH)**2,(AMW/AMCH)**2)
        ENDIF
       ELSEIF (AMCH.LT.XM2) THEN
        XX(1) = XM1-1D0
        XX(2) = XM1
        XX(3) = XM2
        XX(4) = XM2+1D0
        YY(1) = 9.D0*GF**2/16.D0/PI**3*AMW**4*XX(1)
     .         *HVH((AML/XX(1))**2,(AMW/XX(1))**2)
        YY(2) = 9.D0*GF**2/16.D0/PI**3*AMW**4*XX(2)
     .         *HVH((AML/XX(2))**2,(AMW/XX(2))**2)
        CWH=LAMB_HDEC(AML**2/XX(3)**2,AMW**2/XX(3)**2)
     .     *LAMB_HDEC(XX(3)**2/AMW**2,AML**2/AMW**2)**2
        YY(3)=GF/8.D0/DSQRT(2D0)/PI*AMW**4/XX(3)*CWH
        CWH=LAMB_HDEC(AML**2/XX(4)**2,AMW**2/XX(4)**2)
     .     *LAMB_HDEC(XX(4)**2/AMW**2,AML**2/AMW**2)**2
        YY(4)=GF/8.D0/DSQRT(2D0)/PI*AMW**4/XX(4)*CWH
        HWH = FINT_HDEC(AMCH,XX,YY)*GHVV**2
       ELSE
        CWH=LAMB_HDEC(AML**2/AMCH**2,AMW**2/AMCH**2)
     .     *LAMB_HDEC(AMCH**2/AMW**2,AML**2/AMW**2)**2
        HWH=GF/8.D0/DSQRT(2D0)/PI*AMW**4/AMCH*GHVV**2*CWH
       ENDIF
      ELSE
       IF (AMCH.LT.AMW+AML) THEN
        HWH=0
       ELSE
        CWH=LAMB_HDEC(AML**2/AMCH**2,AMW**2/AMCH**2)
     .     *LAMB_HDEC(AMCH**2/AMW**2,AML**2/AMW**2)**2
        HWH=GF/8.D0/DSQRT(2D0)/PI*AMW**4/AMCH*GHVV**2*CWH
       ENDIF
      ENDIF

c     print*,'H+ -> W+ h',hwh

c MMM changed 21/8/2013
c  H+ ---> W H
      if(i2hdm.eq.1) then
      IF(IONSH.EQ.0)THEN
       DLD=3D0
       DLU=5D0
       XM1 = AMW+AMH-DLD
       XM2 = AMW+AMH+DLU
       IF (AMCH.LT.AMH) THEN
        HWHH=0
       ELSEIF (AMCH.LE.XM1) THEN
        IF(AMCH.LE.DABS(AMW-AMH))THEN
         HWHH=0
        ELSE
         HWHH=9.D0*GF**2/16.D0/PI**3*AMW**4*AMCH*GLVV**2
     .      *HVH((AMH/AMCH)**2,(AMW/AMCH)**2)
        ENDIF
       ELSEIF (AMCH.LT.XM2) THEN
        XX(1) = XM1-1D0
        XX(2) = XM1
        XX(3) = XM2
        XX(4) = XM2+1D0
        YY(1) = 9.D0*GF**2/16.D0/PI**3*AMW**4*XX(1)
     .         *HVH((AMH/XX(1))**2,(AMW/XX(1))**2)
        YY(2) = 9.D0*GF**2/16.D0/PI**3*AMW**4*XX(2)
     .         *HVH((AMH/XX(2))**2,(AMW/XX(2))**2)
        CWH=LAMB_HDEC(AMH**2/XX(3)**2,AMW**2/XX(3)**2)
     .     *LAMB_HDEC(XX(3)**2/AMW**2,AMH**2/AMW**2)**2
        YY(3)=GF/8.D0/DSQRT(2D0)/PI*AMW**4/XX(3)*CWH
        CWH=LAMB_HDEC(AMH**2/XX(4)**2,AMW**2/XX(4)**2)
     .     *LAMB_HDEC(XX(4)**2/AMW**2,AMH**2/AMW**2)**2
        YY(4)=GF/8.D0/DSQRT(2D0)/PI*AMW**4/XX(4)*CWH
        HWHH = FINT_HDEC(AMCH,XX,YY)*GLVV**2
       ELSE
        CWH=LAMB_HDEC(AMH**2/AMCH**2,AMW**2/AMCH**2)
     .     *LAMB_HDEC(AMCH**2/AMW**2,AMH**2/AMW**2)**2
        HWHH=GF/8.D0/DSQRT(2D0)/PI*AMW**4/AMCH*GLVV**2*CWH
       ENDIF
      ELSE
       IF (AMCH.LT.AMW+AMH) THEN
        HWHH=0
       ELSE
        CWH=LAMB_HDEC(AMH**2/AMCH**2,AMW**2/AMCH**2)
     .     *LAMB_HDEC(AMCH**2/AMW**2,AMH**2/AMW**2)**2
        HWHH=GF/8.D0/DSQRT(2D0)/PI*AMW**4/AMCH*GLVV**2*CWH
       ENDIF
      ENDIF
      endif

      if(i2hdm.eq.0) then
         HWHH=0.D0
      endif

c     print*,'H+ -> W+ H',hwhh
c end MMM changed 21/8/2013

C  H+ ---> W A
      IF(IONSH.EQ.0)THEN
       IF (AMCH.LT.AMA) THEN
        HWA=0
       ELSEIF (AMCH.LT.AMW+AMA) THEN
        IF(AMCH.LE.DABS(AMW-AMA))THEN
         HWA=0
        ELSE
         HWA=9.D0*GF**2/16.D0/PI**3*AMW**4*AMCH
     .      *HVH((AMA/AMCH)**2,(AMW/AMCH)**2)
        ENDIF
       ELSE
        HWA=0.D0
       ENDIF
      ELSE
       IF (AMCH.LT.AMW+AMA) THEN
        HWA=0
       ELSE
        HWA=0.D0
       ENDIF
      ENDIF

c MMM changed 22/8/2013
      if(i2hdm.eq.1) then
         if(ionsh.eq.0)then
            dld=3d0
            dlu=5d0
            xm1 = amw+ama-dld
            xm2 = amw+ama+dlu
            if (amch.lt.ama) then
               hwa=0
            elseif (amch.le.xm1) then
               if(amch.le.dabs(amw-ama))then
                  hwa=0
               else
                  hwa=9.d0*gf**2/16.d0/pi**3*amw**4*amch
     .                 *hvh((ama/amch)**2,(amw/amch)**2)

                  ivegas(1) = 30000
                  ivegas(2) = 5
                  ivegas(3) = 120000
                  ivegas(4) = 10

                  iprint = 0

                  amhi = amch
                  amhj = amw
                  amhk = ama
                  gamtoti = 1.D0
                  gamtotj = 1.D0
                  gamtotk = gamw

c ---- initialization of VEGAS ----

c                  call RSTART(12,34,56,78)

c                  call INTEG(hvhinteg,2,iprint,ivegas,result,relative)

c                  hwa=9.d0*gf**2/16.d0/pi**3*amw**4*amch*result
               endif
            elseif (amch.lt.xm2) then
               xx(1) = xm1-1d0
               xx(2) = xm1
               xx(3) = xm2
               xx(4) = xm2+1d0
               yy(1) = 9.d0*gf**2/16.d0/pi**3*amw**4*xx(1)
     .              *hvh((ama/xx(1))**2,(amw/xx(1))**2)
               yy(2) = 9.d0*gf**2/16.d0/pi**3*amw**4*xx(2)
     .              *hvh((ama/xx(2))**2,(amw/xx(2))**2)
               cwh=lamb_hdec(ama**2/xx(3)**2,amw**2/xx(3)**2)
     .              *lamb_hdec(xx(3)**2/amw**2,ama**2/amw**2)**2
               yy(3)=gf/8.d0/dsqrt(2d0)/pi*amw**4/xx(3)*cwh
               cwh=lamb_hdec(ama**2/xx(4)**2,amw**2/xx(4)**2)
     .              *lamb_hdec(xx(4)**2/amw**2,ama**2/amw**2)**2
               yy(4)=gf/8.d0/dsqrt(2d0)/pi*amw**4/xx(4)*cwh
               hwa = fint_hdec(amch,xx,yy)
            else
               cwh=lamb_hdec(ama**2/amch**2,amw**2/amch**2)
     .              *lamb_hdec(amch**2/amw**2,ama**2/amw**2)**2
               hwa=gf/8.d0/dsqrt(2d0)/pi*amw**4/amch*cwh
            endif
         else
            if (amch.lt.amw+ama) then
               hwa=0
            else
               cwh=lamb_hdec(ama**2/amch**2,amw**2/amch**2)
     .              *lamb_hdec(amch**2/amw**2,ama**2/amw**2)**2
               hwa=gf/8.d0/dsqrt(2d0)/pi*amw**4/amch*cwh
            endif
         endif
      endif
c end MMM changed 22/8/2013

c     print*,'H+ -> W+ A',hwa

C  ======================= SUSY DECAYS 
      IF(IOFSUSY.EQ.0) THEN
C
C  H+ ----> CHARGINOS+NEUTRALINOS
C
      DO 751 I=1,2
      DO 751 J=1,4
      IF (AMCH.GT.AMCHAR(I)+AMNEUT(J)) THEN
      WHCCN(I,J)=GF*AMW**2/(2*PI*DSQRT(2.D0))/AMCH
     .   *LAMB_HDEC(AMCHAR(I)**2/AMCH**2,AMNEUT(J)**2/AMCH**2)*(
     .   (ACNL(I,J)**2+ACNR(I,J)**2)*(AMCH**2-AMCHAR(I)**2-XMNEUT(J)
     .   **2)-4.D0*ACNL(I,J)*ACNR(I,J)*XMCHAR(I)*XMNEUT(J) )
      ELSE
      WHCCN(I,J)=0.D0
      ENDIF
 751  CONTINUE

      WHCCNT=WHCCN(1,1)+WHCCN(1,2)+WHCCN(1,3)+WHCCN(1,4)
     .      +WHCCN(2,1)+WHCCN(2,2)+WHCCN(2,3)+WHCCN(2,4)
C
C  H+ ----> SLEPTONS 
C
      IF (AMCH.GT.AMSE(1)+AMSN1(1)) THEN
      WHCSL00=2*GF/4.D0/DSQRT(2D0)/PI*AMW**4/AMCH*DSIN(2.D0*B)**2
     .     *LAMB_HDEC(AMSE(1)**2/AMCH**2,AMSN1(1)**2/AMCH**2)
      ELSE 
      WHCSL00=0.D0
      ENDIF

      IF (AMCH.GT.AMSL(1)+AMSN(1)) THEN
      WHCSL11=GF/2.D0/DSQRT(2D0)/PI*AMW**4/AMCH*GCEN(1,1)**2
     .     *LAMB_HDEC(AMSL(1)**2/AMCH**2,AMSN(1)**2/AMCH**2)
      ELSE 
      WHCSL11=0.D0
      ENDIF

      IF (AMCH.GT.AMSL(2)+AMSN(1)) THEN
      WHCSL21=GF/2.D0/DSQRT(2D0)/PI*AMW**4/AMCH*GCEN(1,2)**2
     .     *LAMB_HDEC(AMSL(2)**2/AMCH**2,AMSN(1)**2/AMCH**2)
      ELSE 
      WHCSL21=0.D0
      ENDIF

      WHCSLT=WHCSL00+WHCSL11+WHCSL21

C
C  H+ ----> SQUARKS 
C
      IF (AMCH.GT.AMSU(1)+AMSD(1)) THEN
      WHCSQ=6*GF/4.D0/DSQRT(2D0)/PI*AMW**4/AMCH*DSIN(2.D0*B)**2
     .     *LAMB_HDEC(AMSU(1)**2/AMCH**2,AMSD(1)**2/AMCH**2)
      ELSE 
      WHCSQ=0.D0
      ENDIF
C
      DO 753 I=1,2
      DO 753 J=1,2
      IF(AMCH.GT.AMST(I)+AMSB(J)) THEN
      WHCSTB(I,J)=3*GF*AMW**4/2.D0/DSQRT(2.D0)/PI*GCTB(I,J)**2
     .      *LAMB_HDEC(AMST(I)**2/AMCH**2,AMSB(J)**2/AMCH**2)/AMCH
      ELSE
      WHCSTB(I,J)=0.D0
      ENDIF

 753  CONTINUE
C
      WHCSQT=WHCSQ+WHCSTB(1,1)+WHCSTB(1,2)+WHCSTB(2,1)+WHCSTB(2,2) 

      ELSE 
      WHCCNT=0.D0
      WHCSLT=0.D0
      WHCSQT=0.D0
      WHCGDT=0.D0
C--Change thanks to Elzbieta Richter-Was
      DO I=1,2
       DO J=1,2
        WHCSTB(I,J)=0.D0
       ENDDO
      ENDDO
      DO I=1,2
       DO J=1,4
        WHCCN(I,J)=0.D0
       ENDDO
      ENDDO
      ENDIF

      IF(IGOLD.NE.0)THEN
C   HC ---> GOLDSTINOS
       DO 750 I=1,2
       IF (AMCH.GT.AMCHAR(I)) THEN
        WHCGD(I)=AMCH**5/AXMPL**2/AXMGD**2/48.D0/PI*
     .           (1.D0-AMCHAR(I)**2/AMCH**2)**4*AGDC(I)**2
       ELSE
        WHCGD(I)=0.D0
       ENDIF
 750   CONTINUE
       WHCGDT=WHCGD(1)+WHCGD(2)
      ELSE
       WHCGDT=0
      ENDIF
C
C    ==========  TOTAL WIDTH AND BRANCHING RATIOS 
C
      WTOT=HLN+HMN+HSU+HBU+HSC+HBC+HBT+HWH+HWA+WHCCNT+WHCSLT+WHCSQT
     .    +WHCGDT

c MMM changed 21/8/2013
      WTOT=WTOT+HCD+HST+HDT

      if(i2hdm.eq.1) then
         wtot = wtot+hwhh
         HCBRWHH=hwhh/wtot
      endif
      if(i2hdm.eq.0) then
         hcbrwhh=0.D0
      endif
c end MMM changed 21/8/2013
      HCBRCD=HCD/WTOT
      HCBRTS=HST/WTOT
      HCBRTD=HDT/WTOT

c     print*,'wtot',wtot

      HCBRL=HLN/WTOT
      HCBRM=HMN/WTOT
      HCBRS=HSU/WTOT
      HCBRBU=HBU/WTOT
      HCBRC=HSC/WTOT
      HCBRB=HBC/WTOT
      HCBRT=HBT/WTOT
      HCBRW=HWH/WTOT
      HCBRA=HWA/WTOT
      DO 851 I=1,2
      DO 851 J=1,4
      HCBRSU(I,J)=WHCCN(I,J)/WTOT
851   CONTINUE
      HCBRCNT=WHCCNT/WTOT
      HCBRSL=WHCSLT/WTOT 
      HCBRSQ=WHCSQ/WTOT 
      HCBRSQT=WHCSQT/WTOT 
      DO 853 I=1,2
      DO 853 J=1,2
      HCBRSTB(I,J)=WHCSTB(I,J)/WTOT
853   CONTINUE
      HCBRGD=WHCGDT/WTOT
      HCWDTH=WTOT

      BHCSL00 = WHCSL00/WTOT
      BHCSL11 = WHCSL11/WTOT
      BHCSL21 = WHCSL21/WTOT
      BHCSQ = WHCSQ/WTOT
      DO I = 1,2
       DO J = 1,2
        BHCSTB(I,J) = WHCSTB(I,J)/WTOT
       ENDDO
      ENDDO

      GAMC0 = WTOT

      ENDIF

      IF(IHIGGS.EQ.2.OR.IHIGGS.EQ.5)THEN
     
C        =========================================================
C                       HEAVY CP EVEN HIGGS DECAYS
C        =========================================================
C     =============  RUNNING MASSES 
      RMS = RUNM_HDEC(AMH,3)
      RMC = RUNM_HDEC(AMH,4)
      RMB = RUNM_HDEC(AMH,5)
      RMT = RUNM_HDEC(AMH,6)
      RATCOUP = GHT/GHB
      HIGTOP = AMH**2/AMT**2

      ASH=ALPHAS_HDEC(AMH,3)
      AMC0=1.D8
      AMB0=2.D8
C     AMT0=3.D8
      AS3=ALPHAS_HDEC(AMH,3)
      AMC0=AMC
      AS4=ALPHAS_HDEC(AMH,3)
      AMB0=AMB
C     AMT0=AMT

C     =============== PARTIAL WIDTHS 
C  H ---> G G
       EPS=1.D-8
       NFEXT = 3
       ASG = AS3
       CTT = 4*AMT**2/AMH**2*DCMPLX(1D0,-EPS)
       CTB = 4*AMB**2/AMH**2*DCMPLX(1D0,-EPS)
       CAT = 2*CTT*(1+(1-CTT)*CF(CTT))*GHT
       CAB = 2*CTB*(1+(1-CTB)*CF(CTB))*GHB
       CTC = 4*AMC**2/AMH**2*DCMPLX(1D0,-EPS)
       CAC = 2*CTC*(1+(1-CTC)*CF(CTC))*GHT
C
       IF(IOFSUSY.EQ.0) THEN 
       CSB1= 4*AMSB(1)**2/AMH**2*DCMPLX(1D0,-EPS)
       CSB2= 4*AMSB(2)**2/AMH**2*DCMPLX(1D0,-EPS)
       CST1= 4*AMST(1)**2/AMH**2*DCMPLX(1D0,-EPS)
       CST2= 4*AMST(2)**2/AMH**2*DCMPLX(1D0,-EPS)
C
       CXB1=-AMZ**2/AMSB(1)**2*CSB1*(1-CSB1*CF(CSB1))*GHBB(1,1)
       CXB2=-AMZ**2/AMSB(2)**2*CSB2*(1-CSB2*CF(CSB2))*GHBB(2,2)
       CXT1=-AMZ**2/AMST(1)**2*CST1*(1-CST1*CF(CST1))*GHTT(1,1)
       CXT2=-AMZ**2/AMST(2)**2*CST2*(1-CST2*CF(CST2))*GHTT(2,2)
C
       CSUL = 4*AMSU(1)**2/AMH**2*DCMPLX(1D0,-EPS)
       CSUR = 4*AMSU(2)**2/AMH**2*DCMPLX(1D0,-EPS)
       CSDL = 4*AMSD(1)**2/AMH**2*DCMPLX(1D0,-EPS)
       CSDR = 4*AMSD(2)**2/AMH**2*DCMPLX(1D0,-EPS)
       CXUL=-2*(1.D0/2.D0-2.D0/3.D0*SS)*AMZ**2/AMSU(1)**2*DCOS(A+B)
     .      *CSUL*(1-CSUL*CF(CSUL))
       CXUR=-2*(2.D0/3.D0*SS)*AMZ**2/AMSU(2)**2*DCOS(A+B)
     .      *CSUR*(1-CSUR*CF(CSUR))
       CXDL=-2*(-1.D0/2.D0+1.D0/3.D0*SS)*AMZ**2/AMSD(1)**2*DCOS(A+B)
     .      *CSDL*(1-CSDL*CF(CSDL))
       CXDR=-2*(-1.D0/3.D0*SS)*AMZ**2/AMSD(2)**2*DCOS(A+B)
     .      *CSDR*(1-CSDR*CF(CSDR))
       ELSE
       CXB1=0.D0 
       CXB2=0.D0 
       CXT1=0.D0 
       CXT2=0.D0 
       CXUL=0.D0
       CXUR=0.D0
       CXDL=0.D0
       CXDR=0.D0
       ENDIF

       FQCD=HGGQCD(ASG,NFEXT)
       SQCD=SGGQCD(ASG)
       XFAC = CDABS(CAT+CAB+CAC+CXB1+CXB2+CXT1+CXT2
     .             +CXUL+CXUR+CXDL+CXDR)**2*FQCD
     .      + DREAL(DCONJG(CAT+CAB+CAC+CXB1+CXB2+CXT1+CXT2
     .                    +CXUL+CXUR+CXDL+CXDR)
     .             *(CXB1+CXB2+CXT1+CXT2+CXUL+CXUR+CXDL+CXDR))*SQCD
       HGG=HVV(AMH,0.D0)*(ASG/PI)**2*XFAC/8

c      write(6,*)'ghb, ght: ',ghb,ght

c      print*,''
c      print*,'H decay widths'
c      print*,'hgg_NLO',hgg

C  H ---> G G* ---> G CC   TO BE ADDED TO H ---> CC
       NFEXT = 4
       ASG = AS4
       FQCD=HGGQCD(ASG,NFEXT)
       SQCD=SGGQCD(ASG)
       XFAC = CDABS(CAT+CAB+CAC+CXB1+CXB2+CXT1+CXT2
     .             +CXUL+CXUR+CXDL+CXDR)**2*FQCD
     .      + DREAL(DCONJG(CAT+CAB+CAC+CXB1+CXB2+CXT1+CXT2
     .                    +CXUL+CXUR+CXDL+CXDR)
     .             *(CXB1+CXB2+CXT1+CXT2+CXUL+CXUR+CXDL+CXDR))*SQCD
       DCC=HVV(AMH,0.D0)*(ASG/PI)**2*XFAC/8 - HGG

C  H ---> G G* ---> G BB   TO BE ADDED TO H ---> BB
       NFEXT = 5
       ASG = ASH
       FQCD=HGGQCD(ASG,NFEXT)
       SQCD=SGGQCD(ASG)
       XFAC = CDABS(CAT+CAB+CAC+CXB1+CXB2+CXT1+CXT2
     .             +CXUL+CXUR+CXDL+CXDR)**2*FQCD
     .      + DREAL(DCONJG(CAT+CAB+CAC+CXB1+CXB2+CXT1+CXT2
     .                    +CXUL+CXUR+CXDL+CXDR)
     .             *(CXB1+CXB2+CXT1+CXT2+CXUL+CXUR+CXDL+CXDR))*SQCD
       DBB=HVV(AMH,0.D0)*(ASG/PI)**2*XFAC/8 - HGG - DCC
       HGG=HVV(AMH,0.D0)*(ASG/PI)**2*XFAC/8

C  H ---> G G: FULL NNNLO CORRECTIONS TO TOP LOOPS FOR NF=5
       FQCD0=HGGQCD(ASG,5)
       FQCD=HGGQCD2(ASG,5,AMH,AMT)
       XFAC = CDABS(CAT+CAB)**2*(FQCD-FQCD0)
       HGG=HGG+HVV(AMH,0.D0)*(ASG/PI)**2*XFAC/8

      IF(NFGG.EQ.3)THEN
       HGG = HGG - DBB - DCC
      ELSEIF(NFGG.EQ.4)THEN
       HGG = HGG - DBB
       DCC = 0
      ELSE
       DCC = 0
       DBB = 0
      ENDIF

c      print*,'hgg_NNLO',hgg

C  H ---> MU MU
      XGLM = GLB
      XGHM = GHB
      XGAM = GAB
      if(i2hdm.eq.1) then
         xghm = ghlep
      endif
      IF(IOFSUSY.EQ.0) THEN
       CALL STAUSUSY_HDEC(GLB,GHB,GAB,XGLM,XGHM,XGAM,QSUSY,0)
      ENDIF
      IF(AMH.LE.2*AMMUON) THEN
       HMM = 0
      ELSE
      HMM=HFF(AMH,(AMMUON/AMH)**2)*XGHM**2
      ENDIF

c      print*,'H -> mumu',hmm
C  H ---> LL
      XGLT = GLB
      XGHT = GHB
      XGAT = GAB
      if(i2hdm.eq.1) then
         xght = ghlep
      endif
      IF(IOFSUSY.EQ.0) THEN
       CALL STAUSUSY_HDEC(GLB,GHB,GAB,XGLT,XGHT,XGAT,QSUSY,1)
      ENDIF
      IF(AMH.LE.2*AMTAU) THEN
       HLL = 0
      ELSE
      HLL=HFF(AMH,(AMTAU/AMH)**2)*XGHT**2
      ENDIF

c     write(6,*)'H: tau/mu: ',HLL/HMM*AMMUON**2/AMTAU**2,XGHT**2/XGHM**2
c       print*,'H -> tautau',hll
C  H --> SS
      XGLS = GLB
      XGHS = GHB
      XGAS = GAB
      IF(IOFSUSY.EQ.0) THEN
       CALL STRSUSY_HDEC(GLB,GHB,GAB,XGLS,XGHS,XGAS,QSUSY,LOOP)
      ENDIF
      IF(AMH.LE.2*AMS) THEN
       HSS = 0
      ELSE
       HS1=3.D0*HFF(AMH,(AMS/AMH)**2)
     .    *XGHS**2
     .    *TQCDH(AMS**2/AMH**2)
       HS2=3.D0*HFF(AMH,(RMS/AMH)**2)*XGHS**2
     .    *QCDH(RMS**2/AMH**2)
       IF(HS2.LT.0.D0) HS2 = 0
       RAT = 2*AMS/AMH
       HSS = QQINT_HDEC(RAT,HS1,HS2)
      ENDIF

c      print*,'H -> ss',hss
C  H --> CC
      RATCOUP = 1
      IF(AMH.LE.2*AMC) THEN
       HCC = 0
      ELSE
       HC1=3.D0*HFF(AMH,(AMC/AMH)**2)
     .    *GHT**2
     .    *TQCDH(AMC**2/AMH**2)
       HC2=3.D0*HFF(AMH,(RMC/AMH)**2)*GHT**2
     .    *QCDH(RMC**2/AMH**2)
     .   + DCC
       IF(HC2.LT.0.D0) HC2 = 0
       RAT = 2*AMC/AMH
       HCC = QQINT_HDEC(RAT,HC1,HC2)
      ENDIF

c      print*,'H -> cc',hcc
C  H --> BB :
      QQ = AMB
      SUSY = 0
      XGHB = GHB
      SSUSY = (AMSB(1)+AMSB(2)+AMGLU)/3*QSUSY
      AS0 = ALPHAS_HDEC(SSUSY,3)
      IF(IOFSUSY.EQ.0) THEN
       I0 = 1
       CALL DMBAPP_HDEC(I0,DGLB,DGHB,DGAB,QSUSY,LOOP)
       I0 = 2
       BSC = (AMSQ+AMUR+AMDR)/3
c      XMB = AMB
       DELB0 = -DGAB/(1+1/TGBET**2)
       XMB = RUNM_HDEC(SUSYSCALE,5)/(1+DELB0)
       SUSY = COFSUSY_HDEC(I0,AMB,XMB,QQ)*AS0/PI - 2*DGHB
       CALL BOTSUSY_HDEC(GLB,GHB,GAB,XGLB,XGHB,XGAB,QSUSY,LOOP)
      ENDIF
      RATCOUP = GHT/XGHB
      IF(AMH.LE.2*AMB) THEN
       HBB = 0
      ELSE
       HB1=3.D0*HFF(AMH,(AMB/AMH)**2)
     .    *(XGHB**2+XGHB*GHB*SUSY)
     .    *TQCDH(AMB**2/AMH**2)
       HB2=3.D0*HFF(AMH,(RMB/AMH)**2)
     .    *(XGHB**2+XGHB*GHB*SUSY)
     .    *QCDH(RMB**2/AMH**2)
     .   + DBB
       IF(HB2.LT.0.D0) HB2 = 0
       RAT = 2*AMB/AMH
       HBB = QQINT_HDEC(RAT,HB1,HB2)

c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c      write(6,('A3,4(1X,G15.8)'))'H: ',AMA,AMH,SUSY+2*DGHB,
c    .                             SUSY/(SUSY+2*DGHB)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

      ENDIF

c      print*,'H -> bb',hbb
c      print*
c      print*,'H -> bb:',hbb,ghb,xghb,ghb*susy,2*dghb,susy,2*dghb+susy
C  H ---> TT
      RATCOUP = 0
      if(i2hdm.eq.0) then
      IF(IONSH.EQ.0)THEN
       DLD=3D0
       DLU=5D0
       XM1 = 2D0*AMT-DLD
       XM2 = 2D0*AMT+DLU
       IF (AMH.LE.AMT+AMW+AMB) THEN
        HTT=0.D0
       ELSEIF (AMH.LE.XM1) THEN
        FACTT=6.D0*GF**2*AMH**3*AMT**2/2.D0/128.D0/PI**3
        call HTOTT_hdec(amh,amt,amb,amw,amch,ght,ghb,gat,gab,
     .              ghvv,gzal,htt0)
        HTT=FACTT*HTT0
       ELSEIF (AMH.LE.XM2) THEN
        ZZMA=AMAR
        XX(1) = XM1-1D0
        XX(2) = XM1
        XX(3) = XM2
        XX(4) = XM2+1D0
        CALL AMHAMA_HDEC(2,XX(1),TGBET)
        FACTT=6.D0*GF**2*XX(1)**3*AMT**2/2.D0/128.D0/PI**3
        call HTOTT_hdec(xx(1),amt,amb,amw,amch,ght,ghb,gat,gab,
     .              ghvv,gzal,htt0)
        YY(1)=FACTT*HTT0
        CALL AMHAMA_HDEC(2,XX(2),TGBET)
        FACTT=6.D0*GF**2*XX(2)**3*AMT**2/2.D0/128.D0/PI**3
        call HTOTT_hdec(xx(2),amt,amb,amw,amch,ght,ghb,gat,gab,
     .              ghvv,gzal,htt0)
        YY(2)=FACTT*HTT0
        CALL AMHAMA_HDEC(2,XX(3),TGBET)
        XMT = RUNM_HDEC(XX(3),6)
        HT1=3.D0*HFF(XX(3),(AMT/XX(3))**2)*GHT**2
     .    *TQCDH(AMT**2/XX(3)**2)
        HT2=3.D0*HFF(XX(3),(XMT/XX(3))**2)*GHT**2
     .    *QCDH(XMT**2/XX(3)**2)
        IF(HT2.LT.0.D0) HT2 = 0
        RAT = 2*AMT/XX(3)
        YY(3) = QQINT_HDEC(RAT,HT1,HT2)
        CALL AMHAMA_HDEC(2,XX(4),TGBET)
        XMT = RUNM_HDEC(XX(4),6)
        HT1=3.D0*HFF(XX(4),(AMT/XX(4))**2)*GHT**2
     .    *TQCDH(AMT**2/XX(4)**2)
        HT2=3.D0*HFF(XX(4),(XMT/XX(4))**2)*GHT**2
     .    *QCDH(XMT**2/XX(4)**2)
        IF(HT2.LT.0.D0) HT2 = 0
        RAT = 2*AMT/XX(4)
        YY(4) = QQINT_HDEC(RAT,HT1,HT2)
        AMA = ZZMA
        CALL SUSYCP_HDEC(TGBET)
        HTT=FINT_HDEC(AMH,XX,YY)
       ELSE
        HT1=3.D0*HFF(AMH,(AMT/AMH)**2)*GHT**2
     .    *TQCDH(AMT**2/AMH**2)
        HT2=3.D0*HFF(AMH,(RMT/AMH)**2)*GHT**2
     .    *QCDH(RMT**2/AMH**2)
        IF(HT2.LT.0.D0) HT2 = 0
        RAT = 2*AMT/AMH
        HTT = QQINT_HDEC(RAT,HT1,HT2)
       ENDIF
      ELSE
       IF (AMH.LE.2.D0*AMT) THEN
        HTT=0.D0
       ELSE
        HT1=3.D0*HFF(AMH,(AMT/AMH)**2)*GHT**2
     .    *TQCDH(AMT**2/AMH**2)
        HT2=3.D0*HFF(AMH,(RMT/AMH)**2)*GHT**2
     .    *QCDH(RMT**2/AMH**2)
        IF(HT2.LT.0.D0) HT2 = 0
        RAT = 2*AMT/AMH
        HTT = QQINT_HDEC(RAT,HT1,HT2)
       ENDIF
      ENDIF
      endif

c MMM changed 21/8/2013
      if(i2hdm.eq.1) then
         if(ionsh.eq.0)then
            dld=5.D0
            dlu=3.D0
            xm1 = 2d0*amt-dld
            xm2 = 2d0*amt+dlu
            if (amh.le.amt+amw+amb) then
               htt=0.d0
            elseif (amh.le.xm1) then
               factt=6.d0*gf**2*amh**3*amt**2/2.d0/128.d0/pi**3
               call HTOTT_hdec(amh,amt,amb,amw,amch,ght,ghb,gat,gab,
     .              ghvv,gzal,htt0)
               htt=factt*htt0
            elseif (amh.le.xm2) then
               XX(1) = XM1-1D0
               XX(2) = XM1
               XX(3) = XM2
               XX(4) = XM2+1D0

               factt=6.d0*gf**2*xx(1)**3*amt**2/2.d0/128.d0/pi**3
               call HTOTT_hdec(xx(1),amt,amb,amw,amch,ght,ghb,gat,gab,
     .              ghvv,gzal,htt0)
               yy(1)=factt*htt0

               factt=6.d0*gf**2*xx(2)**3*amt**2/2.d0/128.d0/pi**3
               call HTOTT_hdec(xx(2),amt,amb,amw,amch,ght,ghb,gat,gab,
     .              ghvv,gzal,htt0)
               yy(2)=factt*htt0

               xmt = runm_hdec(xx(3),6)
               ht1=3.d0*hff(xx(3),(amt/xx(3))**2)*ght**2
     .              *tqcdh(amt**2/xx(3)**2)
               ht2=3.d0*hff(xx(3),(xmt/xx(3))**2)*ght**2
     .              *qcdh(xmt**2/xx(3)**2)
               if(ht2.lt.0.d0) ht2 = 0
               rat = 2*amt/xx(3)
               yy(3) = qqint_hdec(rat,ht1,ht2)

               xmt = runm_hdec(xx(4),6)
               ht1=3.d0*hff(xx(4),(amt/xx(4))**2)*ght**2
     .              *tqcdh(amt**2/xx(4)**2)
               ht2=3.d0*hff(xx(4),(xmt/xx(4))**2)*ght**2
     .              *qcdh(xmt**2/xx(4)**2)
               if(ht2.lt.0.d0) ht2 = 0
               rat = 2*amt/xx(4)
               yy(4) = qqint_hdec(rat,ht1,ht2)

               htt=fint_hdec(amh,xx,yy)
            else
               ht1=3.d0*hff(amh,(amt/amh)**2)*ght**2
     .              *tqcdh(amt**2/amh**2)
               ht2=3.d0*hff(amh,(rmt/amh)**2)*ght**2
     .              *qcdh(rmt**2/amh**2)
               if(ht2.lt.0.d0) ht2 = 0
               rat = 2.D0*amt/amh
               htt = qqint_hdec(rat,ht1,ht2)
            endif
         else
            if (amh.le.2.d0*amt) then
               htt=0.d0
            else
               ht1=3.d0*hff(amh,(amt/amh)**2)*ght**2
     .              *tqcdh(amt**2/amh**2)
               ht2=3.d0*hff(amh,(rmt/amh)**2)*ght**2
     .              *qcdh(rmt**2/amh**2)
               if(ht2.lt.0.d0) ht2 = 0
               rat = 2.D0*amt/amh
               htt = qqint_hdec(rat,ht1,ht2)
            endif
         endif
      endif
c end MMM changed 21/8/2103

c      print*,'H -> tt',htt

C  H ---> GAMMA GAMMA
       EPS=1.D-8
       XRMC = RUNM_HDEC(AMH/2,4)*AMC/RUNM_HDEC(AMC,4)
       XRMB = RUNM_HDEC(AMH/2,5)*AMB/RUNM_HDEC(AMB,5)
       XRMT = RUNM_HDEC(AMH/2,6)*AMT/RUNM_HDEC(AMT,6)
       CTT = 4*XRMT**2/AMH**2*DCMPLX(1D0,-EPS)
       CTB = 4*XRMB**2/AMH**2*DCMPLX(1D0,-EPS)
       CTL = 4*AMTAU**2/AMH**2*DCMPLX(1D0,-EPS)
       CTW = 4*AMW**2/AMH**2*DCMPLX(1D0,-EPS)
       CTH = 4*AMCH**2/AMH**2*DCMPLX(1D0,-EPS)
       CTC = 4*XRMC**2/AMH**2*DCMPLX(1D0,-EPS)
       CAC = 4/3D0 * 2*CTC*(1+(1-CTC)*CF(CTC))*GHT
     .     * CFACQ_HDEC(0,AMH,XRMC)
       CAT = 4/3D0 * 2*CTT*(1+(1-CTT)*CF(CTT))*GHT
     .     * CFACQ_HDEC(0,AMH,XRMT)
       CAB = 1/3D0 * 2*CTB*(1+(1-CTB)*CF(CTB))*GHB
     .     * CFACQ_HDEC(0,AMH,XRMB)
       CAL = 1.D0  * 2*CTL*(1+(1-CTL)*CF(CTL))*GHB
       if(i2hdm.eq.1) then
          CAL = 1.D0  * 2*CTL*(1+(1-CTL)*CF(CTL))*ghlep
       endif
       CAW = -(2+3*CTW+3*CTW*(2-CTW)*CF(CTW))*GHVV
       CAH = -AMZ**2/2/AMCH**2*CTH*(1-CTH*CF(CTH))*GHPM
       IF(IOFSUSY.EQ.0) THEN 
        RMSU1 = RUNMS_HDEC(AMH/2,AMSU(1))
        RMSU2 = RUNMS_HDEC(AMH/2,AMSU(2))
        RMSD1 = RUNMS_HDEC(AMH/2,AMSD(1))
        RMSD2 = RUNMS_HDEC(AMH/2,AMSD(2))
        RMSB1 = RUNMS_HDEC(AMH/2,AMSB(1))
        RMSB2 = RUNMS_HDEC(AMH/2,AMSB(2))
        RMST1 = RUNMS_HDEC(AMH/2,AMST(1))
        RMST2 = RUNMS_HDEC(AMH/2,AMST(2))
        CX1 = 4*AMCHAR(1)**2/AMH**2*DCMPLX(1D0,-EPS)
        CX2 = 4*AMCHAR(2)**2/AMH**2*DCMPLX(1D0,-EPS)
        CAX1= AMW/XMCHAR(1) * 2*CX1*(1+(1-CX1)*CF(CX1))*2*AC1(1,1) 
        CAX2= AMW/XMCHAR(2) * 2*CX2*(1+(1-CX2)*CF(CX2))*2*AC1(2,2) 
        CSL1= 4*AMSL(1)**2/AMH**2*DCMPLX(1D0,-EPS)
        CSL2= 4*AMSL(2)**2/AMH**2*DCMPLX(1D0,-EPS)
        CSB1= 4*RMSB1**2/AMH**2*DCMPLX(1D0,-EPS)
        CSB2= 4*RMSB2**2/AMH**2*DCMPLX(1D0,-EPS)
        CST1= 4*RMST1**2/AMH**2*DCMPLX(1D0,-EPS)
        CST2= 4*RMST2**2/AMH**2*DCMPLX(1D0,-EPS)

        CSEL = 4*AMSE(1)**2/AMH**2*DCMPLX(1D0,-EPS)
        CSER = 4*AMSE(2)**2/AMH**2*DCMPLX(1D0,-EPS)
        CSUL = 4*RMSU1**2/AMH**2*DCMPLX(1D0,-EPS)
        CSUR = 4*RMSU2**2/AMH**2*DCMPLX(1D0,-EPS)
        CSDL = 4*RMSD1**2/AMH**2*DCMPLX(1D0,-EPS)
        CSDR = 4*RMSD2**2/AMH**2*DCMPLX(1D0,-EPS)
        CXEL=-2*(-1/2D0+SS)*AMZ**2/AMSE(1)**2*DCOS(A+B)
     .       *CSEL*(1-CSEL*CF(CSEL))
        CXER=2*(SS)*AMZ**2/AMSE(2)**2*DCOS(A+B)
     .       *CSER*(1-CSER*CF(CSER))
        CXUL=-2*4.D0/3.D0*(1.D0/2.D0-2.D0/3.D0*SS)
     .       *AMZ**2/AMSU(1)**2*DCOS(A+B)*CSUL*(1-CSUL*CF(CSUL))
     .      * CFACSQ_HDEC(AMH,RMSU1)
        CXUR=-2*4.D0/3.D0*(2.D0/3.D0*SS)
     .       *AMZ**2/AMSU(2)**2*DCOS(A+B)*CSUR*(1-CSUR*CF(CSUR))
     .      * CFACSQ_HDEC(AMH,RMSU2)
        CXDL=-2/3.D0*(-1.D0/2.D0+1.D0/3.D0*SS)
     .       *AMZ**2/AMSD(1)**2*DCOS(A+B)*CSDL*(1-CSDL*CF(CSDL))
     .      * CFACSQ_HDEC(AMH,RMSD1)
        CXDR=-2/3.D0*(-1.D0/3.D0*SS)
     .       *AMZ**2/AMSD(2)**2*DCOS(A+B)*CSDR*(1-CSDR*CF(CSDR))
     .      * CFACSQ_HDEC(AMH,RMSD2)

        CXB1= -1/3D0*AMZ**2/AMSB(1)**2*CSB1*(1-CSB1*CF(CSB1))*GHBB(1,1)
     .      * CFACSQ_HDEC(AMH,RMSB1)
        CXB2= -1/3D0*AMZ**2/AMSB(2)**2*CSB2*(1-CSB2*CF(CSB2))*GHBB(2,2)
     .      * CFACSQ_HDEC(AMH,RMSB2)
        CXT1= -4/3D0*AMZ**2/AMST(1)**2*CST1*(1-CST1*CF(CST1))*GHTT(1,1)
     .      * CFACSQ_HDEC(AMH,RMST1)
        CXT2= -4/3D0*AMZ**2/AMST(2)**2*CST2*(1-CST2*CF(CST2))*GHTT(2,2)
     .      * CFACSQ_HDEC(AMH,RMST2)
        CXL1=       -AMZ**2/AMSL(1)**2*CSL1*(1-CSL1*CF(CSL1))*GHEE(1,1)
        CXL2=       -AMZ**2/AMSL(2)**2*CSL2*(1-CSL2*CF(CSL2))*GHEE(2,2)
        XFAC = CDABS(CAT+CAB+CAC+CAL+CAW+CAH+CAX1+CAX2
     .       +  CXEL+CXER+CXUL+CXUR+CXDL+CXDR
     .       +  CXB1+CXB2+CXT1+CXT2+CXL1+CXL2)**2
       ELSE 
        XFAC = CDABS(CAT+CAB+CAC+CAL+CAW+CAH)**2
       ENDIF
       HGA=HVV(AMH,0.D0)*(ALPH/PI)**2/16.D0*XFAC

c       print*,'H -> gamgam',hga
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
       XFACQ = CDABS(CAT+CAB+CAC+CAL+CAW+CAH)**2
       XFACS = CDABS(CAT+CAB+CAC+CAL+CAW+CAH+CAX1+CAX2
     .      +  CXL1+CXL2)**2
       XFACSQ = CDABS(CAT+CAB+CAC+CAL+CAW+CAH+CAX1+CAX2
     .      +  CXB1+CXB2+CXT1+CXT2+CXL1+CXL2)**2
       HGA0 = HGA*XFACSQ/XFAC
       CAC0 = 4/3D0 * 2*CTC*(1+(1-CTC)*CF(CTC))*GHT
       CAT0 = 4/3D0 * 2*CTT*(1+(1-CTT)*CF(CTT))*GHT
       CAB0 = 1/3D0 * 2*CTB*(1+(1-CTB)*CF(CTB))*GHB
       CXB10= -1/3D0*AMZ**2/AMSB(1)**2*CSB1*(1-CSB1*CF(CSB1))*GHBB(1,1)
       CXB20= -1/3D0*AMZ**2/AMSB(2)**2*CSB2*(1-CSB2*CF(CSB2))*GHBB(2,2)
       CXT10= -4/3D0*AMZ**2/AMST(1)**2*CST1*(1-CST1*CF(CST1))*GHTT(1,1)
       CXT20= -4/3D0*AMZ**2/AMST(2)**2*CST2*(1-CST2*CF(CST2))*GHTT(2,2)
       XFACLOQ = CDABS(CAT0+CAB0+CAC0+CAL+CAW+CAH)**2
       CXUL0=-2*4.D0/3.D0*(1.D0/2.D0-2.D0/3.D0*SS)
     .      *AMZ**2/AMSU(1)**2*DCOS(A+B)*CSUL*(1-CSUL*CF(CSUL))
       CXUR0=-2*4.D0/3.D0*(2.D0/3.D0*SS)
     .      *AMZ**2/AMSU(2)**2*DCOS(A+B)*CSUR*(1-CSUR*CF(CSUR))
       CXDL0=-2/3.D0*(-1.D0/2.D0+1.D0/3.D0*SS)
     .      *AMZ**2/AMSD(1)**2*DCOS(A+B)*CSDL*(1-CSDL*CF(CSDL))
       CXDR0=-2/3.D0*(-1.D0/3.D0*SS)
     .      *AMZ**2/AMSD(2)**2*DCOS(A+B)*CSDR*(1-CSDR*CF(CSDR))
       XFACLO = CDABS(CAT0+CAB0+CAC0+CAL+CAW+CAH+CAX1+CAX2
     .      +  CXEL+CXER+CXUL0+CXUR0+CXDL0+CXDR0
     .      +  CXB10+CXB20+CXT10+CXT20+CXL1+CXL2)**2
       CSQ = 1+3*ALPHAS_HDEC(AMH,3)
       XFACSQL = CDABS(CAT+CAB+CAC+CAL+CAW+CAH+CAX1+CAX2
     .      +  CXEL+CXER+(CXUL0+CXUR0+CXDL0+CXDR0
     .      +  CXB10+CXB20+CXT10+CXT20)*CSQ+CXL1+CXL2)**2
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
C  H ---> Z GAMMA
      XRMC = RUNM_HDEC(AMH/2,4)*AMC/RUNM_HDEC(AMC,4)
      XRMB = RUNM_HDEC(AMH/2,5)*AMB/RUNM_HDEC(AMB,5)
      XRMT = RUNM_HDEC(AMH/2,6)*AMT/RUNM_HDEC(AMT,6)
c     print*,'xrmc,xrmb,xrmt ',xrmc,xrmb,xrmt
      IF(AMH.LE.AMZ)THEN
       HZGA=0
      ELSE
       TS = SS/CS
       FT = -3*2D0/3*(1-4*2D0/3*SS)/DSQRT(SS*CS)*GHT
       FB = 3*1D0/3*(-1+4*1D0/3*SS)/DSQRT(SS*CS)*GHB
       FC = -3*2D0/3*(1-4*2D0/3*SS)/DSQRT(SS*CS)*GHT
       FL = (-1+4*SS)/DSQRT(SS*CS)*GHB
       if(i2hdm.eq.1) then
          FL = (-1+4*SS)/DSQRT(SS*CS)*ghlep
       endif
       EPS=1.D-8
c      CTT = 4*XRMT**2/AMH**2*DCMPLX(1D0,-EPS)
c      CTB = 4*XRMB**2/AMH**2*DCMPLX(1D0,-EPS)
c      CTC = 4*XRMC**2/AMH**2*DCMPLX(1D0,-EPS)
       CTT = 4*AMT**2/AMH**2*DCMPLX(1D0,-EPS)
       CTB = 4*AMB**2/AMH**2*DCMPLX(1D0,-EPS)
       CTC = 4*AMC**2/AMH**2*DCMPLX(1D0,-EPS)
       CTL = 4*AMTAU**2/AMH**2*DCMPLX(1D0,-EPS)
       CTW = 4*AMW**2/AMH**2*DCMPLX(1D0,-EPS)
       CTH = 4*AMCH**2/AMH**2*DCMPLX(1D0,-EPS)
c      CLT = 4*XRMT**2/AMZ**2*DCMPLX(1D0,-EPS)
c      CLB = 4*XRMB**2/AMZ**2*DCMPLX(1D0,-EPS)
c      CLC = 4*XRMC**2/AMZ**2*DCMPLX(1D0,-EPS)
       CLT = 4*AMT**2/AMZ**2*DCMPLX(1D0,-EPS)
       CLB = 4*AMB**2/AMZ**2*DCMPLX(1D0,-EPS)
       CLC = 4*AMC**2/AMZ**2*DCMPLX(1D0,-EPS)
       CLE = 4*AMTAU**2/AMZ**2*DCMPLX(1D0,-EPS)
       CLW = 4*AMW**2/AMZ**2*DCMPLX(1D0,-EPS)
       CLH = 4*AMCH**2/AMZ**2*DCMPLX(1D0,-EPS)
       CAT = FT*(CI1(CTT,CLT) - CI2(CTT,CLT))
       CAB = FB*(CI1(CTB,CLB) - CI2(CTB,CLB))
       CAC = FC*(CI1(CTC,CLC) - CI2(CTC,CLC))
       CAL = FL*(CI1(CTL,CLE) - CI2(CTL,CLE))
       CAW = -1/DSQRT(TS)*(4*(3-TS)*CI2(CTW,CLW)
     .     + ((1+2/CTW)*TS - (5+2/CTW))*CI1(CTW,CLW))*GHVV
       CAH = (1-2*SS)/DSQRT(SS*CS)*AMZ**2/2/AMCH**2*CI1(CTH,CLH)*GHPM
       XFAC = CDABS(CAT+CAB+CAC+CAL+CAW+CAH)**2
       ACOUP = DSQRT(2D0)*GF*AMZ**2*SS*CS/PI**2
       HZGA = GF/(4.D0*PI*DSQRT(2.D0))*AMH**3*(ALPH/PI)*ACOUP/16.D0
     .        *XFAC*(1-AMZ**2/AMH**2)**3
      ENDIF

c      print*,'H -> Zgam',hzga
C  H ---> W W
      IF(IONWZ.EQ.0)THEN
       CALL HTOVV_HDEC(0,AMH,AMW,GAMW,HTWW)
       HWW = 3D0/2D0*GF*AMW**4/DSQRT(2D0)/PI/AMH**3*HTWW*GHVV**2
      ELSEIF(IONWZ.EQ.-1)THEN
       DLD=2D0
       DLU=2D0
       XM1 = 2D0*AMW-DLD
       XM2 = 2D0*AMW+DLU
       IF (AMH.LE.XM1) THEN
        CALL HTOVV_HDEC(0,AMH,AMW,GAMW,HTWW)
        HWW = 3D0/2D0*GF*AMW**4/DSQRT(2D0)/PI/AMH**3*HTWW*GHVV**2
       ELSEIF (AMH.LE.XM2) THEN
        XX(1) = XM1-1D0
        XX(2) = XM1
        XX(3) = XM2
        XX(4) = XM2+1D0
        CALL HTOVV_HDEC(0,XX(1),AMW,GAMW,HTWW)
        YY(1)=3D0/2D0*GF*AMW**4/DSQRT(2D0)/PI/XX(1)**3*HTWW
        CALL HTOVV_HDEC(0,XX(2),AMW,GAMW,HTWW)
        YY(2)=3D0/2D0*GF*AMW**4/DSQRT(2D0)/PI/XX(2)**3*HTWW
        YY(3)=HVV(XX(3),AMW**2/XX(3)**2)
        YY(4)=HVV(XX(4),AMW**2/XX(4)**2)
        HWW = FINT_HDEC(AMH,XX,YY)*GHVV**2
       ELSE
        HWW=HVV(AMH,AMW**2/AMH**2)*GHVV**2
       ENDIF
      ELSE
      DLD=2D0
      DLU=2D0
      XM1 = 2D0*AMW-DLD
      XM2 = 2D0*AMW+DLU
      IF (AMH.LE.AMW) THEN
       HWW=0
      ELSE IF (AMH.LE.XM1) THEN
       CWW=3.D0*GF**2*AMW**4/16.D0/PI**3
       HWW=HV(AMW**2/AMH**2)*CWW*AMH*GHVV**2
      ELSE IF (AMH.LT.XM2) THEN
       CWW=3.D0*GF**2*AMW**4/16.D0/PI**3
       XX(1) = XM1-1D0
       XX(2) = XM1
       XX(3) = XM2
       XX(4) = XM2+1D0
       YY(1)=HV(AMW**2/XX(1)**2)*CWW*XX(1)
       YY(2)=HV(AMW**2/XX(2)**2)*CWW*XX(2)
       YY(3)=HVV(XX(3),AMW**2/XX(3)**2)
       YY(4)=HVV(XX(4),AMW**2/XX(4)**2)
       HWW = FINT_HDEC(AMH,XX,YY)*GHVV**2
      ELSE
       HWW=HVV(AMH,AMW**2/AMH**2)*GHVV**2
      ENDIF
      ENDIF

c      print*,'H -> WW',hww
C  H ---> Z Z
      IF(IONWZ.EQ.0)THEN
       CALL HTOVV_HDEC(0,AMH,AMZ,GAMZ,HTZZ)
       HZZ = 3D0/4D0*GF*AMZ**4/DSQRT(2D0)/PI/AMH**3*HTZZ*GHVV**2
      ELSEIF(IONWZ.EQ.-1)THEN
       DLD=2D0
       DLU=2D0
       XM1 = 2D0*AMZ-DLD
       XM2 = 2D0*AMZ+DLU
       IF (AMH.LE.XM1) THEN
        CALL HTOVV_HDEC(0,AMH,AMZ,GAMZ,HTZZ)
        HZZ = 3D0/4D0*GF*AMZ**4/DSQRT(2D0)/PI/AMH**3*HTZZ*GHVV**2
       ELSEIF (AMH.LE.XM2) THEN
        XX(1) = XM1-1D0
        XX(2) = XM1
        XX(3) = XM2
        XX(4) = XM2+1D0
        CALL HTOVV_HDEC(0,XX(1),AMZ,GAMZ,HTZZ)
        YY(1)=3D0/4D0*GF*AMZ**4/DSQRT(2D0)/PI/XX(1)**3*HTZZ
        CALL HTOVV_HDEC(0,XX(2),AMZ,GAMZ,HTZZ)
        YY(2)=3D0/4D0*GF*AMZ**4/DSQRT(2D0)/PI/XX(2)**3*HTZZ
        YY(3)=HVV(XX(3),AMZ**2/XX(3)**2)/2
        YY(4)=HVV(XX(4),AMZ**2/XX(4)**2)/2
        HZZ = FINT_HDEC(AMH,XX,YY)*GHVV**2
       ELSE
        HZZ=HVV(AMH,AMZ**2/AMH**2)/2.D0*GHVV**2
       ENDIF
      ELSE
      DLD=2D0
      DLU=2D0
      XM1 = 2D0*AMZ-DLD
      XM2 = 2D0*AMZ+DLU
      IF (AMH.LE.AMZ) THEN
       HZZ=0
      ELSE IF (AMH.LE.XM1) THEN
       CZZ=3.D0*GF**2*AMZ**4/192.D0/PI**3*(7-40/3.D0*SS+160/9.D0*SS**2)
       HZZ=HV(AMZ**2/AMH**2)*CZZ*AMH*GHVV**2
      ELSE IF (AMH.LT.XM2) THEN
       CZZ=3.D0*GF**2*AMZ**4/192.D0/PI**3*(7-40/3.D0*SS+160/9.D0*SS**2)
       XX(1) = XM1-1D0
       XX(2) = XM1
       XX(3) = XM2
       XX(4) = XM2+1D0
       YY(1)=HV(AMZ**2/XX(1)**2)*CZZ*XX(1)
       YY(2)=HV(AMZ**2/XX(2)**2)*CZZ*XX(2)
       YY(3)=HVV(XX(3),AMZ**2/XX(3)**2)/2D0
       YY(4)=HVV(XX(4),AMZ**2/XX(4)**2)/2D0
       HZZ = FINT_HDEC(AMH,XX,YY)*GHVV**2
      ELSE
       HZZ=HVV(AMH,AMZ**2/AMH**2)/2.D0*GHVV**2
      ENDIF
      ENDIF

c      print*,'H -> ZZ',hzz
C  H ---> h h
      if(i2hdm.eq.0) then
      IF(IONSH.EQ.0)THEN
      if(islhai.eq.0)then
       ZZMA = AMAR
       AMREAL = AMH
       AMA = 1.D0
       AMLOW = AMH
12345  CALL SUSYCP_HDEC(TGBET)
       IF(AMLR.LT.0.D0)THEN
        AMA = AMAR + 1
        GOTO 12345
       ENDIF
       AMLOW = AMH
       AMDEL = AMREAL - AMLOW
       DLD = 0.3D0*(TGBET-1.3D0)
       DLD = DMAX1(0.1D0,DLD)
       DLU=DLD
       AMA = ZZMA
       CALL SUSYCP_HDEC(TGBET)
       XM1 = 2*AML-DLD
       XM2 = 2*AML+DLU
       IF (AMH.LE.AML) THEN
        HHH=0
       ELSEIF (AMH.LT.XM1) THEN
        XH=AML**2/AMH**2
        XH1=(XH-1.D0)*(2.D0-.5D0*DLOG(XH))+(1.D0-5.D0*XH)
     .    *(DATAN((2.D0*XH-1.D0)/DSQRT(4.D0*XH-1.D0))
     .     -DATAN(1.D0/DSQRT(4.D0*XH-1.D0)))/DSQRT(4.D0*XH-1.D0)
        XH2=3*GF**2/32.D0/PI**3*AMZ**4/AMH*GHLL**2*GLB**2*AMB**2
        HHH=XH1*XH2
       ELSEIF (AMH.LT.XM2) THEN
        IFLON0 = 0
        IFLON1 = 0
        ZZMA=AMAR
        AMACRIT = AMAR
        AMA0 = AMAR
        AMA1 = AMAR
510     AMA0 = AMA0 - 1
        AMA1 = AMA1 + 1
        AMA = AMA0
        CALL SUSYCP_HDEC(TGBET)
        IF(AMH.LT.2*AML) THEN
         IFLON0 = -1
        ELSE
         IFLON0 = 1
        ENDIF
        AMA = AMA1
        CALL SUSYCP_HDEC(TGBET)
        IF(AMH.LT.2*AML) THEN
         IFLON1 = -1
        ELSE
         IFLON1 = 1
        ENDIF
        IF(IFLON0*IFLON1.NE.-1) GOTO 510
501     AMA = (AMA0+AMA1)/2
        CALL SUSYCP_HDEC(TGBET)
        IF(AMH.LT.2*AML) THEN
         IF(IFLON0.EQ.-1) THEN
          AMA0 = AMAR
         ELSE
          AMA1 = AMAR
         ENDIF
        ELSE
         IF(IFLON0.EQ.-1) THEN
          AMA1 = AMAR
         ELSE
          AMA0 = AMAR
         ENDIF
        ENDIF
        AMACRIT = (AMA0+AMA1)/2
        DEL = 1.D-8
        AMDEL = 2*DABS(AMA1-AMA0)/(AMA1+AMA0)
        IF(AMDEL.GT.DEL) GOTO 501
       AMA = AMACRIT
       CALL SUSYCP_HDEC(TGBET)
       YM1 = AMACRIT
       YM2 = AMACRIT
       AMA0 = AMACRIT
       AMA1 = AMACRIT
       DELSTEP = 1.D0
511    AMA0 = AMA0 - DELSTEP
       AMA1 = AMA1 + DELSTEP
       AMA = AMACRIT
       CALL SUSYCP_HDEC(TGBET)
       IF(AMH.LT.2*AML-DLD) THEN
        IFLONC = -1
       ELSE
        IFLONC = 1
       ENDIF
       AMA = AMA0
       CALL SUSYCP_HDEC(TGBET)
       IF(AMH.LT.2*AML-DLD) THEN
        IFLON0 = -1
       ELSE
        IFLON0 = 1
       ENDIF
       AMA = AMA1
       CALL SUSYCP_HDEC(TGBET)
       IF(AMH.LT.2*AML-DLD) THEN
        IFLON1 = -1
       ELSE
        IFLON1 = 1
       ENDIF
       IF(IFLON0*IFLONC.NE.-1.AND.IFLONC*IFLON1.NE.-1) GOTO 511
       IF(IFLON0*IFLONC.EQ.-1) THEN
         AMA1 = AMACRIT
         IFLON1 = IFLONC
       ELSE
         AMA0 = AMACRIT
         IFLON0 = IFLONC
       ENDIF
512    AMA = (AMA0+AMA1)/2
       CALL SUSYCP_HDEC(TGBET)
       IF(AMH.LT.2*AML-DLD) THEN
        IF(IFLON0.EQ.-1) THEN
         AMA0 = AMAR
        ELSE
         AMA1 = AMAR
        ENDIF
       ELSE
        IF(IFLON0.EQ.-1) THEN
         AMA1 = AMAR
        ELSE
         AMA0 = AMAR
        ENDIF
       ENDIF
       YM1 = (AMA0+AMA1)/2
       DEL = 1.D-8
       AMDEL = 2*DABS(AMA1-AMA0)/(AMA1+AMA0)
       IF(AMDEL.GT.DEL) GOTO 512
       AMA = YM1
       CALL SUSYCP_HDEC(TGBET)
       AMA0 = AMACRIT
       AMA1 = AMACRIT
       DELSTEP = 1.D0
513    AMA0 = AMA0 - DELSTEP
       AMA1 = AMA1 + DELSTEP
       AMA = AMACRIT
       CALL SUSYCP_HDEC(TGBET)
       IF(AMH.LT.2*AML+DLU) THEN
        IFLONC = -1
       ELSE
        IFLONC = 1
       ENDIF
       AMA = AMA0
       CALL SUSYCP_HDEC(TGBET)
       IF(AMH.LT.2*AML+DLU) THEN
        IFLON0 = -1
       ELSE
        IFLON0 = 1
       ENDIF
       AMA = AMA1
       CALL SUSYCP_HDEC(TGBET)
       IF(AMH.LT.2*AML+DLU) THEN
        IFLON1 = -1
       ELSE
        IFLON1 = 1
       ENDIF
       IF(IFLON0*IFLONC.NE.-1.AND.IFLONC*IFLON1.NE.-1) GOTO 513
       IF(IFLON0*IFLONC.EQ.-1) THEN
         AMA1 = AMACRIT
         IFLON1 = IFLONC
       ELSE
         AMA0 = AMACRIT
         IFLON0 = IFLONC
       ENDIF
514    AMA = (AMA0+AMA1)/2
       CALL SUSYCP_HDEC(TGBET)
       IF(AMH.LT.2*AML+DLU) THEN
        IF(IFLON0.EQ.-1) THEN
         AMA0 = AMAR
        ELSE
         AMA1 = AMAR
        ENDIF
       ELSE
        IF(IFLON0.EQ.-1) THEN
         AMA1 = AMAR
        ELSE
         AMA0 = AMAR
        ENDIF
       ENDIF
       YM2 = (AMA0+AMA1)/2
       DEL = 1.D-8
       AMDEL = 2*DABS(AMA1-AMA0)/(AMA1+AMA0)
       IF(AMDEL.GT.DEL) GOTO 514
       AMA = YM2
       CALL SUSYCP_HDEC(TGBET)
       DEL = 1.D-4
        XX(1) = YM1 - DEL
        XX(2) = YM1
        XX(3) = YM2
        XX(4) = YM2 + DEL
        AMAR = ZZMA
        DO J=1,4
         AMA = XX(J)
         CALL SUSYCP_HDEC(TGBET)
         XX(J) = AMH
         IF(AMH.GE.2*AML)THEN
          YY(J)=GF/16D0/DSQRT(2D0)/PI*AMZ**4/XX(J)
     .          *BETA_HDEC(AML**2/XX(J)**2)
         ELSEIF(AMH.LE.AML)THEN
          YY(J) = 0
         ELSE
          XH=AML**2/XX(J)**2
          XH1=(XH-1.D0)*(2.D0-.5D0*DLOG(XH))+(1.D0-5.D0*XH)
     .    *(DATAN((2.D0*XH-1.D0)/DSQRT(4.D0*XH-1.D0))
     .     -DATAN(1.D0/DSQRT(4.D0*XH-1.D0)))/DSQRT(4.D0*XH-1.D0)
          XH2=3*GF**2/32.D0/PI**3*AMZ**4/XX(J)*GLB**2*AMB**2
          YY(J)=XH1*XH2
         ENDIF
        ENDDO
        AMA = ZZMA
        CALL SUSYCP_HDEC(TGBET)
        HHH = FINT_HDEC(AMH,XX,YY)*GHLL**2
       ELSE
        HHH=GF/16D0/DSQRT(2D0)/PI*AMZ**4/AMH*BETA_HDEC(AML**2/AMH**2)
     .      *GHLL**2
       ENDIF
      else
       DLD=0.1D0
       DLU=0.1D0
       XM1 = 2D0*AML-DLD
       XM2 = 2D0*AML+DLU
       IF (AMH.LE.AML) THEN
        HHH = 0D0
       ELSEIF (AMH.LE.XM1) THEN
        XH=AML**2/AMH**2
        XH1=(XH-1.D0)*(2.D0-.5D0*DLOG(XH))+(1.D0-5.D0*XH)
     .  *(DATAN((2.D0*XH-1.D0)/DSQRT(4.D0*XH-1.D0))
     .   -DATAN(1.D0/DSQRT(4.D0*XH-1.D0)))/DSQRT(4.D0*XH-1.D0)
        XH2=3*GF**2/32.D0/PI**3*AMZ**4/AMH*GLB**2*AMB**2
        HHH=XH1*XH2
       ELSEIF (AMH.LE.XM2) THEN
        XX(1) = XM1-1D0
        XX(2) = XM1
        XX(3) = XM2
        XX(4) = XM2+1D0
        XH=AML**2/XX(1)**2
        XH1=(XH-1.D0)*(2.D0-.5D0*DLOG(XH))+(1.D0-5.D0*XH)
     .  *(DATAN((2.D0*XH-1.D0)/DSQRT(4.D0*XH-1.D0))
     .   -DATAN(1.D0/DSQRT(4.D0*XH-1.D0)))/DSQRT(4.D0*XH-1.D0)
        XH2=3*GF**2/32.D0/PI**3*AMZ**4/XX(1)*GLB**2*AMB**2
        YY(1)=XH1*XH2
        XH=AML**2/XX(2)**2
        XH1=(XH-1.D0)*(2.D0-.5D0*DLOG(XH))+(1.D0-5.D0*XH)
     .  *(DATAN((2.D0*XH-1.D0)/DSQRT(4.D0*XH-1.D0))
     .   -DATAN(1.D0/DSQRT(4.D0*XH-1.D0)))/DSQRT(4.D0*XH-1.D0)
        XH2=3*GF**2/32.D0/PI**3*AMZ**4/XX(2)*GLB**2*AMB**2
        YY(2)=XH1*XH2
        YY(3)=GF/16D0/DSQRT(2D0)/PI*AMZ**4/XX(3)
     .        *BETA_HDEC(AML**2/XX(3)**2)
        YY(4)=GF/16D0/DSQRT(2D0)/PI*AMZ**4/XX(4)
     .        *BETA_HDEC(AML**2/XX(4)**2)
        HHH = FINT_HDEC(AMH,XX,YY)*GHLL**2
       ELSE
        HHH=GF/16D0/DSQRT(2D0)/PI*AMZ**4/AMH*BETA_HDEC(AML**2/AMH**2)
     .      *GHLL**2
       ENDIF
      endif
      ELSE
       IF (AMH.LE.2*AML) THEN
        HHH=0
       ELSE
        HHH=GF/16D0/DSQRT(2D0)/PI*AMZ**4/AMH*BETA_HDEC(AML**2/AMH**2)
     .      *GHLL**2
       ENDIF
      ENDIF
      endif

c MMM changed 22/8/2013
      if(i2hdm.eq.1) then
         if(ionsh.eq.0) then
            dld=0.1d0
            dlu=0.1d0
            xm1 = 2d0*aml-dld
            xm2 = 2d0*aml+dlu
            if (amh.le.aml) then
               hhh = 0d0
            elseif (amh.le.xm1) then
               xh=aml**2/amh**2
               xh1=(xh-1.d0)*(2.d0-.5d0*dlog(xh))+(1.d0-5.d0*xh)
     .              *(datan((2.d0*xh-1.d0)/dsqrt(4.d0*xh-1.d0))
     .              -datan(1.d0/dsqrt(4.d0*xh-1.d0)))
     .              /dsqrt(4.d0*xh-1.d0)
               xh2=3*gf**2/32.d0/pi**3*amz**4/amh*glb**2*amb**2
               hhh=xh1*xh2
            elseif (amh.le.xm2) then
               xx(1) = xm1-1d0
               xx(2) = xm1
               xx(3) = xm2
               xx(4) = xm2+1d0
               xh=aml**2/xx(1)**2
               xh1=(xh-1.d0)*(2.d0-.5d0*dlog(xh))+(1.d0-5.d0*xh)
     .              *(datan((2.d0*xh-1.d0)/dsqrt(4.d0*xh-1.d0))
     .              -datan(1.d0/dsqrt(4.d0*xh-1.d0)))
     .              /dsqrt(4.d0*xh-1.d0)
               xh2=3*gf**2/32.d0/pi**3*amz**4/xx(1)*glb**2*amb**2
               yy(1)=xh1*xh2
               xh=aml**2/xx(2)**2
               xh1=(xh-1.d0)*(2.d0-.5d0*dlog(xh))+(1.d0-5.d0*xh)
     .              *(datan((2.d0*xh-1.d0)/dsqrt(4.d0*xh-1.d0))
     .              -datan(1.d0/dsqrt(4.d0*xh-1.d0)))
     .              /dsqrt(4.d0*xh-1.d0)
               xh2=3*gf**2/32.d0/pi**3*amz**4/xx(2)*glb**2*amb**2
               yy(2)=xh1*xh2
               yy(3)=gf/16d0/dsqrt(2d0)/pi*amz**4/xx(3)
     .              *beta_hdec(aml**2/xx(3)**2)
               yy(4)=gf/16d0/dsqrt(2d0)/pi*amz**4/xx(4)
     .              *beta_hdec(aml**2/xx(4)**2)
               hhh = fint_hdec(amh,xx,yy)*ghll**2
            else
               hhh=gf/16d0/dsqrt(2d0)/pi*amz**4/amh*
     .              beta_hdec(aml**2/amh**2)*ghll**2
            endif
         else
            if (amh.le.2*aml) then
               hhh=0
            else
               hhh=gf/16d0/dsqrt(2d0)/pi*amz**4/amh*
     .              beta_hdec(aml**2/amh**2)*ghll**2
            endif
         endif
      endif
c end MMM changed 22/8/2013

c      print*,'H -> hh',HHH,ghll

C  H ---> A A
      if(i2hdm.eq.0) then
      IF(IONSH.EQ.0)THEN
      if(islhai.eq.0)then
       DLD = 0.3D0*(TGBET-1.3D0)
       DLD = DMAX1(0.1D0,DLD)
       DLU=DLD
       ALD = DLD/2
       ALU = DLU/2
       XM1 = 2*AMA-DLD
       XM2 = 2*AMA+DLU
       IF (AMH.LE.AMA) THEN
        HAA=0
       ELSEIF (AMH.LT.XM1) THEN
        XA=AMA**2/AMH**2
        XA1=(XA-1.D0)*(2.D0-.5D0*DLOG(XA))+(1.D0-5.D0*XA)
     .    *(DATAN((2.D0*XA-1.D0)/DSQRT(4.D0*XA-1.D0))
     .     -DATAN(1.D0/DSQRT(4.D0*XA-1.D0)))/DSQRT(4.D0*XA-1.D0)
        XA2=3*GF**2/32.D0/PI**3*AMZ**4/AMH*GHAA**2*GAB**2*AMB**2
        HAA=XA1*XA2
       ELSEIF (AMH.LT.XM2) THEN
        ZZMA=AMAR
        AMACRIT = AMAR
        AMA0 = 10.D0
        AMA1 = AMAR + 50.D0
        AMA = AMA0
        CALL SUSYCP_HDEC(TGBET)
        IF(AMH.LT.2*AMA) THEN
         IFLON0 = -1
        ELSEIF(AMH.EQ.2*AMA) THEN
         IFLON0 = 0
         AMACRIT = AMAR
        ELSE
         IFLON0 = 1
        ENDIF
        AMA = AMA1
        CALL SUSYCP_HDEC(TGBET)
        IF(AMH.LT.2*AMA) THEN
         IFLON1 = -1
        ELSEIF(AMH.EQ.2*AMA) THEN
         IFLON1 = 0
         AMACRIT = AMAR
        ELSE
         IFLON1 = 1
        ENDIF
        IF(IFLON0*IFLON1.EQ.0)THEN
         IFLON0 = 0
         IFLON1 = 0
        ENDIF
        IF(IFLON0.NE.IFLON1)THEN
502      AMA = (AMA0+AMA1)/2
         CALL SUSYCP_HDEC(TGBET)
         IF(AMH.LT.2*AMA) THEN
          IF(IFLON0.EQ.-1) THEN
           AMA0 = AMAR
          ELSE
           AMA1 = AMAR
          ENDIF
         ELSEIF(AMH.EQ.2*AMA) THEN
          IFLON0 = 0
          IFLON1 = 0
          AMACRIT = AMAR
         ELSE
          IF(IFLON0.EQ.-1) THEN
           AMA1 = AMAR
          ELSE
           AMA0 = AMAR
          ENDIF
         ENDIF
         IF(IFLON0.NE.0)THEN
          AMACRIT = (AMA0+AMA1)/2
          DEL = 1.D-8
          AMDEL = 2*DABS(AMA1-AMA0)/(AMA1+AMA0)
          IF(AMDEL.GT.DEL) GOTO 502
         ENDIF
        ENDIF
        DEL = 1.D-4
        XX(1) = AMACRIT - ALD - DEL
        XX(2) = AMACRIT - ALD
        XX(3) = AMACRIT + ALU
        XX(4) = AMACRIT + ALU + DEL
        DO J=1,4
         AMA = XX(J)
         CALL SUSYCP_HDEC(TGBET)
         XX(J) = AMH
         IF(AMH.GE.2*AMA)THEN
          YY(J)=GF/16D0/DSQRT(2D0)/PI*AMZ**4/XX(J)
     .          *BETA_HDEC(AMA**2/XX(J)**2)
         ELSEIF(AMH.LE.AMA)THEN
          YY(J) = 0
         ELSE
          XA=AMA**2/XX(J)**2
          XA1=(XA-1.D0)*(2.D0-.5D0*DLOG(XA))+(1.D0-5.D0*XA)
     .    *(DATAN((2.D0*XA-1.D0)/DSQRT(4.D0*XA-1.D0))
     .     -DATAN(1.D0/DSQRT(4.D0*XA-1.D0)))/DSQRT(4.D0*XA-1.D0)
          XA2=3*GF**2/32.D0/PI**3*AMZ**4/XX(J)*GAB**2*AMB**2
          YY(J)=XA1*XA2
         ENDIF
        ENDDO
        AMA = ZZMA
        CALL SUSYCP_HDEC(TGBET)
        HAA = FINT_HDEC(AMH,XX,YY)*GHAA**2
       ELSE
        HAA=GF/16D0/DSQRT(2D0)/PI*AMZ**4/AMH*BETA_HDEC(AMA**2/AMH**2)
     .       *GHAA**2
       ENDIF
      else
       DLD=0.1D0
       DLU=0.1D0
       XM1 = 2D0*AMA-DLD
       XM2 = 2D0*AMA+DLU
       IF (AMH.LE.AMA) THEN
        HAA = 0D0
       ELSEIF (AMH.LE.XM1) THEN
        XA=AMA**2/AMH**2
        XA1=(XA-1.D0)*(2.D0-.5D0*DLOG(XA))+(1.D0-5.D0*XA)
     .    *(DATAN((2.D0*XA-1.D0)/DSQRT(4.D0*XA-1.D0))
     .     -DATAN(1.D0/DSQRT(4.D0*XA-1.D0)))/DSQRT(4.D0*XA-1.D0)
        XA2=3*GF**2/32.D0/PI**3*AMZ**4/AMH*GHAA**2*GAB**2*AMB**2
        HAA=XA1*XA2
       ELSEIF (AMH.LE.XM2) THEN
        XX(1) = XM1-1D0
        XX(2) = XM1
        XX(3) = XM2
        XX(4) = XM2+1D0
        XA=AMA**2/XX(1)**2
        XA1=(XA-1.D0)*(2.D0-.5D0*DLOG(XA))+(1.D0-5.D0*XA)
     .  *(DATAN((2.D0*XA-1.D0)/DSQRT(4.D0*XA-1.D0))
     .   -DATAN(1.D0/DSQRT(4.D0*XA-1.D0)))/DSQRT(4.D0*XA-1.D0)
        XA2=3*GF**2/32.D0/PI**3*AMZ**4/XX(1)*GAB**2*AMB**2
        YY(1)=XA1*XA2
        XA=AMA**2/XX(2)**2
        XA1=(XA-1.D0)*(2.D0-.5D0*DLOG(XA))+(1.D0-5.D0*XA)
     .  *(DATAN((2.D0*XA-1.D0)/DSQRT(4.D0*XA-1.D0))
     .   -DATAN(1.D0/DSQRT(4.D0*XA-1.D0)))/DSQRT(4.D0*XA-1.D0)
        XA2=3*GF**2/32.D0/PI**3*AMZ**4/XX(2)*GAB**2*AMB**2
        YY(2)=XA1*XA2
        YY(3)=GF/16D0/DSQRT(2D0)/PI*AMZ**4/XX(3)
     .        *BETA_HDEC(AMA**2/XX(3)**2)
        YY(4)=GF/16D0/DSQRT(2D0)/PI*AMZ**4/XX(4)
     .        *BETA_HDEC(AMA**2/XX(4)**2)
        HAA = FINT_HDEC(AMH,XX,YY)*GHAA**2
       ELSE
        HAA=GF/16D0/DSQRT(2D0)/PI*AMZ**4/AMH*BETA_HDEC(AMA**2/AMH**2)
     .       *GHAA**2
       ENDIF
      endif
      ELSE
       IF (AMH.LE.2*AMA) THEN
        HAA=0
       ELSE
        HAA=GF/16D0/DSQRT(2D0)/PI*AMZ**4/AMH*BETA_HDEC(AMA**2/AMH**2)
     .       *GHAA**2
       ENDIF
      ENDIF
      endif

c MMM changed 22/8/2013
      if(i2hdm.eq.1) then
         if(ionsh.eq.0) then
            dld=0.1d0
            dlu=0.1d0
            xm1 = 2d0*ama-dld
            xm2 = 2d0*ama+dlu
            if (amh.le.ama) then
               haa = 0d0
            elseif (amh.le.xm1) then
               xa=ama**2/amh**2
               xa1=(xa-1.d0)*(2.d0-.5d0*dlog(xa))+(1.d0-5.d0*xa)
     .              *(datan((2.d0*xa-1.d0)/dsqrt(4.d0*xa-1.d0))
     .              -datan(1.d0/dsqrt(4.d0*xa-1.d0)))
     .              /dsqrt(4.d0*xa-1.d0)
               xa2=3*gf**2/32.d0/pi**3*amz**4/amh*ghaa**2*gab**2*amb**2
               haa=xa1*xa2
            elseif (amh.le.xm2) then
               xx(1) = xm1-1d0
               xx(2) = xm1
               xx(3) = xm2
               xx(4) = xm2+1d0
               xa=ama**2/xx(1)**2
               xa1=(xa-1.d0)*(2.d0-.5d0*dlog(xa))+(1.d0-5.d0*xa)
     .              *(datan((2.d0*xa-1.d0)/dsqrt(4.d0*xa-1.d0))
     .              -datan(1.d0/dsqrt(4.d0*xa-1.d0)))
     .              /dsqrt(4.d0*xa-1.d0)
               xa2=3*gf**2/32.d0/pi**3*amz**4/xx(1)*gab**2*amb**2
               yy(1)=xa1*xa2
               xa=ama**2/xx(2)**2
               xa1=(xa-1.d0)*(2.d0-.5d0*dlog(xa))+(1.d0-5.d0*xa)
     .              *(datan((2.d0*xa-1.d0)/dsqrt(4.d0*xa-1.d0))
     .              -datan(1.d0/dsqrt(4.d0*xa-1.d0)))
     .              /dsqrt(4.d0*xa-1.d0)
               xa2=3*gf**2/32.d0/pi**3*amz**4/xx(2)*gab**2*amb**2
               yy(2)=xa1*xa2
               yy(3)=gf/16d0/dsqrt(2d0)/pi*amz**4/xx(3)
     .              *beta_hdec(ama**2/xx(3)**2)
               yy(4)=gf/16d0/dsqrt(2d0)/pi*amz**4/xx(4)
     .              *beta_hdec(ama**2/xx(4)**2)
               haa = fint_hdec(amh,xx,yy)*ghaa**2
            else
               haa=gf/16d0/dsqrt(2d0)/pi*amz**4/amh*
     .              beta_hdec(ama**2/amh**2)*ghaa**2
            endif
         else
            if (amh.le.2*ama) then
               haa=0
            else
               haa=gf/16d0/dsqrt(2d0)/pi*amz**4/amh*
     .              beta_hdec(ama**2/amh**2)*ghaa**2
            endif
         endif
      endif
c end MMM changed 22/8/2013

c      print*,'H -> AA',haa,ghaa

C  H ---> H+ H- 

      if(i2hdm.eq.1) then
         if (amh.le.2*amch) then
            hchch=0.D0
         else
            hchch=gf/8d0/dsqrt(2d0)/pi*amz**4/amh*
     .           beta_hdec(amch**2/amh**2)*ghpm**2
         endif
      elseif(i2hdm.eq.0) then
         hchch=0.D0
      endif

c      print*,'H -> H+H-',hchch,ghpm

C  H ---> A Z
      if(i2hdm.eq.0)then
      IF(IONSH.EQ.0)THEN
      if(islhai.eq.0)then
       DLD=1D0
       DLU=8D0
       XM1 = AMA+AMZ-DLD
       XM2 = AMA+AMZ+DLU
       IF (AMH.LT.AMA) THEN
        HAZ=0
       ELSEIF (AMH.LT.XM1) THEN
        IF(AMH.LE.DABS(AMZ-AMA))THEN
         HAZ=0
        ELSE
        HAZ=9.D0*GF**2/16.D0/PI**3*AMZ**4*AMH*GZAH**2*
     .      (7.D0/12.D0-10.D0/9.D0*SS+40.D0/27.D0*SS**2)
     .      *HVH((AMA/AMH)**2,(AMZ/AMH)**2)
        ENDIF
       ELSEIF (AMH.LT.XM2) THEN
        ZZMA=AMAR
165     AMA = AMAR - 1.D0
        CALL SUSYCP_HDEC(TGBET)
        IF(AMH.LT.AMA+AMZ+DLU.AND.AMH.GT.AMA+AMZ-DLD) GOTO 165
        XX(1) = AMAR-1D0
        XX(2) = AMAR
        AMA = ZZMA
        CALL SUSYCP_HDEC(TGBET)
166     AMA = AMAR + 1.D0
        CALL SUSYCP_HDEC(TGBET)
        IF(AMH.LT.AMA+AMZ+DLU.AND.AMH.GT.AMA+AMZ-DLD) GOTO 166
        XX(3) = AMAR
        XX(4) = AMAR+1D0
        DO IJ=1,4
         AMA = XX(IJ)
         CALL SUSYCP_HDEC(TGBET)
         XX(IJ) = AMH
         IF(AMH.LE.AMA+AMZ) THEN
          YY(IJ)=9.D0*GF**2/16.D0/PI**3*AMZ**4*XX(IJ)*
     .          (7.D0/12.D0-10.D0/9.D0*SS+40.D0/27.D0*SS**2)
     .          *HVH((AMA/XX(IJ))**2,(AMZ/XX(IJ))**2)
         ELSE
          CAZ=LAMB_HDEC(AMA**2/XX(IJ)**2,AMZ**2/XX(IJ)**2)
     .       *LAMB_HDEC(XX(IJ)**2/AMZ**2,AMA**2/AMZ**2)**2
          YY(IJ)=GF/8.D0/DSQRT(2D0)/PI*AMZ**4/XX(IJ)*CAZ
         ENDIF
        ENDDO
        AMA = ZZMA
        CALL SUSYCP_HDEC(TGBET)
        HAZ = FINT_HDEC(AMH,XX,YY)*GZAH**2
       ELSE
        CAZ=LAMB_HDEC(AMA**2/AMH**2,AMZ**2/AMH**2)
     .     *LAMB_HDEC(AMH**2/AMZ**2,AMA**2/AMZ**2)**2
        HAZ=GF/8.D0/DSQRT(2D0)/PI*AMZ**4/AMH*CAZ*GZAH**2
       ENDIF
      else
       DLD=1D0
       DLU=8D0
       XM1 = AMA+AMZ-DLD
       XM2 = AMA+AMZ+DLU
       IF (AMH.LT.AMA) THEN
        HAZ=0
       ELSEIF (AMH.LT.XM1) THEN
        IF(AMH.LE.DABS(AMZ-AMA))THEN
         HAZ=0
        ELSE
        HAZ=9.D0*GF**2/16.D0/PI**3*AMZ**4*AMH*GZAH**2*
     .      (7.D0/12.D0-10.D0/9.D0*SS+40.D0/27.D0*SS**2)
     .      *HVH((AMA/AMH)**2,(AMZ/AMH)**2)
        ENDIF
       ELSEIF (AMH.LT.XM2) THEN
        XX(1) = XM1-1D0
        XX(2) = XM1
        XX(3) = XM2
        XX(4) = XM2+1D0
        YY(1)=9.D0*GF**2/16.D0/PI**3*AMZ**4*XX(1)*
     .        (7.D0/12.D0-10.D0/9.D0*SS+40.D0/27.D0*SS**2)
     .        *HVH((AMA/XX(1))**2,(AMZ/XX(1))**2)
        YY(2)=9.D0*GF**2/16.D0/PI**3*AMZ**4*XX(2)*
     .        (7.D0/12.D0-10.D0/9.D0*SS+40.D0/27.D0*SS**2)
     .        *HVH((AMA/XX(2))**2,(AMZ/XX(2))**2)
        CAZ=LAMB_HDEC(AMA**2/XX(3)**2,AMZ**2/XX(3)**2)
     .     *LAMB_HDEC(XX(3)**2/AMZ**2,AMA**2/AMZ**2)**2
        YY(3)=GF/8.D0/DSQRT(2D0)/PI*AMZ**4/XX(3)*CAZ
        CAZ=LAMB_HDEC(AMA**2/XX(4)**2,AMZ**2/XX(4)**2)
     .     *LAMB_HDEC(XX(4)**2/AMZ**2,AMA**2/AMZ**2)**2
        YY(4)=GF/8.D0/DSQRT(2D0)/PI*AMZ**4/XX(4)*CAZ
        HAZ = FINT_HDEC(AMH,XX,YY)*GZAH**2
       ELSE
        CAZ=LAMB_HDEC(AMA**2/AMH**2,AMZ**2/AMH**2)
     .     *LAMB_HDEC(AMH**2/AMZ**2,AMA**2/AMZ**2)**2
        HAZ=GF/8.D0/DSQRT(2D0)/PI*AMZ**4/AMH*CAZ*GZAH**2
       ENDIF
      endif
      ELSE
       IF (AMH.LT.AMZ+AMA) THEN
        HAZ=0
       ELSE
        CAZ=LAMB_HDEC(AMA**2/AMH**2,AMZ**2/AMH**2)
     .     *LAMB_HDEC(AMH**2/AMZ**2,AMA**2/AMZ**2)**2
        HAZ=GF/8.D0/DSQRT(2D0)/PI*AMZ**4/AMH*CAZ*GZAH**2
       ENDIF
      ENDIF
      endif

c MMM changed 22/8/2013
      if(i2hdm.eq.1) then
         if(ionsh.eq.0) then
            dld=1d0
            dlu=8d0
            xm1 = ama+amz-dld
            xm2 = ama+amz+dlu
            if (amh.lt.ama) then
               haz=0
            elseif (amh.lt.xm1) then
               if(amh.le.dabs(amz-ama))then
                  haz=0
               else
                  haz=9.d0*gf**2/16.d0/pi**3*amz**4*amh*gzah**2*
     .                 (7.d0/12.d0-10.d0/9.d0*ss+40.d0/27.d0*ss**2)
     .                 *hvh((ama/amh)**2,(amz/amh)**2)
               endif
            elseif (amh.lt.xm2) then
               xx(1) = xm1-1d0
               xx(2) = xm1
               xx(3) = xm2
               xx(4) = xm2+1d0
               yy(1)=9.d0*gf**2/16.d0/pi**3*amz**4*xx(1)*
     .              (7.d0/12.d0-10.d0/9.d0*ss+40.d0/27.d0*ss**2)
     .              *hvh((ama/xx(1))**2,(amz/xx(1))**2)
               yy(2)=9.d0*gf**2/16.d0/pi**3*amz**4*xx(2)*
     .              (7.d0/12.d0-10.d0/9.d0*ss+40.d0/27.d0*ss**2)
     .              *hvh((ama/xx(2))**2,(amz/xx(2))**2)
               caz=lamb_hdec(ama**2/xx(3)**2,amz**2/xx(3)**2)
     .              *lamb_hdec(xx(3)**2/amz**2,ama**2/amz**2)**2
               yy(3)=gf/8.d0/dsqrt(2d0)/pi*amz**4/xx(3)*caz
               caz=lamb_hdec(ama**2/xx(4)**2,amz**2/xx(4)**2)
     .              *lamb_hdec(xx(4)**2/amz**2,ama**2/amz**2)**2
               yy(4)=gf/8.d0/dsqrt(2d0)/pi*amz**4/xx(4)*caz
               haz = fint_hdec(amh,xx,yy)*gzah**2
            else
               caz=lamb_hdec(ama**2/amh**2,amz**2/amh**2)
     .              *lamb_hdec(amh**2/amz**2,ama**2/amz**2)**2
               haz=gf/8.d0/dsqrt(2d0)/pi*amz**4/amh*caz*gzah**2
            endif
         else
            if (amh.lt.amz+ama) then
               haz=0
            else
               caz=lamb_hdec(ama**2/amh**2,amz**2/amh**2)
     .              *lamb_hdec(amh**2/amz**2,ama**2/amz**2)**2
               haz=gf/8.d0/dsqrt(2d0)/pi*amz**4/amh*caz*gzah**2
            endif
         endif
      endif
c end MMM changed 22/8/2013

c      print*,'H -> AZ',haz

C  H ---> H+ W-
      if(i2hdm.eq.0)then
      IF(IONSH.EQ.0)THEN
      if(islhai.eq.0)then
       DLD=3D0
       DLU=9D0
       XM1 = AMCH+AMW-DLD
       XM2 = AMCH+AMW+DLU
       IF (AMH.LT.AMCH) THEN
        HHW=0.D0
       ELSEIF (AMH.LT.XM1) THEN
        IF(AMH.LE.DABS(AMW-AMCH))THEN
         HHW=0
        ELSE
        HHW=9.D0*GF**2/16.D0/PI**3*AMW**4*AMH*GLVV**2*2
     .      *HVH((AMCH/AMH)**2,(AMW/AMH)**2)
        ENDIF
       ELSEIF (AMH.LT.XM2) THEN
        ZZMA=AMAR
167     AMA = AMAR - 1.D0
        CALL SUSYCP_HDEC(TGBET)
        IF(AMH.LT.AMCH+AMW+DLU) GOTO 167
        XX(1) = AMAR-1D0
        XX(2) = AMAR
        AMA = ZZMA
        CALL SUSYCP_HDEC(TGBET)
168     AMA = AMAR + 1.D0
        CALL SUSYCP_HDEC(TGBET)
        IF(AMH.GT.AMCH+AMW-DLD) GOTO 168
        XX(3) = AMAR
        XX(4) = AMAR+1D0
        AMA = XX(1)
        CALL SUSYCP_HDEC(TGBET)
        XX(1) = AMH
        CHW=LAMB_HDEC(AMCH**2/XX(1)**2,AMW**2/XX(1)**2)
     .     *LAMB_HDEC(XX(1)**2/AMW**2,AMCH**2/AMW**2)**2
        YY(1)=2*GF/8.D0/DSQRT(2D0)/PI*AMW**4/XX(1)*CHW
        AMA = XX(2)
        CALL SUSYCP_HDEC(TGBET)
        XX(2) = AMH
        CHW=LAMB_HDEC(AMCH**2/XX(2)**2,AMW**2/XX(2)**2)
     .     *LAMB_HDEC(XX(2)**2/AMW**2,AMCH**2/AMW**2)**2
        YY(2)=2*GF/8.D0/DSQRT(2D0)/PI*AMW**4/XX(2)*CHW
        AMA = XX(3)
        CALL SUSYCP_HDEC(TGBET)
        XX(3) = AMH
        YY(3)=9.D0*GF**2/16.D0/PI**3*AMW**4*XX(3)*2
     .       *HVH((AMCH/XX(3))**2,(AMW/XX(3))**2)
        AMA = XX(4)
        CALL SUSYCP_HDEC(TGBET)
        XX(4) = AMH
        YY(4)=9.D0*GF**2/16.D0/PI**3*AMW**4*XX(4)*2
     .       *HVH((AMCH/XX(4))**2,(AMW/XX(4))**2)
        AMA = ZZMA
        CALL SUSYCP_HDEC(TGBET)
        HHW=FINT_HDEC(AMH,XX,YY)*GLVV**2
       ELSE
        CHW=LAMB_HDEC(AMCH**2/AMH**2,AMW**2/AMH**2)
     .     *LAMB_HDEC(AMH**2/AMW**2,AMCH**2/AMW**2)**2
        HHW=2*GF/8.D0/DSQRT(2D0)/PI*AMW**4/AMH*CHW*GLVV**2
       ENDIF
      else
       DLD=3D0
       DLU=9D0
       XM1 = AMCH+AMW-DLD
       XM2 = AMCH+AMW+DLU
       IF (AMH.LT.AMCH) THEN
        HHW=0.D0
       ELSEIF (AMH.LT.XM1) THEN
        IF(AMH.LE.DABS(AMW-AMCH))THEN
         HHW=0
        ELSE
        HHW=9.D0*GF**2/16.D0/PI**3*AMW**4*AMH*GLVV**2*2
     .      *HVH((AMCH/AMH)**2,(AMW/AMH)**2)
        ENDIF
       ELSEIF (AMH.LT.XM2) THEN
        XX(1) = XM1-1D0
        XX(2) = XM1
        XX(3) = XM2
        XX(4) = XM2+1D0
        YY(1)=9.D0*GF**2/16.D0/PI**3*AMW**4*XX(1)*2
     .       *HVH((AMCH/XX(1))**2,(AMW/XX(1))**2)
        YY(2)=9.D0*GF**2/16.D0/PI**3*AMW**4*XX(2)*2
     .       *HVH((AMCH/XX(2))**2,(AMW/XX(2))**2)
        CHW=LAMB_HDEC(AMCH**2/XX(3)**2,AMW**2/XX(3)**2)
     .     *LAMB_HDEC(XX(3)**2/AMW**2,AMCH**2/AMW**2)**2
        YY(3)=2*GF/8.D0/DSQRT(2D0)/PI*AMW**4/XX(3)*CHW
        CHW=LAMB_HDEC(AMCH**2/XX(4)**2,AMW**2/XX(4)**2)
     .     *LAMB_HDEC(XX(4)**2/AMW**2,AMCH**2/AMW**2)**2
        YY(4)=2*GF/8.D0/DSQRT(2D0)/PI*AMW**4/XX(4)*CHW
        HHW=FINT_HDEC(AMH,XX,YY)*GLVV**2
       ELSE
        CHW=LAMB_HDEC(AMCH**2/AMH**2,AMW**2/AMH**2)
     .     *LAMB_HDEC(AMH**2/AMW**2,AMCH**2/AMW**2)**2
        HHW=2*GF/8.D0/DSQRT(2D0)/PI*AMW**4/AMH*CHW*GLVV**2
       ENDIF
      endif
      ELSE
       IF (AMH.LT.AMW+AMCH) THEN
        HHW=0.D0
       ELSE
        CHW=LAMB_HDEC(AMCH**2/AMH**2,AMW**2/AMH**2)
     .     *LAMB_HDEC(AMH**2/AMW**2,AMCH**2/AMW**2)**2
        HHW=2*GF/8.D0/DSQRT(2D0)/PI*AMW**4/AMH*CHW*GLVV**2
       ENDIF
      ENDIF
      endif

c MMM changed 22/8/2013
      if(i2hdm.eq.1) then
         if(ionsh.eq.0) then
            dld=3d0
            dlu=9d0
            xm1 = amch+amw-dld
            xm2 = amch+amw+dlu
            if (amh.lt.amch) then
               hhw=0.d0
            elseif (amh.lt.xm1) then
               if(amh.le.dabs(amw-amch))then
                  hhw=0
               else
                  hhw=9.d0*gf**2/16.d0/pi**3*amw**4*amh*glvv**2*2
     .                 *hvh((amch/amh)**2,(amw/amh)**2)
               endif
            elseif (amh.lt.xm2) then
               xx(1) = xm1-1d0
               xx(2) = xm1
               xx(3) = xm2
               xx(4) = xm2+1d0
               yy(1)=9.d0*gf**2/16.d0/pi**3*amw**4*xx(1)*2
     .              *hvh((amch/xx(1))**2,(amw/xx(1))**2)
               yy(2)=9.d0*gf**2/16.d0/pi**3*amw**4*xx(2)*2
     .              *hvh((amch/xx(2))**2,(amw/xx(2))**2)
               chw=lamb_hdec(amch**2/xx(3)**2,amw**2/xx(3)**2)
     .              *lamb_hdec(xx(3)**2/amw**2,amch**2/amw**2)**2
               yy(3)=2*gf/8.d0/dsqrt(2d0)/pi*amw**4/xx(3)*chw
               chw=lamb_hdec(amch**2/xx(4)**2,amw**2/xx(4)**2)
     .              *lamb_hdec(xx(4)**2/amw**2,amch**2/amw**2)**2
               yy(4)=2*gf/8.d0/dsqrt(2d0)/pi*amw**4/xx(4)*chw
               hhw=fint_hdec(amh,xx,yy)*glvv**2
            else
               chw=lamb_hdec(amch**2/amh**2,amw**2/amh**2)
     .              *lamb_hdec(amh**2/amw**2,amch**2/amw**2)**2
               hhw=2*gf/8.d0/dsqrt(2d0)/pi*amw**4/amh*chw*glvv**2
            endif
         else
            if (amh.lt.amw+amch) then
               hhw=0.d0
            else
               chw=lamb_hdec(amch**2/amh**2,amw**2/amh**2)
     .              *lamb_hdec(amh**2/amw**2,amch**2/amw**2)**2
               hhw=2*gf/8.d0/dsqrt(2d0)/pi*amw**4/amh*chw*glvv**2
            endif
         endif
      endif
c end MMM changed 22/8/2013

c      print*,'H -> H+W-',hhw,hhw/2.D0

C ========================== SUSY DECAYS 
C
      IF(IOFSUSY.EQ.0) THEN
C  HH ----> CHARGINOS
      DO 741 I=1,2
      DO 741 J=1,2
      IF (AMH.GT.AMCHAR(I)+AMCHAR(J)) THEN
      WHHCH(I,J)=GF*AMW**2/(2*PI*DSQRT(2.D0))/AMH 
     .     *LAMB_HDEC(AMCHAR(I)**2/AMH**2,AMCHAR(J)**2/AMH**2)
     .     *( (AC1(I,J)**2+AC1(J,I)**2)*(AMH**2-AMCHAR(I)
     .         **2-AMCHAR(J)**2)-4.D0*AC1(I,J)*AC1(J,I)* 
     .         XMCHAR(I)*XMCHAR(J) ) 
      ELSE
      WHHCH(I,J)=0.D0
      ENDIF
 741  CONTINUE
      WHHCHT=WHHCH(1,1)+WHHCH(1,2)+WHHCH(2,1)+WHHCH(2,2)
C
C  HH ----> NEUTRALINOS 
      DO 742 I=1,4
      DO 742 J=1,4
      IF (AMH.GT.AMNEUT(I)+AMNEUT(J)) THEN
      WHHNE(I,J)=GF*AMW**2/(2*PI*DSQRT(2.D0))/AMH
     .         *AN1(I,J)**2*(AMH**2-(XMNEUT(I)+XMNEUT(J))**2)
     .         *LAMB_HDEC(AMNEUT(I)**2/AMH**2,AMNEUT(J)**2/AMH**2)
      ELSE 
      WHHNE(I,J)=0.D0
      ENDIF
 742  CONTINUE
      WHHNET= WHHNE(1,1)+WHHNE(1,2)+WHHNE(1,3)+WHHNE(1,4)
     .       +WHHNE(2,1)+WHHNE(2,2)+WHHNE(2,3)+WHHNE(2,4)
     .       +WHHNE(3,1)+WHHNE(3,2)+WHHNE(3,3)+WHHNE(3,4)
     .       +WHHNE(4,1)+WHHNE(4,2)+WHHNE(4,3)+WHHNE(4,4)
C
C  HH ----> SLEPTONS 
C
      IF (AMH.GT.2.D0*AMSE(1)) THEN
      WHHSLEL=2*GF/2.D0/DSQRT(2D0)/PI*AMZ**4/AMH*DCOS(B+A)**2
     .      *BETA_HDEC(AMSE(1)**2/AMH**2)*(-0.5D0+SS)**2
      ELSE
      WHHSLEL=0.D0
      ENDIF

      IF (AMH.GT.2.D0*AMSE(2)) THEN
      WHHSLER=2*GF/2.D0/DSQRT(2D0)/PI*AMZ**4/AMH*DCOS(B+A)**2
     .      *BETA_HDEC(AMSE(2)**2/AMH**2)*SS**2
      ELSE
      WHHSLER=0.D0
      ENDIF

      WHHSLNL=0.D0
      IF (AMH.GT.2.D0*AMSN1(1)) THEN
      WHHSLNL=2*GF/2.D0/DSQRT(2D0)/PI*AMZ**4/AMH*DCOS(B+A)**2
     .      *BETA_HDEC(AMSN1(1)**2/AMH**2)*0.5D0**2
      ENDIF
      IF (AMH.GT.2.D0*AMSN(1)) THEN
      WHHSLNL=WHHSLNL + GF/2.D0/DSQRT(2D0)/PI*AMZ**4/AMH*DCOS(B+A)**2
     .      *BETA_HDEC(AMSN(1)**2/AMH**2)*0.5D0**2
      ENDIF

      DO 748 I=1,2
      DO 748 J=1,2
      IF(AMH.GT.AMSL(I)+AMSL(J)) THEN
      WHHSTAU(I,J)=GF*AMZ**4/2.D0/DSQRT(2.D0)/PI*GHEE(I,J)**2*
     .      LAMB_HDEC(AMSL(I)**2/AMH**2,AMSL(J)**2/AMH**2)/AMH
      ELSE
      WHHSTAU(I,J)=0.D0
      ENDIF
 748  CONTINUE

      WHHSLT=WHHSTAU(1,1)+WHHSTAU(1,2)+WHHSTAU(2,1)+WHHSTAU(2,2) 
     .       +WHHSLEL+WHHSLER+WHHSLNL
C
C  HH ----> SQUARKS 
C
      IF (AMH.GT.2.D0*AMSU(1)) THEN
      WHHSQUL=6*GF/2.D0/DSQRT(2D0)/PI*AMZ**4/AMH*DCOS(B+A)**2
     .      *BETA_HDEC(AMSU(1)**2/AMH**2)*(0.5D0-2.D0/3.D0*SS)**2
      ELSE
      WHHSQUL=0.D0
      ENDIF

      IF (AMH.GT.2.D0*AMSU(2)) THEN
      WHHSQUR=6*GF/2.D0/DSQRT(2D0)/PI*AMZ**4/AMH*DCOS(B+A)**2
     .      *BETA_HDEC(AMSU(2)**2/AMH**2)*(-2.D0/3.D0*SS)**2
      ELSE
      WHHSQUR=0.D0
      ENDIF

      IF (AMH.GT.2.D0*AMSD(1)) THEN
      WHHSQDL=6*GF/2.D0/DSQRT(2D0)/PI*AMZ**4/AMH*DCOS(B+A)**2
     .      *BETA_HDEC(AMSD(1)**2/AMH**2)*(-0.5D0+1.D0/3.D0*SS)**2
      ELSE
      WHHSQDL=0.D0
      ENDIF

      IF (AMH.GT.2.D0*AMSD(2)) THEN
      WHHSQDR=6*GF/2.D0/DSQRT(2D0)/PI*AMZ**4/AMH*DCOS(B+A)**2
     .      *BETA_HDEC(AMSD(2)**2/AMH**2)*(+1.D0/3.D0*SS)**2
      ELSE
      WHHSQDR=0.D0
      ENDIF

      WHHSQ=WHHSQUL+WHHSQUR+WHHSQDL+WHHSQDR
C
C  HH ----> STOPS 
      SUSY = 1
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     DO K=-10,10
c     DO K=-1,1
c     DO I=1,2
c     DO J=1,2
c     QSQ = AMH*10.D0**(K/10.D0)
c     QSQ = AMH*2.D0**(K)
c     IF(AMH.GT.YMST(I)+YMST(J)) THEN
c      CALL SQMBAPP_HDEC(QSQ)
c      SUSY = 1+SQSUSY_HDEC(2,1,I,J,QSQ,0,1)
c      WHHST0(I,J)=3*GF*AMZ**4/2.D0/DSQRT(2.D0)/PI*YHTT(I,J)**2*
c    .       LAMB_HDEC(YMST(I)**2/AMH**2,YMST(J)**2/AMH**2)/AMH
c      WHHST(I,J)=3*GF*AMZ**4/2.D0/DSQRT(2.D0)/PI*YHTT(I,J)**2*
c    .       LAMB_HDEC(YMST(I)**2/AMH**2,YMST(J)**2/AMH**2)/AMH
c    .      *SUSY
c     ELSE
c     WHHST(I,J)=0.D0
c     ENDIF
c     ENDDO
c     ENDDO
c     write(9,*)'H -> t1 t1: ',QSQ/AMH,WHHST0(1,1),WHHST(1,1)
c     write(9,*)'numbers: ',3*GF/2.D0/DSQRT(2.D0)/PI/AMH,
c    .YHTT(1,1)**2*AMZ**4,LAMB_HDEC(YMST(1)**2/AMH**2,YMST(1)**2/AMH**2)
c    .,YHTT(1,1)*AMZ**2
c    .,3*GF/2.D0/DSQRT(2.D0)/PI/AMH*
c    .YHTT(1,1)**2*AMZ**4*LAMB_HDEC(YMST(1)**2/AMH**2,YMST(1)**2/AMH**2)
c     write(9,*)'H -> t1 t2: ',QSQ/AMH,WHHST0(1,2),WHHST(1,2)
c     write(9,*)'H -> t2 t2: ',QSQ/AMH,WHHST0(2,2),WHHST(2,2)
c     write(901,('1X,G10.4,6(1X,G10.4)'))QSQ/AMH,WHHST0(1,1),WHHST(1,1),
c    .                    WHHST0(1,2),WHHST(1,2),WHHST0(2,2),WHHST(2,2)
c     ENDDO
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      DO 743 I=1,2
      DO 743 J=1,2
c     QSQ = (YMST(I)+YMST(J))/2
      QSQ = AMH
      IF(AMH.GT.YMST(I)+YMST(J)) THEN
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
       CALL SQMBAPP_HDEC(QSQ)
       SUSY = 1+SQSUSY_HDEC(2,1,I,J,QSQ,0,1)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
       WHHST(I,J)=3*GF*AMZ**4/2.D0/DSQRT(2.D0)/PI*YHTT(I,J)**2*
     .       LAMB_HDEC(YMST(I)**2/AMH**2,YMST(J)**2/AMH**2)/AMH
     .      *SUSY
c     write(6,*)'H -> stop: ',I,J,AMH,YMST(I),YMST(J),100*(SUSY-1),'% ',
c    .          WHHST(I,J)/SUSY,WHHST(I,J)
c      if(i.eq.1.and.j.eq.1)write(511,*)AMH,WHHST(I,J),WHHST(I,J)/SUSY
c      if(i.eq.1.and.j.eq.2)write(512,*)AMH,WHHST(I,J),WHHST(I,J)/SUSY
c      if(i.eq.2.and.j.eq.1)write(521,*)AMH,WHHST(I,J),WHHST(I,J)/SUSY
c      if(i.eq.2.and.j.eq.2)write(522,*)AMH,WHHST(I,J),WHHST(I,J)/SUSY
c      write(6,*)'H -> stop: ',I,J,AMH,YMST(I),YMST(J),SUSY-1
      ELSE
      WHHST(I,J)=0.D0
      ENDIF
 743  CONTINUE
C
C  HH ----> SBOTTOMS 
      SUSY = 1
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     DO K=-10,10
c     DO K=-1,1
c     DO I=1,2
c     DO J=1,2
c     QSQ = AMH*10.D0**(K/10.D0)
c     QSQ = AMH*2.D0**(K)
c     IF(AMH.GT.YMSB(I)+YMSB(J)) THEN
c      CALL SQMBAPP_HDEC(QSQ)
c      SUSY = 1+SQSUSY_HDEC(2,2,I,J,QSQ,0,1)
c      WHHSB0(I,J)=3*GF*AMZ**4/2.D0/DSQRT(2.D0)/PI*YHBB(I,J)**2*
c    .       LAMB_HDEC(YMSB(I)**2/AMH**2,YMSB(J)**2/AMH**2)/AMH
c      WHHSB(I,J)=3*GF*AMZ**4/2.D0/DSQRT(2.D0)/PI*YHBB(I,J)**2*
c    .       LAMB_HDEC(YMSB(I)**2/AMH**2,YMSB(J)**2/AMH**2)/AMH
c    .      *SUSY
c     ELSE
c      WHHSB(I,J)=0.D0
c     ENDIF
c     ENDDO
c     ENDDO
c     write(9,*)'H -> b1 b1: ',QSQ/AMH,WHHSB0(1,1),WHHSB(1,1)
c     write(9,*)'H -> b1 b2: ',QSQ/AMH,WHHSB0(1,2),WHHSB(1,2)
c     write(9,*)'H -> b2 b2: ',QSQ/AMH,WHHSB0(2,2),WHHSB(2,2)
c     write(902,('1X,G10.4,6(1X,G10.4)'))QSQ/AMH,WHHSB0(1,1),WHHSB(1,1),
c    .        WHHSB0(1,2),WHHSB(1,2),WHHSB0(2,2),WHHSB(2,2)
c     ENDDO
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     dummy0 = 0
c     dummy1 = 0
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      DO 744 I=1,2
      DO 744 J=1,2
c     QSQ = (YMSB(I)+YMSB(J))/2
      QSQ = AMH
      IF(AMH.GT.YMSB(I)+YMSB(J)) THEN
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
       CALL SQMBAPP_HDEC(QSQ)
       SUSY = 1+SQSUSY_HDEC(2,2,I,J,QSQ,0,1)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
       WHHSB(I,J)=3*GF*AMZ**4/2.D0/DSQRT(2.D0)/PI*YHBB(I,J)**2*
     .       LAMB_HDEC(YMSB(I)**2/AMH**2,YMSB(J)**2/AMH**2)/AMH
     .      *SUSY
c     write(6,*)'H -> sbot: ',I,J,AMH,YMSB(I),YMSB(J),100*(SUSY-1),'% ',
c    .          WHHSB(I,J)/SUSY,WHHSB(I,J)
c      if(i.eq.1.and.j.eq.1)write(611,*)AMH,WHHSB(I,J),WHHSB(I,J)/SUSY
c      if(i.eq.1.and.j.eq.2)write(612,*)AMH,WHHSB(I,J),WHHSB(I,J)/SUSY
c      if(i.eq.2.and.j.eq.1)write(621,*)AMH,WHHSB(I,J),WHHSB(I,J)/SUSY
c      if(i.eq.2.and.j.eq.2)write(622,*)AMH,WHHSB(I,J),WHHSB(I,J)/SUSY
c      write(6,*)'H -> sbot: ',I,J,AMH,YMSB(I),YMSB(J),SUSY-1
c      dummy0 = dummy0 + WHHSB(I,J)/SUSY
c      dummy1 = dummy1 + WHHSB(I,J)
      ELSE
      WHHSB(I,J)=0.D0
      ENDIF
 744  CONTINUE
c     write(6,*)'H -> sbot0: ',YMSB(1),YMSB(2),dummy0,dummy1,
c    .                        100*(dummy1/dummy0-1),'%'
C
      WHHSTT=WHHST(1,1)+WHHST(1,2)+WHHST(2,1)+WHHST(2,2) 
      WHHSBB=WHHSB(1,1)+WHHSB(1,2)+WHHSB(2,1)+WHHSB(2,2) 
      WHHSQT=WHHSTT+WHHSBB+WHHSQ
C
      ELSE 
      WHHCHT=0.D0
      WHHNET=0.D0
      WHHSLT=0.D0
      WHHSQT=0.D0
C--Change thanks to Elzbieta Richter-Was
      DO I=1,2
       DO J=1,2
        WHHCH(I,J)=0.D0
        WHHST(I,J)=0.D0
        WHHSB(I,J)=0.D0
        WHHSTAU(I,J)=0.D0
       ENDDO
      ENDDO
      DO I=1,4
       DO J=1,4
        WHHNE(I,J)=0.D0
       ENDDO
      ENDDO
      ENDIF

      IF(IGOLD.NE.0)THEN
C   HH ---> GOLDSTINOS
       DO 740 I=1,4
       IF (AMH.GT.AMNEUT(I)) THEN
        WHHGD(I)=AMH**5/AXMPL**2/AXMGD**2/48.D0/PI*
     .           (1.D0-AMNEUT(I)**2/AMH**2)**4*AGDH(I)**2
       ELSE
        WHHGD(I)=0.D0
       ENDIF
 740   CONTINUE
       WHHGDT=WHHGD(1)+WHHGD(2)+WHHGD(3)+WHHGD(4)
      ELSE
       WHHGDT=0
      ENDIF
C
C    ==========  TOTAL WIDTH AND BRANCHING RATIOS 
      WTOT=HLL+HMM+HSS+HCC+HBB+HTT+HGG+HGA+HZGA+HWW+HZZ+HHH+HAA+HAZ
     .    +HHW+WHHCHT+WHHNET+WHHSLT+WHHSQT + WHHGDT

c     print*,'wtot',wtot

c maggie changed 21/10/2013
      WTOT=WTOT+HCHCH
      HHBRCHCH=HCHCH/WTOT
c end maggie changed 21/10/2013

c     print*,'wtot',wtot

      HHBRT=HTT/WTOT
      HHBRB=HBB/WTOT
      HHBRL=HLL/WTOT
      HHBRM=HMM/WTOT
      HHBRS=HSS/WTOT
      HHBRC=HCC/WTOT
      HHBRG=HGG/WTOT
      HHBRGA=HGA/WTOT
      HHBRZGA=HZGA/WTOT
      HHBRW=HWW/WTOT
      HHBRZ=HZZ/WTOT
      HHBRH=HHH/WTOT
      HHBRA=HAA/WTOT
      HHBRAZ=HAZ/WTOT
      HHBRHW=HHW/WTOT
      DO 841 I=1,2
      DO 841 J=1,2
      HHBRSC(I,J)=WHHCH(I,J)/WTOT
841   CONTINUE
      DO 842 I=1,4
      DO 842 J=1,4
      HHBRSN(I,J)=WHHNE(I,J)/WTOT
842   CONTINUE
      HHBRCHT=WHHCHT/WTOT 
      HHBRNET=WHHNET/WTOT 
      HHBRSL=WHHSLT/WTOT
      HHBRSQ=WHHSQ/WTOT
      HHBRSQT=WHHSQT/WTOT
      DO 843 I=1,2
      DO 843 J=1,2
      HHBRST(I,J)=WHHST(I,J)/WTOT
843   CONTINUE
      DO 844 I=1,2
      DO 844 J=1,2
      HHBRSB(I,J)=WHHSB(I,J)/WTOT
844   CONTINUE
      HHBRGD =WHHGDT/WTOT
      HHWDTH=WTOT

      BHHSLNL = WHHSLNL/WTOT
      BHHSLEL = WHHSLEL/WTOT
      BHHSLER = WHHSLER/WTOT
      BHHSQUL = WHHSQUL/WTOT
      BHHSQUR = WHHSQUR/WTOT
      BHHSQDL = WHHSQDL/WTOT
      BHHSQDR = WHHSQDR/WTOT
      DO I = 1,2
       DO J = 1,2
        BHHST(I,J) = WHHST(I,J)/WTOT
        BHHSB(I,J) = WHHSB(I,J)/WTOT
        BHHSTAU(I,J) = WHHSTAU( I,J)/WTOT
       ENDDO
      ENDDO

      ENDIF

      IF(IHIGGS.EQ.3.OR.IHIGGS.EQ.5)THEN 
C
C        =========================================================
C                       CP ODD  HIGGS DECAYS
C        =========================================================
C     =============  RUNNING MASSES 
      RMS = RUNM_HDEC(AMA,3)
      RMC = RUNM_HDEC(AMA,4)
      RMB = RUNM_HDEC(AMA,5)
      RMT = RUNM_HDEC(AMA,6)
      RATCOUP = GAT/GAB
      HIGTOP = AMA**2/AMT**2

      ASH=ALPHAS_HDEC(AMA,3)
      AMC0=1.D8
      AMB0=2.D8
C     AMT0=3.D8
      AS3=ALPHAS_HDEC(AMA,3)
      AMC0=AMC
      AS4=ALPHAS_HDEC(AMA,3)
      AMB0=AMB
C     AMT0=AMT

C     =============== PARTIAL WIDTHS
C  A ---> G G
       EPS=1.D-8
       NFEXT = 3
       ASG = AS3
       CTT = 4*AMT**2/AMA**2*DCMPLX(1D0,-EPS)
       CTB = 4*AMB**2/AMA**2*DCMPLX(1D0,-EPS)
       CAT = CTT*CF(CTT)*GAT
       CAB = CTB*CF(CTB)*GAB
       CTC = 4*AMC**2/AMA**2*DCMPLX(1D0,-EPS)
       CAC = CTC*CF(CTC)*GAT
       FQCD=AGGQCD(ASG,NFEXT)
       XFAC = CDABS(CAT+CAB+CAC)**2*FQCD
       HGG=GF/(16.D0*PI*DSQRT(2.D0))*AMA**3*(ASG/PI)**2*XFAC

c      print*
c      print*,'A decay widths'
c      print*,'hgg_NLO',hgg

C  A ---> G G* ---> G CC   TO BE ADDED TO A ---> CC
       NFEXT = 4
       ASG = AS4
       FQCD=AGGQCD(ASG,NFEXT)
       XFAC = CDABS(CAT+CAB+CAC)**2*FQCD
       DCC=GF/(16.D0*PI*DSQRT(2.D0))*AMA**3*(ASG/PI)**2*XFAC
     .     - HGG

C  A ---> G G* ---> G BB   TO BE ADDED TO A ---> BB
       NFEXT = 5
       ASG = ASH
       FQCD=AGGQCD(ASG,NFEXT)
       XFAC = CDABS(CAT+CAB+CAC)**2*FQCD
       DBB=GF/(16.D0*PI*DSQRT(2.D0))*AMA**3*(ASG/PI)**2*XFAC
     .     - HGG - DCC
       HGG=GF/(16.D0*PI*DSQRT(2.D0))*AMA**3*(ASG/PI)**2*XFAC

C  A ---> G G: FULL NNLO CORRECTIONS TO TOP LOOPS FOR NF=5
       FQCD0=AGGQCD(ASG,5)
       FQCD=AGGQCD2(ASG,5,AMA,AMT)
       XFAC = CDABS(CAT+CAB+CAC)**2*(FQCD-FQCD0)
       HGG=HGG+GF/(16.D0*PI*DSQRT(2.D0))*AMA**3*(ASG/PI)**2*XFAC

      IF(NFGG.EQ.3)THEN
       HGG = HGG - DBB - DCC
      ELSEIF(NFGG.EQ.4)THEN
       HGG = HGG - DBB
       DCC = 0
      ELSE
       DCC = 0
       DBB = 0
      ENDIF

c     print*,'hgg_NNLO',hgg

C  A ---> MU MU
      XGLM = GLB
      XGHM = GHB
      XGAM = GAB
      if(i2hdm.eq.1) then
         xgam = galep
      endif
      IF(IOFSUSY.EQ.0) THEN
       CALL STAUSUSY_HDEC(GLB,GHB,GAB,XGLM,XGHM,XGAM,QSUSY,0)
      ENDIF
      IF(AMA.LE.2*AMMUON) THEN
       HMM = 0
      ELSE
      HMM=AFF(AMA,(AMMUON/AMA)**2)*XGAM**2
      ENDIF
      
c     print*,'A -> mumu',hmm

C  A ---> LL
      XGLT = GLB
      XGHT = GHB
      XGAT = GAB
      if(i2hdm.eq.1) then
         xgat = galep
      endif
      IF(IOFSUSY.EQ.0) THEN
       CALL STAUSUSY_HDEC(GLB,GHB,GAB,XGLT,XGHT,XGAT,QSUSY,1)
      ENDIF
      IF(AMA.LE.2*AMTAU) THEN
       HLL = 0
      ELSE
      HLL=AFF(AMA,(AMTAU/AMA)**2)*XGAT**2
      ENDIF

c     write(6,*)'A: tau/mu: ',HLL/HMM*AMMUON**2/AMTAU**2,XGAT**2/XGAM**2
c     print*,'A -> tautau',hll
C  A --> SS
      XGLS = GLB
      XGHS = GHB
      XGAS = GAB
      IF(IOFSUSY.EQ.0) THEN
       CALL STRSUSY_HDEC(GLB,GHB,GAB,XGLS,XGHS,XGAS,QSUSY,LOOP)
      ENDIF
      IF(AMA.LE.2*AMS) THEN
       HSS = 0
      ELSE
       HS1=3.D0*AFF(AMA,(AMS/AMA)**2)
     .    *XGAS**2
     .    *TQCDA(AMS**2/AMA**2)
       HS2=3.D0*AFF(AMA,(RMS/AMA)**2)
     .    *XGAS**2
     .    *QCDA(RMS**2/AMA**2)
       IF(HS2.LT.0.D0) HS2 = 0
       RAT = 2*AMS/AMA
       HSS = QQINT_HDEC(RAT,HS1,HS2)
      ENDIF

c     print*,'A -> ss',hss
C  A --> CC
      RATCOUP = 1
      IF(AMA.LE.2*AMC) THEN
       HCC = 0
      ELSE
       HC1=3.D0*AFF(AMA,(AMC/AMA)**2)
     .    *GAT**2
     .    *TQCDA(AMC**2/AMA**2)
       HC2=3.D0*AFF(AMA,(RMC/AMA)**2)
     .    *GAT**2
     .    *QCDA(RMC**2/AMA**2)
     .   + DCC
       IF(HC2.LT.0.D0) HC2 = 0
       RAT = 2*AMC/AMA
       HCC = QQINT_HDEC(RAT,HC1,HC2)
      ENDIF

c     print*,'A -> cc',hcc

C  A --> BB :
      QQ = AMB
      SUSY = 0
      XGAB = GAB
      SSUSY = (AMSB(1)+AMSB(2)+AMGLU)/3*QSUSY
      AS0 = ALPHAS_HDEC(SSUSY,3)
      IF(IOFSUSY.EQ.0) THEN
       I0 = 1
       CALL DMBAPP_HDEC(I0,DGLB,DGHB,DGAB,QSUSY,LOOP)
       I0 = 3
       BSC = (AMSQ+AMUR+AMDR)/3
c      XMB = AMB
       DELB0 = -DGAB/(1+1/TGBET**2)
       XMB = RUNM_HDEC(SUSYSCALE,5)/(1+DELB0)
       SUSY = COFSUSY_HDEC(I0,AMB,XMB,QQ)*AS0/PI - 2*DGAB
       CALL BOTSUSY_HDEC(GLB,GHB,GAB,XGLB,XGHB,XGAB,QSUSY,LOOP)
      ENDIF
      RATCOUP = GAT/XGAB
      IF(AMA.LE.2*AMB) THEN
       HBB = 0
      ELSE
       HB1=3.D0*AFF(AMA,(AMB/AMA)**2)
     .    *(XGAB**2+XGAB*GAB*SUSY)
     .    *TQCDA(AMB**2/AMA**2)
       HB2=3.D0*AFF(AMA,(RMB/AMA)**2)
     .    *(XGAB**2+XGAB*GAB*SUSY)
     .    *QCDA(RMB**2/AMA**2)
     .   + DBB
       IF(HB2.LT.0.D0) HB2 = 0
       RAT = 2*AMB/AMA
       HBB = QQINT_HDEC(RAT,HB1,HB2)

c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c      write(6,('A3,4(1X,G15.8)'))'A: ',AMA,AMA,SUSY+2*DGAB,
c    .                             SUSY/(SUSY+2*DGAB)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

      ENDIF

c     print*,'A -> bb',hbb
c      print*
c      print*,'A -> bb:',hbb,gab,xgab,gab*susy,2*dgab,susy,2*dgab+susy
C  A --> TT :
      RATCOUP = 0
      IF(IONSH.EQ.0)THEN
       DLD=3D0
       DLU=4D0
       XM1 = 2D0*AMT-DLD
       XM2 = 2D0*AMT+DLU
       IF (AMA.LE.AMT+AMW+AMB) THEN
        HTT=0.D0
       ELSEIF (AMA.LE.XM1) THEN
        FACTT=6.D0*GF**2*AMA**3*AMT**2/2.D0/128.D0/PI**3*GAT**2
        call ATOTT_HDEC(ama,amt,amb,amw,amch,att0,gab,gat)
        HTT=FACTT*ATT0
       ELSEIF (AMA.LE.XM2) THEN
        XX(1) = XM1-1D0
        XX(2) = XM1
        XX(3) = XM2
        XX(4) = XM2+1D0
        FACTT=6.D0*GF**2*XX(1)**3*AMT**2/2.D0/128.D0/PI**3
        call ATOTT_HDEC(xx(1),amt,amb,amw,amch,att0,gab,gat)
        YY(1)=FACTT*ATT0
        FACTT=6.D0*GF**2*XX(2)**3*AMT**2/2.D0/128.D0/PI**3
        call ATOTT_HDEC(xx(2),amt,amb,amw,amch,att0,gab,gat)
        YY(2)=FACTT*ATT0
        XMT = RUNM_HDEC(XX(3),6)
        XYZ1 =3.D0*AFF(XX(3),(AMT/XX(3))**2)
     .    *TQCDA(AMT**2/XX(3)**2)
        XYZ2 =3.D0*AFF(XX(3),(XMT/XX(3))**2)
     .    *QCDA(XMT**2/XX(3)**2)
        IF(XYZ2.LT.0.D0) XYZ2 = 0
        RAT = 2*AMT/XX(3)
        YY(3) = QQINT_HDEC(RAT,XYZ1,XYZ2)
        XMT = RUNM_HDEC(XX(4),6)
        XYZ1 =3.D0*AFF(XX(4),(AMT/XX(4))**2)
     .    *TQCDA(AMT**2/XX(4)**2)
        XYZ2 =3.D0*AFF(XX(4),(XMT/XX(4))**2)
     .    *QCDA(XMT**2/XX(4)**2)
        IF(XYZ2.LT.0.D0) XYZ2 = 0
        RAT = 2*AMT/XX(4)
        YY(4) = QQINT_HDEC(RAT,XYZ1,XYZ2)
        HTT = FINT_HDEC(AMA,XX,YY)*GAT**2
       ELSE
        HT1=3.D0*AFF(AMA,(AMT/AMA)**2)*GAT**2
     .    *TQCDA(AMT**2/AMA**2)
        HT2=3.D0*AFF(AMA,(RMT/AMA)**2)*GAT**2
     .    *QCDA(RMT**2/AMA**2)
        IF(HT2.LT.0.D0) HT2 = 0
        RAT = 2*AMT/AMA
        HTT = QQINT_HDEC(RAT,HT1,HT2)
       ENDIF
      ELSE
       IF (AMA.LE.2.D0*AMT) THEN
        HTT=0.D0
       ELSE
        HT1=3.D0*AFF(AMA,(AMT/AMA)**2)*GAT**2
     .    *TQCDA(AMT**2/AMA**2)
        HT2=3.D0*AFF(AMA,(RMT/AMA)**2)*GAT**2
     .    *QCDA(RMT**2/AMA**2)
        IF(HT2.LT.0.D0) HT2 = 0
        RAT = 2*AMT/AMA
        HTT = QQINT_HDEC(RAT,HT1,HT2)
       ENDIF
      ENDIF

c     print*,'A -> tt',htt

C  A ---> GAMMA GAMMA
       EPS=1.D-8
       XRMC = RUNM_HDEC(AMA/2,4)*AMC/RUNM_HDEC(AMC,4)
       XRMB = RUNM_HDEC(AMA/2,5)*AMB/RUNM_HDEC(AMB,5)
       XRMT = RUNM_HDEC(AMA/2,6)*AMT/RUNM_HDEC(AMT,6)
       CTT = 4*XRMT**2/AMA**2*DCMPLX(1D0,-EPS)
       CTB = 4*XRMB**2/AMA**2*DCMPLX(1D0,-EPS)
       CAT = 4/3D0 * CTT*CF(CTT)*GAT
     .     * CFACQ_HDEC(1,AMA,XRMT)
       CAB = 1/3D0 * CTB*CF(CTB)*GAB
     .     * CFACQ_HDEC(1,AMA,XRMB)
       CTC = 4*XRMC**2/AMA**2*DCMPLX(1D0,-EPS)
       CAC = 4/3D0 * CTC*CF(CTC)*GAT
     .     * CFACQ_HDEC(1,AMA,XRMC)
       CTL = 4*AMTAU**2/AMA**2*DCMPLX(1D0,-EPS)
       CAL = 1.D0  * CTL*CF(CTL)*GAB
       if(i2hdm.eq.1) then
          CAL = 1.D0  * CTL*CF(CTL)*galep
       endif
       IF(IOFSUSY.EQ.0) THEN 
        CX1 = 4*AMCHAR(1)**2/AMA**2*DCMPLX(1D0,-EPS)
        CX2 = 4*AMCHAR(2)**2/AMA**2*DCMPLX(1D0,-EPS)
        CAX1= AMW/XMCHAR(1) * CX1*CF(CX1) * 2*AC3(1,1) 
        CAX2= AMW/XMCHAR(2) * CX2*CF(CX2) * 2*AC3(2,2) 
        XFAC = CDABS(CAT+CAB+CAC+CAL+CAX1+CAX2)**2
       ELSE 
        XFAC = CDABS(CAT+CAB+CAC+CAL)**2
       ENDIF
       HGA=GF/(32.D0*PI*DSQRT(2.D0))*AMA**3*(ALPH/PI)**2*XFAC

c      print*,'A -> gamgam',hga
C  A ---> Z GAMMA
      XRMC = RUNM_HDEC(AMA/2,4)*AMC/RUNM_HDEC(AMC,4)
      XRMB = RUNM_HDEC(AMA/2,5)*AMB/RUNM_HDEC(AMB,5)
      XRMT = RUNM_HDEC(AMA/2,6)*AMT/RUNM_HDEC(AMT,6)
c     print*,'xrmc,xrmb,xrmt ',xrmc,xrmb,xrmt
      IF(AMA.LE.AMZ)THEN
       HZGA=0
      ELSE
       TS = SS/CS
       FT = -3*2D0/3*(1-4*2D0/3*SS)/DSQRT(SS*CS)*GAT
       FB = 3*1D0/3*(-1+4*1D0/3*SS)/DSQRT(SS*CS)*GAB
       FC = -3*2D0/3*(1-4*2D0/3*SS)/DSQRT(SS*CS)*GAT
       FL = (-1+4*SS)/DSQRT(SS*CS)*GAB
       if(i2hdm.eq.1) then
          FL = (-1+4*SS)/DSQRT(SS*CS)*galep
       endif
       EPS=1.D-8
c      CTT = 4*XRMT**2/AMA**2*DCMPLX(1D0,-EPS)
c      CTB = 4*XRMB**2/AMA**2*DCMPLX(1D0,-EPS)
c      CTC = 4*XRMC**2/AMA**2*DCMPLX(1D0,-EPS)
       CTT = 4*AMT**2/AMA**2*DCMPLX(1D0,-EPS)
       CTB = 4*AMB**2/AMA**2*DCMPLX(1D0,-EPS)
       CTC = 4*AMC**2/AMA**2*DCMPLX(1D0,-EPS)
       CTL = 4*AMTAU**2/AMA**2*DCMPLX(1D0,-EPS)
c      CLT = 4*XRMT**2/AMZ**2*DCMPLX(1D0,-EPS)
c      CLB = 4*XRMB**2/AMZ**2*DCMPLX(1D0,-EPS)
c      CLC = 4*XRMC**2/AMZ**2*DCMPLX(1D0,-EPS)
       CLT = 4*AMT**2/AMZ**2*DCMPLX(1D0,-EPS)
       CLB = 4*AMB**2/AMZ**2*DCMPLX(1D0,-EPS)
       CLC = 4*AMC**2/AMZ**2*DCMPLX(1D0,-EPS)
       CLE = 4*AMTAU**2/AMZ**2*DCMPLX(1D0,-EPS)
       CAT = FT*(- CI2(CTT,CLT))
       CAB = FB*(- CI2(CTB,CLB))
       CAC = FC*(- CI2(CTC,CLC))
       CAL = FL*(- CI2(CTL,CLE))
c
c       CTC = 4*AMC**2/AMA**2*DCMPLX(1D0,-EPS)
c       CLC = 4*AMC**2/AMZ**2*DCMPLX(1D0,-EPS)
c       CAC = FT*(- CI2(CTC,CLC))
c
       XFAC = CDABS(CAT+CAB+CAC+CAL)**2
       ACOUP = DSQRT(2D0)*GF*AMZ**2*SS*CS/PI**2
       HZGA = GF/(4.D0*PI*DSQRT(2.D0))*AMA**3*(ALPH/PI)*ACOUP/16.D0
     .        *XFAC*(1-AMZ**2/AMA**2)**3
      ENDIF

c     print*,'A -> Zgam',hzga
C  A ---> h Z* ---> HFF
      IF(IONSH.EQ.0)THEN
       DLD=3D0
       DLU=5D0
       XM1 = AML+AMZ-DLD
       XM2 = AML+AMZ+DLU
       IF (AMA.LE.AML) THEN
        HAZ=0
       ELSEIF (AMA.LE.XM1) THEN
        IF (AMA.LE.DABS(AMZ-AML)) THEN
         HAZ=0
        ELSE
         HAZ=9.D0*GF**2/16.D0/PI**3*AMZ**4*AMA*GZAL**2*
     .      (7.D0/12.D0-10.D0/9.D0*SS+40.D0/27.D0*SS**2)
     .      *HVH((AML/AMA)**2,(AMZ/AMA)**2)
        ENDIF
       ELSEIF (AMA.LE.XM2) THEN
        XX(1) = XM1-1D0
        XX(2) = XM1
        XX(3) = XM2
        XX(4) = XM2+1D0
        YY(1)=9.D0*GF**2/16.D0/PI**3*AMZ**4*XX(1)*
     .      (7.D0/12.D0-10.D0/9.D0*SS+40.D0/27.D0*SS**2)
     .      *HVH((AML/XX(1))**2,(AMZ/XX(1))**2)
        YY(2)=9.D0*GF**2/16.D0/PI**3*AMZ**4*XX(2)*
     .      (7.D0/12.D0-10.D0/9.D0*SS+40.D0/27.D0*SS**2)
     .      *HVH((AML/XX(2))**2,(AMZ/XX(2))**2)
        CAZ=LAMB_HDEC(AML**2/XX(3)**2,AMZ**2/XX(3)**2)
     .     *LAMB_HDEC(XX(3)**2/AMZ**2,AML**2/AMZ**2)**2
        YY(3)=GF/8D0/DSQRT(2D0)/PI*AMZ**4/XX(3)*CAZ
        CAZ=LAMB_HDEC(AML**2/XX(4)**2,AMZ**2/XX(4)**2)
     .     *LAMB_HDEC(XX(4)**2/AMZ**2,AML**2/AMZ**2)**2
        YY(4)=GF/8D0/DSQRT(2D0)/PI*AMZ**4/XX(4)*CAZ
        HAZ = FINT_HDEC(AMA,XX,YY)*GZAL**2
       ELSE
        CAZ=LAMB_HDEC(AML**2/AMA**2,AMZ**2/AMA**2)
     .     *LAMB_HDEC(AMA**2/AMZ**2,AML**2/AMZ**2)**2
        HAZ=GF/8D0/DSQRT(2D0)/PI*AMZ**4/AMA*GZAL**2*CAZ
       ENDIF
      ELSE
       IF (AMA.LE.AMZ+AML) THEN
        HAZ=0
       ELSE
        CAZ=LAMB_HDEC(AML**2/AMA**2,AMZ**2/AMA**2)
     .     *LAMB_HDEC(AMA**2/AMZ**2,AML**2/AMZ**2)**2
        HAZ=GF/8D0/DSQRT(2D0)/PI*AMZ**4/AMA*GZAL**2*CAZ
       ENDIF
      ENDIF

c     print*,'A -> Zh',haz

c MMM changed 23/8/2013
C  A ---> H Z* ---> HFF
      if(i2hdm.eq.1) then
      IF(IONSH.EQ.0)THEN
       DLD=3D0
       DLU=5D0
       XM1 = AMH+AMZ-DLD
       XM2 = AMH+AMZ+DLU
       IF (AMA.LE.AMH) THEN
        HHAZ=0
       ELSEIF (AMA.LE.XM1) THEN
        IF (AMA.LE.DABS(AMZ-AMH)) THEN
         HHAZ=0
        ELSE
         HHAZ=9.D0*GF**2/16.D0/PI**3*AMZ**4*AMA*GZAH**2*
     .      (7.D0/12.D0-10.D0/9.D0*SS+40.D0/27.D0*SS**2)
     .      *HVH((AMH/AMA)**2,(AMZ/AMA)**2)
        ENDIF
       ELSEIF (AMA.LE.XM2) THEN
        XX(1) = XM1-1D0
        XX(2) = XM1
        XX(3) = XM2
        XX(4) = XM2+1D0
        YY(1)=9.D0*GF**2/16.D0/PI**3*AMZ**4*XX(1)*
     .      (7.D0/12.D0-10.D0/9.D0*SS+40.D0/27.D0*SS**2)
     .      *HVH((AMH/XX(1))**2,(AMZ/XX(1))**2)
        YY(2)=9.D0*GF**2/16.D0/PI**3*AMZ**4*XX(2)*
     .      (7.D0/12.D0-10.D0/9.D0*SS+40.D0/27.D0*SS**2)
     .      *HVH((AMH/XX(2))**2,(AMZ/XX(2))**2)
        CAZ=LAMB_HDEC(AMH**2/XX(3)**2,AMZ**2/XX(3)**2)
     .     *LAMB_HDEC(XX(3)**2/AMZ**2,AMH**2/AMZ**2)**2
        YY(3)=GF/8D0/DSQRT(2D0)/PI*AMZ**4/XX(3)*CAZ
        CAZ=LAMB_HDEC(AMH**2/XX(4)**2,AMZ**2/XX(4)**2)
     .     *LAMB_HDEC(XX(4)**2/AMZ**2,AMH**2/AMZ**2)**2
        YY(4)=GF/8D0/DSQRT(2D0)/PI*AMZ**4/XX(4)*CAZ
        HHAZ = FINT_HDEC(AMA,XX,YY)*GZAH**2
       ELSE
        CAZ=LAMB_HDEC(AMH**2/AMA**2,AMZ**2/AMA**2)
     .     *LAMB_HDEC(AMA**2/AMZ**2,AMH**2/AMZ**2)**2
        HHAZ=GF/8D0/DSQRT(2D0)/PI*AMZ**4/AMA*GZAH**2*CAZ
       ENDIF
      ELSE
       IF (AMA.LE.AMZ+AMH) THEN
        HHAZ=0
       ELSE
        CAZ=LAMB_HDEC(AMH**2/AMA**2,AMZ**2/AMA**2)
     .     *LAMB_HDEC(AMA**2/AMZ**2,AMH**2/AMZ**2)**2
        HHAZ=GF/8D0/DSQRT(2D0)/PI*AMZ**4/AMA*GZAH**2*CAZ
       ENDIF
      ENDIF
      endif

      if(i2hdm.eq.0) then
         HHAZ=0.D0
      endif

c     print*,'A -> ZH',hhaz

C  A ---> W+ H-
      if(i2hdm.eq.1) then
         if(ionsh.eq.0)then
            dld=3d0
            dlu=5d0
            xm1 = amw+amch-dld
            xm2 = amw+amch+dlu
            if (ama.lt.amch) then
               hawphm=0
            elseif (ama.le.xm1) then
               if(ama.le.dabs(amw-amch))then
                  hawphm=0
               else
                  hawphm=9.d0*gf**2/16.d0/pi**3*amw**4*ama
     .                 *hvh((amch/ama)**2,(amw/ama)**2)
               endif
            elseif (ama.lt.xm2) then
               xx(1) = xm1-1d0
               xx(2) = xm1
               xx(3) = xm2
               xx(4) = xm2+1d0
               yy(1) = 9.d0*gf**2/16.d0/pi**3*amw**4*xx(1)
     .              *hvh((amch/xx(1))**2,(amw/xx(1))**2)
               yy(2) = 9.d0*gf**2/16.d0/pi**3*amw**4*xx(2)
     .              *hvh((amch/xx(2))**2,(amw/xx(2))**2)
               cwh=lamb_hdec(amch**2/xx(3)**2,amw**2/xx(3)**2)
     .              *lamb_hdec(xx(3)**2/amw**2,amch**2/amw**2)**2
               yy(3)=gf/8.d0/dsqrt(2d0)/pi*amw**4/xx(3)*cwh
               cwh=lamb_hdec(amch**2/xx(4)**2,amw**2/xx(4)**2)
     .              *lamb_hdec(xx(4)**2/amw**2,amch**2/amw**2)**2
               yy(4)=gf/8.d0/dsqrt(2d0)/pi*amw**4/xx(4)*cwh
               hawphm = fint_hdec(ama,xx,yy)
            else
               cwh=lamb_hdec(amch**2/ama**2,amw**2/ama**2)
     .              *lamb_hdec(ama**2/amw**2,amch**2/amw**2)**2
               hawphm=gf/8.d0/dsqrt(2d0)/pi*amw**4/ama*cwh
            endif
         else
            if (ama.lt.amw+amch) then
               hawphm=0
            else
               cwh=lamb_hdec(amch**2/ama**2,amw**2/ama**2)
     .              *lamb_hdec(ama**2/amw**2,amch**2/amw**2)**2
               hawphm=gf/8.d0/dsqrt(2d0)/pi*amw**4/ama*cwh
            endif
         endif
      endif

      if(i2hdm.eq.0) then
         hawphm = 0.D0
      endif

      hawphm = 2.D0*hawphm

c     print*,'A -> W+H- + W-H+',hawphm,hawphm/2.D0
c end MMM changed 23/8/2013

C
C ========================== SUSY DECAYS  
C
      IF(IOFSUSY.EQ.0) THEN 
C  A ----> CHARGINOS
      DO 731 I=1,2
      DO 731 J=1,2
      IF (AMA.GT.AMCHAR(I)+AMCHAR(J)) THEN
      WHACH(I,J)=GF*AMW**2/(2*PI*DSQRT(2.D0))/AMA
     .     *LAMB_HDEC(AMCHAR(I)**2/AMA**2,AMCHAR(J)**2/AMA**2)
     .     *( (AC3(I,J)**2+AC3(J,I)**2)*(AMA**2-AMCHAR(I)
     .         **2-AMCHAR(J)**2)+4.D0*AC3(I,J)*AC3(J,I)* 
     .         XMCHAR(I)*XMCHAR(J) ) 
      ELSE 
      WHACH(I,J)=0.D0
      ENDIF
 731  CONTINUE
      WHACHT=WHACH(1,1)+WHACH(1,2)+WHACH(2,1)+WHACH(2,2)
C  A ----> NEUTRALINOS 
      DO 732 I=1,4
      DO 732 J=1,4
      IF (AMA.GT.AMNEUT(I)+AMNEUT(J)) THEN
      WHANE(I,J)=GF*AMW**2/(2*PI*DSQRT(2.D0))/AMA
     .         *AN3(I,J)**2*(AMA**2-(XMNEUT(I)-XMNEUT(J))**2)
     .         *LAMB_HDEC(AMNEUT(I)**2/AMA**2,AMNEUT(J)**2/AMA**2)
      ELSE 
      WHANE(I,J)=0.D0
      ENDIF
 732  CONTINUE
      WHANET= WHANE(1,1)+WHANE(1,2)+WHANE(1,3)+WHANE(1,4)
     .       +WHANE(2,1)+WHANE(2,2)+WHANE(2,3)+WHANE(2,4)
     .       +WHANE(3,1)+WHANE(3,2)+WHANE(3,3)+WHANE(3,4)
     .       +WHANE(4,1)+WHANE(4,2)+WHANE(4,3)+WHANE(4,4)

C  A ----> STAU'S 
C
      IF(AMA.GT.AMSL(1)+AMSL(2)) THEN
      WHASL=GF*AMZ**4/DSQRT(2.D0)/PI*GAEE**2*
     .      LAMB_HDEC(AMSL(1)**2/AMA**2,AMSL(2)**2/AMA**2)/AMA
      ELSE
      WHASL=0.D0
      ENDIF
C
C  A ----> STOPS 
C
      SUSY = 1
c     QSQ = (YMST(1)+YMST(2))/2
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     DO K=-10,10
c     QSQ = AMA*10.D0**(K/10.D0)
c     DO K=-1,1
c     QSQ = AMA*2.D0**(K)
c     IF(AMA.GT.YMST(1)+YMST(2)) THEN
c      CALL SQMBAPP_HDEC(QSQ)
c      SUSY = 1+SQSUSY_HDEC(3,1,1,2,QSQ,0,1)
c      WHAST0=3*GF*AMZ**4/DSQRT(2.D0)/PI*YATT**2*
c    .       LAMB_HDEC(YMST(1)**2/AMA**2,YMST(2)**2/AMA**2)/AMA
c      WHAST=3*GF*AMZ**4/DSQRT(2.D0)/PI*YATT**2*
c    .       LAMB_HDEC(YMST(1)**2/AMA**2,YMST(2)**2/AMA**2)/AMA
c    .      *SUSY
c     ELSE
c      WHAST=0.D0
c     ENDIF
c     write(9,*)'A -> t1 t2: ',QSQ/AMA,WHAST0,WHAST
c     write(903,('1X,G10.4,2(1X,G10.4)'))QSQ/AMA,WHAST0,WHAST
c     ENDDO
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      QSQ = AMA
      IF(AMA.GT.YMST(1)+YMST(2)) THEN
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
       CALL SQMBAPP_HDEC(QSQ)
       SUSY = 1+SQSUSY_HDEC(3,1,1,2,QSQ,0,1)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
       WHAST=3*GF*AMZ**4/DSQRT(2.D0)/PI*YATT**2*
     .       LAMB_HDEC(YMST(1)**2/AMA**2,YMST(2)**2/AMA**2)/AMA
     .      *SUSY
c      write(6,*)'A -> stop: ',AMA,AMST(1),AMST(2),100*(SUSY-1),'% ',
c    .           WHAST/SUSY,WHAST
c      write(712,*)AMA,WHAST/2,WHAST/SUSY/2
c      write(721,*)AMA,WHAST/2,WHAST/SUSY/2
c      write(6,*)'A -> stop: ',AMA,AMST(1),AMST(2),SUSY-1
      ELSE
      WHAST=0.D0
      ENDIF
C
C  A ----> SBOTTOMS 
C
      SUSY = 1
c     QSQ = (YMSB(1)+YMSB(2))/2
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     DO K=-10,10
c     QSQ = AMA*10.D0**(K/10.D0)
c     DO K=-1,1
c     QSQ = AMA*2.D0**(K)
c     IF(AMA.GT.YMSB(1)+YMSB(2)) THEN
c      CALL SQMBAPP_HDEC(QSQ)
c      SUSY = 1+SQSUSY_HDEC(3,2,1,2,QSQ,0,1)
c      WHASB0=3*GF*AMZ**4/DSQRT(2.D0)/PI*YABB**2*
c    .       LAMB_HDEC(YMSB(1)**2/AMA**2,YMSB(2)**2/AMA**2)/AMA
c      WHASB=3*GF*AMZ**4/DSQRT(2.D0)/PI*YABB**2*
c    .       LAMB_HDEC(YMSB(1)**2/AMA**2,YMSB(2)**2/AMA**2)/AMA
c    .      *SUSY
c     ELSE
c      WHASB=0.D0
c     ENDIF
c     write(9,*)'A -> b1 b2: ',QSQ/AMA,WHASB0,WHASB
c     write(904,('1X,G10.4,2(1X,G10.4)'))QSQ/AMA,WHASB0,WHASB
c     ENDDO
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      QSQ = AMA
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     dummy0 = 0
c     dummy1 = 0
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      IF(AMA.GT.YMSB(1)+YMSB(2)) THEN
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
       CALL SQMBAPP_HDEC(QSQ)
       SUSY = 1+SQSUSY_HDEC(3,2,1,2,QSQ,0,1)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
       WHASB=3*GF*AMZ**4/DSQRT(2.D0)/PI*YABB**2*
     .       LAMB_HDEC(YMSB(1)**2/AMA**2,YMSB(2)**2/AMA**2)/AMA
     .      *SUSY
c      write(6,*)'A -> sbot: ',AMA,YMSB(1),YMSB(2),100*(SUSY-1),'% ',
c    .           WHASB/SUSY,WHASB
c      write(812,*)AMA,WHASB/2,WHASB/SUSY/2
c      write(821,*)AMA,WHASB/2,WHASB/SUSY/2
c      write(6,*)'A -> sbot: ',AMA,AMSB(1),AMSB(2),SUSY-1
c      dummy0 = dummy0 + WHASB/SUSY
c      dummy1 = dummy1 + WHASB
      ELSE
      WHASB=0.D0
      ENDIF
c     write(6,*)'A -> sbot0: ',YMSB(1),YMSB(2),dummy0,dummy1,
c    .                        100*(dummy1/dummy0-1),'%'
C
      ELSE 
      WHACHT=0.D0
      WHANET=0.D0
      WHASL=0.D0
      WHAST=0.D0
      WHASB=0.D0
C--Change thanks to Elzbieta Richter-Was
      DO I=1,2
       DO J=1,2
        WHACH(I,J)=0.D0
       ENDDO
      ENDDO
      DO I=1,4
       DO J=1,4
        WHANE(I,J)=0.D0
       ENDDO
      ENDDO
      ENDIF

      IF(IGOLD.NE.0)THEN
C   HA ---> GOLDSTINOS
       DO 730 I=1,4
       IF (AMA.GT.AMNEUT(I)) THEN
        WHAGD(I)=AMA**5/AXMPL**2/AXMGD**2/48.D0/PI*
     .           (1.D0-AMNEUT(I)**2/AMA**2)**4*AGDA(I)**2
       ELSE
        WHAGD(I)=0.D0
       ENDIF
 730   CONTINUE
       WHAGDT=WHAGD(1)+WHAGD(2)+WHAGD(3)+WHAGD(4)
      ELSE
       WHAGDT=0
      ENDIF
C
C    ==========  TOTAL WIDTH AND BRANCHING RATIOS 
      WTOT=HLL+HMM+HSS+HCC+HBB+HGG+HGA+HZGA+HAZ+HTT
     .    +WHACHT+WHANET+WHASL+WHAST+WHASB + WHAGDT

c MMM changed 23/8/2013
      if(i2hdm.eq.1) then
         wtot = wtot + hhaz + hawphm
         abrhhaz = hhaz/wtot
         abrhawphm = hawphm/wtot
      endif
      if(i2hdm.eq.0) then
         abrhhaz = 0.D0
         abrhawphm = 0.D0
      endif
c end MMM changed 23/8/2013

c     print*,'wtot',wtot

      ABRT=HTT/WTOT
      ABRB=HBB/WTOT
      ABRL=HLL/WTOT
      ABRM=HMM/WTOT
      ABRS=HSS/WTOT
      ABRC=HCC/WTOT
      ABRG=HGG/WTOT
      ABRGA=HGA/WTOT
      ABRZGA=HZGA/WTOT
      ABRZ=HAZ/WTOT
      DO 831 I=1,2
      DO 831 J=1,2
      HABRSC(I,J)=WHACH(I,J)/WTOT
831   CONTINUE
      DO 832 I=1,4
      DO 832 J=1,4
      HABRSN(I,J)=WHANE(I,J)/WTOT
832   CONTINUE
      HABRCHT=WHACHT/WTOT      
      HABRNET=WHANET/WTOT      
      HABRSL=WHASL/WTOT 
      HABRST=WHAST/WTOT 
      HABRSB=WHASB/WTOT 
      HABRGD=WHAGDT/WTOT
C 
      AWDTH=WTOT

      BHASTAU = WHASL/WTOT
      BHASB = WHASB/WTOT
      BHAST = WHAST/WTOT

C    ==============================================================
      ENDIF

      RETURN
      END
 
      DOUBLE PRECISION FUNCTION BIJ_HDEC(X,Y)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DOUBLE PRECISION LAMB_HDEC
      BIJ_HDEC = (1-X-Y)/LAMB_HDEC(X,Y)*(
     .          4*SP_HDEC(XI_HDEC(X,Y)*XI_HDEC(Y,X))
     .        - 2*SP_HDEC(-XI_HDEC(X,Y)) - 2*SP_HDEC(-XI_HDEC(Y,X))
     .        + 2*DLOG(XI_HDEC(X,Y)*XI_HDEC(Y,X))
     .           *DLOG(1-XI_HDEC(X,Y)*XI_HDEC(Y,X))
     .        - DLOG(XI_HDEC(X,Y))*DLOG(1+XI_HDEC(X,Y))
     .        - DLOG(XI_HDEC(Y,X))*DLOG(1+XI_HDEC(Y,X))
     .          )
     .        -4*(DLOG(1-XI_HDEC(X,Y)*XI_HDEC(Y,X))
     .        +XI_HDEC(X,Y)*XI_HDEC(Y,X)/(1-XI_HDEC(X,Y)*XI_HDEC(Y,X))
     .          *DLOG(XI_HDEC(X,Y)*XI_HDEC(Y,X)))
     .        +(LAMB_HDEC(X,Y)+X-Y)/LAMB_HDEC(X,Y)*(DLOG(1+XI_HDEC(X,Y))
     .              - XI_HDEC(X,Y)/(1+XI_HDEC(X,Y))*DLOG(XI_HDEC(X,Y)))
     .        +(LAMB_HDEC(X,Y)-X+Y)/LAMB_HDEC(X,Y)*(DLOG(1+XI_HDEC(Y,X))
     .              - XI_HDEC(Y,X)/(1+XI_HDEC(Y,X))*DLOG(XI_HDEC(Y,X)))
      RETURN
      END

      DOUBLE PRECISION FUNCTION BETA_HDEC(X)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      BETA_HDEC=DSQRT(1.D0-4.D0*X)
      RETURN
      END

      DOUBLE PRECISION FUNCTION LAMB_HDEC(X,Y)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      LAMB_HDEC=DSQRT((1.D0-X-Y)**2-4.D0*X*Y)
      RETURN
      END

      DOUBLE PRECISION FUNCTION XI_HDEC(X,Y)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DOUBLE PRECISION LAMB_HDEC
      XI_HDEC = 2*X/(1-X-Y+LAMB_HDEC(X,Y))
      RETURN
      END

      DOUBLE PRECISION FUNCTION ELW4_HDEC(AMNUP,AMEP,AMTP,AMBP)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      COMMON/PARAM_HDEC/GF,ALPH,AMTAU,AMMUON,AMZ,AMW
      FF(X) = (1-X)*SP_HDEC(1-X)-X/(1-X)*DLOG(X)
     .      * (1+(3+X**2)/2/(1-X)*DLOG(X))
      ELW41(AMA,X,NC) = NC*GF*AMA**2/8/DSQRT(2.D0)/PI**2
     .                * (7*(1+X)/6+X*DLOG(X)/(1-X))
      ELW42(AMA,X,NC) = NC*GF**2*AMA**4/8/PI**4
     .                * ((25-29*X+25*X**2)/24
     .                 +(6-9*X+85*X**2-47*X**3+13*X**4)/96/(1-X)*DLOG(X)
     .                 +(1-X)**2/96/X*(13-14*X+13*X**2)*DLOG(DABS(1-X))
     .                 -X**3*(3-X**2)/16/(1-X)**2*DLOG(X)**2
     .                 +(1-X)/16/X*(1+X**3)*DLOG(X)*DLOG(DABS(1-X))
     .                 -3*(1-X**2)/8*SP_HDEC(1-X)
     .                 +NC*(11*(1+X)**2/128+13*X*(1+X)/96/(1-X)*DLOG(X)
     .                     +3*X**2/32/(1-X)**2*DLOG(X)**2))
      ELW42Q(AMA,X,NC) = NC*GF*AMA**2/8/DSQRT(2.D0)/PI**2 * CF*ALS/PI
     .                 * (-3*(1+X)/4-FF(X)/2)
      ELW410(AMA,NC) = NC*GF*AMA**2/8/DSQRT(2.D0)/PI**2 * 4/3.D0
      ELW420(AMA,NC) = NC*GF**2*AMA**4/8/PI**4 * (3+2*NC)/12
      ELW42Q0(AMA,NC) = NC*GF*AMA**2/8/DSQRT(2.D0)/PI**2*CF*ALS/PI*(-1)
      PI = 4*DATAN(1.D0)
      QQ = (AMTP+AMBP)/2
      ALS = ALPHAS_HDEC(QQ,3)
      CF = 4/3.D0
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     CF = 0
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      NC = 3
      IF(AMTP.NE.AMBP)THEN
       XQ1 = ELW41(AMBP,AMTP**2/AMBP**2,NC)
       XQ2 = ELW42(AMBP,AMTP**2/AMBP**2,NC)
     .     + ELW42Q(AMBP,AMTP**2/AMBP**2,NC)
      ELSE
       XQ1 = ELW410(AMBP,NC)
       XQ2 = ELW420(AMBP,NC) + ELW42Q0(AMBP,NC)
      ENDIF
      NC = 1
      IF(AMEP.NE.AMNUP)THEN
       XL1 = ELW41(AMEP,AMNUP**2/AMEP**2,NC)
       XL2 = ELW42(AMEP,AMNUP**2/AMEP**2,NC)
      ELSE
       XL1 = ELW410(AMEP,NC)
       XL2 = ELW420(AMEP,NC)
      ENDIF
      ELW4_HDEC=2*(XQ1+XL1) + 2*(XQ2+XL2) + (XQ1+XL1)**2
c     write(6,*)'SM4 elw.: ',(XQ1+XL1),(XQ2+XL2)
c     write(6,*)'SM4 elw. ff: ',ELW4_HDEC*100,2*(XQ1+XL1)*100,
c    .                          (2*(XQ2+XL2)+(XQ1+XL1)**2)*100
      RETURN
      END

      DOUBLE PRECISION FUNCTION ELW4V_HDEC(IV,AMNUP,AMEP,AMTP,AMBP)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      COMMON/PARAM_HDEC/GF,ALPH,AMTAU,AMMUON,AMZ,AMW
      ELW41U(AMA,X,NC) = NC*GF*AMA**2/8/DSQRT(2.D0)/PI**2
     .                 * (7*(1+X)/6+X*DLOG(X)/(1-X))
      ELW41V(AMA,X,NC) = NC*GF*AMA**2/8/DSQRT(2.D0)/PI**2*(-2)*(1+X)
      ELW42U(AMA,X,NC) = NC*GF**2*AMA**4/8/PI**4
     .                 * ((25-29*X+25*X**2)/24
     .                 +(6-9*X+85*X**2-47*X**3+13*X**4)/96/(1-X)*DLOG(X)
     .                  +(1-X)**2/96/X*(13-14*X+13*X**2)*DLOG(DABS(1-X))
     .                  -X**3*(3-X**2)/16/(1-X)**2*DLOG(X)**2
     .                  +(1-X)/16/X*(1+X**3)*DLOG(X)*DLOG(DABS(1-X))
     .                  -3*(1-X**2)/8*SP_HDEC(1-X)
     .                  +NC*(11*(1+X)**2/128+13*X*(1+X)/96/(1-X)*DLOG(X)
     .                      +3*X**2/32/(1-X)**2*DLOG(X)**2))
      ELW42W(AMA,X,NC) = NC*GF**2*AMA**4/8/PI**4
     .             * (-3*(1-X+X**2)/4-X**2*(10-5*X+X**2)/8/(1-X)*DLOG(X)
     .                -(1-X)**4/8/X*DLOG(DABS(1-X)))
      ELW42Z(AMA,X,NC) = NC*GF**2*AMA**4/8/PI**4
     .                 * (-15*(1-X)**2/16+X*(3-4*X+X**2)/8*DLOG(X)
     .                    -(1-X)**4/8/X*DLOG(DABS(1-X))
     .                    -NC*(1+X)/8*(1+X+2*X/(1-X)*DLOG(X)))
      ELW41U0(AMA,NC) = NC*GF*AMA**2/8/DSQRT(2.D0)/PI**2 * 4/3.D0
      ELW42U0(AMA,NC) = NC*GF**2*AMA**4/8/PI**4 * (3+2*NC)/12
      ELW41V0(AMA,NC) = NC*GF*AMA**2/8/DSQRT(2.D0)/PI**2*(-4)
      ELW42V0(AMA,NC) = 0
      FF(X) = (1-X)*SP_HDEC(1-X)-X/(1-X)*DLOG(X)
     .      * (1+(3+X**2)/2/(1-X)*DLOG(X))
      ELW42QW(AMA,X,NC) = NC*GF*AMA**2/8/DSQRT(2.D0)/PI**2*CF*ALS/PI
     .             * (9*(1+X)/4-3*X/(1-X)*DLOG(X)-FF(X)/2)
      ELW42QW0(AMA,NC) = NC*GF*AMA**2/8/DSQRT(2.D0)/PI**2*CF*ALS/PI * 8
      ELW42QZ(AMA,X,NC) = NC*GF*AMA**2/8/DSQRT(2.D0)/PI**2*CF*ALS/PI
     .             * (15*(1+X)/4-FF(X)/2)
      ELW42QZ0(AMA,NC) = NC*GF*AMA**2/8/DSQRT(2.D0)/PI**2*CF*ALS/PI * 8
      PI = 4*DATAN(1.D0)
      QQ = (AMTP+AMBP)/2
      ALS = ALPHAS_HDEC(QQ,3)
      CF = 4/3.D0
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     ALS = ALPHAS_HDEC(AMZ,3)
c     ALS = 0.119D0
c     write(6,*)'als = ',als
c     CF = 0
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      IF(IV.EQ.1)THEN
C--HWW
       NC = 3
       IF(AMTP.NE.AMBP)THEN
        XQ1 = ELW41U(AMBP,AMTP**2/AMBP**2,NC)
     .      + ELW41V(AMBP,AMTP**2/AMBP**2,NC)
        XQ2 = ELW42U(AMBP,AMTP**2/AMBP**2,NC)
     .      + ELW42W(AMBP,AMTP**2/AMBP**2,NC)
     .      + ELW42QW(AMBP,AMTP**2/AMBP**2,NC)
        XQU = ELW41U(AMBP,AMTP**2/AMBP**2,NC)
        XQV = ELW41V(AMBP,AMTP**2/AMBP**2,NC)
       ELSE
        XQ1 = ELW41U0(AMBP,NC) + ELW41V0(AMBP,NC)
        XQ2 = ELW42U0(AMBP,NC) + ELW42V0(AMBP,NC) + ELW42QW0(AMBP,NC)
        XQU = ELW41U0(AMBP,NC)
        XQV = ELW41V0(AMBP,NC)
       ENDIF
       NC = 1
       IF(AMEP.NE.AMNUP)THEN
        XL1 = ELW41U(AMEP,AMNUP**2/AMEP**2,NC)
     .      + ELW41V(AMEP,AMNUP**2/AMEP**2,NC)
        XL2 = ELW42U(AMEP,AMNUP**2/AMEP**2,NC)
     .      + ELW42W(AMEP,AMNUP**2/AMEP**2,NC)
        XLU = ELW41U(AMEP,AMNUP**2/AMEP**2,NC)
        XLV = ELW41V(AMEP,AMNUP**2/AMEP**2,NC)
       ELSE
        XL1 = ELW41U0(AMEP,NC) + ELW41V0(AMEP,NC)
        XL2 = ELW42U0(AMEP,NC) + ELW42V0(AMEP,NC)
        XLU = ELW41U0(AMEP,NC)
        XLV = ELW41V0(AMEP,NC)
       ENDIF
      ELSE
C--HZZ
       NC = 3
       IF(AMTP.NE.AMBP)THEN
        XQ1 = ELW41U(AMBP,AMTP**2/AMBP**2,NC)
     .      + ELW41V(AMBP,AMTP**2/AMBP**2,NC)
        XQ2 = ELW42U(AMBP,AMTP**2/AMBP**2,NC)
     .      + ELW42Z(AMBP,AMTP**2/AMBP**2,NC)
     .      + ELW42QZ(AMBP,AMTP**2/AMBP**2,NC)
        XQU = ELW41U(AMBP,AMTP**2/AMBP**2,NC)
        XQV = ELW41V(AMBP,AMTP**2/AMBP**2,NC)
       ELSE
        XQ1 = ELW41U0(AMBP,NC) + ELW41V0(AMBP,NC)
        XQ2 = ELW42U0(AMBP,NC) + ELW42V0(AMBP,NC) + ELW42QZ0(AMBP,NC)
        XQU = ELW41U0(AMBP,NC)
        XQV = ELW41V0(AMBP,NC)
       ENDIF
       NC = 1
       IF(AMEP.NE.AMNUP)THEN
        XL1 = ELW41U(AMEP,AMNUP**2/AMEP**2,NC)
     .      + ELW41V(AMEP,AMNUP**2/AMEP**2,NC)
        XL2 = ELW42U(AMEP,AMNUP**2/AMEP**2,NC)
     .      + ELW42Z(AMEP,AMNUP**2/AMEP**2,NC)
        XLU = ELW41U(AMEP,AMNUP**2/AMEP**2,NC)
        XLV = ELW41V(AMEP,AMNUP**2/AMEP**2,NC)
       ELSE
        XL1 = ELW41U0(AMEP,NC) + ELW41V0(AMEP,NC)
        XL2 = ELW42U0(AMEP,NC) + ELW42V0(AMEP,NC)
        XLU = ELW41U0(AMEP,NC)
        XLV = ELW41V0(AMEP,NC)
       ENDIF
      ENDIF
      X1 = XQ1 + XL1
      X2 = XQ2 + XL2 + (XQU + XLU) * (XQV + XLV)
      ELW4V_HDEC = 2*(X1 + X2) + X1**2
c     write(6,*)'SM4 elw. VV: ',ELW4V_HDEC*100,2*X1*100,
c    .                          ELW4V_HDEC*100-2*X1*100
c     write(6,*)'SM4:  '
c     write(6,*)'du1, dv1: ',
c    . ELW41U(AMBP,AMTP**2/AMBP**2,3)+ELW41U0(AMEP,1),
c    . ELW41V(AMBP,AMTP**2/AMBP**2,3)+ELW41V0(AMEP,1)
c     write(6,*)
c     write(6,*)'du2, dw2, dz2: ',
c    . ELW42U(AMBP,AMTP**2/AMBP**2,3)+ELW42U0(AMEP,1),
c    . ELW42W(AMBP,AMTP**2/AMBP**2,3)+ELW42V0(AMEP,1),
c    . ELW42Z(AMBP,AMTP**2/AMBP**2,3)+ELW42V0(AMEP,1)
c     write(6,*)
c     write(6,*)'dvtot1, dwtot2, dztot2: ',
c    . ELW41U(AMBP,AMTP**2/AMBP**2,3)+ELW41V(AMBP,AMTP**2/AMBP**2,3)
c    .+ELW41U0(AMEP,1) + ELW41V0(AMEP,1),
c    . ELW42U(AMBP,AMTP**2/AMBP**2,3)+ELW42W(AMBP,AMTP**2/AMBP**2,3)
c    .+ELW42QW(AMBP,AMTP**2/AMBP**2,3)+ELW42U0(AMEP,1)+ELW42V0(AMEP,1)
c    .+(ELW41U(AMBP,AMTP**2/AMBP**2,3)+ELW41U0(AMEP,1))
c    .*(ELW41V(AMBP,AMTP**2/AMBP**2,3)+ELW41V0(AMEP,1)),
c    . ELW42U(AMBP,AMTP**2/AMBP**2,3)+ELW42Z(AMBP,AMTP**2/AMBP**2,3)
c    .+ELW42QZ(AMBP,AMTP**2/AMBP**2,3)+ELW42U0(AMEP,1)+ELW42V0(AMEP,1)
c    .+(ELW41U(AMBP,AMTP**2/AMBP**2,3)+ELW41U0(AMEP,1))
c    .*(ELW41V(AMBP,AMTP**2/AMBP**2,3)+ELW41V0(AMEP,1))
c     write(6,*)
c     write(6,*)'dw2ew, dw2qcd, dz2ew, dz2qcd: ',
c    . ELW42U(AMBP,AMTP**2/AMBP**2,3)+ELW42W(AMBP,AMTP**2/AMBP**2,3)
c    .+ELW42U0(AMEP,1)+ELW42V0(AMEP,1)
c    .+(ELW41U(AMBP,AMTP**2/AMBP**2,3)+ELW41U0(AMEP,1))
c    .*(ELW41V(AMBP,AMTP**2/AMBP**2,3)+ELW41V0(AMEP,1)),
c    . ELW42QW(AMBP,AMTP**2/AMBP**2,3),
c    . ELW42U(AMBP,AMTP**2/AMBP**2,3)+ELW42Z(AMBP,AMTP**2/AMBP**2,3)
c    .+ELW42U0(AMEP,1)+ELW42V0(AMEP,1)
c    .+(ELW41U(AMBP,AMTP**2/AMBP**2,3)+ELW41U0(AMEP,1))
c    .*(ELW41V(AMBP,AMTP**2/AMBP**2,3)+ELW41V0(AMEP,1)),
c    . ELW42QZ(AMBP,AMTP**2/AMBP**2,3)
c     write(6,*)'dw2qcd, dz2qcd: ',
c    . ELW42QW(AMBP,AMTP**2/AMBP**2,3),
c    . ELW42QZ(AMBP,AMTP**2/AMBP**2,3)
c     write(6,*)'dw2qcd, dz2qcd: ',
c    . ELW42QW(AMTP,AMBP**2/AMTP**2,3),
c    . ELW42QZ(AMTP,AMBP**2/AMTP**2,3)
c     write(6,*)
      RETURN
      END

C *****************************************************************
C ************* SUBROUTINE FOR THE SUSY COUPLINGS *****************
C *****************************************************************
      SUBROUTINE SUSYCP_HDEC(TGBET)
      IMPLICIT DOUBLE PRECISION (A-H,M,O-Z)
      DOUBLE PRECISION LA1,LA2,LA3,LA4,LA5,LA6,LA7,LA3T
      COMPLEX*16 F0_HDEC
      DIMENSION MST(2),GLTT(2,2),GHTT(2,2),
     .          MSB(2),GLBB(2,2),GHBB(2,2)
      dimension itest(30)
      COMMON/FLAG_HDEC/IHIGGS,NNLO,IPOLE
      COMMON/MODEL_HDEC/IMODEL
      COMMON/MASSES_HDEC/AMS,AMC,AMB,AMT
      COMMON/HMASS_HDEC/AMSM,AMA,AML,AMH,AMCH,AMAR
      COMMON/HMASSR_HDEC/AMLR,AMHR
      COMMON/CHIMASS_HDEC/AMCHI
      COMMON/HSELF_HDEC/LA1,LA2,LA3,LA4,LA5,LA6,LA7
      COMMON/BREAK_HDEC/AMEL,AMER,AMSQ,AMUR,AMDR,AL,AU,AD,AMU,AM2
      COMMON/BREAKSCALE_HDEC/SUSYSCALE
      COMMON/BREAKGLU_HDEC/AMGLU
      COMMON/SFER1ST_HDEC/AMQL1,AMUR1,AMDR1,AMEL1,AMER1
      COMMON/PARAM_HDEC/GF,ALPH,AMTAU,AMMUON,AMZ,AMW
      COMMON/COUP_HDEC/GAT,GAB,GLT,GLB,GHT,GHB,GZAH,GZAL,
     .            GHHH,GLLL,GHLL,GLHH,GHAA,GLAA,GLVV,GHVV,
     .            GLPM,GHPM,B,A
      COMMON/GLUINO_HDEC/AMGLUINO,AMSB1,AMSB2,STHB,CTHB,
     .              XLBB(2,2),XHBB(2,2),XABB(2,2),
     .              AMST1,AMST2,STHT,CTHT,
     .              XLTT(2,2),XHTT(2,2),XATT(2,2)
      COMMON/ALS_HDEC/XLAMBDA,AMC0,AMB0,AMT0,N0
      COMMON/SLHA_vals_HDEC/islhai,islhao
      COMMON/SLHA_hmass_HDEC/slhaml,slhamh,slhamc,slha_alpha
      COMMON/SLHA_gaug_HDEC/slhaneut(4),slhaxneut(4),slhachar(2),
     .          slhau(2,2),slhav(2,2),slhaz(4,4),slhaxchar(2)
c MMM changed 21/8/13
      COMMON/THDM_HDEC/TGBET2HDM,ALPH2HDM,AMHL2HDM,AMHH2HDM,
     .     AMHA2HDM,AMHC2HDM,AM12SQ,A1LAM2HDM,A2LAM2HDM,A3LAM2HDM,
     .     A4LAM2HDM,A5LAM2HDM,ITYPE2HDM,I2HDM,IPARAM2HDM
      COMMON/THDM_TEST/itestcond
      COMMON/THDM_COUP_HDEC/gllep,ghlep,galep
c end MMM changed 21/8/13
      COMMON/HMSSM_HDEC/AMHL10
      FTRIANG(X,Y)= (x**2+y**2)/2.D0-x**2*y**2/(x**2-y**2)*
     .     dlog(x**2/y**2)
      FPTRIANG(X,Y)=-1.D0/3.D0*(4.D0/3.D0-(x**2*dlog(x**2)
     .     -y**2*dlog(y**2))/(x**2-y**2)-(x**2+y**2)/(x**2-y**2)**2*
     .     ((x**2+y**2)/2.D0-x**2*y**2/(x**2-y**2)*
     .     dlog(x**2/y**2)))
      BB1(P2,AM1,AM2,XMU)=((AM1**2-AM2**2)*B02_HDEC(0.D0,AM1,AM2,XMU**2)
     .           -(P2+AM1**2-AM2**2)*B02_HDEC(P2,AM1,AM2,XMU**2))/2/P2
      PMSQ1(QQ,AM,AMG,ALS) = CF*ALS/PI*(AMG**2*DLOG(QQ**2/AMG**2)
     . + AM**2/2*DLOG(AMG**2/AM**2) + AM**2/2 + 3*AMG**2/2
     . + (AMG**2-AM**2)**2/2/AM**2*DLOG(DABS(AMG**2-AM**2)/AMG**2))
      PMSQ10(QQ,AM,ALS) = CF*ALS/PI*(AM**2*DLOG(QQ**2/AM**2)
     . + AM**2/2 + 3*AM**2/2)
      ALS_SUSY(X,XLB,B0,B1)=12.D0*PI/(B0*DLOG(X**2/XLB**2))
     .          *(1.D0-B1*DLOG(DLOG(X**2/XLB**2))
     .           /DLOG(X**2/XLB**2))

      PI=4*DATAN(1D0)
      V=1.D0/DSQRT(DSQRT(2.D0)*GF)
      BET=DATAN(TGBET)
      SB = DSIN(BET)
      CB = DCOS(BET)
      AMAR = AMA
C  ============ TRANSFORMATION OF INPUT FOR SUBH ========== 
      CF = 4/3.D0
      CA = 3
c     Q0 = DSQRT(2*AMSQ**2+AMUR**2+AMDR**2)/2
      Q0 = SUSYSCALE
      ALSP = ALPHAS_HDEC(Q0,3)/PI
      ALTP = RUNM_HDEC(Q0,6)**2/2/PI/V**2/SB**2 / PI
      ALBP = RUNM_HDEC(Q0,5)**2/2/PI/V**2/CB**2 / PI
      RMT = RUNM_HDEC(AMT,6)
      RMB = RUNM_HDEC(AMT,5)
      QT = DSQRT(DMAX1(AMSQ**2+RMT**2,AMUR**2+RMT**2))
      QB = DSQRT(DMAX1(AMSQ**2+RMB**2,AMDR**2+RMB**2))
      AMH12 = AMA**2*SB**2 - AMZ**2/2*(CB**2-SB**2) - AMU**2
      AMH22 = AMA**2*CB**2 + AMZ**2/2*(CB**2-SB**2) - AMU**2
      XB = AMSQ**2 + AMDR**2 + AMH12 + AD**2
      XT = AMSQ**2 + AMUR**2 + AMH22 + AU**2
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     AD0 = AD + (CF*ALSP*AMGLU + 3*ALTP/2*AU + ALBP/4*AD)
c    .         * DLOG(QB**2/Q0**2)
c     AMDL0 = DSQRT(AMSQ**2 + (-CF*ALSP*AMGLU**2 + (ALTP*XT+ALBP*XB)/4)
c    .                      * DLOG(QB**2/Q0**2))
c     AMDR0 = DSQRT(AMDR**2 + (-CF*ALSP*AMGLU**2 + ALBP*XB/4)
c    .                      * DLOG(QB**2/Q0**2))
c     AU0 = AU + (CF*ALSP*AMGLU + ALTP/4*AU + 3*ALBP/2*AD)
c    .         * DLOG(QT**2/Q0**2)
c     AMUL0 = DSQRT(AMSQ**2 + (-CF*ALSP*AMGLU**2 + (ALTP*XT+ALBP*XB)/4)
c    .                      * DLOG(QT**2/Q0**2))
c     AMUR0 = DSQRT(AMUR**2 + (-CF*ALSP*AMGLU**2 + ALTP*XT/4)
c    .                      * DLOG(QT**2/Q0**2))
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     ADQCD = CF*ALSP*AMGLU*DLOG(QB**2/Q0**2)
c     SDQCD = -CF*ALSP*AMGLU**2*DLOG(QB**2/Q0**2)
c     AUQCD = CF*ALSP*AMGLU*DLOG(QT**2/Q0**2)
c     SUQCD = -CF*ALSP*AMGLU**2*DLOG(QT**2/Q0**2)
c     goto 8765
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      AMG = AMGLU
      QQ0 = Q0
      ALS = ALPHAS_HDEC(QQ0,3)
      SW2=1.D0-AMW**2/AMZ**2
C UP SQUARKS: 
      AMSQ0 = DSQRT(AMQL1**2+(0.5D0-2.D0/3.D0*SW2)*AMZ**2*DCOS(2.D0*B))
      IF(DABS(AMG).NE.AMSQ0)THEN
       AMU1L = DSQRT(AMSQ0**2 + PMSQ1(QQ0,AMSQ0,AMG,ALS))
      ELSE
       AMU1L = DSQRT(AMSQ0**2 + PMSQ10(QQ0,AMSQ0,ALS))
      ENDIF 
      AMSQ0 = DSQRT(AMUR1**2+2.D0/3.D0*SW2*AMZ**2*DCOS(2.D0*B))
      IF(DABS(AMG).NE.AMSQ0)THEN
       AMU1R = DSQRT(AMSQ0**2 + PMSQ1(QQ0,AMSQ0,AMG,ALS))
      ELSE
       AMU1R = DSQRT(AMSQ0**2 + PMSQ10(QQ0,AMSQ0,ALS))
      ENDIF
C DOWN SQUARKS
      AMSQ0 = DSQRT(AMQL1**2+(-0.5D0+1.D0/3.D0*SW2)*AMZ**2*DCOS(2.D0*B))
      IF(DABS(AMG).NE.AMSQ0)THEN
       AMD1L = DSQRT(AMSQ0**2 + PMSQ1(QQ0,AMSQ0,AMG,ALS))
      ELSE
       AMD1L = DSQRT(AMSQ0**2 + PMSQ10(QQ0,AMSQ0,ALS))
      ENDIF 
      AMSQ0 = DSQRT(AMDR1**2-1.D0/3.D0*SW2*AMZ**2*DCOS(2.D0*B))
      IF(DABS(AMG).NE.AMSQ0)THEN
       AMD1R = DSQRT(AMSQ0**2 + PMSQ1(QQ0,AMSQ0,AMG,ALS))
      ELSE
       AMD1R = DSQRT(AMSQ0**2 + PMSQ10(QQ0,AMSQ0,ALS))
      ENDIF 
      EPS = 0
      FFB = AMB*2*STHB*CTHB/AMG
      FFT = AMT*2*STHT*CTHT/AMG
      AM3 = AMG*(1-ALPHAS_HDEC(DABS(AMG),3)/4/PI
     .         *(4*CA+3*CA*DLOG(QQ0**2/AMG**2)
     .          + BB1(AMG**2,EPS,AMU1L,QQ0)
     .          + BB1(AMG**2,EPS,AMU1R,QQ0)
     .          + BB1(AMG**2,EPS,AMD1L,QQ0)
     .          + BB1(AMG**2,EPS,AMD1R,QQ0)
     .          + BB1(AMG**2,AMS,AMD1L,QQ0)
     .          + BB1(AMG**2,AMS,AMD1R,QQ0)
     .          + BB1(AMG**2,AMC,AMU1L,QQ0)
     .          + BB1(AMG**2,AMC,AMU1R,QQ0)
     .          + BB1(AMG**2,AMB,AMSB1,QQ0)
     .          + BB1(AMG**2,AMB,AMSB2,QQ0)
     .          + BB1(AMG**2,AMT,AMST1,QQ0)
     .          + BB1(AMG**2,AMT,AMST2,QQ0)
     .          + FFB*(B02_HDEC(AMG**2,AMB,AMSB1,QQ0**2)
     .                -B02_HDEC(AMG**2,AMB,AMSB2,QQ0**2))
     .          + FFT*(B02_HDEC(AMG**2,AMT,AMST1,QQ0**2)
     .                -B02_HDEC(AMG**2,AMT,AMST2,QQ0**2))
     .          ))
      ALS_SUSY0= ALPHAS_HDEC(QQ0,3)*(1+ALPHAS_HDEC(QQ0,3)/PI
     .         * (DLOG(QQ0**2/AMT**2)/6 + DLOG(QQ0**2/AMG**2)/2
     .         + (2*(DLOG(QQ0**2/AMU1L**2)+DLOG(QQ0**2/AMU1R**2)
     .              +DLOG(QQ0**2/AMD1L**2)+DLOG(QQ0**2/AMD1R**2))
     .           +DLOG(QQ0**2/AMSB1**2)+DLOG(QQ0**2/AMSB2**2)
     .           +DLOG(QQ0**2/AMST1**2)+DLOG(QQ0**2/AMST2**2))/24))
      ACC = 1.D-10
      XLB_SUSY = XITSUSY_HDEC(QQ0,ALS_SUSY0,ACC)
      B0 = 9
      B1 = 14.D0/9
      ALS0 = ALS_SUSY(QQ0,XLB_SUSY,B0,B1)
      ALST = ALS_SUSY(QT,XLB_SUSY,B0,B1)
      ALSB = ALS_SUSY(QB,XLB_SUSY,B0,B1)
      AUQCD = AM3*(-16.D0/9*(ALST/ALS0-1)*(1+ALS0/6/PI)
     .             -16*ALS0/27/PI*(ALST**2/ALS0**2-1))
      ADQCD = AM3*(-16.D0/9*(ALSB/ALS0-1)*(1+ALS0/6/PI)
     .             -16*ALS0/27/PI*(ALSB**2/ALS0**2-1))
      SUQCD = AM3**2*(8.D0/9*(ALST**2/ALS0**2-1)*(1+ALS0/6/PI)
     .               -4*ALS0/9/PI*(ALST**3/ALS0**3-1))
      SDQCD = AM3**2*(8.D0/9*(ALSB**2/ALS0**2-1)*(1+ALS0/6/PI)
     .               -4*ALS0/9/PI*(ALSB**3/ALS0**3-1))
8765  continue
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      AD0 = AD + ADQCD + (3*ALTP/2*AU + ALBP/4*AD)*DLOG(QB**2/Q0**2)
      AMDL0 = DSQRT(AMSQ**2 + SDQCD
     .           + ((ALTP*XT+ALBP*XB)/4)*DLOG(QB**2/Q0**2))
      AMDR0 = DSQRT(AMDR**2 + SDQCD
     .           + ALBP*XB/4*DLOG(QB**2/Q0**2))
      AU0 = AU + AUQCD + (ALTP/4*AU + 3*ALBP/2*AD)*DLOG(QT**2/Q0**2)
      AMUL0 = DSQRT(AMSQ**2 + SUQCD
     .           + (ALTP*XT+ALBP*XB)/4*DLOG(QT**2/Q0**2))
      AMUR0 = DSQRT(AMUR**2 + SUQCD + ALTP*XT/4*DLOG(QT**2/Q0**2))
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     write(6,*)
c     write(6,*)
c     write(6,*)'ALS param: ',XLAMBDA,AMC0,AMB0,AMT0,N0
c     write(6,*)'Q0,QT,QB: ',Q0,QT,QB
c     write(6,*)'ALSP,ALTP,ALBP: ',ALSP,ALTP,ALBP
c     write(6,*)'stop params:    ',RUNM_HDEC(Q0,6),V,SB
c     write(6,*)'sbot params:    ',RUNM_HDEC(Q0,5),V,CB
c     write(6,*)'A_t params: ',CF*ALSP*AMGLU,ALTP/4*AU,3*ALBP/2*AD,
c    .                         DLOG(QT**2/Q0**2)
c     write(6,*)'stop: ',AMSQ,AMUR,AU
c     write(6,*)'      ',AMUL0,AMUR0,AU0
c     write(6,*)'sbot: ',AMSQ,AMDR,AD
c     write(6,*)'      ',AMDL0,AMDR0,AD0
c     write(6,*)'      ',AMDR**2,-CF*ALSP*AMGLU**2 * DLOG(QB**2/Q0**2),
c    .                   ALBP*XB/4 * DLOG(QB**2/Q0**2)
c     write(6,*)'      ',AMGLU
c     write(6,*)
c     write(6,*)AMDR**2,SDQCD,ALBP*XB/4*DLOG(QB**2/Q0**2)
c     write(6,*)AMG,AM3
c     write(6,*)AMST1,AMST2,AMSB1,AMSB2
C  ============ HEAVIEST CHARGINO MASS NEEDED FOR SUBH ========== 
      if(islhai.eq.0) then
         AMCHI2=AM2**2+AMU**2+2.D0*AMW**2+DSQRT((AM2**2-AMU**2)**2
     .        +4.D0*AMW**4*DCOS(2.D0*BET)**2+4.D0*AMW**2*
     .        (AM2**2+AMU**2+2.D0*AMU*AM2*DSIN(2.D0*BET) ) ) 
         AMCHI=DSQRT(0.5D0*AMCHI2)
      else
         amchi = slhachar(2)
      endif
C ===============================================================
C ========== RUNNING MASSES
      if(islhai.eq.0) then
      IF(IMODEL.EQ.1)THEN
       CALL SUBH1_HDEC(AMA,TGBET,AMUL0,AMDL0,AMUR0,AMDR0,AMT,AU0,AD0,
     .            AMU,AMCHI,AMLR,AMHR,AMCH,SA,CA,TANBA,AMGLU)
      ELSEIF(IMODEL.EQ.2)THEN
       CALL SUBH2_HDEC(AMA,TGBET,AMUL0,AMUR0,AMT,AU0,AD0,AMU,
     .            AMLR,AMHR,AMCH,SA,CA,TANBA)
      ELSEIF(IMODEL.EQ.3)THEN
       CALL HABER(TGBET,SA,CA)
       AMLR = AML
       AMHR = AMH
      ELSEIF(IMODEL.EQ.4)THEN
C--Use Carena et al. for everything not included in FeynHiggs....
       CALL SUBH1_HDEC(AMA,TGBET,AMUL0,AMDL0,AMUR0,AMDR0,AMT,AU0,AD0,
     .            AMU,AMCHI,AMLR,AMHR,AMCH,SA,CA,TANBA,AMGLU)
       IF(CTHT.GE.0.D0)THEN
        XMST1 = AMST1
        XMST2 = AMST2
        STT = STHT
       ELSE
        XMST1 = AMST1
        XMST2 = AMST2
        STT = CTHT
       ENDIF
       IF(CTHB.GE.0.D0)THEN
        XMSB1 = AMSB1
        XMSB2 = AMSB2
        STB = STHB
       ELSE
        XMSB1 = AMSB1
        XMSB2 = AMSB2
        STB = CTHB
       ENDIF
       CALL FEYNHIGGS(AMA,TGBET,AMT,XMST1,XMST2,STT,XMSB1,
     .                XMSB2,STB,AMU,AMGLU,AM2,AMLR,AMHR,SA,CA)
      ELSEIF(IMODEL.EQ.10)THEN
c--hMSSM?
       aml = amhl10
       c2b = cb**2-sb**2
       amh = dsqrt(((ama**2+amz**2-aml**2)*(amz**2*cb**2+ama**2*sb**2)
     .       -ama**2*amz**2*c2b**2)/(amz**2*cb**2+ama**2*sb**2-aml**2))
       amch = dsqrt(ama**2+amw**2)
       a = -datan((amz**2+ama**2)*sb*cb
     .            /(amz**2*cb**2+ama**2*sb**2-aml**2))
       amlr = aml
       amhr = amh
       sa = dsin(a)
       ca = dcos(a)
       deps = (aml**2*(ama**2+amz**2-aml**2)-ama**2*amz**2*c2b**2)
     .      / (amz**2*cb**2+ama**2*sb**2-aml**2)
      ENDIF
      else
       CALL SUBH1_HDEC(AMA,TGBET,AMUL0,AMDL0,AMUR0,AMDR0,AMT,AU0,AD0,
     .            AMU,AMCHI,AMLR,AMHR,AMCH,SA,CA,TANBA,AMGLU)
       amlr = slhaml
       amhr = slhamh
       aml  = slhaml
       amh  = slhamh
       amch = slhamc
       sa   = dsin(slha_alpha)
       ca   = dcos(slha_alpha)
      endif
c MMM changed 21/8/13
      if(I2HDM.eq.1) then
         ammh2 = AM12SQ/sb/cb
         bb = datan(tgbet2hdm)
         cb = dcos(bb)
         sb = dsin(bb)

         if(iparam2hdm.eq.1) then
            sa = dsin(alph2hdm)
            ca = dcos(alph2hdm)
            amar = AMHA2HDM
            ama = amar
            amlr= AMHL2HDM
            amhr= AMHH2HDM
            aml = AMHL2HDM
            amh = AMHH2HDM
            amch= AMHC2HDM
            amar= AMHA2HDM
c from hep-ph/0408364, Eqs. (26)-(30)
            LA1 = 1.D0/v**2/cb**2*(-sb**2*ammh2+sa**2*amlr**2
     .           +ca**2*amhr**2)
            LA2 = 1.D0/v**2/sb**2*(-cb**2*ammh2+ca**2*amlr**2
     .           +sa**2*amhr**2)
            LA3 = -ammh2/v**2+2.D0*amch**2/v**2+1.D0/v**2*sa*ca/sb/cb*
     .           (amhr**2-amlr**2)
            LA4 = 1.D0/v**2*(ammh2+amar**2-2.D0*amch**2)
            LA5 = 1.D0/v**2*(ammh2-amar**2)
            LA6 = 0.D0
            LA7 = 0.D0
c            print*,'lambdas',la1,la2,la3,la4,la5
         elseif(iparam2hdm.eq.2) then
            la1=a1lam2hdm
            la2=a2lam2hdm
            la3=a3lam2hdm
            la4=a4lam2hdm
            la5=a5lam2hdm
            LA6 = 0.D0
            LA7 = 0.D0
c            print*,'lambdas',la1,la2,la3,la4,la5

            am11h2 = (la1*cb**4+la2*sb**4+2.D0*(la3+la4+la5)*
     .           cb**2*sb**2)*V**2
            am12h2 = (-la1*cb**2+la2*sb**2+(la3+la4+la5)*
     .           (cb**2-sb**2))*cb*sb*V**2
            am22h2 = ammh2 + 1.D0/8.D0*(la1+la2-2.D0*(la3+la4+la5))*
     .           (1.D0-dcos(4.D0*bb))*V**2

            aminb = 1.D0/2.D0*datan(2.D0*am12h2/(am11h2-am22h2))
            alph2hdm = aminb + bb
            sa = dsin(alph2hdm)
            ca = dcos(alph2hdm)

            amlch2 = (dsin(alph2hdm-bb))**2*am11h2
     .           -dsin(2.D0*(alph2hdm-bb))*am12h2 
     .           + (dcos(alph2hdm-bb))**2*am22h2
            
            amhch2 = (dcos(alph2hdm-bb))**2*am11h2
     .           +dsin(2.D0*(alph2hdm-bb))*am12h2 
     .           + (dsin(alph2hdm-bb))**2*am22h2

            amlr = dsqrt(amlch2)
            amhr = dsqrt(amhch2)

            if(amlr.gt.amhr) then
               alph2hdm = alph2hdm - Pi/2.D0 
               sa = dsin(alph2hdm)
               ca = dcos(alph2hdm)
               tmp1 = amlr
               tmp2 = amhr
               amhr = tmp1
               amlr = tmp2
            endif

            aml  = amlr
            amh  = amhr

            amach2 = ammh2-la5*V**2
            amcch2 = ammh2-1.D0/2.D0*(la4+la5)*V**2

            amar = dsqrt(amach2)
            amch = dsqrt(amcch2)

c            print*,'vals',aml,amh,amar,amch,alph2hdm
         endif

         if(itestcond.eq.1) then
c test of vacuum stability (S.Kanemura eal, Phys.Rev.D70(2004)115002)
            do i=1,30,1
               itest(i) = 0
            end do
            if(la1.gt.0.D0) then 
               itest(1)=1
            endif
            if(la1.gt.0.D0) then 
               itest(2)=1
            endif
            if(0.D0.le.la4+la5.and.0.D0.le.la4-la5) then
               amin=0.D0
            elseif(la4+la5.le.0.D0.and.la4+la5.le.la4-la5) then
               amin=la4+la5
            elseif(la4-la5.le.0.D0.and.la4-la5.le.la4+la5) then
               amin=la4-la5
            endif
            valtest = dsqrt(la1*la2)+la3+amin
            if(valtest.gt.0.D0) then
               itest(3)=1
            endif
c            print*,'check',amin,itest(1),itest(2),itest(3)
            iteststab=itest(1)*itest(2)*itest(3)
            if(iteststab.eq.0) then
               print*,'Attention: vacuum stability is not fulfilled.'
            endif

c test for perturbativity (arXiv:1106.0034, Eqs.(3.363)-(3.372))
            pap = 1.5D0*(la1+la2)+dsqrt(9.D0/4.D0*(la1-la2)**2+
     .           (2.D0*la3+la4)**2)
            pam = 1.5D0*(la1+la2)-dsqrt(9.D0/4.D0*(la1-la2)**2+
     .           (2.D0*la3+la4)**2)
            pbp = 0.5D0*(la1+la2)+0.5D0*dsqrt((la1-la2)**2+4.D0*la4**2)
            pbm = 0.5D0*(la1+la2)-0.5D0*dsqrt((la1-la2)**2+4.D0*la4**2)
            pcp = 0.5D0*(la1+la2)+0.5D0*dsqrt((la1-la2)**2+4.D0*la5**2)
            pcm = 0.5D0*(la1+la2)-0.5D0*dsqrt((la1-la2)**2+4.D0*la5**2)
            pe1 = la3+2.D0*la4-3.D0*la5
            pe2 = la3-la5
            pfp = la3+2.D0*la4+3.D0*la5
            pfm = la3+la5
            pf1 = la3+la4
            pp1 = la3-la4

            pertval = 8.D0*Pi

            if(dabs(pap).lt.pertval) then 
               itest(4) = 1
            endif
            if(dabs(pam).lt.pertval) then 
               itest(5) = 1
            endif
            if(dabs(pbp).lt.pertval) then 
               itest(6) = 1
            endif
            if(dabs(pbm).lt.pertval) then 
               itest(7) = 1
            endif
            if(dabs(pcp).lt.pertval) then 
               itest(8) = 1
            endif
            if(dabs(pcm).lt.pertval) then 
               itest(9) = 1
            endif
            if(dabs(pe1).lt.pertval) then 
               itest(10) = 1
            endif
            if(dabs(pe2).lt.pertval) then 
               itest(11) = 1
            endif
            if(dabs(pfp).lt.pertval) then 
               itest(12) = 1
            endif
            if(dabs(pfm).lt.pertval) then 
               itest(13) = 1
            endif
            if(dabs(pf1).lt.pertval) then 
               itest(14) = 1
            endif
            if(dabs(pp1).lt.pertval) then 
               itest(15) = 1
            endif
            itestpert = itest(4)*itest(5)*itest(6)*itest(7)*itest(8)*
     .           itest(9)*itest(10)*itest(11)*itest(12)
            if(itestpert.eq.0) then
            print*,'Attention: perturbative unitarity is not fulfilled.'
            endif

c the S- and T-parameter in the 2HDM (1108.3297, Eqs. (12),(13))
            params2hdm = -1.D0/(4.D0*Pi)*(1.D0/3.D0*dlog(amch**2)
     .           -(dsin(bb-alph2hdm))**2*fptriang(amh,amar)
     .           -(dcos(bb-alph2hdm))**2*fptriang(aml,amar))
            paramt2hdm = -dsqrt(2.D0)*gf/16.D0/Pi**2/alph*(
     .           -ftriang(amar,amch)+(dsin(bb-alph2hdm))**2*(
     .           ftriang(amh,amar)-ftriang(amh,amch))
     .           +(dcos(bb-alph2hdm))**2*(ftriang(aml,amar)
     .           -ftriang(aml,amch)))
c            print*,'par S,T',params2hdm,paramt2hdm

c the S- and T-parameter limits are taken from 1209.2716, Eq.(9)

            tparamlimp = 0.17D0
            tparamlimm = -0.07D0

            sparamlimp = 0.13D0
            sparamlimm = -0.07D0

          if(params2hdm.gt.sparamlimp.or.params2hdm.lt.sparamlimm) then
             print*,'The limits on the S-parameter are not fulfilled.'
          endif
          if(paramt2hdm.gt.tparamlimp.or.paramt2hdm.lt.tparamlimm) then
             print*,'The limits on the T-parameter are not fulfilled.'
          endif

         endif

c -- this is for a check --
c         print*,'alphacheck',alphacheck-2.D0*datan(1.D0)
c         print*,'alpha',datan(dsin(alph2hdm)/dcos(alph2hdm))
c         print*,'alpha2hdm',alph2hdm
c         print*,'check: aml,amh,ama,amch',amlc,amhc,amac,amcc
c         print*,'m12h2',am12sq,am12sq/sb/cb,ammh2
      endif
c end MMM changed 21/8/13
      LA3T=LA3+LA4+LA5
      AMA2=AMAR**2
      AML2=AMLR**2
      AMH2=AMHR**2
      AMP2=AMCH**2
C ========== HIGGS COUPLINGS 
      SBMA = SB*CA-CB*SA
      CBMA = CB*CA+SB*SA
      SBPA = SB*CA+CB*SA
      CBPA = CB*CA-SB*SA
      S2A = 2*SA*CA
      C2A = CA**2-SA**2
      S2B = 2*SB*CB
      C2B = CB**2-SB**2
      GLZZ = 1/V/2*AML2*SBMA
      GHZZ = 1/V/2*AMH2*CBMA
      GLWW = 2*GLZZ
      GHWW = 2*GHZZ
      GLAZ = 1/V*(AML2-AMA2)*CBMA
      GHAZ = -1/V*(AMH2-AMA2)*SBMA
      GLPW = -1/V*(AMP2-AML2)*CBMA
      GLMW = GLPW
      GHPW = 1/V*(AMP2-AMH2)*SBMA
      GHMW = GHPW
      GAPW = 1/V*(AMP2-AMA2)
      GAMW = -GAPW
      GHHH = V/2*(LA1*CA**3*CB + LA2*SA**3*SB + LA3T*SA*CA*SBPA
     .     + LA6*CA**2*(3*SA*CB+CA*SB) + LA7*SA**2*(3*CA*SB+SA*CB))
      GLLL = -V/2*(LA1*SA**3*CB - LA2*CA**3*SB + LA3T*SA*CA*CBPA
     .     - LA6*SA**2*(3*CA*CB-SA*SB) + LA7*CA**2*(3*SA*SB-CA*CB))
      GLHH = -3*V/2*(LA1*CA**2*CB*SA - LA2*SA**2*SB*CA
     .     + LA3T*(SA**3*CB-CA**3*SB+2*SBMA/3)
     .     - LA6*CA*(CB*C2A-SA*SBPA) - LA7*SA*(C2A*SB+CA*SBPA))
      GHLL = 3*V/2*(LA1*SA**2*CB*CA + LA2*CA**2*SB*SA
     .     + LA3T*(SA**3*SB+CA**3*CB-2*CBMA/3)
     .     - LA6*SA*(CB*C2A+CA*CBPA) + LA7*CA*(C2A*SB+SA*CBPA))
      GLAA = -V/2*(LA1*SB**2*CB*SA - LA2*CB**2*SB*CA
     .     - LA3T*(SB**3*CA-CB**3*SA) + 2*LA5*SBMA
     .     - LA6*SB*(CB*SBPA+SA*C2B) - LA7*CB*(C2B*CA-SB*SBPA))
      GHAA = V/2*(LA1*SB**2*CB*CA + LA2*CB**2*SB*SA
     .     + LA3T*(SB**3*SA+CB**3*CA) - 2*LA5*CBMA
     .     - LA6*SB*(CB*CBPA+CA*C2B) + LA7*CB*(SB*CBPA+SA*C2B))
      GLPM = 2*GLAA + V*(LA5 - LA4)*SBMA

c --- this is for a check against hep-ph/0408364 ---

      aa = alph2hdm

      xlll = -1.D0/4.D0/V/(2.D0*cb*sb)*(
     .     (dcos(3.D0*aa-bb)+3.D0*dcos(aa+bb))*amlr**2
     .     -4.D0*(dcos(aa-bb))**2*dcos(aa+bb)*ammh2)

      xlhh = -dsin(aa-bb)/2.D0/V/dsin(2.D0*bb)*(dsin(2.D0*aa)*
     .     (amlr**2+2.D0*amhr**2)-(3.D0*dsin(2.D0*aa)+dsin(2.D0*bb))*
     .     ammh2)

      xhll = -dcos(aa-bb)/2.D0/V/dsin(2.D0*bb)*(dsin(2.D0*aa)*(2.D0*
     .     amlr**2+amhr**2)-(3.D0*dsin(2.D0*aa)-dsin(2.D0*bb))*ammh2)

      xhhh = 1.D0/4.D0/V/dsin(2.D0*b)*((dsin(3.D0*aa-bb)
     .     -3.D0*dsin(aa+bb))*amhr**2+4.D0*(dsin(aa-bb))**2*
     .     dsin(aa+bb)*ammh2)

      xlaa = -1.D0/4.D0/V/dsin(2.D0*bb)*((dcos(aa-3.D0*bb)+3.D0*
     .     dcos(aa+bb))*
     .     amlr**2-4.D0*dsin(2.D0*bb)*dsin(aa-bb)*amar**2-4.D0*
     .     dcos(aa+bb)*ammh2)

      xhaa = -1.D0/4.D0/V/dsin(2.D0*bb)*((dsin(aa-3.D0*bb)+3.D0*
     .     dsin(aa+bb))*amhr**2+4.D0*dsin(2.D0*bb)*dcos(aa-bb)*amar**2
     .     -4.D0*dsin(aa+bb)*ammh2)

      xlpm = -1.D0/2.D0/V/dsin(2.D0*bb)*((dcos(aa-3.D0*bb)+3.D0*
     .     dcos(aa+bb))*
     .     amlr**2-4.D0*dsin(2.D0*bb)*dsin(aa-bb)*amch**2-4.D0*
     .     dcos(aa+bb)*ammh2)

      xhpm = -1.D0/2.D0/V/dsin(2.D0*bb)*((dsin(aa-3.D0*bb)+3.D0*
     .     dsin(aa+bb))*amhr**2+4.D0*dsin(2.D0*bb)*dcos(aa-bb)*
     .     amch**2-4.D0*dsin(aa+bb)*ammh2)

c      print*,'lll',glll,xlll
c      print*,'lhh',glhh,xlhh
c      print*,'hll',ghll,xhll
c      print*,'hhh',ghhh,xhhh
c      print*,'laa',glaa,xlaa
c      print*,'haa',ghaa,xhaa
c      print*,'lH+H-',glpm,xlpm
c      print*,'MA',amar,amch
c      print*,'Mh,MH,MA,MH+',amlr,amhr,amar,amch
c      print*,'alpha,tan(beta)',aa,tgbet2hdm

c ---------------------------------------------------

c      print*,'la1-la5',la1,la2,la3,la4,la5
c      print*,'glaa',glaa
c      print*,'sbma,sb,cb,sa,ca',sbma,sb,cb,sa,ca
c     write(6,*)'GLAA,SB,CB,SA,CA,SBPA,C2B,LA1,LA2,LA3T,LA5,LA6,LA7: ',
c    .           GLAA,SB,CB,SA,CA,SBPA,C2B,LA1,LA2,LA3T,LA5,LA6,LA7
c     write(6,*)'GLPM,GLAA,V,LA5,LA4,SBMA,LA5-LA4: ',
c    .           GLPM,GLAA,V,LA5,LA4,SBMA,LA5-LA4
      GHPM = 2*GHAA + V*(LA5 - LA4)*CBMA
c      print*,'HH+H-',ghpm,xhpm
      GLZZ = 2*GLZZ
      GHZZ = 2*GHZZ
      GLLL = 6*GLLL
      GHHH = 6*GHHH
      GLHH = 2*GLHH
      GHLL = 2*GHLL
      GLAA = 2*GLAA
      GHAA = 2*GHAA
      XNORM = AMZ**2/V
      GLLL = GLLL/XNORM
      GHLL = GHLL/XNORM
      GLHH = GLHH/XNORM
      GHHH = GHHH/XNORM
      GHAA = GHAA/XNORM
      GLAA = GLAA/XNORM
      GLPM = GLPM/XNORM

c--hMSSM?
      IF(IMODEL.EQ.10)THEN
        GHHH = 3*C2A*CBPA            + 3*deps/amz**2*sa**3/sb
        GLLL = 3*C2A*SBPA            + 3*deps/amz**2*ca**3/sb
        GHLL = 2*S2A*SBPA - C2A*CBPA + 3*deps/amz**2*sa*ca**2/sb
        GLHH =-2*S2A*CBPA - C2A*SBPA + 3*deps/amz**2*sa**2*ca/sb
        GHAA =-C2B*CBPA              + deps/amz**2*sa*cb**2/sb
        GLAA = C2B*SBPA              + deps/amz**2*ca*cb**2/sb
        GLPM = 2*AMW**2/AMZ**2*SBMA + C2B*SBPA
        GHPM = 2*AMW**2/AMZ**2*CBMA - C2B*CBPA
      ENDIF


c      print*,'lll',glll
c      print*,'lhh',glhh
c      print*,'hll',ghll
c      print*,'hhh',ghhh
c      print*,'laa',glaa
c      print*,'haa',ghaa
c      print*,'lH+H-',glpm
c      print*,'hH+H-',ghpm*v/amz**2
c      print*,'norm = v/amz**2',v/amz**2

c      print*,'glpm',glpm

      GHPM = GHPM/XNORM
      GAT=1.D0/TGBET
      GAB=TGBET
c MMM changed 21/8/13
      if(i2hdm.eq.0) then
c end MMM changed 2178/13
      GLT=CA/SB
      GLB=-SA/CB
      GHT=SA/SB
      GHB=CA/CB
c MMM changed 21/8/13
      elseif(i2hdm.eq.1) then
         if(itype2hdm.eq.1) then
            glt   = ca/sb
            glb   = ca/sb
            gllep = glb
            ght   = sa/sb
            ghb   = sa/sb
            ghlep = ghb
            gat   = 1.D0/tgbet
            gab   =-1.D0/tgbet
            galep = gab
         elseif(itype2hdm.eq.2) then
            glt   = ca/sb
            glb   =-sa/cb
            gllep = glb
            ght   = sa/sb
            ghb   = ca/cb
            ghlep = ghb
            gat   = 1.D0/tgbet
            gab   = tgbet
            galep = gab
         elseif(itype2hdm.eq.3) then
            glt   = ca/sb
            glb   = ca/sb
            gllep =-sa/cb
            ght   = sa/sb
            ghb   = sa/sb
            ghlep = ca/cb
            gat   = 1.D0/tgbet
            gab   =-1.D0/tgbet
            galep = tgbet
         elseif(itype2hdm.eq.4) then
            glt   = ca/sb
            glb   =-sa/cb
            gllep = ca/sb
            ght   = sa/sb
            ghb   = ca/cb
            ghlep = sa/sb
            gat   = 1.D0/tgbet
            gab   = tgbet
            galep =-1.D0/tgbet
         endif
      endif
c end MMM changed 21/8/13
      GZAL=-CBMA
      GZAH=SBMA
      GLVV=SBMA
      GHVV=CBMA
      B=BET
      IF(CA.EQ.0)THEN
       A = PI/2
      ELSE
       A=DATAN(SA/CA)
      ENDIF
      IF(CA.LT.0D0)THEN
       IF(SA.LT.0D0)THEN
        A = A-PI
       ELSE
        A = A+PI
       ENDIF
      ENDIF
C ===============================================================
C ========== POLE MASSES 
      if(islhai.eq.0) then
      IF(IMODEL.EQ.1)THEN
      IF(IPOLE.EQ.1) THEN 
       MT=RUNM_HDEC(AMT,6)
       MB=RUNM_HDEC(AMT,5)
       SW2=1.D0-AMW**2/AMZ**2
C===== STOP MASSES
       MSTL2=AMSQ**2+(0.5D0-2.D0/3.D0*SW2)*AMZ**2*DCOS(2.D0*B)
       MSTR2=AMUR**2+2.D0/3.D0*SW2*AMZ**2*DCOS(2.D0*B)
       MLRT=AU-AMU/TGBET
       DELT=(MSTL2-MSTR2)**2+4*MT**2*MLRT**2
       MST12=MT**2+0.5D0*(MSTL2+MSTR2-DSQRT(DELT))
       MST22=MT**2+0.5D0*(MSTL2+MSTR2+DSQRT(DELT))
        IF(MST12.LT.0.D0)GOTO 111
       MST(1)=DSQRT(MST12)
       MST(2)=DSQRT(MST22)
       IF(MSTL2.EQ.MSTR2) THEN
        THET = PI/4
       ELSE
        THET=0.5D0*DATAN(2.D0*MT*MLRT / (MSTL2-MSTR2) )
        IF(MSTL2.GT.MSTR2) THET = THET + PI/2
       ENDIF
       CST= DCOS(THET)
       SST= DSIN(THET)
C===== SBOTTOM MASSES
       MSBL2=AMSQ**2+(-0.5D0+1.D0/3.D0*SW2)*AMZ**2*DCOS(2.D0*B)
       MSBR2=AMDR**2-1.D0/3.D0*SW2*AMZ**2*DCOS(2.D0*B)
       MLRB=AD-AMU*TGBET
       DELB=(MSBL2-MSBR2)**2+4*MB**2*MLRB**2
       MSB12=MB**2+0.5D0*(MSBL2+MSBR2-DSQRT(DELB))
       MSB22=MB**2+0.5D0*(MSBL2+MSBR2+DSQRT(DELB))
        IF(MSB12.LT.0.D0)GOTO 111
       MSB(1)=DSQRT(MSB12)
       MSB(2)=DSQRT(MSB22)
       IF(MSBL2.EQ.MSBR2) THEN
        THEB = PI/4
       ELSE
        THEB=0.5D0*DATAN(2.D0*MB*MLRB / (MSBL2-MSBR2) )
        IF(MSBL2.GT.MSBR2) THEB = THEB + PI/2
       ENDIF
       CSB= DCOS(THEB)
       SSB= DSIN(THEB)
C===== LIGHT HIGGS COUPLINGS 
       GLTT(1,1)=-SBPA*(0.5D0*CST**2-2.D0/3.D0*SW2*DCOS(2*THET) )
     .     +MT**2/AMZ**2*GLT + MT*SST*CST/AMZ**2*(AU*GLT+AMU*GHT)
       GLTT(2,2)=-SBPA*(0.5D0*SST**2+2.D0/3.D0*SW2*DCOS(2*THET) )
     .     +MT**2/AMZ**2*GLT - MT*SST*CST/AMZ**2*(AU*GLT+AMU*GHT)
       GLTT(1,2)=-2*SBPA*SST*CST*(2.D0/3.D0*SW2-0.25D0)
     .     + MT*DCOS(2*THET)/2.D0/AMZ**2*(AU*GLT+AMU*GHT)
       GLTT(2,1)=-2*SBPA*SST*CST*(2.D0/3.D0*SW2-0.25D0)
     .     + MT*DCOS(2*THET)/2.D0/AMZ**2*(AU*GLT+AMU*GHT)
       GLBB(1,1)=-SBPA*(-0.5D0*CSB**2+1.D0/3.D0*SW2*DCOS(2*THEB))
     .     +MB**2/AMZ**2*GLB + MB*SSB*CSB/AMZ**2*(AD*GLB-AMU*GHB)
       GLBB(2,2)=-SBPA*(-0.5D0*SSB**2-1.D0/3.D0*SW2*DCOS(2*THEB))
     .     +MB**2/AMZ**2*GLB - MB*SSB*CSB/AMZ**2*(AD*GLB-AMU*GHB)
       GLBB(1,2)=-2*SBPA*SSB*CSB*(-1.D0/3.D0*SW2+0.25D0)
     .    + MB*DCOS(2*THEB)/2.D0/AMZ**2*(AD*GLB-AMU*GHB)
       GLBB(2,1)=-2*SBPA*SSB*CSB*(-1.D0/3.D0*SW2+0.25D0)
     .     + MB*DCOS(2*THEB)/2.D0/AMZ**2*(AD*GLB-AMU*GHB)
C===== HEAVY HIGGS COUPLINGS 
       GHTT(1,1)=CBPA*(0.5D0*CST**2-2.D0/3.D0*SW2*DCOS(2*THET))
     .     +MT**2/AMZ**2*GHT + MT*SST*CST/AMZ**2*(AU*GHT-AMU*GLT)
       GHTT(2,2)=CBPA*(0.5D0*SST**2+2.D0/3.D0*SW2*DCOS(2*THET))
     .     +MT**2/AMZ**2*GHT - MT*SST*CST/AMZ**2*(AU*GHT-AMU*GLT)
       GHTT(1,2)=2*CBPA*SST*CST*(2.D0/3.D0*SW2-0.25D0)
     .     +MT*DCOS(2*THET)/2.D0/AMZ**2*(AU*GHT-AMU*GLT)
       GHTT(2,1)=2*CBPA*SST*CST*(2.D0/3.D0*SW2-0.25D0)
     .     + MT*DCOS(2*THET)/2.D0/AMZ**2*(AU*GHT-AMU*GLT)
       GHBB(1,1)=CBPA*(-0.5D0*CSB**2+1.D0/3.D0*SW2*DCOS(2*THEB))
     .     +MB**2/AMZ**2*GHB + MB*SSB*CSB/AMZ**2*(AD*GHB+AMU*GLB)
       GHBB(2,2)=CBPA*(-0.5D0*SSB**2-1.D0/3.D0*SW2*DCOS(2*THEB))
     .     + MB**2/AMZ**2*GHB - MB*SSB*CSB/AMZ**2*(AD*GHB+AMU*GLB)
       GHBB(1,2)=2*CBPA*SSB*CSB*(-1.D0/3.D0*SW2+0.25D0)
     .     + MB*DCOS(2*THEB)/2.D0/AMZ**2*(AD*GHB+AMU*GLB)
       GHBB(2,1)=2*CBPA*SSB*CSB*(-1.D0/3.D0*SW2+0.25D0)
     .     + MB*DCOS(2*THEB)/2.D0/AMZ**2*(AD*GHB+AMU*GLB)
C===== PSEUDOSCALAR HIGGS COUPLINGS 
       GATT=MT/2.D0/AMZ**2*(AMU+AU*GAT) 
       GABB=MB/2.D0/AMZ**2*(AMU+AD*GAB) 
C======= LOOP CORRECTIONS  
       XDLT=GF/(2.D0*DSQRT(2.D0)*PI**2)*GLT**2*(-2.D0*MT**2+0.5D0*AML2)
     .     *DREAL(F0_HDEC(MT,MT,AML2))
     .     *3*MT**2
       XDLB=GF/(2.D0*DSQRT(2.D0)*PI**2)*GLB**2*(-2.D0*MB**2+0.5D0*AML2)
     .     *DREAL(F0_HDEC(MB,MB,AML2))
     .     *3*MB**2
C--BUG IN CARENA ET AL. FIXED
     .     +GF/(2.D0*DSQRT(2.D0)*PI**2)*GLB**2*(0.5D0*AML2)
     .     *DLOG(MB**2/MT**2)
     .     *3*MB**2
       XDHT=GF/(2.D0*DSQRT(2.D0)*PI**2)*GHT**2*(-2.D0*MT**2+0.5D0*AMH2)
     .     *DREAL(F0_HDEC(MT,MT,AMH2))
     .     *3*MT**2
       XDHB=GF/(2.D0*DSQRT(2.D0)*PI**2)*GHB**2*(-2.D0*MB**2+0.5D0*AMH2)
     .     *DREAL(F0_HDEC(MB,MB,AMH2))
     .     *3*MB**2
C--BUG IN CARENA ET AL. FIXED
     .     +GF/(2.D0*DSQRT(2.D0)*PI**2)*GHB**2*(0.5D0*AMH2)
     .     *DLOG(MB**2/MT**2)
     .     *3*MB**2
       XDAT=GF/(2.D0*DSQRT(2.D0)*PI**2)*GAT**2*(-0.5D0*AMA2)
     .     *DREAL(F0_HDEC(MT,MT,AMA2))
     .     *3*MT**2
       XDAB=GF/(2.D0*DSQRT(2.D0)*PI**2)*GAB**2*(-0.5D0*AMA2)
     .     *DREAL(F0_HDEC(MB,MB,AMA2))
     .     *3*MB**2
C--BUG IN CARENA ET AL. FIXED
     .     +GF/(2.D0*DSQRT(2.D0)*PI**2)*GAB**2*(-0.5D0*AMA2)
     .     *DLOG(MB**2/MT**2)
     .     *3*MB**2
       XDLST=0.D0
       XDLSB=0.D0
       XDHST=0.D0
       XDHSB=0.D0
         DO 311 I=1,2
         DO 311 J=1,2
       XDLST=XDLST+GF/(2.D0*DSQRT(2.D0)*PI**2)*GLTT(I,J)**2*
     .       DREAL(F0_HDEC(MST(I),MST(J),AML2))
     .     *3*AMZ**4
       XDLSB=XDLSB+GF/(2.D0*DSQRT(2.D0)*PI**2)*GLBB(I,J)**2*
     .       DREAL(F0_HDEC(MSB(I),MSB(J),AML2))
     .    *3*AMZ**4
       XDHST=XDHST+GF/(2.D0*DSQRT(2.D0)*PI**2)*GHTT(I,J)**2*
     .       DREAL(F0_HDEC(MST(I),MST(J),AMH2))
     .     *3*AMZ**4
       XDHSB=XDHSB+GF/(2.D0*DSQRT(2.D0)*PI**2)*GHBB(I,J)**2*
     .       DREAL(F0_HDEC(MSB(I),MSB(J),AMH2))
     .     *3*AMZ**4
311    CONTINUE
       XDAST=GF/(1.D0*DSQRT(2.D0)*PI**2)*GATT**2*
     .       DREAL(F0_HDEC(MST(1),MST(2),AMA2))
     .     *3*AMZ**4
       XDASB=GF/(1.D0*DSQRT(2.D0)*PI**2)*GABB**2*
     .       DREAL(F0_HDEC(MSB(1),MSB(2),AMA2))
     .     *3*AMZ**4
      
       AML=DSQRT(AML2+XDLT+XDLB+XDLST+XDLSB)
       AMH=DSQRT(AMH2+XDHT+XDHB+XDHST+XDHSB)  
       AMA=DSQRT(AMA2+XDAT+XDAB+XDAST+XDASB)  
      ELSE
       AML=AMLR
       AMH=AMHR     
       AMA=AMAR     
      ENDIF 
      ELSE
       AML=AMLR
       AMH=AMHR
       AMA=AMAR
      ENDIF
      endif
c MMM changed 21/8/13
      if(I2HDM.eq.1) then
         ama = amar
         amlr= AMHL2HDM
         amhr= AMHH2HDM
         aml = AMHL2HDM
         amh = AMHH2HDM
         amch= AMHC2HDM
      endif
c end MMM changed 21/8/13
      RETURN
111   STOP
      END

C ===================== THE FUNCTION F0 ===============
      COMPLEX*16 FUNCTION F0_HDEC(M1,M2,QSQ)
      IMPLICIT REAL*8 (A-H,M,O-Z)
      COMPLEX*16 CD,CR,CQ2,IEPS,CBET,CXX
      M1SQ = M1*M1
      M2SQ = M2*M2
      AQSQ = DABS(QSQ)
      IEPS = DCMPLX(1.D0,1.D-12)
      CQ2 = QSQ*IEPS
      CD = (M1SQ-M2SQ)/CQ2
      CR = CDSQRT((1+CD)**2 - 4*M1SQ/CQ2)
      IF(QSQ.EQ.0.D0) THEN
       F0_HDEC = 0.D0
      ELSE
       IF(M1.EQ.M2) THEN
        F0_HDEC = -2.D0 + CR*CDLOG(-(1+CR)/(1-CR))
       ELSE
        CBET = CDSQRT(1-4*M1*M2/(CQ2 - (M1-M2)**2))
        CXX = (CBET-1)/(CBET+1)
        F0_HDEC = -1 + ((QSQ+M2SQ-M1SQ)/2/QSQ - M2SQ/(M2SQ-M1SQ))
     .                                           *DLOG(M2SQ/M1SQ)
     .     - (QSQ-(M1-M2)**2)/QSQ*CBET*CDLOG(CXX)
       ENDIF
      ENDIF
      RETURN
      END

C     ************************************************************
C     SUBROUTINE FOR HSM ---> V*V* ---> 4F
C     ************************************************************
      SUBROUTINE HTOVV_HDEC(IV,AMH,AMV,GAMV,HTVV)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      COMMON/VVOFF_HDEC/AMH1,AMV1,GAMV1
      COMMON/VVOFFFLAG_HDEC/IV1
      COMMON/PREC_HDEC/IP
      EXTERNAL FTOVV1_HDEC
c     IP=20
      IP=50
      AMH1=AMH
      AMV1=AMV
      GAMV1=GAMV
      IV1 = IV
      DLT=1D0/IP
      SUM=0D0
      DO 1 I=1,IP
       UU=DLT*I
       DD=UU-DLT
       CALL QGAUS1_HDEC(FTOVV1_HDEC,DD,UU,RES)
       SUM=SUM+RES
1     CONTINUE
      HTVV=SUM
      RETURN
      END

      DOUBLE PRECISION FUNCTION FTOVV1_HDEC(XX)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      COMMON/FIRST_HDEC/X1
      COMMON/PREC_HDEC/IP
      EXTERNAL FTOVV2_HDEC
      X1=XX
      DLT=1D0/IP
      SUM=0D0
      DO 1 I=1,IP
       UU=DLT*I
       DD=UU-DLT
       CALL QGAUS2_HDEC(FTOVV2_HDEC,DD,UU,RES)
       SUM=SUM+RES
1     CONTINUE
      FTOVV1_HDEC=SUM
      RETURN
      END

      DOUBLE PRECISION FUNCTION FTOVV2_HDEC(XX)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION YY(2)
      COMMON/FIRST_HDEC/X1
      YY(1)=X1
      YY(2)=XX
      FTOVV2_HDEC=FTOVV_HDEC(YY)
      RETURN
      END

      DOUBLE PRECISION FUNCTION FTOVV_HDEC(XX)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DOUBLE PRECISION LAMB
      DIMENSION XX(2)
      COMMON/VVOFF_HDEC/AMH,AMW,GAMW
      LAMB(X,Y)=DSQRT((1.D0-X-Y)**2-4.D0*X*Y)
      PI=4D0*DATAN(1D0)
      IF(AMH.LT.2*AMW)THEN
       ICASE = 1
      ELSE
       ICASE = 0
      ENDIF
      IF(ICASE.EQ.0)THEN
       YY = AMH**2
       Y1 = DATAN((YY-AMW**2)/AMW/GAMW)
       Y2 = -DATAN((AMW**2)/AMW/GAMW)
       DJAC = Y1-Y2
       T1 = TAN(Y1*XX(1)+Y2*(1.D0-XX(1)))
       SP = AMW**2 + AMW*GAMW*T1
       YY = (AMH-DSQRT(SP))**2
       Y1 = DATAN((YY-AMW**2)/AMW/GAMW)
       Y2 = -DATAN((AMW**2)/AMW/GAMW)
       DJAC = DJAC*(Y1-Y2)
       T2 = TAN(Y1*XX(2)+Y2*(1.D0-XX(2)))
       SM = AMW**2 + AMW*GAMW*T2
       AM2=AMH**2
       GAM = AM2*LAMB(SP/AM2,SM/AM2)*(1+LAMB(SP/AM2,SM/AM2)**2*AMH**4
     .                               /SP/SM/12)
       PRO1 = SP/AMW**2
       PRO2 = SM/AMW**2
       FTOVV_HDEC = PRO1*PRO2*GAM*DJAC/PI**2
     .            * RADVV_HDEC(SP,SM)
      ELSE
       SP = AMH**2*XX(1)
       SM = (AMH-DSQRT(SP))**2*XX(2)
       DJAC = AMH**2*(AMH-DSQRT(SP))**2/PI**2
       AM2=AMH**2
       GAM = AM2*LAMB(SP/AM2,SM/AM2)*(1+LAMB(SP/AM2,SM/AM2)**2*AMH**4
     .                               /SP/SM/12)
       PRO1 = SP*GAMW/AMW/((SP-AMW**2)**2+AMW**2*GAMW**2)
       PRO2 = SM*GAMW/AMW/((SM-AMW**2)**2+AMW**2*GAMW**2)
       FTOVV_HDEC = PRO1*PRO2*GAM*DJAC
     .            * RADVV_HDEC(SP,SM)
      ENDIF
      RETURN
      END

      DOUBLE PRECISION FUNCTION RADVV_HDEC(Q12,Q22)
      IMPLICIT DOUBLE PRECISION (A-B,D-H,O-Z), COMPLEX*16 (C)
      COMMON/VVOFFFLAG_HDEC/IV
      q1 = dsqrt(q12)
      q2 = dsqrt(q22)
      if(iv.eq.0)then
       radvv_hdec=1
      elseif(iv.eq.1)then
       radvv_hdec=radww_hdec(q1,q2)
      else
       radvv_hdec=radzz_hdec(q1,q2)
      endif
      RETURN
      END

      DOUBLE PRECISION FUNCTION RADWW_HDEC(Q1W,Q2W)
C     ************************************************************
C     ELECTROWEAK CORRECTIONS TO HSM ---> W*W* ---> 4F (APPROX.)
C     (A. BREDENSTEIN ET AL., PHYS. REV. D74 (2006) 013004
C                             [ARXIV:HEP-PH/0604011])
C     ************************************************************
      IMPLICIT DOUBLE PRECISION (A-B,D-H,O-Z), COMPLEX*16 (C)
      DOUBLE PRECISION LAMB
      COMMON/VVOFF_HDEC/AMH,AMW,GAMW
      COMMON/PARAM_HDEC/GF,ALPH,AMTAU,AMMUON,AMZ,AMW0
      COMMON/MASSES_HDEC/AMS0,AMC0,AMB0,AMT
      COMMON/WZWDTH_HDEC/GAMC0,GAMT,GAMT1,GAMW0,GAMZ
      LAMB(X,Y)=DSQRT((1.D0-X-Y)**2-4.D0*X*Y)
      PI=4D0*DATAN(1D0)
      cmut2 = dcmplx(amt**2,-amt*gamt)
      cmuw2 = dcmplx(amw**2,-amw*gamw)
      cmuz2 = dcmplx(amz**2,-amz*gamz*3)
      cbett = cdsqrt(1-4*cmut2/amh**2)
      cbetw = cdsqrt(1-4*cmuw2/amh**2)
      cbetz = cdsqrt(1-4*cmuz2/amh**2)
      cxt = (cbett-1)/(cbett+1)
      cthww = 8+12*cbett**2+3*cbett*(3*cbett**2-1)*cdlog(cxt)
     .      + 3/2.d0*(1-cbett**2)**2*cdlog(cxt)**2
      cbetwb = cdsqrt(dcmplx(amh**4+q1w**4+q2w**4-2*amh**2*q1w**2
     .             -2*amh**2*q2w**2-2*q1w**2*q2w**2))/amh**2
      cgw = (1-cbetwb**2)**2
      dmw = dabs(q1w**2-q2w**2)/amh**2
      cdcoul = alph/cbetwb*dimag(cdlog((cbetw-cbetwb+dmw)
     .                                /(cbetw+cbetwb+dmw)))
c     cgz = 1
      cgz = 0.7d0
      cdcoulz = alph/2/cbetz*dimag(cdlog((cbetz-cbetwb)
     .                                  /(cbetz+cbetwb)))
c     bias = 0.04d0*(1+(100-amh)/500)
      bias = 0.05d0*(1+(100-amh)/500)
      radww_hdec=dreal(1.d0
     .           +gf*cmut2/8.d0/pi**2/dsqrt(2.d0)*(-5+cthww)
     .           +gf*amh**2/16.d0/pi**2/dsqrt(2.d0)*2.800952d0
     .           +(gf*amh**2/16.d0/pi**2/dsqrt(2.d0))**2*62.0308d0
     .           +cgw*cdcoul + cgz*cdcoulz + bias)
      RETURN
      END

      DOUBLE PRECISION FUNCTION RADZZ_HDEC(Q1Z,Q2Z)
C     ************************************************************
C     ELECTROWEAK CORRECTIONS TO HSM ---> Z*Z* ---> 4F (APPROX.)
C     (A. BREDENSTEIN ET AL., PHYS. REV. D74 (2006) 013004
C                             [ARXIV:HEP-PH/0604011])
C     ************************************************************
      IMPLICIT DOUBLE PRECISION (A-B,D-H,O-Z), COMPLEX*16 (C)
      DOUBLE PRECISION LAMB
      COMMON/VVOFF_HDEC/AMH,AMZ,GAMZ
      COMMON/PARAM_HDEC/GF,ALPH,AMTAU,AMMUON,AMZ0,AMW
      COMMON/MASSES_HDEC/AMS0,AMC0,AMB0,AMT
      COMMON/WZWDTH_HDEC/GAMC0,GAMT,GAMT1,GAMW,GAMZ0
      LAMB(X,Y)=DSQRT((1.D0-X-Y)**2-4.D0*X*Y)
      PI=4D0*DATAN(1D0)
      cmut2 = dcmplx(amt**2,-amt*gamt)
      cmuw2 = dcmplx(amw**2,-amw*gamw)
      cmuz2 = dcmplx(amz**2,-amz*gamz*2)
      cbett = cdsqrt(1-4*cmut2/amh**2)
      cbetw = cdsqrt(1-4*cmuw2/amh**2)
      cbetz = cdsqrt(1-4*cmuz2/amh**2)
      cxt = (cbett-1)/(cbett+1)
      cthzz = 20+6*cbett**2+3*cbett*(cbett**2+1)*cdlog(cxt)
     .      + 3*(1-cbett**2)*cdlog(cxt)**2
      cbetzb = cdsqrt(dcmplx(amh**4+q1z**4+q2z**4-2*amh**2*q1z**2
     .             -2*amh**2*q2z**2-2*q1z**2*q2z**2))/amh**2
c     cgw =-0.15d0 * (1-cbetw**2)**2
      cgw =-0.10d0 * (1-cbetw**2)**2
      cdcoulw = alph/2/cbetw*dimag(cdlog((cbetw-cbetzb)
     .                                  /(cbetw+cbetzb)))
      cgz = 1
      cdcoulz = alph/2/cbetz*dimag(cdlog((cbetz-cbetzb)
     .                                  /(cbetz+cbetzb)))
c     bias = 0.02d0*(1+(100-amh)/350)
      bias = 0.02d0*(1+(100-amh)/300)
      radzz_hdec=dreal(1.d0
     .           +gf*cmut2/8.d0/pi**2/dsqrt(2.d0)*(1+cthzz)
     .           +gf*amh**2/16.d0/pi**2/dsqrt(2.d0)*2.800952d0
     .           +(gf*amh**2/16.d0/pi**2/dsqrt(2.d0))**2*62.0308d0
     .           +cgw*cdcoulw + cgz*cdcoulz + bias)
      RETURN
      END

C     ************************************************************
C     SUBROUTINE FOR HSM ---> TT* ---> TBW
C     ************************************************************
      SUBROUTINE HTOTTS_HDEC(AMH,AMT,AMB,AMW,HTTS)
      IMPLICIT REAL*8(A-Z)
      INTEGER IP,K
      COMMON/PREC1_HDEC/IP
      EXTERNAL FUNSTT1_HDEC
      COMMON/IKSY0_HDEC/X1,X2,M1,M2,M3,ECM,S
      COMMON/TOP0_HDEC/AMH0,AMT0,AMB0,AMW0
      AMH0=AMH
      AMT0=AMT
      AMB0=AMB
      AMW0=AMW
      IP=5
      M1=AMB
      M2=AMT
      M3=AMW
C     FIRST INTEGRATE OVER X2, i.e. (1+3) SYSTEM
C        CHECK WHETHER ENOUGH PHASE SPACE
      MASTOT=M1+M2+M3
      IF(MASTOT.GE.AMH) GOTO 12
      ECM=AMH
      S=ECM**2
      U1=(ECM-M2)**2
      D1=(M1+M3)**2
      U=(S-D1+M2**2)/s
      D=(S-U1+M2**2)/s
      DEL=(U-D)/IP
      U=D+DEL
      XSEC=0.D0
      DO K=1,IP
      CALL QGAUS1_HDEC(FUNSTT1_HDEC,D,U,SS)
      D=U
      U=D+DEL
      XSEC=XSEC+SS
      ENDDO
      HTTS=XSEC
12    CONTINUE
      RETURN
      END

      DOUBLE PRECISION FUNCTION FUNSTT1_HDEC(XL)
      IMPLICIT REAL*8(A-Z)
      INTEGER IP,I
      COMMON/IKSY0_HDEC/X1,X2,M1,M2,M3,ECM,S
      COMMON/PREC1_HDEC/IP
      EXTERNAL FUNSTT2_HDEC
      X2=XL
      S13=S-S*X2+M2**2
      TEM=2.D0*DSQRT(S13)
      E2S=(S-S13-M2**2)/TEM
      E3S=(S13+M3**2-M1**2)/TEM
C     SECOND INTEGRAL OVER X1, i.e. (2+3) SYSTEM
      U1=(E2S+E3S)**2-(DSQRT(E2S**2-M2**2)-DSQRT(E3S**2-M3**2))**2
      D1=(E2S+E3S)**2-(DSQRT(E2S**2-M2**2)+DSQRT(E3S**2-M3**2))**2
      U=(S-D1+M1**2)/s
      D=(S-U1+M1**2)/s
      DEL=(U-D)/IP
      FUNSTT1_HDEC=0.d0
      U=D+DEL
      DO I=1,IP
      CALL QGAUS2_HDEC(FUNSTT2_HDEC,D,U,SS)
      FUNSTT1_HDEC=FUNSTT1_HDEC+SS
      D=U
      U=D+DEL
      ENDDO
      RETURN
      END

c --- this is for a check of the off-shell decays h/H -> H+ W-
      double precision function hvhinteg(dum)

      implicit double precision (a-h,o-z)
      double precision dum(2),lamb,kj,kk

      common/DECPARAMETERS/amhi,amhj,amhk,gamtotj,gamtotk

c      lamb(x,y)=dsqrt((1.d0-x-y)**2-4.d0*x*y)
c      pi=4d0*datan(1d0)

c amhi = h/H, amhj = amch, amhk = amw     
      kj = amhj**2/amhi**2
      kk = amhk**2/amhi**2
      gamk = gamtotk**2/amhi**2

c ---- the integration limits ----
c --- the x1 integration ---------
      x1m = 0.D0
      x1p = 1.D0-kj
      x1 = (x1p-x1m)*dum(1)+x1m 
c --- the x2 integration ---------
      x2m = 1.D0-x1-kj
      x2p = 1.D0-kj/(1.D0-x1)
      x2 = (x2p-x2m)*dum(2)+x2m

      djac = (x1p-x1m)*(x2p-x2m)

      hvhinteg = ((1.D0-x1)*(1.D0-x2)-kj)/
     .     ((1.D0-x1-x2-kj+kk)**2+kk*gamk)*djac
      
      return
      end
c ---- end check off-shell decays h/H -> H+ W-

      DOUBLE PRECISION FUNCTION FUNSTT2_HDEC(XK)
      IMPLICIT REAL*8(A-Z)
      COMMON/IKSY0_HDEC/X1,X2,M1,M2,M3,ECM,S
      X1=XK
      CALL ELEMSTT_HDEC(SS)
      FUNSTT2_HDEC=SS
      RETURN
      END

      SUBROUTINE ELEMSTT_HDEC(RES)
      IMPLICIT REAL*8(A-Z)
      COMMON/IKSY0_HDEC/X1,X2,M1,M2,M3,ECM,S
      COMMON/TOP0_HDEC/AMH,AMT,AMB,AMW
      COMMON/WZWDTH_HDEC/GAMC0,GAMT0,GAMT1,GAMW0,GAMZ0
      GAMT=GAMT0**2*AMT**2/AMH**4
      GAMW=GAMW0**2*AMW**2/AMH**4
      W=AMW**2/AMH**2
      T=AMT**2/AMH**2
      Y1=1-X2
      Y2=1-X1
      X0=2.D0-X1-X2
      W1=(1.D0-X2)
      W3=(1.D0-X1-X2)
      W11=1.D0/((1.D0-X2)**2+GAMT)
      W33=1.D0/(W3**2+GAMW)
      W13=W1*W3*W11*W33

      R11=4*T*W-16.*T*W*Y1-4.*T*Y2*Y1+8.*T*Y1+32.*T*W**2-20
     . .*T*Y1**2+8.*W*Y2*Y1+4.*W*Y1**2-4.*Y2*Y1**2-16.*T**2*W-
     .  32.*T**2*Y1+4.*T**2-16.*T**3-8.*W**2+4.*Y1**2-4.*Y1**3
      R33=-4.*T*W+4.*T*W*Y2-2.*T*W*Y2*Y1+4.*T*W*Y1+T*W*Y2**2-
     .  3.*T*W*Y1**2+2.*T*Y2*Y1-3.*T*Y2*Y1**2+4.*T*W**2-4.*T*W**3
     .  +T*Y2**2-3.*T*Y2**2*Y1-T*Y2**3+T*Y1**2-T*Y1**3+4.*T**2
     .  *W-4.*T**2*W*Y2-4.*T**2*W*Y1-2.*T**2*Y2*Y1-4.*T**2*W**2-
     .  T**2*Y2**2-T**2*Y1**2+4.*W**2*Y2*Y1-8.*W**3*Y2-8.*W**3*Y1
     .  +4.*W**3+8.*W**4
      R13=8.*W-24.*T*W+16.*T*W*Y1 -4.*T*Y2+16.*T*Y2*Y1-4.*T*
     .  Y1+16.*T*W**2+4.*T*Y2**2+12.*T*Y1**2-8.*W*Y2-12.*W*Y2*Y1
     .  -8.*W*Y1+4.*W*Y1**2-4.*Y2*Y1+8.*Y2*Y1**2+16.*T**2*W+8.
     .  *T**2*Y2+8.*T**2*Y1+16.*W**2*Y2+24.*W**2*Y1+4.*Y2**2*Y1-
     .  32.*W**3-4.*Y1**2+4.*Y1**3
      RES=R11*W11+4.D0*R33*W33/T-2.D0*R13*W13
      RETURN
      END

C     **************************************************
C     SUBROUTINE FOR A -> TT* -> TBW
C     **************************************************

      subroutine ATOTT_HDEC(ama,amt,amb,amw,amch,att0,gab,gat)

      implicit real*8(a-z)
      integer ip,k
      common/prec1_hdec/ip
      external funatt1_hdec
      common/iksy1_hdec/x1,x2,m1,m2,m3,ecm,s
      common/top1_hdec/ama1,amt1,amb1,amw1,amch1
      common/top6_hdec/gab1,gat1

      ama1=ama
      amt1=amt
      amb1=amb
      amw1=amw
      amch1=amch
      gab1 = gab
      gat1 = gat

      ip=5
      m1=amb
      m2=amt
      m3=amw
c        first integrate over x2, i.e. (1+3) system
c        check whether enough phase space
      mastot=m1+m2+m3
      if(mastot.ge.ama) goto 12
      ecm=ama
      s=ecm**2
      u1=(ecm-m2)**2
      d1=(m1+m3)**2
      u=(s-d1+m2**2)/s
      d=(s-u1+m2**2)/s
      del=(u-d)/ip
      u=d+del
      xsec=0.d0
      do k=1,ip
      call qgaus1_hdec(funatt1_hdec,d,u,ss)
      d=u
      u=d+del
      xsec=xsec+ss
      enddo
      att0=xsec
 12   continue
      return
      end

      DOUBLE PRECISION FUNCTION FUNATT1_HDEC(XL)
      IMPLICIT REAL*8(A-Z)
      INTEGER IP,I
      COMMON/IKSY1_HDEC/X1,X2,M1,M2,M3,ECM,S
      COMMON/PREC1_HDEC/IP
      EXTERNAL FUNATT2_HDEC
      X2=XL
      S13=S-S*X2+M2**2
      TEM=2.D0*DSQRT(S13)
      E2S=(S-S13-M2**2)/TEM
      E3S=(S13+M3**2-M1**2)/TEM
C     SECOND INTEGRAL OVER X1, i.e. (2+3) SYSTEM
      U1=(E2S+E3S)**2-(DSQRT(E2S**2-M2**2)-DSQRT(E3S**2-M3**2))**2
      D1=(E2S+E3S)**2-(DSQRT(E2S**2-M2**2)+DSQRT(E3S**2-M3**2))**2
      U=(S-D1+M1**2)/s
      D=(S-U1+M1**2)/s
      DEL=(U-D)/IP
      FUNATT1_HDEC=0.d0
      U=D+DEL
      DO I=1,IP
      CALL QGAUS2_HDEC(FUNATT2_HDEC,D,U,SS)
      FUNATT1_HDEC=FUNATT1_HDEC+SS
      D=U
      U=D+DEL
      ENDDO
      RETURN
      END

      DOUBLE PRECISION FUNCTION FUNATT2_HDEC(XK)
      IMPLICIT REAL*8(A-Z)
      COMMON/IKSY1_HDEC/X1,X2,M1,M2,M3,ECM,S
      X1=XK
      CALL ELEMATT_HDEC(SS)
      FUNATT2_HDEC=SS
      RETURN
      END

      subroutine ELEMATT_HDEC(res)
      implicit real*8(a-z)
      common/iksy1_hdec/x1,x2,m1,m2,m3,ecm,s
      common/top1_hdec/ama,amt,amb,amw,amch
      common/top6_hdec/gab,gat
      common/wzwdth_hdec/gamc0,gamt0,gamt1,gamw,gamz

      gamt=gamt1**2*amt**2/ama**4
      gamc=gamc0**2*amch**2/ama**4
      ch=amch**2/ama**2
      w=amw**2/ama**2
      t=amt**2/ama**2
      b=amb**2/ama**2
      y1=1-x1
      y2=1-x2
      x0=2.d0-x1-x2
      w1=(1.d0-x2)
      w2=(1.d0-x0+w-ch)
      w22=1.d0/ ((1.d0-x0+w-ch)**2+gamc)
      w11=1.d0/((1.d0-x2)**2+gamt)
      w12=w1*w2*w11*w22
      w3=(1.d0-x1)
      w33=1.d0/w3**2
      w13=w1*w11/w3
      w23=w2*w22/w3

      r11= 4.d0*b**2 - 8.d0*b*t + 4.d0*t**2 + 4.d0*b*w + 4.d0*t*w - 
     .     8.d0*w**2 - 8.d0*b*y2 + 8.d0*t*y2 + 4.d0*b*y1*y2 - 
     .     4.d0*t*y1*y2 + 8.d0*w*y1*y2 + 4.d0*y2**2 + 4.d0*b*y2**2 -
     .     4.d0*t*y2**2 + 4.d0*w*y2**2 - 4.d0*y1*y2**2 - 4.d0*y2**3
      r22= 4.d0/t*(4.d0*w - (y1+y2)**2)*(b**2*gab**2+gat**2*t*
     .     (-1.d0 + t - w + y1 + y2) + b*(4.d0*gab*gat*t + gat**2*t +
     .     gab**2*(-1 + t - w + y1 + y2)))
      r12= 2.d0*(-4.d0*(gat*(-2*w**2 -(t + y2)*(-1 + y1 + y2)*(y1 + y2)
     .     +w*(2*(-1 + t + y1) + (2 + y1)*y2 + y2**2)) + 
     .     b*(gat*(2*w - y1 - y2) + gab*(4*w - (y1 + y2)**2))))
      r33= 4.d0*b**2 - 8.d0*b*t + 4.d0*t**2 + 4.d0*b*w + 4.d0*t*w -
     .     8.d0*w**2 + 8.d0*b*y1 - 8.d0*t*y1 + 4.d0*y1**2 - 
     .     4.d0*b*y1**2 + 4.d0*t*y1**2 + 4.d0*w*y1**2 - 4.d0*y1**3 - 
     .     4.d0*b*y1*y2 + 4.d0*t*y1*y2 + 8.d0*w*y1*y2 - 4.d0*y1**2*y2
      r13= -8.d0*b*(y1**2+2.d0*y1*y2+y2**2+2.d0*w)
      r23= -8.d0*(gat*t*(4.d0*w - (y1 + y2)**2) + gab*(-2.d0*w**2 - 
     .     (y1 + y2)*(t + b*(-1.d0 + y1 + y2) + y1*(-1.d0 + y1 + y2)) +
     .     w*(2.d0*b + 2.d0*t + 
     .     2.d0*(-1.d0 + y2) + y1*(2.d0 + y1 + y2))))

c --- take off charged Higgs exchange ---
      w22 = 0.D0
      w12 = 0.D0
      w23 = 0.D0
c ----------------------------------------

      res= (r11*w11*gat**2+r22*w22+r12*w12*gat+
     .     r33*w33*b/t*gab**2+r23*w23*gab*b/t+
     .     r13*w13*gab*gat)/gat**2

      return

      end

C     ************************************************************
C     SUBROUTINE FOR H ---> TT* ---> TBW
C     ************************************************************
      SUBROUTINE HTOTT_HDEC(AMH,AMT,AMB,AMW,AMCH,GHT,GHB,GAT,GAB,GHVV,
     .     GHIHPWM,HTT0)
      IMPLICIT REAL*8(A-Z)
      INTEGER IP,K
      COMMON/PREC1_HDEC/IP
      EXTERNAL FUNHTT1_HDEC
      COMMON/IKSY2_HDEC/X1,X2,M1,M2,M3,ECM,S
      COMMON/TOP2_HDEC/AMH2,AMT2,AMB2,AMW2,AMCH2,GHT2,GHB2,
     .     GAT2,GAB2,GHVV2
c maggie changed 20/8/2013
      COMMON/TOP4_HDEC/GLVV2
c end maggie changed 20/8/2013
      AMH2=AMH
      AMT2=AMT
      AMB2=AMB
      AMW2=AMW
      AMCH2=AMCH
      GHT2=GHT
      GHB2=GHB
      GAT2=GAT
      GAB2=GAB
      GHVV2=GHVV
c MMM changed 20/8/2013
      GLVV2=GHIHPWM
c end MMM changed 20/8/2013
      IP=5
      M1=AMB
      M2=AMT
      M3=AMW
C     FIRST INTEGRATE OVER X2, i.e. (1+3) SYSTEM
C        CHECK WHETHER ENOUGH PHASE SPACE
      MASTOT=M1+M2+M3
      IF(MASTOT.GE.AMH) GOTO 12
      ECM=AMH
      S=ECM**2
      U1=(ECM-M2)**2
      D1=(M1+M3)**2
      U=(S-D1+M2**2)/s
      D=(S-U1+M2**2)/s
      DEL=(U-D)/IP
      U=D+DEL
      XSEC=0.D0
      DO K=1,IP
      CALL QGAUS1_HDEC(FUNHTT1_HDEC,D,U,SS)
      D=U
      U=D+DEL
      XSEC=XSEC+SS
      ENDDO
      HTT0=XSEC
 12   CONTINUE
      RETURN
      END
 
      DOUBLE PRECISION FUNCTION FUNHTT1_HDEC(XL)
      IMPLICIT REAL*8(A-Z)
      INTEGER IP,I
      COMMON/IKSY2_HDEC/X1,X2,M1,M2,M3,ECM,S
      COMMON/PREC1_HDEC/IP
      EXTERNAL FUNHTT2_HDEC
      X2=XL
      S13=S-S*X2+M2**2
      TEM=2.D0*DSQRT(S13)
      E2S=(S-S13-M2**2)/TEM
      E3S=(S13+M3**2-M1**2)/TEM
C     SECOND INTEGRAL OVER X1, i.e. (2+3) SYSTEM
      U1=(E2S+E3S)**2-(DSQRT(E2S**2-M2**2)-DSQRT(E3S**2-M3**2))**2
      D1=(E2S+E3S)**2-(DSQRT(E2S**2-M2**2)+DSQRT(E3S**2-M3**2))**2
      U=(S-D1+M1**2)/s
      D=(S-U1+M1**2)/s
      DEL=(U-D)/IP
      FUNHTT1_HDEC=0.d0
      U=D+DEL
      DO I=1,IP
      CALL QGAUS2_HDEC(FUNHTT2_HDEC,D,U,SS)
      FUNHTT1_HDEC=FUNHTT1_HDEC+SS
      D=U
      U=D+DEL
      ENDDO
      RETURN
      END

      DOUBLE PRECISION FUNCTION FUNHTT2_HDEC(XK)
      IMPLICIT REAL*8(A-Z)
      COMMON/IKSY2_HDEC/X1,X2,M1,M2,M3,ECM,S
      X1=XK
      CALL ELEMHTT_HDEC(SS)
      FUNHTT2_HDEC=SS
      RETURN
      END

      SUBROUTINE ELEMHTT_HDEC(RES)
      IMPLICIT REAL*8(A-Z)
      COMMON/IKSY2_HDEC/X1,X2,M1,M2,M3,ECM,S
      COMMON/TOP2_HDEC/AMH,AMT,AMB,AMW,AMCH,GHT,GHB,GAT,GAB,GHVV
      COMMON/WZWDTH_HDEC/GAMC0,GAMT0,GAMT1,GAMW0,GAMZ0
      COMMON/TOP4_HDEC/GLVV
      GAMT=GAMT1**2*AMT**2/AMH**4
      GAMC=GAMC0**2*AMCH**2/AMH**4
      GAMW=GAMW0**2*AMW**2/AMH**4
      CH=AMCH**2/AMH**2
      W=AMW**2/AMH**2
      T=AMT**2/AMH**2
      B=AMB**2/AMH**2
      Y1=1-X2
      Y2=1-X1
      X0=2.D0-X1-X2

      W1=(1.D0-X2)
      W2=(1.D0-X0+W-CH)
      W3 = (1.D0-X0)
      W22=1.D0/ (W2**2+GAMC)
      W11=1.D0/(W1**2+GAMT)
      W33=1.D0/(W3**2+GAMW)
      W12=W1*W2*W11*W22
      W13=W1*W3*W11*W33
      W23=W2*W3*W22*W33
      W4=(1.d0-X1)
      W44=1.d0/W4**2
      W14=W1*W11/W4
      W24=W2*W22/W4
      W34=W3*W33/W4

      R11=4.d0*B**2-8.d0*B*T-16.d0*B**2*T+4.d0*T**2+32.d0*B*T**2-
     .     16.d0*T**3+4.d0*B*W+4.d0*T*W-16.d0*B*T*W-16.d0*T**2*W-
     .     8.d0*W**2+32.d0*T*W**2-8.d0*B*Y1+8.d0*T*Y1+32.d0*B*T*Y1-
     .     32.d0*T**2*Y1-16.d0*T*W*Y1+4.d0*Y1**2+4.d0*B*Y1**2-
     .     20.d0*T*Y1**2+4.d0*W*Y1**2-4.d0*Y1**3+4.d0*B*Y1*Y2 - 
     .     4.d0*T*Y1*Y2+8.d0*W*Y1*Y2-4.d0*Y1**2*Y2
      R22= 4.d0*(4.d0*W - (Y1+Y2)**2)*(b**2*GAB**2+GAT**2*t*
     .     (-1.d0 + T - W + Y1 + Y2) + B*(4.d0*GAB*GAT*T + GAT**2*T +
     .     GAB**2*(-1 + T - W + Y1 + Y2)))/T
      R33=-4.d0*B*W+4.d0*B**2*W-4.d0*T*W-8.d0*B*T*W+4.d0*T**2*W+
     .     4.d0*B*W**2-4.d0*B**2*W**2+4.d0*T*W**2+8.d0*B*T*W**2-
     .     4.d0*T**2*W**2+4.d0*W**3-4.d0*B*W**3-4.d0*T*W**3+8.d0*W**4+
     .     4.d0*B*W*Y1-4.d0*B**2*W*Y1+4.d0*T*W*Y1+8.d0*B*T*W*Y1-
     .     4.d0*T**2*W*Y1-8.d0*W**3*Y1+B*Y1**2-B**2*Y1**2+T*Y1**2+
     .     2.d0*B*T*Y1**2-T**2*Y1**2+B*W*Y1**2-3.d0*T*W*Y1**2-B*Y1**3-
     .     T*Y1**3+4.d0*B*W*Y2-4.d0*B**2*W*Y2+4.d0*T*W*Y2+
     .     8.d0*B*T*W*Y2-4.d0*T**2*W*Y2-8.d0*W**3*Y2+2.d0*B*Y1*Y2-
     .     2.d0*B**2*Y1*Y2+2.d0*T*Y1*Y2+4.d0*B*T*Y1*Y2-2.d0*T**2*Y1*Y2-
     .     2.d0*B*W*Y1*Y2-2.d0*T*W*Y1*Y2+4.d0*W**2*Y1*Y2-
     .     3.d0*B*Y1**2*Y2-3.d0*T*Y1**2*Y2+B*Y2**2-B**2*Y2**2+T*Y2**2+
     .     2.d0*B*T*Y2**2-T**2*Y2**2-3.d0*B*W*Y2**2+T*W*Y2**2-
     .     3.d0*B*Y1*Y2**2-3.d0*T*Y1*Y2**2-B*Y2**3-T*Y2**3
      R12=4.d0*(B*GAB*((Y1*(2.d0*T - 2*W + Y1)) + 
     .     2.d0*(T + W)*Y2 - Y2**2 - 2.d0*B*(Y1 + Y2)) +
     .     GAT*(2.d0*W**2 - W*(2.d0*B + 2.d0*T*(3 + Y1 - Y2) +
     .     2.d0*(-1.d0 + Y1 + Y2) + Y1*(Y1 + Y2)) + 
     .     (Y1 + Y2)*(B - 2.d0*B*T +
     .     (T + Y1)*(-1.d0 + 2.d0*T + Y1 + Y2))))
c original HDECAY sign in R13 not correct
      R13=(8.d0*W-8*B*W+16.d0*B**2*W-24.d0*T*W-32.d0*B*T*W+16.d0*T**2*W+
     .     16.d0*B*W**2+16.d0*T*W**2-32.d0*W**3+4.d0*B*Y1+8.d0*B**2*Y1-
     .     4.d0*T*Y1-16.d0*B*T*Y1+8.d0*T**2*Y1-8.d0*W*Y1-16.d0*B*W*Y1+
     .     16.d0*T*W*Y1+24.d0*W**2*Y1-4.d0*Y1**2-4.d0*B*Y1**2+
     .     12.d0*T*Y1**2+4.d0*W*Y1**2+4.d0*Y1**3+4.d0*B*Y2+8.d0*B**2*Y2-
     .     4.d0*T*Y2-16.d0*B*T*Y2+8.d0*T**2*Y2-8.d0*W*Y2+16.d0*W**2*Y2-
     .     4.d0*Y1*Y2+16.d0*T*Y1*Y2-12.d0*W*Y1*Y2+8.d0*Y1**2*Y2+
     .     4.d0*B*Y2**2+4.d0*T*Y2**2+4.d0*Y1*Y2**2)*(-1.d0)
      R23=-4.d0*(B**2*GAB*(2.d0*W*(-2.d0 + Y1 + Y2) + (Y1 + Y2)**2) -
     .     GAT*T*(2*W**2*(-Y1 + Y2) + (Y1 + Y2)**2*(-1 + T + Y1 + Y2) +
     .     W*(4 + (-4 + Y1)*Y1 - Y2*(4 + Y2) + 2*T*(-2 + Y1 + Y2))) + 
     .     B*(GAT*T*(2*W*(-2 + Y1 + Y2) + (Y1 + Y2)**2) + GAB*
     .     (2*W**2*(Y1 - Y2) + (Y1 + Y2)**2*(-1 - T + Y1 + Y2) - 
     .     W*(Y1*(4 + Y1) - (-2 + Y2)**2 + 2*T*(-2 + Y1 + Y2)))))
      R44=4.d0*B**2-16.d0*B**3-8.d0*B*T+32.d0*B**2*T+4.d0*T**2-
     .     16.d0*B*T**2+4.d0*B*W-16.d0*B**2*W+4.d0*T*W-16.d0*B*T*W-
     .     8.d0*W**2+32.d0*B*W**2+8.d0*B*Y2-32.d0*B**2*Y2-8.d0*T*Y2+
     .     32.d0*B*T*Y2-16.d0*B*W*Y2-4.d0*B*Y1*Y2+4.d0*T*Y1*Y2+
     .     8.d0*W*Y1*Y2+4.d0*Y2**2-20.d0*B*Y2**2+4.d0*T*Y2**2+
     .     4.d0*W*Y2**2-4.d0*Y1*Y2**2-4.d0*Y2**3
      R14=-16.d0*B**2+32.d0*B*T-16.d0*T**2+8.d0*W-16.d0*B*W-16.d0*T*W+
     .     32.d0*W**2-16.d0*B*Y2+16.d0*T*Y2-8.d0*W*Y2-4.d0*Y2**2+
     .     16.d0*B*Y1-16.d0*T*Y1-8.d0*W*Y1+8.d0*Y2*Y1-4.d0*Y1**2
      R24= 4.d0*(GAT*T*(Y1*(2.d0*T - 2*W + Y1) + 2.d0*(T + W)*Y2 -
     .     Y2**2 - 2.d0*B*(Y1 + Y2)) + GAB*(-2.d0*W**2 + 
     .     W*(2.d0*B*(3.d0 - Y1 + Y2) + Y2*(Y1 + Y2) + 
     .     2.d0*(-1.d0 + T + Y1 + Y2)) - (Y1 + Y2)*(2.d0*B**2 + T + 
     .     Y2*(-1.d0 + Y1 + Y2) + B*(-1.d0 - 2*T + Y1 + 3.d0*Y2))))
      R34=-8.d0*W+24.d0*B*W-16.d0*B**2*W+8*T*W+32.d0*B*T*W-16.d0*T**2*W-
     .     16.d0*B*W**2-16.d0*T*W**2+32.d0*W**3+4.d0*B*Y1-8.d0*B**2*Y1-
     .     4.d0*T*Y1+16.d0*B*T*Y1-8.d0*T**2*Y1+8.d0*W*Y1-16.d0*W**2*Y1-
     .     4.d0*B*Y1**2-4.d0*T*Y1**2+4.d0*B*Y2-8.d0*B**2*Y2-4.d0*T*Y2+
     .     16.d0*B*T*Y2-8.d0*T**2*Y2+8.d0*W*Y2-16.d0*B*W*Y2+
     .     16.d0*T*W*Y2-24.d0*W**2*Y2+4.d0*Y1*Y2-16.d0*B*Y1*Y2+
     .     12.d0*W*Y1*Y2-4.d0*Y1**2*Y2+4.d0*Y2**2-12.d0*B*Y2**2+
     .     4.d0*T*Y2**2-4.d0*W*Y2**2-8.d0*Y1*Y2**2-4.d0*Y2**3

c - take off the charged H+ and W+ exchange -
      W22 = 0.D0
      W33 = 0.D0
      W12 = 0.D0
      W23 = 0.D0
      W13 = 0.D0
      W24 = 0.D0
      W34 = 0.D0
c -------------------------------------------

      RES=GHT**2*R11*W11+GLVV**2*R22*W22+
     .     4.D0*GHVV**2*R33*W33/T+2.D0*GHT*GLVV*R12*W12+
     .     2.D0*GHT*GHVV*R13*W13+2.D0*GHVV*GLVV*R23*W23/T+
     .     GHB**2*R44*W44*B/T+2.D0*B*GHB*GHT*R14*W14+
     .     2.D0*GHB*GLVV*R24*W24*B/T+
     .     2.d0*GHB*GHVV*R34*W34*B/T
      RETURN
      END

C     ************************************************************
C     SUBROUTINE FOR H+ ---> BT* ---> BBW
C     ************************************************************
      SUBROUTINE CTOTT_HDEC(AMCH,AMT,AMB,AMW,i2hdm,gat,gab,CTT0)
      IMPLICIT REAL*8(A-Z)
      INTEGER IP,K,i2hdm3,i2hdm
      COMMON/PREC1_HDEC/IP
      EXTERNAL FUNCTT1_HDEC
      COMMON/IKSY3_HDEC/X1,X2,M1,M2,M3,ECM,S
      COMMON/TOP3_HDEC/AMH3,AMT3,AMB3,AMW3
      COMMON/TOP5_HDEC/gat3,gab3,i2hdm3
      AMH3=AMCH
      AMT3=AMT
      AMB3=AMB
      AMW3=AMW
      gat3=gat
      gab3=gab
      i2hdm3=i2hdm
      IP=5
      M1=AMB
      M2=AMB
      M3=AMW
C     FIRST INTEGRATE OVER X2, i.e. (1+3) SYSTEM
C        CHECK WHETHER ENOUGH PHASE SPACE
      MASTOT=M1+M2+M3
      IF(MASTOT.GE.AMCH) GOTO 12
      ECM=AMCH
      S=ECM**2
      U1=(ECM-M2)**2
      D1=(M1+M3)**2
      U=(S-D1+M2**2)/s
      D=(S-U1+M2**2)/s
      DEL=(U-D)/IP
      U=D+DEL
      XSEC=0.D0
      DO K=1,IP
      CALL QGAUS1_HDEC(FUNCTT1_HDEC,D,U,SS)
      D=U
      U=D+DEL
      XSEC=XSEC+SS
      ENDDO
      CTT0=XSEC
12    CONTINUE
      RETURN
      END

      DOUBLE PRECISION FUNCTION FUNCTT1_HDEC(XL)
      IMPLICIT REAL*8(A-Z)
      INTEGER IP,I
      COMMON/IKSY3_HDEC/X1,X2,M1,M2,M3,ECM,S
      COMMON/PREC1_HDEC/IP
      EXTERNAL FUNCTT2_HDEC
      X2=XL
      S13=S-S*X2+M2**2
      TEM=2.D0*DSQRT(S13)
      E2S=(S-S13-M2**2)/TEM
      E3S=(S13+M3**2-M1**2)/TEM
C     SECOND INTEGRAL OVER X1, i.e. (2+3) SYSTEM
      U1=(E2S+E3S)**2-(DSQRT(E2S**2-M2**2)-DSQRT(E3S**2-M3**2))**2
      D1=(E2S+E3S)**2-(DSQRT(E2S**2-M2**2)+DSQRT(E3S**2-M3**2))**2
      U=(S-D1+M1**2)/s
      D=(S-U1+M1**2)/s
      DEL=(U-D)/IP
      FUNCTT1_HDEC=0.d0
      U=D+DEL
      DO I=1,IP
      CALL QGAUS2_HDEC(FUNCTT2_HDEC,D,U,SS)
      FUNCTT1_HDEC=FUNCTT1_HDEC+SS
      D=U
      U=D+DEL
      ENDDO
      RETURN
      END

      DOUBLE PRECISION FUNCTION FUNCTT2_HDEC(XK)
      IMPLICIT REAL*8(A-Z)
      COMMON/IKSY3_HDEC/X1,X2,M1,M2,M3,ECM,S
      X1=XK
      CALL ELEMCTT_HDEC(SS)
      FUNCTT2_HDEC=SS
      RETURN
      END

      SUBROUTINE ELEMCTT_HDEC(RES)
      IMPLICIT REAL*8(A-Z)
      integer i2hdm
      COMMON/IKSY3_HDEC/X1,X2,M1,M2,M3,ECM,S
      COMMON/TOP3_HDEC/AMCH,AMT,AMB,AMW
      COMMON/WZWDTH_HDEC/GAMC0,GAMT0,GAMT1,GAMW,GAMZ
      COMMON/TOP5_HDEC/gat,gab,i2hdm
      GAMT=GAMT1**2*AMT**2/AMCH**4
      W=AMW**2/AMCH**2
      T=AMT**2/AMCH**2
      B=AMB**2/AMCH**2

c ---- w/o b-quark mass dependence ----

      RES=((1.D0-X1-W)*(1.D0-X2-W)+W*(X1+X2-1.D0+W))/
     .   ((1.D0-X2+B-T)**2+GAMT)

c ---- w/ b-quark mass dependence ----

      res=(-2.d0*b**3*gab**2*w+gat**2*t**2*((-1.d0+2.d0*w)*(-1.d0+w+x1)+
     .     (-1.d0+2.d0*w+x1)*x2)+b*(-2.d0*gat**2*t**2*w+2*gab*gat*t*
     .     (1.d0+2.d0*w-x2)*(-1.d0+w+x2)+gab**2*(-2.d0*w**2+
     .     (-1.d0+x2)**2*(-1.d0+x1+x2)+w*(-1.d0+x2)*
     .     (-3.d0+2.d0*x1+x2)))+b**2*gab*(-4.d0*gat*t*w+gab*
     .     (2.d0*w**2+w*(3.d0-2.d0*x1)-(-1.d0+x2)*
     .     (-3.d0+x1+2.d0*x2))))/((1.d0-x2+b-t)**2+gamt)/t**2/gat**2

      RETURN
      END

C   *****************  INTEGRATION ROUTINE ***********************
C    Returns SS as integral of FUNC from A to B, by 10-point Gauss-
C    Legendre integration
      SUBROUTINE QGAUS1_HDEC(FUNC,A,B,SS)
      IMPLICIT REAL*8(A-Z)
      INTEGER J
      DIMENSION X(5),W(5)
      EXTERNAL FUNC
      DATA X/.1488743389D0,.4333953941D0,.6794095682D0
     .  ,.8650633666D0,.9739065285D0/
      DATA W/.2955242247D0,.2692667193D0,.2190863625D0
     .  ,.1494513491D0,.0666713443D0/
      XM=0.5D0*(B+A)
      XR=0.5D0*(B-A)
      SS=0.D0
      DO 11 J=1,5
        DX=XR*X(J)
        SS=SS+W(J)*(FUNC(XM+DX)+FUNC(XM-DX))
11    CONTINUE
      SS=XR*SS
      RETURN
      END

C     Returns SS as integral of FUNC from A to B, by 10-point Gauss-
C      Legendre integration
      SUBROUTINE QGAUS2_HDEC(FUNC,A,B,SS)
      IMPLICIT REAL*8(A-Z)
      INTEGER J
      DIMENSION X(5),W(5)
      EXTERNAL FUNC
      DATA X/.1488743389D0,.4333953941D0,.6794095682D0
     .  ,.8650633666D0,.9739065285D0/
      DATA W/.2955242247D0,.2692667193D0,.2190863625D0
     .  ,.1494513491D0,.0666713443D0/
      XM=0.5D0*(B+A)
      XR=0.5D0*(B-A)
      SS=0.D0
      DO 11 J=1,5
        DX=XR*X(J)
        SS=SS+W(J)*(FUNC(XM+DX)+FUNC(XM-DX))
11    CONTINUE
      SS=XR*SS
      RETURN
      END

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      SUBROUTINE AMHAMA_HDEC(ICASE,MH,TANB)
C--CALCULATION OF PSEUDOSCALAR HIGGS MASS FROM HIGGS MASS MH
C--ICASE=0: MH=PSEUDOSCALAR MASS
C--ICASE=1: MH=LIGHT SCALAR MASS
C--ICASE=2: MH=HEAVY SCALAR MASS
C--ICASE=3: MH=CHARGED HIGGS MASS
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      IMPLICIT REAL*8(A-H,L,M,O-Z)
      DIMENSION VH(2,2),M2(2,2),M2P(2,2)
      COMMON/HMASS_HDEC/AMSM,AMA,AML,AMH,AMCH,AMAR
      IF(ICASE.EQ.0)THEN
       MA = MH
      ELSE
       DEL0 = 1.D-4
       MA0 = 1.D0
       MA1 = 1.D4
1      MA = (MA0+MA1)/2
C      CALL SUBH1_HDEC(MA,TANB,MQ,MUR,MD,MTOP,AU,AD,MU,MCHI0,
C    *                 MHP,HMP,MCH,SA,CA,TANBA)
       AMA = MA
       CALL SUSYCP_HDEC(TANB)
       IF(ICASE.EQ.1)THEN
        MX = AML
       ELSEIF(ICASE.EQ.2)THEN
        MX = AMH
       ELSEIF(ICASE.EQ.3)THEN
        MX = AMCH
       ENDIF
       DEL = DABS(MA1 - MA0)/MA
       IF(DEL.GT.DEL0) THEN
        IF(MX.GT.MH) MA1 = MA
        IF(MX.LT.MH) MA0 = MA
        GOTO 1
       ENDIF
       FAC = 1
       MAX = DINT(FAC*MA+0.5D0)/FAC
C      CALL SUBH1_HDEC(MAX,TANB,MQ,MUR,MD,MTOP,AU,AD,MU,MCHI0,
C    *                 MHP,HMP,MCH,SA,CA,TANBA)
       AMA = MAX
       CALL SUSYCP_HDEC(TANB)
       IF(ICASE.EQ.1)THEN
        MX = AML
       ELSEIF(ICASE.EQ.2)THEN
        MX = AMH
       ELSEIF(ICASE.EQ.3)THEN
        MX = AMCH
       ENDIF
       IF(MX.EQ.MH)THEN
        MA = MAX
       ELSE
        DEL0 = 1.D-8
2       MA = (MA0+MA1)/2
C       CALL SUBH1_HDEC(MA,TANB,MQ,MUR,MD,MTOP,AU,AD,MU,MCHI0,
C    *                  MHP,HMP,MCH,SA,CA,TANBA)
        AMA = MA
        CALL SUSYCP_HDEC(TANB)
        IF(ICASE.EQ.1)THEN
         MX = AML
        ELSEIF(ICASE.EQ.2)THEN
         MX = AMH
        ELSEIF(ICASE.EQ.3)THEN
         MX = AMCH
        ENDIF
        DEL = DABS(MA1 - MA0)/MA
        IF(DEL.GT.DEL0) THEN
         IF(MX.GT.MH) MA1 = MA
         IF(MX.LT.MH) MA0 = MA
         GOTO 2
        ENDIF
       ENDIF
      ENDIF
      AMA = MA
      CALL SUSYCP_HDEC(TANB)
      RETURN
      END
C
C     ****************************************************************
C	  CHARGINO AND NEUTRALINO MASS MATRICES AND COUPLINGS
C     ****************************************************************
      SUBROUTINE GAUGINO_HDEC(MU,M2,B,A,MC,MN,XMN,AC1,AC2,AC3,AN1,AN2
     .                 ,AN3,ACNL,ACNR,AGDL,AGDA,AGDH,AGDC)            
      IMPLICIT REAL*8(A-H,K-Z)
      COMPLEX*16 CXA,CXB,CXC,CXD,CX1,CX2,CX3
      DIMENSION MC(2),MN(4),XMN(4),Z(4,4),ZX(4,4),U(2,2),V(2,2),
     .          QQ(4,4),SS(4,4),S(2,2),Q(2,2),AC1(2,2),AC2(2,2),
     .          AC3(2,2),AN1(4,4),AN2(4,4),AN3(4,4),ACNL(2,4),
     .          ACNR(2,4),IORD(4),IREM(2)
      DIMENSION X(2,2)
      DIMENSION YMN(4),YZ(4,4),XMC(2),BU(2),BV(2)
      DIMENSION AGDL(4),AGDA(4),AGDH(4),AGDC(2)
      DIMENSION slhaneut(4),slhaxneut(4),slhachar(2),slhau(2,2),
     .          slhav(2,2),slhaz(4,4),slhaxchar(2)
      COMMON/PARAM_HDEC/GF,ALPH,AMTAU,AMMUON,MZ,MW
      COMMON/GAUGINOMIX_HDEC/ZX,U,V
      COMMON/SLHA_vals_HDEC/islhai,islhao
      COMMON/SLHA_m1_HDEC/am1
      COMMON/SLHA_gaug_HDEC/slhaneut,slhaxneut,slhachar,slhau,slhav,
     .                      slhaz,slhaxchar
      CW=MW/MZ
      SW=DSQRT(1-CW**2)
      PI=4.D0*DATAN(1.D0)
      SB=DSIN(B)
      CB=DCOS(B)
      TW=SW/CW
      if(islhai.eq.0) then
         M1=5.D0/3.D0*TW**2*M2
      else
         M1 = am1
      endif
C     ************  NEUTRALINO MASSES AND MATRIX ELEMENTS ***********
      EPS=-1.D-10
      XC2=(M1*M2-MZ**2-MU**2)-3.D0/8.D0*(M1+M2)**2
      XC3=-1.D0/8.D0*(M1+M2)**3+1.D0/2.D0*(M1+M2)*(M1*M2-MZ**2
     .    -MU**2)+(M1+M2)*MU**2+(M1*CW**2+M2*SW**2)*MZ**2
     .    -MU*MZ**2*DSIN(2.D0*B)
      XC4=+(M1*CW**2+M2*SW**2)*MU*MZ**2*DSIN(2.D0*B)-M1*M2*MU**2
     .    +1.D0/4.D0*(M1+M2)*( (M1+M2)*MU**2+(M1*CW**2+M2*SW**2)
     .    *MZ**2-MU*MZ**2*DSIN(2.D0*B) )+1.D0/16.D0*(M1+M2)**2*
     .    (M1*M2-MZ**2-MU**2)-3.D0/256.D0*(M1+M2)**4
      XS=-XC3**2-2.D0/27.D0*XC2**3+8.D0/3.D0*XC2*XC4
      XU=-1.D0/3.D0*XC2**2-4.D0*XC4
      CXD=(-4*XU**3-27*XS**2)*DCMPLX(1.D0,EPS)
      CXC=1.D0/2.D0*(-XS+DCMPLX(0.D0,1.D0)*CDSQRT(CXD/27.D0))
      CXA=DREAL(CXC**(1.D0/3.D0))*DCMPLX(1.D0,-EPS)
      CXB=8.D0*CXA-8.D0/3.D0*XC2*DCMPLX(1.D0,-EPS)
C     *********** MASSES AND COUPLINGS:
      if(islhai.eq.0) then
         X0=(M1+M2)/4.D0
         CX1= CXA/2.D0-XC2/6.D0*DCMPLX(1.D0,-EPS)
         CX2=-CXA/2.D0-XC2/3.D0*DCMPLX(1.D0,-EPS)
         CX3=XC3*DCMPLX(1.D0,-EPS)/CDSQRT(CXB)
         XMN(1)=X0-CDABS(CDSQRT(CX1))+CDABS(CDSQRT(CX2+CX3))
         XMN(2)=X0+CDABS(CDSQRT(CX1))-CDABS(CDSQRT(CX2-CX3))
         XMN(3)=X0-CDABS(CDSQRT(CX1))-CDABS(CDSQRT(CX2+CX3))
         XMN(4)=X0+CDABS(CDSQRT(CX1))+CDABS(CDSQRT(CX2-CX3))
         DO 10 I=1,4
            MN(I)=DABS(XMN(I))
            YMN(I)=XMN(I)
            ZX(I,2)=-CW/SW*(M1-XMN(I))/(M2-XMN(I))
            ZX(I,3)=(MU*(M2-XMN(I))*
     .           (M1-XMN(I))-MZ**2*SB*CB*((M1-M2)*CW**2
     .           +M2-XMN(I)))/MZ/(M2-XMN(I))/SW/(MU*CB+XMN(I)*SB)
            ZX(I,4)=(-XMN(I)*(M2-XMN(I))*(M1-XMN(I))-MZ**2*CB*CB*
     .           ((M1-M2)*CW**2+M2-XMN(I)))/MZ/(M2-XMN(I))
     .           /SW/(MU*CB+XMN(I)*SB)
            ZX(I,1)=1.D0/DSQRT(1.D0+ZX(I,2)**2+ZX(I,3)**2+ZX(I,4)**2) 
            YZ(I,1)=ZX(I,1)
            YZ(I,2)=ZX(I,2)*ZX(I,1)
            YZ(I,3)=ZX(I,3)*ZX(I,1)
            YZ(I,4)=ZX(I,4)*ZX(I,1)
 10      CONTINUE
      else
         do i=1,4,1
            xmn(i) = slhaxneut(i)
            mn(i)  = dabs(xmn(i))
            ymn(i) = xmn(i)
            do j=1,4,1
               zx(i,j) = slhaz(i,j)
            end do
            yz(i,1)=zx(i,1)
            yz(i,2)=zx(i,2)
            yz(i,3)=zx(i,3)
            yz(i,4)=zx(i,4)
         end do
      endif
C     *************  ORDERING THE DISORDER ******************
      XX0 = DMIN1(MN(1),MN(2),MN(3),MN(4))
      XX1 = DMAX1(MN(1),MN(2),MN(3),MN(4))
      IDUMMY = 1
      DO I = 1,4
       IF(MN(I).EQ.XX0)THEN
        IORD(1) = I
       ELSEIF(MN(I).EQ.XX1)THEN
        IORD(4) = I
       ELSE
        IREM(IDUMMY) = I
        IDUMMY = IDUMMY+1
       ENDIF
      ENDDO
      IF(MN(IREM(1)).LE.MN(IREM(2)))THEN
       IORD(2) = IREM(1)
       IORD(3) = IREM(2)
      ELSE
       IORD(2) = IREM(2)
       IORD(3) = IREM(1)
      ENDIF
C 
      DO 98 J=1,4
      I=IORD(J)
      XMN(J)=YMN(I)
      MN(J) =DABS(YMN(I))
        DO I1=1,4
        Z(J,I1)=YZ(I,I1)
        ENDDO
 98   CONTINUE
C     ************  NEUTRALINO COUPLINGS TO HIGGS BOSONS ***********
	DO 11 I=1,4
	DO 11 J=1,4
	QQ(I,J)=1.D0/2.D0*(Z(I,3)*(Z(J,2)-TW*Z(J,1))+Z(J,3)*
     .		(Z(I,2)-TW*Z(I,1)))
	SS(I,J)=1.D0/2.D0*(Z(I,4)*(Z(J,2)-TW*Z(J,1))+Z(J,4)*
     .		(Z(I,2)-TW*Z(I,1)))
 11	CONTINUE
	DO 21 I=1,4
	DO 21 J=1,4
	AN1(I,J)= QQ(I,J)*DCOS(A)-SS(I,J)*DSIN(A)
	AN2(I,J)=-QQ(I,J)*DSIN(A)-SS(I,J)*DCOS(A)
	AN3(I,J)=-QQ(I,J)*DSIN(B)+SS(I,J)*DCOS(B)
 21	CONTINUE

C       ************* CHARGINO MASSES AND MATRIX ELEMENTS ***********
        if(islhai.eq.0) then
           DELTA=DABS(B-.25*PI)
           DDD=MU*DCOS(B)+M2*DSIN(B)
           CCC=MU*DSIN(B)+M2*DCOS(B)
           IF(DELTA.LT.0.01D0) THEN
              PHIM=PI/4.D0-.5D0*DATAN((M2-MU)/(2.D0*MW))
              PHIP=PHIM
           ELSE IF	(DABS(CCC).LT.1.D-5) THEN
              PHIM=0.D0
              PHIP=DATAN(DSQRT(2.D0)*MW*DSIN(B)/(M2+1.D-5))
           ELSE IF	(DABS(DDD).LT.1.D-5) THEN
              PHIP=0.D0
              PHIM=DATAN(DSQRT(2.D0)*MW*DCOS(B)/(M2+1.D-5))
           ELSE
              RAD=DSQRT((M2**2-MU**2)**2+4.D0*MW**4*DCOS(2.D0*B)**2
     +             +4.D0*MW**2*(M2**2+MU**2+2.D0*M2*MU*DSIN(2.D0*B)))
              PHIP=DATAN((RAD-(M2**2-MU**2+2.D0*MW**2*DCOS(2.D0*B)))
     +             /(2.D0*DSQRT(2.D0)*MW*(MU*DCOS(B)+M2*DSIN(B))))
              PHIM=DATAN((RAD-(M2**2-MU**2-2.D0*MW**2*DCOS(2.D0*B)))
     +             /(2.D0*DSQRT(2.D0)*MW*(MU*DSIN(B)+M2*DCOS(B))))
           ENDIF
           CP=DCOS(PHIP)
           SP=DSIN(PHIP)
           CM=DCOS(PHIM)
           SM=DSIN(PHIM)
C MY CONVENTION
           U(2,2)=CM
           U(2,1)=-SM
           U(1,2)=SM
           U(1,1)=CM
           V(1,1)=CP
           V(1,2)=SP
           V(2,1)=-SP
           V(2,2)=CP
           X(1,1)=M2
           X(1,2)=DSQRT(2.D0)*MW*DSIN(B)
           X(2,1)=DSQRT(2.D0)*MW*DCOS(B)
           X(2,2)=MU
 555       CONTINUE
           XMC(1)=(U(1,1)*X(1,1)+U(1,2)*X(2,1))*V(1,1)
     .          +(U(1,1)*X(1,2)+U(1,2)*X(2,2))*V(1,2)
           XMC(2)=(U(2,1)*X(1,1)+U(2,2)*X(2,1))*V(2,1)
     .          +(U(2,1)*X(1,2)+U(2,2)*X(2,2))*V(2,2)
           IF(XMC(1).LT.0.D0) THEN
              V(1,1)=-CP
              V(1,2)=-SP
              V(2,1)=-SP
              V(2,2)=CP
              GOTO 555
           ENDIF
           IF(XMC(2).LT.0.D0) THEN
              V(1,1)=CP
              V(1,2)=SP
              V(2,1)=SP
              V(2,2)=-CP
              GOTO 555
           ENDIF
           IF(XMC(1).GT.XMC(2)) THEN
              MTEMP=XMC(1)
              XMC(1)=XMC(2)
              XMC(2)=MTEMP
              DO J=1,2
                 BU(J)=U(1,J)
                 U(1,J)=U(2,J)
                 U(2,J)=BU(J)
                 BV(J)=V(1,J)
                 V(1,J)=V(2,J)
                 V(2,J)=BV(J)
              ENDDO
           ENDIF        
           MC(1)=DABS(XMC(1))
           MC(2)=DABS(XMC(2))
           slhaxchar(1) = mc(1)
           slhaxchar(2) = mc(2)
        else
           mc(1) = slhachar(1)
           mc(2) = slhachar(2)
           do i=1,2,1
              do j=1,2,1
                 u(i,j) = slhau(i,j)
                 v(i,j) = slhav(i,j)
              end do
           end do
        endif

C     ************  CHARGINO COUPLINGS TO HIGGS BOSONS ***********
	DO 12 I=1,2
	DO 12 J=1,2
	Q(I,J)=DSQRT(1.D0/2.D0)*U(J,2)*V(I,1)
	S(I,J)=DSQRT(1.D0/2.D0)*U(J,1)*V(I,2)
 12	CONTINUE
	DO 22 I=1,2
	DO 22 J=1,2	
	AC1(I,J)= Q(I,J)*DCOS(A)+S(I,J)*DSIN(A)
	AC2(I,J)=-Q(I,J)*DSIN(A)+S(I,J)*DCOS(A)
	AC3(I,J)=-Q(I,J)*DSIN(B)-S(I,J)*DCOS(B)
 22	CONTINUE
C     **** CHARGINO-NEUTRALINO COUPLINGS TO CHARGED HIGGS BOSONS 
	DO 13 I=1,2
	DO 13 J=1,4
        ACNL(I,J)=DCOS(B)*(Z(J,4)*V(I,1)+(Z(J,2)+Z(J,1)*TW)
     .       *V(I,2)/DSQRT(2.D0)) 
        ACNR(I,J)=DSIN(B)*(Z(J,3)*U(I,1)-(Z(J,2)+Z(J,1)*TW)
     .       *U(I,2)/DSQRT(2.D0)) 
 13     CONTINUE

C   ************* HIGGS--NEUTRALINO--GOLDSTINO COUPLINGS
      DO 51 I=1,4
      AGDL(I)=Z(I,3)*DSIN(A)-Z(I,4)*DCOS(A)
      AGDH(I)=Z(I,3)*DCOS(A)+Z(I,4)*DSIN(A)
      AGDA(I)=Z(I,3)*DSIN(B)+Z(I,4)*DCOS(B)
 51   CONTINUE
C
C   ************* CHARGED HIGGS--CHARGINO--GOLDSTINO COUPLINGS
      AGDC(1)=DSQRT( V(1,2)**2*DCOS(B)**2+ U(1,2)**2*DSIN(B)**2 )
      AGDC(2)=DSQRT( V(2,2)**2*DCOS(B)**2+ U(2,2)**2*DSIN(B)**2 )

       RETURN
       END

C   ****************************************************************
C     SUBROUTINE FOR SFERMION MASSES, MIXING AND COUPLINGS 
C   ****************************************************************

      SUBROUTINE SFERMION_HDEC(TSC,BSC,MQL,MUR,MDR,MEL,MER,AL,AT,AB,MU,
     .                    MST,MSB,MSL,MSU,MSD,MSE,MSN,MSN1,
     .                    GLEE,GLTT,GLBB,GHEE,GHTT,GHBB,
     .                    GAEE,GATT,GABB,GCEN,GCTB)

      IMPLICIT REAL*8(A-H,K-Z)
      DIMENSION MST(2),MSB(2),MSL(2),MSU(2),MSD(2),MSE(2),MSN(2),
     .          MSN1(2),
     .          GCEN(2,2),GCTB(2,2),GLEE(2,2),GLTT(2,2),GLBB(2,2),
     .          GHEE(2,2),GHTT(2,2),GHBB(2,2)
      DIMENSION slhast(2),slhasb(2),slhasu(2),slhasd(2),slhase(2),
     .          slhasl(2),slhasn(2),slhasnl(2)
      COMMON/MASSES_HDEC/AMS,AMC,AMB,AMT
      COMMON/PARAM_HDEC/GF,ALPH,AMTAU,AMMUON,MZ,MW
      COMMON/COUP_HDEC/GAT,GAB,GLT,GLB,GHT,GHB,GZAH,GZAL,GHHH,GLLL,GHLL,
     .            GLHH,GHAA,GLAA,GLVV,GHVV,GLPM,GHPM,B,A
      COMMON/HMASS_HDEC/AMSM,AMA,AML,AMH,AMCH,AMAR
      COMMON/SFER1ST_HDEC/MQL1,MUR1,MDR1,MEL1,MER1
      COMMON/GLUINO_HDEC/AMGLUINO,XMSB1,XMSB2,STHB,CTHB,
     .              XLBB(2,2),XHBB(2,2),XABB(2,2),
     .              XMST1,XMST2,STHT,CTHT,
     .              XLTT(2,2),XHTT(2,2),XATT(2,2)
      COMMON/TAUMIX_HDEC/CL,SL
      COMMON/SLHA_vals_HDEC/islhai,islhao
      COMMON/SLHA_sfer_HDEC/slhast,slhasb,slhasu,slhasd,slhase,slhasl,
     .                 slhasn,slhasnl,slhacot,slhasit,slhacob,slhasib,
     .                 slhacol,slhasil
      COMMON/SQUARKHIGGS_HDEC/THEB,AMG,JONSH,JDTH
      COMMON/QTBNLO_HDEC/SMTOP,SMBOT
      COMMON/SQNLO_HDEC/YMSB(2),YSTHB,YCTHB,YLBB(2,2),YHBB(2,2),YABB,
     .                  YMST(2),YSTHT,YCTHT,YLTT(2,2),YHTT(2,2),YATT
      COMMON/BREAKSCALE_HDEC/SUSYSCALE
      COMMON/TRILINEAR_HDEC/AT00,AB00
      PMSQ1(QQ,AM,AMG,ALS) = 4*ALS/3/PI*(AMG**2*DLOG(QQ**2/AMG**2)
     . + AM**2/2*DLOG(AMG**2/AM**2) + AM**2/2 + 3*AMG**2/2
     . + (AMG**2-AM**2)**2/2/AM**2*DLOG(DABS(AMG**2-AM**2)/AMG**2))
      PMSQ10(QQ,AM,ALS) = 4*ALS/3/PI*(AM**2*DLOG(QQ**2/AM**2)
     . + AM**2/2 + 3*AM**2/2)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c-- comparison with Luminita
c     PMSQ1(QQ,AM,AMG,ALS) = 0
c     PMSQ10(QQ,AM,ALS) = 0
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
C
      PI = 4*DATAN(1.D0)
      SW2=1.D0-MW**2/MZ**2
      TB=DTAN(B)
      AMG = AMGLUINO
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      MT = AMT
      MB = AMB
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     MT = RUNM_HDEC(TSC,6)
c     MB = RUNM_HDEC(BSC,5)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     MT = RUNM_HDEC(AMT,6)
c     MB = RUNM_HDEC(AMT,5)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      ML = AMTAU
C FIRST TWO GENERATIONS:  NO MIXING INCLUDED 
      if(islhai.eq.0) then
C UP SQUARKS: 
         MSQ0=DSQRT(MQL1**2+(0.5D0-2.D0/3.D0*SW2)*MZ**2*DCOS(2.D0*B))
         IF(AMG.NE.MSQ0)THEN
          MSTL2=MSQ0**2
     .         +PMSQ1(SUSYSCALE,MSQ0,AMG,ALPHAS_HDEC(SUSYSCALE,3))
         ELSE
          MSTL2=MSQ0**2
     .         + PMSQ10(SUSYSCALE,MSQ0,ALPHAS_HDEC(SUSYSCALE,3))
         ENDIF
         MSQ0=DSQRT(MUR1**2+2.D0/3.D0*SW2*MZ**2*DCOS(2.D0*B))
         IF(AMG.NE.MSQ0)THEN
          MSTR2=MSQ0**2
     .         + PMSQ1(SUSYSCALE,MSQ0,AMG,ALPHAS_HDEC(SUSYSCALE,3))
         ELSE
          MSTR2=MSQ0**2
     .         + PMSQ10(SUSYSCALE,MSQ0,ALPHAS_HDEC(SUSYSCALE,3))
         ENDIF
         MSU(1)=DSQRT(MSTL2)
         MSU(2)=DSQRT(MSTR2)
C DOWN SQUARKS
         MSQ0=DSQRT(MQL1**2+(-0.5D0+1.D0/3.D0*SW2)*MZ**2*DCOS(2.D0*B))
         IF(AMG.NE.MSQ0)THEN
          MSBL2=MSQ0**2
     .         + PMSQ1(SUSYSCALE,MSQ0,AMG,ALPHAS_HDEC(SUSYSCALE,3))
         ELSE
          MSBL2=MSQ0**2
     .         + PMSQ10(SUSYSCALE,MSQ0,ALPHAS_HDEC(SUSYSCALE,3))
         ENDIF
         MSQ0=DSQRT(MDR1**2-1.D0/3.D0*SW2*MZ**2*DCOS(2.D0*B))
         IF(AMG.NE.MSQ0)THEN
          MSBR2=MSQ0**2
     .         + PMSQ1(SUSYSCALE,MSQ0,AMG,ALPHAS_HDEC(SUSYSCALE,3))
         ELSE
          MSBR2=MSQ0**2
     .         + PMSQ10(SUSYSCALE,MSQ0,ALPHAS_HDEC(SUSYSCALE,3))
         ENDIF
         MSD(1)=DSQRT(MSBL2)
         MSD(2)=DSQRT(MSBR2)
C SLEPTONS
         MSEL2=MEL1**2+(-0.5D0+SW2)*MZ**2*DCOS(2.D0*B)
         MSER2=MER1**2- SW2*MZ**2*DCOS(2.D0*B) 
         MSNL2=MEL1**2+0.5D0*MZ**2*DCOS(2.D0*B)
         MSE(1)=DSQRT(MSEL2)
         MSE(2)=DSQRT(MSER2)
         MSN1(1)=DSQRT(MSNL2)
         MSN1(2)=1.D+15

C NOW THE THIRD GENERATION
C
C STOP MASSES/MIXING
C
      MSTL2=MQL**2+(0.5D0-2.D0/3.D0*SW2)*MZ**2*DCOS(2.D0*B)
      MSTR2=MUR**2+2.D0/3.D0*SW2*MZ**2*DCOS(2.D0*B) 
      MLRT=AT-MU/TB
      DELT=(MSTL2-MSTR2)**2+4*MT**2*MLRT**2
      MST12=MT**2+0.5D0*(MSTL2+MSTR2-DSQRT(DELT))
      MST22=MT**2+0.5D0*(MSTL2+MSTR2+DSQRT(DELT))
c       IF(MST12.LT.0.D0)THEN 
c     PRINT *, 'MSTOP**2 is negative!!!!'
c     GOTO 111 
c     ELSE 
      MST(1)=DSQRT(MST12)
      MST(2)=DSQRT(MST22)
      IF(MSTL2.EQ.MSTR2) THEN
       THET = PI/4
      ELSE
       THET=0.5D0*DATAN(2.D0*MT*MLRT / (MSTL2-MSTR2) )
       IF(MSTL2.GT.MSTR2) THET = THET + PI/2
      ENDIF
c       ENDIF 
      CT= DCOS(THET)
      ST= DSIN(THET) 
c     write(6,*)'stop_LO:  ',CT,ST
C
C SBOTTOM MASSES/MIXING
C
      MSBL2=MQL**2+(-0.5D0+1.D0/3.D0*SW2)*MZ**2*DCOS(2.D0*B)
      MSBR2=MDR**2-1.D0/3.D0*SW2*MZ**2*DCOS(2.D0*B) 
      MLRB=AB-MU*TB
      DELB=(MSBL2-MSBR2)**2+4*MB**2*MLRB**2
      MSB12=MB**2+0.5D0*(MSBL2+MSBR2-DSQRT(DELB))
      MSB22=MB**2+0.5D0*(MSBL2+MSBR2+DSQRT(DELB))
c       IF(MSB12.LT.0.D0)THEN
c     PRINT *, 'MSBOT**2 is negative!!!!'
c     GOTO 111
c       ELSE
      MSB(1)=DSQRT(MSB12)
      MSB(2)=DSQRT(MSB22)
      IF(MSBL2.EQ.MSBR2) THEN
       THEB = PI/4
      ELSE
       THEB=0.5D0*DATAN(2.D0*MB*MLRB / (MSBL2-MSBR2) )
       IF(MSBL2.GT.MSBR2) THEB = THEB + PI/2
      ENDIF
c       ENDIF  
      CB= DCOS(THEB)
      SB= DSIN(THEB) 
c     write(6,*)'MAT_SB:   ',MSBL2,MB*(AB-MU*TB),MSBR2
c     write(6,*)'CTH, STH: ',CB,SB
C
C  STAU MASSES/MIXING
C
      MSEL2=MEL**2+(-0.5D0+SW2)*MZ**2*DCOS(2.D0*B)
      MSER2=MER**2- SW2*MZ**2*DCOS(2.D0*B) 
      MSNL2=MEL**2+0.5D0*MZ**2*DCOS(2.D0*B)
      MLRE=AL-MU*TB
      DELE=(MSEL2-MSER2)**2+4*ML**2*MLRE**2
      MSE12=ML**2+0.5D0*(MSEL2+MSER2-DSQRT(DELE))
      MSE22=ML**2+0.5D0*(MSEL2+MSER2+DSQRT(DELE))
        IF(MSE12.LT.0.D0)THEN
      PRINT *, 'MSTAU**2 is negative!!!!'
      GOTO 111
        ELSE
      MSL(1)=DSQRT(MSE12)
      MSL(2)=DSQRT(MSE22)
      MSN(1)=DSQRT(MSNL2)
      MSN(2)=1.D+15
      IF(MSEL2.EQ.MSER2) THEN
       THEL = PI/4
      ELSE
       THEL=0.5D0*DATAN(2.D0*ML*MLRE / (MSEL2-MSER2) )
       IF(MSEL2.GT.MSER2) THEL = THEL + PI/2
      ENDIF
        ENDIF  
      CL= DCOS(THEL)
      SL= DSIN(THEL) 

      else
         do i=1,2,1
            msu(i) = slhasu(i)
            msd(i) = slhasd(i)
            mse(i) = slhase(i)
            msn1(i) = slhasnl(i)
            mst(i) = slhast(i)
            msb(i) = slhasb(i)
            msl(i) = slhasl(i)
            msn(i) = slhasn(i)
         end do
         ct = slhacot
         st = slhasit
         cb = slhacob
         sb = slhasib
         cl = slhacol
         sl = slhasil
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
         cf = 4.d0/3
c        write(6,*)
c        write(6,*)'theta: ',2*st*ct,2*sb*cb,2*sl*cl
         thet = dasin(2*st*ct)/2
         if(ct**2-st**2.lt.0.d0) thet = pi/2-thet
         ct= dcos(thet)
         st= dsin(thet) 
         theb = dasin(2*sb*cb)/2
         if(cb**2-sb**2.lt.0.d0) theb = pi/2-theb
         cb= dcos(theb)
         sb= dsin(theb) 
         thel = dasin(2*sl*cl)/2
         if(cl**2-sl**2.lt.0.d0) thel = pi/2-thel
         cl= dcos(thel)
         sl= dsin(thel) 
c        write(6,*)'theta: ',2*st*ct,2*sb*cb,2*sl*cl
         delta_b = 2*amg*mu*dtan(b)*t_hdec(msb(1),msb(2),amg)
         alsscb = (msb(1)+msb(2)+dabs(amg))/3
         alssct = (mst(1)+mst(2)+dabs(amg))/3
         rmb = runm_hdec(susyscale,5)
     .       / (1+cf/4*alphas_hdec(alsscb,3)/pi*delta_b)
         rmt = runm_hdec(susyscale,6)
         mstl2=mql**2+(0.5d0-2.d0/3.d0*sw2)*mz**2*dcos(2.d0*b)
         mstr2=mur**2+2.d0/3.d0*sw2*mz**2*dcos(2.d0*b) 
         msbl2=mql**2+(-0.5d0+1.d0/3.d0*sw2)*mz**2*dcos(2.d0*b)
         msbr2=mdr**2-1.d0/3.d0*sw2*mz**2*dcos(2.d0*b) 
         msel2=mel**2+(-0.5d0+sw2)*mz**2*dcos(2.d0*b)
         mser2=mer**2-sw2*mz**2*dcos(2.d0*b) 
         s2t = 2*rmt*(at-mu/dtan(b))/(mst(1)**2-mst(2)**2)
c        c2t = (mstl2-mstr2)/(mst(1)**2-mst(2)**2)
         c2t = dsqrt(1-s2t**2)
         if(mstl2.gt.mstr2) c2t = -c2t
         s2b = 2*rmb*(ab-mu*dtan(b))/(msb(1)**2-msb(2)**2)
c        c2b = (msbl2-msbr2)/(msb(1)**2-msb(2)**2)
         c2b = dsqrt(1-s2b**2)
         if(msbl2.gt.msbr2) c2b = -c2b
c        s2l = 2* ml*(al-mu*dtan(b))/(msl(1)**2-msl(2)**2)
c        c2l = (msel2-mser2)/(msl(1)**2-msl(2)**2)
c        c2l = dsqrt(1-s2l**2)
c        if(msel2.gt.mser2) c2l = -c2l
         thet = datan(s2t/c2t)/2
         if(c2t.lt.0.d0) thet = pi/2+thet
         ct= dcos(thet)
         st= dsin(thet) 
         theb = datan(s2b/c2b)/2
         if(c2b.lt.0.d0) theb = pi/2+theb
         cb= dcos(theb)
         sb= dsin(theb) 
c        thel = datan(s2l/c2l)/2
c        if(c2l.lt.0.d0) thel = pi/2+thel
c        cl= dcos(thel)
c        sl= dsin(thel) 
c        write(6,*)'theta: ',2*st*ct,2*sb*cb,2*sl*cl
c        write(6,*)'theta: ',s2t,s2b
c        write(6,*)'theta: ',st**2+ct**2,sb**2+cb**2,sl**2+cl**2
c        write(6,*)'theta: ',s2t**2+c2t**2,s2b**2+c2b**2
c        write(6,*)'theta: ',rmt,rmb,ml
c    .             ,cf/4*alphas_hdec(alsscb,3)/pi*delta_b
c        write(6,*)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      endif

      C2T = CT**2-ST**2
      C2B = CB**2-SB**2
      C2L = CL**2-SL**2
C
C LIGHT CP--EVEN HIGGS COUPLINGS TO STOPS
C 
      GLTT(1,1)=-DSIN(B+A)*(0.5D0*CT**2-2.D0/3.D0*SW2*C2T) 
     .    + MT**2/MZ**2*GLT + MT*ST*CT/MZ**2*(AT*GLT+MU*GHT)
      GLTT(2,2)=-DSIN(B+A)*(0.5D0*ST**2+2.D0/3.D0*SW2*C2T)
     .    + MT**2/MZ**2*GLT - MT*ST*CT/MZ**2*(AT*GLT+MU*GHT)
      GLTT(1,2)=-2*DSIN(B+A)*ST*CT*(2.D0/3.D0*SW2-0.25D0)
     .    + MT*C2T/2.D0/MZ**2*(AT*GLT+MU*GHT) 
      GLTT(2,1)=-2*DSIN(B+A)*ST*CT*(2.D0/3.D0*SW2-0.25D0)
     .    + MT*C2T/2.D0/MZ**2*(AT*GLT+MU*GHT) 
C
C LIGHT CP--EVEN HIGGS COUPLINGS TO SBOTTOMS
C
      GLBB(1,1)=-DSIN(B+A)*(-0.5D0*CB**2+1.D0/3.D0*SW2*C2B) 
     .    + MB**2/MZ**2*GLB + MB*SB*CB/MZ**2*(AB*GLB-MU*GHB)
      GLBB(2,2)=-DSIN(B+A)*(-0.5D0*SB**2-1.D0/3.D0*SW2*C2B) 
     .    + MB**2/MZ**2*GLB - MB*SB*CB/MZ**2*(AB*GLB-MU*GHB)
      GLBB(1,2)=-2*DSIN(B+A)*SB*CB*(-1.D0/3.D0*SW2+0.25D0)
     .    + MB*C2B/2.D0/MZ**2*(AB*GLB-MU*GHB) 
      GLBB(2,1)=-2*DSIN(B+A)*SB*CB*(-1.D0/3.D0*SW2+0.25D0)
     .    + MB*C2B/2.D0/MZ**2*(AB*GLB-MU*GHB) 

C
C LIGHT CP--EVEN HIGGS COUPLINGS TO STAU'S 
C
      GLEE(1,1)=-DSIN(B+A)*(-0.5D0*CL**2+SW2*C2L) 
     .    + ML**2/MZ**2*GLB + ML*SL*CL/MZ**2*(AL*GLB-MU*GHB)
      GLEE(2,2)=-DSIN(B+A)*(-0.5D0*SL**2-SW2*C2L) 
     .    + ML**2/MZ**2*GLB - ML*SL*CL/MZ**2*(AL*GLB-MU*GHB)
      GLEE(1,2)=-2*DSIN(B+A)*SL*CL*(-SW2+0.25D0)
     .    + ML*C2L/2.D0/MZ**2*(AL*GLB-MU*GHB) 
      GLEE(2,1)=-2*DSIN(B+A)*SL*CL*(-SW2+0.25D0)
     .    + ML*C2L/2.D0/MZ**2*(AL*GLB-MU*GHB) 
C
C HEAVY CP--EVEN HIGGS COUPLINGS TO STOPS
C
      GHTT(1,1)=DCOS(B+A)*(0.5D0*CT**2-2.D0/3.D0*SW2*C2T) 
     .    + MT**2/MZ**2*GHT + MT*ST*CT/MZ**2*(AT*GHT-MU*GLT)
      GHTT(2,2)= DCOS(B+A)*(0.5D0*ST**2+2.D0/3.D0*SW2*C2T)
     .    + MT**2/MZ**2*GHT - MT*ST*CT/MZ**2*(AT*GHT-MU*GLT)
      GHTT(1,2)=2*DCOS(B+A)*ST*CT*(2.D0/3.D0*SW2-0.25D0)
     .    + MT*C2T/2.D0/MZ**2*(AT*GHT-MU*GLT) 
      GHTT(2,1)=2*DCOS(B+A)*ST*CT*(2.D0/3.D0*SW2-0.25D0)
     .    + MT*C2T/2.D0/MZ**2*(AT*GHT-MU*GLT) 
C
C HEAVY CP--EVEN HIGGS COUPLINGS TO SBOTTOMS
C
      GHBB(1,1)= DCOS(B+A)*(-0.5D0*CB**2+1.D0/3.D0*SW2*C2B) 
     .    + MB**2/MZ**2*GHB + MB*SB*CB/MZ**2*(AB*GHB+MU*GLB)
      GHBB(2,2)= DCOS(B+A)*(-0.5D0*SB**2-1.D0/3.D0*SW2*C2B) 
     .    + MB**2/MZ**2*GHB - MB*SB*CB/MZ**2*(AB*GHB+MU*GLB)
      GHBB(1,2)=2*DCOS(B+A)*SB*CB*(-1.D0/3.D0*SW2+0.25D0)
     .    + MB*C2B/2.D0/MZ**2*(AB*GHB+MU*GLB) 
      GHBB(2,1)=2*DCOS(B+A)*SB*CB*(-1.D0/3.D0*SW2+0.25D0)
     .    + MB*C2B/2.D0/MZ**2*(AB*GHB+MU*GLB) 
C
C HEAVY CP--EVEN HIGGS COUPLINGS TO STAU'S 
C
      GHEE(1,1)= DCOS(B+A)*(-0.5D0*CL**2+SW2*C2L) 
     .    + ML**2/MZ**2*GHB + ML*SL*CL/MZ**2*(AL*GHB+MU*GLB)
      GHEE(2,2)= DCOS(B+A)*(-0.5D0*SL**2-SW2*C2L) 
     .    + ML**2/MZ**2*GHB - ML*SL*CL/MZ**2*(AL*GHB+MU*GLB)
      GHEE(1,2)=2*DCOS(B+A)*SL*CL*(-SW2+0.25D0)
     .    + ML*C2L/2.D0/MZ**2*(AL*GHB+MU*GLB) 
      GHEE(2,1)=2*DCOS(B+A)*SL*CL*(-SW2+0.25D0)
     .    + ML*C2L/2.D0/MZ**2*(AL*GHB+MU*GLB) 

C
C PSEUDOSCALAR COUPLINGS 
C
      GATT=MT/2.D0/MZ**2*(MU+AT*GAT) 
      GABB=MB/2.D0/MZ**2*(MU+AB*GAB) 
      GAEE=ML/2.D0/MZ**2*(MU+AL*GAB) 
C
C CHARGED HIGGS COUPLINGS STOPS/SBOTTOMS 
C
      CLL=(MW**2*DSIN(2*B)-MT**2*GAT-MB**2*GAB)/DSQRT(2.D0)/MW**2
      CRR=-MT*MB*(GAT+GAB)/DSQRT(2.D0)/MW**2
      CLR=-MB*(MU+AB*GAB)/DSQRT(2.D0)/MW**2
      CRL=-MT*(MU+AT*GAT)/DSQRT(2.D0)/MW**2
      GCTB(1,1)=+CT*CB*CLL+ST*SB*CRR+CT*SB*CLR+ST*CB*CRL
      GCTB(1,2)=-CT*SB*CLL+ST*CB*CRR+CT*CB*CLR-ST*SB*CRL
      GCTB(2,1)=-ST*CB*CLL+CT*SB*CRR-ST*SB*CLR+CT*CB*CRL
      GCTB(2,2)=+ST*SB*CLL+CT*CB*CRR-ST*CB*CLR-CT*SB*CRL

C
C CHARGED HIGGS COUPLINGS TAU'S AND NEUTRINOS 
C
      CLL=(MW**2*DSIN(2*B)-ML**2*GAB)/DSQRT(2.D0)/MW**2
      CLR=-ML*(MU+AL*GAB)/DSQRT(2.D0)/MW**2
      GCEN(1,1)=CL*CLL+SL*CLR
      GCEN(1,2)=-SL*CLL+CL*CLR
      GCEN(2,1)=0.D0
      GCEN(2,2)=0.D0 

c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     write(6,*)'mst: ',MST(1),MST(2)
c     write(6,*)'msb: ',MSB(1),MSB(2)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      IF(ISLHAI.EQ.0)THEN
       CALL SFERMHO_HDEC(MQL,MUR,MDR,AT,AB,MU,TB,MW,MZ)
       ST = YSTHT
       CT = YCTHT
       SB = YSTHB
       CB = YCTHB
       GATT = YATT
       GABB = YABB
       DO I = 1,2
        MST(I) = YMST(I)
        MSB(I) = YMSB(I)
        DO J = 1,2
         GLTT(I,J) = YLTT(I,J)
         GHTT(I,J) = YHTT(I,J)
         GLBB(I,J) = YLBB(I,J)
         GHBB(I,J) = YHBB(I,J)
        ENDDO
       ENDDO
      ENDIF
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      YSTHT = ST
      YCTHT = CT
      YSTHB = SB
      YCTHB = CB
      YATT = GATT
      YABB = GABB
      DO I = 1,2
       YMST(I) = MST(I)
       YMSB(I) = MSB(I)
       DO J = 1,2
        YLTT(I,J) = GLTT(I,J)
        YHTT(I,J) = GHTT(I,J)
        YLBB(I,J) = GLBB(I,J)
        YHBB(I,J) = GHBB(I,J)
       ENDDO
      ENDDO
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     write(6,*)'stop_NLO: ',CT,ST

C--FILL COMMON BLOCK GLUINO_HDEC FOR SUSY-QCD CORRECTIONS TO
C  HIGGS -> BB, SQUARKS
      XMST1 = MST(1)
      XMST2 = MST(2)
      XMSB1 = MSB(1)
      XMSB2 = MSB(2)
      STHT = ST
      CTHT = CT
      STHB = SB
      CTHB = CB
      DO I=1,2
       DO J=1,2
        XLBB(I,J) = GLBB(I,J)
        XHBB(I,J) = GHBB(I,J)
        XABB(I,J) = 0
        XLTT(I,J) = GLTT(I,J)
        XHTT(I,J) = GHTT(I,J)
        XATT(I,J) = 0
       ENDDO
      ENDDO
      XABB(1,2) = GABB
      XABB(2,1) = -GABB
      XATT(1,2) = GATT
      XATT(2,1) = -GATT

c     write(6,*)'stop:    ',mst(1),mst(2),glt,st,ct,
c    .          gltt(1,1)*mz**2,gltt(1,2)*mz**2,
c    .          gltt(2,1)*mz**2,gltt(2,2)*mz**2
c     write(6,*)'sbottom: ',msb(1),msb(2),glb,sb,cb,
c    .          glbb(1,1)*mz**2,glbb(1,2)*mz**2,
c    .          glbb(2,1)*mz**2,glbb(2,2)*mz**2

c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     write(6,*)'mst: ',MST(1),MST(2)
c     write(6,*)'msb: ',MSB(1),MSB(2)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      goto 117
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

      QQS = (MSB(1)+MSB(2)+DABS(AMG))/3
      ALPSB = ALPHAS_HDEC(QQS,3)
      QQS = (MST(1)+MST(2)+DABS(AMG))/3
      ALPST = ALPHAS_HDEC(QQS,3)

      write(6,*)'light scalar:'
      write(6,*)'============:'
      write(6,*)
      QB = (msb(1)+msb(2))/2
      XX = SQSUSY_HDEC(1,1,1,1,QB,0,1)
      CALL SQMBAPP_HDEC(QB)
      write(6,*)'sbottom:'
      write(6,*)'--------'
      write(6,*)'M_H        =  ',aml,'D0'
      write(6,*)'M_Q        =  ',amb,'D0'
      write(6,*)'M_G        =  ',amgluino,'D0'
      write(6,*)'TG(BETA)   =  ',tb,'D0'
      write(6,*)'MU         =  ',mu,'D0'
      write(6,*)'FACTOR     =  ',-1+glt/glb,'D0'
      write(6,*)'QQ         =  ',QB,'D0'
      write(6,*)'M_Q(QQ)    =  ',smbot,'D0'
      write(6,*)'A_b(QQ)    =  ',AB00,'D0'
      write(6,*)'M_S1       =  ',msb(1),'D0'
      write(6,*)'M_S2       =  ',msb(2),'D0'
      write(6,*)'G_Q^H      =  ',glb,'D0'
      write(6,*)'ALPHAS(QS) =  ',ALPSB,'D0'
      write(6,*)'SIN(THETA) =  ',sb,'D0'
      write(6,*)'COS(THETA) =  ',cb,'D0'
      write(6,*)'GHQQ(1,1)  =  ',ylbb(1,1)*mz**2,'D0'
      write(6,*)'GHQQ(1,2)  =  ',ylbb(1,2)*mz**2,'D0'
      write(6,*)'GHQQ(2,1)  =  ',ylbb(2,1)*mz**2,'D0'
      write(6,*)'GHQQ(2,2)  =  ',ylbb(2,2)*mz**2,'D0'
      write(6,*)
      QT = (mst(1)+mst(2))/2
      XX = SQSUSY_HDEC(1,1,1,1,QT,0,1)
      CALL SQMBAPP_HDEC(QT)
      write(6,*)'stop:'
      write(6,*)'-----'
      write(6,*)'M_H        =  ',aml,'D0'
      write(6,*)'M_Q        =  ',amt,'D0'
      write(6,*)'M_G        =  ',amgluino,'D0'
      write(6,*)'TG(BETA)   =  ',tb,'D0'
      write(6,*)'MU         =  ',mu,'D0'
      write(6,*)'FACTOR     =  ',-1+glt/glb,'D0'
      write(6,*)'QQ         =  ',QT,'D0'
      write(6,*)'M_Q(QQ)    =  ',smtop,'D0'
      write(6,*)'A_t(QQ)    =  ',AT00,'D0'
      write(6,*)'M_S1       =  ',mst(1),'D0'
      write(6,*)'M_S2       =  ',mst(2),'D0'
      write(6,*)'G_Q^H      =  ',glt,'D0'
      write(6,*)'ALPHAS(QS) =  ',ALPST,'D0'
      write(6,*)'SIN(THETA) =  ',st,'D0'
      write(6,*)'COS(THETA) =  ',ct,'D0'
      write(6,*)'GHQQ(1,1)  =  ',yltt(1,1)*mz**2,'D0'
      write(6,*)'GHQQ(1,2)  =  ',yltt(1,2)*mz**2,'D0'
      write(6,*)'GHQQ(2,1)  =  ',yltt(2,1)*mz**2,'D0'
      write(6,*)'GHQQ(2,2)  =  ',yltt(2,2)*mz**2,'D0'
      write(6,*)
      write(6,*)'heavy scalar:'
      write(6,*)'============:'
      write(6,*)
      QB = (msb(1)+msb(2))/2
      XX = SQSUSY_HDEC(1,1,1,1,QB,0,1)
      CALL SQMBAPP_HDEC(QB)
      write(6,*)'sbottom:'
      write(6,*)'--------'
      write(6,*)'M_H        =  ',amh,'D0'
      write(6,*)'M_Q        =  ',amb,'D0'
      write(6,*)'M_G        =  ',amgluino,'D0'
      write(6,*)'TG(BETA)   =  ',tb,'D0'
      write(6,*)'MU         =  ',mu,'D0'
      write(6,*)'FACTOR     =  ',-1+ght/ghb,'D0'
      write(6,*)'QQ         =  ',QB,'D0'
      write(6,*)'M_Q(QQ)    =  ',smbot,'D0'
      write(6,*)'A_b(QQ)    =  ',AB00,'D0'
      write(6,*)'M_S1       =  ',msb(1),'D0'
      write(6,*)'M_S2       =  ',msb(2),'D0'
      write(6,*)'G_Q^H      =  ',ghb,'D0'
      write(6,*)'ALPHAS(QS) =  ',ALPSB,'D0'
      write(6,*)'SIN(THETA) =  ',sb,'D0'
      write(6,*)'COS(THETA) =  ',cb,'D0'
      write(6,*)'GHQQ(1,1)  =  ',yhbb(1,1)*mz**2,'D0'
      write(6,*)'GHQQ(1,2)  =  ',yhbb(1,2)*mz**2,'D0'
      write(6,*)'GHQQ(2,1)  =  ',yhbb(2,1)*mz**2,'D0'
      write(6,*)'GHQQ(2,2)  =  ',yhbb(2,2)*mz**2,'D0'
      write(6,*)
      QT = (mst(1)+mst(2))/2
      XX = SQSUSY_HDEC(1,1,1,1,QT,0,1)
      CALL SQMBAPP_HDEC(QT)
      write(6,*)'stop:'
      write(6,*)'-----'
      write(6,*)'M_H        =  ',amh,'D0'
      write(6,*)'M_Q        =  ',amt,'D0'
      write(6,*)'M_G        =  ',amgluino,'D0'
      write(6,*)'TG(BETA)   =  ',tb,'D0'
      write(6,*)'MU         =  ',mu,'D0'
      write(6,*)'FACTOR     =  ',-1+ght/ghb,'D0'
      write(6,*)'QQ         =  ',QT,'D0'
      write(6,*)'M_Q(QQ)    =  ',smtop,'D0'
      write(6,*)'A_t(QQ)    =  ',AT00,'D0'
      write(6,*)'M_S1       =  ',mst(1),'D0'
      write(6,*)'M_S2       =  ',mst(2),'D0'
      write(6,*)'G_Q^H      =  ',ght,'D0'
      write(6,*)'ALPHAS(QS) =  ',ALPST,'D0'
      write(6,*)'SIN(THETA) =  ',st,'D0'
      write(6,*)'COS(THETA) =  ',ct,'D0'
      write(6,*)'GHQQ(1,1)  =  ',yhtt(1,1)*mz**2,'D0'
      write(6,*)'GHQQ(1,2)  =  ',yhtt(1,2)*mz**2,'D0'
      write(6,*)'GHQQ(2,1)  =  ',yhtt(2,1)*mz**2,'D0'
      write(6,*)'GHQQ(2,2)  =  ',yhtt(2,2)*mz**2,'D0'
      write(6,*)

117   continue

c     write(6,*)'Stops:'
c     write(6,*)'======'
c     write(6,*)'COS(THETA) =  ',ct
c     write(6,*)'SIN(THETA) =  ',st
c     write(6,*)'M_ST1      =  ',mst(1)
c     write(6,*)'M_ST2      =  ',mst(2)
c     write(6,*)
c     write(6,*)'Sbottoms:'
c     write(6,*)'========='
c     write(6,*)'COS(THETA) =  ',cb
c     write(6,*)'SIN(THETA) =  ',sb
c     write(6,*)'M_SB1      =  ',msb(1)
c     write(6,*)'M_SB2      =  ',msb(2)
c     write(6,*)

      RETURN
111   STOP
      END 

C ******************************************************************

C      DOUBLE PRECISION FUNCTION RUNP_HDEC(Q,NF)
C      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C      COMMON/RUN_HDEC/XMSB,XMHAT,XKFAC
C      RUNP_HDEC = RUNM_HDEC(Q,NF)
C      RUNP_HDEC = RUNM_HDEC(Q/2.D0,NF)*XKFAC
C      RETURN
C      END

      DOUBLE PRECISION FUNCTION RUNM_HDEC(Q,NF0)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (NN=8)
      PARAMETER (ZETA3 = 1.202056903159594D0)
      DIMENSION AM(NN),YMSB(NN)
      COMMON/ALS_HDEC/XLAMBDA,AMCA,AMBA,AMTA,N0A
      COMMON/MASSES_HDEC/AMS,AMC,AMB,AMT
      COMMON/STRANGE_HDEC/AMSB
      COMMON/RUN_HDEC/XMSB,XMHAT,XKFAC
      COMMON/FLAG_HDEC/IHIGGS,NNLO,IPOLE
      COMMON/SM4_HDEC/AMTP,AMBP,AMNUP,AMEP,ISM4,IGGELW
      SAVE ISTRANGE
      B0(NF)=(33.D0-2.D0*NF)/12D0
      B1(NF) = (102D0-38D0/3D0*NF)/16D0
      B2(NF) = (2857D0/2D0-5033D0/18D0*NF+325D0/54D0*NF**2)/64D0
      G0(NF) = 1D0
      G1(NF) = (202D0/3D0-20D0/9D0*NF)/16D0
      G2(NF) = (1249D0-(2216D0/27D0+160D0/3D0*ZETA3)*NF
     .       - 140D0/81D0*NF**2)/64D0
      C1(NF) = G1(NF)/B0(NF) - B1(NF)*G0(NF)/B0(NF)**2
      C2(NF) = ((G1(NF)/B0(NF) - B1(NF)*G0(NF)/B0(NF)**2)**2
     .       + G2(NF)/B0(NF) + B1(NF)**2*G0(NF)/B0(NF)**3
     .       - B1(NF)*G1(NF)/B0(NF)**2 - B2(NF)*G0(NF)/B0(NF)**2)/2D0
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     C1(NF) = 1.175d0
c     C2(NF) = 1.501d0
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      TRAN(X,XK)=1D0+4D0/3D0*ALPHAS_HDEC(X,3)/PI
     .              +XK*(ALPHAS_HDEC(X,3)/PI)**2
      CQ(X,NF)=(2D0*B0(NF)*X)**(G0(NF)/B0(NF))
     .            *(1D0+LOOP2*C1(NF)*X+LOOP3*C2(NF)*X**2)
      DATA ISTRANGE/0/
      NF = NF0
      LOOP = 3
      LOOP2 = 1
      LOOP3 = 1
      IF(LOOP.LE.2)LOOP3 = 0
      IF(LOOP.LE.1)LOOP2 = 0
      PI=4D0*DATAN(1D0)
      ACC = 1.D-8
      AM(1) = 0
      AM(2) = 0
C--SCALE OF STRANGE MSBAR-MASS
c     QQS = 1.D0
      QQS = 2.D0
C--------------------------------------------
      IMSBAR = 0
      IF(IMSBAR.EQ.1)THEN
       IF(ISTRANGE.EQ.0)THEN
C--STRANGE POLE MASS FROM MSBAR-MASS AT QQS
        AMSD = XLAMBDA
        AMSU = 1.D8
123     AMS  = (AMSU+AMSD)/2
        AM(3) = AMS
        XMSB = AMS/CQ(ALPHAS_HDEC(AMS,3)/PI,3)
     .            *CQ(ALPHAS_HDEC(QQS,3)/PI,3)/TRAN(AMS,0D0)
        DD = (XMSB-AMSB)/AMSB
        IF(DABS(DD).GE.ACC)THEN
         IF(DD.LE.0.D0)THEN
          AMSD = AM(3)
         ELSE
          AMSU = AM(3)
         ENDIF
         GOTO 123
        ENDIF
        ISTRANGE=1
       ENDIF
       AM(3) = AMSB
      ELSE
       AMS=AMSB
       AM(3) = AMS
      ENDIF
C--------------------------------------------
      AM(3) = AMSB
      AM(4) = AMC
      AM(5) = AMB
      AM(6) = AMT
      AM(7) = 100*AMT
      AM(8) = 200*AMT
      IF(ISM4.NE.0)THEN
       IF(AMBP.LE.AMTP)THEN
        AM(7) = AMBP
        AM(8) = AMTP
       ELSE
        AM(7) = AMTP
        AM(8) = AMBP
        IF(NF0.EQ.7)THEN
         NF = 8
        ENDIF
        IF(NF0.EQ.8)THEN
         NF = 7
        ENDIF
       ENDIF
      ENDIF
      XK = 16.11D0
      DO 1 I=1,NF-1
       XK = XK - 1.04D0*(1.D0-AM(I)/AM(NF))
1     CONTINUE
      IF(NF.GE.4)THEN
       XMSB = AM(NF)/TRAN(AM(NF),0D0)
       XMHAT = XMSB/CQ(ALPHAS_HDEC(AM(NF),3)/PI,NF)
      ELSE
       XMSB = 0
       XMHAT = 0
      ENDIF
      YMSB(3) = AMSB
      IF(NF.EQ.3)THEN
       IF(QQS.LT.AMC)THEN
        YMSB(4) = YMSB(3)*CQ(ALPHAS_HDEC(AM(4),3)/PI,3)/
     .                    CQ(ALPHAS_HDEC(QQS,3)/PI,3)
       ELSE
        YMSB(4) = AMSB*CQ(ALPHAS_HDEC(AM(4),3)/PI,4)/
     .                    CQ(ALPHAS_HDEC(QQS,3)/PI,4)
        YMSB(3) = YMSB(4)
       ENDIF
       YMSB(5) = YMSB(4)*CQ(ALPHAS_HDEC(AM(5),3)/PI,4)/
     .                   CQ(ALPHAS_HDEC(AM(4),3)/PI,4)
       YMSB(6) = YMSB(5)*CQ(ALPHAS_HDEC(AM(6),3)/PI,5)/
     .                   CQ(ALPHAS_HDEC(AM(5),3)/PI,5)
       YMSB(7) = YMSB(6)*CQ(ALPHAS_HDEC(AM(7),3)/PI,6)/
     .                   CQ(ALPHAS_HDEC(AM(6),3)/PI,6)
       YMSB(8) = YMSB(7)*CQ(ALPHAS_HDEC(AM(8),3)/PI,7)/
     .                   CQ(ALPHAS_HDEC(AM(7),3)/PI,7)
      ELSEIF(NF.EQ.4)THEN
       YMSB(4) = XMSB
       YMSB(3) = YMSB(4)*CQ(ALPHAS_HDEC(QQS,3)/PI,3)/
     .                   CQ(ALPHAS_HDEC(AM(4),3)/PI,3)
       YMSB(5) = YMSB(4)*CQ(ALPHAS_HDEC(AM(5),3)/PI,4)/
     .                   CQ(ALPHAS_HDEC(AM(4),3)/PI,4)
       YMSB(6) = YMSB(5)*CQ(ALPHAS_HDEC(AM(6),3)/PI,5)/
     .                   CQ(ALPHAS_HDEC(AM(5),3)/PI,5)
       YMSB(7) = YMSB(6)*CQ(ALPHAS_HDEC(AM(7),3)/PI,6)/
     .                   CQ(ALPHAS_HDEC(AM(6),3)/PI,6)
       YMSB(8) = YMSB(7)*CQ(ALPHAS_HDEC(AM(8),3)/PI,7)/
     .                   CQ(ALPHAS_HDEC(AM(7),3)/PI,7)
      ELSEIF(NF.EQ.5)THEN
       YMSB(5) = XMSB
       YMSB(4) = YMSB(5)*CQ(ALPHAS_HDEC(AM(4),3)/PI,4)/
     .                   CQ(ALPHAS_HDEC(AM(5),3)/PI,4)
       YMSB(3) = YMSB(4)*CQ(ALPHAS_HDEC(QQS,3)/PI,3)/
     .                   CQ(ALPHAS_HDEC(AM(4),3)/PI,3)
       YMSB(6) = YMSB(5)*CQ(ALPHAS_HDEC(AM(6),3)/PI,5)/
     .                   CQ(ALPHAS_HDEC(AM(5),3)/PI,5)
       YMSB(7) = YMSB(6)*CQ(ALPHAS_HDEC(AM(7),3)/PI,6)/
     .                   CQ(ALPHAS_HDEC(AM(6),3)/PI,6)
       YMSB(8) = YMSB(7)*CQ(ALPHAS_HDEC(AM(8),3)/PI,7)/
     .                   CQ(ALPHAS_HDEC(AM(7),3)/PI,7)
      ELSEIF(NF.EQ.6)THEN
       YMSB(6) = XMSB
       YMSB(5) = YMSB(6)*CQ(ALPHAS_HDEC(AM(5),3)/PI,5)/
     .                   CQ(ALPHAS_HDEC(AM(6),3)/PI,5)
       YMSB(4) = YMSB(5)*CQ(ALPHAS_HDEC(AM(4),3)/PI,4)/
     .                   CQ(ALPHAS_HDEC(AM(5),3)/PI,4)
       YMSB(3) = YMSB(4)*CQ(ALPHAS_HDEC(QQS,3)/PI,3)/
     .                   CQ(ALPHAS_HDEC(AM(4),3)/PI,3)
       YMSB(7) = YMSB(6)*CQ(ALPHAS_HDEC(AM(7),3)/PI,6)/
     .                   CQ(ALPHAS_HDEC(AM(6),3)/PI,6)
       YMSB(8) = YMSB(7)*CQ(ALPHAS_HDEC(AM(8),3)/PI,7)/
     .                   CQ(ALPHAS_HDEC(AM(7),3)/PI,7)
      ELSEIF(NF.EQ.7)THEN
       YMSB(7) = XMSB
       YMSB(6) = YMSB(7)*CQ(ALPHAS_HDEC(AM(6),3)/PI,6)/
     .                   CQ(ALPHAS_HDEC(AM(7),3)/PI,6)
       YMSB(5) = YMSB(6)*CQ(ALPHAS_HDEC(AM(5),3)/PI,5)/
     .                   CQ(ALPHAS_HDEC(AM(6),3)/PI,5)
       YMSB(4) = YMSB(5)*CQ(ALPHAS_HDEC(AM(4),3)/PI,4)/
     .                   CQ(ALPHAS_HDEC(AM(5),3)/PI,4)
       YMSB(3) = YMSB(4)*CQ(ALPHAS_HDEC(QQS,3)/PI,3)/
     .                   CQ(ALPHAS_HDEC(AM(4),3)/PI,3)
       YMSB(8) = YMSB(7)*CQ(ALPHAS_HDEC(AM(8),3)/PI,7)/
     .                   CQ(ALPHAS_HDEC(AM(7),3)/PI,7)
      ELSEIF(NF.EQ.8)THEN
       YMSB(8) = XMSB
       YMSB(7) = YMSB(8)*CQ(ALPHAS_HDEC(AM(7),3)/PI,7)/
     .                   CQ(ALPHAS_HDEC(AM(8),3)/PI,7)
       YMSB(6) = YMSB(7)*CQ(ALPHAS_HDEC(AM(6),3)/PI,6)/
     .                   CQ(ALPHAS_HDEC(AM(7),3)/PI,6)
       YMSB(5) = YMSB(6)*CQ(ALPHAS_HDEC(AM(5),3)/PI,5)/
     .                   CQ(ALPHAS_HDEC(AM(6),3)/PI,5)
       YMSB(4) = YMSB(5)*CQ(ALPHAS_HDEC(AM(4),3)/PI,4)/
     .                   CQ(ALPHAS_HDEC(AM(5),3)/PI,4)
       YMSB(3) = YMSB(4)*CQ(ALPHAS_HDEC(QQS,3)/PI,3)/
     .                   CQ(ALPHAS_HDEC(AM(4),3)/PI,3)
      ENDIF
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     IF(Q.LT.AMC)THEN
c      N0=3
c      IF(QQS.LT.AMC)THEN
c       Q0 = QQS
c      ELSE
c       Q0 = AMC
c      ENDIF
c     ELSEIF(Q.LE.AMB)THEN
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     IF(Q.LT.AMB)THEN
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      IF(Q.LT.AMC)THEN
       N0=3
       Q0 = QQS
      ELSEIF(Q.LE.AMB)THEN
       N0=4
       Q0 = AMC
      ELSEIF(Q.LE.AMT)THEN
       N0=5
       Q0 = AMB
      ELSE
       N0=6
       Q0 = AMT
       IF(ISM4.NE.0)THEN
        IF(Q.GT.AM(7))THEN
         IF(Q.LE.AM(8))THEN
          N0=7
          Q0 = AM(7)
         ELSE
          N0=8
          Q0 = AM(8)
         ENDIF
        ENDIF
       ENDIF
      ENDIF
      IF(NNLO.EQ.1.AND.NF.GT.3)THEN
       XKFAC = TRAN(AM(NF),0D0)/TRAN(AM(NF),XK)
      ELSE
       XKFAC = 1D0
      ENDIF
      RUNM_HDEC = YMSB(N0)*CQ(ALPHAS_HDEC(Q,3)/PI,N0)/
     .               CQ(ALPHAS_HDEC(Q0,3)/PI,N0)
     .       * XKFAC
      RETURN
      END

      DOUBLE PRECISION FUNCTION RUNM0_HDEC(Q,NF0)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (NN=8)
      PARAMETER (ZETA3 = 1.202056903159594D0)
      DIMENSION AM(NN),YMSB(NN)
      COMMON/ALS_HDEC/XLAMBDA,AMCA,AMBA,AMTA,N0A
      COMMON/MASSES_HDEC/AMS,AMC,AMB,AMT
      COMMON/STRANGE_HDEC/AMSB
      COMMON/RUN_HDEC/XMSB,XMHAT,XKFAC
      COMMON/FLAG_HDEC/IHIGGS,NNLO,IPOLE
      COMMON/SM4_HDEC/AMTP,AMBP,AMNUP,AMEP,ISM4,IGGELW
      SAVE ISTRANGE
      B0(NF)=(33.D0-2.D0*NF)/12D0
      B1(NF) = (102D0-38D0/3D0*NF)/16D0
      B2(NF) = (2857D0/2D0-5033D0/18D0*NF+325D0/54D0*NF**2)/64D0
      G0(NF) = 1D0
      G1(NF) = (202D0/3D0-20D0/9D0*NF)/16D0
      G2(NF) = (1249D0-(2216D0/27D0+160D0/3D0*ZETA3)*NF
     .       - 140D0/81D0*NF**2)/64D0
      C1(NF) = G1(NF)/B0(NF) - B1(NF)*G0(NF)/B0(NF)**2
      C2(NF) = ((G1(NF)/B0(NF) - B1(NF)*G0(NF)/B0(NF)**2)**2
     .       + G2(NF)/B0(NF) + B1(NF)**2*G0(NF)/B0(NF)**3
     .       - B1(NF)*G1(NF)/B0(NF)**2 - B2(NF)*G0(NF)/B0(NF)**2)/2D0
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     C1(NF) = 1.175d0
c     C2(NF) = 1.501d0
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      TRAN(X,XK)=1D0+4D0/3D0*ALPHAS_HDEC(X,2)/PI
     .              +XK*(ALPHAS_HDEC(X,2)/PI)**2
      CQ(X,NF)=(2D0*B0(NF)*X)**(G0(NF)/B0(NF))
     .            *(1D0+LOOP2*C1(NF)*X+LOOP3*C2(NF)*X**2)
      DATA ISTRANGE/0/
      NF = NF0
      LOOP = 3
      LOOP2 = 1
      LOOP3 = 1
      IF(LOOP.LE.2)LOOP3 = 0
      IF(LOOP.LE.1)LOOP2 = 0
      PI=4D0*DATAN(1D0)
      ACC = 1.D-8
      AM(1) = 0
      AM(2) = 0
C--SCALE OF STRANGE MSBAR-MASS
c     QQS = 1.D0
      QQS = 2.D0
C--------------------------------------------
      IMSBAR = 0
      IF(IMSBAR.EQ.1)THEN
       IF(ISTRANGE.EQ.0)THEN
C--STRANGE POLE MASS FROM MSBAR-MASS AT QQS
        AMSD = XLAMBDA
        AMSU = 1.D8
123     AMS  = (AMSU+AMSD)/2
        AM(3) = AMS
        XMSB = AMS/CQ(ALPHAS_HDEC(AMS,2)/PI,3)
     .            *CQ(ALPHAS_HDEC(QQS,2)/PI,3)/TRAN(AMS,0D0)
        DD = (XMSB-AMSB)/AMSB
        IF(DABS(DD).GE.ACC)THEN
         IF(DD.LE.0.D0)THEN
          AMSD = AM(3)
         ELSE
          AMSU = AM(3)
         ENDIF
         GOTO 123
        ENDIF
        ISTRANGE=1
       ENDIF
       AM(3) = AMSB
      ELSE
       AMS=AMSB
       AM(3) = AMS
      ENDIF
C--------------------------------------------
      AM(3) = AMSB
      AM(4) = AMC
      AM(5) = AMB
      AM(6) = AMT
      AM(7) = 100*AMT
      AM(8) = 200*AMT
      IF(ISM4.NE.0)THEN
       IF(AMBP.LE.AMTP)THEN
        AM(7) = AMBP
        AM(8) = AMTP
       ELSE
        AM(7) = AMTP
        AM(8) = AMBP
        IF(NF0.EQ.7)THEN
         NF = 8
        ELSE
         NF = 7
        ENDIF
       ENDIF
      ENDIF
      XK = 16.11D0
      DO 1 I=1,NF-1
       XK = XK - 1.04D0*(1.D0-AM(I)/AM(NF))
1     CONTINUE
      IF(NF.GE.4)THEN
       XMSB = AM(NF)/TRAN(AM(NF),0D0)
       XMHAT = XMSB/CQ(ALPHAS_HDEC(AM(NF),2)/PI,NF)
      ELSE
       XMSB = 0
       XMHAT = 0
      ENDIF
      YMSB(3) = AMSB
      IF(NF.EQ.3)THEN
       IF(QQS.LT.AMC)THEN
        YMSB(4) = YMSB(3)*CQ(ALPHAS_HDEC(AM(4),2)/PI,3)/
     .                    CQ(ALPHAS_HDEC(QQS,2)/PI,3)
       ELSE
        YMSB(4) = AMSB*CQ(ALPHAS_HDEC(AM(4),2)/PI,4)/
     .                    CQ(ALPHAS_HDEC(QQS,2)/PI,4)
        YMSB(3) = YMSB(4)
       ENDIF
       YMSB(5) = YMSB(4)*CQ(ALPHAS_HDEC(AM(5),2)/PI,4)/
     .                   CQ(ALPHAS_HDEC(AM(4),2)/PI,4)
       YMSB(6) = YMSB(5)*CQ(ALPHAS_HDEC(AM(6),2)/PI,5)/
     .                   CQ(ALPHAS_HDEC(AM(5),2)/PI,5)
       YMSB(7) = YMSB(6)*CQ(ALPHAS_HDEC(AM(7),2)/PI,6)/
     .                   CQ(ALPHAS_HDEC(AM(6),2)/PI,6)
       YMSB(8) = YMSB(7)*CQ(ALPHAS_HDEC(AM(8),2)/PI,7)/
     .                   CQ(ALPHAS_HDEC(AM(7),2)/PI,7)
      ELSEIF(NF.EQ.4)THEN
       YMSB(4) = XMSB
       YMSB(3) = YMSB(4)*CQ(ALPHAS_HDEC(QQS,2)/PI,3)/
     .                   CQ(ALPHAS_HDEC(AM(4),2)/PI,3)
       YMSB(5) = YMSB(4)*CQ(ALPHAS_HDEC(AM(5),2)/PI,4)/
     .                   CQ(ALPHAS_HDEC(AM(4),2)/PI,4)
       YMSB(6) = YMSB(5)*CQ(ALPHAS_HDEC(AM(6),2)/PI,5)/
     .                   CQ(ALPHAS_HDEC(AM(5),2)/PI,5)
       YMSB(7) = YMSB(6)*CQ(ALPHAS_HDEC(AM(7),2)/PI,6)/
     .                   CQ(ALPHAS_HDEC(AM(6),2)/PI,6)
       YMSB(8) = YMSB(7)*CQ(ALPHAS_HDEC(AM(8),2)/PI,7)/
     .                   CQ(ALPHAS_HDEC(AM(7),2)/PI,7)
      ELSEIF(NF.EQ.5)THEN
       YMSB(5) = XMSB
       YMSB(4) = YMSB(5)*CQ(ALPHAS_HDEC(AM(4),2)/PI,4)/
     .                   CQ(ALPHAS_HDEC(AM(5),2)/PI,4)
       YMSB(3) = YMSB(4)*CQ(ALPHAS_HDEC(QQS,2)/PI,3)/
     .                   CQ(ALPHAS_HDEC(AM(4),2)/PI,3)
       YMSB(6) = YMSB(5)*CQ(ALPHAS_HDEC(AM(6),2)/PI,5)/
     .                   CQ(ALPHAS_HDEC(AM(5),2)/PI,5)
       YMSB(7) = YMSB(6)*CQ(ALPHAS_HDEC(AM(7),2)/PI,6)/
     .                   CQ(ALPHAS_HDEC(AM(6),2)/PI,6)
       YMSB(8) = YMSB(7)*CQ(ALPHAS_HDEC(AM(8),2)/PI,7)/
     .                   CQ(ALPHAS_HDEC(AM(7),2)/PI,7)
      ELSEIF(NF.EQ.6)THEN
       YMSB(6) = XMSB
       YMSB(5) = YMSB(6)*CQ(ALPHAS_HDEC(AM(5),2)/PI,5)/
     .                   CQ(ALPHAS_HDEC(AM(6),2)/PI,5)
       YMSB(4) = YMSB(5)*CQ(ALPHAS_HDEC(AM(4),2)/PI,4)/
     .                   CQ(ALPHAS_HDEC(AM(5),2)/PI,4)
       YMSB(3) = YMSB(4)*CQ(ALPHAS_HDEC(QQS,2)/PI,3)/
     .                   CQ(ALPHAS_HDEC(AM(4),2)/PI,3)
       YMSB(7) = YMSB(6)*CQ(ALPHAS_HDEC(AM(7),2)/PI,6)/
     .                   CQ(ALPHAS_HDEC(AM(6),2)/PI,6)
       YMSB(8) = YMSB(7)*CQ(ALPHAS_HDEC(AM(8),2)/PI,7)/
     .                   CQ(ALPHAS_HDEC(AM(7),2)/PI,7)
      ELSEIF(NF.EQ.7)THEN
       YMSB(7) = XMSB
       YMSB(6) = YMSB(7)*CQ(ALPHAS_HDEC(AM(6),2)/PI,6)/
     .                   CQ(ALPHAS_HDEC(AM(7),2)/PI,6)
       YMSB(5) = YMSB(6)*CQ(ALPHAS_HDEC(AM(5),2)/PI,5)/
     .                   CQ(ALPHAS_HDEC(AM(6),2)/PI,5)
       YMSB(4) = YMSB(5)*CQ(ALPHAS_HDEC(AM(4),2)/PI,4)/
     .                   CQ(ALPHAS_HDEC(AM(5),2)/PI,4)
       YMSB(3) = YMSB(4)*CQ(ALPHAS_HDEC(QQS,2)/PI,3)/
     .                   CQ(ALPHAS_HDEC(AM(4),2)/PI,3)
       YMSB(8) = YMSB(7)*CQ(ALPHAS_HDEC(AM(8),2)/PI,7)/
     .                   CQ(ALPHAS_HDEC(AM(7),2)/PI,7)
      ELSEIF(NF.EQ.8)THEN
       YMSB(8) = XMSB
       YMSB(7) = YMSB(8)*CQ(ALPHAS_HDEC(AM(7),2)/PI,7)/
     .                   CQ(ALPHAS_HDEC(AM(8),2)/PI,7)
       YMSB(6) = YMSB(7)*CQ(ALPHAS_HDEC(AM(6),2)/PI,6)/
     .                   CQ(ALPHAS_HDEC(AM(7),2)/PI,6)
       YMSB(5) = YMSB(6)*CQ(ALPHAS_HDEC(AM(5),2)/PI,5)/
     .                   CQ(ALPHAS_HDEC(AM(6),2)/PI,5)
       YMSB(4) = YMSB(5)*CQ(ALPHAS_HDEC(AM(4),2)/PI,4)/
     .                   CQ(ALPHAS_HDEC(AM(5),2)/PI,4)
       YMSB(3) = YMSB(4)*CQ(ALPHAS_HDEC(QQS,2)/PI,3)/
     .                   CQ(ALPHAS_HDEC(AM(4),2)/PI,3)
      ENDIF
      IF(Q.LT.AMC)THEN
       N0=3
c      IF(QQS.LT.AMC)THEN
        Q0 = QQS
c      ELSE
c       Q0 = AMC
c      ENDIF
      ELSEIF(Q.LE.AMB)THEN
       N0=4
       Q0 = AMC
      ELSEIF(Q.LE.AMT)THEN
       N0=5
       Q0 = AMB
      ELSE
       N0=6
       Q0 = AMT
       IF(ISM4.NE.0)THEN
        IF(Q.GT.AM(7))THEN
         IF(Q.LE.AM(8))THEN
          N0=7
          Q0 = AM(7)
         ELSE
          N0=8
          Q0 = AM(8)
         ENDIF
        ENDIF
       ENDIF
      ENDIF
      IF(NNLO.EQ.1.AND.NF.GT.3)THEN
       XKFAC = TRAN(AM(NF),0D0)/TRAN(AM(NF),XK)
      ELSE
       XKFAC = 1D0
      ENDIF
      RUNM0_HDEC = YMSB(N0)*CQ(ALPHAS_HDEC(Q,2)/PI,N0)/
     .               CQ(ALPHAS_HDEC(Q0,2)/PI,N0)
     .       * XKFAC
      RETURN
      END

      DOUBLE PRECISION FUNCTION ALPHAS_HDEC(Q,N)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION XLB(6)
      COMMON/ALSLAM_HDEC/XLB1(6),XLB2(6),XLB3(6)
      COMMON/ALS_HDEC/XLAMBDA,AMC,AMB,AMT,N0
      B0(NF)=33.D0-2.D0*NF
      B1(NF)=6.D0*(153.D0-19.D0*NF)/B0(NF)**2
      B2(NF)=27/2.D0*(2857-5033/9.D0*NF+325/27.D0*NF**2)/B0(NF)**3
      ALS1(NF,X)=12.D0*PI/(B0(NF)*DLOG(X**2/XLB(NF)**2))
      ALS2(NF,X)=12.D0*PI/(B0(NF)*DLOG(X**2/XLB(NF)**2))
     .          *(1.D0-B1(NF)*DLOG(DLOG(X**2/XLB(NF)**2))
     .           /DLOG(X**2/XLB(NF)**2))
      ALS3(NF,X)=12.D0*PI/(B0(NF)*DLOG(X**2/XLB(NF)**2))
     .          *(1.D0-B1(NF)*DLOG(DLOG(X**2/XLB(NF)**2))
     .           /DLOG(X**2/XLB(NF)**2)
     .           +(B1(NF)**2*(DLOG(DLOG(X**2/XLB(NF)**2))**2
     .                      -DLOG(DLOG(X**2/XLB(NF)**2))-1)+B2(NF))
     .           /DLOG(X**2/XLB(NF)**2)**2)
      PI=4.D0*DATAN(1.D0)
c     write(6,*)'ALS param: ',XLAMBDA,AMC,AMB,AMT,N0
      IF(N.EQ.1)THEN
       DO 1 I=1,6
        XLB(I)=XLB1(I)
1      CONTINUE
      ELSEIF(N.EQ.2)THEN
       DO 2 I=1,6
        XLB(I)=XLB2(I)
2      CONTINUE
      ELSE
       DO 3 I=1,6
        XLB(I)=XLB3(I)
3      CONTINUE
      ENDIF
      IF(Q.LT.AMC)THEN
       NF=3
      ELSEIF(Q.LE.AMB)THEN
       NF=4
      ELSEIF(Q.LE.AMT)THEN
       NF=5
      ELSE
       NF=6
      ENDIF
      IF(N.EQ.1)THEN
        ALPHAS_HDEC=ALS1(NF,Q)
      ELSEIF(N.EQ.2)THEN
        ALPHAS_HDEC=ALS2(NF,Q)
      ELSE
        ALPHAS_HDEC=ALS3(NF,Q)
c       ALPHAS_HDEC=ALS2(NF,Q)
      ENDIF
      RETURN
      END

      SUBROUTINE ALSINI_HDEC(ACC)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION XLB(6)
      COMMON/ALSLAM_HDEC/XLB1(6),XLB2(6),XLB3(6)
      COMMON/ALS_HDEC/XLAMBDA,AMC,AMB,AMT,N0
      PI=4.D0*DATAN(1.D0)
      XLB1(1)=0D0
      XLB1(2)=0D0
      XLB2(1)=0D0
      XLB2(2)=0D0
      IF(N0.EQ.3)THEN
       XLB(3)=XLAMBDA
       XLB(4)=XLB(3)*(XLB(3)/AMC)**(2.D0/25.D0)
       XLB(5)=XLB(4)*(XLB(4)/AMB)**(2.D0/23.D0)
       XLB(6)=XLB(5)*(XLB(5)/AMT)**(2.D0/21.D0)
      ELSEIF(N0.EQ.4)THEN
       XLB(4)=XLAMBDA
       XLB(5)=XLB(4)*(XLB(4)/AMB)**(2.D0/23.D0)
       XLB(3)=XLB(4)*(XLB(4)/AMC)**(-2.D0/27.D0)
       XLB(6)=XLB(5)*(XLB(5)/AMT)**(2.D0/21.D0)
      ELSEIF(N0.EQ.5)THEN
       XLB(5)=XLAMBDA
       XLB(4)=XLB(5)*(XLB(5)/AMB)**(-2.D0/25.D0)
       XLB(3)=XLB(4)*(XLB(4)/AMC)**(-2.D0/27.D0)
       XLB(6)=XLB(5)*(XLB(5)/AMT)**(2.D0/21.D0)
      ELSEIF(N0.EQ.6)THEN
       XLB(6)=XLAMBDA
       XLB(5)=XLB(6)*(XLB(6)/AMT)**(-2.D0/23.D0)
       XLB(4)=XLB(5)*(XLB(5)/AMB)**(-2.D0/25.D0)
       XLB(3)=XLB(4)*(XLB(4)/AMC)**(-2.D0/27.D0)
      ENDIF
      DO 1 I=3,6
       XLB1(I)=XLB(I)
1     CONTINUE
      IF(N0.EQ.3)THEN
       XLB(3)=XLAMBDA
       XLB(4)=XLB(3)*(XLB(3)/AMC)**(2.D0/25.D0)
     .             *(2.D0*DLOG(AMC/XLB(3)))**(-107.D0/1875.D0)
       XLB(4)=XITER_HDEC(AMC,XLB(3),3,XLB(4),4,ACC)
       XLB(5)=XLB(4)*(XLB(4)/AMB)**(2.D0/23.D0)
     .             *(2.D0*DLOG(AMB/XLB(4)))**(-963.D0/13225.D0)
       XLB(5)=XITER_HDEC(AMB,XLB(4),4,XLB(5),5,ACC)
       XLB(6)=XLB(5)*(XLB(5)/AMT)**(2.D0/21.D0)
     .            *(2.D0*DLOG(AMT/XLB(5)))**(-321.D0/3381.D0)
       XLB(6)=XITER_HDEC(AMT,XLB(5),5,XLB(6),6,ACC)
      ELSEIF(N0.EQ.4)THEN
       XLB(4)=XLAMBDA
       XLB(5)=XLB(4)*(XLB(4)/AMB)**(2.D0/23.D0)
     .             *(2.D0*DLOG(AMB/XLB(4)))**(-963.D0/13225.D0)
       XLB(5)=XITER_HDEC(AMB,XLB(4),4,XLB(5),5,ACC)
       XLB(3)=XLB(4)*(XLB(4)/AMC)**(-2.D0/27.D0)
     .             *(2.D0*DLOG(AMC/XLB(4)))**(107.D0/2025.D0)
       XLB(3)=XITER_HDEC(AMC,XLB(4),4,XLB(3),3,ACC)
       XLB(6)=XLB(5)*(XLB(5)/AMT)**(2.D0/21.D0)
     .            *(2.D0*DLOG(AMT/XLB(5)))**(-321.D0/3381.D0)
       XLB(6)=XITER_HDEC(AMT,XLB(5),5,XLB(6),6,ACC)
      ELSEIF(N0.EQ.5)THEN
       XLB(5)=XLAMBDA
       XLB(4)=XLB(5)*(XLB(5)/AMB)**(-2.D0/25.D0)
     .             *(2.D0*DLOG(AMB/XLB(5)))**(963.D0/14375.D0)
       XLB(4)=XITER_HDEC(AMB,XLB(5),5,XLB(4),4,ACC)
       XLB(3)=XLB(4)*(XLB(4)/AMC)**(-2.D0/27.D0)
     .             *(2.D0*DLOG(AMC/XLB(4)))**(107.D0/2025.D0)
       XLB(3)=XITER_HDEC(AMC,XLB(4),4,XLB(3),3,ACC)
       XLB(6)=XLB(5)*(XLB(5)/AMT)**(2.D0/21.D0)
     .            *(2.D0*DLOG(AMT/XLB(5)))**(-321.D0/3381.D0)
       XLB(6)=XITER_HDEC(AMT,XLB(5),5,XLB(6),6,ACC)
      ELSEIF(N0.EQ.6)THEN
       XLB(6)=XLAMBDA
       XLB(5)=XLB(6)*(XLB(6)/AMT)**(-2.D0/23.D0)
     .            *(2.D0*DLOG(AMT/XLB(6)))**(321.D0/3703.D0)
       XLB(5)=XITER_HDEC(AMT,XLB(6),6,XLB(5),5,ACC)
       XLB(4)=XLB(5)*(XLB(5)/AMB)**(-2.D0/25.D0)
     .             *(2.D0*DLOG(AMB/XLB(5)))**(963.D0/14375.D0)
       XLB(4)=XITER_HDEC(AMB,XLB(5),5,XLB(4),4,ACC)
       XLB(3)=XLB(4)*(XLB(4)/AMC)**(-2.D0/27.D0)
     .             *(2.D0*DLOG(AMC/XLB(4)))**(107.D0/2025.D0)
       XLB(3)=XITER_HDEC(AMC,XLB(4),4,XLB(3),3,ACC)
      ENDIF
      DO 2 I=3,6
       XLB2(I)=XLB(I)
2     CONTINUE
      IF(N0.EQ.3)THEN
       XLB(3)=XLAMBDA
       XLB(4)=XLB(3)*(XLB(3)/AMC)**(2.D0/25.D0)
     .             *(2.D0*DLOG(AMC/XLB(3)))**(-107.D0/1875.D0)
       XLB(4)=XITER3_HDEC(AMC,XLB(3),3,XLB(4),4,ACC)
       XLB(5)=XLB(4)*(XLB(4)/AMB)**(2.D0/23.D0)
     .             *(2.D0*DLOG(AMB/XLB(4)))**(-963.D0/13225.D0)
       XLB(5)=XITER3_HDEC(AMB,XLB(4),4,XLB(5),5,ACC)
       XLB(6)=XLB(5)*(XLB(5)/AMT)**(2.D0/21.D0)
     .            *(2.D0*DLOG(AMT/XLB(5)))**(-321.D0/3381.D0)
       XLB(6)=XITER3_HDEC(AMT,XLB(5),5,XLB(6),6,ACC)
      ELSEIF(N0.EQ.4)THEN
       XLB(4)=XLAMBDA
       XLB(5)=XLB(4)*(XLB(4)/AMB)**(2.D0/23.D0)
     .             *(2.D0*DLOG(AMB/XLB(4)))**(-963.D0/13225.D0)
       XLB(5)=XITER3_HDEC(AMB,XLB(4),4,XLB(5),5,ACC)
       XLB(3)=XLB(4)*(XLB(4)/AMC)**(-2.D0/27.D0)
     .             *(2.D0*DLOG(AMC/XLB(4)))**(107.D0/2025.D0)
       XLB(3)=XITER3_HDEC(AMC,XLB(4),4,XLB(3),3,ACC)
       XLB(6)=XLB(5)*(XLB(5)/AMT)**(2.D0/21.D0)
     .            *(2.D0*DLOG(AMT/XLB(5)))**(-321.D0/3381.D0)
       XLB(6)=XITER3_HDEC(AMT,XLB(5),5,XLB(6),6,ACC)
      ELSEIF(N0.EQ.5)THEN
       XLB(5)=XLAMBDA
       XLB(4)=XLB(5)*(XLB(5)/AMB)**(-2.D0/25.D0)
     .             *(2.D0*DLOG(AMB/XLB(5)))**(963.D0/14375.D0)
       XLB(4)=XITER3_HDEC(AMB,XLB(5),5,XLB(4),4,ACC)
       XLB(3)=XLB(4)*(XLB(4)/AMC)**(-2.D0/27.D0)
     .             *(2.D0*DLOG(AMC/XLB(4)))**(107.D0/2025.D0)
       XLB(3)=XITER3_HDEC(AMC,XLB(4),4,XLB(3),3,ACC)
       XLB(6)=XLB(5)*(XLB(5)/AMT)**(2.D0/21.D0)
     .            *(2.D0*DLOG(AMT/XLB(5)))**(-321.D0/3381.D0)
       XLB(6)=XITER3_HDEC(AMT,XLB(5),5,XLB(6),6,ACC)
      ELSEIF(N0.EQ.6)THEN
       XLB(6)=XLAMBDA
       XLB(5)=XLB(6)*(XLB(6)/AMT)**(-2.D0/23.D0)
     .            *(2.D0*DLOG(AMT/XLB(6)))**(321.D0/3703.D0)
       XLB(5)=XITER3_HDEC(AMT,XLB(6),6,XLB(5),5,ACC)
       XLB(4)=XLB(5)*(XLB(5)/AMB)**(-2.D0/25.D0)
     .             *(2.D0*DLOG(AMB/XLB(5)))**(963.D0/14375.D0)
       XLB(4)=XITER3_HDEC(AMB,XLB(5),5,XLB(4),4,ACC)
       XLB(3)=XLB(4)*(XLB(4)/AMC)**(-2.D0/27.D0)
     .             *(2.D0*DLOG(AMC/XLB(4)))**(107.D0/2025.D0)
       XLB(3)=XITER3_HDEC(AMC,XLB(4),4,XLB(3),3,ACC)
      ENDIF
      DO 3 I=3,6
       XLB3(I)=XLB(I)
3     CONTINUE
      RETURN
      END

      DOUBLE PRECISION FUNCTION XITER_HDEC(Q,XLB1,NF1,XLB,NF2,ACC)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      B0(NF)=33.D0-2.D0*NF
      B1(NF)=6.D0*(153.D0-19.D0*NF)/B0(NF)**2
      ALS2(NF,X,XLB)=12.D0*PI/(B0(NF)*DLOG(X**2/XLB**2))
     .              *(1.D0-B1(NF)*DLOG(DLOG(X**2/XLB**2))
     .              /DLOG(X**2/XLB**2))
      AA(NF)=12D0*PI/B0(NF)
      BB(NF)=B1(NF)/AA(NF)
      XIT(A,B,X)=A/2.D0*(1D0+DSQRT(1D0-4D0*B*DLOG(X)))
      PI=4.D0*DATAN(1.D0)
      XLB2=XLB
      II=0
1     II=II+1
      X=DLOG(Q**2/XLB2**2)
      ALP=ALS2(NF1,Q,XLB1)
      A=AA(NF2)/ALP
      B=BB(NF2)*ALP
      XX=XIT(A,B,X)
      XLB2=Q*DEXP(-XX/2.D0)
      Y1=ALS2(NF1,Q,XLB1)
      Y2=ALS2(NF2,Q,XLB2)
      DY=DABS(Y2-Y1)/Y1
      IF(DY.GE.ACC) GOTO 1
       XITER_HDEC=XLB2
      RETURN
      END

      DOUBLE PRECISION FUNCTION XITER3_HDEC(Q,XLB1,NF1,XLB,NF2,ACC)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      B0(NF)=33.D0-2.D0*NF
      B1(NF)=6.D0*(153.D0-19.D0*NF)/B0(NF)**2
      B2(NF)=27/2.D0*(2857-5033/9.D0*NF+325/27.D0*NF**2)/B0(NF)**3
      ALS3(NF,X,XLB)=12.D0*PI/(B0(NF)*DLOG(X**2/XLB**2))
     .          *(1.D0-B1(NF)*DLOG(DLOG(X**2/XLB**2))
     .           /DLOG(X**2/XLB**2)
     .           +(B1(NF)**2*(DLOG(DLOG(X**2/XLB**2))**2
     .                      -DLOG(DLOG(X**2/XLB**2))-1)+B2(NF))
     .           /DLOG(X**2/XLB**2)**2)
      AA(NF)=12D0*PI/B0(NF)
      BB(NF)=B1(NF)/AA(NF)
      CC(NF)=B2(NF)/AA(NF)
      XIT(A,B,C,X)=A/2.D0*(1D0+DSQRT(1D0-4D0*B*DLOG(X)
     .          *(1-(A*B*(DLOG(X)**2-DLOG(X)-1)+C/B)/X/DLOG(X))))
      PI=4.D0*DATAN(1.D0)
      XLB2=XLB
      II=0
1     II=II+1
      X=DLOG(Q**2/XLB2**2)
      IF(NF1.LT.NF2)THEN
       DELTA = 7*ALS3(NF1,Q,XLB1)**2/PI**2/24
       ALP=ALS3(NF1,Q,XLB1)*(1+DELTA)
      ELSE
       DELTA = 7*ALS3(NF1,Q,XLB1)**2/PI**2/24
       ALP=ALS3(NF1,Q,XLB1)/(1+DELTA)
      ENDIF
      A=AA(NF2)/ALP
      B=BB(NF2)*ALP
      C=CC(NF2)*ALP
      XX=XIT(A,B,C,X)
      XLB2=Q*DEXP(-XX/2.D0)
      IF(NF1.LT.NF2)THEN
       DELTA = 7*ALS3(NF1,Q,XLB1)**2/PI**2/24
       Y1=ALS3(NF1,Q,XLB1)*(1+DELTA)
       Y2=ALS3(NF2,Q,XLB2)
      ELSE
       DELTA = 7*ALS3(NF1,Q,XLB1)**2/PI**2/24
       Y1=ALS3(NF1,Q,XLB1)/(1+DELTA)
       Y2=ALS3(NF2,Q,XLB2)
      ENDIF
      DY=DABS(Y2-Y1)/Y1
      IF(DY.GE.ACC) GOTO 1
       XITER3_HDEC=XLB2
      RETURN
      END

      DOUBLE PRECISION FUNCTION FINT_HDEC(Z,XX,YY)
C--ONE-DIMENSIONAL CUBIC INTERPOLATION
C--Z  = WANTED POINT
C--XX = ARRAY OF 4 DISCRETE X-VALUES AROUND Z
C--YY = ARRAY OF 4 DISCRETE FUNCTION-VALUES AROUND Z
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION XX(4),YY(4)
      X = DLOG(Z)
      X0=DLOG(XX(1))
      X1=DLOG(XX(2))
      X2=DLOG(XX(3))
      X3=DLOG(XX(4))
      Y0=DLOG(YY(1))
      Y1=DLOG(YY(2))
      Y2=DLOG(YY(3))
      Y3=DLOG(YY(4))
      A0=(X-X1)*(X-X2)*(X-X3)/(X0-X1)/(X0-X2)/(X0-X3)
      A1=(X-X0)*(X-X2)*(X-X3)/(X1-X0)/(X1-X2)/(X1-X3)
      A2=(X-X0)*(X-X1)*(X-X3)/(X2-X0)/(X2-X1)/(X2-X3)
      A3=(X-X0)*(X-X1)*(X-X2)/(X3-X0)/(X3-X1)/(X3-X2)
      FINT_HDEC=DEXP(A0*Y0+A1*Y1+A2*Y2+A3*Y3)
      RETURN
      END

      DOUBLE PRECISION FUNCTION SP_HDEC(X)
C--REAL DILOGARITHM (SPENCE-FUNCTION)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      COMPLEX*16 CX,LI2_HDEC
      CX = DCMPLX(X,0.D0)
      SP_HDEC = DREAL(LI2_HDEC(CX))
      RETURN
      END
 
      COMPLEX*16 FUNCTION LI2_HDEC(X)
C--COMPLEX DILOGARITHM (SPENCE-FUNCTION)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      COMPLEX*16 X,Y,CLI2_HDEC
      COMMON/CONST_HDEC/ZETA2,ZETA3
      ZERO=1.D-16
      XR=DREAL(X)
      XI=DIMAG(X)
      R2=XR*XR+XI*XI
      LI2_HDEC=0
      IF(R2.LE.ZERO)THEN
        LI2_HDEC=X
        RETURN
      ENDIF
      RR=XR/R2
      IF(R2.EQ.1.D0.AND.XI.EQ.0.D0)THEN
        IF(XR.EQ.1.D0)THEN
          LI2_HDEC=DCMPLX(ZETA2)
        ELSE
          LI2_HDEC=-DCMPLX(ZETA2/2.D0)
        ENDIF
        RETURN
      ELSEIF(R2.GT.1.D0.AND.RR.GT.0.5D0)THEN
        Y=(X-1.D0)/X
        LI2_HDEC=CLI2_HDEC(Y)+ZETA2-CDLOG(X)*CDLOG(1.D0-X)
     .          +0.5D0*CDLOG(X)**2
        RETURN
      ELSEIF(R2.GT.1.D0.AND.RR.LE.0.5D0)THEN
        Y=1.D0/X
        LI2_HDEC=-CLI2_HDEC(Y)-ZETA2-0.5D0*CDLOG(-X)**2
        RETURN
      ELSEIF(R2.LE.1.D0.AND.XR.GT.0.5D0)THEN
        Y=1.D0-X
        LI2_HDEC=-CLI2_HDEC(Y)+ZETA2-CDLOG(X)*CDLOG(1.D0-X)
       RETURN
      ELSEIF(R2.LE.1.D0.AND.XR.LE.0.5D0)THEN
        Y=X
        LI2_HDEC=CLI2_HDEC(Y)
        RETURN
      ENDIF
      END
 
      COMPLEX*16 FUNCTION CLI2_HDEC(X)
C--TAYLOR-EXPANSION FOR COMPLEX DILOGARITHM (SPENCE-FUNCTION)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      COMPLEX*16 X,Z
      COMMON/BERNOULLI_HDEC/B2(18),B12(18),B3(18)
      COMMON/POLY_HDEC/NBER
      N=NBER-1
      Z=-CDLOG(1.D0-X)
      CLI2_HDEC=B2(NBER)
      DO 111 I=N,1,-1
        CLI2_HDEC=Z*CLI2_HDEC+B2(I)
111   CONTINUE
      CLI2_HDEC=Z**2*CLI2_HDEC+Z
      RETURN
      END
 
      DOUBLE PRECISION FUNCTION FACTRL_HDEC(N)
C--DOUBLE PRECISION VERSION OF FACTORIAL
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      FACTRL_HDEC=1.D0
      IF(N.EQ.0)RETURN
      DO 999 I=1,N
        FACTRL_HDEC=FACTRL_HDEC*DFLOAT(I)
999   CONTINUE
      RETURN
      END
 
      SUBROUTINE BERNINI_HDEC(N)
C--INITIALIZATION OF COEFFICIENTS FOR POLYLOGARITHMS
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION B(18),PB(19)
      COMMON/BERNOULLI_HDEC/B2(18),B12(18),B3(18)
      COMMON/CONST_HDEC/ZETA2,ZETA3
      COMMON/POLY_HDEC/NBER
 
      NBER=N
      PI=4.D0*DATAN(1.D0)
 
      B(1)=-1.D0/2.D0
      B(2)=1.D0/6.D0
      B(3)=0.D0
      B(4)=-1.D0/30.D0
      B(5)=0.D0
      B(6)=1.D0/42.D0
      B(7)=0.D0
      B(8)=-1.D0/30.D0
      B(9)=0.D0
      B(10)=5.D0/66.D0
      B(11)=0.D0
      B(12)=-691.D0/2730.D0
      B(13)=0.D0
      B(14)=7.D0/6.D0
      B(15)=0.D0
      B(16)=-3617.D0/510.D0
      B(17)=0.D0
      B(18)=43867.D0/798.D0
      ZETA2=PI**2/6.D0
      ZETA3=1.202056903159594D0
 
      DO 995 I=1,18
        B2(I)=B(I)/FACTRL_HDEC(I+1)
        B12(I)=DFLOAT(I+1)/FACTRL_HDEC(I+2)*B(I)/2.D0
        PB(I+1)=B(I)
        B3(I)=0.D0
995   CONTINUE
      PB(1)=1.D0
      DO 996 I=1,18
      DO 996 J=0,I
        B3(I)=B3(I)+PB(J+1)*PB(I-J+1)/FACTRL_HDEC(I-J)/FACTRL_HDEC(J+1)
     .                                            /DFLOAT(I+1)
996   CONTINUE
 
      RETURN
      END

      DOUBLE PRECISION FUNCTION QQINT_HDEC(RAT,H1,H2)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      N = 2
      QQINT_HDEC = RAT**N * H1 + (1-RAT**N) * H2
      RETURN
      END

      DOUBLE PRECISION FUNCTION XITLA_HDEC(NO,ALP,ACC)
C--ITERATION ROUTINE TO DETERMINE IMPROVED LAMBDAS
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      COMMON/PARAM_HDEC/GF,ALPH,AMTAU,AMMUON,AMZ,AMW
      B0(NF)=33.D0-2.D0*NF
      B1(NF)=6.D0*(153.D0-19.D0*NF)/B0(NF)**2
      B2(NF)=27/2.D0*(2857-5033/9.D0*NF+325/27.D0*NF**2)/B0(NF)**3
      ALS2(NF,X,XLB)=12.D0*PI/(B0(NF)*DLOG(X**2/XLB**2))
     .              *(1.D0-B1(NF)*DLOG(DLOG(X**2/XLB**2))
     .              /DLOG(X**2/XLB**2))
      ALS3(NF,X,XLB)=12.D0*PI/(B0(NF)*DLOG(X**2/XLB**2))
     .          *(1.D0-B1(NF)*DLOG(DLOG(X**2/XLB**2))
     .           /DLOG(X**2/XLB**2)
     .           +(B1(NF)**2*(DLOG(DLOG(X**2/XLB**2))**2
     .                      -DLOG(DLOG(X**2/XLB**2))-1)+B2(NF))
     .           /DLOG(X**2/XLB**2)**2)
      AA(NF)=12D0*PI/B0(NF)
      BB(NF)=B1(NF)/AA(NF)
      CC(NF)=B2(NF)/AA(NF)
      XIT(A,B,X)=A/2.D0*(1D0+DSQRT(1D0-4D0*B*DLOG(X)))
      XIT3(A,B,C,X)=A/2.D0*(1D0+DSQRT(1D0-4D0*B*DLOG(X)
     .          *(1-(A*B*(DLOG(X)**2-DLOG(X)-1)+C/B)/X/DLOG(X))))
      PI=4.D0*DATAN(1.D0)
      NF=5
      Q=AMZ
      XLB=Q*DEXP(-AA(NF)/ALP/2.D0)
      IF(NO.EQ.1)GOTO 111
      II=0
1     II=II+1
      X=DLOG(Q**2/XLB**2)
      A=AA(NF)/ALP
      B=BB(NF)*ALP
      C=CC(NF)*ALP
      IF(NO.EQ.2)THEN
       XX=XIT(A,B,X)
      ELSE
       XX=XIT3(A,B,C,X)
      ENDIF
      XLB=Q*DEXP(-XX/2.D0)
      Y1=ALP
      IF(NO.EQ.2)THEN
       Y2=ALS2(NF,Q,XLB)
      ELSE
       Y2=ALS3(NF,Q,XLB)
      ENDIF
      DY=DABS(Y2-Y1)/Y1
      IF(DY.GE.ACC) GOTO 1
111   XITLA_HDEC=XLB
      RETURN
      END

C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      DOUBLE PRECISION FUNCTION COFSUSY_HDEC(IHIGGS,AMB,RMB,QQ)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      COMPLEX*16 C03_HDEC
      COMMON/PARAM_HDEC/GF,ALPH,AMTAU,AMMUON,AMZ,AMW
      COMMON/HMASS_HDEC/AMSM,AMA,AMHL,AMHH,AMCH,AMAR
      COMMON/GLUINO_HDEC/AMG,AMSB1,AMSB2,STH,CTH,
     .              GLBB(2,2),GHBB(2,2),GABB(2,2),
     .              AMST1,AMST2,STHT,CTHT,
     .              GLTT(2,2),GHTT(2,2),GATT(2,2)
      COMMON/COUP_HDEC/GAT,GAB,GLT,GLB,GHT,GHB,GZAH,GZAL,
     .            GHHH,GLLL,GHLL,GLHH,GHAA,GLAA,GLVV,GHVV,
     .            GLPM,GHPM,B,A
      COMMON/BREAK_HDEC/AMEL,AMER,AMSQ,AMUR,AMDR,AL,AU,AD,AMU,AM2
      FC1(VI,VJ,AI,AJ,BI,BJ,BIJ,CIJ,AMI,AMJ) = -1.D0/4*
     .  ((VI*VJ+AI*AJ)/(AMH**2-4*AMB**2)*(BI+BJ-2*BIJ
     .                  +(AMI**2+AMJ**2-2*AMG**2-2*AMB**2)*CIJ)
     .  + AMG/RMB*(VI*VJ-AI*AJ)*CIJ)
      FCA(VI,VJ,AI,AJ,BI,BJ,BIJ,CIJ,AMI,AMJ) = -1.D0/4*(
     .   (VI*AJ+AI*VJ)/AMH**2*(BJ-BI+(AMI**2-AMJ**2)*CIJ)
     .  + AMG/RMB*(AI*VJ-VI*AJ)*CIJ
     .  )
      FC2(VI,AI,A0I,A0G,BI,BPI,AMI) = -1.D0/8*(
     .   (VI**2+AI**2)/2/AMB**2*(A0G-A0I+(AMB**2-AMG**2+AMI**2)*BI
     .                          +2*(AMB**2+AMG**2-AMI**2)*AMB**2*BPI)
     .  + 2*AMG*AMB*(VI**2-AI**2)*BPI
     .  )
      FC3(VI,AI,A0I,A0G,BI,AMI,Q2) = 1.D0/8*(
     .   (VI**2+AI**2)/2/Q2*(A0I-A0G+(Q2+AMG**2-AMI**2)*BI)
     .   +AMG/RMB*(VI**2-AI**2)*BI
     .  )
      FC4(VI,AI,A0I,A0G,BI,BPI,AMI) = 1.D0/8*(
     .   (VI**2+AI**2)/2*((A0G-A0I)/(AMG**2-AMI**2)
     .   +(AMG**2-AMI**2)*BPI)
     .   +AMG/RMB*(VI**2-AI**2)*BI
     .  )
      CF = 4.D0/3.D0
      PI = 4*DATAN(1.D0)
c     write(6,*)
      IF(IHIGGS.EQ.1)THEN
       AMH = AMHL
       GLO = GLB/AMZ**2
       G11 = GLBB(1,1)/GLO
       G12 = GLBB(1,2)/GLO
       G21 = GLBB(2,1)/GLO
       G22 = GLBB(2,2)/GLO
c      write(6,*)'h:'
      ELSEIF(IHIGGS.EQ.2)THEN
       AMH = AMHH
       GLO = GHB/AMZ**2
       G11 = GHBB(1,1)/GLO
       G12 = GHBB(1,2)/GLO
       G21 = GHBB(2,1)/GLO
       G22 = GHBB(2,2)/GLO
c      write(6,*)'H:'
      ELSEIF(IHIGGS.EQ.3)THEN
       AMH = AMA
       GLO = GAB/AMZ**2
       G11 = GABB(1,1)/GLO
       G12 = GABB(1,2)/GLO
       G21 = GABB(2,1)/GLO
       G22 = GABB(2,2)/GLO
c      write(6,*)'A:'
      ENDIF
c     write(6,*)'=='
c     write(6,*)'MB, MSB1, MSB2, MG: ',AMB,AMSB1,AMSB2,AMG
c     write(6,*)'LO,11,12,21,22: ',
c    .          GLO*AMZ**2,G11,G12,G21,G22
c     write(6,*)'SIN(THETA), COS(THETA): ',STH,CTH
      XMU = AMH
      V1 = CTH-STH
      V2 = -CTH-STH
      A1 = CTH+STH
      A2 = CTH-STH
      CC11  = DREAL(C03_HDEC(AMB**2,AMB**2,AMH**2,AMSB1,AMG,AMSB1))
      CC12  = DREAL(C03_HDEC(AMB**2,AMB**2,AMH**2,AMSB1,AMG,AMSB2))
      CC21  = DREAL(C03_HDEC(AMB**2,AMB**2,AMH**2,AMSB2,AMG,AMSB1))
      CC22  = DREAL(C03_HDEC(AMB**2,AMB**2,AMH**2,AMSB2,AMG,AMSB2))
      BB1 = B02_HDEC(AMB**2,AMG,AMSB1,XMU**2)
      BB2 = B02_HDEC(AMB**2,AMG,AMSB2,XMU**2)
      BB11 = B02_HDEC(AMH**2,AMSB1,AMSB1,XMU**2)
      BB12 = B02_HDEC(AMH**2,AMSB1,AMSB2,XMU**2)
      BB21 = B02_HDEC(AMH**2,AMSB2,AMSB1,XMU**2)
      BB22 = B02_HDEC(AMH**2,AMSB2,AMSB2,XMU**2)
      BP1 = BP02_HDEC(AMB**2,AMG,AMSB1,XMU**2)
      BP2 = BP02_HDEC(AMB**2,AMG,AMSB2,XMU**2)
      AA1 = AMSB1**2*(1+DLOG(XMU**2/AMSB1**2))
      AA2 = AMSB2**2*(1+DLOG(XMU**2/AMSB2**2))
      AAG = AMG**2*(1+DLOG(XMU**2/AMG**2))
      BCT1 = B02_HDEC(QQ**2,AMG,AMSB1,XMU**2)
      BCT2 = B02_HDEC(QQ**2,AMG,AMSB2,XMU**2)
      BPCT1 = BP02_HDEC(QQ**2,AMG,AMSB1,XMU**2)
      BPCT2 = BP02_HDEC(QQ**2,AMG,AMSB2,XMU**2)
c     write(6,*)'A0: m1, m2, mg: ',AA1,AA2,AAG
c     write(6,*)'B0: g1, g2, 11, 12, 21, 22: ',BB1,BB2,BB11,BB12,BB21,BB22
c     write(6,*)'B''0: g1, g2: ',BP1,BP2
c     write(6,*)'B''0: g1, g2: ',BPCT1,BPCT2
c     write(6,*)'C0: 11, 12, 21, 22: ',CC11,CC12,CC21,CC22
      IF(IHIGGS.EQ.3)THEN
       COF1 = G11*FCA(V1,V1,A1,A1,BB1,BB1,BB11,CC11,AMSB1,AMSB1)
     .      + G12*FCA(V1,V2,A1,A2,BB1,BB2,BB12,CC12,AMSB1,AMSB2)
     .      + G21*FCA(V2,V1,A2,A1,BB2,BB1,BB21,CC21,AMSB2,AMSB1)
     .      + G22*FCA(V2,V2,A2,A2,BB2,BB2,BB22,CC22,AMSB2,AMSB2)
      ELSE
       COF1 = G11*FC1(V1,V1,A1,A1,BB1,BB1,BB11,CC11,AMSB1,AMSB1)
     .      + G12*FC1(V1,V2,A1,A2,BB1,BB2,BB12,CC12,AMSB1,AMSB2)
     .      + G21*FC1(V2,V1,A2,A1,BB2,BB1,BB21,CC21,AMSB2,AMSB1)
     .      + G22*FC1(V2,V2,A2,A2,BB2,BB2,BB22,CC22,AMSB2,AMSB2)
      ENDIF
      COF2 = FC2(V1,A1,AA1,AAG,BB1,BP1,AMSB1)
     .     + FC2(V2,A2,AA2,AAG,BB2,BP2,AMSB2)
      IF(QQ.EQ.0.D0)THEN
       COF3 = FC4(V1,A1,AA1,AAG,BCT1,BPCT1,AMSB1)
     .      + FC4(V2,A2,AA2,AAG,BCT2,BPCT2,AMSB2)
      ELSE
       COF3 = FC3(V1,A1,AA1,AAG,BCT1,AMSB1,QQ**2)
     .      + FC3(V2,A2,AA2,AAG,BCT2,AMSB2,QQ**2)
      ENDIF
      COF1 = 2*CF*COF1
      COF2 = 2*CF*COF2
      COF3 = 2*CF*COF3
      COFSUSY_HDEC = COF1 + COF2 + COF3
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     COFSUSY_HDEC = 0
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     s2th0 = 2*rmb*(au-amu*dtan(b))/(amsb1**2-amsb2**2)
c     s2th1 = 2*sth*cth
c     s2th2 = 2*rmb*(-amu*dtan(b))/(amsb1**2-amsb2**2)
c     write(6,*)
c     write(6,*)'s2th: ',s2th0,s2th1,s2th2
c     write(6,*)'s2th: ',rmb,au,amu,dtan(b),amsb1,amsb2
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     write(6,*)
c     write(6,*)'SUSY-QCD: ',2*cf*cof1,2*cf*cof2,2*cf*cof3
c     write(6,*)
c     write(6,*)AMSB1,AMSB2,G11,G12,G22
c     write(6,*)
c     write(6,*)2*CF*G11*FC1(V1,V1,A1,A1,BB1,BB1,BB11,CC11,AMSB1,AMSB1)
c    .      , 2*CF*G12*FC1(V1,V2,A1,A2,BB1,BB2,BB12,CC12,AMSB1,AMSB2)
c    .      , 2*CF*G21*FC1(V2,V1,A2,A1,BB2,BB1,BB21,CC21,AMSB2,AMSB1)
c    .      , 2*CF*G22*FC1(V2,V2,A2,A2,BB2,BB2,BB22,CC22,AMSB2,AMSB2)
c     write(6,*)
c     write(6,*)2*CF*G11*FC1(V1,V1,A1,A1,BB1,BB1,0.D0,0.D0,AMSB1,AMSB1),
c    .        2*CF*G11*FC1(V1,V1,A1,A1,0.D0,0.D0,BB11,0.D0,AMSB1,AMSB1),
c    .        2*CF*G11*FC1(V1,V1,A1,A1,0.D0,0.D0,0.D0,CC11,AMSB1,AMSB1),
c    .        -2*CF*G11*(V1*V1+A1*A1)/(AMH**2-4*AMB**2)*
c    .                  (AMSB1**2+AMSB1**2-2*AMG**2-2*AMB**2)*CC11/4,
c    .        -2*CF*G11*AMG/RMB*(V1*V1-A1*A1)*CC11/4,
c    .        RMB,AMB,(V1*V1-A1*A1),(V1*V1+A1*A1),V1,A1
c     FC1(VI,VJ,AI,AJ,BI,BJ,BIJ,CIJ,AMI,AMJ) = -1.D0/4*
c    .  ((VI*VJ+AI*AJ)/(AMH**2-4*AMB**2)*(BI+BJ-2*BIJ
c    .                  +(AMI**2+AMJ**2-2*AMG**2-2*AMB**2)*CIJ)
c    .  + AMG/RMB*(VI*VJ-AI*AJ)*CIJ)
c     write(6,*)
      RETURN
      END
 
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      SUBROUTINE BOTSUSY_HDEC(GLB,GHB,GAB,XGLB,XGHB,XGAB,SCALE,IL)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      ICASE = 0
      CALL DMBAPP_HDEC(ICASE,DGLB,DGHB,DGAB,SCALE,IL)
      XGLB = GLB*(1+DGLB)
      XGHB = GHB*(1+DGHB)
      XGAB = GAB*(1+DGAB)
      RETURN
      END
 
      SUBROUTINE DMBAPP_HDEC(ICASE,DGLB,DGHB,DGAB,SCALE,IL)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION AMCHAR(2),AMNEUT(4),XMNEUT(4),
     .          XMST(2),XMSB(2),AMSL(2),
     .          AMSU(2),AMSD(2),AMSE(2),AMSN(2),AMSN1(2)
      COMMON/PARAM_HDEC/GF,ALPH,AMTAU,AMMUON,AMZ,AMW
      COMMON/MASSES_HDEC/AMS,AMC,AMB,AMT
      COMMON/HMASS_HDEC/AMSM,AMA,AMHL,AMHH,AMCH,AMAR
      COMMON/GLUINO_HDEC/AMG,AMSB1,AMSB2,STH,CTH,
     .              GLBB(2,2),GHBB(2,2),GABB(2,2),
     .              AMST1,AMST2,STHT,CTHT,
     .              GLTT(2,2),GHTT(2,2),GATT(2,2)
      COMMON/COUP_HDEC/GAT,GAB,GLT,GLB,GHT,GHB,GZAH,GZAL,
     .            GHHH,GLLL,GHLL,GLHH,GHAA,GLAA,GLVV,GHVV,
     .            GLPM,GHPM,B,A
      COMMON/BREAK_HDEC/AMEL,AMER,AMSQ,AMUR,AMDR,AL,AU,AD,AMU,AM2
      COMMON/SMASS_HDEC/AMNEUT,XMNEUT,AMCHAR,XMST,XMSB,AMSL,
     .              AMSU,AMSD,AMSE,AMSN,AMSN1
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      COMMON/DAVID/QSUSY1,QSUSY2,LOOP
      COMMON/ALSLAM_HDEC/XLB1(6),XLB2(6),XLB3(6)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      DELTA(AM1,AM2) = (DABS(AM1)-DABS(AM2))/(DABS(AM1)+DABS(AM2))
      PI = 4*DATAN(1.D0)
      V  = 1/DSQRT(2*DSQRT(2D0)*GF)
      TANB = DTAN(B)
      TANA = DTAN(A)
      SB = TANB/DSQRT(1+TANB**2)
      AT = AU
      AB = AD
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      SCALELW = QSUSY2*(AMST1+AMST2+DABS(AMU))/3
      SCALQCD = QSUSY1*(AMSB1+AMSB2+DABS(AMG))/3
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      RMTOP   = RUNM_HDEC(SCALELW,6)
      HT = RMTOP/V/SB
      STOP1 = AMST1
      STOP2 = AMST2
      SBOT1 = AMSB1
      SBOT2 = AMSB2
      CW = AMW**2/AMZ**2
      SW = 1-CW
      AL2  = (2*AMW/V/DSQRT(2D0))**2/4/PI
      AL1  = AL2*SW/CW
      AM1=5.D0/3.D0*SW/CW*AM2

      FELW = 1
      FQCD = 1
      IF(IL.EQ.2)THEN
       ASH = ALPHAS_HDEC(SCALELW,3)
       CELW = FELW_HDEC(SCALELW,AMU,AMG,SBOT1,SBOT2,STOP1,STOP2,AMT)
       CUT = 0.01D0
       IF(DABS(DELTA(STOP1,STOP2)).LT.CUT.OR.
     .    DABS(DELTA(STOP1,AMG)).LT.CUT.OR.
     .    DABS(DELTA(STOP2,AMG)).LT.CUT.OR.
     .    DABS(DELTA(SBOT1,SBOT2)).LT.CUT.OR.
     .    DABS(DELTA(SBOT1,AMG)).LT.CUT.OR.
     .    DABS(DELTA(SBOT2,AMG)).LT.CUT)THEN
        DEL = 2*CUT
        IF(STOP1.LE.AMG)THEN
         IF(STOP2.LE.AMG)THEN
          ST1 = STOP1*(1-2*DEL)
          ST2 = STOP2*(1-DEL)
         ELSE
          ST1 = STOP1*(1-DEL)
          ST2 = STOP2*(1+DEL)
         ENDIF
        ELSE
         ST1 = STOP1*(1+DEL)
         ST2 = STOP2*(1+2*DEL)
        ENDIF
        IF(SBOT1.LE.AMG)THEN
         IF(SBOT2.LE.AMG)THEN
          SB1 = SBOT1*(1-2*DEL)
          SB2 = SBOT2*(1-DEL)
         ELSE
          SB1 = SBOT1*(1-DEL)
          SB2 = SBOT2*(1+DEL)
         ENDIF
        ELSE
         SB1 = SBOT1*(1+DEL)
         SB2 = SBOT2*(1+2*DEL)
        ENDIF
        C1 = FELW_HDEC(SCALELW,AMU,AMG,SB1,SB2,ST1,ST2,AMT)
        DEL = CUT
        IF(STOP1.LE.AMG)THEN
         IF(STOP2.LE.AMG)THEN
          ST1 = STOP1*(1-2*DEL)
          ST2 = STOP2*(1-DEL)
         ELSE
          ST1 = STOP1*(1-DEL)
          ST2 = STOP2*(1+DEL)
         ENDIF
        ELSE
         ST1 = STOP1*(1+DEL)
         ST2 = STOP2*(1+2*DEL)
        ENDIF
        IF(SBOT1.LE.AMG)THEN
         IF(SBOT2.LE.AMG)THEN
          SB1 = SBOT1*(1-2*DEL)
          SB2 = SBOT2*(1-DEL)
         ELSE
          SB1 = SBOT1*(1-DEL)
          SB2 = SBOT2*(1+DEL)
         ENDIF
        ELSE
         SB1 = SBOT1*(1+DEL)
         SB2 = SBOT2*(1+2*DEL)
        ENDIF
        C2 = FELW_HDEC(SCALELW,AMU,AMG,SB1,SB2,ST1,ST2,AMT)
        CELW = C2 + (C2-C1)
c       write(6,*)'elw: ',c1,c2,celw
       ENDIF
       FELW = 1+ASH/PI*CELW
       ASH = ALPHAS_HDEC(SCALQCD,3)
       XXT = AU - AMU/TANB
       CQCD = FQCD_HDEC(SCALQCD,AMT,AMG,SBOT1,SBOT2,STOP1,STOP2,
     .                  AMSU(1),AMSU(2),AMSD(1),AMSD(2),XXT)
       CUT = 0.01D0
       IF(DABS(DELTA(STOP1,STOP2)).LT.CUT.OR.
     .    DABS(DELTA(STOP1,AMG)).LT.CUT.OR.
     .    DABS(DELTA(STOP2,AMG)).LT.CUT.OR.
     .    DABS(DELTA(SBOT1,SBOT2)).LT.CUT.OR.
     .    DABS(DELTA(SBOT1,AMG)).LT.CUT.OR.
     .    DABS(DELTA(SBOT2,AMG)).LT.CUT)THEN
        DEL = 2*CUT
        IF(STOP1.LE.AMG)THEN
         IF(STOP2.LE.AMG)THEN
          ST1 = STOP1*(1-2*DEL)
          ST2 = STOP2*(1-DEL)
         ELSE
          ST1 = STOP1*(1-DEL)
          ST2 = STOP2*(1+DEL)
         ENDIF
        ELSE
         ST1 = STOP1*(1+DEL)
         ST2 = STOP2*(1+2*DEL)
        ENDIF
        IF(SBOT1.LE.AMG)THEN
         IF(SBOT2.LE.AMG)THEN
          SB1 = SBOT1*(1-2*DEL)
          SB2 = SBOT2*(1-DEL)
         ELSE
          SB1 = SBOT1*(1-DEL)
          SB2 = SBOT2*(1+DEL)
         ENDIF
        ELSE
         SB1 = SBOT1*(1+DEL)
         SB2 = SBOT2*(1+2*DEL)
        ENDIF
        C1 = FQCD_HDEC(SCALQCD,AMT,AMG,SB1,SB2,ST1,ST2,
     .                 AMSU(1),AMSU(2),AMSD(1),AMSD(2),XXT)
        DEL = CUT
        IF(STOP1.LE.AMG)THEN
         IF(STOP2.LE.AMG)THEN
          ST1 = STOP1*(1-2*DEL)
          ST2 = STOP2*(1-DEL)
         ELSE
          ST1 = STOP1*(1-DEL)
          ST2 = STOP2*(1+DEL)
         ENDIF
        ELSE
         ST1 = STOP1*(1+DEL)
         ST2 = STOP2*(1+2*DEL)
        ENDIF
        IF(SBOT1.LE.AMG)THEN
         IF(SBOT2.LE.AMG)THEN
          SB1 = SBOT1*(1-2*DEL)
          SB2 = SBOT2*(1-DEL)
         ELSE
          SB1 = SBOT1*(1-DEL)
          SB2 = SBOT2*(1+DEL)
         ENDIF
        ELSE
         SB1 = SBOT1*(1+DEL)
         SB2 = SBOT2*(1+2*DEL)
        ENDIF
        C2 = FQCD_HDEC(SCALQCD,AMT,AMG,SB1,SB2,ST1,ST2,
     .                 AMSU(1),AMSU(2),AMSD(1),AMSD(2),XXT)
        CQCD = C2 + (C2-C1)
c       write(6,*)'qcd: ',c1,c2,cqcd
       ENDIF
       FQCD = 1+ASH/PI*CQCD
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c--comparison with luminita
c     xmg = 1000
c     xmsb1 = 600.7326590270438d0
c     xmsb2 = 960.8906796048134d0
c     xlphas = 0.088374016132112168d0
c     xmg = amg
c     xmsb1 = amsb1
c     xmsb2 = amsb2
c     xscalqcd = (xmsb1+xmsb2+xmg)/3
c     xlphas = alphas_hdec(xscalqcd,3)
c     xxt = au - amu/tanb
c     xfac = fqcd_hdec(scalqcd,amt,xmg,xmsb1,xmsb2,stop1,stop2,
c    .                 amsu(1),amsu(2),amsd(1),amsd(2),xxt)
c     xfac = 1 + xfac*xlphas/pi
c     xdeltamb = 2*xlphas/3/pi*xmg*amu*tanb*t_hdec(xmsb1,xmsb2,xmg)
c     write(6,*)'alpha_s: ',xlphas
c     write(6,*)'scale:   ',xscalqcd
c     write(6,*)'gluino:  ',xmg
c     write(6,*)'sbottom: ',xmsb1,xmsb2
c     write(6,*)'stop:    ',stop1,stop2
c     write(6,*)'sup:     ',amsu(1),amsu(2)
c     write(6,*)'sdown:   ',amsd(1),amsd(2)
c     write(6,*)'NLO, NNLO: ',xdeltamb,xdeltamb*xfac
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c      write(6,*)'QCD: ',CQCD,FQCD-1,
c    .           2*ASH/3/PI*AMG*AMU*TANB*T_HDEC(SBOT1,SBOT2,AMG)
c      write(6,*)'elw: ',CELW,FELW-1
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c  Equal masses
c      EP = 1.D-2
c      CA = 3
c      CF = 4/3.D0
c      TR = 1/2.D0
c      NF = 5
c      BETAL = (11*CA-4*TR*NF)/12.D0
c      SB1 = AMG*(1+EP)
c      SB2 = AMG*(1-EP)
c      ST1 = AMG*(1+EP)
c      ST2 = AMG*(1-EP)
c      SU1 = AMG*(1+EP)
c      SU2 = AMG*(1-EP)
c      SD1 = AMG*(1+EP)
c      SD2 = AMG*(1-EP)
c      XMU = AMG
c      SQ1 = AMG
c      XMT = AMT/100
c      XNN = 1/2.D0
c      XN0 = T_HDEC(SB1,SB2,AMG)*AMG**2
c      write(6,*)
c      write(6,*)'Equal masses'
c      write(6,*)'============'
c      write(6,*)'LO = ',XN0
c      write(6,*)'     ',XNN,XNN/XN0
c      write(6,*)'CA = ',CA/3+11*CA/12*DLOG(SQ1**2/AMG**2)
c      write(6,*)'CF = ',CF
c      write(6,*)'TR = ',TR*((NF+1)/2.D0 + 1/3.D0*DLOG(AMG**2/XMT**2)
c    .                      -4*NF/12.D0*DLOG(SQ1**2/AMG**2))
c      write(6,*)
c      XXT = AU - AMU/TANB
c      XQCD = FQCD_HDEC(SQ1,XMT,AMG,SB1,SB2,ST1,ST2,
c    .                  SU1,SU2,SD1,SD2,XXT)
c      TQ = CA/3 + CF + TR*((NF+1)/2.D0 + 1/3.D0*DLOG(AMG**2/XMT**2))
c    .    + BETAL*DLOG(SQ1**2/AMG**2)
c      write(6,*)
c      write(6,*)'QCD: ',XQCD
c      write(6,*)'     ',TQ,TQ/XQCD
c      write(6,*)'full: ',CQCD
c      write(6,*)'     ',TQ,TQ/CQCD
c      write(6,*)
c      XELW = FELW_HDEC(SQ1,XMU,AMG,SB1,SB2,ST1,ST2,XMT)
c      TE = CF * (7/4.D0 + 3/2.D0*DLOG(SQ1**2/XMT/AMG))
c      write(6,*)'elw: ',XELW
c      write(6,*)'     ',TE,TE/XELW
c      write(6,*)'full: ',CELW
c      write(6,*)'     ',TE,TE/CELW
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c  Large gluino mass
c      FAC = 100
c      CA = 3
c      CF = 4/3.D0
c      TR = 1/2.D0
c      NU = 2
c      ND = 2
c      NF = NU+ND+1
c      BETAL = (11*CA-4*TR*NF)/12.D0
c      XMG = AMG*FAC
c      SB1 = SBOT1
c      SB2 = SBOT2
c      ST1 = STOP1
c      ST2 = STOP2
c      SU1 = AMSU(1)
c      SU2 = AMSU(2)
c      SD1 = AMSD(1)
c      SD2 = AMSD(2)
c      XMU = AMU
c      SQ1 = XMG
c      XMT = AMT
c      ZETA2 = PI**2/6
c      XNN = DLOG(XMG**2/SB2**2)
c    .     - SB1**2*DLOG(SB2**2/SB1**2)/(SB2**2-SB1**2)
c      XN0 = T_HDEC(SB1,SB2,XMG)*XMG**2
c      write(6,*)
c      write(6,*)'Large gluino mass'
c      write(6,*)'================='
c      write(6,*)'LO = ',XN0
c      write(6,*)'     ',XNN,XNN/XN0
c      write(6,*)'CA = ',4*CA/3+11*CA/12*DLOG(SQ1**2/XMG**2)
c    .                  + 0*(ZETA2-1)/2/XNN
c      write(6,*)'CF = ',CF*(XNN+5/2.D0 + 0*(1-4*ZETA2)/4/XNN)
c      write(6,*)'TR = ',TR*(-(NF+1) + 1/3.D0*DLOG(XMG**2/XMT**2)
c    .         + NU/12.D0*DLOG(XMG**4/SU1**2/SU2**2)
c    .         + ND/12.D0*DLOG(XMG**4/SD1**2/SD2**2)
c    .         + 1/12.D0*DLOG(XMG**4/SB1**2/SB2**2)
c    .         + 1/12.D0*DLOG(XMG**4/ST1**2/ST2**2)
c    .                      -4*NF/12.D0*DLOG(SQ1**2/XMG**2))
c      write(6,*)
c      XXT = AU - AMU/TANB
c      XQCD = FQCD_HDEC(SQ1,XMT,XMG,SB1,SB2,ST1,ST2,
c    .                  SU1,SU2,SD1,SD2,XXT)
c      TQ = 4*CA/3 + CF*(XNN+5/2.D0)
c    .    + TR*(-(NF+1) + 1/3.D0*DLOG(XMG**2/XMT**2)
c    .         + NU/12.D0*DLOG(XMG**4/SU1**2/SU2**2)
c    .         + ND/12.D0*DLOG(XMG**4/SD1**2/SD2**2)
c    .         + 1/12.D0*DLOG(XMG**4/SB1**2/SB2**2)
c    .         + 1/12.D0*DLOG(XMG**4/ST1**2/ST2**2))
c    .    + BETAL*DLOG(SQ1**2/XMG**2)
c      write(6,*)
c      write(6,*)'QCD: ',XQCD
c      write(6,*)'     ',TQ,TQ/XQCD
c      write(6,*)'full: ',CQCD
c      write(6,*)'     ',TQ,TQ/CQCD
c      write(6,*)
c      XELW = FELW_HDEC(SQ1,XMU,XMG,SB1,SB2,ST1,ST2,XMT)
c      TE = CF * (23/8.D0 + 3/2.D0*DLOG(SQ1**2/XMT/XMG))
c      write(6,*)'elw: ',XELW
c      write(6,*)'     ',TE,TE/XELW
c      write(6,*)'full: ',CELW
c      write(6,*)'     ',TE,TE/CELW
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      ENDIF
 
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     FQCD = 1
c     FELW = 1
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      ASH = ALPHAS_HDEC(SCALQCD,3)
      IF(ICASE.EQ.0)THEN
       DELTAMB = (2*ASH/3/PI*AMG*AMU*TANB*T_HDEC(SBOT1,SBOT2,AMG)*FQCD
     * + HT**2/(4*PI)**2*AT*AMU*TANB*T_HDEC(STOP1,STOP2,AMU)*FELW
     . - AL1/12/PI*AM1*AMU*TANB*(T_HDEC(SBOT1,SBOT2,AM1)/3
     .      + (CTH**2/2+STH**2)*T_HDEC(SBOT1,AM1,AMU)
     .      + (STH**2/2+CTH**2)*T_HDEC(SBOT2,AM1,AMU))
     . - AL2/4/PI*AM2*AMU*TANB*(CTHT**2*T_HDEC(STOP1,AM2,AMU)
     .                         +STHT**2*T_HDEC(STOP2,AM2,AMU)
     . +(CTH**2*T_HDEC(SBOT1,AM2,AMU)+STH**2*T_HDEC(SBOT2,AM2,AMU))/2))
     *         /(1-2*ASH/3/PI*AMG*AB*T_HDEC(SBOT1,SBOT2,AMG))
       DGLB = -DELTAMB/(1+DELTAMB)*(1+1/TANA/TANB)
       DGHB = -DELTAMB/(1+DELTAMB)*(1-TANA/TANB)
       DGAB = -DELTAMB/(1+DELTAMB)*(1+1/TANB**2)
c      write(6,*)'Delta_b: ',DELTAMB,DGLB,DGHB,DGAB
c      write(6,*)'Delta_b: ',FQCD,FELW,DGLB,DGHB,DGAB
c      write(6,*)'Delta_b: ',DELTAMB,SBOT1,SBOT2,STOP1,STOP2,AMG
c    .          ,2*ASH/3/PI*AMG*AMU*TANB*T_HDEC(SBOT1,SBOT2,AMG),FQCD
c    .         ,HT**2/(4*PI)**2*AT*AMU*TANB*T_HDEC(STOP1,STOP2,AMU),FELW
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c      sb1 = sbot1 * (1-4*ash/pi/3)
c      sb2 = sbot2 * (1-4*ash/pi/3)
c      st1 = stop1 * (1-4*ash/pi/3)
c      st2 = stop2 * (1-4*ash/pi/3)
c      atp = at - 4*ash/pi/3*amg*dlog(4.d0)
c      write(6,*)'Approx_b: ',SBOT1,SBOT2,sb1,sb2,
c    . sb1/SBOT1,sb2/SBOT2,T_HDEC(SB1,SB2,AMG)/T_HDEC(SBOT1,SBOT2,AMG)
c      write(6,*)'Approx_t: ',STOP1,STOP2,st1,st2,
c    . st1/STOP1,st2/STOP2,T_HDEC(ST1,ST2,AMU)/T_HDEC(STOP1,STOP2,AMU)
c      write(6,*)'A_t: ',at,atp,atp/at
c      write(6,*)'als: ',ash/pi
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c      write(6,*)SBOT1,SBOT2,AMG,AMU,TANB,ASH,
c    .           2*ASH/3/PI*AMG*AMU*TANB*T_HDEC(SBOT1,SBOT2,AMG),
c    .           2*ASH/3/PI*AMG*AMU*TANB*T_HDEC(SBOT1,SBOT2,AMG)*FQCD,
c    .           2*ASH/3/PI*AMG*AMU*TANB,T_HDEC(SBOT1,SBOT2,AMG)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c      write(6,*)'deltamb, qcd, elw: ',deltamb,fqcd,felw
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c      sbot1 = 600.733d0
c      sbot2 = 960.891d0
c      amsb1 = sbot1
c      amsb2 = sbot2
c      sc1 = 0.1d0
c      sc2 = 10
c      nsc = 101
c      open(77,file='fort.77')
c      open(78,file='fort.78')
c      do i = 1,nsc
c       scfac = sc1*(sc2/sc1)**((i-1)/(nsc-1.d0))
c       scalelw = scfac*(amst1+amst2+amu)/3
c       scalqcd = scfac*(amsb1+amsb2+amg)/3

c       ash = alphas_hdec(scalqcd,3)
c       dmbqcd = 2*ash/3/pi*amg*amu*tanb*t_hdec(sbot1,sbot2,amg)
c    *         /(1-2*ash/3/pi*amg*ab*t_hdec(sbot1,sbot2,amg))
c      xxt = au - amu/tanb
c       cqcd = fqcd_hdec(scalqcd,amt,amg,sbot1,sbot2,stop1,stop2,
c    .                   amsu(1),amsu(2),amsd(1),amsd(2),xxt)
c       fqcd = 1+ash/pi*cqcd
c       write(6,*)scfac,scalqcd,ash
c       write(6,*)sbot1,sbot2,stop1,stop2
c       write(6,*)amsu,amsd
c       write(6,*)
c       dmbqcd1 = dmbqcd*fqcd
c       rmtop   = runm_hdec(scalelw,6)
c       ht = rmtop/v/sb
c       dmbelw = ht**2/(4*pi)**2*at*amu*tanb*t_hdec(stop1,stop2,amu)
c       ash = alphas_hdec(scalelw,3)
c       celw = felw_hdec(scalelw,amu,amg,sbot1,sbot2,stop1,stop2,amt)
c       felw = 1+ash/pi*celw
c       dmbelw1 = dmbelw*felw
c       write(77,*)scfac,dmbqcd,dmbqcd1
c       write(78,*)scfac,dmbelw,dmbelw1
c      enddo
c      close(77)
c      close(78)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      ELSE
       DELTAMB = 2*ASH/3/PI*AMG*AMU*TANB*T_HDEC(SBOT1,SBOT2,AMG)
       DGLB = -DELTAMB*(1+1/TANA/TANB)
       DGHB = -DELTAMB*(1-TANA/TANB)
       DGAB = -DELTAMB*(1+1/TANB**2)
c      write(6,*)'sub: ',DELTAMB,DGLB,DGHB,DGAB
c      write(6,*)'sub: ',ASH,AMG,AMU,TANB,T_HDEC(SBOT1,SBOT2,AMG)
c      write(6,*)'sub: ',SBOT1,SBOT2,AMG
      ENDIF
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     DGLB = 0
c     DGHB = 0
c     DGAB = 0
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     write(6,*)'delta_b: ',deltamb, ash

      RETURN
      END
 
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      SUBROUTINE STAUSUSY_HDEC(GLB,GHB,GAB,XGLB,XGHB,XGAB,SCALE,IL)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      ICASE = 0
      CALL DMTAUAPP_HDEC(ICASE,DGLB,DGHB,DGAB,SCALE,IL)
      XGLB = GLB*(1+DGLB)
      XGHB = GHB*(1+DGHB)
      XGAB = GAB*(1+DGAB)
      RETURN
      END
 
      SUBROUTINE DMTAUAPP_HDEC(ICASE,DGLB,DGHB,DGAB,SCALE,IL)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION AMCHAR(2),AMNEUT(4),XMNEUT(4),
     .          XMST(2),XMSB(2),AMSL(2),
     .          AMSU(2),AMSD(2),AMSE(2),AMSN(2),AMSN1(2)
      COMMON/PARAM_HDEC/GF,ALPH,AMTAU,AMMUON,AMZ,AMW
      COMMON/MASSES_HDEC/AMS,AMC,AMB,AMT
      COMMON/HMASS_HDEC/AMSM,AMA,AMHL,AMHH,AMCH,AMAR
      COMMON/GLUINO_HDEC/AMG,AMSB1,AMSB2,STH0,CTH0,
     .              GLBB(2,2),GHBB(2,2),GABB(2,2),
     .              AMST1,AMST2,STHT0,CTHT0,
     .              GLTT(2,2),GHTT(2,2),GATT(2,2)
      COMMON/COUP_HDEC/GAT,GAB,GLT,GLB,GHT,GHB,GZAH,GZAL,
     .            GHHH,GLLL,GHLL,GLHH,GHAA,GLAA,GLVV,GHVV,
     .            GLPM,GHPM,B,A
      COMMON/BREAK_HDEC/AMEL,AMER,AMSQ,AMUR,AMDR,AL,AU,AD,AMU,AM2
      COMMON/SMASS_HDEC/AMNEUT,XMNEUT,AMCHAR,XMST,XMSB,AMSL,
     .              AMSU,AMSD,AMSE,AMSN,AMSN1
      COMMON/TAUMIX_HDEC/CL,SL
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      COMMON/DAVID/QSUSY1,QSUSY2,LOOP
      COMMON/ALSLAM_HDEC/XLB1(6),XLB2(6),XLB3(6)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      PI = 4*DATAN(1.D0)
      V  = 1/DSQRT(2*DSQRT(2D0)*GF)
      TANB = DTAN(B)
      TANA = DTAN(A)
      SB = TANB/DSQRT(1+TANB**2)
      AT = AU
      AB = AD
      IF(IL.EQ.0)THEN
       STAU1 = AMSE(1)
       STAU2 = AMSE(2)
       SNEUT = AMSN1(1)
       CTH = 1
       STH = 0
      ELSE
       STAU1 = AMSL(1)
       STAU2 = AMSL(2)
       SNEUT = AMSN(1)
       CTH = CL
       STH = SL
      ENDIF
      CW = AMW**2/AMZ**2
      SW = 1-CW
      AL2  = (2*AMW/V/DSQRT(2D0))**2/4/PI
      AL1  = AL2*SW/CW
      AM1=5.D0/3.D0*SW/CW*AM2

      IF(ICASE.EQ.0)THEN
       DELTAMB = AL1/4/PI*AM1*AMU*TANB*(T_HDEC(STAU1,STAU2,AM1)
     . + (CTH**2/2-STH**2)*T_HDEC(STAU1,AM1,AMU)
     . + (STH**2/2-CTH**2)*T_HDEC(STAU2,AM1,AMU))
     .         - AL2/4/PI*AM2*AMU*TANB*(T_HDEC(SNEUT,AM2,AMU)
     . +(CTH**2*T_HDEC(STAU1,AM2,AMU)+STH**2*T_HDEC(STAU2,AM2,AMU))/2)
       DGLB = -DELTAMB/(1+DELTAMB)*(1+1/TANA/TANB)
       DGHB = -DELTAMB/(1+DELTAMB)*(1-TANA/TANB)
       DGAB = -DELTAMB/(1+DELTAMB)*(1+1/TANB**2)
c      write(6,*)'Delta_tau: ',DELTAMB
      ELSE
       DELTAMB = AL1/4/PI*AM1*AMU*TANB*T_HDEC(STAU1,STAU2,AM1)
     .         - AL2/4/PI*AM2*AMU*TANB*T_HDEC(SNEUT,AM2,AMU)
       DGLB = -DELTAMB*(1+1/TANA/TANB)
       DGHB = -DELTAMB*(1-TANA/TANB)
       DGAB = -DELTAMB*(1+1/TANB**2)
      ENDIF
      RETURN
      END
 
      SUBROUTINE STRSUSY_HDEC(GLB,GHB,GAB,XGLB,XGHB,XGAB,SCALE,IL)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      ICASE = 0
      CALL DMSAPP_HDEC(ICASE,DGLB,DGHB,DGAB,SCALE,IL)
      XGLB = GLB*(1+DGLB)
      XGHB = GHB*(1+DGHB)
      XGAB = GAB*(1+DGAB)
      RETURN
      END
 
      SUBROUTINE DMSAPP_HDEC(ICASE,DGLB,DGHB,DGAB,SCALE,IL)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION AMCHAR(2),AMNEUT(4),XMNEUT(4),
     .          XMST(2),XMSB(2),AMSL(2),
     .          AMSU(2),AMSD(2),AMSE(2),AMSN(2),AMSN1(2)
      COMMON/PARAM_HDEC/GF,ALPH,AMTAU,AMMUON,AMZ,AMW
      COMMON/MASSES_HDEC/AMS,AMC,AMB,AMT
      COMMON/HMASS_HDEC/AMSM,AMA,AMHL,AMHH,AMCH,AMAR
      COMMON/GLUINO_HDEC/AMG,AMSB1,AMSB2,STH0,CTH0,
     .              GLBB(2,2),GHBB(2,2),GABB(2,2),
     .              AMST1,AMST2,STHT0,CTHT0,
     .              GLTT(2,2),GHTT(2,2),GATT(2,2)
      COMMON/COUP_HDEC/GAT,GAB,GLT,GLB,GHT,GHB,GZAH,GZAL,
     .            GHHH,GLLL,GHLL,GLHH,GHAA,GLAA,GLVV,GHVV,
     .            GLPM,GHPM,B,A
      COMMON/BREAK_HDEC/AMEL,AMER,AMSQ,AMUR,AMDR,AL,AU,AD,AMU,AM2
      COMMON/SMASS_HDEC/AMNEUT,XMNEUT,AMCHAR,XMST,XMSB,AMSL,
     .              AMSU,AMSD,AMSE,AMSN,AMSN1
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      COMMON/DAVID/QSUSY1,QSUSY2,LOOP
      COMMON/ALSLAM_HDEC/XLB1(6),XLB2(6),XLB3(6)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      PI = 4*DATAN(1.D0)
      V  = 1/DSQRT(2*DSQRT(2D0)*GF)
      TANB = DTAN(B)
      TANA = DTAN(A)
      SB = TANB/DSQRT(1+TANB**2)
      AT = AU
      AB = AD
      SBOT1 = AMSD(1)
      SBOT2 = AMSD(2)
      STOP1 = AMSU(1)
      STOP2 = AMSU(2)
      CTHT = 1
      STHT = 0
      CTH  = 1
      STH  = 0
      CW = AMW**2/AMZ**2
      SW = 1-CW
      AL2  = (2*AMW/V/DSQRT(2D0))**2/4/PI
      AL1  = AL2*SW/CW
      AM1=5.D0/3.D0*SW/CW*AM2

      FELW = 1
      FQCD = 1
      SCALQCD = (SBOT1+SBOT2+DABS(AMG))/3
      ASH = ALPHAS_HDEC(SCALQCD,3)
      IF(ICASE.EQ.0)THEN
c      DELTAMB = 2*ASH/3/PI*AMG*AMU*TANB*T_HDEC(SBOT1,SBOT2,AMG)*FQCD
c    *         /(1-2*ASH/3/PI*AMG*AB*T_HDEC(SBOT1,SBOT2,AMG))
       DELTAMB = (2*ASH/3/PI*AMG*AMU*TANB*T_HDEC(SBOT1,SBOT2,AMG)*FQCD
     . - AL1/12/PI*AM1*AMU*TANB*(T_HDEC(SBOT1,SBOT2,AM1)/3
     .      + (CTH**2/2+STH**2)*T_HDEC(SBOT1,AM1,AMU)
     .      + (STH**2/2+CTH**2)*T_HDEC(SBOT2,AM1,AMU))
     . - AL2/4/PI*AM2*AMU*TANB*(CTHT**2*T_HDEC(STOP1,AM2,AMU)
     .                         +STHT**2*T_HDEC(STOP2,AM2,AMU)
     . +(CTH**2*T_HDEC(SBOT1,AM2,AMU)+STH**2*T_HDEC(SBOT2,AM2,AMU))/2))
     *         /(1-2*ASH/3/PI*AMG*AB*T_HDEC(SBOT1,SBOT2,AMG))
       DGLB = -DELTAMB/(1+DELTAMB)*(1+1/TANA/TANB)
       DGHB = -DELTAMB/(1+DELTAMB)*(1-TANA/TANB)
       DGAB = -DELTAMB/(1+DELTAMB)*(1+1/TANB**2)
c      write(6,*)'Delta_s: ',DELTAMB
c    .          ,2*ASH/3/PI*AMG*AMU*TANB*T_HDEC(SBOT1,SBOT2,AMG)*FQCD
      ELSE
       DELTAMB = 2*ASH/3/PI*AMG*AMU*TANB*T_HDEC(SBOT1,SBOT2,AMG)
       DGLB = -DELTAMB*(1+1/TANA/TANB)
       DGHB = -DELTAMB*(1-TANA/TANB)
       DGAB = -DELTAMB*(1+1/TANB**2)
      ENDIF
      RETURN
      END
 
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

***********************************************************************
        FUNCTION ETA_HDEC(C1,C2)
***********************************************************************
*       COMPLEX ETA-FUNKTION                                           
*---------------------------------------------------------------------*
*       8.06.90    ANSGAR DENNER                                       
***********************************************************************
        IMPLICIT   LOGICAL(A-Z)                                        
        COMPLEX*16 ETA_HDEC,C1,C2
        REAL*8     PI,IM1,IM2,IM12                                     
                                                                       
        PI     = 4D0*DATAN(1D0)                                        
        IM1    = DIMAG(C1)                                             
        IM2    = DIMAG(C2)                                             
        IM12   = DIMAG(C1*C2)                                          
                                                                       
        IF(IM1.LT.0D0.AND.IM2.LT.0D0.AND.IM12.GT.0D0) THEN             
            ETA_HDEC = DCMPLX(0D0,2D0*PI)
        ELSE IF (IM1.GT.0D0.AND.IM2.GT.0D0.AND.IM12.LT.0D0) THEN       
            ETA_HDEC = DCMPLX(0D0,-2D0*PI)
        ELSE                                                           
            ETA_HDEC = DCMPLX(0D0)
        END IF                                                         
        END                                                            

***********************************************************************
        FUNCTION ETAS_HDEC(Y,R,RS)
***********************************************************************
*       MODIFIED ETA-FUNKTION                                           
*---------------------------------------------------------------------*
*       18.1.94   SD                                       
***********************************************************************
        IMPLICIT   LOGICAL(A-Z)                                        
        COMPLEX*16 ETA_HDEC,ETAS_HDEC,Y,R,RS
        REAL*8     PI,IMY,IMRS
                                                                       
        PI     = 4D0*DATAN(1D0)                                        

	IF( DIMAG(R).NE.0D0 ) THEN
	    ETAS_HDEC = ETA_HDEC(Y,R)
	ELSE	    
	    IF( DREAL(R).GT.0D0 ) THEN
		ETAS_HDEC = DCMPLX(0D0,0D0)
	    ELSE
	 	IMY  = DIMAG(Y)
		IMRS = DIMAG(RS)
		ETAS_HDEC = 2D0*DCMPLX(0D0,PI)*(
     *			(1D0+SIGN(1D0,-IMY))*(1D0+SIGN(1D0,-IMRS))-
     *			(1D0+SIGN(1D0, IMY))*(1D0+SIGN(1D0, IMRS))
     *					  )/4D0
	    ENDIF
	ENDIF
        END                                                            

***********************************************************************
        FUNCTION SQE_HDEC(A,B,C)
***********************************************************************
*       SOLUTION OF QUADRATIC EQUATION				      *
*---------------------------------------------------------------------*
*       13.1.92  SD						      *
***********************************************************************
        IMPLICIT REAL*8 (A-Z)                                        
        COMPLEX*16 A,B,C,SQE_HDEC,X1,X2

	X1=(-B+SQRT(B**2-4D0*A*C))/2D0/A
	X2=(-B-SQRT(B**2-4D0*A*C))/2D0/A

	IF (ABS(X1).GT.ABS(X2)) THEN
	   SQE_HDEC=X1
	ELSE
	   SQE_HDEC=X2
	ENDIF

        END                                                            

************************************************************************
        FUNCTION D04_HDEC(P1,P2,P3,P4,P12,P23,M1,M2,M3,M4)
************************************************************************
*  SCALAR 4-POINT FUNCTION WITH AT LEAST ONE MASS ZERO                 *
*  P1,P2,P3,P4 = SQUARED EXTERNAL MOMENTA			       *
*  P12 = (p1+p2)**2,  P23 = (p2+p3)**2				       *
*----------------------------------------------------------------------*
*  2.1.92  SD	         					       *
************************************************************************
        IMPLICIT REAL*8 (A-Z)
	REAL*8 M(4),P(4,4),K(4,4)
	COMPLEX*16 A1,A2,A3,A4,SWAP
	COMPLEX*16 SS(4), XX(2), X(2,4),RS(4,4)
	COMPLEX*16 S0(4),XX0(2),X0(2,4), R(4,4),G(2)
        COMPLEX*16 C04,D04_HDEC,CSPEN_HDEC,ETA_HDEC,SQE_HDEC,ETAS_HDEC
	COMPLEX*16 AA,BB,CC,DD,IEPS,H,HH,L1,L2,L3,L4
        COMPLEX*16 Z2,B,SC,TC,WP,WM,BS,XS
	INTEGER GEN,I,J

        PI = 4*DATAN(1.D0)

        MM1=M1
        MM2=M2
        MM3=M3
        MM4=M4
        M12=M1*M1
        M22=M2*M2
        M32=M3*M3
        M42=M4*M4
        Q1=P1
        Q2=P2
        Q3=P3
	Q4=P4
        Q12=P12
        Q23=P23

C	IS AT LEAST ONE MASS ZERO ???
	IF (MM1*MM2*MM3*MM4.NE.0D0) GOTO 130

C	PERMUTATE UNTIL MM3=0D0
	GOTO 20
10	CONTINUE
	MM0=MM1
	MM1=MM2
	MM2=MM3
	MM3=MM4
	MM4=MM0
	M02=M12
	M12=M22
	M22=M32
	M32=M42
	M42=M02
	Q00=Q12
	Q12=Q23
	Q23=Q00
	Q0=Q1
	Q1=Q2
	Q2=Q3
	Q3=Q4
	Q4=Q0
20	IF (MM3.NE.0D0) GOTO 10
C	ONLY MM3 IS ZERO
	IF (MM1*MM2*MM4.NE.0D0) GOTO 30
C	ONLY MM3 AND MM4 ARE ZERO ==> 3->2, 4->3...
	IF ((MM1*MM2.NE.0D0).AND.(MM4.EQ.0D0)) GOTO 10
C	ONLY MM2 AND MM3 ARE ZERO
	IF ((MM1*MM4.NE.0D0).AND.(MM2.EQ.0D0)) GOTO 40
	WRITE(*,*)'CASE OF THIS SPECIAL D0-FUNCTION NOT IMPLEMENTED!'
	STOP

C	****** NO MASS EQUAL TO ZERO ******
130	CONTINUE
	EPS=1D-18
	IEPS=DCMPLX(0D0,EPS)

	IF( ABS((MM1**2+MM3**2-Q12)/MM1/MM3).LT.2D0 ) THEN
C	R13 WOULD BE NOT REAL. -> PERMUTATION! -> R(2,4) IS NOT REAL.
	   M(1)=MM2
	   M(2)=MM3
	   M(3)=MM4
	   M(4)=MM1
	   P(1,2)=Q2
	   P(1,3)=Q23
	   P(1,4)=Q1
	   P(2,3)=Q3
	   P(2,4)=Q12
	   P(3,4)=Q4
	ELSE
C	R(1,3) IS REAL.
	   M(1)=MM1
	   M(2)=MM2
	   M(3)=MM3
	   M(4)=MM4
	   P(1,2)=Q1
	   P(1,3)=Q12
	   P(1,4)=Q4
	   P(2,3)=Q2
	   P(2,4)=Q23
	   P(3,4)=Q3
	ENDIF

	DO 11 J=2,4
	DO 11 I=1,J-1
	K(I,J)=(M(I)**2+M(J)**2-P(I,J))/M(I)/M(J)
	R(I,J) =SQE_HDEC(DCMPLX(1D0,0D0),DCMPLX(-K(I,J),0D0),
     *	            DCMPLX(1D0,0D0))
	IF( DIMAG(R(I,J)).EQ.0D0 ) THEN
	   RS(I,J)=SQE_HDEC(DCMPLX(1D0,0D0),DCMPLX(-K(I,J),EPS),
     *	               DCMPLX(1D0,0D0))
	ELSE
	   RS(I,J)=R(I,J)
	ENDIF
11	CONTINUE

	SS(1)=RS(1,2)
	SS(2)=RS(2,3)
	SS(3)=RS(3,4)
	SS(4)=RS(1,4)
	S0(1)=R(1,2)
	S0(2)=R(2,3)
	S0(3)=R(3,4)
	S0(4)=R(1,4)
	AA=K(3,4)/R(2,4)+R(1,3)*K(1,2)-K(1,4)*R(1,3)/R(2,4)-K(2,3)
	BB=(R(2,4)-1D0/R(2,4))*(R(1,3)-1D0/R(1,3))
     *		+K(1,2)*K(3,4)-K(1,4)*K(2,3)
	CC=K(1,2)/R(1,3)+R(2,4)*K(3,4)-K(1,4)*R(2,4)/R(1,3)-K(2,3)
	DD=K(2,3)-R(1,3)*K(1,2)-R(2,4)*K(3,4)+R(1,3)*R(2,4)*K(1,4)
	XX(1)=SQE_HDEC(AA,BB,CC+IEPS*DD)
	XX(2)=(CC+IEPS*DD)/AA/XX(1)
	XX0(1)=SQE_HDEC(AA,BB,CC)
	XX0(2)=CC/AA/XX0(1)
c	IF (ABS(DREAL(XX0(1)-XX(2))).LT.ABS(DREAL(XX0(1)-XX(1)))) THEN
	IF (ABS(XX0(1)-XX(2)).LT.ABS(XX0(1)-XX(1))) THEN
	  SWAP  =XX0(1)
	  XX0(1)=XX0(2)
	  XX0(2)=SWAP
	ENDIF

	DO 12 I=1,2
	G(I)  =SIGN( 1D0,DREAL(AA*(XX(I)-XX(3-I))) )
	 X(I,1)= XX(I)/R(2,4)
	X0(I,1)=XX0(I)/R(2,4)
	 X(I,2)= XX(I)/R(2,4)*R(1,3)
	X0(I,2)=XX0(I)/R(2,4)*R(1,3)
	 X(I,3)= XX(I)*R(1,3)
	X0(I,3)=XX0(I)*R(1,3)
	 X(I,4)= XX(I)
	X0(I,4)=XX0(I)
12	CONTINUE

	D04_HDEC = DCMPLX(0D0,0D0)
	DO 13 I=1,2
	DO 13 J=1,4
	A1 = 1D0+X0(I,J)*S0(J) + ABS(1D0+X0(I,J)*S0(J))*IEPS*
     *				  SIGN(1D0,DIMAG(X(I,J)*SS(J)))
	A2 = 1D0+X0(I,J)/S0(J) + ABS(1D0+X0(I,J)/S0(J))*IEPS*
     *				  SIGN(1D0,DIMAG(X(I,J)/SS(J)))
	D04_HDEC = D04_HDEC + (-1D0)**(I+J)*(
     *		CSPEN_HDEC(A1)+ETA_HDEC(-X(I,J),SS(J))*LOG(A1)
     *	       +CSPEN_HDEC(A2)+ETA_HDEC(-X(I,J),1D0/SS(J))*LOG(A2))
13	CONTINUE

	IF( DIMAG(R(1,3)).EQ.0D0 ) THEN
	DO 14 I=1,2
	   A1 = (K(1,3)-2D0*R(1,3))/XX0(I)
     *		      -R(1,3)*K(1,4)+K(3,4)
     	   A2 = ((K(2,4)-2D0*R(2,4))*R(1,3)*XX0(I)
     *		      -R(2,4)*K(3,4)+K(2,3))/DD
	   A3 = (K(1,3)-2D0*R(1,3))*R(2,4)/XX0(I)
     *		      -R(1,3)*K(1,2)+K(2,3)
	   A4 = ((K(2,4)-2D0*R(2,4))*XX0(I)
     *		      -R(2,4)*K(1,4)+K(1,2))/DD
	   L1 = LOG( A1-ABS(A1)*IEPS )
     	   L2 = LOG( A2+ABS(A2)*IEPS*G(I)*SIGN(1D0,DREAL(R(1,3))
     *				        	  *DIMAG(RS(2,4))) )
	   L3 = LOG( A3-ABS(A3)*IEPS )
	   L4 = LOG( A4+ABS(A4)*IEPS*G(I)*SIGN(1D0,DIMAG(RS(2,4))) )

	   D04_HDEC = D04_HDEC + (3D0-2D0*I)*(
     *		 ETAS_HDEC(-XX(I),R(1,3),RS(1,3))
     *		   *( LOG(R(1,3)*XX(I)) + L1 + L2 )
     *		+ETAS_HDEC(-XX(I),1D0/R(2,4),1D0/RS(2,4))
     *		   *( LOG(XX(I)/R(2,4)) + L3 + L4 )
     *		-( ETAS_HDEC(-XX(I),R(1,3)/R(2,4),RS(1,3)/RS(2,4))
     *		  +ETA_HDEC(RS(1,3),1D0/RS(2,4)) )
     *		   *( LOG(XX(I)*R(1,3)/R(2,4)) + L3 + L2 )
     *	  	+ETA_HDEC(RS(1,3),1D0/RS(2,4))
     *		  *ETAS_HDEC(-XX(I),-R(1,3)/R(2,4),-RS(1,3)/RS(2,4))   )
14	CONTINUE
	ELSE
	DO 15 I=1,2
	   L1 = LOG( R(2,4)/XX0(I)+XX0(I)/R(2,4)+K(1,2)
     *		     -XX0(I)/R(2,4)*EPS*BB*G(I) )
	   L2 = LOG( R(1,3)*XX0(I)+1D0/XX0(I)/R(1,3)+K(3,4)
     *		     -XX0(I)*R(1,3)*EPS*BB*G(I) )
	   L3 = LOG( R(1,3)/R(2,4)*XX0(I)+R(2,4)/XX0(I)/R(1,3)+K(2,3)
     *		     -XX0(I)*R(1,3)/R(2,4)*EPS*BB*G(I) )

	   D04_HDEC = D04_HDEC + (3D0-2D0*I)*(
     *		+ETA_HDEC(-XX(I),1D0/R(2,4))
     *		   *( LOG(XX(I)/R(2,4)) + L1 )
     *		+ETA_HDEC(-XX(I),R(1,3))
     *		   *( LOG(R(1,3)*XX(I)) + L2 )
     *		-( ETA_HDEC(-XX(I),R(1,3)/R(2,4))
     *		  +ETA_HDEC(R(1,3),1D0/R(2,4)) )
     *		   *( LOG(XX(I)*R(1,3)/R(2,4)) + L3 )
     *	  	+ETA_HDEC(R(1,3),1D0/R(2,4))
     *		   *ETA_HDEC(-XX(I),-R(1,3)/R(2,4))
     *		   *(1D0-G(I)*SIGN(1D0,DREAL(BB)))	    )
15	CONTINUE
	ENDIF

	D04_HDEC = D04_HDEC/M(1)/M(2)/M(3)/M(4)/AA/(XX(1)-XX(2))
	RETURN


C--->	***************** SPEZIELL ( --> T.SACK-PROMOTION )
C	D1=Q12-M12
C	D2=Q2 -M22
C	D3=Q3 -M42
C	IF ((D1*D2.LE.0D0).OR.(D2*D3.LE.0D0)) THEN
C	   WRITE(*,*) 'THE CASE OF DIFFERENT SIGNS OF THE D1,D2,D3'
C	   WRITE(*,*) 'IN D04(...) IS NOT IMPLEMENTED !!!'
C	   STOP
C	ENDIF
C	NM1=ABS(MM1/D1)
C	NM2=ABS(MM2/D2)
C	NM3=ABS(MM4/D3)
C	NP1=Q2/D2**2+Q12/D1**2+(Q1-Q2-Q12)/D1/D2
C	NP2=Q2/D2**2+ Q3/D3**2+(Q23-Q2-Q3)/D2/D3
C	NP3=Q3/D3**2+Q12/D1**2+(Q4-Q3-Q12)/D1/D3
C	D04_HDEC=C04(NP1,NP2,NP3,NM1,NM2,NM3)/D1/D2/D3

C	*************** ALLGEMEIN


C	****** ONLY MM3 IS ZERO ******
30	CONTINUE
	EPS=1D-17
	IEPS=DCMPLX(0D0,EPS)
	M(1)=MM1
	M(2)=MM2
	M(3)=10D0
	M(4)=MM4
	P(1,2)=Q1
	P(1,3)=Q12
	P(1,4)=Q4
	P(2,3)=Q2
	P(2,4)=Q23
	P(3,4)=Q3
	DO 1 J=2,4
	DO 1 I=1,J-1
	K(I,J)=(M(I)**2+M(J)**2-P(I,J))/M(I)/M(J)
	IF (I.EQ.3) K(I,J)=K(I,J)-M(I)/M(J)
	IF (J.EQ.3) K(I,J)=K(I,J)-M(J)/M(I)
	R(I,J) =SQE_HDEC(DCMPLX(1D0,0D0),DCMPLX(-K(I,J),0D0),
     *	            DCMPLX(1D0,0D0))
	IF( DIMAG(R(I,J)).EQ.0D0 ) THEN
	   RS(I,J)=SQE_HDEC(DCMPLX(1D0,0D0),DCMPLX(-K(I,J),EPS),
     *	               DCMPLX(1D0,0D0))
	ELSE
	   RS(I,J)=R(I,J)
	ENDIF
1	CONTINUE
	SS(1)=RS(1,2)
	SS(2)=RS(2,3)
	SS(3)=RS(3,4)
	SS(4)=RS(1,4)
	AA=K(3,4)/R(2,4)-K(2,3)
	BB=K(1,3)*(1D0/R(2,4)-R(2,4))+K(1,2)*K(3,4)-K(1,4)*K(2,3)
	CC=K(1,2)*K(1,3)-K(1,3)*K(1,4)*R(2,4)+R(2,4)*K(3,4)-K(2,3)
	DD=K(2,3)-R(2,4)*K(3,4)
	XX(1)=SQE_HDEC(AA,BB,CC+IEPS*DD)
	XX(2)=(CC+IEPS*DD)/AA/XX(1)
	DO 2 I=1,2
	X(I,1)=XX(I)/R(2,4)
	X(I,2)=XX(I)/R(2,4)*R(1,3)
	X(I,3)=XX(I)*R(1,3)
	X(I,4)=XX(I)
2	CONTINUE
	D04_HDEC = DCMPLX(0D0,0D0)
	DO 3 I=1,2
	D04_HDEC = D04_HDEC + (2D0*I-3D0)*(
     *		CSPEN_HDEC(1D0+SS(4)*X(I,4))
     *	       -CSPEN_HDEC(1D0+SS(1)*X(I,1))
     *	       +CSPEN_HDEC(1D0+X(I,4)/SS(4))
     *	       -CSPEN_HDEC(1D0+X(I,1)/SS(1))
     *	       +ETA_HDEC(-X(I,4),SS(4))*LOG(1D0+SS(4)*X(I,4))
     *	       -ETA_HDEC(-X(I,1),SS(1))*LOG(1D0+SS(1)*X(I,1))
     *	       +ETA_HDEC(-X(I,4),1D0/SS(4))*LOG(1D0+X(I,4)/SS(4))
     *	       -ETA_HDEC(-X(I,1),1D0/SS(1))*LOG(1D0+X(I,1)/SS(1))
     *	       -CSPEN_HDEC(1D0+X(I,4)*(K(3,4)-IEPS)/(K(1,3)-IEPS))
     *	       +CSPEN_HDEC(1D0+X(I,1)*(K(2,3)-IEPS)/(K(1,3)-IEPS))
     *	       -ETA_HDEC(-X(I,4),(K(3,4)-IEPS)/(K(1,3)-IEPS))
     *	           *LOG(1D0+X(I,4)*(K(3,4)-IEPS)/(K(1,3)-IEPS))
     *	       +ETA_HDEC(-X(I,1),(K(2,3)-IEPS)/(K(1,3)-IEPS))
     *	           *LOG(1D0+X(I,1)*(K(2,3)-IEPS)/(K(1,3)-IEPS))   )
	IF (DIMAG(R(2,4)).NE.0D0) THEN
	   H=ETA_HDEC(-1D0/XX(I),R(2,4))
	ELSE
	   H=DCMPLX(0D0,0D0)
	   IF (DREAL(R(2,4)).LT.0D0) THEN
	      HH=-1D0/XX(I)
	      IM1=DIMAG(HH)
	      IM2=DIMAG(RS(2,4))
	      IF ((IM1.GT.0D0).AND.(IM2.GT.0D0)) THEN
	         H=-DCMPLX(0D0,2D0*PI)
	      ENDIF
	      IF ((IM1.LT.0D0).AND.(IM2.LT.0D0)) THEN
	         H=+DCMPLX(0D0,2D0*PI)
	      ENDIF
	   ENDIF
	ENDIF
	D04_HDEC = D04_HDEC + (2D0*I-3D0)*
     *	          H*( LOG( (K(1,2)-R(2,4)*K(1,4)
     *			  +XX(I)*(1D0/R(2,4)-R(2,4)))/DD )
     *		     +LOG(K(1,3)-IEPS) )
3	CONTINUE
	D04_HDEC = D04_HDEC/M(1)/M(2)/M(3)/M(4)/AA/(XX(1)-XX(2))
	RETURN

C	****** ONLY MM2 AND MM3 ARE ZERO ******
40	CONTINUE
	EPS=1D-17
	IEPS=DCMPLX(0D0,EPS)

	M(1)=MM1
	M(2)=10D0
	M(3)=10D0
	M(4)=MM4
	P(1,2)=Q1
	P(1,3)=Q12
	P(1,4)=Q4
	P(2,3)=Q2
	P(2,4)=Q23
	P(3,4)=Q3
	DO 4 J=2,4
	DO 4 I=1,J-1
	K(I,J)=(M(I)**2+M(J)**2-P(I,J))/M(I)/M(J)
	IF (I.EQ.2) K(I,J)=K(I,J)-M(I)/M(J)
	IF (J.EQ.2) K(I,J)=K(I,J)-M(J)/M(I)
	IF (I.EQ.3) K(I,J)=K(I,J)-M(I)/M(J)
	IF (J.EQ.3) K(I,J)=K(I,J)-M(J)/M(I)
	R(I,J) =SQE_HDEC(DCMPLX(1D0,0D0),DCMPLX(-K(I,J),0D0),
     *	            DCMPLX(1D0,0D0))
	IF( DIMAG(R(I,J)).EQ.0D0 ) THEN
	   RS(I,J)=SQE_HDEC(DCMPLX(1D0,0D0),DCMPLX(-K(I,J),EPS),
     *	               DCMPLX(1D0,0D0))
	ELSE
	   RS(I,J)=R(I,J)
	ENDIF
4	CONTINUE
	SS(1)=RS(1,2)
	SS(2)=RS(2,3)
	SS(3)=RS(3,4)
	SS(4)=RS(1,4)
	AA=K(2,4)*K(3,4)-K(2,3)
	BB=K(1,3)*K(2,4)+K(1,2)*K(3,4)-K(1,4)*K(2,3)
	CC=K(1,2)*K(1,3)-K(2,3)
	DD=K(2,3)
	XX(1)=SQE_HDEC(AA,BB,CC+IEPS*DD)
	XX(2)=(CC+IEPS*DD)/AA/XX(1)
	DO 5 I=1,2
	X(I,1)=XX(I)/R(2,4)
	X(I,2)=XX(I)/R(2,4)*R(1,3)
	X(I,3)=XX(I)*R(1,3)
	X(I,4)=XX(I)
5	CONTINUE
	D04_HDEC = DCMPLX(0D0,0D0)
	DO 6 I=1,2
	D04_HDEC = D04_HDEC + (2D0*I-3D0)*(
     *		CSPEN_HDEC(1D0+SS(4)*X(I,4))
     *	       +CSPEN_HDEC(1D0+X(I,4)/SS(4))
     *	       +ETA_HDEC(-X(I,4),SS(4))*LOG(1D0+SS(4)*X(I,4))
     *	       +ETA_HDEC(-X(I,4),1D0/SS(4))*LOG(1D0+X(I,4)/SS(4))
     *	       -CSPEN_HDEC(1D0+XX(I)*(K(3,4)-IEPS)/(K(1,3)-IEPS))
     *	       -CSPEN_HDEC(1D0+XX(I)*(K(2,4)-IEPS)/(K(1,2)-IEPS))
     *	       -ETA_HDEC(-XX(I),(K(3,4)-IEPS)/(K(1,3)-IEPS))
     *	           *LOG(1D0+XX(I)*(K(3,4)-IEPS)/(K(1,3)-IEPS))
     *	       -ETA_HDEC(-XX(I),(K(2,4)-IEPS)/(K(1,2)-IEPS))
     *	           *LOG(1D0+XX(I)*(K(2,4)-IEPS)/(K(1,2)-IEPS))
     *	       +LOG(-XX(I))*( LOG(K(1,2)-IEPS)
     *			     +LOG(K(1,3)-IEPS)-LOG(K(2,3)-IEPS) ) )
6	CONTINUE
	D04_HDEC = D04_HDEC/M(1)/M(2)/M(3)/M(4)/AA/(XX(1)-XX(2))

	RETURN

	END

************************************************************************
        FUNCTION C03_HDEC(P1,P2,P3,M1,M2,M3)
************************************************************************
*  SCALAR 3-POINT FUNCTION                                             *
*  P1,P2,P3 = SQUARED EXTERNAL MOMENTA  			       *
*----------------------------------------------------------------------*
*  5.12.96  M. SPIRA    					       *
************************************************************************
      IMPLICIT REAL*8 (A-H,O-Z)
      REAL*8 M1,M2,M3
      REAL*8 R(0:2)
      COMPLEX*16 C03_HDEC,CSPEN_HDEC,ETA_HDEC,IEPS,IM
      COMPLEX*16 ALP(0:2),X(0:2,2),Y0(0:2),Y(0:2,2)
      COMPLEX*16 CDUM
C     REAL*8 KAPPA
      COMPLEX*16 KAPPA
C     KAPPA(A,B,C) = DSQRT(A**2+B**2+C**2-2*(A*B+A*C+B*C))
C     KAPPA(A,B,C) = DSQRT(DABS(A**2+B**2+C**2-2*(A*B+A*C+B*C)))
c     KAPPA(A,B,C) = CDSQRT(DCMPLX(A**2+B**2+C**2-2*(A*B+A*C+B*C)))
      KAPPA(A,B,C,D) = CDSQRT((A**2+B**2+C**2-2*(A*B+A*C+B*C))
     .               * (1+IEPS*D))
      EPS = 1.D-8*(P1+P2+P3)
      IM = DCMPLX(0.D0,1.D0)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      IEPS = DCMPLX(0.D0,1.D-17)
c     IEPS = DCMPLX(0.D0,1.D-20)
      PI = 4*DATAN(1.D0)
      XX = 0.D0
C     IF(P1.LT.0.D0.OR.P2.LT.0.D0.OR.P3.LT.0.D0) XX=1.D0
      IF(P1.NE.0.D0.OR.XX.NE.0.D0)THEN
       Q10 = P1
      ELSE
       Q10 = EPS
      ENDIF
      IF(P3.NE.0.D0.OR.XX.NE.0.D0)THEN
       Q20 = P3
      ELSE
       Q20 = EPS
      ENDIF
      IF(P2.NE.0.D0.OR.XX.NE.0.D0)THEN
       Q21 = P2
      ELSE
       Q21 = EPS
      ENDIF
      R(0) = P2
      R(1) = P3
      R(2) = P1
      SM0 = M1**2
      SM1 = M2**2
      SM2 = M3**2
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     ALPHA  = KAPPA(Q10,Q21,Q20)
c     ALP(0) = KAPPA(Q21,SM1,SM2)*(1+IEPS*Q21)
c     ALP(1) = KAPPA(Q20,SM2,SM0)*(1+IEPS*Q20)
c     ALP(2) = KAPPA(Q10,SM0,SM1)*(1+IEPS*Q10)
      ALPHA  = KAPPA(Q10,Q21,Q20,1.D0)
      ALP(0) = KAPPA(Q21,SM1,SM2,DSIGN(1.D0,Q21))
      ALP(1) = KAPPA(Q20,SM2,SM0,DSIGN(1.D0,Q20))
      ALP(2) = KAPPA(Q10,SM0,SM1,DSIGN(1.D0,Q10))
      X(0,1) = (Q21 - SM1 + SM2 + ALP(0))/2/Q21
      X(0,2) = (Q21 - SM1 + SM2 - ALP(0))/2/Q21
      X(1,1) = (Q20 - SM2 + SM0 + ALP(1))/2/Q20
      X(1,2) = (Q20 - SM2 + SM0 - ALP(1))/2/Q20
      X(2,1) = (Q10 - SM0 + SM1 + ALP(2))/2/Q10
      X(2,2) = (Q10 - SM0 + SM1 - ALP(2))/2/Q10
      Y0(0) = (Q21*(Q21-Q20-Q10+2*SM0-SM1-SM2) - (Q20-Q10)*(SM1-SM2)
     .      + ALPHA*(Q21-SM1+SM2))/2/ALPHA/Q21
      Y0(1) = (Q20*(Q20-Q10-Q21+2*SM1-SM2-SM0) - (Q10-Q21)*(SM2-SM0)
     .      + ALPHA*(Q20-SM2+SM0))/2/ALPHA/Q20
      Y0(2) = (Q10*(Q10-Q21-Q20+2*SM2-SM0-SM1) - (Q21-Q20)*(SM0-SM1)
     .      + ALPHA*(Q10-SM0+SM1))/2/ALPHA/Q10
      Y(0,1) = Y0(0) - X(0,1)
      Y(0,2) = Y0(0) - X(0,2)
      Y(1,1) = Y0(1) - X(1,1)
      Y(1,2) = Y0(1) - X(1,2)
      Y(2,1) = Y0(2) - X(2,1)
      Y(2,2) = Y0(2) - X(2,2)
      CDUM=0.D0
      DO I=0,2
       DO J=1,2
        CDUM = CDUM + CSPEN_HDEC((Y0(I)-1)/Y(I,J))
     .              - CSPEN_HDEC(Y0(I)/Y(I,J))
        CX = ETA_HDEC(1-X(I,J),1/Y(I,J))
        IF(CX.NE.0.D0)THEN
         CDUM = CDUM + CX*CDLOG((Y0(I)-1)/Y(I,J))
        ENDIF
        CY = ETA_HDEC(-X(I,J),1/Y(I,J))
        IF(CY.NE.0.D0)THEN
         CDUM = CDUM - CY*CDLOG(Y0(I)/Y(I,J))
        ENDIF
       ENDDO
       CX = ETA_HDEC(-X(I,1),-X(I,2))
       IF(CX.NE.0.D0)THEN
        CDUM = CDUM - CX*CDLOG((1-Y0(I))/(-Y0(I)))
       ENDIF
       CY = ETA_HDEC(Y(I,1),Y(I,2))
       IF(CY.NE.0.D0)THEN
        CDUM = CDUM + CY*CDLOG((1-Y0(I))/(-Y0(I)))
       ENDIF
       A = -R(I)
       B = -DIMAG(Y(I,1)*Y(I,2))
       IF(A.GT.0.D0.AND.B.GT.0.D0) THEN
        CDUM = CDUM + 2*PI*IM*CDLOG((1-Y0(I))/(-Y0(I)))
       ENDIF
      ENDDO
      C03_HDEC = CDUM/ALPHA
      RETURN
      END

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                                                                     C
C        SUBROUTINE CALCULATING THE FINITE REAL PART OF THE           C
C          GENERAL MASSIVE TWO POINT FUNCTION                         C
C                                                                     C
C           B02(P.P,M1,M2,MU**2)                                      C
C           BP02(P.P,M1,M2,MU**2)                                     C
C                                                                     C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

c ---------------------------------------------------------------------
      real*8 function B02_HDEC(s,m1,m2,mu2)

      implicit none

      real*8     s,m1,m2,mu2,m12,m22
      complex*16 zkappa,x1,x2

      m12 = m1**2
      m22 = m2**2

      zkappa=cdsqrt(dcmplx(s**2+m12**2+m22**2
     &                     -2.D0*(s*m12+s*m22+m12*m22)))

      if (s.eq.0.D0) then
         if (m12.eq.m22) then
            B02_HDEC=-dlog(m12/mu2)
         else
          if (m22.eq.0.D0) then
           if (m12.eq.0.D0) then
            B02_HDEC=1.D0
           else
            B02_HDEC=1.D0 - m12/(m12-m22)*dlog(m12/mu2)
           endif
          else
           if (m12.eq.0.D0) then
            B02_HDEC=1.D0 + m22/(m12-m22)*dlog(m22/mu2)
           else
            B02_HDEC=1.D0 - m12/(m12-m22)*dlog(m12/mu2)
     &                 + m22/(m12-m22)*dlog(m22/mu2)
           endif
          endif
         endif
      else
         if ((m12.eq.0.D0).and.(m22.eq.0.D0)) then
            B02_HDEC=2.D0 - dlog(s/mu2)
         elseif ((m12.eq.s).and.(m22.eq.0.D0)) then
            B02_HDEC=2.D0 - dlog(m12/mu2)
         elseif ((m22.eq.s).and.(m12.eq.0.D0)) then
            B02_HDEC=2.D0 - dlog(m22/mu2)
         elseif (m12.eq.0.D0) then
            B02_HDEC=2.D0 - (s-m22)/s*dlog( dabs(m22-s)/m22 )
     &                 - dlog(m22/mu2)
         elseif (m22.eq.0.D0) then
            B02_HDEC=2.D0 - (s-m12)/s*dlog( dabs(m12-s)/m12 )
     &                 - dlog(m12/mu2)
         else
            x1=dcmplx( (s-m22+m12+zkappa)/(2.D0*s) )
            x2=dcmplx( (s-m22+m12-zkappa)/(2.D0*s) )
            B02_HDEC=dreal( 2.D0+ dlog(mu2/m22) 
     &                       + x1*cdlog(1.D0-1.D0/x1) 
     &                       + x2*cdlog(1.D0-1.D0/x2))
         endif
      endif

      return
      end




c ---------------------------------------------------------------------
      real*8 function BP02_HDEC(s,m1,m2,mu2)

      implicit none

      real*8     s,m1,m2,mu2,m12,m22
      complex*16 zkappa,x1,x2

      m12 = m1**2
      m22 = m2**2

      zkappa=cdsqrt(dcmplx(s**2+m12**2+m22**2
     &                    -2.D0*(s*m12+s*m22+m12*m22)))

      if (s.eq.0.D0) then
         if (m12.eq.m22) then
            BP02_HDEC=1.D0/(6.D0*m12)
         elseif(m12.eq.0.d0) then
            BP02_HDEC=1/m22/2
         elseif(m22.eq.0.d0) then
            BP02_HDEC=1/m12/2
         else
            BP02_HDEC=( (m12+m22)/2.D0
     &        - m12*m22/(m12-m22)*dlog(m12/m22) )/(m12-m22)**2
         endif
      elseif ((s.eq.m12).and.(m22.eq.0.D0)) then
         BP02_HDEC=( -1.D0 + dlog(m12/mu2)/2.D0 )/m12
      elseif ((s.eq.m22).and.(m12.eq.0.D0)) then
         BP02_HDEC=( -1.D0 + dlog(m22/mu2)/2.D0 )/m22
      elseif (m22.eq.0.D0) then
         BP02_HDEC=( -1.D0 - m12/s*dlog((m12-s)/m12) )/s
      elseif (m12.eq.0.D0) then
         BP02_HDEC=( -1.D0 - m22/s*dlog((m22-s)/m22) )/s
      else
         x1=dcmplx( (s-m22+m12+zkappa)/(2.D0*s) )
         x2=dcmplx( (s-m22+m12-zkappa)/(2.D0*s) )
         BP02_HDEC=dreal( -1.D0 + ( x1*(1.D0-x1)*cdlog(1.D0-1.D0/x1)
     &                     - x2*(1.D0-x2)*cdlog(1.D0-1.D0/x2) )
     &                                                  /(x1-x2) )/s
      endif

      return
      end

C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
        FUNCTION CSPEN_HDEC(Z)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C       SPENCE-FUNKTION KOMPLEX, FREI NACH HOLLIK                     C
C---------------------------------------------------------------------C
C       20.07.83    LAST CHANGED 10.05.89        ANSGAR DENNER        C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
        COMPLEX*16 CSPEN_HDEC,W,SUM,Z,U
        REAL*8 RZ,AZ,A1
        REAL*8 B(9)/
     1   0.1666666666666666666666666667D0,
     2  -0.0333333333333333333333333333D0,
     3   0.0238095238095238095238095238D0,
     4  -0.0333333333333333333333333333D0,
     5   0.0757575757575757575757575758D0,
     6  -0.2531135531135531135531135531D0,
     7   1.1666666666666666666666666667D0,
     8  -7.09215686274509804D0         ,
     9  54.97117794486215539D0         /
C     BEACHTE:                 B(N)=B2N
C     B(1)=1./6.
C     B(2)=-1./30.
C     B(3)=1./42.
C     B(4)=-1./30.
C     B(5)=5./66.
C     B(6)=-691./2730.
C     B(7)=7./6.
C     B(8)=-3617./510.
C     B(9)=43867./798.
C     B(10)=-174611./330.
C     B(11)=854513./138.
C     PI=3.1415926535897932384
C     PI*PI/6.=1.6449..., PI*PI/3=3.28986...
C
c      write(*,*) 'z:',z
      Z =Z*DCMPLX(1D0)
      RZ=DREAL(Z)
      AZ=CDABS(Z)
      A1=CDABS(1D0-Z)
c      write(*,*)'z, rz, az, a1:',z,rz,az,a1
C     IF((SNGL(RZ) .EQ. 0.0) .AND. (SNGL(DIMAG(Z)) .EQ. 0.0)) THEN
C ---> CHANGED  10.5.89
      IF(AZ .LT. 1D-20) THEN
        CSPEN_HDEC=-CDLOG(1D0-Z)
c        write(*,*) 'cspen:', cspen_HDEC
        RETURN
      END IF
      IF((SNGL(RZ) .EQ. 1.0) .AND. (SNGL(DIMAG(Z)) .EQ. 0.0)) THEN
        CSPEN_HDEC=1.64493406684822643D0
c        write(*,*) 'cspen:', cspen_HDEC
        RETURN
      END IF
      IF(RZ.GT.5D-1) GOTO 20
      IF(AZ.GT.1D0) GOTO 10
      W=-CDLOG(1D0-Z)
      SUM=W-0.25D0*W*W
      U=W
      IF(CDABS(U).LT.1D-10) GOTO 2
c      write(*,*) 'u:',u
c      write(*,*) 'sum:',sum
      DO 1 K=1,9
      U=U*W*W/DFLOAT(2*K*(2*K+1))
      IF(CDABS(U*B(K)/SUM).LT.1D-20) GOTO 2
      SUM=SUM+U*B(K)
 1    CONTINUE
 2    CSPEN_HDEC=SUM
c        write(*,*) 'cspen:', cspen_HDEC
      RETURN
10    W=-CDLOG(1D0-1D0/Z)
      SUM=W-0.25D0*W*W
      U=W
      IF(CDABS(U).LT.1D-10) GOTO 12

      DO 11 K=1,9
      U=U*W*W/DFLOAT(2*K*(2*K+1))
      IF(CDABS(B(K)*U/SUM).LT.1D-20) GOTO 12
      SUM=SUM+U*B(K)
11    CONTINUE
12    CSPEN_HDEC=-SUM-1.64493406684822643D0-.5D0*CDLOG(-Z)**2
c        write(*,*) 'cspen:', cspen_HDEC
      RETURN
20    IF(A1.GT.1D0) GOTO 30
      W=-CDLOG(Z)
      SUM=W-0.25D0*W*W
      U=W
      IF(CDABS(U).LT.1D-10) GOTO 22
      DO 21 K=1,9
      U=U*W*W/DFLOAT(2*K*(2*K+1))
      IF(CDABS(U*B(K)/SUM).LT.1D-20) GOTO 22
      SUM=SUM+U*B(K)
21    CONTINUE
22    CSPEN_HDEC=-SUM+1.64493406684822643D0-CDLOG(Z)*CDLOG(1D0-Z)
c        write(*,*) 'cspen:', cspen_HDEC
      RETURN
30    W=CDLOG(1D0-1D0/Z)
      SUM=W-0.25D0*W*W
      U=W
      IF(CDABS(U).LT.1D-10) GOTO 32
      DO 31 K=1,9
      U=U*W*W/DFLOAT(2*K*(2*K+1))
      IF(CDABS(U*B(K)/SUM).LT.1D-20) GOTO 32
      SUM=SUM+U*B(K)
31    CONTINUE
32    CSPEN_HDEC=SUM+3.28986813369645287D0
     *               +.5D0*CDLOG(Z-1D0)**2-CDLOG(Z)*CDLOG(1D0-Z)
50    CONTINUE
c        write(*,*) 'cspen:', cspen_HDEC
      END

C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     THIS PROGRAM COMPUTES THE RENORMALIZATION GROUP IMPROVED
C     VALUES OF HIGGS MASSES AND COUPLINGS IN THE MSSM.
C
C     INPUT: MA,TANB = TAN(BETA),MQ,MUR,MDR,MTOP,AU,AD,MU,MCHI
C
C     ALL MASSES IN GEV UNITS. MA IS THE CP-ODD HIGGS MASS,
C     MTOP IS THE PHYSICAL TOP MASS, MQ AND MUR/MDR ARE THE SOFT
C     SUPERSYMMETRY BREAKING MASS PARAMETERS OF LEFT HANDED
C     AND RIGHT HANDED STOPS RESPECTIVELY, AU AND AD ARE THE
C     STOP AND SBOTTOM TRILINEAR SOFT BREAKING TERMS,
C     RESPECTIVELY,  AND MU IS THE SUPERSYMMETRIC
C     HIGGS MASS PARAMETER. WE USE THE  CONVENTIONS FROM
C     THE PHYSICS REPORT OF HABER AND KANE: LEFT RIGHT
C     STOP MIXING TERM PROPORTIONAL TO (AU - MU/TANB).
C     MCHI IS THE HEAVIEST CHARGINO MASS. 
C     WE USE AS INPUT TANB DEFINED AT THE SCALE MTOP.

C     OUTPUT: MH,HM,MCH, SA = SIN(ALPHA), CA= COS(ALPHA), TANBA
C     WHERE MHP AND HPM ARE THE LIGHTEST AND HEAVIEST CP-EVEN
C     HIGGS MASSES, MHCH IS THE CHARGED HIGGS MASS AND
C     ALPHA IS THE HIGGS MIXING ANGLE.
C     TANBA IS THE ANGLE TANB AT THE CP-ODD HIGGS MASS SCALE.

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c       Program based on the work by M. Carena, M. Quiros
c       and C.E.M. Wagner, "Effective potential methods and
c       the Higgs mass spectrum in the MSSM", Nucl. Phys.
c       B461 (1996) 407. 
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      SUBROUTINE SUBH1_HDEC(MA,TANB,MUL,MDL,MUR,MD,MTOP,AU,AD,MU,MCHI0,
     *                 MHP,HMP,MCH,SA,CA,TANBA,MGLU)

      IMPLICIT REAL*8(A-H,L,M,O-Z)
      DIMENSION VH(2,2),M2(2,2),M2P(2,2)
      COMMON/PARAM_HDEC/GF,ALPH,AMTAU,AMMUON,AMZ,AMW
      COMMON/HSELF_HDEC/LAMBDA1,LAMBDA2,LAMBDA3,LAMBDA4,LAMBDA5,
     .             LAMBDA6,LAMBDA7

      MCHI = MCHI0
      TANBA = TANB
      TANBT = TANB
      
      PI = 4*DATAN(1D0)
      MZ = AMZ
      MW = AMW
      V  = 1/DSQRT(2*DSQRT(2D0)*GF)
      CW = AMW**2/AMZ**2
      SW = 1-CW
      ALPHA2  = (2*AMW/V/DSQRT(2D0))**2/4/PI
      ALPHA1  = ALPHA2*SW/CW
      ALPHA3Z = ALPHAS_HDEC(AMZ,3)
      ALPHA3  = ALPHAS_HDEC(MTOP,3)
      MB      = RUNM_HDEC(MTOP,5)
      RMTOP   = RUNM_HDEC(MTOP,6)

      TUL = LOG((MUL**2+MTOP**2)/MTOP**2)
      TDL = LOG((MDL**2+MTOP**2)/MTOP**2)
      TU = LOG((MUR**2 + MTOP**2)/MTOP**2)
      TD = LOG((MD**2 + MTOP**2)/MTOP**2)
      SINB = TANB/DSQRT(1.D0 + TANB**2)
      COSB = SINB/TANB

      IF(MA.GT.MTOP)
     *       TANBA = TANB*(1.D0-3.D0/32.D0/PI**2*
     *       (RMTOP**2/V**2/SINB**2-MB**2/V**2/COSB**2)*
     *       DLOG(MA**2/MTOP**2))
      IF(MA.LT.MTOP.OR.MA.EQ.MTOP) TANBT = TANBA

      SINB = TANBT/DSQRT(1.D0 + TANBT**2)
      COSB = 1.D0/DSQRT(1.D0 + TANBT**2)
      COS2B = (TANBT**2 - 1.D0)/(TANBT**2 + 1.D0)
      G1 = DSQRT(ALPHA1*4.D0*PI)
      G2 = DSQRT(ALPHA2*4.D0*PI)
      G3 = DSQRT(ALPHA3*4.D0*PI)
      HU = RMTOP/V/SINB
      HD =  MB/V/COSB
C

      IF(MUL.GT.MUR) TP = TUL - TU
      IF(MUL.LT.MUR.OR.MUL.EQ.MUR) TP = TU - TUL
      IF(MUL.GT.MUR) TDP = TU
      IF(MUL.LT.MUR.OR.MUL.EQ.MUR) TDP = TUL
      IF(MDL.GT.MD) TPD = TDL - TD
      IF(MDL.LT.MD.OR.MDL.EQ.MD) TPD = TD - TDL
      IF(MDL.GT.MD) TDPD = TD
      IF(MDL.LT.MD.OR.MDL.EQ.MD) TDPD = TDL

      IF(MDL.GT.MD) DLAMBDA1 = 6./96./PI**2*G1**2*HD**2*TPD
      IF(MDL.LT.MD.OR.MDL.EQ.MD) DLAMBDA1 = 3./32./PI**2*
     * HD**2*(G1**2/3.+G2**2)*TPD

      IF(MUL.GT.MUR) DLAMBDA2 =12./96./PI**2*G1**2*HU**2*TP
      IF(MUL.LT.MUR.OR.MUL.EQ.MUR) DLAMBDA2 = 3./32./PI**2*
     * HU**2*(-G1**2/3.+G2**2)*TP

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c  dlambdap1 and dlambdap2 are the new log corrections due to
c  the presence of the gluino mass. They are in general very small,
c  and only present if there is a hierarchy of masses between the
c  two stops.
c
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

        dlambdap2 = 0
        tglu = log(mglu**2/mtop**2)

        if(mglu.lt.mur.or.mglu.lt.mul) then
        if(mul.gt.mur.and.mglu.gt.mur) then
        dlambdap2 = -4./(16.*pi**2)**2*hu**4*(tul**2-tglu**2)
        endif

        if(mul.gt.mur.and.mglu.lt.mur) then
        dlambdap2 = -4./(16.*pi**2)**2*hu**4*(tul**2-tu**2)
        endif

        if(mul.gt.mur.and.mglu.eq.mur) then
        dlambdap2 = -4./(16.*pi**2)**2*hu**4*(tul**2-tu**2)
        endif

        if(mur.gt.mul.and.mglu.gt.mul) then
        dlambdap2 = -4./(16.*pi**2)**2*hu**4*(tu**2-tglu**2)
        endif

        if(mur.gt.mul.and.mglu.lt.mul) then
        dlambdap2 = -4./(16.*pi**2)**2*hu**4*(tu**2-tul**2)
        endif

        if(mur.gt.mul.and.mglu.eq.mul) then
        dlambdap2 = -4./(16.*pi**2)**2*hu**4*(tu**2-tul**2)
        endif
        endif

      DLAMBDA3 = 0.
      DLAMBDA4 = 0.

      IF(MDL.GT.MD) DLAMBDA3 = -1./32./PI**2*G1**2*HD**2*TPD
      IF(MDL.LT.MD.OR.MDL.EQ.MD) DLAMBDA3 = 3./64./PI**2*HD**2*
     *(G2**2-G1**2/3.)*TPD
      
      IF(MUL.GT.MUR) DLAMBDA3 = DLAMBDA3 - 
     *1./16./PI**2*G1**2*HU**2*TP
      IF(MUL.LT.MUR.OR.MUL.EQ.MUR) DLAMBDA3 = DLAMBDA3 + 
     * 3./64./PI**2*HU**2*(G2**2+G1**2/3.)*TP

      IF(MUL.LT.MUR) DLAMBDA4 = -3./32./PI**2*G2**2*HU**2*TP
      IF(MDL.LT.MD) DLAMBDA4 = DLAMBDA4 - 3./32./PI**2*G2**2*
     *                        HD**2*TPD
C
      LAMBDA1 = ((G1**2 + G2**2)/4.)*
     *(1.-3.*HD**2*(TPD + TDPD)/8./PI**2)
     *+(3.*HD**4./16./PI**2) *TPD*(1.   
     *+ (3.*HD**2/2. + HU**2/2.       
     *- 8.*G3**2) * (TPD + 2.*TDPD)/16./PI**2) 
     *+(3.*HD**4./8./PI**2) *TDPD*(1.  + (3.*HD**2/2. + HU**2/2.       
     *- 8.*G3**2) * TDPD/16./PI**2) + DLAMBDA1 
C
      LAMBDA2 = ((G1**2 + G2**2)/4.)*(1.-3.*HU**2*
     *(TP + TDP)/8./PI**2)
     *+(3.*HU**4./16./PI**2) *TP*(1.   
     *+ (3.*HU**2/2. + HD**2/2.       
     *- 8.*G3**2) * (TP + 2.*TDP)/16./PI**2) 
     *+(3.*HU**4./8./PI**2) *TDP*(1. + (3.*HU**2/2. + HD**2/2.       
     *- 8.*G3**2) * TDP/16./PI**2) + DLAMBDA2  + DLAMBDAP2
C
      LAMBDA3 = ((G2**2 - G1**2)/4.)*(1.-3.*
     *(HU**2)*(TP + TDP)/16./PI**2 -3.*
     *(HD**2)*(TPD + TDPD)/16./PI**2) +DLAMBDA3 
C
      LAMBDA4 = (- G2**2/2.)*(1.
     *-3.*(HU**2)*(TP + TDP)/16./PI**2
     *-3.*(HD**2)*(TPD + TDPD)/16./PI**2) +DLAMBDA4
C     
	LAMBDA5 = 0.
	LAMBDA6 = 0.
	LAMBDA7 = 0.


C
C     THIS IS THE CONTRIBUTION FROM LIGHT CHARGINOS/NEUTRALINOS
C     CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  	 MSSUSY=DSQRT(0.5D0*(MUL**2+MUR**2)+MTOP**2)
	IF(MCHI.GT.MSSUSY)GOTO 3790
	IF(MCHI.LT.MTOP) MCHI=MTOP
	TCHAR=LOG(MSSUSY**2/MCHI**2)
	DELTAL12=(9./64./PI**2*G2**4+5./192./PI**2*G1**4)*TCHAR
	DELTAL3P4=(3./64./PI**2*G2**4+7./192./PI**2*G1**4
     *       +4./32/PI**2*G1**2*G2**2)*TCHAR
	DELTAM112=2.*DELTAL12*V**2*COSB**2
	DELTAM222=2.*DELTAL12*V**2*SINB**2
	DELTAM122=2.*DELTAL3P4*V**2*SINB*COSB
C--EXTENSION OF CARENA ET AL.: TRAFO MASS MATRIX -> LAMBDA_I
        DLAM1 = DELTAM112/2.D0/V**2/COSB**2
        DLAM2 = DELTAM222/2.D0/V**2/SINB**2
        DLAM3 = DELTAM122/2.D0/V**2/SINB/COSB
     .        *(G1**2-G2**2)/(G1**2+G2**2)
        DLAM4 = DELTAM122/2.D0/V**2/SINB/COSB
     .        *(2*G2**2)/(G1**2+G2**2)
        LAMBDA1 = LAMBDA1+DLAM1
        LAMBDA2 = LAMBDA2+DLAM2
        LAMBDA3 = LAMBDA3+DLAM3
        LAMBDA4 = LAMBDA4+DLAM4
C--END OF EXTENSION
 3790	CONTINUE
CCCCCCCCCCCCCCC    END OF CHARGINOS AND NEUTRALINOS  CCCCCCCCCCCC 


C--EXTENSION OF CARENA ET AL.: TRAFO MASS MATRIX -> LAMBDA_I
      CALL GFUN_HDEC(MA,TANBA,MUL,MDL,MUR,MD,MTOP,AU,AD,MU,MGLU,
     *                 DLAM1,DLAM2,DLAM3,DLAM4,DLAM5,DLAM6,DLAM7)

      LAMBDA1 = LAMBDA1+DLAM1
      LAMBDA2 = LAMBDA2+DLAM2
      LAMBDA3 = LAMBDA3+DLAM3
      LAMBDA4 = LAMBDA4+DLAM4
      LAMBDA5 = LAMBDA5+DLAM5
      LAMBDA6 = LAMBDA6+DLAM6
      LAMBDA7 = LAMBDA7+DLAM7
      
      M2(1,1) = 2.*V**2*(LAMBDA1*COSB**2+2.*LAMBDA6*
     *COSB*SINB + LAMBDA5*SINB**2) + MA**2*SINB**2
      M2(2,2) = 2.*V**2*(LAMBDA5*COSB**2+2.*LAMBDA7*
     *COSB*SINB + LAMBDA2*SINB**2) + MA**2*COSB**2
      M2(1,2) = 2.*V**2*(LAMBDA6*COSB**2+(LAMBDA3+LAMBDA4)*
     *COSB*SINB + LAMBDA7*SINB**2) - MA**2*SINB*COSB
      M2(2,1) = M2(1,2)

      M2P(1,1) = M2(1,1)
      M2P(2,2) = M2(2,2)
      M2P(1,2) = M2(1,2)
      M2P(2,1) = M2(2,1)

C--END OF EXTENSION

      TRM2P  = M2P(1,1) + M2P(2,2)
      DETM2P = M2P(1,1)*M2P(2,2) - M2P(1,2)*M2P(2,1)

      MH2P = (TRM2P - DSQRT(TRM2P**2 - 4.D0* DETM2P))/2.D0
      HM2P = (TRM2P + DSQRT(TRM2P**2 - 4.D0* DETM2P))/2.D0
C !!!!!!!!!!!!!!!!!!!
      MCH2=MA**2+(LAMBDA5-LAMBDA4)*V**2
C !!!!!!!!!!!!!!!!!!!
      MCH=DSQRT(MCH2)
      HMP = DSQRT(HM2P) 
      IF(MH2P.LT.0.)GOTO 5555
      MHP = DSQRT(MH2P) 
C
      SIN2ALPHA = 2.*M2P(1,2)/DSQRT(TRM2P**2-4.D0*DETM2P)
      COS2ALPHA = (M2P(1,1)-M2P(2,2))/DSQRT(TRM2P**2-4.D0*DETM2P)
      IF(COS2ALPHA.GT.0.) ALPHA = DASIN(SIN2ALPHA)/2.D0
      IF(COS2ALPHA.LT.0.) ALPHA = -PI/2.D0-DASIN(SIN2ALPHA)/2.D0
      SA = DSIN(ALPHA)
      CA = DCOS(ALPHA)  
      SQBMA = (SINB*CA - COSB*SA)**2

5555  RETURN
      END
C
CCCCCCCCCCCCCCCCCCCCCCCC NON DEGENERATE STOP/SBOTTOM EFFECTS CCCCCCCCC
C
        SUBROUTINE GFUN_HDEC(MA,TANB,MUL,MDL,MUR,MD,MTOP,AT,AB,MU,MGLU,
     *                     DLAM1,DLAM2,DLAM3,DLAM4,DLAM5,DLAM6,DLAM7)
        IMPLICIT REAL*8 (A-H,L,M,O-Z)
        DIMENSION VH(2,2),VH1(2,2),VH2(2,2),
     *            VH3T(2,2),VH3B(2,2),AL(2,2)
        COMMON/PARAM_HDEC/GF,ALPH,AMTAU,AMMUON,AMZ,AMW
        G(X,Y) = 2.D0 - (X+Y)/(X-Y)*DLOG(X/Y)

        IF(DABS(MU).LT.0.000001) MU = 0.000001
        MUL2  = MUL**2
        MDL2  = MDL**2
        MUR2  = MUR**2
        MD2   = MD**2
        TANBA = TANB
        SINBA = TANBA/DSQRT(TANBA**2+1.D0)
        COSBA = SINBA/TANBA        
        SINB = TANB/DSQRT(TANB**2+1.D0)
        COSB = SINB/TANB

      MB = RUNM_HDEC(MTOP,5)
      PI = 4*DATAN(1D0)
      MZ = AMZ
      MW = AMW
      V  = 1/DSQRT(2*DSQRT(2D0)*GF)
      CW = AMW**2/AMZ**2
      SW = 1-CW
      ALPHA2  = (2*AMW/V/DSQRT(2D0))**2/4/PI
      ALPHA1  = ALPHA2*SW/CW
      ALPHA3Z = ALPHAS_HDEC(AMZ,3)
      ALPHA3  = ALPHAS_HDEC(MTOP,3)

      G1 = DSQRT(ALPHA1*4.*PI)
      G2 = DSQRT(ALPHA2*4.*PI)
      G3 = DSQRT(ALPHA3*4.*PI)
      
        IF(MUL.GT.MUR) MST = MUL
        IF(MUR.GT.MUL.OR.MUR.EQ.MUL) MST = MUR
        MSUSYT = DSQRT(MST**2  + MTOP**2)

	IF(MDL.GT.MD) MSB = MDL
	IF(MD.GT.MDL.OR.MD.EQ.MDL) MSB = MD
	MSUSYB = DSQRT(MSB**2 + MB**2)

	TT = LOG(MSUSYT**2/MTOP**2)
	TB = LOG(MSUSYB**2/MTOP**2)

        RMTOP   = RUNM_HDEC(MTOP,6)

        HT = RMTOP/V/SINB
        HTST = RMTOP/V
        HB =  MB/V/COSB
        G32 = ALPHA3*4.*PI

        BT2 = -(8.*G32 - 9.*HT**2/2. - HB**2/2.)/(4.*PI)**2
	BB2 = -(8.*G32 - 9.*HB**2/2. - HT**2/2.)/(4.*PI)**2
        AL2 = 3./8./PI**2*HT**2
        BT2ST = -(8.*G32 - 9.*HTST**2/2.)/(4.*PI)**2
        ALST = 3./8./PI**2*HTST**2
        AL1 = 3./8./PI**2*HB**2

        AL(1,1) = AL1
        AL(1,2) = (AL2+AL1)/2.
        AL(2,1) = (AL2+AL1)/2.
        AL(2,2) = AL2

	IF(MA.GT.MTOP) THEN
        VI = V*(1. + 3./32./PI**2*HTST**2*LOG(MTOP**2/MA**2))
        H1I = VI*COSBA
        H2I = VI*SINBA
        H1T = H1I*(1.+3./8./PI**2*HB**2*LOG(MA**2/MSUSYT**2))**.25
        H2T = H2I*(1.+3./8./PI**2*HT**2*LOG(MA**2/MSUSYT**2))**.25
        H1B = H1I*(1.+3./8./PI**2*HB**2*LOG(MA**2/MSUSYB**2))**.25
        H2B = H2I*(1.+3./8./PI**2*HT**2*LOG(MA**2/MSUSYB**2))**.25
	ELSE
	VI =  V
	H1I = VI*COSB
	H2I = VI*SINB
        H1T = H1I*(1.+3./8./PI**2*HB**2*LOG(MTOP**2/MSUSYT**2))**.25
        H2T = H2I*(1.+3./8./PI**2*HT**2*LOG(MTOP**2/MSUSYT**2))**.25
        H1B = H1I*(1.+3./8./PI**2*HB**2*LOG(MTOP**2/MSUSYB**2))**.25
        H2B = H2I*(1.+3./8./PI**2*HT**2*LOG(MTOP**2/MSUSYB**2))**.25
	END IF

        TANBST = H2T/H1T
        SINBT = TANBST/(1.+TANBST**2)**.5
        COSBT = SINBT/TANBST

        TANBSB = H2B/H1B
        SINBB = TANBSB/(1.+TANBSB**2)**.5
        COSBB = SINBB/TANBSB

      CALL DELMB_HDEC(MA,TANB,MUL,MDL,MUR,MD,AT,AB,MU,MGLU,
     .           MTOP,DELTAMT,DELTAMB,STOP12,STOP22,SBOT12,SBOT22)

        IF(STOP22.LT.0.) GOTO 4237
        IF(SBOT22.LT.0.) GOTO 4237

        STOP1 = STOP12**.5
        STOP2 = STOP22**.5
        SBOT1 = SBOT12**.5
        SBOT2 = SBOT22**.5

        mtop4 = rmtop**4.*(1.+2.*bt2*tt- al2*tt - 4.*deltamt)
c     * /(1.+deltamt)**4.
        mbot4 = mb**4.*(1.+2.*bb2*tb - al1*tb)
     * /(1.+deltamb)**4.
        MTOP2 = DSQRT(MTOP4)
        MBOT2 = DSQRT(MBOT4)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        mtop2 = rmtop**2*(1+bt2*tt- al2*tt/2 - 2*deltamt)
c     * /(1+deltamt)**2
        mbot2 = mb**2*(1+bb2*tb - al1*tb/2)
     * /(1+deltamb)**2
        mtop4 = mtop2**2
        mbot4 = mbot2**2
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        mb = mb/(1+deltamb)

        VH1(1,1) = 1./TANBST
        VH1(2,1) = -1.
        VH1(1,2) = -1.
        VH1(2,2) = TANBST
        VH2(1,1) = TANBST
        VH2(1,2) = -1.
        VH2(2,1) = -1.
        VH2(2,2) = 1./TANBST

C CCCCCCCCCCCCCCCCCCCCCCCCCCC  D-terms CCCCCCCCCCCCCCCCCCCCCCCCCCCCC
	STW=SW

	F1T=(MUL2-MUR2)/(STOP12-STOP22)*(.5-4./3.*STW)*
     *         LOG(STOP1/STOP2)
     *        +(.5-2./3.*STW)*LOG(STOP1*STOP2/(MUL2+MTOP2))
     *        + 2./3.*STW*LOG(STOP1*STOP2/(MUR2+MTOP2))

	F1B=(MDL2-MD2)/(SBOT12-SBOT22)*(-.5+2./3.*STW)*
     *        LOG(SBOT1/SBOT2)
     *        +(-.5+1./3.*STW)*LOG(SBOT1*SBOT2/(MDL2+MBOT2))
     *        - 1./3.*STW*LOG(SBOT1*SBOT2/(MD2+MBOT2))

	F2T=1/(STOP12-STOP22)*
     *         (-.5*LOG(STOP12/STOP22)
     *        +(4./3.*STW-.5)*(MUL2-MUR2)/(STOP12-STOP22)*
     *         G(STOP12,STOP22))

	F2B=1/(SBOT12-SBOT22)*
     *         (.5*LOG(SBOT12/SBOT22)
     *        +(-2./3.*STW+.5)*(MDL2-MD2)/(SBOT12-SBOT22)*
     *        G(SBOT12,SBOT22))

C*************************************************************
C
C--EXTENSION OF CARENA ET AL.: TRAFO MASS MATRIX -> LAMBDA_I
C
C TRAFOS APPROXIMATE -> EXACT:
C
C (i)  1/M_{SUSY}^2 -> LOG(M1^2/M2^2) / (M1^2-M2^2)
C
C (ii) 1/M_{SUSY}^4 -> -6 G(M1^2,M2^2) / (M1^2-M2^2)^2
C
C Then use results of Phys. Lett. B355 (1995) 209 in order to
C obtain the results for lambda_1 - lambda_7 according to
C Nucl. Phys. B461 (1996) 407. Perform a full evolution from
C M_SUSY -> m_t for lambdas (anomalous dimensions, v_i).
C
C - ht^2*hb^2 terms neglected in lambda_3,4 (according to
C   Nucl. Phys. B461 (1996) 407)
C
C*************************************************************

        DLAM1T = MTOP4/(SINBT**4)*(MU**2/(STOP1**2
     *    -STOP2**2))**2*G(STOP12,STOP22)
     *  - MZ**2*MTOP2*MU**2/TANBST**2*F2T/COSBT**2

        DLAM1B = MBOT4/(COSBB**4)*(LOG(SBOT1**2*SBOT2**2/
     *    (MDL2+MBOT2)/(MD2+MBOT2))
     *    + 2*AB**2/(SBOT1**2-SBOT2**2)*LOG(SBOT1**2/SBOT2**2))
     *  + MBOT4/(COSBB**4)*(AB**2/
     *    (SBOT1**2-SBOT2**2))**2*G(SBOT12,SBOT22)
     *  + MZ**2*(2*MBOT2*F1B-MBOT2*AB**2*F2B)/COSBB**2

        DLAM2T = MTOP4/(SINBT**4)*(LOG(STOP1**2*STOP2**2/
     *    (MUL2+MTOP2)/(MUR2+MTOP2))
     *  + 2*AT**2/(STOP1**2-STOP2**2)*LOG(STOP1**2/STOP2**2))
     *  + MTOP4/(SINBT**4)*(AT**2/
     *    (STOP1**2-STOP2**2))**2*G(STOP12,STOP22)
     *  + MZ**2*(-2*MTOP2*F1T+MTOP2*AT**2*F2T)/SINBT**2
 
        DLAM2B = MBOT4/(COSBB**4)*MU**4/(SBOT1**2
     *    -SBOT2**2)**2*G(SBOT12,SBOT22)
     *    + MZ**2*MBOT2*MU**2*TANBSB**2*F2B/SINBB**2
 
        DLAM3T = MTOP4/(SINBT**4)*
     *    MU**2/(STOP1**2-STOP2**2)*(LOG(STOP1**2/STOP2**2)/2.D0
     *  + AT**2/(STOP1**2-STOP2**2)*G(STOP12,STOP22))
     *  + MZ**2*(MTOP2/TANBST*F1T-MTOP2*(AT**2-MU**2)/TANBST/2.*F2T)
     *    /SINBT/COSBT/2
c    *  + MTOP2*MBOT2/(SINBT**2*COSBB**2)*(
c    *    LOG(STOP1**2*STOP2**2/(MQ2+MTOP2)/(MUR2+MTOP2))
c    *  + LOG(SBOT1**2*SBOT2**2/(MQ2+MBOT2)/(MD2+MBOT2))
c    *  + ((AT+AB)**2/2-MU**2)*(
c    *      1.D0/(STOP1**2-SBOT1**2)*LOG(STOP1**2/SBOT1**2)
c    *    + 1.D0/(STOP2**2-SBOT2**2)*LOG(STOP2**2/SBOT2**2))
c    *  - (MU**2-AT*AB)**2*(
c    *    - 1.D0/(STOP1**2-SBOT1**2)**2*G(STOP12,SBOT12)
c    *    - 1.D0/(STOP2**2-SBOT2**2)**2*G(STOP22,SBOT22)))

        DLAM3B = MBOT4/(COSBB**4)*MU**2/(SBOT1**2-SBOT2**2)*(
     *    LOG(SBOT1**2/SBOT2**2)/2.D0
     *  + AB**2/(SBOT1**2-SBOT2**2)*G(SBOT12,SBOT22))
     *  + MZ**2*(-MBOT2*TANBSB*F1B+MBOT2*(AB**2-MU**2)*TANBSB/2.*F2B)
     *    /SINBB/COSBB/2

        DLAM4T = MTOP4/(SINBT**4)*
     *    MU**2/(STOP1**2-STOP2**2)*(LOG(STOP1**2/STOP2**2)/2.D0
     *  + AT**2/(STOP1**2-STOP2**2)*G(STOP12,STOP22))
     *  + MZ**2*(MTOP2/TANBST*F1T-MTOP2*(AT**2-MU**2)/TANBST/2.*F2T)
     *    /SINBT/COSBT/2
c    *  - MTOP2*MBOT2/(SINBT**2*COSBB**2)*(
c    *    LOG(STOP1**2*STOP2**2/(MQ2+MTOP2)/(MUR2+MTOP2))
c    *  + LOG(SBOT1**2*SBOT2**2/(MQ2+MBOT2)/(MD2+MBOT2))
c    *  + ((AT+AB)**2/2-MU**2)*(
c    *      1.D0/(STOP1**2-SBOT1**2)*LOG(STOP1**2/SBOT1**2)
c    *    + 1.D0/(STOP2**2-SBOT2**2)*LOG(STOP2**2/SBOT2**2))
c    *  - (MU**2-AT*AB)**2*(
c    *    - 1.D0/(STOP1**2-SBOT1**2)**2*G(STOP12,SBOT12)
c    *    - 1.D0/(STOP2**2-SBOT2**2)**2*G(STOP22,SBOT22)))

        DLAM4B = MBOT4/(COSBB**4)*MU**2/(SBOT1**2-SBOT2**2)*(
     *    LOG(SBOT1**2/SBOT2**2)/2.D0
     *  + AB**2/(SBOT1**2-SBOT2**2)*G(SBOT12,SBOT22))
     *  + MZ**2*(-MBOT2*TANBSB*F1B+MBOT2*(AB**2-MU**2)*TANBSB/2.*F2B)
     *    /SINBB/COSBB/2

        DLAM5T = MTOP4/(SINBT**4)*
     *    (MU**2*AT**2)/(STOP1**2-STOP2**2)**2*G(STOP12,STOP22)

        DLAM5B = MBOT4/(COSBB**4)*
     *    (MU**2*AB**2)/(SBOT1**2-SBOT2**2)**2*G(SBOT12,SBOT22)

        DLAM6T = MTOP4/(SINBT**4)*
     *    (-MU**3*AT)/(STOP1**2-STOP2**2)**2*G(STOP12,STOP22)
     *  + MZ**2*MTOP2*MU*AT/TANBST*F2T/(2*SINBT*COSBT)

        DLAM6B = MBOT4/(COSBB**4)*MU*AB*
     *    (-1.D0/(SBOT1**2-SBOT2**2)*LOG(SBOT1**2/SBOT2**2)
     *    -AB**2/(SBOT1**2-SBOT2**2)**2*G(SBOT12,SBOT22))
     *  - MZ**2*(-MBOT2*AB*MU*TANBSB*F2B)/(2*SINBB*COSBB)

        DLAM7T = MTOP4/(SINBT**4)*MU*AT*
     *    (-1.D0/(STOP1**2-STOP2**2)*LOG(STOP1**2/STOP2**2)
     *    -AT**2/(STOP1**2-STOP2**2)**2*G(STOP12,STOP22))
     *  - MZ**2*MTOP2*AT*MU/TANBST*F2T/(2*SINBT*COSBT)

        DLAM7B = MBOT4/(COSBB**4)*
     *    (-MU**3*AB)/(SBOT1**2-SBOT2**2)**2*G(SBOT12,SBOT22)
     *    - MZ**2*MBOT2*MU*AB*TANBSB*F2B/(2*SINBB*COSBB)

       TQ = LOG((MUL2 + MTOP2)/MTOP2)
       TU = LOG((MUR2+MTOP2)/MTOP2)
       TQD = LOG((MDL2 + MB**2)/MB**2)
       TD = LOG((MD2+MB**2)/MB**2)

        FACT = 3.D0/(16.D0*PI**2*(H1T**2+H2T**2)**2)
        FACB = 3.D0/(16.D0*PI**2*(H1B**2+H2B**2)**2)

        DLAM1 = FACT*DLAM1T*(1.-AL1*TT) + FACB*DLAM1B*(1.-AL1*TB)

        DLAM2 = FACT*DLAM2T*(1.-AL2*TT) + FACB*DLAM2B*(1.-AL2*TB)

        DLAM3 = FACT*DLAM3T*(1.-(AL1+AL2)/2*TT)
     *        + FACB*DLAM3B*(1.-(AL1+AL2)/2*TB)

        DLAM4 = FACT*DLAM4T*(1.-(AL1+AL2)/2*TT)
     *        + FACB*DLAM4B*(1.-(AL1+AL2)/2*TB)

        DLAM5 = FACT*DLAM5T*(1.-(AL1+AL2)/2*TT)
     *        + FACB*DLAM5B*(1.-(AL1+AL2)/2*TB)

        DLAM6 = FACT*DLAM6T*(1.-(3*AL1+AL2)/4*TT)
     *        + FACB*DLAM6B*(1.-(3*AL1+AL2)/4*TB)

        DLAM7 = FACT*DLAM7T*(1.-(AL1+3*AL2)/4*TT)
     *        + FACB*DLAM7B*(1.-(AL1+3*AL2)/4*TB)

        FACTOR = 1.D0
        DLAM1 = DLAM1 * FACTOR
        DLAM2 = DLAM2 * FACTOR
        DLAM3 = DLAM3 * FACTOR
        DLAM4 = DLAM4 * FACTOR
        DLAM5 = DLAM5 * FACTOR
        DLAM6 = DLAM6 * FACTOR
        DLAM7 = DLAM7 * FACTOR

C--END OF EXTENSION

        GOTO 4236
 4237   CONTINUE

        DLAM1 = -1.D+15
        DLAM2 = -1.D+15
        DLAM3 = -1.D+15
        DLAM4 = -1.D+15
        DLAM5 = -1.D+15
        DLAM6 = -1.D+15
        DLAM7 = -1.D+15

4236    RETURN
        END
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c       End of program from M. Carena, M. Quiros and C.E.M. Wagner.
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      FUNCTION T_HDEC(X,Y,Z)
      implicit real*8(a-h,l,m,o-z)
      delta(a,b) = dabs((a-b)/(a+b))
      cut = 0.0001d0
c     if(x.eq.y) x = x - 0.00001
c     if(x.eq.z) x = x - 0.00002
c     if(y.eq.z) y = y - 0.00003
      if(delta(x,y).lt.cut) then
       if(delta(x,z).lt.cut) then
        t_hdec = 1/x**2/2
       else
        t_hdec = (x**2-z**2+z**2*log(z**2/x**2))/(x**2-z**2)**2
       endif
      elseif(delta(x,z).lt.cut) then
       if(delta(x,y).lt.cut) then
        t_hdec = 1/x**2/2
       else
        t_hdec = (x**2-y**2+y**2*log(y**2/x**2))/(x**2-y**2)**2
       endif
      elseif(delta(y,z).lt.cut) then
       if(delta(x,y).lt.cut) then
        t_hdec = 1/x**2/2
       else
        t_hdec = (x**2-y**2+y**2*log(y**2/x**2))/(x**2-y**2)**2
       endif
      else
       t_hdec = (x**2*y**2*log(x**2/y**2) + x**2*z**2*log(z**2/x**2)
     * + y**2*z**2*log(y**2/z**2))/((x**2-y**2)*(y**2-z**2)*(x**2-z**2))
      endif
      return
      end

      SUBROUTINE DELMB_HDEC(MA,TANB,MUL,MDL,MUR,MD,AT,AB,MU,MGLU,
     .           MTOP,DELTAMT,DELTAMB,STOP12,STOP22,SBOT12,SBOT22)
        IMPLICIT REAL*8 (A-H,L,M,O-Z)
        COMMON/PARAM_HDEC/GF,ALPH,AMTAU,AMMUON,AMZ,AMW

        IF(DABS(MU).LT.0.000001) MU = 0.000001
        MUL2  = MUL**2
        MDL2  = MDL**2
        MUR2  = MUR**2
        MD2   = MD**2
        TANBA = TANB
        SINBA = TANBA/DSQRT(TANBA**2+1.D0)
        COSBA = SINBA/TANBA        
        SINB = TANB/DSQRT(TANB**2+1.D0)
        COSB = SINB/TANB

      RMTOP = RUNM_HDEC(MTOP,6)
      MB = RUNM_HDEC(MTOP,5)
      PI = 4*DATAN(1D0)
      MZ = AMZ
      MW = AMW
      V  = 1/DSQRT(2*DSQRT(2D0)*GF)
      CW = AMW**2/AMZ**2
      SW = 1-CW
      ALPHA2  = (2*AMW/V/DSQRT(2D0))**2/4/PI
      ALPHA1  = ALPHA2*SW/CW
      ALPHA3Z = ALPHAS_HDEC(AMZ,3)
      ALPHA3  = ALPHAS_HDEC(MTOP,3)

      G1 = DSQRT(ALPHA1*4.*PI)
      G2 = DSQRT(ALPHA2*4.*PI)
      G3 = DSQRT(ALPHA3*4.*PI)
      
        IF(MUL.GT.MUR) MST = MUL
        IF(MUR.GT.MUL.OR.MUR.EQ.MUL) MST = MUR
        MSUSYT = DSQRT(MST**2  + MTOP**2)

	IF(MDL.GT.MD) MSB = MDL
	IF(MD.GT.MDL.OR.MD.EQ.MDL) MSB = MD
	MSUSYB = DSQRT(MSB**2 + MB**2)

	TT = LOG(MSUSYT**2/MTOP**2)
	TB = LOG(MSUSYB**2/MTOP**2)

        HT = RMTOP/V/SINB
        HTST = RMTOP/V
        HB =  MB/V/COSB
        G32 = ALPHA3*4.*PI

        BT2 = -(8.*G32 - 9.*HT**2/2. - HB**2/2.)/(4.*PI)**2
	BB2 = -(8.*G32 - 9.*HB**2/2. - HT**2/2.)/(4.*PI)**2
        AL2 = 3./8./PI**2*HT**2
        BT2ST = -(8.*G32 - 9.*HTST**2/2.)/(4.*PI)**2
        ALST = 3./8./PI**2*HTST**2
        AL1 = 3./8./PI**2*HB**2

        IF(MA.GT.MTOP) THEN
        VI = V*(1. + 3./32./PI**2*HTST**2*LOG(MTOP**2/MA**2))
        H1I = VI*COSBA
        H2I = VI*SINBA
        H1T = H1I*(1.+3./8./PI**2*HB**2*LOG(MA**2/MSUSYT**2))**.25
        H2T = H2I*(1.+3./8./PI**2*HT**2*LOG(MA**2/MSUSYT**2))**.25
        H1B = H1I*(1.+3./8./PI**2*HB**2*LOG(MA**2/MSUSYB**2))**.25
        H2B = H2I*(1.+3./8./PI**2*HT**2*LOG(MA**2/MSUSYB**2))**.25
        ELSE
        VI =  V
        H1I = VI*COSB
        H2I = VI*SINB
        H1T = H1I*(1.+3./8./PI**2*HB**2*LOG(MTOP**2/MSUSYT**2))**.25
        H2T = H2I*(1.+3./8./PI**2*HT**2*LOG(MTOP**2/MSUSYT**2))**.25
        H1B = H1I*(1.+3./8./PI**2*HB**2*LOG(MTOP**2/MSUSYB**2))**.25
        H2B = H2I*(1.+3./8./PI**2*HT**2*LOG(MTOP**2/MSUSYB**2))**.25
        END IF

        TANBST = H2T/H1T
        SINBT = TANBST/(1.+TANBST**2)**.5
        COSBT = SINBT/TANBST

        TANBSB = H2B/H1B
        SINBB = TANBSB/(1.+TANBSB**2)**.5
        COSBB = SINBB/TANBSB

        deltamt = 0
        deltamb = 0

        mtop4 = rmtop**4.*(1.+2.*bt2*tt- al2*tt - 4.*deltamt)
c     * /(1.+deltamt)**4.
        mbot4 = mb**4.*(1.+2.*bb2*tb - al1*tb)
     * /(1.+deltamb)**4.
        MTOP2 = DSQRT(MTOP4)
	MBOT2 = DSQRT(MBOT4)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        mtop2 = rmtop**2*(1+bt2*tt- al2*tt/2 - 2*deltamt)
c     * /(1+deltamt)**2
        mbot2 = mb**2*(1+bb2*tb - al1*tb/2)
     * /(1+deltamb)**2
        mtop4 = mtop2**2
        mbot4 = mbot2**2
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        STOP12 = (MUL2 + MUR2)*.5 + MTOP2 
     *   +1./8.*(G2**2+G1**2)*(H1T**2-H2T**2)
     *   +(((G2**2-5.*G1**2/3.)/4.*(H1T**2-H2T**2) +
     *   MUL2 - MUR2)**2*0.25 + MTOP2*(AT-MU/TANBST)**2)**.5

        STOP22 = (MUL2 + MUR2)*.5 + MTOP2 
     *  +1./8.*(G2**2+G1**2)*(H1T**2-H2T**2) 
     *   - (((G2**2-5.*G1**2/3.)/4.*(H1T**2-H2T**2) +
     *  MUL2 - MUR2)**2*0.25 
     *  + MTOP2*(AT-MU/TANBST)**2)**.5

        IF(STOP22.LT.0.) GOTO 4237

        SBOT12 = (MDL2 + MD2)*.5  
     *   - 1./8.*(G2**2+G1**2)*(H1B**2-H2B**2)
     *  + (((G1**2/3.-G2**2)/4.*(H1B**2-H2B**2) +
     *  MDL2 - MD2)**2*0.25 + MBOT2*(AB-MU*TANBSB)**2)**.5

        SBOT22 = (MDL2 + MD2)*.5  
     *   - 1./8.*(G2**2+G1**2)*(H1B**2-H2B**2)
     *   - (((G1**2/3.-G2**2)/4.*(H1B**2-H2B**2) +
     *   MDL2 - MD2)**2*0.25 + MBOT2*(AB-MU*TANBSB)**2)**.5

        IF(SBOT22.LT.0.) GOTO 4237

        STOP1 = STOP12**.5
        STOP2 = STOP22**.5
        SBOT1 = SBOT12**.5
        SBOT2 = SBOT22**.5

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     Here is the definition of deltamb and deltamt, which
c     are the vertex corrections to the bottom and top quark
c     mass, keeping the dominant QCD and top Yukawa coupling
c     induced corrections.
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

        deltamb = -2*alpha3/3./pi*mglu*(ab-mu*tanb)*
     *  T_HDEC(sbot1,sbot2,mglu)
     *  + ht**2/(4.*pi)**2*(at-mu/tanb)*mu*tanb*
     *  T_HDEC(stop1,stop2,mu)


        deltamt = -2.*alpha3/3./pi*(at-mu/tanb)*mglu*
     *  T_HDEC(stop1,stop2,mglu)

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c   Here the new values of the top and bottom quark masses at
c   the scale MS are defined, to be used in the effective
c   potential approximation. They are just the old ones, but
c   including the finite corrections deltamt and deltamb.
c   The deltamb corrections can become large and are resummed
c   to all orders, as suggested in the two recent works by M. Carena,
c   S. Mrenna and C.E.M. Wagner, as well as in the work by M. Carena,
c   D. Garcia, U. Nierste and C.E.M. Wagner, to appear. The top
c   quark mass corrections are small and are kept in the perturbative
c   formulation. The function T(X,Y,Z) is necessary for the calculation.
c   the entries are masses and NOT their squares !
c
c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc


        mtop4 = rmtop**4.*(1.+2.*bt2*tt- al2*tt - 4.*deltamt)
c     * /(1.+deltamt)**4.
        mbot4 = mb**4.*(1.+2.*bb2*tb - al1*tb)
     * /(1.+deltamb)**4.
        MTOP2 = DSQRT(MTOP4)
	MBOT2 = DSQRT(MBOT4)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        mtop2 = rmtop**2*(1+bt2*tt- al2*tt/2 - 2*deltamt)
c     * /(1+deltamt)**2
        mbot2 = mb**2*(1+bb2*tb - al1*tb/2)
     * /(1+deltamb)**2
        mtop4 = mtop2**2
        mbot4 = mbot2**2
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        STOP12 = (MUL2 + MUR2)*.5 + MTOP2 
     *   +1./8.*(G2**2+G1**2)*(H1T**2-H2T**2)
     *   +(((G2**2-5.*G1**2/3.)/4.*(H1T**2-H2T**2) +
     *   MUL2 - MUR2)**2*0.25 + MTOP2*(AT-MU/TANBST)**2)**.5

        STOP22 = (MUL2 + MUR2)*.5 + MTOP2 
     *  +1./8.*(G2**2+G1**2)*(H1T**2-H2T**2) 
     *   - (((G2**2-5.*G1**2/3.)/4.*(H1T**2-H2T**2) +
     *  MUL2 - MUR2)**2*0.25 
     *  + MTOP2*(AT-MU/TANBST)**2)**.5

        IF(STOP22.LT.0.) GOTO 4237

        SBOT12 = (MDL2 + MD2)*.5  
     *   - 1./8.*(G2**2+G1**2)*(H1B**2-H2B**2)
     *  + (((G1**2/3.-G2**2)/4.*(H1B**2-H2B**2) +
     *  MDL2 - MD2)**2*0.25 + MBOT2*(AB-MU*TANBSB)**2)**.5

        SBOT22 = (MDL2 + MD2)*.5  
     *   - 1./8.*(G2**2+G1**2)*(H1B**2-H2B**2)
     *   - (((G1**2/3.-G2**2)/4.*(H1B**2-H2B**2) +
     *   MDL2 - MD2)**2*0.25 + MBOT2*(AB-MU*TANBSB)**2)**.5

4237    RETURN
        END

C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     THIS PROGRAM COMPUTES THE RENORMALIZATION GROUP IMPROVED
C     VALUES OF HIGGS MASSES AND COUPLINGS IN THE MSSM.
C
C     INPUT: MA,TANB = TAN(BETA),MQ,MUR,MTOP,AU,AD,MU.
C
C     ALL MASSES IN GEV UNITS. MA IS THE CP-ODD HIGGS MASS,
C     MTOP IS THE PHYSICAL TOP MASS, MQ AND MUR ARE THE SOFT
C     SUPERSYMMETRY BREAKING MASS PARAMETERS OF LEFT HANDED
C     AND RIGHT HANDED STOPS RESPECTIVELY, AU AND AD ARE THE
C     STOP AND SBOTTOM TRILINEAR SOFT BREAKING TERMS,
C     RESPECTIVELY,  AND MU IS THE SUPERSYMMETRIC
C     HIGGS MASS PARAMETER. WE USE THE  CONVENTIONS FROM
C     THE PHYSICS REPORT OF HABER AND KANE: LEFT RIGHT
C     STOP MIXING TERM PROPORTIONAL TO (AU - MU/TANB).
C
C     WE USE AS INPUT TANB DEFINED AT THE SCALE MTOP.
C
C     OUTPUT: MH,HM,MHCH, SA = SIN(ALPHA), CA= COS(ALPHA), TANBA
C
C     WHERE MH AND HM ARE THE LIGHTEST AND HEAVIEST CP-EVEN
C     HIGGS MASSES, MHCH IS THE CHARGED HIGGS MASS AND
C     ALPHA IS THE HIGGS MIXING ANGLE.
C
C     TANBA IS THE ANGLE TANB AT THE CP-ODD HIGGS MASS SCALE.
C
C     RANGE OF VALIDITY:
C
C    (STOP1**2 - STOP2**2)/(STOP2**2 + STOP1**2) < 0.5
C    (SBOT1**2 - SBOT2**2)/(SBOT2**2 + SBOT2**2) < 0.5
C
C     WHERE STOP1, STOP2, SBOT1 AND SBOT2 ARE THE STOP AND
C     ARE THE SBOTTOM  MASS EIGENVALUES, RESPECTIVELY. THIS
C     RANGE AUTOMATICALLY EXCLUDES THE EXISTENCE OF TACHYONS.
C
C
C     FOR THE CHARGED HIGGS MASS COMPUTATION, THE METHOD IS
C     VALID IF
C
C     2 * |MB * AD* TANB|  < M_SUSY**2,  2 * |MTOP * AU| < M_SUSY**2
C
C     2 * |MB * MU * TANB| < M_SUSY**2,  2 * |MTOP * MU| < M_SUSY**2
C
C     WHERE M_SUSY**2 IS THE AVERAGE OF THE SQUARED STOP MASS
C     EIGENVALUES, M_SUSY**2 = (STOP1**2 + STOP2**2)/2. THE SBOTTOM
C     MASSES HAVE BEEN ASSUMED TO BE OF ORDER OF THE STOP ONES.
C
C     M_SUSY**2 = (MQ**2 + MUR**2)*0.5 + MTOP**2
C
C     PROGRAM BASED ON THE WORK BY M. CARENA, J.R. ESPINOSA,
C     M. QUIROS AND C.E.M. WAGNER, PHYS. LETT. B355 (1995) 209
C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      SUBROUTINE SUBH2_HDEC(MA,TANB,MQ,MUR,MTOP,AU,AD,MU,MH,HM,
     * MHCH,SA,CA,TANBA)
      IMPLICIT REAL*8(A-H,L,M,O-Z)
      COMMON/PARAM_HDEC/GF,ALPH,AMTAU,AMMUON,AMZ,AMW
      COMMON/HSELF_HDEC/LAMBDA1,LAMBDA2,LAMBDA3,LAMBDA4,LAMBDA5,
     .             LAMBDA6,LAMBDA7
C     MZ = 91.18
C     ALPHA1 = 0.0101
C     ALPHA2 = 0.0337
C     ALPHA3Z = 0.12
C     V = 174.1
C     PI = 3.14159
      TANBA = TANB
      TANBT = TANB

C     MBOTTOM(MTOP) = 3. GEV
C     MB = 3.
C     ALPHA3 = ALPHA3Z/(1. +(11. - 10./3.)/4./PI*ALPHA3Z*
C    *LOG(MTOP**2/MZ**2))

C     RMTOP= RUNNING TOP QUARK MASS
C     RMTOP = MTOP/(1.+4.*ALPHA3/3./PI)
C>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      MB = RUNM_HDEC(MTOP,5)
      PI = 4*DATAN(1D0)
      MZ = AMZ
      V  = 1/DSQRT(2*DSQRT(2D0)*GF)
      CW = AMW**2/AMZ**2
      SW = 1-CW
      ALPHA2  = (2*AMW/V/DSQRT(2D0))**2/4/PI
      ALPHA1  = ALPHA2*SW/CW
      ALPHA3Z = ALPHAS_HDEC(AMZ,3)
      ALPHA3  = ALPHAS_HDEC(MTOP,3)
      RMTOP   = RUNM_HDEC(MTOP,6)
C>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
C      RMTOP=MTOP
C>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      MS = ((MQ**2 + MUR**2)/2. + MTOP**2)**.5
      T = LOG(MS**2/MTOP**2)
      SINB = TANB/((1. + TANB**2)**.5)
      COSB = SINB/TANB
C      IF(MA.LE.MTOP) TANBA = TANBT
      IF(MA.GT.MTOP)
     *TANBA = TANBT*(1.-3./32./PI**2*
     *(RMTOP**2/V**2/SINB**2-MB**2/V**2/COSB**2)*
     *LOG(MA**2/MTOP**2))

      SINBT = TANBT/((1. + TANBT**2)**.5)
      COSBT = 1./((1. + TANBT**2)**.5)
      COS2BT = (TANBT**2 - 1.)/(TANBT**2 + 1.)
      G1 = (ALPHA1*4.*PI)**.5
      G2 = (ALPHA2*4.*PI)**.5
      G3 = (ALPHA3*4.*PI)**.5
      HU = RMTOP/V/SINBT
      HD =  MB/V/COSBT

C>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
C      G3=0
C      HU=0
C      HD=0
C>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

      XAU = (2.*AU**2/MS**2)*(1. - AU**2/12./MS**2)
      XAD = (2.*AD**2/MS**2)*(1. - AD**2/12./MS**2)
      AUD = (-6.*MU**2/MS**2 - ( MU**2- AD*AU)**2/MS**4.
     *+ 3.*(AU + AD)**2/MS**2)/6.
      LAMBDA1 = ((G1**2 + G2**2)/4.)*(1.-3.*HD**2*T/8./PI**2)
     *+(3.*HD**4/8./PI**2) * (T + XAD/2. + (3.*HD**2/2. + HU**2/2.
     *- 8.*G3**2) * (XAD*T + T**2)/16./PI**2)
     *-(3.*HU**4* MU**4/96./PI**2/MS**4) * (1+ (9.*HU**2 -5.* HD**2
     *-  16.*G3**2) *T/16./PI**2)
      LAMBDA2 = ((G1**2 + G2**2)/4.)*(1.-3.*HU**2*T/8./PI**2)
     *+(3.*HU**4/8./PI**2) * (T + XAU/2. + (3.*HU**2/2. + HD**2/2.
     *- 8.*G3**2) * (XAU*T + T**2)/16./PI**2)
     *-(3.*HD**4* MU**4/96./PI**2/MS**4) * (1+ (9.*HD**2 -5.* HU**2
     *-  16.*G3**2) *T/16./PI**2)
      LAMBDA3 = ((G2**2 - G1**2)/4.)*(1.-3.*
     *(HU**2 + HD**2)*T/16./PI**2)
     *+(6.*HU**2*HD**2/16./PI**2) * (T + AUD/2. + (HU**2 + HD**2
     *- 8.*G3**2) * (AUD*T + T**2)/16./PI**2)
     *+(3.*HU**4/96./PI**2) * (3.*MU**2/MS**2 - MU**2*AU**2/
     *MS**4)* (1.+ (6.*HU**2 -2.* HD**2/2.
     *-  16.*G3**2) *T/16./PI**2)
     *+(3.*HD**4/96./PI**2) * (3.*MU**2/MS**2 - MU**2*AD**2/
     *MS**4)*(1.+ (6.*HD**2 -2.* HU**2
     *-  16.*G3**2) *T/16./PI**2)
      LAMBDA4 = (- G2**2/2.)*(1.-3.*(HU**2 + HD**2)*T/16./PI**2)
     *-(6.*HU**2*HD**2/16./PI**2) * (T + AUD/2. + (HU**2 + HD**2
     *- 8.*G3**2) * (AUD*T + T**2)/16./PI**2)
     *+(3.*HU**4/96./PI**2) * (3.*MU**2/MS**2 - MU**2*AU**2/
     *MS**4)*
     *(1+ (6.*HU**2 -2.* HD**2
     *-  16.*G3**2) *T/16./PI**2)
     *+(3.*HD**4/96./PI**2) * (3.*MU**2/MS**2 - MU**2*AD**2/
     *MS**4)*
     *(1+ (6.*HD**2 -2.* HU**2/2.
     *-  16.*G3**2) *T/16./PI**2)
      LAMBDA5 = -(3.*HU**4* MU**2*AU**2/96./PI**2/MS**4) *
     * (1- (2.*HD**2 -6.* HU**2 + 16.*G3**2) *T/16./PI**2)
     *-(3.*HD**4* MU**2*AD**2/96./PI**2/MS**4) *
     * (1- (2.*HU**2 -6.* HD**2 + 16.*G3**2) *T/16./PI**2)
      LAMBDA6 = (3.*HU**4* MU**3*AU/96./PI**2/MS**4) *
     * (1- (7.*HD**2/2. -15.* HU**2/2. + 16.*G3**2) *T/16./PI**2)
     *+(3.*HD**4* MU *(AD**3/MS**3 - 6.*AD/MS )/96./PI**2/MS) *
     * (1- (HU**2/2. -9.* HD**2/2. + 16.*G3**2) *T/16./PI**2)
      LAMBDA7 = (3.*HD**4* MU**3*AD/96./PI**2/MS**4) *
     * (1- (7.*HU**2/2. -15.* HD**2/2. + 16.*G3**2) *T/16./PI**2)
     *+(3.*HU**4* MU *(AU**3/MS**3 - 6.*AU/MS )/96./PI**2/MS) *
     * (1- (HD**2/2. -9.* HU**2/2. + 16.*G3**2) *T/16./PI**2)
      TRM2 = MA**2 + 2.*V**2* (LAMBDA1* COSBT**2 +
     *2.* LAMBDA6*SINBT*COSBT
     *+ LAMBDA5*SINBT**2 + LAMBDA2* SINBT**2 + 2.* LAMBDA7*SINBT*COSBT
     *+ LAMBDA5*COSBT**2)
      DETM2 = 4.*V**4*(-(SINBT*COSBT*(LAMBDA3 + LAMBDA4) +
     *LAMBDA6*COSBT**2
     *+ LAMBDA7* SINBT**2)**2 + (LAMBDA1* COSBT**2 +
     *2.* LAMBDA6* COSBT*SINBT
     *+ LAMBDA5*SINBT**2)*(LAMBDA2* SINBT**2 +2.* LAMBDA7* COSBT*SINBT
     *+ LAMBDA5*COSBT**2)) + MA**2*2.*V**2 *
     *((LAMBDA1* COSBT**2 +2.*
     *LAMBDA6* COSBT*SINBT + LAMBDA5*SINBT**2)*COSBT**2 +
     *(LAMBDA2* SINBT**2 +2.* LAMBDA7* COSBT*SINBT + LAMBDA5*COSBT**2)
     **SINBT**2
     * +2.*SINBT*COSBT* (SINBT*COSBT*(LAMBDA3
     * + LAMBDA4) + LAMBDA6*COSBT**2
     *+ LAMBDA7* SINBT**2))

      MH2 = (TRM2 - (TRM2**2 - 4.* DETM2)**.5)/2.
      HM2 = (TRM2 + (TRM2**2 - 4.* DETM2)**.5)/2.
      HM = HM2**.5
      MH = MH2**.5
      MHCH2 = MA**2 + (LAMBDA5 - LAMBDA4)* V**2
      MHCH = MHCH2**.5
      MHCH = MHCH2**.5

      SINALPHA = SQRT(((TRM2**2 - 4.* DETM2)**.5) -
     * ((2.*V**2*(LAMBDA1* COSBT**2 + 2.*
     *LAMBDA6* COSBT*SINBT
     *+ LAMBDA5*SINBT**2) + MA**2*SINBT**2)
     *- (2.*V**2*(LAMBDA2* SINBT**2 +2.* LAMBDA7* COSBT*SINBT
     *+ LAMBDA5*COSBT**2) + MA**2*COSBT**2)))/
     *SQRT(((TRM2**2 - 4.* DETM2)**.5))/2.**.5

      COSALPHA = (2.*(2.*V**2*(SINBT*COSBT*(LAMBDA3 + LAMBDA4) +
     *LAMBDA6*COSBT**2 + LAMBDA7* SINBT**2) -
     *MA**2*SINBT*COSBT))/2.**.5/
     *SQRT(((TRM2**2 - 4.* DETM2)**.5)*
     *(((TRM2**2 - 4.* DETM2)**.5) -
     * ((2.*V**2*(LAMBDA1* COSBT**2 + 2.*
     *LAMBDA6* COSBT*SINBT
     *+ LAMBDA5*SINBT**2) + MA**2*SINBT**2)
     *- (2.*V**2*(LAMBDA2* SINBT**2 +2.* LAMBDA7* COSBT*SINBT
     *+ LAMBDA5*COSBT**2) + MA**2*COSBT**2))))

      SA = -SINALPHA
      CA = -COSALPHA

 2242 RETURN
      END

      double precision function elwfull_hdec(amh,amf,amfp,qf,ai3f,xmu)
      implicit double precision (a-h,o-z)
      elwfull_hdec = dhffloop_hdec(amh,amf,amfp,qf,ai3f,xmu)
     .             + dhffct_hdec(amh,amf,amfp,qf,ai3f,xmu)
     .             - delta_r_hdec(amh,xmu)
      return
      end

      double precision function dhffloop_hdec(amh,amf,amfp,qf,ai3f,xmu)
      implicit double precision (a-h,o-z)
      complex*16 c03_hdec
      common/param_hdec/gf,alph,amtau,ammuon,amz,amw
      common/masses_hdec/ams,amc,amb,amt
      c03(p1,p2,p3,am1,am2,am3) = dreal(c03_hdec(p1,p2,p3,am1,am2,am3))
      bb1(s,am1,am2,amu2) = (am2**2-am1**2)/2/s
     .                    * (b02_hdec(s,am1,am2,amu2)
     .                      -b02_hdec(0.d0,am1,am2,amu2))
     .                    - b02_hdec(s,am1,am2,amu2)/2
      cc1p(s,am,am3,amu2) = (b02_hdec(s,am,am,amu2)
     .                        -b02_hdec(amf2,am,am3,amu2)
     .                      + (amf2+am3**2-am**2)
     .                        *c03(s,amf2,amf2,am,am,am3))
     .                       / (4*amf2-amh2)
      cc20(s,am,am3,amu2) = (b02_hdec(s,am,am,amu2)
     .                  + 2*(am**2-am3**2-amf2)*cc1p(s,am,am3,amu2)
     .                  + 2*am3**2*c03(s,amf2,amf2,am,am,am3)+1) /4
      cc2p(s,am,am3,amu2) = ((bb1(amf2,am3,am,amu2)
     .                        +b02_hdec(s,am,am,amu2)
     .                  + 2*(am3**2-am**2+amf2)*cc1p(s,am,am3,amu2))/2
     .                      - cc20(s,am,am3,amu2)) /(4*amf2-amh2)
      cc2m(s,am,am3,amu2) = (-bb1(amf2,am3,am,amu2)/2
     .                      - cc20(s,am,am3,amu2)) /s
      pi = 4*datan(1.d0)
      cw2 = amw**2/amz**2
      sw2 = 1-cw2
      cw = dsqrt(cw2)
      sw = dsqrt(sw2)
      xmu2 = xmu**2
      amf2 = amf**2
      amfp2 = amfp**2
      amh2 = amh**2
      amz2 = amz**2
      amw2 = amw**2
      af = ai3f/sw/cw/2
      vf = (ai3f-2*qf*sw2)/sw/cw/2
      alph0 = dsqrt(2.d0)*gf*amw**2*sw2/pi
c--dT_1
      dt1 = 0
c--dT_2
      c0 = c03(amh2,amf2,amf2,amf,amf,amz)
      c1p = cc1p(amh2,amf,amz,xmu2)
      c20 = cc20(amh2,amf,amz,xmu2)
      c2p = cc2p(amh2,amf,amz,xmu2)
      c2m = cc2m(amh2,amf,amz,xmu2)
      dt2 = 4*(vf**2-af**2)*(4*c20-1+(4*amf2-amh2)*c2p+amh2*c2m
     .                  +(amh2-4*amf2)*c1p+(2*amf2-amh2/2)*c0)
     .    + 4*amf2*(vf**2+af**2)*(2*c1p-c0)
c--dT_6
c     dt6 = amf2/amw2/4/sw2*(4*c20-1/2.d0+(4*amf2-amh2)*c2p+amh2*c2m)
      dt6 =-amf2/amw2/4/sw2*(4*c20-1/2.d0+(4*amf2-amh2)*c2p+amh2*c2m)
c--dT_3
      c0 = c03(amh2,amf2,amf2,amfp,amfp,amw)
      c1p = cc1p(amh2,amfp,amw,xmu2)
      c20 = cc20(amh2,amfp,amw,xmu2)
      c2p = cc2p(amh2,amfp,amw,xmu2)
      c2m = cc2m(amh2,amfp,amw,xmu2)
c     dt3 = amfp**2/sw2*(c0-2*c1p)
      dt3 =-amfp**2/sw2*(c0-2*c1p)
c--dT_4
c     dt4 =-amfp2/amw2/2/sw2*(4*c20-1/2.d0+(4*amf2-amh2)*c2p+amh2*c2m
      dt4 = amfp2/amw2/2/sw2*(4*c20-1/2.d0+(4*amf2-amh2)*c2p+amh2*c2m
     .    + 2*(amfp2-amf2)*c1p)
c--dT_5
      c0 = c03(amh2,amf2,amf2,amf,amf,amh)
      c1p = cc1p(amh2,amf,amh,xmu2)
      c20 = cc20(amh2,amf,amh,xmu2)
      c2p = cc2p(amh2,amf,amh,xmu2)
      c2m = cc2m(amh2,amf,amh,xmu2)
c     dt5 =-amf2/amw2/4/sw2*(4*c20-1/2.d0+(4*amf2-amh2)*c2p+amh2*c2m
      dt5 = amf2/amw2/4/sw2*(4*c20-1/2.d0+(4*amf2-amh2)*c2p+amh2*c2m
     .    + 4*amf2*c0-8*amf2*c1p)
c--dT_7
      c0 = c03(amh2,amf2,amf2,amw,amw,amfp)
      c1p = cc1p(amh2,amw,amfp,xmu2)
      c20 = cc20(amh2,amw,amfp,xmu2)
      c2p = cc2p(amh2,amw,amfp,xmu2)
      c2m = cc2m(amh2,amw,amfp,xmu2)
      dt7 = -2*amw2/sw2*c1p
c--dT_9
      dt9 = -amh2/2/sw2/amw2*((amf2+amfp2)*c1p-amfp2*c0)
c--dT_10+11
      dt10 = 1/sw2/2*(4*c20-1/2.d0+(4*amf2-amh2)*c2p+amh2*c2m
c    .     + 2*(amh2-amf2)*c1p-amfp2*(c0-2*c1p))
     .     + 2*(amh2-amf2)*c1p+amfp2*(c0-2*c1p))
c--dT_8
      c0 = c03(amh2,amf2,amf2,amz,amz,amf)
      c1p = cc1p(amh2,amz,amf,xmu2)
      c20 = cc20(amh2,amz,amf,xmu2)
      c2p = cc2p(amh2,amz,amf,xmu2)
      c2m = cc2m(amh2,amz,amf,xmu2)
      dt8 = -8*amz2*((vf**2+af**2)*c1p-(vf**2-af**2)*c0)
c--dT_12+13
      dt12 = 1/sw2/cw2/4*(4*c20-1/2.d0+(4*amf2-amh2)*c2p+amh2*c2m
c    .     + 2*(amh2-amf2)*c1p+amfp2*(c0-2*c1p))
     .     + 2*(amh2-amf2)*c1p+amf2*(c0-2*c1p))
c--dT_14
      dt14 = amh2/4/sw2/amw2*amf2*(c0-2*c1p)
c--dT_15
      c0 = c03(amh2,amf2,amf2,amh,amh,amf)
      c1p = cc1p(amh2,amh,amf,xmu2)
      dt15 = -3*amh2/4/sw2/amw2*amf2*(c0+2*c1p)
      dhffloop_hdec = alph0/4/pi*2*(dt1+dt2+dt3+dt4+dt5+dt6+dt7+dt8+dt9
     .                             +dt10+dt12+dt14+dt15)
      return
      end

      double precision function dhffct_hdec(amh,amf,amfp,qf,ai3f,xmu)
      implicit double precision (a-h,o-z)
      common/param_hdec/gf,alph,amtau,ammuon,amz,amw
      common/masses_hdec/ams,amc,amb,amt
      pi = 4*datan(1.d0)
      cw2 = amw**2/amz**2
      sw2 = 1-cw2
      cw = dsqrt(cw2)
      sw = dsqrt(sw2)
      alph0 = dsqrt(2.d0)*gf*amw**2*sw2/pi
      add = -7*b02_hdec(0.d0,amw,amw,xmu**2)-2/3.d0
      call elwself_hdec(amh,amf,amfp,qf,ai3f,xmu,sigw0,deltamw2,
     .        deltamz2,siggz0,pigam0,pigamf0,sigph,sigfs,sigpfs,sigpfv)
c--del v/v
      dummy = pigamf0 - deltamz2/amz**2
     .      - (cw2-sw2)/sw2*(deltamz2/amz**2-deltamw2/amw**2)
     .      - sigph + add
      dvbv = alph0/4/pi * dummy/2
c--del Z^f_V + del m_f/m_f
      dummy = sigfs-2*amf**2*(sigpfs+sigpfv)
      dzpdmbm = alph0/4/pi * dummy
c--CT_tot
      dhffct_hdec = 2*(dzpdmbm + dvbv)
      return
      end

      double precision function delta_r_hdec(amh,xmu0)
      implicit double precision (a-h,o-z)
      common/param_hdec/gf,alph,amtau,ammuon,amz,amw
      common/masses_hdec/ams,amc,amb,amt
      pi = 4*datan(1.d0)
      cw2 = amw**2/amz**2
      sw2 = 1-cw2
      cw = dsqrt(cw2)
      sw = dsqrt(sw2)
      alph0 = dsqrt(2.d0)*gf*amw**2*sw2/pi
      xmu = xmu0
c----------------------------
c--dummy parameters
      amf = amb
      amfp = amt
      qf = -1/3.d0
      ai3f = -1/2.d0
c----------------------------
      boxes = (6+(7-4*sw2)/2/sw2*dlog(cw2))/sw2
      call elwself_hdec(amh,amf,amfp,qf,ai3f,xmu,
     .                  sigw0,deltamw2,deltamz2,siggz0,pigam0,
     .                  pigamf0,sigph,sigfs,sigpfs,sigpfv)
c--Total sum
      hatsigw0 = sigw0 - deltamw2
     .         + amw**2*(cw2/sw2*(deltamw2/amw**2-deltamz2/amz**2)
     .                  +2*cw/sw*siggz0/amz**2+pigam0)
      dummy = hatsigw0/amw**2 + boxes
      delta_r_hdec = alph0/4/pi * dummy
      return
      end

      subroutine elwself_hdec(amh,amf0,amfp,qf0,ai3f0,xmu,
     .                        sigw0,deltamw2,deltamz2,siggz0,pigam0,
     .                        pigamf0,sigph,sigfs,sigpfs,sigpfv)
      implicit double precision (a-h,o-z)
      common/param_hdec/gf,alph,amtau,ammuon,amz,amw
      common/masses_hdec/ams,amc,amb,amt
      bb1(s,am1,am2,xmu2) = (am2**2-am1**2)/2/s
     .                    * (b02_hdec(s,am1,am2,xmu2)
     .                      -b02_hdec(0.d0,am1,am2,xmu2))
     .                    - b02_hdec(s,am1,am2,xmu2)/2
      bb1p(s,am1,am2,xmu2) = (am1**2-am2**2)/2/s**2
     .                     * (b02_hdec(s,am1,am2,xmu2)
     .                       -b02_hdec(0.d0,am1,am2,xmu2))
     .                     + (am2**2-am1**2-s)/2/s
     .                     * bp02_hdec(s,am1,am2,xmu2)
      amel = 0.510998910d-3
      amup = 0.066d0
      amdo = 0.066d0
      ams0 = 0.150d0
      amc0 = 1.20d0
      amb0 = 4.30d0
      pi = 4*datan(1.d0)
      cw2 = amw**2/amz**2
      sw2 = 1-cw2
      cw = dsqrt(cw2)
      sw = dsqrt(sw2)
c--sigma^W(0)
c  leptons
      amf = amel
      dume =-(1/sw2/3*(amf**2/2*b02_hdec(0.d0,0.d0,amf,xmu**2)
     .                +amf**2*b02_hdec(0.d0,amf,amf,xmu**2)
     .                +amf**4/2*bp02_hdec(0.d0,0.d0,amf,xmu**2)))
c     dume = 0
      amf = ammuon
      dumm =-(1/sw2/3*(amf**2/2*b02_hdec(0.d0,0.d0,amf,xmu**2)
     .                +amf**2*b02_hdec(0.d0,amf,amf,xmu**2)
     .                +amf**4/2*bp02_hdec(0.d0,0.d0,amf,xmu**2)))
      amf = amtau
      dumt =-(1/sw2/3*(amf**2/2*b02_hdec(0.d0,0.d0,amf,xmu**2)
     .                +amf**2*b02_hdec(0.d0,amf,amf,xmu**2)
     .                +amf**4/2*bp02_hdec(0.d0,0.d0,amf,xmu**2)))
      duml = dume+dumm+dumt
c  quarks
      amp = amup
      amm = amdo
      dum1 =-(1/sw2/3*((amp**2+amm**2)/2*b02_hdec(0.d0,amp,amm,xmu**2)
     .           +amp**2*b02_hdec(0.d0,amp,amp,xmu**2)
     .           +amm**2*b02_hdec(0.d0,amm,amm,xmu**2)
     .           +(amp**2-amm**2)**2/2*bp02_hdec(0.d0,amp,amm,xmu**2)))
      amp = amc
      amm = ams
      dum2 =-(1/sw2/3*((amp**2+amm**2)/2*b02_hdec(0.d0,amp,amm,xmu**2)
     .           +amp**2*b02_hdec(0.d0,amp,amp,xmu**2)
     .           +amm**2*b02_hdec(0.d0,amm,amm,xmu**2)
     .           +(amp**2-amm**2)**2/2*bp02_hdec(0.d0,amp,amm,xmu**2)))
      amp = amt
      amm = amb
      dum3 =-(1/sw2/3*((amp**2+amm**2)/2*b02_hdec(0.d0,amp,amm,xmu**2)
     .           +amp**2*b02_hdec(0.d0,amp,amp,xmu**2)
     .           +amm**2*b02_hdec(0.d0,amm,amm,xmu**2)
     .           +(amp**2-amm**2)**2/2*bp02_hdec(0.d0,amp,amm,xmu**2)))
      dumq = 3*(dum1+dum2+dum3)
c  bosons
      dumb =-(2/3.d0*(2*amw**2*b02_hdec(0.d0,amw,0.d0,xmu**2)
     .               -2*amw**2*b02_hdec(0.d0,amw,amw,xmu**2)
     .               -amw**4*bp02_hdec(0.d0,amw,0.d0,xmu**2))
     .       +1/sw2/12*(((16*cw2+54-10/cw2)*amw**2)
     .                  *b02_hdec(0.d0,amw,amz,xmu**2)
     .                -(16*cw2+2)*(amw**2*b02_hdec(0.d0,amw,amw,xmu**2)
     .                            +amz**2*b02_hdec(0.d0,amz,amz,xmu**2))
     .                -(8*cw2+1)*(amw**2-amz**2)**2
     .                        *bp02_hdec(0.d0,amw,amz,xmu**2))
     .     +1/sw2/12*((2*amh**2-10*amw**2)*b02_hdec(0.d0,amw,amh,xmu**2)
     .                -2*amw**2*b02_hdec(0.d0,amw,amw,xmu**2)
     .                -2*amh**2*b02_hdec(0.d0,amh,amh,xmu**2)
     .                -(amw**2-amh**2)**2
     .                        *bp02_hdec(0.d0,amw,amh,xmu**2)))
      sigw0 = duml+dumq+dumb
c--delta M_W^2
      s = amw**2
c  leptons
      amf = amel
      dume =-(1/sw2/3*(-(s-amf**2/2)*b02_hdec(s,0.d0,amf,xmu**2)+s/3
     .                +amf**2*b02_hdec(0.d0,amf,amf,xmu**2)
     .                +amf**4/2/s*(b02_hdec(s,0.d0,amf,xmu**2)
     .                            -b02_hdec(0.d0,0.d0,amf,xmu**2))))
      amf = ammuon
      dumm =-(1/sw2/3*(-(s-amf**2/2)*b02_hdec(s,0.d0,amf,xmu**2)+s/3
     .                +amf**2*b02_hdec(0.d0,amf,amf,xmu**2)
     .                +amf**4/2/s*(b02_hdec(s,0.d0,amf,xmu**2)
     .                            -b02_hdec(0.d0,0.d0,amf,xmu**2))))
      amf = amtau
      dumt =-(1/sw2/3*(-(s-amf**2/2)*b02_hdec(s,0.d0,amf,xmu**2)+s/3
     .                +amf**2*b02_hdec(0.d0,amf,amf,xmu**2)
     .                +amf**4/2/s*(b02_hdec(s,0.d0,amf,xmu**2)
     .                            -b02_hdec(0.d0,0.d0,amf,xmu**2))))
      duml = dume+dumm+dumt
c     write(6,*)dume,dumm,dumt
c  quarks
      amp = amup
      amm = amdo
      dum1 =-(1/sw2/3*(-(s-(amp**2+amm**2)/2)*b02_hdec(s,amp,amm,xmu**2)
     .                +s/3+amp**2*b02_hdec(0.d0,amp,amp,xmu**2)
     .                +amm**2*b02_hdec(0.d0,amm,amm,xmu**2)
     .                +(amp**2-amm**2)**2/2/s
     .                  *(b02_hdec(s,amp,amm,xmu**2)
     .                   -b02_hdec(0.d0,amp,amm,xmu**2))))
      amp = amc
      amm = ams
      dum2 =-(1/sw2/3*(-(s-(amp**2+amm**2)/2)*b02_hdec(s,amp,amm,xmu**2)
     .                +s/3+amp**2*b02_hdec(0.d0,amp,amp,xmu**2)
     .                +amm**2*b02_hdec(0.d0,amm,amm,xmu**2)
     .                +(amp**2-amm**2)**2/2/s
     .                  *(b02_hdec(s,amp,amm,xmu**2)
     .                   -b02_hdec(0.d0,amp,amm,xmu**2))))
      amp = amt
      amm = amb
      dum3 =-(1/sw2/3*(-(s-(amp**2+amm**2)/2)*b02_hdec(s,amp,amm,xmu**2)
     .                +s/3+amp**2*b02_hdec(0.d0,amp,amp,xmu**2)
     .                +amm**2*b02_hdec(0.d0,amm,amm,xmu**2)
     .                +(amp**2-amm**2)**2/2/s
     .                  *(b02_hdec(s,amp,amm,xmu**2)
     .                   -b02_hdec(0.d0,amp,amm,xmu**2))))
      dumq = 3*(dum1+dum2+dum3)
c  bosons
      dumb =-(2/3.d0*((2*amw**2+5*s)*b02_hdec(s,amw,0.d0,xmu**2)
     .               -2*amw**2*b02_hdec(0.d0,amw,amw,xmu**2)
     .               -amw**4/s*(b02_hdec(s,amw,0.d0,xmu**2)
     .                         -b02_hdec(0.d0,amw,0.d0,xmu**2))+s/3)
     .       +1/sw2/12*(((40*cw2-1)*s+(16*cw2+54-10/cw2)*amw**2)
     .                  *b02_hdec(s,amw,amz,xmu**2)
     .                -(16*cw2+2)*(amw**2*b02_hdec(0.d0,amw,amw,xmu**2)
     .                            +amz**2*b02_hdec(0.d0,amz,amz,xmu**2))
     .                +(4*cw2-1)*2*s/3
     .                -(8*cw2+1)*(amw**2-amz**2)**2/s
     .                        *(b02_hdec(s,amw,amz,xmu**2)
     .                         -b02_hdec(0.d0,amw,amz,xmu**2)))
     .      +1/sw2/12*((2*amh**2-10*amw**2-s)*b02_hdec(s,amw,amh,xmu**2)
     .                -2*amw**2*b02_hdec(0.d0,amw,amw,xmu**2)
     .                -2*amh**2*b02_hdec(0.d0,amh,amh,xmu**2)
     .                -(amw**2-amh**2)**2/s
     .                        *(b02_hdec(s,amw,amh,xmu**2)
     .                         -b02_hdec(0.d0,amw,amh,xmu**2))-2*s/3))
      deltamw2 = duml+dumq+dumb
c--delta M_Z^2
      s = amz**2
c  leptons
      ef = -1
      ai3f = -1/2.d0
      af = ai3f/sw/cw/2
      vf = (ai3f-2*ef*sw2)/sw/cw/2
      amf = amel
      dume = 4/3.d0*(2*af**2*(s*b02_hdec(s,0.d0,0.d0,xmu**2)-s/3)
     .      +(vf**2+af**2)*((s+2*amf**2)*b02_hdec(s,amf,amf,xmu**2)
     .                     -2*amf**2*b02_hdec(0.d0,amf,amf,xmu**2)-s/3)
     .      -3/sw2/cw2/8*amf**2*b02_hdec(s,amf,amf,xmu**2))
      amf = ammuon
      dumm = 4/3.d0*(2*af**2*(s*b02_hdec(s,0.d0,0.d0,xmu**2)-s/3)
     .      +(vf**2+af**2)*((s+2*amf**2)*b02_hdec(s,amf,amf,xmu**2)
     .                     -2*amf**2*b02_hdec(0.d0,amf,amf,xmu**2)-s/3)
     .      -3/sw2/cw2/8*amf**2*b02_hdec(s,amf,amf,xmu**2))
      amf = amtau
      dumt = 4/3.d0*(2*af**2*(s*b02_hdec(s,0.d0,0.d0,xmu**2)-s/3)
     .      +(vf**2+af**2)*((s+2*amf**2)*b02_hdec(s,amf,amf,xmu**2)
     .                     -2*amf**2*b02_hdec(0.d0,amf,amf,xmu**2)-s/3)
     .      -3/sw2/cw2/8*amf**2*b02_hdec(s,amf,amf,xmu**2))
      duml = dume+dumm+dumt
c  quarks
      ef = 2/3.d0
      ai3f = 1/2.d0
      af = ai3f/sw/cw/2
      vf = (ai3f-2*ef*sw2)/sw/cw/2
      amf = amup
      dumu = 4/3.d0*((vf**2+af**2)*(
     .               (s+2*amf**2)*b02_hdec(s,amf,amf,xmu**2)
     .              -2*amf**2*b02_hdec(0.d0,amf,amf,xmu**2)-s/3)
     .              -3/sw2/cw2/8*amf**2*b02_hdec(s,amf,amf,xmu**2))
      amf = amc
      dumc = 4/3.d0*((vf**2+af**2)*(
     .               (s+2*amf**2)*b02_hdec(s,amf,amf,xmu**2)
     .              -2*amf**2*b02_hdec(0.d0,amf,amf,xmu**2)-s/3)
     .              -3/sw2/cw2/8*amf**2*b02_hdec(s,amf,amf,xmu**2))
      amf = amt
      dumt = 4/3.d0*((vf**2+af**2)*(
     .               (s+2*amf**2)*b02_hdec(s,amf,amf,xmu**2)
     .              -2*amf**2*b02_hdec(0.d0,amf,amf,xmu**2)-s/3)
     .              -3/sw2/cw2/8*amf**2*b02_hdec(s,amf,amf,xmu**2))
      ef = -1/3.d0
      ai3f = -1/2.d0
      af = ai3f/sw/cw/2
      vf = (ai3f-2*ef*sw2)/sw/cw/2
      amf = amdo
      dumd = 4/3.d0*((vf**2+af**2)*(
     .               (s+2*amf**2)*b02_hdec(s,amf,amf,xmu**2)
     .              -2*amf**2*b02_hdec(0.d0,amf,amf,xmu**2)-s/3)
     .              -3/sw2/cw2/8*amf**2*b02_hdec(s,amf,amf,xmu**2))
      amf = ams
      dums = 4/3.d0*((vf**2+af**2)*(
     .               (s+2*amf**2)*b02_hdec(s,amf,amf,xmu**2)
     .              -2*amf**2*b02_hdec(0.d0,amf,amf,xmu**2)-s/3)
     .              -3/sw2/cw2/8*amf**2*b02_hdec(s,amf,amf,xmu**2))
      amf = amb
      dumb = 4/3.d0*((vf**2+af**2)*(
     .               (s+2*amf**2)*b02_hdec(s,amf,amf,xmu**2)
     .              -2*amf**2*b02_hdec(0.d0,amf,amf,xmu**2)-s/3)
     .              -3/sw2/cw2/8*amf**2*b02_hdec(s,amf,amf,xmu**2))
      dumq = 3*(dumu+dumd+dumc+dums+dumt+dumb)
c  bosons
      dumb =-(2*(((18*cw2**2+2*cw2-1/2.d0)*s
     .        +(24*cw2**2+16*cw2-10)*amw**2)*b02_hdec(s,amw,amw,xmu**2)
     .        -(24*cw2**2-8*cw2+2)*amw**2*b02_hdec(0.d0,amw,amw,xmu**2)
     .        +(4*cw2-1)*s/3)
     .       +(2*amh**2-10*amz**2-s)*b02_hdec(s,amz,amh,xmu**2)
     .       -2*amz**2*b02_hdec(0.d0,amz,amz,xmu**2)
     .       -2*amh**2*b02_hdec(0.d0,amh,amh,xmu**2)
     .       -(amz**2-amh**2)**2/s
     .       *(b02_hdec(s,amz,amh,xmu**2)-b02_hdec(0.d0,amz,amh,xmu**2))
     .       -2*s/3)/12/cw2/sw2
      deltamz2 = duml+dumq+dumb
c--Sigma^gamma^Z(0)
      siggz0 = 2*amw**2/sw/cw*b02_hdec(0.d0,amw,amw,xmu**2)
c--Pi^gamma(0)
c  leptons
      ef = -1
      amf = amel
      dume = 4/3.d0*ef**2*(-1/3.d0 + b02_hdec(0.D0,amf,amf,xmu**2)
     .                    + 2*amf**2*bp02_hdec(0.d0,amf,amf,xmu**2))
      amf = ammuon
      dumm = 4/3.d0*ef**2*(-1/3.d0 + b02_hdec(0.D0,amf,amf,xmu**2)
     .                    + 2*amf**2*bp02_hdec(0.d0,amf,amf,xmu**2))
      amf = amtau
      dumt = 4/3.d0*ef**2*(-1/3.d0 + b02_hdec(0.D0,amf,amf,xmu**2)
     .                    + 2*amf**2*bp02_hdec(0.d0,amf,amf,xmu**2))
      duml = dume+dumm+dumt
c  quarks
      ef = 2/3.d0
      amf = amup
      dumu = 4/3.d0*ef**2*(-1/3.d0 + b02_hdec(0.D0,amf,amf,xmu**2)
     .                    + 2*amf**2*bp02_hdec(0.d0,amf,amf,xmu**2))
      amf = amc0
      dumc = 4/3.d0*ef**2*(-1/3.d0 + b02_hdec(0.D0,amf,amf,xmu**2)
     .                    + 2*amf**2*bp02_hdec(0.d0,amf,amf,xmu**2))
      amf = amt
      dumt = 4/3.d0*ef**2*(-1/3.d0 + b02_hdec(0.D0,amf,amf,xmu**2)
     .                    + 2*amf**2*bp02_hdec(0.d0,amf,amf,xmu**2))
      ef = -1/3.d0
      amf = amdo
      dumd = 4/3.d0*ef**2*(-1/3.d0 + b02_hdec(0.D0,amf,amf,xmu**2)
     .                    + 2*amf**2*bp02_hdec(0.d0,amf,amf,xmu**2))
      amf = ams0
      dums = 4/3.d0*ef**2*(-1/3.d0 + b02_hdec(0.D0,amf,amf,xmu**2)
     .                    + 2*amf**2*bp02_hdec(0.d0,amf,amf,xmu**2))
      amf = amb0
      dumb = 4/3.d0*ef**2*(-1/3.d0 + b02_hdec(0.D0,amf,amf,xmu**2)
     .                    + 2*amf**2*bp02_hdec(0.d0,amf,amf,xmu**2))
      dumq = 3*(dumu+dumd+dumc+dums+dumt+dumb)
c  bosons
      dumb =-3*b02_hdec(0.d0,amw,amw,xmu**2)
     .     - 4*amw**2*bp02_hdec(0.d0,amw,amw,xmu**2)
      pigam0 = duml+dumq+dumb
      pigamf0 = duml+dumq
c--Sig'_H(MH^2)
      s = amh**2
c  leptons
      amf = amel
      if(s.eq.4*amf**2)then
       dume = 1/sw2/amw**2*amf**2*b02_hdec(s,amf,amf,xmu**2)/2
      else
       dume = 1/sw2/amw**2*amf**2*(b02_hdec(s,amf,amf,xmu**2)/2
     .                    + (s/2-2*amf**2)*bp02_hdec(s,amf,amf,xmu**2))
      endif
      amf = ammuon
      if(s.eq.4*amf**2)then
       dumm = 1/sw2/amw**2*amf**2*b02_hdec(s,amf,amf,xmu**2)/2
      else
       dumm = 1/sw2/amw**2*amf**2*(b02_hdec(s,amf,amf,xmu**2)/2
     .                    + (s/2-2*amf**2)*bp02_hdec(s,amf,amf,xmu**2))
      endif
      amf = amtau
      if(s.eq.4*amf**2)then
       dumt = 1/sw2/amw**2*amf**2*b02_hdec(s,amf,amf,xmu**2)/2
      else
       dumt = 1/sw2/amw**2*amf**2*(b02_hdec(s,amf,amf,xmu**2)/2
     .                    + (s/2-2*amf**2)*bp02_hdec(s,amf,amf,xmu**2))
      endif
      duml = dume+dumm+dumt
c  quarks
      amf = amup
      if(s.eq.4*amf**2)then
       dumu = 1/sw2/amw**2*amf**2*b02_hdec(s,amf,amf,xmu**2)/2
      else
       dumu = 1/sw2/amw**2*amf**2*(b02_hdec(s,amf,amf,xmu**2)/2
     .                    + (s/2-2*amf**2)*bp02_hdec(s,amf,amf,xmu**2))
      endif
      amf = amdo
      if(s.eq.4*amf**2)then
       dumd = 1/sw2/amw**2*amf**2*b02_hdec(s,amf,amf,xmu**2)/2
      else
       dumd = 1/sw2/amw**2*amf**2*(b02_hdec(s,amf,amf,xmu**2)/2
     .                    + (s/2-2*amf**2)*bp02_hdec(s,amf,amf,xmu**2))
      endif
      amf = amc
      if(s.eq.4*amf**2)then
       dumc = 1/sw2/amw**2*amf**2*b02_hdec(s,amf,amf,xmu**2)/2
      else
       dumc = 1/sw2/amw**2*amf**2*(b02_hdec(s,amf,amf,xmu**2)/2
     .                    + (s/2-2*amf**2)*bp02_hdec(s,amf,amf,xmu**2))
      endif
      amf = ams
      if(s.eq.4*amf**2)then
       dums = 1/sw2/amw**2*amf**2*b02_hdec(s,amf,amf,xmu**2)/2
      else
       dums = 1/sw2/amw**2*amf**2*(b02_hdec(s,amf,amf,xmu**2)/2
     .                    + (s/2-2*amf**2)*bp02_hdec(s,amf,amf,xmu**2))
      endif
      amf = amb
      if(s.eq.4*amf**2)then
       dumb = 1/sw2/amw**2*amf**2*b02_hdec(s,amf,amf,xmu**2)/2
      else
       dumb = 1/sw2/amw**2*amf**2*(b02_hdec(s,amf,amf,xmu**2)/2
     .                    + (s/2-2*amf**2)*bp02_hdec(s,amf,amf,xmu**2))
      endif
      amf = amt
      if(s.eq.4*amf**2)then
       dumt = 1/sw2/amw**2*amf**2*b02_hdec(s,amf,amf,xmu**2)/2
      else
       dumt = 1/sw2/amw**2*amf**2*(b02_hdec(s,amf,amf,xmu**2)/2
     .                    + (s/2-2*amf**2)*bp02_hdec(s,amf,amf,xmu**2))
      endif
      dumq = 3*(dumu+dumd+dumc+dums+dumt+dumb)
c  bosons
      dumb = 1/sw2/amw**2*(-amw**2*b02_hdec(s,amw,amw,xmu**2)
     .     + (-amw**2*s+3*amw**4+amh**4/4)*bp02_hdec(s,amw,amw,xmu**2)
     .     - amz**2/2*b02_hdec(s,amz,amz,xmu**2)
     .     + (-amz**2*s+3*amz**4+amh**4/4)/2*bp02_hdec(s,amz,amz,xmu**2)
     .     + 9*amh**4/8*bp02_hdec(s,amh,amh,xmu**2))
      sigph = duml+dumq+dumb
c--Sig_fS(mf^2)   (without QED)
      amf = amf0
      qf = qf0
      ai3f = ai3f0
      af = ai3f/sw/cw/2
      vf = (ai3f-2*qf*sw2)/sw/cw/2
      s = amf**2
      dum1 = -((vf**2-af**2)*(4*b02_hdec(s,amf,amz,xmu**2)-2)
     .        +amfp**2/2/sw2/amw**2*b02_hdec(s,amfp,amw,xmu**2)
     .        +amf**2/4/sw2/amw**2*(b02_hdec(s,amf,amz,xmu**2)
     .                             -b02_hdec(s,amf,amh,xmu**2)))
      dum2 = 0
      sigfs = dum1+dum2
      dum1 = -((vf**2-af**2)*4*bp02_hdec(s,amf,amz,xmu**2)
     .        +amfp**2/2/sw2/amw**2*bp02_hdec(s,amfp,amw,xmu**2)
     .        +amf**2/4/sw2/amw**2*(bp02_hdec(s,amf,amz,xmu**2)
     .                             -bp02_hdec(s,amf,amh,xmu**2)))
      dum2 = 0
      sigpfs = dum1+dum2
      bp11 = bb1p(s,amf,amz,xmu**2)
      bp12 = bb1p(s,amfp,amw,xmu**2)
      bp13 = bb1p(s,amf,amh,xmu**2)
      dum1 = -((vf**2+af**2)*2*bp11+(amf**2+amfp**2)/sw2/amw**2/4*bp12
     .        +1/sw2/4*2*bp12 + amf**2/4/sw2/amw**2*(bp13+bp11))
      dum2 = 0
      sigpfv = dum1+dum2
      return
      end

