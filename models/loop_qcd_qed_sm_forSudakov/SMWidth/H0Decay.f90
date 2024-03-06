MODULE H0Decay
  USE ParamModule
  IMPLICIT NONE
CONTAINS
  FUNCTION SMHWidth(idummy)
    ! (qcdord,qedord):
    ! (0,0): LO
    ! (1,0): NLO QCD
    ! (0,1): NLO QED
    ! (1,1): NLO QCD+QED
    ! Higgs Decay is special, since there is no LO 1 > 2 process with yb=0
    ! yf/MH mass correction, (yf/MH)**2*alpha H > f f~
    ! alpha QED correction, alpha**2 H > V V* > V f f,alpha**3 H > A A,Z A
    ! alphas QCD correction, alphas**2*alpha H > g g
    IMPLICIT NONE
    !INTEGER,INTENT(IN)::qcdord,qedord
    INTEGER,INTENT(IN)::idummy
    REAL(KIND(1d0))::SMHWidth
    IF(.NOT.print_banner)THEN
       INCLUDE "banner.inc"
       print_banner=.TRUE.
    ENDIF
    SMHWidth=HDECAY_HWidth(0)
    RETURN
  END FUNCTION SMHWidth
  
  ! use Hdecay to calculate higgs decay width
  ! see hep-ph/9704448
  FUNCTION HDECAY_HWidth(idummy)
    IMPLICIT NONE
    INTEGER,INTENT(IN)::idummy
    REAL(KIND(1d0))::HDECAY_HWidth
    REAL(KIND(1d0))::AMSM,AMA,AML,AMH,AMCH,AMAR,TGBET,AMABEG,AMAEND
    INTEGER::NMA
    ! branching ratios and width
    REAL(KIND(1d0))::SMBRB,SMBRL,SMBRM,SMBRS,SMBRC,SMBRT,SMBRG,&
         SMBRGA,SMBRZGA,SMBRW,SMBRZ,SMWDTH
    COMMON/HMASS_HDEC/AMSM,AMA,AML,AMH,AMCH,AMAR
    COMMON/WIDTHSM_HDEC/SMBRB,SMBRL,SMBRM,SMBRS,SMBRC,SMBRT,SMBRG,&
         SMBRGA,SMBRZGA,SMBRW,SMBRZ,SMWDTH
    CALL WRITE_HDECAY_INPUT
    CALL READ_HDEC(TGBET,AMABEG,AMAEND,NMA) ! read from hdecay.in
    AMAR = AMABEG
    AMSM = AMAR
    AMA = AMAR
    CALL HDEC(TGBET) ! calculate decay width
    HDECAY_HWidth=SMWDTH ! look for 
                         ! WRITE(NSA,20)AMSM,SMBRB,SMBRL,SMBRM,SMBRS,SMBRC,SMBRT
                         ! in hdecay.f subrutine write_hdec
    RETURN
  END FUNCTION HDECAY_HWidth

  SUBROUTINE WRITE_HDECAY_INPUT
    IMPLICIT NONE
    INTEGER::iunit=1993
    OPEN(UNIT=iunit,FILE="./hdecay/hdecay.in")
    WRITE(iunit,101)"SLHAIN   =",0
    WRITE(iunit,101)"SLHAOUT  =",0
    WRITE(iunit,101)"COUPVAR  =",0
    WRITE(iunit,101)"HIGGS    =",0
    WRITE(iunit,101)"SM4      =",0
    WRITE(iunit,101)"FERMPHOB =",0
    WRITE(iunit,101)"2HDM     =",0
    WRITE(iunit,101)"MODEL    =",1
    WRITE(iunit,100)"TGBET    =",30.D0
    WRITE(iunit,100)"MABEG    =",Decay_MH
    WRITE(iunit,100)"MAEND    =",1000.D0
    WRITE(iunit,101)"NMA      =",1
    WRITE(iunit,*)"********************* hMSSM (MODEL = 10) *********************************"
    WRITE(iunit,100)"MHL      =",125.D0
    WRITE(iunit,*)"**************************************************************************"
    WRITE(iunit,100)"ALS(MZ)  =",alphaQCD2
    WRITE(iunit,100)"MSBAR(2) =",0.100D0
    WRITE(iunit,100)"MC       =",Finite_MC
    WRITE(iunit,100)"MB       =",Finite_MB
    WRITE(iunit,100)"MT       =",Decay_MT
    WRITE(iunit,100)"MTAU     =",Finite_MTAU
    WRITE(iunit,100)"MMUON    =",Finite_MMU
    WRITE(iunit,100)"1/ALPHA  =",137.0359997D0
    WRITE(iunit,100)"GF       =",Decay_Gf
    WRITE(iunit,100)"GAMW     =",2.08856D0
    WRITE(iunit,100)"GAMZ     =",2.49581D0
    WRITE(iunit,100)"MZ       =",Decay_MZ
    WRITE(iunit,100)"MW       =",Decay_MW
    ! CKM Matrix Element
    WRITE(iunit,100)"VTB      =",0.9991D0
    WRITE(iunit,100)"VTS      =",0.0404D0
    WRITE(iunit,100)"VTD      =",0.00867D0
    WRITE(iunit,100)"VCB      =",0.0412D0
    WRITE(iunit,100)"VCS      =",0.97344D0
    WRITE(iunit,100)"VCD      =",0.22520D0
    WRITE(iunit,100)"VUB      =",0.00351D0
    WRITE(iunit,100)"VUS      =",0.22534D0
    WRITE(iunit,100)"VUD      =",0.97427D0
    WRITE(iunit,*)"********************* 4TH GENERATION *************************************"
    WRITE(iunit,*)"  SCENARIO FOR ELW. CORRECTIONS TO H -> GG (EVERYTHING IN GEV):"
    WRITE(iunit,*)"  GG_ELW = 1: MTP = 500    MBP = 450    MNUP = 375    MEP = 450"
    WRITE(iunit,*)"  GG_ELW = 2: MBP = MNUP = MEP = 600    MTP = MBP+50*(1+LOG(M_H/115)/5)"
    WRITE(iunit,*)"                        "
    WRITE(iunit,101)"GG_ELW   =",1
    WRITE(iunit,100)"MTP      =",500.D0
    WRITE(iunit,100)"MBP      =",450.D0
    WRITE(iunit,100)"MNUP     =",375.D0
    WRITE(iunit,100)"MEP      =",450.D0
    WRITE(iunit,*)"************************** 2 Higgs Doublet Model *************************"
    WRITE(iunit,*)"  TYPE: 1 (I), 2 (II), 3 (lepton-specific), 4 (flipped)"
    WRITE(iunit,*)" PARAM: 1 (masses), 2 (lambda_i)"
    WRITE(iunit,*)"   "
    WRITE(iunit,101)"PARAM    =",2
    WRITE(iunit,101)"TYPE     =",1
    WRITE(iunit,*)"********************"
    WRITE(iunit,100)"TGBET2HDM=",1.0D0
    WRITE(iunit,100)"M_12^2   =",25600.D0
    WRITE(iunit,*)"******************** PARAM=1:"
    WRITE(iunit,100)"ALPHA_H  =",-0.14D0
    WRITE(iunit,100)"MHL      =",125.D0
    WRITE(iunit,100)"MHH      =",210.D0
    WRITE(iunit,100)"MHA      =",130.D0
    WRITE(iunit,100)"MH+-     =",130.D0
    WRITE(iunit,*)"******************** PARAM=2:"
    WRITE(iunit,100)"LAMBDA1  =",2.6885665050462264D0
    WRITE(iunit,100)"LAMBDA2  =",0.000156876030254505681D0
    WRITE(iunit,100)"LAMBDA3  =",0.46295674052962260D0
    WRITE(iunit,100)"LAMBDA4  =",0.96605498373771792D0
    WRITE(iunit,100)"LAMBDA5  =",-0.88138084173680198D0
    WRITE(iunit,*)"**************************************************************************"
    WRITE(iunit,100)"SUSYSCALE=",1000.D0
    WRITE(iunit,100)"MU       =",1000.D0
    WRITE(iunit,100)"M2       =",1000.D0
    WRITE(iunit,100)"MGLUINO  =",1000.D0
    WRITE(iunit,100)"MSL1     =",1000.D0
    WRITE(iunit,100)"MER1     =",1000.D0
    WRITE(iunit,100)"MQL1     =",1000.D0
    WRITE(iunit,100)"MUR1     =",1000.D0
    WRITE(iunit,100)"MDR1     =",1000.D0
    WRITE(iunit,100)"MSL      =",1000.D0
    WRITE(iunit,100)"MER      =",1000.D0
    WRITE(iunit,100)"MSQ      =",1000.D0
    WRITE(iunit,100)"MUR      =",1000.D0
    WRITE(iunit,100)"MDR      =",1000.D0
    WRITE(iunit,100)"AL       =",2450.D0
    WRITE(iunit,100)"AU       =",2450.D0
    WRITE(iunit,100)"AD       =",2450.D0
    WRITE(iunit,101)"NNLO (M) =",0
    WRITE(iunit,101)"ON-SHELL =",0
    WRITE(iunit,101)"ON-SH-WZ =",0
    WRITE(iunit,101)"IPOLE    =",0
    WRITE(iunit,101)"OFF-SUSY =",0
    WRITE(iunit,101)"INDIDEC  =",0
    WRITE(iunit,101)"NF-GG    =",5
    WRITE(iunit,101)"IGOLD    =",0
    WRITE(iunit,100)"MPLANCK  =",2.4D18
    WRITE(iunit,100)"MGOLD    =",1.D-13
    WRITE(iunit,*)"******************* VARIATION OF HIGGS COUPLINGS *************************"
    WRITE(iunit,101)"ELWK     =",0
    WRITE(iunit,100)"CW       =",1.D0
    WRITE(iunit,100)"CZ       =",1.D0
    WRITE(iunit,100)"Ctau     =",1.D0
    WRITE(iunit,100)"Cmu      =",1.D0
    WRITE(iunit,100)"Ct       =",1.D0
    WRITE(iunit,100)"Cb       =",1.D0
    WRITE(iunit,100)"Cc       =",1.D0
    WRITE(iunit,100)"Cs       =",1.D0
    WRITE(iunit,100)"Cgaga    =",0.D0
    WRITE(iunit,100)"Cgg      =",0.D0
    WRITE(iunit,100)"CZga     =",0.D0
    WRITE(iunit,*)"********************* 4TH GENERATION *************************************"
    WRITE(iunit,100)"Ctp      =",1.D0
    WRITE(iunit,100)"Cbp      =",1.D0
    WRITE(iunit,100)"Cnup     =",1.D0
    WRITE(iunit,100)"Cep      =",1.D0
    WRITE(iunit,*)"      "
    CLOSE(UNIT=iunit)
100 FORMAT(A10,G30.20)
101 FORMAT(A10,I30)
    RETURN
  END SUBROUTINE WRITE_HDECAY_INPUT
END MODULE H0Decay
