MODULE WDecay
  USE ParamModule
  USE alfas_functions
CONTAINS
  FUNCTION SMWWidth(qcdord,qedord,finitemassin)
    IMPLICIT NONE
    ! (qcdord,qedord):
    ! (0,0): LO
    ! (1,0): NLO QCD
    ! (0,1): NLO QED
    ! (1,1): NLO QCD+QED
    INTEGER,INTENT(IN)::qcdord,qedord
    REAL(KIND(1d0))::SMWWidth
    LOGICAL,INTENT(IN),OPTIONAL::finitemassin
    LOGICAL::finitemass
    REAL(KIND(1d0))::WWidth_LOud,ALPHAS
    IF(.NOT.print_banner)THEN
       INCLUDE "banner.inc"
       print_banner=.TRUE.
    ENDIF
    IF(qcdord.LT.0.OR.qcdord.GT.1)THEN
       WRITE(*,*)"ERROR:Please specify the QCD corr. order to be 0 or 1"
       STOP
    ENDIF
    IF(qedord.LT.0.OR.qedord.GT.1)THEN
       WRITE(*,*)"ERROR:Please specify the QED corr. order to be 0 or 1"
       STOP
    ENDIF
    ! W > u d~
    WWidth_LOud=Decay_e**2*Decay_MW/(16d0*pi*Decay_R_SW2)
    ! W > u d~, c s~, e ve, mu vm, ta vt
    SMWWidth=WWidth_LOud*(2d0+1d0)
    ! include finite quark/lepton mass effect
    IF(PRESENT(finitemassin))THEN
       finitemass=finitemassin
    ELSE
       finitemass=.FALSE. ! it should be turned off with loop_qcd_qed_sm correspondigly
    END IF
    IF(finitemass)THEN
       SMWWidth=SMWWidth+SMWWidth_fmass(Finite_MC,Finite_MTAU,Finite_MMU)
    ENDIF
    IF(qcdord.EQ.0.AND.qedord.EQ.0)RETURN
    IF(qcdord.EQ.1)THEN
       NLOOP=2
       Decay_aS=ALPHAS_MCFM(Decay_MW)
       Decay_gs=DSQRT(4d0*pi*Decay_aS)
       SMWWidth=SMWWidth+WWidth_LOud*2d0*Decay_aS/pi
    ENDIF
    IF(qedord.EQ.1)THEN
       SMWWidth=SMWWidth+SMWWidth_EW(Decay_Scheme)
    ENDIF
    RETURN
  END FUNCTION SMWWidth
  
  FUNCTION SMWWidth_EW(iScheme)
    IMPLICIT NONE
    INTEGER,INTENT(IN)::iScheme ! 1: alpha(MZ) 2: Gmu
    REAL(KIND(1d0))::SMWWidth_EW
    REAL(KIND(1d0)),EXTERNAL::Width_W2ud_EW,Width_W2lv_EW
    ! W+ > u d~ and W+ > c s~
    SMWWidth_EW=2d0*Width_W2ud_EW(iScheme)
    ! W+ > ve e+, W+ > vm m+, W+ > vt tt+
    SMWWidth_EW=SMWWidth_EW+3d0*Width_W2lv_EW(iScheme)
    RETURN
  END FUNCTION SMWWIDTH_EW

  FUNCTION SMWWidth_fmass(mc,mtt,mm)
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::mc,mtt,mm
    REAL(KIND(1d0))::SMWWidth_fmass
    REAL(KIND(1d0)),EXTERNAL::Width_W2cs_qmass,Width_W2lv_lmass
    SMWWidth_fmass=0d0
    IF(mc.GT.0d0)THEN
       SMWWidth_fmass=SMWWidth_fmass+Width_W2cs_qmass(mc,0d0)
    ENDIF
    IF(mtt.GT.0d0)THEN
       SMWWidth_fmass=SMWWidth_fmass+Width_W2lv_lmass(mtt)
    ENDIF
    IF(mm.GT.0d0)THEN
       SMWWidth_fmass=SMWWidth_fmass+Width_W2lv_lmass(mm)
    ENDIF
    RETURN
  END FUNCTION SMWWidth_fmass
END MODULE WDecay
