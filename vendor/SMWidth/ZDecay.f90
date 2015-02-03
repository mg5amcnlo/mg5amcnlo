MODULE ZDecay
  USE ParamModule
  USE alfas_functions
CONTAINS
  FUNCTION SMZWidth(qcdord,qedord,Wradin,finitemassin)
    IMPLICIT NONE
    ! (qcdord,qedord):
    ! (0,0): LO
    ! (1,0): NLO QCD
    ! (0,1): NLO QED
    ! (1,1): NLO QCD+QED
    INTEGER,INTENT(IN)::qcdord,qedord
    LOGICAL,INTENT(IN),OPTIONAL::Wradin,finitemassin
    LOGICAL::Wrad,finitemass
    REAL(KIND(1d0))::SMZWidth
    REAL(KIND(1d0))::ZWidth_LOuu,ZWidth_LOdd,ZWidth_LOll,ZWidth_LOvv
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
    ! Z > u u~
    ZWidth_LOuu=Decay_e**2*Decay_MZ*(9d0-24d0*Decay_R_SW2+32d0*Decay_R_SW2**2)&
         /(288d0*Decay_R_CW2*pi*Decay_R_SW2)
    ! Z > d d~
    ZWidth_LOdd=Decay_e**2*Decay_MZ*(9d0-12d0*Decay_R_SW2+8d0*Decay_R_SW2**2)&
         /(288d0*Decay_R_CW2*pi*Decay_R_SW2)
    ! Z > e+ e-
    ZWidth_LOll=Decay_e**2*Decay_MZ*(1d0-4d0*Decay_R_SW2+8d0*Decay_R_SW2**2)&
         /(96d0*Decay_R_CW2*pi*Decay_R_SW2)
    ! Z > ve ve~
    ZWidth_LOvv=Decay_e**2*Decay_MZ/(96d0*Decay_R_CW2*pi*Decay_R_SW2)
    ! Z > u u~, c c~, d d~, s s~, b b~, l+ l-, v v~
    SMZWidth=ZWidth_LOuu*2d0+ZWidth_LOdd*3d0+ZWidth_LOll*3d0+ZWidth_LOvv*3d0
    ! include finite quark/lepton mass effect
    IF(PRESENT(finitemassin))THEN
       finitemass=finitemassin
    ELSE
       finitemass=.FALSE. ! it should be turned off with loop_qcd_qed_sm correspondigly
    END IF
    IF(finitemass)THEN
       SMZWidth=SMZWidth+SMZWidth_fmass(Finite_MC,Finite_MB,Finite_MTAU,Finite_MMU)
    ENDIF
    IF(qcdord.EQ.0.AND.qedord.EQ.0)RETURN
    IF(qcdord.EQ.1)THEN
       NLOOP=2
       Decay_aS=ALPHAS_MCFM(Decay_MZ)
       Decay_gs=DSQRT(4d0*pi*Decay_aS)
       SMZWidth=SMZWidth+(ZWidth_LOuu*2d0+ZWidth_LOdd*3d0)*Decay_aS/pi
    ENDIF
    IF(qedord.EQ.1)THEN
       IF(PRESENT(Wradin))THEN
          Wrad=Wradin
       ELSE
          Wrad=.FALSE. ! it should be turned off as we don't include W radiation in loop_qcd_qed_sm
       END IF
       SMZWidth=SMZWidth+SMZWidth_EW(Decay_Scheme)
       IF(Wrad)THEN
          SMZWidth=SMZWidth+SM_Z2ffW(0)
       ENDIF
    ENDIF
    RETURN
  END FUNCTION SMZWidth
  
  FUNCTION SMZWidth_EW(iScheme)
    IMPLICIT NONE
    INTEGER,INTENT(IN)::iScheme ! 1: alpha(MZ) 2: Gmu
    REAL(KIND(1d0))::SMZWidth_EW
    REAL(KIND(1d0)),EXTERNAL::Width_Z2uu_EW,Width_Z2dd_EW,Width_Z2ll_EW,Width_Z2vv_EW
    ! Z > u u~ and Z > c c~
    SMZWidth_EW=2d0*Width_Z2uu_EW(iScheme)
    ! Z > d d~, Z > s s~, Z > b b~
    SMZWidth_EW=SMZWidth_EW+3d0*Width_Z2dd_EW(iScheme)
    ! Z > e+ e-, Z > m+ m-, Z > tt+ tt-
    SMZWidth_EW=SMZWidth_EW+3d0*Width_Z2ll_EW(iScheme)
    ! Z > ve ve~, Z > vm vm~, Z > vt vt~
    SMZWidth_EW=SMZWidth_EW+3d0*Width_Z2vv_EW(iScheme)
    RETURN
  END FUNCTION SMZWIDTH_EW

  FUNCTION SM_Z2ffW(idummy)
    IMPLICIT NONE
    INTEGER,INTENT(IN)::idummy
    REAL(KIND(1d0))::SM_Z2ffW
    REAL(KIND(1d0)),EXTERNAL::Width_Z2udxWm,Width_Z2veepWm
    SM_Z2ffW=Width_Z2udxWm(0)*2d0*2d0
    SM_Z2ffW=SM_Z2ffW+Width_Z2veepWm(0)*2d0*3d0
    RETURN
  END FUNCTION SM_Z2ffW

  FUNCTION SMZWidth_fmass(mc,mb,mtt,mm)
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::mc,mb,mtt,mm
    REAL(KIND(1d0)),EXTERNAL::Width_Z2cc_cmass,Width_Z2bb_bmass,Width_Z2tautau_taumass
    REAL(KIND(1d0))::SMZWidth_fmass
    SMZWidth_fmass=0d0
    IF(mc.GT.0d0)THEN
       SMZWidth_fmass=SMZWidth_fmass+Width_Z2cc_cmass(mc)
    ENDIF
    IF(mb.GT.0d0)THEN
       SMZWidth_fmass=SMZWidth_fmass+Width_Z2bb_bmass(mc)
    ENDIF
    IF(mtt.GT.0d0)THEN
       SMZWidth_fmass=SMZWidth_fmass+Width_Z2tautau_taumass(mtt)
    ENDIF
    IF(mm.GT.0d0)THEN
       SMZWidth_fmass=SMZWidth_fmass+Width_Z2tautau_taumass(mm)
    ENDIF
    RETURN
  END FUNCTION SMZWidth_fmass
END MODULE ZDecay
