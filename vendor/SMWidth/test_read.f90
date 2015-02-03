PROGRAM test_read
  USE ReadParam
  USE Func_PSI
  USE WDecay
  USE ZDecay
  USE tDecay
  USE H0Decay
  IMPLICIT NONE
  REAL(KIND(1d0)),EXTERNAL::Width_W2ud_EW,Width_W2lv_EW,Width_Z2uu_EW,Width_Z2dd_EW,Width_Z2ll_EW,Width_Z2vv_EW,&
       Width_Z2udxWm,Width_Z2veepWm
  REAL(KIND(1d0))::res,argarg
  CALL ReadParamCard
  !res=HDecay_HWidth(0)
  !PRINT *,"Higgs Width = ",res
  !STOP
  res=SMtWidth(0,1)
  PRINT *,"NLO QED:",res
  STOP
  res=SMZWidth(0,0,.FALSE.)
  PRINT *, "LO:",res
  res=SMZWidth(1,0,.FALSE.)
  PRINT *, "NLO QCD:",res
  res=SMZWidth(0,1,.FALSE.)
  PRINT *, "NLO QED:",res
  res=SMZWidth(0,1)
  PRINT *, "NLO QED+Wrad:",res
  res=SMZWidth(1,1,.FALSE.)
  PRINT *, "NLO QCD+QED:",res
  res=SMZWidth(1,1)
  PRINT *, "NLO QCD+QED+Wrad:",res
END PROGRAM test_read
