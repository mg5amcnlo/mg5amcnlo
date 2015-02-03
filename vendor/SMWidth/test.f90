PROGRAM test
  USE ReadParam
  USE Func_PSI
  USE WDecay
  USE ZDecay
  USE tDecay
  USE H0Decay
  IMPLICIT NONE
  REAL(KIND(1d0)),EXTERNAL::Width_W2ud_EW,Width_W2lv_EW,Width_Z2uu_EW,Width_Z2dd_EW,Width_Z2ll_EW,Width_Z2vv_EW,&
       Width_Z2udxWm,Width_Z2veepWm
  REAL(KIND(1d0))::reslo,resqcd,resqed,resqcdqed,argarg,resmass
  CHARACTER(len=20)::paramcard
  paramcard="param_card.dat"
  CALL ReadParamCard(paramcard)
  reslo=SMWWidth(0,0)
  resqcd=SMWWidth(1,0)
  resqed=SMWWidth(0,1)
  resqcdqed=SMWWidth(1,1)
  WRITE(*,*)" "
  WRITE(*,*)"=============================="
  WRITE(*,*)"|  W Boson Width (GeV)       |"
  WRITE(*,*)"------------------------------"
  WRITE(*,*)"|                            |"
  WRITE(*,100)"|          LO = ",reslo," |"
  WRITE(*,100)"|     NLO QCD = ",resqcd," |"
  WRITE(*,100)"|     NLO QED = ",resqed," |"
  WRITE(*,100)"| NLO QCD+QED = ",resqcdqed," |"
  WRITE(*,*)"|                            |"
  WRITE(*,*)"=============================="
  reslo=SMZWidth(0,0)
  resqcd=SMZWidth(1,0)
  resqed=SMZWidth(0,1)
  resqcdqed=SMZWidth(1,1)
  WRITE(*,*)" "
  WRITE(*,*)"=============================="
  WRITE(*,*)"|  Z Boson Width (GeV)       |"
  WRITE(*,*)"------------------------------"
  WRITE(*,*)"|                            |"
  WRITE(*,100)"|          LO = ",reslo," |"
  WRITE(*,100)"|     NLO QCD = ",resqcd," |"
  WRITE(*,100)"|     NLO QED = ",resqed," |"
  WRITE(*,100)"| NLO QCD+QED = ",resqcdqed," |"
  WRITE(*,*)"|                            |"
  WRITE(*,*)"=============================="
  reslo=SMtWidth(0,0,.FALSE.,.FALSE.) ! without b mass effect
  resqcd=SMtWidth(1,0,.FALSE.,.FALSE.) ! without finite width effect
  resqed=SMtWidth(0,1,.FALSE.,.FALSE.)
  resqcdqed=SMtWidth(1,1,.FALSE.,.FALSE.)
  WRITE(*,*)" "
  WRITE(*,*)"=============================="
  WRITE(*,*)"|  Top Quark Width (GeV)     |"
  WRITE(*,*)"------------------------------"
  WRITE(*,*)"|                            |"
  WRITE(*,100)"|          LO = ",reslo," |"
  WRITE(*,100)"|     NLO QCD = ",resqcd," |"
  WRITE(*,100)"|     NLO QED = ",resqed," |"
  WRITE(*,100)"| NLO QCD+QED = ",resqcdqed," |"
  WRITE(*,*)"|                            |"
  WRITE(*,*)"=============================="
  reslo=SMHWidth(0)
  WRITE(*,*)" "
  WRITE(*,*)"=============================="
  WRITE(*,*)"|  Higgs Boson Width (GeV)   |"
  WRITE(*,*)"------------------------------"
  WRITE(*,*)"|                            |"
  WRITE(*,100)"|      HDecay = ",reslo," |"
  WRITE(*,*)"|                            |"
  WRITE(*,*)"=============================="
100 FORMAT(1X,A16,E12.6,A2)
END PROGRAM test
