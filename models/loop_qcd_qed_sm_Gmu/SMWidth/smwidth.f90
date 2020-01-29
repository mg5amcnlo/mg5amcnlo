PROGRAM smwidth
  USE ReadParam
  USE Func_PSI
  USE WDecay
  USE ZDecay
  USE tDecay
  USE H0Decay
  IMPLICIT NONE
  REAL(KIND(1d0))::reslo,resqcd,resqed,resqcdqed,argarg,resmass
  CHARACTER(len=100)::paramcard
  INTEGER,PARAMETER::PID_Z=23,PID_W=24,PID_H=25,PID_t=6
  CHARACTER(len=1)::cscheme
  Skip_scheme=.TRUE.
  IF(iargc().NE.3)THEN
     WRITE(*,*)"ERROR: Please input three arguments"
     STOP
  ENDIF
  CALL getarg(1,paramcard)
  CALL getarg(2,identcard)
  CALL getarg(3,cscheme)
  IF(cscheme.EQ."2")THEN
     Decay_scheme=2
  ELSE
     Decay_scheme=1
  ENDIF
  CALL ReadParamCard(paramcard)
  resqcdqed=SMWWidth(1,1)
  WRITE(*,*)" --------------------------"
  WRITE(*,*)" Decay Widths (GeV) in SM: "
  WRITE(*,100)'decay',PID_W,resqcdqed
  resqcdqed=SMZWidth(1,1)
  WRITE(*,100)'decay',PID_Z,resqcdqed
  ! include finite-width effect which is also at NLO EW order
  resqcdqed=SMtWidth(1,1,.FALSE.,.TRUE.)
  WRITE(*,100)'decay',PID_t,resqcdqed
  resqcdqed=SMHWidth(0)
  WRITE(*,100)'decay',PID_H,resqcdqed
  WRITE(*,*)" --------------------------"
  RETURN
100 FORMAT(2X,A5,2X,I2,2X,E12.6)
END PROGRAM smwidth
