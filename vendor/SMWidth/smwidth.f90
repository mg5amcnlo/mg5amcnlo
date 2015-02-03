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
  Skip_scheme=.TRUE.
  OPEN(UNIT=5123,FILE="./input.dat")
  READ(5123,*)paramcard
  READ(5123,*)Decay_scheme
  CLOSE(UNIT=5123)
  CALL ReadParamCard(paramcard)
  resqcdqed=SMWWidth(1,1)
  WRITE(*,*)" --------------------------"
  WRITE(*,*)" Decay Widths (GeV) in SM: "
  WRITE(*,100)'decay',PID_W,resqcdqed
  resqcdqed=SMZWidth(1,1)
  WRITE(*,100)'decay',PID_Z,resqcdqed
  resqcdqed=SMtWidth(1,1)
  WRITE(*,100)'decay',PID_t,resqcdqed
  resqcdqed=SMHWidth(0)
  WRITE(*,100)'decay',PID_H,resqcdqed
  WRITE(*,*)" --------------------------"
  RETURN
100 FORMAT(2X,A5,2X,I2,2X,E12.6)
END PROGRAM smwidth
