ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c      written by the UFO converter
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      SUBROUTINE COUP1()

      IMPLICIT NONE

      DOUBLE PRECISION PI
      PARAMETER  (PI=3.141592653589793D0)
      INCLUDE 'input.inc'
      INCLUDE 'coupl.inc'
      REAL*16 MP__PI
      PARAMETER (MP__PI=3.1415926535897932384626433832795E0_16)
      INCLUDE 'mp_input.inc'
      INCLUDE 'mp_coupl.inc'


      INCLUDE 'model_functions.inc'
      END
