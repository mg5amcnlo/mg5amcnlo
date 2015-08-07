ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c      written by the UFO converter
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      SUBROUTINE COUP1()

      IMPLICIT NONE
      INCLUDE 'model_functions.inc'

      DOUBLE PRECISION PI, ZERO
      PARAMETER  (PI=3.141592653589793D0)
      PARAMETER  (ZERO=0D0)
      INCLUDE 'input.inc'
      INCLUDE 'coupl.inc'
      REAL*16 MP__PI, MP__ZERO
      PARAMETER (MP__PI=3.1415926535897932384626433832795E0_16)
      PARAMETER (MP__ZERO=0E0_16)
      INCLUDE 'mp_input.inc'
      INCLUDE 'mp_coupl.inc'

      END
