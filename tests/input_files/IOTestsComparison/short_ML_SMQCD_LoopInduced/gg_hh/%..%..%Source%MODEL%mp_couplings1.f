ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c      written by the UFO converter
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      SUBROUTINE MP_COUP1( )

      IMPLICIT NONE

      INCLUDE 'model_functions.inc'
      REAL*16 MP__PI, MP__ZERO
      PARAMETER (MP__PI=3.1415926535897932384626433832795E0_16)
      PARAMETER (MP__ZERO=0E0_16)
      INCLUDE 'mp_input.inc'
      INCLUDE 'mp_coupl.inc'

      MP__GC_30 = -6.000000E+00_16*MP__MDL_COMPLEXI*MP__MDL_LAM
     $ *MP__MDL_V
      MP__GC_33 = -((MP__MDL_COMPLEXI*MP__MDL_YB)/MP__MDL_SQRT__2)
      MP__GC_37 = -((MP__MDL_COMPLEXI*MP__MDL_YT)/MP__MDL_SQRT__2)
      END
