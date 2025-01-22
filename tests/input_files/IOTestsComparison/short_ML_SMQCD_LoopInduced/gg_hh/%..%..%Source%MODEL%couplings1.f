ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c      written by the UFO converter
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      SUBROUTINE COUP1( )

      IMPLICIT NONE

      INCLUDE 'model_functions.inc'

      DOUBLE PRECISION PI, ZERO
      PARAMETER  (PI=3.141592653589793D0)
      PARAMETER  (ZERO=0D0)
      INCLUDE 'input.inc'
      INCLUDE 'coupl.inc'
      GC_30 = -6.000000D+00*MDL_COMPLEXI*MDL_LAM*MDL_V
      GC_33 = -((MDL_COMPLEXI*MDL_YB)/MDL_SQRT__2)
      GC_37 = -((MDL_COMPLEXI*MDL_YT)/MDL_SQRT__2)
      END
