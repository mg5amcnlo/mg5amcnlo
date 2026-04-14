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
      GC_30 = -6.000000D+00*MDL_COMPLEXI*MDL_LAM*MDL_V
      MP__GC_30 = -6.000000E+00_16*MP__MDL_COMPLEXI*MP__MDL_LAM
     $ *MP__MDL_V
      GC_33 = -((MDL_COMPLEXI*MDL_YB)/MDL_SQRT__2)
      MP__GC_33 = -((MP__MDL_COMPLEXI*MP__MDL_YB)/MP__MDL_SQRT__2)
      GC_37 = -((MDL_COMPLEXI*MDL_YT)/MDL_SQRT__2)
      MP__GC_37 = -((MP__MDL_COMPLEXI*MP__MDL_YT)/MP__MDL_SQRT__2)
      END
