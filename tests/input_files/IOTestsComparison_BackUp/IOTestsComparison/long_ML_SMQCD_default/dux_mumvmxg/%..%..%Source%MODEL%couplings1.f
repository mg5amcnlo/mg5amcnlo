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
      GC_47 = (MDL_EE*MDL_COMPLEXI*MDL_CONJG__CKM33)/(MDL_SW*MDL_SQRT__
     $ 2)
      MP__GC_47 = (MP__MDL_EE*MP__MDL_COMPLEXI*MP__MDL_CONJG__CKM33)
     $ /(MP__MDL_SW*MP__MDL_SQRT__2)
      END
