ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c      written by the UFO converter
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      SUBROUTINE COUP1( )
      USE MODEL_OBJECT
      IMPLICIT NONE

      INCLUDE 'model_functions.inc'

      DOUBLE PRECISION PI, ZERO
      PARAMETER  (PI=3.141592653589793D0)
      PARAMETER  (ZERO=0D0)
      INCLUDE 'input.inc'
      INCLUDE 'coupl.inc'
      GC_37 = -MDL_EE/(2.000000D+00*MDL_SW)
      GC_51 = (MDL_CW*MDL_EE*MDL_COMPLEXI)/(2.000000D+00*MDL_SW)
      GC_58 = -(MDL_EE*MDL_COMPLEXI*MDL_SW)/(6.000000D+00*MDL_CW)
      GC_60 = -(MDL_CW*MDL_EE)/(2.000000D+00*MDL_SW)-(MDL_EE*MDL_SW)
     $ /(2.000000D+00*MDL_CW)
      GC_68 = -2.000000D+00*MDL_COMPLEXI*MDL_LAM*MDL_VEV
      GC_72 = (MDL_EE__EXP__2*MDL_COMPLEXI*MDL_VEV)/(2.000000D+00
     $ *MDL_SW__EXP__2)
      GC_81 = MDL_EE__EXP__2*MDL_COMPLEXI*MDL_VEV+(MDL_CW__EXP__2
     $ *MDL_EE__EXP__2*MDL_COMPLEXI*MDL_VEV)/(2.000000D+00
     $ *MDL_SW__EXP__2)+(MDL_EE__EXP__2*MDL_COMPLEXI*MDL_SW__EXP__2
     $ *MDL_VEV)/(2.000000D+00*MDL_CW__EXP__2)
      GC_100 = (MDL_EE*MDL_COMPLEXI*MDL_CONJG__CKM1X1)/(MDL_SW
     $ *MDL_SQRT__2)
      GC_FFV_0 = -2.000000D+00*(-(MDL_EE*MDL_COMPLEXI*MDL_SW)
     $ /(6.000000D+00*MDL_CW))
      GC_FFV_1 = 1.000000D+00*(-(MDL_CW*MDL_EE*MDL_COMPLEXI)
     $ /(2.000000D+00*MDL_SW)+(-(MDL_EE*MDL_COMPLEXI*MDL_SW)
     $ /(6.000000D+00*MDL_CW)))
      GC_FFV_2 = 4.000000D+00*(-(MDL_EE*MDL_COMPLEXI*MDL_SW)
     $ /(6.000000D+00*MDL_CW))
      GC_FFV_3 = 1.000000D+00*((MDL_CW*MDL_EE*MDL_COMPLEXI)/(2.000000D
     $ +00*MDL_SW)+(-(MDL_EE*MDL_COMPLEXI*MDL_SW)/(6.000000D+00*MDL_CW)
     $ ))
      END
