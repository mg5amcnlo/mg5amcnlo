ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c      written by the UFO converter
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      SUBROUTINE COUP3()

      IMPLICIT NONE

      DOUBLE PRECISION PI
      PARAMETER  (PI=3.141592653589793D0)
      INCLUDE 'input.inc'
      INCLUDE 'coupl.inc'

      INCLUDE 'model_functions.inc'
      UV_TMASS_1EPS = (COND(DCMPLX(MDL_MT),DCMPLX(0.000000D+00)
     $ ,DCMPLX(MDL_COMPLEXI*((MDL_G__EXP__2)/(1.600000D+01*PI**2))
     $ *3.000000D+00*MDL_CF*MDL_MT)))
      UVWFCT_B_0_1EPS = COND(DCMPLX(MDL_MB),DCMPLX(0.000000D+00)
     $ ,DCMPLX(-((MDL_G__EXP__2)/(2.000000D+00*1.600000D+01*PI**2))
     $ *3.000000D+00*MDL_CF))
      UVWFCT_G_2_1EPS = COND(DCMPLX(MDL_MT),DCMPLX(0.000000D+00)
     $ ,DCMPLX(-((MDL_G__EXP__2)/(2.000000D+00*4.800000D+01*PI**2))
     $ *4.000000D+00*MDL_TF))
      END
