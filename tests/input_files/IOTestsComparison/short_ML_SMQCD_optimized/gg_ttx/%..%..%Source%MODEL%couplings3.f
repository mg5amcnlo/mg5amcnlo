ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c      written by the UFO converter
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      SUBROUTINE COUP3()

      IMPLICIT NONE
      INCLUDE 'model_functions.inc'

      DOUBLE PRECISION PI, ZERO
      PARAMETER  (PI=3.141592653589793D0)
      PARAMETER  (ZERO=0D0)
      INCLUDE 'input.inc'
      INCLUDE 'coupl.inc'
      UVWFCT_T_0 = COND(DCMPLX(MDL_MT),DCMPLX(0.000000D+00),DCMPLX(
     $ -((MDL_G__EXP__2)/(2.000000D+00*1.600000D+01*PI**2))*MDL_CF
     $ *(4.000000D+00-3.000000D+00*REGLOG(DCMPLX(MDL_MT__EXP__2
     $ /MDL_MU_R__EXP__2)))))
      UVWFCT_G_1 = COND(DCMPLX(MDL_MB),DCMPLX(0.000000D+00)
     $ ,DCMPLX(((MDL_G__EXP__2)/(2.000000D+00*4.800000D+01*PI**2))
     $ *4.000000D+00*MDL_TF*REGLOG(DCMPLX(MDL_MB__EXP__2
     $ /MDL_MU_R__EXP__2))))
      UVWFCT_G_2 = COND(DCMPLX(MDL_MT),DCMPLX(0.000000D+00)
     $ ,DCMPLX(((MDL_G__EXP__2)/(2.000000D+00*4.800000D+01*PI**2))
     $ *4.000000D+00*MDL_TF*REGLOG(DCMPLX(MDL_MT__EXP__2
     $ /MDL_MU_R__EXP__2))))
      END
