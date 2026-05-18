ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c      written by the UFO converter
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      SUBROUTINE COUP3( )
      USE MODEL_OBJECT
      IMPLICIT NONE

      INCLUDE 'model_functions.inc'

      DOUBLE PRECISION PI, ZERO
      PARAMETER  (PI=3.141592653589793D0)
      PARAMETER  (ZERO=0D0)
      INCLUDE 'input.inc'
      INCLUDE 'coupl.inc'
      GC_5 = MDL_COMPLEXI*G
      UVWFCT_B_0_1EPS = COND(DCMPLX(MDL_MB),DCMPLX(0.000000D+00)
     $ ,DCMPLX(-((MDL_G__EXP__2)/(2.000000D+00*1.600000D+01*PI**2))
     $ *3.000000D+00*MDL_CF))
      R2_UUA = (2.000000D+00*(MDL_EE*MDL_COMPLEXI)/3.000000D+00)
     $ *MDL_R2MIXEDFACTOR_FIN_
      R2_DDZ_V3 = (-(MDL_EE*MDL_COMPLEXI*MDL_SW)/(6.000000D+00*MDL_CW))
     $ *MDL_R2MIXEDFACTOR_FIN_
      R2_UUZ_V2 = ((MDL_CW*MDL_EE*MDL_COMPLEXI)/(2.000000D+00*MDL_SW))
     $ *MDL_R2MIXEDFACTOR_FIN_
      UVWFCT_T_0 = COND(DCMPLX(MDL_MT),DCMPLX(0.000000D+00),DCMPLX(
     $ -((MDL_G__EXP__2)/(2.000000D+00*1.600000D+01*PI**2))*MDL_CF
     $ *(4.000000D+00-3.000000D+00*REGLOG(DCMPLX(MDL_MT__EXP__2
     $ /MDL_MU_R__EXP__2)))))
      END
