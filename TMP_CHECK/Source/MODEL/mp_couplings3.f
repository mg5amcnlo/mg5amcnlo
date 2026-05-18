ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c      written by the UFO converter
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      SUBROUTINE MP_COUP3( )
      USE MODEL_OBJECT
      IMPLICIT NONE

      INCLUDE 'model_functions.inc'
      REAL*16 MP__PI, MP__ZERO
      PARAMETER (MP__PI=3.1415926535897932384626433832795E0_16)
      PARAMETER (MP__ZERO=0E0_16)
      INCLUDE 'mp_input.inc'
      INCLUDE 'mp_coupl.inc'

      MP__GC_5 = MP__MDL_COMPLEXI*MP__G
      MP__UVWFCT_B_0_1EPS = MP_COND(CMPLX(MP__MDL_MB,KIND=16)
     $ ,CMPLX(0.000000E+00_16,KIND=16),CMPLX(-((MP__MDL_G__EXP__2)
     $ /(2.000000E+00_16*1.600000E+01_16*MP__PI**2))*3.000000E+00_16
     $ *MP__MDL_CF,KIND=16))
      MP__R2_UUA = (2.000000E+00_16*(MP__MDL_EE*MP__MDL_COMPLEXI)
     $ /3.000000E+00_16)*MP__MDL_R2MIXEDFACTOR_FIN_
      MP__R2_DDZ_V3 = (-(MP__MDL_EE*MP__MDL_COMPLEXI*MP__MDL_SW)
     $ /(6.000000E+00_16*MP__MDL_CW))*MP__MDL_R2MIXEDFACTOR_FIN_
      MP__R2_UUZ_V2 = ((MP__MDL_CW*MP__MDL_EE*MP__MDL_COMPLEXI)
     $ /(2.000000E+00_16*MP__MDL_SW))*MP__MDL_R2MIXEDFACTOR_FIN_
      MP__UVWFCT_T_0 = MP_COND(CMPLX(MP__MDL_MT,KIND=16)
     $ ,CMPLX(0.000000E+00_16,KIND=16),CMPLX(-((MP__MDL_G__EXP__2)
     $ /(2.000000E+00_16*1.600000E+01_16*MP__PI**2))*MP__MDL_CF
     $ *(4.000000E+00_16-3.000000E+00_16
     $ *MP_REGLOG(CMPLX((MP__MDL_MT__EXP__2/MP__MDL_MU_R__EXP__2)
     $ ,KIND=16))),KIND=16))
      END
