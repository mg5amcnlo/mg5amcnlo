ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c      written by the UFO converter
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      SUBROUTINE COUP2()
      
      IMPLICIT NONE
      
      INCLUDE 'input.inc'
      INCLUDE 'coupl.inc'
      
      GC_19 = -(EE*COMPLEXI*SW)/(6.000000D+00*CW)
      GC_20 = (EE*COMPLEXI*SW)/(2.000000D+00*CW)
      GC_21 = -(COMPLEXI*GW*SW)
      GC_22 = -2.000000D+00*CW*COMPLEXI*GW__EXP__2*SW
      GC_23 = COMPLEXI*GW__EXP__2*SW__EXP__2
      GC_24 = (CW*EE*COMPLEXI)/(2.000000D+00*SW) + (EE*COMPLEXI
     $ *SW)/(2.000000D+00*CW)
      GC_25 = EE__EXP__2*COMPLEXI + (CW__EXP__2*EE__EXP__2*COMPLEXI)
     $ /(2.000000D+00*SW__EXP__2) + (EE__EXP__2*COMPLEXI*SW__EXP__2)
     $ /(2.000000D+00*CW__EXP__2)
      GC_26 = -6.000000D+00*COMPLEXI*LAM*V
      GC_27 = (EE__EXP__2*COMPLEXI*V)/(2.000000D+00*SW__EXP__2)
      GC_28 = EE__EXP__2*COMPLEXI*V + (CW__EXP__2*EE__EXP__2*COMPLEXI
     $ *V)/(2.000000D+00*SW__EXP__2) + (EE__EXP__2*COMPLEXI*SW__EXP__2
     $ *V)/(2.000000D+00*CW__EXP__2)
      GC_29 = -((COMPLEXI*YB)/SQRT__2)
      GC_30 = -((COMPLEXI*YC)/SQRT__2)
      GC_31 = -((COMPLEXI*YT)/SQRT__2)
      GC_32 = -((COMPLEXI*YTAU)/SQRT__2)
      GC_33 = (EE*COMPLEXI*CONJG__CKM11)/(SW*SQRT__2)
      END
