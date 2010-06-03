ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c      written by the UFO converter
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      SUBROUTINE COUP1()
      
      IMPLICIT NONE
      
      INCLUDE 'input.inc'
      INCLUDE 'coupl.inc'
      
      GC_1 = -(EE*COMPLEXI)/3.000000D+00
      GC_2 = (2.000000D+00*EE*COMPLEXI)/3.000000D+00
      GC_3 = -(EE*COMPLEXI)
      GC_7 = -(CW*COMPLEXI*GW)
      GC_8 = -(COMPLEXI*GW__EXP__2)
      GC_9 = CW__EXP__2*COMPLEXI*GW__EXP__2
      GC_10 = -6.000000D+00*COMPLEXI*LAM
      GC_11 = (EE__EXP__2*COMPLEXI)/(2.000000D+00*SW__EXP__2)
      GC_12 = (EE*COMPLEXI)/(SW*SQRT__2)
      GC_13 = (CKM11*EE*COMPLEXI)/(SW*SQRT__2)
      GC_14 = (CKM12*EE*COMPLEXI)/(SW*SQRT__2)
      GC_15 = (CKM21*EE*COMPLEXI)/(SW*SQRT__2)
      GC_16 = (CKM22*EE*COMPLEXI)/(SW*SQRT__2)
      GC_17 = -(CW*EE*COMPLEXI)/(2.000000D+00*SW)
      GC_18 = (CW*EE*COMPLEXI)/(2.000000D+00*SW)
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
      END
