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
      END
