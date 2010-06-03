ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c      written by the UFO converter
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      SUBROUTINE COUP2()
      
      IMPLICIT NONE
      
      INCLUDE 'input.inc'
      INCLUDE 'coupl.inc'
      
      GC_29 = -((COMPLEXI*YB)/SQRT__2)
      GC_30 = -((COMPLEXI*YC)/SQRT__2)
      GC_31 = -((COMPLEXI*YT)/SQRT__2)
      GC_32 = -((COMPLEXI*YTAU)/SQRT__2)
      GC_33 = (EE*COMPLEXI*CONJG__CKM11)/(SW*SQRT__2)
      GC_34 = (EE*COMPLEXI*CONJG__CKM12)/(SW*SQRT__2)
      GC_35 = (EE*COMPLEXI*CONJG__CKM21)/(SW*SQRT__2)
      GC_36 = (EE*COMPLEXI*CONJG__CKM22)/(SW*SQRT__2)
      END
