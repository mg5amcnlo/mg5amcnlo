ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c      written by the UFO converter
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc


      MODULE MODEL_OBJECT
      TYPE COUPPTR  ! needed to have an array of pointer
        SEQUENCE
        DOUBLE COMPLEX, POINTER :: P
        END TYPE COUPPTR

        TYPE FLV_COUPLING
          SEQUENCE
          INTEGER :: PARTNER(4)
          INTEGER :: PARTNER2(4)
          TYPE(COUPPTR) :: VAL(4)
          END TYPE FLV_COUPLING
          END MODULE MODEL_OBJECT


          SUBROUTINE INIT_FLV_COUPLINGS()
          USE MODEL_OBJECT
          IMPLICIT NONE

          INCLUDE 'coupl.inc'


          FLV_3 % PARTNER(1) = 1
          FLV_3 % PARTNER2(1) = 1
          FLV_3 % VAL(1) %P  =>  GC_FFV_0
          FLV_3 % PARTNER(3) = 3
          FLV_3 % PARTNER2(3) = 3
          FLV_3 % VAL(3) %P  =>  GC_FFV_0
          FLV_3 % PARTNER(2) = 2
          FLV_3 % PARTNER2(2) = 2
          FLV_3 % VAL(2) %P  =>  GC_FFV_2
          FLV_3 % PARTNER(4) = 4
          FLV_3 % PARTNER2(4) = 4
          FLV_3 % VAL(4) %P  =>  GC_51
          FLV_4 % PARTNER(1) = 1
          FLV_4 % PARTNER2(1) = 1
          FLV_4 % VAL(1) %P  =>  GC_FFV_1
          FLV_4 % PARTNER(3) = 3
          FLV_4 % PARTNER2(3) = 3
          FLV_4 % VAL(3) %P  =>  GC_FFV_1
          FLV_4 % PARTNER(2) = 2
          FLV_4 % PARTNER2(2) = 2
          FLV_4 % VAL(2) %P  =>  GC_FFV_3
          FLV_4 % PARTNER(4) = 4
          FLV_4 % PARTNER2(4) = 4
          FLV_4 % VAL(4) %P  =>  GC_58
          FLV_5 % PARTNER(2) = 1
          FLV_5 % PARTNER2(1) = 2
          FLV_5 % VAL(2) %P  =>  GC_100
          FLV_5 % PARTNER(4) = 3
          FLV_5 % PARTNER2(3) = 4
          FLV_5 % VAL(4) %P  =>  GC_100
          FLV_6 % PARTNER(1) = 2
          FLV_6 % PARTNER2(2) = 1
          FLV_6 % VAL(1) %P  =>  GC_100
          FLV_6 % PARTNER(3) = 4
          FLV_6 % PARTNER2(4) = 3
          FLV_6 % VAL(3) %P  =>  GC_100
          END SUBROUTINE INIT_FLV_COUPLINGS

