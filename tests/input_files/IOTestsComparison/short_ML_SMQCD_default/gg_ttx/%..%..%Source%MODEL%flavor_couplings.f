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
          INTEGER :: PARTNER(0)
          INTEGER :: PARTNER2(0)
          TYPE(COUPPTR) :: VAL(0)
          END TYPE FLV_COUPLING
          END MODULE MODEL_OBJECT


          SUBROUTINE INIT_FLV_COUPLINGS()
          USE MODEL_OBJECT
          IMPLICIT NONE

          INCLUDE 'coupl.inc'



          END SUBROUTINE INIT_FLV_COUPLINGS

