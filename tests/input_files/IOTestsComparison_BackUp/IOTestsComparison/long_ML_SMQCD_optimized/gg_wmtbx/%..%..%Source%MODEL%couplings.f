ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c      written by the UFO converter
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      SUBROUTINE COUP()

      IMPLICIT NONE
      DOUBLE PRECISION PI, ZERO
      LOGICAL READLHA
      PARAMETER  (PI=3.141592653589793D0)
      PARAMETER  (ZERO=0D0)
      REAL*16 MP__PI, MP__ZERO
      PARAMETER (MP__PI=3.1415926535897932384626433832795E0_16)
      PARAMETER (MP__ZERO=0E0_16)
      INCLUDE 'mp_input.inc'
      INCLUDE 'mp_coupl.inc'

      INCLUDE 'input.inc'
      INCLUDE 'coupl.inc'
      READLHA = .TRUE.
      INCLUDE 'intparam_definition.inc'
      INCLUDE 'mp_intparam_definition.inc'

      CALL COUP1()
C     
C     couplings needed to be evaluated points by points
C     
      CALL COUP2()
      CALL COUP3()
      CALL MP_COUP2()
      CALL MP_COUP3()

      RETURN
      END

      SUBROUTINE UPDATE_AS_PARAM()

      IMPLICIT NONE
      DOUBLE PRECISION PI, ZERO
      LOGICAL READLHA
      PARAMETER  (PI=3.141592653589793D0)
      PARAMETER  (ZERO=0D0)
      INCLUDE 'input.inc'
      INCLUDE 'coupl.inc'
      READLHA = .FALSE.

      INCLUDE 'intparam_definition.inc'


C     
C     couplings needed to be evaluated points by points
C     
      CALL COUP2()
      CALL COUP3()

      RETURN
      END

      SUBROUTINE MP_UPDATE_AS_PARAM()

      IMPLICIT NONE
      LOGICAL READLHA
      REAL*16 MP__PI, MP__ZERO
      PARAMETER (MP__PI=3.1415926535897932384626433832795E0_16)
      PARAMETER (MP__ZERO=0E0_16)
      INCLUDE 'mp_input.inc'
      INCLUDE 'mp_coupl.inc'

      INCLUDE 'input.inc'
      INCLUDE 'coupl.inc'
      INCLUDE 'actualize_mp_ext_params.inc'
      READLHA = .FALSE.
      INCLUDE 'mp_intparam_definition.inc'


C     
C     couplings needed to be evaluated points by points
C     
      CALL MP_COUP2()
      CALL MP_COUP3()

      RETURN
      END

