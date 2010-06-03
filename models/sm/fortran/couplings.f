ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c      written by the UFO converter
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      SUBROUTINE COUP(READLHA)
      
      IMPLICIT NONE
      LOGICAL READLHA
      DOUBLE PRECISION PI
      PARAMETER  (PI=3.141592653589793D0)
      
      INCLUDE 'input.inc'
      INCLUDE 'coupl.inc'
      INCLUDE 'intparam_definition.inc'
      
      
      
      IF (READLHA) THEN
        CALL COUP1()
        CALL COUP2()
        
      ENDIF
c      
c      couplings needed to be evaluated points by points
c      
      CALL COUP3()
      
      RETURN
      END
