      FUNCTION GET_COLOR(IPDG)
      IMPLICIT NONE
      INTEGER GET_COLOR, IPDG

      IF(IPDG.EQ.-4)THEN
        GET_COLOR=-3
        RETURN
      ELSE IF(IPDG.EQ.4)THEN
        GET_COLOR=3
        RETURN
      ELSE IF(IPDG.EQ.23)THEN
        GET_COLOR=1
        RETURN
      ELSE IF(IPDG.EQ.7)THEN
C       This is dummy particle used in multiparticle vertices
        GET_COLOR=2
        RETURN
      ELSE
        WRITE(*,*)'Error: No color given for pdg ',IPDG
        STOP 1
        RETURN
      ENDIF
      END

      FUNCTION GET_SPIN(IPDG)
      IMPLICIT NONE
      INTEGER GET_SPIN, IPDG

      IF(IPDG.EQ.-4)THEN
        GET_SPIN=2
        RETURN
      ELSE IF(IPDG.EQ.4)THEN
        GET_SPIN=2
        RETURN
      ELSE IF(IPDG.EQ.23)THEN
        GET_SPIN=3
        RETURN
      ELSE IF(IPDG.EQ.7)THEN
C       This is dummy particle used in multiparticle vertices
        GET_SPIN=-2
        RETURN
      ELSE
        WRITE(*,*)'Error: No spin given for pdg ',IPDG
        STOP 1
        RETURN
      ENDIF
      END

