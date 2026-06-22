ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c      written by the UFO converter
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      FUNCTION GET_COLOR(IPDG)
      IMPLICIT NONE
      INTEGER GET_COLOR, IPDG
      SELECT CASE (IPDG)
      CASE(-82)
        GET_COLOR=8
      CASE(-24)
        GET_COLOR=1
      CASE(-16)
        GET_COLOR=1
      CASE(-15)
        GET_COLOR=1
      CASE(-14)
        GET_COLOR=1
      CASE(-13)
        GET_COLOR=1
      CASE(-12)
        GET_COLOR=1
      CASE(-11)
        GET_COLOR=1
      CASE(-6)
        GET_COLOR=-3
      CASE(-5)
        GET_COLOR=-3
      CASE(-4)
        GET_COLOR=-3
      CASE(-3)
        GET_COLOR=-3
      CASE(-2)
        GET_COLOR=-3
      CASE(-1)
        GET_COLOR=-3
      CASE(1)
        GET_COLOR=3
      CASE(2)
        GET_COLOR=3
      CASE(3)
        GET_COLOR=3
      CASE(4)
        GET_COLOR=3
      CASE(5)
        GET_COLOR=3
      CASE(6)
        GET_COLOR=3
      CASE(11)
        GET_COLOR=1
      CASE(12)
        GET_COLOR=1
      CASE(13)
        GET_COLOR=1
      CASE(14)
        GET_COLOR=1
      CASE(15)
        GET_COLOR=1
      CASE(16)
        GET_COLOR=1
      CASE(21)
        GET_COLOR=8
      CASE(22)
        GET_COLOR=1
      CASE(23)
        GET_COLOR=1
      CASE(24)
        GET_COLOR=1
      CASE(25)
        GET_COLOR=1
      CASE(82)
        GET_COLOR=8
      CASE(7)
C       This is dummy particle used in multiparticle vertices
        GET_COLOR=2
      CASE DEFAULT
        WRITE(*,*)'Error: No color given for pdg ',IPDG
        STOP 1
      END SELECT
      END

      FUNCTION GET_SPIN(IPDG)
      IMPLICIT NONE
      INTEGER GET_SPIN, IPDG
      SELECT CASE (IPDG)
      CASE(-82)
        GET_SPIN=1
      CASE(-24)
        GET_SPIN=3
      CASE(-16)
        GET_SPIN=2
      CASE(-15)
        GET_SPIN=2
      CASE(-14)
        GET_SPIN=2
      CASE(-13)
        GET_SPIN=2
      CASE(-12)
        GET_SPIN=2
      CASE(-11)
        GET_SPIN=2
      CASE(-6)
        GET_SPIN=2
      CASE(-5)
        GET_SPIN=2
      CASE(-4)
        GET_SPIN=2
      CASE(-3)
        GET_SPIN=2
      CASE(-2)
        GET_SPIN=2
      CASE(-1)
        GET_SPIN=2
      CASE(1)
        GET_SPIN=2
      CASE(2)
        GET_SPIN=2
      CASE(3)
        GET_SPIN=2
      CASE(4)
        GET_SPIN=2
      CASE(5)
        GET_SPIN=2
      CASE(6)
        GET_SPIN=2
      CASE(11)
        GET_SPIN=2
      CASE(12)
        GET_SPIN=2
      CASE(13)
        GET_SPIN=2
      CASE(14)
        GET_SPIN=2
      CASE(15)
        GET_SPIN=2
      CASE(16)
        GET_SPIN=2
      CASE(21)
        GET_SPIN=3
      CASE(22)
        GET_SPIN=3
      CASE(23)
        GET_SPIN=3
      CASE(24)
        GET_SPIN=3
      CASE(25)
        GET_SPIN=1
      CASE(82)
        GET_SPIN=1
      CASE(7)
C       This is dummy particle used in multiparticle vertices
        GET_SPIN=-2
      CASE DEFAULT
        WRITE(*,*)'Error: No spin given for pdg ',IPDG
        STOP 1
      END SELECT
      END

