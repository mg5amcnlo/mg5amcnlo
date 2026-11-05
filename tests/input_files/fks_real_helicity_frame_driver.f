      PROGRAM REAL_TIE_FRAME
C     Driver of test_real_helicity_sum_survives_a_boost
C     (tests/acceptance_tests/test_cmd_amcatnlo.py).
C     Learn the zero-helicity mask at a point in the real c.m. frame,
C     then compare the sum after a longitudinal boost with all helicities
C     evaluated explicitly. This guards against reintroducing equality
C     inference based on the first point (PR #478).
C     The placeholders between double underscores are filled in by the
C     test from the generated matrix file of the real emission.
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'coupl.inc'
      INTEGER NCOMB,NSQAMPSO
      PARAMETER (NCOMB=__NCOMB__,NSQAMPSO=__NSQAMPSO__)
      INTEGER NHEL(NEXTERNAL,NCOMB),IHEL,I,J,IDEN
      REAL*8 PA(0:3,NEXTERNAL),PB(0:3,NEXTERNAL)
      REAL*8 ANS(0:NSQAMPSO),T(0:NSQAMPSO),HELSUM
__NHEL__
      DATA IDEN/__IDEN__/
      READ(*,*) ((PA(I,J),I=0,3),J=1,NEXTERNAL)
      READ(*,*) ((PB(I,J),I=0,3),J=1,NEXTERNAL)
      CALL SETPARA('param_card.dat')
      G=SQRT(4D0*3.141592653589793D0*0.118D0)
      CALL UPDATE_AS_PARAM()
      CALL SMATRIX__PFX___SPLITORDERS(PA,ANS)
      CALL SMATRIX__PFX___SPLITORDERS(PB,ANS)
      HELSUM=0D0
      DO IHEL=1,NCOMB
        CALL MATRIX___PFX__(PB,NHEL(1,IHEL),T)
        HELSUM=HELSUM+T(0)
      ENDDO
      HELSUM=HELSUM/DBLE(IDEN)
      WRITE(*,'(A,2E25.16)') 'RESULT ',ANS(0),HELSUM
      END


C     SMATRIX<PFX> in the same file refers to it; not called here.
      INTEGER FUNCTION ORDERS_TO_AMP_SPLIT_POS(ORD)
      INTEGER ORD(*)
      ORDERS_TO_AMP_SPLIT_POS=1
      END
