      SUBROUTINE PY_MG5_0_SMATRIXHEL(P,HEL,FLAVOR,ANS)
      IMPLICIT NONE
C
C CONSTANT
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=4)
      INTEGER                 NCOMB
      PARAMETER (             NCOMB=81)
CF2PY INTENT(OUT) :: ANS
CF2PY INTENT(IN) :: HEL
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
CF2PY INTENT(IN) :: FLAVOR(NEXTERNAL)

C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL),ANS
      INTEGER HEL
      INTEGER FLAVOR(NEXTERNAL)

      CALL MG5_0_SMATRIXHEL(P,HEL,FLAVOR,ANS)
      END

      SUBROUTINE PY_MG5_0_SMATRIX(P,FLAVOR,ANS)
C
C
C MadGraph5_aMC@NLO StandAlone Version
C
C Returns amplitude squared summed/avg over colors
c and helicities
c for the point in phase space P(0:3,NEXTERNAL)
C
C Process: w+ w- > w+ w- WEIGHTED<=4
C
      IMPLICIT NONE
C
C CONSTANTS
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=4)
      INTEGER    NINITIAL
      PARAMETER (NINITIAL=2)
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL),ANS
      INTEGER FLAVOR(NEXTERNAL)
CF2PY INTENT(OUT) :: ANS
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
CF2PY INTENT(IN) :: FLAVOR(NEXTERNAL)
      call MG5_0_SMATRIX(P,FLAVOR,ANS)
      END


      REAL*8 FUNCTION PY_MG5_0_MATRIX(P,NHEL,IC,FLAVOR)
C
C
C Returns amplitude squared -- no average over initial state/symmetry factor
c for the point with external lines W(0:6,NEXTERNAL)
C
C Process: w+ w- > w+ w- WEIGHTED<=4
C
      IMPLICIT NONE
C
C CONSTANTS
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=4)
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
      INTEGER FLAVOR(NEXTERNAL)
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
CF2PY INTENT(IN) :: NHEL(NEXTERNAL)
CF2PY INTENT(IN) :: IC(NEXTERNAL)
CF2PY INTENT(IN) :: FLAVOR(NEXTERNAL)
C
C  FUNCTIONS
C
      real*8 MG5_0_MATRIX
      PY_MG5_0_MATRIX = MG5_0_MATRIX(P,NHEL,IC,FLAVOR)
      END

      SUBROUTINE PY_MG5_0_GET_value(P, ALPHAS, NHEL,
     &     FLAVOR, ANS)
      IMPLICIT NONE
C
C CONSTANT
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=4)
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL),ANS
      INTEGER NHEL
      INTEGER FLAVOR(NEXTERNAL)
      DOUBLE PRECISION ALPHAS
CF2PY INTENT(OUT) :: ANS
CF2PY INTENT(IN) :: NHEL
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
CF2PY INTENT(IN) :: ALPHAS
CF2PY INTENT(IN) :: FLAVOR(NEXTERNAL)
      call MG5_0_GET_value(P, ALPHAS, NHEL, FLAVOR, ANS)
      return
      end

      SUBROUTINE PY_MG5_0_INITIALISEMODEL(PATH)
C     ROUTINE FOR F2PY to read the benchmark point.
      IMPLICIT NONE
      CHARACTER*512 PATH
CF2PY INTENT(IN) :: PATH
      call setpara(PATH)  !first call to setup the paramaters
      return
      end

      SUBROUTINE PY_MG5_0_GET_DENSITY(P, POS, N_CHANGING,
     &     ALLOW_HEL, N_COMB, FLAVOR, ALPHAS, INTER)
C     F2PY wrapper around MG5_0_GET_DENSITY so the density-matrix
C     computation is exposed in the standalone matrix2py module.
C     The CF2PY directives mirror the working pattern used by the
C     auto-generated allmatrix2py PY_GET_DENSITY wrapper: they must
C     appear before the Fortran type declarations so that f2py can pick
C     up the per-argument intent/dimension overrides.
      IMPLICIT NONE
CF2PY double precision, intent(in), dimension(0:3,4) :: P
CF2PY integer, intent(in), dimension(*) :: POS
CF2PY integer, intent(in) :: N_CHANGING
CF2PY integer, intent(in), dimension(N_CHANGING*N_COMB) :: ALLOW_HEL
CF2PY integer, intent(in) :: N_COMB
CF2PY integer, intent(in), dimension(4) :: FLAVOR
CF2PY double precision, intent(in) :: ALPHAS
CF2PY double complex, intent(out), dimension(N_COMB*(N_COMB+1)/2) :: INTER
C     ARGUMENTS
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=4)
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER N_CHANGING, N_COMB
      INTEGER POS(*)
      INTEGER ALLOW_HEL(*)
      INTEGER FLAVOR(NEXTERNAL)
      DOUBLE PRECISION ALPHAS
      DOUBLE COMPLEX INTER(*)
      CALL MG5_0_GET_DENSITY(P, POS, N_CHANGING, ALLOW_HEL,
     &     N_COMB, FLAVOR, ALPHAS, INTER)
      RETURN
      END

      LOGICAL FUNCTION PY_MG5_0_IS_BORN_HEL_SELECTED(HELID)
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=4)
C
C     ARGUMENTS
C
      INTEGER HELID
      LOGICAL MG5_0_IS_BORN_HEL_SELECTED
      PY_MG5_0_IS_BORN_HEL_SELECTED = MG5_0_IS_BORN_HEL_SELECTED(HELID)
      RETURN
      END
