C     f2py wrappers. Each entry comes in two flavors: a FLAVOR(NEXTERNAL)
C     variant (back-compat) that resolves the flavor index via
C     GET_FLAVOR_INDEX, and a *_IDX variant taking the flavor index directly.
C     The Python dispatch wrapper (flavor_dispatch.py) picks the right one.
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
      INTEGER MG5_0_GET_FLAVOR_INDEX

      CALL MG5_0_SMATRIXHEL(P,HEL,
     &     MG5_0_GET_FLAVOR_INDEX(FLAVOR),ANS)
      END

      SUBROUTINE PY_MG5_0_SMATRIXHEL_IDX(P,HEL,FLAV_IDX,ANS)
      IMPLICIT NONE
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=4)
CF2PY INTENT(OUT) :: ANS
CF2PY INTENT(IN) :: HEL
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
CF2PY INTENT(IN) :: FLAV_IDX
      REAL*8 P(0:3,NEXTERNAL),ANS
      INTEGER HEL
      INTEGER FLAV_IDX
      CALL MG5_0_SMATRIXHEL(P,HEL,FLAV_IDX,ANS)
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
      INTEGER MG5_0_GET_FLAVOR_INDEX
CF2PY INTENT(OUT) :: ANS
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
CF2PY INTENT(IN) :: FLAVOR(NEXTERNAL)
      call MG5_0_SMATRIX(P,
     &     MG5_0_GET_FLAVOR_INDEX(FLAVOR),ANS)
      END

      SUBROUTINE PY_MG5_0_SMATRIX_IDX(P,FLAV_IDX,ANS)
      IMPLICIT NONE
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=4)
      REAL*8 P(0:3,NEXTERNAL),ANS
      INTEGER FLAV_IDX
CF2PY INTENT(OUT) :: ANS
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
CF2PY INTENT(IN) :: FLAV_IDX
      call MG5_0_SMATRIX(P,FLAV_IDX,ANS)
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
      INTEGER MG5_0_GET_FLAVOR_INDEX
      PY_MG5_0_MATRIX = MG5_0_MATRIX(P,NHEL,IC,
     &     MG5_0_GET_FLAVOR_INDEX(FLAVOR))
      END

      REAL*8 FUNCTION PY_MG5_0_MATRIX_IDX(P,NHEL,IC,FLAV_IDX)
      IMPLICIT NONE
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=4)
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
      INTEGER FLAV_IDX
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
CF2PY INTENT(IN) :: NHEL(NEXTERNAL)
CF2PY INTENT(IN) :: IC(NEXTERNAL)
CF2PY INTENT(IN) :: FLAV_IDX
      real*8 MG5_0_MATRIX
      PY_MG5_0_MATRIX_IDX = MG5_0_MATRIX(P,NHEL,IC,FLAV_IDX)
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

      SUBROUTINE PY_MG5_0_GET_value_idx(P, ALPHAS, NHEL,
     &     FLAV_IDX, ANS)
      IMPLICIT NONE
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=4)
      REAL*8 P(0:3,NEXTERNAL),ANS
      INTEGER NHEL
      INTEGER FLAV_IDX
      DOUBLE PRECISION ALPHAS
CF2PY INTENT(OUT) :: ANS
CF2PY INTENT(IN) :: NHEL
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
CF2PY INTENT(IN) :: ALPHAS
CF2PY INTENT(IN) :: FLAV_IDX
      call MG5_0_GET_value_idx(P, ALPHAS, NHEL, FLAV_IDX, ANS)
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
     &     ALLOW_HEL, N_COMB, FLAVOR, ALPHAS, SCALE2, INTER)
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
CF2PY double precision, intent(in) :: SCALE2
CF2PY double complex, intent(out), dimension(N_COMB*(N_COMB+1)/2) :: INTER
C     ARGUMENTS
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=4)
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER N_CHANGING, N_COMB
      INTEGER POS(*)
      INTEGER ALLOW_HEL(*)
      INTEGER FLAVOR(NEXTERNAL)
      DOUBLE PRECISION ALPHAS, SCALE2
C     INTER must be declared with its explicit size (not INTER(*)): f2py reads
C     this Fortran declaration to size the intent(out) array, and an assumed-size
C     (*) makes it allocate a zero-length buffer, corrupting memory at runtime.
      DOUBLE COMPLEX INTER(N_COMB*(N_COMB+1)/2)
C     GET_DENSITY takes (..., ALPHAS, SCALE2, INTER): SCALE2 must be passed,
C     otherwise INTER lands on the SCALE2 slot and the real INTER pointer is
C     undefined, corrupting memory when the density matrix is written.
      CALL MG5_0_GET_DENSITY(P, POS, N_CHANGING, ALLOW_HEL,
     &     N_COMB, FLAVOR, ALPHAS, SCALE2, INTER)
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
