C     f2py wrappers. Each entry comes in two flavors: a FLAVOR(NEXTERNAL)
C     variant (back-compat) that resolves the flavor index via
C     GET_FLAVOR_INDEX, and a *_IDX variant taking the flavor index directly.
C     The Python dispatch wrapper (flavor_dispatch.py) picks the right one.
      SUBROUTINE PY_%(proc_prefix)sSMATRIXHEL(P,HEL,FLAVOR,ANS)
      IMPLICIT NONE
C
C CONSTANT
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER                 NCOMB
      PARAMETER (             NCOMB=%(ncomb)d)
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
      INTEGER %(proc_prefix)sGET_FLAVOR_INDEX

      CALL %(proc_prefix)sSMATRIXHEL(P,HEL,
     &     %(proc_prefix)sGET_FLAVOR_INDEX(FLAVOR),ANS)
      END

      SUBROUTINE PY_%(proc_prefix)sSMATRIXHEL_IDX(P,HEL,FLAV_IDX,ANS)
      IMPLICIT NONE
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
CF2PY INTENT(OUT) :: ANS
CF2PY INTENT(IN) :: HEL
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
CF2PY INTENT(IN) :: FLAV_IDX
      REAL*8 P(0:3,NEXTERNAL),ANS
      INTEGER HEL
      INTEGER FLAV_IDX
      CALL %(proc_prefix)sSMATRIXHEL(P,HEL,FLAV_IDX,ANS)
      END

      SUBROUTINE PY_%(proc_prefix)sSMATRIX(P,FLAVOR,ANS)
C
C
C MadGraph5_aMC@NLO StandAlone Version
C
C Returns amplitude squared summed/avg over colors
c and helicities
c for the point in phase space P(0:3,NEXTERNAL)
C
%(process_lines)s
C
      IMPLICIT NONE
C
C CONSTANTS
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER    NINITIAL
      PARAMETER (NINITIAL=%(nincoming)d)
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL),ANS
      INTEGER FLAVOR(NEXTERNAL)
      INTEGER %(proc_prefix)sGET_FLAVOR_INDEX
CF2PY INTENT(OUT) :: ANS
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
CF2PY INTENT(IN) :: FLAVOR(NEXTERNAL)
      call %(proc_prefix)sSMATRIX(P,
     &     %(proc_prefix)sGET_FLAVOR_INDEX(FLAVOR),ANS)
      END

      SUBROUTINE PY_%(proc_prefix)sSMATRIX_IDX(P,FLAV_IDX,ANS)
      IMPLICIT NONE
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      REAL*8 P(0:3,NEXTERNAL),ANS
      INTEGER FLAV_IDX
CF2PY INTENT(OUT) :: ANS
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
CF2PY INTENT(IN) :: FLAV_IDX
      call %(proc_prefix)sSMATRIX(P,FLAV_IDX,ANS)
      END


      REAL*8 FUNCTION PY_%(proc_prefix)sMATRIX(P,NHEL,IC,FLAVOR)
C
C
C Returns amplitude squared -- no average over initial state/symmetry factor
c for the point with external lines W(0:6,NEXTERNAL)
C
%(process_lines)s
C
      IMPLICIT NONE
C
C CONSTANTS
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
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
      real*8 %(proc_prefix)sMATRIX
      INTEGER %(proc_prefix)sGET_FLAVOR_INDEX
      PY_%(proc_prefix)sMATRIX = %(proc_prefix)sMATRIX(P,NHEL,IC,
     &     %(proc_prefix)sGET_FLAVOR_INDEX(FLAVOR))
      END

      REAL*8 FUNCTION PY_%(proc_prefix)sMATRIX_IDX(P,NHEL,IC,FLAV_IDX)
      IMPLICIT NONE
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
      INTEGER FLAV_IDX
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
CF2PY INTENT(IN) :: NHEL(NEXTERNAL)
CF2PY INTENT(IN) :: IC(NEXTERNAL)
CF2PY INTENT(IN) :: FLAV_IDX
      real*8 %(proc_prefix)sMATRIX
      PY_%(proc_prefix)sMATRIX_IDX = %(proc_prefix)sMATRIX(P,NHEL,IC,FLAV_IDX)
      END

      SUBROUTINE PY_%(proc_prefix)sGET_value(P, ALPHAS, NHEL,
     &     FLAVOR, ANS)
      IMPLICIT NONE
C
C CONSTANT
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
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
      call %(proc_prefix)sGET_value(P, ALPHAS, NHEL, FLAVOR, ANS)
      return
      end

      SUBROUTINE PY_%(proc_prefix)sGET_value_idx(P, ALPHAS, NHEL,
     &     FLAV_IDX, ANS)
      IMPLICIT NONE
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      REAL*8 P(0:3,NEXTERNAL),ANS
      INTEGER NHEL
      INTEGER FLAV_IDX
      DOUBLE PRECISION ALPHAS
CF2PY INTENT(OUT) :: ANS
CF2PY INTENT(IN) :: NHEL
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
CF2PY INTENT(IN) :: ALPHAS
CF2PY INTENT(IN) :: FLAV_IDX
      call %(proc_prefix)sGET_value_idx(P, ALPHAS, NHEL, FLAV_IDX, ANS)
      return
      end

      SUBROUTINE PY_%(proc_prefix)sINITIALISEMODEL(PATH)
C     ROUTINE FOR F2PY to read the benchmark point.
      IMPLICIT NONE
      CHARACTER*512 PATH
CF2PY INTENT(IN) :: PATH
      call setpara(PATH)  !first call to setup the paramaters
      return
      end

      SUBROUTINE PY_%(proc_prefix)sGET_DENSITY(P, POS, N_CHANGING,
     &     ALLOW_HEL, N_COMB, FLAVOR, ALPHAS, INTER)
C     F2PY wrapper around %(proc_prefix)sGET_DENSITY so the density-matrix
C     computation is exposed in the standalone matrix2py module.
C     The CF2PY directives mirror the working pattern used by the
C     auto-generated allmatrix2py PY_GET_DENSITY wrapper: they must
C     appear before the Fortran type declarations so that f2py can pick
C     up the per-argument intent/dimension overrides.
      IMPLICIT NONE
CF2PY double precision, intent(in), dimension(0:3,%(nexternal)d) :: P
CF2PY integer, intent(in), dimension(*) :: POS
CF2PY integer, intent(in) :: N_CHANGING
CF2PY integer, intent(in), dimension(N_CHANGING*N_COMB) :: ALLOW_HEL
CF2PY integer, intent(in) :: N_COMB
CF2PY integer, intent(in), dimension(%(nexternal)d) :: FLAVOR
CF2PY double precision, intent(in) :: ALPHAS
CF2PY double complex, intent(out), dimension(N_COMB*(N_COMB+1)/2) :: INTER
C     ARGUMENTS
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER N_CHANGING, N_COMB
      INTEGER POS(*)
      INTEGER ALLOW_HEL(*)
      INTEGER FLAVOR(NEXTERNAL)
      DOUBLE PRECISION ALPHAS
      DOUBLE COMPLEX INTER(*)
      CALL %(proc_prefix)sGET_DENSITY(P, POS, N_CHANGING, ALLOW_HEL,
     &     N_COMB, FLAVOR, ALPHAS, INTER)
      RETURN
      END

      LOGICAL FUNCTION PY_%(proc_prefix)sIS_BORN_HEL_SELECTED(HELID)
      IMPLICIT NONE
C
C     CONSTANTS
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
C
C     ARGUMENTS
C
      INTEGER HELID
      LOGICAL %(proc_prefix)sIS_BORN_HEL_SELECTED
      PY_%(proc_prefix)sIS_BORN_HEL_SELECTED = %(proc_prefix)sIS_BORN_HEL_SELECTED(HELID)
      RETURN
      END
