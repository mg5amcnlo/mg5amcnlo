C this is a f2py wrapper for reweight mode at loop-induced level


      SUBROUTINE INITIALISE(PATH)
C     ROUTINE FOR F2PY to read the benchmark point.
      IMPLICIT NONE
      CHARACTER*512 PATH
CF2PY INTENT(IN) :: PATH
      CALL SETPARA(PATH)  !first call to setup the paramaters
      RETURN
      END

      SUBROUTINE CHANGE_PARA(NAME, VALUE)
      IMPLICIT NONE
CF2PY intent(in) :: name
CF2PY intent(in) :: value
 
        CHARACTER*512 NAME
        DOUBLE PRECISION VALUE
        CALL F77_CHANGE_PARA(NAME, VALUE)
 
        RETURN
        END


      SUBROUTINE UPDATE_ALL_COUP()
      IMPLICIT NONE
      call f77_update_all_coup()
      RETURN
      END


      SUBROUTINE SET_MADLOOP_PATH(PATH)
C     Routine to set the path of the folder 'MadLoop5_resources' to
C      MadLoop
      CHARACTER(512) PATH
CF2PY intent(in)::path
      CALL SETMADLOOPPATH(PATH)
      END

      SUBROUTINE SMATRIXHEL(PDGS, PROCID, NPDG, P, ALPHAS, SCALES2,
     $  NHEL, ANS, RETURNCODE)
      IMPLICIT NONE

CF2PY double precision, intent(in), dimension(0:3,npdg) :: p
CF2PY integer, intent(in), dimension(npdg) :: pdgs
CF2PY integer, intent(in):: procid
CF2PY integer, intent(in) :: npdg
CF2PY double precision, intent(out) :: ANS
CF2PY integer, intent(out) :: RETURNCODE
CF2PY double precision, intent(in) :: ALPHAS
CF2PY double precision, intent(in) :: SCALES2

      INTEGER PDGS(*)
      INTEGER NPDG, NHEL, RETURNCODE, PROCID
      DOUBLE PRECISION P(*)
      DOUBLE PRECISION ANS, ALPHAS, SCALES2
      CALL F77_SMATRIXHEL(PDGS, PROCID, NPDG, P,
     & alphas, scales2, nhel, ans, returncode)
      return 
      end

      SUBROUTINE GET_PDG_ORDER(OUT, ALLPROC)
      IMPLICIT NONE
CF2PY INTEGER, intent(out) :: OUT(%(nb_me)i,%(maxpart)i)  
CF2PY INTEGER, intent(out) :: ALLPROC(%(nb_me)i)
      INTEGER OUT(%(nb_me)i,%(maxpart)i)
      INTEGER ALLPROC(%(nb_me)i)

      call f77_get_pdg_order(out, allproc)
      RETURN
      END

      SUBROUTINE GET_PREFIX(PREFIX)
          implicit none
CF2PY CHARACTER*20, intent(out) :: PREFIX(%(nb_me)i)
      character*20 PREFIX(%(nb_me)i)
      call f77_get_prefix(prefix)
      RETURN
      END



      SUBROUTINE %(f2py_prefix)sREFCHOICEP(PREF, PHI, THETA)
          implicit none
CF2PY DOUBLE PRECISION, INTENT(OUT) :: PHI
CF2PY DOUBLE PRECISION INTENT(OUT) :: THETA
CF2PY DOUBLE PRECISION, INTENT(IN) :: PREF(0:3)
          double precision PREF(0:3)
          double precision PHI, THETA

      call f77_refchoicep(PREF, PHI, THETA)
      RETURN
      END

      SUBROUTINE %(f2py_prefix)sROTATIONP(P, PHI, THETA, NEXTERNAL, PROT)
          implicit none
CF2PY DOUBLE PRECISION, INTENT(OUT) :: PROT(0:3, NEXTERNAL)
CF2PY INTEGER, INTENT(IN) :: NEXTERNAL
CF2PY DOUBLE PRECISION, INTENT(IN) :: PHI
CF2PY DOUBLE PRECISION, INTENT(IN) :: THETA
CF2PY DOUBLE PRECISION, INTENT(IN) :: P(0:3, NEXTERNAL)
          double precision P(0:3, NEXTERNAL), PROT(0:3, NEXTERNAL)
          integer NEXTERNAL
          double precision PHI, THETA
      
      call f77_rotationp(P, PHI, THETA, NEXTERNAL, PROT)
      RETURN
      END

      SUBROUTINE %(f2py_prefix)sPY_GET_DENSITY(PDGS, PROCID, P, POS, ALLOW_HEL, ALPHAS, SCALE2, INTER, N_CHANGING, N_COMB, NPDG)

CF2PY double precision, intent(in) :: p
CF2PY integer, intent(in) :: pdgs
CF2PY integer, intent(in) :: procid
CF2PY integer, intent(in) :: pos(N_CHANGING)
CF2PY integer, INTENT(IN) :: ALLOW_HEL(N_CHANGING*N_COMB)
CF2PY double precision INTENT(IN) :: ALPHAS
CF2PY double precision INTENT(IN) :: SCALE2
CF2PY double complex INTENT(OUT), dimension(N_COMB*(N_COMB+1)/2) :: INTER
CF2PY integer, intent(hide), depend(allow_hel, pos) :: N_COMB = len(ALLOW_HEL)/len(pos)
CF2PY integer, intent(hide), depend(pos) :: N_CHANGING = len(pos)
CF2PY integer, intent(hide), depend(pdgs) :: NPDG = len(pdgs)

      INTEGER PDGS(*), N_CHANGING, NPDG
      INTEGER PROCID
      INTEGER POS(N_CHANGING)
      DOUBLE PRECISION ALPHAS, SCALE2
      INTEGER ALLOW_HEL(N_CHANGING*N_COMB)
      DOUBLE COMPLEX INTER(N_COMB*(N_COMB+1)/2) !what value instead of 0:1
      DOUBLE PRECISION P(0:3,*)

      CALL %(f2py_prefix)sF77_DENSITY(PDGS, NPDG, PROCID, P, POS, ALLOW_HEL, ALPHAS, SCALE2, N_CHANGING, N_COMB, INTER)

      RETURN
      END

C     GET_ALL_INTER is not present, it can be added but is not practical to use outside of the main framework because of the way JAMPL_ALL is computed.