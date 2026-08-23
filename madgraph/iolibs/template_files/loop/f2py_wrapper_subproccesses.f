%(python_information)s

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


