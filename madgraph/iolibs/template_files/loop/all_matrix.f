%(python_information)s

      SUBROUTINE f77_INITIALISE(PATH)
C     ROUTINE FOR F2PY to read the benchmark point.
      IMPLICIT NONE
      CHARACTER*512 PATH
CF2PY INTENT(IN) :: PATH
      CALL SETPARA(PATH)  !first call to setup the paramaters
      RETURN
      END

      subroutine f77_CHANGE_PARA(name, value)
      implicit none
CF2PY intent(in) :: name
CF2PY intent(in) :: value

      character*512 name
      double precision value

      include '../Source/MODEL/input.inc'
      include '../Source/MODEL/coupl.inc'
      include '../Source/MODEL/mp_coupl.inc'
      include '../Source/MODEL/mp_input.inc'
      
      SELECT CASE (name)   
         %(parameter_setup)s
         CASE DEFAULT
            write(*,*) 'no parameter matching', name
      END SELECT

      return
      end
      
      subroutine f77_update_all_coup()
      implicit none
      call coup()
      call printout()
      return 
      end
 
 

      subroutine f77_smatrixhel(pdgs, procid, npdg, p, ALPHAS, SCALES2, nhel, ANS, RETURNCODE)
      IMPLICIT NONE

CF2PY double precision, intent(in), dimension(0:3,npdg) :: p
CF2PY integer, intent(in), dimension(npdg) :: pdgs
CF2PY integer, intent(in):: procid
CF2PY integer, intent(in) :: npdg
CF2PY double precision, intent(out) :: ANS
CF2PY integer, intent(out) :: RETURNCODE
CF2PY double precision, intent(in) :: ALPHAS
CF2PY double precision, intent(in) :: SCALES2

     integer pdgs(*)
     integer npdg, nhel, RETURNCODE, procid
     double precision p(*)
     double precision ANS, ALPHAS, PI,SCALES2
    1 continue
%(smatrixhel)s

      return
      end
  
      subroutine f77_get_pdg_order(OUT, ALLPROC)
      IMPLICIT NONE
CF2PY INTEGER, intent(out) :: OUT(%(nb_me)i,%(maxpart)i)  
CF2PY INTEGER, intent(out) :: ALLPROC(%(nb_me)i)
      INTEGER OUT(%(nb_me)i,%(maxpart)i), PDGS(%(nb_me)i,%(maxpart)i)
      INTEGER ALLPROC(%(nb_me)i),PIDs(%(nb_me)i)
      DATA PDGS/ %(pdgs)s /
      DATA PIDS/ %(pids)s /
      OUT=PDGS
      ALLPROC = PIDS
      RETURN
      END
      
      subroutine f77_get_prefix(PREFIX)
      IMPLICIT NONE
CF2PY CHARACTER*20, intent(out) :: PREFIX(%(nb_me)i)
      character*20 PREFIX(%(nb_me)i),PREF(%(nb_me)i)
      DATA PREF / '%(prefix)s'/
      PREFIX = PREF
      RETURN
      END 
  
