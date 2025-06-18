%(python_information)s
  subroutine smatrixhel(pdgs, procid, npdg, p, ALPHAS, SCALE2, nhel, ANS)
  IMPLICIT NONE
C ALPHAS is given at scale2 (SHOULD be different of 0 for loop induced, ignore for LO)  

CF2PY double precision, intent(in), dimension(0:3,npdg) :: p
CF2PY integer, intent(in), dimension(npdg) :: pdgs
CF2PY integer, intent(in):: procid
CF2PY integer, intent(in) :: npdg
CF2PY double precision, intent(out) :: ANS
CF2PY double precision, intent(in) :: ALPHAS
CF2PY double precision, intent(in) :: SCALE2
  integer pdgs(*)
  integer npdg, nhel, procid
  double precision p(*)
  double precision ANS, ALPHAS, PI,SCALE2
  include 'coupl.inc'
  
  
  if (scale2.eq.0)then
       PI = 3.141592653589793D0
       G = 2* DSQRT(ALPHAS*PI)
       CALL UPDATE_AS_PARAM()
  else
       CALL UPDATE_AS_PARAM2(scale2, ALPHAS)
  endif

%(smatrixhel)s

      return
      end
  
      SUBROUTINE INITIALISE(PATH)
C     ROUTINE FOR F2PY to read the benchmark point.
      IMPLICIT NONE
      CHARACTER*512 PATH
CF2PY INTENT(IN) :: PATH
      CALL SETPARA(PATH)  !first call to setup the paramaters
      RETURN
      END
      
      
      subroutine CHANGE_PARA(name, value)
      implicit none
CF2PY intent(in) :: name
CF2PY intent(in) :: value

      character*512 name
      double precision value
      
      %(helreset_def)s

      include '../Source/MODEL/input.inc'
      include '../Source/MODEL/coupl.inc'

      %(helreset_setup)s

      SELECT CASE (name)
         %(parameter_setup)s
         CASE DEFAULT
            write(*,*) 'no parameter matching', name, value
      END SELECT

      return
      end
      
    subroutine update_all_coup()
    implicit none
     call coup()
    return 
    end
      

    subroutine get_pdg_order(PDG, ALLPROC)
  IMPLICIT NONE
CF2PY INTEGER, intent(out) :: PDG(%(nb_me)i,%(maxpart)i)  
CF2PY INTEGER, intent(out) :: ALLPROC(%(nb_me)i)
  INTEGER PDG(%(nb_me)i,%(maxpart)i), PDGS(%(nb_me)i,%(maxpart)i)
  INTEGER ALLPROC(%(nb_me)i),PIDs(%(nb_me)i)
  DATA PDGS/ %(pdgs)s /
  DATA PIDS/ %(pids)s /
  PDG = PDGS
  ALLPROC = PIDS
  RETURN
  END 

    subroutine get_prefix(PREFIX)
  IMPLICIT NONE
CF2PY CHARACTER*20, intent(out) :: PREFIX(%(nb_me)i)
  character*20 PREFIX(%(nb_me)i),PREF(%(nb_me)i)
  DATA PREF / '%(prefix)s'/
  PREFIX = PREF
  RETURN
  END 
 


    subroutine set_fixed_extra_scale(new_value)
    implicit none
CF2PY logical, intent(in) :: new_value
    logical new_value
                logical fixed_extra_scale
            integer maxjetflavor
            double precision mue_over_ref
            double precision mue_ref_fixed
            common/model_setup_running/maxjetflavor,fixed_extra_scale,mue_over_ref,mue_ref_fixed
  
        fixed_extra_scale = new_value
        return 
        end

    subroutine set_mue_over_ref(new_value)
    implicit none
CF2PY double precision, intent(in) :: new_value
    double precision new_value
    logical fixed_extra_scale
    integer maxjetflavor
    double precision mue_over_ref
    double precision mue_ref_fixed
    common/model_setup_running/maxjetflavor,fixed_extra_scale,mue_over_ref,mue_ref_fixed
  
    mue_over_ref = new_value
        
    return 
    end

    subroutine set_mue_ref_fixed(new_value)
    implicit none
CF2PY double precision, intent(in) :: new_value
    double precision new_value
    logical fixed_extra_scale
    integer maxjetflavor
    double precision mue_over_ref
    double precision mue_ref_fixed
    common/model_setup_running/maxjetflavor,fixed_extra_scale,mue_over_ref,mue_ref_fixed
  
    mue_ref_fixed = new_value
        
    return 
    end


    subroutine set_maxjetflavor(new_value)
    implicit none
CF2PY integer, intent(in) :: new_value
    integer new_value
    logical fixed_extra_scale
    integer maxjetflavor
    double precision mue_over_ref
    double precision mue_ref_fixed
    common/model_setup_running/maxjetflavor,fixed_extra_scale,mue_over_ref,mue_ref_fixed
  
    maxjetflavor = new_value
        
    return 
    end


    subroutine set_asmz(new_value)
    implicit none
CF2PY double precision, intent(in) :: new_value
    double precision new_value
          integer nloop
      double precision asmz
      common/a_block/asmz,nloop
    asmz = new_value
    write(*,*) "asmz is set to ", new_value
        
    return 
    end

    subroutine set_nloop(new_value)
    implicit none
CF2PY integer, intent(in) :: new_value
    integer new_value
          integer nloop
      double precision asmz
      common/a_block/asmz,nloop
    nloop = new_value
     write(*,*) "nloop is set to ", new_value
        
    return 
    end

