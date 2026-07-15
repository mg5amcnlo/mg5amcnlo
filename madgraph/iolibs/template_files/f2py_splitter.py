C this is a f2py wrapper for reweight mode at tree level

  subroutine %(f2py_prefix)sf77_smatrixhel(pdgs, procid, npdg, p, ALPHAS, SCALE2, nhel, ANS)
  use model_object
  use aloha_object
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
  integer flavor(%(maxpart)i),I
%(flavor_index_decl)s
  include 'coupl.inc'
  
  
  if (scale2.eq.0)then
       PI = 3.141592653589793D0
       G = 2* DSQRT(ALPHAS*PI)
       CALL UPDATE_AS_PARAM()
  else
       CALL UPDATE_AS_PARAM2(scale2, ALPHAS)
  endif

%(flavormapping)s

%(smatrixhel)s

      return
      end
  
  subroutine %(f2py_prefix)sf77_density(pdgs, npdg, procid, P, POS, N_CHANGING, ALLOW_HEL, N_COMB, ALPHAS, SCALE2, INTER)
  IMPLICIT NONE
CF2PY double precision, intent(in) :: p
CF2PY integer, intent(in) :: pdgs
CF2PY integer, intent(in) :: procid
CF2PY integer, intent(in) :: pos
CF2PY integer, INTENT(IN) :: ALLOW_HEL
CF2PY double precision INTENT(IN) :: ALPHAS
CF2PY double precision INTENT(IN) :: SCALE2
CF2PY double complex INTENT(OUT), dimension(N_COMB*(N_COMB+1)/2) :: INTER
CF2PY integer, intent(in) :: N_COMB
CF2PY integer, intent(in) :: N_CHANGING
CF2PY integer, intent(in) :: NPDG
C     scale is a dummy argument added to have the same syntax as in loop-induced
C
C     Some variables seem unused but they are necessary for density_splitter
C

  integer pdgs(*), procid, n_changing, n_comb, npdg
  double precision p(0:3,*)
  double precision ALPHAS, SCALE2
  INTEGER POS(n_changing)
  INTEGER ALLOW_HEL(n_changing*n_comb)
  DOUBLE COMPLEX INTER(n_comb*(n_comb+1)/2)
  integer flavor(%(maxpart)i),I
C     Update is done insider the direct density call functions

%(flavormapping)s

%(density_splitter)s

            return
            end

  subroutine %(f2py_prefix)sf77_get_all_inter(pdgs, procid, npdg, P, POS, N_CHANGING, ALLOW_HEL, N_COMB, INTER)
  IMPLICIT NONE
C     P momenta
C     NHEL base of helicity that are not changing
C     POS(N_CHNGING): position of the changing helicity
C     n_changing: number of changing helicity
C     ALLOW_HEL(NCOMB, N_CHANGING): combination of helicity to
C      consider (all jamp computed)
C     INTER((NCOMB*NCOMB+1)/2: all interference term (not the
C      symmetric one)

  integer pdgs(*)
  integer npdg, nhel, procid, n_changing
  integer n_comb
  double precision p(*)
  double precision ANS,  PI,SCALE2
  INTEGER POS(*)
  INTEGER ALLOW_HEL(*)
  DOUBLE COMPLEX INTER(*)
  integer flavor(%(maxpart)i),I
C     Update is done insider the direct density call functions

C     Update is done insider the direct density call functions

%(flavormapping)s

%(inter_splitter)s

            return
            end

      SUBROUTINE %(f2py_prefix)sf77_INITIALISE(PATH)
C     ROUTINE FOR F2PY to read the benchmark point.
      IMPLICIT NONE
      CHARACTER*512 PATH
CF2PY INTENT(IN) :: PATH
      CALL SETPARA(PATH)  !first call to setup the paramaters
      RETURN
      END
      
      
      subroutine %(f2py_prefix)sf77_CHANGE_PARA(name, value)
      use model_object
      use aloha_object
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
      
    subroutine %(f2py_prefix)sf77_update_all_coup()
    implicit none
     call coup()
    return 
    end
      
      

    subroutine %(f2py_prefix)sf77_get_pdg_order(PDG, ALLPROC)
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

    subroutine %(f2py_prefix)sf77_get_prefix(prefix)
  IMPLICIT NONE
CF2PY CHARACTER*20, intent(out) :: PREFIX(%(nb_me)i)
  character*20 PREFIX(%(nb_me)i),PREF(%(nb_me)i)
  DATA PREF / '%(prefix)s'/
  PREFIX = PREF
  RETURN
  END 
 


    subroutine %(f2py_prefix)sf77_set_fixed_extra_scale(new_value)
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

    subroutine %(f2py_prefix)sf77_set_mue_over_ref(new_value)
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

    subroutine %(f2py_prefix)sf77_set_mue_ref_fixed(new_value)
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


    subroutine %(f2py_prefix)sf77_set_maxjetflavor(new_value)
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


    subroutine %(f2py_prefix)sf77_set_asmz(new_value)
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

    subroutine %(f2py_prefix)sf77_set_nloop(new_value)
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
    

    %(nhel)s
