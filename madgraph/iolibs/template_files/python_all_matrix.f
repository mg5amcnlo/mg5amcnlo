%(python_information)s


  subroutine smatrixhel(pdgs, procid, npdg, p, ALPHAS, SCALE2, nhel, ANS)
  use model_object
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
  call smatrixhel_internal(pdgs, procid, npdg, p, ALPHAS, SCALE2, nhel, ANS)
  return
  end




  subroutine smatrixhel_internal(pdgs, procid, npdg, p, ALPHAS, SCALE2, nhel, ANS)
  use model_object
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
  
      SUBROUTINE INITIALISE(PATH)
C     ROUTINE FOR F2PY to read the benchmark point.
      IMPLICIT NONE
      CHARACTER*512 PATH
CF2PY INTENT(IN) :: PATH
      CALL SETPARA(PATH)  !first call to setup the paramaters
       %(setpara_for_each_matrix)s
      RETURN
      END
      
      
      subroutine CHANGE_PARA(name, value)
      use model_object
      implicit none
CF2PY intent(in) :: name
CF2PY intent(in) :: value

      character*512 name
      double precision value

      call change_para_internal(name, value)
      return
      end

      subroutine CHANGE_PARA_internal(name, value)
      use model_object
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

C     This function takes the number of a particle (in the generate
C     command) and returns the angles needed to put this particle as a 
c     reference for the helicity basis.
      SUBROUTINE REFCHOICE(P, PNUMBER, SIZEPN, PHI, THETA)
      IMPLICIT NONE
C     
C     CONSTANT
C     
      DOUBLE PRECISION    EPSILON
      PARAMETER (EPSILON=1D-10)

CF2PY DOUBLE PRECISION, INTENT(OUT) :: PHI
CF2PY DOUBLE PRECISION, INTENT(OUT) :: THETA
CF2PY INTEGER, INTENT(IN) :: PNUMBER(*)
CF2PY INTEGER, INTENT(IN) :: SIZEPN
CF2PY DOUBLE PRECISION, INTENT(IN) :: P(0:3, *)
C     
C     ARGUMENT
C     
      REAL*8 P(0:3, *)
      REAL*8 PREF(0:3)
      INTEGER PNUMBER(*), I, SIZEPN
      DOUBLE PRECISION THETA, PHI

C     PREF is the 4-momentum of the particle with particle number =
C     pNUMBER
C We need to initialise PREF to 0
      PREF = [0, 0, 0, 0]
      DO I=1, SIZEPN
        PREF = PREF + P(:, PNUMBER(I))
      ENDDO
C     The angles phi and theta are calculated such that after rotation, 
C     pref = (E, 0, 0, p_k)

      IF (ABS(PREF(1)/PREF(0)) < EPSILON .AND. ABS(PREF(2)/PREF(0)) < EPSILON) THEN
c     If the particle is immobile then we can't do the rotation
        IF (ABS(PREF(3)/PREF(0)) < EPSILON) THEN
          WRITE(*,*) "The chosen particle is immobile. We cant use it",
     &    " as reference for the helicity basis"
          STOP "Error when passing to helicity basis"
c If the particle has no tranverse momentum (we are already in the correct frame)
        ELSE IF (PREF(3) < 0) THEN
          PHI = 0D0
          THETA = 4*ATAN(1D0)
        ELSE IF (PREF(3) > 0) THEN
          PHI = 0D0
          THETA = 0D0
        ENDIF
c If the momentum is anything else:
      ELSE
        PHI = SIGN(1D0, PREF(2)) * DACOS(PREF(1)/DSQRT(PREF(1)**2 +
     &  PREF(2)**2))
        THETA = DACOS(PREF(3)/DSQRT(PREF(1)**2 + PREF(2)**2 + 
     &  PREF(3)**2))
     
      ENDIF


      END



C     This function takes a given 4-momentum and returns the angles
C      needed to put 
C     this particle as a reference for the helicity basis
      SUBROUTINE REFCHOICEP(PREF, PHI, THETA)
      IMPLICIT NONE
C     
C     CONSTANT
C     
      DOUBLE PRECISION    EPSILON
      PARAMETER (EPSILON=1D-10)

CF2PY DOUBLE PRECISION, INTENT(OUT) :: PHI
CF2PY DOUBLE PRECISION INTENT(OUT) :: THETA
CF2PY DOUBLE PRECISION, INTENT(IN) :: PREF(0:3)
C     
C     ARGUMENT
C     
      REAL*8 PREF(0:3)
      DOUBLE PRECISION THETA, PHI

C     The angles phi and theta are calculated such that after rotation, 
C     pref = (E, 0, 0, p_k)

      IF (ABS(PREF(1)/PREF(0)) < EPSILON .AND. ABS(PREF(2)/PREF(0)) < EPSILON) THEN
c     If the particle is immobile then we can't do the rotation
        IF (ABS(PREF(3)/PREF(0)) < EPSILON) THEN
          WRITE(*,*) "The chosen particle is immobile. We cant use it",
     &    " as reference for the helicity basis, using phi = theta = 0"
c          STOP "Error when passing to helicity basis"
c   We chose to put the angles to 0 to not stop the code
          PHI = 0D0
          THETA = 0D0
c If the particle has no tranverse momentum (we are already in the correct frame)
        ELSE IF (PREF(3) < 0) THEN
          PHI = 0D0
          THETA = 4*ATAN(1D0)
        ELSE IF (PREF(3) > 0) THEN
          PHI = 0D0
          THETA = 0D0
        ENDIF
c If the momentum is anything else:
      ELSE
        PHI = SIGN(1D0, PREF(2)) * DACOS(PREF(1)/DSQRT(PREF(1)**2 +
     &  PREF(2)**2))
        THETA = DACOS(PREF(3)/DSQRT(PREF(1)**2 + PREF(2)**2 + 
     &  PREF(3)**2))
     
      ENDIF

      END

C This function needs to be able to take as input momenta of the form all_p = [p1, p2, ...],
C where pi = [(), (), (), ()], where the () are tuples

C     This function takes angles PHI and THETA and rotates the
C     4-momenta of all external particles into the helicity referential 
      SUBROUTINE ROTATIONP(P, PHI, THETA, NEXTERNAL, PROT)
      IMPLICIT NONE
C     The helicity referential is {n, r, k} so the components of P
C     can change even if phi and theta are 0
C     
C     CONSTANT
C     
      DOUBLE PRECISION    EPSILON
      PARAMETER (EPSILON=1D-10)

CF2PY DOUBLE PRECISION, INTENT(OUT) :: PROT(0:3, NEXTERNAL)
CF2PY INTEGER, INTENT(IN) :: NEXTERNAL
CF2PY DOUBLE PRECISION, INTENT(IN) :: PHI
CF2PY DOUBLE PRECISION, INTENT(IN) :: THETA
CF2PY DOUBLE PRECISION, INTENT(IN) :: P(0:3, NEXTERNAL)
C     
C     ARGUMENT
C     
      REAL*8 P(0:3, *)
      REAL*8 PROT(0:3, *)
      DOUBLE PRECISION THETA, PHI
      INTEGER I, NEXTERNAL

      DO I= 1, NEXTERNAL
        PROT(0, I) = P(0, I)

        PROT(1, I) = -DSIN(PHI)*P(1, I) + DCOS(PHI)*P(2, I)  !p_n
        IF (ABS(PROT(1, I)/PROT(0, I)) < EPSILON) THEN
          PROT(1, I) = 0D0
        ENDIF

        PROT(2, I) = -DCOS(PHI)*DCOS(THETA)*P(1, I) - DSIN(PHI)*  ! p_r
     & DCOS(THETA)*P(2, I) + DSIN(THETA)*P(3, I)
        IF (ABS(PROT(2, I)/PROT(0, I)) < EPSILON) THEN
          PROT(2, I) = 0D0
        ENDIF

        PROT(3, I) = DCOS(PHI)*DSIN(THETA)*P(1, I) + DSIN(PHI)*
     & DSIN(THETA)*P(2, I) + DCOS(THETA)*P(3, I) !p_k
        IF (ABS(PROT(3, I)/PROT(0, I)) < EPSILON) THEN
          PROT(3, I) = 0D0
        ENDIF
      ENDDO

      END