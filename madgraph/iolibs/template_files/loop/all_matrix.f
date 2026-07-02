C this is a f2py wrapper for reweight mode at loop-induced level


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
      
C This function needs to be able to take as input momenta of the form all_p = [p1, p2, ...],
C where pi = [(), (), (), ()], where the () are tuples

C     This function takes angles PHI and THETA and rotates the
C     4-momenta of all external particles into the helicity referential 
C     The helicity referential is {n, r, k} so the components of P
C     can change even if phi and theta are 0
      SUBROUTINE f77_rotationp(P, PHI, THETA, NEXTERNAL, PROT)
      IMPLICIT NONE
CF2PY integer, intent(in) :: npdg
CF2PY integer, intent(in):: procid
CF2PY integer, intent(in), dimension(npdg) :: pdgs
CF2PY DOUBLE PRECISION, INTENT(OUT) :: PROT(0:3, NEXTERNAL)
CF2PY INTEGER, INTENT(IN) :: NEXTERNAL
CF2PY DOUBLE PRECISION, INTENT(IN) :: PHI
CF2PY DOUBLE PRECISION, INTENT(IN) :: THETA
CF2PY DOUBLE PRECISION, INTENT(IN) :: P(0:3, NEXTERNAL)
C
C     CONSTANT
C     
      DOUBLE PRECISION    EPSILON
      PARAMETER (EPSILON=1D-10)
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


C     This function takes a given 4-momentum and returns the angles
C     needed to use this particle as a reference for the helicity basis
C     These angles are the input of ROTATIONP.

      SUBROUTINE f77_refchoicep(PREF, PHI, THETA)
      IMPLICIT NONE
CF2PY DOUBLE PRECISION, INTENT(OUT) :: PHI
CF2PY DOUBLE PRECISION INTENT(OUT) :: THETA
CF2PY DOUBLE PRECISION, INTENT(IN) :: PREF(0:3)
C     
C     CONSTANT
C     
      DOUBLE PRECISION    EPSILON
      PARAMETER (EPSILON=1D-10)
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

      SUBROUTINE %(f2py_prefix)sF77_DENSITY(PDGS, NPDG, PROCID, P, POS, ALLOW_HEL, ALPHAS, SCALE2, N_CHANGING, N_COMB, INTER)
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
C
C     Some variables seem unused but they are necessary for density_splitter
C
      INTEGER PDGS(*)
      INTEGER PROCID
      DOUBLE PRECISION P(0:3, *)
      DOUBLE PRECISION ALPHAS, SCALE2
      INTEGER N_CHANGING, N_COMB, NPDG
      INTEGER POS(N_CHANGING)
      INTEGER ALLOW_HEL(N_CHANGING*N_COMB)
      DOUBLE COMPLEX INTER(N_COMB*(N_COMB+1)/2)    

      %(density_splitter)s

      return 
      end