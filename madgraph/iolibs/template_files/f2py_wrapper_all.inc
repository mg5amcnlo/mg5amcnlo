      %(python_information)s

      SUBROUTINE %(f2py_prefix)sSMATRIXHEL(PDGS, PROCID, NPDG, P, ALPHAS, SCALE2,
     $  NHEL, ANS)
      IMPLICIT NONE
C     ALPHAS is given at scale2 (SHOULD be different of 0 for loop
C      induced, ignore for LO)  

CF2PY double precision, intent(in), dimension(0:3,npdg) :: p
CF2PY integer, intent(in), dimension(npdg) :: pdgs
CF2PY integer, intent(in):: procid
CF2PY integer, intent(in) :: npdg
CF2PY double precision, intent(out) :: ANS
CF2PY double precision, intent(in) :: ALPHAS
CF2PY double precision, intent(in) :: SCALE2
      INTEGER PDGS(*)
      INTEGER NPDG, NHEL, PROCID
      DOUBLE PRECISION P(*)
      DOUBLE PRECISION ANS, ALPHAS,SCALE2

      call %(f2py_prefix)sf77_smatrixhel(pdgs, procid, npdg, p, alphas, scale2, nhel, ans)
       
      RETURN
      END

      SUBROUTINE %(f2py_prefix)sPY_GET_DENSITY(PDGS, PROCID, NPDG, P, POS, N_CHANGING,  ALLOW_HEL, N_COMB, ALPHAS, INTER)
C     P momenta
C     NHEL base of helicity that are not changing
C     POS(N_CHNGING): position of the changing helicity
C     n_changing: number of changing helicity
C     ALLOW_HEL(NCOMB, N_CHANGING): combination of helicity to
C     consider (all jamp computed)
C     INTER((NCOMB*NCOMB+1)/2: all interference term (not the
C     symmetric one)

CF2PY double precision, intent(in), dimension(0:3,npdg) :: p
CF2PY integer, intent(in), dimension(npdg) :: pdgs
CF2PY integer, intent(in):: procid
CF2PY integer, intent(in), dimension(*):: pos
CF2PY integer, intent(in) :: npdg
CF2PY integer, intent(in), optional, depend(pos) :: n_changing=len(pos)
CF2PY integer, INTENT(IN) :: ALLOW_HEL(N_CHANGING*N_COMB)
CF2PY integer INTENT(IN) :: N_COMB
CF2PY double precision INTENT(IN) :: ALPHAS
CF2PY double complex INTENT(OUT) :: INTER(N_COMB*(N_COMB+1)/2)
      INTEGER PDGS(*)
      INTEGER NPDG, PROCID, N_CHANGING, N_COMB
      INTEGER POS(*)
      DOUBLE PRECISION ALPHAS
      INTEGER ALLOW_HEL(*)
      DOUBLE COMPLEX INTER(*)
      call f77_density(PDGS, PROCID, NPDG, P, POS, N_CHANGING,  ALLOW_HEL, N_COMB, ALPHAS, INTER)

      return 
      end

      SUBROUTINE %(f2py_prefix)sPY_GET_ALL_INTER(PDGS, PROCID, NPDG, P, POS,N_CHANGING, ALLOW_HEL, N_COMB, INTER)
C     P momenta
C     NHEL base of helicity that are not changing
C     POS(N_CHNGING): position of the changing helicity
C     n_changing: number of changing helicity
C     ALLOW_HEL(NCOMB, N_CHANGING): combination of helicity to
C     consider (all jamp computed)
C     INTER((NCOMB*NCOMB+1)/2: all interference term (not the
C     symmetric one)

CF2PY double precision, intent(in), dimension(0:3,npdg) :: p
CF2PY integer, intent(in), dimension(npdg) :: pdgs
CF2PY integer, intent(in):: procid
CF2PY integer, intent(in), dimension(*):: pos
CF2PY integer, intent(in) :: npdg
CF2PY integer, intent(in), optional, depend(pos) :: n_changing=len(pos)
CF2PY INTENT(IN) :: ALLOW_HEL(N_CHANGING*N_COMB)
CF2PY INTENT(IN) :: N_COMB
      INTEGER PDGS(*)
      INTEGER NPDG, PROCID, N_CHANGING, N_COMB
      INTEGER POS(*)
      INTEGER ALLOW_HEL(*)
      DOUBLE COMPLEX INTER(*)
CF2PY INTENT(OUT) :: INTER(N_COMB*(N_COMB+1)/2)
      call f77_get_all_inter(PDGS, PROCID, NPDG, P, POS, N_CHANGING,  ALLOW_HEL, N_COMB, INTER)

      return 
      end



      SUBROUTINE %(f2py_prefix)sINITIALISE(PATH)
C     ROUTINE FOR F2PY to read the benchmark point.
      IMPLICIT NONE
      CHARACTER*512 PATH
CF2PY INTENT(IN) :: PATH
      CALL SETPARA(PATH)  !first call to setup the paramaters
      RETURN
      END


      SUBROUTINE %(f2py_prefix)sCHANGE_PARA(NAME, VALUE)
      IMPLICIT NONE
CF2PY intent(in) :: name
CF2PY intent(in) :: value

      CHARACTER*512 NAME
      DOUBLE PRECISION VALUE
      call %(f2py_prefix)sf77_change_para(name, value)

      RETURN
      END

      SUBROUTINE %(f2py_prefix)sUPDATE_ALL_COUP()
      IMPLICIT NONE
      CALL %(f2py_prefix)sf77_update_all_coup()
      RETURN
      END


      SUBROUTINE %(f2py_prefix)sGET_PDG_ORDER(PDG, ALLPROC)
      IMPLICIT NONE
CF2PY INTEGER, intent(out) :: PDG(%(nb_me)i,%(maxpart)i)
CF2PY INTEGER, intent(out) :: ALLPROC(%(nb_me)i)
  INTEGER PDG(%(nb_me)i,%(maxpart)i)
  INTEGER ALLPROC(%(nb_me)i)
      call %(f2py_prefix)sf77_get_pdg_order(pdg, allproc)
      RETURN
      END

      SUBROUTINE %(f2py_prefix)sGET_prefix(prefix)
      IMPLICIT NONE
CF2PY CHARACTER*20, intent(out) :: prefix(%(nb_me)i)
      CHARACTER*20 prefix(%(nb_me)i)
      call %(f2py_prefix)sf77_get_prefix(prefix)
      RETURN
      END

      SUBROUTINE %(f2py_prefix)sSET_FIXED_EXTRA_SCALE(NEW_VALUE)
      IMPLICIT NONE
CF2PY logical, intent(in) :: new_value
      LOGICAL NEW_VALUE
      call %(f2py_prefix)sf77_set_fixed_extra_scale(new_value)
      RETURN
      END

      SUBROUTINE %(f2py_prefix)sSET_MUE_OVER_REF(NEW_VALUE)
      IMPLICIT NONE
CF2PY double precision, intent(in) :: new_value
      DOUBLE PRECISION NEW_VALUE

      call %(f2py_prefix)sf77_set_mue_over_ref(new_value)
      RETURN
      END

      SUBROUTINE %(f2py_prefix)sSET_MUE_REF_FIXED(NEW_VALUE)
      IMPLICIT NONE
CF2PY double precision, intent(in) :: new_value
      DOUBLE PRECISION NEW_VALUE

      call %(f2py_prefix)sf77_set_mue_ref_fixed(new_value)
      RETURN
      END


      SUBROUTINE %(f2py_prefix)sSET_MAXJETFLAVOR(NEW_VALUE)
      IMPLICIT NONE
CF2PY integer, intent(in) :: new_value
      INTEGER NEW_VALUE
      call %(f2py_prefix)sf77_set_maxjetflavor(new_value)

      RETURN
      END


      SUBROUTINE %(f2py_prefix)sSET_ASMZ(NEW_VALUE)
      IMPLICIT NONE
CF2PY double precision, intent(in) :: new_value
      DOUBLE PRECISION NEW_VALUE
      call %(f2py_prefix)sf77_set_asmz(new_value)
      RETURN
      END

      SUBROUTINE %(f2py_prefix)sSET_NLOOP(NEW_VALUE)
      IMPLICIT NONE
CF2PY integer, intent(in) :: new_value
      INTEGER NEW_VALUE
      call %(f2py_prefix)sf77_set_nloop(new_value)
      RETURN
      END


      %(nhel)s
      
      SUBROUTINE %(f2py_prefix)sGET_IDENS(idens)
      IMPLICIT NONE
CF2PY integer, intent(out) :: idens(%(nb_me)i)
      integer idens(%(nb_me)i)
      %(idens_value)s
      RETURN
      END



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
