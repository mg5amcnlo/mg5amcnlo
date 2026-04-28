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

