MODULE %(mp_prefix)s%(proc_prefix)sLOOP_POLYNOMIAL
  use %(proc_prefix)spolynomial_constants
  use vertex_polynomial
  implicit none
  
  TYPE %(mp_prefix)sL_COEF
    sequence
    %(complex_format)s COEFS(0:LOOPMAXCOEFS-1)
    LOGICAL IS_ZERO(0:LOOPMAXCOEFS-1)
    INTEGER NON_ZERO_IDS(LOOPMAXCOEFS)
    INTEGER N_NON_ZERO_IDS
  ENDTYPE %(mp_prefix)sL_COEF

  interface %(mp_prefix)s%(proc_prefix)sinitialize_loop_coefs
    module procedure %(mp_prefix)s%(proc_prefix)sinitialize_one_loop_coef, %(mp_prefix)s%(proc_prefix)sinitialize_all_loop_coefs
  end interface %(mp_prefix)s%(proc_prefix)sinitialize_loop_coefs

  interface %(mp_prefix)s%(proc_prefix)sprint_coefs
    module procedure %(mp_prefix)s%(proc_prefix)sprint_one_coefs, %(mp_prefix)s%(proc_prefix)sprint_all_coefs
  end interface %(mp_prefix)s%(proc_prefix)sprint_coefs

CONTAINS

  SUBROUTINE %(mp_prefix)s%(proc_prefix)sinitialize_one_loop_coef(this)
      TYPE(%(mp_prefix)sL_COEF), INTENT(inout) :: this
      this%%COEFS(:)        = (%(zero_def)s,%(zero_def)s)
      this%%IS_ZERO(:)      = .TRUE.
      this%%NON_ZERO_IDS(:) = -1
      this%%N_NON_ZERO_IDS  = 0
  END SUBROUTINE %(mp_prefix)s%(proc_prefix)sinitialize_one_loop_coef

  SUBROUTINE %(mp_prefix)s%(proc_prefix)sinitialize_all_loop_coefs(this)
      TYPE(%(mp_prefix)sL_COEF), INTENT(inout) :: this(MAXLWFSIZE,MAXLWFSIZE)
	  INTEGER I,J
	  DO I=1,MAXLWFSIZE
        DO J=1,MAXLWFSIZE
		  CALL %(mp_prefix)s%(proc_prefix)sinitialize_one_loop_coef(this(I,J))
		ENDDO
	  ENDDO
  END SUBROUTINE %(mp_prefix)s%(proc_prefix)sinitialize_all_loop_coefs

  SUBROUTINE %(mp_prefix)s%(proc_prefix)sPRINT_ONE_COEFS(COEFS)
      TYPE(%(mp_prefix)sL_COEF),INTENT(in)    :: COEFS
	  double complex TMP      
	  INTEGER I
      WRITE(*,*) 'n_non_zero_IDs =',COEFS%%N_NON_ZERO_IDS
	  IF (COEFS%%N_NON_ZERO_IDS.eq.0) THEN
	    WRITE(*,*) 'Empty loop coefs.'
	  ELSE
        DO I=1,COEFS%%N_NON_ZERO_IDS
C Cast in double precision to avoid problems with the known gfortran bug on quadprec writeout
	      TMP = DCMPLX(COEFS%%COEFS(COEFS%%NON_ZERO_IDS(I)))
          WRITE(*,*) 'Loop COEFS(',COEFS%%NON_ZERO_IDS(I),')=',TMP
        ENDDO
	  ENDIF
  END SUBROUTINE %(mp_prefix)s%(proc_prefix)sPRINT_ONE_COEFS

  SUBROUTINE %(mp_prefix)s%(proc_prefix)sPRINT_ALL_COEFS(COEFS)
      TYPE(%(mp_prefix)sL_COEF),INTENT(in)    :: COEFS(MAXLWFSIZE,MAXLWFSIZE)
	  double complex TMP      
	  INTEGER I,J,K
	  logical foundSome
	  foundSome = .False.
	  DO J=1,MAXLWFSIZE
	    DO K=1,MAXLWFSIZE
	      IF (COEFS(J,K)%%N_NON_ZERO_IDS.ne.0) THEN
		    foundSome = .True.
            WRITE(*,*) 'For (',J,',',K,'), n_non_zero_IDs =',COEFS(J,K)%%N_NON_ZERO_IDS
            DO I=1,COEFS(J,K)%%N_NON_ZERO_IDS
C Cast in double precision to avoid problems with the known gfortran bug on quadprec writeout
	          TMP = DCMPLX(COEFS(J,K)%%COEFS(COEFS(J,K)%%NON_ZERO_IDS(I)))
              WRITE(*,*) 'Loop COEFS(',COEFS(J,K)%%NON_ZERO_IDS(I),')=',TMP
            ENDDO
		  ENDIF
		ENDDO
	  ENDDO
      IF(.NOT.foundSome) THEN
	    WRITE(*,*) 'Empty list of loop coefs.' 
	  ENDIF
  END SUBROUTINE %(mp_prefix)s%(proc_prefix)sPRINT_ALL_COEFS

END MODULE %(mp_prefix)s%(proc_prefix)sLOOP_POLYNOMIAL

## if(not AmplitudeReduction){
C THE SUBROUTINE TO CREATE THE COEFFICIENTS FROM LAST LOOP WF AND 
C MULTIPLY BY THE BORN

      SUBROUTINE %(mp_prefix)s%(proc_prefix)sCREATE_LOOP_COEFS(LOOP_WF,RANK,LCUT_SIZE,LOOP_GROUP_NUMBER,SYMFACT,MULTIPLIER,COLOR_ID,HELCONFIG)
	  USE VERTEX_POLYNOMIAL
	  USE %(proc_prefix)sPOLYNOMIAL_CONSTANTS	  
	  implicit none
C  
C CONSTANTS 
C 
## if (not LoopInduced) {
	  INTEGER NBORNAMPS
      PARAMETER (NBORNAMPS=%(nbornamps)d)
## }
	  %(real_format)s ZERO,ONE
	  PARAMETER (ZERO=%(zero_def)s,ONE=%(one_def)s)
	  %(complex_format)s IMAG1
      PARAMETER (IMAG1=(ZERO,ONE))
	  %(complex_format)s CMPLX_ZERO
	  PARAMETER (CMPLX_ZERO=(ZERO,ZERO))
      INTEGER    NCOLORROWS
	  PARAMETER (NCOLORROWS=%(nloopamps)d)
	  INTEGER    NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=%(nloop_groups)d)
	  INTEGER    NCOMB
      PARAMETER (NCOMB=%(ncomb)d)
C     These are constants related to the split orders
      INTEGER    NSO, NSQUAREDSO, NAMPSO
	  PARAMETER (NSO=%(nSO)d, NSQUAREDSO=%(nSquaredSO)d, NAMPSO=%(nAmpSO)d)
C  
C ARGUMENTS 
C  
      %(complex_format)s LOOP_WF(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)
	  INTEGER RANK, COLOR_ID, SYMFACT, MULTIPLIER, LCUT_SIZE, HELCONFIG, LOOP_GROUP_NUMBER
C  
C LOCAL VARIABLES 
C
      %(complex_format)s CFTOT
      %(complex_format)s CONST(NAMPSO)
      INTEGER I,J
C  
C FUNCTIONS
C
      INTEGER %(proc_prefix)sML5SOINDEX_FOR_BORN_AMP, %(proc_prefix)sML5SOINDEX_FOR_LOOP_AMP, %(proc_prefix)sML5SQSOINDEX
C
C GLOBAL VARIABLES
C
	  INTEGER CF_D(NCOLORROWS,%(color_matrix_size)s)
	  INTEGER CF_N(NCOLORROWS,%(color_matrix_size)s)
	  common/%(proc_prefix)sCF/CF_D,CF_N

	  LOGICAL CHECKPHASE
	  LOGICAL HELDOUBLECHECKED
      common/%(proc_prefix)sINIT/CHECKPHASE, HELDOUBLECHECKED

	  INTEGER HELOFFSET
	  INTEGER GOODHEL(NCOMB)
	  LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
	  common/%(proc_prefix)sFilters/GOODAMP,GOODHEL,HELOFFSET

      %(complex_format)s LOOPCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)
	  common/%(proc_prefix)s%(mp_prefix)sLCOEFS/LOOPCOEFS

	  INTEGER HELPICKED
	  common/%(proc_prefix)sHELCHOICE/HELPICKED
      
## if(not LoopInduced) {
	  %(complex_format)s AMP(NBORNAMPS)
	  common/%(proc_prefix)s%(mp_prefix)sAMPS/AMP
## }
      
	  DO I=1,NAMPSO
	    CONST(I)=CMPLX_ZERO
      ENDDO

      DO I=1,NBORNAMPS
	    CFTOT=CMPLX(CF_N(COLOR_ID,I)/(ONE*ABS(CF_D(COLOR_ID,I))),ZERO,KIND=%(kind)d)
	    IF(CF_D(COLOR_ID,I).LT.0) CFTOT=CFTOT*IMAG1
	    CONST(%(proc_prefix)sML5SOINDEX_FOR_BORN_AMP(I))=CONST(%(proc_prefix)sML5SOINDEX_FOR_BORN_AMP(I))+CFTOT*CONJG(AMP(I))
	  ENDDO
	  
      DO I=1,NAMPSO
	    IF (CONST(I).NE.CMPLX_ZERO) THEN
	      CONST(I)=(CONST(I)*MULTIPLIER)/SYMFACT
	      IF (.NOT.CHECKPHASE.AND.HELDOUBLECHECKED.AND.HELPICKED.EQ.-1) THEN
	        CONST(I)=CONST(I)*GOODHEL(HELCONFIG)
	      ENDIF
	      CALL %(mp_prefix)s%(proc_prefix)sMERGE_WL(LOOP_WF,RANK,LCUT_SIZE,CONST(I),LOOPCOEFS(0,%(proc_prefix)sML5SQSOINDEX(I,%(proc_prefix)sML5SOINDEX_FOR_LOOP_AMP(COLOR_ID)),LOOP_GROUP_NUMBER))
	    ENDIF
      ENDDO

	  END
## }else{
C The subroutine to create the loop coefficients form the last loop wf.
C In this case of loop-induced process, the reduction is performed at the loop
C amplitude level so that no multiplication is performed.

      SUBROUTINE %(mp_prefix)s%(proc_prefix)sCREATE_LOOP_COEFS(LOOP_WF,RANK,LCUT_SIZE,LOOP_GROUP_NUMBER,SYMFACT,MULTIPLIER)
	  USE VERTEX_POLYNOMIAL
	  USE %(proc_prefix)sPOLYNOMIAL_CONSTANTS	  
	  implicit none
C  
C CONSTANTS 
C 
	  %(real_format)s ZERO,ONE
	  PARAMETER (ZERO=%(zero_def)s,ONE=%(one_def)s)
	  %(complex_format)s IMAG1
      PARAMETER (IMAG1=(ZERO,ONE))
	  %(complex_format)s CMPLX_ZERO
	  PARAMETER (CMPLX_ZERO=(ZERO,ZERO))
	  INTEGER    NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=%(nloop_groups)d)
	  INTEGER    NCOMB
      PARAMETER (NCOMB=%(ncomb)d)
C  
C ARGUMENTS 
C  
      %(complex_format)s LOOP_WF(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)
	  INTEGER RANK, SYMFACT, LCUT_SIZE, LOOP_GROUP_NUMBER, MULTIPLIER
C  
C LOCAL VARIABLES 
C
      %(complex_format)s CONST
      INTEGER I,J
C
C GLOBAL VARIABLES
C
      %(complex_format)s LOOPCOEFS(0:LOOPMAXCOEFS-1,NLOOPGROUPS)
	  common/%(proc_prefix)s%(mp_prefix)sLCOEFS/LOOPCOEFS
C
C BEGIN CODE
C
	  CONST=CMPLX((ONE*MULTIPLIER)/SYMFACT,ZERO,KIND=16)

      CALL %(mp_prefix)s%(proc_prefix)sMERGE_WL(LOOP_WF,RANK,LCUT_SIZE,CONST,LOOPCOEFS(0,LOOP_GROUP_NUMBER))

	  END
## }

## if(ninja_available){
      SUBROUTINE %(mp_prefix)s%(proc_prefix)sINVERT_MOMENTA_IN_POLYNOMIAL(NCOEFS,POLYNOMIAL)
C       Just a handy subroutine to modify the coefficients for the
C       tranformation q_loop -> -q_loop
C       It is only used for the NINJA interface
        USE %(proc_prefix)sPOLYNOMIAL_CONSTANTS
        IMPLICIT NONE

        INTEGER I, NCOEFS

		%(complex_format)s POLYNOMIAL(0:NCOEFS-1)

        DO I=0,NCOEFS-1
          IF (MOD(COEFTORANK_MAP(I),2).eq.1) then
            POLYNOMIAL(I)=-POLYNOMIAL(I)
          ENDIF
        ENDDO

      END
## }

C Now the routines to update the wavefunctions

