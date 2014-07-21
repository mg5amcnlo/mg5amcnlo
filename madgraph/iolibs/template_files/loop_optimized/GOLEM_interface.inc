	  SUBROUTINE %(proc_prefix)sGOLEMLOOP(NLOOPLINE,PL,M2L,RANK,RES,STABLE)
C  
%(info_lines)s
C 
C Interface between MG5 and Golem95.
C  
%(process_lines)s
C
C
C MODULES
C
          USE MATRICE_S
          USE FORM_FACTOR_TYPE, ONLY: FORM_FACTOR
          USE PRECISION_GOLEM, ONLY: KI
          USE TENS_COMB
	  USE TENS_REC
	  USE FORM_FACTOR_1P, ONLY: A10
	  USE FORM_FACTOR_2P, ONLY: A20
	  USE FORM_FACTOR_3P, ONLY: A30
	  USE FORM_FACTOR_4P, ONLY: A40
	  USE FORM_FACTOR_5P, ONLY: A50
	  USE FORM_FACTOR_6P, ONLY: A60 
C  
C CONSTANTS 
C  
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      LOGICAL checkPConservation
      PARAMETER (checkPConservation=.TRUE.)
      %(real_dp_format)s NORMALIZATION
      PARAMETER (NORMALIZATION = 1.D0/(16.D0*3.14159265358979323846D0**2))
      REAL(KI),DIMENSION(0:3),PARAMETER::NULL_VEC = (/0.0_ki,0.0_ki,0.0_ki,0.0_ki/)
C     GOLEM_RUN_MODE = 1: Use directly MadLoop tensorial coefficients
C     GOLEM_RUN_MODE = 2: Reconstruct the tensorial coefficeints directly from 
C                         numerator using golem internal reconstruction routine
C     GOLEM_RUN_MODE = 3: Cross-checked reconstructed coefficients against
C                         MadLoop internal ones.
      INTEGER GOLEM_RUN_MODE
      PARAMETER (GOLEM_RUN_MODE=1)
C     The following is the acceptance threshold used for GOLEM_RUN_MODE = 3
      REAL*8 coef_check_thrs
      DATA coef_check_thrs/1.0d-13/
      COMMON/%(proc_prefix)coef_check_thrs/coef_check_thrs

      LOGICAL PASS_COEF_CHECK     
C  
C ARGUMENTS 
C
      INTEGER NLOOPLINE, RANK
      %(real_dp_format)s PL(0:3,NLOOPLINE)
      %(real_dp_format)s PCT(0:3,0:NLOOPLINE-1)
      REAL(ki) PGOLEM(NLOOPLINE,0:3)
      %(mass_dp_format)s M2L(NLOOPLINE)
      COMPLEX(ki) M2GOLEM(NLOOPLINE)
      %(complex_dp_format)s RES(3)
	  LOGICAL STABLE	  
C  
C LOCAL VARIABLES 
C
	  INTEGER I, J, K
	  TYPE(FORM_FACTOR)::RES_GOLEM
	  
	  COMPLEX(KI)::COEFFS0,COEFFS0_REC
	  TYPE(COEFF_TYPE_1)::COEFFS1,COEFFS1_REC
	  TYPE(COEFF_TYPE_2)::COEFFS2,COEFFS2_REC
	  TYPE(COEFF_TYPE_3)::COEFFS3,COEFFS3_REC
	  TYPE(COEFF_TYPE_4)::COEFFS4,COEFFS4_REC
	  TYPE(COEFF_TYPE_5)::COEFFS5,COEFFS5_REC
	  TYPE(COEFF_TYPE_6)::COEFFS6,COEFFS6_REC

C         The pinch propagator optimization is not used, so for now it is
C         always 0.
	  INTEGER PINCH
C
C EXTERNAL FUNCTIONS
C
      COMPLEX(KI) %(proc_prefix)GOLEM_LOOPNUM
      EXTERNAL %(proc_prefix)GOLEM_LOOPNUM
C
C GLOBAL VARIABLES
C
      include 'coupl.inc'
	  INTEGER CTMODE
	  %(real_dp_format)s LSCALE
      common/%(proc_prefix)sCT/LSCALE,CTMODE

      INTEGER ID,SQSOINDEX,R
      COMMON/%(proc_prefix)sLOOP/ID,SQSOINDEX,R

C ----------
C BEGIN CODE
C ----------

C INITIALIZE CUTTOOLS IF NEEDED
      IF (CTINIT) THEN
	    CTINIT=.FALSE.
		CALL %(proc_prefix)sINITCT()
      ENDIF

C YOU CAN FIND THE DETAILS ABOUT THE DIFFERENT CTMODE AT THE BEGINNING OF THE FILE CTS_CUTS.F90 IN THE CUTTOOLS DISTRIBUTION

C  CONVERT THE MASSES TO BE COMPLEX
      do I=1,NLOOPLINE
	    M2LCT(I-1)=M2L(I)
	  ENDDO

C  CONVERT THE MOMENTA FLOWING IN THE LOOP LINES TO CT CONVENTIONS
      do I=0,3
        do J=0,(NLOOPLINE-1)
          PCT(I,J)=0.D0
        enddo
      enddo
      do I=0,3
        do J=1,NLOOPLINE
          PCT(I,0)=PCT(I,0)+PL(I,J)
        enddo
      enddo
      if (checkPConservation) then
        if (PCT(0,0).gt.1.d-6) then
          write(*,*) 'energy is not conserved ',PCT(0,0)
          stop 'energy is not conserved'
        elseif (PCT(1,0).gt.1.d-6) then
         write(*,*) 'px is not conserved ',PCT(1,0)
         stop 'px is not conserved'
        elseif (PCT(2,0).gt.1.d-6) then
         write(*,*) 'py is not conserved ',PCT(2,0)
         stop 'py is not conserved'
        elseif (PCT(3,0).gt.1.d-6) then
         write(*,*) 'pz is not conserved ',PCT(3,0)
         stop 'pz is not conserved'
        endif
      endif
      do I=0,3
        do J=1,(NLOOPLINE-1)
          do K=1,J
            PCT(I,J)=PCT(I,J)+PL(I,K)
          enddo
        enddo
      enddo

      CALL CTSXCUT(CTMODE,LSCALE,MU_R,NLOOPLINE,%(proc_prefix)sLOOPNUM,%(proc_prefix)sMPLOOPNUM,RANK,PCT,M2LCT,RES,ACC,R1,STABLE)
	  %(finalize_CT)s
C     WRITE(*,*) 'Loop ID',ID,' =',RES(1),RES(2),RES(3)
      END

	  SUBROUTINE %(proc_prefix)sINITCT()
C 
C INITIALISATION OF CUTTOOLS
C  
C LOCAL VARIABLES 
C
	  %(real_dp_format)s THRS
	  LOGICAL EXT_NUM_FOR_R1
C  
C GLOBAL VARIABLES 
C
	  include 'MadLoopParams.inc'
C ----------
C BEGIN CODE
C ----------

C DEFAULT PARAMETERS FOR CUTTOOLS
C -------------------------------  
C THRS1 IS THE PRECISION LIMIT BELOW WHICH THE MP ROUTINES ACTIVATES
	  THRS=CTSTABTHRES
C LOOPLIB SET WHAT LIBRARY CT USES
C 1 -> LOOPTOOLS
C 2 -> AVH
C 3 -> QCDLOOP
      LOOPLIB=CTLOOPLIBRARY
C MADLOOP'S NUMERATOR IN THE OPEN LOOP IS MUCH FASTER THAN THE RECONSTRUCTED ONE IN CT. SO WE BETTER USE MADLOOP ONE IN THIS CASE.
      EXT_NUM_FOR_R1=.TRUE.
C -------------------------------	  

C The initialization below is for CT v1.8.+
      CALL CTSINIT(THRS,LOOPLIB,EXT_NUM_FOR_R1)
C The initialization below is for the older stable CT v1.7, still used for now in the beta release.
C      CALL CTSINIT(THRS,LOOPLIB)

	  END
