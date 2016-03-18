	  SUBROUTINE %(proc_prefix)sGOLEMLOOP(NLOOPLINE,PL,M2L,RANK,RES,STABLE)
C  
%(info_lines)s
C 
C Interface between MG5 and Golem95.
C The Golem95 version should be higher than 1.3.0.
C It supports RANK = NLOOPLINE + 1 tensor integrals when 1 < NLOOPLINE < 6.
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
	  IMPLICIT NONE
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
      %(real_dp_format)s coef_check_thrs
      DATA coef_check_thrs/1.0d-13/
      COMMON/%(proc_prefix)scoef_check_thrs/coef_check_thrs

      LOGICAL PASS_COEF_CHECK     
C  
C ARGUMENTS 
C
      INTEGER NLOOPLINE, RANK
      %(real_dp_format)s PL(0:3,NLOOPLINE)
      %(real_dp_format)s PCT(0:3,0:NLOOPLINE-1), ABSPCT(0:3)
      %(real_dp_format)s REF_P
      REAL(ki) PGOLEM(NLOOPLINE,0:3)
      %(mass_dp_format)s M2L(NLOOPLINE)
      COMPLEX(ki) M2LGOLEM(NLOOPLINE)
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
      COMPLEX(KI) %(proc_prefix)sGOLEM_LOOPNUM
      EXTERNAL %(proc_prefix)sGOLEM_LOOPNUM
      LOGICAL %(proc_prefix)sCOMPARE_COEFS_0
      LOGICAL %(proc_prefix)sCOMPARE_COEFS_1
      LOGICAL %(proc_prefix)sCOMPARE_COEFS_2
      LOGICAL %(proc_prefix)sCOMPARE_COEFS_3
      LOGICAL %(proc_prefix)sCOMPARE_COEFS_4
      LOGICAL %(proc_prefix)sCOMPARE_COEFS_5
      LOGICAL %(proc_prefix)sCOMPARE_COEFS_6
C
C GLOBAL VARIABLES
C
      include 'coupl.inc'
	  INTEGER CTMODE
	  %(real_dp_format)s LSCALE
      common/%(proc_prefix)sCT/LSCALE,CTMODE

## if(not AmplitudeReduction) {
      INTEGER ID,SQSOINDEX,R
	  common/%(proc_prefix)sLOOP/ID,SQSOINDEX,R
## }else{
      INTEGER ID,R
	  common/%(proc_prefix)sLOOP/ID,R
## }

      LOGICAL CTINIT, TIRINIT, GOLEMINIT, SAMURAIINIT, NINJAINIT
      COMMON/REDUCTIONCODEINIT/CTINIT, TIRINIT,GOLEMINIT,SAMURAIINIT,NINJAINIT

      INTEGER NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=%(nloop_groups)d)
      INTEGER NSQUAREDSO
      PARAMETER (NSQUAREDSO=%(nSquaredSO)d)
      include 'loop_max_coefs.inc'

## if(not AmplitudeReduction){	  
      %(complex_dp_format)s LOOPCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)
## }else{
      %(complex_dp_format)s LOOPCOEFS(0:LOOPMAXCOEFS-1,NLOOPGROUPS)
## }
      COMMON/%(proc_prefix)sLCOEFS/LOOPCOEFS
C ----------
C BEGIN CODE
C ----------

C The CT initialization is also performed here if not done already because it calls MPINIT of OneLOop which is necessary on some system
      IF (CTINIT) THEN
	    CTINIT=.FALSE.
		CALL %(proc_prefix)sINITCT()
      ENDIF

C INITIALIZE GOLEM IF NEEDED
      IF (GOLEMINIT) THEN
	    GOLEMINIT=.FALSE.
	    CALL %(proc_prefix)sINITGOLEM()
      ENDIF

C  No stability test intrisic to Golem95 now
      STABLE=.TRUE.

C  This initialization must be done for each reduction because we
C  have not setup anyoptimization using pinched propagators yet.
      CALL INITGOLEM95(NLOOPLINE)
      PINCH = 0

C YOU CAN FIND THE DETAILS ABOUT THE DIFFERENT CTMODE AT THE BEGINNING OF THE FILE CTS_CUTS.F90 IN THE CUTTOOLS DISTRIBUTION

C  CONVERT THE MASSES TO BE COMPLEX
      do I=1,NLOOPLINE
	    M2LGOLEM(I)=M2L(I)
	  ENDDO

C  CONVERT THE MOMENTA FLOWING IN THE LOOP LINES TO CT CONVENTIONS
      do I=0,3
        ABSPCT(I) = 0.D0
        do J=0,(NLOOPLINE-1)
          PCT(I,J)=0.D0
        enddo
      enddo
      do I=0,3
        do J=1,NLOOPLINE
          PCT(I,0)=PCT(I,0)+PL(I,J)
          ABSPCT(I)=ABSPCT(I)+ABS(PL(I,J))
        enddo
      enddo
      ref_p = max(ABSPCT(0), ABSPCT(1),ABSPCT(2),ABSPCT(3))
       do I=0,3
           ABSPCT(I) = MAX(ref_p*1e-6, ABSPCT(I))
       enddo
      if (checkPConservation.and.ref_p.gt.1d-8) then
        if ((PCT(0,0)/ABSPCT(0)).gt.1.d-6) then
          write(*,*) 'energy is not conserved ',PCT(0,0)
          stop 'energy is not conserved'
        elseif ((PCT(1,0)/ABSPCT(1)).gt.1.d-6) then
         write(*,*) 'px is not conserved ',PCT(1,0)
         stop 'px is not conserved'
        elseif ((PCT(2,0)/ABSPCT(2)).gt.1.d-6) then
         write(*,*) 'py is not conserved ',PCT(2,0)
         stop 'py is not conserved'
        elseif ((PCT(3,0)/ABSPCT(3)).gt.1.d-6) then
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

C     Now convert the loop momenta to Golem95 conventions
      DO i=0,3
      	 PGOLEM(1,i)=0.0e0_ki
	 DO j=2,NLOOPLINE
	    PGOLEM(j,i)=PCT(i,j-1)
	 ENDDO
      ENDDO

C     Fill in the kinematic s-matrix while taking care of on-shell limits.
      CALL %(proc_prefix)sSETUP_KIN_MATRIX(NLOOPLINE,PGOLEM,M2LGOLEM)
C     Construct the golem internal matrices derived from the kinetic one.
      CALL PREPARESMATRIX()

C     Fill in the golem coefficents and compute the loop
      IF(GOLEM_RUN_MODE.EQ.2)THEN
	res_golem = evaluate_b(%(proc_prefix)sGOLEM_LOOPNUM,PGOLEM,0,RANK)
      ELSE
        PASS_COEF_CHECK=.TRUE.
	SELECT CASE(RANK)
	CASE(0)
	    CALL %(proc_prefix)sFILL_GOLEM_COEFFS_0(LOOPCOEFS(0%(loop_induced_sqsoindex)s,ID),COEFFS0)
	    IF(GOLEM_RUN_MODE.EQ.3)THEN
		COEFFS0_REC = %(proc_prefix)sGOLEM_LOOPNUM(NULL_VEC,0.0_ki)
		PASS_COEF_CHECK=%(proc_prefix)sCOMPARE_COEFS_0(COEFFS0,COEFFS0_REC)
	    ENDIF
	CASE(1)
	    CALL %(proc_prefix)sFILL_GOLEM_COEFFS_1(LOOPCOEFS(0%(loop_induced_sqsoindex)s,ID),COEFFS1)
	    IF(GOLEM_RUN_MODE.EQ.3)THEN
		CALL RECONSTRUCT1(%(proc_prefix)sGOLEM_LOOPNUM,COEFFS1_REC)
		PASS_COEF_CHECK=%(proc_prefix)sCOMPARE_COEFS_1(COEFFS1,COEFFS1_REC)
	    ENDIF
	CASE(2)
	    CALL %(proc_prefix)sFILL_GOLEM_COEFFS_2(LOOPCOEFS(0%(loop_induced_sqsoindex)s,ID),COEFFS2)
	    IF(GOLEM_RUN_MODE.EQ.3)THEN
	        CALL RECONSTRUCT2(%(proc_prefix)sGOLEM_LOOPNUM,COEFFS2_REC)
		PASS_COEF_CHECK=%(proc_prefix)sCOMPARE_COEFS_2(COEFFS2,COEFFS2_REC)
	    ENDIF
	CASE(3)
	    CALL %(proc_prefix)sFILL_GOLEM_COEFFS_3(LOOPCOEFS(0%(loop_induced_sqsoindex)s,ID),COEFFS3)
	    IF(GOLEM_RUN_MODE.EQ.3)THEN
		CALL RECONSTRUCT3(%(proc_prefix)sGOLEM_LOOPNUM,COEFFS3_REC)
		PASS_COEF_CHECK=%(proc_prefix)sCOMPARE_COEFS_3(COEFFS3,COEFFS3_REC)
	    ENDIF
	CASE(4)
	    CALL %(proc_prefix)sFILL_GOLEM_COEFFS_4(LOOPCOEFS(0%(loop_induced_sqsoindex)s,ID),COEFFS4)
	    IF(GOLEM_RUN_MODE.EQ.3)THEN
		CALL RECONSTRUCT4(%(proc_prefix)sGOLEM_LOOPNUM,COEFFS4_REC)
		PASS_COEF_CHECK=%(proc_prefix)sCOMPARE_COEFS_4(COEFFS4,COEFFS4_REC)
	    ENDIF
	CASE(5)
	    CALL %(proc_prefix)sFILL_GOLEM_COEFFS_5(LOOPCOEFS(0%(loop_induced_sqsoindex)s,ID),COEFFS5)
	    IF(GOLEM_RUN_MODE.EQ.3)THEN
		CALL RECONSTRUCT5(%(proc_prefix)sGOLEM_LOOPNUM,COEFFS5_REC)
		PASS_COEF_CHECK=%(proc_prefix)sCOMPARE_COEFS_5(COEFFS5,COEFFS5_REC)
	    ENDIF
	CASE(6)
	    CALL %(proc_prefix)sFILL_GOLEM_COEFFS_6(LOOPCOEFS(0%(loop_induced_sqsoindex)s,ID),COEFFS6)
	    IF(GOLEM_RUN_MODE.EQ.3)THEN
		CALL RECONSTRUCT6(%(proc_prefix)sGOLEM_LOOPNUM,COEFFS6_REC)
		PASS_COEF_CHECK=%(proc_prefix)sCOMPARE_COEFS_6(COEFFS6,COEFFS6_REC)
	    ENDIF
	CASE DEFAULT
	    WRITE(*,*)"Not yet implemented in Golem95 for rank= ",RANK
	    STOP
	END SELECT
	
	IF(.NOT.PASS_COEF_CHECK)THEN
	    WRITE(*,*)'Coefs mismatch for ID ',ID,' and rank ',RANK
	    WRITE(*,*)'Coefs form MadLoop5:'
	    SELECT CASE(RANK)
	    CASE(0)
		WRITE(*,*)'Constant coef = ',COEFFS0
	    CASE(1)
		CALL PRINT_COEFFS(COEFFS1)
	    CASE(2)
		CALL PRINT_COEFFS(COEFFS2)
	    CASE(3)
		CALL PRINT_COEFFS(COEFFS3)
	    CASE(4)
		CALL PRINT_COEFFS(COEFFS4)
	    CASE(5)
		CALL PRINT_COEFFS(COEFFS5)
	    CASE(6)
		CALL PRINT_COEFFS(COEFFS6)
	    END SELECT
	    WRITE(*,*)'Coefs reconstructed by Golem95:'
	    SELECT CASE(RANK)
	    CASE(0)
		WRITE(*,*)'Constant coef = ',COEFFS0_REC
	    CASE(1)
		CALL PRINT_COEFFS(COEFFS1_REC)
	    CASE(2)
		CALL PRINT_COEFFS(COEFFS2_REC)
	    CASE(3)
		CALL PRINT_COEFFS(COEFFS3_REC)
	    CASE(4)
		CALL PRINT_COEFFS(COEFFS4_REC)
	    CASE(5)
	        CALL PRINT_COEFFS(COEFFS5_REC)
	    CASE(6)
		CALL PRINT_COEFFS(COEFFS6_REC)
	    END SELECT
	    STOP
      ENDIF

      SELECT CASE(NLOOPLINE)
      CASE(1)
	 WRITE(*,*)'Golem95 cannot handle with tadpole yet'
	 STOP
      CASE(2)
	 SELECT CASE(RANK)
	 CASE(0)
	     RES_GOLEM = COEFFS0*A20(PINCH)
	 CASE(1)
	     RES_GOLEM = CONTRACT2_1(COEFFS1,PGOLEM,PINCH)
	 CASE(2)
	     RES_GOLEM = CONTRACT2_2(COEFFS2,PGOLEM,PINCH)
	 CASE(3)
	     RES_GOLEM = CONTRACT2_3(COEFFS3,PGOLEM,PINCH)
	 CASE DEFAULT
	     WRITE(*,*)"Golem95 cannot handle with: N,r = ",2,RANK
	     STOP
	 END SELECT
       CASE(3)
	  SELECT CASE(RANK)
	  CASE(0)
	      RES_GOLEM = COEFFS0*A30(PINCH)
	  CASE(1)
	      RES_GOLEM = CONTRACT3_1(COEFFS1,PGOLEM,PINCH)
	  CASE(2)
	      RES_GOLEM = CONTRACT3_2(COEFFS2,PGOLEM,PINCH)
	  CASE(3)
	      RES_GOLEM = CONTRACT3_3(COEFFS3,PGOLEM,PINCH)
	  CASE(4)
	      RES_GOLEM = CONTRACT3_4(COEFFS4,PGOLEM,PINCH)
	  CASE DEFAULT
	      WRITE(*,*)"Golem95 cannot handle with: N,r = ",3,RANK
	      STOP
	  END SELECT
       CASE(4)
	  SELECT CASE(RANK)
	  CASE(0)
	      RES_GOLEM = COEFFS0*A40(PINCH)
	  CASE(1)
	      RES_GOLEM = CONTRACT4_1(COEFFS1,PGOLEM,PINCH)
	  CASE(2)
	      RES_GOLEM = CONTRACT4_2(COEFFS2,PGOLEM,PINCH)
	  CASE(3)
	      RES_GOLEM = CONTRACT4_3(COEFFS3,PGOLEM,PINCH)
	  CASE(4)
	      RES_GOLEM = CONTRACT4_4(COEFFS4,PGOLEM,PINCH)
          CASE(5)
	      RES_GOLEM = CONTRACT4_5(COEFFS5,PGOLEM,PINCH)
	  CASE DEFAULT
	      WRITE(*,*)"Golem95 cannot handle with: N,r = ",4,RANK
	      STOP
	  END SELECT
	CASE(5)
	  SELECT CASE(RANK)
	  CASE(0)
	      RES_GOLEM = COEFFS0*A50(PINCH)
	  CASE(1)
	      RES_GOLEM = CONTRACT5_1(COEFFS1,PGOLEM,PINCH)
	  CASE(2)
	      RES_GOLEM = CONTRACT5_2(COEFFS2,PGOLEM,PINCH)
	  CASE(3)
	      RES_GOLEM = CONTRACT5_3(COEFFS3,PGOLEM,PINCH)
	  CASE(4)
	      RES_GOLEM = CONTRACT5_4(COEFFS4,PGOLEM,PINCH)
	  CASE(5)
	      RES_GOLEM = CONTRACT5_5(COEFFS5,PGOLEM,PINCH)
          CASE(6)
              RES_GOLEM = CONTRACT5_6(COEFFS6,PGOLEM,PINCH)
	  CASE DEFAULT
	      WRITE(*,*)"Golem95 cannot handle with: N,r = ",5,RANK
	      STOP
	  END SELECT
        CASE(6)
	  SELECT CASE(RANK)
	  CASE(0)
	      RES_GOLEM = COEFFS0*A60(PINCH)
	  CASE(1)
	      RES_GOLEM = CONTRACT6_1(COEFFS1,PGOLEM,PINCH)
	  CASE(2)
	      RES_GOLEM = CONTRACT6_2(COEFFS2,PGOLEM,PINCH)
	  CASE(3)
	      RES_GOLEM = CONTRACT6_3(COEFFS3,PGOLEM,PINCH)
	  CASE(4)
	      RES_GOLEM = CONTRACT6_4(COEFFS4,PGOLEM,PINCH)
	  CASE(5)
	      RES_GOLEM = CONTRACT6_5(COEFFS5,PGOLEM,PINCH)
	  CASE(6)
	      RES_GOLEM = CONTRACT6_6(COEFFS6,PGOLEM,PINCH)
	  CASE DEFAULT
	      WRITE(*,*)"Golem95 cannot handle with: N,r = ",6,RANK
	      STOP
	  END SELECT
         CASE DEFAULT
	   WRITE(*,*)"Golem95 cannot handle with: N = ",NLOOPLINE
	   STOP
	 END SELECT
      ENDIF

## if(AmplitudeReduction) {
        RES(1)=NORMALIZATION*(RES_GOLEM%%c+2.0*LOG(MU_R)*RES_GOLEM%%b+2.0*LOG(MU_R)**2*RES_GOLEM%%a)
        RES(2)=NORMALIZATION*(RES_GOLEM%%b+2.0*LOG(MU_R)*RES_GOLEM%%a)
        RES(3)=NORMALIZATION*(RES_GOLEM%%a)
## } else {
        RES(1)=NORMALIZATION*2.0D0*DBLE(RES_GOLEM%%c+2.0*LOG(MU_R)*RES_GOLEM%%b+2.0*LOG(MU_R)**2*RES_GOLEM%%a)
        RES(2)=NORMALIZATION*2.0D0*DBLE(RES_GOLEM%%b+2.0*LOG(MU_R)*RES_GOLEM%%a)
        RES(3)=NORMALIZATION*2.0D0*DBLE(RES_GOLEM%%a)
## }
C     WRITE(*,*) 'Loop ID',ID,' =',RES(1),RES(2),RES(3)

C     Finally free golem memory and cache
      CALL EXITGOLEM95()

      END

         FUNCTION %(proc_prefix)sGOLEM_LOOPNUM(Q,mu2)
	 USE PRECISION_GOLEM, ONLY: KI
	 REAL(KI),DIMENSION(0:3),INTENT(IN)::Q
	 REAL(KI),INTENT(IN)::mu2
	 COMPLEX(KI)::%(proc_prefix)sGOLEM_LOOPNUM
	 
	 %(complex_dp_format)s QQ(0:3),num
	 INTEGER i

	    DO i=0,3
		QQ(i)=CMPLX(Q(i),0.0d0,KIND=16)
	    ENDDO

	    CALL %(proc_prefix)sLOOPNUM(QQ,num)
	    %(proc_prefix)sGOLEM_LOOPNUM=num
	    RETURN
	 END FUNCTION

	  SUBROUTINE %(proc_prefix)sINITGOLEM()
C 
C INITIALISATION OF GOLEM
C
C  
C MODULE
C
	  USE PARAMETRE
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

C DEFAULT PARAMETERS FOR GOLEM
C -------------------------------  
C One can chose here to have either just the rational R1 piece 
C or everything but the R2
       RAT_OR_TOT_PAR = TOT

	  END

     SUBROUTINE %(proc_prefix)sSETUP_KIN_MATRIX(NLOOPLINE,PGOLEM,M2L)
C    
C    MODULE
C
     USE MATRICE_S
     USE PRECISION_GOLEM, ONLY: KI
C    
C    ARGUMENTS
C
     INTEGER NLOOPLINE
     REAL(KI) PGOLEM(NLOOPLINE,0:3)
     COMPLEX(KI) M2L(NLOOPLINE)
C
C    LOCAL VARIABLES
C
     INTEGER I,J
     %(complex_dp_format)s S_MAT_FROM_MG(NLOOPLINE,NLOOPLINE)
C    ----------
C    BEGIN CODE
C    ----------

   	   CALL %(proc_prefix)sBUILD_KINEMATIC_MATRIX(NLOOPLINE,PGOLEM,M2L,S_MAT_FROM_MG) 

       do i=1,NLOOPLINE
         do j=1,NLOOPLINE
	       s_mat(i,j)=S_MAT_FROM_MG(i,j)
         ENDDO
       ENDDO
       
      END

      FUNCTION %(proc_prefix)sCOMPARE_COEFS_0(COEFS_A,COEFS_B)

      USE PRECISION_GOLEM, ONLY: KI
      COMPLEX(KI) COEFS_A,COEFS_B
      %(real_dp_format)s coef_check_thrs
      COMMON/%(proc_prefix)scoef_check_thrs/coef_check_thrs
      %(real_dp_format)s DENOM,NUM
      LOGICAL %(proc_prefix)sCOMPARE_COEFS_0

      NUM = ABS(COEFS_A-COEFS_B)
      DENOM = ABS(COEFS_A+COEFS_B)
      IF(DENOM.GT.0d0)THEN
        %(proc_prefix)sCOMPARE_COEFS_0=((NUM/DENOM).LT.coef_check_thrs)
      ELSE
        %(proc_prefix)sCOMPARE_COEFS_0=(NUM.LT.coef_check_thrs)      
      ENDIF

      END

      FUNCTION %(proc_prefix)sCOMPARE_COEFS_1(COEFS_A,COEFS_B)
      
      USE TENS_REC
      TYPE(COEFF_TYPE_1)COEFS_A,COEFS_B
      %(real_dp_format)s COEF_CHECK_THRS
      COMMON/%(proc_prefix)scoef_check_thrs/coef_check_thrs
      %(real_dp_format)s DENOM,NUM
      LOGICAL %(proc_prefix)sCOMPARE_COEFS_1
      
      NUM = ABS(COEFS_A%%c0-COEFS_B%%c0)+SUM(ABS(COEFS_A%%c1-COEFS_B%%c1))
      DENOM = ABS(COEFS_A%%c0+COEFS_B%%c0)+SUM(ABS(COEFS_A%%c1+COEFS_B%%c1))
      
      IF(DENOM.GT.0d0)THEN
        %(proc_prefix)sCOMPARE_COEFS_1=((NUM/DENOM).LT.coef_check_thrs)
      ELSE
        %(proc_prefix)sCOMPARE_COEFS_1=(NUM.LT.coef_check_thrs)
      ENDIF

      END

      FUNCTION %(proc_prefix)sCOMPARE_COEFS_2(COEFS_A,COEFS_B)
      
      USE TENS_REC
      TYPE(COEFF_TYPE_2) COEFS_A,COEFS_B
      %(real_dp_format)s coef_check_thrs
      COMMON/%(proc_prefix)scoef_check_thrs/coef_check_thrs
      %(real_dp_format)s DENOM,NUM
      LOGICAL %(proc_prefix)sCOMPARE_COEFS_2
      
      NUM = ABS(COEFS_A%%c0-COEFS_B%%c0)+SUM(ABS(COEFS_A%%c1-COEFS_B%%c1))+SUM(ABS(COEFS_A%%c2-COEFS_B%%c2))
      DENOM = ABS(COEFS_A%%c0+COEFS_B%%c0)+SUM(ABS(COEFS_A%%c1+COEFS_B%%c1))+SUM(ABS(COEFS_A%%c2+COEFS_B%%c2))
      IF(DENOM.GT.0d0)THEN
        %(proc_prefix)sCOMPARE_COEFS_2=((NUM/DENOM).LT.coef_check_thrs)
      ELSE
        %(proc_prefix)sCOMPARE_COEFS_2=(NUM.LT.coef_check_thrs)
      ENDIF

      END

      FUNCTION %(proc_prefix)sCOMPARE_COEFS_3(COEFS_A,COEFS_B)

      USE TENS_REC
      TYPE(COEFF_TYPE_3) COEFS_A, COEFS_B
      %(real_dp_format)s coef_check_thrs
      COMMON/%(proc_prefix)scoef_check_thrs/coef_check_thrs
      %(real_dp_format)s DENOM,NUM
      LOGICAL %(proc_prefix)sCOMPARE_COEFS_3

      NUM = ABS(COEFS_A%%c0-COEFS_B%%c0)+SUM(ABS(COEFS_A%%c1-COEFS_B%%c1))+SUM(ABS(COEFS_A%%c2-COEFS_B%%c2))+SUM(ABS(COEFS_A%%c3-COEFS_B%%c3))
      DENOM = ABS(COEFS_A%%c0+COEFS_B%%c0)+SUM(ABS(COEFS_A%%c1+COEFS_B%%c1))+SUM(ABS(COEFS_A%%c2+COEFS_B%%c2))+SUM(ABS(COEFS_A%%c3+COEFS_B%%c3))
      IF(DENOM.GT.0d0)THEN
        %(proc_prefix)sCOMPARE_COEFS_3=((NUM/DENOM).LT.coef_check_thrs)
      ELSE
        %(proc_prefix)sCOMPARE_COEFS_3=(NUM.LT.coef_check_thrs)
      ENDIF

      END	     

      FUNCTION %(proc_prefix)sCOMPARE_COEFS_4(COEFS_A,COEFS_B)
      
      USE TENS_REC
      TYPE(COEFF_TYPE_4) COEFS_A, COEFS_B
      %(real_dp_format)s coef_check_thrs
      COMMON/%(proc_prefix)scoef_check_thrs/coef_check_thrs
      %(real_dp_format)s DENOM,NUM
      LOGICAL %(proc_prefix)sCOMPARE_COEFS_4

      NUM = ABS(COEFS_A%%c0-COEFS_B%%c0)+SUM(ABS(COEFS_A%%c1-COEFS_B%%c1))+SUM(ABS(COEFS_A%%c2-COEFS_B%%c2))+SUM(ABS(COEFS_A%%c3-COEFS_B%%c3))+SUM(ABS(COEFS_A%%c4-COEFS_B%%c4))
      DENOM = ABS(COEFS_A%%c0+COEFS_B%%c0)+SUM(ABS(COEFS_A%%c1+COEFS_B%%c1))+SUM(ABS(COEFS_A%%c2+COEFS_B%%c2))+SUM(ABS(COEFS_A%%c3+COEFS_B%%c3))+SUM(ABS(COEFS_A%%c4+COEFS_B%%c4))
      IF(DENOM.GT.0d0)THEN
        %(proc_prefix)sCOMPARE_COEFS_4=((NUM/DENOM).LT.coef_check_thrs)
      ELSE
        %(proc_prefix)sCOMPARE_COEFS_4=(NUM.LT.coef_check_thrs)
      ENDIF

      END

      FUNCTION %(proc_prefix)sCOMPARE_COEFS_5(COEFS_A,COEFS_B)
      
      USE TENS_REC
      TYPE(COEFF_TYPE_5) COEFS_A,COEFS_B
      %(real_dp_format)s coef_check_thrs
      COMMON/%(proc_prefix)scoef_check_thrs/coef_check_thrs
      %(real_dp_format)s DENOM,NUM
      LOGICAL %(proc_prefix)sCOMPARE_COEFS_5

      NUM = ABS(COEFS_A%%c0-COEFS_B%%c0)+SUM(ABS(COEFS_A%%c1-COEFS_B%%c1))+SUM(ABS(COEFS_A%%c2-COEFS_B%%c2))+SUM(ABS(COEFS_A%%c3-COEFS_B%%c3))+SUM(ABS(COEFS_A%%c4-COEFS_B%%c4))
      DENOM = ABS(COEFS_A%%c0+COEFS_B%%c0)+SUM(ABS(COEFS_A%%c1+COEFS_B%%c1))+SUM(ABS(COEFS_A%%c2+COEFS_B%%c2))+SUM(ABS(COEFS_A%%c3+COEFS_B%%c3))+SUM(ABS(COEFS_A%%c4+COEFS_B%%c4))
      IF(DENOM.GT.0d0)THEN
        %(proc_prefix)sCOMPARE_COEFS_5=((NUM/DENOM).LT.coef_check_thrs)
      ELSE
        %(proc_prefix)sCOMPARE_COEFS_5=(NUM.LT.coef_check_thrs)
      ENDIF

      END

      FUNCTION %(proc_prefix)sCOMPARE_COEFS_6(COEFS_A,COEFS_B)
      
      USE TENS_REC
      TYPE(COEFF_TYPE_6) COEFS_A,COEFS_B
      %(real_dp_format)s coef_check_thrs
      COMMON/%(proc_prefix)scoef_check_thrs/coef_check_thrs
      %(real_dp_format)s DENOM,NUM
      LOGICAL %(proc_prefix)sCOMPARE_COEFS_6

      NUM = ABS(COEFS_A%%c0-COEFS_B%%c0)+SUM(ABS(COEFS_A%%c1-COEFS_B%%c1))+SUM(ABS(COEFS_A%%c2-COEFS_B%%c2))+SUM(ABS(COEFS_A%%c3-COEFS_B%%c3))+SUM(ABS(COEFS_A%%c4-COEFS_B%%c4))
      DENOM = ABS(COEFS_A%%c0+COEFS_B%%c0)+SUM(ABS(COEFS_A%%c1+COEFS_B%%c1))+SUM(ABS(COEFS_A%%c2+COEFS_B%%c2))+SUM(ABS(COEFS_A%%c3+COEFS_B%%c3))+SUM(ABS(COEFS_A%%c4+COEFS_B%%c4))
      IF(DENOM.GT.0d0)THEN
        %(proc_prefix)sCOMPARE_COEFS_6=((NUM/DENOM).LT.coef_check_thrs)
      ELSE
        %(proc_prefix)sCOMPARE_COEFS_6=(NUM.LT.coef_check_thrs)
      ENDIF

      END
