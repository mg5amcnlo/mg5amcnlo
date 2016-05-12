C     ===========================================
C     ===== Beginning of CutTools Interface =====
C     ===========================================
      SUBROUTINE %(proc_prefix)sCTLOOP(NLOOPLINE,PL,M2L,RANK,RES,STABLE)
C  
%(info_lines)s
C 
C Interface between MG5 and CutTools.
C  
%(process_lines)s
C  
C  
C CONSTANTS 
C  
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      LOGICAL checkPConservation
      PARAMETER (checkPConservation=.TRUE.)
      %(real_dp_format)s NORMALIZATION
      PARAMETER (NORMALIZATION = 1.D0/(16.D0*3.14159265358979323846D0**2))
C  
C ARGUMENTS 
C
      INTEGER NLOOPLINE, RANK
      %(real_dp_format)s PL(0:3,NLOOPLINE)
      %(real_dp_format)s PCT(0:3,0:NLOOPLINE-1),ABSPCT(0:3)
      %(real_dp_format)s REF_P
      %(mass_dp_format)s M2L(NLOOPLINE)
      %(complex_dp_format)s M2LCT(0:NLOOPLINE-1)	  
      %(complex_dp_format)s RES(3)
	  LOGICAL STABLE	  
C  
C LOCAL VARIABLES 
C
	  %(complex_dp_format)s R1, ACC
	  INTEGER I, J, K
	  LOGICAL CTINIT, TIRINIT, GOLEMINIT, SAMURAIINIT, NINJAINIT
	  COMMON/REDUCTIONCODEINIT/CTINIT,TIRINIT,GOLEMINIT,SAMURAIINIT,NINJAINIT
C
C EXTERNAL FUNCTIONS
C
      EXTERNAL %(proc_prefix)sLOOPNUM
      EXTERNAL %(proc_prefix)sMPLOOPNUM
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
        ABSPCT(I)=0.D0
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
          write(*,*) 'energy is not conserved (flag:CT95)',PCT(0,0)
          stop 'energy is not conserved (flag:CT95)'
        elseif ((PCT(1,0)/ABSPCT(1)).gt.1.d-6) then
         write(*,*) 'px is not conserved (flag:CT95)',PCT(1,0)
         stop 'px is not conserved (flag:CT95)'
        elseif ((PCT(2,0)/ABSPCT(2)).gt.1.d-6) then
         write(*,*) 'py is not conserved (flag:CT95)',PCT(2,0)
         stop 'py is not conserved (flag:CT95)'
        elseif ((PCT(3,0)/ABSPCT(3)).gt.1.d-6) then
         write(*,*) 'pz is not conserved (flag:CT95)',PCT(3,0)
         stop 'pz is not conserved (flag:CT95)'
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
## if(AmplitudeReduction) {
      RES(1)=NORMALIZATION*RES(1)
      RES(2)=NORMALIZATION*RES(2)
      RES(3)=NORMALIZATION*RES(3)
## } else {
      RES(1)=NORMALIZATION*2.0d0*DBLE(RES(1))
      RES(2)=NORMALIZATION*2.0d0*DBLE(RES(2))
      RES(3)=NORMALIZATION*2.0d0*DBLE(RES(3))
## }
C     WRITE(*,*) 'CutTools: Loop ID',ID,' =',RES(1),RES(2),RES(3)
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

	  SUBROUTINE %(proc_prefix)sBUILD_KINEMATIC_MATRIX(NLOOPLINE,P_LOOP,M2L,S_MAT) 
C
C       Helper function that compute the loop kinematic matrix with proper thresholds
C       NLOOPLINE : Number of loop lines
C       P_LOOP    : List of external momenta running in the loop, i.e. q_i in the denominator (l_i+q_i)**2-m_i**2
C       M2L       : List of complex-valued masses running in the loop.
C       S_MAT(N,N): Kinematic matrix output.
C
C    	ARGUMENTS
C
		INTEGER NLOOPLINE
    	%(real_dp_format)s P_LOOP(NLOOPLINE,0:3)
    	%(complex_dp_format)s M2L(NLOOPLINE)
    	%(complex_dp_format)s S_MAT(NLOOPLINE,NLOOPLINE)
C    
C    	GLOBAL VARIABLES
C
     	include 'MadLoopParams.inc'
C
C    	LOCAL VARIABLES
C
     	INTEGER I,J,K
     	%(complex_dp_format)s DIFFSQ
     	%(real_dp_format)s REF_NORMALIZATION

C     	----------
C     	BEGIN CODE
C     	----------

     	do i=1,NLOOPLINE
       	  do j=1,NLOOPLINE
            
		    if(i.eq.j)then
	          s_mat(i,j)=-(M2L(i)+M2L(j))
	        else
	          DIFFSQ = (DCMPLX(P_LOOP(i,0),0.0d0)-DCMPLX(P_LOOP(j,0),0.0d0))**2
	          do k=1,3
	            DIFFSQ = DIFFSQ - (DCMPLX(P_LOOP(i,k),0.0d0)-DCMPLX(P_LOOP(j,k),0.0d0))**2
	          enddo
C             Default value of the kinematic matrix
	          s_mat(i,j)=DIFFSQ-M2L(i)-M2L(j)
C             And we now test various thresholds. Normaly, at most one applies.
	          IF(ABS(M2L(i)).NE.0.0D0)THEN
	            IF(ABS((DIFFSQ-M2L(i))/M2L(i)).LT.OSThres)THEN
	              s_mat(i,j)=-M2L(j)
 	            ENDIF
	          ENDIF
		      IF(ABS(M2L(j)).NE.0.0D0)THEN
	            IF(ABS((DIFFSQ-M2L(j))/M2L(j)).LT.OSThres)THEN
	              s_mat(i,j)=-M2L(i)
	            ENDIF
		      ENDIF
C             Choose what seems the most appropriate way to compare
C             massless onshellness.
              REF_NORMALIZATION=0.0d0
C             Here, we chose to base the threshold only on the energy component
	          do k=0,0
		        REF_NORMALIZATION = REF_NORMALIZATION + ABS(P_LOOP(i,k)) + ABS(P_LOOP(J,k))
		      ENDDO
			  REF_NORMALIZATION = (REF_NORMALIZATION/2.0d0)**2
	          IF(REF_NORMALIZATION.NE.0.0D0)THEN
                IF(ABS(DIFFSQ/REF_NORMALIZATION).LT.OSThres)THEN
	              s_mat(i,j)=-(M2L(i)+M2L(j))
			    ENDIF
	          ENDIF
            ENDIF

          ENDDO
        ENDDO

	  END

	  SUBROUTINE %(proc_prefix)sMP_BUILD_KINEMATIC_MATRIX(NLOOPLINE,P_LOOP,M2L,S_MAT) 
C
C       Helper function that compute the loop kinematic matrix with proper thresholds
C       NLOOPLINE : Number of loop lines
C       P_LOOP    : List of external momenta running in the loop, i.e. q_i in the denominator (l_i+q_i)**2-m_i**2
C       M2L       : List of complex-valued masses running in the loop.
C       S_MAT(N,N): Kinematic matrix output.
C
C    	ARGUMENTS
C
		INTEGER NLOOPLINE
    	%(real_mp_format)s P_LOOP(NLOOPLINE,0:3)
    	%(complex_mp_format)s M2L(NLOOPLINE)
    	%(complex_mp_format)s S_MAT(NLOOPLINE,NLOOPLINE)
C    
C    	GLOBAL VARIABLES
C
     	include 'MadLoopParams.inc'
C
C    	LOCAL VARIABLES
C
     	INTEGER I,J,K
     	%(complex_mp_format)s DIFFSQ
     	%(real_mp_format)s REF_NORMALIZATION

C     	----------
C     	BEGIN CODE
C     	----------

     	do i=1,NLOOPLINE
       	  do j=1,NLOOPLINE
            
		    if(i.eq.j)then
	          s_mat(i,j)=-(M2L(i)+M2L(j))
	        else
	          DIFFSQ = (CMPLX(P_LOOP(i,0),0.0e0_16,KIND=16)-CMPLX(P_LOOP(j,0),0.0e0_16,KIND=16))**2
	          do k=1,3
	            DIFFSQ = DIFFSQ - (CMPLX(P_LOOP(i,k),0.0e0_16,KIND=16)-CMPLX(P_LOOP(j,k),0.0e0_16,KIND=16))**2
	          enddo
C             Default value of the kinematic matrix
	          s_mat(i,j)=DIFFSQ-M2L(i)-M2L(j)
C             And we now test various thresholds. Normaly, at most one applies.
	          IF(ABS(M2L(i)).NE.0.0e0_16)THEN
	            IF(ABS((DIFFSQ-M2L(i))/M2L(i)).LT.OSThres)THEN
	              s_mat(i,j)=-M2L(j)
 	            ENDIF
	          ENDIF
		      IF(ABS(M2L(j)).NE.0.0e0_16)THEN
	            IF(ABS((DIFFSQ-M2L(j))/M2L(j)).LT.OSThres)THEN
	              s_mat(i,j)=-M2L(i)
	            ENDIF
		      ENDIF
C             Choose what seems the most appropriate way to compare
C             massless onshellness.
              REF_NORMALIZATION=0.0e0_16
C             Here, we chose to base the threshold only on the energy component
	          do k=0,0
		        REF_NORMALIZATION = REF_NORMALIZATION + ABS(P_LOOP(i,k)) + ABS(P_LOOP(J,k))
		      ENDDO
			  REF_NORMALIZATION = (REF_NORMALIZATION/2.0e0_16)**2
	          IF(REF_NORMALIZATION.NE.0.0e0_16)THEN
                IF(ABS(DIFFSQ/REF_NORMALIZATION).LT.OSThres)THEN
	              s_mat(i,j)=-(M2L(i)+M2L(j))
			    ENDIF
	          ENDIF
            ENDIF

          ENDDO
        ENDDO

	  END

## if(samurai_available){

C     ===========================================
C     ===== Beginning of Samurai interface  =====
C     ===========================================

      SUBROUTINE %(proc_prefix)sSAMURAI_LOOP(NLOOPLINE,PL,M2L,RANK,RES,STABLE)
C    
C     Module used
C
      use msamurai
      IMPLICIT NONE	  
C  
%(info_lines)s
C 
C Interface between MG5 and Samurai.
C  
%(process_lines)s
C  
C  
C CONSTANTS 
C  
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      LOGICAL checkPConservation
      PARAMETER (checkPConservation=.TRUE.)
      %(real_dp_format)s NORMALIZATION
      PARAMETER (NORMALIZATION = 1.D0/(16.D0*3.14159265358979323846D0**2))
	  INTEGER SAM_REDUCTION_STEPS
C     This tells Samurai to perform the reduction all the way to tadpoles.
	  PARAMETER (SAM_REDUCTION_STEPS=1)
C  
C ARGUMENTS 
C
      INTEGER NLOOPLINE, RANK
      %(real_dp_format)s PL(0:3,NLOOPLINE)
      %(real_dp_format)s P_TMP(0:3,0:NLOOPLINE-1), ABSP_TMP(0:3)
      %(real_dp_format)s REF_P
      %(real_dp_format)s P_SAM(0:NLOOPLINE-1,4)
      %(mass_dp_format)s M2L(NLOOPLINE)
      %(complex_dp_format)s M2L_SAM(0:NLOOPLINE-1)	  
      %(complex_dp_format)s RES(3), SAM_RES(-2:0)
	  LOGICAL STABLE	  
C  
C LOCAL VARIABLES 
C
	  %(complex_dp_format)s R1, ACC
	  INTEGER I, J, K
      %(real_dp_format)s PDEN_DUMMY(0:3,NLOOPLINE-1)
	  LOGICAL CTINIT, TIRINIT, GOLEMINIT, SAMURAIINIT, NINJAINIT
	  COMMON/REDUCTIONCODEINIT/CTINIT,TIRINIT,GOLEMINIT,SAMURAIINIT,NINJAINIT
C
C EXTERNAL FUNCTIONS
C
C     Hook for numerator function for Samurai
      %(complex_dp_format)s %(proc_prefix)sSAMURAI_LOOPNUM
      External %(proc_prefix)sSAMURAI_LOOPNUM
C
C GLOBAL VARIABLES
C
      include 'coupl.inc'

	  %(real_dp_format)s LSCALE
	  INTEGER CTMODE
      common/%(proc_prefix)sCT/LSCALE,CTMODE

      INTEGER ID,SQSOINDEX,R
      COMMON/%(proc_prefix)sLOOP/ID,SQSOINDEX,R

C ----------
C BEGIN CODE
C ----------

C     For the direction test, we must switch the direction in which the loop is read for CTMode equal to 2 or 4.
      CALL %(proc_prefix)sSWITCH_ORDER(CTMODE,NLOOPLINE,PL,PDEN_DUMMY,M2L)

C The CT initialization is also performed here if not done already because it calls MPINIT of OneLOop which is necessary on some system
      IF (CTINIT) THEN
	    CTINIT=.FALSE.
		CALL %(proc_prefix)sINITCT()
      ENDIF

C INITIALIZE SAMURAI IF NEEDED
      IF (SAMURAIINIT) THEN
	    SAMURAIINIT=.FALSE.
		CALL %(proc_prefix)sINITIALIZESAMURAI()
      ENDIF

C  CONVERT THE MASSES TO BE COMPLEX
      do I=1,NLOOPLINE
	    M2L_SAM(I-1)=M2L(I)
	  ENDDO

C  CONVERT THE MOMENTA FLOWING IN THE LOOP LINES TO SAMURAI CONVENTIONS
      do I=0,3
        ABSP_TMP(I)=0.D0		
        do J=0,(NLOOPLINE-1)
          P_TMP(I,J)=0.D0
        enddo
      enddo
      do I=0,3
        do J=1,NLOOPLINE
          P_TMP(I,0)=P_TMP(I,0)+PL(I,J)
          ABSP_TMP(I)=ABSP_TMP(I)+ABS(PL(I,J))
        enddo
      enddo
      ref_p = max(ABSP_TMP(0), ABSP_TMP(1),ABSP_TMP(2),ABSP_TMP(3))
       do I=0,3
           ABSP_TMP(I) = MAX(ref_p*1e-6, ABSP_TMP(I))
       enddo
      if (checkPConservation.and.ref_p.gt.1d-8) then
        if ((P_TMP(0,0)/ABSP_TMP(0)).gt.1.d-6) then
          write(*,*) 'energy is not conserved (flag:CT404)',P_TMP(0,0)
          stop 'energy is not conserved (flag:CT404)'
        elseif ((P_TMP(1,0)/ABSP_TMP(1)).gt.1.d-6) then
         write(*,*) 'px is not conserved (flag:CT404)',P_TMP(1,0)
         stop 'px is not conserved (flag:CT404)'
        elseif ((P_TMP(2,0)/ABSP_TMP(2)).gt.1.d-6) then
         write(*,*) 'py is not conserved  (flag:CT404)',P_TMP(2,0)
         stop 'py is not conserved (flag:CT404)'
        elseif ((P_TMP(3,0)/ABSP_TMP(3)).gt.1.d-6) then
         write(*,*) 'pz is not conserved (flag:CT404)',P_TMP(3,0)
         stop 'pz is not conserved (flag:CT404)'
        endif
      endif
      do I=0,3
        do J=1,(NLOOPLINE-1)
          do K=1,J
            P_TMP(I,J)=P_TMP(I,J)+PL(I,K)
          enddo
        enddo
      enddo

C     In samurai, the first index is the momentum one (with energy last) and the second one is the leg number
      DO I=0,NLOOPLINE-1
        P_SAM(I,1) = P_TMP(1,I)
        P_SAM(I,2) = P_TMP(2,I)
        P_SAM(I,3) = P_TMP(3,I)
        P_SAM(I,4) = P_TMP(0,I)        
      ENDDO

C     Optionally one can set the kinematic matrix fo samurai
      CALL %(proc_prefix)sSET_SAMURAI_S_MAT(NLOOPLINE,P_TMP,M2L)

      CALL samurai(%(proc_prefix)sSAMURAI_LOOPNUM,SAM_RES,R1,P_SAM,M2L_SAM,NLOOPLINE,RANK,SAM_REDUCTION_STEPS,MU_R**2,STABLE)
## if(AmplitudeReduction) {
      RES(1)=NORMALIZATION*SAM_RES(0)
      RES(2)=NORMALIZATION*SAM_RES(-1)
      RES(3)=NORMALIZATION*SAM_RES(-2)
## } else {
      RES(1)=NORMALIZATION*2.0d0*DBLE(SAM_RES(0))
      RES(2)=NORMALIZATION*2.0d0*DBLE(SAM_RES(-1))
      RES(3)=NORMALIZATION*2.0d0*DBLE(SAM_RES(-2))
## }
C     WRITE(*,*) 'Samurai:  Loop ID',ID,' =',RES(1),RES(2),RES(3)
      END

C     WRAPPER for the loop numerator function in Samurai.
C     =====================================================
      FUNCTION %(proc_prefix)sSAMURAI_LOOPNUM(icut_dummy,Q_IN,mu2_dummy)
      implicit none
C     
C     ARGUMENTS 
C     
      %(complex_dp_format)s %(proc_prefix)sSAMURAI_LOOPNUM
      integer icut_dummy
      %(real_dp_format)s mu2_dummy
      %(complex_dp_format)s Q_IN(4),Q(0:3)
      %(complex_dp_format)s RES

      RES=(0.0D0,0.0D0)
      Q(0)=Q_IN(4)
      Q(1)=Q_IN(1)
      Q(2)=Q_IN(2)
      Q(3)=Q_IN(3)
      CALL %(proc_prefix)sLOOPNUM(Q,RES)

      %(proc_prefix)sSAMURAI_LOOPNUM = RES

      END


	  SUBROUTINE %(proc_prefix)sSET_SAMURAI_S_MAT(NLOOPLINE,P_LOOP_IN,M2L)
C    
C     Module used
C
      use mgetkin, only: s_mat
C
C ARGUMENTS
C
	  integer NLOOPLINE
	  %(complex_dp_format)s M2L(NLOOPLINE)
      %(real_dp_format)s P_LOOP_IN(0:3,0:NLOOPLINE-1)
C
C LOCAL VARIABLES 
C
      INTEGER I,J,K
	  %(real_dp_format)s P_LOOP(NLOOPLINE,0:3)
	  %(complex_dp_format)s S_MAT_OUT(NLOOPLINE,NLOOPLINE)	  
C
C BEGIN CODE
C 
      DO I=1,NLOOPLINE
	    DO J=0,3
		  P_LOOP(I,J)=P_LOOP_IN(J,I-1)
		ENDDO
      ENDDO
	  CALL %(proc_prefix)sBUILD_KINEMATIC_MATRIX(NLOOPLINE,P_LOOP,M2L,S_MAT_OUT)
	  IF(ALLOCATED(s_mat)) THEN
	    DEALLOCATE(s_mat)
	  ENDIF
	  ALLOCATE(S_MAT(NLOOPLINE,NLOOPLINE))
	  DO I=1,NLOOPLINE
	    DO J=1,NLOOPLINE
		  S_MAT(I,J) = S_MAT_OUT(I,J)
		ENDDO
	  ENDDO
	  
	  END

	  SUBROUTINE %(proc_prefix)sINITIALIZESAMURAI()
C    
C     Module used
C
      use msamurai
C 
C INITIALISATION OF SAMURAI 
C  
C LOCAL VARIABLES 
C
      INTEGER SAM_verbosity
      INTEGER LOOPLIB, STABILITYTEST
      character*4 NumeratorForm
C  
C GLOBAL VARIABLES 
C
	  include 'MadLoopParams.inc'
C ----------
C BEGIN CODE
C ----------

C     DEFAULT PARAMETERS FOR SAMURAI 
C     -------------------------------  
C     LOOPLIB SET WHAT LIBRARY SAMURAI USES
C     1 -> LOOPTOOLS
C     2 -> AVH
C     3 -> QCDLOOP
      IF (CTLOOPLIBRARY.eq.1) THEN
          write(*,*) "Warning in Samurai initialization. LoopTools is not supported by the Samurai interface. It will use OneLOop instead."
          LOOPLIB = 2
      ELSEIF (CTLOOPLIBRARY.eq.3) THEN
          LOOPLIB = 1
      ELSEIF (CTLOOPLIBRARY.eq.2) THEN
          LOOPLIB = 2
      ELSE
          write(*,*) "Error in Samurai initialization. Loop library ID=",CTLOOPLIBRARY," is not supported. Change variable CTLoopLibrary in MadLoopParams.dat."
          stop 1
      ENDIF
C     Stability tests are:
C      0 : None
C      1 : Global N=N test
C      2 : Local N=N test
C      3 : Power test
      STABILITYTEST = 0
C     Verbosity
C       verbosity=0, no output;
C       verbosity=1, the coefficients are printed;
C       verbosity=2, the value of the MIs are printed as well;
C       verbosity=3, the outcome of the numerical test appears.
      SAM_verbosity = 0

C     Tell samurai our loo propagators don't contain denominators (i.e.
C     otherwise one must specify 'tree')
      NumeratorForm = 'diag'

C     -------------------------------
      call initsamurai(NumeratorForm,LOOPLIB,SAM_verbosity,STABILITYTEST)

      END
## }

## if(ninja_available){

C     ===========================================
C     ===== Beginning of Ninja interface  =====
C     ===========================================

      SUBROUTINE %(proc_prefix)sNINJA_LOOP(NLOOPLINE,PL,M2L,RANK,RES,STABLE)
C    
C     Module used
C
      use mninja
C  
%(info_lines)s
C 
C Interface between MG5 and Ninja.
C  
%(process_lines)s
C  
C  
C CONSTANTS 
C  
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      LOGICAL checkPConservation
      PARAMETER (checkPConservation=.TRUE.)
      %(real_dp_format)s NORMALIZATION
      PARAMETER (NORMALIZATION = 1.D0/(16.D0*3.14159265358979323846D0**2))
      INTEGER NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=%(nloop_groups)d)
C     These are constants related to the split orders
      INTEGER NSQUAREDSO
      PARAMETER (NSQUAREDSO=%(nSquaredSO)d)
      include 'loop_max_coefs.inc'
C  
C ARGUMENTS 
C
      INTEGER NLOOPLINE, RANK
      %(real_dp_format)s PL(0:3,NLOOPLINE)
      %(mass_dp_format)s M2L(NLOOPLINE)
      %(complex_dp_format)s RES(3)
	  LOGICAL STABLE	  
C  
C LOCAL VARIABLES 
C
      %(real_dp_format)s P_TMP(0:3,0:NLOOPLINE-1),ABSP_TMP(0:3)
      %(real_dp_format)s REF_P
      %(real_dp_format)s P_NINJA(0:3,NLOOPLINE)
      %(real_dp_format)s P_S_MAT(NLOOPLINE,0:3)	  
      %(mass_dp_format)s M2L_NINJA(NLOOPLINE)
      %(complex_dp_format)s NINJA_RES(0:2)
	  %(complex_dp_format)s R1
	  INTEGER NINJA_STATUS
	  INTEGER I, J, K
      %(real_dp_format)s PDEN_DUMMY(0:3,NLOOPLINE-1)

      %(complex_dp_format)s S_MAT(NLOOPLINE,NLOOPLINE)
      %(real_dp_format)s REAL_S_MAT(NLOOPLINE,NLOOPLINE)

      INTEGER CURR_MAXCOEF	  
	  %(complex_dp_format)s, ALLOCATABLE :: TENSORCOEFS(:)

C
C GLOBAL VARIABLES
C
      include 'coupl.inc'

	  LOGICAL CTINIT, TIRINIT, GOLEMINIT, SAMURAIINIT, NINJAINIT
	  COMMON/REDUCTIONCODEINIT/CTINIT,TIRINIT,GOLEMINIT,SAMURAIINIT,NINJAINIT

	  %(real_dp_format)s LSCALE
	  INTEGER CTMODE
      common/%(proc_prefix)sCT/LSCALE,CTMODE

      INTEGER ID,SQSOINDEX,R
      COMMON/%(proc_prefix)sLOOP/ID,SQSOINDEX,R
## if(not AmplitudeReduction){
       %(complex_dp_format)s LOOPCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)
## }else{
       %(complex_dp_format)s LOOPCOEFS(0:LOOPMAXCOEFS-1,NLOOPGROUPS)
## }
       COMMON/%(proc_prefix)sLCOEFS/LOOPCOEFS

      LOGICAL FPE_IN_DP_REDUCTION, FPE_IN_QP_REDUCTION
	  COMMON/%(proc_prefix)sFPE_IN_REDUCTION/FPE_IN_DP_REDUCTION, FPE_IN_QP_REDUCTION

C ----------
C BEGIN CODE
C ----------

C     For the direction test, we must switch the direction in which the loop is read for CTMode equal to 2 or 4.
      CALL %(proc_prefix)sSWITCH_ORDER(CTMODE,NLOOPLINE,PL,PDEN_DUMMY,M2L)

C The CT initialization is also performed here if not done already because it calls MPINIT of OneLOop which is necessary on some system
      IF (CTINIT) THEN
	    CTINIT=.FALSE.
		CALL %(proc_prefix)sINITCT()
      ENDIF

C INITIALIZE NINJA IF NEEDED
      IF (NINJAINIT) THEN
	    NINJAINIT=.FALSE.
		CALL %(proc_prefix)sINITNINJA()
      ENDIF

C  CONVERT THE MASSES TO BE COMPLEX
      do I=1,NLOOPLINE
	    M2L_NINJA(I)=M2L(I)
	  ENDDO

C  CONVERT THE MOMENTA FLOWING IN THE LOOP LINES TO NINJA CONVENTIONS
      do I=0,3
        ABSP_TMP = 0.D0
        do J=0,(NLOOPLINE-1)
          P_TMP(I,J)=0.D0
        enddo
      enddo
      do I=0,3
        do J=1,NLOOPLINE
          P_TMP(I,0)=P_TMP(I,0)+PL(I,J)
	  ABSP_TMP(I) = ABSP_TMP(I)+ABS(PL(I,J))
        enddo
      enddo
      ref_p = max(ABSP_TMP(0), ABSP_TMP(1),ABSP_TMP(2),ABSP_TMP(3))
       do I=0,3
           ABSP_TMP(I) = MAX(ref_p*1e-6, ABSP_TMP(I))
       enddo 

      if (checkPConservation.and.ref_p.gt.1d-8) then
        if ((P_TMP(0,0)/ABSP_TMP(0)).gt.1.d-6) then
          write(*,*) 'energy is not conserved (flag:CT692)',P_TMP(0,0)
          stop 'energy is not conserved (flag:CT692)'
        elseif ((P_TMP(1,0)/ABSP_TMP(1)).gt.1.d-6) then
         write(*,*) 'px is not conserved (flag:CT692)',P_TMP(1,0)
         stop 'px is not conserved (flag:CT692)'
        elseif ((P_TMP(2,0)/ABSP_TMP(2)).gt.1.d-6) then
         write(*,*) 'py is not conserved (flag:CT692)',P_TMP(2,0)
         stop 'py is not conserved (flag:CT692)'
        elseif ((P_TMP(3,0)/ABSP_TMP(3)).gt.1.d-6) then
         write(*,*) 'pz is not conserved (flag:CT692)',P_TMP(3,0)
         stop 'pz is not conserved (flag:CT692)'
        endif
      endif
      do I=0,3
        do J=1,(NLOOPLINE-1)
          do K=1,J
            P_TMP(I,J)=P_TMP(I,J)+PL(I,K)
          enddo
        enddo
      enddo
C     In Ninja, the loop line index starts at 1
      DO I=0,NLOOPLINE-1
        P_NINJA(0,I+1) = P_TMP(0,I)
        P_NINJA(1,I+1) = P_TMP(1,I)
        P_NINJA(2,I+1) = P_TMP(2,I)
        P_NINJA(3,I+1) = P_TMP(3,I)
      ENDDO

C     Number of coefficients for the current rank
      CURR_MAXCOEF = 0
      DO I=0,RANK
        CURR_MAXCOEF=CURR_MAXCOEF+(3+I)*(2+I)*(1+I)/6
      ENDDO
C     Now write the tensor coefficients for Ninja
C     It should never be allocated at this stage
      IF (.NOT. ALLOCATED(TENSORCOEFS)) THEN
        ALLOCATE(TENSORCOEFS(0:CURR_MAXCOEF-1))
      ENDIF
      DO I=0,CURR_MAXCOEF-1
## if(AmplitudeReduction) {
        TENSORCOEFS(I) = LOOPCOEFS(I,ID)
## } else {
        TENSORCOEFS(I) = LOOPCOEFS(I,SQSOINDEX,ID)
## }
      ENDDO
C     The loop momentum is in fact q_loop -> -q_loop, so that the
C     coefficients must be changed accordingly
      CALL %(proc_prefix)sINVERT_MOMENTA_IN_POLYNOMIAL(CURR_MAXCOEF,TENSORCOEFS)

C     Compute the kinematic matrix
      DO J=1,NLOOPLINE
        DO I=0,3
          P_S_MAT(J,I)=P_NINJA(I,J)
        ENDDO
      ENDDO
      CALL %(proc_prefix)sBUILD_KINEMATIC_MATRIX(NLOOPLINE,P_S_MAT,M2L,S_MAT)

	  DO I=1,NLOOPLINE
	    DO J=1,NLOOPLINE
		  REAL_S_MAT(I,J) = DBLE(S_MAT(I,J)+M2L(i)+M2L(j))
		ENDDO
	  ENDDO

C     Below is the call specifying the kinematic matrix
	  call ninja_tensor_evaluate(TENSORCOEFS,NLOOPLINE,RANK,REAL_S_MAT,P_NINJA,M2L,MU_R**2,NINJA_RES,R1,NINJA_STATUS)

C     Below is the call without specification of the kinematic matrix
c      call ninja_tensor_evaluate(TENSORCOEFS,NLOOPLINE,RANK,P_NINJA,M2L,MU_R**2,NINJA_RES,R1,NINJA_STATUS)

C     If a floating point exception was found in Ninja (e.g. exactly zero gram. det.)
C     Then warn loop_matrix.f so that it will flag this kinematic point as unstable no matter what.
      IF (NINJA_STATUS.EQ.NINJA_UNSTABLE_KINEMATICS) THEN
	    FPE_IN_DP_REDUCTION = .TRUE. 
	  ENDIF

C     Make sure to deallocate the tensor of coefficients
      IF (ALLOCATED(TENSORCOEFS)) THEN
        DEALLOCATE(TENSORCOEFS)
      ENDIF

## if(AmplitudeReduction) {
      RES(1)=NORMALIZATION*NINJA_RES(0)
      RES(2)=NORMALIZATION*NINJA_RES(1)
      RES(3)=NORMALIZATION*NINJA_RES(2)
## } else {
      RES(1)=NORMALIZATION*2.0d0*DBLE(NINJA_RES(0))
      RES(2)=NORMALIZATION*2.0d0*DBLE(NINJA_RES(1))
      RES(3)=NORMALIZATION*2.0d0*DBLE(NINJA_RES(2))
## }
C     WRITE(*,*) 'Ninja:  Loop ID',ID,' =',RES(1),RES(2),RES(3)
      END

## if(ninja_supports_quad_prec){
C
C Quadruple precision version of loop_ninja
C
      SUBROUTINE %(proc_prefix)sMP_NINJA_LOOP(NLOOPLINE,PL,M2L,RANK,RES,STABLE)
C    
C     Module used
C
      use mninja
C  
%(info_lines)s
C 
C Interface between MG5 and Ninja.
C  
%(process_lines)s
C  
C  
C CONSTANTS 
C  
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      LOGICAL checkPConservation
      PARAMETER (checkPConservation=.TRUE.)
      %(real_dp_format)s NORMALIZATION
      PARAMETER (NORMALIZATION = 1.D0/(16.D0*3.14159265358979323846D0**2))
      INTEGER NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=%(nloop_groups)d)
C     These are constants related to the split orders
      INTEGER NSQUAREDSO
      PARAMETER (NSQUAREDSO=%(nSquaredSO)d)
      include 'loop_max_coefs.inc'
C  
C ARGUMENTS 
C
      INTEGER NLOOPLINE, RANK
      %(real_mp_format)s PL(0:3,NLOOPLINE)
      %(mass_dp_format)s M2L(NLOOPLINE)
      %(complex_dp_format)s RES(3)
	  LOGICAL STABLE	  
C  
C LOCAL VARIABLES 
C
      %(real_mp_format)s NINJA_SCALE
      %(real_mp_format)s P_TMP(0:3,0:NLOOPLINE-1), ABSP_TMP(0:3)
      %(real_dp_format)s REF_P
      real(KI_QNIN) MP_P_NINJA(0:3,NLOOPLINE)
      %(real_mp_format)s MP_P(0:3,NLOOPLINE)	  
      %(real_mp_format)s P_S_MAT(NLOOPLINE,0:3)
      %(complex_mp_format)s MP_M2L(NLOOPLINE)
      complex(KI_QNIN) MP_M2L_NINJA(NLOOPLINE)
      complex(KI_QNIN) NINJA_RES(0:2)
	  complex(KI_QNIN) NINJA_R1
	  %(complex_dp_format)s R1
	  %(complex_dp_format)s DP_RES(0:2)
	  INTEGER NINJA_STATUS
	  INTEGER I, J, K
      %(real_mp_format)s PDEN_DUMMY(0:3,NLOOPLINE-1)

      %(complex_mp_format)s MP_S_MAT(NLOOPLINE,NLOOPLINE)
      %(real_mp_format)s MP_REAL_S_MAT(NLOOPLINE,NLOOPLINE)
      real(KI_QNIN) MP_REAL_S_MAT_NINJA(NLOOPLINE,NLOOPLINE)

      INTEGER CURR_MAXCOEF	  
	  %(complex_mp_format)s, ALLOCATABLE :: MP_TENSORCOEFS(:)
	  complex(KI_QNIN), ALLOCATABLE :: MP_NINJA_TENSORCOEFS(:)

C
C GLOBAL VARIABLES
C
      include 'coupl.inc'

	  LOGICAL CTINIT, TIRINIT, GOLEMINIT, SAMURAIINIT, NINJAINIT
	  COMMON/REDUCTIONCODEINIT/CTINIT,TIRINIT,GOLEMINIT,SAMURAIINIT,NINJAINIT

	  %(real_dp_format)s LSCALE
	  INTEGER CTMODE
      common/%(proc_prefix)sCT/LSCALE,CTMODE

      INTEGER ID,SQSOINDEX,R
      COMMON/%(proc_prefix)sLOOP/ID,SQSOINDEX,R
## if(not AmplitudeReduction){
       %(complex_mp_format)s MP_LOOPCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)
## }else{
       %(complex_mp_format)s MP_LOOPCOEFS(0:LOOPMAXCOEFS-1,NLOOPGROUPS)
## }
       COMMON/%(proc_prefix)sMP_LCOEFS/MP_LOOPCOEFS

      LOGICAL FPE_IN_DP_REDUCTION, FPE_IN_QP_REDUCTION
	  COMMON/%(proc_prefix)sFPE_IN_REDUCTION/FPE_IN_DP_REDUCTION, FPE_IN_QP_REDUCTION

C ----------
C BEGIN CODE
C ----------

C     Cast the masses in complex quadruple precision
      DO I=1,NLOOPLINE
	    MP_M2L(I) = CMPLX(M2L(I),KIND=16)
	  ENDDO

C     For the direction test, we must switch the direction in which the loop is read for CTMode equal to 2 or 4.
      CALL %(proc_prefix)sMP_SWITCH_ORDER(CTMODE,NLOOPLINE,PL,PDEN_DUMMY,MP_M2L)

C The CT initialization is also performed here if not done already because it calls MPINIT of OneLOop which is necessary on some system
      IF (CTINIT) THEN
	    CTINIT=.FALSE.
		CALL %(proc_prefix)sINITCT()
      ENDIF

C INITIALIZE NINJA IF NEEDED
      IF (NINJAINIT) THEN
	    NINJAINIT=.FALSE.
		CALL %(proc_prefix)sINITNINJA()
      ENDIF

C  CONVERT THE MOMENTA FLOWING IN THE LOOP LINES TO NINJA CONVENTIONS
      do I=0,3
        ABSP_TMP(I)=0.e0+0_16
        do J=0,(NLOOPLINE-1)
          P_TMP(I,J)=0.e0+0_16
        enddo
      enddo
      do I=0,3
        do J=1,NLOOPLINE
          P_TMP(I,0)=P_TMP(I,0)+PL(I,J)
          ABSP_TMP(I)=ABSP_TMP(I)+ABS(PL(I,J))
        enddo
      enddo
      ref_p = max(ABSP_TMP(0), ABSP_TMP(1),ABSP_TMP(2),ABSP_TMP(3))
       do I=0,3
           ABSP_TMP(I) = MAX(ref_p*1e-6, ABSP_TMP(I))
       enddo
      if (checkPConservation.and.ref_p.gt.1.e-8_16) then
        if ((P_TMP(0,0)/ABSP_TMP(0)).gt.1.e-6_16) then
          write(*,*) 'energy is not conserved (flag:CT968)',DBLE(P_TMP(0,0))
          stop 'energy is not conserved (flag:CT968)'
        elseif ((P_TMP(1,0)/ABSP_TMP(1)).gt.1.e-6_16) then
         write(*,*) 'px is not conserved (flag:CT968)',DBLE(P_TMP(1,0))
         stop 'px is not conserved (flag:CT968)'
        elseif ((P_TMP(2,0)/ABSP_TMP(2)).gt.1.e-6_16) then
         write(*,*) 'py is not conserved (flag:CT968)',DBLE(P_TMP(2,0))
         stop 'py is not conserved (flag:CT968)'
        elseif ((P_TMP(3,0)/ABSP_TMP(3)).gt.1.e-6_16) then
         write(*,*) 'pz is not conserved (flag:CT968)',DBLE(P_TMP(3,0))
         stop 'pz is not conserved (flag:CT968)'
        endif
      endif
      do I=0,3
        do J=1,(NLOOPLINE-1)
          do K=1,J
            P_TMP(I,J)=P_TMP(I,J)+PL(I,K)
          enddo
        enddo
      enddo
C     In Ninja, the loop line index starts at 1
      DO I=0,NLOOPLINE-1
        MP_P(0,I+1) = P_TMP(0,I)
        MP_P(1,I+1) = P_TMP(1,I)
        MP_P(2,I+1) = P_TMP(2,I)
        MP_P(3,I+1) = P_TMP(3,I)
      ENDDO

C     Number of coefficients for the current rank
      CURR_MAXCOEF = 0
      DO I=0,RANK
        CURR_MAXCOEF=CURR_MAXCOEF+(3+I)*(2+I)*(1+I)/6
      ENDDO
C     Now write the tensor coefficients for Ninja
C     It should never be allocated at this stage
      IF (.NOT. ALLOCATED(MP_TENSORCOEFS)) THEN
        ALLOCATE(MP_TENSORCOEFS(0:CURR_MAXCOEF-1))
      ENDIF
      IF (.NOT. ALLOCATED(MP_NINJA_TENSORCOEFS)) THEN
        ALLOCATE(MP_NINJA_TENSORCOEFS(0:CURR_MAXCOEF-1))
	  ENDIF
      DO I=0,CURR_MAXCOEF-1
## if(AmplitudeReduction) {
        MP_TENSORCOEFS(I) = MP_LOOPCOEFS(I,ID)
## } else {
        MP_TENSORCOEFS(I) = MP_LOOPCOEFS(I,SQSOINDEX,ID)
## }
      ENDDO
C     The loop momentum is in fact q_loop -> -q_loop, so that the
C     coefficients must be changed accordingly
      CALL MP_%(proc_prefix)sINVERT_MOMENTA_IN_POLYNOMIAL(CURR_MAXCOEF,MP_TENSORCOEFS)

C     Compute the kinematic matrix
      DO J=1,NLOOPLINE
        DO I=0,3
          P_S_MAT(J,I)=MP_P(I,J)
        ENDDO
      ENDDO
      CALL %(proc_prefix)sMP_BUILD_KINEMATIC_MATRIX(NLOOPLINE,P_S_MAT,MP_M2L,MP_S_MAT)

	  DO I=1,NLOOPLINE
	    DO J=1,NLOOPLINE
		  MP_REAL_S_MAT(I,J) = REAL(MP_S_MAT(I,J)+MP_M2L(i)+MP_M2L(j),KIND=16)
		ENDDO
	  ENDDO

C     Now typecast to Ninja's quadruple precision format
      DO I=0,CURR_MAXCOEF-1
	    MP_NINJA_TENSORCOEFS(I)=CMPLX(MP_TENSORCOEFS(I),KIND=KI_QNIN)
      ENDDO
	  DO I=1,NLOOPLINE
	    DO J=1,NLOOPLINE
		  MP_REAL_S_MAT_NINJA(I,J) = REAL(MP_REAL_S_MAT(I,J),KIND=KI_QNIN)
		ENDDO
	  ENDDO
      DO I=1,NLOOPLINE
	    MP_M2L_NINJA(I)=CMPLX(MP_M2L(I),KIND=KI_QNIN)
	  ENDDO
      DO I=1,NLOOPLINE
        MP_P_NINJA(0,I) = REAL(MP_P(0,I),KIND=KI_QNIN)
        MP_P_NINJA(1,I) = REAL(MP_P(1,I),KIND=KI_QNIN)
        MP_P_NINJA(2,I) = REAL(MP_P(2,I),KIND=KI_QNIN)
        MP_P_NINJA(3,I) = REAL(MP_P(3,I),KIND=KI_QNIN)
      ENDDO
	  NINJA_SCALE = REAL(MU_R**2,KIND=KI_QNIN)


C     Below is the call specifying the kinematic matrix
	  call ninja_tensor_evaluate(MP_NINJA_TENSORCOEFS,NLOOPLINE,RANK,MP_REAL_S_MAT_NINJA,MP_P_NINJA,MP_M2L_NINJA,NINJA_SCALE,NINJA_RES,NINJA_R1,NINJA_STATUS)
C     Below is the call without specification of the kinematic matrix
c      call ninja_tensor_evaluate(MP_NINJA_TENSORCOEFS,NLOOPLINE,RANK,MP_P_NINJA,MP_M2L_NINJA,NINJA_SCALE,NINJA_RES,NINJA_R1,NINJA_STATUS)

C     If a floating point exception was found in Ninja (e.g. exactly zero gram. det.)
C     Then warn loop_matrix.f so that it will flag this kinematic point as unstable no matter what.
      IF (NINJA_STATUS.EQ.NINJA_UNSTABLE_KINEMATICS) THEN
	    FPE_IN_QP_REDUCTION = .TRUE. 
	  ENDIF

C     Typecast the result back
	  R1 = DCMPLX(R1)
	  DO I=0,2
	    DP_RES(I)=DCMPLX(NINJA_RES(I))
	  ENDDO

C     Make sure to deallocate the tensor of coefficients
      IF (ALLOCATED(MP_TENSORCOEFS)) THEN
        DEALLOCATE(MP_TENSORCOEFS)
      ENDIF
      IF (ALLOCATED(MP_NINJA_TENSORCOEFS)) THEN
        DEALLOCATE(MP_NINJA_TENSORCOEFS)
      ENDIF

## if(AmplitudeReduction) {
      RES(1)=NORMALIZATION*DP_RES(0)
      RES(2)=NORMALIZATION*DP_RES(1)
      RES(3)=NORMALIZATION*DP_RES(2)
## } else {
      RES(1)=NORMALIZATION*2.0d0*DBLE(DP_RES(0))
      RES(2)=NORMALIZATION*2.0d0*DBLE(DP_RES(1))
      RES(3)=NORMALIZATION*2.0d0*DBLE(DP_RES(2))
## }
C     WRITE(*,*) 'QP Ninja:  Loop ID',ID,' =',RES(1),RES(2),RES(3)
      END
## } else {
C 
C The Ninja version installed does not support quadruple precision
C so that the corresponding subroutines are not output.
C
## }

	  SUBROUTINE %(proc_prefix)sINITNINJA()
C    
C     Module used
C
      use mninja 
C 
C     Initialization of Ninja 
C  
C LOCAL VARIABLES 
C
      INTEGER LOOPLIB
C  
C GLOBAL VARIABLES 
C
	  include 'MadLoopParams.inc'
C ----------
C BEGIN CODE
C ----------

C     LOOPLIB SET WHAT LIBRARY NINJA USES
C     1 -> LOOPTOOLS
C     2 -> AVH
C     3 -> QCDLOOP
      IF (CTLOOPLIBRARY.eq.1) THEN
          write(*,*) "Warning in Ninja initialization. LoopTools is not supported by the Ninja interface. It will use OneLOop instead."
          LOOPLIB = 1
      ELSEIF (CTLOOPLIBRARY.eq.3) THEN
          write(*,*) "Warning in Ninja initialization. LoopTools is not supported by the Ninja interface. It will use OneLOop instead."
          LOOPLIB = 1
      ELSEIF (CTLOOPLIBRARY.eq.2) THEN
          LOOPLIB = 1
      ELSE
          write(*,*) "Error in Ninja initialization. Loop library ID=",CTLOOPLIBRARY," is not supported. Change variable CTLoopLibrary in MadLoopParams.dat."
          stop 1
      ENDIF
      call ninja_set_integral_library(LOOPLIB)

      END
## }
