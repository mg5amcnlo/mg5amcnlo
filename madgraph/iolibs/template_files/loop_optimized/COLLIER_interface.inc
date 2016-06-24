	  SUBROUTINE %(proc_prefix)sCOLLIERLOOP(CTMODE, NLOOPLINE,RANK,PL,PDEN,M2L,TIRCOEFS,TIRCOEFSERRORS)
C  
%(info_lines)s
C 
C Interface between MG5 and COLLIER.
C It supports any rank when 1 < NLOOPLINE < 7.
C  
%(process_lines)s
C
C
C MODULES
C
	  USE COLLIER
	  IMPLICIT NONE
C  
C CONSTANTS 
C  
      LOGICAL checkPConservation
      PARAMETER (checkPConservation=.TRUE.)
      %(real_dp_format)s PIVALUE
      PARAMETER(PIVALUE=3.14159265358979323846D0)
	  integer MAXRANK
	  parameter (MAXRANK=%(maxrank)d)
      INTEGER NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=%(nloop_groups)d)
      include 'loop_max_coefs.inc'	  
C  
C ARGUMENTS 
C
      INTEGER NLOOPLINE, RANK, CTMODE
      %(real_dp_format)s PL(0:3,NLOOPLINE)
      %(real_dp_format)s PDEN(0:3,NLOOPLINE-1)	  
      %(mass_dp_format)s M2L(NLOOPLINE)
      %(complex_dp_format)s RES(3)
	  %(complex_dp_format)s TIRCOEFS(0:LOOPMAXCOEFS-1,3)
	  %(complex_dp_format)s TIRCOEFSERRORS(0:LOOPMAXCOEFS-1,3)

C  
C LOCAL VARIABLES 
C
	  INTEGER N, I, J, K, L
	  INTEGER C0,C1,C2,C3
	  INTEGER N_CACHES
	  INTEGER CURR_RANK, SGN
	  INTEGER CURR_INDEX
	  INTEGER CURR_MAXCOEF
	  %(real_dp_format)s RBUFF(0:3)

      INTEGER COEFMAP_ZERO(0:LOOPMAXCOEFS-1)
      INTEGER COEFMAP_ONE(0:LOOPMAXCOEFS-1)
      INTEGER COEFMAP_TWO(0:LOOPMAXCOEFS-1)
      INTEGER COEFMAP_THREE(0:LOOPMAXCOEFS-1)	  
	  %(collier_coefmap)s

	  %(real_dp_format)s REF_NORMALIZATION

	  double complex M2LCOLLIER(0:NLOOPLINE-1)
      double complex MomVec(0:3,NLOOPLINE-1)
	  double complex MomInv((NLOOPLINE*(NLOOPLINE-1))/2)
	 
      double complex TNten(0:RANK,0:RANK,0:RANK,0:RANK)
      double complex TNtenuv(0:RANK,0:RANK,0:RANK,0:RANK)
      double precision TNtenerr(0:RANK)

	  %(real_dp_format)s MaxCoefForRank(0:RANK)

C     These quantities are for the pole evaluation
      double complex TNten_UV(0:RANK,0:RANK,0:RANK,0:RANK)
      double complex TNtenuv_UV(0:RANK,0:RANK,0:RANK,0:RANK)
      double precision TNtenerr_UV(0:RANK)
	  double complex TNten_IR1(0:RANK,0:RANK,0:RANK,0:RANK)
      double complex TNtenuv_IR1(0:RANK,0:RANK,0:RANK,0:RANK)
      double precision TNtenerr_IR1(0:RANK)
      double complex TNten_IR2(0:RANK,0:RANK,0:RANK,0:RANK)
      double complex TNtenuv_IR2(0:RANK,0:RANK,0:RANK,0:RANK)
      double precision TNtenerr_IR2(0:RANK)
C
C GLOBAL VARIABLES
C
      include 'coupl.inc'
      include 'MadLoopParams.inc'
	  include 'unique_id.inc'
	  include 'global_specs.inc'

## if(not AmplitudeReduction) {
      INTEGER ID,SQSOINDEX,R
	  common/%(proc_prefix)sLOOP/ID,SQSOINDEX,R
## }else{
      INTEGER ID,R
	  common/%(proc_prefix)sLOOP/ID,R
## }

C
C     These global variables cache the coefficients already computed by COLLIER
C     so that they can be reused. In particular, in CTMODE 2, the cached ERROR 
C     quantities for eahc rank will be used to make use of COLLIER's internal
C     assessment of the numerical accuracy.
C
      logical LOOP_ID_DONE(NLOOPGROUPS)
	  DATA LOOP_ID_DONE/NLOOPGROUPS*.FALSE./
	  %(complex_dp_format)s TIR_COEFS_DIRECT_MODE(0:LOOPMAXCOEFS-1,3,NLOOPGROUPS)
	  %(real_dp_format)s TIR_COEFS_ERROR(0:MAXRANK,NLOOPGROUPS)
	  common/%(proc_prefix)sCOLLIER_TIR_COEFS/TIR_COEFS_DIRECT_MODE,TIR_COEFS_ERROR,LOOP_ID_DONE

      INTEGER COLLIER_CACHE_ACTIVE, NCALLS_IN_CACHE(4), NCOLLIERCALLS(4)
      LOGICAL MUST_INIT_EVENT
	  COMMON/%(proc_prefix)sCOLLIER_CACHE_STATUS/COLLIER_CACHE_ACTIVE,NCALLS_IN_CACHE, NCOLLIERCALLS,MUST_INIT_EVENT

      LOGICAL CTINIT, TIRINIT, GOLEMINIT, SAMURAIINIT, NINJAINIT, COLLIERINIT
      COMMON/REDUCTIONCODEINIT/CTINIT, TIRINIT,GOLEMINIT,SAMURAIINIT,NINJAINIT, COLLIERINIT

C ----------
C BEGIN CODE
C ----------

	  IF (COLLIERUseCacheForPoles) THEN
	    N_CACHES = 4
	  ELSE
	    N_CACHES =1
	  ENDIF

C     Initialize COLLIER if needed 
      IF (COLLIERINIT) THEN
	    COLLIERINIT=.FALSE.
	    CALL INITCOLLIER()
      ENDIF

C     Initialize the event if it is the first time collier is called on this PS point
      IF(MUST_INIT_EVENT) THEN
	    MUST_INIT_EVENT = .False.
        IF (COLLIERGlobalCache.eq.0) THEN
	      CALL InitEvent_cll()
	    ELSE
		  DO I=1,N_CACHES
C           Record how many events where put in the cache. On the first PS point.
		    IF (NCALLS_IN_CACHE(I).eq.-1.AND.NCOLLIERCALLS(I).gt.0) THEN
		      NCALLS_IN_CACHE(I) = NCOLLIERCALLS(I)
		    ENDIF
	      ENDDO
		  DO I=1,N_CACHES		  
C           Now apply a safety check that our last event had as many calls as the cache is setup for.
C           The only case for now when it can be half of the calls when we are doing the true loop-direction test with also the computation of a rotated PS point (which is computed for one mode only).
            IF (NCALLS_IN_CACHE(I).ne.-1.and..NOT. ( NCOLLIERCALLS(I).eq.NCALLS_IN_CACHE(I).or. ( CTModeRun.eq.-1.and..not.COLLIERUseInternalStabilityTest.and.NRotations_DP.gt.0.and.MOD(NCALLS_IN_CACHE(I),2).eq.0.and.(NCALLS_IN_CACHE(I)/2).eq.NCOLLIERCALLS(I) ) ) ) then
		      WRITE(*,*) 'WARNING: A consistency check in MadLoop failed and, for safety, forced MadLoop to reinitialize the global cache of COLLIER. Report this to MadLoop authors. The problematic cache was number ',I
              IF (COLLIERGLOBALCACHE.EQ.-1) THEN
                CALL INITCACHESYSTEM_CLL(N_CACHES*NPROCS,MAXNEXTERNAL)
              ELSE
		        CALL INITCACHESYSTEM_CLL(N_CACHES*NPROCS,COLLIERGLOBALCACHE)
              ENDIF
C             Make sure all caches are switched off at first.
              call SwitchOffCacheSystem_cll()
C             Reset the cache design property
		      NCALLS_IN_CACHE(:) = -1
			  NCOLLIERCALLS(:)   = 0
		      IF (COLLIER_CACHE_ACTIVE.eq.1) THEN			  
		        CALL SwitchOnCache_cll((unique_id-1)*N_CACHES+1)
			  ENDIF
C             No need to check the other caches since we already had to reset here.
			  EXIT
		    ENDIF
		  ENDDO
	      CALL InitEvent_cll((UNIQUE_ID-1)*N_CACHES+1)
		  NCOLLIERCALLS(1) = 0		  
		  IF(COLLIERComputeUVpoles.and.COLLIERUseCacheForPoles) THEN
	        CALL InitEvent_cll((UNIQUE_ID-1)*N_CACHES+2)
			NCOLLIERCALLS(2) = 0
		  ENDIF
		  IF(COLLIERComputeIRpoles.and.COLLIERUseCacheForPoles) THEN
	        CALL InitEvent_cll((UNIQUE_ID-1)*N_CACHES+3)
			NCOLLIERCALLS(3) = 0
	        CALL InitEvent_cll((UNIQUE_ID-1)*N_CACHES+4)
			NCOLLIERCALLS(4) = 0
		  ENDIF
        ENDIF
        CALL SetDeltaIR_cll(0.0d0,(PIVALUE**2)/6.0d0)
        CALL SetDeltaUV_cll(0.0d0)
        CALL SetMuIR2_cll(mu_r**2)      
        CALL SetMuUV2_cll(mu_r**2)
	  ENDIF

C     Now really start the reduction with COLLIER

C     Number of coefficients for the current rank
      CURR_MAXCOEF = 0
      DO I=0,RANK
        CURR_MAXCOEF=CURR_MAXCOEF+(3+I)*(2+I)*(1+I)/6
      ENDDO

	  IF (CTMODE.ne.1.and.CTMODE.ne.2) THEN
	    WRITE(*,*) 'ERROR: COLLIER only support the computational mode CTMODE equal to 1 or 2, not',CTMODE
	    stop 'Incorrect computational mode specified to the COLLIER MG5aMC interface.'
	  ENDIF

      DO I=0,CURR_MAXCOEF-1
        DO K=1,3			
          TIRCOEFSERRORS(I,K)=DCMPLX(0.0d0,0.0d0)
	    ENDDO
      ENDDO

	  IF (COLLIERUseInternalStabilityTest) THEN
C       Use MADLOOP internal cache dedicated to COLLIER that emulates the CTMODE 2
	    IF (LOOP_ID_DONE(ID)) THEN
		  DO I=0,CURR_MAXCOEF-1
		    DO K=1,3
			  TIRCOEFS(I,K)=TIR_COEFS_DIRECT_MODE(I,K,ID)
			ENDDO
	      ENDDO
		  DO I=0,RANK
			MaxCoefForRank(I) = 0.0d0
	      ENDDO
		  DO I=0,CURR_MAXCOEF-1
		    CURR_RANK = COEFMAP_ZERO(I)+COEFMAP_ONE(I)+COEFMAP_TWO(I)+COEFMAP_THREE(I)			  
			MaxCoefForRank(CURR_RANK)=max(MaxCoefForRank(CURR_RANK),abs(TIR_COEFS_DIRECT_MODE(I,1,ID)))
		  ENDDO
		  DO I=0,CURR_MAXCOEF-1
		    CURR_RANK = COEFMAP_ZERO(I)+COEFMAP_ONE(I)+COEFMAP_TWO(I)+COEFMAP_THREE(I)
            IF (MaxCoefForRank(CURR_RANK).ne.0.0d0) THEN
              DO K=1,3
C               The expression below is like taking the absolute value when summing errors linearly
C                TIRCOEFSERRORS(I,K)=(TIR_COEFS_ERROR(CURR_RANK,ID)/MaxCoefForRank(CURR_RANK))*DCMPLX( ABS(DBLE(TIRCOEFS(I,K))),ABS(DIMAG(TIRCOEFS(I,K))) )
C               But empirically, I observed that retaining the original complex phase leads to slightly more accurate estimates
			    TIRCOEFSERRORS(I,K)=(TIR_COEFS_ERROR(CURR_RANK,ID)/MaxCoefForRank(CURR_RANK))*TIRCOEFS(I,K)           
			  ENDDO
            ENDIF	
	      ENDDO    
		  RETURN
		ENDIF	    
      ELSE
C       Apply the loop-direction switching here.
        CALL %(proc_prefix)sSWITCH_ORDER(CTMODE,NLOOPLINE,PL,PDEN,M2L)	  
	  ENDIF

C     Make sure masses are complex 
      do I=1,NLOOPLINE
	    M2LCOLLIER(I-1)=DCMPLX(M2L(I))
	  ENDDO

C     Now convert the loop offset momenta to COLLIER conventions
      DO i=0,3
	    DO j=1,NLOOPLINE-1
	      MomVec(i,j)=DCMPLX(PDEN(i,j),0.0d0)
	    ENDDO
      ENDDO

C     This first do loop spans over 'N' in '\foreach_N \foreach_i(k_i+k_{i+1}+..+k_{i+N})^2'
      CURR_INDEX = 0
      DO N=0,NLOOPLINE-1
C       We stop whenever we reached the target number of invariants
        IF (CURR_INDEX.GE.((NLOOPLINE*(NLOOPLINE-1))/2)) THEN
		  EXIT
		ENDIF
C       This do loop spans over 'i' in the expression above.
        DO I=1,NLOOPLINE
          IF (CURR_INDEX.GE.((NLOOPLINE*(NLOOPLINE-1))/2)) THEN
		    EXIT
		  ENDIF

          CURR_INDEX = CURR_INDEX+1
          RBUFF(:) = 0.0D0
          DO J=I,I+N
            RBUFF(:) = RBUFF(:) + PL(:,MOD(J-1,NLOOPLINE)+1)
          ENDDO
          MomInv(CURR_INDEX) = DCMPLX(RBUFF(0)**2-RBUFF(1)**2-RBUFF(2)**2-RBUFF(3)**2,0.0d0)

C         Now regularize the onshell behavior of the kinematic invarients
C         All loop masses are tested, but that might be a bit too inclusive.
	      DO K=1,NLOOPLINE
		    IF(ABS(M2L(K)).NE.0.0d0) THEN
              IF(ABS((MomInv(CURR_INDEX)-M2L(K))/M2L(K)).LT.OSTHRES) THEN
                MomInv(CURR_INDEX)=DCMPLX(M2L(K))
              ENDIF
		    ENDIF
		  ENDDO
C         For the massless onshell-case, we base the threshold only on the energy component
          REF_NORMALIZATION=0.0D0
          DO L=0,0
            REF_NORMALIZATION = REF_NORMALIZATION + ABS(RBUFF(L))
          ENDDO
          REF_NORMALIZATION = (REF_NORMALIZATION/(N+1))**2
          IF(REF_NORMALIZATION.NE.0.0D0)THEN
            IF(ABS(MomInv(CURR_INDEX)/REF_NORMALIZATION).LT.OSTHRES)THEN
              MomInv(CURR_INDEX)=DCMPLX(0.0d0,0.0d0)
            ENDIF
          ENDIF

		ENDDO
	  ENDDO

C     We can now call COLLIER
	  IF (NLOOPLINE.ne.1) THEN
        CALL TNten_cll(TNten, TNtenuv, MomVec, MomInv, M2LCOLLIER, NLOOPLINE, RANK, TNtenerr)
      ELSE
        CALL TNten_cll(TNten, TNtenuv, M2LCOLLIER, NLOOPLINE, RANK, TNtenerr)
	  ENDIF
	  IF (COLLIER_CACHE_ACTIVE.eq.1) THEN
	    NCOLLIERCALLS(1) = NCOLLIERCALLS(1)+1
	  ENDIF

C     Now compute the UV poles if asked for
      IF (COLLIERComputeUVpoles) THEN
	    IF(COLLIER_CACHE_ACTIVE.eq.1) THEN
		  CALL SwitchOffCache_cll((unique_id-1)*N_CACHES+1)
		  IF(COLLIERUseCacheForPoles) THEN
		    CALL SwitchOnCache_cll((unique_id-1)*N_CACHES+2)
		  ENDIF
        ENDIF
        CALL SetDeltaUV_cll(1.0d0)
	    IF (NLOOPLINE.ne.1) THEN
          CALL TNten_cll(TNten_UV, TNtenuv_UV, MomVec, MomInv, M2LCOLLIER, NLOOPLINE, RANK, TNtenerr_UV)
        ELSE
          CALL TNten_cll(TNten_UV, TNtenuv_UV, M2LCOLLIER, NLOOPLINE, RANK, TNtenerr_UV)
	    ENDIF
	    IF (COLLIER_CACHE_ACTIVE.eq.1) THEN
	      NCOLLIERCALLS(2) = NCOLLIERCALLS(2)+1
	    ENDIF
	    IF(COLLIER_CACHE_ACTIVE.eq.1) THEN
		  IF(COLLIERUseCacheForPoles) THEN
		    CALL SwitchOffCache_cll((unique_id-1)*N_CACHES+2)
		  ENDIF
		  CALL SwitchOnCache_cll((unique_id-1)*N_CACHES+1)
        ENDIF
        CALL SetDeltaUV_cll(0.0d0)
	  ENDIF

C     Now compute the IR poles if asked for
      IF (COLLIERComputeIRpoles) THEN
	    IF(COLLIER_CACHE_ACTIVE.eq.1) THEN
		  CALL SwitchOffCache_cll((unique_id-1)*N_CACHES+1)
		  IF(COLLIERUseCacheForPoles) THEN
		    CALL SwitchOnCache_cll((unique_id-1)*N_CACHES+3)
		  ENDIF
        ENDIF
        CALL SetDeltaIR_cll(1.0d0,(PIVALUE**2)/6.0d0)
	    IF (NLOOPLINE.ne.1) THEN
          CALL TNten_cll(TNten_IR1, TNtenuv_IR1, MomVec, MomInv, M2LCOLLIER, NLOOPLINE, RANK, TNtenerr_IR1)
        ELSE
          CALL TNten_cll(TNten_IR1, TNtenuv_IR1, M2LCOLLIER, NLOOPLINE, RANK, TNtenerr_IR1)
	    ENDIF
	    IF (COLLIER_CACHE_ACTIVE.eq.1) THEN
	      NCOLLIERCALLS(3) = NCOLLIERCALLS(3)+1
	    ENDIF
        IF(COLLIER_CACHE_ACTIVE.eq.1.AND.COLLIERUseCacheForPoles) THEN
		  CALL SwitchOffCache_cll((unique_id-1)*N_CACHES+3)
		  CALL SwitchOnCache_cll((unique_id-1)*N_CACHES+4)			
        ENDIF
        CALL SetDeltaIR_cll(0.0d0,1.0d0+(PIVALUE**2)/6.0d0)
	    IF (NLOOPLINE.ne.1) THEN
          CALL TNten_cll(TNten_IR2, TNtenuv_IR2, MomVec, MomInv, M2LCOLLIER, NLOOPLINE, RANK, TNtenerr_IR2)
        ELSE
          CALL TNten_cll(TNten_IR2, TNtenuv_IR2, M2LCOLLIER, NLOOPLINE, RANK, TNtenerr_IR2)
	    ENDIF
	    IF (COLLIER_CACHE_ACTIVE.eq.1) THEN
	      NCOLLIERCALLS(4) = NCOLLIERCALLS(4)+1
	    ENDIF
	    IF(COLLIER_CACHE_ACTIVE.eq.1) THEN
		  IF(COLLIERUseCacheForPoles) THEN
		    CALL SwitchOffCache_cll((unique_id-1)*N_CACHES+4)
		  ENDIF
		  CALL SwitchOnCache_cll((unique_id-1)*N_CACHES+1)		  
        ENDIF
		CALL SetDeltaIR_cll(0.0d0,(PIVALUE**2)/6.0d0)
	  ENDIF

	  DO I=0,(CURR_MAXCOEF-1)
	    C0 = COEFMAP_ZERO(I)
		C1 = COEFMAP_ONE(I)
		C2 = COEFMAP_TWO(I)
		C3 = COEFMAP_THREE(I)
	    CURR_RANK = C0+C1+C2+C3
C       Because we must set q -> -q, we apply a minus sign to coefs of odd rank
	    IF (MOD(CURR_RANK,2).eq.1) THEN
		  SGN = -1
		ELSE
		  SGN = 1
		ENDIF
		TIRCOEFS(I,1) = SGN*TNten(C0,C1,C2,C3)
        IF (COLLIERComputeUVpoles) THEN
		  TIRCOEFS(I,2) = SGN*( TNten_UV(C0,C1,C2,C3)-TNten(C0,C1,C2,C3) )
		ELSE
		  TIRCOEFS(I,2) = DCMPLX(0.0d0,0.0d0)
		ENDIF
        IF (COLLIERComputeIRpoles) THEN
		  TIRCOEFS(I,2) = TIRCOEFS(I,2) + SGN*( TNten_IR1(C0,C1,C2,C3)-TNten(C0,C1,C2,C3) )
		  TIRCOEFS(I,3) = SGN*( TNten_IR2(C0,C1,C2,C3)-TNten(C0,C1,C2,C3) )
		ELSE
		  TIRCOEFS(I,3) = DCMPLX(0.0d0,0.0d0)		
		ENDIF
      ENDDO

	  IF (COLLIERUseInternalStabilityTest) THEN
c       Finish by caching the coefficients and error just computed
	    LOOP_ID_DONE(ID) = .TRUE.
	    DO J=0,CURR_MAXCOEF-1
	      DO K=1,3
	        TIR_COEFS_DIRECT_MODE(J,K,ID) = TIRCOEFS(J,K)
	      ENDDO
	    ENDDO
	    DO J=0,RANK
	      TIR_COEFS_ERROR(J,ID)=TNtenerr(J)
	    ENDDO

C       Now compute the errors on each coefficient
        DO I=0,RANK
          MaxCoefForRank(I) = 0.0d0
        ENDDO
        DO I=0,CURR_MAXCOEF-1
          CURR_RANK = COEFMAP_ZERO(I)+COEFMAP_ONE(I)+COEFMAP_TWO(I)+COEFMAP_THREE(I)			  
          MaxCoefForRank(CURR_RANK)=max(MaxCoefForRank(CURR_RANK),abs(TIRCOEFS(I,1)))
        ENDDO
        DO I=0,CURR_MAXCOEF-1
          CURR_RANK = COEFMAP_ZERO(I)+COEFMAP_ONE(I)+COEFMAP_TWO(I)+COEFMAP_THREE(I)
          IF (MaxCoefForRank(CURR_RANK).ne.0.0d0) THEN
            DO K=1,3
C               The expression below is like taking the absolute value when summing errors linearly
C			  TIRCOEFSERRORS(I,K)=(TNtenerr(CURR_RANK)/MaxCoefForRank(CURR_RANK))*DCMPLX( ABS(DBLE(TIRCOEFS(I,K))),ABS(DIMAG(TIRCOEFS(I,K))) )			
C               But empirically, I observed that retaining the original complex phase leads to slightly more accurate estimates
			  TIRCOEFSERRORS(I,K)=(TNtenerr(CURR_RANK)/MaxCoefForRank(CURR_RANK))*TIRCOEFS(I,K)
			ENDDO
          ENDIF
        ENDDO
	    
	  ENDIF

	  END SUBROUTINE %(proc_prefix)sCOLLIERLOOP 

      SUBROUTINE %(proc_prefix)sCLEAR_COLLIER_CACHE()
	    
	    USE COLLIER

        include 'loop_max_coefs.inc'
        INTEGER NLOOPGROUPS
        PARAMETER (NLOOPGROUPS=%(nloop_groups)d)
	    integer MAXRANK
	    parameter (MAXRANK=%(maxrank)d)
		
		integer I,J,K

        include 'MadLoopParams.inc'

        logical LOOP_ID_DONE(NLOOPGROUPS)
	    %(complex_dp_format)s TIR_COEFS_DIRECT_MODE(0:LOOPMAXCOEFS-1,3,NLOOPGROUPS)
	    %(complex_dp_format)s TIR_COEFS_ERROR(0:MAXRANK,NLOOPGROUPS)
	    common/%(proc_prefix)sCOLLIER_TIR_COEFS/TIR_COEFS_DIRECT_MODE,TIR_COEFS_ERROR,LOOP_ID_DONE

        INTEGER COLLIER_CACHE_ACTIVE, NCALLS_IN_CACHE(4), NCOLLIERCALLS(4)
        LOGICAL MUST_INIT_EVENT
	    COMMON/%(proc_prefix)sCOLLIER_CACHE_STATUS/COLLIER_CACHE_ACTIVE,NCALLS_IN_CACHE, NCOLLIERCALLS,MUST_INIT_EVENT

C       Make sure that next time the COLLIER Subroutine is called it will call the subroutine initEvent of Collier.
        MUST_INIT_EVENT = .True.

C       Reinitialize the ML cache for COLLIER
        DO I=1,NLOOPGROUPS
		  LOOP_ID_DONE(I) = .FALSE.
		  DO J=0,LOOPMAXCOEFS-1
		    DO K=1,3
			  TIR_COEFS_DIRECT_MODE(J,K,I) = DCMPLX(0.0d0,0.0d0)
			ENDDO
		  ENDDO
		  DO J=0,MAXRANK
		    TIR_COEFS_ERROR(J,I)=0.0d0
	      ENDDO
		ENDDO

	  END

	  SUBROUTINE %(proc_prefix)sSET_COLLIER_GLOBAL_CACHE(ONOFF)
C
C       This routine is used by loop_matrix.f to turn on the global
C       cache of COLLIER when it the main SLOOP subroutine starts and
C       turn it off when it ends.
C       However several checks are performed to verify that it is safe
C       to turn it on and to reinitialize it if necessary.
C
C       MODULES
C
        USE COLLIER
		implicit none
C
C       ARGUMENTS
C
        LOGICAL ONOFF
C
C       LOCAL VARIABLES
C
        LOGICAL NEED_REINITIALIZATION
		INTEGER N_CACHES
C		
C       GLOBAL VARIABLES
C
C
C       This common blocks saves the relevant ML parameters when activating the
C       global cache of COLLIER so that we know when we must reinitialize it.
C       COLLIER_CACHE_ACTIVE is set to -1 when it has never been turned on yet and
C       to 1 for 'Active' and 0 for 'Inactive'.
C       The integer NCALLS_IN_CACHE saves how many calls the cache is setup for, for each of the up to four caches.
C       When it is the first PS points, it is set to -1.
        INTEGER COLLIER_CACHE_ACTIVE, NCALLS_IN_CACHE(4), NCOLLIERCALLS(4)
	    DATA COLLIER_CACHE_ACTIVE/-1/
		DATA NCALLS_IN_CACHE/-1,-1,-1,-1/
		DATA NCOLLIERCALLS/0,0,0,0/
C       This is a flag to tell the COLLIER subroutine that it must init the event when called.
		LOGICAL MUST_INIT_EVENT
		DATA MUST_INIT_EVENT/.TRUE./
		COMMON/%(proc_prefix)sCOLLIER_CACHE_STATUS/COLLIER_CACHE_ACTIVE, NCALLS_IN_CACHE, NCOLLIERCALLS,MUST_INIT_EVENT

		LOGICAL COLLIERUseInternalStabilityTest_BU
		INTEGER USERHEL_BU, SQSO_TARGET_BU, COLLIERMode_BU,CTModeRun_BU
        COMMON/%(proc_prefix)sCOLLIER_CACHE_RELEVANT_PARAMS/USERHEL_BU,SQSO_TARGET_BU,COLLIERMode_BU,CTModeRun_BU,COLLIERUseInternalStabilityTest_BU

C       The common blocks below are to retrieve the necessary information about
C       MadLoop running mode and store it in the sCOLLIER_CACHE_RELEVANT_PARAMS common block.

        include 'MadLoopParams.inc'
		include 'unique_id.inc'
		include 'global_specs.inc'

        LOGICAL CTINIT, TIRINIT, GOLEMINIT, SAMURAIINIT, NINJAINIT, COLLIERINIT
        COMMON/REDUCTIONCODEINIT/CTINIT, TIRINIT,GOLEMINIT,SAMURAIINIT,NINJAINIT, COLLIERINIT

	    LOGICAL CHECKPHASE
	    LOGICAL HELDOUBLECHECKED
        common/%(proc_prefix)sINIT/CHECKPHASE, HELDOUBLECHECKED

	    INTEGER USERHEL
	    common/%(proc_prefix)sUSERCHOICE/USERHEL

	    INTEGER SQSO_TARGET
	    common/%(proc_prefix)sSOCHOICE/SQSO_TARGET
		
C
C       BEGIN CODE
C

		IF (COLLIERUseCacheForPoles) THEN
		  N_CACHES = 4
		ELSE
		  N_CACHES =1
		ENDIF

C       Do nothing if COLLIER still has to be initialized or if global caches are disabled
        IF(COLLIERINIT.OR.COLLIERGlobalCache.eq.0) THEN
		  RETURN
		ENDIF

C       Never activate anything in the checkphase
        IF (ONOFF.AND.CHECKPHASE) THEN
		  RETURN
		ENDIF

C       Handle the request of turning off the caching
		IF (.NOT.ONOFF) THEN
		  IF (COLLIER_CACHE_ACTIVE.eq.1) THEN
		    CALL SwitchOffCache_cll((unique_id-1)*4+1)
		    COLLIER_CACHE_ACTIVE = 0
		  ENDIF
C         If we were asked to turn the cache off but it was already so, then do nothing		  
		  RETURN
		ENDIF

C       Handle the request of turning on the caching

C       If asked to activate it but already active, then do nothing
		IF (ONOFF.AND.COLLIER_CACHE_ACTIVE.eq.1) THEN
		  RETURN
        ENDIF

C       We are now in the position where we are asked to activate the global cache but it was *not* already active.

C       If we activate it for the first time, make sure to store the value of the relevant parameters, activate and return.
        IF (COLLIER_CACHE_ACTIVE.eq.-1) THEN
		  USERHEL_BU         = USERHEL 
		  SQSO_TARGET_BU     = SQSO_TARGET
		  COLLIERMode_BU     = COLLIERMode
		  COLLIERUseInternalStabilityTest_BU = COLLIERUseInternalStabilityTest
		  CTModeRun_BU       = CTModeRun
		  CALL SwitchOnCache_cll((unique_id-1)*4+1)
		  COLLIER_CACHE_ACTIVE = 1
		  RETURN
		ENDIF

C       Now perform sanity check before the activation to decide if we need to reinitialize the cache system first.
		NEED_REINITIALIZATION = .FALSE.

        IF (SQSO_TARGET.NE.SQSO_TARGET_BU) THEN
		  NEED_REINITIALIZATION = .TRUE.		
		ENDIF

        IF (COLLIERMode.NE.COLLIERMode_BU) THEN
		  NEED_REINITIALIZATION = .TRUE.		
		ENDIF

        IF (COLLIERUseInternalStabilityTest.NEQV.COLLIERUseInternalStabilityTest_BU) THEN
		  NEED_REINITIALIZATION = .TRUE.		
		ENDIF

		IF (CTModeRun_BU.ne.CTModeRun.and.(.not.COLLIERUseInternalStabilityTest)) THEN
		  NEED_REINITIALIZATION = .TRUE.			  
		ENDIF

## if(AmplitudeReduction){
C       When doing amplitude reduction, the parameter USERHEL impacts the number of COLLIER calls/order if
C       it involves a different number of helicity configurations or always if the Loop filter is on, which
C       it should almost never be by now.
        IF(UseLoopFilter.and.(USERHEL.ne.USERHEL_BU)) THEN
		  NEED_REINITIALIZATION = .TRUE.
		ENDIF
		IF((USERHEL.eq.-1).and.(USERHEL_BU.ne.-1)) THEN
		  NEED_REINITIALIZATION = .TRUE.
		ENDIF
		IF((USERHEL.ne.-1).and.(USERHEL_BU.eq.-1)) THEN
		  NEED_REINITIALIZATION = .TRUE.
		ENDIF
## }else{
C       When doing amplitude reduction the parameter USERHEL does not impact the number/order of COLLIER calls
C       except if the LoopFilter is ON which really shouldn't be the case anymore.
        IF(UseLoopFilter.and.(USERHEL.ne.USERHEL_BU)) THEN
		  NEED_REINITIALIZATION = .TRUE.
		ENDIF
## }

		IF(NEED_REINITIALIZATION) THEN
C         Log the event because if it happens a lot of time and floods the screen, the user must see it
C         and either change its usage of MadLoop or turnoff COLLIER cache
          WRITE(*,*) 'INFO: MadLoop detected that the global cache of COLLIER had to be reset because of a change in your use of MadLoop. This should not happend for each event.'
		  USERHEL_BU         = USERHEL 
		  SQSO_TARGET_BU     = SQSO_TARGET
		  COLLIERMode_BU     = COLLIERMode
		  COLLIERUseInternalStabilityTest_BU = COLLIERUseInternalStabilityTest
		  CTModeRun_BU       = CTModeRun		  
		  IF (COLLIERGLOBALCACHE.EQ.-1) THEN
            CALL INITCACHESYSTEM_CLL(N_CACHES*NPROCS,MAXNEXTERNAL)
          ELSE
		    CALL INITCACHESYSTEM_CLL(N_CACHES*NPROCS,COLLIERGLOBALCACHE)
          ENDIF
		  NCOLLIERCALLS(:)   = 0
		  NCALLS_IN_CACHE(:) = -1 
C         Make sure all caches are switched off at first.
          call SwitchOffCacheSystem_cll()
		ENDIF

C       Now we can finally activate the cache
		CALL SwitchOnCache_cll((unique_id-1)*4+1)
		COLLIER_CACHE_ACTIVE = 1

	  END
