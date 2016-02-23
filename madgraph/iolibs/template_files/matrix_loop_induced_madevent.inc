SUBROUTINE SMATRIX%(proc_id)s(P,ANS)
C  
%(info_lines)s
C 
C MadGraph5_aMC@NLO for Madevent Version
C 
C Returns amplitude squared summed/avg over colors
c and helicities
c for the point in phase space P(0:3,NEXTERNAL)
C  
%(process_lines)s
C 
    use DiscreteSampler
    IMPLICIT NONE
C  
C CONSTANTS
C  
    Include 'genps.inc'
    Include 'maxconfigs.inc'
    Include 'nexternal.inc'
    Include 'maxamps.inc'
    INTEGER                 NCOMB         
    PARAMETER (             NCOMB=%(ncomb)d)
C   Below is the maximum config ID for multichannels which is always less than
C   the number of loop diagrams, so we use the latter.
    INTEGER    NLOOPDIAGRAMS
    PARAMETER (NLOOPDIAGRAMS=%(nloopdiags)d)  
C   Below is the number of multi-channel actually contributing to this process
	INTEGER    NMULTICHANNELS
    PARAMETER (NMULTICHANNELS=%(nmultichannels)d)
    INTEGER    NLOOPFLOWS
    PARAMETER (NLoopFlows=%(nLoopFlows)d) 
    INTEGER    THEL
    PARAMETER (THEL=NCOMB)
    INTEGER Hel_Average_Factor
    PARAMETER (Hel_Average_Factor=%(hel_avg_factor)d)
C   It can be that all '|A_i|^2' are zero, in which case one wants to effectively set multichannel factors to be flat.
C   This behavior will occur when  Sum_i(|A_i|^2)/Full_ME < MULTICHANNEL_THRES. 
C   When set negative, the security above is removed
	DOUBLE PRECISION MULTICHANNEL_THRES
	PARAMETER (MULTICHANNEL_THRES=1.0d-5)
C  
C ARGUMENTS 
C  
    REAL*8 P(0:3,NEXTERNAL),ANS
c
c global (due to reading writting) 
c
        LOGICAL GOODHEL(NCOMB)
	INTEGER NTRY
        common/BLOCK_GOODHEL/NTRY,GOODHEL
C  
C LOCAL VARIABLES 
C  
    INTEGER NHEL(NEXTERNAL,NCOMB)
    REAL*8 T,MATRIX%(proc_id)s
    REAL*8 R,SUMHEL,TS(NCOMB)
    INTEGER I,IDEN
    INTEGER IPROC,JC(NEXTERNAL),II
    REAL*8 HWGT, XTOT, XTRY, XREJ, XR, YFRAC(0:NCOMB)
    INTEGER IDUM, NGOOD, IGOOD(NCOMB), JHEL, J, JJ
    REAL     XRAN1
    EXTERNAL XRAN1

C   This stores the different between the full matrix element squared and the sum of multi-channeling factors '|A_i|^2'.
	DOUBLE PRECISION INTERFERENCE_CONTRIB
	
	real t_before

	REAL*8 , ALLOCATABLE :: ANS_ML(:,:)
    REAL*8 , ALLOCATABLE :: PREC_FOUND(:)
	INTEGER ML_RET_CODE
    INTEGER ML_ANS_DIMENSION
C Prec_found ans ans_ml are saved so that they don't have to be re-allocated everytime
	SAVE ML_ANS_DIMENSION,ANS_ML,PREC_FOUND

    DATA IDUM /-1/
    DATA XTRY, XREJ, NGOOD /0,0,0/

	LOGICAL INIT_MADLOOP
	DATA INIT_MADLOOP/.FALSE./
C  
C GLOBAL VARIABLES
C  
    LOGICAL FORCE_ML_HELICITY_SUM 
    common/to_ML_control/FORCE_ML_HELICITY_SUM 

    DOUBLE PRECISION AMP2(MAXAMPS), JAMP2(0:MAXFLOW)
    COMMON/TO_AMPS/  AMP2,       JAMP2
   
    CHARACTER*101        HEL_BUFF
    COMMON/TO_HELICITY/  HEL_BUFF

    %(real_dp_format)s ML_JAMP2(nLoopFlows)
	common/%(ml_prefix)sJAMP2/ML_JAMP2

    %(real_dp_format)s ML_AMP2(NLOOPDIAGRAMS)
	common/%(ml_prefix)sAMP2/ML_AMP2
    
    REAL*8 POL(2)
    COMMON/TO_POLARIZATION/ POL

    logical read_grid_file
    common/read_grid_file/read_grid_file

    INTEGER          ISUM_HEL
    LOGICAL                    MULTI_CHANNEL
    COMMON/TO_MATRIX/ISUM_HEL, MULTI_CHANNEL
%(define_iconfigs_lines)s
    DATA XTRY, XREJ, NGOOD /0,0,0/
    SAVE YFRAC, IGOOD, JHEL

%(helicity_lines)s

C ----------
C BEGIN CODE
C ----------

C  Notice that when forcing helicity sum directly in MadLoop then one doesn't have access to the weights of individual Hel. Configs so that they cannot be specified in the event file. We turned this option off by default.
    FORCE_ML_Helicity_Sum = (ISUM_HEL.eq.0.AND..False.)

	IF(NLOOPDIAGRAMS.GT.MAXAMPS) THEN
	   STOP 'MAXAMPS smaller than to NLOOPDIAGRAMS in matrix.f'
	ENDIF

	IF(.NOT.INIT_MADLOOP) THEN
	  INIT_MADLOOP = .TRUE.
C Unless this is the first iteration, make sure to never double check the helicity filter again
      IF (read_grid_file) THEN
        CALL SET_FORBID_HEL_DOUBLECHECK(.True.)
	  ELSE
C       The double check is turned off here even for the first iteration because the further iteration would anyway be wrong since the new HelFilter is not output. We keep it here as it can be used for diagnostics, for example by triggering a crash if the double-check fails instead of trying to recompute the filter.
        CALL SET_FORBID_HEL_DOUBLECHECK(.True.)
	  ENDIF
C So that TIR integrals can be reused accross helicity configuration.
C Notice that ITR integrals for rotated PS points (i.e. which is part of MadLoop's default stabiliy test procedure), will not work. This is why one should always keep the MadLoop parameter 'NRotations_DP' set to 0 in this case.
C Of course this is only necessary when using MadLoop for several helicity for the same PS point
      IF ((.not.FORCE_ML_Helicity_Sum).and.ISUM_HEL.eq.0) THEN
	    CALL %(ml_prefix)sSET_AUTOMATIC_CACHE_CLEARING(.FALSE.)
	  ENDIF
      CALL %(ml_prefix)sGET_ANSWER_DIMENSION(ML_ANS_DIMENSION)
C In case there is an initialization phase for MadLoop (i.e. everytime except when MC over helicity and disabling the Loop filter, which is the default.
	  CALL %(ml_prefix)sFORCE_STABILITY_CHECK(.TRUE.)
      ALLOCATE(ANS_ML(0:3,0:ML_ANS_DIMENSION))
	  ALLOCATE(PREC_FOUND(0:ML_ANS_DIMENSION))
	ENDIF

    IF ((.not.FORCE_ML_Helicity_Sum).and.ISUM_HEL.eq.0) THEN
C     But then we must clear the cache by hand at the beginning of the computation of each new PS point
C     This is of course only necessary if the automatic TIR Cache clearing was turned off.
	  CALL %(ml_prefix)sCLEAR_CACHES()
    ENDIF

	NTRY=NTRY+1
    DO I=1,NEXTERNAL
       JC(I) = +1
    ENDDO
     
    IF (multi_channel) THEN
        DO I=1,NLOOPDIAGRAMS
            AMP2(I)=0D0
        ENDDO
        JAMP2(0)=%(ncolor)d
        DO I=1,INT(JAMP2(0))
            JAMP2(I)=0D0
        ENDDO
    ENDIF
    ANS = 0D0
    WRITE(HEL_BUFF,'(20I5)') (0,I=1,NEXTERNAL)

    DO I=1,NCOMB
       TS(I)=0d0
    ENDDO

    if (FORCE_ML_Helicity_Sum)then
C Of course this option can also not be used for polarized beams.
      do JJ=1,NINCOMING
	    if (POL(JJ).NE.1d0) FORCE_ML_Helicity_Sum = .false.
	  enddo	       
    endif

    IF(Force_ML_Helicity_Sum )THEN

	    call cpu_time(t_before)
        CALL %(ml_prefix)sSLOOPMATRIX_THRES(P,ANS_ML,-1.0d0,PREC_FOUND,ML_RET_CODE)
		CALL PROCESS_MADLOOP_OUTPUT%(proc_id)s(P,t_before,ANS,ANS_ML(1,0),ANS_ML(2,0),ANS_ML(3,0),PREC_FOUND(0),ML_RET_CODE)
    
	ELSE
    IF (ISUM_HEL.EQ.0.or.(DS_get_dim_status('Helicity').eq.0)) THEN
           DO I=1,NCOMB
              IF (GOODHEL(I) .OR. NTRY .LE. MAXTRIES.OR.(ISUM_HEL.NE.0)) THEN
	            
			    call cpu_time(t_before)
				CALL %(ml_prefix)sSLOOPMATRIXHEL_THRES(P,I,ANS_ML,-1.0d0,PREC_FOUND,ML_RET_CODE)
		        CALL PROCESS_MADLOOP_OUTPUT%(proc_id)s(P,t_before,T,ANS_ML(1,0),ANS_ML(2,0),ANS_ML(3,0),PREC_FOUND(0),ML_RET_CODE)
                
				DO JJ=1,JAMP2(0)
                  JAMP2(JJ) = JAMP2(JJ) + ML_JAMP2(JJ)
                ENDDO
                DO JJ=1,NLOOPDIAGRAMS
                  AMP2(JJ) = AMP2(JJ)+ ML_AMP2(JJ)
                ENDDO

				DO JJ=1,nincoming
               	     IF(POL(JJ).NE.1d0.AND.NHEL(JJ,I).EQ.INT(SIGN(1d0,POL(JJ)))) THEN
                         T=T*ABS(POL(JJ))
                     ELSE IF(POL(JJ).NE.1d0)THEN
                         T=T*(2d0-ABS(POL(JJ)))
                     ENDIF
                  ENDDO
			      IF (ISUM_HEL.NE.0) then
			        call DS_add_entry('Helicity',I,T)
			      endif
                  ANS=ANS+DABS(T)
                  TS(I)=T
              ENDIF
           ENDDO
		IF (ISUM_HEL.NE.0) then
!         We set HEL_PICKED to -1 here so that later on, the call to DS_add_point in dsample.f does not add anything to the grid since it was already done here.
		  HEL_PICKED = -1
!         For safety, hardset the helicity sampling jacobian to 0.0d0 to make sure it is not .
		  hel_jacobian   = 1.0d0
		  IF(DS_get_dim_status('Helicity').eq.1) then 
!           If we finished the initialization we can update the grid so as to start sampling over it.
!           However the grid will now be filled by dsample with different kind of weights (including pdf, flux, etc...) so by setting the grid_mode of the reference grid to 'initialization' we make sure it will be overwritten (as opposed to 'combined') by the running grid at the next update.
            CALL DS_UPDATE_GRID('Helicity')
			CALL DS_SET_GRID_MODE('Helicity','init')
	    call reset_cumulative_variable() ! avoid biais of the initialization
          endif
	    ELSE
           JHEL = 1
           IF(NTRY.LE.MAXTRIES)THEN
             DO I=1,NCOMB
                IF (.NOT.GOODHEL(I) .AND. (TS(I).GT.ANS*LIMHEL/NCOMB)) THEN
                   GOODHEL(I)=.TRUE.
                   NGOOD = NGOOD +1
                   IGOOD(NGOOD) = I        
                   print *,'Adding good helicity ',I,TS(I)/ANS
                ENDIF
             ENDDO
          ENDIF
          IF(NTRY.EQ.MAXTRIES)THEN
              ISUM_HEL=MIN(ISUM_HEL,NGOOD)
          ENDIF
        IF(NTRY.EQ.(MAXTRIES+1)) THEN
           call reset_cumulative_variable() ! avoid biais of the initialization
        ENDIF

		endif
       ELSE              !RANDOM HELICITY
C           The helicity configuration was chosen already by genps and put in a common block defined in genps.inc.
            I = HEL_PICKED
	        
			call cpu_time(t_before) 
			CALL %(ml_prefix)sSLOOPMATRIXHEL_THRES(P,I,ANS_ML,-1.0d0,PREC_FOUND,ML_RET_CODE)
		    CALL PROCESS_MADLOOP_OUTPUT%(proc_id)s(P,t_before,T,ANS_ML(1,0),ANS_ML(2,0),ANS_ML(3,0),PREC_FOUND(0),ML_RET_CODE)

            DO JJ=1,JAMP2(0)
              JAMP2(JJ) = JAMP2(JJ) + ML_JAMP2(JJ)
            ENDDO
            DO JJ=1,NLOOPDIAGRAMS
              AMP2(JJ) = AMP2(JJ)+ ML_AMP2(JJ)
            ENDDO

		     DO JJ=1,nincoming
                IF(POL(JJ).NE.1d0.AND.NHEL(JJ,I).EQ.INT(SIGN(1d0,POL(JJ)))) THEN
                  T=T*ABS(POL(JJ))
                ELSE IF(POL(JJ).NE.1d0)THEN
                  T=T*(2d0-ABS(POL(JJ)))
                ENDIF
             ENDDO

c           Always one helicity at a time
            ANS = T

c           Include the Jacobian from helicity sampling
            ANS = ANS * hel_jacobian 

			 WRITE(HEL_BUFF,'(20i5)')(NHEL(II,I),II=1,NEXTERNAL)
       ENDIF
       IF (ISUM_HEL .NE. 1.or.(HEL_PICKED.eq.-1)) THEN
          R=XRAN1(IDUM)*ANS
          SUMHEL=0d0
          DO I=1,NCOMB
            SUMHEL=SUMHEL+DABS(TS(I))
            IF(R.LT.SUMHEL)THEN
               WRITE(HEL_BUFF,'(20i5)')(NHEL(II,I),II=1,NEXTERNAL)
      	       ANS=DSIGN(ANS,TS(i))
              GOTO 10
            ENDIF
          ENDDO
 10       CONTINUE   
       ENDIF
    ENDIF

   IF (MULTI_CHANNEL) THEN
        XTOT=0D0
        DO I=1,NLOOPDIAGRAMS
            XTOT=XTOT+AMP2(I)
        ENDDO
		IF (MULTICHANNEL_THRES.ge.0.0d0) then
		  IF (XTOT.lt.MULTICHANNEL_THRES*(DABS(ANS)/hel_jacobian)) THEN
			INTERFERENCE_CONTRIB = (MULTICHANNEL_THRES*(DABS(ANS)/hel_jacobian))
            AMP2(%(configID_in_matrix)s) = AMP2(%(configID_in_matrix)s) + (INTERFERENCE_CONTRIB/NMULTICHANNELS)
			XTOT = XTOT + INTERFERENCE_CONTRIB
		  ENDIF 
		ENDIF
        IF (XTOT.NE.0D0) THEN
		    ANS=ANS*AMP2(%(configID_in_matrix)s)/XTOT
        ELSE
            ANS=0D0
        ENDIF
    ENDIF
    IF(.not.Force_ML_Helicity_Sum )THEN
      ANS = ANS/ Hel_Average_Factor    			
    ENDIF

C Amplitude(s) for diagram number %(n_tot_diags)d
C This last line is a tag do not remove it. 
    END
 
	SUBROUTINE PROCESS_MADLOOP_OUTPUT%(proc_id)s(P,time_before, me_answer, finite, OneEps, TwoEps, prec, return_code)
c
c   Simply registers the madloop return code and timing into the
C   global variables
c
    implicit none
C
C   Constants
C
    Include 'nexternal.inc'
C
C	Arguments
C
    REAL*8 P(0:3,NEXTERNAL)
	real time_before
	integer return_code
	double precision me_answer, finite, OneEps, TwoEps, prec
C
C	Local variables
C
	real time_after
	integer u,t,h
c
c	Global variables
c
c   To monitor MadLoop computational aspects.
    INTEGER U_RETURN_CODES(0:9)
    INTEGER T_RETURN_CODES(0:9)
    INTEGER H_RETURN_CODES(0:9)
    DOUBLE PRECISION AVG_TIMING
    DOUBLE PRECISION MAX_PREC, MIN_PREC	
    INTEGER N_EVALS
    COMMON/MADLOOPSTATS/AVG_TIMING,MAX_PREC,MIN_PREC,N_EVALS,U_RETURN_CODES,T_RETURN_CODES,H_RETURN_CODES

    INTEGER WARNING_COUNTERS(2)
	DATA WARNING_COUNTERS/2*0/

c   ----
c   Begin code
c   ----

	call cpu_time(time_after)
    AVG_TIMING = (AVG_TIMING * N_EVALS + (time_after-time_before)) / DBLE(N_EVALS+1)
	N_EVALS = N_EVALS + 1

	U = mod(RETURN_CODE,10)
	U_RETURN_CODES(U) = U_RETURN_CODES(U)+1
	T = (mod(RETURN_CODE,100)-U)/10
	T_RETURN_CODES(T) = T_RETURN_CODES(T)+1	
	H = (mod(RETURN_CODE,1000)-10*T-U)/100
	H_RETURN_CODES(H) = H_RETURN_CODES(H)+1

	MAX_PREC = max(PREC, MAX_PREC)
	MIN_PREC = min(PREC, MIN_PREC)

	IF (H.eq.4) THEN
	  WARNING_COUNTERS(1) = WARNING_COUNTERS(1) + 1
	  IF (WARNING_COUNTERS(1).le.10) THEN
	    WRITE(*,*) "WARNING :: MadLoop encountered an exceptional unstable PS point. Its weight is set to 0. Details should be provided by MadLoop before in this log file."
      ENDIF
	  IF (WARNING_COUNTERS(1).eq.10) THEN
	    WRITE(*,*) "WARNING :: Further warnings about exceptional unstable PS points now suppressed (past the first 10)."
	  ENDIF
	  me_answer = 0.0d0
	  RETURN
	ENDIF

	me_answer = FINITE

	if (finite.eq.0.0d0) then
	  return
	endif

	IF (((OneEps+TwoEps)/finite).gt.(1000d0*PREC).AND.(((OneEps+TwoEps)/finite).gt.1.0d-5)) THEN
	  WARNING_COUNTERS(2) = WARNING_COUNTERS(2) + 1
	  IF (WARNING_COUNTERS(2).le.10) THEN
	    WRITE(*,*) "WARNING : The residue of the single and double pole of the loop matrix element being integrated does not seem to vanish."
		WRITE(*,*) "          Its contribution relative to the finite part is : ",((OneEps+TwoEps)/finite)
		WRITE(*,*) "          while the estimated numerical accuracy is       : ",PREC
		WRITE(*,*) "          MadLoop results (fin, 1eps, 2eps)               : ",finite, OneEps, TwoEps
		WRITE(*,*) " The warning above was triggered when processing the following phase space point:"
        CALL %(ml_prefix)sWRITE_MOM(P)
	  ENDIF
	  IF (WARNING_COUNTERS(2).eq.10) THEN
	    WRITE(*,*) "WARNING :: Further warnings about the relative size of pole residues are now suppressed (past first 10)."
	  ENDIF
	ENDIF

    END  
