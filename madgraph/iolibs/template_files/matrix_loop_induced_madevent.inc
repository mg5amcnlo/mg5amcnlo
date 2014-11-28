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
    INTEGER    NLOOPDIAGRAMS
    PARAMETER (NLOOPDIAGRAMS=%(nloopdiags)d)
    INTEGER    NLOOPFLOWS
    PARAMETER (NLoopFlows=%(nLoopFlows)d) 
    INTEGER    THEL
    PARAMETER (THEL=NCOMB)
    INTEGER Hel_Average_Factor
    PARAMETER (Hel_Average_Factor=%(hel_avg_factor)d)
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
    INTEGER NHEL(NEXTERNAL,NCOMB),NTRY
    REAL*8 T,MATRIX%(proc_id)s
    REAL*8 R,SUMHEL,TS(NCOMB)
    INTEGER I,IDEN
    INTEGER IPROC,JC(NEXTERNAL),II
    LOGICAL GOODHEL(NCOMB)
    REAL*8 HWGT, XTOT, XTRY, XREJ, XR, YFRAC(0:NCOMB)
    INTEGER IDUM, NGOOD, IGOOD(NCOMB), JHEL, J, JJ
    REAL     XRAN1
    EXTERNAL XRAN1

	REAL*8 , ALLOCATABLE :: ANS_ML(:,:)
    REAL*8 , ALLOCATABLE :: PREC_FOUND(:)
	INTEGER ML_RET_CODE
    INTEGER ML_ANS_DIMENSION
C Prec_found ans ans_ml are saved so that they don't have to be re-allocated everytime
	SAVE ML_ANS_DIMENSION,ANS_ML,PREC_FOUND

    DATA IDUM /0,-1/
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

	IF(NLOOPDIAGRAMS.GT.MAXAMPS) THEN
	   STOP 'MAXAMPS smaller than to NLOOPDIAGRAMS in matrix.f'
	ENDIF

	IF(.NOT.INIT_MADLOOP) THEN
	  INIT_MADLOOP = .TRUE.
C So that TIR integrals can be reused accross helicity configuration.
C Notice that ITR integrals for rotated PS points (i.e. which is part of MadLoop's default stabiliy test procedure), will not work. This is why one should always keep the MadLoop parameter 'NRotations_DP' set to 0.
	  CALL %(ml_prefix)sSET_AUTOMATIC_TIR_CACHE_CLEARING(.FALSE.)
      CALL %(ml_prefix)sGET_ANSWER_DIMENSION(ML_ANS_DIMENSION)
      ALLOCATE(ANS_ML(0:3,0:ML_ANS_DIMENSION))
	  ALLOCATE(PREC_FOUND(0:ML_ANS_DIMENSION))
	ENDIF
C But then we must clear the cache by hand at the beginning of the computation of each new PS point
	CALL %(ml_prefix)sCLEAR_TIR_CACHE()
   
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

C  Notice that when forcing helicity sum directly in MadLoop then one doesn't have access to the weights of individual Hel. Configs so that they cannot be specified in the event file. We turned this option off by default.
    FORCE_ML_Helicity_Sum = (ISUM_HEL.eq.0.AND..False.)
    if (FORCE_ML_Helicity_Sum)then
C Of course this option can also not be used for polarized beams.
      do JJ=1,NINCOMING
	    if (POL(JJ).NE.1d0) FORCE_ML_Helicity_Sum = .false.
	  enddo	       
    endif

    IF(Force_ML_Helicity_Sum )THEN

        CALL %(ml_prefix)sSLOOPMATRIX_THRES(P,ANS_ML,-1.0d0,PREC_FOUND,ML_RET_CODE)
		ANS = ANS_ML(1,0)
    
	ELSE
    IF (ISUM_HEL.EQ.0.or.(DS_get_dim_status('Helicity').eq.0)) THEN
           DO I=1,NCOMB
              IF (GOODHEL(I) .OR. NTRY .LE. MAXTRIES.OR.(ISUM_HEL.NE.0)) THEN
	          	
				CALL %(ml_prefix)sSLOOPMATRIXHEL_THRES(P,I,ANS_ML,-1.0d0,PREC_FOUND,ML_RET_CODE)
				T = ANS_ML(1,0)         	 
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
        IF(NTRY(IMIRROR).EQ.(MAXTRIES+1)) THEN
           call reset_cumulative_variable() ! avoid biais of the initialization
        ENDIF

		endif
       ELSE              !RANDOM HELICITY
C           The helicity configuration was chosen already by genps and put in a common block defined in genps.inc.
            I = HEL_PICKED
             
			CALL %(ml_prefix)sSLOOPMATRIXHEL_THRES(P,I,ANS_ML,-1.0d0,PREC_FOUND,ML_RET_CODE)
			T = ANS_ML(1,0)
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
        IF (XTOT.NE.0D0) THEN
%(set_amp2_line)s
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
 
 
