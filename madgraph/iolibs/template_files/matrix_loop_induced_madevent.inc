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
    INTEGER    NDIAGS
    PARAMETER (NDIAGS=%(ndiags)d) 
    INTEGER    NLOOPAMPS
    PARAMETER (NLOOPAMPS=%(nloopamps)d) 
    INTEGER    NLOOPFLOWS
    PARAMETER (NLoopFlows=%(nLoopFlows)d) 
    INTEGER    THEL
    PARAMETER (THEL=NCOMB)
    INTEGER NCTAMPS 
    PARAMETER (NCTAMPS=%(nctamps)d) 
    INTEGER Hel_Average_Factor
    PARAMETER (Hel_Average_Factor=%(hel_avg_factor)d)
C  
C ARGUMENTS 
C  
    REAL*8 P(0:3,NEXTERNAL),ANS
	REAL*8 , ALLOCATABLE :: ANS_ML(:,:)
C  
C LOCAL VARIABLES 
C  
    INTEGER NHEL(NEXTERNAL,NCOMB),NTRY
    REAL*8 T,MATRIX%(proc_id)s
    REAL*8 R,SUMHEL,TS(NCOMB)
    INTEGER I,IDEN
    INTEGER IPROC,JC(NEXTERNAL),II
    INTEGER ML_ANS_DIMENSION
    LOGICAL GOODHEL(NCOMB)
    DATA GOODHEL/THEL*.FALSE./
    REAL*8 HWGT, XTOT, XTRY, XREJ, XR, YFRAC(0:NCOMB)
    INTEGER IDUM, NGOOD, IGOOD(NCOMB), JHEL, J, JJ
    REAL     XRAN1
    EXTERNAL XRAN1

    DATA NTRY,IDUM /0,-1/
    DATA XTRY, XREJ, NGOOD /0,0,0/
C  
C GLOBAL VARIABLES
C  
    LOGICAL FORCE_ML_HELICITY_SUM 
    common/to_ML_control/FORCE_ML_HELICITY_SUM 

    DOUBLE PRECISION AMP2(MAXAMPS), JAMP2(0:MAXFLOW)
    COMMON/TO_AMPS/  AMP2,       JAMP2
    
    %(complex_dp_format)s ML_JAMP2(nLoopFlows)
	common/%(ml_prefix)sJAMP2/ML_JAMP2

    %(real_dp_format)s ML_AMP2(nloopamps)
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
    NTRY=NTRY+1
    DO I=1,NEXTERNAL
       JC(I) = +1
    ENDDO
     
    IF (multi_channel) THEN
        DO I=1,NDIAGS
            AMP2(I)=0D0
        ENDDO
        JAMP2(0)=%(ncolor)d
        DO I=1,INT(JAMP2(0))
            JAMP2(I)=0D0
        ENDDO
    ENDIF
    ANS = 0D0
    
    CALL %(ml_prefix)sGET_ANSWER_DIMENSION(ML_ANS_DIMENSION)
    ALLOCATE(ANS_ML(0:3,0:ML_ANS_DIMENSION))
    DO I=1,NCOMB
       TS(I)=0d0
    ENDDO

    FORCE_ML_Helicity_Sum = ISUM_HEL.eq.0
    if (FORCE_ML_Helicity_Sum)then
        do JJ=1,NINCOMING
	   if (POL(JJ).NE.1d0) FORCE_ML_Helicity_Sum = .false.
	enddo	       
    endif	       


    IF(Force_ML_Helicity_Sum )THEN
        CALL %(ml_prefix)sSLOOPMATRIX(P,ANS_ML)
	ANS = ANS_ML(1,0)
    ELSE
	IF (ISUM_HEL .EQ. 0 .OR. NTRY .LE. MAXTRIES) THEN
           DO I=1,NCOMB
              IF (GOODHEL(I) .OR. NTRY .LE. MAXTRIES) THEN
	          CALL %(ml_prefix)sSLOOPMATRIXHEL(P,I,ANS_ML)
	    	  T = ANS_ML(1,0)              	 
		  DO JJ=1,nincoming
               	     IF(POL(JJ).NE.1d0.AND.NHEL(JJ,I).EQ.INT(SIGN(1d0,POL(JJ)))) THEN
                         T=T*ABS(POL(JJ))
                     ELSE IF(POL(JJ).NE.1d0)THEN
                         T=T*(2d0-ABS(POL(JJ)))
                     ENDIF
                  ENDDO
                  ANS=ANS+DABS(T)
                  TS(I)=T
              ENDIF
           ENDDO
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
       ELSE              !RANDOM HELICITY
          DO J=1,ISUM_HEL
             HWGT = REAL(NGOOD)/REAL(ISUM_HEL)
             I = IGOOD(J)
             CALL %(ml_prefix)sSLOOPMATRIXHEL(P,I,ANS_ML)
	     T = ANS_ML(1,0)
             DO JJ=1,nincoming
                IF(POL(JJ).NE.1d0.AND.NHEL(JJ,I).EQ.INT(SIGN(1d0,POL(JJ)))) THEN
                  T=T*ABS(POL(JJ))
                ELSE IF(POL(JJ).NE.1d0)THEN
                  T=T*(2d0-ABS(POL(JJ)))
                ENDIF
             ENDDO
             ANS=ANS+DABS(T)*HWGT
             TS(I)=T*HWGT
          ENDDO
          IF (ISUM_HEL .EQ. 1) THEN
             ANS=DSIGN(ANS,TS(I))
          ENDIF
       ENDIF
       IF (ISUM_HEL .NE. 1) THEN
          R=XRAN1(IDUM)*ANS
          SUMHEL=0d0
          DO I=1,NCOMB
            SUMHEL=SUMHEL+DABS(TS(I))
            IF(R.LT.SUMHEL)THEN
      	       ANS=DSIGN(ANS,TS(i))
              GOTO 10
            ENDIF
          ENDDO
 10       CONTINUE   
       ENDIF
    ENDIF
    
    DO I=1,JAMP2(0)
       JAMP2(I) = JAMP2(i) + ML_JAMP2(I)
    ENDDO
    DO I=1,NDIAGS
       AMP2(i) = AMP2(i)+ ML_AMP2(I+NCTAMPS)
    ENDDO

   IF (MULTI_CHANNEL) THEN
        XTOT=0D0
        DO I=1,NDIAGS
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

    DEALLOCATE(ANS_ML)
C Amplitude(s) for diagram number %(n_tot_diags)d
C This last line is a tag do not remove it. 
    END
 
 
