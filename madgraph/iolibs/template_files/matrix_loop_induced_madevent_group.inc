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
    INTEGER    NDIAGS
    PARAMETER (NDIAGS=%(ndiags)d) 
    INTEGER    NLOOPAMPS
    PARAMETER (NLOOPAMPS=%(nloopamps)d) 
    INTEGER    NLOOPFLOWS
    PARAMETER (NLoopFlows=%(nLoopFlows)d) 
    INTEGER                 NCOMB         
    PARAMETER (             NCOMB=%(ncomb)d)
    INTEGER    THEL
    PARAMETER (THEL=2*NCOMB)
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
        INTEGER NHEL(NEXTERNAL,NCOMB)
	INTEGER I, J ,JJ
	INTEGER ML_ANS_DIMENSION
        INTEGER NTRY(2)
        INTEGER ISHEL(2)
        REAL*8 R,SUMHEL,TS(NCOMB), T
        LOGICAL GOODHEL(NCOMB,2)
        DATA GOODHEL/THEL*.FALSE./
	REAL*8 HWGT,XTOT
        INTEGER NGOOD(2), IGOOD(NCOMB,2)         
        DATA NTRY /0,0/
        DATA NGOOD /0,0/
        DATA ISHEL/0,0/
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

    CHARACTER*101         HEL_BUFF
    COMMON/TO_HELICITY/  HEL_BUFF
    
    INTEGER IMIRROR
    COMMON/TO_MIRROR/ IMIRROR
 
    REAL*8 POL(2)
    COMMON/TO_POLARIZATION/ POL
    
    INTEGER          ISUM_HEL
    LOGICAL                    MULTI_CHANNEL
    COMMON/TO_MATRIX/ISUM_HEL, MULTI_CHANNEL
%(define_iconfigs_lines)s
%(helicity_lines)s

C ----------
C BEGIN CODE
C ----------
    NTRY(IMIRROR)=NTRY(IMIRROR)+1

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

C   Temporary need to be set in the driver! following the run_card input.
    FORCE_ML_Helicity_Sum = .true.

    IF(Force_ML_Helicity_Sum )THEN
        CALL %(ml_prefix)sSLOOPMATRIX(P,ANS_ML)
	ANS = ANS_ML(1,0)
    ELSE
	IF (ISHEL(IMIRROR) .EQ. 0 .OR. NTRY(IMIRROR) .LE. MAXTRIES) THEN
      	 DO I=1,NCOMB
         IF (GOODHEL(I,IMIRROR) .OR. NTRY(IMIRROR).LE.MAXTRIES) THEN
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
C      JHEL(IMIRROR) = 1
        IF(NTRY(IMIRROR).LE.MAXTRIES)THEN
         DO I=1,NCOMB
            IF (.NOT.GOODHEL(I,IMIRROR) .AND. (DABS(TS(I)).GT.ANS*LIMHEL/NCOMB)) THEN
               GOODHEL(I,IMIRROR)=.TRUE.
               NGOOD(IMIRROR) = NGOOD(IMIRROR) +1
               IGOOD(NGOOD(IMIRROR),IMIRROR) = I
               PRINT *,'Added good helicity ',I,TS(I)*NCOMB/ANS,' in event ',NTRY(IMIRROR)
            ENDIF
         ENDDO
        ENDIF
        IF(NTRY(IMIRROR).EQ.MAXTRIES)THEN
           ISHEL(IMIRROR)=MIN(ISUM_HEL,NGOOD(IMIRROR))
        ENDIF
      ELSE !LOOP OVER GOOD HELICITIES
        DO J=1,ISHEL(IMIRROR)
c            JHEL(IMIRROR)=JHEL(IMIRROR)+1
c            IF (JHEL(IMIRROR) .GT. NGOOD(IMIRROR)) JHEL(IMIRROR)=1
            HWGT = REAL(NGOOD(IMIRROR))/REAL(ISHEL(IMIRROR))
            I = IGOOD(J,IMIRROR)
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
        IF (ISHEL(IMIRROR) .EQ. 1) THEN
c         Set right sign for ANS, based on sign of chosen helicity
          ANS=DSIGN(ANS,TS(I))
        ENDIF
      ENDIF
      IF (ISHEL(IMIRROR) .NE. 1) THEN
         CALL RANMAR(R)
         SUMHEL=0d0
         DO I=1,NCOMB
            SUMHEL=SUMHEL+DABS(TS(I))/ANS
            IF(R.LT.SUMHEL)THEN
c         Set right sign for ANS, based on sign of chosen helicity
               ANS=DSIGN(ANS,TS(I))
               GOTO 10
            ENDIF
         ENDDO
 10      CONTINUE   
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
    ANS=ANS
    IF(.not.Force_ML_Helicity_Sum )THEN
      ANS = ANS/ Hel_Average_Factor    			
    ENDIF

	DEALLOCATE(ANS_ML)
C Amplitude(s) for diagram number %(n_tot_diags)d
C This last line is a tag do not remove it. 
	END
