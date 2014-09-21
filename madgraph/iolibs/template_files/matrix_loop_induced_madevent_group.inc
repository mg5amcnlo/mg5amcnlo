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
C  
C ARGUMENTS 
C 
    REAL*8 P(0:3,NEXTERNAL),ANS
	REAL*8 , ALLOCATABLE :: ANS_ML(:,:)
C  
C LOCAL VARIABLES 
C  
	INTEGER I
	INTEGER ML_ANS_DIMENSION
C  
C GLOBAL VARIABLES
C  
    DOUBLE PRECISION AMP2(MAXAMPS), JAMP2(0:MAXFLOW)
    COMMON/TO_AMPS/  AMP2,       JAMP2
    
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


C ----------
C BEGIN CODE
C ----------
     
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
    
C   Call MadLoop Matrix Element
    CALL %(ml_prefix)sGET_ANSWER_DIMENSION(ML_ANS_DIMENSION)
	ALLOCATE(ANS_ML(0:3,0:ML_ANS_DIMENSION))
	CALL %(ml_prefix)sSLOOPMATRIX(P,ANS_ML)
	ANS = ANS_ML(1,0)
	DEALLOCATE(ANS_ML)

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
    
C Amplitude(s) for diagram number %(n_tot_diags)d

	END
