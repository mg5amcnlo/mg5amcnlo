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
    PARAMETER (             NCOMB=${ncomb})
    INTEGER    NGRAPHS
    PARAMETER (NGRAPHS=%(ngraphs)d) 
    INTEGER    NDIAGS
    PARAMETER (NDIAGS=%(ndiags)d) 
    INTEGER    THEL
    PARAMETER (THEL=2*NCOMB)
C  
C ARGUMENTS 
C  
    REAL*8 P(0:3,NEXTERNAL),ANS
C  
C LOCAL VARIABLES 
C  
        INTEGER CONFSUB(MAXSPROC,LMAXCONFIGS)
        INCLUDE 'config_subproc_map.inc'
        INTEGER NHEL(NEXTERNAL,NCOMB)
        REAL*8 T
        REAL*8 R,SUMHEL,TS(NCOMB)
        INTEGER I,IDEN
        INTEGER JC(NEXTERNAL),II
        REAL*8 XTOT
        INTEGER  J, JJ

      double precision get_channel_cut
      external get_channel_cut

C  
C GLOBAL VARIABLES
C  
    DOUBLE PRECISION AMP2(MAXAMPS), JAMP2(0:MAXFLOW)
    COMMON/TO_AMPS/  AMP2,       JAMP2


C
C   INFORMATION TO WRITE THE HELICITY IN THE EVENT --not memory efficient--
C   
    CHARACTER*101         HEL_BUFF
    COMMON/TO_HELICITY/  HEL_BUFF

    INTEGER NB_SPIN_STATE_in(2)
    common /nb_hel_state/ nb_spin_state_in
    
    REAL*8 POL(2)
    
    COMMON/TO_POLARIZATION/ POL
       double precision tmin_for_channel
       integer sde_strat ! 1 means standard single diagram enhancement strategy,
c      	      	      	   2 means approximation by the	denominator of the propagator
       common/TO_CHANNEL_STRAT/tmin_for_channel,	sde_strat
       
    INTEGER          ISUM_HEL
    LOGICAL                    MULTI_CHANNEL
    COMMON/TO_MATRIX/ISUM_HEL, MULTI_CHANNEL
%(define_iconfigs_lines)s
${helicity_lines}
%(den_factor_line)s

C ----------
C BEGIN CODE
C ----------

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
    WRITE(HEL_BUFF,'(20I5)') (0,I=1,NEXTERNAL)
c   Kiran please check if you need this:    
    DO I=1,NCOMB
       TS(I)=0d0
    ENDDO

    call MATRIX%(proc_id)s(P ,JC(1), TS)
    DO I=1,NCOMB     
       T=TS(I)  
       DO JJ=1,nincoming
          IF(POL(JJ).NE.1d0.AND.NHEL(JJ,I).EQ.INT(SIGN(1d0,POL(JJ)))) THEN
             T=T*ABS(POL(JJ))*NB_SPIN_STATE_IN(JJ)/2d0 ! NB_SPIN_STATE(JJ)/2d0 is added for polarised beam
          ELSE IF(POL(JJ).NE.1d0)THEN
             T=T*(2d0-ABS(POL(JJ)))*NB_SPIN_STATE_IN(JJ)/2d0
          ENDIF
       ENDDO
       ANS=ANS+DABS(T)
       TS(I)=T
    ENDDO

    IF (ANS.ne.0d0) THEN
        CALL RANMAR(R)
        SUMHEL=0d0
	DO I=1,NCOMB
            SUMHEL=SUMHEL+DABS(TS(I))/ANS
            IF(R.LT.SUMHEL)THEN
                WRITE(HEL_BUFF,'(20i5)')(NHEL(II,I),II=1,NEXTERNAL)
c         Set right sign for ANS, based on sign of chosen helicity
                ANS=DSIGN(ANS,TS(I))
                GOTO 10
            ENDIF
        ENDDO
 10     CONTINUE   
    ENDIF
    IF (MULTI_CHANNEL) THEN
        XTOT=0D0
        DO I=1,LMAXCONFIGS
	    J = CONFSUB(%(proc_id)s, I)
	    if (J.ne.0)then
	      if (sde_strat.eq.1)then
	         AMP2(J) = AMP2(J) * GET_CHANNEL_CUT(P, I)
               else
	     	         AMP2(J) = GET_CHANNEL_CUT(P, I)
	       endif
             XTOT=XTOT+AMP2(J)

             endif
        ENDDO
        IF (XTOT.NE.0D0) THEN
%(set_amp2_line)s
        ELSE IF(ANS.ne.0d0) THEN
			write(*,*) "Problem in the multi-channeling. All amp2 are zero but not the total matrix-element"
	        stop 1
        ENDIF
    ENDIF
    ANS=ANS/DBLE(IDEN)
    END
 
 
Subroutine  MATRIX%(proc_id)s(P,IC, TS)
C  
%(info_lines)s
C
C Returns amplitude squared summed/avg over colors
c for the point with external lines W(0:6,NEXTERNAL)
C  
%(process_lines)s
C  
    IMPLICIT NONE
C  
C CONSTANTS
C  
    INTEGER    NGRAPHS
    PARAMETER (NGRAPHS=%(ngraphs)d) 
    include 'genps.inc'
    include 'nexternal.inc'
    include 'maxamps.inc'
    INTEGER    NWAVEFUNCS,     NCOLOR
    PARAMETER (NWAVEFUNCS=${nwavefuncs}, NCOLOR=%(ncolor)d) 
    REAL*8     ZERO
    PARAMETER (ZERO=0D0)
    COMPLEX*16 IMAG1
    PARAMETER (IMAG1=(0D0,1D0))
    INTEGER NAMPSO, NSQAMPSO
    PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
	LOGICAL CHOSEN_SO_CONFIGS(NSQAMPSO)
	DATA CHOSEN_SO_CONFIGS/%(chosen_so_configs)s/
	SAVE CHOSEN_SO_CONFIGS
    INTEGER                 NCOMB         
    PARAMETER (             NCOMB=${ncomb})
C  
C ARGUMENTS 
C
    REAL*8 P(0:3,NEXTERNAL)
    INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
    REAL*8 TS(NCOMB)
C  
C LOCAL VARIABLES 
C  
    INTEGER I,J,M,N,K
    COMPLEX*16 ZTEMP,TMP_JAMP(%(nb_temp_jamp)i)
    COMPLEX*16 TMP(6)
    REAL*8 CF(NCOLOR,NCOLOR)
    COMPLEX*16 AMP(NCOMB,NGRAPHS), JAMP(NCOLOR,NAMPSO)
    COMPLEX*16 W(%(wavefunctionsize)d,NWAVEFUNCS)
C   Needed for v4 models
    COMPLEX*16 DUM0,DUM1
    DATA DUM0, DUM1/(0d0, 0d0), (1d0, 0d0)/

	%(fake_width_declaration)s
	logical first
	data first /.true./
	save first
C
C FUNCTION
C
      INTEGER SQSOINDEX%(proc_id)s

C  
C GLOBAL VARIABLES
C  
    Double Precision amp2(maxamps), jamp2(0:maxflow)
    common/to_amps/  amp2,       jamp2
    include 'coupl.inc'

       double precision tmin_for_channel
       integer sde_strat ! 1 means standard single diagram enhancement strategy,
c      	      	      	   2 means approximation by the	denominator of the propagator
       common/TO_CHANNEL_STRAT/tmin_for_channel,	sde_strat

	double precision small_width_treatment
	common/narrow_width/small_width_treatment
C  
C COLOR DATA
C  
%(color_data_lines)s
C ----------
C BEGIN CODE
C ----------
if (first) then
	first=.false.
	%(fake_width_definitions)s
endif
    AMP(:,:) = (0d0,0d0)
${helas_calls}
    DO K = 1, NCOMB
${jamp_lines}
        TS(K) = 0.D0 
    	DO M = 1, NAMPSO
          DO I = 1, NCOLOR
            ZTEMP = (0.D0,0.D0)
            DO J = 1, NCOLOR
              ZTEMP = ZTEMP + CF(J,I)*JAMP(J,M)
            ENDDO
    		DO N = 1, NAMPSO
			%(select_configs_if)s
    		     TS(K) = TS(K) + ZTEMP*DCONJG(JAMP(I,N))
			  %(select_configs_endif)s
    		ENDDO
          ENDDO
    	ENDDO
	if(sde_strat.eq.1) then
        ${amp2_lines}
	endif
        Do I = 1, NCOLOR
    	  DO M = 1, NAMPSO
    		DO N = 1, NAMPSO
			%(select_configs_if)s
                Jamp2(i)=Jamp2(i)+DABS(DBLE(Jamp(i,m)*dconjg(Jamp(i,n))))
			%(select_configs_endif)s
    		enddo
    	  enddo
        Enddo
    ENDDO

    END


    SUBROUTINE PRINT_ZERO_AMP_%(proc_id)s()

    integer i
    	    i =1
    return
    end