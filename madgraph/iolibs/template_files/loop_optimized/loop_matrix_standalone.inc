      SUBROUTINE SLOOPMATRIXHEL(P,HEL,ANS)
      IMPLICIT NONE
C  
C CONSTANTS
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
C  
C ARGUMENTS 
C  
      %(real_dp_format)s P(0:3,NEXTERNAL)
      %(real_dp_format)s ANS(3)
	  INTEGER HEL, USERHEL
	  common/USERCHOICE/USERHEL
C ----------
C BEGIN CODE
C ----------
	  USERHEL=HEL
      CALL SLOOPMATRIX(P,ANS)
	  END

      SUBROUTINE SLOOPMATRIX(P_IN,ANS)
C  
%(info_lines)s
C
C Returns amplitude squared summed/avg over colors
c and helicities for the point in phase space P(0:3,NEXTERNAL)
c and external lines W(0:6,NEXTERNAL)
C  
%(process_lines)s
C  
      IMPLICIT NONE
C  
C CONSTANTS
C
      CHARACTER*64 paramFileName
      PARAMETER ( paramFileName='MadLoopParams.dat')
	  %(nbornamps_decl)s
      INTEGER    NLOOPS, NLOOPGROUPS, NCTAMPS
      PARAMETER (NLOOPS=%(nloops)d, NLOOPGROUPS=%(nloop_groups)d, NCTAMPS=%(nctamps)d) 
      INTEGER    NCOLORROWS
	  PARAMETER (NCOLORROWS=%(nloopamps)d)
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER    NWAVEFUNCS,NLOOPWAVEFUNCS
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d,NLOOPWAVEFUNCS=%(nloopwavefuncs)d)
	  INTEGER MAXLWFSIZE
	  PARAMETER (MAXLWFSIZE=%(max_lwf_size)d)
	  INTEGER LOOPMAXCOEFS, VERTEXMAXCOEFS
	  PARAMETER (LOOPMAXCOEFS=%(loop_max_coefs)d, VERTEXMAXCOEFS=%(vertex_max_coefs)d)
	  INTEGER    NCOMB
      PARAMETER (NCOMB=%(ncomb)d)
      %(real_dp_format)s     ZERO
      PARAMETER (ZERO=0D0)
	  %(real_mp_format)s     MP__ZERO
      PARAMETER (MP__ZERO=0E0_16)
      %(complex_dp_format)s IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
C     This parameter is designed for the check timing command of MG5
      LOGICAL SKIPLOOPEVAL
      PARAMETER (SKIPLOOPEVAL=.FALSE.)
	  LOGICAL BOOTANDSTOP
      PARAMETER (BOOTANDSTOP=.FALSE.)
	  INTEGER MAXSTABILITYLENGTH
	  DATA MAXSTABILITYLENGTH/20/
	  common/stability_tests/maxstabilitylength
C  
C ARGUMENTS 
C  
      %(real_dp_format)s P_IN(0:3,NEXTERNAL)
      %(real_dp_format)s ANS(3)
C  
C LOCAL VARIABLES 
C  
      INTEGER I,J,K,H,DUMMY
      %(real_dp_format)s P(0:3,NEXTERNAL)	  
C DP_RES STORES THE DOUBLE PRECISION RESULT OBTAINED FROM DIFFERENT EVALUATION METHODS IN ORDER TO ASSESS STABILITY.
C THE STAB_STAGE COUNTER I CORRESPONDANCE GOES AS FOLLOWS
C  I=1 -> ORIGINAL PS, CTMODE=1
C  I=2 -> ORIGINAL PS, CTMODE=2, (ONLY WITH CTMODERUN=-1)
C  I=3 -> PS WITH ROTATION 1, CTMODE=1, (ONLY WITH CTMODERUN=-2)
C  I=4 -> PS WITH ROTATION 2, CTMODE=1, (ONLY WITH CTMODERUN=-3)
C  I=5 -> POSSIBLY MORE EVALUATION METHODS IN THE FUTURE, MAX IS MAXSTABILITYLENGTH
C IF UNSTABLE IT GOES TO THE SAME PATTERN BUT STAB_INDEX IS THEN I+20.
      LOGICAL EVAL_DONE(MAXSTABILITYLENGTH)
	  LOGICAL DOING_QP_EVALS
      INTEGER STAB_INDEX,BASIC_CT_MODE
	  INTEGER N_DP_EVAL, N_QP_EVAL
	  DATA N_DP_EVAL/1/
	  DATA N_QP_EVAL/1/
	  %(real_dp_format)s ACC
	  %(real_dp_format)s DP_RES(3,MAXSTABILITYLENGTH)
C QP_RES STORES THE QUADRUPLE PRECISION RESULT OBTAINED FROM DIFFERENT EVALUATION METHODS IN ORDER TO ASSESS STABILITY.
	  %(real_dp_format)s QP_RES(3,MAXSTABILITYLENGTH)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
	  INTEGER NATTEMPTS
	  DATA NATTEMPTS/0/
	  DATA IC/NEXTERNAL*1/
	  %(real_dp_format)s HELSAVED(3,NCOMB)	  
	  %(real_dp_format)s BUFFR(3),BUFFR_BIS(3),TEMP(3),TEMP1,TEMP2
	  %(complex_dp_format)s COEFS(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)
      %(complex_dp_format)s CFTOT
	  LOGICAL FOUNDHELFILTER,FOUNDLOOPFILTER
	  DATA FOUNDHELFILTER/.TRUE./
	  DATA FOUNDLOOPFILTER/.TRUE./
      LOGICAL LOOPFILTERBUFF(NLOOPGROUPS)
	  DATA (LOOPFILTERBUFF(I),I=1,NLOOPGROUPS)/NLOOPGROUPS*.FALSE./

	  INTEGER IDEN
      %(den_factor_line)s
	  INTEGER HELAVGFACTOR
	  DATA HELAVGFACTOR/%(hel_avg_factor)d/
      LOGICAL DONEHELDOUBLECHECK
      DATA DONEHELDOUBLECHECK/.FALSE./
	  INTEGER NEPS
	  DATA NEPS/0/
C
C FUNCTIONS
C
      INTEGER ISSAME
      LOGICAL ISZERO
C  
C GLOBAL VARIABLES
C  
      include 'coupl.inc'
	  include 'mp_coupl.inc'
	  include 'MadLoopParams.inc'

	  LOGICAL CHECKPHASE
	  DATA CHECKPHASE/.TRUE./
	  LOGICAL HELDOUBLECHECKED
	  DATA HELDOUBLECHECKED/.FALSE./
      common/INIT/CHECKPHASE, HELDOUBLECHECKED
	  INTEGER NTRY
      DATA NTRY/0/
	  %(real_dp_format)s REF
	  DATA REF/0.0d0/

	  LOGICAL MP_DONE
	  DATA MP_DONE/.FALSE./
	  common/MP_DONE/MP_DONE

	  %(real_dp_format)s PS(0:3,NEXTERNAL)
	  common/PSPOINT/PS

	  %(real_dp_format)s LSCALE
	  INTEGER CTMODE	  
      common/CT/LSCALE,CTMODE

C     The parameter below sets the convention for the helicity filter
C     For a given helicity, the attached integer 'i' means
C     'i' in ]-inf;-HELOFFSET[ -> Helicity is equal, up to a sign, to helicity number abs(i+HELOFFSET)
C     'i' == -HELOFFSET        -> Helicity is analytically zero
C     'i' in ]-HELOFFSET,inf[  -> Helicity is contributing with weight 'i'. If it is zero, it is skipped.
C     Typically, the hel_offset is 10000
	  INTEGER HELOFFSET
	  DATA HELOFFSET/%(hel_offset)d/
	  INTEGER GOODHEL(NCOMB)
	  LOGICAL GOODAMP(NLOOPGROUPS)
	  common/Filters/GOODAMP,GOODHEL,HELOFFSET

	  INTEGER HELPICKED
	  DATA HELPICKED/-1/
	  common/HELCHOICE/HELPICKED
	  INTEGER USERHEL
	  DATA USERHEL/-1/
	  common/USERCHOICE/USERHEL

	  %(dp_born_amps_decl)s	  
	  %(complex_dp_format)s W(20,NWAVEFUNCS)
	  common/W/W  

      %(complex_mp_format)s MPW(20,NWAVEFUNCS)
	  common/MP_W/MPW

	  %(complex_dp_format)s WL(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE,0:NLOOPWAVEFUNCS)
	  %(complex_dp_format)s PL(0:3,0:NLOOPWAVEFUNCS)
	  common/WL/WL,PL

	  %(complex_dp_format)s LOOPCOEFS(0:LOOPMAXCOEFS-1,NLOOPS)
	  common/LCOEFS/LOOPCOEFS

      %(complex_dp_format)s AMPL(3,NCTAMPS)
	  common/AMPL/AMPL

      %(complex_dp_format)s LOOPRES(3,NLOOPGROUPS)
	  LOGICAL S(NLOOPGROUPS)
	  common/LOOPRES/LOOPRES,S

	  INTEGER CF_D(NCOLORROWS,%(color_matrix_size)s)
	  INTEGER CF_N(NCOLORROWS,%(color_matrix_size)s)
	  common/CF/CF_D,CF_N

	  INTEGER HELC(NEXTERNAL,NCOMB)
	  common/HELCONFIGS/HELC

C ----------
C BEGIN CODE
C ----------

IF(NTRY.EQ.0) THEN
  CALL MADLOOPPARAMREADER(paramFileName,.TRUE.)
  CALL SET_N_EVALS(N_DP_EVAL,N_QP_EVAL)  
  HELDOUBLECHECKED=.NOT.DoubleCheckHelicityFilter
OPEN(1, FILE="LoopFilter.dat", err=100, status='OLD',           action='READ')
  DO J=1,NLOOPGROUPS
    READ(1,*,END=101) GOODAMP(J)
  ENDDO
  GOTO 101
100  CONTINUE
  FOUNDLOOPFILTER=.FALSE.
  DO J=1,NLOOPGROUPS
	GOODAMP(J)=(.NOT.USELOOPFILTER)
  ENDDO
101  CONTINUE
CLOSE(1)
OPEN(1, FILE="HelFilter.dat", err=102, status='OLD',           action='READ')
  DO I=1,NCOMB
    READ(1,*,END=103) GOODHEL(I)
  ENDDO
  GOTO 103
102  CONTINUE
  FOUNDHELFILTER=.FALSE.
  DO J=1,NCOMB
	GOODHEL(J)=1
  ENDDO
103  CONTINUE
CLOSE(1)
OPEN(1, FILE="ColorNumFactors.dat", err=104, status='OLD',           action='READ')
  DO I=1,NCOLORROWS
    READ(1,*,END=105) (CF_N(I,J),J=1,%(color_matrix_size)s)
  ENDDO
  GOTO 105
104  CONTINUE
  STOP 'Color factors could not be initialized from file ColorNumFactors.dat. File not found' 
105  CONTINUE
CLOSE(1)
OPEN(1, FILE="ColorDenomFactors.dat", err=106, status='OLD',           action='READ')
  DO I=1,NCOLORROWS
    READ(1,*,END=107) (CF_D(I,J),J=1,%(color_matrix_size)s)
  ENDDO
  GOTO 107
106  CONTINUE
  STOP 'Color factors could not be initialized from file ColorDenomFactors.dat. File not found' 
107  CONTINUE
CLOSE(1)
OPEN(1, FILE="HelConfigs.dat", err=108, status='OLD',                  action='READ')
  DO H=1,NCOMB
    READ(1,*,END=109) (HELC(I,H),I=1,NEXTERNAL)
  ENDDO
  GOTO 109
108  CONTINUE
  STOP 'Color helictiy configurations could not be initialized from file HelConfigs.dat. File not found' 
109  CONTINUE
CLOSE(1)

C SETUP OF THE COMMON STARTING EXTERNAL LOOP WAVEFUNCTION
C IT IS ALSO PS POINT INDEPENDENT, SO IT CAN BE DONE HERE.
  DO I=0,3
    PL(I,0)=(0.0d0,0.0d0)
  ENDDO
  DO I=1,MAXLWFSIZE
    DO J=0,LOOPMAXCOEFS-1
	  DO K=1,MAXLWFSIZE
	    IF(I.EQ.K.AND.J.EQ.0) then
        	WL(I,J,K,0)=(1.0d0,0.0d0)
		ELSE
        	WL(I,J,K,0)=(0.0d0,0.0d0)
		ENDIF
	  ENDDO
	ENDDO
  ENDDO
  IF(BOOTANDSTOP) THEN
    WRITE(*,*) 'Stopped by user request.'
    STOP
  ENDIF
ENDIF

STAB_INDEX=0
DOING_QP_EVALS=.FALSE.
EVAL_DONE(1)=.TRUE.
DO I=2,MAXSTABILITYLENGTH
  EVAL_DONE(I)=.FALSE.
ENDDO
IF(DONEHELDOUBLECHECK) THEN
  HELDOUBLECHECKED=.TRUE.
  DONEHELDOUBLECHECK=.FALSE.
ENDIF
NTRY=NTRY+1

CHECKPHASE=(NTRY.LE.CHECKCYCLE).AND.(((.NOT.FOUNDLOOPFILTER).AND.USELOOPFILTER).OR.(.NOT.FOUNDHELFILTER))

IF ((.NOT. CHECKPHASE).AND.(.NOT.FOUNDHELFILTER)) THEN
OPEN(1, FILE="HelFilter.dat", err=110, status='NEW',       action='WRITE')
  DO I=1,NCOMB
    WRITE(1,*) GOODHEL(I)  
  ENDDO
110  CONTINUE
  CLOSE(1)
FOUNDHELFILTER=.TRUE.
ENDIF

IF ((.NOT. CHECKPHASE).AND.(.NOT.FOUNDLOOPFILTER).AND.USELOOPFILTER) THEN
OPEN(1, FILE="LoopFilter.dat", err=111, status='NEW',       action='WRITE')
  DO J=1,NLOOPGROUPS
    WRITE(1,*) GOODAMP(J)
  ENDDO
111  CONTINUE
  CLOSE(1)
FOUNDLOOPFILTER=.TRUE.
ENDIF

IF(CHECKPHASE.OR.(.NOT.HELDOUBLECHECKED)) THEN
  HELPICKED=1
  CTMODE=CTMODEINIT
ELSE
  IF (USERHEL.ne.-1) then
    IF(GOODHEL(USERHEL).eq.-HELOFFSET) THEN
      ANS(1)=0.0d0
      ANS(2)=0.0d0
      ANS(3)=0.0d0
      goto 9999
    ENDIF
  ENDIF
  HELPICKED=USERHEL
  IF (CTMODERUN.NE.-1) THEN
    CTMODE=CTMODERUN
  ELSE
    CTMODE=1
  ENDIF
ENDIF

DO I=1,NEXTERNAL
  DO J=0,3
    P(J,I)=P_IN(J,I)
    PS(J,I)=P_IN(J,I)
  ENDDO
ENDDO

%(set_reference)s

DO K=1, 3
  BUFFR(K)=0.0d0
  DO I=1,NCTAMPS
    AMPL(K,I)=(0.0d0,0.0d0)
  ENDDO
ENDDO

200 CONTINUE

C This improvement seems to yield problems for some PS point of d d~ > w+ w- g
C So it is safer to remove it for double precision.
C Instead of having n-1 particles exactly onshell and perfect energy-momentum
c conservation, this subroutine should be written so to have n exactly onshell part.
C CALL IMPROVE_PS_POINT_PRECISION(P)

LSCALE=DSQRT(ABS((P(0,1)+P(0,2))**2-(P(1,1)+P(1,2))**2-(P(2,1)+P(2,2))**2-(P(3,1)+P(3,2))**2))

DO I=1,NLOOPS
  DO J=0,LOOPMAXCOEFS-1
    LOOPCOEFS(J,I)=(0.0d0,0.0d0)
  ENDDO
ENDDO

DO K=1,3
  ANS(K)=0.0d0
ENDDO

C Check if we directly go to multiple precision
IF (CTMODE.GE.4) THEN
  CALL MP_COMPUTE_LOOP_COEFS(P,BUFFR_BIS)
  MP_DONE=.TRUE.
  goto 300 
ELSE
  MP_DONE=.FALSE.
ENDIF

DO H=1,NCOMB
  IF ((HELPICKED.EQ.H).OR.((HELPICKED.EQ.-1).AND.(CHECKPHASE.OR.(.NOT.HELDOUBLECHECKED).OR.(GOODHEL(H).GT.-HELOFFSET.AND.GOODHEL(H).NE.0)))) THEN
  DO I=1,NEXTERNAL
    NHEL(I)=HELC(I,H)
  ENDDO
  %(born_ct_helas_calls)s
  IF (.NOT.CHECKPHASE.AND.HELDOUBLECHECKED.AND.HELPICKED.EQ.-1) THEN
    DUMMY=GOODHEL(H)
  ELSE
    DUMMY=1
  ENDIF
  DO I=1,%(nctamps_or_nloopamps)s
    DO J=1,%(nbornamps_or_nloopamps)s
	  CFTOT=DCMPLX(CF_N(I,J)/DBLE(ABS(CF_D(I,J))),0.0d0)
      IF(CF_D(I,J).LT.0) CFTOT=CFTOT*IMAG1
	  %(squaring)s
    ENDDO
  ENDDO
  %(coef_construction)s  
  ENDIF
ENDDO

%(coef_merging)s

BUFFR_BIS(1)=ANS(1)
BUFFR_BIS(2)=ANS(2)
BUFFR_BIS(3)=ANS(3)
300 CONTINUE
ANS(1)=BUFFR_BIS(1)
ANS(2)=BUFFR_BIS(2)
ANS(3)=BUFFR_BIS(3)

IF(SKIPLOOPEVAL) THEN
  GOTO 1226
ENDIF

%(loop_CT_calls)s

%(actualize_ans)s

1226 CONTINUE

IF (CHECKPHASE.OR.(.NOT.HELDOUBLECHECKED)) THEN
  IF((USERHEL.EQ.-1).OR.(USERHEL.EQ.HELPICKED)) THEN
C   TO KEEP TRACK OF THE FINAL ANSWER TO BE RETURNED DURING CHECK PHASE
    BUFFR(1)=BUFFR(1)+ANS(1)
    BUFFR(2)=BUFFR(2)+ANS(2)
    BUFFR(3)=BUFFR(3)+ANS(3)
  ENDIF
C SAVE RESULT OF EACH INDEPENDENT HELICITY FOR COMPARISON DURING THE HELICITY FILTER SETUP
  HELSAVED(1,HELPICKED)=ANS(1)
  HELSAVED(2,HELPICKED)=ANS(2)
  HELSAVED(3,HELPICKED)=ANS(3)

  IF (CHECKPHASE) THEN
C   SET THE HELICITY FILTER
    IF(.NOT.FOUNDHELFILTER) THEN
      IF(ISZERO(ABS(ANS(1))+ABS(ANS(2))+ABS(ANS(3)),REF/DBLE(NCOMB),-1)) THEN
        IF(NTRY.EQ.1) THEN
	      GOODHEL(HELPICKED)=-HELOFFSET
	    ELSEIF(GOODHEL(HELPICKED).NE.-HELOFFSET) THEN
	      WRITE(*,*) '##W02 WARNING Inconsistent vanishing helicity ',HELPICKED,' filter setting, starting over.'
	      NTRY=0
	    ENDIF
      ELSE
	    DO H=1,HELPICKED-1
		  IF(GOODHEL(H).GT.-HELOFFSET) THEN
C           Be looser for helicity check, bring a factor 100
		    DUMMY=ISSAME(ANS,HELSAVED(1,H),REF)
		    IF(DUMMY.NE.0) THEN
		      IF(NTRY.EQ.1) THEN
C               Set the matching helicity to be contributing once more
		        GOODHEL(H)=GOODHEL(H)+DUMMY
C               Use an offset to clearly show it is linked to an other one and to avoid overlap
			    GOODHEL(HELPICKED)=-H-HELOFFSET
C             Make sure we have paired this hel config to the same one last PS point
			  ELSEIF(GOODHEL(HELPICKED).NE.(-H-HELOFFSET)) THEN
	      	    WRITE(*,*) '##W02 WARNING Inconsistent matching helicity ',HELPICKED,' filter setting, starting over.'
	      	    NTRY=0
			  ENDIF
			  EXIT
			ENDIF
		  ENDIF
		ENDDO
	  ENDIF
    ENDIF

C   SET THE LOOP FILTER
    IF(.NOT.FOUNDLOOPFILTER.AND.USELOOPFILTER) THEN
  	  DO I=1,NLOOPGROUPS
        IF(.NOT.ISZERO(ABS(LOOPRES(1,I))+ABS(LOOPRES(2,I))+ABS(LOOPRES(3,I)),(REF*1.0d-4),I)) THEN
          IF(NTRY.EQ.1) THEN
	        GOODAMP(I)=.TRUE.
		    LOOPFILTERBUFF(I)=.TRUE.
	      ELSEIF(.NOT.LOOPFILTERBUFF(I)) THEN
	        WRITE(*,*) '##W02 WARNING Inconsistent loop amp ',I,'.'
		    IF(LOOPINITSTARTOVER) THEN
		      WRITE(*,*) '##I01 INFO Initialization starting over because of inconsistency in the loop filter setup.'
	          NTRY=0
		    ELSE
		      GOODAMP(I)=.TRUE.
		    ENDIF
	      ENDIF
        ENDIF
  	  ENDDO
    ENDIF
  ELSEIF (.NOT.HELDOUBLECHECKED)THEN
C   DOUBLE CHECK THE HELICITY FILTER
    IF ((GOODHEL(HELPICKED).EQ.-HELOFFSET).AND.(.NOT.ISZERO(ABS(ANS(1))+ABS(ANS(2))+ABS(ANS(3)),REF/DBLE(NCOMB),-1))) THEN
	  write(*,*) '##E03 Helicity filter could not be successfully double checked.'
	  write(*,*) 'One reason for this is that you have changed sensible parameters which affected what are the zero helicity configurations.'
	  write(*,*) 'Please regenerate the filter by removing HelFilter.dat and rerunning SLOOPMATRIX over a couple of PS points.'
	  STOP
	ELSEIF (GOODHEL(HELPICKED).LT.-HELOFFSET.AND.(ISSAME(ANS,HELSAVED(1,ABS(GOODHEL(HELPICKED)+HELOFFSET)),REF).EQ.0)) THEN
	  write(*,*) '##E03 Helicity filter could not be successfully double checked.'
	  write(*,*) 'One reason for this is that you have changed sensible parameters which affected the dependent helicity relationships.'
	  write(*,*) 'Please regenerate the filter by removing HelFilter.dat and rerunning SLOOPMATRIX over a couple of PS points.'
	  STOP
	ENDIF
C   SET HELDOUBLECHECKED TO .TRUE. WHEN DONE
	IF (HELPICKED.EQ.NCOMB) THEN
	  DONEHELDOUBLECHECK=.TRUE.
	ENDIF
  ENDIF

C GOTO NEXT HELICITY OR FINISH
  IF(HELPICKED.NE.NCOMB) THEN
    HELPICKED=HELPICKED+1
    goto 200
  ELSE
C   Useful printout
c    do I=1,NCOMB
c      write(*,*) 'HELSAVED(1,',I,')=',HELSAVED(1,I)
c      write(*,*) 'HELSAVED(2,',I,')=',HELSAVED(2,I)
c      write(*,*) 'HELSAVED(3,',I,')=',HELSAVED(3,I)
c      write(*,*) '   GOODHEL(',I,')=',GOODHEL(I)
c    ENDDO
    ANS(1)=BUFFR(1)
	ANS(2)=BUFFR(2)
	ANS(3)=BUFFR(3)
	IF(NTRY.EQ.0) THEN
	  NATTEMPTS=NATTEMPTS+1
	  IF(NATTEMPTS.EQ.MAXATTEMPTS) THEN
	    WRITE(*,*) '##E01 ERROR Could not initialize the filters in ',MAXATTEMPTS,' trials'
		STOP
	  ENDIF
	ENDIF
  ENDIF

ENDIF

DO K=1,3
  ANS(K)=ANS(K)/DBLE(IDEN)
  IF (USERHEL.NE.-1) THEN
    ANS(K)=ANS(K)*HELAVGFACTOR
  ENDIF
ENDDO

IF(.NOT.CHECKPHASE.AND.HELDOUBLECHECKED.AND.(CTMODERUN.EQ.-1)) THEN
  STAB_INDEX=STAB_INDEX+1  
  IF(DOING_QP_EVALS) THEN
    QP_RES(1,STAB_INDEX)=ANS(1)
    QP_RES(2,STAB_INDEX)=ANS(2)
    QP_RES(3,STAB_INDEX)=ANS(3)
  ELSE
    DP_RES(1,STAB_INDEX)=ANS(1)
    DP_RES(2,STAB_INDEX)=ANS(2)
    DP_RES(3,STAB_INDEX)=ANS(3)
  ENDIF

  IF(DOING_QP_EVALS) THEN	
      BASIC_CT_MODE=4
  ELSE
      BASIC_CT_MODE=1
  ENDIF

C BEGINNING OF THE DEFINITIONS OF THE DIFFERENT EVALUATION METHODS

  IF(.NOT.EVAL_DONE(2)) THEN
	EVAL_DONE(2)=.TRUE. 
	CTMODE=BASIC_CT_MODE+1
	goto 200
  ENDIF

  CTMODE=BASIC_CT_MODE
  
  IF(.NOT.EVAL_DONE(3).AND. ((DOING_QP_EVALS.AND.NRotations_QP.GE.1).OR.((.NOT.DOING_QP_EVALS).AND.NRotations_DP.GE.1)) ) THEN
	EVAL_DONE(3)=.TRUE.
	CALL ROTATE_PS(P_IN,P,1)
	goto 200
  ENDIF

  IF(.NOT.EVAL_DONE(4).AND. ((DOING_QP_EVALS.AND.NRotations_QP.GE.2).OR.((.NOT.DOING_QP_EVALS).AND.NRotations_DP.GE.2)) ) THEN
	EVAL_DONE(4)=.TRUE.
	CALL ROTATE_PS(P_IN,P,2)
	goto 200
  ENDIF

  CALL ROTATE_PS(P_IN,P,0)

C END OF THE DEFINITIONS OF THE DIFFERENT EVALUATION METHODS

  IF(DOING_QP_EVALS) THEN
    CALL COMPUTE_ACCURACY(QP_RES,N_QP_EVAL,ACC,ANS)
	IF(ACC.GE.MLSTABTHRES) THEN
	  NEPS=NEPS+1
      CALL COMPUTE_ACCURACY(DP_RES,N_DP_EVAL,TEMP1,TEMP)	  
      WRITE(*,*) '##W03 WARNING An unstable PS point was',       ' detected.'
	  WRITE(*,*) '(DP,QP) accuracies : (',TEMP1,',',ACC,')'
	  WRITE(*,*) 'Best estimate (fin,1eps,2eps) :',(ANS(I),I=1,3)
	  IF(NEPS.LE.10) THEN
	    WRITE(*,*) 'Double precision evaluations :',(DP_RES(1,I),I=1,N_DP_EVAL)
	    WRITE(*,*) 'Quad   precision evaluations :',(QP_RES(1,I),I=1,N_QP_EVAL)		 
	    WRITE(*,*) 'PS point specification :'
	    WRITE(*,*) 'Renormalization scale MU_R=',MU_R	
	    DO I=1,NEXTERNAL
          WRITE (*,'(i2,1x,4e27.17)') i, P(0,i),P(1,i),P(2,i),P(3,i) 
        ENDDO
	  ENDIF
    ENDIF
  ELSE
    CALL COMPUTE_ACCURACY(DP_RES,N_DP_EVAL,ACC,ANS)
	IF(ACC.GE.MLSTABTHRES) THEN
	  DOING_QP_EVALS=.TRUE.
	  EVAL_DONE(1)=.TRUE.
	  DO I=2,MAXSTABILITYLENGTH
        EVAL_DONE(I)=.FALSE.
      ENDDO
	  STAB_INDEX=0
	  CTMODE=4
	  goto 200
    ENDIF	
  ENDIF

ENDIF

 9999 CONTINUE
      END


	  logical function IsZero(toTest, reference_value, loop)
      IMPLICIT NONE
C  
C CONSTANTS
C
      INTEGER    NLOOPGROUPS
	  PARAMETER (NLOOPGROUPS=%(nloop_groups)d)
C  
C ARGUMENTS 
C  
	  %(real_dp_format)s toTest, reference_value
	  integer loop
C  
C GLOBAL 
C
      INCLUDE 'MadLoopParams.inc'
      %(complex_dp_format)s LOOPRES(3,NLOOPGROUPS)
	  LOGICAL S(NLOOPGROUPS)
      common/LOOPRES/LOOPRES,S
C ----------
C BEGIN CODE
C ----------
	  IF(abs(reference_value).eq.0.0d0) then
	    IsZero=.FALSE.
		write(*,*) '##E02 ERRROR Reference value for comparison is zero.'
		STOP
	  else
	    IsZero=((abs(toTest)/abs(reference_value)).lt.ZEROTHRES)
	  endif
      
	  IF(loop.NE.-1) THEN
	    IF((.NOT.ISZERO).AND.(.NOT.S(loop))) THEN
	      write(*,*) '##W01 WARNING Contribution ',loop,' is detected as contributing with CR=',(abs(toTest)/abs(reference_value)),' but is unstable.' 
	    ENDIF
	  ENDIF

	  end

      integer function ISSAME(resA,resB,ref)
	  IMPLICIT NONE
C     This function compares the result from two different helicity configuration A and B
C     It returns 0 if they are not related and (+/-wgt) if A=(+/-wgt)*B.
C     For now, the only wgt implemented is the integer 1.
C  
C CONSTANTS
C
      integer N_WGT_TO_TRY
	  parameter (N_WGT_TO_TRY=2)
C  
C ARGUMENTS 
C  
	  %(real_dp_format)s resA(3), resB(3)
	  %(real_dp_format)s ref
C  
C LOCAL VARIABLES
C
      LOGICAL ISZERO
      integer I,J
      integer WGT_TO_TRY(N_WGT_TO_TRY)
	  data WGT_TO_TRY/1,-1/
C ----------
C BEGIN CODE
C ----------
      ISSAME=0

	  DO I=1,N_WGT_TO_TRY
	    DO J=1,3
		  IF (IsZero(ABS(resB(J)),ref,-1)) THEN
		    IF(.NOT.IsZero(ABS(resB(J))+ABS(resA(J)),ref,-1)) THEN
		      GOTO 1231
			ENDIF
C         Be loser for helicity comparison, so bring a factor 100
		  ELSEIF(.NOT.IsZero((resA(J)/resB(J))-DBLE(WGT_TO_TRY(I)),ref*100.0d0,-1)) THEN
		    GOTO 1231		    
		  ENDIF
		ENDDO
		ISSAME = WGT_TO_TRY(I)
		RETURN
 1231   CONTINUE
      ENDDO
	  END

	  SUBROUTINE IMPROVE_PS_POINT_PRECISION(P)
C  
C CONSTANTS 
C 
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
	  %(real_dp_format)s     ZERO
	  PARAMETER (ZERO=0.0d0)
C  
C ARGUMENTS 
C
      %(real_dp_format)s P(0:3,NEXTERNAL)
      %(real_dp_format)s PTOT(0:3,NEXTERNAL)	  
C
C LOCAL VARIABLES 
C
      INTEGER I,J
	  %(real_dp_format)s BUFF,REF
	  %(real_dp_format)s MASSES(NEXTERNAL-1)
	  %(real_dp_format)s SHIFTX(2),SHIFTY(2)
C  
C GLOBAL VARIABLES
C 
      include 'coupl.inc'

	  %(masses_def)s

	  DO I=0,3
	    REF=ZERO
		BUFF=ZERO
		DO J=1,NEXTERNAL
		  REF=REF+ABS(P(I,J))
		ENDDO
        DO J=1,%(n_initial)d		
		  BUFF=BUFF-P(I,J)
		ENDDO
        DO J=%(n_initial)d+1,NEXTERNAL	
		  BUFF=BUFF+P(I,J)
		ENDDO
		IF ((BUFF/REF).GT.1.0d-6) THEN
		  WRITE(*,*) 'ERROR:: Four-momentum conservation is not accurate enough, ',(BUFF/REF)
		  CALL WRITE_MOM(P)
		  STOP
		ENDIF
	  ENDDO

	  DO I=1,NEXTERNAL-2
	    REF=ABS(P(0,I))+ABS(P(1,I))+ABS(P(2,I))+ABS(P(3,I))
		IF ((SQRT(ABS(P(0,I)**2-P(1,I)**2-P(2,I)**2-P(3,I)**2-MASSES(I)**2))/REF).GT.1.0d-2) THEN
		  WRITE(*,*) 'ERROR:: Four-momentum conservation for particle ',I,' is not accurate enough, ', (SQRT(ABS(P(0,I)**2-P(1,I)**2-P(2,I)**2-P(3,I)**2-MASSES(I)**2))/REF)
		  CALL WRITE_MOM(P)
		  STOP
		ELSE
	      P(3,I)=SIGN(SQRT(ABS(P(0,I)**2-P(1,I)**2-P(2,I)**2-MASSES(I)**2)),P(3,I))
		ENDIF
	  ENDDO

	  DO I=0,3
	    BUFF=0.0d0 
        DO J=1,%(n_initial)d
		  BUFF=BUFF-P(I,J)
		ENDDO
		DO J=%(n_initial)d+1,NEXTERNAL-2
		  BUFF=BUFF+P(I,J)		
		ENDDO
		PTOT(I)=BUFF
	  ENDDO

	  END

	  SUBROUTINE WRITE_MOM(P)
	  IMPLICIT NONE
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      double precision dot

	  INTEGER I
C  
C ARGUMENTS 
C
      %(real_dp_format)s P(0:3,NEXTERNAL)
      write (*,*) " Phase space point:"
      write (*,*) "-----------------------------------------------------------------------------"
      write (*,*)  "n        E             px             py              pz               m "
      do i=1,nexternal
         write (*,'(1x,5e25.15)') i, P(0,i),P(1,i),P(2,i),P(3,i),dsqrt(dabs(DOT(p(0,i),p(0,i))))
      enddo
      write (*,*) "-----------------------------------------------------------------------------"
	  END

      double precision function dot(p1,p2)
C ****************************************************************************
C     4-Vector Dot product
C ****************************************************************************
      implicit none
      %(real_dp_format)s p1(0:3),p2(0:3)
      dot=p1(0)*p2(0)-p1(1)*p2(1)-p1(2)*p2(2)-p1(3)*p2(3)
      return
      end

C Rotate_PS rotates the PS point P_IN (without modifying it)
C stores the result in P and modifies the global variables
C PS and MP_DONE accordingly.

	  SUBROUTINE ROTATE_PS(P_IN,P,ROTATION)
C  
C CONSTANTS 
C 
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
C  
C ARGUMENTS 
C
      %(real_dp_format)s P_IN(0:3,NEXTERNAL),P(0:3,NEXTERNAL)
	  INTEGER ROTATION
C
C LOCAL VARIABLES 
C
      INTEGER I,J
C  
C GLOBAL VARIABLES
C 
	  LOGICAL MP_DONE
	  common/MP_DONE/MP_DONE

	  %(real_dp_format)s PS(0:3,NEXTERNAL)
	  common/PSPOINT/PS

C ----------
C BEGIN CODE
C ----------

      DO I=1,NEXTERNAL
C rotation=1 => (xp=z,yp=-x,zp=-y)
		IF(ROTATION.EQ.1) THEN
		  P(0,I)=P_IN(0,I)
		  P(1,I)=P_IN(3,I)
		  P(2,I)=-P_IN(1,I)
		  P(3,I)=-P_IN(2,I)
C rotation=2 => (xp=-z,yp=y,zp=x)
		ELSEIF(ROTATION.EQ.2) THEN
		  P(0,I)=P_IN(0,I)
		  P(1,I)=-P_IN(3,I)
		  P(2,I)=P_IN(2,I)
		  P(3,I)=P_IN(1,I)
		ELSE
		  P(0,I)=P_IN(0,I)
		  P(1,I)=P_IN(1,I)
		  P(2,I)=P_IN(2,I)
		  P(3,I)=P_IN(3,I)
		ENDIF
		DO J=0,3
		  PS(J,I)=P(J,I)
		ENDDO
	  ENDDO

	  MP_DONE = .FALSE.

	  END

      SUBROUTINE compute_accuracy(fulllist, length, acc, estimate)
      implicit none
C  
C PARAMETERS 
C
      integer maxstabilitylength
	  common/stability_tests/maxstabilitylength
C  
C ARGUMENTS 
C
      real*8 fulllist(3,maxstabilitylength)
      integer length
      real*8 acc, estimate(3)
C  
C LOCAL VARIABLES 
C
      logical mask(maxstabilitylength)
	  logical mask3(3)
	  data mask3/.TRUE.,.TRUE.,.TRUE./
      integer i,j
      real*8 avg
      real*8 diff
	  real*8 accuracies(3)
	  real*8 list(maxstabilitylength)

C ----------
C BEGIN CODE
C ----------
      do i=1,length
        mask(i)=.TRUE.
      enddo
      do i=length+1,maxstabilitylength
        mask(i)=.FALSE.      
      enddo

	  do i=1,3
	    do j=1,maxstabilitylength
		  list(j)=fulllist(i,j)
		enddo
        diff=maxval(list,1,mask)-minval(list,1,mask)
        avg=(maxval(list,1,mask)+minval(list,1,mask))/2.0d0
		estimate(i)=avg
        if (avg.eq.0.0d0) then
          accuracies(i)=diff
        else
          accuracies(i)=diff/abs(avg)
        endif
	  enddo

	  acc=maxval(accuracies,1,mask3)

      end

      SUBROUTINE SET_N_EVALS(N_DP_EVALS,N_QP_EVALS)
 
	  IMPLICIT NONE
	  INTEGER N_DP_EVALS, N_QP_EVALS

	  include 'MadLoopParams.inc'

	  IF(CTMODERUN.LE.-1) THEN
	    N_DP_EVALS=2+NRotations_DP
	    N_QP_EVALS=2+NRotations_QP
	  ELSE
	  	N_DP_EVALS=1
	    N_QP_EVALS=1
	  ENDIF

	  IF(N_DP_EVALS.GT.20.OR.N_QP_EVALS.GT.20) THEN
	    WRITE(*,*) 'ERROR:: Increase hardcoded maxstabilitylength.'
		STOP
	  ENDIF

	  END
