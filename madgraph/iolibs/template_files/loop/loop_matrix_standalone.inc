      SUBROUTINE %(proc_prefix)sSLOOPMATRIXHEL(P,HEL,ANS)
      IMPLICIT NONE
C  
C CONSTANTS
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)

      INCLUDE "nsquaredSO.inc"

C  
C ARGUMENTS 
C  
      %(real_dp_format)s P(0:3,NEXTERNAL)
      %(real_dp_format)s ANS(0:3,0:NSQUAREDSO)
	  INTEGER HEL, USERHEL
	  common/%(proc_prefix)sUSERCHOICE/USERHEL
C ----------
C BEGIN CODE
C ----------
	  USERHEL=HEL
      CALL %(proc_prefix)sSLOOPMATRIX(P,ANS)
	  END

	  logical function %(proc_prefix)sIsZero(toTest, reference_value, ampLn)
      IMPLICIT NONE
C  
C CONSTANTS
C
      INTEGER    NLOOPAMPS
	  PARAMETER (NLOOPAMPS=%(nloopamps)d)
C  
C ARGUMENTS 
C  
	  %(real_dp_format)s toTest, reference_value
	  integer ampLn
C  
C GLOBAL 
C
	  include 'MadLoopParams.inc'

      %(complex_dp_format)s AMPL(3,NLOOPAMPS)
	  LOGICAL S(NLOOPAMPS)
	  common/%(proc_prefix)sAMPL/AMPL,S
C ----------
C BEGIN CODE
C ----------
	  IF(abs(reference_value).eq.0.0d0) then
	    %(proc_prefix)sIsZero=.FALSE.
		write(*,*) '##E02 ERRROR Reference value for comparison is zero.'
		STOP
	  else
	    %(proc_prefix)sIsZero=((abs(toTest)/abs(reference_value)).lt.ZeroThres)
	  endif
      IF(AMPLN.NE.-1) THEN 
	    IF((.NOT.%(proc_prefix)sISZERO).AND.(.NOT.S(AMPLN))) THEN
	      write(*,*) '##W01 WARNING Contribution ',ampLn,' is detected as contributing with CR=',(abs(toTest)/abs(reference_value)),' but is unstable.' 
	    ENDIF
	  ENDIF

	  end

      SUBROUTINE %(proc_prefix)sSLOOPMATRIX(P_USER,ANSRETURNED)
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
      CHARACTER*512 paramFName,HelConfigFName,LoopFilterFName
	  CHARACTER*512 colorNumFName,colorDenomFName, HelFilterFName
	  CHARACTER*512 Proc_Prefix
      PARAMETER ( paramFName='MadLoopParams.dat')
      PARAMETER ( HelConfigFName='HelConfigs.dat')
      PARAMETER ( LoopFilterFName='LoopFilter.dat')
      PARAMETER ( HelFilterFName='HelFilter.dat')
      PARAMETER ( ColorNumFName='ColorNumFactors.dat')
      PARAMETER ( ColorDenomFName='ColorDenomFactors.dat')
      PARAMETER ( Proc_Prefix='%(proc_prefix)s')

	  %(nbornamps_decl)s
      INTEGER    NLOOPAMPS, NCTAMPS
      PARAMETER (NLOOPAMPS=%(nloopamps)d, NCTAMPS=%(nctamps)d) 
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d)
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
      INCLUDE "nsquaredSO.inc"
	  INTEGER NSQUAREDSOP1
	  PARAMETER (NSQUAREDSOP1=NSQUAREDSO+1)
	  INTEGER MAXSTABILITYLENGTH
	  DATA MAXSTABILITYLENGTH/20/
	  common/%(proc_prefix)sstability_tests/maxstabilitylength	  
C  
C ARGUMENTS 
C  
      %(real_dp_format)s P_USER(0:3,NEXTERNAL)
      %(real_dp_format)s ANSRETURNED(0:3,0:NSQUAREDSO)
C  
C LOCAL VARIABLES 
C  
      %(real_dp_format)s ANS(0:3)
      INTEGER I,J,K,H

      CHARACTER*512 ParamFN,HelConfigFN,LoopFilterFN,ColorNumFN,ColorDenomFN,HelFilterFN
	  CHARACTER*512 TMP
	  SAVE ParamFN
	  SAVE HelConfigFN
	  SAVE LoopFilterFN
	  SAVE ColorNumFN
	  SAVE ColorDenomFN
	  SAVE HelFilterFN

	  INTEGER HELPICKED_BU, CTMODEINIT_BU
	  %(real_dp_format)s MLSTABTHRES_BU
C P is the actual PS POINT used for the computation, and can be rotated for the stability test purposes.
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
c     This is used for loop-induced where the reference scale for comparisons is infered from
c     the previous points
	  %(real_dp_format)s NEXTREF
	  DATA NEXTREF/ZERO/
	  INTEGER NPSPOINTS
	  DATA NPSPOINTS/0/
      LOGICAL FOUND_VALID_REDUCTION_METHOD
      DATA FOUND_VALID_REDUCTION_METHOD/.FALSE./

	  %(real_dp_format)s ACC
	  %(real_dp_format)s DP_RES(3,MAXSTABILITYLENGTH)
C QP_RES STORES THE QUADRUPLE PRECISION RESULT OBTAINED FROM DIFFERENT EVALUATION METHODS IN ORDER TO ASSESS STABILITY.
	  %(real_dp_format)s QP_RES(3,MAXSTABILITYLENGTH)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
	  INTEGER NATTEMPTS
	  DATA NATTEMPTS/0/
	  DATA IC/NEXTERNAL*1/
	  %(real_dp_format)s BUFFR(3),TEMP(3),TEMP1,TEMP2
      %(complex_dp_format)s CFTOT
	  LOGICAL FOUNDHELFILTER,FOUNDLOOPFILTER
	  DATA FOUNDHELFILTER/.TRUE./
	  DATA FOUNDLOOPFILTER/.TRUE./
	  INTEGER IDEN
      %(den_factor_line)s
	  INTEGER HELAVGFACTOR
	  DATA HELAVGFACTOR/%(hel_avg_factor)d/
      LOGICAL DONEHELDOUBLECHECK
      DATA DONEHELDOUBLECHECK/.FALSE./	  
	  INTEGER NEPS
	  DATA NEPS/0/
C     Below are variables to bypass the checkphase and insure stability check to take place
      LOGICAL OLD_CHECKPHASE, OLD_HELDOUBLECHECKED
	  LOGICAL OLD_GOODHEL(NCOMB)
	  LOGICAL OLD_GOODAMP(NLOOPAMPS,NCOMB)

	  LOGICAL BYPASS_CHECK, ALWAYS_TEST_STABILITY
	  COMMON/%(proc_prefix)sBYPASS_CHECK/BYPASS_CHECK, ALWAYS_TEST_STABILITY 
C
C FUNCTIONS
C
      LOGICAL %(proc_prefix)sISZERO
C  
C GLOBAL VARIABLES
C 
      include 'process_info.inc'
      include 'coupl.inc'
	  include 'mp_coupl.inc'
	  include 'MadLoopParams.inc'

	  INTEGER NTRY
      DATA NTRY/0/
	  LOGICAL CHECKPHASE
	  DATA CHECKPHASE/.TRUE./
	  LOGICAL HELDOUBLECHECKED
	  DATA HELDOUBLECHECKED/.FALSE./
	  %(real_dp_format)s REF
	  DATA REF/0.0d0/
      common/%(proc_prefix)sINIT/NTRY,CHECKPHASE,HELDOUBLECHECKED,REF

C     THE LOGICAL BELOWS ARE JUST TO KEEP TRACK OF WHETHER THE MP_PS HAS BEEN SET YET OR NOT AND WHETER THE MP EXTERNAL WFS HAVE BEEN COMPUTED YET.
	  LOGICAL MP_DONE
	  DATA MP_DONE/.FALSE./
	  common/%(proc_prefix)sMP_DONE/MP_DONE
	  LOGICAL MP_PS_SET
	  DATA MP_PS_SET/.FALSE./
	  common/%(proc_prefix)sMP_PS_SET/MP_PS_SET

C     PS CAN POSSIBILY BE PASSED THROUGH IMPROVE_PS BUT IS NOT MODIFIED FOR THE PURPOSE OF THE STABILITY TEST	  
C     EVEN THOUGH THEY ARE PUT IN COMMON BLOCK, FOR NOW THEY ARE NOT USED ANYWHERE ELSE
	  %(real_dp_format)s PS(0:3,NEXTERNAL)
	  common/%(proc_prefix)sPSPOINT/PS
C     AGAIN BELOW, MP_PS IS THE FIXED (POSSIBLY IMPROVED) MP PS POINT AND MP_P IS THE ONE WHICH CAN BE MODIFIED (I.E. ROTATED ETC.) FOR STABILITY PURPOSE
C     EVEN THOUGH THEY ARE PUT IN COMMON BLOCK, FOR NOW THEY ARE NOT USED ANYWHERE ELSE THAN HERE AND SET_MP_PS()
      %(real_mp_format)s MP_PS(0:3,NEXTERNAL),MP_P(0:3,NEXTERNAL)
	  common/%(proc_prefix)sMP_PSPOINT/MP_PS,MP_P

	  %(real_dp_format)s LSCALE
	  INTEGER CTMODE	  
      common/%(proc_prefix)sCT/LSCALE,CTMODE

	  LOGICAL GOODHEL(NCOMB)
	  LOGICAL GOODAMP(NLOOPAMPS,NCOMB)
	  common/%(proc_prefix)sFilters/GOODAMP,GOODHEL

	  INTEGER HELPICKED
	  DATA HELPICKED/-1/
	  common/%(proc_prefix)sHELCHOICE/HELPICKED
	  INTEGER USERHEL
	  DATA USERHEL/-1/
	  common/%(proc_prefix)sUSERCHOICE/USERHEL

      %(dp_born_amps_decl)s	  
	  %(complex_dp_format)s W(20,NWAVEFUNCS%(ncomb_helas_objs)s)
	  INTEGER VALIDH
	  common/%(proc_prefix)sWFCTS/W  
	  common/%(proc_prefix)sVALIDH/VALIDH

      %(complex_dp_format)s AMPL(3,NLOOPAMPS)
	  LOGICAL S(NLOOPAMPS)
	  common/%(proc_prefix)sAMPL/AMPL,S

	  INTEGER CF_D(NLOOPAMPS,%(color_matrix_size)s)
	  INTEGER CF_N(NLOOPAMPS,%(color_matrix_size)s)
	  common/%(proc_prefix)sCF/CF_D,CF_N

	  INTEGER HELC(NEXTERNAL,NCOMB)
	  common/%(proc_prefix)sHELCONFIGS/HELC

	  %(real_dp_format)s PREC,USER_STAB_PREC
	  DATA USER_STAB_PREC/-1.0d0/	  
	  common/%(proc_prefix)sUSER_STAB_PREC/USER_STAB_PREC

C     Return codes H,T,U correspond to the hundreds, tens and units
C     building returncode, i.e.
C     RETURNCODE=100*RET_CODE_H+10*RET_CODE_T+RET_CODE_U
	  
      INTEGER RET_CODE_H,RET_CODE_T,RET_CODE_U
	  %(real_dp_format)s ACCURACY(0:NSQUAREDSO)
	  DATA (ACCURACY(I),I=0,NSQUAREDSO)/NSQUAREDSOP1*1.0d0/
	  DATA RET_CODE_H,RET_CODE_T,RET_CODE_U/1,1,0/
	  common/%(proc_prefix)sACC/ACCURACY,RET_CODE_H,RET_CODE_T,RET_CODE_U

C     Allows to forbid the zero helicity double check, no matter the value in MadLoopParams.dat
C     This can be accessed with the SET_FORBID_HEL_DOUBLECHECK subroutine of MadLoopCommons.dat
      LOGICAL FORBID_HEL_DOUBLECHECK
	  COMMON/FORBID_HEL_DOUBLECHECK/FORBID_HEL_DOUBLECHECK

	  LOGICAL MP_DONE_ONCE
	  DATA MP_DONE_ONCE/.FALSE./
	  common/%(proc_prefix)sMP_DONE_ONCE/MP_DONE_ONCE

	  character(512) MLPath
      common/MLPATH/MLPath

	  LOGICAL ML_INIT
	  common/ML_INIT/ML_INIT

C     This variable controls the *local* initialization of this particular SubProcess.
C     For example, the reading of the filters must be done independently by each SubProcess.
      LOGICAL LOCAL_ML_INIT
	  data LOCAL_ML_INIT/.TRUE./

C     Variables related to turning off the Lorentz rotation test when spin-2 particles are external
      LOGICAL WARNED_LORENTZ_STAB_TEST_OFF
	  data WARNED_LORENTZ_STAB_TEST_OFF/.FALSE./
	  INTEGER NROTATIONS_DP_BU,NROTATIONS_QP_BU

C ----------
C BEGIN CODE
C ----------

IF(ML_INIT) THEN
  CALL PRINT_MADLOOP_BANNER()
  TMP = 'auto'
  CALL SETMADLOOPPATH(TMP)
  CALL JOINPATH(MLPATH,PARAMFNAME,PARAMFN)
  CALL MADLOOPPARAMREADER(PARAMFN,.TRUE.)
  IF (FORBID_HEL_DOUBLECHECK) THEN
    DoubleCheckHelicityFilter = .False. 
  ENDIF
  ML_INIT = .FALSE.
C For now only CutTools is interfaced in the default mode. Samurai could follow.
  DO I=1,SIZE(MLReductionLib)
    if (MLReductionLib(I).eq.1) then
	  FOUND_VALID_REDUCTION_METHOD = .TRUE.
	endif
  ENDDO
  if (.not.FOUND_VALID_REDUCTION_METHOD) THEN
    WRITE(*,*) 'ERROR:: For now, only CutTools is interfaced to MadLoop in the non-optimized output.'
    WRITE(*,*) 'ERROR:: Make sure to include 1 in the parameter MLReductionLib of the card MadLoopParams.dat'
	STOP 1
  ENDIF
ENDIF
IF (LOCAL_ML_INIT) THEN
C Setup the file paths
  CALL JOINPATH(MLPATH,PARAMFNAME,PARAMFN)
  CALL JOINPATH(MLPATH,PROC_PREFIX,TMP)
  CALL JOINPATH(TMP,HELCONFIGFNAME,HELCONFIGFN)
  CALL JOINPATH(TMP,LOOPFILTERFNAME,LOOPFILTERFN)
  CALL JOINPATH(TMP,COLORNUMFNAME,COLORNUMFN)
  CALL JOINPATH(TMP,COLORDENOMFNAME,COLORDENOMFN)
  CALL JOINPATH(TMP,HELFILTERFNAME,HELFILTERFN)

C Make sure that the loop filter is disabled when there is spin-2 particles for 2>1 or 1>2 processes
  if(MAX_SPIN_EXTERNAL_PARTICLE.gt.3.AND.(NEXTERNAL.LE.3.AND.HelicityFilterLevel.NE.0)) THEN
    WRITE(*,*) '##INFO: Helicity filter deactivated for 2>1 processes involving spin 2 particles.'
    HelicityFilterLevel = 0	
C   We write a dummy filter for structural reasons here
    OPEN(1, FILE=HelFilterFN, err=6116, status='NEW',action='WRITE')
    DO I=1,NCOMB
      WRITE(1,*) 'T' 
    ENDDO
6116  CONTINUE
    CLOSE(1)
  ENDIF

OPEN(1, FILE=ColorNumFN, err=104, status='OLD',           action='READ')
  DO I=1,NLOOPAMPS
    READ(1,*,END=105) (CF_N(I,J),J=1,%(color_matrix_size)s)
  ENDDO
  GOTO 105
104  CONTINUE
  STOP 'Color factors could not be initialized from file %(proc_prefix)sColorNumFactors.dat. File not found' 
105  CONTINUE
CLOSE(1)
OPEN(1, FILE=ColorDenomFN, err=106, status='OLD',           action='READ')
  DO I=1,NLOOPAMPS
    READ(1,*,END=107) (CF_D(I,J),J=1,%(color_matrix_size)s)
  ENDDO
  GOTO 107
106  CONTINUE
  STOP 'Color factors could not be initialized from file %(proc_prefix)sColorDenomFactors.dat. File not found' 
107  CONTINUE
CLOSE(1)
OPEN(1, FILE=HelConfigFN, err=108, status='OLD',                  action='READ')
  DO H=1,NCOMB
    READ(1,*,END=109) (HELC(I,H),I=1,NEXTERNAL)
  ENDDO
  GOTO 109
108  CONTINUE
  STOP 'Color helictiy configurations could not be initialized from file %(proc_prefix)sHelConfigs.dat. File not found' 
109  CONTINUE
CLOSE(1)
IF(BOOTANDSTOP) THEN
  WRITE(*,*) '##Stopped by user request.'
  STOP
ENDIF
LOCAL_ML_INIT = .FALSE.
ENDIF

C Make sure that lorentz rotation tests are not used if there is external loop wavefunction of spin 2 and that one specific helicity is asked
NROTATIONS_DP_BU = NROTATIONS_DP
NROTATIONS_QP_BU = NROTATIONS_QP
if(MAX_SPIN_EXTERNAL_PARTICLE.gt.3.AND.USERHEL.NE.-1) THEN
   if(.NOT.WARNED_LORENTZ_STAB_TEST_OFF) THEN
     WRITE(*,*) '##WARNING: Evaluation of a specific helicity was asked for this PS point, and there is a spin-2 (or higher) particle in the external states.'
	 WRITE(*,*) '##WARNING: As a result, MadLoop disabled the Lorentz rotation test for this phase-space point only.'
	 WRITE(*,*) '##WARNING: Further warning of that type suppressed.'
     WARNED_LORENTZ_STAB_TEST_OFF = .FALSE. 
   ENDIF
   NROTATIONS_QP=0
   NROTATIONS_DP=0
ENDIF

IF(NTRY.EQ.0) THEN
  CALL %(proc_prefix)sSET_N_EVALS(N_DP_EVAL,N_QP_EVAL)
  HELDOUBLECHECKED=(.NOT.DoubleCheckHelicityFilter).OR.(HelicityFilterLevel.eq.0)
  DO J=1,NCOMB
    DO I=1,NCTAMPS
	  GOODAMP(I,J)=.TRUE.
	ENDDO
  ENDDO
OPEN(1, FILE=LoopFilterFN, err=100, status='OLD',           action='READ')
  DO J=1,NCOMB
    READ(1,*,END=101) (GOODAMP(I,J),I=NCTAMPS+1,NLOOPAMPS)
  ENDDO
  GOTO 101
100  CONTINUE
  FOUNDLOOPFILTER=.FALSE.
  DO J=1,NCOMB
    DO I=NCTAMPS+1,NLOOPAMPS
	  GOODAMP(I,J)=(.NOT.USELOOPFILTER)
	ENDDO
  ENDDO
101  CONTINUE
CLOSE(1)
IF (HelicityFilterLevel.eq.0) then
  FOUNDHELFILTER=.TRUE.
  DO J=1,NCOMB
	GOODHEL(J)=.TRUE.
  ENDDO
  GOTO 122
ENDIF
OPEN(1, FILE=HelFilterFN, err=102, status='OLD',           action='READ')
  READ(1,*,END=103) (GOODHEL(I),I=1,NCOMB)
  GOTO 103
102  CONTINUE
  FOUNDHELFILTER=.FALSE.
  DO J=1,NCOMB
	GOODHEL(J)=.TRUE.
  ENDDO
103  CONTINUE
CLOSE(1)
122  CONTINUE
ENDIF

MP_DONE=.FALSE.
MP_DONE_ONCE=.FALSE.
MP_PS_SET=.FALSE.
STAB_INDEX=0
DOING_QP_EVALS=.FALSE.
EVAL_DONE(1)=.TRUE.
DO I=2,MAXSTABILITYLENGTH
  EVAL_DONE(I)=.FALSE.
ENDDO

%(compute_born)s

IF (USER_STAB_PREC.GT.0.0d0) THEN
  MLSTABTHRES_BU=MLSTABTHRES
  MLSTABTHRES=USER_STAB_PREC
C In the initialization, I cannot perform stability test and therefore guarantee any precision
  CTMODEINIT_BU=CTMODEINIT
C So either one choses quad precision directly
C  CTMODEINIT=4
C Or, because this is very slow, we keep the orignal value. The accuracy returned is -1 and tells the MC that he should not trust the evaluation for checks.
  CTMODEINIT=CTMODEINIT_BU
ENDIF

IF(.NOT.BYPASS_CHECK) THEN
  NTRY=NTRY+1
ENDIF

IF(DONEHELDOUBLECHECK.AND.(.NOT.HELDOUBLECHECKED)) THEN
  HELDOUBLECHECKED=.TRUE.
  DONEHELDOUBLECHECK=.FALSE.
ENDIF

CHECKPHASE=(NTRY.LE.CHECKCYCLE).AND.(((.NOT.FOUNDLOOPFILTER).AND.USELOOPFILTER).OR.(.NOT.FOUNDHELFILTER))

IF (WriteOutFilters) THEN
IF ((.NOT. CHECKPHASE).AND.(.NOT.FOUNDHELFILTER)) THEN
OPEN(1, FILE=HelFilterFN, err=110, status='NEW',action='WRITE')
  WRITE(1,*) (GOODHEL(I),I=1,NCOMB)
110  CONTINUE
  CLOSE(1)
FOUNDHELFILTER=.TRUE.
ENDIF

IF ((.NOT. CHECKPHASE).AND.(.NOT.FOUNDLOOPFILTER).AND.USELOOPFILTER) THEN
OPEN(1, FILE=LoopFilterFN, err=111, status='NEW',action='WRITE')
  DO J=1,NCOMB
    WRITE(1,*) (GOODAMP(I,J),I=NCTAMPS+1,NLOOPAMPS)
  ENDDO
111  CONTINUE
  CLOSE(1)
FOUNDLOOPFILTER=.TRUE.
ENDIF
ENDIF

IF (BYPASS_CHECK) THEN
  OLD_CHECKPHASE = CHECKPHASE
  OLD_HELDOUBLECHECKED = HELDOUBLECHECKED
  CHECKPHASE = .FALSE.
  HELDOUBLECHECKED = .TRUE.
  DO I=1,NCOMB
    OLD_GOODHEL(I)=GOODHEL(I)
	GOODHEL(I) = .TRUE.
  ENDDO
  DO I=1,NCOMB
    DO J=1,NLOOPAMPS
      OLD_GOODAMP(J,I)=GOODAMP(J,I)
	  GOODAMP(J,I) = .TRUE.
	ENDDO
  ENDDO
ENDIF

IF(CHECKPHASE.OR.(.NOT.HELDOUBLECHECKED)) THEN
  HELPICKED=1
  CTMODE=CTMODEINIT
ELSE
  IF (USERHEL.ne.-1) THEN
    IF(.NOT.GOODHEL(USERHEL)) THEN
      ANS(1)=0.0d0
      ANS(2)=0.0d0
      ANS(3)=0.0d0
      goto 9999
    ENDIF
  ENDIF
  HELPICKED=USERHEL      
  IF (CTMODERUN.GT.-1) THEN
    CTMODE=CTMODERUN
  ELSE
    CTMODE=1
  ENDIF
ENDIF

DO I=1,NEXTERNAL
  DO J=0,3
    PS(J,I)=P_USER(J,I)
  ENDDO
ENDDO

IF (ImprovePSPoint.ge.0) THEN
C Make the input PS more precise (exact onshell and energy-momentum conservation)
  CALL %(proc_prefix)sIMPROVE_PS_POINT_PRECISION(PS)
ENDIF

DO I=1,NEXTERNAL
  DO J=0,3
    P(J,I)=PS(J,I)
  ENDDO
ENDDO

DO K=1, 3
  BUFFR(K)=0.0d0
  DO I=1,NLOOPAMPS
    AMPL(K,I)=(0.0d0,0.0d0)
  ENDDO
ENDDO

LSCALE=DSQRT(ABS((P(0,1)+P(0,2))**2-(P(1,1)+P(1,2))**2-(P(2,1)+P(2,2))**2-(P(3,1)+P(3,2))**2))

%(set_reference)s

200 CONTINUE

IF (CTMODE.EQ.0.OR.CTMODE.GE.4) THEN
  CALL MP_UPDATE_AS_PARAM()
ENDIF

IF (.NOT.MP_PS_SET.AND.(CTMODE.EQ.0.OR.CTMODE.GE.4)) THEN
  CALL %(proc_prefix)sSET_MP_PS(P_USER)
  MP_PS_SET = .TRUE.
ENDIF

DO K=1,3
  ANS(K)=0.0d0
ENDDO

VALIDH=-1
DO H=1,NCOMB
  IF ((HELPICKED.EQ.H).OR.((HELPICKED.EQ.-1).AND.(CHECKPHASE.OR.(.NOT.HELDOUBLECHECKED).OR.GOODHEL(H)))) THEN
  IF (VALIDH.EQ.-1) VALIDH=H
  DO I=1,NEXTERNAL
    NHEL(I)=HELC(I,H)
  ENDDO
C Check if we are in multiple precision and compute wfs and amps accordingly if needed   
  IF (CTMODE.GE.4) THEN
C   Force that only current helicity is used in the routine below
C   This should always be done, even if MP_DONE is True
C   because the AMPL of the R2 MUST be recomputed for loop induced.
C   (because they are not saved for each hel configuration)
C   (This is not optimal unlike what is done int the loop optimized output)
	HELPICKED_BU = HELPICKED
    HELPICKED = H
    CALL %(proc_prefix)sMP_BORN_AMPS_AND_WFS(MP_P)
    HELPICKED = HELPICKED_BU
    GOTO 300	
  ENDIF
  %(born_ct_helas_calls)s
300 continue
  %(loop_induced_setup)s  
  %(loop_induced_helas_calls)s
  %(loop_induced_finalize)s
  DO I=1,%(nctamps_or_nloopamps)s
    DO J=1,%(nbornamps_or_nloopamps)s
	  CFTOT=DCMPLX(CF_N(I,J)/DBLE(ABS(CF_D(I,J))),0.0d0)
      IF(CF_D(I,J).LT.0) CFTOT=CFTOT*IMAG1
	  %(squaring)s
    ENDDO
  ENDDO
  ENDIF
ENDDO

C WHEN CTMODE IS >=4, then the MP computation of wfs and amps is automatically done.
IF (CTMODE.GE.4) THEN
  MP_DONE = .TRUE.
ENDIF

IF(SKIPLOOPEVAL) THEN
  GOTO 1226
ENDIF

%(loop_helas_calls)s

%(actualize_ans)s

1226 CONTINUE

IF (CHECKPHASE.OR.(.NOT.HELDOUBLECHECKED)) THEN
C  Update of NEXTREF, will be used for loop induced only.
  NEXTREF = NEXTREF + ANS(1) + ANS(2) + ANS(3)
  IF((USERHEL.EQ.-1).OR.(USERHEL.EQ.HELPICKED)) THEN
  BUFFR(1)=BUFFR(1)+ANS(1)
  BUFFR(2)=BUFFR(2)+ANS(2)
  BUFFR(3)=BUFFR(3)+ANS(3)
  ENDIF

  IF (CHECKPHASE) THEN
C   SET THE HELICITY FILTER
    IF(.NOT.FOUNDHELFILTER) THEN
      IF(%(proc_prefix)sISZERO(ABS(ANS(1))+ABS(ANS(2))+ABS(ANS(3)),REF/DBLE(NCOMB),-1)) THEN
        IF(NTRY.EQ.1) THEN
	      GOODHEL(HELPICKED)=.FALSE.
	    ELSEIF(GOODHEL(HELPICKED)) THEN
		  WRITE(*,*) '##W02A WARNING Inconsistent helicity ',HELPICKED
		  IF(HELINITSTARTOVER) THEN
	        WRITE(*,*) '##I01 INFO Initialization starting over because of inconsistency in the helicity filter setup.'
	        NTRY=0
		  ENDIF
	    ENDIF
      ELSE
	    IF(.NOT.GOODHEL(HELPICKED)) THEN
		  WRITE(*,*) '##W02B WARNING Inconsistent helicity ',HELPICKED
		  IF(HELINITSTARTOVER) THEN
	        WRITE(*,*) '##I01 INFO Initialization starting over because of inconsistency in the helicity filter setup.'
	        NTRY=0
		  ELSE
		    GOODHEL(HELPICKED)=.TRUE.
		  ENDIF
		ENDIF
	  ENDIF
    ENDIF

C   SET THE LOOP FILTER
    IF(.NOT.FOUNDLOOPFILTER.AND.USELOOPFILTER) THEN
  	  DO I=NCTAMPS+1,NLOOPAMPS
        IF(.NOT.%(proc_prefix)sISZERO(ABS(AMPL(1,I))+ABS(AMPL(2,I))+ABS(AMPL(3,I)),(REF*1.0d-4),I)) THEN
          IF(NTRY.EQ.1) THEN
	        GOODAMP(I,HELPICKED)=.TRUE.
	      ELSEIF(.NOT.GOODAMP(I,HELPICKED)) THEN
	        WRITE(*,*) '##W02 WARNING Inconsistent loop amp ',I,' for helicity ',HELPICKED,'.'
		    IF(LOOPINITSTARTOVER) THEN
		      WRITE(*,*) '##I01 INFO Initialization starting over because of inconsistency in the loop filter setup.'
	          NTRY=0
		    ELSE
		      GOODAMP(I,HELPICKED)=.TRUE.
		    ENDIF
	      ENDIF
        ENDIF
  	  ENDDO
    ENDIF
  ELSEIF (.NOT.HELDOUBLECHECKED)THEN
    IF ((.NOT.GOODHEL(HELPICKED)).AND.(.NOT.%(proc_prefix)sISZERO(ABS(ANS(1))+ABS(ANS(2))+ABS(ANS(3)),REF/DBLE(NCOMB),-1))) THEN
	  write(*,*) '##W15 Helicity filter could not be successfully double checked.'
	  write(*,*) '##One reason for this is that you have changed sensible parameters which affected what are the zero helicity configurations.'
	  write(*,*) '##MadLoop will try to reset the Helicity filter with the next PS points it receives.'
	  NTRY=0
	  OPEN(30,FILE=HelFilterFN,err=349)
349   CONTINUE
      CLOSE(30,STATUS='delete')
	ENDIF
C   SET HELDOUBLECHECKED TO .TRUE. WHEN DONE
C   even if it failed we do not want to redo the check afterwards if HELINITSTARTOVER=.FALSE.
    IF (HELPICKED.EQ.NCOMB.AND.(NTRY.NE.0.OR..NOT.HELINITSTARTOVER)) THEN
      DONEHELDOUBLECHECK=.TRUE.
	ENDIF
  ENDIF

C GOTO NEXT HELICITY OR FINISH
  IF(HELPICKED.NE.NCOMB) THEN
    HELPICKED=HELPICKED+1
	MP_DONE=.FALSE.
    goto 200
  ELSE
    ANS(1)=BUFFR(1)
	ANS(2)=BUFFR(2)
	ANS(3)=BUFFR(3)
C We add one here to the number of PS points used for building the reference scale for comparison (used only for loop-induced processes).
    NPSPOINTS = NPSPOINTS+1
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

IF(.NOT.CHECKPHASE.AND.HELDOUBLECHECKED.AND.(CTMODERUN.LE.-1)) THEN
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
	CALL %(proc_prefix)sROTATE_PS(PS,P,1)
	IF (DOING_QP_EVALS) CALL %(proc_prefix)sMP_ROTATE_PS(MP_PS,MP_P,1)
	goto 200
  ENDIF

  IF(.NOT.EVAL_DONE(4).AND. ((DOING_QP_EVALS.AND.NRotations_QP.GE.2).OR.((.NOT.DOING_QP_EVALS).AND.NRotations_DP.GE.2)) ) THEN
	EVAL_DONE(4)=.TRUE.
	CALL %(proc_prefix)sROTATE_PS(PS,P,2)
	IF (DOING_QP_EVALS) CALL %(proc_prefix)sMP_ROTATE_PS(MP_PS,MP_P,2)	
	goto 200
  ENDIF

  CALL %(proc_prefix)sROTATE_PS(PS,P,0)
  IF (DOING_QP_EVALS) CALL %(proc_prefix)sMP_ROTATE_PS(MP_PS,MP_P,0)  

C END OF THE DEFINITIONS OF THE DIFFERENT EVALUATION METHODS

  IF(DOING_QP_EVALS) THEN
    CALL %(proc_prefix)sCOMPUTE_ACCURACY(QP_RES,N_QP_EVAL,ACC,ANS(1))
	ACCURACY(0)=ACC	
	RET_CODE_H=3	
	IF(ACC.GE.MLSTABTHRES) THEN
	  RET_CODE_H=4
	  NEPS=NEPS+1
      CALL %(proc_prefix)sCOMPUTE_ACCURACY(DP_RES,N_DP_EVAL,TEMP1,TEMP)	  
      WRITE(*,*) '##W03 WARNING An unstable PS point was',       ' detected.'
	  WRITE(*,*) '##(DP,QP) accuracies : (',TEMP1,',',ACC,')'
	  WRITE(*,*) '##Best estimate (fin,1eps,2eps) :',(ANS(I),I=1,3)
	  IF(NEPS.LE.10) THEN
	    WRITE(*,*) '##Double precision evaluations :',(DP_RES(1,I),I=1,N_DP_EVAL)
	    WRITE(*,*) '##Quad   precision evaluations :',(QP_RES(1,I),I=1,N_QP_EVAL)		 
	    WRITE(*,*) '##PS point specification :'
	    WRITE(*,*) '##Renormalization scale MU_R=',MU_R	
	    DO I=1,NEXTERNAL
          WRITE (*,'(i2,1x,4e27.17)') i, P(0,i),P(1,i),P(2,i),P(3,i) 
        ENDDO
	  ENDIF
	  IF(NEPS.EQ.10) THEN
	    WRITE(*,*) '##Further output of the details of these unstable PS points will now be suppressed.'
	  ENDIF
    ENDIF
  ELSE
    CALL %(proc_prefix)sCOMPUTE_ACCURACY(DP_RES,N_DP_EVAL,ACC,ANS(1))
	IF(ACC.GE.MLSTABTHRES) THEN
	  DOING_QP_EVALS=.TRUE.
	  EVAL_DONE(1)=.TRUE.
	  DO I=2,MAXSTABILITYLENGTH
        EVAL_DONE(I)=.FALSE.
      ENDDO
	  STAB_INDEX=0
	  CTMODE=4
	  goto 200
    ELSE
	  ACCURACY(0)=ACC
      RET_CODE_H=2	
	ENDIF
  ENDIF
ELSE
  RET_CODE_H=1
  ACCURACY=-1.0d0
ENDIF

 9999 continue

C Finalize the return code
IF (MP_DONE_ONCE) THEN
  RET_CODE_T=2
ELSE
  RET_CODE_T=1
ENDIF
IF(CHECKPHASE.OR..NOT.HELDOUBLECHECKED) THEN
  RET_CODE_H=1
  RET_CODE_T=RET_CODE_T+2
  ACCURACY=-1.0d0
ENDIF
IF (RET_CODE_H.eq.4) THEN
  RET_CODE_U=0
ELSE
  RET_CODE_U=1
ENDIF

C Reinitialize the default threshold if it was specified by the user
IF (USER_STAB_PREC.GT.0.0d0) THEN
  MLSTABTHRES=MLSTABTHRES_BU
  CTMODEINIT=CTMODEINIT_BU  
ENDIF

C Reinitialize the Lorentz test if it had been disabled because spin-2 particles are in the external states.
NROTATIONS_DP = NROTATIONS_DP_BU
NROTATIONS_QP = NROTATIONS_QP_BU

C Conform to the returned synthax of split orders even though the default output does not support it (this then done only for compatibility purpose).
ANSRETURNED(0,0)=ANS(0)
ANSRETURNED(1,0)=ANS(1)
ANSRETURNED(2,0)=ANS(2)
ANSRETURNED(3,0)=ANS(3)

C Reinitialize the check phase logicals and the filters if check bypassed
IF (BYPASS_CHECK) THEN
  CHECKPHASE = OLD_CHECKPHASE
  HELDOUBLECHECKED = OLD_HELDOUBLECHECKED
  DO I=1,NCOMB
    GOODHEL(I)=OLD_GOODHEL(I)
  ENDDO
  DO I=1,NCOMB
    DO J=1,NLOOPAMPS
      GOODAMP(J,I)=OLD_GOODAMP(J,I)
	ENDDO
  ENDDO
ENDIF

END

      SUBROUTINE %(proc_prefix)scompute_accuracy(fulllist, length, acc, estimate)
      implicit none
C  
C PARAMETERS 
C
      integer maxstabilitylength
	  common/%(proc_prefix)sstability_tests/maxstabilitylength
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

C     The technique below is too sensitive, typically to
C     unstablities in very small poles
C      ACC=MAXVAL(ACCURACIES,1,MASK3)
C     The following is used instead
      ACC = 0.0d0
      AVG = 0.0d0
      DO I=1,3
        ACC = ACC + ACCURACIES(I)*ABS(ESTIMATE(I))
        AVG = AVG + ESTIMATE(I)
      ENDDO
      ACC  = ACC / ( ABS(AVG) / 3.0d0)

      end

      SUBROUTINE %(proc_prefix)sSET_N_EVALS(N_DP_EVALS,N_QP_EVALS)
 
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
	    WRITE(*,*) '##ERROR:: Increase hardcoded maxstabilitylength.'
		STOP
	  ENDIF

	  END


C THIS SUBROUTINE SIMPLY SET THE GLOBAL PS CONFIGURATION GLOBAL VARIABLES FROM A GIVEN VARIABLE IN DOUBLE PRECISION
	  SUBROUTINE %(proc_prefix)sSET_MP_PS(P)

	    INTEGER    NEXTERNAL
        PARAMETER (NEXTERNAL=%(nexternal)d)
	    %(real_mp_format)s MP_PS(0:3,NEXTERNAL),MP_P(0:3,NEXTERNAL)
	    common/%(proc_prefix)sMP_PSPOINT/MP_PS,MP_P
  	    %(real_dp_format)s P(0:3,NEXTERNAL)

	    DO I=1,NEXTERNAL
	      DO J=0,3
    	    MP_PS(J,I)=P(J,I) 
    	  ENDDO
  	    ENDDO
  	    CALL %(proc_prefix)sMP_IMPROVE_PS_POINT_PRECISION(MP_PS)
  	    DO I=1,NEXTERNAL
    	  DO J=0,3
      	  	MP_P(J,I)=MP_PS(J,I) 
    	  ENDDO
  	  	ENDDO

	  END

	  SUBROUTINE %(proc_prefix)sSET_COUPLINGORDERS_TARGET(SOTARGET)
	  IMPLICIT NONE
C
C     This routine can be accessed by an external user to set the squared split order target.
C     This functionality is only available in the optimized mode, but for compatibility 
C     purposes, a dummy version is also put in this default output.
C
C
C     ARGUMENTS
C
      INTEGER SOTARGET
C ----------
C BEGIN CODE
C ----------
      write(*,*) '##WARNING:: Ignored, the possibility of selecting specific squared order contributions is not available in the default mode.'

	  END

	  SUBROUTINE %(proc_prefix)sFORCE_STABILITY_CHECK(ONOFF)
C
C This function can be called by the MadLoop user so as to always have stability
C checked, even during initialisation, when calling the *_thres routines.
C
      LOGICAL ONOFF

	  LOGICAL BYPASS_CHECK, ALWAYS_TEST_STABILITY
	  DATA BYPASS_CHECK, ALWAYS_TEST_STABILITY /.FALSE.,.FALSE./
	  COMMON/%(proc_prefix)sBYPASS_CHECK/BYPASS_CHECK, ALWAYS_TEST_STABILITY

	  ALWAYS_TEST_STABILITY = ONOFF

	  END

      SUBROUTINE %(proc_prefix)sGET_ANSWER_DIMENSION(ANSDIM)
C
C     Simple subroutine which returns the upper bound of the second dimension of the
C     quantity ANS(0:3,0:ANSDIM) returned by MadLoop. As long as the default output
c     cannot handle split orders, this ANSDIM will always be 0.
C
	  INCLUDE "nsquaredSO.inc"

	  INTEGER ANSDIM

	  ANSDIM=NSQUAREDSO

      END

      SUBROUTINE %(proc_prefix)sGET_NSQSO_LOOP(NSQSO)
C
C     Simple subroutine returning the number of squared split order
C     contributions returned in ANS when calling sloopmatrix 
C
      INCLUDE "nsquaredSO.inc"

	  INTEGER NSQSO

	  NSQSO=NSQUAREDSO

      END

      SUBROUTINE %(proc_prefix)sSLOOPMATRIXHEL_THRES(P,HEL,ANS,PREC_ASKED,PREC_FOUND,RET_CODE)
	  IMPLICIT NONE
C  
C CONSTANTS
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INCLUDE "nsquaredSO.inc"
C  
C ARGUMENTS 
C  
      %(real_dp_format)s P(0:3,NEXTERNAL)
      %(real_dp_format)s ANS(0:3,0:NSQUAREDSO)
	  INTEGER HEL,RET_CODE
	  %(real_dp_format)s PREC_ASKED,PREC_FOUND(0:NSQUAREDSO)
C
C GLOBAL VARIABLES
C
	  %(real_dp_format)s USER_STAB_PREC
	  common/%(proc_prefix)sUSER_STAB_PREC/USER_STAB_PREC

	  INTEGER I

	  INTEGER H,T,U
	  %(real_dp_format)s ACCURACY(0:NSQUAREDSO)
	  common/%(proc_prefix)sACC/ACCURACY,H,T,U

	  LOGICAL BYPASS_CHECK, ALWAYS_TEST_STABILITY
	  COMMON/%(proc_prefix)sBYPASS_CHECK/BYPASS_CHECK, ALWAYS_TEST_STABILITY

C ----------
C BEGIN CODE
C ----------
      USER_STAB_PREC = PREC_ASKED
      CALL %(proc_prefix)sSLOOPMATRIXHEL(P,HEL,ANS)
	  IF(ALWAYS_TEST_STABILITY.AND.(H.eq.1.OR.ACCURACY(0).lt.0.0d0)) THEN
	    BYPASS_CHECK = .TRUE.
        CALL %(proc_prefix)sSLOOPMATRIXHEL(P,HEL,ANS) 
	    BYPASS_CHECK = .FALSE.
C     Make sure we correctly return an initialization-type T code
		IF (T.eq.2) T=4
		IF (T.eq.1) T=3
	  ENDIF
	  
C Reset it to default value not to affect next runs
	  USER_STAB_PREC = -1.0d0	  
      DO I=0,NSQUAREDSO
	    PREC_FOUND(I)=ACCURACY(I)
	  ENDDO
      RET_CODE=100*H+10*T+U

	  END

      SUBROUTINE %(proc_prefix)sSLOOPMATRIX_THRES(P,ANS,PREC_ASKED,PREC_FOUND,RET_CODE)
C
C     Inputs are:
C     P(0:3, Nexternal)  double  :: Kinematic configuration (E,px,py,pz)
C     PEC_ASKED          double  :: Target relative accuracy, -1 for default
C
C     Outputs are:
C     ANS(3)             double  :: Result (finite, single pole, double pole) 
C     PREC_FOUND         double  :: Relative accuracy estimated for the result
C                                   Returns -1 if no stab test could be performed.
C	  RET_CODE			 integer :: Return code. See below for details
C
C     Return code conventions: RET_CODE = H*100 + T*10 + U
C
C     H == 1
C         Stability unknown.
C     H == 2
C         Stable PS (SPS) point.
C         No stability rescue was necessary.
C     H == 3
C         Unstable PS (UPS) point.
C         Stability rescue necessary, and successful.
C     H == 4
C         Exceptional PS (EPS) point.
C         Stability rescue attempted, but unsuccessful.
C
C     T == 1
C         Default computation (double prec.) was performed.
C     T == 2
C         Quadruple precision was used for this PS point.
C     T == 3
C         MadLoop in initialization phase. Only double precision used.
C     T == 4
C         MadLoop in initialization phase. Quadruple precision used.
C
C     U is a number left for future use (always set to 0 for now).
C     example: TIR vs OPP usage.
C
      IMPLICIT NONE
C  
C CONSTANTS
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INCLUDE "nsquaredSO.inc"	  
C  
C ARGUMENTS 
C  
      %(real_dp_format)s P(0:3,NEXTERNAL)
      %(real_dp_format)s ANS(0:3,0:NSQUAREDSO)
	  %(real_dp_format)s PREC_ASKED,PREC_FOUND(0:NSQUAREDSO)
	  INTEGER RET_CODE
C
C GLOBAL VARIABLES
C
	  %(real_dp_format)s USER_STAB_PREC
	  common/%(proc_prefix)sUSER_STAB_PREC/USER_STAB_PREC

	  INTEGER I

	  INTEGER H,T,U
	  %(real_dp_format)s ACCURACY(0:NSQUAREDSO)
	  common/%(proc_prefix)sACC/ACCURACY,H,T,U

	  LOGICAL BYPASS_CHECK, ALWAYS_TEST_STABILITY
	  COMMON/%(proc_prefix)sBYPASS_CHECK/BYPASS_CHECK, ALWAYS_TEST_STABILITY

C ----------
C BEGIN CODE
C ----------
      USER_STAB_PREC = PREC_ASKED
      CALL %(proc_prefix)sSLOOPMATRIX(P,ANS)
	  IF(ALWAYS_TEST_STABILITY.AND.(H.eq.1.OR.ACCURACY(0).lt.0.0d0)) THEN
	    BYPASS_CHECK = .TRUE.
        CALL %(proc_prefix)sSLOOPMATRIX(P,ANS)
		BYPASS_CHECK = .FALSE.
C     Make sure we correctly return an initialization-type T code
		IF (T.eq.2) T=4
		IF (T.eq.1) T=3
	  ENDIF

C Reset it to default value not to affect next runs
	  USER_STAB_PREC = -1.0d0
	  DO I=0,NSQUAREDSO
	    PREC_FOUND(I)=ACCURACY(I)
	  ENDDO
	  RET_CODE=100*H+10*T+U		 
	  
	  END
