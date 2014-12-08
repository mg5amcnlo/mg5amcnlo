C --=========================================--
C   Main subroutine
C --=========================================--

	  SUBROUTINE %(proc_prefix)sSLOOPMATRIX(P_USER,ANS)
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
C     These are constants related to the split orders
      %(get_nsqso_born)s
      INTEGER    NSO, NSQUAREDSO, NAMPSO
	  PARAMETER (NSO=%(nSO)d, NSQUAREDSO=%(nSquaredSO)d, NAMPSO=%(nAmpSO)d)
	  INTEGER ANS_DIMENSION
	  PARAMETER(ANS_DIMENSION=MAX(NSQSO_BORN,NSQUAREDSO))
	  INTEGER NSQSOXNLG
	  PARAMETER (NSQSOXNLG=NSQUAREDSO*NLOOPGROUPS)
	  INTEGER NSQUAREDSOP1
	  PARAMETER (NSQUAREDSOP1=NSQUAREDSO+1)
C     The total number of loop reduction libraries
C     At present, there are only CutTools,PJFry++,IREGI,Golem95
      INTEGER NLOOPLIB
      PARAMETER (NLOOPLIB=4)
C     Only CutTools provides QP
      INTEGER QP_NLOOPLIB
      PARAMETER (QP_NLOOPLIB=1)
C     This parameter is designed for the check timing command of MG5
      LOGICAL SKIPLOOPEVAL
      PARAMETER (SKIPLOOPEVAL=.FALSE.)
	  LOGICAL BOOTANDSTOP
      PARAMETER (BOOTANDSTOP=.FALSE.)
	  INTEGER MAXSTABILITYLENGTH
	  DATA MAXSTABILITYLENGTH/20/
	  common/%(proc_prefix)sstability_tests/maxstabilitylength
C  
C ARGUMENTS 
C  
      %(real_dp_format)s P_USER(0:3,NEXTERNAL)
C
C The zeroth component of the second dimension is the result summed over all
c contributing split orders. The zeroth component of the first one is the Born.
C Notice that the upper bound of the second integer is not number of squared orders
c combination for the loops but the maximum between this number for the Born
c contributions and the loop ones. There are some cases for which the Born contrib.
c has squared split order contributions than the loop does. For example
c
c generate u u~ > d d~ QCD^2<=2 QED^2<=99 [virt=QCD]
c
c It is however somehow academical. This is why ANS_DIMENSION is not just NSQSO but rather MAX(NSQSO,NSQSO_BORN)
C
	  %(real_dp_format)s ANS(0:3,0:ANS_DIMENSION)
C  
C LOCAL VARIABLES 
C  
      INTEGER I,J,K,H,DUMMY,I_QP_LIB

      CHARACTER*512 ParamFN,HelConfigFN,LoopFilterFN,ColorNumFN,ColorDenomFN,HelFilterFN
	  CHARACTER*512 TMP
	  SAVE ParamFN
	  SAVE HelConfigFN
	  SAVE LoopFilterFN
	  SAVE ColorNumFN
	  SAVE ColorDenomFN
	  SAVE HelFilterFN

	  INTEGER CTMODEINIT_BU
	  %(real_dp_format)s MLSTABTHRES_BU
	  INTEGER NEWHELREF	  
	  LOGICAL HEL_INCONSISTENT	  
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
	  %(real_dp_format)s ACC(0:NSQUAREDSO)
	  %(real_dp_format)s DP_RES(3,0:NSQUAREDSO,MAXSTABILITYLENGTH)
C QP_RES STORES THE QUADRUPLE PRECISION RESULT OBTAINED FROM DIFFERENT EVALUATION METHODS IN ORDER TO ASSESS STABILITY.
	  %(real_dp_format)s QP_RES(3,0:NSQUAREDSO,MAXSTABILITYLENGTH)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
	  INTEGER NATTEMPTS
	  DATA NATTEMPTS/0/
	  DATA IC/NEXTERNAL*1/
	  %(real_dp_format)s HELSAVED(3,NCOMB)
	  INTEGER ITEMP
	  LOGICAL LTEMP
	  %(real_dp_format)s BORNBUFF(0:NSQSO_BORN)
	  %(real_dp_format)s BUFFR(3,0:NSQUAREDSO),BUFFR_BIS(3,0:NSQUAREDSO),TEMP(0:3,0:NSQUAREDSO),TEMP1(0:NSQUAREDSO),TEMP2
	  %(complex_dp_format)s COEFS(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)
      %(complex_dp_format)s CFTOT
	  LOGICAL FOUNDHELFILTER,FOUNDLOOPFILTER
	  DATA FOUNDHELFILTER/.TRUE./
	  DATA FOUNDLOOPFILTER/.TRUE./
      LOGICAL LOOPFILTERBUFF(NSQUAREDSO,NLOOPGROUPS)
	  DATA ((LOOPFILTERBUFF(J,I),J=1,NSQUAREDSO),I=1,NLOOPGROUPS)/NSQSOXNLG*.FALSE./

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
	  INTEGER OLD_GOODHEL(NCOMB)
	  LOGICAL OLD_GOODAMP(NSQUAREDSO,NLOOPGROUPS)
	  LOGICAL BYPASS_CHECK, ALWAYS_TEST_STABILITY
	  COMMON/%(proc_prefix)sBYPASS_CHECK/BYPASS_CHECK, ALWAYS_TEST_STABILITY
C
C FUNCTIONS
C
      INTEGER %(proc_prefix)sML5SOINDEX_FOR_BORN_AMP
	  INTEGER %(proc_prefix)sML5SOINDEX_FOR_LOOP_AMP
	  INTEGER %(proc_prefix)sML5SQSOINDEX
      INTEGER %(proc_prefix)sISSAME
      LOGICAL %(proc_prefix)sISZERO
      INTEGER SET_RET_CODE_U
C  
C GLOBAL VARIABLES
C  
      include 'coupl.inc'
	  include 'mp_coupl.inc'
	  include 'MadLoopParams.inc'

      LOGICAL CHOSEN_SO_CONFIGS(NSQUAREDSO)
	  DATA CHOSEN_SO_CONFIGS/%(chosen_so_configs)s/
      COMMON/%(proc_prefix)sCHOSEN_LOOP_SQSO/CHOSEN_SO_CONFIGS

	  LOGICAL CHECKPHASE
	  DATA CHECKPHASE/.TRUE./
	  LOGICAL HELDOUBLECHECKED
	  DATA HELDOUBLECHECKED/.FALSE./
      common/%(proc_prefix)sINIT/CHECKPHASE, HELDOUBLECHECKED
	  INTEGER NTRY
      DATA NTRY/0/
	  %(real_dp_format)s REF
	  DATA REF/0.0d0/

	  LOGICAL MP_DONE
	  DATA MP_DONE/.FALSE./
	  common/%(proc_prefix)sMP_DONE/MP_DONE
C     A FLAG TO DENOTE WHETHER THE CORRESPONDING LOOPLIBS ARE AVAILABLE OR NOT
          LOGICAL LOOPLIBS_AVAILABLE(4)
	  %(data_looplibs_av)s
	  common/%(proc_prefix)sLOOPLIBS_AV/ LOOPLIBS_AVAILABLE
C     A FLAG TO DENOTE WHETHER THE CORRESPONDING DIRECTION TESTS AVAILABLE OR NOT IN THE LOOPLIBS
C     PJFry++ and Golem95 do not support direction test
          LOGICAL LOOPLIBS_DIRECTEST(4)
	  DATA LOOPLIBS_DIRECTEST /.TRUE.,.FALSE.,.TRUE.,.FALSE./

C     PS CAN POSSIBILY BE PASSED THROUGH IMPROVE_PS BUT IS NOT MODIFIED FOR THE PURPOSE OF THE STABILITY TEST
C     EVEN THOUGH THEY ARE PUT IN COMMON BLOCK, FOR NOW THEY ARE NOT USED ANYWHERE ELSE
	  %(real_dp_format)s PS(0:3,NEXTERNAL)
	  common/%(proc_prefix)sPSPOINT/PS
C     AGAIN BELOW, MP_PS IS THE FIXED (POSSIBLY IMPROVED) MP PS POINT AND MP_P IS THE ONE WHICH CAN BE MODIFIED (I.E. ROTATED ETC.) FOR STABILITY PURPOSE
	  %(real_mp_format)s MP_PS(0:3,NEXTERNAL),MP_P(0:3,NEXTERNAL)
	  common/%(proc_prefix)sMP_PSPOINT/MP_PS,MP_P

	  %(real_dp_format)s LSCALE
	  INTEGER CTMODE	  
      common/%(proc_prefix)sCT/LSCALE,CTMODE
	  LOGICAL MP_PS_SET
	  DATA MP_PS_SET/.FALSE./

C     The parameter below sets the convention for the helicity filter
C     For a given helicity, the attached integer 'i' means
C     'i' in ]-inf;-HELOFFSET[ -> Helicity is equal, up to a sign, to helicity number abs(i+HELOFFSET)
C     'i' == -HELOFFSET        -> Helicity is analytically zero
C     'i' in ]-HELOFFSET,inf[  -> Helicity is contributing with weight 'i'. If it is zero, it is skipped.
C     Typically, the hel_offset is 10000
	  INTEGER HELOFFSET
	  DATA HELOFFSET/%(hel_offset)d/
	  INTEGER GOODHEL(NCOMB)
	  LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
	  common/%(proc_prefix)sFilters/GOODAMP,GOODHEL,HELOFFSET

	  INTEGER HELPICKED
	  DATA HELPICKED/-1/
	  common/%(proc_prefix)sHELCHOICE/HELPICKED
	  INTEGER USERHEL
	  DATA USERHEL/-1/
	  common/%(proc_prefix)sUSERCHOICE/USERHEL

C     This integer can be accessed by an external user to set its target squared split order.
C     If set to a value different than -1, the code will try to avoid computing anything which
C     does not contribute to contributions of squared split orders SQSO_TARGET and below.
	  INTEGER SQSO_TARGET
	  DATA SQSO_TARGET/-1/
	  common/%(proc_prefix)sSOCHOICE/SQSO_TARGET
C     The following logical are used to broadcast the fact that the target 'required' CT and
C     loop split orders contributions have been reached already and the rest can be skipped.
	  LOGICAL UVCT_REQ_SO_DONE,MP_UVCT_REQ_SO_DONE,CT_REQ_SO_DONE,MP_CT_REQ_SO_DONE,LOOP_REQ_SO_DONE,MP_LOOP_REQ_SO_DONE,CTCALL_REQ_SO_DONE,FILTER_SO
	  DATA UVCT_REQ_SO_DONE/.FALSE./
	  DATA MP_UVCT_REQ_SO_DONE/.FALSE./	  
	  DATA CT_REQ_SO_DONE/.FALSE./
	  DATA MP_CT_REQ_SO_DONE/.FALSE./	  
	  DATA LOOP_REQ_SO_DONE/.FALSE./
	  DATA MP_LOOP_REQ_SO_DONE/.FALSE./	  
	  DATA CTCALL_REQ_SO_DONE/.FALSE./
	  DATA FILTER_SO/.FALSE./
	  common/%(proc_prefix)sSO_REQS/UVCT_REQ_SO_DONE,MP_UVCT_REQ_SO_DONE,CT_REQ_SO_DONE,MP_CT_REQ_SO_DONE,LOOP_REQ_SO_DONE,MP_LOOP_REQ_SO_DONE,CTCALL_REQ_SO_DONE,FILTER_SO

	  integer I_SO
	  DATA I_SO/1/
	  common/%(proc_prefix)sI_SO/I_SO
	  integer I_LIB
	  DATA I_LIB/1/
	  common/%(proc_prefix)sI_LIB/I_LIB
C TILL NOW, ONLY CUTTOOLS PROVIDE QP
       	  LOGICAL QP_TOOLS_AVAILABLE
       	  DATA QP_TOOLS_AVAILABLE/.FALSE./
       	  INTEGER INDEX_QP_TOOLS(QP_NLOOPLIB+1)
       	  common/%(proc_prefix)sLOOP_TOOLS/QP_TOOLS_AVAILABLE,INDEX_QP_TOOLS

	  %(dp_born_amps_decl)s	  
	  %(complex_dp_format)s W(20,NWAVEFUNCS)
	  common/%(proc_prefix)sW/W  

      %(complex_mp_format)s MPW(20,NWAVEFUNCS)
	  common/%(proc_prefix)sMP_W/MPW

	  %(complex_dp_format)s WL(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE,0:NLOOPWAVEFUNCS)
	  %(complex_dp_format)s PL(0:3,0:NLOOPWAVEFUNCS)
	  common/%(proc_prefix)sWL/WL,PL

	  %(complex_dp_format)s LOOPCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)
	  common/%(proc_prefix)sLCOEFS/LOOPCOEFS

      %(complex_dp_format)s AMPL(3,NCTAMPS)
	  common/%(proc_prefix)sAMPL/AMPL

      %(complex_dp_format)s LOOPRES(3,NSQUAREDSO,NLOOPGROUPS)
	  LOGICAL S(NSQUAREDSO,NLOOPGROUPS)
	  common/%(proc_prefix)sLOOPRES/LOOPRES,S

	  INTEGER CF_D(NCOLORROWS,%(color_matrix_size)s)
	  INTEGER CF_N(NCOLORROWS,%(color_matrix_size)s)
	  common/%(proc_prefix)sCF/CF_D,CF_N

	  INTEGER HELC(NEXTERNAL,NCOMB)
	  common/%(proc_prefix)sHELCONFIGS/HELC

	  %(real_dp_format)s PREC,USER_STAB_PREC
	  DATA USER_STAB_PREC/-1.0d0/	  
	  COMMON/%(proc_prefix)sUSER_STAB_PREC/USER_STAB_PREC

C     Return codes H,T,U correspond to the hundreds, tens and units
C     building returncode, i.e.
C     RETURNCODE=100*RET_CODE_H+10*RET_CODE_T+RET_CODE_U
	  
      INTEGER RET_CODE_H,RET_CODE_T,RET_CODE_U
	  %(real_dp_format)s ACCURACY(0:NSQUAREDSO)
	  DATA (ACCURACY(I),I=0,NSQUAREDSO)/NSQUAREDSOP1*1.0d0/
	  DATA RET_CODE_H,RET_CODE_T,RET_CODE_U/1,1,0/
	  common/%(proc_prefix)sACC/ACCURACY,RET_CODE_H,RET_CODE_T,RET_CODE_U
	  
	  LOGICAL MP_DONE_ONCE
	  DATA MP_DONE_ONCE/.FALSE./
	  common/%(proc_prefix)sMP_DONE_ONCE/MP_DONE_ONCE

	  character(512) MLPath
      common/MLPATH/MLPath

	  LOGICAL ML_INIT
	  common/ML_INIT/ML_INIT

C ----------
C BEGIN CODE
C ----------

IF(ML_INIT) THEN
  CALL PRINT_MADLOOP_BANNER()
  TMP = 'auto'
  CALL setMadLoopPath(TMP)
  CALL JOINPATH(MLPATH,PARAMFNAME,PARAMFN)
  CALL MADLOOPPARAMREADER(PARAMFN,.TRUE.)
  ML_INIT = .FALSE.
ENDIF

IF(NTRY.EQ.0) THEN
C  CALL MADLOOPPARAMREADER(paramFileName,.TRUE.)
  QP_TOOLS_AVAILABLE=.FALSE.
  INDEX_QP_TOOLS(1:QP_NLOOPLIB+1)=0
C SKIP THE ONES THAT NOT AVAILABLE
  J=1
  DO I=1,NLOOPLIB
     IF(MLReductionLib(J).EQ.0)EXIT
     IF(.NOT.LOOPLIBS_AVAILABLE(MLReductionLib(J)))THEN
	MLReductionLib(J:NLOOPLIB-1)=MLReductionLib(J+1:NLOOPLIB)
	MLReductionLib(NLOOPLIB)=0
     ELSE
	J=J+1
     ENDIF
  ENDDO
  IF(MLReductionLib(1).EQ.0)THEN
     STOP "No available loop reduction lib is provided. Make sure MLReductionLib is correct."
  ENDIF
  J=0
  DO I=1,NLOOPLIB
     IF(MLReductionLib(I).EQ.1)THEN
      J=J+1
      IF(.NOT.QP_TOOLS_AVAILABLE)QP_TOOLS_AVAILABLE=.TRUE.
      INDEX_QP_TOOLS(J)=I
     ENDIF	
  ENDDO

C Setup the file paths
  CALL JOINPATH(MLPATH,PARAMFNAME,PARAMFN)
  CALL JOINPATH(MLPATH,PROC_PREFIX,TMP)
  CALL JOINPATH(TMP,HELCONFIGFNAME,HELCONFIGFN)
  CALL JOINPATH(TMP,LOOPFILTERFNAME,LOOPFILTERFN)
  CALL JOINPATH(TMP,COLORNUMFNAME,COLORNUMFN)
  CALL JOINPATH(TMP,COLORDENOMFNAME,COLORDENOMFN)
  CALL JOINPATH(TMP,HELFILTERFNAME,HELFILTERFN)

  CALL %(proc_prefix)sSET_N_EVALS(N_DP_EVAL,N_QP_EVAL)  

  HELDOUBLECHECKED=.NOT.DoubleCheckHelicityFilter
OPEN(1, FILE=LoopFilterFN, err=100, status='OLD',           action='READ')
  DO J=1,NLOOPGROUPS
    READ(1,*,END=101) (GOODAMP(I,J),I=1,NSQUAREDSO)
  ENDDO
  GOTO 101
100  CONTINUE
  FOUNDLOOPFILTER=.FALSE.
  DO J=1,NLOOPGROUPS
    DO I=1,NSQUAREDSO
	  GOODAMP(I,J)=(.NOT.USELOOPFILTER)
	ENDDO
  ENDDO
101  CONTINUE
CLOSE(1)

OPEN(1, FILE=HelFilterFN, err=102, status='OLD',           action='READ')
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

OPEN(1, FILE=ColorNumFN, err=104, status='OLD',           action='READ')
  DO I=1,NCOLORROWS
    READ(1,*,END=105) (CF_N(I,J),J=1,%(color_matrix_size)s)
  ENDDO
  GOTO 105
104  CONTINUE
  STOP 'Color factors could not be initialized from file %(proc_prefix)sColorNumFactors.dat. File not found' 
105  CONTINUE
CLOSE(1)
OPEN(1, FILE=ColorDenomFN, err=106, status='OLD',           action='READ')
  DO I=1,NCOLORROWS
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
    stop
  ENDIF
ENDIF

%(compute_born)s

%(set_reference)s

MP_DONE=.FALSE.
MP_DONE_ONCE=.FALSE.
MP_PS_SET=.FALSE.
STAB_INDEX=0
DOING_QP_EVALS=.FALSE.
EVAL_DONE(1)=.TRUE.
DO I=2,MAXSTABILITYLENGTH
  EVAL_DONE(I)=.FALSE.
ENDDO

IF(.NOT.BYPASS_CHECK) THEN
  NTRY=NTRY+1
ENDIF

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

IF(DONEHELDOUBLECHECK.AND.(.NOT.HELDOUBLECHECKED)) THEN
  HELDOUBLECHECKED=.TRUE.
  DONEHELDOUBLECHECK=.FALSE.
ENDIF

CHECKPHASE=(NTRY.LE.CHECKCYCLE).AND.(((.NOT.FOUNDLOOPFILTER).AND.USELOOPFILTER).OR.(.NOT.FOUNDHELFILTER))

IF (WriteOutFilters) THEN
IF ((.NOT. CHECKPHASE).AND.(.NOT.FOUNDHELFILTER)) THEN

OPEN(1, FILE=HelFilterFN, err=110, status='NEW',action='WRITE')
  DO I=1,NCOMB
    WRITE(1,*) GOODHEL(I)  
  ENDDO
110  CONTINUE
  CLOSE(1)
FOUNDHELFILTER=.TRUE.
ENDIF

IF ((.NOT. CHECKPHASE).AND.(.NOT.FOUNDLOOPFILTER).AND.USELOOPFILTER) THEN
OPEN(1, FILE=LoopFilterFN, err=111, status='NEW',action='WRITE')
  DO J=1,NLOOPGROUPS
    WRITE(1,*) (GOODAMP(I,J),I=1,NSQUAREDSO)
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
	GOODHEL(I)=1
  ENDDO
  DO I=1,NSQUAREDSO
    DO J=1,NLOOPGROUPS
      OLD_GOODAMP(I,J)=GOODAMP(I,J)
	  GOODAMP(I,J)=.TRUE.
    ENDDO
  ENDDO
ENDIF

IF(CHECKPHASE.OR.(.NOT.HELDOUBLECHECKED)) THEN
  HELPICKED=1
  CTMODE=CTMODEINIT
ELSE
  IF (USERHEL.ne.-1) then
    IF(GOODHEL(USERHEL).eq.-HELOFFSET) THEN
	  DO I=0,NSQUAREDSO
        ANS(1,I)=0.0d0
        ANS(2,I)=0.0d0
        ANS(3,I)=0.0d0
	  ENDDO
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
  DO I=0,NSQUAREDSO
    BUFFR(K,I)=0.0d0
  ENDDO
  DO I=1,NCTAMPS
    AMPL(K,I)=(0.0d0,0.0d0)
  ENDDO
ENDDO
C     USE THE FIRST LOOP REDUCTION LIBRARY AND THE FIRST QP LOOP REDUCTION LIBRARY
I_LIB=1
I_QP_LIB=1
200 CONTINUE

IF (.NOT.MP_PS_SET.AND.(CTMODE.EQ.0.OR.CTMODE.GE.4)) THEN
  CALL %(proc_prefix)sSET_MP_PS(P_USER)
  MP_PS_SET = .TRUE.
ENDIF

LSCALE=DSQRT(ABS((P(0,1)+P(0,2))**2-(P(1,1)+P(1,2))**2-(P(2,1)+P(2,2))**2-(P(3,1)+P(3,2))**2))

CTCALL_REQ_SO_DONE=.FALSE.
FILTER_SO = (.NOT.CHECKPHASE).AND.HELDOUBLECHECKED.AND.(SQSO_TARGET.ne.-1)

DO I=1,NLOOPGROUPS
  DO J=0,LOOPMAXCOEFS-1
    DO K=1,NSQUAREDSO
      LOOPCOEFS(J,K,I)=(0.0d0,0.0d0)
	ENDDO
  ENDDO
ENDDO

DO I=1,NLOOPGROUPS
  DO J=1,3
    DO K=1,NSQUAREDSO  
      LOOPRES(J,K,I)=0.0d0
	ENDDO
  ENDDO
ENDDO

DO K=1,3
  DO I=0,NSQUAREDSO
    ANS(K,I)=0.0d0
  ENDDO
ENDDO

C Check if we directly go to multiple precision
IF (CTMODE.GE.4) THEN
  IF (.NOT.MP_DONE) THEN
    CALL %(proc_prefix)sMP_COMPUTE_LOOP_COEFS(MP_P,BUFFR_BIS)
C   It should be safe to directly set MP_DONE to true already here. But maybe I overlooked something.
    MP_DONE=.TRUE.
  ENDIF
C Even if MP_DONE is .TRUE. we should anyway skip the
C double precision evaluation as it as already been
c computed in quadruple precision.
  goto 300
ENDIF

DO H=1,NCOMB
  IF ((HELPICKED.EQ.H).OR.((HELPICKED.EQ.-1).AND.(CHECKPHASE.OR.(.NOT.HELDOUBLECHECKED).OR.(GOODHEL(H).GT.-HELOFFSET.AND.GOODHEL(H).NE.0)))) THEN
  DO I=1,NEXTERNAL
    NHEL(I)=HELC(I,H)
  ENDDO

  UVCT_REQ_SO_DONE=.FALSE.
  CT_REQ_SO_DONE=.FALSE.
  LOOP_REQ_SO_DONE=.FALSE.

C Helas calls for the born amplitudes and counterterms associated to given loops
  %(born_ct_helas_calls)s
2000 CONTINUE
  CT_REQ_SO_DONE=.TRUE.

c Helas calls for the counterterm of type 'UVtree' in the UFO.
c These are generated irrespectively of the produced loops.
c In general, only wavefunction renormalization counterterms
c (if needed by the loop UFO model) are of this type.
c Quite often and in principle for all loop UFO models from 
c FeynRules, there are none of these type of counterterms.
  %(uvct_helas_calls)s
3000 CONTINUE
  UVCT_REQ_SO_DONE=.TRUE.

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
4000 CONTINUE
  LOOP_REQ_SO_DONE=.TRUE.

  ENDIF
ENDDO

%(coef_merging)s

DO I=0,NSQUAREDSO
  DO J=1,3
    BUFFR_BIS(J,I)=ANS(J,I)
  ENDDO
ENDDO

300 CONTINUE
C     Free cache when using IREGI
%(iregi_free_ps)s

DO I=0,NSQUAREDSO
  DO J=1,3
    ANS(J,I)=BUFFR_BIS(J,I)
  ENDDO
ENDDO

IF(SKIPLOOPEVAL) THEN
  GOTO 1226
ENDIF

DO I_SO=1,NSQUAREDSO
  DO J=1,NLOOPGROUPS
    S(I_SO,J)=.TRUE.
  ENDDO
  IF (FILTER_SO.and.SQSO_TARGET.NE.I_SO) GOTO 5001
%(loop_CT_calls)s
  GOTO 5001
5000 CONTINUE
  CTCALL_REQ_SO_DONE=.TRUE.
5001 CONTINUE
ENDDO

%(actualize_ans)s

1226 CONTINUE

IF (CHECKPHASE.OR.(.NOT.HELDOUBLECHECKED)) THEN
  IF((USERHEL.EQ.-1).OR.(USERHEL.EQ.HELPICKED)) THEN
C   TO KEEP TRACK OF THE FINAL ANSWER TO BE RETURNED DURING CHECK PHASE
    DO I=0,NSQUAREDSO
      DO K=1,3
        BUFFR(K,I)=BUFFR(K,I)+ANS(K,I)
	  ENDDO
	ENDDO
  ENDIF
C SAVE RESULT OF EACH INDEPENDENT HELICITY FOR COMPARISON DURING THE HELICITY FILTER SETUP
  HELSAVED(1,HELPICKED)=ANS(1,0)
  HELSAVED(2,HELPICKED)=ANS(2,0)
  HELSAVED(3,HELPICKED)=ANS(3,0)

  IF (CHECKPHASE) THEN
C   SET THE HELICITY FILTER
    IF(.NOT.FOUNDHELFILTER) THEN
	  HEL_INCONSISTENT=.FALSE.
      IF(%(proc_prefix)sIsZero(ABS(ANS(1,0))+ABS(ANS(2,0))+ABS(ANS(3,0)),REF/DBLE(NCOMB),-1,-1)) THEN
        IF(NTRY.EQ.1) THEN
	      GOODHEL(HELPICKED)=-HELOFFSET
	    ELSEIF(GOODHEL(HELPICKED).NE.-HELOFFSET) THEN
		  WRITE(*,*) '##W02A WARNING Inconsistent helicity ',HELPICKED
		  IF(HELINITSTARTOVER) THEN
	        WRITE(*,*) '##I01 INFO Initialization starting over because of inconsistency in the helicity filter setup.'
	        NTRY=0
		  ELSE
		    HEL_INCONSISTENT=.TRUE.
		  ENDIF
		ENDIF
      ELSE
	    DO H=1,HELPICKED-1
		  IF(GOODHEL(H).GT.-HELOFFSET) THEN
C           Be looser for helicity check, bring a factor 100
		    DUMMY=%(proc_prefix)sISSAME(ANS(1,0),HELSAVED(1,H),REF,.FALSE.)
		    IF(DUMMY.NE.0) THEN
		      IF(NTRY.EQ.1) THEN
C               Set the matching helicity to be contributing once more
		        GOODHEL(H)=GOODHEL(H)+DUMMY
C               Use an offset to clearly show it is linked to an other one and to avoid overlap
			    GOODHEL(HELPICKED)=-H-HELOFFSET
C             Make sure we have paired this hel config to the same one last PS point
			  ELSEIF(GOODHEL(HELPICKED).NE.(-H-HELOFFSET)) THEN
			    WRITE(*,*) '##W02B WARNING Inconsistent helicity ',HELPICKED
		        IF(HELINITSTARTOVER) THEN
	              WRITE(*,*) '##I01 INFO Initialization starting over because of inconsistency in the helicity filter setup.'
	              NTRY=0
		        ELSE
				  HEL_INCONSISTENT=.TRUE. 
				ENDIF
			  ENDIF
			ENDIF
		  ENDIF
		ENDDO
	  ENDIF
	  IF(HEL_INCONSISTENT) THEN
C This helicity has unstable filter so we will always compute it by itself.
C We therefore also need to remove it from the multiplicative factor of the corresponding helicity.
	    IF(GOODHEL(HELPICKED).LT.-HELOFFSET) THEN
	      GOODHEL(-GOODHEL(HELPICKED)-HELOFFSET)=GOODHEL(-GOODHEL(HELPICKED)-HELOFFSET)-1
	    ENDIF
C If several helicities were matched to that one, we need to chose another one as reference and redirect the others to this new one
C Of course if it is one, then we do not need to do anything (because with HELINITSTARTOVER=.FALSE. we only support exactly identical Hels.)
	    IF(GOODHEL(HELPICKED).GT.-HELOFFSET.AND.GOODHEL(HELPICKED).NE.1) THEN
		  NEWHELREF=-1
		  DO H=1,NCOMB
		    IF (GOODHEL(H).EQ.(-HELOFFSET-HELPICKED)) THEN
			  IF (NEWHELREF.EQ.-1) THEN
			    NEWHELREF=H
				GOODHEL(H)=GOODHEL(HELPICKED)-1
			  ELSE
				GOODHEL(H)=-NEWHELREF-HELOFFSET
			  ENDIF
			ENDIF
		  ENDDO
		ENDIF
C In all cases, from now on this helicity will be computed independantly of the others.
C In particular, it is the only thing to do if the helicity was flagged not contributing.
		GOODHEL(HELPICKED)=1
	  ENDIF
    ENDIF

C   SET THE LOOP FILTER
    IF(.NOT.FOUNDLOOPFILTER.AND.USELOOPFILTER) THEN
  	  DO I=1,NLOOPGROUPS
	    DO J=1,NSQUAREDSO
          IF(.NOT.%(proc_prefix)sIsZero(ABS(LOOPRES(1,J,I))+ABS(LOOPRES(2,J,I))+ABS(LOOPRES(3,J,I)),(REF*1.0d-4),I,J)) THEN
            IF(NTRY.EQ.1) THEN
	          GOODAMP(J,I)=.TRUE.
		      LOOPFILTERBUFF(J,I)=.TRUE.
	        ELSEIF(.NOT.LOOPFILTERBUFF(J,I)) THEN
	          WRITE(*,*) '##W02 WARNING Inconsistent loop amp ',I,'.'
		      IF(LOOPINITSTARTOVER) THEN
		        WRITE(*,*) '##I01 INFO Initialization starting over because of inconsistency in the loop filter setup.'
	            NTRY=0
		      ELSE
		        GOODAMP(J,I)=.TRUE.
		      ENDIF
	        ENDIF
          ENDIF
  	    ENDDO
	  ENDDO
    ENDIF
  ELSEIF (.NOT.HELDOUBLECHECKED.AND.NTRY.NE.0)THEN
C   DOUBLE CHECK THE HELICITY FILTER
    IF (GOODHEL(HELPICKED).EQ.-HELOFFSET) THEN
	  IF (.NOT.%(proc_prefix)sIsZero(ABS(ANS(1,0))+ABS(ANS(2,0))+ABS(ANS(3,0)),REF/DBLE(NCOMB),-1,-1)) THEN
	    write(*,*) '##W15 Helicity filter could not be successfully double checked.'
	    write(*,*) 'One reason for this is that you might have changed sensible parameters which affected what are the zero helicity configurations.'
	    write(*,*) 'MadLoop will try to reset the Helicity filter with the next PS points it receives.'
	    NTRY=0
	    OPEN(29,FILE=HelFilterFN,err=348)
348     CONTINUE
        CLOSE(29,STATUS='delete')
	  ENDIF
	ENDIF
	IF (GOODHEL(HELPICKED).LT.-HELOFFSET.AND.NTRY.NE.0) THEN
	  IF(%(proc_prefix)sISSAME(ANS(1,0),HELSAVED(1,ABS(GOODHEL(HELPICKED)+HELOFFSET)),REF,.TRUE.).EQ.0) THEN
	    write(*,*) '##W15 Helicity filter could not be successfully double checked.'
	    write(*,*) 'One reason for this is that you might have changed sensible parameters which affected the helicity dependance relations.'
	    write(*,*) 'MadLoop will try to reset the Helicity filter with the next PS points it receives.'
	    NTRY=0
	    OPEN(30,FILE=HelFilterFN,err=349)
349     CONTINUE
        CLOSE(30,STATUS='delete')
	  ENDIF
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
C   Useful printout
c    do I=1,NCOMB
c      write(*,*) 'HELSAVED(1,',I,')=',HELSAVED(1,I)
c      write(*,*) 'HELSAVED(2,',I,')=',HELSAVED(2,I)
c      write(*,*) 'HELSAVED(3,',I,')=',HELSAVED(3,I)
c      write(*,*) '   GOODHEL(',I,')=',GOODHEL(I)
c    ENDDO
    DO I=0,NSQUAREDSO
      DO K=1,3
        ANS(K,I)=BUFFR(K,I)
      ENDDO
	ENDDO
	IF(NTRY.EQ.0) THEN
	  NATTEMPTS=NATTEMPTS+1
	  IF(NATTEMPTS.EQ.MAXATTEMPTS) THEN
	    WRITE(*,*) '##E01 ERROR Could not initialize the filters in ',MAXATTEMPTS,' trials'
		STOP 1
	  ENDIF
	ENDIF
  ENDIF

ENDIF

DO K=1,3
  DO I=0,NSQUAREDSO
    ANS(K,I)=ANS(K,I)/DBLE(IDEN)
    IF (USERHEL.NE.-1) THEN
      ANS(K,I)=ANS(K,I)*HELAVGFACTOR
    ENDIF
  ENDDO
ENDDO

IF(.NOT.CHECKPHASE.AND.HELDOUBLECHECKED.AND.(CTMODERUN.EQ.-1)) THEN
  STAB_INDEX=STAB_INDEX+1  
  IF(DOING_QP_EVALS.AND.MLReductionLib(I_LIB).EQ.1) THEN
C NOW,ONLY CUTTOOLS PROVIDES QP
    DO I=0,NSQUAREDSO
	  DO K=1,3
        QP_RES(K,I,STAB_INDEX)=ANS(K,I)
      ENDDO
	ENDDO
  ELSE
    DO I=0,NSQUAREDSO
	  DO K=1,3
        DP_RES(K,I,STAB_INDEX)=ANS(K,I)
	  ENDDO
	ENDDO
  ENDIF

  IF(DOING_QP_EVALS.AND.MLReductionLib(I_LIB).EQ.1) THEN	
      BASIC_CT_MODE=4
  ELSE
      BASIC_CT_MODE=1
  ENDIF

C BEGINNING OF THE DEFINITIONS OF THE DIFFERENT EVALUATION METHODS

  IF(.NOT.EVAL_DONE(2)) THEN
	EVAL_DONE(2)=.TRUE.
	IF(LOOPLIBS_DIRECTEST(MLReductionLib(I_LIB)))THEN 
	CTMODE=BASIC_CT_MODE+1
	goto 300
	ELSE
C   NO DIRECTION TEST LIBRARIES: e.g. PJFry++ and Golem95
	STAB_INDEX=STAB_INDEX+1
	IF(DOING_QP_EVALS)THEN
	DO I=0,NSQUAREDSO
	DO K=1,3
	  QP_RES(K,I,STAB_INDEX)=ANS(K,I)
	ENDDO
	ENDDO
	ELSE
	DO I=0,NSQUAREDSO
       	DO K=1,3
          DP_RES(K,I,STAB_INDEX)=ANS(K,I)
       	ENDDO
       	ENDDO
	ENDIF
	ENDIF
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

  IF(DOING_QP_EVALS.AND.MLReductionLib(I_LIB).EQ.1) THEN
    CALL %(proc_prefix)sCOMPUTE_ACCURACY(QP_RES,N_QP_EVAL,ACC,ANS)
	DO I=0,NSQUAREDSO
	  ACCURACY(I)=ACC(I)
	ENDDO
	RET_CODE_H=3
	RET_CODE_U=SET_RET_CODE_U(MLReductionLib(I_LIB),.TRUE.,.TRUE.)	
	IF(MAXVAL(ACC).GE.MLSTABTHRES) THEN
	  I_QP_LIB=I_QP_LIB+1
	  IF(I_QP_LIB.GT.QP_NLOOPLIB.OR.INDEX_QP_TOOLS(I_QP_LIB).EQ.0)THEN
	  RET_CODE_H=4
	  RET_CODE_U=SET_RET_CODE_U(MLReductionLib(I_LIB),.TRUE.,.FALSE.)
	  NEPS=NEPS+1
      CALL %(proc_prefix)sCOMPUTE_ACCURACY(DP_RES,N_DP_EVAL,TEMP1,TEMP)
	  IF(NEPS.LE.10) THEN	  
        WRITE(*,*) '##W03 WARNING An unstable PS point was',       ' detected.'
		IF (NSQUAREDSO.NE.1) THEN
		  WRITE(*,*) 'Accuracies for each split order, starting with the summed case'
	      WRITE(*,*) 'DP accuracies (for each split order): ',(TEMP1(I),I=0,NSQUAREDSO)
	      WRITE(*,*) 'QP accuracies (for each split order): ',(ACC(I),I=0,NSQUAREDSO)
		ELSE
	      WRITE(*,*) 'DP accuracy: ',TEMP1(1)
	      WRITE(*,*) 'QP accuracy: ',ACC(1)
		ENDIF
	    DO J=0,NSQUAREDSO
		  IF (NSQUAREDSO.NE.1.OR.J.NE.0) THEN
		    IF (J.EQ.0) THEN
		      WRITE(*,*) 'Details for all split orders summed :'
	        ELSE
		      WRITE(*,*) 'Details for split order index : ',J
		    ENDIF
	        WRITE(*,*) 'Best estimate (fin,1eps,2eps):',(ANS(I,J),I=1,3)
	        WRITE(*,*) 'Finite double precision evaluations :',(DP_RES(1,J,I),I=1,N_DP_EVAL)
	        WRITE(*,*) 'Finite quad   precision evaluations :',(QP_RES(1,J,I),I=1,N_QP_EVAL)
		  ENDIF
		ENDDO
	    WRITE(*,*) 'PS point specification :'
	    WRITE(*,*) 'Renormalization scale MU_R=',MU_R	
	    DO I=1,NEXTERNAL
          WRITE (*,'(i2,1x,4e27.17)') i, P(0,i),P(1,i),P(2,i),P(3,i) 
        ENDDO
	  ENDIF
	  IF(NEPS.EQ.10) THEN
	    WRITE(*,*) 'Further output of the details of these unstable PS points will now be suppressed.'
	  ENDIF
    ELSE
     I_LIB=INDEX_QP_TOOLS(I_QP_LIB)
     EVAL_DONE(1)=.TRUE.
     DO I=2,MAXSTABILITYLENGTH
     EVAL_DONE(I)=.FALSE.
     ENDDO
     STAB_INDEX=0
     IF(NRotations_QP.GE.1)THEN
	goto 200
     ELSE
	goto 300
     ENDIF
    ENDIF
    ENDIF
  ELSEIF(.NOT.DOING_QP_EVALS)THEN
    CALL %(proc_prefix)sCOMPUTE_ACCURACY(DP_RES,N_DP_EVAL,ACC,ANS)
	IF(MAXVAL(ACC).GE.MLSTABTHRES) THEN
	  I_LIB=I_LIB+1
	  IF((I_LIB.GT.NLOOPLIB.OR.MLReductionLib(I_LIB).EQ.0).AND.QP_TOOLS_AVAILABLE)THEN
	        I_LIB=INDEX_QP_TOOLS(1)
		I_QP_LIB=1
		DOING_QP_EVALS=.TRUE.
	  	EVAL_DONE(1)=.TRUE.
	  	DO I=2,MAXSTABILITYLENGTH
        	EVAL_DONE(I)=.FALSE.
      		ENDDO
	  	STAB_INDEX=0
		CTMODE=4
	  	goto 200
	  ELSEIF(I_LIB.LE.NLOOPLIB.AND.MLReductionLib(I_LIB).GT.0)THEN
	        EVAL_DONE(1)=.TRUE.
		DO I=2,MAXSTABILITYLENGTH
		EVAL_DONE(I)=.FALSE.
		ENDDO
		STAB_INDEX=0
		IF(NRotations_DP.GE.1)THEN
		   goto 200
		ELSE
		   goto 300
		ENDIF
	  ELSE
	     DO I=0,NSQUAREDSO
	       ACCURACY(I)=ACC(I)
	     ENDDO
	     RET_CODE_H=4
	     RET_CODE_U=SET_RET_CODE_U(MLReductionLib(I_LIB),.FALSE.,.FALSE.)
	     NEPS=NEPS+1
             IF(NEPS.LE.10) THEN
             WRITE(*,*) '##W03 WARNING An unstable PS point was',       ' detected.'
	     WRITE(*,*) '##W03 WARNING No quadruple precision will be used.'
              IF (NSQUAREDSO.NE.1) THEN
                  WRITE(*,*) 'Accuracies for each split order, starting with the summed case'
              	  WRITE(*,*) 'DP accuracies (for each split order): ',(ACC(I),I=0,NSQUAREDSO)
              ELSE
              	WRITE(*,*) 'DP accuracy: ',ACC(1)
	      ENDIF
              DO J=0,NSQUAREDSO
                  IF (NSQUAREDSO.NE.1.OR.J.NE.0) THEN
                    IF (J.EQ.0) THEN
                      WRITE(*,*) 'Details for all split orders summed :'
                    ELSE
                      WRITE(*,*) 'Details for split order index : ',J
                    ENDIF
                    WRITE(*,*) 'Best estimate (fin,1eps,2eps):',(ANS(I,J),I=1,3)
               	     WRITE(*,*) 'Finite double precision evaluations :',(DP_RES(1,J,I),I=1,N_DP_EVAL)
                  ENDIF
                ENDDO
            WRITE(*,*) 'PS point specification :'
            WRITE(*,*) 'Renormalization scale MU_R=',MU_R
            DO I=1,NEXTERNAL
            WRITE (*,'(i2,1x,4e27.17)') i, P(0,i),P(1,i),P(2,i),P(3,i)
            ENDDO
            ENDIF
            IF(NEPS.EQ.10) THEN
             WRITE(*,*) 'Further output of the details of these unstable PS points will now be suppressed.'
            ENDIF
	  ENDIF
    ELSE
	  DO I=0,NSQUAREDSO
	    ACCURACY(I)=ACC(I)
	  ENDDO
      	  RET_CODE_H=2
	  RET_CODE_U=SET_RET_CODE_U(MLReductionLib(I_LIB),.FALSE.,.TRUE.)	
	  ENDIF	
  ENDIF
ELSE
  RET_CODE_H=1
  DO I=0,NSQUAREDSO
    ACCURACY(I)=-1.0d0
  ENDDO
  RET_CODE_U=SET_RET_CODE_U(MLReductionLib(I_LIB),.FALSE.,.FALSE.)
ENDIF

 9999 CONTINUE

C Finalize the return code
IF (MP_DONE_ONCE) THEN
  RET_CODE_T=2
ELSE
  RET_CODE_T=1
ENDIF
IF(CHECKPHASE.OR..NOT.HELDOUBLECHECKED) THEN
  RET_CODE_H=1
  RET_CODE_U=SET_RET_CODE_U(MLReductionLib(I_LIB),.FALSE.,.FALSE.)
  RET_CODE_T=RET_CODE_T+2
  DO I=0,NSQUAREDSO
    ACCURACY(I)=-1.0d0
  ENDDO
ENDIF

C Finally for the summed result in ANS(1:3,0), make sure to only
c consider the squared order asked for by the user.
C Notice that this filtering using CHOSEN_SO_CONFIGS happens
C here only while everywhere else one always considers the sum.
DO J=1,3
  ANS(J,0)=0.0d0
ENDDO	
DO I=1,NSQUAREDSO
  IF (CHOSEN_SO_CONFIGS(I)) THEN
    DO J=1,3
	  ANS(J,0)=ANS(J,0)+ANS(J,I)
    ENDDO
  ENDIF
ENDDO

C Reinitialize the default threshold if it was specified by the user
IF (USER_STAB_PREC.GT.0.0d0) THEN
  MLSTABTHRES=MLSTABTHRES_BU
  CTMODEINIT=CTMODEINIT_BU  
ENDIF

C Reinitialize the check phase logicals and the filters if check bypassed
IF (BYPASS_CHECK) THEN
  CHECKPHASE = OLD_CHECKPHASE
  HELDOUBLECHECKED = OLD_HELDOUBLECHECKED
  DO I=1,NCOMB
	GOODHEL(I)=OLD_GOODHEL(I)
  ENDDO
  DO I=1,NSQUAREDSO
    DO J=1,NLOOPGROUPS
      GOODAMP(I,J)=OLD_GOODAMP(I,J)
    ENDDO
  ENDDO
ENDIF
      END

C --=========================================--
C   General Helper functions and subroutine
C   for the main sloopmatrix subroutine
C --=========================================--

	  logical function %(proc_prefix)sIsZero(toTest, reference_value, loop, soindex)
      IMPLICIT NONE
C  
C CONSTANTS
C
      INTEGER    NLOOPGROUPS
	  PARAMETER (NLOOPGROUPS=%(nloop_groups)d)
	  INTEGER    NSQUAREDSO
	  PARAMETER (NSQUAREDSO=%(nSquaredSO)d)
C  
C ARGUMENTS 
C  
	  %(real_dp_format)s toTest, reference_value
	  integer loop, soindex
C  
C GLOBAL 
C
      INCLUDE 'MadLoopParams.inc'
      %(complex_dp_format)s LOOPRES(3,NSQUAREDSO,NLOOPGROUPS)
	  LOGICAL S(NSQUAREDSO,NLOOPGROUPS)
      common/%(proc_prefix)sLOOPRES/LOOPRES,S
C ----------
C BEGIN CODE
C ----------
	  IF(abs(reference_value).eq.0.0d0) then
	    %(proc_prefix)sIsZero=.FALSE.
		write(*,*) '##E02 ERRROR Reference value for comparison is zero.'
		STOP 1
	  else
	    %(proc_prefix)sIsZero=((abs(toTest)/abs(reference_value)).lt.ZEROTHRES)
	  endif
      
	  IF(loop.NE.-1) THEN
	    IF((.NOT.%(proc_prefix)sIsZero).AND.(.NOT.S(soindex,loop))) THEN
	      write(*,*) '##W01 WARNING Contribution ',loop,' of split order ',soindex,' is detected as contributing with CR=',(abs(toTest)/abs(reference_value)),' but is unstable.' 
	    ENDIF
	  ENDIF

	  end

      integer function %(proc_prefix)sISSAME(resA,resB,ref,useMax)
	  IMPLICIT NONE
C     This function compares the result from two different helicity configuration A and B
C     It returns 0 if they are not related and (+/-wgt) if A=(+/-wgt)*B.
C     For now, the only wgt implemented is the integer 1 or -1.
C     If useMax is .TRUE., it uses all implemented weights no matter what is HELINITSTARTOVER
C   
C CONSTANTS
C
      integer MAX_WGT_TO_TRY
	  parameter (MAX_WGT_TO_TRY=2)
C  
C ARGUMENTS 
C  
	  %(real_dp_format)s resA(3), resB(3)
	  %(real_dp_format)s ref
	  logical useMax
C  
C LOCAL VARIABLES
C
      LOGICAL %(proc_prefix)sIsZero
      integer I,J
	  integer N_WGT_TO_TRY
      integer WGT_TO_TRY(MAX_WGT_TO_TRY)
	  data WGT_TO_TRY/1,-1/
C
C INCLUDES
C
	  include 'MadLoopParams.inc'
C ----------
C BEGIN CODE
C ----------
      %(proc_prefix)sISSAME=0

C If the helicity can be constructed progressively while allowing inconsistency, then we only allow for weight one comparisons.
	  IF (.NOT.HELINITSTARTOVER.AND..NOT.USEMAX) THEN
	    N_WGT_TO_TRY=1
	  ELSE
	    N_WGT_TO_TRY=MAX_WGT_TO_TRY
	  ENDIF

	  DO I=1,N_WGT_TO_TRY
	    DO J=1,3
		  IF (%(proc_prefix)sIsZero(ABS(resB(J)),ref,-1,-1)) THEN
		    IF(.NOT.%(proc_prefix)sIsZero(ABS(resB(J))+ABS(resA(J)),ref,-1,-1)) THEN
		      GOTO 1231
			ENDIF
C         Be loser for helicity comparison, so bring a factor 100
		  ELSEIF(.NOT.%(proc_prefix)sIsZero((resA(J)/resB(J))-DBLE(WGT_TO_TRY(I)),ref*100.0d0,-1,-1)) THEN
		    GOTO 1231		    
		  ENDIF
		ENDDO
		%(proc_prefix)sISSAME = WGT_TO_TRY(I)
		RETURN
 1231   CONTINUE
      ENDDO
	  END

      SUBROUTINE %(proc_prefix)scompute_accuracy(fulllist, length, acc, estimate)
      implicit none
C  
C PARAMETERS 
C
      integer maxstabilitylength
	  common/%(proc_prefix)sstability_tests/maxstabilitylength
      INTEGER    NSQUAREDSO
	  PARAMETER (NSQUAREDSO=%(nSquaredSO)d)
C  
C ARGUMENTS 
C
      real*8 fulllist(3,0:NSQUAREDSO,maxstabilitylength)
      integer length
      real*8 acc(0:NSQUAREDSO), estimate(0:3,0:NSQUAREDSO)
C  
C LOCAL VARIABLES 
C
      logical mask(maxstabilitylength)
	  logical mask3(3)
	  data mask3/.TRUE.,.TRUE.,.TRUE./
      integer i,j,k
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
C For some architectures, it is necessary to initialize all the elements of fulllist(i,j)
C Beware that if the length provided is incorrect, then this can corrup the fulllist given in argument.
        do j=0,NSQUAREDSO
          do k=1,3
		    fulllist(k,j,i)=0.0d0
		  enddo
		enddo
      enddo

      do k=0,NSQUAREDSO

	    do i=1,3
	      do j=1,maxstabilitylength
		    list(j)=fulllist(i,k,j)
		  enddo
          diff=maxval(list,1,mask)-minval(list,1,mask)
          avg=(maxval(list,1,mask)+minval(list,1,mask))/2.0d0
		  estimate(i,k)=avg
          if (avg.eq.0.0d0) then
            accuracies(i)=diff
          else
            accuracies(i)=diff/abs(avg)
          endif
	    enddo

C       The technique below is too sensitive, typically to
C       unstablities in very small poles
C       acc(k)=MAXVAL(ACCURACIES,1,MASK3)
C       The following is used instead
        acc(k) = 0.0d0
        AVG = 0.0d0
        DO I=1,3
          acc(k) = acc(k) + ACCURACIES(I)*ABS(ESTIMATE(I,K))
          AVG = AVG + ESTIMATE(I,K)
        ENDDO
		if (avg.ne.0.0d0) then
          acc(k)  = acc(k) / ( ABS(AVG) / 3.0d0)
		endif

	  enddo

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
	    WRITE(*,*) 'ERROR:: Increase hardcoded maxstabilitylength.'
		stop 1
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

C --=========================================--
C   Functions for dealing with the ordering
C   and indexing of split order contributions
C --=========================================--

      SUBROUTINE %(proc_prefix)sGET_NSQSO_LOOP(NSQSO)
C
C     Simple subroutine returning the number of squared split order
C     contributions returned in ANS when calling sloopmatrix 
C
      INTEGER    NSQUAREDSO
	  PARAMETER (NSQUAREDSO=%(nSquaredSO)d)

	  INTEGER NSQSO

	  NSQSO=NSQUAREDSO

      END

      SUBROUTINE %(proc_prefix)sGET_ANSWER_DIMENSION(ANS_DIM)
C
C     MadLoop subroutines return an array of dimension ANS(0:3,0:ANS_DIM)
C     In order for the user program to be able to correctly declare this
c     array when calling MadLoop, this subroutine returns its dimension
C
      INTEGER    NSQUAREDSO
	  PARAMETER (NSQUAREDSO=%(nSquaredSO)d)
	  INTEGER ANS_DIM 

	  %(get_nsqso_born)s 

	  ANS_DIM=MAX(NSQSO_BORN,NSQUAREDSO)

      END

      INTEGER FUNCTION %(proc_prefix)sML5SOINDEX_FOR_SQUARED_ORDERS(ORDERS)
C
C This functions returns the integer index identifying the split orders list passed in argument which correspond to the values of the following list of couplings (and in this order):
C %(split_order_str_list)s
C
C CONSTANTS
C
      INTEGER    NSO, NSQSO
	  PARAMETER (NSO=%(nSO)d, NSQSO=%(nSquaredSO)d)
C
C ARGUMENTS
C
      INTEGER ORDERS(NSO)
C
C LOCAL VARIABLES
C
      INTEGER I,J
	  INTEGER SQPLITORDERS(NSQSO,NSO)
	  %(SquaredSO)s
C
C BEGIN CODE
C
      DO I=1,NSQSO
	    DO J=1,NSO
		  IF (ORDERS(J).NE.SQPLITORDERS(I,J)) GOTO 1009
		ENDDO
		%(proc_prefix)sML5SOINDEX_FOR_SQUARED_ORDERS = I
		RETURN
1009    CONTINUE
	  ENDDO

	  WRITE(*,*) 'ERROR:: Stopping function %(proc_prefix)sML5SOINDEX_FOR_SQUARED_ORDERS'
	  WRITE(*,*) 'Could not find squared orders ',(ORDERS(I),I=1,NSO)
	  STOP

      END

      INTEGER FUNCTION %(proc_prefix)sML5SOINDEX_FOR_BORN_AMP(AMPID)
C
C For a given born amplitude number, it returns the ID of the split orders it has
C
C CONSTANTS
C
	  INTEGER NBORNAMPS
	  PARAMETER (NBORNAMPS=%(nBornAmps)d)
C
C ARGUMENTS
C
      INTEGER AMPID
C
C LOCAL VARIABLES
C
	  INTEGER BORNAMPORDERS(NBORNAMPS)
	  %(BornAmpSO)s
C -----------
C BEGIN CODE
C -----------
	  IF (AMPID.gt.NBORNAMPS) THEN
	    WRITE(*,*) 'ERROR:: Born amplitude ID ',AMPID,' above the maximum ',NBORNAMPS	
      ENDIF
	  %(proc_prefix)sML5SOINDEX_FOR_BORN_AMP = BORNAMPORDERS(AMPID)

      END

      INTEGER FUNCTION %(proc_prefix)sML5SOINDEX_FOR_LOOP_AMP(AMPID)
C
C For a given loop amplitude number, it returns the ID of the split orders it has
C
C CONSTANTS
C
      INTEGER    NLOOPAMPS
      PARAMETER (NLOOPAMPS=%(nloopamps)d)
C
C ARGUMENTS
C
      INTEGER AMPID
C
C LOCAL VARIABLES
C
      INTEGER LOOPAMPORDERS(NLOOPAMPS)
	  %(loopAmpSO)s
C -----------
C BEGIN CODE
C -----------
	  IF (AMPID.gt.NLOOPAMPS) THEN
	    WRITE(*,*) 'ERROR:: Loop amplitude ID ',AMPID,' above the maximum ',NLOOPAMPS	
	  ENDIF
	  %(proc_prefix)sML5SOINDEX_FOR_LOOP_AMP = LOOPAMPORDERS(AMPID)

      END


      INTEGER FUNCTION %(proc_prefix)sML5SQSOINDEX(ORDERINDEXA, ORDERINDEXB)
C
C This functions plays the role of the interference matrix. It can be hardcoded or 
C made more elegant using hashtables if its execution speed ever becomes a relevant
C factor. From two split order indices, it return the corresponding index in the squared 
c order canonical ordering.
C
C CONSTANTS
C      

      INTEGER    NSO, NSQUAREDSO, NAMPSO
	  PARAMETER (NSO=%(nSO)d, NSQUAREDSO=%(nSquaredSO)d, NAMPSO=%(nAmpSO)d)
C
C ARGUMENTS
C
	  INTEGER ORDERINDEXA, ORDERINDEXB
C
C LOCAL VARIABLES
C
      INTEGER I, SQORDERS(NSO)
      INTEGER AMPSPLITORDERS(NAMPSO,NSO)
	  %(ampsplitorders)s
C
C FUNCTION
C
      INTEGER %(proc_prefix)sML5SOINDEX_FOR_SQUARED_ORDERS
C
C BEGIN CODE
C
      DO I=1,NSO
	    SQORDERS(I)=AMPSPLITORDERS(ORDERINDEXA,I)+AMPSPLITORDERS(ORDERINDEXB,I)
	  ENDDO
	  %(proc_prefix)sML5SQSOINDEX=%(proc_prefix)sML5SOINDEX_FOR_SQUARED_ORDERS(SQORDERS)
	  END

C --=========================================--
C   Definition of additional access routines
C --=========================================--

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

	  SUBROUTINE %(proc_prefix)sSET_COUPLINGORDERS_TARGET(SOTARGET)
	  IMPLICIT NONE
C
C     This routine can be accessed by an external user to set the squared split order target.
C     If set to a value different than -1, the code will try to avoid computing anything which
C     does not contribute to contributions of squared split orders SQSO_TARGET and below.
C     This can considerably speed up the code. However, keep in mind that any contribution of
C     'squared order index' larger than SQSO_TARGET cannot be trust.
C
C     ARGUMENTS
C
      INTEGER SOTARGET
C
C     GLOBAL
C
	  INTEGER SQSO_TARGET
	  common/%(proc_prefix)sSOCHOICE/SQSO_TARGET
C ----------
C BEGIN CODE
C ----------
      SQSO_TARGET = SOTARGET
	  END

      SUBROUTINE %(proc_prefix)sSLOOPMATRIXHEL(P,HEL,ANS)
      IMPLICIT NONE
C  
C CONSTANTS
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER    NSQUAREDSO
      PARAMETER (NSQUAREDSO=%(nSquaredSO)d)
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

      SUBROUTINE %(proc_prefix)sSLOOPMATRIXHEL_THRES(P,HEL,ANS,PREC_ASKED,PREC_FOUND,RET_CODE)
	  IMPLICIT NONE
C  
C CONSTANTS
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER    NSQUAREDSO
      PARAMETER (NSQUAREDSO=%(nSquaredSO)d)
C  
C ARGUMENTS 
C  
      %(real_dp_format)s P(0:3,NEXTERNAL)
      %(real_dp_format)s ANS(0:3,0:NSQUAREDSO)
	  INTEGER HEL, RET_CODE
	  %(real_dp_format)s PREC_ASKED,PREC_FOUND(0:NSQUAREDSO)
C
C LOCAL VARIABLES
C
      INTEGER I
C
C GLOBAL VARIABLES
C
	  %(real_dp_format)s USER_STAB_PREC
	  COMMON/%(proc_prefix)sUSER_STAB_PREC/USER_STAB_PREC

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
C       Make sure we correctly return an initialization-type T code
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
C     U == 0
C         Not stable.
C     U == 1
C         Stable with CutTools in double precision.
C     U == 2
C         Stable with PJFry++.
C     U == 3
C         Stable with IREGI.
C     U == 4
C         Stable with Golem95
C     U == 9
C         Stable with CutTools in quadruple precision.         
C
      IMPLICIT NONE
C  
C CONSTANTS
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER    NSQUAREDSO
      PARAMETER (NSQUAREDSO=%(nSquaredSO)d)
C  
C ARGUMENTS 
C  
      %(real_dp_format)s P(0:3,NEXTERNAL)
      %(real_dp_format)s ANS(0:3,0:NSQUAREDSO)
	  %(real_dp_format)s PREC_ASKED,PREC_FOUND(0:NSQUAREDSO)
	  INTEGER RET_CODE	  
C
C LOCAL VARIABLES
C
      INTEGER I
C
C GLOBAL VARIABLES
C
	  %(real_dp_format)s USER_STAB_PREC
	  COMMON/%(proc_prefix)sUSER_STAB_PREC/USER_STAB_PREC

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
