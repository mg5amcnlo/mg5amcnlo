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
C Modules
C
      use %(proc_prefix)sPOLYNOMIAL_CONSTANTS      
C
      IMPLICIT NONE
C
C USER CUSTOMIZABLE OPTIONS
C 
## if(ComputeColorFlows) {
C The variables below are just used in the context of a JAMP consistency check turned off by default.
      %(real_dp_format)s JAMP_DOUBLECHECK_THRES
      PARAMETER (JAMP_DOUBLECHECK_THRES=1.0d-9)
      LOGICAL DIRECT_ME_COMPUTATION, ME_COMPUTATION_FROM_JAMP
C     Modify the logicals below to chose how the ME must be computed
C        DIRECT_ME_COMPUTATION    = Each loop amplitude is squared individually against all amplitudes with its own color factor.
C        ME_COMPUTATION_FROM_JAMP = Amplitudes are first projected onto color flows (many less of them) which are then squared to form the ME.
C     When setting both computation method to .TRUE., their systematic comparisons will be printed out.
## if(LoopInduced) {
      DATA DIRECT_ME_COMPUTATION/.FALSE./
c     When using this MadLoop output for integration with MadEvent, ME_COMPUTATION_FROM_JAMP *must* be set to .True. because it is necessary to compute the AMP2 setting up the multichanneling. 
      DATA ME_COMPUTATION_FROM_JAMP/.TRUE./
## } else {
      DATA DIRECT_ME_COMPUTATION/.TRUE./
      DATA ME_COMPUTATION_FROM_JAMP/.FALSE./
## }
## }
C     This parameter is designed for the check timing command of MG5. It skips the loop reduction.
      LOGICAL SKIPLOOPEVAL
      PARAMETER (SKIPLOOPEVAL=.FALSE.)
C     For timing checks. Stops the code after having only initialized its arrays from the external data files 
	  LOGICAL BOOTANDSTOP
      PARAMETER (BOOTANDSTOP=.FALSE.)
## if(TIRCaching) {
      INTEGER TIR_CACHE_SIZE
C To change memory foot-print of MadLoop, you can change this parameter to be 0,1 or 2 *and recompile*.
C Notice that this will impact MadLoop speed performances in the context of stability checks.
      include 'tir_cache_size.inc'
## }
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

## if (not LoopInduced) {
	  INTEGER NBORNAMPS
      PARAMETER (NBORNAMPS=%(nbornamps)d)
## }
      INTEGER    NLOOPS, NLOOPGROUPS, NCTAMPS
      PARAMETER (NLOOPS=%(nloops)d, NLOOPGROUPS=%(nloop_groups)d, NCTAMPS=%(nctamps)d)
      INTEGER    NLOOPAMPS
      PARAMETER (NLOOPAMPS=%(nloopamps)d)
      INTEGER    NCOLORROWS
	  PARAMETER (NCOLORROWS=NLOOPAMPS)
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER    NWAVEFUNCS,NLOOPWAVEFUNCS
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d,NLOOPWAVEFUNCS=%(nloopwavefuncs)d)
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
C     At present, there are only CutTools,PJFry++,IREGI,Golem95,Samurai and Ninja
      INTEGER NLOOPLIB
      PARAMETER (NLOOPLIB=6)
C     Only CutTools or possibly Ninja (if installed with qp support) provide QP
      INTEGER QP_NLOOPLIB
## if(ninja_supports_quad_prec) {
      PARAMETER (QP_NLOOPLIB=2)
## } else {
      PARAMETER (QP_NLOOPLIB=1)
## }
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
      INTEGER I,J,K,H,HEL_MULT,I_QP_LIB,DUMMY

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

## if(LoopInduced){	  
C This is used for loop-induced where the reference scale for comparisons is inferred from the first 100 points at most (notice that the weight of a  given kinematic configuration can appear more than once because of the stability tests).
C When changing this parameter, make sure to correspondingly update the parameter with the same name in MadLoopCommons.f.
      INTEGER MAXNREF_EVALS
      PARAMETER (MAXNREF_EVALS=100)
      %(real_dp_format)s REF_EVALS(MAXNREF_EVALS)
      DATA REF_EVALS/MAXNREF_EVALS*ZERO/
      INTEGER NPSPOINTS
      DATA NPSPOINTS/0/      
## }

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
	  %(real_dp_format)s BORNBUFF(0:NSQSO_BORN),TMPR
	  %(real_dp_format)s BUFFR(3,0:NSQUAREDSO),BUFFR_BIS(3,0:NSQUAREDSO),TEMP(0:3,0:NSQUAREDSO),TEMP1(0:NSQUAREDSO)
	  %(complex_dp_format)s CTEMP
## if(not AmplitudeReduction){
	  %(real_dp_format)s TEMP2
## }else{
	  %(real_dp_format)s TEMP2(3)
## }
## if(ComputeColorFlows) {
	  %(real_dp_format)s BUFFRES(0:3,0:NSQUAREDSO)
## }
      %(complex_dp_format)s COEFS(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)
      %(complex_dp_format)s CFTOT
	  LOGICAL FOUNDHELFILTER,FOUNDLOOPFILTER
	  DATA FOUNDHELFILTER/.TRUE./
	  DATA FOUNDLOOPFILTER/.TRUE./
      LOGICAL LOOPFILTERBUFF(NSQUAREDSO,NLOOPGROUPS)
	  DATA ((LOOPFILTERBUFF(J,I),J=1,NSQUAREDSO),I=1,NLOOPGROUPS)/NSQSOXNLG*.FALSE./

      LOGICAL AUTOMATIC_CACHE_CLEARING
      DATA AUTOMATIC_CACHE_CLEARING/.TRUE./
      COMMON/%(proc_prefix)sRUNTIME_OPTIONS/AUTOMATIC_CACHE_CLEARING

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
	  INTEGER %(proc_prefix)sTIRCACHE_INDEX
      INTEGER %(proc_prefix)sML5SOINDEX_FOR_BORN_AMP
	  INTEGER %(proc_prefix)sML5SOINDEX_FOR_LOOP_AMP
	  INTEGER %(proc_prefix)sML5SQSOINDEX
      INTEGER %(proc_prefix)sISSAME
      LOGICAL %(proc_prefix)sISZERO
      INTEGER SET_RET_CODE_U
## if(LoopInduced){
	  %(real_dp_format)s Median
## }
C  
C GLOBAL VARIABLES
C 
      include 'process_info.inc'
      include 'coupl.inc'
	  include 'mp_coupl.inc'
	  include 'MadLoopParams.inc'

## if(ComputeColorFlows) {
      %(real_dp_format)s RES_FROM_JAMP(0:3,0:NSQUAREDSO)
	  common/%(proc_prefix)sDOUBLECHECK_JAMP/RES_FROM_JAMP,DIRECT_ME_COMPUTATION,ME_COMPUTATION_FROM_JAMP
## }

      LOGICAL CHOSEN_SO_CONFIGS(NSQUAREDSO)
	  DATA CHOSEN_SO_CONFIGS/%(chosen_so_configs)s/
      COMMON/%(proc_prefix)sCHOSEN_LOOP_SQSO/CHOSEN_SO_CONFIGS

	  INTEGER N_DP_EVAL, N_QP_EVAL
	  DATA N_DP_EVAL/1/
	  DATA N_QP_EVAL/1/
	  COMMON/%(proc_prefix)sN_EVALS/N_DP_EVAL,N_QP_EVAL

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
      LOGICAL LOOPLIBS_AVAILABLE(6)
      DATA LOOPLIBS_AVAILABLE/%(data_looplibs_av)s/
	  common/%(proc_prefix)sLOOPLIBS_AV/ LOOPLIBS_AVAILABLE
C     A FLAG TO DENOTE WHETHER THE CORRESPONDING DIRECTION TESTS AVAILABLE OR NOT IN THE LOOPLIBS
      LOGICAL LOOPLIBS_DIRECTEST(6)
	  DATA LOOPLIBS_DIRECTEST /.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE./
C     Specifying for which reduction tool quadruple precision is available.
C     The index 0 is dummy and simply means that the corresponding loop_library is not available
C     in which case neither is its quadruple precision version.
      LOGICAL LOOPLIBS_QPAVAILABLE(0:6)
## if(ninja_supports_quad_prec) {
	  DATA LOOPLIBS_QPAVAILABLE /.FALSE.,.TRUE.,.FALSE.,.FALSE.,.FALSE.,.FALSE.,.TRUE./
## } else {
	  DATA LOOPLIBS_QPAVAILABLE /.FALSE.,.TRUE.,.FALSE.,.FALSE.,.FALSE.,.FALSE.,.FALSE./
## }
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

C     Allows to forbid the zero helicity double check, no matter the value in MadLoopParams.dat
C     This can be accessed with the SET_FORBID_HEL_DOUBLECHECK subroutine of MadLoopCommons.dat
      LOGICAL FORBID_HEL_DOUBLECHECK
	  COMMON/FORBID_HEL_DOUBLECHECK/FORBID_HEL_DOUBLECHECK

	  integer I_SO
	  DATA I_SO/1/
	  common/%(proc_prefix)sI_SO/I_SO
	  integer I_LIB
	  DATA I_LIB/1/
	  common/%(proc_prefix)sI_LIB/I_LIB
C UTIL NOW, ONLY CUTTOOLS PROVIDE QP
       	  LOGICAL QP_TOOLS_AVAILABLE
       	  DATA QP_TOOLS_AVAILABLE/.FALSE./
       	  INTEGER INDEX_QP_TOOLS(QP_NLOOPLIB+1)
       	  common/%(proc_prefix)sLOOP_TOOLS/QP_TOOLS_AVAILABLE,INDEX_QP_TOOLS

## if(not LoopInduced) {
	  %(complex_dp_format)s AMP(NBORNAMPS)
	  common/%(proc_prefix)sAMPS/AMP
## }
	  %(complex_dp_format)s W(20,NWAVEFUNCS)
	  common/%(proc_prefix)sW/W  

      %(complex_mp_format)s MPW(20,NWAVEFUNCS)
	  common/%(proc_prefix)sMP_W/MPW

	  %(complex_dp_format)s WL(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE,0:NLOOPWAVEFUNCS)
	  %(complex_dp_format)s PL(0:3,0:NLOOPWAVEFUNCS)
	  common/%(proc_prefix)sWL/WL,PL

## if(not AmplitudeReduction){
	  %(complex_dp_format)s LOOPCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)
## } else {
	  %(complex_dp_format)s LOOPCOEFS(0:LOOPMAXCOEFS-1,NLOOPGROUPS)
## }
	  common/%(proc_prefix)sLCOEFS/LOOPCOEFS

## if(AmplitudeReduction) {
C This flag is used to prevent the re-computation of the OpenLoop coefficients when changing the CTMode for the stability test.
	  LOGICAL SKIP_LOOPNUM_COEFS_CONSTRUCTION
	  DATA SKIP_LOOPNUM_COEFS_CONSTRUCTION/.FALSE./
	  COMMON/%(proc_prefix)sSKIP_COEFS/SKIP_LOOPNUM_COEFS_CONSTRUCTION
## }

## if(TIRCaching){
	  LOGICAL TIR_DONE(NLOOPGROUPS)
	  COMMON/%(proc_prefix)sTIRCACHING/TIR_DONE
## }

## if(not AmplitudeReduction){
      %(complex_dp_format)s AMPL(3,NCTAMPS)
## } else {
      %(complex_dp_format)s AMPL(3,NLOOPAMPS)
## }
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

C     This variable controls the general initialization which is *common* between all MadLoop SubProcesses.
C     For example setting the MadLoopPath or reading the ML runtime parameters.
	  LOGICAL ML_INIT
	  common/ML_INIT/ML_INIT

C     This variable controls the *local* initialization of this particular SubProcess.
C     For example, the reading of the filters must be done independently by each SubProcess.
      LOGICAL LOCAL_ML_INIT
	  data LOCAL_ML_INIT/.TRUE./

      LOGICAL WARNED_LORENTZ_STAB_TEST_OFF
	  data WARNED_LORENTZ_STAB_TEST_OFF/.FALSE./
	  INTEGER NROTATIONS_DP_BU,NROTATIONS_QP_BU

      LOGICAL FPE_IN_DP_REDUCTION, FPE_IN_QP_REDUCTION
	  data FPE_IN_DP_REDUCTION, FPE_IN_QP_REDUCTION/.FALSE.,.FALSE./      
	  COMMON/%(proc_prefix)sFPE_IN_REDUCTION/FPE_IN_DP_REDUCTION, FPE_IN_QP_REDUCTION
C ----------
C BEGIN CODE
C ----------

IF(ML_INIT) THEN
  ML_INIT = .FALSE.
  CALL PRINT_MADLOOP_BANNER()
  TMP = 'auto'
  CALL setMadLoopPath(TMP)
  CALL JOINPATH(MLPATH,PARAMFNAME,PARAMFN)
  CALL MADLOOPPARAMREADER(PARAMFN,.TRUE.)
  IF (FORBID_HEL_DOUBLECHECK) THEN
    DoubleCheckHelicityFilter = .False. 
  ENDIF 
## if(LoopInduced){
  IF(.NOT.LoopInitStartOver) THEN
    WRITE(*,*) '##INFO: For loop-induced processes it is preferable to always set the parameter LoopInitStartOver to True, so it is hard-set here to True.'
    LoopInitStartOver=.TRUE.
  ENDIF
  IF(.NOT.HelInitStartOver) THEN
    WRITE(*,*) "##INFO: For loop-induced processes it is preferable to always set the parameter HelInitStartOver to True, so it is hard-set here to True.'
    HelInitStartOver=.TRUE.
  ENDIF
  IF (CheckCycle.LT.5) THEN
    WRITE(*,*) "##INFO: Due to the dynamic setting of the reference scale for contributions comparisons, it is preferable to set the parameter CheckCycle to a value larger than 4, so it is hard-set here to 5.'
    CheckCycle=5
  ENDIF
## }

C Make sure that NROTATIONS_QP and NROTATIONS_DP are set to zero if AUTOMATIC_CACHE_CLEARING is disabled.
  if(.NOT.AUTOMATIC_CACHE_CLEARING) THEN
    IF(NROTATIONS_DP.NE.0.or.NROTATIONS_QP.NE.0) THEN
      WRITE(*,*) '##INFO: AUTOMATIC_CACHE_CLEARING is disabled, so MadLoop automatically resets NROTATIONS_DP and NROTATIONS_QP to 0.'
      NROTATIONS_QP=0
      NROTATIONS_DP=0
    ENDIF
  ENDIF

ENDIF

IF (LOCAL_ML_INIT) THEN
  LOCAL_ML_INIT = .FALSE.
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
     IF(LOOPLIBS_QPAVAILABLE(MLReductionLib(I)))THEN
      J=J+1
      IF(.NOT.QP_TOOLS_AVAILABLE) THEN
	    QP_TOOLS_AVAILABLE=.TRUE.
      ENDIF
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

C Make sure that the loop filter is disabled when there is spin-2 particles for 2>1 or 1>2 processes
  if(MAX_SPIN_EXTERNAL_PARTICLE.gt.3.AND.(NEXTERNAL.LE.3.AND.HelicityFilterLevel.NE.0)) THEN
    WRITE(*,*) '##INFO: Helicity filter deactivated for 2>1 processes involving spin 2 particles.'
    HelicityFilterLevel = 0	
C   We write a dummy filter for structural reasons here
    OPEN(1, FILE=HelFilterFN, err=6116, status='NEW',action='WRITE')
    DO I=1,NCOMB
      WRITE(1,*) 1  
    ENDDO
6116  CONTINUE
    CLOSE(1)
  ENDIF

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
    PL(I,0)=DCMPLX(0.0d0,0.0d0)
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
    WRITE(*,*) '##Stopped by user request.'
    stop
  ENDIF
ENDIF

C Make sure that lorentz rotation tests are not used if there is external loop wavefunction of spin 2 and that one specific helicity is asked
NROTATIONS_DP_BU = NROTATIONS_DP
NROTATIONS_QP_BU = NROTATIONS_QP
if(MAX_SPIN_EXTERNAL_PARTICLE.gt.3.AND.USERHEL.NE.-1) THEN
   if(.NOT.WARNED_LORENTZ_STAB_TEST_OFF) THEN
     WRITE(*,*) '##WARNING: Evaluation of a specific helicity was asked for this PS point, and there is a spin-2 (or higher) particle in the external states.'
	 WRITE(*,*) '##WARNING: As a result, MadLoop disabled the Lorentz rotation test for this phase-space point only.'
	 WRITE(*,*) '##WARNING: Further warning of that type suppressed.'
     WARNED_LORENTZ_STAB_TEST_OFF = .TRUE. 
   ENDIF
   NROTATIONS_QP=0
   NROTATIONS_DP=0
   CALL %(proc_prefix)sSET_N_EVALS(N_DP_EVAL,N_QP_EVAL)
ENDIF

IF(NTRY.EQ.0) THEN
  HELDOUBLECHECKED=(.NOT.DoubleCheckHelicityFilter).OR.(HelicityFilterLevel.eq.0)
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

IF (HelicityFilterLevel.eq.0) then
  FOUNDHELFILTER=.TRUE.
  DO J=1,NCOMB
	GOODHEL(J)=1
  ENDDO
  GOTO 122
ENDIF
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
IF (HelicityFilterLevel.eq.1) then
C We must make sure to remove the matching-helicity optimisation, as requested by the user.
  DO J=1,NCOMB
    IF ((GOODHEL(J).GT.1).OR.(GOODHEL(J).LT.-HELOFFSET)) THEN
	  GOODHEL(J)=1
	ENDIF
  ENDDO
ENDIF
122  CONTINUE
ENDIF

## if(LoopInduced){
C The born is of course 0 for loop-induced processes.
DO I=0,NSQUAREDSO
  ANS(0,I)=0.0d0
ENDDO
## }else{
C First compute the borns, it will store them in ANS(0,I)
C It is left untouched for the rest of MadLoop evaluation.
C Notice that the squared split order index I does NOT
C correspond to the same ordering of J for the loop ME 
C results stored in ANS(K,J), with K in [1-3].The ordering 
C of each can be obtained with ML5SOINDEX_FOR_SQUARED_ORDERS 
C and SQSOINDEX_FROM_ORDERS for the loop ME and born ME 
C respectively. For this to work, we assume that there is 
C always more squared split orders in the loop ME than in the
C born ME, which is practically always true. In any case, only
C the split_order summed value I=0 is used in ML5 code.
DO I=0,NSQSO_BORN
  BORNBUFF(I)=0.0d0
ENDDO
CALL %(proc_prefix)sSMATRIXHEL_SPLITORDERS(P_USER,USERHEL,BORNBUFF(0))
DO I=0,NSQSO_BORN
  ANS(0,I)=BORNBUFF(I)
ENDDO
## }

## if(LoopInduced){
C For loop-induced, the reference for comparison is set later from the total contribution of the previous PS point considered.
C But you can edit here the value to be used for the first PS points.    
IF (NPSPOINTS.EQ.0) THEN
  REF=1.0D-50
ELSE
  IF(NPSPOINTS.GE.MAXNREF_EVALS) THEN
    REF=Median(REF_EVALS,MAXNREF_EVALS)
  ELSE
    REF=Median(REF_EVALS,NPSPOINTS)
  ENDIF
ENDIF
## }else{
C We set here the reference to the born summed over all split orders
REF=0.0d0
DO I=1,NSQSO_BORN
REF=REF+ANS(0,I)
ENDDO
## }

MP_DONE=.FALSE.
MP_DONE_ONCE=.FALSE.
MP_PS_SET=.FALSE.
STAB_INDEX=0
DOING_QP_EVALS=.FALSE.
EVAL_DONE(1)=.TRUE.
DO I=2,MAXSTABILITYLENGTH
  EVAL_DONE(I)=.FALSE.
ENDDO

## if(LoopInduced) {
C For loop-induced processes, we should make sure not to use the first points
C to set the filters because it doesn't have a reasonable REF scale yet.
IF(.NOT.BYPASS_CHECK.AND.NPSPOINTS.GE.1) THEN
## } else {
IF(.NOT.BYPASS_CHECK) THEN
## }
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
IF ((HelicityFilterLevel.ne.0).AND.(.NOT. CHECKPHASE).AND.(.NOT.FOUNDHELFILTER)) THEN
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

C Make sure we start with empty caches
IF (AUTOMATIC_CACHE_CLEARING) THEN
  CALL %(proc_prefix)sCLEAR_CACHES()
ENDIF

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
## if(not AmplitudeReduction){
  DO I=1,NCTAMPS
## }else{
  DO I=1,NLOOPAMPS
## }
    AMPL(K,I)=(0.0d0,0.0d0)
  ENDDO
ENDDO

C Start by using the first available loop reduction library and qp library.
I_LIB=1
I_QP_LIB=1

goto 208
C MadLoop jumps to this label during stability checks when it recomputes a rotated PS point
200 CONTINUE
C For the computation of a rotated version of this PS point we must reset the all MadLoop cache since this changes the definition of the loop denominators.
C We don't check for AUTOMATIC_CACHE_CLEARING here because the Lorentz test should anyway be disabled if the flag is turned off.
CALL %(proc_prefix)sCLEAR_CACHES()
208 CONTINUE
## if(AmplitudeReduction) {
SKIP_LOOPNUM_COEFS_CONSTRUCTION=.FALSE.
GOTO 308
C MadLoop jumps to this label during stability checks when it recomputes the same PS point with a different CTMode
300 CONTINUE
C Of course the trick of reusing coefficients when reducing at the amplitude level only works when computing one helicity at a time
if (USERHEL.ne.-1) THEN
  SKIP_LOOPNUM_COEFS_CONSTRUCTION = .TRUE.
ENDIF
308 CONTINUE
## if(ComputeColorFlows) {
C We don't want to re-initialized the following quantities when checking the helicity filter. (which jumps to label 205 to probe each helicity).
C We however want to re-initialize them for each new computation part of the stability check (which jumps to label 200)
C This code is therefore placed before 205 and after 200.
CALL %(proc_prefix)sREINITIALIZE_CUMULATIVE_ARRAYS()
IF (ME_COMPUTATION_FROM_JAMP) THEN
C If both ME computational methods have been used, then the ME computation from color flows was stored in RES_FROM_JAMP and we must reset it here.
  DO I=0,NSQUAREDSO
    DO K=0,3
      RES_FROM_JAMP(K,I)=0.0d0
    ENDDO
  ENDDO
ENDIF
IF ((.NOT.DIRECT_ME_COMPUTATION).AND.ME_COMPUTATION_FROM_JAMP) THEN
C When computing the ME with color flows, the Born ME will be computed as well, so we reset here the result obtained from the smatrix call above.
DO I=0,NSQUAREDSO
  ANS(0,I)=0.0d0
ENDDO
ENDIF
## }

## if(iregi_available) {
C     Free cache when using IREGI
      IF(IREGIRECY.AND.MLReductionLib(I_LIB).EQ.3) THEN
        CALL IREGI_FREE_PS()
      ENDIF
## }

## if(TIRCaching){
C Even if the user did ask to turn off the automatic TIR cache clearing, we must do it now if the CTModeIndex rolls over the size of the TIR cache employed.
C Notice that we must do that only when processing a new CT mode as part of the stability test and not when computing a new helicity as part of the filtering process.
C This we check that we are not in the initialization phase.
C If we are not in CTModeRun=-1, then we never need to clear the cache since the TIR will always be used for a unique computation (not stab test).
C Also, it is clear that if we are running OPP when reaching this line, then we shouldn't clear the TIR cache as it might still be useful later.
C Finally, notice that the conditional statement below should never be true except you have TIR library supporting quadruple precision or when TIR_CACHE_SIZE<2.
IF((.NOT.CHECKPHASE.AND.(HELDOUBLECHECKED)).and.CTMODERUN.eq.-1.and.(MLReductionLib(I_LIB).ne.1.AND.MLReductionLib(I_LIB).ne.5).AND.(%(proc_prefix)sTIRCACHE_INDEX(CTMODE).eq.(TIR_CACHE_SIZE+1))) THEN
  CALL %(proc_prefix)sCLEAR_TIR_CACHE()
ENDIF
## }

## }

C MadLoop jumps to this label during initialization when it goes to the computation of the next helicity.
205 CONTINUE

IF (.NOT.MP_PS_SET.AND.(CTMODE.EQ.0.OR.CTMODE.GE.4)) THEN
  CALL %(proc_prefix)sSET_MP_PS(P_USER)
  MP_PS_SET = .TRUE.
ENDIF

LSCALE=DSQRT(ABS((P(0,1)+P(0,2))**2-(P(1,1)+P(1,2))**2-(P(2,1)+P(2,2))**2-(P(3,1)+P(3,2))**2))

CTCALL_REQ_SO_DONE=.FALSE.
FILTER_SO = (.NOT.CHECKPHASE).AND.HELDOUBLECHECKED.AND.(SQSO_TARGET.ne.-1)

## if(not AmplitudeReduction){
DO I=1,NLOOPGROUPS
  DO J=0,LOOPMAXCOEFS-1
    DO K=1,NSQUAREDSO
      LOOPCOEFS(J,K,I)=(0.0d0,0.0d0)
	ENDDO
  ENDDO
ENDDO
## }

DO I=1,NLOOPGROUPS
  DO J=1,3
    DO K=1,NSQUAREDSO  
      LOOPRES(J,K,I)=(0.0d0,0.0d0)
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
## if(not AmplitudeReduction){
  IF (.NOT.MP_DONE) THEN
    CALL %(proc_prefix)sMP_COMPUTE_LOOP_COEFS(MP_P,BUFFR_BIS)
C   It should be safe to directly set MP_DONE to true already here. But maybe I overlooked something.
    MP_DONE=.TRUE.
  ENDIF
C Even if MP_DONE is .TRUE. we should anyway skip the
C double precision evaluation as it as already been
c computed in quadruple precision.
  goto 300
## }else{
  CALL %(proc_prefix)sMP_COMPUTE_LOOP_COEFS(MP_P,BUFFR_BIS)
## if(ComputeColorFlows) {
    IF ((.NOT.DIRECT_ME_COMPUTATION).AND.ME_COMPUTATION_FROM_JAMP) THEN
C   If the ME's are computed from the color flows only, we must update the NLO part of ANS from BUFFR_BIS and the Born part of ANS using RES_FROM_JAMP(0,*)
      DO I=0,NSQUAREDSO
        ANS(0,I)=RES_FROM_JAMP(0,I)
        DO K=1,3
          ANS(K,I)=BUFFR_BIS(K,I)
        ENDDO
      ENDDO
    ENDIF
## }
C We must skip the double precision computation of both loop amplitudes and CT amplitudes because they will all be computed in MP_COMPUTE_LOOP_COEFS.
  goto 301
## }
ENDIF

DO H=1,NCOMB
  IF ((HELPICKED.EQ.H).OR.((HELPICKED.EQ.-1).AND.(CHECKPHASE.OR.(.NOT.HELDOUBLECHECKED).OR.(GOODHEL(H).GT.-HELOFFSET.AND.GOODHEL(H).NE.0)))) THEN
  DO I=1,NEXTERNAL
    NHEL(I)=HELC(I,H)
  ENDDO
  
  UVCT_REQ_SO_DONE=.FALSE.
  CT_REQ_SO_DONE=.FALSE.
  LOOP_REQ_SO_DONE=.FALSE.
 
  IF (.NOT.CHECKPHASE.AND.HELDOUBLECHECKED.AND.HELPICKED.EQ.-1) THEN
    HEL_MULT=GOODHEL(H)
  ELSE
    HEL_MULT=1
  ENDIF
 
## if(AmplitudeReduction){
  CTCALL_REQ_SO_DONE=.FALSE.
  
C The coefficient were already computed previously with another CTMode, so we can skip them
  IF (SKIP_LOOPNUM_COEFS_CONSTRUCTION) THEN
    GOTO 4000
  ENDIF
  
  DO I=1,NLOOPGROUPS
    DO J=0,LOOPMAXCOEFS-1
      LOOPCOEFS(J,I)=(0.0d0,0.0d0)
    ENDDO
  ENDDO
  
  DO K=1,3
    DO I=1,NLOOPAMPS
      AMPL(K,I)=(0.0d0,0.0d0)
    ENDDO
  ENDDO
## }

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

## if(not AmplitudeReduction) {
  DO J=1,NBORNAMPS
    CTEMP = 2.0d0*HEL_MULT*DCONJG(AMP(J))
    DO I=1,NCTAMPS	
	  CFTOT=DCMPLX(CF_N(I,J)/DBLE(ABS(CF_D(I,J))),0.0d0)
      IF(CF_D(I,J).LT.0) CFTOT=CFTOT*IMAG1
      ITEMP = %(proc_prefix)sML5SQSOINDEX(%(proc_prefix)sML5SOINDEX_FOR_LOOP_AMP(I),%(proc_prefix)sML5SOINDEX_FOR_BORN_AMP(J))
	  IF (.NOT.FILTER_SO.OR.SQSO_TARGET.EQ.ITEMP) THEN
        DO K=1,3
          TEMP2 = DBLE(CFTOT*AMPL(K,I)*CTEMP)
          ANS(K,ITEMP)=ANS(K,ITEMP)+TEMP2
          ANS(K,0)=ANS(K,0)+TEMP2
        ENDDO
      ENDIF
    ENDDO
  ENDDO
## }
  
  %(coef_construction)s
4000 CONTINUE
  LOOP_REQ_SO_DONE=.TRUE.

## if(AmplitudeReduction) {
  IF(SKIPLOOPEVAL) THEN
    GOTO 5000
  ENDIF
  DO I=1,NSQUAREDSO
    DO J=1,NLOOPGROUPS
      S(I,J)=.TRUE.
    ENDDO
  ENDDO
C We need the dummy argument I_SO for the squared order index to conform to the structure that the call to the LOOP* subroutine takes for processes with Born diagrams.
  I_SO=1
%(loop_CT_calls)s
5000 CONTINUE
  CTCALL_REQ_SO_DONE=.TRUE.

## if(ComputeColorFlows) {
  IF (DIRECT_ME_COMPUTATION) THEN
## }
  DO I=1,NLOOPAMPS
## if(not LoopInduced) {
    DO J=1,NBORNAMPS
## } else {
    DO J=1,NLOOPAMPS
## }
	  CFTOT=DCMPLX(CF_N(I,J)/DBLE(ABS(CF_D(I,J))),0.0d0)
      IF(CF_D(I,J).LT.0) CFTOT=CFTOT*IMAG1
## if(not LoopInduced) {
      ITEMP = %(proc_prefix)sML5SQSOINDEX(%(proc_prefix)sML5SOINDEX_FOR_LOOP_AMP(I),%(proc_prefix)sML5SOINDEX_FOR_BORN_AMP(J))
      DO K=1,3
        TEMP2(K) = 2.0d0*HEL_MULT*DBLE(CFTOT*AMPL(K,I)*DCONJG(AMP(J)))
      ENDDO
## } else {
      ITEMP = %(proc_prefix)sML5SQSOINDEX(%(proc_prefix)sML5SOINDEX_FOR_LOOP_AMP(I),%(proc_prefix)sML5SOINDEX_FOR_LOOP_AMP(J))
      TEMP2(1) = HEL_MULT*DBLE(CFTOT*(AMPL(1,I)*DCONJG(AMPL(1,J))))
C Computing the quantities below is not strictly necessary since the result should be finite
C It is however a good cross-check.
      TEMP2(2) = HEL_MULT*DBLE(CFTOT*(AMPL(2,I)*DCONJG(AMPL(1,J)) + AMPL(1,I)*DCONJG(AMPL(2,J))))
      TEMP2(3) = HEL_MULT*DBLE(CFTOT*(AMPL(3,I)*DCONJG(AMPL(1,J)) + AMPL(1,I)*DCONJG(AMPL(3,J))+AMPL(2,I)*DCONJG(AMPL(2,J))))
## }
C To mimick the structure of the squared amplitude reduction, we add here the squared counterterm contribution directly to the result ANS() and put the loop contributions in the LOOPRES array which will be added to ANS later
      IF (I.LE.NCTAMPS) THEN
        IF (.NOT.FILTER_SO.OR.SQSO_TARGET.EQ.ITEMP) THEN
          DO K=1,3
            ANS(K,ITEMP)=ANS(K,ITEMP)+TEMP2(K)
            ANS(K,0)=ANS(K,0)+TEMP2(K)
          ENDDO
        ENDIF
      ELSE
        DO K=1,3
          LOOPRES(K,ITEMP,I-NCTAMPS)=LOOPRES(K,ITEMP,I-NCTAMPS)+TEMP2(K)
C During the evaluation of the AMPL, we had stored the stability in S(1,*) so we now copy over this flag to the relevant contributing Squared orders.
          S(ITEMP,I-NCTAMPS)=S(1,I-NCTAMPS)
        ENDDO
      ENDIF
    ENDDO
  ENDDO
## }
## if(ComputeColorFlows) {
ENDIF
## }

## if(ComputeColorFlows){
C We should compute the color flow either if it contributes to the final result (i.e. not used just for the filtering), or if the computation is only done from the color flows
IF (((.NOT.DIRECT_ME_COMPUTATION).AND.ME_COMPUTATION_FROM_JAMP).OR.(H.EQ.USERHEL.OR.USERHEL.EQ.-1)) THEN
C The cumulative quantities must only be computed if that helicity contributes according to user request (second argument of the subroutine below).
  CALL %(proc_prefix)sCOMPUTE_COLOR_FLOWS(HEL_MULT)
  IF(ME_COMPUTATION_FROM_JAMP) THEN
    CALL %(proc_prefix)sCOMPUTE_RES_FROM_JAMP(BUFFRES,HEL_MULT)
    IF(((.NOT.DIRECT_ME_COMPUTATION).AND.ME_COMPUTATION_FROM_JAMP)) THEN
C     If the computation from the color flow is the only form of computation, we directly update the answer.
      DO K=0,3
        DO I=0,NSQUAREDSO
          ANS(K,I)=ANS(K,I)+BUFFRES(K,I)
        ENDDO
      ENDDO
C   When setting up the loop filter, it is important to set the quantitied LOOPRES.
C   Notice that you may have a more powerful filter with the direct computation mode because it can filter vanishing loop contributions for a particular squared split order only
C   The quantity LOOPRES defined below is not physical, but it's ok since it is only intended for the loop filtering.
     IF(.NOT.FOUNDLOOPFILTER.AND.USELOOPFILTER) THEN
        DO J=1,NLOOPGROUPS
          DO I=1,NSQUAREDSO
            DO K=1,3
              LOOPRES(K,I,J)=LOOPRES(K,I,J)+AMPL(K,NCTAMPS+J)
            ENDDO
          ENDDO
        ENDDO
      ENDIF
C   The if statement below is not strictly necessary but makes it clear when it is executed.
    ELSEIF(H.EQ.USERHEL.OR.USERHEL.EQ.-1) THEN
C     If both computational method is used, then we must just update RES_FROM_JAMP
      DO K=0,3
        DO I=0,NSQUAREDSO
          RES_FROM_JAMP(K,I)=RES_FROM_JAMP(K,I)+BUFFRES(K,I)
        ENDDO
      ENDDO
    ENDIF
	IF (H.EQ.USERHEL.OR.USERHEL.EQ.-1) THEN
	  CALL %(proc_prefix)sCOMPUTE_COLOR_FLOWS_DERIVED_QUANTITIES(HEL_MULT)
	ENDIF
  ENDIF
ENDIF
## }

  ENDIF
ENDDO

## if(not AmplitudeReduction){
%(coef_merging)s
## }

## if (ComputeColorFlows) {
IF(DIRECT_ME_COMPUTATION) THEN
C Lines below are not necessary when computing the ME from color flows
## }
DO I=0,NSQUAREDSO
  DO J=1,3
    BUFFR_BIS(J,I)=ANS(J,I)
  ENDDO
ENDDO
## if (ComputeColorFlows) {
ENDIF
## }


## if(not AmplitudeReduction){
C MadLoop jumps to this label during stability checks when it recomputes the same PS point with a different CTMode
300 CONTINUE

C Make sure that the loop calls are performed since this is new evaluation.
CTCALL_REQ_SO_DONE=.FALSE.

## if(iregi_available) {
C     Free cache when using IREGI
      IF(IREGIRECY.AND.MLReductionLib(I_LIB).EQ.3) THEN
        CALL IREGI_FREE_PS()
      ENDIF
## }

## if(TIRCaching){
C Even if the user did ask to turn off the automatic TIR cache clearing, we must do it now if the CTModeIndex rolls over the size of the TIR cache employed.
C Notice that we must do that only when processing a new CT mode as part of the stability test and not when computing a new helicity as part of the filtering process.
C This we check that we are not in the initialization phase.
C If we are not in CTModeRun=-1, then we never need to clear the cache since the TIR will always be used for a unique computation (not stab test).
C Also, it is clear that if we are running OPP when reaching this line, then we shouldn't clear the TIR cache as it might still be useful later.
C Finally, notice that the conditional statement below should never be true except you have TIR library supporting quadruple precision or when TIR_CACHE_SIZE<2.
IF((.NOT.CHECKPHASE.AND.(HELDOUBLECHECKED)).AND.CTMODERUN.eq.-1.and.(MLReductionLib(I_LIB).ne.1.AND.MLReductionLib(I_LIB).ne.5).AND.(%(proc_prefix)sTIRCACHE_INDEX(CTMODE).eq.(TIR_CACHE_SIZE+1))) THEN
  CALL %(proc_prefix)sCLEAR_TIR_CACHE()
ENDIF
## }

## }else{
C MadLoop jumps to this label just after having called the subroutine %(proc_prefix)sMP_COMPUTE_LOOP_COEFS to compute OpenLoop coefficients in quadruple precision (and not double precision as done above)
301 CONTINUE
## }

## if (ComputeColorFlows) {
IF(DIRECT_ME_COMPUTATION) THEN
C Lines below are not necessary when computing the ME from color flows
## }
DO I=0,NSQUAREDSO
  DO J=1,3
    ANS(J,I)=BUFFR_BIS(J,I)
  ENDDO
ENDDO
## if (ComputeColorFlows) {
ENDIF
## }

## if (ComputeColorFlows) {
IF ((.NOT.DIRECT_ME_COMPUTATION).AND.ME_COMPUTATION_FROM_JAMP) THEN
C We can skip the update of ANS if it was computed from color flows
  GOTO 1226
ENDIF
## }

IF(SKIPLOOPEVAL) THEN
  GOTO 1226
ENDIF

## if(not AmplitudeReduction){
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
## }

DO I=1,NLOOPGROUPS
  LTEMP=.TRUE.
  DO K=1,NSQUAREDSO
    IF (.NOT.FILTER_SO.OR.SQSO_TARGET.EQ.K) THEN
      IF (.NOT.S(K,I)) LTEMP=.FALSE.
      DO J=1,3
        ANS(J,K)=ANS(J,K)+LOOPRES(J,K,I)
        ANS(J,0)=ANS(J,0)+LOOPRES(J,K,I)
      ENDDO
    ENDIF
  ENDDO
  IF((CTMODERUN.NE.-1).AND..NOT.CHECKPHASE.AND.(.NOT.LTEMP)) THEN
    WRITE(*,*) '##W03 WARNING Contribution ',I,' is unstable.'           
  ENDIF
ENDDO

C Make sure that no NaN is present in the result
DO K=1,NSQUAREDSO
  DO J=1,3
    IF (.NOT.(ANS(J,K).eq.ANS(J,K))) THEN
      IF (DOING_QP_EVALS) THEN
        FPE_IN_QP_REDUCTION = .TRUE.
      ELSE
        FPE_IN_DP_REDUCTION = .TRUE.
      ENDIF
    ENDIF
  ENDDO
ENDDO

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
  HELSAVED(1,HELPICKED)=0.0d0
  HELSAVED(2,HELPICKED)=0.0d0
  HELSAVED(3,HELPICKED)=0.0d0
  DO I=1,NSQUAREDSO
    IF (CHOSEN_SO_CONFIGS(I)) THEN
      HELSAVED(1,HELPICKED)=HELSAVED(1,HELPICKED)+ANS(1,I)
      HELSAVED(2,HELPICKED)=HELSAVED(2,HELPICKED)+ANS(2,I)
      HELSAVED(3,HELPICKED)=HELSAVED(3,HELPICKED)+ANS(3,I)
	ENDIF
  ENDDO

## if(LoopInduced){
C We make sure not to perform any check when NTRY is 0 because this means that
C the REF scale has not been set to the result of the evaluation for the first PS point.
## }
  IF (CHECKPHASE.AND.NTRY.NE.0) THEN
C   SET THE HELICITY FILTER
    IF(.NOT.FOUNDHELFILTER) THEN
	  HEL_INCONSISTENT=.FALSE.
      IF(%(proc_prefix)sIsZero(DABS(HELSAVED(1,HELPICKED))+DABS(HELSAVED(2,HELPICKED))+DABS(HELSAVED(3,HELPICKED)),REF/DBLE(NCOMB),-1,-1)) THEN
        IF(NTRY.EQ.1) THEN
	      GOODHEL(HELPICKED)=-HELOFFSET
	    ELSEIF(GOODHEL(HELPICKED).NE.-HELOFFSET) THEN
		  WRITE(*,*) '##W02A WARNING Inconsistent zero helicity ',HELPICKED
		  IF(HELINITSTARTOVER) THEN
	        WRITE(*,*) '##I01 INFO Initialization starting over because of inconsistency in the helicity filter setup.'
	        NTRY=0
		  ELSE
		    HEL_INCONSISTENT=.TRUE.
		  ENDIF
		ENDIF
      ELSEIF(HelicityFilterLevel.gt.1) THEN
	    DO H=1,HELPICKED-1
		  IF(GOODHEL(H).GT.-HELOFFSET) THEN
C           Be looser for helicity check, bring a factor 100
		    DUMMY=%(proc_prefix)sISSAME(HELSAVED(1,HELPICKED),HELSAVED(1,H),REF,.FALSE.)
		    IF(DUMMY.NE.0) THEN
              IF(NTRY.EQ.1) THEN
C               Set the matching helicity to be contributing once more
		        GOODHEL(H)=GOODHEL(H)+DUMMY
C               Use an offset to clearly show it is linked to an other one and to avoid overlap
			    GOODHEL(HELPICKED)=-H-HELOFFSET
C             Make sure we have paired this hel config to the same one last PS point
			  ELSEIF(GOODHEL(HELPICKED).NE.(-H-HELOFFSET)) THEN
			    WRITE(*,*) '##W02B WARNING Inconsistent matching helicity ',HELPICKED
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
	  IF (.NOT.%(proc_prefix)sIsZero(DABS(HELSAVED(1,HELPICKED))+DABS(HELSAVED(2,HELPICKED))+DABS(HELSAVED(2,HELPICKED)),REF/DBLE(NCOMB),-1,-1)) THEN
	    write(*,*) '##W15 Helicity filter could not be successfully double checked.'
	    write(*,*) '##One reason for this is that you might have changed sensible parameters which affected what are the zero helicity configurations.'
	    write(*,*) '##MadLoop will try to reset the Helicity filter with the next PS points it receives.'
	    NTRY=0
	    OPEN(29,FILE=HelFilterFN,err=348)
348     CONTINUE
        CLOSE(29,STATUS='delete')
	  ENDIF
	ENDIF
	IF (GOODHEL(HELPICKED).LT.-HELOFFSET.AND.NTRY.NE.0) THEN
	  IF(%(proc_prefix)sISSAME(HELSAVED(1,HELPICKED),HELSAVED(1,ABS(GOODHEL(HELPICKED)+HELOFFSET)),REF,.TRUE.).EQ.0) THEN
	    write(*,*) '##W15 Helicity filter could not be successfully double checked.'
	    write(*,*) '##One reason for this is that you might have changed sensible parameters which affected the helicity dependance relations.'
	    write(*,*) '##MadLoop will try to reset the Helicity filter with the next PS points it receives.'
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
    goto 205
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
## if(LoopInduced){
C Update of REF_EVALS (only for loop-induced processes).
    TMPR = ABS(ANS(1,0)) + ABS(ANS(2,0)) + ABS(ANS(3,0))
C We add one here to the number of PS points used for building the reference scale for comparisons.
C It might be that when asking for specific helicities, the user started with a vanishing helicity
C not filtered yet. In this case, the new ref would remain zero. So we want to check for this
C and wait for a point for which the evaluation isn't be zero.
    IF(TMPR.ne.0.0d0) THEN
	  REF_EVALS(MOD(NPSPOINTS,MAXNREF_EVALS)+1) = TMPR
      NPSPOINTS = NPSPOINTS+1
	ENDIF
## }
	IF(NTRY.EQ.0) THEN
	  NATTEMPTS=NATTEMPTS+1
	  IF(NATTEMPTS.EQ.MAXATTEMPTS) THEN
	    WRITE(*,*) '##E01 ERROR Could not initialize the filters in ',MAXATTEMPTS,' trials'
		STOP 1
	  ENDIF
	ENDIF
  ENDIF
## if(LoopInduced){
ELSE
C When not in checking mode, update the ref for the first MAXNREF_EVALS points (The ref. scale could still be used after this stage if BYPASS_CHECK was set to true at some point using SLOOPMATRIX_THRES).
C Is is possible to simply remove the if statement below to have a running reference scale which always depends on the MAXNREF_EVALS *last* evaluations. 
    if (NPSPOINTS.LE.MAXNREF_EVALS) THEN
	  TMPR = ABS(ANS(1,0)) + ABS(ANS(2,0)) + ABS(ANS(3,0))
	  IF (TMPR.ne.0.0d0) THEN
        REF_EVALS(MOD(NPSPOINTS,MAXNREF_EVALS)+1) = TMPR
        NPSPOINTS = NPSPOINTS+1
      ENDIF
    ENDIF
## }
ENDIF

## if (ComputeColorFlows) {
C When computing the ME from the color flows, we also compute the born ME from them, so we must apply the normalization factors to the born ME as well.
IF(((.NOT.DIRECT_ME_COMPUTATION).AND.ME_COMPUTATION_FROM_JAMP)) THEN
  ITEMP=0
ELSE
  ITEMP=1
ENDIF
DO K=ITEMP,3
## } else {
DO K=1,3
## }
  DO I=0,NSQUAREDSO
    ANS(K,I)=ANS(K,I)/DBLE(IDEN)
    IF (USERHEL.NE.-1) THEN
      ANS(K,I)=ANS(K,I)*HELAVGFACTOR
    ENDIF
  ENDDO
ENDDO

## if (ComputeColorFlows) {
IF (DIRECT_ME_COMPUTATION.AND.ME_COMPUTATION_FROM_JAMP) THEN
  WRITE(*,*) ' ================================= '
  WRITE(*,*) ' === JAMP double-checking test === '
  WRITE(*,*) ' ================================= '
  CALL %(proc_prefix)sWRITE_MOM(P)
  DO J=1,NSQUAREDSO+1
C We should finish by the summed orders
    I = MOD(J,NSQUAREDSO+1)
    IF (I.EQ.0) THEN
      WRITE(*,*) ' > Checking the sum of all chosen squared split orders'    
    ELSE
      WRITE(*,*) ' > Checking squared split order #',I
    ENDIF
## if (LoopInduced) {
    DO K=1,1
## } else {
    DO K=0,3
## }
      RES_FROM_JAMP(K,I)=RES_FROM_JAMP(K,I)/DBLE(IDEN)
      IF (USERHEL.NE.-1) THEN
        RES_FROM_JAMP(K,I)=RES_FROM_JAMP(K,I)*HELAVGFACTOR
      ENDIF
      IF (K.eq.0) WRITE(*,*) ' || Born :' 
      IF (K.eq.1) WRITE(*,*) ' || Finite part :' 
      IF (K.eq.2) WRITE(*,*) ' || Single pole residue :' 
      IF (K.eq.3) WRITE(*,*) ' || Double pole residue :' 
      WRITE(*,*) ' --> Direct result        =',ANS(K,I)
      WRITE(*,*) ' --> Computed from JAMPS  =',RES_FROM_JAMP(K,I)
      IF((RES_FROM_JAMP(K,I)+ANS(K,I)).EQ.0.0d0) THEN
        TMPR = ABS(RES_FROM_JAMP(K,I)-ANS(K,I))      
      ELSE
        TMPR = ABS((ANS(K,I)-RES_FROM_JAMP(K,I))/((ANS(K,I)+RES_FROM_JAMP(K,I))/2.0d0))
      ENDIF
      WRITE(*,*) ' --> Relative diff.       =',TMPR
      IF(TMPR.GT.JAMP_DOUBLECHECK_THRES) THEN
         STOP 'Consistency cross-check of JAMPS failed.'
      ENDIF
    ENDDO
  ENDDO
  WRITE(*,*) ' ================================= '
ENDIF
## }

IF(.NOT.CHECKPHASE.AND.HELDOUBLECHECKED.AND.(CTMODERUN.EQ.-1)) THEN
  STAB_INDEX=STAB_INDEX+1  
  IF(DOING_QP_EVALS.AND.LOOPLIBS_QPAVAILABLE(MLReductionLib(I_LIB))) THEN
C   Only run over the reduction algorithms which support quadruple precision
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

  IF(DOING_QP_EVALS.AND.LOOPLIBS_QPAVAILABLE(MLReductionLib(I_LIB))) THEN	
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
C     If some TIR library would not support the loop direction test (they all do for now), then we would just copy the answer from mode 1 and carry on.
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

  IF(DOING_QP_EVALS.AND.LOOPLIBS_QPAVAILABLE(MLReductionLib(I_LIB))) THEN
    CALL %(proc_prefix)sCOMPUTE_ACCURACY(QP_RES,N_QP_EVAL,ACC,ANS)
C   If a floating point exception was encountered during the reduction,
C   the result cannot be trusted at all and we hardset all accuracies to 1.0
    IF(FPE_IN_QP_REDUCTION) THEN
      DO I=0,NSQUAREDSO
        ACC(I)=1.0d0
      ENDDO
    ENDIF
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
      CALL %(proc_prefix)sCOMPUTE_ACCURACY(QP_RES,N_QP_EVAL,ACC,ANS)
	  IF(NEPS.LE.10) THEN	  
        WRITE(*,*) '##W03 WARNING An unstable PS point was',       ' detected.'
        IF(FPE_IN_QP_REDUCTION) THEN
          WRITE(*,*) '## The last QP reduction was deemed unstable because a floating point exception was encountered.'        
        ENDIF
		IF (NSQUAREDSO.NE.1) THEN
		  WRITE(*,*) '##Accuracies for each split order, starting with the summed case'
	      WRITE(*,*) '##DP accuracies (for each split order): ',(TEMP1(I),I=0,NSQUAREDSO)
	      WRITE(*,*) '##QP accuracies (for each split order): ',(ACC(I),I=0,NSQUAREDSO)
		ELSE
	      WRITE(*,*) '##DP accuracy: ',TEMP1(1)
	      WRITE(*,*) '##QP accuracy: ',ACC(1)
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
C    A new reduction tool will be used. Reinitialize the FPE flags.
	 FPE_IN_DP_REDUCTION=.False.
	 FPE_IN_QP_REDUCTION=.False.
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
C   If a floating point exception was encountered during the reduction,
C   the result cannot be trusted at all and we hardset all accuracies to 1.0
    IF(FPE_IN_DP_REDUCTION) THEN
      DO I=0,NSQUAREDSO
        ACC(I)=1.0d0
      ENDDO
    ENDIF
	IF(MAXVAL(ACC).GE.MLSTABTHRES) THEN
	  I_LIB=I_LIB+1
	  IF((I_LIB.GT.NLOOPLIB.OR.MLReductionLib(I_LIB).EQ.0).AND.QP_TOOLS_AVAILABLE)THEN
	        I_LIB=INDEX_QP_TOOLS(1)
C       A new reduction tool will be used. Reinitialize the FPE flags.
	    FPE_IN_DP_REDUCTION=.False.
	    FPE_IN_QP_REDUCTION=.False.	    
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
C       A new reduction tool will be used. Reinitialize the FPE flags.
	    FPE_IN_DP_REDUCTION=.False.
	    FPE_IN_QP_REDUCTION=.False.
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
               IF(FPE_IN_DP_REDUCTION) THEN
                 WRITE(*,*) '## The last DP reduction was deemed unstable because a floating point exception was encountered.'        
               ENDIF
               CALL %(proc_prefix)sCOMPUTE_ACCURACY(DP_RES,N_DP_EVAL,ACC,ANS)
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

C Reinitialize the Lorentz test if it had been disabled because spin-2 particles are in the external states.
NROTATIONS_DP = NROTATIONS_DP_BU
NROTATIONS_QP = NROTATIONS_QP_BU

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

C Make sure that we finish by emptying caches
IF (AUTOMATIC_CACHE_CLEARING) THEN
  CALL %(proc_prefix)sCLEAR_CACHES()
ENDIF
      END

      SUBROUTINE %(proc_prefix)sCLEAR_CACHES()
C Clears all the caches used at some point in MadLoop
        CALL %(proc_prefix)sCLEAR_TIR_CACHE()
## if(ninja_available){
        call ninja_clear_integral_cache()
## }
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
C         Be looser for helicity comparison, so bring a factor 100
		  ELSEIF(.NOT.%(proc_prefix)sIsZero(ABS((resA(J)/resB(J))-DBLE(WGT_TO_TRY(I))),1.0d0,-1,-1)) THEN
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
	  COMMON/%(proc_prefix)sML5SQPLITORDERS/SQPLITORDERS
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
	  COMMON/%(proc_prefix)sML5AMPSPLITORDERS/AMPSPLITORDERS
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

C This is the inverse subroutine of ML5SOINDEX_FOR_SQUARED_ORDERS. Not directly useful, but provided nonetheless.
      SUBROUTINE %(proc_prefix)sML5GET_SQUARED_ORDERS_FOR_SOINDEX(SOINDEX,ORDERS)
C
C This functions returns the orders identified by the squared split order index in argument. Order values correspond to following list of couplings (and in this order):
C %(split_order_str_list)s
C
C CONSTANTS
C
      INTEGER    NSO, NSQSO
	  PARAMETER (NSO=%(nSO)d, NSQSO=%(nSquaredSO)d)
C
C ARGUMENTS
C
      INTEGER SOINDEX, ORDERS(NSO)
C
C LOCAL VARIABLES
C
      INTEGER I
	  INTEGER SQPLITORDERS(NSQSO,NSO)
	  COMMON/%(proc_prefix)sML5SQPLITORDERS/SQPLITORDERS	  
C
C BEGIN CODE
C
      IF (SOINDEX.gt.0.and.SOINDEX.le.NSQSO) THEN
        DO I=1,NSO
          ORDERS(I) =  SQPLITORDERS(SOINDEX,I)
        ENDDO
        RETURN
      ENDIF

	  WRITE(*,*) 'ERROR:: Stopping function %(proc_prefix)sML5GET_SQUARED_ORDERS_FOR_SOINDEX'
	  WRITE(*,*) 'Could not find squared orders index ',SOINDEX
	  STOP

      END SUBROUTINE

C This is the inverse subroutine of getting amplitude SO orders. Not directly useful, but provided nonetheless.
      SUBROUTINE %(proc_prefix)sML5GET_ORDERS_FOR_AMPSOINDEX(SOINDEX,ORDERS)
C
C This functions returns the orders identified by the split order index in argument. Order values correspond to following list of couplings (and in this order):
C %(split_order_str_list)s
C
C CONSTANTS
C
      INTEGER    NSO, NAMPSO
	  PARAMETER (NSO=%(nSO)d, NAMPSO=%(nAmpSO)d)
C
C ARGUMENTS
C
      INTEGER SOINDEX, ORDERS(NSO)
C
C LOCAL VARIABLES
C
      INTEGER I
      INTEGER AMPSPLITORDERS(NAMPSO,NSO)
	  COMMON/%(proc_prefix)sML5AMPSPLITORDERS/AMPSPLITORDERS
C
C BEGIN CODE
C
      IF (SOINDEX.gt.0.and.SOINDEX.le.NAMPSO) THEN
        DO I=1,NSO
          ORDERS(I) =  AMPSPLITORDERS(SOINDEX,I)
        ENDDO
        RETURN
      ENDIF

	  WRITE(*,*) 'ERROR:: Stopping function %(proc_prefix)sML5GET_ORDERS_FOR_AMPSOINDEX'
	  WRITE(*,*) 'Could not find amplitude split orders index ',SOINDEX
	  STOP

      END SUBROUTINE


C This function is not directly useful, but included for completeness
      INTEGER FUNCTION %(proc_prefix)sML5SOINDEX_FOR_AMPORDERS(ORDERS)
C
C This functions returns the integer index identifying the amplitude split orders passed in argument which correspond to the values of the following list of couplings (and in this order):
C %(split_order_str_list)s
C
C CONSTANTS
C
      INTEGER    NSO, NAMPSO
	  PARAMETER (NSO=%(nSO)d, NAMPSO=%(nAmpSO)d)
C
C ARGUMENTS
C
      INTEGER ORDERS(NSO)
C
C LOCAL VARIABLES
C
      INTEGER I,J
      INTEGER AMPSPLITORDERS(NAMPSO,NSO)
	  COMMON/%(proc_prefix)sML5AMPSPLITORDERS/AMPSPLITORDERS
C
C BEGIN CODE
C
      DO I=1,NAMPSO
	    DO J=1,NSO
		  IF (ORDERS(J).NE.AMPSPLITORDERS(I,J)) GOTO 1009
		ENDDO
		%(proc_prefix)sML5SOINDEX_FOR_AMPORDERS = I
		RETURN
1009    CONTINUE
	  ENDDO

	  WRITE(*,*) 'ERROR:: Stopping function %(proc_prefix)sML5SOINDEX_FOR_AMPORDERS'
	  WRITE(*,*) 'Could not find squared orders ',(ORDERS(I),I=1,NSO)
	  STOP

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

	  END SUBROUTINE

	  SUBROUTINE %(proc_prefix)sSET_AUTOMATIC_CACHE_CLEARING(ONOFF)
C
C This function can be called by the MadLoop user so as to manually chose when
C to reset the TIR cache.
C
	  	IMPLICIT NONE

	    include 'MadLoopParams.inc'

		LOGICAL ONOFF

      	LOGICAL AUTOMATIC_CACHE_CLEARING
      	DATA AUTOMATIC_CACHE_CLEARING/.TRUE./
      	COMMON/%(proc_prefix)sRUNTIME_OPTIONS/AUTOMATIC_CACHE_CLEARING
      	
	    INTEGER N_DP_EVAL, N_QP_EVAL
	    COMMON/%(proc_prefix)sN_EVALS/N_DP_EVAL,N_QP_EVAL

## if(not TIRCaching){
       WRITE(*,*) 'Warning: No TIR caching implemented. Call to SET_AUTOMATIC_CACHE_CLEARING did nothing.'
## } else {

		AUTOMATIC_CACHE_CLEARING = ONOFF
		
		IF (NRotations_DP.ne.0.or.NRotations_QP.ne.0) THEN
		  WRITE(*,*) 'Warning: One cannot remove the TIR cache automatic clearing while at the same time keeping Lorentz rotations for stability tests.'
		  WRITE(*,*) 'MadLoop will therefore automatically set NRotations_DP and NRotations_QP to 0.'
		  NRotations_DP = 0
		  NRotations_QP = 0
  		  CALL %(proc_prefix)sSET_N_EVALS(N_DP_EVAL,N_QP_EVAL)
		ENDIF
## }
	  END SUBROUTINE
	  
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
C     U == 5
C         Stable with Samurai
C     U == 6
C         Stable with Ninja in double precision
C     U == 8
C         Stable with Ninja in quadruple precision
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
	  
C The subroutine below perform clean-up duties for MadLoop like de-allocating
c arrays
	  SUBROUTINE %(proc_prefix)sEXIT_MADLOOP()
## if(ComputeColorFlows) {
	    CALL %(proc_prefix)sDEALLOCATE_COLOR_FLOWS()
## }
        CONTINUE
	  END
