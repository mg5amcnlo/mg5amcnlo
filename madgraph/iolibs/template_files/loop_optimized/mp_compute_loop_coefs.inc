      SUBROUTINE %(proc_prefix)sMP_COMPUTE_LOOP_COEFS(PS,ANSDP)
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
C CONSTANTS
C
      CHARACTER*64 paramFileName
      PARAMETER ( paramFileName='MadLoopParams.dat')
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
	  %(real_mp_format)s    ZERO
      PARAMETER (ZERO=0E0_16)
      %(complex_mp_format)s IMAG1
      PARAMETER (IMAG1=(0e0_16,1e0_16))
      %(complex_mp_format)s DP_IMAG1
      PARAMETER (DP_IMAG1=(0d0,1d0))
C     These are constants related to the split orders
      INTEGER    NSO, NSQUAREDSO, NAMPSO
	  PARAMETER (NSO=%(nSO)d, NSQUAREDSO=%(nSquaredSO)d, NAMPSO=%(nAmpSO)d)

##if(ComputeColorFlows) {
C The variables below are just used in the context of a JAMP consistency check turned off by default.
      LOGICAL DIRECT_ME_COMPUTATION, ME_COMPUTATION_FROM_JAMP
      %(real_dp_format)s RES_FROM_JAMP(0:3,0:NSQUAREDSO)
	  common/%(proc_prefix)sDOUBLECHECK_JAMP/RES_FROM_JAMP,DIRECT_ME_COMPUTATION,ME_COMPUTATION_FROM_JAMP
## }
	  
C  
C ARGUMENTS 
C  
      %(real_mp_format)s PS(0:3,NEXTERNAL)
      %(real_dp_format)s ANSDP(3,0:NSQUAREDSO)
C  
C LOCAL VARIABLES 
C  
      LOGICAL DPW_COPIED
	  LOGICAL COMPUTE_INTEGRAND_IN_QP
      INTEGER I,J,K,H,HEL_MULT,ITEMP
## if(not AmplitudeReduction){
	  %(real_mp_format)s TEMP2
	  %(real_dp_format)s DP_TEMP2	  
## }else{
	  %(real_mp_format)s TEMP2(3)
	  %(real_dp_format)s DP_TEMP2(3)	  
## }
      %(complex_mp_format)s CTEMP
      %(complex_dp_format)s DP_CTEMP

      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
      %(real_mp_format)s MP_P(0:3,NEXTERNAL)
	  %(real_dp_format)s P(0:3,NEXTERNAL)

	  DATA IC/NEXTERNAL*1/
	  %(real_mp_format)s ANS(3,0:NSQUAREDSO)
## if(ComputeColorFlows) {
	  %(real_dp_format)s BUFFRES(0:3,0:NSQUAREDSO)
## }
	  %(complex_mp_format)s COEFS(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)
      %(complex_mp_format)s CFTOT
      %(complex_dp_format)s DP_CFTOT	  
C
C FUNCTIONS
C
      INTEGER %(proc_prefix)sML5SOINDEX_FOR_BORN_AMP
	  INTEGER %(proc_prefix)sML5SOINDEX_FOR_LOOP_AMP
	  INTEGER %(proc_prefix)sML5SQSOINDEX
C  
C GLOBAL VARIABLES
C  
	  include 'mp_coupl_same_name.inc'

	  include 'MadLoopParams.inc'

	  LOGICAL CHECKPHASE, HELDOUBLECHECKED
      common/%(proc_prefix)sINIT/CHECKPHASE, HELDOUBLECHECKED

	  INTEGER HELOFFSET
	  INTEGER GOODHEL(NCOMB)
	  LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
	  common/%(proc_prefix)sFilters/GOODAMP,GOODHEL,HELOFFSET

	  INTEGER HELPICKED
	  common/%(proc_prefix)sHELCHOICE/HELPICKED

	  INTEGER USERHEL
	  common/%(proc_prefix)sUSERCHOICE/USERHEL

	  INTEGER SQSO_TARGET
	  common/%(proc_prefix)sSOCHOICE/SQSO_TARGET
      
	  LOGICAL UVCT_REQ_SO_DONE,MP_UVCT_REQ_SO_DONE,CT_REQ_SO_DONE,MP_CT_REQ_SO_DONE,LOOP_REQ_SO_DONE,MP_LOOP_REQ_SO_DONE,CTCALL_REQ_SO_DONE,FILTER_SO
	  common/%(proc_prefix)sSO_REQS/UVCT_REQ_SO_DONE,MP_UVCT_REQ_SO_DONE,CT_REQ_SO_DONE,MP_CT_REQ_SO_DONE,LOOP_REQ_SO_DONE,MP_LOOP_REQ_SO_DONE,CTCALL_REQ_SO_DONE,FILTER_SO

## if(not LoopInduced) {
	  %(complex_mp_format)s AMP(NBORNAMPS)
	  common/%(proc_prefix)sMP_AMPS/AMP
	  %(complex_dp_format)s DP_AMP(NBORNAMPS)
	  common/%(proc_prefix)sAMPS/DP_AMP
## }	  
	  %(complex_mp_format)s W(20,NWAVEFUNCS)
	  common/%(proc_prefix)sMP_W/W  

	  %(complex_dp_format)s DPW(20,NWAVEFUNCS)
	  common/%(proc_prefix)sW/DPW

	  %(complex_mp_format)s WL(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE,0:NLOOPWAVEFUNCS)
	  %(complex_mp_format)s PL(0:3,0:NLOOPWAVEFUNCS)
	  common/%(proc_prefix)sMP_WL/WL,PL

      %(complex_dp_format)s DP_WL(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE,0:NLOOPWAVEFUNCS)
	  %(complex_dp_format)s DP_PL(0:3,0:NLOOPWAVEFUNCS)
	  common/%(proc_prefix)sWL/DP_WL,DP_PL

## if(not AmplitudeReduction){
	  %(complex_mp_format)s LOOPCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)
## } else {
	  %(complex_mp_format)s LOOPCOEFS(0:LOOPMAXCOEFS-1,NLOOPGROUPS)
## }
	  common/%(proc_prefix)sMP_LCOEFS/LOOPCOEFS

## if(not AmplitudeReduction){
	  %(complex_dp_format)s DP_LOOPCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)
## } else {
	  %(complex_dp_format)s DP_LOOPCOEFS(0:LOOPMAXCOEFS-1,NLOOPGROUPS)
## }
	  common/%(proc_prefix)sLCOEFS/DP_LOOPCOEFS

## if(AmplitudeReduction) {
C This flag is used to prevent the re-computation of the OpenLoop coefficients when changing the CTMode for the stability test.
	  LOGICAL SKIP_LOOPNUM_COEFS_CONSTRUCTION
	  COMMON/%(proc_prefix)sSKIP_COEFS/SKIP_LOOPNUM_COEFS_CONSTRUCTION
## }

## if(not AmplitudeReduction){
      %(complex_mp_format)s AMPL(3,NCTAMPS)
## } else {
      %(complex_mp_format)s AMPL(3,NLOOPAMPS)
## } 
	  common/%(proc_prefix)sMP_AMPL/AMPL

## if(not AmplitudeReduction){	  
      %(complex_dp_format)s DP_AMPL(3,NCTAMPS)
## } else {
      %(complex_dp_format)s DP_AMPL(3,NLOOPAMPS)
## } 
	  common/%(proc_prefix)sAMPL/DP_AMPL

## if(AmplitudeReduction){
      %(complex_dp_format)s LOOPRES(3,NSQUAREDSO,NLOOPGROUPS)
	  LOGICAL S(NSQUAREDSO,NLOOPGROUPS)
	  common/%(proc_prefix)sLOOPRES/LOOPRES,S

	  integer I_SO
	  common/%(proc_prefix)sI_SO/I_SO
## }

	  INTEGER CF_D(NCOLORROWS,%(color_matrix_size)s)
	  INTEGER CF_N(NCOLORROWS,%(color_matrix_size)s)
	  common/%(proc_prefix)sCF/CF_D,CF_N

	  INTEGER HELC(NEXTERNAL,NCOMB)
	  common/%(proc_prefix)sHELCONFIGS/HELC

	  LOGICAL MP_DONE_ONCE
	  common/%(proc_prefix)sMP_DONE_ONCE/MP_DONE_ONCE

      INTEGER LIBINDEX
      COMMON/%(proc_prefix)sI_LIB/LIBINDEX

C ----------
C BEGIN CODE
C ----------

C Decide whether to really compute the integrand in quadruple precision or to fake it and copy the double precision computation in the quadruple precision variables.
COMPUTE_INTEGRAND_IN_QP = ((MLREDUCTIONLIB(LIBINDEX).eq.6.AND.UseQPIntegrandForNinja) .OR. (MLREDUCTIONLIB(LIBINDEX).eq.1.AND.UseQPIntegrandForCutTools))

C To be on the safe side, we always update the MP params here.
C It can be redundant as this routine can be called a couple of times for the same PS point during the stability checks.
C But it is really not time consuming and I would rather be safe.
CALL MP_UPDATE_AS_PARAM()

MP_DONE_ONCE = .TRUE.

C AS A SAFETY MEASURE WE FIRST COPY HERE THE PS POINT
DO I=1,NEXTERNAL
  DO J=0,3
    MP_P(J,I)=PS(J,I)
	P(J,I) = REAL(PS(J,I),KIND=8)
  ENDDO
ENDDO

DO I=0,3
  PL(I,0)=CMPLX(ZERO,ZERO,KIND=16)
  IF (.NOT.COMPUTE_INTEGRAND_IN_QP) THEN
    DP_PL(I,0)=DCMPLX(0.0d0,0.0d0)
  ENDIF
ENDDO

## if(AmplitudeReduction){
IF (.NOT.SKIP_LOOPNUM_COEFS_CONSTRUCTION) THEN
## }
DO I=1,MAXLWFSIZE
  DO J=0,LOOPMAXCOEFS-1
    DO K=1,MAXLWFSIZE
      IF (I.EQ.K.AND.J.EQ.0) THEN
        WL(I,J,K,0)=(1.0e0_16,ZERO)
      ELSE
        WL(I,J,K,0)=(ZERO,ZERO)
      ENDIF
	  IF (.NOT.COMPUTE_INTEGRAND_IN_QP) THEN
        IF (I.EQ.K.AND.J.EQ.0) THEN
          DP_WL(I,J,K,0)=(1.0d0,0.0d0)
        ELSE
          DP_WL(I,J,K,0)=(0.0d0,0.0d0)
        ENDIF
	  ENDIF
    ENDDO
  ENDDO
ENDDO

DO K=1, 3
## if(not AmplitudeReduction){
  DO I=1,NCTAMPS
## }else{
  DO I=1,NLOOPAMPS
## }
    AMPL(K,I)=(ZERO,ZERO)
	IF (.NOT.COMPUTE_INTEGRAND_IN_QP) THEN
      DP_AMPL(K,I)=(0.0d0,0.0d0)
	ENDIF
  ENDDO
ENDDO

## if(AmplitudeReduction){
ENDIF
## }

## if(not LoopInduced) {
DO I=1, NBORNAMPS
  DP_AMP(I) = (0.0d0,0.0d0)
  AMP(I) = (ZERO, ZERO)
ENDDO
## }

## if(not AmplitudeReduction){
DO I=1,NLOOPGROUPS
  DO J=0,LOOPMAXCOEFS-1
    DO K=1,NSQUAREDSO
      LOOPCOEFS(J,K,I)=(ZERO,ZERO)
      IF (.NOT.COMPUTE_INTEGRAND_IN_QP) THEN
        DP_LOOPCOEFS(J,K,I)=(0.0d0,0.0d0)	  
	  ENDIF
	ENDDO
  ENDDO
ENDDO
## }

DO K=1,3
  DO J=0,NSQUAREDSO
    ANSDP(K,J)=0.0d0
    ANS(K,J)=ZERO
  ENDDO
ENDDO

DPW_COPIED = .False.
DO H=1,NCOMB
  IF ((HELPICKED.EQ.H).OR.((HELPICKED.EQ.-1).AND.(CHECKPHASE.OR.(.NOT.HELDOUBLECHECKED).OR.(GOODHEL(H).GT.-HELOFFSET.AND.GOODHEL(H).NE.0)))) THEN
  DO I=1,NEXTERNAL
    NHEL(I)=HELC(I,H)
  ENDDO

  IF (COMPUTE_INTEGRAND_IN_QP) THEN  
    MP_UVCT_REQ_SO_DONE=.FALSE.
    MP_CT_REQ_SO_DONE=.FALSE.
    MP_LOOP_REQ_SO_DONE=.FALSE.  
  ELSE
    UVCT_REQ_SO_DONE=.FALSE.
    CT_REQ_SO_DONE=.FALSE.
    LOOP_REQ_SO_DONE=.FALSE.
  ENDIF

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
      LOOPCOEFS(J,I)=(ZERO,ZERO)
      IF (.NOT.COMPUTE_INTEGRAND_IN_QP) THEN
        DP_LOOPCOEFS(J,I)=(0.0d0,0.0d0)
	  ENDIF
    ENDDO
  ENDDO
  
 DO K=1, 3
  DO I=1,NLOOPAMPS
    AMPL(K,I)=(ZERO,ZERO)
    IF (.NOT.COMPUTE_INTEGRAND_IN_QP) THEN
      DP_AMPL(K,I)=(0.0d0,0.0d0)
	ENDIF
  ENDDO
ENDDO
## }

  IF (COMPUTE_INTEGRAND_IN_QP) THEN
    %(mp_born_ct_helas_calls)s
    CONTINUE
  ELSE
    %(born_ct_helas_calls)s
    CONTINUE
  ENDIF

2000 CONTINUE
  MP_CT_REQ_SO_DONE=.TRUE.

  IF (COMPUTE_INTEGRAND_IN_QP) THEN
    %(mp_uvct_helas_calls)s
    CONTINUE
  ELSE
    %(uvct_helas_calls)s   
    CONTINUE
  ENDIF

  IF (.NOT.COMPUTE_INTEGRAND_IN_QP) THEN
C   Copy back to the quantities computed in DP in the QP containers (but only those needed)
## if(not LoopInduced) {
    DO I=1,NBORNAMPS
      AMP(I)=CMPLX(DP_AMP(I),KIND=16)
    ENDDO
## }
    DO I=1,NCTAMPS
      DO K=1,3
        AMPL(K,I)=CMPLX(DP_AMPL(K,I),KIND=16)
      ENDDO
    ENDDO
    DO I=1,NWAVEFUNCS
      DO J=1,MAXLWFSIZE+4
        W(J,I)=CMPLX(DPW(J,I),KIND=16)
      ENDDO
    ENDDO
  ENDIF

3000 CONTINUE
  MP_UVCT_REQ_SO_DONE=.TRUE.

## if(not AmplitudeReduction) { 
  IF (COMPUTE_INTEGRAND_IN_QP) THEN

  DO J=1,NBORNAMPS
    CTEMP = HEL_MULT*2.0e0_16*CONJG(AMP(J))
	DO I=1,NCTAMPS	
	  CFTOT=CMPLX(CF_N(I,J)/REAL(ABS(CF_D(I,J)),KIND=16),0.0e0_16,KIND=16)
      IF(CF_D(I,J).LT.0) CFTOT=CFTOT*IMAG1
      ITEMP = %(proc_prefix)sML5SQSOINDEX(%(proc_prefix)sML5SOINDEX_FOR_LOOP_AMP(I),%(proc_prefix)sML5SOINDEX_FOR_BORN_AMP(J))
	  IF (.NOT.FILTER_SO.OR.SQSO_TARGET.EQ.ITEMP) THEN
        DO K=1,3
          TEMP2 = REAL(CFTOT*AMPL(K,I)*CTEMP,KIND=16)
          ANS(K,ITEMP)=ANS(K,ITEMP)+TEMP2
          ANS(K,0)=ANS(K,0)+TEMP2
        ENDDO
      ENDIF
    ENDDO
  ENDDO

  ELSE

  DO J=1,NBORNAMPS  
    DP_CTEMP = HEL_MULT*2.0d0*DCONJG(DP_AMP(J)) 
    DO I=1,NCTAMPS
	  DP_CFTOT=CMPLX(CF_N(I,J)/REAL(ABS(CF_D(I,J)),KIND=8),0.0d0,KIND=8)
      IF(CF_D(I,J).LT.0) DP_CFTOT=DP_CFTOT*DP_IMAG1
      ITEMP = %(proc_prefix)sML5SQSOINDEX(%(proc_prefix)sML5SOINDEX_FOR_LOOP_AMP(I),%(proc_prefix)sML5SOINDEX_FOR_BORN_AMP(J))
	  IF (.NOT.FILTER_SO.OR.SQSO_TARGET.EQ.ITEMP) THEN
        DO K=1,3
          DP_TEMP2 = REAL(DP_CFTOT*DP_AMPL(K,I)*DP_CTEMP,KIND=8)
          ANSDP(K,ITEMP)=ANSDP(K,ITEMP)+DP_TEMP2
          ANSDP(K,0)=ANSDP(K,0)+DP_TEMP2
        ENDDO
      ENDIF
    ENDDO
  ENDDO

  ENDIF

## }

  IF (COMPUTE_INTEGRAND_IN_QP) THEN
    
    %(mp_coef_construction)s  

  ELSE

    %(coef_construction)s

C   Copy back to the coefficients computed in DP in the QP containers
	DO I=0,LOOPMAXCOEFS-1
	  DO K=1,NLOOPGROUPS
## if(not AmplitudeReduction){
	    DO J=1,NSQUAREDSO
	      LOOPCOEFS(I,J,K)=CMPLX(DP_LOOPCOEFS(I,J,K),KIND=16) 
        ENDDO
## } else {
	      LOOPCOEFS(I,K)=CMPLX(DP_LOOPCOEFS(I,K),KIND=16)        
## }
      ENDDO
	ENDDO
  ENDIF

  4000 CONTINUE
  MP_LOOP_REQ_SO_DONE=.TRUE.

## if(AmplitudeReduction) {
  IF (COMPUTE_INTEGRAND_IN_QP) THEN
## if(not LoopInduced) {
C Copy the multiple precision Born amplitudes computed to the AMP double precision array for its use later if necessary (i.e. color flows for example.)
  DO I=1,NBORNAMPS
    DP_AMP(I)=CMPLX(AMP(I),KIND=8)
  ENDDO
## }

C Copy the multiple precision CT amplitudes computed to the AMPL double precision array for its use later if necessary (i.e. color flows for example.)
  DO I=1,NCTAMPS
    DO K=1,3
      DP_AMPL(K,I)=CMPLX(AMPL(K,I),KIND=8)
    ENDDO
  ENDDO

  ENDIF
## }

c Copy the qp wfs to the dp ones as they are used to setup the CT calls.
C This needs to be done once since only the momenta of these WF matters.
  IF(.NOT.DPW_COPIED.AND.COMPUTE_INTEGRAND_IN_QP) THEN
    DO I=1,NWAVEFUNCS
      DO J=1,MAXLWFSIZE+4
        DPW(J,I)=CMPLX(W(J,I),KIND=8)
      ENDDO
    ENDDO
    DPW_COPIED=.True.
  ENDIF

## if(AmplitudeReduction) {
  DO I=1,NSQUAREDSO
    DO J=1,NLOOPGROUPS
      S(I,J)=.TRUE.
    ENDDO
  ENDDO
  
C We need a dummy argument for the squared order index to conform to the 
C structure that the call to the LOOP* subroutine takes for processes with Born diagrams.
  I_SO=1
%(loop_CT_calls)s
5000 CONTINUE
  CTCALL_REQ_SO_DONE=.TRUE.

C Copy the loop amplitudes computed (whose final result was stored in a double
C precision variable) to the AMPL multiple precision array.
  DO I=NCTAMPS+1,NLOOPAMPS
    DO K=1,3
      AMPL(K,I)=CMPLX(DP_AMPL(K,I),KIND=16)
    ENDDO
  ENDDO

## if(ComputeColorFlows) {
  IF (DIRECT_ME_COMPUTATION) THEN
## }
  DO I=1,NLOOPAMPS
## if(not LoopInduced) {
    DO J=1,NBORNAMPS
## } else {
    DO J=1,NLOOPAMPS
## }

      IF (COMPUTE_INTEGRAND_IN_QP) THEN

	  CFTOT=CMPLX(CF_N(I,J)/REAL(ABS(CF_D(I,J)),KIND=16),0.0e0_16,KIND=16)
      IF(CF_D(I,J).LT.0) CFTOT=CFTOT*IMAG1
## if(not LoopInduced) {
      ITEMP = %(proc_prefix)sML5SQSOINDEX(%(proc_prefix)sML5SOINDEX_FOR_LOOP_AMP(I),%(proc_prefix)sML5SOINDEX_FOR_BORN_AMP(J))
      DO K=1,3
        TEMP2(K) = 2.0e0_16*HEL_MULT*REAL(CFTOT*AMPL(K,I)*CONJG(AMP(J)),KIND=16)
      ENDDO
## } else {
      ITEMP = %(proc_prefix)sML5SQSOINDEX(%(proc_prefix)sML5SOINDEX_FOR_LOOP_AMP(I),%(proc_prefix)sML5SOINDEX_FOR_LOOP_AMP(J))
      TEMP2(1) = HEL_MULT*REAL(CFTOT*AMPL(1,I)*CONJG(AMPL(1,J)),KIND=16)
C Computing the quantities below is not strictly necessary since the result should be finite
C It is however a good cross-check.
      TEMP2(2) = HEL_MULT*REAL(CFTOT*(AMPL(2,I)*CONJG(AMPL(1,J)) + AMPL(1,I)*CONJG(AMPL(2,J))),KIND=16)
      TEMP2(3) = HEL_MULT*REAL(CFTOT*(AMPL(3,I)*CONJG(AMPL(1,J)) + AMPL(1,I)*CONJG(AMPL(3,J))+AMPL(2,I)*CONJG(AMPL(2,J))),KIND=16)
## }

      ELSE

	  DP_CFTOT=CMPLX(CF_N(I,J)/REAL(ABS(CF_D(I,J)),KIND=8),0.0d0,KIND=8)
      IF(CF_D(I,J).LT.0) DP_CFTOT=DP_CFTOT*DP_IMAG1
## if(not LoopInduced) {
      ITEMP = %(proc_prefix)sML5SQSOINDEX(%(proc_prefix)sML5SOINDEX_FOR_LOOP_AMP(I),%(proc_prefix)sML5SOINDEX_FOR_BORN_AMP(J))
      DO K=1,3
        DP_TEMP2(K) = 2.0d0*HEL_MULT*REAL(DP_CFTOT*DP_AMPL(K,I)*CONJG(DP_AMP(J)),KIND=8)
      ENDDO
## } else {
      ITEMP = %(proc_prefix)sML5SQSOINDEX(%(proc_prefix)sML5SOINDEX_FOR_LOOP_AMP(I),%(proc_prefix)sML5SOINDEX_FOR_LOOP_AMP(J))
      DP_TEMP2(1) = HEL_MULT*REAL(DP_CFTOT*DP_AMPL(1,I)*DCONJG(DP_AMPL(1,J)),KIND=8)
C Computing the quantities below is not strictly necessary since the result should be finite
C It is however a good cross-check.
      DP_TEMP2(2) = HEL_MULT*REAL(DP_CFTOT*(DP_AMPL(2,I)*DCONJG(DP_AMPL(1,J)) + DP_AMPL(1,I)*DCONJG(DP_AMPL(2,J))),KIND=8)
      DP_TEMP2(3) = HEL_MULT*REAL(DP_CFTOT*(DP_AMPL(3,I)*DCONJG(DP_AMPL(1,J)) + DP_AMPL(1,I)*DCONJG(DP_AMPL(3,J))+DP_AMPL(2,I)*DCONJG(DP_AMPL(2,J))),KIND=8)
## }

	  ENDIF

C To mimic the structure of the non loop-induced processes, we add here the squared counterterm contribution directly the result ANS and put the loop contributions in the LOOPRES array which will be added to ANS later
      IF (I.LE.NCTAMPS) THEN
        IF (.NOT.FILTER_SO.OR.SQSO_TARGET.EQ.ITEMP) THEN
          DO K=1,3
            IF (COMPUTE_INTEGRAND_IN_QP) THEN
              ANS(K,ITEMP)=ANS(K,ITEMP)+TEMP2(K)
              ANS(K,0)=ANS(K,0)+TEMP2(K)
			ELSE
              ANSDP(K,ITEMP)=ANSDP(K,ITEMP)+DP_TEMP2(K)
              ANSDP(K,0)=ANSDP(K,0)+DP_TEMP2(K)
			ENDIF
          ENDDO
        ENDIF
      ELSE
        DO K=1,3
C This LOOPRES array entries will be added to the main result ANS(*,*) later in the loop_matrix.f file. It is in double precision however, so the cast of the temporary variable TEMP2 is necessary.
          IF (COMPUTE_INTEGRAND_IN_QP) THEN
            LOOPRES(K,ITEMP,I-NCTAMPS)=LOOPRES(K,ITEMP,I-NCTAMPS)+DBLE(TEMP2(K))
		  ELSE
            LOOPRES(K,ITEMP,I-NCTAMPS)=LOOPRES(K,ITEMP,I-NCTAMPS)+DP_TEMP2(K)		  
		  ENDIF
C During the evaluation of the AMPL, we had stored the stability in S(1,*) so we now copy over this flag to the relevant contributing Squared orders.
          S(ITEMP,I-NCTAMPS)=S(1,I-NCTAMPS)
        ENDDO
      ENDIF
    ENDDO
  ENDDO

## if(ComputeColorFlows) {
  ENDIF
## }
  
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
      DO K=1,3
        DO I=0,NSQUAREDSO
          IF (COMPUTE_INTEGRAND_IN_QP) THEN		
            ANS(K,I)=ANS(K,I)+REAL(BUFFRES(K,I),KIND=16)
		  ELSE
            ANSDP(K,I)=ANSDP(K,I)+BUFFRES(K,I)
		  ENDIF
        ENDDO
      ENDDO
C     In this case, we temporarily store the compute Born ME in RES_FROM_JAMP(0,I), which will be used to set ANS(0,I) just after the call to this subroutine in loop_matrix.f
      DO I=0,NSQUAREDSO
        RES_FROM_JAMP(0,I)=RES_FROM_JAMP(0,I)+BUFFRES(0,I)
      ENDDO
C   When setting up the loop filter, it is important to set the quantitied LOOPRES.
C   Notice that you may have a more powerful filter with the direct computation mode because it can filter vanishing loop contributions for a particular squared split order only
C   The quantity LOOPRES defined below is not physical, but it's ok since it is only intended for the loop filtering.
C   In principle it is no longer necessary to compute the quantity below once the loop filter is setup, but it takes a negligible amount of time compare to the quad prec computations. 
      DO J=1,NLOOPGROUPS
        DO I=1,NSQUAREDSO
          DO K=1,3
            LOOPRES(K,I,J)=LOOPRES(K,I,J)+DP_AMPL(K,NCTAMPS+J)
          ENDDO
        ENDDO
      ENDDO
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


C If we were not computing the integrand in QP, then we were already updating ANSDP all along, so that fetching it here from the QP ANS(:,:) should not be done.
IF (COMPUTE_INTEGRAND_IN_QP) THEN
  DO I=1,3
    DO J=0,NSQUAREDSO
      ANSDP(I,J)=REAL(ANS(I,J),KIND=8)
    ENDDO
  ENDDO
ENDIF

## if(not AmplitudeReduction) {
%(mp_coef_merging)s
C If some kind of coefficient merging was done above, do not forget to copy back the LOOPCOEFS merged into DP_LOOPCOEFS if COMPUTE_INTEGRAND_IN_QP is False.
## }

END
