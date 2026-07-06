C Set of subroutines to project the Feynman diagrama amplitudes AMPL onto the color flow space.

      SUBROUTINE %(proc_prefix)sCOMPUTE_COLOR_FLOWS(HEL_MULT,DO_CUMULATIVE)
      IMPLICIT NONE
      LOGICAL DO_CUMULATIVE
      INTEGER HEL_MULT
        CALL %(proc_prefix)sREINITIALIZE_JAMPS()
        CALL %(proc_prefix)sDO_COMPUTE_COLOR_FLOWS(.FALSE.)
      END SUBROUTINE

      SUBROUTINE %(proc_prefix)sCOMPUTE_COLOR_FLOWS_DERIVED_QUANTITIES(HEL_MULT)
      IMPLICIT NONE
      INTEGER HEL_MULT      
## if(MadEventOutput) {
        CALL %(proc_prefix)sCOMPUTE_AMPL2(HEL_MULT)
        CALL %(proc_prefix)sCOMPUTE_JAMP2(HEL_MULT)  
## }
        CONTINUE
      END SUBROUTINE
      
      SUBROUTINE %(proc_prefix)sDEALLOCATE_COLOR_FLOWS()
      IMPLICIT NONE
        CALL %(proc_prefix)sDO_COMPUTE_COLOR_FLOWS(.TRUE.)
      END SUBROUTINE
            
      SUBROUTINE %(proc_prefix)sDO_COMPUTE_COLOR_FLOWS(CLEANUP)
      implicit none
C  
C CONSTANTS 
C 
      CHARACTER*512 Proc_Prefix
      PARAMETER ( Proc_Prefix='%(proc_prefix)s')
      CHARACTER*512 LoopColorFlowCoefsName
      PARAMETER ( LoopColorFlowCoefsName='LoopColorFlowCoefs.dat')
## if(not LoopInduced){
      CHARACTER*512 BornColorFlowCoefsName
      PARAMETER ( BornColorFlowCoefsName='BornColorFlowCoefs.dat')
## }
      %(complex_dp_format)s IMAG1
      PARAMETER (IMAG1=(0.0d0,1.0d0))
## if(LoopInduced and MadEventOutput){
      INTEGER NCONFIGS
      PARAMETER (NCONFIGS=%(nmultichannel_configs)i)
## }

## if (not LoopInduced) {
      INTEGER NBORNAMPS
      PARAMETER (NBORNAMPS=%(nbornamps)d)
## }
      INTEGER    NLOOPAMPS
      PARAMETER (NLOOPAMPS=%(nloopamps)d)
      INTEGER    NSQUAREDSO, NLOOPAMPSO
      PARAMETER (NSQUAREDSO=%(nSquaredSO)d, NLOOPAMPSO=%(nAmpSO)d)
      INTEGER NLOOPFLOWS
      PARAMETER (NLOOPFLOWS=%(nLoopFlows)d)   
## if(not LoopInduced){
      INTEGER NBORNFLOWS
      PARAMETER (NBORNFLOWS=%(nBornFlows)d) 
      INTEGER    NBORNAMPSO
      PARAMETER (NBORNAMPSO=NLOOPAMPSO)

## }
C  
C LOCAL VARIABLES 
C
C When this subroutine is called with CLEANUP=True, it deallocates all its
C arrays
      LOGICAL CLEANUP
C Storing concatenated filenames
      CHARACTER*512 TMP
      CHARACTER*512 LoopColorFlowCoefsN
## if(not LoopInduced){
      CHARACTER*512 BornColorFlowCoefsN
## }

      INTEGER I, J, K, SOINDEX, ARRAY_SIZE
      %(complex_dp_format)s PROJ_COEF

## if(LoopInduced and MadEventOutput){
      integer config_index
      integer CONFIG_MAP(NLOOPAMPS)
      %(config_map_definition)s
## }

C  
C FUNCTIONS
C
      INTEGER %(proc_prefix)sML5SOINDEX_FOR_BORN_AMP
      INTEGER %(proc_prefix)sML5SOINDEX_FOR_LOOP_AMP
C
C GLOBAL VARIABLES
C
      character(512) MLPath
      common/MLPATH/MLPath

## if(not LoopInduced) {
      %(complex_dp_format)s AMP(NBORNAMPS)
      common/%(proc_prefix)sAMPS/AMP
## }
      %(complex_dp_format)s AMPL(3,NLOOPAMPS)
      common/%(proc_prefix)sAMPL/AMPL
      %(complex_dp_format)s JAMPL(3,NLOOPFLOWS,NLOOPAMPSO)
      common/%(proc_prefix)sJAMPL/JAMPL
## if(LoopInduced and MadEventOutput){
      %(complex_dp_format)s JAMPL_FOR_AMP2(3,NLOOPFLOWS,NLOOPAMPSO,NCONFIGS)
      common/%(proc_prefix)sJAMPL_FOR_AMP2/JAMPL_FOR_AMP2
## }

## if(not LoopInduced){
      %(complex_dp_format)s JAMPB(NBORNFLOWS,NBORNAMPSO)
      common/%(proc_prefix)sJAMPB/JAMPB
## }

C Now a more advanced data structure for storing the projection coefficient
C This is of course not F77 standard but widely supported by now.
      TYPE ProjCoeffs
        integer, dimension(:), allocatable :: Num
        integer, dimension(:), allocatable :: Denom
        integer, dimension(:), allocatable :: AmpID           
      ENDTYPE ProjCoeffs
      TYPE(ProjCoeffs), dimension(NLOOPFLOWS), save :: LoopColorProjector
## if(not LoopInduced){
      TYPE(ProjCoeffs), dimension(NBORNFLOWS), save :: BornColorProjector
## }

C ----------
C BEGIN CODE
C ----------

C CleanUp duties, i.e. array deallocation
if (CleanUp) then
  DO I=1,NLOOPFLOWS
    if(ALLOCATED(LoopColorProjector(I)%Num)) deallocate(LoopColorProjector(I)%Num)
    if(ALLOCATED(LoopColorProjector(I)%Denom)) deallocate(LoopColorProjector(I)%Denom)
    if(ALLOCATED(LoopColorProjector(I)%AmpID)) deallocate(LoopColorProjector(I)%AmpID)
  ENDDO
## if(not LoopInduced){
  DO I=1,NBORNFLOWS
    if(ALLOCATED(BornColorProjector(I)%Num)) deallocate(BornColorProjector(I)%Num)
    if(ALLOCATED(BornColorProjector(I)%Denom)) deallocate(BornColorProjector(I)%Denom)
    if(ALLOCATED(BornColorProjector(I)%AmpID)) deallocate(BornColorProjector(I)%AmpID)
  ENDDO
## }  
  return
endif

C Initialization; must allocate array from data files. All arrays are allocated at once, so we only need to check if LoopColorProjector(0)%Num is allocated and all other arrays must share the same status.
if(.not.allocated(LoopColorProjector(1)%Num)) then
  CALL JOINPATH(MLPATH,PROC_PREFIX,TMP)
  CALL JOINPATH(TMP,LoopColorFlowCoefsName,LoopColorFlowCoefsN)
## if(not LoopInduced){
  CALL JOINPATH(TMP,BornColorFlowCoefsName,BornColorFlowCoefsN)
## }

C Initialize the LoopColorProjector
  OPEN(1, FILE=LoopColorFlowCoefsN, err=201, status='OLD',           action='READ')
  DO I=1,NLOOPFLOWS
    READ(1,*,END=998) ARRAY_SIZE
    allocate(LoopColorProjector(I)%Num(ARRAY_SIZE))
    allocate(LoopColorProjector(I)%Denom(ARRAY_SIZE))
    allocate(LoopColorProjector(I)%AmpID(ARRAY_SIZE))
    READ(1,*,END=998) (LoopColorProjector(I)%Num(J),J=1,ARRAY_SIZE)
    READ(1,*,END=998) (LoopColorProjector(I)%Denom(J),J=1,ARRAY_SIZE)
    READ(1,*,END=998) (LoopColorProjector(I)%AmpID(J),J=1,ARRAY_SIZE)
  ENDDO
  GOTO 203
201  CONTINUE
  STOP 'Color projection coefficients could not be initialized from file %(proc_prefix)sLoopColorFlowCoefs.dat.' 
203  CONTINUE
  CLOSE(1)

## if (not LoopInduced) {
C Initialize the BornColorProjector
  OPEN(1, FILE=BornColorFlowCoefsN, err=301, status='OLD',           action='READ')
  DO I=1,NBORNFLOWS
    READ(1,*,END=998) ARRAY_SIZE
    allocate(BornColorProjector(I)%Num(ARRAY_SIZE))
    allocate(BornColorProjector(I)%Denom(ARRAY_SIZE))
    allocate(BornColorProjector(I)%AmpID(ARRAY_SIZE))
    READ(1,*,END=998) (BornColorProjector(I)%Num(J),J=1,ARRAY_SIZE)
    READ(1,*,END=998) (BornColorProjector(I)%Denom(J),J=1,ARRAY_SIZE)
    READ(1,*,END=998) (BornColorProjector(I)%AmpID(J),J=1,ARRAY_SIZE)
  ENDDO
  GOTO 303
301  CONTINUE
  STOP 'Color projection coefficients could not be initialized from file %(proc_prefix)sBornColorFlowCoefs.dat.'  
303  CONTINUE
  CLOSE(1)

## }
  GOTO 999
998 CONTINUE
  STOP 'End of file reached. Should not have happened.' 
999 CONTINUE

ENDIF

c Here we calculate JAMPB
## if(not LoopInduced){
C Start by projecting the Born amplitudes
DO I=1,NBORNFLOWS
  DO J=1,SIZE(BornColorProjector(I)%AmpID)
    SOINDEX = %(proc_prefix)sML5SOINDEX_FOR_BORN_AMP(BornColorProjector(I)%AmpID(J))
    PROJ_COEF=DCMPLX(BornColorProjector(I)%Num(J)/DBLE(ABS(BornColorProjector(I)%Denom(J))),0.0d0)
    IF(BornColorProjector(I)%Denom(J).LT.0) PROJ_COEF=PROJ_COEF*IMAG1
    JAMPB(I,SOINDEX) = JAMPB(I,SOINDEX) + PROJ_COEF*AMP(BornColorProjector(I)%AmpID(J))
  ENDDO
ENDDO

## }
C Projection of the loop amplitudes
DO I=1,NLOOPFLOWS
  DO J=1,SIZE(LoopColorProjector(I)%AmpID)
    SOINDEX = %(proc_prefix)sML5SOINDEX_FOR_LOOP_AMP(LoopColorProjector(I)%AmpID(J))
    PROJ_COEF=DCMPLX(LoopColorProjector(I)%Num(J)/DBLE(ABS(LoopColorProjector(I)%Denom(J))),0.0d0)
    IF(LoopColorProjector(I)%Denom(J).LT.0) PROJ_COEF=PROJ_COEF*IMAG1
    DO K=1,3
      JAMPL(K,I,SOINDEX) = JAMPL(K,I,SOINDEX) + PROJ_COEF*AMPL(K,LoopColorProjector(I)%AmpID(J))
## if(LoopInduced and MadEventOutput){
      config_index = CONFIG_MAP(LoopColorProjector(I)%AmpID(J))
      IF (config_index.ne.0) THEN
        JAMPL_FOR_AMP2(K,I,SOINDEX,config_index) = JAMPL_FOR_AMP2(K,I,SOINDEX,config_index) + PROJ_COEF*AMPL(K,LoopColorProjector(I)%AmpID(J))
      ENDIF
## }
    ENDDO
  ENDDO
ENDDO

      END SUBROUTINE

c This subroutine initializes the color flow matrix.
      SUBROUTINE %(proc_prefix)sINITIALIZE_FLOW_COLORMATRIX()
      implicit none
C  
C CONSTANTS 
C 
      CHARACTER*512 Proc_Prefix
      PARAMETER ( Proc_Prefix='%(proc_prefix)s')
      CHARACTER*512 LoopColorFlowMatrixName
      PARAMETER ( LoopColorFlowMatrixName='LoopColorFlowMatrix.dat')
## if(not LoopInduced){
      CHARACTER*512 BornColorFlowMatrixName
      PARAMETER ( BornColorFlowMatrixName='BornColorFlowMatrix.dat')
## }

      INTEGER NLOOPFLOWS
      PARAMETER (NLOOPFLOWS=%(nLoopFlows)d)   
## if(not LoopInduced){
      INTEGER NBORNFLOWS
      PARAMETER (NBORNFLOWS=%(nBornFlows)d)
## }
C  
C LOCAL VARIABLES 
C
C Storing concatenated filenames
      CHARACTER*512 TMP
      CHARACTER*512 LoopColorFlowMatrixN
## if(not LoopInduced){
      CHARACTER*512 BornColorFlowMatrixN
## }
      INTEGER I, J
C
C GLOBAL VARIABLES
C
      character(512) MLPath
      common/MLPATH/MLPath

C Now a more advanced data structure for storing the projection coefficient
C This is of course not F77 standard but widely supported by now.
      TYPE ColorCoeff
        sequence
        integer :: Num
        integer :: Denom          
      ENDTYPE ColorCoeff
## if(not LoopInduced){
      TYPE(ColorCoeff), dimension(NLOOPFLOWS,NBORNFLOWS) :: LoopColorFlowMatrix
      TYPE(ColorCoeff), dimension(NBORNFLOWS,NBORNFLOWS) :: BornColorFlowMatrix
## } else {
      TYPE(ColorCoeff), dimension(NLOOPFLOWS,NLOOPFLOWS) :: LoopColorFlowMatrix
## }
      LOGICAL CMInitialized
      DATA CMInitialized/.FALSE./
## if(not LoopInduced){
      COMMON/%(proc_prefix)sFLOW_COLOR_MATRIX/LoopColorFlowMatrix, BornColorFlowMatrix, CMInitialized
## } else {
      COMMON/%(proc_prefix)sFLOW_COLOR_MATRIX/LoopColorFlowMatrix, CMInitialized
## }

C ----------
C BEGIN CODE
C ----------

C Initialization
if(.NOT.CMInitialized) then
  CMInitialized = .TRUE.
  CALL JOINPATH(MLPATH,PROC_PREFIX,TMP)
  CALL JOINPATH(TMP,LoopColorFlowMatrixName,LoopColorFlowMatrixN)
## if(not LoopInduced){
  CALL JOINPATH(TMP,BornColorFlowMatrixName,BornColorFlowMatrixN)
C Initialize the BornColorFlowMatrix
  OPEN(1, FILE=BornColorFlowMatrixN, err=601, status='OLD',           action='READ')
  DO I=1,NBORNFLOWS
    READ(1,*,END=898) (BornColorFlowMatrix(I,J)%Num,J=1,NBORNFLOWS)
    READ(1,*,END=898) (BornColorFlowMatrix(I,J)%Denom,J=1,NBORNFLOWS)
  ENDDO
  GOTO 603
601  CONTINUE
  STOP 'Color factors could not be initialized from file %(proc_prefix)sBornColorFlowMatrix.dat.' 
603  CONTINUE
  CLOSE(1)
## }
C Initialize the LoopColorFlowMatrix
  OPEN(1, FILE=LoopColorFlowMatrixN, err=701, status='OLD',           action='READ')
  DO I=1,NLOOPFLOWS
## if(not LoopInduced){
    READ(1,*,END=898) (LoopColorFlowMatrix(I,J)%Num,J=1,NBORNFLOWS)
    READ(1,*,END=898) (LoopColorFlowMatrix(I,J)%Denom,J=1,NBORNFLOWS)
## } else {
    READ(1,*,END=898) (LoopColorFlowMatrix(I,J)%Num,J=1,NLOOPFLOWS)
    READ(1,*,END=898) (LoopColorFlowMatrix(I,J)%Denom,J=1,NLOOPFLOWS)
## }
  ENDDO
  GOTO 703
701  CONTINUE
  STOP 'Color factors could not be initialized from file %(proc_prefix)sBornColorFlowMatrix.dat.' 
703  CONTINUE
  CLOSE(1)

  GOTO 899
898 CONTINUE
  STOP 'End of file reached. Should not have happened.' 
899 CONTINUE

ENDIF
      END SUBROUTINE

## if(LoopInduced and MadEventOutput){
C For the loop-inducedd MadEvent this subroutine computes the squared of 
C each individual JAMPL to created the usual JAMP2 array used for the
C color assignation of the events.
      SUBROUTINE %(proc_prefix)sCOMPUTE_JAMP2(HEL_MULT)
      implicit none
C  
C CONSTANTS 
C 
      %(complex_dp_format)s IMAG1
      PARAMETER (IMAG1=(0.0d0,1.0d0))

      INTEGER    NSQUAREDSO, NLOOPAMPSO
      PARAMETER (NSQUAREDSO=%(nSquaredSO)d, NLOOPAMPSO=%(nAmpSO)d)
      INTEGER NLOOPFLOWS
      PARAMETER (NLOOPFLOWS=%(nLoopFlows)d)   
C  
C ARGUMENT 
C
C HEL_MULT is the helicity multiplier which can be more than one if several helicity configuration are mapped onto the one being currently computed.
      integer HEL_MULT
C  
C LOCAL VARIABLES 
C
      INTEGER I, J, M, N, K, ISQSO, DOUBLEFACT
      %(complex_dp_format)s COLOR_COEF
C  
C FUNCTIONS
C
      INTEGER %(proc_prefix)sML5SQSOINDEX
C
C GLOBAL VARIABLES
C
      LOGICAL CHOSEN_SO_CONFIGS(NSQUAREDSO)
      COMMON/%(proc_prefix)sCHOSEN_LOOP_SQSO/CHOSEN_SO_CONFIGS

      %(complex_dp_format)s JAMPL(3,NLOOPFLOWS,NLOOPAMPSO)
      common/%(proc_prefix)sJAMPL/JAMPL

      %(real_dp_format)s JAMP2(NLOOPFLOWS)
      common/%(proc_prefix)sJAMP2/JAMP2

C Now a more advanced data structure for storing the projection coefficient
C This is of course not F77 standard but widely supported by now.
      TYPE ColorCoeff
        sequence
        integer :: Num
        integer :: Denom          
      ENDTYPE ColorCoeff
      TYPE(ColorCoeff), dimension(NLOOPFLOWS,NLOOPFLOWS) :: LoopColorFlowMatrix
      LOGICAL CMInitialized
      COMMON/%(proc_prefix)sFLOW_COLOR_MATRIX/LoopColorFlowMatrix, CMInitialized

C ----------
C BEGIN CODE
C ----------

C Initialization
if(.NOT.CMInitialized) then
  CALL %(proc_prefix)sINITIALIZE_FLOW_COLORMATRIX()
ENDIF

C Compute the Loop ME from the loop color flow amplitudes (JAMPL)
DO I=1,NLOOPFLOWS
  COLOR_COEF=DCMPLX(LoopColorFlowMatrix(I,I)%Num/DBLE(ABS(LoopColorFlowMatrix(I,I)%Denom)),0.0d0)
  IF (LoopColorFlowMatrix(I,I)%Denom.LT.0) COLOR_COEF=COLOR_COEF*IMAG1
  DO M=1,NLOOPAMPSO
    DO N=M,NLOOPAMPSO
      ISQSO = %(proc_prefix)sML5SQSOINDEX(M,N)
      IF(CHOSEN_SO_CONFIGS(ISQSO)) THEN
        IF(M.ne.N) THEN
          DOUBLEFACT=2 
        ELSE 
          DOUBLEFACT=1
        ENDIF
        JAMP2(I)=JAMP2(I)+DABS(DOUBLEFACT*HEL_MULT*DBLE(COLOR_COEF*(JAMPL(1,I,M)*DCONJG(JAMPL(1,I,N)))))
      ENDIF
    ENDDO
  ENDDO
ENDDO

      END SUBROUTINE
## }

c This routine is used as a crosscheck only to make sure the loop ME computed
c from the JAMP is equal to the amplitude computed directly. It is a consistency
c check of the color projection and computation.

      SUBROUTINE %(proc_prefix)sCOMPUTE_RES_FROM_JAMP(RES,HEL_MULT)
      IMPLICIT NONE
C     
C     CONSTANTS 
C     
      INTEGER    NSQUAREDSO, NLOOPAMPSO
      PARAMETER (NSQUAREDSO=%(nSquaredSO)d, NLOOPAMPSO=%(nAmpSO)d)
      INTEGER NLOOPFLOWS
      PARAMETER (NLOOPFLOWS=%(nLoopFlows)d)
C     
C     ARGUMENT 
C     
      DOUBLE COMPLEX RES_CPLX(0:3,0:NSQUAREDSO)
      REAL*8 RES(0:3,0:NSQUAREDSO)
C     HEL_MULT is the helicity multiplier which can be more than one
C      if several helicity configuration are mapped onto the one being
C      currently computed.
      INTEGER HEL_MULT
      DOUBLE COMPLEX JAMPL(3,NLOOPFLOWS,NLOOPAMPSO)
      COMMON/%(proc_prefix)sJAMPL/JAMPL
C     
C     LOCAL VARIABLES 
C     
      INTEGER I, J
C     ----------
C     BEGIN CODE
C     ----------
      CALL %(proc_prefix)sDO_COMPUTE_INTER_JAMP(RES_CPLX, HEL_MULT, JAMPL, JAMPL)
      DO I=0,3
        DO J=0, NSQUAREDSO
          RES(I,J) = DREAL(RES_CPLX(I,J)) !RES is real for matrix element computation and complex for density matrix computation. A conversion is necessary
        ENDDO
      ENDDO
      END SUBROUTINE





INTEGER FUNCTION %(proc_prefix)sGET_HEL_INDEX(NHEL)
c returned th 
c
    implicit none
    integer nexternal
    parameter (nexternal=%(nexternal)i)
    integer NCOMB
    parameter (ncomb=%(ncomb)i)
    LOGICAL FOUND
    INTEGER I
    INTEGER NHEL(NEXTERNAL)
    INTEGER HELC(NEXTERNAL,NCOMB)
    COMMON/ML5_0_HELCONFIGS/HELC
c    
c   START CODE
c
    DO %(proc_prefix)sGET_HEL_INDEX=1,NCOMB
        DO I=1,NEXTERNAL
           IF (HELC(I, %(proc_prefix)sGET_HEL_INDEX).ne.NHEL(I))then 
              exit
           ELSE IF (I.eq.nexternal) then
              return
           endif   
        ENDDO
    ENDDO
END 


## if(LoopInduced and MadEventOutput){
      SUBROUTINE %(proc_prefix)sCOMPUTE_AMPL2(HEL_MULT)
C  
C CONSTANTS 
C 
      INTEGER    NLOOPAMPS
      PARAMETER (NLOOPAMPS=%(nloopamps)d)
      INTEGER    NLOOPDIAGRAMS
      PARAMETER (NLOOPDIAGRAMS=%(nloopdiags)d)
      INTEGER    NCONFIGS
      PARAMETER (NCONFIGS=%(nmultichannel_configs)i)
      INTEGER    NSQUAREDSO, NLOOPAMPSO, NSO
      PARAMETER (NSQUAREDSO=%(nSquaredSO)d, NLOOPAMPSO=%(nAmpSO)d, NSO=%(nSO)d)
C  
C LOCAL VARIABLES 
C
      INTEGER I,J, SOINDEX
      integer config_index_map(NCONFIGS)
      %(config_index_map_definition)s
C
C ARGUMENTS
C 
      INTEGER HEL_MULT
C
C GLOBAL VARIABLES
C
      %(real_dp_format)s AMP2(NLOOPDIAGRAMS)
      common/%(proc_prefix)sAMP2/AMP2
      %(real_dp_format)s  AMP2_ALL(3,NCONFIGS,0:NSQUAREDSO)   
      common/%(proc_prefix)sAMP2_ALL/AMP2_ALL
      LOGICAL CHOSEN_SO_CONFIGS(NSQUAREDSO)
      COMMON/%(proc_prefix)sCHOSEN_LOOP_SQSO/CHOSEN_SO_CONFIGS
C ----------
C BEGIN CODE
C ----------

      do i=1,nconfigs
C The configurations are not necessarily consecutive when grouping subprocesses. So, here, we simply skip those which have no diagrams for this process.
        if (config_index_map(i).eq.0) then
          cycle
        endif
        DO J=1,NSQUAREDSO
C For the multi-channeling, it is in general better to ignore the squared order constraints (which, if they are any, would restrict the AMP2 quantities to be coming from the interference terms only).  
         IF (.TRUE.) THEN
C Chosing the if statement below instead would as a result to apply the squared order constraints to the AMP2 quantities given to MadEvent for the multichanneling.
c         IF (CHOSEN_SO_CONFIGS(I)) THEN
            AMP2(config_index_map(i))=AMP2(config_index_map(i))+DABS(AMP2_ALL(1,i,J))
          ENDIF
        ENDDO
      enddo

      END SUBROUTINE
## }


      SUBROUTINE %(proc_prefix)sDO_COMPUTE_INTER_JAMP(INTER,HEL_MULT,JAMPL1,JAMPL2)
C     Computation of the interference between JAMPL1 and JAMPL2
C     This subroutine is used to compute the matrix element JAMPL1 = JAMPL2 or the density matrix JAMPL1 != JAMPL2
      IMPLICIT NONE
C     
C     CONSTANTS 
C     
      %(complex_dp_format)s IMAG1
      PARAMETER (IMAG1=(0.0d0,1.0d0))

      INTEGER    NSQUAREDSO, NLOOPAMPSO, NSO
      PARAMETER (NSQUAREDSO=%(nSquaredSO)d, NLOOPAMPSO=%(nAmpSO)d, NSO=%(nSO)d)
      INTEGER NLOOPFLOWS
      PARAMETER (NLOOPFLOWS=%(nLoopFlows)d)
## if(LoopInduced and MadEventOutput){
      INTEGER NCONFIGS
      PARAMETER (NCONFIGS=%(nmultichannel_configs)i)
## }
## if(not LoopInduced){
      INTEGER NBORNFLOWS
      PARAMETER (NBORNFLOWS=%(nBornFlows)d)
C     We use the same list of amplitude split order for Born and Loop, so that NBORNAMPSO is always equal to NLOOPAMPSO  
      INTEGER    NBORNAMPSO
      PARAMETER (NBORNAMPSO=NLOOPAMPSO)
## }
C     
C     ARGUMENT 
C     
      %(complex_dp_format)s INTER(0:3,0:NSQUAREDSO)
C     HEL_MULT is the helicity multiplier which can be more than one
C      if several helicity configuration are mapped onto the one being
C      currently computed.
      INTEGER HEL_MULT      
C     
C     LOCAL VARIABLES 
C     
      INTEGER I, J, K, M, N, ISQSO, DOUBLEFACT
## if(LoopInduced) {
      INTEGER LOWERBOUND
## }
## if(LoopInduced and MadEventOutput){
      integer config_i, config_j
## }
      INTEGER ORDERS_A(NSO), ORDERS_B(NSO)
      %(complex_dp_format)s COLOR_COEF
      %(complex_dp_format)s TEMP(3)
## if(LoopInduced and MadEventOutput){
      %(complex_dp_format)s TEMP_AMP2(3,NCONFIGS)
## }
C  
C FUNCTIONS
C
C This function belongs to the loop ME computation (loop_matrix.f) and is prefixed with ML5 
      INTEGER %(proc_prefix)sML5SQSOINDEX
C This function belongs to the Born ME computation (born_matrix.f) and is not prefixed at all
      INTEGER %(proc_prefix)sSQSOINDEX, %(proc_prefix)sSOINDEX_FOR_AMPORDERS
C
C GLOBAL VARIABLES
C
      character(512) MLPath
      common/MLPATH/MLPath
      INTEGER SQSO_TARGET
      common/%(proc_prefix)sSOCHOICE/SQSO_TARGET


      %(complex_dp_format)s JAMPL1(3,NLOOPFLOWS,NLOOPAMPSO)
      %(complex_dp_format)s JAMPL2(3,NLOOPFLOWS,NLOOPAMPSO)

## if(not LoopInduced){
      %(complex_dp_format)s JAMPB1(NBORNFLOWS,NBORNAMPSO)
      %(complex_dp_format)s JAMPB2(NBORNFLOWS,NBORNAMPSO)
C      The common block is broken because there are now 2 JAMP, see how to fix it, Valentin D.
C      common/%(proc_prefix)sJAMPB/JAMPB
## }
## if(LoopInduced and MadEventOutput){
      %(real_dp_format)s AMP2_ALL(3,NCONFIGS,0:NSQUAREDSO)    
      common/%(proc_prefix)sAMP2_ALL/AMP2_ALL
      %(complex_dp_format)s JAMPL_FOR_AMP2(3,NLOOPFLOWS,NLOOPAMPSO,NCONFIGS)
      common/%(proc_prefix)sJAMPL_FOR_AMP2/JAMPL_FOR_AMP2
## }

      LOGICAL UVCT_REQ_SO_DONE,MP_UVCT_REQ_SO_DONE,CT_REQ_SO_DONE,MP_CT_REQ_SO_DONE,LOOP_REQ_SO_DONE,MP_LOOP_REQ_SO_DONE,CTCALL_REQ_SO_DONE,FILTER_SO
      common/%(proc_prefix)sSO_REQS/UVCT_REQ_SO_DONE,MP_UVCT_REQ_SO_DONE,CT_REQ_SO_DONE,MP_CT_REQ_SO_DONE,LOOP_REQ_SO_DONE,MP_LOOP_REQ_SO_DONE,CTCALL_REQ_SO_DONE,FILTER_SO

C Now a more advanced data structure for storing the projection coefficient
C This is of course not F77 standard but widely supported by now.
      TYPE ColorCoeff
        sequence
        integer :: Num
        integer :: Denom          
      ENDTYPE ColorCoeff
## if(not LoopInduced){
      TYPE(ColorCoeff), dimension(NLOOPFLOWS,NBORNFLOWS) :: LoopColorFlowMatrix
      TYPE(ColorCoeff), dimension(NBORNFLOWS,NBORNFLOWS) :: BornColorFlowMatrix
## } else {
      TYPE(ColorCoeff), dimension(NLOOPFLOWS,NLOOPFLOWS) :: LoopColorFlowMatrix
## }
      LOGICAL CMInitialized
## if(not LoopInduced){
      COMMON/%(proc_prefix)sFLOW_COLOR_MATRIX/LoopColorFlowMatrix, BornColorFlowMatrix, CMInitialized
## } else {
      COMMON/%(proc_prefix)sFLOW_COLOR_MATRIX/LoopColorFlowMatrix, CMInitialized
## }

C ----------
C BEGIN CODE
C ----------
## if(LoopInduced and MadEventOutput){
C Always reinitialize AMP2_ALL because it is not a cumulative quantity
CALL %(proc_prefix)sREINITIALIZE_AMP2_ALL()
## }

C Initialization
if(.NOT.CMInitialized) then
  CALL %(proc_prefix)sINITIALIZE_FLOW_COLORMATRIX()
ENDIF

DO I=0,NSQUAREDSO
  DO K=0,3
    INTER(K,I)=0.0d0
  ENDDO
ENDDO

## if(not LoopInduced){
c Compute the Born ME from the Born color flow amplitudes (JAMP)
DO I=1, NBORNFLOWS
  DO J=I, NBORNFLOWS
    COLOR_COEF=DCMPLX(BornColorFlowMatrix(I,J)%Num/DBLE(ABS(BornColorFlowMatrix(I,J)%Denom)),0.0d0)
    IF (BornColorFlowMatrix(I,J)%Denom.LT.0) COLOR_COEF=COLOR_COEF*IMAG1
    DO M=1,NBORNAMPSO
C It may be that this AmpSO index does not receive contribution by the Born amps (because we put the loop and Born amplitude split orders in a common list)
      IF (ABS(JAMPB1(I,M)).eq.0.0d0.or.ABS(JAMPB2(I,M)).eq.0.0d0) CYCLE
      DO N=1,NBORNAMPSO
        IF (ABS(JAMPB1(J,N)).eq.0.0d0.or.ABS(JAMPB2(J,N)).eq.0.0d0) CYCLE
C First fetch what orders the split order indices M, N correspond to
        CALL %(proc_prefix)sML5GET_ORDERS_FOR_AMPSOINDEX(M,ORDERS_A)
        CALL %(proc_prefix)sML5GET_ORDERS_FOR_AMPSOINDEX(N,ORDERS_B)
C Now figure out to which SQSOINDEX these orders together correspond to *in the Born subroutine* (Notice the absence of the ML5 prefix in the functions used then)
        ISQSO = %(proc_prefix)sSQSOINDEX(%(proc_prefix)sSOINDEX_FOR_AMPORDERS(ORDERS_A),%(proc_prefix)sSOINDEX_FOR_AMPORDERS(ORDERS_B))
        IF(J.ne.I) THEN
          DOUBLEFACT=2 
        ELSE 
          DOUBLEFACT=1
        ENDIF
        TEMP(1) = DOUBLEFACT*HEL_MULT*DBLE(COLOR_COEF*JAMPB1(I,M)*DCONJG(JAMPB2(J,N)))
        INTER(0,ISQSO) = INTER(0,ISQSO) + TEMP(1)
        IF((.not.FILTER_SO).or.SQSO_TARGET.eq.-1.or.SQSO_TARGET.eq.ISQSO) THEN
          INTER(0,0) = INTER(0,0) + TEMP(1)
        ENDIF
      ENDDO
    ENDDO
  ENDDO
ENDDO
## }

C Compute the Loop ME from the loop color flow amplitudes (JAMPL)
DO I=1,NLOOPFLOWS
## if(not LoopInduced){
  DO J=1,NBORNFLOWS
## } else {
  DO J=I,NLOOPFLOWS
## }
    COLOR_COEF=DCMPLX(LoopColorFlowMatrix(I,J)%Num/DBLE(ABS(LoopColorFlowMatrix(I,J)%Denom)),0.0d0)
    IF (LoopColorFlowMatrix(I,J)%Denom.LT.0) COLOR_COEF=COLOR_COEF*IMAG1
    DO M=1,NLOOPAMPSO
## if(not LoopInduced){
C It may be that this AmpSO index does not receive contribution by the Loop amps (because we put the loop and Born amplitude split orders in a common list)
      IF((ABS(JAMPL(1,I,M))+ABS(JAMPL(2,I,M))+ABS(JAMPL(3,I,M))).eq.0.0d0) CYCLE
      DO N=1,NBORNAMPSO
C Same for contributions of split order index N by the Born amps
      IF(ABS(JAMPB(J,N)).eq.0.0d0) CYCLE
## } else {
      IF (I.eq.J) THEN
        LOWERBOUND = M
      ELSE
        LOWERBOUND = 1
      ENDIF
      DO N=LOWERBOUND,NLOOPAMPSO
## }
         ISQSO = %(proc_prefix)sML5SQSOINDEX(M,N)
## if(LoopInduced){
        IF(J.ne.I) THEN
          DOUBLEFACT=2 
        ELSEIF (M.ne.N) THEN
          DOUBLEFACT=2
        ELSE
          DOUBLEFACT=1
        ENDIF
C I have removed the conversion to real because JAMPL1 and JAMPL2 are now different. The conversion to dble is done later in COMPUTE_RES_FROM_JAMP if we only want the matrix element
        TEMP(1) = DOUBLEFACT*HEL_MULT*(COLOR_COEF*(JAMPL1(1,I,M)*DCONJG(JAMPL2(1,J,N))))
C Computing the quantities below is not strictly necessary since the result should be finite
C It is however a good cross-check.
        TEMP(2) = DOUBLEFACT*HEL_MULT*(COLOR_COEF*(JAMPL1(2,I,M)*DCONJG(JAMPL2(1,J,N)) + JAMPL1(1,I,M)*DCONJG(JAMPL2(2,J,N))))
        TEMP(3) = DOUBLEFACT*HEL_MULT*(COLOR_COEF*(JAMPL1(3,I,M)*DCONJG(JAMPL2(1,J,N)) + JAMPL1(1,I,M)*DCONJG(JAMPL2(3,J,N))+JAMPL1(2,I,M)*DCONJG(JAMPL2(2,J,N))))
## if(MadEventOutput) {
        DO config_i=1,nconfigs
          DO K=1,3
            TEMP_AMP2(K,config_i) = 0.0d0
          ENDDO
C For the multi-channeling, to choose to square against only among a given configuration, use the do loop below
          DO config_j=config_i,config_i
c For the multi-channeling, to choose to square against all other configuration, use the do loop below
c          DO config_j=1,nconfigs
            TEMP_AMP2(1,config_i) = TEMP_AMP2(1,config_i) + DOUBLEFACT*HEL_MULT*(COLOR_COEF*(JAMPL_FOR_AMP2(1,I,M,config_i)*DCONJG(JAMPL_FOR_AMP2(1,J,N,config_j))))
            TEMP_AMP2(2,config_i) = TEMP_AMP2(2,config_i) + DOUBLEFACT*HEL_MULT*(COLOR_COEF*(JAMPL_FOR_AMP2(2,I,M,config_i)*DCONJG(JAMPL_FOR_AMP2(1,J,N,config_j)) + JAMPL_FOR_AMP2(1,I,M,config_i)*DCONJG(JAMPL_FOR_AMP2(2,J,N,config_j))))
            TEMP_AMP2(3,config_i) = TEMP_AMP2(3,config_i) + DOUBLEFACT*HEL_MULT*(COLOR_COEF*(JAMPL_FOR_AMP2(3,I,M,config_i)*DCONJG(JAMPL_FOR_AMP2(1,J,N,config_j)) + JAMPL_FOR_AMP2(1,I,M,config_i)*DCONJG(JAMPL_FOR_AMP2(3,J,N,config_j))+JAMPL_FOR_AMP2(2,I,M,config_i)*DCONJG(JAMPL_FOR_AMP2(2,J,N,config_j))))

          ENDDO
        ENDDO
## }
## }
        DO K=1,3
## if(not LoopInduced){      
          TEMP(K) = 2.0d0*HEL_MULT*DBLE(COLOR_COEF*JAMPL(K,I,M)*DCONJG(JAMPB(J,N)))
## }
          INTER(K,ISQSO) = INTER(K,ISQSO) + TEMP(K)
## if(LoopInduced and MadEventOutput){
          DO config_i=1,nconfigs
            AMP2_ALL(K,config_i,ISQSO) = AMP2_ALL(K,config_i,ISQSO) + TEMP_AMP2(K,config_i)
          ENDDO
## }
        ENDDO
        IF((.NOT.FILTER_SO).OR.SQSO_TARGET.eq.-1.or.SQSO_TARGET.eq.ISQSO) THEN
          DO K=1,3
            INTER(K,0) = INTER(K,0) + TEMP(K)
## if(LoopInduced and MadEventOutput){
C           Notice here that the squared order constraints will apply to the determination of the AMP2 for multichanneling. Place it outside the corresponding statement if this is not desirable.
            do config_i=1,nconfigs
              AMP2_ALL(K,config_i,0) = AMP2_ALL(K,config_i,0) + TEMP_AMP2(K,config_i)
            enddo
## }
          ENDDO
        ENDIF
      ENDDO
    ENDDO
  ENDDO
ENDDO
      END SUBROUTINE


     SUBROUTINE %(proc_prefix)sSMATRIXHEL(P,HEL,ANS)
      IMPLICIT NONE
C
C CONSTANT
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER                 NCOMB         
      PARAMETER (             NCOMB=%(ncomb)d)
      %(get_nsqso_born)s
      INTEGER NSQUAREDSO
	PARAMETER (NSQUAREDSO=%(nSquaredSO)d)
      INTEGER ANS_DIMENSION
	PARAMETER(ANS_DIMENSION=MAX(NSQSO_BORN,NSQUAREDSO))
CF2PY INTENT(OUT) :: ANS
CF2PY INTENT(IN) ::HEL
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)  

C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL)
      %(real_dp_format)s ANS(0:3,0:ANS_DIMENSION)
      INTEGER HEL
C
C GLOBAL VARIABLES
C
      INTEGER USERHEL
      COMMON/%(proc_prefix)sHELUSERCHOICE/USERHEL
C ----------
C BEGIN CODE
C ----------
      USERHEL=HEL
      CALL %(proc_prefix)sSLOOPMATRIX(P,ANS)
      USERHEL=-1

      END

## if(UseDensity){

C This subroutine resets to 0 the common arrays JAMPL_ALL for the density matrix
SUBROUTINE %(proc_prefix)sREINITIALIZE_JAMP_ALL()
implicit none
C  
C CONSTANTS 
C 
      %(complex_dp_format)s CMPLXZERO
      PARAMETER (CMPLXZERO=(0.0d0,0.0d0))
      %(real_dp_format)s ZERO
      PARAMETER (ZERO=0.0d0)
      INTEGER    NLOOPAMPSO
      PARAMETER (NLOOPAMPSO=%(nAmpSO)d)
      INTEGER NLOOPFLOWS
      PARAMETER (NLOOPFLOWS=%(nLoopFlows)d)
      INTEGER NCOMB
      PARAMETER (NCOMB=%(ncomb)d)
## if(LoopInduced and MadEventOutput){
      INTEGER NCONFIGS
      PARAMETER (NCONFIGS=%(nmultichannel_configs)i)
## }
## if(not LoopInduced){
      INTEGER NBORNFLOWS
      PARAMETER (NBORNFLOWS=%(nBornFlows)d)
C     We use the same list of amplitude split order for Born and Loop, so that NBORNAMPSO is always equal to NLOOPAMPSO
      INTEGER    NBORNAMPSO
      PARAMETER (NBORNAMPSO=NLOOPAMPSO)
## }
C
C LOCAL VARIABLES
C
      INTEGER I,J,K,L
C
C GLOBAL VARIABLES
C
      DOUBLE COMPLEX JAMPL_ALL(3, NLOOPFLOWS,NLOOPAMPSO, NCOMB)
      COMMON/%(proc_prefix)sDENSITY_MATRIX/JAMPL_ALL

C ----------
C BEGIN CODE
C ----------

DO I=1,NLOOPAMPSO
  DO J=1,NLOOPFLOWS
    DO K=1,3
      DO L=1,NCOMB
        JAMPL_ALL(K,J,I,L)=CMPLXZERO
      ENDDO
    ENDDO
  ENDDO
ENDDO
write(*,*) "JAMPL_ALL reinitialised"
END SUBROUTINE
## }

C This subroutine resets to 0 the common arrays JAMPL, JAMPB and possibly JAMPL_FOR_AMP2 if used
c The common array JAMPL_ALL is reseted to 0 in the previous subroutine
SUBROUTINE %(proc_prefix)sREINITIALIZE_JAMPS()
implicit none
C  
C CONSTANTS 
C 
      %(complex_dp_format)s CMPLXZERO
      PARAMETER (CMPLXZERO=(0.0d0,0.0d0))
      %(real_dp_format)s ZERO
      PARAMETER (ZERO=0.0d0)
      INTEGER    NLOOPAMPSO
      PARAMETER (NLOOPAMPSO=%(nAmpSO)d)
      INTEGER NLOOPFLOWS
      PARAMETER (NLOOPFLOWS=%(nLoopFlows)d)
## if(LoopInduced and MadEventOutput){
      INTEGER NCONFIGS
      PARAMETER (NCONFIGS=%(nmultichannel_configs)i)
## }
## if(not LoopInduced){
      INTEGER NBORNFLOWS
      PARAMETER (NBORNFLOWS=%(nBornFlows)d)
C     We use the same list of amplitude split order for Born and Loop, so that NBORNAMPSO is always equal to NLOOPAMPSO
      INTEGER    NBORNAMPSO
      PARAMETER (NBORNAMPSO=NLOOPAMPSO)
## }
C
C LOCAL VARIABLES
C
      INTEGER I,J,K,L
C
C GLOBAL VARIABLES
C
      %(complex_dp_format)s JAMPL(3,NLOOPFLOWS,NLOOPAMPSO)
      common/%(proc_prefix)sJAMPL/JAMPL
## if(not LoopInduced){
      %(complex_dp_format)s JAMPB(NBORNFLOWS,NBORNAMPSO)
      common/%(proc_prefix)sJAMPB/JAMPB
## }
## if(LoopInduced and MadEventOutput){
      %(complex_dp_format)s JAMPL_FOR_AMP2(3,NLOOPFLOWS,NLOOPAMPSO,NCONFIGS)
      common/%(proc_prefix)sJAMPL_FOR_AMP2/JAMPL_FOR_AMP2
## }

C ----------
C BEGIN CODE
C ----------

DO I=1,NLOOPAMPSO
  DO J=1,NLOOPFLOWS
    DO K=1,3
      JAMPL(K,J,I)=CMPLXZERO
    ENDDO
  ENDDO
ENDDO
## if(LoopInduced and MadEventOutput){
DO L=1,NCONFIGS
  DO I=1,NLOOPAMPSO
    DO J=1,NLOOPFLOWS
      DO K=1,3
        JAMPL_FOR_AMP2(K,J,I,L)=CMPLXZERO
      ENDDO
    ENDDO
  ENDDO
ENDDO
## }
## if(not LoopInduced){
DO I=1,NBORNAMPSO
  DO J=1,NBORNFLOWS
    JAMPB(J,I)=CMPLXZERO
  ENDDO
ENDDO
## }

END SUBROUTINE

## if(LoopInduced and MadEventOutput){
C This subroutine resets to 0 the common arrays AMP2_ALL
SUBROUTINE %(proc_prefix)sREINITIALIZE_AMP2_ALL()
implicit none
C  
C CONSTANTS 
C 
      INTEGER    NSQUAREDSO, NLOOPAMPSO, NSO
      PARAMETER (NSQUAREDSO=%(nSquaredSO)d, NLOOPAMPSO=%(nAmpSO)d, NSO=%(nSO)d)
      %(real_dp_format)s ZERO
      PARAMETER (ZERO=0.0d0)
      INTEGER NCONFIGS
      PARAMETER (NCONFIGS=%(nmultichannel_configs)i)
C
C LOCAL VARIABLES
C
      INTEGER I,K,L
C
C GLOBAL VARIABLES
C
      %(real_dp_format)s AMP2_ALL(3,NCONFIGS,0:NSQUAREDSO)
      common/%(proc_prefix)sAMP2_ALL/AMP2_ALL
C ----------
C BEGIN CODE
C ----------

DO I=1,NCONFIGS
  DO K=1,3
    DO L=0,NSQUAREDSO
      AMP2_ALL(K,I,L)=ZERO
    ENDDO
  ENDDO
ENDDO

END SUBROUTINE
## }

C This subroutine resets all cumulative (for each helicity) arrays in common block and deriving from the color flow computation.
C This subroutine is called by loop_matrix.f at appropriate time when these arrays must be reset.
C An example of this is the JAMP2 which must be cumulatively computed as loop_matrix.f loops over helicity configs but must be resets when loop_matrix.f starts over with helicity one (for the stability test for example).
SUBROUTINE %(proc_prefix)sREINITIALIZE_CUMULATIVE_ARRAYS()
implicit none
C  
C CONSTANTS 
C 
      %(complex_dp_format)s CMPLXZERO
      PARAMETER (CMPLXZERO=(0.0d0,0.0d0))
      %(real_dp_format)s ZERO
      PARAMETER (ZERO=0.0d0)
      INTEGER    NLOOPDIAGRAMS
      PARAMETER (NLOOPDIAGRAMS=%(nloopdiags)d)
      INTEGER NLOOPFLOWS
      PARAMETER (NLOOPFLOWS=%(nLoopFlows)d)
## if(LoopInduced and MadEventOutput){
      INTEGER NCONFIGS
      PARAMETER (NCONFIGS=%(nmultichannel_configs)i)
## }
C
C LOCAL VARIABLES
C
      INTEGER I,J,K
C
C GLOBAL VARIABLES
C
## if(LoopInduced and MadEventOutput){
      %(real_dp_format)s AMP2(NLOOPDIAGRAMS)
      common/%(proc_prefix)sAMP2/AMP2
      %(real_dp_format)s JAMP2(NLOOPFLOWS)
      common/%(proc_prefix)sJAMP2/JAMP2
## }

C ----------
C BEGIN CODE
C ----------

## if(LoopInduced and MadEventOutput){
  DO I=1,NLOOPDIAGRAMS
    AMP2(I)=ZERO
  ENDDO
  DO I=1,NLOOPFLOWS
    JAMP2(I)=ZERO
  ENDDO
## }

CONTINUE

END SUBROUTINE

## if(UseDensity) {
# Core of the density matrix calculations
      SUBROUTINE %(proc_prefix)sGET_ALL_INTER(JAMPL_ALL, NHEL, POS, N_CHANGING, ALLOW_HEL, N_COMB_RHO, INTER)
      IMPLICIT NONE

      INCLUDE 'MadLoopParams.inc'
C     
C     CONSTANT
C     
      INTEGER    NSQUAREDSO, NLOOPAMPSO, NSO
      PARAMETER (NSQUAREDSO=%(nSquaredSO)d, NLOOPAMPSO=%(nAmpSO)d, NSO=%(nSO)d)
      INTEGER NLOOPFLOWS
      PARAMETER (NLOOPFLOWS=%(nLoopFlows)d)
      INTEGER NCOMB
      PARAMETER (NCOMB=%(ncomb)d)
      INTEGER HEL_MULT
      PARAMETER (HEL_MULT=1)
C     
C     ARGUMENTS
C     
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)i)
      INTEGER NHEL(NEXTERNAL)
      INTEGER N_CHANGING, N_COMB_RHO
      INTEGER POS(*)
      INTEGER ALLOW_HEL(N_CHANGING * N_COMB_RHO)
      DOUBLE COMPLEX INTER(N_COMB_RHO * (N_COMB_RHO + 1)/2, 0:NSQUAREDSO)

C     
C     Intermediate array
C     
      DOUBLE COMPLEX JAMPL_ALL(3, NLOOPFLOWS,NLOOPAMPSO, NCOMB)
      DOUBLE COMPLEX INTER_TMP(0:3,0:NSQUAREDSO)

C     
C     Common for density
C     
      INTEGER HELC(NEXTERNAL,NCOMB), ORDER_HEL(N_COMB_RHO)
      COMMON/%(proc_prefix)sHELCONFIGS/HELC

C
C     Common for GOODHEL
C
c     the common for goodhel is not good because there is some treatment afterwards in loop_matrix
      INTEGER GOODHEL(NCOMB)
      common/%(proc_prefix)sFilters/GOODHEL
C     The parameter below sets the convention for the helicity filter
C     For a given helicity, the attached integer 'i' means
C     'i' in ]-inf;-HELOFFSET[ -> Helicity is equal, up to a sign, to helicity number abs(i+HELOFFSET)
C     'i' == -HELOFFSET        -> Helicity is analytically zero
C     'i' in ]-HELOFFSET,inf[  -> Helicity is contributing with weight 'i'. If it is zero, it is skipped.
C     Typically, the hel_offset is 10000
      INTEGER HELOFFSET
	DATA HELOFFSET/%(hel_offset)d/
      CHARACTER*512 Proc_Prefix
      PARAMETER ( Proc_Prefix='%(proc_prefix)s')
C     
C     local
C     
      INTEGER I,J, N, SOL, H, K, NUMBERCALL
      CHARACTER*512 HelFilterFN, MLPATH, TMP
      COMMON/MLPATH/MLPATH
      

C Here we setup GOODHEL. If GOODHEL(H) = -10000, its JAMP is 0 so we can skip all its computations
      CALL JOINPATH(MLPATH,PROC_PREFIX,TMP)
      CALL JOINPATH(TMP,'HelFilter.dat',HELFILTERFN)

      OPEN(1, FILE=HelFilterFN, err=102, status='OLD',action='READ')
      DO I=1,NCOMB
      READ(1,*,END=103) GOODHEL(I)
      ENDDO
      GOTO 103
      102  CONTINUE
      DO J=1,NCOMB
            GOODHEL(J)=1
      ENDDO
      103  CONTINUE
      CLOSE(1)

      IF (HelicityFilterLevel.eq.1) THEN
        DO J=1,NCOMB
          IF ((GOODHEL(J).GT.1).OR.(GOODHEL(J).LT.-HELOFFSET)) THEN
            GOODHEL(J)=1
          ENDIF
        ENDDO
      ELSE IF (HelicityFilterLevel.eq.0) THEN
        DO J=1,NCOMB
          GOODHEL(J)=1
        ENDDO
      ELSE
        WRITE(*,*) '### ERROR The parameter HelicityFilterLevel only accepts values of 0 and 1. The value was set to', HelicityFilterLevel
C       We cant stop the code here because in the initialisation phase, HelicityFilterLevel = 2.
      ENDIF

c     Here we determine which helicity state we want to compute and which position it has in the JAMPL_ALL structure
      do I = 1, N_COMB_RHO
      do N = 1, N_CHANGING
      NHEL(POS(N)) = ALLOW_HEL((I-1)*N_CHANGING+N)
      enddo
      CALL %(proc_prefix)sFIND_H(NHEL, H)
      ORDER_HEL(I) = H
      enddo


      NUMBERCALL = 0
      SOL = 0 
      DO I = 1, N_COMB_RHO
        IF (GOODHEL(ORDER_HEL(I)).lt.0) THEN
          DO J = I, N_COMB_RHO
            SOL = SOL + 1
            DO K=0, NSQUAREDSO
              INTER(SOL,K) = 0D0
            ENDDO
          ENDDO
        ELSE
          DO J= I, N_COMB_RHO
            SOL = SOL + 1
            IF (GOODHEL(ORDER_HEL(J)).lt.0) THEN
              DO K=0, NSQUAREDSO
                INTER(SOL,K) = 0D0
              ENDDO
            ELSE
              call %(proc_prefix)sCOMPUTE_INTER_JAMP(INTER_TMP,HEL_MULT,JAMPL_ALL,ORDER_HEL(I), ORDER_HEL(J))
              NUMBERCALL = NUMBERCALL + 1
              DO K=0, NSQUAREDSO
                INTER(SOL,K) = INTER_TMP(1,K) + INTER_TMP(2,K) + INTER_TMP(3,K) !we sum the 3 contributions even though contribution 2 and 3 should be zero
              ENDDO
            ENDIF
          ENDDO
        ENDIF
      ENDDO

c      WRITE(*,*) "NUMBERCALL", NUMBERCALL
      RETURN
      END SUBROUTINE




      SUBROUTINE %(proc_prefix)sGET_DENSITY(P, POS, N_CHANGING, ALLOW_HEL, N_COMB, ALPHAS, SCALE2, INTER)
      IMPLICIT NONE
C     
C     CONSTANT
C     
      INTEGER    NSQUAREDSO, NLOOPAMPSO, NSO
      PARAMETER (NSQUAREDSO=%(nSquaredSO)d, NLOOPAMPSO=%(nAmpSO)d, NSO=%(nSO)d)
      INTEGER NLOOPFLOWS
      PARAMETER (NLOOPFLOWS=%(nLoopFlows)d)
      INTEGER NCOMB
      PARAMETER (NCOMB=%(ncomb)d)
      INTEGER HEL_MULT
      PARAMETER (HEL_MULT=1)
      INTEGER NSQSO_BORN
      PARAMETER (NSQSO_BORN=0)
      INTEGER AUX_DIMENSION
      PARAMETER(AUX_DIMENSION=MAX(NSQSO_BORN,NSQUAREDSO))
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)i)
C     
C     ARGUMENTS
C     
      INTEGER N_CHANGING, N_COMB
      INTEGER THISNHEL(NEXTERNAL)
      INTEGER POS(N_CHANGING)
      INTEGER ALLOW_HEL(N_CHANGING*N_COMB)
      DOUBLE COMPLEX INTER(N_COMB * (N_COMB + 1)/2) !only 1 dimension here, correspond to the 0-element
      DOUBLE COMPLEX TMP_INTER(N_COMB * (N_COMB + 1)/2, 0:NSQUAREDSO)
      DOUBLE PRECISION P(0:3, NEXTERNAL)
      DOUBLE PRECISION ALPHAS, SCALE2

C     
C     Intermediate array
C     

      DOUBLE COMPLEX JAMPL_ALL(3, NLOOPFLOWS,NLOOPAMPSO, NCOMB)
      COMMON/%(proc_prefix)sDENSITY_MATRIX/JAMPL_ALL

      INTEGER HELC(NEXTERNAL,NCOMB), ORDER_HEL(N_COMB)
      COMMON/%(proc_prefix)sHELCONFIGS/HELC
      REAL*8 AUX(0:3,0:AUX_DIMENSION)

C     
C     local
C     
      INTEGER I, IHEL, IPART

c     Maybe should be a .or. ?
      IF ((ALPHAS.ne.0D0).and.(SCALE2.ne.0D0)) THEN
        call UPDATE_AS_PARAM2(SCALE2, ALPHAS)
      ENDIF

      CALL %(proc_prefix)sSLOOPMATRIX(P, AUX)

      do I = 1, N_COMB*(N_COMB+1)/2
        INTER(I) = 0D0
      enddo
      
      DO IHEL =1, NCOMB
        THISNHEL(:) = HELC(:, IHEL)
	    DO IPART=1,N_CHANGING
            if(THISNHEL(POS(IPART)).NE.ALLOW_HEL(IPART)) GOTO 10 !BYPASS COMPUTATION FOR HELICITY
        ENDDO
        call %(proc_prefix)sGET_ALL_INTER(JAMPL_ALL, THISNHEL, POS, N_CHANGING, ALLOW_HEL, N_COMB, TMP_INTER)
        do I = 1, N_COMB*(N_COMB+1)/2
          INTER(I) = INTER(I) + TMP_INTER(I, 0)
        enddo 
 10   enddo
        
c      CALL %(proc_prefix)sNORMALISE_RHO(INTER, N_COMB, RHO_NORM)
c      WRITE(*,*) "DENSITY MATRIX = ", RHO_NORM(:,0)
      RETURN 
      END

     SUBROUTINE %(proc_prefix)sFIND_H(NHEL, H)
C     This subroutine parses HELC to find NHEL in it. It returns H, the position of this helicity state.
C     If it does not find the state in HELC, it returns 0
      IMPLICIT NONE
      INTEGER NLOOPFLOWS, NEXTERNAL
      PARAMETER (NLOOPFLOWS=%(nLoopFlows)d, NEXTERNAL=%(nexternal)i)
      INTEGER NCOMB
      PARAMETER (NCOMB=%(ncomb)d)
      LOGICAL equal
      INTEGER I, H, J
      INTEGER HELC(NEXTERNAL,NCOMB), NHEL(NEXTERNAL)
      COMMON/%(proc_prefix)sHELCONFIGS/HELC

      H = 0
      DO I = 1, NCOMB
        DO J = 1, NEXTERNAL
          equal = (HELC(J,I) == NHEL(J))
          if (.not.equal) exit
        ENDDO
        if (equal) then
        H = I
        exit
        endif
      ENDDO
      RETURN
      END SUBROUTINE

      SUBROUTINE %(proc_prefix)sSTORE_JAMP(H)
C     This subroutine stores the JAMP with helicity state HELC(:,H) in JAMPL_ALL
      IMPLICIT NONE
      INTEGER NLOOPFLOWS, NEXTERNAL
      PARAMETER (NLOOPFLOWS=%(nLoopFlows)d, NEXTERNAL=%(nexternal)i)
      INTEGER NCOMB, NSQUAREDSO
      PARAMETER (NCOMB=%(ncomb)d, NSQUAREDSO=%(nSquaredSO)d)
      INTEGER    NLOOPAMPSO
      PARAMETER (NLOOPAMPSO=%(nAmpSO)d)
      INTEGER H
      INTEGER HELC(NEXTERNAL,NCOMB)

      DOUBLE COMPLEX JAMPL_ALL(3,NLOOPFLOWS,NLOOPAMPSO,NCOMB)
      
      COMMON/%(proc_prefix)sDENSITY_MATRIX/JAMPL_ALL
      COMMON/%(proc_prefix)sHELCONFIGS/HELC
      COMPLEX*16 JAMPL(3,NLOOPFLOWS,NLOOPAMPSO)
      COMMON/%(proc_prefix)sJAMPL/JAMPL

c     THIS JAMPL_ALL CORRESPONDS TO HELC(H)
      JAMPL_ALL(:,:,:,H) = JAMPL(:,:,:) !this line basically stores JAMPL in JAMPL_ALL

c      WRITE(*,*) "JAMP STORED", JAMPL
c      WRITE(*,*) "HELCITY STATE", HELC(:,H)
c      WRITE(*,*) "JAMPL ALL", JAMPL_ALL
c      WRITE(*,*) "----------------"
      RETURN
      END SUBROUTINE


      SUBROUTINE %(proc_prefix)sNORMALISE_RHO(RHO, N_COMB_RHO, RHO_NORM)
C     This subroutine stores the JAMP with helicity state HELC(:,H) in JAMPL_ALL
      IMPLICIT NONE
      INTEGER NLOOPFLOWS, NEXTERNAL
      PARAMETER (NLOOPFLOWS=%(nLoopFlows)d, NEXTERNAL=%(nexternal)i)
      INTEGER N_COMB_RHO
      INTEGER    NLOOPAMPSO, NSQUAREDSO
      PARAMETER (NLOOPAMPSO=%(nAmpSO)d, NSQUAREDSO=%(nSquaredSO)d)

      DOUBLE COMPLEX RHO(N_COMB_RHO * (N_COMB_RHO + 1)/2)
      DOUBLE COMPLEX RHO_NORM(N_COMB_RHO * (N_COMB_RHO + 1)/2)
      DOUBLE PRECISION TRACE
      INTEGER DIAG_INDICES(N_COMB_RHO), COUNTER, IDIAG, I, SOINDEX, J

      COUNTER = 0
      DO IDIAG = 1, N_COMB_RHO
        DIAG_INDICES(IDIAG) = (IDIAG-1) * N_COMB_RHO - COUNTER + 1
        COUNTER = COUNTER + (IDIAG-1)
      ENDDO

      TRACE = 0
      DO I = 1, N_COMB_RHO
        TRACE = TRACE + DREAL(RHO(DIAG_INDICES(I)))
      ENDDO

      DO J = 1, N_COMB_RHO * (N_COMB_RHO + 1)/2
        RHO_NORM(J) = RHO(J) / TRACE
      ENDDO


      RETURN
      END SUBROUTINE



      SUBROUTINE %(proc_prefix)sCOMPUTE_INTER_JAMP(INTER,HEL_MULT,JAMPL_ALL,H1,H2)
C     This function computes the interference between JAMPL_ALL(:,:,:,H1) and JAMPL_ALL(:,:,:,H2)
      IMPLICIT NONE
C     
C     CONSTANTS 
C     
      INTEGER    NSQUAREDSO, NLOOPAMPSO
      PARAMETER (NSQUAREDSO=%(nSquaredSO)d, NLOOPAMPSO=%(nAmpSO)d)
      INTEGER NLOOPFLOWS
      PARAMETER (NLOOPFLOWS=%(nLoopFlows)d)
      INTEGER NCOMB
      PARAMETER (NCOMB=%(ncomb)d)
      INTEGER IDEN
      %(den_factor_line)s
      INTEGER I, SOINDEX
C     
C     ARGUMENTS
C     
      DOUBLE COMPLEX INTER(0:3,0:NSQUAREDSO)
C     HEL_MULT is the helicity multiplier which can be more than one
C      if several helicity configuration are mapped onto the one being
C      currently computed.
      INTEGER HEL_MULT, H1, H2 !helicity states chosen for the 2 JAMPL we compute the interference from
      DOUBLE COMPLEX JAMPL_ALL(3,NLOOPFLOWS,NLOOPAMPSO,NCOMB)
      DOUBLE COMPLEX JAMPL1(3,NLOOPFLOWS,NLOOPAMPSO)
      DOUBLE COMPLEX JAMPL2(3,NLOOPFLOWS,NLOOPAMPSO)

C     ----------
C     BEGIN CODE
C     ----------
C     Initialise JAMPL1 and JAMPL2
      JAMPL1(:,:,:) = JAMPL_ALL(:,:,:,H1)
      JAMPL2(:,:,:) = JAMPL_ALL(:,:,:,H2)

      IF (HEL_MULT.ne.1) THEN 
        ERROR STOP 'HEL_MULT is bigger than 1, it is not yet implemented'
      ENDIF
      CALL %(proc_prefix)sDO_COMPUTE_INTER_JAMP(INTER,HEL_MULT,JAMPL1,JAMPL2)

      DO SOINDEX = 0, NSQUAREDSO
        DO I=0, 3
          INTER(I, SOINDEX) = INTER(I, SOINDEX)/IDEN
        ENDDO
      ENDDO
      
      RETURN
      END SUBROUTINE

## }
