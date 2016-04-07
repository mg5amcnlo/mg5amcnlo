      SUBROUTINE %(proc_prefix)s%(bunch_name)s_%(bunch_number)d(P,NHEL,H,IC)
C
C Modules
C
      use %(proc_prefix)sPOLYNOMIAL_CONSTANTS      
C 
      IMPLICIT NONE
C  
C CONSTANTS
C    
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
	  INTEGER    NCOMB
      PARAMETER (NCOMB=%(ncomb)d)
## if (not LoopInduced) {
	  INTEGER NBORNAMPS
      PARAMETER (NBORNAMPS=%(nbornamps)d)
## }
	  INTEGER    NLOOPS, NLOOPGROUPS, NCTAMPS
      PARAMETER (NLOOPS=%(nloops)d, NLOOPGROUPS=%(nloop_groups)d, NCTAMPS=%(nctamps)d)
	  INTEGER    NLOOPAMPS
	  PARAMETER (NLOOPAMPS=%(nloopamps)d)
      INTEGER    NWAVEFUNCS,NLOOPWAVEFUNCS
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d,NLOOPWAVEFUNCS=%(nloopwavefuncs)d)
      %(real_dp_format)s     ZERO
      PARAMETER (ZERO=0D0)
	  %(real_mp_format)s     MP__ZERO
      PARAMETER (MP__ZERO=0.0e0_16)
C     These are constants related to the split orders
      INTEGER    NSO, NSQUAREDSO, NAMPSO
	  PARAMETER (NSO=%(nSO)d, NSQUAREDSO=%(nSquaredSO)d, NAMPSO=%(nAmpSO)d)
C  
C ARGUMENTS
C
      %(real_dp_format)s P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
	  INTEGER H
C
C LOCAL VARIABLES
C
      INTEGER I,J,K
	  %(complex_dp_format)s COEFS(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)

	  LOGICAL DUMMYFALSE
	  DATA DUMMYFALSE/.FALSE./
C  
C GLOBAL VARIABLES
C  
      include 'coupl.inc'
	  include 'mp_coupl.inc'

	  INTEGER HELOFFSET	  
	  INTEGER GOODHEL(NCOMB)
	  LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
	  common/%(proc_prefix)sFilters/GOODAMP,GOODHEL,HELOFFSET

	  LOGICAL CHECKPHASE
	  LOGICAL HELDOUBLECHECKED
      common/%(proc_prefix)sINIT/CHECKPHASE, HELDOUBLECHECKED

	  INTEGER SQSO_TARGET
	  common/%(proc_prefix)sSOCHOICE/SQSO_TARGET
      
	  LOGICAL UVCT_REQ_SO_DONE,MP_UVCT_REQ_SO_DONE,CT_REQ_SO_DONE,MP_CT_REQ_SO_DONE,LOOP_REQ_SO_DONE,MP_LOOP_REQ_SO_DONE,CTCALL_REQ_SO_DONE,FILTER_SO
	  common/%(proc_prefix)sSO_REQS/UVCT_REQ_SO_DONE,MP_UVCT_REQ_SO_DONE,CT_REQ_SO_DONE,MP_CT_REQ_SO_DONE,LOOP_REQ_SO_DONE,MP_LOOP_REQ_SO_DONE,CTCALL_REQ_SO_DONE,FILTER_SO

	  integer I_SO
	  common/%(proc_prefix)sI_SO/I_SO
	  integer I_LIB
	  common/%(proc_prefix)sI_LIB/I_LIB

## if(not LoopInduced) {
	  %(complex_dp_format)s AMP(NBORNAMPS)
	  common/%(proc_prefix)sAMPS/AMP
## }
	  %(complex_dp_format)s W(20,NWAVEFUNCS)
	  common/%(proc_prefix)sW/W

	  %(complex_dp_format)s WL(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE,0:NLOOPWAVEFUNCS)
	  %(complex_dp_format)s PL(0:3,0:NLOOPWAVEFUNCS)
	  common/%(proc_prefix)sWL/WL,PL

## if(not AmplitudeReduction){
      %(complex_dp_format)s AMPL(3,NCTAMPS)
## } else {
      %(complex_dp_format)s AMPL(3,NLOOPAMPS)
## } 
	  common/%(proc_prefix)sAMPL/AMPL

C
C ----------
C BEGIN CODE
C ----------

C     The target squared split order contribution is already reached if true.
      IF (FILTER_SO.AND.%(required_so_broadcaster)s) THEN
	    GOTO 1001
	  ENDIF

%(helas_calls)s

      GOTO 1001
%(continue_label)d CONTINUE
      %(required_so_broadcaster)s=.TRUE.
1001  CONTINUE
      END
