      SUBROUTINE %(proc_prefix)s%(bunch_name)s_%(bunch_number)d(P,NHEL,H,IC)
C       
      use %(proc_prefix)sPOLYNOMIAL_CONSTANTS      
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
	  %(real_mp_format)s     ZERO
      PARAMETER (ZERO=0.0e0_16)
	  %(complex_mp_format)s     IZERO
      PARAMETER (IZERO=CMPLX(0.0e0_16,0.0e0_16,KIND=16))
C     These are constants related to the split orders
      INTEGER    NSO, NSQUAREDSO, NAMPSO
	  PARAMETER (NSO=%(nSO)d, NSQUAREDSO=%(nSquaredSO)d, NAMPSO=%(nAmpSO)d)
C  
C ARGUMENTS
C
     %(real_mp_format)s P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
	  INTEGER H
C
C LOCAL VARIABLES
C
	  INTEGER I,J,K
	  %(complex_mp_format)s COEFS(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)	  
C  
C GLOBAL VARIABLES
C  
      include 'mp_coupl_same_name.inc'

	  INTEGER GOODHEL(NCOMB)
	  LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
	  common/%(proc_prefix)sFilters/GOODAMP,GOODHEL

	  INTEGER SQSO_TARGET
	  common/%(proc_prefix)sSOCHOICE/SQSO_TARGET
      
	  LOGICAL UVCT_REQ_SO_DONE,MP_UVCT_REQ_SO_DONE,CT_REQ_SO_DONE,MP_CT_REQ_SO_DONE,LOOP_REQ_SO_DONE,MP_LOOP_REQ_SO_DONE,CTCALL_REQ_SO_DONE,FILTER_SO
	  common/%(proc_prefix)sSO_REQS/UVCT_REQ_SO_DONE,MP_UVCT_REQ_SO_DONE,CT_REQ_SO_DONE,MP_CT_REQ_SO_DONE,LOOP_REQ_SO_DONE,MP_LOOP_REQ_SO_DONE,CTCALL_REQ_SO_DONE,FILTER_SO

## if(not LoopInduced) {
	  %(complex_mp_format)s AMP(NBORNAMPS)
	  common/%(proc_prefix)sMP_AMPS/AMP
## }	  
	  %(complex_mp_format)s W(20,NWAVEFUNCS)
	  common/%(proc_prefix)sMP_W/W

	  %(complex_mp_format)s WL(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE,0:NLOOPWAVEFUNCS)
	  %(complex_mp_format)s PL(0:3,0:NLOOPWAVEFUNCS)
	  common/%(proc_prefix)sMP_WL/WL,PL

## if(not AmplitudeReduction){
      %(complex_mp_format)s AMPL(3,NCTAMPS)
## } else {
      %(complex_mp_format)s AMPL(3,NLOOPAMPS)
## } 
	  common/%(proc_prefix)sMP_AMPL/AMPL

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
