C THE CORE SUBROUTINE CALLED BY CUTTOOLS WHICH CONTAINS THE HELAS CALLS BUILDING THE LOOP

      SUBROUTINE %(proc_prefix)sLOOPNUM(Q,RES)
C  
C CONSTANTS 
C 
      INTEGER    NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=%(nloop_groups)d)
      include 'loop_max_coefs.inc'
C     These are constants related to the split orders
      INTEGER    NSQUAREDSO
	  PARAMETER (NSQUAREDSO=%(nSquaredSO)d)
C  
C ARGUMENTS 
C  
      %(complex_dp_format)s Q(0:3)
      %(complex_dp_format)s RES
C
C GLOBAL VARIABLES
C
## if(not AmplitudeReduction) {
      INTEGER ID,SQSOINDEX,RANK
	  common/%(proc_prefix)sLOOP/ID,SQSOINDEX,RANK
## }else{
      INTEGER ID,RANK
	  common/%(proc_prefix)sLOOP/ID,RANK
## }

## if(not AmplitudeReduction) {
	  %(complex_dp_format)s LOOPCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)
## }else{
	  %(complex_dp_format)s LOOPCOEFS(0:LOOPMAXCOEFS-1,NLOOPGROUPS)
## }	  
	  common/%(proc_prefix)sLCOEFS/LOOPCOEFS

	  RES=(0.0d0,0.0d0)

## if(not AmplitudeReduction) {
	  CALL %(proc_prefix)sEVAL_POLY(LOOPCOEFS(0,SQSOINDEX,ID),RANK,-Q,RES)
## }else{
	  CALL %(proc_prefix)sEVAL_POLY(LOOPCOEFS(0,ID),RANK,-Q,RES)
## }
	  END

      SUBROUTINE %(proc_prefix)sMPLOOPNUM(Q,RES)
C
C MODULE
C
      INCLUDE 'cts_mprec.h'
C  
C CONSTANTS 
C 
      INTEGER    NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=%(nloop_groups)d)
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      include 'loop_max_coefs.inc'
C     These are constants related to the split orders
      INTEGER    NSQUAREDSO
	  PARAMETER (NSQUAREDSO=%(nSquaredSO)d)
C  
C ARGUMENTS 
C  
      INCLUDE 'cts_mpc.h'                                             , INTENT(IN), DIMENSION(0:3) :: Q
      INCLUDE 'cts_mpc.h'                                             , INTENT(OUT) :: RES
C  
C LOCAL VARIABLES 
C
      %(complex_mp_format)s QRES
	  %(real_dp_format)s DUMMY(3,0:NSQUAREDSO)
      %(real_mp_format)s QPP(0:3,NEXTERNAL)
      %(complex_mp_format)s QQ(0:3)
      INTEGER I,J
C
C GLOBAL VARIABLES
C
	  LOGICAL MP_DONE
	  common/%(proc_prefix)sMP_DONE/MP_DONE

## if(not AmplitudeReduction) {
      INTEGER ID,SQSOINDEX,RANK
	  common/%(proc_prefix)sLOOP/ID,SQSOINDEX,RANK
## }else{
      INTEGER ID,RANK
	  common/%(proc_prefix)sLOOP/ID,RANK
## }

## if(not AmplitudeReduction) {
	  %(complex_mp_format)s LOOPCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)
## }else{
	  %(complex_mp_format)s LOOPCOEFS(0:LOOPMAXCOEFS-1,NLOOPGROUPS)
## }
	  common/%(proc_prefix)sMP_LCOEFS/LOOPCOEFS

C     MP_PS IS THE FIXED (POSSIBLY IMPROVED) MP PS POINT AND MP_P IS THE ONE WHICH CAN BE MODIFIED (I.E. ROTATED ETC.) FOR STABILITY PURPOSE
	  %(real_mp_format)s MP_PS(0:3,NEXTERNAL),MP_P(0:3,NEXTERNAL)
	  common/%(proc_prefix)sMP_PSPOINT/MP_PS,MP_P

C ----------
C BEGIN CODE
C ----------
      DO I=0,3
        QQ(I) = Q(I)
      ENDDO
	  QRES=(0.0e0_16,0.0e0_16)

## if(not AmplitudeReduction) {
      IF (.NOT.MP_DONE) THEN
	    CALL %(proc_prefix)sMP_COMPUTE_LOOP_COEFS(MP_P,DUMMY)
	    MP_DONE=.TRUE.
	  ENDIF

	  CALL MP_%(proc_prefix)sEVAL_POLY(LOOPCOEFS(0,SQSOINDEX,ID),RANK,-QQ,QRES)
## }else{
	  CALL MP_%(proc_prefix)sEVAL_POLY(LOOPCOEFS(0,ID),RANK,-QQ,QRES)
## }
	  RES=QRES

	  END

      SUBROUTINE %(proc_prefix)sMPLOOPNUM_DUMMY(Q,RES)
C
C MODULE
C
      INCLUDE 'cts_mprec.h'
C  
C ARGUMENTS 
C  
      INCLUDE 'cts_mpc.h'                                             , INTENT(IN), DIMENSION(0:3) :: Q
      INCLUDE 'cts_mpc.h'                                             , INTENT(OUT) :: RES
C  
C LOCAL VARIABLES 
C
      %(complex_dp_format)s DRES
      %(complex_dp_format)s DQ(0:3)
      INTEGER I
C ----------
C BEGIN CODE
C ----------
      DO I=0,3
        DQ(I) = Q(I)
      ENDDO

      CALL %(proc_prefix)sLOOPNUM(DQ,DRES)
      RES=DRES

      END
