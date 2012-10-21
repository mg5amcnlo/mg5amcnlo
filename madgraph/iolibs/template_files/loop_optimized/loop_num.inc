C THE CORE SUBROUTINE CALLED BY CUTTOOLS WHICH CONTAINS THE HELAS CALLS BUILDING THE LOOP

      SUBROUTINE LOOPNUM(Q,RES)
C  
C CONSTANTS 
C 
	  INTEGER    NLOOPS
      PARAMETER (NLOOPS=%(nloops)d)
	  INTEGER LOOPMAXCOEFS
	  PARAMETER (LOOPMAXCOEFS=%(loop_max_coefs)d)
C  
C ARGUMENTS 
C  
      %(complex_dp_format)s Q(0:3)
      %(complex_dp_format)s RES
C
C GLOBAL VARIABLES
C
      INTEGER ID,RANK
      COMMON/LOOP/ID,RANK

	  %(complex_dp_format)s LOOPCOEFS(0:LOOPMAXCOEFS-1,NLOOPS)
	  common/LCOEFS/LOOPCOEFS

	  RES=(0.0d0,0.0d0)

	  CALL EVAL_POLY(LOOPCOEFS(0,ID),RANK,-Q,RES)

	  END

      SUBROUTINE MPLOOPNUM(Q,RES)
C
C MODULE
C
      INCLUDE 'cts_mprec.h'
C  
C CONSTANTS 
C 
	  INTEGER    NLOOPS
      PARAMETER (NLOOPS=%(nloops)d)
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
	  INTEGER LOOPMAXCOEFS
	  PARAMETER (LOOPMAXCOEFS=%(loop_max_coefs)d)
C  
C ARGUMENTS 
C  
      INCLUDE 'cts_mpc.h'                                             , INTENT(IN), DIMENSION(0:3) :: Q
      INCLUDE 'cts_mpc.h'                                             , INTENT(OUT) :: RES
C  
C LOCAL VARIABLES 
C
      %(complex_mp_format)s QRES
	  %(real_dp_format)s DUMMY(3)
      %(real_mp_format)s QPP(0:3,NEXTERNAL)
      %(complex_mp_format)s QQ(0:3)
      INTEGER I,J
C
C GLOBAL VARIABLES
C
	  LOGICAL MP_DONE
	  common/MP_DONE/MP_DONE

      INTEGER ID,RANK
	  COMMON/LOOP/ID,RANK

	  %(complex_mp_format)s LOOPCOEFS(0:LOOPMAXCOEFS-1,NLOOPS)
	  common/MP_LCOEFS/LOOPCOEFS

C     MP_PS IS THE FIXED (POSSIBLY IMPROVED) MP PS POINT AND MP_P IS THE ONE WHICH CAN BE MODIFIED (I.E. ROTATED ETC.) FOR STABILITY PURPOSE
	  %(real_mp_format)s MP_PS(0:3,NEXTERNAL),MP_P(0:3,NEXTERNAL)
	  common/MP_PSPOINT/MP_PS,MP_P

C ----------
C BEGIN CODE
C ----------
      DO I=0,3
        QQ(I) = Q(I)
      ENDDO
	  QRES=(0.0e0_16,0.0e0_16)

      IF (.NOT.MP_DONE) THEN
	    CALL MP_COMPUTE_LOOP_COEFS(MP_P,DUMMY)
	    MP_DONE=.TRUE.
	  ENDIF

	  CALL MP_EVAL_POLY(LOOPCOEFS(0,ID),RANK,-QQ,QRES)

	  RES=QRES

	  END

      SUBROUTINE MPLOOPNUM_DUMMY(Q,RES)
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

      CALL LOOPNUM(DQ,DRES)
      RES=DRES

      END
