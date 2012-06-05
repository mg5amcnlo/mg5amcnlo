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

	  %(real_dp_format)s PS(0:3,NEXTERNAL)
	  common/PSPOINT/PS
C ----------
C BEGIN CODE
C ----------
      DO I=0,3
        QQ(I) = Q(I)
      ENDDO
	  QRES=(0.0e0_16,0.0e0_16)

      IF (.NOT.MP_DONE) THEN
	    DO I=1,NEXTERNAL
		  DO J=0,3
		    QPP(J,I)=PS(J,I)
		  ENDDO
		ENDDO
		CALL MP_UPDATE_AS_PARAM()
		CALL IMPROVE_PS_POINT_PRECISION(QPP)
	    CALL MP_COMPUTE_LOOP_COEFS(QPP,DUMMY)
	    MP_DONE=.TRUE.
	  ENDIF

	  CALL MP_EVAL_POLY(LOOPCOEFS(0,ID),RANK,-QQ,QRES)

	  RES=QRES

	  END

	  SUBROUTINE IMPROVE_PS_POINT_PRECISION(P)
C  
C CONSTANTS 
C 
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
	  %(real_mp_format)s     ZERO
	  PARAMETER (ZERO=0E0_16)
C  
C ARGUMENTS 
C
      %(real_mp_format)s P(0:3,NEXTERNAL)
C
C LOCAL VARIABLES 
C
      INTEGER I,J
	  %(real_mp_format)s BUFF
C  
C GLOBAL VARIABLES
C 
      include 'mp_coupl_same_name.inc'

      %(force_onshell)s

	  DO I=0,3
	    BUFF=0.0E0_16 
        DO J=1,%(n_initial)d
		  BUFF=BUFF+P(I,J)
		ENDDO
		DO J=%(n_initial)d+1,NEXTERNAL-1
		  BUFF=BUFF-P(I,J)		
		ENDDO
		P(I,NEXTERNAL)=BUFF
	  ENDDO

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
