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
		CALL MP_IMPROVE_PS_POINT_PRECISION(QPP)
	    CALL MP_COMPUTE_LOOP_COEFS(QPP,DUMMY)
	    MP_DONE=.TRUE.
	  ENDIF

	  CALL MP_EVAL_POLY(LOOPCOEFS(0,ID),RANK,-QQ,QRES)

	  RES=QRES

	  END

	  SUBROUTINE MP_IMPROVE_PS_POINT_PRECISION(P)
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
	  %(real_mp_format)s BUFF,REF
	  %(real_dp_format)s MASSES(NEXTERNAL-1)	  
C  
C GLOBAL VARIABLES
C 
      include 'mp_coupl_same_name.inc'

	  %(masses_def)s

	  DO I=0,3
	    REF=ZERO
		BUFF=ZERO
		DO J=1,NEXTERNAL
		  REF=REF+ABS(P(I,J))
		ENDDO
        DO J=1,%(n_initial)d		
		  BUFF=BUFF-P(I,J)
		ENDDO
        DO J=%(n_initial)d+1,NEXTERNAL	
		  BUFF=BUFF+P(I,J)
		ENDDO
		IF ((BUFF/REF).GT.1.0E-9_16) THEN
		  STOP 'Four-momentum conservation is not accurate enough.'		  
		ENDIF
	  ENDDO

	  DO I=1,NEXTERNAL-1
	    REF=ABS(P(0,I))+ABS(P(1,I))+ABS(P(2,I))+ABS(P(3,I))
		IF ((SQRT(P(0,I)**2-P(1,I)**2-P(2,I)**2-P(3,I)**2-MASSES(I)**2)/REF).GT.1.0E-4_16) THEN
		  STOP 'Four-momentum conservation is not accurate enough.'
		ELSE
	      P(3,I)=SIGN(SQRT(P(0,I)**2-P(1,I)**2-P(2,I)**2-MASSES(I)**2),P(3,I))
		ENDIF
	  ENDDO

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
