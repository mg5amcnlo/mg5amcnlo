C THE CORE SUBROUTINE CALLED BY CUTTOOLS WHICH CONTAINS THE HELAS CALLS BUILDING THE LOOP

      SUBROUTINE LOOPNUM(Q,RES)
C  
C CONSTANTS 
C 
	  INTEGER    NLOOPS
      PARAMETER (NLOOPS=%(nloops)d)
	  %(complex_dp_format)s IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
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
      INTEGER ID,RANK,SYMFAC
      COMMON/LOOP/ID,RANK,SYMFACT

	  %(complex_dp_format)s LOOPCOEFS(0:LOOPMAXCOEFS-1,NLOOPS)
	  common/LCOEFS/LOOPCOEFS

	  RES=(0.0d0,0.0d0)

	  CALL EVAL_POLY(LOOPCOEFS(ID),RANK,-Q,RES)

	  RES=RES/SYMFACT

	  END

      SUBROUTINE MPLOOPNUM_DUMMY(Q,RES)
C  
C ARGUMENTS 
C  
      INCLUDE 'cts_mprec.h'
      INCLUDE 'cts_mpc.h'                                             , INTENT(IN), DIMENSION(0:3) :: Q
      INCLUDE 'cts_mpc.h'                                             , INTENT(OUT) :: RES
C  
C LOCAL VARIABLES 
C
      COMPLEX*16 DRES
      COMPLEX*16 DQ(0:3)
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
