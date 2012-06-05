      SUBROUTINE %(bunch_name)s_%(bunch_number)d(P,NHEL,H,IC)
C  
      IMPLICIT NONE
C  
C CONSTANTS
C  
	  %(nbornamps_decl)s
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
	  INTEGER    NLOOPS, NLOOPGROUPS, NCTAMPS
      PARAMETER (NLOOPS=%(nloops)d, NLOOPGROUPS=%(nloop_groups)d, NCTAMPS=%(nctamps)d)
      INTEGER    NWAVEFUNCS,NLOOPWAVEFUNCS
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d,NLOOPWAVEFUNCS=%(nloopwavefuncs)d)
	  INTEGER MAXLWFSIZE
	  PARAMETER (MAXLWFSIZE=%(max_lwf_size)d)
	  INTEGER LOOPMAXCOEFS, VERTEXMAXCOEFS
	  PARAMETER (LOOPMAXCOEFS=%(loop_max_coefs)d, VERTEXMAXCOEFS=%(vertex_max_coefs)d)
	  %(real_mp_format)s     MP__ZERO
      PARAMETER (MP__ZERO=0.0e0_16)
	  %(complex_mp_format)s     MP__IZERO
      PARAMETER (MP__IZERO=CMPLX(0.0e0_16,0.0e0_16,KIND=16))
C  
C ARGUMENTS
C
     %(real_mp_format)s P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
	  INTEGER H
C  
C GLOBAL VARIABLES
C  
      include 'mp_coupl_same_name.inc'

	  LOGICAL GOODHEL(NCOMB)
	  LOGICAL GOODAMP(NLOOPGROUPS)
	  common/Filters/GOODAMP,GOODHEL

      %(mp_born_amps_decl)s
	  %(complex_mp_format)s W(20,NWAVEFUNCS)
	  common/MP_W/W

	  %(complex_mp_format)s WL(MAXLWFSIZE,0:LOOP_MAXCOEFS-1,MAXLWFSIZE,0:NLOOPWAVEFUNCS)
	  %(complex_mp_format)s PL(0:3,0:NLOOPWAVEFUNCS)
	  common/MP_WL/WL,PL

	  %(complex_mp_format)s LOOPCOEFS(0:LOOPMAXCOEFS-1,NLOOPS)
	  common/MP_LCOEFS/LOOPCOEFS

      %(complex_dp_format)s AMPL(3,NCTAMPS)
	  common/MP_AMPL/AMPL

      %(complex_dp_format)s LOOPRES(3,NLOOPGROUPS)
	  LOGICAL S(NLOOPGROUPS)
	  common/LOOPRES/LOOPRES,S
C
C ----------
C BEGIN CODE
C ----------

      DO I=0,3
        PL(I,0)=MP__IZERO
      ENDDO
      DO I=1,MAXLWFSIZE
        DO J=0,LOOPMAXCOEFS-1
          DO K=1,MAXLWFSIZE
            IF(I.EQ.K.AND.J.EQ.0) THEN
              WL(I,J,K,0)=CMPLX(1.0e0_16,MP__ZERO,KIND=16)
            ELSE
              WL(I,J,K,0)=MP__IZERO
            ENDIF
          ENDDO
        ENDDO
      ENDDO
	  DO I=1,NLOOPS
  	    DO J=0,LOOPMAXCOEFS-1
    	  LOOPCOEFS(J,I)=MP__IZERO
        ENDDO
      ENDDO

%(helas_calls)s

      END
