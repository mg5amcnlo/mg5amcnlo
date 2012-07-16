      SUBROUTINE %(bunch_name)s_%(bunch_number)d(P,NHEL,H,IC)
C  
      IMPLICIT NONE
C  
C CONSTANTS
C  
	  %(nbornamps_decl)s
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
	  INTEGER    NCOMB
      PARAMETER (NCOMB=%(ncomb)d)
	  INTEGER    NLOOPS, NLOOPGROUPS, NCTAMPS
      PARAMETER (NLOOPS=%(nloops)d, NLOOPGROUPS=%(nloop_groups)d, NCTAMPS=%(nctamps)d)
      INTEGER    NWAVEFUNCS,NLOOPWAVEFUNCS
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d,NLOOPWAVEFUNCS=%(nloopwavefuncs)d)
	  INTEGER MAXLWFSIZE
	  PARAMETER (MAXLWFSIZE=%(max_lwf_size)d)
	  INTEGER LOOPMAXCOEFS, VERTEXMAXCOEFS
	  PARAMETER (LOOPMAXCOEFS=%(loop_max_coefs)d, VERTEXMAXCOEFS=%(vertex_max_coefs)d)
      %(real_dp_format)s     ZERO
      PARAMETER (ZERO=0D0)
	  %(real_mp_format)s     MP__ZERO
      PARAMETER (MP__ZERO=0.0e0_16)
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
C  
C GLOBAL VARIABLES
C  
      include 'coupl.inc'
	  include 'mp_coupl.inc'

	  INTEGER HELOFFSET	  
	  INTEGER GOODHEL(NCOMB)
	  LOGICAL GOODAMP(NLOOPGROUPS)
	  common/Filters/GOODAMP,GOODHEL,HELOFFSET

      %(dp_born_amps_decl)s
	  %(complex_dp_format)s W(20,NWAVEFUNCS)
	  common/W/W

	  %(complex_dp_format)s WL(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE,0:NLOOPWAVEFUNCS)
	  %(complex_dp_format)s PL(0:3,0:NLOOPWAVEFUNCS)
	  common/WL/WL,PL

	  %(complex_dp_format)s LOOPCOEFS(0:LOOPMAXCOEFS-1,NLOOPS)
	  common/LCOEFS/LOOPCOEFS

      %(complex_dp_format)s AMPL(3,NCTAMPS)
	  common/AMPL/AMPL

      %(complex_dp_format)s LOOPRES(3,NLOOPGROUPS)
	  LOGICAL S(NLOOPGROUPS)
	  common/LOOPRES/LOOPRES,S
C
C ----------
C BEGIN CODE
C ----------

%(helas_calls)s

      END
