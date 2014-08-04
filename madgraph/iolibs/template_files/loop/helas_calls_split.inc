      SUBROUTINE %(proc_prefix)s%(bunch_name)s_%(bunch_number)d(P,NHEL,H,IC)
C  
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER    NLOOPAMPS
      PARAMETER (NLOOPAMPS=%(nloopamps)d)
	  %(nbornamps_decl)s
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d)
	  INTEGER    NCOMB
      PARAMETER (NCOMB=%(ncomb)d)
      %(real_dp_format)s     ZERO
      PARAMETER (ZERO=0D0)
	  %(real_mp_format)s     MP__ZERO
      PARAMETER (MP__ZERO=0.0e0_16)
      %(complex_dp_format)s IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
C  
C ARGUMENTS
C
      %(real_dp_format)s P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
	  INTEGER H
C  
C GLOBAL VARIABLES
C  
      include 'coupl.inc'
	  include 'mp_coupl.inc'
      %(dp_born_amps_decl)s
	  %(complex_dp_format)s W(20,NWAVEFUNCS%(ncomb_helas_objs)s)
	  common/%(proc_prefix)sWFCTS/W  

	  LOGICAL S(NLOOPAMPS)
      %(complex_dp_format)s AMPL(3,NLOOPAMPS)
	  common/%(proc_prefix)sAMPL/AMPL,S
C
C ----------
C BEGIN CODE
C ----------

%(helas_calls)s

      END
