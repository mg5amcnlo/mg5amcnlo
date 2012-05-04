      SUBROUTINE MP_BORN_AMPS_AND_WFS(P)
C  
%(info_lines)s
C
C Computes all the AMP and WFS in quadruple precision for the 
C phase space point P(0:3,NEXTERNAL)
C  
%(process_lines)s
C  
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER    NBORNAMPS, NLOOPAMPS
      PARAMETER (NBORNAMPS=%(nbornamps)d, NLOOPAMPS=%(nloopamps)d) 
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d)
	  INTEGER    NCOMB
      PARAMETER (NCOMB=%(ncomb)d)
      %(real_mp_format)s     ZERO
      PARAMETER (ZERO=0E0_16)
	  %(complex_mp_format)s IMAG1
      PARAMETER (IMAG1=(0E0_16,1E0_16))

C  
C ARGUMENTS 
C  
      %(real_mp_format)s P(0:3,NEXTERNAL)
C  
C LOCAL VARIABLES 
C  
      INTEGER I,H
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
	  DATA IC/NEXTERNAL*1/
C  
C GLOBAL VARIABLES
C  
      include 'mp_coupl_same_name.inc'

	  INTEGER NTRY
	  LOGICAL CHECKPHASE
	  %(real_dp_format)s REF
      common/INIT/NTRY,CHECKPHASE,REF

	  LOGICAL GOODHEL(NCOMB)
	  LOGICAL GOODAMP(NLOOPAMPS,NCOMB)
	  common/Filters/GOODAMP,GOODHEL

	  INTEGER HELPICKED
	  common/HELCHOICE/HELPICKED

      %(complex_mp_format)s AMP(NBORNAMPS,NCOMB)
	  %(complex_mp_format)s W(20,NWAVEFUNCS,NCOMB)
	  common/MP_WFS/W,AMP

	  INTEGER HELC(NEXTERNAL,NCOMB)
	  common/HELCONFIGS/HELC

C ----------
C BEGIN CODE
C ---------

DO H=1,NCOMB
  IF (((HELPICKED.EQ.-1).OR.(HELPICKED.EQ.H)).AND.(CHECKPHASE.OR.GOODHEL(H))) THEN
  DO I=1,NEXTERNAL
    NHEL(I)=HELC(I,H)
  ENDDO
  %(born_amps_and_wfs_calls)s
  ENDIF
ENDDO

END
