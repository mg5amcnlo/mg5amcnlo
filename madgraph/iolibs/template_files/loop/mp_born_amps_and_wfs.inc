      SUBROUTINE %(proc_prefix)sMP_BORN_AMPS_AND_WFS(P)
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
      %(nbornamps_decl)s
	  INTEGER    NLOOPAMPS, NCTAMPS
      PARAMETER (NLOOPAMPS=%(nloopamps)d, NCTAMPS=%(nctamps)d)
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
      INTEGER I,J,H
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
	  DATA IC/NEXTERNAL*1/

C  
C GLOBAL VARIABLES
C  
      include 'mp_coupl_same_name.inc'

	  INTEGER NTRY
	  LOGICAL CHECKPHASE,HELDOUBLECHECKED
	  %(real_dp_format)s REF
      common/%(proc_prefix)sINIT/NTRY,CHECKPHASE,HELDOUBLECHECKED,REF

	  LOGICAL GOODHEL(NCOMB)
	  LOGICAL GOODAMP(NLOOPAMPS,NCOMB)
	  common/%(proc_prefix)sFilters/GOODAMP,GOODHEL

	  INTEGER HELPICKED
	  common/%(proc_prefix)sHELCHOICE/HELPICKED

	  %(mp_born_amps_decl)s
	  %(dp_born_amps_decl_in_mp)s
	  %(complex_mp_format)s W(20,NWAVEFUNCS%(ncomb_helas_objs)s)
	  common/%(proc_prefix)sMP_WFS/W

      %(complex_mp_format)s AMPL(3,NCTAMPS)
	  common/%(proc_prefix)sMP_AMPL/AMPL

	  %(complex_dp_format)s DPW(20,NWAVEFUNCS%(ncomb_helas_objs)s)
	  common/%(proc_prefix)sWFCTS/DPW

      %(complex_dp_format)s DPAMPL(3,NLOOPAMPS)
	  LOGICAL S(NLOOPAMPS)
	  common/%(proc_prefix)sAMPL/DPAMPL,S

	  INTEGER HELC(NEXTERNAL,NCOMB)
	  common/%(proc_prefix)sHELCONFIGS/HELC

	  LOGICAL MP_DONE_ONCE
	  common/%(proc_prefix)sMP_DONE_ONCE/MP_DONE_ONCE

C ----------
C BEGIN CODE
C ---------

MP_DONE_ONCE=.TRUE.

C To be on the safe side, we always update the MP params here.
C It can be redundant as this routine can be called a couple of times for the same PS point during the stability checks.
C But it is really not time consuming and I would rather be safe.
CALL MP_UPDATE_AS_PARAM()

DO H=1,NCOMB
  IF ((HELPICKED.EQ.H).OR.((HELPICKED.EQ.-1).AND.((CHECKPHASE.OR..NOT.HELDOUBLECHECKED).OR.GOODHEL(H)))) THEN
  DO I=1,NEXTERNAL
    NHEL(I)=HELC(I,H)
  ENDDO
  %(born_amps_and_wfs_calls)s
C  Copy the qp wfs to the dp ones as they are used to setup the CT calls.
  DO I=1,NWAVEFUNCS
    DO J=1,20
      DPW(J,I%(h_w_suffix)s)=W(J,I%(h_w_suffix)s)
    ENDDO
  ENDDO
C  Same for the counterterms amplitudes
  DO I=1,NCTAMPS
    DO J=1,3
	  DPAMPL(J,I)=AMPL(J,I)
	  S(I)=.TRUE.
	ENDDO
  ENDDO
  %(copy_mp_to_dp_born_amps)s
  ENDIF  
ENDDO

END
