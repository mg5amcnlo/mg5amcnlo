      SUBROUTINE %(proc_prefix)sMP_COMPUTE_LOOP_COEFS(PS,ANSDP)
C  
%(info_lines)s
C
C Returns amplitude squared summed/avg over colors
c and helicities for the point in phase space P(0:3,NEXTERNAL)
c and external lines W(0:6,NEXTERNAL)
C 
%(process_lines)s
C  
      IMPLICIT NONE
C  
C CONSTANTS
C
      CHARACTER*64 paramFileName
      PARAMETER ( paramFileName='MadLoopParams.dat')
	  %(nbornamps_decl)s
      INTEGER    NLOOPS, NLOOPGROUPS, NCTAMPS
      PARAMETER (NLOOPS=%(nloops)d, NLOOPGROUPS=%(nloop_groups)d, NCTAMPS=%(nctamps)d) 
      INTEGER    NCOLORROWS
	  PARAMETER (NCOLORROWS=%(nloopamps)d)
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER    NWAVEFUNCS,NLOOPWAVEFUNCS
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d,NLOOPWAVEFUNCS=%(nloopwavefuncs)d)
	  INTEGER MAXLWFSIZE
	  PARAMETER (MAXLWFSIZE=%(max_lwf_size)d)
	  INTEGER LOOPMAXCOEFS, VERTEXMAXCOEFS
	  PARAMETER (LOOPMAXCOEFS=%(loop_max_coefs)d, VERTEXMAXCOEFS=%(vertex_max_coefs)d)
	  INTEGER    NCOMB
      PARAMETER (NCOMB=%(ncomb)d)
	  %(real_mp_format)s    ZERO
      PARAMETER (ZERO=0E0_16)
      %(complex_mp_format)s IMAG1
      PARAMETER (IMAG1=(0e0_16,1e0_16))
C     These are constants related to the split orders
      INTEGER    NSO, NSQUAREDSO, NAMPSO
	  PARAMETER (NSO=%(nSO)d, NSQUAREDSO=%(nSquaredSO)d, NAMPSO=%(nAmpSO)d)
C  
C ARGUMENTS 
C  
      %(real_mp_format)s PS(0:3,NEXTERNAL)
      %(real_dp_format)s ANSDP(3,0:NSQUAREDSO)
C  
C LOCAL VARIABLES 
C  
      INTEGER I,J,K,H,DUMMY,ITEMP
      %(real_mp_format)s TEMP2
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
      %(real_mp_format)s P(0:3,NEXTERNAL)
	  DATA IC/NEXTERNAL*1/
	  %(real_mp_format)s ANS(3,0:NSQUAREDSO)
	  %(complex_mp_format)s COEFS(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)
      %(complex_mp_format)s CFTOT
C
C FUNCTIONS
C
      INTEGER %(proc_prefix)sML5SOINDEX_FOR_BORN_AMP
	  INTEGER %(proc_prefix)sML5SOINDEX_FOR_LOOP_AMP
	  INTEGER %(proc_prefix)sML5SQSOINDEX
C  
C GLOBAL VARIABLES
C  
	  include 'mp_coupl_same_name.inc'

	  LOGICAL CHECKPHASE, HELDOUBLECHECKED
      common/%(proc_prefix)sINIT/CHECKPHASE, HELDOUBLECHECKED

	  INTEGER HELOFFSET
	  INTEGER GOODHEL(NCOMB)
	  LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
	  common/%(proc_prefix)sFilters/GOODAMP,GOODHEL,HELOFFSET

	  INTEGER HELPICKED
	  common/%(proc_prefix)sHELCHOICE/HELPICKED

	  INTEGER SQSO_TARGET
	  common/%(proc_prefix)sSOCHOICE/SQSO_TARGET
      
	  LOGICAL UVCT_REQ_SO_DONE,MP_UVCT_REQ_SO_DONE,CT_REQ_SO_DONE,MP_CT_REQ_SO_DONE,LOOP_REQ_SO_DONE,MP_LOOP_REQ_SO_DONE,CTCALL_REQ_SO_DONE,FILTER_SO
	  common/%(proc_prefix)sSO_REQS/UVCT_REQ_SO_DONE,MP_UVCT_REQ_SO_DONE,CT_REQ_SO_DONE,MP_CT_REQ_SO_DONE,LOOP_REQ_SO_DONE,MP_LOOP_REQ_SO_DONE,CTCALL_REQ_SO_DONE,FILTER_SO

	  %(mp_born_amps_decl)s	  
	  %(complex_mp_format)s W(20,NWAVEFUNCS)
	  common/%(proc_prefix)sMP_W/W  

	  %(complex_dp_format)s DPW(20,NWAVEFUNCS)
	  common/%(proc_prefix)sW/DPW

	  %(complex_mp_format)s WL(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE,0:NLOOPWAVEFUNCS)
	  %(complex_mp_format)s PL(0:3,0:NLOOPWAVEFUNCS)
	  common/%(proc_prefix)sMP_WL/WL,PL

	  %(complex_mp_format)s LOOPCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)
	  common/%(proc_prefix)sMP_LCOEFS/LOOPCOEFS

      %(complex_mp_format)s AMPL(3,NCTAMPS)
	  common/%(proc_prefix)sMP_AMPL/AMPL

	  INTEGER CF_D(NCOLORROWS,%(color_matrix_size)s)
	  INTEGER CF_N(NCOLORROWS,%(color_matrix_size)s)
	  common/%(proc_prefix)sCF/CF_D,CF_N

	  INTEGER HELC(NEXTERNAL,NCOMB)
	  common/%(proc_prefix)sHELCONFIGS/HELC

	  LOGICAL MP_DONE_ONCE
	  common/%(proc_prefix)sMP_DONE_ONCE/MP_DONE_ONCE

C ----------
C BEGIN CODE
C ----------

C To be on the safe side, we always update the MP params here.
C It can be redundant as this routine can be called a couple of times for the same PS point during the stability checks.
C But it is really not time consuming and I would rather be safe.
CALL MP_UPDATE_AS_PARAM()

MP_DONE_ONCE = .TRUE.

C AS A SAFETY MEASURE WE FIRST COPY HERE THE PS POINT
DO I=1,NEXTERNAL
  DO J=0,3
    P(J,I)=PS(J,I)
  ENDDO
ENDDO

DO I=0,3
  PL(I,0)=(ZERO,ZERO)
ENDDO
DO I=1,MAXLWFSIZE
  DO J=0,LOOPMAXCOEFS-1
    DO K=1,MAXLWFSIZE
      IF(I.EQ.K.AND.J.EQ.0) THEN
        WL(I,J,K,0)=(1.0e0_16,ZERO)
      ELSE
        WL(I,J,K,0)=(ZERO,ZERO)
      ENDIF
    ENDDO
  ENDDO
ENDDO

DO K=1, 3
  DO I=1,NCTAMPS
    AMPL(K,I)=(ZERO,ZERO)
  ENDDO
ENDDO

DO I=1,NLOOPGROUPS
  DO J=0,LOOPMAXCOEFS-1
    DO K=1,NSQUAREDSO  
      LOOPCOEFS(J,K,I)=(ZERO,ZERO)
	ENDDO
  ENDDO
ENDDO

DO K=1,3
  DO J=0,NSQUAREDSO
    ANSDP(K,J)=0.0d0
    ANS(K,J)=ZERO
  ENDDO
ENDDO

DO H=1,NCOMB
  IF ((HELPICKED.EQ.H).OR.((HELPICKED.EQ.-1).AND.(CHECKPHASE.OR.(.NOT.HELDOUBLECHECKED).OR.(GOODHEL(H).GT.-HELOFFSET.AND.GOODHEL(H).NE.0)))) THEN
  DO I=1,NEXTERNAL
    NHEL(I)=HELC(I,H)
  ENDDO

  MP_UVCT_REQ_SO_DONE=.FALSE.
  MP_CT_REQ_SO_DONE=.FALSE.
  MP_LOOP_REQ_SO_DONE=.FALSE.

  %(mp_born_ct_helas_calls)s
2000 CONTINUE
  MP_CT_REQ_SO_DONE=.TRUE.

  %(mp_uvct_helas_calls)s
3000 CONTINUE
  MP_UVCT_REQ_SO_DONE=.TRUE.

  IF (.NOT.CHECKPHASE.AND.HELDOUBLECHECKED.AND.HELPICKED.EQ.-1) THEN
    DUMMY=GOODHEL(H)
  ELSE
    DUMMY=1
  ENDIF
  DO I=1,%(nctamps_or_nloopamps)s
    DO J=1,%(nbornamps_or_nloopamps)s
	  CFTOT=CMPLX(CF_N(I,J)/REAL(ABS(CF_D(I,J)),KIND=16),0.0e0_16,KIND=16)
      IF(CF_D(I,J).LT.0) CFTOT=CFTOT*IMAG1
	  %(mp_squaring)s
    ENDDO
  ENDDO 

  %(mp_coef_construction)s  
  4000 CONTINUE
  MP_LOOP_REQ_SO_DONE=.TRUE.

  ENDIF
ENDDO

c Copy the qp wfs to the dp ones as they are used to setup the CT calls.
DO I=1,NWAVEFUNCS
  DO J=1,MAXLWFSIZE+4
    DPW(J,I)=W(J,I)
  ENDDO
ENDDO

DO I=1,3
  DO J=0,NSQUAREDSO
    ANSDP(I,J)=ANS(I,J)
  ENDDO
ENDDO

%(mp_coef_merging)s

END
