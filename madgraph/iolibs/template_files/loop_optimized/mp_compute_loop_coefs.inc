      SUBROUTINE MP_COMPUTE_LOOP_COEFS(P,ANS)
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
C  
C ARGUMENTS 
C  
      %(real_mp_format)s P(0:3,NEXTERNAL)
      %(real_dp_format)s ANS(3)
C  
C LOCAL VARIABLES 
C  
      INTEGER I,J,K,H
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
	  DATA IC/NEXTERNAL*1/
	  %(complex_mp_format)s COEFS(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)
      %(complex_mp_format)s CFTOT
C  
C GLOBAL VARIABLES
C  
	  include 'mp_coupl_same_name.inc'

	  LOGICAL CHECKPHASE
      common/INIT/CHECKPHASE

	  LOGICAL GOODHEL(NCOMB)
	  LOGICAL GOODAMP(NLOOPGROUPS)
	  common/Filters/GOODAMP,GOODHEL

	  INTEGER HELPICKED
	  common/HELCHOICE/HELPICKED

	  %(mp_born_amps_decl)s	  
	  %(complex_mp_format)s W(20,NWAVEFUNCS)
	  common/MP_W/W  

	  %(complex_mp_format)s WL(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE,0:NLOOPWAVEFUNCS)
	  %(complex_mp_format)s PL(0:3,0:NLOOPWAVEFUNCS)
	  common/MP_WL/WL,PL

	  %(complex_mp_format)s LOOPCOEFS(0:LOOPMAXCOEFS-1,NLOOPS)
	  common/MP_LCOEFS/LOOPCOEFS

      %(complex_mp_format)s AMPL(3,NCTAMPS)
	  common/MP_AMPL/AMPL

	  INTEGER CF_D(NCOLORROWS,%(color_matrix_size)s)
	  INTEGER CF_N(NCOLORROWS,%(color_matrix_size)s)
	  common/CF/CF_D,CF_N

	  INTEGER HELC(NEXTERNAL,NCOMB)
	  common/HELCONFIGS/HELC

C ----------
C BEGIN CODE
C ----------

DO K=1, 3
  DO I=1,NCTAMPS
    AMPL(K,I)=(ZERO,ZERO)
  ENDDO
ENDDO

DO I=1,NLOOPS
  DO J=1,LOOPMAXCOEFS
    LOOPCOEFS(J,I)=(ZERO,ZERO)
  ENDDO
ENDDO

DO K=1,3
  ANS(K)=0.0d0
ENDDO

DO H=1,NCOMB
  IF (((HELPICKED.EQ.-1).OR.(HELPICKED.EQ.H)).AND.(CHECKPHASE.OR.GOODHEL(H))) THEN
  DO I=1,NEXTERNAL
    NHEL(I)=HELC(I,H)
  ENDDO
  %(mp_born_ct_helas_calls)s
  DO I=1,%(nctamps_or_nloopamps)s
    DO J=1,%(nbornamps_or_nloopamps)s
	  CFTOT=DCMPLX(CF_N(I,J)/DBLE(ABS(CF_D(I,J))),0.0d0)
      IF(CF_D(I,J).LT.0) CFTOT=CFTOT*IMAG1
	  %(mp_squaring)s
    ENDDO
  ENDDO
  %(mp_coef_construction)s  
  ENDIF
ENDDO

END
