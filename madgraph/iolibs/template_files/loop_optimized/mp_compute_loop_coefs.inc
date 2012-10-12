      SUBROUTINE MP_COMPUTE_LOOP_COEFS(PS,ANSDP)
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
      %(real_dp_format)s PS(0:3,NEXTERNAL)
      %(real_dp_format)s ANSDP(3)
C  
C LOCAL VARIABLES 
C  
      INTEGER I,J,K,H,DUMMY
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
      %(real_mp_format)s P(0:3,NEXTERNAL)
	  DATA IC/NEXTERNAL*1/
	  %(real_mp_format)s ANS(3)
	  %(complex_mp_format)s COEFS(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)
      %(complex_mp_format)s CFTOT
C  
C GLOBAL VARIABLES
C  
	  include 'mp_coupl_same_name.inc'

	  LOGICAL CHECKPHASE, HELDOUBLECHECKED
      common/INIT/CHECKPHASE, HELDOUBLECHECKED

	  INTEGER HELOFFSET
	  INTEGER GOODHEL(NCOMB)
	  LOGICAL GOODAMP(NLOOPGROUPS)
	  common/Filters/GOODAMP,GOODHEL,HELOFFSET

	  INTEGER HELPICKED
	  common/HELCHOICE/HELPICKED

	  %(mp_born_amps_decl)s	  
	  %(complex_mp_format)s W(20,NWAVEFUNCS)
	  common/MP_W/W  

	  %(complex_dp_format)s DPW(20,NWAVEFUNCS)
	  common/W/DPW

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

CALL MP_UPDATE_AS_PARAM()

DO I=1,NEXTERNAL
  DO J=0,3
    P(J,I)=PS(J,I)
  ENDDO
ENDDO
CALL MP_IMPROVE_PS_POINT_PRECISION(P)

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

DO I=1,NLOOPS
  DO J=0,LOOPMAXCOEFS-1
    LOOPCOEFS(J,I)=(ZERO,ZERO)
  ENDDO
ENDDO

DO K=1,3
  ANSDP(K)=0.0d0
  ANS(K)=ZERO
ENDDO

DO H=1,NCOMB
  IF ((HELPICKED.EQ.H).OR.((HELPICKED.EQ.-1).AND.(CHECKPHASE.OR.(.NOT.HELDOUBLECHECKED).OR.(GOODHEL(H).GT.-HELOFFSET.AND.GOODHEL(H).NE.0)))) THEN
  DO I=1,NEXTERNAL
    NHEL(I)=HELC(I,H)
  ENDDO
  %(mp_born_ct_helas_calls)s
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
  ENDIF
ENDDO

c Copy the qp wfs to the dp ones as they are used to setup the CT calls.
DO I=1,NWAVEFUNCS
  DO J=1,MAXLWFSIZE+4
    DPW(J,I)=W(J,I)
  ENDDO
ENDDO

DO I=1,3
  ANSDP(I)=ANS(I)
ENDDO

%(mp_coef_merging)s

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
		IF ((BUFF/REF).GT.1.0E-6_16) THEN
		  WRITE(*,*) 'ERROR:: MP, Four-momentum conservation is not accurate enough, ',(BUFF/REF)
		  CALL MP_WRITE_MOM(P)
		  STOP
		ENDIF
	  ENDDO

	  DO I=1,NEXTERNAL-1
	    REF=ABS(P(0,I))+ABS(P(1,I))+ABS(P(2,I))+ABS(P(3,I))
		IF ((SQRT(ABS(P(0,I)**2-P(1,I)**2-P(2,I)**2-P(3,I)**2-MASSES(I)**2))/REF).GT.1.0E-2_16) THEN
		  WRITE(*,*) 'ERROR:: MP, Onshellness of the momentum of particle ',I,' of mass ',MASSES(I),' is not accurate enough, ', (SQRT(ABS(P(0,I)**2-P(1,I)**2-P(2,I)**2-P(3,I)**2-MASSES(I)**2))/REF)
		  CALL MP_WRITE_MOM(P)
		  STOP
		ELSE
	      P(3,I)=SIGN(SQRT(ABS(P(0,I)**2-P(1,I)**2-P(2,I)**2-MASSES(I)**2)),P(3,I))
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

	  %(real_mp_format)s function mp_dot(p1,p2)
C ****************************************************************************
C     4-Vector Dot product in multiple precision
C ****************************************************************************
      implicit none
      %(real_mp_format)s p1(0:3),p2(0:3)
      mp_dot=p1(0)*p2(0)-p1(1)*p2(1)-p1(2)*p2(2)-p1(3)*p2(3)
      return
      end

	  SUBROUTINE MP_WRITE_MOM(P)
	  IMPLICIT NONE
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      %(real_mp_format)s mp_dot

	  INTEGER I,J
C  
C ARGUMENTS 
C
      %(real_mp_format)s P(0:3,NEXTERNAL), PSUM(0:3)
	  DO I=0,3  
        PSUM(I)=0.0e0_16
        DO J=1,%(n_initial)d
          PSUM(I)=PSUM(I)+P(I,J)
        ENDDO
        DO J=%(n_initial)d+1,NEXTERNAL
          PSUM(I)=PSUM(I)-P(I,J)
        ENDDO
      ENDDO
      WRITE (*,*) ' Phase space point:'
      WRITE (*,*) '    ---------------------'
      WRITE (*,*) '    E | px | py | pz | m '
      DO I=1,NEXTERNAL
        WRITE (*,'(1x,5e25.15)') P(0,I),P(1,I),P(2,I),P(3,I),SQRT(ABS(MP_DOT(P(0,I),P(0,I))))
      ENDDO
      WRITE (*,*) '    Four-momentum conservation sum:'
      WRITE (*,'(1x,4e25.15)') PSUM(0),PSUM(1),PSUM(2),PSUM(3)
      WRITE (*,*) '   ---------------------'
	  END
