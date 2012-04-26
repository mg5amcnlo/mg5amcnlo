C THE CORE SUBROUTINE CALLED BY CUTTOOLS WHICH CONTAINS THE HELAS CALLS BUILDING THE LOOP

      SUBROUTINE LOOPNUM(Q,RES)
C  
C CONSTANTS 
C 
	  INTEGER    NCOMB
      PARAMETER (NCOMB=%(ncomb)d)
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER    NBORNAMPS, NLOOPAMPS
      PARAMETER (NBORNAMPS=%(nbornamps)d, NLOOPAMPS=%(nloopamps)d)
	  INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d)
	  INTEGER    MAXLCOUPLINGS
      PARAMETER (MAXLCOUPLINGS=%(maxlcouplings)d)
	  %(complex_dp_format)s IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
C  
C ARGUMENTS 
C  
      %(complex_dp_format)s Q(0:3)
      %(complex_dp_format)s RES
C  
C LOCAL VARIABLES 
C
      %(complex_dp_format)s CFTOT
      %(complex_dp_format)s BUFF
      INTEGER I,H
C
C GLOBAL VARIABLES
C
      INTEGER WE(NEXTERNAL)	  
      INTEGER ID, SYMFACT,AMPLNUM
      COMMON/LOOP/WE,ID,SYMFACT,AMPLNUM

	  LOGICAL GOODHEL(NCOMB)
	  LOGICAL GOODAMP(NLOOPAMPS,NCOMB)
	  common/Filters/GOODAMP,GOODHEL

	  INTEGER NTRY
	  LOGICAL CHECKPHASE
	  %(real_dp_format)s REF
      common/INIT/NTRY,CHECKPHASE,REF

	  INTEGER CF_D(NLOOPAMPS,NBORNAMPS)
	  INTEGER CF_N(NLOOPAMPS,NBORNAMPS)
	  common/CF/CF_D,CF_N

      %(complex_dp_format)s AMP(NBORNAMPS,NCOMB)
	  %(complex_dp_format)s W(20,NWAVEFUNCS,NCOMB)
	  INTEGER VALIDH
	  common/OPTIMIZATION/W,AMP,VALIDH

	  INTEGER HELPICKED
	  common/HELCHOICE/HELPICKED

	  RES=(0.0d0,0.0d0)

  DO H=1,NCOMB
  IF (((HELPICKED.EQ.-1).OR.(HELPICKED.EQ.H)).AND.(CHECKPHASE.OR.(GOODHEL(H).AND.GOODAMP(AMPLNUM,H)))) THEN
    CALL LOOPNUMHEL(Q,BUFF,H)
	DO I=1,NBORNAMPS
	  CFTOT=DCMPLX(CF_N(AMPLNUM,I)/DBLE(ABS(CF_D(AMPLNUM,I))),0.0d0)
	  IF(CF_D(AMPLNUM,I).LT.0) CFTOT=CFTOT*IMAG1
	  RES=RES+CFTOT*BUFF*DCONJG(AMP(I,H))
	ENDDO
  ENDIF
  ENDDO
	  RES=RES/SYMFACT

	  END

      SUBROUTINE LOOPNUMHEL(Q,RES,H)
C  
C CONSTANTS 
C  
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
	  INTEGER    MAXLCOUPLINGS
      PARAMETER (MAXLCOUPLINGS=%(maxlcouplings)d)
	  INTEGER    NMAXLOOPWFS
      PARAMETER (NMAXLOOPWFS=(NEXTERNAL+1))
	  %(real_dp_format)s     ZERO
	  PARAMETER (ZERO=0.D0)
	  INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d)
      INTEGER    NLOOPAMPS, NBORNAMPS
      PARAMETER (NLOOPAMPS=%(nloopamps)d, NBORNAMPS=%(nbornamps)d)
	  INTEGER    NCOMB
      PARAMETER (NCOMB=%(ncomb)d)
C  
C ARGUMENTS 
C  
      %(complex_dp_format)s Q(0:3)
      %(complex_dp_format)s RES
	  INTEGER H
C  
C LOCAL VARIABLES 
C
      %(complex_dp_format)s BUFF(12)
      %(complex_dp_format)s WL(20,NMAXLOOPWFS)
      INTEGER I
C
C GLOBAL VARIABLES
C
	  %(complex_dp_format)s LC(MAXLCOUPLINGS)
	  %(mass_dp_format)s ML(NEXTERNAL+1)
      COMMON/DP_LOOP/LC,ML

      INTEGER WE(NEXTERNAL)  
      INTEGER ID, SYMFACT,AMPLNUM
      COMMON/LOOP/WE,ID,SYMFACT,AMPLNUM

      %(complex_dp_format)s AMP(NBORNAMPS,NCOMB)
	  %(complex_dp_format)s W(20,NWAVEFUNCS,NCOMB)
	  INTEGER VALIDH
	  common/OPTIMIZATION/W,AMP,VALIDH

C ----------
C BEGIN CODE
C ----------
      RES=(0.D0,0.D0)
      %(loop_helas_calls)s
      END

      SUBROUTINE MPLOOPNUM(Q,RES)
	  
      INCLUDE 'cts_mprec.h'
	  IMPLICIT NONE	  
C  
C CONSTANTS 
C 
	  INTEGER    NCOMB
      PARAMETER (NCOMB=%(ncomb)d)
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER    NBORNAMPS, NLOOPAMPS
      PARAMETER (NBORNAMPS=%(nbornamps)d, NLOOPAMPS=%(nloopamps)d)
	  INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d)
	  INTEGER    MAXLCOUPLINGS
      PARAMETER (MAXLCOUPLINGS=%(maxlcouplings)d)
	  %(complex_mp_format)s IMAG1
      PARAMETER (IMAG1=(0E0_16,1E0_16))
C  
C ARGUMENTS 
C  
      INCLUDE 'cts_mpc.h'                                             , INTENT(IN), DIMENSION(0:3) :: Q
      INCLUDE 'cts_mpc.h'                                             , INTENT(OUT) :: RES
C  
C LOCAL VARIABLES 
C
      %(complex_mp_format)s QPRES
      %(complex_mp_format)s QPQ(0:3)
      %(real_mp_format)s QPP(0:3,NEXTERNAL)
      INTEGER I,J,H
	  %(complex_mp_format)s CFTOT
      %(complex_mp_format)s BUFF
C  
C GLOBAL VARIABLES
C 
	  LOGICAL MP_DONE
	  common/MP_DONE/MP_DONE

	  %(real_dp_format)s PS(0:3,NEXTERNAL)
	  common/PSPOINT/PS

      INTEGER WE(NEXTERNAL)  
      INTEGER ID, SYMFACT,AMPLNUM
      COMMON/LOOP/WE,ID,SYMFACT,AMPLNUM

	  LOGICAL GOODHEL(NCOMB)
	  LOGICAL GOODAMP(NLOOPAMPS,NCOMB)
	  common/Filters/GOODAMP,GOODHEL

	  INTEGER NTRY
	  LOGICAL CHECKPHASE
	  %(real_dp_format)s REF
      common/INIT/NTRY,CHECKPHASE,REF

	  INTEGER CF_D(NLOOPAMPS,NBORNAMPS)
	  INTEGER CF_N(NLOOPAMPS,NBORNAMPS)
	  common/CF/CF_D,CF_N

      %(complex_mp_format)s AMP(NBORNAMPS,NCOMB)
	  %(complex_mp_format)s W(20,NWAVEFUNCS,NCOMB)
	  common/MP_WFS/W,AMP	

	  INTEGER HELPICKED
	  common/HELCHOICE/HELPICKED
C ----------
C BEGIN CODE
C ----------
      DO I=0,3
        QPQ(I) = Q(I)
      ENDDO
	  QPRES=(0.0E0_16,0.0E0_16)	  

      IF(.NOT.MP_DONE) THEN
	    DO I=1,NEXTERNAL
		  DO J=0,3
		    QPP(J,I)=PS(J,I)
		  ENDDO
		ENDDO
		CALL IMPROVE_PS_POINT_PRECISION(QPP)
	    CALL MP_BORN_AMPS_AND_WFS(QPP)
	    MP_DONE=.TRUE.
	  ENDIF

  DO H=1,NCOMB
  IF (((HELPICKED.EQ.-1).OR.(HELPICKED.EQ.H)).AND.(CHECKPHASE.OR.(GOODHEL(H).AND.GOODAMP(AMPLNUM,H)))) THEN
    CALL MPLOOPNUMHEL(QPQ,BUFF,H)
	DO I=1,NBORNAMPS
	  CFTOT=CMPLX(CF_N(AMPLNUM,I)/(1.0E0_16*ABS(CF_D(AMPLNUM,I))),0.0E0_16,KIND=16)
	  IF(CF_D(AMPLNUM,I).LT.0) CFTOT=CFTOT*IMAG1
	  QPRES=QPRES+CFTOT*BUFF*CONJG(AMP(I,H))
	ENDDO
  ENDIF
  ENDDO
	  QPRES=QPRES/SYMFACT

	  RES=QPRES
      END

	  SUBROUTINE IMPROVE_PS_POINT_PRECISION(P)
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
	  %(real_mp_format)s BUFF
C  
C GLOBAL VARIABLES
C 
      include 'mp_coupl.inc'

      %(force_onshell)s

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

      SUBROUTINE MPLOOPNUMHEL(Q,RES,H)
C  
C CONSTANTS 
C  
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
	  INTEGER    MAXLCOUPLINGS
      PARAMETER (MAXLCOUPLINGS=%(maxlcouplings)d)
	  INTEGER    NMAXLOOPWFS
      PARAMETER (NMAXLOOPWFS=(NEXTERNAL+1))
	  %(real_mp_format)s     ZERO
	  PARAMETER (ZERO=0E0_16)
	  INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d)
      INTEGER    NLOOPAMPS, NBORNAMPS
      PARAMETER (NLOOPAMPS=%(nloopamps)d, NBORNAMPS=%(nbornamps)d)
	  INTEGER    NCOMB
      PARAMETER (NCOMB=%(ncomb)d)
C  
C ARGUMENTS 
C  
      %(complex_mp_format)s Q(0:3)
      %(complex_mp_format)s RES
	  INTEGER H
C  
C LOCAL VARIABLES 
C
      %(complex_mp_format)s BUFF(12)
      %(complex_mp_format)s WL(20,NMAXLOOPWFS)
      INTEGER I
C
C GLOBAL VARIABLES
C
	  %(complex_mp_format)s LC(MAXLCOUPLINGS)
	  %(mass_mp_format)s ML(NEXTERNAL+1)
      COMMON/MP_LOOP/LC,ML

      INTEGER WE(NEXTERNAL)
      INTEGER ID, SYMFACT,AMPLNUM
      COMMON/LOOP/WE,ID,SYMFACT,AMPLNUM

      %(complex_mp_format)s AMP(NBORNAMPS,NCOMB)
	  %(complex_mp_format)s W(20,NWAVEFUNCS,NCOMB)
	  common/MP_WFS/W,AMP
C ----------
C BEGIN CODE
C ----------
      RES=(0E0_16,0E0_16)
      %(mp_loop_helas_calls)s
      END
