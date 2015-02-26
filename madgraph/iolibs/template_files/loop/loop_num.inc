C THE CORE SUBROUTINE CALLED BY CUTTOOLS WHICH CONTAINS THE HELAS CALLS BUILDING THE LOOP

      SUBROUTINE %(proc_prefix)sLOOPNUM(Q,RES)
C  
C CONSTANTS 
C 
	  INTEGER    NCOMB
      PARAMETER (NCOMB=%(ncomb)d)
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
	  %(nbornamps_decl)s
      INTEGER    NLOOPAMPS
      PARAMETER (NLOOPAMPS=%(nloopamps)d)
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
      INTEGER ID, SYMFACT, MULTIPLIER, AMPLNUM
      common/%(proc_prefix)sLOOP/WE,ID,SYMFACT,MULTIPLIER,AMPLNUM

	  LOGICAL GOODHEL(NCOMB)
	  LOGICAL GOODAMP(NLOOPAMPS,NCOMB)
	  common/%(proc_prefix)sFilters/GOODAMP,GOODHEL

	  INTEGER NTRY
	  LOGICAL CHECKPHASE,HELDOUBLECHECKED
	  %(real_dp_format)s REF
      common/%(proc_prefix)sINIT/NTRY,CHECKPHASE,HELDOUBLECHECKED,REF

	  INTEGER CF_D(NLOOPAMPS,%(color_matrix_size)s)
	  INTEGER CF_N(NLOOPAMPS,%(color_matrix_size)s)
	  common/%(proc_prefix)sCF/CF_D,CF_N

	  %(dp_born_amps_decl)s
	  %(complex_dp_format)s W(20,NWAVEFUNCS%(ncomb_helas_objs)s)
	  common/%(proc_prefix)sWFCTS/W  

	  INTEGER HELPICKED
	  common/%(proc_prefix)sHELCHOICE/HELPICKED

	  RES=(0.0d0,0.0d0)

  DO H=1,NCOMB
  IF (((HELPICKED.EQ.-1).OR.(HELPICKED.EQ.H)).AND.((CHECKPHASE.OR..NOT.HELDOUBLECHECKED).OR.(GOODHEL(H).AND.GOODAMP(AMPLNUM,H)))) THEN
    CALL %(proc_prefix)sLOOPNUMHEL(-Q,BUFF,H)
	%(dp_squaring)s
  ENDIF
  ENDDO
	  RES=(RES*MULTIPLIER)/SYMFACT

	  END

      SUBROUTINE %(proc_prefix)sLOOPNUMHEL(Q,RES,H)
C  
C CONSTANTS 
C  
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
	  INTEGER    MAXLCOUPLINGS
      PARAMETER (MAXLCOUPLINGS=%(maxlcouplings)d)
	  INTEGER    NMAXLOOPWFS
      PARAMETER (NMAXLOOPWFS=(NEXTERNAL+2))
	  %(real_dp_format)s     ZERO
	  PARAMETER (ZERO=0.D0)
	  INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d)
	  %(nbornamps_decl)s
      INTEGER    NLOOPAMPS
      PARAMETER (NLOOPAMPS=%(nloopamps)d)
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
      %(complex_dp_format)s BUFF(4)
      %(complex_dp_format)s WL(20,NMAXLOOPWFS)
      INTEGER I
C
C GLOBAL VARIABLES
C
	  %(complex_dp_format)s LC(MAXLCOUPLINGS)
	  %(mass_dp_format)s ML(NEXTERNAL+2)
      common/%(proc_prefix)sDP_LOOP/LC,ML

      INTEGER WE(NEXTERNAL)  
      INTEGER ID, SYMFACT,MULTIPLIER,AMPLNUM
      common/%(proc_prefix)sLOOP/WE,ID,SYMFACT,MULTIPLIER,AMPLNUM

	  %(dp_born_amps_decl)s	  
	  %(complex_dp_format)s W(20,NWAVEFUNCS%(ncomb_helas_objs)s)
	  common/%(proc_prefix)sWFCTS/W  

C ----------
C BEGIN CODE
C ----------
      RES=(0.D0,0.D0)
      %(loop_helas_calls)s
      END

      SUBROUTINE %(proc_prefix)sMPLOOPNUM(Q,RES)
	  
      INCLUDE 'cts_mprec.h'
	  IMPLICIT NONE	  
C  
C CONSTANTS 
C 
	  INTEGER    NCOMB
      PARAMETER (NCOMB=%(ncomb)d)
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
	  %(nbornamps_decl)s	  
      INTEGER    NLOOPAMPS
      PARAMETER (NLOOPAMPS=%(nloopamps)d)
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
	  common/%(proc_prefix)sMP_DONE/MP_DONE

	  %(real_mp_format)s MP_PS(0:3,NEXTERNAL),MP_P(0:3,NEXTERNAL)
	  common/%(proc_prefix)sMP_PSPOINT/MP_PS,MP_P

	  %(real_dp_format)s LSCALE
	  INTEGER CTMODE	  
      common/%(proc_prefix)sCT/LSCALE,CTMODE

      INTEGER WE(NEXTERNAL)  
      INTEGER ID, SYMFACT,MULTIPLIER,AMPLNUM
      common/%(proc_prefix)sLOOP/WE,ID,SYMFACT,MULTIPLIER,AMPLNUM

	  LOGICAL GOODHEL(NCOMB)
	  LOGICAL GOODAMP(NLOOPAMPS,NCOMB)
	  common/%(proc_prefix)sFilters/GOODAMP,GOODHEL

	  INTEGER NTRY
	  LOGICAL CHECKPHASE,HELDOUBLECHECKED
	  %(real_dp_format)s REF
      common/%(proc_prefix)sINIT/NTRY,CHECKPHASE,HELDOUBLECHECKED,REF

	  INTEGER CF_D(NLOOPAMPS,%(color_matrix_size)s)
	  INTEGER CF_N(NLOOPAMPS,%(color_matrix_size)s)
	  common/%(proc_prefix)sCF/CF_D,CF_N

	  %(mp_born_amps_decl)s	  
	  %(complex_mp_format)s W(20,NWAVEFUNCS%(ncomb_helas_objs)s)
	  common/%(proc_prefix)sMP_WFS/W	

	  INTEGER HELPICKED
	  common/%(proc_prefix)sHELCHOICE/HELPICKED
C ----------
C BEGIN CODE
C ----------
      DO I=0,3
        QPQ(I) = Q(I)
      ENDDO
	  QPRES=(0.0E0_16,0.0E0_16)	  

      IF(.NOT.MP_DONE.AND.CTMODE.EQ.0) THEN
C       This is just to compute the wfs in quad prec
	    CALL %(proc_prefix)sMP_BORN_AMPS_AND_WFS(MP_P)
	    MP_DONE=.TRUE.
	  ENDIF

  DO H=1,NCOMB
  IF (((HELPICKED.EQ.-1).OR.(HELPICKED.EQ.H)).AND.((CHECKPHASE.OR..NOT.HELDOUBLECHECKED).OR.(GOODHEL(H).AND.GOODAMP(AMPLNUM,H)))) THEN
    CALL %(proc_prefix)sMPLOOPNUMHEL(-QPQ,BUFF,H)
	%(mp_squaring)s
  ENDIF
  ENDDO
	  QPRES=(QPRES*MULTIPLIER)/SYMFACT

	  RES=QPRES
      END

      SUBROUTINE %(proc_prefix)sMPLOOPNUMHEL(Q,RES,H)
C  
C CONSTANTS 
C  
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
	  INTEGER    MAXLCOUPLINGS
      PARAMETER (MAXLCOUPLINGS=%(maxlcouplings)d)
	  INTEGER    NMAXLOOPWFS
      PARAMETER (NMAXLOOPWFS=(NEXTERNAL+2))
	  %(real_mp_format)s     ZERO
	  PARAMETER (ZERO=0E0_16)
	  INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d)
	  %(nbornamps_decl)s	  
      INTEGER    NLOOPAMPS
      PARAMETER (NLOOPAMPS=%(nloopamps)d)
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
      %(complex_mp_format)s BUFF(4)
      %(complex_mp_format)s WL(20,NMAXLOOPWFS)
      INTEGER I
C
C GLOBAL VARIABLES
C
	  %(complex_mp_format)s LC(MAXLCOUPLINGS)
	  %(mass_mp_format)s ML(NEXTERNAL+2)
      common/%(proc_prefix)sMP_LOOP/LC,ML

      INTEGER WE(NEXTERNAL)
      INTEGER ID, SYMFACT,MULTIPLIER,AMPLNUM
      common/%(proc_prefix)sLOOP/WE,ID,SYMFACT,MULTIPLIER,AMPLNUM

	  %(mp_born_amps_decl)s	  
	  %(complex_mp_format)s W(20,NWAVEFUNCS%(ncomb_helas_objs)s)
	  common/%(proc_prefix)sMP_WFS/W
C ----------
C BEGIN CODE
C ----------
      RES=(0E0_16,0E0_16)
      %(mp_loop_helas_calls)s
      END

      SUBROUTINE %(proc_prefix)sMPLOOPNUM_DUMMY(Q,RES)
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

      CALL %(proc_prefix)sLOOPNUM(DQ,DRES)
      RES=DRES

      END
