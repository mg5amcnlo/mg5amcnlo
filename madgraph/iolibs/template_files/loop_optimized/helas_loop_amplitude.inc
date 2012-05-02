      SUBROUTINE LOOP_%(nloopline)d_%(ncplsargs)d( LID, %(wfsargs)s%(margs)s%(cplsargs)s RANK, LSYMFACT, AMPLN, RES,STABLE)
C  
C CONSTANTS 
C  
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
	  INTEGER    MAXLCOUPLINGS
      PARAMETER (MAXLCOUPLINGS=%(maxlcouplings)d)
      INTEGER    NLOOPLINE
      PARAMETER (NLOOPLINE=%(nloopline)d)
      INTEGER    NBORNAMPS
      PARAMETER (NBORNAMPS=%(nbornamps)d)
	  INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d)
	  INTEGER    NCOMB
      PARAMETER (NCOMB=%(ncomb)d)
C  
C ARGUMENTS
C  
      INTEGER %(wfsargsdecl)s
	  INTEGER %(momposdecl)s	  
      %(mass_dp_format)s %(margsdecl)s
      %(mass_mp_format)s %(mp_margsdecl)s	  
      %(complex_dp_format)s %(cplsdecl)s
      %(complex_mp_format)s %(mp_cplsdecl)s
      %(complex_dp_format)s RES(3)
	  INTEGER LID, RANK, LSYMFACT
	  INTEGER AMPLN
	  LOGICAL STABLE
C  
C LOCAL VARIABLES 
C
      %(real_dp_format)s PL(0:3,NLOOPLINE)
      %(mass_dp_format)s M2L(NLOOPLINE)
	  INTEGER MOMPOS(%(nwfsargs)d)	  
	  INTEGER I, J
C
C GLOBAL VARIABLES
C
      INTEGER WE(NEXTERNAL)
	  INTEGER ID, SYMFACT, AMPLNUM
	  COMMON/LOOP/WE,ID,SYMFACT,AMPLNUM

	  %(complex_dp_format)s LC(MAXLCOUPLINGS)	  
      %(mass_dp_format)s ML(NEXTERNAL+2)
	  COMMON/DP_LOOP/LC,ML

	  %(complex_mp_format)s MP_LC(MAXLCOUPLINGS)
	  %(mass_mp_format)s MP_ML(NEXTERNAL+2)
      COMMON/MP_LOOP/MP_LC,MP_ML

      %(complex_dp_format)s AMP(NBORNAMPS,NCOMB)
	  %(complex_dp_format)s W(20,NWAVEFUNCS,NCOMB)
	  INTEGER VALIDH
	  common/OPTIMIZATION/W,AMP,VALIDH
C ----------
C BEGIN CODE
C ----------
	  
      %(weset)s
	  %(momposset)s	  
      %(mset)s
	  %(mset2)s
      %(cplset)s
	  AMPLNUM=AMPLN
	  ID=LID
	  SYMFACT=LSYMFACT	  
	  DO I=0,3
	    DO J=1,NLOOPLINE
	      PL(I,J)=SIGN(1,MOMPOS(J))*DBLE(W(ABS(MOMPOS(J))+I,WE(J),VALIDH))
		ENDDO
	  ENDDO
      CALL CTLOOP(NLOOPLINE,PL,M2L,RANK,RES,STABLE)

	  END
