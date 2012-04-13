      SUBROUTINE LOOP_%(nloopline)d_%(ncplsargs)d( LID, %(wfsargs)s%(margs)s%(cplsargs)s RANK, LSYMFACT, RES,CONTRIBUTE)
C  
C CONSTANTS 
C  
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
	  INTEGER    MAXLCOUPLINGS
      PARAMETER (MAXLCOUPLINGS=%(maxlcouplings)d)
      INTEGER    NLOOPLINE
      PARAMETER (NLOOPLINE=%(nloopline)d)
C  
C ARGUMENTS
C  
      COMPLEX*16 %(wfsargsdecl)s
	  INTEGER %(momposdecl)s	  
      %(mass_format)s %(margsdecl)s
      COMPLEX*16 %(cplsdecl)s	  
      COMPLEX*16 RES(3)
	  INTEGER LID, RANK, LSYMFACT
	  LOGICAL CONTRIBUTE
C  
C LOCAL VARIABLES 
C
      REAL*8 PL(0:3,NLOOPLINE)
      %(mass_format)s M2L(NLOOPLINE)
	  INTEGER MOMPOS(%(nwfsargs)d)	  
	  INTEGER I, J
C
C GLOBAL VARIABLES
C
      COMPLEX*16 WE(20,NEXTERNAL)
	  COMPLEX*16 LC(MAXLCOUPLINGS)	  
      %(mass_format)s ML(NEXTERNAL+1)
	  INTEGER ID, SYMFACT
	  COMMON/LOOP/WE,LC,ML,ID,SYMFACT
C ----------
C BEGIN CODE
C ----------
      IF (.not.CONTRIBUTE) THEN
	    RES(1)=(0.D0,0.D0)
	    RES(2)=(0.D0,0.D0)
	    RES(3)=(0.D0,0.D0)
	  ELSE
	  DO I=1,20
        %(weset)s
	  ENDDO
	  %(momposset)s	  
      %(mset)s
	  %(mset2)s
      %(cplset)s	  
	  ID=LID
	  SYMFACT=LSYMFACT	  
	  DO I=0,3
	    DO J=1,NLOOPLINE
	      PL(I,J)=SIGN(1,MOMPOS(J))*DBLE(WE(ABS(MOMPOS(J))+I,J))
		ENDDO
	  ENDDO
      CALL CTLOOP(NLOOPLINE,PL,M2L,RANK,RES)
	  ENDIF

	  END
