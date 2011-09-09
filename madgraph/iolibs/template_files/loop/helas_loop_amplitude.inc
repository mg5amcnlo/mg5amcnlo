      SUBROUTINE LOOP_%(nloopline)d_%(ncplsargs)d( LID, %(wfsargs)s%(margs)s%(cplsargs)s RANK, RES)
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
      REAL*8 %(margsdecl)s
      COMPLEX*16 %(cplsdecl)s	  
      COMPLEX*16 RES(3)
	  INTEGER LID, RANK
C  
C LOCAL VARIABLES 
C
      REAL*8 PL(0:3,NLOOPLINE)
      REAL*8 M2L(NLOOPLINE)
	  INTEGER MOMPOS(%(nwfsargs)d)	  
	  INTEGER I, J
C
C GLOBAL VARIABLES
C
      COMPLEX*16 WE(20,NEXTERNAL)
	  COMPLEX*16 LC(MAXLCOUPLINGS)	  
      REAL*8 ML(NEXTERNAL+1)
	  INTEGER ID
	  COMMON/LOOP/WE,LC,ML,ID
C ----------
C BEGIN CODE
C ----------
	  DO I=1,20
        %(weset)s
	  ENDDO
	  %(momposset)s	  
      %(mset)s
	  %(mset2)s
      %(cplset)s	  
	  ID=LID
	  DO I=0,3
	    DO J=1,NLOOPLINE
	      PL(I,J)=SIGN(1,MOMPOS(J))*DBLE(WE(ABS(MOMPOS(J))+I,J))
		ENDDO
	  ENDDO
      CALL CTLOOP(NLOOPLINE,PL,M2L,RANK,RES)

	  END
