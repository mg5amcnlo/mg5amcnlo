      SUBROUTINE LOOP( LID, %(wfsargs)s%(margs)s RES)
C  
C CONSTANTS 
C  
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER    NLOOPLINE
      PARAMETER (NLOOPLINE=%(nloopline)d)
C  
C ARGUMENTS 
C  
      REAL*8 %(wfsargsdecl)s
      REAL*8 %(margsdecl)s
      COMPLEX*16 RES(3)
	  INTEGER LID
C  
C LOCAL VARIABLES 
C
      REAL*8 PL(0:3,NLOOPLINE)
      REAL*8 M2L(NLOOPLINE)
	  INTEGER I, J
C
C GLOBAL VARIABLES
C
      COMPLEX*16 WE(20,NEXTERNAL)
	  INTEGER ID
	  COMMON/LOOP/WE,ID
C ----------
C BEGIN CODE
C ----------
	  DO I=1,20
        %(weset)s
	  ENDDO
      %(mset)s	  
	  ID=LID
	  DO I=0,3
	    DO J=NLOOPLINE
	      PL(I,J)=DBLE(WE(17+I,J)
		ENDDO
	  ENDDO
      CALL CTLOOP(NLOOPLINE,PL,M2L,RES)

	  END
