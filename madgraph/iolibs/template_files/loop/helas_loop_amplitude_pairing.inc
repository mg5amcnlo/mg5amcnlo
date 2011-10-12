      SUBROUTINE LOOP_%(nloopline)d_%(nwfsargs)d_%(ncplsargs)d( LID, %(pairingargs)s%(wfsargs)s%(margs)s%(cplsargs)s RANK, LSYMFACT, RES, CONTRIBUTE)

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
      INTEGER %(pairingdecl)s
      COMPLEX*16 RES(3)
      INTEGER LID, RANK, LSYMFACT
	  LOGICAL CONTRIBUTE
C
C LOCAL VARIABLES 
C
      REAL*8 PL(0:3,NLOOPLINE)
      REAL*8 M2L(NLOOPLINE)
      INTEGER PAIRING(NLOOPLINE)
	  INTEGER MOMPOS(%(nwfsargs)d)
      INTEGER I, J, K, TEMP
C
C GLOBAL VARIABLES
C
      COMPLEX*16 WE(20,NEXTERNAL)
	  COMPLEX*16 LC(MAXLCOUPLINGS)	  
      REAL*8 ML(NEXTERNAL+1)	  
      INTEGER ID, SYMFACT
      COMMON/LOOP/WE,LC,ML,ID, SYMFACT
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
      %(pairingset)s
      %(cplset)s
      ID=LID
	  SYMFACT=LSYMFACT
      DO I=0,3
        TEMP=1
        DO J=1,NLOOPLINE
		  PL(I,J)=0.D0
          DO K=TEMP,(TEMP+PAIRING(J)-1)
            PL(I,J)=PL(I,J)+SIGN(1,MOMPOS(K))*DBLE(WE(ABS(MOMPOS(K))+I,K))
          ENDDO
          TEMP=TEMP+PAIRING(J)
        ENDDO
      ENDDO
      CALL CTLOOP(NLOOPLINE,PL,M2L,RANK,RES)
	  ENDIF

      END
