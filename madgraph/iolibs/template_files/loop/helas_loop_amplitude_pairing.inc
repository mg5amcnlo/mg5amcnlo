      SUBROUTINE LOOP_%(nloopline)d_%(nwfsargs)d_%(ncplsargs)d( LID, %(pairingargs)s%(wfsargs)s%(margs)s%(cplsargs)s RES)

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
      REAL*8 %(margsdecl)s
      COMPLEX*16 %(cplsdecl)s
      INTEGER %(pairingdecl)s
      COMPLEX*16 RES(3)
      INTEGER LID
C
C LOCAL VARIABLES 
C
      REAL*8 PL(0:3,NLOOPLINE)
      REAL*8 M2L(NLOOPLINE)
      INTEGER PAIRING(NLOOPLINE)
      INTEGER I, J, K, TEMP
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
      %(mset)s
	  %(mset2)s	  
      %(pairingset)s
      %(cplset)s	  
      ID=LID
      DO I=0,3
        TEMP=1
        DO J=1,NLOOPLINE
          DO K=TEMP,(TEMP+PAIRING(J)-1)
            PL(I,J)=DBLE(WE(5+I,K))
          ENDDO
          TEMP=TEMP+PAIRING(J)
        ENDDO
      ENDDO
      CALL CTLOOP(NLOOPLINE,PL,M2L,RES)

      END
