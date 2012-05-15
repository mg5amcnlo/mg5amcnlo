      SUBROUTINE LOOP_%(nloopline)d%(nwfsargs_header)s( LID, %(pairingargs)s%(wfsargs)s%(margs)s RANK, LSYMFACT, LOOPNUM, RES, STABLE)

      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER    NLOOPLINE
      PARAMETER (NLOOPLINE=%(nloopline)d)
	  INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d)
C
C ARGUMENTS 
C
      INTEGER %(wfsargsdecl)s
	  INTEGER %(momposdecl)s
      %(mass_dp_format)s %(margsdecl)s
      %(pairingdecl)s
      %(complex_dp_format)s RES(3)
      INTEGER LID, RANK, LSYMFACT
	  INTEGER LOOPNUM
	  LOGICAL STABLE
C
C LOCAL VARIABLES 
C
      %(real_dp_format)s PL(0:3,NLOOPLINE)
      %(mass_dp_format)s M2L(NLOOPLINE)
      INTEGER PAIRING(NLOOPLINE)
	  INTEGER MOMPOS(%(nwfsargs)d)
      INTEGER I, J, K, TEMP
C
C GLOBAL VARIABLES
C
      INTEGER ID,R,SYMFACT
      COMMON/LOOP/ID,R,SYMFACT

	  LOGICAL GOODHEL(NCOMB)
	  LOGICAL GOODAMP(NLOOPS)
	  common/Filters/GOODAMP,GOODHEL

	  %(complex_dp_format)s W(20,NWAVEFUNCS)
	  common/W/W  

C ----------
C BEGIN CODE
C ----------

	  IF (CHECKPHASE.OR.GOODAMP(LOOPNUM)) THEN
	  %(momposset)s
      %(mset)s
      %(pairingset)s
	  R=RANK
      ID=LID
	  SYMFACT=LSYMFACT
      DO I=0,3
        TEMP=1
        DO J=1,NLOOPLINE
		  PL(I,J)=0.D0
          DO K=TEMP,(TEMP+PAIRING(J)-1)
            PL(I,J)=PL(I,J)+SIGN(1,MOMPOS(K))*DBLE(W(ABS(MOMPOS(K))+I,WE(K)%(validh_or_nothing)s))
          ENDDO
          TEMP=TEMP+PAIRING(J)
        ENDDO
      ENDDO
      CALL CTLOOP(NLOOPLINE,PL,M2L,RANK,RES,STABLE)
	  ELSE
	    RES(1)=0.0d0
	    RES(2)=0.0d0
	    RES(3)=0.0d0
		STABLE=.TRUE.
	  ENDIF
      END
