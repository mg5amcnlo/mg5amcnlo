      SUBROUTINE LOOP_%(nloopline)d%(nwfsargs_header)s( LID, %(pairingargs)s%(wfsargs)s%(margs)s RANK, RES, STABLE, LOOPNUM)

      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER    NLOOPLINE
      PARAMETER (NLOOPLINE=%(nloopline)d)
	  INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d)
      INTEGER    NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=%(nloop_groups)d)
	  INTEGER    NCOMB
      PARAMETER (NCOMB=%(ncomb)d)
C
C ARGUMENTS 
C
      INTEGER %(wfsargsdecl)s
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
      INTEGER PAIRING(NLOOPLINE),WE(%(nwfsargs)d)
      INTEGER I, J, K, TEMP
C
C GLOBAL VARIABLES
C
      INTEGER ID,R
      COMMON/LOOP/ID,R

	  LOGICAL CHECKPHASE, HELDOUBLECHECKED
      common/INIT/CHECKPHASE, HELDOUBLECHECKED

	  INTEGER HELOFFSET
	  INTEGER GOODHEL(NCOMB)
	  LOGICAL GOODAMP(NLOOPGROUPS)
	  common/Filters/GOODAMP,GOODHEL,HELOFFSET

	  %(complex_dp_format)s W(20,NWAVEFUNCS)
	  common/W/W  

C ----------
C BEGIN CODE
C ----------

	  IF (CHECKPHASE.OR.(.NOT.HELDOUBLECHECKED).OR.GOODAMP(LOOPNUM)) THEN
	  %(weset)s
      %(mset)s
      %(pairingset)s
	  R=RANK
      ID=LID
      DO I=0,3
        TEMP=1
        DO J=1,NLOOPLINE
		  PL(I,J)=0.D0
          DO K=TEMP,(TEMP+PAIRING(J)-1)
            PL(I,J)=PL(I,J)-DBLE(W(1+I,WE(K)))
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
