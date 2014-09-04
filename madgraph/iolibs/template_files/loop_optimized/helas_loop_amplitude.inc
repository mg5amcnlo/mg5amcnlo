      SUBROUTINE %(proc_prefix)sLOOP_%(nloopline)d%(nwfsargs_header)s(%(pairingargs)s%(wfsargs)s%(margs)s RANK, SQUAREDSOINDEX, LOOPNUM)

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
C     These are constants related to the split orders
      INTEGER    NSQUAREDSO
	  PARAMETER (NSQUAREDSO=%(nSquaredSO)d)
C
C ARGUMENTS 
C
      INTEGER %(wfsargsdecl)s
      %(mass_dp_format)s %(margsdecl)s
      %(pairingdecl)s
      INTEGER RANK, LSYMFACT
	  INTEGER SQUAREDSOINDEX, LOOPNUM
C
C LOCAL VARIABLES 
C
      %(real_dp_format)s PL(0:3,NLOOPLINE)
      %(mass_dp_format)s M2L(NLOOPLINE)
      INTEGER PAIRING(NLOOPLINE),WE(%(nwfsargs)d)
      INTEGER I, J, K, TEMP,I_LIB
      LOGICAL complex_mass,doing_qp
C
C GLOBAL VARIABLES
C
      INCLUDE 'MadLoopParams.inc'
      INTEGER ID,SQSOINDEX,R
      common/%(proc_prefix)sLOOP/ID,SQSOINDEX,R

	  LOGICAL CHECKPHASE, HELDOUBLECHECKED
      common/%(proc_prefix)sINIT/CHECKPHASE, HELDOUBLECHECKED

	  INTEGER HELOFFSET
	  INTEGER GOODHEL(NCOMB)
	  LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
	  common/%(proc_prefix)sFilters/GOODAMP,GOODHEL,HELOFFSET

	  %(complex_dp_format)s LOOPRES(3,NSQUAREDSO,NLOOPGROUPS)
	  LOGICAL S(NSQUAREDSO,NLOOPGROUPS)
	  common/%(proc_prefix)sLOOPRES/LOOPRES,S

	  %(complex_dp_format)s W(20,NWAVEFUNCS)
	  common/%(proc_prefix)sW/W 
	  %(real_dp_format)s LSCALE 
	  INTEGER CTMODE
          common/%(proc_prefix)sCT/LSCALE,CTMODE
	  INTEGER LIBINDEX
	  common/%(proc_prefix)sI_LIB/LIBINDEX

C ----------
C BEGIN CODE
C ----------

	  IF (CHECKPHASE.OR.(.NOT.HELDOUBLECHECKED).OR.GOODAMP(SQUAREDSOINDEX,LOOPNUM)) THEN
	  %(weset)s
      %(mset)s
      %(pairingset)s
	  R=RANK
      ID=LOOPNUM
	  SQSOINDEX=SQUAREDSOINDEX
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
C Determine whether the integral is with complex masses or not
C since some reduction libraries, e.g.PJFry++ and IREGI, are still
C not able to deal with complex masses
      complex_mass=.FALSE.
      DO I=1,NLOOPLINE
      	 IF(DIMAG(M2L(I)).EQ.0d0)CYCLE
      	 IF(ABS(DIMAG(M2L(I)))/MAX(ABS(M2L(I)),1D-2).GT.1d-15)THEN
	   complex_mass=.TRUE.
	   EXIT
	 ENDIF
      ENDDO
C Determine it uses qp or not
      doing_qp=.FALSE.
      IF(CTMODE.GE.4)doing_qp=.TRUE.
C Choose the correct loop library
      CALL %(proc_prefix)sCHOOSE_LOOPLIB(LIBINDEX,NLOOPLINE,RANK,complex_mass,doing_qp,I_LIB)
      IF(MLReductionLib(I_LIB).EQ.1)THEN
C USE CUTTOOLS
	CALL %(proc_prefix)sCTLOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
      ELSE
C USE TIR
	CALL %(proc_prefix)sTIRLOOP(SQUAREDSOINDEX,LOOPNUM,I_LIB,NLOOPLINE,PL,M2L,RANK,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM)) 
      ENDIF
	  ELSE
	    LOOPRES(1,SQUAREDSOINDEX,LOOPNUM)=0.0d0
	    LOOPRES(2,SQUAREDSOINDEX,LOOPNUM)=0.0d0
	    LOOPRES(3,SQUAREDSOINDEX,LOOPNUM)=0.0d0
		S(SQUAREDSOINDEX,LOOPNUM)=.TRUE.
	  ENDIF
      END
