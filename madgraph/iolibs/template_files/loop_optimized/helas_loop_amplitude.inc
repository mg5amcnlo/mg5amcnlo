      SUBROUTINE %(proc_prefix)sLOOP_%(nloopline)d%(nwfsargs_header)s(%(pairingargs)s%(wfsargs)s%(margs)s RANK, SQUAREDSOINDEX, LOOPNUM)
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER    NLOOPLINE
      PARAMETER (NLOOPLINE=%(nloopline)d)
## if(AmplitudeReduction){
      INTEGER    NLOOPAMPS
      PARAMETER (NLOOPAMPS=%(nloopamps)d)
      INTEGER    NCTAMPS
      PARAMETER (NCTAMPS=%(nctamps)d)
## }
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
## if(AmplitudeReduction){
C In this case of a loop-induced process, the SQUAREDSOINDEX argument is dummy
## }
	  INTEGER LOOPNUM, SQUAREDSOINDEX
C
C LOCAL VARIABLES 
C
      %(real_dp_format)s PL(0:3,NLOOPLINE)
      %(real_mp_format)s MP_PL(0:3,NLOOPLINE)	  
      %(mass_dp_format)s M2L(NLOOPLINE)
      INTEGER PAIRING(NLOOPLINE),WE(%(nwfsargs)d)
      INTEGER I, J, K, TEMP,I_LIB
      LOGICAL complex_mass,doing_qp
## if(AmplitudeReduction){
      LOGICAL AMP_CONTRIBUTES
## }
C
C GLOBAL VARIABLES
C
      INCLUDE 'MadLoopParams.inc'
## if(not AmplitudeReduction){
      INTEGER ID,SQSOINDEX,R
      common/%(proc_prefix)sLOOP/ID,SQSOINDEX,R
## }else{
      INTEGER ID,R
      common/%(proc_prefix)sLOOP/ID,R
## }

	  LOGICAL CHECKPHASE, HELDOUBLECHECKED
      common/%(proc_prefix)sINIT/CHECKPHASE, HELDOUBLECHECKED

	  INTEGER HELOFFSET
	  INTEGER GOODHEL(NCOMB)
	  LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
	  common/%(proc_prefix)sFilters/GOODAMP,GOODHEL,HELOFFSET

	  %(complex_dp_format)s LOOPRES(3,NSQUAREDSO,NLOOPGROUPS)
	  LOGICAL S(NSQUAREDSO,NLOOPGROUPS)
	  common/%(proc_prefix)sLOOPRES/LOOPRES,S

## if(AmplitudeReduction){
      %(complex_dp_format)s AMPL(3,NLOOPAMPS)
	  common/%(proc_prefix)sAMPL/AMPL
## }

	  %(complex_dp_format)s W(20,NWAVEFUNCS)
	  common/%(proc_prefix)sW/W
      %(complex_mp_format)s MP_W(20,NWAVEFUNCS)
      common/%(proc_prefix)sMP_W/MP_W

	  %(real_dp_format)s LSCALE 
	  INTEGER CTMODE
      common/%(proc_prefix)sCT/LSCALE,CTMODE
	  INTEGER LIBINDEX
	  common/%(proc_prefix)sI_LIB/LIBINDEX

C ----------
C BEGIN CODE
C ----------

C Determine it uses qp or not
      DOING_QP = (CTMODE.GE.4)

## if(not AmplitudeReduction){
	  IF (CHECKPHASE.OR.(.NOT.HELDOUBLECHECKED).OR.GOODAMP(SQUAREDSOINDEX,LOOPNUM)) THEN
## }else{
C For loop-induced processes, we must reduce this loop amplitude if it contributes to any squared split order contribution.
      AMP_CONTRIBUTES = .FALSE.
      DO I=1,NSQUAREDSO
        IF (GOODAMP(I,LOOPNUM)) AMP_CONTRIBUTES=.TRUE.
	  ENDDO
	  IF (CHECKPHASE.OR.(.NOT.HELDOUBLECHECKED).OR.AMP_CONTRIBUTES) THEN
## }
	  %(weset)s
      %(mset)s
      %(pairingset)s
	  R=RANK
      ID=LOOPNUM
## if(not AmplitudeReduction){
	  SQSOINDEX=SQUAREDSOINDEX
## }
      DO I=0,3
        TEMP=1
        DO J=1,NLOOPLINE
		  PL(I,J)=0.D0
		  IF (DOING_QP) THEN
		    MP_PL(I,J)=0.0e+0_16		    
		  ENDIF
          DO K=TEMP,(TEMP+PAIRING(J)-1)
            PL(I,J)=PL(I,J)-DBLE(W(1+I,WE(K)))
			IF (DOING_QP) THEN
		      MP_PL(I,J)=MP_PL(I,J)-REAL(MP_W(1+I,WE(K)),KIND=16)	    
		    ENDIF
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
C Choose the correct loop library
      CALL %(proc_prefix)sCHOOSE_LOOPLIB(LIBINDEX,NLOOPLINE,RANK,complex_mass,ID,doing_qp,I_LIB)
      IF(MLReductionLib(I_LIB).EQ.1)THEN
C CutTools is used
## if(not AmplitudeReduction){
	CALL %(proc_prefix)sCTLOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
## if(samurai_available){
	  ELSEIF (MLReductionLib(I_LIB).EQ.5) THEN
C Samurai is used
	CALL %(proc_prefix)sSAMURAI_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
##	}
## if(ninja_available){
	  ELSEIF (MLReductionLib(I_LIB).EQ.6) THEN
C Ninja is used
    IF (.NOT.DOING_QP) THEN
	  CALL %(proc_prefix)sNINJA_LOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
	ELSE
## if(ninja_supports_quad_prec){
	  CALL %(proc_prefix)sMP_NINJA_LOOP(NLOOPLINE,MP_PL,M2L,RANK,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
## } else {
      WRITE(*,*) 'ERROR: Ninja should not be called in quadruple precision since the installed version considered does not support it.'
	  STOP 9
## }
	ENDIF
##	}
      ELSE
C Tensor Integral Reduction is used 
	CALL %(proc_prefix)sTIRLOOP(SQUAREDSOINDEX,LOOPNUM,I_LIB,NLOOPLINE,PL,M2L,RANK,LOOPRES(1,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))
## }else{
	CALL %(proc_prefix)sCTLOOP(NLOOPLINE,PL,M2L,RANK,AMPL(1,NCTAMPS+LOOPNUM),S(1,LOOPNUM))
## if(samurai_available){
	  ELSEIF (MLReductionLib(I_LIB).EQ.5) THEN
	CALL %(proc_prefix)sSAMURAI_LOOP(NLOOPLINE,PL,M2L,RANK,AMPL(1,NCTAMPS+LOOPNUM),S(1,LOOPNUM))
##	}
## if(ninja_available){
	  ELSEIF (MLReductionLib(I_LIB).EQ.6) THEN
        IF (.NOT.DOING_QP) THEN
	      CALL %(proc_prefix)sNINJA_LOOP(NLOOPLINE,PL,M2L,RANK,AMPL(1,NCTAMPS+LOOPNUM),S(1,LOOPNUM))
	    ELSE
## if(ninja_supports_quad_prec){
	      CALL %(proc_prefix)sMP_NINJA_LOOP(NLOOPLINE,MP_PL,M2L,RANK,AMPL(1,NCTAMPS+LOOPNUM),S(1,LOOPNUM))
## } else {
          WRITE(*,*) 'ERROR: Ninja should not be called in quadruple precision since the installed version considered does not support it.'
	      STOP 9
## }
	    ENDIF

##	}
      ELSE
C Tensor Integral Reduction is used 
	CALL %(proc_prefix)sTIRLOOP(1,LOOPNUM,I_LIB,NLOOPLINE,PL,M2L,RANK,AMPL(1,NCTAMPS+LOOPNUM),S(1,LOOPNUM))
## }
      ENDIF
	  ELSE
## if(not AmplitudeReduction){
	    LOOPRES(1,SQUAREDSOINDEX,LOOPNUM)=(0.0d0,0.0d0)
	    LOOPRES(2,SQUAREDSOINDEX,LOOPNUM)=(0.0d0,0.0d0)
	    LOOPRES(3,SQUAREDSOINDEX,LOOPNUM)=(0.0d0,0.0d0)
		S(SQUAREDSOINDEX,LOOPNUM)=.TRUE.
## }else{
	    AMPL(1,NCTAMPS+LOOPNUM)=(0.0d0,0.0d0)
	    AMPL(2,NCTAMPS+LOOPNUM)=(0.0d0,0.0d0)
	    AMPL(3,NCTAMPS+LOOPNUM)=(0.0d0,0.0d0)
		S(1,LOOPNUM)=.TRUE.
## }
	  ENDIF
      END
