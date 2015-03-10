      SUBROUTINE %(proc_prefix)sLOOP_%(nloopline)d%(nwfsargs_header)s_%(ncplsargs)d( LID, %(pairingargs)s%(wfsargs)s%(margs)s%(cplsargs)s RANK, LSYMFACT, LMULTIPLIER, AMPLN, RES, STABLE)

      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
	  INTEGER    MAXLCOUPLINGS
      PARAMETER (MAXLCOUPLINGS=%(maxlcouplings)d)
      INTEGER    NLOOPLINE
      PARAMETER (NLOOPLINE=%(nloopline)d)
	  INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d)
	  INTEGER    NCOMB
      PARAMETER (NCOMB=%(ncomb)d)
C
C ARGUMENTS 
C
      INTEGER %(wfsargsdecl)s
      %(mass_dp_format)s %(margsdecl)s
      %(mass_mp_format)s %(mp_margsdecl)s	  
      %(complex_dp_format)s %(cplsdecl)s
      %(complex_mp_format)s %(mp_cplsdecl)s	  
      %(pairingdecl)s
      %(complex_dp_format)s RES(3)
      INTEGER LID, RANK, LSYMFACT, LMULTIPLIER
	  INTEGER AMPLN
	  LOGICAL STABLE
C
C LOCAL VARIABLES 
C
      %(real_dp_format)s PL(0:3,NLOOPLINE)
      %(mass_dp_format)s M2L(NLOOPLINE)
      INTEGER PAIRING(NLOOPLINE)
      INTEGER I, J, K, TEMP
C
C GLOBAL VARIABLES
C
      INTEGER WE(NEXTERNAL)  
      INTEGER ID, SYMFACT, MULTIPLIER, AMPLNUM 
      common/%(proc_prefix)sLOOP/WE,ID, SYMFACT, MULTIPLIER,AMPLNUM

	  %(complex_dp_format)s LC(MAXLCOUPLINGS)	  
      %(mass_dp_format)s ML(NEXTERNAL+2)	
      common/%(proc_prefix)sDP_LOOP/LC,ML

	  %(complex_mp_format)s MP_LC(MAXLCOUPLINGS)
	  %(mass_mp_format)s MP_ML(NEXTERNAL+2)
      common/%(proc_prefix)sMP_LOOP/MP_LC,MP_ML

	  %(complex_dp_format)s W(20,NWAVEFUNCS%(ncomb_helas_objs)s)
	  INTEGER VALIDH
	  common/%(proc_prefix)sWFCTS/W  
	  common/%(proc_prefix)sVALIDH/VALIDH

C ----------
C BEGIN CODE
C ----------

      %(weset)s
      %(mset)s
	  %(mset2)s	  
      %(pairingset)s
      %(cplset)s
	  AMPLNUM=AMPLN
      ID=LID
	  SYMFACT=LSYMFACT
	  MULTIPLIER=LMULTIPLIER
      DO I=0,3
        TEMP=1
        DO J=1,NLOOPLINE
		  PL(I,J)=0.D0
          DO K=TEMP,(TEMP+PAIRING(J)-1)
            PL(I,J)=PL(I,J)-DBLE(W(1+I,WE(K)%(validh_or_nothing)s))
          ENDDO
          TEMP=TEMP+PAIRING(J)
        ENDDO
      ENDDO
      CALL %(proc_prefix)sCTLOOP(NLOOPLINE,PL,M2L,RANK,RES,STABLE)

      END
