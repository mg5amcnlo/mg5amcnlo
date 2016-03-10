	  SUBROUTINE %(proc_prefix)sTIRLOOP(I_SQSO,I_LOOPGROUP,I_LIB,NLOOPLINE,PL,M2L,RANK,RES,STABLE)
C  
%(info_lines)s
C 
C Interface between MG5 and TIR.
C  
%(process_lines)s
C  
C  
C CONSTANTS 
C 
      INTEGER NLOOPGROUPS
      PARAMETER (NLOOPGROUPS=%(nloop_groups)d)
C     These are constants related to the split orders
      INTEGER NSQUAREDSO
      PARAMETER (NSQUAREDSO=%(nSquaredSO)d)
      include 'loop_max_coefs.inc'
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      LOGICAL checkPConservation
      PARAMETER (checkPConservation=.TRUE.)
      %(real_dp_format)s NORMALIZATION
      PARAMETER (NORMALIZATION = 1.D0/(16.D0*3.14159265358979323846D0**2))
## if(TIRCaching){
      INTEGER TIR_CACHE_SIZE
      include 'tir_cache_size.inc'
## }
C  
C ARGUMENTS 
C
      INTEGER I_SQSO,I_LOOPGROUP,I_LIB
      INTEGER NLOOPLINE, RANK
      %(real_dp_format)s PL(0:3,NLOOPLINE)
      %(real_dp_format)s PCT(0:3,0:NLOOPLINE-1),ABSPCT(0:3)
      %(real_dp_format)s REF_P
      %(real_dp_format)s PDEN(0:3,NLOOPLINE-1)
      %(mass_dp_format)s M2L(NLOOPLINE)
      %(complex_dp_format)s M2LCT(0:NLOOPLINE-1)	  
      %(complex_dp_format)s RES(3)
	  LOGICAL STABLE	  
C  
C LOCAL VARIABLES 
C
	  INTEGER I, J, K
	  INTEGER NLOOPCOEFS
	  LOGICAL CTINIT, TIRINIT, GOLEMINIT, SAMURAIINIT, NINJAINIT
	  COMMON/REDUCTIONCODEINIT/CTINIT,TIRINIT,GOLEMINIT,SAMURAIINIT,NINJAINIT

c This variable will be used to detect changes in the TIR library used so as to force the reset of the TIR filter.
	  INTEGER LAST_LIB_USED
	  DATA LAST_LIB_USED/-1/

       %(complex_dp_format)s TIRCOEFS(0:LOOPMAXCOEFS-1,3)
       %(complex_dp_format)s PJCOEFS(0:LOOPMAXCOEFS-1,3)
## if(TIRCaching){
       integer TIR_CACHE_INDEX 
       %(complex_dp_format)s TIRCOEFS_CACHING(3,0:LOOPMAXCOEFS-1,NLOOPGROUPS,TIR_CACHE_SIZE)
       SAVE TIRCOEFS_CACHING
## }
C
C EXTERNAL FUNCTIONS
C
## if(TIRCaching){
	integer %(proc_prefix)sTIRCACHE_INDEX
## }
C
C GLOBAL VARIABLES
C
      include 'MadLoopParams.inc'
      include 'coupl.inc'
	  INTEGER CTMODE
	  %(real_dp_format)s LSCALE
      common/%(proc_prefix)sCT/LSCALE,CTMODE

C     The variables below are just for monitoring purposes. 
      INTEGER ID,SQSOINDEX,R
      COMMON/%(proc_prefix)sLOOP/ID,SQSOINDEX,R

c The argument ILIB is the TIR library to be used for that specific library.
      INTEGER LIBINDEX
      COMMON/%(proc_prefix)sI_LIB/LIBINDEX

## if(not AmplitudeReduction){
       %(complex_dp_format)s LOOPCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)
## }else{
       %(complex_dp_format)s LOOPCOEFS(0:LOOPMAXCOEFS-1,NLOOPGROUPS)
## }
       COMMON/%(proc_prefix)sLCOEFS/LOOPCOEFS
## if(TIRCaching){
C The index 0 of the second dimensions is filled with .false. as it makes the implementation simpler this way when the user sets TIR_CACHE_SIZE to be 0.
	   LOGICAL TIR_DONE(NLOOPGROUPS,0:TIR_CACHE_SIZE)
	   COMMON/%(proc_prefix)sTIRCACHING/TIR_DONE
## }
C ----------
C BEGIN CODE
C ----------

C Initialize for the very first time ML is called LAST_ILIB with the first ILIB used.
IF(LAST_LIB_USED.eq.-1) THEN
  LAST_LIB_USED = MLREDUCTIONLIB(LIBINDEX)
ELSE
C We changed the TIR library so we must refresh the cache.
  IF(MLREDUCTIONLIB(LIBINDEX).NE.LAST_LIB_USED) THEN
    LAST_LIB_USED = MLREDUCTIONLIB(LIBINDEX)
	CALL %(proc_prefix)sCLEAR_TIR_CACHE()
  ENDIF
ENDIF

IF (MLReductionLib(I_LIB).EQ.4) THEN
## if(golem_available) {
C Using Golem95
C PDEN is dummy for Golem95 so we just initialize it to zero here so as to use it for the function SWITCHORDER
  DO I=0,3
    DO J=1,NLOOPLINE-1
      PDEN(I,J)=0.0d0
    ENDDO
  ENDDO
  CALL %(proc_prefix)sSWITCH_ORDER(CTMODE,NLOOPLINE,PL,PDEN,M2L)
  CALL %(proc_prefix)sGOLEMLOOP(NLOOPLINE,PL,M2L,RANK,RES,STABLE)
  RETURN
## } else {
C Golem95 not available
  WRITE(*,*) 'ERROR:: Golem95 is not interfaced.'
  STOP
## }
ENDIF

C INITIALIZE TIR IF NEEDED
      IF (TIRINIT) THEN
	    TIRINIT=.FALSE.
		CALL %(proc_prefix)sINITTIR()
      ENDIF

C  CONVERT THE MASSES TO BE COMPLEX
      do I=1,NLOOPLINE
	    M2LCT(I-1)=M2L(I)
	  ENDDO

C  CONVERT THE MOMENTA FLOWING IN THE LOOP LINES TO CT CONVENTIONS
      do I=0,3
        ABSPCT(I) = 0.D0
        do J=0,(NLOOPLINE-1)
          PCT(I,J)=0.D0
        enddo
      enddo
      do I=0,3
        do J=1,NLOOPLINE
          PCT(I,0)=PCT(I,0)+PL(I,J)
          ABSPCT(I)=ABSPCT(I)+ABS(PL(I,J))
        enddo
      enddo
      ref_p = max(ABSPCT(0), ABSPCT(1),ABSPCT(2),ABSPCT(3))
       do I=0,3
           ABSPCT(I) = MAX(ref_p*1e-6, ABSPCT(I))
       enddo

      if (checkPConservation.and.ref_p.gt.1d-8) then
        if ((PCT(0,0)/ABSPCT(0)).gt.1.d-6) then
          write(*,*) 'energy is not conserved (flag: TIR)',PCT(0,0)
          stop 'energy is not conserved (flag: TIR)'
        elseif ((PCT(1,0)/ABSPCT(1)).gt.1.d-6) then
         write(*,*) 'px is not conserved (flag: TIR)',PCT(1,0)
         stop 'px is not conserved (flag: TIR)'
        elseif ((PCT(2,0)/ABSPCT(2)).gt.1.d-6) then
         write(*,*) 'py is not conserved (flag: TIR)',PCT(2,0)
         stop 'py is not conserved (flag: TIR)'
        elseif ((PCT(3,0)/ABSPCT(3)).gt.1.d-6) then
         write(*,*) 'pz is not conserved (flag: TIR)',PCT(3,0)
         stop 'pz is not conserved (flag: TIR)'
        endif
      endif
      do I=0,3
        do J=1,(NLOOPLINE-1)
          do K=1,J
            PCT(I,J)=PCT(I,J)+PL(I,K)
          enddo
        enddo
      enddo

      do I=0,3
      	 do J=1,(NLOOPLINE-1)
	    PDEN(I,J)=PCT(I,J)
	 enddo
      enddo
C NUMBER OF INDEPEDENT LOOPCOEFS FOR RANK=RANK
      NLOOPCOEFS=0
      DO I=0,RANK
         NLOOPCOEFS=NLOOPCOEFS+(3+I)*(2+I)*(1+I)/6
      ENDDO
## if(TIRCaching){
      if (TIR_CACHE_SIZE.eq.0) THEN
	    TIR_CACHE_INDEX = 0
	  ELSE
	    TIR_CACHE_INDEX = %(proc_prefix)sTIRCACHE_INDEX(CTMODE)
C This way whenever the CTMode exceeds the cache size it rolls back to the beginning and MadLoop will have taken care of having re-initialized TIR_DONE between labels 200 and 205 in loop_matrix.f
		TIR_CACHE_INDEX = MOD(TIR_CACHE_INDEX-1,TIR_CACHE_SIZE)+1 
	  ENDIF

	  IF(.NOT.TIR_DONE(I_LOOPGROUP,TIR_CACHE_INDEX))THEN
## } 
      SELECT CASE(MLReductionLib(I_LIB))
      CASE(2)
C     PJFry++
## if(pjfry_available){
      CALL %(proc_prefix)sSWITCH_ORDER(CTMODE,NLOOPLINE,PL,PDEN,M2L)
      CALL PMLOOP(NLOOPLINE,RANK,PL,PDEN,M2L,MU_R,PJCOEFS(0:NLOOPCOEFS-1,1:3),STABLE)
C     CONVERT TO MADLOOP CONVENTION
      CALL %(proc_prefix)sCONVERT_PJFRY_COEFFS(RANK,PJCOEFS,TIRCOEFS)
## } else {
      WRITE(*,*) 'ERROR:: PJFRY++ is not interfaced.'
      STOP
## }
      CASE(3)
C     IREGI
## if(iregi_available){
      CALL IMLOOP(CTMODE,IREGIMODE,NLOOPLINE,LOOPMAXCOEFS,RANK,PDEN,M2L,MU_R,PJCOEFS,STABLE)
C     CONVERT TO MADLOOP CONVENTION
      CALL %(proc_prefix)sCONVERT_IREGI_COEFFS(RANK,PJCOEFS,TIRCOEFS)  
## } else {
      WRITE(*,*) 'ERROR:: IREGI is not interfaced.'
      STOP
## }
      END SELECT
## if(TIRCaching){
C The zero index is dummy and means no caching at all since TIR_CACHE_SIZE=0.
	    IF(TIR_CACHE_INDEX.NE.0) THEN
	      DO I=1,3
	  	    DO J=0,NLOOPCOEFS-1
	         TIRCOEFS_CACHING(I,J,I_LOOPGROUP,TIR_CACHE_INDEX)=TIRCOEFS(J,I)
	        ENDDO
	      ENDDO
	      TIR_DONE(I_LOOPGROUP,TIR_CACHE_INDEX)=.TRUE.
	    ENDIF
	  ELSE
	    DO I=1,3
	      DO J=0,NLOOPCOEFS-1
	        TIRCOEFS(J,I)=TIRCOEFS_CACHING(I,J,I_LOOPGROUP,TIR_CACHE_INDEX)
	      ENDDO
	    ENDDO
	  ENDIF
## }
      DO I=1,3
	     RES(I)=(0.0d0,0.0d0)
      	 DO J=0,NLOOPCOEFS-1
## if(not AmplitudeReduction){
	       RES(I)=RES(I)+LOOPCOEFS(J,I_SQSO,I_LOOPGROUP)*TIRCOEFS(J,I)
## }else{
	       RES(I)=RES(I)+LOOPCOEFS(J,I_LOOPGROUP)*TIRCOEFS(J,I)
## }
	 ENDDO
      ENDDO
## if(AmplitudeReduction){
      RES(1)=NORMALIZATION*RES(1)
      RES(2)=NORMALIZATION*RES(2)
      RES(3)=NORMALIZATION*RES(3)
## } else {
      RES(1)=NORMALIZATION*2.0d0*DBLE(RES(1))
      RES(2)=NORMALIZATION*2.0d0*DBLE(RES(2))
      RES(3)=NORMALIZATION*2.0d0*DBLE(RES(3))
## }
C     IF(MLReductionLib(I_LIB).EQ.2) THEN
C       WRITE(*,*) 'PJFry: Loop ID',ID,' =',RES(1),RES(2),RES(3)
C     ELSEIF(MLReductionLib(I_LIB).EQ.3) THEN
C       WRITE(*,*) 'Iregi: Loop ID',ID,' =',RES(1),RES(2),RES(3)
C     ENDIF
      END

      SUBROUTINE %(proc_prefix)sSWITCH_ORDER(CTMODE,NLOOPLINE,PL,PDEN,M2L)
      IMPLICIT NONE

      INTEGER CTMODE,NLOOPLINE

      %(real_dp_format)s PL(0:3,NLOOPLINE)
      %(real_dp_format)s PDEN(0:3,NLOOPLINE-1)
      %(complex_dp_format)s M2L(NLOOPLINE)
      %(real_dp_format)s NEW_PL(0:3,NLOOPLINE)
      %(real_dp_format)s NEW_PDEN(0:3,NLOOPLINE-1)
      %(complex_dp_format)s NEW_M2L(NLOOPLINE)

      INTEGER I,J,K

      IF (CTMode.ne.2.and.CTMode.ne.5) THEN
        RETURN 
      ENDIF

      IF (NLOOPLINE.le.2) THEN
        RETURN
      ENDIF

      DO I=1,NLOOPLINE-1
        DO J=0,3
           NEW_PDEN(J,NLOOPLINE-I) = PDEN(J,I)
        ENDDO
      ENDDO
      DO I=1,NLOOPLINE-1
        DO J=0,3
           PDEN(J,I) = NEW_PDEN(J,I)
        ENDDO
      ENDDO

      DO I=2,NLOOPLINE
        NEW_M2L(I)=M2L(NLOOPLINE-I+2)
      ENDDO
      DO I=2,NLOOPLINE
        M2L(I)=NEW_M2L(I)
      ENDDO


      DO I=1,NLOOPLINE
        DO J=0,3
          NEW_PL(J,I) = -PL(J,NLOOPLINE+1-I)
        ENDDO
      ENDDO
      DO I=1,NLOOPLINE
        DO J=0,3
           PL(J,I) = NEW_PL(J,I)
        ENDDO
      ENDDO

      end

      SUBROUTINE %(proc_prefix)sMP_SWITCH_ORDER(CTMODE,NLOOPLINE,PL,PDEN,M2L)
      IMPLICIT NONE

      INTEGER CTMODE,NLOOPLINE

      %(real_mp_format)s PL(0:3,NLOOPLINE)
      %(real_mp_format)s PDEN(0:3,NLOOPLINE-1)
      %(complex_mp_format)s M2L(NLOOPLINE)
      %(real_mp_format)s NEW_PL(0:3,NLOOPLINE)
      %(real_mp_format)s NEW_PDEN(0:3,NLOOPLINE-1)
      %(complex_mp_format)s NEW_M2L(NLOOPLINE)

      INTEGER I,J,K

      IF (CTMode.ne.2.and.CTMode.ne.5) THEN
        RETURN 
      ENDIF

      IF (NLOOPLINE.le.2) THEN
        RETURN
      ENDIF

      DO I=1,NLOOPLINE-1
        DO J=0,3
           NEW_PDEN(J,NLOOPLINE-I) = PDEN(J,I)
        ENDDO
      ENDDO
      DO I=1,NLOOPLINE-1
        DO J=0,3
           PDEN(J,I) = NEW_PDEN(J,I)
        ENDDO
      ENDDO

      DO I=2,NLOOPLINE
        NEW_M2L(I)=M2L(NLOOPLINE-I+2)
      ENDDO
      DO I=2,NLOOPLINE
        M2L(I)=NEW_M2L(I)
      ENDDO


      DO I=1,NLOOPLINE
        DO J=0,3
          NEW_PL(J,I) = -PL(J,NLOOPLINE+1-I)
        ENDDO
      ENDDO
      DO I=1,NLOOPLINE
        DO J=0,3
           PL(J,I) = NEW_PL(J,I)
        ENDDO
      ENDDO

      end

	  SUBROUTINE %(proc_prefix)sINITTIR()
C 
C INITIALISATION OF TIR
C  
C LOCAL VARIABLES 
C
	  %(real_dp_format)s THRS
	  LOGICAL EXT_NUM_FOR_R1
C  
C GLOBAL VARIABLES 
C
	  include 'MadLoopParams.inc'
	  LOGICAL CTINIT, TIRINIT, GOLEMINIT, SAMURAIINIT, NINJAINIT
	  COMMON/REDUCTIONCODEINIT/CTINIT,TIRINIT,GOLEMINIT,SAMURAIINIT,NINJAINIT
	  
C ----------
C BEGIN CODE
C ----------

C DEFAULT PARAMETERS FOR TIR
C -------------------------------  
C THRS1 IS THE PRECISION LIMIT BELOW WHICH THE MP ROUTINES ACTIVATES
C USE THE SAME MADLOOP PARAMETER IN CUTTOOLS AND TIR
C IT IS NECESSARY TO INITIALIZE CT BECAUSE IREGI USES THE VERSION OF ONELOOP
C FROM CUTTOOLS LIBRARY
	  THRS=CTSTABTHRES
C LOOPLIB SET WHAT LIBRARY CT USES
C 1 -> LOOPTOOLS
C 2 -> AVH
C 3 -> QCDLOOP
      LOOPLIB=CTLOOPLIBRARY
## if (iregi_available) {
      CALL INITIREGI(IREGIRECY,LOOPLIB,1d-6)
## }
C      The initialization below is for CT v1.9.2+
       IF (CTINIT) THEN
	     CTINIT=.FALSE.
		 CALL %(proc_prefix)sINITCT()
	   ENDIF
	  END

	  SUBROUTINE %(proc_prefix)sCHOOSE_LOOPLIB(LIBINDEX,NLOOPLINE,RANK,complex_mass,loop_ID,doing_qp,I_LIB)
C
C CHOOSE THE CORRECT LOOP LIB
C Example:
C    MLReductionLib=3|2|1 and LIBINDEX=3
C    IF THE LOOP IS BEYOND THE SCOPE OF LOOP LIB MLReductionLib(3)=1
C    USE LIBINDEX=1, and LIBINDEX=2 ...
C    IF IT IS STILL NOT GOOD,STOP
C
     	   IMPLICIT NONE
C
C CONSTANTS
C
            INTEGER NLOOPLIB
			PARAMETER (NLOOPLIB=4)
			INTEGER QP_NLOOPLIB
## if(ninja_supports_quad_prec) {
      		PARAMETER (QP_NLOOPLIB=2)
## } else {
      		PARAMETER (QP_NLOOPLIB=1)
## }
            INTEGER NLOOPGROUPS
            PARAMETER (NLOOPGROUPS=%(nloop_groups)d)
C
C ARGUMENTS
C
            INTEGER LIBINDEX,NLOOPLINE,RANK,I_LIB,loop_ID
            LOGICAL complex_mass,doing_qp
C
C LOCAL VARIABLES
C
            INTEGER I,J_LIB,LIBNUM,SELECT_LIBINDEX
	        LOGICAL LPASS
C           This list specifies what loop involve an Higgs effective vertex so that CutTools limitations can be correctly implemented
            LOGICAL HAS_AN_HEFT_VERTEX(NLOOPGROUPS)
            %(has_HEFT_list)s
C
C GLOBAL VARIABLES
C
	        INCLUDE 'MadLoopParams.inc'
			INCLUDE 'process_info.inc'
C Change the list 'LOOPLIBS_QPAVAILABLE' in loop_matrix_standalone.inc to change the list of QPTools availables
            LOGICAL QP_TOOLS_AVAILABLE
            INTEGER INDEX_QP_TOOLS(QP_NLOOPLIB+1)
            common/%(proc_prefix)sLOOP_TOOLS/QP_TOOLS_AVAILABLE,INDEX_QP_TOOLS

C ----------
C BEGIN CODE
C ----------

	    IF(doing_qp)THEN
C  QP EVALUATION, ONLY CUTTOOLS
		IF(.NOT.QP_TOOLS_AVAILABLE)THEN
		   STOP 'No qp tools available, please make sure MLReductionLib is correct'
		ENDIF
		J_LIB=0
		SELECT_LIBINDEX=LIBINDEX
		DO WHILE(J_LIB.EQ.0)
		   DO I=1,QP_NLOOPLIB	
		      IF(INDEX_QP_TOOLS(I).EQ.SELECT_LIBINDEX)THEN
		        J_LIB=I
		        EXIT
		      ENDIF
		   ENDDO
		   IF(J_LIB.EQ.0)THEN
                      SELECT_LIBINDEX=SELECT_LIBINDEX+1		
                      IF(SELECT_LIBINDEX.GT.NLOOPLIB.OR.MLReductionLib(SELECT_LIBINDEX).EQ.0)SELECT_LIBINDEX=1
		   ENDIF
		ENDDO
		I=J_LIB
		I_LIB=SELECT_LIBINDEX
		LIBNUM=MLReductionLib(I_LIB)
		DO
		  CALL DETECT_LOOPLIB(LIBNUM,NLOOPLINE,RANK,complex_mass,HAS_AN_HEFT_VERTEX(loop_ID),MAX_SPIN_CONNECTED_TO_LOOP,LPASS)
		  IF(LPASS)EXIT
		  I=I+1
		  IF(I.GT.QP_NLOOPLIB.AND.INDEX_QP_TOOLS(I).EQ.0)THEN
		     I=1
		  ENDIF
		  IF(I.EQ.J_LIB)THEN
		    STOP 'No qp loop library can deal with this integral'
		  ENDIF
		  I_LIB=INDEX_QP_TOOLS(I)
		  LIBNUM=MLReductionLib(I_LIB)
		ENDDO
	    ELSE
C  DP EVALUATION
      		I_LIB=LIBINDEX
		LIBNUM=MLReductionLib(I_LIB)
		DO
		  CALL DETECT_LOOPLIB(LIBNUM,NLOOPLINE,RANK,complex_mass,HAS_AN_HEFT_VERTEX(loop_ID),MAX_SPIN_CONNECTED_TO_LOOP,LPASS)
		  IF(LPASS)EXIT
		  I_LIB=I_LIB+1
		  IF(I_LIB.GT.NLOOPLIB.OR.MLReductionLib(I_LIB).EQ.0)THEN
		     I_LIB=1
		  ENDIF
		  IF(I_LIB.EQ.LIBINDEX)THEN
		     STOP 'No dp loop library can deal with this integral'
		  ENDIF
		  LIBNUM=MLReductionLib(I_LIB)
		ENDDO 
	    ENDIF
	    RETURN
	  END

	  SUBROUTINE %(proc_prefix)sCLEAR_TIR_CACHE()
## if(TIRCaching){
	    IMPLICIT NONE

		INTEGER NLOOPGROUPS
        PARAMETER (NLOOPGROUPS=%(nloop_groups)d)
        INTEGER TIR_CACHE_SIZE
        include 'tir_cache_size.inc'

		INTEGER I,J

	    LOGICAL TIR_DONE(NLOOPGROUPS,0:TIR_CACHE_SIZE)
	    COMMON/%(proc_prefix)sTIRCACHING/TIR_DONE
        
		DO I=0,TIR_CACHE_SIZE
		  DO J=1,NLOOPGROUPS
            TIR_DONE(J,I)=.FALSE.
          ENDDO
		ENDDO
## } else {
C No TIR caching implemented, this is dummy. (The subroutine is kept as it might be called by the MC).
	    CONTINUE
## }
	  END SUBROUTINE

## if(TIRCaching){
	  integer function %(proc_prefix)sTIRCACHE_INDEX(CTMODE)
C Mapping function from CTMode to index and the TIR_cache
        INTEGER TIR_CACHE_SIZE
        include 'tir_cache_size.inc'

		INTEGER CTMODE

		%(proc_prefix)sTIRCACHE_INDEX=CTMODE
		IF(%(proc_prefix)sTIRCACHE_INDEX.gt.2) THEN
C This way, the CTMode 4 and 5 are sent to 3 and 4 respectively.
		  %(proc_prefix)sTIRCACHE_INDEX = %(proc_prefix)sTIRCACHE_INDEX-1
		ENDIF

C In principle if we wanted to cache other modes or rotations, we could do it here
	    RETURN 
      END
## }
