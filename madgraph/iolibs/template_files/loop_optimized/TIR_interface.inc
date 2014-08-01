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
      INTEGER LOOPMAXCOEFS
      PARAMETER (LOOPMAXCOEFS=%(loop_max_coefs)d)
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      LOGICAL checkPConservation
      PARAMETER (checkPConservation=.TRUE.)
      %(real_dp_format)s NORMALIZATION
      PARAMETER (NORMALIZATION = 1.D0/(16.D0*3.14159265358979323846D0**2))
C  
C ARGUMENTS 
C
      INTEGER I_SQSO,I_LOOPGROUP,I_LIB
      INTEGER NLOOPLINE, RANK
      %(real_dp_format)s PL(0:3,NLOOPLINE)
      %(real_dp_format)s PCT(0:3,0:NLOOPLINE-1)
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
	  LOGICAL CTINIT, TIRINIT, GOLEMINIT
	  COMMON/REDUCTIONCODEINIT/CTINIT,TIRINIT,GOLEMINIT

       %(complex_dp_format)s TIRCOEFS(0:LOOPMAXCOEFS-1,3)
       %(complex_dp_format)s PJCOEFS(0:LOOPMAXCOEFS-1,3)
C
C EXTERNAL FUNCTIONS
C
C
C GLOBAL VARIABLES
C
      include 'MadLoopParams.inc'
      include 'coupl.inc'
	  INTEGER CTMODE
	  %(real_dp_format)s LSCALE
      common/%(proc_prefix)sCT/LSCALE,CTMODE


       %(complex_dp_format)s LOOPCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)
       COMMON/%(proc_prefix)sLCOEFS/LOOPCOEFS

C ----------
C BEGIN CODE
C ----------

C INITIALIZE TIR IF NEEDED
      %(golem_calling)s
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
        do J=0,(NLOOPLINE-1)
          PCT(I,J)=0.D0
        enddo
      enddo
      do I=0,3
        do J=1,NLOOPLINE
          PCT(I,0)=PCT(I,0)+PL(I,J)
        enddo
      enddo
      if (checkPConservation) then
        if (PCT(0,0).gt.1.d-6) then
          write(*,*) 'energy is not conserved ',PCT(0,0)
          stop 'energy is not conserved'
        elseif (PCT(1,0).gt.1.d-6) then
         write(*,*) 'px is not conserved ',PCT(1,0)
         stop 'px is not conserved'
        elseif (PCT(2,0).gt.1.d-6) then
         write(*,*) 'py is not conserved ',PCT(2,0)
         stop 'py is not conserved'
        elseif (PCT(3,0).gt.1.d-6) then
         write(*,*) 'pz is not conserved ',PCT(3,0)
         stop 'pz is not conserved'
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
 
      SELECT CASE(MLReductionLib(I_LIB))
      CASE(2)
C     PJFry++
%(pjfry_calling)s
      CASE(3)
C     IREGI
%(iregi_calling)s
      END SELECT

      DO I=1,3
      	 RES(I)=LOOPCOEFS(0,I_SQSO,I_LOOPGROUP)*TIRCOEFS(0,I)
      	 DO J=1,NLOOPCOEFS-1
	    RES(I)=RES(I)+LOOPCOEFS(J,I_SQSO,I_LOOPGROUP)*TIRCOEFS(J,I)
	 ENDDO
      ENDDO
          %(finalize_TIR)s
C     WRITE(*,*) 'Loop ID',ID,' =',RES(1),RES(2),RES(3)
      END

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
	  LOGICAL CTINIT, TIRINIT, GOLEMINIT
	  COMMON/REDUCTIONCODEINIT/CTINIT,TIRINIT,GOLEMINIT
	  
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
%(initiregi)s
C      The initialization below is for CT v1.9.2+
       IF (CTINIT) THEN
	     CTINIT=.FALSE.
		 CALL %(proc_prefix)sINITCT()
	   ENDIF
	  END

	  SUBROUTINE %(proc_prefix)sCHOOSE_LOOPLIB(LIBINDEX,NLOOPLINE,RANK,complex_mass,doing_qp,I_LIB)
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
            INTEGER NLOOPLIB,QP_NLOOPLIB
            PARAMETER (NLOOPLIB=3,QP_NLOOPLIB=1)
C
C ARGUMENTS
C
            INTEGER LIBINDEX,NLOOPLINE,RANK,I_LIB
            LOGICAL complex_mass,doing_qp
C
C LOCAL VARIABLES
C
            INTEGER I,J_LIB,LIBNUM
	    LOGICAL LPASS
C
C GLOBAL VARIABLES
C
	    INCLUDE 'MadLoopParams.inc'
C TILL NOW, ONLY CUTTOOLS PROVIDE QP
            LOGICAL QP_TOOLS_AVAILABLE
            INTEGER INDEX_QP_TOOLS(QP_NLOOPLIB)
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
		DO I=1,QP_NLOOPLIB
		   IF(INDEX_QP_TOOLS(I).EQ.LIBINDEX)THEN
		     J_LIB=I
		     EXIT
		   ENDIF
		ENDDO
		IF(J_LIB.EQ.0)THEN
		   STOP 'CANNOT find the correct qp tool'
		ENDIF
		I=J_LIB
		I_LIB=LIBINDEX
		LIBNUM=MLReductionLib(I_LIB)
		DO
		  CALL DETECT_LOOPLIB(LIBNUM,NLOOPLINE,RANK,complex_mass,LPASS)
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
		  CALL DETECT_LOOPLIB(LIBNUM,NLOOPLINE,RANK,complex_mass,LPASS)
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
