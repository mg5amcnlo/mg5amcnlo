	  SUBROUTINE TIRLOOP(I_SQSO,I_LOOPGROUP,NLOOPLINE,PL,M2L,RANK,RES,STABLE)
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
      INTEGER I_SQSO,I_LOOPGROUP
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
	  LOGICAL INIT
	  DATA INIT /.TRUE./
	  SAVE INIT
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
      common/CT/LSCALE,CTMODE


       %(complex_dp_format)s LOOPCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)
       COMMON/LCOEFS/LOOPCOEFS

C ----------
C BEGIN CODE
C ----------

C INITIALIZE TIR IF NEEDED
      IF (INIT) THEN
	    INIT=.FALSE.
		CALL INITTIR()
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
 
      SELECT CASE(MLReductionLib(1))
      CASE(2)
C     PJFry++
	CALL PMLOOP(NLOOPLINE,RANK,PL,PDEN,M2L,MU_R,PJCOEFS,STABLE)
C       CONVERT TO MADLOOP CONVENTION
	CALL SORT_PJCOEFS(RANK,NLOOPCOEFS,PJCOEFS,TIRCOEFS)
      CASE(3)
C     IREGI
	CALL IMLOOP(CTMODE,IREGIMODE,NLOOPLINE,LOOPMAXCOEFS,RANK,PDEN,M2L,MU_R,TIRCOEFS,STABLE)
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

	  SUBROUTINE INITTIR()
C 
C INITIALISATION OF TIR
C  
C LOCAL VARIABLES 
C
	  %(real_dp_format)s THRS
C  
C GLOBAL VARIABLES 
C
	  include 'MadLoopParams.inc'
C ----------
C BEGIN CODE
C ----------

C DEFAULT PARAMETERS FOR TIR
C -------------------------------  
C THRS1 IS THE PRECISION LIMIT BELOW WHICH THE MP ROUTINES ACTIVATES
C USE THE SAME MADLOOP PARAMETER IN CUTTOOLS AND TIR
	  THRS=CTSTABTHRES
C LOOPLIB SET WHAT LIBRARY CT USES
C 1 -> LOOPTOOLS
C 2 -> AVH
C 3 -> QCDLOOP
      LOOPLIB=CTLOOPLIBRARY
      CALL INITIREGI(IREGIRECY,LOOPLIB,1d-6)
	  END

	  SUBROUTINE SORT_PJCOEFS(RANK,NLOOPCOEFS,PJCOEFS,TIRCOEFS)
C
C CONVERT THE OUTPUT OF PJFRY++ TO THAT OF MADLOOP 
C
C THE OUTPUT OF COEFS FROM PJFRY++ IS
C RANK=0: (,)
C RANK=1: (0,),(1,),(2,),(3,)
C RANK=2: (0,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)
C ...
C THE OLD OUTPUT OF COEFS FROM MADLOOP IS
C RANK=0: (,)
C RANK=1: (0,),(1,),(2,),(3,)
C RANK=2: (0,0),(0,1),(0,2),(0,3),(1,1),(2,1),(3,1),(2,2),(2,3),(3,3)
C ...
C
C
C CONSTANTS
C
            INTEGER LOOPMAXCOEFS
            PARAMETER (LOOPMAXCOEFS=%(loop_max_coefs)d)
C ARGUMENTS
            INTEGER RANK,NLOOPCOEFS
	    %(complex_dp_format)s PJCOEFS(0:LOOPMAXCOEFS-1,3)
	    %(complex_dp_format)s TIRCOEFS(0:LOOPMAXCOEFS-1,3)
C
C LOCAL VARIABLES
C
            INTEGER I
	    INTEGER POS(0:LOOPMAXCOEFS-1)
	    SAVE POS
            LOGICAL INIT
            DATA INIT /.TRUE./
            SAVE INIT
C ----------
C BEGIN CODE
C ----------
            
	    IF(INIT)THEN
		INIT=.FALSE.
C ASSIGN THE POSITION OF POS FOR SWAP
		CALL ASSIGN_PJPOS(POS)
	    ENDIF

	    DO I=0,NLOOPCOEFS-1
	       TIRCOEFS(POS(I),1:3)=PJCOEFS(I,1:3)
	    ENDDO

	  END
	  

	  SUBROUTINE ASSIGN_PJPOS(POS)
C
C ASSIGN THE POSITION OF POS FOR SWAP
C
C
C CONSTANTS
C
            INTEGER LOOPMAXCOEFS,MAXRANK
            PARAMETER (LOOPMAXCOEFS=%(loop_max_coefs)d)
	    PARAMETER (MAXRANK=%(maxrank)d)
C
C ARGUMENTS
C
            INTEGER POS(0:LOOPMAXCOEFS-1)
C
C EXTERNAL FUNCTIONS
C
            INTEGER QPOLYPOS
            EXTERNAL QPOLYPOS
C
C LOCAL VARIABLES
C
            INTEGER I,J,K,SHIFT,DN
	    INTEGER POSINDEX(MAXRANK),PJPOSINDEX(MAXRANK)
C ----------
C BEGIN CODE
C ----------
	    POS(0)=0
	    IF(MAXRANK.LE.0)RETURN
	    DO I=1,4
	       POS(I)=I
	    ENDDO
	    IF(MAXRANK.LE.1)RETURN
	    SHIFT=4
	    DO J=2,MAXRANK
	       DN=(J+3)*(J+2)*(J+1)/6
	       POSINDEX(1:MAXRANK)=0
	       PJPOSINDEX(1:MAXRANK)=0
	       DO I=1,DN
	       	  IF(I.GT.1)CALL NEXTINDEX(J,POSINDEX)
		  CALL CONVERT_PJPOSINDEX(J,POSINDEX,PJPOSINDEX)
	       	  K=DN-QPOLYPOS(J,PJPOSINDEX)+1+SHIFT
	       	  POS(K)=I+SHIFT
	       ENDDO	       
	       SHIFT=SHIFT+DN
	    ENDDO

          END

	  SUBROUTINE NEXTINDEX(RANK,POSINDEX)
C
C CALL FOR THE NEXT INDEX
C
C
C CONSTANTS
C
            INTEGER MAXRANK
            PARAMETER (MAXRANK=%(maxrank)d)
C
C ARGUMENTS
C
            INTEGER RANK
            INTEGER POSINDEX(MAXRANK)
C
C LOCAL VARIABLES
C
            INTEGER I
C ----------
C BEGIN CODE
C ----------
	    DO I=1,RANK
	      POSINDEX(I)=POSINDEX(I)+1
	      IF(POSINDEX(I).GT.3)THEN
	        POSINDEX(I)=0
		IF(I.EQ.RANK)THEN
		   RETURN
		ENDIF
	      ELSE
		IF(I.GT.1)THEN
		  POSINDEX(1:I-1)=POSINDEX(I)
		ENDIF
		RETURN
	      ENDIF
	    ENDDO

	  END

	  SUBROUTINE CONVERT_PJPOSINDEX(RANK,POSINDEX,PJPOSINDEX)
C
C CONVERT POSINDEX TO PJPOSINDEX
C
C
C CONSTANTS
C
            INTEGER MAXRANK
            PARAMETER (MAXRANK=%(maxrank)d)
C
C ARGUMENTS
C
            INTEGER RANK
            INTEGER POSINDEX(MAXRANK),PJPOSINDEX(MAXRANK)
C
C LOCAL VARIABLES
C
            INTEGER I
C ----------
C BEGIN CODE
C ----------
            DO I=1,RANK
	      PJPOSINDEX(RANK+1-I)=3-POSINDEX(I)
	    ENDDO
	    RETURN
	  END

	  FUNCTION QPOLYPOS(RANK,POSINDEX)
C
C COMPUTATION THE RELATIVE POSITION OF INDEX WITH RANK
C
C
C CONSTANTS
C
            INTEGER MAXRANK
            PARAMETER (MAXRANK=%(maxrank)d)
C
C ARGUMENTS
C
            INTEGER RANK
            INTEGER POSINDEX(MAXRANK)
	    INTEGER QPOLYPOS
C
C EXTERNAL FUNCTIONS
C
            INTEGER QPOLYNUMBER
	    EXTERNAL QPOLYNUMBER
C
C LOCAL VARIABLES
C
            INTEGER I,J,IMIN
C ----------
C BEGIN CODE
C ----------

            IF(RANK.EQ.0)THEN
	      QPOLYPOS=1
	      RETURN
	    ENDIF

	    IF(RANK.EQ.1)THEN
	      QPOLYPOS=POSINDEX(1)+1
	      RETURN
	    ENDIF

	    QPOLYPOS=POSINDEX(1)-POSINDEX(2)+1
	    DO I=2,RANK
	      IF(I.EQ.RANK)THEN
		IMIN=0
	      ELSE
	        IMIN=POSINDEX(I+1)
	      ENDIF
	      DO J=IMIN,POSINDEX(I)-1
	        QPOLYPOS=QPOLYPOS+QPOLYNUMBER(J,I-1)
	      ENDDO
	    ENDDO
	    RETURN
	  END

	  FUNCTION QPOLYNUMBER(I,RANK)
C
C THE INDEPENDENT NUMBER OF Q POLY WITH \MU=I,...,3 AND RANK
C
C
C CONSTANTS
C
C
C ARGUMENTS
C
            INTEGER I,RANK
	    INTEGER QPOLYNUMBER
C
C LOCAL VARIABLES
C
C ----------
C BEGIN CODE
C ----------
            SELECT CASE(I)
	    CASE(0)
	      QPOLYNUMBER=(3+RANK)*(2+RANK)*(1+RANK)/6
	    CASE(1)
	      QPOLYNUMBER=(2+RANK)*(1+RANK)/2
            CASE(2)
              QPOLYNUMBER=(1+RANK)
	    CASE(3)
	      QPOLYNUMBER=1
	    CASE DEFAULT
	      STOP 'I must be >= 0 and <=3 in QPOLYNUMBER.'
	    END SELECT
	    RETURN
          END