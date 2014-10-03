	  SUBROUTINE %(proc_prefix)sCTLOOP(NLOOPLINE,PL,M2L,RANK,RES,STABLE)
C  
%(info_lines)s
C 
C Interface between MG5 and CutTools.
C  
%(process_lines)s
C  
C  
C CONSTANTS 
C  
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      LOGICAL checkPConservation
      PARAMETER (checkPConservation=.TRUE.)
      %(real_dp_format)s NORMALIZATION
      PARAMETER (NORMALIZATION = 1.D0/(16.D0*3.14159265358979323846D0**2))
C  
C ARGUMENTS 
C
      INTEGER NLOOPLINE, RANK
      %(real_dp_format)s PL(0:3,NLOOPLINE)
      %(real_dp_format)s PCT(0:3,0:NLOOPLINE-1)
      %(mass_dp_format)s M2L(NLOOPLINE)
      %(complex_dp_format)s M2LCT(0:NLOOPLINE-1)	  
      %(complex_dp_format)s RES(3)
	  LOGICAL STABLE	  
C  
C LOCAL VARIABLES 
C
	  %(complex_dp_format)s R1, ACC
	  INTEGER I, J, K
	  LOGICAL CTINIT, TIRINIT, GOLEMINIT
	  COMMON/REDUCTIONCODEINIT/CTINIT,TIRINIT,GOLEMINIT
C
C EXTERNAL FUNCTIONS
C
      EXTERNAL %(proc_prefix)sLOOPNUM
      EXTERNAL %(proc_prefix)sMPLOOPNUM
C
C GLOBAL VARIABLES
C
      include 'coupl.inc'
	  INTEGER CTMODE
	  %(real_dp_format)s LSCALE
      common/%(proc_prefix)sCT/LSCALE,CTMODE

      INTEGER ID,SQSOINDEX,R
      COMMON/%(proc_prefix)sLOOP/ID,SQSOINDEX,R

C ----------
C BEGIN CODE
C ----------

C INITIALIZE CUTTOOLS IF NEEDED
      IF (CTINIT) THEN
	    CTINIT=.FALSE.
		CALL %(proc_prefix)sINITCT()
      ENDIF

C YOU CAN FIND THE DETAILS ABOUT THE DIFFERENT CTMODE AT THE BEGINNING OF THE FILE CTS_CUTS.F90 IN THE CUTTOOLS DISTRIBUTION

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

      CALL CTSXCUT(CTMODE,LSCALE,MU_R,NLOOPLINE,%(proc_prefix)sLOOPNUM,%(proc_prefix)sMPLOOPNUM,RANK,PCT,M2LCT,RES,ACC,R1,STABLE)
## if(AmplitudeReduction) {
      RES(1)=NORMALIZATION*RES(1)
      RES(2)=NORMALIZATION*RES(2)
      RES(3)=NORMALIZATION*RES(3)
## } else {
      RES(1)=NORMALIZATION*2.0d0*DBLE(RES(1))
      RES(2)=NORMALIZATION*2.0d0*DBLE(RES(2))
      RES(3)=NORMALIZATION*2.0d0*DBLE(RES(3))
## }
C     WRITE(*,*) 'Loop ID',ID,' =',RES(1),RES(2),RES(3)
      END

	  SUBROUTINE %(proc_prefix)sINITCT()
C 
C INITIALISATION OF CUTTOOLS
C  
C LOCAL VARIABLES 
C
	  %(real_dp_format)s THRS
	  LOGICAL EXT_NUM_FOR_R1
C  
C GLOBAL VARIABLES 
C
	  include 'MadLoopParams.inc'
C ----------
C BEGIN CODE
C ----------

C DEFAULT PARAMETERS FOR CUTTOOLS
C -------------------------------  
C THRS1 IS THE PRECISION LIMIT BELOW WHICH THE MP ROUTINES ACTIVATES
	  THRS=CTSTABTHRES
C LOOPLIB SET WHAT LIBRARY CT USES
C 1 -> LOOPTOOLS
C 2 -> AVH
C 3 -> QCDLOOP
      LOOPLIB=CTLOOPLIBRARY
C MADLOOP'S NUMERATOR IN THE OPEN LOOP IS MUCH FASTER THAN THE RECONSTRUCTED ONE IN CT. SO WE BETTER USE MADLOOP ONE IN THIS CASE.
      EXT_NUM_FOR_R1=.TRUE.
C -------------------------------	  

C The initialization below is for CT v1.8.+
      CALL CTSINIT(THRS,LOOPLIB,EXT_NUM_FOR_R1)
C The initialization below is for the older stable CT v1.7, still used for now in the beta release.
C      CALL CTSINIT(THRS,LOOPLIB)

	  END
