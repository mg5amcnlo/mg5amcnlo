	  SUBROUTINE CTLOOP(NLOOPLINE,PL,M2L,RANK,RES)
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
      REAL*8 NORMALIZATION
      PARAMETER (NORMALIZATION = 1.D0/(16.D0*3.14159265358979323846D0**2))
C  
C ARGUMENTS 
C
      INTEGER NLOOPLINE, RANK
      REAL*8 PL(0:3,NLOOPLINE)
      REAL*8 PCT(0:3,0:NLOOPLINE-1)
      %(mass_format)s M2L(NLOOPLINE)
      COMPLEX*16 M2LCT(0:NLOOPLINE-1)	  
      COMPLEX*16 RES(3)
C  
C LOCAL VARIABLES 
C
	  COMPLEX*16 R1, ACC
	  INTEGER I, J, K
	  INTEGER CTMODE
	  LOGICAL STABLE
	  LOGICAL INIT
	  DATA INIT /.TRUE./
	  SAVE INIT
C
C EXTERNAL FUNCTIONS
C
      EXTERNAL LOOPNUM
      EXTERNAL MPLOOPNUM
C
C GLOBAL VARIABLES
C
      include 'coupl.inc'
	  REAL*8 LSCALE
      common/CT/LSCALE

C ----------
C BEGIN CODE
C ----------

C INITIALIZE CUTTOOLS IF NEEDED
      IF (INIT) THEN
	    INIT=.FALSE.
		CALL INITCT()
      ENDIF

C DECIDE WHAT CT MODE TO USE FOR TREATING THIS PARTICULAR LOOP
C IN THE ACTUAL SETUP, WE ALWAYS USE ZERO.
C YOU CAN FIND THE DETAILS ABOUT THE DIFFERENT MODE AT THE BEGINNING OF THE FILE CTS_CUTS.F90 IN THE CUTTOOLS DISTRIBUTION

      CTMODE=0

C  CONVERT THE MASSES TO BE COMPLEX
      do I=1,NLOOPLINE
	    M2LCT(I-1)=DCMPLX(M2L(I),0.D0)
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

      CALL CTSXCUT(CTMODE,LSCALE,MU_R,NLOOPLINE,LOOPNUM,MPLOOPNUM,RANK,PCT,M2LCT,RES,ACC,R1,STABLE)	  
      RES(1)=NORMALIZATION*RES(1)
      RES(2)=NORMALIZATION*RES(2)
      RES(3)=NORMALIZATION*RES(3)
      END

	  SUBROUTINE INITCT()
C 
C INITIALISATION OF CUTTOOLS
C  
C LOCAL VARIABLES 
C
	  REAL*8 THRS

C ----------
C BEGIN CODE
C ----------

C DEFAULT PARAMETERS FOR CUTTOOLS
C -------------------------------  
C THRS1 IS THE PRECISION LIMIT BELOW WHICH THE MP ROUTINES ACTIVATES
	  THRS=1.D-2
C LOOPLIB SET WHAT LIBRARY CT USES
C 1 -> LOOPTOOLS
C 2 -> AVH
C 3 -> QCDLOOP
      LOOPLIB=2
C -------------------------------	  

      CALL CTSINIT(THRS,LOOPLIB)

	  END
