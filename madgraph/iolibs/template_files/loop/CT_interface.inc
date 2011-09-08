	  SUBROUTINE CTLOOP(NLOOPLINE,PL,M2L,RES)
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
C  
C ARGUMENTS 
C
      INTEGER NLOOPLINE
      REAL*8 PL(0:3,NLOOPLINE)
      REAL*8 PCT(0:3,0:NLOOPLINE-1)
      REAL*8 M2L(NLOOPLINE)
      COMPLEX*16 M2LCT(0:NLOOPLINE-1)	  
      COMPLEX*16 RES(3)
C  
C LOCAL VARIABLES 
C
	  REAL*8 THRS1
	  REAL*8 THRS2
	  COMPLEX*16 R1
	  INTEGER MPDIG
	  INTEGER RANK, I, J, K
	  LOGICAL STABLE,FORCEMP
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

C DEFAULT PARAMETERS FOR CUTTOOLS
C -------------------------------  
      RANK=NLOOPLINE
C TO FORCE CUT-TOOL TO USE MULTIPLE PRECISION
      FORCEMP=.FALSE.
C MPDIG IS THE NUMBER OF DIGITS USED IN THE MULTIPLE PRECISION SUBROUTINES
	  MPDIG=64
C THRS1 IS THE PRECISION LIMIT BELOW WHICH THE MP ROUTINES ACTIVATES
	  THRS1=1.D-2
C THRS1 IS THE NUMERICAL THRESHOLD FOR THE COLLINEAR/SOFT DIVERGENCES IN AVH_OLO ROUTINES	  
	  THRS2=1.D-6
C LOOPLIB SET WHAT LIBRARY CT USES
C 1 -> LOOPTOOLS
C 2 -> AVH
C 3 -> QCDLOOP
      LOOPLIB=2
C -------------------------------	  

      CALL CTSINIT(LSCALE,THRS1,MPDIG,LOOPLIB,MU_R,THRS2)

C  Convert the masses to be complex
      do I=1,NLOOPLINE
	    M2LCT(I-1)=DCMPLX(M2L(I),0.D0)
	  ENDDO

C  Convert the flowing in loop momentum to CT convention
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

      CALL CTSXCUT(LSCALE,MU_R,NLOOPLINE,LOOPNUM,MPLOOPNUM,RANK,PCT,M2LCT,RES,R1,STABLE,FORCEMP)
	  RES(1)=RES(1)+R1

	  END
