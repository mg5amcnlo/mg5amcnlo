      SUBROUTINE CLOSE_V(Q,M,AMPS,RES)
      
      COMPLEX*16 Q(0:3)
      COMPLEX*16 RES
      COMPLEX*16 AMPS(4)

      IF (M.NE.0.D0) THEN
        STOP 'Massive vector L-cut particle not supported'
      ENDIF
      RES=(0.D0,1.D0)*(-AMPS(1)+AMPS(2)+AMPS(3)+AMPS(4))

      END

      SUBROUTINE CLOSE_F(Q,M,AMPS,RES)      
      
      COMPLEX*16 Q(0:3)
      COMPLEX*16 RES, QNORM
      REAL*8 M
      COMPLEX*16 AMPS(4) 
      COMPLEX*16 PMM, PPM

      PPM=AMPS(1)+AMPS(2)
      PMM=AMPS(3)+AMPS(4)
      IF (M.NE.0.D0) THEN
        QNORM=CSQRT(Q(0)**2-Q(1)**2-Q(2)**2-Q(3)**2)      
        RES=(0.D0,0.5D0)*((PPM+PMM)+(PPM-PMM)*(M/QNORM))
      ELSE
        RES=(0.D0,0.5D0)*(PPM+PMM)        
      ENDIF

      END

      SUBROUTINE CLOSE_S(Q,AMP,RES)

      COMPLEX*16 Q(0:3)
      COMPLEX*16 RES
      COMPLEX*16 AMP

      RES=(0.D0,1.D0)*AMP
      
      END


      SUBROUTINE LCUT_F(Q,M,CFIG,SCD,W)

      COMPLEX*16 Q(0:3)
      INTEGER CFIG
      LOGICAL SCD
      REAL*8 M
      COMPLEX*16 W(20)

      IF (CFIG.EQ.1) THEN
        IF SCD THEN
C         UBAR, HEL=-1        
          CALL ILXXXX(Q(0),M,-1,1,W(1))
        ELSE
C         U, HEL=-1        
          CALL ILXXXX(Q(0),M,-1,-1,W(1))
        ENDIF
      ELSEIF (CFIG.EQ.2) THEN
        IF SCD THEN
C         UBAR, HEL=1        
          CALL ILXXXX(Q(0),M,1,1,W(1))
        ELSE
C         U, HEL=1        
          CALL ILXXXX(Q(0),M,1,-1,W(1))
        ENDIF
      ELSEIF (CFIG.EQ.3) THEN
        IF SCD THEN
C         VBAR, HEL=-1,        
          CALL OLXXXX(Q(0),M,-1,-1,W(1))
        ELSE
C         V, HEL=-1        
          CALL OLXXXX(Q(0),M,-1,1,W(1))
        ENDIF
      ELSEIF (CFIG.EQ.4) THEN
        IF SCD THEN
C         VBAR, HEL=1        
          CALL OLXXXX(Q(0),M,1,-1,W(1))
        ELSE
C         V, HEL=1    
          CALL OLXXXX(Q(0),M,1,1,W(1))
        ENDIF
      ENDIF
C     REVERSE THE MOMENTUM IN THE WF FOR THE SECOND L-CUT SPINORS      
      IF SCD THEN
        WF(5)=-Q(1)
        WF(6)=-Q(2)
        WF(7)=-Q(3)
        WF(8)=-Q(4)
      ENDIF

      END

      SUBROUTINE LCUT_CF(Q,M,CFIG,SCD,W)

      COMPLEX*16 Q(0:3)
      INTEGER CFIG
      LOGICAL SCD
      REAL*8 M
      COMPLEX*16 W(20)

      IF (CFIG.EQ.1) THEN
        IF SCD THEN
C         UBAR, HEL=-1        
          CALL ICLXXXX(Q(0),M,-1,1,W(1))
        ELSE
C         U, HEL=-1        
          CALL ICLXXXX(Q(0),M,-1,-1,W(1))
        ENDIF
      ELSEIF (CFIG.EQ.2) THEN
        IF SCD THEN
C         UBAR, HEL=1        
          CALL ICLXXXX(Q(0),M,1,1,W(1))
        ELSE
C         U, HEL=1        
          CALL ICLXXXX(Q(0),M,1,-1,W(1))
        ENDIF
      ELSEIF (CFIG.EQ.3) THEN
        IF SCD THEN
C         VBAR, HEL=-1,        
          CALL OCLXXXX(Q(0),M,-1,-1,W(1))
        ELSE
C         V, HEL=-1        
          CALL OCLXXXX(Q(0),M,-1,1,W(1))
        ENDIF
      ELSEIF (CFIG.EQ.4) THEN
        IF SCD THEN
C         VBAR, HEL=1        
          CALL OCLXXXX(Q(0),M,1,-1,W(1))
        ELSE
C         V, HEL=1    
          CALL OCLXXXX(Q(0),M,1,1,W(1))
        ENDIF
      ENDIF
C     REVERSE THE MOMENTUM IN THE WF FOR THE SECOND L-CUT SPINORS      
      IF SCD THEN
        WF(5)=-Q(1)
        WF(6)=-Q(2)
        WF(7)=-Q(3)
        WF(8)=-Q(4)
      ENDIF

      END

      SUBROUTINE LCUT_V(Q,M,CFIG,SCD,W)

      COMPLEX*16 Q(0:3)
      INTEGER CFIG
      LOGICAL SCD      
      REAL*8 M
      COMPLEX*16 W(20)

      IF M.NE.0.D0 THEN
        STOP 'Massive vector L-cut particle not supported'
      ENDIF
      W(1)=(0.d0,0.d0)
      W(2)=(0.d0,0.d0)
      W(3)=(0.d0,0.d0)
      W(4)=(0.d0,0.d0)
      IF (CFIG.EQ.1) THEN
        W(1)=(1.d0,0.d0)
      ELSEIF (CFIG.EQ.2) THEN
        W(2)=(1.d0,0.d0)
      ELSEIF (CFIG.EQ.3) THEN
        W(3)=(1.d0,0.d0)
      ELSEIF (CFIG.EQ.4) THEN
        W(4)=(1.d0,0.d0)
      ENDIF
C     REVERSE THE MOMENTUM IN THE WF FOR THE SECOND L-CUT SPINORS      
      IF SCD THEN
        WF(5)=-Q(1)
        WF(6)=-Q(2)
        WF(7)=-Q(3)
        WF(8)=-Q(4)
      ENDIF

      END

      SUBROUTINE LCUT_S(Q,SCD,W)

      COMPLEX*16 Q(0:3)
      LOGICAL CJG      
      COMPLEX*16 W(20)

      WF(1)=(1.D0,0.D0)

C     REVERSE THE MOMENTUM IN THE WF FOR THE SECOND SPINORS      
      IF SCD THEN
        WF(2)=-Q(1)
        WF(3)=-Q(2)
        WF(4)=-Q(3)
        WF(5)=-Q(4)
      ENDIF

      END

      SUBROUTINE MPLOOPNUM(Q,RES)
C  
C ARGUMENTS 
C  
      INCLUDE 'cts_mprec.h'
      INCLUDE 'cts_mc.h', INTENT(IN), DIMENSION(0:3) :: Q
      INCLUDE 'cts_mc.h', INTENT(OUT) :: RES
C  
C LOCAL VARIABLES 
C
      COMPLEX*16 DRES
      COMPLEX*16 DQ(0:3)
      INTEGER I
C ----------
C BEGIN CODE
C ----------
      DO I=0,3
        DQ(I) = Q(I)
      ENDDO

      CALL LOOPNUM(DQ,DRES)
      RES=DRES

      END

C THE CORE SUBROUTINE CALLED BY CUTTOOLS WHICH CONTAINS THE HELAS CALLS BUILDING THE LOOP

      SUBROUTINE LOOPNUM(Q,RES)
C  
C CONSTANTS 
C  
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
	  INTEGER    MAXLCOUPLINGS
      PARAMETER (NEXTERNAL=%(maxlcouplings)d)
	  INTEGER    NMAXLOOPWFS
      PARAMETER (NMAXLOOPWFS=(NEXTERNAL+1))
C
C WHEN NOT USING A COMPLEX SCHEME, PARTICLES IN THE LOOP MUST HAVE ZERO WIDTH
C
	  REAL*8     LW
	  PARAMETER (LW = 0.D0)
C  
C ARGUMENTS 
C  
      COMPLEX*16 Q(0:3)
      COMPLEX*16 RES
C  
C LOCAL VARIABLES 
C
      COMPLEX*16 BUFF(4)
      COMPLEX*16 WL(20,NMAXLOOPWFS)
      INTEGER I
C
C GLOBAL VARIABLES
C
      COMPLEX*16 WE(20,NEXTERNAL)
	  COMPLEX*16 LC(MAXLCOUPLINGS)
	  REAL*8 ML(NEXTERNAL)	  
      INTEGER ID
      COMMON/LOOP/WE,LC,ML,ID
C ----------
C BEGIN CODE
C ----------
      RES=(0.D0,0.D0)
      %(loop_helas_calls)s
      END
