c***********************************************************************
c     Routines for generating momentum of decay products
c***********************************************************************

      SUBROUTINE PHASESPACE(PD,PSWGT)
C************************************************************************
C     WRAPPER FOR CALLING THE PHASE SPACE ROUTINES
C************************************************************************
      IMPLICIT NONE
      include 'decay.inc'
C
C     ARGUMENTS
C
      REAL*8 PD(0:3,MaxDecPart)
      REAL*8 PSWGT
C
C     LOCAL
C
      LOGICAL DONE
      INTEGER IMAX
C
C----------
C     START
C----------
C
      PSWGT=0d0
      IMAX=0

      IF(ND.EQ.2) THEN
         CALL TWOMOM(PD(0,1),PD(0,2),PD(0,3),PSWGT)
      ELSEIF(ND.EQ.3) THEN
         CALL THREEMOM(PD(0,1),PD(0,2),PD(0,3),PD(0,4),PSWGT)
      ELSEIF(ND.EQ.4) THEN
         CALL FOURMOM(PD(0,1),PD(0,2),PD(0,3),PD(0,4),PD(0,5),PSWGT)
      ELSE
         WRITE(*,*) 'NO PS AVAILABLE '
         STOP
      ENDIF


      RETURN
      END


      SUBROUTINE TWOMOM(P1,P2,P3,PSWGT)
C************************************************************************
C     GENERIC TWO BODY PHACE SPACE FOR THE DECAYS P1->P2+P3
C
C     GIVEN P1 MOMENTUM SETS UP THE MOMENTA P2,P3
C     ALSO SETS UP THE APPROPRIATE PHASESPACE WEIGTH PSWGT
C
C     X(1) = COSTHETA OF FIRST DECAY  
C     X(2) = PHI OF FIRST DECAY       
C
C************************************************************************
      IMPLICIT NONE
      include 'decay.inc'
C
C     ARGUMENTS
C
      REAL*8 P1(0:3),P2(0:3),P3(0:3)
      REAL*8 PSWGT
C     
C     LOCAL
C
      REAL*8 COSTH,PHI,JACPOLE
      REAL*8 RS
      REAL*8 XA2,XB2,S,JAC
      REAL*8 a
      integer i
C
C     EXTERNAL
C
      REAL*8 LAMBDA,DOT
C
C----------
C     BEGIN
C----------
C
C     CMS ENERGY = (P1)^2
C
      S=DOT(P1,P1)
      RS=SQRT(S)
C     
C     PICK VALUE OF THETA AND PHI 
C     
      COSTH=-1.D0+2.D0*X(1)
      PHI  = 2D0*PI*X(2)
C     
C     DETERMINE JACOBIAN FOR THETA AND PHI
C     
      JAC =  4D0*PI
C     
C     CALCULATE COMPONENTS OF MOMENTUM FOUR-VECTORS
C     OF THE FINAL STATE MASSLESS PARTONS IN THEIR CM FRAME
C     
      CALL MOM2CX(RS,M2,M3,COSTH,PHI,P2,P3) !DECAY P1
      CALL BOOSTX(P2,P1,P2)    !BOOST DECAY TO LAB
      CALL BOOSTX(P3,P1,P3)    !BOOST DECAY TO LAB  
C     
      JAC = JAC /(2D0*PI)**2    !FOR THE (2 PI)^3N-4 OUT FRONT
C     
C     CALCULATE  PHASE SPACE FACTOR DSIG/DCOS(THETA)
C     
      PSWGT = 1.D0/(2.D0*RS)    !FLUX FACTOR
      XA2 =   M2*M2/S
      XB2 =   M3*M3/S
      PSWGT = PSWGT*.5D0*PI*SQRT(LAMBDA(ONE,XA2,XB2))/(4.D0*PI) 
 
      PSWGT=PSWGT*JAC

c      write(*,*) 'from decay_mom',x(1),x(2),pswgt
      RETURN
      END
 


      SUBROUTINE THREEMOM(P1,P2,P3,P4,PSWGT)
C************************************************************************
C     This routine generates the momenta for the decay products of
C
C     1 -> 2 W -> 2 3 4 
C
C     when the W can be on-shell or not.
C     It ALSO SETS UP THE APPROPRIATE PHASESPACE WEIGTH PSWGT
C
C     X(1) = COSTHETA OF FIRST DECAY TOP -> W*+B
C     X(2) = PHI OF FIRST DECAY      TOP -> W*+B
C     X(3) = M_W*
C     X(4) = COSTHETA OF  DECAY      W* -> E+VE
C     X(5) = PHI OF       DECAY      W* -> E+VE
C
C************************************************************************
      IMPLICIT NONE
      include 'decay.inc'
C
C     ARGUMENTS
C
      REAL*8 P1(0:3),P2(0:3),P3(0:3),P4(0:3),PW(0:3)
      REAL*8 PSWGT
C
C     LOCAL
C
      REAL*8 COSTH,PHI,JACPOLE
      REAL*8 RS
      REAL*8 COSTH1,PHI1,S1,RS1
      REAL*8 COSTH2,PHI2,S2,RS2
      REAL*8 XA2,XB2,S,JAC
      REAL*8 a
      INTEGER I
C
C     EXTERNAL
C
      REAL*8 LAMBDA,DOT
C
C     LOGICAL 
C
      LOGICAL FIRSTTIME
      DATA FIRSTTIME /.TRUE./
C
C----------
C     BEGIN
C----------
C
C     CMS ENERGY 
C

      S=DOT(P1,P1)
      RS=SQRT(S)
      
      IF(RS.GT.(M2+MV)) THEN   ! W CAN BE ON SHELL
         CALL TRANSPOLE(MV**2/S,MV*GV/S,X(3),S2,JACPOLE)
         S2 = S2*S
         RS2 = SQRT(S2)
         JAC=JACPOLE*S
      ELSE                     ! W is off-shell
         S2=(S-M2*M2)*X(3)+M2*M2     ! generated uniformily
         RS2 = SQRT(S2)
         JAC=(S-M2*M2)
      ENDIF   
C
C     check total energy conservation
C
      IF(RS2.GE.(RS-M2).or.RS2.LT.(M3+M4)) THEN
         PSWGT=0D0
         RETURN
      ENDIF
C     
C     PICK VALUE OF THETA AND PHI FOR FIRST DECAY TOP -> W*+B
C     
      COSTH=-1.D0+2.D0*X(1)
      PHI = 2D0*PI*X(2)
C     
C     DETERMINE JACOBIAN FOR THETA AND PHI
C     
      JAC =  JAC * 4D0*PI
C     
C     CHOOSE COS(THETA) AND PHI FOR DECAY W*-> E+VE
C     
      COSTH1 = -1 + 2.*X(4)
      PHI1   = 2D0*PI*X(5)
C     
C     JACOBIAN FACTOR FOR DECAY OF W*-> E+VE
C     
      JAC=JAC*4D0*PI
C     
C     CALCULATE COMPONENTS OF MOMENTUM FOUR-VECTORS
C     OF THE FINAL STATE MASSLESS PARTONS IN THEIR CM FRAME
C     
      CALL MOM2CX(RS,RS2,M2,COSTH ,PHI ,PW,P2) !DECAY 1
      CALL MOM2CX(RS2,M3,M4,COSTH1,PHI1,P3,P4) !DECAY W*
      CALL BOOSTX(P2,P1,P2)     !BOOST DECAY TO LAB
      CALL BOOSTX(PW,P1,PW)     !BOOST DECAY TO LAB
      CALL BOOSTX(P3,PW,P3)     !BOOST DECAY TO LAB
      CALL BOOSTX(P4,PW,P4)     !BOOST DECAY TO LAB  
      
      JAC = JAC /(2D0*PI)**5    !FOR THE (2 PI)^3N-4 OUT FRONT
C     
C     CALCULATE  PHASE SPACE FACTOR DSIG/DCOS(THETA)
C     
      
      PSWGT = 1.D0/(2.D0*RS)    !FLUX FACTOR
      XA2 =   S2/S
      XB2 =   M2*M2/S
      PSWGT = PSWGT*.5D0*PI*SQRT(LAMBDA(ONE,XA2,XB2))/(4.D0*PI) !T->W*+B
      XA2 =   M3*M3/S
      XB2 =   M4*M4/S
      PSWGT = PSWGT*.5D0*PI*SQRT(LAMBDA(ONE,XA2,XB2))/(4.D0*PI) !W*->E+VE
 
      
      PSWGT=PSWGT*JAC

      RETURN
      END

 
      SUBROUTINE FOURMOM(P1,P2,P3,P4,P5,PSWGT)
C************************************************************************
C     This routine generates the momenta for the decay products of
C
C     1 -> v v'* -> 2 3 4 5
C
C     where the v'* can be on-shell or not. 
c
C     GIVEN X(8) AND THE P1 MOMENTUM SETS UP THE MOMENTA
C     ALSO SETS UP THE APPROPRIATE PHASESPACE WEIGTH PSWGT
C
C     X(1) = COSTHETA OF FIRST DECAY SH -> M_(2+3)+M_(4,5)
C     X(2) = PHI OF FIRST DECAY      SH -> M_(2+3)+M_(4,5)
C     X(3) = M_(2,3)
C     X(4) = M_(4,5)
C     X(5) = COSTHETA OF  DECAY M_(2,3) -> M2+M3
C     X(6) = PHI OF FIRST DECAY M_(2,3) -> M2+M3
C     X(7) = COSTHETA OF  DECAY M_(4,5) -> M4+M5
C     X(8) = PHI OF FIRST DECAY M_(4,5) -> M4+M5
C
C************************************************************************
      IMPLICIT NONE
      include "decay.inc"
C
C     ARGUMENTS
C
      REAL*8 P1(0:3),P2(0:3),P3(0:3),P4(0:3),P5(0:3)
      REAL*8 PSWGT
C
C     LOCAL
C
      REAL*8 JAC,JACPOLE
      REAL*8 RSH,SH,COSINIT,PHIINIT,COSTH,PHI
      REAL*8 COSTH1,PHI1,S1,PTEMP1(0:3),X3MIN,X3MAX,RS1
      REAL*8 COSTH2,PHI2,S2,PTEMP2(0:3),X4MIN,X4MAX,RS2
      REAL*8 XA2,XB2,SMIN
      REAL*8 TEMP
      INTEGER I
C
C     EXTERNAL
C
      REAL*8 LAMBDA,DOT
      real xran1
      integer iseed
      data iseed/1/  !no effect if already called
C
C     LOGICAL 
C
      LOGICAL FIRSTTIME
      DATA FIRSTTIME /.TRUE./
C
C----------
C     BEGIN
C----------
c
      JAC=1D0
C
C     CMS ENERGY = (HIGGS MASS)^2
C
      SH=DOT(P1,P1)
      RSH=SQRT(SH)
C     
C     PICK VALUE OF THETA AND PHI FOR FIRST DECAY SH->M3+S1
C     
      COSTH=-1.D0+2.D0*X(1)
      PHI  =  2D0*PI*X(2)
C     
C     DETERMINE JACOBIAN FOR THETA AND PHI
C     
      JAC =  JAC * 4D0*PI
C
C     MASSES OF THE DECAY 1->V V
C
      IF(RSH.GT.MV) THEN        ! AT LEAST A W CAN BE ON SHELL
         CALL TRANSPOLE(MV**2/SH,MV*GV/SH,X(3),S1,JACPOLE)
         S1 = S1*SH
         RS1 = SQRT(S1)
         JAC=JACPOLE*SH*JAC
      ELSE                      ! NO W'S CAN BE ON SHELL
         S1=SH*X(3)              ! generated uniformily
         RS1 = SQRT(S1)
         JAC=SH*JAC
      ENDIF   

      IF(RS1.GE.RSH) THEN   ! ENERGY CONSERVATION
         PSWGT=0D0
         RETURN
      ENDIF
C     
C     DETERMINE INVARIENT MASS OF SYSTEM M4+M5
C  
      IF(RSH-RS1.GT.MV) THEN    ! THE SECOND W CAN BE ON-SHELL
         CALL TRANSPOLE(MV**2/SH,MV*GV/SH,X(4),S2,JACPOLE)
         S2 = S2*SH
         RS2 = SQRT(S2)
         JAC=JACPOLE*SH*JAC
      else                      ! the second w cannot be on-shell
c         X4MIN = (M4+M5)**2/SH
         X4MIN = 0d0   ! less efficient but symmetric in m2,m3 <-> m4,m5
         X4MAX = (1D0-RS1/RSH)**2
         S2    = ((X4MAX-X4MIN)*X(4)+X4MIN)*SH
         RS2   = SQRT(S2)
         JAC   = JAC*SH*(X4MAX-X4MIN)
      ENDIF
  
      IF(RS2+RS1.GT.RSH) THEN  ! ENERGY CONSERVATION
         PSWGT=0D0
         RETURN
      ENDIF
c      
c     switch order to ensure no bias
c
      IF(xran1(iseed).lt.0.5e0)then
         temp=rs1
         rs1=rs2
         rs2=temp
         temp=s1
         s1=s2
         s2=temp
      ENDIF
      
      IF(RS1 .LT. (M2+M3)) THEN  ! MINIMUM inv mass
         PSWGT=0D0
         RETURN
      ENDIF

      IF(RS2 .LT. (M4+M5)) THEN  ! MINIMUM inv mass
         PSWGT=0D0
         RETURN
      ENDIF
      

C
C     CHOOSE COS(THETA) AND PHI FOR DECAY S1->M3+M4
C     
      COSTH1 = -1 + 2.*X(5)
      PHI1   = 2D0*PI*X(6)
C     
C     CHOOSE COS(THETA) AND PHI FOR DECAY S2->M4+M5
C     
      COSTH2 = -1 + 2.*X(7)
      PHI2   = 2D0*PI*X(8)
C     
C     JACOBIAN FACTOR FOR DECAY OF S1->M3+M4
C     
      JAC=JAC*4D0*PI
C     
C     JACOBIAN FACTOR FOR DECAY OF S2->M4+M5
C     
      JAC=JAC*4D0*PI
C     
C     CALCULATE COMPONENTS OF MOMENTUM FOUR-VECTORS
C     OF THE FINAL STATE
C     
      CALL MOM2CX(RSH,RS1,RS2,COSTH,PHI,PTEMP1,PTEMP2)
      CALL MOM2CX(RS1,M2,M3,COSTH1,PHI1,P2,P3) !DECAY PTEMP1
      CALL MOM2CX(RS2,M4,M5,COSTH2,PHI2,P4,P5) !DECAY PTEMP2
      CALL BOOSTX(PTEMP1,P1,PTEMP1) !BOOST DECAY TO CM
      CALL BOOSTX(PTEMP2,P1,PTEMP2) !BOOST DECAY TO CM
      CALL BOOSTX(P2,PTEMP1,P2) !BOOST DECAY TO CM
      CALL BOOSTX(P3,PTEMP1,P3) !BOOST DECAY TO CM
      CALL BOOSTX(P4,PTEMP2,P4) !BOOST DECAY TO CM
      CALL BOOSTX(P5,PTEMP2,P5) !BOOST DECAY TO CM
C      
C        
      JAC = JAC /(2D0*PI)**8    !FOR THE (2 PI)^3N-4 OUT FRONT
C      
C     
C     CALCULATE  PHASE SPACE FACTOR DSIG/DCOS(THETA)
C     
      
      PSWGT = 1.D0/(2.D0*RSH)   !FLUX FACTOR
      XA2 =   S1/SH
      XB2 =   S2/SH
      PSWGT = PSWGT*.5D0*PI*SQRT(LAMBDA(ONE,XA2,XB2))/(4.D0*PI) !S->S1+S2
      XA2 =   M2*M2/S1
      XB2 =   M3*M3/S1
      PSWGT = PSWGT*.5D0*PI*SQRT(LAMBDA(ONE,XA2,XB2))/(4.D0*PI) !S1->M2+M3
      XA2 =   M4*M4/S2
      XB2 =   M5*M5/S2
      PSWGT = PSWGT*.5D0*PI*SQRT(LAMBDA(ONE,XA2,XB2))/(4.D0*PI) !S2->M4+M5
      
      PSWGT=PSWGT*JAC



      RETURN
      END

    
      REAL*8 FUNCTION LAMBDA(S,MA2,MB2)
      IMPLICIT NONE
C****************************************************************************
C     THIS IS THE LAMBDA FUNCTION FROM VERNONS BOOK COLLIDER PHYSICS P 662
C     MA2 AND MB2 ARE THE MASS SQUARED OF THE FINAL STATE PARTICLES
C     2-D PHASE SPACE = .5*PI*SQRT(1.,MA2/S^2,MB2/S^2)*(D(OMEGA)/4PI)
C****************************************************************************
      REAL*8 MA2,MB2,S
      LAMBDA=S**2+MA2**2+MB2**2-2.*S*MA2-2.*MA2*MB2-2.*S*MB2
      RETURN
      END
      

      SUBROUTINE TRANSPOLE(POLE,WIDTH,X,Y,JAC)
C**********************************************************************
C     THIS ROUTINE TRANSFERS EVENLY SPACED X VALUES BETWEEN 0 AND 1
C     TO Y VALUES WITH A POLE AT Y=POLE WITH WIDTH WIDTH AND RETURNS
C     THE APPROPRIATE JACOBIAN FOR THIS 
C**********************************************************************
      IMPLICIT NONE
C
C     ARGUMENTS
C
      REAL*8 POLE,WIDTH,Y,JAC
      REAL*4 X
C
C     LOCAL
C
      REAL*8 Z,ZMIN,ZMAX
c-----
c  Begin Code
c-----

      ZMIN = ATAN((-POLE)/WIDTH)/WIDTH
      ZMAX = ATAN((1.-POLE)/WIDTH)/WIDTH
      Z = ZMIN+(ZMAX-ZMIN)*X
      Y = POLE+WIDTH*TAN(WIDTH*Z)
      JAC= (WIDTH/COS(WIDTH*Z))**2*(ZMAX-ZMIN)     
      RETURN
      END


