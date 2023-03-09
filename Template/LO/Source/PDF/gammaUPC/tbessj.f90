!************************************************************************
!*                                                                      *
!*    Program to calculate the first kind Bessel function of integer    *
!*    order N, for any REAL X, using the function BESSJ(N,X).           *
!*                                                                      *
!* -------------------------------------------------------------------- *
!*                                                                      *
!*    SAMPLE RUN:                                                       *
!*                                                                      *
!*    (Calculate Bessel function for N=2, X=0.75).                      *
!*                                                                      *
!*    Bessel function of order  2 for X =  0.7500:                      *
!*                                                                      *
!*         Y =  0.67073997E-01                                          *
!*                                                                      *
!* -------------------------------------------------------------------- *
!*   Reference: From Numath Library By Tuan Dang Trong in Fortran 77.   *
!*                                                                      *
!*                               F90 Release 1.0 By J-P Moreau, Paris.  *
!*                                      all variables declared          *
!*                                         (www.jpmoreau.fr)            *
!************************************************************************

      REAL*8 FUNCTION ZEROJP(N,K)
!--------------------------------------------------------------------
!     CALCULATE THE Kth ZERO OF THE DERIVATIVE OF BESSEL FUNCTION
!     OF ORDER N, J(N,X) 
!--------------------------------------------------------------------
!     CALLING MODE:
!       RES = ZEROJP(N,K)
!
!     INPUTS:
!       N    ORDER OF BESSEL FUNCTION J (INTEGER >= 0)            I*4
!       K    RANK OF ZERO (INTEGER > 0)                           I*4
!     OUTPUT:
!       ZEROJP                                                    R*8
!     REFERENCE:
!     ABRAMOWITZ M. & STEGUN IRENE A.
!     HANDBOOK OF MATHEMATICAL FUNCTIONS
!---------------------------------------------------------------------
      REAL*8 BESSJP,B0,B1,B2,B3,B5,B7,T0,T1,T3,T5,T7,PI,FN,FK,  &
      C1,C2,C3,C4,F1,F2,F3,P,DP,P0,P1,Q0,Q1,TOL
      LOGICAL IMPROV
      DATA TOL/1.D-7/,NITMX/15/
      DATA C1,C2,C3,C4 /0.8086165D0,0.072490D0,.05097D0,.0094D0/
      DATA IMPROV/.TRUE./
      
      PI = 4.d0*ATAN(1.d0)

        FN = DFLOAT(N)
      FK = DFLOAT(K)

      IF (K.GT.1) GO TO 10

!     SI N = 0 ET K = 1

      IF (N.EQ.0) THEN
      ZEROJP= 0.D0
      RETURN

!     TCHEBYCHEV'S SERIES FOR K <= N

      ELSE

      F1 = FN**(1.D0/3.D0)
      F2 = F1*F1*FN
      ZEROJP = FN+C1*F1+(C2/F1)-(C3/FN)+(C4/F2)
      GO TO 20
      ENDIF

!     MAC MAHON'S SERIES FOR K >> N

   10 B0 = (FK+.5D0*FN-.75D0)*PI
      B1 = 8.D0*B0
      B2 = B1*B1
      B3 = 3.D0*B1*B2
      B5 = 5.D0*B3*B2
      B7 = 7.D0*B5*B2
      T0 = 4.D0*FN*FN
      T1 = T0+3.D0
      T3 = 4.D0*((7.D0*T0+82.D0)*T0-9.D0)
      T5 = 32.D0*(((83.D0*T0+2075.D0)*T0-3039.D0)*T0+3537.D0)
      T7 = 64.D0*((((6949.D0*T0+296492.D0)*T0-1248002.D0)*T0  &
                      +7414380.D0)*T0-5853627.D0)
      ZEROJP = B0-(T1/B1)-(T3/B3)-(T5/B5)-(T7/B7)

   20 IF (IMPROV) THEN

!     IMPROVE SOLUTION BY SECANT METHOD WHEN K > N
!     AND IMPROV = .TRUE.
      P0 = 0.9D0*ZEROJP
      P1 = ZEROJP
      IER = 0
      NEV = 2
      Q0 = BESSJP(N,P0)
      Q1 = BESSJP(N,P1)
      DO 30 IT = 1,NITMX
      P = P1-Q1*(P1-P0)/(Q1-Q0)
      DP = P-P1
      IF (IT.EQ.1) GO TO 25
      IF (ABS(DP).LT.TOL) GO TO 40
   25 NEV = NEV+1
      P0 = P1
      Q0 = Q1
      P1 = P
      Q1 = BESSJP(N,P1)
   30 CONTINUE
      IER = 1
      WRITE(*,'(1X,A)') '** ZEROJP ** NITMX EXCEEDED'
      RETURN
   40 ZEROJP = P
      ENDIF
      RETURN
      END

      FUNCTION BESSJP (N,X)
! ----------------------------------------------------------------------
!    NAME  :  BESSJP
!    DATE  :  06/01/1982
!    IV    :  1
!    IE    :  1
!    AUTHOR:  DANG TRONG TUAN
! ......................................................................
!
!    FIRST DERIVATIVE OF FIRST KIND BESSEL FUNCTION OF ORDER N, FOR REAL X
!
!                          MODULE BESSJP                               .
! ......................................................................
!
!    THIS SUBROUTINE CALCULATES THE FIRST DERIVATIVE OF FIRST KIND BESSEL 
!    FUNCTION OF ORDER N, FOR REAL X.
!                                                                     .
! ......................................................................
!
!  I  VARIABLE DIMENSION/TYPE  DESCRIPTION  (INPUTS)
!        N       I*4           ORDER OF FUNCTION                             .
!        X       R*8           ABSCISSA OF FUNCTION BESSJP(N,X)             .
!
!  O  VARIABLE,DIMENSION/TYPE  DESCRIPTION  (OUTPUT)
!
!      BESSJP    R*8           FUNCTION EVALUATION AT X                        .
!.......................................................................
!    CALLED SUBROUTINE                                                  
!
!      BESSJ     FIRST KIND BESSEL FUNCTION                            
!
! ----------------------------------------------------------------------
      DOUBLE PRECISION X,BESSJP,BESSJ
      IF (N.EQ.0) THEN
      BESSJP=-BESSJ(1,X)
      ELSE IF(X.EQ.0.D0) THEN
      X=1.D-30
      ELSE
      BESSJP=BESSJ(N-1,X)-( FLOAT(N)/X)*BESSJ(N,X)
      ENDIF
      RETURN
      END

     FUNCTION BESSJ (N,X)

!     This subroutine calculates the first kind modified Bessel function
!     of integer order N, for any REAL X. We use here the classical
!     recursion formula, when X > N. For X < N, the Miller's algorithm
!     is used to avoid overflows. 
!     REFERENCE:
!     C.W.CLENSHAW, CHEBYSHEV SERIES FOR MATHEMATICAL FUNCTIONS,
!     MATHEMATICAL TABLES, VOL.5, 1962.

      IMPLICIT NONE
      INTEGER, PARAMETER :: IACC = 40
      REAL*8, PARAMETER :: BIGNO = 1.D10, BIGNI = 1.D-10
      INTEGER M, N, J, JSUM
      REAL *8 X,BESSJ,BESSJ0,BESSJ1,TOX,BJM,BJ,BJP,SUM
      IF (N.EQ.0) THEN
      BESSJ = BESSJ0(X)
      RETURN
      ENDIF
      IF (N.EQ.1) THEN
      BESSJ = BESSJ1(X)
      RETURN
      ENDIF
      IF (X.EQ.0.) THEN
      BESSJ = 0.
      RETURN
      ENDIF
      TOX = 2./X
      IF (X.GT.FLOAT(N)) THEN
      BJM = BESSJ0(X)
      BJ  = BESSJ1(X)
      DO 11 J = 1,N-1
      BJP = J*TOX*BJ-BJM
      BJM = BJ
      BJ  = BJP
   11 CONTINUE
      BESSJ = BJ
      ELSE
      M = 2*((N+INT(SQRT(FLOAT(IACC*N))))/2)
      BESSJ = 0.
      JSUM = 0
      SUM = 0.
      BJP = 0.
      BJ  = 1.
      DO 12 J = M,1,-1
      BJM = J*TOX*BJ-BJP
      BJP = BJ
      BJ  = BJM
      IF (ABS(BJ).GT.BIGNO) THEN
      BJ  = BJ*BIGNI
      BJP = BJP*BIGNI
      BESSJ = BESSJ*BIGNI
      SUM = SUM*BIGNI
      ENDIF
      IF (JSUM.NE.0) SUM = SUM+BJ
      JSUM = 1-JSUM
      IF (J.EQ.N) BESSJ = BJP
   12 CONTINUE
      SUM = 2.*SUM-BJ
      BESSJ = BESSJ/SUM
      ENDIF
      RETURN
      END

      FUNCTION BESSJ0 (X)
      IMPLICIT NONE
      REAL *8 X,BESSJ0,AX,FR,FS,Z,FP,FQ,XX

!     This subroutine calculates the First Kind Bessel Function of
!     order 0, for any real number X. The polynomial approximation by
!     series of Chebyshev polynomials is used for 0<X<8 and 0<8/X<1.
!     REFERENCES:
!     M.ABRAMOWITZ,I.A.STEGUN, HANDBOOK OF MATHEMATICAL FUNCTIONS, 1965.
!     C.W.CLENSHAW, NATIONAL PHYSICAL LABORATORY MATHEMATICAL TABLES,
!     VOL.5, 1962.

      REAL *8 Y,P1,P2,P3,P4,P5,R1,R2,R3,R4,R5,R6  &
               ,Q1,Q2,Q3,Q4,Q5,S1,S2,S3,S4,S5,S6
      DATA P1,P2,P3,P4,P5 /1.D0,-.1098628627D-2,.2734510407D-4, &
      -.2073370639D-5,.2093887211D-6 /
      DATA Q1,Q2,Q3,Q4,Q5 /-.1562499995D-1,.1430488765D-3, &
      -.6911147651D-5,.7621095161D-6,-.9349451520D-7 /
      DATA R1,R2,R3,R4,R5,R6 /57568490574.D0,-13362590354.D0, &
      651619640.7D0,-11214424.18D0,77392.33017D0,-184.9052456D0 /
      DATA S1,S2,S3,S4,S5,S6 /57568490411.D0,1029532985.D0, &
      9494680.718D0,59272.64853D0,267.8532712D0,1.D0 /
      IF(X.EQ.0.D0) GO TO 1
      AX = ABS (X)
      IF (AX.LT.8.) THEN
      Y = X*X
      FR = R1+Y*(R2+Y*(R3+Y*(R4+Y*(R5+Y*R6))))
      FS = S1+Y*(S2+Y*(S3+Y*(S4+Y*(S5+Y*S6))))
      BESSJ0 = FR/FS
      ELSE
      Z = 8./AX
      Y = Z*Z
      XX = AX-.785398164
      FP = P1+Y*(P2+Y*(P3+Y*(P4+Y*P5)))
      FQ = Q1+Y*(Q2+Y*(Q3+Y*(Q4+Y*Q5)))
      BESSJ0 = SQRT(.636619772/AX)*(FP*COS(XX)-Z*FQ*SIN(XX))
      ENDIF
      RETURN
    1 BESSJ0 = 1.D0
      RETURN
      END
! ---------------------------------------------------------------------------
      FUNCTION BESSJ1 (X)
      IMPLICIT NONE
      REAL *8 X,BESSJ1,AX,FR,FS,Z,FP,FQ,XX
!     This subroutine calculates the First Kind Bessel Function of
!     order 1, for any real number X. The polynomial approximation by
!     series of Chebyshev polynomials is used for 0<X<8 and 0<8/X<1.
!     REFERENCES:
!     M.ABRAMOWITZ,I.A.STEGUN, HANDBOOK OF MATHEMATICAL FUNCTIONS, 1965.
!     C.W.CLENSHAW, NATIONAL PHYSICAL LABORATORY MATHEMATICAL TABLES,
!     VOL.5, 1962.
      REAL *8 Y,P1,P2,P3,P4,P5,P6,R1,R2,R3,R4,R5,R6  &
               ,Q1,Q2,Q3,Q4,Q5,S1,S2,S3,S4,S5,S6
      DATA P1,P2,P3,P4,P5 /1.D0,.183105D-2,-.3516396496D-4,  &
      .2457520174D-5,-.240337019D-6 /,P6 /.636619772D0 /
      DATA Q1,Q2,Q3,Q4,Q5 /.04687499995D0,-.2002690873D-3,   &
      .8449199096D-5,-.88228987D-6,.105787412D-6 /
      DATA R1,R2,R3,R4,R5,R6 /72362614232.D0,-7895059235.D0, & 
      242396853.1D0,-2972611.439D0,15704.48260D0,-30.16036606D0 /
      DATA S1,S2,S3,S4,S5,S6 /144725228442.D0,2300535178.D0, &
      18583304.74D0,99447.43394D0,376.9991397D0,1.D0 /

      AX = ABS(X)
      IF (AX.LT.8.) THEN
      Y = X*X
      FR = R1+Y*(R2+Y*(R3+Y*(R4+Y*(R5+Y*R6))))
      FS = S1+Y*(S2+Y*(S3+Y*(S4+Y*(S5+Y*S6))))
      BESSJ1 = X*(FR/FS)
      ELSE
      Z = 8./AX
      Y = Z*Z
      XX = AX-2.35619491
      FP = P1+Y*(P2+Y*(P3+Y*(P4+Y*P5)))
      FQ = Q1+Y*(Q2+Y*(Q3+Y*(Q4+Y*Q5)))
      BESSJ1 = SQRT(P6/AX)*(COS(XX)*FP-Z*SIN(XX)*FQ)*SIGN(S6,X)
      ENDIF
      RETURN
      END

!End of file Tbessj.f90
