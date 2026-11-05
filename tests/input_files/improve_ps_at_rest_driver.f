      PROGRAM IMPROVE_PS_AT_REST
C     *****************************************************************
C     Feeds IMPROVE_PS_POINT_PRECISION a g g > z z point boosted to the
C     rest frame of one of the two Z's, with that leg's three-momentum
C     set to exactly zero the way boost_to_frame does it, and prints
C     what comes back out.
C
C     A polarised matrix element needs the exact zero to survive: HELAS
C     reads the spin quantisation axis of a massive vector off
C     'pp.eq.rZero'. Both entry points are exercised: the double
C     precision wrapper, and the quad routine that SET_MP_PS calls on
C     the stability-escalation path.
C     *****************************************************************
      IMPLICIT NONE
      INTEGER NEXTERNAL
      PARAMETER (NEXTERNAL=4)
      DOUBLE PRECISION P(0:3,NEXTERNAL),PB(0:3,NEXTERNAL),PQ(0:3)
      DOUBLE PRECISION PW(0:3,NEXTERNAL)
      REAL*16 QW(0:3,NEXTERNAL)
      INTEGER I,J,MODE,IREST
      LOGICAL KEEP_OFFSHELL(NEXTERNAL)
      DOUBLE PRECISION MZ,EE,PP,TH,PH,SQS
      INCLUDE 'MadLoopParams.inc'
      INCLUDE 'mp_coupl.inc'

      DO I=1,NEXTERNAL
        KEEP_OFFSHELL(I)=.FALSE.
      ENDDO

      MZ = 9.11880000000000D+01
      MP__MDL_MZ = 9.11880000000000E+01_16

      SQS = 500D0
      EE = SQS/2D0
      PP = SQRT(EE**2-MZ**2)
      TH = 1.0D0
      PH = 0.7D0
      P(0,1)=EE
      P(1,1)=0D0
      P(2,1)=0D0
      P(3,1)=EE
      P(0,2)=EE
      P(1,2)=0D0
      P(2,2)=0D0
      P(3,2)=-EE
      P(0,3)=EE
      P(1,3)=PP*SIN(TH)*COS(PH)
      P(2,3)=PP*SIN(TH)*SIN(PH)
      P(3,3)=PP*COS(TH)
      P(0,4)=EE
      P(1,4)=-P(1,3)
      P(2,4)=-P(2,3)
      P(3,4)=-P(3,3)

      DO IREST=3,4
C       boost to the rest frame of leg IREST, as boost_to_frame does
        PQ(0)=P(0,IREST)
        DO J=1,3
          PQ(J)=-P(J,IREST)
        ENDDO
        DO I=1,NEXTERNAL
          CALL IPS_BOOSTX(P(0,I),PQ,PB(0,I))
        ENDDO
C       ... and then enforce the exact zero, as boost_to_frame does
        PB(1,IREST)=0D0
        PB(2,IREST)=0D0
        PB(3,IREST)=0D0

        DO MODE=1,2
          IMPROVEPSPOINT = MODE
C         entry point 1: the double precision wrapper
          DO I=1,NEXTERNAL
            DO J=0,3
              PW(J,I)=PB(J,I)
            ENDDO
          ENDDO
          CALL __PFX__IMPROVE_PS_POINT_PRECISION(KEEP_OFFSHELL,PW)
          WRITE(*,'(A,I2,I2,A,3E26.16)') 'RESULT DP ',IREST,MODE,
     &     ' ',PW(1,IREST),PW(2,IREST),PW(3,IREST)
C         entry point 2: the quad routine SET_MP_PS calls
          DO I=1,NEXTERNAL
            DO J=0,3
              QW(J,I)=PB(J,I)
            ENDDO
          ENDDO
          CALL __PFX__MP_IMPROVE_PS_POINT_PRECISION(KEEP_OFFSHELL,QW)
          WRITE(*,'(A,I2,I2,A,3E26.16)') 'RESULT QP ',IREST,MODE,
     &     ' ',DBLE(QW(1,IREST)),DBLE(QW(2,IREST)),DBLE(QW(3,IREST))
        ENDDO
      ENDDO
      END

      SUBROUTINE IPS_BOOSTX(P,Q,PBOOST)
C     same as the DHELAS boostx used by boost_to_frame
      IMPLICIT NONE
      DOUBLE PRECISION P(0:3),Q(0:3),PBOOST(0:3)
      DOUBLE PRECISION PQ,QQ,M,LF
      QQ=Q(1)**2+Q(2)**2+Q(3)**2
      IF (QQ.NE.0D0) THEN
        PQ=P(1)*Q(1)+P(2)*Q(2)+P(3)*Q(3)
        M=SQRT(Q(0)**2-QQ)
        LF=((Q(0)-M)*PQ/QQ+P(0))/M
        PBOOST(0) = (P(0)*Q(0)+PQ)/M
        PBOOST(1) = P(1)+Q(1)*LF
        PBOOST(2) = P(2)+Q(2)*LF
        PBOOST(3) = P(3)+Q(3)*LF
      ELSE
        PBOOST(0)=P(0)
        PBOOST(1)=P(1)
        PBOOST(2)=P(2)
        PBOOST(3)=P(3)
      ENDIF
      END
