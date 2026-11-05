      PROGRAM MADLOOP_LOOP_FILTER_FRAME
C     Learn filters with the W at rest, then compare a rotated and
C     boosted event against an evaluation bypassing all filters.
      IMPLICIT NONE
      INCLUDE 'coupl.inc'
      INCLUDE 'nsquaredSO.inc'
      REAL*8 P(0:3,3),Q(0:3,3),RESULT(0:3,0:NSQUAREDSO)
      REAL*8 REFERENCE(0:3,0:NSQUAREDSO),ACC(0:NSQUAREDSO)
      REAL*8 CA,SA,BZ,GZ,PY,PZ,TOL
      INTEGER I,K,RET
      LOGICAL BYPASS,ALWAYS
      COMMON/__PFX__BYPASS_CHECK/BYPASS,ALWAYS
      CALL SETPARA('../MadLoop5_resources/param_card.dat')
      P=0D0
      P(0,1)=MDL_MW/2D0
      P(3,1)=MDL_MW/2D0
      P(0,2)=MDL_MW/2D0
      P(3,2)=-MDL_MW/2D0
      P(0,3)=MDL_MW
      TOL=1D-7
      DO K=1,7
        CALL __PFX__SLOOPMATRIX_THRES(P,RESULT,TOL,ACC,RET)
      ENDDO
      CA=COS(1D0)
      SA=SIN(1D0)
      BZ=0.5D0
      GZ=1D0/SQRT(1D0-BZ**2)
      DO I=1,3
        PY=CA*P(2,I)-SA*P(3,I)
        PZ=SA*P(2,I)+CA*P(3,I)
        Q(0,I)=GZ*(P(0,I)+BZ*PZ)
        Q(1,I)=P(1,I)
        Q(2,I)=PY
        Q(3,I)=GZ*(PZ+BZ*P(0,I))
      ENDDO
      CALL __PFX__SLOOPMATRIX_THRES(Q,RESULT,TOL,ACC,RET)
      BYPASS=.TRUE.
      CALL __PFX__SLOOPMATRIX_THRES(Q,REFERENCE,TOL,ACC,RET)
      WRITE(*,'(A,6E25.16)') 'RESULT ',RESULT(1:3,0),
     $  REFERENCE(1:3,0)
      END
