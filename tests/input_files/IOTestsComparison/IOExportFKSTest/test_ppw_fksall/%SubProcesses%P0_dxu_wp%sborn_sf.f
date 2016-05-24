      SUBROUTINE SBORN_SF(P_BORN,M,N,WGT)
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      DOUBLE PRECISION P_BORN(0:3,NEXTERNAL-1),WGT
      DOUBLE COMPLEX WGT1(2)
      INTEGER M,N

C     b_sf_001 links partons 1 and 2 
      IF ((M.EQ.1 .AND. N.EQ.2).OR.(M.EQ.2 .AND. N.EQ.1)) THEN
        CALL SB_SF_001(P_BORN,WGT)

      ELSE
        WGT = 0D0
      ENDIF

      RETURN
      END
