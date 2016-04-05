      SUBROUTINE IMPROVE_PS_POINT_PRECISION(P)
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=3)
C     
C     ARGUMENTS 
C     
      DOUBLE PRECISION P(0:3,NEXTERNAL)
      REAL*16 QP_P(0:3,NEXTERNAL)
C     
C     LOCAL VARIABLES 
C     
      INTEGER I,J

C     ----------
C     BEGIN CODE
C     ----------

      DO I=1,NEXTERNAL
        DO J=0,3
          QP_P(J,I)=P(J,I)
        ENDDO
      ENDDO

      CALL MP_IMPROVE_PS_POINT_PRECISION(QP_P)

      DO I=1,NEXTERNAL
        DO J=0,3
          P(J,I)=QP_P(J,I)
        ENDDO
      ENDDO

      END


      SUBROUTINE MP_IMPROVE_PS_POINT_PRECISION(P)
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=3)
C     
C     ARGUMENTS 
C     
      REAL*16 P(0:3,NEXTERNAL)
C     
C     LOCAL VARIABLES 
C     
      INTEGER I,J
      INTEGER ERRCODE,ERRCODETMP
      REAL*16 NEWP(0:3,NEXTERNAL)
C     
C     FUNCTIONS
C     
      LOGICAL MP_IS_PHYSICAL
C     
C     SAVED VARIABLES
C     
      INCLUDE 'MadLoopParams.inc'
C     
C     SAVED VARIABLES
C     
      INTEGER WARNED
      DATA WARNED/0/

      LOGICAL TOLD_SUPPRESS
      DATA TOLD_SUPPRESS/.FALSE./
C     ----------
C     BEGIN CODE
C     ----------

C     ERROR CODES CONVENTION
C     
C     1         ::  None physical PS point input
C     100-1000  ::  Error in the origianl method for restoring
C      precision
C     1000-9999 ::  Error when restoring precision ala PSMC
C     
      ERRCODETMP=0
      ERRCODE=0

      DO J=1,NEXTERNAL
        DO I=0,3
          NEWP(I,J)=P(I,J)
        ENDDO
      ENDDO

C     Check the sanity of the original PS point
      IF (.NOT.MP_IS_PHYSICAL(NEWP,WARNED)) THEN
        ERRCODE = 1
        WRITE(*,*) 'ERROR:: The input PS point is not precise enough.'
        GOTO 100
      ENDIF

C     Now restore the precision
      IF (IMPROVEPSPOINT.EQ.1) THEN
        CALL MP_PSMC_IMPROVE_PS_POINT_PRECISION(NEWP,ERRCODE,WARNED)
      ELSEIF((IMPROVEPSPOINT.EQ.2).OR.(IMPROVEPSPOINT.LE.0)) THEN
        CALL MP_ORIG_IMPROVE_PS_POINT_PRECISION(NEWP,ERRCODE,WARNED)
      ENDIF
      IF (ERRCODE.NE.0) THEN
        IF (WARNED.LT.20) THEN
          WRITE(*,*) 'INFO:: Attempting to rescue the precision'
     $     //' improvement with an alternative method.'
          WARNED=WARNED+1
        ENDIF
        IF (IMPROVEPSPOINT.EQ.1) THEN
          CALL MP_ORIG_IMPROVE_PS_POINT_PRECISION(NEWP,ERRCODETMP
     $     ,WARNED)
        ELSEIF((IMPROVEPSPOINT.EQ.2).OR.(IMPROVEPSPOINT.LE.0)) THEN
          CALL MP_PSMC_IMPROVE_PS_POINT_PRECISION(NEWP,ERRCODETMP
     $     ,WARNED)
        ENDIF
        IF (ERRCODETMP.NE.0) GOTO 100
      ENDIF

C     Report to the user or update the PS point.

      GOTO 101
 100  CONTINUE
      IF (WARNED.LT.20) THEN
        WRITE(*,*) 'WARNING:: This PS point could not be improved.'
     $   //' Error code = ',ERRCODE,ERRCODETMP
        CALL MP_WRITE_MOM(P)
        WARNED = WARNED +1
      ENDIF
      GOTO 102
 101  CONTINUE
      DO J=1,NEXTERNAL
        DO I=0,3
          P(I,J)=NEWP(I,J)
        ENDDO
      ENDDO
 102  CONTINUE

      IF (WARNED.GE.20.AND..NOT.TOLD_SUPPRESS) THEN
        WRITE(*,*) 'INFO:: Further warnings from the improve_ps'
     $   //' routine will now be supressed.'
        TOLD_SUPPRESS=.TRUE.
      ENDIF

      END


      FUNCTION MP_IS_CLOSE(P,NEWP,WARNED)
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=3)
      REAL*16 ZERO
      PARAMETER (ZERO=0.0E+00_16)
      REAL*16 THRS_CLOSE
      PARAMETER (THRS_CLOSE=1.0E-02_16)
C     
C     ARGUMENTS 
C     
      REAL*16 P(0:3,NEXTERNAL), NEWP(0:3,NEXTERNAL)
      LOGICAL MP_IS_CLOSE
      INTEGER WARNED
C     
C     LOCAL VARIABLES 
C     
      INTEGER I,J
      REAL*16 REF,REF2
      DOUBLE PRECISION BUFFDP

C     NOW MAKE SURE THE SHIFTED POINT IS NOT TOO FAR FROM THE ORIGINAL
C      ONE
      MP_IS_CLOSE = .TRUE.
      REF  = ZERO
      REF2 = ZERO
      DO J=1,NEXTERNAL
        DO I=0,3
          REF2 = REF2 + ABS(P(I,J))
          REF = REF + ABS(P(I,J)-NEWP(I,J))
        ENDDO
      ENDDO

      IF ((REF/REF2).GT.THRS_CLOSE) THEN
        MP_IS_CLOSE = .FALSE.
        IF (WARNED.LT.20) THEN
          BUFFDP = (REF/REF2)
          WRITE(*,*) 'WARNING:: The improved PS point is too far from'
     $     //' the original one',BUFFDP
          WARNED=WARNED+1
        ENDIF
      ENDIF

      END

      FUNCTION MP_IS_PHYSICAL(P,WARNED)
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=3)
      INTEGER    NINITIAL
      PARAMETER (NINITIAL=2)
      REAL*16 ZERO
      PARAMETER (ZERO=0.0E+00_16)
      REAL*16 MP__ZERO
      PARAMETER (MP__ZERO=ZERO)
      REAL*16 ONE
      PARAMETER (ONE=1.0E+00_16)
      REAL*16 TWO
      PARAMETER (TWO=2.0E+00_16)
      REAL*16 THRES_ONSHELL
      PARAMETER (THRES_ONSHELL=1.0E-02_16)
      REAL*16 THRES_FOURMOM
      PARAMETER (THRES_FOURMOM=1.0E-06_16)
C     
C     ARGUMENTS 
C     
      REAL*16 P(0:3,NEXTERNAL)
      LOGICAL MP_IS_PHYSICAL
      INTEGER WARNED
C     
C     LOCAL VARIABLES 
C     
      INTEGER I,J
      REAL*16 BUFF,REF
      REAL*16 MASSES(NEXTERNAL)
      DOUBLE PRECISION BUFFDPA,BUFFDPB
C     
C     GLOBAL VARIABLES
C     
      INCLUDE 'mp_coupl.inc'

      MASSES(1)=MP__ZERO
      MASSES(2)=MP__ZERO
      MASSES(3)=MP__MDL_MW

C     ----------
C     BEGIN CODE
C     ----------

      MP_IS_PHYSICAL = .TRUE.

C     WE FIRST CHECK THAT THE INPUT PS POINT IS REASONABLY PHYSICAL
C     FOR THAT WE NEED A REFERENCE SCALE
      REF=ZERO
      DO J=1,NEXTERNAL
        REF=REF+ABS(P(0,J))
      ENDDO
      DO I=0,3
        BUFF=ZERO
        DO J=1,NINITIAL
          BUFF=BUFF-P(I,J)
        ENDDO
        DO J=NINITIAL+1,NEXTERNAL
          BUFF=BUFF+P(I,J)
        ENDDO
        IF ((BUFF/REF).GT.THRES_FOURMOM) THEN
          IF (WARNED.LT.20) THEN
            BUFFDPA = (BUFF/REF)
            WRITE(*,*) 'ERROR:: Four-momentum conservation is not'
     $       //' accurate enough, ',BUFFDPA
            CALL MP_WRITE_MOM(P)
            WARNED=WARNED+1
          ENDIF
          MP_IS_PHYSICAL = .FALSE.
        ENDIF
      ENDDO
      REF = REF / (ONE*NEXTERNAL)
      DO I=1,NEXTERNAL
        REF=ABS(P(0,I))+ABS(P(1,I))+ABS(P(2,I))+ABS(P(3,I))
        IF ((SQRT(ABS(P(0,I)**2-P(1,I)**2-P(2,I)**2-P(3,I)**2-MASSES(I)
     $   **2))/REF).GT.THRES_ONSHELL) THEN
          IF (WARNED.LT.20) THEN
            BUFFDPA=MASSES(I)
            BUFFDPB=(SQRT(ABS(P(0,I)**2-P(1,I)**2-P(2,I)**2-P(3,I)**2
     $       -MASSES(I)**2))/REF)
            WRITE(*,*) 'ERROR:: Onshellness of the momentum of'
     $       //' particle ',I,' of mass ',BUFFDPA,' is not accurate'
     $       //' enough, ',BUFFDPB
            CALL MP_WRITE_MOM(P)
            WARNED=WARNED+1
          ENDIF
          MP_IS_PHYSICAL = .FALSE.
        ENDIF
      ENDDO

      END

      SUBROUTINE WRITE_MOM(P)
      IMPLICIT NONE
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=3)
      INTEGER    NINITIAL
      PARAMETER (NINITIAL=2)
      DOUBLE PRECISION ZERO
      PARAMETER (ZERO=0.0D0)
      DOUBLE PRECISION MDOT

      INTEGER I,J

C     
C     ARGUMENTS 
C     
      DOUBLE PRECISION P(0:3,NEXTERNAL),PSUM(0:3)
      DO I=0,3
        PSUM(I)=ZERO
        DO J=1,NINITIAL
          PSUM(I)=PSUM(I)+P(I,J)
        ENDDO
        DO J=NINITIAL+1,NEXTERNAL
          PSUM(I)=PSUM(I)-P(I,J)
        ENDDO
      ENDDO
      WRITE (*,*) ' Phase space point:'
      WRITE (*,*) '    ---------------------'
      WRITE (*,*) '    E | px | py | pz | m '
      DO I=1,NEXTERNAL
        WRITE (*,'(1x,5e27.17)') P(0,I),P(1,I),P(2,I),P(3,I)
     $   ,SQRT(ABS(MDOT(P(0,I),P(0,I))))
      ENDDO
      WRITE (*,*) '    Four-momentum conservation sum:'
      WRITE (*,'(1x,4e27.17)') PSUM(0),PSUM(1),PSUM(2),PSUM(3)
      WRITE (*,*) '   ---------------------'
      END

      DOUBLE PRECISION FUNCTION MDOT(P1,P2)
      IMPLICIT NONE
      DOUBLE PRECISION P1(0:3),P2(0:3)
      MDOT=P1(0)*P2(0)-P1(1)*P2(1)-P1(2)*P2(2)-P1(3)*P2(3)
      RETURN
      END

      SUBROUTINE MP_WRITE_MOM(P)
      IMPLICIT NONE
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=3)
      INTEGER    NINITIAL
      PARAMETER (NINITIAL=2)
      REAL*16 ZERO
      PARAMETER (ZERO=0.0E+00_16)
      REAL*16 MP_MDOT

      INTEGER I,J

C     
C     ARGUMENTS 
C     
      REAL*16 P(0:3,NEXTERNAL),PSUM(0:3),DOT
      DOUBLE PRECISION DP_P(0:3,NEXTERNAL),DP_PSUM(0:3),DP_DOT

      DO I=0,3
        PSUM(I)=ZERO
        DO J=1,NINITIAL
          PSUM(I)=PSUM(I)+P(I,J)
        ENDDO
        DO J=NINITIAL+1,NEXTERNAL
          PSUM(I)=PSUM(I)-P(I,J)
        ENDDO
      ENDDO

C     The GCC4.7 compiler on SLC machines has trouble to write out
C      quadruple precision variable with the write(*,*) statement. I
C      therefore perform the cast by hand
      DO I=0,3
        DP_PSUM(I)=PSUM(I)
        DO J=1,NEXTERNAL
          DP_P(I,J)=P(I,J)
        ENDDO
      ENDDO

      WRITE (*,*) ' Phase space point:'
      WRITE (*,*) '    ---------------------'
      WRITE (*,*) '    E | px | py | pz | m '
      DO I=1,NEXTERNAL
        DOT=SQRT(ABS(MP_MDOT(P(0,I),P(0,I))))
        DP_DOT=DOT
        WRITE (*,'(1x,5e27.17)') DP_P(0,I),DP_P(1,I),DP_P(2,I),DP_P(3
     $   ,I),DP_DOT
      ENDDO
      WRITE (*,*) '    Four-momentum conservation sum:'
      WRITE (*,'(1x,4e27.17)') DP_PSUM(0),DP_PSUM(1),DP_PSUM(2)
     $ ,DP_PSUM(3)
      WRITE (*,*) '   ---------------------'
      END

      REAL*16 FUNCTION MP_MDOT(P1,P2)
      IMPLICIT NONE
      REAL*16 P1(0:3),P2(0:3)
      MP_MDOT=P1(0)*P2(0)-P1(1)*P2(1)-P1(2)*P2(2)-P1(3)*P2(3)
      RETURN
      END

C     Rotate_PS rotates the PS point PS (without modifying it)
C     stores the result in P and for the quadruple precision 
C     version , it also modifies the global variables
C     PS and MP_DONE accordingly.

      SUBROUTINE ROTATE_PS(P_IN,P,ROTATION)
      IMPLICIT NONE
C     
C     CONSTANTS 
C     
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=3)
C     
C     ARGUMENTS 
C     
      DOUBLE PRECISION P_IN(0:3,NEXTERNAL),P(0:3,NEXTERNAL)
      INTEGER ROTATION
C     
C     LOCAL VARIABLES 
C     
      INTEGER I,J

C     ----------
C     BEGIN CODE
C     ----------

      DO I=1,NEXTERNAL
C       rotation=1 => (xp=z,yp=-x,zp=-y)
        IF(ROTATION.EQ.1) THEN
          P(0,I)=P_IN(0,I)
          P(1,I)=P_IN(3,I)
          P(2,I)=-P_IN(1,I)
          P(3,I)=-P_IN(2,I)
C         rotation=2 => (xp=-z,yp=y,zp=x)
        ELSEIF(ROTATION.EQ.2) THEN
          P(0,I)=P_IN(0,I)
          P(1,I)=-P_IN(3,I)
          P(2,I)=P_IN(2,I)
          P(3,I)=P_IN(1,I)
        ELSE
          P(0,I)=P_IN(0,I)
          P(1,I)=P_IN(1,I)
          P(2,I)=P_IN(2,I)
          P(3,I)=P_IN(3,I)
        ENDIF
      ENDDO

      END


      SUBROUTINE MP_ROTATE_PS(P_IN,P,ROTATION)
      IMPLICIT NONE
C     
C     CONSTANTS 
C     
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=3)
C     
C     ARGUMENTS 
C     
      REAL*16 P_IN(0:3,NEXTERNAL),P(0:3,NEXTERNAL)
      INTEGER ROTATION
C     
C     LOCAL VARIABLES 
C     
      INTEGER I,J
C     
C     GLOBAL VARIABLES
C     
      LOGICAL MP_DONE
      COMMON/MP_DONE/MP_DONE

C     ----------
C     BEGIN CODE
C     ----------

      DO I=1,NEXTERNAL
C       rotation=1 => (xp=z,yp=-x,zp=-y)
        IF(ROTATION.EQ.1) THEN
          P(0,I)=P_IN(0,I)
          P(1,I)=P_IN(3,I)
          P(2,I)=-P_IN(1,I)
          P(3,I)=-P_IN(2,I)
C         rotation=2 => (xp=-z,yp=y,zp=x)
        ELSEIF(ROTATION.EQ.2) THEN
          P(0,I)=P_IN(0,I)
          P(1,I)=-P_IN(3,I)
          P(2,I)=P_IN(2,I)
          P(3,I)=P_IN(1,I)
        ELSE
          P(0,I)=P_IN(0,I)
          P(1,I)=P_IN(1,I)
          P(2,I)=P_IN(2,I)
          P(3,I)=P_IN(3,I)
        ENDIF
      ENDDO

      MP_DONE = .FALSE.

      END

C     *****************************************************************
C     Beginning of the routine for restoring precision with V.H. method
C     *****************************************************************

      SUBROUTINE MP_ORIG_IMPROVE_PS_POINT_PRECISION(P,ERRCODE,WARNED)
      IMPLICIT NONE
C     
C     CONSTANTS 
C     
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=3)
      INTEGER    NINITIAL
      PARAMETER (NINITIAL=2)
      REAL*16 ZERO
      PARAMETER (ZERO=0.0E+00_16)
      REAL*16 MP__ZERO
      PARAMETER (MP__ZERO=ZERO)
      REAL*16 ONE
      PARAMETER (ONE=1.0E+00_16)
      REAL*16 TWO
      PARAMETER (TWO=2.0E+00_16)
      REAL*16 THRS_TEST
      PARAMETER (THRS_TEST=1.0E-15_16)
C     
C     ARGUMENTS 
C     
      REAL*16 P(0:3,NEXTERNAL)
      INTEGER ERRCODE, WARNED
C     
C     FUNCTIONS
C     
      LOGICAL MP_IS_CLOSE
C     
C     LOCAL VARIABLES 
C     
      INTEGER I,J, P1, P2
C     PT STANDS FOR PTOT
      REAL*16 PT(0:3), NEWP(0:3,NEXTERNAL)
      REAL*16 BUFF,REF,REF2,DISCR
      REAL*16 MASSES(NEXTERNAL)
      REAL*16 SHIFTE(2),SHIFTZ(2)
C     
C     GLOBAL VARIABLES
C     
      INCLUDE 'mp_coupl.inc'

      MASSES(1)=MP__ZERO
      MASSES(2)=MP__ZERO
      MASSES(3)=MP__MDL_MW

C     ----------
C     BEGIN CODE
C     ----------
      ERRCODE = 0

C     NOW WE MAKE SURE THAT THE PS POINT CAN BE IMPROVED BY THE
C      ALGORITHM
      REF=ZERO
      DO J=1,NEXTERNAL
        REF=REF+ABS(P(0,J))
      ENDDO

      IF (NINITIAL.NE.2) ERRCODE = 100

      IF (ABS(P(1,1)/REF).GT.THRS_TEST.OR.ABS(P(2,1)/REF)
     $ .GT.THRS_TEST.OR.ABS(P(1,2)/REF).GT.THRS_TEST.OR.ABS(P(2,2)/REF)
     $ .GT.THRS_TEST) ERRCODE = 200

      IF (MASSES(1).NE.ZERO.OR.MASSES(2).NE.ZERO) ERRCODE = 300

      DO I=1,NEXTERNAL
        IF (P(0,I).LT.ZERO) ERRCODE = 400 + I
      ENDDO

      IF (ERRCODE.NE.0) GOTO 100

C     WE FIRST SHIFT ALL THE FINAL STATE PARTICLES TO MAKE THEM
C      EXACTLY ONSHELL

      DO I=0,3
        PT(I)=ZERO
      ENDDO
      DO I=NINITIAL+1,NEXTERNAL
        DO J=0,3
          IF (J.EQ.3) THEN
            NEWP(3,I)=SIGN(SQRT(ABS(P(0,I)**2-P(1,I)**2-P(2,I)**2
     $       -MASSES(I)**2)),P(3,I))
          ELSE
            NEWP(J,I)=P(J,I)
          ENDIF
          PT(J)=PT(J)+NEWP(J,I)
        ENDDO
      ENDDO

C     WE CHOOSE P1 IN THE ALGORITHM TO ALWAYS BE THE PARTICLE WITH
C      POSITIVE PZ
      IF (P(3,1).GT.ZERO) THEN
        P1=1
        P2=2
      ELSEIF (P(3,2).GT.ZERO) THEN
        P1=2
        P2=1
      ELSE
        ERRCODE = 500
        GOTO 100
      ENDIF

C     Now we calculate the shift to bring to P1 and P2
C     Mathematica gives
C     ptotC = {ptotE, ptotX, ptotY, ptotZ};
C     pm1C = {pm1E + sm1E, pm1X, pm1Y, pm1Z + sm1Z};
C     {pm0E + sm0E, ptotX - pm1X, ptotY - pm1Y, pm0Z + sm0Z};
C     sol = Solve[{ptotC[[1]] - pm1C[[1]] - pm0C[[1]] == 0,  
C     ptotC[[4]] - pm1C[[4]] - pm0C[[4]] == 0,
C     pm1C[[1]]^2 - pm1C[[2]]^2 - pm1C[[3]]^2 - pm1C[[4]]^2 == m1M^2,
C     pm0C[[1]]^2 - pm0C[[2]]^2 - pm0C[[3]]^2 - pm0C[[4]]^2 == m2M^2},
C     {sm1E, sm1Z, sm0E, sm0Z}] // FullSimplify;
C     (solC[[1]] /. {m1M -> 0, m2M -> 0} /. {pm1X -> 0, pm1Y -> 0})
C     END
C     
      DISCR = -PT(0)**2 + PT(1)**2 + PT(2)**2 + PT(3)**2
      IF (DISCR.LT.ZERO) DISCR = -DISCR

      SHIFTE(1) = (PT(0)*(-TWO*P(0,P1)*PT(0) + PT(0)**2 + PT(1)**2 +
     $  PT(2)**2) + (TWO*P(0,P1) - PT(0))*PT(3)**2 + PT(3)*DISCR)/(TWO
     $ *(PT(0) - PT(3))*(PT(0) + PT(3)))
      SHIFTE(2) = -(PT(0)*(TWO*P(0,P2)*PT(0) - PT(0)**2 + PT(1)**2 +
     $  PT(2)**2) + (-TWO*P(0,P2) + PT(0))*PT(3)**2 + PT(3)*DISCR)
     $ /(TWO*(PT(0) - PT(3))*(PT(0) + PT(3)))
      SHIFTZ(1) = (-TWO*P(3,P1)*(PT(0)**2 - PT(3)**2) + PT(3)*(PT(0)*
     $ *2 + PT(1)**2 + PT(2)**2 - PT(3)**2) + PT(0)*DISCR)/(TWO*(PT(0)
     $ **2 - PT(3)**2))
      SHIFTZ(2) = -(TWO*P(3,P2)*(PT(0)**2 - PT(3)**2) + PT(3)*(-PT(0)*
     $ *2 + PT(1)**2 + PT(2)**2 + PT(3)**2) + PT(0)*DISCR)/(TWO*(PT(0)
     $ **2 - PT(3)**2))
      NEWP(0,P1) = P(0,P1)+SHIFTE(1)
      NEWP(3,P1) = P(3,P1)+SHIFTZ(1)
      NEWP(0,P2) = P(0,P2)+SHIFTE(2)
      NEWP(3,P2) = P(3,P2)+SHIFTZ(2)
      NEWP(1,P2) = P(1,P2)
      NEWP(2,P2) = P(2,P2)
      DO J=1,2
        REF=ZERO
        DO I=NINITIAL+1,NEXTERNAL
          REF = REF + P(J,I)
        ENDDO
        REF = REF - P(J,P2)
        NEWP(J,P1) = REF
      ENDDO

      IF (.NOT.MP_IS_CLOSE(P,NEWP,WARNED)) THEN
        ERRCODE=999
        GOTO 100
      ENDIF

      DO J=1,NEXTERNAL
        DO I=0,3
          P(I,J)=NEWP(I,J)
        ENDDO
      ENDDO

 100  CONTINUE

      END

C     *****************************************************************
C     Beginning of the routine for restoring precision a la PSMC
C     *****************************************************************

      SUBROUTINE MP_PSMC_IMPROVE_PS_POINT_PRECISION(P,ERRCODE,WARNED)
      IMPLICIT NONE
C     
C     CONSTANTS 
C     
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=3)
      INTEGER    NINITIAL
      PARAMETER (NINITIAL=2)
      REAL*16 ZERO
      PARAMETER (ZERO=0.0E+00_16)
      REAL*16 MP__ZERO
      PARAMETER (MP__ZERO=ZERO)
      REAL*16 ONE
      PARAMETER (ONE=1.0E+00_16)
      REAL*16 TWO
      PARAMETER (TWO=2.0E+00_16)
      REAL*16 CONSISTENCY_THRES
      PARAMETER (CONSISTENCY_THRES=1.0E-25_16)

      INTEGER    NAPPROXZEROS
      PARAMETER (NAPPROXZEROS=3)

C     
C     ARGUMENTS 
C     
      REAL*16 P(0:3,NEXTERNAL)
      INTEGER ERRCODE,ERROR,WARNED
C     
C     FUNCTIONS
C     
      LOGICAL MP_IS_CLOSE
C     
C     LOCAL VARIABLES 
C     
      INTEGER I,J, P1, P2
      REAL*16 NEWP(0:3,NEXTERNAL), PBUFF(0:3)
      REAL*16 BUFF, BUFF2, XSCALE, APPROX_ZEROS(NAPPROXZEROS)
      REAL*16 MASSES(NEXTERNAL)
C     
C     GLOBAL VARIABLES
C     
      INCLUDE 'mp_coupl.inc'

C     ----------
C     BEGIN CODE
C     ----------

      MASSES(1)=MP__ZERO
      MASSES(2)=MP__ZERO
      MASSES(3)=MP__MDL_MW

      ERRCODE = 0
      XSCALE = ONE

C     Define the seeds which should be tried
      APPROX_ZEROS(1)=1.0E+00_16
      APPROX_ZEROS(2)=1.1E+00_16
      APPROX_ZEROS(3)=0.9E+00_16

C     Start by copying the momenta
      DO I=1,NEXTERNAL
        DO J=0,3
          NEWP(J,I)=P(J,I)
        ENDDO
      ENDDO

C     First make sur that the space like momentum is exactly conserved
      DO J=0,3
        PBUFF(J)=ZERO
      ENDDO
      DO I=1,NINITIAL
        DO J=1,3
          PBUFF(J)=PBUFF(J)+NEWP(J,I)
        ENDDO
      ENDDO
      DO I=NINITIAL+1,NEXTERNAL-1
        DO J=1,3
          PBUFF(J)=PBUFF(J)-NEWP(J,I)
        ENDDO
      ENDDO
      DO J=1,3
        NEWP(J,NEXTERNAL)=PBUFF(J)
      ENDDO

C     Now find the 'x' rescaling factor
      DO I=1,NAPPROXZEROS
        CALL FINDX(NEWP,APPROX_ZEROS(I),XSCALE,ERROR)
        IF(ERROR.EQ.0) THEN
          GOTO 1001
        ELSE
          ERRCODE=ERRCODE+(10**(I-1))*ERROR
        ENDIF
      ENDDO
      IF (WARNED.LT.20) THEN
        WRITE(*,*) 'WARNING:: Could not find the proper rescaling'
     $   //' factor x. Restoring precision ala PSMC will therefore not'
     $   //' be used.'
        WARNED=WARNED+1
      ENDIF
      IF (ERRCODE.LT.1000) THEN
        ERRCODE=ERRCODE+1000
      ENDIF
      GOTO 1000
 1001 CONTINUE
      ERRCODE = 0

C     Apply the rescaling
      DO I=1,NEXTERNAL
        DO J=1,3
          NEWP(J,I)=NEWP(J,I)*XSCALE
        ENDDO
      ENDDO

C     Now restore exact onshellness of the particles.
      DO I=1,NEXTERNAL
        BUFF=MASSES(I)**2
        DO J=1,3
          BUFF=BUFF+NEWP(J,I)**2
        ENDDO
        NEWP(0,I)=SQRT(BUFF)
      ENDDO

C     Consistency check
      BUFF=ZERO
      BUFF2=ZERO
      DO I=1,NINITIAL
        BUFF=BUFF-NEWP(0,I)
        BUFF2=BUFF2+NEWP(0,I)
      ENDDO
      DO I=NINITIAL+1,NEXTERNAL
        BUFF=BUFF+NEWP(0,I)
        BUFF2=BUFF2+NEWP(0,I)
      ENDDO
      IF ((ABS(BUFF)/BUFF2).GT.CONSISTENCY_THRES) THEN
        IF (WARNED.LT.20) THEN
          WRITE(*,*) 'WARNING:: The consistency check in the a la PSMC'
     $     //' precision restoring algorithm failed. The result will'
     $     //' therefore not be used.'
          WARNED=WARNED+1
        ENDIF
        ERRCODE = 1000
        GOTO 1000
      ENDIF

      IF (.NOT.MP_IS_CLOSE(P,NEWP,WARNED)) THEN
        ERRCODE=999
        GOTO 1000
      ENDIF

      DO J=1,NEXTERNAL
        DO I=0,3
          P(I,J)=NEWP(I,J)
        ENDDO
      ENDDO

 1000 CONTINUE

      END


      SUBROUTINE FINDX(P,SEED,XSCALE,ERROR)
      IMPLICIT NONE
C     
C     CONSTANTS 
C     
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=3)
      INTEGER    NINITIAL
      PARAMETER (NINITIAL=2)
      REAL*16 ZERO
      PARAMETER (ZERO=0.0E+00_16)
      REAL*16 MP__ZERO
      PARAMETER (MP__ZERO=ZERO)
      REAL*16 ONE
      PARAMETER (ONE=1.0E+00_16)
      REAL*16 TWO
      PARAMETER (TWO=2.0E+00_16)
      INTEGER MAXITERATIONS
      PARAMETER (MAXITERATIONS=8)
      REAL*16 CONVERGED
      PARAMETER (CONVERGED=1.0E-26_16)
C     
C     ARGUMENTS 
C     
      REAL*16 P(0:3,NEXTERNAL),SEED,XSCALE
      INTEGER ERROR
C     
C     LOCAL VARIABLES 
C     
      INTEGER I,J,ERR
      REAL*16 PVECSQ(NEXTERNAL)
      REAL*16 XN, XNP1,FVAL,DVAL

C     ----------
C     BEGIN CODE
C     ----------

      ERROR = 0
      XSCALE = SEED
      XN = SEED
      XNP1 = SEED

      DO I=1,NEXTERNAL
        PVECSQ(I)=P(1,I)**2+P(2,I)**2+P(3,I)**2
      ENDDO

      DO I=1,MAXITERATIONS
        CALL FUNCT(PVECSQ(1),XN,.FALSE.,ERR, FVAL)
        IF (ERR.NE.0) THEN
          ERROR=ERR
          GOTO 710
        ENDIF
        CALL FUNCT(PVECSQ(1),XN,.TRUE.,ERR, DVAL)
        IF (ERR.NE.0) THEN
          ERROR=ERR
          GOTO 710
        ENDIF
        XNP1=XN-(FVAL/DVAL)
        IF((ABS(((XNP1-XN)*TWO)/(XNP1+XN))).LT.CONVERGED) THEN
          XN=XNP1
          GOTO 700
        ENDIF
        XN=XNP1
      ENDDO
      ERROR=9
      GOTO 710

 700  CONTINUE
C     For good measure, we iterate one last time
      CALL FUNCT(PVECSQ(1),XN,.FALSE.,ERR, FVAL)
      IF (ERR.NE.0) THEN
        ERROR=ERR
        GOTO 710
      ENDIF
      CALL FUNCT(PVECSQ(1),XN,.TRUE.,ERR, DVAL)
      IF (ERR.NE.0) THEN
        ERROR=ERR
        GOTO 710
      ENDIF

      XSCALE=XN-(FVAL/DVAL)

 710  CONTINUE

      END

      SUBROUTINE FUNCT(PVECSQ,X,DERIVATIVE,ERROR,RES)
      IMPLICIT NONE
C     
C     CONSTANTS 
C     
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=3)
      INTEGER    NINITIAL
      PARAMETER (NINITIAL=2)
      REAL*16 ZERO
      PARAMETER (ZERO=0.0E+00_16)
      REAL*16 MP__ZERO
      PARAMETER (MP__ZERO=ZERO)
      REAL*16 ONE
      PARAMETER (ONE=1.0E+00_16)
      REAL*16 TWO
      PARAMETER (TWO=2.0E+00_16)
C     
C     ARGUMENTS 
C     
      REAL*16 PVECSQ(NEXTERNAL),X,RES
      INTEGER ERROR
      LOGICAL DERIVATIVE
C     
C     LOCAL VARIABLES 
C     
      INTEGER I,J
      REAL*16 BUFF,FACTOR
      REAL*16 MASSES(NEXTERNAL)
C     
C     GLOBAL VARIABLES
C     
      INCLUDE 'mp_coupl.inc'

C     ----------
C     BEGIN CODE
C     ----------

      MASSES(1)=MP__ZERO
      MASSES(2)=MP__ZERO
      MASSES(3)=MP__MDL_MW

      ERROR=0
      RES=ZERO
      BUFF=ZERO

      DO I=1,NEXTERNAL
        IF (I.LE.NINITIAL) THEN
          FACTOR=-ONE
        ELSE
          FACTOR=ONE
        ENDIF
        BUFF=MASSES(I)**2+PVECSQ(I)*X**2
        IF (BUFF.LT.ZERO) THEN
          RES=ZERO
          ERROR = 1
          GOTO 800
        ENDIF
        IF (DERIVATIVE) THEN
          RES=RES + FACTOR*((X*PVECSQ(I))/SQRT(BUFF))
        ELSE
          RES=RES + FACTOR*SQRT(BUFF)
        ENDIF
      ENDDO

 800  CONTINUE

      END

