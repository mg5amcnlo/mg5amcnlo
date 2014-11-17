      SUBROUTINE %(proc_prefix)sIMPROVE_PS_POINT_PRECISION(P)
	  IMPLICIT NONE
C
C CONSTANTS
C
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
C  
C ARGUMENTS 
C
      DOUBLE PRECISION P(0:3,NEXTERNAL) 
      %(real_format)s QP_P(0:3,NEXTERNAL) 
C
C LOCAL VARIABLES 
C
      INTEGER I,J

C ----------
C BEGIN CODE
C ----------

      DO I=1,NEXTERNAL
	    DO J=0,3
		  QP_P(J,I)=P(J,I)
		ENDDO
	  ENDDO

	  CALL %(proc_prefix)s%(mp_prefix)sIMPROVE_PS_POINT_PRECISION(QP_P)

	  DO I=1,NEXTERNAL
	    DO J=0,3
		  P(J,I)=QP_P(J,I)
		ENDDO
	  ENDDO

	  END


      SUBROUTINE %(proc_prefix)s%(mp_prefix)sIMPROVE_PS_POINT_PRECISION(P)
	  IMPLICIT NONE
C
C CONSTANTS
C
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
C  
C ARGUMENTS 
C
      %(real_format)s P(0:3,NEXTERNAL)
C
C LOCAL VARIABLES 
C
      INTEGER I,J
	  INTEGER ERRCODE,ERRCODETMP
	  %(real_format)s NEWP(0:3,NEXTERNAL)
C  
C FUNCTIONS
C
      LOGICAL %(proc_prefix)s%(mp_prefix)sIS_PHYSICAL
C
C SAVED VARIABLES
C
	  include 'MadLoopParams.inc'
C
C SAVED VARIABLES
C
      INTEGER WARNED
	  DATA WARNED/0/

	  LOGICAL TOLD_SUPPRESS
	  DATA TOLD_SUPPRESS/.FALSE./
C ----------
C BEGIN CODE
C ----------

C     ERROR CODES CONVENTION
C
C     1         ::  None physical PS point input
C     100-1000  ::  Error in the origianl method for restoring precision
C     1000-9999 ::  Error when restoring precision ala PSMC
C
      ERRCODETMP=0
	  ERRCODE=0

      DO J=1,NEXTERNAL
	    DO I=0,3
		  NEWP(I,J)=P(I,J)
		ENDDO
	  ENDDO

C Check the sanity of the original PS point
      IF (.NOT.%(proc_prefix)s%(mp_prefix)sIS_PHYSICAL(NEWP,WARNED)) THEN
	    ERRCODE = 1
		WRITE(*,*) 'ERROR:: The input PS point is not precise enough.'
		GOTO 100
	  ENDIF

C Now restore the precision
      IF (ImprovePSPoint.eq.1) THEN
        CALL %(proc_prefix)s%(mp_prefix)sPSMC_IMPROVE_PS_POINT_PRECISION(NEWP,ERRCODE,WARNED)	    
	  ELSEIF((ImprovePSPoint.eq.2).or.(ImprovePSPoint.le.0)) THEN
	    CALL %(proc_prefix)s%(mp_prefix)sORIG_IMPROVE_PS_POINT_PRECISION(NEWP,ERRCODE,WARNED)
	  ENDIF
	  IF (ERRCODE.NE.0) THEN
        IF (WARNED.LT.20) THEN
		  WRITE(*,*) 'INFO:: Attempting to rescue the precision improvement with an alternative method.'
		  WARNED=WARNED+1
		ENDIF
	    IF (ImprovePSPoint.eq.1) THEN
	       CALL %(proc_prefix)s%(mp_prefix)sORIG_IMPROVE_PS_POINT_PRECISION(NEWP,ERRCODETMP,WARNED)
	  	ELSEIF((ImprovePSPoint.eq.2).or.(ImprovePSPoint.le.0)) THEN
           CALL %(proc_prefix)s%(mp_prefix)sPSMC_IMPROVE_PS_POINT_PRECISION(NEWP,ERRCODETMP,WARNED)		  
        ENDIF
		IF (ERRCODETMP.NE.0) GOTO 100
	  ENDIF

C Report to the user or update the PS point.

      GOTO 101
100  CONTINUE
      IF (WARNED.LT.20) THEN
	    WRITE(*,*) 'WARNING:: This PS point could not be improved. Error code = ',ERRCODE,ERRCODETMP
		CALL %(proc_prefix)s%(mp_prefix)sWRITE_MOM(P)
	    WARNED = WARNED +1 
	  ENDIF
	  GOTO 102
101   CONTINUE
      DO J=1,NEXTERNAL
	    DO I=0,3
		  P(I,J)=NEWP(I,J)
		ENDDO
	  ENDDO
102   CONTINUE

      IF (WARNED.GE.20.AND..NOT.TOLD_SUPPRESS) THEN
	    WRITE(*,*) 'INFO:: Further warnings from the improve_ps routine will now be supressed.'
		TOLD_SUPPRESS=.TRUE.
      ENDIF

	  END
      

      FUNCTION %(proc_prefix)s%(mp_prefix)sIS_CLOSE(P,NEWP,WARNED)
	  IMPLICIT NONE
C
C CONSTANTS
C
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
	  %(real_format)s ZERO
	  PARAMETER (ZERO=0.0%(exp_letter)s+00%(mp_specifier)s)
	  %(real_format)s THRS_CLOSE
	  PARAMETER (THRS_CLOSE=1.0%(exp_letter)s-02%(mp_specifier)s)
C  
C ARGUMENTS 
C
      %(real_format)s P(0:3,NEXTERNAL), NEWP(0:3,NEXTERNAL)
	  LOGICAL %(proc_prefix)s%(mp_prefix)sIS_CLOSE
	  INTEGER WARNED
C
C LOCAL VARIABLES 
C
      INTEGER I,J
	  %(real_format)s REF,REF2
	  DOUBLE PRECISION BUFFDP

C NOW MAKE SURE THE SHIFTED POINT IS NOT TOO FAR FROM THE ORIGINAL ONE
     %(proc_prefix)s%(mp_prefix)sIS_CLOSE = .TRUE.
     REF  = ZERO
	 REF2 = ZERO
	 DO J=1,NEXTERNAL
	   DO I=0,3
	     REF2 = REF2 + ABS(P(I,J))	   
	     REF = REF + ABS(P(I,J)-NEWP(I,J)) 
	   ENDDO
	 ENDDO

	  IF ((REF/REF2).GT.THRS_CLOSE) THEN
	    %(proc_prefix)s%(mp_prefix)sIS_CLOSE = .FALSE.
	    IF (WARNED.LT.20) THEN
		  BUFFDP = (REF/REF2)
	      WRITE(*,*) 'WARNING:: The improved PS point is too far from the original one',BUFFDP
		  WARNED=WARNED+1
		ENDIF
	  ENDIF

	  END

      FUNCTION %(proc_prefix)s%(mp_prefix)sIS_PHYSICAL(P,WARNED)
	  IMPLICIT NONE
C
C CONSTANTS
C
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
	  INTEGER    NINITIAL
      PARAMETER (NINITIAL=%(ninitial)d)
	  %(real_format)s ZERO
	  PARAMETER (ZERO=0.0%(exp_letter)s+00%(mp_specifier)s)
	  %(real_format)s MP__ZERO
	  PARAMETER (MP__ZERO=ZERO)	  
	  %(real_format)s ONE
	  PARAMETER (ONE=1.0%(exp_letter)s+00%(mp_specifier)s)
	  %(real_format)s TWO
	  PARAMETER (TWO=2.0%(exp_letter)s+00%(mp_specifier)s)
	  %(real_format)s THRES_ONSHELL
	  PARAMETER (THRES_ONSHELL=1.0%(exp_letter)s-02%(mp_specifier)s)
	  %(real_format)s THRES_FOURMOM
	  PARAMETER (THRES_FOURMOM=1.0%(exp_letter)s-06%(mp_specifier)s)
C  
C ARGUMENTS 
C
      %(real_format)s P(0:3,NEXTERNAL)
	  LOGICAL %(proc_prefix)s%(mp_prefix)sIS_PHYSICAL
	  INTEGER WARNED
C
C LOCAL VARIABLES 
C
      INTEGER I,J
	  %(real_format)s BUFF,REF
	  %(real_format)s MASSES(NEXTERNAL)
	  double precision BUFFDPA,BUFFDPB
C  
C GLOBAL VARIABLES
C 
      include '%(coupl_inc_name)s'

	  %(masses_def)s

C ----------
C BEGIN CODE
C ----------

      %(proc_prefix)s%(mp_prefix)sIS_PHYSICAL = .TRUE.

C WE FIRST CHECK THAT THE INPUT PS POINT IS REASONABLY PHYSICAL
C FOR THAT WE NEED A REFERENCE SCALE
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
		    WRITE(*,*) 'ERROR:: Four-momentum conservation is not accurate enough, ',BUFFDPA
		    CALL %(proc_prefix)s%(mp_prefix)sWRITE_MOM(P)
			WARNED=WARNED+1
		  ENDIF
		  %(proc_prefix)s%(mp_prefix)sIS_PHYSICAL = .FALSE.
		ENDIF
	  ENDDO
	  REF = REF / (ONE*NEXTERNAL)
	  DO I=1,NEXTERNAL
	    REF=ABS(P(0,I))+ABS(P(1,I))+ABS(P(2,I))+ABS(P(3,I))
		IF ((SQRT(ABS(P(0,I)**2-P(1,I)**2-P(2,I)**2-P(3,I)**2-MASSES(I)**2))/REF).GT.THRES_ONSHELL) THEN
		  IF (WARNED.LT.20) THEN
		    BUFFDPA=MASSES(I)
			BUFFDPB=(SQRT(ABS(P(0,I)**2-P(1,I)**2-P(2,I)**2-P(3,I)**2-MASSES(I)**2))/REF)
		    WRITE(*,*) 'ERROR:: Onshellness of the momentum of particle ',I,' of mass ',BUFFDPA,' is not accurate enough, ',BUFFDPB
		    CALL %(proc_prefix)s%(mp_prefix)sWRITE_MOM(P)
			WARNED=WARNED+1
		  ENDIF		  
		  %(proc_prefix)s%(mp_prefix)sIS_PHYSICAL = .FALSE.
		ENDIF
	  ENDDO

	  END

	  SUBROUTINE %(proc_prefix)sWRITE_MOM(P)
	  IMPLICIT NONE
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
	  INTEGER    NINITIAL
      PARAMETER (NINITIAL=%(ninitial)d)
	  DOUBLE PRECISION ZERO
	  PARAMETER (ZERO=0.0d0)
      DOUBLE PRECISION %(proc_prefix)smdot

	  INTEGER I,J

C  
C ARGUMENTS 
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
        WRITE (*,'(1x,5e27.17)') P(0,I),P(1,I),P(2,I),P(3,I),SQRT(ABS(%(proc_prefix)sMDOT(P(0,I),P(0,I))))
      ENDDO
      WRITE (*,*) '    Four-momentum conservation sum:'
      WRITE (*,'(1x,4e27.17)') PSUM(0),PSUM(1),PSUM(2),PSUM(3)
      WRITE (*,*) '   ---------------------'
	  END

	  DOUBLE PRECISION function %(proc_prefix)smdot(p1,p2)
      implicit none
      DOUBLE PRECISION p1(0:3),p2(0:3)
      %(proc_prefix)smdot=p1(0)*p2(0)-p1(1)*p2(1)-p1(2)*p2(2)-p1(3)*p2(3)
      return
      end

	  SUBROUTINE %(proc_prefix)s%(mp_prefix)sWRITE_MOM(P)
	  IMPLICIT NONE
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
	  INTEGER    NINITIAL
      PARAMETER (NINITIAL=%(ninitial)d)
	  %(real_format)s ZERO
	  PARAMETER (ZERO=0.0%(exp_letter)s+00%(mp_specifier)s)
      %(real_format)s %(proc_prefix)s%(mp_prefix)smdot

	  INTEGER I,J

C  
C ARGUMENTS 
C
      %(real_format)s P(0:3,NEXTERNAL),PSUM(0:3),DOT
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
	  
C The GCC4.7 compiler on SLC machines has trouble to write out quadruple precision variable with the write(*,*) statement. I therefore perform the cast by hand
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
	    DOT=SQRT(ABS(%(proc_prefix)s%(mp_prefix)sMDOT(P(0,I),P(0,I))))
		DP_DOT=DOT
        WRITE (*,'(1x,5e27.17)') DP_P(0,I),DP_P(1,I),DP_P(2,I),DP_P(3,I),DP_DOT
      ENDDO
      WRITE (*,*) '    Four-momentum conservation sum:'
      WRITE (*,'(1x,4e27.17)') DP_PSUM(0),DP_PSUM(1),DP_PSUM(2),DP_PSUM(3)
      WRITE (*,*) '   ---------------------'
	  END

      %(real_format)s function %(proc_prefix)s%(mp_prefix)smdot(p1,p2)
      implicit none
      %(real_format)s p1(0:3),p2(0:3)
      %(proc_prefix)s%(mp_prefix)smdot=p1(0)*p2(0)-p1(1)*p2(1)-p1(2)*p2(2)-p1(3)*p2(3)
      return
      end

C Rotate_PS rotates the PS point PS (without modifying it)
C stores the result in P and for the quadruple precision 
c version , it also modifies the global variables
C PS and MP_DONE accordingly.

	  SUBROUTINE %(proc_prefix)sROTATE_PS(P_IN,P,ROTATION)
	  IMPLICIT NONE
C  
C CONSTANTS 
C 
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
C  
C ARGUMENTS 
C
      DOUBLE PRECISION P_IN(0:3,NEXTERNAL),P(0:3,NEXTERNAL)
	  INTEGER ROTATION
C
C LOCAL VARIABLES 
C
      INTEGER I,J

C ----------
C BEGIN CODE
C ----------

      DO I=1,NEXTERNAL
C rotation=1 => (xp=z,yp=-x,zp=-y)
		IF(ROTATION.EQ.1) THEN
		  P(0,I)=P_IN(0,I)
		  P(1,I)=P_IN(3,I)
		  P(2,I)=-P_IN(1,I)
		  P(3,I)=-P_IN(2,I)
C rotation=2 => (xp=-z,yp=y,zp=x)
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


	  SUBROUTINE %(proc_prefix)s%(mp_prefix)sROTATE_PS(P_IN,P,ROTATION)
	  IMPLICIT NONE
C  
C CONSTANTS 
C 
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
C  
C ARGUMENTS 
C
      %(real_format)s P_IN(0:3,NEXTERNAL),P(0:3,NEXTERNAL)
	  INTEGER ROTATION
C
C LOCAL VARIABLES 
C
      INTEGER I,J
C  
C GLOBAL VARIABLES
C 
	  LOGICAL MP_DONE
	  common/%(proc_prefix)sMP_DONE/MP_DONE

C ----------
C BEGIN CODE
C ----------

      DO I=1,NEXTERNAL
C rotation=1 => (xp=z,yp=-x,zp=-y)
		IF(ROTATION.EQ.1) THEN
		  P(0,I)=P_IN(0,I)
		  P(1,I)=P_IN(3,I)
		  P(2,I)=-P_IN(1,I)
		  P(3,I)=-P_IN(2,I)
C rotation=2 => (xp=-z,yp=y,zp=x)
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

C *****************************************************************
C Beginning of the routine for restoring precision with V.H. method
C *****************************************************************

      SUBROUTINE %(proc_prefix)s%(mp_prefix)sORIG_IMPROVE_PS_POINT_PRECISION(P,ERRCODE,WARNED)
	  IMPLICIT NONE
C  
C CONSTANTS 
C 
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
	  INTEGER    NINITIAL
      PARAMETER (NINITIAL=%(ninitial)d)
	  %(real_format)s ZERO
	  PARAMETER (ZERO=0.0%(exp_letter)s+00%(mp_specifier)s)
	  %(real_format)s MP__ZERO
	  PARAMETER (MP__ZERO=ZERO)	  
	  %(real_format)s ONE
	  PARAMETER (ONE=1.0%(exp_letter)s+00%(mp_specifier)s)	  
	  %(real_format)s TWO	  
	  PARAMETER (TWO=2.0%(exp_letter)s+00%(mp_specifier)s)
	  %(real_format)s THRS_TEST
	  PARAMETER (THRS_TEST=1.0%(exp_letter)s-15%(mp_specifier)s)	  
C  
C ARGUMENTS 
C
      %(real_format)s P(0:3,NEXTERNAL)
	  INTEGER ERRCODE, WARNED
C  
C FUNCTIONS
C
      LOGICAL %(proc_prefix)s%(mp_prefix)sIS_CLOSE 
C
C LOCAL VARIABLES 
C
      INTEGER I,J, P1, P2 
C     PT STANDS FOR PTOT
      %(real_format)s PT(0:3), NEWP(0:3,NEXTERNAL)
	  %(real_format)s BUFF,REF,REF2,DISCR
	  %(real_format)s MASSES(NEXTERNAL)
	  %(real_format)s SHIFTE(2),SHIFTZ(2)
C  
C GLOBAL VARIABLES
C 
      include '%(coupl_inc_name)s'

	  %(masses_def)s

C ----------
C BEGIN CODE
C ----------
	  ERRCODE = 0

C NOW WE MAKE SURE THAT THE PS POINT CAN BE IMPROVED BY THE ALGORITHM
      REF=ZERO
	  DO J=1,NEXTERNAL
		REF=REF+ABS(P(0,J))
	  ENDDO

     IF (NINITIAL.NE.2) ERRCODE = 100
	 
	 IF (ABS(P(1,1)/REF).GT.THRS_TEST.OR.ABS(P(2,1)/REF).GT.THRS_TEST.OR.ABS(P(1,2)/REF).GT.THRS_TEST.OR.ABS(P(2,2)/REF).GT.THRS_TEST) ERRCODE = 200
	   
	 IF (MASSES(1).NE.ZERO.OR.MASSES(2).NE.ZERO) ERRCODE = 300
	 
	 DO I=1,NEXTERNAL
	   IF (P(0,I).LT.ZERO) ERRCODE = 400 + I
     ENDDO

     IF (ERRCODE.NE.0) GOTO 100

C WE FIRST SHIFT ALL THE FINAL STATE PARTICLES TO MAKE THEM EXACTLY ONSHELL

      DO I=0,3
	    PT(I)=ZERO
	  ENDDO
	  DO I=NINITIAL+1,NEXTERNAL
	    DO J=0,3
		  IF (J.EQ.3) THEN
	        NEWP(3,I)=SIGN(SQRT(ABS(P(0,I)**2-P(1,I)**2-P(2,I)**2-MASSES(I)**2)),P(3,I))
		  ELSE
		    NEWP(J,I)=P(J,I)
		  ENDIF
		  PT(J)=PT(J)+NEWP(J,I)
		ENDDO
	  ENDDO

C WE CHOOSE P1 IN THE ALGORITHM TO ALWAYS BE THE PARTICLE WITH POSITIVE PZ
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

C Now we calculate the shift to bring to P1 and P2
C Mathematica gives
C ptotC = {ptotE, ptotX, ptotY, ptotZ};
C pm1C = {pm1E + sm1E, pm1X, pm1Y, pm1Z + sm1Z};
C {pm0E + sm0E, ptotX - pm1X, ptotY - pm1Y, pm0Z + sm0Z};
C sol = Solve[{ptotC[[1]] - pm1C[[1]] - pm0C[[1]] == 0,  
C  ptotC[[4]] - pm1C[[4]] - pm0C[[4]] == 0,
C  pm1C[[1]]^2 - pm1C[[2]]^2 - pm1C[[3]]^2 - pm1C[[4]]^2 == m1M^2,
C  pm0C[[1]]^2 - pm0C[[2]]^2 - pm0C[[3]]^2 - pm0C[[4]]^2 == m2M^2},
C  {sm1E, sm1Z, sm0E, sm0Z}] // FullSimplify;
C  (solC[[1]] /. {m1M -> 0, m2M -> 0} /. {pm1X -> 0, pm1Y -> 0})
C END
C
     DISCR = -PT(0)**2 + PT(1)**2 + PT(2)**2 + PT(3)**2
	 IF (DISCR.LT.ZERO) DISCR = -DISCR

	 SHIFTE(1) = (PT(0)*(-TWO*P(0,P1)*PT(0) + PT(0)**2 + PT(1)**2 + PT(2)**2) + (TWO*P(0,P1) - PT(0))*PT(3)**2 + PT(3)*DISCR)/(TWO*(PT(0) - PT(3))*(PT(0) + PT(3))) 
	 SHIFTE(2) = -(PT(0)*(TWO*P(0,P2)*PT(0) - PT(0)**2 + PT(1)**2 + PT(2)**2) + (-TWO*P(0,P2) + PT(0))*PT(3)**2 + PT(3)*DISCR)/(TWO*(PT(0) - PT(3))*(PT(0) + PT(3)))
	 SHIFTZ(1) = (-TWO*P(3,P1)*(PT(0)**2 - PT(3)**2) + PT(3)*(PT(0)**2 + PT(1)**2 + PT(2)**2 - PT(3)**2) + PT(0)*DISCR)/(TWO*(PT(0)**2 - PT(3)**2))
	 SHIFTZ(2) = -(TWO*P(3,P2)*(PT(0)**2 - PT(3)**2) + PT(3)*(-PT(0)**2 + PT(1)**2 + PT(2)**2 + PT(3)**2) + PT(0)*DISCR)/(TWO*(PT(0)**2 - PT(3)**2))
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
 
      IF (.NOT.%(proc_prefix)s%(mp_prefix)sIS_CLOSE(P,NEWP,WARNED)) THEN
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

C *****************************************************************
C Beginning of the routine for restoring precision a la PSMC
C *****************************************************************

      SUBROUTINE %(proc_prefix)s%(mp_prefix)sPSMC_IMPROVE_PS_POINT_PRECISION(P,ERRCODE,WARNED)
	  IMPLICIT NONE
C  
C CONSTANTS 
C 
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
	  INTEGER    NINITIAL
      PARAMETER (NINITIAL=%(ninitial)d)
	  %(real_format)s ZERO
	  PARAMETER (ZERO=0.0%(exp_letter)s+00%(mp_specifier)s)
	  %(real_format)s MP__ZERO
	  PARAMETER (MP__ZERO=ZERO)	  
	  %(real_format)s ONE
	  PARAMETER (ONE=1.0%(exp_letter)s+00%(mp_specifier)s)	  
	  %(real_format)s TWO	  
	  PARAMETER (TWO=2.0%(exp_letter)s+00%(mp_specifier)s)
	  %(real_format)s CONSISTENCY_THRES
	  PARAMETER (CONSISTENCY_THRES=1.0%(exp_letter)s-25%(mp_specifier)s)	

	  INTEGER    NAPPROXZEROS
      PARAMETER (NAPPROXZEROS=3)

C  
C ARGUMENTS 
C
      %(real_format)s P(0:3,NEXTERNAL)
	  INTEGER ERRCODE,ERROR,WARNED
C  
C FUNCTIONS
C
      LOGICAL %(proc_prefix)s%(mp_prefix)sIS_CLOSE
C
C LOCAL VARIABLES 
C
      INTEGER I,J, P1, P2 
      %(real_format)s NEWP(0:3,NEXTERNAL), PBUFF(0:3)
	  %(real_format)s BUFF, BUFF2, XSCALE, APPROX_ZEROS(NAPPROXZEROS)
	  %(real_format)s MASSES(NEXTERNAL)
C
C GLOBAL VARIABLES
C 
      include '%(coupl_inc_name)s'

C ----------
C BEGIN CODE
C ----------

	  %(masses_def)s

	  ERRCODE = 0
	  XSCALE = ONE

C     Define the seeds which should be tried
      APPROX_ZEROS(1)=1.0%(exp_letter)s+00%(mp_specifier)s
      APPROX_ZEROS(2)=1.1%(exp_letter)s+00%(mp_specifier)s
      APPROX_ZEROS(3)=0.9%(exp_letter)s+00%(mp_specifier)s

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
	    CALL %(proc_prefix)sFINDX(NEWP,APPROX_ZEROS(I),XSCALE,ERROR)
		IF(ERROR.EQ.0) THEN
		  GOTO 1001
		ELSE
		  ERRCODE=ERRCODE+(10**(I-1))*ERROR
		ENDIF
	  ENDDO
	  IF (WARNED.LT.20) THEN
		WRITE(*,*) 'WARNING:: Could not find the proper rescaling factor x. Restoring precision ala PSMC will therefore not be used.'
		WARNED=WARNED+1
      ENDIF
	  IF (ERRCODE.lt.1000) THEN
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
	  IF ((ABS(BUFF)/BUFF2).gt.CONSISTENCY_THRES) THEN
	    IF (WARNED.LT.20) THEN
		  WRITE(*,*) 'WARNING:: The consistency check in the a la PSMC precision restoring algorithm failed. The result will therefore not be used.'
		  WARNED=WARNED+1
		ENDIF
	    ERRCODE = 1000
		GOTO 1000
	  ENDIF

      IF (.NOT.%(proc_prefix)s%(mp_prefix)sIS_CLOSE(P,NEWP,WARNED)) THEN
	    ERRCODE=999
		GOTO 1000
	  ENDIF

	  DO J=1,NEXTERNAL
	    DO I=0,3
		  P(I,J)=NEWP(I,J)
		ENDDO
	  ENDDO
	 
 1000  CONTINUE

      END


	  SUBROUTINE %(proc_prefix)sFINDX(P,SEED,XSCALE,ERROR)
	  IMPLICIT NONE
C  
C CONSTANTS 
C 
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
	  INTEGER    NINITIAL
      PARAMETER (NINITIAL=%(ninitial)d)
	  %(real_format)s ZERO
	  PARAMETER (ZERO=0.0%(exp_letter)s+00%(mp_specifier)s)
	  %(real_format)s MP__ZERO
	  PARAMETER (MP__ZERO=ZERO)	  
	  %(real_format)s ONE
	  PARAMETER (ONE=1.0%(exp_letter)s+00%(mp_specifier)s)	  
	  %(real_format)s TWO	  
	  PARAMETER (TWO=2.0%(exp_letter)s+00%(mp_specifier)s)
	  INTEGER MAXITERATIONS
      PARAMETER (MAXITERATIONS=8)
	  %(real_format)s CONVERGED
	  PARAMETER (CONVERGED=1.0%(exp_letter)s-26%(mp_specifier)s)
C  
C ARGUMENTS 
C
      %(real_format)s P(0:3,NEXTERNAL),SEED,XSCALE
	  INTEGER ERROR
C
C LOCAL VARIABLES 
C
      INTEGER I,J,ERR
      %(real_format)s PVECSQ(NEXTERNAL)
	  %(real_format)s XN, XNP1,FVAL,DVAL

C ----------
C BEGIN CODE
C ----------

      ERROR = 0
      XSCALE = SEED
      XN = SEED
	  XNP1 = SEED

      DO I=1,NEXTERNAL
	    PVECSQ(I)=P(1,I)**2+P(2,I)**2+P(3,I)**2
	  ENDDO

	  DO I=1,MAXITERATIONS
	    CALL %(proc_prefix)sFUNCT(PVECSQ(1),XN,.FALSE.,ERR, FVAL)
		IF (ERR.NE.0) THEN
		  ERROR=ERR
		  GOTO 710
		ENDIF
	    CALL %(proc_prefix)sFUNCT(PVECSQ(1),XN,.TRUE.,ERR, DVAL)
		IF (ERR.NE.0) THEN
		  ERROR=ERR
		  GOTO 710
		ENDIF
        XNP1=XN-(FVAL/DVAL)
		IF((ABS(((XNP1-XN)*TWO)/(XNP1+XN))).lt.CONVERGED) THEN
		  XN=XNP1
		  GOTO 700
		ENDIF
		XN=XNP1
	  ENDDO
	  ERROR=9
	  GOTO 710

  700 CONTINUE
C     For good measure, we iterate one last time
	  CALL %(proc_prefix)sFUNCT(PVECSQ(1),XN,.FALSE.,ERR, FVAL)
	  IF (ERR.NE.0) THEN
		ERROR=ERR
		GOTO 710
      ENDIF
	  CALL %(proc_prefix)sFUNCT(PVECSQ(1),XN,.TRUE.,ERR, DVAL)
	  IF (ERR.NE.0) THEN
	    ERROR=ERR
		GOTO 710
      ENDIF
      
	  XSCALE=XN-(FVAL/DVAL)

  710 CONTINUE

	  END

	  SUBROUTINE %(proc_prefix)sFUNCT(PVECSQ,X,DERIVATIVE,ERROR,RES)
	  IMPLICIT NONE
C  
C CONSTANTS 
C 
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
	  INTEGER    NINITIAL
      PARAMETER (NINITIAL=%(ninitial)d)
	  %(real_format)s ZERO
	  PARAMETER (ZERO=0.0%(exp_letter)s+00%(mp_specifier)s)
	  %(real_format)s MP__ZERO
	  PARAMETER (MP__ZERO=ZERO)	  
	  %(real_format)s ONE
	  PARAMETER (ONE=1.0%(exp_letter)s+00%(mp_specifier)s)	  
	  %(real_format)s TWO	  
	  PARAMETER (TWO=2.0%(exp_letter)s+00%(mp_specifier)s)
C  
C ARGUMENTS 
C
      %(real_format)s PVECSQ(NEXTERNAL),X,RES
	  INTEGER ERROR
	  LOGICAL DERIVATIVE
C
C LOCAL VARIABLES 
C
      INTEGER I,J 
	  %(real_format)s BUFF,FACTOR
	  %(real_format)s MASSES(NEXTERNAL)
C
C GLOBAL VARIABLES
C 
      include '%(coupl_inc_name)s'

C ----------
C BEGIN CODE
C ----------

	  %(masses_def)s

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

  800 CONTINUE

	  END
