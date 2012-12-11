      SUBROUTINE %(mp_prefix)sIMPROVE_PS_POINT_PRECISION(P)
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
	  %(real_format)s THRS_CLOSE
	  PARAMETER (THRS_CLOSE=1.0%(exp_letter)s-02%(mp_specifier)s)
	  %(real_format)s THRS_TEST
	  PARAMETER (THRS_TEST=1.0%(exp_letter)s-15%(mp_specifier)s)
C  
C ARGUMENTS 
C
      %(real_format)s P(0:3,NEXTERNAL)
C
C LOCAL VARIABLES 
C
      INTEGER I,J, P1, P2, ERRCODE
C     PT STANDS FOR PTOT
      %(real_format)s PT(0:3), NEWP(0:3,NEXTERNAL),OLDP(0:3,NEXTERNAL)
	  %(real_format)s BUFF,REF,REF2,DISCR
	  %(real_format)s MASSES(NEXTERNAL)
	  %(real_format)s SHIFTE(2),SHIFTZ(2)
      %(real_format)s cosphi,sinphi,costh,sinth,pp
C
C SAVED VARIABLES
C
      INTEGER WARNED
	  DATA WARNED/0/
C  
C GLOBAL VARIABLES
C 
      include '%(coupl_inc_name)s'

	  %(masses_def)s

C ----------
C BEGIN CODE
C ----------
	  ERRCODE = 0

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
		  WRITE(*,*) 'ERROR:: Four-momentum conservation is not accurate enough, ',(BUFF/REF)
		  CALL %(mp_prefix)sWRITE_MOM(P)
		  STOP
		ENDIF
	  ENDDO
	  REF = REF / (ONE*NEXTERNAL)
	  DO I=1,NEXTERNAL
	    REF=ABS(P(0,I))+ABS(P(1,I))+ABS(P(2,I))+ABS(P(3,I))
		IF ((SQRT(ABS(P(0,I)**2-P(1,I)**2-P(2,I)**2-P(3,I)**2-MASSES(I)**2))/REF).GT.THRES_ONSHELL) THEN
		  WRITE(*,*) 'ERROR:: Onshellness of the momentum of particle ',I,' of mass ',MASSES(I),' is not accurate enough, ', (SQRT(ABS(P(0,I)**2-P(1,I)**2-P(2,I)**2-P(3,I)**2-MASSES(I)**2))/REF)
		  CALL %(mp_prefix)sWRITE_MOM(P)
		  STOP
		ENDIF
	  ENDDO

C NOW WE MAKE SURE THAT THE PS POINT CAN BE IMPROVED BY THE ALGORITHM

     IF (NINITIAL.NE.2) ERRCODE = 100
	 
	 IF (P(1,1)/REF.GT.THRS_TEST.OR.P(2,1)/REF.GT.THRS_TEST.OR.P(1,2)/REF.GT.THRS_TEST.OR.P(2,2)/REF.GT.THRS_TEST) ERRCODE = 200
	   
	 IF (MASSES(1).NE.ZERO.OR.MASSES(2).NE.ZERO) ERRCODE = 300
	 
	 DO I=1,NEXTERNAL
	   IF (P(0,I).LT.ZERO.OR.P(0,I).LT.ZERO) ERRCODE = 400 + I
     ENDDO

     IF (ERRCODE.NE.0) GOTO 100

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
C Rotate the P(1:3,P2), such that it is in the z direction
     cosphi = 1.0%(exp_letter)s+00%(mp_specifier)s
     sinphi = ZERO
     pp=SQRT(P(1,P2)**2+P(2,P2)**2+P(3,P2)**2)
     IF(pp.GT.ZERO)THEN
       IF(SQRT(P(1,P2)**2+P(2,P2)**2)/pp.GT.THRS_TEST)THEN
          cosphi=P(1,P2)/SQRT(P(1,P2)**2+P(2,P2)**2)
          sinphi=P(2,P2)/SQRT(P(1,P2)**2+P(2,P2)**2)
       ENDIF
     ENDIF
     costh = 1.0%(exp_letter)s+00%(mp_specifier)s
     sinth = ZERO
     IF(pp.GT.ZERO)THEN
       costh = P(3,P2)/pp
       sinth = SQRT(P(1,P2)**2+P(2,P2)**2)/pp
     ENDIF
     OLDP(0:3,1:NEXTERNAL)=P(0:3,1:NEXTERNAL)
     CALL %(mp_prefix)sROTATE(P,cosphi,sinphi,costh,sinth)
     IF(pp.GT.ZERO)THEN
        IF(SQRT(P(1,P2)**2+P(2,P2)**2)/pp.GT.THRS_TEST)THEN
           WRITE(*,*)"Failed in rotate the initial momentum into the z direction ",SQRT(P(1,P2)**2+P(2,P2)**2)/pp
           P(0:3,1:NEXTERNAL)=OLDP(0:3,1:NEXTERNAL)
           ERRCODE = 600
           GOTO 100
        ENDIF
     ENDIF

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

C NOW MAKE SURE THE SHIFTED POINT IS NOT TOO FAR FROM THE ORIGINAL ONE
     REF  = ZERO
	 REF2 = ZERO
	 DO J=1,NEXTERNAL
	   DO I=0,3
	     REF2 = REF2 + ABS(P(I,J))	   
	     REF = REF + ABS(P(I,J)-NEWP(I,J)) 
	   ENDDO
	 ENDDO

	  IF ((REF/REF2).GT.THRS_CLOSE) THEN
	    IF (WARNED.LT.4) THEN
	      WRITE(*,*) '##W23 The improved PS point is too far from the original one',(REF/REF2)
		ENDIF
		ERRCODE = 700
		GOTO 100
	  ENDIF
      
C Rotate back

      CALL %(mp_prefix)sROTATE_INV(NEWP,cosphi,sinphi,costh,sinth)
	  GOTO 101
100   CONTINUE
      IF (WARNED.LT.4) THEN
	      WRITE(*,*) '##W23 THIS PS point could not be improved. Error code = ',ERRCODE
		  CALL %(mp_prefix)sWRITE_MOM(P)
	  ELSEIF (WARNED.EQ.10) THEN
	      WRITE(*,*) '##W23 Further warnings from the improve_ps routine will now be supressed.'	     
	  ENDIF
      WARNED = WARNED+1
      P(0:3,1:NEXTERNAL)=OLDP(0:3,1:NEXTERNAL)
	  GOTO 102
101   CONTINUE
      DO J=1,NEXTERNAL
	    DO I=0,3
		  P(I,J)=NEWP(I,J)
		ENDDO
	  ENDDO
102   CONTINUE
	  END

	  SUBROUTINE %(mp_prefix)sWRITE_MOM(P)
	  IMPLICIT NONE
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
	  INTEGER    NINITIAL
      PARAMETER (NINITIAL=%(ninitial)d)
	  %(real_format)s ZERO
	  PARAMETER (ZERO=0.0%(exp_letter)s+00%(mp_specifier)s)
      %(real_format)s %(mp_prefix)smdot

	  INTEGER I,J

C  
C ARGUMENTS 
C
      %(real_format)s P(0:3,NEXTERNAL),PSUM(0:3)
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
        WRITE (*,'(1x,5e25.15)') P(0,I),P(1,I),P(2,I),P(3,I),SQRT(ABS(%(mp_prefix)sMDOT(P(0,I),P(0,I))))
      ENDDO
      WRITE (*,*) '    Four-momentum conservation sum:'
      WRITE (*,'(1x,4e25.15)') PSUM(0),PSUM(1),PSUM(2),PSUM(3)
      WRITE (*,*) '   ---------------------'
	  END

      %(real_format)s function %(mp_prefix)smdot(p1,p2)
C ****************************************************************************
C     4-Vector Dot product
C ****************************************************************************
      implicit none
      %(real_format)s p1(0:3),p2(0:3)
      %(mp_prefix)smdot=p1(0)*p2(0)-p1(1)*p2(1)-p1(2)*p2(2)-p1(3)*p2(3)
      return
      end

C Rotate_PS rotates the PS point PS (without modifying it)
C stores the result in P and modifies the global variables
C PS and MP_DONE accordingly.

	  SUBROUTINE %(mp_prefix)sROTATE_PS(P_IN,P,ROTATION)
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
	  common/MP_DONE/MP_DONE

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

      SUBROUTINE %(mp_prefix)sROTATE(P,cosphi,sinphi,costh,sinth)
C  
C CONSTANTS 
C 
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
C  
C ARGUMENTS 
C
      %(real_format)s P(0:3,NEXTERNAL),cosphi,sinphi,costh,sinth
      %(real_format)s NEWP(0:3,NEXTERNAL)
C
C LOCAL VARIABLES 
C
      INTEGER I,J

C ----------
C BEGIN CODE
C ----------

      DO I=1,NEXTERNAL
         NEWP(1,I)=P(1,I)*cosphi*costh+P(2,I)*costh*sinphi-P(3,I)*sinth
         NEWP(2,I)=P(2,I)*cosphi-P(1,I)*sinphi
         NEWP(3,I)=P(3,I)*costh+(P(1,I)*cosphi+P(2,I)*sinphi)*sinth
	  ENDDO

      P(1:3,1:NEXTERNAL)=NEWP(1:3,1:NEXTERNAL)
	  END
      
      SUBROUTINE %(mp_prefix)sROTATE_INV(P,cosphi,sinphi,costh,sinth)
C  
C CONSTANTS 
C 
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
C  
C ARGUMENTS 
C
      %(real_format)s P(0:3,NEXTERNAL),cosphi,sinphi,costh,sinth
      %(real_format)s NEWP(0:3,NEXTERNAL)
C
C LOCAL VARIABLES 
C
      INTEGER I,J

C ----------
C BEGIN CODE
C ----------

      DO I=1,NEXTERNAL
         NEWP(1,I)=P(1,I)*cosphi*costh-P(2,I)*sinphi+P(3,I)*sinth*cosphi
         NEWP(2,I)=P(2,I)*cosphi+sinphi*(P(1,I)*costh+P(3,I)*sinth)
         NEWP(3,I)=P(3,I)*costh-P(1,I)*sinth
	  ENDDO

      P(1:3,1:NEXTERNAL)=NEWP(1:3,1:NEXTERNAL)
	  END
