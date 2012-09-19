C===================================================================
C A package of utility routines which allow to compute a large set
C of QCD event shape distributions
C
C Guenther Dissertori,  28.5.2001
C===================================================================


      Subroutine ThrustDG(Pext,np,dirs,Miss)
C-----------------------------------------------------------------------
C Thrust,Major/Minor calculation, taking into account missing
C momentum and calling then routine Thrust (based on ALPHA-routine)
C ===================================================================
C
C Input : R*4 p(4,N)    : array with 4-vectors
C         I*4 np        : number of 4-vectors in PP
C         Log miss      : .True. -> use also  missing momentum 
C                       : .False.-> disregard missing momentum
C Output: R*4 dirs(4,3) : full results for Thrust, Major and Minor
C 
C       The return argument DIRS contains the information:
C
C       DIRS(1..3,1) : thrust direction,  DIRS(4,1) : thrust value 
C       DIRS(1..3,2) : major  direction,  DIRS(4,2) : major  value 
C       DIRS(1..3,3) : minor  direction,  DIRS(4,3) : minor  value 
C
C Note: Negative values for thrust signify an error condition:
C
C       Thrust = -1 : too few tracks
C       Thrust = -2 : too many tracks
C
C Author: Guenther Dissertori / 02-Mar-1995
C-----------------------------------------------------------------------

      Implicit  NONE

      Integer   NpMax
      Parameter (NpMax =200) 
      
      Integer   np,i,j,imax,neff
      Real*4    Pext(4,np),p(4,NpMax),dirs(4,3)
      Real*4    pmiss(4),psum,pmax
      Logical   Miss

      SAVE
C-----------------------------------------------------------------------

C -- Check for critical values of particlenumber
 
      if (np.le. 0) then
        dirs(4,1) = -1.
        RETURN
      endif
      if (np.gt.(NpMax-1)) then
        dirs(4,1) = -2.
        RETURN
      endif

C -- find missing momentum 
      do i=1,4
        pmiss(i) = 0.
      enddo

      imax = 0
      psum = 0.
      do i=1,np
        do j=1,4
          p(j,i)   = Pext(j,i)
          pmiss(j) = pmiss(j) + p(j,i)
        enddo
      enddo

      do i=1,4
        p(i,np+1) = (-1.)*pmiss(i)
      enddo

      if (Miss) then
        neff = np+1
      else
        neff = np
      endif

      call Thrust(p,neff,dirs)

C -- find fastest particle and put Thrust in same hemisphere
      imax = 0
      pmax = 0.     
      do i=1,neff
        psum = sqrt(p(1,i)**2+p(2,i)**2+p(3,i)**2)
        if (psum.gt.pmax) then
          pmax = psum
          imax = i
        endif
      enddo

      if ((dirs(1,1)*p(1,imax)+
     >     dirs(2,1)*p(2,imax)+
     >     dirs(3,1)*p(3,imax)).lt.0.) then
        dirs(1,1) = (-1.)*dirs(1,1)
        dirs(2,1) = (-1.)*dirs(2,1)
        dirs(3,1) = (-1.)*dirs(3,1)
      endif

      RETURN
      END



C-----------------------------------------------------------------------
C-----------------------------------------------------------------------

      Subroutine Thrust(p,np,dirs)
C-----------------------------------------------------------------------
C Stand-alone self-contained Thrust,Major/Minor calculation,
C based on a Thrustcalculation by H.Albrecht
C ================================================================
C
C Input : R*4 p(4,N)    : array with 4-vectors
C         I*4 np        : number of 4-vectors in PP 
C Output: R*4 dirs(4,3) : full results for Thrust, Major and Minor
C 
C       The return argument DIRS contains the information:
C
C       DIRS(1..3,1) : thrust direction,  DIRS(4,1) : thrust value 
C       DIRS(1..3,2) : major  direction,  DIRS(4,2) : major  value 
C       DIRS(1..3,3) : minor  direction,  DIRS(4,3) : minor  value 
C
C Note: Negative values for thrust signify an error condition:
C
C       Thrust = -1 : too few tracks
C       Thrust = -2 : too many tracks
C       Thrust = -3 : zero or negative Thrust value obtained
C
C Author: Guenther Dissertori / 10-Feb-1995
C-----------------------------------------------------------------------

      Implicit  NONE

      Integer   NpMax
      Parameter (NpMax=200)
      
      Integer   np,i,j,npt
      Real*4    p(4,np),pt(4,NpMax),dirs(4,3),ptot,temp(4,NpMax)
      Real*4    t,dt(3),major,dmajor(3),minor,dminor(3)

      SAVE
C-----------------------------------------------------------------------

C -- Check for critical values of particlenumber

      if (np.le. 0) then
        dirs(4,1) = -1.
        RETURN
      endif
      if (np.gt.NpMax) then
        dirs(4,1) = -2.
        RETURN
      endif

C -- Calculate first Thrust and its axis
      if (np.gt.2) then
        call AThrust(p,np,t,dt)
      else
        call RThrust(p,np,t,dt)
      endif

      if (t.le.0.) then
        dirs(4,1) = -3.
        RETURN
      endif

      dirs(4,1) = t
      do i=1,3
        dirs(i,1) = dt(i)
      enddo

C -- Now calculate normal vectors to Thrustaxis
      npt = 0
      do i=1,np
        temp(1,i) = p(2,i)*dt(3) - p(3,i)*dt(2)
        temp(2,i) = p(3,i)*dt(1) - p(1,i)*dt(3)
        temp(3,i) = p(1,i)*dt(2) - p(2,i)*dt(1)        
        temp(4,i) = sqrt(temp(1,i)**2+temp(2,i)**2+temp(3,i)**2)
        if (temp(4,i).ne.0.) then
          npt = npt+1
          pt(1,npt) = temp(1,i)
          pt(2,npt) = temp(2,i)
          pt(3,npt) = temp(3,i)
          pt(4,npt) = temp(4,i)
        endif
      enddo

C -- Check against balanced 2-particle final state
      if (npt.eq.0) then
        do i=1,4
          dirs(i,2) = 0.
          dirs(i,3) = 0.
        enddo
        goto 10
      endif
        
C -- Use normal vectors to compute Minor
C -- Note : The axis we get is rotated by 90deg with respect
C           to the Major_axis -> Minor

      if (npt.gt.2) then
        call AThrust(pt,npt,minor,dminor)
      else
        call RThrust(pt,npt,minor,dminor)
      endif

      minor = 0.
      ptot  = 0.
      do i=1,np
        ptot  = ptot + sqrt(p(1,i)**2+p(2,i)**2+p(3,i)**2)
        minor = minor + abs(p(1,i)*dminor(1)+
     >                      p(2,i)*dminor(2)+
     >                      p(3,i)*dminor(3))
      enddo
      dirs(4,3) = minor/ptot
      do i=1,3
        dirs(i,3) = dminor(i)/max(1.e-17,
     >              (sqrt(dminor(1)**2+dminor(2)**2+dminor(3)**2)))
      enddo

C -- now major axis and major values
      do i=1,3
        dmajor(1) = dt(2)*dminor(3) - dt(3)*dminor(2)
        dmajor(2) = dt(3)*dminor(1) - dt(1)*dminor(3)
        dmajor(3) = dt(1)*dminor(2) - dt(2)*dminor(1)
      enddo

      major = 0.
      do i=1,np
        major = major + abs(p(1,i)*dmajor(1)+
     >                      p(2,i)*dmajor(2)+
     >                      p(3,i)*dmajor(3))
      enddo

      dirs(4,2) = major/ptot
      do i=1,3
        dirs(i,2) = dmajor(i)/max(1.e-17,
     >              (sqrt(dmajor(1)**2+dmajor(2)**2+dmajor(3)**2)))
      enddo

 10   RETURN
      END


C=======================================================================
C-----------------------------------------------------------------------
C-----------------------------------------------------------------------


      SUBROUTINE RThrust(p,np,t,dt)
C-----------------------------------------------------------------------
C Calculate Thrust for a set of 4-momenta
C =======================================
C
C Input : R*4 p(4,np) : array of 4-vectors
C         I*4 np      : number of 4-vectors 
C Output: R*4 t       : thrust
C         R*4 dt(3)   : thrust axis
C 
C Description
C ===========
C An exact calculation of Thrust is performed for a set of not too
C many balanced momentum vectors, i.e. which add up to zero. This 
C routine exploits the facts, that 
C  a) the Thrust axis is given by the longest vector that can be 
C     formed from any subset of the available momentum vectors, 
C     and that
C  b) all partions of a set of N elements into 2 subsets are given 
C     by the pattern of 0 and 1 of a binary counter which ranges 
C     from 0 to 2**N-1. 
C As the momenta are assumed to be balanced only 2**(N-1)-1 
C combinations actually have to be tried. Note that the limitation
C for this routine is at np=32 - which already requires a few hours
C CPU-time on an ALPHA 3000-600.  Up to np=8 this routine is 
C faster than the LUND algorithm, as far as only the calculation of 
C Thrust is required.
C
C Author: Michael Schmelling / 9-Feb-1995
C-----------------------------------------------------------------------

      Implicit NONE

      Integer   np
      Real      p(4,np),t,dt(3)

      Real      px,py,pz,temp
      Integer   i,n,nmax,icnt,nact
C-----------------------------------------------------------------------

C--      initialization

      t     = 0.
      dt(1) = 0.
      dt(2) = 0.
      dt(3) = 0.

      if(np.le. 0) RETURN
      if(np.gt.32) STOP 'RTHRUST: too many momentum vectors'  

C--      loop over all possible subsets 

      nact =    np   - 1
      nmax = 2**nact - 1
      icnt = 0
      do n=1,nmax
         icnt = icnt + 1
         px = 0.
         py = 0.
         pz = 0.
         do i=1,nact
            if(btest(icnt,i-1)) then
               px = px + p(1,i)
               py = py + p(2,i)
               pz = pz + p(3,i)
            endif
         enddo
         temp = px**2+py**2+pz**2
         if(t.lt.temp) then 
            t     = temp
            dt(1) = px
            dt(2) = py
            dt(3) = pz
         endif
      enddo
      if(t.le.0.) RETURN

C--      fill return arguments

      t     = sqrt(t) 
      dt(1) = dt(1)/t
      dt(2) = dt(2)/t
      dt(3) = dt(3)/t
      temp  = 0.
      do i=1,np
         temp = temp + sqrt(p(1,i)**2+p(2,i)**2+p(3,i)**2)
      enddo
      t = 2.*t/temp

      RETURN
      END


C***********************************************************************
C***********************************************************************


      SUBROUTINE ATHRUST(PP,N,t,dt)
C-----------------------------------------------------------------------
C ALPHA-topological event analysis: thrust
C                                                   H. Albrecht. feb 82
C
C Algorithm: try initial trust axis from sum of projections along
C            all cross products between pairs of particles. Take 
C            the axis with maximum thrust value and iterate once 
C            more.
C-----------------------------------------------------------------------
  
      Implicit NONE

      Integer n
      Real    pp(4,n),t,dt(3)

      Integer NpMax
      Parameter (NpMax=200)

      Integer ktbi,i,j,l,iter
      Real    vcx,vcy,vcz,vlx,vly,vlz,vnx,vny,vnz,vmx,vmy,vmz
      Real    vmax,vsum,vnew,qtbox,qtboy,qtboz
      Real    qtbix(NpMax),qtbiy(NpMax),qtbiz(NpMax)
C-----------------------------------------------------------------------

      KTBI = n

      IF (KTBI.LE.    2)  STOP 'no thrust for 2 or less tracks'
      IF (KTBI.GT.NpMax)  STOP 'no thrust for more than 100 tracks'

      do i=1,n
         qtbix(i) = pp(1,i)
         qtbiy(i) = pp(2,i)
         qtbiz(i) = pp(3,i)
      enddo

C
      VMAX = 0.
      DO 30 I=1,KTBI-1
        DO 20 J=I+1,KTBI
          VCX = QTBIY(I) * QTBIZ(J) - QTBIY(J) * QTBIZ(I)
          VCY = QTBIZ(I) * QTBIX(J) - QTBIZ(J) * QTBIX(I)
          VCZ = QTBIX(I) * QTBIY(J) - QTBIX(J) * QTBIY(I)
          VLX = 0.
          VLY = 0.
          VLZ = 0.
C
          DO 10 L=1,KTBI
            IF (L .EQ. I .OR. L .EQ. J)  GO TO 10
            IF (QTBIX(L) * VCX + QTBIY(L) * VCY +
     &          QTBIZ(L) * VCZ .GE. 0.)  THEN
              VLX = VLX + QTBIX(L)
              VLY = VLY + QTBIY(L)
              VLZ = VLZ + QTBIZ(L)
            ELSE
              VLX = VLX - QTBIX(L)
              VLY = VLY - QTBIY(L)
              VLZ = VLZ - QTBIZ(L)
            ENDIF
   10     CONTINUE
C
C --- make all four sign-combinations for I,J
C
          VNX = VLX + QTBIX(J) + QTBIX(I)
          VNY = VLY + QTBIY(J) + QTBIY(I)
          VNZ = VLZ + QTBIZ(J) + QTBIZ(I)
          VNEW = VNX**2 + VNY**2 + VNZ**2
          IF (VNEW .GT. VMAX)  THEN
            VMAX = VNEW
            VMX = VNX
            VMY = VNY
            VMZ = VNZ
          ENDIF
C
          VNX = VLX + QTBIX(J) - QTBIX(I)
          VNY = VLY + QTBIY(J) - QTBIY(I)
          VNZ = VLZ + QTBIZ(J) - QTBIZ(I)
          VNEW = VNX**2 + VNY**2 + VNZ**2
          IF (VNEW .GT. VMAX)  THEN
            VMAX = VNEW
            VMX = VNX
            VMY = VNY
            VMZ = VNZ
          ENDIF
C
          VNX = VLX - QTBIX(J) + QTBIX(I)
          VNY = VLY - QTBIY(J) + QTBIY(I)
          VNZ = VLZ - QTBIZ(J) + QTBIZ(I)
          VNEW = VNX**2 + VNY**2 + VNZ**2
          IF (VNEW .GT. VMAX)  THEN
            VMAX = VNEW
            VMX = VNX
            VMY = VNY
            VMZ = VNZ
          ENDIF
C
          VNX = VLX - QTBIX(J) - QTBIX(I)
          VNY = VLY - QTBIY(J) - QTBIY(I)
          VNZ = VLZ - QTBIZ(J) - QTBIZ(I)
          VNEW = VNX**2 + VNY**2 + VNZ**2
          IF (VNEW .GT. VMAX)  THEN
            VMAX = VNEW
            VMX = VNX
            VMY = VNY
            VMZ = VNZ
          ENDIF
   20   CONTINUE
   30 CONTINUE
C
C -- sum momenta of all particles and iterate
C
      DO 50 ITER=1,4
        QTBOX = 0.
        QTBOY = 0.
        QTBOZ = 0.
        DO 40  I=1,KTBI
          IF (VMX * QTBIX(I) + VMY * QTBIY(I) +
     +        VMZ * QTBIZ(I) .GE. 0.)  THEN
            QTBOX = QTBOX + QTBIX(I)
            QTBOY = QTBOY + QTBIY(I)
            QTBOZ = QTBOZ + QTBIZ(I)
          ELSE
            QTBOX = QTBOX - QTBIX(I)
            QTBOY = QTBOY - QTBIY(I)
            QTBOZ = QTBOZ - QTBIZ(I)
          ENDIF
   40   CONTINUE
        VNEW = QTBOX**2 + QTBOY**2 + QTBOZ**2
        IF (VNEW .EQ. VMAX)  GO TO 70
        VMAX = VNEW
        VMX = QTBOX
        VMY = QTBOY
        VMZ = QTBOZ
   50 CONTINUE
C
C
C --- normalize thrust -division by total momentum-
C
   70 VSUM = 0.
      DO 80 I=1,KTBI
        VSUM = VSUM + SQRT (QTBIX(I)**2 + QTBIY(I)**2 + QTBIZ(I)**2)
 80   CONTINUE

      t     = SQRT (VNEW) / VSUM
      vnew  = sqrt(qtbox**2+qtboy**2+qtboz**2)
      dt(1) = QTBOX/vnew
      dt(2) = QTBOY/vnew
      dt(3) = QTBOZ/vnew
C  
      RETURN
      END

C***********************************************************************
C***********************************************************************


      FUNCTION Thrust_Lund(PP,N,DIRS)
C-----------------------------------------------------------------------
C Stand-alone self-contained thrust calculation based on 4-vectors
C ================================================================
C
C Input : R*4 PP(4,N)   : array with 4-vectors
C         I*4 N         : number of 4-vectors in PP 
C Output: R*4 Thrust_Lund    : trust value
C         R*4 DIRS(4,3) : full results for Thrust, Major and Minor
C 
C Note: Negative values for thrust signifif am error condition:
C
C       Thrust_Lund = -1 : too few tracks
C       Thrust_Lund = -2 : too many tracks
C 
C       The return argument DIRS contains additional information:
C
C       DIRS(1..3,1) : thrust direction,  DIRS(4,1) : thrust value 
C       DIRS(1..3,2) : major  direction,  DIRS(4,2) : major  value 
C       DIRS(1..3,3) : minor  direction,  DIRS(4,3) : minor  value 
C
C       This routine was adapted from a hack-up of the original 
C       LUND 6.3 program by Glen Cowan.
C
C Author: Michael Schmelling  / 08-Dec-1990 
C         Guenther Dissertori / 13-Dec-1994  (small bug fix regarding
C                                             THRUST = -1,-2)
C-----------------------------------------------------------------------

      PARAMETER (NMX=300)   ! BUFFER SIZE
      PARAMETER (EPS=1.E-4) ! CONVERGENCE PARAMETER

      REAL PP(4,*),DIRS(4,*),P(NMX,5),TDI(3),TPR(3),ROT(3,3),PV(3)

      DATA JRAN,IM,IA,IC/1234,134456,8121,28411/  ! IN-LINE RNDM


      Thrust_Lund = 0.

C -- Check for extreme values 

      IF(N.LT.3) THEN
        Thrust_Lund = -1
        RETURN
      ENDIF

      IF(N.GT.NMX-25) THEN
         Thrust_Lund = -2.
         RETURN
      ENDIF

C--      COPY DATA TO INTERNAL ARRAYS

      DO I=1,N
        P(I,1) = PP(1,I)
        P(I,2) = PP(2,I)
        P(I,3) = PP(3,I)
      ENDDO
 
C--      START THE LUND ALOGORITHM

      NP=0
      PS=0.
      DO 280 LD=1,2
      IF(LD.EQ.2) THEN
C...THRUST AXIS ALONG Z DIRECTION FOR MAJOR AXIS SEARCH
        PHI    = ATAN2(P(N+1,2),P(N+1,1))
        NROT   = N+1
        IFLAG  = 1
        THEROT = 0.
        PHIROT = -PHI
        GOTO 7777
 7771   CONTINUE
        THE    = ATAN2(P(N+1,1),P(N+1,3))
        IFLAG  = 2
        THEROT = -THE
        PHIROT = 0.
        GOTO 7777
 7772   CONTINUE
      ENDIF
 
C...FIND AND ORDER PARTICLES WITH HIGHEST P (PT FOR MAJOR)
      DO 100 LF=N+4,N+8
  100 P(LF,4)=0.
      DO 140 I=1,N
      IF(LD.EQ.1) THEN
        NP=NP+1
        PA=SQRT(P(I,1)**2+P(I,2)**2+P(I,3)**2)
        P(I,5)=1.
        PS=PS+P(I,5)*PA
      ELSE
        PA=SQRT(P(I,1)**2+P(I,2)**2)
      ENDIF
      DO 110 LF=N+7,N+4,-1
      IF(PA.LE.P(LF,4)) GOTO 120
      DO 110 J=1,5
  110 P(LF+1,J)=P(LF,J)
      LF=N+3
  120 DO 130 J=1,3
  130 P(LF+1,J)=P(I,J)
      P(LF+1,4)=PA
      P(LF+1,5)=P(I,5)
  140 CONTINUE
 
      IF(NP.LE.1) THEN
C...VERY LOW MULTIPLICITIES (0 OR 1) NOT CONSIDERED
        Thrust_Lund =-1.
        RETURN
      ENDIF
 
C...FIND AND ORDER INITIAL AXES WITH HIGHEST THRUST
      DO 150 LG=N+9,N+19
  150 P(LG,4)=0.
      NC=2**(MIN(4,NP)-1)
      DO 210 LC=1,NC
      DO 160 J=1,3
  160 TDI(J)=0.
      DO 170 LF=1,MIN(4,NP)
      SGN=P(N+LF+3,5)
      IF(2**LF*((LC+2**(LF-1)-1)/2**LF).GE.LC) SGN=-SGN
      DO 170 J=1,4-LD
  170 TDI(J)=TDI(J)+SGN*P(N+LF+3,J)
      TDS=TDI(1)**2+TDI(2)**2+TDI(3)**2
      DO 180 LG=N+8+MIN(LC,10),N+9,-1
      IF(TDS.LE.P(LG,4)) GOTO 190
      DO 180 J=1,4
  180 P(LG+1,J)=P(LG,J)
      LG=N+8
  190 DO 200 J=1,3
  200 P(LG+1,J)=TDI(J)
      P(LG+1,4)=TDS
  210 CONTINUE
 
C...ITERATE DIRECTION OF AXIS UNTIL STABLE MAXIMUM
      P(N+LD,4)=0.
      LG=0
  220 LG=LG+1
      THP=0.
  230 THPS=THP
      DO 240 J=1,3
      IF(THP.LE.1E-10) TDI(J)=P(N+8+LG,J)
      IF(THP.GT.1E-10) TDI(J)=TPR(J)
  240 TPR(J)=0.
      DO 260 I=1,N
      SGN=SIGN(P(I,5),TDI(1)*P(I,1)+TDI(2)*P(I,2)+TDI(3)*P(I,3))
      DO 250 J=1,4-LD
  250 TPR(J)=TPR(J)+SGN*P(I,J)
  260 CONTINUE
      THP=SQRT(TPR(1)**2+TPR(2)**2+TPR(3)**2)/PS
      IF(THP.GE.THPS+EPS) GOTO 230
 
C...SAVE GOOD AXIS, TRY NEW INITIAL AXIS UNTIL A NUMBER OF TRIES AGREE
      IF(THP.LT.P(N+LD,4)-EPS.AND.LG.LT.MIN(10,NC)) GOTO 220
      IF(THP.GT.P(N+LD,4)+EPS) THEN
        LAGR=0
        JRAN = MOD(JRAN*IA+IC,IM)
        RAN  = FLOAT(JRAN)/FLOAT(IM)
        SGN=(-1.)**INT(RAN+0.5)
        DO 270 J=1,3
  270   P(N+LD,J)=SGN*TPR(J)/(PS*THP)
        P(N+LD,4)=THP
      ENDIF
      LAGR=LAGR+1
  280 IF(LAGR.LT.2.AND.LG.LT.MIN(10,NC)) GOTO 220
 
C...FIND MINOR AXIS AND VALUE BY ORTHOGONALITY
      JRAN = MOD(JRAN*IA+IC,IM)
      RAN  = FLOAT(JRAN)/FLOAT(IM)
      SGN=(-1.)**INT(RAN+0.5)
      P(N+3,1)=-SGN*P(N+2,2)
      P(N+3,2)=SGN*P(N+2,1)
      P(N+3,3)=0.
      THP=0.
      DO 290 I=1,N
      THP=THP+P(I,5)*ABS(P(N+3,1)*P(I,1)+P(N+3,2)*P(I,2))
      P(I,5)=SQRT(MAX(P(I,4)**2-P(I,1)**2-P(I,2)**2-P(I,3)**2,0.))
  290 CONTINUE
      P(N+3,4)=THP/PS
 
C...RESET UNUSED COMPONENTS, ROTATE BACK TO ORIGINAL COORDINATE SYSTEM
      DO 300 LD=1,3
  300 P(N+LD,5)=0.
      NROT   = N + 3
      THEROT = THE
      PHIROT = PHI
      IFLAG  = 3
      GOTO 7777
 7773 CONTINUE

C--      COPY RESULTS TO OUTPUT BUFFER AND RETURN
 
      DO I=1,4
        DIRS(I,1) = P(N+1,I)
        DIRS(I,2) = P(N+2,I)
        DIRS(I,3) = P(N+3,I)
      ENDDO
 
      Thrust_Lund = DIRS(4,1)        
      RETURN

C-----------------------------------------------------------------------
C PATCH THE PART OF LUROBO NEEDED FOR THRUST INTO THIS ROUTINE

 7777 CONTINUE
      IF(NROT.GT.0.AND.THEROT**2+PHIROT**2.GT.1E-20) THEN
C...ROTATE (TYPICALLY FROM Z AXIS TO DIRECTION THETA,PHI)
        ROT(1,1)=COS(THEROT)*COS(PHIROT)
        ROT(1,2)=-SIN(PHIROT)
        ROT(1,3)=SIN(THEROT)*COS(PHIROT)
        ROT(2,1)=COS(THEROT)*SIN(PHIROT)
        ROT(2,2)=COS(PHIROT)
        ROT(2,3)=SIN(THEROT)*SIN(PHIROT)
        ROT(3,1)=-SIN(THEROT)
        ROT(3,2)=0.
        ROT(3,3)=COS(THEROT)
        DO 7120 I=1,NROT
        DO 7100 J=1,3
 7100   PV(J)=P(I,J)
        DO 7110 J=1,3
 7110   P(I,J)=ROT(J,1)*PV(1)+ROT(J,2)*PV(2)+ROT(J,3)*PV(3)
 7120   CONTINUE
      ENDIF

      IF(IFLAG.EQ.1) GOTO 7771
      IF(IFLAG.EQ.2) GOTO 7772
      IF(IFLAG.EQ.3) GOTO 7773
      END



C***********************************************************************
C***********************************************************************


      Subroutine Cparameter(p,n,Cpar,Dpar)
C-----------------------------------------------------------------------
C Calculate the C-parameter
C =========================
C
C Input : R*4 p(4,*) : array of 4-vectors
C         I*4 n      : number of 4-vectors
C Output: R*4 Cpar   : C-parameter
C         R*4 Dpar   : D-parameter
C
C Author: Michael Schmelling  / 14-Dec-1990 
C         Guenther Dissertori / 05-May-1995  (D-parameter added)
C-----------------------------------------------------------------------

      Implicit NONE

      Real    p(4,*),txx,tyy,tzz,txy,txz,tyz,sum,pi
      Real    Cpar,Dpar
      Integer n,i

      txx = 0.
      tyy = 0.
      tzz = 0.
      txy = 0.
      txz = 0.
      tyz = 0.
      sum = 0.
      
      do 111 i=1,n
      pi  = sqrt(p(1,i)*p(1,i)+p(2,i)*p(2,i)+p(3,i)*p(3,i))
      sum = sum + pi
      if (pi.ne.0.) then
        txx = txx + p(1,i)*p(1,i)/pi
        tyy = tyy + p(2,i)*p(2,i)/pi
        tzz = tzz + p(3,i)*p(3,i)/pi
        txy = txy + p(1,i)*p(2,i)/pi
        txz = txz + p(1,i)*p(3,i)/pi
        tyz = tyz + p(2,i)*p(3,i)/pi
      endif
 111  continue

      Cpar = txx*tyy + tyy*tzz + tzz*txx - txy*txy - txz*txz - tyz*tyz
      Cpar = 3.*Cpar/sum/sum

      Dpar = txx*tyy*tzz + 2*txy*txz*tyz - tyy*(txz**2) - txx*(tyz**2)
     >       - tzz*(txy**2)
      Dpar = 27.*Dpar/sum/sum/sum

      RETURN
      END


C*********************************************************************
C*********************************************************************


      SUBROUTINE JetMass(p,np,dir,mh2,ml2,md2)
C-----------------------------------------------------------------------
C Calculate mh**2/s and md**2/s
C =============================
C
C Input : R*4 p(4,*) : array of 4-vectors
C         I*4 np     : number of 4-vectors in p
C         R*4 dir(3) : direction that defines hemispheres
C Output: R*4 mh2    : high mass**2/s
C         R*4 ml2    : low mass**2/s
C         R*4 md2    : mass**2 difference/s
C
C Author: Michael Schmelling / 8-Dec-1990 
C-----------------------------------------------------------------------

      Implicit NONE

      Real     p(4,*),dir(3),mh2,ml2,md2,ppos(4),pneg(4),s,mpos,mneg
      Integer  np,i

      do 1 i=1,4
      ppos(i) = 0.
      pneg(i) = 0.
 1    continue

      do 10 i=1,np
      if((p(1,i)*dir(1)+p(2,i)*dir(2)+p(3,i)*dir(3)).gt.0.) then
         ppos(1) = ppos(1) + p(1,i)
         ppos(2) = ppos(2) + p(2,i)
         ppos(3) = ppos(3) + p(3,i)
         ppos(4) = ppos(4) + p(4,i)
      else
         pneg(1) = pneg(1) + p(1,i)
         pneg(2) = pneg(2) + p(2,i)
         pneg(3) = pneg(3) + p(3,i)
         pneg(4) = pneg(4) + p(4,i)
      endif
 10   continue

      s    = (ppos(4)+pneg(4))**2 
      mpos = (ppos(4)**2-ppos(1)**2-ppos(2)**2-ppos(3)**2) / s
      mneg = (pneg(4)**2-pneg(1)**2-pneg(2)**2-pneg(3)**2) / s
      mh2  = max(mpos,mneg)
      ml2  = min(mpos,mneg)
      md2  = abs(mpos-mneg)
    
      RETURN
      END

C*********************************************************************
C*********************************************************************


      SUBROUTINE HJetMass(p,np,dir,mh2,ml2,md2,h)
C-----------------------------------------------------------------------
C Calculate mh**2/s and md**2/s
C =============================
C
C Input : R*4 p(4,*) : array of 4-vectors
C         I*4 np     : number of 4-vectors in p
C         R*4 dir(3) : direction that defines hemispheres
C Output: R*4 mh2    : high mass**2/s
C         R*4 ml2    : low mass**2/s
C         R*4 md2    : mass**2 difference/s
C         I*4 h      : hemisphere of mh2: +1 ptot || thrust, -1 else
C
C Author: Michael Schmelling / 8-Dec-1990 
C-----------------------------------------------------------------------

      Implicit NONE

      Real     p(4,*),dir(3),mh2,ml2,md2,ppos(4),pneg(4),s,mpos,mneg
      Integer  np,i,h

      do 1 i=1,4
      ppos(i) = 0.
      pneg(i) = 0.
 1    continue

      do 10 i=1,np
      if((p(1,i)*dir(1)+p(2,i)*dir(2)+p(3,i)*dir(3)).gt.0.) then
         ppos(1) = ppos(1) + p(1,i)
         ppos(2) = ppos(2) + p(2,i)
         ppos(3) = ppos(3) + p(3,i)
         ppos(4) = ppos(4) + p(4,i)
      else
         pneg(1) = pneg(1) + p(1,i)
         pneg(2) = pneg(2) + p(2,i)
         pneg(3) = pneg(3) + p(3,i)
         pneg(4) = pneg(4) + p(4,i)
      endif
 10   continue

      s    = (ppos(4)+pneg(4))**2 
      mpos = (ppos(4)**2-ppos(1)**2-ppos(2)**2-ppos(3)**2) / s
      mneg = (pneg(4)**2-pneg(1)**2-pneg(2)**2-pneg(3)**2) / s
      mh2  = max(mpos,mneg)
      ml2  = min(mpos,mneg)
      md2  = abs(mpos-mneg)

      if (mpos.gt.mneg) then
        h =  1
      else
        h = -1
      endif
    
      RETURN
      END


C*********************************************************************
C*********************************************************************


      SUBROUTINE JetBroad(p,np,dir,Btot,Bw)
C-----------------------------------------------------------------------
C Calculate Jet Broadening Measures Btot and Bw
C =============================================
C
C Input : R*4 p(4,*) : array of 4-vectors
C         I*4 np     : number of 4-vectors in p
C         R*4 dir(3) : direction that defines hemispheres
C Output: R*4 Btot   : total broadening
C         R*4 Bw     : highest broadening of the 2 hemispheres
C
C Author: Guenther Dissertori / 16-Dec-1994 
C-----------------------------------------------------------------------

      Implicit NONE

      Real     p(4,*),dir(3),Btot,Bw,ptpos,ptneg,pt(3),Ptot
      Integer  np,i

      Ptot  = 0.
      ptpos = 0.
      ptneg = 0.
      
      do i=1,np
        pt(1) = p(2,i)*dir(3) - p(3,i)*dir(2)
        pt(2) = p(3,i)*dir(1) - p(1,i)*dir(3)
        pt(3) = p(1,i)*dir(2) - p(2,i)*dir(1)
        Ptot  = Ptot + sqrt(p(1,i)**2+p(2,i)**2+p(3,i)**2)
        if((p(1,i)*dir(1)+p(2,i)*dir(2)+p(3,i)*dir(3)).gt.0.) then
           ptpos = ptpos + sqrt(pt(1)**2+pt(2)**2+pt(3)**2)
        else
           ptneg = ptneg + sqrt(pt(1)**2+pt(2)**2+pt(3)**2)
        endif
      enddo

      Btot = (ptpos + ptneg)/(2.*Ptot)
      Bw   = (max(ptpos,ptneg))/(2.*Ptot)
    
      RETURN
      END


C*********************************************************************
C*********************************************************************


      SUBROUTINE HJetBroad(p,np,dir,Btot,Bw,h)
C-----------------------------------------------------------------------
C Calculate Jet Broadening Measures Btot and Bw
C =============================================
C
C Input : R*4 p(4,*) : array of 4-vectors
C         I*4 np     : number of 4-vectors in p
C         R*4 dir(3) : direction that defines hemispheres
C Output: R*4 Btot   : total broadening
C         R*4 Bw     : highest broadening of the 2 hemispheres
C         I*4 h      : hemisphere of Bw, +1 if || with thrust, -1 else
C
C Author: Guenther Dissertori / 9-Jun-1998 
C-----------------------------------------------------------------------

      Implicit NONE

      Real     p(4,*),dir(3),Btot,Bw,ptpos,ptneg,pt(3),Ptot
      Integer  np,i,h

      Ptot  = 0.
      ptpos = 0.
      ptneg = 0.
      
      do i=1,np
        pt(1) = p(2,i)*dir(3) - p(3,i)*dir(2)
        pt(2) = p(3,i)*dir(1) - p(1,i)*dir(3)
        pt(3) = p(1,i)*dir(2) - p(2,i)*dir(1)
        Ptot  = Ptot + sqrt(p(1,i)**2+p(2,i)**2+p(3,i)**2)
        if((p(1,i)*dir(1)+p(2,i)*dir(2)+p(3,i)*dir(3)).gt.0.) then
           ptpos = ptpos + sqrt(pt(1)**2+pt(2)**2+pt(3)**2)
        else
           ptneg = ptneg + sqrt(pt(1)**2+pt(2)**2+pt(3)**2)
        endif
      enddo

      Btot = (ptpos + ptneg)/(2.*Ptot)
      Bw   = (max(ptpos,ptneg))/(2.*Ptot)

      if (ptpos.gt.ptneg) then
        h =  1
      else
        h = -1
      endif

      RETURN
      END


C*********************************************************************
C*********************************************************************


      Subroutine DurhamYn(Pext,Np,Scheme,Yn,Error)
C=====================================================================
C Routine to calculate Yn, where Yn is minimum metric in case
C of a n-jet event
C
C Input : R*4 Pext(4,Np)   Array of external momenta
C         I*4 Np           Number of external particles
C         I*4 Scheme       Recombination Scheme
C                          Scheme = 0 : E0 - Scheme
C                          Scheme = 1 : E  - Scheme
C Output: R*4 Yn(Np)       Array of Yn values
C         I*4 Error        Error = 0 : o.k.
C                          Error = 1 : to few particles
C                          Error = 2 : to many particles
C                          Error = 3 : division by zero
C
C Author: Guenther Dissertori / 14-Dec-1994
C---------------------------------------------------------------------

      Implicit NONE

      Integer   NpMax
      Parameter (NpMax = 300)   

C -- External variables
      Real    Pext(4,*),Yn(*)
      Integer Np,Scheme,Error

C -- Internal Variables
      Real    P(4,NpMax),Y(NpMax,NpMax),Ymin
      Real    P2i,P2j,Cij,Evis,EnClus,PnClus
      Integer Iflag(NpMax),Njets
      Integer i,j,Imin,Jmin,IminC

C=====================================================================

C -- Check if # of particles is o.k.
      
      Error = 0
      if (Np.lt.3) then
        Error = 1
        Return
      endif
      if (Np.gt.NpMax) then
        Error = 2
        Return
      endif

C -- Copy momenta to internal array, initialize
C -- rescale all momenta to zero mass in case E0
      
      Evis = 0.
      do i=1,Np
        Iflag(i) = 0
        Evis = Evis + Pext(4,i)
        EnClus = Pext(4,i)
        PnClus = sqrt(Pext(1,i)**2+Pext(2,i)**2+Pext(3,i)**2)
        do j=1,3
          if (Scheme.eq.0) P(j,i)=Pext(j,i)*EnClus/(max(1.e-7,PnClus))
          if (Scheme.eq.1) P(j,i)=Pext(j,i) 
        enddo
        P(4,i) = EnClus
      enddo


      Njets = Np
      IminC = 0

C -- loop over # of jets

 100  continue
      if (Njets.lt.3) RETURN
      Ymin  = 100000.

C -- loop over combination of particles
      
      do i=1,Np-1
        if (Iflag(i).eq.1) goto 300
        do j=i+1,Np
          if (Iflag(j).eq.1) goto 200
C -- if nothing has changed for i and j
          if ((i.ne.IminC).and.(j.ne.IminC).and.(Njets.ne.Np)) then
            if (Y(i,j).lt.Ymin) then
              Ymin = Y(i,j)
              Imin = i
              Jmin = j
            endif
          else 
C -- else compute metric now
            Cij = P(1,i)*P(1,j)+P(2,i)*P(2,j)+P(3,i)*P(3,j)
            P2i = P(1,i)*P(1,i)+P(2,i)*P(2,i)+P(3,i)*P(3,i)
            P2j = P(1,j)*P(1,j)+P(2,j)*P(2,j)+P(3,j)*P(3,j)
            if ((P2i.le.1.e-14).or.(P2j.le.1.e-14)) then
              Y(i,j) = 0.
            else
              Y(i,j) = 2.*min((P(4,i)/Evis)**2,(P(4,j)/Evis)**2)*
     >                 (1. - Cij/(sqrt(P2i*P2j)))
            endif

            if (Y(i,j).lt.Ymin) then
              Ymin = Y(i,j)
              Imin = i
              Jmin = j
            endif
          endif
 200    enddo
 300  enddo

C -- store Ymin in Yn
      Yn(Njets-2) = Ymin
      Njets = Njets - 1

C -- recombine according to E-Scheme
      if (Scheme.eq.1) then
        do i=1,4
          P(i,Imin) = P(i,Imin) + P(i,Jmin)
        enddo
      endif

C -- recombine according to E0-Scheme
      if (Scheme.eq.0) then
        P(4,Imin) = P(4,Imin) + P(4,Jmin)
        P2i = sqrt((P(1,Imin) + P(1,Jmin))**2 +
     >             (P(2,Imin) + P(2,Jmin))**2 +
     >             (P(3,Imin) + P(3,Jmin))**2) 
        if (P2i.le.1.e-14) then
          Error = 3
          RETURN
        endif
        do i=1,3
          P(i,Imin) = P(4,Imin) * (P(i,Imin) + P(i,Jmin))/P2i
        enddo
      endif

C -- lock the other track
      Iflag(Jmin) = 1
      IminC = Imin

      goto 100

      END


C*********************************************************************
C*********************************************************************


      Subroutine DurhamJn(Pext,Np,Scheme,Ycut,Njet,Pjet,Error)
C=====================================================================
C Routine to calculate number of jet and jet momenta for a given Ycut
C
C Input : R*4 Pext(4,Np)   Array of external momenta
C         I*4 Np           Number of external particles
C         I*4 Scheme       Recombination Scheme
C                          Scheme = 0 : E0 - Scheme
C                          Scheme = 1 : E  - Scheme
C         R*4 Ycut         Ycut value
C Output: I*4 Njet         Number of Jets
C         R*4 Pjet(4,Njet) Array of Jet momenta
C         I*4 Error        Error = 0 : o.k.
C                          Error = 1 : to few particles
C                          Error = 2 : to many particles
C                          Error = 3 : division by zero
C
C Author: Guenther Dissertori / 15-Jun-1995
C---------------------------------------------------------------------

      Implicit NONE

      Integer   NpMax
      Parameter (NpMax = 300)   

C -- External variables
      Real    Pext(4,*),Pjet(4,*),Ycut
      Integer Np,Scheme,Njet,Error

C -- Internal Variables
      Real    P(4,NpMax),Y(NpMax,NpMax),Ymin
      Real    P2i,P2j,Cij,Evis,EnClus,PnClus
      Integer Iflag(NpMax),Iclus
      Integer i,j,Imin,Jmin,IminC

C=====================================================================

C -- Check if # of particles is o.k.
      
      Error = 0
      if (Np.lt.2) then
        Error = 1
        Return
      endif
      if (Np.gt.NpMax) then
        Error = 2
        Return
      endif

C -- Copy momenta to internal array, initialize
C -- rescale all momenta to zero mass in case E0
      
      Evis = 0.
      do i=1,Np
        Iflag(i) = 0
        Evis = Evis + Pext(4,i)
        EnClus = Pext(4,i)
        PnClus = sqrt(Pext(1,i)**2+Pext(2,i)**2+Pext(3,i)**2)
        do j=1,3
          if (Scheme.eq.0) P(j,i)=Pext(j,i)*EnClus/(max(1.e-7,PnClus))
          if (Scheme.eq.1) P(j,i)=Pext(j,i) 
        enddo
        P(4,i) = EnClus
      enddo


      Njet  = Np
      IminC = 0

C -- loop over # of jets

 100  continue
      if (Njet.lt.1) RETURN
      Ymin  = 100000.

C -- loop over combination of particles
      
      do i=1,Np-1
        if (Iflag(i).eq.1) goto 300
        do j=i+1,Np
          if (Iflag(j).eq.1) goto 200
C -- if nothing has changed for i and j
          if ((i.ne.IminC).and.(j.ne.IminC).and.(Njet.ne.Np)) then
            if (Y(i,j).lt.Ymin) then
              Ymin = Y(i,j)
              Imin = i
              Jmin = j
            endif
          else 
C -- else compute metric now
            Cij = P(1,i)*P(1,j)+P(2,i)*P(2,j)+P(3,i)*P(3,j)
            P2i = P(1,i)*P(1,i)+P(2,i)*P(2,i)+P(3,i)*P(3,i)
            P2j = P(1,j)*P(1,j)+P(2,j)*P(2,j)+P(3,j)*P(3,j)
            if ((P2i.le.1.e-14).or.(P2j.le.1.e-14)) then
              Y(i,j) = 0.
            else
              Y(i,j) = 2.*min((P(4,i)/Evis)**2,(P(4,j)/Evis)**2)*
     >                 (1. - Cij/(sqrt(P2i*P2j)))
            endif

            if (Y(i,j).lt.Ymin) then
              Ymin = Y(i,j)
              Imin = i
              Jmin = j
            endif
          endif
 200    enddo
 300  enddo

C -- compare Ymin with Ycut
      if (Ymin.gt.Ycut) then

C -- fill array of cluster momenta 
        i = 1
        Iclus = 1
 101    if (i.le.Np) then
          if (Iflag(i).eq.0) then
            do j=1,4
              Pjet(j,Iclus) = P(j,i)
            enddo

            Iclus = Iclus+1
          endif
          i=i+1
          goto 101
        endif

        Njet = Iclus-1
        goto 1000

      endif

C -- else recombine according to E-Scheme
      if (Scheme.eq.1) then
        do i=1,4
          P(i,Imin) = P(i,Imin) + P(i,Jmin)
        enddo
      endif

C -- recombine according to E0-Scheme
      if (Scheme.eq.0) then
        P(4,Imin) = P(4,Imin) + P(4,Jmin)
        P2i = sqrt((P(1,Imin) + P(1,Jmin))**2 +
     >             (P(2,Imin) + P(2,Jmin))**2 +
     >             (P(3,Imin) + P(3,Jmin))**2) 
        if (P2i.le.1.e-14) then
          Error = 3
          RETURN
        endif
        do i=1,3
          P(i,Imin) = P(4,Imin) * (P(i,Imin) + P(i,Jmin))/P2i
        enddo
      endif

C -- lock the other track
      Iflag(Jmin) = 1
      IminC = Imin
      Njet = Njet - 1

      goto 100

 1000 Continue
      
      END


C*********************************************************************
C*********************************************************************


      Subroutine EvShapes(Pext,Np,Miss,EvS,ThD,Error)
C=====================================================================
C Routine to return an array of eventshape-values
C
C Input : R*4 Pext(4,Np)   Array of external momenta
C         I*4 Np           Number of external particles
C         Log Miss      : .True. -> use also  missing momentum in Thrustcalc 
C                       : .False.-> disregard missing momentum
C
C Output: R*4 EvS(20)      Array of eventshape-values
C                          EvS(1) = Thrust
C                          EvS(2) = Th-Major
C                          EvS(3) = Th-Minor
C                          EvS(4) = Oblatness
C                          EvS(5) = C-Parameter
C                          EvS(6) = Heavy Jet Mass High
C                          EvS(7) = Heavy Jet Mass Low
C                          EvS(8) = Mass Difference
C                          EvS(9) = Total Jet Broadening
C                          EvS(10)= Wide Jet Broadening
C                          EvS(11)= D-Parameter
C                          EvS(12)= float(h) for heavy jet mass
C                          EvS(13)= float(h) for wide jet broad.
C                          EvS(14-20) not yet assigned
C
C         R*4 ThD(4,3)     Thrust and Major Directions, see subr ThrustDG
C         I*4 Error        Error = 0 : o.k.
C                          Error <>0 : a problem occured in Thrust         I
C         
C
C Author: Guenther Dissertori / 19-Dec-1994
C---------------------------------------------------------------------

      Implicit NONE
      
      Integer   NpMax
      Parameter (NpMax = 300)   

C -- External variables
      Real    Pext(4,*),EvS(20),ThD(4,3)
      Integer Np,Error
      Logical Miss

C -- Internal Variables
      Real    Dirs(4,3),ThruD(3),ExChange(4)
      Real    Thrus,Major,Minor,Oblat,Cpar,Dpar,mh2,ml2,md2,Btot,Bw
      Integer i,j,h1,h2

C=====================================================================      

      Error = 0

C -- first Thrust and related quantities
      call ThrustDG(Pext,Np,Dirs,Miss)
 
      Thrus = Dirs(4,1)
      Major = Dirs(4,2)
      Minor = Dirs(4,3)

      if ((Thrus.eq.-1.).or.(Thrus.eq.-2.).or.(Thrus.eq.-3.)) then
        Error = Nint(Thrus)
        RETURN
      endif

C -- Check against bad features of Lundroutine
C -- e.g. exchange Major and Minor if Major < Minor
      if (Major.lt.Minor) then
        do i=1,4
          ExChange(i) = Dirs(i,2)
          Dirs(i,2)   = Dirs(i,3)
          Dirs(i,3)   = ExChange(i)
        enddo
        Major = Dirs(4,2)
        Minor = Dirs(4,3)
      endif

      Oblat = Major - Minor
      do i = 1,3
        ThruD(i) = Dirs(i,1)
        do j = 1,4
          ThD(j,i) = Dirs(j,i)
        enddo
      enddo

C -- C-parameter
      call Cparameter(Pext,Np,Cpar,Dpar)

C -- Heavy-Jet Mass
      call HJetMass(Pext,Np,ThruD,mh2,ml2,md2,h1)

C -- Jet Broadening
      call HJetBroad(Pext,Np,ThruD,Btot,Bw,h2)

C -- store accumulated Results
      EvS(1) = Thrus
      EvS(2) = Major
      EvS(3) = Minor
      EvS(4) = Oblat
      EvS(5) = Cpar
      EvS(6) = mh2  
      EvS(7) = ml2  
      EvS(8) = md2  
      EvS(9) = Btot 
      EvS(10)= Bw    
      EvS(11)= Dpar
      EvS(12)= float(h1)
      EvS(13)= float(h2)

      RETURN
      END

C*********************************************************************
C*********************************************************************


      SUBROUTINE Fourjets(Pclus,Abz,Aks,Anr,A34,Error)
C-----------------------------------------------------------------------
C Routine which calculates 4-Jet related quantities
C ==========================================================
C
C  Input  Real*4  Pclus(4,*) :  4 vector of * jets  (* should be 4) 
C 
C  Output Real*4  Abz        :  Bengtsson-Zerwas   Angle in radian
C                 Aks        :  Koerner-Schierholz  --  "  --
C                 Anr        :  Nachtmann-Reiter
C                 A34        :  Angle between jet 3 and 4 in radian
C         Integer Error      :  0 if everything OK        
C
C Author: Guenther Dissertori  / 12.4.95
C-----------------------------------------------------------------------

      Implicit NONE

      Real       Pclus(4,*),Abz,Aks,Anr,A34
      Real       Eclus(4),P1(3),P2(3)
      Real       Aks1,Aks2

      Integer    i,j,k,l,IEO(4),Error

      Real       PI
      Parameter (PI=3.141592654)

      SAVE
C-----------------------------------------------------------------------

      Error = 1

C -- order Jetenergies 
      do i=1,4
        Eclus(i) = Pclus(4,i)

C -- check against pathological cases
        if (Eclus(i).le.0.) then
          write(6,*)' ** FOURJETS : Eclus = ',Eclus(i),' !!!!'
          RETURN 
        endif

      enddo
      call SORTZV(Eclus,IEO,4,1,1,0)


C -- calculate BZ-Angle
      i=IEO(1)
      j=IEO(2)

      P1(1) = Pclus(2,i)*Pclus(3,j)-Pclus(3,i)*Pclus(2,j)
      P1(2) = Pclus(3,i)*Pclus(1,j)-Pclus(1,i)*Pclus(3,j)
      P1(3) = Pclus(1,i)*Pclus(2,j)-Pclus(2,i)*Pclus(1,j)

      i=IEO(3)
      j=IEO(4)

      P2(1) = Pclus(2,i)*Pclus(3,j)-Pclus(3,i)*Pclus(2,j)
      P2(2) = Pclus(3,i)*Pclus(1,j)-Pclus(1,i)*Pclus(3,j)
      P2(3) = Pclus(1,i)*Pclus(2,j)-Pclus(2,i)*Pclus(1,j)

      Abz =        (P1(1)*P2(1)+P1(2)*P2(2)+P1(3)*P2(3))/
     >      sqrt(max(1.e-10,P1(1)**2+P1(2)**2+P1(3)**2))/
     >      sqrt(max(1.e-10,P2(1)**2+P2(2)**2+P2(3)**2))

      Abz = acos(max(min(Abz,1.),-1.))

      if (Abz.ge.(PI/2.)) Abz = PI - Abz


C -- calculate KS-Angle
      i=IEO(1)
      j=IEO(4)

      P1(1) = Pclus(2,i)*Pclus(3,j)-Pclus(3,i)*Pclus(2,j)
      P1(2) = Pclus(3,i)*Pclus(1,j)-Pclus(1,i)*Pclus(3,j)
      P1(3) = Pclus(1,i)*Pclus(2,j)-Pclus(2,i)*Pclus(1,j)

      i=IEO(2)
      j=IEO(3)
      P2(1) = Pclus(2,i)*Pclus(3,j)-Pclus(3,i)*Pclus(2,j)
      P2(2) = Pclus(3,i)*Pclus(1,j)-Pclus(1,i)*Pclus(3,j)
      P2(3) = Pclus(1,i)*Pclus(2,j)-Pclus(2,i)*Pclus(1,j)

      Aks1 =       (P1(1)*P2(1)+P1(2)*P2(2)+P1(3)*P2(3))/
     >      sqrt(max(1.e-10,P1(1)**2+P1(2)**2+P1(3)**2))/
     >      sqrt(max(1.e-10,P2(1)**2+P2(2)**2+P2(3)**2))

      Aks1 = acos(max(min(Aks1,1.),-1.))

      i=IEO(1)
      j=IEO(3)

      P1(1) = Pclus(2,i)*Pclus(3,j)-Pclus(3,i)*Pclus(2,j)
      P1(2) = Pclus(3,i)*Pclus(1,j)-Pclus(1,i)*Pclus(3,j)
      P1(3) = Pclus(1,i)*Pclus(2,j)-Pclus(2,i)*Pclus(1,j)

      i=IEO(2)
      j=IEO(4)

      P2(1) = Pclus(2,i)*Pclus(3,j)-Pclus(3,i)*Pclus(2,j)
      P2(2) = Pclus(3,i)*Pclus(1,j)-Pclus(1,i)*Pclus(3,j)
      P2(3) = Pclus(1,i)*Pclus(2,j)-Pclus(2,i)*Pclus(1,j)

      Aks2 =       (P1(1)*P2(1)+P1(2)*P2(2)+P1(3)*P2(3))/
     >      sqrt(max(1.e-10,P1(1)**2+P1(2)**2+P1(3)**2))/
     >      sqrt(max(1.e-10,P2(1)**2+P2(2)**2+P2(3)**2))

      Aks2 = acos(max(min(Aks2,1.),-1.))

      Aks = 0.5*(Aks1+Aks2)


C -- generalized NR-Angle
      i=IEO(1)
      j=IEO(2)

      P1(1) = Pclus(1,i)-Pclus(1,j)
      P1(2) = Pclus(2,i)-Pclus(2,j)
      P1(3) = Pclus(3,i)-Pclus(3,j)

      i=IEO(3)
      j=IEO(4)      

      P2(1) = Pclus(1,i)-Pclus(1,j)
      P2(2) = Pclus(2,i)-Pclus(2,j)
      P2(3) = Pclus(3,i)-Pclus(3,j)
      Anr =       (P1(1)*P2(1)+P1(2)*P2(2)+P1(3)*P2(3))/
     >      sqrt(max(1.e-10,P1(1)**2+P1(2)**2+P1(3)**2))/
     >      sqrt(max(1.e-10,P2(1)**2+P2(2)**2+P2(3)**2))

      Anr = acos(max(min(Anr,1.),-1.))

      if (Anr.ge.(PI/2.)) Anr = PI - Anr

     
C -- Angle between 3 and 4
      i=IEO(3)

      P1(1) = Pclus(1,i)
      P1(2) = Pclus(2,i)
      P1(3) = Pclus(3,i)

      i=IEO(4)

      P2(1) = Pclus(1,i)
      P2(2) = Pclus(2,i)
      P2(3) = Pclus(3,i)

      A34 =       (P1(1)*P2(1)+P1(2)*P2(2)+P1(3)*P2(3))/
     >      sqrt(max(1.e-10,P1(1)**2+P1(2)**2+P1(3)**2))/
     >      sqrt(max(1.e-10,P2(1)**2+P2(2)**2+P2(3)**2))

      A34 = acos(max(min(A34,1.),-1.))

      Error = 0

      RETURN

      END
