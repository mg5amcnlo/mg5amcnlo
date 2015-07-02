      PROGRAM example
      IMPLICIT NONE
      INTEGER iset,f1,f2,alphaSorder
      DOUBLE PRECISION x1,x2,GetOnePDF,tst
      DOUBLE PRECISION BCorr0, BCorr1, BCorrm0, BCorrm1
      DOUBLE PRECISION mu, muMad, muhREAL, muh2, mum
      COMPLEX muh, alphaC, alphamC, alphah, ALPHASCOMP
      DOUBLE PRECISION alphaSMZ, ALPHAS, alphaSQ0,mCharm,mBottom
      DOUBLE PRECISION alpha,alpham,alphahREAL
      DOUBLE PRECISION Q, Q2, Efull, E1NNLL, E1NLL, EfullLO
      CHARACTER prefix*50
      DOUBLE PRECISION Hcompensationfactor,Hcompfactorm
      DOUBLE PRECISION HfactREALmuh,HfactCOMPLEXmuh,pi
      parameter (pi=3.14159265358979324d0)
C     Parameters for the jet veto
      include 'JetVetoParameters.inc'

      prefix = "Grids/BeRn2014NNLO" ! prefix for the grid files
c
 100  continue
c
c     Scale chosen in Madgraph can be left fixed
c     Only the PDFs depend on it; dependence goes away when reweighting 
c     with beam functions
c
c------TEST values---------------------------------------
c
c        Flavors (in PDG notation) 
c         f1 = -1
c         f2 = 1
c        momentum fractions
c         x1 = 0.00482046d0
c         x2 = 0.035196d0
c        Invariant Mass
c         Q= 91.1778d0
c        Scale 
c         muMad = 91.1778d0
c------------------------------------------------------
      read (*,*) f1,f2,x1,x2,Q,muMad
      Q2 = Q*Q
c
c-----Renormalization/factorization scales----------------------
c     Low scale, associated with the emissions
      mu = MuScalefactor*ptjmax

c     High scale, should be of order Q (REAL or IMAGINARY)
      muh = MuHScalefactor*Q
      muhREAL=REAL(muh)
      muh2=ABS(REAL(muh**2))
      mum= MuMScalefactor*Q ! matching scale
c     
C-----MSTW alpha_s routine-----------------------------------------
      mcharm=1.5
      mbottom=4.25
      alphasMZ=0.11707d0
      CALL INITALPHAS(2,1.D0,91.1876D0,alphasMZ,
     &     mcharm,mbottom,1.D10)
c
C-----Compensation Factors for the Hard function-------------------
c     Principal value prescription for imaginary mu_h choice
c
      HfactREALmuh=(2d0*(pi**2 + 18d0*Log(muMad**2/Q2) +  
     -   6d0*Log(muMad**2/Q2)**2 + 18d0*Log(Q2/muh2) - 
     -   6d0*Log(Q2/muh2)**2))/9d0
      HfactCOMPLEXmuh = (2d0*(-5d0*pi**2 + 18d0*Log(muMad**2/Q2) +  
     -   6d0*Log(muMad**2/Q2)**2 + 18d0*Log(Q2/muh2) - 
     -   6d0*Log(Q2/muh2)**2))/9d0
      Hcompfactorm = (2d0*(pi**2 + 18d0*Log(muMad**2/Q2) +  
     -   6d0*Log(muMad**2/Q2)**2 + 18d0*Log(Q2/mum**2) - 
     -   6d0*Log(Q2/mum**2)**2))/9d0
c
c-----Different Cases depending on muh REAL or IMAGINARY----
      IF (AIMAG(muh) .EQ. 0) THEN
      	alphaC = ALPHAS(mu)
        alphamC = ALPHAS(mum)
      	alphah = ALPHAS(muhREAL)
        Hcompensationfactor=HfactREALmuh
      ELSE
      	alphaC = ALPHASCOMP(CMPLX(mu,0.0))
        alphamC = ALPHASCOMP(CMPLX(mum,0.0))
      	alphah = ALPHASCOMP(muh)
        Hcompensationfactor=HfactCOMPLEXmuh
      END IF
      alpha=REAL(alphaC)
      alpham=REAL(alphamC)
      alphahREAL=REAL(alphah)
c-----Call various functions-------------------------------------
      call Anomaly(Q2, alpha, alphah, mu, muh, ptjmax, 
     $		JETRADIUS, Efull)
      call AnomalyLO(Q2, alpha, alphah, mu, muh, ptjmax, 
     $		JETRADIUS, EfullLO)
      call AnomalyExp(Q2, alpham, mum, mu, muh, ptjmax,
     $          E1NNLL, E1NLL)
      call BeamCorr(x1, x2, mu, mum, muMad, f1, f2, 
     $           ptjmax, alpha, alpham, 
     $           BCorr0, BCorrm0, BCorr1, BCorrm1)
c
c$$$      WRITE(6,*) "BeamCorrection = "
c$$$      WRITE(6,*) "Efull = ", Efull
c$$$      WRITE(6,*) "a*Hcomp=", alphahREAL*Hcompensationfactor
c
c
      write (*,*) mode, Bcorr0, Bcorr1, 
     $            Bcorrm0, Bcorrm1,
     $            Efull, EfullLO,
     $            Hcompensationfactor,
     $            Hcompfactorm,
     $            alphahREAL, alpham,
     $            E1NNLL, E1NLL, alpha
      goto 100
C----------------------------------------------------------------------
      END
C----------------------------------------------------------------------


      COMPLEX FUNCTION ALPHASCOMP(mu)
      IMPLICIT NONE
C
C  GLOBAL Parameters
C 
      include 'parameters.inc'
      COMPLEX mu
      ALPHASCOMP=4*Pi*(1/(beta0*Log(25.587705545274503*mu**2))- 
     &    (beta1*Log(Log(25.587705545274503*mu**2)))/
     &     (beta0**3*Log(25.587705545274503*mu**2)**2)+ 
     &    (beta2/beta0 + (beta1**2*
     &          (-1 - Log(Log(25.587705545274503*mu**2))+ 
     &            Log(Log(25.587705545274503*mu**2))**2))/beta0**2)/
     &     (beta0**3*Log(25.587705545274503*mu**2)**3))
C
      END
