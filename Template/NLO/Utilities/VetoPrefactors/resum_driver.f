      PROGRAM example
      IMPLICIT NONE
      INTEGER iset,f1,f2,alphaSorder
      DOUBLE PRECISION x1,x2,mu,muh,muMad,ptjmax,BCorr,GetOnePDF,tst
      DOUBLE PRECISION alphaSMZ, ALPHAS, alphaSQ0,mCharm,mBottom
      DOUBLE PRECISION Q,Q2, JETRADIUS, alpha, alphah, Efull, E1
      CHARACTER prefix*50
      double precision Hcompensationfactor,pi
      parameter (pi=3.14159265358979324d0)

      prefix = "Grids/BeRn2014NNLO" ! prefix for the grid files

c     (some example values to test)
c     These should be read out of the event
c
 100  continue

c     Flavors (in PDG notation) 
c$$$      f1 = -1
c$$$      f2 = 1
c     momentum fractions
c$$$      x1 = 0.00482046d0
c$$$      x2 = 0.035196d0
      
c$$$      Q= 2.0d0*79.0d0

c     Scale chosen in Madgraph can be left fixed
c     Only the PDFs depend on it; dependence goes away when reweighting 
c     with beam functions
c$$$      muMad = 91.1778d0
      read (*,*) f1,f2,x1,x2,Q,muMad
      Q2 = Q*Q

C     Parameters for the jet veto
      JETRADIUS = 0.5d0
      ptjmax = 20.d0


c     Renormalization/factorization scales

c     Low scale, associated with the emissions
      mu = 2.0d0*ptjmax

c     High scale, should be of order Q
      muh = Q


C     MSTW alpha_s routine

      mcharm=1.5
      mbottom=4.25
      alphasMZ=0.11707d0
      CALL INITALPHAS(2,1.D0,91.1876D0,alphasMZ,
     &     mcharm,mbottom,1.D10)


      alpha = ALPHAS(mu)
      alphah = ALPHAS(muh)

      call Anomaly(Q2, alpha, alphah, mu, muh, ptjmax, 
     $		JETRADIUS, Efull)
      call AnomalyExp(Q2, alpha, mu, ptjmax, E1)
      call BeamCorr(x1,x2,mu,muMad,f1,f2,ptjmax,alpha,BCorr)

c$$$      WRITE(6,*) "BeamCorrection = ", BCorr
c$$$      WRITE(6,*) "Efull = ", Efull
c$$$      WRITE(6,*) "E1 = ", E1
      Hcompensationfactor = (2d0*(Pi**2 + 24d0*Log(muMad/muh)**2 +
     $     Log(muMad/muh)*(36d0 - 48d0*Log(Q/muh))))/9d0

      write (*,*) Bcorr*Efull,Hcompensationfactor,alphah
      goto 100
C----------------------------------------------------------------------
      END
C----------------------------------------------------------------------
