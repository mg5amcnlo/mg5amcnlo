	subroutine Anomaly(Q2, alpha, alphah, mu, muh, ptjmax,
     $ 		JETRADIUS, Efull)
c This is Eq.(8) of arXiv:1412.8408
	implicit none
c Q2 denotes the momentum squared (q1+q2)^2
	DOUBLE PRECISION, intent(in)  :: Q2, ptjmax, JETRADIUS
  	DOUBLE PRECISION, intent(in)  :: alpha, mu
        COMPLEX, intent(in) ::  alphah, muh
  	DOUBLE PRECISION, intent(out) :: Efull  
C
C  GLOBAL Parameters
C 
	include 'parameters.inc'
C-----
C  BEGIN Definition
C-----
  	DOUBLE PRECISION Fexp2, hexp2
        COMPLEX LogU2, eps
        eps=(0.0,0.000000000000000001d0)
C-----
	Fexp2= alpha/(4.*Pi)*CF*Gamma0*Log(mu**2/ptjmax**2) + 
     $     	 (alpha/(4.*Pi))**2*(-88.07485937057658*CF -
     $		27.10420771393072*CF*JETRADIUS**2 + 
     $     	26.31894506957162*CF**2*JETRADIUS**2 + 
     $     	2.2030236514296107*CF*JETRADIUS**4 -
     $		2.*CF**2*JETRADIUS**4 - 
     $     	0.05627657806902043*CF*JETRADIUS**6 + 
     $     	0.0015419253059560723*CF*JETRADIUS**8 - 
     $     	0.0001084140443080469*CF*JETRADIUS**10 + 
     $     	119.3841690105608*CF*Log(JETRADIUS) + 
     $		CF*Gamma1*Log(mu**2/ptjmax**2) + 
     $     	(CF*beta0*Gamma0*Log(mu**2/ptjmax**2)**2)/2.)
C----- 
	hexp2=-(alpha*gq0*Log(mu**2/ptjmax**2))/(4.*Pi) + 
     $  	(alpha*CF*Gamma0*Log(mu**2/ptjmax**2)**2)/(16.*Pi)
C-----
	LogU2= -((alpha - alphah)*(-(beta1*gq0) + beta0*gq1))/
     -   (4.*beta0**2*Pi) - ((-(beta1*gq0) + beta0*gq1)*
     -     (alpha - CONJG(alphah)))/(4.*beta0**2*Pi) - 
     -  (gq0*Log(alpha/alphah))/beta0 - 
     -  (CF*(alpha**2*beta1**2*Gamma0 - 
     -       2*alpha*alphah*beta1**2*Gamma0 + 
     -       alphah**2*beta1**2*Gamma0 - 
     -       alpha**2*beta0*beta2*Gamma0 + 
     -       alphah**2*beta0*beta2*Gamma0 - 
     -       alpha**2*beta0*beta1*Gamma1 + 
     -       4*alpha*alphah*beta0*beta1*Gamma1 - 
     -       3*alphah**2*beta0*beta1*Gamma1 + 
     -       alpha**2*beta0**2*Gamma2 - 
     -       2*alpha*alphah*beta0**2*Gamma2 + 
     -       alphah**2*beta0**2*Gamma2 + 
     -       2*alpha*alphah*beta1**2*Gamma0*Log(alpha/alphah) - 
     -       2*alphah**2*beta1**2*Gamma0*Log(alpha/alphah) + 
     -       2*alphah**2*beta0*beta2*Gamma0*Log(alpha/alphah) - 
     -       2*alpha*alphah*beta0*beta1*Gamma1*Log(alpha/alphah)))/
     -   (16.*alphah*beta0**4*Pi) + 
     -  2*CF*(((-1 + alpha/alphah)*(beta1*Gamma0 - beta0*Gamma1))/
     -      (4.*beta0**3) + ((-(beta1*Gamma0) + beta0*Gamma1)*
     -        Log(alpha/alphah))/(4.*beta0**3) + 
     -     (beta1*Gamma0*Log(alpha/alphah)**2)/(8.*beta0**3) + 
     -     (Gamma0*Pi*(-1 + alpha/alphah - 
     -          (alpha*Log(alpha/alphah))/alphah))/(alpha*beta0**2))-
     -   ((alpha - alphah)*CF*(-(beta1*Gamma0) + beta0*Gamma1)*
     -     Log((-eps - Q2)/muh**2))/(8.*beta0**2*Pi) - 
     -  (CF*Gamma0*Log(alpha/alphah)*Log((-eps - Q2)/muh**2))/
     -   (2.*beta0) - (CF*(-(beta1*Gamma0) + beta0*Gamma1)*
     -     (alpha - CONJG(alphah))*Log((eps - Q2)/muh**2))/
     -   (8.*beta0**2*Pi) - (gq0*Log(alpha/CONJG(alphah)))/beta0 - 
     -  (CF*Gamma0*Log((eps - Q2)/muh**2)*Log(alpha/CONJG(alphah)))/
     -   (2.*beta0) - (CF*(alpha**2*beta1**2*Gamma0 - 
     -       alpha**2*beta0*beta2*Gamma0 - 
     -       alpha**2*beta0*beta1*Gamma1 + alpha**2*beta0**2*Gamma2 - 
     -       2*alpha*beta1**2*Gamma0*CONJG(alphah) + 
     -       4*alpha*beta0*beta1*Gamma1*CONJG(alphah) - 
     -       2*alpha*beta0**2*Gamma2*CONJG(alphah) + 
     -       beta1**2*Gamma0*CONJG(alphah)**2 + 
     -       beta0*beta2*Gamma0*CONJG(alphah)**2 - 
     -       3*beta0*beta1*Gamma1*CONJG(alphah)**2 + 
     -       beta0**2*Gamma2*CONJG(alphah)**2 + 
     -       2*alpha*beta1**2*Gamma0*CONJG(alphah)*
     -        Log(alpha/CONJG(alphah)) - 
     -       2*alpha*beta0*beta1*Gamma1*CONJG(alphah)*
     -        Log(alpha/CONJG(alphah)) - 
     -       2*beta1**2*Gamma0*CONJG(alphah)**2*
     -        Log(alpha/CONJG(alphah)) + 
     -       2*beta0*beta2*Gamma0*CONJG(alphah)**2*
     -        Log(alpha/CONJG(alphah))))/
     -   (16.*beta0**4*Pi*CONJG(alphah)) + 
     -  2*CF*(((beta1*Gamma0 - beta0*Gamma1)*
     -        (-1 + alpha/CONJG(alphah)))/(4.*beta0**3) + 
     -     ((-(beta1*Gamma0) + beta0*Gamma1)*
     -        Log(alpha/CONJG(alphah)))/(4.*beta0**3) + 
     -     (beta1*Gamma0*Log(alpha/CONJG(alphah))**2)/(8.*beta0**3) + 
     -     (Gamma0*Pi*(-1 + alpha/CONJG(alphah) - 
     -          (alpha*Log(alpha/CONJG(alphah)))/CONJG(alphah)))/
     -      (alpha*beta0**2))
c 
	Efull=EXP((2*hexp2 + REAL(LogU2)))/(Q2/ptjmax**2)**Fexp2
	end subroutine Anomaly

	subroutine AnomalyLO(Q2, alpha, alphah, mu, muh, ptjmax,
     $ 		JETRADIUS, EfullLO)
	implicit none
c Q2 denotes the momentum squared (q1+q2)^2
	DOUBLE PRECISION, intent(in)  :: Q2, ptjmax, JETRADIUS
  	DOUBLE PRECISION, intent(in)  :: alpha, mu
        COMPLEX, intent(in) ::  alphah, muh        	
  	DOUBLE PRECISION, intent(out) :: EfullLO  
C
C  GLOBAL Parameters
C 
	include 'parameters.inc'
C-----
C  BEGIN Definition
C-----
  	DOUBLE PRECISION Fexp2LO, hexp2LO
        COMPLEX LogU2LO, eps
        eps=(0.0,0.000000000000000001d0)
C----
	Fexp2LO= alpha/(4.*Pi)*CF*Gamma0*Log(mu**2/ptjmax**2)
C---- 
	hexp2LO=0
C----
	LogU2LO=   -((gq0*Log(alpha/alphah))/beta0) + 
     -   2*CF*(((-1 + alpha/alphah)*(beta1*Gamma0 - beta0*Gamma1))/
     -   (4.*beta0**3) + ((-(beta1*Gamma0) + beta0*Gamma1)*
     -   Log(alpha/alphah))/(4.*beta0**3) + 
     -   (beta1*Gamma0*Log(alpha/alphah)**2)/(8.*beta0**3) + 
     -   (Gamma0*Pi*(-1 + alpha/alphah - 
     -   (alpha*Log(alpha/alphah))/alphah))/(alpha*beta0**2))-
     -   (CF*Gamma0*Log(alpha/alphah)*Log((-eps - Q2)/muh**2))/
     -   (2.*beta0) - (gq0*Log(alpha/CONJG(alphah)))/beta0 - 
     -   (CF*Gamma0*Log((eps - Q2)/muh**2)*Log(alpha/CONJG(alphah)))/
     -   (2.*beta0) + 2*CF*(((beta1*Gamma0 - beta0*Gamma1)*
     -   (-1 + alpha/CONJG(alphah)))/(4.*beta0**3) + 
     -   ((-(beta1*Gamma0) + beta0*Gamma1)*
     -    Log(alpha/CONJG(alphah)))/(4.*beta0**3) + 
     -   (beta1*Gamma0*Log(alpha/CONJG(alphah))**2)/(8.*beta0**3) + 
     -   (Gamma0*Pi*(-1 + alpha/CONJG(alphah) - 
     -   (alpha*Log(alpha/CONJG(alphah)))/CONJG(alphah)))/
     -   (alpha*beta0**2))
c 
	EfullLO=EXP((2*hexp2LO + REAL(LogU2LO)))/(Q2/ptjmax**2)**Fexp2LO
	end subroutine AnomalyLO



	subroutine AnomalyExp(Q2, alpham, mum, mu, muh, ptjmax,
     $           E1NNLL, E1NLL)
	implicit none
	DOUBLE PRECISION, intent(in)  :: Q2, alpham, mu, ptjmax, mum
        COMPLEX, intent(in) ::  muh        	                	
  	DOUBLE PRECISION, intent(out) :: E1NNLL, E1NLL
C
C  GLOBAL Parameters
C 
        include 'parameters.inc'
C-----
C  BEGIN Definition
C-----
        COMPLEX E1NLLComp, eps
        eps=(0.0,0.000000000000000001d0)
c
        E1NNLL=(alpham*(-4*gq0*Log(mum**2/ptjmax**2)
     $   + CF*Gamma0*Log(mum**2/ptjmax**2)**2 - 
     $   2*CF*Gamma0*Log(mum**2/ptjmax**2)*Log(Q2/ptjmax**2)))/(8.*Pi)
c
        E1NLLComp=(alpham*(-2*gq0*CONJG(Log(muh/mum)) - 
     -      CF*Gamma0*CONJG(Log(muh/mum))**2 + 4*gq0*Log(mu/mum) + 
     -      2*CF*Gamma0*CONJG(Log(muh/mum))*Log(mu/mum) - 
     -      2*CF*Gamma0*Log(mu/mum)**2 - 2*gq0*Log(muh/mum) + 
     -      2*CF*Gamma0*Log(mu/mum)*Log(muh/mum) - 
     -      CF*Gamma0*Log(muh/mum)**2 - 
     -      CF*Gamma0*CONJG(Log(muh/mum))*Log((eps - Q2)/muh**2) + 
     -      CF*Gamma0*Log(mu/mum)*Log((eps - Q2)/muh**2) - 
     -      CF*Gamma0*Log(mu**2/ptjmax**2)*Log(Q2/ptjmax**2) + 
     -      CF*Gamma0*Log(mu/mum)*Log(-((eps + Q2)/muh**2)) - 
     -      CF*Gamma0*Log(muh/mum)*Log(-((eps + Q2)/muh**2))))/(4.*Pi)
c
        E1NLL=REAL(E1NLLComp)
	end subroutine AnomalyExp
