	subroutine Anomaly(Q2, alpha, alphah, mu, muh, ptjmax,
     $ 		JETRADIUS, Efull)
	implicit none
c Q2 denotes the momentum squared (q1+q2)^2
	DOUBLE PRECISION, intent(in)  :: Q2, ptjmax, JETRADIUS
  	DOUBLE PRECISION, intent(in)  :: alpha, alphah, mu, muh         	
  	DOUBLE PRECISION, intent(out) :: Efull  
C
C  GLOBAL Parameters
C 
	include 'parameters.inc'
C-----
C  BEGIN Definition
C-----
  	DOUBLE PRECISION Fexp2, hexp2, LogU2
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
c 
	hexp2=-(alpha*gq0*Log(mu**2/ptjmax**2))/(4.*Pi) + 
     $  	(alpha*CF*Gamma0*Log(mu**2/ptjmax**2)**2)/(16.*Pi)
c
	LogU2=2*(-((alpha - alphah)*(-(beta1*gq0) + beta0*gq1))/
     $     (4.*beta0**2*Pi) - 
     $     (CF*(alpha**2*beta1**2*Gamma0 - 
     $     2*alpha*alphah*beta1**2*Gamma0 + 
     $     alphah**2*beta1**2*Gamma0 - 
     $     alpha**2*beta0*beta1*Gamma1 + 
     $     4*alpha*alphah*beta0*beta1*Gamma1 - 
     $     3*alphah**2*beta0*beta1*Gamma1 - 
     $     alpha**2*beta0*Gamma0*beta2 + 
     $     alphah**2*beta0*Gamma0*beta2 + alpha**2*beta0**2*Gamma2 - 
     $     2*alpha*alphah*beta0**2*Gamma2 + alphah**2*beta0**2*Gamma2 + 
     $     2*alpha*alphah*beta1**2*Gamma0*Log(alpha/alphah) - 
     $     2*alphah**2*beta1**2*Gamma0*Log(alpha/alphah) - 
     $     2*alpha*alphah*beta0*beta1*Gamma1*Log(alpha/alphah) + 
     $     2*alphah**2*beta0*Gamma0*beta2*Log(alpha/alphah)))/
     $     (16.*alphah*beta0**4*Pi) - 
     $     ((alpha - alphah)*CF*(-(beta1*Gamma0) + beta0*Gamma1)*
     $     Log(Q2/muh**2))/(8.*beta0**2*Pi)) + 
     $     2*(-((gq0*Log(alpha/alphah))/beta0) + 
     $     2*CF*(((-1 + alpha/alphah)*(beta1*Gamma0 - beta0*Gamma1))/
     $     (4.*beta0**3) + 
     $     ((-(beta1*Gamma0) + beta0*Gamma1)*Log(alpha/alphah))/
     $     (4.*beta0**3) + 
     $     (beta1*Gamma0*Log(alpha/alphah)**2)/(8.*beta0**3) + 
     $     (Gamma0*Pi*(-1 + alpha/alphah - 
     $     (alpha*Log(alpha/alphah))/alphah))/(alpha*beta0**2)
     $     ) - (CF*Gamma0*Log(alpha/alphah)*Log(Q2/muh**2))/
     $     (2.*beta0))
c 
	Efull=EXP((2*hexp2 + LogU2))/(Q2/ptjmax**2)**Fexp2
	end subroutine Anomaly


	subroutine AnomalyExp(Q2, alpha, mu, ptjmax, E1)
	implicit none
	DOUBLE PRECISION, intent(in)  :: Q2, alpha, mu, ptjmax        	
  	DOUBLE PRECISION, intent(out) :: E1  
C
C  GLOBAL Parameters
C 
	include 'parameters.inc'
C-----
C  BEGIN Definition
C-----
	E1=(alpha*(-4*gq0*Log(mu**2/ptjmax**2) + 
     $     CF*Gamma0*Log(mu**2/ptjmax**2)**2 - 
     $     2*CF*Gamma0*Log(mu**2/ptjmax**2)*Log(Q2/ptjmax**2)))/(8.*Pi)
	end subroutine AnomalyExp
