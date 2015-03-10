	subroutine Anomaly(Q2, alpha, alphah, mu, muh, ptjmax,
     $ 		JETRADIUS, Efull)
c This is Eq.(8) of arXiv:1412.8408
	implicit none
c Q2 denotes the momentum squared (q1+q2)^2
	DOUBLE PRECISION, intent(in)  :: Q2, ptjmax, JETRADIUS
  	DOUBLE PRECISION, intent(in)  :: alpha, alphah, mu, muh         	
  	DOUBLE PRECISION, intent(out) :: Efull  
C
C  GLOBAL Parameters
C 
	include 'veto_xsec.inc'
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
c This is the O(alphaS) expansion of Eq.(8) of arXiv:1412.8408
	implicit none
	DOUBLE PRECISION, intent(in)  :: Q2, alpha, mu, ptjmax        	
  	DOUBLE PRECISION, intent(out) :: E1  
C
C  GLOBAL Parameters
C 
	include 'veto_xsec.inc'
C-----
C  BEGIN Definition
C-----
	E1=(alpha*(-4*gq0*Log(mu**2/ptjmax**2) + 
     $     CF*Gamma0*Log(mu**2/ptjmax**2)**2 - 
     $     2*CF*Gamma0*Log(mu**2/ptjmax**2)*Log(Q2/ptjmax**2)))/(8.*Pi)
	end subroutine AnomalyExp



      
      subroutine compute_veto_compensating_factor(H1_factor_virt,
     $     born_wgt,muSoft,muHard,veto_compensating_factor)
c 2nd term on 3rd line of eq.(20) of arXiv:1412.8408
      implicit none
      include 'q_es.inc'
      include 'coupl.inc'
      include 'cuts.inc'
	double precision H1_factor_virt,born_wgt,veto_compensating_factor
     &     ,muSoft,muHard
      double precision Q2,ptjmax,mu,alpha,E1,H1_factor,muMad,alphah
     $     ,Q,muh,Efull,H1_comp,alphas
      external alphas
      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat
      integer izero
      parameter (izero=0)
      double precision pi
      parameter (pi=3.1415926535897932385d0)
      call set_cms_stuff(izero)
      Q=sqrtshat
      Q2=shat
      ptjmax=ptj
      mu=ptjmax*muSoft
      if (abs(QES2-ptjmax**2).gt.1d-7) then
         write (*,*) 'ERROR in VETO XSec: Ellis-Sexton '/
     $        /'scale should be equal to the veto scale',QES2
     $        ,ptjmax**2
         stop
      endif
      muMad=sqrt(QES2)
      alpha=alphas(mu)
      call AnomalyExp(Q2, alpha, mu, ptjmax, E1)
c compensating factor for difference between muMad and the soft scale mu
      H1_comp=(2d0*(Pi**2 + 24d0*Log(muMad/mu)**2 + Log(muMad/mu)*(36d0
     $     - 48d0*Log(Q/mu))))/9d0
      veto_compensating_factor=(H1_factor_virt + H1_comp
     $     /(2d0*pi) + E1/alpha) * born_wgt / (4d0*pi)

      return
      end

      subroutine compute_veto_multiplier(H1_factor_virt,muSoft,muHard
     &     ,veto_multiplier)
c 2nd line of eq.(20) of arXiv:1412.8408
      implicit none
      include 'nexternal.inc'
      include 'q_es.inc'
      include 'coupl.inc'
      include 'cuts.inc'
      double precision muSoft,muHard,H1_factor_virt,veto_multiplier
      double precision Q2,ptjmax,mu,alpha,E1,H1_factor,muMad,alphah
     $     ,Q,muh,Efull,H1_comp,alphas
      external alphas
      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat
      double precision p1_cnt(0:3,nexternal,-2:2)
      double precision wgt_cnt(-2:2)
      double precision pswgt_cnt(-2:2)
      double precision jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
      integer izero
      parameter (izero=0)
      double precision pi
      parameter (pi=3.1415926535897932385d0)
c     set sqrt(\hat(s)) correctly to be the one of the n-body kinematics
      call set_cms_stuff(izero)
      Q=sqrtshat
      Q2=shat
      ptjmax=ptj
c     set muMad to be the ren scale that was used in the virtual
      call set_alphaS(p1_cnt(0,1,0))
      if (abs(QES2-ptjmax**2).gt.1d-7) then
         write (*,*) 'ERROR in VETO XSec: Ellis-Sexton '/
     $        /'scale should be equal to the veto scale',QES2
     $        ,ptjmax**2
         stop
      endif
      muMad=sqrt(QES2)
      muh=sqrt(Q2)*muHard       ! hard scale
      mu=ptjmax*muSoft          ! soft scale
      alpha=alphas(mu)
      alphah=alphas(muh)
c     compensating factor for difference between muMad and the hard
c     scale muh
      H1_comp=(2d0*(Pi**2 + 24d0*Log(muMad/muh)**2 + Log(muMad/muh)
     $     *(36d0 - 48d0*Log(Q/muh))))/9d0
c     (first order of) the Hard function
      H1_factor=H1_factor_virt + H1_comp/(2d0*pi)
      call Anomaly(Q2, alpha, alphah, mu, muh, ptjmax, JETRADIUS, Efull)
      veto_multiplier=(1d0+alphah*H1_factor)*Efull
      return
      end
