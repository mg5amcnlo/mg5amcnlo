	double precision Pi, Zeta3, CA, CF, TF, nf
	double precision Gamma0, Gamma1, beta0, beta1, gq0
	double precision Gamma2, beta2, gq1
	parameter (Pi=3.1415926535897932385d0)
	parameter (Zeta3=1.20205690315959428539973d0)
  	parameter (CA=3.d0)
	parameter (CF=4/3.d0)
	parameter (TF=1/2.d0)
	parameter (nf=5.)
	parameter (Gamma0=4.)
	parameter (Gamma1=4*(CA*(67/9. - Pi**2/3.) - (20*nf*TF)/9.))
	parameter (Gamma2=16*(-nf**2/27. - CF*nf*(2.2916666666666665 
     $    - 2*Zeta3) + CA**2*(10.208333333333334 - (67*Pi**2)/54. + 
     $    (11*Pi**4)/180. + (11*Zeta3)/6.) - 
     $    CA*nf*(1.9351851851851851 - (5*Pi**2)/27. + (7*Zeta3)/3.)) )
	parameter (beta0=(11*CA)/3. - (4*nf*TF)/3.)
	parameter (beta1=(34*CA**2)/3. - (20*CA*nf*TF)/3. - 4*CF*nf*TF)
	parameter (beta2= (2857*CA**3)/54. + ((-1415*CA**2)/54. -  
     $    (205*CA*CF)/18. +CF**2)*nf + ((79*CA)/54. + (11*CF)/9.)*nf**2)
	parameter (gq0=-3*CF)
	parameter (gq1=CF*nf*(4.814814814814815 + (2*Pi**2)/3.)*TF + 
     $    CF**2*(-1.5 + 2*Pi**2 - 24*Zeta3) + 
     $    CA*CF*(-17.796296296296298 - (11*Pi**2)/6. + 26*Zeta3))
