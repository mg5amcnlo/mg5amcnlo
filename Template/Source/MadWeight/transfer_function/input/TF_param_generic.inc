c*********************************************************
c                  TF_param.inc                         **
c*********************************************************
      include 'phasespace.inc'
c
c external functions
c
        double precision pt,eta,phi,theta,rap,rho
	external pt,eta,phi,theta,rap,rho
c
c local
c
	double precision prov1,prov2,prov3,prov4,prov5
	double precision prov6,prov7,prov8,prov9,prov10
c
c	MET tag
c
	integer met_lhco,opt_lhco
	common/LHCO_met_tag/met_lhco,opt_lhco
c
c	generated
c
c
c     global
c
      double precision momenta(0:3,-max_branches:2*max_particles)
      double precision mvir2(-max_branches:2*max_particles)    
      common /to_diagram_kin/ momenta, mvir2
c
c parameters to init in transfert_card.dat
c	
