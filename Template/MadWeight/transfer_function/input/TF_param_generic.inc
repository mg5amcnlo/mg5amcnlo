c*********************************************************
c                  TF_param.inc                         **
c*********************************************************
      include 'phasespace.inc'
      
      integer curr_tf
      common/transfer_fct_curr_tf/curr_tf
      
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
c LHCO input
c
      integer tag_init(3:max_particles),type(max_particles),run_number,trigger
      double precision eta_init(max_particles),phi_init(max_particles),
     &pt_init(max_particles),j_mass(max_particles),ntrk(max_particles),
     &btag(max_particles),had_em(max_particles),dummy1(max_particles),
     &dummy2(max_particles)
      common/LHCO_input/eta_init,phi_init,pt_init,
     &j_mass,ntrk,btag,had_em,dummy1,dummy2,tag_init,type,run_number,
     &trigger

c
c	generated
c
c
c     global
c
      double precision momenta(0:3,-max_branches:2*max_particles)
      double precision mvir2(-max_branches:2*max_particles)    
      common /to_diagram_kin/ momenta, mvir2

      double precision permutation_weight
      common /to_permutation_weight/permutation_weight

c
c parameters to init in transfert_card.dat
c	
