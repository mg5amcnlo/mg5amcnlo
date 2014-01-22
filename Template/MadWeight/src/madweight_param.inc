C	definition of variables for permutation	
C		definition also present in get_feynman_content.f
C
       logical use_perm,perm_with_b, montecarlo_perm
       common/global_run_perm/use_perm,perm_with_b, montecarlo_perm
C
C	definition of variables for the number of events in the integration
C		definition also present in driver.f
C
       integer nevents, nevents_refine
       common/number_integration_points/nevents,nevents_refine
C
C	definition of variables for the optimization of the driver
C		definition also present in driver.f
C
       integer max_it_step1,max_it_step2,max_it_step3,max_it_step4,
     &         max_it_step5
       common/number_of_iteration/max_it_step1,max_it_step2,
     &      max_it_step3,max_it_step4,max_it_step5
C
C       definition of variables for the optimization of the driver
C               definition also present in driver.f
C
       double precision min_prec_cut1,min_prec_cut2,value_cut1,
     &      value_cut2,final_prec, min_perm_cut
       common/global_precision/ min_prec_cut1,min_prec_cut2,value_cut1,
     &      value_cut2,final_prec, min_perm_cut
C
C   variable for use histogram
C	 	
      logical histo
      common/to_histo/histo
c
c   narrow width approximation/optimization PS
c
      double precision NWA
      logical pretrained
      common /to_NWA/NWA,pretrained
c
c     ISR mode
c
      integer integrator
      common /to_integrator/integrator
      integer ISR
      common /to_correct_ISR/ISR
c
c     CUTS
c
      logical use_cut, bw_cut
      common /to_use_cut/ use_cut, bw_cut
c
c     activate / desactivate sobol generator
c
      logical use_sobol
      common /to_use_sobol/use_sobol


