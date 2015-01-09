mw_run  nb_exp_events   integer  10                     # number of experimental events         
mw_run  MW_int_points   integer 10000                  # number of points for MadWeight integration
mw_run  use_cut         logical  F                      # use the cut defined in run_card.dat for weight computation
mw_run  22              logical  0                      # put on 1 to add a new events in the analysis (for step 5)
mw_run  bw_cut          logical  T
mw_run  histo           logical F                       # put one creates histogram for value defined in topgraph.f
mw_run  histo_by_events logical F                       # decide if we create the histograms for each events
mw_run  nb_event_by_node integer  10                    # decide how many events weight are computed in a single cluster submission
mw_run  event_packing    integer  20                    # the input event files will be splitted in event_packing*nb_event_by_node 
mw_run  log_level        string   'channel'             # from low level of log to extensive log: weight, permutation, channel, iteration, full     
mw_run  nwa              float    0.1                   # width below NWA is used instead of performing the actual integrals
mw_run  integrator       integer  0                   # default integrator method (0/1 -> vegas/mint)
mw_run  pretrained_tf    logical   T                    # if we pre-trained the grid for the TF
mw_run  use_sobol        logical   F                    # don t use sobol generator by default
mw_run  inputfile        string   './Events/input.lhco' # path to the inputfile

mw_parameter mode integer 1                         
mw_parameter 1    integer mw_parameter@@mode # choose the mode for the generation of the param_card.dat
mw_parameter 2    logical 0                  # put on 1 to add new param_card.dat (for step 1) creates the corresponding new events dir if step 5 already performs

mw_perm  permutation    logical T                       # make the permutation between identical particles/jets
mw_perm  bjet_is_jet    logical T                       #consider bjet like normal jet for permutation
mw_perm  bjet_is_jet_for_selection  logical mw_perm@@bjet_is_jet  #don't use btagging for selceting events (follow precedent rules)
mw_perm  preselect      string   'default'              # How to pre-select permutation (None/default/path)
mw_perm  min_perm_cut   float    5e-4                   # permutation to remove if relative weight lower than
mw_perm  combine_mode    integer 1        # how to combine the permutation 1: average
                                          #                                2: first permutation 
                                          #                                3: higher permutation only
mw_perm  montecarlo     logical T         # Perform the MC over permutation or not...

       
mw_gen     1         integer 1            # choose which propagator to allign in ECS:
                                # 0: favours the  BW at the very end of the decay chain
                                # 1: favours the thinest BW
mw_gen     2         integer 1            # Same condition but for blobs
mw_gen     3         integer 4            # maximal multiplicity for ECS sector
# Preference in blob generation (If more than one "True", we will perform a multi-channel techniques)
mw_gen     4         logical F            #      maximizes  the number of aligned BW
mw_gen     5         logical T            #      choose all momenta as  variables of integration
mw_gen     6         logical T            #      mixed choice between the two above options
mw_gen     10        logical T            # Authorizes ECS A  (0 neut)
mw_gen     11        logical T            # Authorizes ECS B  (1 neut)
mw_gen     12        logical T            # Authorizes ECS C  (1neut+1 visible(m=0))
mw_gen     13        logical T            # Authorizes ECS D  (2 neut ttbar-like)
mw_gen     14        logical T            # Authorizes ECS E  (2 neut HWW-like)
mw_gen     15        logical T            # Authorizes ECS F  (2 neut [WW in T-channel]-like)
mw_gen     force_nwa float   2.0          # width below the mass should ALWAYS be align.

