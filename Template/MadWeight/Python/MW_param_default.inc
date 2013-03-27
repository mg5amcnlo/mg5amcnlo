mw_run  cluster         integer  0                      # cluster type
mw_run  1               integer  mw_run@@cluster
mw_run  queue           string   ''                     # queue restriction                    
mw_run  packet          integer  1                      # packet submission
mw_run  max_submission  integer  999999999999           # maximal jobs submitted in one packet [PBS cluster only]
mw_run  11              string   mw_run@@queue     
mw_run  nb_exp_events   integer  10                     # number of experimental events
mw_run  2               integer  mw_run@@nb_exp_events          
mw_run  3               logical  F                      # control efficiency
mw_run  write_log       logical  mw_run@@3              # write the log/out/err for each job
mw_run  normalize       logical  F                      # normalize wegiht by cross section #NOT SUPPORTED !!!
mw_run  4               logical  mw_run@@normalize
mw_run  acceptance_run  logical  F                      # compute the acceptance run 
mw_run  ME_int_points   interger 10000                  # number of points for MadEvent integration
mw_run  5               integer  mw_run@@ME_int_points     
mw_run  MW_int_points   integer 10000                  # number of points for MadWeight integration
mw_run  6               integer  mw_run@@MW_int_points   
mw_run  accuracy_cross  float    0.001                  # accuracy asked for the cross section computation
mw_run  nb_loop_cross   integer  3                      # number of loop in order to be sure of the precision of the cross section
mw_run  use_cut         logical  F                      # use the cut defined in run_card.dat for weight computation / cross-section
mw_run  p_cut           logical  mw_run@@use_cut        # use the cut defined in run_card.dat for cross-section
mw_run  91              logical  mw_run@@p_cut          # use the cut defined in run_card.dat for cross-section computation
mw_run  mw_cut          logical  mw_run@@use_cut        # use the cut defined in run_card.dat for weight computation
mw_run  9               logical  mw_run@@mw_cut         # use the cut defined in run_card.dat for weight computation
mw_run  22              logical  0                      # put on 1 to add a new events in the analysis (for step 5)
mw_run  bw_cut          logical  T
mw_run  me_bw_cut       logical  mw_run@@bw_cut         # use the BW cut for the cross-section computation. (is important)
mw_run  mw_bw_cut       logical  mw_run@@bw_cut         # use the BW cut for the weight computation. (is important)
mw_run  histo           logical F                       # put one creates histogram for value defined in topgraph.f
mw_run  histo_by_events logical F                       # decide if we create the histograms for each events
mw_run  weight_format   string  %(card)s.%(event)s\t%(value)s\t%(error)s\n # decide how to write the weight output file
#mw_run  cross_format    string  %(card)s\t%(value)s\t%(error)s\n   # decide how to write the cross-section output files
mw_run  nb_event_by_node integer  10                    # decide how many events weight are computed in a single cluster submission
mw_run  event_packing    integer  20                    # the input event files will be splitted in event_packing*nb_event_by_node 
mw_run  log_level        string   'channel'             # from low level of log to extensive log: weight, permutation, channel, iteration, full

mw_parameter mode logical 1                         
mw_parameter 1    logical mw_parameter@@mode # choose the mode for the generation of the param_card.dat
mw_parameter 2    logical 0                  # put on 1 to add new param_card.dat (for step 1) creates the corresponding new events dir if step 5 already performs

mw_perm  permutation    logical T                       # make the permutation between identical particles/jets
mw_perm  1              logical mw_perm@@permutation
mw_perm  bjet_is_jet    logical T                       #consider bjet like normal jet for permutation
mw_perm      2          logical mw_perm@@bjet_is_jet   
mw_perm  bjet_is_jet_for_selection  logical mw_perm@@2  #don't use btagging for selceting events (follow precedent rules)
mw_perm      21         logical mw_perm@@bjet_is_jet_for_selection  #don't use btagging for selceting events (follow precedent rules)
mw_perm  combine_mode    integer 1        # how to combine the permutation 1: average
                                          #                                2: first permutation 
                                          #                                3: higher permutation only


       
mw_gen     1         integer 1            # choose which propagator to allign in ECS:
                                # 0: favours the  BW at the very end of the decay chain
                                # 1: favours the thinest BW
mw_gen     2         integer 1            # Same condition but for blobs
mw_gen     3         integer 4            # maximal multiplicity for ECS sector
# Preference in blob generation (If more than one "True", we will perform a multi-channel techniques)
mw_gen     4         logical T            #      maximizes  the number of aligned BW
mw_gen     5         logical T            #      choose all momenta as  variables of integration
mw_gen     6         logical T            #      mixed choice between the two above options
mw_gen     10        logical T            # Authorizes ECS A  (0 neut)
mw_gen     11        logical T            # Authorizes ECS B  (1 neut)
mw_gen     12        logical T            # Authorizes ECS C  (1neut+1 visible(m=0))
mw_gen     13        logical T            # Authorizes ECS D  (2 neut ttbar-like)
mw_gen     14        logical T            # Authorizes ECS E  (2 neut HWW-like)
mw_gen     15        logical T            # Authorizes ECS F  (2 neut [WW in T-channel]-like)
