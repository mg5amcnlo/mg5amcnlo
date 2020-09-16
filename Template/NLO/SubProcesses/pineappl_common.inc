*     -*-fortran-*-
************************************************************************
*     NLO mode of aMC@NLO
*     Common blocks needed for the PineAPPL interface
************************************************************************
      include 'orders.inc'
*     Amp split size
      integer appl_amp_split_size 
      integer appl_qcdpower(amp_split_size), appl_qedpower(amp_split_size)
      common / appl_common_fixed / appl_amp_split_size, appl_qcdpower, appl_qedpower

*     Initial state map to be used together with the info in appl_flavmap
      integer appl_nlumi
      include 'pineappl_maxproc.inc'
      integer appl_nproc(mxpdflumi)
      integer appl_lumimap(2,max_nproc,mxpdflumi)
      common / appl_common_lumi / appl_lumimap,appl_nproc,appl_nlumi

*     Event weights, kinematics, observables etc. that are different event by event
      double precision appl_x1(4), 
     1                 appl_x2(4)
      double precision appl_muF2(4),
     1                 appl_muR2(4),
     2                 appl_QES2(4)
      double precision appl_w0(4, amp_split_size),
     1                 appl_wR(4, amp_split_size),
     2                 appl_wF(4, amp_split_size),
     3                 appl_wB(4, amp_split_size)
      integer appl_flavmap(4)
      common / appl_common_weights / appl_x1,appl_x2,
     1                               appl_muF2,appl_muR2,appl_QES2,
     2                               appl_w0,appl_wR,appl_wF,appl_wB,
     3                               appl_flavmap

*     Parameters of the grids.
*     These parameters can optionally be singularly specified by the user,
*     but if no specification is given, the code will use the default values.
      double precision appl_Q2min,appl_Q2max
      double precision appl_xmin,appl_xmax
      integer appl_nQ2,appl_Q2order
      integer appl_nx,appl_xorder
      common / appl_common_grid / appl_Q2min,appl_Q2max,
     1                            appl_xmin,appl_xmax,
     2                            appl_nQ2,appl_Q2order,
     3                            appl_nx,appl_xorder

*     Parameters of the histograms
      double precision appl_www_histo,appl_norm_histo
      double precision appl_obs_histo
      double precision appl_obs_min,appl_obs_max
      double precision appl_obs_bins(0:100)
      integer appl_obs_nbins
      integer appl_itype
      integer appl_amp_pos
      integer appl_obs_num
      common / appl_common_histokin / appl_www_histo,appl_norm_histo,
     1                                appl_obs_histo,
     2                                appl_obs_min,appl_obs_max,
     3                                appl_obs_bins,
     4                                appl_obs_nbins,
     5                                appl_itype,
     5                                appl_amp_pos,
     6                                appl_obs_num

*     This is the actual hadronic event weight, computed with the LHAPDF set
*     defined above, needed for checks in the C++ pineappl interface
*     Save also the vegas weight, needed to fill histograms and for checks
*     appl_www_histo is the same weight used to will the amcatnlo
*     histograms in the madfks routine
      double precision appl_event_weight,appl_vegaswgt
      double precision appl_xsec12,appl_xsec11,appl_xsec20
      common / appl_common_reco / appl_event_weight,appl_vegaswgt,
     1                            appl_xsec12,appl_xsec11,appl_xsec20

*     Common variables that are not fed to the bridge and used
*     for the indentifications of the observables
      integer nh_obs
      integer nmax_obs
      parameter(nmax_obs=50)
      integer ih_obs(nmax_obs)
      common / appl_ident / nh_obs,ih_obs

