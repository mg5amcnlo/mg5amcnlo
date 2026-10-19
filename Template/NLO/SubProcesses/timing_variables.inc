*     -*-fortran-*-
c Exclusive CPU categories for the sampled integration-point profile.
c Keep the inexpensive guard at call sites: unsampled points read no clocks.
      integer timing_phase_space,timing_born,timing_real,timing_virtual
      integer timing_fks_local,timing_fks_integrated,timing_mc
      integer timing_clustering,timing_cuts,timing_scales,timing_pdf
      integer timing_reweight,timing_ewsudakov,timing_analysis
      integer timing_weights,timing_ncategories,timing_stack_size
      parameter(timing_phase_space=1,timing_born=2,timing_real=3,
     $     timing_virtual=4,timing_fks_local=5,timing_fks_integrated=6,
     $     timing_mc=7,timing_clustering=8,timing_cuts=9,
     $     timing_scales=10,
     $     timing_pdf=11,timing_reweight=12,timing_ewsudakov=13,
     $     timing_analysis=14,timing_weights=15,timing_ncategories=15,
     $     timing_stack_size=64)
      logical timing_sampled
      common/timing_sample_active/timing_sampled
