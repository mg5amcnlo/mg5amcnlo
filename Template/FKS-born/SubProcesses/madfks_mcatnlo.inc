c Set kinematic-dependent upper bound (veto) in MC subtraction terms
      logical dampMCsubt
      parameter (dampMCsubt=.true.)

c Define lower and upper veto range (see MC subtraction terms)
      double precision frac_low,frac_upp
      parameter (frac_low=0.1d0)
      parameter (frac_upp=1.0d0)

c Other control switches on veto (see MC subtraction terms)
      double precision scaleMClow,scaleMCdelta
      parameter (scaleMClow=10.d0)
      parameter (scaleMCdelta=20.d0)

c If .true., writes more in LH event file
      logical AddInfoLHE
      parameter (AddInfoLHE=.true.)

c If .true., applies Delta terms
      logical UseSudakov
      parameter (UseSudakov=.true.)

c If .true., multiplies MC subtraction terms by S_ev
      logical UseSfun
      parameter (UseSfun=.false.)

c If .true., reads CKKW scale from cuts.f
      logical UseCKKW
      parameter (UseCKKW=.false.)

