c Define lower and upper veto range (see MC subtraction terms)
      double precision frac_low,frac_upp
      parameter (frac_low=0.1d0)
      parameter (frac_upp=1.0d0)

c Other control switches on veto (see MC subtraction terms)
      double precision scaleMClow,scaleMCdelta
      parameter (scaleMClow=10.d0)
      parameter (scaleMCdelta=20.d0)

c IR cutoff for shower scales
      double precision scaleMCcut
      parameter (scaleMCcut=3.d0)

c If .true., multiplies MC subtraction terms by S_ev
      logical UseSfun
      parameter (UseSfun=.false.)

c If .true., reads CKKW scale from cuts.f
      logical UseCKKW
      parameter (UseCKKW=.false.)

