c Set kinematic-dependent upper bound (veto) in MC subtraction terms
      logical dampMCsubt
      parameter (dampMCsubt=.true.)

c Damping can occur independently of the above, if CKKW is used
      logical dampMCsubt_used
      common/cdampMCsubt_used/dampMCsubt_used

c Define lower and upper veto range (see MC subtraction terms)
      double precision frac_low,frac_upp
      parameter (frac_low=0.05d0)
      parameter (frac_upp=0.5d0)

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

c If .true., does multi-parton merging
      logical UseCKKW
      parameter (UseCKKW=.true.)

c CKKW scale limits
      double precision CKKWscalemin,CKKWscalemax
      parameter (CKKWscalemin=15.d0)
      parameter (CKKWscalemax=30.d0)

c Minimal scale for MC showers
      double precision scaleMCmin
      parameter (scaleMCmin=2.d0)

c Controls smooth transition between CKKWscalemin and CKKWscalemax
      double precision alphaCKKW
      parameter (alphaCKKW=1.0d0)

c Largest parton multiplicity at the Born level
      integer NmaxCKKW
      parameter (NmaxCKKW=1)

c If .true., uses a sharp D function (with mu_Q = CKKWscalemax)
      logical DsharpCKKW
      parameter (DsharpCKKW=.false.)

c CKKW scales. Only one is used at present
      integer nscaleCKKW
      parameter (nscaleCKKW=10)
      double precision scale_CKKW(nscaleCKKW)
      common/cscale_CKKW/scale_CKKW

c ikintype=1 for counterevents, ikintype=2 for events
      integer ikintype
      common/cikintype/ikintype

c d_i used in CKKW
      double precision di_cnt_CKKW(-1:nexternal),di_ev_CKKW(-1:nexternal)
      common/cdi_CKKW/di_cnt_CKKW,di_ev_CKKW

c This is the i of the i-parton sample
      integer iCKKWnlo
      common/ciCKKWnlo/iCKKWnlo

c Weights used by CKKW
      double precision wCKKW_ev_D,wCKKW_ev_omD,wCKKW_cnt_D,wCKKW_cnt_omD
      common/cwCKKW1/wCKKW_ev_D,wCKKW_ev_omD,wCKKW_cnt_D,wCKKW_cnt_omD

      double precision rwgt_S_MC,rwgt_S_me,rwgt_H_MC,rwgt_H_me
      common/cwCKKW2/rwgt_S_MC,rwgt_S_me,rwgt_H_MC,rwgt_H_me
