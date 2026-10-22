program check_pooled_virtual_weights
  use pooled_weight_fixture
  use weight_lines
  implicit none
  include 'run.inc'
  character*32 mode
  character*7 event_norm
  common/event_normalisation/event_norm
  double precision :: b(3),v(3),k(3),residual(3),approximation(3),expected(3)
  double precision :: z,truth(4),frozen(4),prediction,lkl
  integer :: nloops,nprepares,nhelicities,j,carrier_type
  integer,external :: find_virtual_mint_carrier
  logical fixed_order,nlo_ps
  common/c_fnlo_nlops/fixed_order,nlo_ps
  call get_command_argument(1,mode)
  call weight_lines_allocated(5,8,1,2)
  icontr=0

  select case(trim(mode))
  case('surrogate_cancellation')
     b=[2d0,0d0,-3d0];v=[0d0,7d0,-.5d0];k=2d0
     call set_point(b,v,k)
     call enable_surrogate(2d0)
     call exercise_virtual_stage()
     if(.not.vs_ratio_set.or..not.vs_vcomp)error stop 'fitted surrogate not selected'
     call close_vector(amp_split_avv,2d0*b,'fitted approximation')
     call close_vector(amp_split_avv+amp_split_virt,v,'fitted cancellation at zero virtual')
     residual=amp_split_virt
     approximation=amp_split_avv
     nloops=loop_calls
     fold=1;average_virtual=-777d0;exact_virtual=999d0
     call exercise_virtual_stage()
     if(loop_calls/=nloops)error stop 'fold repeated fitted virtual work'
     call close_vector(amp_split_virt,residual,'fitted fold residual')
     call close_vector(amp_split_avv,approximation,'fitted fold approximation')
  case('surrogate_sampling')
     b=[2d0,1d0,-1d0];v=[7d0,-4d0,1d0];k=2d0
     call set_point(b,v,k)
     call enable_surrogate(2d0)
     virtual_fraction=.25d0
     call exercise_virtual_stage()
     approximation=amp_split_avv
     call close_vector(amp_split_avv+.25d0*amp_split_virt,v,'fitted sampling mean')
     nloops=loop_calls;uniform=.9d0
     call exercise_virtual_stage()
     if(loop_calls/=nloops.or.vs_vcomp)error stop 'unsampled point entered the fit'
     call close_vector(amp_split_virt,[0d0,0d0,0d0],'unsampled fitted residual')
     call close_vector(amp_split_avv,approximation,'unsampled fitted approximation')
     abrv='virt';amp_split_avv=0d0
     call exercise_virtual_stage()
     call close_vector(amp_split_virt,v-approximation,'separate fitted virtual stream')
     if(vs_fvirt/=1d0)error stop 'separate stream sampling fraction'
     abrv='all'
     do j=1,2
        born_spread_phase=j;nloops=loop_calls
        call exercise_virtual_stage()
        if(loop_calls/=nloops.or.vs_vcomp)error stop 'calibration evaluated fitted residual'
        call close_vector(amp_split_avv,approximation,'calibration retains fitted average')
     enddo
  case('surrogate_fit_weights')
     call set_point([1d0,0d0,0d0],[2d0,0d0,0d0],[0d0,0d0,0d0])
     use_pooled_virtual=.false.;VirtTrickSurrogate=2
     vs_vcomp=.true.;vs_featok=.true.;vs_born=1d0;vs_fvirt=.25d0
     vs_feat=0d0;vs_feat(1)=1d0;vs_nfeat=4
     average_virtual(0,1)=1d0
     icontr=2;itype(1:2)=[14,15]
     do j=1,30
        ! Target 2 has integrand factor 3; target 4 has factor 1.
        ! The fitted constant is (9*2+4)/(9+1) = 2.2.
        vs_virt=2d0;wgts(1,1:2)=[12d0,3d0]
        call virt_surrogate_fit_point
        vs_virt=4d0;wgts(1,1:2)=[12d0,1d0]
        call virt_surrogate_fit_point
     enddo
     vs_virt=1000d0;wgts(1,1:2)=[3996d0,1d0]
     imode=1;call virt_surrogate_fit_point
     ! A barrier-trained model freezes the native fit during application.
     imode=0;use_pooled_virtual=.true.;pooled_virtual_applying=.true.
     call virt_surrogate_fit_point
     use_pooled_virtual=.false.;VirtTrickSurrogate=0;call virt_surrogate_fit_point
     VirtTrickSurrogate=2;vs_vcomp=.false.;call virt_surrogate_fit_point
     call vs_solve(1,.false.)
     if(abs(vs_value(1)-2.2d0)>1d-8)error stop 'fit weights or grid-setup gating changed'
     ! Collection still adapts its native surrogate. One more target 4
     ! with unit integrand factor changes the weighted mean to 664/301.
     use_pooled_virtual=.true.;pooled_virtual_applying=.false.
     VirtTrickSurrogate=2;vs_vcomp=.true.;vs_virt=4d0
     wgts(1,1:2)=[12d0,1d0]
     call virt_surrogate_fit_point
     call vs_solve(1,.false.)
     if(abs(vs_value(1)-664d0/301d0)>1d-8)error stop 'pooled collection did not adapt the native fit'
  case('surrogate_fitting')
     call vs_reset
     truth=[1d0,2d0,-.5d0,.25d0]
     do j=1,120
        z=(j-60d0)/30d0
        vs_feat(1:4)=[1d0,z,z*z,sin(2d0*z)]
        prediction=dot_product(truth,vs_feat(1:4))
        call vs_accumulate(1,prediction,0d0,1d0+.1d0*j)
     enddo
     vs_featok=.true.
     call vs_select(1,1d0)
     if(vs_ratio_set.or.vs_value(1)/=0d0)error stop 'fit changed within iteration'
     call vs_solve(1,.false.)
     call vs_select(1,1d0)
     if(.not.vs_ratio_set)error stop 'fit did not improve the grid'
     do j=1,4
        vs_feat=0d0;vs_feat(j)=1d0
        if(abs(vs_value(1)-truth(j))>1d-6)error stop 'weighted fit lost known correlation'
     enddo
     call load_surrogate_model(0d0,2)
     do j=1,120
        z=(j-60d0)/30d0
        vs_feat(1:4)=[1d0,z,z*z,sin(2d0*z)]
        call vs_accumulate(1,1d0,1.1d0,1d0)
        call vs_accumulate(2,1d0,1.1d0,1d0)
     enddo
     if(vs_value(1)/=0d0)error stop 'validation refitted its own points'
     call vs_solve(1,.true.)
     call vs_solve(2,.false.)
     vs_featok=.true.
     call vs_select(1,1d0)
     if(.not.vs_ratio_set)error stop 'sign-aware fallback selection'
     call vs_select(2,1d0)
     if(vs_ratio_set)error stop 'variance fallback selection'
     call vs_reset
     vs_featok=.true.;vs_feat(1)=1d0
     call vs_select(1,1d0)
     if(vs_ratio_set.or.vs_value(1)/=0d0)error stop 'grid reset retained a fit'
  case('surrogate_features')
     call set_point([2d0,0d0,0d0],[1d0,0d0,0d0],[0d0,0d0,0d0])
     VirtTrickSurrogate=2
     p_born(:,1)=[10d0,0d0,0d0,10d0];p_born(:,2)=[10d0,0d0,0d0,-10d0]
     p_born(:,3)=[5d0,3d0,0d0,4d0];p_born(:,4)=[5d0,-3d0,0d0,-4d0]
     QES2=4d0
     ! The unresolved real leg is 3: real links (1,4) and (5,5)
     ! refer to native Born links (1,3) and (4,4).
     call vs_add_link(p_born,QES2,3,1,4,2d0)
     call vs_add_link(p_born,QES2,3,5,5,3d0)
     call vs_features(p_born,2d0,QES2,VirtTrickSurrogate)
     if(.not.vs_featok.or.vs_nfeat/=16)error stop 'missing kinematic features'
     lkl=log(5d0)
     if(abs(vs_feat(2)-2d0*lkl)>1d-12.or.abs(vs_feat(3)-2d0*lkl*lkl)>1d-12) &
          error stop 'real-to-Born colour link mapping'
     if(vs_feat(1)/=2d0.or.vs_feat(4)/=3d0)error stop 'Born or massive self-link feature'
     if(abs(vs_feat(5)-2d0*log(100d0))>1d-12)error stop 'all-pair kinematic log'
     frozen=vs_feat(1:4)
     VirtTrickSurrogate=1;call vs_features(p_born,2d0,QES2,VirtTrickSurrogate)
     if(vs_nfeat/=4.or.any(vs_feat(1:4)/=frozen))error stop 'dipole feature mode'
     QES2=0d0;call vs_features(p_born,2d0,QES2,VirtTrickSurrogate)
     if(vs_featok)error stop 'invalid reference scale used by fit'
  case('born_spread_calibration')
     b=[2d0,-1d0,3d0];v=[7d0,-4d0,1d0];k=[.5d0,-2d0,1d0]
     do j=1,2
        call set_point(b,v,k)
        born_spread_phase=j
        ! Distinguish the MINT mean from stale pooled point state.
        average_virtual(1:3,1)=k
        pooled_virtual_ratio=999d0
        proposal_ratio=-777d0
        nloops=loop_calls
        nprepares=pooled_prepare_calls
        nhelicities=pooled_helicity_calls
        call exercise_virtual_stage()
        if(loop_calls.ne.nloops)error stop 'calibration evaluated a loop'
        if(pooled_prepare_calls.ne.nprepares.or.pooled_helicity_calls.ne.nhelicities) &
             error stop 'unused calibration virtual features prepared'
        if(pooled_virtual_sampled)error stop 'calibration marked a virtual observation'
        call close_vector(amp_split_avv,b*k,'calibration retains the current MINT mean')
        call close_vector(pooled_virtual_actual_ratio,k,'calibration actual follows MINT')
        call close_vector(amp_split_virt,[0d0,0d0,0d0],'calibration has no residual')
        ! An already loaded model must still predict when one is available.
        pooled_virtual_applying=.true.
        proposal_ratio=2d0*k
        call exercise_virtual_stage()
        if(loop_calls.ne.nloops)error stop 'model calibration evaluated a loop'
        if(pooled_prepare_calls.ne.nprepares+1.or.pooled_helicity_calls.ne.nhelicities+1) &
             error stop 'calibration skipped the loaded model'
        call close_vector(amp_split_avv,2d0*b*k,'calibration keeps a loaded prediction')
     enddo
  case('conditional_actual_subtraction')
     b=[2d0,-1d0,3d0];v=[7d0,-4d0,1d0];k=[.5d0,-2d0,1d0]
     call set_point(b,v,k)
     selected_ratio=[1d0,3d0,-.5d0]
     conditional_selection=.true.
     virtual_fraction=.25d0
     call exercise_virtual_stage()
     call close_vector(amp_split_avv,b*k,'mean approximation uses weighted head mean')
     call close_vector(amp_split_virt,(v-b*selected_ratio)/.25d0,'selected actual subtraction')
     call close_vector(.25d0*amp_split_virt+b*pooled_virtual_actual_ratio,v, &
          'capture target reconstructs selected virtual')
  case('deferred_veto_xsec')
     ! Veto-xsec consumes the virtual before the deferred closing callback.
     b=[2d0,1d0,-1d0];v=[3d0,-4d0,8d0];k=2d0
     call set_point(b,v,k)
     abrv='virt'
     ickkw=-1
     virtual_defer_requested=.true.
     nloops=loop_calls
     call exercise_virtual_stage()
     if(loop_calls.ne.nloops+1.or.virtual_deferred_ready)error stop 'veto-xsec was deferred'
     call close_vector(amp_split_virt,v,'veto-xsec needs immediate virtual')
  case('deferred_zero_carrier')
     ! A zero total absolute Born carrier cannot bound a finite loop.
     ! It must take the ordinary exact branch even when deferral is enabled.
     b=0d0;v=[2d0,-3d0,7d0];k=[100d0,-20d0,30d0]
     call set_point(b,v,k)
     abrv='virt'
     virtual_defer_requested=.true.
     nloops=loop_calls
     call exercise_virtual_stage()
     if(loop_calls.ne.nloops+1.or.virtual_deferred_ready)error stop 'zero carrier was deferred'
     call close_vector(amp_split_virt,v,'zero carrier finite virtual lost')
     if(virtual_over_born.ne.0d0)error stop 'zero carrier diagnostic ratio'
     if(.not.pooled_virtual_sampled.or.pooled_probability.ne.1d0)error stop 'fallback not exact'
  case('pooled_cancellation')
     ! Three coupling orders: zero virtual, zero Born, and signed Born.
     b=[2d0,0d0,-3d0]; v=[0d0,7d0,-0.5d0]; k=[0.25d0,99d0,0.5d0]
     call set_point(b,v,k)
     call exercise_virtual_stage()
     call close_vector(amp_split_virt+amp_split_avv,v,'exact stream cancellation')
     call close_vector(amp_split_avv,[0.5d0,0d0,-1.5d0],'per-order approximation')
     if(avv_wgt.ne.-1d0.or.virt_wgt.ne.7.5d0)error stop 'incorrect total virtual weights'
     residual=amp_split_virt
     approximation=amp_split_avv
     nloops=loop_calls; nprepares=pooled_prepare_calls
     exact_virtual=999d0; proposal_ratio=-999d0
     average_virtual=-777d0 ! MINT resets its own marginals between folds.
     fold=1
     call exercise_virtual_stage()
     if(loop_calls.ne.nloops.or.pooled_prepare_calls.ne.nprepares)error stop 'fold repeated virtual work'
     call close_vector(amp_split_virt,residual,'fold residual changed')
     call close_vector(amp_split_avv,approximation,'fold approximation changed')
     call set_point([0d0,0d0,0d0],v,k)
     call exercise_virtual_stage()
     call close_vector(amp_split_virt,v,'Born zero removed finite virtual')
     if(virtual_over_born.ne.0d0)error stop 'zero Born diagnostic division'
     ! A zero residual in one coupling order must retain its Born
     ! observation while the other orders produce genuine type-14 lines.
     b=[2d0,0d0,-3d0]; v=[0.5d0,7d0,-0.5d0]
     call set_point(b,v,k)
     call exercise_virtual_stage()
     icontr=0
     call exercise_virtual_tracking()
     if(icontr.ne.2)error stop 'incorrect number of nonzero split residuals'
     call close_vector(born_wgt_mint(1:3),b*f_nb,'partial zero residual lost Born')
     call close_vector(virt_wgt_mint(1:3),(v-b*k)*f_nb,'split residual tracking')
  case('pooled_subsampling')
     b=[2d0,1d0,-3d0]; v=[0d0,7d0,-0.5d0]; k=[0.25d0,0.5d0,0.5d0]
     call set_point(b,v,k)
     virtual_fraction=0.25d0
     call exercise_virtual_stage()
     if(.not.pooled_virtual_sampled.or.pooled_probability.ne.0.25d0) &
          error stop 'lost loop sampling probability'
     residual=amp_split_virt
     approximation=amp_split_avv
     call close_vector(approximation+0.25d0*residual,v,'subsampled exact mean')
     uniform=0.9d0
     nloops=loop_calls
     call exercise_virtual_stage()
     if(pooled_virtual_sampled.or.loop_calls.ne.nloops)error stop 'unsampled point trained'
     call close_vector(amp_split_virt,[0d0,0d0,0d0],'unsampled residual')
     call close_vector(amp_split_avv,approximation,'unsampled approximation')
     ! Generation explicitly requests the residual stream with probability one.
     abrv='virt'
     amp_split_avv=0d0
     call exercise_virtual_stage()
     if(pooled_probability.ne.1d0)error stop 'virtual stream probability'
     call close_vector(amp_split_virt,v-approximation,'explicit residual stream')
     call close_vector(amp_split_avv,[0d0,0d0,0d0],'explicit stream added approximation')
  case('pooled_tail_probability')
     b=[2d0,-3d0,0d0];v=[0d0,7d0,-5d0];k=[10d0,-3d0,2d0]
     call set_point(b,v,k)
     virtual_fraction=.01d0
     pooled_exact_tail=.true.
     uniform=.99d0
     call exercise_virtual_stage
     if(.not.pooled_virtual_sampled.or.pooled_probability/=1d0)error stop 'tail probability'
     call close_vector(amp_split_avv+amp_split_virt,v,'exact tail correction uses actual probability')
     if(virtual_fraction(1)/=.01d0)error stop 'tail changed channel fraction'
  case('pooled_normalization','surrogate_normalization')
     fixed_order=.true.; nlo_ps=.false.; flavour_bias=[0,1]; lpp=0
     event_norm='bias'
     b=[2d0,0d0,-3d0]; k=[0.25d0,99d0,0.5d0]
     if(mode=='surrogate_normalization')k=2d0
     v=b*k
     do j=1,4
        call set_point(b,v,k)
        if(mode=='surrogate_normalization')call enable_surrogate(2d0)
        call exercise_virtual_stage()
        icontr=0
        call exercise_virtual_tracking()
        if(icontr.ne.0)error stop 'perfect approximation created residual record'
        call close_vector(born_wgt_mint(1:3),b*f_nb,'zero-residual Born tracking')
        ! Prefer type14 when present; otherwise the zero residual still
        ! needs normalization using type15, integrated counterterms or Born.
        carrier_type=14
        if(j.eq.2)carrier_type=15
        if(j.eq.3)carrier_type=3
        if(j.eq.4)carrier_type=2
        call prepare_weight_lines(carrier_type)
        if(find_virtual_mint_carrier().ne.2)error stop 'wrong n-body normalization carrier'
        call include_PDF_and_alphas()
        call close_vector(born_wgt_mint(1:3),b*f_nb*5d0,'PDF factor missing or repeated')
        call include_bias_wgt()
        expected=b*f_nb*5d0*7d0
        call close_vector(born_wgt_mint(1:3),expected,'bias factor missing or repeated')
        call close_vector(virt_wgt_mint(1:3),[0d0,0d0,0d0],'zero residual changed')
        call close_vector(k*born_wgt_mint(1:3), &
             v*f_nb*5d0*7d0,'reconstructed weighted virtual')
     enddo
     ! No valid carrier must omit the observation, not fit an unscaled Born.
     call set_point(b,v,k)
     if(mode=='surrogate_normalization')call enable_surrogate(2d0)
     call exercise_virtual_stage()
     icontr=0
     call exercise_virtual_tracking()
     call prepare_weight_lines(1)
     call include_PDF_and_alphas()
     if(pooled_virtual_sampled.or.vs_vcomp)error stop 'unscaled Born observation retained'
  case default
     error stop 'unknown test mode'
  end select
  call deallocate_weight_lines()
  write(*,*)'PASS '//trim(mode)
contains
  subroutine close_vector(actual,reference,label)
    double precision actual(3),reference(3)
    character(*) label
    if(any(abs(actual-reference).gt.1d-12*max(1d0,maxval(abs(reference)))))then
       write(*,*)label,actual,reference
       error stop 'weight identity failed'
    endif
  end subroutine
  subroutine prepare_weight_lines(carrier)
    integer carrier
    icontr=3
    itype(1:3)=[1,carrier,carrier]
    nFKS=1; native_ids=0; event_nFKS=1; ipr=0
    bjx=0.25d0; scales2=16d0; wgt=0d0; wgt(1,:)=1d0
    wgt_ME_tree=1d0; g_strong=2d0; QCDpower=2; cpower=0d0
    H_event=.false.; H_event(1)=.true.
    y_bst=0d0; orderstag=1; momenta=1d0; momenta_m=1d0
  end subroutine
end program
