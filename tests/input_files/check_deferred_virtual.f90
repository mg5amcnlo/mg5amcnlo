! Run the production delayed-loop callback with controlled loop amplitudes.
module fks_phase_space_data
  implicit none
  double precision :: p_born(0:3,4)
end module

module mc_native_context
  implicit none
  integer :: active_history=0
end module

module mint_module
  implicit none
  type pooled_virtual_context
    double precision :: mean(3)=0d0,actual(3)=0d0
  end type
  logical :: virtual_deferred_ready=.false.,pooled_virtual_sampled=.false.
  logical :: accept_carrier=.true.
  integer :: screen_calls=0,mark_calls=0
  double precision :: pooled_virtual_ratio(3)=0d0
  double precision :: pooled_virtual_actual_ratio(3)=0d0
  double precision :: virt_wgt_mint(0:3)=0d0,born_wgt_mint(0:3)=0d0
  double precision :: carrier_seen=0d0,other_seen=0d0
contains
  subroutine pooled_virtual_save_context(context)
    type(pooled_virtual_context),intent(out) :: context
    context%mean=pooled_virtual_ratio
    context%actual=pooled_virtual_actual_ratio
  end subroutine
  subroutine pooled_virtual_restore_context(context)
    type(pooled_virtual_context),intent(in) :: context
    pooled_virtual_ratio=context%mean
    pooled_virtual_actual_ratio=context%actual
  end subroutine
  subroutine screen_virtual_carrier(carrier,other,accepted)
    double precision,intent(in) :: carrier,other
    logical,intent(out) :: accepted
    screen_calls=screen_calls+1
    carrier_seen=carrier
    other_seen=other
    accepted=accept_carrier
  end subroutine
  subroutine pooled_virtual_mark_sampled(probability)
    double precision,intent(in) :: probability
    if(probability.ne.1d0)error stop 'delayed loop is always exact'
    mark_calls=mark_calls+1
    pooled_virtual_sampled=.true.
  end subroutine
end module

module deferred_fixture
  use fks_phase_space_data
  use mc_native_context
  use mint_module
  use weight_lines
  implicit none
  double precision g,scale,qes2,mu_r,virtual_over_born
  double precision amp_split_virt(3),amp_split_born_for_virt(3),amp_split_avv(3)
  double precision amp_split_finite_ML(3)
  integer nFKSprocess
  logical updateloop
  common/test_g/g
  common/test_scale/scale
  common/coupl_es/qes2
  common/test_mu_r/mu_r
  common/c_vob/virtual_over_born
  common/c_nFKSprocess/nFKSprocess
  common/to_amp_split_virt/amp_split_virt,amp_split_born_for_virt,amp_split_avv
  common/to_amp_split_finite/amp_split_finite_ML
  common/to_updateloop/updateloop
  integer :: loop_calls=0,update_calls=0,fks_calls=0,history_calls=0
  integer :: fks_seen(2)=0,history_seen(2)=0
  double precision :: expected_p(0:3,4),expected_born
  double precision :: coupling_cache=0d0
  double precision :: first_born(3)=[2d0,-1d0,0d0]
  double precision :: first_ratio(3)=[.5d0,3d0,20d0]
  double precision :: selected_ratio(3)=[2d0,-4d0,8d0]
  logical :: conditional_selection=.false.
  double precision :: exact_virtual(3)=[5d0,-10d0,3d0]
  double precision :: initial_wgt(3,4),initial_wgts(2,4),initial_parton(3,4)
contains
  subroutine initialize(zero_signed_born)
    logical,intent(in) :: zero_signed_born
    integer i
    if(zero_signed_born)first_born=[2d0,-2d0,0d0]
    allocate(wgt(3,4),wgts(2,4),parton_iproc(3,4))
    allocate(itype(4),niproc(4),amppos(4),nFKS(4),native_ids(4,4))
    icontr=4
    itype=[14,8,14,14]
    amppos=[1,2,3,2]
    niproc=[2,3,1,2]
    nFKS=[11,22,33,44]
    native_ids=123
    initial_wgt=reshape([(dble(i),i=1,12)],[3,4])
    initial_wgts=reshape([13d0,900d0,17d0,901d0,19d0,902d0,23d0,903d0],[2,4])
    initial_parton=reshape([8d0,-5d0,800d0,2d0,-3d0,7d0,19d0,801d0,802d0, &
                           -10d0,13d0,803d0],[3,4])
    wgt=initial_wgt
    wgts=initial_wgts
    parton_iproc=initial_parton
    p_born=reshape([(dble(i),i=1,16)],[4,4])
    expected_p=p_born
    expected_born=sum(first_born)
    amp_split_born_for_virt=first_born
    pooled_virtual_ratio=first_ratio
    pooled_virtual_actual_ratio=first_ratio
    g=2d0
    scale=3d0
    qes2=17d0
    mu_r=5d0
    nFKSprocess=7
    active_history=3
    call timing_reset()
    virtual_over_born=1234d0
    updateloop=.false.
    call deferred_virtual_context(0,expected_born)
    if(mu_r.ne.sqrt(17d0).or.coupling_cache.ne.4d0+sqrt(17d0).or.updateloop) &
      error stop 'cheap pass lost Binoth model-parameter update'
    if(update_calls.ne.1.or.loop_calls.ne.0)error stop 'cheap pass coupling/loop count'
    ! The final fold changes all mutable input. The loop must use only the
    ! saved first-fold state while retaining these final-fold values after it.
    p_born=-expected_p
    amp_split_born_for_virt=100d0
    pooled_virtual_ratio=200d0
    pooled_virtual_actual_ratio=300d0
    g=11d0
    scale=13d0
    qes2=29d0
    mu_r=7d0
    nFKSprocess=19
    active_history=5
    coupling_cache=g*g+mu_r
    virt_wgt_mint=[63d0,12d0,21d0,30d0]
    born_wgt_mint=[6d0,1d0,2d0,3d0]
    virtual_deferred_ready=.true.
    pooled_virtual_sampled=.true.
  end subroutine

  subroutine check_context
    if(g.ne.11d0.or.scale.ne.13d0.or.qes2.ne.29d0.or.mu_r.ne.7d0) &
      error stop 'final-fold scales not restored'
    if(nFKSprocess.ne.19.or.active_history.ne.5)error stop 'final-fold history not restored'
    if(coupling_cache.ne.128d0)error stop 'derived couplings not restored'
    if(any(p_born.ne.-expected_p))error stop 'global momenta overwritten'
    if(any(nFKS.ne.[11,22,33,44]).or.any(native_ids.ne.123))error stop 'record identity changed'
  end subroutine
end module

subroutine BinothLHA(p,born,virtual)
  use deferred_fixture
  implicit none
  double precision p(0:3,4),born,virtual
  loop_calls=loop_calls+1
  if(any(p.ne.expected_p).or.born.ne.expected_born)error stop 'wrong first-fold Born point'
  if(nFKSprocess.ne.7.or.active_history.ne.3)error stop 'wrong first-fold native history'
  if(g.ne.2d0.or.scale.ne.3d0.or.qes2.ne.17d0.or.mu_r.ne.5d0) &
    error stop 'wrong first-fold scales'
  if(any(pooled_virtual_ratio.ne.first_ratio))error stop 'wrong saved mean predictor'
  if(any(pooled_virtual_actual_ratio.ne.first_ratio))error stop 'wrong saved conditional context'
  if(conditional_selection)pooled_virtual_actual_ratio=selected_ratio
  ! Model the real Binoth wrapper's update of MadLoop scale and parameters.
  mu_r=sqrt(qes2)
  coupling_cache=g*g+mu_r
  amp_split_finite_ML=exact_virtual
  virtual=sum(exact_virtual)
end subroutine

subroutine update_as_param
  use deferred_fixture
  implicit none
  update_calls=update_calls+1
  if(update_calls.eq.1.and..not.updateloop)error stop 'loop model parameters not updated'
  if(update_calls.eq.2.and.updateloop)error stop 'unexpected restoration mode'
  coupling_cache=g*g+mu_r
end subroutine

subroutine update_fks_dir(fks)
  use deferred_fixture
  implicit none
  integer fks
  fks_calls=fks_calls+1
  if(fks_calls.gt.2)error stop 'unexpected context recomputation'
  fks_seen(fks_calls)=fks
  nFKSprocess=fks
  active_history=0
end subroutine

subroutine mc_set_history(history)
  use deferred_fixture
  implicit none
  integer history
  history_calls=history_calls+1
  if(history_calls.gt.2)error stop 'unexpected history resampling'
  history_seen(history_calls)=history
  active_history=history
end subroutine

program check_deferred_virtual
  use deferred_fixture
  implicit none
  character(40) mode
  double precision ratios(3),expected(3),want_wgt(3,4),want_wgts(2,4),want_parton(3,4)
  integer i,k
  call get_command_argument(1,mode)
  conditional_selection=trim(mode).eq.'conditional_selected'
  call initialize(trim(mode).eq.'zero_signed_born')
  if(trim(mode).eq.'veto')accept_carrier=.false.
  if(trim(mode).eq.'inactive')virtual_deferred_ready=.false.
  call deferred_virtual_context(1,expected_born)
  call check_context
  if(trim(mode).eq.'inactive')then
    if(screen_calls.ne.0.or.loop_calls.ne.0.or.mark_calls.ne.0)error stop 'inactive callback work'
    if(icontr.ne.4.or.any(wgt.ne.initial_wgt).or.any(wgts.ne.initial_wgts).or. &
       any(parton_iproc.ne.initial_parton))error stop 'inactive records changed'
  else
    ! Absolute flavour sums, not the possibly cancelling central weight.
    if(screen_calls.ne.1.or.carrier_seen.ne.55d0.or.other_seen.ne.12d0) &
      error stop 'wrong PDF weighted carrier'
    if(virtual_deferred_ready)error stop 'pending callback not cleared'
    if(trim(mode).eq.'veto')then
      if(loop_calls.ne.0.or.fks_calls.ne.0.or.update_calls.ne.1.or.mark_calls.ne.0) &
        error stop 'veto evaluated loop or mutated context'
      if(icontr.ne.0.or.any(virt_wgt_mint.ne.0d0).or.any(born_wgt_mint.ne.0d0).or. &
         pooled_virtual_sampled)error stop 'veto records retained'
    else
      if(loop_calls.ne.1.or.mark_calls.ne.1.or.update_calls.ne.2)error stop 'wrong exact-loop count'
      if(any(fks_seen.ne.[7,19]).or.any(history_seen.ne.[3,5]))error stop 'context ordering'
      ratios=(exact_virtual-first_ratio*first_born)/sum(abs(first_born))
      if(conditional_selection)ratios=(exact_virtual-selected_ratio*first_born)/sum(abs(first_born))
      want_wgt=initial_wgt
      want_wgts=initial_wgts
      want_parton=initial_parton
      do i=1,4
        if(itype(i).ne.14)cycle
        k=amppos(i)
        want_wgt(:,i)=want_wgt(:,i)*ratios(k)
        want_wgts(1,i)=want_wgts(1,i)*ratios(k)
        want_parton(1:niproc(i),i)=want_parton(1:niproc(i),i)*ratios(k)
      enddo
      if(any(wgt.ne.want_wgt).or.any(wgts.ne.want_wgts).or.any(parton_iproc.ne.want_parton)) &
        error stop 'mixed-order weights or inactive records corrupted'
      expected=[12d0,21d0,30d0]*ratios
      if(any(virt_wgt_mint(1:3).ne.expected).or.virt_wgt_mint(0).ne.sum(expected)) &
        error stop 'wrong MINT residual'
      if(any(born_wgt_mint.ne.[6d0,1d0,2d0,3d0]))error stop 'Born MINT carrier changed'
      if(expected_born.eq.0d0)then
        if(virtual_over_born.ne.0d0)error stop 'zero Born ratio not guarded'
      else
        if(virtual_over_born.ne.sum(exact_virtual)/expected_born)error stop 'wrong diagnostic ratio'
      endif
      if(.not.pooled_virtual_sampled)error stop 'exact sample flag lost'
    endif
  endif
  print *, 'PASS '//trim(mode)
end program
