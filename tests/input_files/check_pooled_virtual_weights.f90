program check_pooled_virtual_weights
  use pooled_weight_fixture
  use weight_lines
  implicit none
  include 'run.inc'
  character*32 mode
  character*7 event_norm
  common/event_normalisation/event_norm
  double precision :: b(3),v(3),k(3),residual(3),approximation(3),expected(3)
  integer :: nloops,nprepares,j,carrier_type
  integer,external :: find_virtual_mint_carrier
  logical fixed_order,nlo_ps
  common/c_fnlo_nlops/fixed_order,nlo_ps
  call get_command_argument(1,mode)
  call weight_lines_allocated(5,8,1,2)
  icontr=0

  select case(trim(mode))
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
  case('pooled_normalization')
     fixed_order=.true.; nlo_ps=.false.; flavour_bias=[0,1]; lpp=0
     event_norm='bias'
     b=[2d0,0d0,-3d0]; k=[0.25d0,99d0,0.5d0]; v=b*k
     do j=1,4
        call set_point(b,v,k)
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
        call close_vector(pooled_virtual_ratio*born_wgt_mint(1:3), &
             v*f_nb*5d0*7d0,'reconstructed weighted virtual')
     enddo
     ! No valid carrier must omit the observation, not fit an unscaled Born.
     call set_point(b,v,k)
     call exercise_virtual_stage()
     icontr=0
     call exercise_virtual_tracking()
     call prepare_weight_lines(1)
     call include_PDF_and_alphas()
     if(pooled_virtual_sampled)error stop 'unscaled Born observation retained'
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
