! Controlled native histories around the production flow dispatcher.
! The oracle keeps the correlation of each flow's P, MC kernel and G term.
module fks_phase_space_data
  implicit none
  real(8) :: p_born(0:3,4),p1_cnt(0:3,5,-2:2)
end module

module process_module
  implicit none
  integer :: ndelH=5
  logical,allocatable :: valid_dipole_n1(:,:)
end module

module scale_module
  implicit none
  integer :: born_flow_picked=2
  real(8),allocatable :: shower_scale_n1body(:,:),emsca_H(:,:,:,:)
  integer,allocatable :: event_colour_H(:,:,:,:)
contains
  subroutine compute_shower_scale_n1body(p,i_fks,j_fks,emitter_mass)
    real(8),intent(in) :: p(0:3,5),emitter_mass
    integer,intent(in) :: i_fks,j_fks
    shower_scale_n1body=100d0*born_flow_picked
  end subroutine
end module

module mc_counterterms
  implicit none
  real(8) :: gfactsf,gfactcl,gfactazi
contains
  real(8) function mc_shower_scale_mass()
    mc_shower_scale_mass=0d0
  end function
end module

module weight_lines
  implicit none
  logical :: mc_H_only=.false.,mc_S_only=.false.
  integer :: rejected_born_flow_points=0
end module

module flow_test_state
  use scale_module
  use process_module
  use mc_counterterms
  implicit none
  integer,parameter :: max_bcol=3,nexternal=5
  real(8) :: ordinary(9),matching(8),pdfscheme(3)
  common/factor_n1body/ordinary
  common/factor_n1body_NLOPS/matching
  common/factor_pdfsch/pdfscheme
  integer :: nFKSprocess,fold,ifold_counter,i_fks,j_fks
  common/c_nFKSprocess/nFKSprocess
  common/cfl/fold,ifold_counter
  common/fks_indices/i_fks,j_fks
  integer :: MCcntcalled
  common/c_MCcntcalled/MCcntcalled
  integer :: icolup_s(2,4),icolup_h(2,5)
  common/colour_connections/icolup_s,icolup_h
  integer :: ickkw
  common/test_run/ickkw
  character(4) :: abrv
  common/to_abrv/abrv
  logical :: calculatedBorn
  common/ccalculatedBorn/calculatedBorn
  logical :: is_leading_cflow(3)
  integer :: num_leading_cflows
  common/c_leading_cflows/is_leading_cflow,num_leading_cflows
  real(8) :: raw_weights(3),kernel(3),probabilities(3),gsoft(3),gcoll(3)
  real(8) :: real_me=40d0,subtraction=6d0
  real(8) :: totals(2),by_flow(2,3),seen_factors(20,3)
  integer :: evaluations(3),refreshes=0
  logical :: mutate_prefactors=.false.
contains
  subroutine setup()
    integer i
    if (.not.allocated(shower_scale_n1body)) then
      allocate(shower_scale_n1body(5,5),emsca_H(2,2,5,5))
      allocate(event_colour_H(2,5,2,2),valid_dipole_n1(5,5))
    endif
    ordinary=[(1d0+0.1d0*i,i=1,9)]
    matching=[2.1d0,2.1d0,2.2d0,2.2d0,2.3d0,2.3d0,2.4d0,2.4d0]
    pdfscheme=[3.1d0,3.2d0,3.3d0]
    raw_weights=[1d0,3d0,0d0]
    kernel=[2d0,9d0,0d0]
    probabilities=[0.2d0,0.7d0,0.4d0]
    gsoft=[0.1d0,0.8d0,0.3d0]
    gcoll=[0.9d0,0.2d0,0.5d0]
    totals=0d0
    by_flow=0d0
    seen_factors=0d0
    evaluations=0
    refreshes=0
    mutate_prefactors=.false.
    is_leading_cflow=[.true.,.true.,.false.]
    num_leading_cflows=2
    born_flow_picked=2
    shower_scale_n1body=777d0
    emsca_H=888d0
    event_colour_H=999
    valid_dipole_n1=.false.
    valid_dipole_n1(1,2)=.true.
    icolup_s=123
    icolup_h=456
    gfactsf=0.91d0
    gfactcl=0.92d0
    gfactazi=0.93d0
    ickkw=0
    abrv='all '
    calculatedBorn=.false.
    MCcntcalled=0
    nFKSprocess=2
    fold=0
    ifold_counter=1
    i_fks=5
    j_fks=3
  end subroutine

  subroutine assert_close(actual,expected,label)
    real(8),intent(in) :: actual,expected
    character(*),intent(in) :: label
    if (abs(actual-expected).gt.1d-11*max(1d0,abs(expected))) then
      print *, label,actual,expected
      error stop 'flow sum mismatch'
    endif
  end subroutine

  subroutine oracle(flow,fraction,base,expected)
    integer,intent(in) :: flow
    real(8),intent(in) :: fraction,base(20)
    real(8),intent(out) :: expected(2)
    real(8) :: p,real_weight,mc_weight,g_weight,c_weight
    p=probabilities(flow)
    real_weight=fraction*base(1)*real_me
    mc_weight=base(16)*kernel(flow)
    g_weight=fraction*(1d0-gsoft(flow))*(base(10)*7d0 &
         -(1d0-gcoll(flow))*base(12)*5d0)
    c_weight=fraction*base(2)*subtraction
    expected(1)=p*(real_weight-mc_weight-g_weight)
    expected(2)=(1d0-p)*real_weight+p*(mc_weight+g_weight)-c_weight
  end subroutine
end module

program check_mc_flow_sum
  use FKSParams
  use fks_phase_space_data
  use flow_test_state
  implicit none
  real(8) :: p(0:3,5),probne,base(20),expected(2),flow_expected(2),fraction
  real(8) :: one_flow(2),saved_scales(5,5)
  logical :: weights_valid,saved_dipoles(5,5)
  integer :: i,mode,zero_case
  p=1d0
  p_born=p(:,1:4)
  p1_cnt=1d0

  ! A sum must retain the flow correlation of P with both M and G.
  ! A zero Born weight can have a nonzero MC term at another coupling.
  do zero_case=0,1
    call setup()
    if (zero_case.eq.1) then
      is_leading_cflow(3)=.true.
      num_leading_cflows=3
      kernel(3)=4d0
    endif
    MCSubtractionAtFixedFlow=.false.
    base=[ordinary,matching,pdfscheme]
    saved_scales=shower_scale_n1body
    saved_dipoles=valid_dipole_n1
    expected=0d0
    do i=1,3
      if (.not.is_leading_cflow(i)) cycle
      call oracle(i,raw_weights(i)/sum(raw_weights),base,flow_expected)
      expected=expected+flow_expected
    enddo
    mutate_prefactors=.true.
    call compute_NLOPS_flow_weights(p,p,p,1d0,.true.,.true.,probne,weights_valid)
    if (.not.weights_valid) error stop 'valid flows rejected'
    do i=1,2
      call assert_close(totals(i),expected(i),'summed H/S contribution')
    enddo
    call assert_close(sum(totals),base(1)*real_me-base(2)*subtraction,'H+S cancellation')
    if (any(evaluations(1:2).ne.1)) error stop 'positive flow omitted or repeated'
    if (evaluations(3).ne.zero_case) error stop 'zero leading flow omitted'
    do i=1,3
      if (.not.is_leading_cflow(i)) cycle
      fraction=raw_weights(i)/sum(raw_weights)
      if (maxval(abs(seen_factors(1:15,i)-fraction*base(1:15))).gt.1d-12) &
           error stop 'ordinary or G factors not reset per flow'
      if (maxval(abs(seen_factors(16:17,i)-base(16:17))).gt.1d-12) &
           error stop 'MC flow fraction applied twice'
      if (maxval(abs(seen_factors(18:20,i)-fraction*base(18:20))).gt.1d-12) &
           error stop 'PDF factors not reset per flow'
    enddo
    if (maxval(abs([ordinary,matching,pdfscheme]-base)).gt.1d-12) &
         error stop 'prefactors not restored'
    if (born_flow_picked.ne.2) error stop 'owner flow not restored'
    if (any(shower_scale_n1body.ne.saved_scales)) error stop 'n1 scales not restored'
    if (any(valid_dipole_n1.neqv.saved_dipoles)) error stop 'n1 dipoles not restored'
    if (any(emsca_H(2,1,:,:).ne.20d0)) error stop 'Delta owner scales not retained'
    if (any(emsca_H(1,:,:,:).ne.888d0).or.any(emsca_H(:,2,:,:).ne.888d0)) &
         error stop 'unrelated Delta scales changed'
    if (any(event_colour_H.ne.999)) error stop 'event colour owner changed'
    if (any(icolup_s.ne.1002).or.any(icolup_h.ne.2002)) error stop 'owner working colours not retained'
    if (maxval(abs([gfactsf,gfactcl,gfactazi]-[gsoft(2),gcoll(2),2d0])).gt.1d-12) &
         error stop 'owner G state not retained'
    if (MCcntcalled.ne.2) error stop 'owner MC call status not retained'
    call assert_close(probne,sum(raw_weights*probabilities)/sum(raw_weights),'flow-averaged P')
  enddo

  ! Fixed mode receives the existing importance correction and forwards
  ! exactly one native call without refreshing Born data or colour state.
  call setup()
  MCSubtractionAtFixedFlow=.true.
  base=[ordinary,matching,pdfscheme]
  fraction=raw_weights(2)/sum(raw_weights)
  call oracle(2,fraction,base,expected)
  call include_born_flow_weight(fraction,fraction)
  call compute_NLOPS_flow_weights(p,p,p,1d0,.true.,.true.,probne,weights_valid)
  do i=1,2
    call assert_close(totals(i),expected(i)/fraction,'fixed flow sampling')
  enddo
  if (any(evaluations.ne.[0,1,0])) error stop 'fixed mode did not forward one flow'
  if (refreshes.ne.0) error stop 'fixed mode refreshed Born evaluation'
  call assert_close(probne,probabilities(2),'fixed mode P')

  ! With one colour flow the two estimators coincide, including Delta/G.
  do mode=1,2
    call setup()
    is_leading_cflow=[.false.,.true.,.false.]
    num_leading_cflows=1
    raw_weights=[0d0,3d0,0d0]
    MCSubtractionAtFixedFlow=mode.eq.1
    if (MCSubtractionAtFixedFlow) call include_born_flow_weight(1d0,1d0)
    call compute_NLOPS_flow_weights(p,p,p,1d0,.true.,.true.,probne,weights_valid)
    if (mode.eq.1) then
      one_flow=totals
    else
      do i=1,2
        call assert_close(totals(i),one_flow(i),'one-flow equivalence')
      enddo
    endif
  enddo

  call setup()
  MCSubtractionAtFixedFlow=.false.
  raw_weights=[-1d0,3d0,0d0]
  call compute_NLOPS_flow_weights(p,p,p,1d0,.true.,.true.,probne,weights_valid)
  if (weights_valid.or.any(evaluations.ne.0)) error stop 'invalid colour weights accepted'
  print *, 'PASS Born-flow sum, sampling, compensation and owner restoration'
end program

subroutine compute_native_NLOPS_weights(p,p_lab,p_cms,jacPS,cuts_born,cuts_real,probne)
  use flow_test_state
  implicit none
  real(8) :: p(0:3,5),p_lab(0:3,5),p_cms(0:3,5),jacPS,probne
  logical :: cuts_born,cuts_real
  real(8) :: real_weight,mc_weight,g_weight,c_weight,contribution(2)
  integer :: flow
  flow=born_flow_picked
  evaluations(flow)=evaluations(flow)+1
  MCcntcalled=flow
  seen_factors(:,flow)=[ordinary,matching,pdfscheme]
  probne=probabilities(flow)
  real_weight=ordinary(1)*real_me
  mc_weight=matching(7)*kernel(flow)
  g_weight=(1d0-gsoft(flow))*(matching(1)*7d0-(1d0-gcoll(flow))*matching(3)*5d0)
  c_weight=ordinary(2)*subtraction
  contribution(1)=probne*(real_weight-mc_weight-g_weight)
  contribution(2)=(1d0-probne)*real_weight+probne*(mc_weight+g_weight)-c_weight
  totals=totals+contribution
  by_flow(:,flow)=by_flow(:,flow)+contribution
  ! Real native evaluation updates working colours, Delta scales and G.
  emsca_H(nFKSprocess,ifold_counter,:,:)=10d0*flow
  icolup_s=1000+flow
  icolup_h=2000+flow
  gfactsf=gsoft(flow)
  gfactcl=gcoll(flow)
  gfactazi=flow
  if (mutate_prefactors) then
    ordinary=-11d0*ordinary
    matching=-12d0*matching
    pdfscheme=-13d0*pdfscheme
  endif
end subroutine

subroutine init_process_module_n1body_flow(flow,preserve_event_owner)
  use flow_test_state
  implicit none
  integer :: flow
  logical :: preserve_event_owner
  if (.not.preserve_event_owner) error stop 'flow sum overwrites owner'
  valid_dipole_n1=.false.
  valid_dipole_n1(flow,5)=.true.
end subroutine

subroutine set_cms_stuff(flag)
  implicit none
  integer :: flag
end subroutine

subroutine set_alphaS(p)
  implicit none
  real(8) :: p(0:3,*)
end subroutine

subroutine set_FxFx_scale(flag,p)
  implicit none
  integer :: flag
  real(8) :: p(0:3,*)
end subroutine

subroutine sborn_native(p,wgt)
  use flow_test_state,only: refreshes
  implicit none
  real(8) :: p(0:3,*),wgt
  refreshes=refreshes+1
  wgt=1d0
end subroutine

real(8) function mc_born_flow_weight(flow)
  use flow_test_state,only: raw_weights
  implicit none
  integer :: flow
  mc_born_flow_weight=raw_weights(flow)
end function
