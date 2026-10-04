program check_born_flow_rejection
  use, intrinsic :: ieee_arithmetic
  use flow_fixture
  use mint_module
  use scale_module
  implicit none
  character(16) :: mode
  double precision :: factor,nan,inf,answer(6),dummy,x(5),sigintF,virtual_over_born
  integer :: flow,i,old_draws,old_calls,fold,ifold_counter,nndim,ickkw,ifks,jfks
  logical :: leading(3),mcatnlo_delta
  integer :: nleading
  common/c_leading_cflows/leading,nleading
  common/cfl/fold,ifold_counter
  common/tosigint/nndim
  common/test_run/ickkw,mcatnlo_delta
  common/fks_indices/ifks,jfks
  common/c_vob/virtual_over_born
  character(4) :: abrv
  common/to_abrv/abrv
  external sigintF
  call get_command_argument(1,mode)
  nan=ieee_value(0d0,ieee_quiet_nan)
  inf=ieee_value(0d0,ieee_positive_inf)
  leading=[.true.,.true.,.false.]
  nleading=2
  if(trim(mode).eq.'selection')then
    uniform=0.1d0
    call get_born_flow(flow,factor)
    if(flow.ne.1.or.factor.ne.0.25d0)error stop 'changed valid first-flow draw'
    uniform=0.5d0
    call get_born_flow(flow,factor)
    if(flow.ne.2.or.factor.ne.0.75d0)error stop 'changed valid second-flow draw'
    values(3)=nan
    call get_born_flow(flow,factor)
    if(flow.ne.2.or.factor.ne.0.75d0)error stop 'used inactive colour weight'
    old_draws=draws
    do i=1,6
      values=[1d0,3d0,0d0]
      select case(i)
      case(1)
        values(2)=nan
      case(2)
        values(2)=inf
      case(3)
        values(1)=-inf
      case(4)
        values(1)=-1d0
      case(5)
        values=0d0
      case(6)
        values=huge(0d0)
      end select
      flow=19
      factor=17d0
      call get_born_flow(flow,factor)
      if(flow.ne.0.or.factor.ne.0d0)error stop 'invalid draw retained a colour'
    enddo
    if(draws.ne.old_draws)error stop 'sampled an invalid colour distribution'
    if(rejected_born_flow_points.ne.6)error stop 'incorrect rejection count'
  else
    nndim=5
    ickkw=0
    mcatnlo_delta=.false.
    ifks=5
    jfks=3
    abrv='all'
    x=0.5d0
    call weight_lines_allocated(5,32,1,1)
    if(trim(mode).eq.'native'.or.trim(mode).eq.'explicit')then
      fail_native_fold=2
      if(trim(mode).eq.'explicit')FKSExplicitSum=.true.
    else
      values(2)=nan
      select case(trim(mode))
      case('outer')
      case('born')
        abrv='born'
      case('virtual')
        abrv='virt'
      case('uncut')
        born_cuts=.false.
      case('legacy')
        MCExplicitKLSum=.false.
      case default
        error stop 'unknown test mode'
      end select
    endif
    dummy=sigintF(x,1d0,0,answer)
    if(fail_native_fold.eq.2.and.icontr.ne.4)error stop 'missing earlier-fold records'
    dummy=sigintF(x,1d0,1,answer)
    old_calls=born_calls
    dummy=sigintF(x,1d0,1,answer)
    if(born_calls.ne.old_calls)error stop 'evaluated a later fold of a rejected point'
    dummy=sigintF(x,1d0,2,answer)
    if(any(answer.ne.0d0).or.icontr.ne.0)error stop 'retained part of a rejected point'
    if(any(virt_wgt_mint.ne.0d0).or.any(born_wgt_mint.ne.0d0).or.virtual_over_born.ne.0d0) &
         error stop 'retained virtual/Born estimates'
    if(pass_cuts_check.or.real_point_active)error stop 'retained point/cache flags'
    if(last_grid_weight.ne.0d0)error stop 'nonzero adaptive weight after rejection'
    if(ifold_counter.ne.3)error stop 'lost folding position'
    if(rejected_born_flow_points.ne.1)error stop 'counted rejected point more than once'
    ! The next point must be evaluated normally, with no stale records.
    values=[1d0,3d0,0d0]
    fail_native_fold=0
    abrv='all'
    born_cuts=.true.
    dummy=sigintF(x,1d0,0,answer)
    dummy=sigintF(x,1d0,1,answer)
    dummy=sigintF(x,1d0,1,answer)
    dummy=sigintF(x,1d0,2,answer)
    if(icontr.ne.3*merge(4,3,MCExplicitKLSum).or. &
         any(answer.ne.merge(30d0,18d0,MCExplicitKLSum)))error stop 'next valid point changed'
    if(last_grid_weight.ne.answer(1).or.real_point_active)error stop 'valid-point state changed'
    call deallocate_weight_lines()
  endif
  write(*,*) 'PASS '//trim(mode)
end program

double precision function mc_born_flow_weight(i)
  use flow_fixture
  integer :: i
  mc_born_flow_weight=values(i)
end function
double precision function ran2()
  use flow_fixture
  draws=draws+1
  ran2=uniform
end function
subroutine setup_proc_map(isum,proc_map,ini_fin)
  integer :: isum,proc_map(0:1,0:1),ini_fin
  isum=1
  proc_map=1
  ini_fin=0
end subroutine
subroutine get_born_nFKSprocess(i,j)
  integer :: i,j
  j=i
end subroutine
subroutine update_vegas_x(xx,x)
  double precision :: xx(5),x(99)
  x=0d0
  x(1:5)=xx
end subroutine
subroutine get_MC_integer(i,j,k,vol)
  integer :: i,j,k
  double precision :: vol
  k=1
  vol=1d0
end subroutine
subroutine fill_MC_integer(i,j,weight)
  use flow_fixture
  integer :: i,j
  double precision :: weight
  last_grid_weight=weight
end subroutine
logical function passcuts(p,rwgt)
  use flow_fixture
  double precision :: p(0:3,5),rwgt
  rwgt=1d0
  passcuts=born_cuts
end function
subroutine compute_born()
  use flow_fixture
  use mint_module
  born_calls=born_calls+1
  born_wgt_mint=5d0
  call add_record(1d0)
end subroutine
subroutine compute_nbody_noborn()
  use flow_fixture
  use mint_module
  double precision :: virtual_over_born
  common/c_vob/virtual_over_born
  virt_wgt_mint=7d0
  virtual_over_born=9d0
  call add_record(2d0)
end subroutine
subroutine sborn_native(p,weight)
  double precision :: p(0:3,4),weight
  weight=1d0
end subroutine
subroutine determine_partner(flow,father,partner)
  integer :: flow,father,partner
  father=3
  partner=1
end subroutine
subroutine init_process_module_n1body_wrapper(flow)
  integer :: flow
  if(flow.eq.0)error stop 'invalid flow reached shower setup'
end subroutine
subroutine include_born_flow_weight(probability,proposal)
  double precision :: probability,proposal
  if(probability.le.0d0.or.proposal.le.0d0)error stop 'invalid flow reached weight evaluation'
end subroutine
subroutine compute_native_NLOPS_weights(p,pl,pc,jac,cb,cr,probne)
  use flow_fixture
  double precision :: p(0:3,5),pl(0:3,5),pc(0:3,5),jac,probne
  logical :: cb,cr
  call add_record(3d0)
end subroutine
subroutine repartition_MC_H(first,p,pl,pc,jac,vw,sw,bf,point,valid)
  use, intrinsic :: ieee_arithmetic
  use flow_fixture
  use fks_phase_space
  implicit none
  integer :: first,fold,ifold_counter,flow
  common/cfl/fold,ifold_counter
  double precision :: p(0:3,5),pl(0:3,5),pc(0:3,5),jac,vw,sw,bf,factor
  type(fks_phase_space_point),intent(in) :: point
  logical,intent(out) :: valid
  call add_record(4d0)
  if(ifold_counter.eq.fail_native_fold)values(2)=ieee_value(0d0,ieee_quiet_nan)
  call get_born_flow(flow,factor)
  valid=flow.ne.0
end subroutine
subroutine fill_mint_function_NLOPS(answer,n1body)
  use flow_fixture
  double precision :: answer(6),n1body
  answer=sum(wgt(1,1:icontr))
  n1body=answer(1)
end subroutine
