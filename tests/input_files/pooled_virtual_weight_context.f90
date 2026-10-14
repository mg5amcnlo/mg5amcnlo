! Controlled matrix elements around the production subtraction and weight code.
module FKSParams
  implicit none
  logical :: use_poly_virtual=.false.
  logical :: use_pooled_virtual=.false.
  logical :: separate_flavour_configs=.false.
end module

module mint_module
  use FKSParams
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  logical :: virtual_defer_requested=.false.,virtual_deferred_ready=.false.
  logical :: pooled_virtual_sampled=.false.
  logical :: pooled_exact_tail=.false.
  integer :: pooled_prepare_calls=0
  double precision :: pooled_virtual_ratio(3)=0d0,pooled_probability=0d0
  double precision :: pooled_virtual_actual_ratio(3)=0d0
  integer :: ichan=1,born_spread_phase=0
  double precision :: virtual_fraction(1)=1d0,average_virtual(0:3,1)=0d0,polyfit(0:3)=0d0
  double precision :: virt_wgt_mint(0:3)=0d0,born_wgt_mint(0:3)=0d0
  double precision :: proposal_ratio(3)=0d0
contains
  subroutine observe_virtual_ratio(virtual,born)
    double precision,intent(in) :: virtual(:),born(:)
  end subroutine
  double precision function pooled_virtual_call_probability(base) result(probability)
    double precision,intent(in) :: base
    probability=base
    if(pooled_exact_tail) probability=1d0
  end function
  subroutine pooled_virtual_prepare(p,mass,incoming,pdg,mur2,ratio,qes2,conditional,context_valid)
    double precision p(0:3,4),mass(4),mur2,ratio(:),qes2
    logical incoming(4)
    logical,optional :: conditional,context_valid
    integer pdg(4)
    if(any(mass.ne.[0d0,0d0,2d0,3d0]))error stop 'pooled Born mass ordering'
    if(any(pdg.ne.[21,21,6,-6]))error stop 'pooled Born PDG ordering'
    if(any(incoming.neqv.[.true.,.true.,.false.,.false.]))error stop 'pooled incoming ordering'
    if(mur2.ne.25d0)error stop 'pooled scale'
    if(qes2.ne.9d0)error stop 'pooled loop reference scale'
    pooled_prepare_calls=pooled_prepare_calls+1
    pooled_virtual_ratio=proposal_ratio
    pooled_virtual_actual_ratio=proposal_ratio
    pooled_virtual_sampled=.false.
  end subroutine
  subroutine pooled_virtual_mark_sampled(probability)
    double precision probability
    pooled_virtual_sampled=.true.
    pooled_probability=probability
  end subroutine
end module

subroutine prepare_pooled_helicities(p)
  double precision p(0:3,4)
end subroutine

subroutine deferred_virtual_context(action,born)
  integer action
  double precision born
  error stop 'deferred branch requires its dedicated native fixture'
end subroutine

module pooled_weight_fixture
  use mint_module
  implicit none
  integer,parameter :: nexternal=5,nincoming=2,amp_split_size=3,qcd_pos=1,qed_pos=2
  integer :: i,iamp,orders(2),fold=0,ickkw=0,i_fks=3,loop_calls=0
  integer :: active_context=1,local_context=1
  type fixture_born_metadata
     integer :: born_ids(4,1)=reshape([21,21,6,-6],[4,1])
  end type
  type(fixture_born_metadata)::native_metadata
  double precision :: p_born(0:3,4)=1d0,pmass(5)=[0d0,0d0,99d0,2d0,3d0]
  double precision :: feature_mass(4),QES2=9d0,scale=5d0
  logical :: feature_incoming(4)
  integer :: feature_pdg(4)
  complex*16 :: amp_split_cnt(3,2,2)=(0d0,0d0)
  double precision :: amp_split_virt(3)=0d0,amp_split_born_for_virt(3)=0d0
  double precision :: amp_split_avv(3)=0d0,amp_split_finite_ML(3)=0d0
  double precision :: amp_split_virt_save(3)=0d0,virt_wgt_save=0d0
  double precision :: exact_virtual(3)=0d0,uniform=0d0
  double precision :: selected_ratio(3)=0d0
  logical :: conditional_selection=.false.
  double precision :: wgt1=0d0,born_wgt=0d0,virt_wgt=0d0,avv_wgt=0d0
  double precision :: virtual_over_born=0d0,tBefore,tAfter,tOLP=0d0,virtual_prob=0d0
  double precision :: f_nb=2d0,f_b=3d0,g=2d0
  character*4 :: abrv='all'
  double precision,external :: ran2
  integer,external :: get_orders_tag
contains
  subroutine set_point(born,virtual,ratio)
    double precision born(3),virtual(3),ratio(3)
    amp_split_cnt=(0d0,0d0)
    amp_split_cnt(:,1,1)=born
    exact_virtual=virtual
    proposal_ratio=ratio
    use_pooled_virtual=.true.
    pooled_virtual_sampled=.false.
    pooled_exact_tail=.false.
    conditional_selection=.false.
    born_wgt=sum(born)
    virt_wgt_mint=0d0
    born_wgt_mint=0d0
    amp_split_avv=0d0
    avv_wgt=0d0
    fold=0
    uniform=0d0
    abrv='all'
    virtual_fraction=1d0
  end subroutine
end module

subroutine sborn_native(p,weight)
  use pooled_weight_fixture, only: born_wgt
  implicit none
  double precision p(0:3,4),weight
  weight=born_wgt
end subroutine

subroutine BinothLHA(p,born,virtual)
  use pooled_weight_fixture, only: exact_virtual,amp_split_finite_ML,loop_calls, &
       conditional_selection,selected_ratio,pooled_virtual_actual_ratio
  implicit none
  double precision p(0:3,4),born,virtual
  loop_calls=loop_calls+1
  virtual=sum(exact_virtual)
  amp_split_finite_ML=exact_virtual
  if(conditional_selection)pooled_virtual_actual_ratio=selected_ratio
end subroutine

double precision function ran2()
  use pooled_weight_fixture, only: uniform
  ran2=uniform
end function

subroutine amp_split_pos_to_orders(i,orders)
  implicit none
  integer i,orders(2)
  orders=[2*i,0]
end subroutine

integer function get_orders_tag(orders)
  implicit none
  integer orders(2)
  get_orders_tag=orders(1)
end function

subroutine add_wgt(kind,orders,w1,w2,w3)
  ! The tracking block should not add a zero residual contribution.
  use weight_lines, only: icontr,itype
  implicit none
  integer kind,orders(2)
  double precision w1,w2,w3
  if(w1.eq.0d0.and.w2.eq.0d0.and.w3.eq.0d0)error stop 'zero residual record'
  icontr=icontr+1
  itype(icontr)=kind
end subroutine

double precision function dlum()
  implicit none
  integer iproc
  double precision pd(0:2)
  common/SUBPROC/pd,iproc
  iproc=2
  pd=[-5d0,2d0,3d0]
  dlum=5d0
end function

subroutine mc_set_history(history)
  integer history
end subroutine

subroutine set_pdg_codes(iproc,pd,fks,ict)
  use weight_lines
  implicit none
  integer iproc,fks,ict,k
  double precision pd(0:2)
  niproc(ict)=iproc
  do k=1,iproc
     parton_iproc(k,ict)=pd(k)
     parton_pdg(:,k,ict)=k
     parton_pdg_uborn(:,k,ict)=k
  enddo
end subroutine

double precision function rwgt_muR_dep_fac(mu1,mu2,power)
  implicit none
  double precision mu1,mu2,power
  rwgt_muR_dep_fac=1d0
end function

subroutine separate_flavour_config(i)
  integer i
  error stop 'unexpected separate flavours'
end subroutine

subroutine bias_weight_function(p,pdg,bias)
  implicit none
  double precision p(0:3,5),bias
  integer pdg(5)
  bias=7d0
end subroutine
