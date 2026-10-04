program check_fks_sum_born_chart
  use mint_module
  use mc_native_context
  use fks_phase_space_data
  implicit none
  integer :: nFKSprocess,itree(2,-2:-1),iconf
  common/c_nFKSprocess/nFKSprocess
  common/to_itree/itree,iconf
  logical :: is_a_j(5),is_a_lp(5),is_a_lm(5),is_a_ph(5)
  common/to_specisa/is_a_j,is_a_lp,is_a_lm,is_a_ph
  double precision :: emass(5)
  common/to_mass/emass
  double precision :: etmin(3:4),etmax(3:4),mxxmin(3:4,3:4)
  common/to_cuts/etmin,etmax,mxxmin
  integer :: idup(5,1),mothup(2,5,1),icolup(2,5,1),niprocs
  common/c_leshouche_inc/idup,mothup,icolup,niprocs
  integer :: granny_sector
  common/test_granny/granny_sector
  double precision :: bounds(3),sampled_final(3),native_bounds(3)
  integer :: sector

  itree(:,-1)=[3,4]
  itree(:,-2)=[1,2]
  iconf=1
  is_a_j=[.false.,.false.,.true.,.false.,.true.]
  is_a_lp=.false.
  is_a_lm=.false.
  is_a_ph=.false.
  emass=[0d0,0d0,0d0,100d0,0d0]
  etmin=0d0
  etmax=-1d0
  mxxmin=0d0
  idup(:,1)=[2,1,2,23,21]

  ! Establish that ordinary ISR and FSR use different jet-cut thresholds.
  nFKSprocess=1
  call set_tau_min()
  bounds=[tau_Born_lower_bound,tau_lower_bound,tau_lower_bound_resonance]
  nFKSprocess=2
  call set_tau_min()
  sampled_final=[tau_Born_lower_bound,tau_lower_bound,tau_lower_bound_resonance]
  if(sampled_final(1).le.bounds(1))error stop 'fixture has no threshold difference'

  FKSExplicitSum=.true.
  do sector=1,2
    nFKSprocess=sector
    call set_tau_min()
    if(any([tau_Born_lower_bound,tau_lower_bound,tau_lower_bound_resonance].ne.bounds)) &
      error stop 'explicit sum used different Born charts'
    if(granny_sector.ne.sector)error stop 'recoil lost its native FKS sector'
  enddo
  ! Event replay again selects exactly the same common sampling bounds.
  nFKSprocess=2
  call set_tau_min()
  if(tau_Born_lower_bound.ne.bounds(1))error stop 'event replay changed Born chart'

  nlo_ps=.false.
  call set_tau_min()
  if(any([tau_Born_lower_bound,tau_lower_bound,tau_lower_bound_resonance].ne.sampled_final)) &
    error stop 'fixed-order chart changed'
  nlo_ps=.true.
  native_mapping=.true.
  FKSExplicitSum=.false.
  call set_tau_min()
  native_bounds=[tau_Born_lower_bound,tau_lower_bound,tau_lower_bound_resonance]
  FKSExplicitSum=.true.
  call set_tau_min()
  if(any([tau_Born_lower_bound,tau_lower_bound,tau_lower_bound_resonance].ne.native_bounds)) &
    error stop 'auxiliary matching chart changed'
  write(*,*) 'PASS common Born chart and local recoil'
end program

subroutine set_granny(sector,config,masses)
  implicit none
  integer :: sector,config,granny_sector
  double precision :: masses(*)
  common/test_granny/granny_sector
  granny_sector=sector
end subroutine
