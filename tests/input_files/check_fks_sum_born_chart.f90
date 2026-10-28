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
  double precision :: foreign_bounds(3),foreign_native_bounds(3)
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

  ! Recoil selection temporarily clears native_mapping while preparing an
  ! inner history. Its own cut bounds must survive a cached outer chart.
  active_history=1
  call set_tau_min()
  if(any([tau_Born_lower_bound,tau_lower_bound,tau_lower_bound_resonance].ne.sampled_final)) &
    error stop 'inner history reused the outer common Born chart'

  ! A foreign history can have no integrated outer sector at all.
  active_context=2
  nFKSprocess=3
  call set_tau_min()
  if(any([tau_Born_lower_bound,tau_lower_bound,tau_lower_bound_resonance].ne.bounds)) &
    error stop 'foreign history changed its own Born chart'
  if(granny_sector.ne.3)error stop 'foreign history lost its recoil sector'

  active_context=1
  nFKSprocess=2
  active_history=0
  call set_tau_min()
  if(any([tau_Born_lower_bound,tau_lower_bound,tau_lower_bound_resonance].ne.bounds)) &
    error stop 'inner history changed the cached outer chart'

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
  ! An auxiliary history can use a Born context absent from the integrated
  ! sectors. Its recoil selection still needs the ordinary generation cuts.
  nFKSprocess=3
  active_context=2
  native_mapping=.false.
  FKSExplicitSum=.false.
  call set_tau_min()
  foreign_bounds=[tau_Born_lower_bound,tau_lower_bound,tau_lower_bound_resonance]
  native_mapping=.true.
  call set_tau_min()
  foreign_native_bounds=[tau_Born_lower_bound,tau_lower_bound,tau_lower_bound_resonance]
  if(all(foreign_bounds.eq.foreign_native_bounds))error stop 'fixture has no native cut difference'
  FKSExplicitSum=.true.
  do sector=1,2
    ! Exercise both a fresh calculation and cached recoil/physical bounds.
    call select_native_recoil(1)
    if(.not.native_mapping.or.active_context.ne.2)error stop 'native context changed'
    if(granny_sector.ne.3)error stop 'foreign recoil lost its FKS sector'
    if(any([tau_Born_lower_bound,tau_lower_bound,tau_lower_bound_resonance].ne.foreign_bounds)) &
      error stop 'foreign recoil lost its generation cuts'
    call set_tau_min()
    if(any([tau_Born_lower_bound,tau_lower_bound,tau_lower_bound_resonance].ne.foreign_native_bounds)) &
      error stop 'foreign native map lost its physical bounds'
  enddo
  active_context=1
  nFKSprocess=2
  native_mapping=.false.
  call set_tau_min()
  if(any([tau_Born_lower_bound,tau_lower_bound,tau_lower_bound_resonance].ne.bounds)) &
    error stop 'outer chart changed after foreign recoil'
  write(*,*) 'PASS common Born chart and local recoil'
end program

subroutine set_granny(config,masses)
  implicit none
  integer :: sector,config,granny_sector
  double precision :: masses(*)
  integer :: igranny,iaunt
  logical :: granny_is_res,granny_chain(-5:5),granny_chain_real_final(-5:5)
  common/c_nFKSprocess/sector
  common/test_granny/granny_sector
  common/c_granny_res/igranny,iaunt,granny_is_res,granny_chain,granny_chain_real_final
  granny_sector=sector
  igranny=0
  iaunt=0
  granny_is_res=.false.
  granny_chain=.false.
  granny_chain_real_final=.false.
end subroutine

subroutine select_fks_recoil(members,allow)
  implicit none
  logical :: members(5),allow
end subroutine
