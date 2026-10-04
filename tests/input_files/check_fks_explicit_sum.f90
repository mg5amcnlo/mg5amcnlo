program check_fks_explicit_sum
  use FKSParams
  implicit none
  include 'nFKSconfigs.inc'
  include 'run.inc'
  integer :: map(0:fks_configs,0:fks_configs),oldmap(0:fks_configs,0:fks_configs)
  integer :: mode,channel,i,j,n,picked,seen(fks_configs),oldseen(fks_configs)
  double precision :: volume
  logical :: flat_grid
  common/to_readgrid/flat_grid
  character(512) :: arg
  call get_command_argument(1,arg)
  select case(trim(arg))
  case('card')
    if(FKSExplicitSum)error stop 'declaration default'
    call FKSParamReader('explicit.dat',.false.,.true.)
    if(.not.FKSExplicitSum)error stop 'explicit card'
    if(.not.MCExplicitKLSum)error stop 'changed matching default'
    call FKSParamReader('legacy_matching.dat',.false.,.true.)
    if(.not.FKSExplicitSum.or.MCExplicitKLSum)error stop 'matching switches coupled'
    call FKSParamReader('sampled.dat',.false.,.true.)
    if(FKSExplicitSum)error stop 'sampled card'
    call FKSParamReader('explicit.dat',.false.,.true.)
    call FKSParamReader('old.dat',.false.,.true.)
    if(FKSExplicitSum)error stop 'old card default'
    FKSExplicitSum=.true.
    call DefaultFKSParam()
    if(FKSExplicitSum)error stop 'default reset'
    call get_command_argument(2,arg)
    call FKSParamReader(trim(arg),.true.,.true.)
    if(FKSExplicitSum)error stop 'template default'
    write(*,*) 'PASS card'
  case('map')
    call get_command_argument(2,arg)
    read(arg,*)channel
    ickkw=0
    call setup_proc_map(mode,oldmap,channel)
    if(mode.ne.3)error stop 'sampled mode'
    n=3
    if(channel.eq.1)n=1
    if(channel.eq.2)n=2
    if(oldmap(0,0).ne.n)error stop 'sampled groups'
    oldseen=0
    do i=1,oldmap(0,0)
      if(oldmap(i,0).ne.2)error stop 'sampled group size'
      if(oldmap(i,1).gt.3)error stop 'sampled owner not soft'
      do j=1,oldmap(i,0)
        oldseen(oldmap(i,j))=oldseen(oldmap(i,j))+1
      enddo
    enddo
    FKSExplicitSum=.true.
    call setup_proc_map(mode,map,channel)
    if(mode.ne.1.or.map(0,0).ne.1)error stop 'explicit mode'
    if(map(1,0).ne.2*n)error stop 'explicit group size'
    if(map(1,1).ne.oldmap(1,1))error stop 'lost soft owner'
    seen=0
    do j=1,map(1,0)
      seen(map(1,j))=seen(map(1,j))+1
    enddo
    if(any(seen.ne.oldseen).or.any(seen.gt.1))error stop 'sector coverage'
    if(seen(7).ne.0)error stop 'integrated auxiliary native sector'
    flat_grid=.false.
    call get_MC_integer(1,map(0,0),picked,volume)
    if(picked.ne.1.or.volume.ne.1d0)error stop 'explicit sampling weight'
    ! Born is counted once; every real/MC/G sector has unit weight.
    if(1d0/(map(0,0)*volume).ne.1d0)error stop 'Born normalization'
    if(dble(map(1,0))/volume.ne.2*n)error stop 'real normalization'
    call fill_MC_integer(1,picked,2d0)
    call regrid_MC_integer()
    call get_MC_integer(1,map(0,0),picked,volume)
    if(volume.ne.1d0)error stop 'adapted explicit weight'
    write(*,*) 'PASS map and sampling'
  case('unlops')
    ickkw=4
    call setup_proc_map(mode,map,0)
    if(mode.ne.0.or.map(0,0).ne.6)error stop 'sampled UNLOPS changed'
    FKSExplicitSum=.true.
    call setup_proc_map(mode,map,0)
    error stop 'accepted explicit UNLOPS'
  case default
    error stop 'unknown test'
  end select
end program

subroutine fks_inc_chooser()
  implicit none
  include 'nexternal.inc'
  integer :: nFKSprocess,i_fks,j_fks
  common/c_nFKSprocess/nFKSprocess
  common/fks_indices/i_fks,j_fks
  integer :: partners(nexternal,0:nexternal),particle_type(nexternal),pdg_type(nexternal)
  common/c_fks_inc/partners,particle_type,pdg_type
  logical :: need_color_links,need_charge_links
  common/c_need_links/need_color_links,need_charge_links
  i_fks=5
  j_fks=mod(nFKSprocess-1,3)+1
  pdg_type=21
  if(nFKSprocess.gt.3)pdg_type(i_fks)=2
  need_color_links=nFKSprocess.le.3
  need_charge_links=.false.
end subroutine

double precision function dlum()
  implicit none
  include 'genps.inc'
  integer :: iproc
  double precision :: pd(0:maxproc)
  common/subproc/pd,iproc
  iproc=1
  dlum=1d0
end function

double precision function ran2()
  error stop 'explicit sum sampled a random sector'
end function
