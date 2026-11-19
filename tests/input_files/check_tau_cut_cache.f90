module tau_cache_fixture
  use fks_phase_space_data
  implicit none
  integer,parameter :: ne=7
  integer :: builds(2)=0,sector=2,tree(2,-4:-1),config=1
  integer :: emitter=7,spectator=3,granny,aunt,bw_max,bw(-ne:-1),bw_level(-ne:-1)
  integer :: ids(ne,1),mothers(2,ne,1),colours(2,ne,1),nprocs,forest(2,-4:-1,2)
  logical :: jets(ne),positive(ne),negative(ne),photons(ne),is_res,chain(-ne:ne),real_chain(-ne:ne)
  real(8) :: masses(ne),etmin(3:6),etmax(3:6),mxxmin(3:6,3:6),beams(2)
  real(8) :: ptj,ptgmin,ptl,mll,mll_sf,resmass(2),reswidth(2),parentmass(2),parentwidth(2),g
  real(8) :: bw_mass(-1:1,-ne:-1),bw_width(-1:1,-ne:-1),s_mass(-ne:ne)
  common/c_nfksprocess/sector
  common/to_itree/tree,config
  common/fks_indices/emitter,spectator
  common/c_granny_res/granny,aunt,is_res,chain,real_chain
  common/c_conflictingBW/bw_mass,bw_width,bw_max,bw,bw_level
  common/to_phase_space_s_channel/s_mass
  common/c_leshouche_inc/ids,mothers,colours,nprocs
  common/to_specisa/jets,positive,negative,photons
  common/to_mass/masses
  common/to_cuts/etmin,etmax,mxxmin
  common/test_beams/beams
  common/test_cuts/ptj,ptgmin,ptl,mll,mll_sf
  common/test_model/resmass,reswidth,parentmass,parentwidth,g
  common/test_forest/forest
contains
  subroutine initialize
    tree(:,-1)=[3,4]
    tree(:,-2)=[-1,5]
    tree(:,-3)=[1,-2]
    tree(:,-4)=[2,6]
    forest(:,:,1)=tree
    forest(:,:,2)=tree
    ids(:,1)=[2,-1,11,-11,21,35,21]
    mothers=0
    colours=0
    nprocs=1
    jets=.false.
    jets(5)=.true.
    jets(7)=.true.
    positive=.false.
    positive(4)=.true.
    negative=.false.
    negative(3)=.true.
    photons=.false.
    masses=0d0
    masses(6)=30d0
    etmin=0d0
    etmax=-1d0
    mxxmin=0d0
    beams=[500d0,650d0]
    ptj=40d0
    ptgmin=10d0
    ptl=5d0
    mll=0d0
    mll_sf=0d0
    resmass=[80d0,90d0]
    reswidth=[2d0,4d0]
    parentmass=[100d0,110d0]
    parentwidth=[3d0,5d0]
    g=1d0
  end subroutine

  subroutine verify(expected_builds)
    integer,intent(in) :: expected_builds
    integer :: before,integers(3+2*ne)
    real(8) :: bounds(3),s(-ne:ne),bwm(-1:1,-ne:-1),bww(-1:1,-ne:-1)
    logical :: logicals(4*ne+3)
    before=builds(1)
    call set_tau_min_fresh()
    bounds=[tau_Born_lower_bound,tau_lower_bound,tau_lower_bound_resonance]
    integers=[granny,aunt,bw_max,bw,bw_level]
    logicals=[is_res,chain,real_chain]
    s=s_mass
    bwm=bw_mass
    bww=bw_width
    ! Other maps and point restoration overwrite all these shared outputs.
    tau_Born_lower_bound=-1d0
    tau_lower_bound=-2d0
    tau_lower_bound_resonance=-3d0
    s_mass=-4d0
    bw_mass=-5d0
    bw_width=-6d0
    bw=-7
    bw_level=-8
    bw_max=-9
    granny=-10
    aunt=-11
    is_res=.not.logicals(1)
    chain=.not.logicals(2:2*ne+2)
    real_chain=.not.logicals(2*ne+3:4*ne+3)
    call set_tau_min()
    if(builds(1)-before.ne.expected_builds)then
      write(*,*) 'unexpected rebuild count',builds(1)-before,expected_builds,builds
      error stop 1
    endif
    if(any([tau_Born_lower_bound,tau_lower_bound,tau_lower_bound_resonance].ne.bounds)) error stop 'bounds'
    if(any([granny,aunt,bw_max,bw,bw_level].ne.integers)) error stop 'integer metadata'
    if(any([is_res,chain,real_chain].neqv.logicals)) error stop 'recoil metadata'
    if(any(s_mass.ne.s)) error stop 's-channel masses'
    if(any(bw_mass(-1,:).ne.bwm(-1,:)).or.any(bw_mass(1,:).ne.bwm(1,:))) error stop 'BW masses'
    if(any(bw_width(-1,:).ne.bww(-1,:)).or.any(bw_width(1,:).ne.bww(1,:))) error stop 'BW widths'
  end subroutine
end module

program check_tau_cut_cache
  use mint_module
  use mc_native_context
  use tau_cache_fixture
  implicit none
  character(32) :: mode
  integer :: i
  call get_command_argument(1,mode)
  call initialize()
  call verify(1)
  call verify(0)
  select case(trim(mode))
  case('reuse')
    native_epoch=native_epoch+1
    g=2d0
    p_born=123d0
    jets(1:2)=.true.
    call verify(0)
    active_context=2
    sector=4
    native_epoch=native_epoch+1
    call verify(1)
    active_context=1
    sector=2
    native_epoch=native_epoch+1
    call verify(0)
    sector=1
    spectator=1
    call verify(1)
    sector=2
    spectator=3
    call verify(0)
    iconfig=2
    config=2
    call verify(1)
    iconfig=1
    config=1
    call verify(0)
    ichan=2
    call verify(1)
    ichan=1
    call verify(0)
    tree(:,-1)=[3,5]
    tree(:,-2)=[-1,4]
    forest(:,:,1)=tree
    call verify(1)
  case('inputs')
    ptj=45d0
    call verify(1)
    ptl=6d0
    call verify(1)
    ptgmin=12d0
    call verify(1)
    mll=20d0
    call verify(1)
    mll_sf=25d0
    call verify(1)
    beams(1)=550d0
    call verify(1)
    masses(6)=35d0
    call verify(1)
    etmin(6)=15d0
    call verify(1)
    mxxmin(6,6)=10d0
    call verify(1)
    resmass(1)=85d0
    call verify(1)
    reswidth(1)=2.5d0
    call verify(1)
    parentmass(1)=105d0
    call verify(1)
    parentwidth(1)=3.5d0
    call verify(1)
    ids(4,1)=-13
    call verify(1)
    jets(5)=.false.
    photons(5)=.true.
    call verify(1)
    call verify(0)
  case('mapping')
    native_mapping=.true.
    call verify(1)
    native_mapping=.false.
    call verify(0)
    FKSExplicitSum=.true.
    call verify(1)
    active_history=1
    ! Native cut preparation clears native_mapping temporarily. It must
    ! retain its own bounds and must not reuse the outer common-chart entry.
    call verify(0)
    active_history=0
    call verify(0)
    nlo_ps=.false.
    call verify(1)
    nlo_ps=.true.
    call verify(0)
    FKSExplicitSum=.false.
    call verify(0)
  case('resonance')
    if(.not.is_res.or..not.real_chain(7)) error stop 'fixture needs FSR resonance'
    if(bw(-2).ne.1.or.bw_mass(1,-2).le.0d0.or.bw_mass(-1,-1).le.0d0) &
      error stop 'fixture needs conflicting resonances'
    reswidth(1)=0d0
    call verify(1)
    if(is_res)error stop 'zero-width resonance'
    reswidth(1)=2d0
    call verify(0)
    ptl=60d0
    call verify(1)
    if(is_res)error stop 'cut excluded resonance'
    ptl=5d0
    call verify(0)
  case('eviction')
    sector=1
    spectator=1
    call verify(1)
    sector=2
    spectator=3
    call verify(0)
    do i=1,70
      ptj=100d0+i
      call verify(1)
    enddo
    ptj=40d0
    sector=1
    spectator=1
    call verify(1)
    sector=2
    spectator=3
    call verify(1)
    call verify(0)
  case default
    error stop 'unknown tau cache mode'
  end select
  write(*,*) 'PASS tau cache '//trim(mode)
end program

subroutine fixture_props(pmass,pwidth,pow)
  use mc_native_context
  use tau_cache_fixture,only:resmass,reswidth,parentmass,parentwidth
  implicit none
  real(8) :: pmass(-7:0,2),pwidth(-7:0,2)
  integer :: pow(-7:0,2),i
  pmass=0d0
  pwidth=0d0
  pow=0
  do i=1,2
    pmass(-1,i)=resmass(active_context)+5d0*(i-1)
    pwidth(-1,i)=reswidth(active_context)
    pmass(-2,i)=parentmass(active_context)+5d0*(i-1)
    pwidth(-2,i)=parentwidth(active_context)
  enddo
end subroutine

subroutine count_tau_builds(tag)
  use tau_cache_fixture,only:builds
  implicit none
  integer :: tag
  builds(tag)=builds(tag)+1
end subroutine
