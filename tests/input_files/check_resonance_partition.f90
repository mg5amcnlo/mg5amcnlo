module weight_lines
  logical :: mc_H_only=.false.
end module

module mc_native_context
  integer :: active_context=1,local_context=1
  type result_type
    double precision :: diagrams(3)
  end type
  type(result_type) :: native_result
end module

program check_resonance_partition
  use fks_phase_space_data, only: pb => p_born,pbe => p_born_ev,pe => p_ev
  use mc_native_context
  implicit none
  integer,parameter :: nexternal=5,lmaxconfigs=3,max_branch=3,fks_configs=1
  double precision :: g,total,outer,native,mc,soft
  common/test_coupling/g
  double precision :: masses(-5:0,3,0:1),widths(-5:0,3,0:1)
  integer :: forests(2,-3:-1,3,0:1),sprops(-3:-1,3,0:1),tprids(-3:-1,3,0:1),maps(0:3,0:1)
  common/c_configurations/masses,widths,forests,sprops,tprids,maps
  integer :: mapconfig(0:3),iforest(2,-3:-1,3),sprop(-3:-1,3),tprid(-3:-1,3)
  logical :: forcebw(-3:-1,3)
  common/MC_NATIVE_BORN_TOPOLOGY/mapconfig,iforest,sprop,tprid,forcebw
  integer :: config,c,group_of(3)
  common/to_mconfigs/config
  logical :: calculated,evpr,granny,chain(-5:5),realchain(-5:5)
  integer :: igranny,iaunt
  common/ccalculatedBorn/calculated
  common/to_use_evpr/evpr
  common/c_granny_res/igranny,iaunt,granny,chain,realchain
  double precision :: symmetry,fr,fs,fc,fdc,fsc,fdsc(4),fms,fmh,fss,fsh,fcs,fch,fscs,fsch
  common/dsymfactor/symmetry
  common/factor_n1body/fr,fs,fc,fdc,fsc,fdsc
  common/factor_n1body_NLOPS/fss,fsh,fcs,fch,fscs,fsch,fms,fmh
  double precision,external :: mc_outer_channel_weight,native_recoil_weight

  maps=0
  maps(:,0)=[3,1,2,3]
  mapconfig=maps(:,0)
  group_of=[1,2,2]
  evpr=.true.
  symmetry=1d0
  pe=0d0
  pe(0,1:2)=300d0
  pbe=0d0
  pbe(0,1:2)=200d0
  total=0d0
  do c=1,3
    config=c
    pb=pbe
    ! Production and decay charts have different Born projections, and
    ! mixed Born orders respond differently to the coupling scale.
    if(c.gt.1)pb(0,1:2)=100d0
    granny=c.gt.1
    call set_alphas(pe)
    calculated=.false.
    fr=1d0
    call include_multichannel_enhance(2)
    outer=mc_outer_channel_weight()
    if(abs(fr-outer).gt.1d-14)error stop 'outer real partition differs'
    if(abs(g-3d0).gt.1d-14)error stop 'real coupling not restored'
    total=total+fr
  enddo
  if(abs(total-1d0).gt.1d-14)error stop 'real charts do not sum to unity'

  pb(0,1:2)=100d0
  mc=0d0
  soft=0d0
  do c=2,3
    config=c
    fms=1d0
    fmh=1d0
    call set_alphas(pe)
    calculated=.false.
    call include_multichannel_enhance(4)
    mc=mc+fms
    if(abs(g-3d0).gt.1d-14)error stop 'MC coupling not restored'
    fs=1d0
    calculated=.false.
    call set_alphas(pb)
    call include_multichannel_enhance(3)
    soft=soft+fs
  enddo
  do active_context=1,2
    native=native_recoil_weight(pb,2,group_of)
    if(abs(mc-native).gt.1d-14.or.abs(soft-native).gt.1d-14) &
         error stop 'native recoil group differs from outer MC/G weights'
    if(abs(g-3d0).gt.1d-14)error stop 'native coupling not restored'
  enddo
  print *, 'PASS resonance partitions and mixed Born orders'
end program

subroutine set_alphas(p)
  implicit none
  double precision :: p(0:3,*),g
  common/test_coupling/g
  g=(p(0,1)+p(0,2))/200d0
end subroutine

subroutine sborn_native(p,ans)
  use mc_native_context
  implicit none
  double precision :: p(0:3,*),ans,g,amp2(3),jamp2(0:1)
  common/test_coupling/g
  common/to_amps/amp2,jamp2
  native_result%diagrams=[1d0,g**2,g**4]
  ans=sum(native_result%diagrams)
  amp2=native_result%diagrams
  if(active_context.ne.local_context)amp2=-1d0
end subroutine

subroutine sborn(p,ans)
  implicit none
  double precision :: p(0:3,*),ans
  call sborn_native(p,ans)
end subroutine
