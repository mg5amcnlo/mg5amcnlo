program check_fks_sum_weights
  use mint_module
  use weight_lines
  implicit none
  integer :: pdg_type_d(2,5),fks_i_d(2)
  common/test_fks_info/pdg_type_d,fks_i_d
  integer :: iproc_save(2),eto(1,2),etoi(1,2),maxproc_found
  common/cproc_combination/iproc_save,eto,etoi,maxproc_found
  double precision :: virtual_over_born
  common/c_vob/virtual_over_born
  double precision :: f(nintegrals),n1body,mc_signed,mc_abs,prob(2),negative_mc,negative_sum
  integer :: sector

  ndim=1
  ifold=1
  imode=0
  only_virt=.false.
  born_spread_phase=0
  virt_wgt_mint=0d0
  born_wgt_mint=0d0
  virtual_over_born=0d0
  iproc_save=1
  eto=1
  etoi=1
  pdg_type_d=21
  fks_i_d=5
  call weight_lines_allocated(5,5,1,1)
  niproc=1
  ifold_cnt=1
  event_nFKS=0
  parton_pdg=21
  momenta=0d0
  prob=[0.25d0,0.75d0]
  mc_signed=0d0
  mc_abs=0d0
  do sector=1,2
    icontr=3
    H_event(1:3)=[.false.,.false.,.true.]
    itype(1:3)=[2,12,1]
    nFKS=sector
    ! Each sampled group contains half the Born; its real terms occur once.
    parton_iproc(1,1:3)=[1d0,3d0,2d0]
    if(sector.eq.2)parton_iproc(1,1:3)=[1d0,-4d0,-1d0]
    parton_iproc(1,1:3)=parton_iproc(1,1:3)/prob(sector)
    wgts(1,1:3)=parton_iproc(1,1:3)
    call sum_identical_contributions()
    call fill_mint_function_NLOPS(f,n1body)
    mc_abs=mc_abs+prob(sector)*f(1)
    mc_signed=mc_signed+prob(sector)*f(2)
  enddo

  icontr=5
  H_event(1:5)=[.false.,.false.,.false.,.true.,.true.]
  itype(1:5)=[2,12,12,1,1]
  nFKS(1:5)=[1,1,2,1,2]
  ! The explicit list includes one full Born and both sectors at unit weight.
  parton_iproc(1,1:5)=[2d0,3d0,-4d0,2d0,-1d0]
  wgts(1,1:5)=parton_iproc(1,1:5)
  ! Distinct real momenta must not be cancelled against each other.
  momenta(0,1,4)=100d0
  momenta(0,1,5)=200d0
  call sum_identical_contributions()
  call fill_mint_function_NLOPS(f,n1body)
  if(group_size(1).ne.3.or.group_size(4).ne.1.or.group_size(5).ne.1) &
    error stop 'incorrect S/H grouping'
  if(abs(mc_signed-2d0).gt.1d-12.or.abs(f(2)-mc_signed).gt.1d-12) &
    error stop 'signed integral changed'
  if(abs(mc_abs-10d0).gt.1d-12.or.abs(f(1)-4d0).gt.1d-12) &
    error stop 'absolute integral'
  negative_mc=(mc_abs-mc_signed)/(2d0*mc_abs)
  negative_sum=(f(1)-f(2))/(2d0*f(1))
  if(abs(negative_mc-0.4d0).gt.1d-12.or.abs(negative_sum-0.25d0).gt.1d-12) &
    error stop 'unweighted negative fractions'
  write(*,*) 'PASS signed integral and negative fractions',mc_signed,negative_mc,negative_sum
end program

double precision function ran2()
  error stop 'unexpected random draw'
end function
