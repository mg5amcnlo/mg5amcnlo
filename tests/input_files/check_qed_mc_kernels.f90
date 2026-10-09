! Exercise the production QED kernels with independent analytic references in
! test_qed_mc_kernels.py. The Born stub isolates correction-order dispatch.
program check_qed_mc_kernels
  use mc_counterterms
  use process_module, only: shower_mc_mod, mcatnlo_delta_mod, ickkw_mod, nincoming_mod
  use scale_module
  implicit none
  include 'orders.inc'
  type(mc_kernel_limits) :: limits
  character(16) :: mode
  integer :: code, ios, selected
  double precision :: z, t, jac, kernels(2), azimuth(2)
  double precision :: p(0:3,5), pb(0:3,4), born(amp_split_size), spin(amp_split_size)
  double precision :: ch_i, ch_j, ch_m, g
  integer :: i_type, j_type, m_type, j_pdg
  integer :: ifks,jfks,num_leading_cflows,iextra_cnt,isplitorder_born,isplitorder_cnt
  logical :: is_leading_cflow(1),calculatedBorn
  double precision :: iden_comp,iden_comp_FKS(1)
  double complex :: gal(2)
  logical :: split_type(nsplitorders)
  common /cparticle_types/ ch_i,ch_j,ch_m,i_type,j_type,m_type,j_pdg
  common /test_couplings/ g,gal
  common /c_split_type/ split_type
  common /fks_indices/ ifks,jfks
  common /c_leading_cflows/ is_leading_cflow,num_leading_cflows
  common /c_extra_cnt/ iextra_cnt,isplitorder_born,isplitorder_cnt
  common /c_iden_comp/ iden_comp,iden_comp_FKS
  common /ccalculatedBorn/ calculatedBorn
  call get_command_argument(1,mode)
  shower_mc_mod='PYTHIA8'
  mcatnlo_delta_mod=.false.
  ickkw_mod=0
  nincoming_mod=2
  born_flow_picked=1
  ifks=5
  jfks=3
  iextra_cnt=0
  is_leading_cflow=.true.
  num_leading_cflows=1
  calculatedBorn=.false.
  iden_comp=1d0
  ch_m=0d0
  ch_j=1d0
  ch_i=-1d0
  m_type=1
  i_type=1
  j_type=1
  ileg=4
  if (mode.eq.'dispatch') then
    read(*,*) split_type,shower_mc_mod,mcatnlo_delta_mod,ickkw_mod
    p=0d0
    pb=0d0
    call prepare_MCsubtraction_born(p,0.2d0,0.5d0,pb,selected,born,spin)
    write(*,*) selected,born,spin
    stop
  endif
  if (mode.eq.'born') then
    read(*,*) i_type,j_type,m_type,ch_i,ch_j,ch_m
    p=0d0
    pb=0d0
    call get_mbar(p,0.2d0,0.5d0,pb,ileg,1,qed_pos,born,spin)
    write(*,*) born,spin
    stop
  endif
  shat_n1=1000d0
  x=0.7d0
  yij=0.2d0
  xij=0.6d0
  xm12=0d0
  w2=10d0
  kn=100d0
  kn0=sqrt(kn**2+4d0)
  knbar=120d0
  z=0.6d0
  t=20d0
  jac=7d0
  gal=cmplx(0.3d0,0d0,kind=8)
  limits%nonsoft=.true.
  if (mode.eq.'radiators') read(*,*) shower_mc_mod,z
  do
    read(*,*,iostat=ios) ileg,limits%collinear,i_type,j_type,m_type, &
         ch_i,ch_j,ch_m,j_pdg,g
    if (ios.ne.0) exit
    call compute_splitting_kernels(kernels,azimuth,z,t,jac,limits)
    write(*,'(4ES25.16)') kernels,azimuth
  enddo
end program

subroutine sborn_native(pb,weight)
  implicit none
  include 'orders.inc'
  double precision :: pb(0:3,4),weight
  integer :: iord
  double complex :: ans_cnt(2,nsplitorders)
  common /c_born_cnt/ ans_cnt
  do iord=1,nsplitorders
    ans_cnt(:,iord)=dble(iord)
    amp_split_cnt(:,:,iord)=dble(iord)
  enddo
  weight=1d0
end subroutine

double precision function mc_born_flow_weight(iflow)
  implicit none
  integer :: iflow
  mc_born_flow_weight=1d0
end function

double complex function mc_born_azimuth_phase(p,xi,y,rotate_beam)
  implicit none
  double precision :: p(0:3,5),xi,y
  logical :: rotate_beam
  mc_born_azimuth_phase=cmplx(1d0,0d0,kind=8)
end function

subroutine getaziangles(p,cphi,sphi)
  implicit none
  double precision :: p(0:3),cphi,sphi
  cphi=1d0
  sphi=0d0
end subroutine
