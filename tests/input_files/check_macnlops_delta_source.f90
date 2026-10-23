! Controlled amplitudes around the ACTUAL compute_MC_subt_term routine.
module process_module
  logical :: mcatnlo_delta_mod=.true.
end module
module scale_module
  integer :: born_flow_picked=1
end module
module fks_phase_space_data
  real(8) :: xi_i_fks_ev=.5d0,y_ij_fks_ev=0d0
  real(8) :: p_born(0:3,4)=0d0,xbjrk_cnt(2,-2:2)=.1d0
end module
module extra_weights
  integer :: QCD_power=2,orders_tag=1,amp_pos=1
  real(8) :: wgtcpower=0d0
end module
module mc_counterterms
  real(8) :: gfactsf=.3d0,gfactcl=.4d0,gfactazi=.1d0
  real(8) :: requested_delta=1d0
  integer :: delta_calls=0,bogus_calls=0
contains
  subroutine compute_MCsubtraction_kl(i,j,xi,y,p,pcms,pborn,gfun,z,n,amp,scalar)
    integer :: i,j,n
    real(8) :: xi,y,p(0:3,5),pcms(0:3,5),pborn(0:3,4),z(2),amp(1,2)
    logical :: gfun,scalar
    if(.not.gfun.or.scalar)error stop 'Delta source lost G construction'
    z=.7d0;n=1;amp=0d0;amp(1,1)=40d0
  end subroutine
  subroutine compute_delta(p,probability)
    real(8) :: p(0:3,5),probability
    delta_calls=delta_calls+1
    probability=requested_delta
  end subroutine
  real(8) function bogus_probne_fun(q)
    real(8) :: q
    bogus_calls=bogus_calls+1
    bogus_probne_fun=.123d0
  end function
  real(8) function get_qMC(xi,y)
    real(8) :: xi,y
    get_qMC=2d0
  end function
end module
module source_observation
  real(8) :: source_s=0d0,source_h=0d0
end module

program check_delta_source
  use weight_lines
  use mc_counterterms
  use source_observation
  implicit none
  logical :: macnlops,macnlops_resolved_source
  common/test_source_mode/macnlops,macnlops_resolved_source
  real(8) :: p(0:3,5),probability,values(3),r,d,h,s
  real(8) :: factors(8)
  common/factor_n1body_NLOPS/factors
  integer :: legs(2),i,calls
  common/fks_indices/legs
  macnlops=.true.;macnlops_resolved_source=.false.
  p=1d0;legs=[5,1];factors=1d0
  values=[0d0,.37d0,1d0]
  do i=1,3
    call reset_delta_contexts()
    call begin_delta_context(1,.4d0)
    delta_history=1;delta_recoil=1
    requested_delta=values(i)
    source_s=0d0;source_h=0d0
    call compute_MC_subt_term(p,p,probability)
    if(abs(source_s-10d0*values(i))>1d-13)stop 1
    if(abs(source_h+10d0*values(i))>1d-13)stop 2
    if(probability/=values(i).or.bogus_calls/=0)stop 3
    if(delta_contexts(1)%count/=1)stop 4
    ! The MC term uses Delta once. Replay returns the same probability
    ! without re-evaluating tables or drawing a new conditional sample.
    delta_replay=.true.;calls=delta_calls
    source_s=0d0;source_h=0d0
    call compute_MC_subt_term(p,p,probability)
    if(delta_calls/=calls.or..not.complete_delta_replay())stop 5
    if(abs(source_s-10d0*values(i))>1d-13)stop 6
    ! Include the real and G transfers using that SAME returned factor.
    r=23d0;d=10d0+7d0
    h=values(i)*r+source_h-7d0*values(i)
    s=(1d0-values(i))*r+source_s+7d0*values(i)
    if(abs(h-values(i)*(r-d))>1d-13.or.abs(h+s-r)>1d-13)stop 7
  enddo
  print *, 'PASS native Delta MC term, no bogus product, replay and compensation'
end program

real(8) function fks_Sij(p,i,j,xi,y)
  integer :: i,j
  real(8) :: p(0:3,5),xi,y
  fks_Sij=1d0
end function
real(8) function fks_Hij(p,i,j)
  integer :: i,j
  real(8) :: p(0:3,5)
  fks_Hij=1d0
end function
subroutine get_mc_lum(j,z,xi,lum,xborn)
  integer :: j
  real(8) :: z,xi,lum,xborn(2)
  lum=1d0
end subroutine
subroutine amp_split_pos_to_orders(i,orders)
  integer :: i,orders(1)
  orders=2
end subroutine
integer function get_orders_tag(orders)
  integer :: orders(1)
  get_orders_tag=1
end function
subroutine add_wgt(kind,orders,w1,w2,w3)
  use source_observation
  integer :: kind,orders(1)
  real(8) :: w1,w2,w3
  if(kind==12)source_s=source_s+w1
  if(kind==13)source_h=source_h+w1
end subroutine
subroutine timing_enter(kind)
  integer :: kind
end subroutine
subroutine timing_leave(kind)
  integer :: kind
end subroutine
