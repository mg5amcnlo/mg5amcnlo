! Controlled amplitudes around the ACTUAL compute_MC_subt_term and
! compute_native_NLOPS_weights routines. The production wrapper must send
! one source probability to the MC, G and real-emission contributions.
module process_module
  logical :: mcatnlo_delta_mod=.true.
end module
module scale_module
  integer :: born_flow_picked=1
end module
module fks_phase_space_data
  real(8) :: xi_i_fks_ev=.5d0,y_ij_fks_ev=0d0
  real(8) :: p_born(0:3,4)=0d0,xbjrk_cnt(2,-2:2)=.1d0
  real(8) :: p1_cnt(0:3,5,-2:2)=1d0
end module
module mint_module
  logical :: pass_cuts_check=.false.
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
    gfactsf=.3d0;gfactcl=.4d0;gfactazi=.1d0
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
  use weight_lines, only: mc_H_only,mc_S_only
  real(8) :: source_s=0d0,source_h=0d0
  real(8) :: counter_arguments(3)=0d0,real_argument=0d0
  integer :: contribution_calls(4)=0
contains
  subroutine reset_observation()
    source_s=0d0;source_h=0d0
    counter_arguments=-1d0;real_argument=-1d0;contribution_calls=0
  end subroutine
  subroutine record_weight(hard,weight)
    logical, intent(in) :: hard
    real(8), intent(in) :: weight
    ! Model the contribution sink's H/S selection; the production wrapper
    ! must preserve these caller flags throughout its contribution calls.
    if(hard)then
      if(.not.mc_S_only)source_h=source_h+weight
    else
      if(.not.mc_H_only)source_s=source_s+weight
    endif
  end subroutine
  subroutine record_counterterm(index,coefficient,amplitude)
    integer, intent(in) :: index
    real(8), intent(in) :: coefficient,amplitude
    contribution_calls(index)=contribution_calls(index)+1
    counter_arguments(index)=coefficient
    call record_weight(.true.,-amplitude*coefficient)
    call record_weight(.false.,amplitude*coefficient)
  end subroutine
end module

program check_delta_source
  use weight_lines
  use mc_counterterms
  use source_observation
  use process_module
  use mint_module
  implicit none
  logical :: macnlops,macnlops_resolved_source
  common/test_source_mode/macnlops,macnlops_resolved_source
  real(8) :: p(0:3,5),probability,values(3)
  real(8) :: factors(8)
  common/factor_n1body_NLOPS/factors
  integer :: legs(2),i,calls,mode,old_delta,old_bogus
  common/fks_indices/legs
  character(4) :: abrv
  common/to_abrv/abrv
  macnlops=.true.;macnlops_resolved_source=.false.
  p=1d0;legs=[5,1];factors=1d0;abrv='all '
  values=[0d0,.37d0,1d0]
  do i=1,3
    call reset_delta_contexts()
    call begin_delta_context(1,.4d0)
    delta_history=1;delta_recoil=1
    requested_delta=values(i)
    call reset_observation()
    call compute_MC_subt_term(p,p,probability)
    if(abs(source_s-10d0*values(i))>1d-13)stop 1
    if(abs(source_h+10d0*values(i))>1d-13)stop 2
    if(probability/=values(i).or.bogus_calls/=0)stop 3
    if(delta_contexts(1)%count/=1)stop 4
    ! The MC term uses Delta once. Replay returns the same probability
    ! without re-evaluating tables or drawing a new conditional sample.
    delta_replay=.true.;calls=delta_calls
    call reset_observation()
    call compute_MC_subt_term(p,p,probability)
    if(delta_calls/=calls.or..not.complete_delta_replay())stop 5
    if(abs(source_s-10d0*values(i))>1d-13)stop 6
  enddo
  print *, 'PASS native Delta MC term, no bogus product and replay'

  ! Exercise the actual source assembly, including its G and real calls.
  ! The controlled amplitudes below supply the responses, not the call
  ! order or probability coefficients being checked here.
  do i=1,3
    do mode=0,2
      mc_H_only=mode==1;mc_S_only=mode==2
      call reset_delta_contexts()
      call begin_delta_context(1,.4d0)
      delta_history=1;delta_recoil=1
      requested_delta=values(i)
      old_delta=delta_calls;old_bogus=bogus_calls
      call run_wrapper(values(i),mode)
      if(delta_calls/=old_delta+1.or.bogus_calls/=old_bogus)stop 8
      if(delta_contexts(1)%count/=1)stop 9

      ! Changing the table answer must have no effect during H replay.
      delta_replay=.true.;requested_delta=1d0-values(i)
      call run_wrapper(values(i),mode)
      if(delta_calls/=old_delta+1.or.bogus_calls/=old_bogus)stop 10
      if(.not.complete_delta_replay())stop 11
    enddo
  enddo
  print *, 'PASS native Delta wrapper: real/G compensation, filters and replay'

  ! The same assembly must still use the ordinary bogus factor exactly
  ! once when Delta is off, with neither source context nor Delta calls.
  call reset_delta_contexts()
  mcatnlo_delta_mod=.false.
  do mode=0,2
    mc_H_only=mode==1;mc_S_only=mode==2
    old_delta=delta_calls;old_bogus=bogus_calls
    call run_wrapper(.123d0,mode)
    if(delta_calls/=old_delta.or.bogus_calls/=old_bogus+1)stop 12
    if(delta_context_count/=0)stop 13
  enddo
  print *, 'PASS native ordinary bogus-P wrapper'
contains
  subroutine assert_close(actual,expected,label)
    real(8), intent(in) :: actual,expected
    character(*), intent(in) :: label
    if(abs(actual-expected)>1d-12)then
      print *, label,': got ',actual,' expected ',expected
      error stop 'Incorrect native source assembly'
    endif
  end subroutine
  subroutine run_wrapper(expected_probability,selection)
    real(8), intent(in) :: expected_probability
    integer, intent(in) :: selection
    real(8) :: expected_h,expected_s
    call reset_observation()
    pass_cuts_check=.false.
    call compute_native_NLOPS_weights(p,p,.true.,.true.,probability)
    call assert_close(probability,expected_probability,'source probability')
    if(.not.pass_cuts_check.or.macnlops_gate_active)stop 14
    if((mc_H_only.neqv.(selection==1)).or. &
       (mc_S_only.neqv.(selection==2)))stop 15
    if(any(contribution_calls/=1))stop 16
    call assert_close(counter_arguments(1),.7d0*expected_probability,'soft G coefficient')
    call assert_close(counter_arguments(2),.42d0*expected_probability,'collinear G coefficient')
    call assert_close(counter_arguments(3),.42d0*expected_probability,'soft-collinear G coefficient')
    call assert_close(real_argument,expected_probability,'real coefficient')
    ! R=23 and D_G=10+4*.7+6*.42-2*.42=14.48 for the chosen
    ! amplitudes. These are the paired-source identities under test.
    expected_h=expected_probability*(23d0-14.48d0)
    expected_s=23d0-expected_h
    if(selection==1)expected_s=0d0
    if(selection==2)expected_h=0d0
    call assert_close(source_h,expected_h,'H contribution')
    call assert_close(source_s,expected_s,'S contribution')
    if(selection==0)call assert_close(source_h+source_s,23d0,'paired total')
  end subroutine
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
  if(kind==12)call record_weight(.false.,w1)
  if(kind==13)call record_weight(.true.,w1)
end subroutine
subroutine compute_soft_counter_term(coefficient)
  use source_observation
  real(8) :: coefficient
  call record_counterterm(1,coefficient,4d0)
end subroutine
subroutine compute_collinear_counter_term(coefficient)
  use source_observation
  real(8) :: coefficient
  call record_counterterm(2,coefficient,6d0)
end subroutine
subroutine compute_soft_collinear_counter_term(coefficient)
  use source_observation
  real(8) :: coefficient
  call record_counterterm(3,coefficient,-2d0)
end subroutine
subroutine compute_real_emission(p,probability)
  use source_observation
  real(8) :: p(0:3,5),probability
  contribution_calls(4)=contribution_calls(4)+1
  real_argument=probability
  call record_weight(.true.,23d0*probability)
  call record_weight(.false.,23d0*(1d0-probability))
end subroutine
subroutine set_cms_stuff(frame)
  integer :: frame
end subroutine
subroutine set_FxFx_scale(mode,p)
  integer :: mode
  real(8) :: p(0:3,*)
  error stop 'Unmerged source unexpectedly requested an FxFx scale'
end subroutine
subroutine set_alphaS(p)
  real(8) :: p(0:3,*)
end subroutine
subroutine include_multichannel_enhance(kind)
  integer :: kind
end subroutine
subroutine timing_enter(kind)
  integer :: kind
end subroutine
subroutine timing_leave(kind)
  integer :: kind
end subroutine
