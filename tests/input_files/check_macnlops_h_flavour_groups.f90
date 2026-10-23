program check_h_groups
  use weight_lines
  use mint_module
  use scale_module
  use extra_weights
  use mc_native_context
  implicit none
  logical macnlops,Hevents
  integer Flavour_Bias(2),i,j,k,first_pdg(5,3,2),first_ids(3,2)
  double precision first_weights(3,2),inv_bias
  character*7 event_norm
  common/test_mac_mode/macnlops,Flavour_Bias
  common/SHevents/Hevents
  common/event_normalisation/event_norm

  call weight_lines_allocated(5,2,1,3)
  icontr=2
  H_event=.true.; niproc=3; event_nFKS=1; nFKS(1:2)=[1,2]
  ifold_cnt=1; itype=13; momenta=0d0; momenta(0,1,:)=100d0
  momenta_m=0d0; momenta_m(0,1,:,:)=100d0
  parton_pdg=0; parton_pdg_uborn=0
  native_ids(:,1)=[7,8,1];native_ids(:,2)=[9,10,2]
  do i=1,2
    do k=1,5
      history_permutations(k,i)=k
    enddo
  enddo
  do j=1,3
    parton_pdg(:,j,1)=[j,21,6,-6,j]
  enddo
  parton_pdg(:,1,2)=parton_pdg(:,2,1)
  parton_pdg(:,2,2)=parton_pdg(:,3,1)
  parton_pdg(:,3,2)=parton_pdg(:,1,1)
  ! Each native contribution has opposite sign, but the complete H is
  ! positive in every physical channel: d=4, u=3, s=5.
  parton_iproc(1:3,1)=[-5d0,-7d0,-11d0]
  parton_iproc(1:3,2)=[10d0,16d0,9d0]
  first_pdg=parton_pdg(:,1:3,1:2)
  first_weights=parton_iproc(1:3,1:2)
  first_ids=native_ids(:,1:2)
  macnlops=.true.
  call sum_identical_contributions
  if(group_size(1)/=2.or.group_size(2)/=0)stop 1
  if(any(unwgt(1:3,1)/=[4d0,3d0,5d0]))stop 2
  if(any(parton_pdg(:,1:3,1:2)/=first_pdg))stop 3
  if(any(parton_iproc(1:3,1:2)/=first_weights))stop 4
  if(any(native_ids(:,1:2)/=first_ids))stop 5
  if(group_flavour_row(1,1,2)/=3)stop 6
  if(group_flavour_row(1,2,2)/=1)stop 7
  if(group_flavour_row(1,3,2)/=2)stop 8

  ! Exercise both downstream consumers with a selected physical u row.
  ! A same-index lookup would use d instead and fail the weight check.
  Hevents=.true.;icontr_picked=1;iproc_picked=2
  event_norm='bias';Flavour_Bias=[0,1];bias_wgt=2d0
  call include_inverse_bias_wgt(inv_bias)
  if(abs(inv_bias-.5d0)>1d-14)stop 9
  wgt=0d0;wgt_ME_tree=0d0;bjx=.1d0;scales2=100d0;g_strong=1d0
  QCDpower=6;orderstag=6
  call fill_rwgt_lines
  if(n_ctr_found/=2)stop 10
  do i=1,n_ctr_found
    write(*,'(a)')'RWGT '//trim(n_ctr_str(i))
  enddo

  ! Ordinary generation retains its original independently sampled rows.
  macnlops=.false.
  call sum_identical_contributions
  if(any(group_size(1:2)/=[1,1]))stop 11
  if(any(unwgt(1:3,1:2)/=first_weights))stop 12

  ! Repeated labels map by occurrence; missing multiplicity must fail.
  parton_pdg(:,2,1)=parton_pdg(:,1,1)
  parton_pdg(:,1,2)=parton_pdg(:,1,1)
  parton_pdg(:,2,2)=parton_pdg(:,3,1)
  parton_pdg(:,3,2)=parton_pdg(:,1,1)
  if(group_flavour_row(1,1,2)/=1)stop 13
  if(group_flavour_row(1,2,2)/=3)stop 14
  if(group_flavour_row(1,3,2)/=2)stop 15
  parton_pdg(:,3,2)=parton_pdg(:,3,1)
  if(group_flavour_row(1,2,2)/=0)stop 16
  macnlops=.true.
  call sum_identical_contributions
  if(any(group_size(1:2)/=[1,1]))stop 17
  print *,'PASS native H flavour groups, bias and reweight rows'
end program

logical function momenta_equal(p,q)
  implicit none
  double precision p(0:3,5),q(0:3,5)
  momenta_equal=all(p==q)
end function

logical function pdg_equal(p,q)
  implicit none
  integer p(5),q(5)
  pdg_equal=all(p==q)
end function

logical function colour_con_equal()
  colour_con_equal=.true.
end function
