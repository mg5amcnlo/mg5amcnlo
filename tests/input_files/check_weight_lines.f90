! Exercise contribution growth, including growth of the other dimensions.
program check_weight_lines
  use weight_lines
  implicit none
  integer :: i,j,k,n,old_capacity,old_proc,old_wgt,growths,requested
  call weight_lines_allocated(5,1,1,1)
  pdg(:,0)=-91
  pdg_uborn(:,0)=-92
  growths=0
  do n=1,257
     old_capacity=max_contr
     old_proc=max_iproc
     old_wgt=max_wgt
     requested=n
     call weight_lines_allocated(5,requested,merge(9,1,n.ge.30),merge(5,1,n.ge.19))
     if(requested.ne.n)error stop 'allocation changed requested contribution count'
     if(max_contr.ne.old_capacity)growths=growths+1
     if(max_contr.lt.n.or.max_contr.gt.max(32,2*n))error stop 'invalid reserved capacity'
     if(any(pdg(:,0).ne.-91).or.any(pdg_uborn(:,0).ne.-92))error stop 'lost zero-index PDGs'
     do i=1,n-1
        call check_record(i,old_proc,old_wgt)
     enddo
     ! Initialize new flavour/weight entries as those dimensions increase.
     do i=1,n
        call fill_record(i)
     enddo
  enddo
  if(growths.gt.6)error stop 'contribution storage still grows per append'
  icontr=257
  group_size=0
  do i=1,icontr
     call add_group_member(1+mod(i-1,7),i)
  enddo
  call pack_contribution_groups()
  do i=1,7
     k=0
     do j=i,icontr,7
        k=k+1
        if(group_member(k,i).ne.j)error stop 'unstable compact membership'
     enddo
     if(group_size(i).ne.k)error stop 'wrong compact group count'
  enddo
  ! Reweighting passes module capacities as arguments; this must be safe
  ! when another dimension grows and must preserve packed membership.
  call weight_lines_allocated(5,max_contr,13,max_iproc)
  call weight_lines_allocated(5,max_contr,max_wgt,7)
  if(group_member(2,3).ne.10)error stop 'reweight growth lost membership'
  do i=1,icontr
     call check_payload(i,5,9)
  enddo
  call deallocate_weight_lines()
  if(allocated(group_members).or.allocated(native_ids))error stop 'storage not released'
  call weight_lines_allocated(5,3,2,2)
  if(any(group_size.ne.0).or.any(native_ids.ne.0))error stop 'stale initial storage'
  call deallocate_weight_lines()
  write(*,*)'PASS contribution storage'
contains
  subroutine fill_record(ict)
    integer,intent(in) :: ict
    H_event(ict)=mod(ict,2).eq.0
    itype(ict)=ict
    nFKS(ict)=ict
    event_nFKS(ict)=ict+1
    native_ids(:,ict)=[ict,ict+1,ict+2]
    QCDpower(ict)=ict
    pdg(:,ict)=ict
    pdg_uborn(:,ict)=ict
    parton_pdg_uborn(:,:,ict)=ict
    parton_pdg(:,:,ict)=ict
    plot_id(ict)=ict
    ifold_cnt(ict)=ict
    niproc(ict)=ict
    ipr(ict)=ict
    orderstag(ict)=ict
    amppos(ict)=ict
    parton_pdf(:,:,ict)=ict
    group_size(ict)=ict
    group_start(ict)=ict
    group_members(ict)=ict
    group_owner(ict)=ict
    group_cursor(ict)=ict
    momenta(:,:,ict)=ict
    momenta_m(:,:,:,ict)=ict
    wgt(:,ict)=ict
    wgt_ME_tree(:,ict)=ict
    bjx(:,ict)=ict
    scales2(:,ict)=ict
    g_strong(ict)=ict
    wgts(:,ict)=ict
    parton_iproc(:,ict)=ict
    y_bst(ict)=ict
    cpower(ict)=ict
    bias_wgt(ict)=ict
    plot_wgts(:,ict)=ict
    shower_scale(ict)=ict
    shower_scale_a(ict,:,:)=ict
    unwgt(:,ict)=ict
    need_match(:,ict)=ict
  end subroutine
  subroutine check_record(ict,nproc,nwgt)
    integer,intent(in) :: ict,nproc,nwgt
    call check_payload(ict,nproc,nwgt)
    if(any([group_size(ict),group_start(ict),group_members(ict), &
         group_owner(ict),group_cursor(ict)].ne.ict))error stop 'lost group storage'
  end subroutine
  subroutine check_payload(ict,nproc,nwgt)
    integer,intent(in) :: ict,nproc,nwgt
    if(H_event(ict).neqv.(mod(ict,2).eq.0))error stop 'lost event type'
    if(any([itype(ict),nFKS(ict),QCDpower(ict),plot_id(ict),ifold_cnt(ict), &
         niproc(ict),ipr(ict),orderstag(ict),amppos(ict)].ne.ict))error stop 'lost record IDs'
    if(event_owner(ict).ne.ict+1.or.any(native_ids(:,ict).ne.[ict,ict+1,ict+2])) &
         error stop 'lost native provenance or event owner'
    if(any(pdg(:,ict).ne.ict).or.any(pdg_uborn(:,ict).ne.ict).or. &
         any(parton_pdg_uborn(:,1:nproc,ict).ne.ict).or. &
         any(parton_pdg(:,1:nproc,ict).ne.ict).or. &
         any(parton_pdf(:,1:nproc,ict).ne.ict).or.any(need_match(:,ict).ne.ict)) &
         error stop 'lost flavour arrays'
    if(any(momenta(:,:,ict).ne.ict).or.any(momenta_m(:,:,:,ict).ne.ict).or. &
         any(wgt(:,ict).ne.ict).or.any(wgt_ME_tree(:,ict).ne.ict).or. &
         any(bjx(:,ict).ne.ict).or.any(scales2(:,ict).ne.ict))error stop 'lost kinematic arrays'
    if(any([g_strong(ict),y_bst(ict),cpower(ict),bias_wgt(ict),shower_scale(ict)].ne.ict).or. &
         any(shower_scale_a(ict,:,:).ne.ict))error stop 'lost scale data'
    if(any(wgts(1:nwgt,ict).ne.ict).or.any(plot_wgts(1:nwgt,ict).ne.ict).or. &
         any(parton_iproc(1:nproc,ict).ne.ict).or.any(unwgt(1:nproc,ict).ne.ict)) &
         error stop 'lost reweight data'
  end subroutine
end program
