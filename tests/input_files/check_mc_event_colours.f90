module controlled_colour_random
  double precision :: next_random=0.25d0
end module

program check_mc_event_colours
  use process_module
  use scale_module
  use weight_lines
  use controlled_colour_random
  use mint_module
  implicit none
  include 'nexternal.inc'
  include 'genps.inc'
  include 'born_nhel.inc'
  ! Shared with the production grouping routine through the test include.
  integer :: pdg_type_d(2,5),fks_i_d(2)
  logical :: need_color_links_d(2),need_charge_links_d(2)
  common/test_fks_info/pdg_type_d,fks_i_d,need_color_links_d,need_charge_links_d
  integer :: idup(nexternal-1,maxproc),mothup(2,nexternal-1,maxproc)
  integer :: icolup(2,nexternal-1,max_bcol)
  include 'born_leshouche.inc'
  integer :: real_ids(nexternal,maxproc),mothers(2,nexternal,maxproc)
  integer :: colours(2,nexternal,maxflow),niprocs
  common /c_leshouche_inc/real_ids,mothers,colours,niprocs
  integer :: fks_j_from_i(nexternal,0:nexternal),particle_type(nexternal),pdg_type(nexternal)
  common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
  integer :: i_fks,j_fks,nFKSprocess,fold,ifold_counter
  common /fks_indices/i_fks,j_fks
  common /c_nFKSprocess/nFKSprocess
  common /cfl/fold,ifold_counter
  logical :: split_type(2)
  common /c_split_type/split_type
  integer :: colour_connections(2,nexternal)
  common /colour_connections_to_write/colour_connections
  integer :: saved(2,nexternal),picked,fold_picked
  integer, parameter :: ntest=257
  integer :: i,j,ii,mode,soft,reference(0:ntest,ntest),provenance(3,ntest)
  integer :: iproc_save(2),eto(1,2),etoi(1,2),maxproc_found
  common/cproc_combination/iproc_save,eto,etoi,maxproc_found
  double precision :: reference_weight(ntest)
  logical :: momenta_equal,pdg_equal
  external momenta_equal,pdg_equal

  call init_process_module_global('PYTHIA8   ','all ',5,2,.true.,13000d0,1,1,0)
  call init_scale_module(5,1d0,2,2)
  idup(:,1)=[21,21,6,-6]
  icolup(:,:,1)=reshape([501,502,502,503,501,0,0,503],[2,4])
  real_ids(:,1)=[21,21,6,-6,21]
  particle_type=[8,8,3,-3,8]
  pdg_type=real_ids(:,1)
  split_type=[.true.,.false.]
  i_fks=5
  j_fks=1
  nFKSprocess=1
  ifold_counter=1
  mc_H_only=.false.
  call init_process_module_n1body_wrapper(1)
  saved=event_colour_H(:,:,1,1)
  if (all(saved.eq.0)) error stop 'outer colours missing'

  ! A later sector/fold chooses the opposite gluon insertion.
  next_random=0.75d0
  nFKSprocess=2
  call init_process_module_n1body_wrapper(1)
  if (all(saved.eq.event_colour_H(:,:,2,1))) error stop 'test did not change colours'
  nFKSprocess=1
  ifold_counter=2
  call init_process_module_n1body_wrapper(1)
  if (all(saved.eq.event_colour_H(:,:,1,2))) error stop 'test did not change fold'

  ! Native evaluation and outer restoration reuse the insertion variate,
  ! but must not replace the saved ownership record.
  mc_H_only=.true.
  ifold_counter=1
  call init_process_module_n1body_wrapper(1)
  if (any(saved.ne.event_colour_H(:,:,1,1))) error stop 'native overwrote owner'
  mc_H_only=.false.

  call weight_lines_allocated(5,2,1,1)
  icontr=1
  niproc=1
  H_event=.true.
  native_ids=0
  nFKS=2
  event_nFKS=1
  ifold_cnt=1
  unwgt=1d0
  need_match=0
  emsca_H=100d0
  next_random=0.25d0
  call pick_unweight_contr(picked,fold_picked)
  if (picked.ne.1.or.fold_picked.ne.1) error stop 'wrong event owner'
  if (any(saved.ne.colour_connections)) error stop 'colours from a later sector'

  ! Equal momenta/scales with different colours are distinct events.
  icontr=2
  event_nFKS(2)=2
  parton_pdg(:,1,1)=real_ids(:,1)
  parton_pdg(:,1,2)=real_ids(:,1)
  momenta=1d0
  parton_iproc(1,1)=1d0
  parton_iproc(1,2)=-0.5d0
  call sum_identical_contributions()
  if (any(group_size(1:2).ne.1)) error stop 'merged distinct colours'
  event_colour_H(:,:,2,1)=saved
  call sum_identical_contributions()
  if (group_size(1).ne.2.or.group_size(2).ne.0) &
       error stop 'failed to merge identical owners'
  if (abs(unwgt(1,1)-0.5d0).gt.1d-14) error stop 'changed total weight'

  ! Compare the packed groups with the previous dense membership for
  ! mixed native H records and S counterevents. Keep insertion order,
  ! signed cancellations, different colours/folds and native provenance.
  call weight_lines_allocated(5,ntest,1,1)
  icontr=ntest
  iproc_save=1
  eto=1
  etoi=1
  pdg_type_d=21
  fks_i_d=5
  need_color_links_d=.true.
  need_charge_links_d=.false.
  event_colour_H(:,:,2,1)=event_colour_H(:,:,1,2)
  do i=1,icontr
     H_event(i)=mod(i,7).ne.0
     nFKS(i)=1+mod(i,2)
     event_nFKS(i)=1+mod(i,3)/2
     ifold_cnt(i)=1+mod(i,5)/3
     itype(i)=1
     if(.not.H_event(i))itype(i)=mod(i,3)+13
     niproc(i)=1
     parton_pdg(:,1,i)=real_ids(:,1)
     if(mod(i,11).eq.0)parton_pdg(1:2,1,i)=[1,-1]
     momenta(:,:,i)=1d0+mod(i,4)
     parton_iproc(1,i)=(-1d0)**i*(1d0+dble(i)/ntest)
     native_ids(:,i)=[i,mod(i,3)+1,0]
  enddo
  provenance=native_ids(:,1:icontr)
  do mode=0,1
     imode=mode
     reference=0
     reference_weight=0d0
     soft=7
     do i=1,icontr
        if(H_event(i))then
           do ii=1,i
              if(.not.H_event(ii))cycle
              if(any(emsca_H(event_owner(ii),ifold_cnt(ii),:,:).ne. &
                   emsca_H(event_owner(i),ifold_cnt(i),:,:)))cycle
              if(any(event_colour_H(:,:,event_owner(ii),ifold_cnt(ii)).ne. &
                   event_colour_H(:,:,event_owner(i),ifold_cnt(i))))cycle
              if(.not.pdg_equal(parton_pdg(:,1,ii),parton_pdg(:,1,i)))cycle
              if(.not.momenta_equal(momenta(:,:,ii),momenta(:,:,i)))cycle
              exit
           enddo
        else
           ii=soft
        endif
        reference(0,ii)=reference(0,ii)+1
        reference(reference(0,ii),ii)=i
        if(.not.H_event(i).and.itype(i).eq.14.and.imode.eq.1)cycle
        reference_weight(ii)=reference_weight(ii)+parton_iproc(1,i)
     enddo
     call sum_identical_contributions()
     if(any(group_size(1:icontr).ne.reference(0,:)))error stop 'group sizes changed'
     if(any(unwgt(1,1:icontr).ne.reference_weight))error stop 'group weights changed'
     do i=1,icontr
        do j=1,group_size(i)
           if(group_member(j,i).ne.reference(j,i))error stop 'group order changed'
        enddo
     enddo
     if(any(native_ids(:,1:icontr).ne.provenance))error stop 'grouping changed provenance'
  enddo
  call deallocate_weight_lines()
  write(*,*) 'PASS event colours'
end program

double precision function ran2()
  use controlled_colour_random
  ran2=next_random
end function

integer function get_color(id)
  integer :: id
  get_color=3
  if(id.eq.21)get_color=8
end function

double precision function get_mass_from_id(id)
  integer :: id
  get_mass_from_id=0d0
  if(abs(id).eq.6)get_mass_from_id=173d0
end function

subroutine update_shower_scale_Sevents(ifold)
  integer :: ifold
  error stop 'S event entered the H test'
end subroutine
