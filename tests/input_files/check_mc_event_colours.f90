module controlled_colour_random
  double precision :: next_random=0.25d0
end module

program check_mc_event_colours
  use process_module
  use scale_module
  use weight_lines
  use controlled_colour_random
  implicit none
  include 'nexternal.inc'
  include 'genps.inc'
  include 'born_nhel.inc'
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
  if (any(icontr_sum(0,1:2).ne.1)) error stop 'merged distinct colours'
  event_colour_H(:,:,2,1)=saved
  call sum_identical_contributions()
  if (icontr_sum(0,1).ne.2.or.icontr_sum(0,2).ne.0) &
       error stop 'failed to merge identical owners'
  if (abs(unwgt(1,1)-0.5d0).gt.1d-14) error stop 'changed total weight'
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
