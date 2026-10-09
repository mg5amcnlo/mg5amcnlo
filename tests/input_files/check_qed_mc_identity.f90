module mc_native_context
contains
  subroutine activate_native_context(ifks)
    implicit none
    integer :: ifks
  end subroutine
end module

program check_qed_mc_identity
  implicit none
  include 'orders.inc'
  include 'nexternal.inc'
  include 'fks_info.inc'
  integer :: nFKSprocess,particle_type_born(4),sector,leg,i
  logical :: split_type(nsplitorders)
  double precision :: particle_charge_born(4)
  common /c_nFKSprocess/ nFKSprocess
  common /c_particle_type_born/ particle_type_born
  common /c_charges_born/ particle_charge_born
  common /c_split_type/ split_type
  fks_i_D=5
  extra_cnt_D=0
  isplitorder_born_D=0
  isplitorder_cnt_D=0
  need_color_links_D=.false.
  need_charge_links_D=.false.
  fks_j_from_i_D=0
  particle_tag_D=.false.
  split_type_D=.false.
  split_type_D(1,qcd_pos)=.true.
  split_type_D(2,qed_pos)=.true.
  do leg=1,3,2
    fks_j_D=leg
    particle_type_D=1
    pdg_type_D=22
    particle_charge_D=0d0
    particle_type_D(:,5)=-3
    particle_type_D(:,leg)=3
    particle_charge_D(:,5)=1d0/3d0
    particle_charge_D(:,leg)=-1d0/3d0
    if (leg.eq.1) then
      particle_type_D(:,5)=3
      particle_charge_D(:,5)=-1d0/3d0
    endif
    split_type=split_type_D(1,:)
    do i=1,3
      nFKSprocess=2-mod(i+1,2)
      call fks_inc_chooser()
      write(*,*) particle_type_born(leg),particle_charge_born(leg)
    enddo
  enddo
end program
