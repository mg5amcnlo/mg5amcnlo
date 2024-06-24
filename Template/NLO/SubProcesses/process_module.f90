module process_module
  implicit none
  integer :: next_n,next_n1,nexternal_mod,nincoming_mod,max_flows_n,max_flows_n1
  double precision,allocatable,dimension(:) :: mass_n,mass_n1
  integer,allocatable,dimension(:) :: colour_n,colour_n1,iRtoB
  logical,allocatable,dimension(:,:,:) :: valid_dipole_n,valid_dipole_n1
  double precision :: shat_n1,collider_energy
  character*10 :: shower_mc_mod
  character*4 :: abrv_mod
  logical :: mcatnlo_delta_mod

  public :: init_process_module_global,init_process_module_nbody, &
       init_process_module_n1body

contains

  subroutine init_process_module_global(shower_mc_in, abrv_in,nexternal_in,nincoming_in, &
       mcatnlo_delta_in,collider_energy_in,max_flows_n_in,max_flows_n1_in)
    implicit none
    integer :: nexternal_in,nincoming_in,max_flows_n_in,max_flows_n1_in
    character*10 :: shower_mc_in
    character*4 :: abrv_in
    logical :: mcatnlo_delta_in
    double precision :: collider_energy_in
    ! global:
    nexternal_mod=nexternal_in
    next_n1=nexternal_in
    nincoming_mod=nincoming_in
    shower_mc_mod=shower_mc_in
    abrv_mod=abrv_in
    mcatnlo_delta_mod=mcatnlo_delta_in
    collider_energy=collider_energy_in
    max_flows_n=max_flows_n_in
    max_flows_n1=max_flows_n1_in
    ! n-body:
    if (.not.allocated(mass_n)) allocate(mass_n(1:nexternal_mod-1))
    if (.not.allocated(colour_n)) allocate(colour_n(1:nexternal_mod-1))
    if (.not.allocated(valid_dipole_n)) allocate(valid_dipole_n(1:nexternal_mod-1,1:nexternal_mod-1,1:max_flows_n))
    ! n+1-body:
    if (.not.allocated(mass_n1)) allocate(mass_n1(1:nexternal_mod))
    if (.not.allocated(colour_n1)) allocate(colour_n1(1:nexternal_mod))
    if (.not.allocated(valid_dipole_n1)) allocate(valid_dipole_n1(1:nexternal_mod,1:nexternal_mod,1:max_flows_n1))
    if (.not.allocated(iRtoB)) allocate(iRtoB(1:nexternal_mod))
  end subroutine init_process_module_global
  
  subroutine init_process_module_nbody(nexternal_in, mass_in, colour_in, &
       max_flows_in,valid_dipole_in)
    implicit none
    integer :: nexternal_in,max_flows_in
    double precision,dimension(nexternal_in) :: mass_in
    integer,dimension(nexternal_in) :: colour_in
    logical,dimension(nexternal_in,nexternal_in,max_flows_in) :: valid_dipole_in
    next_n=nexternal_in
    if (next_n.ne.nexternal_mod-1) then
       write (*,*) 'ERROR #1 in setting up process module',next_n,nexternal_mod
       stop 1
    endif
    if (max_flows_in.ne.max_flows_n) then
       write (*,*) 'ERROR #3 in setting up process module',max_flows_in,max_flows_n
       stop 1
    endif
    mass_n=mass_in
    colour_n=colour_in
    valid_dipole_n=valid_dipole_in
  end subroutine init_process_module_nbody

  subroutine init_process_module_n1body(nexternal_in, mass_in, colour_in, &
       max_flows_in,valid_dipole_in, shat_in)
    implicit none
    integer :: nexternal_in,max_flows_in
    double precision,dimension(nexternal_in) :: mass_in
    integer,dimension(nexternal_in) :: colour_in
    logical,dimension(nexternal_in,nexternal_in,max_flows_in) :: valid_dipole_in
    double precision :: shat_in
    if (next_n1.ne.nexternal_mod) then
       write (*,*) 'ERROR #2 in setting up process module',next_n1,nexternal_mod
       stop 1
    endif
    if (max_flows_in.ne.max_flows_n1) then
       write (*,*) 'ERROR #4 in setting up process module',max_flows_in,max_flows_n1
       stop 1
    endif
    mass_n1=mass_in
    colour_n1=colour_in
    valid_dipole_n1=valid_dipole_in
    shat_n1=shat_in
    
  end subroutine init_process_module_n1body

  subroutine RealToBornMapping(i_fks)
    implicit none
    integer :: i,i_fks
    do i=1,next_n1
       if(i.lt.i_fks)then
          iRtoB(i)=i
       elseif(i.eq.i_fks)then
          iRtoB(i)=-1
       elseif(i.gt.i_fks)then
          iRtoB(i)=i-1
       endif
    enddo
  end subroutine RealToBornMapping
  
end module process_module
