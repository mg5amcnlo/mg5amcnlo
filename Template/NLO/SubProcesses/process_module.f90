module process_module
  implicit none
  integer :: next_n,next_n1,nexternal_mod,nincoming_mod
  double precision,allocatable,dimension(:) :: mass_n,mass_n1
  integer,allocatable,dimension(:) :: colour_n,colour_n1
  logical,allocatable,dimension(:,:) :: valid_dipole_n,valid_dipole_n1
  double precision :: shat_n,shat_n1,collider_energy
  character(len=10) :: shower_mc
  character(len=4) :: abrv
  logical :: mcatnlo_delta

  public :: init_process_module_global,init_process_module_nbody, &
       init_process_module_n1body

contains

  subroutine init_process_module_global(shower_mc_in, abrv_in,nexternal_in,nincoming_in,mcatnlo_delta_in,collider_energy_in)
    implicit none
    integer :: nexternal_in,nincoming_in
    character(len=10) :: shower_mc_in
    character(len=4) :: abrv_in
    logical :: mcatnlo_delta_in
    double precision :: collider_energy_in
    ! global:
    nexternal_mod=nexternal_in
    nincoming_mod=nincoming_in
    shower_mc=shower_mc_in
    abrv=abrv_in
    mcatnlo_delta=mcatnlo_delta_in
    collider_energy=collider_energy_in
    ! n-body:
    if (.not.allocated(mass_n)) allocate(mass_n(1:nexternal_mod-1))
    if (.not.allocated(colour_n)) allocate(colour_n(1:nexternal_mod-1))
    if (.not.allocated(valid_dipole_n)) allocate(valid_dipole_n(1:nexternal_mod-1,1:nexternal_mod-1))
    ! n+1-body:
    if (.not.allocated(mass_n1)) allocate(mass_n1(1:nexternal_mod))
    if (.not.allocated(colour_n1)) allocate(colour_n1(1:nexternal_mod))
    if (.not.allocated(valid_dipole_n1)) allocate(valid_dipole_n1(1:nexternal_mod,1:nexternal_mod))
  end subroutine init_process_module_global
  
  subroutine init_process_module_nbody(nexternal_in, mass_in, colour_in, &
       valid_dipole_in, shat_in)
    implicit none
    integer :: nexternal_in
    double precision,dimension(nexternal_in) :: mass_in
    integer,dimension(nexternal_in) :: colour_in
    logical,dimension(nexternal_in,nexternal_in) :: valid_dipole_in
    double precision :: shat_in
    next_n=nexternal_in
    if (next_n.ne.nexternal_mod-1) then
       write (*,*) 'ERROR #1 in setting up process module',next_n,nexternal_mod
       stop 1
    endif
    mass_n=mass_in
    colour_n=colour_in
    valid_dipole_n=valid_dipole_in
    shat_n=shat_in
  end subroutine init_process_module_nbody

  subroutine init_process_module_n1body(nexternal_in, mass_in, colour_in, &
       valid_dipole_in, shat_in)
    implicit none
    integer :: nexternal_in
    double precision,dimension(nexternal_in) :: mass_in
    integer,dimension(nexternal_in) :: colour_in
    logical,dimension(nexternal_in,nexternal_in) :: valid_dipole_in
    double precision :: shat_in
    next_n1=nexternal_in
    if (next_n1.ne.nexternal_mod) then
       write (*,*) 'ERROR #2 in setting up process module',next_n1,nexternal_mod
       stop 1
    endif
    mass_n1=mass_in
    colour_n1=colour_in
    valid_dipole_n1=valid_dipole_in
    shat_n1=shat_in
  end subroutine init_process_module_n1body

end module process_module
