module flow_fixture
  use weight_lines
  implicit none
  double precision :: values(3)=[1d0,3d0,0d0],uniform=0.5d0,last_grid_weight=-1d0
  integer :: draws=0,born_calls=0,fail_native_fold=0
  logical :: real_point_active=.false.,born_cuts=.true.
contains
  subroutine add_record(value)
    double precision :: value
    icontr=icontr+1
    wgt(:,icontr)=value
  end subroutine
end module

module fks_phase_space_data
  implicit none
  double precision :: p_born(0:3,4)=1d0,p1_cnt(0:3,5,0:2)=1d0,jac_cnt(0:2)=1d0
end module

module fks_phase_space
  use fks_phase_space_data
  implicit none
  type born_fixture
    logical :: valid=.false.,event_projection=.true.
    double precision :: p(0:3,4)=1d0,xbjrk(2)=0.2d0
  end type
  type fks_phase_space_point
    double precision :: p(0:3,5)=1d0,p_lab(0:3,5)=1d0,p_cms(0:3,5)=1d0
    type(born_fixture) :: born
  end type
contains
  subroutine generate_born_contribution(ndim,config,jac,x,point)
    integer :: ndim,config
    double precision :: jac,x(99)
    type(fks_phase_space_point) :: point
    point=fks_phase_space_point()
    p_born=1d0
  end subroutine
  subroutine generate_real_phase_space(ndim,config,jac,x,point)
    integer :: ndim,config
    double precision :: jac,x(99)
    type(fks_phase_space_point) :: point
    point=fks_phase_space_point()
  end subroutine
end module

module mc_native_context
  use flow_fixture, only: real_point_active
  implicit none
contains
  subroutine mc_begin_real_point(p)
    double precision :: p(0:3,5)
    if(real_point_active)error stop 'real-point cache was not closed'
    real_point_active=.true.
  end subroutine
  subroutine mc_end_real_point()
    if(.not.real_point_active)error stop 'real-point cache closed twice'
    real_point_active=.false.
  end subroutine
end module

module mint_module
  use FKSParams
  implicit none
  integer,parameter :: ndimmax=5,nintegrals=6,max_fold=8
  integer :: iconfig=1,ndim=5,ifold(5)=[1,1,3,1,1]
  integer :: ifold_energy=3,ifold_yij=4,ifold_phi=5
  integer :: born_spread_current_bin=1,born_spread_bin_fold(8),born_spread_sector_fold(8)
  logical :: new_point=.true.,pass_cuts_check=.false.
  double precision :: virt_wgt_mint(0:1)=0d0,born_wgt_mint(0:1)=0d0
contains
  subroutine born_spread_set_point(x,y)
    double precision :: x,y
  end subroutine
end module

module mc_counterterms
  implicit none
  integer :: fksfather=3
  double precision :: gfactsf=1d0,gfactcl=1d0
contains
  double precision function mc_shower_scale_mass()
    mc_shower_scale_mass=0d0
  end function
end module

module process_module
  integer,parameter :: ndelH=5
end module

module scale_module
  implicit none
  integer :: born_flow_picked=0
  double precision :: emsca_H(1,8,5,5)=1d0
  double precision :: shower_scale_n1body(5,5)=1d0,shower_scale_nbody(4,4)=1d0
end module
