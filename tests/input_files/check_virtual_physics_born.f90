! Exercise the actual FKS helper with the worker's static-local compilation.
module virtual_pooled
  implicit none
  integer :: predictions=0
contains
  logical function pooled_virtual_needs_helicities()
    pooled_virtual_needs_helicities=.true.
  end function
  logical function pooled_virtual_uses_physics()
    pooled_virtual_uses_physics=.true.
  end function
  subroutine pooled_virtual_set_helicity_weights(weights,dipoles)
    real(8),intent(in) :: weights(:,:)
    real(8),intent(in),optional :: dipoles(:,:)
    if(.not.present(dipoles)) error stop 'missing colour inputs'
    if(any(weights/=2d0)) error stop 'Born helicity weights changed'
    if(abs(dipoles(1,2)+4d0/3d0)>1d-12) error stop 'colour Born used as normalization'
    predictions=predictions+1
  end subroutine
end module

subroutine mc_born_local(p,request,result)
  use mc_born_types
  implicit none
  real(8),intent(in) :: p(0:3,4)
  type(BornRequest),intent(in) :: request
  type(BornResult),intent(out) :: result
  include 'coupl.inc'
  allocate(result%split_counterterms(1,1,1))
  result%split_counterterms=2d0
  if(request%helicities) then
    if(request%colour.or.request%charge) error stop 'stale correlation request'
    allocate(result%helicity_orders(1,1))
    result%helicity_orders=2d0
  endif
  if(request%colour) then
    allocate(result%soft(1))
    result%soft=-8d0/3d0*g**2
    result%split_counterterms=-8d0/3d0
  endif
end subroutine

subroutine update_as_param()
end subroutine

program check_virtual_physics_born
  use virtual_pooled
  implicit none
  include 'coupl.inc'
  include 'q_es.inc'
  include '../../Source/MODEL/input.inc'
  real(8) :: p(0:3,4)
  logical :: updateloop
  common/to_updateloop/updateloop
  p=0d0;g=2d0;qes2=100d0;mu_r=20d0;updateloop=.false.
  call prepare_pooled_helicities(p,[8,8,3,-3])
  call prepare_pooled_helicities(p,[8,8,3,-3])
  if(predictions/=2) error stop 'missing second prediction'
  if(mu_r/=20d0.or.updateloop) error stop 'loop parameter state not restored'
end program
