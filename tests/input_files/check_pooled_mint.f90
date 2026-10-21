module pooled_mint_fixture
  implicit none
  integer, parameter :: ndimmax=4,n_ord_virt=2,norders=n_ord_virt
  integer :: ndim=4,ichan=1,icell(4)=1
  double precision :: xacc(1,4,1)=0d0,f(10)=0d0,virtual_fraction(1)=.25d0
  double precision :: average_virtual(0:2,1),polyfit(0:2)=0d0,wgt_mult=1d0
  double precision :: recorded_virtual(0:2),recorded_born(0:2)
  integer :: recorded_calls(0:2)=0
  logical :: use_poly_virtual=.false.
  logical :: use_pooled_virtual=.false.,pooled_virtual_sampled=.false.
  logical :: vs_vcomp=.false.,vs_ratio_set=.false.
  double precision :: vs_ratio=0d0
  double precision :: pooled_virtual_ratio(2)=0d0,pooled_virtual_probability=.25d0
  double precision :: pooled_virtual_actual_ratio(2)=0d0
contains
  include 'pooled_mint_routine.inc'
  subroutine fail(message)
    character(len=*), intent(in) :: message
    write(*,*) message
    stop 1
  end subroutine
  subroutine fill_ave_virt(x,order,virtual,born)
    double precision, intent(in) :: x(:),virtual,born
    integer, intent(in) :: order
    recorded_virtual(order)=virtual
    recorded_born(order)=born
    recorded_calls(order)=recorded_calls(order)+1
  end subroutine
end module

program check_pooled_mint
  use pooled_mint_fixture
  implicit none
  double precision :: x(4),saved(10)
  x=.5d0
  ! A pooled model has its own frozen ratio; grid fitting must reconstruct
  ! the exact virtual even though the traditional MINT ratio differs.
  use_pooled_virtual=.true.
  pooled_virtual_sampled=.true.
  ! Conditional predictions distinguish the q-weighted mean from the head
  ! actually subtracted by the sampled native loop.
  pooled_virtual_ratio=[99d0,-88d0]
  pooled_virtual_actual_ratio=[2d0,3d0]
  average_virtual(:,1)=[4d0,99d0,88d0]
  f([8,10])=[6d0,-10d0]
  f([7,9])=[32d0,240d0]
  f(3)=sum(f([7,9]));f(6)=sum(f([8,10]))
  saved=f
  call add_point_to_grids(x)
  if(any(abs(recorded_virtual-[50d0,20d0,30d0])>1d-12)) error stop 'pooled grid target changed'
  if(any(f/=saved)) error stop 'pooled reconstruction changed physical residual'
  f([3,7,9])=0d0
  call add_point_to_grids(x)
  if(any(recorded_virtual/=[-18d0,12d0,-30d0])) error stop 'pooled zero residual grid target'
  if(any(f([6,8,10])/=[-4d0,6d0,-10d0])) error stop 'pooled zero residual lost Born'
  ! A guarded tail is sampled with probability one even when the channel
  ! budget remains one quarter. Grid targets must use the actual probability.
  pooled_virtual_probability=1d0
  f([7,9])=[8d0,60d0]
  f(3)=sum(f([7,9]))
  call add_point_to_grids(x)
  if(any(abs(recorded_virtual-[50d0,20d0,30d0])>1d-12)) error stop 'pooled tail grid target changed'
  if(virtual_fraction(1)/=.25d0) error stop 'tail changed the channel budget'
  pooled_virtual_sampled=.false.
  f([3,7,9])=0d0
  call add_point_to_grids(x)
  if(any(f([6,8,10])/=0d0)) error stop 'unsampled Born stream retained'
  ! A per-channel fit must undo its own ratio when training the fallback.
  use_pooled_virtual=.false.;vs_ratio_set=.true.;vs_vcomp=.true.;vs_ratio=2d0
  f([8,10])=[6d0,-10d0];f([7,9])=[32d0,200d0]
  f(3)=sum(f([7,9]));f(6)=sum(f([8,10]))
  call add_point_to_grids(x)
  if(any(abs(recorded_virtual-[50d0,20d0,30d0])>1d-12))error stop 'fitted surrogate grid target'
  ! A perfect fit in one order still supplies its weighted Born target.
  f(7)=0d0;f(3)=f(9)
  call add_point_to_grids(x)
  if(any(recorded_virtual/=[42d0,12d0,30d0]))error stop 'fitted zero residual target'
  if(any(f([6,8,10])/=[-4d0,6d0,-10d0]))error stop 'fitted zero residual lost Born'
  print *, 'PASS pooled MINT original targets and zero residuals'
end program

subroutine add_point_polyfit(channel,order,x,ratio,weight)
  integer, intent(in) :: channel,order
  double precision, intent(in) :: x(*),ratio,weight
  error stop 'unexpected polynomial fit'
end subroutine
