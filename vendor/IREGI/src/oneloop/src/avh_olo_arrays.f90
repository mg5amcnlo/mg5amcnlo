!!
!! Copyright (C) 2014 Andreas van Hameren. 
!!
!! This file is part of OneLOop-3.4.
!!
!! OneLOop-3.4 is free software: you can redistribute it and/or modify
!! it under the terms of the GNU General Public License as published by
!! the Free Software Foundation, either version 3 of the License, or
!! (at your option) any later version.
!!
!! OneLOop-3.4 is distributed in the hope that it will be useful,
!! but WITHOUT ANY WARRANTY; without even the implied warranty of
!! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!! GNU General Public License for more details.
!!
!! You should have received a copy of the GNU General Public License
!! along with OneLOop-3.4.  If not, see <http://www.gnu.org/licenses/>.
!!


module avh_olo_arrays
  use avh_olo_units
  use avh_olo_kinds !|RCPROG=intrinsic
!#  use ddmodule     !|RCPROG=ddfun90
!#  use qdmodule     !|RCPROG=qdcpp
!#  use mpmodule     !|RCPROG=mpfun90
!#  use mpmodule     !|RCPROG=arprec
  implicit none
  private
  public :: shift1,shift2,shift3,resize,enlarge

! Increase the size of the last dimension by one,
! and move  x(...,n:nsize)  to  x(...,n+1:nsize+1).
  interface shift1 ! for x(:)
    module procedure shift1_r,shift1_i
  end interface
  interface shift2 ! for x(:,:)
    module procedure shift2_r,shift2_i
  end interface
  interface shift3 ! for x(:,:,:)
    module procedure shift3_r,shift3_i
  end interface

! Resize x to the new bounds. Anything that doesn't fit anymore is lost.
  interface resize
    module procedure resize1_r,resize2_r
  end interface

! Resize x to the maximum of the bounds it has and then new bounds.
  interface enlarge
    module procedure enlarge1_r,enlarge2_r
  end interface

contains

  subroutine shift1_r( xx ,nn )
  include 'avh_olo_real.h90'
    ,allocatable ,intent(inout) :: xx(:)
  integer        ,intent(in   ) :: nn
  include 'avh_olo_real.h90'
    ,allocatable :: tt(:)
  integer ,parameter :: dm=1
  integer :: lb(dm),ub(dm)
  if (.not.allocated(xx)) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop shift1_r'
    stop
  endif
  lb=lbound(xx) ;ub=ubound(xx)
  allocate(tt(lb(dm):ub(dm)))
  tt = xx
  deallocate(xx)
  ub(dm) = ub(dm)+1
  allocate(xx(lb(dm):ub(dm)))
  xx(lb(dm):nn-1) = tt(lb(dm):nn-1)
  xx(nn+1:ub(dm)) = tt(nn:ub(dm)-1)
  deallocate(tt)
  end subroutine

  subroutine shift1_i( xx ,nn )
  integer ,allocatable ,intent(inout) :: xx(:)
  integer              ,intent(in   ) :: nn
  integer ,allocatable :: tt(:)
  integer ,parameter :: dm=1
  integer :: lb(dm),ub(dm)
  if (.not.allocated(xx)) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop shift1_i'
    stop
  endif
  lb=lbound(xx) ;ub=ubound(xx)
  allocate(tt(lb(dm):ub(dm)))
  tt = xx
  deallocate(xx)
  ub(dm) = ub(dm)+1
  allocate(xx(lb(dm):ub(dm)))
  xx(lb(dm):nn-1) = tt(lb(dm):nn-1)
  xx(nn+1:ub(dm)) = tt(nn:ub(dm)-1)
  deallocate(tt)
  end subroutine

  subroutine shift2_r( xx ,nn )
  include 'avh_olo_real.h90'
          ,allocatable ,intent(inout) :: xx(:,:)
  integer              ,intent(in   ) :: nn
  include 'avh_olo_real.h90'
          ,allocatable :: tt(:,:)
  integer ,parameter :: dm=2
  integer :: lb(dm),ub(dm)
  if (.not.allocated(xx)) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop shift2_r'
    stop
  endif
  lb=lbound(xx) ;ub=ubound(xx)
  allocate(tt(lb(1):ub(1),lb(dm):ub(dm)))
  tt = xx
  deallocate(xx)
  ub(dm) = ub(dm)+1
  allocate(xx(lb(1):ub(1),lb(dm):ub(dm)))
  xx(:,lb(dm):nn-1) = tt(:,lb(dm):nn-1)
  xx(:,nn+1:ub(dm)) = tt(:,nn:ub(dm)-1)
  deallocate(tt)
  end subroutine

  subroutine shift2_i( xx ,nn )
  integer ,allocatable ,intent(inout) :: xx(:,:)
  integer              ,intent(in   ) :: nn
  integer ,allocatable :: tt(:,:)
  integer ,parameter :: dm=2
  integer :: lb(dm),ub(dm)
  if (.not.allocated(xx)) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop shift2_i'
    stop
  endif
  lb=lbound(xx) ;ub=ubound(xx)
  allocate(tt(lb(1):ub(1),lb(dm):ub(dm)))
  tt = xx
  deallocate(xx)
  ub(dm) = ub(dm)+1
  allocate(xx(lb(1):ub(1),lb(dm):ub(dm)))
  xx(:,lb(dm):nn-1) = tt(:,lb(dm):nn-1)
  xx(:,nn+1:ub(dm)) = tt(:,nn:ub(dm)-1)
  deallocate(tt)
  end subroutine

  subroutine shift3_r( xx ,nn )
  include 'avh_olo_real.h90'
    ,allocatable ,intent(inout) :: xx(:,:,:)
  integer        ,intent(in   ) :: nn
  include 'avh_olo_real.h90'
    ,allocatable :: tt(:,:,:)
  integer ,parameter :: dm=3
  integer :: lb(dm),ub(dm)
  if (.not.allocated(xx)) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop shift3_r'
    stop
  endif
  lb=lbound(xx) ;ub=ubound(xx)
  allocate(tt(lb(1):ub(1),lb(2):ub(2),lb(dm):ub(dm)))
  tt = xx
  deallocate(xx)
  ub(dm) = ub(dm)+1
  allocate(xx(lb(1):ub(1),lb(2):ub(2),lb(dm):ub(dm)))
  xx(:,:,lb(dm):nn-1) = tt(:,:,lb(dm):nn-1)
  xx(:,:,nn+1:ub(dm)) = tt(:,:,nn:ub(dm)-1)
  deallocate(tt)
  end subroutine

  subroutine shift3_i( xx ,nn )
  integer ,allocatable ,intent(inout) :: xx(:,:,:)
  integer              ,intent(in   ) :: nn
  integer ,allocatable :: tt(:,:,:)
  integer ,parameter :: dm=3
  integer :: lb(dm),ub(dm)
  if (.not.allocated(xx)) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop shift3_i'
    stop
  endif
  lb=lbound(xx) ;ub=ubound(xx)
  allocate(tt(lb(1):ub(1),lb(2):ub(2),lb(dm):ub(dm)))
  tt = xx
  deallocate(xx)
  ub(dm) = ub(dm)+1
  allocate(xx(lb(1):ub(1),lb(2):ub(2),lb(dm):ub(dm)))
  xx(:,:,lb(dm):nn-1) = tt(:,:,lb(dm):nn-1)
  xx(:,:,nn+1:ub(dm)) = tt(:,:,nn:ub(dm)-1)
  deallocate(tt)
  end subroutine

 
  subroutine resize1_r( xx ,l1,u1 )
  include 'avh_olo_real.h90'
    ,allocatable ,intent(inout) :: xx(:)
  integer        ,intent(in   ) :: l1,u1
  include 'avh_olo_real.h90'
    ,allocatable :: tt(:)
  integer :: lb(1),ub(1)
  if (.not.allocated(xx)) then
    allocate(xx(l1:u1))
    return
  endif
  lb=lbound(xx) ;ub=ubound(xx)
  allocate(tt(lb(1):ub(1)))
  tt = xx
  deallocate(xx)
  allocate( xx(l1:u1) )
  lb(1)=max(l1,lb(1)) ;ub(1)=min(u1,ub(1))
  xx(lb(1):ub(1)) = tt(lb(1):ub(1))
  deallocate(tt)
  end subroutine 

  subroutine resize2_r( xx ,l1,u1 ,l2,u2 )
  include 'avh_olo_real.h90'
    ,allocatable ,intent(inout) :: xx(:,:)
  integer        ,intent(in   ) :: l1,u1,l2,u2
  include 'avh_olo_real.h90'
    ,allocatable :: tt(:,:)
  integer :: lb(2),ub(2)
  if (.not.allocated(xx)) then
    allocate(xx(l1:u1,l2:u2))
    return
  endif
  lb=lbound(xx) ;ub=ubound(xx)
  allocate(tt(lb(1):ub(1),lb(2):ub(2)))
  tt = xx
  deallocate(xx)
  allocate( xx(l1:u1,l2:u2) )
  lb(1)=max(l1,lb(1)) ;ub(1)=min(u1,ub(1))
  lb(2)=max(l2,lb(2)) ;ub(2)=min(u2,ub(2))
  xx(lb(1):ub(1),lb(2):ub(2)) = &
  tt(lb(1):ub(1),lb(2):ub(2))
  deallocate(tt)
  end subroutine 


  subroutine enlarge1_r( xx ,l1,u1 )
  include 'avh_olo_real.h90'
    ,allocatable ,intent(inout) :: xx(:)
  integer        ,intent(in   ) :: l1,u1
  include 'avh_olo_real.h90'
    ,allocatable :: tt(:)
  integer :: lb(1),ub(1)
  if (.not.allocated(xx)) then
    allocate(xx(l1:u1))
    return
  endif
  lb=lbound(xx) ;ub=ubound(xx)
  if (lb(1).le.l1.and.u1.le.ub(1)) return
  if (lb(1).gt.ub(1)) then
    deallocate( xx )
    allocate( xx(min(l1,lb(1)):max(u1,ub(1))) )
    return
  endif
  allocate(tt(lb(1):ub(1)))
  tt = xx
  deallocate(xx)
  allocate( xx(min(l1,lb(1)):max(u1,ub(1))) )
  xx(lb(1):ub(1)) = tt(lb(1):ub(1))
  deallocate(tt)
  end subroutine 

  subroutine enlarge2_r( xx ,l1,u1 ,l2,u2 )
  include 'avh_olo_real.h90'
    ,allocatable ,intent(inout) :: xx(:,:)
  integer        ,intent(in   ) :: l1,u1,l2,u2
  include 'avh_olo_real.h90'
    ,allocatable :: tt(:,:)
  integer :: lb(2),ub(2)
  if (.not.allocated(xx)) then
    allocate(xx(l1:u1,l2:u2))
    return
  endif
  lb=lbound(xx) ;ub=ubound(xx)
  if (lb(1).le.l1.and.u1.le.ub(1).and. &
      lb(2).le.l2.and.u2.le.ub(2)      ) return
  if (lb(1).gt.ub(1).or.lb(2).gt.ub(2)) then
    deallocate( xx )
    allocate( xx(min(l1,lb(1)):max(u1,ub(1))  &
                ,min(l2,lb(2)):max(u2,ub(2))) )
    return
  endif
  allocate(tt(lb(1):ub(1),lb(2):ub(2)))
  tt = xx
  deallocate(xx)
  allocate( xx(min(l1,lb(1)):max(u1,ub(1))  &
              ,min(l2,lb(2)):max(u2,ub(2))) )
  xx(lb(1):ub(1),lb(2):ub(2)) = &
  tt(lb(1):ub(1),lb(2):ub(2))
  deallocate(tt)
  end subroutine 

end module
