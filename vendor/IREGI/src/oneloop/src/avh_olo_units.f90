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


module avh_olo_units
  implicit none
! By default these values are set to 6. However, they can potentially clog
! the logs so we want to force them off, unless explicitely turned on
  integer :: eunit=0
  integer :: wunit=0
  integer :: munit=0
  integer :: punit=0 ! print all
  protected :: eunit,wunit,munit,punit !]PROTECTED
contains
  subroutine set_unit( message ,val )
!***********************************************************************
! message is intended to be one of the following:
! 'printall', 'message' ,'warning' ,'error'
!***********************************************************************
  character(*) ,intent(in) :: message
  integer      ,intent(in) :: val
  if (.false.) then
  elseif (message(1:8).eq.'printall') then ;punit=val
  elseif (message(1:7).eq.'message' ) then ;munit=val
  elseif (message(1:7).eq.'warning' ) then ;wunit=val
  elseif (message(1:5).eq.'error'   ) then ;eunit=val
  else
    eunit=val
    wunit=val
    munit=val
    punit=0
  endif
  end subroutine
end module
