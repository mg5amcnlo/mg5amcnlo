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


module avh_olo

!{dp=yes
  use avh_olo_dp ,only: &
     olo_dp_kind=>olo_kind &
    ,olo_dp_scale=>olo_get_scale &
    ,olo_dp_onshell=>olo_get_onshell &
    ,olo_dp_precision=>olo_get_precision &
    ,olo,olo_a0,olo_an,olo_b0,olo_b11,olo_bn,olo_c0,olo_d0
!}dp=yes
!{qp=yes
!#  use avh_olo_qp ,only: &
!#     olo_qp_kind=>olo_kind &
!#    ,olo_qp_scale=>olo_get_scale &
!#    ,olo_qp_onshell=>olo_get_onshell &
!#    ,olo_qp_precision=>olo_get_precision &
!#    ,olo,olo_a0,olo_an,olo_b0,olo_b11,olo_bn,olo_c0,olo_d0
!}qp=yes
!{dd=yes
!#  use avh_olo_dd ,only: &
!#     olo_dd_kind=>olo_kind &
!#    ,olo_dd_scale=>olo_get_scale &
!#    ,olo_dd_onshell=>olo_get_onshell &
!#    ,olo_dd_precision=>olo_get_precision &
!#    ,olo,olo_a0,olo_an,olo_b0,olo_b11,olo_bn,olo_c0,olo_d0
!}dd=yes
!{qd=yes
!#  use avh_olo_qd ,only: &
!#     olo_qd_kind=>olo_kind &
!#    ,olo_qd_scale=>olo_get_scale &
!#    ,olo_qd_onshell=>olo_get_onshell &
!#    ,olo_qd_precision=>olo_get_precision &
!#    ,olo,olo_a0,olo_an,olo_b0,olo_b11,olo_bn,olo_c0,olo_d0
!}qd=yes
!{mp=yes
!#  use avh_olo_mp ,only: &
!#     olo_mp_kind=>olo_kind &
!#    ,olo_mp_scale=>olo_get_scale &
!#    ,olo_mp_onshell=>olo_get_onshell &
!#    ,olo_mp_precision=>olo_get_precision &
!#    ,olo,olo_a0,olo_an,olo_b0,olo_b11,olo_bn,olo_c0,olo_d0
!}mp=yes

  implicit none

contains

  subroutine olo_unit( val ,message )
  use avh_olo_version
  use avh_olo_units ,only: set_unit
  integer     ,intent(in) :: val
  character(*),intent(in),optional :: message
  call olo_version
  if (present(message)) then ;call set_unit( message ,val )
  else                       ;call set_unit( 'all'   ,val )
  endif
  end subroutine

  subroutine olo_precision( ndec )
  use avh_olo_dp ,only: dp_sub=>olo_precision !|dp=yes
!#  use avh_olo_qp ,only: qp_sub=>olo_precision !|qp=yes
!#  use avh_olo_dd ,only: dd_sub=>olo_precision !|dd=yes
!#  use avh_olo_qd ,only: qd_sub=>olo_precision !|qd=yes
!#  use avh_olo_mp ,only: mp_sub=>olo_precision !|mp=yes
  integer ,intent(in) :: ndec
  call dp_sub( ndec ) !|dp=yes
!#  call qp_sub( ndec ) !|qp=yes
!#  call dd_sub( ndec ) !|dd=yes
!#  call qd_sub( ndec ) !|qd=yes
!#  call mp_sub( ndec ) !|mp=yes
  end subroutine

  subroutine olo_scale( val )
  use avh_olo_dp ,only: dp_sub=>olo_scale !|dp=yes
!#  use avh_olo_qp ,only: qp_sub=>olo_scale !|qp=yes
!#  use avh_olo_dd ,only: dd_sub=>olo_scale !|dd=yes
!#  use avh_olo_qd ,only: qd_sub=>olo_scale !|qd=yes
!#  use avh_olo_mp ,only: mp_sub=>olo_scale !|mp=yes
  real(kind(1d0)) ,intent(in) :: val
  call dp_sub( val ) !|dp=yes
!#  call qp_sub( val ) !|qp=yes
!#  call dd_sub( val ) !|dd=yes
!#  call qd_sub( val ) !|qd=yes
!#  call mp_sub( val ) !|mp=yes
  end subroutine

  subroutine olo_onshell( val )
  use avh_olo_dp ,only: dp_sub=>olo_onshell !|dp=yes
!#  use avh_olo_qp ,only: qp_sub=>olo_onshell !|qp=yes
!#  use avh_olo_dd ,only: dd_sub=>olo_onshell !|dd=yes
!#  use avh_olo_qd ,only: qd_sub=>olo_onshell !|qd=yes
!#  use avh_olo_mp ,only: mp_sub=>olo_onshell !|mp=yes
  real(kind(1d0)) ,intent(in) :: val
  call dp_sub( val ) !|dp=yes
!#  call qp_sub( val ) !|qp=yes
!#  call dd_sub( val ) !|dd=yes
!#  call qd_sub( val ) !|qd=yes
!#  call mp_sub( val ) !|mp=yes
  end subroutine

  subroutine olo_setting( iunit )
  use avh_olo_units
  use avh_olo_version
  integer,optional,intent(in) :: iunit
  integer :: nunit
  call olo_version
  nunit = munit
  if (present(iunit)) nunit = iunit
  if (nunit.le.0) return
  write(nunit,*) 'ERROR in OneLOop: subroutine olo_setting  is not available,'
  write(nunit,*) 'ERROR in OneLOop: use  function olo_get_scale  etc. instead.'
  end subroutine

end module
