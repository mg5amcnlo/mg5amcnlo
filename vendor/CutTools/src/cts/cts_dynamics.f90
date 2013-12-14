!
 module inout
  include 'cts_mprec.h'
  implicit none 
  private
  logical, public :: mprec
  include 'cts_mpc.h'
   , public, dimension(0:3) :: mpq
  include 'cts_dpc.h'
   , public, dimension(0:3) :: dpq
  include 'cts_mpc.h'
   , public :: mpres
  include 'cts_dpc.h'
   , public :: dpres
 endmodule inout
!
 subroutine numfunc(numdummy)
!
! Numerator function
!
  use inout
  include 'cts_mprec.h'
  implicit none 
  external numdummy
  if (.not.mprec) then
    call numdummy(dpq,dpres)
  else
    call numdummy(mpq,mpres)
  endif
 end subroutine numfunc
