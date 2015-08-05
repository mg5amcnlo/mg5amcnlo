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


module avh_olo_print
  use avh_olo_prec
  implicit none
  private
  public :: myprint

  integer ,parameter :: novh=10 !maximally 6 decimals for exponent
  integer ,parameter :: nxtr=4  !extra decimals

  interface myprint
    module procedure printr,printc,printi
  end interface

contains

  function printc( zz ,ndec ) result(rslt)
  include 'avh_olo_complex.h90'
    ,intent(in) :: zz
  integer,optional,intent(in) :: ndec
  character((ndecim(prcpar)+nxtr+novh)*2+3) :: rslt
  if (present(ndec)) then
    rslt = '('//trim(printr(areal(zz),ndec)) &
         //','//trim(printr(aimag(zz),ndec)) &
         //')'
  else
    rslt = '('//trim(printr(areal(zz))) &
         //','//trim(printr(aimag(zz))) &
         //')'
  endif
  rslt = adjustl(rslt)
  end function

  function printr( xx_in ,ndec_in ) result(rslt)
  include 'avh_olo_real.h90'
                  ,intent(in) :: xx_in
  integer,optional,intent(in) :: ndec_in
  character(ndecim(prcpar)+nxtr+novh  ) :: rslt
  character(ndecim(prcpar)+nxtr+novh+1) :: cc
  character(10) :: aa,bb
  integer :: ndec
  double precision :: xx     !|RCTYPE=intrinsic
!#  real(kind(1d0)) :: xx !|RCTYPE=ddtype
!#  real(kind(1d0)) :: xx !|RCTYPE=qdtype
!#  real(kind(1d0)) :: xx !|RCTYPE=mptype
  xx = xx_in
  if (present(ndec_in)) then ;ndec=ndec_in
                        else ;ndec=ndecim(prcpar)+nxtr
  endif
  write(aa,'(i10)') min(len(cc),ndec+novh+1) ;aa=adjustl(aa)
  write(bb,'(i10)') min(len(cc),ndec       ) ;bb=adjustl(bb)
  aa = '(e'//trim(aa)//'.'//trim(bb)//')'
  write(cc,aa) xx  ;cc=adjustl(cc)
  if (cc(1:2).eq.'-0') then ;rslt = '-'//cc(3:len(cc))
  else                      ;rslt = ' '//cc(2:len(cc))
  endif
  end function

  function printi( ii ) result(rslt)
  integer ,intent(in) :: ii
  character(ndecim(prcpar)) :: rslt
  character(ndecim(prcpar)) :: cc
  character(10) :: aa
  write(aa,'(i10)') ndecim(prcpar) ;aa=adjustl(aa)
  aa = '(i'//trim(aa)//')'
  write(cc,aa) ii ;cc=adjustl(cc)
  if (cc(1:1).ne.'-') then ;rslt=' '//cc
  else                     ;rslt=cc 
  endif
  end function

end module
