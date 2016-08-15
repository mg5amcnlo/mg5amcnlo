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


module avh_olo_bub
  use avh_olo_units
  use avh_olo_prec
  use avh_olo_auxfun
  use avh_olo_bnlog
  use avh_olo_qmplx
  implicit none
  private
  public :: tadp ,tadpn ,bub0 ,bub1 ,bub11 ,bub111 ,bub1111

contains

  subroutine tadp( rslt ,mm ,amm ,rmu2 )
!*******************************************************************
! The 1-loop scalar 1-point function.
!*******************************************************************
  include 'avh_olo_complex.h90'
    ,intent(out) :: rslt(0:2)
  include 'avh_olo_complex.h90'
    ,intent(in)  :: mm
  include 'avh_olo_real.h90'
    ,intent(in)  :: amm,rmu2
![CALLINGME  write(*,*) 'MESSAGE from OneLOop tadp: you are calling me'
!
  rslt(2) = 0
  if (amm.eq.RZRO.or.mm.eq.CZRO) then
    rslt(1) = 0
    rslt(0) = 0
  else
    rslt(1) = mm
    rslt(0) = mm - mm*logc( qonv(mm/rmu2,-1) )
  endif
  end subroutine


  subroutine tadpn( rslt ,rank ,mm ,amm ,rmu2 )
!*******************************************************************
! The 1-loop tensor 1-point functions.
!   rslt(:,0) = A0
!   rslt(:,1) = A00
!   rslt(:,2) = A0000  etc.
! For input  rank  only  rslt(:,0:rank/2)  is filled.
!*******************************************************************
  include 'avh_olo_complex.h90'
    ,intent(out) :: rslt(0:,0:)
  include 'avh_olo_complex.h90'
    ,intent(in)  :: mm
  include 'avh_olo_real.h90'
    ,intent(in)  :: amm,rmu2
  integer ,intent(in) :: rank
  include 'avh_olo_complex.h90'
    :: aa
  include 'avh_olo_real.h90'
    :: bb
  integer :: ii
![CALLINGME  write(*,*) 'MESSAGE from OneLOop tadpn: you are calling me'
!
  do ii=0,rank
    rslt(2,ii) = 0
    rslt(1,ii) = 0
    rslt(0,ii) = 0
  enddo
  if (amm.eq.RZRO.or.mm.eq.CZRO) then
    return
  else
    rslt(1,0) = mm
    rslt(0,0) = mm - mm*logc( qonv(mm/rmu2,-1) )
    aa = 1
    bb = 0
    do ii=1,rank/2
      aa = aa*mm/(2*(ii+1))
      bb = bb + RONE/(ii+1)
      rslt(1,ii) = aa*( rslt(1,0) )
      rslt(0,ii) = aa*( rslt(0,0) + mm*bb )
    enddo
  endif
  end subroutine


!*******************************************************************
! Return the Passarino-Veltman functions
!
!      C   /      d^(Dim)q
!   ------ | -------------------- = b0
!   i*pi^2 / [q^2-m0][(q+p)^2-m1]
!
!      C   /    d^(Dim)q q^mu
!   ------ | -------------------- = p^mu b1
!   i*pi^2 / [q^2-m0][(q+p)^2-m1]
!
!      C   /  d^(Dim)q q^mu q^nu
!   ------ | -------------------- = g^{mu,nu} b00 + p^mu p^nu b11
!   i*pi^2 / [q^2-m0][(q+p)^2-m1]
!
!   etc.
!
! Based on the formulas from
! A. Denner, M. Dittmaier, Nucl.Phys. B734 (2006) 62-115
!*******************************************************************

  subroutine bub0( b0 &
                  ,pp,m0i,m1i ,app,am0i,am1i ,rmu2 )
  include 'avh_olo_complex.h90'
    ,intent(out) :: b0(0:2)
  include 'avh_olo_bub.h90' !?r1=no !?r2=no !?r3=no !?r4=no
  end subroutine

  subroutine bub1( b1,b0 &
                  ,pp,m0i,m1i ,app,am0i,am1i ,rmu2 )
  include 'avh_olo_complex.h90'
    ,intent(out) :: b1(0:2),b0(0:2)
  include 'avh_olo_bub.h90' !?r1=yes !?r2=no !?r3=no !?r4=no
  end subroutine

  subroutine bub11( b11,b00,b1,b0 &
                   ,pp,m0i,m1i ,app,am0i,am1i ,rmu2 )
  include 'avh_olo_complex.h90'
    ,intent(out) :: b11(0:2),b00(0:2),b1(0:2),b0(0:2)
  include 'avh_olo_bub.h90' !?r1=yes !?r2=yes !?r3=no !?r4=no
  end subroutine

  subroutine bub111( b111,b001,b11,b00,b1,b0 &
                    ,pp,m0i,m1i ,app,am0i,am1i ,rmu2 )
  include 'avh_olo_complex.h90'
    ,intent(out) :: b111(0:2),b001(0:2),b11(0:2),b00(0:2),b1(0:2),b0(0:2)
  include 'avh_olo_bub.h90' !?r1=yes !?r2=yes !?r3=yes !?r4=no
  end subroutine

  subroutine bub1111( b1111,b0011,b0000,b111,b001,b11,b00,b1,b0 &
                    ,pp,m0i,m1i ,app,am0i,am1i ,rmu2 )
  include 'avh_olo_complex.h90'
    ,intent(out) :: b1111(0:2),b0011(0:2),b0000(0:2) &
                   ,b111(0:2),b001(0:2),b11(0:2),b00(0:2),b1(0:2),b0(0:2)
  include 'avh_olo_bub.h90' !?r1=yes !?r2=yes !?r3=yes !?r4=yes
  end subroutine

end module
