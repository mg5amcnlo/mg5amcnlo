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


module avh_olo_qmplx
  use avh_olo_units
  use avh_olo_prec
  use avh_olo_auxfun
  use avh_olo_olog
  use avh_olo_dilog

  implicit none
  private
  public :: qmplx_type,qonv,directly,sheet,logc,logc2,li2c,li2c2
  public :: operator (*) ,operator (/)

  type :: qmplx_type
  include 'avh_olo_complex.h90'
          :: c
  integer :: p
  end type

  interface qonv
    module procedure qonv_cr,qonv_ci,qonv_c,qonv_i
  end interface

  interface operator (*)
    module procedure prduct_qq,prduct_qr
  end interface
  interface operator (/)
    module procedure ratio_qq,ratio_qr
  end interface

contains


  function qonv_cr(xx,sgn) result(rslt)
!*******************************************************************
! zz=rslt%c ,iz=rslt%p
! Determine  zz,iz  such that  xx = zz*exp(iz*imag*pi)  and  Re(zz)
! is positive. If  Im(x)=0  and  Re(x)<0  then  iz  becomes the
! sign of  sgn .
!*******************************************************************
  include 'avh_olo_complex.h90'
    ,intent(in) :: xx
  include 'avh_olo_real.h90'
    ,intent(in) :: sgn
  type(qmplx_type) :: rslt
  include 'avh_olo_real.h90'
    :: xre,xim
  xre = areal(xx)
  if (xre.ge.RZRO) then
    rslt%c = xx
    rslt%p = 0
  else
    xim = aimag(xx)
    if (xim.eq.RZRO) then
      rslt%c = -xre
      rslt%p = sgnRe(sgn)
    else
      rslt%c = -xx
      rslt%p = sgnRe(xim) ! xim = -Im(rslt%c)
    endif
  endif
  end function

  function qonv_ci(xx,sgn) result(rslt)
!*******************************************************************
! zz=rslt%c ,iz=rslt%p
! Determine  zz,iz  such that  xx = zz*exp(iz*imag*pi)  and  Re(zz)
! is positive. If  Im(x)=0  and  Re(x)<0  then  iz  becomes the
! sign of  sgn .
!*******************************************************************
  include 'avh_olo_complex.h90'
    ,intent(in) :: xx
  integer         ,intent(in) :: sgn
  type(qmplx_type) :: rslt
  include 'avh_olo_real.h90'
    :: xre,xim
  xre = areal(xx)
  if (xre.ge.RZRO) then
    rslt%c = xx
    rslt%p = 0
  else
    xim = aimag(xx)
    if (xim.eq.RZRO) then
      rslt%c = -xre
      rslt%p = sign(1,sgn)
    else
      rslt%c = -xx
      rslt%p = sgnRe(xim) ! xim = -Im(rslt%c)
    endif
  endif
  end function

  function qonv_c(xx) result(rslt)
!*******************************************************************
! zz=rslt%c ,iz=rslt%p
! Determine  zz,iz  such that  xx = zz*exp(iz*imag*pi)  and  Re(zz)
! is positive. If  Im(x)=0  and  Re(x)<0  then  iz=1
!*******************************************************************
  include 'avh_olo_complex.h90'
    ,intent(in) :: xx
  type(qmplx_type) :: rslt
  include 'avh_olo_real.h90'
    :: xre,xim
  xre = areal(xx)
  if (xre.ge.RZRO) then
    rslt%c = xx
    rslt%p = 0
  else
    xim = aimag(xx)
    if (xim.eq.RZRO) then
      if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop qonv_c: ' &
        ,'negative input with undefined sign for the imaginary part, ' &
        ,'putting +ieps'
      rslt%c = -xre
      rslt%p = 1
    else
      rslt%c = -xx
      rslt%p = sgnRe(xim) ! xim = -Im(rslt%c)
    endif
  endif
  end function

  function qonv_i(xx) result(rslt)
!*******************************************************************
! zz=rslt%c ,iz=rslt%p
! Determine  zz,iz  such that  xx = zz*exp(iz*imag*pi)  and  Re(zz)
! is positive. If  Im(x)=0  and  Re(x)<0  then  iz=1
!*******************************************************************
  integer ,intent(in) :: xx
  type(qmplx_type) :: rslt
  if (xx.ge.0) then
    rslt%c = xx
    rslt%p = 0
  else
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop qonv_i: ' &
      ,'negative input with undefined sign for the imaginary part, ' &
      ,'putting +ieps'
    rslt%c = -xx
    rslt%p = 1
  endif
  end function

  function directly(xx,ix) result(rslt)
!*******************************************************************
!*******************************************************************
  include 'avh_olo_complex.h90'
    ,intent(in) :: xx
  integer         ,intent(in) :: ix
  type(qmplx_type) :: rslt
  rslt%c = xx
  rslt%p = ix
  end function


  function sheet(xx) result(ii)
!*******************************************************************
! Returns the number of the Riemann-sheet (times 2) for the complex
! number  xx*exp(ix*imag*pi) . The real part of xx is assumed to be
! positive or zero. Examples:
! xx=1+imag, ix=-1 -> ii= 0 
! xx=1+imag, ix= 1 -> ii= 2 
! xx=1-imag, ix=-1 -> ii=-2 
! xx=1-imag, ix= 1 -> ii= 0 
! xx=1     , ix= 1 -> ii= 0  convention that log(-1)=pi on
! xx=1     , ix=-1 -> ii=-2  the principal Riemann-sheet
!*******************************************************************
  type(qmplx_type) ,intent(in) :: xx
  integer :: ii,jj
  include 'avh_olo_real.h90'
    :: xim
  jj = mod(xx%p,2)
  ii = xx%p-jj
  xim = aimag(xx%c)
  if (xim.le.RZRO) then ! also xim=0 <==> log(-1)=pi, not -pi
    if (jj.eq.-1) ii = ii-2
  else
    if (jj.eq. 1) ii = ii+2
  endif
  end function


  function prduct_qq(yy,xx) result(zz)
!*******************************************************************
! Return the product  zz  of  yy  and  xx  
! keeping track of (the multiple of pi of) the phase %p such that
! the real part of  zz%c  remains positive 
!*******************************************************************
  type(qmplx_type) ,intent(in) :: yy,xx
  type(qmplx_type) :: zz
  zz%c = yy%c*xx%c
  zz%p = yy%p+xx%p
  if (areal(zz%c).lt.RZRO) then
    zz%p = zz%p + sgnIm(xx%c)
    zz%c = -zz%c
  endif
  end function

  function prduct_qr(yy,xx) result(zz)
!*******************************************************************
! Return the product  zz  of  yy  and  xx  
! keeping track of (the multiple of pi of) the phase %p such that
! the real part of  zz%c  remains positive 
!*******************************************************************
  type(qmplx_type) ,intent(in) :: yy
  include 'avh_olo_real.h90'
    ,intent(in) :: xx
  type(qmplx_type) :: zz
  zz%c = yy%c*abs(xx)
  zz%p = yy%p
  end function

  function ratio_qq(yy,xx) result(zz)
!*******************************************************************
! Return the ratio  zz  of  yy  and  xx  
! keeping track of (the multiple of pi of) the phase %p such that
! the real part of  zz%c  remains positive 
!*******************************************************************
  type(qmplx_type) ,intent(in) :: yy,xx
  type(qmplx_type) :: zz
  zz%c = yy%c/xx%c
  zz%p = yy%p-xx%p
  if (areal(zz%c).lt.RZRO) then
    zz%p = zz%p - sgnIm(xx%c)
    zz%c = -zz%c
  endif
  end function

  function ratio_qr(yy,xx) result(zz)
!*******************************************************************
!*******************************************************************
  type(qmplx_type) ,intent(in) :: yy
  include 'avh_olo_real.h90'
    ,intent(in) :: xx
  type(qmplx_type) :: zz
  zz%c = yy%c/abs(xx)
  zz%p = yy%p
  end function


  function logc(xx) result(rslt)
!*******************************************************************
! log(xx)
!*******************************************************************
  type(qmplx_type) ,intent(in) :: xx
  include 'avh_olo_complex.h90'
    :: rslt
!  rslt = olog(acmplx(xx%c),xx%p)
  rslt = olog(xx%c,xx%p)
  end function

  function logc2(xx) result(rslt)
!*******************************************************************
! log(xx)/(1-xx)
!*******************************************************************
  type(qmplx_type) ,intent(in) :: xx
  include 'avh_olo_complex.h90'
    :: rslt
!  rslt = -olog2(acmplx(xx%c),xx%p)
  rslt = -olog2(xx%c,xx%p)
  end function

  function li2c(xx) result(rslt)
!*******************************************************************
!    /1    ln(1-(1-xx)*t)
!  - |  dt -------------- 
!    /0        t
!*******************************************************************
  type(qmplx_type) ,intent(in) :: xx
  include 'avh_olo_complex.h90'
    :: rslt
!  rslt = dilog(acmplx(xx%c),xx%p)
  rslt = dilog(xx%c,xx%p)
  end function

  function li2c2(xx,yy) result(rslt)
!*******************************************************************
! ( li2(xx) - li2(yy) )/(xx-yy)
!*******************************************************************
  type(qmplx_type) ,intent(in) :: xx,yy
  include 'avh_olo_complex.h90'
    :: rslt
!  rslt = dilog( acmplx(xx%c),xx%p ,acmplx(yy%c),yy%p )
!  write(*,*) 'li2c2 x:',xx%c,xx%p !DEBUG
!  write(*,*) 'li2c2 y:',yy%c,yy%p !DEBUG
  rslt = dilog( xx%c,xx%p ,yy%c,yy%p )
!  write(*,*) 'li2c2 out:',rslt !DEBUG
  end function


end module
