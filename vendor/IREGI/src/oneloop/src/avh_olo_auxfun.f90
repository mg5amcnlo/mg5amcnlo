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


module avh_olo_auxfun
  use avh_olo_units
  use avh_olo_prec

  implicit none
  private
  public :: mysqrt,eta5,eta3,eta2,sgnIm,sgnRe,kallen
  public :: solabc,rfun,rfun0,solabc_rcc

  interface mysqrt
    module procedure mysqrt_c,mysqrt_cr,mysqrt_ci
  end interface

  interface eta5
    module procedure eta5_0
  end interface
  interface eta3
    module procedure eta3_r,eta3_0
  end interface
  interface eta2
    module procedure eta2_r,eta2_0
  end interface

  interface sgnIm
    module procedure sgnIm_c,sgnIm_ci
  end interface
  interface sgnRe
    module procedure sgnRe_c,sgnRe_r,sgnRe_ri
  end interface

contains


  function mysqrt_c(xx) result(rslt)
!*******************************************************************
! Returns the square-root of xx .
! If  Im(xx)  is equal zero and  Re(xx)  is negative, the result is
! negative imaginary.
!*******************************************************************
  include 'avh_olo_complex.h90'
    ,intent(in) :: xx
  include 'avh_olo_complex.h90'
    :: rslt ,zz
  include 'avh_olo_real.h90'
    :: xim,xre
  xim = aimag(xx)
  if (xim.eq.RZRO) then
    xre = areal(xx)
    if (xre.ge.RZRO) then
      zz = acmplx(sqrt(xre),0)
    else
      zz = acmplx(0,-sqrt(-xre))
    endif
  else
    zz = sqrt(xx)
  endif
  rslt = zz
  end function

  function mysqrt_cr(xx,sgn) result(rslt)
!*******************************************************************
! Returns the square-root of xx .
! If  Im(xx)  is equal zero and  Re(xx)  is negative, the result is
! imaginary and has the same sign as  sgn .
!*******************************************************************
  include 'avh_olo_complex.h90'
    ,intent(in) :: xx
  include 'avh_olo_real.h90'
    ,intent(in) :: sgn
  include 'avh_olo_complex.h90'
    :: rslt ,zz
  include 'avh_olo_real.h90'
    :: xim,xre
  xim = aimag(xx)
  if (xim.eq.RZRO) then
    xre = areal(xx)
    if (xre.ge.RZRO) then
      zz = acmplx(sqrt(xre),0)
    else
      zz = acmplx(0,sign(sqrt(-xre),sgn))
    endif
  else
    zz = sqrt(xx)
  endif
  rslt = zz
  end function

  function mysqrt_ci(xx,sgn) result(rslt)
!*******************************************************************
! Returns the square-root of xx .
! If  Im(xx)  is equal zero and  Re(xx)  is negative, the result is
! imaginary and has the same sign as  sgn .
!*******************************************************************
  include 'avh_olo_complex.h90'
          ,intent(in) :: xx
  integer ,intent(in) :: sgn
  include 'avh_olo_complex.h90'
    :: rslt ,zz
  include 'avh_olo_real.h90'
    :: xim,xre,hh
  xim = aimag(xx)
  if (xim.eq.RZRO) then
    xre = areal(xx)
    if (xre.ge.RZRO) then
      zz = acmplx(sqrt(xre),0)
    else
      hh = sgn
      zz = acmplx(0,sign(sqrt(-xre),hh))
    endif
  else
    zz = sqrt(xx)
  endif
  rslt = zz
  end function


  subroutine solabc( x1,x2 ,dd ,aa,bb,cc ,imode )
!*******************************************************************
! Returns the solutions  x1,x2  to the equation  aa*x^2+bb*x+cc=0
! Also returns  dd = aa*(x1-x2)
! If  imode=/=0  it uses  dd  as input as value of  sqrt(b^2-4*a*c)
!*******************************************************************
  include 'avh_olo_complex.h90'
    ,intent(out)   :: x1,x2
  include 'avh_olo_complex.h90'
    ,intent(inout) :: dd
  include 'avh_olo_complex.h90'
    ,intent(in) :: aa,bb,cc
  integer         ,intent(in) :: imode
  include 'avh_olo_complex.h90'
    :: qq,hh
  include 'avh_olo_real.h90'
    :: r1,r2

  if (aa.eq.CZRO) then
    if (bb.eq.CZRO) then
      if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop solabc: ' &
        ,'no solutions, returning 0'
      x1 = 0
      x2 = 0
      dd = 0
    else
      x1 = -cc/bb
      x2 = x1
      dd = bb
    endif
  elseif (cc.eq.CZRO) then
    dd = -bb
    x1 = dd/aa
    x2 = 0
  else
    if (imode.eq.0) dd = sqrt(bb*bb - 4*aa*cc)
    qq = -bb+dd
    hh = -bb-dd
    r1 = abs(qq)
    r2 = abs(hh)
    if (r1.ge.r2) then
      x1 = qq/(2*aa)
      x2 = (2*cc)/qq
    else
      qq = hh
      x2 = qq/(2*aa)
      x1 = (2*cc)/qq
    endif
  endif
  end subroutine


  subroutine solabc_rcc( x1,x2 ,aa,bb,cc )
!*******************************************************************
! Tested
!*******************************************************************
  intent(out) :: x1,x2
  intent(in ) :: aa,bb,cc
  include 'avh_olo_complex.h90'
    :: x1,x2,bb,cc ,t1,t2
  include 'avh_olo_real.h90'
    :: aa,xx,yy,pp,qq,uu,vv,pq1,pq2,uv1,uv2,dd,xd1,xd2,yd1,yd2 &
      ,gg,hh,rx1,rx2,ix1,ix2
  if (aa.eq.RZRO) then
    if (bb.eq.CZRO) then
      if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop solabc: ' &
        ,'no solutions, returning 0'
      x1 = 0
      x2 = 0
    else
      x1 = -cc/bb
      x2 = x1
    endif
  elseif (cc.eq.CZRO) then
    x1 = -bb/aa
    x2 = 0
  else
    t1 = cc/aa          ;xx= areal(t1) ;yy= aimag(t1)
    t2 = bb/(aa*2)      ;pp=-areal(t2) ;uu=-aimag(t2)
    t2 = sqrt(t2*t2-t1) ;qq= areal(t2) ;vv= aimag(t2)
    pq1=pp+qq ;uv1=uu+vv
    pq2=pp-qq ;uv2=uu-vv
    dd=pq1*pq1+uv1*uv1 ;xd1=xx/dd ;yd1=yy/dd
    dd=pq2*pq2+uv2*uv2 ;xd2=xx/dd ;yd2=yy/dd
    if (abs(pq1).gt.abs(pq2)) then
      rx1 = pq1
      gg=xd1*pq1 ;hh=yd1*uv1
      rx2 = gg+hh
      if (abs(rx2).lt.neglig(prcpar)*max(abs(gg),abs(hh))) rx2 = 0
    else
      rx2 = pq2
      gg=xd2*pq2 ;hh=yd2*uv2
      rx1 = gg+hh
      if (abs(rx1).lt.neglig(prcpar)*max(abs(gg),abs(hh))) rx1 = 0
    endif
    if (abs(uv1).gt.abs(uv2)) then
      ix1 = uv1
      gg=yd1*pq1 ;hh=xd1*uv1
      ix2 = gg-hh
      if (abs(ix2).lt.neglig(prcpar)*max(abs(gg),abs(hh))) ix2 = 0
    else
      ix2 = uv2
      gg=yd2*pq2 ;hh=xd2*uv2
      ix1 = gg-hh
      if (abs(ix1).lt.neglig(prcpar)*max(abs(gg),abs(hh))) ix1 = 0
    endif
    x1 = acmplx(rx1,ix1)
    x2 = acmplx(rx2,ix2)
  endif
  end subroutine


  subroutine rfun(rr,dd ,qq)
!*******************************************************************
! Returns  rr  such that  qq = rr + 1/rr  and  Im(rr)  has the same
! sign as  Im(qq) .
! If  Im(qq)  is zero, then  Im(rr)  is negative or zero.
! If  Im(rr)  is zero, then  |rr| > 1/|rr| .
! Also returns  dd = rr - 1/rr .
!*******************************************************************
  include 'avh_olo_complex.h90'
    ,intent(out) :: rr,dd
  include 'avh_olo_complex.h90'
    ,intent(in)  :: qq
  include 'avh_olo_complex.h90'
    :: r2
  include 'avh_olo_real.h90'
    :: aa,bb
  integer :: ir,ik
  dd = sqrt(qq*qq-4)
  rr = qq+dd
  r2 = qq-dd
  aa = abs(rr)
  bb = abs(r2)
  if (bb.gt.aa) then
    rr = r2
    dd = -dd
  endif
  aa = aimag(qq)
  bb = aimag(rr)
  if (aa.eq.RZRO) then
    if (bb.le.RZRO) then
      rr = rr/2
    else
      rr = 2/rr
      dd = -dd
    endif
  else
    ik = sgnRe(aa)
    ir = sgnRe(bb)
    if (ir.eq.ik) then
      rr = rr/2
    else
      rr = 2/rr
      dd = -dd
    endif
  endif
  end subroutine

  subroutine rfun0(rr ,dd,qq)
!*******************************************************************
! Like rfun, but now  dd  is input, which may get a minus sign
!*******************************************************************
  include 'avh_olo_complex.h90'
    ,intent(out)   :: rr
  include 'avh_olo_complex.h90'
    ,intent(inout) :: dd
  include 'avh_olo_complex.h90'
    ,intent(in)  :: qq
  include 'avh_olo_complex.h90'
    :: r2
  include 'avh_olo_real.h90'
    :: aa,bb
  integer :: ir,ik
  rr = qq+dd
  r2 = qq-dd
  aa = abs(rr)
  bb = abs(r2)
  if (bb.gt.aa) then
    rr = r2
    dd = -dd
  endif
  aa = aimag(qq)
  bb = aimag(rr)
  if (aa.eq.RZRO) then
    if (bb.le.RZRO) then
      rr = rr/2
    else
      rr = 2/rr
      dd = -dd
    endif
  else
    ik = sgnRe(aa)
    ir = sgnRe(bb)
    if (ir.eq.ik) then
      rr = rr/2
    else
      rr = 2/rr
      dd = -dd
    endif
  endif
  end subroutine


  function eta3_r( aa,sa ,bb,sb ,cc,sc ) result(rslt)
!*******************************************************************
! 2*pi*imag times the result of
!     theta(-Im(a))*theta(-Im(b))*theta( Im(c))
!   - theta( Im(a))*theta( Im(b))*theta(-Im(c))
! where a,b,c are interpreted as a+i|eps|sa, b+i|eps|sb, c+i|eps|sc
!*******************************************************************
  include 'avh_olo_complex.h90'
    ,intent(in) :: aa,bb,cc
  include 'avh_olo_real.h90'
    ,intent(in) :: sa,sb,sc
  include 'avh_olo_complex.h90'
    :: rslt
  include 'avh_olo_real.h90'
    :: ima,imb,imc
  ima = aimag(aa)
  imb = aimag(bb)
  imc = aimag(cc)
  if (ima.eq.RZRO) ima = sa
  if (imb.eq.RZRO) imb = sb
  if (imc.eq.RZRO) imc = sc
  ima = sgnRe(ima)
  imb = sgnRe(imb)
  imc = sgnRe(imc)
  if (ima.eq.imb.and.ima.ne.imc) then
    rslt = acmplx(0,imc*TWOPI)
  else
    rslt = 0
  endif
  end function

  function eta3_0( aa ,bb ,cc ) result(rslt)
!*******************************************************************
! 2*pi*imag times the result of
!     theta(-Im(a))*theta(-Im(b))*theta( Im(c))
!   - theta( Im(a))*theta( Im(b))*theta(-Im(c))
! where a,b,c are interpreted as a+i|eps|sa, b+i|eps|sb, c+i|eps|sc
!*******************************************************************
  include 'avh_olo_complex.h90'
    ,intent(in) :: aa,bb,cc
  include 'avh_olo_complex.h90'
    :: rslt
  include 'avh_olo_real.h90'
    :: ima,imb,imc
  ima = sgnIm(aa)
  imb = sgnIm(bb)
  imc = sgnIm(cc)
  if (ima.eq.imb.and.ima.ne.imc) then
    rslt = acmplx(0,imc*TWOPI)
  else
    rslt = 0
  endif
  end function

  function eta5_0( aa ,b1,c1 ,b2,c2 ) result(rslt)
!*******************************************************************
! eta3(aa,b1,c1) - eta3(aa,b2,c2)
!*******************************************************************
  include 'avh_olo_complex.h90'
    ,intent(in) :: aa,b1,c1 ,b2,c2
  include 'avh_olo_complex.h90'
    :: rslt
  include 'avh_olo_real.h90'
    :: imaa,imb1,imc1,imb2,imc2
  imaa = sgnIm(aa)
  imb1 = sgnIm(b1)
  imb2 = sgnIm(b2)
  imc1 = sgnIm(c1)
  imc2 = sgnIm(c2)
  if (imaa.eq.imb1) then
    if (imaa.eq.imb2) then
      if (imc1.eq.imc2) then
        rslt = 0
      elseif (imaa.ne.imc1) then
        rslt = acmplx(0, imc1*TWOPI)
      else
        rslt = acmplx(0,-imc2*TWOPI)
      endif
    elseif (imaa.ne.imc1) then
      rslt = acmplx(0, imc1*TWOPI)
    else
      rslt = 0
    endif
  elseif (imaa.eq.imb2.and.imaa.ne.imc2) then
    rslt = acmplx(0,-imc2*TWOPI)
  else
    rslt = 0
  endif
  end function

  function eta2_r( aa,sa ,bb,sb ) result(rslt)
!*******************************************************************
! The same as  eta3, but with  c=a*b, so that
!   eta(a,b) = log(a*b) - log(a) - log(b)
!*******************************************************************
  include 'avh_olo_complex.h90'
    ,intent(in) :: aa,bb
  include 'avh_olo_real.h90'
    ,intent(in) :: sa,sb
  include 'avh_olo_complex.h90'
    :: rslt
  include 'avh_olo_real.h90'
    :: rea,reb,ima,imb,imab
  rea = areal(aa)  ;ima = aimag(aa)
  reb = areal(bb)  ;imb = aimag(bb)
  imab = rea*imb + reb*ima
  if (ima .eq.RZRO) ima = sa
  if (imb .eq.RZRO) imb = sb
  if (imab.eq.RZRO) imab = sign(rea,sb) + sign(reb,sa)
  ima  = sgnRe(ima)
  imb  = sgnRe(imb)
  imab = sgnRe(imab)
  if (ima.eq.imb.and.ima.ne.imab) then
    rslt = acmplx(0,imab*TWOPI)
  else
    rslt = 0
  endif
  end function
 
  function eta2_0( aa ,bb ) result(rslt)
!*******************************************************************
!*******************************************************************
  include 'avh_olo_complex.h90'
    ,intent(in) :: aa,bb
  include 'avh_olo_complex.h90'
    :: rslt
  include 'avh_olo_real.h90'
    :: rea,reb,ima,imb,imab
  rea = areal(aa)  ;ima = aimag(aa)
  reb = areal(bb)  ;imb = aimag(bb)
  rea = rea*imb
  reb = reb*ima
  imab = rea+reb
  ima  = sgnRe(ima)
  imb  = sgnRe(imb)
  imab = sgnRe(imab)
  if (ima.eq.imb.and.ima.ne.imab) then
    rslt = acmplx(0,imab*TWOPI)
  else
    rslt = 0
  endif
  end function 


  function kallen( p1,p2,p3 ) result(rslt)
!*******************************************************************
!  p1^2 + p2^2 + p3^2 - 2*p1*p2 - 2*p2*p3 - 2*p3*p1
!*******************************************************************
  include 'avh_olo_complex.h90'
    ,intent(in) :: p1,p2,p3
  include 'avh_olo_complex.h90'
    :: rslt ,y1,y2,y3
  include 'avh_olo_real.h90'
    :: b1,b2,b3
  y1=p2*p3 ;b1=areal(y1)
  y2=p3*p1 ;b2=areal(y2)
  y3=p1*p2 ;b3=areal(y3)
      if (b1.le.RZRO) then  ;rslt = (p1-p2-p3)**2 - 4*y1
  elseif (b2.le.RZRO) then  ;rslt = (p2-p3-p1)**2 - 4*y2
  elseif (b3.le.RZRO) then  ;rslt = (p3-p1-p2)**2 - 4*y3
  elseif (b1.le.b2.and.b1.le.b3) then  ;rslt = (p1-p2-p3)**2 - 4*y1
  elseif (b2.le.b3.and.b2.le.b1) then  ;rslt = (p2-p3-p1)**2 - 4*y2
                                 else  ;rslt = (p3-p1-p2)**2 - 4*y3
  endif
  end function


  function sgnIm_c(zz) result(rslt)
!*******************************************************************
!*******************************************************************
  include 'avh_olo_complex.h90'
    ,intent(in) :: zz
  integer :: rslt
  include 'avh_olo_real.h90'
    :: imz
  imz = aimag(zz)
  if (imz.ge.RZRO) then ;rslt= 1
                   else ;rslt=-1
  endif
  end function

  function sgnIm_ci(zz,ii) result(rslt)
!*******************************************************************
!*******************************************************************
  include 'avh_olo_complex.h90'
          ,intent(in) :: zz
  integer ,intent(in) :: ii
  integer :: rslt
  include 'avh_olo_real.h90'
    :: imz
  imz = aimag(zz)
  if     (imz.gt.RZRO) then ;rslt= 1
  elseif (imz.lt.RZRO) then ;rslt=-1
                       else ;rslt= sign(1,ii)
  endif
  end function

  function sgnRe_c(zz) result(rslt)
!*******************************************************************
!*******************************************************************
  include 'avh_olo_complex.h90'
    ,intent(in) :: zz
  integer :: rslt
  include 'avh_olo_real.h90'
    :: rez
  rez = zz
  if (rez.ge.RZRO) then ;rslt= 1
                   else ;rslt=-1
  endif
  end function

  function sgnRe_r(rez) result(rslt)
!*******************************************************************
!*******************************************************************
  include 'avh_olo_real.h90'
    ,intent(in) :: rez
  integer :: rslt
  if (rez.ge.RZRO) then ;rslt= 1
                   else ;rslt=-1
  endif
  end function

  function sgnRe_ri(rez,ii) result(rslt)
!*******************************************************************
!*******************************************************************
  include 'avh_olo_real.h90'
          ,intent(in) :: rez
  integer ,intent(in) :: ii
  integer :: rslt
  if     (rez.gt.RZRO) then ;rslt= 1
  elseif (rez.lt.RZRO) then ;rslt=-1
                       else ;rslt=sign(1,ii)
  endif
  end function

end module
