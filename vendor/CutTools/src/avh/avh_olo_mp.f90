!
!  Copyright (C) 2011 Andreas van Hameren. 
!
!  This is the package OneLOop-2.2.
! 
!  OneLOop-2.2 is free software: you can redistribute it and/or modify
!  it under the terms of the GNU General Public License as published by
!  the Free Software Foundation, either version 3 of the License, or
!  (at your option) any later version.
!
!  OneLOop-2.2 is distributed in the hope that it will be useful,
!  but WITHOUT ANY WARRANTY; without even the implied warranty of
!  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!  GNU General Public License for more details.
!
!  You should have received a copy of the GNU General Public License
!  along with OneLOop-2.2.  If not, see <http://www.gnu.org/licenses/>.
!
module avh_olo_mp_xkind
! use !XKIND_MODULE
!
  implicit none
  private
  public :: olo_mp_xkind
!
  integer ,parameter :: olo_mp_xkind=kind(1d0) !XKIND_SETPAR
!  
end module

module avh_olo_mp_kinds
  include 'cts_mprec.h'
  use avh_olo_mp_xkind
  implicit none
  private
  public :: kindr2,kindc2 &
           ,R0P0,R1P0,R5M1,TWOPI,SQRT2,C0P0,C1P0,CiP0,pi,pi2o6,ipi &
           ,mp_eps,oieps,pi2
  public :: avh_olo_mp_load_constants,mp_rl,mp_ig,mp_cx,mp_sn,mp_ml
  public :: countdigits
  private mp_epsilon
  include 'cts_mpr.h'
   :: R0P0,R1P0,R5M1,TWOPI,SQRT2,pi,pi2o6,mp_eps
  include 'cts_mpc.h'
   :: C0P0,C1P0,CiP0,ipi,oieps,pi2
  integer ,parameter :: kindr2 = olo_mp_xkind
  integer ,parameter :: kindc2 = kindr2
!
  interface epsilon
   procedure mp_epsilon
  end interface
  contains
!
  function mp_epsilon(arg)
  type(mp_real), intent(in) :: arg
  type(mp_real)  :: mp_epsilon,mp_aus
  mp_aus= 10.d0
  mp_aus= mp_aus**28
  mp_epsilon= 1/mp_aus
  end function
!
  subroutine countdigits(mcountd)
  implicit none
  include 'cts_mpr.h'
   :: x
  integer :: i,digit
  integer, intent(out) :: mcountd
  x      = 1.d0
  x      = x/3.d0
  mcountd= 0; digit  = 3
  do while(digit.eq.3)
   digit  = nint(10*x-x)
   mcountd= mcountd+1
   x      = 10*x-3
  enddo
  mcountd  = mcountd-1
  end subroutine
!
  subroutine avh_olo_mp_load_constants                               
  implicit none                                                      
  R0P0  = 0.d0                                                       
  R1P0  = 1.d0                                                       
  C0P0  = dcmplx(R0P0,R0P0)           
  C1P0  = dcmplx(R1P0,R0P0)                              
  CiP0  = dcmplx(R0P0,R1P0)     
  R5M1  = 0.5d0                                                      
  TWOPI = 1.d0                                                       
  TWOPI = 8.d0*atan(TWOPI)                                           
  SQRT2 = 2.d0                                                       
  SQRT2 = sqrt(SQRT2)                                                
  pi    = TWOPI/2.d0                                                 
  pi2o6 = C1P0*TWOPI*TWOPI/24                                        
  ipi   = CiP0*TWOPI/2
  pi2   = C1P0*TWOPI*TWOPI/4
  mp_eps= epsilon(R1P0)
  oieps = C1P0+CiP0*mp_eps**2
  end subroutine                                                     
!
  function mp_rl(arg)                                                
  implicit none                                                      
  include 'cts_mpr.h'                                                
   :: mp_rl,aus                                                      
  include 'cts_mpc.h'                                                
   ,intent(in) :: arg                                                
  aus= (arg+conjg(arg))/2.d0                                         
  mp_rl= aus                                                         
  end function                                                       
!                                                                
  function mp_ig(arg)                                                
  implicit none                                                      
  include 'cts_mpr.h'                                                
   :: mp_ig,aus                                                      
  include 'cts_mpc.h'                                                
   ,intent(in) :: arg 
  aus= (arg-conjg(arg))/(2.d0*CiP0)                                  
  mp_ig= aus                                                         
  end function                                                       
!                                                                
  function mp_cx(arg1,arg2)                                          
  implicit none                                                      
  include 'cts_mpc.h'                                                
   :: mp_cx,aus                                                      
  include 'cts_mpr.h'                                                
   ,intent(in) :: arg1,arg2                                          
  aus= arg1+arg2*CiP0                                                
  mp_cx= aus                                                         
  end function                                                           
!                                                                
  function mp_sn(arg1,arg2)                                          
  implicit none                                                      
  include 'cts_mpr.h'                                                
   :: mp_sn,aus                                                      
  real(kind(1.d0)),parameter :: m1= -1.d0,zero= 0.d0 
  include 'cts_mpr.h'                                                
   ,intent(in) :: arg1,arg2                                          
  aus= arg1 
  if (arg2.ge.zero) then
   if (arg1.lt.zero) aus= aus*m1
  else
   if (arg1.ge.zero) aus= aus*m1
  endif
  mp_sn= aus                                                         
  end function
!                                                                
  function mp_ml(a,n)                                          
!
! this is the mp version of maxval for a vector with n elements
!
  implicit none                                                      
  include 'cts_mpr.h'                                                
   :: mp_ml,aus                                                      
  integer, intent(in) :: n
  include 'cts_mpr.h'                                                
   ,intent(in) :: a(n)                                          
  integer :: i
  aus= a(1)
  do i= 2,n
   if (a(i).ge.a(i-1)) aus= a(i)
  enddo
  mp_ml= aus                                                         
  end function
!
end module

module avh_olo_mp_units
  implicit none
  integer :: eunit=6 !PROTECTED
  integer :: wunit=6 !PROTECTED
  integer :: munit=6 !PROTECTED
  integer :: punit=0 !PROTECTED ! print all
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


module avh_olo_mp_print
  include 'cts_mprec.h'
  use avh_olo_mp_kinds
  implicit none
  private
  public :: myprint,init_print
!
  integer ,parameter :: noverh=10 !maximally 6 decimals for exponent
  integer :: ndigits=19
  integer :: nefrmt=19+noverh
!
  interface myprint
    module procedure printr,printc,printi
  end interface
!
contains
!
  subroutine init_print( ndig )
  integer ,intent(in) :: ndig
  ndigits = ndig+ndig/4+1
  nefrmt  = ndigits+noverh
  end subroutine
! 
  function printc( zz ) result(rslt)
  include 'cts_mpc.h'
   ,intent(in) :: zz
  include 'cts_mpr.h'
   :: zzr,zzi
  character(nefrmt*2+3) :: rslt
  zzr= mp_rl(zz)
  zzi= mp_ig(zz)
  rslt = '('//trim(printr(zzr)) &
       //','//trim(printr(zzi           )) &
       //')'
  rslt = adjustl(rslt)
  end function
!
  function printr( xx ) result(rslt)
  include 'cts_mpr.h'
   ,intent(in) :: xx
  include 'cts_dpr.h'
   :: xxr
  character(nefrmt  ) :: rslt
  character(nefrmt+1) :: cc
  character(10) :: aa,bb
  xxr= xx
  write(aa,'(i10)') nefrmt+1  ;aa=adjustl(aa)
  write(bb,'(i10)') ndigits   ;bb=adjustl(bb)
  aa = '(e'//trim(aa)//'.'//trim(bb)//')'
  write(cc,aa) xxr  ;cc=adjustl(cc)
  if (cc(1:2).eq.'-0') then ;rslt = '-'//cc(3:ndigits*2)
  else                      ;rslt = ' '//cc(2:ndigits*2)
  endif
  end function
!
  function printi( ii ) result(rslt)
  integer ,intent(in) :: ii
  character(ndigits) :: rslt
  character(ndigits) :: cc
  character(10) :: aa
  write(aa,'(i10)') ndigits ;aa=adjustl(aa)
  aa = '(i'//trim(aa)//')'
  write(cc,aa) ii ;cc=adjustl(cc)
  if (cc(1:1).ne.'-') then ;rslt=' '//cc
  else                     ;rslt=cc 
  endif
  end function
!
end module


module avh_olo_mp_func
  include 'cts_mprec.h'
  use avh_olo_mp_kinds
  use avh_olo_mp_units
!
  implicit none
!
  type :: qmplx_type
   include 'cts_mpc.h'
    :: c
   integer         :: p
  end type
!
  interface mysqrt
    module procedure mysqrt_0,mysqrt_r,mysqrt_i
  end interface
!
  interface qonv
    module procedure qonv_r,qonv_0,qonv_i
  end interface
!
  interface operator (*)
    module procedure prduct,prduct_r
  end interface
  interface operator (/)
    module procedure ratio,ratio_r
  end interface
!
contains
!
!
   function mysqrt_0(xx) result(rslt)
!*******************************************************************
! Returns the square-root of xx .
! If  Im(xx)  is equal zero and  Re(xx)  is negative, the result is
! negative imaginary.
!*******************************************************************
   include 'cts_mpc.h'
    ,intent(in) :: xx
   include 'cts_mpc.h'
    :: rslt ,zz
   include 'cts_mpr.h'
    :: xim,xre
   xim = mp_ig(xx)
   if (xim.eq.R0P0) then
     xre = mp_rl(xx)
     if (xre.ge.R0P0) then
       zz = mp_cx(sqrt(xre),R0P0)
     else
       zz = mp_cx(R0P0,-sqrt(-xre))
     endif
   else
     zz = sqrt(xx)
   endif
   rslt = zz
   end function

   function mysqrt_r(xx,sgn) result(rslt)
!*******************************************************************
! Returns the square-root of xx .
! If  Im(xx)  is equal zero and  Re(xx)  is negative, the result is
 ! imaginary and has the same sign as  sgn .
!*******************************************************************
   include 'cts_mpc.h'
    ,intent(in) :: xx
   include 'cts_mpr.h'
    ,intent(in) :: sgn
   include 'cts_mpc.h'
    :: rslt ,zz
   include 'cts_mpr.h'
    :: xim,xre
   xim = mp_ig(xx)
   if (xim.eq.R0P0) then
     xre = mp_rl(xx)
     if (xre.ge.R0P0) then
       zz = mp_cx(sqrt(xre),R0P0)
     else
       zz = mp_cx(R0P0,mp_sn(sqrt(-xre),sgn))
     endif
   else
     zz = sqrt(xx)
   endif
   rslt = zz
   end function

   function mysqrt_i(xx,sgn) result(rslt)
!*******************************************************************
! Returns the square-root of xx .
! If  Im(xx)  is equal zero and  Re(xx)  is negative, the result is
! imaginary and has the same sign as  sgn .
!*******************************************************************
   include 'cts_mpc.h'
    ,intent(in) :: xx
   integer         ,intent(in) :: sgn
   include 'cts_mpc.h'
    :: rslt ,zz,csgn
   include 'cts_mpr.h'
    :: xim,xre
   csgn= sgn*C1P0
   xim = mp_ig(xx)
   if (xim.eq.R0P0) then
     xre = mp_rl(xx)
     if (xre.ge.R0P0) then
       zz = mp_cx(sqrt(xre),R0P0)
     else
       zz = mp_cx(R0P0,mp_sn(sqrt(-xre),mp_rl(csgn)))
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
   include 'cts_mpc.h'
    ,intent(out)   :: x1,x2
   include 'cts_mpc.h'
    ,intent(inout) :: dd
   include 'cts_mpc.h'
    ,intent(in) :: aa,bb,cc
   integer         ,intent(in) :: imode
   include 'cts_mpc.h'
    :: qq,hh
   include 'cts_mpr.h'
    :: r1,r2
!
   if (aa.eq.C0P0) then
     if (bb.eq.C0P0) then
       if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop solabc: ' &
         ,'no solutions, returning 0'
       x1 = C0P0
       x2 = C0P0
       dd = C0P0
     else
       x1 = -cc/bb
       x2 = x1
       dd = bb
     endif
   elseif (cc.eq.C0P0) then
     dd = -bb
     x1 = dd/aa
     x2 = C0P0
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


   subroutine rfun(rr,dd ,qq)
!*******************************************************************
! Returns  rr  such that  qq = rr + 1/rr  and  Im(rr)  has the same
! sign as  Im(qq) .
! If  Im(qq)  is zero, then  Im(rr)  is negative or zero.
! If  Im(rr)  is zero, then  |rr| > 1/|rr| .
! Also returns  dd = rr - 1/rr .
!*******************************************************************
   include 'cts_mpc.h'
    ,intent(out) :: rr,dd
   include 'cts_mpc.h'
    ,intent(in)  :: qq
   include 'cts_mpc.h'
    :: r2
   include 'cts_mpr.h'
    :: aa,bb
   integer :: ir,ik
   include 'cts_mpc.h'
    :: two,four
   two=2.d0*C1P0
   four=4.d0*C1P0
   dd = sqrt(qq*qq-four)
   rr = qq+dd
   r2 = qq-dd
   aa = abs(rr)
   bb = abs(r2)
   if (bb.gt.aa) then
     rr = r2
     dd = -dd
   endif
   aa = mp_ig(qq)
   bb = mp_ig(rr)
   if (aa.eq.R0P0) then
     if (bb.le.R0P0) then
       rr = rr/two
     else
       rr = two/rr
       dd = -dd
     endif
   else
     ik = int(mp_sn(R1P0,aa))
     ir = int(mp_sn(R1P0,bb))
     if (ir.eq.ik) then
       rr = rr/two
     else
       rr = two/rr
       dd = -dd
     endif
   endif
   end subroutine

   subroutine rfun0(rr ,dd,qq)
!*******************************************************************
! Like rfun, but now  dd  is input, which may get a minus sign
!*******************************************************************
   include 'cts_mpc.h'
    ,intent(out)   :: rr
   include 'cts_mpc.h'
    ,intent(inout) :: dd
   include 'cts_mpc.h'
    ,intent(in)  :: qq
   include 'cts_mpc.h'
    :: r2
   include 'cts_mpr.h'
    :: aa,bb
   integer :: ir,ik
   include 'cts_mpc.h'
    :: two
   two=2.d0*C1P0
   rr = qq+dd
   r2 = qq-dd
   aa = abs(rr)
   bb = abs(r2)
   if (bb.gt.aa) then
     rr = r2
     dd = -dd
   endif
   aa = mp_ig(qq)
   bb = mp_ig(rr)
   if (aa.eq.R0P0) then
     if (bb.le.R0P0) then
       rr = rr/two
     else
       rr = two/rr
       dd = -dd
     endif
   else
     ik = int(mp_sn(R1P0,aa))
     ir = int(mp_sn(R1P0,bb))
     if (ir.eq.ik) then
       rr = rr/two
     else
       rr = two/rr
       dd = -dd
     endif
   endif
   end subroutine


   function qonv_r(xx,sgn) result(rslt)
!*******************************************************************
! zz=rslt%c ,iz=rslt%p
! Determine  zz,iz  such that  xx = zz*exp(iz*imag*pi)  and  Re(zz)
! is positive. If  Im(x)=0  and  Re(x)<0  then  iz  becomes the
! sign of  sgn .
!*******************************************************************
   include 'cts_mpc.h'
    ,intent(in) :: xx
   include 'cts_mpr.h'
    ,intent(in) :: sgn
   type(qmplx_type) :: rslt
   include 'cts_mpr.h'
    :: xre,xim
   xre = mp_rl(xx)
   if (xre.ge.R0P0) then
     rslt%c = xx
     rslt%p = 0
   else
     xim = mp_ig(xx)
     if (xim.eq.R0P0) then
       rslt%c = mp_cx(-xre,R0P0)
       rslt%p = int(mp_sn(R1P0,sgn))
     else
       rslt%c = -xx
       rslt%p = int(mp_sn(R1P0,xim)) ! xim = -Im(rslt%c)
     endif
   endif
   end function
!
   function qonv_i(xx,sgn) result(rslt)
!*******************************************************************
! zz=rslt%c ,iz=rslt%p
! Determine  zz,iz  such that  xx = zz*exp(iz*imag*pi)  and  Re(zz)
! is positive. If  Im(x)=0  and  Re(x)<0  then  iz  becomes the
! sign of  sgn .
!*******************************************************************
   include 'cts_mpc.h'
    ,intent(in) :: xx
   integer         ,intent(in) :: sgn
   type(qmplx_type) :: rslt
   include 'cts_mpr.h'
    :: xre,xim
   xre = mp_rl(xx)
   if (xre.ge.R0P0) then
     rslt%c = xx
     rslt%p = 0
   else
     xim = mp_ig(xx)
     if (xim.eq.R0P0) then
       rslt%c = mp_cx(-xre,R0P0)
       rslt%p = sign(1,sgn)
     else
       rslt%c = -xx
       rslt%p = int(mp_sn(R1P0,xim)) ! xim = -Im(rslt%c)
     endif
   endif
   end function
!
   function qonv_0(xx) result(rslt)
!*******************************************************************
! zz=rslt%c ,iz=rslt%p
! Determine  zz,iz  such that  xx = zz*exp(iz*imag*pi)  and  Re(zz)
! is positive. If  Im(x)=0  and  Re(x)<0  then  iz=1
!*******************************************************************
   include 'cts_mpc.h'
    ,intent(in) :: xx
   type(qmplx_type) :: rslt
   include 'cts_mpr.h'
    :: xre,xim
   xre = mp_rl(xx)
   if (xre.ge.R0P0) then
     rslt%c = xx
     rslt%p = 0
   else
     xim = mp_ig(xx)
     if (xim.eq.R0P0) then
       if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop qonv: ' &
         ,'negative input with undefined sign for the imaginary part, ' &
         ,'putting +ieps'
       rslt%c = mp_cx(-xre,R0P0)
       rslt%p = 1
     else
       rslt%c = -xx
       rslt%p = int(mp_sn(R1P0,xim)) ! xim = -Im(rslt%c)
     endif
   endif
   end function
!
   function directly(xx,ix) result(rslt)
!*******************************************************************
!*******************************************************************
   include 'cts_mpc.h'
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
   include 'cts_mpr.h'
    :: xim
   ii = xx%p/2*2
   jj = xx%p-ii
   xim = mp_ig(xx%c)
   if (xim.le.R0P0) then ! also xim=0 <==> log(-1)=pi, not -pi
     if (jj.eq.-1) ii = ii-2
   else
     if (jj.eq. 1) ii = ii+2
   endif
   end function


   function prduct(yy,xx) result(zz)
!*******************************************************************
! Return the product  zz  of  yy  and  xx  
! keeping track of (the multiple of pi of) the phase %p such that
! the real part of  zz%c  remains positive 
!*******************************************************************
   type(qmplx_type) ,intent(in) :: yy,xx
   type(qmplx_type) :: zz
   zz%c = yy%c*xx%c
   zz%p = yy%p+xx%p
   if (mp_rl(zz%c).lt.R0P0) then
     zz%p = zz%p + int(mp_sn(R1P0,mp_ig(xx%c)))
     zz%c = -zz%c
   endif
   end function

   function prduct_r(yy,xx) result(zz)
!*******************************************************************
! Return the product  zz  of  yy  and  xx  
! keeping track of (the multiple of pi of) the phase %p such that
! the real part of  zz%c  remains positive 
!*******************************************************************
   type(qmplx_type) ,intent(in) :: yy
   include 'cts_mpr.h'
    ,intent(in) :: xx
   type(qmplx_type) :: zz
   zz%c = yy%c*abs(xx)
   zz%p = yy%p
   end function

   function ratio(yy,xx) result(zz)
!*******************************************************************
! Return the ratio  zz  of  yy  and  xx  
! keeping track of (the multiple of pi of) the phase %p such that
! the real part of  zz%c  remains positive 
!*******************************************************************
   type(qmplx_type) ,intent(in) :: yy,xx
   type(qmplx_type) :: zz
   zz%c = yy%c/xx%c
   zz%p = yy%p-xx%p
   if (mp_rl(zz%c).lt.R0P0) then
     zz%p = zz%p - int(mp_sn(R1P0,mp_ig(xx%c)))
     zz%c = -zz%c
   endif
   end function
!
   function ratio_r(yy,xx) result(zz)
!*******************************************************************
!*******************************************************************
   type(qmplx_type) ,intent(in) :: yy
   include 'cts_mpr.h'
    ,intent(in) :: xx
   type(qmplx_type) :: zz
   zz%c = yy%c/abs(xx)
   zz%p = yy%p
   end function
!
!
   function eta3( aa,sa ,bb,sb ,cc,sc ) result(rslt)
!*******************************************************************
! 2*pi*imag times the result of
!     theta(-Im(a))*theta(-Im(b))*theta( Im(c))
!   - theta( Im(a))*theta( Im(b))*theta(-Im(c))
! where a,b,c are interpreted as a+i|eps|sa, b+i|eps|sb, c+i|eps|sc
!*******************************************************************
   include 'cts_mpc.h'
    ,intent(in) :: aa,bb,cc
   include 'cts_mpr.h'
    ,intent(in) :: sa,sb,sc
   include 'cts_mpc.h'
    :: rslt
   include 'cts_mpr.h'
    :: ima,imb,imc
   ima = mp_ig(aa)
   imb = mp_ig(bb)
   imc = mp_ig(cc)
   if (ima.eq.R0P0) ima = sa
   if (imb.eq.R0P0) imb = sb
   if (imc.eq.R0P0) imc = sc
   ima = mp_sn(R1P0,ima)
   imb = mp_sn(R1P0,imb)
   imc = mp_sn(R1P0,imc)
   if (ima.eq.imb.and.ima.ne.imc) then
     rslt = mp_cx(R0P0,imc*TWOPI)
   else
     rslt = R0P0
   endif
   end function
 
   function eta2( aa,sa ,bb,sb ) result(rslt)
!*******************************************************************
! The same as  eta3, but with  c=a*b, so that
!   eta(a,b) = log(a*b) - log(a) - log(b)
!*******************************************************************
   include 'cts_mpc.h'
    ,intent(in) :: aa,bb
   include 'cts_mpr.h'
    ,intent(in) :: sa,sb
   include 'cts_mpr.h'
    :: rslt ,rea,reb,ima,imb,imab
   rea = mp_rl(aa)  ;ima = mp_ig(aa)
   reb = mp_rl(bb)  ;imb = mp_ig(bb)
   imab = rea*imb + reb*ima
   if (ima.eq.R0P0) ima = sa
   if (imb.eq.R0P0) imb = sb
   if (imab.eq.R0P0) imab = mp_sn(rea,sb) + mp_sn(reb,sa)
   ima  = mp_sn(R1P0,ima)
   imb  = mp_sn(R1P0,imb)
   imab = mp_sn(R1P0,imab)
   if (ima.eq.imb.and.ima.ne.imab) then
     rslt = mp_cx(R0P0,imab*TWOPI)
   else
     rslt = R0P0
   endif
   end function 
!
end module


module avh_olo_mp_loga
!*******************************************************************
! log( |xx|*exp(imag*pi*iph) ) = log|xx| + imag*pi*iph
!*******************************************************************
  use avh_olo_mp_kinds
  use avh_olo_mp_units
  use avh_olo_mp_func
  implicit none
  private
  public :: loga
contains
!
  function loga(xx,iph) result(rslt)
  include 'cts_mpr.h'
   ,intent(in) :: xx
  integer      ,intent(in) :: iph
  include 'cts_mpc.h'
   :: rslt
  include 'cts_mpr.h'
   :: rr
  rr = abs(xx)
  if (rr.eq.R0P0) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop loga: ' &
      ,'|xx|=',rr
  endif
  rslt = mp_cx(log(rr),iph*pi)
  end function
!
end module


module avh_olo_mp_bern
!*******************************************************************
! the first nn Bernoulli numbers
!*******************************************************************
  include 'cts_mprec.h'
  use avh_olo_mp_kinds
  implicit none
  private
  public :: init_bern,rbern,cbern
  integer ,parameter :: nn=40
  include 'cts_mpr.h'
   :: rbern(nn) !PROTECTED
  include 'cts_mpc.h'
   :: cbern(nn) !PROTECTED
  integer :: ndigits=0
contains

  subroutine init_bern(ndig)
  integer ,intent(in) :: ndig
  integer :: jj
  integer ,parameter :: d=kindr2
  if (ndigits.eq.ndig) return ;ndigits=ndig
  rbern(1:nn) = R0P0
  rbern( 1) = -1._d
  rbern( 1) = rbern(1)/2._d

  rbern( 2) =  1._d
  rbern( 2) = rbern(2)/6._d

  rbern( 4) = -1._d
  rbern( 4) = rbern(4)/30._d

  rbern( 6) =  1._d
  rbern( 6) = rbern(6)/42._d

  rbern( 8) = -1._d
  rbern( 8) = rbern(8)/30._d

  rbern(10) =  5._d
  rbern(10) = rbern(10)/66._d

  rbern(12) = -691._d
  rbern(12) = rbern(12)/2730._d

  rbern(14) =  7._d
  rbern(14) = rbern(14)/6._d

  rbern(16) = -3617._d
  rbern(16) = rbern(16)/510._d

  rbern(18) =  43867._d
  rbern(18) = rbern(18)/798._d

  rbern(20) = -174611._d
  rbern(20) = rbern(20)/330._d

  rbern(22) =  854513._d
  rbern(22) = rbern(22)/138._d

  rbern(24) = -236364091._d
  rbern(24) = rbern(24)/2730._d

  rbern(26) =  8553103._d
  rbern(26) = rbern(26)/6._d

  rbern(28) = -23749461029._d
  rbern(28) = rbern(28)/870._d

  rbern(30) =  8615841276005._d
  rbern(30) = rbern(30)/14322._d

  rbern(32) = -7709321041217._d
  rbern(32) = rbern(32)/510._d

  rbern(34) =  2577687858367._d
  rbern(34) =  rbern(34)/6._d

  rbern(36) = -26315271553053477373._d
  rbern(36) = rbern(36)/1919190._d

  rbern(38) =  2929993913841559._d
  rbern(38) = rbern(38)/6._d

  rbern(40) = -261082718496449122051._d
  rbern(40) = rbern(40)/13530._d

  do jj=1,nn
    cbern(jj) = mp_cx(rbern(jj),R0P0)
  enddo
  end subroutine
!
end module


module avh_olo_mp_li2a
!*******************************************************************
!                  /1    ln(1-zz*t)
! avh_olo_mp_li2a = - |  dt ---------- 
!                  /0        t
! with  zz = 1 - |xx|*exp(imag*pi*iph)
! Examples:
!   In order to get the dilog of  1.1  use  xx=1.1, iph=0
!   In order to get the dilog of -1.1  use  xx=1.1, iph=1
! Add multiples of  2  to  iph  in order to get the result on
! different Riemann-sheets.
!*******************************************************************
  use avh_olo_mp_kinds
  use avh_olo_mp_func
  use avh_olo_mp_bern
  implicit none
  private
  public :: init_li2a,li2a
!  include 'cts_mpr.h'
!   ,parameter :: pi=TWOPI/2
!  include 'cts_mpc.h'
!   ,parameter :: pi2o6=C1P0*TWOPI*TWOPI/24
  integer :: nn=16
  integer :: ndigits=0
contains
!
  subroutine init_li2a(ndig)
  integer ,intent(in) :: ndig
  if (ndigits.eq.ndig) return ;ndigits=ndig
  call init_bern(ndigits)
  if     (ndigits.lt.24) then
    nn = 16
  else
    nn = 30
  endif
  end subroutine

  function li2a(xx,iph) result(rslt)
  include 'cts_mpr.h'
   ,intent(in) :: xx
  integer      ,intent(in) :: iph
  include 'cts_mpc.h'
   :: rslt
  include 'cts_mpr.h'
   :: rr,yy,lyy,loy,zz,z2,liox
  integer :: ii,ntwo,ione
  logical :: positive , r_gt_1 , y_lt_h
!
  rr = abs(xx)
  ntwo = iph/2*2
  ione = iph - ntwo
  positive = (ione.eq.0)
! 
  if     (rr.eq.R0P0) then
    rslt = pi2o6
  elseif (rr.eq.R1P0.and.positive) then
    rslt = C0P0
  else
    yy  = rr
    lyy = log(rr)
    if (.not.positive) yy = -yy
!
    r_gt_1 = (rr.gt.R1P0)
    if (r_gt_1) then
      yy   = R1P0/yy
      lyy  = -lyy
      ntwo = -ntwo
      ione = -ione
    endif
    loy = log(R1P0-yy) ! log(1-yy) is always real
!
    y_lt_h = (yy.lt.R5M1)
    if (y_lt_h) then
      zz = -loy ! log(1-yy) is real
    else
      zz = -lyy ! yy>0.5 => log(yy) is real
    endif
!
    z2 = zz*zz
    liox = rbern(nn)
    do ii=nn,4,-2
      liox = rbern(ii-2) + liox*z2/(ii*(ii+1))
    enddo
    liox = rbern(1) + liox*zz/3
    liox = zz + liox*z2/2
!
    rslt = mp_cx(liox,R0P0)
!
    if (y_lt_h) then
      rslt = pi2o6 - rslt - mp_cx(loy*lyy,loy*pi*ione)
    endif
!
    rslt = rslt + mp_cx( R0P0 , -loy*pi*ntwo)
!
    if (r_gt_1) rslt = -rslt - mp_cx(-lyy,iph*pi)**2/2
  endif
  end function
!
end module


module avh_olo_mp_loga2
!*******************************************************************
! log(xx)/(1-xx)  with  xx = log|xx| + imag*pi*iph
!*******************************************************************
  use avh_olo_mp_kinds
  use avh_olo_mp_units
  use avh_olo_mp_func
  implicit none
  private
  public :: init_loga2,loga2
  include 'cts_mpr.h'
   :: thrs
  integer :: ndigits=0
contains
!
  subroutine init_loga2(ndig)
  integer ,intent(in) :: ndig
  thrs= mp_eps
  if (ndigits.eq.ndig) return ;ndigits=ndig
  thrs = 10*thrs
  end subroutine
!
  function loga2(xx,iph) result(rslt)
  use avh_olo_mp_loga ,only : loga
  include 'cts_mpr.h'
   ,intent(in) :: xx
  integer      ,intent(in) :: iph
  include 'cts_mpc.h'
   :: rslt
  include 'cts_mpr.h'
   :: omx
  if (mod(iph,2).eq.0) then
    omx = R1P0-abs(xx)
  else
    omx = R1P0+abs(xx)
  endif
!
  if (iph.ne.0) then
    if (omx.eq.R0P0) then
      if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop loga2: ' &
        ,'1-xx,iph=',omx,iph
      rslt = C0P0
    else
      rslt = loga(xx,iph)/mp_cx(omx,R0P0)
    endif
  else
    if (abs(omx).lt.thrs) then
      rslt = mp_cx(-R1P0-omx/2,R0P0)
    else
      rslt = loga(xx,iph)/mp_cx(omx,R0P0)
    endif
  endif
  end function
!
end module


module avh_olo_mp_logc
!*******************************************************************
! Returns  log( |Re(xx)| + imag*Im(xx) ) + imag*pi*iph
!*******************************************************************
  use avh_olo_mp_kinds
  use avh_olo_mp_units
  use avh_olo_mp_func
  implicit none
  private
  public :: logc
contains
!
  function logc(xx) result(rslt)
  type(qmplx_type) ,intent(in) :: xx
  include 'cts_mpc.h'
   :: rslt
  if (xx%c.eq.C0P0) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop logc: xx%c =',xx%c
    rslt = C0P0
  else
    rslt = log(mp_cx(abs(mp_rl(xx%c)),mp_ig(xx%c)) ) &
         + ipi*xx%p
  endif
  end function
!
end module


module avh_olo_mp_li2c
!*******************************************************************
!                  /1    ln(1-zz*t)
! avh_olo_mp_li2c = - |  dt ---------- 
!                  /0        t
! with  zz = 1 - ( |Re(xx)| + imag*Im(xx) )*exp(imag*pi*iph)
! Examples:
!   In order to get the dilog of  1+imag  use  xx=1+imag, iph= 0
!   In order to get the dilog of  1-imag  use  xx=1-imag, iph= 0
!   In order to get the dilog of -1+imag  use  xx=1-imag, iph= 1
!   In order to get the dilog of -1-imag  use  xx=1+imag, iph=-1
! Add multiples of  2  to  iph  in order to get the result on
! different Riemann-sheets.
!*******************************************************************
  use avh_olo_mp_kinds
  use avh_olo_mp_func
  use avh_olo_mp_bern
  use avh_olo_mp_li2a
  implicit none
  private
  public :: init_li2c,li2c
  integer :: nn=18
  integer :: ndigits=0
contains
!
  subroutine init_li2c(ndig)
  integer ,intent(in) :: ndig
  if (ndigits.eq.ndig) return ;ndigits=ndig
  call init_li2a(ndigits)
  call init_bern(ndigits)
  if     (ndigits.lt.24) then
    nn = 18
  else
    nn = 36
  endif
  end subroutine

  function li2c(xx) result(rslt)
  type(qmplx_type) :: xx
  include 'cts_mpc.h'
   :: rslt ,yy,lyy,loy,zz,z2
  include 'cts_mpr.h'
   :: rex,imx
  integer :: ii,iyy
  logical :: x_gt_1 , y_lt_h
!
  rex = mp_rl(xx%c)
  imx = mp_ig(xx%c)
! 
  if (imx.eq.R0P0) then
    rslt = li2a(rex,xx%p)
  else
    rex = abs(rex)
!
    if (mod(xx%p,2).eq.0) then
      yy = mp_cx(rex,imx)
      iyy = xx%p
    else
      yy = mp_cx(-rex,-imx)
! Notice that  iyy=xx%p/2*2  does not deal correctly with the
! situation when  xx%p-xx%p/2*2 = sign(Im(xx%c)) . The following does:
      iyy = xx%p + nint(mp_sn(R1P0,imx))
    endif
!
    x_gt_1 = (abs(xx%c).gt.R1P0)
    if (x_gt_1) then
      yy = C1P0/yy
      iyy = -iyy
    endif
    lyy = log(yy)
    loy = log(C1P0-yy)
!
    y_lt_h = (mp_rl(yy).lt.R5M1)
    if (y_lt_h) then
      zz = -loy
    else
      zz = -lyy
    endif
!
    z2 = zz*zz
    rslt = cbern(nn)
    do ii=nn,4,-2
      rslt = cbern(ii-2) + rslt*z2/(ii*(ii+1))
    enddo
    rslt = cbern(1) + rslt*zz/3
    rslt = zz + rslt*z2/2
!
    if (y_lt_h) rslt = pi2o6 - rslt - loy*lyy
!
    rslt = rslt - loy*ipi*iyy
!
    if (x_gt_1) rslt = -rslt - (lyy + ipi*iyy)**2/2
  endif
  end function
!
end module


module avh_olo_mp_logc2
!*******************************************************************
! log(xx)/(1-xx)
! with  log(xx) = log( |Re(xx)| + imag*Im(xx) ) + imag*pi*iph
!*******************************************************************
  use avh_olo_mp_kinds
  use avh_olo_mp_units
  use avh_olo_mp_func
  implicit none
  private
  public :: init_logc2,logc2
  include 'cts_mpr.h'
   :: thrs
  integer :: ndigits=0
contains
!
  subroutine init_logc2(ndig)
  integer ,intent(in) :: ndig
  thrs= mp_eps
  if (ndigits.eq.ndig) return ;ndigits=ndig
  thrs = 10*thrs
  end subroutine
!
  function logc2(xx) result(rslt)
  use avh_olo_mp_logc ,only : logc
  type(qmplx_type) ,intent(in) :: xx
  include 'cts_mpc.h'
   :: rslt ,omx
  if (mod(xx%p,2).eq.0) then
    omx = mp_cx(1d0-abs(mp_rl(xx%c)),-mp_ig(xx%c))
  else
    omx = mp_cx(1d0+abs(mp_rl(xx%c)), mp_ig(xx%c))
  endif
  if (xx%p.ne.0) then
    if (omx.eq.C0P0) then
      if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop logc2: ' &
        ,'1-xx%c,xx%p=',omx,xx%p
      rslt = C0P0
    else
      rslt = logc(xx)/omx
    endif
  else
    if (abs(omx).lt.thrs) then
      rslt = -C1P0-omx/2
    else
      rslt = logc(xx)/omx
    endif
  endif
  end function
!
end module


module avh_olo_mp_li2c2
!*******************************************************************
! avh_olo_mp_li2c2 = ( li2(x1) - li2(x2) )/(x1%c-x2%c)
!
!                    /1    ln(1-zz*t)
! where  li2(x1) = - |  dt ----------
!                    /0        t
! with  zz = 1 - ( |Re(x1%c)| + imag*Im(x1%c) )*exp(imag*pi*x1%p)
! and similarly for li2(x2)
!*******************************************************************
  use avh_olo_mp_kinds
  use avh_olo_mp_units
  use avh_olo_mp_func
  use avh_olo_mp_li2c
  use avh_olo_mp_logc2
  implicit none
  private
  public :: init_li2c2,li2c2
  include 'cts_mpr.h'
   :: thrs1
  include 'cts_mpr.h'
   :: thrs
  integer      :: nmax=12
  integer :: ndigits=0
contains
!
  subroutine init_li2c2(ndig)
  integer ,intent(in) :: ndig
  thrs1= mp_eps
  thrs = 0.11_kindr2
  if (ndigits.eq.ndig) return ;ndigits=ndig
  call init_logc2(ndigits)
  call init_li2c(ndigits)
  if     (ndigits.lt.16) then
    thrs = 0.11_kindr2 ! double precision
    nmax = 12
  elseif (ndigits.lt.24) then
    thrs = 0.02_kindr2 ! guess
    nmax = 12
  else
    thrs = 0.008_kindr2 ! quadruple precision
    nmax = 12
  endif
  end subroutine
!
  function li2c2(x1,x2) result(rslt)
  type(qmplx_type) ,intent(in) :: x1,x2
  include 'cts_mpc.h'
   :: rslt
  include 'cts_mpc.h'
   :: x1r,x2r,delta,xx,xr,omx,del,hh,ff(0:20),zz
  integer :: ih,ii
!
  if (mod(x1%p,2).eq.0) then
    x1r = mp_cx( abs(mp_rl(x1%c)), mp_ig(x1%c))
  else
    x1r = mp_cx(-abs(mp_rl(x1%c)),-mp_ig(x1%c))
  endif     
  if (mod(x2%p,2).eq.0) then
    x2r = mp_cx( abs(mp_rl(x2%c)), mp_ig(x2%c))
  else
    x2r = mp_cx(-abs(mp_rl(x2%c)),-mp_ig(x2%c))
  endif
  delta = x1r-x2r
!
  if (x1%p.ne.x2%p) then
    if (delta.eq.C0P0) then
      if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop li2c2: ' &
        ,'x1%p,x2%p,delta=',x1%p,x2%p,delta
      rslt = C0P0
    else
      rslt = ( li2c(x1)-li2c(x2) )/delta
    endif
  else
    if (abs(delta/x1%c).gt.thrs) then
      rslt = ( li2c(x1)-li2c(x2) )/delta
    else
      xx  = x1%c
      xr  = x1r
      omx = C1P0-xr
      del = delta
      hh = C1P0-x2r
      if (abs(hh).gt.abs(omx)) then
        xx = x2%c
        xr = x2r
        omx = hh
        del = -delta
      endif
      if (abs(omx).lt.thrs1) then
        zz = -C1P0-omx/2-del/4
      else
        ih = x1%p - x1%p/2*2
        ff(0) = logc2(directly(xx,ih))
        hh = -C1P0
        do ii=1,nmax
          hh = -hh/xr
          ff(ii) = ( hh/ii + ff(ii-1) )/omx
        enddo
        zz = ff(nmax)/(nmax+1)
        do ii=nmax-1,0,-1
          zz = ff(ii)/(ii+1) - zz*del
        enddo
      endif
      ih = x1%p-ih
      if (ih.ne.0) then
        omx = C1P0-x1r
        zz = zz - ih*ipi*logc2(qonv((C1P0-x2r)/omx))/omx
      endif
      rslt = zz
    endif
  endif
  end function
!
end module


module avh_olo_mp_bub
  include 'cts_mprec.h'                                              
  use avh_olo_mp_kinds
  use avh_olo_mp_units
  implicit none
  private
  public :: init_bub ,tadp ,bub0 ,bub11
  integer ,parameter :: d=kindr2
  integer ,parameter :: ntrmmax=20
  include 'cts_dpr.h'
   ,parameter :: thrslistd(ntrmmax)=&
    (/5e-5_d,5e-3_d,0.05_d,0.10_d,0.15_d,0.20_d,0.30_d,0.40_d &
     ,0.50_d,0.60_d,0.65_d,0.68_d,0.72_d,0.74_d,0.76_d,0.78_d &
     ,0.80_d,0.82_d,0.83_d,0.84_d/)
  include 'cts_dpr.h'
   ,parameter :: thrslisth(ntrmmax)=&
    (/7e-8_d,5e-4_d,2e-3_d,1e-2_d,3e-2_d,6e-2_d,0.11_d,0.17_d &
     ,0.22_d,0.28_d,0.33_d,0.37_d,0.42_d,0.47_d,0.51_d,0.54_d &
     ,0.58_d,0.60_d,0.62_d,0.65_d/)
  include 'cts_dpr.h'
   ,parameter :: thrslistq(ntrmmax)=&
    (/1e-10_d,5e-5_d,1e-4_d,1e-3_d,7e-3_d,0.02_d,0.04_d,0.07_d &
      ,0.10_d,0.13_d,0.17_d,0.20_d,0.25_d,0.30_d,0.34_d,0.38_d &
      ,0.42_d,0.44_d,0.47_d,0.50_d/)
  include 'cts_dpr.h'
   :: thrs=0.07_d
  include 'cts_dpr.h'
   :: thrsexp=0.01_d
  include 'cts_dpr.h'
   :: thrslist(1:ntrmmax)=thrslistd(1:ntrmmax)
  integer      :: ntrm=11,nnexp=7
  include 'cts_dpc.h'
   :: aaexp(8)= 0._d
  integer :: ndigits=0
contains
!
  subroutine init_bub(ndig)
  integer ,intent(in) :: ndig
  integer :: ii
  if (ndigits.eq.ndig) return ;ndigits=ndig
  if     (ndigits.lt.16) then
    thrs = 0.07_kindr2    ! double precision,
    ntrm = 11             ! tested to suit also b11
    thrsexp = 0.01_kindr2 !
    nnexp = 7             !
    thrslist = thrslistd  ! double precision
  elseif (ndigits.lt.24) then
    thrs = 0.02_kindr2     ! guess
    ntrm = 11              !
    thrsexp = 0.001_kindr2 !
    nnexp = 7              !
    thrslist = thrslisth   !
  else
    thrs = 0.005_kindr2     ! quadruple precision, not tested
    ntrm = 11               !
    thrsexp = 0.0001_kindr2 !
    nnexp = 7               !
    thrslist = thrslistq    ! quadruple precision
  endif
  do ii=1,nnexp
    aaexp(ii) = C1P0/(ii*(ii+1))
  enddo
  end subroutine


  subroutine tadp( rslt ,mm ,amm ,rmu2 )
!*******************************************************************
! The 1-loop scalar 1-point function.
!*******************************************************************
  use avh_olo_mp_func
  use avh_olo_mp_logc ,only : logc
  include 'cts_mpc.h'
   ,intent(out) :: rslt(0:2)
  include 'cts_mpc.h'
   ,intent(in)  :: mm
  include 'cts_mpr.h'
   ,intent(in)  :: amm,rmu2
!
!  write(*,*) 'MESSAGE from OneLOop tadp: you are calling me' !CALLINGME
!
  rslt(2) = C0P0
  if (amm.eq.R0P0) then
    rslt(1) = C0P0
    rslt(0) = C0P0
  else
    rslt(1) = mm
    rslt(0) = mm - mm*logc( qonv(mm/rmu2,-1) )
  endif
  end subroutine


  subroutine bub0( rslt ,pp,m1i,m2i ,app,am1i,am2i ,rmu2 )
!*******************************************************************
! The 1-loop scalar 2-point function. Based on the formulas from
! A. Denner, Fortsch.Phys.41:307-420,1993 arXiv:0709.1075 [hep-ph]
!*******************************************************************
  use avh_olo_mp_func
  use avh_olo_mp_logc ,only: logc
  use avh_olo_mp_logc2 ,only: logc2
  include 'cts_mpc.h'
   ,intent(out) :: rslt(0:2)
  include 'cts_mpc.h'
   ,intent(in)  :: pp,m1i,m2i
  include 'cts_mpr.h'
   ,intent(in)  :: app,am1i,am2i,rmu2
  include 'cts_mpc.h'
   :: cc(0:ntrmmax),m1,m2,hh,aa,bb,rr,dd
  type(qmplx_type) :: qmm,qz1
  include 'cts_mpc.h'
   :: two
  include 'cts_mpr.h'
   :: am1,am2,tt
  integer :: ii
  two=C1P0*2.d0
!
!  write(*,*) 'MESSAGE from OneLOop bub0: you are calling me' !CALLINGME
!
  tt = max(am1i,am2i)
  if (am1i.lt.tt) then
    m1=m1i ;am1=am1i
    m2=m2i ;am2=am2i
  else
    m1=m2i ;am1=am2i
    m2=m1i ;am2=am1i
  endif
!
  rslt(2) = C0P0     
  rslt(1) = C1P0
!
  if (am2.eq.R0P0) then
    if (app.eq.R0P0) then
      rslt(1) = C0P0     
      rslt(0) = C0P0     
    else
      rslt(0) = two - logc(qonv(-pp/rmu2,-1))
    endif
  else!if(am2.ne.R0P0)
    tt = app/tt
    if (am1.eq.R0P0) then
      qmm = qonv(m2/rmu2,-1)
      if     (app.eq.R0P0) then
        rslt(0) = C1P0 - logc(qmm)
      elseif (pp.eq.m2) then
        rslt(0) = two - logc(qmm)
      elseif (tt.lt.R1P0) then
        hh = m2-pp
        rslt(0) = two + (hh/pp)*logc(qonv(hh/rmu2,-1)/qmm) - logc(qmm)
      else!if (tt.ge.R1P0) then
        hh = m2-pp
        rslt(0) = two - (m2/pp)*logc(qmm) + (hh/pp)*logc(qonv(hh/rmu2,-1))
      endif
    else!if(am1.ne.R0P0)
      if (app.eq.R0P0) then
         qz1 = qonv(m1/rmu2,-1)
         rslt(0) = C1P0 + logc2(qz1/qonv(m2/rmu2,-1)) - logc(qz1)
      else!if(pp.ne.C0P0)
        if     (tt.le.thrs) then
          call expans( cc ,m1,m2 ,am1,am2 ,rmu2 ,ntrm)
          rslt(0) = cc(ntrm)
          do ii=ntrm-1,0,-1
            rslt(0) = cc(ii) + pp*rslt(0)
          enddo
        elseif (tt.lt.R1P0) then
          hh = mysqrt(m1)
          bb = mysqrt(m2)
          aa = hh*bb ! sm1*sm2
          bb = hh/bb ! sm1/sm2
          hh = (m1+m2-pp)/aa
          dd = (m2-m1)**2 + ( pp - 2*(m1+m2) )*pp
          dd = mysqrt(dd)/aa
          call rfun0( rr ,dd ,hh )
          qz1 = qonv(bb,-1) ! sm1/sm2
          rslt(0) = two - logc(qonv(m2/rmu2,-1)) &
                        + logc(qz1)*two*m1/(aa*rr-m1) &
                        + logc2(qz1*qonv(rr,-1))*dd*aa/(aa*rr-m1+pp)
        else
          hh = mysqrt(m1)
          bb = mysqrt(m2)
          aa = hh*bb ! sm1*sm2
          bb = hh/bb ! sm1/sm2
          hh = (m1+m2-pp)/aa
          call rfun( rr,dd ,hh )
          rslt(0) = two - logc(qonv(aa/rmu2,-1)) &
                  + (logc(qonv(bb,-1))*(m2-m1) + logc(qonv(rr,-1))*dd*aa)/pp
        endif
!        call expans( cc ,m1,m2 ,am1,am2 ,rmu2 ,ntrm) !DEBUG
!        hh = cc(ntrm)                            !DEBUG
!        do ii=ntrm-1,0,-1                        !DEBUG
!          hh = cc(ii) + pp*hh                    !DEBUG
!        enddo                                    !DEBUG
!        write(*,'(a4,2d24.16)') 'exp:',hh        !DEBUG
      endif
    endif
  endif
  end subroutine


  subroutine bub11( b11,b00,b1,b0 ,pp,m0,m1 ,app,am0,am1 ,rmu2 )
!*******************************************************************
! Return the Passarino-Veltman functions b11,b00,b1,b0 , for
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
!*******************************************************************
  include 'cts_mpc.h'
   ,intent(out) :: b11(0:2),b00(0:2),b1(0:2),b0(0:2)
  include 'cts_mpc.h'
   ,intent(in)  :: pp,m0,m1
  include 'cts_mpr.h'
   ,intent(in)  :: app,am0,am1,rmu2
  include 'cts_mpc.h'
   :: a1(0:2),a0(0:2),ff,gg,c1,c2,cc(0:ntrmmax)
  include 'cts_mpr.h'
   :: rr,maxm
  integer :: ii
!
  maxm = max(am0,am1)
  if (maxm.eq.R0P0) then
    if (app.eq.R0P0) then
      b0  = C0P0
      b1  = C0P0
      b00 = C0P0
      b11 = C0P0
      return
    endif
    rr = R1P0+thrs
  else
    rr = app/maxm
  endif
!
  ff = pp - m1 + m0
  gg = m0 + m1 - pp/3
  b0(2)  = C0P0
  b1(2)  = C0P0
  b00(2) = C0P0
  b11(2) = C0P0
  b0(1)  = C1P0
  b1(1)  = -C1P0/2
  b00(1) = gg/4
  b11(1) = C1P0/3
  call tadp( a1 ,m0 ,am0 ,rmu2 )
  call tadp( a0 ,m1 ,am1 ,rmu2 )
!
  if (rr.le.thrs) then
!    write(*,*) 'expansion' !DEBUG
    call expans( cc ,m0,m1 ,am0,am1 ,rmu2 ,ntrm )
    c2 = cc(ntrm)
    do ii=ntrm-1,2,-1
      c2 = cc(ii) + pp*c2
    enddo
    c1 = cc(1) + pp*c2
    b0(0)  = cc(0) + pp*c1
    b1(0)  = -( cc(0) + ff*c1 )/2
    b00(0) = ( a0(0) + ff*b1(0) + 2*m0*b0(0) + gg )/6
    b11(0) = cc(0) + (ff+m0-m1)*cc(1) + ff*ff*c2 - m0*c1
    b11(0) = ( b11(0) + C1P0/6 )/3
  else
    call bub0( b0 ,pp,m0,m1 ,app,am0,am1 ,rmu2 )
    b1(0)  = ( a1(0) - a0(0) - ff*b0(0) )/(2*pp)
    b00(0) = ( a0(0) + ff*b1(0) + 2*m0*b0(0) + gg )/6
    b11(0) = ( a0(0) - 2*ff*b1(0) - m0*b0(0) - gg/2 )/(3*pp)
  endif
!
  end subroutine


  subroutine expans( cc ,m1i,m2i ,am1i,am2i ,rmu2 ,ntrm )
!*******************************************************************
! Returns the first 1+ntrm coefficients of the expansion in p^2 of
! the finite part of the 1-loop scalar 2-point function 
!*******************************************************************
  use avh_olo_mp_func
  use avh_olo_mp_logc ,only: logc
  integer         ,intent(in)  :: ntrm
  include 'cts_mpc.h'
   ,intent(out) :: cc(0:ntrm)
  include 'cts_mpc.h'
   ,intent(in)  :: m1i,m2i
  include 'cts_mpr.h'
   ,intent(in)  :: am1i,am2i,rmu2
  include 'cts_mpc.h'
   :: m1,m2,zz,oz,xx,logz,tt(ntrm)
  type(qmplx_type) :: qm1,qm2,qzz
  include 'cts_mpr.h'
   :: am1,am2
  integer :: ii
  include 'cts_mpr.h'
   ::rr
!
!  write(*,*) 'MESSAGE from OneLOop bub expans: you are calling me' !CALLINGME
!
  if (am1i.lt.am2i) then
    m1=m1i ;am1=am1i
    m2=m2i ;am2=am2i
  else
    m1=m2i ;am1=am2i
    m2=m1i ;am2=am1i
  endif
!
  if (am2.eq.R0P0) then
!
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop bub expans: ' &
      ,'m1=m2=0, returning 0'
    do ii=0,ntrm
      cc(ii) = C0P0
    enddo
!
  else
!
    qm1 = qonv(m1/rmu2,-1) 
    qm2 = qonv(m2/rmu2,-1)
    qzz = qm1/qm2
    if (mod(qzz%p,2).eq.0) then
      zz = qzz%c
    else
      zz = -qzz%c
    endif
!
    if (m1.eq.C0P0) then
      cc(0) = C1P0 - logc(qm2)
    else
      oz = C1P0-zz
      rr = abs(oz)
      if (rr.lt.thrsexp) then
        xx = aaexp(nnexp)
        do ii=nnexp-1,1,-1
          xx = aaexp(ii) + oz*xx
        enddo
        xx = oz*xx
      else
        logz = logc( qzz )
        xx = zz*logz + oz
        xx = xx/oz
      endif
      cc(0) = xx - logc(qm2)
    endif
!
    zz = C1P0-zz
    xx = C1P0
    call expans1(tt ,ntrm,zz)
    do ii=1,ntrm
      xx = xx*m2
      cc(ii) = tt(ii)/(ii*xx)
    enddo
!
  endif
  end subroutine


  subroutine expans1(tt ,ntrm,zz)
!*******************************************************************
! Returns  tt(n) = int( ( x*(1-x)/(1-zz*x) )^n , x=0..1 )
! for  n=1...ntrm  and  |zz|=<1
!
! Gives at least 2 correct digits (4 at quad.) for tt(ntrm),
! and increasingly more digits for tt(i<ntrm)
!
! Uses recursion on integrals of the type
!    int( x^m * (1-x)^n / (1-z*x)^n , x=0..1 )
! and
!    int( x^m * (1-x)^n / (1+y*x)^(n+2) , x=0..1 )
! where  y = z/(1-z)
! The latter integrals are related to the original ones via the
! substitution  x <- 1-x  followed by  x <- (1-x)/(1+y*x)
!*******************************************************************
  integer         ,intent(in)  :: ntrm
  include 'cts_mpc.h'
   ,intent(in)  :: zz
  include 'cts_mpc.h'
   ,intent(out) :: tt(ntrm)
  include 'cts_mpc.h'
   :: tu(ntrm),tv(ntrm) ,tt0,tu0,tv0,yy,y2,oy
  include 'cts_mpr.h'
   :: rr
  integer :: nn,ii,jj
!
  rr = mp_rl(zz)
  nn = ntrm
  if (nn.lt.1) nn = 1
  if (nn.gt.ntrmmax) then
    if (wunit.gt.0) write(wunit,*) 'WARNING from OneLOop bub expans1: ' &
      ,'ntrm =',nn,' > nmax =',ntrmmax,', using ntrm=nmax'
    nn = ntrmmax
    do ii=nn+1,ntrm
      tt(ii) = C0P0
    enddo
  endif
!
  if (zz.eq.C1P0) then
!    write(*,'(a16,i4)') 'simple expansion',nn !DEBUG
    do ii=1,nn
      tt(ii) = mp_cx(R1P0/(ii+1),R0P0)
    enddo
  elseif (rr.lt.(thrslist(nn)*R1P0)) then
! Backward recursion, number of correct decimals constant, so need
! full precision from the start
!    write(*,'(a8,i4,d24.16)') 'Backward',nn,rr !DEBUG
    call expans2(tt(nn),tv(nn-1),tu(nn-1) ,nn,zz)
    do ii=nn-1,2,-1
      jj = ii+1
      tt(ii  ) = 2*tv(ii) - (zz*tt(jj)*ii)/jj
      tu(ii-1) = (2+R1P0/ii)*tt(ii) - zz*tu(ii)
      tv(ii-1) = (C1P0-zz)*tu(ii-1) + zz*( 2*tt(ii) - zz*tu(ii) )
    enddo
    tt(1) = 2*tv(1) - zz*tt(2)/2
  else
! Foreward recursion, number of correct decimals decreases
!    write(*,'(a8,i4,d24.16)') 'Foreward',nn,rr !DEBUG
    yy = zz/(C1P0-zz)
    y2 = yy*yy
    oy = C1P0+yy ! C1P0/(C1P0-zz)
    tt0 = C1P0-zz ! 1/(1+y)
    tu0 = ( oy*log(oy)-yy )/( y2*oy )
    tv0 = tt0/2
    tt(1) = ( tt0-2*tu0 )/( 2*yy )
    tv(1) = ( tv0 - 3*tt(1) )/( 3*yy )
    tu(1) = ( oy*tu0 - 2*yy*tt(1) - tv0 )/y2
    do ii=2,nn
      jj = ii-1
      tt(ii) = ii*( tt(jj)-2*tu(jj) )/( (ii+1)*yy )
      tv(ii) = ( ii*tv(jj) - (ii+ii+1)*tt(ii) )/( (ii+2)*yy )
      tu(ii) = ( oy*tu(jj) - 2*yy*tt(ii) - tv(jj) )/y2
    enddo
    yy = oy
    do ii=1,nn
      oy = oy*yy
      tt(ii) = oy*tt(ii)
    enddo
  endif
  end subroutine


  subroutine expans2(ff,fa,fb ,nn_in,zz)
!*******************************************************************
! ff = Beta(nn+1,nn+1) * 2F1(nn  ,nn+1;2*nn+2;zz)
! fa = Beta(nn+1,nn  ) * 2F1(nn-1,nn+1;2*nn+1;zz)
! fb = Beta(nn  ,nn+1) * 2F1(nn  ,nn  ;2*nn+1;zz)
!*******************************************************************
  include 'cts_mpc.h'
   ,intent(out) :: ff,fa,fb
  include 'cts_mpc.h'
   ,intent(in)  :: zz
  integer         ,intent(in)  :: nn_in
  integer ,parameter :: nmax=100
  integer :: aa,bb,cc,ii,ntrm
  include 'cts_mpc.h'
   ,save :: qq(0:nmax),qa(0:nmax),qb(0:nmax),gg,ga
  include 'cts_mpr.h'
   ,save :: logprec
  integer ,save :: nn=0
  include 'cts_mpr.h'
   :: ac0,bc0,ai,bi,ci,ac,bc
   logprec=-36.0d0
  if (nn.ne.nn_in) then
    nn = nn_in
    aa = nn-1
    bb = nn
    cc = nn+nn+1
    qq(0) = C1P0
    qa(0) = C1P0
    qb(0) = C1P0
    ac0 = mp_rl(aa*C1P0)/mp_rl(cc*C1P0)
    bc0 = mp_rl(bb*C1P0)/mp_rl(cc*C1P0)
    ntrm = nmax
    do ii=1,ntrm
      ai = mp_rl(aa*C1P0+ii*C1P0)
      bi = mp_rl(bb*C1P0+ii*C1P0)
      ci = mp_rl(cc*C1P0+ii*C1P0)
      ac = ai/ci
      bc = bi/ci
      qq(ii) = qq(ii-1) * ai*bc  / ii
      qa(ii) = qa(ii-1) * ac0*bi / ii
      qb(ii) = qb(ii-1) * ai*bc0 / ii
      ac0 = ac
      bc0 = bc
    enddo
    ai = R1P0
    do ii=2,nn-1
      ai = ai*ii
    enddo
    ci = ai
    cc = nn+nn
    do ii=nn,cc
      ci = ci*ii
    enddo
    bi = ai*nn
    gg = bi*bi/(ci*(cc+1))
    ga = ai*bi/ci
    logprec = log(mp_eps)
  endif
!
  ai = abs(zz)
  if (ai.gt.R0P0) then
    ntrm = 1 + int(logprec/log(ai))
  else
    ntrm = 1
  endif
  if (ntrm.gt.nmax) then
    if (wunit.gt.0) write(wunit,*) 'WARNING from OneLOop bub expans2: ' &
      ,'ntrm =',ntrm,' > nmax =',nmax,', putting ntrm=nmax'
    ntrm = nmax
  endif
!
  ff = qq(ntrm)
  fa = qa(ntrm)
  fb = qb(ntrm)
  do ii=ntrm-1,0,-1
    ff = qq(ii) + ff*zz
    fa = qa(ii) + fa*zz
    fb = qb(ii) + fb*zz
  enddo
  ff = gg*ff
  fa = ga*fa
  fb = ga*fb
  end subroutine
!
end module


module avh_olo_mp_tri
  use avh_olo_mp_kinds
  use avh_olo_mp_units
  use avh_olo_mp_func
  implicit none
  private
  public :: tria0,tria1,tria2,tria3,tria4,trif0,trif1,trif2,trif3 &
           ,permtable,casetable,base
  integer ,parameter :: permtable(3,0:7)=reshape((/ &
       1,2,3 &! 0, 0 masses non-zero, no permutation
      ,1,2,3 &! 1, 1 mass non-zero,   no permutation
      ,3,1,2 &! 2, 1 mass non-zero,   1 cyclic permutation
      ,1,2,3 &! 3, 2 masses non-zero, no permutation
      ,2,3,1 &! 4, 1 mass non-zero,   2 cyclic permutations
      ,2,3,1 &! 5, 2 masses non-zero, 2 cyclic permutations
      ,3,1,2 &! 6, 2 masses non-zero, 1 cyclic permutation
      ,1,2,3 &! 7, 3 masses non-zero, no permutation
      /) ,(/3,8/))                     ! 0,1,2,3,4,5,6,7
  integer ,parameter :: casetable(0:7)=(/0,1,1,2,1,2,2,3/)
  integer ,parameter :: base(3)=(/4,2,1/)

contains

   subroutine tria4( rslt ,cpp,cm2,cm3 ,rmu2 )
!*******************************************************************
! calculates
!               C   /             d^(Dim)q
!            ------ | ----------------------------------
!            i*pi^2 / q^2 [(q+k1)^2-m2] [(q+k1+k2)^2-m3]
!
! with  k1^2=m2, k2^2=pp, (k1+k2)^2=m3.
! m2,m3 should NOT be identically 0d0.
!*******************************************************************
   use avh_olo_mp_logc ,only: logc
   use avh_olo_mp_logc2 ,only: logc2
   use avh_olo_mp_li2c2 ,only: li2c2
   include 'cts_mpc.h'
    ,intent(out) :: rslt(0:2)
   include 'cts_mpc.h'
    ,intent(in)  :: cm2,cm3,cpp
   include 'cts_mpr.h'
    ,intent(in)  :: rmu2
   type(qmplx_type) :: q23,qm3,q32
   include 'cts_mpc.h'
    :: sm2,sm3,k23,r23,d23,cc
!
!  write(*,*) 'MESSAGE from OneLOop tria4: you are calling me' !CALLINGME
!
   sm2 = mysqrt(cm2)
   sm3 = mysqrt(cm3)
   k23 = (cm2+cm3-cpp)/(sm2*sm3)
   call rfun( r23,d23, k23 )
   if (r23.eq.-C1P0) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop tria4: ' &
       ,'threshold singularity, returning 0'
     rslt = C0P0
     return
   endif
   q23 = qonv(r23,-1)
   qm3 = qonv(cm3/rmu2,-1)
   q32 = qonv(sm3)/qonv(sm2)
!
   rslt(2) = C0P0
   cc = logc2(q23) * r23/(C1P0+r23)/(sm2*sm3)
   rslt(1) = -cc
   rslt(0) = cc*( logc(qm3) - logc(q23) ) &
           - li2c2(q32*q23,q32/q23) / cm2 &
           + li2c2(q23*q23,qonv(C1P0)) * r23/(sm2*sm3)
   end subroutine


   subroutine tria3( rslt ,cp2,cp3,cm3 ,rmu2 )
!*******************************************************************
! calculates
!               C   /          d^(Dim)q
!            ------ | -----------------------------
!            i*pi^2 / q^2 (q+k1)^2 [(q+k1+k2)^2-m3]
!
! with  p2=k2^2, p3=(k1+k2)^2.
! mm should NOT be identically 0d0,
! and p2 NOR p3 should be identical to mm. 
!*******************************************************************
   use avh_olo_mp_logc ,only: logc
   use avh_olo_mp_logc2 ,only: logc2
   use avh_olo_mp_li2c2 ,only: li2c2
   include 'cts_mpc.h'
    ,intent(out) :: rslt(0:2)
   include 'cts_mpc.h'
    ,intent(in)  :: cp2,cp3,cm3
   include 'cts_mpr.h'
    ,intent(in)  :: rmu2
   type(qmplx_type) :: q13,q23,qm3,x1,x2
   include 'cts_mpc.h'
    :: r13,r23
!
!  write(*,*) 'MESSAGE from OneLOop tria3: you are calling me' !CALLINGME
!
   r13 = cm3-cp3
   r23 = cm3-cp2
   q13 = qonv(r13,-1)
   q23 = qonv(r23,-1)
   qm3 = qonv(cm3,-1)
   x1 = q23/qm3
   x2 = q13/qm3
   rslt(2) = C0P0
   rslt(1) = -logc2( q23/q13 )/r13
   rslt(0) = -li2c2( x1,x2 )/cm3 &
           - rslt(1)*( logc(x1*x2)+logc(qm3/rmu2) )
   end subroutine


   subroutine tria2( rslt ,cp3,cm3 ,rmu2 )
!*******************************************************************
! calculates
!               C   /          d^(Dim)q
!            ------ | -----------------------------
!            i*pi^2 / q^2 (q+k1)^2 [(q+k1+k2)^2-m3]
!
! with  k1^2 = 0 , k2^2 = m3  and  (k1+k2)^2 = p3.
! mm should NOT be identically 0d0,
! and pp should NOT be identical to mm. 
!*******************************************************************
   use avh_olo_mp_logc ,only: logc
   use avh_olo_mp_li2c ,only: li2c
   include 'cts_mpc.h'
    ,intent(out) :: rslt(0:2)
   include 'cts_mpc.h'
    ,intent(in)  :: cp3,cm3
   include 'cts_mpr.h'
    ,intent(in)  :: rmu2
   include 'cts_mpc.h'
     :: const
   include 'cts_mpc.h'
     :: half
   type(qmplx_type) :: q13,qm3,qxx
   include 'cts_mpc.h'
    :: r13,logm,z2,z1,z0,cc
   const= C1P0*TWOPI*TWOPI/9
   half = C1P0/2
!
!  write(*,*) 'MESSAGE from OneLOop tria2: you are calling me' !CALLINGME
!
   r13 = cm3-cp3
   q13 = qonv(r13,-1)
   qm3 = qonv(cm3,-1)
   logm = logc( qm3/rmu2 )
   qxx = qm3/q13
   z2 = half
   z1 = logc(qxx)
   z0 = const + z1*z1/2 - li2c(qxx)
   cc = -C1P0/r13
   rslt(2) = cc*z2
   rslt(1) = cc*(z1 - z2*logm)
   rslt(0) = cc*(z0 + (z2*logm/2-z1)*logm)
   end subroutine


   subroutine tria1( rslt ,cm3 ,rmu2 )
!*******************************************************************
! calculates
!               C   /          d^(Dim)q
!            ------ | -----------------------------
!            i*pi^2 / q^2 (q+k1)^2 [(q+k1+k2)^2-m3]
!
! with  k1^2 = (k1+k2)^2 = m3.
! mm should NOT be identically 0d0.
!*******************************************************************
   use avh_olo_mp_logc ,only: logc
   include 'cts_mpc.h'
    ,intent(out) :: rslt(0:2)
   include 'cts_mpc.h'
    ,intent(in)  :: cm3
   include 'cts_mpr.h'
    ,intent(in)  :: rmu2
   include 'cts_mpc.h'
    :: zm
!
!  write(*,*) 'MESSAGE from OneLOop tria1: you are calling me' !CALLINGME
!
   zm = C1P0/(2*cm3)
   rslt(2) = C0P0
   rslt(1) = -zm
   rslt(0) = zm*( 2*C1P0 + logc(qonv(cm3/rmu2,-1)) )
   end subroutine


   subroutine tria0( rslt ,cp ,ap ,rmu2 )
!*******************************************************************
! calculates
!               C   /         d^(Dim)q
!            ------ | ------------------------
!            i*pi^2 / q^2 (q+k1)^2 (q+k1+k2)^2
!
! with  Dim = 4-2*eps
!         C = pi^eps * mu^(2*eps) * exp(gamma_Euler*eps)
!
! input:  p1 = k1^2,  p2 = k2^2,  p3 = k3^2
! output: rslt(0) = eps^0   -coefficient
!         rslt(1) = eps^(-1)-coefficient
!         rslt(2) = eps^(-2)-coefficient
!
! If any of these numbers is IDENTICALLY 0d0, the corresponding
! IR-singular case is returned.
!*******************************************************************
   use avh_olo_mp_loga ,only: loga
   use avh_olo_mp_loga2 ,only: loga2
   include 'cts_mpc.h'
    ,intent(out) :: rslt(0:2)
   include 'cts_mpc.h'
    ,intent(in)  :: cp(3)
   include 'cts_mpr.h'
    ,intent(in)  :: ap(3),rmu2
   include 'cts_mpr.h'
    :: pp(3),rp1,rp2,rp3
   include 'cts_mpc.h'
    :: const
   include 'cts_mpc.h'
    :: log2,log3
   integer :: icase,i1,i2,i3
   const= C1P0*TWOPI*TWOPI/48
!
   pp(1)=mp_rl(cp(1))
   pp(2)=mp_rl(cp(2))
   pp(3)=mp_rl(cp(3))
!
   icase = 0
   if (ap(1).gt.R0P0) icase = icase + base(1)
   if (ap(2).gt.R0P0) icase = icase + base(2)
   if (ap(3).gt.R0P0) icase = icase + base(3)
   rp1 = pp(permtable(1,icase))
   rp2 = pp(permtable(2,icase))
   rp3 = pp(permtable(3,icase))
   icase  = casetable(  icase)
!
   i1=0 ;if (-rp1.lt.R0P0) i1=-1
   i2=0 ;if (-rp2.lt.R0P0) i2=-1
   i3=0 ;if (-rp3.lt.R0P0) i3=-1
!
   if     (icase.eq.0) then
! 0 masses non-zero
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop tria0: ' &
       ,'all external masses equal zero, returning 0'
     rslt = C0P0
   elseif (icase.eq.1) then
! 1 mass non-zero
!  write(*,*) 'MESSAGE from OneLOop tria0 1: you are calling me' !CALLINGME
    log3 = loga( -rp3/rmu2 , i3 )
    rslt(2) = mp_cx( R1P0/rp3 ,R0P0)
    rslt(1) = -log3/rp3
    rslt(0) = ( log3**2/2 - const )/rp3
  elseif (icase.eq.2) then
! 2 masses non-zero
!  write(*,*) 'MESSAGE from OneLOop tria0 2: you are calling me' !CALLINGME
    log2 = loga( -rp2/rmu2 ,i2 )
    log3 = loga( -rp3/rmu2 ,i3 )
    rslt(2) = C0P0
    rslt(1) = loga2( rp3/rp2 ,i3-i2 )/rp2
    rslt(0) = -rslt(1)*(log3+log2)/2
  elseif (icase.eq.3) then
! 3 masses non-zero
    call trif0( rslt ,cp(1),cp(2),cp(3) )
  endif
  end subroutine


   subroutine trif0( rslt ,p1,p2,p3 )
!*******************************************************************
! Finite 1-loop scalar 3-point function with all internal masses
! equal zero. Obtained from the formulas for 4-point functions in
! A. Denner, U. Nierste, R. Scharf, Nucl.Phys.B367(1991)637-656
! by sending one internal mass to infinity.
!*******************************************************************
   use avh_olo_mp_logc ,only: logc
   use avh_olo_mp_logc2 ,only: logc2
   use avh_olo_mp_li2c2 ,only: li2c2
   include 'cts_mpc.h'
    ,intent(out) :: rslt(0:2)
   include 'cts_mpc.h'
    ,intent(in)  :: p1,p2,p3
   type(qmplx_type) :: q23,q24,q34,qx1,qx2
   include 'cts_mpc.h'
    :: r23,r24,r34,aa,bb,cc,dd,x1,x2
   include 'cts_mpr.h'
    :: hh
!
!  write(*,*) 'MESSAGE from OneLOop trif0: you are calling me' !CALLINGME
!
   r23 = -p1
   r24 = -p3
   r34 = -p2
!
   aa = r34*r24
   bb = r24 + r34 - r23
   cc = C1P0
   hh = mp_rl(r23)
   dd = mysqrt( bb*bb - 4*aa*cc , -mp_rl(aa)*hh )
   call solabc( x1,x2,dd ,aa,bb,cc ,1 )
   x1 = -x1
   x2 = -x2
!
   qx1 = qonv(x1, hh)
   qx2 = qonv(x2,-hh)
   q23 = qonv(r23,-1)
   q24 = qonv(r24,-1)
   q34 = qonv(r34,-1)
!
   rslt = C0P0
!
   rslt(0) = li2c2( qx1*q34 ,qx2*q34 )*r34 &
           + li2c2( qx1*q24 ,qx2*q24 )*r24 &
           - logc2( qx1/qx2 )*logc( qx1*qx2 )/(x2*2) &
           - logc2( qx1/qx2 )*logc( q23 )/x2
!
   rslt(0) = rslt(0)/aa
   end subroutine


   subroutine trif1( rslt ,p1i,p2i,p3i ,m3i )
!*******************************************************************
! Finite 1-loop scalar 3-point function with one internal masses
! non-zero. Obtained from the formulas for 4-point functions in
! A. Denner, U. Nierste, R. Scharf, Nucl.Phys.B367(1991)637-656
! by sending one internal mass to infinity.
!*******************************************************************
   use avh_olo_mp_logc ,only: logc
   use avh_olo_mp_logc2 ,only: logc2
   use avh_olo_mp_li2c2 ,only: li2c2
   include 'cts_mpc.h'
    ,intent(out) :: rslt(0:2)
   include 'cts_mpc.h'
    ,intent(in)  :: p1i,p2i,p3i ,m3i 
   type(qmplx_type) :: q23,q24,q34,qm4,qx1,qx2,qss
   include 'cts_mpc.h'
    :: p2,p3,p4,p12,p23,m4,sm2,sm3,sm4 &
                     ,aa,bb,cc,dd,x1,x2,r23,r24,r34
   include 'cts_mpr.h'
    :: mhh
!
!  write(*,*) 'MESSAGE from OneLOop trif1: you are calling me' !CALLINGME
!
!   p1 = nul
   p2 = p1i
   p3 = p2i
   p4 = p3i
   p12 = p1i
   p23 = p3i
!   m1 = infinite
!   m2 = m1i = C0P0
!   m3 = m2i = C0P0
   m4 = m3i
!
   sm4 = mysqrt(m4)
   mhh = abs(sm4)
   sm3 = mp_cx(mhh,R0P0)
   sm2 = sm3
!
   r24 = C0P0
   r34 = C0P0
                  r23 = (   -p2 *oieps )/(sm2*sm3)
   if (m4.ne.p23) r24 = ( m4-p23*oieps )/(sm2*sm4)
   if (m4.ne.p3 ) r34 = ( m4-p3 *oieps )/(sm3*sm4)     
!
   aa = r34*r24 - r23
!
   if (aa.eq.C0P0) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop trif1: ' &
       ,'threshold singularity, returning 0'
     rslt = C0P0
     return
   endif
!
   bb = r24/sm3 + r34/sm2 - r23/sm4
   cc = C1P0/(sm2*sm3)
!   hh = real(r23)
!   dd = mysqrt( bb*bb - 4*aa*cc , -real(aa)*hh )
   call solabc( x1,x2,dd ,aa,bb,cc ,0 )
   x1 = -x1
   x2 = -x2
!
   qx1 = qonv(x1 ,1) ! x1 SHOULD HAVE im. part
   qx2 = qonv(x2 ,1) ! x2 SHOULD HAVE im. part
   q23 = qonv(r23,-1)
   q24 = qonv(r24,-1)
   q34 = qonv(r34,-1)
   qm4 = qonv(sm4,-1)
!
   rslt = C0P0
!
   rslt(0) = -logc2( qx1/qx2 )*logc( qx1*qx2/(qm4*qm4) )/(x2*2) &
             -li2c2( qx1*qm4 ,qx2*qm4 )*sm4
!
   if (r34.ne.C0P0) then
     qss = q34*mhh
     rslt(0) = rslt(0) + li2c2( qx1*qss ,qx2*qss )*r34*sm3
   endif
!
   if (r24.ne.C0P0) then
     qss = q24*mhh
     rslt(0) = rslt(0) + li2c2( qx1*qss ,qx2*qss )*r24*sm2
   endif
!
   rslt(0) = rslt(0) - logc2( qx1/qx2 )*logc( q23*(mhh*mhh) )/x2
!
   rslt(0) = rslt(0)/(aa*sm2*sm3*sm4)
   end subroutine


   subroutine trif2( rslt ,p1i,p2i,p3i ,m2i,m3i )
!*******************************************************************
! Finite 1-loop scalar 3-point function with two internal masses
! non-zero. Obtained from the formulas for 4-point functions in
! A. Denner, U. Nierste, R. Scharf, Nucl.Phys.B367(1991)637-656
! by sending one internal mass to infinity.
!*******************************************************************
   use avh_olo_mp_logc ,only: logc
   use avh_olo_mp_logc2 ,only: logc2
   use avh_olo_mp_li2c2 ,only: li2c2
   include 'cts_mpc.h'
    ,intent(out) :: rslt(0:2)
   include 'cts_mpc.h'
    ,intent(in)  :: p1i,p2i,p3i ,m2i,m3i
   type(qmplx_type) :: q23,q34,q24,qm2,qm3,qm4,qx1,qx2,qss,qy1,qy2
   include 'cts_mpc.h'
    :: p2,p3,p23,m2,m4,sm2,sm3,sm4,aa,bb,cc,dd,x1,x2 &
                     ,r23,k24,r34,r24,d24
!
!  write(*,*) 'MESSAGE from OneLOop trif2: you are calling me' !CALLINGME
!
!   p1 = nul
   p2 = p3i
   p3 = p1i
!   p4 = p2i
!   p12 = p3i
   p23 = p2i
!   m1 = infinite
   m2 = m3i
!   m3 = m1i = C0P0
   m4 = m2i
!
!   sm1 = infinite
   sm2 = mysqrt(m2)
   sm3 = mp_cx(abs(sm2),R0P0) !mysqrt(m3)
   sm4 = mysqrt(m4)
!
   r23 = C0P0
   k24 = C0P0
   r34 = C0P0
   if (m2   .ne.p2 ) r23 = (    m2-p2 *oieps )/(sm2*sm3) ! p2
   if (m2+m4.ne.p23) k24 = ( m2+m4-p23*oieps )/(sm2*sm4) ! p2+p3
   if (m4   .ne.p3 ) r34 = (    m4-p3 *oieps )/(sm3*sm4) ! p3
!
   call rfun( r24,d24 ,k24 )
!
   aa = r34/r24 - r23
!
   if (aa.eq.C0P0) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop trif2: ' &
       ,'threshold singularity, returning 0'
     rslt = C0P0
     return
   endif
!
   bb = -d24/sm3 + r34/sm2 - r23/sm4
   cc = (sm4/sm2 - r24)/(sm3*sm4)
!   hh = dreal(r23 - r24*r34)
!   dd = mysqrt( bb*bb - 4*aa*cc , -dreal(aa)*hh )
   call solabc(x1,x2,dd ,aa,bb,cc ,0)
   x1 = -x1
   x2 = -x2
!
   qx1 = qonv(x1 ,1 ) ! x1 SHOULD HAVE im. part
   qx2 = qonv(x2 ,1 ) ! x2 SHOULD HAVE im. part
   q23 = qonv(r23,-1)
   q24 = qonv(r24,-1)
   q34 = qonv(r34,-1)
   qm2 = qonv(sm2,-1)
   qm3 = qonv(sm3,-1)
   qm4 = qonv(sm4,-1)
!
   rslt = C0P0
!
   qy1 = qx1/q24
   qy2 = qx2/q24
!
   rslt(0) = li2c2( qy1*qm2 ,qy2*qm2 )/r24*sm2
!
   if (x2.ne.C0P0) then ! better to put a threshold on cc 
     rslt(0) = rslt(0) + ( logc2( qy1/qy2 )*logc( qy1*qy2/(qm2*qm2) ) &
                          -logc2( qx1/qx2 )*logc( qx1*qx2/(qm4*qm4) ) )/(x2*2)
   endif
!
   rslt(0) = rslt(0) - li2c2( qx1*qm4 ,qx2*qm4 )*sm4
!
   if (r23.ne.C0P0) then
     qss = q23*qm3/q24
     rslt(0) = rslt(0) - li2c2( qx1*qss ,qx2*qss )*r23*sm3/r24
   endif
!
   if (r34.ne.C0P0) then
     qss = q34*qm3
     rslt(0) = rslt(0) + li2c2( qx1*qss ,qx2*qss )*r34*sm3
   endif
!
   rslt(0) = rslt(0)/(aa*sm2*sm3*sm4)
   end subroutine


   subroutine trif3( rslt ,p1i,p2i,p3i ,m1i,m2i,m3i )
!*******************************************************************
! Finite 1-loop scalar 3-point function with all internal masses
! non-zero. Obtained from the formulas for 4-point functions in
! A. Denner, U. Nierste, R. Scharf, Nucl.Phys.B367(1991)637-656
! by sending one internal mass to infinity.
!*******************************************************************
   use avh_olo_mp_logc ,only: logc
   use avh_olo_mp_logc2 ,only: logc2
   use avh_olo_mp_li2c2 ,only: li2c2
   include 'cts_mpc.h'
    ,intent(out) :: rslt(0:2)
   include 'cts_mpc.h'
    ,intent(in)  :: p1i,p2i,p3i,m1i,m2i,m3i
   type(qmplx_type) :: q12,q13,q23,qm1,qm2,qm3,qx1,qx2,qz1,qz2,qtt
   include 'cts_mpc.h'
    :: p1,p2,p3,m1,m2,m3,sm1,sm2,sm3,aa,bb,cc,dd,x1,x2 &
                     ,k12,k13,k23,r12,r13,r23,d12,d13,d23 
   include 'cts_mpr.h'
    :: h1,h2,h3
!
!  write(*,*) 'MESSAGE from OneLOop trif3: you are calling me' !CALLINGME
!
   h1 = -mp_ig(m1i)
   h2 = -mp_ig(m2i)
   h3 = -mp_ig(m3i)
   if (h2.ge.h1.and.h2.ge.h3) then
     p1=p3i ;p2=p1i ;p3=p2i ;m1=m3i ;m2=m1i ;m3=m2i
   else
     p1=p1i ;p2=p2i ;p3=p3i ;m1=m1i ;m2=m2i ;m3=m3i
   endif
!
   sm1 = mysqrt(m1)
   sm2 = mysqrt(m2)
   sm3 = mysqrt(m3)
!
   k12 = C0P0
   k13 = C0P0
   k23 = C0P0
   if (m1+m2.ne.p1) k12 = ( m1+m2-p1*oieps )/(sm1*sm2) ! p1
   if (m1+m3.ne.p3) k13 = ( m1+m3-p3*oieps )/(sm1*sm3) ! p1+p2 => p12
   if (m2+m3.ne.p2) k23 = ( m2+m3-p2*oieps )/(sm2*sm3) ! p2
!
   call rfun( r12,d12 ,k12 )
   call rfun( r13,d13 ,k13 )
   call rfun( r23,d23 ,k23 )
!
   aa = sm2/sm3 - k23 + r13*(k12 - sm2/sm1)
!
   if (aa.eq.C0P0) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop trif3: ' &
       ,'threshold singularity, returning 0'
     rslt = C0P0
     return
   endif
!
   bb = d13/sm2 + k12/sm3 - k23/sm1
   cc = ( sm1/sm3 - C1P0/r13 )/(sm1*sm2)
!   hh = dreal( (r13-sm1/sm3)/(sm1*sm2) )
!   dd = mysqrt( bb*bb - 4*aa*cc , -dreal(aa)*hh )
   call solabc( x1,x2,dd ,aa,bb,cc ,0 )
   x1 = -x1
   x2 = -x2
!
   qx1 = qonv(x1 ,1) ! x1 SHOULD HAVE im. part
   qx2 = qonv(x2 ,1) ! x2 SHOULD HAVE im. part
   q12 = qonv(r12,-1)
   q13 = qonv(r13,-1)
   q23 = qonv(r23,-1)
   qm1 = qonv(sm1,-1)
   qm2 = qonv(sm2,-1)
   qm3 = qonv(sm3,-1)
!
   rslt = C0P0
!
   qz1 = qx1*qm2
   qz2 = qx2*qm2
   rslt(0) = rslt(0) + ( li2c2( qz1*q12 ,qz2*q12 )*r12 &
                        +li2c2( qz1/q12 ,qz2/q12 )/r12 )*sm2
   qtt = q13*qm2
   qz1 = qx1*qtt
   qz2 = qx2*qtt
   rslt(0) = rslt(0) - ( li2c2( qz1*q23 ,qz2*q23 )*r23 &
                        +li2c2( qz1/q23 ,qz2/q23 )/r23 )*r13*sm2
   qz1 = qx1*q13
   qz2 = qx2*q13
   rslt(0) = rslt(0) + li2c2( qz1*qm3 ,qz2*qm3 )*r13*sm3 &
                     - li2c2( qx1*qm1 ,qx2*qm1 )*sm1
   if (x2.ne.C0P0) then
     rslt(0) = rslt(0) + ( logc2( qz1/qz2 )*logc( qz1*qz2/(qm3*qm3) ) &
                          -logc2( qx1/qx2 )*logc( qx1*qx2/(qm1*qm1) ) )/(x2*2)
   endif
!
   rslt(0) = rslt(0)/(aa*sm1*sm2*sm3)
   end subroutine

end module


module avh_olo_mp_box
  use avh_olo_mp_kinds
  use avh_olo_mp_units
  use avh_olo_mp_func
  implicit none
  private
  public :: box00,box03,box05,box06,box07,box08,box09,box10,box11,box12 &
           ,box13,box14,box15,box16,boxf1,boxf2,boxf3,boxf5,boxf4 &
           ,permtable,casetable,base
  integer ,parameter ::  permtable(6,0:15)=reshape((/ &
     1,2,3,4 ,5,6 &! 0, 0 masses non-zero,           no perm
    ,1,2,3,4 ,5,6 &! 1, 1 mass non-zero,             no perm
    ,4,1,2,3 ,6,5 &! 2, 1 mass non-zero,             1 cyclic perm
    ,1,2,3,4 ,5,6 &! 3, 2 neighbour masses non-zero, no perm
    ,3,4,1,2 ,5,6 &! 4, 1 mass   non-zero,           2 cyclic perm's
    ,1,2,3,4 ,5,6 &! 5, 2 opposite masses non-zero,  no perm
    ,4,1,2,3 ,6,5 &! 6, 2 neighbour masses non-zero, 1 cyclic perm
    ,1,2,3,4 ,5,6 &! 7, 3 masses non-zero,           no perm
    ,2,3,4,1 ,6,5 &! 8, 1 mass   non-zero,           3 cyclic perm's
    ,2,3,4,1 ,6,5 &! 9, 2 neighbour masses non-zero, 3 cyclic perm's
    ,4,1,2,3 ,6,5 &!10, 2 opposite masses non-zero,  1 cyclic perm
    ,2,3,4,1 ,6,5 &!11, 3 masses non-zero,           3 cyclic perm's
    ,3,4,1,2 ,5,6 &!12, 2 neighbour masses non-zero, 2 cyclic perm's
    ,3,4,1,2 ,5,6 &!13, 3 masses non-zero,           2 cyclic perm's
    ,4,1,2,3 ,6,5 &!14, 3 masses non-zero,           1 cyclic perm
    ,1,2,3,4 ,5,6 &!15, 4 masses non-zero,           no perm
    /),(/6,16/)) !          0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
  integer ,parameter :: casetable(0:15)= &
                          (/0,1,1,2,1,5,2,3,1,2, 5, 3, 2, 3, 3, 4/)
  integer ,parameter :: base(4)=(/8,4,2,1/)
contains

   subroutine box16( rslt ,p2,p3,p12,p23 ,m2,m3,m4 ,rmu )
!*******************************************************************
! calculates
!
!    C   /                     d^(Dim)q
! ------ | ------------------------------------------------------
! i*pi^2 / q^2 [(q+k1)^2-m2] [(q+k1+k2)^2-m3] [(q+k1+k2+k3)^2-m4]
!
! with  k1^2=m2, k2^2=p2, k3^2=p3, (k1+k2+k3)^2=m4
! m2,m4 should NOT be identically 0d0
!*******************************************************************
   use avh_olo_mp_logc ,only: logc
   use avh_olo_mp_logc2 ,only: logc2
   use avh_olo_mp_li2c2 ,only: li2c2
   include 'cts_mpc.h'
    ,intent(out) :: rslt(0:2)
   include 'cts_mpc.h'
    ,intent(in)  :: p2,p3,p12,p23 ,m2,m3,m4
   include 'cts_mpr.h'
    ,intent(in)  :: rmu
   include 'cts_mpc.h'
    :: cp2,cp3,cp12,cp23,cm2,cm3,cm4,sm1,sm2,sm3,sm4 &
                     ,r13,r23,r24,r34,d23,d24,d34,log24,cc
   type(qmplx_type) :: q13,q23,q24,q34,qss,qy1,qy2,qz1,qz2
!
!  write(*,*) 'MESSAGE from OneLOop box16: you are calling me' !CALLINGME
!
   if (abs(m2).gt.abs(m4)) then
     cm2=m2 ;cm4=m4 ;cp2=p2 ;cp3=p3
   else
     cm2=m4 ;cm4=m2 ;cp2=p3 ;cp3=p2
   endif
   cm3=m3 ;cp12=p12 ;cp23=p23
!
   if (cp12.eq.cm3) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box16: ' &
       ,'p12=m3, returning 0'
     rslt = C0P0
     return
   endif
!
   sm1 = mp_cx(abs(rmu),R0P0)
   sm2 = mysqrt(cm2)
   sm3 = mysqrt(cm3)
   sm4 = mysqrt(cm4)
!
   r13 = (cm3-cp12)/(sm1*sm3)
   call rfun( r23,d23 ,(cm2+cm3-cp2 )/(sm2*sm3) )
   call rfun( r24,d24 ,(cm2+cm4-cp23)/(sm2*sm4) )
   call rfun( r34,d34 ,(cm3+cm4-cp3 )/(sm3*sm4) )
   q13 = qonv(r13,-1)
   q23 = qonv(r23,-1)
   q24 = qonv(r24,-1)
   q34 = qonv(r34,-1)
!
   if (r24.eq.-C1P0) then 
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box16: ' &
       ,'threshold singularity, returning 0'
     rslt = C0P0
     return
   endif
!
   qss = q23*q34
   qy1 = qss*q24
   qy2 = qss/q24
!
   qss = q23/q34
   qz1 = qss*q24
   qz2 = qss/q24
!
   qss = q13*q23
   qss = (qss*qss)/q24
!
   cc = C1P0/( sm2*sm4*(cp12-cm3) )
   log24 = logc2(q24)*r24/(C1P0+r24)
   rslt(2) = C0P0
   rslt(1) = -log24
   rslt(0) = log24*logc(qss) + li2c2(q24*q24,qonv(C1P0))*r24 &
           - li2c2(qy1,qy2)*r23*r34 - li2c2(qz1,qz2)*r23/r34
   rslt(1) = cc*rslt(1)
   rslt(0) = cc*rslt(0)
   end subroutine


   subroutine box15( rslt ,p2,p3,p12,p23 ,m2,m4 ,rmu )
!*******************************************************************
! calculates
!
!    C   /                  d^(Dim)q
! ------ | -------------------------------------------------
! i*pi^2 / q^2 [(q+k1)^2-m2] (q+k1+k2)^2 [(q+k1+k2+k3)^2-m4]
!
! with  k1^2=m2, k2^2=p2, k3^2=p3, (k1+k2+k3)^2=m4
! m2,m4 should NOT be identically 0d0
!*******************************************************************
   use avh_olo_mp_logc ,only: logc
   use avh_olo_mp_logc2 ,only: logc2
   use avh_olo_mp_li2c2 ,only: li2c2
   include 'cts_mpc.h'
    ,intent(out) :: rslt(0:2)
   include 'cts_mpc.h'
    ,intent(in)  :: p2,p3,p12,p23 ,m2,m4
   include 'cts_mpr.h'
    ,intent(in)  :: rmu
   include 'cts_mpc.h'
    :: cp2,cp3,cp12,cp23,cm2,cm4,sm1,sm2,sm3,sm4 &
                     ,r13,r23,r24,r34,d24,log24,cc
   type(qmplx_type) :: q13,q23,q24,q34,qss,qz1,qz2
!
!  write(*,*) 'MESSAGE from OneLOop box15: you are calling me' !CALLINGME
!
   if (abs(m2-p2).gt.abs(m4-p3)) then
     cm2=m2 ;cm4=m4 ;cp2=p2 ;cp3=p3
   else
     cm2=m4 ;cm4=m2 ;cp2=p3 ;cp3=p2
   endif
   cp12=p12 ;cp23=p23
!
   if (cp12.eq.C0P0) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box15: ' &
       ,'p12=0, returning 0'
     rslt = C0P0
     return
   endif
!
   sm1 = mp_cx(abs(rmu),R0P0)
   sm2 = mysqrt(cm2)
   sm4 = mysqrt(cm4)
   sm3 = mp_cx(abs(sm2),R0P0)
   r13 = (       -cp12)/(sm1*sm3)
   r23 = (cm2    -cp2 )/(sm2*sm3)
   r34 = (    cm4-cp3 )/(sm3*sm4)
   call rfun( r24,d24 ,(cm2+cm4-cp23)/(sm2*sm4) )
!
   if (r24.eq.-C1P0) then 
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box15: ' &
       ,'threshold singularity, returning 0'
     rslt = C0P0
     return
   endif
!
   q13 = qonv(r13,-1)
   q23 = qonv(r23,-1)
   q24 = qonv(r24,-1)
   q34 = qonv(r34,-1)
!
   qss = q13/q23
   qss = (qss*qss)/q24
!
   cc = r24/(sm2*sm4*cp12)
   log24 = logc2(q24)/(C1P0+r24)
   rslt(2) = C0P0
   rslt(1) = -log24
   rslt(0) = log24 * logc(qss) + li2c2(q24*q24,qonv(C1P0))
   if (r34.ne.C0P0) then
     qss = q34/q23
     qz1 = qss*q24
     qz2 = qss/q24
     rslt(0) = rslt(0) - li2c2(qz1,qz2)*r34/(r23*r24)
   endif
   rslt(1) = cc*rslt(1)
   rslt(0) = cc*rslt(0)
   end subroutine


   subroutine box14( rslt ,cp12,cp23 ,cm2,cm4 ,rmu )
!*******************************************************************
! calculates
!
!    C   /                  d^(Dim)q
! ------ | -------------------------------------------------
! i*pi^2 / q^2 [(q+k1)^2-m2] (q+k1+k2)^2 [(q+k1+k2+k3)^2-m4]
!
! with  k1^2=m2, k2^2=m2, k3^2=m4, (k1+k2+k3)^2=m4
! m2,m4 should NOT be identically 0d0
!*******************************************************************
   use avh_olo_mp_logc ,only: logc
   use avh_olo_mp_logc2 ,only: logc2
   include 'cts_mpc.h'
    ,intent(out) :: rslt(0:2)
   include 'cts_mpc.h'
    ,intent(in)  :: cp12,cp23,cm2,cm4
   include 'cts_mpr.h'
    ,intent(in)  :: rmu
   include 'cts_mpc.h'
    :: sm2,sm4,r24,d24,cc
!
!  write(*,*) 'MESSAGE from OneLOop box14: you are calling me' !CALLINGME
!
   if (cp12.eq.C0P0) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box14: ' &
       ,'p12=0, returning 0'
     rslt = C0P0
     return
   endif
!
   sm2 = mysqrt(cm2)
   sm4 = mysqrt(cm4)
   call rfun( r24,d24 ,(cm2+cm4-cp23)/(sm2*sm4) )
!
   if (r24.eq.-C1P0) then 
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box14: ' &
       ,'threshold singularity, returning 0'
     rslt = C0P0
     return
   endif
!
   cc = -2*logc2(qonv(r24,-1))*r24/(C1P0+r24)/(sm2*sm4*cp12)
!
   rslt(2) = C0P0
   rslt(1) = cc
   rslt(0) = -cc*logc(qonv(-cp12/(rmu*rmu),-1))
   end subroutine


   subroutine box13( rslt ,p2,p3,p4,p12,p23 ,m3,m4 ,rmu )
!*******************************************************************
! calculates
!
!    C   /                  d^(Dim)q
! ------ | -------------------------------------------------
! i*pi^2 / q^2 (q+k1)^2 [(q+k1+k2)^2-m3] [(q+k1+k2+k3)^2-m4]
!
! with  k1^2=0, k2^2=p2, k3^2=p3, (k1+k2+k3)^2=p4
! m3,m4 should NOT be identically 0d0
! p4 should NOT be identical to m4
! p2 should NOT be identical to m3
!*******************************************************************
   use avh_olo_mp_logc ,only: logc
   use avh_olo_mp_logc2 ,only: logc2
   use avh_olo_mp_li2c2 ,only: li2c2
   include 'cts_mpc.h'
    ,intent(out) :: rslt(0:2)
   include 'cts_mpc.h'
    ,intent(in)  :: p2,p3,p4,p12,p23,m3,m4
   include 'cts_mpr.h'
    ,intent(in)  :: rmu
   include 'cts_mpc.h'
    :: cp2,cp3,cp4,cp12,cp23,cm3,cm4,sm3,sm4,sm1,sm2 &
             ,r13,r14,r23,r24,r34,d34,cc,logd,li2d,loge,li2f,li2b,li2e
   type(qmplx_type) :: q13,q14,q23,q24,q34,qy1,qy2
   include 'cts_mpr.h'
    :: h1,h2
!
!  write(*,*) 'MESSAGE from OneLOop box13: you are calling me' !CALLINGME
!
   if (p12.eq.m3) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box13: ' &
       ,'p12=m3, returning 0'
     rslt = C0P0
     return
   endif
   if (p23.eq.m4) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box13: ' &
       ,'p23=m4, returning 0'
     rslt = C0P0
     return
   endif
!
   h1 = abs((m3-p12)*(m4-p23))
   h2 = abs((m3-p2 )*(m4-p4 ))
   if (h1.ge.h2) then
     cp2=p2  ;cp3=p3 ;cp4=p4  ;cp12=p12 ;cp23=p23 ;cm3=m3 ;cm4=m4
   else
     cp2=p12 ;cp3=p3 ;cp4=p23 ;cp12=p2  ;cp23=p4  ;cm3=m3 ;cm4=m4
   endif
!
   sm3 = mysqrt(cm3)
   sm4 = mysqrt(cm4)
   sm1 = mp_cx(abs(rmu),R0P0)
   sm2 = sm1
!
   r13 = (cm3-cp12)/(sm1*sm3)
   r14 = (cm4-cp4 )/(sm1*sm4)
   r23 = (cm3-cp2 )/(sm2*sm3)
   r24 = (cm4-cp23)/(sm2*sm4)
   call rfun( r34,d34 ,(cm3+cm4-cp3)/(sm3*sm4) )
!
   q13 = qonv(r13,-1)
   q14 = qonv(r14,-1)
   q23 = qonv(r23,-1)
   q24 = qonv(r24,-1)
   q34 = qonv(r34,-1) 
!
   qy1 = q14*q23/q13/q24
   logd = logc2(qy1     )/(r13*r24)
   li2d = li2c2(qy1,qonv(C1P0))/(r13*r24)
   loge = logc(q13)
!
   qy1 = q23/q24
   qy2 = q13/q14
   li2f = li2c2( qy1*q34,qy2*q34 )*r34/(r14*r24)
   li2b = li2c2( qy1/q34,qy2/q34 )/(r34*r14*r24)
   li2e = li2c2( q14/q24,q13/q23 )/(r23*r24)
!
   rslt(2) = C0P0
   rslt(1) = logd
   rslt(0) = li2f + li2b + 2*li2e - 2*li2d - 2*logd*loge
   cc = sm1*sm2*sm3*sm4
   rslt(1) = rslt(1)/cc
   rslt(0) = rslt(0)/cc
   end subroutine


   subroutine box12( rslt ,cp3,cp4,cp12,cp23 ,cm3,cm4 ,rmu )
!*******************************************************************
! calculates
!
!    C   /                  d^(Dim)q
! ------ | -------------------------------------------------
! i*pi^2 / q^2 (q+k1)^2 [(q+k1+k2)^2-m3] [(q+k1+k2+k3)^2-m4]
!
! with  k1^2=0, k2^2=m3, k3^2=p3, (k1+k2+k3)^2=p4
! m3,m4 should NOT be indentiallcy 0d0
! p4 should NOT be identical to m4
!*******************************************************************
   use avh_olo_mp_logc ,only: logc
   use avh_olo_mp_li2c ,only: li2c
   include 'cts_mpc.h'
    ,intent(out) :: rslt(0:2)
   include 'cts_mpc.h'
    ,intent(in)  :: cp3,cp4,cp12,cp23,cm3,cm4
   include 'cts_mpr.h'
    ,intent(in)  :: rmu
   include 'cts_mpc.h'
    :: sm3,sm4,sm1,sm2,r13,r14,r24,r34,d34,cc &
                     ,log13,log14,log24,log34,li2f,li2b,li2d
   type(qmplx_type) :: q13,q14,q24,q34,qyy
   include 'cts_mpc.h'
    :: const
!
!  write(*,*) 'MESSAGE from OneLOop box12: you are calling me' !CALLINGME
!
   const= C1P0*TWOPI*TWOPI/32
   if (cp12.eq.cm3) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box12: ' &
       ,'p12=m3, returning 0'
     rslt = C0P0
     return
   endif
   if (cp23.eq.cm4) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box12: ' &
       ,'p23=m4, returning 0'
     rslt = C0P0
     return
   endif
!
   sm3 = mysqrt(cm3)
   sm4 = mysqrt(cm4)
   sm1 = mp_cx(abs(rmu),R0P0)
   sm2 = sm1
!
   r13 = (cm3-cp12)/(sm1*sm3)
   r14 = (cm4-cp4 )/(sm1*sm4)
   r24 = (cm4-cp23)/(sm2*sm4)
   call rfun( r34,d34 ,(cm3+cm4-cp3)/(sm3*sm4) )
!
   q13 = qonv(r13,-1)
   q14 = qonv(r14,-1)
   q24 = qonv(r24,-1)
   q34 = qonv(r34,-1) 
!
   log13 = logc(q13) 
   log14 = logc(q14) 
   log24 = logc(q24) 
   log34 = logc(q34) 
!
   qyy = q14/q13
   li2f = li2c(qyy*q34)
   li2b = li2c(qyy/q34)
   li2d = li2c(q14/q24)
!
   rslt(2) = C1P0/2
   rslt(1) = log14 - log24 - log13
   rslt(0) = 2*log13*log24 - log14*log14 - log34*log34 &
           - 2*li2d - li2f - li2b - const
   cc = (cm3-cp12)*(cm4-cp23) ! = sm1*sm2*sm3*sm4*r13*r24
   rslt(2) = rslt(2)/cc
   rslt(1) = rslt(1)/cc
   rslt(0) = rslt(0)/cc
   end subroutine


   subroutine box11( rslt ,cp3,cp12,cp23 ,cm3,cm4 ,rmu )
!*******************************************************************
! calculates
!
!    C   /                  d^(Dim)q
! ------ | -------------------------------------------------
! i*pi^2 / q^2 (q+k1)^2 [(q+k1+k2)^2-m3] [(q+k1+k2+k3)^2-m4]
!
! with  k1^2=0, k2^2=m3, k3^2=p3, (k1+k2+k3)^2=m4
! m3,m4 should NOT be indentiallcy 0d0
!*******************************************************************
   use avh_olo_mp_logc ,only: logc
   include 'cts_mpc.h'
    ,intent(out) :: rslt(0:2)
   include 'cts_mpc.h'
    ,intent(in)  :: cp3,cp12,cp23,cm3,cm4
   include 'cts_mpr.h'
    ,intent(in)  :: rmu
   include 'cts_mpc.h'
    :: sm3,sm4,sm1,sm2,r13,r24,r34,d34 &
                     ,cc,log13,log24,log34
   include 'cts_mpc.h'
    :: const
!
!  write(*,*) 'MESSAGE from OneLOop box11: you are calling me' !CALLINGME
!
   const=(C1P0*TWOPI*TWOPI*7)/48
   if (cp12.eq.cm3) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box11: ' &
       ,'p12=m3, returning 0'
     rslt = C0P0
     return
   endif
   if (cp23.eq.cm4) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box11: ' &
       ,'p23=m4, returning 0'
     rslt = C0P0
     return
   endif
!
   sm3 = mysqrt(cm3)
   sm4 = mysqrt(cm4)
   sm1 = mp_cx(abs(rmu),R0P0)
   sm2 = sm1
!
   r13 = (cm3-cp12)/(sm1*sm3)
   r24 = (cm4-cp23)/(sm2*sm4)
   call rfun( r34,d34 ,(cm3+cm4-cp3 )/(sm3*sm4) )
!
   log13 = logc(qonv(r13,-1)) 
   log24 = logc(qonv(r24,-1)) 
   log34 = logc(qonv(r34,-1)) 
!
   rslt(2) = C1P0
   rslt(1) = -log13-log24
   rslt(0) = 2*log13*log24 - log34*log34 - const
   cc = (cm3-cp12)*(cm4-cp23) ! = sm1*sm2*sm3*sm4*r13*r24
   rslt(2) = rslt(2)/cc
   rslt(1) = rslt(1)/cc
   rslt(0) = rslt(0)/cc
   end subroutine


   subroutine box10( rslt ,p2,p3,p4,p12,p23 ,m4 ,rmu )
!*******************************************************************
! calculates
!
!     C   /               d^(Dim)q
!  ------ | --------------------------------------------
!  i*pi^2 / q^2 (q+k1)^2 (q+k1+k2)^2 [(q+k1+k2+k3)^2-m4]
!
! with  k1^2=0, k2^2=p2, k3^2=p3, (k1+k2+k3)^2=p4
! m4 should NOT be identically 0d0
! p2 should NOT be identically 0d0
! p4 should NOT be identical to m4
!*******************************************************************
   use avh_olo_mp_logc ,only: logc
   use avh_olo_mp_logc2 ,only: logc2
   use avh_olo_mp_li2c2 ,only: li2c2
   include 'cts_mpc.h'
    ,intent(out) :: rslt(0:2)
   include 'cts_mpc.h'
    ,intent(in)  :: p2,p3,p4,p12,p23,m4
   include 'cts_mpr.h'
    ,intent(in)  :: rmu
   include 'cts_mpc.h'
    :: cp2,cp3,cp4,cp12,cp23,cm4,r13,r14,r23,r24,r34,z1,z0
   type(qmplx_type) :: q13,q14,q23,q24,q34,qm4,qxx,qx1,qx2
   include 'cts_mpr.h'
    :: h1,h2
!
!  write(*,*) 'MESSAGE from OneLOop box10: you are calling me' !CALLINGME
!
   if (p12.eq.C0P0) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box10: ' &
       ,'p12=0, returning 0'
     rslt = C0P0
     return
   endif
   if (p23.eq.m4) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box10: ' &
       ,'p23=mm, returning 0'
     rslt = C0P0
     return
   endif
!
   h1 = abs(p12*(m4-p23))
   h2 = abs( p2*(m4-p4 ))
   if (h1.ge.h2) then
     cp2=p2  ;cp3=p3 ;cp4=p4  ;cp12=p12 ;cp23=p23 ;cm4=m4
   else
     cp2=p12 ;cp3=p3 ;cp4=p23 ;cp12=p2  ;cp23=p4  ;cm4=m4
   endif
!
   r23 =    -cp2
   r13 =    -cp12
   r34 = cm4-cp3
   r14 = cm4-cp4
   r24 = cm4-cp23
   q23 = qonv(r23,-1)
   q13 = qonv(r13,-1)
   q34 = qonv(r34,-1)
   q14 = qonv(r14,-1)
   q24 = qonv(r24,-1)
   qm4 = qonv(cm4,-1)
!
   if (r34.ne.C0P0) then
     qx1 = q34/qm4
     qx2 = qx1*q14/q13
     qx1 = qx1*q24/q23
     z0 = -li2c2(qx1,qx2)*r34/(2*cm4*r23)
   else
     z0 = C0P0
   endif
!
   qx1 = q23/q13
   qx2 = q24/q14
   qxx = qx1/qx2
   z1 = -logc2(qxx)/r24
   z0 = z0 - li2c2(qx1,qx2)/r14
   z0 = z0 + li2c2(qxx,qonv(C1P0))/r24
   z0 = z0 + z1*( logc(qm4/q24) - logc(qm4/(rmu*rmu))/2 )
!
   rslt(2) = C0P0
   rslt(1) = -z1/r13
   rslt(0) = -2*z0/r13
   end subroutine


   subroutine box09( rslt ,cp2,cp3,cp12,cp23 ,cm4 ,rmu )
!*******************************************************************
! calculates
!
!     C   /               d^(Dim)q
!  ------ | --------------------------------------------
!  i*pi^2 / q^2 (q+k1)^2 (q+k1+k2)^2 [(q+k1+k2+k3)^2-m4]
!
! with  k1^2=0, k2^2=p2, k3^2=p3, (k1+k2+k3)^2=m4
! m4 should NOT be identically 0d0
! p2 should NOT be identically 0d0
!*******************************************************************
   use avh_olo_mp_logc ,only: logc
   use avh_olo_mp_li2c ,only: li2c
   include 'cts_mpc.h'
    ,intent(out) :: rslt(0:2)
   include 'cts_mpc.h'
    ,intent(in)  :: cp2,cp3,cp12,cp23,cm4
   include 'cts_mpr.h'
    ,intent(in)  :: rmu
   include 'cts_mpc.h'
    :: logm,log12,log23,li12,li23,z2,z1,z0,cc &
                     ,r13,r23,r24,r34
   type(qmplx_type) :: q13,q23,q24,q34,qm4,qxx
   include 'cts_mpc.h'
    :: const
!
!  write(*,*) 'MESSAGE from OneLOop box09: you are calling me' !CALLINGME
!
   const=C1P0*TWOPI*TWOPI/96
   if (cp12.eq.C0P0) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box09: ' &
       ,'p12=0, returning 0'
     rslt = C0P0
     return
   endif
   if (cp23.eq.cm4) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box09: ' &
       ,'p23=mm, returning 0'
     rslt = C0P0
     return
   endif
!
   r23 =    -cp2
   r13 =    -cp12
   r34 = cm4-cp3
   r24 = cm4-cp23
   q23 = qonv(r23,-1)
   q13 = qonv(r13,-1)
   q34 = qonv(r34,-1)
   q24 = qonv(r24,-1)
   qm4 = qonv(cm4,-1)
!
   logm  = logc(qm4/(rmu*rmu))
   qxx = q13/q23
   log12 = logc(qxx)
   li12  = li2c(qxx)
!
   qxx = q24/qm4
   log23 = logc(qxx)
   li23  = li2c(qxx*q34/q23)
!
   z2 = C1P0/2
   z1 = -log12 - log23
   z0 = li23 + 2*li12 + z1*z1 + const
   cc = C1P0/(r13*r24)
   rslt(2) = cc*z2
   rslt(1) = cc*(z1 - z2*logm)
   rslt(0) = cc*(z0 + (z2*logm/2-z1)*logm)
   end subroutine


   subroutine box08( rslt ,cp3,cp4,cp12,cp23 ,cm4 ,rmu )
!*******************************************************************
! calculates
!
!     C   /               d^(Dim)q
!  ------ | --------------------------------------------
!  i*pi^2 / q^2 (q+k1)^2 (q+k1+k2)^2 [(q+k1+k2+k3)^2-m4]
!
! with  k1^2=k2^2=0, k3^2=p3, (k1+k2+k3)^2=p4
! mm should NOT be identically 0d0
! p3 NOR p4 should be identically m4
!*******************************************************************
   use avh_olo_mp_logc ,only: logc
   use avh_olo_mp_li2c ,only: li2c
   include 'cts_mpc.h'
    ,intent(out) :: rslt(0:2)
   include 'cts_mpc.h'
    ,intent(in)  :: cp3,cp4,cp12,cp23,cm4
   include 'cts_mpr.h'
    ,intent(in)  :: rmu
   type(qmplx_type) :: q13,q14,q24,q34,qm4,qxx,qx1,qx2,qx3
   include 'cts_mpc.h'
    :: r13,r14,r24,r34,z1,z0,cc
   include 'cts_mpr.h'
    :: rmu2
   include 'cts_mpc.h'
    :: const
!
!  write(*,*) 'MESSAGE from OneLOop box08: you are calling me' !CALLINGME
!
   const=C1P0*TWOPI*TWOPI/16
   if (cp12.eq.C0P0) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box08: ' &
       ,'p12=0, returning 0'
     rslt = C0P0
     return
   endif
   if (cp23.eq.cm4) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box08: ' &
       ,'p23=mm, returning 0'
     rslt = C0P0
     return
   endif
!
   rmu2 = rmu*rmu
   r13 =    -cp12
   r34 = cm4-cp3
   r14 = cm4-cp4
   r24 = cm4-cp23
   q13 = qonv(r13,-1)
   q34 = qonv(r34,-1)
   q14 = qonv(r14,-1)
   q24 = qonv(r24,-1)
   qm4 = qonv(cm4,-1)
!
   qx1 = q34/q24
   qx2 = q14/q24
   qx3 = q13/rmu2
   z1 = logc(qx1*qx2/qx3)
   z0 = 2*( logc(q24/rmu2)*logc(qx3) - (li2c(qx1)+li2c(qx2)) )
!
   qx1 = q34/rmu2
   qx2 = q14/rmu2
   qxx = qx1*qx2/qx3
   z0 = z0 - logc(qx1)**2 - logc(qx2)**2 &
           + logc(qxx)**2/2 + li2c(qm4/qxx/rmu2)
!
   cc = C1P0/(r13*r24)
   rslt(2) = cc
   rslt(1) = cc*z1
   rslt(0) = cc*( z0 - const )
   end subroutine


   subroutine box07( rslt ,cp4,cp12,cp23 ,cm4 ,rmu )
!*******************************************************************
! calculates
!
!     C   /               d^(Dim)q
!  ------ | --------------------------------------------
!  i*pi^2 / q^2 (q+k1)^2 (q+k1+k2)^2 [(q+k1+k2+k3)^2-m4]
!
! with  k1^2=k2^2=0, k3^2=m4, (k1+k2+k3)^2=p4
! m3 should NOT be identically 0d0
! p4 should NOT be identically m4
!*******************************************************************
   use avh_olo_mp_logc ,only: logc
   use avh_olo_mp_li2c ,only: li2c
   include 'cts_mpc.h'
    ,intent(out) :: rslt(0:2)
   include 'cts_mpc.h'
    ,intent(in)  :: cp4,cp12,cp23,cm4
   include 'cts_mpr.h'
    ,intent(in)  :: rmu
   type(qmplx_type) :: q13,q14,q24,qm4
   include 'cts_mpc.h'
    :: r13,r14,r24,logm,log12,log23,log4,li423 &
                     ,z2,z1,z0,cc
   include 'cts_mpc.h'
    :: const
!
!  write(*,*) 'MESSAGE from OneLOop box07: you are calling me' !CALLINGME
!
   const=(C1P0*TWOPI*TWOPI*13)/96
   if (cp12.eq.C0P0) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box07: ' &
       ,'p12=0, returning 0'
     rslt = C0P0
     return
   endif
   if (cp23.eq.cm4) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box07: ' &
       ,'p23=mm, returning 0'
     rslt = C0P0
     return
   endif
!
   r13 =    -cp12
   r14 = cm4-cp4
   r24 = cm4-cp23
   q13 = qonv(r13,-1)
   q14 = qonv(r14,-1)
   q24 = qonv(r24,-1)
   qm4 = qonv(cm4,-1)
!
   logm  = logc(qm4/(rmu*rmu))
   log12 = logc(q13/qm4)
   log23 = logc(q24/qm4)
   log4  = logc(q14/qm4)
   li423 = li2c(q14/q24)
!
   z2 = (C1P0*3)/2
   z1 = -2*log23 - log12 + log4
   z0 = 2*(log12*log23 - li423) - log4*log4 - const
   cc = C1P0/(r13*r24)
   rslt(2) = cc*z2
   rslt(1) = cc*(z1 - z2*logm)
   rslt(0) = cc*(z0 + (z2*logm/2-z1)*logm)
   end subroutine


   subroutine box06( rslt ,cp12,cp23 ,cm4 ,rmu )
!*******************************************************************
! calculates
!
!     C   /               d^(Dim)q
!  ------ | --------------------------------------------
!  i*pi^2 / q^2 (q+k1)^2 (q+k1+k2)^2 [(q+k1+k2+k3)^2-m4]
!
! with  k1^2=k2^2=0, k3^2=(k1+k2+k3)^2=m4
! m3 should NOT be identically 0d0
!*******************************************************************
   use avh_olo_mp_logc ,only: logc
   include 'cts_mpc.h'
    ,intent(out) :: rslt(0:2)
   include 'cts_mpc.h'
    ,intent(in)  :: cp12,cp23,cm4
   include 'cts_mpr.h'
    ,intent(in)  :: rmu
   type(qmplx_type) :: q13,q24,qm4
   include 'cts_mpc.h'
    :: r13,r24,logm,log1,log2,z2,z1,z0,cc
   include 'cts_mpc.h'
    :: const
!
!  write(*,*) 'MESSAGE from OneLOop box06: you are calling me' !CALLINGME
!
   const=C1P0*TWOPI*TWOPI/12
   if (cp12.eq.C0P0) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box06: ' &
       ,'p12=0, returning 0'
     rslt = C0P0
     return
   endif
   if (cp23.eq.cm4) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box06: ' &
       ,'p23=mm, returning 0'
     rslt = C0P0
     return
   endif
!
   r13 =    -cp12
   r24 = cm4-cp23
   q13 = qonv(r13,-1)
   q24 = qonv(r24,-1)
   qm4 = qonv(cm4,-1)
!
   logm = logc(qm4/(rmu*rmu))
   log1 = logc(q13/qm4)
   log2 = logc(q24/qm4)
!
   z2 = C1P0*2
   z1 = -2*log2 - log1
   z0 = 2*(log2*log1 - const)
   cc = C1P0/(r13*r24)
   rslt(2) = cc*z2
   rslt(1) = cc*(z1 - z2*logm)
   rslt(0) = cc*(z0 + (z2*logm/2-z1)*logm)
   end subroutine


   subroutine box03( rslt ,p2,p4,p5,p6 ,rmu )
!*******************************************************************
! calculates
!
!     C   /               d^(Dim)q
!  ------ | ---------------------------------------
!  i*pi^2 / q^2 (q+k1)^2 (q+k1+k2)^2 (q+k1+k2+k3)^2
!
! with  k1^2=k3^2=0
!*******************************************************************
   use avh_olo_mp_logc ,only: logc
   use avh_olo_mp_logc2 ,only: logc2
   use avh_olo_mp_li2c2 ,only: li2c2
   include 'cts_mpc.h'
    ,intent(out) :: rslt(0:2)
   include 'cts_mpc.h'
    ,intent(in)  :: p2,p4,p5,p6 
   include 'cts_mpr.h'
    ,intent(in)  :: rmu
   type(qmplx_type) :: q2,q4,q5,q6,q26,q54,qy
   include 'cts_mpc.h'
    :: logy
   include 'cts_mpr.h'
    :: rmu2
!
!  write(*,*) 'MESSAGE from OneLOop box03: you are calling me' !CALLINGME
!
   rmu2 = rmu*rmu
   q2 = qonv(-p2,-1)
   q4 = qonv(-p4,-1)
   q5 = qonv(-p5,-1)
   q6 = qonv(-p6,-1)
   q26 = q2/q6
   q54 = q5/q4
   qy = q26/q54
   logy = logc2(qy)/(p5*p6)
   rslt(1) = logy
   rslt(0) = li2c2(q6/q4,q2/q5)/(p4*p5) &
           + li2c2(q54,q26)/(p4*p6)     &
           - li2c2(qonv(C1P0),qy)/(p5*p6) &
           - logy*logc(q54*q2*q6/(rmu2*rmu2))/2
   rslt(2) = C0P0
   rslt(1) = 2*rslt(1)
   rslt(0) = 2*rslt(0)
   end subroutine


   subroutine box05( rslt ,p2,p3,p4,p5,p6 ,rmu )
!*******************************************************************
! calculates
!
!     C   /               d^(Dim)q
!  ------ | ---------------------------------------
!  i*pi^2 / q^2 (q+k1)^2 (q+k1+k2)^2 (q+k1+k2+k3)^2
!
! with  k1^2=0
!*******************************************************************
   use avh_olo_mp_logc ,only: logc
   use avh_olo_mp_logc2 ,only: logc2
   use avh_olo_mp_li2c2 ,only: li2c2
   include 'cts_mpc.h'
    ,intent(out) :: rslt(0:2)
   include 'cts_mpc.h'
    ,intent(in)  :: p2,p3,p4,p5,p6
   include 'cts_mpr.h'
    ,intent(in)  :: rmu
   type(qmplx_type) ::q2,q3,q4,q5,q6 ,q25,q64,qy,qz
   include 'cts_mpc.h'
    :: logy
   include 'cts_mpr.h'
    :: rmu2
!
!  write(*,*) 'MESSAGE from OneLOop box05: you are calling me' !CALLINGME
!
   rmu2 = rmu*rmu
   q2 = qonv(-p2,-1)
   q3 = qonv(-p3,-1)
   q4 = qonv(-p4,-1)
   q5 = qonv(-p5,-1)
   q6 = qonv(-p6,-1)
   q25 = q2/q5
   q64 = q6/q4
   qy = q25/q64
   qz = q64*q2*q5*q6*q6/q3/q3/(rmu2*rmu2)
!
   logy = logc2(qy)/(p5*p6)
   rslt(2) = C0P0
   rslt(1) = logy
   rslt(0) = li2c2(q64,q25)/(p4*p5) &
           - li2c2(qonv(C1P0),qy)/(p5*p6) &
           - logy*logc(qz)/4
   rslt(0) = 2*rslt(0)
   end subroutine


   subroutine box00( rslt ,cp ,api ,rmu )
!*******************************************************************
! calculates
!               C   /              d^(Dim)q
!            ------ | ---------------------------------------
!            i*pi^2 / q^2 (q+k1)^2 (q+k1+k2)^2 (q+k1+k2+k3)^2
!
! with  Dim = 4-2*eps
!         C = pi^eps * mu^(2*eps) * exp(gamma_Euler*eps)
!
! input:  p1 = k1^2,  p2 = k2^2,  p3 = k3^2,  p4 = (k1+k2+k3)^2,
!         p12 = (k1+k2)^2,  p23 = (k2+k3)^2
! output: rslt(0) = eps^0   -coefficient
!         rslt(1) = eps^(-1)-coefficient
!         rslt(2) = eps^(-2)-coefficient
!
! If any of these numbers is IDENTICALLY 0d0, the corresponding
! IR-singular case is returned.
!*******************************************************************
   use avh_olo_mp_loga ,only: loga
   use avh_olo_mp_li2a ,only: li2a
   include 'cts_mpc.h'
    ,intent(out) :: rslt(0:2)
   include 'cts_mpc.h'
    ,intent(in)  :: cp(6)
   include 'cts_mpr.h'
    ,intent(in)  :: api(6),rmu
   include 'cts_mpc.h'
    :: log3,log4,log5,log6,li24,li25,li26 &
                     ,li254,li263
   include 'cts_mpr.h'
    :: rp1,rp2,rp3,rp4,rp5,rp6,pp(6),ap(6),gg,ff,hh,arg,rmu2
   integer :: icase,sf,sgn,i3,i4,i5,i6
   integer ,parameter :: base(4)=(/8,4,2,1/)
!
   rmu2 = rmu*rmu
   ff = api(5)*api(6)
   gg = api(2)*api(4)
   hh = api(1)*api(3)
   if     (ff.ge.gg.and.ff.ge.hh) then
     pp(1)=mp_rl(cp(1)) ;ap(1)=api(1)
     pp(2)=mp_rl(cp(2)) ;ap(2)=api(2)
     pp(3)=mp_rl(cp(3)) ;ap(3)=api(3)
     pp(4)=mp_rl(cp(4)) ;ap(4)=api(4)
     pp(5)=mp_rl(cp(5)) ;ap(5)=api(5)
     pp(6)=mp_rl(cp(6)) ;ap(6)=api(6)
   elseif (gg.ge.ff.and.gg.ge.hh) then
     pp(1)=mp_rl(cp(1)) ;ap(1)=api(1)
     pp(2)=mp_rl(cp(6)) ;ap(2)=api(6)
     pp(3)=mp_rl(cp(3)) ;ap(3)=api(3)
     pp(4)=mp_rl(cp(5)) ;ap(4)=api(5)
     pp(5)=mp_rl(cp(4)) ;ap(5)=api(4)
     pp(6)=mp_rl(cp(2)) ;ap(6)=api(2)
   else
     pp(1)=mp_rl(cp(5)) ;ap(1)=api(5)
     pp(2)=mp_rl(cp(2)) ;ap(2)=api(2)
     pp(3)=mp_rl(cp(6)) ;ap(3)=api(6)
     pp(4)=mp_rl(cp(4)) ;ap(4)=api(4)
     pp(5)=mp_rl(cp(1)) ;ap(5)=api(1)
     pp(6)=mp_rl(cp(3)) ;ap(6)=api(3)
   endif
!
   icase = 0
   if (ap(1).gt.R0P0) icase = icase + base(1)
   if (ap(2).gt.R0P0) icase = icase + base(2)
   if (ap(3).gt.R0P0) icase = icase + base(3)
   if (ap(4).gt.R0P0) icase = icase + base(4)
   rp1 = pp(permtable(1,icase))
   rp2 = pp(permtable(2,icase))
   rp3 = pp(permtable(3,icase))
   rp4 = pp(permtable(4,icase))
   rp5 = pp(permtable(5,icase))
   rp6 = pp(permtable(6,icase))
   icase = casetable(   icase)
!
   i3=0 ;if (-rp3.lt.R0P0) i3=-1
   i4=0 ;if (-rp4.lt.R0P0) i4=-1
   i5=0 ;if (-rp5.lt.R0P0) i5=-1
   i6=0 ;if (-rp6.lt.R0P0) i6=-1
!
   if     (icase.eq.0) then
! 0 masses non-zero
!  write(*,*) 'MESSAGE from OneLOop box00 0: you are calling me' !CALLINGME
     gg = R1P0/( rp5 * rp6 )
     log5 = loga(-rp5/rmu2, i5 )
     log6 = loga(-rp6/rmu2, i6 )
     rslt(2) = gg*( 4*C1P0 )
     rslt(1) = gg*(-2*(log5 + log6) )
     rslt(0) = gg*( log5**2 + log6**2 - loga( rp5/rp6 ,i5-i6 )**2 - (pi2*4)/3 )
   elseif (icase.eq.1) then
! 1 mass non-zero
!  write(*,*) 'MESSAGE from OneLOop box00 1: you are calling me' !CALLINGME
     gg = R1P0/( rp5 * rp6 )
     ff =  gg*( rp5 + rp6 - rp4 )
     log4 = loga(-rp4/rmu2,i4)
     log5 = loga(-rp5/rmu2,i5)
     log6 = loga(-rp6/rmu2,i6)
     sf = nint(mp_sn(R1P0,ff))
     sgn = 0
       arg = rp4*ff 
       if (arg.lt.R0P0) sgn = sf
       li24 = li2a(arg,sgn)
     sgn = 0
       arg = rp5*ff 
       if (arg.lt.R0P0) sgn = sf
       li25 = li2a(arg,sgn)
     sgn = 0
       arg = rp6*ff 
       if (arg.lt.R0P0) sgn = sf
       li26 = li2a(arg,sgn)
     rslt(2) = gg*( 2*C1P0 )
     rslt(1) = gg*( 2*(log4-log5-log6) )
     rslt(0) = gg*( log5**2 + log6**2 - log4**2 - pi2/2 &
                   + 2*(li25 + li26 - li24) )
   elseif (icase.eq.2) then
! 2 neighbour masses non-zero
!  write(*,*) 'MESSAGE from OneLOop box00 2: you are calling me' !CALLINGME
     gg = R1P0/( rp5 * rp6 )
     ff =  gg*( rp5 + rp6 - rp4 )
     log3 = loga(-rp3/rmu2,i3)
     log4 = loga(-rp4/rmu2,i4)
     log5 = loga(-rp5/rmu2,i5)
     log6 = loga(-rp6/rmu2,i6)
     li254 = li2a( rp4/rp5 ,i4-i5 )
     li263 = li2a( rp3/rp6 ,i3-i6 )
     sf = nint(mp_sn(R1P0,ff))
     sgn = 0
       arg = rp4*ff 
       if (arg.lt.R0P0) sgn = sf
       li24 = li2a(arg,sgn)
     sgn = 0
       arg = rp5*ff 
       if (arg.lt.R0P0) sgn = sf
       li25 = li2a(arg,sgn)
     sgn = 0
       arg = rp6*ff 
       if (arg.lt.R0P0) sgn = sf
       li26 = li2a(arg,sgn)
     rslt(2) = gg
     rslt(1) = gg*( log4 + log3 - log5 - 2*log6 )
     rslt(0) = gg*( log5**2 + log6**2 - log3**2 - log4**2 &
                   + (log3 + log4 - log5)**2/2 &
                   - pi2/12 + 2*(li254 - li263 + li25 + li26 - li24) )
   elseif (icase.eq.5) then
! 2 opposite masses non-zero
     call box03( rslt ,mp_cx(rp2,R0P0),mp_cx(rp4,R0P0) &
                      ,mp_cx(rp5,R0P0),mp_cx(rp6,R0P0) ,rmu )
   elseif (icase.eq.3) then
! 3 masses non-zero
     call box05( rslt ,mp_cx(rp2,R0P0),mp_cx(rp3,R0P0) &
                      ,mp_cx(rp4,R0P0),mp_cx(rp5,R0P0) &
                      ,mp_cx(rp6,R0P0) ,rmu )
   elseif (icase.eq.4) then
! 4 masses non-zero
     call boxf0( rslt ,mp_cx(rp1,R0P0),mp_cx(rp2,R0P0) &
                      ,mp_cx(rp3,R0P0),mp_cx(rp4,R0P0) &
                      ,mp_cx(rp5,R0P0),mp_cx(rp6,R0P0) )
   endif
   end subroutine

  
  subroutine boxf0( rslt ,p1,p2,p3,p4,p12,p23 )
!*******************************************************************
! Finite 1-loop scalar 4-point function with all internal masses
! equal zero. Based on the formulas from
! A. Denner, U. Nierste, R. Scharf, Nucl.Phys.B367(1991)637-656
!*******************************************************************
  use avh_olo_mp_logc ,only: logc
  use avh_olo_mp_logc2 ,only: logc2
  use avh_olo_mp_li2c2 ,only: li2c2
  include 'cts_mpc.h'
   ,intent(out) :: rslt(0:2) 
  include 'cts_mpc.h'
   ,intent(in) :: p1,p2,p3,p4,p12,p23
  type(qmplx_type) :: q12,q13,q14,q23,q24,q34,qx1,qx2,qss
  include 'cts_mpc.h'
   :: aa,bb,cc,dd,x1,x2,ss,r12,r13,r14,r23,r24,r34
  include 'cts_mpr.h'
   :: hh
!
!  write(*,*) 'MESSAGE from OneLOop boxf0: you are calling me' !CALLINGME
!
  r12 = -p1  !  p1
  r13 = -p12 !  p1+p2
  r14 = -p4  !  p1+p2+p3
  r23 = -p2  !  p2
  r24 = -p23 !  p2+p3
  r34 = -p3  !  p3      
!
  aa = r34*r24
!
  if (r13.eq.C0P0.or.aa.eq.C0P0) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop boxf0: ' &
       ,'threshold singularity, returning 0'
    rslt = C0P0
    return
  endif
!
  bb = r13*r24 + r12*r34 - r14*r23
  cc = r12*r13
  hh = mp_rl(r23)
  dd = mysqrt( bb*bb - 4*aa*cc , -mp_rl(aa)*hh )
  call solabc(x1,x2,dd ,aa,bb,cc ,1)
  x1 = -x1
  x2 = -x2
!
  qx1 = qonv(x1 , hh)
  qx2 = qonv(x2 ,-hh)
  q12 = qonv(r12,-1)
  q13 = qonv(r13,-1)
  q14 = qonv(r14,-1)
  q23 = qonv(r23,-1)
  q24 = qonv(r24,-1)
  q34 = qonv(r34,-1)
!
  rslt = C0P0
!
  qss = q34/q13
  rslt(0) = rslt(0) + li2c2(qx1*qss,qx2*qss) * r34/r13
!
  qss = q24/q12
  rslt(0) = rslt(0) + li2c2(qx1*qss,qx2*qss) * r24/r12
!
  ss = -logc2(qx1/qx2) / x2
  rslt(0) = rslt(0) + ss*( logc(qx1*qx2)/2 - logc(q12*q13/q14/q23) )
!
  rslt(0) = -rslt(0) / aa
  end subroutine


  subroutine boxf1( rslt ,p1,p2,p3,p4,p12,p23 ,m4 )
!*******************************************************************
! Finite 1-loop scalar 4-point function with one internal mass
! non-zero. Based on the formulas from
! A. Denner, U. Nierste, R. Scharf, Nucl.Phys.B367(1991)637-656
!*******************************************************************
  use avh_olo_mp_logc ,only: logc
  use avh_olo_mp_logc2 ,only: logc2
  use avh_olo_mp_li2c2 ,only: li2c2
  include 'cts_mpc.h'
   ,intent(out) :: rslt(0:2) 
  include 'cts_mpc.h'
   ,intent(in) :: p1,p2,p3,p4,p12,p23 ,m4
  type(qmplx_type) :: qx1,qx2,qss,q12,q13,q14,q23,q24,q34
  include 'cts_mpc.h'
   :: smm,sm4,aa,bb,cc,dd,x1,x2,r12,r13,r14,r23,r24,r34
  logical :: r12zero,r13zero
!
!  write(*,*) 'MESSAGE from OneLOop boxf1: you are calling me' !CALLINGME
!
  sm4 = mysqrt(m4)
  smm = mp_cx(abs(sm4),R0P0) 
!
  r12 = C0P0
  r13 = C0P0
  r14 = C0P0
  r23 = C0P0
  r24 = C0P0
  r34 = C0P0
  if (m4.ne.p4 ) r12 = ( m4-p4 *oieps )/(smm*sm4)
  if (m4.ne.p23) r13 = ( m4-p23*oieps )/(smm*sm4)
  if (m4.ne.p3 ) r14 = ( m4-p3 *oieps )/(smm*sm4)
                 r23 = (   -p1 *oieps )/(smm*smm)
                 r24 = (   -p12*oieps )/(smm*smm)
                 r34 = (   -p2 *oieps )/(smm*smm)
!
  r12zero = (r12.eq.C0P0)
  r13zero = (r13.eq.C0P0)
!
  aa = r34*r24
!
  if (aa.eq.C0P0) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop boxf1: ' &
       ,'threshold singularity, returning 0'
    rslt = C0P0
    return
  endif
!
  bb = r13*r24 + r12*r34 - r14*r23
  cc = r12*r13 - r23
  call solabc(x1,x2,dd ,aa,bb,cc ,0)
  x1 = -x1
  x2 = -x2
!
  qx1 = qonv(x1 ,1 )
  qx2 = qonv(x2 ,1 )
  q12 = qonv(r12,-1)
  q13 = qonv(r13,-1)
  q14 = qonv(r14,-1)
  q23 = qonv(r23,-1)
  q24 = qonv(r24,-1)
  q34 = qonv(r34,-1)
!
  rslt = C0P0
!
  if (r12zero.and.r13zero) then
    qss = qx1*qx2*q34*q24/q23
    qss = qss*qss
    rslt(0) = rslt(0) + logc2( qx1/qx2 )*logc( qss )/(x2*2)
  else
    if (r13zero) then
      qss = q34*q12/q23
      qss = qx1*qx2*qss*qss
      rslt(0) = rslt(0) + logc2( qx1/qx2 )*logc( qss )/(x2*2)
    else
      qss = q34/q13
      rslt(0) = rslt(0) + li2c2( qx1*qss ,qx2*qss )*r34/r13
    endif
    if (r12zero) then
      qss = q24*q13/q23
      qss = qx1*qx2*qss*qss
      rslt(0) = rslt(0) + logc2( qx1/qx2 )*logc( qss )/(x2*2)
    else
      qss = q24/q12
      rslt(0) = rslt(0) + li2c2( qx1*qss ,qx2*qss )*r24/r12
    endif
    if (.not.r12zero.and..not.r13zero) then
      rslt(0) = rslt(0) + logc2( qx1/qx2 )*logc( q12*q13/q23 )/x2
    endif
  endif
!
  if (r14.ne.C0P0) then
    rslt(0) = rslt(0) - li2c2( qx1*q14 ,qx2*q14 )*r14
  endif
!
  rslt(0) = -rslt(0)/(aa*smm*smm*smm*sm4)
  end subroutine


  subroutine boxf5( rslt ,p1,p2,p3,p4,p12,p23, m2,m4 )
!*******************************************************************
! Finite 1-loop scalar 4-point function with two opposite internal
! masses non-zero. Based on the formulas from
! A. Denner, U. Nierste, R. Scharf, Nucl.Phys.B367(1991)637-656
!*******************************************************************
  include 'cts_mpc.h'
   ,intent(out) :: rslt(0:2) 
  include 'cts_mpc.h'
   ,intent(in) :: p1,p2,p3,p4,p12,p23,m2,m4
  call boxf2( rslt ,p12,p2,p23,p4,p1,p3 ,m2,m4 )
  end subroutine


  subroutine boxf2( rslt ,p1,p2,p3,p4,p12,p23 ,m3,m4 )
!*******************************************************************
! Finite 1-loop scalar 4-point function with two adjacent internal
! masses non-zero. Based on the formulas from
! A. Denner, U. Nierste, R. Scharf, Nucl.Phys.B367(1991)637-656
!*******************************************************************
  use avh_olo_mp_logc ,only: logc
  use avh_olo_mp_logc2 ,only: logc2
  use avh_olo_mp_li2c2 ,only: li2c2
  include 'cts_mpc.h'
   ,intent(out) :: rslt(0:2) 
  include 'cts_mpc.h'
   ,intent(in) :: p1,p2,p3,p4,p12,p23,m3,m4
  type(qmplx_type) :: qx1,qx2,qss,q12,q13,q14,q23,q24,q34
  include 'cts_mpc.h'
   :: smm,sm3,sm4,aa,bb,cc,dd,x1,x2 &
                    ,r12,r13,r14,r23,r24,r34,d14,k14
  logical :: r12zero,r13zero,r24zero,r34zero
!
!  write(*,*) 'MESSAGE from OneLOop boxf2: you are calling me' !CALLINGME
!
  sm3 = mysqrt(m3)
  sm4 = mysqrt(m4)
!
  smm = mp_cx(abs(sm3),R0P0) 
!
  r12 = C0P0
  r13 = C0P0
  k14 = C0P0
  r23 = C0P0
  r24 = C0P0
  r34 = C0P0
  if (   m4.ne.p4 ) r12 = (    m4-p4 *oieps )/(smm*sm4)
  if (   m4.ne.p23) r13 = (    m4-p23*oieps )/(smm*sm4)
  if (m3+m4.ne.p3 ) k14 = ( m3+m4-p3 *oieps )/(sm3*sm4)
                    r23 = (      -p1 *oieps )/(smm*smm)
  if (   m3.ne.p12) r24 = (    m3-p12*oieps )/(smm*sm3)
  if (   m3.ne.p2 ) r34 = (    m3-p2 *oieps )/(smm*sm3)
!
  r12zero = (r12.eq.C0P0)
  r13zero = (r13.eq.C0P0)
  r24zero = (r24.eq.C0P0)
  r34zero = (r34.eq.C0P0)
  if (r12zero.and.r24zero) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop boxf2: ' &
       ,'m4=p4 and m3=p12, returning 0'
    rslt = C0P0
    return
  endif
  if (r13zero.and.r34zero) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop boxf2: ' &
       ,'m4=p23 and m3=p2, returning 0'
    rslt = C0P0
    return
  endif
!
  call rfun( r14,d14 ,k14 )
!
  aa = r34*r24 - r23
!
  if (aa.eq.C0P0) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop boxf2: ' &
       ,'threshold singularity, returning 0'
    rslt = C0P0
    return
  endif
!
  bb = r13*r24 + r12*r34 - k14*r23
  cc = r12*r13 - r23
  call solabc(x1,x2,dd ,aa,bb,cc ,0)
  x1 = -x1
  x2 = -x2
!
  qx1 = qonv(x1 ,1 )
  qx2 = qonv(x2 ,1 )
  q12 = qonv(r12,-1)
  q13 = qonv(r13,-1)
  q14 = qonv(r14,-1)
  q23 = qonv(r23,-1)
  q24 = qonv(r24,-1)
  q34 = qonv(r34,-1)
!
  rslt = C0P0
!
  rslt(0) = rslt(0) - li2c2( qx1*q14 ,qx2*q14 )*r14
  rslt(0) = rslt(0) - li2c2( qx1/q14 ,qx2/q14 )/r14
!
  if (r12zero.and.r13zero) then
    qss = qx1*qx2*q34*q24/q23
    qss = qss*qss
    rslt(0) = rslt(0) + logc2( qx1/qx2 )*logc( qss )/(x2*2)
  else
    if (r13zero) then
      qss = q34*q12/q23
      qss = qx1*qx2*qss*qss
      rslt(0) = rslt(0) + logc2( qx1/qx2 )*logc( qss )/(x2*2)
    elseif (.not.r34zero) then
      qss = q34/q13
      rslt(0) = rslt(0) + li2c2( qx1*qss ,qx2*qss )*r34/r13
    endif
    if (r12zero) then
      qss = q24*q13/q23
      qss = qx1*qx2*qss*qss
      rslt(0) = rslt(0) + logc2( qx1/qx2 )*logc( qss )/(x2*2)
    elseif (.not.r24zero) then
      qss = q24/q12
      rslt(0) = rslt(0) + li2c2( qx1*qss ,qx2*qss )*r24/r12
    endif
    if (.not.r12zero.and..not.r13zero) then
      rslt(0) = rslt(0) + logc2( qx1/qx2 )*logc( q12*q13/q23 )/x2 
    endif
  endif
!
  rslt(0) = -rslt(0)/(aa*smm*smm*sm3*sm4)
  end subroutine


  subroutine boxf3( rslt ,pp ,mm )
!*******************************************************************
! Finite 1-loop scalar 4-point function with three internal masses
! non-zero.
!*******************************************************************
  include 'cts_mpc.h'
   ,intent(out) :: rslt(0:2) 
  include 'cts_mpc.h'
   ,intent(in) :: pp(6),mm(4)
  integer :: j
  integer ,parameter :: ip(6)=(/4,5,2,6,3,1/)
  integer ,parameter :: im(4)=(/4,1,3,2/)
  integer ,parameter :: ic(4,6)=reshape((/1,2,3,4 ,2,3,4,1 ,3,4,1,2 &
                                  ,4,1,2,3 ,5,6,5,6 ,6,5,6,5/),(/4,6/))
!
  if     (mm(1).eq.C0P0) then ;j=3
  elseif (mm(2).eq.C0P0) then ;j=4
  elseif (mm(3).eq.C0P0) then ;j=1
  else                        ;j=2
  endif
  call boxf33( rslt ,pp(ic(j,ip(1))) ,pp(ic(j,ip(2))) ,pp(ic(j,ip(3))) &
                    ,pp(ic(j,ip(4))) ,pp(ic(j,ip(5))) ,pp(ic(j,ip(6))) &
                    ,mm(ic(j,im(1))) ,mm(ic(j,im(2))) ,mm(ic(j,im(4))) )
  end subroutine

  subroutine boxf33( rslt ,p1,p2,p3,p4,p12,p23, m1,m2,m4 )
!*******************************************************************
! Finite 1-loop scalar 4-point function with three internal masses
! non-zero, and m3=0. Based on the formulas from
! A. Denner, U. Nierste, R. Scharf, Nucl.Phys.B367(1991)637-656
!*******************************************************************
  use avh_olo_mp_logc ,only: logc
  use avh_olo_mp_logc2 ,only: logc2
  use avh_olo_mp_li2c2 ,only: li2c2
  include 'cts_mpc.h'
   ,intent(out) :: rslt(0:2) 
  include 'cts_mpc.h'
   ,intent(in) :: p1,p2,p3,p4,p12,p23,m1,m2,m4
  type(qmplx_type) :: qx1,qx2,qss,q12,q13,q14,q23,q24,q34,qy1,qy2
  include 'cts_mpc.h'
   :: sm1,sm2,sm3,sm4 ,aa,bb,cc,dd,x1,x2 &
                    ,r12,r13,r14,r23,r24,r34,d12,d14,d24,k12,k14,k24
  logical ::r13zero,r23zero,r34zero
!
!  write(*,*) 'MESSAGE from OneLOop boxf33: you are calling me' !CALLINGME
!
  sm1 = mysqrt(m1)
  sm2 = mysqrt(m2)
  sm4 = mysqrt(m4)
  sm3 = mp_cx(abs(sm2),R0P0) 
!
  k12 = C0P0
  r13 = C0P0
  k14 = C0P0
  r23 = C0P0
  k24 = C0P0
  r34 = C0P0
  if (m1+m2.ne.p1 ) k12 = ( m1 + m2 - p1 *oieps )/(sm1*sm2) ! p1
  if (m1   .ne.p12) r13 = ( m1      - p12*oieps )/(sm1*sm3) ! p1+p2
  if (m1+m4.ne.p4 ) k14 = ( m1 + m4 - p4 *oieps )/(sm1*sm4) ! p1+p2+p3
  if (m2   .ne.p2 ) r23 = ( m2      - p2 *oieps )/(sm2*sm3) ! p2
  if (m2+m4.ne.p23) k24 = ( m2 + m4 - p23*oieps )/(sm2*sm4) ! p2+p3
  if (   m4.ne.p3 ) r34 = (      m4 - p3 *oieps )/(sm3*sm4) ! p3
!
  r13zero = (r13.eq.C0P0)
  r23zero = (r23.eq.C0P0)
  r34zero = (r34.eq.C0P0)
  if (r13zero) then
    if     (r23zero) then
      if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop boxf33: ' &
       ,'m4=p4 and m3=p12, returning 0'
      rslt = C0P0
      return
    elseif (r34zero) then
      if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop boxf33: ' &
       ,'m2=p1 and m3=p12, returning 0'
      rslt = C0P0
      return
    endif
  endif
!
  call rfun( r12,d12 ,k12 )
  call rfun( r14,d14 ,k14 )
  call rfun( r24,d24 ,k24 )
!
  aa = r34/r24 - r23
!
  if (aa.eq.C0P0) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop boxf33: ' &
       ,'threshold singularity, returning 0'
    rslt = C0P0
    return
  endif
!
  bb = -r13*d24 + k12*r34 - k14*r23
  cc = k12*r13 + r24*r34 - k14*r24*r13 - r23
  call solabc(x1,x2,dd ,aa,bb,cc ,0)
  x1 = -x1
  x2 = -x2
!
  qx1 = qonv(x1 ,1 ) ! x1 SHOULD HAVE im. part
  qx2 = qonv(x2 ,1 ) ! x2 SHOULD HAVE im. part
  q12 = qonv(r12,-1)
  q13 = qonv(r13,-1)
  q14 = qonv(r14,-1)
  q23 = qonv(r23,-1)
  q24 = qonv(r24,-1)
  q34 = qonv(r34,-1)
!
  rslt = C0P0
!
  qy1 = qx1/q24
  qy2 = qx2/q24
  rslt(0) = rslt(0) + li2c2( qy1*q12 ,qy2*q12 )/r24*r12
  rslt(0) = rslt(0) + li2c2( qy1/q12 ,qy2/q12 )/r24/r12
  rslt(0) = rslt(0) - li2c2( qx1*q14 ,qx2*q14 )*r14
  rslt(0) = rslt(0) - li2c2( qx1/q14 ,qx2/q14 )/r14
!
  if (.not.r13zero) then
    if (.not.r23zero) then
      qss = q23/q13/q24
      rslt(0) = rslt(0) - li2c2( qx1*qss ,qx2*qss )*r23/(r13*r24)
    endif
    if (.not.r34zero) then
      qss = q34/q13
      rslt(0) = rslt(0) + li2c2( qx1*qss ,qx2*qss )*r34/r13
    endif
  else
    rslt(0) = rslt(0) - logc2( qx1/qx2 )*logc( q23/q24/q34 )/x2 
  endif
!
  rslt(0) = -rslt(0)/(aa*sm1*sm2*sm3*sm4)
  end subroutine


  subroutine boxf4( rslt ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4 )
!*******************************************************************
! Finite 1-loop scalar 4-point function with all internal masses
! non-zero. Based on the formulas from
! A. Denner, U. Nierste, R. Scharf, Nucl.Phys.B367(1991)637-656
!*******************************************************************
  use avh_olo_mp_li2c2 ,only: li2c2
  include 'cts_mpc.h'
   ,intent(out) :: rslt(0:2) 
  include 'cts_mpc.h'
   ,intent(in) :: p1,p2,p3,p4,p12,p23,m1,m2,m3,m4
  type(qmplx_type) :: q12,q13,q14,q23,q24,q34,qx1,qx2,qy1,qy2,qtt
  include 'cts_mpc.h'
   :: sm1,sm2,sm3,sm4 ,aa,bb,cc,dd,x1,x2,tt &
                    ,k12,k13,k14,k23,k24,k34 &
                    ,r12,r13,r14,r23,r24,r34 &
                    ,d12,d13,d14,d23,d24,d34
  include 'cts_mpr.h'
   :: h1,h2
!
!  write(*,*) 'MESSAGE from OneLOop boxf4: you are calling me' !CALLINGME
!
  sm1 = mysqrt(m1)
  sm2 = mysqrt(m2)
  sm3 = mysqrt(m3)
  sm4 = mysqrt(m4)
!
  k12 = C0P0
  k13 = C0P0
  k14 = C0P0
  k23 = C0P0
  k24 = C0P0
  k34 = C0P0
  if (m1+m2.ne.p1 ) k12 = ( m1 + m2 - p1 *oieps)/(sm1*sm2) ! p1
  if (m1+m3.ne.p12) k13 = ( m1 + m3 - p12*oieps)/(sm1*sm3) ! p1+p2
  if (m1+m4.ne.p4 ) k14 = ( m1 + m4 - p4 *oieps)/(sm1*sm4) ! p1+p2+p3
  if (m2+m3.ne.p2 ) k23 = ( m2 + m3 - p2 *oieps)/(sm2*sm3) ! p2
  if (m2+m4.ne.p23) k24 = ( m2 + m4 - p23*oieps)/(sm2*sm4) ! p2+p3
  if (m3+m4.ne.p3 ) k34 = ( m3 + m4 - p3 *oieps)/(sm3*sm4) ! p3
!
  call rfun( r12,d12 ,k12 )
  call rfun( r13,d13 ,k13 )
  call rfun( r14,d14 ,k14 )
  call rfun( r23,d23 ,k23 )
  call rfun( r24,d24 ,k24 )
  call rfun( r34,d34 ,k34 )
!
  aa = k34/r24 + r13*k12 - k14*r13/r24 - k23
!
  if (aa.eq.C0P0) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop boxf4: ' &
       ,'threshold singularity, returning 0'
    rslt = C0P0
    return
  endif
!
  bb = d13*d24 + k12*k34 - k14*k23
  cc = k12/r13 + r24*k34 - k14*r24/r13 - k23
  call solabc(x1,x2,dd ,aa,bb,cc ,0)
!
  h1 = mp_rl(k23 - r13*k12 - r24*k34 + r13*r24*k14)
  h2 = h1*mp_rl(aa)*mp_rl(x1)
  h1 = h1*mp_rl(aa)*mp_rl(x2)
!
  qx1 = qonv(-x1,-h1) ! x1 should have im. part
  qx2 = qonv(-x2,-h2) ! x2 should have im. part
  q12 = qonv(r12,-1)
  q13 = qonv(r13,-1)
  q14 = qonv(r14,-1)
  q23 = qonv(r23,-1)
  q24 = qonv(r24,-1)
  q34 = qonv(r34,-1)
!
  rslt = C0P0
!
  qy1 = qx1/q24
  qy2 = qx2/q24
  rslt(0) = rslt(0) + ( li2c2( qy1*q12 ,qy2*q12 )*r12 &
                      + li2c2( qy1/q12 ,qy2/q12 )/r12 )/r24
  tt = r13/r24
  qtt = qonv(tt,-mp_rl(r24) )
  qy1 = qx1*qtt
  qy2 = qx2*qtt
  rslt(0) = rslt(0) - ( li2c2( qy1*q23 ,qy2*q23 )*r23 &
                      + li2c2( qy1/q23 ,qy2/q23 )/r23 )*tt
  qy1 = qx1*q13
  qy2 = qx2*q13
  rslt(0) = rslt(0) + ( li2c2( qy1*q34 ,qy2*q34 )*r34 &
                      + li2c2( qy1/q34 ,qy2/q34 )/r34 )*r13
!
  rslt(0) = rslt(0) - ( li2c2( qx1*q14 ,qx2*q14 )*r14 &
                      + li2c2( qx1/q14 ,qx2/q14 )/r14 )
!
  rslt(0) = -rslt(0)/(aa*sm1*sm2*sm3*sm4)
  end subroutine

end module


module avh_olo_mp_boxc
   use avh_olo_mp_kinds
   use avh_olo_mp_units
   use avh_olo_mp_func
   implicit none
   private
   public :: boxc,init_boxc
   include 'cts_mpr.h'
    :: thrss3fun
   integer :: ndigits=0
contains

   subroutine init_boxc(ndig)
   integer ,intent(in) :: ndig
   thrss3fun= mp_eps*1000
   if (ndigits.eq.ndig) return ;ndigits=ndig
   if     (ndigits.lt.16) then ;thrss3fun = mp_eps*1000
   elseif (ndigits.lt.24) then ;thrss3fun = mp_eps*30000
   else                        ;thrss3fun = mp_eps*1000000
   endif
   end subroutine
   

   subroutine boxc( rslt ,pp_in,mm_in ,ap_in )
!*******************************************************************
! Finite 1-loop scalar 4-point function for complex internal masses
! Based on the formulas from
!   Dao Thi Nhung and Le Duc Ninh, arXiv:0902.0325 [hep-ph]
!   G. 't Hooft and M.J.G. Veltman, Nucl.Phys.B153:365-401,1979 
!*******************************************************************
   use avh_olo_mp_box ,only: base,casetable,ll=>permtable
   include 'cts_mpc.h'
    ,intent(out)   :: rslt(0:2)
   include 'cts_mpc.h'
    ,intent(inout) :: pp_in(6),mm_in(4)
   include 'cts_mpr.h'
    ,intent(in)  :: ap_in(6)
   include 'cts_mpc.h'
    :: pp(6),mm(4)
   include 'cts_mpr.h'
    :: ap(6),aptmp(6),rem,imm,hh
   include 'cts_mpc.h'
    :: a,b,c,d,e,f,g,h,j,k,dpe,epk,x1,x2,sdnt,o1,j1,e1
   integer :: icase,jcase,ii,jj
   integer ,parameter :: lp(6,3)=&
            reshape((/1,2,3,4,5,6 ,5,2,6,4,1,3 ,1,6,3,5,4,2/),(/6,3/))
   integer ,parameter :: lm(4,3)=&
            reshape((/1,2,3,4     ,1,3,2,4     ,1,2,4,3    /),(/4,3/))
   include 'cts_mpr.h'
    :: small
!
!  write(*,*) 'MESSAGE from OneLOop boxc: you are calling me' !CALLINGME
!
   small= mp_eps
   rslt = C0P0
!
   hh = R0P0
   do ii=1,6
     aptmp(ii) = ap_in(ii)
     if (aptmp(ii).gt.hh) hh = aptmp(ii)
   enddo
   hh = 100*small*hh
   do ii=1,6
     if (aptmp(ii).lt.hh) aptmp(ii) = R0P0
   enddo
!
   if (aptmp(5).eq.R0P0.or.aptmp(6).eq.R0P0) then
     if (aptmp(1).eq.R0P0.or.aptmp(3).eq.R0P0) then
       if (aptmp(2).eq.R0P0.or.aptmp(4).eq.R0P0) then
         if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop boxc: ' &
           ,'no choice with |s| and |t| large enough, putting them by hand'
         if (aptmp(5).eq.R0P0) then
           aptmp(5) = hh
           pp_in(5) = mp_cx(mp_sn(hh,mp_rl(pp_in(5))),R0P0)
         endif
         if (aptmp(6).eq.R0P0) then
           aptmp(6) = hh
           pp_in(6) = mp_cx(mp_sn(hh,mp_rl(pp_in(6))),R0P0)
         endif
         jj = 1
       else
         jj = 3
       endif
     else
       jj = 2
     endif
   else
     jj = 1
   endif
   do ii=1,6
     ap(ii) = aptmp(lp(ii,jj))
     if (ap(ii).gt.R0P0) then ;pp(ii) = pp_in(lp(ii,jj))
     else                     ;pp(ii) = C0P0
     endif
   enddo
   do ii=1,4
     rem =  mp_rl(mm_in(lm(ii,jj)))
     imm = mp_ig(mm_in(lm(ii,jj)))
     hh = small*abs(rem)
     if (abs(imm).lt.hh) imm = -hh
     mm(ii) = mp_cx(rem,imm)
   enddo
!
   icase = 0
   do ii=1,4
     if (ap(ii).gt.R0P0) icase = icase + base(ii)
   enddo
!
   if (icase.lt.15) then
! at least one exernal mass equal zero
     jcase = casetable(icase)
     if (jcase.eq.0.or.jcase.eq.1.or.jcase.eq.5) then
! two opposite masses equal zero
       a = pp(ll(5,icase)) - pp(ll(1,icase))
       c = pp(ll(4,icase)) - pp(ll(5,icase)) - pp(ll(3,icase))
       g = pp(ll(2,icase))
       h = pp(ll(6,icase)) - pp(ll(2,icase)) - pp(ll(3,icase))
       d = (mm(ll(3,icase)) - mm(ll(4,icase))) - pp(ll(3,icase))
       e = (mm(ll(1,icase)) - mm(ll(3,icase))) + pp(ll(3,icase)) - pp(ll(4,icase))
       f = mm(ll(4,icase))
       j = (mm(ll(2,icase)) - mm(ll(3,icase))) - pp(ll(6,icase)) + pp(ll(3,icase))
       rslt(0) = t13fun( a,c,g,h ,d,e,f,j )
     else
       a = pp(ll(3,icase))
       b = pp(ll(2,icase))
       c = pp(ll(6,icase)) - pp(ll(2,icase)) - pp(ll(3,icase))
       h = pp(ll(4,icase)) - pp(ll(5,icase)) - pp(ll(6,icase)) + pp(ll(2,icase))
       j = pp(ll(5,icase)) - pp(ll(1,icase)) - pp(ll(2,icase))
       d = (mm(ll(3,icase)) - mm(ll(4,icase))) - pp(ll(3,icase))
       e = (mm(ll(2,icase)) - mm(ll(3,icase))) - pp(ll(6,icase)) + pp(ll(3,icase))
       k = (mm(ll(1,icase)) - mm(ll(2,icase))) + pp(ll(6,icase)) - pp(ll(4,icase))
       f = mm(ll(4,icase))
       epk = (mm(ll(1,icase)) - mm(ll(3,icase))) + pp(ll(3,icase)) - pp(ll(4,icase))
       rslt(0) = tfun( a,b  ,c  ,h,j ,d,e  ,f,k ) &
               - tfun( a,b+j,c+h,h,j ,d,epk,f,k )
     endif
   else
! no extenal mass equal zero
     if    (mp_rl((pp(5)-pp(1)-pp(2))**2-4*pp(1)*pp(2)).gt.R0P0)then ;icase=0 !12, no permutation
     elseif(mp_rl((pp(6)-pp(2)-pp(3))**2-4*pp(2)*pp(3)).gt.R0P0)then ;icase=8 !23, 1 cyclic permutation
     elseif(mp_rl((pp(4)-pp(5)-pp(3))**2-4*pp(5)*pp(3)).gt.R0P0)then ;icase=4 !34, 2 cyclic permutations
     elseif(mp_rl((pp(4)-pp(1)-pp(6))**2-4*pp(1)*pp(6)).gt.R0P0)then ;icase=2 !41, 3 cyclic permutations
     else
       if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop boxc: ' &
         ,'no positive lambda, returning 0'
       return
     endif
     a = pp(ll(3,icase))
     b = pp(ll(2,icase))
     g = pp(ll(1,icase))
     c = pp(ll(6,icase)) - pp(ll(2,icase)) - pp(ll(3,icase))
     h = pp(ll(4,icase)) - pp(ll(5,icase)) - pp(ll(6,icase)) + pp(ll(2,icase))
     j = pp(ll(5,icase)) - pp(ll(1,icase)) - pp(ll(2,icase))
     d = (mm(ll(3,icase)) - mm(ll(4,icase))) - pp(ll(3,icase))
     e = (mm(ll(2,icase)) - mm(ll(3,icase))) - pp(ll(6,icase)) + pp(ll(3,icase))
     k = (mm(ll(1,icase)) - mm(ll(2,icase))) + pp(ll(6,icase)) - pp(ll(4,icase))
     f = mm(ll(4,icase))
     dpe = (mm(ll(2,icase)) - mm(ll(4,icase))) - pp(ll(6,icase))
     epk = (mm(ll(1,icase)) - mm(ll(3,icase))) + pp(ll(3,icase)) - pp(ll(4,icase))
     call solabc( x1,x2 ,sdnt ,g,j,b ,0 )
     if (mp_ig(sdnt).ne.R0P0) then
       if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop boxc: ' &
         ,'no real solution for alpha, returning 0'
       return
     endif
!BAD        if (abs(real(x1)).gt.abs(real(x2))) then
     if (abs(mp_rl(x1)).lt.abs(mp_rl(x2))) then !BETTER
       sdnt = x1
       x1 = x2
       x2 = sdnt
     endif
     o1 = C1P0-x1
     j1 = j+2*g*x1
     e1 = e+k*x1
     rslt(0) =   -tfun( a+b+c,g    ,j+h,c+2*b+(h+j)*x1,j1    ,dpe,k  ,f,e1 ) &
             + o1*tfun( a    ,b+g+j,c+h,c+h*x1        ,o1*j1 ,d  ,epk,f,e1 ) &
             + x1*tfun( a    ,b    ,c  ,c+h*x1        ,-j1*x1,d  ,e  ,f,e1 )
   endif
   end subroutine


   function t13fun( aa,cc,gg,hh ,dd,ee,ff,jj ) result(rslt)
!*******************************************************************
! /1   /x                             y
! | dx |  dy -----------------------------------------------------
! /0   /0    (gy^2 + hxy + dx + jy + f)*(ax^2 + cxy + dx + ey + f)
!
! jj should have negative imaginary part
!*******************************************************************
   include 'cts_mpc.h'
    ,intent(in) :: aa,cc,gg,hh ,dd,ee,ff,jj
   include 'cts_mpc.h'
    :: rslt ,kk,ll,nn,y1,y2,sdnt,ieps
   include 'cts_mpr.h'
    :: small
!
!  write(*,*) 'MESSAGE from OneLOop t13fun: you are calling me' !CALLINGME
!
   small=mp_eps**2
   ieps = mp_cx(R0P0,small*abs(mp_rl(ff)))
!
   kk = hh*aa - cc*gg
   ll = aa*dd + hh*ee - dd*gg - cc*jj
   nn = dd*(ee - jj) + (hh - cc)*(ff-ieps)
   call solabc( y1,y2 ,sdnt ,kk,ll,nn ,0 )
!
   rslt = - s3fun( y1,y2 ,C0P0,C1P0 ,aa   ,ee+cc,dd+ff ) &
          + s3fun( y1,y2 ,C0P0,C1P0 ,gg   ,jj+hh,dd+ff ) &
          - s3fun( y1,y2 ,C0P0,C1P0 ,gg+hh,dd+jj,ff    ) &
          + s3fun( y1,y2 ,C0P0,C1P0 ,aa+cc,ee+dd,ff    )
!
   rslt = rslt/kk
   end function


   function t1fun( aa,cc,gg,hh ,dd,ee,ff,jj ) result(rslt)
!*******************************************************************
! /1   /x                         1
! | dx |  dy ----------------------------------------------
! /0   /0    (g*x + h*x + j)*(a*x^2 + c*xy + d*x + e*y + f)
!
! jj should have negative imaginary part
!*******************************************************************
   include 'cts_mpc.h'
    ,intent(in) :: aa,cc,gg,hh ,dd,ee,ff,jj
   include 'cts_mpc.h'
    ::rslt ,kk,ll,nn,y1,y2,sdnt,ieps
   include 'cts_mpr.h'
    :: small
!
!  write(*,*) 'MESSAGE from OneLOop t1fun: you are calling me' !CALLINGME
!
   small= mp_eps**2
   ieps = mp_cx(R0P0,small*abs(mp_rl(ff)))
!
   kk = hh*aa - cc*gg
   ll = hh*dd - cc*jj - ee*gg
   nn = hh*(ff-ieps) - ee*jj
   call solabc( y1,y2 ,sdnt ,kk,ll,nn ,0 )
!
   rslt = - s3fun( y1,y2 ,C0P0,C1P0 ,aa+cc,dd+ee,ff ) &
          + s3fun( y1,y2 ,C0P0,C1P0 ,C0P0 ,gg+hh,jj ) &
          - s3fun( y1,y2 ,C0P0,C1P0 ,C0P0 ,gg   ,jj ) &
          + s3fun( y1,y2 ,C0P0,C1P0 ,aa   ,dd   ,ff )
!
   rslt = rslt/kk
   end function


   function tfun( aa,bb,cc ,gin,hin ,dd,ee,ff ,jin ) result(rslt)
!*******************************************************************
! /1   /x                             1
! | dx |  dy ------------------------------------------------------
! /0   /0    (g*x + h*x + j)*(a*x^2 + b*y^2 + c*xy + d*x + e*y + f)
!*******************************************************************
   include 'cts_mpc.h'
    ,intent(in) :: aa,bb,cc ,gin,hin ,dd,ee,ff ,jin
   include 'cts_mpc.h'
    :: rslt ,gg,hh,jj,zz(2),beta,tmpa(2),tmpb(2) &
                  ,tmpc(2),kiz(2),ll,nn,kk,y1,y2,yy(2,2),sdnt,ieps
   include 'cts_mpr.h'
    :: sj,ab1,ab2,ac1,ac2,abab,acac,abac,det,ap1,ap2 &
                  ,apab,apac,x1(2,2),x2(2,2),xmin
   integer :: iz,iy,izmin
   logical :: pp(2,2),p1,p2
   include 'cts_mpr.h'
    :: small
!
!  write(*,*) 'MESSAGE from OneLOop tfun: you are calling me' !CALLINGME
!
   small= mp_eps**2
   sj = mp_ig(jin)
   if (sj.eq.R0P0) then
     sj = -R1P0
   else
     sj = mp_sn(R1P0,mp_ig(jin))
   endif
   gg = -sj*gin
   hh = -sj*hin
   jj = -sj*jin
!
   if     (bb.eq.C0P0) then
     rslt = -sj*t1fun( aa,cc,gg,hh ,dd,ee,ff,jj )
     return
   elseif (aa.eq.C0P0) then
     rslt = -sj*t1fun( bb+cc,-cc,-gg-hh,gg,-dd-ee-2*(bb+cc),dd+cc,dd+ee+bb+cc+ff,gg+hh+jj )
     return
   endif
!
   ieps = mp_cx(R0P0,small*abs(mp_rl(ff)))
!
   call solabc( zz(1),zz(2) ,sdnt ,bb,cc,aa ,0 )
   if (abs(zz(1)).gt.abs(zz(2))) then
     beta = zz(1)
     zz(1) = zz(2)
     zz(2) = beta
   endif
!
   do iz=1,2
     beta = zz(iz)
     tmpa(iz) = gg + beta*hh
     tmpb(iz) = cc + 2*beta*bb
     tmpc(iz) = dd + beta*ee
     kiz(iz) =        bb*tmpa(iz)               - hh*tmpb(iz)
     ll      =        ee*tmpa(iz) - hh*tmpc(iz) - jj*tmpb(iz)
     nn      = (ff-ieps)*tmpa(iz) - jj*tmpc(iz)
     call solabc( yy(iz,1),yy(iz,2) ,sdnt ,kiz(iz),ll,nn ,0 )
     if (abs(mp_ig(beta)).ne.R0P0) then
       ab1 =  mp_rl(-beta)
       ab2 = mp_ig(-beta)
       ac1 = ab1+R1P0 !real(C1P0-beta)
       ac2 = ab2     !aimag(C1P0-beta)
       abab = ab1*ab1 + ab2*ab2
       acac = ac1*ac1 + ac2*ac2
       abac = ab1*ac1 + ab2*ac2
       det = abab*acac - abac*abac
       do iy=1,2
         ap1 =  mp_rl(yy(iz,iy))
         ap2 = mp_ig(yy(iz,iy))
         apab = ap1*ab1 + ap2*ab2
         apac = ap1*ac1 + ap2*ac2
         x1(iz,iy) = ( acac*apab - abac*apac )/det
         x2(iz,iy) = (-abac*apab + abab*apac )/det
       enddo
     else
       do iy=1,2
         x1(iz,iy) = -R1P0
         x2(iz,iy) = -R1P0
       enddo
     endif
   enddo
   xmin = R1P0
   izmin = 2
   do iz=1,2
   do iy=1,2
     if ( x1(iz,iy).ge.R0P0.and.x2(iz,iy).ge.R0P0 &
                 .and.x1(iz,iy)+x2(iz,iy).le.R1P0 ) then
       pp(iz,iy) = .true.
       if (x1(iz,iy).lt.xmin) then
         xmin = x1(iz,iy)
         izmin = iz
       endif
       if (x2(iz,iy).lt.xmin) then
         xmin = x2(iz,iy)
         izmin = iz
       endif
     else
       pp(iz,iy) = .false.
     endif
   enddo
   enddo
   iz = izmin+1
   if (iz.eq.3) iz = 1
!
   beta = zz(iz)
   kk = kiz(iz)
   y1 = yy(iz,1)
   y2 = yy(iz,2)
   p1 = pp(iz,1)
   p2 = pp(iz,2)
!
   rslt =   s3fun( y1,y2 ,beta ,C1P0      ,C0P0    ,hh   ,gg+jj    ) &
          - s3fun( y1,y2 ,C0P0 ,C1P0-beta ,C0P0    ,gg+hh,   jj    ) &
          + s3fun( y1,y2 ,C0P0 ,    -beta ,C0P0    ,gg   ,   jj    ) &
          - s3fun( y1,y2 ,beta ,C1P0      ,bb      ,cc+ee,aa+dd+ff ) &
          + s3fun( y1,y2 ,C0P0 ,C1P0-beta ,aa+bb+cc,dd+ee,ff       ) &
          - s3fun( y1,y2 ,C0P0 ,    -beta ,aa      ,dd   ,ff       )
!
   sdnt = plnr( y1,y2 ,p1,p2, tmpa(iz),tmpb(iz),tmpc(iz) )
   if (mp_ig(beta).le.R0P0) then ;rslt = rslt + sdnt
   else                          ;rslt = rslt - sdnt
   endif
!
   rslt = -sj*rslt/kk
   end function


   function s3fun( y1i,y2i ,dd,ee ,aa,bb,cin ) result(rslt)
!*******************************************************************
! Calculate
!            ( S3(y1i) - S3(y2i) )/( y1i - y2i )
! where
!               /1    ee * ln( aa*x^2 + bb*x + cc )
!       S3(y) = |  dx -----------------------------
!               /0           ee*x - y - dd
!
! y1i,y2i should have a non-zero imaginary part
!*******************************************************************
   use avh_olo_mp_logc ,only: logc
   include 'cts_mpc.h'
    ,intent(in) ::  y1i,y2i ,dd,ee ,aa,bb,cin
   include 'cts_mpc.h'
    :: rslt ,y1,y2,fy1y2,z1,z2,tmp,cc
   include 'cts_mpr.h'
    ::rea,reb,rez1,rez2,imz1,imz2,simc
   include 'cts_mpr.h'
    :: small
!
   small=mp_eps**2
   if (ee.eq.C0P0) then
     rslt = C0P0
     return
   endif
!
   cc = cin
   rea = abs(aa)
   reb = abs(bb)
   simc = abs(cc)
   if (simc.lt.thrss3fun*min(rea,reb)) cc = C0P0
!
   simc = mp_ig(cc)
   if (simc.eq.R0P0) then
     simc = mp_ig(bb)
     if (simc.eq.R0P0) simc = -R1P0
   endif
   simc = mp_sn(R1P0,simc)
!
   y1 = (dd+y1i)/ee
   y2 = (dd+y2i)/ee
   if (mp_ig(y1).eq.R0P0) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop s3fun: ' &
       ,'y1 has zero imaginary part'
   endif
   if (mp_ig(y2).eq.R0P0) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop s3fun: ' &
       ,'y2 has zero imaginary part'
   endif
   fy1y2 = r0fun( y1,y2 )
!
   if     (aa.ne.C0P0) then
!
     call solabc( z1,z2 ,tmp ,aa,bb,cc ,0 )
     rea  = mp_sn(R1P0,mp_rl(aa))
     rez1 = mp_rl(z1)
     rez2 = mp_rl(z2) 
     imz1 = mp_ig(z1) ! sign(Im(a*z1*z2)) = simc
     imz2 = mp_ig(z2)
     if (imz1.eq.R0P0) imz1 = simc*rea*mp_sn(R1P0,rez2)*abs(small*rez1)
     if (imz2.eq.R0P0) imz2 = simc*rea*mp_sn(R1P0,rez1)*abs(small*rez2)
     z1 = mp_cx( rez1,imz1)
     z2 = mp_cx( rez2,imz2)
     rslt = fy1y2 * ( logc(qonv(aa,simc)) &
                    + eta3( -z1,-imz1,-z2,-imz2,C0P0,simc*rea ) ) &
          + r1fun( z1,y1,y2,fy1y2 ) &
          + r1fun( z2,y1,y2,fy1y2 )
!
   elseif (bb.ne.C0P0) then
!
     z1 = -cc/bb ! - i|eps|Re(b)
     reb  = mp_rl(bb)
     rez1 = mp_rl(z1)
     imz1 = mp_ig(z1)
     if (abs(imz1).eq.R0P0) then
       imz1 = -simc*reb*abs(small*rez1/reb)
       z1 = mp_cx( rez1,imz1)
     endif
     rslt = fy1y2 * ( logc(qonv(bb,simc)) &
                    + eta3(bb,simc ,-z1,-imz1 ,cc,simc) ) &
          + r1fun( z1,y1,y2,fy1y2 )
!
   elseif (cc.ne.C0P0) then
!
     rslt = logc( qonv(cc,simc) )*fy1y2
!
   else!if (aa=bb=cc=0)
!
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop s3fun: ' &
       ,'cc equal zero, returning 0'
     rslt = C0P0
!
   endif
!
   rslt = rslt/ee
   end function


   function r1fun( zz,y1,y2,fy1y2 ) result(rslt)
!*******************************************************************
! calculates  ( R1(y1,z) - R1(y2,z) )/( y1 - y2 )
! where
!                          /     / 1-y \       / 1-z \ \
!      R1(y,z) = ln(y-z) * | log |-----| - log |-----| |
!                          \     \ -y  /       \ -z  / / 
!
!                      /    y-z \       /    y-z \
!                - Li2 |1 - ----| + Li2 |1 - ----|
!                      \    -z  /       \    1-z /
!
!                                     / 1-y1 \       / 1-y2 \
!                                 log |------| - log |------| 
! input fy1y2 should be equal to      \  -y1 /       \  -y2 /
!                                 ---------------------------
!                                           y1 - y2
!*******************************************************************
   use avh_olo_mp_logc ,only: logc
   use avh_olo_mp_li2c ,only: li2c
   use avh_olo_mp_logc2 ,only: logc2
   use avh_olo_mp_li2c2 ,only: li2c2
   include 'cts_mpc.h'
    ,intent(in) :: y1,y2,zz,fy1y2
   include 'cts_mpc.h'
    :: rslt ,oz
   type(qmplx_type) :: q1z,q2z,qq
   include 'cts_mpr.h'
    :: h12,hz1,hz2,hzz,hoz
   logical :: zzsmall,ozsmall
!
   oz = C1P0-zz
   h12 = abs(y1-y2)
   hz1 = abs(y1-zz)
   hz2 = abs(y2-zz)
   hzz = abs(zz)
   hoz = abs(oz)
   q1z = qonv(y1-zz)
   q2z = qonv(y2-zz)
!
   zzsmall = .false.
   ozsmall = .false.
   if     (hzz.lt.hz1.and.hzz.lt.hz2.and.hzz.lt.hoz) then ! |z| < |y1-z|,|y2-z|
     zzsmall = .true.
     rslt = fy1y2*logc( q1z ) &
          - ( logc(q1z*q2z)/2 + logc(qonv((y2-C1P0)/y2)) &
                                        - logc(qonv(oz)) )*logc2(q1z/q2z)/(y2-zz)
   elseif (hoz.lt.hz1.and.hoz.lt.hz2) then ! |1-z| < |y1-z|,|y2-z|
     ozsmall = .true.
     rslt = fy1y2*logc( q1z ) &
          - (-logc(q1z*q2z)/2 + logc(qonv((y2-C1P0)/y2)) &
                                       + logc(qonv(-zz)) )*logc2(q1z/q2z)/(y2-zz)
   elseif (h12.le.hz2.and.hz2.le.hz1) then ! |y1-y2| < |y2-z| < |y1-z|
     rslt = fy1y2*logc( q1z ) - r0fun( y2,zz )*logc2( q1z/q2z )        
   elseif (h12.le.hz1.and.hz1.le.hz2) then ! |y1-y2| < |y2-z| < |y1-z|
     rslt = fy1y2*logc( q2z ) - r0fun( y1,zz )*logc2( q2z/q1z )        
   else!if(hz1.lt.h12.or.hz2.lt.h12) then ! |y2-z|,|y1-z| < |y1-y2|
     rslt = C0P0
     if (hz1.ne.R0P0) rslt = rslt + (y1-zz)*logc( q1z )*r0fun( y1,zz )
     if (hz2.ne.R0P0) rslt = rslt - (y2-zz)*logc( q2z )*r0fun( y2,zz )
     rslt = rslt/(y1-y2)
   endif
!
   if (zzsmall) then ! |z| < |y1-z|,|y2-z|
     qq  = qonv(-zz)
     rslt = rslt + ( li2c( qq/q1z ) - li2c( qq/q2z ) )/(y1-y2)
   else
     qq  = qonv(-zz)
     rslt = rslt + li2c2( q1z/qq ,q2z/qq )/zz
   endif
!
   if (ozsmall) then ! |1-z| < |y1-z|,|y2-z|
     qq  = qonv(oz)
     rslt = rslt - ( li2c( qq/q1z ) - li2c( qq/q2z ) )/(y1-y2)
   else
     qq = qonv(oz)
     rslt = rslt + li2c2( q1z/qq ,q2z/qq )/oz
   endif
   end function


   function r0fun( y1,y2 ) result(rslt)
!*******************************************************************
!      / 1-y1 \       / 1-y2 \
!  log |------| - log |------| 
!      \  -y1 /       \  -y2 /
!  ---------------------------
!            y1 - y2
!
! y1,y2 should have non-zero imaginary parts
!*******************************************************************
   use avh_olo_mp_logc2 ,only: logc2
   include 'cts_mpc.h'
    ,intent(in) :: y1,y2
   include 'cts_mpc.h'
    :: rslt ,oy1,oy2
   oy1 = C1P0-y1
   oy2 = C1P0-y2
   rslt = logc2( qonv(-y2)/qonv(-y1) )/y1 &
        + logc2( qonv(oy2)/qonv(oy1) )/oy1
   end function


   function plnr( y1,y2 ,p1,p2 ,aa,bb,cc ) result(rslt)
!*******************************************************************
!                   /   a    \          /   a    \
!            p1*log |--------| - p2*log |--------| 
!                   \ b*y1+c /          \ b*y2+c /
! 2*pi*imag* -------------------------------------
!                           y1 - y2
! 
! p1,p2 are logical, to be interpreted as 0,1 in the formula above 
!*******************************************************************
   use avh_olo_mp_logc ,only: logc
   use avh_olo_mp_logc2 ,only: logc2
   include 'cts_mpc.h'
    ,intent(in) :: y1,y2 ,aa,bb,cc
   logical         ,intent(in) :: p1,p2
   include 'cts_mpc.h'
    :: rslt ,x1,x2,xx
   type(qmplx_type) :: q1,q2
   include 'cts_mpc.h'
    :: twopii
!
   twopii=CiP0*TWOPI 
   if (p1) then
     x1 = bb*y1 + cc
     xx = aa/x1
     if (mp_ig(xx).eq.R0P0) then
       if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop plnr: ' &
         ,'aa/x1 has zero imaginary part'
     endif
     q1 = qonv(xx)
   endif
   if (p2) then
     x2 = bb*y2 + cc
     xx = aa/x2
     if (mp_ig(xx).eq.R0P0) then
       if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop plnr: ' &
         ,'aa/x2 has zero imaginary part'
     endif
     q2 = qonv(xx)
   endif
   if (p1) then
     if (p2) then
       rslt = logc2( q2/q1 ) * twopii*bb/x2
     else
       rslt = logc( q1 ) * twopii/(y1-y2)
     endif
   elseif (p2) then
     rslt = logc( q2 ) * twopii/(y2-y1) ! minus sign
   else
     rslt = C0P0
   endif
   end function


end module


module avh_olo_mp
  include 'cts_mprec.h'
  use avh_olo_mp_kinds
  use avh_olo_mp_units
  use avh_olo_mp_print
!
  implicit none
  private
  public :: olo_mp_kind ,olo_mp_unit ,olo_mp_scale ,olo_mp_onshell ,olo_mp_setting &
           ,olo_mp_a0 ,olo_mp_b0 ,olo_mp_b11 ,olo_mp_c0 ,olo_mp_d0
  private :: mp_radix,mp_digits
!
  integer ,parameter :: olo_mp_kind = kindr2
!
  integer      :: ndigits = 0        ! corrected in subroutine hello
  include 'cts_mpr.h'
   :: onshellthrs ! set in subroutine hello
  logical      :: nonzerothrs = .false.
!
  include 'cts_mpr.h'
   :: muscale   ! set in subroutine hello
!
  character(99) ,parameter :: warnonshell=&
       'it seems you forgot to put some input explicitly on shell. ' &
     //'You may  call olo_mp_onshell  to cure this.'
!
  logical :: intro=.true.
!
  interface olo_mp_a0
    module procedure a0r,a0rr,a0c,a0cr
  end interface 
  interface olo_mp_b0
    module procedure b0rr,b0rrr,b0rc,b0rcr,b0cc,b0ccr
  end interface 
  interface olo_mp_b11
    module procedure b11rr,b11rrr,b11rc,b11rcr,b11cc,b11ccr
  end interface 
  interface olo_mp_c0
    module procedure c0rr,c0rrr,c0rc,c0rcr,c0cc,c0ccr
  end interface 
  interface olo_mp_d0
    module procedure d0rr,d0rrr,d0rc,d0rcr,d0cc,d0ccr
  end interface 
!
  interface radix
   procedure mp_radix
  end interface
!
  interface digits
   procedure mp_digits
  end interface
!
contains
  function mp_radix(arg)
  type(mp_real), intent(in) :: arg
  integer :: mp_radix
  mp_radix= 2
  end function
!
  function mp_digits(arg)
  type(mp_real), intent(in) :: arg
  integer :: mp_digits
  mp_digits= 64 
  end function
!
  subroutine hello
!*******************************************************************
!*******************************************************************
  use avh_olo_mp_loga2 ,only: init_loga2
  use avh_olo_mp_li2c2 ,only: init_li2c2
  use avh_olo_mp_bub   ,only: init_bub
  use avh_olo_mp_boxc  ,only: init_boxc
!
  intro = .false.
!
  write(*,'(a72)') '########################################################################'
  write(*,'(a72)') '#                                                                      #'
  write(*,'(a72)') '#          You are using OneLOop in multiple precision                 #'
  write(*,'(a72)') '#                                                                      #'
  write(*,'(a72)') '#             obtained by R. Pittau (pittau@ugr.es)                    #'
  write(*,'(a72)') '#             from the original OneLOop-2.2 package                    #'                       
  write(*,'(a72)') '#                                                                      #'
!
  muscale= 1.d0
  call avh_olo_mp_load_constants
  call countdigits(ndigits)
  if (ndigits.gt.50) then
   ndigits = 32
   write(*,'(a72)') '#                 Internal mproutines detected.                        #'
   write(*,'(a72)') '#                                                                      #'
   write(*,'(a72)') '########################################################################'
  else
   ndigits = int(digits(R1P0)*log(radix(R1P0)*R1P0)/log(R1P0*10))
   write(*,'(a17,i4.2,a51)')'#   Compiler with',ndigits,'  significant digits detetected in OneLOop.       #'
   write(*,'(a72)') '#                                                                      #'
   write(*,'(a72)') '########################################################################'
  endif
      if (ndigits.lt.16) then ;onshellthrs = mp_eps*100
  elseif (ndigits.lt.24) then ;onshellthrs = mp_eps*1000
  elseif (ndigits.lt.32) then ;onshellthrs = mp_eps*10000
  else                        ;onshellthrs = mp_eps*1000000
  endif
!
  call init_print( ndigits )
  call init_loga2( ndigits )
  call init_li2c2( ndigits )
  call init_bub(   ndigits )
  call init_boxc(  ndigits )
!
  end subroutine
 
 
  subroutine olo_mp_unit( val ,message )
!*******************************************************************
!*******************************************************************
  integer     ,intent(in) :: val
  character(*),intent(in),optional :: message
  if (intro) call hello
  if (present(message)) then ;call set_unit( message ,val )
  else                       ;call set_unit( 'all'   ,val )
  endif
  end subroutine
 
 
  subroutine olo_mp_scale( val )
!*******************************************************************
!*******************************************************************
  include 'cts_dpr.h'
   ,intent(in) :: val
  if (intro) call hello
  muscale = val
  end subroutine
  
  subroutine olo_mp_onshell( thrs )
!*******************************************************************
!*******************************************************************
  include 'cts_dpr.h'
   ,intent(in) :: thrs
  if (intro) call hello
  nonzerothrs = .true.
  onshellthrs = thrs
  end subroutine


  subroutine olo_mp_setting( iunit )
!*******************************************************************
!*******************************************************************
  integer,optional,intent(in) :: iunit
  integer :: nunit
  if (intro) call hello
  nunit = munit
  if (present(iunit)) nunit = iunit
  if (nunit.le.0) return
!
  write(nunit,*) 'MESSAGE from OneLOop: real kind parameter =',trim(myprint(kindr2))
  write(nunit,*) 'MESSAGE from OneLOop: significant digits =',trim(myprint(ndigits))
!
  if (nonzerothrs) then
    write(nunit,*) 'MESSAGE from OneLOop: on-shell threshold =',trim(myprint(onshellthrs))
  else
    write(nunit,*) 'MESSAGE from OneLOop: on-shell threshold is not set'
  endif
!
  write(nunit,*) 'MESSAGE from OneLOop: scale (mu, not mu^2) =',trim(myprint(muscale))
!
  end subroutine
 
 
!*******************************************************************
!
!           C   / d^(Dim)q
! rslt = ------ | -------- 
!        i*pi^2 / (q^2-mm)
!
! with  Dim = 4-2*eps
!         C = pi^eps * mu^(2*eps) * exp(gamma_Euler*eps)
!
! input:  mm = mass squared
! output: rslt(0) = eps^0   -coefficient
!         rslt(1) = eps^(-1)-coefficient
!         rslt(2) = eps^(-2)-coefficient
!
! Check the comments in  subroutine olo_mp_onshell  to find out how
! this routine decides when to return IR-divergent cases.
!*******************************************************************
!
  subroutine a0r( rslt ,mm  )
!*******************************************************************
!*******************************************************************
  use avh_olo_mp_bub ,only: tadp
  include 'cts_mpc.h'
   ,intent(out) :: rslt(0:2)
  include 'cts_mpr.h'
   ,intent(in)  :: mm
!
  include 'cts_mpc.h'
   :: ss
  include 'cts_mpr.h'
   :: am,thrs,mulocal,mulocal2
  character(25+99) ,parameter :: warning=&
                     'WARNING from OneLOop a0: '//warnonshell
  if (intro) call hello
!
  mulocal = muscale
!
  am = abs(mm)
!
  mulocal2 = mulocal*mulocal
!
  if (nonzerothrs) then
    thrs = onshellthrs
    if (am.lt.thrs) am = R0P0
  elseif (wunit.gt.0) then
    thrs = onshellthrs*max(am,mulocal2)
    if (R0P0.lt.am.and.am.lt.thrs) write(wunit,*) warning
  endif
!
  ss = mm
  call tadp( rslt ,ss ,am ,mulocal2 )
!
  if (punit.gt.0) then
    if (nonzerothrs) write(punit,*) 'onshell:',trim(myprint(onshellthrs))
    write(punit,*) 'muscale:',trim(myprint(mulocal))
    write(punit,*) ' mm:',trim(myprint(mm))
    write(punit,*) 'a0(2):',trim(myprint(rslt(2)))
    write(punit,*) 'a0(1):',trim(myprint(rslt(1)))
    write(punit,*) 'a0(0):',trim(myprint(rslt(0)))
  endif
!
  end subroutine
!
  subroutine a0rr( rslt ,mm ,rmu )
!*******************************************************************
!*******************************************************************
  use avh_olo_mp_bub ,only: tadp
  include 'cts_mpc.h'
   ,intent(out) :: rslt(0:2)
  include 'cts_mpr.h'
   ,intent(in)  :: mm
  include 'cts_mpr.h'
   ,intent(in)  :: rmu
!
  include 'cts_mpc.h'
   :: ss
  include 'cts_mpr.h'
   :: am,thrs,mulocal,mulocal2
  character(25+99) ,parameter :: warning=&
                     'WARNING from OneLOop a0: '//warnonshell
  if (intro) call hello
!
  mulocal = rmu
!
  am = abs(mm)
!
  mulocal2 = mulocal*mulocal
!
  if (nonzerothrs) then
    thrs = onshellthrs
    if (am.lt.thrs) am = R0P0
  elseif (wunit.gt.0) then
    thrs = onshellthrs*max(am,mulocal2)
    if (R0P0.lt.am.and.am.lt.thrs) write(wunit,*) warning
  endif
!
  ss = mm
  call tadp( rslt ,ss ,am ,mulocal2 )
!
  if (punit.gt.0) then
    if (nonzerothrs) write(punit,*) 'onshell:',trim(myprint(onshellthrs))
    write(punit,*) 'muscale:',trim(myprint(mulocal))
    write(punit,*) ' mm:',trim(myprint(mm))
    write(punit,*) 'a0(2):',trim(myprint(rslt(2)))
    write(punit,*) 'a0(1):',trim(myprint(rslt(1)))
    write(punit,*) 'a0(0):',trim(myprint(rslt(0)))
  endif
!
  end subroutine
!
  subroutine a0c( rslt ,mm )
!*******************************************************************
!*******************************************************************
  use avh_olo_mp_bub ,only: tadp
  include 'cts_mpc.h'
   ,intent(out) :: rslt(0:2)
  include 'cts_mpc.h'
   ,intent(in)  :: mm
!
  include 'cts_mpc.h'
   :: ss
  include 'cts_mpr.h'
   :: am,thrs,mulocal,mulocal2
  character(25+99) ,parameter :: warning=&
                     'WARNING from OneLOop a0: '//warnonshell
  if (intro) call hello
!
  mulocal = muscale
!
  am = abs(mm)
!
  mulocal2 = mulocal*mulocal
!
  if (nonzerothrs) then
    thrs = onshellthrs
    if (am.lt.thrs) am = R0P0
  elseif (wunit.gt.0) then
    thrs = onshellthrs*max(am,mulocal2)
    if (R0P0.lt.am.and.am.lt.thrs) write(wunit,*) warning
  endif
!
  ss = mm
  call tadp( rslt ,ss ,am ,mulocal2 )
!
  if (punit.gt.0) then
    if (nonzerothrs) write(punit,*) 'onshell:',trim(myprint(onshellthrs))
    write(punit,*) 'muscale:',trim(myprint(mulocal))
    write(punit,*) ' mm:',trim(myprint(mm))
    write(punit,*) 'a0(2):',trim(myprint(rslt(2)))
    write(punit,*) 'a0(1):',trim(myprint(rslt(1)))
    write(punit,*) 'a0(0):',trim(myprint(rslt(0)))
  endif
!
  end subroutine
!
  subroutine a0cr( rslt ,mm ,rmu )
!*******************************************************************
!*******************************************************************
  use avh_olo_mp_bub ,only: tadp
  include 'cts_mpc.h'
   ,intent(out) :: rslt(0:2)
  include 'cts_mpc.h'
   ,intent(in)  :: mm
  include 'cts_mpr.h'
   ,intent(in)  :: rmu
!
  include 'cts_mpc.h'
   :: ss
  include 'cts_mpr.h'
   :: am,thrs,mulocal,mulocal2
  character(25+99) ,parameter :: warning=&
                     'WARNING from OneLOop a0: '//warnonshell
  if (intro) call hello
!
  mulocal = rmu
!
  am = abs(mm)
!
  mulocal2 = mulocal*mulocal
!
  if (nonzerothrs) then
    thrs = onshellthrs
    if (am.lt.thrs) am = R0P0
  elseif (wunit.gt.0) then
    thrs = onshellthrs*max(am,mulocal2)
    if (R0P0.lt.am.and.am.lt.thrs) write(wunit,*) warning
  endif
!
  ss = mm
  call tadp( rslt ,ss ,am ,mulocal2 )
!
  if (punit.gt.0) then
    if (nonzerothrs) write(punit,*) 'onshell:',trim(myprint(onshellthrs))
    write(punit,*) 'muscale:',trim(myprint(mulocal))
    write(punit,*) ' mm:',trim(myprint(mm))
    write(punit,*) 'a0(2):',trim(myprint(rslt(2)))
    write(punit,*) 'a0(1):',trim(myprint(rslt(1)))
    write(punit,*) 'a0(0):',trim(myprint(rslt(0)))
  endif
!
  end subroutine


!*******************************************************************
!
!           C   /      d^(Dim)q
! rslt = ------ | --------------------
!        i*pi^2 / [q^2-m1][(q+k)^2-m2]
!
! with  Dim = 4-2*eps
!         C = pi^eps * mu^(2*eps) * exp(gamma_Euler*eps)
!
! input:  pp = k^2, m1,m2 = mass squared
! output: rslt(0) = eps^0   -coefficient
!         rslt(1) = eps^(-1)-coefficient
!         rslt(2) = eps^(-2)-coefficient
!
! Check the comments in  subroutine olo_mp_onshell  to find out how
! this routine decides when to return IR-divergent cases.
!*******************************************************************
!
  subroutine b0rr( rslt ,pp ,m1,m2 )
!*******************************************************************
!*******************************************************************
  use avh_olo_mp_bub ,only: bub0
  include 'cts_mpc.h'
   ,intent(out) :: rslt(0:2)
  include 'cts_mpr.h'
   ,intent(in)  :: pp,m1,m2
!
  include 'cts_mpc.h'
   :: ss,r1,r2
  include 'cts_mpr.h'
   :: app,am1,am2,thrs,thrsa,thrsb,mulocal,mulocal2
  character(25+99) ,parameter :: warning=&
                     'WARNING from OneLOop b0: '//warnonshell
  if (intro) call hello
!
  ss = pp
  r1 = m1
  r2 = m2
!
  mulocal = muscale
!
  app = abs(pp)
  am1 = abs(m1)
  am2 = abs(m2)
!
  mulocal2 = mulocal*mulocal
!
  if (nonzerothrs) then
    thrs = onshellthrs
    if (app.lt.thrs) app = R0P0
    if (am1.lt.thrs) am1 = R0P0
    if (am2.lt.thrs) am2 = R0P0
  elseif (wunit.gt.0) then
    thrsa = onshellthrs*max(app,am1)
    thrsb = onshellthrs*max(am2,mulocal2)
    thrs = max(thrsa,thrsb)
    if (R0P0.lt.app.and.app.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.am1.and.am1.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.am2.and.am2.lt.thrs) write(wunit,*) warning
  endif
!
  call bub0( rslt ,ss,r1,r2 ,app,am1,am2 ,mulocal2 )
!
  if (punit.gt.0) then
    if (nonzerothrs) write(punit,*) 'onshell:',trim(myprint(onshellthrs))
    write(punit,*) 'muscale:',trim(myprint(mulocal))
    write(punit,*) ' pp:',trim(myprint(pp))
    write(punit,*) ' m1:',trim(myprint(m1))
    write(punit,*) ' m2:',trim(myprint(m2))
    write(punit,*) 'b0(2):',trim(myprint(rslt(2)))
    write(punit,*) 'b0(1):',trim(myprint(rslt(1)))
    write(punit,*) 'b0(0):',trim(myprint(rslt(0)))
  endif
!
  end subroutine
!
  subroutine b0rrr( rslt ,pp ,m1,m2 ,rmu )
!*******************************************************************
!*******************************************************************
  use avh_olo_mp_bub ,only: bub0
  include 'cts_mpc.h'
   ,intent(out) :: rslt(0:2)
  include 'cts_mpr.h'
   ,intent(in)  :: pp,m1,m2,rmu
!
  include 'cts_mpc.h'
   :: ss,r1,r2
  include 'cts_mpr.h'
   :: app,am1,am2,thrs,thrsa,thrsb,mulocal,mulocal2
  character(25+99) ,parameter :: warning=&
                     'WARNING from OneLOop b0: '//warnonshell
  if (intro) call hello
!
  ss = pp
  r1 = m1
  r2 = m2
!
  mulocal = rmu
!
  app = abs(pp)
  am1 = abs(m1)
  am2 = abs(m2)
!
  mulocal2 = mulocal*mulocal
!
  if (nonzerothrs) then
    thrs = onshellthrs
    if (app.lt.thrs) app = R0P0
    if (am1.lt.thrs) am1 = R0P0
    if (am2.lt.thrs) am2 = R0P0
  elseif (wunit.gt.0) then
    thrsa = onshellthrs*max(app,am1)
    thrsb = onshellthrs*max(am2,mulocal2)
    thrs = max(thrsa,thrsb)
    if (R0P0.lt.app.and.app.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.am1.and.am1.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.am2.and.am2.lt.thrs) write(wunit,*) warning
  endif
!
  call bub0( rslt ,ss,r1,r2 ,app,am1,am2 ,mulocal2 )
!
  if (punit.gt.0) then
    if (nonzerothrs) write(punit,*) 'onshell:',trim(myprint(onshellthrs))
    write(punit,*) 'muscale:',trim(myprint(mulocal))
    write(punit,*) ' pp:',trim(myprint(pp))
    write(punit,*) ' m1:',trim(myprint(m1))
    write(punit,*) ' m2:',trim(myprint(m2))
    write(punit,*) 'b0(2):',trim(myprint(rslt(2)))
    write(punit,*) 'b0(1):',trim(myprint(rslt(1)))
    write(punit,*) 'b0(0):',trim(myprint(rslt(0)))
  endif
!
  end subroutine
!
  subroutine b0rc( rslt ,pp ,m1,m2 )
!*******************************************************************
!*******************************************************************
  use avh_olo_mp_bub ,only: bub0
  include 'cts_mpc.h'
   ,intent(out) :: rslt(0:2)
  include 'cts_mpr.h'
   ,intent(in)  :: pp
  include 'cts_mpc.h'
   ,intent(in)  :: m1,m2
  include 'cts_mpr.h'
   :: hh
!
  include 'cts_mpc.h'
   :: ss,r1,r2
  include 'cts_mpr.h'
   :: app,am1,am2,thrs,thrsa,thrsb,mulocal,mulocal2
  character(25+99) ,parameter :: warning=&
                     'WARNING from OneLOop b0: '//warnonshell
  if (intro) call hello
!
  ss = pp
  r1 = m1
  r2 = m2
!
  mulocal = muscale
!
  app = abs(pp)
!
  am1 = mp_rl(r1)
  hh  = mp_ig(r1)
  if (hh.gt.R0P0) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop b0: ' &
      ,'r1 has positive imaginary part, switching its sign.'
    r1 = mp_cx( am1 ,-hh )
  endif
  am1 = abs(am1) + abs(hh)
!
  am2 = mp_rl(r2)
  hh  = mp_ig(r2)
  if (hh.gt.R0P0) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop b0: ' &
      ,'r2 has positive imaginary part, switching its sign.'
    r2 = mp_cx( am2 ,-hh )
  endif
  am2 = abs(am2) + abs(hh)
!
  mulocal2 = mulocal*mulocal
!
  if (nonzerothrs) then
    thrs = onshellthrs
    if (app.lt.thrs) app = R0P0
    if (am1.lt.thrs) am1 = R0P0
    if (am2.lt.thrs) am2 = R0P0
  elseif (wunit.gt.0) then
    thrsa = onshellthrs*max(app,am1)
    thrsb = onshellthrs*max(am2,mulocal2)
    thrs = max(thrsa,thrsb)
    if (R0P0.lt.app.and.app.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.am1.and.am1.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.am2.and.am2.lt.thrs) write(wunit,*) warning
  endif
!
  call bub0( rslt ,ss,r1,r2 ,app,am1,am2 ,mulocal2 )
!
  if (punit.gt.0) then
    if (nonzerothrs) write(punit,*) 'onshell:',trim(myprint(onshellthrs))
    write(punit,*) 'muscale:',trim(myprint(mulocal))
    write(punit,*) ' pp:',trim(myprint(pp))
    write(punit,*) ' m1:',trim(myprint(m1))
    write(punit,*) ' m2:',trim(myprint(m2))
    write(punit,*) 'b0(2):',trim(myprint(rslt(2)))
    write(punit,*) 'b0(1):',trim(myprint(rslt(1)))
    write(punit,*) 'b0(0):',trim(myprint(rslt(0)))
  endif
!
  end subroutine
!
  subroutine b0rcr( rslt ,pp,m1,m2 ,rmu )
!*******************************************************************
!*******************************************************************
  use avh_olo_mp_bub ,only: bub0
  include 'cts_mpc.h'
   ,intent(out) :: rslt(0:2)
  include 'cts_mpr.h'
   ,intent(in)  :: pp ,rmu
  include 'cts_mpc.h'
   ,intent(in)  :: m1,m2
  include 'cts_mpr.h'
   :: hh
!
  include 'cts_mpc.h'
   :: ss,r1,r2
  include 'cts_mpr.h'
   :: app,am1,am2,thrs,thrsa,thrsb,mulocal,mulocal2
  character(25+99) ,parameter :: warning=&
                     'WARNING from OneLOop b0: '//warnonshell
  if (intro) call hello
!
  ss = pp
  r1 = m1
  r2 = m2
!
  mulocal = rmu
!
  app = abs(pp)
!
  am1 = mp_rl(r1)
  hh  = mp_ig(r1)
  if (hh.gt.R0P0) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop b0: ' &
      ,'r1 has positive imaginary part, switching its sign.'
    r1 = mp_cx( am1 ,-hh )
  endif
  am1 = abs(am1) + abs(hh)
!
  am2 = mp_rl(r2)
  hh  = mp_ig(r2)
  if (hh.gt.R0P0) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop b0: ' &
      ,'r2 has positive imaginary part, switching its sign.'
    r2 = mp_cx( am2 ,-hh )
  endif
  am2 = abs(am2) + abs(hh)
!
  mulocal2 = mulocal*mulocal
!
  if (nonzerothrs) then
    thrs = onshellthrs
    if (app.lt.thrs) app = R0P0
    if (am1.lt.thrs) am1 = R0P0
    if (am2.lt.thrs) am2 = R0P0
  elseif (wunit.gt.0) then
    thrsa = onshellthrs*max(app,am1)
    thrsb = onshellthrs*max(am2,mulocal2)
    thrs = max(thrsa,thrsb)
    if (R0P0.lt.app.and.app.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.am1.and.am1.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.am2.and.am2.lt.thrs) write(wunit,*) warning
  endif
!
  call bub0( rslt ,ss,r1,r2 ,app,am1,am2 ,mulocal2 )
!
  if (punit.gt.0) then
    if (nonzerothrs) write(punit,*) 'onshell:',trim(myprint(onshellthrs))
    write(punit,*) 'muscale:',trim(myprint(mulocal))
    write(punit,*) ' pp:',trim(myprint(pp))
    write(punit,*) ' m1:',trim(myprint(m1))
    write(punit,*) ' m2:',trim(myprint(m2))
    write(punit,*) 'b0(2):',trim(myprint(rslt(2)))
    write(punit,*) 'b0(1):',trim(myprint(rslt(1)))
    write(punit,*) 'b0(0):',trim(myprint(rslt(0)))
  endif
!
  end subroutine
!
  subroutine b0cc( rslt ,pp,m1,m2 )
!*******************************************************************
!*******************************************************************
  use avh_olo_mp_bub ,only: bub0
  include 'cts_mpc.h'
   ,intent(out) :: rslt(0:2)
  include 'cts_mpc.h'
   ,intent(in)  :: pp,m1,m2
  include 'cts_mpr.h'
   :: hh
!
  include 'cts_mpc.h'
   :: ss,r1,r2
  include 'cts_mpr.h'
   :: app,am1,am2,thrs,thrsa,thrsb,mulocal,mulocal2
  character(25+99) ,parameter :: warning=&
                     'WARNING from OneLOop b0: '//warnonshell
  if (intro) call hello
!
  ss = pp
  r1 = m1
  r2 = m2
!
  mulocal = muscale
!
  app = mp_rl(ss)
  if (mp_ig(ss).ne.R0P0) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop b0: ' &
      ,'ss has non-zero imaginary part, putting it to zero.'
    ss = mp_cx( app,R0P0 )
  endif
  app = abs(app)
!
  am1 = mp_rl(r1)
  hh  = mp_ig(r1)
  if (hh.gt.R0P0) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop b0: ' &
      ,'r1 has positive imaginary part, switching its sign.'
    r1 = mp_cx( am1 ,-hh )
  endif
  am1 = abs(am1) + abs(hh)
!
  am2 = mp_rl(r2)
  hh  = mp_ig(r2)
  if (hh.gt.R0P0) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop b0: ' &
      ,'r2 has positive imaginary part, switching its sign.'
    r2 = mp_cx( am2 ,-hh )
  endif
  am2 = abs(am2) + abs(hh)
!
  mulocal2 = mulocal*mulocal
!
  if (nonzerothrs) then
    thrs = onshellthrs
    if (app.lt.thrs) app = R0P0
    if (am1.lt.thrs) am1 = R0P0
    if (am2.lt.thrs) am2 = R0P0
  elseif (wunit.gt.0) then
    thrsa = onshellthrs*max(app,am1)
    thrsb = onshellthrs*max(am2,mulocal2)
    thrs = max(thrsa,thrsb)
    if (R0P0.lt.app.and.app.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.am1.and.am1.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.am2.and.am2.lt.thrs) write(wunit,*) warning
  endif
!
  call bub0( rslt ,ss,r1,r2 ,app,am1,am2 ,mulocal2 )
!
  if (punit.gt.0) then
    if (nonzerothrs) write(punit,*) 'onshell:',trim(myprint(onshellthrs))
    write(punit,*) 'muscale:',trim(myprint(mulocal))
    write(punit,*) ' pp:',trim(myprint(pp))
    write(punit,*) ' m1:',trim(myprint(m1))
    write(punit,*) ' m2:',trim(myprint(m2))
    write(punit,*) 'b0(2):',trim(myprint(rslt(2)))
    write(punit,*) 'b0(1):',trim(myprint(rslt(1)))
    write(punit,*) 'b0(0):',trim(myprint(rslt(0)))
  endif
!
  end subroutine
!
  subroutine b0ccr( rslt ,pp,m1,m2 ,rmu )
!*******************************************************************
!*******************************************************************
  use avh_olo_mp_bub ,only: bub0
  include 'cts_mpc.h'
   ,intent(out) :: rslt(0:2)
  include 'cts_mpc.h'
   ,intent(in)  :: pp,m1,m2
  include 'cts_mpr.h'
   ,intent(in)  :: rmu
  include 'cts_mpr.h'
   :: hh
!
  include 'cts_mpc.h'
   :: ss,r1,r2
  include 'cts_mpr.h'
   :: app,am1,am2,thrs,thrsa,thrsb,mulocal,mulocal2
  character(25+99) ,parameter :: warning=&
                     'WARNING from OneLOop b0: '//warnonshell
  if (intro) call hello
!
  ss = pp
  r1 = m1
  r2 = m2
!
  mulocal = rmu
!
  app = mp_rl(ss)
  if (mp_ig(ss).ne.R0P0) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop b0: ' &
      ,'ss has non-zero imaginary part, putting it to zero.'
    ss = mp_cx( app,R0P0 )
  endif
  app = abs(app)
!
  am1 = mp_rl(r1)
  hh  = mp_ig(r1)
  if (hh.gt.R0P0) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop b0: ' &
      ,'r1 has positive imaginary part, switching its sign.'
    r1 = mp_cx( am1 ,-hh )
  endif
  am1 = abs(am1) + abs(hh)
!
  am2 = mp_rl(r2)
  hh  = mp_ig(r2)
  if (hh.gt.R0P0) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop b0: ' &
      ,'r2 has positive imaginary part, switching its sign.'
    r2 = mp_cx( am2 ,-hh )
  endif
  am2 = abs(am2) + abs(hh)
!
  mulocal2 = mulocal*mulocal
!
  if (nonzerothrs) then
    thrs = onshellthrs
    if (app.lt.thrs) app = R0P0
    if (am1.lt.thrs) am1 = R0P0
    if (am2.lt.thrs) am2 = R0P0
  elseif (wunit.gt.0) then
    thrsa = onshellthrs*max(app,am1)
    thrsb = onshellthrs*max(am2,mulocal2)
    thrs = max(thrsa,thrsb)
    if (R0P0.lt.app.and.app.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.am1.and.am1.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.am2.and.am2.lt.thrs) write(wunit,*) warning
  endif
!
  call bub0( rslt ,ss,r1,r2 ,app,am1,am2 ,mulocal2 )
!
  if (punit.gt.0) then
    if (nonzerothrs) write(punit,*) 'onshell:',trim(myprint(onshellthrs))
    write(punit,*) 'muscale:',trim(myprint(mulocal))
    write(punit,*) ' pp:',trim(myprint(pp))
    write(punit,*) ' m1:',trim(myprint(m1))
    write(punit,*) ' m2:',trim(myprint(m2))
    write(punit,*) 'b0(2):',trim(myprint(rslt(2)))
    write(punit,*) 'b0(1):',trim(myprint(rslt(1)))
    write(punit,*) 'b0(0):',trim(myprint(rslt(0)))
  endif
!
  end subroutine


!*******************************************************************
! Return the Papparino-Veltman functions b11,b00,b1,b0 , for
!
!      C   /      d^(Dim)q
!   ------ | -------------------- = b0
!   i*pi^2 / [q^2-m1][(q+p)^2-m2]
!
!      C   /    d^(Dim)q q^mu
!   ------ | -------------------- = p^mu b1
!   i*pi^2 / [q^2-m1][(q+p)^2-m2]
!
!      C   /  d^(Dim)q q^mu q^nu
!   ------ | -------------------- = g^{mu,nu} b00 + p^mu p^nu b11
!   i*pi^2 / [q^2-m1][(q+p)^2-m2]
!
! Check the comments in  subroutine olo_mp_onshell  to find out how
! this routine decides when to return IR-divergent cases.
!*******************************************************************
!
  subroutine b11rr( b11,b00,b1,b0 ,pp,m1,m2 )
!*******************************************************************
!*******************************************************************
  use avh_olo_mp_bub ,only: bub11
  include 'cts_mpc.h'
   ,intent(out) :: b11(0:2),b00(0:2),b1(0:2),b0(0:2)
  include 'cts_mpr.h'
   ,intent(in)  :: pp,m1,m2
!
  include 'cts_mpc.h'
   :: ss,r1,r2
  include 'cts_mpr.h'
   :: app,am1,am2,thrs,thrsa,thrsb,mulocal,mulocal2
  character(26+99) ,parameter :: warning=&
                     'WARNING from OneLOop b11: '//warnonshell
  if (intro) call hello
!
  ss = pp
  r1 = m1
  r2 = m2
!
  mulocal = muscale
!
  app = abs(pp)
  am1 = abs(m1)
  am2 = abs(m2)
!
  mulocal2 = mulocal*mulocal
!
  if (nonzerothrs) then
    thrs = onshellthrs
    if (app.lt.thrs) app = R0P0
    if (am1.lt.thrs) am1 = R0P0
    if (am2.lt.thrs) am2 = R0P0
  elseif (wunit.gt.0) then
    thrsa = onshellthrs*max(app,am1)
    thrsb = onshellthrs*max(am2,mulocal2)
    thrs = max(thrsa,thrsb)
    if (R0P0.lt.app.and.app.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.am1.and.am1.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.am2.and.am2.lt.thrs) write(wunit,*) warning
  endif
!
  call bub11( b11,b00,b1,b0 ,ss,r1,r2 ,app,am1,am2 ,mulocal2 )
!
  if (punit.gt.0) then
    if (nonzerothrs) write(punit,*) 'onshell:',trim(myprint(onshellthrs))
    write(punit,*) 'muscale:',trim(myprint(mulocal))
    write(punit,*) ' pp:',trim(myprint(pp))
    write(punit,*) ' m1:',trim(myprint(m1))
    write(punit,*) ' m2:',trim(myprint(m2))
    write(punit,*) 'b11(2):',trim(myprint(b11(2)))
    write(punit,*) 'b11(1):',trim(myprint(b11(1)))
    write(punit,*) 'b11(0):',trim(myprint(b11(0)))
    write(punit,*) 'b00(2):',trim(myprint(b00(2)))
    write(punit,*) 'b00(1):',trim(myprint(b00(1)))
    write(punit,*) 'b00(0):',trim(myprint(b00(0)))
    write(punit,*) ' b1(2):',trim(myprint(b1(2) ))
    write(punit,*) ' b1(1):',trim(myprint(b1(1) ))
    write(punit,*) ' b1(0):',trim(myprint(b1(0) ))
    write(punit,*) ' b0(2):',trim(myprint(b0(2) ))
    write(punit,*) ' b0(1):',trim(myprint(b0(1) ))
    write(punit,*) ' b0(0):',trim(myprint(b0(0) ))
  endif
!
  end subroutine
!
  subroutine b11rrr( b11,b00,b1,b0 ,pp,m1,m2 ,rmu )
!*******************************************************************
!*******************************************************************
  use avh_olo_mp_bub ,only: bub11
  include 'cts_mpc.h'
   ,intent(out) :: b11(0:2),b00(0:2),b1(0:2),b0(0:2)
  include 'cts_mpr.h'
   ,intent(in)  :: pp,m1,m2,rmu
!
  include 'cts_mpc.h'
   :: ss,r1,r2
  include 'cts_mpr.h'
   :: app,am1,am2,thrs,thrsa,thrsb,mulocal,mulocal2
  character(26+99) ,parameter :: warning=&
                     'WARNING from OneLOop b11: '//warnonshell
  if (intro) call hello
!
  ss = pp
  r1 = m1
  r2 = m2
!
  mulocal = rmu
!
  app = abs(pp)
  am1 = abs(m1)
  am2 = abs(m2)
!
  mulocal2 = mulocal*mulocal
!
  if (nonzerothrs) then
    thrs = onshellthrs
    if (app.lt.thrs) app = R0P0
    if (am1.lt.thrs) am1 = R0P0
    if (am2.lt.thrs) am2 = R0P0
  elseif (wunit.gt.0) then
    thrsa = onshellthrs*max(app,am1)
    thrsb = onshellthrs*max(am2,mulocal2)
    thrs = max(thrsa,thrsb)
    if (R0P0.lt.app.and.app.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.am1.and.am1.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.am2.and.am2.lt.thrs) write(wunit,*) warning
  endif
!
  call bub11( b11,b00,b1,b0 ,ss,r1,r2 ,app,am1,am2 ,mulocal2 )
!
  if (punit.gt.0) then
    if (nonzerothrs) write(punit,*) 'onshell:',trim(myprint(onshellthrs))
    write(punit,*) 'muscale:',trim(myprint(mulocal))
    write(punit,*) ' pp:',trim(myprint(pp))
    write(punit,*) ' m1:',trim(myprint(m1))
    write(punit,*) ' m2:',trim(myprint(m2))
    write(punit,*) 'b11(2):',trim(myprint(b11(2)))
    write(punit,*) 'b11(1):',trim(myprint(b11(1)))
    write(punit,*) 'b11(0):',trim(myprint(b11(0)))
    write(punit,*) 'b00(2):',trim(myprint(b00(2)))
    write(punit,*) 'b00(1):',trim(myprint(b00(1)))
    write(punit,*) 'b00(0):',trim(myprint(b00(0)))
    write(punit,*) ' b1(2):',trim(myprint(b1(2) ))
    write(punit,*) ' b1(1):',trim(myprint(b1(1) ))
    write(punit,*) ' b1(0):',trim(myprint(b1(0) ))
    write(punit,*) ' b0(2):',trim(myprint(b0(2) ))
    write(punit,*) ' b0(1):',trim(myprint(b0(1) ))
    write(punit,*) ' b0(0):',trim(myprint(b0(0) ))
  endif
!
  end subroutine
!
  subroutine b11rc( b11,b00,b1,b0 ,pp,m1,m2 )
!*******************************************************************
!*******************************************************************
  use avh_olo_mp_bub ,only: bub11
  include 'cts_mpc.h'
   ,intent(out) :: b11(0:2),b00(0:2),b1(0:2),b0(0:2)
  include 'cts_mpr.h'
   ,intent(in)  :: pp
  include 'cts_mpc.h'
   ,intent(in)  :: m1,m2
  include 'cts_mpr.h'
   :: hh
!
  include 'cts_mpc.h'
   :: ss,r1,r2
  include 'cts_mpr.h'
   :: app,am1,am2,thrs,thrsa,thrsb,mulocal,mulocal2
  character(26+99) ,parameter :: warning=&
                     'WARNING from OneLOop b11: '//warnonshell
  if (intro) call hello
!
  ss = pp
  r1 = m1
  r2 = m2
!
  mulocal = muscale
!
  app = abs(pp)
!
  am1 = mp_rl(r1)
  hh  = mp_ig(r1)
  if (hh.gt.R0P0) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop b11: ' &
      ,'r1 has positive imaginary part, switching its sign.'
    r1 = mp_cx( am1 ,-hh )
  endif
  am1 = abs(am1) + abs(hh)
!
  am2 = mp_rl(r2)
  hh  = mp_ig(r2)
  if (hh.gt.R0P0) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop b11: ' &
      ,'r2 has positive imaginary part, switching its sign.'
    r2 = mp_cx( am2 ,-hh )
  endif
  am2 = abs(am2) + abs(hh)
!
  mulocal2 = mulocal*mulocal
!
  if (nonzerothrs) then
    thrs = onshellthrs
    if (app.lt.thrs) app = R0P0
    if (am1.lt.thrs) am1 = R0P0
    if (am2.lt.thrs) am2 = R0P0
  elseif (wunit.gt.0) then
    thrsa = onshellthrs*max(app,am1)
    thrsb = onshellthrs*max(am2,mulocal2)
    thrs = max(thrsa,thrsb)
    if (R0P0.lt.app.and.app.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.am1.and.am1.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.am2.and.am2.lt.thrs) write(wunit,*) warning
  endif
!
  call bub11( b11,b00,b1,b0 ,ss,r1,r2 ,app,am1,am2 ,mulocal2 )
!
  if (punit.gt.0) then
    if (nonzerothrs) write(punit,*) 'onshell:',trim(myprint(onshellthrs))
    write(punit,*) 'muscale:',trim(myprint(mulocal))
    write(punit,*) ' pp:',trim(myprint(pp))
    write(punit,*) ' m1:',trim(myprint(m1))
    write(punit,*) ' m2:',trim(myprint(m2))
    write(punit,*) 'b11(2):',trim(myprint(b11(2)))
    write(punit,*) 'b11(1):',trim(myprint(b11(1)))
    write(punit,*) 'b11(0):',trim(myprint(b11(0)))
    write(punit,*) 'b00(2):',trim(myprint(b00(2)))
    write(punit,*) 'b00(1):',trim(myprint(b00(1)))
    write(punit,*) 'b00(0):',trim(myprint(b00(0)))
    write(punit,*) ' b1(2):',trim(myprint(b1(2) ))
    write(punit,*) ' b1(1):',trim(myprint(b1(1) ))
    write(punit,*) ' b1(0):',trim(myprint(b1(0) ))
    write(punit,*) ' b0(2):',trim(myprint(b0(2) ))
    write(punit,*) ' b0(1):',trim(myprint(b0(1) ))
    write(punit,*) ' b0(0):',trim(myprint(b0(0) ))
  endif
!
  end subroutine
!
  subroutine b11rcr( b11,b00,b1,b0 ,pp,m1,m2 ,rmu )
!*******************************************************************
!*******************************************************************
  use avh_olo_mp_bub ,only: bub11
  include 'cts_mpc.h'
   ,intent(out) :: b11(0:2),b00(0:2),b1(0:2),b0(0:2)
  include 'cts_mpr.h'
   ,intent(in)  :: pp ,rmu
  include 'cts_mpc.h'
   ,intent(in)  :: m1,m2
  include 'cts_mpr.h'
   :: hh
!
  include 'cts_mpc.h'
   :: ss,r1,r2
  include 'cts_mpr.h'
   :: app,am1,am2,thrs,thrsa,thrsb,mulocal,mulocal2
  character(26+99) ,parameter :: warning=&
                     'WARNING from OneLOop b11: '//warnonshell
  if (intro) call hello
!
  ss = pp
  r1 = m1
  r2 = m2
!
  mulocal = rmu
!
  app = abs(pp)
!
  am1 = mp_rl(r1)
  hh  = mp_ig(r1)
  if (hh.gt.R0P0) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop b11: ' &
      ,'r1 has positive imaginary part, switching its sign.'
    r1 = mp_cx( am1 ,-hh )
  endif
  am1 = abs(am1) + abs(hh)
!
  am2 = mp_rl(r2)
  hh  = mp_ig(r2)
  if (hh.gt.R0P0) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop b11: ' &
      ,'r2 has positive imaginary part, switching its sign.'
    r2 = mp_cx( am2 ,-hh )
  endif
  am2 = abs(am2) + abs(hh)
!
  mulocal2 = mulocal*mulocal
!
  if (nonzerothrs) then
    thrs = onshellthrs
    if (app.lt.thrs) app = R0P0
    if (am1.lt.thrs) am1 = R0P0
    if (am2.lt.thrs) am2 = R0P0
  elseif (wunit.gt.0) then
    thrsa = onshellthrs*max(app,am1)
    thrsb = onshellthrs*max(am2,mulocal2)
    thrs = max(thrsa,thrsb)
    if (R0P0.lt.app.and.app.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.am1.and.am1.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.am2.and.am2.lt.thrs) write(wunit,*) warning
  endif
!
  call bub11( b11,b00,b1,b0 ,ss,r1,r2 ,app,am1,am2 ,mulocal2 )
!
  if (punit.gt.0) then
    if (nonzerothrs) write(punit,*) 'onshell:',trim(myprint(onshellthrs))
    write(punit,*) 'muscale:',trim(myprint(mulocal))
    write(punit,*) ' pp:',trim(myprint(pp))
    write(punit,*) ' m1:',trim(myprint(m1))
    write(punit,*) ' m2:',trim(myprint(m2))
    write(punit,*) 'b11(2):',trim(myprint(b11(2)))
    write(punit,*) 'b11(1):',trim(myprint(b11(1)))
    write(punit,*) 'b11(0):',trim(myprint(b11(0)))
    write(punit,*) 'b00(2):',trim(myprint(b00(2)))
    write(punit,*) 'b00(1):',trim(myprint(b00(1)))
    write(punit,*) 'b00(0):',trim(myprint(b00(0)))
    write(punit,*) ' b1(2):',trim(myprint(b1(2) ))
    write(punit,*) ' b1(1):',trim(myprint(b1(1) ))
    write(punit,*) ' b1(0):',trim(myprint(b1(0) ))
    write(punit,*) ' b0(2):',trim(myprint(b0(2) ))
    write(punit,*) ' b0(1):',trim(myprint(b0(1) ))
    write(punit,*) ' b0(0):',trim(myprint(b0(0) ))
  endif
!
  end subroutine
!
  subroutine b11cc( b11,b00,b1,b0 ,pp,m1,m2 )
!*******************************************************************
!*******************************************************************
  use avh_olo_mp_bub ,only: bub11
  include 'cts_mpc.h'
   ,intent(out) :: b11(0:2),b00(0:2),b1(0:2),b0(0:2)
  include 'cts_mpc.h'
   ,intent(in)  :: pp,m1,m2
  include 'cts_mpr.h'
   :: hh
!
  include 'cts_mpc.h'
   :: ss,r1,r2
  include 'cts_mpr.h'
   :: app,am1,am2,thrs,thrsa,thrsb,mulocal,mulocal2
  character(26+99) ,parameter :: warning=&
                     'WARNING from OneLOop b11: '//warnonshell

  if (intro) call hello
!
  ss = pp
  r1 = m1
  r2 = m2
!
  mulocal = muscale
! 
  app = mp_rl(ss)
  if (mp_ig(ss).ne.R0P0) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop b11: ' &
      ,'ss has non-zero imaginary part, putting it to zero.'
    ss = mp_cx( app,R0P0 )
  endif
  app = abs(app)
!
  am1 = mp_rl(r1)
  hh  = mp_ig(r1)
  if (hh.gt.R0P0) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop b11: ' &
      ,'r1 has positive imaginary part, switching its sign.'
    r1 = mp_cx( am1 ,-hh )
  endif
  am1 = abs(am1) + abs(hh)
!
  am2 = mp_rl(r2)
  hh  = mp_ig(r2)
  if (hh.gt.R0P0) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop b11: ' &
      ,'r2 has positive imaginary part, switching its sign.'
    r2 = mp_cx( am2 ,-hh )
  endif
  am2 = abs(am2) + abs(hh)
!
  mulocal2 = mulocal*mulocal
!
  if (nonzerothrs) then
    thrs = onshellthrs
    if (app.lt.thrs) app = R0P0
    if (am1.lt.thrs) am1 = R0P0
    if (am2.lt.thrs) am2 = R0P0
  elseif (wunit.gt.0) then
    thrsa = onshellthrs*max(app,am1)
    thrsb = onshellthrs*max(am2,mulocal2)
    thrs = max(thrsa,thrsb)
    if (R0P0.lt.app.and.app.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.am1.and.am1.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.am2.and.am2.lt.thrs) write(wunit,*) warning
  endif
!
  call bub11( b11,b00,b1,b0 ,ss,r1,r2 ,app,am1,am2 ,mulocal2 )
!
  if (punit.gt.0) then
    if (nonzerothrs) write(punit,*) 'onshell:',trim(myprint(onshellthrs))
    write(punit,*) 'muscale:',trim(myprint(mulocal))
    write(punit,*) ' pp:',trim(myprint(pp))
    write(punit,*) ' m1:',trim(myprint(m1))
    write(punit,*) ' m2:',trim(myprint(m2))
    write(punit,*) 'b11(2):',trim(myprint(b11(2)))
    write(punit,*) 'b11(1):',trim(myprint(b11(1)))
    write(punit,*) 'b11(0):',trim(myprint(b11(0)))
    write(punit,*) 'b00(2):',trim(myprint(b00(2)))
    write(punit,*) 'b00(1):',trim(myprint(b00(1)))
    write(punit,*) 'b00(0):',trim(myprint(b00(0)))
    write(punit,*) ' b1(2):',trim(myprint(b1(2) ))
    write(punit,*) ' b1(1):',trim(myprint(b1(1) ))
    write(punit,*) ' b1(0):',trim(myprint(b1(0) ))
    write(punit,*) ' b0(2):',trim(myprint(b0(2) ))
    write(punit,*) ' b0(1):',trim(myprint(b0(1) ))
    write(punit,*) ' b0(0):',trim(myprint(b0(0) ))
  endif
!
  end subroutine
!
  subroutine b11ccr( b11,b00,b1,b0 ,pp,m1,m2 ,rmu )
!*******************************************************************
!*******************************************************************
  use avh_olo_mp_bub ,only: bub11
  include 'cts_mpc.h'
   ,intent(out) :: b11(0:2),b00(0:2),b1(0:2),b0(0:2)
  include 'cts_mpc.h'
   ,intent(in)  :: pp,m1,m2
  include 'cts_mpr.h'
   ,intent(in)  :: rmu
  include 'cts_mpr.h'
   :: hh
!
  include 'cts_mpc.h'
   :: ss,r1,r2
  include 'cts_mpr.h'
   :: app,am1,am2,thrs,thrsa,thrsb,mulocal,mulocal2
  character(26+99) ,parameter :: warning=&
                     'WARNING from OneLOop b11: '//warnonshell
  if (intro) call hello
!
  ss = pp
  r1 = m1
  r2 = m2
!
  mulocal = rmu
!
  app = mp_rl(ss)
  if (mp_ig(ss).ne.R0P0) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop b11: ' &
      ,'ss has non-zero imaginary part, putting it to zero.'
    ss = mp_cx( app,R0P0 )
  endif
  app = abs(app)
!
  am1 = mp_rl(r1)
  hh  = mp_ig(r1)
  if (hh.gt.R0P0) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop b11: ' &
      ,'r1 has positive imaginary part, switching its sign.'
    r1 = mp_cx( am1 ,-hh )
  endif
  am1 = abs(am1) + abs(hh)
!
  am2 = mp_rl(r2)
  hh  = mp_ig(r2)
  if (hh.gt.R0P0) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop b11: ' &
      ,'r2 has positive imaginary part, switching its sign.'
    r2 = mp_cx( am2 ,-hh )
  endif
  am2 = abs(am2) + abs(hh)
!
  mulocal2 = mulocal*mulocal
!
  if (nonzerothrs) then
    thrs = onshellthrs
    if (app.lt.thrs) app = R0P0
    if (am1.lt.thrs) am1 = R0P0
    if (am2.lt.thrs) am2 = R0P0
  elseif (wunit.gt.0) then
    thrsa = onshellthrs*max(app,am1)
    thrsb = onshellthrs*max(am2,mulocal2)
    thrs = max(thrsa,thrsb)
    if (R0P0.lt.app.and.app.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.am1.and.am1.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.am2.and.am2.lt.thrs) write(wunit,*) warning
  endif
!
  call bub11( b11,b00,b1,b0 ,ss,r1,r2 ,app,am1,am2 ,mulocal2 )
!
  if (punit.gt.0) then
    if (nonzerothrs) write(punit,*) 'onshell:',trim(myprint(onshellthrs))
    write(punit,*) 'muscale:',trim(myprint(mulocal))
    write(punit,*) ' pp:',trim(myprint(pp))
    write(punit,*) ' m1:',trim(myprint(m1))
    write(punit,*) ' m2:',trim(myprint(m2))
    write(punit,*) 'b11(2):',trim(myprint(b11(2)))
    write(punit,*) 'b11(1):',trim(myprint(b11(1)))
    write(punit,*) 'b11(0):',trim(myprint(b11(0)))
    write(punit,*) 'b00(2):',trim(myprint(b00(2)))
    write(punit,*) 'b00(1):',trim(myprint(b00(1)))
    write(punit,*) 'b00(0):',trim(myprint(b00(0)))
    write(punit,*) ' b1(2):',trim(myprint(b1(2) ))
    write(punit,*) ' b1(1):',trim(myprint(b1(1) ))
    write(punit,*) ' b1(0):',trim(myprint(b1(0) ))
    write(punit,*) ' b0(2):',trim(myprint(b0(2) ))
    write(punit,*) ' b0(1):',trim(myprint(b0(1) ))
    write(punit,*) ' b0(0):',trim(myprint(b0(0) ))
  endif
!
  end subroutine


!*******************************************************************
! calculates
!               C   /               d^(Dim)q
!            ------ | ---------------------------------------
!            i*pi^2 / [q^2-m1] [(q+k1)^2-m2] [(q+k1+k2)^2-m3]
!
! with  Dim = 4-2*eps
!         C = pi^eps * mu^(2*eps)
!             * GAMMA(1-2*eps)/GAMMA(1-eps)^2/GAMMA(1+eps)
!
! input:  p1=k1^2, p2=k2^2, p3=(k1+k2)^2,  m1,m2,m3=squared masses
! output: rslt(0) = eps^0   -coefficient
!         rslt(1) = eps^(-1)-coefficient
!         rslt(2) = eps^(-2)-coefficient
!
! Check the comments in  subroutine olo_mp_onshell  to find out how
! this routine decides when to return IR-divergent cases.
!*******************************************************************
!
  subroutine c0rr( rslt ,p1,p2,p3 ,m1,m2,m3 )
!*******************************************************************
!*******************************************************************
  use avh_olo_mp_tri
  include 'cts_mpc.h'
   ,intent(out) :: rslt(0:2)
  include 'cts_mpr.h'
   ,intent(in)  :: p1,p2,p3 ,m1,m2,m3
  include 'cts_mpr.h'
   :: pp(3),mm(3)
!
  include 'cts_mpc.h'
   :: ss(3),rr(3)
  include 'cts_mpr.h'
   :: smax,ap(3),am(3),as(3),ar(3),thrs,s1r2,s2r3,s3r3
  include 'cts_mpr.h'
   :: mulocal,mulocal2
  integer :: icase,ii
  include 'cts_mpc.h'
   :: const
  character(25+99) ,parameter :: warning=&
                     'WARNING from OneLOop c0: '//warnonshell
  const=C1P0*TWOPI*TWOPI/48
  if (intro) call hello
!
  pp(1) = p1
  pp(2) = p2
  pp(3) = p3
  mm(1) = m1
  mm(2) = m2
  mm(3) = m3
!
  smax = R0P0
!
  mulocal = muscale
!
  do ii=1,3
    ap(ii) = abs(pp(ii))
    if (ap(ii).gt.smax) smax = ap(ii)
  enddo
!
  do ii=1,3
    am(ii) = abs(mm(ii))
    if (am(ii).gt.smax) smax = am(ii)
  enddo
!
  mulocal2 = mulocal*mulocal
!
  if (smax.eq.R0P0) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop c0: ' &
      ,'all input equal zero, returning 0'
    rslt = C0P0
    return
  endif
!
  if (mulocal2.gt.smax) smax = mulocal2
!
  if (nonzerothrs) then
    thrs = onshellthrs
    do ii=1,3
      if (ap(ii).lt.thrs) ap(ii) = R0P0
      if (am(ii).lt.thrs) am(ii) = R0P0
    enddo
  else
    thrs = onshellthrs*smax
    if (wunit.gt.0) then
    do ii=1,3
      if (R0P0.lt.ap(ii).and.ap(ii).lt.thrs) write(wunit,*) warning
      if (R0P0.lt.am(ii).and.am(ii).lt.thrs) write(wunit,*) warning
    enddo
    endif
  endif
!
  icase = 0
  do ii=1,3
    if (am(ii).gt.R0P0) icase = icase + base(ii)
  enddo
  ss(1)=pp(permtable(1,icase)) ;as(1)=ap(permtable(1,icase))
  ss(2)=pp(permtable(2,icase)) ;as(2)=ap(permtable(2,icase))
  ss(3)=pp(permtable(3,icase)) ;as(3)=ap(permtable(3,icase))
  rr(1)=mm(permtable(1,icase)) ;ar(1)=am(permtable(1,icase))
  rr(2)=mm(permtable(2,icase)) ;ar(2)=am(permtable(2,icase))
  rr(3)=mm(permtable(3,icase)) ;ar(3)=am(permtable(3,icase))
  icase = casetable(icase)
!
  s1r2 = abs(ss(1)-rr(2))
  s2r3 = abs(ss(2)-rr(3))
  s3r3 = abs(ss(3)-rr(3))
  if (nonzerothrs) then
    if (s1r2.lt.thrs) s1r2 = R0P0
    if (s2r3.lt.thrs) s2r3 = R0P0
    if (s3r3.lt.thrs) s3r3 = R0P0
  elseif (wunit.gt.0) then
    if (R0P0.lt.s1r2.and.s1r2.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.s2r3.and.s2r3.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.s3r3.and.s3r3.lt.thrs) write(wunit,*) warning
  endif
!
  if     (icase.eq.3) then
! 3 non-zero internal masses
    call trif3( rslt ,ss(1),ss(2),ss(3) ,rr(1),rr(2),rr(3) )
  elseif (icase.eq.2) then
! 2 non-zero internal masses
    if (s1r2.ne.R0P0.or.s3r3.ne.R0P0) then
      call trif2( rslt ,ss(1),ss(2),ss(3) ,rr(2),rr(3) )
    else
      call tria4( rslt ,ss(2) ,rr(2),rr(3) ,mulocal2 )
    endif
  elseif (icase.eq.1) then
! 1 non-zero internal mass
    if     (as(1).ne.R0P0) then
      call trif1( rslt ,ss(1),ss(2),ss(3), rr(3) )
    elseif (s2r3.ne.R0P0) then
      if   (s3r3.ne.R0P0) then
        call tria3( rslt ,ss(2),ss(3) ,rr(3) ,mulocal2 )
      else
        call tria2( rslt ,ss(2) ,rr(3) ,mulocal2 )
      endif
    elseif (s3r3.ne.R0P0) then
      call tria2( rslt ,ss(3) ,rr(3) ,mulocal2 )
    else
      call tria1( rslt ,rr(3) ,mulocal2 )
    endif
  else
! 0 non-zero internal masses
    call tria0( rslt ,ss ,as ,mulocal2 )
  endif
! exp(eps*gamma_EULER) -> GAMMA(1-2*eps)/GAMMA(1-eps)^2/GAMMA(1+eps)
  rslt(0) = rslt(0) + const*rslt(2)
!
  if (punit.gt.0) then
    if (nonzerothrs) write(punit,*) 'onshell:',trim(myprint(onshellthrs))
    write(punit,*) 'muscale:',trim(myprint(mulocal))
    write(punit,*) ' p1:',trim(myprint(p1))
    write(punit,*) ' p2:',trim(myprint(p2))
    write(punit,*) ' p3:',trim(myprint(p3))
    write(punit,*) ' m1:',trim(myprint(m1))
    write(punit,*) ' m2:',trim(myprint(m2))
    write(punit,*) ' m3:',trim(myprint(m3))
    write(punit,*) 'c0(2):',trim(myprint(rslt(2)))
    write(punit,*) 'c0(1):',trim(myprint(rslt(1)))
    write(punit,*) 'c0(0):',trim(myprint(rslt(0)))
  endif
!
  end subroutine
!
  subroutine c0rrr( rslt ,p1,p2,p3 ,m1,m2,m3 ,rmu )
!*******************************************************************
!*******************************************************************
  use avh_olo_mp_tri
  include 'cts_mpc.h'
   ,intent(out) :: rslt(0:2)
  include 'cts_mpr.h'
   ,intent(in)  :: p1,p2,p3 ,m1,m2,m3 ,rmu
  include 'cts_mpr.h'
   :: pp(3),mm(3)
!
  include 'cts_mpc.h'
   :: ss(3),rr(3)
  include 'cts_mpr.h'
   :: smax,ap(3),am(3),as(3),ar(3),thrs,s1r2,s2r3,s3r3
  include 'cts_mpr.h'
   :: mulocal,mulocal2
  integer :: icase,ii
  include 'cts_mpc.h'
   :: const
  character(25+99) ,parameter :: warning=&
                     'WARNING from OneLOop c0: '//warnonshell
  const=C1P0*TWOPI*TWOPI/48
  if (intro) call hello
!
  pp(1) = p1
  pp(2) = p2
  pp(3) = p3
  mm(1) = m1
  mm(2) = m2
  mm(3) = m3
!
  smax = R0P0
!
  mulocal = rmu
!
  do ii=1,3
    ap(ii) = abs(pp(ii))
    if (ap(ii).gt.smax) smax = ap(ii)
  enddo
!
  do ii=1,3
    am(ii) = abs(mm(ii))
    if (am(ii).gt.smax) smax = am(ii)
  enddo
!
  mulocal2 = mulocal*mulocal
!
  if (smax.eq.R0P0) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop c0: ' &
      ,'all input equal zero, returning 0'
    rslt = C0P0
    return
  endif
!
  if (mulocal2.gt.smax) smax = mulocal2
!
  if (nonzerothrs) then
    thrs = onshellthrs
    do ii=1,3
      if (ap(ii).lt.thrs) ap(ii) = R0P0
      if (am(ii).lt.thrs) am(ii) = R0P0
    enddo
  else
    thrs = onshellthrs*smax
    if (wunit.gt.0) then
    do ii=1,3
      if (R0P0.lt.ap(ii).and.ap(ii).lt.thrs) write(wunit,*) warning
      if (R0P0.lt.am(ii).and.am(ii).lt.thrs) write(wunit,*) warning
    enddo
    endif
  endif
!
  icase = 0
  do ii=1,3
    if (am(ii).gt.R0P0) icase = icase + base(ii)
  enddo
  ss(1)=pp(permtable(1,icase)) ;as(1)=ap(permtable(1,icase))
  ss(2)=pp(permtable(2,icase)) ;as(2)=ap(permtable(2,icase))
  ss(3)=pp(permtable(3,icase)) ;as(3)=ap(permtable(3,icase))
  rr(1)=mm(permtable(1,icase)) ;ar(1)=am(permtable(1,icase))
  rr(2)=mm(permtable(2,icase)) ;ar(2)=am(permtable(2,icase))
  rr(3)=mm(permtable(3,icase)) ;ar(3)=am(permtable(3,icase))
  icase = casetable(icase)
!
  s1r2 = abs(ss(1)-rr(2))
  s2r3 = abs(ss(2)-rr(3))
  s3r3 = abs(ss(3)-rr(3))
  if (nonzerothrs) then
    if (s1r2.lt.thrs) s1r2 = R0P0
    if (s2r3.lt.thrs) s2r3 = R0P0
    if (s3r3.lt.thrs) s3r3 = R0P0
  elseif (wunit.gt.0) then
    if (R0P0.lt.s1r2.and.s1r2.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.s2r3.and.s2r3.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.s3r3.and.s3r3.lt.thrs) write(wunit,*) warning
  endif
!
  if     (icase.eq.3) then
! 3 non-zero internal masses
    call trif3( rslt ,ss(1),ss(2),ss(3) ,rr(1),rr(2),rr(3) )
  elseif (icase.eq.2) then
! 2 non-zero internal masses
    if (s1r2.ne.R0P0.or.s3r3.ne.R0P0) then
      call trif2( rslt ,ss(1),ss(2),ss(3) ,rr(2),rr(3) )
    else
      call tria4( rslt ,ss(2) ,rr(2),rr(3) ,mulocal2 )
    endif
  elseif (icase.eq.1) then
! 1 non-zero internal mass
    if     (as(1).ne.R0P0) then
      call trif1( rslt ,ss(1),ss(2),ss(3), rr(3) )
    elseif (s2r3.ne.R0P0) then
      if   (s3r3.ne.R0P0) then
        call tria3( rslt ,ss(2),ss(3) ,rr(3) ,mulocal2 )
      else
        call tria2( rslt ,ss(2) ,rr(3) ,mulocal2 )
      endif
    elseif (s3r3.ne.R0P0) then
      call tria2( rslt ,ss(3) ,rr(3) ,mulocal2 )
    else
      call tria1( rslt ,rr(3) ,mulocal2 )
    endif
  else
! 0 non-zero internal masses
    call tria0( rslt ,ss ,as ,mulocal2 )
  endif
! exp(eps*gamma_EULER) -> GAMMA(1-2*eps)/GAMMA(1-eps)^2/GAMMA(1+eps)
  rslt(0) = rslt(0) + const*rslt(2)
!
  if (punit.gt.0) then
    if (nonzerothrs) write(punit,*) 'onshell:',trim(myprint(onshellthrs))
    write(punit,*) 'muscale:',trim(myprint(mulocal))
    write(punit,*) ' p1:',trim(myprint(p1))
    write(punit,*) ' p2:',trim(myprint(p2))
    write(punit,*) ' p3:',trim(myprint(p3))
    write(punit,*) ' m1:',trim(myprint(m1))
    write(punit,*) ' m2:',trim(myprint(m2))
    write(punit,*) ' m3:',trim(myprint(m3))
    write(punit,*) 'c0(2):',trim(myprint(rslt(2)))
    write(punit,*) 'c0(1):',trim(myprint(rslt(1)))
    write(punit,*) 'c0(0):',trim(myprint(rslt(0)))
  endif
!
  end subroutine
!
  subroutine c0rc( rslt ,p1,p2,p3 ,m1,m2,m3 )
!*******************************************************************
!*******************************************************************
  use avh_olo_mp_tri
  include 'cts_mpc.h'
   ,intent(out) :: rslt(0:2)
  include 'cts_mpr.h'
   ,intent(in)  :: p1,p2,p3
  include 'cts_mpc.h'
   ,intent(in)  :: m1,m2,m3
  include 'cts_mpr.h'
   :: pp(3)
  include 'cts_mpc.h'
   :: mm(3)
  include 'cts_mpr.h'
   :: hh
!
  include 'cts_mpc.h'
   :: ss(3),rr(3)
  include 'cts_mpr.h'
   :: smax,ap(3),am(3),as(3),ar(3),thrs,s1r2,s2r3,s3r3
  include 'cts_mpr.h'
   :: mulocal,mulocal2
  integer :: icase,ii
  include 'cts_mpc.h'
   :: const
  character(25+99) ,parameter :: warning=&
                     'WARNING from OneLOop c0: '//warnonshell
  const=C1P0*TWOPI*TWOPI/48
  if (intro) call hello
!
  pp(1) = p1
  pp(2) = p2
  pp(3) = p3
  mm(1) = m1
  mm(2) = m2
  mm(3) = m3
!
  smax = R0P0
!
  mulocal = muscale
!
  do ii=1,3
    ap(ii) = abs(pp(ii))
    if (ap(ii).gt.smax) smax = ap(ii)
  enddo
!
  do ii=1,3
    am(ii) = mp_rl(mm(ii))
    hh     = mp_ig(mm(ii))
    if (hh.gt.R0P0) then
      if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop c0: ' &
        ,'mass-squared has positive imaginary part, switching its sign.'
      mm(ii) = mp_cx( am(ii) ,-hh )
    endif
    am(ii) = abs(am(ii)) + abs(hh)
    if (am(ii).gt.smax) smax = am(ii)
  enddo
!
  mulocal2 = mulocal*mulocal
!
  if (smax.eq.R0P0) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop c0: ' &
      ,'all input equal zero, returning 0'
    rslt = C0P0
    return
  endif
!
  if (mulocal2.gt.smax) smax = mulocal2
!
  if (nonzerothrs) then
    thrs = onshellthrs
    do ii=1,3
      if (ap(ii).lt.thrs) ap(ii) = R0P0
      if (am(ii).lt.thrs) am(ii) = R0P0
    enddo
  else
    thrs = onshellthrs*smax
    if (wunit.gt.0) then
    do ii=1,3
      if (R0P0.lt.ap(ii).and.ap(ii).lt.thrs) write(wunit,*) warning
      if (R0P0.lt.am(ii).and.am(ii).lt.thrs) write(wunit,*) warning
    enddo
    endif
  endif
!
  icase = 0
  do ii=1,3
    if (am(ii).gt.R0P0) icase = icase + base(ii)
  enddo
  ss(1)=pp(permtable(1,icase)) ;as(1)=ap(permtable(1,icase))
  ss(2)=pp(permtable(2,icase)) ;as(2)=ap(permtable(2,icase))
  ss(3)=pp(permtable(3,icase)) ;as(3)=ap(permtable(3,icase))
  rr(1)=mm(permtable(1,icase)) ;ar(1)=am(permtable(1,icase))
  rr(2)=mm(permtable(2,icase)) ;ar(2)=am(permtable(2,icase))
  rr(3)=mm(permtable(3,icase)) ;ar(3)=am(permtable(3,icase))
  icase = casetable(icase)
!
  s1r2 = abs(ss(1)-rr(2))
  s2r3 = abs(ss(2)-rr(3))
  s3r3 = abs(ss(3)-rr(3))
  if (nonzerothrs) then
    if (s1r2.lt.thrs) s1r2 = R0P0
    if (s2r3.lt.thrs) s2r3 = R0P0
    if (s3r3.lt.thrs) s3r3 = R0P0
  elseif (wunit.gt.0) then
    if (R0P0.lt.s1r2.and.s1r2.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.s2r3.and.s2r3.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.s3r3.and.s3r3.lt.thrs) write(wunit,*) warning
  endif
!
  if     (icase.eq.3) then
! 3 non-zero internal masses
    call trif3( rslt ,ss(1),ss(2),ss(3) ,rr(1),rr(2),rr(3) )
  elseif (icase.eq.2) then
! 2 non-zero internal masses
    if (s1r2.ne.R0P0.or.s3r3.ne.R0P0) then
      call trif2( rslt ,ss(1),ss(2),ss(3) ,rr(2),rr(3) )
    else
      call tria4( rslt ,ss(2) ,rr(2),rr(3) ,mulocal2 )
    endif
  elseif (icase.eq.1) then
! 1 non-zero internal mass
    if     (as(1).ne.R0P0) then
      call trif1( rslt ,ss(1),ss(2),ss(3), rr(3) )
    elseif (s2r3.ne.R0P0) then
      if   (s3r3.ne.R0P0) then
        call tria3( rslt ,ss(2),ss(3) ,rr(3) ,mulocal2 )
      else
        call tria2( rslt ,ss(2) ,rr(3) ,mulocal2 )
      endif
    elseif (s3r3.ne.R0P0) then
      call tria2( rslt ,ss(3) ,rr(3) ,mulocal2 )
    else
      call tria1( rslt ,rr(3) ,mulocal2 )
    endif
  else
! 0 non-zero internal masses
    call tria0( rslt ,ss ,as ,mulocal2 )
  endif
! exp(eps*gamma_EULER) -> GAMMA(1-2*eps)/GAMMA(1-eps)^2/GAMMA(1+eps)
  rslt(0) = rslt(0) + const*rslt(2)
!
  if (punit.gt.0) then
    if (nonzerothrs) write(punit,*) 'onshell:',trim(myprint(onshellthrs))
    write(punit,*) 'muscale:',trim(myprint(mulocal))
    write(punit,*) ' p1:',trim(myprint(p1))
    write(punit,*) ' p2:',trim(myprint(p2))
    write(punit,*) ' p3:',trim(myprint(p3))
    write(punit,*) ' m1:',trim(myprint(m1))
    write(punit,*) ' m2:',trim(myprint(m2))
    write(punit,*) ' m3:',trim(myprint(m3))
    write(punit,*) 'c0(2):',trim(myprint(rslt(2)))
    write(punit,*) 'c0(1):',trim(myprint(rslt(1)))
    write(punit,*) 'c0(0):',trim(myprint(rslt(0)))
  endif
!
  end subroutine
!
  subroutine c0rcr( rslt ,p1,p2,p3 ,m1,m2,m3 ,rmu )
!*******************************************************************
!*******************************************************************
  use avh_olo_mp_tri
  include 'cts_mpc.h'
   ,intent(out) :: rslt(0:2)
  include 'cts_mpr.h'
   ,intent(in)  :: p1,p2,p3 ,rmu
  include 'cts_mpc.h'
   ,intent(in)  :: m1,m2,m3
  include 'cts_mpr.h'
   :: pp(3)
  include 'cts_mpc.h'
   :: mm(3)
  include 'cts_mpr.h'
   :: hh
!
  include 'cts_mpc.h'
   :: ss(3),rr(3)
  include 'cts_mpr.h'
   :: smax,ap(3),am(3),as(3),ar(3),thrs,s1r2,s2r3,s3r3
  include 'cts_mpr.h'
   :: mulocal,mulocal2
  integer :: icase,ii
  include 'cts_mpc.h'
   :: const
  character(25+99) ,parameter :: warning=&
                     'WARNING from OneLOop c0: '//warnonshell
  const=C1P0*TWOPI*TWOPI/48
  if (intro) call hello
!
  pp(1) = p1
  pp(2) = p2
  pp(3) = p3
  mm(1) = m1
  mm(2) = m2
  mm(3) = m3
!
  smax = R0P0
!
  mulocal = rmu
!
  do ii=1,3
    ap(ii) = abs(pp(ii))
    if (ap(ii).gt.smax) smax = ap(ii)
  enddo
!
  do ii=1,3
    am(ii) = mp_rl(mm(ii))
    hh     = mp_ig(mm(ii))
    if (hh.gt.R0P0) then
      if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop c0: ' &
        ,'mass-squared has positive imaginary part, switching its sign.'
      mm(ii) = mp_cx( am(ii) ,-hh )
    endif
    am(ii) = abs(am(ii)) + abs(hh)
    if (am(ii).gt.smax) smax = am(ii)
  enddo
!
  mulocal2 = mulocal*mulocal
!
  if (smax.eq.R0P0) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop c0: ' &
      ,'all input equal zero, returning 0'
    rslt = C0P0
    return
  endif
!
  if (mulocal2.gt.smax) smax = mulocal2
!
  if (nonzerothrs) then
    thrs = onshellthrs
    do ii=1,3
      if (ap(ii).lt.thrs) ap(ii) = R0P0
      if (am(ii).lt.thrs) am(ii) = R0P0
    enddo
  else
    thrs = onshellthrs*smax
    if (wunit.gt.0) then
    do ii=1,3
      if (R0P0.lt.ap(ii).and.ap(ii).lt.thrs) write(wunit,*) warning
      if (R0P0.lt.am(ii).and.am(ii).lt.thrs) write(wunit,*) warning
    enddo
    endif
  endif
!
  icase = 0
  do ii=1,3
    if (am(ii).gt.R0P0) icase = icase + base(ii)
  enddo
  ss(1)=pp(permtable(1,icase)) ;as(1)=ap(permtable(1,icase))
  ss(2)=pp(permtable(2,icase)) ;as(2)=ap(permtable(2,icase))
  ss(3)=pp(permtable(3,icase)) ;as(3)=ap(permtable(3,icase))
  rr(1)=mm(permtable(1,icase)) ;ar(1)=am(permtable(1,icase))
  rr(2)=mm(permtable(2,icase)) ;ar(2)=am(permtable(2,icase))
  rr(3)=mm(permtable(3,icase)) ;ar(3)=am(permtable(3,icase))
  icase = casetable(icase)
!
  s1r2 = abs(ss(1)-rr(2))
  s2r3 = abs(ss(2)-rr(3))
  s3r3 = abs(ss(3)-rr(3))
  if (nonzerothrs) then
    if (s1r2.lt.thrs) s1r2 = R0P0
    if (s2r3.lt.thrs) s2r3 = R0P0
    if (s3r3.lt.thrs) s3r3 = R0P0
  elseif (wunit.gt.0) then
    if (R0P0.lt.s1r2.and.s1r2.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.s2r3.and.s2r3.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.s3r3.and.s3r3.lt.thrs) write(wunit,*) warning
  endif
!
  if     (icase.eq.3) then
! 3 non-zero internal masses
    call trif3( rslt ,ss(1),ss(2),ss(3) ,rr(1),rr(2),rr(3) )
  elseif (icase.eq.2) then
! 2 non-zero internal masses
    if (s1r2.ne.R0P0.or.s3r3.ne.R0P0) then
      call trif2( rslt ,ss(1),ss(2),ss(3) ,rr(2),rr(3) )
    else
      call tria4( rslt ,ss(2) ,rr(2),rr(3) ,mulocal2 )
    endif
  elseif (icase.eq.1) then
! 1 non-zero internal mass
    if     (as(1).ne.R0P0) then
      call trif1( rslt ,ss(1),ss(2),ss(3), rr(3) )
    elseif (s2r3.ne.R0P0) then
      if   (s3r3.ne.R0P0) then
        call tria3( rslt ,ss(2),ss(3) ,rr(3) ,mulocal2 )
      else
        call tria2( rslt ,ss(2) ,rr(3) ,mulocal2 )
      endif
    elseif (s3r3.ne.R0P0) then
      call tria2( rslt ,ss(3) ,rr(3) ,mulocal2 )
    else
      call tria1( rslt ,rr(3) ,mulocal2 )
    endif
  else
! 0 non-zero internal masses
    call tria0( rslt ,ss ,as ,mulocal2 )
  endif
! exp(eps*gamma_EULER) -> GAMMA(1-2*eps)/GAMMA(1-eps)^2/GAMMA(1+eps)
  rslt(0) = rslt(0) + const*rslt(2)
!
  if (punit.gt.0) then
    if (nonzerothrs) write(punit,*) 'onshell:',trim(myprint(onshellthrs))
    write(punit,*) 'muscale:',trim(myprint(mulocal))
    write(punit,*) ' p1:',trim(myprint(p1))
    write(punit,*) ' p2:',trim(myprint(p2))
    write(punit,*) ' p3:',trim(myprint(p3))
    write(punit,*) ' m1:',trim(myprint(m1))
    write(punit,*) ' m2:',trim(myprint(m2))
    write(punit,*) ' m3:',trim(myprint(m3))
    write(punit,*) 'c0(2):',trim(myprint(rslt(2)))
    write(punit,*) 'c0(1):',trim(myprint(rslt(1)))
    write(punit,*) 'c0(0):',trim(myprint(rslt(0)))
  endif
!
  end subroutine
!
  subroutine c0cc( rslt ,p1,p2,p3 ,m1,m2,m3 )
!*******************************************************************
!*******************************************************************
  use avh_olo_mp_tri
  include 'cts_mpc.h'
   ,intent(out) :: rslt(0:2)
  include 'cts_mpc.h'
   ,intent(in)  :: p1,p2,p3 ,m1,m2,m3
  include 'cts_mpc.h'
   :: pp(3),mm(3)
  include 'cts_mpr.h'
   :: hh
!
  include 'cts_mpc.h'
   :: ss(3),rr(3)
  include 'cts_mpr.h'
   :: smax,ap(3),am(3),as(3),ar(3),thrs,s1r2,s2r3,s3r3
  include 'cts_mpr.h'
   :: mulocal,mulocal2
  integer :: icase,ii
  include 'cts_mpc.h'
   :: const
  character(25+99) ,parameter :: warning=&
                     'WARNING from OneLOop c0: '//warnonshell
  const=C1P0*TWOPI*TWOPI/48
  if (intro) call hello
!
  pp(1) = p1
  pp(2) = p2
  pp(3) = p3
  mm(1) = m1
  mm(2) = m2
  mm(3) = m3
!
  smax = R0P0
!
  mulocal = muscale
!
  do ii=1,3
    ap(ii) = mp_rl(pp(ii))
    if (mp_ig(pp(ii)).ne.R0P0) then
      if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop c0: ' &
        ,'momentum with non-zero imaginary part, putting it to zero.'
      pp(ii) = mp_cx( ap(ii),R0P0 )
    endif
    ap(ii) = abs(ap(ii))
    if (ap(ii).gt.smax) smax = ap(ii)
  enddo
!
  do ii=1,3
    am(ii) = mp_rl(mm(ii))
    hh     = mp_ig(mm(ii))
    if (hh.gt.R0P0) then
      if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop c0: ' &
        ,'mass-squared has positive imaginary part, switching its sign.'
      mm(ii) = mp_cx( am(ii) ,-hh )
    endif
    am(ii) = abs(am(ii)) + abs(hh)
    if (am(ii).gt.smax) smax = am(ii)
  enddo
!
  mulocal2 = mulocal*mulocal
!
  if (smax.eq.R0P0) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop c0: ' &
      ,'all input equal zero, returning 0'
    rslt = C0P0
    return
  endif
!
  if (mulocal2.gt.smax) smax = mulocal2
!
  if (nonzerothrs) then
    thrs = onshellthrs
    do ii=1,3
      if (ap(ii).lt.thrs) ap(ii) = R0P0
      if (am(ii).lt.thrs) am(ii) = R0P0
    enddo
  else
    thrs = onshellthrs*smax
    if (wunit.gt.0) then
    do ii=1,3
      if (R0P0.lt.ap(ii).and.ap(ii).lt.thrs) write(wunit,*) warning
      if (R0P0.lt.am(ii).and.am(ii).lt.thrs) write(wunit,*) warning
    enddo
    endif
  endif
!
  icase = 0
  do ii=1,3
    if (am(ii).gt.R0P0) icase = icase + base(ii)
  enddo
  ss(1)=pp(permtable(1,icase)) ;as(1)=ap(permtable(1,icase))
  ss(2)=pp(permtable(2,icase)) ;as(2)=ap(permtable(2,icase))
  ss(3)=pp(permtable(3,icase)) ;as(3)=ap(permtable(3,icase))
  rr(1)=mm(permtable(1,icase)) ;ar(1)=am(permtable(1,icase))
  rr(2)=mm(permtable(2,icase)) ;ar(2)=am(permtable(2,icase))
  rr(3)=mm(permtable(3,icase)) ;ar(3)=am(permtable(3,icase))
  icase = casetable(icase)
!
  s1r2 = abs(ss(1)-rr(2))
  s2r3 = abs(ss(2)-rr(3))
  s3r3 = abs(ss(3)-rr(3))
  if (nonzerothrs) then
    if (s1r2.lt.thrs) s1r2 = R0P0
    if (s2r3.lt.thrs) s2r3 = R0P0
    if (s3r3.lt.thrs) s3r3 = R0P0
  elseif (wunit.gt.0) then
    if (R0P0.lt.s1r2.and.s1r2.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.s2r3.and.s2r3.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.s3r3.and.s3r3.lt.thrs) write(wunit,*) warning
  endif
!
  if     (icase.eq.3) then
! 3 non-zero internal masses
    call trif3( rslt ,ss(1),ss(2),ss(3) ,rr(1),rr(2),rr(3) )
  elseif (icase.eq.2) then
! 2 non-zero internal masses
    if (s1r2.ne.R0P0.or.s3r3.ne.R0P0) then
      call trif2( rslt ,ss(1),ss(2),ss(3) ,rr(2),rr(3) )
    else
      call tria4( rslt ,ss(2) ,rr(2),rr(3) ,mulocal2 )
    endif
  elseif (icase.eq.1) then
! 1 non-zero internal mass
    if     (as(1).ne.R0P0) then
      call trif1( rslt ,ss(1),ss(2),ss(3), rr(3) )
    elseif (s2r3.ne.R0P0) then
      if   (s3r3.ne.R0P0) then
        call tria3( rslt ,ss(2),ss(3) ,rr(3) ,mulocal2 )
      else
        call tria2( rslt ,ss(2) ,rr(3) ,mulocal2 )
      endif
    elseif (s3r3.ne.R0P0) then
      call tria2( rslt ,ss(3) ,rr(3) ,mulocal2 )
    else
      call tria1( rslt ,rr(3) ,mulocal2 )
    endif
  else
! 0 non-zero internal masses
    call tria0( rslt ,ss ,as ,mulocal2 )
  endif
! exp(eps*gamma_EULER) -> GAMMA(1-2*eps)/GAMMA(1-eps)^2/GAMMA(1+eps)
  rslt(0) = rslt(0) + const*rslt(2)
!
  if (punit.gt.0) then
    if (nonzerothrs) write(punit,*) 'onshell:',trim(myprint(onshellthrs))
    write(punit,*) 'muscale:',trim(myprint(mulocal))
    write(punit,*) ' p1:',trim(myprint(p1))
    write(punit,*) ' p2:',trim(myprint(p2))
    write(punit,*) ' p3:',trim(myprint(p3))
    write(punit,*) ' m1:',trim(myprint(m1))
    write(punit,*) ' m2:',trim(myprint(m2))
    write(punit,*) ' m3:',trim(myprint(m3))
    write(punit,*) 'c0(2):',trim(myprint(rslt(2)))
    write(punit,*) 'c0(1):',trim(myprint(rslt(1)))
    write(punit,*) 'c0(0):',trim(myprint(rslt(0)))
  endif
!
  end subroutine
!
  subroutine c0ccr( rslt ,p1,p2,p3 ,m1,m2,m3 ,rmu )
!*******************************************************************
!*******************************************************************
  use avh_olo_mp_tri
  include 'cts_mpc.h'
   ,intent(out) :: rslt(0:2)
  include 'cts_mpc.h'
   ,intent(in)  :: p1,p2,p3 ,m1,m2,m3
  include 'cts_mpr.h'
   ,intent(in)  :: rmu
  include 'cts_mpc.h'
   :: pp(3),mm(3)
  include 'cts_mpr.h'
   :: hh
!
  include 'cts_mpc.h'
   :: ss(3),rr(3)
  include 'cts_mpr.h'
   :: smax,ap(3),am(3),as(3),ar(3),thrs,s1r2,s2r3,s3r3
  include 'cts_mpr.h'
   :: mulocal,mulocal2
  integer :: icase,ii
  include 'cts_mpc.h'
   :: const
  character(25+99) ,parameter :: warning=&
                     'WARNING from OneLOop c0: '//warnonshell
  const=C1P0*TWOPI*TWOPI/48
  if (intro) call hello
!
  pp(1) = p1
  pp(2) = p2
  pp(3) = p3
  mm(1) = m1
  mm(2) = m2
  mm(3) = m3
!
  smax = R0P0
!
  mulocal = rmu
!
  do ii=1,3
    ap(ii) = mp_rl(pp(ii))
    if (mp_ig(pp(ii)).ne.R0P0) then
      if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop c0: ' &
        ,'momentum with non-zero imaginary part, putting it to zero.'
      pp(ii) = mp_cx( ap(ii),R0P0 )
    endif
    ap(ii) = abs(ap(ii))
    if (ap(ii).gt.smax) smax = ap(ii)
  enddo
!
  do ii=1,3
    am(ii) = mp_rl(mm(ii))
    hh     = mp_ig(mm(ii))
    if (hh.gt.R0P0) then
      if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop c0: ' &
        ,'mass-squared has positive imaginary part, switching its sign.'
      mm(ii) = mp_cx( am(ii) ,-hh )
    endif
    am(ii) = abs(am(ii)) + abs(hh)
    if (am(ii).gt.smax) smax = am(ii)
  enddo
!
  mulocal2 = mulocal*mulocal
!
  if (smax.eq.R0P0) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop c0: ' &
      ,'all input equal zero, returning 0'
    rslt = C0P0
    return
  endif
!
  if (mulocal2.gt.smax) smax = mulocal2
!
  if (nonzerothrs) then
    thrs = onshellthrs
    do ii=1,3
      if (ap(ii).lt.thrs) ap(ii) = R0P0
      if (am(ii).lt.thrs) am(ii) = R0P0
    enddo
  else
    thrs = onshellthrs*smax
    if (wunit.gt.0) then
    do ii=1,3
      if (R0P0.lt.ap(ii).and.ap(ii).lt.thrs) write(wunit,*) warning
      if (R0P0.lt.am(ii).and.am(ii).lt.thrs) write(wunit,*) warning
    enddo
    endif
  endif
!
  icase = 0
  do ii=1,3
    if (am(ii).gt.R0P0) icase = icase + base(ii)
  enddo
  ss(1)=pp(permtable(1,icase)) ;as(1)=ap(permtable(1,icase))
  ss(2)=pp(permtable(2,icase)) ;as(2)=ap(permtable(2,icase))
  ss(3)=pp(permtable(3,icase)) ;as(3)=ap(permtable(3,icase))
  rr(1)=mm(permtable(1,icase)) ;ar(1)=am(permtable(1,icase))
  rr(2)=mm(permtable(2,icase)) ;ar(2)=am(permtable(2,icase))
  rr(3)=mm(permtable(3,icase)) ;ar(3)=am(permtable(3,icase))
  icase = casetable(icase)
!
  s1r2 = abs(ss(1)-rr(2))
  s2r3 = abs(ss(2)-rr(3))
  s3r3 = abs(ss(3)-rr(3))
  if (nonzerothrs) then
    if (s1r2.lt.thrs) s1r2 = R0P0
    if (s2r3.lt.thrs) s2r3 = R0P0
    if (s3r3.lt.thrs) s3r3 = R0P0
  elseif (wunit.gt.0) then
    if (R0P0.lt.s1r2.and.s1r2.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.s2r3.and.s2r3.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.s3r3.and.s3r3.lt.thrs) write(wunit,*) warning
  endif
!
  if     (icase.eq.3) then
! 3 non-zero internal masses
    call trif3( rslt ,ss(1),ss(2),ss(3) ,rr(1),rr(2),rr(3) )
  elseif (icase.eq.2) then
! 2 non-zero internal masses
    if (s1r2.ne.R0P0.or.s3r3.ne.R0P0) then
      call trif2( rslt ,ss(1),ss(2),ss(3) ,rr(2),rr(3) )
    else
      call tria4( rslt ,ss(2) ,rr(2),rr(3) ,mulocal2 )
    endif
  elseif (icase.eq.1) then
! 1 non-zero internal mass
    if     (as(1).ne.R0P0) then
      call trif1( rslt ,ss(1),ss(2),ss(3), rr(3) )
    elseif (s2r3.ne.R0P0) then
      if   (s3r3.ne.R0P0) then
        call tria3( rslt ,ss(2),ss(3) ,rr(3) ,mulocal2 )
      else
        call tria2( rslt ,ss(2) ,rr(3) ,mulocal2 )
      endif
    elseif (s3r3.ne.R0P0) then
      call tria2( rslt ,ss(3) ,rr(3) ,mulocal2 )
    else
      call tria1( rslt ,rr(3) ,mulocal2 )
    endif
  else
! 0 non-zero internal masses
    call tria0( rslt ,ss ,as ,mulocal2 )
  endif
! exp(eps*gamma_EULER) -> GAMMA(1-2*eps)/GAMMA(1-eps)^2/GAMMA(1+eps)
  rslt(0) = rslt(0) + const*rslt(2)
!
  if (punit.gt.0) then
    if (nonzerothrs) write(punit,*) 'onshell:',trim(myprint(onshellthrs))
    write(punit,*) 'muscale:',trim(myprint(mulocal))
    write(punit,*) ' p1:',trim(myprint(p1))
    write(punit,*) ' p2:',trim(myprint(p2))
    write(punit,*) ' p3:',trim(myprint(p3))
    write(punit,*) ' m1:',trim(myprint(m1))
    write(punit,*) ' m2:',trim(myprint(m2))
    write(punit,*) ' m3:',trim(myprint(m3))
    write(punit,*) 'c0(2):',trim(myprint(rslt(2)))
    write(punit,*) 'c0(1):',trim(myprint(rslt(1)))
    write(punit,*) 'c0(0):',trim(myprint(rslt(0)))
  endif
!
  end subroutine


!*******************************************************************
! calculates
!
!    C   /                      d^(Dim)q
! ------ | --------------------------------------------------------
! i*pi^2 / [q^2-m1][(q+k1)^2-m2][(q+k1+k2)^2-m3][(q+k1+k2+k3)^2-m4]
!
! with  Dim = 4-2*eps
!         C = pi^eps * mu^(2*eps)
!             * GAMMA(1-2*eps)/GAMMA(1-eps)^2/GAMMA(1+eps)
!
! input:  p1=k1^2, p2=k2^2, p3=k3^2, p4=(k1+k2+k3)^2, 
!         p12=(k1+k2)^2, p23=(k2+k3)^2, 
!         m1,m2,m3,m4=squared masses
! output: rslt(0) = eps^0   -coefficient
!         rslt(1) = eps^(-1)-coefficient
!         rslt(2) = eps^(-2)-coefficient
!
! Check the comments in  avh_olo_mp_onshell  to find out how this
! routines decides when to return IR-divergent cases.
!*******************************************************************
!
  subroutine d0rr( rslt ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4 )
!*******************************************************************
!*******************************************************************
  use avh_olo_mp_box
  use avh_olo_mp_boxc
  include 'cts_mpc.h'
   ,intent(out) :: rslt(0:2)
  include 'cts_mpr.h'
   ,intent(in)  :: p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4
  include 'cts_mpr.h'
   :: pp(6),mm(4)
!
  include 'cts_mpc.h'
   :: ss(6),rr(4)
  include 'cts_mpr.h'
   :: smax,ap(6),am(4),as(6),ar(4),s1r2,s2r2,s2r3,s3r4,s4r4
  include 'cts_mpr.h'
   :: mulocal,mulocal2,small,thrs
  integer :: icase,ii
  logical :: useboxc
  include 'cts_mpc.h'
   :: const
  character(25+99) ,parameter :: warning=&
                 'WARNING from OneLOop d0: '//warnonshell
  const=C1P0*TWOPI*TWOPI/48

  if (intro) call hello
!
  pp(1) = p1
  pp(2) = p2
  pp(3) = p3
  pp(4) = p4
  pp(5) = p12
  pp(6) = p23
  mm(1) = m1
  mm(2) = m2
  mm(3) = m3
  mm(4) = m4
!
  smax = R0P0
!
  mulocal = muscale
!
  do ii=1,6
    ap(ii) = abs(pp(ii))
    if (ap(ii).gt.smax) smax = ap(ii)
  enddo
!
  do ii=1,4
    am(ii) = abs(mm(ii))
    if (am(ii).gt.smax) smax = am(ii)
  enddo
!
  small = mp_ml(ap,6)*mp_eps*100
  mulocal2 = mulocal*mulocal
!
  if (smax.eq.R0P0) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop d0: ' &
      ,'all input equal zero, returning 0'
    rslt = C0P0
    return
  endif
!
  if (mulocal2.gt.smax) smax = mulocal2
!
  if (nonzerothrs) then
    thrs = onshellthrs
    do ii=1,4
      if (ap(ii).lt.thrs) ap(ii) = R0P0
      if (am(ii).lt.thrs) am(ii) = R0P0
    enddo
  else
    thrs = onshellthrs*smax
    if (wunit.gt.0) then
    do ii=1,4
      if (R0P0.lt.ap(ii).and.ap(ii).lt.thrs) write(wunit,*) warning
      if (R0P0.lt.am(ii).and.am(ii).lt.thrs) write(wunit,*) warning
    enddo
    endif
  endif
!
  icase = 0
  do ii=1,4
    if (am(ii).gt.R0P0) icase = icase + base(ii)
  enddo
  ss(1)=pp(permtable(1,icase)) ;as(1)=ap(permtable(1,icase))
  ss(2)=pp(permtable(2,icase)) ;as(2)=ap(permtable(2,icase))
  ss(3)=pp(permtable(3,icase)) ;as(3)=ap(permtable(3,icase))
  ss(4)=pp(permtable(4,icase)) ;as(4)=ap(permtable(4,icase))
  ss(5)=pp(permtable(5,icase)) ;as(5)=ap(permtable(5,icase))
  ss(6)=pp(permtable(6,icase)) ;as(6)=ap(permtable(6,icase))
  rr(1)=mm(permtable(1,icase)) ;ar(1)=am(permtable(1,icase))
  rr(2)=mm(permtable(2,icase)) ;ar(2)=am(permtable(2,icase))
  rr(3)=mm(permtable(3,icase)) ;ar(3)=am(permtable(3,icase))
  rr(4)=mm(permtable(4,icase)) ;ar(4)=am(permtable(4,icase))
  icase = casetable(icase)
!
  s1r2 = abs(mp_rl(ss(1)-rr(2))) + abs(mp_ig(ss(1)-rr(2)))
  s2r2 = abs(mp_rl(ss(2)-rr(2))) + abs(mp_ig(ss(2)-rr(2)))
  s2r3 = abs(mp_rl(ss(2)-rr(3))) + abs(mp_ig(ss(2)-rr(3)))
  s3r4 = abs(mp_rl(ss(3)-rr(4))) + abs(mp_ig(ss(3)-rr(4)))
  s4r4 = abs(mp_rl(ss(4)-rr(4))) + abs(mp_ig(ss(4)-rr(4)))
  if (nonzerothrs) then
    if (s1r2.lt.thrs) s1r2 = R0P0
    if (s2r2.lt.thrs) s2r2 = R0P0
    if (s2r3.lt.thrs) s2r3 = R0P0
    if (s3r4.lt.thrs) s3r4 = R0P0
    if (s4r4.lt.thrs) s4r4 = R0P0
  elseif (wunit.gt.0) then
    if (R0P0.lt.s1r2.and.s1r2.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.s2r2.and.s2r2.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.s2r3.and.s2r3.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.s3r4.and.s3r4.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.s4r4.and.s4r4.lt.thrs) write(wunit,*) warning
  endif
!
  if     (icase.eq.4) then
!4 non-zero internal masses
    useboxc = (    (ar(1).ne.R0P0.and.mp_ig(rr(1)).ne.R0P0) &
               .or.(ar(2).ne.R0P0.and.mp_ig(rr(2)).ne.R0P0) &
               .or.(ar(3).ne.R0P0.and.mp_ig(rr(3)).ne.R0P0) &
               .or.(ar(4).ne.R0P0.and.mp_ig(rr(4)).ne.R0P0) &
               .or.(     mp_rl(ss(1)).ge.-small  &
                    .and.mp_rl(ss(2)).ge.-small  &
                    .and.mp_rl(ss(3)).ge.-small  &
                    .and.mp_rl(ss(4)).ge.-small) )
    if (useboxc) then
      call boxc( rslt ,ss,rr ,as )
    else
      call boxf4( rslt ,ss(1),ss(2),ss(3),ss(4),ss(5),ss(6) ,rr(1),rr(2),rr(3),rr(4) )
    endif
  elseif (icase.eq.3) then
!3 non-zero internal masses
    if (s1r2.ne.R0P0.or.s4r4.ne.R0P0) then
      useboxc = (    (ar(1).ne.R0P0.and.mp_ig(rr(1)).ne.R0P0) &
                 .or.(ar(2).ne.R0P0.and.mp_ig(rr(2)).ne.R0P0) &
                 .or.(ar(3).ne.R0P0.and.mp_ig(rr(3)).ne.R0P0) &
                 .or.(ar(4).ne.R0P0.and.mp_ig(rr(4)).ne.R0P0) &
                 .or.(     mp_rl(ss(1)).ge.-small  &
                      .and.mp_rl(ss(2)).ge.-small  &
                      .and.mp_rl(ss(3)).ge.-small  &
                      .and.mp_rl(ss(4)).ge.-small) )
      if (useboxc) then
        call boxc( rslt ,ss,rr ,as )
      else
        call boxf3( rslt, ss,rr )
      endif
    else
      call box16( rslt ,ss(2),ss(3),ss(5),ss(6) ,rr(2),rr(3),rr(4) ,mulocal )
    endif
  elseif (icase.eq.5) then
!2 non-zero internal masses, opposite case
    if     (s1r2.ne.R0P0.or.s4r4.ne.R0P0) then
      if     (s2r2.ne.R0P0.or.s3r4.ne.R0P0) then
        call boxf5( rslt ,ss(1),ss(2),ss(3),ss(4),ss(5),ss(6) ,rr(2),rr(4) )
      else
        call box15( rslt ,ss(1),ss(4),ss(5),ss(6) ,rr(2),rr(4) ,mulocal )
      endif
    elseif (s2r2.ne.R0P0.or.s3r4.ne.R0P0) then
      call box15( rslt ,ss(2),ss(3),ss(5),ss(6) ,rr(2),rr(4) ,mulocal )
    else
      call box14( rslt ,ss(5),ss(6) ,rr(2),rr(4) ,mulocal )
    endif
  elseif (icase.eq.2) then
!2 non-zero internal masses, adjacent case
    if     (as(1).ne.R0P0) then
      call boxf2( rslt ,ss(1),ss(2),ss(3),ss(4),ss(5),ss(6) ,rr(3),rr(4) )
    elseif (s2r3.ne.R0P0) then
      if     (s4r4.ne.R0P0) then
        call box13( rslt ,ss(2),ss(3),ss(4),ss(5),ss(6) ,rr(3),rr(4) ,mulocal )
      else
        call box12( rslt ,ss(3),ss(2),ss(6),ss(5) ,rr(4),rr(3) ,mulocal )
      endif
    elseif (s4r4.ne.R0P0) then
      call box12( rslt ,ss(3),ss(4),ss(5),ss(6) ,rr(3),rr(4) ,mulocal )
    else
      call box11( rslt ,ss(3),ss(5),ss(6) ,rr(3),rr(4) ,mulocal )
    endif
  elseif (icase.eq.1) then
!1 non-zero internal mass
    if     (as(1).ne.R0P0) then
      if      (as(2).ne.R0P0) then
        call boxf1( rslt ,ss(1),ss(2),ss(3),ss(4),ss(5),ss(6) ,rr(4) )
      else
        if     (s3r4.ne.R0P0) then
          call box10( rslt ,ss(1),ss(4),ss(3),ss(5),ss(6) ,rr(4) ,mulocal )
        else
          call box09( rslt ,ss(1),ss(4),ss(5),ss(6) ,rr(4) ,mulocal )
        endif
      endif
    elseif (as(2).ne.R0P0) then
      if      (s4r4.ne.R0P0) then
        call box10( rslt ,ss(2),ss(3),ss(4),ss(5),ss(6) ,rr(4) ,mulocal )
      else
        call box09( rslt ,ss(2),ss(3),ss(5),ss(6) ,rr(4) ,mulocal )
      endif
    else
      if     (s3r4.ne.R0P0) then
        if     (s4r4.ne.R0P0) then
          call box08( rslt ,ss(3),ss(4),ss(5),ss(6) ,rr(4) ,mulocal )
        else
          call box07( rslt ,ss(3),ss(5),ss(6) ,rr(4) ,mulocal )
        endif
      elseif (s4r4.ne.R0P0) then
        call box07( rslt ,ss(4),ss(5),ss(6) ,rr(4) ,mulocal )
      else
        call box06( rslt ,ss(5),ss(6) ,rr(4) ,mulocal )
      endif
    endif
  else
!0 non-zero internal mass
    call box00( rslt ,ss ,as ,mulocal )
  endif
!exp(eps*gamma_EULER) -> GAMMA(1-2*eps)/GAMMA(1-eps)^2/GAMMA(1+eps)
  rslt(0) = rslt(0) + const*rslt(2)
!
  if (punit.gt.0) then
    if (nonzerothrs) write(punit,*) 'onshell:',trim(myprint(onshellthrs))
    write(punit,*) 'muscale:',trim(myprint(mulocal))
    write(punit,*) ' p1:',trim(myprint(p1))
    write(punit,*) ' p2:',trim(myprint(p2))
    write(punit,*) ' p3:',trim(myprint(p3))
    write(punit,*) ' p4:',trim(myprint(p4))
    write(punit,*) 'p12:',trim(myprint(p12))
    write(punit,*) 'p23:',trim(myprint(p23))
    write(punit,*) ' m1:',trim(myprint(m1))
    write(punit,*) ' m2:',trim(myprint(m2))
    write(punit,*) ' m3:',trim(myprint(m3))
    write(punit,*) ' m4:',trim(myprint(m4))
    write(punit,*) 'd0(2):',trim(myprint(rslt(2)))
    write(punit,*) 'd0(1):',trim(myprint(rslt(1)))
    write(punit,*) 'd0(0):',trim(myprint(rslt(0)))
  endif
!
  end subroutine
!
  subroutine d0rrr( rslt ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4 ,rmu )
!*******************************************************************
!*******************************************************************
  use avh_olo_mp_box
  use avh_olo_mp_boxc
  include 'cts_mpc.h'
   ,intent(out) :: rslt(0:2)
  include 'cts_mpr.h'
   ,intent(in)  :: p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4 ,rmu
  include 'cts_mpr.h'
   :: pp(6),mm(4)
!
  include 'cts_mpc.h'
   :: ss(6),rr(4)
  include 'cts_mpr.h'
   :: smax,ap(6),am(4),as(6),ar(4),s1r2,s2r2,s2r3,s3r4,s4r4
  include 'cts_mpr.h'
   :: mulocal,mulocal2,small,thrs
  integer :: icase,ii
  logical :: useboxc
  include 'cts_mpc.h'
   :: const
  character(25+99) ,parameter :: warning=&
                 'WARNING from OneLOop d0: '//warnonshell
  const=C1P0*TWOPI*TWOPI/48
  if (intro) call hello
!
  pp(1) = p1
  pp(2) = p2
  pp(3) = p3
  pp(4) = p4
  pp(5) = p12
  pp(6) = p23
  mm(1) = m1
  mm(2) = m2
  mm(3) = m3
  mm(4) = m4
!
  smax = R0P0
!
  mulocal = rmu
!
  do ii=1,6
    ap(ii) = abs(pp(ii))
    if (ap(ii).gt.smax) smax = ap(ii)
  enddo
!
  do ii=1,4
    am(ii) = abs(mm(ii))
    if (am(ii).gt.smax) smax = am(ii)
  enddo
!
  small = mp_ml(ap,6)*mp_eps*100
  mulocal2 = mulocal*mulocal
!
  if (smax.eq.R0P0) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop d0: ' &
      ,'all input equal zero, returning 0'
    rslt = C0P0
    return
  endif
!
  if (mulocal2.gt.smax) smax = mulocal2
!
  if (nonzerothrs) then
    thrs = onshellthrs
    do ii=1,4
      if (ap(ii).lt.thrs) ap(ii) = R0P0
      if (am(ii).lt.thrs) am(ii) = R0P0
    enddo
  else
    thrs = onshellthrs*smax
    if (wunit.gt.0) then
    do ii=1,4
      if (R0P0.lt.ap(ii).and.ap(ii).lt.thrs) write(wunit,*) warning
      if (R0P0.lt.am(ii).and.am(ii).lt.thrs) write(wunit,*) warning
    enddo
    endif
  endif
!
  icase = 0
  do ii=1,4
    if (am(ii).gt.R0P0) icase = icase + base(ii)
  enddo
  ss(1)=pp(permtable(1,icase)) ;as(1)=ap(permtable(1,icase))
  ss(2)=pp(permtable(2,icase)) ;as(2)=ap(permtable(2,icase))
  ss(3)=pp(permtable(3,icase)) ;as(3)=ap(permtable(3,icase))
  ss(4)=pp(permtable(4,icase)) ;as(4)=ap(permtable(4,icase))
  ss(5)=pp(permtable(5,icase)) ;as(5)=ap(permtable(5,icase))
  ss(6)=pp(permtable(6,icase)) ;as(6)=ap(permtable(6,icase))
  rr(1)=mm(permtable(1,icase)) ;ar(1)=am(permtable(1,icase))
  rr(2)=mm(permtable(2,icase)) ;ar(2)=am(permtable(2,icase))
  rr(3)=mm(permtable(3,icase)) ;ar(3)=am(permtable(3,icase))
  rr(4)=mm(permtable(4,icase)) ;ar(4)=am(permtable(4,icase))
  icase = casetable(icase)
!
  s1r2 = abs(mp_rl(ss(1)-rr(2))) + abs(mp_ig(ss(1)-rr(2)))
  s2r2 = abs(mp_rl(ss(2)-rr(2))) + abs(mp_ig(ss(2)-rr(2)))
  s2r3 = abs(mp_rl(ss(2)-rr(3))) + abs(mp_ig(ss(2)-rr(3)))
  s3r4 = abs(mp_rl(ss(3)-rr(4))) + abs(mp_ig(ss(3)-rr(4)))
  s4r4 = abs(mp_rl(ss(4)-rr(4))) + abs(mp_ig(ss(4)-rr(4)))
  if (nonzerothrs) then
    if (s1r2.lt.thrs) s1r2 = R0P0
    if (s2r2.lt.thrs) s2r2 = R0P0
    if (s2r3.lt.thrs) s2r3 = R0P0
    if (s3r4.lt.thrs) s3r4 = R0P0
    if (s4r4.lt.thrs) s4r4 = R0P0
  elseif (wunit.gt.0) then
    if (R0P0.lt.s1r2.and.s1r2.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.s2r2.and.s2r2.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.s2r3.and.s2r3.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.s3r4.and.s3r4.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.s4r4.and.s4r4.lt.thrs) write(wunit,*) warning
  endif
!
  if     (icase.eq.4) then
!4 non-zero internal masses
    useboxc = (    (ar(1).ne.R0P0.and.mp_ig(rr(1)).ne.R0P0) &
               .or.(ar(2).ne.R0P0.and.mp_ig(rr(2)).ne.R0P0) &
               .or.(ar(3).ne.R0P0.and.mp_ig(rr(3)).ne.R0P0) &
               .or.(ar(4).ne.R0P0.and.mp_ig(rr(4)).ne.R0P0) &
               .or.(     mp_rl(ss(1)).ge.-small  &
                    .and.mp_rl(ss(2)).ge.-small  &
                    .and.mp_rl(ss(3)).ge.-small  &
                    .and.mp_rl(ss(4)).ge.-small) )
    if (useboxc) then
      call boxc( rslt ,ss,rr ,as )
    else
      call boxf4( rslt ,ss(1),ss(2),ss(3),ss(4),ss(5),ss(6) ,rr(1),rr(2),rr(3),rr(4) )
    endif
  elseif (icase.eq.3) then
!3 non-zero internal masses
    if (s1r2.ne.R0P0.or.s4r4.ne.R0P0) then
      useboxc = (    (ar(1).ne.R0P0.and.mp_ig(rr(1)).ne.R0P0) &
                 .or.(ar(2).ne.R0P0.and.mp_ig(rr(2)).ne.R0P0) &
                 .or.(ar(3).ne.R0P0.and.mp_ig(rr(3)).ne.R0P0) &
                 .or.(ar(4).ne.R0P0.and.mp_ig(rr(4)).ne.R0P0) &
                 .or.(     mp_rl(ss(1)).ge.-small  &
                      .and.mp_rl(ss(2)).ge.-small  &
                      .and.mp_rl(ss(3)).ge.-small  &
                      .and.mp_rl(ss(4)).ge.-small) )
      if (useboxc) then
        call boxc( rslt ,ss,rr ,as )
      else
        call boxf3( rslt, ss,rr )
      endif
    else
      call box16( rslt ,ss(2),ss(3),ss(5),ss(6) ,rr(2),rr(3),rr(4) ,mulocal )
    endif
  elseif (icase.eq.5) then
!2 non-zero internal masses, opposite case
    if     (s1r2.ne.R0P0.or.s4r4.ne.R0P0) then
      if     (s2r2.ne.R0P0.or.s3r4.ne.R0P0) then
        call boxf5( rslt ,ss(1),ss(2),ss(3),ss(4),ss(5),ss(6) ,rr(2),rr(4) )
      else
        call box15( rslt ,ss(1),ss(4),ss(5),ss(6) ,rr(2),rr(4) ,mulocal )
      endif
    elseif (s2r2.ne.R0P0.or.s3r4.ne.R0P0) then
      call box15( rslt ,ss(2),ss(3),ss(5),ss(6) ,rr(2),rr(4) ,mulocal )
    else
      call box14( rslt ,ss(5),ss(6) ,rr(2),rr(4) ,mulocal )
    endif
  elseif (icase.eq.2) then
!2 non-zero internal masses, adjacent case
    if     (as(1).ne.R0P0) then
      call boxf2( rslt ,ss(1),ss(2),ss(3),ss(4),ss(5),ss(6) ,rr(3),rr(4) )
    elseif (s2r3.ne.R0P0) then
      if     (s4r4.ne.R0P0) then
        call box13( rslt ,ss(2),ss(3),ss(4),ss(5),ss(6) ,rr(3),rr(4) ,mulocal )
      else
        call box12( rslt ,ss(3),ss(2),ss(6),ss(5) ,rr(4),rr(3) ,mulocal )
      endif
    elseif (s4r4.ne.R0P0) then
      call box12( rslt ,ss(3),ss(4),ss(5),ss(6) ,rr(3),rr(4) ,mulocal )
    else
      call box11( rslt ,ss(3),ss(5),ss(6) ,rr(3),rr(4) ,mulocal )
    endif
  elseif (icase.eq.1) then
!1 non-zero internal mass
    if     (as(1).ne.R0P0) then
      if      (as(2).ne.R0P0) then
        call boxf1( rslt ,ss(1),ss(2),ss(3),ss(4),ss(5),ss(6) ,rr(4) )
      else
        if     (s3r4.ne.R0P0) then
          call box10( rslt ,ss(1),ss(4),ss(3),ss(5),ss(6) ,rr(4) ,mulocal )
        else
          call box09( rslt ,ss(1),ss(4),ss(5),ss(6) ,rr(4) ,mulocal )
        endif
      endif
    elseif (as(2).ne.R0P0) then
      if      (s4r4.ne.R0P0) then
        call box10( rslt ,ss(2),ss(3),ss(4),ss(5),ss(6) ,rr(4) ,mulocal )
      else
        call box09( rslt ,ss(2),ss(3),ss(5),ss(6) ,rr(4) ,mulocal )
      endif
    else
      if     (s3r4.ne.R0P0) then
        if     (s4r4.ne.R0P0) then
          call box08( rslt ,ss(3),ss(4),ss(5),ss(6) ,rr(4) ,mulocal )
        else
          call box07( rslt ,ss(3),ss(5),ss(6) ,rr(4) ,mulocal )
        endif
      elseif (s4r4.ne.R0P0) then
        call box07( rslt ,ss(4),ss(5),ss(6) ,rr(4) ,mulocal )
      else
        call box06( rslt ,ss(5),ss(6) ,rr(4) ,mulocal )
      endif
    endif
  else
!0 non-zero internal mass
    call box00( rslt ,ss ,as ,mulocal )
  endif
!exp(eps*gamma_EULER) -> GAMMA(1-2*eps)/GAMMA(1-eps)^2/GAMMA(1+eps)
  rslt(0) = rslt(0) + const*rslt(2)
!
  if (punit.gt.0) then
    if (nonzerothrs) write(punit,*) 'onshell:',trim(myprint(onshellthrs))
    write(punit,*) 'muscale:',trim(myprint(mulocal))
    write(punit,*) ' p1:',trim(myprint(p1))
    write(punit,*) ' p2:',trim(myprint(p2))
    write(punit,*) ' p3:',trim(myprint(p3))
    write(punit,*) ' p4:',trim(myprint(p4))
    write(punit,*) 'p12:',trim(myprint(p12))
    write(punit,*) 'p23:',trim(myprint(p23))
    write(punit,*) ' m1:',trim(myprint(m1))
    write(punit,*) ' m2:',trim(myprint(m2))
    write(punit,*) ' m3:',trim(myprint(m3))
    write(punit,*) ' m4:',trim(myprint(m4))
    write(punit,*) 'd0(2):',trim(myprint(rslt(2)))
    write(punit,*) 'd0(1):',trim(myprint(rslt(1)))
    write(punit,*) 'd0(0):',trim(myprint(rslt(0)))
  endif
!
  end subroutine
!
  subroutine d0rc( rslt ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4 )
!*******************************************************************
!*******************************************************************
  use avh_olo_mp_box
  use avh_olo_mp_boxc
  include 'cts_mpc.h'
   ,intent(out) :: rslt(0:2)
  include 'cts_mpr.h'
   ,intent(in)  :: p1,p2,p3,p4,p12,p23
  include 'cts_mpc.h'
   ,intent(in)  :: m1,m2,m3,m4
  include 'cts_mpr.h'
   :: pp(6)
  include 'cts_mpc.h'
   :: mm(4)
  include 'cts_mpr.h'
   :: hh
!
  include 'cts_mpc.h'
   :: ss(6),rr(4)
  include 'cts_mpr.h'
   :: smax,ap(6),am(4),as(6),ar(4),s1r2,s2r2,s2r3,s3r4,s4r4
  include 'cts_mpr.h'
   :: mulocal,mulocal2,small,thrs
  integer :: icase,ii
  logical :: useboxc
  include 'cts_mpc.h'
   :: const
  character(25+99) ,parameter :: warning=&
                 'WARNING from OneLOop d0: '//warnonshell
  const=C1P0*TWOPI*TWOPI/48
  if (intro) call hello
!
  pp(1) = p1
  pp(2) = p2
  pp(3) = p3
  pp(4) = p4
  pp(5) = p12
  pp(6) = p23
  mm(1) = m1
  mm(2) = m2
  mm(3) = m3
  mm(4) = m4
!
  smax = R0P0
!
  mulocal = muscale
!
  do ii=1,6
    ap(ii) = abs(pp(ii))
    if (ap(ii).gt.smax) smax = ap(ii)
  enddo
!
  do ii=1,4
    am(ii) = mp_rl(mm(ii))
    hh = mp_ig(mm(ii))
    if (hh.gt.R0P0) then
      if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop d0: ' &
        ,'mass-squared has positive imaginary part, switching its sign.'
      mm(ii) = mp_cx( am(ii) ,-hh )
    endif
    am(ii) = abs(am(ii)) + abs(hh)
    if (am(ii).gt.smax) smax = am(ii)
  enddo
!
  small = mp_ml(ap,6)*mp_eps*100
  mulocal2 = mulocal*mulocal
!
  if (smax.eq.R0P0) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop d0: ' &
      ,'all input equal zero, returning 0'
    rslt = C0P0
    return
  endif
!
  if (mulocal2.gt.smax) smax = mulocal2
!
  if (nonzerothrs) then
    thrs = onshellthrs
    do ii=1,4
      if (ap(ii).lt.thrs) ap(ii) = R0P0
      if (am(ii).lt.thrs) am(ii) = R0P0
    enddo
  else
    thrs = onshellthrs*smax
    if (wunit.gt.0) then
    do ii=1,4
      if (R0P0.lt.ap(ii).and.ap(ii).lt.thrs) write(wunit,*) warning
      if (R0P0.lt.am(ii).and.am(ii).lt.thrs) write(wunit,*) warning
    enddo
    endif
  endif
!
  icase = 0
  do ii=1,4
    if (am(ii).gt.R0P0) icase = icase + base(ii)
  enddo
  ss(1)=pp(permtable(1,icase)) ;as(1)=ap(permtable(1,icase))
  ss(2)=pp(permtable(2,icase)) ;as(2)=ap(permtable(2,icase))
  ss(3)=pp(permtable(3,icase)) ;as(3)=ap(permtable(3,icase))
  ss(4)=pp(permtable(4,icase)) ;as(4)=ap(permtable(4,icase))
  ss(5)=pp(permtable(5,icase)) ;as(5)=ap(permtable(5,icase))
  ss(6)=pp(permtable(6,icase)) ;as(6)=ap(permtable(6,icase))
  rr(1)=mm(permtable(1,icase)) ;ar(1)=am(permtable(1,icase))
  rr(2)=mm(permtable(2,icase)) ;ar(2)=am(permtable(2,icase))
  rr(3)=mm(permtable(3,icase)) ;ar(3)=am(permtable(3,icase))
  rr(4)=mm(permtable(4,icase)) ;ar(4)=am(permtable(4,icase))
  icase = casetable(icase)
!
  s1r2 = abs(mp_rl(ss(1)-rr(2))) + abs(mp_ig(ss(1)-rr(2)))
  s2r2 = abs(mp_rl(ss(2)-rr(2))) + abs(mp_ig(ss(2)-rr(2)))
  s2r3 = abs(mp_rl(ss(2)-rr(3))) + abs(mp_ig(ss(2)-rr(3)))
  s3r4 = abs(mp_rl(ss(3)-rr(4))) + abs(mp_ig(ss(3)-rr(4)))
  s4r4 = abs(mp_rl(ss(4)-rr(4))) + abs(mp_ig(ss(4)-rr(4)))
  if (nonzerothrs) then
    if (s1r2.lt.thrs) s1r2 = R0P0
    if (s2r2.lt.thrs) s2r2 = R0P0
    if (s2r3.lt.thrs) s2r3 = R0P0
    if (s3r4.lt.thrs) s3r4 = R0P0
    if (s4r4.lt.thrs) s4r4 = R0P0
  elseif (wunit.gt.0) then
    if (R0P0.lt.s1r2.and.s1r2.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.s2r2.and.s2r2.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.s2r3.and.s2r3.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.s3r4.and.s3r4.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.s4r4.and.s4r4.lt.thrs) write(wunit,*) warning
  endif
!
  if     (icase.eq.4) then
!4 non-zero internal masses
    useboxc = (    (ar(1).ne.R0P0.and.mp_ig(rr(1)).ne.R0P0) &
               .or.(ar(2).ne.R0P0.and.mp_ig(rr(2)).ne.R0P0) &
               .or.(ar(3).ne.R0P0.and.mp_ig(rr(3)).ne.R0P0) &
               .or.(ar(4).ne.R0P0.and.mp_ig(rr(4)).ne.R0P0) &
               .or.(     mp_rl(ss(1)).ge.-small  &
                    .and.mp_rl(ss(2)).ge.-small  &
                    .and.mp_rl(ss(3)).ge.-small  &
                    .and.mp_rl(ss(4)).ge.-small) )
    if (useboxc) then
      call boxc( rslt ,ss,rr ,as )
    else
      call boxf4( rslt ,ss(1),ss(2),ss(3),ss(4),ss(5),ss(6) ,rr(1),rr(2),rr(3),rr(4) )
    endif
  elseif (icase.eq.3) then
!3 non-zero internal masses
    if (s1r2.ne.R0P0.or.s4r4.ne.R0P0) then
      useboxc = (    (ar(1).ne.R0P0.and.mp_ig(rr(1)).ne.R0P0) &
                 .or.(ar(2).ne.R0P0.and.mp_ig(rr(2)).ne.R0P0) &
                 .or.(ar(3).ne.R0P0.and.mp_ig(rr(3)).ne.R0P0) &
                 .or.(ar(4).ne.R0P0.and.mp_ig(rr(4)).ne.R0P0) &
                 .or.(     mp_rl(ss(1)).ge.-small  &
                      .and.mp_rl(ss(2)).ge.-small  &
                      .and.mp_rl(ss(3)).ge.-small  &
                      .and.mp_rl(ss(4)).ge.-small) )
      if (useboxc) then
        call boxc( rslt ,ss,rr ,as )
      else
        call boxf3( rslt, ss,rr )
      endif
    else
      call box16( rslt ,ss(2),ss(3),ss(5),ss(6) ,rr(2),rr(3),rr(4) ,mulocal )
    endif
  elseif (icase.eq.5) then
!2 non-zero internal masses, opposite case
    if     (s1r2.ne.R0P0.or.s4r4.ne.R0P0) then
      if     (s2r2.ne.R0P0.or.s3r4.ne.R0P0) then
        call boxf5( rslt ,ss(1),ss(2),ss(3),ss(4),ss(5),ss(6) ,rr(2),rr(4) )
      else
        call box15( rslt ,ss(1),ss(4),ss(5),ss(6) ,rr(2),rr(4) ,mulocal )
      endif
    elseif (s2r2.ne.R0P0.or.s3r4.ne.R0P0) then
      call box15( rslt ,ss(2),ss(3),ss(5),ss(6) ,rr(2),rr(4) ,mulocal )
    else
      call box14( rslt ,ss(5),ss(6) ,rr(2),rr(4) ,mulocal )
    endif
  elseif (icase.eq.2) then
!2 non-zero internal masses, adjacent case
    if     (as(1).ne.R0P0) then
      call boxf2( rslt ,ss(1),ss(2),ss(3),ss(4),ss(5),ss(6) ,rr(3),rr(4) )
    elseif (s2r3.ne.R0P0) then
      if     (s4r4.ne.R0P0) then
        call box13( rslt ,ss(2),ss(3),ss(4),ss(5),ss(6) ,rr(3),rr(4) ,mulocal )
      else
        call box12( rslt ,ss(3),ss(2),ss(6),ss(5) ,rr(4),rr(3) ,mulocal )
      endif
    elseif (s4r4.ne.R0P0) then
      call box12( rslt ,ss(3),ss(4),ss(5),ss(6) ,rr(3),rr(4) ,mulocal )
    else
      call box11( rslt ,ss(3),ss(5),ss(6) ,rr(3),rr(4) ,mulocal )
    endif
  elseif (icase.eq.1) then
!1 non-zero internal mass
    if     (as(1).ne.R0P0) then
      if      (as(2).ne.R0P0) then
        call boxf1( rslt ,ss(1),ss(2),ss(3),ss(4),ss(5),ss(6) ,rr(4) )
      else
        if     (s3r4.ne.R0P0) then
          call box10( rslt ,ss(1),ss(4),ss(3),ss(5),ss(6) ,rr(4) ,mulocal )
        else
          call box09( rslt ,ss(1),ss(4),ss(5),ss(6) ,rr(4) ,mulocal )
        endif
      endif
    elseif (as(2).ne.R0P0) then
      if      (s4r4.ne.R0P0) then
        call box10( rslt ,ss(2),ss(3),ss(4),ss(5),ss(6) ,rr(4) ,mulocal )
      else
        call box09( rslt ,ss(2),ss(3),ss(5),ss(6) ,rr(4) ,mulocal )
      endif
    else
      if     (s3r4.ne.R0P0) then
        if     (s4r4.ne.R0P0) then
          call box08( rslt ,ss(3),ss(4),ss(5),ss(6) ,rr(4) ,mulocal )
        else
          call box07( rslt ,ss(3),ss(5),ss(6) ,rr(4) ,mulocal )
        endif
      elseif (s4r4.ne.R0P0) then
        call box07( rslt ,ss(4),ss(5),ss(6) ,rr(4) ,mulocal )
      else
        call box06( rslt ,ss(5),ss(6) ,rr(4) ,mulocal )
      endif
    endif
  else
!0 non-zero internal mass
    call box00( rslt ,ss ,as ,mulocal )
  endif
!exp(eps*gamma_EULER) -> GAMMA(1-2*eps)/GAMMA(1-eps)^2/GAMMA(1+eps)
  rslt(0) = rslt(0) + const*rslt(2)
!
  if (punit.gt.0) then
    if (nonzerothrs) write(punit,*) 'onshell:',trim(myprint(onshellthrs))
    write(punit,*) 'muscale:',trim(myprint(mulocal))
    write(punit,*) ' p1:',trim(myprint(p1))
    write(punit,*) ' p2:',trim(myprint(p2))
    write(punit,*) ' p3:',trim(myprint(p3))
    write(punit,*) ' p4:',trim(myprint(p4))
    write(punit,*) 'p12:',trim(myprint(p12))
    write(punit,*) 'p23:',trim(myprint(p23))
    write(punit,*) ' m1:',trim(myprint(m1))
    write(punit,*) ' m2:',trim(myprint(m2))
    write(punit,*) ' m3:',trim(myprint(m3))
    write(punit,*) ' m4:',trim(myprint(m4))
    write(punit,*) 'd0(2):',trim(myprint(rslt(2)))
    write(punit,*) 'd0(1):',trim(myprint(rslt(1)))
    write(punit,*) 'd0(0):',trim(myprint(rslt(0)))
  endif
!
  end subroutine
!
  subroutine d0rcr( rslt ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4 ,rmu )
!*******************************************************************
!*******************************************************************
  use avh_olo_mp_box
  use avh_olo_mp_boxc
  include 'cts_mpc.h'
   ,intent(out) :: rslt(0:2)
  include 'cts_mpr.h'
   ,intent(in)  :: p1,p2,p3,p4,p12,p23 ,rmu
  include 'cts_mpc.h'
   ,intent(in)  :: m1,m2,m3,m4
  include 'cts_mpr.h'
   :: pp(6)
  include 'cts_mpc.h'
   :: mm(4)
  include 'cts_mpr.h'
   :: hh
!
  include 'cts_mpc.h'
   :: ss(6),rr(4)
  include 'cts_mpr.h'
   :: smax,ap(6),am(4),as(6),ar(4),s1r2,s2r2,s2r3,s3r4,s4r4
  include 'cts_mpr.h'
   :: mulocal,mulocal2,small,thrs
  integer :: icase,ii
  logical :: useboxc
  include 'cts_mpc.h'
   :: const
  character(25+99) ,parameter :: warning=&
                 'WARNING from OneLOop d0: '//warnonshell
  const=C1P0*TWOPI*TWOPI/48
  if (intro) call hello
!
  pp(1) = p1
  pp(2) = p2
  pp(3) = p3
  pp(4) = p4
  pp(5) = p12
  pp(6) = p23
  mm(1) = m1
  mm(2) = m2
  mm(3) = m3
  mm(4) = m4
!
  smax = R0P0
!
  mulocal = rmu
!
  do ii=1,6
    ap(ii) = abs(pp(ii))
    if (ap(ii).gt.smax) smax = ap(ii)
  enddo
!
  do ii=1,4
    am(ii) = mp_rl(mm(ii))
    hh = mp_ig(mm(ii))
    if (hh.gt.R0P0) then
      if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop d0: ' &
        ,'mass-squared has positive imaginary part, switching its sign.'
      mm(ii) = mp_cx( am(ii) ,-hh )
    endif
    am(ii) = abs(am(ii)) + abs(hh)
    if (am(ii).gt.smax) smax = am(ii)
  enddo
!
  small = mp_ml(ap,6)*mp_eps*100
  mulocal2 = mulocal*mulocal
!
  if (smax.eq.R0P0) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop d0: ' &
      ,'all input equal zero, returning 0'
    rslt = C0P0
    return
  endif
!
  if (mulocal2.gt.smax) smax = mulocal2
!
  if (nonzerothrs) then
    thrs = onshellthrs
    do ii=1,4
      if (ap(ii).lt.thrs) ap(ii) = R0P0
      if (am(ii).lt.thrs) am(ii) = R0P0
    enddo
  else
    thrs = onshellthrs*smax
    if (wunit.gt.0) then
    do ii=1,4
      if (R0P0.lt.ap(ii).and.ap(ii).lt.thrs) write(wunit,*) warning
      if (R0P0.lt.am(ii).and.am(ii).lt.thrs) write(wunit,*) warning
    enddo
    endif
  endif
!
  icase = 0
  do ii=1,4
    if (am(ii).gt.R0P0) icase = icase + base(ii)
  enddo
  ss(1)=pp(permtable(1,icase)) ;as(1)=ap(permtable(1,icase))
  ss(2)=pp(permtable(2,icase)) ;as(2)=ap(permtable(2,icase))
  ss(3)=pp(permtable(3,icase)) ;as(3)=ap(permtable(3,icase))
  ss(4)=pp(permtable(4,icase)) ;as(4)=ap(permtable(4,icase))
  ss(5)=pp(permtable(5,icase)) ;as(5)=ap(permtable(5,icase))
  ss(6)=pp(permtable(6,icase)) ;as(6)=ap(permtable(6,icase))
  rr(1)=mm(permtable(1,icase)) ;ar(1)=am(permtable(1,icase))
  rr(2)=mm(permtable(2,icase)) ;ar(2)=am(permtable(2,icase))
  rr(3)=mm(permtable(3,icase)) ;ar(3)=am(permtable(3,icase))
  rr(4)=mm(permtable(4,icase)) ;ar(4)=am(permtable(4,icase))
  icase = casetable(icase)
!
  s1r2 = abs(mp_rl(ss(1)-rr(2))) + abs(mp_ig(ss(1)-rr(2)))
  s2r2 = abs(mp_rl(ss(2)-rr(2))) + abs(mp_ig(ss(2)-rr(2)))
  s2r3 = abs(mp_rl(ss(2)-rr(3))) + abs(mp_ig(ss(2)-rr(3)))
  s3r4 = abs(mp_rl(ss(3)-rr(4))) + abs(mp_ig(ss(3)-rr(4)))
  s4r4 = abs(mp_rl(ss(4)-rr(4))) + abs(mp_ig(ss(4)-rr(4)))
  if (nonzerothrs) then
    if (s1r2.lt.thrs) s1r2 = R0P0
    if (s2r2.lt.thrs) s2r2 = R0P0
    if (s2r3.lt.thrs) s2r3 = R0P0
    if (s3r4.lt.thrs) s3r4 = R0P0
    if (s4r4.lt.thrs) s4r4 = R0P0
  elseif (wunit.gt.0) then
    if (R0P0.lt.s1r2.and.s1r2.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.s2r2.and.s2r2.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.s2r3.and.s2r3.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.s3r4.and.s3r4.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.s4r4.and.s4r4.lt.thrs) write(wunit,*) warning
  endif
!
  if     (icase.eq.4) then
!4 non-zero internal masses
    useboxc = (    (ar(1).ne.R0P0.and.mp_ig(rr(1)).ne.R0P0) &
               .or.(ar(2).ne.R0P0.and.mp_ig(rr(2)).ne.R0P0) &
               .or.(ar(3).ne.R0P0.and.mp_ig(rr(3)).ne.R0P0) &
               .or.(ar(4).ne.R0P0.and.mp_ig(rr(4)).ne.R0P0) &
               .or.(     mp_rl(ss(1)).ge.-small  &
                    .and.mp_rl(ss(2)).ge.-small  &
                    .and.mp_rl(ss(3)).ge.-small  &
                    .and.mp_rl(ss(4)).ge.-small) )
    if (useboxc) then
      call boxc( rslt ,ss,rr ,as )
    else
      call boxf4( rslt ,ss(1),ss(2),ss(3),ss(4),ss(5),ss(6) ,rr(1),rr(2),rr(3),rr(4) )
    endif
  elseif (icase.eq.3) then
!3 non-zero internal masses
    if (s1r2.ne.R0P0.or.s4r4.ne.R0P0) then
      useboxc = (    (ar(1).ne.R0P0.and.mp_ig(rr(1)).ne.R0P0) &
                 .or.(ar(2).ne.R0P0.and.mp_ig(rr(2)).ne.R0P0) &
                 .or.(ar(3).ne.R0P0.and.mp_ig(rr(3)).ne.R0P0) &
                 .or.(ar(4).ne.R0P0.and.mp_ig(rr(4)).ne.R0P0) &
                 .or.(     mp_rl(ss(1)).ge.-small  &
                      .and.mp_rl(ss(2)).ge.-small  &
                      .and.mp_rl(ss(3)).ge.-small  &
                      .and.mp_rl(ss(4)).ge.-small) )
      if (useboxc) then
        call boxc( rslt ,ss,rr ,as )
      else
        call boxf3( rslt, ss,rr )
      endif
    else
      call box16( rslt ,ss(2),ss(3),ss(5),ss(6) ,rr(2),rr(3),rr(4) ,mulocal )
    endif
  elseif (icase.eq.5) then
!2 non-zero internal masses, opposite case
    if     (s1r2.ne.R0P0.or.s4r4.ne.R0P0) then
      if     (s2r2.ne.R0P0.or.s3r4.ne.R0P0) then
        call boxf5( rslt ,ss(1),ss(2),ss(3),ss(4),ss(5),ss(6) ,rr(2),rr(4) )
      else
        call box15( rslt ,ss(1),ss(4),ss(5),ss(6) ,rr(2),rr(4) ,mulocal )
      endif
    elseif (s2r2.ne.R0P0.or.s3r4.ne.R0P0) then
      call box15( rslt ,ss(2),ss(3),ss(5),ss(6) ,rr(2),rr(4) ,mulocal )
    else
      call box14( rslt ,ss(5),ss(6) ,rr(2),rr(4) ,mulocal )
    endif
  elseif (icase.eq.2) then
!2 non-zero internal masses, adjacent case
    if     (as(1).ne.R0P0) then
      call boxf2( rslt ,ss(1),ss(2),ss(3),ss(4),ss(5),ss(6) ,rr(3),rr(4) )
    elseif (s2r3.ne.R0P0) then
      if     (s4r4.ne.R0P0) then
        call box13( rslt ,ss(2),ss(3),ss(4),ss(5),ss(6) ,rr(3),rr(4) ,mulocal )
      else
        call box12( rslt ,ss(3),ss(2),ss(6),ss(5) ,rr(4),rr(3) ,mulocal )
      endif
    elseif (s4r4.ne.R0P0) then
      call box12( rslt ,ss(3),ss(4),ss(5),ss(6) ,rr(3),rr(4) ,mulocal )
    else
      call box11( rslt ,ss(3),ss(5),ss(6) ,rr(3),rr(4) ,mulocal )
    endif
  elseif (icase.eq.1) then
!1 non-zero internal mass
    if     (as(1).ne.R0P0) then
      if      (as(2).ne.R0P0) then
        call boxf1( rslt ,ss(1),ss(2),ss(3),ss(4),ss(5),ss(6) ,rr(4) )
      else
        if     (s3r4.ne.R0P0) then
          call box10( rslt ,ss(1),ss(4),ss(3),ss(5),ss(6) ,rr(4) ,mulocal )
        else
          call box09( rslt ,ss(1),ss(4),ss(5),ss(6) ,rr(4) ,mulocal )
        endif
      endif
    elseif (as(2).ne.R0P0) then
      if      (s4r4.ne.R0P0) then
        call box10( rslt ,ss(2),ss(3),ss(4),ss(5),ss(6) ,rr(4) ,mulocal )
      else
        call box09( rslt ,ss(2),ss(3),ss(5),ss(6) ,rr(4) ,mulocal )
      endif
    else
      if     (s3r4.ne.R0P0) then
        if     (s4r4.ne.R0P0) then
          call box08( rslt ,ss(3),ss(4),ss(5),ss(6) ,rr(4) ,mulocal )
        else
          call box07( rslt ,ss(3),ss(5),ss(6) ,rr(4) ,mulocal )
        endif
      elseif (s4r4.ne.R0P0) then
        call box07( rslt ,ss(4),ss(5),ss(6) ,rr(4) ,mulocal )
      else
        call box06( rslt ,ss(5),ss(6) ,rr(4) ,mulocal )
      endif
    endif
  else
!0 non-zero internal mass
    call box00( rslt ,ss ,as ,mulocal )
  endif
!exp(eps*gamma_EULER) -> GAMMA(1-2*eps)/GAMMA(1-eps)^2/GAMMA(1+eps)
  rslt(0) = rslt(0) + const*rslt(2)
!
  if (punit.gt.0) then
    if (nonzerothrs) write(punit,*) 'onshell:',trim(myprint(onshellthrs))
    write(punit,*) 'muscale:',trim(myprint(mulocal))
    write(punit,*) ' p1:',trim(myprint(p1))
    write(punit,*) ' p2:',trim(myprint(p2))
    write(punit,*) ' p3:',trim(myprint(p3))
    write(punit,*) ' p4:',trim(myprint(p4))
    write(punit,*) 'p12:',trim(myprint(p12))
    write(punit,*) 'p23:',trim(myprint(p23))
    write(punit,*) ' m1:',trim(myprint(m1))
    write(punit,*) ' m2:',trim(myprint(m2))
    write(punit,*) ' m3:',trim(myprint(m3))
    write(punit,*) ' m4:',trim(myprint(m4))
    write(punit,*) 'd0(2):',trim(myprint(rslt(2)))
    write(punit,*) 'd0(1):',trim(myprint(rslt(1)))
    write(punit,*) 'd0(0):',trim(myprint(rslt(0)))
  endif
!
  end subroutine
!
  subroutine d0cc( rslt ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4 )
!*******************************************************************
!*******************************************************************
  use avh_olo_mp_box
  use avh_olo_mp_boxc
  include 'cts_mpc.h'
   ,intent(out) :: rslt(0:2)
  include 'cts_mpc.h'
   ,intent(in)  :: p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4
  include 'cts_mpc.h'
   :: pp(6),mm(4)
  include 'cts_mpr.h'
   :: hh
!
  include 'cts_mpc.h'
   :: ss(6),rr(4)
  include 'cts_mpr.h'
   :: smax,ap(6),am(4),as(6),ar(4),s1r2,s2r2,s2r3,s3r4,s4r4
  include 'cts_mpr.h'
   :: mulocal,mulocal2,small,thrs
  integer :: icase,ii
  logical :: useboxc
  include 'cts_mpc.h'
   :: const
  character(25+99) ,parameter :: warning=&
                 'WARNING from OneLOop d0: '//warnonshell
  const=C1P0*TWOPI*TWOPI/48

  if (intro) call hello
!
  pp(1) = p1
  pp(2) = p2
  pp(3) = p3
  pp(4) = p4
  pp(5) = p12
  pp(6) = p23
  mm(1) = m1
  mm(2) = m2
  mm(3) = m3
  mm(4) = m4
!
  smax = R0P0
!
  mulocal = muscale
!
  do ii=1,6
    ap(ii) = mp_rl(pp(ii))
    if (mp_ig(pp(ii)).ne.R0P0) then
      if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop d0: ' &
        ,'momentum with non-zero imaginary part, putting it to zero.'
      pp(ii) = mp_cx( ap(ii) ,R0P0 )
    endif
    ap(ii) = abs(ap(ii))
    if (ap(ii).gt.smax) smax = ap(ii)
  enddo
!
  do ii=1,4
    am(ii) = mp_rl(mm(ii))
    hh = mp_ig(mm(ii))
    if (hh.gt.R0P0) then
      if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop d0: ' &
        ,'mass-squared has positive imaginary part, switching its sign.'
      mm(ii) = mp_cx( am(ii) ,-hh )
    endif
    am(ii) = abs(am(ii)) + abs(hh)
    if (am(ii).gt.smax) smax = am(ii)
  enddo
!
  small = mp_ml(ap,6)*mp_eps*100
  mulocal2 = mulocal*mulocal
!
  if (smax.eq.R0P0) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop d0: ' &
      ,'all input equal zero, returning 0'
    rslt = C0P0
    return
  endif
!
  if (mulocal2.gt.smax) smax = mulocal2
!
  if (nonzerothrs) then
    thrs = onshellthrs
    do ii=1,4
      if (ap(ii).lt.thrs) ap(ii) = R0P0
      if (am(ii).lt.thrs) am(ii) = R0P0
    enddo
  else
    thrs = onshellthrs*smax
    if (wunit.gt.0) then
    do ii=1,4
      if (R0P0.lt.ap(ii).and.ap(ii).lt.thrs) write(wunit,*) warning
      if (R0P0.lt.am(ii).and.am(ii).lt.thrs) write(wunit,*) warning
    enddo
    endif
  endif
!
  icase = 0
  do ii=1,4
    if (am(ii).gt.R0P0) icase = icase + base(ii)
  enddo
  ss(1)=pp(permtable(1,icase)) ;as(1)=ap(permtable(1,icase))
  ss(2)=pp(permtable(2,icase)) ;as(2)=ap(permtable(2,icase))
  ss(3)=pp(permtable(3,icase)) ;as(3)=ap(permtable(3,icase))
  ss(4)=pp(permtable(4,icase)) ;as(4)=ap(permtable(4,icase))
  ss(5)=pp(permtable(5,icase)) ;as(5)=ap(permtable(5,icase))
  ss(6)=pp(permtable(6,icase)) ;as(6)=ap(permtable(6,icase))
  rr(1)=mm(permtable(1,icase)) ;ar(1)=am(permtable(1,icase))
  rr(2)=mm(permtable(2,icase)) ;ar(2)=am(permtable(2,icase))
  rr(3)=mm(permtable(3,icase)) ;ar(3)=am(permtable(3,icase))
  rr(4)=mm(permtable(4,icase)) ;ar(4)=am(permtable(4,icase))
  icase = casetable(icase)
!
  s1r2 = abs(mp_rl(ss(1)-rr(2))) + abs(mp_ig(ss(1)-rr(2)))
  s2r2 = abs(mp_rl(ss(2)-rr(2))) + abs(mp_ig(ss(2)-rr(2)))
  s2r3 = abs(mp_rl(ss(2)-rr(3))) + abs(mp_ig(ss(2)-rr(3)))
  s3r4 = abs(mp_rl(ss(3)-rr(4))) + abs(mp_ig(ss(3)-rr(4)))
  s4r4 = abs(mp_rl(ss(4)-rr(4))) + abs(mp_ig(ss(4)-rr(4)))
  if (nonzerothrs) then
    if (s1r2.lt.thrs) s1r2 = R0P0
    if (s2r2.lt.thrs) s2r2 = R0P0
    if (s2r3.lt.thrs) s2r3 = R0P0
    if (s3r4.lt.thrs) s3r4 = R0P0
    if (s4r4.lt.thrs) s4r4 = R0P0
  elseif (wunit.gt.0) then
    if (R0P0.lt.s1r2.and.s1r2.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.s2r2.and.s2r2.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.s2r3.and.s2r3.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.s3r4.and.s3r4.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.s4r4.and.s4r4.lt.thrs) write(wunit,*) warning
  endif
!
  if     (icase.eq.4) then
!4 non-zero internal masses
    useboxc = (    (ar(1).ne.R0P0.and.mp_ig(rr(1)).ne.R0P0) &
               .or.(ar(2).ne.R0P0.and.mp_ig(rr(2)).ne.R0P0) &
               .or.(ar(3).ne.R0P0.and.mp_ig(rr(3)).ne.R0P0) &
               .or.(ar(4).ne.R0P0.and.mp_ig(rr(4)).ne.R0P0) &
               .or.(     mp_rl(ss(1)).ge.-small  &
                    .and.mp_rl(ss(2)).ge.-small  &
                    .and.mp_rl(ss(3)).ge.-small  &
                    .and.mp_rl(ss(4)).ge.-small) )
    if (useboxc) then
      call boxc( rslt ,ss,rr ,as )
    else
      call boxf4( rslt ,ss(1),ss(2),ss(3),ss(4),ss(5),ss(6) ,rr(1),rr(2),rr(3),rr(4) )
    endif
  elseif (icase.eq.3) then
!3 non-zero internal masses
    if (s1r2.ne.R0P0.or.s4r4.ne.R0P0) then
      useboxc = (    (ar(1).ne.R0P0.and.mp_ig(rr(1)).ne.R0P0) &
                 .or.(ar(2).ne.R0P0.and.mp_ig(rr(2)).ne.R0P0) &
                 .or.(ar(3).ne.R0P0.and.mp_ig(rr(3)).ne.R0P0) &
                 .or.(ar(4).ne.R0P0.and.mp_ig(rr(4)).ne.R0P0) &
                 .or.(     mp_rl(ss(1)).ge.-small  &
                      .and.mp_rl(ss(2)).ge.-small  &
                      .and.mp_rl(ss(3)).ge.-small  &
                      .and.mp_rl(ss(4)).ge.-small) )
      if (useboxc) then
        call boxc( rslt ,ss,rr ,as )
      else
        call boxf3( rslt, ss,rr )
      endif
    else
      call box16( rslt ,ss(2),ss(3),ss(5),ss(6) ,rr(2),rr(3),rr(4) ,mulocal )
    endif
  elseif (icase.eq.5) then
!2 non-zero internal masses, opposite case
    if     (s1r2.ne.R0P0.or.s4r4.ne.R0P0) then
      if     (s2r2.ne.R0P0.or.s3r4.ne.R0P0) then
        call boxf5( rslt ,ss(1),ss(2),ss(3),ss(4),ss(5),ss(6) ,rr(2),rr(4) )
      else
        call box15( rslt ,ss(1),ss(4),ss(5),ss(6) ,rr(2),rr(4) ,mulocal )
      endif
    elseif (s2r2.ne.R0P0.or.s3r4.ne.R0P0) then
      call box15( rslt ,ss(2),ss(3),ss(5),ss(6) ,rr(2),rr(4) ,mulocal )
    else
      call box14( rslt ,ss(5),ss(6) ,rr(2),rr(4) ,mulocal )
    endif
  elseif (icase.eq.2) then
!2 non-zero internal masses, adjacent case
    if     (as(1).ne.R0P0) then
      call boxf2( rslt ,ss(1),ss(2),ss(3),ss(4),ss(5),ss(6) ,rr(3),rr(4) )
    elseif (s2r3.ne.R0P0) then
      if     (s4r4.ne.R0P0) then
        call box13( rslt ,ss(2),ss(3),ss(4),ss(5),ss(6) ,rr(3),rr(4) ,mulocal )
      else
        call box12( rslt ,ss(3),ss(2),ss(6),ss(5) ,rr(4),rr(3) ,mulocal )
      endif
    elseif (s4r4.ne.R0P0) then
      call box12( rslt ,ss(3),ss(4),ss(5),ss(6) ,rr(3),rr(4) ,mulocal )
    else
      call box11( rslt ,ss(3),ss(5),ss(6) ,rr(3),rr(4) ,mulocal )
    endif
  elseif (icase.eq.1) then
!1 non-zero internal mass
    if     (as(1).ne.R0P0) then
      if      (as(2).ne.R0P0) then
        call boxf1( rslt ,ss(1),ss(2),ss(3),ss(4),ss(5),ss(6) ,rr(4) )
      else
        if     (s3r4.ne.R0P0) then
          call box10( rslt ,ss(1),ss(4),ss(3),ss(5),ss(6) ,rr(4) ,mulocal )
        else
          call box09( rslt ,ss(1),ss(4),ss(5),ss(6) ,rr(4) ,mulocal )
        endif
      endif
    elseif (as(2).ne.R0P0) then
      if      (s4r4.ne.R0P0) then
        call box10( rslt ,ss(2),ss(3),ss(4),ss(5),ss(6) ,rr(4) ,mulocal )
      else
        call box09( rslt ,ss(2),ss(3),ss(5),ss(6) ,rr(4) ,mulocal )
      endif
    else
      if     (s3r4.ne.R0P0) then
        if     (s4r4.ne.R0P0) then
          call box08( rslt ,ss(3),ss(4),ss(5),ss(6) ,rr(4) ,mulocal )
        else
          call box07( rslt ,ss(3),ss(5),ss(6) ,rr(4) ,mulocal )
        endif
      elseif (s4r4.ne.R0P0) then
        call box07( rslt ,ss(4),ss(5),ss(6) ,rr(4) ,mulocal )
      else
        call box06( rslt ,ss(5),ss(6) ,rr(4) ,mulocal )
      endif
    endif
  else
!0 non-zero internal mass
    call box00( rslt ,ss ,as ,mulocal )
  endif
!exp(eps*gamma_EULER) -> GAMMA(1-2*eps)/GAMMA(1-eps)^2/GAMMA(1+eps)
  rslt(0) = rslt(0) + const*rslt(2)
!
  if (punit.gt.0) then
    if (nonzerothrs) write(punit,*) 'onshell:',trim(myprint(onshellthrs))
    write(punit,*) 'muscale:',trim(myprint(mulocal))
    write(punit,*) ' p1:',trim(myprint(p1))
    write(punit,*) ' p2:',trim(myprint(p2))
    write(punit,*) ' p3:',trim(myprint(p3))
    write(punit,*) ' p4:',trim(myprint(p4))
    write(punit,*) 'p12:',trim(myprint(p12))
    write(punit,*) 'p23:',trim(myprint(p23))
    write(punit,*) ' m1:',trim(myprint(m1))
    write(punit,*) ' m2:',trim(myprint(m2))
    write(punit,*) ' m3:',trim(myprint(m3))
    write(punit,*) ' m4:',trim(myprint(m4))
    write(punit,*) 'd0(2):',trim(myprint(rslt(2)))
    write(punit,*) 'd0(1):',trim(myprint(rslt(1)))
    write(punit,*) 'd0(0):',trim(myprint(rslt(0)))
  endif
!
  end subroutine
!
  subroutine d0ccr( rslt ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4 ,rmu )
!*******************************************************************
!*******************************************************************
  use avh_olo_mp_box
  use avh_olo_mp_boxc
  include 'cts_mpc.h'
   ,intent(out) :: rslt(0:2)
  include 'cts_mpc.h'
   ,intent(in)  :: p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4
  include 'cts_mpr.h'
   ,intent(in)  :: rmu
  include 'cts_mpc.h'
   :: pp(6),mm(4)
  include 'cts_mpr.h'
   :: hh
!
  include 'cts_mpc.h'
   :: ss(6),rr(4)
  include 'cts_mpr.h'
   :: smax,ap(6),am(4),as(6),ar(4),s1r2,s2r2,s2r3,s3r4,s4r4
  include 'cts_mpr.h'
   :: mulocal,mulocal2,small,thrs
  integer :: icase,ii
  logical :: useboxc
  include 'cts_mpc.h'
   :: const
  character(25+99) ,parameter :: warning=&
                 'WARNING from OneLOop d0: '//warnonshell
  const=C1P0*TWOPI*TWOPI/48
  if (intro) call hello
!
  pp(1) = p1
  pp(2) = p2
  pp(3) = p3
  pp(4) = p4
  pp(5) = p12
  pp(6) = p23
  mm(1) = m1
  mm(2) = m2
  mm(3) = m3
  mm(4) = m4
!
  smax = R0P0
!
  mulocal = rmu
!
  do ii=1,6
    ap(ii) = mp_rl(pp(ii))
    if (mp_ig(pp(ii)).ne.R0P0) then
      if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop d0: ' &
        ,'momentum with non-zero imaginary part, putting it to zero.'
      pp(ii) = mp_cx( ap(ii) ,R0P0 )
    endif
    ap(ii) = abs(ap(ii))
    if (ap(ii).gt.smax) smax = ap(ii)
  enddo
!
  do ii=1,4
    am(ii) = mp_rl(mm(ii))
    hh = mp_ig(mm(ii))
    if (hh.gt.R0P0) then
      if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop d0: ' &
        ,'mass-squared has positive imaginary part, switching its sign.'
      mm(ii) = mp_cx( am(ii) ,-hh )
    endif
    am(ii) = abs(am(ii)) + abs(hh)
    if (am(ii).gt.smax) smax = am(ii)
  enddo
!
  small = mp_ml(ap,6)*mp_eps*100
  mulocal2 = mulocal*mulocal
!
  if (smax.eq.R0P0) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop d0: ' &
      ,'all input equal zero, returning 0'
    rslt = C0P0
    return
  endif
!
  if (mulocal2.gt.smax) smax = mulocal2
!
  if (nonzerothrs) then
    thrs = onshellthrs
    do ii=1,4
      if (ap(ii).lt.thrs) ap(ii) = R0P0
      if (am(ii).lt.thrs) am(ii) = R0P0
    enddo
  else
    thrs = onshellthrs*smax
    if (wunit.gt.0) then
    do ii=1,4
      if (R0P0.lt.ap(ii).and.ap(ii).lt.thrs) write(wunit,*) warning
      if (R0P0.lt.am(ii).and.am(ii).lt.thrs) write(wunit,*) warning
    enddo
    endif
  endif
!
  icase = 0
  do ii=1,4
    if (am(ii).gt.R0P0) icase = icase + base(ii)
  enddo
  ss(1)=pp(permtable(1,icase)) ;as(1)=ap(permtable(1,icase))
  ss(2)=pp(permtable(2,icase)) ;as(2)=ap(permtable(2,icase))
  ss(3)=pp(permtable(3,icase)) ;as(3)=ap(permtable(3,icase))
  ss(4)=pp(permtable(4,icase)) ;as(4)=ap(permtable(4,icase))
  ss(5)=pp(permtable(5,icase)) ;as(5)=ap(permtable(5,icase))
  ss(6)=pp(permtable(6,icase)) ;as(6)=ap(permtable(6,icase))
  rr(1)=mm(permtable(1,icase)) ;ar(1)=am(permtable(1,icase))
  rr(2)=mm(permtable(2,icase)) ;ar(2)=am(permtable(2,icase))
  rr(3)=mm(permtable(3,icase)) ;ar(3)=am(permtable(3,icase))
  rr(4)=mm(permtable(4,icase)) ;ar(4)=am(permtable(4,icase))
  icase = casetable(icase)
!
  s1r2 = abs(mp_rl(ss(1)-rr(2))) + abs(mp_ig(ss(1)-rr(2)))
  s2r2 = abs(mp_rl(ss(2)-rr(2))) + abs(mp_ig(ss(2)-rr(2)))
  s2r3 = abs(mp_rl(ss(2)-rr(3))) + abs(mp_ig(ss(2)-rr(3)))
  s3r4 = abs(mp_rl(ss(3)-rr(4))) + abs(mp_ig(ss(3)-rr(4)))
  s4r4 = abs(mp_rl(ss(4)-rr(4))) + abs(mp_ig(ss(4)-rr(4)))
  if (nonzerothrs) then
    if (s1r2.lt.thrs) s1r2 = R0P0
    if (s2r2.lt.thrs) s2r2 = R0P0
    if (s2r3.lt.thrs) s2r3 = R0P0
    if (s3r4.lt.thrs) s3r4 = R0P0
    if (s4r4.lt.thrs) s4r4 = R0P0
  elseif (wunit.gt.0) then
    if (R0P0.lt.s1r2.and.s1r2.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.s2r2.and.s2r2.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.s2r3.and.s2r3.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.s3r4.and.s3r4.lt.thrs) write(wunit,*) warning
    if (R0P0.lt.s4r4.and.s4r4.lt.thrs) write(wunit,*) warning
  endif
!
  if     (icase.eq.4) then
!4 non-zero internal masses
    useboxc = (    (ar(1).ne.R0P0.and.mp_ig(rr(1)).ne.R0P0) &
               .or.(ar(2).ne.R0P0.and.mp_ig(rr(2)).ne.R0P0) &
               .or.(ar(3).ne.R0P0.and.mp_ig(rr(3)).ne.R0P0) &
               .or.(ar(4).ne.R0P0.and.mp_ig(rr(4)).ne.R0P0) &
               .or.(     mp_rl(ss(1)).ge.-small  &
                    .and.mp_rl(ss(2)).ge.-small  &
                    .and.mp_rl(ss(3)).ge.-small  &
                    .and.mp_rl(ss(4)).ge.-small) )
    if (useboxc) then
      call boxc( rslt ,ss,rr ,as )
    else
      call boxf4( rslt ,ss(1),ss(2),ss(3),ss(4),ss(5),ss(6) ,rr(1),rr(2),rr(3),rr(4) )
    endif
  elseif (icase.eq.3) then
!3 non-zero internal masses
    if (s1r2.ne.R0P0.or.s4r4.ne.R0P0) then
      useboxc = (    (ar(1).ne.R0P0.and.mp_ig(rr(1)).ne.R0P0) &
                 .or.(ar(2).ne.R0P0.and.mp_ig(rr(2)).ne.R0P0) &
                 .or.(ar(3).ne.R0P0.and.mp_ig(rr(3)).ne.R0P0) &
                 .or.(ar(4).ne.R0P0.and.mp_ig(rr(4)).ne.R0P0) &
                 .or.(     mp_rl(ss(1)).ge.-small  &
                      .and.mp_rl(ss(2)).ge.-small  &
                      .and.mp_rl(ss(3)).ge.-small  &
                      .and.mp_rl(ss(4)).ge.-small) )
      if (useboxc) then
        call boxc( rslt ,ss,rr ,as )
      else
        call boxf3( rslt, ss,rr )
      endif
    else
      call box16( rslt ,ss(2),ss(3),ss(5),ss(6) ,rr(2),rr(3),rr(4) ,mulocal )
    endif
  elseif (icase.eq.5) then
!2 non-zero internal masses, opposite case
    if     (s1r2.ne.R0P0.or.s4r4.ne.R0P0) then
      if     (s2r2.ne.R0P0.or.s3r4.ne.R0P0) then
        call boxf5( rslt ,ss(1),ss(2),ss(3),ss(4),ss(5),ss(6) ,rr(2),rr(4) )
      else
        call box15( rslt ,ss(1),ss(4),ss(5),ss(6) ,rr(2),rr(4) ,mulocal )
      endif
    elseif (s2r2.ne.R0P0.or.s3r4.ne.R0P0) then
      call box15( rslt ,ss(2),ss(3),ss(5),ss(6) ,rr(2),rr(4) ,mulocal )
    else
      call box14( rslt ,ss(5),ss(6) ,rr(2),rr(4) ,mulocal )
    endif
  elseif (icase.eq.2) then
!2 non-zero internal masses, adjacent case
    if     (as(1).ne.R0P0) then
      call boxf2( rslt ,ss(1),ss(2),ss(3),ss(4),ss(5),ss(6) ,rr(3),rr(4) )
    elseif (s2r3.ne.R0P0) then
      if     (s4r4.ne.R0P0) then
        call box13( rslt ,ss(2),ss(3),ss(4),ss(5),ss(6) ,rr(3),rr(4) ,mulocal )
      else
        call box12( rslt ,ss(3),ss(2),ss(6),ss(5) ,rr(4),rr(3) ,mulocal )
      endif
    elseif (s4r4.ne.R0P0) then
      call box12( rslt ,ss(3),ss(4),ss(5),ss(6) ,rr(3),rr(4) ,mulocal )
    else
      call box11( rslt ,ss(3),ss(5),ss(6) ,rr(3),rr(4) ,mulocal )
    endif
  elseif (icase.eq.1) then
!1 non-zero internal mass
    if     (as(1).ne.R0P0) then
      if      (as(2).ne.R0P0) then
        call boxf1( rslt ,ss(1),ss(2),ss(3),ss(4),ss(5),ss(6) ,rr(4) )
      else
        if     (s3r4.ne.R0P0) then
          call box10( rslt ,ss(1),ss(4),ss(3),ss(5),ss(6) ,rr(4) ,mulocal )
        else
          call box09( rslt ,ss(1),ss(4),ss(5),ss(6) ,rr(4) ,mulocal )
        endif
      endif
    elseif (as(2).ne.R0P0) then
      if      (s4r4.ne.R0P0) then
        call box10( rslt ,ss(2),ss(3),ss(4),ss(5),ss(6) ,rr(4) ,mulocal )
      else
        call box09( rslt ,ss(2),ss(3),ss(5),ss(6) ,rr(4) ,mulocal )
      endif
    else
      if     (s3r4.ne.R0P0) then
        if     (s4r4.ne.R0P0) then
          call box08( rslt ,ss(3),ss(4),ss(5),ss(6) ,rr(4) ,mulocal )
        else
          call box07( rslt ,ss(3),ss(5),ss(6) ,rr(4) ,mulocal )
        endif
      elseif (s4r4.ne.R0P0) then
        call box07( rslt ,ss(4),ss(5),ss(6) ,rr(4) ,mulocal )
      else
        call box06( rslt ,ss(5),ss(6) ,rr(4) ,mulocal )
      endif
    endif
  else
!0 non-zero internal mass
    call box00( rslt ,ss ,as ,mulocal )
  endif
!exp(eps*gamma_EULER) -> GAMMA(1-2*eps)/GAMMA(1-eps)^2/GAMMA(1+eps)
  rslt(0) = rslt(0) + const*rslt(2)
!
  if (punit.gt.0) then
    if (nonzerothrs) write(punit,*) 'onshell:',trim(myprint(onshellthrs))
    write(punit,*) 'muscale:',trim(myprint(mulocal))
    write(punit,*) ' p1:',trim(myprint(p1))
    write(punit,*) ' p2:',trim(myprint(p2))
    write(punit,*) ' p3:',trim(myprint(p3))
    write(punit,*) ' p4:',trim(myprint(p4))
    write(punit,*) 'p12:',trim(myprint(p12))
    write(punit,*) 'p23:',trim(myprint(p23))
    write(punit,*) ' m1:',trim(myprint(m1))
    write(punit,*) ' m2:',trim(myprint(m2))
    write(punit,*) ' m3:',trim(myprint(m3))
    write(punit,*) ' m4:',trim(myprint(m4))
    write(punit,*) 'd0(2):',trim(myprint(rslt(2)))
    write(punit,*) 'd0(1):',trim(myprint(rslt(1)))
    write(punit,*) 'd0(0):',trim(myprint(rslt(0)))
  endif
!
  end subroutine

end module

      subroutine avh_olo_mp_mu_set(mu)
      use avh_olo_mp
      implicit none
      include 'cts_dpr.h'
       ,intent(in) :: mu
      call olo_mp_scale( mu )
      end subroutine

      subroutine avh_olo_mp_onshell(thrs)
      use avh_olo_mp
      implicit none
      include 'cts_dpr.h'
       ,intent(in) :: thrs
      call olo_mp_onshell( thrs )
      end subroutine

      subroutine avh_olo_mp_unit( unit_in )
      use avh_olo_mp
      implicit none
      integer ,intent(in) :: unit_in
      call olo_mp_unit( unit_in ,'all' )
      end subroutine

      subroutine avh_olo_mp_printall( unit_in )
      use avh_olo_mp
      implicit none
      integer ,intent(in) :: unit_in
      call olo_mp_unit( unit_in ,'printall' )
      end subroutine

      subroutine avh_olo_mp_a0c( rslt ,mm )
      include 'cts_mprec.h'
      use avh_olo_mp
      implicit none
      include 'cts_mpc.h'
       ,intent(out) :: rslt(0:2)
      include 'cts_mpc.h'
       ,intent(in)  :: mm
      call olo_mp_a0( rslt ,mm )
      end subroutine

      subroutine avh_olo_mp_b0c( rslt ,pp,m1,m2 )
      include 'cts_mprec.h'
      use avh_olo_mp
      implicit none
      include 'cts_mpc.h'
       ,intent(out) :: rslt(0:2)
      include 'cts_mpc.h'
       ,intent(in)  :: pp,m1,m2
      call olo_mp_b0( rslt ,pp,m1,m2 )
      end subroutine

      subroutine avh_olo_mp_b11c( b11,b00,b1,b0 ,pp,m1,m2 )
      include 'cts_mprec.h'
      use avh_olo_mp
      implicit none
      include 'cts_mpc.h'
       ,intent(out) :: b11(0:2),b00(0:2),b1(0:2),b0(0:2)
      include 'cts_mpc.h'
       ,intent(in)  :: pp,m1,m2
      call olo_mp_b11( b11,b00,b1,b0 ,pp,m1,m2 )
      end subroutine

      subroutine avh_olo_mp_c0c( rslt ,p1,p2,p3 ,m1,m2,m3 )
      include 'cts_mprec.h'
      use avh_olo_mp
      implicit none
      include 'cts_mpc.h'
       ,intent(out) :: rslt(0:2)
      include 'cts_mpc.h'
       ,intent(in)  :: p1,p2,p3 ,m1,m2,m3
      call olo_mp_c0( rslt ,p1,p2,p3 ,m1,m2,m3 )
      end subroutine

      subroutine avh_olo_mp_d0c( rslt ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4 )
      include 'cts_mprec.h'
      use avh_olo_mp
      implicit none
      include 'cts_mpc.h'
       ,intent(out) :: rslt(0:2)
      include 'cts_mpc.h'
       ,intent(in)  :: p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4
      call olo_mp_d0( rslt ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4 )
      end subroutine

      subroutine avh_olo_mp_a0m( rslt ,mm )
      include 'cts_mprec.h'
      use avh_olo_mp
      implicit none
      include 'cts_mpc.h'
       ,intent(out) :: rslt(0:2)
      include 'cts_mpr.h'
       ,intent(in)  :: mm
      call olo_mp_a0( rslt ,mm )
      end subroutine

      subroutine avh_olo_mp_b0m( rslt ,pp,m1,m2 )
      include 'cts_mprec.h'
      use avh_olo_mp
      implicit none
      include 'cts_mpc.h'
       ,intent(out) :: rslt(0:2)
      include 'cts_mpr.h'
       ,intent(in)  :: pp,m1,m2
      call olo_mp_b0( rslt ,pp,m1,m2 )
      end subroutine

      subroutine avh_olo_mp_b11m( b11,b00,b1,b0 ,pp,m1,m2 )
      include 'cts_mprec.h'
      use avh_olo_mp
      implicit none
      include 'cts_mpc.h'
       ,intent(out) :: b11(0:2),b00(0:2),b1(0:2),b0(0:2)
      include 'cts_mpr.h'
       ,intent(in)  :: pp,m1,m2
      call olo_mp_b11( b11,b00,b1,b0 ,pp,m1,m2 )
      end subroutine

      subroutine avh_olo_mp_c0m( rslt ,p1,p2,p3 ,m1,m2,m3 )
      include 'cts_mprec.h'
      use avh_olo_mp
      implicit none
      include 'cts_mpc.h'
       ,intent(out) :: rslt(0:2)
      include 'cts_mpr.h'
       ,intent(in)  :: p1,p2,p3 ,m1,m2,m3
      call olo_mp_c0( rslt ,p1,p2,p3 ,m1,m2,m3 )
      end subroutine

      subroutine avh_olo_mp_d0m( rslt ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4 )
      include 'cts_mprec.h'
      use avh_olo_mp
      implicit none
      include 'cts_mpc.h'
       ,intent(out) :: rslt(0:2)
      include 'cts_mpr.h'
       ,intent(in)  :: p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4
      call olo_mp_d0( rslt ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4 )
      end subroutine
