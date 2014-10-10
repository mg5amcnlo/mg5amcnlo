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


module avh_olo_prec
  use mpmodule

  implicit none
  public
  private :: IMAG,acmplx_r,acmplx_rr,acmplx_ir,acmplx_ri,acmplx_c

  integer             ,save :: prcpar=1
  integer ,allocatable,save :: ndecim(:)
  type(mp_real) &
          ,allocatable,save :: epsilo(:),neglig(:)

  type(mp_real) &
    ,save :: RZRO ,RONE ,EPSN ,EPSN2 ,TWOPI ,ONEPI
  type(mp_complex) &
    ,save :: IEPS ,CZRO ,CONE ,IMAG ,PISQo24 ,IPI

  protected :: prcpar,ndecim,epsilo,neglig      !]PROTECTED
  protected :: RZRO,RONE,EPSN,EPSN2,TWOPI,ONEPI !]PROTECTED
  protected :: IEPS,CZRO,CONE,PISQo24,IPI       !]PROTECTED

  interface acmplx
    module procedure acmplx_r,acmplx_rr,acmplx_ir,acmplx_ri,acmplx_c
  end interface

contains

  subroutine set_precision( ndec ,newprc )
!***********************************************************************
!***********************************************************************
  use avh_olo_units
  use avh_olo_arrays
  logical ,intent(out) :: newprc
  integer ,intent(in) :: ndec                     
  integer :: i0,i1,ii                             
  if (allocated(ndecim)) then                     
    i0 = 0                                        
    i1 = 1+ubound(ndecim,1)                       
    do ;if (i0+1.eq.i1) exit                      
      ii = (i0+i1)/2                              
      if     (ndecim(ii).gt.ndec) then            
        i1 = ii                                   
      elseif (ndecim(ii).lt.ndec) then            
        i0 = ii                                   
      else                                        
        exit                                      
      endif                                       
    enddo                                         
    newprc = (ndecim(ii).ne.ndec)
    if (newprc) then
      prcpar = i0+1                              
      call shift1( ndecim ,prcpar )               
      call shift1( epsilo ,prcpar )               
      call shift1( neglig ,prcpar )               
      call set_epsn          
    else           
      prcpar = ii
      EPSN = epsilo(prcpar)                       
    endif                                         
  else                                            
    allocate(ndecim(1:1),epsilo(1:1),neglig(1:1)) 
    call set_epsn            
    newprc = .true.                               
  endif                                           
  if (prcpar.eq.ubound(ndecim,1)) then
    RZRO=0
    RONE=1
    IMAG=cmplx(0,1,kind=kind(1d0))
    CZRO=RZRO
    CONE=RONE
    ONEPI=4*atan(RONE)
    TWOPI=2*ONEPI
    PISQo24=CONE*ONEPI*ONEPI/24
    IPI=IMAG*ONEPI
  endif
  EPSN2 = EPSN !EPSN*EPSN
  IEPS  = EPSN2*IMAG
!
  contains
!
  subroutine set_epsn
  type(mp_real) &
    :: ten
  ten = 10                                       
  EPSN = ten**(-ndec)                            
  ndecim(prcpar) = ndec                         
  epsilo(prcpar) = EPSN                         
  neglig(prcpar) = EPSN*ten**(ndec/7)            
  end subroutine
!
  end subroutine


  function adble(xx) result(rslt)
!***********************************************************************
! Turn mp_real into kind(1d0)
!***********************************************************************
  type(mp_real) ,intent(in) :: xx
  real(kind(1d0)) :: rslt
  rslt = xx
  end function

  function convert(xx) result(rslt)
!***********************************************************************
! Turn kind(1d0) into mp_real
!***********************************************************************
  real(kind(1d0)) ,intent(in) :: xx
  type(mp_real) :: rslt
  rslt = xx
  end function

  function areal(zz) result(rslt)
!***********************************************************************
! Get real part of a complex
!***********************************************************************
  type(mp_complex) &
    ,intent(in) :: zz
  type(mp_real) &
    :: rslt
  rslt = zz
  end function

  function acmplx_r(xx) result(rslt)
!***********************************************************************
! Turn a real into a complex
!***********************************************************************
  type(mp_real) &
    ,intent(in) :: xx
  type(mp_complex) &
    :: rslt
  rslt = xx
  end function
  
  function acmplx_rr(xx,yy) result(rslt)
!***********************************************************************
! Turn two reals into one complex
!***********************************************************************
  type(mp_real) &
    ,intent(in) :: xx,yy
  type(mp_complex) &
    :: rslt
  rslt = xx + yy*IMAG
  end function
  
  function acmplx_ri(xx,yy) result(rslt)
!***********************************************************************
! Turn a real and an integer into one complex
!***********************************************************************
  type(mp_real) &
          ,intent(in) :: xx
  integer ,intent(in) :: yy
  type(mp_complex) &
    :: rslt
  rslt = xx + yy*IMAG
  end function
  
  function acmplx_ir(xx,yy) result(rslt)
!***********************************************************************
! Turn an integer and a real into one complex
!***********************************************************************
  integer ,intent(in) :: xx
  type(mp_real) &
          ,intent(in) :: yy
  type(mp_complex) &
    :: rslt
  rslt = xx + yy*IMAG
  end function
  
  function acmplx_c(zz) result(rslt)
!***********************************************************************
! Replaces the real part of zz by its absolute value
!***********************************************************************
  type(mp_complex) &
    ,intent(in) :: zz
  type(mp_complex) &
    :: rslt
  type(mp_real) &
    :: xx,yy
  xx = zz
  xx = abs(xx)
  yy = aimag(zz)
  rslt = xx + yy*IMAG
  end function
  
end module
