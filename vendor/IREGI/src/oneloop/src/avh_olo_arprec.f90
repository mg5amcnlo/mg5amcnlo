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
  private :: qrFROMqc,qrFROMi,qcFROMi
  private :: prduct_qr_i,prduct_i_qr
  private :: prduct_qc_i,prduct_i_qc
  private :: ratio_qr_i,ratio_i_qr
  private :: ratio_qc_i,ratio_i_qc
  private :: plus_qr_i,plus_i_qr
  private :: plus_qc_i,plus_i_qc
  private :: minus_qr_i,minus_i_qr
  private :: minus_qc_i,minus_i_qc

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

  interface assignment (=)
    module procedure qrFROMqc,qcFROMi
  end interface
  interface operator (*)
    module procedure prduct_qc_i,prduct_i_qc
  end interface
  interface operator (/)
    module procedure ratio_qc_i!,ratio_i_qc
  end interface
  interface operator (+)
    module procedure plus_qc_i,plus_i_qc
  end interface
  interface operator (-)
    module procedure minus_qc_i,minus_i_qc
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
        i1 = ii                                   
        exit                                      
      endif                                       
    enddo                                         
    if (ndecim(i0).eq.ndec) i0=i1                 
    prcpar = i1                                   
    newprc = (ndecim(prcpar).ne.ndec)             
    if (newprc) then                              
      call shift1( ndecim ,prcpar )               
      call shift1( epsilo ,prcpar )               
      call shift1( neglig ,prcpar )               
      call set_epsn         
    else                                          
      EPSN = epsilo(prcpar)                       
    endif                                         
  else                                            
    allocate(ndecim(1:1),epsilo(1:1),neglig(1:1)) 
    call set_epsn            
    newprc = .true.                               
  endif                                           
  if (prcpar.eq.ubound(ndecim,1)) then
    RZRO=mpreal(0)
    RONE=mpreal(1)
    IMAG=cmplx(0d0,1d0,kind=kind(1d0))
    CZRO=RZRO
    CONE=RONE
    ONEPI=mpreal(4)*atan(RONE)
    TWOPI=mpreal(2)*ONEPI
    PISQo24=CONE*ONEPI*ONEPI/mpreal(24)
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
  rslt = mpreal(xx)
  end function

  function areal(zz) result(rslt)
!***********************************************************************
! Get real part of a complex
!***********************************************************************
  type(mp_complex) &
    ,intent(in) :: zz
  type(mp_real) &
    :: rslt
  rslt = mpreal(zz)
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
  rslt = xx + mpreal(yy)*IMAG
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
  rslt = mpreal(xx) + yy*IMAG
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
  xx = mpreal(zz)
  xx = abs(xx)
  yy = aimag(zz)
  rslt = xx + yy*IMAG
  end function
 
  
  subroutine qrFROMqc(xx,zz)
  intent(out) :: xx
  intent(in ) :: zz
  type(mp_real) &
    ::xx
  type(mp_complex) &
    :: zz
  xx = mpreal(zz)
  end subroutine

  subroutine qrFROMi(xx,ii)
  intent(out) :: xx
  intent(in ) :: ii
  type(mp_real) &
    ::xx
  integer :: ii 
  xx = mpreal(ii)
  end subroutine

  include 'avh_olo_intrf.h90'  
 
end module
