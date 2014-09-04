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
  use avh_olo_kinds

  implicit none
  public
  private :: IMAG,acmplx_r,acmplx_rr,acmplx_ir,acmplx_ri,acmplx_c

  integer ,save :: prcpar=0
  integer ,save :: ndecim(1)
  real(kindr2) &
          ,save :: epsilo(1),neglig(1)

  real(kindr2) &
    ,save :: RZRO ,RONE ,EPSN ,EPSN2 ,TWOPI ,ONEPI
  complex(kindr2) &
    ,save :: IEPS ,CZRO ,CONE ,IMAG ,PISQo24 ,IPI

  protected :: prcpar,ndecim,epsilo,neglig      !]PROTECTED
  protected :: RZRO,RONE,EPSN,EPSN2,TWOPI,ONEPI !]PROTECTED
  protected :: IEPS,CZRO,CONE,PISQo24,IPI       !]PROTECTED
                                                !]PROTECTED
  interface acmplx
    module procedure acmplx_r,acmplx_rr,acmplx_ir,acmplx_ri,acmplx_c
  end interface

contains


  subroutine set_precision( newprc )
!***********************************************************************
!***********************************************************************
  use avh_olo_units
  logical ,intent(out) :: newprc
  integer :: ndec                                  
  if (prcpar.eq.1) then                    
    newprc = .false.                             
    return                                       
  endif
  prcpar = 1                                   
  call set_epsn
  newprc = .true.                              
  RZRO=0
  RONE=1
  IMAG=cmplx(0,1,kind=kind(IMAG))
  CZRO=RZRO
  CONE=RONE
  ONEPI=4*atan(RONE)
  TWOPI=2*ONEPI
  PISQo24=CONE*ONEPI*ONEPI/24
  IPI=IMAG*ONEPI
  EPSN2= EPSN*EPSN
  IEPS= EPSN2*IMAG
!
  contains
!
  subroutine set_epsn
  EPSN = epsilon(EPSN)                         
  ndec = -log10(EPSN)                            
  ndecim(prcpar) = ndec                          
  epsilo(prcpar) = EPSN                        
  neglig(prcpar) = EPSN*10**(ndec/7)       
  end subroutine
!
  end subroutine


  function adble(xx) result(rslt)
!***********************************************************************
! Turn real(kindr2) into kind(1d0)
!***********************************************************************
  real(kindr2) ,intent(in) :: xx
  real(kind(1d0)) :: rslt
  rslt = real(xx,kind=kind(rslt))
  end function

  function convert(xx) result(rslt)
!***********************************************************************
! Turn kind(1d0) into real(kindr2)
!***********************************************************************
  real(kind(1d0)) ,intent(in) :: xx
  real(kindr2) :: rslt
  rslt = real(xx,kind=kind(rslt))
  end function

  function areal(zz) result(rslt)
!***********************************************************************
! Get real part of a complex
!***********************************************************************
  complex(kindr2) &
    ,intent(in) :: zz
  real(kindr2) &
    :: rslt
  rslt = zz
  end function

  function acmplx_r(xx) result(rslt)
!***********************************************************************
! Turn a real into a complex
!***********************************************************************
  real(kindr2) &
    ,intent(in) :: xx
  complex(kindr2) &
    :: rslt
  rslt = xx
  end function
  
  function acmplx_rr(xx,yy) result(rslt)
!***********************************************************************
! Turn two reals into one complex
!***********************************************************************
  real(kindr2) &
    ,intent(in) :: xx,yy
  complex(kindr2) &
    :: rslt
  rslt = cmplx(xx,yy,kind=kind(rslt))
  end function
  
  function acmplx_ri(xx,yy) result(rslt)
!***********************************************************************
! Turn a real and an integer into one complex
!***********************************************************************
  real(kindr2) &
           ,intent(in) :: xx
  integer  ,intent(in) :: yy
  complex(kindr2) &
    :: rslt
  rslt = cmplx(xx,yy,kind=kind(rslt))
  end function
  
  function acmplx_ir(xx,yy) result(rslt)
!***********************************************************************
! Turn an integer and a real into one complex
!***********************************************************************
  integer ,intent(in) :: xx
  real(kindr2) &
          ,intent(in) :: yy
  complex(kindr2) &
    :: rslt
  rslt = cmplx(xx,yy,kind=kind(rslt))
  end function
  
  function acmplx_c(zz) result(rslt)
!***********************************************************************
! Replaces the real part of zz by its absolute value
!***********************************************************************
  complex(kindr2) &
    ,intent(in) :: zz
  complex(kindr2) &
    :: rslt
  real(kindr2) &
    :: xx,yy
  xx = zz
  xx = abs(xx)
  yy = aimag(zz)
  rslt = cmplx(xx,yy,kind=kind(rslt))
  end function
  
end module
