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
  use avh_olo_units
  use avh_olo_print
  use avh_olo_prec
!
  implicit none
  private
  public :: olo_unit ,olo_scale ,olo_onshell ,olo_setting
  public :: olo_precision
  public :: olo_a0 ,olo_b0 ,olo_b11 ,olo_c0 ,olo_d0
  public :: olo_an ,olo_bn
  public :: olo
  public :: olo_get_scale ,olo_get_onshell ,olo_get_precision
!{cppINTERFACE=yes
  public :: a0_r,a0rr,a0_c,a0cr
  public :: an_r,anrr,an_c,ancr
  public :: b0rr,b0rrr,b0rc,b0rcr,b0cc,b0ccr
  public :: b11rr,b11rrr,b11rc,b11rcr,b11cc,b11ccr
  public :: bnrr,bnrrr,bnrc,bnrcr,bncc,bnccr
  public :: c0rr,c0rrr,c0rc,c0rcr,c0cc,c0ccr
  public :: d0rr,d0rrr,d0rc,d0rcr,d0cc,d0ccr
!}cppINTERFACE=yes
!
  integer ,public ,parameter :: olo_kind=kindr2    !|RCTYPE=intrinsic
!# integer ,public ,parameter :: olo_kind=kind(1d0) !|RCTYPE=ddtype
!# integer ,public ,parameter :: olo_kind=kind(1d0) !|RCTYPE=qdtype
!# integer ,public ,parameter :: olo_kind=kind(1d0) !|RCTYPE=mptype
!
  include 'avh_olo_real.h90'
         ,save :: onshellthrs
  logical,save :: nonzerothrs = .false.
!
  include 'avh_olo_real.h90'
         ,save :: muscale
!
  character(99) ,parameter :: warnonshell=&
       'it seems you forgot to put some input explicitly on shell. ' &
     //'You may  call olo_onshell  to cure this.'
!
  logical ,save :: initz=.true.
!
  interface olo_a0
    module procedure a0_r,a0rr,a0_c,a0cr
  end interface 
  interface olo_an
    module procedure an_r,anrr,an_c,ancr
  end interface 
  interface olo_b0
    module procedure b0rr,b0rrr,b0rc,b0rcr,b0cc,b0ccr
  end interface 
  interface olo_b11
    module procedure b11rr,b11rrr,b11rc,b11rcr,b11cc,b11ccr
  end interface 
  interface olo_bn
    module procedure bnrr,bnrrr,bnrc,bnrcr,bncc,bnccr
  end interface 
  interface olo_c0
    module procedure c0rr,c0rrr,c0rc,c0rcr,c0cc,c0ccr
  end interface 
  interface olo_d0
    module procedure d0rr,d0rrr,d0rc,d0rcr,d0cc,d0ccr
  end interface 
!
  interface olo
    module procedure a0_r,a0rr,a0_c,a0cr
    module procedure an_r,anrr,an_c,ancr
    module procedure b0rr,b0rrr,b0rc,b0rcr,b0cc,b0ccr
    module procedure b11rr,b11rrr,b11rc,b11rcr,b11cc,b11ccr
    module procedure bnrr,bnrrr,bnrc,bnrcr,bncc,bnccr
    module procedure c0rr,c0rrr,c0rc,c0rcr,c0cc,c0ccr
    module procedure d0rr,d0rrr,d0rc,d0rcr,d0cc,d0ccr
  end interface 

contains

 
  subroutine init( ndec )
!*******************************************************************
!*******************************************************************
  use avh_olo_version
  integer,optional,intent(in) :: ndec
!
  call olo_version
!
  initz = .false.
!
  if (present(ndec)) then
    call olo_precision( ndec )
  else
    call olo_precision( 15 )
  endif
!
  onshellthrs = 0
  muscale = 1
  if (.not.nonzerothrs) onshellthrs = neglig(prcpar)
!
  end subroutine
 
 
  recursive subroutine olo_precision( ndec )
!*******************************************************************
!*******************************************************************
  use avh_olo_olog  ,only: update_olog
  use avh_olo_dilog ,only: update_dilog
  use avh_olo_bnlog ,only: update_bnlog
  integer ,intent(in) :: ndec
  logical :: newprc
  if (initz) then
    call init( ndec )
  else
    call set_precision( newprc )       !|RCTYPE=intrinsic
!#   call set_precision( newprc )       !|RCTYPE=ddtype
!#   call set_precision( newprc )       !|RCTYPE=qdtype
!#   call set_precision( ndec ,newprc ) !|RCTYPE=mptype
    if (newprc) then
      call update_olog
      call update_dilog
      call update_bnlog
    endif
    if (.not.nonzerothrs) onshellthrs = neglig(prcpar)
  endif
  end subroutine

 
  subroutine olo_unit( val ,message )
!*******************************************************************
!*******************************************************************
  integer     ,intent(in) :: val
  character(*),intent(in),optional :: message
  if (initz) call init
  if (present(message)) then ;call set_unit( message ,val )
  else                       ;call set_unit( 'all'   ,val )
  endif
  end subroutine
 
 
  subroutine olo_scale( val )
!*******************************************************************
!*******************************************************************
  real(kind(1d0)) ,intent(in) :: val
  if (initz) call init
  muscale = convert(val)
  end subroutine
 
 
  subroutine olo_onshell( thrs )
!*******************************************************************
!*******************************************************************
  real(kind(1d0)) ,intent(in) :: thrs
  if (initz) call init
  nonzerothrs = .true.
  onshellthrs = convert(thrs)
  end subroutine


  function olo_get_precision() result(rslt)
!*******************************************************************
!*******************************************************************
  use avh_olo_prec ,only: ndecim,prcpar
  integer :: rslt
  if (initz) call init
  rslt = ndecim(prcpar)
  end function

  function olo_get_scale() result(rslt)
!*******************************************************************
!*******************************************************************
  real(kind(1d0)) :: rslt
  if (initz) call init
  rslt = adble(muscale)
  end function

  function olo_get_onshell() result(rslt)
!*******************************************************************
!*******************************************************************
  real(kind(1d0)) :: rslt
  if (initz) call init
  rslt = adble(onshellthrs)
  end function


  subroutine olo_setting( iunit )
!*******************************************************************
!*******************************************************************
  integer,optional,intent(in) :: iunit
  integer :: nunit
  if (initz) call init
  nunit = munit
  if (present(iunit)) nunit = iunit
  if (nunit.le.0) return
!
  write(nunit,*) 'MESSAGE from OneLOop: real kind parameter =',trim(myprint(kindr2)) !|RCTYPE=intrinsic
!# write(nunit,*) 'MESSAGE from OneLOop: operating at double double precision'        !|RCTYPE=ddtype
!# write(nunit,*) 'MESSAGE from OneLOop: operating at quad double precision'          !|RCTYPE=qdtype
!# write(nunit,*) 'MESSAGE from OneLOop: operating at arbitrary precision'            !|RCTYPE=mptype
  write(nunit,*) 'MESSAGE from OneLOop: number of decimals  =',trim(myprint(ndecim(prcpar)))
!
  if (nonzerothrs) then
    write(nunit,*) 'MESSAGE from OneLOop: on-shell threshold =',trim(myprint(onshellthrs,12))
  else
    write(nunit,*) 'MESSAGE from OneLOop: on-shell threshold is not set'
  endif
!
  write(nunit,*) 'MESSAGE from OneLOop: default scale (mu, not mu^2) =',trim(myprint(muscale,12))
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
! Check the comments in  subroutine olo_onshell  to find out how
! this routine decides when to return IR-divergent cases.
!*******************************************************************

  subroutine a0_c( rslt ,mm )
  include 'avh_olo_a0.h90' !?masses=complex !?mulocal=muscale
  end subroutine

  subroutine a0cr( rslt ,mm ,rmu )
  include 'avh_olo_a0.h90' !?masses=complex !?mulocal=rmu
  end subroutine

  subroutine a0_r( rslt ,mm  )
  include 'avh_olo_a0.h90' !?masses=real !?mulocal=muscale
  end subroutine

  subroutine a0rr( rslt ,mm ,rmu )
  include 'avh_olo_a0.h90' !?masses=real !?mulocal=rmu
  end subroutine


  subroutine an_c( rslt ,rank ,mm )
  include 'avh_olo_an.h90' !?masses=complex !?mulocal=muscale
  end subroutine

  subroutine ancr( rslt ,rank ,mm ,rmu )
  include 'avh_olo_an.h90' !?masses=complex !?mulocal=rmu
  end subroutine

  subroutine an_r( rslt ,rank ,mm  )
  include 'avh_olo_an.h90' !?masses=real !?mulocal=muscale
  end subroutine

  subroutine anrr( rslt ,rank ,mm ,rmu )
  include 'avh_olo_an.h90' !?masses=real !?mulocal=rmu
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
! Check the comments in  subroutine olo_onshell  to find out how
! this routine decides when to return IR-divergent cases.
!*******************************************************************

  subroutine b0cc( rslt ,pp,m1,m2 )
  include 'avh_olo_b0.h90' !?momenta=complex !?masses=complex !?mulocal=muscale
  end subroutine

  subroutine b0ccr( rslt ,pp,m1,m2 ,rmu )
  include 'avh_olo_b0.h90' !?momenta=complex !?masses=complex !?mulocal=rmu
  end subroutine

  subroutine b0rc( rslt ,pp ,m1,m2 )
  include 'avh_olo_b0.h90' !?momenta=real !?masses=complex !?mulocal=muscale
  end subroutine

  subroutine b0rcr( rslt ,pp,m1,m2 ,rmu )
  include 'avh_olo_b0.h90' !?momenta=real !?masses=complex !?mulocal=rmu
  end subroutine

  subroutine b0rr( rslt ,pp ,m1,m2 )
  include 'avh_olo_b0.h90' !?momenta=real !?masses=real !?mulocal=muscale
  end subroutine

  subroutine b0rrr( rslt ,pp ,m1,m2 ,rmu )
  include 'avh_olo_b0.h90' !?momenta=real !?masses=real !?mulocal=rmu
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
! Check the comments in  subroutine olo_onshell  to find out how
! this routine decides when to return IR-divergent cases.
!*******************************************************************

  subroutine b11cc( b11,b00,b1,b0 ,pp,m1,m2 )
  include 'avh_olo_b11.h90' !?momenta=complex !?masses=complex !?mulocal=muscale
  end subroutine

  subroutine b11ccr( b11,b00,b1,b0 ,pp,m1,m2 ,rmu )
  include 'avh_olo_b11.h90' !?momenta=complex !?masses=complex !?mulocal=rmu
  end subroutine

  subroutine b11rc( b11,b00,b1,b0 ,pp,m1,m2 )
  include 'avh_olo_b11.h90' !?momenta=real !?masses=complex !?mulocal=muscale
  end subroutine

  subroutine b11rcr( b11,b00,b1,b0 ,pp,m1,m2 ,rmu )
  include 'avh_olo_b11.h90' !?momenta=real !?masses=complex !?mulocal=rmu
  end subroutine

  subroutine b11rr( b11,b00,b1,b0 ,pp,m1,m2 )
  include 'avh_olo_b11.h90' !?momenta=real !?masses=real !?mulocal=muscale
  end subroutine

  subroutine b11rrr( b11,b00,b1,b0 ,pp,m1,m2 ,rmu )
  include 'avh_olo_b11.h90' !?momenta=real !?masses=real !?mulocal=rmu
  end subroutine


  subroutine bncc( rslt ,rank ,pp,m1,m2 )
  include 'avh_olo_bn.h90' !?momenta=complex !?masses=complex !?mulocal=muscale
  end subroutine

  subroutine bnccr( rslt ,rank ,pp,m1,m2 ,rmu )
  include 'avh_olo_bn.h90' !?momenta=complex !?masses=complex !?mulocal=rmu
  end subroutine

  subroutine bnrc( rslt ,rank ,pp,m1,m2 )
  include 'avh_olo_bn.h90' !?momenta=real !?masses=complex !?mulocal=muscale
  end subroutine

  subroutine bnrcr( rslt ,rank ,pp,m1,m2 ,rmu )
  include 'avh_olo_bn.h90' !?momenta=real !?masses=complex !?mulocal=rmu
  end subroutine

  subroutine bnrr( rslt ,rank ,pp,m1,m2 )
  include 'avh_olo_bn.h90' !?momenta=real !?masses=real !?mulocal=muscale
  end subroutine

  subroutine bnrrr( rslt ,rank ,pp,m1,m2 ,rmu )
  include 'avh_olo_bn.h90' !?momenta=real !?masses=real !?mulocal=rmu
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
! Check the comments in  subroutine olo_onshell  to find out how
! this routine decides when to return IR-divergent cases.
!*******************************************************************

  subroutine c0cc( rslt ,p1,p2,p3 ,m1,m2,m3 )
  include 'avh_olo_c0.h90' !?momenta=complex !?masses=complex !?mulocal=muscale
  end subroutine

  subroutine c0ccr( rslt ,p1,p2,p3 ,m1,m2,m3 ,rmu )
  include 'avh_olo_c0.h90' !?momenta=complex !?masses=complex !?mulocal=rmu
  end subroutine

  subroutine c0rc( rslt ,p1,p2,p3 ,m1,m2,m3 )
  include 'avh_olo_c0.h90' !?momenta=real !?masses=complex !?mulocal=muscale
  end subroutine

  subroutine c0rcr( rslt ,p1,p2,p3 ,m1,m2,m3 ,rmu )
  include 'avh_olo_c0.h90' !?momenta=real !?masses=complex !?mulocal=rmu
  end subroutine

  subroutine c0rr( rslt ,p1,p2,p3 ,m1,m2,m3 )
  include 'avh_olo_c0.h90' !?momenta=real !?masses=real !?mulocal=muscale
  end subroutine

  subroutine c0rrr( rslt ,p1,p2,p3 ,m1,m2,m3 ,rmu )
  include 'avh_olo_c0.h90' !?momenta=real !?masses=real !?mulocal=rmu
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
! Check the comments in  avh_olo_onshell  to find out how this
! routines decides when to return IR-divergent cases.
!*******************************************************************

  subroutine d0cc( rslt ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4 )
  include 'avh_olo_d0.h90' !?momenta=complex !?masses=complex !?mulocal=muscale
  end subroutine

  subroutine d0ccr( rslt ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4 ,rmu )
  include 'avh_olo_d0.h90' !?momenta=complex !?masses=complex !?mulocal=rmu
  end subroutine

  subroutine d0rc( rslt ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4 )
  include 'avh_olo_d0.h90' !?momenta=real !?masses=complex !?mulocal=muscale
  end subroutine

  subroutine d0rcr( rslt ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4 ,rmu )
  include 'avh_olo_d0.h90' !?momenta=real !?masses=complex !?mulocal=rmu
  end subroutine

  subroutine d0rr( rslt ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4 )
  include 'avh_olo_d0.h90' !?momenta=real !?masses=real !?mulocal=muscale
  end subroutine

  subroutine d0rrr( rslt ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4 ,rmu )
  include 'avh_olo_d0.h90' !?momenta=real !?masses=real !?mulocal=rmu
  end subroutine

end module
