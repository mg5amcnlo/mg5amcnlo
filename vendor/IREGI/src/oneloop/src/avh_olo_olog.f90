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


module avh_olo_olog
!***********************************************************************
! Provides the functions
!   olog(x,n) = log(x) + n*pi*imag  
!   olog2(x,n) = olog(x,n)/(x-1)
! In the vicinity of x=1,n=0, the logarithm of complex argument is
! evaluated with a series expansion.
!***********************************************************************
  use avh_olo_units
  use avh_olo_prec
  use avh_olo_print
  use avh_olo_auxfun
  implicit none
  private
  public :: update_olog,olog,olog2

  include 'avh_olo_real.h90'
         ,allocatable,save :: thrs(:,:)
  integer,allocatable,save :: ntrm(:,:)
  integer,parameter :: nStp=6

  interface olog
    module procedure log_c,log_r
  end interface
  interface olog2
    module procedure log2_c,log2_r
  end interface

contains

  subroutine update_olog
!***********************************************************************
!***********************************************************************
  use avh_olo_arrays
  include 'avh_olo_real.h90'
    :: tt
  integer :: nn,mm,ii,jj
!  real(kind(1d0)) :: xx(6) !DEBUG
  if (allocated(thrs)) then
    call shift2( thrs ,prcpar )
    call shift2( ntrm ,prcpar )
  else
    allocate(thrs(1:nStp,1:1))
    allocate(ntrm(1:nStp,1:1))
    if (prcpar.ne.1) then
      if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop update_olog'
      stop
    endif
  endif
  if (prcpar.gt.1) then ;nn=ntrm(nStp,prcpar-1)-1
                   else ;nn=1
  endif
  do
    nn = nn+1
    mm = 2*nn-1
    tt = 1
    tt = (EPSN*mm)**(tt/(mm-1))
    tt = 2*tt/(1-tt)
! expansion from x=1+d with |d|=1/1000
    if (1000*tt.gt.RONE) exit
  enddo
  ntrm(nStp,prcpar) = nn
  thrs(nStp,prcpar) = tt
  nn = max(1,nint(nn*1d0/nStp))
  do ii=nStp-1,1,-1
    ntrm(ii,prcpar) = ntrm(ii+1,prcpar)-nn
    if (ntrm(ii,prcpar).le.1) then
      do jj=1,ii
        ntrm(jj,prcpar) = ntrm(ii,prcpar)
        thrs(jj,prcpar) = 0 
      enddo
      exit
    endif
    mm = 2*ntrm(ii,prcpar)-1
    tt = 1
    tt = (EPSN*mm)**(tt/(mm-1))
    thrs(ii,prcpar) = 2*tt/(1-tt)
  enddo
!  do ii=lbound(thrs,2),ubound(thrs,2) !DEBUG
!    do jj=1,nStp                      !DEBUG
!      xx(jj) = thrs(jj,ii)            !DEBUG
!    enddo                             !DEBUG
!    write(*,'(99e10.3)') xx(:)        !DEBUG
!    write(*,'(99i10)'  ) ntrm(:,ii)   !DEBUG
!  enddo                               !DEBUG
  end subroutine


  function log_c(xx,iph) result(rslt)
!***********************************************************************
!***********************************************************************
  include 'avh_olo_complex.h90'
          ,intent(in) :: xx
  integer ,intent(in) :: iph
  include 'avh_olo_complex.h90'
    :: rslt ,yy,zz,z2
  include 'avh_olo_real.h90'
    :: aa,rex,imx
  integer :: nn,ii,iyy
!
  rex = areal(xx)
  imx = aimag(xx)
  iyy = iph
!
  if (abs(imx).le.EPSN*abs(rex)) then
    if (rex.ge.RZRO) then
      rslt = log_r( rex, iyy )
    else
      rslt = log_r(-rex, iyy+sgnRe(imx) )
    endif
    return
  endif
!
  if (mod(iyy,2).eq.0) then
    yy = acmplx(rex,imx)
  else
    yy = acmplx(-rex,-imx)
    iyy = iyy+sgnRe(imx)
  endif
!
  if (iyy.ne.0) then
    rslt = log(yy) + IPI*iyy
    return
  endif
!
  zz = yy-1
  aa = abs(zz)
  if     (aa.ge.thrs(6,prcpar)) then
    rslt = log(yy)
    return
  elseif (aa.ge.thrs(5,prcpar)) then ;nn=ntrm(6,prcpar)
  elseif (aa.ge.thrs(4,prcpar)) then ;nn=ntrm(5,prcpar)
  elseif (aa.ge.thrs(3,prcpar)) then ;nn=ntrm(4,prcpar)
  elseif (aa.ge.thrs(2,prcpar)) then ;nn=ntrm(3,prcpar)
  elseif (aa.ge.thrs(1,prcpar)) then ;nn=ntrm(2,prcpar)
                                else ;nn=ntrm(1,prcpar)
  endif
  zz = zz/(yy+1)
  z2 = zz*zz
  aa = 2
  nn = 2*nn-1
  rslt = aa/nn
  do ii=nn-2,1,-2
    rslt = aa/ii + z2*rslt
  enddo
  rslt = zz*rslt
  end function


  function log_r(xx,iph) result(rslt)
!***********************************************************************
!***********************************************************************
  include 'avh_olo_real.h90'
          ,intent(in) :: xx
  integer ,intent(in) :: iph
  include 'avh_olo_complex.h90'
    :: rslt
  include 'avh_olo_real.h90'
    :: rr
  integer :: jj
!
  if (xx.eq.RZRO) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop log_r: ' &
       ,'xx =',trim(myprint(xx)),', returning 0'
    rslt = 0
    return
  elseif (xx.gt.RZRO) then ;rr= xx ;jj= iph
                      else ;rr=-xx ;jj= iph+1 ! log(-1)=i*pi
  endif
!
  rslt = log(rr) + IPI*jj
  end function


  function log2_c(xx,iph) result(rslt)
!***********************************************************************
!***********************************************************************
  include 'avh_olo_complex.h90'
          ,intent(in) :: xx
  integer ,intent(in) :: iph
  include 'avh_olo_complex.h90'
    :: rslt ,yy,zz,z2
  include 'avh_olo_real.h90'
    :: aa,rex,imx
  integer :: nn,ii,jj
!
  rex = areal(xx)
  imx = aimag(xx)
!
  if (abs(imx).le.EPSN*abs(rex)) then
    if (rex.ge.RZRO) then
      rslt = log2_r( rex, iph )
    else
      rslt = log2_r(-rex, iph+sgnRe(imx) )
    endif
    return
  endif
!
  if (mod(iph,2).eq.0) then ;yy= xx ;jj=iph
                       else ;yy=-xx ;jj=iph+sgnRe(imx)
  endif
!
  if (jj.ne.0) then
    rslt = ( log(yy) + IPI*jj )/(yy-1)
    return
  endif
!
  zz = yy-1
  aa = abs(zz)
  if     (aa.ge.thrs(6,prcpar)) then
    rslt = log(yy)/zz
    return
  elseif (aa.ge.thrs(5,prcpar)) then ;nn=ntrm(6,prcpar)
  elseif (aa.ge.thrs(4,prcpar)) then ;nn=ntrm(5,prcpar)
  elseif (aa.ge.thrs(3,prcpar)) then ;nn=ntrm(4,prcpar)
  elseif (aa.ge.thrs(2,prcpar)) then ;nn=ntrm(3,prcpar)
  elseif (aa.ge.thrs(1,prcpar)) then ;nn=ntrm(2,prcpar)
                                else ;nn=ntrm(1,prcpar)
  endif
  zz = zz/(yy+1)
  z2 = zz*zz
  aa = 2
  nn = 2*nn-1
  rslt = aa/nn
  do ii=nn-2,1,-2
    rslt = aa/ii + z2*rslt
  enddo
  rslt = rslt/(yy+1)
  end function


  function log2_r(xx,iph) result(rslt)
!***********************************************************************
!***********************************************************************
  include 'avh_olo_real.h90'
          ,intent(in) :: xx
  integer ,intent(in) :: iph
  include 'avh_olo_complex.h90'
    :: rslt
  include 'avh_olo_real.h90'
    :: rr,yy
  integer :: jj
!  include 'avh_olo_real.h90'
!    :: aa,zz,z2
!  integer :: nn,ii
!
  if (xx.eq.RZRO) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop log2_r: ' &
       ,'xx =',trim(myprint(xx)),', returning 0'
    rslt = 0
    return
  elseif (xx.gt.RZRO) then ;rr= xx ;jj=iph
                      else ;rr=-xx ;jj=iph+1 ! log(-1)=i*pi
  endif
!
  yy=rr ;if (mod(jj,2).ne.0) yy=-rr
!
  if (abs(yy-1).le.10*EPSN) then
    if (jj.ne.0) then
      if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop log2_r: ' &
        ,'rr,jj =',trim(myprint(rr)),jj,', putting jj to 0'
    endif
    rslt = 1 - (yy-1)/2
    return
  endif
!
  rslt = ( log(rr) + IPI*jj )/(yy-1)
  end function

end module
