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


module avh_olo_bnlog
!***********************************************************************
!                      /1    
!   bnlog(n,x) = (n+1) |  dt t^n ln(1-t/x) 
!                      /0 
!***********************************************************************
  use avh_olo_units
  use avh_olo_prec
  use avh_olo_auxfun
  use avh_olo_arrays
  use avh_olo_olog
  use avh_olo_print
  implicit none
  private
  public :: update_bnlog,bnlog

  include 'avh_olo_real.h90'
         ,allocatable,save :: coeff(:,:)
  include 'avh_olo_real.h90'
         ,allocatable,save :: thrs(:,:,:)
  integer,allocatable,save :: ntrm(:,:,:)
  integer,parameter :: nStp=6
  integer,parameter :: rank=4
  integer,parameter :: aCoef(0:rank,0:rank)=reshape((/ &
                         1, 0, 0, 0, 0 & ! 1
                       , 1, 2, 0, 0, 0 & ! 1/2,1
                       , 2, 3, 6, 0, 0 & ! 1/3,1/2,1
                       , 3, 4, 6,12, 0 & ! 1/4,1/3,1/2,1
                       ,12,15,20,30,60 & ! 1/5,1/4,1/3,1/2,1
                       /),(/rank+1,rank+1/))

  interface bnlog
    module procedure bnlog_c,bnlog_r
  end interface

contains


  subroutine update_bnlog
!***********************************************************************
!***********************************************************************
  include 'avh_olo_real.h90'
    :: tt
  integer :: nn,ii,jj,n1,nmax,irank
  logical :: highestSoFar
!  real(kind(1d0)) :: xx(6) !DEBUG
!
  if (allocated(thrs)) then
    call shift3( thrs ,prcpar )
    call shift3( ntrm ,prcpar )
  else
    allocate(thrs(1:nStp,0:rank,1:1))
    allocate(ntrm(1:nStp,0:rank,1:1))
    if (prcpar.ne.1) then
      if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop update_bnlog'
      stop
    endif
  endif
!
  highestSoFar = prcpar.eq.ubound(ntrm,3)
!
  if (highestSoFar) then
    if (allocated(coeff)) deallocate(coeff)
    allocate(coeff(0:-1,0:2)) ! allocate at size=0
  endif
!
  nmax = 0
!
  do irank=0,rank
!
    n1 = 2+irank
!
    if (prcpar.gt.1) then ;nn=ntrm(nStp,irank,prcpar-1)-1
                     else ;nn=n1
    endif
!  
    do
      nn = nn+1
      if (highestSoFar.and.nn.gt.ubound(coeff,1)) call update_coeff( 2*nn )
      tt = 1
      tt = (EPSN*abs(coeff(n1,irank)/coeff(nn,irank)))**(tt/(nn-n1))
      if (8*(irank+1)*tt.gt.RONE) exit
    enddo
!
    if (nn.gt.nmax) nmax=nn
!  
    ntrm(nStp,irank,prcpar) = nn
    thrs(nStp,irank,prcpar) = tt
    nn = max(1,nint(nn*1d0/nStp))
    do ii=nStp-1,1,-1
      ntrm(ii,irank,prcpar) = ntrm(ii+1,irank,prcpar)-nn
      if (ntrm(ii,irank,prcpar).le.n1) then
        do jj=1,ii
          ntrm(jj,irank,prcpar) = max(n1,ntrm(ii,irank,prcpar))
          thrs(jj,irank,prcpar) = 0 
        enddo
        exit
      endif
      jj = ntrm(ii,irank,prcpar)
      tt = 1
      tt = (EPSN*abs(coeff(n1,irank)/coeff(jj,irank)))**(tt/(jj-n1))
      thrs(ii,irank,prcpar) = tt
    enddo
!  
  enddo!irank=1,nrank
!  
  if (highestSoFar) call resize( coeff ,2,nmax ,0,rank )
!
!  do ii=lbound(thrs,3),ubound(thrs,3)        !DEBUG
!  do irank=0,rank                            !DEBUG
!    do jj=1,nStp                             !DEBUG
!      xx(jj) = thrs(jj,irank,ii)             !DEBUG
!    enddo                                    !DEBUG
!    write(*,'(i2,99e10.3)') irank,xx(:)      !DEBUG
!    write(*,'(2x,99i10)'  ) ntrm(:,irank,ii) !DEBUG
!  enddo                                      !DEBUG
!  enddo                                      !DEBUG
  end subroutine


  subroutine update_coeff( ncf )
!*******************************************************************
! Coefficients of the expansion of
!   f(n,x) = -int( t^n*log(1-t*x) ,t=0..1 )
! in terms of log(1-x)
!*******************************************************************
  integer ,intent(in) :: ncf
  integer :: ii,jj
  include 'avh_olo_real.h90'
    :: fact,tt(rank)
!
  call enlarge( coeff ,2,ncf ,0,rank )
!
  do jj=0,rank
  do ii=2,1+jj
    coeff(ii,jj) = 0
  enddo
  enddo
  fact = 1
  do ii=1,rank ;tt(ii)=1 ;enddo
  do ii=2,ncf
    fact = fact*ii
    coeff(ii,0) = (ii-1)/fact
    if (ii.eq.2) cycle
    do jj=1,rank ;tt(jj)=tt(jj)*(jj+1) ;enddo
    coeff(ii,1) = coeff(ii,0)*(1-tt(1))
    if (ii.eq.3) cycle
    coeff(ii,2) = coeff(ii,0)*(1-2*tt(1)+tt(2))
    if (ii.eq.4) cycle
    coeff(ii,3) = coeff(ii,0)*(1-3*tt(1)+3*tt(2)-tt(3))
    if (ii.eq.5) cycle
    coeff(ii,4) = coeff(ii,0)*(1-4*tt(1)+6*tt(2)-4*tt(3)+tt(4))
!   if (ii.eq.n+1) cycle
!   coeff(ii,n) = coeff(ii,0)
!               * ( 1 - binom(n,1)*tt(1) + binom(n,2)*tt(2)...)
  enddo
!
  end subroutine


  function bnlog_c( irank ,xx ) result(rslt)
!*******************************************************************
!*******************************************************************
  integer ,intent(in) :: irank
  include 'avh_olo_complex.h90'
    ,intent(in) :: xx
  include 'avh_olo_complex.h90'
    :: rslt,yy,omx
  include 'avh_olo_real.h90'
    :: aa,rex,imx
  integer :: ii,nn
!
  rex = areal(xx)
  imx = aimag(xx)
!
  if (abs(imx).le.EPSN*abs(rex)) then
    rslt = bnlog_r( irank ,rex ,sgnRe(imx,1) )
    return
  endif
!
  if (abs(xx-1).le.EPSN*10) then
    aa = 1
    rslt = -1
    do ii=2,irank+1
      rslt = rslt - aa/ii
    enddo
    return
  endif
!
  yy = olog(1-1/xx,0)
  aa = abs(yy)
  if     (aa.ge.thrs(6,irank,prcpar)) then
     omx = 1
    rslt = aCoef(irank,irank)
    do ii=irank,1,-1
       omx = 1 + xx*omx
      rslt = aCoef(ii-1,irank) + xx*rslt
    enddo
     omx = (1-xx)*omx
    rslt = omx*yy - rslt/aCoef(irank,irank)
!    if     (irank.eq.0) then
!      rslt = (1-xx)*yy - 1
!    elseif (irank.eq.1) then
!      rslt = (1-xx)*(1+xx)*yy - (1+xx*2)/2
!    elseif (irank.eq.2) then
!      rslt = (1-xx)*(1+xx*(1+xx))*yy - (2+xx*(3+xx*6))/6
!    elseif (irank.eq.3) then
!      rslt = (1-xx)*(1+xx*(1+xx*(1+xx)))*yy &
!           - (3+xx*(4+xx*(6+xx*12)))/12
!    elseif (irank.eq.4) then
!      rslt = (1-xx)*(1+xx*(1+xx*(1+xx*(1+xx))))*yy &
!           - (12+xx*(15+xx*(20+xx*(30+xx*60))))/60
!    endif
    return
  elseif (aa.ge.thrs(5,irank,prcpar)) then ;nn=ntrm(6,irank,prcpar)
  elseif (aa.ge.thrs(4,irank,prcpar)) then ;nn=ntrm(5,irank,prcpar)
  elseif (aa.ge.thrs(3,irank,prcpar)) then ;nn=ntrm(4,irank,prcpar)
  elseif (aa.ge.thrs(2,irank,prcpar)) then ;nn=ntrm(3,irank,prcpar)
  elseif (aa.ge.thrs(1,irank,prcpar)) then ;nn=ntrm(2,irank,prcpar)
                                      else ;nn=ntrm(1,irank,prcpar)
  endif
!
  rslt = coeff(nn,irank)
  do ii=nn-1,2+irank,-1
    rslt = coeff(ii,irank) + yy*rslt
  enddo
  rslt = -(irank+1)*rslt*yy*(yy*xx)**(irank+1)
!
  aa = areal(rslt)
  if (abs(aimag(rslt)).le.EPSN*abs(aa)) rslt = acmplx(aa)
!
  end function


  function bnlog_r( irank ,xx ,sgn ) result(rslt)
!*******************************************************************
!*******************************************************************
  integer ,intent(in) :: irank
  include 'avh_olo_real.h90'
          ,intent(in) :: xx
  integer ,intent(in) :: sgn
  include 'avh_olo_complex.h90'
    :: rslt
  include 'avh_olo_real.h90'
    :: yy,aa,omx
  integer :: ii,nn
  logical :: y_lt_0
!
  if (abs(xx).eq.RZRO) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop bnlog_r: ' &
      ,'argument xx=',trim(myprint(xx,8)),', returning 0'
    rslt = 0
    return
  elseif (abs(xx-1).le.EPSN*10) then
    aa = 1
    rslt = -1
    do ii=2,irank+1
      rslt = rslt - aa/ii
    enddo
    return
  endif
!
  yy = 1-1/xx
  y_lt_0 = (yy.lt.RZRO)
  if (y_lt_0) then 
    yy = log(-yy)
    aa = sqrt(yy*yy+ONEPI*ONEPI)
  else
    yy = log( yy)
    aa = abs(yy)
  endif
!
  omx = 1
  do ii=irank,1,-1
    omx = 1+xx*omx
  enddo
  omx = (1-xx)*omx ! (1-x^{rank+1})
!
  if     (aa.ge.thrs(6,irank,prcpar)) then
    rslt = aCoef(irank,irank)
    do ii=irank,1,-1
      rslt = aCoef(ii-1,irank) + xx*rslt
    enddo
    rslt = omx*yy - rslt/aCoef(irank,irank)
!    if     (irank.eq.0) then
!      rslt = omx*yy - 1
!    elseif (irank.eq.1) then
!      rslt = omx*yy - (1+xx*2)/2
!    elseif (irank.eq.2) then
!      rslt = omx*yy - (2+xx*(3+xx*6))/6
!    elseif (irank.eq.3) then
!      rslt = omx*yy - (3+xx*(4+xx*(6+xx*12)))/12
!    elseif (irank.eq.4) then
!      rslt = omx*yy - (12+xx*(15+xx*(20+xx*(30+xx*60))))/60
!    endif
    if (y_lt_0) rslt = rslt + sgn*omx*IPI
    return
  elseif (aa.ge.thrs(5,irank,prcpar)) then ;nn=ntrm(6,irank,prcpar)
  elseif (aa.ge.thrs(4,irank,prcpar)) then ;nn=ntrm(5,irank,prcpar)
  elseif (aa.ge.thrs(3,irank,prcpar)) then ;nn=ntrm(4,irank,prcpar)
  elseif (aa.ge.thrs(2,irank,prcpar)) then ;nn=ntrm(3,irank,prcpar)
  elseif (aa.ge.thrs(1,irank,prcpar)) then ;nn=ntrm(2,irank,prcpar)
                                      else ;nn=ntrm(1,irank,prcpar)
  endif
!
  aa = coeff(nn,irank)
  do ii=nn-1,2+irank,-1
    aa = coeff(ii,irank) + yy*aa
  enddo
  rslt = -(irank+1)*aa*yy*(yy*xx)**(irank+1)
  if (y_lt_0) rslt = rslt + sgn*omx*IPI
!  
  end function

end module
