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


module avh_olo_dilog
!***********************************************************************
!                     /1    ln(1-zz*t)
!   dilog(xx,iph) = - |  dt ---------- 
!                     /0        t
! with  zz = 1 - xx*exp(imag*pi*iph)  [pi, NOT 2*pi]
!
!   dilog(x1,i1,x2,i2) = ( dilog(x1,i1)-dilog(x2,i2) )/( x1-x2 )
!
! Arguments xx,x1,x2, may be all real or all complex,
! arguments iph,i1,i2 must be all integer.
!***********************************************************************
  use avh_olo_units
  use avh_olo_prec
  use avh_olo_print
  use avh_olo_auxfun
  use avh_olo_arrays
  implicit none
  private
  public :: update_dilog,dilog

  include 'avh_olo_real.h90'
         ,allocatable,save :: coeff(:)
  include 'avh_olo_real.h90'
         ,allocatable,save :: thrs(:,:)
  integer,allocatable,save :: ntrm(:,:)
  integer,parameter :: nStp=6

  include 'avh_olo_real.h90'
         ,allocatable :: bern(:),fact(:)

  interface dilog
    module procedure dilog_c,dilog_r,dilog2_c,dilog2_r
  end interface

contains

  subroutine update_dilog
!***********************************************************************
!***********************************************************************
  include 'avh_olo_real.h90'
    :: tt
  integer :: nn,ii,jj
  logical :: highestSoFar
!  real(kind(1d0)) :: xx(6) !DEBUG
!
  if (allocated(thrs)) then
    call shift2( thrs ,prcpar )
    call shift2( ntrm ,prcpar )
  else
    allocate(thrs(1:nStp,1:1))
    allocate(ntrm(1:nStp,1:1))
    if (prcpar.ne.1) then
      if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop update_dilog'
      stop
    endif
  endif
!
  highestSoFar = prcpar.eq.ubound(ntrm,2)
  if (highestSoFar) then
    if (allocated(coeff)) deallocate(coeff)
    allocate(coeff(0:-1)) ! allocate at size=0
  endif
!
  if (prcpar.gt.1) then ;nn=ntrm(nStp,prcpar-1)-1
                   else ;nn=2
  endif
!
  do
    nn = nn+1
    if (nn.gt.ubound(coeff,1)) call update_coeff( 2*nn )
    tt = 1
    tt = (EPSN/abs(coeff(nn)))**(tt/(2*nn))
! expansion parameter is smaller than 1.05
    if (100*tt.gt.105*RONE) exit
  enddo
!
  if (highestSoFar) call resize( coeff ,0,nn )
!
  ntrm(nStp,prcpar) = nn
  thrs(nStp,prcpar) = tt
  nn = max(1,nint(nn*1d0/nStp))
  do ii=nStp-1,1,-1
    ntrm(ii,prcpar) = ntrm(ii+1,prcpar)-nn
    if (ntrm(ii,prcpar).le.2) then
      do jj=1,ii
        ntrm(jj,prcpar) = max(2,ntrm(ii,prcpar))
        thrs(jj,prcpar) = 0 
      enddo
      exit
    endif
    jj = ntrm(ii,prcpar)
    tt = 1
    tt = (EPSN/abs(coeff(jj)))**(tt/(2*jj))
    thrs(ii,prcpar) = tt
  enddo
!
  if (allocated(bern)) deallocate(bern)
  if (allocated(fact)) deallocate(fact)
!
!  do ii=lbound(thrs,2),ubound(thrs,2) !DEBUG
!    do jj=1,nStp                      !DEBUG
!      xx(jj) = thrs(jj,ii)            !DEBUG
!    enddo                             !DEBUG
!    write(*,'(99e10.3)') xx(:)        !DEBUG
!    write(*,'(99i10)'  ) ntrm(:,ii)   !DEBUG
!  enddo                               !DEBUG
  end subroutine


  subroutine update_coeff( ncf )
!*******************************************************************
!   coeff(0)=-1/4
!   coeff(n)=bern(2*n)/(2*n+1)
!    bern(n)=bernoulli(n)/n!
!    fact(n)=n!
! DO NOT SKIP THE ODD bern IN THE RECURSIVE LOOP
! DO NOT PUT THE ODD bern TO ZERO
!*******************************************************************
  integer ,intent(in) :: ncf
  integer :: ii,jj,nbern,nold
!
  if (allocated(bern)) then ;nold=ubound(bern,1)
                       else ;nold=0
  endif
!
  nbern = 2*ncf
!
  call enlarge( bern  ,1,nbern   )
  call enlarge( fact  ,0,nbern+1 )
  call enlarge( coeff ,0,ncf     )
!
  fact(0) = 1
  do ii=nold+1,nbern+1
    fact(ii) = fact(ii-1)*ii
  enddo
!
  do ii=nold+1,nbern
    bern(ii) = -1/fact(ii+1)
    do jj=1,ii-1
      bern(ii) = bern(ii) - bern(jj)/fact(ii+1-jj)
    enddo
  enddo
!
  coeff(0) = 1
  coeff(0) =-coeff(0)/4
  do ii=nold+2,nbern,2
    coeff(ii/2) = bern(ii)/(ii+1)
  enddo
!
  end subroutine


  function dilog_c(xx,iph) result(rslt)
!*******************************************************************
!*******************************************************************
  include 'avh_olo_complex.h90'
          ,intent(in) :: xx
  integer ,intent(in) :: iph
  include 'avh_olo_complex.h90'
    :: rslt ,yy,lyy,loy,zz,z2
  include 'avh_olo_real.h90'
    :: rex,imx,az
  integer :: ii,jj,ntwo,odd,nn
  logical :: r_gt_1 , y_lt_h
!
  rex = areal(xx)
  imx = aimag(xx)
!
  if (abs(imx).le.EPSN*abs(rex)) then
    if (rex.ge.RZRO) then
      rslt = dilog_r( rex, iph )
    else
      rslt = dilog_r(-rex, iph+sgnRe(imx) )
    endif
    return
  endif
!
  if (rex.gt.RZRO) then ;yy= xx ;jj=iph
                   else ;yy=-xx ;jj=iph+sgnRe(imx)
  endif
!
  odd = mod(jj,2)
  ntwo = jj-odd
! 
  r_gt_1 = (rex*rex+imx*imx.gt.RONE)
  lyy = log(yy)
  if (odd.ne.0) yy = -yy
!
  if (r_gt_1) then
    yy   = 1/yy
    lyy  =-lyy
    ntwo =-ntwo
    odd  =-odd
  endif
  loy = log(1-yy)
!
  y_lt_h = (2*areal(yy).lt.RONE)
  if (y_lt_h) then ;zz=-loy
              else ;zz=-lyy
  endif
!
  az = abs(zz)
! if (az.gt.thrs(6,prcpar)) ERROR az to big 
  if     (az.ge.thrs(5,prcpar)) then ;nn=ntrm(6,prcpar)
  elseif (az.ge.thrs(4,prcpar)) then ;nn=ntrm(5,prcpar)
  elseif (az.ge.thrs(3,prcpar)) then ;nn=ntrm(4,prcpar)
  elseif (az.ge.thrs(2,prcpar)) then ;nn=ntrm(3,prcpar)
  elseif (az.ge.thrs(1,prcpar)) then ;nn=ntrm(2,prcpar)
                                else ;nn=ntrm(1,prcpar)
  endif
  z2 = zz*zz
  rslt = coeff(nn)
  do ii=nn,2,-1
    rslt = coeff(ii-1) + z2*rslt
  enddo
  rslt = zz*( 1 + zz*( coeff(0) + zz*rslt ) )
!
  if (y_lt_h) then
    rslt = 4*PISQo24 - rslt - loy*(lyy+IPI*(ntwo+odd))
  else
    rslt = rslt - loy*IPI*ntwo
  endif
!
  if (r_gt_1) rslt = -rslt - (lyy+IPI*(ntwo+odd))**2/2
  end function



  function dilog_r(xx,iph) result(rslt)
!*******************************************************************
!*******************************************************************
  include 'avh_olo_real.h90'
          ,intent(in) :: xx
  integer ,intent(in) :: iph
  include 'avh_olo_complex.h90'
    :: rslt
  include 'avh_olo_real.h90'
    :: yy,lyy,loy,zz,z2,liox,az
  integer :: jj,ii,ntwo,odd,nn
  logical :: r_gt_1 , y_lt_h
!
  if (xx.eq.RZRO) then
    rslt = 4*PISQo24
    return
  elseif (xx.gt.RZRO) then ;yy= xx ;jj=iph
                      else ;yy=-xx ;jj=iph+1 ! log(-1)=i*pi
  endif
!
  odd = mod(jj,2)
  ntwo = jj-odd
! 
  if (yy.eq.RONE.and.odd.eq.0) then
    if (ntwo.ne.0) then
      if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop dilog_r: ' &
        ,'|x|,iph = ',trim(myprint(yy)),',',jj,', returning 0'
    endif
    rslt = 0
    return
  endif
!
  r_gt_1 = (yy.gt.RONE)
  lyy = log(yy)
  if (odd.ne.0) yy = -yy
!
  if (r_gt_1) then
    yy   = 1/yy
    lyy  =-lyy
    ntwo =-ntwo
    odd  =-odd
  endif
  loy = log(1-yy) ! log(1-yy) is always real
!
  y_lt_h = (2*yy.lt.RONE)
  if (y_lt_h) then
    zz = -loy ! log(1-yy) is real
  else
    zz = -lyy ! yy>0.5 => log(yy) is real
  endif
!
  az = abs(zz)
! if (az.gt.thrs(6,prcpar)) ERROR az to big 
  if     (az.ge.thrs(5,prcpar)) then ;nn=ntrm(6,prcpar)
  elseif (az.ge.thrs(4,prcpar)) then ;nn=ntrm(5,prcpar)
  elseif (az.ge.thrs(3,prcpar)) then ;nn=ntrm(4,prcpar)
  elseif (az.ge.thrs(2,prcpar)) then ;nn=ntrm(3,prcpar)
  elseif (az.ge.thrs(1,prcpar)) then ;nn=ntrm(2,prcpar)
                                else ;nn=ntrm(1,prcpar)
  endif
  z2 = zz*zz
  liox = coeff(nn)
  do ii=nn,2,-1
    liox = coeff(ii-1) + z2*liox
  enddo
  liox = zz*( 1 + zz*( coeff(0) + zz*liox ) )
!
  rslt = acmplx(liox)
!
  if (y_lt_h) then
    rslt = 4*PISQo24 - rslt - acmplx(loy*lyy,loy*ONEPI*(ntwo+odd))
  else
    rslt = rslt + acmplx( 0 ,-loy*ONEPI*ntwo )
  endif
!
  if (r_gt_1) rslt = -rslt - acmplx(lyy,ONEPI*(ntwo+odd))**2/2
  end function


  function dilog2_c( x1,i1 ,x2,i2 ) result(rslt)
!*******************************************************************
!*******************************************************************
  use avh_olo_olog
  include 'avh_olo_complex.h90'
          ,intent(in) :: x1,x2
  integer ,intent(in) :: i1,i2
  include 'avh_olo_complex.h90'
    :: rslt ,y1,y2 ,ff,gg,logr1,logr2,logo1,logo2,r1,r2,rr
  include 'avh_olo_real.h90'
    :: eps ,re1,im1,re2,im2,a1,a2,aa,ao1,ao2
  integer :: j1,j2,ii,nn,oo
  integer,parameter :: pp(-1:1,-1:1)=&
                      reshape((/-2,-2,2 ,-2,0,2 ,-2,2,2/),(/3,3/))
!
  re1=areal(x1) ;re2=areal(x2)
  im1=aimag(x1) ;im2=aimag(x2)
!
  if (abs(im1).le.EPSN*abs(re1).and.abs(im2).le.EPSN*abs(re2)) then
    if (re1.ge.RZRO) then
      if (re2.ge.RZRO) then
        rslt = dilog2_r( re1,i1 , re2,i2 )
      else
        rslt = dilog2_r( re1,i1 ,-re2,i2+sgnRe(im2) )
      endif
    elseif (re2.ge.RZRO) then
      rslt = dilog2_r(-re1,i1+sgnRe(im1) , re2,i2 )
    else
      rslt = dilog2_r(-re1,i1+sgnRe(im1) ,-re2,i2+sgnRe(im2) )
    endif
    return
  endif
!
  if (re1.ge.RZRO) then ;r1= x1 ;j1=i1
                   else ;r1=-x1 ;j1=i1+sgnRe(im1,1)
  endif
  if (re2.ge.RZRO) then ;r2= x2 ;j2=i2
                   else ;r2=-x2 ;j2=i2+sgnRe(im2,1)
  endif
!
  a1=abs(r1) ;a2=abs(r2)
  if (a1.gt.a2) then
    aa=a1;a1=a2;a2=aa
    rr=r1;r1=r2;r2=rr
    ii=j1;j1=j2;j2=ii
  endif
!
  oo=mod(j1,2) ;nn=j1-oo ;y1=r1 ;if (oo.ne.0) y1=-y1
  oo=mod(j2,2) ;nn=j2-oo ;y2=r2 ;if (oo.ne.0) y2=-y2
!
  eps = 10*EPSN
!
  if (j1.ne.j2) then
    if (r1.eq.r2) then
      if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop dilog2_c: ' &
        ,'j1,j2,r1-r2',j1,j2,',',trim(myprint(r1-r2)),', returning 0'
      rslt = 0
!      write(*,*) 'dilog2_c j1=/=j2,r1=r2' !DEBUG
      return
    else
      rslt = ( dilog_c(r1,j1)-dilog_c(r2,j2) )/(y1-y2)
!      write(*,*) 'dilog2_c j1=/=j2' !DEBUG
      return
    endif
  endif
!
  if (a1.lt.eps) then
    if (a2.lt.eps) then
      if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop dilog2_c: ' &
        ,'r1,r2 =',trim(myprint(r1)),',',trim(myprint(r2)),', returning 0'
      rslt = 0
!      write(*,*) 'dilog2_c r1<eps,r2<eps' !DEBUG
      return
    else
      rslt = (dilog_c(r2,j2)-4*PISQo24)/y2
!      write(*,*) 'dilog2_c r1<eps' !DEBUG
      return
    endif
  endif
!
  logr1=log(r1) ;logr2=log(r2)
!
  ao1=abs(1-y1) ;ao2=abs(1-y2)
  if (10*ao1.lt.RONE.or.10*ao2.lt.RONE) then
    aa = abs(r1/r2-1)
    if (10*aa.gt.RONE) then
      rslt = (dilog_c(r1,j1)-dilog_c(r2,j2))/(y1-y2)
!      write(*,*) 'dilog2_c ||1-y1|/|1-y2|-1|>0.1' !DEBUG
      return
    elseif (oo.eq.0.and.ao1.lt.eps) then
      if (nn.ne.0.and.eunit.gt.0) write(eunit,*) 'ERROR in OneLOop dilog2_c: ' &
        ,'r1,oo,nn =',trim(myprint(r1)),',',oo,nn,', putting nn=0'
      if (ao2.lt.eps) then
        rslt = -1
!        write(*,*) 'dilog2_c |1-y1|' !DEBUG
        return
      else
        y1=1-eps ;nn=0 ;logr1=0 ;r1=1-eps
      endif
    elseif (oo.eq.0.and.ao2.lt.eps) then
      if (nn.ne.0.and.eunit.gt.0) write(eunit,*) 'ERROR in OneLOop dilog2_c: ' &
        ,'r2,oo,nn =',trim(myprint(r2)),',',oo,nn,', putting nn=0'
      y2=1-eps ;nn=0 ;logr2=0 ;r2=1-eps
    endif
  else
    aa = abs((logr1+oo*IPI)/(logr2+oo*IPI)-1)
    if (10*aa.gt.RONE) then
      rslt = (dilog_c(r1,j1)-dilog_c(r2,j2))/(y1-y2)
!      write(*,*) 'dilog2_c |logr1/logr2-1|>0.1',logr1,logr2 !DEBUG
      return
    elseif (aa.lt.eps) then
      ii = 0
      if (a1.gt.RONE) ii = ii + (nn+pp(oo,sgnIm(y2)))
      if (a2.gt.RONE) ii = ii - (nn+pp(oo,sgnIm(y2)))
      ii = nn*ii
      if (ii.ne.0.and.eunit.gt.0) write(eunit,*) 'ERROR in OneLOop dilog2_c: ' &
        ,'r1,r2,nn =',trim(myprint(r1)),',',trim(myprint(r2)),',',nn &
        ,', putting nn=0'
      rslt = -olog2(y2,0)
!      write(*,*) 'dilog2_c |logr1/lorg2|<eps' !DEBUG
      return
    endif
  endif
!
  if (a1.gt.RONE) then
    y1=1/y1 ;logr1=-logr1
    y2=1/y2 ;logr2=-logr2
    nn=-nn ;oo=-oo
  endif
!
  ff=y1/y2         ;ff=-olog2(ff,0)/y2
  gg=(1-y1)/(1-y2) ;gg=-olog2(gg,0)/(1-y2)
!
  if (2*areal(y1).ge.RONE) then
!    write(*,*) 'dilog2_c re>1/2' !DEBUG
    rslt = ff*sumterms_c(-logr1,-logr2) - nn*IPI*gg
  else
!    write(*,*) 'dilog2_c re<1/2' !DEBUG
    logo1 = log(1-y1)
    logo2 = log(1-y2)
    rslt = gg*( sumterms_c(-logo1,-logo2) - (nn+oo)*IPI - logr2 ) + ff*logo1
  endif
!
  if (a1.gt.RONE) then !implies also r2>1
!    write(*,*) 'dilog2_c r1>1,r2>1' !DEBUG
    rslt = y1*y2*( rslt - ff*((logr1+logr2)/2 + (nn+oo)*IPI) )
  elseif (a2.gt.RONE.and.nn.ne.0) then
!    write(*,*) 'dilog2_c r1<1,r2>1',oo,sgnIm(y2)!DEBUG
    rslt = rslt - 12*nn*( nn + pp(oo,sgnIm(y2)) )*PISQo24/(y1-y2)
  endif
!
  end function


  function dilog2_r( x1,i1 ,x2,i2 ) result(rslt)
!*******************************************************************
!*******************************************************************
  use avh_olo_olog
  include 'avh_olo_real.h90'
          ,intent(in) :: x1,x2
  integer ,intent(in) :: i1,i2
  include 'avh_olo_complex.h90'
    :: rslt
  include 'avh_olo_real.h90'
    :: y1,y2 ,ff,gg,logr1,logr2,logo1,logo2
  include 'avh_olo_real.h90'
    :: eps,r1,r2,rr,ro1,ro2
  integer :: j1,j2,ii,nn,oo
!
  if (x1.ge.RZRO) then ;r1= x1 ;j1=i1
                  else ;r1=-x1 ;j1=i1+1 ! log(-1)=i*pi
  endif
  if (x2.ge.RZRO) then ;r2= x2 ;j2=i2
                  else ;r2=-x2 ;j2=i2+1 ! log(-1)=i*pi
  endif
!
  if (r1.gt.r2) then
    rr=r1;r1=r2;r2=rr
    ii=j1;j1=j2;j2=ii
  endif
!
  oo=mod(j1,2) ;nn=j1-oo ;y1=r1 ;if (oo.ne.0) y1=-y1
  oo=mod(j2,2) ;nn=j2-oo ;y2=r2 ;if (oo.ne.0) y2=-y2
!
  eps = 10*EPSN
!
  if (j1.ne.j2) then
    if (r1.eq.r2) then
      if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop dilog2_r: ' &
        ,'j1,j2,r1-r2',j1,j2,',',trim(myprint(r1-r2)),', returning 0'
      rslt = 0
!      write(*,*) 'dilog2_r j1=/=j2,r1=r2' !DEBUG
      return
    else
      rslt = ( dilog_r(r1,j1)-dilog_r(r2,j2) )/(y1-y2)
!      write(*,*) 'dilog2_r j1=/=j2' !DEBUG
      return
    endif
  endif
!
  if (r1.lt.eps) then
    if (r2.lt.eps) then
      if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop dilog2_r: ' &
        ,'r1,r2 =',trim(myprint(r1)),',',trim(myprint(r2)),', returning 0'
      rslt = 0
!      write(*,*) 'dilog2_r r1<eps,r2<eps' !DEBUG
      return
    else
      rslt = (dilog_r(r2,j2)-4*PISQo24)/y2
!      write(*,*) 'dilog2_r r1<eps' !DEBUG
      return
    endif
  endif
!
  logr1=log(r1) ;logr2=log(r2)
!
  ro1=abs(1-y1) ;ro2=abs(1-y2)
  if (10*ro1.lt.RONE.or.10*ro2.lt.RONE) then
    rr = abs(r1/r2-1)
    if (10*rr.gt.RONE) then
      rslt = (dilog_r(r1,j1)-dilog_r(r2,j2))/(y1-y2)
!      write(*,*) 'dilog2_r ||1-y1|/|1-y2|-1|>0.1' !DEBUG
      return
    elseif (oo.eq.0.and.ro1.lt.eps) then
      if (nn.ne.0.and.eunit.gt.0) write(eunit,*) 'ERROR in OneLOop dilog2_r: ' &
        ,'r1,oo,nn =',trim(myprint(r1)),',',oo,nn,', putting nn=0'
      if (ro2.lt.eps) then
        rslt = -1
!        write(*,*) 'dilog2_r |1-y1|' !DEBUG
        return
      else
        y1=1-eps ;nn=0 ;logr1=0 ;r1=1-eps
      endif
    elseif (oo.eq.0.and.ro2.lt.eps) then
      if (nn.ne.0.and.eunit.gt.0) write(eunit,*) 'ERROR in OneLOop dilog2_r: ' &
        ,'r2,oo,nn =',trim(myprint(r2)),',',oo,nn,', putting nn=0'
      y2=1-eps ;nn=0 ;logr2=0 ;r2=1-eps
    endif
  else
    rr = abs((logr1+oo*IPI)/(logr2+oo*IPI)-1)
    if (10*rr.gt.RONE) then
      rslt = (dilog_r(r1,j1)-dilog_r(r2,j2))/(y1-y2)
!      write(*,*) 'dilog2_r |logr1/logr2-1|>0.1',logr1,logr2 !DEBUG
      return
    elseif (rr.lt.eps) then
      ii = 0
      if (r1.gt.RONE) ii = ii + (nn+2*oo)
      if (r2.gt.RONE) ii = ii - (nn+2*oo)
      ii = nn*ii
      if (ii.ne.0.and.eunit.gt.0) write(eunit,*) 'ERROR in OneLOop dilog2_r: ' &
        ,'r1,r2,nn =',trim(myprint(r1)),',',trim(myprint(r2)),',',nn &
        ,', putting nn=0'
      rslt = -olog2(y2,0)
!      write(*,*) 'dilog2_r |logr1/lorg2|<eps' !DEBUG
      return
    endif
  endif
!
  if (r1.gt.RONE) then
    y1=1/y1 ;logr1=-logr1
    y2=1/y2 ;logr2=-logr2
    nn=-nn ;oo=-oo
  endif
!
  ff=y1/y2         ;ff=-olog2(ff,0)/y2
  gg=(1-y1)/(1-y2) ;gg=-olog2(gg,0)/(1-y2)
!
  if (2*y1.ge.RONE) then
!    write(*,*) 'dilog2_r re>1/2' !DEBUG
    rslt = ff*sumterms_r(-logr1,-logr2) - nn*IPI*gg
  else
!    write(*,*) 'dilog2_r re<1/2' !DEBUG
    logo1 = log(1-y1)
    logo2 = log(1-y2)
    rslt = gg*( sumterms_r(-logo1,-logo2) - (nn+oo)*IPI - logr2 ) + ff*logo1
  endif
!
  if (r1.gt.RONE) then !implies also r2>1
!    write(*,*) 'dilog2_r r1>1,r2>1' !DEBUG
    rslt = y1*y2*( rslt - ff*((logr1+logr2)/2 + (nn+oo)*IPI) )
  elseif (r2.gt.RONE.and.nn.ne.0) then
!    write(*,*) 'dilog2_r r1<1,r2>1' !DEBUG
    rslt = rslt - 12*nn*PISQo24*(nn+2*oo)/(y1-y2)
  endif
!
  end function


  function sumterms_c( z1,z2 ) result(rslt)
!***********************************************************************
! ( f(z1)-f(z2) )/( z1-z2 ), where
! f(z)= z + c0*z^2 + c1*z^3 + c2*z^5 + c3*z^7 + ...
!***********************************************************************
  include 'avh_olo_complex.h90'
    ,intent(in) :: z1,z2
  include 'avh_olo_complex.h90'
    :: rslt,yy,zz
  include 'avh_olo_real.h90'
    :: az
  integer :: ii,nn
  az = max(abs(z1),abs(z2))
  if     (az.ge.thrs(5,prcpar)) then ;nn=ntrm(6,prcpar)
  elseif (az.ge.thrs(4,prcpar)) then ;nn=ntrm(5,prcpar)
  elseif (az.ge.thrs(3,prcpar)) then ;nn=ntrm(4,prcpar)
  elseif (az.ge.thrs(2,prcpar)) then ;nn=ntrm(3,prcpar)
  elseif (az.ge.thrs(1,prcpar)) then ;nn=ntrm(2,prcpar)
                                else ;nn=ntrm(1,prcpar)
  endif
! calculates all z(i)=(z1^i-z2^i)/(z1-z2) numerically stable
!  zz(1) = 1
!  yy    = 1
!  do ii=2,2*nn+1
!    yy = z2*yy
!    zz(ii) = z1*zz(ii-1) + yy
!  enddo
  zz = 1
  yy = 1
  rslt = zz
  yy = z2*yy
  zz = z1*zz+yy
  rslt = rslt + coeff(0)*zz
  do ii=1,nn
    yy = z2*yy
    zz = z1*zz+yy
    rslt = rslt + coeff(ii)*zz
    yy = z2*yy
    zz = z1*zz+yy
  enddo
  end function  


  function sumterms_r( z1,z2 ) result(rslt)
!***********************************************************************
! ( f(z1)-f(z2) )/( z1-z2 ), where
! f(z)= z + c0*z^2 + c1*z^3 + c2*z^5 + c3*z^7 + ...
!***********************************************************************
  include 'avh_olo_real.h90'
    ,intent(in) :: z1,z2
  include 'avh_olo_real.h90'
    :: rslt,yy,zz
  include 'avh_olo_real.h90'
    :: az
  integer :: ii,nn
  az = max(abs(z1),abs(z2))
  if     (az.ge.thrs(5,prcpar)) then ;nn=ntrm(6,prcpar)
  elseif (az.ge.thrs(4,prcpar)) then ;nn=ntrm(5,prcpar)
  elseif (az.ge.thrs(3,prcpar)) then ;nn=ntrm(4,prcpar)
  elseif (az.ge.thrs(2,prcpar)) then ;nn=ntrm(3,prcpar)
  elseif (az.ge.thrs(1,prcpar)) then ;nn=ntrm(2,prcpar)
                                else ;nn=ntrm(1,prcpar)
  endif
  zz = 1
  yy = 1
  rslt = zz
  yy = z2*yy
  zz = z1*zz+yy
  rslt = rslt + coeff(0)*zz
  do ii=1,nn
    yy = z2*yy
    zz = z1*zz+yy
    rslt = rslt + coeff(ii)*zz
    yy = z2*yy
    zz = z1*zz+yy
  enddo
  end function  

end module
