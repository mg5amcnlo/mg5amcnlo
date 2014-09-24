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


module avh_olo_box
  use avh_olo_units
  use avh_olo_prec
  use avh_olo_auxfun
  use avh_olo_qmplx
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
   include 'avh_olo_complex.h90'
     ,intent(out) :: rslt(0:2)
   include 'avh_olo_complex.h90'
     ,intent(in)  :: p2,p3,p12,p23 ,m2,m3,m4
   include 'avh_olo_real.h90'
     ,intent(in)  :: rmu
   include 'avh_olo_complex.h90'
     :: cp2,cp3,cp12,cp23,cm2,cm3,cm4,sm1,sm2,sm3,sm4 &
                     ,r13,r23,r24,r34,d23,d24,d34,log24,cc
   type(qmplx_type) :: q13,q23,q24,q34,qss,qy1,qy2,qz1,qz2
![CALLINGME  write(*,*) 'MESSAGE from OneLOop box16: you are calling me'
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
     rslt(0)=0 ;rslt(1)=0 ;rslt(2)=0
     return
   endif
!
   sm1 = abs(rmu)
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
   if (r24.eq.-CONE) then 
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box16: ' &
       ,'threshold singularity, returning 0'
     rslt(0)=0 ;rslt(1)=0 ;rslt(2)=0
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
   cc = 1/( sm2*sm4*(cp12-cm3) )
   log24 = logc2(q24)*r24/(1+r24)
   rslt(2) = 0
   rslt(1) = -log24
   rslt(0) = log24*logc(qss) + li2c2(q24*q24,qonv(1))*r24 &
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
   include 'avh_olo_complex.h90'
     ,intent(out) :: rslt(0:2)
   include 'avh_olo_complex.h90'
     ,intent(in)  :: p2,p3,p12,p23 ,m2,m4
   include 'avh_olo_real.h90'
     ,intent(in)  :: rmu
   include 'avh_olo_complex.h90'
     :: cp2,cp3,cp12,cp23,cm2,cm4,sm1,sm2,sm3,sm4 &
                     ,r13,r23,r24,r34,d24,log24,cc
   type(qmplx_type) :: q13,q23,q24,q34,qss,qz1,qz2
![CALLINGME  write(*,*) 'MESSAGE from OneLOop box15: you are calling me'
!
   if (abs(m2-p2).gt.abs(m4-p3)) then
     cm2=m2 ;cm4=m4 ;cp2=p2 ;cp3=p3
   else
     cm2=m4 ;cm4=m2 ;cp2=p3 ;cp3=p2
   endif
   cp12=p12 ;cp23=p23
!
   if (cp12.eq.CZRO) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box15: ' &
       ,'p12=0, returning 0'
     rslt(0)=0 ;rslt(1)=0 ;rslt(2)=0
     return
   endif
!
   sm1 = abs(rmu)
   sm2 = mysqrt(cm2)
   sm4 = mysqrt(cm4)
   sm3 = abs(sm2)
   r13 = (       -cp12)/(sm1*sm3)
   r23 = (cm2    -cp2 )/(sm2*sm3)
   r34 = (    cm4-cp3 )/(sm3*sm4)
   call rfun( r24,d24 ,(cm2+cm4-cp23)/(sm2*sm4) )
!
   if (r24.eq.-CONE) then 
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box15: ' &
       ,'threshold singularity, returning 0'
     rslt(0)=0 ;rslt(1)=0 ;rslt(2)=0
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
   log24 = logc2(q24)/(1+r24)
   rslt(2) = 0
   rslt(1) = -log24
   rslt(0) = log24 * logc(qss) + li2c2(q24*q24,qonv(1))
   if (r34.ne.CZRO) then
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
   include 'avh_olo_complex.h90'
     ,intent(out) :: rslt(0:2)
   include 'avh_olo_complex.h90'
     ,intent(in)  :: cp12,cp23,cm2,cm4
   include 'avh_olo_real.h90'
     ,intent(in)  :: rmu
   include 'avh_olo_complex.h90'
     :: sm2,sm4,r24,d24,cc
![CALLINGME  write(*,*) 'MESSAGE from OneLOop box14: you are calling me'
!
   if (cp12.eq.CZRO) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box14: ' &
       ,'p12=0, returning 0'
     rslt(0)=0 ;rslt(1)=0 ;rslt(2)=0
     return
   endif
!
   sm2 = mysqrt(cm2)
   sm4 = mysqrt(cm4)
   call rfun( r24,d24 ,(cm2+cm4-cp23)/(sm2*sm4) )
!
   if (r24.eq.-CONE) then 
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box14: ' &
       ,'threshold singularity, returning 0'
     rslt(0)=0 ;rslt(1)=0 ;rslt(2)=0
     return
   endif
!
   cc = -2*logc2(qonv(r24,-1))*r24/(1+r24)/(sm2*sm4*cp12)
!
   rslt(2) = 0
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
   include 'avh_olo_complex.h90'
     ,intent(out) :: rslt(0:2)
   include 'avh_olo_complex.h90'
     ,intent(in)  :: p2,p3,p4,p12,p23,m3,m4
   include 'avh_olo_real.h90'
     ,intent(in)  :: rmu
   include 'avh_olo_complex.h90'
     :: cp2,cp3,cp4,cp12,cp23,cm3,cm4,sm3,sm4,sm1,sm2 &
             ,r13,r14,r23,r24,r34,d34,cc,logd,li2d,loge,li2f,li2b,li2e
   type(qmplx_type) :: q13,q14,q23,q24,q34,qy1,qy2
   include 'avh_olo_real.h90'
     :: h1,h2
![CALLINGME  write(*,*) 'MESSAGE from OneLOop box13: you are calling me'
!
   if (p12.eq.m3) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box13: ' &
       ,'p12=m3, returning 0'
     rslt(0)=0 ;rslt(1)=0 ;rslt(2)=0
     return
   endif
   if (p23.eq.m4) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box13: ' &
       ,'p23=m4, returning 0'
     rslt(0)=0 ;rslt(1)=0 ;rslt(2)=0
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
   sm1 = abs(rmu)
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
   li2d = li2c2(qy1,qonv(1))/(r13*r24)
   loge = logc(q13)
!
   qy1 = q23/q24
   qy2 = q13/q14
   li2f = li2c2( qy1*q34,qy2*q34 )*r34/(r14*r24)
   li2b = li2c2( qy1/q34,qy2/q34 )/(r34*r14*r24)
   li2e = li2c2( q14/q24,q13/q23 )/(r23*r24)
!
   rslt(2) = 0
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
   include 'avh_olo_complex.h90'
     ,intent(out) :: rslt(0:2)
   include 'avh_olo_complex.h90'
     ,intent(in)  :: cp3,cp4,cp12,cp23,cm3,cm4
   include 'avh_olo_real.h90'
     ,intent(in)  :: rmu
   include 'avh_olo_complex.h90'
     :: sm3,sm4,sm1,sm2,r13,r14,r24,r34,d34,cc &
                     ,log13,log14,log24,log34,li2f,li2b,li2d
   type(qmplx_type) :: q13,q14,q24,q34,qyy
![CALLINGME  write(*,*) 'MESSAGE from OneLOop box12: you are calling me'
!
   if (cp12.eq.cm3) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box12: ' &
       ,'p12=m3, returning 0'
     rslt(0)=0 ;rslt(1)=0 ;rslt(2)=0
     return
   endif
   if (cp23.eq.cm4) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box12: ' &
       ,'p23=m4, returning 0'
     rslt(0)=0 ;rslt(1)=0 ;rslt(2)=0
     return
   endif
!
   sm3 = mysqrt(cm3)
   sm4 = mysqrt(cm4)
   sm1 = abs(rmu)
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
   rslt(2) = 1
   rslt(2) = rslt(2)/2
   rslt(1) = log14 - log24 - log13
   rslt(0) = 2*log13*log24 - log14*log14 - log34*log34 &
           - 2*li2d - li2f - li2b - 3*PISQo24
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
   include 'avh_olo_complex.h90'
     ,intent(out) :: rslt(0:2)
   include 'avh_olo_complex.h90'
     ,intent(in)  :: cp3,cp12,cp23,cm3,cm4
   include 'avh_olo_real.h90'
     ,intent(in)  :: rmu
   include 'avh_olo_complex.h90'
     :: sm3,sm4,sm1,sm2,r13,r24,r34,d34 &
                     ,cc,log13,log24,log34
![CALLINGME  write(*,*) 'MESSAGE from OneLOop box11: you are calling me'
!
   if (cp12.eq.cm3) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box11: ' &
       ,'p12=m3, returning 0'
     rslt(0)=0 ;rslt(1)=0 ;rslt(2)=0
     return
   endif
   if (cp23.eq.cm4) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box11: ' &
       ,'p23=m4, returning 0'
     rslt(0)=0 ;rslt(1)=0 ;rslt(2)=0
     return
   endif
!
   sm3 = mysqrt(cm3)
   sm4 = mysqrt(cm4)
   sm1 = abs(rmu)
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
   rslt(2) = 1
   rslt(1) = -log13-log24
   rslt(0) = 2*log13*log24 - log34*log34 - 14*PISQo24
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
   include 'avh_olo_complex.h90'
     ,intent(out) :: rslt(0:2)
   include 'avh_olo_complex.h90'
     ,intent(in)  :: p2,p3,p4,p12,p23,m4
   include 'avh_olo_real.h90'
     ,intent(in)  :: rmu
   include 'avh_olo_complex.h90'
     :: cp2,cp3,cp4,cp12,cp23,cm4,r13,r14,r23,r24,r34,z1,z0
   type(qmplx_type) :: q13,q14,q23,q24,q34,qm4,qxx,qx1,qx2
   include 'avh_olo_real.h90'
     :: h1,h2
![CALLINGME  write(*,*) 'MESSAGE from OneLOop box10: you are calling me'
!
   if (p12.eq.CZRO) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box10: ' &
       ,'p12=0, returning 0'
     rslt(0)=0 ;rslt(1)=0 ;rslt(2)=0
     return
   endif
   if (p23.eq.m4) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box10: ' &
       ,'p23=mm, returning 0'
     rslt(0)=0 ;rslt(1)=0 ;rslt(2)=0
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
   if (r34.ne.CZRO) then
     qx1 = q34/qm4
     qx2 = qx1*q14/q13
     qx1 = qx1*q24/q23
     z0 = -li2c2(qx1,qx2)*r34/(2*cm4*r23)
   else
     z0 = 0
   endif
!
   qx1 = q23/q13
   qx2 = q24/q14
   qxx = qx1/qx2
   z1 = -logc2(qxx)/r24
   z0 = z0 - li2c2(qx1,qx2)/r14
   z0 = z0 + li2c2(qxx,qonv(1))/r24
   z0 = z0 + z1*( logc(qm4/q24) - logc(qm4/(rmu*rmu))/2 )
!
   rslt(2) = 0
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
   include 'avh_olo_complex.h90'
     ,intent(out) :: rslt(0:2)
   include 'avh_olo_complex.h90'
     ,intent(in)  :: cp2,cp3,cp12,cp23,cm4
   include 'avh_olo_real.h90'
     ,intent(in)  :: rmu
   include 'avh_olo_complex.h90'
     :: logm,log12,log23,li12,li23,z2,z1,z0,cc &
                     ,r13,r23,r24,r34
   type(qmplx_type) :: q13,q23,q24,q34,qm4,qxx
![CALLINGME  write(*,*) 'MESSAGE from OneLOop box09: you are calling me'
!
   if (cp12.eq.CZRO) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box09: ' &
       ,'p12=0, returning 0'
     rslt(0)=0 ;rslt(1)=0 ;rslt(2)=0
     return
   endif
   if (cp23.eq.cm4) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box09: ' &
       ,'p23=mm, returning 0'
     rslt(0)=0 ;rslt(1)=0 ;rslt(2)=0
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
   z2 = 1
   z2 = z2/2
   z1 = -log12 - log23
   z0 = li23 + 2*li12 + z1*z1 + PISQo24
   cc = 1/(r13*r24)
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
   include 'avh_olo_complex.h90'
     ,intent(out) :: rslt(0:2)
   include 'avh_olo_complex.h90'
     ,intent(in)  :: cp3,cp4,cp12,cp23,cm4
   include 'avh_olo_real.h90'
     ,intent(in)  :: rmu
   type(qmplx_type) :: q13,q14,q24,q34,qm4,qxx,qx1,qx2,qx3
   include 'avh_olo_complex.h90'
     :: r13,r14,r24,r34,z1,z0,cc
   include 'avh_olo_real.h90'
     :: rmu2
![CALLINGME  write(*,*) 'MESSAGE from OneLOop box08: you are calling me'
!
   if (cp12.eq.CZRO) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box08: ' &
       ,'p12=0, returning 0'
     rslt(0)=0 ;rslt(1)=0 ;rslt(2)=0
     return
   endif
   if (cp23.eq.cm4) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box08: ' &
       ,'p23=mm, returning 0'
     rslt(0)=0 ;rslt(1)=0 ;rslt(2)=0
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
   cc = 1/(r13*r24)
   rslt(2) = cc
   rslt(1) = cc*z1
   rslt(0) = cc*( z0 - 6*PISQo24 )
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
   include 'avh_olo_complex.h90'
     ,intent(out) :: rslt(0:2)
   include 'avh_olo_complex.h90'
     ,intent(in)  :: cp4,cp12,cp23,cm4
   include 'avh_olo_real.h90'
     ,intent(in)  :: rmu
   type(qmplx_type) :: q13,q14,q24,qm4
   include 'avh_olo_complex.h90'
     :: r13,r14,r24,logm,log12,log23,log4,li423 &
                     ,z2,z1,z0,cc
![CALLINGME  write(*,*) 'MESSAGE from OneLOop box07: you are calling me'
!
   if (cp12.eq.CZRO) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box07: ' &
       ,'p12=0, returning 0'
     rslt(0)=0 ;rslt(1)=0 ;rslt(2)=0
     return
   endif
   if (cp23.eq.cm4) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box07: ' &
       ,'p23=mm, returning 0'
     rslt(0)=0 ;rslt(1)=0 ;rslt(2)=0
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
   z2 = 3
   z2 = z2/2
   z1 = -2*log23 - log12 + log4
   z0 = 2*(log12*log23 - li423) - log4*log4 - 13*PISQo24
   cc = 1/(r13*r24)
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
   include 'avh_olo_complex.h90'
     ,intent(out) :: rslt(0:2)
   include 'avh_olo_complex.h90'
     ,intent(in)  :: cp12,cp23,cm4
   include 'avh_olo_real.h90'
     ,intent(in)  :: rmu
   type(qmplx_type) :: q13,q24,qm4
   include 'avh_olo_complex.h90'
     :: r13,r24,logm,log1,log2,z2,z1,z0,cc
![CALLINGME  write(*,*) 'MESSAGE from OneLOop box06: you are calling me'
!
   if (cp12.eq.CZRO) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box06: ' &
       ,'p12=0, returning 0'
     rslt(0)=0 ;rslt(1)=0 ;rslt(2)=0
     return
   endif
   if (cp23.eq.cm4) then
     if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop box06: ' &
       ,'p23=mm, returning 0'
     rslt(0)=0 ;rslt(1)=0 ;rslt(2)=0
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
   z2 = 2
   z1 = -2*log2 - log1
   z0 = 2*(log2*log1 - 8*PISQo24)
   cc = 1/(r13*r24)
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
   include 'avh_olo_complex.h90'
     ,intent(out) :: rslt(0:2)
   include 'avh_olo_complex.h90'
     ,intent(in)  :: p2,p4,p5,p6 
   include 'avh_olo_real.h90'
     ,intent(in)  :: rmu
   type(qmplx_type) :: q2,q4,q5,q6,q26,q54,qy
   include 'avh_olo_complex.h90'
     :: logy
   include 'avh_olo_real.h90'
     :: rmu2
![CALLINGME  write(*,*) 'MESSAGE from OneLOop box03: you are calling me'
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
           - li2c2(qonv(1),qy)/(p5*p6) &
           - logy*logc(q54*q2*q6/(rmu2*rmu2))/2
   rslt(2) = 0
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
   include 'avh_olo_complex.h90'
     ,intent(out) :: rslt(0:2)
   include 'avh_olo_complex.h90'
     ,intent(in)  :: p2,p3,p4,p5,p6
   include 'avh_olo_real.h90'
     ,intent(in)  :: rmu
   type(qmplx_type) ::q2,q3,q4,q5,q6 ,q25,q64,qy,qz
   include 'avh_olo_complex.h90'
     :: logy
   include 'avh_olo_real.h90'
     :: rmu2
![CALLINGME  write(*,*) 'MESSAGE from OneLOop box05: you are calling me'
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
   rslt(2) = 0
   rslt(1) = logy
   rslt(0) = li2c2(q64,q25)/(p4*p5) &
           - li2c2(qonv(1),qy)/(p5*p6) &
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
   use avh_olo_olog
   use avh_olo_dilog
   include 'avh_olo_complex.h90'
     ,intent(out) :: rslt(0:2)
   include 'avh_olo_complex.h90'
     ,intent(in)  :: cp(6)
   include 'avh_olo_real.h90'
     ,intent(in)  :: api(6),rmu
   include 'avh_olo_complex.h90'
     :: log3,log4,log5,log6,li24,li25,li26 &
                     ,li254,li263
   include 'avh_olo_real.h90'
     :: rp1,rp2,rp3,rp4,rp5,rp6,pp(6),ap(6),gg,ff,hh,arg,rmu2
   integer :: icase,sf,sgn,i3,i4,i5,i6
   integer ,parameter :: base(4)=(/8,4,2,1/)
!
   rmu2 = rmu*rmu
   ff = api(5)*api(6)
   gg = api(2)*api(4)
   hh = api(1)*api(3)
   if     (ff.ge.gg.and.ff.ge.hh) then
     pp(1)=areal(cp(1)) ;ap(1)=api(1)
     pp(2)=areal(cp(2)) ;ap(2)=api(2)
     pp(3)=areal(cp(3)) ;ap(3)=api(3)
     pp(4)=areal(cp(4)) ;ap(4)=api(4)
     pp(5)=areal(cp(5)) ;ap(5)=api(5)
     pp(6)=areal(cp(6)) ;ap(6)=api(6)
   elseif (gg.ge.ff.and.gg.ge.hh) then
     pp(1)=areal(cp(1)) ;ap(1)=api(1)
     pp(2)=areal(cp(6)) ;ap(2)=api(6)
     pp(3)=areal(cp(3)) ;ap(3)=api(3)
     pp(4)=areal(cp(5)) ;ap(4)=api(5)
     pp(5)=areal(cp(4)) ;ap(5)=api(4)
     pp(6)=areal(cp(2)) ;ap(6)=api(2)
   else
     pp(1)=areal(cp(5)) ;ap(1)=api(5)
     pp(2)=areal(cp(2)) ;ap(2)=api(2)
     pp(3)=areal(cp(6)) ;ap(3)=api(6)
     pp(4)=areal(cp(4)) ;ap(4)=api(4)
     pp(5)=areal(cp(1)) ;ap(5)=api(1)
     pp(6)=areal(cp(3)) ;ap(6)=api(3)
   endif
!
   icase = 0
   if (ap(1).gt.RZRO) icase = icase + base(1)
   if (ap(2).gt.RZRO) icase = icase + base(2)
   if (ap(3).gt.RZRO) icase = icase + base(3)
   if (ap(4).gt.RZRO) icase = icase + base(4)
   rp1 = pp(permtable(1,icase))
   rp2 = pp(permtable(2,icase))
   rp3 = pp(permtable(3,icase))
   rp4 = pp(permtable(4,icase))
   rp5 = pp(permtable(5,icase))
   rp6 = pp(permtable(6,icase))
   icase = casetable(   icase)
!
   i3=0 ;if (-rp3.lt.RZRO) i3=-1
   i4=0 ;if (-rp4.lt.RZRO) i4=-1
   i5=0 ;if (-rp5.lt.RZRO) i5=-1
   i6=0 ;if (-rp6.lt.RZRO) i6=-1
!
   if     (icase.eq.0) then
! 0 masses non-zero
![CALLINGME  write(*,*) 'MESSAGE from OneLOop box00 0: you are calling me'
     gg = 1/( rp5 * rp6 )
     log5 = olog(abs(rp5/rmu2),i5)
     log6 = olog(abs(rp6/rmu2),i6)
     rslt(2) = gg*( 4 )
     rslt(1) = gg*(-2*(log5 + log6) )
     rslt(0) = gg*( log5**2 + log6**2 - olog(abs(rp5/rp6),i5-i6)**2 - 32*PISQo24 )
   elseif (icase.eq.1) then
! 1 mass non-zero
![CALLINGME  write(*,*) 'MESSAGE from OneLOop box00 1: you are calling me'
     gg = 1/( rp5 * rp6 )
     ff =  gg*( rp5 + rp6 - rp4 )
     log4 = olog(abs(rp4/rmu2),i4)
     log5 = olog(abs(rp5/rmu2),i5)
     log6 = olog(abs(rp6/rmu2),i6)
     sf = sgnRe(ff)
     sgn = 0
       arg = rp4*ff 
       if (arg.lt.RZRO) sgn = sf
       li24 = dilog(abs(arg),sgn)
     sgn = 0
       arg = rp5*ff 
       if (arg.lt.RZRO) sgn = sf
       li25 = dilog(abs(arg),sgn)
     sgn = 0
       arg = rp6*ff 
       if (arg.lt.RZRO) sgn = sf
       li26 = dilog(abs(arg),sgn)
     rslt(2) = gg*( 2 )
     rslt(1) = gg*( 2*(log4-log5-log6) )
     rslt(0) = gg*( log5**2 + log6**2 - log4**2 - 12*PISQo24 &
                   + 2*(li25 + li26 - li24) )
   elseif (icase.eq.2) then
! 2 neighbour masses non-zero
![CALLINGME  write(*,*) 'MESSAGE from OneLOop box00 2: you are calling me'
     gg = 1/( rp5 * rp6 )
     ff =  gg*( rp5 + rp6 - rp4 )
     log3 = olog(abs(rp3/rmu2),i3)
     log4 = olog(abs(rp4/rmu2),i4)
     log5 = olog(abs(rp5/rmu2),i5)
     log6 = olog(abs(rp6/rmu2),i6)
     li254 = dilog( abs(rp4/rp5) ,i4-i5 )
     li263 = dilog( abs(rp3/rp6) ,i3-i6 )
     sf = sgnRe(ff)
     sgn = 0
       arg = rp4*ff 
       if (arg.lt.RZRO) sgn = sf
       li24 = dilog(abs(arg),sgn)
     sgn = 0
       arg = rp5*ff 
       if (arg.lt.RZRO) sgn = sf
       li25 = dilog(abs(arg),sgn)
     sgn = 0
       arg = rp6*ff 
       if (arg.lt.RZRO) sgn = sf
       li26 = dilog(abs(arg),sgn)
     rslt(2) = gg
     rslt(1) = gg*( log4 + log3 - log5 - 2*log6 )
     rslt(0) = gg*( log5**2 + log6**2 - log3**2 - log4**2 &
                   + (log3 + log4 - log5)**2/2 &
                   - 2*PISQo24 + 2*(li254 - li263 + li25 + li26 - li24) )
   elseif (icase.eq.5) then
! 2 opposite masses non-zero
     call box03( rslt ,acmplx(rp2),acmplx(rp4) &
                      ,acmplx(rp5),acmplx(rp6) ,rmu )
   elseif (icase.eq.3) then
! 3 masses non-zero
     call box05( rslt ,acmplx(rp2),acmplx(rp3) &
                      ,acmplx(rp4),acmplx(rp5) &
                      ,acmplx(rp6) ,rmu )
   elseif (icase.eq.4) then
! 4 masses non-zero
     call boxf0( rslt ,acmplx(rp1),acmplx(rp2) &
                      ,acmplx(rp3),acmplx(rp4) &
                      ,acmplx(rp5),acmplx(rp6) )
   endif
   end subroutine

  
  subroutine boxf0( rslt ,p1,p2,p3,p4,p12,p23 )
!*******************************************************************
! Finite 1-loop scalar 4-point function with all internal masses
! equal zero. Based on the formulas from
! A. Denner, U. Nierste, R. Scharf, Nucl.Phys.B367(1991)637-656
!*******************************************************************
  include 'avh_olo_complex.h90'
    ,intent(out) :: rslt(0:2) 
  include 'avh_olo_complex.h90'
    ,intent(in) :: p1,p2,p3,p4,p12,p23
  type(qmplx_type) :: q12,q13,q14,q23,q24,q34,qx1,qx2,qss
  include 'avh_olo_complex.h90'
    :: aa,bb,cc,dd,x1,x2,ss,r12,r13,r14,r23,r24,r34
  include 'avh_olo_real.h90'
    :: hh
![CALLINGME  write(*,*) 'MESSAGE from OneLOop boxf0: you are calling me'
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
  if (r13.eq.CZRO.or.aa.eq.CZRO) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop boxf0: ' &
       ,'threshold singularity, returning 0'
    rslt(0)=0 ;rslt(1)=0 ;rslt(2)=0
    return
  endif
!
  bb = r13*r24 + r12*r34 - r14*r23
  cc = r12*r13
  hh = areal(r23)
  dd = mysqrt( bb*bb - 4*aa*cc , -areal(aa)*hh )
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
  rslt(0)=0 ;rslt(1)=0 ;rslt(2)=0
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
  include 'avh_olo_complex.h90'
    ,intent(out) :: rslt(0:2) 
  include 'avh_olo_complex.h90'
    ,intent(in) :: p1,p2,p3,p4,p12,p23 ,m4
  type(qmplx_type) :: qx1,qx2,qss,q12,q13,q14,q23,q24,q34
  include 'avh_olo_complex.h90'
    :: smm,sm4,aa,bb,cc,dd,x1,x2,r12,r13,r14,r23,r24,r34
  logical :: r12zero,r13zero,r14zero
![CALLINGME  write(*,*) 'MESSAGE from OneLOop boxf1: you are calling me'
!
  sm4 = mysqrt(m4)
  smm = abs(sm4) 
!
  r12 = ( m4-p4 -p4 *IEPS )/(smm*sm4)
  r13 = ( m4-p23-p23*IEPS )/(smm*sm4)
  r14 = ( m4-p3 -p3 *IEPS )/(smm*sm4)
  r23 = (   -p1 -p1 *IEPS )/(smm*smm)
  r24 = (   -p12-p12*IEPS )/(smm*smm)
  r34 = (   -p2 -p2 *IEPS )/(smm*smm)
!
  r12zero=(abs(areal(r12))+abs(aimag(r12)).lt.neglig(prcpar))
  r13zero=(abs(areal(r13))+abs(aimag(r13)).lt.neglig(prcpar))
  r14zero=(abs(areal(r14))+abs(aimag(r14)).lt.neglig(prcpar))
!
  aa = r34*r24
!
  if (aa.eq.CZRO) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop boxf1: ' &
       ,'threshold singularity, returning 0'
    rslt(0)=0 ;rslt(1)=0 ;rslt(2)=0
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
  rslt(0)=0 ;rslt(1)=0 ;rslt(2)=0
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
  if (.not.r14zero) then
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
  include 'avh_olo_complex.h90'
    ,intent(out) :: rslt(0:2) 
  include 'avh_olo_complex.h90'
    ,intent(in) :: p1,p2,p3,p4,p12,p23,m2,m4
  call boxf2( rslt ,p12,p2,p23,p4,p1,p3 ,m2,m4 )
  end subroutine


  subroutine boxf2( rslt ,p1,p2,p3,p4,p12,p23 ,m3,m4 )
!*******************************************************************
! Finite 1-loop scalar 4-point function with two adjacent internal
! masses non-zero. Based on the formulas from
! A. Denner, U. Nierste, R. Scharf, Nucl.Phys.B367(1991)637-656
!*******************************************************************
  include 'avh_olo_complex.h90'
    ,intent(out) :: rslt(0:2) 
  include 'avh_olo_complex.h90'
    ,intent(in) :: p1,p2,p3,p4,p12,p23,m3,m4
  type(qmplx_type) :: qx1,qx2,qss,q12,q13,q14,q23,q24,q34
  include 'avh_olo_complex.h90'
    :: smm,sm3,sm4,aa,bb,cc,dd,x1,x2 &
                    ,r12,r13,r14,r23,r24,r34,d14,k14
  logical :: r12zero,r13zero,r24zero,r34zero
![CALLINGME  write(*,*) 'MESSAGE from OneLOop boxf2: you are calling me'
!
  sm3 = mysqrt(m3)
  sm4 = mysqrt(m4)
!
  smm = abs(sm3)
!
  r12 = (    m4-p4 -p4 *IEPS )/(smm*sm4)
  r13 = (    m4-p23-p23*IEPS )/(smm*sm4)
  k14 = ( m3+m4-p3 -p3 *IEPS )/(sm3*sm4)
  r23 = (      -p1 -p1 *IEPS )/(smm*smm)
  r24 = (    m3-p12-p12*IEPS )/(smm*sm3)
  r34 = (    m3-p2 -p2 *IEPS )/(smm*sm3)
!
  r12zero = (abs(areal(r12))+abs(aimag(r12)).lt.neglig(prcpar))
  r13zero = (abs(areal(r13))+abs(aimag(r13)).lt.neglig(prcpar))
  r24zero = (abs(areal(r24))+abs(aimag(r24)).lt.neglig(prcpar))
  r34zero = (abs(areal(r34))+abs(aimag(r34)).lt.neglig(prcpar))
!
  if (r12zero.and.r24zero) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop boxf2: ' &
       ,'m4=p4 and m3=p12, returning 0'
    rslt(0)=0 ;rslt(1)=0 ;rslt(2)=0
    return
  endif
  if (r13zero.and.r34zero) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop boxf2: ' &
       ,'m4=p23 and m3=p2, returning 0'
    rslt(0)=0 ;rslt(1)=0 ;rslt(2)=0
    return
  endif
!
  call rfun( r14,d14 ,k14 )
!
  aa = r34*r24 - r23
!
  if (aa.eq.CZRO) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop boxf2: ' &
       ,'threshold singularity, returning 0'
    rslt(0)=0 ;rslt(1)=0 ;rslt(2)=0
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
  rslt(0)=0 ;rslt(1)=0 ;rslt(2)=0
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
  include 'avh_olo_complex.h90'
    ,intent(out) :: rslt(0:2) 
  include 'avh_olo_complex.h90'
    ,intent(in) :: pp(6),mm(4)
  integer :: j
  integer ,parameter :: ip(6)=(/4,5,2,6,3,1/)
  integer ,parameter :: im(4)=(/4,1,3,2/)
  integer ,parameter :: ic(4,6)=reshape((/1,2,3,4 ,2,3,4,1 ,3,4,1,2 &
                                  ,4,1,2,3 ,5,6,5,6 ,6,5,6,5/),(/4,6/))
!
  if     (mm(1).eq.CZRO) then ;j=3
  elseif (mm(2).eq.CZRO) then ;j=4
  elseif (mm(3).eq.CZRO) then ;j=1
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
  include 'avh_olo_complex.h90'
    ,intent(out) :: rslt(0:2) 
  include 'avh_olo_complex.h90'
    ,intent(in) :: p1,p2,p3,p4,p12,p23,m1,m2,m4
  type(qmplx_type) :: qx1,qx2,qss,q12,q13,q14,q23,q24,q34,qy1,qy2
  include 'avh_olo_complex.h90'
    :: sm1,sm2,sm3,sm4 ,aa,bb,cc,dd,x1,x2 &
                    ,r12,r13,r14,r23,r24,r34,d12,d14,d24,k12,k14,k24
  logical ::r13zero,r23zero,r34zero
![CALLINGME  write(*,*) 'MESSAGE from OneLOop boxf33: you are calling me'
!
  sm1 = mysqrt(m1)
  sm2 = mysqrt(m2)
  sm4 = mysqrt(m4)
  sm3 = abs(sm2)
!
  k12 = ( m1+m2-p1 -p1 *IEPS )/(sm1*sm2) ! p1
  r13 = ( m1   -p12-p12*IEPS )/(sm1*sm3) ! p1+p2
  k14 = ( m1+m4-p4 -p4 *IEPS )/(sm1*sm4) ! p1+p2+p3
  r23 = ( m2   -p2 -p2 *IEPS )/(sm2*sm3) ! p2
  k24 = ( m2+m4-p23-p23*IEPS )/(sm2*sm4) ! p2+p3
  r34 = (    m4-p3 -p3 *IEPS )/(sm3*sm4) ! p3
!
  r13zero = (abs(areal(r13))+abs(aimag(r13)).lt.neglig(prcpar))
  r23zero = (abs(areal(r23))+abs(aimag(r23)).lt.neglig(prcpar))
  r34zero = (abs(areal(r34))+abs(aimag(r34)).lt.neglig(prcpar))
!
  if (r13zero) then
    if     (r23zero) then
      if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop boxf33: ' &
       ,'m4=p4 and m3=p12, returning 0'
      rslt(0)=0 ;rslt(1)=0 ;rslt(2)=0
      return
    elseif (r34zero) then
      if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop boxf33: ' &
       ,'m2=p1 and m3=p12, returning 0'
      rslt(0)=0 ;rslt(1)=0 ;rslt(2)=0
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
  if (aa.eq.CZRO) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop boxf33: ' &
       ,'threshold singularity, returning 0'
    rslt(0)=0 ;rslt(1)=0 ;rslt(2)=0
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
  rslt(0)=0 ;rslt(1)=0 ;rslt(2)=0
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
  include 'avh_olo_complex.h90'
    ,intent(out) :: rslt(0:2) 
  include 'avh_olo_complex.h90'
    ,intent(in) :: p1,p2,p3,p4,p12,p23,m1,m2,m3,m4
  type(qmplx_type) :: q12,q13,q14,q23,q24,q34,qx1,qx2,qy1,qy2,qtt
  include 'avh_olo_complex.h90'
    :: sm1,sm2,sm3,sm4 ,aa,bb,cc,dd,x1,x2,tt &
                    ,k12,k13,k14,k23,k24,k34 &
                    ,r12,r13,r14,r23,r24,r34 &
                    ,d12,d13,d14,d23,d24,d34
  include 'avh_olo_real.h90'
    :: h1,h2
![CALLINGME  write(*,*) 'MESSAGE from OneLOop boxf4: you are calling me'
!
  sm1 = mysqrt(m1)
  sm2 = mysqrt(m2)
  sm3 = mysqrt(m3)
  sm4 = mysqrt(m4)
!
  k12 = ( m1+m2-p1 -p1 *IEPS)/(sm1*sm2) ! p1
  k13 = ( m1+m3-p12-p12*IEPS)/(sm1*sm3) ! p1+p2
  k14 = ( m1+m4-p4 -p4 *IEPS)/(sm1*sm4) ! p1+p2+p3
  k23 = ( m2+m3-p2 -p2 *IEPS)/(sm2*sm3) ! p2
  k24 = ( m2+m4-p23-p23*IEPS)/(sm2*sm4) ! p2+p3
  k34 = ( m3+m4-p3 -p3 *IEPS)/(sm3*sm4) ! p3
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
  if (aa.eq.CZRO) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop boxf4: ' &
       ,'threshold singularity, returning 0'
    rslt(0)=0 ;rslt(1)=0 ;rslt(2)=0
    return
  endif
!
  bb = d13*d24 + k12*k34 - k14*k23
  cc = k12/r13 + r24*k34 - k14*r24/r13 - k23
  call solabc(x1,x2,dd ,aa,bb,cc ,0)
!
  h1 = areal(k23 - r13*k12 - r24*k34 + r13*r24*k14)
  h2 = h1*areal(aa)*areal(x1)
  h1 = h1*areal(aa)*areal(x2)
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
  rslt(0)=0 ;rslt(1)=0 ;rslt(2)=0
!
  qy1 = qx1/q24
  qy2 = qx2/q24
  rslt(0) = rslt(0) + ( li2c2( qy1*q12 ,qy2*q12 )*r12 &
                      + li2c2( qy1/q12 ,qy2/q12 )/r12 )/r24
  tt = r13/r24
  qtt = qonv(tt,-areal(r24) )
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
