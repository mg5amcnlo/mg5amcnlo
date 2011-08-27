!
! scaloop= 1 -> looptools 1-loop scalar functions (not implemented yet)
! scaloop= 2 -> avh       1-loop scalar functions (massive with complex masses)
! scaloop= 3 -> qcdloop   1-loop scalar functions (Ellis and Zanderighi)
!
 module loopfunctions
  use denominators
  use scale
  use dimensions
  implicit none
!
! variables for the 1-point sector:
!
  include 'cts_dpc.h'
   , dimension(:,:), public, allocatable :: aloopfun
!
! variables for the 2-point sector:
!
  include 'cts_dpc.h'
   , dimension(:,:), public, allocatable :: bloopfun
  include 'cts_dpc.h'
   , dimension(:,:), public, allocatable :: b1loopfun
  include 'cts_dpc.h'
   , dimension(:,:), public, allocatable :: b11loopfun
!
! variables for the 3-point sector:
!
  include 'cts_dpc.h'
   , dimension(:,:), public, allocatable :: cloopfun
!
! variables for the 4-point sector:
!
  include 'cts_dpc.h'
   , dimension(:,:), public, allocatable :: dloopfun
  save aloopfun,bloopfun,b1loopfun,b11loopfun,cloopfun,dloopfun
!
  contains
!
  subroutine allocate_loopfun(nprop)
   integer, intent(in) :: nprop
   if (nprop.ge.1) call allocate_loopfuna
   if (nprop.ge.2) call allocate_loopfunb
   if (nprop.ge.3) call allocate_loopfunc
   if (nprop.ge.4) call allocate_loopfund
  end subroutine allocate_loopfun
!
  subroutine allocate_loopfuna
   integer :: nprop
   nprop= dmns
   allocate (aloopfun(0:2,2**(nprop-1)), stat=ierr)
   aloopfun= 0.d0
  end subroutine allocate_loopfuna
!
  subroutine allocate_loopfunb
   integer :: nprop
   nprop= dmns
   allocate (  bloopfun(0:2,2**(nprop-1)+2**(nprop-2)), stat=ierr)
   allocate ( b1loopfun(0:2,2**(nprop-1)+2**(nprop-2)), stat=ierr)
   allocate (b11loopfun(0:2,2**(nprop-1)+2**(nprop-2)), stat=ierr)
     bloopfun= 0.d0 
    b1loopfun= 0.d0
   b11loopfun= 0.d0
  end subroutine allocate_loopfunb
!
  subroutine allocate_loopfunc
   integer :: nprop
   nprop= dmns
   allocate (cloopfun(0:2,2**(nprop-1)+2**(nprop-2)+2**(nprop-3)), stat=ierr)
   cloopfun= 0.d0
  end subroutine allocate_loopfunc
!
  subroutine allocate_loopfund
   integer :: nprop
   nprop= dmns
   allocate (dloopfun(0:2,2**(nprop-1)+2**(nprop-2)+2**(nprop-3) &
                                      +2**(nprop-4)), stat=ierr)
   dloopfun= 0.d0
  end subroutine allocate_loopfund
!
  subroutine getloop(nprop)
   integer, intent(in) :: nprop
   if (nprop.ge.1) call getaloop(nprop)
   if (nprop.ge.2) call getbloop(nprop)
   if (nprop.ge.3) call getcloop(nprop)
   if (nprop.ge.4) call getdloop(nprop)
  end subroutine getloop
!
  subroutine getaloop(number_propagators)
!
!  A 1-loop scalar function
!
   integer, intent(in) :: number_propagators
   integer :: np
   integer :: i,ib
   include 'cts_dpc.h'
    , dimension(0:2) :: value
   include 'cts_dpr.h'    
    :: m12
   include 'cts_dpc.h' 
    :: cm12
   include 'cts_dpc.h' 
    :: a0,qlI1
   np= number_propagators
   do i= 1,nbn1(np)
    ib= mbn1(np,i)
    if     (scaloop.eq.1) then
!     aloopfun(0,ib)= a0(m12)
     stop 'value of scaloop not implemented'
    elseif (scaloop.eq.2) then
     cm12= den(bn1(np,1,i))%m2
     call avh_olo_a0c(value,cm12)
     aloopfun(:,ib)= value(:) 
    elseif (scaloop.eq.3) then
     m12= dreal(den(bn1(np,1,i))%m2)
     aloopfun(2,ib)= qlI1(m12,musq,-2)     
     aloopfun(1,ib)= qlI1(m12,musq,-1)     
     aloopfun(0,ib)= qlI1(m12,musq,0)     
    else
     stop 'value of scaloop not implemented'
    endif
   enddo
  end subroutine getaloop
!
  subroutine getbloop(number_propagators)
!
!  B,B1 and B11 1-loop scalar functions
!
   use tensor_operations  
   integer, intent(in) :: number_propagators
   integer :: np
   integer :: i,ib
   include 'cts_dpr.h' 
    :: k12
   include 'cts_dpc.h' 
    :: ck12
   include 'cts_dpr.h'
    , dimension(0:3) :: k1
   include 'cts_dpc.h'
    , dimension(0:2) :: value
   include 'cts_dpc.h'
    , dimension(0:2) :: valb11,valb00,valb1,valb0
   include 'cts_dpr.h'    
    :: m12,m22
   include 'cts_dpc.h' 
    :: cm12,cm22
   include 'cts_dpc.h' 
    :: b0,b1,b11,qlI2
   np= number_propagators
   do i= 1,nbn2(np)
    ib= mbn2(np,i)
    k1= den(bn2(np,2,i))%p-den(bn2(np,1,i))%p
    call contr(k1,k1,k12)
!!    if (dabs(k12/roots/roots).lt.1.d-8) k12= 0.d0
    if     (scaloop.eq.1) then
!     bloopfun(0,ib)  = b0(k12,m12,m22)
!     b1loopfun(0,ib) = b1(k12,m12,m22)
!     b11loopfun(0,ib)= b11(k12,m12,m22)
     stop 'value of scaloop not implemented'
    elseif (scaloop.eq.2) then
     cm12= den(bn2(np,1,i))%m2
     cm22= den(bn2(np,2,i))%m2
     ck12= k12
     call avh_olo_b11c(valb11,valb00,valb1,valb0,ck12,cm12,cm22)
     bloopfun(:,ib)  =  valb0(:)
     b1loopfun(:,ib) =  valb1(:)
     b11loopfun(:,ib)=  valb11(:)
    elseif (scaloop.eq.3) then
     m12= dreal(den(bn2(np,1,i))%m2)
     m22= dreal(den(bn2(np,2,i))%m2)
     bloopfun(2,ib)= qlI2(k12,m12,m22,musq,-2)     
     bloopfun(1,ib)= qlI2(k12,m12,m22,musq,-1)     
     bloopfun(0,ib)= qlI2(k12,m12,m22,musq,0)     
     call avh_olo_b11m(valb11,valb00,valb1,valb0,k12,m12,m22)
     b1loopfun(:,ib) =  valb1(:)
     b11loopfun(:,ib)=  valb11(:)
    else 
     stop 'value of scaloop not implemented'
    endif
   enddo
  end subroutine getbloop
!
  subroutine getcloop(number_propagators)
!
!  C 1-loop scalar function
!
   use tensor_operations  
   integer, intent(in) :: number_propagators
   integer :: np
   integer :: i,ib
   include 'cts_dpr.h' 
    :: k12,k22,k32
   include 'cts_dpc.h' 
    :: ck12,ck22,ck32
   include 'cts_dpr.h'
    , dimension(0:3) :: k1,k2,k3
   include 'cts_dpc.h'
    , dimension(0:2) :: value
   include 'cts_dpr.h'    
    :: m12,m22,m32
   include 'cts_dpc.h' 
    :: cm12,cm22,cm32
   include 'cts_dpc.h' 
    :: c0,qlI3
   np= number_propagators
   do i= 1,nbn3(np)
    ib= mbn3(np,i)
    k1= den(bn3(np,2,i))%p-den(bn3(np,1,i))%p
    k2= den(bn3(np,3,i))%p-den(bn3(np,2,i))%p
    k3= den(bn3(np,3,i))%p-den(bn3(np,1,i))%p
    call contr(k1,k1,k12)
    call contr(k2,k2,k22)
    call contr(k3,k3,k32)
!!    if (dabs(k12/roots/roots).lt.1.d-8) k12= 0.d0
!!    if (dabs(k22/roots/roots).lt.1.d-8) k22= 0.d0
!!    if (dabs(k32/roots/roots).lt.1.d-8) k32= 0.d0
    if     (scaloop.eq.1) then
!     cloopfun(0,ib)  =  c0(k12,k22,k32,m12,m22,m32) 
     stop 'value of scaloop not implemented'
    elseif (scaloop.eq.2) then
     cm12= den(bn3(np,1,i))%m2
     cm22= den(bn3(np,2,i))%m2
     cm32= den(bn3(np,3,i))%m2
     ck12= k12
     ck22= k22
     ck32= k32
     call avh_olo_c0c(value,ck12,ck22,ck32,cm12,cm22,cm32)
     cloopfun(:,ib)  =        value(:) 
    elseif (scaloop.eq.3) then
     m12= dreal(den(bn3(np,1,i))%m2)
     m22= dreal(den(bn3(np,2,i))%m2)
     m32= dreal(den(bn3(np,3,i))%m2)
     cloopfun(2,ib)= qlI3(k12,k22,k32,m12,m22,m32,musq,-2)     
     cloopfun(1,ib)= qlI3(k12,k22,k32,m12,m22,m32,musq,-1)     
     cloopfun(0,ib)= qlI3(k12,k22,k32,m12,m22,m32,musq,0)     
    else 
     stop 'value of scaloop not implemented'
    endif
   enddo
  end subroutine getcloop
!
  subroutine getdloop(number_propagators)
!
!  D 1-loop scalar function
!
   use tensor_operations  
   integer, intent(in) :: number_propagators
   integer :: np
   integer :: i,ib
   include 'cts_dpr.h' 
    :: k12,k22,k32,k42,k122,k232
   include 'cts_dpc.h' 
    :: ck12,ck22,ck32,ck42,ck122,ck232
   include 'cts_dpr.h'
    , dimension(0:3) :: k1,k2,k3,k4,p12,p23
   include 'cts_dpc.h'
    , dimension(0:2) :: value
   include 'cts_dpr.h'    
    :: m12,m22,m32,m42
   include 'cts_dpc.h' 
    :: cm12,cm22,cm32,cm42
   include 'cts_dpc.h' 
    :: d0,qlI4
   np= number_propagators
   do i= 1,nbn4(np)
    ib= mbn4(np,i)
    k1 = den(bn4(np,2,i))%p-den(bn4(np,1,i))%p
    k2 = den(bn4(np,3,i))%p-den(bn4(np,2,i))%p
    k3 = den(bn4(np,4,i))%p-den(bn4(np,3,i))%p
    k4 = den(bn4(np,4,i))%p-den(bn4(np,1,i))%p
    p12= den(bn4(np,3,i))%p-den(bn4(np,1,i))%p
    p23= den(bn4(np,4,i))%p-den(bn4(np,2,i))%p
    call contr(k1 ,k1 ,k12)
    call contr(k2 ,k2 ,k22)
    call contr(k3 ,k3 ,k32)
    call contr(k4 ,k4 ,k42)
    call contr(p12,p12,k122)
    call contr(p23,p23,k232)
!!    if (dabs(k12/roots/roots).lt.1.d-8)  k12 = 0.d0
!!    if (dabs(k22/roots/roots).lt.1.d-8)  k22 = 0.d0
!!    if (dabs(k32/roots/roots).lt.1.d-8)  k32 = 0.d0
!!    if (dabs(k42/roots/roots).lt.1.d-8)  k42 = 0.d0
!!    if (dabs(k122/roots/roots).lt.1.d-8) k122= 0.d0
!!    if (dabs(k232/roots/roots).lt.1.d-8) k232= 0.d0
    if     (scaloop.eq.1) then
!     dloopfun(0,ib)  = d0(k12,k22,k32,k42,k122,k232,m12,m22,m32,m42)
     stop 'value of scaloop not implemented'
    elseif (scaloop.eq.2) then
     cm12= den(bn4(np,1,i))%m2
     cm22= den(bn4(np,2,i))%m2
     cm32= den(bn4(np,3,i))%m2
     cm42= den(bn4(np,4,i))%m2
     ck12= k12
     ck22= k22
     ck32= k32
     ck42= k42
     ck122= k122
     ck232= k232
     call avh_olo_d0c(value,ck12,ck22,ck32,ck42,ck122,ck232,cm12,cm22,cm32,cm42)
     dloopfun(:,ib)  =        value(:) 
    elseif (scaloop.eq.3) then
     m12= dreal(den(bn4(np,1,i))%m2)
     m22= dreal(den(bn4(np,2,i))%m2)
     m32= dreal(den(bn4(np,3,i))%m2)
     m42= dreal(den(bn4(np,4,i))%m2)
     dloopfun(2,ib)= qlI4(k12,k22,k32,k42,k122,k232,m12,m22,m32,m42,musq,-2)   
     dloopfun(1,ib)= qlI4(k12,k22,k32,k42,k122,k232,m12,m22,m32,m42,musq,-1)   
     dloopfun(0,ib)= qlI4(k12,k22,k32,k42,k122,k232,m12,m22,m32,m42,musq,0)   
    else 
     stop 'value of scaloop not implemented'
    endif
   enddo
  end subroutine getdloop
!
 end module loopfunctions





