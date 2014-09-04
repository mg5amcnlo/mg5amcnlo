!
! scaloop= 1 -> looptools 1-loop scalar functions (not implemented yet)
! scaloop= 2 -> avh       1-loop scalar functions (massive with complex masses)
! scaloop= 3 -> qcdloop   1-loop scalar functions (Ellis and Zanderighi)
!
 module loopfunctions
  use denominators
  use scale
  use dimensions
  use avh_olo
  implicit none
  private
  public :: allocate_loopfun,getloop
  public :: aloopfun,bloopfun,b1loopfun,b11loopfun,cloopfun,dloopfun
  integer, private :: ierr
!
! variables for the 1-point sector:
!
  include 'cts_dpc.h'
   , dimension(:,:), allocatable :: aloopfun
!
! variables for the 2-point sector:
!
  include 'cts_dpc.h'
   , dimension(:,:), allocatable :: bloopfun
  include 'cts_dpc.h'
   , dimension(:,:), allocatable :: b1loopfun
  include 'cts_dpc.h'
   , dimension(:,:), allocatable :: b11loopfun
!
! variables for the 3-point sector:
!
  include 'cts_dpc.h'
   , dimension(:,:), allocatable :: cloopfun
!
! variables for the 4-point sector:
!
  include 'cts_dpc.h'
   , dimension(:,:), allocatable :: dloopfun
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
   ierr= -1 
   allocate (aloopfun(0:2,dmns_a), stat=ierr)
   if (ierr.ne.0) STOP "Allocation error in allocate_loopfuna"
   aloopfun= 0.d0
  end subroutine allocate_loopfuna
!
  subroutine allocate_loopfunb
   ierr= -1 
   allocate (  bloopfun(0:2,dmns_b), stat=ierr)
   allocate ( b1loopfun(0:2,dmns_b), stat=ierr)
   allocate (b11loopfun(0:2,dmns_b), stat=ierr)
   if (ierr.ne.0) STOP "Allocation error in allocate_loopfunb"
     bloopfun= 0.d0 
    b1loopfun= 0.d0
   b11loopfun= 0.d0
  end subroutine allocate_loopfunb
!
  subroutine allocate_loopfunc
   ierr= -1 
   allocate (cloopfun(0:2,dmns_c), stat=ierr)
   if (ierr.ne.0) STOP "Allocation error in allocate_loopfunc"
   cloopfun= 0.d0
  end subroutine allocate_loopfunc
!
  subroutine allocate_loopfund
   ierr= -1 
   allocate (dloopfun(0:2,dmns_d), stat=ierr)
   if (ierr.ne.0) STOP "Allocation error in allocate_loopfund"
   dloopfun= 0.d0
  end subroutine allocate_loopfund
!
  subroutine getloop(nprop)
   integer, intent(in) :: nprop
   if (nprop.ge.1) call getaloop
   if (nprop.ge.2) call getbloop
   if (nprop.ge.3) call getcloop
   if (nprop.ge.4) call getdloop
  end subroutine getloop
!
  subroutine getaloop
!
!  A 1-loop scalar function
!
   integer :: ib
   include 'cts_dpc.h'
    , dimension(0:2) :: value
   include 'cts_dpr.h'    
    :: m12
   include 'cts_dpc.h' 
    :: cm12
   include 'cts_dpc.h' 
    :: a0,qlI1
   do ib= 1,dmns_1
    if     (scaloop.eq.1) then
!     aloopfun(0,ib)= a0(m12)
     stop 'value of scaloop not implemented'
    elseif (scaloop.eq.2) then
     cm12= den(bbn1(1,ib))%m2
     call olo(value,cm12)
     aloopfun(:,ib)= value(:) 
    elseif (scaloop.eq.3) then
     m12= dreal(den(bbn1(1,ib))%m2)
     aloopfun(2,ib)= qlI1(m12,musq,-2)     
     aloopfun(1,ib)= qlI1(m12,musq,-1)     
     aloopfun(0,ib)= qlI1(m12,musq,0)     
    else
     stop 'value of scaloop not implemented'
    endif
   enddo
  end subroutine getaloop
!
  subroutine getbloop
!
!  B,B1 and B11 1-loop scalar functions
!
   use tensor_operations  
   integer :: ib
   include 'cts_dpr.h' 
    :: k12
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
   do ib= 1,dmns_2
    k1= den(bbn2(2,ib))%p-den(bbn2(1,ib))%p
    call contr(k1,k1,k12)
!!    if (dabs(k12/roots/roots).lt.1.d-8) k12= 0.d0
    if     (scaloop.eq.1) then
!     bloopfun(0,ib)  = b0(k12,m12,m22)
!     b1loopfun(0,ib) = b1(k12,m12,m22)
!     b11loopfun(0,ib)= b11(k12,m12,m22)
     stop 'value of scaloop not implemented'
    elseif (scaloop.eq.2) then
     cm12= den(bbn2(1,ib))%m2
     cm22= den(bbn2(2,ib))%m2
     call olo(valb11,valb00,valb1,valb0,k12,cm12,cm22)
     bloopfun(:,ib)  =  valb0(:)
     b1loopfun(:,ib) =  valb1(:)
     b11loopfun(:,ib)=  valb11(:)
    elseif (scaloop.eq.3) then
     m12= dreal(den(bbn2(1,ib))%m2)
     m22= dreal(den(bbn2(2,ib))%m2)
     bloopfun(2,ib)= qlI2(k12,m12,m22,musq,-2)     
     bloopfun(1,ib)= qlI2(k12,m12,m22,musq,-1)     
     bloopfun(0,ib)= qlI2(k12,m12,m22,musq,0)     
     call olo(valb11,valb00,valb1,valb0,k12,m12,m22)
     b1loopfun(:,ib) =  valb1(:)
     b11loopfun(:,ib)=  valb11(:)
    else 
     stop 'value of scaloop not implemented'
    endif
   enddo
  end subroutine getbloop
!
  subroutine getcloop
!
!  C 1-loop scalar function
!
   use tensor_operations  
   integer :: ib
   include 'cts_dpr.h' 
    :: k12,k22,k32
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
   do ib= 1,dmns_3
    k1= den(bbn3(2,ib))%p-den(bbn3(1,ib))%p
    k2= den(bbn3(3,ib))%p-den(bbn3(2,ib))%p
    k3= den(bbn3(3,ib))%p-den(bbn3(1,ib))%p
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
     cm12= den(bbn3(1,ib))%m2
     cm22= den(bbn3(2,ib))%m2
     cm32= den(bbn3(3,ib))%m2
     call olo(value,k12,k22,k32,cm12,cm22,cm32)
     cloopfun(:,ib)  =        value(:) 
    elseif (scaloop.eq.3) then
     m12= dreal(den(bbn3(1,ib))%m2)
     m22= dreal(den(bbn3(2,ib))%m2)
     m32= dreal(den(bbn3(3,ib))%m2)
     cloopfun(2,ib)= qlI3(k12,k22,k32,m12,m22,m32,musq,-2)     
     cloopfun(1,ib)= qlI3(k12,k22,k32,m12,m22,m32,musq,-1)     
     cloopfun(0,ib)= qlI3(k12,k22,k32,m12,m22,m32,musq,0)     
    else 
     stop 'value of scaloop not implemented'
    endif
   enddo
  end subroutine getcloop
!
  subroutine getdloop
!
!  D 1-loop scalar function
!
   use tensor_operations  
   integer :: ib
   include 'cts_dpr.h' 
    :: k12,k22,k32,k42,k122,k232
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
   do ib= 1,dmns_4
    k1 = den(bbn4(2,ib))%p-den(bbn4(1,ib))%p
    k2 = den(bbn4(3,ib))%p-den(bbn4(2,ib))%p
    k3 = den(bbn4(4,ib))%p-den(bbn4(3,ib))%p
    k4 = den(bbn4(4,ib))%p-den(bbn4(1,ib))%p
    p12= den(bbn4(3,ib))%p-den(bbn4(1,ib))%p
    p23= den(bbn4(4,ib))%p-den(bbn4(2,ib))%p
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
     cm12= den(bbn4(1,ib))%m2
     cm22= den(bbn4(2,ib))%m2
     cm32= den(bbn4(3,ib))%m2
     cm42= den(bbn4(4,ib))%m2
     call olo(value,k12,k22,k32,k42,k122,k232,cm12,cm22,cm32,cm42)
     dloopfun(:,ib)  =        value(:) 
    elseif (scaloop.eq.3) then
     m12= dreal(den(bbn4(1,ib))%m2)
     m22= dreal(den(bbn4(2,ib))%m2)
     m32= dreal(den(bbn4(3,ib))%m2)
     m42= dreal(den(bbn4(4,ib))%m2)
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
!
!
! scaloop= 2 -> avh 1-loop scalar functions (massive with complex masses)
! WARNING: scaloop= 3 return double precision results
!
!
 module mp_loopfunctions
  include 'cts_mprec.h'
  use denominators
  use scale
  use dimensions
  use avh_olo
  implicit none
  private
  public :: allocate_mp_loopfun,get_mp_loop
  public :: mp_aloopfun,mp_bloopfun,mp_b1loopfun,mp_b11loopfun,mp_cloopfun,mp_dloopfun
  include 'cts_mpc.h'
   , private :: mp_czero
  integer, private :: ic,ierr
!
! variables for the 1-point sector:
!
  include 'cts_mpc.h'
   , dimension(:,:), allocatable :: mp_aloopfun
!
! variables for the 2-point sector:
!
  include 'cts_mpc.h'
   , dimension(:,:), allocatable :: mp_bloopfun
  include 'cts_mpc.h'
   , dimension(:,:), allocatable :: mp_b1loopfun
  include 'cts_mpc.h'
   , dimension(:,:), allocatable :: mp_b11loopfun
!
! variables for the 3-point sector:
!
  include 'cts_mpc.h'
   , dimension(:,:), allocatable :: mp_cloopfun
!
! variables for the 4-point sector:
!
  include 'cts_mpc.h'
   , dimension(:,:), allocatable :: mp_dloopfun
  save mp_aloopfun,mp_bloopfun,mp_b1loopfun,mp_b11loopfun
  save mp_cloopfun,mp_dloopfun
!
  contains
!
  subroutine allocate_mp_loopfun(nprop)
   integer, intent(in) :: nprop
   mp_czero= 0.d0
   if (nprop.ge.1) call allocate_mp_loopfuna
   if (nprop.ge.2) call allocate_mp_loopfunb
   if (nprop.ge.3) call allocate_mp_loopfunc
   if (nprop.ge.4) call allocate_mp_loopfund
  end subroutine allocate_mp_loopfun
!
  subroutine allocate_mp_loopfuna
   ierr= -1 
   allocate (mp_aloopfun(0:2,dmns_a), stat=ierr)
   if (ierr.ne.0) STOP "Allocation error in allocate_mp_loopfuna"
   mp_aloopfun= mp_czero
  end subroutine allocate_mp_loopfuna
!
  subroutine allocate_mp_loopfunb
   ierr= -1 
   allocate (  mp_bloopfun(0:2,dmns_b), stat=ierr)
   allocate ( mp_b1loopfun(0:2,dmns_b), stat=ierr)
   allocate (mp_b11loopfun(0:2,dmns_b), stat=ierr)
   if (ierr.ne.0) STOP "Allocation error in allocate_mp_loopfunb"
     mp_bloopfun= mp_czero 
    mp_b1loopfun= mp_czero
   mp_b11loopfun= mp_czero
  end subroutine allocate_mp_loopfunb
!
  subroutine allocate_mp_loopfunc
   ierr= -1 
   allocate (mp_cloopfun(0:2,dmns_c), stat=ierr)
   if (ierr.ne.0) STOP "Allocation error in allocate_mp_loopfunc"
   mp_cloopfun= mp_czero
  end subroutine allocate_mp_loopfunc
!
  subroutine allocate_mp_loopfund
   ierr= -1 
   allocate (mp_dloopfun(0:2,dmns_d), stat=ierr)
   if (ierr.ne.0) STOP "Allocation error in allocate_mp_loopfund"
   mp_dloopfun= mp_czero
  end subroutine allocate_mp_loopfund
!
  subroutine get_mp_loop(nprop)
   integer, intent(in) :: nprop
   if (nprop.ge.1) call get_mp_aloop
   if (nprop.ge.2) call get_mp_bloop
   if (nprop.ge.3) call get_mp_cloop
   if (nprop.ge.4) call get_mp_dloop
  end subroutine get_mp_loop
!
  subroutine get_mp_aloop
!
!  A 1-loop scalar function
!
   integer :: ib
   include 'cts_mpc.h'
    , dimension(0:2) :: value
   include 'cts_mpr.h'    
    :: m12
   include 'cts_mpc.h' 
    :: cm12
   include 'cts_dpc.h' 
    :: qlI1
   include 'cts_dpr.h'    
    :: dm12
   do ib= 1,dmns_1
    if     (scaloop.eq.2) then
     cm12= mp_den(bbn1(1,ib))%m2
     call olo(value,cm12)
     mp_aloopfun(:,ib)= value(:) 
    elseif (scaloop.eq.3) then
     dm12= mp_den(bbn1(1,ib))%m2
     mp_aloopfun(2,ib)= qlI1(dm12,musq,-2)     
     mp_aloopfun(1,ib)= qlI1(dm12,musq,-1)     
     mp_aloopfun(0,ib)= qlI1(dm12,musq,0)     
    else
     stop 'value of scaloop not implemented'
    endif
   enddo
  end subroutine get_mp_aloop
!
  subroutine get_mp_bloop
!
!  B,B1 and B11 1-loop scalar functions
!
   use tensor_operations  
   integer :: ib
   include 'cts_mpr.h' 
    :: k12
   include 'cts_mpr.h'
    , dimension(0:3) :: k1
   include 'cts_mpc.h'
    , dimension(0:2) :: value
   include 'cts_mpc.h'
    , dimension(0:2) :: valb11,valb00,valb1,valb0
   include 'cts_mpr.h'    
    :: m12,m22
   include 'cts_mpc.h' 
    :: cm12,cm22
   include 'cts_dpc.h' 
    :: qlI2
   include 'cts_dpc.h'
    , dimension(0:2) :: dvalb11,dvalb00,dvalb1,dvalb0
   include 'cts_dpr.h'    
    :: dm12,dm22
   include 'cts_dpr.h' 
    :: dk12
   do ib= 1,dmns_2
    do ic= 0,3
     k1(ic)= mp_den(bbn2(2,ib))%p(ic) &
            -mp_den(bbn2(1,ib))%p(ic)
    enddo
    call contr(k1,k1,k12)
!!    if (dabs(k12/roots/roots).lt.1.d-8) k12= 0.d0
    if     (scaloop.eq.2) then
     cm12= mp_den(bbn2(1,ib))%m2
     cm22= mp_den(bbn2(2,ib))%m2
     call olo(valb11,valb00,valb1,valb0,k12,cm12,cm22)
     mp_bloopfun(:,ib)  =  valb0(:)
     mp_b1loopfun(:,ib) =  valb1(:)
     mp_b11loopfun(:,ib)=  valb11(:)
    elseif (scaloop.eq.3) then
     dm12= mp_den(bbn2(1,ib))%m2
     dm22= mp_den(bbn2(2,ib))%m2
     dk12= k12 
     mp_bloopfun(2,ib)= qlI2(dk12,dm12,dm22,musq,-2)     
     mp_bloopfun(1,ib)= qlI2(dk12,dm12,dm22,musq,-1)     
     mp_bloopfun(0,ib)= qlI2(dk12,dm12,dm22,musq,0)     
     call olo(dvalb11,dvalb00,dvalb1,dvalb0,dk12,dm12,dm22)
     mp_b1loopfun(0,ib) =  dvalb1(0)
     mp_b11loopfun(0,ib)=  dvalb11(0)
     mp_b1loopfun(1,ib) =  dvalb1(1)
     mp_b11loopfun(1,ib)=  dvalb11(1)
     mp_b1loopfun(2,ib) =  dvalb1(2)
     mp_b11loopfun(2,ib)=  dvalb11(2)
    else
     stop 'value of scaloop not implemented'
    endif
   enddo
  end subroutine get_mp_bloop
!
  subroutine get_mp_cloop
!
!  C 1-loop scalar function
!
   use tensor_operations  
   integer :: ib
   include 'cts_mpr.h' 
    :: k12,k22,k32
   include 'cts_mpr.h'
    , dimension(0:3) :: k1,k2,k3
   include 'cts_mpc.h'
    , dimension(0:2) :: value
   include 'cts_mpr.h'    
    :: m12,m22,m32
   include 'cts_mpc.h' 
    :: cm12,cm22,cm32
   include 'cts_dpc.h' 
    :: qlI3
   include 'cts_dpr.h'    
    :: dm12,dm22,dm32
   include 'cts_dpr.h' 
    :: dk12,dk22,dk32
   do ib= 1,dmns_3
    do ic= 0,3
     k1(ic)= mp_den(bbn3(2,ib))%p(ic) &
            -mp_den(bbn3(1,ib))%p(ic)
     k2(ic)= mp_den(bbn3(3,ib))%p(ic) &
            -mp_den(bbn3(2,ib))%p(ic)
     k3(ic)= mp_den(bbn3(3,ib))%p(ic) &
            -mp_den(bbn3(1,ib))%p(ic)
    enddo
    call contr(k1,k1,k12)
    call contr(k2,k2,k22)
    call contr(k3,k3,k32)
!!    if (dabs(k12/roots/roots).lt.1.d-8) k12= 0.d0
!!    if (dabs(k22/roots/roots).lt.1.d-8) k22= 0.d0
!!    if (dabs(k32/roots/roots).lt.1.d-8) k32= 0.d0
    if     (scaloop.eq.2) then
     cm12= mp_den(bbn3(1,ib))%m2
     cm22= mp_den(bbn3(2,ib))%m2
     cm32= mp_den(bbn3(3,ib))%m2
     call olo(value,k12,k22,k32,cm12,cm22,cm32)
     mp_cloopfun(:,ib)  =        value(:) 
    elseif (scaloop.eq.3) then
     dm12= mp_den(bbn3(1,ib))%m2
     dm22= mp_den(bbn3(2,ib))%m2
     dm32= mp_den(bbn3(3,ib))%m2
     dk12= k12
     dk22= k22
     dk32= k32
     mp_cloopfun(2,ib)= qlI3(dk12,dk22,dk32,dm12,dm22,dm32,musq,-2)     
     mp_cloopfun(1,ib)= qlI3(dk12,dk22,dk32,dm12,dm22,dm32,musq,-1)     
     mp_cloopfun(0,ib)= qlI3(dk12,dk22,dk32,dm12,dm22,dm32,musq,0)     
    else
     stop 'value of scaloop not implemented'
    endif
   enddo
  end subroutine get_mp_cloop
!
  subroutine get_mp_dloop
!
!  D 1-loop scalar function
!
   use tensor_operations  
   integer :: ib
   include 'cts_mpr.h' 
    :: k12,k22,k32,k42,k122,k232
   include 'cts_mpr.h'
    , dimension(0:3) :: k1,k2,k3,k4,p12,p23
   include 'cts_mpc.h'
    , dimension(0:2) :: value
   include 'cts_mpr.h'    
    :: m12,m22,m32,m42
   include 'cts_mpc.h' 
    :: cm12,cm22,cm32,cm42
   include 'cts_dpc.h' 
    :: qlI4
   include 'cts_dpr.h'    
    :: dm12,dm22,dm32,dm42
   include 'cts_dpr.h' 
    :: dk12,dk22,dk32,dk42,dk122,dk232
   do ib= 1,dmns_4
    do ic= 0,3
     k1(ic) = mp_den(bbn4(2,ib))%p(ic) &
             -mp_den(bbn4(1,ib))%p(ic)
     k2(ic) = mp_den(bbn4(3,ib))%p(ic) &
             -mp_den(bbn4(2,ib))%p(ic)
     k3(ic) = mp_den(bbn4(4,ib))%p(ic) &
             -mp_den(bbn4(3,ib))%p(ic)
     k4(ic) = mp_den(bbn4(4,ib))%p(ic) &
             -mp_den(bbn4(1,ib))%p(ic)
     p12(ic)= mp_den(bbn4(3,ib))%p(ic) &
             -mp_den(bbn4(1,ib))%p(ic)
     p23(ic)= mp_den(bbn4(4,ib))%p(ic) &
             -mp_den(bbn4(2,ib))%p(ic)
    enddo
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
    if     (scaloop.eq.2) then
     cm12= mp_den(bbn4(1,ib))%m2
     cm22= mp_den(bbn4(2,ib))%m2
     cm32= mp_den(bbn4(3,ib))%m2
     cm42= mp_den(bbn4(4,ib))%m2
     call olo(value,k12,k22,k32,k42,k122,k232,cm12,cm22,cm32,cm42)
     mp_dloopfun(:,ib)  =        value(:) 
    elseif (scaloop.eq.3) then
     dm12= mp_den(bbn4(1,ib))%m2
     dm22= mp_den(bbn4(2,ib))%m2
     dm32= mp_den(bbn4(3,ib))%m2
     dm42= mp_den(bbn4(4,ib))%m2
     dk12 = k12
     dk22 = k22
     dk32 = k32
     dk42 = k42
     dk122= k122
     dk232= k232
     mp_dloopfun(2,ib)= qlI4(dk12,dk22,dk32,dk42,dk122,dk232,dm12,dm22,dm32,dm42,musq,-2)   
     mp_dloopfun(1,ib)= qlI4(dk12,dk22,dk32,dk42,dk122,dk232,dm12,dm22,dm32,dm42,musq,-1)   
     mp_dloopfun(0,ib)= qlI4(dk12,dk22,dk32,dk42,dk122,dk232,dm12,dm22,dm32,dm42,musq,0)   
    else
     stop 'value of scaloop not implemented'
    endif
   enddo
  end subroutine get_mp_dloop
!
 end module mp_loopfunctions





