!
! in this version numfunc or numfuncrec is used if 
! ext_num_for_r1=.true. or .false, in  ctsinit
!
 module scale
  implicit none                    
  private
  include 'cts_dpr.h'
   , public :: roots,limit,musq,muscale
  integer, public :: scaloop
  logical, public :: stablen=.true.,ext_num_for_r1=.true.
  include 'cts_dpr.h'
   , public :: precstablen,n_mp= 0,n_unst= 0,n_tot= 0
  include 'cts_dpr.h'
   , public :: llimit= 0.3d0
 end module scale
!
 module qt2value
  include 'cts_mprec.h'
  implicit none
  private
  include 'cts_dpc.h'
   , public :: qt2
  include 'cts_mpc.h'
   , public :: mpqt2
  logical, public :: rational=.false. 
 end module qt2value
!
 module denominators
  include 'cts_mprec.h'
  use def_propagator                                       
  use def_mp_propagator                                       
  use qt2value
  use maxnumden 
  implicit none
  private
  public :: load_denominators,value,load_vden,dp_allocate_den,mp_allocate_den
  type(propagator), dimension(:), public, allocatable :: den
  type(mp_propagator), dimension(:), public, allocatable :: mp_den
  include 'cts_dpc.h'
   , dimension(:,:), public, allocatable :: vden
  include 'cts_mpc.h'
   , dimension(:,:), public, allocatable :: mp_vden
  integer, private :: ierr,icomp1,icomp2
  save den,vden
!
  interface load_denominators
    module procedure load_dp_denominators
    module procedure load_mp_denominators
  end interface!load_denominators
!
  interface initden
    module procedure dp_initden
    module procedure mp_initden
  end interface!initden
!
  interface value
    module procedure dp_value
    module procedure mp_value
  end interface!value
!
  interface load_vden
    module procedure dp_load_vden
    module procedure mp_load_vden
  end interface!load_vden
!
  contains
!
  subroutine load_dp_denominators(denvec,nden)
   integer :: nden
   type(propagator), intent(in) :: denvec(0:nden-1)
   include 'cts_dpr.h' 
    :: p
   integer :: k,kplus1
   if (nden.gt.maxden) stop 'too many denominators in input!'
   call initden(p)
   do k=0,nden-1
      kplus1= k+1
      den(kplus1)=denvec(k)
      den(kplus1)%i=kplus1
   enddo
  end subroutine load_dp_denominators
!
  subroutine load_mp_denominators(denvec,nden)
   integer :: nden
   type(mp_propagator), intent(in) :: denvec(0:nden-1)
   include 'cts_mpr.h' 
    :: p
   integer :: k,kplus1
   if (nden.gt.maxden) stop 'too many denominators in input!'
   call initden(p)
   do k=0,nden-1
      kplus1= k+1
      mp_den(kplus1)=denvec(k)
      mp_den(kplus1)%i=kplus1
   enddo
  end subroutine load_mp_denominators
!
  subroutine dp_initden(p)
  use dimensions
  include 'cts_dpr.h' 
   :: p
  integer :: kk,k
  do kk= 1,dmns
   den(kk)%i = -1
   den(kk)%m2= 0.d0
   do k= 0,3
    den(kk)%p(k) = 0.d0
   enddo
  enddo
  end subroutine dp_initden
!
  subroutine mp_initden(p)
  use dimensions
  include 'cts_mpr.h' 
   :: p
  integer :: kk,k
  do kk= 1,dmns
   mp_den(kk)%i = -1
   mp_den(kk)%m2= 0.d0
   do k= 0,3
    mp_den(kk)%p(k) = 0.d0
   enddo
  enddo
  end subroutine mp_initden
!
  function dp_value(den,q)
   use tensor_operations  
   include 'cts_dpc.h' 
    :: dp_value
   type(propagator), intent (in) :: den
   include 'cts_dpc.h'
    , intent(in), dimension(0:3) :: q
   include 'cts_dpc.h'
    , dimension(0:3) :: qp
   include 'cts_dpc.h' 
    :: qp2
   integer :: k
   do k= 0,3
    qp(k)= q(k)+den%p(k)
   enddo
   call contr(qp,qp,qp2)
   dp_value = qp2-den%m2+qt2
  end function dp_value
!
  function mp_value(den,q)
   use tensor_operations  
   include 'cts_mpc.h' 
    :: mp_value
   type(mp_propagator), intent (in) :: den
   include 'cts_mpc.h'
    , intent(in), dimension(0:3) :: q
   include 'cts_mpc.h'
    , dimension(0:3) :: qp
   include 'cts_mpc.h' 
    :: qp2
   integer :: k
   do k= 0,3
    qp(k)= q(k)+den%p(k)
   enddo
   call contr(qp,qp,qp2)
   mp_value = qp2-den%m2+mpqt2
  end function mp_value
!
  subroutine dp_allocate_den
   use dimensions
   use maxsolutions
   ierr= -1
   allocate (vden(1:dmns_a, max_solutions), stat=ierr)
   allocate (den(1:dmns_a), stat=ierr)
   if (ierr.ne.0) STOP "Allocation error in dp_allocate_den"
   vden= 0.d0
   do icomp2= 1,dmns_a
    den(icomp2)%i= 0
    den(icomp2)%m2= 0.d0
    den(icomp2)%p(0)= 0.d0
    den(icomp2)%p(1)= 0.d0
    den(icomp2)%p(2)= 0.d0
    den(icomp2)%p(3)= 0.d0
   enddo
  end subroutine dp_allocate_den
!
  subroutine mp_allocate_den
   use dimensions
   use maxsolutions
   ierr= -1
   allocate (mp_vden(1:dmns_a, max_solutions), stat=ierr)
   allocate (mp_den(1:dmns_a), stat=ierr)
   if (ierr.ne.0) STOP "Allocation error in mp_allocate_den"
   do icomp2= 1,dmns_a
    do icomp1= 1,max_solutions 
      mp_vden(icomp2,icomp1)= 0.d0
    enddo
    mp_den(icomp2)%i= 0
    mp_den(icomp2)%m2= 0.d0
    mp_den(icomp2)%p(0)= 0.d0
    mp_den(icomp2)%p(1)= 0.d0
    mp_den(icomp2)%p(2)= 0.d0
    mp_den(icomp2)%p(3)= 0.d0
   enddo
  end subroutine mp_allocate_den
!
  subroutine dp_load_vden(q,i,j,p0,m20)
   use tensor_operations  
   include 'cts_dpc.h'
    , intent(in), dimension(0:3) :: q
   include 'cts_dpr.h'
    , intent(in), dimension(0:3) :: p0
   include 'cts_dpc.h'
    :: m20
   include 'cts_dpc.h'
    :: ps(0:3),psq(0:3),pspsq,res 
   integer, intent(in) :: i,j
   integer :: k
!!   if (den(i)%i.gt.0) vden(i,j) = value(den(i),q)
   do k= 0,3
    ps(k)  = den(i)%p(k)-p0(k)
    psq(k) = ps(k)+2.d0*(q(k)+p0(k))
   enddo
   call contr(ps,psq,pspsq)
   res= m20-den(i)%m2
   res= res+pspsq
   vden(i,j)= res
  end subroutine dp_load_vden
!
  subroutine mp_load_vden(q,i,j,p0,m20)
   use tensor_operations  
   include 'cts_mpc.h'
    , intent(in), dimension(0:3) :: q
   include 'cts_mpr.h'
    , intent(in), dimension(0:3) :: p0
   include 'cts_mpc.h'
    :: m20
   include 'cts_mpc.h'
    :: ps(0:3),psq(0:3),pspsq,res 
   integer, intent(in) :: i,j
   integer :: k
!!   if (mp_den(i)%i.gt.0) mp_vden(i,j) = value(mp_den(i),q)
   do k= 0,3
    ps(k)  = den(i)%p(k)-p0(k)
    psq(k) = ps(k)+2.d0*(q(k)+p0(k))
   enddo
   call contr(ps,psq,pspsq)
   res= m20-den(i)%m2
   res= res+pspsq
   mp_vden(i,j)= res
  end subroutine mp_load_vden
!
 end module denominators
!
 module cuttings
  include 'cts_mprec.h'
  use tensor_operations  
  use def_solcut
  use def_mp_solcut
  use denominators
  use constants
  use def_propagator                                       
  use def_mp_propagator                                       
  use qt2value
  implicit none
  private
  public :: cut 
  integer, private :: i,kk
  interface cut
   module procedure dp_cutting4
   module procedure dp_cutting3
   module procedure dp_cutting2
   module procedure dp_cutting1_newbase
   module procedure mp_cutting4
   module procedure mp_cutting3
   module procedure mp_cutting2
   module procedure mp_cutting1_newbase
  end interface!cut
!
  interface build_l
   module procedure dp_build_l
   module procedure mp_build_l
  end interface!build_l
  contains
!
  subroutine dp_build_l(k1,k2in,l1,l2,l3,l4,al1,al2,bet,ga)  
   use scale
   include 'cts_dpr.h' 
    :: p
   include 'cts_dpr.h'
    , intent(in), dimension(0:3) :: k1,k2in
   include 'cts_dpr.h'
    , dimension(0:3) :: k2
   include 'cts_dpc.h'
    , intent(out), dimension(0:3) :: l1,l2,l3,l4
   include 'cts_dpc.h'
    , intent(out) :: al1,al2,bet,ga 
   include 'cts_dpr.h'
    :: k1k1,k1k2,k2k2
   include 'cts_dpr.h'
    :: d12a,d12b,d12c,d12
   include 'cts_dpr.h'
    , dimension(1:3) :: s10,s20,d10,d20
   include 'cts_dpc.h' 
    :: del12,rdel12,b1p,b1m,b2p,b2m,c1p,c1m,c2p,c2m,ausp,ausm
   integer :: k
   k2= k2in
 1 do k= 1,3
    s10(k)= k1(0)+k1(k); d10(k)= k1(0)-k1(k) 
    s20(k)= k2(0)+k2(k); d20(k)= k2(0)-k2(k) 
   enddo
   d12a= (k2(1)*d10(3)-k1(1)*d20(3))*(k2(1)*s10(3)-k1(1)*s20(3))
   d12b= (k2(2)*s10(1)-k1(2)*s20(1))*(k2(2)*d10(1)-k1(2)*d20(1))
   d12c= (k2(3)*s10(2)-k1(3)*s20(2))*(k2(3)*d10(2)-k1(3)*d20(2))
   d12 =  d12a+d12b+d12c
   if (abs(d12/roots**4).lt.1.d-60) then
     k2(0)= k2(0)*1.0000000000001d0 
     goto 1 
   endif
   del12= d12*c1(p)
   rdel12= sqrt(del12)
   call contr(k1,k1,k1k1)
   call contr(k1,k2,k1k2)
   call contr(k2,k2,k2k2)
   if (k1k2.gt.0.d0) then
    k = 1 
   else
    k =-1 
   endif
   ga= k1k2+k*rdel12
   al1= k1k1/ga
   al2= k2k2/ga
   if (dreal(al1*al2).lt.0.d0) then
     bet= c1(p)/(c1(p)-al1*al2)
   else
     bet= k*ga**2*(c1(p)+al1*al2)/4.d0/k1k2/rdel12
   endif
   do k= 0,3
    l1(k)= bet*(k1(k)-al1*k2(k))
    l2(k)= bet*(k2(k)-al2*k1(k))
   enddo
   ausm= l1(0)-l1(3)
   ausp= l1(0)+l1(3)
   if (abs(ausm).gt.abs(ausp)) then 
    c1p= sqrt(ausm)
    c1m= c1p
    b1p= (l1(1)-ci(p)*l1(2))/c1p
    b1m= (l1(1)+ci(p)*l1(2))/c1p
   else
    b1p= sqrt(ausp)
    b1m= b1p
    c1p= (l1(1)+ci(p)*l1(2))/b1p
    c1m= (l1(1)-ci(p)*l1(2))/b1p
   endif
   ausm= l2(0)-l2(3)
   ausp= l2(0)+l2(3)
   if (abs(ausm).gt.abs(ausp)) then 
    c2p= sqrt(ausm)
    c2m= c2p
    b2p= (l2(1)-ci(p)*l2(2))/c2p
    b2m= (l2(1)+ci(p)*l2(2))/c2p
   else
    b2p= sqrt(ausp)
    b2m= b2p
    c2m= (l2(1)-ci(p)*l2(2))/b2p
    c2p= (l2(1)+ci(p)*l2(2))/b2p
   endif
   l3(0)=     b1m*b2p  + c1m*c2p
   l3(1)=     b1m*c2p + c1m*b2p
   l3(2)= ci(p)*(c1m*b2p - b1m*c2p)
   l3(3)=     b1m*b2p  - c1m*c2p  
   l4(0)=     b2m*b1p  + c2m*c1p
   l4(1)=     b2m*c1p + c2m*b1p
   l4(2)= ci(p)*(c2m*b1p - b2m*c1p)
   l4(3)=     b2m*b1p  - c2m*c1p
  end subroutine dp_build_l
!
  subroutine mp_build_l(k1,k2in,l1,l2,l3,l4,al1,al2,bet,ga)  
   use scale
   include 'cts_mpr.h' 
    :: p
   include 'cts_mpr.h'
    , intent(in), dimension(0:3) :: k1,k2in
   include 'cts_mpr.h'
    , dimension(0:3) :: k2
   include 'cts_mpc.h'
    , intent(out), dimension(0:3) :: l1,l2,l3,l4
   include 'cts_mpc.h'
    , intent(out) :: al1,al2,bet,ga 
   include 'cts_mpr.h'
    :: k1k1,k1k2,k2k2,r12
   include 'cts_mpr.h'
    :: d12a,d12b,d12c,d12
   include 'cts_mpr.h'
    , dimension(1:3) :: s10,s20,d10,d20
   include 'cts_mpc.h' 
    :: del12,rdel12,b1p,b1m,b2p,b2m,c1p,c1m,c2p,c2m,ausp,ausm
   integer :: k
   k2= k2in
 1 do k= 1,3
    s10(k)= k1(0)+k1(k); d10(k)= k1(0)-k1(k) 
    s20(k)= k2(0)+k2(k); d20(k)= k2(0)-k2(k) 
   enddo
   d12a= (k2(1)*d10(3)-k1(1)*d20(3))*(k2(1)*s10(3)-k1(1)*s20(3))
   d12b= (k2(2)*s10(1)-k1(2)*s20(1))*(k2(2)*d10(1)-k1(2)*d20(1))
   d12c= (k2(3)*s10(2)-k1(3)*s20(2))*(k2(3)*d10(2)-k1(3)*d20(2))
   d12 =  d12a+d12b+d12c
   if (abs(d12/roots**4).lt.1.d-120) then
     k2(0)= k2(0)*1.0000000000001d0 
     goto 1 
   endif
   del12= d12*c1(p)
   rdel12= sqrt(del12)
   call contr(k1,k1,k1k1)
   call contr(k1,k2,k1k2)
   call contr(k2,k2,k2k2)
   if (k1k2.gt.0.d0) then
    k = 1
   else
    k =-1
   endif
   ga= k1k2+k*rdel12
   al1= k1k1/ga
   al2= k2k2/ga
   r12= (al1*al2+conjg(al1*al2))
   if (r12.lt.0.d0) then
     bet= c1(p)/(c1(p)-al1*al2)
   else
     bet= k*ga**2*(c1(p)+al1*al2)/4.d0/k1k2/rdel12
   endif
   do k= 0,3
    l1(k)= bet*(k1(k)-al1*k2(k))
    l2(k)= bet*(k2(k)-al2*k1(k))
   enddo
   ausm= l1(0)-l1(3)
   ausp= l1(0)+l1(3)
   if (abs(ausm).gt.abs(ausp)) then 
    c1p= sqrt(ausm)
    c1m= c1p
    b1p= (l1(1)-ci(p)*l1(2))/c1p
    b1m= (l1(1)+ci(p)*l1(2))/c1p
   else
    b1p= sqrt(ausp)
    b1m= b1p
    c1p= (l1(1)+ci(p)*l1(2))/b1p
    c1m= (l1(1)-ci(p)*l1(2))/b1p
   endif
   ausm= l2(0)-l2(3)
   ausp= l2(0)+l2(3)
   if (abs(ausm).gt.abs(ausp)) then 
    c2p= sqrt(ausm)
    c2m= c2p
    b2p= (l2(1)-ci(p)*l2(2))/c2p
    b2m= (l2(1)+ci(p)*l2(2))/c2p
   else
    b2p= sqrt(ausp)
    b2m= b2p
    c2m= (l2(1)-ci(p)*l2(2))/b2p
    c2p= (l2(1)+ci(p)*l2(2))/b2p
   endif
   l3(0)=     b1m*b2p  + c1m*c2p
   l3(1)=     b1m*c2p + c1m*b2p
   l3(2)= ci(p)*(c1m*b2p - b1m*c2p)
   l3(3)=     b1m*b2p  - c1m*c2p  
   l4(0)=     b2m*b1p  + c2m*c1p
   l4(1)=     b2m*c1p + c2m*b1p
   l4(2)= ci(p)*(c2m*b1p - b2m*c1p)
   l4(3)=     b2m*b1p  - c2m*c1p
  end subroutine mp_build_l
!
  subroutine dp_cutting4(den0,den1,den2,den3,cut4)
   use dimensions
   include 'cts_dpr.h' 
    :: p
   type(propagator), intent(in) :: den0,den1,den2,den3
   type(solcut4) ,intent(out) :: cut4
   include 'cts_dpr.h'
    , dimension(0:3) :: p0,p1,p2,p3,k1,k2,k3
   include 'cts_dpr.h'
    :: k1k1,k2k2,k3k3
   include 'cts_dpc.h'
    :: m02,m12,m22,m32
   include 'cts_dpc.h'
    :: dd0,dd1,dd2,dd3
   include 'cts_dpc.h'
    :: l1k3,l2k3,l3k3,l4k3,x10,x20,x3p,x4p,x3m,x4m
   include 'cts_dpc.h'
    :: cc,ca3,cb3,root,cb3p,cb3m,al1,al2,bet,gm,z
   include 'cts_dpc.h'
    , dimension(0:3) :: l1,l2,l3,l4
   integer :: k
!
   p0= den0%p 
   p1= den1%p 
   p2= den2%p 
   p3= den3%p 
!
   m02= den0%m2-qt2 
   m12= den1%m2-qt2 
   m22= den2%m2-qt2 
   m32= den3%m2-qt2 
!
   do k= 0,3
    k1(k)= p1(k)-p0(k)
    k2(k)= p2(k)-p0(k)
    k3(k)= p3(k)-p0(k)
   enddo
! 
   call contr(k1,k1,k1k1)
   call contr(k2,k2,k2k2)
   call contr(k3,k3,k3k3)
   call build_l(k1,k2,l1,l2,l3,l4,al1,al2,bet,gm)
!
   call contr(l1,k3,l1k3)
   call contr(l2,k3,l2k3)
   call contr(l3,k3,l3k3)
   call contr(l4,k3,l4k3)
   z= bet/gm
   dd0= m02
   dd1= m12-k1k1
   dd2= m22-k2k2
   dd3= m32-k3k3
   x10= z*(dd2-al2*dd1-dd0*(1.d0-al2))
   x20= z*(dd1-al1*dd2-dd0*(1.d0-al1))
   cc = 0.25d0*(x10*x20-dd0/gm)
   ca3= -l3k3/l4k3
   cb3= (dd3-dd0-2.d0*x10*l1k3-2.d0*x20*l2k3)/2.d0/l4k3
   root= sqrt(cb3**2+4.d0*cc*ca3)
   cb3p= cb3+root
   cb3m= cb3-root
   if (abs(cb3m).ge.abs(cb3p)) then 
     x3p= (-cb3m)/2.d0/ca3
     x3m= -cc/ca3/x3p
     x4m= (cb3m)/2.d0
     x4p= -cc*ca3/x4m
! 
!     x4p= cc/x3p
!     x4m= cc/x3m
!
   else
     x3m= (-cb3p)/2.d0/ca3
     x3p= -cc/ca3/x3m
     x4p= (cb3p)/2.d0
     x4m= -cc*ca3/x4p
!      
!     x4p= cc/x3p
!     x4m= cc/x3m
!
   endif   
!
!  the 2 solutions, the basis and the vector t
! 
!   q(1)= q^+
!   q(2)= q^-
!
   do i= 0,3
    cut4%q(i,1)= -p0(i)+x10*l1(i)+x20*l2(i)+x3p*l3(i)+x4p*l4(i)
    cut4%q(i,2)= -p0(i)+x10*l1(i)+x20*l2(i)+x3m*l3(i)+x4m*l4(i)
    cut4%l1(i)= l1(i)
    cut4%l2(i)= l2(i)
    cut4%l3(i)= l3(i)
    cut4%l4(i)= l4(i)
    cut4%t(i) = l3(i)*l4k3-l4(i)*l3k3
   enddo
!
!  computing all denominators at the solutions
! 
   do i= 1,2
    do kk= 1,dmns
     if (kk.eq.den0%i.or. &
         kk.eq.den1%i.or. &
         kk.eq.den2%i.or. &
         kk.eq.den3%i) then
      vden(kk,i)= c0(p)
     else
      call load_vden(cut4%q(:,i),kk,i,p0,den0%m2)
     endif 
    enddo
   enddo
  end subroutine dp_cutting4  
!
  subroutine dp_cutting3(den0,den1,den2,cut3,dmr)
   use dimensions
   include 'cts_dpr.h' 
    :: p
   type(propagator), intent(in) :: den0,den1,den2
   type(solcut3), intent(out) :: cut3
   integer, intent(in) ::  dmr
   include 'cts_dpr.h'
    , dimension(0:3) :: p0,p1,p2,k1,k2
   include 'cts_dpr.h'
    :: k1k1,k2k2,k1k2,phi
   include 'cts_dpc.h'
    :: m02,m12,m22
   include 'cts_dpc.h'
    :: dd0,dd1,dd2
   include 'cts_dpc.h'
    :: x10,x20,x3,x4,q3caus
   include 'cts_dpc.h'
    :: cc,tau,ca3,al1,al2,bet,gm,z,cc4 
   include 'cts_dpc.h'
    , dimension(0:3)  :: l1,l2,l3,l4
   integer :: k,nsol
   p0= den0%p 
   p1= den1%p 
   p2= den2%p 
!
   m02= den0%m2-qt2 
   m12= den1%m2-qt2 
   m22= den2%m2-qt2 
!
   do k= 0,3
    k1(k)= p1(k)-p0(k)
    k2(k)= p2(k)-p0(k)
   enddo
!
   call contr(k1,k1,k1k1)
   call contr(k2,k2,k2k2)
   call build_l(k1,k2,l1,l2,l3,l4,al1,al2,bet,gm)
   if (dmr.eq.-1) then
    call contr(k1,k2,k1k2)
    cut3%rat1= k1k2-k1k1-k2k2+2.d0*(den0%m2+den1%m2+den2%m2)
   else
    cut3%rat1= 0.d0
   endif
   z= bet/gm
   dd0= m02
   dd1= m12-k1k1
   dd2= m22-k2k2
   x10= z*(dd2-al2*dd1-dd0*(1.d0-al2))
   x20= z*(dd1-al1*dd2-dd0*(1.d0-al1))
   cc = 0.25d0*(x10*x20-dd0/gm)
!  comment: the next line to avoid underflows 
   if (abs(cc).lt.1.d-60) cc= 0.d0
   cut3%gm= gm
   cut3%cc= cc
   do i= 0,3
    cut3%l1(i)= l1(i)
    cut3%l2(i)= l2(i)
    cut3%l3(i)= l3(i)
    cut3%l4(i)= l4(i)
   enddo
   cc4= cc**4
   if  (abs(cc4).gt.my_tiny(p)) then
    phi= atan2(dimag(cc4),dreal(cc4))
   else
    phi= 0.d0
   endif
   tau= exp(ci(p)/8.d0*(pi(p)-phi))
   cut3%tau= tau
   if (dmr.eq.-1) then
! comment: later on c_5 and c_6 not to be computed with rational=.true. 
    nsol= 7
    do i= 0,3
     q3caus= -p0(i)+x10*l1(i)+x20*l2(i)
     x4= c1(p)
     x3= cc/x4
     cut3%q(i,1) = q3caus+x3*tau*l3(i)+x4/tau*l4(i)   
!
     x4= (cexp4(p))**2
     x3= cc/x4
     cut3%q(i,2) = q3caus+x3*tau*l3(i)+x4/tau*l4(i)   
! 
     x4= (cexp4(p))**4
     x3= cc/x4
     cut3%q(i,3) = q3caus+x3*tau*l3(i)+x4/tau*l4(i)   
!
     x4= (cexp4(p))**6
     x3= cc/x4
     cut3%q(i,4) = q3caus+x3*tau*l3(i)+x4/tau*l4(i)   
! 
     x3= c1(p)
     x4= cc/x3
     cut3%q(i,5) = q3caus+x3/tau*l3(i)+x4*tau*l4(i)   
!
     x3= (cexp4(p))**2
     x4= cc/x3
     cut3%q(i,6) = q3caus+x3/tau*l3(i)+x4*tau*l4(i)   
!
     x3= (cexp4(p))**4
     x4= cc/x3
     cut3%q(i,7) = q3caus+x3/tau*l3(i)+x4*tau*l4(i)   
    enddo
   elseif (dmr.eq.0) then
    if (rational) then
     nsol= 3
     do i= 0,3
      q3caus= -p0(i)+x10*l1(i)+x20*l2(i)
      x4= c1(p)
      x3= cc/x4
      cut3%q(i,1) = q3caus+x3*tau*l3(i)+x4/tau*l4(i)   
!
      x4= (cexp4(p))**4
      x3= cc/x4
      cut3%q(i,2) = q3caus+x3*tau*l3(i)+x4/tau*l4(i)   
!
      x3= c1(p)
      x4= cc/x3
      cut3%q(i,3) = q3caus+x3/tau*l3(i)+x4*tau*l4(i)   
     enddo
    else
     nsol= 7
     do i= 0,3
      q3caus= -p0(i)+x10*l1(i)+x20*l2(i)
      x4= c1(p)
      x3= cc/x4
      cut3%q(i,1) = q3caus+x3*tau*l3(i)+x4/tau*l4(i)   
!
      x4= (cexp4(p))**2
      x3= cc/x4
      cut3%q(i,2) = q3caus+x3*tau*l3(i)+x4/tau*l4(i)   
!
      x4= (cexp4(p))**4
      x3= cc/x4
      cut3%q(i,3) = q3caus+x3*tau*l3(i)+x4/tau*l4(i)   
!
      x4= (cexp4(p))**6
      x3= cc/x4
      cut3%q(i,4) = q3caus+x3*tau*l3(i)+x4/tau*l4(i)   
!
      x3= c1(p)
      x4= cc/x3
      cut3%q(i,5) = q3caus+x3/tau*l3(i)+x4*tau*l4(i)   
!
      x3= (cexp4(p))**2
      x4= cc/x3
      cut3%q(i,6) = q3caus+x3/tau*l3(i)+x4*tau*l4(i)   
!
      x3= (cexp4(p))**4
      x4= cc/x3
      cut3%q(i,7) = q3caus+x3/tau*l3(i)+x4*tau*l4(i)   
     enddo
    endif
   elseif (dmr.eq.1) then
    if (rational) then
     nsol= 2
     do i= 0,3
      q3caus= -p0(i)+x10*l1(i)+x20*l2(i)
      x4= c1(p)
      x3= cc/x4
      cut3%q(i,1) = q3caus+x3*tau*l3(i)+x4/tau*l4(i)   
!
      x4= (cexp4(p))**4
      x3= cc/x4
      cut3%q(i,2) = q3caus+x3*tau*l3(i)+x4/tau*l4(i)   
     enddo
    else
     nsol= 5
     do i= 0,3
      q3caus= -p0(i)+x10*l1(i)+x20*l2(i)
      x4= c1(p)
      x3= cc/x4
      cut3%q(i,1) = q3caus+x3*tau*l3(i)+x4/tau*l4(i)   
!
      x4= (cexp4(p))**4
      x3= cc/x4
      cut3%q(i,2) = q3caus+x3*tau*l3(i)+x4/tau*l4(i)   
!
      x4= (cexp4(p))**6
      x3= cc/x4
      cut3%q(i,3) = q3caus+x3*tau*l3(i)+x4/tau*l4(i)   
!
      x3= c1(p)
      x4= cc/x3
      cut3%q(i,4) = q3caus+x3/tau*l3(i)+x4*tau*l4(i)   
!
      x3= (cexp4(p))**4
      x4= cc/x3
      cut3%q(i,5) = q3caus+x3/tau*l3(i)+x4*tau*l4(i)   
     enddo
    endif
   elseif (dmr.eq.2) then
    nsol= 3
    do i= 0,3
     q3caus= -p0(i)+x10*l1(i)+x20*l2(i)
     x4= c1(p)
     x3= cc/x4
     cut3%q(i,1) = q3caus+x3*tau*l3(i)+x4/tau*l4(i)   
!
     x4= (cexp4(p))**4
     x3= cc/x4
     cut3%q(i,2) = q3caus+x3*tau*l3(i)+x4/tau*l4(i)   
!
     x3= c1(p)
     x4= cc/x3
     cut3%q(i,3) = q3caus+x3/tau*l3(i)+x4*tau*l4(i)   
    enddo
   else
    nsol= 1
    do i= 0,3
     q3caus= -p0(i)+x10*l1(i)+x20*l2(i)
     x4= c1(p)
     x3= cc/x4
     cut3%q(i,1) = q3caus+x3*tau*l3(i)+x4/tau*l4(i)   
    enddo
   endif
!
!  computing all denominators at the solutions
! 
   do i= 1,nsol
    do kk= 1,dmns
     if (kk.eq.den0%i.or. &
         kk.eq.den1%i.or. &
         kk.eq.den2%i) then
      vden(kk,i)= c0(p)
     else
      call load_vden(cut3%q(:,i),kk,i,p0,den0%m2)
     endif 
    enddo
   enddo
   end subroutine dp_cutting3  
!
  subroutine dp_cutting2(den0,den1,cut2,dmr)
   use dimensions
   include 'cts_dpr.h' 
    :: p,a,b,c
   type(propagator), intent(in) :: den0,den1
   type(solcut2), intent(out) :: cut2
   integer, intent(in) ::  dmr
   include 'cts_dpr.h'
    , dimension(0:3) :: p0,p1,k1,v
   include 'cts_dpr.h'
    :: k1k1,k1v,factor,phi
   include 'cts_dpc.h'
    :: m02,m12,ak,a0,a1
   include 'cts_dpc.h'
    :: dd0,dd1
   integer :: k,nsol
   include 'cts_dpc.h' 
    :: yv0,yvsigma,yvlambda,y3,y4
   include 'cts_dpc.h' 
    :: tau,taul,cc3,al,al2,bet,gm,cf0,cflambda,cfsigma,q2caus
   include 'cts_dpc.h'
    , dimension(0:3)  :: l1,l2,l3,l4

   p0= den0%p 
   p1= den1%p 
!
   m02= den0%m2-qt2 
   m12= den1%m2-qt2 
!
   do k= 0,3
    k1(k)= p1(k)-p0(k)
   enddo
!
   call contr(k1,k1,k1k1)
!
!  define the arbitrary massless 4-vector v
!
!-comment
    a= 1.d0   
    v(0)= sign(a,k1(0))         
    v(1)=-sign(a/root3(p),k1(1))
    v(2)=-sign(a/root3(p),k1(2))
    v(3)=-sign(a/root3(p),k1(3))
!-comment
   call build_l(k1,v,l1,l2,l3,l4,al,al2,bet,gm)
   cut2%gm= gm
   cut2%rat1= (den0%m2+den1%m2-k1k1/3.d0) 
   if (dmr.eq.-1) then 
    call contr(k1,v,k1v)
    cut2%rat1t= k1v*(2.d0*den0%m2+4.d0*den1%m2-k1k1) 
   else
    cut2%rat1t= 0.d0
   endif
   dd0= m02
   dd1= m12-k1k1
   do i= 0,3
    cut2%l3(i)= l3(i)
    cut2%l4(i)= l4(i)
    cut2%k1(i)= k1(i)
    cut2%v(i) = v(i)
   enddo
   cf0     = -0.25d0/gm*(m02*c1(p))
   cut2%cf0= cf0 
   yv0     =  (dd1-dd0)*c1(p)/gm
   if (dmr.le.0) then
    a0= (sigma(p)+c1(p))
    ak= sigma(p)*a0
    a1= -sigma(p)
    cfsigma= a0*m02+a1*m12+ak*k1k1
    cfsigma= -0.25d0/gm*cfsigma
    cut2%cfsigma = cfsigma
    a0= -c1(p)
    a1=  c1(p)
    ak=-(c1(p)+2.d0*sigma(p))
    yvsigma= a0*m02+a1*m12+ak*k1k1
    yvsigma= yvsigma/gm
   endif
   if (dmr.le.1) then
    a0= (lambda(p)+c1(p))
    ak= lambda(p)*a0
    a1= -lambda(p)
    cflambda= a0*m02+a1*m12+ak*k1k1
    cflambda= -0.25d0/gm*cflambda
    cut2%cflambda= cflambda
    a0= -c1(p)
    a1=  c1(p)
    ak=-(c1(p)+2.d0*lambda(p))
    yvlambda= a0*m02+a1*m12+ak*k1k1
    yvlambda= yvlambda/gm
   endif
   if (dmr.eq.-1) then
! comment: later on b_4-b_8 not to be computed with rational=.true. 
    cc3= cf0**3
    if  (abs(cc3).gt.my_tiny(p)) then
     phi= atan2(dimag(cc3),dreal(cc3))
    else
     phi= 0.d0
    endif
    tau= exp(ci(p)/6.d0*(pi(p)-phi))
    cut2%tau= tau
    nsol= 9
    cc3= cflambda**2
    if  (abs(cc3).gt.my_tiny(p)) then
     phi= atan2(dimag(cc3),dreal(cc3))
    else
     phi= 0.d0
    endif
    taul= exp(ci(p)/4.d0*(pi(p)-phi))
    cut2%taul= taul
    do i= 0,3
     q2caus= -p0(i)*c1(p)+yv0*v(i)
     y4= c1(p)
     y3= cf0/y4 
     cut2%q(i,1) = q2caus+y3*tau*l3(i)+y4/tau*l4(i)   
!
     y4= cexp3(p)**2
     y3= cf0/y4 
     cut2%q(i,2) = q2caus+y3*tau*l3(i)+y4/tau*l4(i)   
!
     y4= cexp3(p)**4
     y3= cf0/y4 
     cut2%q(i,3) = q2caus+y3*tau*l3(i)+y4/tau*l4(i)   
!
     y3= c1(p)
     y4= cf0/y3 
     cut2%q(i,4) = q2caus+y3*l3(i)/tau+y4*tau*l4(i)   
!
     y3= cexp3(p)**2
     y4= cf0/y3 
     cut2%q(i,5) = q2caus+y3*l3(i)/tau+y4*tau*l4(i)   
!
     q2caus= -p0(i)*c1(p)+lambda(p)*c1(p)*k1(i)+yvlambda*v(i)
     y4= c1(p)
     y3= cflambda/y4 
     cut2%q(i,6) = q2caus+y3*taul*l3(i)+y4/taul*l4(i)   
!
     cut2%q(i,7) = q2caus-y3*taul*l3(i)-y4/taul*l4(i)   
!
     y3= c1(p)
     y4= cflambda/y3 
     cut2%q(i,8) = q2caus+y3*l3(i)/taul+y4*taul*l4(i)   
!
     q2caus= -p0(i)*c1(p)+sigma(p)*c1(p)*k1(i)+yvsigma*v(i)
     y4= c1(p)
     y3= cfsigma/y4 
     cut2%q(i,9) = q2caus+y3*taul*l3(i)+y4/taul*l4(i)   
    enddo
   elseif (dmr.eq.0) then
    cc3= cf0**3
    if  (abs(cc3).gt.my_tiny(p)) then
     phi= atan2(dimag(cc3),dreal(cc3))
    else
     phi= 0.d0
    endif
    tau= exp(ci(p)/6.d0*(pi(p)-phi))
    cut2%tau= tau
    if (rational) then
     nsol= 1
     do i= 0,3
      q2caus= -p0(i)*c1(p)+yv0*v(i)
      y4= c1(p)
      y3= cf0/y4 
      cut2%q(i,1) = q2caus+y3*tau*l3(i)+y4/tau*l4(i)   
     enddo
    else
     nsol= 9
     cc3= cflambda**2
     if  (abs(cc3).gt.my_tiny(p)) then
      phi= atan2(dimag(cc3),dreal(cc3))
     else
      phi= 0.d0
     endif
     taul= exp(ci(p)/4.d0*(pi(p)-phi))
     cut2%taul= taul
     do i= 0,3
      q2caus= -p0(i)*c1(p)+yv0*v(i)
      y4= c1(p)
      y3= cf0/y4 
      cut2%q(i,1) = q2caus+y3*tau*l3(i)+y4/tau*l4(i)   
!
      y4= cexp3(p)**2
      y3= cf0/y4 
      cut2%q(i,2) = q2caus+y3*tau*l3(i)+y4/tau*l4(i)   
!
      y4= cexp3(p)**4
      y3= cf0/y4 
      cut2%q(i,3) = q2caus+y3*tau*l3(i)+y4/tau*l4(i)   
!
      y3= c1(p)
      y4= cf0/y3 
      cut2%q(i,4) = q2caus+y3*l3(i)/tau+y4*tau*l4(i)   
!
      y3= cexp3(p)**2
      y4= cf0/y3 
      cut2%q(i,5) = q2caus+y3*l3(i)/tau+y4*tau*l4(i)   
!
      q2caus= -p0(i)*c1(p)+lambda(p)*c1(p)*k1(i)+yvlambda*v(i)
      y4= c1(p)
      y3= cflambda/y4 
      cut2%q(i,6) = q2caus+y3*taul*l3(i)+y4/taul*l4(i)   
!
      cut2%q(i,7) = q2caus-y3*taul*l3(i)-y4/taul*l4(i)   
!
      y3= c1(p)
      y4= cflambda/y3 
      cut2%q(i,8) = q2caus+y3*l3(i)/taul+y4*taul*l4(i)   
!
      q2caus= -p0(i)*c1(p)+sigma(p)*c1(p)*k1(i)+yvsigma*v(i)
      y4= c1(p)
      y3= cfsigma/y4 
      cut2%q(i,9) = q2caus+y3*taul*l3(i)+y4/taul*l4(i)   
     enddo
    endif
   elseif (dmr.eq.1) then
    nsol= 4
    cc3= cf0**2
    if  (abs(cc3).gt.my_tiny(p)) then
     phi= atan2(dimag(cc3),dreal(cc3))
    else
     phi= 0.d0
    endif
    tau= exp(ci(p)/4.d0*(pi(p)-phi))
    cut2%tau= tau
    taul= c1(p)
    cut2%taul= taul
    do i= 0,3
     q2caus= -p0(i)*c1(p)+yv0*v(i)
     y4= c1(p)
     y3= cf0/y4 
     cut2%q(i,1) = q2caus+y3*tau*l3(i)+y4/tau*l4(i)   
!
     y4= cexp1(p)
     y3= cf0/y4 
     cut2%q(i,2) = q2caus+y3*tau*l3(i)+y4/tau*l4(i)   
!
     y3= c1(p)
     y4= cf0/y3 
     cut2%q(i,3) = q2caus+y3*l3(i)/tau+y4*tau*l4(i)   
!
     q2caus= -p0(i)*c1(p)+lambda(p)*c1(p)*k1(i)+yvlambda*v(i)
     y4= c1(p)
     y3= cflambda/y4 
     cut2%q(i,4) = q2caus+y3*taul*l3(i)+y4/taul*l4(i)   
    enddo
   elseif (dmr.eq.2) then
    nsol= 1
    tau = c1(p)
    cut2%tau= tau
    do i= 0,3
     q2caus= -p0(i)*c1(p)+yv0*v(i)
     y4= c1(p)
     y3= cf0/y4 
     cut2%q(i,1) = q2caus+y3*tau*l3(i)+y4/tau*l4(i)   
    enddo
   endif
!
!  computing all denominators at the solutions
! 
   do i= 1,nsol
    do kk= 1,dmns
     if (kk.eq.den0%i.or. &
         kk.eq.den1%i) then
      vden(kk,i)= c0(p)
     else
      call load_vden(cut2%q(:,i),kk,i,p0,den0%m2)
     endif 
    enddo
   enddo
  end subroutine dp_cutting2  
!
  subroutine dp_cutting1_oldbase(den0,cut1,dmr)
   use dimensions
   include 'cts_dpr.h' 
    :: p,a,b,c
   type(propagator), intent(in) :: den0
   type(solcut1), intent(out) :: cut1
   include 'cts_dpr.h'
    , dimension(0:3) :: p0,v,k
   include 'cts_dpc.h' 
    :: m02
   include 'cts_dpc.h' 
    :: al1,al2,bet,gm
   include 'cts_dpc.h' 
    :: cf0
   integer :: nsol,ky
   integer, intent(in) :: dmr
   include 'cts_dpc.h'
    , dimension(0:3)  :: l1,l2,l3,l4
   logical :: computing=.true.
   save computing,v,k
   if (computing) then
     computing=.false.
!    define the arbitrary massless 4-vectors v and k
!-comment
     v(0)= 1.d0
     v(1)= v(0)/root3(p)
     v(2)= v(0)/root3(p)
     v(3)= v(0)/root3(p)
!
     k(0)= 1.d0
     k(1)=-k(0)/root3(p)
     k(2)=-k(0)/root3(p)
     k(3)=-k(0)/root3(p)
!-comment
   endif
   p0 = den0%p 
   m02= den0%m2-qt2 
   call build_l(v,k,l1,l2,l3,l4,al1,al2,bet,gm)
   cut1%gm= gm
! 
   cf0      = -0.25d0/gm*(m02*c1(p))
   cut1%cf0 = cf0
   if (dmr.eq.0) then
    nsol= 5
    do i= 0,3
     cut1%q(i,1) = -p0(i)+cf0*tau11(p)*l3(i)+l4(i)/tau11(p)
     cut1%q(i,2) = -p0(i)-cf0*tau11(p)*l3(i)-l4(i)/tau11(p)
     cut1%q(i,3) =  cut1%q(i,1)+v(i)
     cut1%q(i,4) =  cut1%q(i,1)+k(i)
     cut1%q(i,5) = -p0(i)+cexpk1(p)*tau12(p)*tau11(p)*l3(i) &
                    +cf0/cexpk1(p)*l4(i)/tau11(p)/tau12(p)
     cut1%v(i)   = v(i)
     cut1%k(i)   = k(i)
     cut1%l3(i)  = l3(i)
     cut1%l4(i)  = l4(i)
    enddo
   elseif (dmr.eq.1) then
    nsol= 1
    do i= 0,3
     cut1%q(i,1) = -p0(i)+cf0*tau11(p)*l3(i)+l4(i)/tau11(p)
    enddo
   endif
!
!  computing all denominators at the solutions
! 
   do i= 1,nsol
    do kk= 1,dmns
     if (kk.eq.den0%i) then
      vden(kk,i)= c0(p)
     else
      call load_vden(cut1%q(:,i),kk,i,p0,den0%m2)
     endif 
    enddo
   enddo
  end subroutine dp_cutting1_oldbase
!
  subroutine dp_cutting1_newbase(den0,cut1,dmr)
   use dimensions
   include 'cts_dpr.h' 
    :: p,a,b,c
   type(propagator), intent(in) :: den0
   type(solcut1), intent(out) :: cut1
   integer, intent(in) ::  dmr
   include 'cts_dpr.h'
    , dimension(0:3) :: p0,v,k
   include 'cts_dpc.h' 
    :: m02,apar,root 
   include 'cts_dpc.h' 
    :: al1,al2,bet,gm
   include 'cts_dpc.h' 
    :: cf0
   include 'cts_dpc.h'
    , dimension(0:3)  :: l1,l2,l3,l4
   integer :: ky,nsol
   logical :: computing=.true.
   save computing,v,k,l3,l4 
   if (computing) then
!    define the arbitrary massless 4-vectors v and k
     computing=.false.
!-comment
     k(0) = 1.d0
     k(1) = 0.d0 
     k(2) = 0.d0
     k(3) = 0.d0

     v(0) = 0.d0
     v(1) = 1.d0 
     v(2) = 0.d0
     v(3) = 0.d0

     l3(0)= 0.d0
     l3(1)= 0.d0 
     l3(2)= 1.d0
     l3(3)= 0.d0

     l4(0)= 0.d0
     l4(1)= 0.d0 
     l4(2)= 0.d0
     l4(3)= 1.d0
!-comment
   endif
   do i= 0,3
    cut1%v(i)   = v(i)
    cut1%k(i)   = k(i)
    cut1%l3(i)  = l3(i)
    cut1%l4(i)  = l4(i)
   enddo
   p0 = den0%p 
   m02= den0%m2-qt2 
   if (dmr.eq.-1) then
    cut1%rat1= (den0%m2)**2
   else
    cut1%rat1= 0.d0
   endif
   ky= 0
 1 ky= ky+1
   apar= sqrt(c1(p)*ky+qt2)
   root= sqrt((apar**2+m02)/3.d0)
   if (abs(root).le.abs(c1(p)/10)) goto 1
   cut1%apar= apar
   cut1%root= root
   if ((dmr.eq.0).or.(dmr.eq.-1)) then
    nsol= 5
!
    cut1%q(0,1) = -p0(0)+ci(p)*apar
    cut1%q(1,1) = -p0(1)+ci(p)*root
    cut1%q(2,1) = -p0(2)+ci(p)*root
    cut1%q(3,1) = -p0(3)+ci(p)*root
!
    cut1%q(0,2) = -p0(0)-ci(p)*apar
    cut1%q(1,2) = -p0(1)-ci(p)*root
    cut1%q(2,2) = -p0(2)-ci(p)*root
    cut1%q(3,2) = -p0(3)-ci(p)*root
!
    cut1%q(0,3) = -p0(0)+ci(p)*apar
    cut1%q(1,3) = -p0(1)-ci(p)*root
    cut1%q(2,3) = -p0(2)+ci(p)*root
    cut1%q(3,3) = -p0(3)+ci(p)*root
!
    cut1%q(0,4) = -p0(0)+ci(p)*apar
    cut1%q(1,4) = -p0(1)+ci(p)*root
    cut1%q(2,4) = -p0(2)-ci(p)*root
    cut1%q(3,4) = -p0(3)+ci(p)*root
!
    cut1%q(0,5) = -p0(0)+ci(p)*apar
    cut1%q(1,5) = -p0(1)+ci(p)*root
    cut1%q(2,5) = -p0(2)+ci(p)*root
    cut1%q(3,5) = -p0(3)-ci(p)*root
   elseif (dmr.eq.1) then
    nsol= 1
!
    cut1%q(0,1) = -p0(0)+ci(p)*apar
    cut1%q(1,1) = -p0(1)+ci(p)*root
    cut1%q(2,1) = -p0(2)+ci(p)*root
    cut1%q(3,1) = -p0(3)+ci(p)*root
   endif
!
!  computing all denominators at the solutions
! 
   do i= 1,nsol
    do kk= 1,dmns
     if (kk.eq.den0%i) then
      vden(kk,i)= c0(p)
     else
      call load_vden(cut1%q(:,i),kk,i,p0,den0%m2)
     endif 
    enddo
   enddo
  end subroutine dp_cutting1_newbase 
!
  subroutine mp_cutting4(den0,den1,den2,den3,cut4)
   use dimensions
   include 'cts_mpr.h' 
    :: p
   type(mp_propagator), intent(in) :: den0,den1,den2,den3
   type(mp_solcut4) ,intent(out) :: cut4
   include 'cts_mpr.h'
    , dimension(0:3) :: p0,p1,p2,p3,k1,k2,k3
   include 'cts_mpr.h'
    :: k1k1,k2k2,k3k3
   include 'cts_mpc.h'
    :: m02,m12,m22,m32
   include 'cts_mpc.h'
    :: dd0,dd1,dd2,dd3
   include 'cts_mpc.h'
    :: l1k3,l2k3,l3k3,l4k3,x10,x20,x3p,x4p,x3m,x4m
   include 'cts_mpc.h'
    :: cc,ca3,cb3,root,cb3p,cb3m,al1,al2,bet,gm,z
   include 'cts_mpc.h'
    , dimension(0:3) :: l1,l2,l3,l4
   integer :: k
!
   p0= den0%p 
   p1= den1%p 
   p2= den2%p 
   p3= den3%p 
!
   m02= den0%m2-mpqt2 
   m12= den1%m2-mpqt2 
   m22= den2%m2-mpqt2 
   m32= den3%m2-mpqt2 
!
   do k= 0,3
    k1(k)= p1(k)-p0(k)
    k2(k)= p2(k)-p0(k)
    k3(k)= p3(k)-p0(k)
   enddo
!
   call contr(k1,k1,k1k1)
   call contr(k2,k2,k2k2)
   call contr(k3,k3,k3k3)
   call build_l(k1,k2,l1,l2,l3,l4,al1,al2,bet,gm)
!
   call contr(l1,k3,l1k3)
   call contr(l2,k3,l2k3)
   call contr(l3,k3,l3k3)
   call contr(l4,k3,l4k3)
   z= bet/gm
   dd0= m02
   dd1= m12-k1k1
   dd2= m22-k2k2
   dd3= m32-k3k3
   x10= z*(dd2-al2*dd1-dd0*(1.d0-al2))
   x20= z*(dd1-al1*dd2-dd0*(1.d0-al1))
   cc = 0.25d0*(x10*x20-dd0/gm)
   ca3= -l3k3/l4k3
   cb3= (dd3-dd0-2.d0*x10*l1k3-2.d0*x20*l2k3)/2.d0/l4k3
   root= sqrt(cb3**2+4.d0*cc*ca3)
   cb3p= cb3+root
   cb3m= cb3-root
   if (abs(cb3m).ge.abs(cb3p)) then 
     x3p= (-cb3m)/2.d0/ca3
     x3m= -cc/ca3/x3p
     x4m= (cb3m)/2.d0
     x4p= -cc*ca3/x4m
! 
!     x4p= cc/x3p
!     x4m= cc/x3m
!
   else
     x3m= (-cb3p)/2.d0/ca3
     x3p= -cc/ca3/x3m
     x4p= (cb3p)/2.d0
     x4m= -cc*ca3/x4p
!      
!     x4p= cc/x3p
!     x4m= cc/x3m
!
   endif   
!
!  the 2 solutions, the basis and the vector t
! 
!   q(1)= q^+
!   q(2)= q^-
!
   do i= 0,3
    cut4%q(i,1)= -p0(i)+x10*l1(i)+x20*l2(i)+x3p*l3(i)+x4p*l4(i)
    cut4%q(i,2)= -p0(i)+x10*l1(i)+x20*l2(i)+x3m*l3(i)+x4m*l4(i)
    cut4%l1(i)= l1(i)
    cut4%l2(i)= l2(i)
    cut4%l3(i)= l3(i)
    cut4%l4(i)= l4(i)
    cut4%t(i) = l3(i)*l4k3-l4(i)*l3k3
   enddo
!
!  computing all denominators at the solutions
! 
   do i= 1,2
    do kk= 1,dmns
     if (kk.eq.den0%i.or. &
         kk.eq.den1%i.or. &
         kk.eq.den2%i.or. &
         kk.eq.den3%i) then
      mp_vden(kk,i)= c0(p)
     else
      call load_vden(cut4%q(:,i),kk,i,p0,den0%m2)
     endif 
    enddo
   enddo
  end subroutine mp_cutting4  
!
  subroutine mp_cutting3(den0,den1,den2,cut3,dmr)
   use dimensions
   include 'cts_mpr.h'
    :: p
   type(mp_propagator), intent(in) :: den0,den1,den2
   type(mp_solcut3), intent(out) :: cut3
   integer, intent(in) ::  dmr
   include 'cts_mpr.h'
    , dimension(0:3) :: p0,p1,p2,k1,k2
   include 'cts_mpr.h'
    :: k1k1,k2k2,k1k2,phi,rpart,ipart
   include 'cts_mpc.h'
    :: m02,m12,m22
   include 'cts_mpc.h'
    :: dd0,dd1,dd2
   include 'cts_mpc.h'
    :: x10,x20,x3,x4,q3caus
   include 'cts_mpc.h'
    :: cc,tau,ca3,al1,al2,bet,gm,z,cc4 
   include 'cts_mpc.h'
    , dimension(0:3)  :: l1,l2,l3,l4
   integer :: k,nsol
   p0= den0%p 
   p1= den1%p 
   p2= den2%p 
!
   m02= den0%m2-mpqt2 
   m12= den1%m2-mpqt2 
   m22= den2%m2-mpqt2 
!
   do k= 0,3
    k1(k)= p1(k)-p0(k)
    k2(k)= p2(k)-p0(k)
   enddo
!
   call contr(k1,k1,k1k1)
   call contr(k2,k2,k2k2)
   call build_l(k1,k2,l1,l2,l3,l4,al1,al2,bet,gm)
   if (dmr.eq.-1) then
    call contr(k1,k2,k1k2)
    cut3%rat1= k1k2-k1k1-k2k2+2.d0*(den0%m2+den1%m2+den2%m2)
   else
    cut3%rat1= 0.d0
   endif
   z= bet/gm
   dd0= m02
   dd1= m12-k1k1
   dd2= m22-k2k2
   x10= z*(dd2-al2*dd1-dd0*(1.d0-al2))
   x20= z*(dd1-al1*dd2-dd0*(1.d0-al1))
   cc = 0.25d0*(x10*x20-dd0/gm)
   cut3%gm= gm
   cut3%cc= cc
   do i= 0,3
    cut3%l1(i)= l1(i)
    cut3%l2(i)= l2(i)
    cut3%l3(i)= l3(i)
    cut3%l4(i)= l4(i)
   enddo
   cc4= cc**4
   if  (abs(cc4).gt.my_tiny(p)) then
    rpart= (cc4+conjg(cc4))/2.d0  
    ipart= (cc4-conjg(cc4))/(2.d0*ci(p))  
    phi= atan2(ipart,rpart) 
   else
    phi= 0.d0
   endif
   tau= exp(ci(p)/8.d0*(pi(p)-phi))
   cut3%tau= tau
   if (dmr.eq.-1) then
! comment: later on c_5 and c_6 not to be computed with rational=.true. 
    nsol= 7
    do i= 0,3
     q3caus= -p0(i)+x10*l1(i)+x20*l2(i)
     x4= c1(p)
     x3= cc/x4
     cut3%q(i,1) = q3caus+x3*tau*l3(i)+x4/tau*l4(i)   
!
     x4= (cexp4(p))**2
     x3= cc/x4
     cut3%q(i,2) = q3caus+x3*tau*l3(i)+x4/tau*l4(i)   
!
     x4= (cexp4(p))**4
     x3= cc/x4
     cut3%q(i,3) = q3caus+x3*tau*l3(i)+x4/tau*l4(i)   
!
     x4= (cexp4(p))**6
     x3= cc/x4
     cut3%q(i,4) = q3caus+x3*tau*l3(i)+x4/tau*l4(i)   
!
     x3= c1(p)
     x4= cc/x3
     cut3%q(i,5) = q3caus+x3/tau*l3(i)+x4*tau*l4(i)   
!
     x3= (cexp4(p))**2
     x4= cc/x3
     cut3%q(i,6) = q3caus+x3/tau*l3(i)+x4*tau*l4(i)   
!
     x3= (cexp4(p))**4
     x4= cc/x3
     cut3%q(i,7) = q3caus+x3/tau*l3(i)+x4*tau*l4(i)   
    enddo
   elseif (dmr.eq.0) then
    if (rational) then
     nsol= 3
     do i= 0,3
      q3caus= -p0(i)+x10*l1(i)+x20*l2(i)
      x4= c1(p)
      x3= cc/x4
      cut3%q(i,1) = q3caus+x3*tau*l3(i)+x4/tau*l4(i)   
!
      x4= (cexp4(p))**4
      x3= cc/x4
      cut3%q(i,2) = q3caus+x3*tau*l3(i)+x4/tau*l4(i)   
!
      x3= c1(p)
      x4= cc/x3
      cut3%q(i,3) = q3caus+x3/tau*l3(i)+x4*tau*l4(i)   
     enddo
    else
     nsol= 7
     do i= 0,3
      q3caus= -p0(i)+x10*l1(i)+x20*l2(i)
      x4= c1(p)
      x3= cc/x4
      cut3%q(i,1) = q3caus+x3*tau*l3(i)+x4/tau*l4(i)   
!
      x4= (cexp4(p))**2
      x3= cc/x4
      cut3%q(i,2) = q3caus+x3*tau*l3(i)+x4/tau*l4(i)   
!
      x4= (cexp4(p))**4
      x3= cc/x4
      cut3%q(i,3) = q3caus+x3*tau*l3(i)+x4/tau*l4(i)   
!
      x4= (cexp4(p))**6
      x3= cc/x4
      cut3%q(i,4) = q3caus+x3*tau*l3(i)+x4/tau*l4(i)   
!
      x3= c1(p)
      x4= cc/x3
      cut3%q(i,5) = q3caus+x3/tau*l3(i)+x4*tau*l4(i)   
!
      x3= (cexp4(p))**2
      x4= cc/x3
      cut3%q(i,6) = q3caus+x3/tau*l3(i)+x4*tau*l4(i)   
!
      x3= (cexp4(p))**4
      x4= cc/x3
      cut3%q(i,7) = q3caus+x3/tau*l3(i)+x4*tau*l4(i)   
     enddo
    endif
   elseif (dmr.eq.1) then
    if (rational) then
     nsol= 2
     do i= 0,3
      q3caus= -p0(i)+x10*l1(i)+x20*l2(i)
      x4= c1(p)
      x3= cc/x4
      cut3%q(i,1) = q3caus+x3*tau*l3(i)+x4/tau*l4(i)   
!
      x4= (cexp4(p))**4
      x3= cc/x4
      cut3%q(i,2) = q3caus+x3*tau*l3(i)+x4/tau*l4(i)   
     enddo
    else
     nsol= 5
     do i= 0,3
      q3caus= -p0(i)+x10*l1(i)+x20*l2(i)
      x4= c1(p)
      x3= cc/x4
      cut3%q(i,1) = q3caus+x3*tau*l3(i)+x4/tau*l4(i)   
!
      x4= (cexp4(p))**4
      x3= cc/x4
      cut3%q(i,2) = q3caus+x3*tau*l3(i)+x4/tau*l4(i)   
!
      x4= (cexp4(p))**6
      x3= cc/x4
      cut3%q(i,3) = q3caus+x3*tau*l3(i)+x4/tau*l4(i)   
!
      x3= c1(p)
      x4= cc/x3
      cut3%q(i,4) = q3caus+x3/tau*l3(i)+x4*tau*l4(i)   
!
      x3= (cexp4(p))**4
      x4= cc/x3
      cut3%q(i,5) = q3caus+x3/tau*l3(i)+x4*tau*l4(i)   
     enddo
    endif
   elseif (dmr.eq.2) then
    nsol= 3
    do i= 0,3
     q3caus= -p0(i)+x10*l1(i)+x20*l2(i)
     x4= c1(p)
     x3= cc/x4
     cut3%q(i,1) = q3caus+x3*tau*l3(i)+x4/tau*l4(i)   
!
     x4= (cexp4(p))**4
     x3= cc/x4
     cut3%q(i,2) = q3caus+x3*tau*l3(i)+x4/tau*l4(i)   
!
     x3= c1(p)
     x4= cc/x3
     cut3%q(i,3) = q3caus+x3/tau*l3(i)+x4*tau*l4(i)   
    enddo
   else
    nsol= 1
    do i= 0,3
     q3caus= -p0(i)+x10*l1(i)+x20*l2(i)
     x4= c1(p)
     x3= cc/x4
     cut3%q(i,1) = q3caus+x3*tau*l3(i)+x4/tau*l4(i)   
    enddo
   endif
!
!  computing all denominators at the solutions
! 
   do i= 1,nsol  
    do kk= 1,dmns
     if (kk.eq.den0%i.or. &
         kk.eq.den1%i.or. &
         kk.eq.den2%i) then
      mp_vden(kk,i)= c0(p)
     else
      call load_vden(cut3%q(:,i),kk,i,p0,den0%m2)
     endif 
    enddo
   enddo
   end subroutine mp_cutting3  
!
  subroutine mp_cutting2(den0,den1,cut2,dmr)
   use dimensions
   include 'cts_mpr.h'
    :: p,a,b,c
   type(mp_propagator), intent(in) :: den0,den1
   type(mp_solcut2), intent(out) :: cut2
   integer, intent(in) ::  dmr
   include 'cts_mpr.h'
    , dimension(0:3) :: p0,p1,k1,v
   include 'cts_mpr.h'
    :: k1k1,k1v,factor,phi,rpart,ipart
   include 'cts_mpc.h'
    :: m02,m12,ak,a0,a1
   include 'cts_mpc.h'
    :: dd0,dd1
   integer :: k,nsol
   include 'cts_mpc.h'
    :: yv0,yvsigma,yvlambda,y3,y4
   include 'cts_mpc.h'
    :: tau,taul,cc3,al,al2,bet,gm,cf0,cflambda,cfsigma,q2caus
   include 'cts_mpc.h'
    , dimension(0:3)  :: l1,l2,l3,l4

   p0= den0%p 
   p1= den1%p 
!
   m02= den0%m2-mpqt2 
   m12= den1%m2-mpqt2 
!
   do k= 0,3
    k1(k)= p1(k)-p0(k)
   enddo
!
   call contr(k1,k1,k1k1)
!
!  define the arbitrary massless 4-vector v
!
!-comment
    a= 1.d0   
    v(0)= sign(a,k1(0))         
    v(1)=-sign(a/root3(p),k1(1))
    v(2)=-sign(a/root3(p),k1(2))
    v(3)=-sign(a/root3(p),k1(3))
!-comment
   call build_l(k1,v,l1,l2,l3,l4,al,al2,bet,gm)
   cut2%gm= gm
   cut2%rat1= (den0%m2+den1%m2-k1k1/3.d0) 
   if (dmr.eq.-1) then
    call contr(k1,v,k1v)
    cut2%rat1t= k1v*(2.d0*den0%m2+4.d0*den1%m2-k1k1) 
   else
    cut2%rat1t= 0.d0
   endif
   dd0= m02
   dd1= m12-k1k1
   do i= 0,3
    cut2%l3(i)= l3(i)
    cut2%l4(i)= l4(i)
    cut2%k1(i)= k1(i)
    cut2%v(i) = v(i)
   enddo
!
   cf0     = -0.25d0/gm*(m02*c1(p))
   cut2%cf0= cf0 
   yv0     =  (dd1-dd0)*c1(p)/gm
   if (dmr.le.0) then
    a0= (sigma(p)+c1(p))
    ak= sigma(p)*a0
    a1= -sigma(p)
    cfsigma= a0*m02+a1*m12+ak*k1k1
    cfsigma= -0.25d0/gm*cfsigma
    cut2%cfsigma = cfsigma
    a0= -c1(p)
    a1=  c1(p)
    ak=-(c1(p)+2.d0*sigma(p))
    yvsigma= a0*m02+a1*m12+ak*k1k1
    yvsigma= yvsigma/gm
   endif
   if (dmr.le.1) then
    a0= (lambda(p)+c1(p))
    ak= lambda(p)*a0
    a1= -lambda(p)
    cflambda= a0*m02+a1*m12+ak*k1k1
    cflambda= -0.25d0/gm*cflambda
    cut2%cflambda= cflambda
    a0= -c1(p)
    a1=  c1(p)
    ak=-(c1(p)+2.d0*lambda(p))
    yvlambda= a0*m02+a1*m12+ak*k1k1
    yvlambda= yvlambda/gm
   endif
   if (dmr.eq.-1) then
! comment: later on b_4-b_8 not to be computed with rational=.true. 
    cc3= cf0**3
    if  (abs(cc3).gt.my_tiny(p)) then
     rpart= (cc3+conjg(cc3))/2.d0  
     ipart= (cc3-conjg(cc3))/(2.d0*ci(p))  
     phi= atan2(ipart,rpart)
    else
     phi= 0.d0
    endif
    tau= exp(ci(p)/6.d0*(pi(p)-phi))
    cut2%tau= tau
    nsol= 9
    cc3= cflambda**2
    if  (abs(cc3).gt.my_tiny(p)) then
     rpart= (cc3+conjg(cc3))/2.d0  
     ipart= (cc3-conjg(cc3))/(2.d0*ci(p))  
     phi= atan2(ipart,rpart)
    else
     phi= 0.d0
    endif
    taul= exp(ci(p)/4.d0*(pi(p)-phi))
    cut2%taul= taul
    do i= 0,3
     q2caus= -p0(i)*c1(p)+yv0*v(i)
     y4= c1(p)
     y3= cf0/y4 
     cut2%q(i,1) = q2caus+y3*tau*l3(i)+y4/tau*l4(i)   
!
     y4= cexp3(p)**2
     y3= cf0/y4 
     cut2%q(i,2) = q2caus+y3*tau*l3(i)+y4/tau*l4(i)   
!
     y4= cexp3(p)**4
     y3= cf0/y4 
     cut2%q(i,3) = q2caus+y3*tau*l3(i)+y4/tau*l4(i)   
!
     y3= c1(p)
     y4= cf0/y3 
     cut2%q(i,4) = q2caus+y3*l3(i)/tau+y4*tau*l4(i)   
!
     y3= cexp3(p)**2
     y4= cf0/y3 
     cut2%q(i,5) = q2caus+y3*l3(i)/tau+y4*tau*l4(i)   
!
     q2caus= -p0(i)*c1(p)+lambda(p)*c1(p)*k1(i)+yvlambda*v(i)
     y4= c1(p)
     y3= cflambda/y4 
     cut2%q(i,6) = q2caus+y3*taul*l3(i)+y4/taul*l4(i)   
!
     cut2%q(i,7) = q2caus-y3*taul*l3(i)-y4/taul*l4(i)   
!
     y3= c1(p)
     y4= cflambda/y3 
     cut2%q(i,8) = q2caus+y3*l3(i)/taul+y4*taul*l4(i)   
!
     q2caus= -p0(i)*c1(p)+sigma(p)*c1(p)*k1(i)+yvsigma*v(i)
     y4= c1(p)
     y3= cfsigma/y4 
     cut2%q(i,9) = q2caus+y3*taul*l3(i)+y4/taul*l4(i)   
    enddo
   elseif (dmr.eq.0) then
    cc3= cf0**3
    if  (abs(cc3).gt.my_tiny(p)) then
     rpart= (cc3+conjg(cc3))/2.d0  
     ipart= (cc3-conjg(cc3))/(2.d0*ci(p))  
     phi= atan2(ipart,rpart)
    else
     phi= 0.d0
    endif
    tau= exp(ci(p)/6.d0*(pi(p)-phi))
    cut2%tau= tau
    if (rational) then
     nsol= 1
     do i= 0,3
      q2caus= -p0(i)*c1(p)+yv0*v(i)
      y4= c1(p)
      y3= cf0/y4 
      cut2%q(i,1) = q2caus+y3*tau*l3(i)+y4/tau*l4(i)   
     enddo
    else
     nsol= 9
     cc3= cflambda**2
     if  (abs(cc3).gt.my_tiny(p)) then
      rpart= (cc3+conjg(cc3))/2.d0  
      ipart= (cc3-conjg(cc3))/(2.d0*ci(p))  
      phi= atan2(ipart,rpart)
     else
      phi= 0.d0
     endif
     taul= exp(ci(p)/4.d0*(pi(p)-phi))
     cut2%taul= taul
     do i= 0,3
      q2caus= -p0(i)*c1(p)+yv0*v(i)
      y4= c1(p)
      y3= cf0/y4 
      cut2%q(i,1) = q2caus+y3*tau*l3(i)+y4/tau*l4(i)   
!
      y4= cexp3(p)**2
      y3= cf0/y4 
      cut2%q(i,2) = q2caus+y3*tau*l3(i)+y4/tau*l4(i)   
!
      y4= cexp3(p)**4
      y3= cf0/y4 
      cut2%q(i,3) = q2caus+y3*tau*l3(i)+y4/tau*l4(i)   
!
      y3= c1(p)
      y4= cf0/y3 
      cut2%q(i,4) = q2caus+y3*l3(i)/tau+y4*tau*l4(i)   
!
      y3= cexp3(p)**2
      y4= cf0/y3 
      cut2%q(i,5) = q2caus+y3*l3(i)/tau+y4*tau*l4(i)   
!
      q2caus= -p0(i)*c1(p)+lambda(p)*c1(p)*k1(i)+yvlambda*v(i)
      y4= c1(p)
      y3= cflambda/y4 
      cut2%q(i,6) = q2caus+y3*taul*l3(i)+y4/taul*l4(i)   
!
      cut2%q(i,7) = q2caus-y3*taul*l3(i)-y4/taul*l4(i)   
!
      y3= c1(p)
      y4= cflambda/y3 
      cut2%q(i,8) = q2caus+y3*l3(i)/taul+y4*taul*l4(i)   
!
      q2caus= -p0(i)*c1(p)+sigma(p)*c1(p)*k1(i)+yvsigma*v(i)
      y4= c1(p)
      y3= cfsigma/y4 
      cut2%q(i,9) = q2caus+y3*taul*l3(i)+y4/taul*l4(i)   
     enddo
    endif
   elseif (dmr.eq.1) then
    nsol= 4
    cc3= cf0**2
    if  (abs(cc3).gt.my_tiny(p)) then
     rpart= (cc3+conjg(cc3))/2.d0  
     ipart= (cc3-conjg(cc3))/(2.d0*ci(p))  
     phi= atan2(ipart,rpart)
    else
     phi= 0.d0
    endif
    tau= exp(ci(p)/4.d0*(pi(p)-phi))
    cut2%tau= tau
    taul= c1(p)
    cut2%taul= taul
    do i= 0,3
     q2caus= -p0(i)*c1(p)+yv0*v(i)
     y4= c1(p)
     y3= cf0/y4 
     cut2%q(i,1) = q2caus+y3*tau*l3(i)+y4/tau*l4(i)   
!
     y4= cexp1(p)
     y3= cf0/y4 
     cut2%q(i,2) = q2caus+y3*tau*l3(i)+y4/tau*l4(i)   
!
     y3= c1(p)
     y4= cf0/y3 
     cut2%q(i,3) = q2caus+y3*l3(i)/tau+y4*tau*l4(i)   
!
     q2caus= -p0(i)*c1(p)+lambda(p)*c1(p)*k1(i)+yvlambda*v(i)
     y4= c1(p)
     y3= cflambda/y4 
     cut2%q(i,4) = q2caus+y3*taul*l3(i)+y4/taul*l4(i)   
    enddo
   elseif (dmr.eq.2) then
    nsol= 1
    tau = c1(p)
    cut2%tau= tau
    do i= 0,3
     q2caus= -p0(i)*c1(p)+yv0*v(i)
     y4= c1(p)
     y3= cf0/y4 
     cut2%q(i,1) = q2caus+y3*tau*l3(i)+y4/tau*l4(i)   
    enddo
   endif
!
!  computing all denominators at the solutions
! 
   do i= 1,nsol
    do kk= 1,dmns
     if (kk.eq.den0%i.or. &
         kk.eq.den1%i) then
      mp_vden(kk,i)= c0(p)
     else
      call load_vden(cut2%q(:,i),kk,i,p0,den0%m2)
     endif 
    enddo
   enddo
  end subroutine mp_cutting2  
!
  subroutine mp_cutting1_oldbase(den0,cut1,dmr)
   use dimensions
   include 'cts_mpr.h'
    :: p,a,b,c
   type(mp_propagator), intent(in) :: den0
   type(mp_solcut1), intent(out) :: cut1
   include 'cts_mpr.h'
    , dimension(0:3) :: p0,v,k
   include 'cts_mpc.h'
    :: m02
   include 'cts_mpc.h'
    :: al1,al2,bet,gm
   include 'cts_mpc.h'
    :: cf0
   integer :: nsol,ky
   integer, intent(in) :: dmr
   include 'cts_mpc.h'
    , dimension(0:3)  :: l1,l2,l3,l4
   logical :: computing=.true.
   save computing,v,k
   if (computing) then
     computing=.false.
!    define the arbitrary massless 4-vectors v and k
!-comment
     v(0)= 1.d0
     v(1)= v(0)/root3(p)
     v(2)= v(0)/root3(p)
     v(3)= v(0)/root3(p)
!
     k(0)= 1.d0
     k(1)=-k(0)/root3(p)
     k(2)=-k(0)/root3(p)
     k(3)=-k(0)/root3(p)
!-comment
   endif
   p0 = den0%p 
   m02= den0%m2-mpqt2 
   call build_l(v,k,l1,l2,l3,l4,al1,al2,bet,gm)
   cut1%gm= gm
! 
   cf0      = -0.25d0/gm*(m02*c1(p))
   cut1%cf0 = cf0
   if (dmr.eq.0) then
    nsol= 5
    do i= 0,3
     cut1%q(i,1) = -p0(i)+cf0*tau11(p)*l3(i)+l4(i)/tau11(p)
     cut1%q(i,2) = -p0(i)-cf0*tau11(p)*l3(i)-l4(i)/tau11(p)
     cut1%q(i,3) =  cut1%q(i,1)+v(i)
     cut1%q(i,4) =  cut1%q(i,1)+k(i)
     cut1%q(i,5) = -p0(i)+cexpk1(p)*tau12(p)*tau11(p)*l3(i) &
                    +cf0/cexpk1(p)*l4(i)/tau11(p)/tau12(p)
     cut1%v(i)   = v(i)
     cut1%k(i)   = k(i)
     cut1%l3(i)  = l3(i)
     cut1%l4(i)  = l4(i)
    enddo
   elseif (dmr.eq.1) then
    nsol= 1
    do i= 0,3
     cut1%q(i,1) = -p0(i)+cf0*tau11(p)*l3(i)+l4(i)/tau11(p)
    enddo
   endif
!
!  computing all denominators at the solutions
! 
   do i= 1,nsol
    do kk= 1,dmns
     if (kk.eq.den0%i) then
      mp_vden(kk,i)= c0(p)
     else
      call load_vden(cut1%q(:,i),kk,i,p0,den0%m2)
     endif 
    enddo
   enddo
  end subroutine mp_cutting1_oldbase
!
  subroutine mp_cutting1_newbase(den0,cut1,dmr)
   use dimensions
   include 'cts_mpr.h' 
    :: p,a,b,c
   type(mp_propagator), intent(in) :: den0
   type(mp_solcut1), intent(out) :: cut1
   integer, intent(in) ::  dmr
   include 'cts_mpr.h'
    , dimension(0:3) :: p0,v,k
   include 'cts_mpc.h' 
    :: m02,apar,root 
   include 'cts_mpc.h' 
    :: al1,al2,bet,gm
   include 'cts_mpc.h' 
    :: cf0
   include 'cts_mpc.h'
    , dimension(0:3)  :: l1,l2,l3,l4
   integer :: ky,nsol
   logical :: computing=.true.
   save computing,v,k,l3,l4 
   if (computing) then
!    define the arbitrary massless 4-vectors v and k
     computing=.false.
!-comment
     k(0) = 1.d0
     k(1) = 0.d0 
     k(2) = 0.d0
     k(3) = 0.d0

     v(0) = 0.d0
     v(1) = 1.d0 
     v(2) = 0.d0
     v(3) = 0.d0

     l3(0)= 0.d0
     l3(1)= 0.d0 
     l3(2)= 1.d0
     l3(3)= 0.d0

     l4(0)= 0.d0
     l4(1)= 0.d0 
     l4(2)= 0.d0
     l4(3)= 1.d0
!-comment
   endif
   do i= 0,3
    cut1%v(i)   = v(i)
    cut1%k(i)   = k(i)
    cut1%l3(i)  = l3(i)
    cut1%l4(i)  = l4(i)
   enddo
   p0 = den0%p 
   m02= den0%m2-mpqt2 
   if (dmr.eq.-1) then
    cut1%rat1= (den0%m2)**2
   else
    cut1%rat1= 0.d0
   endif
   ky= 0
 1 ky= ky+1
   apar= sqrt(c1(p)*ky+mpqt2)
   root= sqrt((apar**2+m02)/3.d0)
   if (abs(root).le.abs(c1(p)/10)) goto 1
   cut1%apar= apar
   cut1%root= root
   if ((dmr.eq.0).or.(dmr.eq.-1)) then
    nsol= 5
!
    cut1%q(0,1) = -p0(0)+ci(p)*apar
    cut1%q(1,1) = -p0(1)+ci(p)*root
    cut1%q(2,1) = -p0(2)+ci(p)*root
    cut1%q(3,1) = -p0(3)+ci(p)*root
!
    cut1%q(0,2) = -p0(0)-ci(p)*apar
    cut1%q(1,2) = -p0(1)-ci(p)*root
    cut1%q(2,2) = -p0(2)-ci(p)*root
    cut1%q(3,2) = -p0(3)-ci(p)*root
!
    cut1%q(0,3) = -p0(0)+ci(p)*apar
    cut1%q(1,3) = -p0(1)-ci(p)*root
    cut1%q(2,3) = -p0(2)+ci(p)*root
    cut1%q(3,3) = -p0(3)+ci(p)*root
!
    cut1%q(0,4) = -p0(0)+ci(p)*apar
    cut1%q(1,4) = -p0(1)+ci(p)*root
    cut1%q(2,4) = -p0(2)-ci(p)*root
    cut1%q(3,4) = -p0(3)+ci(p)*root
!
    cut1%q(0,5) = -p0(0)+ci(p)*apar
    cut1%q(1,5) = -p0(1)+ci(p)*root
    cut1%q(2,5) = -p0(2)+ci(p)*root
    cut1%q(3,5) = -p0(3)-ci(p)*root
   elseif (dmr.eq.1) then
    nsol= 1
!
    cut1%q(0,1) = -p0(0)+ci(p)*apar
    cut1%q(1,1) = -p0(1)+ci(p)*root
    cut1%q(2,1) = -p0(2)+ci(p)*root
    cut1%q(3,1) = -p0(3)+ci(p)*root
   endif
!
!  computing all denominators at the solutions
! 
   do i= 1,nsol
    do kk= 1,dmns
     if (kk.eq.den0%i) then
      mp_vden(kk,i)= c0(p)
     else
      call load_vden(cut1%q(:,i),kk,i,p0,den0%m2)
     endif 
    enddo
   enddo
  end subroutine mp_cutting1_newbase
!
 end module cuttings
!
 module coefficients
  include 'cts_mprec.h'
  use cuttings
  use dimensions
  use constants
  use def_solcut
  use def_mp_solcut
  use denominators
  use qt2value
  use tensor_operations
  implicit none
  private
  public :: get_coefficients,dp_allocate_arrays,mp_allocate_arrays
  integer, private :: ierr,icomp1,icomp2
!
! variables for the rational terms
!
  logical, private :: inf=.false.,inf_sv
  include 'cts_dpc.h'
   , public :: rat1,save_rat1 
  include 'cts_mpc.h'
   , public :: mp_rat1,save_mp_rat1
!
! variables for the 4-point sector:
!
  include 'cts_dpc.h'
   , dimension(:,:), public, allocatable :: dcoeff
  include 'cts_dpc.h'
   , dimension(:,:), public, allocatable :: save_dcoeff
  include 'cts_dpc.h'
   , dimension(:,:), public, allocatable :: tvec,p0vecd
  include 'cts_mpc.h'
   , dimension(:,:), public, allocatable :: mp_dcoeff
  include 'cts_mpc.h'
   , dimension(:,:), public, allocatable :: save_mp_dcoeff
  include 'cts_mpc.h'
   , dimension(:,:), public, allocatable :: mp_tvec,mp_p0vecd
!
! variables for the 3-point sector:
!
  include 'cts_dpc.h'
   , dimension(:,:), public, allocatable :: ccoeff
  include 'cts_dpc.h'
   , dimension(:,:), public, allocatable :: ccoeff_2
  include 'cts_dpc.h'
   , dimension(:,:), public, allocatable :: save_ccoeff
  include 'cts_dpc.h'
   , dimension(:,:), public, allocatable :: p0vecc
  include 'cts_dpc.h'
   , dimension(:,:), public, allocatable :: l3vec,l4vec    
  include 'cts_dpc.h'
   , dimension(:), public, allocatable   :: c4_rat1    
  include 'cts_mpc.h'
   , dimension(:,:), public, allocatable :: mp_ccoeff
  include 'cts_mpc.h'
   , dimension(:,:), public, allocatable :: mp_ccoeff_2
  include 'cts_mpc.h'
   , dimension(:,:), public, allocatable :: save_mp_ccoeff
  include 'cts_mpc.h'
   , dimension(:,:), public, allocatable :: mp_p0vecc
  include 'cts_mpc.h'
   , dimension(:,:), public, allocatable :: mp_l3vec,mp_l4vec 
  include 'cts_mpc.h'
   , dimension(:), public, allocatable   :: mp_c4_rat1 
!
! variables for the 2-point sector:
!
  include 'cts_dpc.h'
   , dimension(:,:), public, allocatable :: bcoeff
  include 'cts_dpc.h'
   , dimension(:)  , public, allocatable :: bcoeff_2
  include 'cts_dpc.h'
   , dimension(:,:), public, allocatable :: save_bcoeff
  include 'cts_dpc.h'
   , dimension(:,:), public, allocatable :: p0vecb
  include 'cts_dpc.h'
   , dimension(:,:), public, allocatable :: l5vec,l6vec    
  include 'cts_dpc.h'
   , dimension(:,:), public, allocatable :: vvecb
  include 'cts_dpc.h'
   , dimension(:), public, allocatable :: vveck1,b_rat1,b3_rat1
  include 'cts_mpc.h'
   , dimension(:,:), public, allocatable :: mp_bcoeff
  include 'cts_mpc.h'
   , dimension(:), public, allocatable :: mp_bcoeff_2
  include 'cts_mpc.h'
   , dimension(:,:), public, allocatable :: save_mp_bcoeff
  include 'cts_mpc.h'
   , dimension(:,:), public, allocatable :: mp_p0vecb
  include 'cts_mpc.h'
   , dimension(:,:), public, allocatable :: mp_l5vec,mp_l6vec 
  include 'cts_mpc.h'
   , dimension(:,:), public, allocatable :: mp_vvecb  
  include 'cts_mpc.h'
   , dimension(:), public, allocatable :: mp_vveck1,mp_b_rat1,mp_b3_rat1
!
! variables for the 1-point sector:
!
  include 'cts_dpc.h'
   , dimension(:,:), public, allocatable :: acoeff
  include 'cts_dpc.h'
   , dimension(:), public, allocatable :: acoeff_2
  include 'cts_dpc.h'
   , dimension(:,:), public, allocatable :: save_acoeff
  include 'cts_dpc.h'
   , dimension(:,:), public, allocatable :: p0veca
  include 'cts_dpc.h'
   , dimension(:,:), public, allocatable :: l7vec,l8vec    
  include 'cts_dpc.h'
   , dimension(:,:), public, allocatable :: vveca
  include 'cts_dpc.h'
   , dimension(:,:), public, allocatable :: kvec
  include 'cts_dpc.h'
   , dimension(:), public, allocatable   :: a_rat1
  include 'cts_mpc.h'
   , dimension(:,:), public, allocatable :: mp_acoeff
  include 'cts_mpc.h'
   , dimension(:,:), public, allocatable :: save_mp_acoeff
  include 'cts_mpc.h'
   , dimension(:,:), public, allocatable :: mp_p0veca
  include 'cts_mpc.h'
   , dimension(:,:), public, allocatable :: mp_l7vec,mp_l8vec 
  include 'cts_mpc.h'
   , dimension(:,:), public, allocatable :: mp_vveca  
  include 'cts_mpc.h'
   , dimension(:,:), public, allocatable :: mp_kvec
  include 'cts_mpc.h'
   , dimension(:), public, allocatable   :: mp_a_rat1
!
  save dcoeff,tvec,p0veca,p0vecb,p0vecc,p0vecd
  save ccoeff,ccoeff_2,l3vec,l4vec,c4_rat1
  save bcoeff,bcoeff_2,l5vec,l6vec
  save vveca,vvecb,vveck1,b_rat1,b3_rat1
  save acoeff,acoeff_2,kvec,a_rat1,l7vec,l8vec
  save mp_dcoeff,mp_tvec,mp_p0veca,mp_p0vecb,mp_p0vecc,mp_p0vecd
  save mp_ccoeff,mp_ccoeff_2,mp_l3vec,mp_l4vec,mp_c4_rat1
  save mp_bcoeff,mp_bcoeff_2,mp_l5vec,mp_l6vec
  save mp_vveca,mp_vvecb,mp_vveck1,mp_b_rat1,mp_b3_rat1
  save mp_acoeff,mp_kvec,mp_a_rat1
  save save_dcoeff,save_ccoeff,save_bcoeff,save_acoeff
  save save_mp_dcoeff,save_mp_ccoeff,save_mp_bcoeff,save_mp_acoeff
  save rat1,save_rat1,mp_rat1,save_mp_rat1,mp_l7vec,mp_l8vec 
!
  interface get_coefficients
    module procedure dp_get_coefficients
    module procedure mp_get_coefficients
  end interface!get_coefficients
!
  interface getd
    module procedure dp_getd
    module procedure mp_getd
  end interface!getd
!
  interface getc
    module procedure dp_getc
    module procedure mp_getc
  end interface!getc
!
  interface getb
    module procedure dp_getb
    module procedure mp_getb
  end interface!getb
!
  interface geta
    module procedure dp_geta_newbase
    module procedure mp_geta_newbase
  end interface!geta
!
  interface numd
    module procedure dp_numd
    module procedure mp_numd
  end interface!numd
!
  interface numc
    module procedure dp_numc
    module procedure mp_numc
  end interface!numc
!
  interface numb
    module procedure dp_numb
    module procedure mp_numb
  end interface!numb
!
  interface numa
    module procedure dp_numa
    module procedure mp_numa
  end interface!numa
!
  interface numfuncrec
    module procedure dp_numfuncrec
    module procedure mp_numfuncrec
  end interface!numfuncrec
!
  interface test
    module procedure dp_test
    module procedure mp_test
  end interface!test
!
  interface put_dcoeff
    module procedure dp_put_dcoeff
    module procedure mp_put_dcoeff
  end interface!put_dcoeff
!
  interface get_dcoeff
    module procedure dp_get_dcoeff
    module procedure mp_get_dcoeff
  end interface!get_dcoeff
!
  interface put_ccoeff
    module procedure dp_put_ccoeff
    module procedure mp_put_ccoeff
  end interface!put_ccoeff
!
  interface get_ccoeff
    module procedure dp_get_ccoeff
    module procedure mp_get_ccoeff
  end interface!get_ccoeff
!
  interface put_bcoeff
    module procedure dp_put_bcoeff
    module procedure mp_put_bcoeff
  end interface!put_bcoeff
!
  interface get_bcoeff
    module procedure dp_get_bcoeff
    module procedure mp_get_bcoeff
  end interface!get_bcoeff
!
  interface put_acoeff
    module procedure dp_put_acoeff
    module procedure mp_put_acoeff
  end interface!put_acoeff
!
  interface get_acoeff
    module procedure dp_get_acoeff
    module procedure mp_get_acoeff
  end interface!get_acoeff
!
  contains
!
  subroutine dp_allocate_arrays(np)
  integer, intent(in) :: np
  if (np.ge.4) call dp_allocate_vectorsd
  if (np.ge.3) call dp_allocate_vectorsc
  if (np.ge.2) call dp_allocate_vectorsb
  if (np.ge.1) call dp_allocate_vectorsa
  end subroutine dp_allocate_arrays
!
  subroutine dp_allocate_vectorsd
   ierr= -1
   allocate      (p0vecd(0:3,dmns_d), stat=ierr)
   allocate      (dcoeff(0:1,dmns_d), stat=ierr)
   allocate (save_dcoeff(0:1,dmns_d), stat=ierr)
   allocate        (tvec(0:3,dmns_d), stat=ierr)
   if (ierr.ne.0) STOP "Allocation error in dp_allocate_vectorsd"
   p0vecd= 0.d0
   dcoeff= 0.d0
   save_dcoeff= 0.d0
   tvec= 0.d0
  end subroutine dp_allocate_vectorsd
!
  subroutine dp_allocate_vectorsc
   ierr= -1
   allocate      (p0vecc(0:3,dmns_c), stat=ierr)
   allocate       (l3vec(0:3,dmns_c), stat=ierr)  
   allocate       (l4vec(0:3,dmns_c), stat=ierr)  
   allocate      (ccoeff(0:6,dmns_c), stat=ierr)
   allocate    (ccoeff_2(0:2,dmns_c), stat=ierr)
   allocate (save_ccoeff(0:6,dmns_c), stat=ierr)
   allocate         (c4_rat1(dmns_c), stat=ierr)  
   if (ierr.ne.0) STOP "Allocation error in dp_allocate_vectorsc"
   p0vecc= 0.d0
   l3vec= 0.d0  
   l4vec= 0.d0  
   ccoeff= 0.d0
   ccoeff_2= 0.d0
   save_ccoeff= 0.d0
   c4_rat1= 0.d0  
  end subroutine dp_allocate_vectorsc
!
  subroutine dp_allocate_vectorsb
   ierr= -1
   allocate      (p0vecb(0:3,dmns_b), stat=ierr)
   allocate       (l5vec(0:3,dmns_b), stat=ierr)  
   allocate       (l6vec(0:3,dmns_b), stat=ierr)  
   allocate      (bcoeff(0:8,dmns_b), stat=ierr)
   allocate        (bcoeff_2(dmns_b), stat=ierr)
   allocate (save_bcoeff(0:8,dmns_b), stat=ierr)
   allocate       (vvecb(0:3,dmns_b), stat=ierr)
   allocate          (vveck1(dmns_b), stat=ierr)
   allocate          (b_rat1(dmns_b), stat=ierr)
   allocate         (b3_rat1(dmns_b), stat=ierr)
   if (ierr.ne.0) STOP "Allocation error in dp_allocate_vectorsb"
   p0vecb= 0.d0
   l5vec= 0.d0  
   l6vec= 0.d0  
   bcoeff= 0.d0
   bcoeff_2= 0.d0
   save_bcoeff= 0.d0
   vvecb= 0.d0
   vveck1= 0.d0
   b_rat1= 0.d0
   b3_rat1= 0.d0
  end subroutine dp_allocate_vectorsb
!
  subroutine dp_allocate_vectorsa
   ierr= -1
   allocate  (vveca(0:3,dmns_a),      stat=ierr)
   allocate (p0veca(0:3,dmns_a),      stat=ierr)
   allocate  (l7vec(0:3,dmns_a),      stat=ierr)  
   allocate  (l8vec(0:3,dmns_a),      stat=ierr)  
   allocate (acoeff(0:4,dmns_a),      stat=ierr)
   allocate   (acoeff_2(dmns_a),      stat=ierr)
   allocate (save_acoeff(0:4,dmns_a), stat=ierr)
   allocate   (kvec(0:3,dmns_a),      stat=ierr) 
   allocate     (a_rat1(dmns_a),      stat=ierr) 
   if (ierr.ne.0) STOP "Allocation error in dp_allocate_vectorsa"
   vveca= 0.d0
   p0veca= 0.d0
   l7vec= 0.d0 
   l8vec= 0.d0 
   acoeff= 0.d0
   acoeff_2= 0.d0
   save_acoeff= 0.d0
   kvec= 0.d0
   a_rat1= 0.d0 
  end subroutine dp_allocate_vectorsa
!
  subroutine mp_allocate_arrays(np)
  integer, intent(in) :: np
  if (np.ge.4) call mp_allocate_vectorsd
  if (np.ge.3) call mp_allocate_vectorsc
  if (np.ge.2) call mp_allocate_vectorsb
  if (np.ge.1) call mp_allocate_vectorsa
  end subroutine mp_allocate_arrays
!
  subroutine mp_allocate_vectorsd
   ierr= -1
   allocate (mp_p0vecd(0:3,dmns_d), stat=ierr)
   allocate (mp_dcoeff(0:1,dmns_d), stat=ierr)
   allocate (save_mp_dcoeff(0:1,dmns_d), stat=ierr)
   allocate (mp_tvec(0:3,dmns_d), stat=ierr)
   if (ierr.ne.0) STOP "Allocation error in mp_allocate_vectorsd"
   do icomp2= 1,dmns_d
    do icomp1= 0,3
     mp_p0vecd(icomp1,icomp2)= 0.d0
     mp_tvec(icomp1,icomp2)= 0.d0
    enddo
    do icomp1= 0,1
     mp_dcoeff(icomp1,icomp2)= 0.d0
     save_mp_dcoeff(icomp1,icomp2)= 0.d0
    enddo
   enddo
  end subroutine mp_allocate_vectorsd
!
  subroutine mp_allocate_vectorsc
   ierr= -1
   allocate      (mp_p0vecc(0:3,dmns_c), stat=ierr)
   allocate       (mp_l3vec(0:3,dmns_c), stat=ierr)  
   allocate       (mp_l4vec(0:3,dmns_c), stat=ierr)  
   allocate      (mp_ccoeff(0:6,dmns_c), stat=ierr)
   allocate    (mp_ccoeff_2(0:2,dmns_c), stat=ierr)
   allocate (save_mp_ccoeff(0:6,dmns_c), stat=ierr)
   allocate         (mp_c4_rat1(dmns_c), stat=ierr)  
   if (ierr.ne.0) STOP "Allocation error in mp_allocate_vectorsc"
   do icomp2= 1,dmns_c
    do icomp1= 0,3
     mp_p0vecc(icomp1,icomp2)= 0.d0
     mp_l3vec(icomp1,icomp2)= 0.d0 
     mp_l4vec(icomp1,icomp2)= 0.d0  
    enddo
    do icomp1= 0,6
     mp_ccoeff(icomp1,icomp2)= 0.d0 
     save_mp_ccoeff(icomp1,icomp2)= 0.d0 
    enddo
    do icomp1= 0,2
     mp_ccoeff_2(icomp1,icomp2)= 0.d0
    enddo
    mp_c4_rat1(icomp2)= 0.d0
   enddo
  end subroutine mp_allocate_vectorsc
!
  subroutine mp_allocate_vectorsb
   ierr= -1
   allocate      (mp_p0vecb(0:3,dmns_b), stat=ierr)
   allocate       (mp_l5vec(0:3,dmns_b), stat=ierr)  
   allocate       (mp_l6vec(0:3,dmns_b), stat=ierr)  
   allocate      (mp_bcoeff(0:8,dmns_b), stat=ierr)
   allocate        (mp_bcoeff_2(dmns_b), stat=ierr)
   allocate (save_mp_bcoeff(0:8,dmns_b), stat=ierr)
   allocate       (mp_vvecb(0:3,dmns_b), stat=ierr)
   allocate          (mp_vveck1(dmns_b), stat=ierr)
   allocate          (mp_b_rat1(dmns_b), stat=ierr)
   allocate         (mp_b3_rat1(dmns_b), stat=ierr)
   if (ierr.ne.0) STOP "Allocation error in mp_allocate_vectorsb"
   do icomp2= 1,dmns_b
    do icomp1= 0,3
     mp_p0vecb(icomp1,icomp2)= 0.d0
     mp_l5vec(icomp1,icomp2)= 0.d0
     mp_l6vec(icomp1,icomp2)= 0.d0
     mp_vvecb(icomp1,icomp2)= 0.d0
    enddo 
    do icomp1= 0,8
     mp_bcoeff(icomp1,icomp2)= 0.d0
     save_mp_bcoeff(icomp1,icomp2)= 0.d0 
    enddo
    mp_bcoeff_2(icomp2)= 0.d0
    mp_vveck1(icomp2)= 0.d0
    mp_b_rat1(icomp2)= 0.d0
    mp_b3_rat1(icomp2)= 0.d0
   enddo
  end subroutine mp_allocate_vectorsb
!
  subroutine mp_allocate_vectorsa
   ierr= -1
   allocate  (mp_vveca(0:3,dmns_a)     , stat=ierr)
   allocate (save_mp_acoeff(0:4,dmns_a), stat=ierr)
   allocate (mp_p0veca(0:3,dmns_a),      stat=ierr)
   allocate  (mp_l7vec(0:3,dmns_a),      stat=ierr)  
   allocate  (mp_l8vec(0:3,dmns_a),      stat=ierr)  
   allocate (mp_acoeff(0:4,dmns_a),      stat=ierr)
   allocate   (mp_kvec(0:3,dmns_a),      stat=ierr) 
   allocate     (mp_a_rat1(dmns_a),      stat=ierr) 
   if (ierr.ne.0) STOP "Allocation error in mp_allocate_vectorsa"
   do icomp2= 1,dmns_a
    do icomp1= 0,3
     mp_vveca(icomp1,icomp2)= 0.d0
     mp_p0veca(icomp1,icomp2)= 0.d0
     mp_l7vec(icomp1,icomp2)= 0.d0
     mp_l8vec(icomp1,icomp2)= 0.d0
     mp_kvec(icomp1,icomp2)= 0.d0
    enddo
    do icomp1= 0,4
     save_mp_acoeff(icomp1,icomp2)= 0.d0
     mp_acoeff(icomp1,icomp2)= 0.d0
    enddo
    mp_a_rat1(icomp2)= 0.d0
   enddo
  end subroutine mp_allocate_vectorsa
! 
  subroutine dp_get_coefficients(p,numdummy,number_propagators,dmr,ql)
   use scale
   external numdummy
   include 'cts_dpr.h'
    , intent(in) :: p
   include 'cts_dpc.h'
    , dimension(0:3) :: q
   include 'cts_dpr.h'
    , intent(in) :: ql
   integer, intent(in) :: number_propagators
   integer, intent(in) :: dmr
   include 'cts_dpc.h'
    , dimension(0:3) :: qvalue
   include 'cts_dpc.h' 
    :: dummy1,dummy2
   include 'cts_dpr.h' 
    :: prec
   include 'cts_dpc.h' 
    :: rat1_d,rat1_c,rat1_b,rat1_a
   include 'cts_dpc.h' 
    :: rat_aus,rat_aus1,rat_aus2
   include 'cts_dpc.h' 
    :: qt2_0,qt2_1,qt2_2,qt2_inf,qt2_sv
   integer :: k,ib,iteration,itermax,dmr1
   logical :: computing=.true.
   include 'cts_dpc.h' 
    :: aset(dmns_a,1:4)
   include 'cts_dpc.h' 
    :: bset(dmns_b,1:4),bset3(dmns_b,1:4)
   include 'cts_dpc.h' 
    :: cset(dmns_c,1:4)
   include 'cts_dpc.h' 
    :: dset(dmns_d,1:4)
   include 'cts_dpc.h' 
    :: rset(1:2)
   save computing,qt2_0,qt2_1,qt2_inf
   stablen=.true.
   if (computing) then
    computing=.false.
    qt2_0=  0.d0
!
!   value of q used for computing R_1
!
    q    =  0.d0
   endif
   qt2_1=  ql*ql*0.001d0   
   qt2_2=  ql*ql*0.003d0   
   qt2_inf =  ql*ql*1.d8
   qt2= qt2_0
   rational=.false.              
   if (number_propagators.ge.4) call getd(p,numdummy,number_propagators,dmr)
   if (number_propagators.ge.3) call getc(p,numdummy,number_propagators,dmr)
   if (number_propagators.ge.2) call getb(p,numdummy,number_propagators,dmr)
   if (number_propagators.ge.1) call geta(p,numdummy,number_propagators,dmr)
   call load_set(1)
   if (number_propagators.ge.4) call put_dcoeff(p)
   if (number_propagators.ge.3) call put_ccoeff(p)
   if (number_propagators.ge.2) call put_bcoeff(p)
   if (number_propagators.ge.1) call put_acoeff(p)
!-comment
   qvalue(0)= ql/2.d0
   qvalue(1)=-ql/3.d0
   qvalue(2)= ql/4.d0
   qvalue(3)=-ql/5.d0
   call test(p,numdummy,number_propagators,qvalue,dummy1,dummy2,prec)
!   print *,'           '
!   print *,'dummy1=',dummy1,'to be compared with'
!   print *,'dummy2=',dummy2
!   print *,'prec  in dp_get_coefficients:',prec
   precstablen= prec
   if (prec.gt.llimit) stablen=.false.
!-comment
   rational=.true. 
   if (dmr.ge.0) then
!-comment-new
!    do iteration= 1,2
    do iteration= 1,1
!-comment-new
!
!     iteration= 2 is to get the main coefficients in a different way 
!     in order to test the numerical accuracy of the final result
!
      if (iteration.eq.1) then
       qt2= qt2_1
       dmr1= dmr
      else
       qt2=-qt2_1
       dmr1= 0
      endif
      if (number_propagators.ge.4) call getd(p,numdummy,number_propagators,dmr1)
      if (number_propagators.ge.3) call getc(p,numdummy,number_propagators,dmr1)
      if (number_propagators.ge.2) call getb(p,numdummy,number_propagators,dmr1)
      if (iteration.eq.2) then
       if (number_propagators.ge.1)call geta(p,numdummy,number_propagators,dmr1)
      endif
      call load_set(iteration+1)
      rat1_d= 0.d0 
      rat1_c= 0.d0 
      rat1_b= 0.d0 
      if     (dmr1.ge.2) then
      elseif (dmr1.eq.1) then
       if (number_propagators.ge.3) call getc_2(rat1_c)
      elseif (dmr1.eq.0) then
       if (number_propagators.ge.2) call getb_2(rat1_b)
       if (number_propagators.ge.3) call getc_2(rat1_c)
       if (number_propagators.ge.4) then
         if (inf) then
          call getd_last3(q,rat1_d) 
         else
          call getd_last1(q,rat1_d) ! reconstructed from the other coefficients
!         call getd_last2(q,rat1_d) ! computed by re-fitting the D sector
         endif 
       endif 
      endif
      rat1= rat1_d+rat1_c+rat1_b
      call load_rat(iteration)
    enddo
   elseif (dmr.eq.-1) then
!-comment-new
!    do iteration= 1,3
    do iteration= 1,2
!-comment-new
!
!     iteration= 3 is to get the main coefficients in a different way 
!     in order to test the numerical accuracy of the final result
!
      if     (iteration.eq.1) then
       qt2 = qt2_1
      elseif (iteration.eq.2) then
       qt2 =-qt2_1
      else
       qt2 = qt2_2
      endif
      if (number_propagators.ge.4) call getd(p,numdummy,number_propagators,dmr)
      if (number_propagators.ge.3) call getc(p,numdummy,number_propagators,dmr)
      if ((iteration.eq.1).or.(iteration.eq.3)) then  
       if (number_propagators.ge.2)call getb(p,numdummy,number_propagators,dmr)
       if (number_propagators.ge.1)call geta(p,numdummy,number_propagators,dmr)
      endif
      call load_set(iteration+1)
    enddo
!
!   First determination of R_1
!
    rat1_d = 0.d0
    rat1_c = 0.d0
    rat1_b = 0.d0
    rat1_a = 0.d0
    if (number_propagators.ge.1) then
     do ib= 1,dmns_1
      rat1_a = rat1_a-0.5d0*a_rat1(ib)*(aset(ib,2)-aset(ib,1))/qt2_1 
     enddo
    endif
    if (number_propagators.ge.2) then
     do ib= 1,dmns_2
      rat1_b = rat1_b-0.5d0*b_rat1(ib)*(bset(ib,2) -bset(ib,1) )/qt2_1 
      rat1_b = rat1_b+     b3_rat1(ib)*(bset3(ib,2)-bset3(ib,1))/qt2_1/12.d0 
     enddo
    endif
    if (number_propagators.ge.3) then
     do ib= 1,dmns_3
      rat_aus= (cset(ib,2)-cset(ib,3))/2.d0/qt2_1
      rat1_c= rat1_c-0.5d0*rat_aus 
      rat1_c= rat1_c-c4_rat1(ib)*(cset(ib,2)-cset(ib,1) &
              -rat_aus*qt2_1)/qt2_1/qt2_1/12.d0
     enddo
    endif
    if (number_propagators.eq.4) then
     do ib= 1,dmns_4
      rat_aus= (dset(ib,2)-dset(ib,3))/2.d0/qt2_1
      rat1_d= rat1_d-(dset(ib,2)-dset(ib,1)-rat_aus*qt2_1)/qt2_1/qt2_1/6.d0
     enddo
    endif
    if (number_propagators.gt.4) then
     inf_sv= inf
     qt2_sv= qt2
     inf=.true. 
     qt2= qt2_inf
     call getd(p,numdummy,number_propagators,dmr)
     do ib= 1,dmns_4
      rat1_d= rat1_d-dcoeff(0,ib)/qt2/qt2/6.d0
     enddo
     qt2= qt2_sv
     inf= inf_sv 
    endif
    rat1= rat1_d+rat1_c+rat1_b+rat1_a
    call load_rat(1)
!-comment-new
!!$!
!!$!   Second determination of R_1
!!$!
!!$    rat1_d = 0.d0
!!$    rat1_c = 0.d0
!!$    rat1_b = 0.d0
!!$    rat1_a = 0.d0
!!$    if (number_propagators.ge.1) then
!!$     do ib= 1,dmns_1
!!$      rat1_a = rat1_a-0.5d0*a_rat1(ib)*(aset(ib,4)- aset(ib,1))/qt2_2 
!!$     enddo
!!$    endif
!!$    if (number_propagators.ge.2) then
!!$     do ib= 1,dmns_2
!!$      rat1_b = rat1_b-0.5d0*b_rat1(ib)*(bset(ib,4)- bset(ib,1) )/qt2_2 
!!$      rat1_b = rat1_b+     b3_rat1(ib)*(bset3(ib,4)-bset3(ib,1))/qt2_2/12.d0 
!!$     enddo
!!$    endif
!!$    rat_aus = (qt2_2-qt2_1)
!!$    if (number_propagators.ge.3) then
!!$     do ib= 1,dmns_3
!!$      rat_aus1= (cset(ib,2)-cset(ib,1))/qt2_1
!!$      rat_aus2= (cset(ib,4)-cset(ib,1))/qt2_2
!!$      rat1_c = rat1_c-0.5d0*(rat_aus1*qt2_2-rat_aus2*qt2_1)/rat_aus 
!!$      rat1_c = rat1_c-c4_rat1(ib)*(rat_aus2-rat_aus1)/rat_aus/12.d0
!!$     enddo
!!$    endif
!!$    if (number_propagators.ge.4) then
!!$     do ib= 1,dmns_4
!!$      rat_aus1= (dset(ib,2)-dset(ib,1))/qt2_1
!!$      rat_aus2= (dset(ib,4)-dset(ib,1))/qt2_2
!!$      rat1_d  = rat1_d-(rat_aus2-rat_aus1)/rat_aus/6.d0
!!$     enddo
!!$    endif
!!$    rat1= rat1_d+rat1_c+rat1_b+rat1_a
!!$    call load_rat(2)
!-comment-new
   else 
    print*,'dmr=',dmr,' not allowed' 
    stop
   endif
!-comment-new
!   call finalize_sets
   save_rat1= rset(1) 
!-comment-new
   qt2= qt2_0
!  comment
!  call compare
   contains
!   
   subroutine compare
    if (number_propagators.ge.1) then
     print*,'   '
     print*,'   '
     do ib= 1,dmns_1
      print*,'     acoeff(0,',ib,')=',     acoeff(0,ib)
      print*,'save_acoeff(0,',ib,')=',save_acoeff(0,ib)
     enddo
    endif
    if (number_propagators.ge.2) then
     print*,'   '
     print*,'   '
     do ib= 1,dmns_2
      print*,'     bcoeff(0,',ib,')=',     bcoeff(0,ib)
      print*,'save_bcoeff(0,',ib,')=',save_bcoeff(0,ib)
      print*,'     bcoeff(3,',ib,')=',     bcoeff(3,ib)
      print*,'save_bcoeff(3,',ib,')=',save_bcoeff(3,ib)
     enddo
    endif
    if (number_propagators.ge.3) then
     print*,'   '
     print*,'   '
     do ib= 1,dmns_3
      print*,'     ccoeff(0,',ib,')=',     ccoeff(0,ib)
      print*,'save_ccoeff(0,',ib,')=',save_ccoeff(0,ib)
     enddo
    endif
    if (number_propagators.ge.4) then
     print*,'   '
     print*,'   '
     do ib= 1,dmns_4
      print*,'     dcoeff(0,',ib,')=',     dcoeff(0,ib)
      print*,'save_dcoeff(0,',ib,')=',save_dcoeff(0,ib)
     enddo
    endif
    print*,'   '
    print*,'   '
    print*,'save_rat1=',save_rat1
    print*,'     rat1=',rat1
    print*,' rat1_d  =',rat1_d
    print*,' rat1_c  =',rat1_c
    print*,' rat1_b  =',rat1_b
    print*,' rat1_a  =',rat1_a
    print*,'   '
    print*,'   '
   end subroutine compare
!
   subroutine getb_2(rat1_b)
   include 'cts_dpc.h'
    , intent(out) :: rat1_b
   rat1_b= 0.d0
   do ib= 1,dmns_2
     bcoeff_2(ib)= (bcoeff(0,ib)-save_bcoeff(0,ib))/qt2  
     rat1_b= rat1_b+bcoeff_2(ib)*b_rat1(ib)
   enddo
   rat1_b= -0.5d0*rat1_b 
   end subroutine getb_2 
!
   subroutine getc_2(rat1_c)
   include 'cts_dpc.h'
    , intent(out) :: rat1_c
   rat1_c= 0.d0
   do ib= 1,dmns_3
     ccoeff_2(0,ib)= (ccoeff(0,ib)-save_ccoeff(0,ib))/qt2  
     ccoeff_2(1,ib)= (ccoeff(1,ib)-save_ccoeff(1,ib))/qt2  
     ccoeff_2(2,ib)= (ccoeff(2,ib)-save_ccoeff(2,ib))/qt2  
     rat1_c= rat1_c+ccoeff_2(0,ib)
   enddo
   rat1_c= -0.5d0*rat1_c
   end subroutine getc_2 
!
   subroutine getd_last1(q,rat1_d)
   include 'cts_dpc.h'
    , intent(in), dimension(0:3) :: q
   include 'cts_dpc.h'
    , intent(out) :: rat1_d
   integer :: i,np
   include 'cts_dpc.h'
    , dimension(0:3) :: qp0
   include 'cts_dpc.h' 
    :: l3qp0,l4qp0,vqp0,kqp0
   include 'cts_dpc.h' 
    :: start,sumdena,sumdenb,num0,num2,num3
   np= number_propagators
   start= 0.d0 
!
!  contribution from the a sector:
!
   if (iteration.eq.1) then
    do ib= 1,dmns_1
     do k= 0,3
      qp0(k)= q(k)+p0veca(k,ib) 
     enddo
     call contr(l7vec(:,ib),qp0,l3qp0)
     call contr(l8vec(:,ib),qp0,l4qp0)
     call contr(vveca(:,ib),qp0,vqp0)
     call contr(kvec(:,ib),qp0,kqp0)
     sumdena= 0.d0 
     do k= 2,np
      sumdena= sumdena+(value(den(bbn1(k,ib)),q)-qt2)
     enddo
     start= start+(                    &
              save_acoeff(0,ib)        & 
             +save_acoeff(1,ib)*kqp0   &
             +save_acoeff(2,ib)*vqp0   &
             +save_acoeff(3,ib)*l3qp0  &
             +save_acoeff(4,ib)*l4qp0)*sumdena 
    enddo  
   else
    do ib= 1,dmns_1
     do k= 0,3
      qp0(k)= q(k)+p0veca(k,ib) 
     enddo
     call contr(l7vec(:,ib),qp0,l3qp0)
     call contr(l8vec(:,ib),qp0,l4qp0)
     call contr(vveca(:,ib),qp0,vqp0)
     call contr(kvec(:,ib),qp0,kqp0)
     sumdena= 0.d0 
     do k= 2,np
      sumdena= sumdena+(value(den(bbn1(k,ib)),q)-qt2)
     enddo
     start= start+(               &
              acoeff(0,ib)        & 
             +acoeff(1,ib)*kqp0   &
             +acoeff(2,ib)*vqp0   &
             +acoeff(3,ib)*l3qp0  &
             +acoeff(4,ib)*l4qp0)*sumdena 
    enddo  
   endif
!
!  contribution from the b sector:
!
   do ib= 1,dmns_2
    do k= 0,3
     qp0(k)= q(k)+p0vecb(k,ib) 
    enddo
    call contr(l5vec(:,ib),qp0,l3qp0)
    call contr(l6vec(:,ib),qp0,l4qp0)
    call contr(vvecb(:,ib),qp0,vqp0)
    sumdenb= 0.d0 
    do k= 3,np
     sumdenb= sumdenb+(value(den(bbn2(k,ib)),q)-qt2)
    enddo
    if (iteration.eq.1) start= start+save_bcoeff(0,ib)
    if (iteration.eq.2) start= start+(bset(ib,2)+bset(ib,3))/2.d0
    start=  start+(                            & 
             save_bcoeff(1,ib)*l3qp0           &
            +save_bcoeff(2,ib)*l4qp0           &
            +save_bcoeff(3,ib)*vqp0            &
            +save_bcoeff(4,ib)*(l3qp0)**2      &
            +save_bcoeff(5,ib)*(l4qp0)**2      &
            +save_bcoeff(6,ib)*vqp0**2         &
            +save_bcoeff(7,ib)*vqp0*l3qp0      &
            +save_bcoeff(8,ib)*vqp0*l4qp0)     &    
            +bcoeff_2(ib)*sumdenb
   enddo  
!
!  contribution from the c sector:
!
   do ib= 1,dmns_3
    do k= 0,3 
     qp0(k)= q(k)+p0vecc(k,ib) 
    enddo
    call contr(l3vec(:,ib),qp0,l3qp0)
    call contr(l4vec(:,ib),qp0,l4qp0)
    start= start+(ccoeff_2(0,ib)               &
                 +ccoeff_2(1,ib)*l3qp0         &
                 +ccoeff_2(2,ib)*l4qp0)        
   enddo
   rat1_d= -start
   rat1_d=-rat1_d/6.d0 ! multiplying by the R_2 integral
   end subroutine getd_last1
!
   subroutine getd_last2(q,rat1_d)
    include 'cts_dpc.h'
     , intent(in), dimension(0:3) :: q
    include 'cts_dpc.h'
     , intent(out) :: rat1_d
    include 'cts_dpc.h'
     , dimension(1:(number_propagators-2)) :: x
    include 'cts_dpc.h'
     , dimension(0:(number_propagators-2)) :: dres
    include 'cts_dpc.h' 
     :: qt2_save 
    integer :: n,ntot,kmax 
    qt2_save= qt2        ! store the current value of qt2
    dres(0)= numd(number_propagators,q,0) ! compute sumd at x(0)= qt2
    kmax= number_propagators-2
    ntot= kmax+1
    do n= 1,ntot-1
     x(n)= qt2*exp(-2.d0*ci(p)*pi(p)*n/ntot)
    enddo
    do n= 1,ntot-1
     qt2= x(n)
     call getd(p,numdummy,number_propagators,dmr1)  
     dres(n)= numd(number_propagators,q,0) ! compute sumd at x(n)
    enddo
    rat1_d= 0.d0
    do n= 0,ntot-1
     rat1_d= rat1_d+dres(n)*exp(2.d0*ci(p)*pi(p)*n*kmax/ntot)
    enddo
    qt2= qt2_save         ! reload the initial value of qt2
    rat1_d= rat1_d/ntot/qt2**kmax
    rat1_d=-rat1_d/6.d0   ! multiplying by the R_2 integral
   end subroutine getd_last2
!
   subroutine getd_last3(q,rat1_d)
    include 'cts_dpc.h'
     , intent(in), dimension(0:3) :: q
    include 'cts_dpc.h'
     , intent(out) :: rat1_d
    include 'cts_dpc.h' 
     :: qt2_save 
    qt2_save= qt2        ! store the current value of qt2
    qt2= qt2_inf 
    call getd(p,numdummy,number_propagators,dmr1)  
    rat1_d= numd(number_propagators,q,0) ! compute sumd at qt2_inf
    rat1_d= rat1_d/qt2**(number_propagators-2)
    rat1_d=-rat1_d/6.d0   ! multiplying by the R_2 integral
    qt2= qt2_save         ! reload the initial value of qt2
   end subroutine getd_last3
!
   subroutine load_set(n)
   integer, intent(in) :: n
   if (number_propagators.ge.1) then
    do ib= 1,dmns_1
     aset(ib,n)= acoeff(0,ib)
    enddo
   endif
   if (number_propagators.ge.2) then
    do ib= 1,dmns_2
     bset (ib,n)= bcoeff(0,ib)
     bset3(ib,n)= bcoeff(3,ib)
    enddo
   endif
   if (number_propagators.ge.3) then
    do ib= 1,dmns_3
     cset(ib,n)= ccoeff(0,ib)
    enddo
   endif
   if (number_propagators.ge.4) then
    do ib= 1,dmns_4
     dset(ib,n)= dcoeff(0,ib)
    enddo
   endif
   end subroutine load_set
!
   subroutine load_rat(n)
   integer, intent(in) :: n
   rset(n)= rat1 
   end subroutine load_rat
!
   subroutine finalize_sets
   include 'cts_dpc.h' 
    :: r_aus,r_aus1,r_aus2,r_aus3
   if     (dmr.ge.0) then
    if (number_propagators.ge.1) then
     do ib= 1,dmns_1
      acoeff(0,ib) = aset(ib,3)
     enddo
    endif
    if (number_propagators.ge.2) then
     do ib= 1,dmns_2
      bcoeff(0,ib)= (bset(ib,2)+bset(ib,3))/2.d0 
     enddo
    endif
    if (number_propagators.ge.3) then
     do ib= 1,dmns_3
      ccoeff(0,ib)= (cset(ib,2)+cset(ib,3))/2.d0
      ccoeff(1,ib)= save_ccoeff(1,ib) ! to put it to the true value  
      ccoeff(2,ib)= save_ccoeff(2,ib) ! to put it to the true value
     enddo
    endif
   elseif (dmr.eq.-1) then
!
!   Restores all values
!
!!    if (number_propagators.ge.4) call get_dcoeff(p)
!!    if (number_propagators.ge.3) call get_ccoeff(p)
!!    if (number_propagators.ge.2) call get_bcoeff(p)
!!    if (number_propagators.ge.1) call get_acoeff(p)
!
!   The new determinations of the non-spurious coefficients 
!
    r_aus = (qt2_2-qt2_1)
    r_aus3= (qt2_2+qt2_1)
    if (number_propagators.ge.1) then
     do ib= 1,dmns_1
      acoeff(0,ib)= (qt2_2*aset(ib,2)-qt2_1*aset(ib,4))/r_aus
     enddo
    endif
    if (number_propagators.ge.2) then
     do ib= 1,dmns_2
      bcoeff(0,ib)= (qt2_2* bset(ib,2) -qt2_1*bset(ib,4))/r_aus
      bcoeff(3,ib)= (qt2_2*bset3(ib,2)-qt2_1*bset3(ib,4))/r_aus
     enddo
    endif
    if (number_propagators.ge.3) then
     do ib= 1,dmns_3
      r_aus1= (cset(ib,2)+cset(ib,3))/2.d0
      r_aus2= cset(ib,4)-qt2_2*(cset(ib,2)-cset(ib,3))/2.d0/qt2_1
      ccoeff(0,ib)= (r_aus1*qt2_2**2-r_aus2*qt2_1**2)/r_aus/r_aus3
     enddo
    endif
    if (number_propagators.ge.4) then
     do ib= 1,dmns_4
      r_aus1= (dset(ib,2)+dset(ib,3))/2.d0
      r_aus2= dset(ib,4)-qt2_2*(dset(ib,2)-dset(ib,3))/2.d0/qt2_1
      dcoeff(0,ib)= (r_aus1*qt2_2**2-r_aus2*qt2_1**2)/r_aus/r_aus3
     enddo
    endif
   else
     stop 'dmr value not allowed in subroutine finalize_sets!'
   endif
   save_rat1= rset(1) 
   rat1     = rset(2) 
   end subroutine finalize_sets
!
  end subroutine dp_get_coefficients
! 
  subroutine mp_get_coefficients(p,numdummy,number_propagators,dmr,ql)
   use scale
   external numdummy
   include 'cts_mpr.h'
    , intent(in) :: p
   include 'cts_dpc.h'
    , dimension(0:3) :: q
   include 'cts_dpr.h'
    , intent(in) :: ql
   integer, intent(in) :: number_propagators
   integer, intent(in) :: dmr
   include 'cts_dpc.h'
    , dimension(0:3) :: qvalue
   include 'cts_mpc.h' 
    :: dummy1,dummy2
   include 'cts_dpr.h' 
    :: prec
   include 'cts_mpc.h' 
    :: rat1_d,rat1_c,rat1_b,rat1_a
   include 'cts_mpc.h' 
    :: rat_aus,rat_aus1,rat_aus2
   include 'cts_mpc.h' 
    :: qt2_0,qt2_1,qt2_2,qt2_inf,mpqt2_sv
   integer :: k,ib,iteration,dmr1
   logical :: computing=.true.
   include 'cts_mpc.h' 
    :: aset(dmns_a,1:4)
   include 'cts_mpc.h' 
    :: bset(dmns_b,1:4),bset3(dmns_b,1:4)
   include 'cts_mpc.h' 
    :: cset(dmns_c,1:4)
   include 'cts_mpc.h' 
    :: dset(dmns_d,1:4)
   include 'cts_mpc.h' 
    :: rset(1:2)
   save computing,qt2_0,qt2_1,qt2_inf
   stablen=.true.
   if (computing) then
    computing=.false.
    qt2_0=  0.d0
    q    =  0.d0
   endif
   qt2_1=  ql*ql*0.001d0    
   qt2_2=  ql*ql*0.003d0   
   qt2_inf =  ql*ql*1.d16 
   mpqt2= qt2_0
   rational=.false.              
   if (number_propagators.ge.4) call getd(p,numdummy,number_propagators,dmr)
   if (number_propagators.ge.3) call getc(p,numdummy,number_propagators,dmr)
   if (number_propagators.ge.2) call getb(p,numdummy,number_propagators,dmr)
   if (number_propagators.ge.1) call geta(p,numdummy,number_propagators,dmr)
   call load_set(1)
   if (number_propagators.ge.4) call put_dcoeff(p)
   if (number_propagators.ge.3) call put_ccoeff(p)
   if (number_propagators.ge.2) call put_bcoeff(p)
   if (number_propagators.ge.1) call put_acoeff(p)
!-comment
   qvalue(0)= ql/2.d0
   qvalue(1)=-ql/3.d0
   qvalue(2)= ql/4.d0
   qvalue(3)=-ql/5.d0
   call test(p,numdummy,number_propagators,qvalue,dummy1,dummy2,prec)
!   print *,'           '
!   aus= dummy1
!   print *,'dummy1=',aus,'to be compared with'
!   aus= dummy2
!   print *,'dummy2=',aus
!   print *,'prec  in mp_get_coefficients:',prec
   precstablen= prec
   if (prec.gt.llimit) stablen=.false.
!-comment
   rational=.true.          
   if (dmr.ge.0) then
!-comment-new
!    do iteration= 1,2
    do iteration= 1,1
!-comment-new
!
!     iteration= 2 is to get the main coefficients in a different way 
!     in order to test the numerical accuracy of the final result
!
      if (iteration.eq.1) then
       mpqt2= qt2_1
       dmr1= dmr
      else
       mpqt2=-qt2_1
       dmr1= 0
      endif
      if (number_propagators.ge.4) call getd(p,numdummy,number_propagators,dmr1)
      if (number_propagators.ge.3) call getc(p,numdummy,number_propagators,dmr1)
      if (number_propagators.ge.2) call getb(p,numdummy,number_propagators,dmr1)
      if (iteration.eq.2) then
       if (number_propagators.ge.1)call geta(p,numdummy,number_propagators,dmr1)
      endif
      call load_set(iteration+1)
      rat1_d= 0.d0 
      rat1_c= 0.d0 
      rat1_b= 0.d0 
      if     (dmr1.ge.2) then
      elseif (dmr1.eq.1) then
       if (number_propagators.ge.3) call getc_2(rat1_c)
      elseif (dmr1.eq.0) then
       if (number_propagators.ge.2) call getb_2(rat1_b)
       if (number_propagators.ge.3) call getc_2(rat1_c)
       if (number_propagators.ge.4) then
         if (inf) then
          call getd_last3(q,rat1_d) 
         else
          call getd_last1(q,rat1_d) ! reconstructed from the other coefficients
!         call getd_last2(q,rat1_d) ! computed by re-fitting the D sector
         endif 
       endif 
      endif
      mp_rat1= rat1_d+rat1_c+rat1_b
      call load_rat(iteration)
    enddo
   elseif (dmr.eq.-1) then
!-comment-new
!    do iteration= 1,3
    do iteration= 1,2
!-comment-new
!
!     iteration= 3 is to get the main coefficients in a different way 
!     in order to test the numerical accuracy of the final result
!
      if     (iteration.eq.1) then
       mpqt2 = qt2_1
      elseif (iteration.eq.2) then
       mpqt2 =-qt2_1
      else
       mpqt2 = qt2_2
      endif
      if (number_propagators.ge.4) call getd(p,numdummy,number_propagators,dmr)
      if (number_propagators.ge.3) call getc(p,numdummy,number_propagators,dmr)
      if ((iteration.eq.1).or.(iteration.eq.3)) then  
       if (number_propagators.ge.2)call getb(p,numdummy,number_propagators,dmr)
       if (number_propagators.ge.1)call geta(p,numdummy,number_propagators,dmr)
      endif
      call load_set(iteration+1)
    enddo
!
!   First determination of R_1
!
    rat1_d = 0.d0
    rat1_c = 0.d0
    rat1_b = 0.d0
    rat1_a = 0.d0
    if (number_propagators.ge.1) then
     do ib= 1,dmns_1
      rat1_a = rat1_a-0.5d0*mp_a_rat1(ib)*(aset(ib,2)-aset(ib,1))/qt2_1 
     enddo
    endif
    if (number_propagators.ge.2) then
     do ib= 1,dmns_2
      rat1_b = rat1_b-0.5d0*mp_b_rat1(ib)*(bset(ib,2) -bset(ib,1))/qt2_1 
      rat1_b = rat1_b+     mp_b3_rat1(ib)*(bset3(ib,2)-bset3(ib,1))/qt2_1/12.d0 
     enddo
    endif
    if (number_propagators.ge.3) then
     do ib= 1,dmns_3
      rat_aus= (cset(ib,2)-cset(ib,3))/2.d0/qt2_1
      rat1_c= rat1_c-0.5d0*rat_aus 
      rat1_c= rat1_c-mp_c4_rat1(ib)*(cset(ib,2)-cset(ib,1) &
             -rat_aus*qt2_1)/qt2_1/qt2_1/12.d0
     enddo
    endif
    if (number_propagators.eq.4) then
     do ib= 1,dmns_4
      rat_aus= (dset(ib,2)-dset(ib,3))/2.d0/qt2_1
      rat1_d = rat1_d-(dset(ib,2)-dset(ib,1)-rat_aus*qt2_1)/qt2_1/qt2_1/6.d0
     enddo
    endif
    if (number_propagators.gt.4) then
     inf_sv= inf
     mpqt2_sv= mpqt2
     inf=.true.
     mpqt2= qt2_inf
     call getd(p,numdummy,number_propagators,dmr)
     do ib= 1,dmns_4
      rat1_d= rat1_d-mp_dcoeff(0,ib)/mpqt2/mpqt2/6.d0
     enddo
     mpqt2= mpqt2_sv
     inf= inf_sv 
    endif
    mp_rat1= rat1_d+rat1_c+rat1_b+rat1_a
    call load_rat(1)
!-comment-new
!!$!
!!$!   Second determination of R_1
!!$!
!!$    rat1_d = 0.d0
!!$    rat1_c = 0.d0
!!$    rat1_b = 0.d0
!!$    rat1_a = 0.d0
!!$    if (number_propagators.ge.1) then
!!$     do ib= 1,dmns_1
!!$      rat1_a = rat1_a-0.5d0*mp_a_rat1(ib)*(aset(ib,4)- aset(ib,1))/qt2_2 
!!$     enddo
!!$    endif
!!$    if (number_propagators.ge.2) then
!!$     do ib= 1,dmns_2
!!$      rat1_b = rat1_b-0.5d0*mp_b_rat1(ib)*(bset(ib,4)- bset(ib,1))/qt2_2 
!!$      rat1_b = rat1_b+     mp_b3_rat1(ib)*(bset3(ib,4)-bset3(ib,1))/qt2_2/12.d0 
!!$     enddo
!!$    endif
!!$    rat_aus = (qt2_2-qt2_1)
!!$    if (number_propagators.ge.3) then
!!$     do ib= 1,dmns_3
!!$      rat_aus1= (cset(ib,2)-cset(ib,1))/qt2_1
!!$      rat_aus2= (cset(ib,4)-cset(ib,1))/qt2_2
!!$      rat1_c = rat1_c-0.5d0*(rat_aus1*qt2_2-rat_aus2*qt2_1)/rat_aus 
!!$      rat1_c = rat1_c-mp_c4_rat1(ib)*(rat_aus2-rat_aus1)/rat_aus/12.d0
!!$     enddo
!!$    endif
!!$    if (number_propagators.ge.4) then
!!$     do ib= 1,dmns_4
!!$      rat_aus1= (dset(ib,2)-dset(ib,1))/qt2_1
!!$      rat_aus2= (dset(ib,4)-dset(ib,1))/qt2_2
!!$      rat1_d  = rat1_d-(rat_aus2-rat_aus1)/rat_aus/6.d0
!!$     enddo
!!$    endif
!!$    mp_rat1= rat1_d+rat1_c+rat1_b+rat1_a
!!$    call load_rat(2)
!-comment-new
   else 
    print*,'dmr=',dmr,' not allowed' 
    stop
   endif
!-comment-new
!   call finalize_sets
   save_mp_rat1= rset(1) 
!-comment-new
   mpqt2= qt2_0
!  comment
!  call compare
   contains
!   
   subroutine compare
    include 'cts_dpc.h' 
     :: aus
    if (number_propagators.ge.1) then
     print*,'   '
     print*,'   '
     do ib= 1,dmns_1
      aus= mp_acoeff(0,ib)
      print*,'     mp_acoeff(0,ib)=',aus     
      aus= save_mp_acoeff(0,ib)
      print*,'save_mp_acoeff(0,ib)=',aus
     enddo
    endif 
    if (number_propagators.ge.2) then
     print*,'   '
     print*,'   '
     do ib= 1,dmns_2
      aus= mp_bcoeff(0,ib)
      print*,'     mp_bcoeff(0,ib)=',aus     
      aus= save_mp_bcoeff(0,ib)
      print*,'save_mp_bcoeff(0,ib)=',aus
      aus= mp_bcoeff(3,ib)
      print*,'     mp_bcoeff(3,ib)=',aus     
      aus= save_mp_bcoeff(3,ib)
      print*,'save_mp_bcoeff(3,ib)=',aus
     enddo
    endif
    if (number_propagators.ge.3) then
     print*,'   '
     print*,'   '
     do ib= 1,dmns_3
      aus= mp_ccoeff(0,ib)
      print*,'     mp_ccoeff(0,ib)=',aus     
      aus= save_mp_ccoeff(0,ib)
      print*,'save_mp_ccoeff(0,ib)=',aus
     enddo
    endif
    if (number_propagators.ge.4) then
     print*,'   '
     print*,'   '
     do ib= 1,dmns_4
      aus= mp_dcoeff(0,ib)
      print*,'     mp_dcoeff(0,',ib,')=',aus
      aus= save_mp_dcoeff(0,ib)
      print*,'save_mp_dcoeff(0,',ib,')=',aus
     enddo
    endif
    print*,'   '
    print*,'   '
    aus= save_mp_rat1
    print*,'save_mp_rat1=',aus
    aus= mp_rat1
    print*,'     mp_rat1=',aus
    aus= rat1_d
    print*,' rat1_d  =',aus
    aus= rat1_c
    print*,' rat1_c  =',aus
    aus= rat1_b
    print*,' rat1_b  =',aus
    aus= rat1_a
    print*,' rat1_a  =',aus
    print*,'   '
    print*,'   '
   end subroutine compare
!
   subroutine getb_2(rat1_b)
   include 'cts_mpc.h'
    , intent(out) :: rat1_b
   rat1_b= 0.d0
   do ib= 1,dmns_2
     mp_bcoeff_2(ib)= (mp_bcoeff(0,ib)-save_mp_bcoeff(0,ib))/mpqt2
     rat1_b= rat1_b+mp_bcoeff_2(ib)*mp_b_rat1(ib)
   enddo
   rat1_b= -0.5d0*rat1_b 
   end subroutine getb_2 
!
   subroutine getc_2(rat1_c)
   include 'cts_mpc.h'
    , intent(out) :: rat1_c
   rat1_c= 0.d0
   do ib= 1,dmns_3
     mp_ccoeff_2(0,ib)= (mp_ccoeff(0,ib)-save_mp_ccoeff(0,ib))/mpqt2  
     mp_ccoeff_2(1,ib)= (mp_ccoeff(1,ib)-save_mp_ccoeff(1,ib))/mpqt2  
     mp_ccoeff_2(2,ib)= (mp_ccoeff(2,ib)-save_mp_ccoeff(2,ib))/mpqt2  
     rat1_c= rat1_c+mp_ccoeff_2(0,ib)
   enddo
   rat1_c= -0.5d0*rat1_c
   end subroutine getc_2 
!
   subroutine getd_last1(q,rat1_d)
   include 'cts_dpc.h'
    , intent(in), dimension(0:3) :: q
   include 'cts_mpc.h'
    , dimension(0:3) :: mpq
   include 'cts_mpc.h'
    , intent(out) :: rat1_d
   integer :: i,np
   include 'cts_mpc.h'
    , dimension(0:3) :: qp0
   include 'cts_mpc.h' 
    :: l3qp0,l4qp0,vqp0,kqp0
   include 'cts_mpc.h' 
    :: start,sumdena,sumdenb,num0,num2,num3
   np= number_propagators
   start= 0.d0 
   do k= 0,3; mpq(k)= q(k)*c1(p); enddo
!
!  contribution from the a sector:
!
   if (iteration.eq.1) then
    do ib= 1,dmns_1
     do k= 0,3
      qp0(k)= mpq(k)+mp_p0veca(k,ib) 
     enddo
     call contr(mp_l7vec(:,ib),qp0,l3qp0)
     call contr(mp_l8vec(:,ib),qp0,l4qp0)
     call contr(mp_vveca(:,ib),qp0,vqp0)
     call contr(mp_kvec(:,ib),qp0,kqp0)
     sumdena= 0.d0 
     do k= 2,np
      sumdena= sumdena+(value(mp_den(bbn1(k,ib)),mpq)-mpqt2)
     enddo
     start= start+(                    &
              save_mp_acoeff(0,ib)        & 
             +save_mp_acoeff(1,ib)*kqp0   &
             +save_mp_acoeff(2,ib)*vqp0   &
             +save_mp_acoeff(3,ib)*l3qp0  &
             +save_mp_acoeff(4,ib)*l4qp0)*sumdena 
    enddo  
   else
    do ib= 1,dmns_1
     do k= 0,3
      qp0(k)= mpq(k)+mp_p0veca(k,ib) 
     enddo
     call contr(mp_l7vec(:,ib),qp0,l3qp0)
     call contr(mp_l8vec(:,ib),qp0,l4qp0)
     call contr(mp_vveca(:,ib),qp0,vqp0)
     call contr(mp_kvec(:,ib),qp0,kqp0)
     sumdena= 0.d0 
     do k= 2,np
      sumdena= sumdena+(value(mp_den(bbn1(k,ib)),mpq)-mpqt2)
     enddo
     start= start+(                  &
              mp_acoeff(0,ib)        & 
             +mp_acoeff(1,ib)*kqp0   &
             +mp_acoeff(2,ib)*vqp0   &
             +mp_acoeff(3,ib)*l3qp0  &
             +mp_acoeff(4,ib)*l4qp0)*sumdena 
    enddo  
   endif
!
!  contribution from the b sector:
!
   do ib= 1,dmns_2
    do k= 0,3
     qp0(k)= mpq(k)+mp_p0vecb(k,ib) 
    enddo
    call contr(mp_l5vec(:,ib),qp0,l3qp0)
    call contr(mp_l6vec(:,ib),qp0,l4qp0)
    call contr(mp_vvecb(:,ib),qp0,vqp0)
    sumdenb= 0.d0 
    do k= 3,np
     sumdenb= sumdenb+(value(mp_den(bbn2(k,ib)),mpq)-mpqt2)
    enddo
    if (iteration.eq.1) start= start+save_mp_bcoeff(0,ib)
    if (iteration.eq.2) start= start+(bset(ib,2)+bset(ib,3))/2.d0
    start=  start+(                            & 
             save_mp_bcoeff(1,ib)*l3qp0           &
            +save_mp_bcoeff(2,ib)*l4qp0           &
            +save_mp_bcoeff(3,ib)*vqp0            &
            +save_mp_bcoeff(4,ib)*(l3qp0)**2      &
            +save_mp_bcoeff(5,ib)*(l4qp0)**2      &
            +save_mp_bcoeff(6,ib)*vqp0**2         &
            +save_mp_bcoeff(7,ib)*vqp0*l3qp0      &
            +save_mp_bcoeff(8,ib)*vqp0*l4qp0)     &    
            +mp_bcoeff_2(ib)*sumdenb
   enddo  
!
!  contribution from the c sector:
!
   do ib= 1,dmns_3
    do k= 0,3 
     qp0(k)= mpq(k)+mp_p0vecc(k,ib) 
    enddo
    call contr(mp_l3vec(:,ib),qp0,l3qp0)
    call contr(mp_l4vec(:,ib),qp0,l4qp0)
    start= start+(mp_ccoeff_2(0,ib)               &
                 +mp_ccoeff_2(1,ib)*l3qp0         &
                 +mp_ccoeff_2(2,ib)*l4qp0)        
   enddo
   rat1_d= -start
   rat1_d=-rat1_d/6.d0 ! multiplying by the R_2 integral
   end subroutine getd_last1
!
   subroutine getd_last2(q,rat1_d)
    include 'cts_dpc.h'
     , intent(in), dimension(0:3) :: q
    include 'cts_mpc.h'
     , dimension(0:3) :: mpq
    include 'cts_mpc.h'
     , intent(out) :: rat1_d
    include 'cts_mpc.h'
     , dimension(1:(number_propagators-2)) :: x
    include 'cts_mpc.h'
     , dimension(0:(number_propagators-2)) :: dres
    include 'cts_mpc.h' 
     :: mpqt2_save 
    integer :: n,ntot,kmax 
    do k= 0,3; mpq(k)= q(k)*c1(p); enddo
    mpqt2_save= mpqt2        ! store the current value of qt2
    dres(0)= numd(number_propagators,mpq,0) ! compute sumd at x(0)= qt2
    kmax= number_propagators-2
    ntot= kmax+1
    do n= 1,ntot-1
     x(n)= mpqt2*exp(-2.d0*ci(p)*pi(p)*n/ntot)
    enddo
    do n= 1,ntot-1
     mpqt2= x(n)
     call getd(p,numdummy,number_propagators,dmr1)  
     dres(n)= numd(number_propagators,mpq,0) ! compute sumd at x(n)
    enddo
    rat1_d= 0.d0
    do n= 0,ntot-1
     rat1_d= rat1_d+dres(n)*exp(2.d0*ci(p)*pi(p)*n*kmax/ntot)
    enddo
    mpqt2= mpqt2_save         ! reload the initial value of qt2
    rat1_d= rat1_d/ntot/mpqt2**kmax
    rat1_d=-rat1_d/6.d0   ! multiplying by the R_2 integral
   end subroutine getd_last2
!
   subroutine getd_last3(q,rat1_d)
    include 'cts_dpc.h'
     , intent(in), dimension(0:3) :: q
    include 'cts_mpc.h'
     , dimension(0:3) :: mpq
    include 'cts_mpc.h'
     , intent(out) :: rat1_d
    include 'cts_mpc.h' 
     :: mpqt2_save 
    do k= 0,3; mpq(k)= q(k)*c1(p); enddo
    mpqt2_save= mpqt2        ! store the current value of qt2
    mpqt2= qt2_inf 
    call getd(p,numdummy,number_propagators,dmr1)  
    rat1_d= numd(number_propagators,mpq,0) ! compute sumd at qt2_inf
    rat1_d= rat1_d/mpqt2**(number_propagators-2)
    rat1_d=-rat1_d/6.d0   ! multiplying by the R_2 integral
    mpqt2= mpqt2_save         ! reload the initial value of qt2
   end subroutine getd_last3
!
   subroutine load_set(n)
   integer, intent(in) :: n
   if (number_propagators.ge.1) then
    do ib= 1,dmns_1
     aset(ib,n)= mp_acoeff(0,ib)
    enddo
   endif
   if (number_propagators.ge.2) then
    do ib= 1,dmns_2
     bset (ib,n)= mp_bcoeff(0,ib)
     bset3(ib,n)= mp_bcoeff(3,ib)
    enddo
   endif
   if (number_propagators.ge.3) then
    do ib= 1,dmns_3
     cset(ib,n)= mp_ccoeff(0,ib)
    enddo
   endif
   if (number_propagators.ge.4) then
    do ib= 1,dmns_4
     dset(ib,n)= mp_dcoeff(0,ib)
    enddo
   endif
   end subroutine load_set
!
   subroutine load_rat(n)
   integer, intent(in) :: n
   rset(n)= mp_rat1 
   end subroutine load_rat
!
   subroutine finalize_sets
   include 'cts_mpc.h' 
    :: r_aus,r_aus1,r_aus2,r_aus3
   if     (dmr.ge.0) then
    if (number_propagators.ge.1) then
     do ib= 1,dmns_1
      mp_acoeff(0,ib) = aset(ib,3)
     enddo
    endif
    if (number_propagators.ge.2) then
     do ib= 1,dmns_2
      mp_bcoeff(0,ib)= (bset(ib,2)+bset(ib,3))/2.d0 
     enddo
    endif
    if (number_propagators.ge.3) then
     do ib= 1,dmns_3
      mp_ccoeff(0,ib)= (cset(ib,2)+cset(ib,3))/2.d0
      mp_ccoeff(1,ib)= save_mp_ccoeff(1,ib) ! to put it to the true value  
      mp_ccoeff(2,ib)= save_mp_ccoeff(2,ib) ! to put it to the true value
     enddo
    endif
   elseif (dmr.eq.-1) then
!
!   Restores all values
!
!!    if (number_propagators.ge.4) call get_dcoeff(p)
!!    if (number_propagators.ge.3) call get_ccoeff(p)
!!    if (number_propagators.ge.2) call get_bcoeff(p)
!!    if (number_propagators.ge.1) call get_acoeff(p)
!
!   The new determinations of the non-spurious coefficients 
!
    r_aus = (qt2_2-qt2_1)
    r_aus3= (qt2_2+qt2_1)
    if (number_propagators.ge.1) then
     do ib= 1,dmns_1
      mp_acoeff(0,ib)= (qt2_2*aset(ib,2)-qt2_1*aset(ib,4))/r_aus
     enddo
    endif
    if (number_propagators.ge.2) then
     do ib= 1,dmns_2
      mp_bcoeff(0,ib)= (qt2_2* bset(ib,2) -qt2_1*bset(ib,4))/r_aus
      mp_bcoeff(3,ib)= (qt2_2*bset3(ib,2)-qt2_1*bset3(ib,4))/r_aus
     enddo
    endif
    if (number_propagators.ge.3) then
     do ib= 1,dmns_3
      r_aus1= (cset(ib,2)+cset(ib,3))/2.d0
      r_aus2= cset(ib,4)-qt2_2*(cset(ib,2)-cset(ib,3))/2.d0/qt2_1
      mp_ccoeff(0,ib)= (r_aus1*qt2_2**2-r_aus2*qt2_1**2)/r_aus/r_aus3
     enddo
    endif
    if (number_propagators.ge.4) then
     do ib= 1,dmns_4
      r_aus1= (dset(ib,2)+dset(ib,3))/2.d0
      r_aus2= dset(ib,4)-qt2_2*(dset(ib,2)-dset(ib,3))/2.d0/qt2_1
      mp_dcoeff(0,ib)= (r_aus1*qt2_2**2-r_aus2*qt2_1**2)/r_aus/r_aus3
     enddo
    endif
   else
     stop 'dmr value not allowed in subroutine finalize_sets!'
   endif
   save_mp_rat1= rset(1) 
   mp_rat1     = rset(2) 
   end subroutine finalize_sets
!
  end subroutine mp_get_coefficients
!
  subroutine dp_put_dcoeff(p)
   include 'cts_dpr.h'
    , intent(in) :: p
   integer :: i,j
   do i= 0,1; do j= 1,dmns_d
    save_dcoeff(i,j)= dcoeff(i,j)
   enddo; enddo
  end subroutine dp_put_dcoeff
!
  subroutine dp_get_dcoeff(p)
   include 'cts_dpr.h'
    , intent(in) :: p
   integer :: i,j
   do i= 0,1; do j= 1,dmns_d
    dcoeff(i,j)= save_dcoeff(i,j) 
   enddo; enddo
  end subroutine dp_get_dcoeff
!
  subroutine dp_put_ccoeff(p)
   include 'cts_dpr.h'
    , intent(in) :: p
   integer :: i,j
   do i= 0,6; do j= 1,dmns_c
    save_ccoeff(i,j)= ccoeff(i,j)
   enddo; enddo
  end subroutine dp_put_ccoeff
!
  subroutine dp_get_ccoeff(p)
   include 'cts_dpr.h'
    , intent(in) :: p
   integer :: i,j
   do i= 0,6; do j= 1,dmns_c
    ccoeff(i,j)= save_ccoeff(i,j) 
   enddo; enddo
  end subroutine dp_get_ccoeff
!
  subroutine dp_put_bcoeff(p)
   include 'cts_dpr.h'
    , intent(in) :: p
   integer :: i,j
   do i= 0,8; do j= 1,dmns_b
    save_bcoeff(i,j)= bcoeff(i,j)
   enddo; enddo
  end subroutine dp_put_bcoeff
!
  subroutine dp_get_bcoeff(p)
   include 'cts_dpr.h'
    , intent(in) :: p
   integer :: i,j
   do i= 0,8; do j= 1,dmns_b
    bcoeff(i,j)= save_bcoeff(i,j) 
   enddo; enddo
  end subroutine dp_get_bcoeff
!
  subroutine dp_put_acoeff(p)
   include 'cts_dpr.h'
    , intent(in) :: p
   integer :: i,j
   do i= 0,4; do j= 1,dmns_a
    save_acoeff(i,j)= acoeff(i,j)
   enddo; enddo
  end subroutine dp_put_acoeff
!
  subroutine dp_get_acoeff(p)
   include 'cts_dpr.h'
    , intent(in) :: p
   integer :: i,j
   do i= 0,4; do j= 1,dmns_a
    acoeff(i,j)= save_acoeff(i,j) 
   enddo; enddo
  end subroutine dp_get_acoeff
!
  subroutine mp_put_dcoeff(p)
   include 'cts_mpr.h'
    , intent(in) :: p
   integer :: i,j
   do i= 0,1; do j= 1,dmns_d
    save_mp_dcoeff(i,j)= mp_dcoeff(i,j)
   enddo; enddo
  end subroutine mp_put_dcoeff
!
  subroutine mp_get_dcoeff(p)
   include 'cts_mpr.h'
    , intent(in) :: p
   integer :: i,j
   do i= 0,1; do j= 1,dmns_d
    mp_dcoeff(i,j)= save_mp_dcoeff(i,j) 
   enddo; enddo
  end subroutine mp_get_dcoeff
!
  subroutine mp_put_ccoeff(p)
   include 'cts_mpr.h'
    , intent(in) :: p
   integer :: i,j
   do i= 0,6; do j= 1,dmns_c
    save_mp_ccoeff(i,j)= mp_ccoeff(i,j)
   enddo; enddo
  end subroutine mp_put_ccoeff
!
  subroutine mp_get_ccoeff(p)
   include 'cts_mpr.h'
    , intent(in) :: p
   integer :: i,j
   do i= 0,6; do j= 1,dmns_c
    mp_ccoeff(i,j)= save_mp_ccoeff(i,j) 
   enddo; enddo
  end subroutine mp_get_ccoeff
!
  subroutine mp_put_bcoeff(p)
   include 'cts_mpr.h'
    , intent(in) :: p
   integer :: i,j
   do i= 0,8; do j= 1,dmns_b
    save_mp_bcoeff(i,j)= mp_bcoeff(i,j)
   enddo; enddo
  end subroutine mp_put_bcoeff
!
  subroutine mp_get_bcoeff(p)
   include 'cts_mpr.h'
    , intent(in) :: p
   integer :: i,j
   do i= 0,8; do j= 1,dmns_b
    mp_bcoeff(i,j)= save_mp_bcoeff(i,j) 
   enddo; enddo
  end subroutine mp_get_bcoeff
!
  subroutine mp_put_acoeff(p)
   include 'cts_mpr.h'
    , intent(in) :: p
   integer :: i,j
   do i= 0,4; do j= 1,dmns_a
    save_mp_acoeff(i,j)= mp_acoeff(i,j)
   enddo; enddo
  end subroutine mp_put_acoeff
!
  subroutine mp_get_acoeff(p)
   include 'cts_mpr.h'
    , intent(in) :: p
   integer :: i,j
   do i= 0,4; do j= 1,dmns_a
    mp_acoeff(i,j)= save_mp_acoeff(i,j) 
   enddo; enddo
  end subroutine mp_get_acoeff
!
  subroutine dp_getd(p,numdummy,number_propagators,dmr)
   external numdummy
   include 'cts_dpr.h'
    , intent(in) :: p
   integer, intent(in) :: dmr
   integer, intent(in) :: number_propagators
   type(solcut4) :: cut4 
   integer :: i,ib,k
   integer :: np
   include 'cts_dpc.h'
    , dimension(0:3) :: qpp0
   include 'cts_dpc.h' 
    :: tqpp0
   include 'cts_dpc.h' 
    :: f1,f2
   dcoeff= c0(p)
   np= number_propagators
   if     (np.lt.4) then
   else
    do ib= 1,dmns_4
     call cut(den(bbn4(1,ib)),den(bbn4(2,ib)),den(bbn4(3,ib)), &
              den(bbn4(4,ib)),cut4) 
     do k= 0,3
      tvec(k,ib) =  cut4%t(k)
      p0vecd(k,ib)=  den(bbn4(1,ib))%p(k)
      qpp0(k)= cut4%q(k,1)+p0vecd(k,ib)
     enddo
     call contr(tvec(:,ib),qpp0,tqpp0)
     f1= fnum(1) 
     f2= fnum(2) 
     dcoeff(0,ib)= 0.5d0*(f1+f2)
     dcoeff(1,ib)= 0.5d0*(f1-f2)/tqpp0
    enddo
   endif
   contains
!
   function fnum(j)
   use inout
   use scale
   include 'cts_dpr.h' 
    :: p
   include 'cts_dpc.h' 
    :: fnum,allden
   integer, intent(in) :: j
   integer :: k
   allden= c1(p) 
   do k= 5,np
    allden= allden*vden(den(bbn4(k,ib))%i,j)
   enddo
   mprec=.false.
   dpq  = cut4%q(:,j)
! comment
   if (ext_num_for_r1) then
    call numfunc(numdummy)
   else 
    if (rational.and.(.not.inf)) then 
     call numfuncrec(np,cut4%q(:,j),j,dpres) 
    else
     call numfunc(numdummy)
    endif
   endif
! comment
   fnum= dpres/allden
   end function fnum
  end subroutine dp_getd 
!
  subroutine dp_getc(p,numdummy,number_propagators,dmr)
   external numdummy
   include 'cts_dpr.h'
    , intent(in) :: p
   integer, intent(in) :: number_propagators
   integer, intent(in) :: dmr
   type(solcut3) :: cut3 
   integer :: i,ib,k,m 
   integer :: np
   include 'cts_dpc.h' 
    :: gm,cc,tau,cph,f1,f2,f3,f4,&
      f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16
   include 'cts_dpc.h' 
    :: sum,fv(0:3),gv(0:3),r(0:7),ccpar(-3:3)
   include 'cts_dpc.h' 
    :: cdelta,rp(0:7)= 0.d0,fvp(0:3),cden1,cden2,c34
   include 'cts_dpc.h' 
    :: ca1,ca2,ca3,ca4,ca5
   if (.not.rational) ccoeff= c0(p)
   if (rational)     call get_ccoeff(p)
   np= number_propagators
   if (dmr.ge.4) return
   if     (np.lt.3) then
   else   
    if     (dmr.eq.3) then
     do ib= 1,dmns_3
      call cut(den(bbn3(1,ib)),den(bbn3(2,ib)),den(bbn3(3,ib)),cut3,dmr) 
      ccoeff(0,ib)= fnum(1)
      c4_rat1(ib) = 0.d0
      do k= 0,3
       l3vec(k,ib) =  cut3%l3(k)
       l4vec(k,ib) =  cut3%l4(k)
       p0vecc(k,ib) =  den(bbn3(1,ib))%p(k) 
      enddo
     enddo
    elseif (dmr.eq.2) then
     do ib= 1,dmns_3
      call cut(den(bbn3(1,ib)),den(bbn3(2,ib)),den(bbn3(3,ib)),cut3,dmr) 
      do k= 0,3
       l3vec(k,ib) =  cut3%l3(k)
       l4vec(k,ib) =  cut3%l4(k)
       p0vecc(k,ib) =  den(bbn3(1,ib))%p(k) 
      enddo
      cc = cut3%cc
      tau= cut3%tau
      c34= -2.d0*cut3%gm
      cden1= cc*tau
      cden2= c1(p)/tau**4-cc**4*tau**4
      cdelta= cden2*c34**4
      r(0) = fnum(1)
      r(4) = fnum(2)
      rp(0)= fnum(3) 
      ca1= 0.5d0*(r(0)-r(4))
      ca2= 0.5d0*(r(0)+r(4))
      ca3= rp(0)-ca2      
      ccoeff(0,ib)= ca2
      ccoeff(1,ib)= c34**3*(ca1/tau-cden1*ca3)*(c1(p)/tau**2+cden1**2)/cdelta
      ccoeff(2,ib)= c34**3*(ca3/tau-cden1*ca1)*(c1(p)/tau**2+cden1**2)/cdelta
      c4_rat1(ib) = 0.d0
     enddo
    elseif (dmr.eq.1) then
     do ib= 1,dmns_3
      call cut(den(bbn3(1,ib)),den(bbn3(2,ib)),den(bbn3(3,ib)),cut3,dmr) 
      if (rational) then
       r(0) = fnum(1)-fnuminv(1,1)
       r(4) = fnum(2)-fnuminv(2,1)
       ccoeff(0,ib)= 0.5d0*(r(0)+r(4))    
       c4_rat1(ib) = 0.d0
       cycle
      else
       do k= 0,3
        l3vec(k,ib) =  cut3%l3(k)
        l4vec(k,ib) =  cut3%l4(k)
        p0vecc(k,ib) =  den(bbn3(1,ib))%p(k) 
       enddo
      endif 
      cc = cut3%cc
      tau= cut3%tau
      c34= -2.d0*cut3%gm
      cden1= cc*tau
      cden2= c1(p)/tau**4-cc**4*tau**4
      cdelta= cden2*c34**4
      r(0) = fnum(1)
      r(4) = fnum(2)
      r(6) = fnum(3)
      rp(0)= fnum(4) 
      rp(4)= fnum(5) 
      ca1= 0.5d0*(r(0)-r(4))
      ca2= 0.5d0*(r(0)+r(4))
      ca3= 0.5d0*(rp(0)-rp(4)) 
      ca4= 0.5d0*(rp(0)+rp(4)) 
      ccoeff(1,ib)= c34**3*(ca1/tau-cden1*ca3)*(c1(p)/tau**2+cden1**2)/cdelta
      ccoeff(2,ib)= c34**3*(ca3/tau-cden1*ca1)*(c1(p)/tau**2+cden1**2)/cdelta
      ca5= r(6)+ci(p)*c34*(ccoeff(1,ib)/tau-ccoeff(2,ib)*cden1)
      ccoeff(0,ib)= 0.5d0*(ca2+ca5)
      ca1= ca2-ccoeff(0,ib)
      ca2= ca4-ccoeff(0,ib)
      ccoeff(3,ib)= c34**2/cdelta*(ca1/tau**2-cden1**2*ca2)
      ccoeff(4,ib)= c34**2/cdelta*(ca2/tau**2-cden1**2*ca1)
      c4_rat1(ib) = 0.d0
     enddo
    elseif (dmr.eq.0) then
     do ib= 1,dmns_3
      call cut(den(bbn3(1,ib)),den(bbn3(2,ib)),den(bbn3(3,ib)),cut3,dmr) 
      cc = cut3%cc
      tau= cut3%tau
      c34= -2.d0*cut3%gm
      cden1= cc*tau
      cden2= c1(p)/tau**4-cc**4*tau**4
      cdelta= cden2*c34**4
      if (rational) then
       r(0) = fnum(1)-fnuminv(1,2)
       r(4) = fnum(2)-fnuminv(2,2)
       rp(0)= fnum(3)-fnuminv(3,2)
       ca1= 0.5d0*(r(0)-r(4))
       ca2= 0.5d0*(r(0)+r(4))
       ca3= rp(0)-ca2      
       ccoeff(0,ib)= ca2
       ccoeff(1,ib)= c34**3*(ca1/tau-cden1*ca3)*(c1(p)/tau**2+cden1**2)/cdelta
       ccoeff(2,ib)= c34**3*(ca3/tau-cden1*ca1)*(c1(p)/tau**2+cden1**2)/cdelta
       c4_rat1(ib) = 0.d0
       cycle
      else
       do k= 0,3
        l3vec(k,ib) =  cut3%l3(k)
        l4vec(k,ib) =  cut3%l4(k)
        p0vecc(k,ib) =  den(bbn3(1,ib))%p(k) 
       enddo
      endif
      r(0) = fnum(1)
      r(2) = fnum(2)
      r(4) = fnum(3)
      r(6) = fnum(4)
      rp(0)= fnum(5) 
      rp(2)= fnum(6) 
      rp(4)= fnum(7) 
      do k= 0,3
       sum= 0.d0
       do m= 0,3
         sum= sum+r(2*m)*(cexp2(p))**(-m*k) 
       enddo
       fv(k)= sum/4.d0 
      enddo
      rp(6)= 4.d0*fv(0)-rp(0)-rp(2)-rp(4) 
      do k= 0,3
       sum= 0.d0
       do m= 0,3
         sum= sum+rp(2*m)*(cexp2(p))**(-m*k) 
       enddo
       fvp(k)= sum/4.d0 
      enddo
      ccoeff(0,ib)= fv(0)
      ccoeff(3,ib)= (fv(2)/tau**2-fvp(2)*cden1**2)*c34**2/cdelta
      ccoeff(4,ib)= (fvp(2)/tau**2-fv(2)*cden1**2)*c34**2/cdelta
      ccoeff(1,ib)= (fv(1)/tau**3-fvp(3)*cden1**3)*c34**3/cdelta
      ccoeff(6,ib)= (fvp(3)/tau**1-fv(1)*cden1**1)*c34**1/cdelta
      ccoeff(5,ib)= (fv(3)/tau**1-fvp(1)*cden1**1)*c34**1/cdelta
      ccoeff(2,ib)= (fvp(1)/tau**3-fv(3)*cden1**3)*c34**3/cdelta
      c4_rat1(ib) = 0.d0
     enddo
    elseif (dmr.eq.-1) then
! comment: later on c_5 and c_6 not to be computed with rational=.true. 
     do ib= 1,dmns_3
      call cut(den(bbn3(1,ib)),den(bbn3(2,ib)),den(bbn3(3,ib)),cut3,dmr) 
      do k= 0,3
       l3vec(k,ib) =  cut3%l3(k)
       l4vec(k,ib) =  cut3%l4(k)
       p0vecc(k,ib) =  den(bbn3(1,ib))%p(k) 
      enddo
      cc = cut3%cc
      tau= cut3%tau
      c34= -2.d0*cut3%gm
      cden1= cc*tau
      cden2= c1(p)/tau**4-cc**4*tau**4
      cdelta= cden2*c34**4
      r(0) = fnum(1)
      r(2) = fnum(2)
      r(4) = fnum(3)
      r(6) = fnum(4)
      rp(0)= fnum(5) 
      rp(2)= fnum(6) 
      rp(4)= fnum(7) 
      do k= 0,3
       sum= 0.d0
       do m= 0,3
         sum= sum+r(2*m)*(cexp2(p))**(-m*k) 
       enddo
       fv(k)= sum/4.d0 
      enddo
      rp(6)= 4.d0*fv(0)-rp(0)-rp(2)-rp(4) 
      do k= 0,3
       sum= 0.d0
       do m= 0,3
         sum= sum+rp(2*m)*(cexp2(p))**(-m*k) 
       enddo
       fvp(k)= sum/4.d0 
      enddo
      ccoeff(0,ib)= fv(0)
      ccoeff(3,ib)= (fv(2)/tau**2-fvp(2)*cden1**2)*c34**2/cdelta
      ccoeff(4,ib)= (fvp(2)/tau**2-fv(2)*cden1**2)*c34**2/cdelta
      ccoeff(1,ib)= (fv(1)/tau**3-fvp(3)*cden1**3)*c34**3/cdelta
      ccoeff(6,ib)= (fvp(3)/tau**1-fv(1)*cden1**1)*c34**1/cdelta
      ccoeff(5,ib)= (fv(3)/tau**1-fvp(1)*cden1**1)*c34**1/cdelta
      ccoeff(2,ib)= (fvp(1)/tau**3-fv(3)*cden1**3)*c34**3/cdelta
      c4_rat1(ib) = cut3%rat1
     enddo
    else
     print*,'In subroutine getc '
     print*,'dmr=',dmr,' not allowed'
     stop
    endif
   endif
   contains
!
   function fnum(j)
   use inout
   use scale
   include 'cts_dpr.h' 
    :: p
   include 'cts_dpc.h' 
    :: fnum,allden
   integer, intent(in) :: j
   integer :: k
   allden= c1(p) 
   do k= 4,np
    allden= allden*vden(den(bbn3(k,ib))%i,j)
   enddo
   mprec=.false.
   dpq  = cut3%q(:,j)
! comment
   if (ext_num_for_r1) then
    call numfunc(numdummy)
   else 
    if (rational) then
     call numfuncrec(np,cut3%q(:,j),j,dpres) 
    else
     call numfunc(numdummy)
    endif
   endif
! comment
   fnum=  (dpres-numd(np,cut3%q(:,j),j))/allden
   end function fnum
! 
   function fnuminv(j,level)
   include 'cts_dpr.h' 
    :: p
   include 'cts_dpc.h' 
    :: sum,fnuminv,l3qp0,l4qp0
   include 'cts_dpc.h'
    , dimension(0:3) :: qp0
   integer, intent(in) :: j,level
   integer :: k
   do k= 0,3
    qp0(k)= cut3%q(k,j)+p0vecc(k,ib)
   enddo
   call contr(qp0,l3vec(:,ib),l3qp0)
   call contr(qp0,l4vec(:,ib),l4qp0)
   sum= c0(p)
   if (level.le.1) sum= sum+ccoeff(1,ib)*l3qp0   +ccoeff(2,ib)*l4qp0        
   if (level.le.2) sum= sum+ccoeff(3,ib)*l3qp0**2+ccoeff(4,ib)*l4qp0**2     
   if (level.le.3) sum= sum+ccoeff(5,ib)*l3qp0**3+ccoeff(6,ib)*l4qp0**3     
   fnuminv= sum
   end function fnuminv
!
  end subroutine dp_getc 
!
  subroutine dp_getb(p,numdummy,number_propagators,dmr)
   external numdummy
   include 'cts_dpr.h'
    , intent(in) :: p
   integer, intent(in) :: number_propagators
   integer, intent(in) :: dmr
   type(solcut2) :: cut2 
   integer :: i,ib,k
   integer :: np
   include 'cts_dpc.h' 
    :: gm,ulambda,usigma,zlambda,zsigma
   include 'cts_dpc.h' 
    :: rden,caus1,caus2,caus3,cflambda,cfsigma
   include 'cts_dpc.h' 
    :: f1,f2,f3,f4,f5,f6,f7,f8,f9,f10
   include 'cts_dpc.h' 
    :: f11,f12,f13,f14,f15,f16,f17,f18
   include 'cts_dpc.h' 
    :: r(0:4)= 0,rp(0:2)= 0,fv(0:2)= 0,fvp(0:2)= 0
   include 'cts_dpc.h' 
    :: rr0,rr2,tau,taul,cf0,cdelta,sf6,sf9
   if (.not.rational) bcoeff= c0(p)
   if (rational)     call get_bcoeff(p)
   np= number_propagators  
   if (dmr.ge.3) return
   if     (np.lt.2) then
   else
    if     (dmr.eq.2) then
     do ib= 1,dmns_2
      call cut(den(bbn2(1,ib)),den(bbn2(2,ib)),cut2,dmr) 
      do k= 0,3 
       p0vecb(k,ib)=  den(bbn2(1,ib))%p(k) 
       l5vec(k,ib)=  cut2%l3(k)
       l6vec(k,ib)=  cut2%l4(k)
       vvecb(k,ib)=  cut2%v(k)
      enddo
      gm= cut2%gm
      bcoeff(0,ib)=  fnum(1)
      vveck1(ib) =  gm/2.d0
      b_rat1(ib) = cut2%rat1
      b3_rat1(ib)= 0.d0
     enddo
    elseif (dmr.eq.1) then
     do ib= 1,dmns_2
      call cut(den(bbn2(1,ib)),den(bbn2(2,ib)),cut2,dmr) 
      do k= 0,3 
       p0vecb(k,ib)=  den(bbn2(1,ib))%p(k) 
       l5vec(k,ib)=  cut2%l3(k)
       l6vec(k,ib)=  cut2%l4(k)
       vvecb(k,ib)=  cut2%v(k)
      enddo 
      tau= cut2%tau 
      taul= cut2%taul
      gm = cut2%gm
      cf0= cut2%cf0 
      cdelta= c1(p)-tau**4*cf0**2
      r(0) = fnum(1)
      r(3) = fnum(2)
      rp(0)= fnum(3)
      bcoeff(0,ib)= 0.5d0*(r(0)+r(3))
      fv(0) = 0.5d0*(r(0)-r(3))
      fvp(0)= rp(0)-bcoeff(0,ib)
      bcoeff(1,ib)= -0.5d0*tau/gm/cdelta*(fv(0)-tau**2*cf0*fvp(0))
      bcoeff(2,ib)= -0.5d0*tau/gm/cdelta*(fvp(0)-tau**2*cf0*fv(0))
      bcoeff(4,ib)= 0.d0
      bcoeff(5,ib)= 0.d0
      bcoeff(3,ib)= 2.d0/gm/lambda(p)*sfun(4)
      vveck1(ib) =  gm/2.d0
      b_rat1(ib) = cut2%rat1
      b3_rat1(ib)= 0.d0
     enddo
    elseif (dmr.eq.0) then
     do ib= 1,dmns_2
      call cut(den(bbn2(1,ib)),den(bbn2(2,ib)),cut2,dmr) 
      if (rational) then
       bcoeff(0,ib)= 0.d0
       bcoeff(0,ib)= sfun(1)
       cycle
      else
       do k= 0,3 
        p0vecb(k,ib)=  den(bbn2(1,ib))%p(k) 
        l5vec(k,ib)=  cut2%l3(k)
        l6vec(k,ib)=  cut2%l4(k)
        vvecb(k,ib)=  cut2%v(k)
       enddo
      endif 
      tau= cut2%tau
      taul= cut2%taul
      gm = cut2%gm
      cf0= cut2%cf0 
      cdelta= c1(p)-tau**6*cf0**3
      r(0) = fnum(1)
      r(2) = fnum(2)
      r(4) = fnum(3)
      rp(0)= fnum(4)
      rp(2)= fnum(5)
      fv(0)= (r(0)+r(2)+r(4))/3.d0
      fv(1)= (r(0)+r(2)/cexp3(p)**2+r(4)/cexp3(p)**4)/3.d0
      fv(2)= (r(0)+r(2)/cexp3(p)**4+r(4)/cexp3(p)**8)/3.d0
      rr0= rp(0)-fv(0)
      rr2= rp(2)-fv(0)
      fvp(2)= (rr2-rr0*cexp3(p)**2)/(cexp3(p)**4-cexp3(p)**2)
      fvp(1)= rr0-fvp(2)
      bcoeff(0,ib)= fv(0) 
      bcoeff(1,ib)=   -tau/2.d0/gm*(fv(1)-tau**4*cf0**2*fvp(2))/cdelta
      bcoeff(2,ib)=   -tau/2.d0/gm*(fvp(1)-tau**4*cf0**2*fv(2))/cdelta
      bcoeff(4,ib)= tau**2/4.d0/gm/gm*(fv(2)-tau**2*cf0*fvp(1))/cdelta
      bcoeff(5,ib)= tau**2/4.d0/gm/gm*(fvp(2)-tau**2*cf0*fv(1))/cdelta
      cflambda= cut2%cflambda
      cfsigma = cut2%cfsigma
      r(0) = sfun(6)
      r(2) = sfun(7)
      fv(0)= (r(0)+r(2))/2.d0
      fv(1)= (r(0)-r(2))/2.d0
      rp(0)= sfun(8)-fv(0)
      fvp(1)= rp(0)
      cdelta= c1(p)-taul**4*cflambda**2
      bcoeff(8,ib)=-taul/gm/gm/lambda(p)*(fvp(1)-fv(1)*cflambda*taul**2)/cdelta
      bcoeff(7,ib)=-taul/gm/gm/lambda(p)*(fv(1)-fvp(1)*cflambda*taul**2)/cdelta
      sf6= r(0)
      sf9= sfun(9)
      ulambda= ufun(6,sf6)
      usigma = ufun(9,sf9)
      cdelta= lambda(p)*sigma(p)*(sigma(p)-lambda(p))
      bcoeff(3,ib)= 2.d0/gm/cdelta*(ulambda*sigma(p)**2-usigma*lambda(p)**2)
      bcoeff(6,ib)= 4.d0/gm/gm/cdelta*(usigma*lambda(p)-ulambda*sigma(p))
      vveck1(ib) = gm/2.d0
      b_rat1(ib) = cut2%rat1
      b3_rat1(ib)= 0.d0
     enddo
    elseif (dmr.eq.-1) then
! comment: later on b_4-b_8 not to be computed with rational=.true. 
     do ib= 1,dmns_2
      call cut(den(bbn2(1,ib)),den(bbn2(2,ib)),cut2,dmr) 
      do k= 0,3 
       p0vecb(k,ib)=  den(bbn2(1,ib))%p(k) 
       l5vec(k,ib)=  cut2%l3(k)
       l6vec(k,ib)=  cut2%l4(k)
       vvecb(k,ib)=  cut2%v(k)
      enddo
      tau= cut2%tau
      taul= cut2%taul
      gm = cut2%gm
      cf0= cut2%cf0 
      cdelta= c1(p)-tau**6*cf0**3
      r(0) = fnum(1)
      r(2) = fnum(2)
      r(4) = fnum(3)
      rp(0)= fnum(4)
      rp(2)= fnum(5)
      fv(0)= (r(0)+r(2)+r(4))/3.d0
      fv(1)= (r(0)+r(2)/cexp3(p)**2+r(4)/cexp3(p)**4)/3.d0
      fv(2)= (r(0)+r(2)/cexp3(p)**4+r(4)/cexp3(p)**8)/3.d0
      rr0= rp(0)-fv(0)
      rr2= rp(2)-fv(0)
      fvp(2)= (rr2-rr0*cexp3(p)**2)/(cexp3(p)**4-cexp3(p)**2)
      fvp(1)= rr0-fvp(2)
      bcoeff(0,ib)= fv(0) 
      bcoeff(1,ib)=   -tau/2.d0/gm*(fv(1)-tau**4*cf0**2*fvp(2))/cdelta
      bcoeff(2,ib)=   -tau/2.d0/gm*(fvp(1)-tau**4*cf0**2*fv(2))/cdelta
      bcoeff(4,ib)= tau**2/4.d0/gm/gm*(fv(2)-tau**2*cf0*fvp(1))/cdelta
      bcoeff(5,ib)= tau**2/4.d0/gm/gm*(fvp(2)-tau**2*cf0*fv(1))/cdelta
      cflambda= cut2%cflambda
      cfsigma = cut2%cfsigma
      r(0) = sfun(6)
      r(2) = sfun(7)
      fv(0)= (r(0)+r(2))/2.d0
      fv(1)= (r(0)-r(2))/2.d0
      rp(0)= sfun(8)-fv(0)
      fvp(1)= rp(0)
      cdelta= c1(p)-taul**4*cflambda**2
      bcoeff(8,ib)=-taul/gm/gm/lambda(p)*(fvp(1)-fv(1)*cflambda*taul**2)/cdelta
      bcoeff(7,ib)=-taul/gm/gm/lambda(p)*(fv(1)-fvp(1)*cflambda*taul**2)/cdelta
      sf6= r(0)
      sf9= sfun(9)
      ulambda= ufun(6,sf6)
      usigma = ufun(9,sf9)
      cdelta= lambda(p)*sigma(p)*(sigma(p)-lambda(p))
      bcoeff(3,ib)= 2.d0/gm/cdelta*(ulambda*sigma(p)**2-usigma*lambda(p)**2)
      bcoeff(6,ib)= 4.d0/gm/gm/cdelta*(usigma*lambda(p)-ulambda*sigma(p))
      vveck1(ib) = gm/2.d0
      b_rat1(ib) = cut2%rat1
      b3_rat1(ib)= cut2%rat1t
     enddo
    else
     print*,'In subroutine getb '
     print*,'dmr=',dmr,' not allowed'
     stop
    endif
   endif
   contains
!
   function fnum(j)
   use inout
   use scale
   include 'cts_dpr.h' 
    :: p
   include 'cts_dpc.h' 
    :: fnum,allden
   integer, intent(in) :: j
   integer :: k
   allden= c1(p) 
   do k= 3,np
    allden= allden*vden(den(bbn2(k,ib))%i,j)
   enddo
   mprec=.false.
   dpq  = cut2%q(:,j)
! comment
   if (ext_num_for_r1) then
    call numfunc(numdummy)
   else 
    if (rational) then
     call numfuncrec(np,cut2%q(:,j),j,dpres) 
    else
     call numfunc(numdummy)
    endif
   endif
! comment
   fnum =  (dpres                                  &
           -numd(number_propagators,cut2%q(:,j),j) &
           -numc(number_propagators,cut2%q(:,j),j))/allden
   end function fnum
!
   function sfun(j) ! function added
   integer, intent(in) :: j
   include 'cts_dpc.h' 
    :: sfun,l3qp0,l4qp0
   include 'cts_dpc.h'
    , dimension(0:3) :: qp0
   integer :: k
   do k= 0,3
    qp0(k)= cut2%q(k,j)+p0vecb(k,ib)
   enddo
   call contr(qp0,l5vec(:,ib),l3qp0)
   call contr(qp0,l6vec(:,ib),l4qp0)
   sfun= fnum(j)-bcoeff(0,ib)          &
                -bcoeff(1,ib)*l3qp0    &
                -bcoeff(2,ib)*l4qp0    &
                -bcoeff(4,ib)*l3qp0**2 &
                -bcoeff(5,ib)*l4qp0**2
   end function sfun
!
   function ufun(j,sf) ! function added
   integer, intent(in) :: j
   include 'cts_dpc.h'
    , intent(in) :: sf
   include 'cts_dpc.h' 
    :: ufun,l3qp0,l4qp0,vqp0
   include 'cts_dpc.h'
    , dimension(0:3) :: qp0
   integer :: k
   do k= 0,3
    qp0(k)= cut2%q(k,j)+p0vecb(k,ib)
   enddo
   call contr(qp0,l5vec(:,ib),l3qp0)
   call contr(qp0,l6vec(:,ib),l4qp0)
   call contr(qp0,vvecb(:,ib),vqp0)
   ufun= sf-bcoeff(7,ib)*vqp0*l3qp0 &
           -bcoeff(8,ib)*vqp0*l4qp0
   end function ufun
  end subroutine dp_getb 
!
  subroutine dp_geta_oldbase(p,numdummy,number_propagators,dmr)
   external numdummy
   include 'cts_dpr.h'
    , intent(in) :: p
   integer, intent(in) :: dmr
   integer, intent(in) :: number_propagators
   type(solcut1) :: cut1 
   integer :: i,ib,k
   integer :: np
   include 'cts_dpc.h' 
    :: gm,cf0,r1,r2,rden,f1,f2,f3,f4,f5
   acoeff= c0(p)
   np= number_propagators
   if (dmr.ge.2) return
   if     (np.lt.1) then
    print*,'In subroutine geta '
    print*,'number_propagators=', number_propagators,' not allowed'
    stop
   else 
    if     (dmr.eq.1) then
     do ib= 1,dmns_1
      call cut(den(bbn1(1,ib)),cut1,dmr) 
      acoeff(0,ib)= fnum(1)
      do k= 0,3 
       p0veca(k,ib)= den(bbn1(1,ib))%p(k)
       l7vec(k,ib)= cut1%l3(k)
       l8vec(k,ib)= cut1%l4(k)
       vveca(k,ib)= cut1%v(k)
       kvec(k,ib) = cut1%k(k)
      enddo
     enddo
    elseif (dmr.eq.0) then
     do ib= 1,dmns_1
      call cut(den(bbn1(1,ib)),cut1,dmr) 
      f1= fnum(1)
      f2= fnum(2)
      f3= fnum(3)
      f4= fnum(4)
      f5= fnum(5)
      gm = cut1%gm
      cf0= cut1%cf0 
      acoeff(0,ib)=   0.5d0*(f1+f2)
      acoeff(1,ib)= 2.d0/gm*(f3-f1)
      acoeff(2,ib)= 2.d0/gm*(f4-f1)
      r1= -0.5d0*(f1-acoeff(0,ib))/gm
      r2= -0.5d0*(f5-acoeff(0,ib))/gm
      rden= cexpk1(p)*tau12(p)-cf0**2/cexpk1(p)/tau12(p)
      acoeff(3,ib)= (r1*cexpk1(p)*tau12(p)-cf0*r2)/rden*tau11(p)
      acoeff(4,ib)= (r2-cf0*r1/cexpk1(p)/tau12(p))/rden/tau11(p)
      do k= 0,3 
       p0veca(k,ib)= den(bbn1(1,ib))%p(k)
       l7vec(k,ib)= cut1%l3(k)
       l8vec(k,ib)= cut1%l4(k)
       vveca(k,ib)= cut1%v(k)
       kvec(k,ib) = cut1%k(k)
      enddo
     enddo
    else
     print*,'In subroutine geta '
     print*,'dmr=',dmr,' not allowed'
     stop
    endif
   endif
   contains
!
   function fnum(j)
   use inout
   use scale
   include 'cts_dpr.h' 
    :: p
   include 'cts_dpc.h' 
    :: fnum,allden
   integer, intent(in) :: j
   integer :: k
   allden= c1(p) 
   do k= 2,np
    allden= allden*vden(den(bbn1(k,ib))%i,j)
   enddo
   mprec=.false.
   dpq= cut1%q(:,j)
! comment
   if (ext_num_for_r1) then
    call numfunc(numdummy)
   else 
    if (rational) then
     call numfuncrec(np,cut1%q(:,j),j,dpres) 
    else
     call numfunc(numdummy)
    endif
   endif
! comment
   fnum =  (dpres                                  &
           -numd(number_propagators,cut1%q(:,j),j) &
           -numc(number_propagators,cut1%q(:,j),j) &
           -numb(number_propagators,cut1%q(:,j),j))/allden
   end function fnum
  end subroutine dp_geta_oldbase 
!
  subroutine dp_geta_newbase(p,numdummy,number_propagators,dmr)
   external numdummy
   include 'cts_dpr.h'
    , intent(in) :: p
   integer, intent(in) :: number_propagators
   integer, intent(in) :: dmr
   type(solcut1) :: cut1 
   integer :: i,ib,k
   integer :: np
   include 'cts_dpc.h' 
    :: gm,cf0,r1,r2,rden,f1,f2,f3,f4,f5
   acoeff= c0(p)
   np= number_propagators
   if (dmr.ge.2) return
   if     (np.lt.1) then
    print*,'In subroutine geta '
    print*,'number_propagators=', number_propagators,' not allowed'
    stop
   else   
    if     (dmr.eq.1) then
     do ib= 1,dmns_1
      call cut(den(bbn1(1,ib)),cut1,dmr) 
      do k= 0,3 
       p0veca(k,ib)= den(bbn1(1,ib))%p(k)
       l7vec(k,ib)= cut1%l3(k)
       l8vec(k,ib)= cut1%l4(k)
       vveca(k,ib)= cut1%v(k)
       kvec(k,ib) = cut1%k(k)
      enddo
      acoeff(0,ib)= fnum(1)
     enddo
    elseif ((dmr.eq.0).or.(dmr.eq.-1)) then
     do ib= 1,dmns_1
      call cut(den(bbn1(1,ib)),cut1,dmr) 
      do k= 0,3 
       p0veca(k,ib)= den(bbn1(1,ib))%p(k)
       l7vec(k,ib)= cut1%l3(k)
       l8vec(k,ib)= cut1%l4(k)
       vveca(k,ib)= cut1%v(k)
       kvec(k,ib) = cut1%k(k)
      enddo
      if (dmr.eq.-1) then
        a_rat1(ib) = cut1%rat1
      else
        a_rat1(ib) = 0.d0
      endif
      f1= fnum(1)
      f2= fnum(2)
      acoeff(0,ib)= 0.5d0*(f1+f2)
      f3= fnum(3)
      f4= fnum(4)
      f5= fnum(5)
      acoeff(1,ib)= 0.5d0*(f5+f4+f3-f2-2.d0*f1)/ci(p)/cut1%apar !0
      acoeff(2,ib)= 0.5d0*(f3-f1)/ci(p)/cut1%root               !x
      acoeff(3,ib)= 0.5d0*(f4-f1)/ci(p)/cut1%root               !y
      acoeff(4,ib)= 0.5d0*(f5-f1)/ci(p)/cut1%root               !z
     enddo
    else
     print*,'In subroutine geta '
     print*,'dmr=',dmr,' not allowed'
     stop
    endif
   endif
   contains
!
   function fnum(j)
   use inout
   use scale
   include 'cts_dpr.h' 
    :: p
   include 'cts_dpc.h' 
    :: fnum,allden
   integer, intent(in) :: j
   integer :: k
   allden= c1(p) 
   do k= 2,np
    allden= allden*vden(den(bbn1(k,ib))%i,j)
   enddo
   mprec=.false.
   dpq= cut1%q(:,j)
! comment
   if (ext_num_for_r1) then
    call numfunc(numdummy)
   else 
    if (rational) then
     call numfuncrec(np,cut1%q(:,j),j,dpres) 
    else
     call numfunc(numdummy)
    endif
   endif
! comment
   fnum =  (dpres                                  &
           -numd(number_propagators,cut1%q(:,j),j) &
           -numc(number_propagators,cut1%q(:,j),j) &
           -numb(number_propagators,cut1%q(:,j),j))/allden
   end function fnum
  end subroutine dp_geta_newbase 
!
  function dp_numd(number_propagators,q,j)
   include 'cts_dpr.h' 
    :: p
   integer, intent(in) :: number_propagators
   include 'cts_dpc.h'
    , intent(in), dimension(0:3) :: q
   integer, intent(in) :: j
   integer :: jj,i,ib,k,np
   include 'cts_dpc.h'
    , dimension(0:3) :: qp0
   include 'cts_dpc.h' 
    :: dp_numd,start,tqp0  
   include 'cts_dpc.h' 
    :: allden
   jj= abs(j)
   start= c0(p)
   np= number_propagators
   if     (np.lt.4) then
   else   
    do ib= 1,dmns_4
     allden= c1(p)
     if     (j.eq.0) then
      do k= 5,np
       allden= allden*value(den(bbn4(k,ib)),q) 
      enddo
     elseif (j.lt.0) then
      do k= 5,np
       allden= allden*(vden(den(bbn4(k,ib))%i,jj)-qt2) 
      enddo
     else
      do k= 5,np
       allden= allden*vden(den(bbn4(k,ib))%i,j)
       if (allden.eq.c0(p)) cycle
      enddo
     endif
     if (allden.eq.c0(p)) cycle
     do k= 0,3
      qp0(k)= q(k)+p0vecd(k,ib) 
     enddo
     call contr(tvec(:,ib),qp0,tqp0)
     if (j.lt.0) then
      start= start+(save_dcoeff(0,ib)+save_dcoeff(1,ib)*tqp0)&
            *allden
     else
      start= start+(dcoeff(0,ib)+dcoeff(1,ib)*tqp0)&
            *allden
     endif
    enddo  
   endif
   dp_numd= start
  end function dp_numd
!
  function dp_numc(number_propagators,q,j)
   include 'cts_dpr.h' 
    :: p
   integer, intent(in) :: number_propagators
   include 'cts_dpc.h'
    , intent(in), dimension(0:3) :: q
   integer, intent(in) :: j
   integer :: jj,i,ib,k,np
   include 'cts_dpc.h'
    , dimension(0:3) :: qp0
   include 'cts_dpc.h' 
    :: l3qp0,l4qp0
   include 'cts_dpc.h' 
    :: dp_numc,start
   include 'cts_dpc.h' 
    :: allden
   jj= abs(j)
   start= c0(p)
   np= number_propagators
   if     (np.lt.3) then
   else
    do ib= 1,dmns_3
     allden= c1(p)
     if     (j.eq.0) then
      do k= 4,np
       allden= allden*value(den(bbn3(k,ib)),q)
      enddo
     elseif (j.lt.0) then
      do k= 4,np
       allden= allden*(vden(den(bbn3(k,ib))%i,jj)-qt2)
      enddo
     else
      do k= 4,np
       allden= allden*vden(den(bbn3(k,ib))%i,j)
       if (allden.eq.c0(p)) cycle
      enddo
     endif
     if (allden.eq.c0(p)) cycle
     do k= 0,3 
      qp0(k)= q(k)+p0vecc(k,ib) 
     enddo
     call contr(l3vec(:,ib),qp0,l3qp0)
     call contr(l4vec(:,ib),qp0,l4qp0)
     if (j.lt.0) then
      start= start+(save_ccoeff(0,ib)               &
                   +save_ccoeff(1,ib)*l3qp0         &
                   +save_ccoeff(2,ib)*l4qp0         &
                   +save_ccoeff(3,ib)*l3qp0**2      &
                   +save_ccoeff(4,ib)*l4qp0**2      &
                   +save_ccoeff(5,ib)*l3qp0**3      &
                   +save_ccoeff(6,ib)*l4qp0**3)     &
                   *allden 
     else
      start= start+(ccoeff(0,ib)               &
                   +ccoeff(1,ib)*l3qp0         &
                   +ccoeff(2,ib)*l4qp0         &
                   +ccoeff(3,ib)*l3qp0**2      &
                   +ccoeff(4,ib)*l4qp0**2      &
                   +ccoeff(5,ib)*l3qp0**3      &
                   +ccoeff(6,ib)*l4qp0**3)     &
                   *allden 
     endif
    enddo  
   endif
   dp_numc= start
  end function dp_numc
!
  function dp_numb(number_propagators,q,j)
   include 'cts_dpr.h' 
    :: p
   integer, intent(in) :: number_propagators
   include 'cts_dpc.h'
    , intent(in), dimension(0:3) :: q
   integer, intent(in) :: j
   integer :: jj,i,ib,k,np
   include 'cts_dpc.h'
    , dimension(0:3) :: qp0
   include 'cts_dpc.h' 
    :: l3qp0,l4qp0,vqp0
   include 'cts_dpc.h' 
    :: dp_numb,start
   include 'cts_dpc.h' 
    :: allden
   jj= abs(j)
   start= c0(p)
   np= number_propagators
   if     (np.lt.2) then
   else    
    do ib= 1,dmns_2
     allden= c1(p)
     if     (j.eq.0) then
      do k= 3,np
       allden= allden*value(den(bbn2(k,ib)),q)
      enddo
     elseif (j.lt.0) then
      do k= 3,np
       allden= allden*(vden(den(bbn2(k,ib))%i,jj)-qt2)
      enddo
     else
      do k= 3,np
       allden= allden*vden(den(bbn2(k,ib))%i,j)
       if (allden.eq.c0(p)) cycle
      enddo
     endif
     if (allden.eq.c0(p)) cycle
     do k= 0,3
      qp0(k)= q(k)+p0vecb(k,ib) 
     enddo
     call contr(l5vec(:,ib),qp0,l3qp0)
     call contr(l6vec(:,ib),qp0,l4qp0)
     call contr(vvecb(:,ib),qp0,vqp0)
     if (j.lt.0) then
      start= start+(                             &
               save_bcoeff(0,ib)                 &
              +save_bcoeff(1,ib)*l3qp0           &
              +save_bcoeff(2,ib)*l4qp0           &
              +save_bcoeff(3,ib)*vqp0            &
              +save_bcoeff(4,ib)*(l3qp0)**2      &
              +save_bcoeff(5,ib)*(l4qp0)**2      &
              +save_bcoeff(6,ib)*vqp0**2         &
              +save_bcoeff(7,ib)*vqp0*l3qp0      &
              +save_bcoeff(8,ib)*vqp0*l4qp0)     &
              *allden
     else
      start= start+(                        &
               bcoeff(0,ib)                 &
              +bcoeff(1,ib)*l3qp0           &
              +bcoeff(2,ib)*l4qp0           &
              +bcoeff(3,ib)*vqp0            &
              +bcoeff(4,ib)*(l3qp0)**2      &
              +bcoeff(5,ib)*(l4qp0)**2      &
              +bcoeff(6,ib)*vqp0**2         &
              +bcoeff(7,ib)*vqp0*l3qp0      &
              +bcoeff(8,ib)*vqp0*l4qp0)     &
              *allden
     endif 
    enddo  
   endif
   dp_numb= start
  end function dp_numb
!
  function dp_numa(number_propagators,q,j)
   include 'cts_dpr.h' 
    :: p
   integer, intent(in) :: number_propagators
   include 'cts_dpc.h'
    , intent(in), dimension(0:3) :: q
   integer, intent(in) :: j
   integer :: jj,i,ib,k,np
   include 'cts_dpc.h'
    , dimension(0:3) :: qp0
   include 'cts_dpc.h' 
    :: l3qp0,l4qp0,vqp0,kqp0
   include 'cts_dpc.h' 
    :: dp_numa,start
   include 'cts_dpc.h' 
    :: allden
   jj= abs(j)
   start= c0(p)
   np= number_propagators
   if     (np.lt.1) then
    print*,'In function numa '
    print*,'number_propagators=', number_propagators,' not allowed'
    stop
   else 
    do ib= 1,dmns_1
     allden= c1(p)
     if     (j.eq.0) then
      do k= 2,np
       allden= allden*value(den(bbn1(k,ib)),q)
      enddo
     elseif (j.lt.0) then
      do k= 2,np
       allden= allden*(vden(den(bbn1(k,ib))%i,jj)-qt2)
      enddo
     else
      do k= 2,np
       allden= allden*vden(den(bbn1(k,ib))%i,j)
       if (allden.eq.c0(p)) cycle
      enddo
     endif  
     if (allden.eq.c0(p)) cycle
     do k= 0,3
      qp0(k)= q(k)+p0veca(k,ib) 
     enddo
     call contr(l7vec(:,ib),qp0,l3qp0)
     call contr(l8vec(:,ib),qp0,l4qp0)
     call contr(vveca(:,ib),qp0,vqp0)
     call contr(kvec(:,ib),qp0,kqp0)
     if (j.lt.0) then 
      start= start+(                    &
               save_acoeff(0,ib)        & 
              +save_acoeff(1,ib)*kqp0   &
              +save_acoeff(2,ib)*vqp0   &
              +save_acoeff(3,ib)*l3qp0  &
              +save_acoeff(4,ib)*l4qp0) &
              *allden
     else
      start= start+(               &
               acoeff(0,ib)        & 
              +acoeff(1,ib)*kqp0   &
              +acoeff(2,ib)*vqp0   &
              +acoeff(3,ib)*l3qp0  &
              +acoeff(4,ib)*l4qp0) &
              *allden
     endif
    enddo  
   endif
   dp_numa= start
  end function dp_numa
!
  subroutine dp_numfuncrec(number_propagators,q,j,dpres) 
   integer, intent(in) :: number_propagators,j
   include 'cts_dpc.h'
    , intent(in), dimension(0:3) :: q
   include 'cts_dpc.h'
    , intent(out) :: dpres
   include 'cts_dpc.h' 
    :: num_d, num_c, num_b, num_a
   num_d= numd(number_propagators,q,-j)
   num_c= numc(number_propagators,q,-j)
   num_b= numb(number_propagators,q,-j)
   num_a= numa(number_propagators,q,-j)
   dpres= num_d+num_c+num_b+num_a
  end subroutine dp_numfuncrec
!
  subroutine mp_getd(p,numdummy,number_propagators,dmr)
   external numdummy
   include 'cts_mpr.h'
    , intent(in) :: p
   integer, intent(in) :: dmr
   integer, intent(in) :: number_propagators
   type(mp_solcut4) :: cut4 
   integer :: i,ib,k
   integer :: np
   include 'cts_mpc.h'
    , dimension(0:3) :: qpp0
   include 'cts_mpc.h' 
    :: tqpp0
   include 'cts_mpc.h' 
    :: f1,f2
   mp_dcoeff= c0(p)
   np= number_propagators
   if     (np.lt.4) then
   else   
    do ib= 1,dmns_4
     call cut(mp_den(bbn4(1,ib)),mp_den(bbn4(2,ib)),mp_den(bbn4(3,ib)), &
              mp_den(bbn4(4,ib)),cut4) 
     do k= 0,3
      mp_tvec(k,ib) =  cut4%t(k)
      mp_p0vecd(k,ib)=  mp_den(bbn4(1,ib))%p(k)
      qpp0(k)= cut4%q(k,1)+mp_p0vecd(k,ib)
     enddo
     call contr(mp_tvec(:,ib),qpp0,tqpp0)
     f1= fnum(1) 
     f2= fnum(2) 
     mp_dcoeff(0,ib)= 0.5d0*(f1+f2)
     mp_dcoeff(1,ib)= 0.5d0*(f1-f2)/tqpp0
    enddo
   endif
   contains
!
   function fnum(j)
   use inout
   use scale
   include 'cts_mpr.h' 
    :: p
   include 'cts_mpc.h' 
    :: fnum,allden
   integer, intent(in) :: j
   integer :: k
   allden= c1(p) 
   do k= 5,np
    allden= allden*mp_vden(mp_den(bbn4(k,ib))%i,j)
   enddo
   mprec=.true.
   mpq= cut4%q(:,j)
! comment
   if (ext_num_for_r1) then
    call numfunc(numdummy)
   else 
    if (rational.and.(.not.inf)) then 
     call numfuncrec(np,cut4%q(:,j),j,mpres) 
    else
     call numfunc(numdummy)
    endif
   endif
! comment
   fnum= mpres/allden
   end function fnum
  end subroutine mp_getd 
!
  subroutine mp_getc(p,numdummy,number_propagators,dmr)
   external numdummy
   include 'cts_mpr.h'
    , intent(in) :: p
   integer, intent(in) :: number_propagators
   integer, intent(in) :: dmr
   type(mp_solcut3) :: cut3 
   integer :: i,ib,k,m 
   integer :: np
   include 'cts_mpc.h' 
    :: gm,cc,tau,cph,f1,f2,f3,f4,&
      f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16
   include 'cts_mpc.h' 
    :: sum,fv(0:3),gv(0:3),r(0:7),ccpar(-3:3)
   include 'cts_mpc.h' 
    :: cdelta,rp(0:7),fvp(0:3),cden1,cden2,c34
   include 'cts_mpc.h' 
    :: ca1,ca2,ca3,ca4,ca5
   if (.not.rational) mp_ccoeff= c0(p)
   if (rational)     call get_ccoeff(p)
   np= number_propagators
   if (dmr.ge.4) return
   if     (np.lt.3) then
   else   
    if     (dmr.eq.3) then
     do ib= 1,dmns_3
      call cut(mp_den(bbn3(1,ib)),mp_den(bbn3(2,ib)),mp_den(bbn3(3,ib)),cut3,dmr) 
      mp_ccoeff(0,ib)= fnum(1)
      mp_c4_rat1(ib) = 0.d0
      do k= 0,3
       mp_l3vec(k,ib) =  cut3%l3(k)
       mp_l4vec(k,ib) =  cut3%l4(k)
       mp_p0vecc(k,ib) =  mp_den(bbn3(1,ib))%p(k) 
      enddo
     enddo
    elseif (dmr.eq.2) then
     do ib= 1,dmns_3
      call cut(mp_den(bbn3(1,ib)),mp_den(bbn3(2,ib)),mp_den(bbn3(3,ib)),cut3,dmr) 
      do k= 0,3
       mp_l3vec(k,ib) =  cut3%l3(k)
       mp_l4vec(k,ib) =  cut3%l4(k)
       mp_p0vecc(k,ib) =  den(bbn3(1,ib))%p(k) 
      enddo
      cc = cut3%cc
      tau= cut3%tau
      c34= -2.d0*cut3%gm
      cden1= cc*tau
      cden2= c1(p)/tau**4-cc**4*tau**4
      cdelta= cden2*c34**4
      r(0) = fnum(1)
      r(4) = fnum(2)
      rp(0)= fnum(3) 
      ca1= 0.5d0*(r(0)-r(4))
      ca2= 0.5d0*(r(0)+r(4))
      ca3= rp(0)-ca2      
      mp_ccoeff(0,ib)= ca2
      mp_ccoeff(1,ib)= c34**3*(ca1/tau-cden1*ca3)*(c1(p)/tau**2+cden1**2)/cdelta
      mp_ccoeff(2,ib)= c34**3*(ca3/tau-cden1*ca1)*(c1(p)/tau**2+cden1**2)/cdelta
      mp_c4_rat1(ib) = 0.d0
     enddo
    elseif (dmr.eq.1) then
     do ib= 1,dmns_3
      call cut(mp_den(bbn3(1,ib)),mp_den(bbn3(2,ib)),mp_den(bbn3(3,ib)),cut3,dmr) 
      if (rational) then
       r(0) = fnum(1)-fnuminv(1,1)
       r(4) = fnum(2)-fnuminv(2,1)
       mp_ccoeff(0,ib)= 0.5d0*(r(0)+r(4))    
       mp_c4_rat1(ib) = 0.d0
       cycle
      else
       do k= 0,3
        mp_l3vec(k,ib) =  cut3%l3(k)
        mp_l4vec(k,ib) =  cut3%l4(k)
        mp_p0vecc(k,ib) =  mp_den(bbn3(1,ib))%p(k) 
       enddo
      endif 
      cc = cut3%cc
      tau= cut3%tau
      c34= -2.d0*cut3%gm
      cden1= cc*tau
      cden2= c1(p)/tau**4-cc**4*tau**4
      cdelta= cden2*c34**4
      r(0) = fnum(1)
      r(4) = fnum(2)
      r(6) = fnum(3)
      rp(0)= fnum(4) 
      rp(4)= fnum(5) 
      ca1= 0.5d0*(r(0)-r(4))
      ca2= 0.5d0*(r(0)+r(4))
      ca3= 0.5d0*(rp(0)-rp(4)) 
      ca4= 0.5d0*(rp(0)+rp(4)) 
      mp_ccoeff(1,ib)= c34**3*(ca1/tau-cden1*ca3)*(c1(p)/tau**2+cden1**2)/cdelta
      mp_ccoeff(2,ib)= c34**3*(ca3/tau-cden1*ca1)*(c1(p)/tau**2+cden1**2)/cdelta
      ca5= r(6)+ci(p)*c34*(mp_ccoeff(1,ib)/tau-mp_ccoeff(2,ib)*cden1)
      mp_ccoeff(0,ib)= 0.5d0*(ca2+ca5)
      ca1= ca2-mp_ccoeff(0,ib)
      ca2= ca4-mp_ccoeff(0,ib)
      mp_ccoeff(3,ib)= c34**2/cdelta*(ca1/tau**2-cden1**2*ca2)
      mp_ccoeff(4,ib)= c34**2/cdelta*(ca2/tau**2-cden1**2*ca1)
      mp_c4_rat1(ib) = 0.d0
     enddo
    elseif (dmr.eq.0) then
     do ib= 1,dmns_3
      call cut(mp_den(bbn3(1,ib)),mp_den(bbn3(2,ib)),mp_den(bbn3(3,ib)),cut3,dmr) 
      cc = cut3%cc
      tau= cut3%tau
      c34= -2.d0*cut3%gm
      cden1= cc*tau
      cden2= c1(p)/tau**4-cc**4*tau**4
      cdelta= cden2*c34**4
      if (rational) then
       r(0) = fnum(1)-fnuminv(1,2)
       r(4) = fnum(2)-fnuminv(2,2)
       rp(0)= fnum(3)-fnuminv(3,2)
       ca1= 0.5d0*(r(0)-r(4))
       ca2= 0.5d0*(r(0)+r(4))
       ca3= rp(0)-ca2      
       mp_ccoeff(0,ib)= ca2
       mp_ccoeff(1,ib)= c34**3*(ca1/tau-cden1*ca3)*(c1(p)/tau**2+cden1**2)/cdelta
       mp_ccoeff(2,ib)= c34**3*(ca3/tau-cden1*ca1)*(c1(p)/tau**2+cden1**2)/cdelta
       mp_c4_rat1(ib) = 0.d0
       cycle
      else
       do k= 0,3
        mp_l3vec(k,ib) =  cut3%l3(k)
        mp_l4vec(k,ib) =  cut3%l4(k)
        mp_p0vecc(k,ib) =  mp_den(bbn3(1,ib))%p(k) 
       enddo
      endif
      r(0) = fnum(1)
      r(2) = fnum(2)
      r(4) = fnum(3)
      r(6) = fnum(4)
      rp(0)= fnum(5) 
      rp(2)= fnum(6) 
      rp(4)= fnum(7) 
      do k= 0,3
       sum= 0.d0
       do m= 0,3
         sum= sum+r(2*m)*(cexp2(p))**(-m*k) 
       enddo
       fv(k)= sum/4.d0 
      enddo
      rp(6)= 4.d0*fv(0)-rp(0)-rp(2)-rp(4) 
      do k= 0,3
       sum= 0.d0
       do m= 0,3
         sum= sum+rp(2*m)*(cexp2(p))**(-m*k) 
       enddo
       fvp(k)= sum/4.d0 
      enddo
      mp_ccoeff(0,ib)= fv(0)
      mp_ccoeff(3,ib)= (fv(2)/tau**2-fvp(2)*cden1**2)*c34**2/cdelta
      mp_ccoeff(4,ib)= (fvp(2)/tau**2-fv(2)*cden1**2)*c34**2/cdelta
      mp_ccoeff(1,ib)= (fv(1)/tau**3-fvp(3)*cden1**3)*c34**3/cdelta
      mp_ccoeff(6,ib)= (fvp(3)/tau**1-fv(1)*cden1**1)*c34**1/cdelta
      mp_ccoeff(5,ib)= (fv(3)/tau**1-fvp(1)*cden1**1)*c34**1/cdelta
      mp_ccoeff(2,ib)= (fvp(1)/tau**3-fv(3)*cden1**3)*c34**3/cdelta
      mp_c4_rat1(ib) = 0.d0
     enddo
    elseif (dmr.eq.-1) then
! comment: later on c_5 and c_6 not to be computed with rational=.true. 
     do ib= 1,dmns_3
      call cut(mp_den(bbn3(1,ib)),mp_den(bbn3(2,ib)),mp_den(bbn3(3,ib)),cut3,dmr) 
      do k= 0,3
       mp_l3vec(k,ib) =  cut3%l3(k)
       mp_l4vec(k,ib) =  cut3%l4(k)
       mp_p0vecc(k,ib) =  mp_den(bbn3(1,ib))%p(k) 
      enddo
      cc = cut3%cc
      tau= cut3%tau
      c34= -2.d0*cut3%gm
      cden1= cc*tau
      cden2= c1(p)/tau**4-cc**4*tau**4
      cdelta= cden2*c34**4
      r(0) = fnum(1)
      r(2) = fnum(2)
      r(4) = fnum(3)
      r(6) = fnum(4)
      rp(0)= fnum(5) 
      rp(2)= fnum(6) 
      rp(4)= fnum(7) 
      do k= 0,3
       sum= 0.d0
       do m= 0,3
         sum= sum+r(2*m)*(cexp2(p))**(-m*k) 
       enddo
       fv(k)= sum/4.d0 
      enddo
      rp(6)= 4.d0*fv(0)-rp(0)-rp(2)-rp(4) 
      do k= 0,3
       sum= 0.d0
       do m= 0,3
         sum= sum+rp(2*m)*(cexp2(p))**(-m*k) 
       enddo
       fvp(k)= sum/4.d0 
      enddo
      mp_ccoeff(0,ib)= fv(0)
      mp_ccoeff(3,ib)= (fv(2)/tau**2-fvp(2)*cden1**2)*c34**2/cdelta
      mp_ccoeff(4,ib)= (fvp(2)/tau**2-fv(2)*cden1**2)*c34**2/cdelta
      mp_ccoeff(1,ib)= (fv(1)/tau**3-fvp(3)*cden1**3)*c34**3/cdelta
      mp_ccoeff(6,ib)= (fvp(3)/tau**1-fv(1)*cden1**1)*c34**1/cdelta
      mp_ccoeff(5,ib)= (fv(3)/tau**1-fvp(1)*cden1**1)*c34**1/cdelta
      mp_ccoeff(2,ib)= (fvp(1)/tau**3-fv(3)*cden1**3)*c34**3/cdelta
      mp_c4_rat1(ib) = cut3%rat1
     enddo
    else
     print*,'In subroutine getc '
     print*,'dmr=',dmr,' not allowed'
     stop
    endif
   endif
   contains
!
   function fnum(j)
   use inout
   use scale
   include 'cts_mpr.h' 
    :: p
   include 'cts_mpc.h' 
    :: fnum,allden
   integer, intent(in) :: j
   integer :: k
   allden= c1(p) 
   do k= 4,np
    allden= allden*mp_vden(mp_den(bbn3(k,ib))%i,j)
   enddo
   mprec=.true.
   mpq= cut3%q(:,j)
! comment
   if (ext_num_for_r1) then
    call numfunc(numdummy)
   else 
    if (rational) then
     call numfuncrec(np,cut3%q(:,j),j,mpres) 
    else
     call numfunc(numdummy)
    endif
   endif
! comment
   fnum=  (mpres-numd(np,cut3%q(:,j),j))/allden
   end function fnum
! 
   function fnuminv(j,level) 
   include 'cts_mpr.h' 
    :: p
   include 'cts_mpc.h' 
    :: sum,fnuminv,l3qp0,l4qp0
   include 'cts_mpc.h'
    , dimension(0:3) :: qp0
   integer, intent(in) :: j,level
   integer :: k
   do k= 0,3
    qp0(k)= cut3%q(k,j)+mp_p0vecc(k,ib) 
   enddo
   call contr(qp0,mp_l3vec(:,ib),l3qp0)
   call contr(qp0,mp_l4vec(:,ib),l4qp0)
   sum= c0(p)
   if (level.le.1) sum= sum+mp_ccoeff(1,ib)*l3qp0   +mp_ccoeff(2,ib)*l4qp0    
   if (level.le.2) sum= sum+mp_ccoeff(3,ib)*l3qp0**2+mp_ccoeff(4,ib)*l4qp0**2 
   if (level.le.3) sum= sum+mp_ccoeff(5,ib)*l3qp0**3+mp_ccoeff(6,ib)*l4qp0**3 
   fnuminv= sum
   end function fnuminv
  end subroutine mp_getc 
!
  subroutine mp_getb(p,numdummy,number_propagators,dmr)
   external numdummy
   include 'cts_mpr.h'
    , intent(in) :: p
   integer, intent(in) :: number_propagators
   integer, intent(in) :: dmr
   type(mp_solcut2) :: cut2 
   integer :: i,ib,k
   integer :: np
   include 'cts_mpc.h' 
    :: gm,ulambda,usigma,zlambda,zsigma
   include 'cts_mpc.h' 
    :: rden,caus1,caus2,caus3,cflambda,cfsigma
   include 'cts_mpc.h' 
    :: f1,f2,f3,f4,f5,f6,f7,f8,f9,f10
   include 'cts_mpc.h' 
    :: f11,f12,f13,f14,f15,f16,f17,f18
   include 'cts_mpc.h' 
    :: r(0:4),rp(0:2),fv(0:2),fvp(0:2)
   include 'cts_mpc.h' 
    :: rr0,rr2,tau,taul,cf0,cdelta,sf6,sf9
   if (.not.rational) mp_bcoeff= c0(p)
   if (rational)     call get_bcoeff(p)
   np= number_propagators  
   if (dmr.ge.3) return
   if     (np.lt.2) then
   else
    if     (dmr.eq.2) then
     do ib= 1,dmns_2
      call cut(mp_den(bbn2(1,ib)),mp_den(bbn2(2,ib)),cut2,dmr) 
      do k= 0,3 
       mp_p0vecb(k,ib)=  mp_den(bbn2(1,ib))%p(k) 
       mp_l5vec(k,ib)=  cut2%l3(k)
       mp_l6vec(k,ib)=  cut2%l4(k)
       mp_vvecb(k,ib)=  cut2%v(k)
      enddo
      gm= cut2%gm
      mp_bcoeff(0,ib)=  fnum(1)
      mp_vveck1(ib) =  gm/2.d0
      mp_b_rat1(ib) = cut2%rat1
      mp_b3_rat1(ib)= 0.d0
     enddo
    elseif (dmr.eq.1) then
     do ib= 1,dmns_2
      call cut(mp_den(bbn2(1,ib)),mp_den(bbn2(2,ib)),cut2,dmr) 
      do k= 0,3 
       mp_p0vecb(k,ib)=  mp_den(bbn2(1,ib))%p(k) 
       mp_l5vec(k,ib)=  cut2%l3(k)
       mp_l6vec(k,ib)=  cut2%l4(k)
       mp_vvecb(k,ib)=  cut2%v(k)
      enddo 
      tau= cut2%tau 
      taul= cut2%taul
      gm = cut2%gm
      cf0= cut2%cf0 
      cdelta= c1(p)-tau**4*cf0**2
      r(0) = fnum(1)
      r(3) = fnum(2)
      rp(0)= fnum(3)
      mp_bcoeff(0,ib)= 0.5d0*(r(0)+r(3))
      fv(0) = 0.5d0*(r(0)-r(3))
      fvp(0)= rp(0)-mp_bcoeff(0,ib)
      mp_bcoeff(1,ib)= -0.5d0*tau/gm/cdelta*(fv(0)-tau**2*cf0*fvp(0))
      mp_bcoeff(2,ib)= -0.5d0*tau/gm/cdelta*(fvp(0)-tau**2*cf0*fv(0))
      mp_bcoeff(4,ib)= 0.d0
      mp_bcoeff(5,ib)= 0.d0
      mp_bcoeff(3,ib)= 2.d0/gm/lambda(p)*sfun(4)
      mp_vveck1(ib) =  gm/2.d0
      mp_b_rat1(ib) = cut2%rat1
      mp_b3_rat1(ib)= 0.d0
     enddo
    elseif (dmr.eq.0) then
     do ib= 1,dmns_2
      call cut(mp_den(bbn2(1,ib)),mp_den(bbn2(2,ib)),cut2,dmr) 
      if (rational) then
       mp_bcoeff(0,ib)= 0.d0
       mp_bcoeff(0,ib)= sfun(1)
       cycle
      else
       do k= 0,3 
        mp_p0vecb(k,ib)=  mp_den(bbn2(1,ib))%p(k) 
        mp_l5vec(k,ib)=  cut2%l3(k)
        mp_l6vec(k,ib)=  cut2%l4(k)
        mp_vvecb(k,ib)=  cut2%v(k)
       enddo
      endif 
      tau= cut2%tau
      taul= cut2%taul
      gm = cut2%gm
      cf0= cut2%cf0 
      cdelta= c1(p)-tau**6*cf0**3
      r(0) = fnum(1)
      r(2) = fnum(2)
      r(4) = fnum(3)
      rp(0)= fnum(4)
      rp(2)= fnum(5)
      fv(0)= (r(0)+r(2)+r(4))/3.d0
      fv(1)= (r(0)+r(2)/cexp3(p)**2+r(4)/cexp3(p)**4)/3.d0
      fv(2)= (r(0)+r(2)/cexp3(p)**4+r(4)/cexp3(p)**8)/3.d0
      rr0= rp(0)-fv(0)
      rr2= rp(2)-fv(0)
      fvp(2)= (rr2-rr0*cexp3(p)**2)/(cexp3(p)**4-cexp3(p)**2)
      fvp(1)= rr0-fvp(2)
      mp_bcoeff(0,ib)= fv(0) 
      mp_bcoeff(1,ib)=   -tau/2.d0/gm*(fv(1)-tau**4*cf0**2*fvp(2))/cdelta
      mp_bcoeff(2,ib)=   -tau/2.d0/gm*(fvp(1)-tau**4*cf0**2*fv(2))/cdelta
      mp_bcoeff(4,ib)= tau**2/4.d0/gm/gm*(fv(2)-tau**2*cf0*fvp(1))/cdelta
      mp_bcoeff(5,ib)= tau**2/4.d0/gm/gm*(fvp(2)-tau**2*cf0*fv(1))/cdelta
      cflambda= cut2%cflambda
      cfsigma = cut2%cfsigma
      r(0) = sfun(6)
      r(2) = sfun(7)
      fv(0)= (r(0)+r(2))/2.d0
      fv(1)= (r(0)-r(2))/2.d0
      rp(0)= sfun(8)-fv(0)
      fvp(1)= rp(0)
      cdelta= c1(p)-taul**4*cflambda**2
      mp_bcoeff(8,ib)=-taul/gm/gm/lambda(p)*(fvp(1)-fv(1)*cflambda*taul**2)/cdelta
      mp_bcoeff(7,ib)=-taul/gm/gm/lambda(p)*(fv(1)-fvp(1)*cflambda*taul**2)/cdelta
      sf6= r(0)
      sf9= sfun(9)
      ulambda= ufun(6,sf6)
      usigma = ufun(9,sf9)
      cdelta= lambda(p)*sigma(p)*(sigma(p)-lambda(p))
      mp_bcoeff(3,ib)= 2.d0/gm/cdelta*(ulambda*sigma(p)**2-usigma*lambda(p)**2)
      mp_bcoeff(6,ib)= 4.d0/gm/gm/cdelta*(usigma*lambda(p)-ulambda*sigma(p))
      mp_vveck1(ib) = gm/2.d0
      mp_b_rat1(ib) = cut2%rat1
      mp_b3_rat1(ib)= 0.d0
     enddo
    elseif (dmr.eq.-1) then
! comment: later on b_4-b_8 not to be computed with rational=.true. 
     do ib= 1,dmns_2
      call cut(mp_den(bbn2(1,ib)),mp_den(bbn2(2,ib)),cut2,dmr) 
      do k= 0,3 
       mp_p0vecb(k,ib)=  mp_den(bbn2(1,ib))%p(k) 
       mp_l5vec(k,ib)=  cut2%l3(k)
       mp_l6vec(k,ib)=  cut2%l4(k)
       mp_vvecb(k,ib)=  cut2%v(k)
      enddo
      tau= cut2%tau
      taul= cut2%taul
      gm = cut2%gm
      cf0= cut2%cf0 
      cdelta= c1(p)-tau**6*cf0**3
      r(0) = fnum(1)
      r(2) = fnum(2)
      r(4) = fnum(3)
      rp(0)= fnum(4)
      rp(2)= fnum(5)
      fv(0)= (r(0)+r(2)+r(4))/3.d0
      fv(1)= (r(0)+r(2)/cexp3(p)**2+r(4)/cexp3(p)**4)/3.d0
      fv(2)= (r(0)+r(2)/cexp3(p)**4+r(4)/cexp3(p)**8)/3.d0
      rr0= rp(0)-fv(0)
      rr2= rp(2)-fv(0)
      fvp(2)= (rr2-rr0*cexp3(p)**2)/(cexp3(p)**4-cexp3(p)**2)
      fvp(1)= rr0-fvp(2)
      mp_bcoeff(0,ib)= fv(0) 
      mp_bcoeff(1,ib)=   -tau/2.d0/gm*(fv(1)-tau**4*cf0**2*fvp(2))/cdelta
      mp_bcoeff(2,ib)=   -tau/2.d0/gm*(fvp(1)-tau**4*cf0**2*fv(2))/cdelta
      mp_bcoeff(4,ib)= tau**2/4.d0/gm/gm*(fv(2)-tau**2*cf0*fvp(1))/cdelta
      mp_bcoeff(5,ib)= tau**2/4.d0/gm/gm*(fvp(2)-tau**2*cf0*fv(1))/cdelta
      cflambda= cut2%cflambda
      cfsigma = cut2%cfsigma
      r(0) = sfun(6)
      r(2) = sfun(7)
      fv(0)= (r(0)+r(2))/2.d0
      fv(1)= (r(0)-r(2))/2.d0
      rp(0)= sfun(8)-fv(0)
      fvp(1)= rp(0)
      cdelta= c1(p)-taul**4*cflambda**2
      mp_bcoeff(8,ib)=-taul/gm/gm/lambda(p)*(fvp(1)-fv(1)*cflambda*taul**2)/cdelta
      mp_bcoeff(7,ib)=-taul/gm/gm/lambda(p)*(fv(1)-fvp(1)*cflambda*taul**2)/cdelta
      sf6= r(0)
      sf9= sfun(9)
      ulambda= ufun(6,sf6)
      usigma = ufun(9,sf9)
      cdelta= lambda(p)*sigma(p)*(sigma(p)-lambda(p))
      mp_bcoeff(3,ib)= 2.d0/gm/cdelta*(ulambda*sigma(p)**2-usigma*lambda(p)**2)
      mp_bcoeff(6,ib)= 4.d0/gm/gm/cdelta*(usigma*lambda(p)-ulambda*sigma(p))
      mp_vveck1(ib) = gm/2.d0
      mp_b_rat1(ib) = cut2%rat1
      mp_b3_rat1(ib)= cut2%rat1t
     enddo
    else
     print*,'In subroutine getb '
     print*,'dmr=',dmr,' not allowed'
     stop
    endif
   endif
   contains
!
   function fnum(j)
   use inout
   use scale
   include 'cts_mpr.h' 
    :: p
   include 'cts_mpc.h' 
    :: fnum,allden
   integer, intent(in) :: j
   integer :: k
   allden= c1(p) 
   do k= 3,np
    allden= allden*mp_vden(mp_den(bbn2(k,ib))%i,j)
   enddo
   mprec=.true.
   mpq= cut2%q(:,j)
! comment
   if (ext_num_for_r1) then
    call numfunc(numdummy)
   else 
    if (rational) then
     call numfuncrec(np,cut2%q(:,j),j,mpres) 
    else
     call numfunc(numdummy)
    endif
   endif
! comment
   fnum =  (mpres                                  &
           -numd(number_propagators,cut2%q(:,j),j) &
           -numc(number_propagators,cut2%q(:,j),j))/allden
   end function fnum
!
   function sfun(j) ! function added
   integer, intent(in) :: j
   include 'cts_mpc.h' 
    :: sfun,l3qp0,l4qp0
   include 'cts_mpc.h'
    , dimension(0:3) :: qp0
   integer :: k
   do k= 0,3
    qp0(k)= cut2%q(k,j)+mp_p0vecb(k,ib)
   enddo
   call contr(qp0,mp_l5vec(:,ib),l3qp0)
   call contr(qp0,mp_l6vec(:,ib),l4qp0)
   sfun= fnum(j)-mp_bcoeff(0,ib)          &
                -mp_bcoeff(1,ib)*l3qp0    &
                -mp_bcoeff(2,ib)*l4qp0    &
                -mp_bcoeff(4,ib)*l3qp0**2 &
                -mp_bcoeff(5,ib)*l4qp0**2
   end function sfun
!
   function ufun(j,sf) ! function added
   integer, intent(in) :: j
   include 'cts_mpc.h'
    , intent(in) :: sf
   include 'cts_mpc.h' 
    :: ufun,l3qp0,l4qp0,vqp0
   include 'cts_mpc.h'
    , dimension(0:3) :: qp0
   integer :: k
   do k= 0,3
    qp0(k)= cut2%q(k,j)+mp_p0vecb(k,ib)
   enddo
   call contr(qp0,mp_l5vec(:,ib),l3qp0)
   call contr(qp0,mp_l6vec(:,ib),l4qp0)
   call contr(qp0,mp_vvecb(:,ib),vqp0)
   ufun= sf-mp_bcoeff(7,ib)*vqp0*l3qp0 &
           -mp_bcoeff(8,ib)*vqp0*l4qp0
   end function ufun
  end subroutine mp_getb 
!
  subroutine mp_geta_oldbase(p,numdummy,number_propagators,dmr)
   external numdummy
   include 'cts_mpr.h'
    , intent(in) :: p
   integer, intent(in) :: dmr
   integer, intent(in) :: number_propagators
   type(mp_solcut1) :: cut1 
   integer :: i,ib,k
   integer :: np
   include 'cts_mpc.h' 
    :: gm,cf0,r1,r2,rden,f1,f2,f3,f4,f5
   mp_acoeff= c0(p)
   np= number_propagators
   if (dmr.ge.2) return
   if     (np.lt.1) then
    print*,'In subroutine geta '
    print*,'number_propagators=', number_propagators,' not allowed'
    stop
   else 
    if     (dmr.eq.1) then
     do ib= 1,dmns_1
      call cut(mp_den(bbn1(1,ib)),cut1,dmr) 
      mp_acoeff(0,ib)= fnum(1)
      do k= 0,3 
       mp_p0veca(k,ib)= mp_den(bbn1(1,ib))%p(k)
       mp_l7vec(k,ib)= cut1%l3(k)
       mp_l8vec(k,ib)= cut1%l4(k)
       mp_vveca(k,ib)= cut1%v(k)
       mp_kvec(k,ib) = cut1%k(k)
      enddo
     enddo
    elseif (dmr.eq.0) then
     do ib= 1,dmns_1
      call cut(mp_den(bbn1(1,ib)),cut1,dmr) 
      f1= fnum(1)
      f2= fnum(2)
      f3= fnum(3)
      f4= fnum(4)
      f5= fnum(5)
      gm = cut1%gm
      cf0= cut1%cf0 
      mp_acoeff(0,ib)=   0.5d0*(f1+f2)
      mp_acoeff(1,ib)= 2.d0/gm*(f3-f1)
      mp_acoeff(2,ib)= 2.d0/gm*(f4-f1)
      r1= -0.5d0*(f1-mp_acoeff(0,ib))/gm
      r2= -0.5d0*(f5-mp_acoeff(0,ib))/gm
      rden= cexpk1(p)*tau12(p)-cf0**2/cexpk1(p)/tau12(p)
      mp_acoeff(3,ib)= (r1*cexpk1(p)*tau12(p)-cf0*r2)/rden*tau11(p)
      mp_acoeff(4,ib)= (r2-cf0*r1/cexpk1(p)/tau12(p))/rden/tau11(p)
      do k= 0,3 
       mp_p0veca(k,ib)= mp_den(bbn1(1,ib))%p(k)
       mp_l7vec(k,ib)= cut1%l3(k)
       mp_l8vec(k,ib)= cut1%l4(k)
       mp_vveca(k,ib)= cut1%v(k)
       mp_kvec(k,ib) = cut1%k(k)
      enddo
     enddo
    else
     print*,'In subroutine geta '
     print*,'dmr=',dmr,' not allowed'
     stop
    endif
   endif
   contains
!
   function fnum(j)
   use inout
   use scale
   include 'cts_mpr.h' 
    :: p
   include 'cts_mpc.h' 
    :: fnum,allden
   integer, intent(in) :: j
   integer :: k
   allden= c1(p) 
   do k= 2,np
    allden= allden*mp_vden(mp_den(bbn1(k,ib))%i,j)
   enddo
   mprec=.true.
   mpq= cut1%q(:,j)
! comment
   if (ext_num_for_r1) then
    call numfunc(numdummy)
   else 
    if (rational) then
     call numfuncrec(np,cut1%q(:,j),j,mpres) 
    else
     call numfunc(numdummy)
    endif
   endif
! comment
   fnum =  (mpres                                  &
           -numd(number_propagators,cut1%q(:,j),j) &
           -numc(number_propagators,cut1%q(:,j),j) &
           -numb(number_propagators,cut1%q(:,j),j))/allden
   end function fnum
  end subroutine mp_geta_oldbase 
!
  subroutine mp_geta_newbase(p,numdummy,number_propagators,dmr)
   external numdummy
   include 'cts_mpr.h'
    , intent(in) :: p
   integer, intent(in) :: number_propagators
   integer, intent(in) :: dmr
   type(mp_solcut1) :: cut1 
   integer :: i,ib,k
   integer :: np
   include 'cts_mpc.h' 
    :: gm,cf0,r1,r2,rden,f1,f2,f3,f4,f5
   mp_acoeff= c0(p)
   np= number_propagators
   if (dmr.ge.2) return
   if     (np.lt.1) then
    print*,'In subroutine geta '
    print*,'number_propagators=', number_propagators,' not allowed'
    stop
   else   
    if     (dmr.eq.1) then
     do ib= 1,dmns_1
      call cut(mp_den(bbn1(1,ib)),cut1,dmr) 
      do k= 0,3 
       mp_p0veca(k,ib)= mp_den(bbn1(1,ib))%p(k)
       mp_l7vec(k,ib)= cut1%l3(k)
       mp_l8vec(k,ib)= cut1%l4(k)
       mp_vveca(k,ib)= cut1%v(k)
       mp_kvec(k,ib) = cut1%k(k)
      enddo
      mp_acoeff(0,ib)= fnum(1)
     enddo
    elseif ((dmr.eq.0).or.(dmr.eq.-1)) then
     do ib= 1,dmns_1
      call cut(mp_den(bbn1(1,ib)),cut1,dmr) 
      do k= 0,3 
       mp_p0veca(k,ib)= mp_den(bbn1(1,ib))%p(k)
       mp_l7vec(k,ib)= cut1%l3(k)
       mp_l8vec(k,ib)= cut1%l4(k)
       mp_vveca(k,ib)= cut1%v(k)
       mp_kvec(k,ib) = cut1%k(k)
      enddo
      if (dmr.eq.-1) then
        mp_a_rat1(ib) = cut1%rat1
      else
        mp_a_rat1(ib) = 0.d0
      endif 
      f1= fnum(1)
      f2= fnum(2)
      mp_acoeff(0,ib)= 0.5d0*(f1+f2)
      f3= fnum(3)
      f4= fnum(4)
      f5= fnum(5)
      mp_acoeff(0,ib)= 0.5d0*(f1+f2)
      mp_acoeff(1,ib)= 0.5d0*(f5+f4+f3-f2-2.d0*f1)/ci(p)/cut1%apar !0
      mp_acoeff(2,ib)= 0.5d0*(f3-f1)/ci(p)/cut1%root               !x
      mp_acoeff(3,ib)= 0.5d0*(f4-f1)/ci(p)/cut1%root               !y
      mp_acoeff(4,ib)= 0.5d0*(f5-f1)/ci(p)/cut1%root               !z
     enddo
    else
     print*,'In subroutine geta '
     print*,'dmr=',dmr,' not allowed'
     stop
    endif
   endif
   contains
!
   function fnum(j)
   use inout
   use scale
   include 'cts_mpr.h' 
    :: p
   include 'cts_mpc.h' 
    :: fnum,allden
   integer, intent(in) :: j
   integer :: k
   allden= c1(p) 
   do k= 2,np
    allden= allden*mp_vden(mp_den(bbn1(k,ib))%i,j)
   enddo
   mprec=.true.
   mpq= cut1%q(:,j)
! comment
   if (ext_num_for_r1) then
    call numfunc(numdummy)
   else 
    if (rational) then
     call numfuncrec(np,cut1%q(:,j),j,mpres) 
    else
     call numfunc(numdummy)
    endif
   endif
! comment
   fnum =  (mpres                                  &
           -numd(number_propagators,cut1%q(:,j),j) &
           -numc(number_propagators,cut1%q(:,j),j) &
           -numb(number_propagators,cut1%q(:,j),j))/allden
   end function fnum
  end subroutine mp_geta_newbase 
!
  function mp_numd(number_propagators,q,j)
   include 'cts_mpr.h' 
    :: p
   integer, intent(in) :: number_propagators
   include 'cts_mpc.h'
    , intent(in), dimension(0:3) :: q
   integer, intent(in) :: j
   integer :: jj,i,ib,k,np
   include 'cts_mpc.h'
    , dimension(0:3) :: qp0
   include 'cts_mpc.h' 
    :: mp_numd,start,tqp0  
   include 'cts_mpc.h' 
    :: allden
   jj= abs(j)
   start= c0(p)
   np= number_propagators
   if     (np.lt.4) then
   else   
    do ib= 1,dmns_4
     allden= c1(p)
     if     (j.eq.0) then
      do k= 5,np
       allden= allden*value(mp_den(bbn4(k,ib)),q) 
      enddo
     elseif (j.lt.0) then
      do k= 5,np
       allden= allden*(mp_vden(mp_den(bbn4(k,ib))%i,jj)-mpqt2) 
      enddo
     else
      do k= 5,np
       allden= allden*mp_vden(mp_den(bbn4(k,ib))%i,j)
       if (allden.eq.c0(p)) cycle
      enddo
     endif
     if (allden.eq.c0(p)) cycle
     do k= 0,3
      qp0(k)= q(k)+mp_p0vecd(k,ib) 
     enddo
     call contr(mp_tvec(:,ib),qp0,tqp0)
     if (j.lt.0) then
      start= start+(save_mp_dcoeff(0,ib)+save_mp_dcoeff(1,ib)*tqp0)&
            *allden
     else
      start= start+(mp_dcoeff(0,ib)+mp_dcoeff(1,ib)*tqp0)&
            *allden
     endif
    enddo  
   endif
   mp_numd= start
  end function mp_numd
!
  function mp_numc(number_propagators,q,j)
   include 'cts_mpr.h' 
    :: p
   integer, intent(in) :: number_propagators
   include 'cts_mpc.h'
    , intent(in), dimension(0:3) :: q
   integer, intent(in) :: j
   integer :: jj,i,ib,k,np
   include 'cts_mpc.h'
    , dimension(0:3) :: qp0
   include 'cts_mpc.h' 
    :: l3qp0,l4qp0
   include 'cts_mpc.h' 
    :: mp_numc,start
   include 'cts_mpc.h' 
    :: allden
   jj= abs(j)
   start= c0(p)
   np= number_propagators
   if     (np.lt.3) then
   else   
    do ib= 1,dmns_3
     allden= c1(p)
     if     (j.eq.0) then
      do k= 4,np
       allden= allden*value(mp_den(bbn3(k,ib)),q)
      enddo
     elseif (j.lt.0) then
      do k= 4,np
       allden= allden*(mp_vden(mp_den(bbn3(k,ib))%i,jj)-mpqt2)
      enddo
     else
      do k= 4,np
       allden= allden*mp_vden(mp_den(bbn3(k,ib))%i,j)
       if (allden.eq.c0(p)) cycle
      enddo
     endif
     if (allden.eq.c0(p)) cycle
     do k= 0,3 
      qp0(k)= q(k)+mp_p0vecc(k,ib) 
     enddo
     call contr(mp_l3vec(:,ib),qp0,l3qp0)
     call contr(mp_l4vec(:,ib),qp0,l4qp0)
     if (j.lt.0) then
      start= start+(save_mp_ccoeff(0,ib)               &
                   +save_mp_ccoeff(1,ib)*l3qp0         &
                   +save_mp_ccoeff(2,ib)*l4qp0         &
                   +save_mp_ccoeff(3,ib)*l3qp0**2      &
                   +save_mp_ccoeff(4,ib)*l4qp0**2      &
                   +save_mp_ccoeff(5,ib)*l3qp0**3      &
                   +save_mp_ccoeff(6,ib)*l4qp0**3)     &
                   *allden 
     else
      start= start+(mp_ccoeff(0,ib)               &
                   +mp_ccoeff(1,ib)*l3qp0         &
                   +mp_ccoeff(2,ib)*l4qp0         &
                   +mp_ccoeff(3,ib)*l3qp0**2      &
                   +mp_ccoeff(4,ib)*l4qp0**2      &
                   +mp_ccoeff(5,ib)*l3qp0**3      &
                   +mp_ccoeff(6,ib)*l4qp0**3)     &
                   *allden 
     endif
    enddo  
   endif
   mp_numc= start
  end function mp_numc
!
  function mp_numb(number_propagators,q,j)
   include 'cts_mpr.h' 
    :: p
   integer, intent(in) :: number_propagators
   include 'cts_mpc.h'
    , intent(in), dimension(0:3) :: q
   integer, intent(in) :: j
   integer :: jj,i,ib,k,np
   include 'cts_mpc.h'
    , dimension(0:3) :: qp0
   include 'cts_mpc.h' 
    :: l3qp0,l4qp0,vqp0
   include 'cts_mpc.h' 
    :: mp_numb,start
   include 'cts_mpc.h' 
    :: allden
   jj= abs(j)
   start= c0(p)
   np= number_propagators
   if     (np.lt.2) then
   else   
    do ib= 1,dmns_2
     allden= c1(p)
     if     (j.eq.0) then
      do k= 3,np
       allden= allden*value(mp_den(bbn2(k,ib)),q)
      enddo
     elseif (j.lt.0) then
      do k= 3,np
       allden= allden*(mp_vden(mp_den(bbn2(k,ib))%i,jj)-mpqt2)
      enddo
     else
      do k= 3,np
       allden= allden*mp_vden(mp_den(bbn2(k,ib))%i,j)
       if (allden.eq.c0(p)) cycle
      enddo
     endif
     if (allden.eq.c0(p)) cycle
     do k= 0,3
      qp0(k)= q(k)+mp_p0vecb(k,ib) 
     enddo
     call contr(mp_l5vec(:,ib),qp0,l3qp0)
     call contr(mp_l6vec(:,ib),qp0,l4qp0)
     call contr(mp_vvecb(:,ib),qp0,vqp0)
     if (j.lt.0) then
      start= start+(                             &
               save_mp_bcoeff(0,ib)                 &
              +save_mp_bcoeff(1,ib)*l3qp0           &
              +save_mp_bcoeff(2,ib)*l4qp0           &
              +save_mp_bcoeff(3,ib)*vqp0            &
              +save_mp_bcoeff(4,ib)*(l3qp0)**2      &
              +save_mp_bcoeff(5,ib)*(l4qp0)**2      &
              +save_mp_bcoeff(6,ib)*vqp0**2         &
              +save_mp_bcoeff(7,ib)*vqp0*l3qp0      &
              +save_mp_bcoeff(8,ib)*vqp0*l4qp0)     &
              *allden
     else
      start= start+(                        &
               mp_bcoeff(0,ib)                 &
              +mp_bcoeff(1,ib)*l3qp0           &
              +mp_bcoeff(2,ib)*l4qp0           &
              +mp_bcoeff(3,ib)*vqp0            &
              +mp_bcoeff(4,ib)*(l3qp0)**2      &
              +mp_bcoeff(5,ib)*(l4qp0)**2      &
              +mp_bcoeff(6,ib)*vqp0**2         &
              +mp_bcoeff(7,ib)*vqp0*l3qp0      &
              +mp_bcoeff(8,ib)*vqp0*l4qp0)     &
              *allden
     endif 
    enddo  
   endif
   mp_numb= start
  end function mp_numb
!
  function mp_numa(number_propagators,q,j)
   include 'cts_mpr.h' 
    :: p
   integer, intent(in) :: number_propagators
   include 'cts_mpc.h'
    , intent(in), dimension(0:3) :: q
   integer, intent(in) :: j
   integer :: jj,i,ib,k,np
   include 'cts_mpc.h'
    , dimension(0:3) :: qp0
   include 'cts_mpc.h' 
    :: l3qp0,l4qp0,vqp0,kqp0
   include 'cts_mpc.h' 
    :: mp_numa,start
   include 'cts_mpc.h' 
    :: allden
   jj= abs(j)
   start= c0(p)
   np= number_propagators
   if     (np.lt.1) then
    print*,'In function numa '
    print*,'number_propagators=', number_propagators,' not allowed'
    stop
   else 
    do ib= 1,dmns_1
     allden= c1(p)
     if     (j.eq.0) then
      do k= 2,np
       allden= allden*value(mp_den(bbn1(k,ib)),q)
      enddo
     elseif (j.lt.0) then
      do k= 2,np
       allden= allden*(mp_vden(mp_den(bbn1(k,ib))%i,jj)-mpqt2)
      enddo
     else
      do k= 2,np
       allden= allden*mp_vden(mp_den(bbn1(k,ib))%i,j)
       if (allden.eq.c0(p)) cycle
      enddo
     endif  
     if (allden.eq.c0(p)) cycle
     do k= 0,3
      qp0(k)= q(k)+mp_p0veca(k,ib) 
     enddo
     call contr(mp_l7vec(:,ib),qp0,l3qp0)
     call contr(mp_l8vec(:,ib),qp0,l4qp0)
     call contr(mp_vveca(:,ib),qp0,vqp0)
     call contr(mp_kvec(:,ib),qp0,kqp0)
     if (j.lt.0) then 
      start= start+(                    &
               save_mp_acoeff(0,ib)        & 
              +save_mp_acoeff(1,ib)*kqp0   &
              +save_mp_acoeff(2,ib)*vqp0   &
              +save_mp_acoeff(3,ib)*l3qp0  &
              +save_mp_acoeff(4,ib)*l4qp0) &
              *allden
     else
      start= start+(               &
               mp_acoeff(0,ib)        & 
              +mp_acoeff(1,ib)*kqp0   &
              +mp_acoeff(2,ib)*vqp0   &
              +mp_acoeff(3,ib)*l3qp0  &
              +mp_acoeff(4,ib)*l4qp0) &
              *allden
     endif
    enddo  
   endif
   mp_numa= start 
!
  end function mp_numa
!
  subroutine mp_numfuncrec(number_propagators,q,j,mpres) 
   integer, intent(in) :: number_propagators,j
   include 'cts_mpc.h'
    , intent(in), dimension(0:3) :: q
   include 'cts_mpc.h'
    , intent(out) :: mpres
   include 'cts_mpc.h' 
    :: num_d, num_c, num_b, num_a
   num_d= numd(number_propagators,q,-j)
   num_c= numc(number_propagators,q,-j)
   num_b= numb(number_propagators,q,-j)
   num_a= numa(number_propagators,q,-j)
   mpres= num_d+num_c+num_b+num_a
  end subroutine mp_numfuncrec
!
  subroutine dp_test(p,numdummy,number_propagators,q,numerator,numrec,prec)
   use inout
   external numdummy
   include 'cts_dpr.h'
    , intent(in) :: p
   integer, intent(in) :: number_propagators
!
!  Routine to perform the test N(q) = N(q)
!
   include 'cts_dpc.h'
    , intent(in), dimension(0:3) :: q
   include 'cts_dpc.h' 
    :: numerator, num_d, num_c, num_b, num_a, numrec
   include 'cts_dpr.h' 
    :: prec,precr,preci,abnum
   intent(out) :: numerator,numrec,prec
   mprec=.false.
   dpq= q
   call numfunc(numdummy)
   numerator= dpres
   num_d= numd(number_propagators,dpq,0)
   num_c= numc(number_propagators,dpq,0)
   num_b= numb(number_propagators,dpq,0)
   num_a= numa(number_propagators,dpq,0)
   numrec= (num_d+num_c+num_b+num_a)
   abnum= max(my_tiny(p),abs(numerator)) 
   precr = dabs(dreal(numerator)-dreal(numrec))/abnum
   preci = dabs(dimag(numerator)-dimag(numrec))/abnum
   prec= max(precr,preci)
  end subroutine dp_test
!
  subroutine mp_test(p,numdummy,number_propagators,q,numerator,numrec,prec)
   use inout
   external numdummy
   include 'cts_mpr.h'
    , intent(in) :: p 
   integer, intent(in) :: number_propagators
!
!  Routine to perform the test N(q) = N(q)
!
   include 'cts_dpc.h'
    , intent(in), dimension(0:3) :: q
   include 'cts_mpc.h' 
    :: numerator, num_d, num_c, num_b, num_a, numrec
   include 'cts_mpc.h' 
    :: diff,partreal,imagpart,partreal1,imagpart1
   include 'cts_mpr.h' 
    :: abnum
   include 'cts_dpr.h' 
    :: prec,precr,preci
   integer :: k
   intent(out) :: numerator,numrec,prec
   mprec=.true.
   do k= 0,3; mpq(k)= q(k); enddo
   call numfunc(numdummy)
   numerator= mpres
   num_d= numd(number_propagators,mpq,0)
   num_c= numc(number_propagators,mpq,0)
   num_b= numb(number_propagators,mpq,0)
   num_a= numa(number_propagators,mpq,0)
   numrec= (num_d+num_c+num_b+num_a)
   diff= numerator
   partreal= (diff+conjg(diff))/2.d0  
   imagpart= (diff-conjg(diff))/(2.d0*ci(p))  
   diff= numrec
   partreal1= (diff+conjg(diff))/2.d0  
   imagpart1= (diff-conjg(diff))/(2.d0*ci(p))  
   abnum= max(abs(numerator),my_tiny(p)) 
   precr = abs(partreal-partreal1)/abnum
   preci = abs(imagpart-imagpart1)/abnum
   prec= max(precr,preci)
  end subroutine mp_test
 end module coefficients
 






 
