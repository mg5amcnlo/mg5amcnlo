 module def_propagator
  implicit none
  private
  public :: propagator 
  type propagator
   integer :: i
   include 'cts_dpc.h'
    :: m2
   include 'cts_dpr.h'
    , dimension(0:3) :: p
  end type propagator
 end module def_propagator                        
!
 module def_solcut
  implicit none
  private
  public :: solcut4,solcut3,solcut2,solcut1
  type solcut4
   include 'cts_dpc.h'
    , dimension(0:3,1:2) :: q
   include 'cts_dpc.h'
    , dimension(0:3) :: l1,l2,l3,l4
   include 'cts_dpc.h'
    , dimension(0:3) :: t
  end type solcut4 
  type solcut3
   include 'cts_dpc.h'
    , dimension(0:3,1:7) :: q 
   include 'cts_dpc.h'
    , dimension(0:3) :: l1,l2,l3,l4
   include 'cts_dpc.h'
    :: gm,cc,tau,rat1
  end type solcut3
  type solcut2
   include 'cts_dpc.h'
    , dimension(0:3,1:9) :: q 
   include 'cts_dpr.h'
    , dimension(0:3) :: k1,v
   include 'cts_dpc.h'
    , dimension(0:3) :: l3,l4
   include 'cts_dpc.h'
    :: gm,cflambda,cfsigma,rat1,rat1t,tau,taul,cf0 
  end type solcut2 
  type solcut1
   include 'cts_dpc.h'
    , dimension(0:3,1:5) :: q
   include 'cts_dpr.h'
    , dimension(0:3) :: k,v
   include 'cts_dpc.h'
    , dimension(0:3) :: l3,l4
   include 'cts_dpc.h'
    :: gm,cf0,apar,root,rat1
  end type solcut1 
 end module def_solcut                       
!
 module def_mp_propagator
  include 'cts_mprec.h'
  implicit none
  private
  public :: mp_propagator
  type mp_propagator
   integer :: i
   include 'cts_mpc.h'
    :: m2 
   include 'cts_mpr.h'
    :: qt
   include 'cts_mpr.h'
    , dimension(0:3) :: p
  end type mp_propagator
 end module def_mp_propagator                        
!
 module def_mp_solcut
  include 'cts_mprec.h'
  implicit none
  private
  public :: mp_solcut4,mp_solcut3,mp_solcut2,mp_solcut1
  type mp_solcut4
   include 'cts_mpc.h'
    , dimension(0:3,1:2) :: q
   include 'cts_mpc.h'
    , dimension(0:3) :: l1,l2,l3,l4
   include 'cts_mpc.h'
    , dimension(0:3) :: t
  end type mp_solcut4 
  type mp_solcut3
   include 'cts_mpc.h'
    , dimension(0:3,1:7) :: q
   include 'cts_mpc.h'
    , dimension(0:3) :: l1,l2,l3,l4
   include 'cts_mpc.h'
    :: gm,cc,tau,rat1
  end type mp_solcut3
  type mp_solcut2
   include 'cts_mpc.h'
    , dimension(0:3,1:9) :: q
   include 'cts_mpr.h'
    , dimension(0:3) :: k1,v
   include 'cts_mpc.h'
    , dimension(0:3) :: l3,l4
   include 'cts_mpc.h'
    :: gm,cflambda,cfsigma,rat1,rat1t,tau,taul,cf0
  end type mp_solcut2 
  type mp_solcut1
   include 'cts_mpc.h'
    , dimension(0:3,1:5) :: q
   include 'cts_mpr.h'
    , dimension(0:3) :: k,v
   include 'cts_mpc.h'
    , dimension(0:3) :: l3,l4
   include 'cts_mpc.h'
    :: gm,cf0,apar,root,rat1
  end type mp_solcut1 
 end module def_mp_solcut                       


