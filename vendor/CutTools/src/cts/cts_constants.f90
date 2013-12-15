!
! define some constants 
!
 module constants  
  include 'cts_mprec.h'
  implicit none
  private
  public root3,pi,lambda,sigma,c0,c1,ci,cexp1,cexp2,cexp3,cexp4
  public cexpk1,tau11,tau12,my_tiny
!
  interface root3
    module procedure dp_root3
    module procedure mp_root3
  end interface!root3
!
  interface pi
    module procedure dp_pi
    module procedure mp_pi
  end interface!pi
!
  interface lambda
    module procedure dp_lambda
    module procedure mp_lambda
  end interface!lambda
!
  interface sigma
    module procedure dp_sigma
    module procedure mp_sigma
  end interface!sigma
!
  interface c0
    module procedure dp_c0
    module procedure mp_c0
  end interface!c0
!
  interface c1
    module procedure dp_c1
    module procedure mp_c1
  end interface!c1
!
  interface ci
    module procedure dp_ci
    module procedure mp_ci
  end interface!ci
!
  interface cexp1
    module procedure dp_cexp1
    module procedure mp_cexp1
  end interface!cexp1
!
  interface cexp2
    module procedure dp_cexp2
    module procedure mp_cexp2
  end interface!cexp2
!
  interface cexp3
    module procedure dp_cexp3
    module procedure mp_cexp3
  end interface!cexp3
!
  interface cexp4
    module procedure dp_cexp4
    module procedure mp_cexp4
  end interface!cexp4
!
  interface cexpk1
    module procedure dp_cexpk1
    module procedure mp_cexpk1
  end interface!cexpk1
!
  interface tau11
    module procedure dp_tau11
    module procedure mp_tau11
  end interface!tau11
!
  interface tau12
    module procedure dp_tau12
    module procedure mp_tau12
  end interface!tau12
!
  interface my_tiny
    module procedure dp_tiny
    module procedure mp_tiny
  end interface!my_tiny
  contains
!
  function dp_root3(p)
   include 'cts_dpr.h'
    , intent(in) :: p
   include 'cts_dpr.h' 
    :: dp_root3,aus  
   logical :: computing=.true.
   save aus,computing
   if (computing) then
    computing=.false.
    aus= sqrt(3.d0)
   endif
   dp_root3= aus
  end function dp_root3
!
  function dp_pi(p)
   include 'cts_dpr.h'
    , intent(in) :: p
   include 'cts_dpr.h' 
    :: dp_pi,aus  
   logical :: computing=.true.
   save aus,computing
   if (computing) then
    computing=.false.
    aus= 4.d0*atan(1.d0)
   endif
   dp_pi= aus
  end function dp_pi
!
  function dp_lambda(p)
   include 'cts_dpr.h'
    , intent(in) :: p
   include 'cts_dpr.h' 
    :: dp_lambda,aus
   logical :: computing=.true.
   save aus,computing
   if (computing) then
    computing=.false.
    aus= -1.d0
   endif
   dp_lambda= aus          
  end function dp_lambda
!
  function dp_sigma(p)
   include 'cts_dpr.h'
    , intent(in) :: p
   include 'cts_dpr.h' 
    :: dp_sigma,aus
   logical :: computing=.true.
   save aus,computing
   if (computing) then
    computing=.false.
    aus= -0.5d0
   endif
   dp_sigma= aus          
  end function dp_sigma
!
  function dp_c0(p)
   include 'cts_dpr.h'
    , intent(in) :: p
   include 'cts_dpc.h' 
    :: dp_c0,aus
   logical :: computing=.true.
   save aus,computing
   if (computing) then
    computing=.false.
    aus= dcmplx(0.d0,0.d0)
   endif
   dp_c0= aus
  end function dp_c0
!
  function dp_c1(p)
   include 'cts_dpr.h'
    , intent(in) :: p
   include 'cts_dpc.h' 
    :: dp_c1,aus
   logical :: computing=.true.
   save aus,computing
   if (computing) then
    computing=.false.
    aus= dcmplx(1.d0,0.d0)
   endif
   dp_c1= aus
  end function dp_c1
!
  function dp_ci(p)
   include 'cts_dpr.h'
    , intent(in) :: p
   include 'cts_dpc.h' 
    :: dp_ci,aus
   logical :: computing=.true.
   save aus,computing
   if (computing) then
    computing=.false.
    aus= dcmplx(0.d0,1.d0)
   endif
   dp_ci= aus 
  end function dp_ci
!
  function dp_cexp1(p)
   include 'cts_dpr.h'
    , intent(in) :: p
   include 'cts_dpc.h' 
    :: dp_cexp1,aus  
   logical :: computing=.true.
   save aus,computing
   if (computing) then
    computing=.false.
    aus= exp(ci(p)*pi(p)/1.d0)
   endif
   dp_cexp1= aus
  end function dp_cexp1
!
  function dp_cexp2(p)
   include 'cts_dpr.h'
    , intent(in) :: p
   include 'cts_dpc.h' 
    :: dp_cexp2,aus  
   logical :: computing=.true.
   save aus,computing
   if (computing) then
    computing=.false.
    aus= exp(ci(p)*pi(p)/2.d0)
   endif
   dp_cexp2= aus
  end function dp_cexp2
!
  function dp_cexp3(p)
   include 'cts_dpr.h'
    , intent(in) :: p
   include 'cts_dpc.h' 
    :: dp_cexp3,aus  
   logical :: computing=.true.
   save aus,computing
   if (computing) then
    computing=.false.
    aus= exp(ci(p)*pi(p)/3.d0)
   endif
   dp_cexp3= aus
  end function dp_cexp3
!
  function dp_cexp4(p)
   include 'cts_dpr.h'
    , intent(in) :: p
   include 'cts_dpc.h' 
    :: dp_cexp4,aus  
   logical :: computing=.true.
   save aus,computing
   if (computing) then
    computing=.false.
    aus= exp(ci(p)*pi(p)/4.d0)
   endif
   dp_cexp4= aus
  end function dp_cexp4
!
  function dp_cexpk1(p)
   include 'cts_dpr.h'
    , intent(in) :: p
   include 'cts_dpc.h' 
    :: dp_cexpk1,aus  
   logical :: computing=.true.
   save aus,computing
   if (computing) then
    computing=.false.
    aus= exp(ci(p)*pi(p)/3.d0)
   endif
   dp_cexpk1= aus
  end function dp_cexpk1
!
  function dp_tau11(p)
   include 'cts_dpr.h'
    , intent(in) :: p
   include 'cts_dpc.h' 
    :: dp_tau11,aus
   logical :: computing=.true.
   save aus,computing
   if (computing) then
    computing=.false.
    aus= 1.d0
   endif
   dp_tau11= aus
  end function dp_tau11
!
  function dp_tau12(p)
   include 'cts_dpr.h'
    , intent(in) :: p
   include 'cts_dpc.h' 
    :: dp_tau12,aus
   logical :: computing=.true.
   save aus,computing
   if (computing) then
    computing=.false.
    aus= 1.d0 
   endif
   dp_tau12= aus
  end function dp_tau12
!
  function dp_tiny(p)
   include 'cts_dpr.h'
    , intent(in) :: p
   include 'cts_dpr.h' 
    :: dp_tiny,aus
   logical :: computing=.true.
   save aus,computing
   if (computing) then
    computing=.false.
    aus= tiny(p) 
   endif
   dp_tiny= aus
  end function dp_tiny 
!
  function mp_root3(p)
   include 'cts_mpr.h'
    , intent(in) :: p
   include 'cts_mpr.h' 
    :: mp_root3,aus  
   logical :: computing=.true.
   save aus,computing
   if (computing) then
    computing=.false.
    aus= 3.d0
    aus= sqrt(aus)
   endif
   mp_root3= aus
  end function mp_root3
!
  function mp_pi(p)
   include 'cts_mpr.h'
    , intent(in) :: p
   include 'cts_mpr.h' 
    :: mp_pi,aus  
   logical :: computing=.true.
   save aus,computing
   if (computing) then
    computing=.false.
    aus= 1.d0
    aus= 4.d0*atan(aus)
   endif
   mp_pi= aus
  end function mp_pi
!
  function mp_lambda(p)
   include 'cts_mpr.h'
    , intent(in) :: p
   include 'cts_mpr.h' 
    :: mp_lambda,aus
   logical :: computing=.true.
   save aus,computing
   if (computing) then
    computing=.false.
    aus= -1.d0
   endif
   mp_lambda= aus
  end function mp_lambda
!
  function mp_sigma(p)
   include 'cts_mpr.h'
    , intent(in) :: p
   include 'cts_mpr.h' 
    :: mp_sigma,aus
   logical :: computing=.true.
   save aus,computing
   if (computing) then
    computing=.false.
    aus= -0.5d0
   endif
   mp_sigma= aus
  end function mp_sigma
!
  function mp_c0(p)
   include 'cts_mpr.h'
    , intent(in) :: p
   include 'cts_mpc.h' 
    :: mp_c0,aus
   logical :: computing=.true.
   save aus,computing
   if (computing) then
    computing=.false.
    aus= dcmplx(0.d0,0.d0)
   endif
   mp_c0= aus
  end function mp_c0
!
  function mp_c1(p)
   include 'cts_mpr.h'
    , intent(in) :: p
   include 'cts_mpc.h' 
    :: mp_c1,aus
   logical :: computing=.true.
   save aus,computing
   if (computing) then
    computing=.false.
    aus= dcmplx(1.d0,0.d0)
   endif
   mp_c1= aus
  end function mp_c1
!
  function mp_ci(p)
   include 'cts_mpr.h'
    , intent(in) :: p
   include 'cts_mpc.h' 
    :: mp_ci,aus
   logical :: computing=.true.
   save aus,computing
   if (computing) then
    computing=.false.
    aus= dcmplx(0.d0,1.d0)
   endif
   mp_ci= aus 
  end function mp_ci
!
  function mp_cexp1(p)
   include 'cts_mpr.h'
    , intent(in) :: p
   include 'cts_mpc.h' 
    :: mp_cexp1,aus  
   logical :: computing=.true.
   save aus,computing
   if (computing) then
    computing=.false.
    aus= exp(ci(p)*pi(p)/1.d0)
   endif
   mp_cexp1= aus
  end function mp_cexp1
!
  function mp_cexp2(p)
   include 'cts_mpr.h'
    , intent(in) :: p
   include 'cts_mpc.h' 
    :: mp_cexp2,aus  
   logical :: computing=.true.
   save aus,computing
   if (computing) then
    computing=.false.
    aus= exp(ci(p)*pi(p)/2.d0)
   endif
   mp_cexp2= aus
  end function mp_cexp2
!
  function mp_cexp3(p)
   include 'cts_mpr.h'
    , intent(in) :: p
   include 'cts_mpc.h' 
    :: mp_cexp3,aus  
   logical :: computing=.true.
   save aus,computing
   if (computing) then
    computing=.false.
    aus= exp(ci(p)*pi(p)/3.d0)
   endif
   mp_cexp3= aus
  end function mp_cexp3
!
  function mp_cexp4(p)    
   include 'cts_mpr.h'
    , intent(in) :: p
   include 'cts_mpc.h' 
    :: mp_cexp4,aus  
   logical :: computing=.true.
   save aus,computing
   if (computing) then
    computing=.false.
    aus= exp(ci(p)*pi(p)/4.d0)
   endif
   mp_cexp4= aus
  end function mp_cexp4
!
  function mp_cexpk1(p)
   include 'cts_mpr.h'
    , intent(in) :: p
   include 'cts_mpc.h' 
    :: mp_cexpk1,aus  
   logical :: computing=.true.
   save aus,computing
   if (computing) then
    computing=.false.
    aus= exp(ci(p)*pi(p)/3.d0)
   endif
   mp_cexpk1= aus
  end function mp_cexpk1
!
  function mp_tau11(p)
   include 'cts_mpr.h'
    , intent(in) :: p
   include 'cts_mpc.h' 
    :: mp_tau11,aus
   logical :: computing=.true.
   save aus,computing
   if (computing) then
    computing=.false.
    aus= 1.d0
   endif
   mp_tau11= aus
  end function mp_tau11
!
  function mp_tau12(p)
   include 'cts_mpr.h'
    , intent(in) :: p
   include 'cts_mpc.h' 
    :: mp_tau12,aus
   logical :: computing=.true.
   save aus,computing
   if (computing) then
    computing=.false.
    aus= 1.d0 
   endif
   mp_tau12= aus
  end function mp_tau12
!
  function mp_tiny(p)
   include 'cts_mpr.h'
    , intent(in) :: p
   include 'cts_mpr.h' 
    :: mp_tiny,aus
   logical :: computing=.true.
   save aus,computing
   if (computing) then
    computing=.false.
    aus= tiny(1.d0)
   endif
   mp_tiny= aus
  end function mp_tiny 
 end module constants

