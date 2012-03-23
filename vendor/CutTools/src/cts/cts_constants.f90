!
! define some constants 
!
 module constants  
  include 'cts_mprec.h'
  implicit none
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
  interface eps
    module procedure dp_eps
    module procedure mp_eps
  end interface!eps
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
  interface kappa
    module procedure dp_kappa
    module procedure mp_kappa
  end interface!kappa
!
  interface kappa1
    module procedure dp_kappa1
    module procedure mp_kappa1
  end interface!kappa1
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
  interface cexp6
    module procedure dp_cexp6
    module procedure mp_cexp6
  end interface!cexp6
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
   dp_root3 = aus
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
    aus    = 4.d0*atan(1.d0)
   endif
   dp_pi = aus
  end function dp_pi
!
  function dp_eps(p)
   include 'cts_dpr.h'
    , intent(in) :: p
   include 'cts_dpr.h' 
    :: dp_eps
   dp_eps   = 1.d-15
  end function dp_eps
!
  function dp_lambda(p)
   include 'cts_dpr.h'
    , intent(in) :: p
   include 'cts_dpr.h' 
    :: dp_lambda
   dp_lambda=-1.d0          
  end function dp_lambda
!
  function dp_sigma(p)
   include 'cts_dpr.h'
    , intent(in) :: p
   include 'cts_dpr.h' 
    :: dp_sigma
   dp_sigma =-0.5d0          
  end function dp_sigma
!
  function dp_kappa(p)
   include 'cts_dpr.h'
    , intent(in) :: p
   include 'cts_dpr.h' 
    :: dp_kappa
   dp_kappa = 3.d0
  end function dp_kappa
!
  function dp_kappa1(p)
   include 'cts_dpr.h'
    , intent(in) :: p
   include 'cts_dpr.h' 
    :: dp_kappa1
   dp_kappa1= 3.d0
  end function dp_kappa1
!
  function dp_c0(p)
   include 'cts_dpr.h'
    , intent(in) :: p
   include 'cts_dpc.h' 
    :: dp_c0
   dp_c0    = dcmplx(0.d0,0.d0)
  end function dp_c0
!
  function dp_c1(p)
   include 'cts_dpr.h'
    , intent(in) :: p
   include 'cts_dpc.h' 
    :: dp_c1
   dp_c1    = dcmplx(1.d0,0.d0)
  end function dp_c1
!
  function dp_ci(p)
   include 'cts_dpr.h'
    , intent(in) :: p
   include 'cts_dpc.h' 
    :: dp_ci
   dp_ci    = dcmplx(0.d0,1.d0)
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
    aus = exp(ci(p)*pi(p)/1.d0)
   endif
   dp_cexp1 = aus
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
    aus = exp(ci(p)*pi(p)/2.d0)
   endif
   dp_cexp2 = aus
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
    aus = exp(ci(p)*pi(p)/3.d0)
   endif
   dp_cexp3 = aus
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
    aus = exp(ci(p)*pi(p)/4.d0)
   endif
   dp_cexp4 = aus
  end function dp_cexp4
!
! begin end_09
!
  function dp_cexp6(p)
   include 'cts_dpr.h'
    , intent(in) :: p
   include 'cts_dpc.h' 
    :: dp_cexp6,aus  
   logical :: computing=.true.
   save aus,computing
   if (computing) then
    computing=.false.
    aus = exp(ci(p)*pi(p)/6.d0)
   endif
   dp_cexp6 = aus
  end function dp_cexp6
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
    aus = exp(ci(p)*pi(p)/kappa1(p))
   endif
   dp_cexpk1= aus
  end function dp_cexpk1
!
  function dp_tau11(p)
   include 'cts_dpr.h'
    , intent(in) :: p
   include 'cts_dpc.h' 
    :: dp_tau11
   dp_tau11  = 1.d0
  end function dp_tau11
!
  function dp_tau12(p)
   include 'cts_dpr.h'
    , intent(in) :: p
   include 'cts_dpc.h' 
    :: dp_tau12
   dp_tau12  = 1.d0
  end function dp_tau12
!
  function mp_tiny(p)
   include 'cts_mpr.h'
    , intent(in) :: p
   include 'cts_mpr.h' 
    :: mp_tiny
   mp_tiny   = 1.d-60
  end function mp_tiny 
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
   mp_root3 = aus
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
    aus    = 1.d0
    aus    = 4.d0*atan(aus)
   endif
   mp_pi = aus
  end function mp_pi
!
  function mp_eps(p)
   include 'cts_mpr.h'
    , intent(in) :: p
   include 'cts_mpr.h' 
    :: mp_eps
   mp_eps   = 1.d-15
  end function mp_eps
!
  function mp_lambda(p)
   include 'cts_mpr.h'
    , intent(in) :: p
   include 'cts_mpr.h' 
    :: mp_lambda
   mp_lambda=-1.d0
  end function mp_lambda
!
  function mp_sigma(p)
   include 'cts_mpr.h'
    , intent(in) :: p
   include 'cts_mpr.h' 
    :: mp_sigma
   mp_sigma = -0.5d0
  end function mp_sigma
!
  function mp_kappa(p)
   include 'cts_mpr.h'
    , intent(in) :: p
   include 'cts_mpr.h' 
    :: mp_kappa
   mp_kappa = 3.d0
  end function mp_kappa
!
  function mp_kappa1(p)
   include 'cts_mpr.h'
    , intent(in) :: p
   include 'cts_mpr.h' 
    :: mp_kappa1
   mp_kappa1= 3.d0
  end function mp_kappa1
!
  function mp_c0(p)
   include 'cts_mpr.h'
    , intent(in) :: p
   include 'cts_mpc.h' 
    :: mp_c0
   mp_c0    = dcmplx(0.d0,0.d0)
  end function mp_c0
!
  function mp_c1(p)
   include 'cts_mpr.h'
    , intent(in) :: p
   include 'cts_mpc.h' 
    :: mp_c1
   mp_c1    = dcmplx(1.d0,0.d0)
  end function mp_c1
!
  function mp_ci(p)
   include 'cts_mpr.h'
    , intent(in) :: p
   include 'cts_mpc.h' 
    :: mp_ci
   mp_ci    = dcmplx(0.d0,1.d0)
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
    aus = exp(ci(p)*pi(p)/1.d0)
   endif
   mp_cexp1 = aus
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
    aus = exp(ci(p)*pi(p)/2.d0)
   endif
   mp_cexp2 = aus
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
    aus = exp(ci(p)*pi(p)/3.d0)
   endif
   mp_cexp3 = aus
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
    aus = exp(ci(p)*pi(p)/4.d0)
   endif
   mp_cexp4 = aus
  end function mp_cexp4
!
  function mp_cexp6(p)
   include 'cts_mpr.h'
    , intent(in) :: p
   include 'cts_mpc.h' 
    :: mp_cexp6,aus  
   logical :: computing=.true.
   save aus,computing
   if (computing) then
    computing=.false.
    aus = exp(ci(p)*pi(p)/6.d0)
   endif
   mp_cexp6 = aus
  end function mp_cexp6
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
    aus = exp(ci(p)*pi(p)/kappa1(p))
   endif
   mp_cexpk1= aus
  end function mp_cexpk1
!
  function mp_tau11(p)
   include 'cts_mpr.h'
    , intent(in) :: p
   include 'cts_mpc.h' 
    :: mp_tau11
   mp_tau11  = 1.d0
  end function mp_tau11
!
  function mp_tau12(p)
   include 'cts_mpr.h'
    , intent(in) :: p
   include 'cts_mpc.h' 
    :: mp_tau12
   mp_tau12  = 1.d0
  end function mp_tau12
 end module constants
