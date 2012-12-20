 module tensor_operations                                                 
  include 'cts_mprec.h'
  implicit none
  private
  public :: contr
  interface contr 
!
!  Real Tensor by Real Tensor
!
   module procedure  contract11_1rr 
!
!  Complex Tensor by Complex Tensor
!
   module procedure  contract11_1cc 
!
!  Real Tensor by Complex Tensor
!
   module procedure  contract11_1rc 
!
!  Complex Tensor by Real Tensor
!
   module procedure  contract11_1cr 
!
!  mp_ Real Tensor by mp_Real Tensor
!
   module procedure  mp_contract11_1rr 
!
!  mp_Complex Tensor by mp_Complex Tensor
!
   module procedure  mp_contract11_1cc 
!
!  mp_Real Tensor by mp_Complex Tensor
!
   module procedure  mp_contract11_1rc 
!
!  mp_Complex Tensor by mp_Real Tensor
!
   module procedure  mp_contract11_1cr 
  end interface!contr
  contains                                                                 
!
!  Real by Real:
!
  subroutine contract11_1rr(a,b,aus1) 
   include 'cts_dpr.h'
    , intent(out) :: aus1            
   include 'cts_dpr.h'
    , intent(in), dimension(0:3) :: a                          
   include 'cts_dpr.h'
    , intent(in), dimension(0:3) :: b                          
   aus1= a(0)*b(0)-a(1)*b(1)-a(2)*b(2)-a(3)*b(3)
  end subroutine contract11_1rr                                               
!
!  Complex by Complex:
!
  subroutine contract11_1cc(a,b,aus1) 
   include 'cts_dpc.h'
    , intent(out)  :: aus1
   include 'cts_dpc.h'
    , intent(in), dimension(0:3) :: a
   include 'cts_dpc.h'
    , intent(in), dimension(0:3) :: b
   aus1= a(0)*b(0)-a(1)*b(1)-a(2)*b(2)-a(3)*b(3)
  end subroutine contract11_1cc                                               
!
!  Real by Complex:
!
  subroutine contract11_1rc(a,b,aus1) 
   include 'cts_dpc.h'
    , intent(out)  :: aus1
   include 'cts_dpr.h'
    , intent(in), dimension(0:3) :: a                          
   include 'cts_dpc.h'
    , intent(in), dimension(0:3) :: b   
   aus1= a(0)*b(0)-a(1)*b(1)-a(2)*b(2)-a(3)*b(3)
  end subroutine contract11_1rc                                               
!
!  Complex by Real:
!
  subroutine contract11_1cr(a,b,aus1) 
   include 'cts_dpc.h'
    , intent(out)  :: aus1
   include 'cts_dpc.h'
    , intent(in), dimension(0:3) :: a 
   include 'cts_dpr.h'
    , intent(in), dimension(0:3) :: b                          
   aus1= a(0)*b(0)-a(1)*b(1)-a(2)*b(2)-a(3)*b(3)
  end subroutine contract11_1cr                                               
!
!  mp_Real by mp_Real:
!
  subroutine mp_contract11_1rr(a,b,aus1) 
   include 'cts_mpr.h'
    , intent(out) :: aus1
   include 'cts_mpr.h'
    , intent(in), dimension(0:3) :: a                          
   include 'cts_mpr.h'
    , intent(in), dimension(0:3) :: b                          
   aus1= a(0)*b(0)-a(1)*b(1)-a(2)*b(2)-a(3)*b(3)
  end subroutine mp_contract11_1rr     
!
!  mp_Complex by mp_Complex:
!
  subroutine mp_contract11_1cc(a,b,aus1) 
   include 'cts_mpc.h'
    , intent(out)  :: aus1
   include 'cts_mpc.h'
    , intent(in), dimension(0:3) :: a                          
   include 'cts_mpc.h'
    , intent(in), dimension(0:3) :: b                          
   aus1= a(0)*b(0)-a(1)*b(1)-a(2)*b(2)-a(3)*b(3)
  end subroutine mp_contract11_1cc                                     
!
!  mp_Real by mp_Complex:
!
  subroutine mp_contract11_1rc(a,b,aus1) 
   include 'cts_mpc.h'
    , intent(out)  :: aus1
   include 'cts_mpr.h'
    , intent(in), dimension(0:3) :: a                          
   include 'cts_mpc.h'
    , intent(in), dimension(0:3) :: b                          
   aus1= a(0)*b(0)-a(1)*b(1)-a(2)*b(2)-a(3)*b(3)
  end subroutine mp_contract11_1rc
!
!  mp_Complex by mp_Real:
!
  subroutine mp_contract11_1cr(a,b,aus1) 
   include 'cts_mpc.h'
    , intent(out)  :: aus1
   include 'cts_mpc.h'
    , intent(in), dimension(0:3) :: a                          
   include 'cts_mpr.h'
    , intent(in), dimension(0:3) :: b                          
   aus1= a(0)*b(0)-a(1)*b(1)-a(2)*b(2)-a(3)*b(3)
  end subroutine mp_contract11_1cr  
 end module tensor_operations                                             



