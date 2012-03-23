!---------------------------------------------------------------------
!
!
!     In this file (numerators.f) 
!     you should place all numerator functions
!
!
!---------------------------------------------------------------------
!
      subroutine test(q,amp)
!
!-------------------------------
!                               
!     "Test" numerator           
!                               
!-------------------------------
!
      implicit none 
      complex*16 q(0:3)
      complex*16 amp
      integer rango
      common/rango/rango
      amp  = 2.d0+(4.d0+q(0)*2.22333d0-q(1)*30.666d0+q(2)*1.55d0 
     &       +q(3)*3.6541122d0)**rango

!      amp= (2.d0+(4.d0+q(0)*2.22333d0-q(1)*30.666d0+q(2)*1.55d0 
!     &            +q(3)*3.6541122d0)**2
!     &     )*(q(0)**2-q(1)**2-q(2)**2-q(3)**2+7.d0)*
!     &     (q(0)*1.12333d0-q(1)*10.666d0+q(2)*5.55d0 
!     &            -q(3)*2.6541122d0)
      end subroutine test
