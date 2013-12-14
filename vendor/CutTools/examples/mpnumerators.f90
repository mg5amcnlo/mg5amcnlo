!---------------------------------------------------------------------
!
!
!     In this file (mpnumerators.f90) 
!     you should place all multi-precision numerator functions
!
!
!---------------------------------------------------------------------
!
      subroutine dummy(q,amp)
!
!-----------------------------------------------
!                               
!     dummy subroutine, to be calld when idig= 0           
!                               
!----------------------------------------------

      include 'cts_mprec.h'
      implicit none
      include 'cts_mpc.h'
       , intent(in), dimension(0:3) :: q
      include 'cts_mpc.h'
       , intent(out) :: amp
      amp = 0
      end subroutine dummy
!      
      subroutine mptest(q,amp)
!
!-------------------------------
!                               
!     "Test" numerator           
!                               
!-------------------------------
!
      include 'cts_mprec.h'
      implicit none
      include 'cts_mpc.h'
       , intent(in), dimension(0:3) :: q
      include 'cts_mpc.h'
       , intent(out) :: amp
      complex(kind(1.d0)) :: dpamp,dpq(0:3)
      integer ::  j
      integer rango
      common/rango/rango
!
! comment: the double precision version of the function is called here 
!
!      amp  = 2.d0+(4.d0+q(0)*2.22333d0-q(1)*30.666d0+q(2)*1.55d0 &
!     &       +q(3)*3.6541122d0)**rango

!      amp= (2.d0+(4.d0+q(0)*2.22333d0-q(1)*30.666d0+q(2)*1.55d0 &
!                  +q(3)*3.6541122d0)**2                         &
!           )*(q(0)**2-q(1)**2-q(2)**2-q(3)**2+7.d0)*            &
!           (q(0)*1.12333d0-q(1)*10.666d0+q(2)*5.55d0            &  
!                  -q(3)*2.6541122d0)

      do j= 0,3; dpq(j)= q(j); enddo
      call test(dpq,dpamp)
      amp= dpamp
! comment
      end subroutine mptest

      subroutine mpfortest(q,amp)
!
!-------------------------------
!                               
!     Numerator for testing the 
!     implementation of the 
!     multiprecision routines           
!                               
!-------------------------------
!
      include 'cts_mprec.h'
      implicit none
      include 'cts_mpc.h'
       , intent(in), dimension(0:3) :: q
      include 'cts_mpc.h'
       , intent(out) :: amp
      complex(kind(1.d0)) :: dpamp,dpq(0:3)
      integer ::  j
      integer rango
      common/rango/rango
      amp  = 2.d0+(4.d0+q(0)*2.22333d0-q(1)*30.666d0+q(2)*1.55d0 &
     &       +q(3)*3.6541122d0)**rango

      end subroutine mpfortest
