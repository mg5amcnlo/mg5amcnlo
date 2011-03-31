      subroutine vvtcxx(v1,v2,tc,gt,vmass , vertex)
c-------------------CP3  2009.10-----------------
c This subroutine computes an amplitude of the three-point coupling of
c two gauge bosons and a non-propagating tensor boson.
c
c input:
c       complex v1(6)          : first  vector                            v1
c       complex v2(6)          : second vector                        v2
c       complex tc(18)         : input  tensor                           T
c       complex gt             : coupling constant                   gt=gs
c       real    vmass          : vector boson mass                   m_v
c
c output:
c       complex vertex         : amplitude                gamma(v1,v2,T)
c
c     
      implicit none
      double complex v1(6), v2(6), tc(18), vertex
      double precision vmass, gt

      double complex ft(6,4)
      double complex  dum

      integer i, j

      double complex cZero
      double precision rZero, rTwo
      parameter( rZero = 0.0d0, rTwo = 2.0d0 )
      parameter( cZero = ( 0.0d0, 0.0d0 ) )

      
      ft(1,1) = tc(1)
      ft(1,2) = tc(2)
      ft(1,3) = tc(3)
      ft(1,4) = tc(4)
      ft(2,1) = tc(5)
      ft(2,2) = tc(6)
      ft(2,3) = tc(7)
      ft(2,4) = tc(8)
      ft(3,1) = tc(9)
      ft(3,2) = tc(10)
      ft(3,3) = tc(11)
      ft(3,4) = tc(12)
      ft(4,1) = tc(13)
      ft(4,2) = tc(14)
      ft(4,3) = tc(15)
      ft(4,4) = tc(16)
      ft(5,1) = tc(17)
      ft(6,1) = tc(18)

        vertex=ft(1,1)*v1(1)*v2(1) -ft(2,1)*v1(2)*v2(1) - 
     -  ft(3,1)*v1(3)*v2(1) - ft(4,1)*v1(4)*v2(1) - 
     -  ft(1,2)*v1(1)*v2(2) + ft(2,2)*v1(2)*v2(2) + 
     -  ft(3,2)*v1(3)*v2(2) + ft(4,2)*v1(4)*v2(2) - 
     -  ft(1,3)*v1(1)*v2(3) + ft(2,3)*v1(2)*v2(3) + 
     -  ft(3,3)*v1(3)*v2(3) + ft(4,3)*v1(4)*v2(3) - 
     -  ft(1,4)*v1(1)*v2(4) + ft(2,4)*v1(2)*v2(4) + 
     -  ft(3,4)*v1(3)*v2(4) + ft(4,4)*v1(4)*v2(4)

      vertex = vertex * gt

      return
      end
