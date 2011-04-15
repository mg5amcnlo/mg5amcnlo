      subroutine sstxxx(s1,s2,tc,gt,smass , vertex)
c
c This subroutine computes an amplitude of the three-point coupling of
c two scalar and a tensor boson.
c
c input:
c       complex s1(3)          : first  scalar                        s1
c       complex s2(3)          : second scalar                        s2
c       complex tc(18)         : input  tensor                         T
c       complex gt             : coupling constant         gts=-1/Lambda
c       real    smass          : scalar mass                         m_s
c
c output:
c       complex vertex         : amplitude                gamma(s1,s2,T)
c
c- by Q.Li - OCT. 2006
c     
      implicit none
      double complex s1(3), s2(3), tc(18), vertex
      double complex gt
      double precision smass

      double complex ft(6,4)
      double complex T12, T13, T14, T23, T24, T34
      double complex TKK
      double precision ps1(4), ps2(4)
	integer i
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

      ps1(1) = dreal(s1(2))
      ps1(2) = dreal(s1(3))
      ps1(3) = dimag(s1(3))
      ps1(4) = dimag(s1(2))

      ps2(1) = -dreal(s2(2))
      ps2(2) = -dreal(s2(3))
      ps2(3) = -dimag(s2(3))
      ps2(4) = -dimag(s2(2))

      T12 = ft(1,2) + ft(2,1)
      T13 = ft(1,3) + ft(3,1)
      T14 = ft(1,4) + ft(4,1)
      T23 = ft(2,3) + ft(3,2)
      T24 = ft(2,4) + ft(4,2)
      T34 = ft(3,4) + ft(4,3)


      TKK   = cZero
    
      do i = 1,4
         TKK=TKK+ft(i,i)*ps1(i)*ps2(i)
      end do

      TKK   = rTwo*TKK

      TKK = TKK - T12*(ps1(1)*ps2(2) + ps1(2)*ps2(1))
     &          - T13*(ps1(1)*ps2(3) + ps1(3)*ps2(1))
     &          - T14*(ps1(1)*ps2(4) + ps1(4)*ps2(1))
     &          + T23*(ps1(2)*ps2(3) + ps1(3)*ps2(2))
     &          + T24*(ps1(2)*ps2(4) + ps1(4)*ps2(2))
     &          + T34*(ps1(3)*ps2(4) + ps1(4)*ps2(3))


      vertex = TKK+(ft(1,1)-ft(2,2)-ft(3,3)-ft(4,4))
     &	*(smass**2-ps1(1)*ps2(1)+ps1(2)*ps2(2)
     &      +ps1(3)*ps2(3)+ps1(4)*ps2(4))

      vertex = vertex * gt*s1(1)*s2(1)

      return
      end
