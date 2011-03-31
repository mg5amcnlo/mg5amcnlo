      subroutine vvtxxx(v1,v2,tc,gt,vmass , vertex)
c
c This subroutine computes an amplitude of the three-point coupling of
c two gauge bosons and a tensor boson.
c
c input:
c       complex v1(6)          : first  vector                        v1
c       complex v2(6)          : second vector                        v2
c       complex tc(18)         : input  tensor                         T
c       complex gt             : coupling constant         gtv=-1/Lambda
c       real    vmass          : vector boson mass                   m_v
c
c output:
c       complex vertex         : amplitude                gamma(v1,v2,T)
c
c- by Q.Li - OCT. 2006
c     
      implicit none
      double complex v1(6), v2(6), tc(18), gt, vertex
      double precision vmass

      double complex ft(6,4)
      double complex T12, T13, T14, T23, T24, T34
      double complex V1V2, K1V2, K2V1
c     new
     &,K1V1,K2V2
c     new
      double complex TKK, TVV, TK1V2, TK2V1, dum
      double precision pv1(4), pv2(4), F

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

      pv1(1) = dreal(v1(5))
      pv1(2) = dreal(v1(6))
      pv1(3) = dimag(v1(6))
      pv1(4) = dimag(v1(5))
      pv2(1) = dreal(v2(5))
      pv2(2) = dreal(v2(6))
      pv2(3) = dimag(v2(6))
      pv2(4) = dimag(v2(5))

      T12 = ft(1,2) + ft(2,1)
      T13 = ft(1,3) + ft(3,1)
      T14 = ft(1,4) + ft(4,1)
      T23 = ft(2,3) + ft(3,2)
      T24 = ft(2,4) + ft(4,2)
      T34 = ft(3,4) + ft(4,3)

      V1V2 =  v1(1)*v2(1) -  v1(2)*v2(2) -  v1(3)*v2(3) -  v1(4)*v2(4)
      K1V2 = pv1(1)*v2(1) - pv1(2)*v2(2) - pv1(3)*v2(3) - pv1(4)*v2(4)
      K2V1 = pv2(1)*v1(1) - pv2(2)*v1(2) - pv2(3)*v1(3) - pv2(4)*v1(4)
c     new
      K1V1 = pv1(1)*v1(1) - pv1(2)*v1(2) - pv1(3)*v1(3) - pv1(4)*v1(4)
      K2V2 = pv2(1)*v2(1) - pv2(2)*v2(2) - pv2(3)*v2(3) - pv2(4)*v2(4)
c     new

      F = pv1(1)*pv2(1) - pv1(2)*pv2(2) - pv1(3)*pv2(3) - pv1(4)*pv2(4)
      if ( vmass.ne.rZero ) then
         F = F + vmass**2
      end if

      TKK   = cZero
      TVV   = cZero
      TK1V2 = cZero
      TK2V1 = cZero

      do i = 1,4
         dum   = ft(i,i)*pv1(i)
         TKK   = TKK   + dum*pv2(i)
         TK1V2 = TK1V2 + dum*v2(i)
         dum   = ft(i,i)*v1(i)
         TVV   = TVV   + dum*v2(i)
         TK2V1 = TK2V1 + dum*pv2(i)
      end do

      TKK   = rTwo*TKK
      TVV   = rTwo*TVV
      TK1V2 = rTwo*TK1V2
      TK2V1 = rTwo*TK2V1

      TKK = TKK - T12*(pv1(1)*pv2(2) + pv1(2)*pv2(1))
     &          - T13*(pv1(1)*pv2(3) + pv1(3)*pv2(1))
     &          - T14*(pv1(1)*pv2(4) + pv1(4)*pv2(1))
     &          + T23*(pv1(2)*pv2(3) + pv1(3)*pv2(2))
     &          + T24*(pv1(2)*pv2(4) + pv1(4)*pv2(2))
     &          + T34*(pv1(3)*pv2(4) + pv1(4)*pv2(3))

      TK1V2 = TK1V2 - T12*(pv1(1)*v2(2) + pv1(2)*v2(1))
     &              - T13*(pv1(1)*v2(3) + pv1(3)*v2(1))
     &              - T14*(pv1(1)*v2(4) + pv1(4)*v2(1))
     &              + T23*(pv1(2)*v2(3) + pv1(3)*v2(2))
     &              + T24*(pv1(2)*v2(4) + pv1(4)*v2(2))
     &              + T34*(pv1(3)*v2(4) + pv1(4)*v2(3))

      TVV = TVV - T12*(v1(1)*v2(2) + v1(2)*v2(1))
     &          - T13*(v1(1)*v2(3) + v1(3)*v2(1))
     &          - T14*(v1(1)*v2(4) + v1(4)*v2(1))
     &          + T23*(v1(2)*v2(3) + v1(3)*v2(2))
     &          + T24*(v1(2)*v2(4) + v1(4)*v2(2))
     &          + T34*(v1(3)*v2(4) + v1(4)*v2(3))

      TK2V1 = TK2V1 - T12*(v1(1)*pv2(2) + v1(2)*pv2(1))
     &              - T13*(v1(1)*pv2(3) + v1(3)*pv2(1))
     &              - T14*(v1(1)*pv2(4) + v1(4)*pv2(1))
     &              + T23*(v1(2)*pv2(3) + v1(3)*pv2(2))
     &              + T24*(v1(2)*pv2(4) + v1(4)*pv2(2))
     &              + T34*(v1(3)*pv2(4) + v1(4)*pv2(3))

      vertex =  (ft(1,1)-ft(2,2)-ft(3,3)-ft(4,4))*( K1V2*K2V1 - V1V2*F )
     &        + F*TVV + V1V2*TKK - K2V1*TK1V2 - K1V2*TK2V1

C      vertex = F*TVV + V1V2*TKK - K2V1*TK1V2 - K1V2*TK2V1

c     new, additonal gauge fixing term in Feyman gauge
      if ( vmass.eq.rZero ) then
         vertex = vertex 
     &+(ft(1,1)-ft(2,2)-ft(3,3)-ft(4,4))*(K1V1*K1V2+K2V1*K2V2+K1V1*K2V2)
     &-K1V1*TK1V2-K2V2*TK2V1	   
      endif	
c     new    

      vertex = vertex * gt

      return
      end
