      subroutine vvvtxx(va,vb,vc,tc,gc,gt , vertex)
c
c This subroutine computes an amplitude of the four-point coupling of
c three gauge bosons and a tensor boson.
c
c input:
c       complex va(6)          : first  vector                        va
c       complex vb(6)          : second vector                        vb
c       complex vc(6)          : third  vector                        vc
c       complex tc(18)         : input  tensor                         T
c       real    gc             : coupling constant       gs (for gluons)
c       complex gt             : coupling constant         gtv=-1/Lambda
c
c output:
c       complex vertex         : amplitude             gamma(va,vb,vc,T)
c
c- by Q.Li - OCT. 2006
c     
      implicit none
      double complex va(6), vb(6), vc(6),  tc(18), gt, vertex
      double precision gc
 
      double complex ft(6,4)
      double complex T00, T12, T13, T14, T23, T24, T34
      double complex V1V2,V1V3,V2V3, K1V2, K1V3, K2V1
     &,K2V3,K3V1,K3V2

      double complex TV12,TV13,TV23,TKV1,TKV2,TKV3, dum
      double precision pva(4), pvb(4), pvc(4),p31(4),p23(4),p12(4)

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

      pva(1) = dreal(va(5))
      pva(2) = dreal(va(6))
      pva(3) = dimag(va(6))
      pva(4) = dimag(va(5))

      pvb(1) = dreal(vb(5))
      pvb(2) = dreal(vb(6))
      pvb(3) = dimag(vb(6))
      pvb(4) = dimag(vb(5))

      pvc(1) = dreal(vc(5))
      pvc(2) = dreal(vc(6))
      pvc(3) = dimag(vc(6))
      pvc(4) = dimag(vc(5))

      p31(1) = pvc(1)-pva(1)
      p31(2) = pvc(2)-pva(2)
      p31(3) = pvc(3)-pva(3)
      p31(4) = pvc(4)-pva(4)
      
      p12(1) = pva(1)-pvb(1)
      p12(2) = pva(2)-pvb(2)
      p12(3) = pva(3)-pvb(3)
      p12(4) = pva(4)-pvb(4)
      
      p23(1) = pvb(1)-pvc(1)
      p23(2) = pvb(2)-pvc(2)
      p23(3) = pvb(3)-pvc(3)
      p23(4) = pvb(4)-pvc(4)
      
      T00 = ft(1,1)-ft(2,2)-ft(3,3)-ft(4,4)
      T12 = ft(1,2) + ft(2,1)
      T13 = ft(1,3) + ft(3,1)
      T14 = ft(1,4) + ft(4,1)
      T23 = ft(2,3) + ft(3,2)
      T24 = ft(2,4) + ft(4,2)
      T34 = ft(3,4) + ft(4,3)

      V1V2 =  va(1)*vb(1) -  va(2)*vb(2) -  va(3)*vb(3) -  va(4)*vb(4)
      V1V3 =  va(1)*vc(1) -  va(2)*vc(2) -  va(3)*vc(3) -  va(4)*vc(4)
      V2V3 =  vc(1)*vb(1) -  vc(2)*vb(2) -  vc(3)*vb(3) -  vc(4)*vb(4)
      K1V2 = pva(1)*vb(1) - pva(2)*vb(2) - pva(3)*vb(3) - pva(4)*vb(4)
      K1V3 = pva(1)*vc(1) - pva(2)*vc(2) - pva(3)*vc(3) - pva(4)*vc(4)
      K2V1 = pvb(1)*va(1) - pvb(2)*va(2) - pvb(3)*va(3) - pvb(4)*va(4)
      K2V3 = pvb(1)*vc(1) - pvb(2)*vc(2) - pvb(3)*vc(3) - pvb(4)*vc(4)
      K3V1 = pvc(1)*va(1) - pvc(2)*va(2) - pvc(3)*va(3) - pvc(4)*va(4)
      K3V2 = pvc(1)*vb(1) - pvc(2)*vb(2) - pvc(3)*vb(3) - pvc(4)*vb(4)


      TV12   = cZero
      TV13   = cZero
      TV23   = cZero
      TKV1   = cZero
      TKV2   = cZero
      TKV3   = cZero

      TV12 = rtwo*(ft(1,1)*va(1)*vb(1)+ft(2,2)*va(2)*vb(2)
     &+ft(3,3)*va(3)*vb(3)+ft(4,4)*va(4)*vb(4))

      TV13 = rtwo*(ft(1,1)*va(1)*vc(1)+ft(2,2)*va(2)*vc(2)
     &+ft(3,3)*va(3)*vc(3)+ft(4,4)*va(4)*vc(4))

      TV23 = rtwo*(ft(1,1)*vb(1)*vc(1)+ft(2,2)*vb(2)*vc(2)
     &+ft(3,3)*vb(3)*vc(3)+ft(4,4)*vb(4)*vc(4))

      TKV1 = rtwo*(ft(1,1)*p23(1)*va(1)+ft(2,2)*p23(2)*va(2)
     &+ft(3,3)*p23(3)*va(3)+ft(4,4)*p23(4)*va(4))

      TKV2 = rtwo*(ft(1,1)*p31(1)*vb(1)+ft(2,2)*p31(2)*vb(2)
     &+ft(3,3)*p31(3)*vb(3)+ft(4,4)*p31(4)*vb(4))
     
      TKV3 = rtwo*(ft(1,1)*p12(1)*vc(1)+ft(2,2)*p12(2)*vc(2)
     &+ft(3,3)*p12(3)*vc(3)+ft(4,4)*p12(4)*vc(4))	


      TV12 = TV12 - T12*(va(1)*vb(2) + va(2)*vb(1))
     &            - T13*(va(1)*vb(3) + va(3)*vb(1))
     &            - T14*(va(1)*vb(4) + va(4)*vb(1))
     &            + T23*(va(2)*vb(3) + va(3)*vb(2))
     &            + T24*(va(2)*vb(4) + va(4)*vb(2))
     &            + T34*(va(3)*vb(4) + va(4)*vb(3))

      TV13 = TV13 - T12*(va(1)*vc(2) + va(2)*vc(1))
     &            - T13*(va(1)*vc(3) + va(3)*vc(1))
     &            - T14*(va(1)*vc(4) + va(4)*vc(1))
     &            + T23*(va(2)*vc(3) + va(3)*vc(2))
     &            + T24*(va(2)*vc(4) + va(4)*vc(2))
     &            + T34*(va(3)*vc(4) + va(4)*vc(3))

      TV23 = TV23 - T12*(vb(1)*vc(2) + vb(2)*vc(1))
     &            - T13*(vb(1)*vc(3) + vb(3)*vc(1))
     &            - T14*(vb(1)*vc(4) + vb(4)*vc(1))
     &            + T23*(vb(2)*vc(3) + vb(3)*vc(2))
     &            + T24*(vb(2)*vc(4) + vb(4)*vc(2))
     &            + T34*(vb(3)*vc(4) + vb(4)*vc(3))


      TKV1 = TKV1 - T12*(p23(1)*va(2) + p23(2)*va(1))
     &            - T13*(p23(1)*va(3) + p23(3)*va(1))
     &            - T14*(p23(1)*va(4) + p23(4)*va(1))
     &            + T23*(p23(2)*va(3) + p23(3)*va(2))
     &            + T24*(p23(2)*va(4) + p23(4)*va(2))
     &            + T34*(p23(3)*va(4) + p23(4)*va(3))

      TKV2 = TKV2 - T12*(p31(1)*vb(2) + p31(2)*vb(1))
     &            - T13*(p31(1)*vb(3) + p31(3)*vb(1))
     &            - T14*(p31(1)*vb(4) + p31(4)*vb(1))
     &            + T23*(p31(2)*vb(3) + p31(3)*vb(2))
     &            + T24*(p31(2)*vb(4) + p31(4)*vb(2))
     &            + T34*(p31(3)*vb(4) + p31(4)*vb(3))

      TKV3 = TKV3 - T12*(p12(1)*vc(2) + p12(2)*vc(1))
     &            - T13*(p12(1)*vc(3) + p12(3)*vc(1))
     &            - T14*(p12(1)*vc(4) + p12(4)*vc(1))
     &            + T23*(p12(2)*vc(3) + p12(3)*vc(2))
     &            + T24*(p12(2)*vc(4) + p12(4)*vc(2))
     &            + T34*(p12(3)*vc(4) + p12(4)*vc(3))


      vertex = TKV3*V1V2-T00*K1V3*V1V2+T00*K2V3*V1V2+TKV2*V1V3
     &+TV23*(K2V1-K3V1)+TKV1*V2V3-T00*K2V1*V2V3+T00*K3V1*V2V3
     &-TV13*(K1V2-K3V2)+T00*K1V2*V1V3-T00*V1V3*K3V2+TV12*(K1V3-K2V3)

      vertex= -vertex * gc*gt

      return
      end
