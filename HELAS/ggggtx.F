      subroutine ggggtx(va,vb,vc,vd,tc,gc,gt , vertex)
c      
c This subroutine computes the portion of the amplitude of the five-point 
c coupling of a tensor boson with 4 massless color octet gauge bosons
c (gluons) corresponding to the color structure f^{a,b,e} f{c,d,e}. 
c
c To obtain the complete amplitude, this coupling must be called three
c times (once for each color structure) with the following permutations:
c     call ggggtx(va,vb,vc,vd,tc,gc,gt , vertex1)
c     call ggggtx(va,vc,vd,vb,tc,gc,gt , vertex2)
c     call ggggtx(va,vd,vb,vc,tc,gc,gt , vertex3)
c corresponding to
c	f^{a,b,e} f^{c,d,e}
c	f^{a,c,e} f^{d,b,e}
c	f^{a,d,e} f^{b,c,e}
c                                                                       
c input:                                                                
c       complex va(6)          : boson with adjoint color index a     va
c       complex vb(6)          : boson with adjoint color index b     vb
c       complex vc(6)          : boson with adjoint color index c     vc
c       complex vd(6)          : boson with adjoint color index d     vd
c       complex tc(18)         : input tensor                          T
c       real    gc             : coupling constant                    gs
c       complex gt             : coupling constant         gtv=-1/Lambda
c
c output:
c       complex vertex         : amplitude          gamma(va,vb,vc,vd,T)
c
c- by Q.Li - OCT. 2006
c
      implicit none
      double complex va(6), vb(6), vc(6), vd(6), tc(18), gt, vertex
      double precision gc

      double complex vab,vac,vad,vbc,vbd,vcd,ft(6,4),dvertx
      double complex T00, T12, T13, T14, T23, T24, T34	
      double complex TV24,TV23,TV14,TV13
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

      T00 = ft(1,1)-ft(2,2)-ft(3,3)-ft(4,4)
      T12 = ft(1,2) + ft(2,1)
      T13 = ft(1,3) + ft(3,1)
      T14 = ft(1,4) + ft(4,1)
      T23 = ft(2,3) + ft(3,2)
      T24 = ft(2,4) + ft(4,2)
      T34 = ft(3,4) + ft(4,3)

      TV14 = rtwo*(ft(1,1)*va(1)*vd(1)+ft(2,2)*va(2)*vd(2)
     &+ft(3,3)*va(3)*vd(3)+ft(4,4)*va(4)*vd(4))

      TV13 = rtwo*(ft(1,1)*va(1)*vc(1)+ft(2,2)*va(2)*vc(2)
     &+ft(3,3)*va(3)*vc(3)+ft(4,4)*va(4)*vc(4))

      TV24 = rtwo*(ft(1,1)*vb(1)*vd(1)+ft(2,2)*vb(2)*vd(2)
     &+ft(3,3)*vb(3)*vd(3)+ft(4,4)*vb(4)*vd(4))

      TV23 = rtwo*(ft(1,1)*vb(1)*vc(1)+ft(2,2)*vb(2)*vc(2)
     &+ft(3,3)*vb(3)*vc(3)+ft(4,4)*vb(4)*vc(4))

	
      TV14 = TV14- T12*(va(1)*vd(2) + va(2)*vd(1))
     &          - T13*(va(1)*vd(3) + va(3)*vd(1))
     &          - T14*(va(1)*vd(4) + va(4)*vd(1))
     &          + T23*(va(2)*vd(3) + va(3)*vd(2))
     &          + T24*(va(2)*vd(4) + va(4)*vd(2))
     &          + T34*(va(3)*vd(4) + va(4)*vd(3))
      
      TV13 = TV13 - T12*(va(1)*vc(2) + va(2)*vc(1))
     &          - T13*(va(1)*vc(3) + va(3)*vc(1))
     &          - T14*(va(1)*vc(4) + va(4)*vc(1))
     &          + T23*(va(2)*vc(3) + va(3)*vc(2))
     &          + T24*(va(2)*vc(4) + va(4)*vc(2))
     &          + T34*(va(3)*vc(4) + va(4)*vc(3))

      TV24 = TV24 - T12*(vb(1)*vd(2) + vb(2)*vd(1))
     &          - T13*(vb(1)*vd(3) + vb(3)*vd(1))
     &          - T14*(vb(1)*vd(4) + vb(4)*vd(1))
     &          + T23*(vb(2)*vd(3) + vb(3)*vd(2))
     &          + T24*(vb(2)*vd(4) + vb(4)*vd(2))
     &          + T34*(vb(3)*vd(4) + vb(4)*vd(3))

      TV23 = TV23 - T12*(vb(1)*vc(2) + vb(2)*vc(1))
     &          - T13*(vb(1)*vc(3) + vb(3)*vc(1))
     &          - T14*(vb(1)*vc(4) + vb(4)*vc(1))
     &          + T23*(vb(2)*vc(3) + vb(3)*vc(2))
     &          + T24*(vb(2)*vc(4) + vb(4)*vc(2))
     &          + T34*(vb(3)*vc(4) + vb(4)*vc(3))
     	

      vab = va(1)*vb(1)-va(2)*vb(2)-va(3)*vb(3)-va(4)*vb(4)
      vac = va(1)*vc(1)-va(2)*vc(2)-va(3)*vc(3)-va(4)*vc(4)
      vad = va(1)*vd(1)-va(2)*vd(2)-va(3)*vd(3)-va(4)*vd(4)
      vbc = vb(1)*vc(1)-vb(2)*vc(2)-vb(3)*vc(3)-vb(4)*vc(4)
      vbd = vb(1)*vd(1)-vb(2)*vd(2)-vb(3)*vd(3)-vb(4)*vd(4)
      vcd = vc(1)*vd(1)-vc(2)*vd(2)-vc(3)*vd(3)-vc(4)*vd(4)

      dvertx = -TV13*vbd-TV24*vac+TV23*vad+TV14*vbc 
     &+vbd*vac*T00-vad*vbc*T00

      vertex = -dvertx * gc*gc*gt

      return
      end
