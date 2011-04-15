      subroutine iovtxx(fi,fo,vc,tc,gc,gt , vertex)
c
c This subroutine computes an amplitude of the four-point coupling of
c a vector boson, two fermions and a tensor boson.
c
c input:
c       complex fi(6)          : flow-in  fermion                   |fi>
c       complex fo(6)          : flow-out fermion                   <fo|
c       complex vc(6)          : input    vector                       v
c       complex tc(18)         : input    tensor                       T
c       complex gc(2)          : coupling constants                  gvf
c       complex gt             : coupling constant      gtfv=-1/Lambda/2
c
c output:
c       complex vertex         : amplitude                   <fo|v,T|fi>
c     
c- by Q.Li - OCT. 2006
c
      implicit none
      double complex fi(6), fo(6), vc(6), tc(18), gc(2), gt, vertex

      double complex ft(6,4)
      double complex T00,T12, T13, T14, T23, T24, T34

      double precision rZero, r2
      parameter( rZero = 0.0d0, r2 = 2.0d0 )
      double complex cone
      parameter( cone = ( 0.0d0, 1.0d0 ) )

      
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


      vertex =-gt*(fi(4)*gc(2)*(fo(1)*(T12*vc(1) - cone*T13*vc(1)
     &	 - r2*ft(2,2)*vc(2) + cone*r2*ft(3,3)*vc(3) - 
     &          T23*(-(cone*vc(2)) + vc(3)) + r2*T00*(-vc(2) 
     &+ cone*vc(3)) - T24*vc(4) + cone*T34*vc(4)) + 
     &       fo(2)*(-(r2*ft(1,1)*vc(1)) + T12*vc(2) + T24*vc(2)
     & + T13*vc(3) + T34*vc(3) + r2*ft(4,4)*vc(4) + 
     &          T14*(-vc(1) + vc(4)) + r2*T00*(vc(1) + vc(4)))) + 
     &    fi(1)*gc(1)*(fo(4)*(-(T12*vc(1)) - cone*T13*vc(1) 
     &+ r2*ft(2,2)*vc(2) - T23*(-(cone*vc(2)) - vc(3)) + 
     &          cone*r2*ft(3,3)*vc(3) + r2*T00*(vc(2) 
     &+ cone*vc(3)) + T24*vc(4) + cone*T34*vc(4)) + 
     &       fo(3)*(-(r2*ft(1,1)*vc(1)) + T12*vc(2) 
     &+ T24*vc(2) + T13*vc(3) + T34*vc(3) + r2*ft(4,4)*vc(4) + 
     &          T14*(-vc(1) + vc(4)) + r2*T00*(vc(1) + vc(4)))) + 
     &    fi(3)*gc(2)*(fo(2)*(T12*vc(1) + cone*T13*vc(1)
     & - r2*ft(2,2)*vc(2) - cone*r2*ft(3,3)*vc(3)
     & - T23*(cone*vc(2) + vc(3)) + 
     &          r2*T00*(-vc(2) - cone*vc(3)) - T24*vc(4) 
     &- cone*T34*vc(4)) + 
     &       fo(1)*(-(r2*ft(1,1)*vc(1)) + T12*vc(2) 
     &- T24*vc(2) + T13*vc(3) - T34*vc(3) + r2*T00*(vc(1) - vc(4)) - 
     &          r2*ft(4,4)*vc(4) + T14*(vc(1) + vc(4)))) + 
     &    fi(2)*gc(1)*(fo(3)*(-(T12*vc(1)) + cone*T13*vc(1)
     & + r2*ft(2,2)*vc(2) - T23*(cone*vc(2) - vc(3)) - 
     &          cone*r2*ft(3,3)*vc(3) + r2*T00*(vc(2) 
     &- cone*vc(3)) + T24*vc(4) - cone*T34*vc(4)) + 
     &       fo(4)*(-(r2*ft(1,1)*vc(1)) + T12*vc(2) 
     &- T24*vc(2) + T13*vc(3) - T34*vc(3) + r2*T00*(vc(1) - vc(4)) - 
     &          r2*ft(4,4)*vc(4) + T14*(vc(1) + vc(4)))))

      return
      end
