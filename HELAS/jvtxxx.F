      subroutine jvtxxx(vc,tc,gt,vmass,vwidth , jvt)
c
c This subroutine computes an off-shell vector current from 
c the coupling of two gauge bosons and a tensor boson.
c
c input:
c       complex vc(6)          : input vector                          v
c       complex tc(18)         : input tensor                          T
c       complex gt             : coupling constant         gtv=-1/Lambda
c       real    vmass          : mass  of output vector v'
c       real    vwidth         : width of output vector v'
c
c output:
c       complex jvt(6)         : vector current             j^mu(v':v,T)
c     
c- by Q.Li - OCT. 2006
c
      implicit none
      double complex vc(6), tc(18), gt, jvt(6)
      double precision vmass, vwidth

      double complex ft(6,4),TVM(4),TKM(4)
      double precision MET(4,4)
      double complex T12, T13, T14, T23, T24, T34, T00
      double complex K2V1,K1V1
      double complex TKK,TK2V1, dum
      double precision pv1(4), pv2(4), F, pp2

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
 
      jvt(5) = vc(5)+ft(5,1)
      jvt(6) = vc(6)+ft(6,1)

      pv1(1) = -dreal(vc(5))
      pv1(2) = -dreal(vc(6))
      pv1(3) = -dimag(vc(6))
      pv1(4) = -dimag(vc(5))

      pv2(1) = dreal(jvt(5))
      pv2(2) = dreal(jvt(6))
      pv2(3) = dimag(jvt(6))
      pv2(4) = dimag(jvt(5))

      do i=1,4
         do j=1,4
            MET(i,j) = 0.0d0
         enddo 
      enddo
      MET(1,1) =  1.0d0
      MET(2,2) = -1.0d0
      MET(3,3) = -1.0d0
      MET(4,4) = -1.0d0
      
      T00 = ft(1,1)-ft(2,2)-ft(3,3)-ft(4,4)
      T12 = ft(1,2) + ft(2,1)
      T13 = ft(1,3) + ft(3,1)
      T14 = ft(1,4) + ft(4,1)
      T23 = ft(2,3) + ft(3,2)
      T24 = ft(2,4) + ft(4,2)
      T34 = ft(3,4) + ft(4,3)

      K2V1 = pv2(1)*vc(1) - pv2(2)*vc(2) - pv2(3)*vc(3) - pv2(4)*vc(4)
      K1V1 = pv1(1)*vc(1) - pv1(2)*vc(2) - pv1(3)*vc(3) - pv1(4)*vc(4)
      F = pv1(1)*pv2(1) - pv1(2)*pv2(2) - pv1(3)*pv2(3) - pv1(4)*pv2(4)
      pp2 = pv2(1)**2 - pv2(2)**2 - pv2(3)**2 - pv2(4)**2

      TKK   = cZero
      TK2V1 = cZero

      do i = 1,4
         dum   = ft(i,i)*pv2(i)
         TKK   = TKK   + dum*pv1(i)
         dum   = ft(i,i)*vc(i)
         TK2V1 = TK2V1 + dum*pv2(i)
      end do
      
      TKK   = rTwo*TKK
      TK2V1 = rTwo*TK2V1
      
      TKK = TKK - T12*(pv1(1)*pv2(2) + pv1(2)*pv2(1))
     &          - T13*(pv1(1)*pv2(3) + pv1(3)*pv2(1))
     &          - T14*(pv1(1)*pv2(4) + pv1(4)*pv2(1))
     &          + T23*(pv1(2)*pv2(3) + pv1(3)*pv2(2))
     &          + T24*(pv1(2)*pv2(4) + pv1(4)*pv2(2))
     &          + T34*(pv1(3)*pv2(4) + pv1(4)*pv2(3))

      TK2V1 = TK2V1 - T12*(vc(1)*pv2(2) + vc(2)*pv2(1))
     &              - T13*(vc(1)*pv2(3) + vc(3)*pv2(1))
     &              - T14*(vc(1)*pv2(4) + vc(4)*pv2(1))
     &              + T23*(vc(2)*pv2(3) + vc(3)*pv2(2))
     &              + T24*(vc(2)*pv2(4) + vc(4)*pv2(2))
     &              + T34*(vc(3)*pv2(4) + vc(4)*pv2(3))


      do j=1,4

         TVM(j) =
     &MET(j,1)*(ft(1,1)*vc(1)-ft(2,1)*vc(2)
     &-ft(3,1)*vc(3)-ft(4,1)*vc(4))
     &-MET(j,2)*(ft(1,2)*vc(1)-ft(2,2)*vc(2)
     &-ft(3,2)*vc(3)-ft(4,2)*vc(4))
     &-MET(j,3)*(ft(1,3)*vc(1)-ft(2,3)*vc(2)
     &-ft(3,3)*vc(3)-ft(4,3)*vc(4))
     &-MET(j,4)*(ft(1,4)*vc(1)-ft(2,4)*vc(2)
     &-ft(3,4)*vc(3)-ft(4,4)*vc(4))
     &+
     &MET(j,1)*(ft(1,1)*vc(1)-ft(1,2)*vc(2)
     &-ft(1,3)*vc(3)-ft(1,4)*vc(4))
     &-MET(j,2)*(ft(2,1)*vc(1)-ft(2,2)*vc(2)
     &-ft(2,3)*vc(3)-ft(2,4)*vc(4))
     &-MET(j,3)*(ft(3,1)*vc(1)-ft(3,2)*vc(2)
     &-ft(3,3)*vc(3)-ft(3,4)*vc(4))
     &-MET(j,4)*(ft(4,1)*vc(1)-ft(4,2)*vc(2)
     &-ft(4,3)*vc(3)-ft(4,4)*vc(4))

         TKM(j) =
     &MET(j,1)*(ft(1,1)*pv1(1)-ft(2,1)*pv1(2)
     &-ft(3,1)*pv1(3)-ft(4,1)*pv1(4))
     &-MET(j,2)*(ft(1,2)*pv1(1)-ft(2,2)*pv1(2)
     &-ft(3,2)*pv1(3)-ft(4,2)*pv1(4))
     &-MET(j,3)*(ft(1,3)*pv1(1)-ft(2,3)*pv1(2)
     &-ft(3,3)*pv1(3)-ft(4,3)*pv1(4))
     &-MET(j,4)*(ft(1,4)*pv1(1)-ft(2,4)*pv1(2)
     &-ft(3,4)*pv1(3)-ft(4,4)*pv1(4))
     &+
     &MET(j,1)*(ft(1,1)*pv1(1)-ft(1,2)*pv1(2)
     &-ft(1,3)*pv1(3)-ft(1,4)*pv1(4))
     &-MET(j,2)*(ft(2,1)*pv1(1)-ft(2,2)*pv1(2)
     &-ft(2,3)*pv1(3)-ft(2,4)*pv1(4))
     &-MET(j,3)*(ft(3,1)*pv1(1)-ft(3,2)*pv1(2)
     &-ft(3,3)*pv1(3)-ft(3,4)*pv1(4))
     &-MET(j,4)*(ft(4,1)*pv1(1)-ft(4,2)*pv1(2)
     &-ft(4,3)*pv1(3)-ft(4,4)*pv1(4))
     
      enddo

      if ( vmass.ne.rZero ) then

         do i=1,4

            jvt(i) = -(vmass**2+F)*T00*vc(i)
     &+T00*K2V1*(pv1(i)+(1.0d0+F/vmass**2)*pv2(i)
     &-F/vmass**2*pv2(i))
     &+(vmass**2+F)*TVM(i)
     &-TK2V1*pv1(i)
     &-TK2V1*pv2(i)*(1.0d0+F/vmass**2)
     &+TKK*vc(i)
     &-K2V1*TKM(i)
     &+F/vmass**2*TK2V1*pv2(i)
	 
            jvt(i)=jvt(i)*gt/dcmplx(pp2-vmass**2, vmass*vwidth )

         enddo

      else
	
         do i=1,4

            jvt(i) = -F*T00*vc(i)
     &+K1V1*T00*(pv1(i)+pv2(i))
     &+T00*K2V1*(pv1(i)+pv2(i))
     &+F*TVM(i)
     &-TK2V1*(pv1(i)+pv2(i))
     &+TKK*vc(i)
     &-(K2V1+K1V1)*TKM(i)
  
            jvt(i)=jvt(i)*gt/dcmplx(pp2, 0.0d0)
            
         enddo

         
      endif

      return
      end
