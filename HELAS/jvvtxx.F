      subroutine jvvtxx(va,vb,tc,gc,gt,vmass,vwidth , jvvt)
c
c This subroutine computes an off-shell vector boson wavefunction from
c the four-point coupling of three gauge bosons and a tensor boson.
c
c input:
c       complex va(6)          : first  vector                        va
c       complex vb(6)          : second vector                        vb
c       complex tc(18)         : input  tensor                         T
c       real    gc             : coupling constant       gs (for gluons)
c       complex gt             : coupling constant         gtv=-1/Lambda
c       real    vmass          : mass  of output vector v
c       real    vwidth         : width of output vector v

c
c output:
c       complex jvvt(6)        : vector current          j^mu(v:va,vb,T)
c
c- by Q.Li - OCT. 2006
c     
      implicit none
      double complex va(6), vb(6), tc(18), gt, jvvt(6)
      double precision gc,vmass,vwidth

      double complex ft(6,4)
      double complex TV1M(4),TV2M(4),TK12M(4)
      double precision MET(4,4)
      double complex d,T00, T12, T13, T14, T23, T24, T34
      double complex V1V2,K1V2, K2V1
     &,K3V1,K3V2
      double precision K1K2

      double complex TV12,TKV1,TKV2,
     &TK3V1,TK3V2,TK312,dum
      double precision pva(4), pvb(4), pvc(4),
     &pv32,p31(4),p23(4),p12(4),K1K3,K2K3

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


      jvvt(5) = va(5)+vb(5)+ft(5,1)
      jvvt(6) = va(6)+vb(6)+ft(6,1)

      pva(1) = dreal(va(5))
      pva(2) = dreal(va(6))
      pva(3) = dimag(va(6))
      pva(4) = dimag(va(5))

      pvb(1) = dreal(vb(5))
      pvb(2) = dreal(vb(6))
      pvb(3) = dimag(vb(6))
      pvb(4) = dimag(vb(5))

      pvc(1) = -dreal(jvvt(5))
      pvc(2) = -dreal(jvvt(6))
      pvc(3) = -dimag(jvvt(6))
      pvc(4) = -dimag(jvvt(5))
	
      pv32=pvc(1)**2-pvc(2)**2-pvc(3)**2-pvc(4)**2
      if ( vmass.gt.rZero ) then
         d =  gc/dcmplx( pv32-vmass**2, vmass*vwidth )
      else
         d =  gc/dcmplx( pv32, rZero )
      end if

      do i=1,4
         do j=1,4
            MET(i,j) = 0.0d0
         enddo 
      enddo
      MET(1,1) = 1.0d0
      MET(2,2) = -1.0d0
      MET(3,3) = -1.0d0
      MET(4,4) = -1.0d0
      
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
      K1V2 = pva(1)*vb(1) - pva(2)*vb(2) - pva(3)*vb(3) - pva(4)*vb(4)
      K2V1 = pvb(1)*va(1) - pvb(2)*va(2) - pvb(3)*va(3) - pvb(4)*va(4)
      K3V1 = pvc(1)*va(1) - pvc(2)*va(2) - pvc(3)*va(3) - pvc(4)*va(4)
      K3V2 = pvc(1)*vb(1) - pvc(2)*vb(2) - pvc(3)*vb(3) - pvc(4)*vb(4)
      
      K1K3 = pva(1)*pvc(1)-pva(2)*pvc(2)-pva(3)*pvc(3)-pva(4)*pvc(4)
      K2K3 = pvb(1)*pvc(1)-pvb(2)*pvc(2)-pvb(3)*pvc(3)-pvb(4)*pvc(4)
      
      TV12 = rtwo*(ft(1,1)*va(1)*vb(1)+ft(2,2)*va(2)*vb(2)
     &+ft(3,3)*va(3)*vb(3)+ft(4,4)*va(4)*vb(4))


      TKV1 = rtwo*(ft(1,1)*p23(1)*va(1)+ft(2,2)*p23(2)*va(2)
     &+ft(3,3)*p23(3)*va(3)+ft(4,4)*p23(4)*va(4))

      TKV2 = rtwo*(ft(1,1)*p31(1)*vb(1)+ft(2,2)*p31(2)*vb(2)
     &+ft(3,3)*p31(3)*vb(3)+ft(4,4)*p31(4)*vb(4))

     
      TK3V1 = rtwo*(ft(1,1)*pvc(1)*va(1)+ft(2,2)*pvc(2)*va(2)
     &+ft(3,3)*pvc(3)*va(3)+ft(4,4)*pvc(4)*va(4))	

      TK3V2 = rtwo*(ft(1,1)*pvc(1)*vb(1)+ft(2,2)*pvc(2)*vb(2)
     &+ft(3,3)*pvc(3)*vb(3)+ft(4,4)*pvc(4)*vb(4))	


      TK312 = rtwo*(ft(1,1)*pvc(1)*p12(1)+ft(2,2)*pvc(2)*p12(2)
     &+ft(3,3)*pvc(3)*p12(3)+ft(4,4)*pvc(4)*p12(4))	


      TV12 = TV12 - T12*(va(1)*vb(2) + va(2)*vb(1))
     &          - T13*(va(1)*vb(3) + va(3)*vb(1))
     &          - T14*(va(1)*vb(4) + va(4)*vb(1))
     &          + T23*(va(2)*vb(3) + va(3)*vb(2))
     &          + T24*(va(2)*vb(4) + va(4)*vb(2))
     &          + T34*(va(3)*vb(4) + va(4)*vb(3))


      TKV1 = TKV1 - T12*(p23(1)*va(2) + p23(2)*va(1))
     &              - T13*(p23(1)*va(3) + p23(3)*va(1))
     &              - T14*(p23(1)*va(4) + p23(4)*va(1))
     &              + T23*(p23(2)*va(3) + p23(3)*va(2))
     &              + T24*(p23(2)*va(4) + p23(4)*va(2))
     &              + T34*(p23(3)*va(4) + p23(4)*va(3))

      TKV2 = TKV2 - T12*(p31(1)*vb(2) + p31(2)*vb(1))
     &              - T13*(p31(1)*vb(3) + p31(3)*vb(1))
     &              - T14*(p31(1)*vb(4) + p31(4)*vb(1))
     &              + T23*(p31(2)*vb(3) + p31(3)*vb(2))
     &              + T24*(p31(2)*vb(4) + p31(4)*vb(2))
     &              + T34*(p31(3)*vb(4) + p31(4)*vb(3))

      TK3V1 = TK3V1 - T12*(pvc(1)*va(2) + pvc(2)*va(1))
     &              - T13*(pvc(1)*va(3) + pvc(3)*va(1))
     &              - T14*(pvc(1)*va(4) + pvc(4)*va(1))
     &              + T23*(pvc(2)*va(3) + pvc(3)*va(2))
     &              + T24*(pvc(2)*va(4) + pvc(4)*va(2))
     &              + T34*(pvc(3)*va(4) + pvc(4)*va(3))

      TK3V2 = TK3V2 - T12*(pvc(1)*vb(2) + pvc(2)*vb(1))
     &              - T13*(pvc(1)*vb(3) + pvc(3)*vb(1))
     &              - T14*(pvc(1)*vb(4) + pvc(4)*vb(1))
     &              + T23*(pvc(2)*vb(3) + pvc(3)*vb(2))
     &              + T24*(pvc(2)*vb(4) + pvc(4)*vb(2))
     &              + T34*(pvc(3)*vb(4) + pvc(4)*vb(3))

      TK312 = TK312 - T12*(pvc(1)*p12(2) + pvc(2)*p12(1))
     &              - T13*(pvc(1)*p12(3) + pvc(3)*p12(1))
     &              - T14*(pvc(1)*p12(4) + pvc(4)*p12(1))
     &              + T23*(pvc(2)*p12(3) + pvc(3)*p12(2))
     &              + T24*(pvc(2)*p12(4) + pvc(4)*p12(2))
     &              + T34*(pvc(3)*p12(4) + pvc(4)*p12(3))

      do j=1,4

         TV1M(j) =
     &MET(j,1)*(ft(1,1)*va(1)-ft(2,1)*va(2)
     &-ft(3,1)*va(3)-ft(4,1)*va(4))
     &-MET(j,2)*(ft(1,2)*va(1)-ft(2,2)*va(2)
     &-ft(3,2)*va(3)-ft(4,2)*va(4))
     &-MET(j,3)*(ft(1,3)*va(1)-ft(2,3)*va(2)
     &-ft(3,3)*va(3)-ft(4,3)*va(4))
     &-MET(j,4)*(ft(1,4)*va(1)-ft(2,4)*va(2)
     &-ft(3,4)*va(3)-ft(4,4)*va(4))
     &+
     &MET(j,1)*(ft(1,1)*va(1)-ft(1,2)*va(2)
     &-ft(1,3)*va(3)-ft(1,4)*va(4))
     &-MET(j,2)*(ft(2,1)*va(1)-ft(2,2)*va(2)
     &-ft(2,3)*va(3)-ft(2,4)*va(4))
     &-MET(j,3)*(ft(3,1)*va(1)-ft(3,2)*va(2)
     &-ft(3,3)*va(3)-ft(3,4)*va(4))
     &-MET(j,4)*(ft(4,1)*va(1)-ft(4,2)*va(2)
     &-ft(4,3)*va(3)-ft(4,4)*va(4))

         TV2M(j) =
     &MET(j,1)*(ft(1,1)*vb(1)-ft(2,1)*vb(2)
     &-ft(3,1)*vb(3)-ft(4,1)*vb(4))
     &-MET(j,2)*(ft(1,2)*vb(1)-ft(2,2)*vb(2)
     &-ft(3,2)*vb(3)-ft(4,2)*vb(4))
     &-MET(j,3)*(ft(1,3)*vb(1)-ft(2,3)*vb(2)
     &-ft(3,3)*vb(3)-ft(4,3)*vb(4))
     &-MET(j,4)*(ft(1,4)*vb(1)-ft(2,4)*vb(2)
     &-ft(3,4)*vb(3)-ft(4,4)*vb(4))
     &+
     &MET(j,1)*(ft(1,1)*vb(1)-ft(1,2)*vb(2)
     &-ft(1,3)*vb(3)-ft(1,4)*vb(4))
     &-MET(j,2)*(ft(2,1)*vb(1)-ft(2,2)*vb(2)
     &-ft(2,3)*vb(3)-ft(2,4)*vb(4))
     &-MET(j,3)*(ft(3,1)*vb(1)-ft(3,2)*vb(2)
     &-ft(3,3)*vb(3)-ft(3,4)*vb(4))
     &-MET(j,4)*(ft(4,1)*vb(1)-ft(4,2)*vb(2)
     &-ft(4,3)*vb(3)-ft(4,4)*vb(4))


         TK12M(j) =
     &MET(j,1)*(ft(1,1)*p12(1)-ft(2,1)*p12(2)
     &-ft(3,1)*p12(3)-ft(4,1)*p12(4))
     &-MET(j,2)*(ft(1,2)*p12(1)-ft(2,2)*p12(2)
     &-ft(3,2)*p12(3)-ft(4,2)*p12(4))
     &-MET(j,3)*(ft(1,3)*p12(1)-ft(2,3)*p12(2)
     &-ft(3,3)*p12(3)-ft(4,3)*p12(4))
     &-MET(j,4)*(ft(1,4)*p12(1)-ft(2,4)*p12(2)
     &-ft(3,4)*p12(3)-ft(4,4)*p12(4))
     &+
     &MET(j,1)*(ft(1,1)*p12(1)-ft(1,2)*p12(2)
     &-ft(1,3)*p12(3)-ft(1,4)*p12(4))
     &-MET(j,2)*(ft(2,1)*p12(1)-ft(2,2)*p12(2)
     &-ft(2,3)*p12(3)-ft(2,4)*p12(4))
     &-MET(j,3)*(ft(3,1)*p12(1)-ft(3,2)*p12(2)
     &-ft(3,3)*p12(3)-ft(3,4)*p12(4))
     &-MET(j,4)*(ft(4,1)*p12(1)-ft(4,2)*p12(2)
     &-ft(4,3)*p12(3)-ft(4,4)*p12(4))
     
      enddo
        
      do i=1,4
         jvvt(i) = TV12*p12(i)+TKV1*vb(i)+TKV2*va(i)
     &+(-p12(i)*V1V2-vb(i)*K2V1+vb(i)*K3V1
     &  +va(i)*K1V2-va(i)*K3V2)*T00
     &-K1V2*TV1M(i)+K3V2*TV1M(i)
     &+V1V2*TK12M(i)
     &+K2V1*TV2M(i)-K3V1*TV2M(i)
      enddo 

      if ( vmass.gt.rZero ) then
         do i=1,4
            jvvt(i) = jvvt(i)
     &+(K1V2*pvc(i)
     &-K3V2*TKV1*pvc(i)-K3V2*TK3V1*pvc(i)
     &-K1K3*TV12*pvc(i)+K2K3*TV12*pvc(i)
     &-V1V2*TK312*pvc(i)
     &-K2V1*TK3V2*pvc(i)+K3V1*TK3V2*pvc(i)
     &-K3V1*TKV2*pvc(i))/vmass**2
     &+(-pvc(i)*K1V2*K3V1+pvc(i)*K2V1*K3V2
     &+pvc(i)*V1V2*K1K3-pvc(i)*V1V2*K2K3
     &)/vmass**2*T00
         enddo
           
      endif	
      
      do i=1,4
         jvvt(i) = -jvvt(i) * d*gt
      enddo
      
      return
      end
