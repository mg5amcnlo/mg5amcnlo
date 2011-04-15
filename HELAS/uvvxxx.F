      subroutine uvvxxx(v1,v2,gt,vmass,tmass,twidth , uvv)
c
c This subroutine computes an off-shell tensor current from 
c the two gauge bosons-tensor boson coupling.
c
c input:
c       complex v1(3)          : first  vector                        v1
c       complex v2(3)          : second vector                        v2
c       complex gt             : coupling constant         gtv=-1/Lambda
c       real    vmass          : vector boson mass                   m_v
c       real    tmass          : mass  of output tensor T
c       real    twidth         : width of output tensor T
c
c output:
c       complex uvv(18)        : tensor current         j^mu^nu(T:v1,v2)
c
c- by Q.Li - OCT. 2006
c- Added massless tensor - P. de Aquino - Oct. 2009 
c
      implicit none
      double complex v1(6), v2(6), gt, uvv(18)
      double precision vmass, tmass, twidth

      double complex yvv(6,4)
      double complex KTE1,KTE2,K2E1,K1E2,K1E1,K2E2,E1E2
      integer i,j
      double precision pv1(4), pv2(4), pT(4)
      double precision MET(4,4)
      double complex cZero, d
      double precision rZero, rTwo
      double precision K1K2,KT2,K1KT,K2KT
      parameter( rZero = 0.0d0, rTwo = 2.0d0 )
      parameter( cZero = ( 0.0d0, 0.0d0 ) )

      
      yvv(5,1) = v1(5)+v2(5)
      yvv(6,1) = v1(6)+v2(6)
      
      pv1(1) = dreal(v1(5))
      pv1(2) = dreal(v1(6))
      pv1(3) = dimag(v1(6))
      pv1(4) = dimag(v1(5))
      
      pv2(1) = dreal(v2(5))
      pv2(2) = dreal(v2(6))
      pv2(3) = dimag(v2(6))
      pv2(4) = dimag(v2(5))
      
      pT(1) = dreal(yvv(5,1))
      pT(2) = dreal(yvv(6,1))
      pT(3) = dimag(yvv(6,1))
      pT(4) = dimag(yvv(5,1))
      
      do i=1,4
         do j=1,4
            MET(i,j) = 0.0d0
         enddo 
      enddo
      
      MET(1,1) =  1.0d0
      MET(2,2) = -1.0d0
      MET(3,3) = -1.0d0
      MET(4,4) = -1.0d0
      
      K1K2 = pv1(1)*pv2(1)-pv1(2)*pv2(2)-pv1(3)*pv2(3)-pv1(4)*pv2(4)
      K1KT = pv1(1)*pT(1)-pv1(2)*pT(2)-pv1(3)*pT(3)-pv1(4)*pT(4)
      K2KT = pT(1)*pv2(1)-pT(2)*pv2(2)-pT(3)*pv2(3)-pT(4)*pv2(4)
      KT2 = pT(1)**2-pT(2)**2-pT(3)**2-pT(4)**2
      
      KTE1 = pT(1)*v1(1)-pT(2)*v1(2)-pT(3)*v1(3)-pT(4)*v1(4)
      KTE2 = pT(1)*v2(1)-pT(2)*v2(2)-pT(3)*v2(3)-pT(4)*v2(4)
      K1E1 = pv1(1)*v1(1)-pv1(2)*v1(2)-pv1(3)*v1(3)-pv1(4)*v1(4)
      K1E2 = pv1(1)*v2(1)-pv1(2)*v2(2)-pv1(3)*v2(3)-pv1(4)*v2(4)
      K2E1 = pv2(1)*v1(1)-pv2(2)*v1(2)-pv2(3)*v1(3)-pv2(4)*v1(4)
      K2E2 = pv2(1)*v2(1)-pv2(2)*v2(2)-pv2(3)*v2(3)-pv2(4)*v2(4)
      E1E2 = v2(1)*v1(1)-v2(2)*v1(2)-v2(3)*v1(3)-v2(4)*v1(4)
      
      if ( tmass.eq.rZero ) then
         d = - gt/dcmplx( KT2, rZero )
    
         do i = 1,4
            do j=1,4
               yvv(i,j) = -(E1E2*K1K2*MET(i,j)) + K1E2*K2E1*MET(i,j)  
     &+  E1E2*pv1(j)*pv2(i) + E1E2*pv1(i)*pv2(j)  
     &-  K1E2*pv2(j)*v1(i) - K1E2*pv2(i)*v1(j)  
     &-  K2E1*pv1(j)*v2(i) + k1k2*v1(j)*v2(i)  
     &-  K2E1*pv1(i)*v2(j) + k1k2*v1(i)*v2(j) 
               
            
               if ( vmass.ne.rZero ) then
                  yvv(i,j) = yvv(i,j)
     &              + vmass**2*v1(j)*v2(i) + vmass**2*v1(i)*v2(j)
               else
c     gauge fixing term for zero mass photon/gluon
                  yvv(i,j) =  yvv(i,j)
     &             -(K1E1*K2E2*MET(i,j)) - K2E2*pv2(j)*v1(i) - 
     &  K2E2*pv2(i)*v1(j) - K1E1*pv1(j)*v2(i) - 
     &  K1E1*pv1(i)*v2(j)
               endif
               yvv(i,j) = yvv(i,j)*d
            end do
         enddo
      
      else if ( tmass.gt.rZero ) then
            d = - gt/dcmplx( KT2-tmass**2, tmass*twidth )
    
            do i = 1,4
               do j=1,4
                  yvv(i,j) = 2.0d0*K1K2*(v1(i)*v2(j)+v1(j)*v2(i))
     &-2.0d0*K1K2*KTE2/tmass**2*(PT(i)*v1(j)+PT(j)*v1(i))
     &-2.0d0*K1E2*(pv2(i)*v1(j)+pv2(j)*v1(i))
     &+2.0d0*K1E2*K2KT/tmass**2*(PT(i)*v1(j)+PT(j)*v1(i))
     &-2.0d0/3.0d0*E1E2*K1K2*MET(i,j)
     &+8.0d0/3.0d0*K1K2*E1E2/tmass**2*PT(i)*PT(j)
     &+2.0d0*E1E2*(pv1(i)*pv2(j)+pv1(j)*pv2(i)) 
     &-2.0d0*K1K2*KTE1/tmass**2*(PT(i)*v2(j)+PT(j)*v2(i))
     &-2.0d0*K2E1*(pv1(i)*v2(j)+pv1(j)*v2(i))
     &+4.0d0*K1K2*KTE1*KTE2/3.0d0/tmass**2*MET(i,j)
     &+8.0d0*K1K2*KTE1*KTE2/3.0d0/tmass**4*PT(i)*PT(j)
     &+2.0d0*K2E1*KTE2/tmass**2*(pv1(i)*PT(j)+pv1(j)*PT(i))
     &+2.0d0*KTE1*K1E2/tmass**2*(pv2(i)*PT(j)+pv2(j)*PT(i))
     &+2.0d0*K2E1*K1E2*MET(i,j)
     &-4.0d0*K2E1*K1E2/tmass**2*PT(i)*PT(j)
     &-2.0d0/3.0d0/tmass**2*KT2*E1E2*K1K2*MET(i,j)
     &-4.0d0/3.0d0/tmass**4*KT2*E1E2*K1K2*PT(i)*PT(j)
     &+2.0d0/3.0d0/tmass**2*K2E1*K1E2*KT2*MET(i,j)
     &+4.0d0/3.0d0/tmass**4*K2E1*K1E2*KT2*PT(i)*PT(j)
     &-2.0d0/tmass**2*E1E2*K1KT*(pv2(i)*PT(j)+pv2(j)*PT(i))
     &+2.0d0/tmass**2*K2E1*K1KT*(PT(i)*v2(j)+PT(j)*v2(i))
     &-4.0d0/3.0d0/tmass**2*K2E1*KTE2*K1KT*MET(i,j)
     &-8.0d0/3.0d0/tmass**4*K2E1*KTE2*K1KT*PT(i)*PT(j)
     &-2.0d0/tmass**2*E1E2*K2KT*(PT(i)*pv1(j)+PT(j)*pv1(i))
     &-4.0d0/3.0d0/tmass**2*KTE1*K1E2*K2KT*MET(i,j)
     &-8.0d0/3.0d0/tmass**4*KTE1*K1E2*K2KT*PT(i)*PT(j)
     &+4.0d0/3.0d0/tmass**2*E1E2*K2KT*K1KT*MET(i,j)
     &+8.0d0/3.0d0/tmass**4*E1E2*K2KT*K1KT*PT(i)*PT(j)
     &-4.0d0/3.0d0*E1E2*K1K2*MET(i,j)
     &+4.0d0/3.0d0/tmass**2*E1E2*K1K2
     &*PT(i)*PT(j)
            
                  if ( vmass.ne.rZero ) then
                     yvv(i,j) = 
     &                 yvv(i,j)+2*vmass**2*(v1(i)*v2(j)+v1(j)*v2(i))
     &-2.0d0/3.0d0*vmass**2*E1E2*MET(i,j)
     &+8.0d0/3.0d0/tmass**2*vmass**2*E1E2*PT(i)*PT(j)
     &-2.0d0/tmass**2*vmass**2*KTE1*(PT(i)*v2(j)+PT(j)*v2(i)) 
     &-2.0d0/tmass**2*vmass**2*KTE2*(PT(i)*v1(j)+PT(j)*v1(i)) 
     &+4.0d0/3.0d0/tmass**2*vmass**2*KTE1*KTE2*MET(i,j)
     &+8.0d0/3.0d0/tmass**4*vmass**2*KTE1*KTE2*PT(i)*PT(j)
     &-2.0d0/3.0d0/tmass**2*vmass**2*KT2*E1E2*MET(i,j)
     &-4.0d0/3.0d0/tmass**4*vmass**2*KT2*E1E2*PT(i)*PT(j)
                  else
c     gauge fixing term for zero mass photon/gluon
                     yvv(i,j) = 
     &                yvv(i,j)-2.0d0*K1E1*(pv1(i)*v2(j)+pv1(j)*v2(i))
     &+2.0d0/tmass**2*KTE2*K1E1*(PT(i)*pv1(j)+PT(j)*pv1(i))
     &+2.0d0/3.0d0*K1E2*K1E1*MET(i,j)
     &-8.0d0/3.0d0/tmass**2*K1E2*K1E1*PT(i)*PT(j)
     &-2.0d0/3.0d0*K2E2*K1E1*MET(i,j)
     &-4.0d0/3.0d0/tmass**2*K2E2*K1E1*PT(i)*PT(j)
     &+2.0d0/3.0d0/tmass**2*K1E2*KT2*K1E1*MET(i,j)
     &+4.0d0/3.0d0/tmass**4*K1E2*KT2*K1E1*PT(i)*PT(j)
     &+2.0d0/3.0d0/tmass**2*K2E2*KT2*K1E1*MET(i,j)
     &+4.0d0/3.0d0/tmass**4*K2E2*KT2*K1E1*PT(i)*PT(j)
     &+2.0d0/tmass**2*K1KT*K1E1*(PT(i)*v2(j)+PT(j)*v2(i))
     &-4.0d0/3.0d0/tmass**2*K1KT*KTE2*K1E1*MET(i,j)
     &-8.0d0/3.0d0/tmass**4*K1KT*KTE2*K1E1*PT(i)*PT(j)
     &-2.0d0*K2E2*(pv2(i)*v1(j)+pv2(j)*v1(i))
     &+2.0d0/tmass**2*K2E2*KTE1*(pv2(i)*PT(j)+pv2(j)*PT(i))
     &+2.0d0/3.0d0*K2E1*K2E2*MET(i,j)
     &-8.0d0/3.0d0/tmass**2*K2E1*K2E2*PT(i)*PT(j)
     &+2.0d0/3.0d0/tmass**2*KT2*K2E2*K2E1*MET(i,j)
     &+4.0d0/3.0d0/tmass**4*KT2*K2E2*K2E1*PT(i)*PT(j)
     &+2.0d0/tmass**2*K2E2*K2KT*(PT(i)*v1(j)+PT(j)*v1(i))
     &-4.0d0/3.0d0/tmass**2*K2E2*K2KT*KTE1*MET(i,j)
     &-8.0d0/3.0d0/tmass**4*K2E2*K2KT*KTE1
     &*PT(i)*PT(j)
                  endif

                  yvv(i,j) = yvv(i,j)*d/2.0d0

               end do
            enddo
         else
            write(*,*) 'invalid tensor mass'
            stop
      end if
      
      uvv(1) = yvv(1,1)
      uvv(2) = yvv(1,2)
      uvv(3) = yvv(1,3)
      uvv(4) = yvv(1,4)
      uvv(5) = yvv(2,1)
      uvv(6) = yvv(2,2)
      uvv(7) = yvv(2,3)
      uvv(8) = yvv(2,4)
      uvv(9) = yvv(3,1)
      uvv(10) = yvv(3,2)
      uvv(11) = yvv(3,3)
      uvv(12) = yvv(3,4)
      uvv(13) = yvv(4,1)
      uvv(14) = yvv(4,2)
      uvv(15) = yvv(4,3)
      uvv(16) = yvv(4,4)
      uvv(17) = yvv(5,1)
      uvv(18) = yvv(6,1)

      return
      end
