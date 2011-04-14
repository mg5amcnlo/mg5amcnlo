      subroutine uvvvxx(va,vb,vc,gc,gt,tmass,twidth , uvvv)
c
c This subroutine computes an off-shell tensor current 
c from the four-point coupling of three gauge bosons and a tensor boson.
c
c input:
c       complex va(6)          : first  vector                        va
c       complex vb(6)          : second vector                        vb
c       complex vc(6)          : third  vector                        vc      
c       real gc                : coupling constant       gs (for gluons)
c       complex gt             : coupling constant         gtv=-1/Lambda
c       real tmass             : mass  of output tensor T
c       real twidth            : width of output tensor T 
c
c output:
c       complex uvvv(18)       : tensor current      j^mu^nu(T:va,vb,vc)
c
c- by Q.Li - OCT. 2006
c- Added massless tensor - P. de Aquino - Oct. 2009 
c     
      implicit none
      double complex va(6), vb(6), vc(6), gt, uvvv(18)
      double precision gc, tmass, twidth

      double complex yvvv(6,4)
      double precision MET(4,4)
      double complex d
      double complex V1V2,V1V3,V2V3,KTV1,KTV2,KTV3,
     &K12V3,K23V1,K31V2
      double precision KTK12,KTK31,KTK23 

      double precision pva(4), pvb(4), pvc(4),pt(4),
     &pt2,p31(4),p23(4),p12(4)

      integer i, j

      double complex cZero
      double precision rZero, r2, r3,r4
      parameter( rZero = 0.0d0, r2 = 2.0d0, r3=3.d0,r4=4.d0 )
      parameter( cZero = ( 0.0d0, 0.0d0 ) )


      yvvv(5,1) = va(5)+vb(5)+vc(5)
      yvvv(6,1) = va(6)+vb(6)+vc(6)

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

      pt(1) = dreal(yvvv(5,1))
      pt(2) = dreal(yvvv(6,1))
      pt(3) = dimag(yvvv(6,1))
      pt(4) = dimag(yvvv(5,1))
	
      pt2=pt(1)**2-pt(2)**2-pt(3)**2-pt(4)**2
      
      do i=1,4
         do j=1,4
            MET(i,j) = 0.0d0
         enddo 
      enddo
      MET(1,1) =  1.0d0
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
      
	
      V1V2 =  va(1)*vb(1) -  va(2)*vb(2) -  va(3)*vb(3) -  va(4)*vb(4)
      V1V3 =  va(1)*vc(1) -  va(2)*vc(2) -  va(3)*vc(3) -  va(4)*vc(4)
      V2V3 =  vc(1)*vb(1) -  vc(2)*vb(2) -  vc(3)*vb(3) -  vc(4)*vb(4)
      K31V2 = p31(1)*vb(1) - p31(2)*vb(2) - p31(3)*vb(3) - p31(4)*vb(4)
      K12V3 = p12(1)*vc(1) - p12(2)*vc(2) - p12(3)*vc(3) - p12(4)*vc(4)
      K23V1 = p23(1)*va(1) - p23(2)*va(2) - p23(3)*va(3) - p23(4)*va(4)
      
      KTV1 = pt(1)*va(1) - pt(2)*va(2) - pt(3)*va(3) - pt(4)*va(4)
      KTV2 = pt(1)*vb(1) - pt(2)*vb(2) - pt(3)*vb(3) - pt(4)*vb(4)
      KTV3 = pt(1)*vc(1) - pt(2)*vc(2) - pt(3)*vc(3) - pt(4)*vc(4)
      
      KTK12 =pt(1)*p12(1)-pt(2)*p12(2)-pt(3)*p12(3)-pt(4)*p12(4)
      KTK31 =pt(1)*p31(1)-pt(2)*p31(2)-pt(3)*p31(3)-pt(4)*p31(4)
      KTK23 =pt(1)*p23(1)-pt(2)*p23(2)-pt(3)*p23(3)-pt(4)*p23(4)
      
      if ( tmass.eq.rZero ) then
         d =  -gc/dcmplx( pt2, rZero )  
      
      do i=1,4
         do j=1,4

            yvvv(i,j) =  -(K12V3*V1V2*MET(i,j)) 
     &- K31V2*V1V3*MET(i,j) - K23V1*V2V3*MET(i,j)  
     &+ V2V3*p23(j)*va(i) + V2V3*p23(i)*va(j) 
     &+ V1V3*p31(j)*vb(i) + K12V3*va(j)*vb(i)  
     &+ V1V3*p31(i)*vb(j) + K12V3*va(i)*vb(j) 
     &+ V1V2*p12(j)*vc(i) + K31V2*va(j)*vc(i)  
     &+ K23V1*vb(j)*vc(i) + V1V2*p12(i)*vc(j) 
     &+ K31V2*va(i)*vc(j) + K23V1*vb(i)*vc(j)
	
            yvvv(i,j) = -yvvv(i,j)*d*gt
	
         enddo
      enddo

      else
         if ( tmass.gt.rZero ) then
            d =  -gc/dcmplx( pt2-tmass**2, tmass*twidth )

         do i=1,4
            do j=1,4

              yvvv(i,j) = (r2*K12V3*KTV1*KTV2*r2*MET(i,j))/(r3*tmass**2)
     &	 + (r2*K31V2*KTV1*KTV3*r2*MET(i,j))/(r3*tmass**2) + 
     &  (r2*K23V1*KTV2*KTV3*r2*MET(i,j))/(r3*tmass**2) 
     &- r2*K12V3*V1V2*MET(i,j) + 
     &  (r2*KTK12*KTV3*r2*V1V2*MET(i,j))/(r3*tmass**2) 
     &- (K12V3*pt2*r2*V1V2*MET(i,j))/(r3*tmass**2) - 
     &  r2*K31V2*V1V3*MET(i,j)
     & + (r2*KTK31*KTV2*r2*V1V3*MET(i,j))/(r3*tmass**2) - 
     &  (K31V2*pt2*r2*V1V3*MET(i,j))/(r3*tmass**2) 
     &- r2*K23V1*V2V3*MET(i,j) + 
     &  (r2*KTK23*KTV1*r2*V2V3*MET(i,j))/(r3*tmass**2)
     & - (K23V1*pt2*r2*V2V3*MET(i,j))/(r3*tmass**2) - 
     &  (r2*KTV3*V1V2*p12(j)*pt(i))/tmass**2 
     &- (r2*KTV1*V2V3*p23(j)*pt(i))/tmass**2 
     &- (r2*KTV2*V1V3*p31(j)*pt(i))/tmass**2 - 
     &  (r2*KTV3*V1V2*p12(i)*pt(j))/tmass**2
     & - (r2*KTV1*V2V3*p23(i)*pt(j))/tmass**2 
     &- (r2*KTV2*V1V3*p31(i)*pt(j))/tmass**2 + 
     &  (r4*K12V3*KTV1*KTV2*pt(i)*pt(j))/tmass**4
     & + (r4*K31V2*KTV1*KTV3*pt(i)*pt(j))/tmass**4 + 
     &  (r4*K23V1*KTV2*KTV3*pt(i)*pt(j))/tmass**4 
     &- (r2*K12V3*KTV1*KTV2*r2*pt(i)*pt(j))/(r3*tmass**4) - 
     &  (r2*K31V2*KTV1*KTV3*r2*pt(i)*pt(j))/(r3*tmass**4)
     & - (r2*K23V1*KTV2*KTV3*r2*pt(i)*pt(j))/(r3*tmass**4) + 
     &  (r4*KTK12*KTV3*V1V2*pt(i)*pt(j))/tmass**4 
     &- (r2*K12V3*pt2*V1V2*pt(i)*pt(j))/tmass**4 - 
     &  (r2*KTK12*KTV3*r2*V1V2*pt(i)*pt(j))/(r3*tmass**4)
     & + (K12V3*pt2*r2*V1V2*pt(i)*pt(j))/(r3*tmass**4) + 
     &  (r4*K12V3*V1V2*pt(i)*pt(j))/tmass**2 
     &+ (r4*KTK31*KTV2*V1V3*pt(i)*pt(j))/tmass**4 - 
     &  (r2*K31V2*pt2*V1V3*pt(i)*pt(j))/tmass**4
     & - (r2*KTK31*KTV2*r2*V1V3*pt(i)*pt(j))/(r3*tmass**4) + 
     &  (K31V2*pt2*r2*V1V3*pt(i)*pt(j))/(r3*tmass**4)
     & + (r4*K31V2*V1V3*pt(i)*pt(j))/tmass**2 + 
     &  (r4*KTK23*KTV1*V2V3*pt(i)*pt(j))/tmass**4
     & - (r2*K23V1*pt2*V2V3*pt(i)*pt(j))/tmass**4 - 
     &  (r2*KTK23*KTV1*r2*V2V3*pt(i)*pt(j))/(r3*tmass**4)
     & + (K23V1*pt2*r2*V2V3*pt(i)*pt(j))/(r3*tmass**4) + 
     &  (r4*K23V1*V2V3*pt(i)*pt(j))/tmass**2 
     &+ r2*V2V3*p23(j)*va(i) - (r2*K12V3*KTV2*pt(j)*va(i))/tmass**2 - 
     &  (r2*K31V2*KTV3*pt(j)*va(i))/tmass**2 
     &- (r2*KTK23*V2V3*pt(j)*va(i))/tmass**2 + 2*V2V3*p23(i)*va(j) - 
     &  (r2*K12V3*KTV2*pt(i)*va(j))/tmass**2 
     &- (r2*K31V2*KTV3*pt(i)*va(j))/tmass**2 
     &- (r2*KTK23*V2V3*pt(i)*va(j))/tmass**2 + 
     &  r2*V1V3*p31(j)*vb(i) - (r2*K12V3*KTV1*pt(j)*vb(i))/tmass**2 
     &- (r2*K23V1*KTV3*pt(j)*vb(i))/tmass**2 - 
     &  (r2*KTK31*V1V3*pt(j)*vb(i))/tmass**2 
     &+ r2*K12V3*va(j)*vb(i) + r2*V1V3*p31(i)*vb(j)
     & - (r2*K12V3*KTV1*pt(i)*vb(j))/tmass**2 - 
     &  (r2*K23V1*KTV3*pt(i)*vb(j))/tmass**2 
     &- (r2*KTK31*V1V3*pt(i)*vb(j))/tmass**2 
     &+ r2*K12V3*va(i)*vb(j) + r2*V1V2*p12(j)*vc(i) - 
     &  (r2*K31V2*KTV1*pt(j)*vc(i))/tmass**2 
     &- (r2*K23V1*KTV2*pt(j)*vc(i))/tmass**2 
     &- (r2*KTK12*V1V2*pt(j)*vc(i))/tmass**2 + 
     &  r2*K31V2*va(j)*vc(i) + r2*K23V1*vb(j)*vc(i) 
     &+ r2*V1V2*p12(i)*vc(j) - (r2*K31V2*KTV1*pt(i)*vc(j))/tmass**2 - 
     &  (r2*K23V1*KTV2*pt(i)*vc(j))/tmass**2
     & - (r2*KTK12*V1V2*pt(i)*vc(j))/tmass**2
     & + r2*K31V2*va(i)*vc(j) + r2*K23V1*vb(i)*vc(j)
	
              yvvv(i,j) = -yvvv(i,j)*d/2.0d0*gt
	
            enddo
         enddo
         else
            write(*,*) '’nvalid tensor mass'
         end if
      end if
      
      uvvv(1) = yvvv(1,1)
      uvvv(2) = yvvv(1,2)
      uvvv(3) = yvvv(1,3)
      uvvv(4) = yvvv(1,4)
      uvvv(5) = yvvv(2,1)
      uvvv(6) = yvvv(2,2)
      uvvv(7) = yvvv(2,3)
      uvvv(8) = yvvv(2,4)
      uvvv(9) = yvvv(3,1)
      uvvv(10) = yvvv(3,2)
      uvvv(11) = yvvv(3,3)
      uvvv(12) = yvvv(3,4)
      uvvv(13) = yvvv(4,1)
      uvvv(14) = yvvv(4,2)
      uvvv(15) = yvvv(4,3)
      uvvv(16) = yvvv(4,4)
      uvvv(17) = yvvv(5,1)
      uvvv(18) = yvvv(6,1)

      return
      end
