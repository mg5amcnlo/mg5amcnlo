      subroutine jgggtx(va,vb,vc,tc,gc,gt , jgggt)
c
c This subroutine computes an off-shell vector boson current from
c the 4 gluons-tensor boson coupling, corresponding 
c to the color structure f^{x,a,e} f{b,c,e}. 
c
c To obtain the complete amplitude, this subroutine must be called three
c times (once for each color structure) with the following permutations:
c     call jgggtx(va,vb,vc,tc,gc,gt , jgggt1)
c     call jgggtx(vb,vc,va,tc,gc,gt , jgggt2)
c     call jgggtx(vc,va,vb,tc,gc,gt , jgggt3)
c corresponding to 
c	f^{x,a,e} f^{b,c,e}
c	f^{x,b,e} f^{c,a,e}
c	f^{x,c,e} f^{a,b,e}
c
c input:
c       complex va(6)          : boson with adjoint color index a     va
c       complex vb(6)          : boson with adjoint color index b     vb
c       complex vc(6)          : boson with adjoint color index c     vc
c       complex tc(18)         : input tensor                          T
c       real    gc             : coupling constant                    gs
c       complex gt             : coupling constant         gtv=-1/Lambda
c
c output:
c       complex jgggt(6)       : gluon current        j^mu(v:va,vb,vc,T)
c
c- by Q.Li - JAN. 2008
c     
      implicit none
      double complex va(6), vb(6), vc(6), tc(18), gt,  jgggt(6)
      double precision gc

      double complex ft(6,4)
      double complex TVBVC, TVBVD, TVDM(4), TVCM(4)
      double complex T00,T12,T13,T14,T23,T24,T34,EBEC,EBED

      integer a,b,i,j

      double precision pb(4),pc(4),pd(4),pT(4),px(4)

      double precision MET(4,4)
      double complex cZero, d
      double precision rZero, rTwo,px2,r3
      parameter( rZero = 0.0d0, rTwo = 2.0d0, r3=3.0d0)
      parameter(cZero=(0.0d0,0.0d0))


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

      jgggt(5)=va(5)+vb(5)+vc(5)+ft(5,1)
      jgggt(6)=ft(6,1)+va(6)+vb(6)+vc(6)

      px(1) = dreal(jgggt(5))
      px(2) = dreal(jgggt(6))
      px(3) = dimag(jgggt(6))
      px(4) = dimag(jgggt(5))

      pb(1) = dreal(va(5))
      pb(2) = dreal(va(6))
      pb(3) = dimag(va(6))
      pb(4) = dimag(va(5))

      pc(1) = dreal(vb(5))
      pc(2) = dreal(vb(6))
      pc(3) = dimag(vb(6))
      pc(4) = dimag(vb(5))

      pd(1) = dreal(vc(5))
      pd(2) = dreal(vc(6))
      pd(3) = dimag(vc(6))
      pd(4) = dimag(vc(5))

     
      do i=1,4
         do j=1,4
            MET(i,j)=0.0d0
         enddo 
      enddo

      MET(1,1)=1.0d0
      MET(2,2)=-1.0d0
      MET(3,3)=-1.0d0
      MET(4,4)=-1.0d0

      px2 = px(1)**2-px(2)**2-px(3)**2-px(4)**2
      EBEC = va(1)*vb(1)-va(2)*vb(2)-va(3)*vb(3)-va(4)*vb(4)
      EBED = va(1)*vc(1)-va(2)*vc(2)-va(3)*vc(3)-va(4)*vc(4)
      
      d = 1.0d0/dcmplx( px2, 0.0d0 )
      
      T00 = ft(1,1)-ft(2,2)-ft(3,3)-ft(4,4)
      T12 = ft(1,2) + ft(2,1)
      T13 = ft(1,3) + ft(3,1)
      T14 = ft(1,4) + ft(4,1)
      T23 = ft(2,3) + ft(3,2)
      T24 = ft(2,4) + ft(4,2)
      T34 = ft(3,4) + ft(4,3)

      TVBVC =2.0d0*(ft(1,1)*va(1)*vb(1)
     &+ft(2,2)*va(2)*vb(2)+ft(3,3)*va(3)*vb(3)
     &+ft(4,4)*va(4)*vb(4)
     &)
     & - T12*(va(1)*vb(2) + va(2)*vb(1))
     &          - T13*(va(1)*vb(3) + va(3)*vb(1))
     &          - T14*(va(1)*vb(4) + va(4)*vb(1))
     &          + T23*(va(2)*vb(3) + va(3)*vb(2))
     &          + T24*(va(2)*vb(4) + va(4)*vb(2))
     &          + T34*(va(3)*vb(4) + va(4)*vb(3))

      TVBVD =2.0d0*(ft(1,1)*va(1)*vc(1)
     &+ft(2,2)*va(2)*vc(2)+ft(3,3)*va(3)*vc(3)
     &+ft(4,4)*va(4)*vc(4)
     &)
     & - T12*(va(1)*vc(2) + va(2)*vc(1))
     &          - T13*(va(1)*vc(3) + va(3)*vc(1))
     &          - T14*(va(1)*vc(4) + va(4)*vc(1))
     &          + T23*(va(2)*vc(3) + va(3)*vc(2))
     &          + T24*(va(2)*vc(4) + va(4)*vc(2))
     &          + T34*(va(3)*vc(4) + va(4)*vc(3))

      do j=1,4
         TVDM(j)=
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
	
	TVCM(j)=
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

      enddo
	    
      do a=1,4
	
         jgggt(a)=vc(a)*TVBVC-vb(a)*TVBVD
     &        -T00*vc(a)*EBEC+T00*vb(a)*EBED
     &        +EBEC*TVDM(a)-EBED*TVCM(a)
	 
         jgggt(a)=-jgggt(a)*d*gc*gc*gt
         
      enddo 
 
      return
      end
