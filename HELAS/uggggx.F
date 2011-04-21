      subroutine uggggx(va,vb,vc,vd,gc,gt,tmass,twidth , ugggg)
c
c This subroutine computes an off-shell tensor boson 
c current from the 4 gluons-tensor boson coupling, 
c corresponding to the color structure f^{a,b,e} f{c,d,e}. 
c
c To obtain the complete amplitude, this subroutine must be called three
c times (once for each color structure) with the following permutations:
c     call uggggx(ga,gb,gc,gd,g,gt,tmass,twidth , ugggg1)
c     call uggggx(ga,gc,gd,gb,g,gt,tmass,twidth , ugggg2)
c     call uggggx(ga,gd,gb,gv,g,gt,tmass,twidth , ugggg3)
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
c       real    gc             : coupling constant                    gs
c       complex gt             : coupling constant         gtv=-1/Lambda
c       real    tmass          : mass  of output tensor T
c       real    twidth         : width of output tensor T
c
c output:
c       complex ugggg(18)      : tensor current   j^mu^nu(T:va,vb,vc,vd)
c     
c- by Q.Li - JAN. 2008
c
      implicit none
      double complex va(6), vb(6), vc(6), vd(6), gt, ugggg(18)
      double precision gc, tmass, twidth

      double complex tc(6,4)
      double complex E1E3,E1E4,E2E3,E2E4,E1PT,E2PT,E3PT,E4PT

      integer a,b,i,j

      double precision pa(4), pb(4),pc(4),pd(4),pT(4)
      double precision MET(4,4)
      double complex cZero, d
      double precision rZero, rTwo,PT2,r3
      parameter( rZero = 0.0d0, rTwo = 2.0d0, r3=3.0d0)
      parameter(cZero=(0.0d0,0.0d0))

      
      tc(5,1) = va(5)+vb(5)+vc(5)+vd(5)
      tc(6,1) = va(6)+vb(6)+vc(6)+vd(6)

      pa(1) = dreal(va(5))
      pa(2) = dreal(va(6))
      pa(3) = dimag(va(6))
      pa(4) = dimag(va(5))

      pb(1) = dreal(vb(5))
      pb(2) = dreal(vb(6))
      pb(3) = dimag(vb(6))
      pb(4) = dimag(vb(5))

      pc(1) = dreal(vc(5))
      pc(2) = dreal(vc(6))
      pc(3) = dimag(vc(6))
      pc(4) = dimag(vc(5))

      pd(1) = dreal(vd(5))
      pd(2) = dreal(vd(6))
      pd(3) = dimag(vd(6))
      pd(4) = dimag(vd(5))

     
      pT(1) = dreal(tc(5,1))
      pT(2) = dreal(tc(6,1))
      pT(3) = dimag(tc(6,1))
      pT(4) = dimag(tc(5,1))

      do i=1,4
         do j=1,4
            MET(i,j) = 0.0d0
         enddo 
      enddo
      MET(1,1) =  1.0d0
      MET(2,2) = -1.0d0
      MET(3,3) = -1.0d0
      MET(4,4) = -1.0d0

      PT2 = pT(1)**2-pT(2)**2-pT(3)**2-pT(4)**2
      
      E1PT = pT(1)*va(1)-pT(2)*va(2)-pT(3)*va(3)-pT(4)*va(4)
      E2PT = pT(1)*vb(1)-pT(2)*vb(2)-pT(3)*vb(3)-pT(4)*vb(4)
      E3PT = pT(1)*vc(1)-pT(2)*vc(2)-pT(3)*vc(3)-pT(4)*vc(4)
      E4PT = pT(1)*vd(1)-pT(2)*vd(2)-pT(3)*vd(3)-pT(4)*vd(4)
      
      E1E3 = va(1)*vc(1)-va(2)*vc(2)-va(3)*vc(3)-va(4)*vc(4)
      E1E4 = va(1)*vd(1)-va(2)*vd(2)-va(3)*vd(3)-va(4)*vd(4)
      E2E3 = vb(1)*vc(1)-vb(2)*vc(2)-vb(3)*vc(3)-vb(4)*vc(4)
      E2E4 = vb(1)*vd(1)-vb(2)*vd(2)-vb(3)*vd(3)-vb(4)*vd(4)
   
      if ( tmass.gt.rZero ) then
         d = - 1.0d0/dcmplx( PT2-tmass**2, tmass*twidth )
      else
         d = - 1.0d0/dcmplx( PT2, rZero )
      end if
      
	    
      do a = 1,4
         do b=1,4
            
            tc(a,b) = -2*E2E4*va(b)*vc(a) + 2*E1E4*vb(b)*vc(a)
     &- 2*E2E4*va(a)*vc(b) + 2*E1E4*vb(a)*vc(b) + 2*E2E3*va(b)*vd(a)
     & - 2*E1E3*vb(b)*vd(a) + 2*E2E3*va(a)*vd(b) - 
     &  2*E1E3*vb(a)*vd(b) - 2*E1E4*E2E3*MET(a,b) + 2*E1E3*E2E4*MET(a,b) 
     &- (2*E1PT*E2E4*E3PT*rtwo*MET(a,b))/(r3*tmass**2) + 
     &  (2*E1E4*E2PT*E3PT*rtwo*MET(a,b))/(r3*tmass**2) 
     &+ (2*E1PT*E2E3*E4PT*rtwo*MET(a,b))/(r3*tmass**2)
     & - (2*E1E3*E2PT*E4PT*rtwo*MET(a,b))/(r3*tmass**2) - 
     &  (E1E4*E2E3*PT2*rtwo*MET(a,b))/(r3*tmass**2) 
     &+ (E1E3*E2E4*PT2*rtwo*MET(a,b))/(r3*tmass**2)
     & + (2*E2E4*E3PT*va(b)*PT(a))/tmass**2 - 
     &  (2*E2E3*E4PT*va(b)*PT(a))/tmass**2
     & - (2*E1E4*E3PT*vb(b)*PT(a))/tmass**2
     & + (2*E1E3*E4PT*vb(b)*PT(a))/tmass**2
     & + (2*E1PT*E2E4*vc(b)*PT(a))/tmass**2 - 
     &  (2*E1E4*E2PT*vc(b)*PT(a))/tmass**2
     & - (2*E1PT*E2E3*vd(b)*PT(a))/tmass**2
     & + (2*E1E3*E2PT*vd(b)*PT(a))/tmass**2
     & + (2*E2E4*E3PT*va(a)*PT(b))/tmass**2 - 
     &  (2*E2E3*E4PT*va(a)*PT(b))/tmass**2 
     &- (2*E1E4*E3PT*vb(a)*PT(b))/tmass**2
     & + (2*E1E3*E4PT*vb(a)*PT(b))/tmass**2 
     &+ (2*E1PT*E2E4*vc(a)*PT(b))/tmass**2 - 
     &  (2*E1E4*E2PT*vc(a)*PT(b))/tmass**2 
     &- (2*E1PT*E2E3*vd(a)*PT(b))/tmass**2 
     &+ (2*E1E3*E2PT*vd(a)*PT(b))/tmass**2
     & - (4*E1PT*E2E4*E3PT*PT(a)*PT(b))/tmass**4 + 
     &  (4*E1E4*E2PT*E3PT*PT(a)*PT(b))/tmass**4 
     &+ (4*E1PT*E2E3*E4PT*PT(a)*PT(b))/tmass**4
     & - (4*E1E3*E2PT*E4PT*PT(a)*PT(b))/tmass**4 - 
     &  (2*E1E4*E2E3*PT2*PT(a)*PT(b))/tmass**4
     & + (2*E1E3*E2E4*PT2*PT(a)*PT(b))/tmass**4
     & + (2*E1PT*E2E4*E3PT*rtwo*PT(a)*PT(b))/(r3*tmass**4) - 
     &  (2*E1E4*E2PT*E3PT*rtwo*PT(a)*PT(b))/(r3*tmass**4)
     & - (2*E1PT*E2E3*E4PT*rtwo*PT(a)*PT(b))/(r3*tmass**4)
     & + (2*E1E3*E2PT*E4PT*rtwo*PT(a)*PT(b))/(r3*tmass**4) + 
     &  (E1E4*E2E3*PT2*rtwo*PT(a)*PT(b))/(r3*tmass**4)
     & - (E1E3*E2E4*PT2*rtwo*PT(a)*PT(b))/(r3*tmass**4) 
     &+ (4*E1E4*E2E3*PT(a)*PT(b))/tmass**2 - 
     &  (4*E1E3*E2E4*PT(a)*PT(b))/tmass**2

            tc(a,b) = -tc(a,b)*d/2.0d0*gc*gc*gt
c     2.0 factor from propagator convention

         enddo
      enddo

      ugggg(1) = tc(1,1)
      ugggg(2) = tc(1,2)
      ugggg(3) = tc(1,3)
      ugggg(4) = tc(1,4)
      ugggg(5) = tc(2,1)
      ugggg(6) = tc(2,2)
      ugggg(7) = tc(2,3)
      ugggg(8) = tc(2,4)
      ugggg(9) = tc(3,1)
      ugggg(10) = tc(3,2)
      ugggg(11) = tc(3,3)
      ugggg(12) = tc(3,4)
      ugggg(13) = tc(4,1)
      ugggg(14) = tc(4,2)
      ugggg(15) = tc(4,3)
      ugggg(16) = tc(4,4)
      ugggg(17) = tc(5,1)
      ugggg(18) = tc(6,1)

      return
      end
