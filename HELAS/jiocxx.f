c ----------------------------------------------------------------------
c
      subroutine jiocxx(fic,foc,gc,vmass,vwidth , jioc)
c
c This subroutine computes an off-shell vector current from an external 
c antifermion pair. The vector boson propagator is given in Feynman     
c gauge for a massless vector and in unitary gauge for a massive vector.
c                                                                       
c input:                                                                
c       complex fic(6)         : flow-in  antifermion              |fic>
c       complex foc(6)         : flow-out antifermion              <foc|
c       complex gc(2)          : coupling constants                  gvf
c       real    vmass          : mass  of output vector v               
c       real    vwidth         : width of output vector v               
c                                                                       
c output:                                                               
c       complex jioc(6)        : vector current        j^mu(<foc|v|fic>)
c
      implicit none
      double complex fic(6),foc(6),gc(2),jioc(6),c0,c1,c2,c3,cs,d
      double precision q(0:3),vmass,vwidth,q2,vm2,dd
      double complex cm2 ! mass**2- I Gamma mass (Benj&Claude)

      double precision rZero, rOne
      parameter( rZero = 0.0d0, rOne = 1.0d0 )
      double complex cImag, cZero
      parameter( cImag = ( 0.0d0, 1.0d0 ), cZero = ( 0.0d0, 0.0d0 ) )
c
      jioc(5) = foc(5)-fic(5)
      jioc(6) = foc(6)-fic(6)

      q(0) = dble( jioc(5))
      q(1) = dble( jioc(6))
      q(2) = dimag(jioc(6))
      q(3) = dimag(jioc(5))
      q2 = q(0)**2-(q(1)**2+q(2)**2+q(3)**2)
      vm2 = vmass**2

      if ( vmass.ne.rZero ) then

         d = -rOne/dcmplx( q2-vm2, vmass*vwidth )
c  for the running width, use below instead of the above d.
c         d = -rOne/dcmplx( q2-vm2, max(vwidth*q2/vmass,rZero) )

         if ( gc(2).ne.cZero ) then
            c0=  gc(2)*( foc(3)*fic(1)+foc(4)*fic(2))
     &          +gc(1)*( foc(1)*fic(3)+foc(2)*fic(4))
            c1= -gc(2)*( foc(3)*fic(2)+foc(4)*fic(1))
     &          +gc(1)*( foc(1)*fic(4)+foc(2)*fic(3))
            c2=( gc(2)*( foc(3)*fic(2)-foc(4)*fic(1)) 
     &          +gc(1)*(-foc(1)*fic(4)+foc(2)*fic(3)))*cImag
            c3=  gc(2)*(-foc(3)*fic(1)+foc(4)*fic(2))
     &          +gc(1)*( foc(1)*fic(3)-foc(2)*fic(4))
         else
            d = d*gc(1)
            c0 =  foc(1)*fic(3)+foc(2)*fic(4)
            c1 =  foc(1)*fic(4)+foc(2)*fic(3)
            c2 =(-foc(1)*fic(4)+foc(2)*fic(3))*cImag
            c3 =  foc(1)*fic(3)-foc(2)*fic(4)
         end if
	 
c     Fabio's implementation of the fixed width (Benj&Claude)
         cm2=dcmplx( vm2, -vmass*vwidth )
c         cs = (q(0)*c0-q(1)*c1-q(2)*c2-q(3)*c3)/vm2
         cs = (q(0)*c0-q(1)*c1-q(2)*c2-q(3)*c3)/cm2
	 
         jioc(1) = (c0-cs*q(0))*d
         jioc(2) = (c1-cs*q(1))*d
         jioc(3) = (c2-cs*q(2))*d
         jioc(4) = (c3-cs*q(3))*d

      else

         d = dcmplx( -rOne/q2, rZero )
         if ( gc(2).ne.cZero ) then
            jioc(1) = ( gc(2)*( foc(3)*fic(1)+foc(4)*fic(2))
     &                 +gc(1)*( foc(1)*fic(3)+foc(2)*fic(4)) )*d
            jioc(2) = (-gc(2)*( foc(3)*fic(2)+foc(4)*fic(1))
     &                 +gc(1)*( foc(1)*fic(4)+foc(2)*fic(3)) )*d
            jioc(3) = ( gc(2)*( foc(3)*fic(2)-foc(4)*fic(1))
     &                 +gc(1)*(-foc(1)*fic(4)+foc(2)*fic(3)) )
     &                *d*cImag
            jioc(4) = ( gc(2)*(-foc(3)*fic(1)+foc(4)*fic(2))
     &                 +gc(1)*( foc(1)*fic(3)-foc(2)*fic(4)) )*d
         else
            d = d*gc(1)
            jioc(1) = ( foc(1)*fic(3)+foc(2)*fic(4))*d
            jioc(2) = ( foc(1)*fic(4)+foc(2)*fic(3))*d
            jioc(3) = (-foc(1)*fic(4)+foc(2)*fic(3))*d*cImag
            jioc(4) = ( foc(1)*fic(3)-foc(2)*fic(4))*d
         end if

      end if
c
      return
      end
