c
c ----------------------------------------------------------------------
c
      subroutine jiogld(fi,fo,gc,vmass,vwidth,idecay , jio)
c
c This subroutine computes an off-shell vector current for the NLSP-
c Goldstino vertex from an external fermion pair. The vector boson 
c propagator is given in feynman gauge for a massless vector and in 
c unitary gauge for a massive vector. The h.c. of the NLSP decay is
c handled via the input parameter idecay (picks out correct
c Goldstino momentum).
c
c input:
c       complex fi(6)          : flow-in  fermion                   |fi>
c       complex fo(6)          : flow-out fermion                   <fo|
c       complex gc(2)          : coupling constants                  gvf
c       real    vmass          : mass  of output vector v
c       real    vwidth         : width of output vector v
c       integer idecay         :  1 for NLSP decay to Goldstino
c                              : -1 for Goldstino to NLSP (h.c. of above)
c
c output:
c       complex jio(6)         : vector current          j^mu(<fo|v|fi>)
c
      implicit none
      double complex  fi(6), fo(6), gc(2), jio(6), c0, c1, c2, c3, cs
      double complex  d, dum, p14p, p14m, p23p, p23m
      double precision  q(0:3), vmass, vwidth, q2, vm2, dd
      double precision  pG(1:4), pdotpG
      integer idecay

      double precision rZero, rOne
      parameter( rZero = 0.0d0, rOne = 1.0d0 )
      double complex ci, cZero
      parameter( ci = ( 0.0d0, 1.0d0 ), cZero = ( 0.0d0, 0.0d0 ) )
c
      if ( idecay.eq.1 ) then
         pG(1) =  dble(fo(5))
         pG(2) =  dble(fo(6))
         pG(3) = dimag(fo(6))
         pG(4) = dimag(fo(5))
      else if ( idecay.eq.-1 ) then
         pG(1) =  dble(fi(5))
         pG(2) =  dble(fi(6))
         pG(3) = dimag(fi(6))
         pG(4) = dimag(fi(5))
      else
         write(6,*) 'error in idecay of JIOGLD'
         stop
      end if

      jio(5) = fo(5) - fi(5)
      jio(6) = fo(6) - fi(6)

      q(0) = dble( jio(5))
      q(1) = dble( jio(6))
      q(2) = dimag(jio(6))
      q(3) = dimag(jio(5))
      q2  = q(0)**2 - q(1)**2 - q(2)**2 - q(3)**2
      vm2 = vmass**2

      pdotpG = q(0)*pG(1) - q(1)*pG(2) - q(2)*pG(3) - q(3)*pG(4)

      p14p = q(0) + q(3)
      p14m = q(0) - q(3)
      p23p = jio(6)
      p23m = dconjg(jio(6))

      if ( vmass.ne.rZero ) then

         d = rOne/dcmplx( q2-vm2, vmass*vwidth )
         d = d*idecay
c  for the running width, use below instead of the above d.
c         d = rOne/dcmplx( q2-vm2, max(vwidth*q2/vmass,rZero) )

         if ( gc(2).ne.cZero ) then
            dum =  ( (fo(3)*p14p + fo(4)*p23p)*fi(1)
     &              +(fo(3)*p23m + fo(4)*p14m)*fi(2) )*gc(1)
     &           + ( (fo(1)*p14m - fo(2)*p23p)*fi(3)
     &              -(fo(1)*p23m - fo(2)*p14p)*fi(4) )*gc(2)

            c0 =  dum*pG(1)
     &           -pdotpG*( gc(1)*( fo(3)*fi(1) + fo(4)*fi(2) )
     &                    +gc(2)*( fo(1)*fi(3) + fo(2)*fi(4) ) )
            c1 =  dum*pG(2)
     &           -pdotpG*(-gc(1)*( fo(4)*fi(1) + fo(3)*fi(2) )
     &                    +gc(2)*( fo(2)*fi(3) + fo(1)*fi(4) ) )
            c2 =  dum*pG(3)
     &           -pdotpG*( gc(1)*(-fo(4)*fi(1) + fo(3)*fi(2) )
     &                    +gc(2)*( fo(2)*fi(3) - fo(1)*fi(4) ) )*ci
            c3 =  dum*pG(4)
     &           -pdotpG*( gc(1)*(-fo(3)*fi(1) + fo(4)*fi(2) )
     &                    +gc(2)*( fo(1)*fi(3) - fo(2)*fi(4) ) )
         else
            d = d*gc(1)
            dum =  (fo(3)*p14p + fo(4)*p23p)*fi(1)
     &            +(fo(3)*p23m + fo(4)*p14m)*fi(2)

            c0 = dum*pG(1) - ( fo(3)*fi(1) + fo(4)*fi(2) )*pdotpG
            c1 = dum*pG(2) + ( fo(4)*fi(1) + fo(3)*fi(2) )*pdotpG
            c2 = dum*pG(3) + ( fo(4)*fi(1) - fo(3)*fi(2) )*pdotpG*ci
            c3 = dum*pG(4) + ( fo(3)*fi(1) - fo(4)*fi(2) )*pdotpG
         end if

         cs = (q(0)*c0 - q(1)*c1 - q(2)*c2 - q(3)*c3) / vm2

         jio(1) = (c0-cs*q(0))*d
         jio(2) = (c1-cs*q(1))*d
         jio(3) = (c2-cs*q(2))*d
         jio(4) = (c3-cs*q(3))*d

      else
         dd = idecay*rOne/q2

         if ( gc(2).ne.cZero ) then
            dum =  ( (fo(3)*p14p + fo(4)*p23p)*fi(1)
     &              +(fo(3)*p23m + fo(4)*p14m)*fi(2) )*gc(1)
     &           + ( (fo(1)*p14m - fo(2)*p23p)*fi(3)
     &              -(fo(1)*p23m - fo(2)*p14p)*fi(4) )*gc(2)

            jio(1) = ( dum*pG(1) - pdotpG*(
     &                 gc(1)*( fo(3)*fi(1) + fo(4)*fi(2) )
     &                +gc(2)*( fo(1)*fi(3) + fo(2)*fi(4) ) ) )*dd
            jio(2) = ( dum*pG(2) - pdotpG*(
     &                -gc(1)*( fo(4)*fi(1) + fo(3)*fi(2) )
     &                +gc(2)*( fo(2)*fi(3) + fo(1)*fi(4) ) ) )*dd
            jio(3) = ( dum*pG(3) - pdotpG*ci*(
     &                 gc(1)*(-fo(4)*fi(1) + fo(3)*fi(2) )
     &                +gc(2)*( fo(2)*fi(3) - fo(1)*fi(4) ) ) )*dd
            jio(4) = ( dum*pG(4) - pdotpG*(
     &                 gc(1)*(-fo(3)*fi(1) + fo(4)*fi(2) )
     &                +gc(2)*( fo(1)*fi(3) - fo(2)*fi(4) ) ) )*dd

         else
            dd = dd*gc(1)
            dum =  (fo(3)*p14p + fo(4)*p23p)*fi(1)
     &            +(fo(3)*p23m + fo(4)*p14m)*fi(2)

            jio(1)=dd*(dum*pG(1) - pdotpG*(fo(3)*fi(1) + fo(4)*fi(2)))
            jio(2)=dd*(dum*pG(2) + pdotpG*(fo(4)*fi(1) + fo(3)*fi(2)))
            jio(3)=dd*(dum*pG(3) + ci*pdotpG*(fo(4)*fi(1)-fo(3)*fi(2)))
            jio(4)=dd*(dum*pG(4) + pdotpG*(fo(3)*fi(1) - fo(4)*fi(2)))
         end if
      end if
c
      return
      end
