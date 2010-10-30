      subroutine jvsxxx(vc,sc,gc,vmass,vwidth , jvs)
c
c This subroutine computes an off-shell vector current from the vector-
c vector-scalar coupling.  The vector propagator is given in Feynman
c gauge for a massless vector and in unitary gauge for a massive vector.
c
c input:
c       complex vc(6)          : input vector                          v
c       complex sc(3)          : input scalar                          s
c       complex gc             : coupling constant                  gvvh
c       real    vmass          : mass  of output vector v'
c       real    vwidth         : width of output vector v'
c
c output:
c       complex jvs(6)         : vector current             j^mu(v':v,s)
c     
      implicit none
      double complex vc(6),sc(3),gc,jvs(6),dg,vk
      double complex cm2        ! mass**2- I Gamma mass (Fabio)
      double precision q(0:3),vmass,vwidth,q2,vm2

      double precision rZero
      parameter( rZero = 0.0d0 )

#ifdef HELAS_CHECK
      double complex cZero
      parameter( cZero = ( 0.0d0, 0.0d0 ) )
      integer stdo
      parameter( stdo = 6 )
#endif
c
#ifdef HELAS_CHECK
      if ( abs(vc(1))+abs(vc(2))+abs(vc(3))+abs(vc(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : vc in jvsxxx is zero vector'
      endif
      if ( abs(vc(5))+abs(vc(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : vc in jvsxxx has zero momentum'
      endif
      if ( sc(1).eq.cZero ) then
         write(stdo,*) ' helas-warn  : sc in jvsxxx is zero scalar'
      endif
      if ( abs(sc(2))+abs(sc(3)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : sc in jvsxxx has zero momentum'
      endif
      if ( gc.eq.cZero ) then
         write(stdo,*) ' helas-error : gc in jvsxxx is zero coupling'
      endif
      if ( vmass.le.rZero ) then
         write(stdo,*) ' helas-error : vmass in jvsxxx is not positive'
         write(stdo,*) '             : vmass = ',vmass
      endif
      if ( vwidth.lt.rZero ) then
         write(stdo,*) ' helas-error : vwidth in jvsxxx is negative'
         write(stdo,*) '             : vwidth = ',vwidth
      endif
#endif

      jvs(5) = vc(5)+sc(2)
      jvs(6) = vc(6)+sc(3)

      q(0) = dble( jvs(5))
      q(1) = dble( jvs(6))
      q(2) = dimag(jvs(6))
      q(3) = dimag(jvs(5))
      q2 = q(0)**2-(q(1)**2+q(2)**2+q(3)**2)
      vm2 = vmass**2

#ifdef HELAS_CHECK
      if ( abs(jvs(5))+abs(jvs(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : jvs in jvsxxx has zero momentum'
      endif
      if ( vwidth.eq.rZero .and. q2.eq.vm2 ) then
         write(stdo,*)
     &        ' helas-error : jvs in jvsxxx is on vmass pole'
         write(stdo,*) 
     &        '             : q     = ',q(0),q(1),q(2),q(3)
         write(stdo,*) 
     &        '             : abs(q)= ',sqrt(abs(q2))
         jvs(1)=cmplx(rZero)
         jvs(2)=cmplx(rZero)
         jvs(3)=cmplx(rZero)
         jvs(4)=cmplx(rZero)
         return
      endif
#endif

      if ( vmass.ne.rZero ) then

         dg = gc*sc(1)/dcmplx( q2-vm2, vmass*vwidth )
c  For the running width, use below instead of the above dg.
c         dg = g*sc(1)/dcmplx( q2-vm2, max(vwidth*q2/vmass,rZero) )

c     Fabio's implementation of the fixed width
         cm2=dcmplx( vm2, -vmass*vwidth )
c     vk = (-q(0)*vc(1)+q(1)*vc(2)+q(2)*vc(3)+q(3)*vc(4))/vm2
         vk = (-q(0)*vc(1)+q(1)*vc(2)+q(2)*vc(3)+q(3)*vc(4))/cm2

         jvs(1) = dg*(q(0)*vk+vc(1))
         jvs(2) = dg*(q(1)*vk+vc(2))
         jvs(3) = dg*(q(2)*vk+vc(3))
         jvs(4) = dg*(q(3)*vk+vc(4))

      else

         dg=gc*sc(1)/q2

         jvs(1) = dg*vc(1)
         jvs(2) = dg*vc(2)
         jvs(3) = dg*vc(3)
         jvs(4) = dg*vc(4)

      endif
c
      return
      end
