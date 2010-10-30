      subroutine jvssxx(vc,s1,s2,gc,vmass,vwidth , jvss)
c
c This subroutine computes an off-shell vector current from the vector-
c vector-scalar-scalar coupling.  The vector propagator is given in
c Feynman gauge for a massless vector and in unitary gauge for a massive
c vector.
c
c input:
c       complex vc(6)          : input  vector                        v
c       complex s1(3)          : first  scalar                        s1
c       complex s2(3)          : second scalar                        s2
c       complex gc             : coupling constant                 gvvhh
c       real    vmass          : mass  of output vector v'
c       real    vwidth         : width of output vector v'
c
c output:
c       complex jvss(6)        : vector current         j^mu(v':v,s1,s2)
c     
      implicit none
      double complex vc(6),s1(3),s2(3),gc,jvss(6),dg
      double complex cm2        ! mass**2- I Gamma mass (Fabio)
      double precision q(0:3),vmass,vwidth,q2,vm2
      double complex vk

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
         write(stdo,*) ' helas-warn  : vc in jvssxx is zero vector'
      endif
      if ( abs(vc(5))+abs(vc(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : vc in jvssxx has zero momentum'
      endif
      if ( s1(1).eq.cZero ) then
         write(stdo,*) ' helas-warn  : s1 in jvssxx is zero scalar'
      endif
      if ( abs(s1(2))+abs(s1(3)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : s1 in jvssxx has zero momentum'
      endif
      if ( s2(1).eq.cZero ) then
         write(stdo,*) ' helas-warn  : s2 in jvssxx is zero scalar'
      endif
      if ( abs(s2(2))+abs(s2(3)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : s2 in jvssxx has zero momentum'
      endif
      if ( gc.eq.cZero ) then
         write(stdo,*) ' helas-error : gc in jvssxx is zero coupling'
      endif
      if ( vmass.lt.rZero ) then
         write(stdo,*) ' helas-error : vmass in jvssxx is negative'
         write(stdo,*) '             : vmass = ',vmass
      endif
      if ( vwidth.lt.rZero ) then
         write(stdo,*) ' helas-error : vwidth in jvssxx is negative'
         write(stdo,*) '             : vwidth = ',vwidth
      endif
#endif

      jvss(5) = vc(5)+s1(2)+s2(2)
      jvss(6) = vc(6)+s1(3)+s2(3)

      q(0) = dble( jvss(5))
      q(1) = dble( jvss(6))
      q(2) = dimag(jvss(6))
      q(3) = dimag(jvss(5))
      q2 = q(0)**2-(q(1)**2+q(2)**2+q(3)**2)
      vm2 = vmass**2

#ifdef HELAS_CHECK
      if ( abs(jvss(5))+abs(jvss(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : jvss in jvssxx has zero momentum'
      endif
      if ( vwidth.eq.rZero .and. q2.eq.vm2 ) then
         write(stdo,*)
     &        ' helas-error : jvss in jvssxx is on vmass pole'
         write(stdo,*)
     &        '             : q     = ',q(0),q(1),q(2),q(3)
         write(stdo,*)
     &        '             : abs(q)= ',sqrt(abs(q2))
         jvss(1) = cZero
         jvss(2) = cZero
         jvss(3) = cZero
         jvss(4) = cZero
         return
      endif
#endif

      if ( vmass.ne.rZero ) then

         dg = gc*s1(1)*s2(1)/dcmplx( q2-vm2, vmass*vwidth )
c  For the running width, use below instead of the above dg.
c         dg = gc*s1(1)*s2(1)/cmplx( q2-vm2 , max( vwidth*q2/vmass ,rZero))

c     Fabio's implementation of the fixed width
         cm2=dcmplx( vm2, -vmass*vwidth )
c     vk = (q(0)*vc(1)-q(1)*vc(2)-q(2)*vc(3)-q(3)*vc(4))/vm2
         vk = (q(0)*vc(1)-q(1)*vc(2)-q(2)*vc(3)-q(3)*vc(4))/cm2

         jvss(1) = dg*(vc(1)-vk*q(0))
         jvss(2) = dg*(vc(2)-vk*q(1))
         jvss(3) = dg*(vc(3)-vk*q(2))
         jvss(4) = dg*(vc(4)-vk*q(3))

      else

         dg = gc*s1(1)*s2(1)/q2

         jvss(1) = dg*vc(1)
         jvss(2) = dg*vc(2)
         jvss(3) = dg*vc(3)
         jvss(4) = dg*vc(4)

      endif
c
      return
      end
