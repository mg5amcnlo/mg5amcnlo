      subroutine iovdmx(fi,fo,vc,gc, vertex)
c
c This subroutine computes an amplitude of the fermion-fermion-vector
c dipole moment (non-renormalizable) coupling.
c
c input:
c       complex fi(6)          : flow-in  fermion                   |fi>
c       complex fo(6)          : flow-out fermion                   <fo|
c       complex vc(6)          : input    vector                      v
c       complex gc(2,2)        : coupling constants                  gvf
c                              : first index is L,R as normal
c                              : second index is EDM,-MDM
c
c output:
c       complex vertex         : amplitude                     <fo|v|fi>
c
      implicit none
      double complex fi(6), fo(6), vc(6), vertex, gc(2,2)

      double complex q(5:6), dum1, dum2
      double complex f1122, f12, f21, f3344, f34, f43
      double complex f12p21, f12m21, f34p43, f34m43
      double complex kvc21, kvc31, kvc41, kvc32, kvc42, kvc43
      double precision  rZero, rOne
      parameter( rZero = 0.0d0, rOne = 1.0d0 )
      double complex cImag, cZero
      parameter( cImag = ( 0.0d0, 1.0d0 ), cZero = ( 0.0d0, 0.0d0 ) )

#ifdef HELAS_CHECK
      double precision p0,p1,p2,p3,q0,q1,q2,q3,r0,r1,r2,r3,pm
      double precision epsi
      parameter( epsi = 4.0d-5 )
      integer stdo
      parameter( stdo = 6 )
#endif
c
#ifdef HELAS_CHECK
      p0 = -dble( fi(5))
      p1 = -dble( fi(6))
      p2 = -dimag(fi(6))
      p3 = -dimag(fi(5))
      q0 = dble( fo(5))
      q1 = dble( fo(6))
      q2 = dimag(fo(6))
      q3 = dimag(fo(5))
      r0 = dble( vc(5))
      r1 = dble( vc(6))
      r2 = dimag(vc(6))
      r3 = dimag(vc(5))
      if ( abs(fi(1))+abs(fi(2))+abs(fi(3))+abs(fi(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : fi in iovdmx is zero spinor'
      endif
      if ( abs(fi(5))+abs(fi(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : fi in iovdmx has zero momentum'
      endif
      if ( abs(fo(1))+abs(fo(2))+abs(fo(3))+abs(fo(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : fo in iovdmx is zero spinor'
      endif
      if ( abs(fo(5))+abs(fo(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : fo in iovdmx has zero momentum'
      endif
      if ( abs(vc(1))+abs(vc(2))+abs(vc(3))+abs(vc(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : vc in iovdmx is zero vector'
      endif
      if ( abs(vc(5))+abs(vc(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : vc in iovdmx has zero momentum'
      endif
      pm = max( abs(p0),abs(q0),abs(r0),abs(p1),abs(q1),abs(r1),
     &          abs(p2),abs(q2),abs(r2),abs(p3),abs(q3),abs(r3) )
      if ( abs(-fi(5)+fo(5)+vc(5))+abs(-fi(6)+fo(6)+vc(6))
     &                                               .ge.pm*epsi) then
         write(stdo,*)
     &        ' helas-error : fi,fo,vc in iovdmx'
         write(stdo,*)
     &        '                        have not balanced momenta'
      endif
      if ( gc(1,1).eq.cZero .and. gc(2,1).eq.cZero .and.
     &     gc(1,2).eq.cZero .and. gc(2,2).eq.cZero      ) then
         write(stdo,*)
     &        ' helas-error : gc in iovdmx is zero coupling'
      endif
#endif

      q(5) = fi(5) - fo(5)
      q(6) = fi(6) - fo(6)

      f1122  = fo(1)*fi(1) - fo(2)*fi(2)
      f12    = fo(1)*fi(2)
      f21    = fo(2)*fi(1)
      f12p21 = f12 + f21
      f12m21 = f12 - f21

      kvc21 = ( dble(q(6))*vc(1) -  dble(q(5))*vc(2))*cImag
      kvc31 =  dimag(q(6))*vc(1) -  dble(q(5))*vc(3)
      kvc41 = (dimag(q(5))*vc(1) -  dble(q(5))*vc(4))*cImag
      kvc32 =  dimag(q(6))*vc(2) -  dble(q(6))*vc(3)
      kvc42 = (dimag(q(5))*vc(2) -  dble(q(6))*vc(4))*cImag
      kvc43 =  dimag(q(5))*vc(3) - dimag(q(6))*vc(4)

      dum1 =   ( kvc31 + kvc42 )*f12m21
     &       + ( kvc32 + kvc41 )*f1122
     &       + ( kvc43 + kvc21 )*f12p21

c     (-) from gamma^5 in EDM only
      vertex = ( -gc(1,1) + cImag*gc(1,2) )*dum1    

      if ( gc(2,1).ne.cZero .or.
     &     gc(2,2).ne.cZero      ) then
         f3344  = fo(3)*fi(3) - fo(4)*fi(4)
         f34    = fo(3)*fi(4)
         f43    = fo(4)*fi(3)
         f34p43 = f34 + f43
         f34m43 = f34 - f43
         dum2 =   (-kvc31 + kvc42 )*f34m43
     &          + ( kvc32 - kvc41 )*f3344
     &          + ( kvc43 - kvc21 )*f34p43
         vertex = vertex + ( gc(2,1) + cImag*gc(2,2) )*dum2
      end if
c
      return
      end
