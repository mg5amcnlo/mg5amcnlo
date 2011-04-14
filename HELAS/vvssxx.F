      subroutine vvssxx(v1,v2,s1,s2,gc , vertex)
c
c This subroutine computes an amplitude of the vector-vector-scalar-
c scalar coupling.
c
c input:
c       complex v1(6)          : first  vector                        v1
c       complex v2(6)          : second vector                        v2
c       complex s1(3)          : first  scalar                        s1
c       complex s2(3)          : second scalar                        s2
c       complex gc             : coupling constant                 gvvhh
c
c output:
c       complex vertex         : amplitude            gamma(v1,v2,s1,s2)
c     
      implicit none
      double complex v1(6),v2(6),s1(3),s2(3),gc,vertex

#ifdef HELAS_CHECK
      double precision p0,p1,p2,p3,q0,q1,q2,q3,r0,r1,r2,r3
      double precision o0,o1,o2,o3,pm
      double precision epsi
      parameter( epsi = 4.0d-5 )
      double precision rZero
      parameter( rZero = 0.0d0 )
      double complex cZero
      parameter( cZero = ( 0.0d0, 0.0d0 ) )
      integer stdo
      parameter( stdo = 6 )
#endif
c
#ifdef HELAS_CHECK
      p0 = dble( v1(2))
      p1 = dble( v1(3))
      p2 = dimag(v1(3))
      p3 = dimag(v1(2))
      q0 = dble( v2(2))
      q1 = dble( v2(3))
      q2 = dimag(v2(3))
      q3 = dimag(v2(2))
      r0 = dble( s1(2))
      r1 = dble( s1(3))
      r2 = dimag(s1(3))
      r3 = dimag(s1(2))
      o0 = dble( s2(2))
      o1 = dble( s2(3))
      o2 = dimag(s2(3))
      o3 = dimag(s2(2))
      if ( abs(v1(1))+abs(v1(2))+abs(v1(3))+abs(v1(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : v1 in vvssxx is zero vector'
      endif
      if ( abs(v1(5))+abs(v1(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : v1 in vvssxx has zero momentum'
      endif
      if ( abs(v2(1))+abs(v2(2))+abs(v2(3))+abs(v2(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : v2 in vvssxx is zero vector'
      endif
      if ( abs(v2(5))+abs(v2(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : v2 in vvssxx has zero momentum'
      endif
      if ( abs(s1(1)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : s1 in vvssxx is zero scalar'
      endif
      if ( abs(s1(2))+abs(s1(3)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : s1 in vvssxx has zero momentum'
      endif
      if ( abs(s2(1)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : s2 in vvssxx is zero scalar'
      endif
      if ( abs(s2(2))+abs(s2(3)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : s2 in vvssxx has zero momentum'
      endif
      pm = max( abs(p0),abs(q0),abs(r0),abs(o0),
     &          abs(p1),abs(q1),abs(r1),abs(o1),
     &          abs(p2),abs(q2),abs(r2),abs(o2),
     &          abs(p3),abs(q3),abs(r3),abs(o3) )
      if (  abs(v1(5)+v2(5)+s1(2)+s2(2))
     &     +abs(v1(6)+v2(6)+s1(3)+s2(3)).ge.pm*epsi ) then
         write(stdo,*)
     &        ' helas-error : v1,v2,s1,s2 in vvssxx'
         write(stdo,*) 
     &        '             : have not balanced momenta'
       endif
       if ( gc.eq.cZero ) then
          write(stdo,*) ' helas-error : gc in vvssxx is zero coupling'
       endif
#endif

      vertex = gc*s1(1)*s2(1)
     &        *(v1(1)*v2(1)-v1(2)*v2(2)-v1(3)*v2(3)-v1(4)*v2(4))
c
      return
      end
