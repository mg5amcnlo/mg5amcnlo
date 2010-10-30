      subroutine ssssxx(s1,s2,s3,s4,gc , vertex)
c
c This subroutine computes an amplitude of the four-scalar coupling.
c
c input:
c       complex s1(3)          : first  scalar                        s1
c       complex s2(3)          : second scalar                        s2
c       complex s3(3)          : third  scalar                        s3
c       complex s4(3)          : fourth scalar                        s4
c       complex gc             : coupling constant                 ghhhh
c
c output:
c       complex vertex         : amplitude            gamma(s1,s2,s3,s4)
c     
      implicit none
      double complex s1(3),s2(3),s3(3),s4(3),gc,vertex

#ifdef HELAS_CHECK
      double precision p0,p1,p2,p3,q0,q1,q2,q3,r0,r1,r2,r3,o0,o1,o2,o3,
     &     pm
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
      p0 = dble( s1(2))
      p1 = dble( s1(3))
      p2 = dimag(s1(3))
      p3 = dimag(s1(2))
      q0 = dble( s2(2))
      q1 = dble( s2(3))
      q2 = dimag(s2(3))
      q3 = dimag(s2(2))
      r0 = dble( s3(2))
      r1 = dble( s3(3))
      r2 = dimag(s3(3))
      r3 = dimag(s3(2))
      o0 = dble( s4(2))
      o1 = dble( s4(3))
      o2 = dimag(s4(3))
      o3 = dimag(s4(2))
      if ( s1(1).eq.cZero ) then
         write(stdo,*) ' helas-warn  : s1 in ssssxx is zero scalar'
      endif
      if ( abs(s1(2))+abs(s1(3)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : s1 in ssssxx has zero momentum'
      endif
      if ( s2(1).eq.cZero ) then
         write(stdo,*) ' helas-warn  : s2 in ssssxx is zero scalar'
      endif
      if ( abs(s2(2))+abs(s2(3)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : s2 in ssssxx has zero momentum'
      endif
      if ( s3(1).eq.cZero ) then
         write(stdo,*) ' helas-warn  : s3 in ssssxx is zero scalar'
      endif
      if ( abs(s3(2))+abs(s3(3)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : s3 in ssssxx has zero momentum'
      endif
      if ( s4(1).eq.cZero ) then
         write(stdo,*) ' helas-warn  : s4 in ssssxx is zero scalar'
      endif
      if ( abs(s4(2))+abs(s4(3)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : s4 in ssssxx has zero momentum'
      endif
      pm = max( abs(p0),abs(q0),abs(r0),abs(o0),
     &          abs(p1),abs(q1),abs(r1),abs(o1),
     &          abs(p2),abs(q2),abs(r2),abs(o2),
     &          abs(p3),abs(q3),abs(r3),abs(o3) )
      if (  abs(s1(2)+s2(2)+s3(2)+s4(2))
     &     +abs(s1(3)+s2(3)+s3(3)+s4(3)).ge.pm*epsi ) then
         write(stdo,*)
     &        ' helas-error : s1,s2,s3,s4 in ssssxx'
         write(stdo,*)
     &         '            : have not balanced momenta'
       endif
       if ( gc.eq.cZero ) then
          write(stdo,*) ' helas-error : g in ssssxx is zero coupling'
       endif
#endif

      vertex = gc*s1(1)*s2(1)*s3(1)*s4(1)
c
      return
      end
