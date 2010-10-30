      subroutine vssxxx(vc,s1,s2,gc , vertex)
c
c This subroutine computes an amplitude from the vector-scalar-scalar
c coupling.  The coupling is absent in the minimal SM in unitary gauge.
c
c       complex vc(6)          : input  vector                        v
c       complex s1(3)          : first  scalar                        s1
c       complex s2(3)          : second scalar                        s2
c       complex gc             : coupling constant (s1 charge)
c
c examples of the coupling constant gc for SUSY particles are as follows:
c   -----------------------------------------------------------
c   |    s1    | (q,i3) of s1  ||   v=a   |   v=z   |   v=w   |
c   -----------------------------------------------------------
c   | nu~_l    | (  0  , +1/2) ||   ---   |  gzn(1) |  gwf(1) |
c   | e~_l     | ( -1  , -1/2) ||  gal(1) |  gzl(1) |  gwf(1) |
c   | u~_l     | (+2/3 , +1/2) ||  gau(1) |  gzu(1) |  gwf(1) |
c   | d~_l     | (-1/3 , -1/2) ||  gad(1) |  gzd(1) |  gwf(1) |
c   -----------------------------------------------------------
c   | e~_r-bar | ( +1  ,  0  ) || -gal(2) | -gzl(2) | -gwf(2) |
c   | u~_r-bar | (-2/3 ,  0  ) || -gau(2) | -gzu(2) | -gwf(2) |
c   | d~_r-bar | (+1/3 ,  0  ) || -gad(2) | -gzd(2) | -gwf(2) |
c   -----------------------------------------------------------
c where the s1 charge is defined by the flowing-OUT quantum number.
c
c output:
c       complex vertex         : amplitude                gamma(v,s1,s2)
c     
      implicit none
      double complex vc(6),s1(3),s2(3),gc,vertex
      double precision p(0:3)

#ifdef HELAS_CHECK
      double precision p0,p1,p2,p3,q0,q1,q2,q3,r0,r1,r2,r3,pm
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
      r0 = dble( vc(5))
      r1 = dble( vc(6))
      r2 = dimag(vc(6))
      r3 = dimag(vc(5))
      if ( abs(vc(1))+abs(vc(2))+abs(vc(3))+abs(vc(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : vc in vssxxx is zero vector'
      endif
      if ( abs(vc(5))+abs(vc(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : vc in vssxxx has zero momentum'
      endif
      if ( s1(1).eq.cZero ) then
         write(stdo,*) ' helas-warn  : s1 in vssxxx is zero scalar'
      endif
      if ( abs(s1(2))+abs(s1(3)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : s1 in vssxxx has zero momentum'
      endif
      if ( s2(1).eq.cZero ) then
         write(stdo,*) ' helas-warn  : s2 in vssxxx is zero scalar'
      endif
      if ( abs(s2(2))+abs(s2(3)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : s2 in vssxxx has zero momentum'
      endif
      pm = max( abs(p0),abs(q0),abs(r0),abs(p1),abs(q1),abs(r1),
     &          abs(p2),abs(q2),abs(r2),abs(p3),abs(q3),abs(r3) )
      if ( abs(vc(5)+s1(2)+s2(2))+abs(vc(6)+s1(3)+s2(3))
     &                                              .ge.pm*epsi ) then
         write(stdo,*)
     &        ' helas-error : vc,s1,s2 in vssxxx'
         write(stdo,*)
     &        '             : have not balanced momenta'
      endif
      if ( gc.eq.cZero ) then
         write(stdo,*) ' helas-error : g in vssxxx is zero coupling'
      endif
#endif

      p(0) = dble( s1(2)-s2(2))
      p(1) = dble( s1(3)-s2(3))
      p(2) = dimag(s1(3)-s2(3))
      p(3) = dimag(s1(2)-s2(2))

      vertex = gc*s1(1)*s2(1)
     &        *(vc(1)*p(0)-vc(2)*p(1)-vc(3)*p(2)-vc(4)*p(3))
c
      return
      end
