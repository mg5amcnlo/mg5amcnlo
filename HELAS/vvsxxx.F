      subroutine vvsxxx(v1,v2,sc,gc , vertex)
c
c This subroutine computes an amplitude of the vector-vector-scalar
c coupling.
c
c input:
c       complex v1(6)          : first  vector                        v1
c       complex v2(6)          : second vector                        v2
c       complex sc(3)          : input  scalar                        s
c       complex gc             : coupling constant                  gvvh
c
c output:
c       complex vertex         : amplitude                gamma(v1,v2,s)
c     
      implicit none
      double complex v1(6),v2(6),sc(3),gc,vertex

#ifdef HELAS_CHECK
      double precision p10,p11,p12,p13,p20,p21,p22,p23,q0,q1,q2,q3,pm
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
      p10 = dble( v1(5))
      p11 = dble( v1(6))
      p12 = dimag(v1(6))
      p13 = dimag(v1(5))
      p20 = dble( v2(5))
      p21 = dble( v2(6))
      p22 = dimag(v2(6))
      p23 = dimag(v2(5))
      q0  = dble( sc(2))
      q1  = dble( sc(3))
      q2  = dimag(sc(3))
      q3  = dimag(sc(2))
      if ( abs(v1(1))+abs(v1(2))+abs(v1(3))+abs(v1(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : v1 in vvsxxx is zero vector'
      endif
      if ( abs(v1(5))+abs(v1(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : v1 in vvsxxx has zero momentum'
      endif
      if ( abs(v2(1))+abs(v2(2))+abs(v2(3))+abs(v2(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : v2 in vvsxxx is zero vector'
      endif
      if ( abs(v2(5))+abs(v2(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : v2 in vvsxxx has zero momentum'
      endif
      if ( abs(sc(1)).eq.rZero) then
         write(stdo,*) ' helas-warn  : sc in vvsxxx is zero scalar'
      endif
      if ( abs(sc(2))+abs(sc(3)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : sc in vvsxxx has zero momentum'
      endif
      pm = max( abs(p10),abs(p20),abs(q0),abs(p11),abs(p21),abs(q1),
     &          abs(p12),abs(p22),abs(q2),abs(p13),abs(p23),abs(q3) )
      if ( abs(v1(5)+v2(5)+sc(2))+abs(v1(6)+v2(6)+sc(3))
     &                                                  .ge.pm*epsi ) then
         write(stdo,*)
     &        ' helas-error : v1,v2,sc in vvsxxx'
         write(stdo,*)
     &        '             : have not balanced momenta'
      endif
      if (gc.eq.cZero) then
         write(stdo,*) ' helas-error : gc in vvsxxx is zero coupling'
      endif
#endif

      vertex = gc*sc(1)
     &        *(v1(1)*v2(1)-v1(2)*v2(2)-v1(3)*v2(3)-v1(4)*v2(4))
c
      return
      end
