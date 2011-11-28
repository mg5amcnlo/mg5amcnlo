      subroutine iosxxx(fi,fo,sc,gc , vertex)
c
c This subroutine computes an amplitude of the fermion-fermion-scalar
c coupling.
c
c input:
c       complex fi(6)          : flow-in  fermion                   |fi>
c       complex fo(6)          : flow-out fermion                   <fo|
c       complex sc(3)          : input    scalar                      s
c       complex gc(2)          : coupling constants                 gchf
c
c output:
c       complex vertex         : amplitude                     <fo|s|fi>
c     
      implicit none
      double complex fi(6),fo(6),sc(3),gc(2),vertex
      INTEGER DIM
      PARAMETER(DIM=18)

c      include 'dimension.inc'
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
      p0 = -dble( fi(5))
      p1 = -dble( fi(6))
      p2 = -dimag(fi(6))
      p3 = -dimag(fi(5))
      q0 = dble( fo(5))
      q1 = dble( fo(6))
      q2 = dimag(fo(6))
      q3 = dimag(fo(5))
      r0 = dble( sc(2))
      r1 = dble( sc(3))
      r2 = dimag(sc(3))
      r3 = dimag(sc(2))
      if ( abs(fi(1))+abs(fi(2))+abs(fi(3))+abs(fi(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : fi in iosxxx is zero spinor'
      endif
      if ( abs(fi(5))+abs(fi(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : fi in iosxxx has zero momentum'
      endif
      if ( abs(fo(1))+abs(fo(2))+abs(fo(3))+abs(fo(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : fo in iosxxx is zero spinor'
      endif
      if ( abs(fo(5))+abs(fo(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : fo in iosxxx has zero momentum'
      endif
      if ( sc(1).eq.cZero ) then
         write(stdo,*) ' helas-warn  : sc in iosxxx is zero scalar'
      endif
      if ( abs(sc(2))+abs(sc(3)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : sc in iosxxx has zero momentum'
      endif
      pm = max( abs(p0),abs(q0),abs(r0),abs(p1),abs(q1),abs(r1),
     &          abs(p2),abs(q2),abs(r2),abs(p3),abs(q3),abs(r3) )
      if ( abs(-fi(5)+fo(5)+sc(2))+abs(-fi(6)+fo(6)+sc(3))
     &                                               .ge.pm*epsi) then
         write(stdo,*)
     &        ' helas-error : fi,fo,sc in iosxxx'
         write(stdo,*)
     &        '             :          have not balanced momenta'
      endif
      if ( gc(1).eq.cZero .and. gc(2).eq.cZero ) then
         write(stdo,*)
     &        ' helas-error : gc in iosxxx is zero coupling'
      endif
#endif

      vertex = sc(1)*( gc(1)*(fi(1)*fo(1)+fi(2)*fo(2))
     &                +gc(2)*(fi(3)*fo(3)+fi(4)*fo(4)) )
c
      return
      end
