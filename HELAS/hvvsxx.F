      subroutine hvvsxx(v1,v2,sc,gc,smass,swidth , hvvs)
c
c This subroutine computes an off-shell scalar current of the vector-
c vector-scalar-scalar coupling.
c
c input:
c       complex v1(6)          : first  vector                        v1
c       complex v2(6)          : second vector                        v2
c       complex sc(3)          : input  scalar                        s
c       complex gc             : coupling constant                 gvvhh
c       real    smass          : mass  of OUTPUT scalar s'
c       real    swidth         : width of OUTPUT scalar s'
c
c output:
c       complex hvvs(3)        : scalar current            j(s':v1,v2,s)
c     
      implicit none
      double complex v1(6),v2(6),sc(3),gc,hvvs(3),dg
      double precision q(0:3),smass,swidth,q2

#ifdef HELAS_CHECK
      double precision rZero
      parameter( rZero = 0.0d0 )
      double complex cZero
      parameter( cZero = ( 0.0d0, 0.0d0 ) )
      integer stdo
      parameter( stdo = 6 )
#endif
c
#ifdef HELAS_CHECK
      if ( abs(v1(1))+abs(v1(2))+abs(v1(3))+abs(v1(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : v1 in hvvsxx is zero vector'
      endif
      if ( abs(v1(5))+abs(v1(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : v1 in hvvsxx has zero momentum'
      endif
      if ( abs(v2(1))+abs(v2(2))+abs(v2(3))+abs(v2(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : v2 in hvvsxx is zero vector'
      endif
      if ( abs(v2(5))+abs(v2(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : v2 in hvvsxx has zero momentum'
      endif
      if ( sc(1).eq.cZero ) then
         write(stdo,*) ' helas-warn  : sc in hvvsxx is zero scalar'
      endif
      if ( abs(sc(2))+abs(sc(3)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : sc in hvvsxx has zero momentum'
      endif
      if ( gc.eq.cZero ) then
         write(stdo,*) ' helas-error : gc in hvvsxx is zero coupling'
      endif
      if ( smass.lt.rZero ) then
         write(stdo,*) ' helas-error : smass in hvvsxx is negative'
         write(stdo,*) '             : smass = ',smass
      endif
      if ( swidth.lt.rZero ) then
         write(stdo,*) ' helas-error : swidth in hvvsxx is negative'
         write(stdo,*) '             : swidth = ',swidth
      endif
#endif

      hvvs(2) = v1(5)+v2(5)+sc(2)
      hvvs(3) = v1(6)+v2(6)+sc(3)

      q(0) = dble( hvvs(2))
      q(1) = dble( hvvs(3))
      q(2) = dimag(hvvs(3))
      q(3) = dimag(hvvs(2))
      q2 = q(0)**2-(q(1)**2+q(2)**2+q(3)**2)

#ifdef HELAS_CHECK
      if ( abs(hvvs(2))+abs(hvvs(3)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : hvvs in hvvsxx has zero momentum'
      endif
      if ( swidth.eq.rZero .and. q2.eq.smass**2 ) then
         write(stdo,*)
     &        ' helas-error : hvvs in hvvsxx is on smass pole'
         write(stdo,*)
     &        '             : q     = ',q(0),q(1),q(2),q(3)
         write(stdo,*)
     &        '             : abs(q)= ',sqrt(abs(q2))
         hvvs(1) = cZero
         return
      endif
#endif

      dg = -gc/dcmplx( q2-smass**2, smass*swidth )

      hvvs(1) = dg*sc(1)
     &         *(v1(1)*v2(1)-v1(2)*v2(2)-v1(3)*v2(3)-v1(4)*v2(4))
c
      return
      end
