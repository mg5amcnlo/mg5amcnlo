      subroutine hvvxxx(v1,v2,gc,smass,swidth , hvv)
c
c This subroutine computes an off-shell scalar current from the vector-
c vector-scalar coupling.
c
c input:
c       complex v1(6)          : first  vector                        v1
c       complex v2(6)          : second vector                        v2
c       complex gc             : coupling constant                  gvvh
c       real    smass          : mass  of OUTPUT scalar s
c       real    swidth         : width of OUTPUT scalar s
c
c output:
c       complex hvv(3)         : off-shell scalar current     j(s:v1,v2)
c     
      implicit none
      double complex v1(6),v2(6),gc,hvv(3),dg
      double precision q(0:3),smass,swidth,q2

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
      if ( abs(v1(1))+abs(v1(2))+abs(v1(3))+abs(v1(4)).eq.rZero) then
         write(stdo,*) ' helas-warn  : v1 in hvvxxx is zero vector'
      endif
      if ( abs(v1(5))+abs(v1(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : v1 in hvvxxx has zero momentum'
      endif
      if ( abs(v2(1))+abs(v2(2))+abs(v2(3))+abs(v2(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : v2 in hvvxxx is zero vector'
      endif
      if ( abs(v2(5))+abs(v2(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : v2 in hvvxxx has zero momentum'
      endif
      if ( gc.eq.cZero ) then
         write(stdo,*) ' helas-error : gc in hvvxxx is zero coupling'
      endif
      if ( smass.lt.rZero ) then
         write(stdo,*) ' helas-error : smass in hvvxxx is negative'
         write(stdo,*) '             : smass = ',smass
      endif
      if ( swidth.lt.rZero ) then
         write(stdo,*) ' helas-error : swidth in hvvxxx is negative'
         write(stdo,*) '             : swidth = ',swidth
      endif
#endif

      hvv(2) = v1(5)+v2(5)
      hvv(3) = v1(6)+v2(6)

      q(0) = dble( hvv(2))
      q(1) = dble( hvv(3))
      q(2) = dimag(hvv(3))
      q(3) = dimag(hvv(2))
      q2 = q(0)**2-(q(1)**2+q(2)**2+q(3)**2)

#ifdef HELAS_CHECK
      if ( abs(hvv(2))+abs(hvv(3)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : hvv in hvvxxx has zero momentum'
      endif
      if ( swidth.eq.rZero .and. q2.eq.smass**2 ) then
         write(stdo,*)
     &        ' helas-error : hvv in hvvxxx is on smass pole'
         write(stdo,*) 
     &        '             : q     = ',q(0),q(1),q(2),q(3)
         write(stdo,*)
     &        '             : abs(q)= ',sqrt(abs(q2))
         hvv(1) = cZero
         return
      endif
#endif

      dg = -gc/dcmplx( q2-smass**2, smass*swidth )

      hvv(1) = dg*(v1(1)*v2(1)-v1(2)*v2(2)-v1(3)*v2(3)-v1(4)*v2(4))
c
      return
      end
