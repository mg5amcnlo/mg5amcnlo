      subroutine hioxxx(fi,fo,gc,smass,swidth , hio)
c
c This subroutine computes an off-shell scalar current from an external
c fermion pair.
c
c input:
c       complex fi(6)          : flow-in  fermion                   |fi>
c       complex fo(6)          : flow-out fermion                   <fo|
c       complex gc(2)          : coupling constants                 gchf
c       real    smass          : mass  of OUTPUT scalar s
c       real    swidth         : width of OUTPUT scalar s
c
c output:
c       complex hio(3)         : scalar current             j(<fi|s|fo>)
c
      implicit none
      double complex fi(6),fo(6),hio(3),gc(2),dn
      double precision q(0:3),smass,swidth,q2

#ifdef HELAS_CHECK
      double precision rZero, cZero
      parameter( rZero = 0.0d0 )
      double complex cZero
      parameter( cZero = ( 0.0d0, 0.0d0 ) )
      integer stdo
      parameter( stdo = 6 )
#endif
c
#ifdef HELAS_CHECK
      if ( abs(fi(1))+abs(fi(2))+abs(fi(3))+abs(fi(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : fi in hioxxx is zero spinor'
      endif
      if ( abs(fi(5))+abs(fi(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : fi in hioxxx has zero momentum'
      endif
      if ( abs(fo(1))+abs(fo(2))+abs(fo(3))+abs(fo(4)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-warn  : fo in hioxxx is zero spinor'
      endif
      if ( abs(fo(5))+abs(fo(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : fo in hioxxx has zero momentum'
      endif
      if ( gc(1).eq.cZero .and. gc(2).eq.cZero ) then
         write(stdo,*)
     &        ' helas-error : gc in hioxxx is zero coupling'
      endif
      if ( smass.lt.rZero ) then
         write(stdo,*) ' helas-error : smass in hioxxx is negative'
         write(stdo,*) '             : smass = ',smass
      endif
      if ( swidth.lt.rZero ) then
         write(stdo,*) ' helas-error : swidth in hioxxx is negative'
         write(stdo,*) '             : swidth = ',swidth
      endif
#endif

      hio(2) = fo(5)-fi(5)
      hio(3) = fo(6)-fi(6)

      q(0) = dble( hio(2))
      q(1) = dble( hio(3))
      q(2) = dimag(hio(3))
      q(3) = dimag(hio(2))
      q2 = q(0)**2-(q(1)**2+q(2)**2+q(3)**2)

#ifdef HELAS_CHECK
      if ( abs(hio(2))+abs(hio(3)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : hio in hioxxx has zero momentum'
      endif
      if ( swidth.eq.rZero .and. q2.eq.smass**2 ) then
         write(stdo,*)
     &        ' helas-error : hio in hioxxx is on smass pole'
         write(stdo,*)
     &        '             : q     = ',q(0),q(1),q(2),q(3)
         write(stdo,*)
     &        '             : abs(q)= ',sqrt(abs(q2))
         hio(1) = cZero
         return
      endif
#endif

      dn = -dcmplx( q2-smass**2, smass*swidth )

      hio(1) = ( gc(1)*(fo(1)*fi(1)+fo(2)*fi(2))
     &          +gc(2)*(fo(3)*fi(3)+fo(4)*fi(4)) )/dn
c
      return
      end
