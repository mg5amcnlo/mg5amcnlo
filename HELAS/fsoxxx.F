      subroutine fsoxxx(fo,sc,gc,fmass,fwidth , fso)
c
c This subroutine computes an off-shell fermion wavefunction from a
c flowing-OUT external fermion and a vector boson.
c
c input:
c       complex fo(6)          : flow-out fermion                   <fo|
c       complex sc(6)          : input    scalar                      s
c       complex gc(2)          : coupling constants                 gchf
c       real    fmass          : mass  of OUTPUT fermion f'
c       real    fwidth         : width of OUTPUT fermion f'
c
c output:
c       complex fso(6)         : off-shell fermion             <fo,s,f'|
c
      implicit none
      double complex fo(6),sc(6),fso(6),gc(2),sl1,sl2,sr1,sr2,ds
      double precision pf(0:3),fmass,fwidth,pf2,p0p3,p0m3

#ifdef HELAS_CHECK
      double precision rZero, cZero
      parameter( rZero = 0.0d0, cZero = ( 0.0d0, 0.0d0 ) )
      integer stdo
      parameter( stdo = 6 )
#endif
c
#ifdef HELAS_CHECK
      if ( abs(fo(1))+abs(fo(2))+abs(fo(3))+abs(fo(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : fo in fsoxxx is zero spinor'
      endif
      if ( abs(fo(5))+abs(fo(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : fo in fsoxxx has zero momentum'
      endif
      if ( sc(1).eq.cZero ) then
         write(stdo,*) ' helas-warn  : sc in fsoxxx is zero scalar'
      endif
      if ( abs(sc(2))+abs(sc(3)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : sc in fsoxxx has zero momentum'
      endif
      if ( gc(1).eq.cZero .and. gc(2).eq.cZero ) then
         write(stdo,*) ' helas-error : gc in fsoxxx is zero coupling'
      endif
      if ( fmass.lt.rZero ) then
         write(stdo,*) ' helas-error : fmass in fsoxxx is negative'
         write(stdo,*) '             : fmass = ',fmass
      endif
      if ( fwidth.lt.rZero ) then
         write(stdo,*) ' helas-error : fwidth in fsoxxx is negative'
         write(stdo,*) '               fwidth = ',fwidth
      endif
#endif

      fso(5) = fo(5)+sc(2)
      fso(6) = fo(6)+sc(3)

      pf(0) = dble( fso(5))
      pf(1) = dble( fso(6))
      pf(2) = dimag(fso(6))
      pf(3) = dimag(fso(5))
      pf2 = pf(0)**2-(pf(1)**2+pf(2)**2+pf(3)**2)

#ifdef HELAS_CHECK
      if ( abs(fso(5))+abs(fso(6)).eq.rZero ) then
          write(stdo,*)
     &        ' helas-error : fso in fsoxxx has zero momentum'
       endif
       if ( fwidth.eq.rZero .and. pf2.eq.fmass**2 ) then
          write(stdo,*)
     &         ' helas-error : fso in fsoxxx is on fmass pole'
          write(stdo,*)
     &         '             : p     = ',pf(0),pf(1),pf(2),pf(3)
          write(stdo,*)
     &         '             : abs(p)= ',sqrt(abs(pf2))
         fso(1) = cZero
         fso(2) = cZero
         fso(3) = cZero
         fso(4) = cZero
         return
      endif
#endif

      ds = -sc(1)/dcmplx( pf2-fmass**2, fmass*fwidth )
      p0p3 = pf(0)+pf(3)
      p0m3 = pf(0)-pf(3)
      sl1 = gc(2)*(p0p3*fo(3)       +fso(6) *fo(4))
      sl2 = gc(2)*(p0m3*fo(4)+dconjg(fso(6))*fo(3))
      sr1 = gc(1)*(p0m3*fo(1)       -fso(6) *fo(2))
      sr2 = gc(1)*(p0p3*fo(2)-dconjg(fso(6))*fo(1))

      fso(1) = ( gc(1)*fmass*fo(1) + sl1 )*ds
      fso(2) = ( gc(1)*fmass*fo(2) + sl2 )*ds
      fso(3) = ( gc(2)*fmass*fo(3) + sr1 )*ds
      fso(4) = ( gc(2)*fmass*fo(4) + sr2 )*ds
c
      return
      end
