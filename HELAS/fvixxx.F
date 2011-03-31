      subroutine fvixxx(fi,vc,gc,fmass,fwidth , fvi)
c
c This subroutine computes an off-shell fermion wavefunction from a
c flowing-IN external fermion and a vector boson.
c
c input:
c       complex fi(6)          : flow-in  fermion                   |fi>
c       complex vc(6)          : input    vector                      v
c       complex gc(2)          : coupling constants                  gvf
c       real    fmass          : mass  of output fermion f'
c       real    fwidth         : width of output fermion f'
c
c output:
c       complex fvi(6)         : off-shell fermion             |f',v,fi>
c
      implicit none
      double complex fi(6),vc(6),gc(2),fvi(6),sl1,sl2,sr1,sr2,d
      double precision pf(0:3),fmass,fwidth,pf2
      
      double precision rZero, rOne
      parameter( rZero = 0.0d0, rOne = 1.0d0 )
      double complex cImag, cZero
      parameter( cImag = ( 0.0d0, 1.0d0 ), cZero = ( 0.0d0, 0.0d0 ) )

#ifdef HELAS_CHECK
      integer stdo
      parameter( stdo = 6 )
#endif
c
#ifdef HELAS_CHECK
      if ( abs(fi(1))+abs(fi(2))+abs(fi(3))+abs(fi(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : fi in fvixxx is zero spinor'
      endif
      if ( abs(fi(5))+abs(fi(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : fi in fvixxx has zero momentum'
      endif
      if ( abs(vc(1))+abs(vc(2))+abs(vc(3))+abs(vc(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : vc in fvixxx is zero vector'
      endif
      if ( abs(vc(5))+abs(vc(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : vc in fvixxx has zero momentum'
      endif
      if ( gc(1).eq.cZero .and. gc(2).eq.cZero ) then
         write(stdo,*)
     &        ' helas-error : gc in fvixxx is zero coupling'
      endif
      if ( fmass.lt.rZero ) then
         write(stdo,*) ' helas-error : fmass in fvixxx is negative'
         write(stdo,*) '             : fmass = ',fmass
      endif
      if ( fwidth.lt.rZero ) then
         write(stdo,*) ' helas-error : fwidth in fvixxx is negative'
         write(stdo,*) '             : fwidth = ',fwidth
      endif
#endif

      fvi(5) = fi(5)-vc(5)
      fvi(6) = fi(6)-vc(6)

      pf(0) = dble( fvi(5))
      pf(1) = dble( fvi(6))
      pf(2) = dimag(fvi(6))
      pf(3) = dimag(fvi(5))
      pf2 = pf(0)**2-(pf(1)**2+pf(2)**2+pf(3)**2)

#ifdef HELAS_CHECK
      if ( abs(fvi(5))+abs(fvi(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : fvi in fvixxx has zero momentum'
      endif
      if ( fwidth.eq.rZero .and. pf2.eq.fmass**2 ) then
         write(stdo,*)
     &        ' helas-error : fvi in fvixxx is on fmass pole'
         write(stdo,*)
     &        '             : p     = ',pf(0),pf(1),pf(2),pf(3)
         write(stdo,*)
     &        '             : abs(p)= ',dsqrt(dabs(pf2))
         fvi(1) = cZero
         fvi(2) = cZero
         fvi(3) = cZero
         fvi(4) = cZero
         return
      endif
#endif

      d = -rOne/dcmplx( pf2-fmass**2, fmass*fwidth )
      sl1 =   (vc(1)+      vc(4))*fi(1)
     &      + (vc(2)-cImag*vc(3))*fi(2)
      sl2 =   (vc(2)+cImag*vc(3))*fi(1)
     &      + (vc(1)-      vc(4))*fi(2)

      if ( gc(2).ne.cZero ) then
         sr1 =   (vc(1)-      vc(4))*fi(3)
     &         - (vc(2)-cImag*vc(3))*fi(4)
         sr2 = - (vc(2)+cImag*vc(3))*fi(3)
     &         + (vc(1)+      vc(4))*fi(4)

         fvi(1) = ( gc(1)*((pf(0)-pf(3))*sl1 - dconjg(fvi(6))*sl2)
     &             +gc(2)*fmass*sr1 )*d
         fvi(2) = ( gc(1)*(      -fvi(6)*sl1 +  (pf(0)+pf(3))*sl2)
     &             +gc(2)*fmass*sr2 )*d
         fvi(3) = ( gc(2)*((pf(0)+pf(3))*sr1 + dconjg(fvi(6))*sr2)
     &             +gc(1)*fmass*sl1 )*d
         fvi(4) = ( gc(2)*(       fvi(6)*sr1 +  (pf(0)-pf(3))*sr2)
     &             +gc(1)*fmass*sl2 )*d

      else
         d = d * gc(1)
         fvi(1) = ((pf(0)-pf(3))*sl1 - dconjg(fvi(6))*sl2)*d
         fvi(2) = (      -fvi(6)*sl1 +  (pf(0)+pf(3))*sl2)*d
         fvi(3) = fmass*sl1*d
         fvi(4) = fmass*sl2*d
      end if
c
      return
      end
