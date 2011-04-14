      subroutine fvidmx(fi,vc,gc,fmass,fwidth , fvi)
c
c This subroutine computes a dipole moment off-shell fermion
c wavefunction from a flowing-IN external fermion and a vector boson.
c
c input:
c       complex fi(6)          : flow-in  fermion                   |fi>
c       complex vc(6)          : input    vector                      v
c       complex gc(2,2)        : coupling constants                  gvf
c                              : first index is L,R as normal
c                              : second index is EDM,-MDM
c       real    fmass          : mass  of output fermion f'
c       real    fwidth         : width of output fermion f'
c
c output:
c       complex fvi(6)         : off-shell fermion             |f',v,fi>
c
      implicit none
      double complex fi(6), vc(6), fvi(6), sl1, sl2, sr1, sr2, d
      double complex gc(2,2), gL, gR
      double precision pf(0:3), fmass, fwidth, pf2

      double complex kvc21, kvc31, kvc41, kvc32, kvc42, kvc43
      double precision k(1:4)
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
         write(stdo,*) ' helas-warn  : fi in fvidmx is zero spinor'
      endif
      if ( abs(fi(5))+abs(fi(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : fi in fvidmx has zero momentum'
      endif
      if ( abs(vc(1))+abs(vc(2))+abs(vc(3))+abs(vc(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : vc in fvidmx is zero vector'
      endif
      if ( abs(vc(5))+abs(vc(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : vc in fvidmx has zero momentum'
      endif
      if ( gc(1,1).eq.cZero .and. gc(2,1).eq.cZero .and.
     &     gc(1,2).eq.cZero .and. gc(2,2).eq.cZero      ) then
         write(stdo,*)
     &        ' helas-error : gc in fvidmx is zero coupling'
      endif
      if ( fmass.lt.rZero ) then
         write(stdo,*) ' helas-error : fmass in fvidmx is negative'
         write(stdo,*) '             : fmass = ',fmass
      endif
      if ( fwidth.lt.rZero ) then
         write(stdo,*) ' helas-error : fwidth in fvidmx is negative'
         write(stdo,*) '             : fwidth = ',fwidth
      endif
#endif

      gL = -gc(1,1) + cImag*gc(1,2)
      gR =  gc(2,1) + cImag*gc(2,2)

c k in vertex formula defined as (pi - po)
      k(1) = dble( vc(5))
      k(2) = dble( vc(6))
      k(3) = dimag(vc(6))
      k(4) = dimag(vc(5))

      kvc21 = (k(2)*vc(1) - k(1)*vc(2))*cImag
      kvc31 =  k(3)*vc(1) - k(1)*vc(3)
      kvc41 = (k(4)*vc(1) - k(1)*vc(4))*cImag
      kvc32 =  k(3)*vc(2) - k(2)*vc(3)
      kvc42 = (k(4)*vc(2) - k(2)*vc(4))*cImag
      kvc43 =  k(4)*vc(3) - k(3)*vc(4)

      fvi(5) = fi(5) - vc(5)
      fvi(6) = fi(6) - vc(6)

      pf(0) = dble( fvi(5))
      pf(1) = dble( fvi(6))
      pf(2) = dimag(fvi(6))
      pf(3) = dimag(fvi(5))
      pf2 = pf(0)**2-(pf(1)**2+pf(2)**2+pf(3)**2)

#ifdef HELAS_CHECK
      if ( abs(fvi(5))+abs(fvi(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : fvi in fvidmx has zero momentum'
      endif
      if ( fwidth.eq.rZero .and. pf2.eq.fmass**2 ) then
         write(stdo,*)
     &        ' helas-error : fvi in fvidmx is on fmass pole'
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

      sl1 = gL*(  fi(1)*(kvc41 + kvc32)
     &          + fi(2)*(kvc42 + kvc21 + kvc43 + kvc31) )
      sl2 = gL*(- fi(1)*(kvc42 - kvc21 - kvc43 + kvc31)
     &          - fi(2)*(kvc41 + kvc32)                 )

      if ( gc(2,1).ne.cZero .or.
     &     gc(2,2).ne.cZero      ) then
         sr1 = gR*(- fi(3)*(kvc41 - kvc32)
     &             + fi(4)*(kvc42 - kvc21 + kvc43 - kvc31) )
         sr2 = gR*(- fi(3)*(kvc42 + kvc21 - kvc43 - kvc31)
     &             + fi(4)*(kvc41 - kvc32)                 )

         fvi(1) = ( (pf(0)-pf(3))*sr1 - dconjg(fvi(6))*sr2
     &             + fmass*sl1                             )*d
         fvi(2) = (      - fvi(6)*sr1 +  (pf(0)+pf(3))*sr2
     &             + fmass*sl2                             )*d
         fvi(3) = ( (pf(0)+pf(3))*sl1 + dconjg(fvi(6))*sl2
     &             + fmass*sr1                             )*d
         fvi(4) = (        fvi(6)*sl1 +  (pf(0)-pf(3))*sl2
     &             + fmass*sr2                             )*d

      else
         fvi(1) = fmass*sl1*d
         fvi(2) = fmass*sl2*d
         fvi(3) = ( (pf(0)+pf(3))*sl1 + dconjg(fvi(6))*sl2 )*d
         fvi(4) = (        fvi(6)*sl1 +  (pf(0)-pf(3))*sl2 )*d
      end if
c
      return
      end
