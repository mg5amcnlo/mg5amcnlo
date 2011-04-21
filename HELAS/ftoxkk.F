      subroutine ftoxkk(fo,tc,g,fmass,fwidth , fto)
c
c This subroutine computes an off-shell fermion wavefunction from a
c flowing-OUT external fermion and a KK tensor boson.
c
c input:
c       complex fo(6)          : flow-out fermion                   <fo|
c       complex tc(6,4)        : input    tensor                      t
c       real    g              : coupling constant                   gtf
c       real    fmass          : mass  of OUTPUT fermion f'
c       real    fwidth         : width of OUTPUT fermion f'
c
c output:
c       complex fvo(6)         : off-shell fermion             <fo,t,f'|
c
      implicit none
      double complex fo(6), tc(6,4), fto(6)
      double precision g, fmass, fwidth

      double complex k14p, k14m, k23, k23s, p14p, p14m
      double complex D1, D2, D3, D4, Tii, mTii, d
      double complex T11, T22, T33, T44, T12, T13, T14, T23, T24, T34
      double precision pf(4), k(4), pf2, m2
      integer i

      double precision rZero, rTwo
      parameter( rZero = 0.0d0, rTwo = 2.0d0 )
      double complex ci
      parameter( ci = ( 0.0d0, 1.0d0 ) )
c
      m2 = rTwo*fmass

      fto(5) = fo(5) + tc(5,1)
      fto(6) = fo(6) + tc(6,1)

      pf(1) = dreal(fto(5))
      pf(2) = dreal(fto(6))
      pf(3) = dimag(fto(6))
      pf(4) = dimag(fto(5))
      pf2 = pf(1)**2 - pf(2)**2 - pf(3)**2 - pf(4)**2

      k(1) = dreal(fo(5)) + pf(1)
      k(2) = dreal(fo(6)) + pf(2)
      k(3) = dimag(fo(6)) + pf(3)
      k(4) = dimag(fo(5)) + pf(4)

      k14p = dcmplx( k(1)+k(4), rZero )
      k14m = dcmplx( k(1)-k(4), rZero )
      k23  = dcmplx( k(2), k(3) )
      k23s = dconjg( k23 )
      p14p = dcmplx( pf(1)+pf(4), rZero )
      p14m = dcmplx( pf(1)-pf(4), rZero )

      T11 = rTwo*tc(1,1)
      T22 = rTwo*tc(2,2)
      T33 = rTwo*tc(3,3)
      T44 = rTwo*tc(4,4)
      T12 = tc(1,2) + tc(2,1)
      T13 = tc(1,3) + tc(3,1)
      T14 = tc(1,4) + tc(4,1)
      T23 = tc(2,3) + tc(3,2)
      T24 = tc(2,4) + tc(4,2)
      T34 = tc(3,4) + tc(4,3)

      D1 =   k(1)*(T11-T14) + k(2)*(T24-T12)
     &     + k(3)*(T34-T13) + k(4)*(T44-T14)

      D2 =   k(1)*(ci*T13-T12) + k(2)*(T22-ci*T23)
     &     + k(3)*(T23-ci*T33) + k(4)*(T24-ci*T34)

      D3 = - k(1)*(T12+ci*T13) + k(2)*(T22+ci*T23)
     &     + k(3)*(T23+ci*T33) + k(4)*(T24+ci*T34)

      D4 =   k(1)*(T11+T14) - k(2)*(T12+T24)
     &     - k(3)*(T13+T34) - k(4)*(T14+T44)

      Tii  = T11 - T22 - T33 - T44
      mTii = fmass*Tii

      if ( fmass.gt.rZero ) then
         d = - g/dcmplx( pf2-fmass**2, fmass*fwidth )
      else
         d = - g/dcmplx( pf2, rZero )
      end if

      fto(1) =   fo(1)*(p14p*D1 + fto(6)*D2)
     &         + fo(2)*(p14p*D3 + fto(6)*D4)
     &         + Tii*(- fo(1)*(p14p*k14m - fto(6)*k23s)
     &                + fo(2)*(p14p*k23  - fto(6)*k14p) )

      fto(2) =   fo(1)*(dconjg(fto(6))*D1 + p14m*D2)
     &         + fo(2)*(dconjg(fto(6))*D3 + p14m*D4)
     &         + Tii*(  fo(1)*(p14m*k23s - dconjg(fto(6))*k14m)
     &                - fo(2)*(p14m*k14p - dconjg(fto(6))*k23 ) )

      fto(3) =   fo(3)*(p14m*D4 + fto(6)*D2)
     &         + fo(4)*(p14m*D3 - fto(6)*D1)
     &         + Tii*(  fo(3)*(fto(6)*k23s - p14m*k14p)
     &                + fo(4)*(fto(6)*k14m - p14m*k23 ) )

      fto(4) = - fo(3)*(dconjg(fto(6))*D4 + p14p*D2)
     &         + fo(4)*(dconjg(fto(6))*D3 - p14p*D1)
     &         + Tii*(  fo(3)*(dconjg(fto(6))*k14p - p14p*k23s)
     &                + fo(4)*(dconjg(fto(6))*k23  - p14p*k14m) )

      if ( fmass.gt.rZero ) then
         fto(1) = fto(1) + fmass*( D4*fo(3) - D3*fo(4) )
         fto(2) = fto(2) + fmass*(-D2*fo(3) + D1*fo(4) )
         fto(3) = fto(3) + fmass*( D1*fo(1) + D3*fo(2) )
         fto(4) = fto(4) + fmass*( D2*fo(1) + D4*fo(2) )
         do i = 1,4
            fto(i) = fto(i) + mTii*m2*fo(i)
         end do
         fto(1) = fto(1) + mTii*(  fo(3)*(rTwo*p14p - k14p)
     &                           + fo(4)*(rTwo*fto(6) - k23) )
         fto(2) = fto(2) + mTii*(  fo(3)*(rTwo*dconjg(fto(6)) - k23s)
     &                           + fo(4)*(rTwo*p14m - k14m) )
         fto(3) = fto(3) + mTii*(  fo(1)*(rTwo*p14m - k14m)
     &                           + fo(2)*(k23 - rTwo*fto(6)) )
         fto(4) = fto(4) + mTii*(  fo(1)*(k23s - rTwo*dconjg(fto(6)))
     &                           + fo(2)*(rTwo*p14p - k14p) )
      end if

      do i = 1,4
         fto(i) = fto(i)*d
      end do
c
      return
      end
