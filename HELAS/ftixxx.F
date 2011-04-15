      subroutine ftixxx(fi,tc,gt,fmass,fwidth , fti)
c
c This subroutine computes an off-shell fermion wavefunction from a
c flowing-IN external fermion and a tensor boson.
c
c input:
c       complex fi(6)          : flow-in  fermion                   |fi>
c       complex tc(18)         : input    tensor                       T
c       complex gt             : coupling constant       gtf=-1/Lambda/4
c       real    fmass          : mass  of output fermion f'
c       real    fwidth         : width of output fermion f'
c
c output:
c       complex fti(6)         : off-shell fermion             |f':T,fi>
c
c- by Q.Li - OCT. 2006
c
      implicit none
      double complex fi(6), tc(18), gt, fti(6)
      double precision fmass, fwidth

      double complex ft(6,4)
      double complex k14p, k14m, k23, k23s, p14p, p14m
      double complex D1, D2, D3, D4, Tii, mTii, d
      double complex T11, T22, T33, T44, T12, T13, T14, T23, T24, T34
      double precision pf(4), k(4), pf2, m2
      integer i
      
      double precision rZero, rTwo
      parameter( rZero = 0.0d0, rTwo = 2.0d0 )
      double complex ci
      parameter( ci = ( 0.0d0, 1.0d0 ) )


      m2 = rTwo*fmass
      ft(1,1) = tc(1)
      ft(1,2) = tc(2)
      ft(1,3) = tc(3)
      ft(1,4) = tc(4)
      ft(2,1) = tc(5)
      ft(2,2) = tc(6)
      ft(2,3) = tc(7)
      ft(2,4) = tc(8)
      ft(3,1) = tc(9)
      ft(3,2) = tc(10)
      ft(3,3) = tc(11)
      ft(3,4) = tc(12)
      ft(4,1) = tc(13)
      ft(4,2) = tc(14)
      ft(4,3) = tc(15)
      ft(4,4) = tc(16)
      ft(5,1) = tc(17)
      ft(6,1) = tc(18)

      fti(5) = fi(5) - ft(5,1)
      fti(6) = fi(6) - ft(6,1)

      pf(1) = dreal(fti(5))
      pf(2) = dreal(fti(6))
      pf(3) = dimag(fti(6))
      pf(4) = dimag(fti(5))
      pf2 = pf(1)**2 - pf(2)**2 - pf(3)**2 - pf(4)**2

      k(1) = dreal(fi(5)) + pf(1)
      k(2) = dreal(fi(6)) + pf(2)
      k(3) = dimag(fi(6)) + pf(3)
      k(4) = dimag(fi(5)) + pf(4)

      k14p = dcmplx( k(1)+k(4), rZero )
      k14m = dcmplx( k(1)-k(4), rZero )
      k23  = dcmplx( k(2), k(3) )
      k23s = dconjg( k23 )
      p14p = dcmplx( pf(1)+pf(4), rZero )
      p14m = dcmplx( pf(1)-pf(4), rZero )

      T11 = rTwo*ft(1,1)
      T22 = rTwo*ft(2,2)
      T33 = rTwo*ft(3,3)
      T44 = rTwo*ft(4,4)
      T12 = ft(1,2) + ft(2,1)
      T13 = ft(1,3) + ft(3,1)
      T14 = ft(1,4) + ft(4,1)
      T23 = ft(2,3) + ft(3,2)
      T24 = ft(2,4) + ft(4,2)
      T34 = ft(3,4) + ft(4,3)

      D1 =   k(1)*(T11-T14) - k(2)*(T12-T24)
     &     - k(3)*(T13-T34) - k(4)*(T14-T44)

      D2 = - k(1)*(T12-ci*T13) + k(2)*(T22-ci*T23)
     &     + k(3)*(T23-ci*T33) + k(4)*(T24-ci*T34)

      D3 = - k(1)*(T12+ci*T13) + k(2)*(T22+ci*T23)
     &     + k(3)*(T23+ci*T33) + k(4)*(T24+ci*T34)

      D4 =   k(1)*(T11+T14) - k(2)*(T12+T24)
     &     - k(3)*(T13+T34) - k(4)*(T14+T44)

      Tii  = T11 - T22 - T33 - T44
      mTii = fmass*Tii

      if ( fmass.gt.rZero ) then
         d = - gt/dcmplx( pf2-fmass**2, fmass*fwidth )
      else
         d = - gt/dcmplx( pf2, rZero )
      end if

      fti(1) =   fi(1)*(p14m*D4 + dconjg(fti(6))*D3)
     &         - fi(2)*(p14m*D2 + dconjg(fti(6))*D1)
     &         + Tii*(  fi(1)*(dconjg(fti(6))*k23  - p14m*k14p)
     &                + fi(2)*(dconjg(fti(6))*k14m - p14m*k23s) )

      fti(2) = - fi(1)*(fti(6)*D4 + p14p*D3)
     &         + fi(2)*(fti(6)*D2 + p14p*D1)
     &         + Tii*(  fi(1)*(fti(6)*k14p - p14p*k23 )
     &                + fi(2)*(fti(6)*k23s - p14p*k14m) )

      fti(3) =   fi(3)*(p14p*D1 + dconjg(fti(6))*D3)
     &         + fi(4)*(p14p*D2 + dconjg(fti(6))*D4)
     &         + Tii*(  fi(3)*(dconjg(fti(6))*k23  - p14p*k14m)
     &                - fi(4)*(dconjg(fti(6))*k14p - p14p*k23s) )

      fti(4) =   fi(3)*(fti(6)*D1 + p14m*D3)
     &         + fi(4)*(fti(6)*D2 + p14m*D4)
     &         + Tii*(  fi(3)*(p14m*k23  - fti(6)*k14m)
     &                - fi(4)*(p14m*k14p - fti(6)*k23s) )

      if ( fmass.gt.rZero ) then
         fti(1) = fti(1) + fmass*( D1*fi(3) + D2*fi(4) )
         fti(2) = fti(2) + fmass*( D3*fi(3) + D4*fi(4) )
         fti(3) = fti(3) + fmass*( D4*fi(1) - D2*fi(2) )
         fti(4) = fti(4) + fmass*(-D3*fi(1) + D1*fi(2) )
         do i = 1,4
            fti(i) = fti(i) + mTii*m2*fi(i)
         end do
         fti(1) = fti(1) + mTii*(  fi(3)*(rTwo*p14m - k14m)
     &                           + fi(4)*(k23 - rTwo*dconjg(fti(6))) )
         fti(2) = fti(2) + mTii*(  fi(3)*(k23 - rTwo*fti(6))
     &                           + fi(4)*(rTwo*p14p - k14p) )
         fti(3) = fti(3) + mTii*(  fi(1)*(rTwo*p14p - k14p)
     &                           + fi(2)*(rTwo*dconjg(fti(6)) - k23s) )
         fti(4) = fti(4) + mTii*(  fi(1)*(rTwo*fti(6) - k23)
     &                           + fi(2)*(rTwo*p14m - k14m) )
      end if

      do i = 1,4
         fti(i) = fti(i)*d
      end do

      return
      end
