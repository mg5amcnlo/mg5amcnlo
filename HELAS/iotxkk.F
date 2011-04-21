      subroutine iotxkk(fi,fo,tc,g,fmass , vertex)
c
c This subroutine computes an amplitude of the fermion-fermion-vector
c coupling.
c
c input:
c       complex fi(6)          : flow-in  fermion                   |fi>
c       complex fo(6)          : flow-out fermion                   <fo|
c       complex tc(6,4)        : input    tensor                      t
c       real    g              : coupling constant                 -kappa/8
c       real    fmass          : fermion mass                        m_f
c
c output:
c       complex vertex         : amplitude                        <fo|t|fi>
c     
      implicit none
      double complex fi(6), fo(6), tc(6,4), vertex
      double precision g, fmass

      double complex k23, k23s, D1, D2, D3, D4, Tii
      double complex T11, T22, T33, T44, T12, T13, T14, T23, T24, T34
      double complex f13, f14, f23, f24, f31, f32, f41, f42
      double precision k(4), k14p, k14m, m2

      double precision rZero, rTwo
      parameter( rZero = 0.0d0, rTwo = 2.0d0 )
      double complex ci
      parameter( ci = ( 0.0d0, 1.0d0 ) )
c
      m2 = rTwo*fmass

      k(1) = dreal(fi(5)+fo(5))
      k(2) = dreal(fi(6)+fo(6))
      k(3) = dimag(fi(6)+fo(6))
      k(4) = dimag(fi(5)+fo(5))
      k23  = dcmplx( k(2),k(3) )
      k23s = dconjg( k23 )
      k14p = k(1) + k(4)
      k14m = k(1) - k(4)

      f13 = fo(1)*fi(3)
      f14 = fo(1)*fi(4)
      f23 = fo(2)*fi(3)
      f24 = fo(2)*fi(4)
      f31 = fo(3)*fi(1)
      f32 = fo(3)*fi(2)
      f41 = fo(4)*fi(1)
      f42 = fo(4)*fi(2)

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

      D1 =   k(1)*(T11-T14) - k(2)*(T12-T24)
     &     - k(3)*(T13-T34) - k(4)*(T14-T44)

      D2 = - k(1)*(T12-ci*T13) + k(2)*(T22-ci*T23)
     &     + k(3)*(T23-ci*T33) + k(4)*(T24-ci*T34)

      D3 = - k(1)*(T12+ci*T13) + k(2)*(T22+ci*T23)
     &     + k(3)*(T23+ci*T33) + k(4)*(T24+ci*T34)

      D4 =   k(1)*(T11+T14) - k(2)*(T12+T24)
     &     - k(3)*(T13+T34) - k(4)*(T14+T44)

      Tii = T11 - T22 - T33 - T44

      vertex = D1*(f13+f42) + D2*(f14-f32) + D3*(f23-f41) + D4*(f24+f31)

      vertex = vertex + Tii*( - k14p*(f24+f31) - k14m*(f13+f42)
     &                        +  k23*(f23-f41) + k23s*(f14-f32) )

      if ( fmass.ne.rZero ) then
         vertex = vertex + m2*Tii*(  fo(1)*fi(1) + fo(2)*fi(2)
     &                             + fo(3)*fi(3) + fo(4)*fi(4) )
      end if

      vertex = vertex * g
c
      return
      end
