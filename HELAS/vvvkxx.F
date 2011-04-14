      subroutine vvvkxx(wm,wp,tc,g, vertex)
c
c This subroutine computes an amplitude of the three-point coupling of
c two gauge bosons and a Kaluza-Klein tensor boson.
c
c input:
c       complex wm(6)          : vector               flow-in  V
c       complex wp(6)          : vector               flow-out V~
c       complex tc(6,4)        : tensor               KK mode T
c       complex g(1)           : coupling constant    -kappa/2
c       real    g(2)           : V boson mass          m_V
c
c output:
c       complex vertex         : amplitude            gamma(wm,wp,tc)
c     
      implicit none
      double complex wm(18), wp(18), tc(18), vertex,g(2)
      double precision vmass

      double complex T12, T13, T14, T23, T24, T34
      double complex V1V2, k1V2, k2V1
      double complex Tkk, TVV, Tk1V2, Tk2V1, dum
      double precision pwm(4), pwp(4), F

      integer i, j

      double complex cZero
      double precision rZero, rTwo
      parameter( rZero = 0.0d0, rTwo = 2.0d0 )
      parameter( cZero = ( 0.0d0, 0.0d0 ) )
c
      vmass = dreal(g(2))
      pwm(1) = dreal(wm(5))
      pwm(2) = dreal(wm(6))
      pwm(3) = dimag(wm(6))
      pwm(4) = dimag(wm(5))
      pwp(1) = dreal(wp(5))
      pwp(2) = dreal(wp(6))
      pwp(3) = dimag(wp(6))
      pwp(4) = dimag(wp(5))

      T12 = tc( 2) + tc( 5)
      T13 = tc( 3) + tc( 9)
      T14 = tc( 4) + tc(13)
      T23 = tc( 7) + tc(10)
      T24 = tc( 8) + tc(14)
      T34 = tc(12) + tc(15)

      V1V2 =  wm(1)*wp(1) -  wm(2)*wp(2) -  wm(3)*wp(3) -  wm(4)*wp(4)
      k1V2 = pwm(1)*wp(1) - pwm(2)*wp(2) - pwm(3)*wp(3) - pwm(4)*wp(4)
      k2V1 = pwp(1)*wm(1) - pwp(2)*wm(2) - pwp(3)*wm(3) - pwp(4)*wm(4)

      F = pwm(1)*pwp(1) - pwm(2)*pwp(2) - pwm(3)*pwp(3) - pwm(4)*pwp(4)
      if ( vmass.ne.rZero ) then
         F = F + vmass**2
      end if

      Tkk   = cZero
      TVV   = cZero
      Tk1V2 = cZero
      Tk2V1 = cZero

      do i = 1,4
         dum   = tc(i+4*(i-1))*pwm(i)
         Tkk   = Tkk   + dum*pwp(i)
         Tk1V2 = Tk1V2 + dum*wp(i)
         dum   = tc(i+4*(i-1))*wm(i)
         TVV   = TVV   + dum*wp(i)
         Tk2V1 = Tk2V1 + dum*pwp(i)
      end do

      Tkk   = rTwo*Tkk
      TVV   = rTwo*TVV
      Tk1V2 = rTwo*Tk1V2
      Tk2V1 = rTwo*Tk2V1

      Tkk = Tkk - T12*(pwm(1)*pwp(2) + pwm(2)*pwp(1))
     &          - T13*(pwm(1)*pwp(3) + pwm(3)*pwp(1))
     &          - T14*(pwm(1)*pwp(4) + pwm(4)*pwp(1))
     &          + T23*(pwm(2)*pwp(3) + pwm(3)*pwp(2))
     &          + T24*(pwm(2)*pwp(4) + pwm(4)*pwp(2))
     &          + T34*(pwm(3)*pwp(4) + pwm(4)*pwp(3))

      Tk1V2 = Tk1V2 - T12*(pwm(1)*wp(2) + pwm(2)*wp(1))
     &              - T13*(pwm(1)*wp(3) + pwm(3)*wp(1))
     &              - T14*(pwm(1)*wp(4) + pwm(4)*wp(1))
     &              + T23*(pwm(2)*wp(3) + pwm(3)*wp(2))
     &              + T24*(pwm(2)*wp(4) + pwm(4)*wp(2))
     &              + T34*(pwm(3)*wp(4) + pwm(4)*wp(3))

      TVV = TVV - T12*(wm(1)*wp(2) + wm(2)*wp(1))
     &          - T13*(wm(1)*wp(3) + wm(3)*wp(1))
     &          - T14*(wm(1)*wp(4) + wm(4)*wp(1))
     &          + T23*(wm(2)*wp(3) + wm(3)*wp(2))
     &          + T24*(wm(2)*wp(4) + wm(4)*wp(2))
     &          + T34*(wm(3)*wp(4) + wm(4)*wp(3))

      Tk2V1 = Tk2V1 - T12*(wm(1)*pwp(2) + wm(2)*pwp(1))
     &              - T13*(wm(1)*pwp(3) + wm(3)*pwp(1))
     &              - T14*(wm(1)*pwp(4) + wm(4)*pwp(1))
     &              + T23*(wm(2)*pwp(3) + wm(3)*pwp(2))
     &              + T24*(wm(2)*pwp(4) + wm(4)*pwp(2))
     &              + T34*(wm(3)*pwp(4) + wm(4)*pwp(3))

      vertex =  (tc(1)-tc(6)-tc(11)-tc(16))*( k1V2*k2V1 - V1V2*F )
     &        + F*TVV + V1V2*Tkk - k2V1*Tk1V2 - k1V2*Tk2V1

C      vertex = F*TVV + V1V2*Tkk - k2V1*Tk1V2 - k1V2*Tk2V1

      vertex = vertex * g(1)
c
      return
      end
