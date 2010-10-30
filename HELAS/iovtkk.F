      subroutine iovtkk(fi,fo,vc,tc,g , vertex)
c
c This subroutine computes an amplitude of the three-point coupling of
c two gauge bosons and a Kaluza-Klein tensor boson.
c
c input:
c       complex fi(6)          : flow-in  fermion     SM |fi>
c       complex fo(6)          : flow-out fermion     SM <fo|
c       complex vc(6)          : vector               SM   v
c       complex tc(6,4)        : tensor               KK   t
c       real    g(2)           : coupling constant    -g(L,R)*kappa/4
c
c output:
c       complex vertex         : amplitude            gamma(fi,fo,vc,tc)
c     
      implicit none
      double complex fi(6), fo(6), vc(6), tc(6,4), vertex
      double precision g(2)

      double complex f13, f14, f23, f24, f31, f32, f41, f42
      double complex fs1L, fs1R, fs2L, fs2R, fs3L, fs3R, fs4L, fs4R
      double complex T12, T13, T14, T23, T24, T34

      double precision rZero, rTwo
      parameter( rZero = 0.0d0, rTwo = 2.0d0 )
      double complex ci
      parameter( ci = ( 0.0d0, 1.0d0 ) )
c
      f31 = fo(3)*fi(1)*g(1)
      f32 = fo(3)*fi(2)*g(1)
      f41 = fo(4)*fi(1)*g(1)
      f42 = fo(4)*fi(2)*g(1)

      fs1L =  f31 + f42
      fs2L = -f32 - f41
      fs3L = (f32 - f41)*ci
      fs4L = -f31 + f42

      if ( g(2).ne.rZero ) then
         f14 = fo(1)*fi(4)*g(2)
         f13 = fo(1)*fi(3)*g(2)
         f23 = fo(2)*fi(3)*g(2)
         f24 = fo(2)*fi(4)*g(2)
         fs1R =  f13 + f24
         fs2R =  f23 + f14
         fs3R = (f23 - f14)*ci
         fs4R =  f13 - f24
      end if

      T12 = tc(1,2) + tc(2,1)
      T13 = tc(1,3) + tc(3,1)
      T14 = tc(1,4) + tc(4,1)
      T23 = tc(2,3) + tc(3,2)
      T24 = tc(2,4) + tc(4,2)
      T34 = tc(3,4) + tc(4,3)

      if ( g(2).ne.rZero ) then

         vertex =  (fs1L + fs1R)*(  vc(1)*rTwo*tc(1,1)
     &                            - vc(2)*T12 - vc(3)*T13 - vc(4)*T14 )

     &           + (fs2L + fs2R)*(  vc(2)*rTwo*tc(2,2)
     &                            - vc(1)*T12 + vc(3)*T23 + vc(4)*T24 )

     &           + (fs3L + fs3R)*(  vc(3)*rTwo*tc(3,3)
     &                            - vc(1)*T13 + vc(2)*T23 + vc(4)*T34 )

     &           + (fs4L + fs4R)*(  vc(4)*rTwo*tc(4,4)
     &                            - vc(1)*T14 + vc(2)*T24 + vc(3)*T34 )

         vertex = vertex - rTwo*( tc(1,1)-tc(2,2)-tc(3,3)-tc(4,4) )
     &                         *(  (vc(1)+      vc(4))*(f31+f24)
     &                           + (vc(1)-      vc(4))*(f13+f42)
     &                           + (vc(2)+ci*vc(3))*(f41-f23)
     &                           + (vc(2)-ci*vc(3))*(f32-f14) )

      else

         vertex =  fs1L*(  vc(1)*rTwo*tc(1,1)
     &                   - vc(2)*T12 - vc(3)*T13 - vc(4)*T14 )

     &           + fs2L*(  vc(2)*rTwo*tc(2,2)
     &                   - vc(1)*T12 + vc(3)*T23 + vc(4)*T24 )

     &           + fs3L*(  vc(3)*rTwo*tc(3,3)
     &                   - vc(1)*T13 + vc(2)*T23 + vc(4)*T34 )

     &           + fs4L*(  vc(4)*rTwo*tc(4,4)
     &                   - vc(1)*T14 + vc(2)*T24 + vc(3)*T34 )

         vertex = vertex - rTwo*( tc(1,1)-tc(2,2)-tc(3,3)-tc(4,4) )
     &                         *(  (vc(1)+      vc(4))*f31
     &                           + (vc(1)-      vc(4))*f42
     &                           + (vc(2)+ci*vc(3))*f41
     &                           + (vc(2)-ci*vc(3))*f32 )

      end if
c
      return
      end
