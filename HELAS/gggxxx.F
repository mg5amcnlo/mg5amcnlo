      subroutine gggxxx(wm,wp,w3,g , vertex)
c
c This subroutine computes an amplitude of the three-point coupling of
c the gauge bosons.
c
c input:
c       complex wm(6)          : vector               flow-out W-
c       complex wp(6)          : vector               flow-out W+
c       complex w3(6)          : vector               j3 or A    or Z
c       real    g              : coupling constant    gw or gwwa or gwwz
c
c output:
c       complex vertex         : amplitude               gamma(wm,wp,w3)
c
      implicit none
      double complex wm(6),wp(6),w3(6),vertex
      double complex xv1,xv2,xv3,v12,v23,v31
      double complex p12,p13,p21,p23,p31,p32
      double precision pwm(0:3),pwp(0:3),pw3(0:3),g

      double precision rZero, rTenth
      parameter( rZero = 0.0d0, rTenth = 0.1d0 )

#ifdef HELAS_CHECK
      double precision pm
      double precision epsi
      parameter( epsi = 4.0d-5 )
      double complex cZero
      parameter( cZero = ( 0.0d0, 0.0d0 ) )
      integer stdo
      parameter( stdo = 6 )
#endif
c
      pwm(0) = dble( wm(5))
      pwm(1) = dble( wm(6))
      pwm(2) = dimag(wm(6))
      pwm(3) = dimag(wm(5))
      pwp(0) = dble( wp(5))
      pwp(1) = dble( wp(6))
      pwp(2) = dimag(wp(6))
      pwp(3) = dimag(wp(5))
      pw3(0) = dble( w3(5))
      pw3(1) = dble( w3(6))
      pw3(2) = dimag(w3(6))
      pw3(3) = dimag(w3(5))

#ifdef HELAS_CHECK
      if (  abs(wm(1))+abs(wm(2))
     &     +abs(wm(3))+abs(wm(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : wm in gggxxx is zero vector'
      endif
      if ( abs(wm(5))+abs(wm(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : wm in gggxxx has zero momentum'
      endif
      if (  abs(wp(1))+abs(wp(2))
     &     +abs(wp(3))+abs(wp(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : wp in gggxxx is zero vector'
      endif
      if ( abs(wp(5))+abs(wp(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : wp in gggxxx has zero momentum'
      endif
      if (  abs(w3(1))+abs(w3(2))
     &    +abs(w3(3))+abs(w3(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : w3 in gggxxx is zero vector'
      endif
      if ( abs(w3(5))+abs(w3(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : w3 in gggxxx has zero momentum'
      endif
      pm = max( abs(pwm(0)),abs(pwp(0)),abs(pw3(0)),
     &          abs(pwm(1)),abs(pwp(1)),abs(pw3(1)),
     &          abs(pwm(2)),abs(pwp(2)),abs(pw3(2)),
     &          abs(pwm(3)),abs(pwp(3)),abs(pw3(3)) )
      if ( abs(wm(5)+wp(5)+w3(5))+abs(wm(6)+wp(6)+w3(6))
     &                                           .ge.pm*epsi) then
         write(stdo,*)
     &        ' helas-error : wm,wp,w3 in gggxxx'
         write(stdo,*)
     &        '             : have not balanced momenta'
      endif
      if ( g.eq.rZero ) then
         write(stdo,*) ' helas-error : g in gggxxx is zero coupling'
      endif
#endif

      v12 = wm(1)*wp(1)-wm(2)*wp(2)-wm(3)*wp(3)-wm(4)*wp(4)
      v23 = wp(1)*w3(1)-wp(2)*w3(2)-wp(3)*w3(3)-wp(4)*w3(4)
      v31 = w3(1)*wm(1)-w3(2)*wm(2)-w3(3)*wm(3)-w3(4)*wm(4)
      xv1 = rZero
      xv2 = rZero
      xv3 = rZero
      if ( abs(wm(1)).ne.rZero ) then
         if ( abs(wm(1)).ge.max(abs(wm(2)),abs(wm(3)),abs(wm(4)))
     &        *rTenth )
     &      xv1 = pwm(0)/wm(1)
      endif
      if ( abs(wp(1)).ne.rZero ) then
         if ( abs(wp(1)).ge.max(abs(wp(2)),abs(wp(3)),abs(wp(4)))
     &        *rTenth )
     &      xv2 = pwp(0)/wp(1)
      endif
      if ( abs(w3(1)).ne.rZero ) then
         if ( abs(w3(1)).ge.max(abs(w3(2)),abs(w3(3)),abs(w3(4)))
     &        *rTenth )
     &      xv3 = pw3(0)/w3(1)
      endif

      p12 = (pwm(0)-xv1*wm(1))*wp(1)-(pwm(1)-xv1*wm(2))*wp(2)
     &     -(pwm(2)-xv1*wm(3))*wp(3)-(pwm(3)-xv1*wm(4))*wp(4)
      p13 = (pwm(0)-xv1*wm(1))*w3(1)-(pwm(1)-xv1*wm(2))*w3(2)
     &     -(pwm(2)-xv1*wm(3))*w3(3)-(pwm(3)-xv1*wm(4))*w3(4)
      p21 = (pwp(0)-xv2*wp(1))*wm(1)-(pwp(1)-xv2*wp(2))*wm(2)
     &     -(pwp(2)-xv2*wp(3))*wm(3)-(pwp(3)-xv2*wp(4))*wm(4)
      p23 = (pwp(0)-xv2*wp(1))*w3(1)-(pwp(1)-xv2*wp(2))*w3(2)
     &     -(pwp(2)-xv2*wp(3))*w3(3)-(pwp(3)-xv2*wp(4))*w3(4)
      p31 = (pw3(0)-xv3*w3(1))*wm(1)-(pw3(1)-xv3*w3(2))*wm(2)
     &     -(pw3(2)-xv3*w3(3))*wm(3)-(pw3(3)-xv3*w3(4))*wm(4)
      p32 = (pw3(0)-xv3*w3(1))*wp(1)-(pw3(1)-xv3*w3(2))*wp(2)
     &     -(pw3(2)-xv3*w3(3))*wp(3)-(pw3(3)-xv3*w3(4))*wp(4)

      vertex = -(v12*(p13-p23)+v23*(p21-p31)+v31*(p32-p12))*g
c
      return
      end
