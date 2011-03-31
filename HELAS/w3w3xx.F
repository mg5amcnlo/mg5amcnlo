      subroutine w3w3xx(wm,w31,wp,w32,g31,g32, vertex)
c
c This subroutine computes an amplitude of the four-point coupling of
c the W-, W+ and two W3/Z/A.
c If one sets wmass=0.0, then the gggg vertex is given
c (see sect 2.9.1 of the manual).
c
c input:
c       complex wm(0:3)        : flow-out W-                         wm
c       complex w31(0:3)       : first    W3/Z/A                     w31
c       complex wp(0:3)        : flow-out W+                         wp
c       complex w32(0:3)       : second   W3/Z/A                     w32
c       real    g31            : coupling of w31 with W-/W+
c       real    g32            : coupling of w32 with W-/W+
c                                                  (see the table below)
c       real    wmass          : mass  of W
c       real    wwidth         : width of W
c
c the possible sets of the inputs are as follows:
c   -------------------------------------------
c   |  wm  |  w31 |  wp  |  w32 |  g31 |  g32 |
c   -------------------------------------------
c   |  W-  |  W3  |  W+  |  W3  |  gw  |  gw  |
c   |  W-  |  W3  |  W+  |  Z   |  gw  | gwwz |
c   |  W-  |  W3  |  W+  |  A   |  gw  | gwwa |
c   |  W-  |  Z   |  W+  |  Z   | gwwz | gwwz |
c   |  W-  |  Z   |  W+  |  A   | gwwz | gwwa |
c   |  W-  |  A   |  W+  |  A   | gwwa | gwwa |
c   -------------------------------------------
c where all the bosons are defined by the flowing-OUT quantum number.
c
c output:
c       complex vertex         : amplitude          gamma(wm,w31,wp,w32)
c     
      implicit none
      double complex wm(6),w31(6),wp(6),w32(6),vertex
      double complex dv1(0:3),dv2(0:3),dv3(0:3),dv4(0:3),dvertx
      double complex v12,v13,v14,v23,v24,v34
      double precision pwm(0:3),pw31(0:3),pwp(0:3),pw32(0:3)
      double precision g31,g32

      double precision rZero, rOne, rTwo
      parameter( rZero = 0.0d0, rOne = 1.0d0, rTwo = 2.0d0 )

#ifdef HELAS_CHECK
      double precision pm
      double precision epsi
      parameter( epsi = 4.0d-5 )
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
      pw31(0) = dble( w31(5))
      pw31(1) = dble( w31(6))
      pw31(2) = dimag(w31(6))
      pw31(3) = dimag(w31(5))
      pw32(0) = dble( w32(5))
      pw32(1) = dble( w32(6))
      pw32(2) = dimag(w32(6))
      pw32(3) = dimag(w32(5))

#ifdef HELAS_CHECK
      if (  abs(wm(1))+abs(wm(2))
     &     +abs(wm(3))+abs(wm(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : wm in w3w3xx is zero vector'
      endif
      if ( abs(wm(5))+abs(wm(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : wm in w3w3xx has zero momentum'
      endif
      if (  abs(w31(1))+abs(w31(2))
     &     +abs(w31(3))+abs(w31(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : w31 in w3w3xx is zero vector'
      endif
      if ( abs(w31(5))+abs(w31(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : w31 in w3w3xx has zero momentum'
      endif
      if (  abs(wp(1))+abs(wp(2))
     &     +abs(wp(3))+abs(wp(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : wp in w3w3xx is zero vector'
      endif
      if ( abs(wp(5))+abs(wp(5)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : wp in w3w3xx has zero momentum'
      endif
      if (  abs(w32(1))+abs(w32(2))
     &     +abs(w32(3))+abs(w32(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : w32 in w3w3xx is zero vector'
      endif
      if ( abs(w32(5))+abs(w32(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : w32 in w3w3xx has zero momentum'
      endif
      pm = max( abs(pwm(0)),abs(pw31(0)),abs(pwp(0)),abs(pw32(0)),
     &          abs(pwm(1)),abs(pw31(1)),abs(pwp(1)),abs(pw32(1)),
     &          abs(pwm(2)),abs(pw31(2)),abs(pwp(2)),abs(pw32(2)),
     &          abs(pwm(3)),abs(pw31(3)),abs(pwp(3)),abs(pw32(3)) )
      if (  abs(wm(5)+w31(5)+wp(5)+w32(5))
     &     +abs(wm(6)+w31(6)+wp(6)+w32(6)).ge.pm*epsi ) then
         write(stdo,*)
     &        ' helas-error : wm,w31,wp,w32 in w3w3xx'
         write(stdo,*)
     &        '             : have not balanced momenta'
      endif
      if ( g31.eq.rZero ) then
         write(stdo,*) ' helas-error : g31 in w3w3xx is zero coupling'
      endif
      if ( g32.eq.rZero ) then
         write(stdo,*) ' helas-error : g32 in w3w3xx is zero coupling'
      endif
      if ( g31.lt.rZero ) then
         write(stdo,*)
     &        ' helas-warn  : g31 in w3w3xx is non-standard coupling'
         write(stdo,*) 
     &        '             : g31 = ',g31
      endif
      if ( g32.lt.rZero ) then
         write(stdo,*)
     &        ' helas-warn  : g32 in w3w3xx is non-standard coupling'
         write(stdo,*)
     &        '             : g32 = ',g32
      endif
#endif

      dv1(0) = dcmplx(wm(1))
      dv1(1) = dcmplx(wm(2))
      dv1(2) = dcmplx(wm(3))
      dv1(3) = dcmplx(wm(4))
      dv2(0) = dcmplx(w31(1))
      dv2(1) = dcmplx(w31(2))
      dv2(2) = dcmplx(w31(3))
      dv2(3) = dcmplx(w31(4))
      dv3(0) = dcmplx(wp(1))
      dv3(1) = dcmplx(wp(2))
      dv3(2) = dcmplx(wp(3))
      dv3(3) = dcmplx(wp(4))
      dv4(0) = dcmplx(w32(1))
      dv4(1) = dcmplx(w32(2))
      dv4(2) = dcmplx(w32(3))
      dv4(3) = dcmplx(w32(4))

      v12 = dv1(0)*dv2(0)-dv1(1)*dv2(1)-dv1(2)*dv2(2)-dv1(3)*dv2(3)
      v13 = dv1(0)*dv3(0)-dv1(1)*dv3(1)-dv1(2)*dv3(2)-dv1(3)*dv3(3)
      v14 = dv1(0)*dv4(0)-dv1(1)*dv4(1)-dv1(2)*dv4(2)-dv1(3)*dv4(3)
      v23 = dv2(0)*dv3(0)-dv2(1)*dv3(1)-dv2(2)*dv3(2)-dv2(3)*dv3(3)
      v24 = dv2(0)*dv4(0)-dv2(1)*dv4(1)-dv2(2)*dv4(2)-dv2(3)*dv4(3)
      v34 = dv3(0)*dv4(0)-dv3(1)*dv4(1)-dv3(2)*dv4(2)-dv3(3)*dv4(3)

      dvertx = v12*v34 + v14*v23 - rTwo*v13*v24
      
      vertex = dcmplx( dvertx ) * (g31*g32)
c
      return
      end
