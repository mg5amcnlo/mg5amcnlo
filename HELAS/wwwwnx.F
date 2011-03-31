      subroutine wwwwnx(wm1,wp1,wm2,wp2,gwwz,gwwa , vertex)

c
c This subroutine computes an amplitude of the four-point W-/W+ coupling.
c
c input:
c       complex wm1(0:3)       : first  flow-out W-                  wm1
c       complex wp1(0:3)       : first  flow-out W+                  wp1
c       complex wm2(0:3)       : second flow-out W-                  wm2
c       complex wp2(0:3)       : second flow-out W+                  wp2
c       real    gwwa           : coupling constant of W and A       gwwa
c       real    gwwz           : coupling constant of W and Z       gwwz
c       real    zmass          : mass  of Z
c       real    zwidth         : width of Z
c
c output:
c       complex vertex         : amplitude        gamma(wm1,wp1,wm2,wp2)
c     
      implicit none
      double complex wm1(6),wp1(6),wm2(6),wp2(6),vertex
      double complex dv1(0:3),dv2(0:3),dv3(0:3),dv4(0:3),dvertx
      double complex v12,v13,v14,v23,v24,v34
      double precision pwm1(0:3),pwp1(0:3),pwm2(0:3),pwp2(0:3)
      double precision gwwa,gwwz,gtemp

      double precision rZero, rOne, rTwo
      parameter( rZero = 0.0d0, rOne = 1.0d0, rTwo = 2.0d0 )

#ifdef HELAS_CHECK
      double precision pm
      double precision epsi
      parameter( epsi = 2.0d-5 )
      integer stdo
      parameter( stdo = 6 )
#endif
c

c Benj's modif in order to have FR running
      gtemp=rZero
      if(gwwa.eq.rZero) gtemp=gwwz
      if(gwwz.eq.rZero) gtemp=gwwa
c End of Benj'S modif


      pwm1(0) = dble( wm1(5))
      pwm1(1) = dble( wm1(6))
      pwm1(2) = dimag(wm1(6))
      pwm1(3) = dimag(wm1(5))
      pwp1(0) = dble( wp1(5))
      pwp1(1) = dble( wp1(6))
      pwp1(2) = dimag(wp1(6))
      pwp1(3) = dimag(wp1(5))
      pwm2(0) = dble( wm2(5))
      pwm2(1) = dble( wm2(6))
      pwm2(2) = dimag(wm2(6))
      pwm2(3) = dimag(wm2(5))
      pwp2(0) = dble( wp2(5))
      pwp2(1) = dble( wp2(6))
      pwp2(2) = dimag(wp2(6))
      pwp2(3) = dimag(wp2(5))

#ifdef HELAS_CHECK
      if (  abs(wm1(1))+abs(wm1(2))
     &     +abs(wm1(3))+abs(wm1(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : wm1 in wwwwxx is zero vector'
      endif
      if ( abs(wm1(5))+abs(wm1(5)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : wm1 in wwwwxx has zero momentum'
      endif
      if (  abs(wp1(1))+abs(wp1(2))
     &     +abs(wp1(3))+abs(wp1(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : wp1 in wwwwxx is zero vector'
      endif
      if ( abs(wp1(5))+abs(wp1(5)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : wp1 in wwwwxx has zero momentum'
      endif
      if (  abs(wm2(1))+abs(wm2(2))
     &     +abs(wm2(3))+abs(wm2(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : wm2 in wwwwxx is zero vector'
      endif
      if ( abs(wm2(5))+abs(wm2(5)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : wm2 in wwwwxx has zero momentum'
      endif
      if (  abs(wp2(1))+abs(wp2(2))
     &     +abs(wp2(3))+abs(wp2(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : wp2 in wwwwxx is zero vector'
      endif
      if ( abs(wp2(5))+abs(wp2(5)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : wp2 in wwwwxx has zero momentum'
      endif
      pm = max( abs(pwm1(0)),abs(pwp1(0)),abs(pwm2(0)),abs(pwp2(0)),
     &          abs(pwm1(1)),abs(pwp1(1)),abs(pwm2(1)),abs(pwp2(1)),
     &          abs(pwm1(2)),abs(pwp1(2)),abs(pwm2(2)),abs(pwp2(2)),
     &          abs(pwm1(3)),abs(pwp1(3)),abs(pwm2(3)),abs(pwp2(3)) )
      if (  abs(wm1(5)+wp1(5)+wm2(5)+wp2(5))
     &     +abs(wm1(6)+wp1(6)+wm2(6)+wp2(6)).ge.pm*epsi ) then
         write(stdo,*)
     &        ' helas-error : wm1,wp1,wm2,wp2 in wwwwxx'
         write(stdo,*)
     &        '             : have not balanced momenta'
      endif

c     Neil edited the following to allow 3-site couplings.
c      if ( gwwa.eq.rZero ) then
c         write(stdo,*) ' helas-error : gwwa in wwwwxx is zero coupling'
c      endif
c      if ( gwwz.eq.rZero ) then
c         write(stdo,*)
c     &        ' helas-error : gwwz in wwwwxx is zero coupling'
c      endif
c      if ( gwwa.lt.rZero .or. gwwa.ge.gwwz ) then
c         write(stdo,*)
c     &  ' helas-warn  : gwwa/gwwz in wwwwxx are non-standard couplings'
c         write(stdo,*) 
c     &  '             : gwwa = ',gwwa,'  gwwz = ',gwwz
c      endif
c     End Neil's edit.
#endif

      dv1(0) = dcmplx(wm1(1))
      dv1(1) = dcmplx(wm1(2))
      dv1(2) = dcmplx(wm1(3))
      dv1(3) = dcmplx(wm1(4))
      dv2(0) = dcmplx(wp1(1))
      dv2(1) = dcmplx(wp1(2))
      dv2(2) = dcmplx(wp1(3))
      dv2(3) = dcmplx(wp1(4))
      dv3(0) = dcmplx(wm2(1))
      dv3(1) = dcmplx(wm2(2))
      dv3(2) = dcmplx(wm2(3))
      dv3(3) = dcmplx(wm2(4))
      dv4(0) = dcmplx(wp2(1))
      dv4(1) = dcmplx(wp2(2))
      dv4(2) = dcmplx(wp2(3))
      dv4(3) = dcmplx(wp2(4))

      v12 = dv1(0)*dv2(0)-dv1(1)*dv2(1)-dv1(2)*dv2(2)-dv1(3)*dv2(3)
      v13 = dv1(0)*dv3(0)-dv1(1)*dv3(1)-dv1(2)*dv3(2)-dv1(3)*dv3(3)
      v14 = dv1(0)*dv4(0)-dv1(1)*dv4(1)-dv1(2)*dv4(2)-dv1(3)*dv4(3)
      v23 = dv2(0)*dv3(0)-dv2(1)*dv3(1)-dv2(2)*dv3(2)-dv2(3)*dv3(3)
      v24 = dv2(0)*dv4(0)-dv2(1)*dv4(1)-dv2(2)*dv4(2)-dv2(3)*dv4(3)
      v34 = dv3(0)*dv4(0)-dv3(1)*dv4(1)-dv3(2)*dv4(2)-dv3(3)*dv4(3)

c      dvertx = (v12*v34 + v14*v23 - rTwo*v13*v24)*(gwwa**2+gwwz**2)
c     Neil edited this vertex to allow implementation of 3-site model.
c     Now, the coupling gwwa is the full coupling squared of this vertex.
c      dvertx = (v12*v34 + v14*v23 - rTwo*v13*v24)*(gwwa)
c     End Neil's edit

c Start of Benj'S modif
      dvertx = (v12*v34 + v14*v23 - rTwo*v13*v24)*(gtemp)
c End of Benj'S modif

c Start of Claude'S modif (Removed minus sign)
      vertex = dcmplx( dvertx )
c End of Claude'S modif
c
      return
      end
