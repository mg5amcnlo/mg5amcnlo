      subroutine momntx(energy,mass,costh,phi , p)
c
c This subroutine sets up a four-momentum from the four inputs.
c
c input:
c       real    energy         : energy
c       real    mass           : mass
c       real    costh          : cos(theta)
c       real    phi            : azimuthal angle
c
c output:
c       real    p(0:3)         : four-momentum
c     
      implicit none
      double precision p(0:3),energy,mass,costh,phi,pp,sinth

      double precision rZero, rOne
      parameter( rZero = 0.0d0, rOne = 1.0d0 )

#ifdef HELAS_CHECK
      double precision rPi, rTwo
      parameter( rPi = 3.14159265358979323846d0, rTwo = 2.d0 )
      integer stdo
      parameter( stdo = 6 )
#endif
c
#ifdef HELAS_CHECK
      if (energy.lt.mass) then
         write(stdo,*)
     &        ' helas-error : energy in momntx is less than mass'
         write(stdo,*) 
     &        '             : energy = ',energy,' : mass = ',mass
      endif
      if (mass.lt.rZero) then
         write(stdo,*) ' helas-error : mass in momntx is negative'
         write(stdo,*) '             : mass = ',mass
      endif
      if (abs(costh).gt.rOne) then
         write(stdo,*) ' helas-error : costh in momntx is improper'
         write(stdo,*) '             : costh = ',costh
      endif
      if (phi.lt.rZero .or. phi.gt.rTwo*rPi) then
         write(stdo,*)
     &   ' helas-warn  : phi in momntx does not lie on 0.0 thru 2.0*pi'
         write(stdo,*) 
     &   '             : phi = ',phi
      endif
#endif

      p(0) = energy

      if ( energy.eq.mass ) then

         p(1) = rZero
         p(2) = rZero
         p(3) = rZero

      else

         pp = sqrt((energy-mass)*(energy+mass))
         sinth = sqrt((rOne-costh)*(rOne+costh))
         p(3) = pp*costh
         if ( phi.eq.rZero ) then
            p(1) = pp*sinth
            p(2) = rZero
         else
            p(1) = pp*sinth*cos(phi)
            p(2) = pp*sinth*sin(phi)
         endif

      endif
c
      return
      end
