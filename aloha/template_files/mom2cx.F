      subroutine mom2cx(esum,mass1,mass2,costh1,phi1 , p1,p2)
c
c This subroutine sets up two four-momenta in the two particle rest
c frame.
c
c input:
c       real    esum           : energy sum of particle 1 and 2
c       real    mass1          : mass            of particle 1
c       real    mass2          : mass            of particle 2
c       real    costh1         : cos(theta)      of particle 1
c       real    phi1           : azimuthal angle of particle 1
c
c output:
c       real    p1(0:3)        : four-momentum of particle 1
c       real    p2(0:3)        : four-momentum of particle 2
c     
      implicit none
      double precision p1(0:3),p2(0:3),
     &     esum,mass1,mass2,costh1,phi1,md2,ed,pp,sinth1

      double precision rZero, rHalf, rOne, rTwo
      parameter( rZero = 0.0d0, rHalf = 0.5d0 )
      parameter( rOne = 1.0d0, rTwo = 2.0d0 )

#ifdef HELAS_CHECK
      double precision rPi
      parameter( rPi = 3.14159265358979323846d0 )
      integer stdo
      parameter( stdo = 6 )
#endif
c
#ifdef HELAS_CHECK
      if (esum.lt.mass1+mass2) then
         write(stdo,*)
     &        ' helas-error : esum in mom2cx is less than mass1+mass2'
         write(stdo,*)
     &        '             : esum = ',esum,
     &        ' : mass1+mass2 = ',mass1,mass2
      endif
      if (mass1.lt.rZero) then
         write(stdo,*) ' helas-error : mass1 in mom2cx is negative'
         write(stdo,*) '             : mass1 = ',mass1
      endif
      if (mass2.lt.rZero) then
         write(stdo,*) ' helas-error : mass2 in mom2cx is negative'
         write(stdo,*) '             : mass2 = ',mass2
      endif
      if (abs(costh1).gt.1.) then
         write(stdo,*) ' helas-error : costh1 in mom2cx is improper'
         write(stdo,*) '             : costh1 = ',costh1
      endif
      if (phi1.lt.rZero .or. phi1.gt.rTwo*rPi) then
         write(stdo,*)
     &   ' helas-warn  : phi1 in mom2cx does not lie on 0.0 thru 2.0*pi'
         write(stdo,*) 
     &   '             : phi1 = ',phi1
      endif
#endif

      md2 = (mass1-mass2)*(mass1+mass2)
      ed = md2/esum
      if ( mass1*mass2.eq.rZero ) then
         pp = (esum-abs(ed))*rHalf
      else
         pp = sqrt((md2/esum)**2-rTwo*(mass1**2+mass2**2)+esum**2)*rHalf
      endif
      sinth1 = sqrt((rOne-costh1)*(rOne+costh1))

      p1(0) = max((esum+ed)*rHalf,rZero)
      p1(1) = pp*sinth1*cos(phi1)
      p1(2) = pp*sinth1*sin(phi1)
      p1(3) = pp*costh1

      p2(0) = max((esum-ed)*rHalf,rZero)
      p2(1) = -p1(1)
      p2(2) = -p1(2)
      p2(3) = -p1(3)
c
      return
      end
