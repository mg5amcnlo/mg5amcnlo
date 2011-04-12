      subroutine j3xxxx(fi,fo,gaf,gzf,zmass,zwidth , j3)
c
c This subroutine computes the sum of photon and Z currents with the
c suitable weights ( j(W3) = cos(theta_W) j(Z) + sin(theta_W) j(A) ).
c The output j3 is useful as an input of vvvxxx, jvvxxx or w3w3xx.
c The photon propagator is given in Feynman gauge, and the Z propagator
c is given in unitary gauge.
c
c input:
c       complex fi(6)          : flow-in  fermion                   |fi>
c       complex fo(6)          : flow-out fermion                   <fo|
c       complex gaf(2)         : fi couplings with A                 gaf
c       complex gzf(2)         : fi couplings with Z                 gzf
c       real    zmass          : mass  of Z
c       real    zwidth         : width of Z
c
c output:
c       complex j3(6)          : W3 current             j^mu(<fo|w3|fi>)
c     
      implicit none
      double complex fi(6),fo(6),j3(6),gaf(2),gzf(2)
      double complex c0l,c1l,c2l,c3l,csl,c0r,c1r,c2r,c3r,csr,dz,ddif
      double complex gn,gz3l,ga3l
      double complex cm2  ! mass**2- I Gamma mass (Fabio)
      double precision q(0:3),zmass,zwidth,zm2,zmw
      double precision q2,da,ww,cw,sw

      double precision rZero, rOne
      parameter( rZero = 0.0d0, rOne = 1.0d0 )
      double complex cImag, cZero
      parameter( cImag = ( 0.0d0, 1.0d0 ), cZero = ( 0.0d0, 0.0d0 ) )

#ifdef HELAS_CHECK
      integer stdo
      parameter( stdo = 6 )
#endif
c
#ifdef HELAS_CHECK
      if ( abs(fi(1))+abs(fi(2))+abs(fi(3))+abs(fi(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : fi in j3xxxx is zero spinor'
      endif
      if ( abs(fi(5))+abs(fi(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : fi in j3xxxx has zero momentum'
      endif
      if ( abs(fo(1))+abs(fo(2))+abs(fo(3))+abs(fo(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : fo in j3xxxx is zero spinor'
      endif
      if ( abs(fo(5))+abs(fo(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : fo in j3xxxx has zero momentum'
      endif
      if ( gaf(1).eq.cZero .and. gaf(2).eq.cZero ) then
         write(stdo,*)
     &        ' helas-error : gaf in j3xxxx is zero coupling'
      endif
      if ( gzf(1).eq.cZero .and. gzf(2).eq.cZero ) then
         write(stdo,*)
     &        ' helas-error : gzf in j3xxxx is zero coupling'
      endif
      if ( gaf(1).ne.gaf(2) ) then
         write(stdo,*)
     &        ' helas-warn  : gaf in j3xxxx is non-standard coupling'
         write(stdo,*) 
     &        '             : gaf = ( ',gaf(1),gaf(2),' )'
      endif
      if ( abs(gzf(1))*abs(gzf(2)).gt.rZero .or.
     &     abs(gzf(1)).le.abs(gzf(2))           ) then
         write(stdo,*)
     &        ' helas-warn  : gzf in j3xxxx is non-standard coupling'
         write(stdo,*) 
     &        '             : gzf = ( ',gzf(1),gzf(2),' )'
      endif
      if ( zmass.le.rZero ) then
         write(stdo,*) ' helas-error : zmass in j3xxxx is not positive'
         write(stdo,*) '             : zmass = ',zmass
      endif
      if ( zwidth.lt.rZero ) then
         write(stdo,*) ' helas-error : zwidth in j3xxxx is negative'
         write(stdo,*) '             : zwidth = ',zwidth
      endif
#endif

      j3(5) = fo(5)-fi(5)
      j3(6) = fo(6)-fi(6)

      q(0) = -dble( j3(5))
      q(1) = -dble( j3(6))
      q(2) = -dimag(j3(6))
      q(3) = -dimag(j3(5))
      q2 = q(0)**2-(q(1)**2+q(2)**2+q(3)**2)
      zm2 = zmass**2
      zmw = zmass*zwidth

#ifdef HELAS_CHECK
      if ( abs(j3(5))+abs(j3(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : j3 in j3xxxx has zero momentum'
      endif
      if ( q2.eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : j3 in j3xxxx is on photon pole'
         write(stdo,*)
     &        '             : q = ',q(0),q(1),q(2),q(3)
         j3(1) = cZero
         j3(2) = cZero
         j3(3) = cZero
         j3(4) = cZero
         return
      endif
      if ( zwidth.eq.rZero .and. q2.eq.zm2 ) then
         write(stdo,*) ' helas-error : j3 in j3xxxx is on z pole'
         write(stdo,*) '             : q        = ',q(0),q(1),q(2),q(3)
         write(stdo,*) '             : abs(q**2)= ',sqrt(abs(q2))
         j3(1) = cZero
         j3(2) = cZero
         j3(3) = cZero
         j3(4) = cZero
         return
      endif
#endif

      da = rOne/q2
C      ww = max(dsign(zmw,q2), rZero)
      dz = rOne/dcmplx( q2-zm2, zmw )
      ddif = dcmplx( -zm2, zmw )*da*dz

c ddif is the difference : ddif=da-dz
c  For the running width, use below instead of the above ww,dz and ddif.
c      ww = max( zwidth*q2/zmass, rZero )
c      dz = rOne/dcmplx( q2-zm2, zmw )
c      ddif = dcmplx( -zm2, zmw )*da*dz



      cw = rOne/sqrt(rOne+(gzf(2)/gaf(2))**2)
      sw = sqrt((rOne-cw)*(rOne+cw))
      gn = gaf(2)*sw
      gz3l = gzf(1)*cw
      ga3l = gaf(1)*sw
      c0l =   fo(3)*fi(1)+fo(4)*fi(2)
      c0r =   fo(1)*fi(3)+fo(2)*fi(4)
      c1l = -(fo(3)*fi(2)+fo(4)*fi(1))
      c1r =   fo(1)*fi(4)+fo(2)*fi(3)
      c2l =  (fo(3)*fi(2)-fo(4)*fi(1))*cImag
      c2r = (-fo(1)*fi(4)+fo(2)*fi(3))*cImag
      c3l =  -fo(3)*fi(1)+fo(4)*fi(2)
      c3r =   fo(1)*fi(3)-fo(2)*fi(4)

c     Fabio's implementation of the fixed width
      cm2=dcmplx( zm2, -zmw )
c     csl = (q(0)*c0l-q(1)*c1l-q(2)*c2l-q(3)*c3l)/zm2
c     csr = (q(0)*c0r-q(1)*c1r-q(2)*c2r-q(3)*c3r)/zm2
      csl = (q(0)*c0l-q(1)*c1l-q(2)*c2l-q(3)*c3l)/cm2
      csr = (q(0)*c0r-q(1)*c1r-q(2)*c2r-q(3)*c3r)/cm2
      
      j3(1) =  gz3l*dz*(c0l-csl*q(0))+ga3l*c0l*da
     &       + gn*(c0r*ddif+csr*q(0)*dz)
      j3(2) =  gz3l*dz*(c1l-csl*q(1))+ga3l*c1l*da
     &       + gn*(c1r*ddif+csr*q(1)*dz)
      j3(3) =  gz3l*dz*(c2l-csl*q(2))+ga3l*c2l*da
     &       + gn*(c2r*ddif+csr*q(2)*dz)
      j3(4) =  gz3l*dz*(c3l-csl*q(3))+ga3l*c3l*da
     &       + gn*(c3r*ddif+csr*q(3)*dz)
c
      return
      end
