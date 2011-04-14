      subroutine sxxxxx(p,nss , sc)
c
c This subroutine computes a complex SCALAR wavefunction.
c
c input:
c       real    p(0:3)         : four-momentum of scalar boson
c       integer nss  = -1 or 1 : +1 for final, -1 for initial
c
c output:
c       complex sc(3)          : scalar wavefunction                   s
c     
      implicit none
      double complex sc(3)
      double precision p(0:3)
      integer nss

      double precision rOne
      parameter( rOne = 1.0d0 )

#ifdef HELAS_CHECK
      double precision p2
      double precision epsi
      parameter( epsi = 2.0d-5 )
      double precision rZero
      parameter( rZero = 0.0d0 )
      integer stdo
      parameter( stdo = 6 )
#endif
c
#ifdef HELAS_CHECK
      if ( abs(p(0))+abs(p(1))+abs(p(2))+abs(p(3)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : p(0:3) in sxxxxx is zero momentum'
      endif
      if ( p(0).le.rZero ) then
         write(stdo,*)
     &        ' helas-error : p(0:3) in sxxxxx has non-positive energy'
         write(stdo,*)
     &        '             : p(0) = ',p(0)
      endif
      p2 = p(0)**2-(p(1)**2+p(2)**2+p(3)**2)
      if ( p2.lt.-p(0)**2*epsi ) then
         write(stdo,*) ' helas-error : p(0:3) in sxxxxx is spacelike'
         write(stdo,*) '             : p**2 = ',p2
      endif
      if ( abs(nss).ne.1 ) then
         write(stdo,*) ' helas-error : nss in sxxxxx is not -1,1'
         write(stdo,*) '             : nss = ',nss
      endif
#endif

      sc(1) = dcmplx( rOne )
      sc(2) = dcmplx(p(0),p(3))*nss
      sc(3) = dcmplx(p(1),p(2))*nss
c
      return
      end
