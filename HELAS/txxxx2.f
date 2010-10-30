      subroutine txxxx2(p,tmass,nhel,nst , tc)
c
c This subroutine computes k^mu*e^nu where e is delta(i,nhel).
c It is used to test for gauge invariance of the tensor routines.
c
c input:
c       real    p(0:3)             : four-momentum of tensor boson
c       real    tmass              : mass          of tensor boson
c       integer nhel = 1..4        : construction of e^nu
c       integer nst  = -1 or 1     : +1 for final, -1 for initial
c
c output:
c       complex tc(6,4)            : tensor wavefunction       epsilon^mu^nu(t)
c     
      implicit none
      double complex tc(6,4), ep(4), em(4)
      double precision p(0:3), tmass
      integer nhel, nst, i, j

      double precision rZero, rOne
      parameter( rZero = 0.0d0, rOne = 1.0d0 )
c
      tc(5,1) = dcmplx(p(0),p(3))*nst
      tc(6,1) = dcmplx(p(1),p(2))*nst

      ep(1) = dcmplx( p(0), rZero )
      ep(2) = dcmplx( p(1), rZero )
      ep(3) = dcmplx( p(2), rZero )
      ep(4) = dcmplx( p(3), rZero )

      if ( nhel.eq.1 ) then
         em(1) = dcmplx( rOne , rZero )
         em(2) = dcmplx( rZero, rZero )
         em(3) = dcmplx( rZero, rZero )
         em(4) = dcmplx( rZero, rZero )
      else if ( nhel.eq.2 ) then
         em(1) = dcmplx( rZero, rZero )
         em(2) = dcmplx( rOne , rZero )
         em(3) = dcmplx( rZero, rZero )
         em(4) = dcmplx( rZero, rZero )
      else if ( nhel.eq.3 ) then
         em(1) = dcmplx( rZero, rZero )
         em(2) = dcmplx( rZero, rZero )
         em(3) = dcmplx( rOne , rZero )
         em(4) = dcmplx( rZero, rZero )
      else if ( nhel.eq.4 ) then
         em(1) = dcmplx( rZero, rZero )
         em(2) = dcmplx( rZero, rZero )
         em(3) = dcmplx( rZero, rZero )
         em(4) = dcmplx( rOne , rZero )
      end if

      do j = 1,4
         do i = 1,4
            tc(i,j) = ep(i)*em(j)
         end do
      end do
c
      return
      end
