      subroutine pxxxxx(p,tmass,nhel,nst , tc)

c    CP3 2009.NOV

c This subroutine computes a PSEUDOR wavefunction.
c
c input:
c       real    p(0:3)         : four-momentum of tensor boson
c       real    tmass          : mass          of tensor boson
c       integer nhel           : helicity      of tensor boson
c                = -2,-1,0,1,2 : (0 is forbidden if tmass=0.0)
c       integer nst  = -1 or 1 : +1 for final, -1 for initial
c
c output:
c       complex tc(18)         : PSEUDOR  wavefunction    epsilon^mu^nu(t)
c     
      implicit none
      double precision p(0:3), tmass
      integer nhel, nst
      double complex tc(18)

      double complex ft(6,4), ep(4), em(4), e0(4)
      double precision pt, pt2, pp, pzpt, emp, sqh, sqs
      integer i, j

      double precision rZero, rHalf, rOne, rTwo
      parameter( rZero = 0.0d0, rHalf = 0.5d0 )
      parameter( rOne = 1.0d0, rTwo = 2.0d0 )


      tc(1)=NHEL
      tc(17) = dcmplx(p(0),p(3))*nst
      tc(18) = dcmplx(p(1),p(2))*nst

      return
      end
