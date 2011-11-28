      subroutine jvtaxx(ga,tc,g,xm1,xw,jw)
c
c- by RF - Feb. 2006
c
c This subroutine computes the off-shell vector current
c in terms of the vector w1 and tensor jt 
c
c     input:
c         complex ga                  : incoming vector boson
c         complex tc                  : incoming tensor particle
c         real    g                   : coupling constant
c
c     output:
c         complex jw                  : outgoing vector current
c
c     not used:
c         real xm1, xw
c

      implicit none

c dimension of the current set to arbitrary length
      integer DIM
      parameter (DIM=18)
c      include "dimension.inc"
      double complex ga(DIM),jw(DIM),tc(DIM)
      double complex gt1(4), gt2(4)
      double precision q(0:3),g,q2,dv
      double precision xm1,xw
      double precision sqrTwo
      parameter( sqrTwo = 1.41421356237309514547462185873882845044d0 )

      jw(5) = ga(5) + tc(17)
      jw(6) = ga(6) + tc(18)

      q(0) = - dble( jw(5))
      q(1) = - dble( jw(6))
      q(2) = - dimag(jw(6))
      q(3) = - dimag(jw(5))

      q2 = q(0)**2 -q(1)**2 -q(2)**2 -q(3)**2



c Gluon propagator:

      dv = - g /sqrTwo / q2


c First take the inner product of the incoming vector with the
c second index of the tensor

      gt2(1) = ga(1)*tc( 1) - ga(2)*tc( 2) - ga(3)*tc( 3) - ga(4)*tc( 4)
      gt2(2) = ga(1)*tc( 5) - ga(2)*tc( 6) - ga(3)*tc( 7) - ga(4)*tc( 8)
      gt2(3) = ga(1)*tc( 9) - ga(2)*tc(10) - ga(3)*tc(11) - ga(4)*tc(12)
      gt2(4) = ga(1)*tc(13) - ga(2)*tc(14) - ga(3)*tc(15) - ga(4)*tc(16)

c and with the first index of the tensor

      gt1(1) = ga(1)*tc( 1) - ga(2)*tc( 5) - ga(3)*tc( 9) - ga(4)*tc(13)
      gt1(2) = ga(1)*tc( 2) - ga(2)*tc( 6) - ga(3)*tc(10) - ga(4)*tc(14)
      gt1(3) = ga(1)*tc( 3) - ga(2)*tc( 7) - ga(3)*tc(11) - ga(4)*tc(15)
      gt1(4) = ga(1)*tc( 4) - ga(2)*tc( 8) - ga(3)*tc(12) - ga(4)*tc(16)

c The current is the difference of gt1 and gt2 with the remaining
c tensor indices the indices of the outgoing vector current

      jw(1) = dv * (gt1(1) - gt2(1))
      jw(2) = dv * (gt1(2) - gt2(2))
      jw(3) = dv * (gt1(3) - gt2(3))
      jw(4) = dv * (gt1(4) - gt2(4))

      return
      end

