      subroutine hvvtxx(w1,w2,g,xm,xw,jt)
c
c- by RF - Feb. 2006
c
c This subroutine computes the portion of the off-shell current
c for the color-octect tensor t3 in terms of w1 and w2 
c
c input:
c       complex w1(6)          : first  vector                        w1
c       complex w2(6)          : second vector                        w2
c       real    g              : first  coupling constant
c       real    xm             : not used
c       real    xw             : not used
c
c output:
c       complex jt(18)        : tensor current  j^(mu,nu)(w':w1,w2,w3)
c
      implicit none

c dimension of the current set to arbitrary length
      integer DIM
      parameter (DIM=18)
c      include "dimension.inc"
      double complex w1(DIM),w2(DIM),jt(DIM)

      double precision xm,xw,g,s2g

      double precision sqrTwo
      parameter( sqrTwo = 1.41421356237309514547462185873882845044d0 )

      s2g = g * sqrTwo
      
      jt( 1) = 0 ! dcmplx( s2g * (w1(1)*w2(1)-w1(1)*w2(1)) )
      jt( 2) =  dcmplx( s2g * (w1(1)*w2(2)-w1(2)*w2(1)) )
      jt( 3) =  dcmplx( s2g * (w1(1)*w2(3)-w1(3)*w2(1)) )
      jt( 4) =  dcmplx( s2g * (w1(1)*w2(4)-w1(4)*w2(1)) )

      jt( 5) =  dcmplx( s2g * (w1(2)*w2(1)-w1(1)*w2(2)) )
      jt( 6) = 0 ! dcmplx( s2g * (w1(2)*w2(2)-w1(2)*w2(2)) )
      jt( 7) =  dcmplx( s2g * (w1(2)*w2(3)-w1(3)*w2(2)) )
      jt( 8) =  dcmplx( s2g * (w1(2)*w2(4)-w1(4)*w2(2)) )

      jt( 9) =  dcmplx( s2g * (w1(3)*w2(1)-w1(1)*w2(3)) )
      jt(10) =  dcmplx( s2g * (w1(3)*w2(2)-w1(2)*w2(3)) )
      jt(11) = 0 ! dcmplx( s2g * (w1(3)*w2(3)-w1(3)*w2(3)) )
      jt(12) =  dcmplx( s2g * (w1(3)*w2(4)-w1(4)*w2(3)) )

      jt(13) =  dcmplx( s2g * (w1(4)*w2(1)-w1(1)*w2(4)) )
      jt(14) =  dcmplx( s2g * (w1(4)*w2(2)-w1(2)*w2(4)) )
      jt(15) =  dcmplx( s2g * (w1(4)*w2(3)-w1(3)*w2(4)) )
      jt(16) = 0 ! dcmplx( s2g * (w1(4)*w2(4)-w1(4)*w2(4)) )

      jt(17) = w1(5) + w2(5)
      jt(18) = w1(6) + w2(6)

      return
      end
