      subroutine httxxx(tc1,tc2,gc,mass,width,jsc)
c
c- by RF - Mar. 2006 
c
c This subroutine computes an off-shell scalar from the tts coupling.
c
c     input:
c          complex tc1(18)           : Incoming tensor particle
c          complex tc2(18)           : Incoming tensor particle
c          complex gc(2)             : coupling constant: gt(1) scalar
c                                                         gt(2) not used
c          real    mass              : mass of the outgoing scalar
c          real    width             : width of the outgoing scalar
c
c     output:
c          complex sc(3)             : Incoming scalar particle
c
c

      implicit none
c--   dimension of the current set to arbitrary length
c      INTEGER DIM
c      PARAMETER(DIM=18)
      include "dimension.inc"
      double complex tc1(DIM),tc2(DIM),jsc(DIM)
      double complex vertex,dj,gc(2)
      double precision mass,width,q2,q(4)

c Take the inner product between the tensor particles. The 
c Note that the tensor particle is antisymmetric, thus all diagonal terms
c are zero.

      jsc(2)=tc1(17)+tc2(17)
      jsc(3)=tc1(18)+tc2(18)

      if (gc(1).NE.(0D0,0D0)) then

      q(1) = -dble( jsc(2))
      q(2) = -dble( jsc(3))
      q(3) = -dimag(jsc(3))
      q(4) = -dimag(jsc(2))

      q2 = q(1)**2 - q(2)**2 - q(3)**2 - q(4)**2

      dj = gc(1) /dcmplx( q2-mass**2, mass*width )

      jsc(1) = dj* (
c     &                       + tc1( 1) * tc2( 1)
     &                       - tc1( 2) * tc2( 2)
     &                       - tc1( 3) * tc2( 3)
     &                       - tc1( 4) * tc2( 4)

     &                       - tc1( 5) * tc2( 5)
c     &                       + tc1( 6) * tc2( 6)
     &                       + tc1( 7) * tc2( 7)
     &                       + tc1( 8) * tc2( 8)

     &                       - tc1( 9) * tc2( 9)
     &                       + tc1(10) * tc2(10)
c     &                       + tc1(11) * tc2(11)
     &                       + tc1(12) * tc2(12)

     &                       - tc1(13) * tc2(13)
     &                       + tc1(14) * tc2(14)
     &                       + tc1(15) * tc2(15)
c     &                       + tc1(16) * tc2(16)
     &                                           )

      else
         jsc( 1)=(0D0,0D0)
      endif

      return
      end
