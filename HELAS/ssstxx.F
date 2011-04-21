      subroutine ssstxx(tc1,tc2,sc,gt,vertex)
c
c- by RF - Feb. 2006 
c
c This subroutine computes an amplitude of the tts coupling.
c
c     input:
c          complex tc1               : Incoming tensor particle
c          complex tc2               : Incoming tensor particle
c          complex sc                : Incoming scalar particle (Higgs)
c          real    gt                : coupling constant for the tts vertex
c
c     output:
c          complex vertex            : amplitude for a tts vertex
c

      implicit none
c--   dimension of the current set to arbitrary length
      INTEGER DIM
      PARAMETER(DIM=18)
c      include "dimension.inc"
      double complex tc1(DIM),tc2(DIM),sc(DIM)

      double complex vertex
      double precision gt

c Take the inner product between the tensor particles
c and multiply it with the scalar particle and the coupling constant.
c Note that the tensor particle is antisymmetric, thus all diagonal terms
c are zero.

      vertex = gt * sc(1) * (
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

      return
      end
