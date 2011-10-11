      subroutine ttssxx(tc1,tc2,sc1,sc2,g1,g2,vertex)
c
c- by RF - Mar. 2006 
c
c This subroutine computes an amplitude of the tts coupling.
c
c     input:
c          complex tc1               : Incoming tensor particle
c          complex tc2               : Incoming tensor particle
c          complex sc1               : Incoming scalar particle (Higgs)
c          complex sc2               : Incoming scalar particle (Higgs)
c          complex g1(2)             : coupling constant (Higgs Eff. Thr.)
c          real    g2                : coupling constant
c
c     output:
c          complex vertex            : amplitude for a tts vertex
c

      implicit none
c--   dimension of the current set to arbitrary length
      INTEGER DIM
      PARAMETER(DIM=18)
c      include "dimension.inc"
      double complex tc1(DIM),tc2(DIM),sc1(DIM),sc2(DIM)

      double complex vertex, g1(2)
      double precision g2


c Take the inner product between the tensor particles
c and multiply it with the scalar particles and the coupling constants.
c Note that the tensor particles are antisymmetric, thus all diagonal terms
c are zero.

      if (g1(1).NE.(0D0,0D0)) then

      vertex = g1(1)*g2*sc1(1)*sc2(1)* (
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
      vertex = (0D0,0D0)
      endif      


      return
      end
