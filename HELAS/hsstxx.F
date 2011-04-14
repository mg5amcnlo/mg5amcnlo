      subroutine hsstxx(tc1,sc,gt,xm,xw,jts)
c
c- by RF - Feb. 2006 
c
c This subroutine computes an off-shell tensor current from the tts coupling.
c
c     input:
c          complex tc1               : Incoming tensor particle
c          complex sc                : Incoming scalar particle (Higgs)
c          real    gt                : coupling constant for the tts vertex
c
c     output:
c          complex jts               : off-shell tensor current
c
c     not used:
c          xm, xw
c

      implicit none
c--   dimension of the current set to arbitrary length
      INTEGER DIM
      PARAMETER(DIM=18)
c      include 'dim.inc'
      double complex tc1(DIM),jts(DIM),sc(DIM)
      double precision gt, xm, xw

c The outgoing tensor current is the same as the incoming multiplied by the
c coupling constant and the scalar particle.
c Note that the diagonal tensor terms are always zero because
c the tensor particle is anti-symmetric.

      jts( 1) = 0 !gt * sc(1) * tc1( 1)
      jts( 2) =  gt * sc(1) * tc1( 2)
      jts( 3) =  gt * sc(1) * tc1( 3)
      jts( 4) =  gt * sc(1) * tc1( 4)

      jts( 5) =  gt * sc(1) * tc1( 5)
      jts( 6) = 0 !gt * sc(1) * tc1( 6)
      jts( 7) =  gt * sc(1) * tc1( 7)
      jts( 8) =  gt * sc(1) * tc1( 8)

      jts( 9) =  gt * sc(1) * tc1( 9)
      jts(10) =  gt * sc(1) * tc1(10)
      jts(11) = 0 !gt * sc(1) * tc1(11)
      jts(12) =  gt * sc(1) * tc1(12)

      jts(13) =  gt * sc(1) * tc1(13)
      jts(14) =  gt * sc(1) * tc1(14)
      jts(15) =  gt * sc(1) * tc1(15)
      jts(16) = 0 !gt * sc(1) * tc1(16)

      jts(17) = sc(2) + tc1(17)
      jts(18) = sc(3) + tc1(18)

      return
      end
