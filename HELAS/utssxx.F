      subroutine utssxx(tc1,sc1,sc2,g1,g2,xm,xw,jts)
c
c- by RF - Mar. 2006 
c
c This subroutine computes an off-shell tensor current from the ttss coupling.
c
c     input:
c          complex tc1(18)           : Incoming tensor particle
c          complex sc1(3)            : Incoming scalar particle (Higgs)
c          complex sc2(3)            : second incoming scalar particle (Higgs)
c          complex g1(2)             : coupling constant (Higgs effc. theor)
c          real    g2                : coupling constant (include extra Higgs)
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
c      include "dimension.inc"
      double complex tc1(DIM),jts(DIM),sc1(DIM),sc2(DIM),g1(2)
      double precision g2, xm, xw

c The outgoing tensor current is the same as the incoming multiplied by the
c coupling constants and the scalar particles.
c Note that the diagonal tensor terms are always zero because
c the tensor particle is anti-symmetric.


      jts(17) = sc1(2) + sc2(2) + tc1(17)
      jts(18) = sc1(3) + sc2(3) + tc1(18)

 
      if (g1(1).NE.(0D0,0D0)) then

         jts( 1) = (0D0,0D0)   ! g1(1)* g2 * sc1(1) * sc2(1) * tc1( 1)
         jts( 2) =  g1(1)* g2 * sc1(1) * sc2(1) * tc1( 2)
         jts( 3) =  g1(1)* g2 * sc1(1) * sc2(1) * tc1( 3)
         jts( 4) =  g1(1)* g2 * sc1(1) * sc2(1) * tc1( 4)

         jts( 5) =  g1(1)* g2 * sc1(1) * sc2(1) * tc1( 5)
         jts( 6) = (0D0,0D0)   ! g1(1)* g2 * sc1(1) * sc2(1) * tc1( 6)
         jts( 7) =  g1(1)* g2 * sc1(1) * sc2(1) * tc1( 7)
         jts( 8) =  g1(1)* g2 * sc1(1) * sc2(1) * tc1( 8)

         jts( 9) =  g1(1)* g2 * sc1(1) * sc2(1) * tc1( 9)
         jts(10) =  g1(1)* g2 * sc1(1) * sc2(1) * tc1(10)
         jts(11) = (0D0,0D0)   ! g1(1)* g2 * sc1(1) * sc2(1) * tc1(11)
         jts(12) =  g1(1)* g2 * sc1(1) * sc2(1) * tc1(12)
         
         jts(13) =  g1(1)* g2 * sc1(1) * sc2(1) * tc1(13)
         jts(14) =  g1(1)* g2 * sc1(1) * sc2(1) * tc1(14)
         jts(15) =  g1(1)* g2 * sc1(1) * sc2(1) * tc1(15)
         jts(16) = (0D0,0D0)   ! g1(1)* g2 * sc1(1) * sc2(1) * tc1(16)

      else
         jts( 1)=(0D0,0D0)
         jts( 2)=(0D0,0D0)
         jts( 3)=(0D0,0D0)
         jts( 4)=(0D0,0D0)
         jts( 5)=(0D0,0D0)
         jts( 6)=(0D0,0D0)
         jts( 7)=(0D0,0D0)
         jts( 8)=(0D0,0D0)
         jts( 9)=(0D0,0D0)
         jts(10)=(0D0,0D0)
         jts(11)=(0D0,0D0)
         jts(12)=(0D0,0D0)
         jts(13)=(0D0,0D0)
         jts(14)=(0D0,0D0)
         jts(15)=(0D0,0D0)
         jts(16)=(0D0,0D0)
      endif


      return
      end
