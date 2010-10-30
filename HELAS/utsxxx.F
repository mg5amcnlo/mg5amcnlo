      subroutine utsxxx(tc1,sc,gt,xm,xw,jts)
c
c- by RF - Feb. 2006 
c
c This subroutine computes an off-shell tensor current from the tts coupling.
c
c     input:
c          complex tc1(18)           : Incoming tensor particle
c          complex sc(3)             : Incoming scalar particle
c          complex gt(2)             : coupling constant: gt(1) scalar
c                                                         gt(2) not used
c
c     output:
c          complex jts               : off-shell tensor current
c
c     not used:
c          xm, xw
c

      implicit none
c--   dimension of the current set to arbitrary length
c      INTEGER DIM
c      PARAMETER(DIM=18)
      include "dimension.inc"
      double complex tc1(DIM),jts(DIM),sc(DIM), gt(2)
      double precision xm, xw

c The outgoing tensor current is the same as the incoming multiplied by the
c coupling constant and the scalar particle. Note that the diagonal tensor
c terms are always zero because the tensor particle is anti-symmetric. The 
c tensor particle does not propagate, thus no multiplication with the tensor 
c propagator.

      jts(17) = sc(2) + tc1(17)
      jts(18) = sc(3) + tc1(18)

 
      if (gt(1).NE.(0D0,0D0)) then

         jts( 1) = (0D0,0D0)        !-gt(1) * sc(1) * tc1( 1)
         jts( 2) =  -gt(1) * sc(1) * tc1( 2)
         jts( 3) =  -gt(1) * sc(1) * tc1( 3)
         jts( 4) =  -gt(1) * sc(1) * tc1( 4)

         jts( 5) =  -gt(1) * sc(1) * tc1( 5)
         jts( 6) = (0D0,0D0)        !-gt(1) * sc(1) * tc1( 6)
         jts( 7) =  -gt(1) * sc(1) * tc1( 7)
         jts( 8) =  -gt(1) * sc(1) * tc1( 8)

         jts( 9) =  -gt(1) * sc(1) * tc1( 9)
         jts(10) =  -gt(1) * sc(1) * tc1(10)
         jts(11) = (0D0,0D0)        !-gt(1) * sc(1) * tc1(11)
         jts(12) =  -gt(1) * sc(1) * tc1(12)
         
         jts(13) =  -gt(1) * sc(1) * tc1(13)
         jts(14) =  -gt(1) * sc(1) * tc1(14)
         jts(15) =  -gt(1) * sc(1) * tc1(15)
         jts(16) = (0D0,0D0)        !-gt(1) * sc(1) * tc1(16)

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
