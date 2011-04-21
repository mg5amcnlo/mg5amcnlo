      subroutine ttscxx(tc1,tc2,sc,gc,vertex)
c
c- by RF - Feb. 2006 
c  CP3  Modified Nov. 2009 
c
c This subroutine computes an amplitude of the tts coupling.
c
c     input:
c          complex tc1(18)           : Incoming tensor particle
c          complex tc2(18)           : Incoming tensor particle
c          complex sc(3)             : Incoming scalar particle (Higgs)
c          complex gc(2)             : coupling constant: gt(1) scalar
c                                                         gt(2) not used
c
c     output:
c          complex vertex            : amplitude for a tts vertex
c

      implicit none
c--   dimension of the current set to arbitrary length
c      INTEGER DIM
c      PARAMETER(DIM=18)
      include "dimension.inc"
      double complex tc1(DIM),tc2(DIM),sc(DIM),t1(6,4),t2(6,4)
      double complex vertex, gc(2)


c Take the inner product between the tensor particles
c and multiply it with the scalar particle and the coupling constant.
c Note that the tensor particle is antisymmetric, thus all diagonal terms
c are zero.

      T1(1,1) = tc1(1)
      T1(1,2) = tc1(2)
      T1(1,3) = tc1(3)
      T1(1,4) = tc1(4)
      T1(2,1) = tc1(5)
      T1(2,2) = tc1(6)
      T1(2,3) = tc1(7)
      T1(2,4) = tc1(8)
      T1(3,1) = tc1(9)
      T1(3,2) = tc1(10)
      T1(3,3) = tc1(11)
      T1(3,4) = tc1(12)
      T1(4,1) = tc1(13)
      T1(4,2) = tc1(14)
      T1(4,3) = tc1(15)
      T1(4,4) = tc1(16)
      T1(5,1) = tc1(17)
      T1(6,1) = tc1(18)

      T2(1,1) = tc2(1)
      T2(1,2) = tc2(2)
      T2(1,3) = tc2(3)
      T2(1,4) = tc2(4)
      T2(2,1) = tc2(5)
      T2(2,2) = tc2(6)
      T2(2,3) = tc2(7)
      T2(2,4) = tc2(8)
      T2(3,1) = tc2(9)
      T2(3,2) = tc2(10)
      T2(3,3) = tc2(11)
      T2(3,4) = tc2(12)
      T2(4,1) = tc2(13)
      T2(4,2) = tc2(14)
      T2(4,3) = tc2(15)
      T2(4,4) = tc2(16)
      T2(5,1) = tc2(17)
      T2(6,1) = tc2(18)

      if (gc(1).NE.(0D0,0D0)) then

      vertex =  gc(1)*(SC(1)*T1(1,2)*T2(1,2) - SC(1)*T1(2,1)*T2(1,2) + 
     -  SC(1)*T1(1,3)*T2(1,3) - SC(1)*T1(3,1)*T2(1,3) + 
     -  SC(1)*T1(1,4)*T2(1,4) - SC(1)*T1(4,1)*T2(1,4) - 
     -  SC(1)*T1(1,2)*T2(2,1) + SC(1)*T1(2,1)*T2(2,1) - 
     -  SC(1)*T1(2,3)*T2(2,3) + SC(1)*T1(3,2)*T2(2,3) - 
     -  SC(1)*T1(2,4)*T2(2,4) + SC(1)*T1(4,2)*T2(2,4) - 
     -  SC(1)*T1(1,3)*T2(3,1) + SC(1)*T1(3,1)*T2(3,1) + 
     -  SC(1)*T1(2,3)*T2(3,2) - SC(1)*T1(3,2)*T2(3,2) - 
     -  SC(1)*T1(3,4)*T2(3,4) + SC(1)*T1(4,3)*T2(3,4) - 
     -  SC(1)*T1(1,4)*T2(4,1) + SC(1)*T1(4,1)*T2(4,1) + 
     -  SC(1)*T1(2,4)*T2(4,2) - SC(1)*T1(4,2)*T2(4,2) + 
     -  SC(1)*T1(3,4)*T2(4,3) - SC(1)*T1(4,3)*T2(4,3))


      else
      vertex = (0D0,0D0)
      endif      


      return
      end
