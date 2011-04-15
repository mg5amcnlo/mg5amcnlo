      subroutine utscxx(tc1,sc,gt,xm,xw,jts)
c
c- by RF - Feb. 2006 
c  CP3  Modified Nov. 2009 
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
      double complex tc1(DIM),jts(DIM),sc(DIM), gt(2),t1(6,4),t2(6,4)
      double precision xm, xw
      integer m1,m3  
      double precision MT(4,4)

c The outgoing tensor current is the same as the incoming multiplied by the
c coupling constant and the scalar particle. Note that the diagonal tensor
c terms are always zero because the tensor particle is anti-symmetric. The 
c tensor particle does not propagate, thus no multiplication with the tensor 
c propagator.

      jts(17) = sc(2) + tc1(17)
      jts(18) = sc(3) + tc1(18)

      T2(1,1) = tc1(1)
      T2(1,2) = tc1(2)
      T2(1,3) = tc1(3)
      T2(1,4) = tc1(4)
      T2(2,1) = tc1(5)
      T2(2,2) = tc1(6)
      T2(2,3) = tc1(7)
      T2(2,4) = tc1(8)
      T2(3,1) = tc1(9)
      T2(3,2) = tc1(10)
      T2(3,3) = tc1(11)
      T2(3,4) = tc1(12)
      T2(4,1) = tc1(13)
      T2(4,2) = tc1(14)
      T2(4,3) = tc1(15)
      T2(4,4) = tc1(16)
      T2(5,1) = tc1(17)
      T2(6,1) = tc1(18)

      do m1=1,4
         do m3=1,4
            MT(m1,m3) = 0.0d0
         enddo 
      enddo
      MT(1,1) =  1.0d0
      MT(2,2) = -1.0d0
      MT(3,3) = -1.0d0
      MT(4,4) = -1.0d0

 
      if (gt(1).NE.(0D0,0D0)) then

       do  m1=1,4
         do m3=1,4
        T1(m1,m3)=gt(1)*(-(MT(1,m3)*MT(2,m1)*SC(1)*T2(1,2)) + 
     -  MT(1,m1)*MT(2,m3)*SC(1)*T2(1,2) - 
     -  MT(1,m3)*MT(3,m1)*SC(1)*T2(1,3) + 
     -  MT(1,m1)*MT(3,m3)*SC(1)*T2(1,3) - 
     -  MT(1,m3)*MT(4,m1)*SC(1)*T2(1,4) + 
     -  MT(1,m1)*MT(4,m3)*SC(1)*T2(1,4) + 
     -  MT(1,m3)*MT(2,m1)*SC(1)*T2(2,1) - 
     -  MT(1,m1)*MT(2,m3)*SC(1)*T2(2,1) + 
     -  MT(2,m3)*MT(3,m1)*SC(1)*T2(2,3) - 
     -  MT(2,m1)*MT(3,m3)*SC(1)*T2(2,3) + 
     -  MT(2,m3)*MT(4,m1)*SC(1)*T2(2,4) - 
     -  MT(2,m1)*MT(4,m3)*SC(1)*T2(2,4) + 
     -  MT(1,m3)*MT(3,m1)*SC(1)*T2(3,1) - 
     -  MT(1,m1)*MT(3,m3)*SC(1)*T2(3,1) - 
     -  MT(2,m3)*MT(3,m1)*SC(1)*T2(3,2) + 
     -  MT(2,m1)*MT(3,m3)*SC(1)*T2(3,2) + 
     -  MT(3,m3)*MT(4,m1)*SC(1)*T2(3,4) - 
     -  MT(3,m1)*MT(4,m3)*SC(1)*T2(3,4) + 
     -  MT(1,m3)*MT(4,m1)*SC(1)*T2(4,1) - 
     -  MT(1,m1)*MT(4,m3)*SC(1)*T2(4,1) - 
     -  MT(2,m3)*MT(4,m1)*SC(1)*T2(4,2) + 
     -  MT(2,m1)*MT(4,m3)*SC(1)*T2(4,2) - 
     -  MT(3,m3)*MT(4,m1)*SC(1)*T2(4,3) + 
     -  MT(3,m1)*MT(4,m3)*SC(1)*T2(4,3))
       enddo
       enddo  

       jts(1) = T1(1,1)
       jts(2) = T1(1,2)
       jts(3) = T1(1,3)
       jts(4) = T1(1,4)
       jts(5) = T1(2,1)
       jts(6) = T1(2,2)
       jts(7) = T1(2,3)
       jts(8) = T1(2,4)
       jts(9) = T1(3,1)
       jts(10) = T1(3,2)
       jts(11) = T1(3,3)
       jts(12) = T1(3,4)
       jts(13) = T1(4,1)
       jts(14) = T1(4,2)
       jts(15) = T1(4,3)
       jts(16) = T1(4,4)


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
