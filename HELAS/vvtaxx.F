      subroutine vvtaxx(ga,gb,tc,g,xm, vertex)
c
c- by RF - Feb. 2006
c
c This subroutine computes the portion of the amplitude of the four-point 
c coupling of 2 massless color octet gauge bosons
C with an effective color octect antisymmetric tensor.
c                                                                       
c input:
c     complex ga(6)                       : first vector  (gluon)
c     complex gb(6)                       : second vector (gluon)
c     complex tc(18)                      : tensor current
c     real    g                           : coupling constant
c 
c output:
c     complex vertex                      : amplitude
c

      implicit none

c dimension of the current set to arbitrary length
      integer DIM
      parameter (DIM=18)
c      include "dimension.inc"
      double complex ga(DIM),gb(DIM),tc(DIM)
      double precision xm,xw,g
      double precision sqrTwo
      parameter( sqrTwo=1.41421356237309514547462185873882845044d0 )

      double complex dvertx, vertex


      dvertx = ! + tc( 1)*( ga(1)*gb(1) - ga(1)*gb(1) )  Always zero
     &          - tc( 2)*( ga(1)*gb(2) - ga(2)*gb(1) ) 
     &          - tc( 3)*( ga(1)*gb(3) - ga(3)*gb(1) ) 
     &          - tc( 4)*( ga(1)*gb(4) - ga(4)*gb(1) ) 
     
     &          - tc( 5)*( ga(2)*gb(1) - ga(1)*gb(2) ) 
     &         ! + tc( 6)*( ga(2)*gb(2) - ga(2)*gb(2) )  Always zero
     &          + tc( 7)*( ga(2)*gb(3) - ga(3)*gb(2) ) 
     &          + tc( 8)*( ga(2)*gb(4) - ga(4)*gb(2) ) 
     
     &          - tc( 9)*( ga(3)*gb(1) - ga(1)*gb(3) ) 
     &          + tc(10)*( ga(3)*gb(2) - ga(2)*gb(3) ) 
     &         ! + tc(11)*( ga(3)*gb(3) - ga(3)*gb(3) )  Always zero
     &          + tc(12)*( ga(3)*gb(4) - ga(4)*gb(3) ) 
     
     &          - tc(13)*( ga(4)*gb(1) - ga(1)*gb(4) ) 
     &          + tc(14)*( ga(4)*gb(2) - ga(2)*gb(4) ) 
     &          + tc(15)*( ga(4)*gb(3) - ga(3)*gb(4) ) 
     &         ! + tc(16)*( ga(4)*gb(4) - ga(4)*gb(4) )  Always zero

      vertex = g * dvertx /sqrTwo


      return
      end
