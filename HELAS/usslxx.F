      subroutine usslxx(s1,s2,gc,xm,xw , uss)
c- by RF - Mar. 2006
c
c This subroutine computes an internal particle current from the three-
c scalar coupling.
c
c input:
c       complex s1(3)          : first  scalar                        s1
c       complex s2(3)          : second scalar                        s2
c       complex gc             : coupling constant                  ghhh
c
c output:
c       complex uss(3)         : internal particle current (scalar)
c    
c not used:
c       xm,xw
c

      implicit none
      INTEGER DIM
      PARAMETER(DIM=18)

c      include "dimension.inc"
      double complex s1(DIM),s2(DIM),uss(DIM)
      double precision xm,xw,gc

      uss(2) = s1(2)+s2(2)
      uss(3) = s1(3)+s2(3)

c the internal particle does not propagate, so no multiplication
c with a propagator necessary.

      uss(1) = - gc*s1(1)*s2(1)
c
      return
      end
