      subroutine sstlxx(s1,s2,t3,gc , vertex)
c- by RF - Mar. 2006
c
c This subroutine computes an amplitude of the three-scalar coupling.
c
c input:
c       complex s1(3)          : first  scalar                        s1
c       complex s2(3)          : second scalar                        s2
c       complex t3(3)          : internal particle                    s3
c       real    gc             : coupling constant                    gv
c
c output:
c       complex vertex         : amplitude               gamma(s1,s2,s3)
c     
      implicit none
      INTEGER DIM
      PARAMETER(DIM=18)

c      include "dimension.inc"
      double complex s1(DIM),s2(DIM),t3(DIM),vertex
      double precision gc

      vertex = - gc*s1(1)*s2(1)*t3(1)

      return
      end
