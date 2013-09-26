      subroutine stdfixpart(i)

C... force a particle to lie on the mass shell
C    assume that phep(5) is the correct mass

      IMPLICIT NONE

#include "stdhep.inc"

      integer i
      double precision msq,p4,newp,oldp,newroot,ms

      msq = phep(4,i)**2-phep(1,i)**2-phep(2,i)**2-phep(3,i)**2
      p4 = phep(4,i)**2+phep(5,i)**2-msq
      if(p4.lt.0.d0)then
          p4 = phep(5,i)**2+phep(1,i)**2+phep(2,i)**2+phep(3,i)**2
      endif
      phep(4,i) = sqrt(p4)
      return
      end
