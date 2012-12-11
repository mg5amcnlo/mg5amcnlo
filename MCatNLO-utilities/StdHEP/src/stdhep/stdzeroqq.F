
      subroutine STDZEROQQ
C
C... zero elements of QQTRAK, etc.

      IMPLICIT NONE

#include "qqpars.inc"
#include "qqtrak.inc"
#include "qqvrtx.inc"
#include "qqmxcp.inc"
#include "qqipcd.inc"

       NTRKQQ = 0
       NSTBQQ = 0
       NCHGQQ = 0
       call VZERO(IPCDQQ,MCTRK)
       call VZERO(IPRNTV,MCTRK)
       call VZERO(IVPROD,MCTRK)
       call VZERO(IVDECA,MCTRK)
       call VZERO(ITRKOU,MCTRK)
       call VFILL(HELCQQ,MCTRK,-100.)
       call VZERO(ILDECA,250)
       call VZERO(CTTAU,250)
       return
       end
