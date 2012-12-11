
      subroutine STDFLHWXSEC

C   fill stdhep common block 1 with Herwig information
C
C   STDECOM  - center-of-mass energy
C   STDXSEC  - cross-section
C   STDSEED1 - random number seed
C   STDSEED2 - random number seed
C   NEVTREQ  - number of events to be generated
C   NEVTGEN  - number of events actually generated
C   NEVTWRT  - number of events written to output file
c   nevtlh   - number of Les Houches events written to output file
C

#include "herwig.inc"
#include "stdcm1.inc"
#include "stdcnt.inc"
c$$$ THIS HAS BEEN ADDED BY PT
      double precision xsecup2
      common/cxsecup/xsecup2
c$$$
      real XM1,XM2,A1,A2

      STDSEED1 = IBRN(1)
      STDSEED2 = IBRN(2)
      XM1 = RMASS(IPART1)
      XM2 = RMASS(IPART2)
      A1 = SQRT(XM1**2 + PBEAM1**2)
      A2 = SQRT(XM2**2 + PBEAM2**2)
      STDECOM = SQRT((A1+A2)**2 - (PBEAM1 - PBEAM2)**2)
      STDXSEC = EVWGT/1000.
c$$$ THIS HAS BEEN ADDED BY PT
      STDXSEC = xsecup2
c$$$
      NEVTREQ = MAXEV
      NEVTGEN = NEVHEP
      NEVTWRT = nstdwrt
      nevtlh  = nlhwrt
      generatorname = "Herwig"
      pdfname = " "
      return
      end
