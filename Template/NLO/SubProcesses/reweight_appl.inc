c Common blocks for NLO or aMC@NLO reweighting
c USed only for interfacing with applgrid
c Conventions for array relevant to NLO computations:
c   i=1,2,3,4  -> event, soft, collinear, soft-collinear
c   ==>  i = icountevts+2 for counterevents

c The following is introduced since this file may be included in codes
c that need not know nexternal. Make sure that maxparticles>=nexternal
      integer maxparticles
      parameter (maxparticles=20)

