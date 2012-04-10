c Common blocks for NLO or aMC@NLO reweighting
c Conventions for array relevant to NLO computations:
c   i=1,2,3,4  -> event, soft, collinear, soft-collinear
c   ==>  i = icountevts+2 for counterevents

c Kinematic configurations
      double precision wgtkin(0:3,nexternal,4)
      common/cwgt2/wgtkin

c MC stuff: MC counterterms proper, and Bjorken x's
      double precision wgtwmcxsec(nexternal),wgtmcxbj(2,nexternal)
      common/cwgt7/wgtwmcxsec,wgtmcxbj

