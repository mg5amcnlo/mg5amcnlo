c Common blocks for NLO or aMC@NLO reweighting
c Conventions for array relevant to NLO computations:
c   i=1,2,3,4  -> event, soft, collinear, soft-collinear
c   ==>  i = icountevts+2 for counterevents

c The following is introduced since this file may be included in codes
c that need not know nexternal. Make sure that maxparticles>=nexternal
      integer maxparticles
      parameter (maxparticles=20)

c Reference weight and Ellis-Sexton scales
      double precision wgtref,wgtqes2(4)
      common/cwgt0/wgtref,wgtqes2

c Bjorken x's
      double precision wgtxbj(2,4)
      common/cwgt1/wgtxbj

c Kinematic configurations
      double precision wgtkinE(0:3,maxparticles,4)
      common/cwgt2E/wgtkinE

c Reference renormalization and factorization scales squared
      double precision wgtmuR2(4),wgtmuF12(4),wgtmuF22(4)
      common/cwgt2mu/wgtmuR2,wgtmuF12,wgtmuF22

c Real-emission weights, scale independent
      double precision wgtwreal(4)
      common/cwgt3/wgtwreal

c (n+1)-body degenerate weights, scale independent
      double precision wgtwdeg(4)
      common/cwgt4/wgtwdeg

c (n+1)-body degenerate weights, muF dependence
      double precision wgtwdegmuf(4)
      common/cwgt5/wgtwdegmuf

c n-body weights: Born, n-body NLO scale independent,
c   n-body NLO scale muF dependence, n-body NLO scale muR dependence
      double precision wgtwborn(2:2),wgtwns(2:2),
     #                 wgtwnsmuf(2:2),wgtwnsmur(2:2)
      common/cwgt6/wgtwborn,wgtwns,wgtwnsmuf,wgtwnsmur

c MC stuff: MC counterterms proper, and Bjorken x's
      double precision wgtwmcxsecE(maxparticles),
     #                 wgtmcxbjE(2,maxparticles)
      common/cwgt7E/wgtwmcxsecE,wgtmcxbjE

c MC stuff: number of father's colour partners
      integer iwgtnumpartn
      common/cwgt8/iwgtnumpartn

c Auxiliary quantities used when writing events
      integer jwgtinfo,mexternal
      common/cwgtaux0/jwgtinfo,mexternal

c
c Cross sections after variations of scales and PDFs
c
      integer maxscales,maxPDFs
      parameter (maxscales=3)
      parameter (maxPDFs=200)

c Number of scale factors (for each scale) and PDF error pairs
      integer numscales,numPDFpairs
      common/cwgxsec1/numscales,numPDFpairs

c Cross sections at different scales and PDFs
      double precision wgtxsecmu(maxscales,maxscales),
     #                 wgtxsecPDF(0:maxPDFs)
      common/cwgxsec2/wgtxsecmu,wgtxsecPDF
