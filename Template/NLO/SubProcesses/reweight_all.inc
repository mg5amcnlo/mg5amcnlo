      include 'reweight.inc'

c Common blocks for NLO or aMC@NLO reweighting
c Conventions for array relevant to NLO computations:
c   i=1,2,3,4  -> event, soft, collinear, soft-collinear
c   ==>  i = icountevts+2 for counterevents

c Second argument specifies the nFKSprocess: 
c 2 ... nFKSprocess*2 (in steps of 2) for the H-events,
c 1 ... nFKSprocess*2-1 (in steps of 2) for the S-events

c Maximum of flavor structures combined in the parton_lum_*.f files for the array sizes
      integer mproc
      parameter (mproc=100)
c The actual maximum used in the code
      integer maxproc_save
      common/cwgtmaxproc_save/maxproc_save

c Reference weight and Ellis-Sexton scales
      double precision wgtref_all(2*maxFKSconfigs,mproc),
     &	               wgtref_nbody_all(mproc),
     &                 wgtqes2_all(4,0:2*maxFKSconfigs)
      common/cwgt0_all/wgtref_all,wgtref_nbody_all,wgtqes2_all

c Bjorken x's
      double precision wgtxbj_all(2,4,0:2*maxFKSconfigs)
      common/cwgt1_all/wgtxbj_all

c Reference renormalization and factorization scales squared
      double precision wgtmuR2_all(4,0:2*maxFKSconfigs),
     &     wgtmuF12_all(4,0:2*maxFKSconfigs),
     &	   wgtmuF22_all(4,0:2*maxFKSconfigs)
      common/cwgt2mu_all/wgtmuR2_all,wgtmuF12_all,wgtmuF22_all

c Real-emission weights, scale independent
      double precision wgtwreal_all(4,2*maxFKSconfigs)
      common/cwgt3_all/wgtwreal_all

c (n+1)-body degenerate weights, scale independent
      double precision wgtwdeg_all(4,2*maxFKSconfigs)
      common/cwgt4_all/wgtwdeg_all

c (n+1)-body degenerate weights, muF dependence
      double precision wgtwdegmuf_all(4,2*maxFKSconfigs)
      common/cwgt5_all/wgtwdegmuf_all

c n-body weights: Born, n-body NLO scale independent,
c   n-body NLO scale muF dependence, n-body NLO scale muR dependence
      double precision wgtwborn_all,wgtwns_all,
     #                 wgtwnsmuf_all,wgtwnsmur_all
      common/cwgt6_all/wgtwborn_all,wgtwns_all,
     &                 wgtwnsmuf_all,wgtwnsmur_all

c MC stuff: number of father's colour partners
      integer iwgtnumpartn_all(2*maxFKSconfigs)
      common/cwgt8_all/iwgtnumpartn_all

c Common blocks for NLO or aMC@NLO reweighting
c Conventions for array relevant to NLO computations:
c   i=1,2,3,4  -> event, soft, collinear, soft-collinear
c   ==>  i = icountevts+2 for counterevents

c Kinematic configurations
      double precision wgtkin_all(0:3,maxparticles,4,0:2*maxFKSconfigs)
      common/cwgt2_all/wgtkin_all

c MC stuff: MC counterterms proper, and Bjorken x's
      double precision wgtwmcxsec_all(maxparticles,2*maxFKSconfigs),
     &                 wgtmcxbj_all(2,maxparticles,2*maxFKSconfigs)
      common/cwgt7_all/wgtwmcxsec_all,wgtmcxbj_all

c Some extra info about the number of S contributions
      integer nScontributions,nFKSprocess_reweight(0:maxFKSconfigs)
      common/cwgt8_all/nScontributions,nFKSprocess_reweight
