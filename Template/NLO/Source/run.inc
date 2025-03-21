c**************************************************************
c    run.inc 
c**************************************************************
c
c     Scales
c
      real*8          scale,scalefact,ellissextonfact,alpsfact, mue_ref_fixed, mue_over_ref
      logical         fixed_ren_scale,fixed_fac_scale,fixed_couplings,hmult, fixed_extra_scale
      integer         ickkw,nhmult,dynamical_scale_choice
      common/to_scale/scale,scalefact,ellissextonfact,alpsfact,fixed_ren_scale,
     $                mue_ref_fixed, mue_over_ref, fixed_extra_scale, 
     $                fixed_fac_scale,fixed_couplings,ickkw,nhmult,hmult,
     $                dynamical_scale_choice

c
c New stuff for scales (NLO)
c

c Same as fixed_ren_scale and fixed_fac_scale, for Ellis-Sexton scale
      logical fixed_QES_scale
      common/cfxQES/fixed_QES_scale

c Ratios of current values of scales over their reference values. The
c latter may be fixed or dynamic
      double precision muR_over_ref,muF1_over_ref,muF2_over_ref,
     &                 QES_over_ref,muF_over_ref
      common/cscales_fact/muR_over_ref,muF1_over_ref,muF2_over_ref,
     &                    QES_over_ref,muF_over_ref

c Values of fixed reference scales, to be given in input
      double precision muR_ref_fixed,muF1_ref_fixed,
     &                 muF2_ref_fixed,QES_ref_fixed,muF_ref_fixed
      common/cscales_fixed_values/muR_ref_fixed,muF1_ref_fixed,
     &                 muF2_ref_fixed,QES_ref_fixed,muF_ref_fixed

c Values of the current scales squared. The information is partly
c redundant with that of common block /to_scale/
      double precision muR2_current,muF12_current,
     &                 muF22_current,QES2_current
      common/cscales_current_values/muR2_current,muF12_current,
     &                              muF22_current,QES2_current

c Re-scale of the shower starting scale
      double precision shower_scale_factor
      common/cshower_scale_factor/shower_scale_factor
c
c     Collider
c
      integer                                        lpp(2)
      double precision    ebeam(2), xbk(2),q2fact(2)
      common/to_collider/ ebeam   , xbk   ,q2fact,   lpp
c
c     BW treatment
c      
      double precision bwcutoff
      common/to_bwcutoff/ bwcutoff
c
c     kT/pT scheme for xqcut, clustering according to channel
c
      integer ktscheme
      logical chcluster,pdfwgt
      common/to_cluster/ktscheme,chcluster,pdfwgt

c
c     Shower MonteCarlo
c     
      character*10 shower_mc
      common /cMonteCarloType/shower_mc

      logical mcatnlo_delta
      common /cMCatNLO_Delta/ mcatnlo_delta

c
c     Reweight info
c
      logical do_rwgt_scale, do_rwgt_pdf, store_rwgt_info
      integer pdf_set_min, pdf_set_max
      double precision rw_Fscale_down, rw_Fscale_up, rw_Rscale_down, rw_Rscale_up
      common/to_rwgt/ do_rwgt_scale, rw_Fscale_down, rw_Fscale_up, rw_Rscale_down, rw_Rscale_up,
     & do_rwgt_pdf, pdf_set_min, pdf_set_max,
     & store_rwgt_info 
c
c    For FO run (with lhe type of analysis
c
      double precision FO_LHE_weight_ratio
      common /FO_ANALYSIS_LHW/FO_LHE_weight_ratio
c
c
c
      integer pdg_cut(0:25)
      double precision ptmin4pdg(0:25) 
      double precision ptmax4pdg(0:25) 
      double precision mxxmin4pdg(0:25) 
      logical mxxpart_antipart(1:25) 
      common/TO_PDG_SPECIFIC_CUT/pdg_cut, ptmin4pdg,ptmax4pdg, mxxmin4pdg, mxxpart_antipart

C stuff for alpha running in lepton collisions
      integer alphascheme, nlep_run, nupq_run, ndnq_run, w_run
      common/to_alphascheme/alphascheme, nlep_run, nupq_run, ndnq_run, w_run

      logical photons_from_lepton, has_bstrahl
      common /to_afromee/ photons_from_lepton
C  PDFs with beamstrahlung use specific initialisation/evaluation
      common /to_has_bs/ has_bstrahl


c for biassed generation based on the particle flavours of the processes
      integer flavour_bias(0:2)
      common /c_flavour_bias/ flavour_bias