c**************************************************************
c    run.inc 
c**************************************************************
c
c     Scales
c
      real*8          scale,scalefact,ellissextonfact,alpsfact
      logical         fixed_ren_scale,fixed_fac_scale,fixed_couplings,hmult
      integer         ickkw,nhmult,dynamical_scale_choice
      common/to_scale/scale,scalefact,ellissextonfact,alpsfact,fixed_ren_scale,
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
     #                 QES_over_ref,muF_over_ref
      common/cscales_fact/muR_over_ref,muF1_over_ref,muF2_over_ref,
     #                    QES_over_ref,muF_over_ref

c Values of fixed reference scales, to be given in input
      double precision muR_ref_fixed,muF1_ref_fixed,
     #                 muF2_ref_fixed,QES_ref_fixed,muF_ref_fixed
      common/cscales_fixed_values/muR_ref_fixed,muF1_ref_fixed,
     #                 muF2_ref_fixed,QES_ref_fixed,muF_ref_fixed

c Values of the current scales squared. The information is partly
c redundant with that of common block /to_scale/
      double precision muR2_current,muF12_current,
     #                 muF22_current,QES2_current
      common/cscales_current_values/muR2_current,muF12_current,
     #                              muF22_current,QES2_current

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
c     Sudakov grid file name
c
      character*130 issgridfile
      common/to_sgridfile/issgridfile
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
c
c     Reweight info
c
      logical do_rwgt_scale, do_rwgt_pdf, store_rwgt_info
      integer pdf_set_min, pdf_set_max
      double precision rw_Fscale_down, rw_Fscale_up, rw_Rscale_down, rw_Rscale_up
      common/to_rwgt/ do_rwgt_scale, rw_Fscale_down, rw_Fscale_up, rw_Rscale_down, rw_Rscale_up,
     # do_rwgt_pdf, pdf_set_min, pdf_set_max,
     # store_rwgt_info 

