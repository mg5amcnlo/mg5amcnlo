c**************************************************************
c    run.inc 
c**************************************************************
c
c     Scales
c
      real*8          scale,scalefact,alpsfact
      logical         fixed_ren_scale,fixed_fac_scale,fixed_couplings,hmult
      integer         ickkw,nhmult,asrwgtflavor,dynamical_scale_choice
      common/to_scale/scale,scalefact,alpsfact,fixed_ren_scale,fixed_fac_scale,
     $                fixed_couplings,ickkw,nhmult,hmult,asrwgtflavor,dynamical_scale_choice
c
c     Collider
c
      integer                                        lpp(2)
      double precision    ebeam(2), xbk(2),q2fact(2)
      common/to_collider/ ebeam   , xbk   ,q2fact,   lpp
c
c     Number of events
c
c      integer nevents, gevents
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
c     Parameters for systematics variations info
c
c      logical use_syst
c     Common block for systematics variations
c      DOUBLE PRECISION s_scale
c      INTEGER n_qcd,n_alpsem
c      DOUBLE PRECISION s_qalps(max_particles-2)
c      INTEGER n_pdfrw(2),i_pdgpdf(max_particles-2,2)
c      DOUBLE PRECISION s_xpdf(max_particles-2,2),s_qpdf(max_particles-2,2)
c      DOUBLE PRECISION s_rwfact
c      COMMON/TO_SYST/use_syst,n_qcd,n_alpsem,n_pdfrw,i_pdgpdf,
c     $               s_scale,s_qalps,s_xpdf,s_qpdf,s_rwfact
c
c     Include <clustering> tag for Pythia 8 CKKW-L matching
c
      logical clusinfo
      COMMON/TO_CINFO/clusinfo
