c**************************************************************
c    run.inc 
c**************************************************************
c
c     Scales
c
      real*8          scale,scalefact,alpsfact
      logical         fixed_ren_scale,fixed_fac_scale,fixed_couplings,hmult
      integer         ickkw,nhmult
      common/to_scale/scale,scalefact,alpsfact,fixed_ren_scale,fixed_fac_scale,
     $                fixed_couplings,ickkw,nhmult,hmult
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
