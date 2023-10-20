c**************************************************************
c    run.inc 
c**************************************************************
c
c     Scales
c
      real*8          scale,scalefact,alpsfact,mue_ref_fixed,mue_over_ref
      logical         fixed_ren_scale,fixed_fac_scale1, fixed_fac_scale2,fixed_couplings,hmult
      logical         fixed_extra_scale
      integer         ickkw,nhmult,asrwgtflavor, dynamical_scale_choice,ievo_eva
      
      common/to_scale/scale,scalefact,alpsfact, mue_ref_fixed, mue_over_ref,
     $                fixed_ren_scale,fixed_fac_scale1, fixed_fac_scale2,
     $                fixed_couplings, fixed_extra_scale,ickkw,nhmult,hmult,asrwgtflavor,
     $                dynamical_scale_choice
      common/to_eva/ievo_eva
c
c     Collider
c
      integer                                        lpp(2)
      double precision    ebeam(2), xbk(2),q2fact(2)
      common/to_collider/ ebeam   , xbk   ,q2fact,   lpp

      integer frame_id
      common/to_frame_me/frame_id
c
c     Number of events
c
      integer nevents, gevents
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
      logical use_syst
c     Common block for systematics variations
      DOUBLE PRECISION s_scale
      INTEGER n_qcd,n_alpsem
      DOUBLE PRECISION s_qalps(max_particles-2)
      INTEGER n_pdfrw(2),i_pdgpdf(max_particles-2,2)
      DOUBLE PRECISION s_xpdf(max_particles-2,2),s_qpdf(max_particles-2,2)
      DOUBLE PRECISION s_rwfact
      COMMON/TO_SYST/use_syst,n_qcd,n_alpsem,n_pdfrw,i_pdgpdf,
     $               s_scale,s_qalps,s_xpdf,s_qpdf,s_rwfact
c
c     Flag on how to write the LHE events 
c     Include <clustering> tag for Pythia 8 CKKW-L matching
c
      logical clusinfo
      double precision lhe_version
      COMMON/TO_LHEFORMAT/lhe_version,clusinfo

c     
C     Controls wheter to perform Monte-Carlo sampling over grouped subprocesses
C
      logical MC_grouped_subproc
      common/to_MC_grouped_subproc/MC_grouped_subproc

C
C     block for heavy ion beam
C
      integer nb_proton(2), nb_neutron(2)
      common/to_heavyion_pdg/ nb_proton, nb_neutron
      double precision mass_ion(2)
      common/to_heavyion_mass/mass_ion

C
C Controls what are the PDGs included in the CKKWl merging procedure, i.e. what
C are the PDGs subject to the ktdurham cut
C
      integer pdgs_for_merging_cut(0:1000)
      common/TO_MERGE/pdgs_for_merging_cut
c
c
c
      integer pdg_cut(0:25)
      double precision ptmin4pdg(0:25) 
      double precision ptmax4pdg(0:25) 
      double precision Emin4pdg(0:25) 
      double precision Emax4pdg(0:25) 
      double precision etamin4pdg(0:25) 
      double precision etamax4pdg(0:25) 
      double precision mxxmin4pdg(0:25) 
      logical mxxpart_antipart(1:25)
      common/TO_PDG_SPECIFIC_CUT/pdg_cut, ptmin4pdg,ptmax4pdg, Emin4pdg, Emax4pdg, etamin4pdg,
     &etamax4pdg, mxxmin4pdg,mxxpart_antipart

       double precision small_width_treatment
       common/narrow_width/small_width_treatment

       double precision tmin_for_channel
       integer sde_strat ! 1 means standard single diagram enhancement strategy,
c                          2 means approximation by the denominator of the propagator
       common/TO_CHANNEL_STRAT/tmin_for_channel, sde_strat