c Test fixture: ordinary source repartition at a fixed outer real point.
      subroutine source_h_audit(owner,ids,lab,h,normalization)
      use fks_phase_space, only: fks_phase_space_point,
     $     generate_native_momenta
      use fks_phase_space_data, only: p_born,p_born_ev,p_ev,xi_i_fks_ev,
     $     y_ij_fks_ev,xinorm_ev
      use mc_native_context, only: set_native_history,native_mapping,
     $     mc_begin_real_point,mc_end_real_point
      use mc_counterterms, only: fksfather,mc_shower_scale_mass
      use mint_module, only: iconfig
      use weight_lines
      use process_module
      use scale_module
      implicit none
      include 'nexternal.inc'
      include 'genps.inc'
      include 'nFKSconfigs.inc'
      include 'mc_histories.inc'
      type(fks_phase_space_point) outer
      integer owner,ids(nexternal),i_fks,j_fks,nconfigs,
     $     configs(lmaxconfigs),groups(lmaxconfigs)
      common/fks_indices/i_fks,j_fks
      integer this_config
      common/to_mconfigs/this_config
      double precision lab(0:3,nexternal),p(0:3,nexternal),
     $     pl(0:3,nexternal),pc(0:3,nexternal),h,jac,born,bfac,
     $     fks_Sij,mc_outer_channel_weight,macnlops_flavour_sum,
     $     normalization,sector
      external fks_Sij,mc_outer_channel_weight,macnlops_flavour_sum
      double precision dsym
      common/dsymfactor/dsym
      double precision sym,symborn,symdeg
      integer ngluons,nquarks(-6:6),nphotons
      common/numberofparticles/sym,symborn,symdeg,
     $     ngluons,nquarks,nphotons
      logical valid,ok
      native_mapping=.true.
      call update_fks_dir(owner)
      call set_native_history(MC_HIST_OWN(owner))
      call init_process_module_nbody_wrapper()
      call update_coltype_and_charge(owner,i_fks,j_fks)
      call native_recoil_groups(nconfigs,configs,groups)
      iconfig=configs(1)
      this_config=iconfig
      call select_native_recoil(iconfig)
      call generate_native_momenta(lab,p,pl,pc,jac,valid,outer)
      if(.not.valid)stop 'invalid audit outer'
      p_born_ev=p_born
      p_ev=p
      call set_cms_stuff(0)
      call set_alphaS(p_born)
      call sborn_native(p_born,born)
      call get_born_flow(born_flow_picked,bfac)
      call init_process_module_n1body_wrapper(born_flow_picked)
      call compute_shower_scale_nbody(p_born,-fksfather,
     $     mc_shower_scale_mass())
      call compute_shower_scale_n1body(p,i_fks,j_fks,
     $     mc_shower_scale_mass())
      call compute_prefactors_n1body(1d0,jac)
      emsca_H=0d0
      call set_cms_stuff(-100)
      sector=fks_Sij(p,i_fks,j_fks,xi_i_fks_ev,y_ij_fks_ev)
      dsym=1d0
      normalization=sector*xinorm_ev*xi_i_fks_ev*jac*sym*
     $     mc_outer_channel_weight()
      icontr=0
      mc_H_only=.false.
      mc_S_only=.false.
      call mc_begin_real_point(p)
      call repartition_MC_H(1,p,pl,pc,jac,1d0,1d0,bfac,outer,ok)
      if(.not.ok)stop 'source repartition failed'
      call include_PDF_and_alphas
      h=macnlops_flavour_sum(ids)/normalization
      call mc_end_real_point()
      icontr=0
      call set_native_history(0)
      native_mapping=.false.
      end
