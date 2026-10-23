c Test fixture: independent native H/S real and MC record pairing.
      subroutine source_pairs_audit(owner,ids,lab,result)
      use fks_phase_space, only: generate_native_momenta
      use fks_phase_space_data, only: p_born,p_ev,p_born_ev,
     $     xi_i_fks_ev,xinorm_ev
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
      integer owner,ids(nexternal),i_fks,j_fks,nconfigs,
     $     configs(lmaxconfigs),groups(lmaxconfigs),i,j,k
      common/fks_indices/i_fks,j_fks
      integer this_config
      common/to_mconfigs/this_config
      double precision lab(0:3,nexternal),p(0:3,nexternal),
     $     pl(0:3,nexternal),pc(0:3,nexternal),jac,born,bfac,
     $     result(6),normalization,probne
      double precision sym,symborn,symdeg
      double precision dsym
      common/dsymfactor/dsym
      integer ngluons,nquarks(-6:6),nphotons
      common/numberofparticles/sym,symborn,symdeg,
     $     ngluons,nquarks,nphotons
      logical valid
      result=0d0
      native_mapping=.true.
      call update_fks_dir(owner)
      call set_native_history(MC_HIST_OWN(owner))
      call init_process_module_nbody_wrapper()
      call update_coltype_and_charge(owner,i_fks,j_fks)
      call native_recoil_groups(nconfigs,configs,groups)
      iconfig=configs(1)
      this_config=iconfig
      call select_native_recoil(iconfig)
      call generate_native_momenta(lab,p,pl,pc,jac,valid)
      if(.not.valid)stop 'invalid pairs-audit outer'
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
      dsym=1d0
      emsca_H=0d0
      normalization=xinorm_ev*xi_i_fks_ev*jac*sym
      icontr=0
      mc_H_only=.false.
      mc_S_only=.false.
      call mc_begin_real_point(p)
      call compute_native_NLOPS_weights(p,pc,.true.,.true.,probne)
      result(6)=probne
      call include_PDF_and_alphas
      do i=1,icontr
         k=0
         if(itype(i).eq.1)k=1
         if(itype(i).eq.11)k=2
         if(itype(i).eq.12)k=3
         if(itype(i).eq.13)k=4
         if(k.eq.0)cycle
         do j=1,niproc(i)
            if(all(parton_pdg(:,j,i).eq.ids))
     $           result(k)=result(k)+parton_iproc(j,i)/normalization
         enddo
      enddo
c An independent undivided real call checks normalization, rather than
c comparing only the two complementary factors against each other.
      icontr=0
      macnlops_gate_active=.false.
      call set_cms_stuff(-100)
      call set_alphaS(p)
      call compute_real_emission(p,1d0)
      call include_PDF_and_alphas
      do i=1,icontr
         if(itype(i).ne.1)cycle
         do j=1,niproc(i)
            if(all(parton_pdg(:,j,i).eq.ids))
     $           result(5)=result(5)+parton_iproc(j,i)/normalization
         enddo
      enddo
      call mc_end_real_point()
      icontr=0
      call set_native_history(0)
      native_mapping=.false.
      end
