      subroutine repartition_MC_H(first_native,x_outer,p,p_lab,p_cms,
     $     jacPS,vegas_wgt,sampling_wgt,born_flow_factor)
! At a fixed real point form Hhat_a = S_a sum_b P_b (S_b R - M_b).
! The ordinary S records have already been made and are not changed.
! Each M_b includes its native G replacement, luminosities and Born map.
! Each inner history samples its OWN colour flow and includes 1/q_b,c.
! The outer event colour is only the event owner, not an inner proposal.
! Thus the colour-sampled summand is
! P_b,c*(p_b,c*S_b*R-M_b,c)/q_b,c, with M_b,c already flow-weighted.
      use mc_native_context, only: native_metadata,set_native_history,
     $     native_mapping,active_context,local_context
      use weight_lines, only: icontr,H_event,wgt,event_nFKS,momenta,
     $     itype,
     $     momenta_m,y_bst,need_match,mc_H_only
      use mint_module, only: ndim,iconfig
      use process_module, only: ndelH
      use kinematics_module
      use scale_module
      implicit none
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      include 'genps.inc'
      include 'run.inc'
      include 'fks_symmetry.inc'
      include 'mc_histories.inc'
      integer first_native,last_native,first_alt,owner,iFKS,ii,jj,ihist,
     $     ict,flow_save,called_save,owner_match(nexternal),
     $     outer_config,native_config
      double precision x_outer(99),p(0:3,nexternal),
     $     p_lab(0:3,nexternal),p_cms(0:3,nexternal),jacPS,vegas_wgt,
     $     sampling_wgt,born_flow_factor,outer_measure,native_measure,
     $     sector_weight,factor,xx(99),jac_native,xbjrk_born(2),
     $     p_flipped(0:3,nexternal),pn(0:3,nexternal),
     $     pn_lab(0:3,nexternal),pn_cms(0:3,nexternal),probne_native,
     $     flow_factor_native,rwgt,outer_boost,gfun_save(3),
     $     outer_channel,mc_outer_channel_weight
      double precision nbody_scales_save(nexternal-1,nexternal-1,3),
     $     n1body_scales_save(nexternal,nexternal),
     $     emsca_save(fks_configs,ndelH,ndelH)
      logical cuts_born,cuts_real,passcuts
      double precision fks_Sij
      external fks_Sij,passcuts
      double precision born_weight
      external mc_outer_channel_weight
      integer nFKSprocess,i_fks,j_fks
      common/c_nFKSprocess/nFKSprocess
      common/fks_indices/i_fks,j_fks
      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born
      double precision pmass(nexternal)
      common/to_mass/pmass
      double precision xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev(0:3),
     $     p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt
      double precision xinorm_ev
      common/cxinormev/xinorm_ev
      double precision fkssymmetryfactor,fkssymmetryfactorBorn,
     $     fkssymmetryfactorDeg
      integer ngluons,nquarks(-6:6),nphotons
      common/numberofparticles/fkssymmetryfactor,fkssymmetryfactorBorn,
     $     fkssymmetryfactorDeg,ngluons,nquarks,nphotons
      double precision p1_cnt(0:3,nexternal,-2:2),wgt_cnt(-2:2),
     $     pswgt_cnt(-2:2),jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     $     sqrtshat,shat
      integer fold,ifold_counter,MCcntcalled
      common/cfl/fold,ifold_counter
      common/c_MCcntcalled/MCcntcalled
      integer need_matching_S(nexternal),need_matching_H(nexternal)
      common/c_need_matching/need_matching_S,need_matching_H
      integer icolup_s(2,nexternal-1),icolup_h(2,nexternal),
     $     colours_s_save(2,nexternal-1),colours_h_save(2,nexternal)
      common/colour_connections/icolup_s,icolup_h
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn

      integer,save::audit_points=0,audit_foreign=0
      integer audit_valid
      double precision audit_ref(3,15),audit_now(3,15),audit_rel
      if (p(0,1).le.0d0 .or. jacPS.le.0d0) return
      if (MC_HIST_COUNT.eq.0) return
      call set_cms_stuff(-100)
      owner=nFKSprocess
      if (.not.MC_HIST_COMPLETE(owner)) then
         write (*,*) 'Incomplete MC history sum for FKS sector',owner
         write (*,*) 'Required global histories are unresolved or',
     $        ' ambiguous; see Source/BornSupport/registry.json'
         stop 1
      endif
      outer_config=iconfig
      outer_channel=mc_outer_channel_weight()
      outer_boost=ybst_til_tolab
      owner_match=need_matching_H
      sector_weight=fks_Sij(p,i_fks,j_fks,xi_i_fks_ev,y_ij_fks_ev)
      last_native=icontr
      do ict=first_native,last_native
         if (H_event(ict)) wgt(:,ict)=0d0
      enddo
      if (sector_weight.eq.0d0) return

! K_a is the full outer real measure, including its sampling probability
! and orbit multiplicity, and the outer matrix-element channel weight.
      outer_measure=xinorm_ev*xi_i_fks_ev*jacPS*vegas_wgt*
     $     sampling_wgt*fkssymmetryfactor*outer_channel
      if (outer_measure.le.0d0) return
      if (fkssymmetryfactor.ne.
     $     dble(FKS_FAC_I_D(owner)*FKS_FAC_J_D(owner))) then
         write (*,*) 'Incorrect outer MC H orbit factor',owner
         stop 1
      endif
      if (MC_HIST_OWN(owner).lt.MC_HIST_FIRST(owner) .or.
     $     MC_HIST_OWN(owner).gt.MC_HIST_LAST(owner)) then
         write (*,*) 'Missing native entry in MC history table',owner
         stop 1
      endif
      gfun_save=[gfactsf,gfactcl,gfactazi]
      flow_save=born_flow_picked
      called_save=MCcntcalled
      colours_s_save=icolup_s
      colours_h_save=icolup_h
      nbody_scales_save(:,:,1)=shower_scale_nbody
      nbody_scales_save(:,:,2)=shower_scale_nbody_min
      nbody_scales_save(:,:,3)=shower_scale_nbody_max
      n1body_scales_save=shower_scale_n1body
      emsca_save=emsca_H(:,ifold_counter,:,:)
      mc_H_only=.true.
      native_mapping=.true.

! The exporter supplies unique ordered histories, including both gg
! orientations, with native Born/order identities and checked label maps.
! Every row has unit multiplicity. Recompute the owner too, using the
! same measure conversion and channel-free density as all other rows.
      do ihist=MC_HIST_FIRST(owner),MC_HIST_LAST(owner)
         iFKS=MC_HIST_NATIVE(ihist)
         ii=MC_HIST_I(ihist)
         jj=MC_HIST_J(ihist)
         iconfig=1
         call update_fks_dir(iFKS)
         call set_native_history(ihist)
         call init_process_module_nbody_wrapper()
         call update_coltype_and_charge(iFKS,i_fks,j_fks)
         if (fkssymmetryfactor.ne.
     $        dble(FKS_FAC_I_D(iFKS)*FKS_FAC_J_D(iFKS)) .or.
     $        MC_HIST_PERM(i_fks,ihist).ne.ii .or.
     $        MC_HIST_PERM(j_fks,ihist).ne.jj) then
            write (*,*) 'Incorrect native MC H orbit',owner,ihist
            stop 1
         endif
         call apply_momentum_permutation(MC_HIST_PERM(:,ihist),
     $        p_lab,p_flipped)
! Select the first native mapping with an invertible physical point.
! The outer channel index has no meaning in a different Born topology.
         audit_valid=0
         do native_config=1,native_metadata%configurations(0)
            iconfig=native_config
            xx=0d0
            jac_native=1d0
            call generate_lab_momenta_inverse(ndim,iconfig,
     $           jac_native,xx,p_flipped,xbjrk_born)
            if(jac_native.le.0d0)cycle
         if (jac_native.le.0d0) then
            write (*,*) 'Cannot invert native MC H history',
     $           owner,iFKS,ii,jj
            stop 1
         endif

! Inversion alone does not fill the native FKS counterevents. Replay
! the forward map to obtain those points AND their limit measures.
         calculatedBorn=.false.
         jac_native=1d0
         call generate_momenta(ndim,iconfig,jac_native,xx,
     $        pn,pn_lab,pn_cms)
         if (jac_native.le.0d0 .or. pn(0,1).le.0d0 .or.
     $        p_born(0,1).le.0d0) then
            write (*,*) 'Cannot replay native MC H history',
     $           owner,iFKS,ii,jj
            stop 1
         endif
         if (maxval(abs(pn_lab-p_flipped)).gt.
     $        1d-7*max(1d0,maxval(abs(p_flipped)))) then
            write (*,*) 'MC H inverse/forward point mismatch',
     $           owner,iFKS,ii,jj
            stop 1
         endif
         native_measure=xinorm_ev*xi_i_fks_ev*jac_native*
     $        fkssymmetryfactor
         if (native_measure.le.0d0) then
            write (*,*) 'Invalid native MC H measure',
     $           owner,iFKS,ii,jj,native_measure
            stop 1
         endif

! Divide out the native real measure of the COMPLETE generated H weight
! and insert K_a. This is not an extra Jacobian on a raw MC density:
! its native K_b cancels exactly. The counter/real measure ratios inside
! the G replacement, however, must be retained. The inner orbit factor
! cancels too, since the labelled histories are explicitly enumerated.
         factor=sector_weight*outer_measure/native_measure
         MCcntcalled=0
         call fill_kinematics_module(pn_cms,i_fks,j_fks,
     $        xi_i_fks_ev,y_ij_fks_ev,pmass(j_fks),.false.)
         call compute_prefactors_n1body(1d0,jac_native)
         if (ickkw.eq.3) then
            call set_FxFx_scale(0,pn)
            call set_cms_stuff(0)
            call set_FxFx_scale(2,p1_cnt(0,1,0))
            call set_cms_stuff(-100)
            call set_FxFx_scale(3,pn)
         endif
         call set_cms_stuff(0)
         if (ickkw.eq.3) call set_FxFx_scale(-2,p1_cnt(0,1,0))
! Sample in this history's own Born basis. Reusing the outer label
! would require a flow map and support at a different Born point.
! q_b,c=p_b,c here; no additional outer 1/q_a,c belongs on this term.
         call set_alphaS(p1_cnt(0,1,0))
         calculatedBorn=.false.
         call sborn_native(p_born,born_weight)
         call get_born_flow(born_flow_picked,flow_factor_native)
         calculatedBorn=.false.
         call include_born_flow_weight(flow_factor_native,
     $        flow_factor_native)
         call init_process_module_n1body_wrapper(born_flow_picked)
         call compute_shower_scale_nbody(p_born,-fksfather)
         call compute_shower_scale_n1body(pn,i_fks,j_fks)
         cuts_born=passcuts(p1_cnt(0,1,0),rwgt)
         call set_cms_stuff(-100)
         if (ickkw.eq.3) call set_FxFx_scale(-3,pn)
         cuts_real=passcuts(pn,rwgt)
         first_alt=icontr+1
         call compute_native_NLOPS_weights(pn,pn_lab,pn_cms,
     $        jac_native,cuts_born,cuts_real,probne_native)
         do ict=first_alt,icontr
            if (.not.H_event(ict)) then
               write (*,*) 'S event entered the inner MC H sum',ihist
               stop 1
            endif
            wgt(:,ict)=wgt(:,ict)*factor
! Keep BOTH native momentum sets for ME reweighting, expressed in the
! outer frame. Only event kinematics and shower ownership are outer.
            event_nFKS(ict)=owner
            call boost_n1_to_lab(momenta_m(:,:,1,ict),pn_cms,
     $           y_bst(ict)-outer_boost)
            momenta_m(:,:,1,ict)=pn_cms
            call boost_n1_to_lab(momenta_m(:,:,2,ict),pn_cms,
     $           y_bst(ict)-outer_boost)
            momenta_m(:,:,2,ict)=pn_cms
            momenta(:,:,ict)=p
            y_bst(ict)=outer_boost
            need_match(:,ict)=owner_match
         enddo
         audit_now=0d0
         do ict=first_alt,icontr
            audit_now(:,itype(ict))=audit_now(:,itype(ict))+wgt(:,ict)
         enddo
         audit_valid=audit_valid+1
         if(audit_valid.eq.1)then
            audit_ref=audit_now
         else
            audit_rel=maxval(abs(audit_now-audit_ref))/
     $           max(1d-50,maxval(abs(audit_now)),maxval(abs(audit_ref)))
            if(audit_rel.gt.1d-6)then
               write(*,*)'FAILED native mapping audit',owner,iFKS,
     $              ihist,native_config,audit_rel
               write(*,*)audit_ref
               write(*,*)audit_now
               stop 77
            endif
            audit_points=audit_points+1
            if(active_context.ne.local_context)audit_foreign=
     $           audit_foreign+1
            if(audit_points.ge.1000)then
               write(*,*)'PASS native mapping audit',audit_points,
     $              audit_foreign
               stop 78
            endif
         endif
         icontr=first_alt-1
         enddo
         if(audit_valid.eq.0)stop 'No valid audit map'
      enddo
! Replaying the saved OUTER random numbers restores all FKS event and
! counterevent COMMON blocks, including Born/spin and Bjorken data.
      call set_native_history(0)
      native_mapping=.false.
      iconfig=outer_config
      call update_fks_dir(owner)
      call init_process_module_nbody_wrapper()
      call update_coltype_and_charge(owner,i_fks,j_fks)
      calculatedBorn=.false.
      jac_native=sampling_wgt
      call generate_momenta(ndim,iconfig,jac_native,x_outer,
     $     pn,pn_lab,pn_cms)
      if (jac_native.le.0d0 .or. p_born(0,1).le.0d0) then
         write (*,*) 'Could not restore outer FKS point after MC H sum'
         stop 1
      endif
      born_flow_picked=flow_save
      call init_process_module_n1body_wrapper(born_flow_picked)
      shower_scale_nbody=nbody_scales_save(:,:,1)
      shower_scale_nbody_min=nbody_scales_save(:,:,2)
      shower_scale_nbody_max=nbody_scales_save(:,:,3)
      shower_scale_n1body=n1body_scales_save
      emsca_H(:,ifold_counter,:,:)=emsca_save
      icolup_s=colours_s_save
      icolup_h=colours_h_save
      MCcntcalled=called_save
      call fill_kinematics_module(p_cms,i_fks,j_fks,
     $     xi_i_fks_ev,y_ij_fks_ev,pmass(j_fks),.false.)
      gfactsf=gfun_save(1)
      gfactcl=gfun_save(2)
      gfactazi=gfun_save(3)
      call compute_prefactors_n1body(vegas_wgt,jac_native)
      call include_born_flow_weight(born_flow_factor,born_flow_factor)
      if (ickkw.eq.3) then
         call set_FxFx_scale(0,p)
         call set_cms_stuff(0)
         call set_FxFx_scale(2,p1_cnt(0,1,0))
         call set_cms_stuff(-100)
         call set_FxFx_scale(3,p)
      endif
      call set_cms_stuff(-100)
      call set_alphaS(p)
      calculatedBorn=.false.
      mc_H_only=.false.
      end

