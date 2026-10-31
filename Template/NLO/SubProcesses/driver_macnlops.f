      subroutine macnlops_evaluator_setup()
c One-time evaluator state, shared by the line protocol in
c macnlops_evaluator.f and the in-process first-emission program
c (macnlops_capi.f).
      use FKSParams
      use mint_module
      use process_module
      use scale_module
      use extra_weights
      implicit none
      include 'nexternal.inc'
      include 'genps.inc'
      include 'nFKSconfigs.inc'
      include 'born_nhel.inc'
      include 'run.inc'
      double precision rng_init,ran2
      external ran2
      double precision beam_xmax(2),ptmin_isr,ptmin_fsr
      common/macnlops_shower_config/beam_xmax,ptmin_isr,ptmin_fsr
      character*4 abrv
      common/to_abrv/abrv
      logical softtest,colltest,nbody,multi_channel
      common/sctests/softtest,colltest
      common/cnbody/nbody
      integer isum_hel
      common/to_matrix/isum_hel,multi_channel
      double precision alsf,besf,alazi,beazi
      common/cgfunsfp/alsf,besf
      common/cgfunazi/alazi,beazi
      integer fold,ifold_counter,lbw(0:nexternal)
      common/cfl/fold,ifold_counter
      common/to_BW/lbw

      call timing_reset()
      fixed_order=.false.
      nlo_ps=.true.
      call FKSParamReader(paramFileName,.true.,.false.)
      call setrun
      if(macnlops)call macnlops_validate_profile
      if(nincoming.ne.2.or.shower_mc.ne.'PYTHIA8'.or.
     $     ickkw.ne.0.or.(mcatnlo_delta.and.
     $     macnlops_resolved_source)) then
         write(*,*) 'MacNLOPS requires two incoming beams, ',
     $        'PYTHIA8 and unmerged ordinary/Delta MC@NLO input'
         stop 1
      endif
      call setpara('param_card.dat')
      call setcuts
      call fill_configurations_common
      call check_amp_split
      call init_process_module_global(shower_mc,'all ',nexternal,
     $     nincoming,mcatnlo_delta,ebeam(1)+ebeam(2),max_bcol,maxflow,0)
      call init_scale_module(nexternal,shower_scale_factor,
     $     fks_configs,1)
c Use the source driver's input_app.txt defaults. Ordinary matching
c retains G/P; macnlops_resolved_source selects the legacy convex source.
      alsf=1d0
      besf=-0.1d0
      alazi=-1d0
      beazi=-0.1d0
      softtest=.false.
      colltest=.false.
      nbody=.false.
      abrv='all '
      multi_channel=.false.
      isum_hel=0
      fold=0
      ifold_counter=1
      lbw=0
      imode=1
      iconfig=1
c Initialize ntuple's one-time RANMAR setup before explicit per-request
c reseeding. A SAMPLE request never depends on a previous point or on
c the actual S event's SCALUP/colour/shower random numbers.
      if(mcatnlo_delta)rng_init=ran2()
      doreweight=.false.
      separate_flavour_configs=.false.
c Apply physical flow weights explicitly below. Global requests sum all
c flows; local requests retain only their selected flow.
      MCSubtractionAtFixedFlow=.false.
c Standalone density audits can request the uncut scalar kernel. The
c production runner supplies actual shower beam bounds and cutoffs.
      beam_xmax=1d0
      ptmin_isr=0d0
      ptmin_fsr=0d0
      end


      subroutine macnlops_density(owner,ids,lab,r,d,k,status)
c Preserve the global V1/V2 audit interface. The V3 path below selects
c one native sector, physical diagram partition and Born flow instead.
      implicit none
      include 'nexternal.inc'
      integer owner,ids(nexternal),status
      integer born_ids(nexternal-1),born_colours(2,nexternal-1)
      double precision lab(0:3,nexternal),r,d,k,
     $     born_lab(0:3,nexternal-1),born_masses(nexternal-1)
      born_ids=0
      born_colours=0
      born_lab=0d0
      born_masses=0d0
      call macnlops_density_impl(owner,ids,lab,.false.,0,0,0,0,
     $     1d0,.false.,born_ids,born_colours,born_lab,born_masses,
     $     r,d,k,status)
      end


      subroutine macnlops_component_density(owner,ids,lab,
     $     component_config,component_graph,component_flow,damping_mode,
     $     diagram_symmetry,has_component_born,born_ids,born_colours,
     $     born_lab,born_masses,r,d,k,status)
c Source sampling probabilities never enter these physical densities.
c Ordinary matching retains its bogus P in H; Delta matching requests
c H before its physical Delta multiplier, without evaluating a Delta.
      implicit none
      include 'nexternal.inc'
      integer owner,ids(nexternal),component_config,component_graph,
     $     component_flow,damping_mode,status,born_ids(nexternal-1),
     $     born_colours(2,nexternal-1)
      double precision lab(0:3,nexternal),diagram_symmetry,r,d,k,
     $     born_lab(0:3,nexternal-1),born_masses(nexternal-1)
      logical has_component_born
      call macnlops_density_impl(owner,ids,lab,.true.,
     $     component_config,component_graph,component_flow,damping_mode,
     $     diagram_symmetry,has_component_born,born_ids,born_colours,
     $     born_lab,born_masses,r,d,k,status)
      end


      subroutine macnlops_density_impl(owner,ids,lab,
     $     local_component,component_config,component_graph,component_flow,
     $     damping_mode,diagram_symmetry,has_component_born,born_ids,
     $     born_colours,born_lab,born_masses,r,d,k,status)
      use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
      use fks_phase_space, only: generate_native_momenta,
     $     fks_phase_space_point
      use fks_phase_space_data, only: p_born,p1_cnt,xinorm_ev,
     $     xi_i_fks_ev,y_ij_fks_ev,xbjrk_cnt
      use fks_phase_space_helpers, only: apply_momentum_permutation
      use mc_native_context, only: set_native_history,native_mapping,
     $     mc_begin_real_point,mc_end_real_point,active_context,
     $     local_context
      use mc_counterterms, only: fksfather,mc_shower_scale_mass,
     $     get_qMC,macnlops_delta_profile
      use process_module
      use scale_module
      use weight_lines
      use mint_module, only: iconfig
      implicit none
      include 'nexternal.inc'
      include 'genps.inc'
      include 'nFKSconfigs.inc'
      include 'born_nhel.inc'
      include 'born_conf.inc'
      include 'mc_histories.inc'
      integer owner,ids(nexternal),status,hist,sector,flow,
     $     n_recoil,recoil,configs(lmaxconfigs),
     $     groups(lmaxconfigs),i_fks,j_fks,MCcntcalled,kernel_status,
     $     leg,hist_first,hist_last,recoil_first,recoil_last,
     $     component_config,component_graph,component_flow,damping_mode,
     $     born_ids(nexternal-1),born_colours(2,nexternal-1),
     $     equivalent_emitters,mother,sister
      common/fks_indices/i_fks,j_fks
      common/c_MCcntcalled/MCcntcalled
      double precision lab(0:3,nexternal),r,d,k,h,
     $     flipped(0:3,nexternal),p(0:3,nexternal),
     $     plab(0:3,nexternal),pcms(0:3,nexternal),jac,measure,
     $     born,flows(max_bcol),total,probne,rwgt,reference,
     $     value,macnlops_flavour_sum,model_mass,mass2,tolerance,
     $     get_mass_from_id,hard_scale2,diagram_symmetry,
     $     fks_Sij,fks_Hij
      double precision component_normalization
      double precision born_lab(0:3,nexternal-1),
     $     born_masses(nexternal-1)
      external macnlops_flavour_sum,get_mass_from_id,fks_Sij,fks_Hij
      double precision factors(9),mc_factors(8),pdf_factors(3),
     $     factors_save(9),mc_save(8),pdf_save(3)
      common/factor_n1body/factors
      common/factor_n1body_NLOPS/mc_factors
      common/factor_pdfsch/pdf_factors
      double precision sym,symborn,symdeg
      integer ngluons,nquarks(-6:6),nphotons
      common/numberofparticles/sym,symborn,symdeg,
     $     ngluons,nquarks,nphotons
      logical valid,cuts_born,cuts_real,passcuts,
     $     passcuts_native_born,is_leading(max_bcol),local_component,
     $     has_component_born
      integer nleading
      common/c_leading_cflows/is_leading,nleading
      integer this_config
      common/to_mconfigs/this_config
      external passcuts,passcuts_native_born
      type(fks_phase_space_point) native_point

      status=1
      r=0d0
      macnlops_sudakov_ready=.false.
      h=0d0
      d=0d0
      k=0d0
      hist=0
      sector=owner
      recoil=0
      flow=0
      native_mapping=.false.
      mc_H_only=.false.
      mc_S_only=.false.
      macnlops_gate_active=.false.
      macnlops_proposal_active=.false.
      if(owner.lt.1.or.owner.gt.fks_configs)goto 900
c Pythia's forced heavy-flavour conversion can assign a massive emitted
c c/b sister even when particleData and the hard model use mass zero.
c Such a point has no inverse in the massless hard real phase space.
c Diagnose it before native reconstruction obscures the mass mismatch.
      hard_scale2=max(1d0,sum(abs(lab(0,1:nincoming)))**2,
     $     maxval(abs(lab))**2)
      do leg=nincoming+1,nexternal
         if(abs(ids(leg)).ne.4.and.abs(ids(leg)).ne.5)cycle
         model_mass=get_mass_from_id(ids(leg))
         mass2=lab(0,leg)**2-sum(lab(1:3,leg)**2)
c Shower boosts use the whole hard system. Their cancellation error
c can exceed the emitted soft leg's own E^2 scale. Momenta are written
c with 17 significant digits. Chained massive-recoil boosts accumulate
c cancellation beyond a single invariant evaluation. Allow that error
c at event scale while still resolving physical heavy-mass shifts.
         tolerance=4096d0*epsilon(1d0)*max(hard_scale2,
     $        lab(0,leg)**2,
     $        sum(lab(1:3,leg)**2),model_mass**2)
         if(abs(mass2-model_mass**2).gt.tolerance)then
            status=24
            write(*,'(a,2(1x,i6),3(1x,es26.17e3))')
     $           'MACNLOPS_MASS_MISMATCH_V1',leg,ids(leg),
     $           model_mass**2,mass2,tolerance
            goto 900
         endif
      enddo
      if(.not.local_component.and..not.MC_HIST_COMPLETE(owner))then
         status=2
         goto 900
      endif
      hist_first=MC_HIST_FIRST(owner)
      hist_last=MC_HIST_LAST(owner)
      if(local_component)then
         if(component_flow.lt.1.or.component_flow.gt.max_bcol.or.
     $      component_config.lt.1.or.component_graph.lt.1.or.
     $      damping_mode.lt.0.or.
     $      damping_mode.gt.1.or.
     $      (mcatnlo_delta_mod.neqv.(damping_mode.eq.1)).or.
     $      .not.ieee_is_finite(diagram_symmetry).or.
     $      diagram_symmetry.le.0d0)then
            status=30
            goto 900
         endif
         hist_first=MC_HIST_OWN(owner)
         hist_last=hist_first
         if(hist_first.lt.1.or.hist_first.gt.MC_HIST_COUNT)then
            status=31
            goto 900
         endif
         if(MC_HIST_NATIVE(hist_first).ne.owner.or.
     $      any(MC_HIST_PERM(:,hist_first).ne.
     $      [(leg,leg=1,nexternal)]))then
            status=31
            goto 900
         endif
      endif
      native_mapping=.true.
      mc_H_only=.true.
      mc_S_only=.false.
      nonfinite_contribution=.false.
      call mc_begin_real_point(lab)
      do hist=hist_first,hist_last
         delta_history=hist
         sector=MC_HIST_NATIVE(hist)
         call update_fks_dir(sector)
         call set_native_history(hist)
         if(local_component.and.active_context.ne.local_context)then
c A flow or graph ordinal has meaning only in its original Born basis.
c Cross-provider component transport needs an explicit certified map.
            status=35
            goto 900
         endif
         call init_process_module_nbody_wrapper()
         call update_coltype_and_charge(sector,i_fks,j_fks)
         call apply_momentum_permutation(MC_HIST_PERM(:,hist),
     $        lab,flipped,valid)
         if(.not.valid)then
            status=3
            goto 900
         endif
         call native_recoil_groups(n_recoil,configs,groups)
         recoil_first=1
         recoil_last=n_recoil
         if(local_component)then
            if(component_config.gt.lmaxconfigs)then
               status=32
               goto 900
            endif
            if(groups(component_config).le.0)then
               status=32
               goto 900
            endif
            if(mapconfig(component_config).ne.component_graph)then
               status=34
               goto 900
            endif
            recoil_first=groups(component_config)
            recoil_last=recoil_first
         endif
         do recoil=recoil_first,recoil_last
            iconfig=configs(recoil)
            if(local_component)iconfig=component_config
            delta_recoil=iconfig
            call select_native_recoil(iconfig)
            if(local_component.and.has_component_born)then
               call generate_native_momenta(flipped,p,plab,pcms,
     $              jac,valid,native_point)
            else
               call generate_native_momenta(flipped,p,plab,pcms,
     $              jac,valid)
            endif
            if(.not.valid)then
               status=4
               goto 900
            endif
            this_config=iconfig
            if(local_component.and.has_component_born)then
               call macnlops_validate_component_born(component_flow,
     $              born_ids,born_colours,born_lab,born_masses,
     $              native_point%born%xbjrk,status)
               if(status.ne.0)goto 900
            endif
            measure=xinorm_ev*xi_i_fks_ev*jac*sym
            if(.not.ieee_is_finite(measure).or.measure.le.0d0)then
               status=5
               goto 900
            endif
            call compute_prefactors_n1body(1d0,jac)
            if(local_component)then
c Individual diagram weights differ between real and projected Born
c contributions. Keep those physical partitions, even though the
c positive sampling factors q_sector and q_flow cancel in the ratio.
               call macnlops_include_diagram_weights(p,
     $              diagram_symmetry)
            elseif(n_recoil.gt.1)then
               call include_native_recoil_weights(recoil,groups)
            endif
            factors_save=factors
            mc_save=mc_factors
            pdf_save=pdf_factors
            call set_cms_stuff(0)
            call set_alphaS(p1_cnt(0,1,0))
            call sborn_native(p_born,born)
            call get_born_flow_weights(flows,total)
            if(total.le.0d0)then
               status=6
               goto 900
            endif
            if(local_component)then
               if(.not.is_leading(component_flow))then
                  status=33
                  goto 900
               endif
            endif
            call compute_shower_scale_nbody(p_born,-fksfather,
     $           mc_shower_scale_mass())
            cuts_born=passcuts_native_born(p1_cnt(0,1,0),rwgt,
     $           reference)
            call set_cms_stuff(-100)
            cuts_real=passcuts(p,rwgt)
            do flow=1,max_bcol
               if(.not.is_leading(flow))cycle
               if(local_component.and.flow.ne.component_flow)cycle
               born_flow_picked=flow
               if(.not.local_component)then
c Only Delta evaluation needs the H colour insertion and starting
c scales. A local unit replay never evaluates Delta, and ordinary
c local H uses only its scalar bogus P. Avoid unused H clustering.
                  call init_process_module_n1body_flow(flow,.true.)
                  call compute_shower_scale_n1body(p,i_fks,j_fks,
     $                 mc_shower_scale_mass())
                  emsca_H=0d0
               endif
               factors=factors_save
               mc_factors=mc_save
               pdf_factors=pdf_save
               call include_born_flow_weight(flows(flow)/total,1d0)
               MCcntcalled=0
               icontr=0
               if(local_component.and.damping_mode.eq.1.and.
     $            cuts_born)then
c A unit replay removes only Delta. Switching to ordinary matching
c here would incorrectly introduce its separate bogus Sudakov P.
c compute_MC_subt_term returns before reading Delta when both FKS
c partitions vanish. Do not invent an unused replay row in that case.
                  if(fks_Sij(p,i_fks,j_fks,xi_i_fks_ev,
     $                 y_ij_fks_ev).ne.0d0.or.
     $                 fks_Hij(p,i_fks,j_fks).ne.0d0)then
                     call begin_delta_context(owner,0.5d0)
                     delta_history=hist
                     delta_recoil=iconfig
                     call append_delta_probability(flow,1d0)
                     delta_replay=.true.
                  endif
               endif
               call compute_native_NLOPS_weights(p,pcms,
     $              cuts_born,cuts_real,probne)
               if(nonfinite_contribution)then
                  status=7
                  goto 900
               endif
               call include_PDF_and_alphas
               value=macnlops_flavour_sum(ids)
               h=h+value/measure

c The same history's undamped real contribution defines R. Taking
c the positive part of H here would incorrectly clip its summands.
               icontr=0
               macnlops_gate_active=.false.
               call set_cms_stuff(-100)
               call set_alphaS(p)
               if(cuts_real)call compute_real_emission(p,1d0)
               if(nonfinite_contribution)then
                  status=8
                  goto 900
               endif
               call include_PDF_and_alphas
               r=r+macnlops_flavour_sum(ids)/measure

c K keeps the starting-scale CDF and luminosity Jacobian. In a prepared
c MacNLOPS run its proposal support is profiled independently of the
c source construction. It never receives source G or bogus P factors.
               icontr=0
               if(cuts_born)then
                  call macnlops_raw_kernel(p,pcms,
     $                 ids(MC_HIST_PERM(:,hist)),kernel_status)
                  if(kernel_status.ne.0)then
                     status=20+kernel_status
                     goto 900
                  endif
               endif
               if(nonfinite_contribution)then
                  status=9
                  goto 900
               endif
               call include_PDF_and_alphas
               k=k+macnlops_flavour_sum(ids)/measure
            enddo
         enddo
      enddo
      d=r-h
      if(local_component.and.macnlops_sudakov_enabled.and.
     $     h.lt.0d0.and.k.gt.0d0)then
         macnlops_sudakov_pt=get_qMC(xi_i_fks_ev,y_ij_fks_ev)
         if(macnlops_sudakov_requested_pt.gt.0d0)then
            if(abs(macnlops_sudakov_pt-
     $           macnlops_sudakov_requested_pt).gt.
     $           1d-5*max(1d0,macnlops_sudakov_pt))then
               status=40
               goto 900
            endif
            macnlops_sudakov_pt=macnlops_sudakov_requested_pt
         endif
         call macnlops_delta_profile(p,p_born,ids,
     $        macnlops_sudakov_pt,macnlops_sudakov_source,
     $        macnlops_sudakov_n,macnlops_sudakov_q,
     $        macnlops_sudakov_w,macnlops_sudakov_fsr,
     $        macnlops_sudakov_cdf,kernel_status)
         if(kernel_status.ne.0)then
            status=41
            goto 900
         endif
         macnlops_sudakov_x=xbjrk_cnt(1:2,0)
         macnlops_sudakov_mass=sqrt(max(0d0,
     $        (p_born(0,1)+p_born(0,2))**2-
     $        sum((p_born(1:3,1)+p_born(1:3,2))**2)))
         macnlops_sudakov_ids=ids(1:2)
         if(j_fks.le.nincoming)then
            mother=ids(j_fks)
            sister=ids(i_fks)
            if(mother.eq.21)then
               if(sister.ne.21)
     $              macnlops_sudakov_ids(j_fks)=-sister
            elseif(sister.ne.21)then
               macnlops_sudakov_ids(j_fks)=21
            endif
         endif
         macnlops_sudakov_ready=.true.
      endif
      if(local_component)then
c The generated native Born is in the real-state statistical measure.
c For an actual, ancestry-labelled shower emission, restore I_R/I_B.
c The reduced source represents O=fac_i*fac_j/(I_R/I_B) equivalent
c Born emitters. Its H is their sum, while this request conditions on
c one actual emitter. Apply the same positive conversion to all three
c densities: neither the veto nor the H-residual ratio changes.
         call macnlops_component_normalization(owner,
     $        component_normalization,equivalent_emitters,status)
         if(status.ne.0)goto 900
         r=r*component_normalization
         d=d*component_normalization
         k=k*component_normalization
      endif
      if(.not.complete_delta_replay())then
         status=11
         goto 900
      endif
      if(.not.all(ieee_is_finite([r,d,k])))then
         status=10
         goto 900
      endif
      status=0
 900  continue
      if(status.ne.0)then
         write(*,'(a,6(1x,i8))')'MACNLOPS_DIAGNOSTIC_V1',
     $        status,owner,hist,sector,recoil,flow
         r=0d0
         d=0d0
         k=0d0
      endif
      icontr=0
      call mc_end_real_point()
      call set_native_history(0)
      native_mapping=.false.
      mc_H_only=.false.
      macnlops_gate_active=.false.
      macnlops_proposal_active=.false.
      end


      subroutine macnlops_validate_component_born(flow,ids,colours,
     $     lab,masses,projected_x,status)
c The supplied point is the actual pre-emission S Born, with inherited
c leg identities in native Born order. An inverse endpoint projection
c may not silently substitute a different Born geometry or colour flow.
      use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
      use fks_phase_space_data, only: p_born
      implicit none
      include 'nexternal.inc'
      include 'genps.inc'
      include 'born_nhel.inc'
      include 'run.inc'
      integer flow,ids(nexternal-1),colours(2,nexternal-1),status,
     $     idup(nexternal-1,maxproc),mothup(2,nexternal-1,maxproc),
     $     icolup(2,nexternal-1,max_bcol),i,j
      include 'born_leshouche.inc'
      double precision lab(0:3,nexternal-1),masses(nexternal-1),
     $     projected_x(2),
     $     model_mass,get_mass_from_id,mass2,native_dot,supplied_dot,
     $     scale2,tolerance,fraction
      double precision,parameter :: closure_rtol=1d-7
      logical flavour_match,macnlops_same_colour_topology
      external get_mass_from_id,macnlops_same_colour_topology
      status=36
      if(flow.lt.1.or.flow.gt.max_bcol)return
      flavour_match=.false.
      do i=1,iproc_born
         if(all(ids.eq.idup(:,i)))flavour_match=.true.
      enddo
      if(.not.flavour_match)return
      if(.not.macnlops_same_colour_topology(nexternal-1,colours,
     $     icolup(:,:,flow)))return
      status=37
      if(.not.all(ieee_is_finite(lab)).or.
     $   .not.all(ieee_is_finite(masses)).or.any(masses.lt.0d0).or.
     $   any(lab(0,:).le.0d0))return
      scale2=max(1d0,maxval(abs(lab))**2,maxval(abs(p_born))**2)
      tolerance=closure_rtol*scale2
      do i=1,nexternal-1
         model_mass=get_mass_from_id(ids(i))
         if(abs(masses(i)-model_mass).gt.closure_rtol*
     $      max(1d0,masses(i),model_mass))return
         mass2=lab(0,i)**2-sum(lab(1:3,i)**2)
         if(abs(mass2-model_mass**2).gt.tolerance)return
      enddo
      status=38
      do j=1,nexternal-1
         do i=1,j
            native_dot=p_born(0,i)*p_born(0,j)-
     $           sum(p_born(1:3,i)*p_born(1:3,j))
            supplied_dot=lab(0,i)*lab(0,j)-
     $           sum(lab(1:3,i)*lab(1:3,j))
            if(abs(native_dot-supplied_dot).gt.tolerance)then
               write(*,'(a,2(1x,i6),3(1x,es26.17e3))')
     $              'MACNLOPS_BORN_MISMATCH_V1',i,j,native_dot,
     $              supplied_dot,tolerance
               return
            endif
         enddo
      enddo
      do i=1,nincoming
         fraction=lab(0,i)/ebeam(i)
c The massive-FSR second solution has no soft counterevent. Its slot
c is invalid and xbjrk_cnt(:,0)=0, while the projected Born is valid.
c Validate against the immutable Born returned by the inverse map.
         if(.not.ieee_is_finite(projected_x(i)).or.
     $      projected_x(i).le.0d0)return
         if(abs(fraction-projected_x(i)).gt.closure_rtol*
     $      max(1d-12,abs(fraction),abs(projected_x(i))))then
            write(*,'(a,1x,i6,2(1x,es26.17e3))')
     $           'MACNLOPS_BORN_FRACTION_V1',i,projected_x(i),fraction
            return
         endif
      enddo
      status=0
      end


      logical function macnlops_same_colour_topology(nlegs,first,second)
c A bijective relabelling of colour tags preserves every ordered slot
c equality. Keep colour and anticolour slots distinct: an adjacency
c matrix alone loses the two orientations of a gluon colour flow.
      implicit none
      integer nlegs,first(2,nlegs),second(2,nlegs),i,j,a,b
      macnlops_same_colour_topology=.false.
      if(any(first.lt.0).or.any(second.lt.0))return
      if(any((first.eq.0).neqv.(second.eq.0)))return
      do j=1,nlegs
         do b=1,2
            do i=1,nlegs
               do a=1,2
                  if((first(a,i).eq.first(b,j)).neqv.
     $               (second(a,i).eq.second(b,j)))return
               enddo
            enddo
         enddo
      enddo
      macnlops_same_colour_topology=.true.
      end


      subroutine macnlops_include_diagram_weights(p,diagram_symmetry)
c Apply the same individual graph partition used by source generation,
c at each contribution's own projection and coupling scale. H-only
c evaluation suppresses the integration driver calls to this routine;
c temporarily lift that guard here, exactly once, before evaluating H.
      use fks_phase_space_data, only: p1_cnt
      use weight_lines, only: mc_H_only
      implicit none
      include 'nexternal.inc'
      double precision p(0:3,nexternal),diagram_symmetry,
     $     diagramsymmetryfactor,saved_symmetry
      common/dsymfactor/diagramsymmetryfactor
      logical saved_h_only
      saved_h_only=mc_H_only
      saved_symmetry=diagramsymmetryfactor
      mc_H_only=.false.
      diagramsymmetryfactor=diagram_symmetry
      call set_cms_stuff(-100)
      call set_alphaS(p)
      call include_multichannel_enhance(4)
      call set_cms_stuff(0)
      call set_alphaS(p1_cnt(0,1,0))
      call include_multichannel_enhance(3)
      call set_cms_stuff(-100)
      call set_alphaS(p)
      call include_multichannel_enhance(2)
      diagramsymmetryfactor=saved_symmetry
      mc_H_only=saved_h_only
      end


      subroutine macnlops_component_normalization(owner,normalization,
     $     emitters,status)
c Statistical factors come from complete exported external states,
c including their tags and helicities, rather than a PDG-only factorial.
      use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
      implicit none
      include 'nFKSconfigs.inc'
      include 'fks_symmetry.inc'
      integer owner,emitters,status
      double precision normalization,orbit,tolerance
      status=39
      normalization=0d0
      emitters=0
      if(owner.lt.1.or.owner.gt.fks_configs)return
      if(FKS_FAC_I_D(owner).le.0.or.FKS_FAC_J_D(owner).le.0.or.
     $   .not.ieee_is_finite(FKS_IDEN_BORN_D(owner)).or.
     $   .not.ieee_is_finite(FKS_IDEN_REAL_D(owner)).or.
     $   FKS_IDEN_BORN_D(owner).le.0d0.or.
     $   FKS_IDEN_REAL_D(owner).le.0d0)return
      normalization=FKS_IDEN_REAL_D(owner)/FKS_IDEN_BORN_D(owner)
      if(.not.ieee_is_finite(normalization).or.
     $   normalization.le.0d0)return
      orbit=dble(FKS_FAC_I_D(owner))*dble(FKS_FAC_J_D(owner))/
     $     normalization
      if(.not.ieee_is_finite(orbit).or.orbit.lt.1d0.or.
     $   orbit.gt.dble(huge(emitters)/2))return
      emitters=nint(orbit)
      tolerance=64d0*epsilon(1d0)*max(1d0,orbit)
      if(abs(orbit-dble(emitters)).gt.tolerance)return
      status=0
      end


      double precision function macnlops_flavour_sum(ids)
      use weight_lines
      implicit none
      include 'nexternal.inc'
      integer ids(nexternal),i,j
      macnlops_flavour_sum=0d0
      do i=1,icontr
         do j=1,niproc(i)
            if(all(parton_pdg(:,j,i).eq.ids))
     $           macnlops_flavour_sum=macnlops_flavour_sum+
     $           parton_iproc(j,i)
         enddo
      enddo
      end


      subroutine macnlops_raw_kernel(p,pcms,ids,status)
      use mc_counterterms, only: compute_MCsubtraction_kl,get_qMC
      use mc_pythia8_kernel, only: py8_isr_ordinary_weight,
     $     py8_fsr_ordinary_weight
      use fks_phase_space_data, only: p_born,xi_i_fks_ev,
     $     y_ij_fks_ev,xbjrk_cnt
      use extra_weights
      use weight_lines, only: macnlops_gate_active,
     $     macnlops_current_z,macnlops_proposal_active,
     $     macnlops_kernel_alphas,macnlops_kernel_q2fact
      implicit none
      include 'nexternal.inc'
      include 'orders.inc'
      include 'coupl.inc'
      include 'run.inc'
      double precision p(0:3,nexternal),pcms(0:3,nexternal),
     $     amps(amp_split_size,2),z(2),lum,weight,sevmc,fks_Hij,
     $     xi,y,mc_factors(8),proposal_weight,shat_born,
     $     total_born(0:3),pt2,mcharm,mbottom,msister,
     $     get_mass_from_id
      double precision beam_xmax(2),ptmin_isr,ptmin_fsr
      common/macnlops_shower_config/beam_xmax,ptmin_isr,ptmin_fsr
      common/factor_n1body_NLOPS/mc_factors
      double precision pi
      parameter (pi=3.1415926535897932385d0)
      integer i,j,nconnect,orders(nsplitorders),i_fks,j_fks,
     $     get_orders_tag,ids(nexternal),status,daughter,mother,sister
      common/fks_indices/i_fks,j_fks
      external fks_Hij,get_mass_from_id
      status=0
      macnlops_gate_active=.false.
      macnlops_proposal_active=macnlops
      xi=xi_i_fks_ev
      y=y_ij_fks_ev
      call set_cms_stuff(-100)
      call set_alphaS(p)
c Every type-13 record of this real point takes its couplings and PDFs
c at these scales. The S veto replaces one power by the shower's
c emission alpha_s and the emission's PDF ratio by the shower's.
      macnlops_kernel_alphas=g**2/(4d0*pi)
      macnlops_kernel_q2fact=q2fact
      sevmc=fks_Hij(p,i_fks,j_fks)
      call compute_MCsubtraction_kl(i_fks,j_fks,xi,y,p,pcms,
     $     p_born,.false.,z,nconnect,amps,.true.)
      pt2=get_qMC(xi,y)**2
      if(.not.macnlops)then
         if(j_fks.le.nincoming)then
            if(pt2.lt.ptmin_isr**2)return
         else
            if(pt2.lt.ptmin_fsr**2)return
         endif
      endif
      do j=1,nconnect
         if(all(amps(:,j).eq.0d0))cycle
         macnlops_current_z=z(j)
         proposal_weight=1d0
         if(j_fks.le.nincoming.and..not.macnlops)then
c Flavour grouping may put charm or bottom in a light representative's
c luminosity table. Use the requested flavour for shower support.
            mother=ids(j_fks)
            sister=ids(i_fks)
            if(abs(mother).ge.1.and.abs(mother).le.6.and.
     $         abs(mother).gt.macnlops_nquark_in)cycle
            if(mother.eq.21)then
               daughter=21
               if(sister.ne.21)daughter=-sister
            else
               daughter=mother
               if(sister.ne.21)daughter=21
            endif
            mcharm=get_mass_from_id(4)
            mbottom=get_mass_from_id(5)
            msister=get_mass_from_id(sister)
            total_born=sum(p_born(:,1:nincoming),dim=2)
            shat_born=total_born(0)**2-sum(total_born(1:3)**2)
            call py8_isr_ordinary_weight(daughter,mother,sister,
     $           z(j),pt2,shat_born,xbjrk_cnt(j_fks,0),
     $           beam_xmax(j_fks),
     $           mcharm,mbottom,msister,.false.,proposal_weight,status)
            if(status.ne.0)return
         elseif(j_fks.gt.nincoming.and..not.macnlops)then
            mother=ids(j_fks)
            sister=ids(i_fks)
            daughter=mother
            if(sister.ne.21)then
               if(mother.eq.21)then
                  daughter=sister
               else
                  daughter=21
               endif
            endif
            call py8_fsr_ordinary_weight(daughter,mother,sister,
     $           z(j),pt2,get_mass_from_id(daughter),
     $           get_mass_from_id(mother),get_mass_from_id(sister),
     $           ptmin_fsr,macnlops_ngluon_to_quark,
     $           proposal_weight,status)
            if(status.ne.0)return
         endif
         call get_mc_lum(j_fks,z(j),xi,lum,xbjrk_cnt(1,0))
         do i=1,amp_split_size
            if(amps(i,j).eq.0d0)cycle
            call amp_split_pos_to_orders(i,orders)
            QCD_power=orders(qcd_pos)
            wgtcpower=0d0
            if(cpower_pos.gt.0)
     $           wgtcpower=dble(orders(max(1,cpower_pos)))
            orders_tag=get_orders_tag(orders)
            amp_pos=i
            weight=sevmc*mc_factors(8)*lum*amps(i,j)*
     $           xi**2*(1d0-y)/g**QCD_power*proposal_weight
c A positive type-13 record gives this proposal exactly the same
c PDF, coupling and native flavour treatment as the source subtraction.
            call add_wgt(13,orders,weight,0d0,0d0)
         enddo
      enddo
      macnlops_gate_active=.false.
      macnlops_proposal_active=.false.
      end
