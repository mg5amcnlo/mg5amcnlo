      program macnlops_evaluator
c Evaluate the source hard remainder and the scalar shower proposal at
c a supplied, flavour-resolved real point. No integration-channel or
c random colour-flow weight belongs in these differential densities.
      use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
      use FKSParams
      use mint_module
      use process_module
      use scale_module
      use extra_weights
      use weight_lines, only: reset_delta_contexts,begin_delta_context,
     $     delta_contexts,delta_context_active,delta_history,
     $     delta_recoil,delta_replay,append_delta_probability
      implicit none
      include 'nexternal.inc'
      include 'genps.inc'
      include 'nFKSconfigs.inc'
      include 'born_nhel.inc'
      include 'run.inc'
      include 'mc_histories.inc'
      integer ids(nexternal),owner,nlegs,request,status,ios,i,
     $     rows,seed,key(3),row
      integer nFKSprocess
      common/c_nFKSprocess/nFKSprocess
      double precision lab(0:3,nexternal),real_density,
     $     subtraction_density,shower_density
      character*80 marker
      character*256 request_line
      character*64 registry
      double precision insertion,probability,rng_init,ran2
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
c These are the source driver's input_app.txt defaults, written by
c aMCatNLOCmd.write_input_file. Ordinary-source audits retain G and P;
c The default source retains ordinary G/P subtraction. The explicit
c macnlops_resolved_source option selects the legacy convex source.
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
c Exact colour summation is needed even if the integration sampled it.
      MCSubtractionAtFixedFlow=.false.
c Standalone density audits can request the uncut scalar kernel. The
c production runner supplies actual shower beam bounds and cutoffs.
      beam_xmax=1d0
      ptmin_isr=0d0
      ptmin_fsr=0d0
      write(*,'(a)') 'MACNLOPS_READY_V1'
      flush(6)
      do
         read(*,'(a)',iostat=ios) request_line
         if(ios.lt.0)exit
         if(ios.ne.0)stop 'Invalid MacNLOPS request line'
         read(request_line,*,iostat=ios) marker
         if(ios.ne.0)stop 'Invalid MacNLOPS request marker'
         if(marker.eq.'MACNLOPS_CONFIG_V1')then
            read(request_line,*,iostat=ios) marker,beam_xmax,
     $           ptmin_isr,ptmin_fsr
            if(ios.ne.0)stop 'Invalid MacNLOPS shower configuration'
            if(.not.all(ieee_is_finite([beam_xmax,ptmin_isr,
     $           ptmin_fsr])).or.any(beam_xmax.le.0d0).or.
     $           any(beam_xmax.gt.1d0).or.ptmin_isr.lt.0d0.or.
     $           ptmin_fsr.lt.0d0)
     $           stop 'Invalid MacNLOPS beam bounds or shower cutoffs'
            write(*,'(a)') 'MACNLOPS_CONFIGURED_V1'
            flush(6)
            cycle
         endif
         read(request_line,*,iostat=ios) marker,request,owner,nlegs
         if(ios.ne.0.or.(marker.ne.'MACNLOPS_POINT_V1'.and.
     $        marker.ne.'MACNLOPS_POINT_V2').or.
     $        nlegs.ne.nexternal) stop 'Invalid MacNLOPS point header'
         if(mcatnlo_delta.and.marker.ne.'MACNLOPS_POINT_V2')
     $      stop 'Delta input requires a conditional V2 context'
         do i=1,nexternal
            read(*,*,iostat=ios) ids(i),lab(:,i)
            if(ios.ne.0)stop 'Invalid MacNLOPS momentum row'
         enddo
         call reset_delta_contexts()
         if(mcatnlo_delta)then
            read(*,'(a)',iostat=ios)request_line
            if(ios.ne.0)stop 'Missing Delta request context'
            read(request_line,*,iostat=ios)marker
            if(ios.ne.0)stop 'Invalid Delta request context'
            if(marker.eq.'DELTA_REPLAY')then
               read(request_line,*,iostat=ios)marker,rows,insertion,
     $              registry
               if(ios.ne.0.or.rows.lt.0.or.rows.gt.8192.or.
     $            registry.ne.MC_HIST_REGISTRY_SHA256)
     $           stop 'Invalid Delta replay registry or size'
               call begin_delta_context(owner,insertion)
               do row=1,rows
                  read(*,*,iostat=ios)key,probability
                  if(ios.ne.0)stop 'Invalid Delta replay row'
                  delta_history=key(1)
                  delta_recoil=key(2)
                  call append_delta_probability(key(3),probability)
               enddo
               delta_replay=.true.
            elseif(marker.eq.'DELTA_SAMPLE')then
               read(request_line,*,iostat=ios)marker,seed
               if(ios.ne.0.or.seed.lt.1.or.seed.gt.899999999)
     $           stop 'Invalid Delta auxiliary seed'
               call rmarin(mod(seed,31328),mod(seed/31328,30081))
               insertion=ran2()
               call begin_delta_context(owner,insertion)
            else
               stop 'Unknown Delta context mode'
            endif
            call set_macnlops_colour_insertion(insertion)
         endif
         read(*,*,iostat=ios) marker
         if(ios.ne.0.or.marker.ne.'END_POINT')
     $        stop 'Missing MacNLOPS point terminator'
         status=1
         real_density=0d0
         subtraction_density=0d0
         shower_density=0d0
         if(owner.ge.1.and.owner.le.fks_configs.and.
     $        all(ieee_is_finite(lab))) then
            call macnlops_density(owner,ids,lab,real_density,
     $           subtraction_density,shower_density,status)
         endif
         write(*,'(a,2(1x,i12),3(1x,es26.17e3))')
     $        'MACNLOPS_DENSITY_V1',request,status,real_density,
     $        subtraction_density,shower_density
         flush(6)
         call reset_delta_contexts()
      enddo
      end


      subroutine macnlops_density(owner,ids,lab,r,d,k,status)
      use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
      use fks_phase_space, only: generate_native_momenta
      use fks_phase_space_data, only: p_born,p1_cnt,xinorm_ev,
     $     xi_i_fks_ev,y_ij_fks_ev
      use fks_phase_space_helpers, only: apply_momentum_permutation
      use mc_native_context, only: set_native_history,native_mapping,
     $     mc_begin_real_point,mc_end_real_point
      use mc_counterterms, only: fksfather,mc_shower_scale_mass
      use process_module
      use scale_module
      use weight_lines
      use mint_module, only: iconfig
      implicit none
      include 'nexternal.inc'
      include 'genps.inc'
      include 'nFKSconfigs.inc'
      include 'born_nhel.inc'
      include 'mc_histories.inc'
      integer owner,ids(nexternal),status,hist,sector,flow,
     $     n_recoil,recoil,configs(lmaxconfigs),
     $     groups(lmaxconfigs),i_fks,j_fks,MCcntcalled,kernel_status,
     $     leg
      common/fks_indices/i_fks,j_fks
      common/c_MCcntcalled/MCcntcalled
      double precision lab(0:3,nexternal),r,d,k,h,
     $     flipped(0:3,nexternal),p(0:3,nexternal),
     $     plab(0:3,nexternal),pcms(0:3,nexternal),jac,measure,
     $     born,flows(max_bcol),total,probne,rwgt,reference,
     $     value,macnlops_flavour_sum,model_mass,mass2,tolerance,
     $     get_mass_from_id,hard_scale2
      external macnlops_flavour_sum,get_mass_from_id
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
     $     passcuts_native_born,is_leading(max_bcol)
      integer nleading
      common/c_leading_cflows/is_leading,nleading
      integer this_config
      common/to_mconfigs/this_config
      external passcuts,passcuts_native_born

      status=1
      r=0d0
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
      if(.not.MC_HIST_COMPLETE(owner))then
         status=2
         goto 900
      endif
      native_mapping=.true.
      mc_H_only=.true.
      mc_S_only=.false.
      nonfinite_contribution=.false.
      call mc_begin_real_point(lab)
      do hist=MC_HIST_FIRST(owner),MC_HIST_LAST(owner)
         delta_history=hist
         sector=MC_HIST_NATIVE(hist)
         call update_fks_dir(sector)
         call set_native_history(hist)
         call init_process_module_nbody_wrapper()
         call update_coltype_and_charge(sector,i_fks,j_fks)
         call apply_momentum_permutation(MC_HIST_PERM(:,hist),
     $        lab,flipped,valid)
         if(.not.valid)then
            status=3
            goto 900
         endif
         call native_recoil_groups(n_recoil,configs,groups)
         do recoil=1,n_recoil
            iconfig=configs(recoil)
            delta_recoil=iconfig
            call select_native_recoil(iconfig)
            call generate_native_momenta(flipped,p,plab,pcms,
     $           jac,valid)
            if(.not.valid)then
               status=4
               goto 900
            endif
            this_config=iconfig
            measure=xinorm_ev*xi_i_fks_ev*jac*sym
            if(.not.ieee_is_finite(measure).or.measure.le.0d0)then
               status=5
               goto 900
            endif
            call compute_prefactors_n1body(1d0,jac)
            if(n_recoil.gt.1)call include_native_recoil_weights(
     $           recoil,groups)
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
            call compute_shower_scale_nbody(p_born,-fksfather,
     $           mc_shower_scale_mass())
            cuts_born=passcuts_native_born(p1_cnt(0,1,0),rwgt,
     $           reference)
            call set_cms_stuff(-100)
            cuts_real=passcuts(p,rwgt)
            do flow=1,max_bcol
               if(.not.is_leading(flow))cycle
               born_flow_picked=flow
               call init_process_module_n1body_flow(flow,.true.)
               call compute_shower_scale_n1body(p,i_fks,j_fks,
     $              mc_shower_scale_mass())
               emsca_H=0d0
               factors=factors_save
               mc_factors=mc_save
               pdf_factors=pdf_save
               call include_born_flow_weight(flows(flow)/total,1d0)
               MCcntcalled=0
               icontr=0
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
     $     macnlops_current_z,macnlops_proposal_active
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
