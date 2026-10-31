      program macnlops_evaluator
c Evaluate the source hard remainder and the scalar shower proposal at
c a supplied, flavour-resolved real point. Sampling probabilities never
c belong in these densities. V3 retains the selected physical diagram
c partition and Born-flow fraction instead of summing components.
c The densities are in driver_macnlops.f; macnlops_capi.f exposes the
c same routines to the in-process first-emission program.
      use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
      use FKSParams, only: delta_common_scale
      use weight_lines, only: reset_delta_contexts,begin_delta_context,
     $     delta_contexts,delta_context_active,delta_history,
     $     delta_recoil,delta_replay,append_delta_probability,
     $     macnlops_kernel_alphas
      implicit none
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      include 'run.inc'
      include 'mc_histories.inc'
      integer ids(nexternal),owner,nlegs,request,status,ios,i,
     $     rows,seed,key(3),row,component_config,component_flow,
     $     damping_mode,component_graph,born_ids(nexternal-1),
     $     born_colours(2,nexternal-1),born_legs
      double precision lab(0:3,nexternal),real_density,
     $     subtraction_density,shower_density
      character*80 marker
      character*256 request_line
      character*64 registry
      double precision insertion,probability,ran2
      double precision diagram_symmetry,born_lab(0:3,nexternal-1),
     $     born_masses(nexternal-1)
      logical component_request,has_component_born
      external ran2
      double precision beam_xmax(2),ptmin_isr,ptmin_fsr
      common/macnlops_shower_config/beam_xmax,ptmin_isr,ptmin_fsr

      call macnlops_evaluator_setup()
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
     $        marker.ne.'MACNLOPS_POINT_V2'.and.
     $        marker.ne.'MACNLOPS_POINT_V3').or.
     $        nlegs.ne.nexternal) stop 'Invalid MacNLOPS point header'
         component_request=marker.eq.'MACNLOPS_POINT_V3'
         if(mcatnlo_delta.and.marker.ne.'MACNLOPS_POINT_V2'.and.
     $      .not.component_request)
     $      stop 'Delta input requires a conditional V2 context'
         do i=1,nexternal
            read(*,*,iostat=ios) ids(i),lab(:,i)
            if(ios.ne.0)stop 'Invalid MacNLOPS momentum row'
         enddo
         call reset_delta_contexts()
         has_component_born=.false.
         born_ids=0
         born_colours=0
         born_lab=0d0
         born_masses=0d0
         if(component_request)then
            read(*,'(a)',iostat=ios)request_line
            if(ios.ne.0)stop 'Missing local component selector'
            component_config=0
            component_graph=0
            component_flow=0
            damping_mode=-1
            diagram_symmetry=0d0
            read(request_line,*,iostat=ios)marker,component_config,
     $           component_graph,component_flow,damping_mode,
     $           diagram_symmetry
            if(ios.ne.0.or.marker.ne.'COMPONENT_LOCAL_V1'.or.
     $         component_config.lt.1.or.component_graph.lt.1.or.
     $         component_flow.lt.1.or.
     $         damping_mode.lt.0.or.damping_mode.gt.3.or.
     $         damping_mode.eq.2.or.
     $         .not.ieee_is_finite(diagram_symmetry).or.
     $         diagram_symmetry.le.0d0)
     $         stop 'Invalid local component selector'
            if((mcatnlo_delta.neqv.(damping_mode.gt.0)).or.
     $         (delta_common_scale.neqv.(damping_mode.eq.3)))
     $         stop 'Local damping mode differs from source matching'
         elseif(mcatnlo_delta)then
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
         read(*,'(a)',iostat=ios)request_line
         if(ios.ne.0)stop 'Missing MacNLOPS point terminator'
         read(request_line,*,iostat=ios)marker
         if(ios.ne.0)stop 'Invalid MacNLOPS point terminator'
         if(component_request.and.marker.eq.'COMPONENT_BORN_V1')then
            born_legs=0
            read(request_line,*,iostat=ios)marker,born_legs
            if(ios.ne.0.or.born_legs.ne.nexternal-1)
     $         stop 'Invalid local Born particle count'
            do i=1,nexternal-1
               read(*,*,iostat=ios)born_ids(i),born_colours(:,i),
     $              born_lab(:,i),born_masses(i)
               if(ios.ne.0.or.any(born_colours(:,i).lt.0).or.
     $            .not.all(ieee_is_finite(born_lab(:,i))).or.
     $            .not.ieee_is_finite(born_masses(i)).or.
     $            born_masses(i).lt.0d0)
     $            stop 'Invalid local Born particle row'
            enddo
            has_component_born=.true.
            read(*,*,iostat=ios)marker
         endif
         if(ios.ne.0.or.marker.ne.'END_POINT')
     $        stop 'Missing MacNLOPS point terminator'
         status=1
         real_density=0d0
         subtraction_density=0d0
         shower_density=0d0
         macnlops_kernel_alphas=0d0
         if(owner.ge.1.and.owner.le.fks_configs.and.
     $        all(ieee_is_finite(lab))) then
            if(component_request)then
               call macnlops_component_density(owner,ids,lab,
     $              component_config,component_graph,component_flow,
     $              damping_mode,
     $              diagram_symmetry,has_component_born,born_ids,
     $              born_colours,born_lab,born_masses,real_density,
     $              subtraction_density,shower_density,status)
            else
               call macnlops_density(owner,ids,lab,real_density,
     $              subtraction_density,shower_density,status)
            endif
         endif
c Precedes its density record; zero when no kernel was evaluated.
         write(*,'(a,1x,i12,1x,es26.17e3)')
     $        'MACNLOPS_KERNEL_COUPLING_V1',request,macnlops_kernel_alphas
         write(*,'(a,2(1x,i12),3(1x,es26.17e3))')
     $        'MACNLOPS_DENSITY_V1',request,status,real_density,
     $        subtraction_density,shower_density
         flush(6)
         call reset_delta_contexts()
      enddo
      end