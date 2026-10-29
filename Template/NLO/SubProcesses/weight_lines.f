*     -*-fortran-*-

      module weight_lines
         use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
         implicit none
         integer max_contr,max_wgt,max_iproc,icontr,iwgt,icontr_picked
     $        ,iproc_picked
         integer :: born_spread_fold_capacity=0
! Only complete native H weights are retained during repartitioning.
         logical :: mc_H_only=.false.
         logical :: mc_S_only=.false.
! The MacNLOPS matching gate is applied after the actual luminosity
! flavours are known. Preserve the generating history rather than the
! representative flavour used by the matrix-element table.
         logical :: macnlops_gate_active=.false.
! Proposal records use the certified physical support even when the
! source keeps ordinary G/P subtraction. Never damp this K by bogus P.
         logical :: macnlops_proposal_active=.false.
         double precision :: macnlops_current_z=0d0
! alpha_s of the last MacNLOPS real point, reported with its density,
! and the squared factorisation scales of that point's kernel PDFs.
         double precision :: macnlops_kernel_alphas=0d0
         double precision :: macnlops_kernel_q2fact(2)=0d0
! Per contribution: qMC**2, shat_Born, xBorn(1:2), connection z,
! and a flag: 1=legacy resolved-source gate, 2=physical proposal support.
         double precision, allocatable :: macnlops_meta(:,:)
         integer :: rejected_born_flow_points=0
         integer :: rejected_mc_kinematics_points=0
         integer :: rejected_nonfinite_points=0
         logical :: nonfinite_contribution=.false.
         integer :: nonfinite_type=0,nonfinite_sector=0,
     $        nonfinite_history=0
         logical, allocatable :: H_event(:)
         integer, allocatable :: itype(:),nFKS(:),QCDpower(:),pdg(:,:)
     $        ,pdg_uborn(:,:),parton_pdg_uborn(:,:,:),parton_pdg(:,:,:)
     $        ,plot_id(:),niproc(:),ipr(:)
     $        ,ifold_cnt(:)
     $        ,orderstag(:),amppos(:),need_match(:,:)
! Keep the event/shower owner separate from the native PDF/ME history.
         integer, allocatable :: event_nFKS(:),native_ids(:,:)
! A Delta H sum is conditional on one joint draw of all its native
! starting scales. Keep that draw with the whole group, not its last row.
         type delta_weight_context
            integer :: owner=0,count=0
            double precision :: insertion=0d0
            integer, allocatable :: keys(:,:)
            double precision, allocatable :: probability(:)
            logical, allocatable :: used(:)
         end type delta_weight_context
         type(delta_weight_context), allocatable :: delta_contexts(:)
         integer, allocatable :: delta_line_context(:)
         integer :: delta_context_count=0,delta_context_active=0
         integer :: delta_history=0,delta_recoil=0
         logical :: delta_replay=.false.
! A sampled native source component is a conditional signed measure.
! Keep its identity through contribution growth/copying and never merge
! two independently sampled components merely because their momenta agree.
         type sampled_component_context
            integer :: sector=0,config=0,graph=0,flow=0
            double precision :: flow_probability=0d0,insertion=0d0,
     $           source_factor=1d0,diagram_symmetry=1d0
         end type sampled_component_context
         type(sampled_component_context), allocatable ::
     $        component_contexts(:)
         integer, allocatable :: component_line_context(:)
         integer :: component_context_count=0,component_context_active=0
! The integration job has one Born diagram; its physical graph identity
! is retained separately from the ordinal native recoil configuration.
         integer :: source_born_config=0,source_born_graph=0
         double precision :: source_flow_probability=0d0
         double precision :: source_diagram_symmetry=1d0
         double precision, allocatable :: component_source_born(:,:)
         double precision :: component_source_xbj(2)=0d0
! Packed contribution groups retain insertion order using linear storage.
         integer, allocatable :: group_size(:),group_start(:)
     $        ,group_members(:),group_owner(:),group_cursor(:)
         double precision, allocatable :: momenta(:,:,:),momenta_m(:,:,:
     $        ,:),wgt(:,:),wgt_ME_tree(:,:),bjx(:,:),scales2(:,:)
     $        ,g_strong(:),wgts(:,:),parton_iproc(:,:),y_bst(:)
     $        ,cpower(:),plot_wgts(:,:),shower_scale(:),unwgt(:,:)
     $        ,unwgt_B(:,:),unwgt_noB(:,:)
     $        ,bias_wgt(:)
         save
      contains
         subroutine reset_delta_contexts()
         if(allocated(delta_contexts))deallocate(delta_contexts)
         delta_context_count=0
         delta_context_active=0
         delta_history=0
         delta_recoil=0
         delta_replay=.false.
         call reset_component_contexts()
         end subroutine reset_delta_contexts

         subroutine reset_component_contexts()
         if(allocated(component_contexts))deallocate(component_contexts)
         component_context_count=0
         component_context_active=0
         if(allocated(component_source_born))
     $      deallocate(component_source_born)
         source_flow_probability=0d0
         end subroutine reset_component_contexts

         subroutine check_component_source_born(p,xbj)
! S records are combined across sectors/folds. A sampled-source packet
! may describe them only when their pre-shuffle Born state is common.
         double precision, intent(in) :: p(0:,:)
         double precision, intent(in) :: xbj(2)
         if(size(p,1).ne.4.or..not.all(ieee_is_finite(p)))
     $      error stop 'Invalid sampled S source Born point'
         if(.not.all(ieee_is_finite(xbj)).or.any(xbj.le.0d0))
     $      error stop 'Invalid sampled S source Bjorken fractions'
         if(.not.allocated(component_source_born))then
            allocate(component_source_born(0:3,size(p,2)))
            component_source_born=p
            component_source_xbj=xbj
         elseif(any(shape(p).ne.shape(component_source_born)))then
            error stop 'Sampled S source Born shape changed'
         elseif(maxval(abs(p-component_source_born)).gt.
     $          1d-10*max(1d0,maxval(abs(component_source_born))))then
            error stop 'Sampled S source combines different Born points'
         elseif(maxval(abs(xbj-component_source_xbj)).gt.1d-10)then
            error stop 'Sampled S source combines different lab boosts'
         endif
         end subroutine check_component_source_born

         subroutine begin_component_context(sector,config,graph,flow,
     $        flow_probability,insertion,diagram_symmetry)
         integer, intent(in) :: sector,config,graph,flow
         double precision, intent(in) :: flow_probability,insertion,
     $        diagram_symmetry
         type(sampled_component_context), allocatable :: grown(:)
         integer capacity
         if(min(sector,config,graph,flow).lt.1.or.
     $      .not.(flow_probability.gt.0d0.and.flow_probability.le.1d0).or.
     $      .not.(insertion.ge.0d0.and.insertion.le.1d0))
     $      error stop 'Invalid sampled source component'
         if(.not.ieee_is_finite(diagram_symmetry).or.
     $      .not.(diagram_symmetry.gt.0d0))
     $      error stop 'Invalid sampled diagram symmetry'
         if(component_context_active.ne.0)
     $      error stop 'Nested sampled source components'
         if(.not.allocated(component_contexts))
     $      allocate(component_contexts(8))
         if(component_context_count.eq.size(component_contexts))then
            capacity=2*size(component_contexts)
            allocate(grown(capacity))
            grown(1:component_context_count)=component_contexts
            call move_alloc(grown,component_contexts)
         endif
         component_context_count=component_context_count+1
         component_context_active=component_context_count
         associate(packet=>component_contexts(component_context_active))
         packet%sector=sector
         packet%config=config
         packet%graph=graph
         packet%flow=flow
         packet%flow_probability=flow_probability
         packet%insertion=insertion
         packet%source_factor=1d0
         packet%diagram_symmetry=diagram_symmetry
         end associate
         end subroutine begin_component_context

         subroutine end_component_context(source_factor)
         double precision, intent(in) :: source_factor
         if(component_context_active.lt.1.or.
     $      .not.(source_factor.ge.0d0.and.source_factor.le.1d0))
     $      error stop 'Invalid sampled source multiplier'
         component_contexts(component_context_active)%source_factor=
     $      source_factor
         component_context_active=0
         end subroutine end_component_context

         logical function same_source_component(left,right)
         integer, intent(in) :: left,right
         same_source_component=component_line_context(left).eq.
     $      component_line_context(right)
         end function same_source_component

         subroutine begin_delta_context(owner,insertion)
         integer, intent(in) :: owner
         double precision, intent(in) :: insertion
         type(delta_weight_context), allocatable :: grown(:)
         integer capacity
         if(owner.le.0.or..not.(insertion.ge.0d0.and.
     $      insertion.le.1d0))error stop 'Invalid Delta context'
         if(.not.allocated(delta_contexts))allocate(delta_contexts(8))
         if(delta_context_count.eq.size(delta_contexts))then
            capacity=2*size(delta_contexts)
            allocate(grown(capacity))
            grown(1:delta_context_count)=delta_contexts
            call move_alloc(grown,delta_contexts)
         endif
         delta_context_count=delta_context_count+1
         delta_context_active=delta_context_count
         delta_contexts(delta_context_active)%owner=owner
         delta_contexts(delta_context_active)%insertion=insertion
         delta_contexts(delta_context_active)%count=0
         allocate(delta_contexts(delta_context_active)%keys(3,16),
     $      delta_contexts(delta_context_active)%probability(16),
     $      delta_contexts(delta_context_active)%used(16))
         delta_replay=.false.
         end subroutine begin_delta_context

         subroutine append_delta_probability(flow,probability)
         integer, intent(in) :: flow
         double precision, intent(in) :: probability
         integer n,capacity,key(3),i
         integer, allocatable :: keys(:,:)
         double precision, allocatable :: values(:)
         logical, allocatable :: used(:)
         if(delta_context_active.eq.0.or.delta_replay)return
         key=[delta_history,delta_recoil,flow]
         if(any(key.le.0).or..not.(probability.ge.0d0.and.
     $      probability.le.1d0))error stop 'Invalid Delta probability'
         associate(packet=>delta_contexts(delta_context_active))
         n=packet%count
         do i=1,n
            if(all(packet%keys(:,i).eq.key))
     $         error stop 'Duplicate Delta probability'
         enddo
         if(.not.allocated(packet%keys))then
            allocate(packet%keys(3,16),packet%probability(16),
     $           packet%used(16))
         elseif(n.eq.size(packet%probability))then
            capacity=2*size(packet%probability)
            allocate(keys(3,capacity),values(capacity),used(capacity))
            keys(:,1:n)=packet%keys
            values(1:n)=packet%probability
            used(1:n)=packet%used
            call move_alloc(keys,packet%keys)
            call move_alloc(values,packet%probability)
            call move_alloc(used,packet%used)
         endif
         packet%count=n+1
         packet%keys(:,n+1)=key
         packet%probability(n+1)=probability
         packet%used(n+1)=.false.
         end associate
         end subroutine append_delta_probability

         subroutine replay_delta_probability(flow,probability,replayed)
         integer, intent(in) :: flow
         double precision, intent(out) :: probability
         logical, intent(out) :: replayed
         integer i,key(3)
         replayed=.false.
         probability=1d0
         if(.not.delta_replay)return
         if(delta_context_active.eq.0)
     $      error stop 'Missing Delta replay packet'
         key=[delta_history,delta_recoil,flow]
         associate(packet=>delta_contexts(delta_context_active))
         do i=1,packet%count
            if(.not.all(packet%keys(:,i).eq.key))cycle
            if(packet%used(i))error stop 'Repeated Delta replay key'
            probability=packet%probability(i)
            packet%used(i)=.true.
            replayed=.true.
            return
         enddo
         end associate
         error stop 'Missing Delta replay key'
         end subroutine replay_delta_probability

         logical function complete_delta_replay()
         complete_delta_replay=.true.
         if(.not.delta_replay)return
         associate(packet=>delta_contexts(delta_context_active))
         if(packet%count.gt.0)complete_delta_replay=
     $      all(packet%used(1:packet%count))
         end associate
         end function complete_delta_replay

         integer function event_owner(ict)
         integer, intent(in) :: ict
         event_owner=nFKS(ict)
         if (event_nFKS(ict).gt.0) event_owner=event_nFKS(ict)
         end function event_owner

         integer function group_flavour_row(igroup,ipro,ict)
! Keep native flavour tables intact. An H group uses its owner's row
! order, while its member may enumerate the same physical flavours in
! a different order. Match repeated rows by occurrence for a bijection.
         integer, intent(in) :: igroup,ipro,ict
         integer i,occurrence,matched
         group_flavour_row=0
         if(ipro.lt.1.or.ipro.gt.niproc(igroup))return
         occurrence=0
         do i=1,ipro
            if(all(parton_pdg(:,i,igroup).eq.
     $           parton_pdg(:,ipro,igroup)))occurrence=occurrence+1
         enddo
         matched=0
         do i=1,niproc(ict)
            if(.not.all(parton_pdg(:,i,ict).eq.
     $           parton_pdg(:,ipro,igroup)))cycle
            matched=matched+1
            if(matched.ne.occurrence)cycle
            group_flavour_row=i
            return
         enddo
         end function group_flavour_row

         logical function group_flavour_complete(igroup,ipro,compatible)
! A physical flavour can occur in partially overlapping native tables.
! Its signed sum is global only if every compatible H record containing
! that flavour belongs to the selected group, irrespective of row order.
         integer, intent(in) :: igroup,ipro
         logical, intent(in) :: compatible(:)
         integer i,j
         group_flavour_complete=.false.
         if(igroup.lt.1.or.igroup.gt.icontr)return
         if(ipro.lt.1.or.ipro.gt.niproc(igroup))return
         if(size(compatible).lt.icontr)return
         do i=1,icontr
            if(.not.compatible(i).or..not.H_event(i))cycle
            do j=1,niproc(i)
               if(.not.all(parton_pdg(:,j,i).eq.
     $              parton_pdg(:,ipro,igroup)))cycle
               if(group_owner(i).ne.igroup)return
            enddo
         enddo
         group_flavour_complete=.true.
         end function group_flavour_complete

         logical function sampled_flavour_complete(igroup,ipro)
! The evaluator sums physical flavours within one sampled component.
! Certify the same sum before source unweighting, including reordered
! member tables. Duplicate physical owner rows would classify a split
! density and are not qualified by this packet format.
         integer, intent(in) :: igroup,ipro
         logical compatible(icontr)
         integer i,component,matches
         sampled_flavour_complete=.false.
         if(igroup.lt.1.or.igroup.gt.icontr)return
         if(ipro.lt.1.or.ipro.gt.niproc(igroup))return
         component=component_line_context(igroup)
         if(component.lt.1)return
         matches=0
         do i=1,niproc(igroup)
            if(all(parton_pdg(:,i,igroup).eq.
     $         parton_pdg(:,ipro,igroup)))matches=matches+1
         enddo
         if(matches.ne.1)return
         compatible=H_event(1:icontr).and.
     $      component_line_context(1:icontr).eq.component
         sampled_flavour_complete=group_flavour_complete(
     $      igroup,ipro,compatible)
         end function sampled_flavour_complete

         subroutine reset_macnlops_line(ict)
         integer, intent(in) :: ict
         macnlops_meta(:,ict)=0d0
         delta_line_context(ict)=0
         component_line_context(ict)=component_context_active
         end subroutine reset_macnlops_line

         subroutine copy_macnlops_line(source,target)
         integer, intent(in) :: source,target
         macnlops_meta(:,target)=macnlops_meta(:,source)
         delta_line_context(target)=delta_line_context(source)
         component_line_context(target)=component_line_context(source)
         end subroutine copy_macnlops_line

         subroutine add_group_member(igroup,ict)
         integer, intent(in) :: igroup,ict
         group_size(igroup)=group_size(igroup)+1
         group_owner(ict)=igroup
         end subroutine add_group_member

         subroutine pack_contribution_groups
! Membership is assigned once per contribution. Packing by its original
! index preserves the order of additions, unweighting and reweighting.
         integer i,igroup,offset
         offset=1
         do i=1,icontr
            group_start(i)=offset
            group_cursor(i)=offset
            offset=offset+group_size(i)
         enddo
         do i=1,icontr
            igroup=group_owner(i)
            group_members(group_cursor(igroup))=i
            group_cursor(igroup)=group_cursor(igroup)+1
         enddo
         end subroutine pack_contribution_groups

         integer function group_member(imember,igroup)
         integer, intent(in) :: imember,igroup
         group_member=group_members(group_start(igroup)+imember-1)
         end function group_member

         subroutine allocate_born_spread_lines(nfolds)
         integer, intent(in) :: nfolds
         double precision, allocatable :: temp(:,:)
         if (.not.allocated(unwgt_B)) then
            allocate(unwgt_B(max_iproc,nfolds))
            allocate(unwgt_noB(max_iproc,nfolds))
            born_spread_fold_capacity=nfolds
         elseif (nfolds.gt.born_spread_fold_capacity) then
            allocate(temp(max_iproc,nfolds))
            temp(:,1:born_spread_fold_capacity)=unwgt_B
            call move_alloc(temp,unwgt_B)
            allocate(temp(max_iproc,nfolds))
            temp(:,1:born_spread_fold_capacity)=unwgt_noB
            call move_alloc(temp,unwgt_noB)
            born_spread_fold_capacity=nfolds
         endif
         unwgt_B=0d0
         unwgt_noB=0d0
         end subroutine allocate_born_spread_lines

         subroutine release_born_spread_lines
         if (allocated(unwgt_B)) deallocate(unwgt_B)
         if (allocated(unwgt_noB)) deallocate(unwgt_noB)
         born_spread_fold_capacity=0
         end subroutine release_born_spread_lines
      end module weight_lines

      subroutine deallocate_born_spread_lines
      use weight_lines, only: release_born_spread_lines
      implicit none
      call release_born_spread_lines
      return
      end


      subroutine weight_lines_allocated(nexternal,n_contr,n_wgt,n_proc)
      use weight_lines
      implicit none
      integer n_contr,n_wgt,n_proc,nexternal,contr_capacity
      logical, allocatable :: ltemp1(:)
      integer, allocatable :: itemp1(:),itemp2(:,:),itemp3(:,:,:)
      double precision, allocatable :: temp1(:),temp2(:,:),temp3(:,:,:)
     $     ,temp4(:,:,:,:)
c Check if we arrays are allocated and if we need to increase the size
c of the allocated arrays.
      if (.not. allocated(itype)) then
         call allocate_weight_lines(nexternal)
      endif
c --- increase size of max_iproc ---
      if (n_proc.gt.max_iproc) then
c parton_pdg_uborn
         allocate(itemp3(nexternal,n_proc,max_contr))
         itemp3(1:nexternal,1:max_iproc,1:max_contr)=parton_pdg_uborn
         call move_alloc(itemp3,parton_pdg_uborn)
c parton_pdg
         allocate(itemp3(nexternal,n_proc,max_contr))
         itemp3(1:nexternal,1:max_iproc,1:max_contr)=parton_pdg
         call move_alloc(itemp3,parton_pdg)
c parton_iproc
         allocate(temp2(n_proc,max_contr))
         temp2(1:max_iproc,1:max_contr)=parton_iproc
         call move_alloc(temp2,parton_iproc)
c unwgt
         allocate(temp2(n_proc,max_contr))
         temp2(1:max_iproc,1:max_contr)=unwgt
         call move_alloc(temp2,unwgt)
c Born-spreading decomposition is allocated only during calibration.
         if (allocated(unwgt_B)) then
            allocate(temp2(n_proc,born_spread_fold_capacity))
            temp2(1:max_iproc,1:born_spread_fold_capacity)=unwgt_B
            call move_alloc(temp2,unwgt_B)
         endif
         if (allocated(unwgt_noB)) then
            allocate(temp2(n_proc,born_spread_fold_capacity))
            temp2(1:max_iproc,1:born_spread_fold_capacity)=unwgt_noB
            call move_alloc(temp2,unwgt_noB)
         endif
c update maximum
         max_iproc=n_proc
      endif
c --- increase size of max_wgt ---
      if (n_wgt.gt.max_wgt) then
c wgts
         allocate(temp2(n_wgt,max_contr))
         temp2(1:max_wgt,1:max_contr)=wgts
         call move_alloc(temp2,wgts)
c plot_wgts
         allocate(temp2(n_wgt,max_contr))
         temp2(1:max_wgt,1:max_contr)=plot_wgts
         call move_alloc(temp2,plot_wgts)
c update maximum
         max_wgt=n_wgt
      endif
c --- increase size of max_contr ---
      if (n_contr.gt.max_contr) then
c Reserve geometrically: adding one native history must not copy all
c contribution arrays on every append. Keep the input arguments intact;
c callers can pass module capacities themselves as actual arguments.
         contr_capacity=max(n_contr,32,2*max_contr)
c H_event
         allocate(ltemp1(contr_capacity))
         ltemp1(1:max_contr)=H_event
         call move_alloc(ltemp1,H_event)
c itype
         allocate(itemp1(contr_capacity))
         itemp1(1:max_contr)=itype
         call move_alloc(itemp1,itype)
c nFKS
         allocate(itemp1(contr_capacity))
         itemp1(1:max_contr)=nFKS
         call move_alloc(itemp1,nFKS)
c event_nFKS
         allocate(itemp1(contr_capacity))
         itemp1=0
         itemp1(1:max_contr)=event_nFKS
         call move_alloc(itemp1,event_nFKS)
c native provenance
         allocate(itemp2(3,contr_capacity))
         itemp2=0
         itemp2(:,1:max_contr)=native_ids
         call move_alloc(itemp2,native_ids)
c MacNLOPS generating-history metadata (new slots are inactive).
         allocate(temp2(6,contr_capacity))
         temp2=0d0
         temp2(:,1:max_contr)=macnlops_meta
         call move_alloc(temp2,macnlops_meta)
c QCDpower         
         allocate(itemp1(contr_capacity))
         itemp1(1:max_contr)=QCDpower
         call move_alloc(itemp1,QCDpower)
c pdg
         allocate(itemp2(nexternal,0:contr_capacity))
         itemp2(1:nexternal,0:max_contr)=pdg
         call move_alloc(itemp2,pdg)
c pdg_uborn
         allocate(itemp2(nexternal,0:contr_capacity))
         itemp2(1:nexternal,0:max_contr)=pdg_uborn
         call move_alloc(itemp2,pdg_uborn)
c parton_pdg_uborn
         allocate(itemp3(nexternal,max_iproc,contr_capacity))
         itemp3(1:nexternal,1:max_iproc,1:max_contr)=parton_pdg_uborn
         call move_alloc(itemp3,parton_pdg_uborn)
c parton_pdg
         allocate(itemp3(nexternal,max_iproc,contr_capacity))
         itemp3(1:nexternal,1:max_iproc,1:max_contr)=parton_pdg
         call move_alloc(itemp3,parton_pdg)
c plot_id
         allocate(itemp1(contr_capacity))
         itemp1(1:max_contr)=plot_id
         call move_alloc(itemp1,plot_id)
c ifold_cnt
         allocate(itemp1(contr_capacity))
         itemp1=0
         itemp1(1:max_contr)=delta_line_context
         call move_alloc(itemp1,delta_line_context)
         allocate(itemp1(contr_capacity))
         itemp1=0
         itemp1(1:max_contr)=component_line_context
         call move_alloc(itemp1,component_line_context)
         allocate(itemp1(contr_capacity))
         itemp1(1:max_contr)=ifold_cnt
         call move_alloc(itemp1,ifold_cnt)
c niproc
         allocate(itemp1(contr_capacity))
         itemp1(1:max_contr)=niproc
         call move_alloc(itemp1,niproc)
c ipr
         allocate(itemp1(contr_capacity))
         itemp1(1:max_contr)=ipr
         call move_alloc(itemp1,ipr)
c orderstag
         allocate(itemp1(contr_capacity))
         itemp1(1:max_contr)=orderstag
         call move_alloc(itemp1,orderstag)
c amppos
         allocate(itemp1(contr_capacity))
         itemp1(1:max_contr)=amppos
         call move_alloc(itemp1,amppos)
c group_size
         allocate(itemp1(contr_capacity))
         itemp1=0
         itemp1(1:max_contr)=group_size
         call move_alloc(itemp1,group_size)
c group_start
         allocate(itemp1(contr_capacity))
         itemp1=0
         itemp1(1:max_contr)=group_start
         call move_alloc(itemp1,group_start)
c group_members
         allocate(itemp1(contr_capacity))
         itemp1=0
         itemp1(1:max_contr)=group_members
         call move_alloc(itemp1,group_members)
c group_owner
         allocate(itemp1(contr_capacity))
         itemp1=0
         itemp1(1:max_contr)=group_owner
         call move_alloc(itemp1,group_owner)
c group_cursor
         allocate(itemp1(contr_capacity))
         itemp1=0
         itemp1(1:max_contr)=group_cursor
         call move_alloc(itemp1,group_cursor)
c momemta
         allocate(temp3(0:3,nexternal,contr_capacity))
         temp3(0:3,1:nexternal,1:max_contr)=momenta
         call move_alloc(temp3,momenta)
c momemta_m
         allocate(temp4(0:3,nexternal,2,contr_capacity))
         temp4(0:3,1:nexternal,1:2,1:max_contr)=momenta_m
         call move_alloc(temp4,momenta_m)
c wgt
         allocate(temp2(3,contr_capacity))
         temp2(1:3,1:max_contr)=wgt
         call move_alloc(temp2,wgt)
c wgt_ME_tree
         allocate(temp2(2,contr_capacity))
         temp2(1:2,1:max_contr)=wgt_ME_tree
         call move_alloc(temp2,wgt_ME_tree)
c bjx
         allocate(temp2(2,contr_capacity))
         temp2(1:2,1:max_contr)=bjx
         call move_alloc(temp2,bjx)
c scales2
         allocate(temp2(3,contr_capacity))
         temp2(1:3,1:max_contr)=scales2
         call move_alloc(temp2,scales2)
c g_strong
         allocate(temp1(contr_capacity))
         temp1(1:max_contr)=g_strong
         call move_alloc(temp1,g_strong)
c wgts
         allocate(temp2(max_wgt,contr_capacity))
         temp2(1:max_wgt,1:max_contr)=wgts
         call move_alloc(temp2,wgts)
c parton_iproc
         allocate(temp2(max_iproc,contr_capacity))
         temp2(1:max_iproc,1:max_contr)=parton_iproc
         call move_alloc(temp2,parton_iproc)
c y_bst
         allocate(temp1(contr_capacity))
         temp1(1:max_contr)=y_bst
         call move_alloc(temp1,y_bst)
c cpower
         allocate(temp1(contr_capacity))
         temp1(1:max_contr)=cpower
         call move_alloc(temp1,cpower)
c bias_wgt
         allocate(temp1(contr_capacity))
         temp1(1:max_contr)=bias_wgt
         call move_alloc(temp1,bias_wgt)
c plot_wgts
         allocate(temp2(max_wgt,contr_capacity))
         temp2(1:max_wgt,1:max_contr)=plot_wgts
         call move_alloc(temp2,plot_wgts)
c shower_scale
         allocate(temp1(contr_capacity))
         temp1(1:max_contr)=shower_scale
         call move_alloc(temp1,shower_scale)
c unwgt
         allocate(temp2(max_iproc,contr_capacity))
         temp2(1:max_iproc,1:max_contr)=unwgt
         call move_alloc(temp2,unwgt)
c need_match
         allocate(itemp2(nexternal,1:contr_capacity))
         itemp2(1:nexternal,1:max_contr)=need_match
         call move_alloc(itemp2,need_match)
c update maximum
         max_contr=contr_capacity
      endif
      return
      end

      subroutine allocate_weight_lines(nexternal)
      use weight_lines
      implicit none
      integer nexternal
      allocate(H_event(1))
      allocate(itype(1))
      allocate(nFKS(1))
      allocate(event_nFKS(1))
      event_nFKS=0
      allocate(native_ids(3,1))
      native_ids=0
      allocate(delta_line_context(1))
      delta_line_context=0
      allocate(component_line_context(1))
      component_line_context=0
      allocate(macnlops_meta(6,1))
      macnlops_meta=0d0
      allocate(QCDpower(1))
      allocate(pdg(nexternal,0:1))
      allocate(pdg_uborn(nexternal,0:1))
      allocate(parton_pdg_uborn(nexternal,1,1))
      allocate(parton_pdg(nexternal,1,1))
      allocate(plot_id(1))
      allocate(ifold_cnt(1))
      allocate(niproc(1))
      allocate(ipr(1))
      allocate(orderstag(1))
      allocate(amppos(1))
      allocate(group_size(1))
      group_size=0
      allocate(group_start(1))
      group_start=0
      allocate(group_members(1))
      group_members=0
      allocate(group_owner(1))
      group_owner=0
      allocate(group_cursor(1))
      group_cursor=0
      allocate(momenta(0:3,nexternal,1))
      allocate(momenta_m(0:3,nexternal,2,1))
      allocate(wgt(3,1))
      allocate(wgt_ME_tree(2,1))
      allocate(bjx(2,1))
      allocate(scales2(3,1))
      allocate(g_strong(1))
      allocate(wgts(1,1))
      allocate(parton_iproc(1,1))
      allocate(y_bst(1))
      allocate(cpower(1))
      allocate(bias_wgt(1))
      allocate(plot_wgts(1,1))
      allocate(shower_scale(1))
      allocate(unwgt(1,1))
      allocate(need_match(nexternal,1))
      max_contr=1
      max_wgt=1
      max_iproc=1
      return
      end

      subroutine deallocate_weight_lines
      use weight_lines
      implicit none
      max_contr=0
      max_wgt=0
      max_iproc=0
      if (allocated(H_event)) deallocate(H_event)
      if (allocated(itype)) deallocate(itype)
      if (allocated(nFKS)) deallocate(nFKS)
      if (allocated(event_nFKS)) deallocate(event_nFKS)
      if (allocated(native_ids)) deallocate(native_ids)
      if (allocated(delta_line_context)) deallocate(delta_line_context)
      if (allocated(component_line_context))deallocate(component_line_context)
      call reset_delta_contexts()
      if (allocated(macnlops_meta)) deallocate(macnlops_meta)
      macnlops_gate_active=.false.
      macnlops_proposal_active=.false.
      macnlops_current_z=0d0
      macnlops_kernel_alphas=0d0
      macnlops_kernel_q2fact=0d0
      if (allocated(QCDpower)) deallocate(QCDpower)
      if (allocated(pdg)) deallocate(pdg)
      if (allocated(pdg_uborn)) deallocate(pdg_uborn)
      if (allocated(parton_pdg_uborn)) deallocate(parton_pdg_uborn)
      if (allocated(parton_pdg)) deallocate(parton_pdg)
      if (allocated(plot_id)) deallocate(plot_id)
      if (allocated(ifold_cnt)) deallocate(ifold_cnt)
      if (allocated(niproc)) deallocate(niproc)
      if (allocated(ipr)) deallocate(ipr)
      if (allocated(orderstag)) deallocate(orderstag)
      if (allocated(amppos)) deallocate(amppos)
      if (allocated(group_size)) deallocate(group_size)
      if (allocated(group_start)) deallocate(group_start)
      if (allocated(group_members)) deallocate(group_members)
      if (allocated(group_owner)) deallocate(group_owner)
      if (allocated(group_cursor)) deallocate(group_cursor)
      if (allocated(momenta)) deallocate(momenta)
      if (allocated(momenta_m)) deallocate(momenta_m)
      if (allocated(wgt)) deallocate(wgt)
      if (allocated(wgt_ME_tree)) deallocate(wgt_ME_tree)
      if (allocated(bjx)) deallocate(bjx)
      if (allocated(scales2)) deallocate(scales2)
      if (allocated(g_strong)) deallocate(g_strong)
      if (allocated(wgts)) deallocate(wgts)
      if (allocated(parton_iproc)) deallocate(parton_iproc)
      if (allocated(y_bst)) deallocate(y_bst)
      if (allocated(cpower)) deallocate(cpower)
      if (allocated(bias_wgt)) deallocate(bias_wgt)
      if (allocated(plot_wgts)) deallocate(plot_wgts)
      if (allocated(shower_scale)) deallocate(shower_scale)
      if (allocated(unwgt)) deallocate(unwgt)
      call release_born_spread_lines
      if (allocated(need_match)) deallocate(need_match)
      return
      end
