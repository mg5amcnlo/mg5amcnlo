*     -*-fortran-*-

      module weight_lines
         implicit none
         integer max_contr,max_wgt,max_iproc,icontr,iwgt,icontr_picked
     $        ,iproc_picked
! Only complete native H weights are retained during repartitioning.
         logical :: mc_H_only=.false.
         logical :: mc_S_only=.false.
         logical, allocatable :: H_event(:)
         integer, allocatable :: itype(:),nFKS(:),QCDpower(:),pdg(:,:)
     $        ,pdg_uborn(:,:),parton_pdg_uborn(:,:,:),parton_pdg(:,:,:)
     $        ,plot_id(:),niproc(:),ipr(:),parton_pdf(:,:,:)
     $        ,ifold_cnt(:)
     $        ,orderstag(:),amppos(:),need_match(:,:)
! Keep the event/shower owner separate from the native PDF/ME history.
         integer, allocatable :: event_nFKS(:),native_ids(:,:)
! Packed contribution groups retain insertion order using linear storage.
         integer, allocatable :: group_size(:),group_start(:)
     $        ,group_members(:),group_owner(:),group_cursor(:)
         double precision, allocatable :: momenta(:,:,:),momenta_m(:,:,:
     $        ,:),wgt(:,:),wgt_ME_tree(:,:),bjx(:,:),scales2(:,:)
     $        ,g_strong(:),wgts(:,:),parton_iproc(:,:),y_bst(:)
     $        ,cpower(:),plot_wgts(:,:),shower_scale(:),unwgt(:,:)
     $        ,bias_wgt(:),shower_scale_a(:,:,:)
         save
      contains
         integer function event_owner(ict)
         integer, intent(in) :: ict
         event_owner=nFKS(ict)
         if (event_nFKS(ict).gt.0) event_owner=event_nFKS(ict)
         end function event_owner

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
      end module weight_lines


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
c parton_pdf
         allocate(itemp3(nexternal,n_proc,max_contr))
         itemp3(1:nexternal,1:max_iproc,1:max_contr)=parton_pdf
         call move_alloc(itemp3,parton_pdf)
c unwgt
         allocate(temp2(n_proc,max_contr))
         temp2(1:max_iproc,1:max_contr)=unwgt
         call move_alloc(temp2,unwgt)
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
c parton_pdf
         allocate(itemp3(nexternal,max_iproc,contr_capacity))
         itemp3(1:nexternal,1:max_iproc,1:max_contr)=parton_pdf
         call move_alloc(itemp3,parton_pdf)
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
c shower_scale_a
         allocate(temp3(contr_capacity,nexternal,nexternal))
         temp3(1:max_contr,1:nexternal,1:nexternal)=shower_scale_a
         call move_alloc(temp3,shower_scale_a)
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
      allocate(parton_pdf(nexternal,1,1))
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
      allocate(shower_scale_a(1,nexternal,nexternal))
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
      if (allocated(parton_pdf)) deallocate(parton_pdf)
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
      if (allocated(shower_scale_a)) deallocate(shower_scale_a)
      if (allocated(unwgt)) deallocate(unwgt)
      if (allocated(need_match)) deallocate(need_match)
      return
      end
