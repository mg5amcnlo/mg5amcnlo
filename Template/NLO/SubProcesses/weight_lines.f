*     -*-fortran-*-

      module weight_lines
         implicit none
         integer max_contr,max_wgt,max_iproc,icontr,iwgt,icontr_picked
     $        ,iproc_picked
         logical, allocatable :: H_event(:)
         integer, allocatable :: itype(:),nFKS(:),QCDpower(:),pdg(:,:)
     $        ,pdg_uborn(:,:),parton_pdg_uborn(:,:,:),parton_pdg(:,:,:)
     $        ,plot_id(:),niproc(:),parton_pdf(:,:,:),icontr_sum(:,:)
         double precision, allocatable :: momenta(:,:,:),momenta_m(:,:,:
     $        ,:),wgt(:,:),wgt_ME_tree(:,:),bjx(:,:),scales2(:,:)
     $        ,g_strong(:),wgts(:,:),parton_iproc(:,:),y_bst(:)
     $        ,plot_wgts(:,:),shower_scale(:),unwgt(:,:),bias_wgt(:)
         save
      end module weight_lines


      subroutine weight_lines_allocated(nexternal,n_contr,n_wgt,n_proc)
      use weight_lines
      implicit none
      integer n_contr,n_wgt,n_proc,nexternal
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
c parton_pdf
         allocate(itemp3(nexternal,n_proc,max_contr))
         itemp3(1:nexternal,1:max_iproc,1:max_contr)=parton_pdf
         call move_alloc(itemp3,parton_pdf)
c parton_iproc
         allocate(temp2(n_proc,max_contr))
         temp2(1:max_iproc,1:max_contr)=parton_iproc
         call move_alloc(temp2,parton_iproc)
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
c --- increase size of max_iproc ---
      if (n_contr.gt.max_contr) then
c H_event
         allocate(ltemp1(n_contr))
         ltemp1(1:max_contr)=H_event
         call move_alloc(ltemp1,H_event)
c itype
         allocate(itemp1(n_contr))
         itemp1(1:max_contr)=itype
         call move_alloc(itemp1,itype)
c nFKS
         allocate(itemp1(n_contr))
         itemp1(1:max_contr)=nFKS
         call move_alloc(itemp1,nFKS)
c QCDpower         
         allocate(itemp1(n_contr))
         itemp1(1:max_contr)=QCDpower
         call move_alloc(itemp1,QCDpower)
c pdg
         allocate(itemp2(nexternal,0:n_contr))
         itemp2(1:nexternal,0:max_contr)=pdg
         call move_alloc(itemp2,pdg)
c pdg_uborn
         allocate(itemp2(nexternal,0:n_contr))
         itemp2(1:nexternal,0:max_contr)=pdg_uborn
         call move_alloc(itemp2,pdg_uborn)
c parton_pdg_uborn
         allocate(itemp3(nexternal,max_iproc,n_contr))
         itemp3(1:nexternal,1:max_iproc,1:max_contr)=parton_pdg_uborn
         call move_alloc(itemp3,parton_pdg_uborn)
c parton_pdg
         allocate(itemp3(nexternal,max_iproc,n_contr))
         itemp3(1:nexternal,1:max_iproc,1:max_contr)=parton_pdg
         call move_alloc(itemp3,parton_pdg)
c plot_id
         allocate(itemp1(n_contr))
         itemp1(1:max_contr)=plot_id
         call move_alloc(itemp1,plot_id)
c niproc
         allocate(itemp1(n_contr))
         itemp1(1:max_contr)=niproc
         call move_alloc(itemp1,niproc)
c parton_pdf
         allocate(itemp3(nexternal,max_iproc,n_contr))
         itemp3(1:nexternal,1:max_iproc,1:max_contr)=parton_pdf
         call move_alloc(itemp3,parton_pdf)
c icontr_sum
         allocate(itemp2(0:n_contr,n_contr))
         itemp2(0:max_contr,1:max_contr)=icontr_sum
         call move_alloc(itemp2,icontr_sum)
c momemta
         allocate(temp3(0:3,nexternal,n_contr))
         temp3(0:3,1:nexternal,1:max_contr)=momenta
         call move_alloc(temp3,momenta)
c momemta_m
         allocate(temp4(0:3,nexternal,2,n_contr))
         temp4(0:3,1:nexternal,1:2,1:max_contr)=momenta_m
         call move_alloc(temp4,momenta_m)
c wgt
         allocate(temp2(3,n_contr))
         temp2(1:3,1:max_contr)=wgt
         call move_alloc(temp2,wgt)
c wgt_ME_tree
         allocate(temp2(2,n_contr))
         temp2(1:2,1:max_contr)=wgt_ME_tree
         call move_alloc(temp2,wgt_ME_tree)
c bjx
         allocate(temp2(2,n_contr))
         temp2(1:2,1:max_contr)=bjx
         call move_alloc(temp2,bjx)
c scales2
         allocate(temp2(3,n_contr))
         temp2(1:3,1:max_contr)=scales2
         call move_alloc(temp2,scales2)
c g_strong
         allocate(temp1(n_contr))
         temp1(1:max_contr)=g_strong
         call move_alloc(temp1,g_strong)
c wgts
         allocate(temp2(max_wgt,n_contr))
         temp2(1:max_wgt,1:max_contr)=wgts
         call move_alloc(temp2,wgts)
c parton_iproc
         allocate(temp2(max_iproc,n_contr))
         temp2(1:max_iproc,1:max_contr)=parton_iproc
         call move_alloc(temp2,parton_iproc)
c y_bst
         allocate(temp1(n_contr))
         temp1(1:max_contr)=y_bst
         call move_alloc(temp1,y_bst)
c bias_wgt
         allocate(temp1(n_contr))
         temp1(1:max_contr)=bias_wgt
         call move_alloc(temp1,bias_wgt)
c plot_wgts
         allocate(temp2(max_wgt,n_contr))
         temp2(1:max_wgt,1:max_contr)=plot_wgts
         call move_alloc(temp2,plot_wgts)
c shower_scale
         allocate(temp1(n_contr))
         temp1(1:max_contr)=shower_scale
         call move_alloc(temp1,shower_scale)
c unwgt
         allocate(temp2(max_iproc,n_contr))
         temp2(1:max_iproc,1:max_contr)=unwgt
         call move_alloc(temp2,unwgt)
c update maximum
         max_contr=n_contr
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
      allocate(QCDpower(1))
      allocate(pdg(nexternal,0:1))
      allocate(pdg_uborn(nexternal,0:1))
      allocate(parton_pdg_uborn(nexternal,1,1))
      allocate(parton_pdg(nexternal,1,1))
      allocate(plot_id(1))
      allocate(niproc(1))
      allocate(parton_pdf(nexternal,1,1))
      allocate(icontr_sum(0:1,1))
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
      allocate(bias_wgt(1))
      allocate(plot_wgts(1,1))
      allocate(shower_scale(1))
      allocate(unwgt(1,1))
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
      if (allocated(QCDpower)) deallocate(QCDpower)
      if (allocated(pdg)) deallocate(pdg)
      if (allocated(pdg_uborn)) deallocate(pdg_uborn)
      if (allocated(parton_pdg_uborn)) deallocate(parton_pdg_uborn)
      if (allocated(parton_pdg)) deallocate(parton_pdg)
      if (allocated(plot_id)) deallocate(plot_id)
      if (allocated(niproc)) deallocate(niproc)
      if (allocated(parton_pdf)) deallocate(parton_pdf)
      if (allocated(icontr_sum)) deallocate(icontr_sum)
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
      if (allocated(bias_wgt)) deallocate(bias_wgt)
      if (allocated(plot_wgts)) deallocate(plot_wgts)
      if (allocated(shower_scale)) deallocate(shower_scale)
      if (allocated(unwgt)) deallocate(unwgt)
      return
      end
