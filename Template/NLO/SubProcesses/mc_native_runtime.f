      subroutine mc_sync_native_tables()
! Activate immutable native metadata. Numerical Born COMMONs remain local.
      use mc_native_context, only: native_metadata
      implicit none
      include 'nexternal.inc'
      include 'genps.inc'
      include 'born_nhel.inc'
      include 'nFKSconfigs.inc'
      integer idup(nexternal-1,maxproc),mothup(2,nexternal-1,maxproc)
      integer icolup(2,nexternal-1,max_bcol)
      include 'born_leshouche.inc'
      include 'born_conf.inc'
      include 'born_coloramps.inc'
      integer nb,nc,np,nf,pow(-nexternal:0,lmaxconfigs)
      double precision masses(-nexternal:0,lmaxconfigs,0:fks_configs)
      double precision widths(-nexternal:0,lmaxconfigs,0:fks_configs)
      integer forest_all(2,-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer sprop_all(-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer tprid_all(-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer map_all(0:lmaxconfigs,0:fks_configs)
      common/c_configurations/masses,widths,forest_all,sprop_all,
     $     tprid_all,map_all
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      nb=size(native_metadata%tree,2)
      nc=native_metadata%configurations(0)
      np=native_metadata%nprocesses
      nf=native_metadata%ncolor
      idup=0
      mothup=0
      icolup=0
      idup(:,1:np)=native_metadata%born_ids
      mothup(:,:,1:np)=native_metadata%born_mothers
      icolup(:,:,1:nf)=native_metadata%born_colours
      mapconfig=0
      iforest=0
      sprop=0
      tprid=0
      gforcebw=.false.
      mapconfig(0:nc)=native_metadata%configurations(0:nc)
      iforest(:,-nb:-1,1:nc)=native_metadata%tree(:,:,1:nc)
      sprop(-nb:-1,1:nc)=native_metadata%sprop(:,1:nc)
      tprid(-nb:-1,1:nc)=native_metadata%tprid(:,1:nc)
      gforcebw(-nb:-1,1:nc)=native_metadata%force_bw(:,1:nc)
      iproc_born=np
      icolamp=.false.
      icolamp(1:nf,1:nc,1)=native_metadata%colour_amplitudes(:,1:nc,1)
      forest_all(:,:,:,0)=iforest
      sprop_all(:,:,0)=sprop
      tprid_all(:,:,0)=tprid
      map_all(:,0)=mapconfig
      call mc_native_props(masses(:,:,0),widths(:,:,0),pow)
      calculatedBorn=.false.
      end

      subroutine mc_native_masses(pmass)
      use mc_native_context, only: ensure_native_context
      implicit none
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      include 'fks_info.inc'
      double precision pmass(nexternal),get_mass_from_id
      external get_mass_from_id
      integer nFKSprocess,k,sector
      common/c_nFKSprocess/nFKSprocess
      call ensure_native_context()
      sector=max(1,nFKSprocess)
      if(sector.gt.fks_configs)stop 'Invalid native mass context'
      do k=1,nexternal
         pmass(k)=get_mass_from_id(pdg_type_d(sector,k))
      enddo
      end

      subroutine mc_set_history(history)
      use mc_native_context, only: set_native_history
      implicit none
      integer history
      call set_native_history(history)
      end

      subroutine mc_filter_native_lum(lum)
      use mc_native_context, only: active_history,history_flavours
      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
      double precision lum,pd(0:maxproc)
      integer iproc,k
      common/subproc/pd,iproc
      if(active_history.eq.0)return
      do k=1,iproc
         if(.not.any(history_flavours(:,active_history).eq.k))pd(k)=0d0
      enddo
      lum=sum(pd(1:iproc))
      if(nincoming.eq.2)lum=lum*389379660d0
      end

      subroutine mc_native_evaluate(p,m,n,extra)
      use mc_native_context
      implicit none
      include 'nexternal.inc'
      include 'orders.inc'
      double precision p(0:3,nexternal-1),charges(nexternal-1)
      common/c_charges_born/charges
      integer m,n,extra,status,a,b,j,k
      logical need_color_links,need_charge_links
      common/c_need_links/need_color_links,need_charge_links
      double complex ans_cnt(2,nsplitorders)
      common/c_born_cnt/ans_cnt
      double precision amp_split_soft(amp_split_size)
      common/to_amp_split_soft/amp_split_soft
      double precision wgt_me_born,wgt_me_real
      common/c_wgt_ME_tree/wgt_me_born,wgt_me_real
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      type(BornRequest),save::request
      type(BornModelState),save::state
      call ensure_native_context()
      request%colour=.false.
      request%charge=.false.
      request%sector=active_sector
      request%charges=charges
      request%m=m
      request%n=n
      request%extra=extra
      if(m.gt.0.and.n.gt.0)then
         request%colour=need_color_links
         request%charge=need_charge_links
      endif
      call mc_capture_model_state(state)
      call born_evaluate(context_providers(active_context),active_context,
     $     p,state,request,native_result,status)
      if(status.ne.0)then
         write(*,*)'Native Born evaluation failed',active_context,
     $        active_sector,status
         stop 1
      endif
      if(m.eq.0.and.n.eq.0.and.extra.eq.0)amp_split=0d0
      amp_split_cnt=(0d0,0d0)
      ans_cnt=(0d0,0d0)
      amp_split_soft=0d0
      do j=1,native_metadata%nsplitorders
         k=native_order_map(j,active_context)
         ans_cnt(:,k)=native_result%counterterms(:,j)
      enddo
      do a=1,native_metadata%namplitudes
         b=native_amplitude_map(a,active_context)
         if(b.eq.0)then
            if(any(native_result%split_counterterms(a,:,:).ne.
     $           (0d0,0d0)))stop 'Unmapped native counterterm order'
            cycle
         endif
         if(m.eq.0.and.n.eq.0.and.extra.eq.0)
     $        amp_split(b)=native_result%amplitudes(a)
         do j=1,native_metadata%nsplitorders
            k=native_order_map(j,active_context)
            amp_split_cnt(b,:,k)=native_result%split_counterterms(a,:,j)
         enddo
         if(native_result%has_soft)
     $        amp_split_soft(b)=native_result%soft(a)
      enddo
      wgt_me_born=native_result%born
      calculatedBorn=.true.
      end

      subroutine sborn_native(p,ans)
      use mc_native_context, only: active_context,local_context,
     $     ensure_native_context,native_result
      implicit none
      include 'nexternal.inc'
      double precision p(0:3,nexternal-1),ans
      call ensure_native_context()
      if(active_context.eq.local_context)then
         call sborn(p,ans)
      else
         call mc_native_evaluate(p,0,0,0)
         ans=native_result%born
      endif
      end

      subroutine sborn_sf_native(p,m,n,ans)
      use mc_native_context, only: active_context,local_context,
     $     native_result
      implicit none
      include 'nexternal.inc'
      integer m,n
      double precision p(0:3,nexternal-1),ans
      if(active_context.eq.local_context)then
         call sborn_sf(p,m,n,ans)
      else
         call mc_native_evaluate(p,m,n,0)
         ans=native_result%correlation
      endif
      end

      subroutine extra_cnt_native(p,icnt,ans)
      use mc_native_context
      implicit none
      include 'nexternal.inc'
      include 'orders.inc'
      integer icnt,j,k
      double precision p(0:3,nexternal-1)
      double complex ans(2,nsplitorders)
      if(active_context.eq.local_context)then
         call extra_cnt(p,icnt,ans)
      else
         call mc_native_evaluate(p,0,0,icnt)
         ans=(0d0,0d0)
         do j=1,native_metadata%nsplitorders
            k=native_order_map(j,active_context)
            ans(:,k)=native_result%extra(:,j)
         enddo
      endif
      end

      double precision function mc_born_flow_weight(i)
      use mc_native_context, only: active_context,local_context,
     $     native_result
      implicit none
      include 'genps.inc'
      integer i
      double precision amp2(ngraphs),jamp2(0:ncolor)
      common/to_amps/amp2,jamp2
      mc_born_flow_weight=0d0
      if(active_context.eq.local_context)then
         if(i.le.ncolor)mc_born_flow_weight=jamp2(i)
      else
         if(i.le.size(native_result%flows))
     $        mc_born_flow_weight=native_result%flows(i)
      endif
      end

      double precision function mc_outer_channel_weight()
! The only integration-channel weight on the complete native H sum.
      implicit none
      include 'nexternal.inc'
      include 'genps.inc'
      include 'born_conf.inc'
      double precision p_born(0:3,nexternal-1),ans,total
      common/pborn/p_born
      double precision amp2(ngraphs),jamp2(0:ncolor)
      common/to_amps/amp2,jamp2
      double precision diagramsymmetryfactor
      common/dsymfactor/diagramsymmetryfactor
      integer this_config,i
      common/to_mconfigs/this_config
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      mc_outer_channel_weight=0d0
      if(p_born(0,1).le.0d0)return
      calculatedBorn=.false.
      call sborn(p_born,ans)
      total=0d0
      do i=1,mapconfig(0)
         total=total+amp2(mapconfig(i))
      enddo
      if(total.gt.0d0)mc_outer_channel_weight=
     $     amp2(mapconfig(this_config))/total*diagramsymmetryfactor
      end
