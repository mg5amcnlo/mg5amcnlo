c Process-facing virtual approximation helpers. Kinematic kernels and
c frozen model state live in virtual_features and virtual_pooled.

      subroutine capture_pooled_event_weights(i_soft)
c Retain the actual fully folded S flavours for the model promotion guard.
c The residual is generated separately. Check that the native virtual
c records share a PDF/bias distribution before replaying a new prediction.
      use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
      use weight_lines
      use mint_module, only: pooled_virtual_sampled,pooled_virtual_ratio,
     $     born_wgt_mint,n_ord_virt,wgt_mult
      use virtual_pooled, only: pooled_virtual_set_event_weights
      implicit none
      include 'nexternal.inc'
      include 'genps.inc'
      include 'nFKSconfigs.inc'
      integer i_soft,i,j,jj,nt,source
      integer iproc_save(fks_configs),eto(maxproc,fks_configs),
     $     etoi(maxproc,fks_configs),maxproc_found
      common/cproc_combination/iproc_save,eto,etoi,maxproc_found
      double precision soft(maxproc),avv(maxproc),res(maxproc)
     $     ,fraction(maxproc),aligned(maxproc),norm,expected
      if(.not.pooled_virtual_sampled.or.i_soft.eq.0)return
      nt=niproc(i_soft)
      soft=0d0
      avv=0d0
      res=0d0
      fraction=0d0
      source=0
      do i=1,icontr
         if(H_event(i))cycle
         if((itype(i).eq.14.or.itype(i).eq.15).and.
     $      abs(wgts(1,i)).gt.0d0.and.source.eq.0)source=i
         do jj=1,iproc_save(nFKS(i))
            j=eto(jj,nFKS(i))
            if(j.lt.1.or.j.gt.nt)return
            if(itype(i).eq.14)then
               res(j)=res(j)+parton_iproc(jj,i)
            elseif(itype(i).eq.15)then
               avv(j)=avv(j)+parton_iproc(jj,i)
            else
               soft(j)=soft(j)+parton_iproc(jj,i)
            endif
         enddo
      enddo
      if(source.eq.0)return
      do jj=1,iproc_save(nFKS(source))
         j=eto(jj,nFKS(source))
         fraction(j)=fraction(j)+parton_iproc(jj,source)/wgts(1,source)
      enddo
      if(.not.all(ieee_is_finite(fraction(1:nt))))return
      if(any(fraction(1:nt).lt.0d0))return
      if(abs(sum(fraction(1:nt))-1d0).gt.1d-9)return
c Check each order/fold before summing: cancellations between orders must
c not hide distinct flavour distributions from the promotion guard.
      do i=1,icontr
         if(H_event(i))cycle
         if(itype(i).ne.14.and.itype(i).ne.15)cycle
         aligned=0d0
         do jj=1,iproc_save(nFKS(i))
            j=eto(jj,nFKS(i))
            aligned(j)=aligned(j)+parton_iproc(jj,i)
         enddo
         norm=max(maxval(abs(aligned(1:nt))),abs(wgts(1,i)),
     $        1d-100)
         if(any(abs(aligned(1:nt)-wgts(1,i)*fraction(1:nt))
     $        .gt.1d-9*norm))return
      enddo
      norm=max(maxval(abs(avv(1:nt))),maxval(abs(res(1:nt))),
     $     1d-100)
      if(any(abs(avv(1:nt)-sum(avv(1:nt))*fraction(1:nt))
     $     .gt.1d-9*norm))return
      if(any(abs(res(1:nt)-sum(res(1:nt))*fraction(1:nt))
     $     .gt.1d-9*norm))return
      expected=sum(pooled_virtual_ratio*
     $     born_wgt_mint(1:n_ord_virt))
      if(abs(sum(avv(1:nt))-expected).gt.1d-9*
     $     max(norm,abs(expected)))return
      if(.not.ieee_is_finite(wgt_mult).or.wgt_mult.le.0d0)return
      call pooled_virtual_set_event_weights(soft(1:nt)/wgt_mult,
     $     fraction(1:nt))
      end


      subroutine prepare_pooled_helicities(p,colours)
c Compute precisely the native mixture of nonzero Born squared orders,
c without drawing a helicity or evaluating a virtual matrix element.
      use mc_born_types
      use virtual_pooled, only: pooled_virtual_needs_helicities,
     $ pooled_virtual_set_helicity_weights,pooled_virtual_uses_physics
      implicit none
      include 'nexternal.inc'
      include 'coupl.inc'
      include 'q_es.inc'
      include 'orders.inc'
      include '../../Source/MODEL/input.inc'
      double precision p(0:3,nexternal-1),saved_mu_r
      integer colours(nexternal-1),i,j
      double precision dipoles(nexternal-1,nexternal-1),norm
      double precision,allocatable,save :: weights(:,:)
      logical updateloop,saved_updateloop
      common/to_updateloop/updateloop
      type(BornRequest) request
      type(BornResult),save :: result
      if(.not.pooled_virtual_needs_helicities())return
c Binoth's sampler sees couplings at the Ellis-Sexton scale. Restore all
c model parameters after this cheap prediction pass for subsequent terms.
      saved_mu_r=mu_r
      saved_updateloop=updateloop
      mu_r=sqrt(QES2)
      updateloop=.true.
      call update_as_param()
      updateloop=.false.
c Retain reusable output buffers explicitly. Reset the request before
c obtaining the uncorrelated Born used to normalize the colour contractions.
      request=BornRequest()
      request%helicities=.true.
      call mc_born_local(p,request,result)
      weights=result%helicity_orders
      if(pooled_virtual_uses_physics())then
c The cached tree amplitudes have already been built for the soft terms.
c Contract their colour links without extra helicity draws or loop calls.
         dipoles=0d0
         norm=sum(abs(result%split_counterterms(:,1,qcd_pos)))*g**2
         if(norm.gt.0d0)then
            request%helicities=.false.
            request%colour=.true.
            do i=1,nexternal-2
               if(abs(colours(i)).eq.1)cycle
               do j=i+1,nexternal-1
                  if(abs(colours(j)).eq.1)cycle
                  request%m=i
                  request%n=j
                  call mc_born_local(p,request,result)
                  dipoles(i,j)=sum(result%soft)/norm
               enddo
            enddo
         endif
         call pooled_virtual_set_helicity_weights(weights,dipoles)
      else
         call pooled_virtual_set_helicity_weights(weights)
      endif
      mu_r=saved_mu_r
      updateloop=.true.
      call update_as_param()
      updateloop=saved_updateloop
      end


      subroutine deferred_virtual_context(action,born_wgt)
      use fks_phase_space_data, only: p_born
      use mc_native_context, only: active_history
      use weight_lines
      use mint_module
      implicit none
      include 'nexternal.inc'
      include 'orders.inc'
      include 'coupl.inc'
      include 'q_es.inc'
      include 'run.inc'
      include '../../Source/MODEL/input.inc'
      include 'timing_variables.inc'
      integer action,i,k,saved_fks,saved_history
      integer nFKSprocess,current_fks,current_history
      common/c_nFKSprocess/nFKSprocess
      double precision born_wgt,saved_p(0:3,nexternal-1),saved_born
      double precision saved_amp_born(amp_split_size)
      double precision carrier_raw
      type(pooled_virtual_context),save :: saved_pool_context
      double precision saved_g,saved_scale,saved_qes,virt_wgt
      double precision saved_mu_r
      double precision current_g,current_scale,current_qes,current_mu_r
      double precision carrier,other,ratio(amp_split_size)
      double precision amp_split_virt(amp_split_size),
     $ amp_split_born_for_virt(amp_split_size),
     $ amp_split_avv(amp_split_size)
      common/to_amp_split_virt/amp_split_virt,
     $ amp_split_born_for_virt,amp_split_avv
      double precision amp_split_finite_ML(amp_split_size)
      common/to_amp_split_finite/amp_split_finite_ML
      double precision virtual_over_born
      common/c_vob/virtual_over_born
      logical accepted,updateloop
      common/to_updateloop/updateloop
      save saved_p,saved_born,saved_amp_born,carrier_raw,
     $ saved_g,saved_scale,saved_qes,saved_fks,saved_history,saved_mu_r
      if(action.eq.0)then
         saved_p=p_born
         saved_born=born_wgt
         saved_amp_born=amp_split_born_for_virt
         call pooled_virtual_save_context(saved_pool_context)
         carrier_raw=sum(abs(saved_amp_born))
         saved_g=g
         saved_scale=scale
         saved_qes=QES2
         saved_mu_r=mu_r
         saved_fks=nFKSprocess
         saved_history=active_history
c Match Binoth's cheap model-parameter update before subsequent Born
c and scheme-counterterm evaluations, including running BSM couplings.
         mu_r=sqrt(QES2)
         updateloop=.true.
         call update_as_param()
         updateloop=.false.
         return
      endif
      if(.not.virtual_deferred_ready)return
      carrier=0d0
      other=0d0
      do i=1,icontr
         if(itype(i).eq.14)then
            carrier=carrier+sum(abs(parton_iproc(1:niproc(i),i)))
         else
            other=other+sum(abs(parton_iproc(1:niproc(i),i)))
         endif
      enddo
      call screen_virtual_carrier(carrier,other,accepted)
      if(.not.accepted)then
         icontr=0
         virt_wgt_mint=0d0
         born_wgt_mint=0d0
         pooled_virtual_sampled=.false.
         virtual_deferred_ready=.false.
         return
      endif
c Evaluate at exactly the first fold's saved native Born context, then
c restore the final fold's context. Floating-point scale recomputation
c alone can change QES2 by an ulp across folds.
      current_g=g
      current_scale=scale
      current_qes=QES2
      current_mu_r=mu_r
      current_fks=nFKSprocess
      current_history=active_history
      call update_fks_dir(saved_fks)
      call mc_set_history(saved_history)
      g=saved_g
      scale=saved_scale
      QES2=saved_qes
      mu_r=saved_mu_r
      call pooled_virtual_restore_context(saved_pool_context)
      if(timing_sampled)call timing_enter(timing_virtual)
      call BinothLHA(saved_p,saved_born,virt_wgt)
      if(timing_sampled)call timing_leave(timing_virtual)
      g=current_g
      scale=current_scale
      QES2=current_qes
      mu_r=current_mu_r
      call update_as_param()
      call update_fks_dir(current_fks)
      call mc_set_history(current_history)
      ratio=(amp_split_finite_ML-pooled_virtual_actual_ratio
     $     *saved_amp_born)
     $     /carrier_raw
      virtual_over_born=0d0
      if(saved_born.ne.0d0)virtual_over_born=virt_wgt/saved_born
      do i=1,icontr
         if(itype(i).ne.14)cycle
         k=amppos(i)
         if(k.lt.1.or.k.gt.amp_split_size)stop 'bad deferred order'
         wgt(1:3,i)=wgt(1:3,i)*ratio(k)
         wgts(1,i)=wgts(1,i)*ratio(k)
         parton_iproc(1:niproc(i),i)=
     $        parton_iproc(1:niproc(i),i)*ratio(k)
      enddo
      virt_wgt_mint(1:amp_split_size)=
     $     virt_wgt_mint(1:amp_split_size)*ratio
      virt_wgt_mint(0)=sum(virt_wgt_mint(1:amp_split_size))
      call pooled_virtual_mark_sampled(1d0)
      virtual_deferred_ready=.false.
      end


      subroutine virt_surrogate_fit_point
c Convert the final PDF/bias weights to a fit weight c**2, where
c c = (residual * virtual_fraction + surrogate) / exact_virtual.
      use weight_lines, only: icontr,itype,wgts
      use mint_module, only: imode,ichan,use_poly_virtual,
     $     use_pooled_virtual,pooled_virtual_applying,
     $     VirtTrickSurrogate,polyfit,average_virtual
      use virt_surrogate, only: vs_vcomp,vs_featok,vs_virt,vs_fvirt,
     $     vs_born,vs_accumulate
      implicit none
      integer i
      double precision s14,s15,c,agrid
      if (VirtTrickSurrogate.lt.1.or.imode.ne.0) return
c The collection surrogate adapts at iteration boundaries. Application
c retains the saved fit and the barrier-trained correction unchanged.
      if (use_pooled_virtual.and.pooled_virtual_applying) return
      if (.not.vs_vcomp .or. .not.vs_featok) return
      if (vs_virt.eq.0d0) return
      s14=0d0
      s15=0d0
      do i=1,icontr
         if (itype(i).eq.14) s14=s14+wgts(1,i)
         if (itype(i).eq.15) s15=s15+wgts(1,i)
      enddo
      if (s14.eq.0d0 .and. s15.eq.0d0) return
      c=(s14*vs_fvirt+s15)/vs_virt
      if (use_poly_virtual) then
         agrid=polyfit(0)*vs_born
      else
         agrid=average_virtual(0,ichan)*vs_born
      endif
      call vs_accumulate(ichan,vs_virt,agrid,c**2)
      return
      end
