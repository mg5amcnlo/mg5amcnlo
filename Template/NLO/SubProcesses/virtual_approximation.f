      subroutine virt_surrogate_fit_point
c Convert the final PDF/bias weights to a fit weight c**2, where
c c = (residual * virtual_fraction + surrogate) / exact_virtual.
      use weight_lines, only: icontr,itype,amppos,wgts
      use mint_module, only: imode,ichan,use_poly_virtual,
     $     VirtTrickSurrogate,polyfit,average_virtual
      use virt_surrogate, only: vs_vcomp,vs_featok,vs_virt,vs_fvirt,
     $     vs_born,vs_accumulate,vs_residual_order,vs_residual_reference
      implicit none
      integer i
      double precision s14,s15,c,agrid
      if (VirtTrickSurrogate.lt.1.or.imode.ne.0) return
      if (.not.vs_vcomp .or. .not.vs_featok) return
      if (vs_virt.eq.0d0) return
      s14=0d0
      s15=0d0
      do i=1,icontr
         if (itype(i).eq.14) s14=s14+wgts(1,i)
         if (itype(i).eq.15) s15=s15+wgts(1,i)
      enddo
      if (vs_residual_order.gt.0) then
c The separate virtual event omits the type-15 approximation. Normalize
c a nonzero raw residual order with all its final flavour/fold records.
         if (vs_residual_reference.eq.0d0) return
         s14=0d0
         do i=1,icontr
            if (itype(i).eq.14.and.amppos(i).eq.vs_residual_order)
     $           s14=s14+wgts(1,i)
         enddo
         c=s14/vs_residual_reference
      else
         if (s14.eq.0d0 .and. s15.eq.0d0) return
         c=(s14*vs_fvirt+s15)/vs_virt
      endif
      if (use_poly_virtual) then
         agrid=polyfit(0)*vs_born
      else
         agrid=average_virtual(0,ichan)*vs_born
      endif
      call vs_accumulate(ichan,vs_virt,agrid,c**2)
      return
      end
