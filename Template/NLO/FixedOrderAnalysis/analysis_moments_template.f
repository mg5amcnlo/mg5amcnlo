c Example of smooth moments alongside ordinary HwU rate histograms.
c Inclusive energy flow in cos(theta) is soft/collinear safe: a soft
c particle has vanishing weight and collinear energies add. This simple
c example is most useful for e+ e- annihilation. It needs no jet library.
      module moments_analysis_data
      implicit none
      double precision, allocatable :: flow_weights(:)
      end module

      subroutine analysis_begin(nwgt,weights_info)
      use moments_analysis_data
      implicit none
      integer nwgt
      character*(*) weights_info(*)
      if (allocated(flow_weights)) deallocate(flow_weights)
      allocate(flow_weights(nwgt))
      call HwU_inithist(nwgt,weights_info)
c Ordinary HwU_book retains literal bins for rates/discrete quantities.
      call HwU_book(1,'total rate |T@NLO',1,0d0,1d0)
      call HwU_book(2,'total rate |T@LO',1,0d0,1d0)
c Last argument is the number of moments, including the constant mode.
c The 40 bins are only a presentation choice; the expansion has 8 modes.
      call Moments_book(3,'angular energy flow |T@NLO',40,
     &     -1d0,1d0,8)
      call Moments_book(4,'angular energy flow |T@LO',40,
     &     -1d0,1d0,8)
      end

      subroutine analysis_end(dummy)
      implicit none
      double precision dummy
      call HwU_write_file
      end

      subroutine analysis_fill(p,istatus,ipdg,wgts,ibody)
      use moments_analysis_data
      implicit none
      include 'nexternal.inc'
      integer istatus(nexternal),ipdg(nexternal),ibody,i
      double precision p(0:4,nexternal),wgts(*)
      double precision total_energy,norm3,costheta,fraction
      call HwU_fill(1,0.5d0,wgts)
      if (ibody.eq.3) call HwU_fill(2,0.5d0,wgts)
      total_energy=0d0
      do i=1,nexternal
         if (istatus(i).eq.1) then
            total_energy=total_energy+max(0d0,p(0,i))
         endif
      enddo
      if (total_energy.le.0d0) return
      do i=1,nexternal
         if (istatus(i).ne.1.or.p(0,i).le.0d0) cycle
         norm3=sqrt(sum(p(1:3,i)**2))
         if (norm3.le.0d0) cycle
         costheta=max(-1d0,min(1d0,p(3,i)/norm3))
         fraction=p(0,i)/total_energy
         flow_weights=wgts(1:size(flow_weights))*fraction
         call HwU_fill(3,costheta,flow_weights)
         if (ibody.eq.3) call HwU_fill(4,costheta,flow_weights)
      enddo
      end
