      subroutine BinothLHA(p_born,born_wgt,virt_wgt)
      implicit none
      include "nexternal.inc"
      double precision p_born(0:3,nexternal-1),virt_wgt,born_wgt
      virt_wgt=0d0
      return
      end

      subroutine BinothLHAInit(filename)
      implicit none
      character*(*) filename
      return
      end
      
      subroutine ctsstatistics(n_mp,n_disc)
      implicit none
      integer n_mp,n_disc
      n_mp=0
      n_disc=0
      return
      end

      subroutine sloopmatrix(p_born,virt_wgts)
      implicit none
      include "nexternal.inc"
      double precision p_born(0:3,nexternal-1),virt_wgts(3)
      virt_wgts(1)=0d0
      virt_wgts(2)=0d0
      virt_wgts(3)=0d0
      return
      end
