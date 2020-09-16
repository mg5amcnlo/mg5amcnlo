      subroutine open_topdrawer_file
      implicit none
      logical useitmax
      common/cuseitmax/useitmax
      open(unit=99,file='MADatNLO.top',status='unknown')
      useitmax=.false.
      return
      end

      subroutine close_topdrawer_file
      implicit none
      close(99)
      return
      end

      subroutine open_root_file
      implicit none
      logical useitmax
      common/cuseitmax/useitmax
      call rinit('MADatNLO.root')
      useitmax=.true.
      return
      end

      subroutine close_root_file
      implicit none
      call rwrit()
      call rclos()
      return
      end

      subroutine HwU_write_file
      implicit none
      double precision xnorm
c     PineAPPL commons (this may not be the best place to put it)
      include "reweight_pineappl.inc"
      include "pineappl_common.inc"
      logical pineappl
      common /for_pineappl/ pineappl
      integer j
      if(pineappl)then
         do j=1,nh_obs
           appl_obs_num = j
           call APPL_term
         enddo
      endif
      open (unit=99,file='MADatNLO.HwU',status='unknown')
      xnorm=1d0
      call HwU_output(99,xnorm)
      close (99)
      return
      end
