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
      open (unit=99,file='MADatNLO.HwU',status='unknown')
      xnorm=1d0
      call HwU_output(99,xnorm)
      close (99)
      return
      end
