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

