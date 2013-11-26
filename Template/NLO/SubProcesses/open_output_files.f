      subroutine open_topdrawer_file
      implicit none
      open(unit=99,file='MADatNLO.top',status='unknown')
      return
      end

      subroutine close_topdrawer_file
      implicit none
      close(99)
      return
      end

      subroutine open_root_file
      implicit none
      call rinit('MADatNLO.root')
      return
      end

      subroutine close_root_file
      implicit none
      return
      end

