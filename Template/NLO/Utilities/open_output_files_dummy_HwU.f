      subroutine rinit(string)
      character*(*) string
      write (*,*) 'rinit (open_output_files_dummy_HwU.f) should'/
     $     /' not be called'
      stop
      end


      subroutine rwrit()
      write (*,*) 'rwrit (open_output_files_dummy_HwU.f) should'/
     $     /' not be called'
      stop
      end


      subroutine rclos()
      write (*,*) 'rclos (open_output_files_dummy_HwU.f) should'/
     $     /' not be called'
      stop
      end


      subroutine mclear
      return
      end
