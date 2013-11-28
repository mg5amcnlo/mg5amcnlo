      subroutine rinit(string)
      character*(*) string
      write (*,*) 'rinit (open_output_files_dummy.f) should'/
     $     /' not be called'
      stop
      end


      subroutine rwrit()
      write (*,*) 'rwrit (open_output_files_dummy.f) should'/
     $     /' not be called'
      stop
      end


      subroutine rclos()
      write (*,*) 'rclos (open_output_files_dummy.f) should'/
     $     /' not be called'
      stop
      end
