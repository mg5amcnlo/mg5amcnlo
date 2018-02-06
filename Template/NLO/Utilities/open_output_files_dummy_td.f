      subroutine rinit(string)
      character*(*) string
      write (*,*) 'rinit (open_output_files_dummy_td.f) should'/
     $     /' not be called'
      stop
      end


      subroutine rwrit()
      write (*,*) 'rwrit (open_output_files_dummy_td.f) should'/
     $     /' not be called'
      stop
      end


      subroutine rclos()
      write (*,*) 'rclos (open_output_files_dummy_td.f) should'/
     $     /' not be called'
      stop
      end


      subroutine HwU_output(unit,xnorm)
      write (*,*) 'HwU_output (open_output_files_dummy_td.f) should'/
     $     /' not be called'
      stop
      end


      subroutine HwU_add_points()
      return
      end


      subroutine set_error_estimation(idummy)
      return
      end


      subroutine finalize_histograms(nevts)
      return
      end
