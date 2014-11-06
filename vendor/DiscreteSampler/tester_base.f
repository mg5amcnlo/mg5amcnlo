      program tester

          use DiscreteSampler

          type(binID) myBinID
          real*8 mValue
          integer i

          call DS_register_dimension('Dimension1',10)
          call DS_add_bin('Dimension1',11)
          mValue=0.0d0
          do i=1,10
            mValue=mValue+1.0d0
            call DS_add_entry('Dimension1',i,mValue)
          enddo

          write(*,*) 'Before grid update'
          call DS_print_global_info()
          call DS_update_grid()
          write(*,*) 'After grid update'
          call DS_print_global_info()
      end program tester
