      program tester

          use DiscreteSampler

          type(binID) myBinID
          real*8 mValue
          integer i

          call DS_register_dimension('Dimension1',10)
          call DS_register_dimension('Dimension2',20)

          call DS_add_bin('Dimension2')
          call DS_add_bin('Dimension1',90)
          call DS_add_entry('Dimension1',90,13.0d0)

          mValue=0.0d0
          do i=1,10
            mValue=mValue+1.0d0
            call DS_add_entry('Dimension2',i,mValue)
          enddo

          call DS_remove_bin('Dimension1',DS_binID(90))
          call DS_add_bin('Dimension1',94)
          call DS_remove_bin('Dimension1',94)

          call DS_remove_dimension('Dimension1')

          call DS_print_global_info()
           
      end program tester
