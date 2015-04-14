      program tester

          use DiscreteSampler

          type(binID) myBinID
          real*8 mValue
          integer i
          real*8 jacobian
          integer bid

          call DS_register_dimension('Dimension1',20)
          mValue=1.0d1
          do i=1,20
            mValue=mValue+1.0d1
            call DS_add_entry('Dimension1',i,mValue)
          enddo
          do i=1,20
            mValue=mValue+1.0d1
            call DS_add_entry('Dimension1',i,mValue)
            if (i.eq.10) then
            call DS_add_entry('Dimension1',i,1.0d0)
            endif
          enddo
          call DS_update_grid()
          call DS_write_grid('grids.dsg')
          call DS_clear()
          call DS_register_dimension('Dimension2',30)
          do i=1,30
            mValue=mValue+1.0d1
            call DS_add_entry('Dimension2',i,mValue)
          enddo
          do i=1,30
            mValue=mValue+1.0d1
            call DS_add_entry('Dimension2',i,mValue)
            if (i.eq.10) then
            call DS_add_entry('Dimension2',i,1.0d0)
            endif
          enddo
          call DS_update_grid()
          call DS_write_grid('grids.dsg')
          call DS_clear()
          write(*,*) 'Before grid load'
          call DS_print_global_info()          
          call DS_load_grid('grids.dsg')
          write(*,*) 'after grid load'
          call DS_print_global_info()


          call DS_get_point('Dimension1',
     &           0.4d0, bid, jacobian, 'norm')
          write(*,*) 'I got point ',bid,' with jac',jacobian

          call DS_get_point('Dimension1',
     &           0.1d0, bid, jacobian, 'norm')
          write(*,*) 'I got point ',bid,' with jac',jacobian

          call DS_get_point('Dimension2',
     &           0.7d0, bid, jacobian,'variance')
          write(*,*) 'I got point ',bid,' with jac',jacobian

      end program tester
