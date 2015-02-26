      program tester

          use DiscreteSampler
          use StringCast
          real*8 jac, to_test
          REAL*8 integral_target, computed_int, variance
          integer i,j
          integer pts_per_it(4), pts
          data pts_per_it/3,6,9,12/
          real*8 avg_acc
          real*8 this_acc
          real*8 mvariance


          mvariance=0.0d0
          do i=1,10
            mvariance = mvariance + (1.0/12.0)*to_test(i,.True.)**2
          enddo

          integral_target = 0.0d0
          do i=1,10
            integral_target = integral_target + to_test(i,.True.)
          enddo
          call DS_register_dimension('TestDim',10)
          avg_acc = 0.0d0
          call DS_set_min_points(10)
          do i=1,10
            call DS_add_entry('conv',i,0.0d0)
          enddo
          call DS_add_entry('conv',7,0.0d0,reset=.True.)
          do j=1,2
            if(j.le.size(pts_per_it)) then
              pts = pts_per_it(j)
            else
              pts = 100000
            endif
c            write(*,*) 'Grid before iteration'            
c            call DS_print_global_info()
            call approx(pts,computed_int,'norm')
            write(*,*) '=== ITERATION #'//trim(toStr(j))//' with '//
     & trim(toStr(pts))//' points.'
            write(*,*) 'exact integral is :',integral_target
            write(*,*) 'computed integral :',computed_int
            this_acc = abs(integral_target-computed_int)/
     &                                               integral_target
            avg_acc = (avg_acc*float(j-1)+this_acc)/float(j)
            write(*,*) 'relative diff (%) :   ',
     &       trim(toStr(this_acc*100.0d0,'Fw.3'))
            write(*,*) 'avg rel. diff (%) :   ',
     &       trim(toStr(avg_acc*100.0d0,'Fw.3'))
            write(*,*) 'Grid before update'            
            call DS_print_global_info()
            call DS_update_grid('TestDim')
            write(*,*) 'Grid after update'
            call DS_print_global_info('TestDim')
          enddo
          call DS_write_grid('rrrr.dsg')
      end program tester


      subroutine approx(n_trials,res,mode)
          use DiscreteSampler          
         integer n_trials
         character(len=*) mode
         real*8 res, to_test, computed_int, jac, func
         REAL*8 r(1)
         integer picked,i
          computed_int = 0.0d0
          do i=1,n_trials
            CALL RANDOM_NUMBER(r)
            call DS_get_point('TestDim',r(1),picked,jac,
     &      mode
     &     ,(/'conv'/))
            func = to_test(picked,.false.)
            computed_int = (computed_int*float(i-1)
     &       + func*jac)/float(i)
            call DS_add_entry('TestDim',picked,func)
          enddo
          res = computed_int

      end subroutine approx

      function to_test(sector,exact)
          integer sector
          real*8 r(1), to_test
          real*8 values(10)
          logical           exact
          data values/5.0d0, -34.0d0, 2.0d0, 3.0d0, 4.0d0,
     &    5.0d0, 2.0d0, 3.0d0, 1.0d0, 2.5d0/

          ! When using the function below, one recovers exact result
          ! with as little as one point using the updated grid
          if (exact) then
            to_test = values(sector)
          else
            CALL RANDOM_NUMBER(r)
            ! This simulates a convergence process
            to_test = (values(sector)*(1.0d0))*(0.5+r(1))
          endif
          return

      end function to_test
