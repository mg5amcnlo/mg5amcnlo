      subroutine HwU_output(idummy,dummy)
      integer idummy
      double precision dummy
      write (*,*) 'HwU_output should not be called',idummy
      stop 1
      end

      subroutine HwU_add_points
      end

      subroutine HwU_accum_iter(ldummy,idummy,dummy)
      logical ldummy
      integer idummy
      double precision dummy(2)
      end
