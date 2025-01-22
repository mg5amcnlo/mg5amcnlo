      Program DRIVER
c**************************************************************************
c     This is the driver for the whole calulation
c**************************************************************************
      implicit none
      include 'nexternal.inc'
      double precision p(0:3, nexternal), gs, res(3)
      integer i 
      double precision t0, t1
 
      ! read the info from stdin (gstrong and momenta)
200   write(*,*) 'enter gstrong'
      read(*,*) gs
      call update_as_param()
      write(*,*) 'enter momenta'
      do i = 1, nexternal -1
        read(*,*) p(0:3,i)
      enddo

      call cpu_time(t0)
      call ewsudakov(p, gs, res)
      call cpu_time(t1)
      write(*,*) "RES", res
      write(*,*) "TIME", t1-t0
      goto 200

      return

      end




