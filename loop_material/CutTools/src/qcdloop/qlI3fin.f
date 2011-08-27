      subroutine qlI3fin(Ival0,xpi,ier)
      implicit none
      double complex Ival0 
      double precision xpi(6)
      integer ier 
      ier = 0 
      call ffxc0(Ival0, xpi, ier)
      end
