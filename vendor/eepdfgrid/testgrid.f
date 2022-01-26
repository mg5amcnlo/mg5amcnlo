      program testcalc
      implicit none
      real*8 eepdf_tilde_calc,eepdf_tilde
      real*8 x,Q2,jkb
      real*8 res_y,res_x,res_grid
      real*8 factor
      integer id
      x=1.00000000d0
      Q2=10d0*10d0
      read *,id
      do while (x .gt. 1.97d-4)
      res_x = eepdf_tilde_calc(x,Q2,id,11,11)
      res_grid = eepdf_tilde(x,Q2,id,11,11)
      print *,x,res_grid,res_x
      x=x*0.99999d0
      if (x.lt.0.99d0)then
      x=x*0.999d0
      endif
      enddo
      end program
