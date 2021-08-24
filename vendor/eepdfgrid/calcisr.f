      real*8 function isr_tilde(x,beta)
      implicit none
      real*8 x,beta
      real*8 res
      real*8 PI
      PI=4.D0*DATAN(1.D0)
      res=0d0
      if (x .lt. 0.9999999d0) then
      res=-beta*(1d0+x)/2d0
      res=res-beta*beta*(
     c   (1d0+3d0*x*x)/(1d0-x)*dlog(x)
     c      +4d0*(1d0+x)*dlog(1d0-x)+5d0+x)
     c    /8d0
      res=res*(1d0-x)**(1-beta)
      endif
      res=res+
     c (1d0+3d0/4d0*beta+(27d0-8d0*PI*PI)/96d0*beta*beta)
     c  *beta
      isr_tilde=res
      end
