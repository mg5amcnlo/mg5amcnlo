      real*8 function intl1(t1)
      implicit none
      real*8 t1,x1,z,Q2,k1,k2
      real*8 eepdf_tilde,eepdf_tilde_power
      real*8 eepdf_tilde_calc
      integer id
      common /intl/ z,Q2,k1,k2,id
      real*8 res
      x1=1d0-t1**(1d0/(1d0-k1))
      res=eepdf_tilde(x1,Q2,id,11,11)/(1d0-k1)
c     print *,t1,z,x1,z/x1,res
      intl1=res
      end

      real*8 function intl2(t2)
      implicit none
      real*8 t2,x2,z,Q2,k1,k2
      real*8 eepdf_tilde,eepdf_tilde_power
      real*8 eepdf_tilde_calc
      integer id
      common /intl/ z,Q2,k1,k2,id
      real*8 res
      x2=1d0-t2**(1d0/(1d0-k2))
      res=eepdf_tilde(x2,Q2,id,-11,-11)/(1d0-k2)
      intl2=res
      end

      real*8 function totlumi(Q2par,idpar)
      implicit none
      real*8 eepdf_tilde
      real*8 eepdf_tilde_power
      real*8 intl1,intl2
      real*8 dgauss
      external eepdf_tilde,eepdf_tilde_power,intl1,intl2,dgauss
      real*8 zpar,Q2par
      integer idpar
      real*8 z,Q2,k1,k2
      real*8 res_y,res_x,res_grid
      real*8 factor
      integer id
      common /intl/ z,Q2,k1,k2,id
      real*8 res1,res2
      Q2=Q2par
      id=idpar
      k1=eepdf_tilde_power(Q2,id,11,11)
      k2=eepdf_tilde_power(Q2,id,-11,-11)
      res1=dgauss(intl1,0d0,1d0,1d-6)
      res2=dgauss(intl2,0d0,1d0,1d-6)
c     print *,z,id,res1,res2
      totlumi=res1*res2
c     print *,id,res1,res2,totlumi
      end

      program main
      implicit none
      real*8 totlumi,eepdf_tilde_power
      external totlumi,eepdf_tilde_power
      real*8 z,Q2
      real*8 res1,res2,res3,res4,res
      Q2=137d0*137d0
      res1=totlumi(Q2,1)
      res2=totlumi(Q2,2)
      res3=totlumi(Q2,3)
      res4=totlumi(Q2,4)
      res=res1+res2+res3+res4
      print *,res
      end program
