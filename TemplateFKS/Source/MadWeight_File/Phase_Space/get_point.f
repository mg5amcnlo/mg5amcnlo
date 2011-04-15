      subroutine get_component(c_point,gam,x,gen_point,jac,var_num,Emax)
c
c     this subroutine gets the value of gen_point (PT,phi or y) 
c     from uniform variable x and gives the jacobian of the transformation. 
c     If uniform = false, gen_point is generated according to a Breit Wigner
c     arount c_point,  with width gam.
c
c     input : c_point : central value
c             gam : associated width
c             x: random number
c             var_num = 1 if the variable is a rapidity
c                       2 if the variable is a phi
c                       3 if the variable is a PT

c     output : gen_point: the generated variable
c              jac: the associated jacobian
c
c     arguments
c
      double precision c_point,gam,x,gen_point,jac,Emax
      integer var_num
c
c     local
c
      double precision point_min, point_max
c
c     Parameter
c
      double precision pi,zero
      parameter (pi=3.141592654d0,zero=0d0)

c
c----
c begin code
c----
c
c

c**********************************
c     I.  set width and bounds    *
c**********************************



c     var_num=1 means that we generate a theta
       if (var_num.eq.1) then

      point_max=c_point +5d0*gam
      if (point_max.gt.pi) point_max=pi
      point_min=c_point -5d0*gam
      if (point_min.lt.0d0) point_min=0d0

       gen_point=(point_max-point_min)*x+point_min

c     var_num=2 means that we generate a phi (note that phi is a cyclic variable) 
      elseif(var_num.eq.2) then

      if(gam.lt.(2d0*pi/10)) then
      point_max=c_point +5d0*gam
      point_min=c_point -5d0*gam
      else
      point_max=2*pi
      point_min=0d0
      endif

      gen_point=dble(mod(((point_max-point_min)*x+point_min),2d0*pi))
      if(gen_point.lt.zero) then
      gen_point=gen_point+2d0*pi    ! this is true since phi is cyclic
      endif



c     var_num=3 means that we generate a rho
       elseif (var_num.eq.3) then
      point_max=dble(min(c_point +5d0*gam,Emax))
      point_min=dble(max(c_point -5d0*gam,0.d0))
      if (point_max.le.point_min) then
      jac=-1d0
      return
      endif
       gen_point=(point_max-point_min)*x+point_min

      endif

 


c      write(*,*) '--get point --'
c      write(*,*) 'x', x
c      write(*,*) 'point_max :', point_max
c
c      write(*,*) 'point_min :', point_min
c      write(*,*) 'c_point :',c_point
c      write(*,*) 'ypeak :', ypeak
c
c      write(*,*) 'gam :', gam
c      write(*,*) 'gam2 :', gam2
c      write(*,*) '--end get point --'


c**************************************************************
c     III.  compute the jacobian                              *
c**************************************************************

      jac=(point_max-point_min)

      return
      end

      subroutine get_bjk_fraction(c_point,gam,x,gen_point,jac)
c
c     this subroutine gets the value of Bjorken fraction
c     from uniform variable x and gives the jacobian of the transformation.
c
c     input : c_point : central value
c             gam : associated width
c             x: random number

c     output : gen_point: the generated variable
c              jac: the associated jacobian
c
c     arguments
c
      double precision c_point,gam,x,gen_point,jac
c
c     local
c
      double precision point_min, point_max
c
c     Parameter
c
      double precision zero
      parameter (zero=0d0)
c---
c Begin code
c---

c**********************************
c     I.  set width and bounds    *
c**********************************


      point_max=c_point +5d0*gam
      if (point_max.gt.1d0) point_max=1d0
      point_min=c_point -5d0*gam
      if (point_min.lt.0d0) point_min=0d0

       gen_point=(point_max-point_min)*x+point_min

c**************************************************************
c     II.  compute the jacobian                              *
c**************************************************************

      jac=(point_max-point_min)


      return
      end
