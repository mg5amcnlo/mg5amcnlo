      subroutine get_PS_point(x)

      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
      double precision pi
      parameter (pi=3.141592654d0)
c
c     argument
c
      double precision x(20)
c
c     local
c
      integer n_var
c
c     Global
c
      double precision              S,X1,X2,PSWGT,JAC
      common /PHASESPACE/ S,X1,X2,PSWGT,JAC
      double precision momenta(0:3,-max_branch:2*nexternal)  ! records the momenta of external/intermediate legs     (MG order)
      double precision mvir2(-max_branch:2*nexternal)                  ! records the sq invariant masses of intermediate particles (MG order)
      common /to_diagram_kin/ momenta, mvir2
      integer config,perm_pos
      common /to_config/config,perm_pos
 
c
      include 'data.inc'
c---
c Begin code
c---
      jac=1d0/((2d0*pi)**(3*(nexternal-2)-4))
      n_var=0

      call generate_visible(x,n_var)
       if (jac.le.0d0) then
       return
       endif

c
      if (num_propa(config).gt.0) call generate_propa(x,n_var)
c
      call main_code(x,n_var)
c
      end
