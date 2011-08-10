      subroutine generate_visible(x,n_var)
c************************************************************************
c   This subroutine generates the momenta associated to visibleparticles
c
c   inputs   1. x(20): random number given by Vegas
c
c   outputs  
c            1. jac_visible: jacobian for the generation of visible momenta
c            2. n_var: dimension of the subspace associated to visible particles
c
c
c   in common  1. Etot: energy of visible particles
c              2. pztot:   momentum of visible particles of visible particles
c              3. misspx: missing momentum along x
c              4. misspy: missing momentum along y
c*************************************************************************
      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
      include 'data.inc'
c
c     arguments
c
      integer n_var
      double precision x(20)
      double precision jac_visible
c
c     local
c
       double precision jac_temp,Emax,sqrts
       integer i,j,k,nu
c
c     global
c
      double precision momenta(0:3,-max_branch:2*nexternal)  ! records the momenta of external/intermediate legs     (MG order)
      double precision mvir2(-max_branch:2*nexternal)        ! records the sq invariant masses of intermediate particles (MG order)
      common /to_diagram_kin/ momenta, mvir2

      double precision Etot,pztot,misspx,misspy
      common /to_missingP/Etot,pztot,misspx,misspy

      double precision c_point(1:nexternal,3,2)
      common/ph_sp_init/c_point
      double precision pmass(1:nexternal)
      common / to_mass/pmass
c
      double precision gen_var(nexternal,3)
      common /to_generate_var/gen_var
      double precision              S,X1,X2,PSWGT,JAC
      common /PHASESPACE/ S,X1,X2,PSWGT,JAC

      integer config,perm_pos
      common / to_config/config,perm_pos
c---
c begin code
c---

c
c     initialize variables
c
      sqrts=dsqrt(s)
      Emax=sqrts
      jac_visible=1d0
      misspx=0d0
      misspy=0d0
      Etot=0d0
      pztot=0d0
      n_var=0        ! n_var labels the variables of integration
c
c     start the loop over visible particles (k=phase space generation order)
c
c      write(*,*) 'data information'
c      write(*,*) 'config',config
c      write(*,*) 'vis_NB',vis_nb
c      write(*,*) 'num_vis',num_vis


      if (num_vis(config).ge.1) then
      do k=1,num_vis(config)
c
c     determine the label in MG order
c
        i=vis_nb(k,config)  ! i = MG order
c        write(*,*) 'generate' ,i
c
c      start the loop over
c              theta   (j=1),
c              phi     (j=2),
c      and     rho     (j=3).
c
        do j=1,3

c
c         if width is zero, just take the exp. component (TF=delta function)
c
          if(c_point(i,j,2).eq.0d0) then
            gen_var(i,j)=c_point(i,j,1)
c            write(*,*) "var ",j," is a delta"
c
c         if width is positive, generate the component
c
          elseif(c_point(i,j,2).gt.0d0) then

             n_var=n_var+1     ! update the component of random variable

c              write(*,*) "var ",j," is generate ",
c     &          "following Transfer functions"
c              write(*,*) "input",c_point(i,j,1),c_point(i,j,2),x(n_var),
c     &                          gen_var(i,j),jac_temp,j

            call get_component(c_point(i,j,1),c_point(i,j,2),x(n_var),
     &                          gen_var(i,j),jac_temp,j,Emax)
 
            jac_visible=jac_visible*jac_temp
            if (jac_temp.le.0d0) then
              jac=-1d0 
              return
            endif

c            write(*,*) "output part",i,c_point(i,j,1),c_point(i,j,2)
c     & ,x(n_var), gen_var(i,j),jac_temp,j

          else
c
c          width < 0  means that the observable is fixed by conservation of P
c
            write(*,*)'Error : wrong definition ',
     & ' of the width for madgraph num: ',i
            STOP
          endif
c
        enddo
c-----------------------------------------------------------------
c       Now theta,phi and |p| of particle i (MG label) are defined.
c       define the momentum in a Lorentz fashion,
c       and record the result in momenta(#,i)
c------------------------------------------------------------------
        call four_momentum(gen_var(i,1),gen_var(i,2),gen_var(i,3),
     &    pmass(i),momenta(0,i))

c         write(*,*) 'p',i,(momenta(nu,i),nu=0,3)

c----------------------------------------------
c     update missing transverse momentum
c----------------------------------------------

        misspx=misspx-momenta(1,i)
        misspy=misspy-momenta(2,i)

c----------------------------------------------
c     update Etot and pztot for visible particles
c----------------------------------------------
        Etot=Etot+momenta(0,i)
        pztot=pztot+momenta(3,i)
        Emax=sqrts-Etot
c----------------------------------------------
c     update jacobian
c----------------------------------------------
      jac_visible=jac_visible
     & *gen_var(i,3)**2*dsin(gen_var(i,1))/(2d0*momenta(0,i))

c       write(*,*) 'p',i,(momenta(j,i),j=0,3)
c       pause
       enddo

c
c  --- end loop over visible particle ---
c
c       write(*,*) 'Etot,pz,px,py',Etot,pztot,misspx,misspy
       jac=jac*jac_visible
       endif
       return
       end

