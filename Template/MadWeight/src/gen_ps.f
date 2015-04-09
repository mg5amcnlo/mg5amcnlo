      subroutine get_PS_point(x)
c***************************************************************************
c     This routine generates the phase-space point and compute the jacobian
c***************************************************************************
      implicit none
      include 'phasespace.inc'
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
      double precision Emiss_weight
c
c     Global
c
      double precision  missPhi_EXP, missPT_EXP
      common /to_missEXP/  missPhi_EXP, missPT_EXP
      double precision pxISR, pyISR
      common /to_ISR/  pxISR, pyISR
      double precision              S,X1,X2,PSWGT,JAC
      common /PHASESPACE/ S,X1,X2,PSWGT,JAC
      double precision momenta(0:3,-max_branches:2*max_particles)   ! momenta of external/intermediate legs     (MG order)
      double precision mvir2(-max_branches:2*max_particles)         ! squared invariant masses of intermediate particles (MG order)
      common /to_diagram_kin/ momenta, mvir2

c
      include 'data.inc'
c---
c Begin code
c---
      jac=1d0/((2d0*pi)**(3*(nexternal-2)-4))
      n_var=0

c     first generate the reconstructed quantities
      call generate_visible(x,n_var)
      if (jac.le.0d0) then
        return
      endif

      if (missPT_EXP.ge.0d0) then  ! using reconstructed missing pT to apply the boost correction
         call generate_miss_parton(Emiss_weight)
         jac=jac*Emiss_weight
      else
         pxISR=0d0
         pyISR=0d0
      endif

c     then generate the invariant mass of mapped resonances, if any
      if (num_propa(config_pos).gt.0) call generate_propa(x,n_var)

c     finally, solve the equations associated with the change of variables in each block
      call main_code(x,n_var)

      end


      subroutine generate_visible(x,n_var)
c************************************************************************
c   This subroutine generates the momenta associated to visible particles
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
      include 'phasespace.inc'
      include 'nexternal.inc'
      include 'data.inc'
      include 'permutation.inc'
c
c     arguments
c
      integer n_var
      integer local_var
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
      double precision momenta(0:3,-max_branches:2*max_particles)  ! records the momenta of external/intermediate legs     (MG order)
      double precision mvir2(-max_branches:2*max_particles)        ! records the sq invariant masses of intermediate particles (MG order)
      common /to_diagram_kin/ momenta, mvir2

      double precision Etot,pztot,misspx,misspy
      common /to_missingP/Etot,pztot,misspx,misspy

      double precision c_point(NPERM,1:max_particles,3,2)
      common/ph_sp_init/c_point
      double precision pmass(1:max_particles)
      common / to_mass/pmass
c
      double precision gen_var(nexternal,3)
      common /to_generate_var/gen_var
      double precision              S,X1,X2,PSWGT,JAC
      common /PHASESPACE/ S,X1,X2,PSWGT,JAC

c---
c begin code
c---

c     initialize variables
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
      if (num_vis(config_pos).ge.1) then
      do k=1,num_vis(config_pos)
c
c     determine the label in MG order
        i=vis_nb(k,config_pos)  ! i = MG order

c      start the loop over
c              theta   (j=1),
c              phi     (j=2),
c      and     rho     (j=3).
        do j=1,3
c
c         write(*,*) "c_point ",i,j,c_point(curr_perm,i,j,2)
c         if width is zero, just take the exp. component (TF=delta function)
          if(c_point(curr_perm,i,j,2).lt.1d-6) then
            gen_var(i,j)=c_point(curr_perm,i,j,1)
c
c         if width is positive, generate the component
          elseif(c_point(curr_perm,i,j,2).gt.0d0) then
             local_var = var2random(3*i-j-3,config_pos)
             n_var=n_var+1     ! update the component of random variable
            call get_component(c_point(curr_perm,i,j,1),c_point(curr_perm,i,j,2),
     &       x(local_var), gen_var(i,j),jac_temp,j,Emax)

            jac_visible=jac_visible*jac_temp
            if (jac_temp.le.0d0) then
              jac=-1d0 
              return
            endif

          else
c
c          width < 0  means that the observable is fixed by conservation of P
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

c       write(*,*) "momentum ",i,momenta(0,i),momenta(1,i),momenta(2,i),momenta(3,i)
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

       enddo

c
c  --- end loop over visible particle ---
c
       jac=jac*jac_visible
       endif
       return
       end

      subroutine get_component(c_point,gam,x,gen_point,jac,var_num,Emax)
c******************************************************************************
c
c     This subroutine gets the value of gen_point (PT,phi or y) 
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
c******************************************************************************
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

      if (gam.gt.0) then
      point_max=c_point +5d0*gam
      if (point_max.gt.pi) point_max=pi
      point_min=c_point -5d0*gam
      if (point_min.lt.0d0) point_min=0d0
      else
        point_max=pi
        point_min=0d0
      endif

       gen_point=(point_max-point_min)*x+point_min

c     var_num=2 means that we generate a phi (note that phi is a cyclic variable) 
      elseif(var_num.eq.2) then

      if(gam.lt.(2d0*pi/10).and.gam.gt.0d0) then
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
            if (gam.ge.100) then
               point_max = 1d3
               point_min = 0
            else
                point_max=dble(min(c_point +5d0*gam,Emax))
                point_min=dble(max(c_point -5d0*gam,0.d0))
c                if (point_max.le.point_min) then
c                    jac=-1d0
c                return
            endif
       gen_point=(point_max-point_min)*x+point_min

      endif

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


      subroutine generate_propa(x,n_var)
c******************************************************************************
c     This subroutine generates the invariant masses mapped onto variable of integration 
c******************************************************************************
      implicit none
c
      include 'phasespace.inc'
      include 'nexternal.inc'
      include 'data.inc' 
      include 'coupl.inc' 
      include 'madweight_param.inc' 
c
c     argument
c
      double precision x(20)
      integer n_var
      integer local_var
c
c     parameter
c
      double precision zero, pi
      parameter (zero=0d0, pi=3.14159265358979d0)
c
c     local
c
      double precision upper_bound,lower_bound,jac_temp,y,pole,gam
      integer i,j

c
c     global
c
      double precision momenta(0:3,-max_branches:2*max_particles)  ! records the momenta of external/intermediate legs     (MG order)
      double precision mvir2(-max_branches:2*max_particles)        ! records the sq invariant masses of intermediate particles (MG order)
      common /to_diagram_kin/ momenta, mvir2
      double precision pmass2(max_particles)     ! records the pole mass of any particle of the diagram  (MG order)
      common / to_mass/pmass2
c
      double precision prmass(-max_branches:0,max_configs)
      double precision prwidth(-max_branches:0,max_configs)
      double precision pow(-max_branches:0,max_configs)
      common /to_MWconfig/prmass, prwidth,pow
c
      double precision              S,X1,X2,PSWGT,JAC
      common /PHASESPACE/ S,X1,X2,PSWGT,JAC

c---
c Begin code
c---
      local_var = 0
      include 'props.inc'
      do i=1,num_propa(config_pos)
        if (prwidth(propa_cont(i,config_pos),1).le.nwa) then
          mvir2(propa_cont(i,config_pos))=prmass(propa_cont(i,config_pos),1)**2
          jac=jac*prmass(propa_cont(i,config_pos),1)
     .    *prwidth(propa_cont(i,config_pos),1)*pi
          goto 15
        endif
c     below we do not use the narrow width approximation
c     upper bound
        if (propa_max(i,1,config_pos).lt.0) then
          upper_bound=dsqrt(mvir2(propa_max(i,1,config_pos)))
          do j=2,max_branches
            if (propa_max(i,j,config_pos).lt.0) then
             upper_bound=upper_bound-dsqrt(mvir2(propa_max(i,j,config_pos)))
            elseif (propa_max(i,j,config_pos).gt.0) then
             upper_bound=upper_bound-pmass2(propa_max(i,j,config_pos))
            elseif (propa_max(i,j,config_pos).eq.0) then
              upper_bound=upper_bound**2
              goto 13
            endif
          enddo
        else
          upper_bound=s
        endif
13      continue
c
c     lower bound
        if (propa_min(i,1,config_pos).gt.0) then
          lower_bound=pmass2(propa_min(i,1,config_pos))
          do j=2,max_branches
            if (propa_min(i,j,config_pos).gt.0) then
              lower_bound=lower_bound+pmass2(propa_min(i,j,config_pos))
            elseif (propa_min(i,j,config_pos).lt.0) then
            lower_bound=lower_bound+dsqrt(mvir2(propa_min(i,j,config_pos)))
            elseif (propa_min(i,j,config_pos).eq.0) then
              lower_bound=lower_bound**2
              goto 14
            endif
          enddo
        else
          lower_bound=0d0
        endif

14      continue
        pole=(prmass(propa_cont(i,config_pos),1)**2-lower_bound)/
     & (upper_bound-lower_bound)
        gam=(prwidth(propa_cont(i,config_pos),1)*
     & prmass(propa_cont(i,config_pos),1))/(upper_bound-lower_bound)
        n_var=n_var+1
        local_var = local_var +1
        call transpole(pole,gam,x(local_var),y,jac)
        jac=jac*(upper_bound-lower_bound)
        mvir2(propa_cont(i,config_pos))=y*(upper_bound-lower_bound)
     & +lower_bound
 15   continue
      enddo

      return
      end

      Double Precision function  Breit_Wigner_for_part(MG_num,M,W)
C-------------------------------------------------------
C     Retruns the Breit-Wigner weight for resonance MG_num              
C-------------------------------------------------------
      implicit none
C
C     parameter
C
      double precision pi
      parameter (pi=3.141592d0)
C
C     argument
C
      integer MG_num            !MG number of the propagator
      double precision M,W      !Mass and Width
C      
C     global
C
      include 'phasespace.inc'
      include 'nexternal.inc'
      include 'run.inc'
      double precision momenta(0:3,-max_branches:2*max_particles)
      double precision mvir2(-max_branches:2*max_particles)
      common/to_diagram_kin/momenta,mvir2
C     
C     local
C
      double precision E,pp2
      
      E=momenta(0,MG_num)
      pp2=momenta(1,MG_num)**2+momenta(2,MG_num)**2+momenta(3,MG_num)**2
      Breit_Wigner_for_part=(E**2-pp2-M**2)**2+M**2*W**2
      Breit_Wigner_for_part=M*W/Breit_Wigner_for_part/pi
c     Normalisation for the measure d(sqrts)  (and not ds)
      Breit_Wigner_for_part=Breit_Wigner_for_part*2d0*E
      return
      end

      subroutine smear_missing_reco(misspx_reco,misspy_reco,weight)
      IMPLICIT NONE
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c                                                                          c
c    IN THIS FILE, THE USER CAN DEFINE A SMEARING FUNCTION FOR THE         c
c    THE RECONSTRUCTED MISSING ENERGY, AND ADJUST THE WEIGHT               c 
c                                                                          c
c    By default, the transfer function is a delta function                 c
C                                                                          c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      double precision weight
      double precision misspx_reco, misspy_reco

      weight=1d0
      return
      end

      subroutine generate_miss_parton(weight)
      IMPLICIT NONE
      integer k
      double precision weight
      double precision misspx_reco, misspy_reco

      double precision  missPhi_EXP, missPT_EXP
      common /to_missEXP/  missPhi_EXP, missPT_EXP
 
      double precision pxISR, pyISR
      common /to_ISR/  pxISR, pyISR

      double precision Etot,pztot,misspx,misspy
      common /to_missingP/Etot,pztot,misspx,misspy

      double precision px_visible,py_visible
      common /to_pTrec_visible/px_visible,py_visible
c     I. first define the transfer function, smear the information 
c           (by default: TF= delta)

      misspx_reco=missPT_EXP*dcos(missPhi_EXP)
      misspy_reco=missPT_EXP*dsin(missPhi_EXP)

      call smear_missing_reco(misspx_reco,misspy_reco,weight)

c     II. Now define the transverse momentum associated with the ISR
c          = -pT_miss_reco - [ sum (pT of visible particles) ] 
c
      pxISR =-misspx_reco -px_visible 
      pyISR =-misspy_reco -py_visible
c      Note that the last two equalities hold in the case (.not. ISR .and. num_inv.eq.0)

c     III. Finally: the values in the variables misspx, misspy are matched 
c          to -px(neutrinos) and -pz(neutrinos) later on in the code, so 
c          these variables should be shifted by px_ISR and py_ISR  
c
c         Note that [ sum (pT of visible particles) ] = (-misspx, -misspy)
c         in the code
      misspx=misspx-pxISR
      misspy=misspy-pyISR

      return
      end

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine generate_variable(x,varnb, mgid, out, jac_loc)
ccccccccccccccccc
c
c     varnb: variable type: 1 if the variable is a rapidity
c                           2 if the variable is a phi
c                           3 if the variable is a PT
c     mgid : mg number associated to the particle
c     out: output variable
c     jac: this variable is going to be updated with the local jac factor
c
cccccccccccccccccc
      double precision x(20)
      integer varnb, mgid
      double precision out, jac_loc
c      local
      integer local_var
      double precision jac_temp

      include 'permutation.inc'
      include 'phasespace.inc'
      include 'data.inc'

      double precision c_point(NPERM,1:max_particles,3,2)
      common/ph_sp_init/c_point

      double precision    S,X1,X2,PSWGT,JAC
      common /PHASESPACE/ S,X1,X2,PSWGT,JAC



      if(c_point(curr_perm,mgid,varnb,2).eq.0d0) then
            out=c_point(curr_perm,mgid,varnb,1)
      elseif(c_point(curr_perm,mgid,varnb,2).gt.0d0.or.c_point(curr_perm,mgid,varnb,2).lt.0d0) then
            local_var = var2random(3*mgid-varnb-3,config_pos)
            call get_component(c_point(curr_perm,mgid,varnb,1),
     &                         c_point(curr_perm,mgid,varnb,2),
     &                         x(local_var),
     &                         out,
     &                         jac_temp,
     &                         varnb,
     &                         sqrt(S))

      jac_loc=jac_loc*jac_temp
      endif
      return
      end

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine generate_bjk_fraction(x,POS, bjk, jac)
ccccccccccccccccc

      integer POS
      double precision bjk,jac_temp, jac

      include 'permutation.inc'
      include 'phasespace.inc'
      include 'data.inc'

      double precision c_point(NPERM,1:max_particles,3,2)
      common/ph_sp_init/c_point

      double precision x(20)



      if (c_point(curr_perm,POS,1,2).gt.0d0) then
        call  get_bjk_fraction(c_point(curr_perm,POS,1,1),
     &  c_point(curr_perm,1,1,2),x(var2random(POS,config_pos)),
     &                  bjk,jac_temp)
        jac=jac*jac_temp
      elseif(c_point(curr_perm,POS,1,2).eq.0d0) then
        bjk=c_point(curr_perm,POS,1,1)
      elseif (c_point(curr_perm,POS,1,2).lt.0d0) then
        bjk=x(var2random(1,config_pos))
      endif
      return
      end

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine generate_flat(x,varnb, vmin, vmax, out, jac_loc)
ccccccccccccccccc

      integer varnb
      double precision vmin, vmax, out, jac_loc

      include 'permutation.inc'
      include 'phasespace.inc'
      include 'data.inc'

      double precision c_point(NPERM,1:max_particles,3,2)
      common/ph_sp_init/c_point

      double precision x(20)

      out     = (VMAX-VMIN)*x(var2random(varnb,config_pos))+VMIN
      jac_loc = jac_loc * (VMAX-VMIN)

      return
      end
