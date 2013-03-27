      subroutine class_h(x,n_var,var2random,p1,p2)
c***************************************************************************
c     ECS in CLASS H
c  
c
c***************************************************************************
      implicit none
      include '../../../SubProcesses/phasespace.inc'
c
c     argument
c
      double precision x(20) 
      integer n_var, p1, p2,local_var
      integer var2random(*)
c
c     local
c
      double precision pboost(0:3), CMS_mom(0:3,max_particles)
      double precision Ptot(0:3),PtotCMS(0:3),Emax
      double precision measureLAB, measureCMS
      double precision jac_temp,det,sqrts
      double precision jac_loc
      integer j,nu,MG,k,i
      double precision gen_var(2,3)
      integer vis(2)
c
c     global
c
      double precision c_point(1:max_particles,3,2)
      common/ph_sp_init/c_point
c
      double precision momenta(0:3,-max_branches:2*max_particles)  ! records the momenta of external/intermediate legs     (MG order)
      double precision mvir2(-max_branches:2*max_particles)        ! records the sq invariant masses of intermediate particles (MG order)
      common /to_diagram_kin/ momenta, mvir2
      double precision pmass(max_particles)     ! records the pole mass of any particle of the diagram  (MG order)
      common / to_mass/pmass
      double precision Etot,pztot,misspx,misspy
      common /to_missingP/Etot,pztot,misspx,misspy
      double precision              S,X1,X2,PSWGT,JAC
      common /PHASESPACE/ S,X1,X2,PSWGT,JAC
      double precision ISRpx,ISRpy
      common /to_ISR/ ISRpx,ISRpy
c
      integer matching_type_part(3:max_particles)
      integer inv_matching_type_part(3:max_particles)
      common/madgraph_order_type/matching_type_part,
     & inv_matching_type_part
      integer nexternal, num_inv
      COMMON/to_num_inv/nexternal, num_inv
c
c     external
c
      double precision dot
      external dot
c---
c Begin code
c---
      jac_loc=1d0
      sqrts=dsqrt(s)
      if((Etot+dsqrt(misspx**2+misspy**2)).gt.sqrts) then
      jac=-1d0
      return
      endif

      vis(1)=p1
      vis(2)=p2
      do i=1,2
c      start the loop over
c              theta   (j=1),
c              phi     (j=2),
c      and     rho     (j=3).
        do j=1,3
c
c         write(*,*) "c_point ",i,j,c_point(i,j,2)
c         if width is zero, just take the exp. component (TF=delta function)
          if(c_point(vis(i),j,2).lt.1d-6) then
            gen_var(i,j)=c_point(vis(i),j,1)
c
c         if width is positive, generate the component
          elseif(c_point(vis(i),j,2).gt.0d0) then

             n_var=n_var+1     ! update the component of random variable
             local_var = var2random(3*vis(i)-j-3)
            call get_component(c_point(vis(i),j,1),c_point(vis(i),j,2),
     &           x(local_var), gen_var(i,j),jac_temp,j,sqrts-Etot)

            jac_loc=jac_loc*jac_temp
            if (jac_temp.le.0d0) then
              jac=-1d0
              return
            endif
          endif
        enddo

c-----------------------------------------------------------------
c       Now theta,phi and |p| of particle i (MG label) are defined.
c       define the momentum in a Lorentz fashion,
c       and record the result in momenta(#,i)
c------------------------------------------------------------------
        call four_momentum(gen_var(i,1),gen_var(i,2),gen_var(i,3),
     &    pmass(vis(i)),momenta(0,vis(i)))

c----------------------------------------------
c     update missing transverse momentum
c----------------------------------------------
        misspx=misspx-momenta(1,vis(i))
        misspy=misspy-momenta(2,vis(i))
c----------------------------------------------
c     update Etot and pztot for visible particles
c----------------------------------------------
        Etot=Etot+momenta(0,vis(i))
        pztot=pztot+momenta(3,vis(i))
        Emax=sqrts-Etot
c----------------------------------------------
c     update jacobian
c----------------------------------------------
      jac_loc=jac_loc
     & *gen_var(i,3)**2*dsin(gen_var(i,1))/(2d0*momenta(0,vis(i)))

      enddo


c      write(*,*) "jac init class h", jac
c      write(*,*) "mvir2(-2) 1",mvir2(-2)
      ISRpx=misspx
      ISRpy=misspy
c
c     Apply the boost correction
c

c     First evaluated the total momentum in the LAB frame
      do j=0,3
      Ptot(j)=0d0
        do k=3,nexternal
          Ptot(j)=Ptot(j)+momenta(j,k)
        enddo
      pboost(j)=Ptot(j)
      enddo
 
c     Then calculate the momenta in the CMS frame
      pboost(1)=-pboost(1)
      pboost(2)=-pboost(2)
      pboost(3)=0d0
       do j=3,nexternal
c         write(*,*) "p",j,momenta(0,j), momenta(1,j),momenta(2,j),momenta(3,j)
         call boostx(momenta(0,j),pboost,CMS_mom(0,j))
       enddo
       call boostx(Ptot,pboost,PtotCMS)

c     Evaluate the initial momenta in the CMS frame
      x1=(PtotCMS(0)+PtotCMS(3))/sqrts
      x2=(PtotCMS(0)-PtotCMS(3))/sqrts

      if (dabs(x1-0.5).gt.0.5d0.or.dabs(x2-0.5).gt.0.5d0) then
        jac=-1d0
        write(*,*) "Warning: x1 or x2 larger than 1"
      endif

      CMS_mom(0,1)=sqrts*x1/2d0
      CMS_mom(1,1)=0d0
      CMS_mom(2,1)=0d0
      CMS_mom(3,1)=sqrts*x1/2d0
      CMS_mom(0,2)=sqrts*x2/2d0
      CMS_mom(1,2)=0d0
      CMS_mom(2,2)=0d0
      CMS_mom(3,2)=-sqrts*x2/2d0

c     Evaluate the initial momenta in the LAB frame
      pboost(1)=Ptot(1)
      pboost(2)=Ptot(2)
      call boostx(CMS_mom(0,1),pboost,momenta(0,1))
      call boostx(CMS_mom(0,2),pboost,momenta(0,2))


      measureLAB=1d0
       do j=3,nexternal-num_inv
         MG=inv_matching_type_part(j)
         measureLAB=measureLAB*dsqrt(momenta(1,MG)**2+momenta(2,MG)**2)
       enddo

      measureCMS=1d0
       do j=3,nexternal-num_inv
         MG=inv_matching_type_part(j)
         measureCMS=measureCMS*dsqrt(CMS_mom(1,MG)**2+CMS_mom(2,MG)**2)
       enddo

      jac=jac*measureCMS/measureLAB
c
c     flux factor
c
      jac_loc=jac_loc/(2d0*S*x1*x2)  ! flux 
      jac=jac*jac_loc

      return
      end
