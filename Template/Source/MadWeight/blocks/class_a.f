      subroutine class_a(x,n_var,p1,p2)
c***************************************************************************
c     ECS in CLASS A
c  
c
c***************************************************************************
      implicit none
      include '../../../SubProcesses/phasespace.inc'
c
c     arguments
c      
      double precision x(20)
      integer p1,p2,n_var
c
c     local
c
      double precision normp1,normp2,jac_temp,det,sqrts
      double precision angles(2,2),px(2),py(2),jac_loc
      integer j
c
c     global
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
      double precision c_point(1:max_particles,3,2)
      common/ph_sp_init/c_point
c
c     external
c
      double precision phi
      external phi
c---
c Begin code
c---
      jac_loc=1d0
      sqrts=dsqrt(s)
      if((Etot+dsqrt(misspx**2+misspy**2)).gt.sqrts) then
      jac=-1d0
      return
      endif

      do j=1,2
c
c     generate angles associated to p1
         if(c_point(p1,j,2).eq.0d0) then
            angles(1,j)=c_point(p1,j,1)
         elseif(c_point(p1,j,2).gt.0d0) then
            n_var=n_var+1     ! update the component of random variable
            call get_component(c_point(p1,j,1),c_point(p1,j,2),x(n_var),
     &                          angles(1,j),jac_temp,j,S)
            jac_loc=jac_loc*jac_temp
         endif
c
c     generate angles associated to p2
         if(c_point(p2,j,2).eq.0d0) then
            angles(2,j)=c_point(p2,j,1)
         elseif(c_point(p2,j,2).gt.0d0) then
            n_var=n_var+1     ! update the component of random variable
            call get_component(c_point(p2,j,1),c_point(p2,j,2),x(n_var),
     &                          angles(2,j),jac_temp,j,S)
            jac_loc=jac_loc*jac_temp
         endif
c
      enddo


c      angles(1,1)=dacos(momenta(3,p1)/
c     & (dsqrt(momenta(1,p1)**2+momenta(2,p1)**2+momenta(3,p1)**2)))
c      angles(2,1)=dacos(momenta(3,p2)/
c     & (dsqrt(momenta(1,p2)**2+momenta(2,p2)**2+momenta(3,p2)**2)))
c      angles(1,2)=phi(momenta(0,p1))
c      angles(2,2)=phi(momenta(0,p2))


c-------------------------------------------------------------------------
c    determine the momentum of the 2 particle fixed by PT conservation
c-------------------------------------------------------------------------
c
c     we have to resolve Px,Py momentum conservation
c         px(1)|p1|+px(2)|p2|-px_miss=0
c         py(1)|p1|+py(2)|p2|-py_miss=0
c
c         px(1)=cos(phi_1)*sin(theta_1)
c         px(2)=cos(phi_2)*sin(theta_2)
c         py(1)=sin(phi_1)*sin(theta_1)
c         py(2)=sin(phi_2)*sin(theta_2)

        px(1)=dcos(angles(1,2))*dsin(angles(1,1))
        px(2)=dcos(angles(2,2))*dsin(angles(2,1))
        py(1)=dsin(angles(1,2))*dsin(angles(1,1))
        py(2)=dsin(angles(2,2))*dsin(angles(2,1))

c        write(*,*) 'px', px(1),px(2)
c        write(*,*) 'py', py(1),py(2)

        det=px(1)*py(2)-py(1)*px(2)
        normp1=(misspx*py(2)-misspy*px(2))/det
        normp2=(misspy*px(1)-misspx*py(1))/det

c        write(*,*) 'norm 1,2', normp1,normp2
c        pause

        if(normp1.le.0d0.or.normp2.le.0d0.or.dabs(det).lt.1d-6) then
        jac=-1d0
        momenta(0,p1)=-1
        momenta(0,p2)=-1
        return
        endif


         
        call four_momentum(angles(1,1),angles(1,2),normp1,
     &    pmass(p1),momenta(0,p1))
        call four_momentum(angles(2,1),angles(2,2),normp2,
     &    pmass(p2),momenta(0,p2))
        jac_loc=jac_loc/dabs(det)


c
c    fill initial momenta
c
      x1=((Etot+momenta(0,p1)+momenta(0,p2))
     & +(pztot+momenta(3,p1)+momenta(3,p2)))/sqrts
      x2=((Etot+momenta(0,p1)+momenta(0,p2))-
     & (pztot+momenta(3,p1)+momenta(3,p2)))/sqrts


        if(dabs(x1-0.5d0).gt.0.5d0.or.dabs(x2-0.5d0).gt.0.5d0) then
        jac=-1d0
        momenta(0,p1)=-1
        momenta(0,p2)=-1
        return
        endif

      jac_loc=jac_loc*normp1**2*dsin(angles(1,1))/(2d0*momenta(0,p1))
      jac_loc=jac_loc*normp2**2*dsin(angles(2,1))/(2d0*momenta(0,p2))

      momenta(0,1)=sqrts*x1/2d0
      momenta(1,1)=0d0
      momenta(2,1)=0d0
      momenta(3,1)=sqrts*x1/2d0
      momenta(0,2)=sqrts*x2/2d0
      momenta(1,2)=0d0
      momenta(2,2)=0d0
      momenta(3,2)=-sqrts*x2/2d0
      misspx=0d0
      misspy=0d0
c
c     flux factor
c
      jac_loc=jac_loc/(S**2*x1*x2) ! flux + jac x1,x2 -> Etot, Pztot
      jac=jac*jac_loc

      return
      end
