      subroutine class_a(x,p1,p2)
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
      integer p1,p2,n_var,local_var
c
c     local
c
      double precision pboost(0:3), CMS_mom(0:3,max_particles)
      double precision Ptot(0:3),PtotCMS(0:3)
      double precision measureLAB, measureCMS
      double precision normp1,normp2,jac_temp,det,sqrts
      double precision angles(2,2),px(2),py(2),jac_loc
      integer j,MG,k
c
c     global
c
      double precision x(20)
      double precision momenta(0:3,-max_branches:2*max_particles)  ! records the momenta of external/intermediate legs     (MG order)
      double precision mvir2(-max_branches:2*max_particles)        ! records the sq invariant masses of intermediate particles (MG order)
      common /to_diagram_kin/ momenta, mvir2
      double precision pmass(max_particles)     ! records the pole mass of any particle of the diagram  (MG order)
      common / to_mass/pmass
      double precision Etot,pztot,misspx,misspy
      common /to_missingP/Etot,pztot,misspx,misspy
      double precision              S,X1,X2,PSWGT,JAC
      common /PHASESPACE/ S,X1,X2,PSWGT,JAC
      double precision pxISR, pyISR
      common /to_ISR/  pxISR, pyISR
      integer matching_type_part(3:max_particles)
      integer inv_matching_type_part(3:max_particles)
      common/madgraph_order_type/matching_type_part,
     & inv_matching_type_part
      integer nexternal, num_inv
      COMMON/to_num_inv/nexternal, num_inv
      integer ISR_mode
      common /to_correct_ISR/ISR_mode
c
c     external
c
      double precision phi
      external phi
c---
c Begin code
c---

c if pTmiss reconstructed is NOT used, 
c the boost is defined by the pT balancing the visible particles
c in class_h 
       if (ISR_mode.eq.3) then
         call class_h(x,p1,p2)
         return
       endif

c otherwise, boost the event based on ISR. 
c
      jac_loc=1d0
      sqrts=dsqrt(s)
      if((Etot+dsqrt(misspx**2+misspy**2)).gt.sqrts) then
      jac=-1d0
      return
      endif

      do j=1,2
c       generate angles associated to p1
        call  generate_variable(x,j,p1, angles(1,j), jac_loc)
c       generate angles associated to p2
        call  generate_variable(x,j,p2, angles(2,j), jac_loc)
      enddo

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

      jac_loc=jac_loc*normp1**2*dsin(angles(1,1))/(2d0*momenta(0,p1))
      jac_loc=jac_loc*normp2**2*dsin(angles(2,1))/(2d0*momenta(0,p2))

C Apply the boost correction 
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
        momenta(0,p1)=-1
        momenta(0,p2)=-1
        return
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

c     correction from the measure to translate the weight to the CM frame
c     ONLY if isr != 1 

      if (isr_mode.ne.1) then
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
      endif
c
c     flux factor
c
      jac_loc=jac_loc/(S**2*x1*x2)  ! flux + jac x1,x2 -> Etot, Pztot
      jac=jac*jac_loc

      misspx=0d0
      misspy=0d0

      return
      end
