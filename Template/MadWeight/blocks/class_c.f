      subroutine class_c(x,p1,p2,p3,r1,r2)
c
c     this subroutine catches the momenta p1 and p3 for the topology
c
c                     * 1 (nu)
c     r2     r1     *
c      -----------*
c        *          *  
c          *          * 2 (e)
c            * 3 (b)
c
c
c        *
c       * *
c      * ! *     m3=0 !!!
c     * * * *
c
c
c     A  variables: (rho3= sqrt[p1x^2+p1y^2+p1z^2]):
c      
c             
c      (miss_px,miss_py,m12sq,m123sq-m12sq)  <-->  (p1x,p1y,p1z,rho3)
c               ^                                       ^
c               |                                       |
c            var w in the following              var u in the following
c            
c     B the change of variables can be written
c 
c      w = A u + E1 C1 + alpha C2 + C3
c
c     with   Amat  =  a 4x4 matrix
c            C1 = (0,0,2E1,0)
c            C2 = (0,0,0,1)
c            C3 = (0,0,m1**2+m2**2, m3**2)
c            alpha = 2 rho3 (E1-sin (theta3) cos (theta3) p1x 
c                              -sin (theta3) sin (theta3) p1y )
c                              -cos (theta3)  p1z )
c

      implicit none 
      include '../../../SubProcesses/phasespace.inc'
c
c      arguments
c
      integer p1,p2,p3,r1,r2,n_var,local_var
c
c     parameters
c
      double precision x(20)
      double precision thres
      parameter (thres=1d0)
      double complex i_num
      parameter (i_num=(0d0,1d0))
c
c     local
c

      INTEGER IDUM
      DATA IDUM/0/
      SAVE IDUM
      double precision pboost(0:3), CMS_mom(0:3,max_particles)
      double precision Ptot(0:3),PtotCMS(0:3)
      double precision measureLAB, measureCMS
      double precision theta3,phi3,jac_temp,jac_factor
      double precision sintheta3,costheta3,sinphi3,cosphi3
      double precision rho3,Delta,jac_loc
      double precision p2x,p2y,p2z,E2,sqrts,chi
c
      double precision p1x_ti,p1x_E1, p1x_alpha
      double precision p1y_ti,p1y_E1, p1y_alpha
      double precision p1z_ti,p1z_E1, p1z_alpha
      double precision rho3_ti,rho3_E1, rho3_alpha
      double precision invA(4,4)
      real var
c
      double precision g22,g11,g12,g10,g20,g00
      double precision h22,h11,h12,h10,h20,h00
c
      double complex solc1(4),solc2(4)
      double precision  E1, alpha,i_part1,i_part2
      double precision solr1(4),solr2(4)
      integer j,k,MG
      logical sol(4)
      double precision p1x,p1y,p1z
c
      double precision inv_jac 
c
c     global
c
      double precision momenta(0:3,-max_branches:2*max_particles)  ! records the momenta of external/intermediate legs     (MG order)
      double precision mvir2(-max_branches:2*max_particles)        ! records the sq invariant masses of intermediate particles (MG order)
      common /to_diagram_kin/ momenta, mvir2
      double precision Etot,pztot,misspx,misspy
      common /to_missingP/Etot,pztot,misspx,misspy
      double precision              S,X1,X2,PSWGT,JAC
      common /PHASESPACE/ S,X1,X2,PSWGT,JAC
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
      REAL XRAN1
      EXTERNAL XRAN1
c
c Begin code
c---
       IDUM=0
c
c     energies are initialized to a negative real
c     (= no solution status)
c
      momenta(0,p1)=-1D0
      momenta(0,p3)=-1D0
      jac_loc=1d0
      sqrts=dsqrt(s)
      if((Etot+dsqrt(misspx**2+misspy**2)).gt.sqrts) then
      jac=-1d0
      return
      endif
c
c     generate theta3, phi3
      call  generate_variable(x,1, p3, theta3, jac_loc)
      call  generate_variable(x,2, p3, phi3, jac_loc)
c
c      theta3=dacos(momenta(3,p3)/
c     & (dsqrt(momenta(1,p3)**2+momenta(2,p3)**2+momenta(3,p3)**2)))
c      phi3=phi(momenta(0,p3))

c
c      initialisation
c
        sintheta3=dsin(theta3)
        costheta3=dcos(theta3)
        sinphi3=dsin(phi3)
        cosphi3=dcos(phi3)
c
        p2x=momenta(1,p2)
        p2y=momenta(2,p2)
        p2z=momenta(3,p2)
        E2=momenta(0,p2)
        Delta=2d0*(E2-sintheta3*cosphi3*p2x-sintheta3*sinphi3*p2y
     & -costheta3*p2z )
c
c       define invA
c
        invA(1,1)=1d0
        invA(1,2)=0d0
        invA(1,3)=0d0
        invA(1,4)=-sintheta3*cosphi3/Delta
c
        invA(2,1)=0d0
        invA(2,2)=1d0
        invA(2,3)=0d0
        invA(2,4)=-sintheta3*sinphi3/Delta
c
        invA(3,1)=-p2x/p2z
        invA(3,2)=-p2y/p2z
        invA(3,3)=-1d0/(2d0*p2z)
        invA(3,4)=(sintheta3*cosphi3*p2x+sintheta3*sinphi3*p2y)
     & /(Delta*p2z)
c
        invA(4,1)=0d0
        invA(4,2)=0d0
        invA(4,3)=0d0
        invA(4,4)=1/Delta
c
c       write
c
c       p1x = p1x_ti + p1x_E1 * E1 + p1x_alpha * alpha
c       p1y = p1y_ti + p1y_E1 * E1 + p1y_alpha * alpha
c       p1z = p1z_ti + p1z_E1 * E1 + p1z_alpha * alpha
c       rho3 = rho3_ti + rho3_E1 * E1 + rho3_alpha * alpha
c
      p1x_ti=misspx+invA(1,4)*(mvir2(r2)-mvir2(r1))
      p1y_ti=misspy+invA(2,4)*(mvir2(r2)-mvir2(r1))
      p1z_ti=invA(3,1)*misspx+invA(3,2)*misspy+
     & invA(3,3)*(mvir2(r1)-mvir2(p1)-mvir2(p2))+invA(3,4)
     & *(mvir2(r2)-mvir2(r1))
      rho3_ti=invA(4,4)*(mvir2(r2)-mvir2(r1))
c
      p1x_E1=0d0
      p1y_E1=0d0
      p1z_E1=-invA(3,3)*2d0*E2
      rho3_E1=0d0
c
      p1x_alpha=-invA(1,4)
      p1y_alpha=-invA(2,4)
      p1z_alpha=-invA(3,4)
      rho3_alpha=-invA(4,4)
c
c     define the coefficients of the two quadratic equations
c
c     g11 E1^2 + g22 alpha^2 + g12 E1 alpha + g10 E1 + g20 alpha +g00 =0   
c     h11 E1^2 + h22 alpha^2 + h12 E1 alpha + h10 E1 + h20 alpha +h00 =0   
c
      g11=1d0-p1x_E1**2-p1y_E1**2-p1z_E1**2
      g22=-p1x_alpha**2-p1y_alpha**2-p1z_alpha**2
      g12=-2d0*(p1x_E1*p1x_alpha+p1y_E1*p1y_alpha+p1z_E1*p1z_alpha)
      g10=-2d0*(p1x_E1*p1x_ti+p1y_E1*p1y_ti+p1z_E1*p1z_ti)
      g20=-2d0*(p1x_ti*p1x_alpha+p1y_ti*p1y_alpha+p1z_ti*p1z_alpha)
      g00=-p1x_ti**2-p1y_ti**2-p1z_ti**2-mvir2(p1)
c
      h11=0d0
      h22=-2d0*rho3_alpha*(sintheta3*cosphi3*p1x_alpha+
     & sintheta3*sinphi3*p1y_alpha+costheta3*p1z_alpha)
      h12=2d0*rho3_alpha*(1d0-costheta3*p1z_E1)
      h10=2d0*rho3_ti*(1d0-sintheta3*cosphi3*p1x_E1-
     & sintheta3*sinphi3*p1y_E1-costheta3*p1z_E1)
      h20=-1d0+2d0*rho3_ti*(-sintheta3*cosphi3*p1x_alpha-
     & sintheta3*sinphi3*p1y_alpha-costheta3*p1z_alpha)
     & +2d0*rho3_alpha*(-sintheta3*cosphi3*p1x_ti-
     & sintheta3*sinphi3*p1y_ti-costheta3*p1z_ti)
      h00=+2d0*rho3_ti*(-sintheta3*cosphi3*p1x_ti-
     & sintheta3*sinphi3*p1y_ti-costheta3*p1z_ti)


       call solve_2quadeqs(g11,g22,g12,g10,g20,g00,
     & h11,h22,h12,h10,h20,h00,solc1,solc2,sol)

c---------------------------------------------------------------------------------------------------
c     At this stage, we have got (max) four solutions for E1, alpha, but
c     some of them are complex. The following  loop over the 4 solutions
c     determines which sol is real.
c     In the two vectors (E1(1),E1(2),E1(3),E1(4)) and  (alpha(1),alpha(2),alpha(3),alpha(4))
c     the real solutions are set in first positions. The other component
c     are set to -1.
c---------------------------------------------------------------------------------------------------
c
      k=0
      do j=1,4
c
c     look if we have a real solution
c
      if (sol(j)) then
      i_part1=dble((solc1(j)-dconjg(solc1(j)))/(2d0*i_num))
      i_part2=dble((solc2(j)-dconjg(solc2(j)))/(2d0*i_num))

c      write(*,*) 'sol ',j
c      write(*,*) 'i_num ',i_num
c      write(*,*) 'dble ',dble(solc1(j))
c      pause
c      write(*,*) 'imag ',i_part1


      if (dabs(dble(solc1(j))).gt.1.0D+5*dabs(i_part1).and.
     & dabs(dble(solc2(j))).gt.1.0D+5*dabs(i_part2 ).and.
     &      dble(solc1(j)).gt.0d0) then

c      write(*,*) 'we have a real number'


c       determine the fraction of initial energies
c

      E1=dble(solc1(j))
      alpha=dble(solc2(j))

      p1x=p1x_ti+p1x_E1*E1+p1x_alpha*alpha
      p1y=p1y_ti+p1y_E1*E1+p1y_alpha*alpha
      p1z=p1z_ti+p1z_E1*E1+p1z_alpha*alpha
      rho3=rho3_ti+rho3_E1*E1+rho3_alpha*alpha

      x1=((Etot+E1+rho3)+(pztot+P1z+rho3*dcos(theta3)))/sqrts
      x2=((Etot+E1+rho3)-(pztot+P1z+rho3*dcos(theta3)))/sqrts

      if(dabs(x1-0.5d0).lt.0.5d0.and.dabs(x2-0.5d0).lt.0.5d0.and.
     &  rho3.gt.0d0)then
            k=k+1 ! count the number of good sol
            solr1(k)=E1
            solr2(k)=alpha
c            write(*,*) 'sol',k
c            write(*,*) solr1(k)
c            write(*,*) solr2(k)
       endif

       endif
       endif
       enddo

      if (k.eq.0) then 
        jac=-1d0
      return
      elseif (k.eq.1) then
        E1=solr1(1)
        alpha=solr2(1)
        jac_factor=1d0
      elseif (k.eq.2) then
        jac_factor=2d0
c        call ntuple(var,0.0,1.0,max_particles)
        var=xran1(IDUM)
        if(var.gt.0.5) then
          E1=solr1(1)
          alpha=solr2(1)
        else
          E1=solr1(2)
          alpha=solr2(2)
        endif
      elseif (k.eq.3) then
        jac_factor=3d0
c        call ntuple(var,0.0,1.0,max_particles)
        var=xran1(IDUM)
        if(var.lt.1.0/3.0) then
          E1=solr1(1)
          alpha=solr2(1)
        elseif(var.ge.1.0/3.0.and.var.lt.2.0/3.0) then
          E1=solr1(2)
          alpha=solr2(2)
        else
          E1=solr1(3)
          alpha=solr2(3)
        endif
      elseif (k.eq.4) then
        jac_factor=4d0
c        call ntuple(var,0.0,1.0,max_particles)
        var=xran1(IDUM)
        if(var.lt.1.0/4.0) then
          E1=solr1(1)
          alpha=solr2(1)
        elseif(var.ge.1.0/4.0.and.var.lt.1.0/2.0) then
          E1=solr1(2)
          alpha=solr2(2)
        elseif(var.ge.1.0/2.0.and.var.lt.3.0/4.0) then
          E1=solr1(3)
          alpha=solr2(3)
         else
          E1=solr1(4)
          alpha=solr2(4)
        endif
      endif
c      write(*,*) 'jac_factor',jac_factor
c
c     fix other components of neutrino momenta
c
      p1x=p1x_ti+p1x_E1*E1+p1x_alpha*alpha
      p1y=p1y_ti+p1y_E1*E1+p1y_alpha*alpha
      p1z=p1z_ti+p1z_E1*E1+p1z_alpha*alpha
      rho3=rho3_ti+rho3_E1*E1+rho3_alpha*alpha
c
      momenta(0,p1)=E1
      momenta(1,p1)=p1x
      momenta(2,p1)=p1y
      momenta(3,p1)=p1z
c
      momenta(0,p3)=rho3
      momenta(1,p3)=rho3*sintheta3*cosphi3
      momenta(2,p3)=rho3*sintheta3*sinphi3
      momenta(3,p3)=rho3*costheta3
c
c    fill initial momenta
c
      x1=((Etot+E1+rho3)+(pztot+P1z+rho3*dcos(theta3)))/sqrts
      x2=((Etot+E1+rho3)-(pztot+P1z+rho3*dcos(theta3)))/sqrts


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

c    fill intermediate momenta

      do j=0,3
      momenta(j,r1)=momenta(j,p1)+momenta(j,p2)
      momenta(j,r2)=momenta(j,r1)+momenta(j,p3)
      enddo

c
c       define jac of the transformation
c
c     (m12sq, m123sq, Etot, pztot, misspx, misspy, E1^2)
c                     
c           =>    (p1z,|p3|,x1,x2,p1x,p1y,E1)   -> [Jac_g^-1]=E^5
c          g^-1                                
c

c       A=-2d0*(p2x+rho3*dsin(theta3)*dcos(phi3))  ! dim E
c       B=-2d0*(p2y+rho3*dsin(theta3)*dsin(phi3))  ! dim E
c       C=-2d0*(p2z+rho3*dcos(theta3))             ! dim E
c       D=2d0*((E1+E2)-(p1x+p2x)*dsin(theta3)*dcos(phi3) ! dim E
c     & -(p1y+p2y)*dsin(theta3)*dsin(phi3)-(p1z+p2z)*dcos(theta3))
c       F=2d0*(E2+rho3)   ! dim E
       chi=2d0*(E1+E2-(p1x+p2x)*dcos(phi3)*dsin(theta3)-
     & (p1y+p2y)*dsin(phi3)*dsin(theta3)-
     & (p1z+p2z)*dcos(theta3))
 
c       inv_jac= -2d0*(D*(E2*p1z - E1*p2z) +
c     -     (cosphi3*(C*E2*p1x - A*E2*p1z - C*E1*p2x - F*p1z*p2x +
c     -           A*E1*p2z + F*p1x*p2z) +
c     -        (C*E2*p1y - B*E2*p1z - C*E1*p2y - F*p1z*p2y +
c     -           B*E1*p2z + F*p1y*p2z)*sinphi3)*sintheta3)*s

        inv_jac=   (4d0*chi*E2*p1z - 4d0*chi*E1*p2z - 
     -   8d0*dcos(phi3)*dcos(theta3)*E2*p1x*rho3*dsin(theta3) + 
     -   8d0*dcos(phi3)*dcos(theta3)*E1*p2x*rho3*dsin(theta3) - 
     -   8d0*dcos(phi3)*p1z*p2x*rho3*dsin(theta3) + 
     -   8d0*dcos(phi3)*p1x*p2z*rho3*dsin(theta3) - 
     -   8d0*dcos(theta3)*E2*p1y*rho3*dsin(phi3)*dsin(theta3) + 
     -   8d0*dcos(theta3)*E1*p2y*rho3*dsin(phi3)*dsin(theta3) - 
     -   8d0*p1z*p2y*rho3*dsin(phi3)*dsin(theta3) + 
     -   8d0*p1y*p2z*rho3*dsin(phi3)*dsin(theta3) + 
     -   8d0*dcos(phi3)**2*E2*p1z*rho3*dsin(theta3)**2 - 
     -   8d0*dcos(phi3)**2*E1*p2z*rho3*dsin(theta3)**2 + 
     -   8d0*E2*p1z*rho3*dsin(phi3)**2*dsin(theta3)**2 - 
     -   8d0*E1*p2z*rho3*dsin(phi3)**2*dsin(theta3)**2)*s/2d0


       if (dabs(inv_jac).gt.thres) then
c
       jac_loc=jac_loc*dsin(theta3)*rho3/(2d0*dabs(inv_jac))*jac_factor
       else
       write(*,*) 'warning: jac^-1 is close to zero'
       endif

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
      x1=(PtotCMS(0)+Ptot(3))/sqrts
      x2=(PtotCMS(0)-Ptot(3))/sqrts

      if (dabs(x1-0.5).gt.0.5d0.or.dabs(x2-0.5).gt.0.5d0) then
        jac=-1d0
c        write(*,*) "Warning: x1 or x2 larger than 1"
        momenta(0,1)=-1
        momenta(0,2)=-1
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
c     ONLY if isr = 2 

      if (isr_mode.eq.2) then
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

      jac=jac*jac_loc/(2d0*x1*x2*s)

      return
      end

