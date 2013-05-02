      subroutine block_b(x,p1,p2,p3,r1,r2)
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     This block corresponds to the reduced diagram
c
c              *P3    * P2
c            *       *
c     ***************
c     MV_r2  MV_r1   * 
c                     * |P1| (missing)
c
c     Pz1,PT1 are fixed to reproduce MV_r1 and MV_r2; phi_1 is generated randomly
c
c
c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      implicit none
      include '../../../SubProcesses/phasespace.inc'
c
c     argument
c
      integer p1,p2,p3,r1,r2,n_var,config, local_var
c
c     local
c
      INTEGER IDUM
      DATA IDUM/0/
      SAVE IDUM
      double precision b,c,rho,jac_factor,dem
      double precision pt1_ti,pt1_E1,pz1_ti,pz1_E1,PT1
      double precision  phi_miss,jac_temp,invjac
      real rand
      double precision pz2,pz3,pt2,pt3,E2,E3,cphi13,cphi12
      double precision Ma(2),Mc(2),MinvB(2,2),detB
      double precision sol(2),PT1sol(2),jac_loc
      double precision x(20)
      integer nu,index_sol
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
c
c     external
c
      double precision pt, phi,dot
      external pt,phi,dot
      REAL XRAN1
      EXTERNAL XRAN1
c---
c Begin code
c---
c
c     First generate  phi of the missing particle
c
      jac_loc = 1d0
      call generate_variable(x,2,p1, phi_miss, jac_loc)
c      write(*,*) 'phi, x,nvar',phi_miss,x(n_var),n_var
c
      cphi13=dcos(phi_miss-phi(momenta(0,p3)))
      cphi12=dcos(phi_miss-phi(momenta(0,p2)))
      pz2=momenta(3,p2)
      pz3=momenta(3,p3)
      pt2=pt(momenta(0,p2))
      pt3=pt(momenta(0,p3))
      E2=momenta(0,p2)
      E3=momenta(0,p3)

c     Express the change of variables as
c       | a1 | = | b11 b12 | | P1T | + | C1 | E1
c       | a2 |   | b21 b22 | | P1Z |   | C2 |
c
      Ma(1)=mvir2(r1)-mvir2(p1)-mvir2(p2)
      Ma(2)=mvir2(r2)-mvir2(r1)-mvir2(p3)-
     & 2d0*dot(momenta(0,p2),momenta(0,p3))
c
      detB=-4d0*cphi13*pz2*pt3+4d0*cphi12*pz3*pt2
c
      MinvB(1,1)=-2d0*pz3/detB
      MinvB(1,2)=2d0*pz2/detB
      MinvB(2,1)=2d0*cphi13*pt3/detB
      MinvB(2,2)=-2d0*cphi12*pt2/detB
c
      Mc(1)=2d0*momenta(0,p2)
      Mc(2)=2d0*momenta(0,p3)
c
      PT1_ti=MinvB(1,1)*Ma(1)+MinvB(1,2)*Ma(2)
      Pz1_ti=MinvB(2,1)*Ma(1)+MinvB(2,2)*Ma(2)
      PT1_E1=-MinvB(1,1)*Mc(1)-MinvB(1,2)*Mc(2)
      Pz1_E1=-MinvB(2,1)*Mc(1)-MinvB(2,2)*Mc(2)
c    
c     solve the mass shell equation 
c    
c     E1^2-|P1|^2=m_1^2   <=>   E1^2-b*E1+c=0 
c
      jac_factor=1d0
      dem=1d0-PT1_E1**2-Pz1_E1**2

      if (dabs(dem).lt.0.0001d0) then
        momenta(0,p1)=-(Pz1_ti**2+PT1_ti**2+mvir2(p1))/
     &  (2d0*PT1_ti*PT1_E1+2d0*Pz1_ti*Pz1_E1)
        PT1=PT1_ti+PT1_E1*momenta(0,p1)
        if  (momenta(0,p1).le.0d0.or.PT1.le.0d0) then
          jac=-1d0
          return
        endif
      else
        b=-(2d0*PT1_ti*PT1_E1+2d0*Pz1_ti*Pz1_E1)/dem
        c=-(Pz1_ti**2+PT1_ti**2+mvir2(p1))/dem
        rho=b**2-4d0*c

        if (rho.eq.0d0.and.b.lt.0d0) then   ! max 1 sol
          momenta(0,p1)=-b/2d0
          PT1=PT1_ti+PT1_E1*momenta(0,p1) 
          if (PT1.le.0d0) then
            jac=-1d0
            momenta(0,p1)=-1d0
            return
          endif
        elseif (rho.gt.b**2)then ! max 1 sol
          momenta(0,p1)=(-b+dsqrt(rho))/2d0
          PT1=PT1_ti+PT1_E1*momenta(0,p1)
          if (PT1.le.0d0) then
            jac=-1d0
            momenta(0,p1)=-1d0
            return
          endif
        elseif (rho.gt.0d0.and.dsqrt(rho).lt.(-b)) then   ! max 2 sol
          sol(1)=(-b+dsqrt(rho))/2d0
          sol(2)=(-b-dsqrt(rho))/2d0
          PT1sol(1)=PT1_ti+PT1_E1*sol(1)
          PT1sol(2)=PT1_ti+PT1_E1*sol(2)
          if (PT1sol(1).gt.0d0.and.PT1sol(2).gt.0d0) then
            index_sol=1
c            call ntuple(rand,0.0,1.0,p1)
             rand=xran1(IDUM)
            if (rand.gt.0.5) index_sol=2
            momenta(0,p1)=sol(index_sol)
            PT1=PT1sol(index_sol)
            jac_factor=2d0
          elseif (PT1sol(1).gt.0d0.and.PT1sol(2).le.0d0) then
            momenta(0,p1)=sol(1)
            PT1=PT1sol(1)
          elseif (PT1sol(2).gt.0d0.and.PT1sol(1).le.0d0) then
            momenta(0,p1)=sol(2)
            PT1=PT1sol(2)
          else
            jac=-1d0
            momenta(0,p1)=-1d0
            return
          endif
        else                                ! 0 sol
          momenta(0,p1)=-1d0
          jac=-1d0
          return 
        endif
      endif
c
c     fill the momentum p1
c
      momenta(1,p1)=PT1*dcos(phi_miss)
      momenta(2,p1)=PT1*dsin(phi_miss)
      momenta(3,p1)=pz1_ti+pz1_E1*momenta(0,p1)
c
c     fill momenta r1,r2
c
      do nu=0,3
        momenta(nu,r1)=momenta(nu,p1)+momenta(nu,p2)
        momenta(nu,r2)=momenta(nu,r1)+momenta(nu,p3)
      enddo
c
c     compute missing momentum
c
       misspx=misspx-momenta(1,p1)
       misspy=misspy-momenta(2,p1)
       Etot=Etot+momenta(0,p1)
       pztot=pztot+momenta(3,p1)
c
c     compute the jacobian:
c
      jac_loc=jac_loc*PT1       ! PS weight for the initial set of variable
c
      invjac=-8d0*cphi12*E3*pt2*momenta(3,p1)
     & +8d0*cphi13*E2*pt3*momenta(3,p1) 
     & +8d0*E3*pt1*pz2 - 8d0*cphi13*momenta(0,p1)*pt3*pz2 
     & -8d0*E2*pt1*pz3 + 8d0*cphi12*momenta(0,p1)*pt2*pz3      
         !  the factor 2*E1 has been removed since it factorizes with the factor 1/2E1 in the PS weight 

      jac_loc=jac_loc*jac_factor/dabs(invjac)   
      jac=jac*jac_loc   

      return
      end

