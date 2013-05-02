      subroutine block_c(x,p1,p2,r1)
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     This block corresponds to the reduced diagram
c
c               * P2
c              *
c     *********
c      MV_r1   * 
c               * |P1| (missing)
c
c     |P_1| is fixed to reproduce MV_r1, theta_1 and phi_1 are generated randomly
c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      implicit none
      include '../../../SubProcesses/phasespace.inc'
c
c     argument
c
      integer p1,p2,r1,n_var, local_var, config
c
c     local
c
      double precision x(20)
      INTEGER IDUM
      DATA IDUM/0/
      SAVE IDUM
      double precision b,c,rho,jac_factor,normP1
      double precision p1_ti,p1_E1,dem,jac_loc
      double precision theta_miss, phi_miss,jac_temp
      real rand
      integer nu,index_sol
      double precision sol(2),p1sol(2)
      double precision phi
      external phi
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
      REAL XRAN1
      EXTERNAL XRAN1
c---
c Begin code
c---
c
c     First generate theta and phi of the missing particle
c
      jac_loc =1d0
      call  generate_variable(x,1,p1, theta_miss, jac_loc)
      call  generate_variable(x,2,p1, phi_miss, jac_loc)

c      write(*,*) 'phi, x,nvar',phi_miss,x(n_var),n_var
c
c      theta_miss=dacos(momenta(3,p1)/dsqrt(momenta(1,p1)**2
c     & +momenta(2,p1)**2+momenta(3,p1)**2)) 
c      phi_miss=phi(momenta(0,p1))
      momenta(1,p1)=dsin(theta_miss)*dcos(phi_miss)
      momenta(2,p1)=dsin(theta_miss)*dsin(phi_miss)
      momenta(3,p1)=dcos(theta_miss)
c
c     Express |P_1| in term of E_1:  |P_1|=p1_ti+p1_E1*E_1
c
      dem=2d0*(momenta(1,p1)*momenta(1,p2)+momenta(2,p1)*momenta(2,p2)
     & +momenta(3,p1)*momenta(3,p2))
      p1_ti=(mvir2(p1)+mvir2(p2)-mvir2(r1))/dem
      p1_E1=2d0*momenta(0,p2)/dem
c
c     solve the equation 
c     E1^2-|P1|^2=m_1^2   <=>   (1-p1_E1^2)*E1^2-2*p1_ti*p1_E1*E1-p1_ti^2-m1^2=0 
c

          jac_factor=1d0
            index_sol=1

      if (dabs(p1_E1**2-1d0).eq.0.0001d0) then
        momenta(0,p1)=-(p1_ti**2+mvir2(p1))/2d0/p1_ti/p1_E1
        normP1=p1_ti+p1_E1*momenta(0,p1)
        if  (momenta(0,p1).le.0d0.or.normP1.le.0d0) then
          momenta(0,p1)=-1d0
          jac_loc=-1d0
          jac=-1d0
          return
        endif
      else
        b=-2d0*p1_ti*p1_E1/(1d0-p1_E1**2)
        c=-(p1_ti**2+mvir2(p1))/(1d0-p1_E1**2)
        rho=b**2-4d0*c
        if (rho.eq.0d0.and.b.lt.0d0) then   ! max 1 sol
          momenta(0,p1)=-b/2d0
          normP1=p1_ti+p1_E1*momenta(0,p1)
          if  (normP1.le.0d0) then
            momenta(0,p1)=-1d0
            jac_loc=-1d0
            jac=-1d0
            return
          endif
        elseif (rho.gt.b**2)then ! max 1 sol
          momenta(0,p1)=(-b+dsqrt(rho))/2d0
          normP1=p1_ti+p1_E1*momenta(0,p1)
          if  (normP1.le.0d0) then
            momenta(0,p1)=-1d0
            jac_loc=-1d0
            jac=-1d0
            return
          endif
        elseif (rho.gt.0d0.and.dsqrt(rho).lt.(-b)) then   ! max 2 sol for E1
          sol(1)=(-b+dsqrt(rho))/2d0
          sol(2)=(-b-dsqrt(rho))/2d0
          p1sol(1)=p1_ti+p1_E1*sol(1)
          p1sol(2)=p1_ti+p1_E1*sol(2)
          if (p1sol(1).gt.0d0.and.p1sol(2).gt.0d0) then
            index_sol=1
c            call ntuple(rand,0.0,1.0,p1)
             rand=xran1(IDUM)
c              write(*,*) 'rand',rand
            if (rand.gt.0.5) then
              index_sol=2
c              write(*,*) 'hello'
            endif
c            if (rand.gt.0.5) then
c              jac_loc=-1d0
c              jac=-1d0
c            endif
              momenta(0,p1)=sol(index_sol)
              normP1=p1sol(index_sol)
              jac_factor=2d0
          elseif (p1sol(1).gt.0d0.and.p1sol(2).le.0d0) then  
              momenta(0,p1)=sol(1)
              normP1=p1sol(1)
          elseif (p1sol(1).le.0d0.and.p1sol(2).gt.0d0) then  
              momenta(0,p1)=sol(2)
              normP1=p1sol(2)
          else 
            momenta(0,p1)=-1d0
            jac_loc=-1d0
            jac=-1d0
            return
          endif
        else                                                 ! 0 sol
          momenta(0,p1)=-1d0
          jac_loc=-1d0
          jac=-1d0
          return 
        endif
      endif

c      if (index_sol.eq.1) then
c          jac_loc=-1d0
c          jac=-1d0
c          return
c      endif

      

c
c     fill the momentum p1,r1
c
      momenta(0,r1)=momenta(0,p1)+momenta(0,p2)
      do nu=1,3
           momenta(nu,p1)=normP1*momenta(nu,p1)
           momenta(nu,r1)=momenta(nu,p1)+momenta(nu,p2)
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
      jac_loc=jac_loc*dsin(theta_miss)*normP1**2       ! PS weight for the initial set of variable
c
      jac_loc=jac_loc*jac_factor
     &/dabs(4d0*normP1*momenta(0,p2)-2d0*momenta(0,p1)*dem)  ! jac for the change of variables 
                         !  the factor 2*E1 has been removed since it factorizes with the factor 1/2E1 in the PS weight 

      jac=jac*jac_loc

      return
      end

