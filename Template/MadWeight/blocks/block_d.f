      subroutine block_d(x,p1,p2,r1)
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     This block corresponds to the reduced diagram
c
c               * P2 (visible)
c              *
c     *********
c      MV_r1   * 
c               * P1 (visible)
c
c     |P1| will be left unfactorized, MV_r1 is aligned.
c     The choice is based on the width of the transfer function (done in the init part)
c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      implicit none
      include '../../../SubProcesses/phasespace.inc'
c
c     argument
c
c      integer parg1,parg2 !old variable
      integer p1,p2,r1,n_var, local_var
c
c     local
c
      double precision x(20)
      INTEGER IDUM
      DATA IDUM/0/
      SAVE IDUM
      double precision b,c,rho,jac_factor,normP1new
      double precision p1_ti,p1_E1,dem,jac_temp,jac_loc
      real rand
c      integer p1,p2
	  integer nu,index_sol,j
      double precision sol(2),p1sol(2)
      double precision gen_var(3)
      double precision dot
      external dot
c
c     global
c
      double precision momenta(0:3,-max_branches:2*max_particles)  ! records the momenta of external/intermediate legs     (MG order)
      double precision mvir2(-max_branches:2*max_particles)        ! records the sq invariant masses of intermediate particles (MG order)
      common /to_diagram_kin/ momenta, mvir2
      double precision pmass(max_particles)     ! records the  mass of external particles   (MG order)
      common / to_mass/pmass
      double precision Etot,pztot,misspx,misspy
      common /to_missingP/Etot,pztot,misspx,misspy
      double precision              S,X1,X2,PSWGT,JAC
      common /PHASESPACE/ S,X1,X2,PSWGT,JAC
      REAL XRAN1
      EXTERNAL XRAN1
c---
c Begin code
c---
      jac_loc=1d0
c
c     First decide which |p| is left unfactorized.
c
cccccccccccccccccccc NOW this is done in the init step p2 is always factorized  
c
c      if (p1.gt.0.and.p2.gt.0) then
c        if (c_point(p1,3,2).gt.c_point(p2,3,2)) then
c          p1=parg1
c          p2=parg2
c        else
c          p1=parg2
c          p2=parg1
c        endif
c      elseif (parg1.gt.0.and.parg2.lt.0) then
c        p1=parg1
c        p2=parg2
c      elseif (parg1.lt.0.and.parg2.gt.0) then
c        p1=parg2
c        p2=parg1
c      else 
c        write(*,*) 'Warning: wrong phase space parametrization'
c        stop
c      endif
c      write(*,*) 'p1,p2',p1,p2
c
CCCCCCCCCCCCCCCCCCCCCCC END of modif


c     Now |p1| will be left unfactorized, |p2| will be factorized 
c
c----------------------------------------------
c     generate  p2
c----------------------------------------------
      if (p2.gt.0) then
      do j=1,3
        call  generate_variable(x,j,p2, gen_var(j),jac_loc)
      enddo

c            write(*,*) "C_point(p2,3,2)",C_point(p2,3,2)
c            write(*,*) gen_var(1),gen_var(2),gen_var(3),
c     &    pmass(p2)
        call four_momentum(gen_var(1),gen_var(2),gen_var(3),
     &    pmass(p2),momenta(0,p2))
c       write(*,*) "mom p2", (momenta(j,p2),j=0,3)
c       pause
c----------------------------------------------
c     update missing transverse momentum
c----------------------------------------------

        misspx=misspx-momenta(1,p2)
        misspy=misspy-momenta(2,p2)

c----------------------------------------------
c     update Etot and pztot for visible particles
c----------------------------------------------
        Etot=Etot+momenta(0,p2)
        pztot=pztot+momenta(3,p2)

c----------------------------------------------
c     update jacobian
c----------------------------------------------
      jac_loc=jac_loc
     & *gen_var(3)**2*dsin(gen_var(1))/(2d0*momenta(0,p2))
      endif

c      write(*,*) 'jac_loc ', jac_factor
c----------------------------------------------
c     generate angles associated to p1
c----------------------------------------------
      do j=1,2
         call  generate_variable(x,j,p1, gen_var(j),jac_loc)
      enddo

        call four_momentum(gen_var(1),gen_var(2),1d0,
     &    pmass(p1),momenta(0,p1))

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
c      write(*,*) 'jac_factor ', jac_factor
      if (dabs(p1_E1**2-1d0).le.0.0001d0) then
        momenta(0,p1)=-(p1_ti**2+mvir2(p1))/2d0/p1_ti/p1_E1
        normP1new=p1_ti+p1_E1*momenta(0,p1)
        if  (momenta(0,p1).le.0d0.or.normP1new.le.0d0) then
          momenta(0,p1)=-1d0
          jac=-1d0
          return
        endif
      else
        b=-2d0*p1_ti*p1_E1/(1d0-p1_E1**2)
        c=-(p1_ti**2+mvir2(p1))/(1d0-p1_E1**2)
        rho=b**2-4d0*c


        if (rho.eq.0d0.and.b.lt.0d0) then   ! max 1 sol
          momenta(0,p1)=-b/2d0
          normP1new=p1_ti+p1_E1*momenta(0,p1)
          if  (normP1new.le.0d0) then
            momenta(0,p1)=-1d0
            jac=-1d0
            return
          endif
        elseif (rho.gt.b**2)then ! max 1 sol
          momenta(0,p1)=(-b+dsqrt(rho))/2d0
          normP1new=p1_ti+p1_E1*momenta(0,p1)
          if  (normP1new.le.0d0) then
            momenta(0,p1)=-1d0
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
            if (rand.gt.0.5) index_sol=2
            momenta(0,p1)=sol(index_sol)
            normP1new=p1sol(index_sol)
            jac_factor=2d0
          elseif (p1sol(1).gt.0d0.and.p1sol(2).le.0d0) then  
              momenta(0,p1)=sol(1)
              normP1new=p1sol(1)
          elseif (p1sol(1).le.0d0.and.p1sol(2).gt.0d0) then  
              momenta(0,p1)=sol(2)
              normP1new=p1sol(2)
          else 
            momenta(0,p1)=-1d0
            jac=-1d0
          endif
        else                                                 ! 0 sol
          momenta(0,p1)=-1d0
          jac=-1d0
          return 
        endif
      endif

c
c     fill the momentum p1,r1
c
      momenta(0,r1)=momenta(0,p1)+momenta(0,p2)
      do nu=1,3
           momenta(nu,p1)=normP1new*momenta(nu,p1)
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

      jac_loc=jac_loc*dsin(gen_var(1))*normP1new**2
c      write(*,*) 'jac ', jac_loc
c
      jac_loc=jac_loc
     & /dabs(4d0*normP1new*momenta(0,p2)-2d0*momenta(0,p1)*dem)  ! jac for the change of variables 
     & *jac_factor      !  the factor 2*E1 has been removed since it factorizes with the factor 1/2E1 in the PS weight 

      jac=jac*jac_loc

c      write(*,*) 'jac_factor ', jac_factor
      return
      end

