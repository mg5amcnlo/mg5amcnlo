      subroutine block_e(x,p1,p2,p3,r1,r2)
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     This block corresponds to the reduced diagram
c
c            *P3(blob) * P2 (visible) m2=0 !!!
c           *         *
c        *************
c        r2      r1   * 
c                      * P1 (visible)
c
c     |P1| and |P2| must be left unfactorized, MV_r1, MV_r2 are aligned.
c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      implicit none
      include '../../../SubProcesses/phasespace.inc'
c
c     parameters
c
c      double precision thres
c      parameter (thres=1000d0)
      double complex i_num
      parameter (i_num=(0d0,1d0))
c
c     argument
c
      double precision x(20)
      integer p1,p2,p3,r1,r2,n_var,local_var
c
c     local
c
      INTEGER IDUM
      DATA IDUM/0/
      SAVE IDUM
      double precision b,c,rho,jac_factor,normp1,normp2,E3
      double precision p1_ti,p1_p2,jac_temp,jac_loc,inv_jac
      double complex E1c(4), E2c(4)
      logical checksol(4)
      real rand
      integer nu,index_sol,j,k
      double precision sol(2),p1sol(2),f12,f13,f23
      double precision gen_var(3),delta_ma2,delta_mb2
      double precision g11,g22,g12,g10,g20,g00,normp1real(4)
      double precision h11,h22,h12,h10,h20,h00,normp2real(4)
      double precision i_part1,i_part2
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
      REAL XRAN1
      EXTERNAL XRAN1
c---
c Begin code
c---

      jac_loc=1d0

c----------------------------------------------
c     generate  p1 angles
c----------------------------------------------
      do j=1,2
        call  generate_variable(x,j,p1, gen_var(j), jac_loc)
      enddo
c            write(*,*) "C_point(p1,3,2)",C_point(p1,3,2)
        call four_momentum(gen_var(1),gen_var(2),1d0,
     &    pmass(p1),momenta(0,p1))
c
c----------------------------------------------
c     update jacobian
c----------------------------------------------
      jac_loc=jac_loc*dsin(gen_var(1))
c
c----------------------------------------------
c     generate  p2
c----------------------------------------------
      do j=1,2
        call  generate_variable(x,j,p2, gen_var(j), jac_loc)
      enddo

c            write(*,*) "C_point(p2,3,2)",C_point(p2,3,2)
        call four_momentum(gen_var(1),gen_var(2),1d0,
     &    0d0,momenta(0,p2))
c       write(*,*) "mom p2", (momenta(j,p2),j=0,3)

c----------------------------------------------
c     update jacobian
c----------------------------------------------
      jac_loc=jac_loc*dsin(gen_var(1))

c----------------------------------------------
c     usefull definitions for change of var.
c----------------------------------------------

      delta_ma2=mvir2(r1)-mvir2(p1)
      delta_mb2=mvir2(r2)-mvir2(r1)-mvir2(p3)
c
      f13=momenta(1,p1)*momenta(1,p3)+momenta(2,p1)*momenta(2,p3)+
     & momenta(3,p1)*momenta(3,p3)
      f23=momenta(1,p2)*momenta(1,p3)+momenta(2,p2)*momenta(2,p3)+
     & momenta(3,p2)*momenta(3,p3)
      f12=momenta(1,p1)*momenta(1,p2)+momenta(2,p1)*momenta(2,p2)+
     & momenta(3,p1)*momenta(3,p2)
      E3=momenta(0,p3)


      if(pmass(p1).gt.0d0) then
c
c    the two quardratic equations can be written as
c
c      g11 p1^2 + g22 p2^2 + g12 p1*p2 + g10 p1 +g20 p2 +g00 =0
c      h11 p1^2 + h22 p2^2 + h12 p1*p2 + h10 p1 +h20 p2 +h00 =0

      g11=0d0
      g22=f23/E3-1d0
      g12=f13/E3-f12
      g10=0d0
      g20=delta_mb2/(2d0*E3)
      g00=-delta_ma2/2d0
c
      h11=f13**2/E3**2-1d0
      h22=f23**2/E3**2-2d0*f23/E3+1d0
      h12=2d0*(f13*f23/E3**2-f13/E3)
      h10=f13*delta_mb2/E3**2
      h20=(f23/E3-1d0)*delta_mb2/E3
      h00=delta_mb2**2/4d0/E3**2-mvir2(p1)

       call solve_2quadeqs(g11,g22,g12,g10,g20,g00,
     & h11,h22,h12,h10,h20,h00,E1c,E2c,checksol)

c        write(*,*) 'root 1'
c        write(*,*) E1c(1)
c       write(*,*) E2c(1)
c       write(*,*) 'root 2'
c       write(*,*) E1c(2)
c      write(*,*) E2c(2)
c       write(*,*) 'root 3'
c       write(*,*) E1c(3)
c       write(*,*) E2c(3)
c      write(*,*) 'root 4'
c      write(*,*) E1c(4)
c      write(*,*) E2c(4)
c       write(*,*) 'verif  eq 1 '
c       do j=1,4
c       write(*,*) g11*E1c(j)**2+g22*E2c(j)**2+g12*E1c(j)*E2c(j)
c     & +g10*E1c(j)+g20*E2c(j)+g00
c       enddo
c       write(*,*) 'verif  eq 2 '
c       do j=1,4
c       write(*,*) h11*E1c(j)**2+h22*E2c(j)**2+h12*E1c(j)*E2c(j)
c     & +h10*E1c(j)+h20*E2c(j)+h00
c       enddo


c---------------------------------------------------------------------------------------------------     
c     At this stage, we have got (max) four solution for E1, E2, but
c     some of them are complex. The following  loop over the 4 solutions
c     determines which sol is real.
c     In the two vectors (E1(1),E1(2),E1(3),E1(4)) and  (E2(1),E2(2),E2(3),E2(4))
c     the real solutions are set in first positions. The other component
c     are set to -1.  
c---------------------------------------------------------------------------------------------------     
c
      k=0
      do j=1,4
c
c     look if we have a real solution
c
        if (checksol(j)) then
          i_part1=dble((E1c(j)-dconjg(E1c(j)))/(2d0*i_num))
          i_part2=dble((E2c(j)-dconjg(E2c(j)))/(2d0*i_num))

c      write(*,*) 'sol ',j
c      write(*,*) 'i_num ',i_num
c      write(*,*) 'dble ',dble(E1c(j))
c      write(*,*) 'imag ',i_part1


          if (dabs(dble(E1c(j))).gt.1.0D+5*dabs(i_part1).and.
     & dabs(dble(E2c(j))).gt.1.0D+5*dabs(i_part2 ).and.
     &      dble(E1c(j)).gt.0d0.and.dble(E2c(j)).gt.0d0) then

c       write(*,*) 'we have a real number'   


            k=k+1 ! count the number of good sol
            normp1real(k)=dble(E1c(j))
            normp2real(k)=dble(E2c(j))
c      pause
          endif
          endif
      enddo

      if (k.eq.0) then 
        jac=-1d0
      return
      elseif (k.eq.1) then
        normp1=normp1real(1)
        normp2=normp2real(1)
        jac_factor=1d0
      elseif (k.eq.2) then
        jac_factor=2d0
c        call ntuple(rand,0.0,1.0,15)
             rand=xran1(IDUM)
c        write(*,*) rand
        if(rand.gt.0.5) then
          normp1=normp1real(1)
          normp2=normp2real(1)
        else
          normp1=normp1real(2)
          normp2=normp2real(2)
        endif
      elseif (k.eq.3) then
        jac_factor=3d0
c        call ntuple(rand,0.0,1.0,max_particles)
             rand=xran1(IDUM)
        if(rand.lt.1.0/3.0) then
          normp1=normp1real(1)
          normp2=normp2real(1)
        elseif(rand.ge.1.0/3.0.and.rand.lt.2.0/3.0) then
          normp1=normp1real(2)
          normp2=normp2real(2)
        else
          normp1=normp1real(3)
          normp2=normp2real(3)
        endif
      elseif (k.eq.4) then
        jac_factor=4d0
c        call ntuple(rand,0.0,1.0,max_particles)
             rand=xran1(IDUM)
        if(rand.lt.1.0/4.0) then
          normp1=normp1real(1)
          normp2=normp2real(1)
        elseif(rand.ge.1.0/4.0.and.rand.lt.1.0/2.0) then
          normp1=normp1real(2)
          normp2=normp2real(2)
        elseif(rand.ge.1.0/2.0.and.rand.lt.3.0/4.0) then
          normp1=normp1real(3)
          normp2=normp2real(3)
         else
          normp1=normp1real(3)
          normp2=normp2real(3)
        endif
      endif

       ! end case m1 >0
      elseif(pmass(p1).eq.0) then
      p1_ti=delta_mb2/2d0/(E3-f13)
      p1_p2=-(E3-f23)/(E3-f13)

c
c     solve the equation 
c     p1_p2 * |p2|^2 + p1_ti*|p2|-delta_ma2/2/(1-f12)=0
c

          jac_factor=1d0

      if (dabs(p1_p2).le.0.0001d0) then
        normp2=delta_ma2/2d0/(1d0-f12)/p1_ti
        normp1=p1_ti
        if  (normp1.le.0d0.or.normp2.le.0d0) then
          momenta(0,p1)=-1d0
          jac=-1d0
          return
        endif
      else
        b=p1_ti/p1_p2
        c=-delta_ma2/2d0/(1d0-f12)/p1_p2
        rho=b**2-4d0*c


        if (rho.eq.0d0.and.b.lt.0d0) then   ! max 1 sol
          normp2=-b/2d0
          normp1=p1_ti+p1_p2*normp2
          if  (normp1.le.0d0) then
            momenta(0,p1)=-1d0
            jac=-1d0
            return
          endif
        elseif (rho.gt.b**2)then ! max 1 sol
          normp2=(-b+dsqrt(rho))/2d0
          normp1=p1_ti+p1_p2*normp2
          if  (normp1.le.0d0) then
            momenta(0,p1)=-1d0
            jac=-1d0
            return
          endif
        elseif (rho.gt.0d0.and.dsqrt(rho).lt.(-b)) then   ! max 2 sol for p2
          sol(1)=(-b+dsqrt(rho))/2d0
          sol(2)=(-b-dsqrt(rho))/2d0
          p1sol(1)=p1_ti+p1_p2*sol(1)
          p1sol(2)=p1_ti+p1_p2*sol(2)
          if (p1sol(1).gt.0d0.and.p1sol(2).gt.0d0) then
            index_sol=1
c            call ntuple(rand,0.0,1.0,p1)
             rand=xran1(IDUM)
            if (rand.gt.0.5) index_sol=2
              normp2=sol(index_sol)
              normp1=p1sol(index_sol)
              jac_factor=2d0
          elseif (p1sol(1).gt.0d0.and.p1sol(2).le.0d0) then  
              normp2=sol(1)
              normp1=p1sol(1)
          elseif (p1sol(1).le.0d0.and.p1sol(2).gt.0d0) then  
              normp2=sol(2)
              normp1=p1sol(2)
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

      endif


      do nu=1,3
        momenta(nu,p1)=momenta(nu,p1)*normp1
        momenta(nu,p2)=momenta(nu,p2)*normp2
        momenta(nu,r1)=momenta(nu,p1)+momenta(nu,p2)
        momenta(nu,r2)=momenta(nu,r1)+momenta(nu,p3)
      enddo

      momenta(0,p1)=dsqrt(mvir2(p1)+normp1**2)
      momenta(0,p2)=normp2
      momenta(0,r1)=momenta(0,p1)+momenta(0,p2)
      momenta(0,r2)=momenta(0,r1)+momenta(0,p3)
c
c     compute missing momentum
c
       misspx=misspx-momenta(1,p1)- momenta(1,p2) 
       misspy=misspy-momenta(2,p1)-momenta(2,p2)
       Etot=Etot+momenta(0,p1)+momenta(0,p2)
       pztot=pztot+momenta(3,p1)+momenta(3,p2)
c
c     compute the jacobian:
c
      jac_loc=jac_loc*normP1**2*normp2/(4d0*momenta(0,p1))*jac_factor

      inv_jac=4d0*((normp1*normp2/momenta(0,p1)-normp2*f12)*
     & (E3-f23)-(E3*normp1/momenta(0,p1)-f13)*
     & (momenta(0,p1)-f12*normp1)) 
      
c      write(*,*) 'jac ', jac_loc

      jac=jac*jac_loc/dabs(inv_jac)

c      write(*,*) 'jac_factor ', jac_factor

      return
      end

