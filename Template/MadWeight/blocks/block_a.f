      subroutine block_a(x,p1,p2,p3,p4,r1,r2,r3)
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     This block corresponds to the reduced diagram
c
c            *P4     *P3    * P2
c           *      *       *
c     *********************
c     MV_r3  MV_r2  MV_r1  * 
c                           * P1 (missing)
c
c     the whole momentum p1 is fixed to reproduce MV_r1, MV_r2, MV_r3
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      implicit none
      include '../../../SubProcesses/phasespace.inc'
c
c     argument
c
      integer p1,p2,p3,p4,r1,r2,r3
c
c     local
c
      INTEGER IDUM
      DATA IDUM/0/
      SAVE IDUM
      double precision jac_loc
      double precision b,c,rho,sign_root,jac_factor,dem
      double precision p1x_ti,p1x_E1,p1y_ti,p1y_E1,p1z_ti,p1z_E1
      double precision invjac
      real rand
      double precision p1x,p1y,p1z,p2x,p2y,p2z,p3x,p3y,p3z,p4x,p4y,p4z
      double precision E1,E2,E3,E4
      double precision Ma(3),Mc(3),MinvB(3,3),detB
      integer nu
      double precision x(20)
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
c      write(*,*) 'P2', (momenta(nu,p2),nu=0,3)
c      write(*,*) 'P3', (momenta(nu,p3),nu=0,3)
c      write(*,*) 'P4', (momenta(nu,p4),nu=0,3)
c      write(*,*) 'mvir2',r1,mvir2(r1)
c      write(*,*) 'mvir2',r2,mvir2(r2)
c      write(*,*) 'mvir2',r3,mvir2(r3)
c      write(*,*) 'mvir2 -4',mvir2(-4)
c      pause
      p2x=momenta(1,p2)
      p3x=momenta(1,p3)
      p4x=momenta(1,p4)
      p2y=momenta(2,p2)
      p3y=momenta(2,p3)
      p4y=momenta(2,p4)
      p2z=momenta(3,p2)
      p3z=momenta(3,p3)
      p4z=momenta(3,p4)
      E2=momenta(0,p2)
      E3=momenta(0,p3)
      E4=momenta(0,p4)

c     Express the change of variables as
c       | a1 |   | b11 b12 b13| | P1x |   | C1 | 
c       | a2 | = | b21 b22 b23| | P1y | + | C2 | E1
c       | a3 |   | b31 b32 b33| | P1z | + | C3 |

      Ma(1)=mvir2(r1)-mvir2(p1)-mvir2(p2)
      Ma(2)=mvir2(r2)-mvir2(r1)-mvir2(p3)
     & -2d0*dot(momenta(0,p2),momenta(0,p3))
      Ma(3)=mvir2(r3)-mvir2(r2)-mvir2(p4)
     & -2d0*dot(momenta(0,p2),momenta(0,p4))
     & -2d0*dot(momenta(0,p3),momenta(0,p4))
c
      detB=8d0*(p2z*p3y*p4x-p2y*p3z*p4x-p2z*p3x*p4y+p2x*p3z*p4y 
     & + p2y*p3x*p4z - p2x*p3y*p4z)
c
      MinvB(1,1)=(-4d0*p3z*p4y + 4d0*p3y*p4z)/detB
      MinvB(1,2)=(4d0*p2z*p4y - 4d0*p2y*p4z)/detB
      MinvB(1,3)=(-4d0*p2z*p3y + 4d0*p2y*p3z)/detB
      MinvB(2,1)=(4d0*p3z*p4x - 4d0*p3x*p4z)/detB
      MinvB(2,2)=(-4d0*p2z*p4x + 4d0*p2x*p4z)/detB
      MinvB(2,3)=(4d0*p2z*p3x - 4d0*p2x*p3z)/detB
      MinvB(3,1)=(-4d0*p3y*p4x + 4d0*p3x*p4y)/detB
      MinvB(3,2)=(4d0*p2y*p4x - 4d0*p2x*p4y)/detB
      MinvB(3,3)=(-4d0*p2y*p3x + 4d0*p2x*p3y)/detB
c
      Mc(1)=2d0*momenta(0,p2)
      Mc(2)=2d0*momenta(0,p3)
      Mc(3)=2d0*momenta(0,p4)
c
c     Inverse the system
c       | p1x |   | p1x_ti+p1x_E1*E1 | 
c       | p1y | = | p1y_ti+p1y_E1*E1 |
c       | p1z |   | p1z_ti+p1z_E1*E1 |
c
      P1x_ti=MinvB(1,1)*Ma(1)+MinvB(1,2)*Ma(2)+MinvB(1,3)*Ma(3)
      P1y_ti=MinvB(2,1)*Ma(1)+MinvB(2,2)*Ma(2)+MinvB(2,3)*Ma(3)
      P1z_ti=MinvB(3,1)*Ma(1)+MinvB(3,2)*Ma(2)+MinvB(3,3)*Ma(3)
c
      p1x_E1=-MinvB(1,1)*Mc(1)-MinvB(1,2)*Mc(2)-MinvB(1,3)*Mc(3)
      p1y_E1=-MinvB(2,1)*Mc(1)-MinvB(2,2)*Mc(2)-MinvB(2,3)*Mc(3)
      p1z_E1=-MinvB(3,1)*Mc(1)-MinvB(3,2)*Mc(2)-MinvB(3,3)*Mc(3)
c    
c     solve the mass shell equation 
c    
c     E1^2-|P1|^2=m_1^2   <=>   E1^2-b*E1+c=0 
c
      jac_factor=1d0
      dem=1d0-P1x_E1**2-P1y_E1**2-P1z_E1**2

      if (dabs(dem).lt.0.0001d0) then
        momenta(0,p1)=-(P1x_ti**2+P1y_ti**2+P1z_ti**2+mvir2(p1))/
     &  (2d0*P1x_ti*P1x_E1+2d0*P1y_ti*P1y_E1+2d0*P1z_ti*P1z_E1)
        if  (momenta(0,p1).lt.0d0) then
          jac=-1d0
          return
        endif
        jac_factor=1d0
      else
        b=-(2d0*P1x_ti*P1x_E1+2d0*P1y_ti*P1y_E1+2d0*P1z_ti*P1z_E1)/dem
        c=-(P1x_ti**2+P1y_ti**2+P1z_ti**2+mvir2(p1))/dem
        rho=b**2-4d0*c
 
        if (rho.eq.0d0.and.b.lt.0d0) then   ! 1 sol
          momenta(0,p1)=-b/2d0
          jac_factor=1d0
        elseif (rho.gt.b**2)then ! 1 sol
          momenta(0,p1)=(-b+dsqrt(rho))/2d0
          jac_factor=1d0
        elseif (rho.gt.0d0.and.dsqrt(rho).lt.(-b)) then   ! 2 sol
          sign_root=1d0
c          call ntuple(rand,0.0,1.0,p1)
           rand=xran1(IDUM)
          if (rand.gt.0.5) sign_root=-1.0d0
          momenta(0,p1)=(-b+sign_root*dsqrt(rho))/2d0
          jac_factor=2d0
        else                                ! 0 sol
          momenta(0,p1)=-1d0
          jac=-1d0
          return 
        endif
      endif
c
c     fill the momentum p1
c
      E1=momenta(0,p1)
      momenta(1,p1)=p1x_ti+p1x_E1*E1
      momenta(2,p1)=p1y_ti+p1y_E1*E1
      momenta(3,p1)=p1z_ti+p1z_E1*E1
      p1x=momenta(1,p1)
      p1y=momenta(2,p1)
      p1z=momenta(3,p1)
c      normP1sq=p1x**2+p1y**2+p1z**2
c      sin_theta_miss=dsqrt(1d0-(p1x**2+p1y**2)/normP1sq)
c
c     fill the momenta r1,r2,r3
c
      do nu=0,3
      momenta(nu,r1)=momenta(nu,p1)+momenta(nu,p2)
      momenta(nu,r2)=momenta(nu,r1)+momenta(nu,p3)
      momenta(nu,r3)=momenta(nu,r2)+momenta(nu,p4)
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
c      jac_loc=jac_loc*sin_theta_miss*normP1sq       ! PS weight for the initial set of variable
                   !  the factor 2*E1 has been removed since it factorizes with the factor 2E1 in the PS param.
c
      invjac=-16*(E4*(p1z*p2y*p3x - p1y*p2z*p3x - 
     -        p1z*p2x*p3y + p1x*p2z*p3y + p1y*p2x*p3z - p1x*p2y*p3z)
     -      + E2*p1z*p3y*p4x - E1*p2z*p3y*p4x - E2*p1y*p3z*p4x + 
     -     E1*p2y*p3z*p4x - E2*p1z*p3x*p4y + E1*p2z*p3x*p4y + 
     -     E2*p1x*p3z*p4y - E1*p2x*p3z*p4y + 
     -     (E2*p1y*p3x - E1*p2y*p3x - E2*p1x*p3y + E1*p2x*p3y)*p4z + 
     -     E3*(-(p1z*p2y*p4x) + p1y*p2z*p4x + p1z*p2x*p4y - 
     -        p1x*p2z*p4y - p1y*p2x*p4z + p1x*p2y*p4z))

      jac_loc=jac_factor/dabs(invjac)     
      jac=jac*jac_loc   

      return
      end

