      subroutine class_h(nexternal)
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
      integer nexternal
c
c     local
c
      double precision jac_temp,det,sqrts
      double precision jac_loc
      integer j,nu
      double precision  pboost(0:3),parton_mom(0:3,max_particles)
      double precision shat,yhat,k1(0:3),k2(0:3),Ptot(0:3),Ep
      double precision measure1, measure2
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
      double precision ISRpx,ISRpy
      common /to_ISR/ ISRpx,ISRpy
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

c      write(*,*) "jac init class h", jac
c      write(*,*) "mvir2(-2) 1",mvir2(-2)
      ISRpx=misspx
      ISRpy=misspy
c
c     calculate s_hat, y_hat = invariant mass and rapitity of the parton-level system (hard-scattering)
c
      do nu=0,3
        Ptot(nu)=0d0
      enddo

      do j=3,nexternal
       do nu=0,3
        Ptot(nu)=Ptot(nu)+momenta(nu,j)  ! momentum of the hard particles in the FS
       enddo
      enddo
      shat=dot(Ptot,Ptot)
      Ep=dsqrt(Ptot(0)**2-Ptot(1)**2-Ptot(2)**2)
      yhat=0.5*dlog((Ep+Ptot(3))/(Ep-Ptot(3)))

c      write(*,*) "mvir2(-2) 2",mvir2(-2)
c     define x1,x2
c
      x1=dsqrt(shat/s)*dexp(yhat)
      x2=dsqrt(shat/s)*dexp(-yhat)

        if(dabs(x1-0.5d0).gt.0.5d0.or.dabs(x2-0.5d0).gt.0.5d0) then
        jac=-1d0
        momenta(0,1)=-1
        momenta(0,2)=-1
        return
        endif

c      write(*,*) "mvir2(-2) 3",mvir2(-2)
c     define initial momenta in the parton-level system (hard-scattering)
      k1(0)=sqrts*x1/2d0
      k1(1)=0d0
      k1(2)=0d0
      k1(3)=sqrts*x1/2d0
c
      k2(0)=sqrts*x2/2d0
      k2(1)=0d0
      k2(2)=0d0
      k2(3)=-sqrts*x2/2d0

c     apply the boost to lab frame
      pboost(0)=Etot
      pboost(1)=-misspx
      pboost(2)=-misspy
      pboost(3)=0d0

c      write(*,*) "mvir2(-2) 4",mvir2(-2)
      call boostx(k1,pboost,momenta(0,1))
      call boostx(k2,pboost,momenta(0,2))

c      write(*,*) "mvir2(-2) 5",mvir2(-2)
c
c     flux factor
c
      jac_loc=jac_loc/(S**2*x1*x2) ! flux + jac x1,x2 -> Etot, Pztot
      jac=jac*jac_loc
c      write(*,*) "jac fin class h", jac

c
c     also need to rescale the weight to compensate for the transformation of the 
c     probability density under boosts:
c
      measure1=1d0
       do j=3,nexternal
         measure1=measure1*(momenta(1,j)**2+momenta(2,j)**2+momenta(3,j)**2)
     & *dsqrt(momenta(1,j)**2+momenta(2,j)**2)/momenta(0,j)**2
       enddo

      pboost(1)=-pboost(1)
      pboost(2)=-pboost(2)
       do j=3,nexternal
c         write(*,*) "p",j,momenta(0,j), momenta(1,j),momenta(2,j),momenta(3,j)
         call boostx(momenta(0,j),pboost,parton_mom(0,j))
       enddo

      measure2=1d0
       do j=3,nexternal
         measure2=measure2*(parton_mom(1,j)**2+parton_mom(2,j)**2+parton_mom(3,j)**2)
     & *dsqrt(parton_mom(1,j)**2+parton_mom(2,j)**2)/parton_mom(0,j)**2
       enddo
      
      jac=jac*measure2/measure1


      return
      end
