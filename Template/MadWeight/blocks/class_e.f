      subroutine class_e(x,p1,p2,p3,p4,r1,r2,r3)
c********************************************************************
c     This subroutine gets the two missing momenta for 
c     the following topology:
c
c                          ________ missing    p1
c                     r1  *________ visible    p3
c                        *
c     ------------------*
c            r3          * ________ missing    p2
c                      r2 *________ visible    p4
c
c********************************************************************
c
      implicit none
      include '../../../SubProcesses/phasespace.inc'
      double precision thres
      parameter (thres=1700d0)
c
c     arguments
c
      double precision x(20)
      integer p1,p2,p3,p4,r1,r2,r3,n_var,local_var

c      double precision m1,m2,misspx,misspy,Etv,pztv,Einit,pzinit
c      double precision p3(0:3),p4(0:3),p1(0:3,2),p2(0:3,2),mvir13,mvir24
c      double precision jac_loc(2)
c
c     local
c
      INTEGER IDUM
      DATA IDUM/0/
      SAVE IDUM
      double precision E3,E4,p3x,p3y,p3z,p4x,p4y,p4z
      double precision A1,A2,A3,A4,B1,B2,B3,B4
      double precision p1z_ti,p1z_E2,p1z_p2y
      double precision p2z_ti,p2z_E2,p2z_p2y
      double precision p2x_ti,p2x_E2,p2x_p2y
      double precision E1_ti,E1_E2,E1_p2y
      double precision p1x_ti, p1x_E2, p1x_p2y, p1y_ti, p1y_E2, p1y_p2y
      double precision E2,p2x,p2y,p2z
      double precision p1x,p1y,p1z,E1
      double precision g11,g12,g10,g22,g20,g00
c      double precision h11,h12,h22
      double precision h10,h20,h00
      double precision inv_jac,jac_loc
      double precision dem,b,c,alpha,beta,rho,sqrts
      double precision x1s1,x2s1,x1s2,x2s2,jac_factor
      double precision sol(2),trialp2y(2),trialE1(2)
      double precision trialp1z(2),trialp2z(2)
      double precision etamax,etamin,eta
      double precision pboost(0:3)
      double precision measure1,measure2
      double precision CMS_mom(0:3,max_particles)
      integer index_sol, MG
      real rand
      REAL XRAN1
      EXTERNAL XRAN1
      integer j
c
c     global
c
      double precision pxISR, pyISR
      common /to_ISR/  pxISR, pyISR
      integer matching_type_part(3:max_particles) !modif/link between our order by type for permutation
      integer inv_matching_type_part(3:max_particles)
      common/madgraph_order_type/matching_type_part,
     & inv_matching_type_part
      integer nexternal, num_inv
      COMMON/to_num_inv/nexternal, num_inv
      double precision momenta(0:3,-max_branches:2*max_particles)  ! records the momenta of external/intermediate legs     (MG order)
      double precision mvir2(-max_branches:2*max_particles)        ! records the sq invariant masses of intermediate particles (MG order)
      common /to_diagram_kin/ momenta, mvir2
      double precision Etot,pztot,miss_px,miss_py
      common /to_missingP/Etot,pztot,miss_px,miss_py
      double precision              S,X1,X2,PSWGT,JAC
      common /PHASESPACE/ S,X1,X2,PSWGT,JAC
      integer ISR_mode
      common /to_correct_ISR/ISR_mode

       IDUM=0
      sqrts=dsqrt(s)
      momenta(0,p1)=-1d0
      momenta(0,p2)=-1d0
      jac_factor=1d0
      if((Etot+dsqrt(miss_px**2+miss_py**2)).gt.sqrts) then
      jac=-1d0
      return
      endif
c
c      if (mvir2(r1).lt.(dsqrt(mvir2(p1))+dsqrt(mvir2(p3)))**2.or.
c     & mvir2(r2).lt.(dsqrt(mivr2(p2))+dsqrt(mvir2(p4)))**2 ) then
c      jac_loc=-1d0
c      jac=-1d0
c      endif

c
c     generate rapidity
c
      n_var=n_var+1
      ETAMIN = .5D0*LOG(mvir2(r3)/s)
      ETAMAX = -ETAMIN
      jac_loc = 1d0
      call  generate_flat(x,1, ETAMIN, ETAMAX, ETA, jac_loc)
      

      X1 = dsqrt(mvir2(r3)/s)*DEXP(ETA)
      X2 = dsqrt(mvir2(r3)/s)*DEXP(-ETA)
c
c     initialization in the CMS of initial particles
c 
      momenta(0,1)=sqrts*x1/2d0
      momenta(1,1)=0d0
      momenta(2,1)=0d0
      momenta(3,1)=sqrts*x1/2d0
      momenta(0,2)=sqrts*x2/2d0
      momenta(1,2)=0d0
      momenta(2,2)=0d0
      momenta(3,2)=-sqrts*x2/2d0
c
c     Here we need to boost the initial momenta in case of ISR:
c     the momentum associated with the boost can be obtained from 
c     the momentum of resonance r3, with 
c         m_inv(r3)^2 =   shat
c         pt_inv(r3)  = - pt(ISR)
c
c      =>  pboost = ( sqrt( pt(ISR)**2 + m_inv(r3)^2 ), -pxISR, -pyISR )
c
c     apply the boost to lab frame
      pboost(0)=dsqrt( mvir2(r3) + pxISR**2 + pyISR**2)*dcosh(ETA) 
      pboost(1)=-pxISR
      pboost(2)=-pyISR
      pboost(3)=0d0

      call boostx(momenta(0,1),pboost,momenta(0,1))
      call boostx(momenta(0,2),pboost,momenta(0,2))

      E3=momenta(0,p3)
      E4=momenta(0,p4)
      p3x=momenta(1,p3)
      p3y=momenta(2,p3)
      p3z=momenta(3,p3)
      p4x=momenta(1,p4)
      p4y=momenta(2,p4)
      p4z=momenta(3,p4)

c      write(*,*) 'E3', E3
c      write(*,*) 'E4', E4
c      write(*,*) 'p3x', p3x
c      write(*,*) 'p3y', p3y
c      write(*,*) 'p3z', p3z
c      write(*,*) 'p4x', p4x
c      write(*,*) 'p4y', p4y
c      write(*,*) 'p4z', p4z
c      write(*,*) 'mvir2',mvir2
c      write(*,*) 'sqrts',sqrts
c      write(*,*) 'miss', miss_px,miss_py
c      write(*,*) 'etot,pztot',etot,pztot
c      write(*,*) 'x/nvar',x,n_var
c      pause
c
c     start from the following expressions for p1z and p2x:
c
c     p1z=A1 + A2 E2 + A3 p2y + A4 p2x
c     p2x=B1 + B2 E2 + B3 p2y + A4 p1z
c
      A1=(mvir2(p1)+mvir2(p3)+2d0*E3*
     & (momenta(0,1)+momenta(0,2)-Etot)
     &-2d0*p3x*miss_px-2d0*p3y*miss_py -mvir2(r1))/(2d0*p3z)
      A2=-E3/p3z
      A3=p3y/p3z
      A4=p3x/p3z
c
      B1=(mvir2(p2)+mvir2(p4)-2d0*p4z*
     & (momenta(3,1)+momenta(3,2)-pztot)-mvir2(r2))
     & /(2d0*p4x)
      B2=E4/p4x
      B3=-p4y/p4x
      B4=p4z/p4x
c
      p1z_ti=(A1+A4*B1)/(1d0-A4*B4)
      p1z_E2=(A2+A4*B2)/(1d0-A4*B4)
      p1z_p2y=(A3+A4*B3)/(1d0-A4*B4)

c
      p2x_ti=(B1+B4*A1)/(1d0-A4*B4)
      p2x_E2=(B2+B4*A2)/(1d0-A4*B4)
      p2x_p2y=(B3+B4*A3)/(1d0-A4*B4)
c
c      write(*,*) 'A4*B4',A4*B4
c
      p1x_ti=miss_px-p2x_ti
      p1x_E2=-p2x_E2
      p1x_p2y=-p2x_p2y
c
      p1y_ti=miss_py
      p1y_p2y=-1d0
      p1y_E2=0d0
c
      E1_ti=momenta(0,1)+momenta(0,2)-Etot
      E1_E2=-1d0
      E1_p2y=0d0
c
      p2z_ti=momenta(3,1)+momenta(3,2)-p1z_ti-pztot
      p2z_E2=-p1z_E2
      p2z_p2y=-p1z_p2y

c
c     mass shell conditions:     
c    
c     E1^2-p1x^2-p1y^2-p1z^2=m1^2   (1)
c     E2^2-p2x^2-p2y^2-p2z^2=m2^2   (2) 
c
c     equivalent to
c
c     g11*E2^2 + g22*p2y^2 + g12*E2*p2y + g10*E2 + g20*p2y + g00 = 0  (1)
c     h11*E2^2 + h22*p2y^2 + h12*E2*p2y + h10*E2 + h20*p2y + h00 = 0  (2)
c
      g11=E1_E2**2-p1x_E2**2-p1y_E2**2-p1z_E2**2
c
      g22=E1_p2y**2-p1x_p2y**2-p1y_p2y**2-p1z_p2y**2
c
      g12=2d0*(E1_E2*E1_p2y-p1x_E2*p1x_p2y-p1y_E2*p1y_p2y-
     & p1z_E2*p1z_p2y)
c
      g10=2d0*(E1_E2*E1_ti-p1x_E2*p1x_ti-p1y_E2*p1y_ti-p1z_E2*p1z_ti)
c 
      g20=2d0*(E1_ti*E1_p2y-p1x_ti*p1x_p2y-p1y_ti*p1y_p2y-
     & p1z_ti*p1z_p2y)
c
      g00=E1_ti**2-p1x_ti**2-p1y_ti**2-p1z_ti**2-mvir2(p1)
c
c       write(*,*) 'g11,g22,g12,g10,g20,g00',g11,g22,g12,g10,g20,g00
c      h11=1d0-p2x_E2**2-p2z_E2**2
c      h22=-1d0-p2x_p2y**2-p2z_p2y**2
c      h12=-2d0*(p2x_p2y*p2x_E2+p2z_p2y*p2z_E2)
c
      h10=-2d0*(p2x_ti*p2x_E2+p2z_ti*p2z_E2)
c
      h20=-2d0*(p2x_p2y*p2x_ti+p2z_p2y*p2z_ti)
c
      h00=-p2x_ti**2-p2z_ti**2-mvir2(p2)
c
c     note that the two quartic equations are fake:
c
c     h11=g11
c     h22=g22
c     h12=g12
c
c     We are left with the equation
c
c     (g10-h10) E2 + (g20-h20) p2y + g00-h00 = 0
c
c      <=>   p2y = alpha + beta E2
c
c      write(*,*) 'g20,h20',g20,h20 
c      write(*,*) 'g10,h10',g10,h10 
c      write(*,*) 'g00,h00',g00,h00 
      if (dabs(g20-h20).gt.1d-5) then
      alpha=-(g00-h00)/(g20-h20)
      beta=-(g10-h10)/(g20-h20)
      elseif (dabs(g10-h10).gt.1d-5) then
        E2=-(g00-h00)/(g10-h10)
        if (E2.gt.0d0) then
          dem = g22
          b=(g12*E2+g20)/dem
          c=(g11*E2**2+g10*E2+g00)/dem
          rho=b**2-4d0*c
          if (rho.eq.0d0) then   ! max 1 sol
            p2y=-b/2d0
            goto 13
          elseif (rho.gt.0d0) then
            sol(1)=(-b+dsqrt(rho))/2d0
            sol(2)=(-b-dsqrt(rho))/2d0
            index_sol=1
c            call ntuple(rand,0.0,1.0,p1)
            rand=xran1(IDUM)
            if (rand.gt.0.5) index_sol=2
            p2y=sol(index_sol)
            jac_factor=2d0
            goto 13 
          else
            jac_loc=-1d0
            jac=-1d0
            return
          endif
        else
          jac_loc=-1d0
          jac=-1d0
          return
        endif
      else
c     write(*,*) 'warning : num instability at code lksdqf'
      jac_loc=-1d0
      jac=-1d0
      return
      endif

      dem=g11+g22*beta**2+g12*beta
c      c1=2D0*g22*alpha*beta+g12*alpha+g10+g20*beta
c       c0=g22*alpha**2+g20*alpha+g00

      if (dabs(dem).lt.0.0001d0) then
        E2=-(g22*alpha**2+g20*alpha+g00)/
     & (2D0*g22*alpha*beta+g12*alpha+g10+g20*beta)
        if  (E2.le.0d0) then
          jac_loc=-1d0
          jac=-1d0
          return
        endif
      else
        b=(2D0*g22*alpha*beta+g12*alpha+g10+g20*beta)/dem
        c=(g22*alpha**2+g20*alpha+g00)/dem
        rho=b**2-4d0*c

c        write(*,*) 'b,c,rho',b,c,rho
        
        if (rho.eq.0d0.and.b.lt.0d0) then   ! max 1 sol
          E2=-b/2d0
        elseif (rho.gt.b**2)then ! max 1 sol
          E2=(-b+dsqrt(rho))/2d0
        elseif (rho.gt.0d0.and.dsqrt(rho).lt.(-b)) then   ! max 2 sol
          sol(1)=(-b+dsqrt(rho))/2d0
          sol(2)=(-b-dsqrt(rho))/2d0

c        write(*,*) 'E2',sol(1),sol(2)

          trialp2y(1)=alpha+beta*sol(1)
          trialE1(1) =E1_ti + E1_E2*sol(1) + E1_p2y*trialp2y(1)
          trialp1z(1)=p1z_ti + p1z_E2*sol(1) + p1z_p2y*trialp2y(1)
          trialp2z(1)=p2z_ti + p2z_E2*sol(1) + p2z_p2y*trialp2y(1)

          trialp2y(2)=alpha+beta*sol(2)
          trialE1(2) =E1_ti + E1_E2*sol(2) + E1_p2y*trialp2y(2)
          trialp1z(2)=p1z_ti + p1z_E2*sol(2) + p1z_p2y*trialp2y(2)
          trialp2z(2)=p2z_ti + p2z_E2*sol(2) + p2z_p2y*trialp2y(2)


          x1s1=((Etot+sol(1)+trialE1(1))
     & +(pztot+trialp1z(1)+trialp2z(1)))/sqrts
          x2s1=((Etot+sol(1)+trialE1(1))
     & -(pztot+trialp1z(1)+trialp2z(1)))/sqrts

          x1s2=((Etot+sol(2)+trialE1(2))
     & +(pztot+trialp1z(2)+trialp2z(2)))/sqrts
          x2s2=((Etot+sol(2)+trialE1(2))
     & -(pztot+trialp1z(2)+trialp2z(2)))/sqrts


      if(dabs(x1s1-0.5d0).lt.0.5d0.and.dabs(x2s1-0.5d0).lt.0.5d0.and. !analyse bjk fractions
     & dabs(x1s2-0.5d0).lt.0.5d0.and.dabs(x2s2-0.5d0).lt.0.5d0) then

            index_sol=1
c            call ntuple(rand,0.0,1.0,p1)
            rand=xran1(IDUM)
            if (rand.gt.0.5) index_sol=2
            E2=sol(index_sol)
            jac_factor=2d0

      elseif(dabs(x1s1-0.5d0).gt.0.5d0.or.dabs(x2s1-0.5d0).gt.0.5d0) 
     & then

          if (dabs(x1s2-0.5d0).lt.0.5d0.and.dabs(x2s2-0.5d0).lt.0.5d0) 
     & then
              E2=sol(2)
          else
            jac_loc=-1d0
            jac=-1d0
            return
          endif

      elseif(dabs(x1s2-0.5d0).gt.0.5d0.or.dabs(x2s2-0.5d0).gt.0.5d0) 
     & then
          if (dabs(x1s1-0.5d0).lt.0.5d0.and.dabs(x2s1-0.5d0).lt.0.5d0) 
     & then
              E2=sol(1)
          else
            jac_loc=-1d0
            jac=-1d0
            return
          endif
      endif
      else 
      jac_loc=-1d0
      jac=-1d0
      return
      endif
      endif

c      c2=g11+g22*beta**2+g12*beta
c      c1=2D0*g22*alpha*beta+g12*alpha+g10+g20*beta
c      c0=g22*alpha**2+g20*alpha+g00

c      write(*,*) 'E2',E2
      p2y=alpha+beta*E2
c
13    E1 =E1_ti + E1_E2*E2 + E1_p2y*p2y
      if (E1.lt.0d0) then
         jac=-1d0
         return
      endif
      p1x=p1x_ti + p1x_E2*E2 + p1x_p2y*p2y
      p1y=p1y_ti + p1y_E2*E2 + p1y_p2y*p2y
      p1z=p1z_ti + p1z_E2*E2 + p1z_p2y*p2y

      p2x=p2x_ti + p2x_E2*E2 + p2x_p2y*p2y
      p2z=p2z_ti + p2z_E2*E2 + p2z_p2y*p2y
c
c     Now fill the set of missing momenta k
c
      momenta(0,p1)=E1
      momenta(1,p1)=p1x
      momenta(2,p1)=p1y
      momenta(3,p1)=p1z
c
      momenta(0,p2)=E2
      momenta(1,p2)=p2x
      momenta(2,p2)=p2y
      momenta(3,p2)=p2z

c      fill intermediate momenta
      do j=0,3
       momenta(j,r1)=momenta(j,p1)+momenta(j,p3)
       momenta(j,r2)=momenta(j,p2)+momenta(j,p4)
       momenta(j,r3)=momenta(j,r1)+momenta(j,r2)
      enddo

      miss_px=0d0
      miss_py=0d0

c
c     inverse jacobian from mathematica: [inv_jac]=
c
      inv_jac=16d0*(E4*(p1z*p2y*p3x - p1y*p2z*p3x - p1z*p2x*p3y + 
     -        p1x*p2z*p3y + p1y*p2x*p3z - p1x*p2y*p3z) + 
     -     E2*p1z*p3y*p4x - E1*p2z*p3y*p4x - E2*p1y*p3z*p4x + 
     -     E1*p2y*p3z*p4x - E2*p1z*p3x*p4y + E1*p2z*p3x*p4y + 
     -     E2*p1x*p3z*p4y - E1*p2x*p3z*p4y + 
     -     (E2*p1y*p3x - E1*p2y*p3x - E2*p1x*p3y + E1*p2x*p3y)*p4z + 
     -     E3*(-(p1z*p2y*p4x) + p1y*p2z*p4x + p1z*p2x*p4y - 
     -        p1x*p2z*p4y - p1y*p2x*p4z + p1x*p2y*p4z))


       if (dabs(inv_jac).gt.thres) then
c
c      dimension of jac_loc is
c
       jac_loc=jac_loc/dabs(inv_jac)*jac_factor/(2d0*x1*x2*s)/s
c       write(*,*) 'jac_loc',jac_loc
       else
       write(*,*) 'warning: jac^-1 is close to zero'
       write(*,*) 'inv_jac',inv_jac
       jac_loc=-1d0
       jac=-1d0
       return
       endif
c      write(*,*) 'jac_factor',jac_factor
      jac=jac*jac_loc


c     correction from the measure to translate the weight to the CM frame
c     ONLY if isr = 2 

      if (isr_mode.eq.2) then
      measure1=1d0
       do j=3,nexternal-num_inv
         MG=inv_matching_type_part(j)
         measure1=measure1*dsqrt(momenta(1,MG)**2+momenta(2,MG)**2)
       enddo

      pboost(1)=-pboost(1)
      pboost(2)=-pboost(2)
       do j=3,nexternal -num_inv
c         write(*,*) "p",j,momenta(0,j), momenta(1,j),momenta(2,j),momenta(3,j)
         MG=inv_matching_type_part(j)
         call boostx(momenta(0,MG),pboost,CMS_mom(0,MG))
       enddo

      measure2=1d0
       do j=3,nexternal-num_inv
         MG=inv_matching_type_part(j)
         measure2=measure2*dsqrt(CMS_mom(1,MG)**2+CMS_mom(2,MG)**2)
       enddo

      jac=jac*measure2/measure1
      endif

      return
      end

