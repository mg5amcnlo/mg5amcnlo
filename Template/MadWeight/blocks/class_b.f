      subroutine class_b(x,p1,p2,r1)
c***************************************************************************
c
c             *  p2 (visible)
c            *
c     *******
c       r1   *
c             *  p1 (missing)
c
c***************************************************************************


c***************************************************************************
      implicit none
      include '../../../SubProcesses/phasespace.inc'
c
c     arguments
c
      integer p1,p2,r1
c
c     local
c
      double precision x(20)
      INTEGER IDUM
      DATA IDUM/0/
      SAVE IDUM
      double precision pboost(0:3), CMS_mom(0:3,max_particles)
      double precision Ptot(0:3),PtotCMS(0:3)
      double precision measureLAB, measureCMS
      double precision jac_loc
      double precision p1z_ti,p1z_E1,jac_factor,dem
      double precision b,c,rho,sqrts
      double precision sol(2),trialpz(2)
      double precision x1s1,x2s1,x1s2,x2s2
      integer index_sol,i,j,k,MG
      real rand
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

      REAL XRAN1
      EXTERNAL XRAN1
c--------
c Begin code 
c--------
      IDUM=0
      sqrts=dsqrt(s)
      if((Etot+dsqrt(misspx**2+misspy**2)).gt.sqrts) then
      jac=-1d0
      return
      endif

c      write(*,*) "mom p2", (momenta(i,p2),i=0,3) 
c      write(*,*) "mvir2",r1, mvir2(r1)
c      write(*,*) "mvir2",p1, mvir2(p1)
c      write(*,*) "mvir2",p2, mvir2(p2)
c      write(*,*) "misspx",misspx
c      write(*,*) "misspy",misspy
c      write(*,*) "Etot",Etot
c      write(*,*) "Pztot",Pztot

c     fill p1x, p1y
c
      momenta(1,p1)=misspx
      momenta(2,p1)=misspy
      momenta(0,p1)=-1d0

c
c     p1z=p1z_ti+p1z_E1*E1
c
      p1z_ti=(mvir2(p1)+mvir2(p2)-mvir2(r1)
     & -2d0*momenta(1,p1)*momenta(1,p2)-2d0*momenta(2,p1)*momenta(2,p2))
     & /(2d0*momenta(3,p2))
      p1z_E1=momenta(0,p2)/momenta(3,p2)
c
c     the mass-shell equation reads (1-p1z_ti^2)*E1^2 -2 p1z_ti*p1z_E1*E1-p1z_ti**2-p1x^2-p1y^2 -m1^2
c
     
      jac_factor=1d0
      index_sol=1
      dem=1d0-p1z_E1**2

      if (dabs(dem).lt.0.0001d0) then
        momenta(0,p1)=-(P1z_ti**2+misspx**2+misspy**2+mvir2(p1))/
     &  (2d0*P1z_ti*P1z_E1)
 
        if  (momenta(0,p1).le.0d0) then
          jac_loc=-1d0
          jac=-1d0
          return
        endif
      else
        b=-2d0*P1z_ti*P1z_E1/dem
        c=-(P1z_ti**2+misspx**2+misspy**2+mvir2(p1))/dem
        rho=b**2-4d0*c


        if (rho.eq.0d0.and.b.lt.0d0) then   ! max 1 sol
          momenta(0,p1)=-b/2d0
        elseif (rho.gt.b**2)then ! max 1 sol
          momenta(0,p1)=(-b+dsqrt(rho))/2d0
        elseif (rho.gt.0d0.and.dsqrt(rho).lt.(-b)) then   ! max 2 sol
          sol(1)=(-b+dsqrt(rho))/2d0
          sol(2)=(-b-dsqrt(rho))/2d0
          trialpz(1)=p1z_ti+p1z_E1*sol(1)
          trialpz(2)=p1z_ti+p1z_E1*sol(2)
          x1s1=((Etot+sol(1))+(pztot+trialpz(1)))/sqrts
          x2s1=((Etot+sol(1))-(pztot+trialpz(1)))/sqrts
          x1s2=((Etot+sol(2))+(pztot+trialpz(2)))/sqrts
          x2s2=((Etot+sol(2))-(pztot+trialpz(2)))/sqrts

c          write(*,*) 'E 1',sol(1)
c          write(*,*) 'E 2',sol(2)
         
c          write(*,*) 'x1,x2',x1s1,x2s1
c          write(*,*) 'x1,x2',x1s2,x2s2

      if(dabs(x1s1-0.5d0).lt.0.5d0.and.dabs(x2s1-0.5d0).lt.0.5d0.and. ! analyse bjk fractions
     & dabs(x1s2-0.5d0).lt.0.5d0.and.dabs(x2s2-0.5d0).lt.0.5d0) then
           
            index_sol=1
c            call ntuple(rand,0.0,1.0,p1)
            rand=xran1(IDUM)
            if (rand.gt.0.5) index_sol=2
c            if (rand.gt.0.5) then
c              jac=-1d0
c              jac_loc=-1d0
c            endif
            momenta(0,p1)=sol(index_sol)
            jac_factor=2d0

      elseif(dabs(x1s1-0.5d0).gt.0.5d0.or.dabs(x2s1-0.5d0).gt.0.5d0)then
      if (dabs(x1s2-0.5d0).lt.0.5d0.and.dabs(x2s2-0.5d0).lt.0.5d0) then
              momenta(0,p1)=sol(2)
          else 
            jac_loc=-1d0
            jac=-1d0
            return
          endif

      elseif(dabs(x1s2-0.5d0).gt.0.5d0.or.dabs(x2s2-0.5d0).gt.0.5d0)then
      if (dabs(x1s1-0.5d0).lt.0.5d0.and.dabs(x2s1-0.5d0).lt.0.5d0) then
              momenta(0,p1)=sol(1)
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

c      if (index_sol.eq.1) then
c        jac_loc=-1d0
c        jac=-1d0
c        return
c      endif


      momenta(3,p1)=p1z_ti+p1z_E1*momenta(0,p1)
c      write(*,*) 'momenta(3,p1)',momenta(3,p1)
      x1=((Etot+momenta(0,p1))+(pztot+momenta(3,p1)))/sqrts
      x2=((Etot+momenta(0,p1))-(pztot+momenta(3,p1)))/sqrts
      if (dabs(x1-0.5d0).gt.0.5d0.or.dabs(x2-0.5d0).gt.0.5d0) then
        jac_loc=-1d0
        jac=-1d0
        return
      endif

c     fill intermediate momenta
      momenta(0,r1)=momenta(0,p1)+momenta(0,p2)
      momenta(1,r1)=momenta(1,p1)+momenta(1,p2)
      momenta(2,r1)=momenta(2,p1)+momenta(2,p2)
      momenta(3,r1)=momenta(3,p1)+momenta(3,p2)
      misspx=0d0
      misspy=0d0
c
c     jacobian factors
c
c     p1x,p1y,p1z,E1 -> misspx, misspy, m12^2, E1^2
c
      jac_loc=1d0/(4d0*dabs(momenta(3,p2)*momenta(0,p1)
     & -momenta(3,p1)*momenta(0,p2)))
c
c     x1,x2 > Etot,pztot
c
      jac_loc=jac_factor*jac_loc*2d0/s

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
c
c     flux
c
      jac_loc=jac_loc/(2d0*x1*x2*s)
      jac=jac*jac_loc
c      write(*,*) 'jac_factor', jac_factor


      return
      end
