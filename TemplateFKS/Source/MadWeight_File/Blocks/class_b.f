      subroutine class_b(p1,p2,r1)
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
      include '../../genps.inc'
      include '../../nexternal.inc'
c
c     arguments
c
      integer p1,p2,r1
c
c     local
c
      double precision jac_loc
      double precision p1z_ti,p1z_E1,jac_factor,dem
      double precision b,c,rho,sqrts
      double precision sol(2),trialpz(2)
      double precision x1s1,x2s1,x1s2,x2s2
      integer index_sol,i
      real rand
c
c     global
c
      double precision momenta(0:3,-max_branch:2*nexternal)  ! records the momenta of external/intermediate legs     (MG order)
      double precision mvir2(-max_branch:2*nexternal)                  ! records the sq invariant masses of intermediate particles (MG order)
      common /to_diagram_kin/ momenta, mvir2
      double precision Etot,pztot,misspx,misspy
      common /to_missingP/Etot,pztot,misspx,misspy
      double precision              S,X1,X2,PSWGT,JAC
      common /PHASESPACE/ S,X1,X2,PSWGT,JAC

c--------
c Begin code 
c--------
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
            call ntuple(rand,0.0,1.0,p1)
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
c
c     fill initial momenta
c
      momenta(0,1)=sqrts*x1/2d0
      momenta(1,1)=0d0
      momenta(2,1)=0d0
      momenta(3,1)=sqrts*x1/2d0
      momenta(0,2)=sqrts*x2/2d0
      momenta(1,2)=0d0
      momenta(2,2)=0d0
      momenta(3,2)=-sqrts*x2/2d0

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
c     flux
c
      jac_loc=jac_loc/(2d0*x1*x2*s)
      jac=jac*jac_loc
c      write(*,*) 'jac_factor', jac_factor
      return
      end
