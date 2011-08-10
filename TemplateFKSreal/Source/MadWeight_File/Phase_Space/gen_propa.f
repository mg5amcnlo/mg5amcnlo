      subroutine generate_propa(x,n_var)
      implicit none
c
      include 'genps.inc'
      include 'nexternal.inc'
      include 'data.inc' 
      include 'coupl.inc' 
c
c     argument
c
      double precision x(20)
      integer n_var
c
c     parameter
c
      double precision zero
      parameter (zero=0d0)
c
c     local
c
      double precision upper_bound,lower_bound,jac_temp,y,pole,gam
      integer i,j
      double precision pmass(-max_branch:0,lmaxconfigs)
      double precision pwidth(-max_branch:0,lmaxconfigs)
      double precision pow(-max_branch:0,lmaxconfigs)
c
c     global
c
      double precision momenta(0:3,-max_branch:2*nexternal)  ! records the momenta of external/intermediate legs     (MG order)
      double precision mvir2(-max_branch:2*nexternal)        ! records the sq invariant masses of intermediate particles (MG order)
      common /to_diagram_kin/ momenta, mvir2
      double precision pmass2(nexternal)     ! records the pole mass of any particle of the diagram  (MG order)
      common / to_mass/pmass2
      double precision              S,X1,X2,PSWGT,JAC
      common /PHASESPACE/ S,X1,X2,PSWGT,JAC

      integer config,perm_pos
      common /to_config/config,perm_pos
c---
c Begin code
c---
      include 'props.inc'
      do i=1,num_propa(config)
c        write(*,*) 'propa_nb(',i,'%)',(propa_nb(i,j),j=1,4)
c
c     upper bound
c
        if (propa_max(i,1,config).lt.0) then
          upper_bound=dsqrt(mvir2(propa_max(i,1,config)))
          do j=2,max_branch
            if (propa_max(i,j,config).lt.0) then
             upper_bound=upper_bound-dsqrt(mvir2(propa_max(i,j,config)))
            elseif (propa_max(i,j,config).gt.0) then
             upper_bound=upper_bound-pmass2(propa_max(i,j,config))
            elseif (propa_max(i,j,config).eq.0) then
              upper_bound=upper_bound**2
              goto 13
            endif
          enddo
        else
          upper_bound=s
c          write(*,*) 's',s
        endif
13      continue
c
c     lower bound
c
        if (propa_min(i,1,config).gt.0) then
          lower_bound=pmass2(propa_min(i,1,config))
          do j=2,max_branch
            if (propa_min(i,j,config).gt.0) then
              lower_bound=lower_bound+pmass2(propa_min(i,j,config))
            elseif (propa_min(i,j,config).lt.0) then
            lower_bound=lower_bound+dsqrt(mvir2(propa_min(i,j,config)))
            elseif (propa_min(i,j,config).eq.0) then
              lower_bound=lower_bound**2
              goto 14
            endif
          enddo
        else
          lower_bound=0d0
c          write(*,*) 's',s
        endif

14      continue
c        write(*,*) 'upper_bound', dsqrt(upper_bound)
c        write(*,*) 'lower_bound', dsqrt(lower_bound)
c         write(*,*) 'pmass(',propa_cont(i,config),',',config,')',pmass(propa_cont(i,1),1)
c        write(*,*) 'wmass',wmass
c        write(*,*) 'tmass',tmass
        pole=(pmass(propa_cont(i,config),1)**2-lower_bound)/
     & (upper_bound-lower_bound)
        gam=(pwidth(propa_cont(i,config),1)*
     & pmass(propa_cont(i,config),1))/(upper_bound-lower_bound)
        n_var=n_var+1
c         write(*,*) 'inputs',pole,gam,x(n_var),upper_bound-lower_bound
c        pause      
        call transpole(pole,gam,x(n_var),y,jac)
c        write(*,*) 'outputs',y,jac
        jac=jac*(upper_bound-lower_bound)
c        jac=jac*upper_bound
        mvir2(propa_cont(i,config))=y*(upper_bound-lower_bound)
     & +lower_bound
c        write(*,*) 'mvir2',propa_cont(i,1),dsqrt(mvir2(propa_cont(i,1)))
c        pause
      enddo

      return
      end
