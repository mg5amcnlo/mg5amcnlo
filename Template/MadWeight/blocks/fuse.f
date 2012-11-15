      subroutine fuse(p1,p2,r1)
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     This subroutine returns the momentum r1=p1+p2
c
c               * P2 (visible)
c              *
c     *********
c      MV_r1   * 
c               * P1 (visible)
c
c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      implicit none
      include '../../../SubProcesses/phasespace.inc'
c
c     argument
c
      integer p1,p2,r1
c
c     local
c
      integer nu
c
c     enternal
c
      double precision dot
      external dot
c
c     global
c
      double precision momenta(0:3,-max_branches:2*max_particles)  ! records the momenta of external/intermediate legs     (MG order)
      double precision mvir2(-max_branches:2*max_particles)        ! records the sq invariant masses of intermediate particles (MG order)
      common /to_diagram_kin/ momenta, mvir2
      double precision pmass(max_particles)     ! records the pole mass of any particle of the diagram  (MG order)
      common / to_mass/pmass
c---
c Begin code
c---
      do nu=0,3
        momenta(nu,r1)=momenta(nu,p1)+momenta(nu,p2)
      enddo

      mvir2(r1)=dot(momenta(0,r1),momenta(0,r1))

      end
