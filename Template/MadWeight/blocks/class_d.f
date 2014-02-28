      subroutine class_d(x,p1,p2,p3,p4,p5,p6,r1,r3,r2,r4)
c***************************************************************************
c
c
c                                                      local order
c             -.-.-.-.-.-  -------------------- blob       4
c                 r3       -.-.-.-.-.-  ------- blob       3
c                               r1      ------- missing    1
c
c                               r2      -------- missing   2
c                 r4       -.-.-.-.-.-  -------- blob      5
c             -.-.-.-.-.-  --------------------- blob      6
c
c***************************************************************************
      implicit none
      include '../../../SubProcesses/phasespace.inc'
c
c     arguments
c      
      integer p1,p2,p3,p4,p5,p6,r1,r2,r3,r4
c
c     parameters
c
      double precision x(20)
      double precision thres
      parameter (thres=1000d0)
      double complex i_num
      parameter (i_num=(0d0,1d0))
c
c     local
c
      INTEGER IDUM
      DATA IDUM/0/
      SAVE IDUM
      double precision pboost(0:3), CMS_mom(0:3,max_particles)
      double precision Ptot(0:3),PtotCMS(0:3)
      double precision measureLAB, measureCMS
      double precision jac_loc
      integer j,k,MG
      double precision inv_jac,jac_factor
      double precision E1, E2,E1real(4),E2real(4)
      double complex E1c(4), E2c(4)
      logical sol(4)
      real var
      double precision E3,E34,E5,E56
      double precision p1x,p1y,p1z
      double precision p2x,p2y,p2z
      double precision p3x,p3y,p3z
      double precision p34x,p34y,p34z
      double precision p5x,p5y,p5z
      double precision p56x,p56y,p56z
      double precision m34sq,m56sq
      double precision g11,g12,g10,g22,g20,g00
      double precision h11,h12,h10,h22,h20,h00
c
      double precision beta0(4),beta1(4),beta2(4),inv_alpha(4,4)
      double precision p1x_ti,p1x_E1,p1x_E2,p1y_ti,p1y_E1,p1y_E2
      double precision p1z_ti,p1z_E1,p1z_E2,p2z_ti,p2z_E1,p2z_E2
      double precision p2x_ti,p2x_E1,p2x_E2,p2y_ti,p2y_E1,p2y_E2
      double precision det_alpha
      double precision i_part1,i_part2,sqrts
c
c     global
c
      double precision momenta(0:3,-max_branches:2*max_particles)  ! records the momenta of external/intermediate legs     (MG order)
      double precision mvir2(-max_branches:2*max_particles)        ! records the sq invariant masses of intermediate particles (MG order)
      common /to_diagram_kin/ momenta, mvir2
      double precision Etot,pztot,miss_px,miss_py
      common /to_missingP/Etot,pztot,miss_px,miss_py
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
c
c     external
c
      REAL XRAN1
      EXTERNAL XRAN1
      double precision dot
      external dot



      sqrts=dsqrt(s)

      if((Etot+dsqrt(miss_px**2+miss_py**2)).gt.sqrts) then
      jac=-1d0
      return
      endif

c      write(*,*) 'sqrts',sqrts
c      write(*,*) 'p',p3,(momenta(j,p3),j=0,3)
c      write(*,*) 'p',p4,(momenta(j,p4),j=0,3)
c      write(*,*) 'p',p5,(momenta(j,p5),j=0,3)
c      write(*,*) 'p',p6,(momenta(j,p6),j=0,3)
c
c     energies are initialized to a negative real
c     (= no solution status)
c
      momenta(0,p1)=-1D0
      momenta(0,p2)=-1D0

       m34sq=mvir2(p3)+mvir2(p4)
     & +2d0*dot(momenta(0,p3),momenta(0,p4))
       m56sq=mvir2(p5)+mvir2(p6)
     & +2d0*dot(momenta(0,p5),momenta(0,p6))
c
c       write(*,*) 'm3sq ',m3sq
c       write(*,*) 'm4sq ',m4sq
c       write(*,*) 'm5sq ',m5sq
c       write(*,*) 'm6sq ',m6sq
c       write(*,*) 'm34sq ',m34sq
c       write(*,*) 'm56sq ',m56sq
c       write(*,*) '   '
c
c
       E3=momenta(0,p3)
       E5=momenta(0,p5)
       E34=momenta(0,p3)+momenta(0,p4)
       E56=momenta(0,p5)+momenta(0,p6)
c
       p3x=momenta(1,p3)
       p5x=momenta(1,p5)
       p34x=momenta(1,p3)+momenta(1,p4)
       p56x=momenta(1,p5)+momenta(1,p6)
c
       p3y=momenta(2,p3)
       p5y=momenta(2,p5)
       p34y=momenta(2,p3)+momenta(2,p4)
       p56y=momenta(2,p5)+momenta(2,p6)
c
       p3z=momenta(3,p3)
       p5z=momenta(3,p5)
       p34z=momenta(3,p3)+momenta(3,p4)
       p56z=momenta(3,p5)+momenta(3,p6)
c
c---------------------------------------------------------------------------------------------------     
c      the reduced change of variables can be expressed as
c 
c      (new) = alpha * (previous) + beta0 + E1*beta1 + E2* beta2 
c
c      with (previous) = 4x1 = (p1x,p1y,p1z,p2z)^t
c           (new)=4x1 = ( p13^2-mvir13**2,p134^2-mvir134**2,p25^2-mvir25**2,p256^2-mvir256**2 )^t    
c                           (->new is (0,0,0,0)^t !!)           
c           beta0, beta1, beta2 = 4x1
c           alpha = 4x4
c
c      SO, 
c      (previous)= - alpha^(-1) ( beta0 + E1*beta1 + E2* beta2  ) 
c---------------------------------------------------------------------------------------------------
c
       beta0(1)=mvir2(p1)+mvir2(p3)-mvir2(r1)       
       beta0(2)=mvir2(p1)+m34sq-mvir2(r3)  
       beta0(3)=mvir2(p2)+mvir2(p5)
     &       -2d0*miss_px*p5x-2d0*miss_py*p5y-mvir2(r2)        
       beta0(4)=mvir2(p2)+m56sq
     &       -2d0*miss_px*p56x-2d0*miss_py*p56y-mvir2(r4)        
c
       beta1(1)=2d0*E3
       beta1(2)=2d0*E34
c
       beta2(3)=2d0*E5
       beta2(4)=2d0*E56
c
c       do j=1,4
c       write(*,*) 'beta0',j,' : ', beta0(j)
c       if (j.ne.3.and.j.ne.4) write(*,*) 'beta1',j,' : ', beta1(j)
c       if (j.ne.1.and.j.ne.2) write(*,*) 'beta2',j,' : ', beta2(j)
c       enddo

       det_alpha=-16d0*p34z*p3y*p56z*p5x + 16d0*p34y*p3z*p56z*p5x + 
     -   16d0*p34z*p3x*p56z*p5y - 16d0*p34x*p3z*p56z*p5y + 
     -   16d0*p34z*p3y*p56x*p5z - 16d0*p34y*p3z*p56x*p5z - 
     -   16d0*p34z*p3x*p56y*p5z + 16d0*p34x*p3z*p56y*p5z

c        write(*,*) 'det_alpha', det_alpha
c
c
       inv_alpha(1,1)=( 8d0*p34z*(-p56z*p5y + p56y*p5z) )/det_alpha
       inv_alpha(2,1)=( 8d0*p34z*( p56z*p5x -  p56x*p5z) )/det_alpha
       inv_alpha(3,1)=(8d0*(-p34y*p56z*p5x+p34x*p56z*p5y+p34y*p56x*p5z
     &  - p34x*p56y*p5z)  )/det_alpha
       inv_alpha(4,1)=( 8d0*p34z*(p56y*p5x - p56x*p5y)  )/det_alpha
c
       inv_alpha(1,2)=(8d0*p3z*( p56z*p5y - p56y*p5z) )/det_alpha
       inv_alpha(2,2)=(8d0*p3z*(-p56z*p5x + p56x*p5z)   )/det_alpha
       inv_alpha(3,2)=(8d0*(p3y*p56z*p5x-p3x*p56z*p5y -p3y*p56x*p5z
     &  +p3x*p56y*p5z)  )/det_alpha
       inv_alpha(4,2)=(8d0*p3z*(-p56y*p5x + p56x*p5y)  )/det_alpha
c
       inv_alpha(1,3)=(8d0*(-p34z*p3y + p34y*p3z)*p56z  )/det_alpha
       inv_alpha(2,3)=(8d0*p56z*(p34z*p3x - p34x*p3z  ))/det_alpha
       inv_alpha(3,3)=(8d0*(-p34y*p3x + p34x*p3y)*p56z )/det_alpha
       inv_alpha(4,3)=(-8d0*p34z*p3y*p56x + 8d0*p34y*p3z*p56x 
     &   + 8d0*p34z*p3x*p56y - 8d0*p34x*p3z*p56y  )/det_alpha
c
       inv_alpha(1,4)=(8d0*p34z*p3y*p5z-8d0*p34y*p3z*p5z )/det_alpha
       inv_alpha(2,4)=(8d0*(-p34z*p3x + p34x*p3z)*p5z  )/det_alpha
       inv_alpha(3,4)=(8d0*p5z*(p34y*p3x- p34x*p3y ))/det_alpha
       inv_alpha(4,4)=(8d0*(p34z*p3y*p5x - p34y*p3z*p5x - 
     &  p34z*p3x*p5y + p34x*p3z*p5y)  )/det_alpha
c
c---------------------------------------------------------------------------------------------------     
c      Now, the variables p1x,p1y,p1z,p2x,p2y,p2z
c      can be expressed as a combili of E1,E2 
c      + an independant term (=ti)
c
c      for example,
c           p1x = p1x_ti + p1x_E1 * E1 + p1x_E2 * E2       
c---------------------------------------------------------------------------------------------------     
c
       p1x_ti=-inv_alpha(1,1)*beta0(1)-inv_alpha(1,2)*beta0(2)
     & -inv_alpha(1,3)*beta0(3)-inv_alpha(1,4)*beta0(4)
       p1x_E1=-inv_alpha(1,1)*beta1(1)-inv_alpha(1,2)*beta1(2)
       p1x_E2=-inv_alpha(1,3)*beta2(3)-inv_alpha(1,4)*beta2(4)
c
       p1y_ti=-inv_alpha(2,1)*beta0(1)-inv_alpha(2,2)*beta0(2)
     & -inv_alpha(2,3)*beta0(3)-inv_alpha(2,4)*beta0(4)
       p1y_E1=-inv_alpha(2,1)*beta1(1)-inv_alpha(2,2)*beta1(2)
       p1y_E2=-inv_alpha(2,3)*beta2(3)-inv_alpha(2,4)*beta2(4)
c
       p1z_ti=-inv_alpha(3,1)*beta0(1)-inv_alpha(3,2)*beta0(2)
     & -inv_alpha(3,3)*beta0(3)-inv_alpha(3,4)*beta0(4)
       p1z_E1=-inv_alpha(3,1)*beta1(1)-inv_alpha(3,2)*beta1(2)
       p1z_E2=-inv_alpha(3,3)*beta2(3)-inv_alpha(3,4)*beta2(4)
c 
       p2z_ti=-inv_alpha(4,1)*beta0(1)-inv_alpha(4,2)*beta0(2)
     & -inv_alpha(4,3)*beta0(3)-inv_alpha(4,4)*beta0(4)
       p2z_E1=-inv_alpha(4,1)*beta1(1)-inv_alpha(4,2)*beta1(2)
       p2z_E2=-inv_alpha(4,3)*beta2(3)-inv_alpha(4,4)*beta2(4)
c

c       write(*,*) 'p1x_ti', p1x_ti
c       write(*,*) 'p1x_E1', p1x_E1
c       write(*,*) 'p1x_E2', p1x_E2
c
c       write(*,*) 'p1y_ti', p1y_ti
c       write(*,*) 'p1y_E1', p1y_E1
c       write(*,*) 'p1y_E2', p1y_E2
c
c
c       write(*,*) 'p1z_ti', p1z_ti
c       write(*,*) 'p1z_E1', p1z_E1
c       write(*,*) 'p1z_E2', p1z_E2

c       write(*,*) 'p2z_ti', p2z_ti
c       write(*,*) 'p2z_E1', p2z_E1
c       write(*,*) 'p2z_E2', p2z_E2
c
       p2x_ti=miss_px-p1x_ti
       p2x_E1=-p1x_E1
       p2x_E2=-p1x_E2
c
       p2y_ti=miss_py-p1y_ti
       p2y_E1=-p1y_E1
       p2y_E2=-p1y_E2
c
c---------------------------------------------------------------------------------------------------     
c      define the coefficients of the two quadratic equations
c
c      E1^2-p1x^2-p1y^2-p1z^2-m1^2=0   (1)
c      E2^2-p2x^2-p2y^2-p2z^2-m2^2=0   (2)
c
c      <=>
c
c      g11 E1^2 +g22 E2^2 + g12 E1*E2 + g10 E1 + g20 *E2 + g00 = 0   (1)
c      h11 E1^2 +h22 E2^2 + h12 E1*E2 + h10 E1 + h20 *E2 + h00 = 0   (2)
c---------------------------------------------------------------------------------------------------     
c
       g11=1d0-p1x_E1**2-p1y_E1**2-p1z_E1**2
       g22=-p1x_E2**2-p1y_E2**2-p1z_E2**2
       g12=-2d0*p1x_E1*p1x_E2-2d0*p1y_E1*p1y_E2-2d0*p1z_E1*p1z_E2
       g10=-2d0*p1x_ti*p1x_E1-2d0*p1y_ti*p1y_E1-2d0*p1z_ti*p1z_E1
       g20=-2d0*p1x_ti*p1x_E2-2d0*p1y_ti*p1y_E2-2d0*p1z_ti*p1z_E2
       g00=-mvir2(p1)-p1x_ti**2-p1y_ti**2-p1z_ti**2
c
       h11=-p2x_E1**2-p2y_E1**2-p2z_E1**2
       h22=1d0-p2x_E2**2-p2y_E2**2-p2z_E2**2
       h12=-2d0*p2x_E1*p2x_E2-2d0*p2y_E1*p2y_E2-2d0*p2z_E1*p2z_E2
       h10=-2d0*p2x_ti*p2x_E1-2d0*p2y_ti*p2y_E1-2d0*p2z_ti*p2z_E1
       h20=-2d0*p2x_ti*p2x_E2-2d0*p2y_ti*p2y_E2-2d0*p2z_ti*p2z_E2
       h00=-mvir2(p2)-p2x_ti**2-p2y_ti**2-p2z_ti**2

c       write(*,*) 'g11', g11
c       write(*,*) 'g22', g22
c       write(*,*) 'g12', g12
c       write(*,*) 'g10', g10
c       write(*,*) 'g20', g20
c       write(*,*) 'g00', g00


c       write(*,*) 'h11', h11
c       write(*,*) 'h22', h22
c       write(*,*) 'h12', h12
c       write(*,*) 'h10', h10
c       write(*,*) 'h20', h20
c       write(*,*) 'h00', h00
c       pause
c
       call solve_2quadeqs(g11,g22,g12,g10,g20,g00,
     & h11,h22,h12,h10,h20,h00,E1c,E2c,sol)

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
        if (sol(j)) then
          i_part1=dble((E1c(j)-dconjg(E1c(j)))/(2d0*i_num))
          i_part2=dble((E2c(j)-dconjg(E2c(j)))/(2d0*i_num))

c      write(*,*) 'sol ',j
c      write(*,*) 'i_num ',i_num
c      write(*,*) 'dble ',dble(E1c(j)),dble(E2c(j))
c      write(*,*) 'imag ',i_part1, i_part2


          if (dabs(dble(E1c(j))).gt.1.0D+5*dabs(i_part1).and.
     & dabs(dble(E2c(j))).gt.1.0D+5*dabs(i_part2 ).and.
     &      dble(E1c(j)).gt.0d0.and.dble(E2c(j)).gt.0d0) then

c       write(*,*) 'we have a real number'   
c
c       determine the fraction of initial energies
c

      p1z=p1z_ti+p1z_E1*dble(E1c(j))+p1z_E2*dble(E2c(j))
      p2z=p2z_ti+p2z_E1*dble(E1c(j))+p2z_E2*dble(E2c(j))
 
      x1=((Etot+dble(E1c(j))+dble(E2c(j)))+(pztot+P1z+P2z))/sqrts
      x2=((Etot+dble(E1c(j))+dble(E2c(j)))-(pztot+P1z+P2z))/sqrts

c      write(*,*) "x1,x2",x1,x2

      if(dabs(x1-0.5d0).lt.0.5d0.and.dabs(x2-0.5d0).lt.0.5d0)then
            k=k+1 ! count the number of good sol
            E1real(k)=dble(E1c(j))
            E2real(k)=dble(E2c(j))
c            write(*,*) 'sol',k
c            write(*,*) E1real(k)
c            write(*,*) E2real(k)
c      pause
          endif
          endif
        endif
      enddo

      IDUM=0
      if (k.eq.0) then 
        jac=-1d0
      return
      elseif (k.eq.1) then
        E1=E1real(1)
        E2=E2real(1)
        jac_factor=1d0
      elseif (k.eq.2) then
        jac_factor=2d0
c        call ntuple(var,0.0,1.0,15)
        var=xran1(IDUM)
        if(var.gt.0.5) then
          E1=E1real(1)
          E2=E2real(1)
        else
          E1=E1real(2)
          E2=E2real(2)
        endif
      elseif (k.eq.3) then
        jac_factor=3d0
c        call ntuple(var,0.0,1.0,max_particles)
        var=xran1(IDUM)
        if(var.lt.1.0/3.0) then
          E1=E1real(1)
          E2=E2real(1)
        elseif(var.ge.1.0/3.0.and.var.lt.2.0/3.0) then
          E1=E1real(2)
          E2=E2real(2)
        else
          E1=E1real(3)
          E2=E2real(3)
        endif
      elseif (k.eq.4) then
        jac_factor=4d0
c        call ntuple(var,0.0,1.0,max_particles)
        var=xran1(IDUM)
        if(var.lt.1.0/4.0) then
          E1=E1real(1)
          E2=E2real(1)
        elseif(var.ge.1.0/4.0.and.var.lt.1.0/2.0) then
          E1=E1real(2)
          E2=E2real(2)
        elseif(var.ge.1.0/2.0.and.var.lt.3.0/4.0) then
          E1=E1real(3)
          E2=E2real(3)
         else
          E1=E1real(4)
          E2=E2real(4)
        endif
      endif

c
c     fix other components of neutrino momenta
c
      p1x=p1x_ti+p1x_E1*E1+p1x_E2*E2
      p1y=p1y_ti+p1y_E1*E1+p1y_E2*E2
      p1z=p1z_ti+p1z_E1*E1+p1z_E2*E2
      p2x=p2x_ti+p2x_E1*E1+p2x_E2*E2
      p2y=p2y_ti+p2y_E1*E1+p2y_E2*E2
      p2z=p2z_ti+p2z_E1*E1+p2z_E2*E2
    
      momenta(0,p1)=E1
      momenta(1,p1)=p1x
      momenta(2,p1)=p1y
      momenta(3,p1)=p1z

      momenta(0,p2)=E2
      momenta(1,p2)=p2x
      momenta(2,p2)=p2y
      momenta(3,p2)=p2z

c
c       define jac of the transformation    [inv_jac]=E^8, 
c       include jac factor for change x1,x2 -> Etot, pztot
c
      inv_jac=    -32d0*(E3*(E5*
     -         (p34z*(p1y*p2z*p56x - p1x*p2z*p56y - p1y*p2x*p56z + 
     -              p1x*p2y*p56z) + 
     -           p1z*(-(p2z*p34y*p56x) + p2z*p34x*p56y - 
     -              p2y*p34x*p56z + p2x*p34y*p56z)) + 
     -        (E56*p2z - E2*p56z)*
     -         (p1z*p34y*p5x - p1y*p34z*p5x - p1z*p34x*p5y + 
     -           p1x*p34z*p5y) + 
     -        (E56*(p1z*p2y*p34x - p1z*p2x*p34y + p1y*p2x*p34z - 
     -              p1x*p2y*p34z) + 
     -           E2*(p1z*p34y*p56x - p1y*p34z*p56x - p1z*p34x*p56y + 
     -              p1x*p34z*p56y))*p5z) + 
     -     E34*(E5*p2z*(p1z*p3y*p56x - p1y*p3z*p56x - p1z*p3x*p56y + 
     -           p1x*p3z*p56y) + 
     -        E5*(p1z*p2y*p3x - p1z*p2x*p3y + p1y*p2x*p3z - 
     -           p1x*p2y*p3z)*p56z - 
     -        (E56*p2z - E2*p56z)*
     -         (p1z*p3y*p5x - p1y*p3z*p5x - p1z*p3x*p5y + p1x*p3z*p5y)
     -          - (E56*(p1z*p2y*p3x - p1z*p2x*p3y + p1y*p2x*p3z - 
     -              p1x*p2y*p3z) + 
     -           E2*(p1z*p3y*p56x - p1y*p3z*p56x - p1z*p3x*p56y + 
     -              p1x*p3z*p56y))*p5z) + 
     -     E1*(E5*(p2z*(-(p34z*p3y*p56x) + p34y*p3z*p56x + 
     -              p34z*p3x*p56y - p34x*p3z*p56y) + 
     -           (-(p2y*p34z*p3x) + p2x*p34z*p3y + p2y*p34x*p3z - 
     -              p2x*p34y*p3z)*p56z) + 
     -        (E56*p2z - E2*p56z)*
     -         (p34z*p3y*p5x - p34y*p3z*p5x - p34z*p3x*p5y + 
     -           p34x*p3z*p5y) + 
     -        (E56*(p2y*p34z*p3x - p2x*p34z*p3y - p2y*p34x*p3z + 
     -              p2x*p34y*p3z) + 
     -           E2*(p34z*p3y*p56x - p34y*p3z*p56x - p34z*p3x*p56y + 
     -              p34x*p3z*p56y))*p5z))*s
c
       if (dabs(inv_jac).gt.thres) then
c
c      dimension of jac_loc is
c
       jac_loc=1d0/dabs(inv_jac)*jac_factor
       else
       write(*,*) 'warning: jac^-1 is close to zero'
         jac=-1d0
       return
       endif

c    fill intermediate momenta
      do j=0,3
        momenta(j,r1)=momenta(j,p1)+momenta(j,p3)
        momenta(j,r2)=momenta(j,p2)+momenta(j,p5)
        momenta(j,r3)=momenta(j,r1)+momenta(j,p4)
        momenta(j,r4)=momenta(j,r2)+momenta(j,p6)
      enddo
      miss_px=0d0
      miss_py=0d0
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
c      write(*,*) Ptot

c     Then calculate the momenta in the CMS frame
      pboost(1)=-pboost(1)
      pboost(2)=-pboost(2)
      pboost(3)=0d0
       do j=3,nexternal
c         write(*,*) "p",j,momenta(0,j), momenta(1,j),momenta(2,j),momenta(3,j)
         call boostx(momenta(0,j),pboost,CMS_mom(0,j))
c         write(*,*) "p",j,CMS_mom(0,j),CMS_mom(1,j),CMS_mom(2,j),CMS_mom(3,j)
       enddo
       call boostx(Ptot,pboost,PtotCMS)
c      write(*,*) PtotCMS

c     Evaluate the initial momenta in the CMS frame
      x1=(PtotCMS(0)+PtotCMS(3))/sqrts
      x2=(PtotCMS(0)-PtotCMS(3))/sqrts

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
c         write(*,*) "pCMS",MG,CMS_mom(0,MG),CMS_mom(1,MG),CMS_mom(2,MG),CMS_mom(3,MG)
c         write(*,*) "pLAB",MG,momenta(0,MG),momenta(1,MG),momenta(2,MG),momenta(3,MG)
c         write(*,*) ""
       enddo

c      write(*,*) "mW ", dsqrt(dot(momenta(0,r1),momenta(0,r1)))
c      write(*,*) "mW ", dsqrt(dot(momenta(0,r2),momenta(0,r2)))
c      write(*,*) "mt ", dsqrt(dot(momenta(0,r3),momenta(0,r3)))
c      write(*,*) "mt ", dsqrt(dot(momenta(0,r4),momenta(0,r4)))

      measureCMS=1d0
       do j=3,nexternal-num_inv
         MG=inv_matching_type_part(j)
         measureCMS=measureCMS*dsqrt(CMS_mom(1,MG)**2+CMS_mom(2,MG)**2)
       enddo
      jac=jac*measureCMS/measureLAB
      endif
c
c     flux factor
c
      jac_loc=jac_loc/(2d0*S*x1*x2)  ! flux 
      jac=jac*jac_loc

c
c     fill intermediate momenta
c
c      do j=0,3
c      momenta(j,r1)= momenta(j,p1)+ momenta(j,p3)
c      momenta(j,r3)= momenta(j,r1)+ momenta(j,p4)
c      momenta(j,r2)= momenta(j,p2)+ momenta(j,p5)
c      momenta(j,r4)= momenta(j,r2)+ momenta(j,p6)
c      enddo
c      write(*,*) 'k',k
      return
      end
