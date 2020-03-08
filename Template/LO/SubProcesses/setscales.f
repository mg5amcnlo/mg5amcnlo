      subroutine set_ren_scale(P,rscale)
c----------------------------------------------------------------------
c     This is the USER-FUNCTION to calculate the renormalization
c     scale on event-by-event basis.
c----------------------------------------------------------------------      
      implicit none
      real*8   alphas
      external alphas
c
c     INCLUDE and COMMON
c
      include 'genps.inc'
      include 'nexternal.inc'
      include 'coupl.inc'

      integer i
      include 'maxamps.inc'
      integer idup(nexternal,maxproc,maxsproc)
      integer mothup(2,nexternal)
      integer icolup(2,nexternal,maxflow,maxsproc)
      include 'leshouche.inc'
      include 'run.inc'

      double precision pmass(nexternal)
      common/to_mass/  pmass

      real*8 xptj,xptb,xpta,xptl,xmtc
      real*8 xetamin,xqcut,deltaeta
      common /to_specxpt/xptj,xptb,xpta,xptl,xmtc,xetamin,xqcut,deltaeta

c
c     ARGUMENTS
c      
      REAL*8 P(0:3,nexternal)
      REAL*8 rscale
c
c     EXTERNAL
c
      REAL*8 R2,DOT,ET,ETA,DJ,SumDot,PT

c----------
c     start
c----------

      if (dynamical_scale_choice.eq.-1) then
c         Cluster external states until reducing the system to a 2->2 topology whose transverse mass is used for setting the scale.
c         This is not done in this file due to the clustering.
         rscale=0d0
      elseif(dynamical_scale_choice.eq.1) then
c         Total transverse energy of the event.         
          rscale=0d0
          do i=3,nexternal
             rscale=rscale+et(P(0,i))
          enddo      
      elseif(dynamical_scale_choice.eq.2) then
c         sum of the transverse mass
c         m^2+pt^2=p(0)^2-p(3)^2=(p(0)+p(3))*(p(0)-p(3))
          rscale=0d0
          do i=3,nexternal
            rscale=rscale+dsqrt(max(0d0,(P(0,i)+P(3,i))*(P(0,i)-P(3,i))))
          enddo
          rscale=rscale
      elseif(dynamical_scale_choice.eq.3) then
c         sum of the transverse mass divide by 2
c         m^2+pt^2=p(0)^2-p(3)^2=(p(0)+p(3))*(p(0)-p(3))
          rscale=0d0
          do i=3,nexternal
            rscale=rscale+dsqrt(max(0d0,(P(0,i)+P(3,i))*(P(0,i)-P(3,i))))
          enddo
          rscale=rscale/2d0
      elseif(dynamical_scale_choice.eq.4) then
c         \sqrt(s), partonic energy
          rscale=dsqrt(max(0d0,2d0*dot(P(0,1),P(0,2))))
      elseif(dynamical_scale_choice.eq.5) then
c         \decaying particle mass, for decays
          rscale=dsqrt(max(0d0,dot(P(0,1),P(0,1))))
      elseif(dynamical_scale_choice.eq.0) then
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cc      USER-DEFINED SCALE: ENTER YOUR CODE HERE                                 cc
cc      to use this code you must set                                            cc
cc                 dynamical_scale_choice = 0                                    cc
cc      in the run_card (run_card.dat)                                           cc
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
         write(*,*) "User-defined scale not set"
         stop 21
         rscale = 0
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cc      USER-DEFINED SCALE: END OF USER CODE                                     cc
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      else
        write(*,*)'Unknown option in scale_global_reference',dynamical_scale_choice
        stop
      endif
      rscale = scalefact*rscale
      return
      end


      subroutine set_fac_scale(P,q2factorization)
c----------------------------------------------------------------------
c     This is the USER-FUNCTION to calculate the factorization 
c     scales^2 on event-by-event basis.
c----------------------------------------------------------------------      
      implicit none

c     INCLUDE and COMMON
c
      include 'genps.inc'
      include 'nexternal.inc'
      include 'coupl.inc'
      include 'run.inc'
c--masses and poles
c
c     ARGUMENTS
c      
      REAL*8 P(0:3,nexternal)
      real*8 q2factorization(2)
c
c     EXTERNAL
c
      REAL*8 R2,DOT,ET,ETA,DJ,SumDot,PT
c
c     LOCAL
c
      integer i
      logical first
      data first/.true./

c----------
c     start
c----------
      
      if (dynamical_scale_choice.eq.-1) then
c         Cluster external states until reducing the system to a 2->2 topology whose transverse mass is used for setting the scale.
c         This is not done in this file due to the clustering.
         q2factorization(1)=0d0          !factorization scale**2 for pdf1
         q2factorization(2)=0d0          !factorization scale**2 for pdf2
      elseif(dynamical_scale_choice.eq.0) then
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cc      USER DEFINE SCALE: ENTER YOUR CODE HERE                                  cc
cc      to use this code you need to set                                         cc
cc                 dymamical_scale_choice to 0 in the run_card                   cc
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c         default: use the renormalization scale
          call set_ren_scale(P,q2factorization(1))
          q2factorization(1)=q2factorization(1)**2
          q2factorization(2)=q2factorization(1)   !factorization scale**2 for pdf2

c
c-some examples of dynamical scales
c

c---------------------------------------
c-- total transverse energy of the event 
c---------------------------------------
c     q2factorization(1)=0d0
c     do i=3,nexternal
c      q2factorization(1)= q2factorization(1)+et(P(0,i))**2
c     enddo
c     q2factorization(2)=q2factorization(1)  

c--------------------------------------
c-- scale^2 = \sum_i  (pt_i^2+m_i^2)  
c--------------------------------------
c     q2factorization(1)=0d0
c     do i=3,nexternal
c      q2factorization(1)=q2factorization(1)+pt(P(0,i))**2+dot(p(0,i),p(0,i))
c     enddo
c     q2factorization(2)=q2factorization(1)  

c--------------------------------------
c-- \sqrt(s): partonic energy
c--------------------------------------
c     q2factorization(1)=2d0*dot(P(0,1),P(0,2))
c     q2factorization(2)=q2factorization(1)  



ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cc      USER DEFINE SCALE: END of USER CODE                                      cc
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      else
          call set_ren_scale(P,q2factorization(1))
          q2factorization(1)=q2factorization(1)**2
          q2factorization(2)=q2factorization(1)   !factorization scale**2 for pdf2
      endif


      

      
      return
      end


