      subroutine set_ren_scale(P,rscale)
c----------------------------------------------------------------------
c     This is the USER-FUNCTION to calculate the renormalization
c     scale on event-by-event basis.
c----------------------------------------------------------------------      
      implicit none
      integer    maxexternal
      parameter (maxexternal=15)
      real*8 Pi
      parameter( Pi = 3.14159265358979323846d0 )
      real*8   alphas
      external alphas
c
c     INCLUDE and COMMON
c
      include 'genps.inc'
      include "nexternal.inc"
      include 'coupl.inc'

      integer    maxflow, i
      parameter (maxflow=999)
      integer idup(nexternal,maxproc)
      integer mothup(2,nexternal,maxproc)
      integer icolup(2,nexternal,maxflow)
      include 'leshouche.inc'
      include 'run.inc'

      double precision pmass(nexternal)
      common/to_mass/  pmass

      real*8 xptj,xptb,xpta,xptl
      real*8 xetamin,xqcut,deltaeta
      common /to_specxpt/xptj,xptb,xpta,xptl,xetamin,xqcut,deltaeta

      logical first
      data first/.true./

c
c     ARGUMENTS
c      
      REAL*8 P(0:3,maxexternal)
      REAL*8 rscale
c
c     EXTERNAL
c
      REAL*8 R2,DOT,ET,ETA,DJ,SumDot,PT

c----------
c     start
c----------

      if(fixed_ren_scale) then
         rscale=scale
      else

      if(ickkw.gt.0.or.xqcut.gt.0)then
c     alpha_s reweighted due to clustering in reweight.f
         rscale=scale
         return
      endif

      rscale=0d0

      if(first) then
         write(*,*) 'Using event- by event '//
     &        'renormalization/factorization scale:'
         write(*,*) 'scalefact^2*(Max of squared masses '//
     &        'of final-state particles + '
         write(*,*) '             sum of pT^2 for jets and '//
     &        'massless particles)'
      endif
      do i=3,nexternal
         rscale=max(rscale,pmass(i)**2)
      enddo
      do i=3,nexternal
         if(iabs(idup(i,1)).le.5.or.idup(i,1).eq.21.or.
     &                                 pmass(i).eq.0d0)then
            rscale=rscale+pt(p(0,i))**2
         endif
      enddo

      rscale=sqrt(rscale)

      if(first)then
         write(*,*) 'mu_R = ',scalefact,' * ',rscale
         first=.false.
      endif

      rscale=rscale*scalefact

c
c-some examples of dynamical scales
c

c---------------------------------------
c-- total transverse energy of the event 
c---------------------------------------
c     scale=0d0
c     do i=3,nexternal
c      scale=scale+et(P(0,i))
c     enddo

c--------------------------------------
c-- scale^2 = \sum_i  (pt_i^2+m_i^2)  
c--------------------------------------
c     scale=0d0
c     do i=3,nexternal
c      scale=scale+pt(P(0,i))**2+dot(p(0,i),p(0,i))
c     enddo
c     scale=dsqrt(scale)

c--------------------------------------
c-- \sqrt(s): partonic energy
c--------------------------------------
c     scale=dsqrt(2d0*dot(P(0,1),P(0,2)))

      endif                     ! fixed_ren_scale

      G = SQRT(4d0*PI*ALPHAS(rscale))
      
      return
      end


      subroutine set_fac_scale(P,q2fact)
c----------------------------------------------------------------------
c     This is the USER-FUNCTION to calculate the factorization 
c     scales^2 on event-by-event basis.
c----------------------------------------------------------------------      
      implicit none
      integer    maxexternal
      parameter (maxexternal=15)
c
c     INCLUDE and COMMON
c
      include 'genps.inc'
      include "nexternal.inc"
      include 'coupl.inc'
      include 'message.inc'
c--masses and poles
c
c     ARGUMENTS
c      
      REAL*8 P(0:3,maxexternal)
      real*8 q2fact(2)
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
      
      
      q2fact(1)=0d0             !factorization scale**2 for pdf1

      call set_ren_scale(P,q2fact(1))

      if(first)then
        write(*,*) 'mu_F = mu_R'
        first=.false.
      endif
      
      if (btest(mlevel,3)) then
        write(*,*)'setscales.f: Setting fact scale to ',q2fact(1)
      endif

      q2fact(1)=q2fact(1)**2

      q2fact(2)=q2fact(1)       !factorization scale**2 for pdf2
      

c
c-some examples of dynamical scales
c

c---------------------------------------
c-- total transverse energy of the event 
c---------------------------------------
c     q2fact(1)=0d0
c     do i=3,nexternal
c      q2fact(1)= q2fact(1)+et(P(0,i))**2
c     enddo
c     q2fact(2)=q2fact(1)  

c--------------------------------------
c-- scale^2 = \sum_i  (pt_i^2+m_i^2)  
c--------------------------------------
c     q2fact(1)=0d0
c     do i=3,nexternal
c      q2fact(1)=q2fact(1)+pt(P(0,i))**2+dot(p(0,i),p(0,i))
c     enddo
c     q2fact(2)=q2fact(1)  

c--------------------------------------
c-- \sqrt(s): partonic energy
c--------------------------------------
c     q2fact(1)=2d0*dot(P(0,1),P(0,2))
c     q2fact(2)=q2fact(1)  

      
      return
      end




      subroutine set_alphaS(P)
c     This subroutine sets the value of the strong coupling.
c     It updates the scales in run.inc and coupling itself in coupl.inc.
c     For an event-by-event scale choice, this should in general only
c     be called if the event passes the cuts.
      implicit none
      include "genps.inc"
      include "nexternal.inc"
      REAL*8 P(0:3,nexternal),rwgt
      integer i,j
      include "run.inc"
      include "coupl.inc"
      include "q_es.inc"

c put momenta in common block for couplings.f
      double precision PP(0:3,max_particles)
      COMMON /MOMENTA_PP/PP

      logical firsttime,firsttime2
      data firsttime,firsttime2 /.true.,.true./

c After recomputing alphaS, be sure to set 'calculatedBorn' to false
      double precision hel_fac
      logical calculatedBorn
      integer get_hel,skip
      common/cBorn/hel_fac,calculatedBorn,get_hel,skip


      if (firsttime) then
         firsttime=.false.
c
c     Set the strong coupling
c
         call set_ren_scale(P,scale)
c
c     Check that the user funtions for setting the scales
c     have been edited if the choice of an event-by-event
c     scale choice has been made 
c
         if(.not.fixed_ren_scale) then
            if(scale.eq.0d0) then
               write(6,*) 
               write(6,*) '* >>>>>>>>>ERROR<<<<<<<<<<<<<<<<<<<<<<<*'
               write(6,*) ' Dynamical renormalization scale choice '
               write(6,*) ' selected but user subroutine' 
               write(6,*) ' set_ren_scale not edited in file:setpara.f'
               write(6,*) ' Switching to a fixed_ren_scale choice'
               write(6,*) ' with scale=zmass'
               scale=91.2d0
               write(6,*) 'scale=',scale
               fixed_ren_scale=.true.
               call set_ren_scale(P,scale)
            endif
         endif
         
         if(.not.fixed_fac_scale) then
            call set_fac_scale(P,q2fact)
            if(q2fact(1).eq.0d0.or.q2fact(2).eq.0d0) then
               write(6,*) 
               write(6,*) '* >>>>>>>>>ERROR<<<<<<<<<<<<<<<<<<<<<<<*'
               write(6,*) ' Dynamical renormalization scale choice '
               write(6,*) ' selected but user subroutine' 
               write(6,*) ' set_fac_scale not edited in file:setpara.f'
               write(6,*) ' Switching to a fixed_fac_scale choice'
               write(6,*) ' with q2fact(i)=zmass**2'
               fixed_fac_scale=.true.
               q2fact(1)=91.2d0**2
               q2fact(2)=91.2d0**2
               write(6,*) 'scales=',q2fact(1),q2fact(2)
            endif
         endif

         if(fixed_ren_scale) then
            call setpara('param_card.dat',.false.)
         endif

c     Put momenta in the common block to zero to start
         do i=0,3
            do j=1,max_particles
               pp(i,j) = 0d0
            enddo
         enddo
      endif
      
c
c     Here we reset factorization and renormalization
c     scales on an event-by-event basis
c
      if(.not.fixed_ren_scale) then
         call set_ren_scale(P,scale)
      endif

      if(.not.fixed_fac_scale) then
         call set_fac_scale(P,q2fact)
      endif
c

c
c Set the Ellis-Sexton scale (should be set before the call to
c setpara(), because some of the R2 coupling constant depend on
c this scale.
      QES2=q2fact(1)


C...Set strong couplings if event passed cuts
      if(.not.fixed_ren_scale.or..not.fixed_couplings) then
         if (.not.fixed_couplings)then
            do i=0,3
               do j=1,nexternal
                  PP(i,j)=p(i,j)
               enddo
            enddo
         endif
         call setpara('param_card.dat',.false.)
      endif

      IF (FIRSTTIME2) THEN
         FIRSTTIME2=.FALSE.
         write(6,*) 'alpha_s for scale ',scale,' is ',
     &        G**2/(16d0*atan(1d0))
      ENDIF


c Reset calculatedBorn, because the couplings might have been changed.
c This is needed in particular for the MC events, because there the
c coupling should be set according to the real-emission kinematics,
c even when computing the Born matrix elements.
      calculatedBorn=.false.

      return
      end
