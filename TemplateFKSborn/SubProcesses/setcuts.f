      SUBROUTINE SETCUTS
C**************************************************************************
C     SET THE CUTS 
C**************************************************************************
      IMPLICIT NONE
c
c     INCLUDE
c
      include 'genps.inc'
      include "nexternal.inc"
      include 'coupl.inc'
      include 'run.inc'
      include 'cuts.inc'
c
c     Constants
c
      double precision zero
      parameter       (ZERO = 0d0)
      real*8 Pi
      parameter( Pi = 3.14159265358979323846d0 )
      integer    lun
      parameter (lun=22)
c
c     LOCAL
c
      integer i,j
      integer icollider,detail_level
      logical  do_cuts(nexternal)
      integer ncheck
      logical done,fopened
C     
C     GLOBAL
C
c--masses and poles
      double precision pmass(nexternal)
      common/to_mass/  pmass
      double precision      spole(maxinvar),swidth(maxinvar),bwjac
      common/to_brietwigner/spole          ,swidth          ,bwjac
c--cuts
      double precision etmin(nincoming+1:nexternal)
      double precision etamax(nincoming+1:nexternal)
      double precision emin(nincoming+1:nexternal)
      double precision r2min(nincoming+1:nexternal,nincoming+1:nexternal)
      double precision s_min(nexternal,nexternal)
      double precision etmax(nincoming+1:nexternal)
      double precision etamin(nincoming+1:nexternal)
      double precision emax(nincoming+1:nexternal)
      double precision r2max(nincoming+1:nexternal,nincoming+1:nexternal)
      double precision s_max(nexternal,nexternal)
      common/to_cuts/  etmin, emin, etamax, r2min, s_min,
     $     etmax, emax, etamin, r2max, s_max

      double precision ptjmin4(4),ptjmax4(4),htjmin4(2:4),htjmax4(2:4)
      logical jetor
      common/to_jet_cuts/ ptjmin4,ptjmax4,htjmin4,htjmax4,jetor

c
c     les houches accord stuff to identify neutrinos
c
      integer    maxflow
      parameter (maxflow=999)
      integer idup(nexternal,maxproc)
      integer mothup(2,nexternal,maxproc)
      integer icolup(2,nexternal,maxflow)
      include 'leshouche.inc'
C
C $B$ TO_SPECISA $B$ !this is a tag
      LOGICAL  IS_A_J(NEXTERNAL),IS_A_L(NEXTERNAL)
      LOGICAL  IS_A_B(NEXTERNAL),IS_A_A(NEXTERNAL)
      LOGICAL  IS_A_NU(NEXTERNAL),IS_HEAVY(NEXTERNAL)
      COMMON /TO_SPECISA/IS_A_J,IS_A_A,IS_A_L,IS_A_B,IS_A_NU,IS_HEAVY
C $E$ TO_SPECISA $E$ !this is a tag
c
c
c     reading parameters
      integer maxpara
      parameter (maxpara=100)
      character*20 param(maxpara),value(maxpara)
      integer npara
c
c     setup masses for the final-state particles
c
      include 'pmass.inc'

C-----
C  BEGIN CODE
C-----
c
c     read the cuts from the run_card.dat - this should already be done in main program
c
c      call setrun

c
c     No pdfs for decay processes - set here since here we know the nincoming
c     Also set stot here, and use mass of incoming particle for ren scale
c
         if(nincoming.eq.1)then
            lpp(1)=0
            lpp(2)=0
            ebeam(1)=pmass(1)/2d0
            ebeam(2)=pmass(1)/2d0
            scale=pmass(1)
            fixed_ren_scale=.true.
         endif
c
c     check if I have to apply cuts on the particles
c
C $B$ IDENTIFY_TYPE $B$
      do i=nincoming+1,nexternal
         do_cuts(i)=.true.
         if(nincoming.eq.1) do_cuts(i)=.false.
         is_a_j(i)=.false.
         is_a_l(i)=.false.
         is_a_b(i)=.false.
         is_a_a(i)=.false.
         is_a_nu(i)=.false.


c-do not apply cuts to these
         if (pmass(i).gt.10d0)     do_cuts(i)=.false.  ! no cuts on top,W,Z,H
         if (abs(idup(i,1)).eq.12) do_cuts(i)=.false.  ! no cuts on ve ve~
         if (abs(idup(i,1)).eq.14) do_cuts(i)=.false.  ! no cuts on vm vm~
         if (abs(idup(i,1)).eq.16) do_cuts(i)=.false.  ! no cuts on vt vt~
c-flavor-jets
         if (abs(idup(i,1)).le.min(maxjetflavor,5)) then
              is_a_j(i)=.true.
c              write(*,*)'jet:ithe pdg is ',abs(idup(i,1)),' maxflavor=',maxjetflavor
         else if (abs(idup(i,1)).ge.maxjetflavor+1 .and. abs(idup(i,1)).le.5) then
              is_a_b(i)=.true.
c              write(*,*)'bjet:the pdg is ',abs(idup(i,1)),' maxflavor=',maxjetflavor
         endif

         if (abs(idup(i,1)).eq.21)  is_a_j(i)=.true. ! gluon is a jet
c-charged-leptons
         if (abs(idup(i,1)).eq.11)  is_a_l(i)=.true. ! e+  e-
         if (abs(idup(i,1)).eq.13)  is_a_l(i)=.true. ! mu+ mu-
         if (abs(idup(i,1)).eq.15)  is_a_l(i)=.true. ! ta+ ta-
c-b-quarks
c         if (abs(idup(i,1)).eq.5)   is_a_b(i)=.true. ! b b~
c-photon
         if (idup(i,1).eq.22)   is_a_a(i)=.true. ! photon
c-neutrino's for missing et
         if (abs(idup(i,1)).eq.12) is_a_nu(i)=.true.  ! no cuts on ve ve~
         if (abs(idup(i,1)).eq.14) is_a_nu(i)=.true.  ! no cuts on vm vm~
         if (abs(idup(i,1)).eq.16) is_a_nu(i)=.true.  ! no cuts on vt vt~
         if (pmass(i).gt.10d0)     is_heavy(i)=.true. ! heavy fs particle
      enddo
C $E$ IDENTIFY_TYPE $E$      

c
c     et and eta cuts
c
      do i=nincoming+1,nexternal

         etmin(i)  = 0d0
         etmax(i)  = 1d5

         emin(i)   = 0d0
         emax(i)   = 1d5

         etamin(i) = 0d0
         etamax(i) = 1d2

         if(do_cuts(i)) then

            if(is_a_j(i)) etmin(i)=ptj
            if(is_a_l(i)) etmin(i)=ptl
            if(is_a_b(i)) etmin(i)=ptb
            if(is_a_a(i)) etmin(i)=pta

            if(is_a_j(i)) etmax(i)=ptjmax
            if(is_a_l(i)) etmax(i)=ptlmax
            if(is_a_b(i)) etmax(i)=ptbmax
            if(is_a_a(i)) etmax(i)=ptamax

            if(is_a_j(i)) emin(i)=ej
            if(is_a_l(i)) emin(i)=el
            if(is_a_b(i)) emin(i)=eb
            if(is_a_a(i)) emin(i)=ea

            if(is_a_j(i)) emax(i)=ejmax
            if(is_a_l(i)) emax(i)=elmax
            if(is_a_b(i)) emax(i)=ebmax
            if(is_a_a(i)) emax(i)=eamax

            if(is_a_j(i)) etamax(i)=etaj
            if(is_a_l(i)) etamax(i)=etal
            if(is_a_b(i)) etamax(i)=etab
            if(is_a_a(i)) etamax(i)=etaa

            if(is_a_j(i)) etamin(i)=etajmin
            if(is_a_l(i)) etamin(i)=etalmin
            if(is_a_b(i)) etamin(i)=etabmin
            if(is_a_a(i)) etamin(i)=etaamin
         endif
      enddo
c
c     delta r cut
c
      do i=nincoming+1,nexternal-1
         do j=i+1,nexternal
            r2min(j,i)=0d0
            r2max(j,i)=1d2
            if(do_cuts(i).and.do_cuts(j)) then

               if(is_a_j(i).and.is_a_j(j)) r2min(j,i)=drjj
               if(is_a_b(i).and.is_a_b(j)) r2min(j,i)=drbb
               if(is_a_l(i).and.is_a_l(j)) r2min(j,i)=drll
               if(is_a_a(i).and.is_a_a(j)) r2min(j,i)=draa

               if((is_a_b(i).and.is_a_j(j)).or.
     &           (is_a_j(i).and.is_a_b(j))) r2min(j,i)=drbj
               if((is_a_a(i).and.is_a_j(j)).or.
     &           (is_a_j(i).and.is_a_a(j))) r2min(j,i)=draj
               if((is_a_l(i).and.is_a_j(j)).or.
     &           (is_a_j(i).and.is_a_l(j))) r2min(j,i)=drjl
               if((is_a_b(i).and.is_a_a(j)).or.
     &           (is_a_a(i).and.is_a_b(j))) r2min(j,i)=drab
               if((is_a_b(i).and.is_a_l(j)).or.
     &           (is_a_l(i).and.is_a_b(j))) r2min(j,i)=drbl
               if((is_a_l(i).and.is_a_a(j)).or.
     &           (is_a_a(i).and.is_a_l(j))) r2min(j,i)=dral

               if(is_a_j(i).and.is_a_j(j)) r2max(j,i)=drjjmax
               if(is_a_b(i).and.is_a_b(j)) r2max(j,i)=drbbmax
               if(is_a_l(i).and.is_a_l(j)) r2max(j,i)=drllmax
               if(is_a_a(i).and.is_a_a(j)) r2max(j,i)=draamax

               if((is_a_b(i).and.is_a_j(j)).or.
     &           (is_a_j(i).and.is_a_b(j))) r2max(j,i)=drbjmax
               if((is_a_a(i).and.is_a_j(j)).or.
     &           (is_a_j(i).and.is_a_a(j))) r2max(j,i)=drajmax
               if((is_a_l(i).and.is_a_j(j)).or.
     &           (is_a_j(i).and.is_a_l(j))) r2max(j,i)=drjlmax
               if((is_a_b(i).and.is_a_a(j)).or.
     &           (is_a_a(i).and.is_a_b(j))) r2max(j,i)=drabmax
               if((is_a_b(i).and.is_a_l(j)).or.
     &           (is_a_l(i).and.is_a_b(j))) r2max(j,i)=drblmax
               if((is_a_l(i).and.is_a_a(j)).or.
     &           (is_a_a(i).and.is_a_l(j))) r2max(j,i)=dralmax
 
            endif
         enddo
      enddo
c     
c     smin cut
c
      do i=nincoming+1,nexternal-1
         do j=i+1,nexternal
            s_min(j,i)=0.0d0**2
            s_max(j,i)=1d5**2
            if(do_cuts(i).and.do_cuts(j)) then
               if(is_a_j(i).and.is_a_j(j)) s_min(j,i)=mmjj*dabs(mmjj)   
               if(is_a_a(i).and.is_a_a(j)) s_min(j,i)=mmaa*dabs(mmaa)  
               if( is_a_b(i).and.is_a_b(j) ) s_min(j,i)=mmbb*dabs(mmbb)     
               if((is_a_l(i).and.is_a_l(j)).and.
     &            (abs(idup(i,1)).eq.abs(idup(j,1))).and.
     &            (idup(i,1)*idup(j,1).lt.0)) 
     &            s_min(j,i)=mmll*dabs(mmll)  !only on l+l- pairs (same flavour) 

               if(is_a_j(i).and.is_a_j(j)) s_max(j,i)=mmjjmax*dabs(mmjjmax)   
               if(is_a_a(i).and.is_a_a(j)) s_max(j,i)=mmaamax*dabs(mmaamax)  
               if( is_a_b(i).and.is_a_b(j) ) s_max(j,i)=mmbbmax*dabs(mmbbmax)     
               if((is_a_l(i).and.is_a_l(j)).and.
     &            (abs(idup(i,1)).eq.abs(idup(j,1))).and.
     &            (idup(i,1)*idup(j,1).lt.0)) 
     &            s_max(j,i)=mmllmax*dabs(mmllmax)  !only on l+l- pairs (same flavour)

            endif
         enddo
      enddo      

c
c   EXTRA JET CUTS
c
      ptjmin4(1)=ptj1min
      ptjmin4(2)=ptj2min
      ptjmin4(3)=ptj3min
      ptjmin4(4)=ptj4min

      ptjmax4(1)=ptj1max
      ptjmax4(2)=ptj2max
      ptjmax4(3)=ptj3max
      ptjmax4(4)=ptj4max

      Htjmin4(2)=ht2min
      htjmin4(3)=ht3min
      htjmin4(4)=ht4min

      htjmax4(2)=ht2max
      htjmax4(3)=ht3max
      htjmax4(4)=ht4max

      jetor = cutuse.eq.0d0
c
c    ERROR TRAPS 
c
        do i=nincoming+1,nexternal
           if(is_a_j(i).and.etmin(i).eq.0.and.emin(i).eq.0) then
              write (*,*) "Warning: pt or E min of a jet should in general be >0"
           endif
           if(is_a_a(i).and.etmin(i).eq.0.and.emin(i).eq.0) then
              write (*,*) "Warning: pt or E min of a gamma should in general be >0"
           endif
        enddo

c    count number of jets to see if special cuts are applicable or not

        ncheck=0
        do i=nincoming+1,nexternal
           if(is_a_j(i)) ncheck=ncheck+1
        enddo

        if(ncheck.eq.0.and. xptj .gt. 0d0) then
           write (*,*) "Warning: cuts on the jet will be ignored"
           xptj = 0d0
        endif

        if(ncheck.lt.2.and. xetamin .gt. 0 .and. deltaeta .gt.0) then
           write (*,*) "Warning: WBF cuts not will be ignored"
           xetamin = 0d0
           deltaeta =0d0
        endif

c    count number of photons to see if special cuts are applicable or not

        ncheck=0
        do i=nincoming+1,nexternal
           if(is_a_a(i)) ncheck=ncheck+1
        enddo

        if(ncheck.eq.0.and. xpta .gt. 0d0) then
           write (*,*) "Warning: cuts on the photon will be ignored"
           xpta =0d0
        endif

c    count number of b-quarks to see if special cuts are applicable or not

        Ncheck=0
        do i=nincoming+1,nexternal
           if(is_a_b(i)) ncheck=ncheck+1
        enddo

        if(ncheck.eq.0.and. xptb .gt. 0d0) then
           write (*,*) "Warning: cuts on the b-quarks will be ignored"
           xptb = 0d0
        endif

c    count number of leptons to see if special cuts are applicable or not

        ncheck=0
        do i=nincoming+1,nexternal
           if(is_a_l(i)) ncheck=ncheck+1
        enddo

        if(ncheck.eq.0.and. xptl .gt. 0d0) then
           write (*,*) "Warning: cuts on the lepton will be ignored"
           xptl = 0d0
        endif
c
c Set minimum tau, to be used in one_tree
      call set_tau_min()

c      call write_cuts()
      RETURN

      END


      subroutine set_tau_min()
c Sets the lower bound for tau=x1*x2, using information on particle
c masses and on the jet minimum pt, as entered in run_card.dat, 
c variable ptj
      implicit none
      double precision zero
      parameter (zero=0.d0)
      include 'cuts.inc'
      include 'run.inc'
      include 'genps.inc'
      include "nexternal.inc"
      include 'coupl.inc'
      LOGICAL  IS_A_J(NEXTERNAL),IS_A_L(NEXTERNAL)
      LOGICAL  IS_A_B(NEXTERNAL),IS_A_A(NEXTERNAL)
      LOGICAL  IS_A_NU(NEXTERNAL),IS_HEAVY(NEXTERNAL)
      COMMON /TO_SPECISA/IS_A_J,IS_A_A,IS_A_L,IS_A_B,IS_A_NU,IS_HEAVY
c
      double precision pmass(-nexternal:0,lmaxconfigs)
      double precision pwidth(-nexternal:0,lmaxconfigs)
      integer pow(-nexternal:0,lmaxconfigs)
      integer iforest(2,-max_branch:-1,lmaxconfigs)
      common/to_forest/ iforest
      integer            mapconfig(0:lmaxconfigs), this_config
      common/to_mconfigs/mapconfig, this_config

      double precision taumin,stot
      integer i
      double precision xm(-nexternal:nexternal),xmmax
      integer tsign,iconfig
      double precision tau_lower_bound,tau_lower_bound_soft
      common/ctau_lower_bound/tau_lower_bound,tau_lower_bound_soft


c
      real*8         emass(nexternal)
      common/to_mass/emass
c      double precision pmass(nexternal)
c      include "pmass.inc"
      include "props.inc"
c
      if(.not.IS_A_J(NEXTERNAL))then
        write(*,*)'Fatal error in set_tau_min'
        stop
      endif
c The following assumes that light QCD particles are at the end of the list.
c Exclude one of them to set tau bound at the Born level
      taumin=0.d0
      do i=nincoming+1,nexternal-1
        if(IS_A_J(i))then
          taumin=taumin+sqrt(ptj**2+emass(i)**2)
        else
          taumin=taumin+emass(i)
        endif
        xm(i)=emass(i)
      enddo
      xm(nexternal)=emass(nexternal)
      stot = 4d0*ebeam(1)*ebeam(2)
      tau_lower_bound=taumin**2/stot


c Also find the minimum lower bound if all internal s-channel particles
c were on-shell
      tsign=-1
      iconfig=this_config
      xmmax=0d0
      do i=-1,-(nexternal-3),-1                ! All propagators
         if ( iforest(1,i,iconfig) .eq. 1 .or.
     &        iforest(1,i,iconfig) .eq. 2 ) tsign=1
         if (tsign.eq.-1) then   ! Only s-channels
            xm(i)=max(pmass(i,iconfig),
     &           xm(iforest(1,i,iconfig))+xm(iforest(2,i,iconfig)))
         else
            xm(i)=0d0
         endif
         xmmax=max(xmmax,xm(i))
      enddo
      
      tau_lower_bound_soft=xmmax**2/stot
      tau_lower_bound_soft=max(tau_lower_bound,tau_lower_bound_soft)

      write (*,*) 'lower bound for tau is ',
     &     tau_lower_bound,taumin,dsqrt(stot)
      write (*,*) 'soft lower bound for tau is ',
     &     tau_lower_bound_soft,xmmax,dsqrt(stot)

      return
      end


