      SUBROUTINE SETCUTS
C**************************************************************************
C     SET THE CUTS 
C**************************************************************************
      IMPLICIT NONE
c
c     INCLUDE
c
      include 'genps.inc'
      include 'nexternal.inc'
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
      integer ncheck
      logical done,fopened
      logical from_decay(-(nexternal+3):nexternal)
C     
C     GLOBAL
C
c--masses and poles
      double precision pmass(nexternal)
      common/to_mass/  pmass
      DOUBLE PRECISION       qmass(2)
      COMMON/to_qmass/qmass
      double precision      spole(maxinvar),swidth(maxinvar),bwjac
      common/to_brietwigner/spole          ,swidth          ,bwjac

      double precision Smin
      common/to_smin/ Smin
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
      double precision ptll_min(nexternal,nexternal),ptll_max(nexternal,nexternal)
      double precision inclHtmin,inclHtmax
      common/to_cuts/  etmin, emin, etamax, r2min, s_min,
     $     etmax, emax, etamin, r2max, s_max, ptll_min, ptll_max, inclHtmin,inclHtmax

      double precision ptjmin4(4),ptjmax4(4),htjmin4(2:4),htjmax4(2:4)
      logical jetor
      common/to_jet_cuts/ ptjmin4,ptjmax4,htjmin4,htjmax4,jetor

      double precision ptlmin4(4),ptlmax4(4)
      common/to_lepton_cuts/ ptlmin4,ptlmax4

      double precision xqcutij(nexternal,nexternal),xqcuti(nexternal)
      common/to_xqcuts/xqcutij,xqcuti
c
c     les houches accord stuff to identify neutrinos
c
      include 'maxamps.inc'
      integer idup(nexternal,maxproc,maxsproc)
      integer mothup(2,nexternal)
      integer icolup(2,nexternal,maxflow,maxsproc)
      include 'leshouche.inc'
C
      LOGICAL  IS_A_J(NEXTERNAL),IS_A_L(NEXTERNAL)
      LOGICAL  IS_A_B(NEXTERNAL),IS_A_A(NEXTERNAL),IS_A_ONIUM(NEXTERNAL)
      LOGICAL  IS_A_NU(NEXTERNAL),IS_HEAVY(NEXTERNAL)
      logical  do_cuts(nexternal)
      COMMON /TO_SPECISA/IS_A_J,IS_A_A,IS_A_L,IS_A_B,IS_A_NU,IS_HEAVY,
     . IS_A_ONIUM, do_cuts 
c
c
c     reading parameters
      include 'run_config.inc'
      character*20 param(maxpara),value(maxpara)
      integer npara
c For checking the consistency of the grouping and the cuts defined here
      integer iproc
      logical equal
      LOGICAL  IS_A_J_SAVE(NEXTERNAL),IS_A_L_SAVE(NEXTERNAL)
     $     ,IS_A_B_SAVE(NEXTERNAL),IS_A_A_SAVE(NEXTERNAL)
     $     ,IS_A_ONIUM_SAVE(NEXTERNAL),IS_A_NU_SAVE(NEXTERNAL)
     $     ,IS_HEAVY_SAVE(NEXTERNAL),DO_CUTS_SAVE(NEXTERNAL)
      double precision r2min_save(nexternal,nexternal)
     $     ,r2max_save(nexternal,nexternal),s_min_save(nexternal
     $     ,nexternal),s_max_save(nexternal,nexternal)
     $     ,ptll_min_save(nexternal,nexternal),ptll_max_save(nexternal
     $     ,nexternal),etmin_save(nexternal),etmax_save(nexternal)
     $     ,emin_save(nexternal) ,emax_save(nexternal)
     $     ,etamin_save(nexternal),etamax_save(nexternal)
      save  IS_A_J_SAVE,IS_A_L_SAVE,IS_A_B_SAVE,IS_A_A_SAVE
     $     ,IS_A_ONIUM_SAVE,IS_A_NU_SAVE,IS_HEAVY_SAVE
     $     ,r2min_save,r2max_save,s_min_save,s_max_save,ptll_min_save
     $     ,ptll_max_save,etmin_save,etmax_save,emin_save,emax_save
     $     ,etamin_save,etamax_save
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
            fixed_fac_scale=.true.
            use_syst=.false.
         endif
c
c     set ptj and s_min if xqcut and ktscheme = 1, to improve
c     integration speed, and set drjj and drjl to 0.
c
        if(xqcut.gt.0) then
           if(auto_ptj_mjj.and.ptj.ge.0d0.and.ktscheme.eq.1)then
            ptj=xqcut
            write(*,*) 'Warning! ptj set to xqcut=',xqcut,
     $            ' to improve integration efficiency'
            write(*,*) 'Note that this might affect non-radiated jets,'
            write(*,*) 'e.g. from decays. Use cut_decays=F in run_card.'
          else if(ptj.gt.xqcut)then
            ptj=0d0
            write(*,*) 'Warning! ptj set to 0 since xqcut > 0 and'
            write(*,*) '         auto_ptj_mjj = F or ktscheme > 1'
          endif
          if(auto_ptj_mjj.and.mmjj.ge.0d0)then
            mmjj=xqcut
            write(*,*) 'Warning! mmjj set to xqcut=',xqcut,
     $            ' to improve integration efficiency'
            write(*,*) 'Note that this might affect non-radiated jets,'
            write(*,*) 'e.g. from decays. Use cut_decays=F in run_card.'
          else if(mmjj.gt.xqcut)then
c           In principle this should never happen since the banner.py
c           is expected to correct this already.
            mmjj=0d0
            write(*,*) 'Warning! mmjj set to 0 since xqcut > 0 and'
            write(*,*) '         auto_ptj_mjj = F'
          endif
          if(drjj.gt.0d0) then
            write(*,*) 'Warning! drjj > 0 with xqcut > 0, set to 0'
            drjj = 0d0
          endif
          if(drjl.gt.0d0) then
            write(*,*) 'Warning! drjl > 0 with xqcut > 0, set to 0'
            drjl = 0d0
          endif
        endif

c     Check which particles come from decays
      if(.not.cut_decays)
     $       call check_decay(from_decay)


      do iproc=1,maxsproc

c
c     check if I have to apply cuts on the particles
c
      do i=nincoming+1,nexternal
         do_cuts(i)=.true.
         if(.not.cut_decays.and.from_decay(i)) do_cuts(i)=.false.
         is_a_j(i)=.false.
         is_a_l(i)=.false.
         is_a_b(i)=.false.
         is_a_a(i)=.false.
         is_a_nu(i)=.false.


c-do not apply cuts to these
         if (pmass(i).gt.20d0)     do_cuts(i)=.false.  ! no cuts on top,W,Z,H
         if (abs(idup(i,1,iproc)).eq.12) do_cuts(i)=.false.  ! no cuts on ve ve~
         if (abs(idup(i,1,iproc)).eq.14) do_cuts(i)=.false.  ! no cuts on vm vm~
         if (abs(idup(i,1,iproc)).eq.16) do_cuts(i)=.false.  ! no cuts on vt vt~
c-flavor-jets
         if (abs(idup(i,1,iproc)).le.min(maxjetflavor,6)) then
              is_a_j(i)=.true.
c              write(*,*)'jet:ithe pdg is ',abs(idup(i,1,iproc)),' maxflavor=',maxjetflavor
         else if (abs(idup(i,1,iproc)).ge.maxjetflavor+1 .and. abs(idup(i,1,iproc)).le.5) then
              is_a_b(i)=.true.
c              write(*,*)'bjet:the pdg is ',abs(idup(i,1,iproc)),' maxflavor=',maxjetflavor
         endif

         if (abs(idup(i,1,iproc)).eq.21)  is_a_j(i)=.true. ! gluon is a jet
c-charged-leptons
         if (abs(idup(i,1,iproc)).eq.11)  is_a_l(i)=.true. ! e+  e-
         if (abs(idup(i,1,iproc)).eq.13)  is_a_l(i)=.true. ! mu+ mu-
         if (abs(idup(i,1,iproc)).eq.15)  is_a_l(i)=.true. ! ta+ ta-
c-b-quarks
c         if (abs(idup(i,1,iproc)).eq.5)   is_a_b(i)=.true. ! b b~
c-photon
         if (idup(i,1,iproc).eq.22)   is_a_a(i)=.true. ! photon
c-neutrino's for missing et
         if (abs(idup(i,1,iproc)).eq.12) is_a_nu(i)=.true.  ! no cuts on ve ve~
         if (abs(idup(i,1,iproc)).eq.14) is_a_nu(i)=.true.  ! no cuts on vm vm~
         if (abs(idup(i,1,iproc)).eq.16) is_a_nu(i)=.true.  ! no cuts on vt vt~
         if (pmass(i).gt.10d0)     is_heavy(i)=.true. ! heavy fs particle
c-onium
         if (idup(i,1,iproc).eq.441)   is_a_onium(i)=.true. ! charmonium
         if (idup(i,1,iproc).eq.10441)   is_a_onium(i)=.true. ! charmonium
         if (idup(i,1,iproc).eq.100441)   is_a_onium(i)=.true. ! charmonium
         if (idup(i,1,iproc).eq.443)   is_a_onium(i)=.true. ! charmonium
         if (idup(i,1,iproc).eq.10443)   is_a_onium(i)=.true. ! charmonium
         if (idup(i,1,iproc).eq.20443)   is_a_onium(i)=.true. ! charmonium
         if (idup(i,1,iproc).eq.100443)   is_a_onium(i)=.true. ! charmonium
         if (idup(i,1,iproc).eq.30443)   is_a_onium(i)=.true. ! charmonium
         if (idup(i,1,iproc).eq.9000443)   is_a_onium(i)=.true. ! charmonium
         if (idup(i,1,iproc).eq.9010443)   is_a_onium(i)=.true. ! charmonium
         if (idup(i,1,iproc).eq.9020443)   is_a_onium(i)=.true. ! charmonium
         if (idup(i,1,iproc).eq.445)   is_a_onium(i)=.true. ! charmonium
         if (idup(i,1,iproc).eq.9000445)   is_a_onium(i)=.true. ! charmonium

         if (idup(i,1,iproc).eq.551)   is_a_onium(i)=.true. ! bottomonium
         if (idup(i,1,iproc).eq.10551)   is_a_onium(i)=.true. ! bottomonium
         if (idup(i,1,iproc).eq.100551)   is_a_onium(i)=.true. ! bottomonium
         if (idup(i,1,iproc).eq.110551)   is_a_onium(i)=.true. ! bottomonium
         if (idup(i,1,iproc).eq.200551)   is_a_onium(i)=.true. ! bottomonium
         if (idup(i,1,iproc).eq.210551)   is_a_onium(i)=.true. ! bottomonium
         if (idup(i,1,iproc).eq.553)   is_a_onium(i)=.true. ! bottomonium
         if (idup(i,1,iproc).eq.10553)   is_a_onium(i)=.true. ! bottomonium
         if (idup(i,1,iproc).eq.20553)   is_a_onium(i)=.true. ! bottomonium
         if (idup(i,1,iproc).eq.30553)   is_a_onium(i)=.true. ! bottomonium
         if (idup(i,1,iproc).eq.100553)   is_a_onium(i)=.true. ! bottomonium
         if (idup(i,1,iproc).eq.110553)   is_a_onium(i)=.true. ! bottomonium
         if (idup(i,1,iproc).eq.120553)   is_a_onium(i)=.true. ! bottomonium
         if (idup(i,1,iproc).eq.130553)   is_a_onium(i)=.true. ! bottomonium
         if (idup(i,1,iproc).eq.200553)   is_a_onium(i)=.true. ! bottomonium
         if (idup(i,1,iproc).eq.210553)   is_a_onium(i)=.true. ! bottomonium
         if (idup(i,1,iproc).eq.220553)   is_a_onium(i)=.true. ! bottomonium
         if (idup(i,1,iproc).eq.300553)   is_a_onium(i)=.true. ! bottomonium
         if (idup(i,1,iproc).eq.9000553)   is_a_onium(i)=.true. ! bottomonium
         if (idup(i,1,iproc).eq.9010553)   is_a_onium(i)=.true. ! bottomonium
         if (idup(i,1,iproc).eq.555)   is_a_onium(i)=.true. ! bottomonium
         if (idup(i,1,iproc).eq.10555)   is_a_onium(i)=.true. ! bottomonium
         if (idup(i,1,iproc).eq.20555)   is_a_onium(i)=.true. ! bottomonium
         if (idup(i,1,iproc).eq.100555)   is_a_onium(i)=.true. ! bottomonium
         if (idup(i,1,iproc).eq.110555)   is_a_onium(i)=.true. ! bottomonium
         if (idup(i,1,iproc).eq.200555)   is_a_onium(i)=.true. ! bottomonium
         if (idup(i,1,iproc).eq.557)   is_a_onium(i)=.true. ! bottomonium
         if (idup(i,1,iproc).eq.100557)   is_a_onium(i)=.true. ! bottomonium

         if (idup(i,1,iproc).eq.541)   is_a_onium(i)=.true. ! Bc
         if (idup(i,1,iproc).eq.10541)   is_a_onium(i)=.true. ! Bc
         if (idup(i,1,iproc).eq.543)   is_a_onium(i)=.true. ! Bc
         if (idup(i,1,iproc).eq.10543)   is_a_onium(i)=.true. ! Bc
         if (idup(i,1,iproc).eq.20543)   is_a_onium(i)=.true. ! Bc
         if (idup(i,1,iproc).eq.545)   is_a_onium(i)=.true. ! Bc
      enddo

c
c     et and eta cuts
c
         Smin = 0d0
      do i=nincoming+1,nexternal
         etmin(i)  = 0d0
         etmax(i)  = -1

         emin(i)   = 0d0
         emax(i)   = -1

         etamin(i) = 0d0
         etamax(i) = -1

         if(do_cuts(i)) then

c        JET
            if(is_a_j(i))then
                 etmin(i)=ptj
                 SMIN = SMIN + max(ptj,ej)
                 etmax(i)=ptjmax
                 emin(i)=ej
                 emax(i)=ejmax
                 etamax(i)=etaj
                 etamin(i)=etajmin
            endif
c        LEPTON
            if(is_a_l(i))then
                 etmin(i)=ptl
                 SMIN = SMIN + max(ptl,el)
                 etmax(i)=ptlmax
                 emin(i)=el
                 emax(i)=elmax
                 etamax(i)=etal
                 etamin(i)=etalmin
            endif
c        BJET
            if(is_a_b(i))then
                 etmin(i)=ptb
                 SMIN = SMIN + max(ptb,eb)
                 etmax(i)=ptbmax
                 emin(i)=eb
                 emax(i)=ebmax
                 etamax(i)=etab
                 etamin(i)=etabmin
            endif
c        PHOTON
            if(is_a_a(i))then
                 etmin(i) = max(pta, ptgmin, ea)
                 SMIN = SMIN + etmin(i)
                 etmax(i)=ptamax
                 emin(i)=ea
                 emax(i)=eamax
                 etamax(i)=etaa
                 etamin(i)=etaamin
            endif
c        QUARKONIUM
            if(is_a_onium(i))then
                 etmin(i)=ptonium
                 SMIN = SMIN + ptonium
                 etamax(i)=etaonium
            endif
         endif
      enddo
      SMIN = SMIN **2
c
c     delta r cut
c
      do i=nincoming+1,nexternal-1
         do j=i+1,nexternal
            r2min(j,i)=0d0
            r2max(j,i)=-1
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
            s_max(j,i)=-1
            if(do_cuts(i).and.do_cuts(j)) then
               if(is_a_j(i).and.is_a_j(j)) s_min(j,i)=mmjj*dabs(mmjj)   
               if(is_a_a(i).and.is_a_a(j)) s_min(j,i)=mmaa*dabs(mmaa)  
               if( is_a_b(i).and.is_a_b(j) ) s_min(j,i)=mmbb*dabs(mmbb)     
               if((is_a_l(i).and.is_a_l(j)).and.
     &            (abs(idup(i,1,iproc)).eq.abs(idup(j,1,iproc))).and.
     &            (idup(i,1,iproc)*idup(j,1,iproc).lt.0)) 
     &            s_min(j,i)=mmll*dabs(mmll)  !only on l+l- pairs (same flavour) 

               if(is_a_j(i).and.is_a_j(j)) s_max(j,i)=mmjjmax*dabs(mmjjmax)   
               if(is_a_a(i).and.is_a_a(j)) s_max(j,i)=mmaamax*dabs(mmaamax)  
               if( is_a_b(i).and.is_a_b(j) ) s_max(j,i)=mmbbmax*dabs(mmbbmax)     
               if((is_a_l(i).and.is_a_l(j)).and.
     &            (abs(idup(i,1,iproc)).eq.abs(idup(j,1,iproc))).and.
     &            (idup(i,1,iproc)*idup(j,1,iproc).lt.0)) 
     &            s_max(j,i)=mmllmax*dabs(mmllmax)  !only on l+l- pairs (same flavour)
            endif
         enddo
      enddo      

c     
c     ptll cut (min and max)
c

      do i=nincoming+1,nexternal-1
         do j=i+1,nexternal
            ptll_min(j,i)=0.0d0**2
            ptll_max(j,i)=-1
            if(((is_a_l(i).and.is_a_l(j)).and.
     &           (abs(idup(i,1,iproc)).eq.abs(idup(j,1,iproc))).and.
     &           (idup(i,1,iproc)*idup(j,1,iproc).lt.0)) ! Leptons from same flavor but different charge
     &           .or.(is_a_nu(i).and.is_a_l(j))
     &           .or.(is_a_l(i).and.is_a_nu(j)) !a lepton and a neutrino
     &           .or.(is_a_nu(i).and.is_a_nu(j))) then ! two neutrinos 
               ptll_min(j,i)=ptllmin*dabs(ptllmin)
               ptll_max(j,i)=ptllmax*dabs(ptllmax)
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
   
      inclHtmin=ihtmin
      inclHtmax=ihtmax

      jetor = cutuse.eq.0

c
c   EXTRA LEPTON CUTS
c
      ptlmin4(1)=ptl1min
      ptlmin4(2)=ptl2min
      ptlmin4(3)=ptl3min
      ptlmin4(4)=ptl4min

      ptlmax4(1)=ptl1max
      ptlmax4(2)=ptl2max
      ptlmax4(3)=ptl3max
      ptlmax4(4)=ptl4max


c
c     Compute Smin (for efficiency
c
      do i=nincoming+1,nexternal-1
      do j=nincoming+1,nexternal-1
         if(j.lt.i)then
            s_min(i,j) = max(s_min(j,i),s_min(i,j))
         else
            smin=0.0d0**2

            if(do_cuts(i).and.do_cuts(j)) then
               if(is_a_j(i).and.is_a_j(j)) s_min(j,i)=mmjj*dabs(mmjj)
               if(is_a_a(i).and.is_a_a(j)) s_min(j,i)=mmaa*dabs(mmaa)
               if( is_a_b(i).and.is_a_b(j) ) s_min(j,i)=mmbb*dabs(mmbb)
               if((is_a_l(i).and.is_a_l(j)).and.
     &            (abs(idup(i,1,iproc)).eq.abs(idup(j,1,iproc))).and.
     &            (idup(i,1,iproc)*idup(j,1,iproc).lt.0))
     &            s_min(j,i)=mmll*dabs(mmll)  !only on l+l- pairs (same flavour)

               if(is_a_j(i).and.is_a_j(j)) s_max(j,i)=mmjjmax*dabs(mmjjmax)
               if(is_a_a(i).and.is_a_a(j)) s_max(j,i)=mmaamax*dabs(mmaamax)
               if( is_a_b(i).and.is_a_b(j) ) s_max(j,i)=mmbbmax*dabs(mmbbmax)
               if((is_a_l(i).and.is_a_l(j)).and.
     &            (abs(idup(i,1,iproc)).eq.abs(idup(j,1,iproc))).and.
     &            (idup(i,1,iproc)*idup(j,1,iproc).lt.0))
     &            s_max(j,i)=mmllmax*dabs(mmllmax)  !only on l+l- pairs (same flavour)
            endif
         endif
      enddo
      enddo

c Check that results are consistent among all the grouped subprocesses

      if (iproc.eq.1) then
         do i=nincoming+1,nexternal
            do_cuts_save(i)=do_cuts(i)
            is_a_j_save(i)=is_a_j(i)
            is_a_b_save(i)=is_a_b(i)
            is_a_l_save(i)=is_a_l(i)
            is_a_a_save(i)=is_a_a(i)
            is_a_nu_save(i)=is_a_nu(i)
            is_a_onium_save(i)=is_a_onium(i)
            is_heavy_save(i)=is_heavy(i) 
            etmin_save(i)=etmin(i)
            etmax_save(i)=etmax(i)
            emin_save(i)=emin(i)
            emax_save(i)=emax(i)
            etamin_save(i)=etamin(i)
            etamax_save(i)=etamax(i)
            if (i.eq.nexternal) cycle
            do j=i+1,nexternal
               r2min_save(j,i) = r2min(j,i)
               r2max_save(j,i) = r2max(j,i)
               s_min_save(j,i) = s_min(j,i)
               s_max_save(j,i) = s_max(j,i)
               ptll_min_save(j,i) = ptll_min(j,i)
               ptll_max_save(j,i) = ptll_max(j,i)
            enddo
         enddo
      else
         equal=.true.
         do i=nincoming+1,nexternal
            if (do_cuts_save(i).neqv.do_cuts(i)) equal=.false.
            if (is_a_j_save(i).neqv.is_a_j(i)) then
               if (ptjmin4(1).gt.0d0 .or. ptjmax4(1).ge.0d0) equal=.false.
               if (ptjmin4(2).gt.0d0 .or. ptjmax4(2).ge.0d0) equal=.false.
               if (ptjmin4(3).gt.0d0 .or. ptjmax4(3).ge.0d0) equal=.false.
               if (ptjmin4(4).gt.0d0 .or. ptjmax4(4).ge.0d0) equal=.false.
               if (Htjmin4(2).gt.0d0 .or. Htjmax4(2).ge.0d0) equal=.false.
               if (Htjmin4(3).gt.0d0 .or. Htjmax4(3).ge.0d0) equal=.false.
               if (Htjmin4(4).gt.0d0 .or. Htjmax4(4).ge.0d0) equal=.false.
               if (inclHtmin.gt.0d0 .or. inclHtmax.ge.0d0) equal=.false.
               if (htjmin.gt.0d0 .or. htjmax.ge.0d0) equal=.false.
               if (xptj.gt.0d0) equal=.false.
               if (xetamin.gt.0d0 .or. deltaeta.gt.0d0) equal=.false.
               if (ptgmin.ne.0d0) equal=.false.
               if (kt_durham.gt.0d0) equal=.false.
            endif
            if (is_a_b_save(i).neqv.is_a_b(i)) then
               if (inclHtmin.gt.0d0 .or. inclHtmax.ge.0d0) equal=.false.
               if (xptb.gt.0d0) equal=.false.
            endif
            if (is_a_a_save(i).neqv.is_a_a(i)) then
               if (xpta.gt.0d0) equal=.false.
               if (ptgmin.ne.0d0) equal=.false.
            endif
            if (is_a_l_save(i).neqv.is_a_l(i)) then
               if (ptlmin4(1).gt.0d0 .or. ptlmax4(1).ge.0d0) equal=.false.
               if (ptlmin4(2).gt.0d0 .or. ptlmax4(2).ge.0d0) equal=.false.
               if (ptlmin4(3).gt.0d0 .or. ptlmax4(3).ge.0d0) equal=.false.
               if (ptlmin4(4).gt.0d0 .or. ptlmax4(4).ge.0d0) equal=.false.
               if (mmnl.gt.0d0 .or. mmnlmax.ge.0d0) equal=.false.
               if (xptl.gt.0d0) equal=.false.
               if (ptgmin.ne.0d0 .and. isoEM) equal=.false.
            endif
            if (is_a_nu_save(i).neqv.is_a_nu(i)) then
               if (misset.gt.0d0 .or. missetmax.ge.0d0) equal=.false.
               if (mmnl.gt.0d0 .or. mmnlmax.ge.0d0) equal=.false.
            endif
            if (is_heavy_save(i).neqv.is_heavy(i)) then
               if (ptheavy.gt.0d0) equal=.false.
            endif
            if (etmin_save(i).ne.etmin(i)) equal=.false.
            if (etmax_save(i).ne.etmax(i)) equal=.false.
            if (emin_save(i).ne.emin(i)) equal=.false.
            if (emax_save(i).ne.emax(i)) equal=.false.
            if (etamin_save(i).ne.etamin(i)) equal=.false.
            if (etamax_save(i).ne.etamax(i)) equal=.false.
            if (i.eq.nexternal) cycle
            do j=i+1,nexternal
               if (r2min_save(j,i).ne.r2min(j,i)) equal=.false.
               if (r2max_save(j,i).ne.r2max(j,i)) equal=.false.
               if (s_min_save(j,i).ne.s_min(j,i)) equal=.false.
               if (s_max_save(j,i).ne.s_max(j,i)) equal=.false.
               if (ptll_min_save(j,i).ne.ptll_min(j,i)) equal=.false.
               if (ptll_max_save(j,i).ne.ptll_max(j,i)) equal=.false.
            enddo
         enddo
         if (.not.equal) then
            write (*,*) 'Grouping of subprocesses not'/
     $           /' consistent with setcuts.f. Either change'/
     $           /' your cuts and/or turn grouping of subprocesses off.'
            open(unit=26,file='../../../error',status='unknown')
            write(26,*) 'Error: grouping of subprocesses not'/
     $           /' consistent with setcuts.f. Either change'/
     $           /' your cuts and/or turn grouping of subprocesses off.'
            stop 1
         endif
      endif
      enddo




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

c     set possible xqcut combinations (for better grid preparation)
        if(xqcut.gt.0)
     $       call setxqcuts(do_cuts)

c      call write_cuts()
      RETURN

      END


      subroutine setxqcuts(do_cuts)
c**************************************************
c     Set xqcuti and xqcutij between all particles
c     to allow for grid preparation based on xqcut
c**************************************************
      implicit none
      include 'genps.inc'
      include 'maxconfigs.inc'
      include 'nexternal.inc'
      include 'cuts.inc'

      logical  do_cuts(nexternal)

      double precision pmass(nexternal)
      common/to_mass/  pmass
      integer iforest(2,-max_branch:-1,lmaxconfigs)
      common/to_forest/ iforest
      integer mapconfig(0:lmaxconfigs), this_config
      common/to_mconfigs/mapconfig, this_config

      double precision xqcutij(nexternal,nexternal),xqcuti(nexternal)
      common/to_xqcuts/xqcutij,xqcuti

      
      integer i,j,k
      logical foundpartner
      include 'maxamps.inc'
      integer idup(nexternal,maxproc,maxsproc)
      integer mothup(2,nexternal)
      integer icolup(2,nexternal,maxflow,maxsproc)
      include 'leshouche.inc'

      do i=3,nexternal
         xqcuti(i)=0d0
         do j=3,nexternal
            xqcutij(i,j)=0d0
         enddo
      enddo

      do i=3,nexternal
         do j=1,mapconfig(0)
            foundpartner=.false.
            do k=-1,-(nexternal-3),-1
               if(iforest(1,k,j).eq.i.and.iforest(2,k,j).gt.2.or.
     $              iforest(2,k,j).eq.i.and.iforest(1,k,j).gt.2)then
                  foundpartner=.true.
                  if(iabs(idup(iforest(2,k,j),1,1)).le.maxjetflavor.or.
     $                 idup(iforest(2,k,j),1,1).eq.21.or.
     $                 iabs(idup(iforest(2,k,j),1,1)).le.maxjetflavor.or.
     $                 idup(iforest(2,k,j),1,1).eq.21)then
                     if (do_cuts(iforest(2,k,j)).and. do_cuts(iforest(1,k,j)))then
                        xqcutij(iforest(2,k,j),iforest(1,k,j))=xqcut
                        xqcutij(iforest(1,k,j),iforest(2,k,j))=xqcut
                     endif
                  endif
               endif
            enddo
            if(.not.foundpartner.and.(iabs(idup(i,1,1)).le.maxjetflavor.or.
     $           idup(i,1,1).eq.21).and.do_cuts(i)) xqcuti(i)=max(0d0,sqrt(xqcut**2-pmass(i)**2))
         enddo
      enddo

      end


c************************************************************************
c     Subroutine to check which external particles are from decays
c************************************************************************
      subroutine check_decay(from_decay)
      implicit none

      include 'nexternal.inc'
      include 'maxconfigs.inc'
      include 'genps.inc'
      include 'maxamps.inc'
c
c    Arguments
c
      logical from_decay(-(nexternal+3):nexternal)
c
c     Global
c
      integer iforest(2,-max_branch:-1,lmaxconfigs)
      common/to_forest/ iforest
      integer sprop(maxsproc,-max_branch:-1,lmaxconfigs)
      integer tprid(-max_branch:-1,lmaxconfigs)
      common/to_sprop/sprop,tprid
      integer gForceBW(-max_branch:-1,lmaxconfigs)  ! Forced BW
      include 'decayBW.inc'

c
c     Local
c
      integer i,j

      do i=-(nexternal-3),nexternal
         from_decay(i)=.false.
      enddo

c     Set who comes from decay based on forced BW
      do i=-(nexternal-3),-1
         if(tprid(i,1).eq.0.and.gForceBW(i,1).eq.1.or.
     $        from_decay(i)) then
            from_decay(i)=.true.
            from_decay(iforest(1,i,1))=.true.
            from_decay(iforest(2,i,1))=.true.
         endif
      enddo

      end
