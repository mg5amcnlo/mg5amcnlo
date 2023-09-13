c
c This file contains the default cuts (as defined in the run_card.dat)
c and can easily be extended by the user to include other.  This
c function should return true if event passes cuts
c (passcuts_user=.true.) and false otherwise (passcuts_user=.false.).
c
c NOTE THAT ONLY IRC-SAFE CUTS CAN BE APPLIED OTHERWISE THE INTEGRATION
c MIGHT NOT CONVERGE
c
      logical function passcuts_user(p,istatus,ipdg)
      implicit none
c This includes the 'nexternal' parameter that labels the number of
c particles in the (n+1)-body process
      include 'nexternal.inc'
c This include file contains common blocks filled with the cuts defined
c in the run_card.dat
      include 'cuts.inc'
c
c This is an array which is '-1' for initial state and '1' for final
c state particles
      integer istatus(nexternal)
c This is an array with (simplified) PDG codes for the particles. Note
c that channels that are combined (i.e. they have the same matrix
c elements) are given only 1 set of PDG codes. This means, e.g., that
c when using a 5-flavour scheme calculation (massless b quark), no
c b-tagging can be applied.
      integer iPDG(nexternal)
c The array of the momenta and masses of the initial and final state
c particles in the lab frame. The format is "E, px, py, pz, mass", while
c the second dimension loops over the particles in the process. Note
c that these are the (n+1)-body particles; for the n-body there is one
c momenta equal to all zero's (this is not necessarily the last particle
c in the list). If one uses IR-safe obserables only, there should be no
c difficulty in using this.
      double precision p(0:4,nexternal)
c
C     external functions that can be used. Some are defined in this
C     file, others are in ./Source/kin_functions.f
      REAL*8 R2_04,invm2_04,pt_04,eta_04,pt,eta
      external R2_04,invm2_04,pt_04,eta_04,pt,eta
C     recombination of photons
      double precision p_reco(0:4,nexternal), R_reco
      integer iPDG_reco(nexternal),nphiso
c local integers
      integer i,j
c bare parton algorithm
      integer nPART
      double precision pPART(0:3,nexternal)
c jet cluster algorithm
      integer nQCD,NJET,JET(nexternal)
      double precision pQCD(0:3,nexternal),PJET(0:3,nexternal)
      integer njet_eta
      double precision pgamma(0:3,nexternal),pgamma_iso(0:3,nexternal)
      integer nph
      include "run.inc" ! includes the ickkw parameter
c logicals that define if particles are leptons, jets or photons. These
c are filled from the PDG codes (iPDG array) in this function.
      logical is_a_lp(nexternal),is_a_lm(nexternal),is_a_j(nexternal)
     $,is_a_ph(nexternal),is_nph_iso(nexternal),is_nextph_iso(nexternal)
     $,is_nextph_iso_reco(nexternal)
      logical is_a_lp_reco(nexternal),is_a_lm_reco(nexternal)
      logical passcuts_leptons, passcuts_unlops_jv, passcuts_photons, 
     $        passcuts_jets, passcuts_pdgs,passcuts_fxfx

      passcuts_user=.true. ! event is okay; otherwise it is changed

C***************************************************************
C***************************************************************
C Cuts from the run_card.dat
C***************************************************************
C***************************************************************

      ! Find the bare QCD partons
      ! This is used only as input to the photon isolation
      call identify_PART_partons(p,istatus,ipdg,pPART,nPART,is_a_lp,is_a_lm)

      ! Apply the Photon cuts on isolated photons based on the bare particles
      passcuts_user = passcuts_user .and.
     $ passcuts_photons(p,istatus,ipdg,is_a_lp,is_a_lm,pPART,nPART,
     $ pgamma,nph,is_nph_iso,is_nextph_iso)
      if (.not.passcuts_user) return

      ! Recombine the photons and fermions
      call recombine_momenta(rphreco, etaphreco, lepphreco, quarkphreco,
     $     p, iPDG, is_nextph_iso,  p_reco, iPDG_reco, is_nextph_iso_reco)

      ! Apply the reco lepton cuts
      passcuts_user = passcuts_user .and. 
     $                  passcuts_leptons(p_reco,istatus,ipdg_reco,is_a_lp_reco,is_a_lm_reco)
      if (.not.passcuts_user) return

      ! Find the reco QCD partons including 
      ! A. All photons if gamma_is_j is on
      ! B. Non-iso, non-reco photons if reco is on
      call identify_QCD_partons(is_nextph_iso_reco,p_reco,istatus,ipdg_reco,is_a_j,pQCD,nQCD)

      ! Apply the UNLOPS/JetVeto cuts
      passcuts_user = passcuts_user .and. 
     $                  passcuts_unlops_jv(p_reco,istatus,ipdg_reco,pQCD,nQCD,ickkw)
      if (.not.passcuts_user) return

      ! Apply the Jet cuts
      if (ickkw.ne.3) then
         passcuts_user = passcuts_user .and. 
     $                  passcuts_jets(p_reco,pQCD,nQCD,pgamma,nph,is_nph_iso,ickkw)
         if (.not.passcuts_user) return
      else
         passcuts_user=passcuts_user .and.
     $                  passcuts_fxfx(p_reco,pQCD,nQCD)
         if (.not.passcuts_user) return
      endif

      ! Apply PDG specific cuts
      passcuts_user = passcuts_user .and. 
     $                  passcuts_pdgs(p_reco,istatus,ipdg_reco)
      if (.not.passcuts_user) return

C***************************************************************
C***************************************************************
C PUT HERE YOUR USER-DEFINED CUTS
C***************************************************************
C***************************************************************
C
c$$$C EXAMPLE: cut on top quark pT
c$$$C          Note that PDG specific cut are more optimised than simple user cut
c$$$      do i=1,nexternal   ! loop over all external particles
c$$$         if (istatus(i).eq.1    ! final state particle
c$$$     &        .and. abs(ipdg(i)).eq.6) then    ! top quark
c$$$C apply the pT cut (pT should be large than 200 GeV for the event to
c$$$C pass cuts)
c$$$            if ( p(1,i)**2+p(2,i)**2 .lt. 200d0**2 ) then
c$$$C momenta do not pass cuts. Set passcuts_user to false and return
c$$$               passcuts_user=.false.
c$$$               return
c$$$            endif
c$$$         endif
c$$$      enddo
c
      return
      end

      subroutine identify_PART_partons(p,istatus,ipdg,pPART,nPART,is_a_lp,is_a_lm)
      implicit none
      include 'nexternal.inc'
      integer istatus(nexternal)
      integer iPDG(nexternal)
      double precision p(0:4,nexternal)
      integer nPART
      double precision pPART(0:3,nexternal)
      logical is_a_lp(nexternal),is_a_lm(nexternal)
      include "run.inc"
      include "cuts.inc"

      integer i, j
c
c Bare partons and leptons
c
      nPART=0
      do j=1,nexternal
         is_a_lp(j)=.false.
         is_a_lm(j)=.false.
c Partons
        if (istatus(j).eq.1 .and.
     &     (abs(ipdg(j)).le.maxjetflavor .or. ipdg(j).eq.21)
     &) then
            nPART=nPART+1
            do i=0,3
               pPART(i,nPART)=p(i,j)
            enddo
        endif
c Leptons
         if (ipdg(j).eq.11 .or. ipdg(j).eq.13
     $      .or.  ipdg(j).eq.15) then
            is_a_lm(j)=.true.
         endif
         if (ipdg(j).eq.-11 .or. ipdg(j).eq.-13
     $      .or.  ipdg(j).eq.-15) then
            is_a_lp(j)=.true.
         endif
      enddo

      return
      end

      logical function passcuts_photons(p,istatus,ipdg,is_a_lp,is_a_lm,
     $pPART,nPART,pgamma,nph,is_nph_iso,is_nextph_iso)
      implicit none
      include 'nexternal.inc'
      integer istatus(nexternal)
      integer iPDG(nexternal)
      double precision p(0:4,nexternal)
      logical is_a_lp(nexternal),is_a_lm(nexternal)
      integer nPART, nph
      double precision pPART(0:3,nexternal), pgamma(0:3,nexternal)
      double precision pgamma_iso(0:3,nexternal)
      logical is_nph_iso(nexternal),is_nextph_iso(nexternal)
      include "cuts.inc"
      include "run.inc"
      integer i,j,k,mu
c Sort array of results: ismode>0 for real, isway=0 for ascending order
      integer ismode,isway,izero,isorted(nexternal)
      parameter (ismode=1)
      parameter (isway=0)
      parameter (izero=0)

c Photon isolation
      integer nem,nin,nphiso
      double precision ptg,chi_gamma_iso,iso_getdrv40
      double precision Etsum(0:nexternal)
      real drlist(nexternal)
      double precision pem(0:3,nexternal)

      logical alliso,isolated
      integer get_n_tagged_photons
      logical is_a_ph(nexternal)

      REAL*8 pt,eta
      external pt,eta

      include "orders.inc"
      logical split_type_used(nsplitorders)
      common/to_split_type_used/split_type_used

      integer n_needed_photons
 
      passcuts_photons = .true.

c
c PHOTON (ISOLATION) CUTS
c
c Initialise common logical iso
      do i=nincoming+1,nexternal
        is_nextph_iso(i)=.False.
      enddo
c find the photons
      do i=nincoming+1,nexternal
         if (ipdg(i).eq.22 .and. .not.gamma_is_j) then
            is_a_ph(i)=.true.
         else
            is_a_ph(i)=.false.
         endif
      enddo

      if (ptgmin.ne.0d0) then
         nph=0
         do j=nincoming+1,nexternal
            if (is_a_ph(j)) then
               nph=nph+1
               do i=0,3
                  pgamma(i,nph)=p(i,j)
               enddo
            endif
         enddo
         if(nph.eq.0) return
         
         if(isoEM)then
            nem=nph
            do k=1,nem
               do i=0,3
                  pem(i,k)=pgamma(i,k)
               enddo
            enddo
            do j=nincoming+1,nexternal
               if (is_a_lp(j).or.is_a_lm(j)) then
                  nem=nem+1
                  do i=0,3
                     pem(i,nem)=p(i,j)
                  enddo
               endif
            enddo
         endif
         
         nphiso=0

         j=0
c Loop over all photons
         do while(j.lt.nph)

            j=j+1
            is_nph_iso(j)=.False. 
            ptg=pt(pgamma(0,j))
            if(ptg.lt.ptgmin)then
               cycle
            endif
            if (etagamma.gt.0d0) then
               if (abs(eta(pgamma(0,j))).gt.etagamma) then
                  cycle
               endif
            endif
         
c Isolate from hadronic energy
            do i=1,nPART
               drlist(i)=sngl(iso_getdrv40(pgamma(0,j),pPART(0,i)))
            enddo
            call sortzv(drlist,isorted,nPART,ismode,isway,izero)
            Etsum(0)=0.d0
            nin=0
            do i=1,nPART
               if(dble(drlist(isorted(i))).le.R0gamma)then
                  nin=nin+1
                  Etsum(nin)=Etsum(nin-1)+pt(pPART(0,isorted(i)))
               endif
            enddo
            isolated=.True.
            do i=1,nin
               if(Etsum(i).gt.chi_gamma_iso(dble(drlist(isorted(i))),
     $             R0gamma,xn,epsgamma,ptg)) then
                   isolated=.False.
                   exit
               endif
            enddo
            if(.not.isolated)cycle
            
c Isolate from EM energy
            if(isoEM.and.nem.gt.1)then
               do i=1,nem
                  drlist(i)=sngl(iso_getdrv40(pgamma(0,j),pem(0,i)))
               enddo
               call sortzv(drlist,isorted,nem,ismode,isway,izero)
c First of list must be the photon: check this, and drop it
                 if(isorted(1).ne.j.or.drlist(isorted(1)).gt.1.e-4)then
                    write(*,*)'Error #1 in photon isolation'
                    write(*,*)j,isorted(1),drlist(isorted(1))
                    stop
                 endif
               Etsum(0)=0.d0
               nin=0
               do i=2,nem
                  if(dble(drlist(isorted(i))).le.R0gamma)then
                     nin=nin+1
                     Etsum(nin)=Etsum(nin-1)+pt(pem(0,isorted(i)))
                  endif
               enddo
               isolated=.True.
               do i=1,nin
                 if(Etsum(i).gt.chi_gamma_iso(dble(drlist(isorted(i))),
     $               R0gamma,xn,epsgamma,ptg)) then
                    isolated=.False.
                    exit
                 endif
               enddo
               if(.not.isolated)cycle
            endif
            is_nph_iso(j)=.True.
            nphiso=nphiso+1

           if (nphiso.gt.0) then
             do mu=0,3
               pgamma_iso(mu,nphiso)=pgamma(mu,j)
             enddo

             do i=nincoming+1,nexternal
               if ( ipdg(i).eq.22 .and. 
     $              pt(p(0,i)).eq.pt(pgamma_iso(0,nphiso)) ) then
                 is_nextph_iso(i)=.True.
               endif
             enddo
           endif
         enddo
c End of loop over photons

C now check that there are enough photons
         if (split_type_used(QED_pos)) then
         ! if the process has QED splittings, use the 
         ! get_n_tagged_photons function
             n_needed_photons = get_n_tagged_photons()
         else
         ! otherwise, just use the number of photons
         ! that has been counted
             n_needed_photons = nph
         endif

         if(nphiso.lt.n_needed_photons)then
            passcuts_photons=.false.
            return
         endif
      endif

      return
      end


      subroutine identify_QCD_partons(is_iso,p,istatus,ipdg,is_a_j,pQCD,nQCD)
      implicit none
      include 'nexternal.inc'
      integer istatus(nexternal)
      integer iPDG(nexternal)
      double precision p(0:4,nexternal)
      logical is_a_j(nexternal)
      integer nQCD
      double precision pQCD(0:3,nexternal)
      logical is_iso(nexternal)
      REAL*8 pt,eta
      external pt,eta
      include "run.inc" 
      include "cuts.inc"

      integer i, j 

c
c JET CUTS
c
c find the jets
      do i=1,nexternal
         if (istatus(i).eq.1 .and.
     &        (  abs(ipdg(i)).le.maxjetflavor .or. ipdg(i).eq.21
     &         .or. (ipdg(i).eq.22.and.gamma_is_j) .or.
     &         (ipdg(i).eq.22.and. .not.is_iso(i))  )
     &) then
            is_a_j(i)=.true.
         else
            is_a_j(i)=.false.
         endif
      enddo
c If we do not require a mimimum jet energy, there's no need to apply
c jet clustering and all that.
      if (ptj.ne.0d0.or.ptgmin.ne.0d0) then
c Put all (light) QCD partons in momentum array for jet clustering.
c From the run_card.dat, maxjetflavor defines if b quark should be
c considered here (via the logical variable 'is_a_jet').  nQCD becomes
c the number of (light) QCD partons at the real-emission level (i.e. one
c more than the Born).
         nQCD=0
         do j=nincoming+1,nexternal
            if (is_a_j(j)) then
               nQCD=nQCD+1
               do i=0,3
                  pQCD(i,nQCD)=p(i,j)
               enddo
            endif
         enddo
      endif

      return
      end


      logical function passcuts_fxfx(p,pQCD,nQCD)
c In case of FxFx merging, use the lowest clustering scale to apply the cut
      implicit none
      include 'nexternal.inc'
      include 'cuts.inc'
      double precision p(0:4,nexternal)
      integer nQCD
      double precision pQCD(0:3,nexternal)
      integer NJET,JET(nexternal)
      double precision rfj,sycut,palg,amcatnlo_fastjetdmerge,etaj_max
      double precision PJET(0:3,nexternal)
      integer nFxFx_ren_scales
      double precision FxFx_ren_scales(0:nexternal),
     $                 FxFx_fac_scale(2)
      common/c_FxFx_scales/FxFx_ren_scales,nFxFx_ren_scales,
     $                     FxFx_fac_scale
      passcuts_fxfx=.true.
c First apply a numerical stability cut
c Define jet clustering parameters with a pTmin=1 GeV
      palg=1.0                  ! jet algorithm: 1.0=kt, 0.0=C/A, -1.0 = anti-kt
      rfj=1.0                   ! the radius parameter
      sycut=1.0                 ! minimum transverse momentum
      etaj_max=1000d0
c     call FASTJET to get all the jets
      call amcatnlo_fastjetppgenkt_etamax_timed(
     $     pQCD,nQCD,rfj,sycut,etaj_max,palg,pjet,njet,jet)
c Apply the jet cut
      if (njet .ne. nQCD .and. njet .ne. nQCD-1) then
         passcuts_fxfx=.false.
         return
      endif
c Second apply the actual ptj cut on the minimum FxFx_ren_scales(i)
      if (minval(FxFx_ren_scales(0:nFxFx_ren_scales)).lt.ptj) then
         passcuts_fxfx=.false.
         return
      endif
      return
      end
      
      logical function passcuts_jets(p,pQCD,nQCD,pgamma,nph,is_nph_iso,ickkw)
      implicit none
      include 'nexternal.inc'
      double precision p(0:4,nexternal)
      integer nQCD, nph
      double precision pQCD(0:3,nexternal), pgamma(0:3,nexternal)
      logical is_nph_iso(nexternal)
      integer ickkw
      include "cuts.inc"

      integer NJET,JET(nexternal)
      double precision rfj,sycut,palg,amcatnlo_fastjetdmerge
      double precision PJET(0:3,nexternal)
      integer mm
      integer i,j

      integer get_n_tagged_photons

      REAL*8 pt,eta
      external pt,eta

      passcuts_jets=.true.

c JET CUTS

C       do nothing if ickkw=4 (UNLOPS)
      if (ickkw.eq.4)return

      if (ptj.gt.0d0.and.nQCD.gt.1) then

c Cut some peculiar momentum configurations, i.e. two partons very soft.
c This is needed to get rid of numerical instabilities in the Real emission
c matrix elements when the Born has a massless final-state parton, but
c no possible divergence related to it (e.g. t-channel single top)
         mm=0
         do j=1,nQCD
            if(abs(pQCD(0,j)/p(0,1)).lt.1.d-8) mm=mm+1
         enddo
         if(mm.gt.1)then
            passcuts_jets=.false.
            return
         endif

c Define jet clustering parameters (from cuts.inc via the run_card.dat)
           palg=JETALGO         ! jet algorithm: 1.0=kt, 0.0=C/A, -1.0 = anti-kt
           rfj=JETRADIUS        ! the radius parameter
           sycut=ptj            ! minimum transverse momentum

c******************************************************************************
c     call FASTJET to get all the jets
c
c     INPUT:
c     input momenta:               pQCD(0:3,nexternal), energy is 0th component
c     number of input momenta:     nQCD
c     radius parameter:            rfj
c     minumum jet pt:              sycut
c     jet algorithm:               palg, 1.0=kt, 0.0=C/A, -1.0 = anti-kt
c
c     OUTPUT:
c     jet momenta:                           pjet(0:3,nexternal), E is 0th cmpnt
c     the number of jets (with pt > SYCUT):  njet
c     the jet for a given particle 'i':      jet(i),   note that this is the
c                                            particle in pQCD, which doesn't
c                                            necessarily correspond to the particle
c                                            label in the process
c
           call amcatnlo_fastjetppgenkt_etamax_timed(
     $          pQCD,nQCD,rfj,sycut,etaj,palg,pjet,njet,jet)
c
c******************************************************************************

c Apply the jet cuts
         if (njet .ne. nQCD .and. njet .ne. nQCD-1) then
            passcuts_jets=.false.
            return
         endif
      endif

      return
      end



      logical function passcuts_unlops_jv(p,istatus,ipdg,pQCD,nQCD,ickkw)
      implicit none
      include 'nexternal.inc'
      integer istatus(nexternal)
      integer iPDG(nexternal)
      double precision p(0:4,nexternal)
      logical is_a_j(nexternal)
      integer nQCD
      double precision pQCD(0:3,nexternal)
      integer ickkw
      include "cuts.inc"
      double precision p_unlops(0:3,nexternal)
      logical passUNLOPScuts
      integer i, j 

      REAL*8 pt
      external pt

      passcuts_unlops_jv=.true.


c THE UNLOPS CUT:
      if (ickkw.eq.4 .and. ptj.gt.0d0) then
c Use special pythia pt cut for minimal pT
         do i=1,nexternal
            do j=0,3
               p_unlops(j,i)=p(j,i)
            enddo
         enddo
         call pythia_UNLOPS(p_unlops,passUNLOPScuts)
         if (.not. passUNLOPScuts) then
            passcuts_unlops_jv=.false.
            return
         endif
c THE VETO XSEC CUT:
      elseif (ickkw.eq.-1 .and. ptj.gt.0d0) then
c Use veto'ed Xsec for analytic NNLL resummation
         if (nQCD.ne.1) then
            write (*,*) 'ERROR: more than one QCD parton in '/
     $           /'this event in cuts.f. There should only be one'
            stop
         endif
         if (pt(pQCD(0,1)) .gt. ptj) then
            passcuts_unlops_jv=.false.
            return
         endif
      endif
      return
      end



      logical function passcuts_leptons(p,istatus,ipdg,is_a_lp_reco,is_a_lm_reco)
      implicit none
      include 'nexternal.inc'
      integer istatus(nexternal)
      integer iPDG(nexternal)
      double precision p(0:4,nexternal)
      logical is_a_lp_reco(nexternal),is_a_lm_reco(nexternal)

      REAL*8 R2_04,invm2_04,pt_04,eta_04,pt,eta
      external R2_04,invm2_04,pt_04,eta_04,pt,eta
      integer i,j

      include 'cuts.inc'

      passcuts_leptons=.true.

c
c CHARGED LEPTON CUTS
c
c find the charged leptons (also used in the photon isolation cuts below)
      do i=1,nexternal
         if(istatus(i).eq.1 .and.
     &    (ipdg(i).eq.11 .or. ipdg(i).eq.13 .or. ipdg(i).eq.15)) then
            is_a_lm_reco(i)=.true.
         else
            is_a_lm_reco(i)=.false.
         endif
         if(istatus(i).eq.1 .and.
     &    (ipdg(i).eq.-11 .or. ipdg(i).eq.-13 .or. ipdg(i).eq.-15)) then
            is_a_lp_reco(i)=.true.
         else
            is_a_lp_reco(i)=.false.
         endif
      enddo
c apply the charged lepton cuts
      do i=nincoming+1,nexternal
         if (is_a_lp_reco(i).or.is_a_lm_reco(i)) then
c transverse momentum
            if (ptl.gt.0d0) then
               if (pt_04(p(0,i)).lt.ptl) then
                  passcuts_leptons=.false.
                  return
               endif
            endif
c pseudo-rapidity
            if (etal.gt.0d0) then
               if (abs(eta_04(p(0,i))).gt.etal) then
                  passcuts_leptons=.false.
                  return
               endif
            endif
c DeltaR and invariant mass cuts
            if (is_a_lp_reco(i)) then
               do j=nincoming+1,nexternal
                  if (is_a_lm_reco(j)) then
                     if (drll.gt.0d0) then
                        if (R2_04(p(0,i),p(0,j)).lt.drll**2) then
                           passcuts_leptons=.false.
                           return
                        endif
                     endif
                     if (mll.gt.0d0) then
                        if (invm2_04(p(0,i),p(0,j),1d0).lt.mll**2) then
                           passcuts_leptons=.false.
                           return
                        endif
                     endif
                     if (ipdg(i).eq.-ipdg(j)) then
                        if (drll_sf.gt.0d0) then
                           if (R2_04(p(0,i),p(0,j)).lt.drll_sf**2) then
                              passcuts_leptons=.false.
                              return
                           endif
                        endif
                        if (mll_sf.gt.0d0) then
                           if (invm2_04(p(0,i),p(0,j),1d0).lt.mll_sf**2)
     $                          then
                              passcuts_leptons=.false.
                              return
                           endif
                        endif
                     endif
                  endif
               enddo
            endif
         endif
      enddo

      return
      end

      logical function passcuts_pdgs(p,istatus,ipdg)
      implicit none
      include 'nexternal.inc'
      integer istatus(nexternal)
      integer iPDG(nexternal)
      double precision p(0:4,nexternal)
c PDG specific cut
      double precision etmin(nincoming+1:nexternal-1)
      double precision etmax(nincoming+1:nexternal-1)
      double precision mxxmin(nincoming+1:nexternal-1,nincoming+1:nexternal-1)
      common /to_cuts/etmin,etmax,mxxmin
      REAL*8 invm2_04,pt_04
      external invm2_04,pt_04
c temporary variable for caching locally computation
      double precision tmpvar
      integer i,j

      passcuts_pdgs = .true.


C
C     PDG SPECIFIC CUTS (PT/M_IJ)
C
      do i=nincoming+1,nexternal-1
         if(etmin(i).gt.0d0 .or. etmax(i).gt.0d0)then
            tmpvar = pt_04(p(0,i))
            if (tmpvar.lt.etmin(i)) then
               passcuts_pdgs=.false.
               return
            elseif (tmpvar.gt.etmax(i) .and. etmax(i).gt.0d0) then
               passcuts_pdgs=.false.
               return
            endif
         endif
         do j=i+1, nexternal-1
            if (mxxmin(i,j).gt.0d0)then
               if (invm2_04(p(0,i),p(0,j),1d0).lt.mxxmin(i,j)**2)then
                  passcuts_pdgs=.false.
                  return
               endif
            endif
         enddo  
      enddo
      return  
      end 


C***************************************************************
C***************************************************************
C NO NEED TO CHANGE ANY OF THE FUNCTIONS BELOW
C***************************************************************
C***************************************************************
      logical function passcuts(p,rwgt)
      implicit none
      include "nexternal.inc"
      include 'run.inc'
      include 'genps.inc'
      include 'cuts.inc'
      include 'timing_variables.inc'
      REAL*8 P(0:3,nexternal),rwgt
      integer i,j,istatus(nexternal),iPDG(nexternal)
c For boosts
      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat
      double precision chybst,shybst,chybstmo
      double precision xd(1:3)
      data (xd(i),i=1,3)/0,0,1/
c Momenta of the particles
      double precision plab(0:3, nexternal),pp(0:4, nexternal)
c Masses of external particles
      double precision pmass(nexternal)
      common/to_mass/pmass
c PDG codes of particles
      integer idup(nexternal,maxproc),mothup(2,nexternal,maxproc),
     &     icolup(2,nexternal,maxflow),niprocs
      common /c_leshouche_inc/idup,mothup,icolup,niprocs
      logical passcuts_user
      external passcuts_user
      call cpu_time(tBefore)
c Make sure have reasonable 4-momenta
      if (p(0,1) .le. 0d0) then
         passcuts=.false.
         return
      endif
c Also make sure there's no INF or NAN
      do i=1,nexternal
         do j=0,3
            if(p(j,i).gt.1d32.or.p(j,i).ne.p(j,i))then
               passcuts=.false.
               return
            endif
         enddo
      enddo

      rwgt=1d0
c Boost the momenta p(0:3,nexternal) to the lab frame plab(0:3,nexternal)
      chybst=cosh(ybst_til_tolab)
      shybst=sinh(ybst_til_tolab)
      chybstmo=chybst-1.d0
      do i=1,nexternal
         call boostwdir2(chybst,shybst,chybstmo,xd,
     &        p(0,i),plab(0,i))
      enddo
c Fill the arrays (momenta, status and PDG):
      do i=1,nexternal
         if (i.le.nincoming) then
            istatus(i)=-1
         else
            istatus(i)=1
         endif
         do j=0,3
            pp(j,i)=plab(j,i)
         enddo
         pp(4,i)=pmass(i)
         ipdg(i)=idup(i,1)
         if (ipdg(i).eq.-21) ipdg(i)=21
      enddo
c Call the actual cuts function  
      passcuts = passcuts_user(pp,istatus,ipdg)
      call cpu_time(tAfter)
      t_cuts=t_cuts+(tAfter-tBefore)
      RETURN
      END


      function chi_gamma_iso(dr,R0,xn,epsgamma,pTgamma)
c Eq.(3.4) of Phys.Lett. B429 (1998) 369-374 [hep-ph/9801442]
      implicit none
      real*8 chi_gamma_iso,dr,R0,xn,epsgamma,pTgamma
      real*8 tmp,axn
c
      axn=abs(xn)
      tmp=epsgamma*pTgamma
      if(axn.ne.0.d0)then
        tmp=tmp*( (1-cos(dr))/(1-cos(R0)) )**axn
      endif
      chi_gamma_iso=tmp
      return
      end


*
* $Id: sortzv.F,v 1.1.1.1 1996/02/15 17:49:50 mclareni Exp $
*
* $Log: sortzv.F,v $
* Revision 1.1.1.1  1996/02/15 17:49:50  mclareni
* Kernlib
*
*
c$$$#include "kerngen/pilot.h"
      SUBROUTINE SORTZV (A,INDEX,N1,MODE,NWAY,NSORT)
C
C CERN PROGLIB# M101    SORTZV          .VERSION KERNFOR  3.15  820113
C ORIG. 02/10/75
C
      DIMENSION A(N1),INDEX(N1)
C
C
      N = N1
      IF (N.LE.0)            RETURN
      IF (NSORT.NE.0) GO TO 2
      DO 1 I=1,N
    1 INDEX(I)=I
C
    2 IF (N.EQ.1)            RETURN
      IF (MODE)    10,20,30
   10 STOP 5 ! CALL SORTTI (A,INDEX,N)
      GO TO 40
C
   20 STOP 5 ! CALL SORTTC(A,INDEX,N)
      GO TO 40
C
   30 CALL SORTTF (A,INDEX,N)
C
   40 IF (NWAY.EQ.0) GO TO 50
      N2 = N/2
      DO 41 I=1,N2
      ISWAP = INDEX(I)
      K = N+1-I
      INDEX(I) = INDEX(K)
   41 INDEX(K) = ISWAP
   50 RETURN
      END
*     ========================================
      SUBROUTINE SORTTF (A,INDEX,N1)
C
      DIMENSION A(N1),INDEX(N1)
C
      N = N1
      DO 3 I1=2,N
      I3 = I1
      I33 = INDEX(I3)
      AI = A(I33)
    1 I2 = I3/2
      IF (I2) 3,3,2
    2 I22 = INDEX(I2)
      IF (AI.LE.A (I22)) GO TO 3
      INDEX (I3) = I22
      I3 = I2
      GO TO 1
    3 INDEX (I3) = I33
    4 I3 = INDEX (N)
      INDEX (N) = INDEX (1)
      AI = A(I3)
      N = N-1
      IF (N-1) 12,12,5
    5 I1 = 1
    6 I2 = I1 + I1
      IF (I2.LE.N) I22= INDEX(I2)
      IF (I2-N) 7,9,11
    7 I222 = INDEX (I2+1)
      IF (A(I22)-A(I222)) 8,9,9
    8 I2 = I2+1
      I22 = I222
    9 IF (AI-A(I22)) 10,11,11
   10 INDEX(I1) = I22
      I1 = I2
      GO TO 6
   11 INDEX (I1) = I3
      GO TO 4
   12 INDEX (1) = I3
      RETURN
      END
*     ========================================
      SUBROUTINE SORTTI (A,INDEX,N1)
C
      INTEGER A,AI
      DIMENSION A(N1),INDEX(N1)
C
      N = N1
      DO 3 I1=2,N
      I3 = I1
      I33 = INDEX(I3)
      AI = A(I33)
    1 I2 = I3/2
      IF (I2) 3,3,2
    2 I22 = INDEX(I2)
      IF (AI.LE.A (I22)) GO TO 3
      INDEX (I3) = I22
      I3 = I2
      GO TO 1
    3 INDEX (I3) = I33
    4 I3 = INDEX (N)
      INDEX (N) = INDEX (1)
      AI = A(I3)
      N = N-1
      IF (N-1) 12,12,5
    5 I1 = 1
    6 I2 = I1 + I1
      IF (I2.LE.N) I22= INDEX(I2)
      IF (I2-N) 7,9,11
    7 I222 = INDEX (I2+1)
      IF (A(I22)-A(I222)) 8,9,9
    8 I2 = I2+1
      I22 = I222
    9 IF (AI-A(I22)) 10,11,11
   10 INDEX(I1) = I22
      I1 = I2
      GO TO 6
   11 INDEX (I1) = I3
      GO TO 4
   12 INDEX (1) = I3
      RETURN
      END
*     ========================================
      SUBROUTINE SORTTC (A,INDEX,N1)
C
      INTEGER A,AI
      DIMENSION A(N1),INDEX(N1)
C
      N = N1
      DO 3 I1=2,N
      I3 = I1
      I33 = INDEX(I3)
      AI = A(I33)
    1 I2 = I3/2
      IF (I2) 3,3,2
    2 I22 = INDEX(I2)
      IF(ICMPCH(AI,A(I22)))3,3,21
   21 INDEX (I3) = I22
      I3 = I2
      GO TO 1
    3 INDEX (I3) = I33
    4 I3 = INDEX (N)
      INDEX (N) = INDEX (1)
      AI = A(I3)
      N = N-1
      IF (N-1) 12,12,5
    5 I1 = 1
    6 I2 = I1 + I1
      IF (I2.LE.N) I22= INDEX(I2)
      IF (I2-N) 7,9,11
    7 I222 = INDEX (I2+1)
      IF (ICMPCH(A(I22),A(I222))) 8,9,9
    8 I2 = I2+1
      I22 = I222
    9 IF (ICMPCH(AI,A(I22))) 10,11,11
   10 INDEX(I1) = I22
      I1 = I2
      GO TO 6
   11 INDEX (I1) = I3
      GO TO 4
   12 INDEX (1) = I3
      RETURN
      END
*     ========================================
      FUNCTION ICMPCH(IC1,IC2)
C     FUNCTION TO COMPARE TWO 4 CHARACTER EBCDIC STRINGS - IC1,IC2
C     ICMPCH=-1 IF HEX VALUE OF IC1 IS LESS THAN IC2
C     ICMPCH=0  IF HEX VALUES OF IC1 AND IC2 ARE THE SAME
C     ICMPCH=+1 IF HEX VALUES OF IC1 IS GREATER THAN IC2
      I1=IC1
      I2=IC2
      IF(I1.GE.0.AND.I2.GE.0)GOTO 40
      IF(I1.GE.0)GOTO 60
      IF(I2.GE.0)GOTO 80
      I1=-I1
      I2=-I2
      IF(I1-I2)80,70,60
 40   IF(I1-I2)60,70,80
 60   ICMPCH=-1
      RETURN
 70   ICMPCH=0
      RETURN
 80   ICMPCH=1
      RETURN
      END


      function iso_getdrv40(p1,p2)
      implicit none
      real*8 iso_getdrv40,p1(0:3),p2(0:3)
      real*8 iso_getdr
c
      iso_getdrv40=iso_getdr(p1(0),p1(1),p1(2),p1(3),
     #                       p2(0),p2(1),p2(2),p2(3))
      return
      end


      function iso_getdr(en1,ptx1,pty1,pl1,en2,ptx2,pty2,pl2)
      implicit none
      real*8 iso_getdr,en1,ptx1,pty1,pl1,en2,ptx2,pty2,pl2,deta,dphi,
     # iso_getpseudorap,iso_getdelphi
c
      deta=iso_getpseudorap(en1,ptx1,pty1,pl1)-
     #     iso_getpseudorap(en2,ptx2,pty2,pl2)
      dphi=iso_getdelphi(ptx1,pty1,ptx2,pty2)
      iso_getdr=sqrt(dphi**2+deta**2)
      return
      end


      function iso_getpseudorap(en,ptx,pty,pl)
      implicit none
      real*8 iso_getpseudorap,en,ptx,pty,pl,tiny,pt,eta,th
      parameter (tiny=1.d-5)
c
      pt=sqrt(ptx**2+pty**2)
      if(pt.lt.tiny.and.abs(pl).lt.tiny)then
        eta=sign(1.d0,pl)*1.d8
      else
        th=atan2(pt,pl)
        eta=-log(tan(th/2.d0))
      endif
      iso_getpseudorap=eta
      return
      end


      function iso_getdelphi(ptx1,pty1,ptx2,pty2)
      implicit none
      real*8 iso_getdelphi,ptx1,pty1,ptx2,pty2,tiny,pt1,pt2,tmp
      parameter (tiny=1.d-5)
c
      pt1=sqrt(ptx1**2+pty1**2)
      pt2=sqrt(ptx2**2+pty2**2)
      if(pt1.ne.0.d0.and.pt2.ne.0.d0)then
        tmp=ptx1*ptx2+pty1*pty2
        tmp=tmp/(pt1*pt2)
        if(abs(tmp).gt.1.d0+tiny)then
          write(*,*)'Cosine larger than 1'
          stop
        elseif(abs(tmp).ge.1.d0)then
          tmp=sign(1.d0,tmp)
        endif
        tmp=acos(tmp)
      else
        tmp=1.d8
      endif
      iso_getdelphi=tmp
      return
      end



      DOUBLE PRECISION FUNCTION R2_04(P1,P2)
c************************************************************************
c     Distance in eta,phi between two particles.
c************************************************************************
      IMPLICIT NONE
c
c     Arguments
c
      double precision p1(0:4),p2(0:4),p1a(0:3),p2a(0:3)
      integer i
c
c     External
c
      double precision eta,DELTA_PHI
      external eta,delta_phi
c-----
c  Begin Code
c-----
      do i=0,3
         p1a(i)=p1(i)
         p2a(i)=p2(i)
      enddo
      R2_04 = (DELTA_PHI(P1a,P2a))**2+(eta(p1a)-eta(p2a))**2
      RETURN
      END

      double precision function pt_04(p)
c************************************************************************
c     Returns transverse momentum of particle
c************************************************************************
      IMPLICIT NONE
c
c     Arguments
c
      double precision p(0:4)
c-----
c  Begin Code
c-----

      pt_04 = dsqrt(p(1)**2+p(2)**2)

      return
      end


      double precision function eta_04(p)
c************************************************************************
c     Returns pseudo rapidity of particle
c************************************************************************
      IMPLICIT NONE
c
c     Arguments
c
      double precision p(0:4),pa(0:3)
      integer i
c
c     external
c
      double precision theta,tp,pi
      parameter (pi=3.14159265358979323846264338327950d0)
      external theta
c-----
c  Begin Code
c-----
      do i=0,3
         pa(i)=p(i)
      enddo
      tp=theta(pa)
      if (abs(tp).lt.1d-5) then
         eta_04=25d0
      elseif (abs(tp-pi).lt.1d-5) then
         eta_04=-25d0
      else
         eta_04=-dlog(dtan(theta(pa)/2d0))
      endif

      return
      end



      DOUBLE PRECISION FUNCTION invm2_04(P1,P2,dsign)
c************************************************************************
c     Invarient mass of 2 particles
c************************************************************************
      IMPLICIT NONE
c
c     Arguments
c
      double precision p1(0:4),p2(0:4),dsign
c
c     Local
c      
      integer i
      double precision ptot(0:3)
c
c     External
c
      double precision dot
      external dot
c-----
c  Begin Code
c-----

      do i=0,3
         ptot(i)=p1(i)+dsign*p2(i)
      enddo
      invm2_04 = dot(ptot,ptot)
      RETURN
      END


      subroutine get_ID_H(IDUP_tmp)
      implicit none
      include "genps.inc"
      include 'nexternal.inc'
      integer idup(nexternal,maxproc),mothup(2,nexternal,maxproc),
     &     icolup(2,nexternal,maxflow),niprocs
c      include 'leshouche.inc'
      common /c_leshouche_inc/idup,mothup,icolup,niprocs
      integer IDUP_tmp(nexternal),i
c
      do i=1,nexternal
         IDUP_tmp(i)=IDUP(i,1)
      enddo
c
      return
      end

      subroutine get_ID_S(IDUP_tmp)
      implicit none
      include "genps.inc"
      include 'nexternal.inc'
      integer idup(nexternal,maxproc)
      integer mothup(2,nexternal,maxproc)
      integer icolup(2,nexternal,maxflow)
      include 'born_leshouche.inc'
      integer IDUP_tmp(nexternal),i
c
      do i=1,nexternal-1
         IDUP_tmp(i)=IDUP(i,1)
      enddo
      IDUP_tmp(nexternal)=0
c
      return
      end


      subroutine bias_weight_function(p,ipdg,bias_wgt)
c This is a user-defined function to which to bias the event generation.
c A non-flat distribution will generate events with a certain weight
c inversely proportinal to the bias_wgt. This is particularly useful to
c generate more events (with smaller weight) in tails of distributions.
c It computes the bias_wgt factor from the momenta and multiplies the
c weight that goes into MINT (or vegas) with this factor.  Before
c writing out the events (or making the plots), this factor is again
c divided out. A value different from 1 makes that MINT (or vegas) does
c not list the correct cross section, but the cross section can still be
c computed from summing all the weights of the events (and dividing by
c the number of events). Since the weights of the events are no longer
c identical for all events, the statistical uncertainty on this total
c cross section can be much larger than without including the bias.
c
c The 'bias_wgt' should be a IR-safe function of the momenta.
c      
c For this to be used, the 'event_norm' option in the run_card should be
c set to
c      'bias' = event_norm      
c
      implicit none
      include 'nexternal.inc'
      double precision bias_wgt,p(0:3,nexternal),H_T
      integer ipdg(nexternal),i

      bias_wgt=1d0

c How to enhance the tails is very process dependent. For example for
c top quark production one could use:
c      do i=1,nexternal
c         if (ipdg(i).eq.6) then
c            bias_wgt=sqrt(p(1,i)**2+p(2,i)**2)**3
c         endif
c      enddo
c Or to use H_T^2 one does     
c      H_T=0d0
c      do i=3,nexternal
c         H_T=H_T+sqrt(max(0d0,(p(0,i)+p(3,i))*(p(0,i)-p(3,i))))
c      enddo
c      bias_wgt=H_T**2
      return
      end

      integer function get_n_tagged_photons()
      implicit none
      integer i
      include "nexternal.inc"
      logical particle_tag(nexternal)
      common /c_particle_tag/particle_tag
      get_n_tagged_photons = 0

      do i = nincoming+1, nexternal
        if (particle_tag(i))
     $     get_n_tagged_photons = get_n_tagged_photons+1
      enddo

      return
      end



