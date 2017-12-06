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
c
c     LOCAL
c
      integer i,j,k
C     
C     GLOBAL
C
c--masses and poles
      double precision pmass(nexternal)
      common/to_mass/  pmass
c
c     les houches accord stuff to identify particles
c
      integer idup(nexternal,maxproc),mothup(2,nexternal,maxproc),
     &     icolup(2,nexternal,maxflow),niprocs
      common /c_leshouche_inc/idup,mothup,icolup,niprocs
C
      LOGICAL  IS_A_J(NEXTERNAL),IS_A_LP(NEXTERNAL),IS_A_LM(NEXTERNAL)
      LOGICAL  IS_A_PH(NEXTERNAL)
      COMMON /TO_SPECISA/IS_A_J,IS_A_LP,IS_A_LM,IS_A_PH
c 
      double precision etmin(nincoming+1:nexternal-1)
      double precision etmax(nincoming+1:nexternal-1)
      double precision mxxmin(nincoming+1:nexternal-1,nincoming+1:nexternal-1)
      common /to_cuts/etmin,etmax, mxxmin
c
c     setup masses for the final-state particles (fills the /to_mass/ common block)
c
      include 'pmass.inc'

C-----
C  BEGIN CODE
C-----
c
c     No pdfs for decay processes - set here since here we know the nincoming
c     Also set stot here, and use mass of incoming particle for ren scale
c
         if(nincoming.eq.1)then
            lpp(1)=0
            lpp(2)=0
            ebeam(1)=pmass(1)/2d0
            ebeam(2)=pmass(1)/2d0
         endif
c-check consistency of maxjetflavor in the run_card and with Nf
c specified in coupl.inc
      if (maxjetflavor.lt.int(Nf)) then
         write(*,'(a,i3,a,i3)') "WARNING: the value of maxjetflavor"/
     $        /"specified in the run_card (",maxjetflavor,") is"/
     $        /" inconsistent with the number of light flavours in"/
     $        /"the model. Hence it will be set to:", int(Nf)
          maxjetflavor = int(Nf)
      endif

      do i=nincoming+1,nexternal
         is_a_j(i)=.false.
         is_a_lp(i)=.false.
         is_a_lm(i)=.false.
         is_a_ph(i)=.false.

c-light-jets
         if (abs(idup(i,1)).le.maxjetflavor) then
              is_a_j(i)=.true.
         endif
         if (abs(idup(i,1)).eq.21)  is_a_j(i)=.true. ! gluon is a jet

c-charged-leptons
         if (idup(i,1).eq.11)  is_a_lm(i)=.true. !  e-
         if (idup(i,1).eq.13)  is_a_lm(i)=.true. !  mu-
         if (idup(i,1).eq.15)  is_a_lm(i)=.true. !  ta-
         if (idup(i,1).eq.-11) is_a_lp(i)=.true. !  e-
         if (idup(i,1).eq.-13) is_a_lp(i)=.true. !  mu-
         if (idup(i,1).eq.-15) is_a_lp(i)=.true. !  ta-

c-photons
         if (idup(i,1).eq.22)  is_a_ph(i)=.true. !  photon
      enddo
c
c     check for pdg specific cut (pt/eta)
c
      do i=nincoming+1, nexternal-1 ! never include last particle
         etmin(i) = 0d0
         etmax(i) = -1d0
         do j = i, nexternal-1
            mxxmin(i,j) = 0d0
         enddo
      enddo

      if (pdg_cut(1).ne.0)then
         do j=1,pdg_cut(0)
            do i=nincoming+1, nexternal-1 ! never include last particle
               if (abs(idup(i,1)).ne.pdg_cut(j)) cycle
c     fully ensure that only massive particles are allowed at NLO
               if(pmass(i).eq.0d0) then
                  write(*,*) 'Illegal use of pdg specific cut.'
                  write(*,*) 'For NLO process, '/
     $                 /'only massive particle can be included'
                  stop 1
               endif
c     fully ensure that this is not a jet/lepton/photon
               if(is_a_lp(i) .or. is_a_lm(i) .or. is_a_j(i) .or.
     $              is_a_ph(i)) then
                  write(*,*) 'Illegal use of pdg specific cut.'
                  write(*,*) 'This can not be used for '/
     $                 /'jet/lepton/photon/gluon'
                  stop 1
               endif
               etmin(i) = ptmin4pdg(j)
               etmax(i) = ptmax4pdg(j)
!     add the invariant mass cut
               if(mxxmin4pdg(j).ne.0d0)then
                  do k=i+1, nexternal-1
                     if (mxxpart_antipart(j))then
                        if (idup(k, 1).eq.-1*idup(i,1))then
                           mxxmin(i,k) = mxxmin4pdg(j)
                        endif
                     else
                        if (abs(idup(k, 1)).eq.pdg_cut(j))then
                           mxxmin(i,k) = mxxmin4pdg(j)
                        endif
                     endif
                  enddo
               endif
            enddo
         enddo
      endif


      RETURN
      END


      subroutine set_tau_min()
c Sets the lower bound for tau=x1*x2, using information on particle
c masses and on the jet minimum pt, as entered in run_card.dat, 
c variable ptj
      implicit none
      double precision zero,vtiny
      parameter (zero=0.d0,vtiny=1d-8)
      include 'cuts.inc'
      include 'run.inc'
      include 'genps.inc'
      include 'nexternal.inc'
      include 'coupl.inc'
      include 'nFKSconfigs.inc'
      include "fks_info.inc"
      include "mint.inc"
      LOGICAL  IS_A_J(NEXTERNAL),IS_A_LP(NEXTERNAL),IS_A_LM(NEXTERNAL)
      LOGICAL  IS_A_PH(NEXTERNAL)
      COMMON /TO_SPECISA/IS_A_J,IS_A_LP,IS_A_LM,IS_A_PH
c
      double precision pmass(-nexternal:0,lmaxconfigs)
      double precision pwidth(-nexternal:0,lmaxconfigs)
      integer pow(-nexternal:0,lmaxconfigs)
      integer itree(2,-max_branch:-1),iconf
      common /to_itree/itree,iconf
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      double precision taumin(fks_configs,maxchannels)
     $     ,taumin_s(fks_configs,maxchannels),taumin_j(fks_configs
     $     ,maxchannels),stot,xk(nexternal)
      save  taumin,taumin_s,taumin_j
      integer i,j,k,d1,d2,iFKS,nt
      double precision xm(-nexternal:nexternal),xm1,xm2,xmi
      double precision xw(-nexternal:nexternal),xw1,xw2,xwi
      integer tsign,j_fks
      double precision tau_Born_lower_bound,tau_lower_bound_resonance
     &     ,tau_lower_bound
      common/ctau_lower_bound/tau_Born_lower_bound
     &     ,tau_lower_bound_resonance,tau_lower_bound
c BW stuff
      double precision mass_min(-nexternal:nexternal),masslow(
     $     -nexternal:-1),widthlow(-nexternal:-1),sum_all_s
      integer t_channel
      integer cBW_FKS_level_max(fks_configs,maxchannels),
     &     cBW_FKS(fks_configs,-nexternal:-1,maxchannels),
     &     cBW_FKS_level(fks_configs,-nexternal:-1,maxchannels)
      double precision cBW_FKS_mass(fks_configs,-1:1,-nexternal:-1
     $     ,maxchannels),cBW_FKS_width(fks_configs,-1:1,-nexternal:-1
     $     ,maxchannels)
      save cBW_FKS_level_max,cBW_FKS,cBW_FKS_level,cBW_FKS_mass
     $     ,cBW_FKS_width
      integer cBW_level_max,cBW(-nexternal:-1),cBW_level(-nexternal:-1)
      double precision cBW_mass(-1:1,-nexternal:-1),
     &     cBW_width(-1:1,-nexternal:-1)
      common/c_conflictingBW/cBW_mass,cBW_width,cBW_level_max,cBW
     $     ,cBW_level
      double precision s_mass(-nexternal:nexternal)
     $     ,s_mass_FKS(fks_configs,-nexternal:nexternal,maxchannels)
      save s_mass_FKS
      common/to_phase_space_s_channel/s_mass
c Les Houches common block
      integer idup(nexternal,maxproc),mothup(2,nexternal,maxproc),
     &     icolup(2,nexternal,maxflow),niprocs
      common /c_leshouche_inc/idup,mothup,icolup,niprocs
      real*8         emass(nexternal)
      common/to_mass/emass
c     block for the (simple) cut bsed on the pdg
      double precision etmin(nincoming+1:nexternal-1)
      double precision etmax(nincoming+1:nexternal-1)
      double precision mxxmin(nincoming+1:nexternal-1,nincoming+1:nexternal-1)
      common /to_cuts/etmin,etmax,mxxmin
      double precision smin_update , mxx
      integer nb_iden_pdg
c
      logical firsttime,firsttime_chans(maxchannels)
      data firsttime /.true./
      data firsttime_chans/maxchannels*.true./
      if (firsttime) then
         do i = 1,lmaxconfigs
            do j = -nexternal,0
               pmass(j,i) = 0d0
               pwidth(j,i) = 0d0
            enddo
         enddo
         firsttime=.false.
      endif
      include "born_props.inc"

      if(.not.IS_A_J(NEXTERNAL))then
        write(*,*)'Fatal error in set_tau_min'
        stop
      endif
c The following assumes that light QCD particles are at the end of the
c list. Exclude one of them to set tau bound at the Born level This
c sets a hard cut in the minimal shat of the Born phase-space
c generation.
c
c The contribution from ptj should be treated only as a 'soft lower
c bound' if j_fks is initial state: the real-emission i_fks parton is
c not necessarily the softest.  Therefore, it could be that even though
c the Born does not have enough energy to pass the cuts set by ptj, the
c event could.
      if (firsttime_chans(ichan)) then
         do i=-nexternal,nexternal
            xm(i)=0d0
            xw(i)=0d0
            mass_min(i)=0d0
         end do
         firsttime_chans(ichan)=.false.
         do iFKS=1,fks_configs
            j_fks=FKS_J_D(iFKS)
            taumin(iFKS,ichan)=0.d0
            taumin_s(iFKS,ichan)=0.d0
            taumin_j(iFKS,ichan)=0.d0
            do i=nincoming+1,nexternal
c Add the minimal jet pTs to tau
               if(IS_A_J(i) .and. i.ne.nexternal)then
                  if  (j_fks.gt.nincoming .and. j_fks.lt.nexternal) then
                     taumin(iFKS,ichan)=taumin(iFKS,ichan)+dsqrt(ptj**2 +emass(i)**2)
                     taumin_s(iFKS,ichan)=taumin_s(iFKS,ichan)+dsqrt(ptj**2 +emass(i)**2)
                     taumin_j(iFKS,ichan)=taumin_j(iFKS,ichan)+dsqrt(ptj**2 +emass(i)**2)
                  elseif (j_fks.ge.1 .and. j_fks.le.nincoming) then
                     taumin(iFKS,ichan)=taumin(iFKS,ichan)+emass(i)
                     taumin_s(iFKS,ichan)=taumin_s(iFKS,ichan)+dsqrt(ptj**2 +emass(i)**2)
                     taumin_j(iFKS,ichan)=taumin_j(iFKS,ichan)+dsqrt(ptj**2 +emass(i)**2)
                  elseif (j_fks.eq.nexternal) then
                     write (*,*)
     &                    'ERROR, j_fks cannot be the final parton'
     &                    ,j_fks
                     stop
                  else
                     write (*,*) 'ERROR, j_fks not correctly defined'
     &                    ,j_fks
                     stop
                  endif
                  xm(i)=emass(i)+ptj
c Add the minimal photon pTs to tau
               elseif(IS_A_PH(i))then
                  if (abs(emass(i)).gt.vtiny) then
                     write (*,*) 'Error in set_tau_min in setcuts.f:'
                     write (*,*) 'mass of a photon should be zero',i
     &                    ,emass(i)
                     stop
                  endif
                  if  (j_fks.gt.nincoming)
     &                 taumin(iFKS,ichan)=taumin(iFKS,ichan)+ptgmin
                  taumin_s(iFKS,ichan)=taumin_s(iFKS,ichan)+ptgmin
                  taumin_j(iFKS,ichan)=taumin_j(iFKS,ichan)+ptgmin
                  xm(i)=emass(i)+ptgmin
               elseif (is_a_lp(i)) then
c Add the postively charged lepton pTs to tau
                  if (j_fks.gt.nincoming) then
                     taumin(iFKS,ichan)=taumin(iFKS,ichan)+dsqrt(ptl**2+emass(i)**2)
                  else
                     taumin(iFKS,ichan)=taumin(iFKS,ichan)+emass(i)
                  endif
                  taumin_s(iFKS,ichan)=taumin_s(iFKS,ichan)+dsqrt(emass(i)**2+ptl**2)
                  taumin_j(iFKS,ichan)=taumin_j(iFKS,ichan)+dsqrt(emass(i)**2+ptl**2)
                  xm(i)=emass(i)+ptl
c Add the lepton invariant mass to tau if there is at least another
c lepton of opposite charge. (Only add half of it, i.e. 'the part
c contributing from this lepton'). Remove possible overcounting with the
c lepton pT
                  do j=nincoming+1,nexternal
                     if (is_a_lm(j) .and. idup(i,1).eq.-idup(j,1) .and.
     $                    (mll_sf.ne.0d0 .or. mll.ne.0d0) ) then
                        if (j_fks.gt.nincoming)
     &                       taumin(iFKS,ichan) = taumin(iFKS,ichan)-dsqrt(ptl**2+emass(i)**2) +
     &                              max(mll/2d0,mll_sf/2d0,dsqrt(ptl**2+emass(i)**2))
                        taumin_s(iFKS,ichan) = taumin_s(iFKS,ichan)-dsqrt(ptl**2+emass(i)**2)
     $                       + max(mll/2d0,mll_sf/2d0,dsqrt(ptl**2+emass(i)**2))
                        taumin_j(iFKS,ichan) = taumin_j(iFKS,ichan)-dsqrt(ptl**2+emass(i)**2)
     $                       + max(mll/2d0,mll_sf/2d0,dsqrt(ptl**2+emass(i)**2))
                        xm(i)=xm(i)-ptl-emass(i)+max(mll/2d0,mll_sf/2d0
     $                       ,ptl+emass(i))
                        exit
                     elseif (is_a_lm(j) .and. mll.ne.0d0) then
                        if (j_fks.gt.nincoming)
     &                       taumin(iFKS,ichan)= taumin(iFKS,ichan)-dsqrt(ptl**2+emass(i)**2) +
     &                                     max(mll/2d0,dsqrt(ptl**2+emass(i)**2))
                        taumin_s(iFKS,ichan) = taumin_s(iFKS,ichan)-dsqrt(ptl**2+emass(i)**2)
     $                       + max(mll/2d0, dsqrt(ptl**2+emass(i)**2))
                        taumin_j(iFKS,ichan) = taumin_j(iFKS,ichan)-dsqrt(ptl**2+emass(i)**2)
     $                       + max(mll/2d0,dsqrt(ptl**2+emass(i)**2))
                        xm(i)=xm(i)-ptl-emass(i)+max(mll/2d0,ptl
     $                       +emass(i))
                        exit
                     endif
                  enddo
               elseif (is_a_lm(i)) then
c Add the negatively charged lepton pTs to tau
                  if (j_fks.gt.nincoming) then
                     taumin(iFKS,ichan)=taumin(iFKS,ichan)+dsqrt(ptl**2+emass(i)**2)
                  else
                     taumin(iFKS,ichan)=taumin(iFKS,ichan)+emass(i)
                  endif
                  taumin_s(iFKS,ichan)=taumin_s(iFKS,ichan)+dsqrt(ptl**2+emass(i)**2)
                  taumin_j(iFKS,ichan)=taumin_j(iFKS,ichan)+dsqrt(ptl**2+emass(i)**2)
                  xm(i)=emass(i)+ptl
c Add the lepton invariant mass to tau if there is at least another
c lepton of opposite charge. (Only add half of it, i.e. 'the part
c contributing from this lepton'). Remove possible overcounting with the
c lepton pT
                  do j=nincoming+1,nexternal
                     if (is_a_lp(j) .and. idup(i,1).eq.-idup(j,1) .and.
     $                    (mll_sf.ne.0d0 .or. mll.ne.0d0) ) then
                        if (j_fks.gt.nincoming)
     &                       taumin(iFKS,ichan) = taumin(iFKS,ichan)-dsqrt(ptl**2+emass(i)**2) +
     &                              max(mll/2d0,mll_sf/2d0,dsqrt(ptl**2+emass(i)**2))
                        taumin_s(iFKS,ichan) = taumin_s(iFKS,ichan)-dsqrt(ptl**2+emass(i)**2)
     $                       + max(mll/2d0,mll_sf/2d0,dsqrt(ptl**2+emass(i)**2))
                        taumin_j(iFKS,ichan) = taumin_j(iFKS,ichan)-dsqrt(ptl**2+emass(i)**2)
     $                       + max(mll/2d0,mll_sf/2d0,dsqrt(ptl**2+emass(i)**2))
                        xm(i)=xm(i)-ptl-emass(i)+max(mll/2d0,mll_sf/2d0
     $                       ,ptl+emass(i))
                        exit
                     elseif (is_a_lp(j) .and. mll.ne.0d0) then
                        if (j_fks.gt.nincoming)
     &                       taumin(iFKS,ichan) = taumin(iFKS,ichan)-dsqrt(ptl**2+emass(i)**2) +
     &                                      max(mll/2d0,dsqrt(ptl**2+emass(i)**2))
                        taumin_s(iFKS,ichan) = taumin_s(iFKS,ichan)-dsqrt(ptl**2+emass(i)**2)
     $                       + max(mll/2d0,dsqrt(ptl**2+emass(i)**2))
                        taumin_j(iFKS,ichan) = taumin_j(iFKS,ichan)-dsqrt(ptl**2+emass(i)**2)
     $                       + max(mll/2d0,dsqrt(ptl**2+emass(i)**2))
                        xm(i)=xm(i)-ptl-emass(i)+max(mll/2d0,ptl
     $                       +emass(i))
                        exit
                     endif
                  enddo
               else
                  if (i.eq.nexternal)then
                        taumin(iFKS,ichan)=taumin(iFKS,ichan) + emass(i)
                        taumin_s(iFKS,ichan)=taumin_s(iFKS,ichan) +  emass(i)
                        taumin_j(iFKS,ichan)=taumin_j(iFKS,ichan) + emass(i)
                        xm(i) = emass(i)
                  else
                     smin_update = 0
                     nb_iden_pdg = 1
                     mxx = 0d0
c                    assume smin apply always on the same set of particle
                     do j=nincoming+1,nexternal-1
                        if (mxxmin(i,j).ne.0d0.or.mxxmin(j,i).ne.0d0) then
                           nb_iden_pdg = nb_iden_pdg +1
                           if (mxx.eq.0d0) mxx = max(mxxmin(i,j), mxxmin(j,i))
                        endif
                     enddo
                     ! S >= (2*N-N^2)*M1^2 + (N^2-N)/2 * Mxx^2
                     smin_update = nb_iden_pdg*((2-nb_iden_pdg)*emass(i)**2 + (nb_iden_pdg-1)/2.*mxx**2)
                     ! compare with the update from pt cut
                     if (smin_update.lt.nb_iden_pdg**2*(etmin(i)**2 + emass(i)**2))then
                        ! the pt is more restrictive
                        smin_update = dsqrt(etmin(i)**2 + emass(i)**2)
                     else
                        smin_update = dsqrt(smin_update)/nb_iden_pdg ! share over N particle, and change dimension
                     endif
                     smin_update = emass(i)
                     ! update in sqrt(s) so take the 
                     if  (j_fks.gt.nincoming) then
                        taumin(iFKS,ichan)=taumin(iFKS,ichan) + smin_update
                     else
                        taumin(iFKS,ichan)=taumin(iFKS,ichan) + emass(i)
                     endif
                     taumin_s(iFKS,ichan)=taumin_s(iFKS,ichan) + smin_update
                     taumin_j(iFKS,ichan)=taumin_j(iFKS,ichan) + smin_update
                     xm(i) = smin_update
                  endif
               endif
               xw(i)=0d0
            enddo
            stot = 4d0*ebeam(1)*ebeam(2)
            tau_Born_lower_bound=taumin(iFKS,ichan)**2/stot
            tau_lower_bound=taumin_j(iFKS,ichan)**2/stot
c         
c Also find the minimum lower bound if all internal s-channel particles
c were on-shell
            tsign=-1
            nt=0
            do i=-1,-(nexternal-3),-1 ! All propagators
               if ( itree(1,i) .eq. 1 .or. itree(1,i) .eq. 2 ) tsign=1
               if (tsign.eq.-1) then ! s-channels
                  d1=itree(1,i)
                  d2=itree(2,i)
c If daughter is a jet, we should treat the ptj as a mass. Except if
c d1=nexternal, because we check the Born, so final parton should be
c skipped. [This is already done above; also for the leptons]
                  xm1=xm(d1)
                  xm2=xm(d2)
                  xw1=xw(d1)
                  xw2=xw(d2)
c On-shell mass of the intermediate resonance
                  xmi=pmass(i,iconf)
c Width of the intermediate resonance
                  xwi=pwidth(i,iconf)
c Set the intermediate mass equal to the max of its actual mass and
c the sum of the masses of the two daugters.
                  if (xmi.gt.xm1+xm2) then
                     xm(i)=xmi
                     xw(i)=xwi
                  else
                     xm(i)=xm1+xm2
                     xw(i)=xw1+xw2 ! just sum the widths
                  endif
c Add the new mass to the bound. To avoid double counting, we should
c subtract the daughters, because they are already included above or in
c the previous iteration of the loop
                  taumin_s(iFKS,ichan)=taumin_s(iFKS,ichan)+xm(i)-xm1-xm2
               else             ! t-channels
                  if (i.eq.-(nexternal-3)) then
                     xm(i)=0d0
                     cycle
                  endif
                  nt=nt+1
                  d1=itree(2,i) ! only use 2nd daughter (which is the outgoing one)
                  xm1=xm(d1)
                  if (nt.gt.1) xm1=max(xm1,xk(nt-1))
                  xk(nt)=xm1
                  j=i-1         ! this is the closest to p2
                  d2=itree(2,j)
                  xm2=xm(d2)
                  xm(i)=min(xm1,xm2)
               endif
            enddo
      
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c Determine the "minimal" s-channel invariant masses
            do i=nincoming+1,nexternal-1
               s_mass_FKS(iFKS,i,ichan)=xm(i)**2
            enddo
            do i=-1,-(nexternal-3),-1 ! All propagators 
               s_mass_FKS(iFKS,i,ichan)=xm(i)**2
            enddo
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c Determine the conflicting Breit-Wigner's. Note that xm(i) contains the
c mass of the BW
            do i=nincoming+1,nexternal-1
               mass_min(i)=xm(i) ! minimal allowed resonance mass (including masses set by cuts)
            enddo
            cBW_FKS_level_max(iFKS,ichan)=0
            t_channel=0
            do i=-1,-(nexternal-3),-1 ! All propagators
               cBW_FKS_mass(iFKS,1,i,ichan)=0d0
               cBW_FKS_width(iFKS,1,i,ichan)=0d0
               cBW_FKS_mass(iFKS,-1,i,ichan)=0d0
               cBW_FKS_width(iFKS,-1,i,ichan)=0d0
               masslow(i)=9d99
               widthlow(i)=0d0
               if ( itree(1,i).eq.1 .or. itree(1,i).eq.2 ) t_channel=i
               if (t_channel.ne.0) exit ! only s-channels
               mass_min(i)=mass_min(itree(1,i))+mass_min(itree(2,i))
               if (xm(i).lt.mass_min(i)-vtiny) then
                  write (*,*)
     $                 'ERROR in the determination of conflicting BW',i
     $                 ,xm(i),mass_min(i)
                  stop
               endif
               if (pmass(i,iconf).lt.xm(i) .and.
     $              pwidth(i,iconf).gt.0d0) then
c     Possible conflict in BW
                  if (pmass(i,iconf).lt.mass_min(i)) then
c     Resonance can never go on-shell due to the kinematics of the event
                     cBW_FKS(iFKS,i,ichan)=2
                     cBW_FKS_level(iFKS,i,ichan)=0
                  elseif(pmass(i,iconf).lt.xm(i)) then
c     Conflicting Breit-Wigner
                     cBW_FKS(iFKS,i,ichan)=1
                     cBW_FKS_level(iFKS,i,ichan)=1
                     cBW_FKS_level_max(iFKS,ichan)=max(cBW_FKS_level_max(iFKS,ichan)
     $                    ,cBW_FKS_level(iFKS,i,ichan))
c     Set here the mass (and width) of the alternative mass; it's the
c     sum of daughter masses. (2nd argument is '1', because this
c     alternative mass is LARGER than the resonance mass).
                     cBW_FKS_mass(iFKS,1,i,ichan)=xm(i)
                     cBW_FKS_width(iFKS,1,i,ichan)=xw(i)
                  endif
c     set the daughters also as conflicting (recursively)
                  masslow(i)=pmass(i,iconf)
                  widthlow(i)=pwidth(i,iconf)
                  do j=i,-1
                     if (cBW_FKS(iFKS,j,ichan).eq.0) cycle
                     do k=1,2   ! loop over the 2 daughters
                        if (itree(k,j).ge.0) cycle
                        if (cBW_FKS(iFKS,itree(k,j),ichan).eq.2) cycle
                        cBW_FKS(iFKS,itree(k,j),ichan)=1
                        cBW_FKS_level(iFKS,itree(k,j),ichan)=
     $                       cBW_FKS_level(iFKS,j,ichan)+1
                        cBW_FKS_level_max(iFKS,ichan)=
     $                       max(cBW_FKS_level_max(iFKS,ichan)
     $                       ,cBW_FKS_level(iFKS,itree(k,j),ichan))
c     Set here the mass (and width) of the alternative mass; it's the
c     difference between the mother and the sister masses. (3rd argument
c     is '-1', because this alternative mass is SMALLER than the
c     resonance mass).
                        masslow(itree(k,j))=min(masslow(itree(k,j)),
     &                       max(masslow(j)-xm(itree(3-k,j)),0d0)) ! mass difference
                        widthlow(itree(k,j))=max(widthlow(itree(k,j)),
     &                       widthlow(j)+xw(itree(3-k,j))) ! sum of widths
                        if (pwidth(itree(k,j),iconf).eq.0d0 .or.
     $                       masslow(itree(k,j)).ge.pmass(itree(k,j)
     $                       ,iconf)) cycle
                        cBW_FKS_mass(iFKS,-1,itree(k,j),ichan)=
     $                       masslow(itree(k,j))
                        cBW_FKS_width(iFKS,-1,itree(k,j),ichan)=
     $                       widthlow(itree(k,j))
                     enddo
                  enddo
               else
c     Normal Breit-Wigner
                  cBW_FKS(iFKS,i,ichan)=0
               endif
            enddo
c loop over t-channel to make sure that s-hat is consistent with sum of
c s-channel masses
            if (t_channel.ne.0) then
               sum_all_s=0d0
               do i=t_channel,-(nexternal-3),-1
c Breit-wigner can never go on-shell:
                  if (itree(2,i).gt.0) cycle
                  if ( pmass(itree(2,i),iconf).gt.sqrt(stot) .and.
     $                 pwidth(itree(2,i),iconf).gt.0d0) then
                     cBW_FKS(iFKS,itree(2,i),ichan)=2
                  endif
c     s-channel is always 2nd argument of itree, sum it to sum_all_s
                  sum_all_s=sum_all_s+xm(itree(2,i))
               enddo
               if (sum_all_s.gt.sqrt(stot)) then
c     conflicting BWs: set all s-channels as conflicting
                  do i=t_channel,-(nexternal-3),-1
                     if (itree(2,i).gt.0) cycle
                     if (cBW_FKS(iFKS,itree(2,i),ichan).ne.2) then
                        cBW_FKS(iFKS,itree(2,i),ichan)=1
                        cBW_FKS_mass(iFKS,-1,itree(2,i),ichan)=sqrt(stot)/2d0
                        cBW_FKS_width(iFKS,-1,itree(2,i),ichan)=xw(itree(2,i))
                     endif
                  enddo
               endif
            endif


c Conflicting BW's determined. They are saved in cBW_FKS
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c If the lower bound found here is smaller than the hard bound,
c simply set the soft bound equal to the hard bound.
            taumin_s(iFKS,ichan)=
     &           max(taumin_j(iFKS,ichan),taumin_s(iFKS,ichan))
c
c For the bound, we have to square and divide by stot.
            tau_lower_bound_resonance=taumin_s(iFKS,ichan)**2/stot
c
            if (j_fks.gt.nincoming) then
               write (*,'(a7,x,i3,x,i5,x,a1,3(e12.5,x)))') 'tau_min'
     $              ,iFKS,ichan,':',taumin(iFKS,ichan),taumin_j(iFKS
     $              ,ichan),taumin_s(iFKS,ichan)
            else
               write (*,'(a7,x,i3,x,i5,x,a1,e12.5,x,a13,e12.5,x))')
     $              'tau_min',iFKS,ichan,':',taumin(iFKS,ichan)
     $              ,'     --      ',taumin_s(iFKS,ichan)
            endif
         enddo
      endif
      tau_Born_lower_bound=taumin(nFKSprocess,ichan)**2/stot
      tau_lower_bound=taumin_j(nFKSprocess,ichan)**2/stot
      tau_lower_bound_resonance=taumin_s(nFKSprocess,ichan)**2/stot
      do i=-nexternal,-1
         cBW(i)=cBW_FKS(nFKSprocess,i,ichan)
         cBW_level(i)=cBW_FKS_level(nFKSprocess,i,ichan)
         do j=-1,1,2
            cBW_mass(j,i)=cBW_FKS_mass(nFKSprocess,j,i,ichan)
            cBW_width(j,i)=cBW_FKS_width(nFKSprocess,j,i,ichan)
         enddo
      enddo
      do i=-nexternal,nexternal
         s_mass(i)=s_mass_FKS(nFKSprocess,i,ichan)
      enddo
      cBW_level_max=cBW_FKS_level_max(nFKSprocess,ichan)
      return
      end


      subroutine sChan_order(ns_channel,order)
      implicit none
      include 'nexternal.inc'
      include 'maxparticles.inc'
      include 'maxconfigs.inc'
      integer itree(2,-max_branch:-1),iconf
      common /to_itree/itree,iconf
      logical new_point
      common /c_new_point/new_point
      double precision ran2,rnd
      integer i,j,order(-nexternal:0),ipos,ns_channel,npos
     $     ,pos(nexternal),ord(-nexternal:0)
      logical done(-nexternal:nexternal)
      external ran2
      save ord
      if (.not. new_point) then
         do j=-1,-ns_channel,-1
            order(j)=ord(j)
         enddo
         return
      endif
      do i=-ns_channel,0
         done(i)=.false.
      enddo
      do i=1,nexternal
         done(i)=.true.
      enddo
      do j=-1,-ns_channel,-1
         npos=0
         do i=-1,-ns_channel,-1
            if((.not. done(i)) .and.
     &           done(itree(1,i))  .and. done(itree(2,i))) then
               npos=npos+1
               pos(npos)=i
            endif
         enddo
         if (npos.gt.1) then
            rnd=ran2()
            ipos=min(int(rnd*npos)+1,npos)
            ord(j)=pos(ipos)
            done(pos(ipos))=.true.
         elseif (npos.eq.1) then
            ord(j)=pos(npos)
            done(pos(npos))=.true.
         else
            write (*,*) 'ERROR in sChan_order',npos
         endif
         order(j)=ord(j)
      enddo
      new_point=.false.
      return
      end
