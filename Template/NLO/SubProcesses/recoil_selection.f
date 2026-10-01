      subroutine set_fks_recoilers(recoilers,pass)
c Request either one incoming recoiler or a nonempty final-state system.
c Labels are those of the current real process; omit emitter and emission.
c The request overrides FKSFinalRecoil until clear_fks_recoilers is called.
c Set after the process, FKS indices and beam configuration are initialized.
c An invalid request leaves any previously accepted request unchanged.
      implicit none
      include 'nexternal.inc'
      include 'recoil_selection.inc'
      logical recoilers(nexternal),pass
      call validate_fks_recoilers(recoilers,pass)
      if(.not.pass)return
      fks_requested_recoilers=recoilers
      fks_recoil_requested=.true.
      end


      subroutine clear_fks_recoilers()
c Resume the card's recoil policy at the next phase-space selection.
      implicit none
      include 'nexternal.inc'
      include 'recoil_selection.inc'
      fks_recoil_requested=.false.
      fks_requested_recoilers=.false.
      end


      subroutine validate_fks_recoilers(recoilers,pass)
      implicit none
      include 'nexternal.inc'
      include 'run.inc'
      logical recoilers(nexternal),pass
      integer i_fks,j_fks,beam
      common /fks_indices/i_fks,j_fks
      double precision pmass(nexternal)
      common /to_mass/pmass
      pass=.false.
      if(i_fks.le.nincoming.or.i_fks.gt.nexternal.or.
     $     j_fks.le.nincoming.or.j_fks.gt.nexternal.or.
     $     i_fks.eq.j_fks)return
      if(recoilers(i_fks).or.recoilers(j_fks))return
      if(.not.any(recoilers))return
      if(any(recoilers(1:nincoming)))then
         if(nincoming.ne.2.or.count(recoilers).ne.1)return
         do beam=1,nincoming
            if(.not.recoilers(beam))cycle
            if(abs(lpp(beam)).ne.1.and.abs(lpp(beam)).ne.2)return
            if(pmass(beam).ne.0d0)return
         enddo
      endif
      pass=.true.
      end


      subroutine select_fks_recoil(default_members,enable)
      use fks_phase_space_data,only: resonance_momentum,resonance_mass2,resonance_recoil,
     $     resonance_members,initial_recoil_leg
c Install the active radiation frame policy. default_members is the
c automatic resonance subsystem, INCLUDING emitter and emission. Explicit
c requests instead contain recoilers only and survive point generation.
      use FKSParams, only: FKSFinalRecoil
      implicit none
      include 'nexternal.inc'
      include 'recoil_selection.inc'
      logical default_members(nexternal),enable,
     $     recoilers(nexternal),pass,defaults(nexternal)
      logical fixed_order,nlo_ps
      common /c_fnlo_nlops/fixed_order,nlo_ps
      integer i_fks,j_fks,beam
      common /fks_indices/i_fks,j_fks

      defaults=default_members
      resonance_recoil=.false.
      resonance_members=.false.
      resonance_momentum=0d0
      resonance_mass2=0d0
      initial_recoil_leg=0
      if(.not.enable.or.j_fks.le.nincoming)return

      recoilers=.false.
      if(fks_recoil_requested)then
         recoilers=fks_requested_recoilers
      elseif(FKSFinalRecoil.ne.0)then
         if(FKSFinalRecoil.lt.1.or.FKSFinalRecoil.gt.nincoming)then
            write(*,*) 'Invalid FKSFinalRecoil for incoming particles',
     $           FKSFinalRecoil,nincoming
            stop 1
         endif
         recoilers(FKSFinalRecoil)=.true.
      else
         if(.not.any(defaults))return
         recoilers=defaults
         if(i_fks.gt.nincoming.and.i_fks.le.nexternal)
     $        recoilers(i_fks)=.false.
         if(j_fks.gt.nincoming.and.j_fks.le.nexternal)
     $        recoilers(j_fks)=.false.
      endif

      call validate_fks_recoilers(recoilers,pass)
      if(.not.pass)then
         write(*,*) 'Invalid FKS recoiler selection for sector',
     $        i_fks,j_fks
         write(*,*) 'Use one massless incoming hadron parton or a',
     $        ' nonempty final-state system excluding emitter/emission'
         stop 1
      endif
      resonance_recoil=.true.
      if(any(recoilers(1:nincoming)))then
         if(nlo_ps)then
            write(*,*) 'Initial FKS recoil is available at fixed order',
     $           ' only; MC@NLO counterterms require their own map'
            stop 1
         endif
         do beam=1,nincoming
            if(recoilers(beam))initial_recoil_leg=beam
         enddo
      else
         resonance_members=recoilers
         resonance_members(i_fks)=.true.
         resonance_members(j_fks)=.true.
      endif
      end


      block data fks_recoil_selection_init
      implicit none
      include 'nexternal.inc'
      include 'recoil_selection.inc'
      data fks_recoil_requested/.false./
      data fks_requested_recoilers/nexternal*.false./
      end
