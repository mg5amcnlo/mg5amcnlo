c NOTE THAT ONLY IRC-SAFE CUTS CAN BE APPLIED OTHERWISE THE INTEGRATION
c MIGHT NOT CONVERGE
c
      logical function dummy_cuts(p,istatus,ipdg)
      implicit none
c This includes the 'nexternal' parameter that labels the number of
c particles in the (n+1)-body process
      include 'nexternal.inc'
c This include file contains common blocks filled with the cuts defined
c     in the run_card.dat (including custom set entry)
      include 'run.inc'
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
C     external functions that can be used. Some are defined in cuts.f
C     others are in ./Source/kin_functions.f
      REAL*8 R2_04,invm2_04,pt_04,eta_04,pt,eta
      external R2_04,invm2_04,pt_04,eta_04,pt,eta

      dummy_cuts = .true.       ! event is okay; otherwise it is changed
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

      double precision function user_dynamical_scale(P)
c     allow to define your own dynamical scale, need to set dynamical_scale_choice to 0 (or 10) to use it
      implicit none
      include 'nexternal.inc'
      double precision P(0:3, nexternal)
c This include file contains common blocks filled with the cuts defined
c     in the run_card.dat (including custom set entry)
      include 'run.inc'
      
      character*80 temp_scale_id
      common/ctemp_scale_id/temp_scale_id

c     default behavior: fixed scale (for retro compatibility)
      user_dynamical_scale = muR_ref_fixed
      temp_scale_id='fixed scale' ! use a meaningful string
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

