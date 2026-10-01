c FKS phase-space entry points and event/counterevent bookkeeping.
c Numerical kernels are grouped by responsibility:
c   genps_fks_born.f    Born trees and invariant-mass sampling
c   genps_fks_beams.f   incoming fractions and rapidities
c   genps_fks_fsr.f     final-state forward and inverse maps
c   genps_fks_isr.f     initial-state forward and inverse maps
c   genps_fks_native.f  projection at a fixed real point
c   genps_fks_helpers.f shared kinematics and radiation boundaries
c Local final/initial recoilers live in resonance_recoil.f and
c initial_recoil.f. Public routines and COMMON layouts stay unchanged.

      subroutine generate_momenta(ndim,iconfig,wgt,xx,p,p_lab,p_cms)
      use kinematics_module
      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      include 'timing_variables.inc'
      integer ndim,iconfig
      double precision wgt,xx(99),p(0:3,nexternal),p_lab(0:3,nexternal)
     $     ,p_cms(0:3,nexternal),dummy
      double precision pmass(-nexternal:0,lmaxconfigs,0:fks_configs)
      double precision pwidth(-nexternal:0,lmaxconfigs,0:fks_configs)
      integer iforest(2,-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer sprop(-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer tprid(-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer mapconfig(0:lmaxconfigs,0:fks_configs)
      common /c_configurations/pmass,pwidth,iforest,sprop,tprid
     $     ,mapconfig
      double precision qmass(-nexternal:0),qwidth(-nexternal:0),jac
      integer i,j
      double precision zero
      parameter (zero=0d0)
      integer itree(2,-max_branch:-1),iconf
      common /to_itree/itree,iconf
      double precision p1_cnt(0:3,nexternal,-2:2)
      double precision wgt_cnt(-2:2)
      double precision pswgt_cnt(-2:2)
      double precision jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
      integer iconfig0
      common/ciconfig0/iconfig0
      double precision qmass_common(-nexternal:0),qwidth_common(
     &     -nexternal:0)
      common /c_qmass_qwidth/qmass_common,qwidth_common
      double precision xvar(99)
      common /c_vegas_x/xvar
      integer            this_config
      common/to_mconfigs/this_config
      double precision tau_cnt(-2:2),ycm_cnt(-2:2)
      common/cbjrk12_cnt/tau_cnt,ycm_cnt
c
      call cpu_time(tBefore)
      this_config=iconfig
      iconf=iconfig
      iconfig0=iconfig
      do i=-max_branch,-1
         do j=1,2
            itree(j,i)=iforest(j,i,iconfig,0)
         enddo
      enddo

      do i=-nexternal,0
         qmass(i)=pmass(i,iconfig,0)
         qwidth(i)=pwidth(i,iconfig,0)
         qmass_common(i)=qmass(i)
         qwidth_common(i)=qwidth(i)
      enddo
      do i=1,ndim
         xvar(i)=xx(i)
      enddo
c

      call generate_momenta_conf_wrapper(ndim,jac,xx,itree,qmass,qwidth,p)
c If the input weight 'wgt' to this subroutine was not equal to one,
c make sure we update all the (counter-event) jacobians and return also
c the updated wgt (i.e. the jacobian for the event)
      do i=-2,2
         jac_cnt(i)=jac_cnt(i)*wgt
      enddo
      wgt=wgt*jac
c
      call cpu_time(tAfter)
      tGenPS=tGenPS+(tAfter-tBefore)

      if (p(0,1).gt.0d0) then ! valid real-emission momenta
         call boost_n1_to_its_cms(p,p_cms,dummy)
         call boost_n1_to_lab(p,p_lab,-ycm_cnt(0))
      endif
      
      return
      end


      subroutine generate_momenta_conf_wrapper(nndim,jac,x,itree,qmass
     $     ,qwidth,p)
c Select the recoil policy after the Born channel has identified its
c resonance descendants. The explicit API and card may override it.
      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      include 'resonance_recoil.inc'
      integer nndim,itree(2,-max_branch:-1),nFKSprocess
      double precision jac,x(99),p(0:3,nexternal),
     $     qmass(-nexternal:0),qwidth(-nexternal:0)
      logical granny_is_res,granny_chain(-nexternal:nexternal),
     $     granny_chain_real_final(-nexternal:nexternal)
      integer igranny,iaunt
      common /c_granny_res/igranny,iaunt,granny_is_res,granny_chain,
     $     granny_chain_real_final
      common /c_nFKSprocess/nFKSprocess
      logical write_granny(fks_configs)
      integer which_is_granny(fks_configs)
      common /write_granny_resonance/which_is_granny,write_granny
      character*4 abrv
      common /to_abrv/abrv

      resonance_recoil=.false.
      resonance_members=.false.
      resonance_momentum=0d0
      resonance_mass2=0d0
      initial_recoil_leg=0
      call set_tau_min()
      if(granny_is_res)
     $     resonance_members=granny_chain_real_final(1:nexternal)
      call select_fks_recoil(resonance_members,abrv.ne.'born')
      write_granny(nFKSprocess)=.true.
      which_is_granny(nFKSprocess)=igranny
      call generate_momenta_conf(nndim,jac,x,itree,qmass,qwidth,p)
      end


      subroutine generate_momenta_conf(ndim,jac,x,itree,qmass,qwidth,p)
c
c x(1)...x(ndim-5) --> invariant mass & angles for the Born
c x(ndim-4) --> tau_born
c x(ndim-3) --> y_born
c x(ndim-2) --> xi_i_fks
c x(ndim-1) --> y_ij_fks
c x(ndim) --> phi_i
c
      use mc_native_context, only: native_epoch
      implicit none
      integer,save::epoch_save=-1
      include 'genps.inc'
      include 'nexternal.inc'
      include 'run.inc'
c arguments
      integer ndim
      double precision jac,x(99),p(0:3,nexternal)
      integer itree(2,-max_branch:-1)
      double precision qmass(-nexternal:0),qwidth(-nexternal:0)
c common
c     Arguments have the following meanings:
c     -2 soft-collinear, incoming leg, - direction as in FKS paper
c     -1 collinear, incoming leg, - direction as in FKS paper
c     0 soft
c     1 collinear
c     2 soft-collinear
      double precision p1_cnt(0:3,nexternal,-2:2)
      double precision wgt_cnt(-2:2)
      double precision pswgt_cnt(-2:2)
      double precision jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      logical nocntevents
      common/cnocntevents/nocntevents
      integer iconfig0,iconfigsave
      common/ciconfig0/iconfig0
      save iconfigsave
c Masses of particles. Should be filled in setcuts.f
      double precision pmass(nexternal)
      common /to_mass/pmass
c local
      integer i,j,nbranch,ns_channel,nt_channel,ionebody
     &     ,isolsign
      double precision M(-max_branch:max_particles),totmassin,totmass
     &     ,stot,xjac0,S(-max_branch:max_particles)
     &     ,tau_born,ycm_born,ycmhat,fksmass,xbjrk_born(2),shat_born
     &     ,sqrtshat_born,xpswgt0,m_born(nexternal-1)
      logical one_body,pass
      logical use_evpr
      common /to_use_evpr/use_evpr
c external
      double precision lambda
      external lambda
c parameters
      logical fks_as_is
      parameter (fks_as_is=.false.)
      real*8 pi
      parameter (pi=3.1415926535897932d0)
      logical firsttime
      data firsttime/.true./
      double precision zero
      parameter (zero=0d0)
c saves
      save m,stot,totmassin,totmass
     &     ,ionebody,fksmass,nbranch
      common /c_isolsign/isolsign

      pass=.true.
      do i=1,nexternal-1
         if (i.lt.i_fks) then
            m(i)=pmass(i)
         else
            m(i)=pmass(i+1)
         endif
      enddo
      if(firsttime.or.iconfig0.ne.iconfigsave.or.
     $     epoch_save.ne.native_epoch)then
         if (nincoming.eq.2) then
            stot = 4d0*ebeam(1)*ebeam(2)
         else
            stot=pmass(1)**2
         endif
c Make sure have enough mass for external particles
         totmassin=0d0
         do i=1,nincoming
            totmassin=totmassin+m(i)
         enddo
         totmass=0d0
         do i=nincoming+1,nexternal-1
            totmass=totmass+m(i)
         enddo
         fksmass=totmass
         if (stot .lt. max(totmass,totmassin)**2) then
            write (*,*) 'Fatal error #0 in one_tree:'/
     &           /'insufficient collider energy'
            stop
         endif

         firsttime=.false.
         iconfigsave=iconfig0
         epoch_save=native_epoch
      endif                     ! firsttime
      call fill_genmom_born_commons(itree,m)
c
      xjac0=1d0
      xpswgt0=1d0

      ! generate tau and y
      call generate_tau_y_wrapper(
     $ qmass,qwidth,totmass,stot,x(ndim-4:ndim-3),tau_born,ycm_born,ycmhat,xjac0)
      ! filter unphysical configurations
      if (xjac0.lt.0d0) goto 222

c Compute Bjorken x's from tau and y
      xbjrk_born(1)=sqrt(tau_born)*exp(ycm_born)
      xbjrk_born(2)=sqrt(tau_born)*exp(-ycm_born)
c Compute shat and sqrt(shat)
      if(.not.one_body)then
        shat_born=tau_born*stot
        sqrtshat_born=sqrt(shat_born)
      else
c Trivial, but prevents loss of accuracy
        shat_born=totmass**2
        sqrtshat_born=totmass
      endif

      !!!!! for dressed-lepton collisions only !!!!
      ! if j_fks is initial state, then use the mapping without
      ! event-projection if use_evpr is set to false
      !(note that in e+e- collisions, if tau is generated with a BW
      ! then use_evpr is set to true)
      if ((abs(lpp(1)).eq.1.and.abs(lpp(2)).eq.1).or.
     $   (abs(lpp(1)).eq.2.and.abs(lpp(2)).eq.2).or.
     $   (lpp(1).eq.0.and.lpp(2).eq.0)) then
          use_evpr = .true.
      else if ((abs(lpp(1)).eq.3.and.abs(lpp(2)).eq.3).or.
     $         (abs(lpp(1)).eq.4.and.abs(lpp(2)).eq.4)) then
          use_evpr = use_evpr.or.j_fks.gt.nincoming 
      endif

      if (use_evpr) then
        ! standard mapping with event-projection
        call generate_momenta_born(x,shat_born,sqrtshat_born,totmass,
     $      m,s,
     $        qmass,qwidth,m_born,xpswgt0,xjac0)
        call generate_FKS_kinematics(x,ndim,xjac0,xpswgt0,
     $      stot,shat_born,sqrtshat_born,tau_born,ycm_born,ycmhat,
     $      xbjrk_born,m,m_born,jac,p,pass)
      else
        ! new mapping without event-projection, suitable for e+e-
        ! collisions with ISR(+beamstrahlung)
        call generate_noevpr_kinematics(x,ndim,xjac0,xpswgt0,
     $      stot,shat_born,sqrtshat_born,tau_born,ycm_born,ycmhat,totmass,
     $      xbjrk_born,m,s,qmass,qwidth,m_born,jac,p,pass)
      endif

      !MZ check that adding .or.xjac0<0 does not screw things up
      if(.not.pass.or.xjac0.lt.0d0)goto 222
      return

 222  continue
c
c Born momenta have not been generated. Neither events nor counterevents exist.
c Set all to negative values and exit
      jac=-222
      jac_cnt(0)=-222
      jac_cnt(1)=-222
      jac_cnt(2)=-222
      p(0,1)=-99
      do i=-2,2
        p1_cnt(0,1,i)=-99
      enddo
      p_born(0,1)=-99
      nocntevents=.true.

      return
      end


      subroutine reset_fks_kinematics()
c Invalidate radiation and counterevent state before either generation
c path. Unused entries must never retain a previous sector's values.
      implicit none
      include 'nexternal.inc'
      double precision xi_i_fks_ev,y_ij_fks_ev,
     $     p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,
     $     p_i_fks_ev,p_i_fks_cnt
      double precision xiimax_ev,xiimax_cnt(-2:2)
      common/cxiimaxev/xiimax_ev
      common/cxiimaxcnt/xiimax_cnt
      double precision p1_cnt(0:3,nexternal,-2:2),wgt_cnt(-2:2),
     $     pswgt_cnt(-2:2),jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     $     sqrtshat,shat
      double complex xij_aor
      common/cxij_aor/xij_aor
      integer i

      p_i_fks_ev(0)=-1.d0
      xiimax_ev=-1.d0
      do i=-2,2
         p_i_fks_cnt(0,i)=-1.d0
         xiimax_cnt(i)=-1.d0
         jac_cnt(i)=-1.d0
      enddo
      ybst_til_tolab=1.d14
      ybst_til_tocm=1.d14
      sqrtshat=0.d0
      shat=0.d0
      xij_aor=(0.d0,0.d0)
      end


      subroutine generate_FKS_kinematics(x,ndim,xjac0,xpswgt0,
     $  stot,shat_born,sqrtshat_born,tau_born,ycm_born,ycmhat,
     $  xbjrk_born,m,m_born,jac,p,pass)
      implicit none

      include 'genps.inc'
      include 'nexternal.inc'
      include 'resonance_recoil.inc'

      double precision xjac0,xpswgt0,x(99),p(0:3,nexternal),
     $   stot,shat_born,sqrtshat_born,tau_born,ycm_born,ycmhat,jac
      double precision xbjrk_born(2)
      double precision M(-max_branch:max_particles),m_born(nexternal-1)
      integer ndim
      logical pass

      integer icountevts
      integer ixEi,ixyij,ixpi,imother
      double precision xmrec2,m_j_fks,phi_i_fks,tau,
     $   xi_i_fks,y_ij_fks,xi_i_hat,xiimax,xinorm,xjac,xpswgt,
     $   ycm,xp(0:3,nexternal),xbjrk(2),p_i_fks(0:3),beam_ratio
      integer i,j

      real*8 pi
      parameter (pi=3.1415926535897932d0)

      double precision pmass(nexternal)
      common /to_mass/pmass

      double precision p1_cnt(0:3,nexternal,-2:2)
      double precision wgt_cnt(-2:2)
      double precision pswgt_cnt(-2:2)
      double precision jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born
      double precision p_born_l(0:3,nexternal-1)
      common/pborn_l/p_born_l
      double precision p_born_ev(0:3,nexternal-1)
      common/pborn_ev/p_born_ev

      logical nocntevents
      common/cnocntevents/nocntevents

      logical nbody
      common/cnbody/nbody

      double precision xi_i_hat_ev,xi_i_hat_cnt(-2:2)
      common /cxi_i_hat/xi_i_hat_ev,xi_i_hat_cnt

      double complex xij_aor
      common/cxij_aor/xij_aor

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     &                        sqrtshat,shat

      double precision xi_i_fks_ev,y_ij_fks_ev
      double precision p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt

      integer isolsign
      common /c_isolsign/isolsign

      double precision xiimax_ev
      common /cxiimaxev/xiimax_ev
      double precision xiimax_cnt(-2:2)
      common /cxiimaxcnt/xiimax_cnt

      logical fks_as_is
      parameter (fks_as_is=.false.)

      ! check that the starting PS point is meaningful
      if (xjac0.lt.0d0) then
         pass = .false.
         return
      endif
c
c Here we start with the FKS Stuff
c
c icountevts=-100 is the event, -2 to 2 the counterevents
      icountevts = -100
      call reset_fks_kinematics()
c
c These will correspond to the vegas x's for the FKS variables xi_i,
c y_ij and phi_i (changing this also requires changing folding parameters)
      ixEi=ndim-2
      ixyij=ndim-1
      ixpi=ndim
c
      imother=min(j_fks,i_fks)
      m_j_fks=pmass(j_fks)
c
c For final state j_fks, compute the recoil invariant mass
      if (j_fks.gt.nincoming.and..not.resonance_recoil) then
         call get_recoil(p_born_l,imother,shat_born,xmrec2,pass)
         if (.not.pass) then
            xjac0=-44
            return
         endif
      endif

c Here is the beginning of the loop over the momenta for the event and
c counter-events. This will fill the xp momenta with the event and
c counter-event momenta.
 111  continue
      xjac   = xjac0
      xpswgt = xpswgt0
c
c Put the Born momenta in the xp momenta, making sure that the mapping
c is correct; put i_fks momenta equal to zero.
      do i=1,nexternal
         if(i.lt.i_fks) then
            do j=0,3
               xp(j,i)=p_born_l(j,i)
            enddo
            m(i)=m_born(i)
         elseif(i.eq.i_fks) then
            do j=0,3
               xp(j,i)=0d0
            enddo
            m(i)=0d0
         elseif(i.ge.i_fks) then
            do j=0,3
               xp(j,i)=p_born_l(j,i-1)
            enddo
            m(i)=m_born(i-1)
         endif
      enddo

c
c set-up phi_i_fks
c
      phi_i_fks=2d0*pi*x(ixpi)
      xjac=xjac*2d0*pi
c To keep track of the special phase-space region with massive j_fks
      isolsign=0
c
c consider the three cases:
c case 1: j_fks is massless final state
c case 2: j_fks is massive final state
c case 3: j_fks is initial state
      if (j_fks.gt.nincoming) then
         shat=shat_born
         sqrtshat=sqrtshat_born
         tau=tau_born
         ycm=ycm_born
         xbjrk(1)=xbjrk_born(1)
         xbjrk(2)=xbjrk_born(2)
         if (initial_recoil_leg.gt.0) then
            call generate_momenta_initial_recoil(icountevts,
     $           isolsign,i_fks,j_fks,m_j_fks,initial_recoil_leg,
     $           xbjrk_born(initial_recoil_leg),x(ixEi),phi_i_fks,
     $           xp,xiimax,xinorm,xi_i_fks,y_ij_fks,xi_i_hat,
     $           p_i_fks,xjac,xpswgt,resonance_momentum,
     $           resonance_mass2,beam_ratio,pass)
            if (.not.pass) goto 112
c The radiation routines leave all momenta in the Born CM. Only the
c selected physical incoming momentum (and its PDF argument) changes.
            xbjrk(initial_recoil_leg)=
     $           xbjrk_born(initial_recoil_leg)*beam_ratio
            tau=tau_born*beam_ratio
            shat=shat_born*beam_ratio
            sqrtshat=sqrt(shat)
            ycm=ycm_born+sign(0.5d0,1.5d0-initial_recoil_leg)
     $           *log(beam_ratio)
         elseif (resonance_recoil) then
            call generate_momenta_resonance_final(icountevts,
     $           isolsign,i_fks,j_fks,m_j_fks,resonance_members,
     $           x(ixEi),phi_i_fks,xp,xiimax,xinorm,xi_i_fks,
     $           y_ij_fks,xi_i_hat,p_i_fks,xjac,xpswgt,
     $           resonance_momentum,resonance_mass2,pass)
            if (.not.pass) goto 112
         elseif (m_j_fks.eq.0d0) then
            isolsign=1
            call generate_momenta_massless_final(icountevts,i_fks,j_fks
     &           ,p_born_l(0,imother),shat,sqrtshat,x(ixEi),xmrec2,xp
     &           ,phi_i_fks,xiimax,xinorm,xi_i_fks,y_ij_fks,xi_i_hat
     &           ,p_i_fks,xjac,xpswgt,pass)
            if (.not.pass) goto 112
         elseif(m_j_fks.gt.0d0) then
            call generate_momenta_massive_final(icountevts,isolsign
     &           ,i_fks,j_fks,p_born_l(0,imother)
     &           ,shat,sqrtshat,m_j_fks,x(ixEi),xmrec2,xp,phi_i_fks
     &           ,xiimax,xinorm,xi_i_fks,y_ij_fks,xi_i_hat,p_i_fks,xjac
     &           ,xpswgt,pass)
            if (.not.pass) goto 112
         endif
      elseif(j_fks.le.nincoming) then
         isolsign=1
         call generate_momenta_initial(icountevts,i_fks,j_fks,xbjrk_born
     &        ,tau_born,ycm_born,ycmhat,shat_born,phi_i_fks,xp,x(ixEi)
     &        ,shat,stot,sqrtshat,tau,ycm,xbjrk,p_i_fks,xiimax,xinorm
     &        ,xi_i_fks,y_ij_fks,xi_i_hat,xpswgt,xjac ,pass)
         if (.not.pass) goto 112
      else
         write (*,*) 'Error #2 in genps_fks.f',j_fks
         stop
      endif
c At this point, the phase space lacks a factor xi_i_fks, which need be 
c excluded in an NLO computation according to FKS, being taken into 
c account elsewhere
c
c All done, so check four-momentum conservation
      if(xjac.gt.0.d0)then
         call phspncheck_nocms(nexternal,sqrtshat,m,xp,pass)
         if (.not.pass) then
            xjac=-199
            goto 112
         endif
      endif

c All real channels use the same reference projection for their
c partition of unity, even when their subtraction maps differ.
c The selected FI beam is independent of the Born diagram, so its
c projection already supplies a common point. A global FF projection
c does not exist for a one-particle Born final state.
      if(resonance_recoil.and.initial_recoil_leg.eq.0.and.
     $     icountevts.eq.-100.and.xjac.gt.0d0)then
         if(xi_i_fks.eq.0d0.or.
     $        (m_j_fks.eq.0d0.and.y_ij_fks.eq.1d0))then
            p_born_ev=p_born_l
         else
            call project_global_fsr_partition(xp,i_fks,j_fks,m_j_fks,
     $           p_born_ev,pass)
         endif
         if(.not.pass)then
            xjac=-147d0
            goto 112
         endif
      endif
      call compute_flux(shat,sqrtshat,m(1),m(2),xpswgt,xjac)
c      
 112  continue
      call fill_FKS_commons(icountevts,tau,ycm,ycm_born,shat,sqrtshat,xbjrk,
     $      xiimax,xinorm,xi_i_fks,xi_i_hat,p_i_fks,y_ij_fks,xp,p,xjac,jac,m_j_fks,i_fks,j_fks)

c
      if(icountevts.eq.-100)then
         if( (j_fks.eq.1.or.j_fks.eq.2).and.fks_as_is )then
            icountevts=-2
         else
            icountevts=0
         endif
c skips counterevents when integrating over second fold for massive
c j_fks
         if( isolsign.eq.-1 )icountevts=5
      else
         icountevts=icountevts+1
      endif
      if( (icountevts.le.2.and.m_j_fks.eq.0.d0.and.(.not.nbody)).or.
     &    (icountevts.eq.0.and.m_j_fks.eq.0.d0.and.nbody) .or.
     &    (icountevts.eq.0.and.m_j_fks.ne.0.d0) )then
         goto 111 ! back to the top of the loop

      elseif(icountevts.eq.5) then
c icountevts=5 only when integrating over the second fold with j_fks
c massive. The counterevents have been skipped, so make sure their
c momenta are unphysical. Born are physical if event was generated, and
c must stay so for the computation of enhancement factors.
         do i=0,2
            jac_cnt(i)=-299
            p1_cnt(0,1,i)=-99
         enddo
      endif
      nocntevents=(jac_cnt(0).le.0.d0) .and.
     &            (jac_cnt(1).le.0.d0) .and.
     &            (jac_cnt(2).le.0.d0)
      call xmom_compare(i_fks,j_fks,jac,jac_cnt,p,p1_cnt,pass)
c
      return
      end


      subroutine generate_noevpr_kinematics(x,ndim,xjac0,xpswgt0,
     $  stot,shat_born,sqrtshat_born,tau_born,ycm_born,ycmhat,totmass,
     $  xbjrk_born,m,s,qmass,qwidth,m_born,jac,p,pass)
      ! generate the kinematics without event projection.
      ! In this case, the bjorken x's are kept the same for all contributions
      ! (event and coutnerevents).
      ! First one generates the radiation (I_fks), then the reduced Born 
      ! system with a com energy sborn=(1-xi)*shat
      implicit none

      include 'genps.inc'
      include 'nexternal.inc'

      double precision xjac0,xpswgt0,x(99),p(0:3,nexternal),
     $   stot,shat_born,sqrtshat_born,tau_born,ycm_born,ycmhat,totmass,jac
      double precision xbjrk_born(2)
      double precision M(-max_branch:max_particles),S(-max_branch:max_particles),
     $   m_born(nexternal-1)
      integer ndim
      logical pass
      double precision qmass(-nexternal:0),qwidth(-nexternal:0)

      integer icountevts
      integer ixEi,ixyij,ixpi,imother
      double precision xmrec2,m_j_fks,phi_i_fks,tau,
     $   xi_i_fks,y_ij_fks,xi_i_hat,xiimax,xinorm,xjac,xpswgt,
     $   ycm,xp(0:3,nexternal),xbjrk(2),p_i_fks(0:3)
      integer i,j

      real*8 pi
      parameter (pi=3.1415926535897932d0)

      double precision pmass(nexternal)
      common /to_mass/pmass

      double precision p1_cnt(0:3,nexternal,-2:2)
      double precision wgt_cnt(-2:2)
      double precision pswgt_cnt(-2:2)
      double precision jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born
      double precision p_born_l(0:3,nexternal-1)
      common/pborn_l/p_born_l
      double precision p_born_ev(0:3,nexternal-1)
      common/pborn_ev/p_born_ev
      double precision p_born_coll(0:3,nexternal-1)
      common/pborn_coll/p_born_coll
      double precision p_born_norad(0:3,nexternal-1)
      common/pborn_norad/p_born_norad

      logical nocntevents
      common/cnocntevents/nocntevents

      logical nbody
      common/cnbody/nbody

      double precision xi_i_hat_ev,xi_i_hat_cnt(-2:2)
      common /cxi_i_hat/xi_i_hat_ev,xi_i_hat_cnt

      double complex xij_aor
      common/cxij_aor/xij_aor

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     &                        sqrtshat,shat

      double precision xi_i_fks_ev,y_ij_fks_ev
      double precision p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt

      integer isolsign
      common /c_isolsign/isolsign

      double precision xiimax_ev
      common /cxiimaxev/xiimax_ev
      double precision xiimax_cnt(-2:2)
      common /cxiimaxcnt/xiimax_cnt

      integer skip
      double precision srec
      double precision pb(0:3,-max_branch:nexternal-1)

      logical fks_as_is
      parameter (fks_as_is=.false.)

c Generate the momenta for the initial state of the Born system
      if(nincoming.eq.2) then
        call mom2cx(sqrtshat_born,m(1),m(2),1d0,0d0,pb(0,1),pb(0,2))
      else
         pb(0,1)=sqrtshat_born
         do i=1,2
            pb(i,1)=0d0
         enddo
         p(3,1)=1e-14           ! For HELAS routine ixxxxx for neg. mass
      endif

c
c Here we start with the FKS Stuff
c
c icountevts=-100 is the event, -2 to 2 the counterevents
      icountevts = -100
      call reset_fks_kinematics()

      ! if we do not do event projection, we first generate y/xi FKS
      ! and p_i_fks, then the other momenta
      isolsign=1
c
c These will correspond to the vegas x's for the FKS variables xi_i,
c y_ij and phi_i (changing this also requires changing folding parameters)
      ixEi=ndim-2
      ixyij=ndim-1
      ixpi=ndim

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCc
c Here is the beginning of the loop over the momenta for the event and
c counter-events. This will fill the xp momenta with the event and
c counter-event momenta.
 111  continue
      xjac   = xjac0
      xpswgt = xpswgt0
c
c set-up phi_i_fks
c
      phi_i_fks=2d0*pi*x(ixpi)
      xjac=xjac*2d0*pi

      call generate_momenta_initial_noevpr(icountevts,i_fks,j_fks,xbjrk_born
     &        ,tau_born,ycm_born,ycmhat,shat_born,phi_i_fks,xp,x(ixEi)
     &        ,shat,stot,sqrtshat,tau,ycm,xbjrk,p_i_fks,xiimax,xinorm
     &        ,xi_i_fks,y_ij_fks,xi_i_hat,xpswgt,xjac,srec,pass)

      ! here we should call generate_momenta_born
      call generate_momenta_born(x,srec,dsqrt(srec),totmass,
     $      m,s,
     $      qmass,qwidth,m_born,xpswgt,xjac)

      ! if anything goes wrong with the generation of this 
      ! specific icountevts configuration, just set the corresponding 
      ! Born momenta to -100 so that they will be filtered out
      ! by setcuts. Do not return (this allows e.g. configurations
      ! to have the Born/soft counterevents but not the real-emission)
      if (.not.pass.or.xjac.lt.0d0) then
          p_born(0,1) = -100d0
          p_born_l(0,1) = -100d0
          p_born_ev(0,1) = -100d0
          goto 112
      endif

C If we are not doing event projection, we need to boost the 
C   born momenta in the partonic com frame 
      call boost_born_momenta_noevpr(p_born_l,xp,xi_i_fks,
     &                      i_fks,shat,srec)

C In the case of the event, store the born momenta without the radiation
C It will be employed to compute the multi-channel enhancement factor
      if (icountevts.eq.-100) then
         do i=1,nexternal-1
           p_born_norad(0:3,i) = p_born_l(0:3,i)
         enddo
      endif

C in the collinear limit, the momenta entering the collinear
C  CT for initial-state splittings are different wrt the Born ones
      !write(*,*) 'XP BEFORE PBORN_COLL', icountevts
      !do i =1, nexternal
      !  write(*,*) xp(:,i)
      !enddo
      if (icountevts.eq.1.and.j_fks.le.nincoming) then
        skip = 0
        do i = 1, nexternal
          if (i.eq.i_fks) then
            skip = skip + 1
            p_born_coll(0:3,j_fks) = p_born_coll(0:3,j_fks) - xp(0:3,i)
            !!write(*,*) 'JJ', j_fks, skip, p_born_coll(:, j_fks)
            cycle
          endif
          p_born_coll(0:3,i-skip) = xp(0:3,i)
          !!write(*,*) 'II', i, skip, p_born_coll(:, i-skip)
        enddo
      elseif (icountevts.eq.1) then
        do i = 1, nexternal
          p_born_coll(0:3,i) = p_born(0:3,i)
        enddo
      endif
c
c  Assign the masses; put i_fks mass equal to zero.
      do i=1,nexternal
         if(i.lt.i_fks) then
            m(i)=m_born(i)
         elseif(i.eq.i_fks) then
            m(i)=0d0
         elseif(i.ge.i_fks) then
            m(i)=m_born(i-1)
         endif
      enddo

c All done, so check four-momentum conservation

      if(xjac.gt.0.d0)then
         call phspncheck_nocms(nexternal,sqrtshat,m,xp,pass)
         if (.not.pass) then
            xjac=-199
            goto 112
         endif
      endif

      call compute_flux(shat,sqrtshat,m(1),m(2),xpswgt,xjac)
c      
 112  continue

      call fill_FKS_commons(icountevts,tau,ycm,ycm_born,shat,sqrtshat,xbjrk,
     $      xiimax,xinorm,xi_i_fks,xi_i_hat,p_i_fks,y_ij_fks,xp,p,xjac,jac,m_j_fks,i_fks,j_fks)
c
      if(icountevts.eq.-100)then
         if( (j_fks.eq.1.or.j_fks.eq.2).and.fks_as_is )then
            icountevts=-2
         else
            icountevts=0
         endif
c skips counterevents when integrating over second fold for massive
c j_fks
         if( isolsign.eq.-1 )icountevts=5
      else
         icountevts=icountevts+1
      endif
      if( (icountevts.le.2.and.m_j_fks.eq.0.d0.and.(.not.nbody)).or.
     &    (icountevts.eq.0.and.m_j_fks.eq.0.d0.and.nbody) .or.
     &    (icountevts.eq.0.and.m_j_fks.ne.0.d0) )then
         goto 111 ! back to the top of the loop

      elseif(icountevts.eq.5) then
c icountevts=5 only when integrating over the second fold with j_fks
c massive. The counterevents have been skipped, so make sure their
c momenta are unphysical. Born are physical if event was generated, and
c must stay so for the computation of enhancement factors.
         do i=0,2
            jac_cnt(i)=-299
            p1_cnt(0,1,i)=-99
         enddo
      endif

      nocntevents=(jac_cnt(0).le.0.d0) .and.
     &            (jac_cnt(1).le.0.d0) .and.
     &            (jac_cnt(2).le.0.d0)
CC      call xmom_compare(i_fks,j_fks,jac,jac_cnt,p,p1_cnt,pass)
c
      return
      end


      subroutine boost_born_momenta_noevpr(pborn,xp,xi_i_fks,i_fks,shat,srec)
      implicit none
      include 'nexternal.inc'
      include 'genps.inc'
      double precision pborn(0:3,nexternal-1),xp(0:3,nexternal)
      double precision xi_i_fks, shat, srec
      integer i_fks
      double precision chy_tbst, shy_tbst, chy_tbstmo, xdir_t(3)

      integer i, skip
      double precision p_i_fks(0:3), p_i_fks_red(0:3)

      p_i_fks(0:3) = xp(0:3,i_fks)
      p_i_fks_red(0:3) = p_i_fks(0:3) / (dsqrt(shat)/2d0*xi_i_fks)

      ! pborn are in the recoil center of frame. Must be boosted in the
      ! partonic center of frame, where p_rec+p_i_fks = (sqrtshat,0,0,0)
      ! Note that p_rec^2 = srec == (1-xi)*shat
      ! In the partonic com frame one must have 
      ! P_rec = sqrtshat/2 * ( 2-xi, -p_i_fks_red(1:3) * xi )

      xdir_t(1:3) = p_i_fks_red(1:3)
      chy_tbst = (1-xi_i_fks/2d0)/dsqrt(1-xi_i_fks)
      chy_tbstmo = (1-xi_i_fks/2d0)/dsqrt(1-xi_i_fks)-1d0
      shy_tbst = (xi_i_fks/2d0)/dsqrt(1-xi_i_fks)

      pborn(0,1) = sqrt(shat)/2d0
      pborn(1,1) = 0d0
      pborn(2,1) = 0d0
      pborn(3,1) = sqrt(shat)/2d0

      pborn(0,2) = sqrt(shat)/2d0
      pborn(1,2) = 0d0
      pborn(2,2) = 0d0
      pborn(3,2) =-sqrt(shat)/2d0

c Boost the momenta
      skip=0
      do i=1,nexternal-1
        if (i.le.nincoming) then
          xp(0:3,i)=pborn(0:3,i) 
        else
          if (i.eq.i_fks) skip = skip+1
          !if(i.ne.i_fks.and.shy_tbst.ne.0.d0)
          if (shy_tbst.ne.0.d0) then
            call boostwdir2(chy_tbst,shy_tbst,chy_tbstmo,xdir_t,
     &                        pborn(0,i),xp(0,i+skip)) 
          else
            xp(0:3,i+skip)=pborn(0:3,i)
          endif
        endif
      enddo
        
      return
      end


      subroutine compute_flux(shat,sqrtshat,m1,m2,xpswgt,xjac)
      implicit none
      include 'nexternal.inc'
      double precision shat,sqrtshat,m1,m2,xpswgt,xjac
      double precision flux
      double precision lambda
      external lambda
      real*8 pi
      parameter (pi=3.1415926535897932d0)

      if(nincoming.eq.2)then
         flux  = 1d0 /(2.D0*SQRT(LAMBDA(shat,m1**2,m2**2)))
      else                      ! Decays
         flux = 1d0/(2d0*sqrtshat)
      endif
c The pi-dependent factor inserted below is due to the fact that the
c weight computed above is relevant to R_n, as defined in Kajantie's
c book, eq.(III.3.1), while we need the full n-body phase space
      flux  = flux / (2d0*pi)**(3 * (nexternal-nincoming) - 4)
c This extra pi-dependent factor is due to the fact that the phase-space
c part relevant to i_fks and j_fks does contain all the pi's needed for 
c the correct normalization of the phase space
      flux  = flux * (2d0*pi)**3
c
      xjac=xjac*xpswgt*flux
c
      return
      end


      subroutine fill_FKS_commons(icountevts,tau,ycm,ycm_born,shat,sqrtshat,xbjrk,
     $      xiimax,xinorm,xi_i_fks,xi_i_hat,p_i_fks,y_ij_fks,xp,p,xjac,jac,m_j_fks,i_fks,j_fks)
      use kinematics_module
      implicit none
      integer icountevts
      include 'nexternal.inc'
      double precision tau,ycm,ycm_born,shat,sqrtshat,xbjrk(2),xiimax,xinorm,
     $ xi_i_fks,xi_i_hat,p_i_fks(0:3),y_ij_fks,xp(0:3,nexternal),p(0:3,nexternal),
     $ xjac,jac,m_j_fks

      integer i,j,i_fks,j_fks

      double precision xi_i_fks_ev,y_ij_fks_ev
      double precision p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt

      double precision xi_i_fks_cnt(-2:2)
      common /cxiifkscnt/xi_i_fks_cnt

      double precision xi_i_hat_ev,xi_i_hat_cnt(-2:2)
      common /cxi_i_hat/xi_i_hat_ev,xi_i_hat_cnt

      double precision xbjrk_ev(2),xbjrk_cnt(2,-2:2)
      common/cbjorkenx/xbjrk_ev,xbjrk_cnt

      double precision sqrtshat_ev,shat_ev
      common/parton_cms_ev/sqrtshat_ev,shat_ev
      double precision sqrtshat_cnt(-2:2),shat_cnt(-2:2)
      common/parton_cms_cnt/sqrtshat_cnt,shat_cnt

      double precision tau_ev,ycm_ev
      common/cbjrk12_ev/tau_ev,ycm_ev
      double precision tau_cnt(-2:2),ycm_cnt(-2:2)
      common/cbjrk12_cnt/tau_cnt,ycm_cnt

      double precision xiimax_ev
      common /cxiimaxev/xiimax_ev
      double precision xiimax_cnt(-2:2)
      common /cxiimaxcnt/xiimax_cnt

      double precision xinorm_ev
      common /cxinormev/xinorm_ev
      double precision xinorm_cnt(-2:2)
      common /cxinormcnt/xinorm_cnt

      double precision p1_cnt(0:3,nexternal,-2:2)
      double precision wgt_cnt(-2:2)
      double precision pswgt_cnt(-2:2)
      double precision jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt

      double precision p_ev(0:3,nexternal)
      common/pev/p_ev
      logical              fixed_order,nlo_ps
      common /c_fnlo_nlops/fixed_order,nlo_ps

c Catch the points for which there is no viable phase-space generation
c (still fill the common blocks with some information that is needed
c (e.g. ycm_cnt)).
      if (xjac .le. 0d0 ) then
         xp(0,1)=-99d0
      endif
c
c Fill common blocks
      if (icountevts.eq.-100) then
         tau_ev=tau
         ycm_ev=ycm
c The second massive-FSR solution has no counterevents. Its lab boost
c still needs the current Born rapidity, not the previous event's value.
         ycm_cnt(0)=ycm_born
         shat_ev=shat
         sqrtshat_ev=sqrtshat
         xbjrk_ev(1)=xbjrk(1)
         xbjrk_ev(2)=xbjrk(2)
         xiimax_ev=xiimax
         xinorm_ev=xinorm
         xi_i_fks_ev=xi_i_fks
         xi_i_hat_ev=xi_i_hat
         do i=0,3
            p_i_fks_ev(i)=p_i_fks(i)
         enddo
         y_ij_fks_ev=y_ij_fks
         do i=1,nexternal
            do j=0,3
               p(j,i)=xp(j,i)
               p_ev(j,i)=xp(j,i)
            enddo
         enddo
         jac=xjac
      else
         tau_cnt(icountevts)=tau
c Special fix in the case the soft counter-events are not generated but
c the Born and real are. (This can happen if ptj>0 in the
c run_card). This fix is needed for set_cms_stuff to work properly.
         if (icountevts.eq.0) then
            ycm=ycm_born
         endif
         ycm_cnt(icountevts)=ycm
         shat_cnt(icountevts)=shat
         sqrtshat_cnt(icountevts)=sqrtshat
         xbjrk_cnt(1,icountevts)=xbjrk(1)
         xbjrk_cnt(2,icountevts)=xbjrk(2)
         xiimax_cnt(icountevts)=xiimax
         xinorm_cnt(icountevts)=xinorm
         xi_i_fks_cnt(icountevts)=xi_i_fks
         xi_i_hat_cnt(icountevts)=xi_i_hat
         do i=0,3
            p_i_fks_cnt(i,icountevts)=p_i_fks(i)
         enddo
         do i=1,nexternal
            do j=0,3
               p1_cnt(j,i,icountevts)=xp(j,i)
            enddo
         enddo
         jac_cnt(icountevts)=xjac
c the following two are obsolete, but still part of some common block:
c so give some non-physical values
         wgt_cnt(icountevts)=-1d99
         pswgt_cnt(icountevts)=-1d99
      endif
      return
      end
