      module fks_phase_space
      use fks_phase_space_data, only: p_born,p_born_l,p_born_ev,p1_cnt,jac_cnt
c Public phase-space stages and their private implementation. Compiler
c interfaces come from contained procedures, not repeated declarations.
c Sampling and radiation kernels live in their own lower-level modules.
      use fks_born_sampling, only: generate_momenta_born,
     $     initialize_born_chart,generate_tau_y_wrapper
      use fks_radiation_maps, only: generate_momenta_massless_final,
     $     generate_momenta_massive_final,generate_momenta_initial,
     $     generate_momenta_massless_final_inverse,
     $     generate_momenta_massive_final_inverse,
     $     generate_momenta_initial_inverse,
     $     generate_momenta_initial_noevpr
      use fks_phase_space_helpers, only: lambda,get_recoil,
     $     boost_isr_recoil
      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
      private
      public fks_born_point,fks_phase_space_configuration,
     $     fks_phase_space_point,capture_fks_phase_space,
     $     restore_fks_phase_space,initialize_fks_phase_space,
     $     sample_fks_born_point,generate_fks_radiation,
     $     generate_born_contribution,generate_born_event,
     $     generate_real_phase_space,generate_prepared_fks_point,
     $     generate_momenta,generate_native_momenta,invert_fks_radiation

c A reusable Born point is an input to radiation generation. Its
c sampling measure and provenance remain independent of any radiation
c point generated from it, including a rejected radiation trial.
      type fks_born_point
         logical :: valid=.false.,sampled=.false.
         logical :: event_projection=.true.
         double precision :: p(0:3,nexternal-1)=0d0
         double precision :: mass(nexternal-1)=0d0
         double precision :: stot=0d0,shat=0d0,sqrtshat=0d0
         double precision :: tau=0d0,ycm=0d0,ycmhat=0d0
         double precision :: xbjrk(2)=0d0,xjac=1d0,xpswgt=1d0
         double precision :: bounds(3)=0d0,omx(2)=0d0
c Provenance for reuse by the integration drivers. Imported/projected
c Born points have sampled=.false.; radiation itself needs no chart.
         integer :: ndim=0,config=0,sector=0,epoch=-1,recoil_leg=0
         integer :: beams(2)=0
         logical :: native=.false.,limit_tests(2)=.false.
         double precision :: coordinates(99)=0d0
         double precision :: external_mass(nexternal)=0d0
         double precision :: beam_energy(2)=0d0
      end type fks_born_point

c One real or endpoint configuration. All momenta use the generation
c frame: with event projection this is the underlying Born CM, which
c need not be the CM of this configuration. direction is p_i/xi and
c remains meaningful at a soft endpoint. xi_hat is the sampled energy
c coordinate; its soft endpoint value must not be replaced by zero.
      type fks_phase_space_configuration
         logical :: valid=.false.
         double precision :: p(0:3,nexternal)=0d0
         double precision :: jacobian=-1d0
         double precision :: tau=0d0,ycm=0d0,shat=0d0,sqrtshat=0d0
         double precision :: xbjrk(2)=0d0
         double precision :: xi=0d0,y=0d0,xi_hat=0d0
         double precision :: xi_max=-1d0,xi_norm=0d0
         double precision :: direction(0:3)=0d0
      end type fks_phase_space_configuration

c A complete generated point, including its subtraction endpoints.
c The Born input is reusable; all other members describe this particular
c radiation trial. Endpoint validity is independent of real validity.
c p_lab is the symmetric hadron frame used by the generation API; the
c additional boost for unequal beam energies is applied by consumers.
      type fks_phase_space_point
         logical :: valid=.false.
         type(fks_born_point) :: born
         double precision :: p(0:3,nexternal)=0d0
         double precision :: p_lab(0:3,nexternal)=0d0
         double precision :: p_cms(0:3,nexternal)=0d0
         double precision :: weight=-1d0
c Radiation coordinates belong to this trial, not the cached Born chart.
         double precision :: radiation_coordinates(3)=0d0
         logical :: has_radiation=.false.,nbody_only=.false.
         type(fks_phase_space_configuration) :: event
         type(fks_phase_space_configuration) :: counterevent(-2:2)
         double precision :: p_born(0:3,nexternal-1)=0d0
         double precision :: p_born_l(0:3,nexternal-1)=0d0
         double precision :: p_born_ev(0:3,nexternal-1)=0d0
c Without event projection the real and collinear reduced Born systems
c differ from the soft Born. Their availability is recorded explicitly.
         double precision :: p_born_coll(0:3,nexternal-1)=0d0
         double precision :: p_born_norad(0:3,nexternal-1)=0d0
         logical :: has_collinear_born=.false.
         logical :: has_reduced_born=.false.
         double complex :: spin_phase=(0d0,0d0)
         double precision :: radiation_energies(3)=0d0
         integer :: solution_sign=0
         logical :: no_counterevents=.true.,event_projection=.true.
         double precision :: bounds(3)=0d0,omx(2)=0d0
c Cursor used by set_cms_stuff: lab/CM boosts, sqrt(shat), shat.
         double precision :: active_cms(4)=0d0
         double precision :: resonance_momentum(0:3)=0d0
         double precision :: resonance_mass2=0d0
         logical :: resonance_recoil=.false.
         logical :: resonance_members(nexternal)=.false.
         integer :: initial_recoil_leg=0
         integer :: sector=0,i_fks=0,j_fks=0,config=0
      end type fks_phase_space_point

c Unlike the other endpoint variables, y had no legacy COMMON slot.
      double precision :: counter_y(-2:2)=0d0


      contains

      subroutine generate_momenta(ndim,iconfig,wgt,x,p,p_lab,p_cms)
c Compatibility entry: generate both Born and radiation, retaining the
c caller's counterevent mode. Some tools require real momenta even when
c cnbody is true, so this entry must not mean "Born momenta only".
      implicit none
      include 'timing_variables.inc'
      integer ndim,iconfig
      double precision wgt,x(99),p(0:3,nexternal),
     $     p_lab(0:3,nexternal),p_cms(0:3,nexternal),started,finished
      logical nbody
      common/cnbody/nbody
      type(fks_phase_space_point) point

      call cpu_time(started)
      point=fks_phase_space_point()
      call initialize_fks_phase_space(iconfig)
      call generate_prepared_fks_point(ndim,x,nbody,wgt,
     $     point)
      p=point%p
      p_lab=point%p_lab
      p_cms=point%p_cms
      call cpu_time(finished)
      tGenPS=tGenPS+finished-started
      end subroutine generate_momenta


      subroutine generate_born_contribution(ndim,iconfig,wgt,x,
     $     point)
c Born/virtual integration still uses the regulated FKS soft-endpoint
c measure and massive-branch support. Keep that adapter distinct from
c sampling a plain Born point, whose measure contains no radiation.
      implicit none
      include 'timing_variables.inc'
      integer ndim,iconfig
      double precision wgt,x(99),started,finished
      type(fks_phase_space_point),intent(out) :: point

      call cpu_time(started)
      point=fks_phase_space_point()
      call initialize_fks_phase_space(iconfig)
      call generate_prepared_fks_point(ndim,x,.true.,wgt,
     $     point)
      call cpu_time(finished)
      tGenPS=tGenPS+finished-started
      end subroutine generate_born_contribution


      subroutine generate_born_event(ndim,iconfig,wgt,x,
     $     point)
      use fks_phase_space_data, only: nocntevents
c Plain Born kinematics for S-event output and mass reshuffling. Slot
c zero contains the Born point with a zero emitted momentum. Its weight
c is the Born measure, not an FKS endpoint measure; integration callers
c must use generate_born_contribution instead.
      use fks_phase_space_helpers, only: boost_n1_to_its_cms,boost_n1_to_lab
      implicit none
      include 'timing_variables.inc'
      integer ndim,iconfig,i,iborn,i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      double precision wgt,x(99),p(0:3,nexternal),
     $     p_lab(0:3,nexternal),p_cms(0:3,nexternal),
     $     xp(0:3,nexternal),p_i_fks(0:3),jac,xpswgt,dummy,
     $     started,finished,ycm,zero
      parameter(zero=0d0)
      type(fks_born_point) born
      type(fks_phase_space_point),intent(out) :: point
      logical pass

      call cpu_time(started)
      call initialize_fks_phase_space(iconfig)
      call sample_fks_born_point(ndim,x,born,pass)
      if(.not.pass)then
         call reject_fks_phase_space(wgt,p,p_lab,p_cms)
         goto 900
      endif
      call reset_fks_kinematics()
      xp=0d0
      iborn=0
      do i=1,nexternal
         if(i.eq.i_fks)cycle
         iborn=iborn+1
         xp(:,i)=born%p(:,iborn)
      enddo
      p=xp
      p_i_fks=0d0
      jac=born%xjac
      xpswgt=born%xpswgt
      call compute_flux(born%shat,born%sqrtshat,
     $     born%mass(1),born%mass(2),xpswgt,jac)
      jac=jac*wgt
      wgt=jac
      ycm=born%ycm
      call fill_fks_point_data(0,born%tau,ycm,born%ycm,
     $     born%shat,born%sqrtshat,born%xbjrk,zero,zero,zero,
     $     zero,p_i_fks,zero,xp,p,jac,dummy)
      nocntevents=.false.
      call boost_n1_to_its_cms(p,p_cms,dummy)
      call boost_n1_to_lab(p,p_lab,-born%ycm)
 900  continue
      call record_fks_phase_space(point,born,wgt,p,p_lab,p_cms)
      point%nbody_only=.true.
      call cpu_time(finished)
      tGenPS=tGenPS+finished-started
      end subroutine generate_born_event


      subroutine generate_real_phase_space(ndim,iconfig,wgt,x,
     $     point)
c Reuse the caller's sampled Born point only when its sampling context
c agrees. Otherwise sample a new one, including the coupled lepton map.
      implicit none
      include 'timing_variables.inc'
      integer ndim,iconfig
      double precision wgt,x(99),started,finished
      type(fks_phase_space_point),intent(inout) :: point

      call cpu_time(started)
      call initialize_fks_phase_space(iconfig)
      call generate_prepared_fks_point(ndim,x,.false.,wgt,
     $     point)
      call cpu_time(finished)
      tGenPS=tGenPS+finished-started
      end subroutine generate_real_phase_space


      subroutine initialize_fks_phase_space(iconfig)
c Install the integration channel, sampling chart and recoil context.
c This routine neither samples momenta nor applies an integration weight.
      implicit none
      include 'nFKSconfigs.inc'
      integer,intent(in) :: iconfig
      double precision pmass(-nexternal:0,lmaxconfigs,0:fks_configs),
     $     pwidth(-nexternal:0,lmaxconfigs,0:fks_configs)
      integer iforest(2,-max_branch:-1,lmaxconfigs,0:fks_configs),
     $     sprop(-max_branch:-1,lmaxconfigs,0:fks_configs),
     $     tprid(-max_branch:-1,lmaxconfigs,0:fks_configs),
     $     mapconfig(0:lmaxconfigs,0:fks_configs)
      common/c_configurations/pmass,pwidth,iforest,sprop,tprid,
     $     mapconfig
      integer itree(2,-max_branch:-1),iconf,this_config
      common/to_itree/itree,iconf
      common/to_mconfigs/this_config
      double precision qmass(-nexternal:0),qwidth(-nexternal:0)
      common/c_qmass_qwidth/qmass,qwidth

      this_config=iconfig
      iconf=iconfig
      itree=iforest(:,:,iconfig,0)
      qmass=pmass(:,iconfig,0)
      qwidth=pwidth(:,iconfig,0)
      call initialize_fks_recoil()
      call initialize_born_chart(itree)
      end subroutine initialize_fks_phase_space


      subroutine initialize_fks_recoil()
      use fks_phase_space_data,only: resonance_momentum,resonance_mass2,resonance_recoil,
     $     resonance_members,initial_recoil_leg
c set_tau_min installs the sector's cut bounds, conflicting BWs and
c resonance descendants. Recoil selection must follow that operation.
      implicit none
      include 'nFKSconfigs.inc'
      logical granny_is_res,granny_chain(-nexternal:nexternal),
     $     granny_chain_real_final(-nexternal:nexternal)
      integer igranny,iaunt,nFKSprocess
      common/c_granny_res/igranny,iaunt,granny_is_res,granny_chain,
     $     granny_chain_real_final
      common/c_nFKSprocess/nFKSprocess
      logical write_granny(fks_configs)
      integer which_is_granny(fks_configs)
      common/write_granny_resonance/which_is_granny,write_granny
      character*4 abrv
      common/to_abrv/abrv

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
      end subroutine initialize_fks_recoil


      subroutine sample_fks_born_point(ndim,x,born,pass)
c Sample only a Born point in the initialized channel. For lepton maps
c without event projection this is the soft Born point; it cannot be
c reused for their radiation-dependent reduced Born systems.
      implicit none
      integer ndim
      double precision x(99),m(-max_branch:max_particles),
     $     s(-max_branch:max_particles),totmass
      type(fks_born_point) born
      logical pass

      call prepare_fks_beam_point(ndim,x,born,m,s,totmass,pass)
      if(.not.pass)return
      call sample_prepared_fks_born(x,born,m,s,totmass,pass)
      end subroutine sample_fks_born_point


      subroutine prepare_fks_beam_point(ndim,x,born,m,s,totmass,pass)
      use fks_phase_space_data, only: initial_recoil_leg,tau_Born_lower_bound,
     $     tau_lower_bound_resonance,tau_lower_bound
c Prepare current masses and sample incoming fractions. No mass sums
c are cached: event output may temporarily replace the external masses.
      use mc_native_context, only: native_epoch,native_mapping
      implicit none
      include 'run.inc'
      integer ndim,i,i_fks,j_fks,this_config,nFKSprocess
      double precision x(99),m(-max_branch:max_particles),
     $     s(-max_branch:max_particles),totmass,totmassin,beam_coordinates(2)
      type(fks_born_point) born
      logical pass,use_evpr,softtest,colltest,one_body
      common/fks_indices/i_fks,j_fks
      common/to_mconfigs/this_config
      common/c_nFKSprocess/nFKSprocess
      common/to_use_evpr/use_evpr
      common/sctests/softtest,colltest
      double precision pmass(nexternal),qmass(-nexternal:0),qwidth(-nexternal:0),omx(2)
      common/to_mass/pmass
      common/c_qmass_qwidth/qmass,qwidth
      common/to_ee_omx1/omx

      born=fks_born_point()
      pass=.false.
      if(ndim.lt.3.or.ndim.gt.size(x))then
         write(*,*) 'Unsupported FKS integration dimension',ndim
         stop 1
      endif
      m=0d0
      s=0d0
      do i=1,nexternal-1
         if(i.lt.i_fks)then
            m(i)=pmass(i)
         else
            m(i)=pmass(i+1)
         endif
      enddo
      born%mass=m(1:nexternal-1)
      if(nincoming.eq.2)then
         born%stot=4d0*ebeam(1)*ebeam(2)
      else
         born%stot=pmass(1)**2
      endif
      totmassin=sum(m(1:nincoming))
      totmass=sum(m(nincoming+1:nexternal-1))
      if(born%stot.lt.max(totmass,totmassin)**2)then
         write(*,*) 'Fatal error #0 in one_tree:'/
     $        /'insufficient collider energy'
         stop
      endif
c A one-particle Born final state fixes tau. Its rapidity uses x(1),
c with no preceding tau coordinate; never form the unused x(0:1) slice.
      beam_coordinates=0.5d0
      if(ndim.gt.4)beam_coordinates(1)=x(ndim-4)
      if(ndim.gt.3)beam_coordinates(2)=x(ndim-3)
      call generate_tau_y_wrapper(qmass,qwidth,totmass,born%stot,
     $     beam_coordinates,born%tau,born%ycm,born%ycmhat,
     $     born%xjac)
      if(born%xjac.lt.0d0)return
      born%xbjrk(1)=sqrt(born%tau)*exp(born%ycm)
      born%xbjrk(2)=sqrt(born%tau)*exp(-born%ycm)
c The old local one_body was uninitialized. Match the multiplicity
c condition in initialize_born_chart and avoid threshold roundoff.
      one_body=(nexternal-nincoming).eq.2
      if(one_body)then
         born%shat=totmass**2
         born%sqrtshat=totmass
      else
         born%shat=born%tau*born%stot
         born%sqrtshat=sqrt(born%shat)
      endif
      if((abs(lpp(1)).eq.1.and.abs(lpp(2)).eq.1).or.
     $   (abs(lpp(1)).eq.2.and.abs(lpp(2)).eq.2).or.
     $   (lpp(1).eq.0.and.lpp(2).eq.0))then
         use_evpr=.true.
      elseif((abs(lpp(1)).eq.3.and.abs(lpp(2)).eq.3).or.
     $       (abs(lpp(1)).eq.4.and.abs(lpp(2)).eq.4))then
         use_evpr=use_evpr.or.j_fks.gt.nincoming
      endif
      born%event_projection=use_evpr
      born%bounds=[tau_Born_lower_bound,tau_lower_bound_resonance,tau_lower_bound]
      born%omx=omx
      born%ndim=ndim
      born%config=this_config
      born%sector=nFKSprocess
      born%epoch=native_epoch
      born%native=native_mapping
      born%recoil_leg=initial_recoil_leg
      born%coordinates(1:ndim)=x(1:ndim)
      born%external_mass=pmass
      born%beam_energy=ebeam
      born%beams=lpp
      born%limit_tests=[softtest,colltest]
      pass=.true.
      end subroutine prepare_fks_beam_point


      subroutine sample_prepared_fks_born(x,born,m,s,totmass,pass)
      implicit none
      double precision x(99),m(-max_branch:max_particles),
     $     s(-max_branch:max_particles),totmass
      type(fks_born_point) born
      logical pass
      double precision qmass(-nexternal:0),qwidth(-nexternal:0)
      common/c_qmass_qwidth/qmass,qwidth

      call generate_momenta_born(x,born%shat,born%sqrtshat,totmass,
     $     m,s,qmass,qwidth,born%mass,born%xpswgt,born%xjac)
      pass=born%xjac.ge.0d0
      born%valid=pass
      born%sampled=pass
      if(pass)born%p=p_born_l
      end subroutine sample_prepared_fks_born


      logical function fks_born_matches(born,ndim,x)
      use fks_phase_space_data, only: initial_recoil_leg,tau_Born_lower_bound,
     $     tau_lower_bound_resonance,tau_lower_bound
c Deliberately require the same sector as well as the same Born chart.
c Equal Born flavours alone do not imply equal cuts/BW sampling maps.
      use mc_native_context, only: native_epoch,native_mapping
      implicit none
      include 'run.inc'
      type(fks_born_point) born
      integer ndim,this_config,nFKSprocess
      double precision x(99),pmass(nexternal)
      logical softtest,colltest
      common/to_mconfigs/this_config
      common/c_nFKSprocess/nFKSprocess
      common/to_mass/pmass
      common/sctests/softtest,colltest

      fks_born_matches=.false.
      if(.not.born%valid.or..not.born%sampled.or.
     $     .not.born%event_projection)return
      if(born%ndim.ne.ndim.or.born%config.ne.this_config.or.
     $     born%sector.ne.nFKSprocess.or.born%epoch.ne.native_epoch)
     $     return
      if(born%native.neqv.native_mapping)return
      if(born%recoil_leg.ne.initial_recoil_leg)return
      if(any(born%external_mass.ne.pmass).or.
     $     any(born%beam_energy.ne.ebeam).or.
     $     any(born%beams.ne.lpp).or.
     $     any(born%bounds.ne.[tau_Born_lower_bound,
     $     tau_lower_bound_resonance,tau_lower_bound]))return
      if(any(born%limit_tests.neqv.[softtest,colltest]))return
      if(any(born%coordinates(1:ndim-3).ne.x(1:ndim-3)))return
      fks_born_matches=.true.
      end function fks_born_matches


      subroutine generate_prepared_fks_point(ndim,x,nbody_only,wgt,
     $     point)
c Compose Born sampling and radiation in a channel already initialized.
c Only the dressed-lepton map without event projection must generate
c radiation first and a different reduced Born system for each endpoint.
      use fks_phase_space_helpers, only: boost_n1_to_its_cms,boost_n1_to_lab
      implicit none
      integer ndim
      logical,intent(in) :: nbody_only
      logical pass
      type(fks_born_point) born
      type(fks_phase_space_point),intent(inout) :: point
      double precision x(99),wgt,p(0:3,nexternal),
     $     p_lab(0:3,nexternal),p_cms(0:3,nexternal),
     $     m(-max_branch:max_particles),s(-max_branch:max_particles),
     $     totmass,jac,dummy,qmass(-nexternal:0),qwidth(-nexternal:0)
      common/c_qmass_qwidth/qmass,qwidth

c Copy the reusable input before replacing the generated result.
      born=point%born
      if(.not.fks_born_matches(born,ndim,x))then
         call prepare_fks_beam_point(ndim,x,born,m,s,totmass,pass)
         if(.not.pass)goto 900
         if(born%event_projection)then
            call sample_prepared_fks_born(x,born,m,s,totmass,pass)
            if(.not.pass)goto 900
         else
            p=0d0
            p(0,1)=-99d0
            p_lab=p
            p_cms=p
            call generate_noevpr_kinematics(x,ndim,nbody_only,born%xjac,
     $           born%xpswgt,born%stot,born%shat,born%sqrtshat,
     $           born%tau,born%ycm,born%ycmhat,totmass,born%xbjrk,
     $           m,s,qmass,qwidth,born%mass,jac,p,pass)
            if(.not.pass.or.born%xjac.lt.0d0)goto 900
            jac_cnt=jac_cnt*wgt
            wgt=wgt*jac
            if(p(0,1).gt.0d0)then
               call boost_n1_to_its_cms(p,p_cms,dummy)
               call boost_n1_to_lab(p,p_lab,-born%ycm)
            endif
            call record_fks_phase_space(point,born,wgt,
     $           p,p_lab,p_cms)
            point%radiation_coordinates=x(ndim-2:ndim)
            point%has_radiation=.true.
            point%nbody_only=nbody_only
            return
         endif
      endif
      call generate_fks_radiation(born,x(ndim-2:ndim),nbody_only,
     $     wgt,point,pass)
      return

 900  continue
      born%valid=.false.
      call reject_fks_phase_space(wgt,p,p_lab,p_cms)
      call record_fks_phase_space(point,born,wgt,p,p_lab,p_cms)
      point%radiation_coordinates=x(ndim-2:ndim)
      point%has_radiation=.true.
      point%nbody_only=nbody_only
      end subroutine generate_prepared_fks_point


      subroutine generate_fks_radiation(born,rad,nbody_only,
     $     wgt,point,pass)
      use fks_phase_space_data, only: tau_Born_lower_bound,tau_lower_bound_resonance,tau_lower_bound
c Generate radiation from supplied Born CM momenta, without any Born
c chart, beam sampling or recoil selection. The active sector/recoiler
c must already be installed. The immutable Born measures never include
c the caller weight: apply it once to real and counterevent Jacobians.
      use fks_phase_space_helpers, only: boost_n1_to_its_cms,boost_n1_to_lab
      implicit none
      type(fks_born_point),intent(in) :: born
      type(fks_phase_space_point),intent(out) :: point
      logical,intent(in) :: nbody_only
      logical,intent(out) :: pass
      logical use_evpr
      double precision rad(3),wgt,p(0:3,nexternal),
     $     p_lab(0:3,nexternal),p_cms(0:3,nexternal),
     $     xjac,jac,dummy
      double precision omx(2)
      common/to_ee_omx1/omx
      common/to_use_evpr/use_evpr

      pass=.false.
      p=0d0
      p(0,1)=-99d0
      p_lab=p
      p_cms=p
      if(.not.born%valid.or..not.born%event_projection)then
         call reject_fks_phase_space(wgt,p,p_lab,p_cms)
         goto 900
      endif
      p_born=born%p
      p_born_l=born%p
      p_born_ev=born%p
      tau_Born_lower_bound=born%bounds(1)
      tau_lower_bound_resonance=born%bounds(2)
      tau_lower_bound=born%bounds(3)
      omx=born%omx
      use_evpr=.true.
      xjac=born%xjac
      call generate_FKS_kinematics(rad,nbody_only,xjac,born%xpswgt,born%stot,
     $     born%shat,born%sqrtshat,born%tau,born%ycm,born%ycmhat,
     $     born%xbjrk,born%mass,jac,p,pass)
      if(.not.pass.or.xjac.lt.0d0)then
         pass=.false.
         call reject_fks_phase_space(wgt,p,p_lab,p_cms)
         goto 900
      endif
      jac_cnt=jac_cnt*wgt
      wgt=wgt*jac
      if(p(0,1).gt.0d0)then
         call boost_n1_to_its_cms(p,p_cms,dummy)
         call boost_n1_to_lab(p,p_lab,-born%ycm)
      endif
 900  continue
      call record_fks_phase_space(point,born,wgt,p,p_lab,p_cms)
      point%radiation_coordinates=rad
      point%has_radiation=.true.
      point%nbody_only=nbody_only
      end subroutine generate_fks_radiation


      subroutine reject_fks_phase_space(wgt,p,p_lab,p_cms)
      use fks_phase_space_data, only: p_ev,nocntevents
c A rejected Born sample invalidates every event slot, including slots
c unused by the current sector. Negative values are validity sentinels.
      implicit none
      double precision wgt,p(0:3,nexternal),p_lab(0:3,nexternal),p_cms(0:3,nexternal)
      call reset_fks_kinematics()
      jac_cnt=-222d0*abs(wgt)
      wgt=-222d0*abs(wgt)
      p=0d0
      p(0,1)=-99d0
      p_lab=p
      p_cms=p
      p_ev=p
      p1_cnt=0d0
      p1_cnt(0,1,:)=-99d0
      p_born=0d0
      p_born_l=0d0
      p_born_ev=0d0
      p_born(0,1)=-99d0
      p_born_l(0,1)=-99d0
      p_born_ev(0,1)=-99d0
      nocntevents=.true.
      end subroutine reject_fks_phase_space


      subroutine reset_fks_kinematics()
      use fks_phase_space_data, only: resonance_momentum,resonance_mass2,p_ev,p_born_coll,
     $     p_born_norad,xi_ev => xi_i_fks_ev,y_ev => y_ij_fks_ev,p_i_ev => p_i_fks_ev,
     $     p_i_cnt => p_i_fks_cnt,xi_cnt => xi_i_fks_cnt,xi_hat_ev => xi_i_hat_ev,
     $     xi_hat_cnt => xi_i_hat_cnt,xbjrk_ev,xbjrk_cnt,sqrtshat_ev,shat_ev,sqrtshat_cnt,shat_cnt,
     $     tau_ev,ycm_ev,tau_cnt,ycm_cnt,xi_max_ev => xiimax_ev,xi_max_cnt => xiimax_cnt,
     $     xi_norm_ev => xinorm_ev,xi_norm_cnt => xinorm_cnt,spin_phase => xij_aor,veckn_ev,
     $     veckbarn_ev,xp0jfks,solution_sign => isolsign,no_counterevents => nocntevents,
     $     ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
c Start a fresh real/endpoint result. Preserve the supplied Born point
c and selected recoil inputs, but invalidate every generated output so
c absent endpoints cannot inherit data from a previous radiation trial.
      implicit none

      p_ev=0d0
      p_ev(0,1)=-99d0
      p1_cnt=0d0
      p1_cnt(0,1,:)=-99d0
      jac_cnt=-1d0
      xi_ev=0d0
      y_ev=0d0
      p_i_ev=0d0
      p_i_ev(0)=-1d0
      p_i_cnt=0d0
      p_i_cnt(0,:)=-1d0
      xi_cnt=0d0
      xi_hat_ev=0d0
      xi_hat_cnt=0d0
      counter_y=0d0
      xbjrk_ev=0d0
      xbjrk_cnt=0d0
      sqrtshat_ev=0d0
      shat_ev=0d0
      sqrtshat_cnt=0d0
      shat_cnt=0d0
      tau_ev=0d0
      ycm_ev=0d0
      tau_cnt=0d0
      ycm_cnt=0d0
      xi_max_ev=-1d0
      xi_max_cnt=-1d0
      xi_norm_ev=0d0
      xi_norm_cnt=0d0
      spin_phase=(0d0,0d0)
      veckn_ev=0d0
      veckbarn_ev=0d0
      xp0jfks=0d0
      solution_sign=0
      no_counterevents=.true.
      ybst_til_tolab=1d14
      ybst_til_tocm=1d14
      sqrtshat=0d0
      shat=0d0
      resonance_momentum=0d0
      resonance_mass2=0d0
      p_born_coll=0d0
      p_born_coll(0,1)=-99d0
      p_born_norad=0d0
      p_born_norad(0,1)=-99d0
      end subroutine reset_fks_kinematics


      subroutine generate_FKS_kinematics(rad,nbody_only,xjac0,xpswgt0,
     $  stot,shat_born,sqrtshat_born,tau_born,ycm_born,ycmhat,
     $  xbjrk_born,m_born,jac,p,pass)
      use fks_phase_space_data, only: resonance_momentum,resonance_mass2,resonance_recoil,
     $     resonance_members,initial_recoil_leg,nocntevents,sqrtshat,shat,isolsign
      implicit none

      double precision xjac0,xpswgt0,p(0:3,nexternal),
     $   stot,shat_born,sqrtshat_born,tau_born,ycm_born,ycmhat,jac
      double precision xbjrk_born(2)
c The momentum checker expects the full legacy branch-indexed layout.
      double precision m(-max_branch:max_particles),m_born(nexternal-1)
      double precision,intent(in) :: rad(3)
      logical,intent(in) :: nbody_only
      logical pass

      integer icountevts
      integer imother
      double precision xmrec2,m_j_fks,phi_i_fks,tau,
     $   xi_i_fks,y_ij_fks,xi_i_hat,xiimax,xinorm,xjac,xpswgt,
     $   ycm,xp(0:3,nexternal),xbjrk(2),p_i_fks(0:3),beam_ratio
      integer i,ireal

      real*8 pi
      parameter (pi=3.1415926535897932d0)

      double precision pmass(nexternal)
      common /to_mass/pmass


      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks


      logical fks_as_is
      parameter (fks_as_is=.false.)

      ! check that the starting PS point is meaningful
      if (xjac0.lt.0d0) then
         pass = .false.
         return
      endif
      m=0d0
c
c Here we start with the FKS Stuff
c
c icountevts=-100 is the event, -2 to 2 the counterevents
      icountevts = -100
      call reset_fks_kinematics()
c The sampled energy coordinate is shared with subsequent endpoints.
      xi_i_hat=0d0
c
c The three radiation coordinates retain the integration fold order:
c energy, polar angle, azimuth.
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
c A rejected map may return before assigning every output. Keep its
c snapshot deterministic while retaining the Born boost for consumers.
      tau=tau_born
      ycm=ycm_born
      shat=shat_born
      sqrtshat=sqrtshat_born
      xbjrk=xbjrk_born
      xiimax=-1d0
      xinorm=0d0
      xi_i_fks=0d0
      y_ij_fks=0d0
      p_i_fks=0d0
      p_i_fks(0)=-1d0
c
c Put the Born momenta in the xp momenta, making sure that the mapping
c is correct; put i_fks momenta equal to zero.
      xp(:,i_fks)=0d0
      m(i_fks)=0d0
      do i=1,nexternal-1
         ireal=i
         if(i.ge.i_fks)ireal=i+1
         xp(:,ireal)=p_born_l(:,i)
         m(ireal)=m_born(i)
      enddo


c
c set-up phi_i_fks
c
      phi_i_fks=2d0*pi*rad(3)
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
     $           xbjrk_born(initial_recoil_leg),rad(1:2),phi_i_fks,
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
     $           rad(1:2),phi_i_fks,xp,xiimax,xinorm,xi_i_fks,
     $           y_ij_fks,xi_i_hat,p_i_fks,xjac,xpswgt,
     $           resonance_momentum,resonance_mass2,pass)
            if (.not.pass) goto 112
         elseif (m_j_fks.eq.0d0) then
            isolsign=1
            call generate_momenta_massless_final(icountevts,i_fks,j_fks
     &           ,p_born_l(0,imother),shat,sqrtshat,rad(1:2),xmrec2,xp
     &           ,phi_i_fks,xiimax,xinorm,xi_i_fks,y_ij_fks,xi_i_hat
     &           ,p_i_fks,xjac,xpswgt,pass)
            if (.not.pass) goto 112
         elseif(m_j_fks.gt.0d0) then
            call generate_momenta_massive_final(icountevts,isolsign
     &           ,i_fks,j_fks,p_born_l(0,imother)
     &           ,shat,sqrtshat,m_j_fks,rad(1:2),xmrec2,xp,phi_i_fks
     &           ,xiimax,xinorm,xi_i_fks,y_ij_fks,xi_i_hat,p_i_fks,xjac
     &           ,xpswgt,pass)
            if (.not.pass) goto 112
         endif
      elseif(j_fks.le.nincoming) then
         isolsign=1
         call generate_momenta_initial(icountevts,i_fks,j_fks,xbjrk_born
     &        ,tau_born,ycm_born,ycmhat,shat_born,phi_i_fks,xp,rad(1:2)
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
      call fill_fks_point_data(icountevts,tau,ycm,ycm_born,shat,sqrtshat,xbjrk,
     $      xiimax,xinorm,xi_i_fks,xi_i_hat,p_i_fks,y_ij_fks,xp,p,xjac,jac)

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
      if( (icountevts.le.2.and.m_j_fks.eq.0.d0.and.(.not.nbody_only)).or.
     &    (icountevts.eq.0.and.m_j_fks.eq.0.d0.and.nbody_only) .or.
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
      end subroutine generate_FKS_kinematics


      subroutine generate_noevpr_kinematics(x,ndim,nbody_only,xjac0,xpswgt0,
     $  stot,shat_born,sqrtshat_born,tau_born,ycm_born,ycmhat,totmass,
     $  xbjrk_born,m,s,qmass,qwidth,m_born,jac,p,pass)
      use fks_phase_space_data, only: p_born_coll,p_born_norad,nocntevents,sqrtshat,shat,isolsign
      ! generate the kinematics without event projection.
      ! In this case, the bjorken x's are kept the same for all contributions
      ! (event and coutnerevents).
      ! First one generates the radiation (I_fks), then the reduced Born 
      ! system with a com energy sborn=(1-xi)*shat
      implicit none


      double precision xjac0,xpswgt0,x(99),p(0:3,nexternal),
     $   stot,shat_born,sqrtshat_born,tau_born,ycm_born,ycmhat,totmass,jac
      double precision xbjrk_born(2)
      double precision M(-max_branch:max_particles),S(-max_branch:max_particles),
     $   m_born(nexternal-1)
c Keep checker masses in real indexing separate from the Born chart.
      double precision real_mass(-max_branch:max_particles)
      integer ndim
      logical pass
      double precision qmass(-nexternal:0),qwidth(-nexternal:0)

      integer icountevts
      integer ixEi,ixpi
      double precision m_j_fks,phi_i_fks,tau,
     $   xi_i_fks,y_ij_fks,xi_i_hat,xiimax,xinorm,xjac,xpswgt,
     $   ycm,xp(0:3,nexternal),xbjrk(2),p_i_fks(0:3)
      integer i,ireal

      real*8 pi
      parameter (pi=3.1415926535897932d0)

      double precision pmass(nexternal)
      common /to_mass/pmass


      logical,intent(in) :: nbody_only


      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks


      integer skip
      double precision srec

      logical fks_as_is
      parameter (fks_as_is=.false.)

c The coupled map is used for incoming emitters. Set the mass before
c deciding which counterevents exist; do not rely on static storage.
      m_j_fks=pmass(j_fks)
      real_mass=0d0
c Here we start with the FKS Stuff
c
c icountevts=-100 is the event, -2 to 2 the counterevents
      icountevts = -100
      call reset_fks_kinematics()
c The sampled energy coordinate is shared with subsequent endpoints.
      xi_i_hat=0d0

      ! if we do not do event projection, we first generate y/xi FKS
      ! and p_i_fks, then the other momenta
      isolsign=1
c
c These will correspond to the vegas x's for the FKS variables xi_i,
c y_ij and phi_i (changing this also requires changing folding parameters)
      ixEi=ndim-2
      ixpi=ndim

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCc
c Here is the beginning of the loop over the momenta for the event and
c counter-events. This will fill the xp momenta with the event and
c counter-event momenta.
 111  continue
      xjac   = xjac0
      xpswgt = xpswgt0
c A rejected map may return before assigning every output. Keep its
c snapshot deterministic while retaining the Born boost for consumers.
      tau=tau_born
      ycm=ycm_born
      shat=shat_born
      sqrtshat=sqrtshat_born
      xbjrk=xbjrk_born
      xiimax=-1d0
      xinorm=0d0
      xi_i_fks=0d0
      y_ij_fks=0d0
      p_i_fks=0d0
      p_i_fks(0)=-1d0
      xp=0d0
c
c set-up phi_i_fks
c
      phi_i_fks=2d0*pi*x(ixpi)
      xjac=xjac*2d0*pi

      call generate_momenta_initial_noevpr(icountevts,i_fks,j_fks,xbjrk_born
     &        ,tau_born,ycm_born,ycmhat,shat_born,phi_i_fks,xp,x(ixEi)
     &        ,shat,stot,sqrtshat,tau,ycm,xbjrk,p_i_fks,xiimax,xinorm
     &        ,xi_i_fks,y_ij_fks,xi_i_hat,xpswgt,xjac,srec,pass)

c A rejected radiation map has no reduced Born invariant to sample.
      if(.not.pass.or.xjac.lt.0d0)then
         p_born(0,1)=-100d0
         p_born_l(0,1)=-100d0
         p_born_ev(0,1)=-100d0
         goto 112
      endif

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
     &     y_ij_fks,phi_i_fks,i_fks,j_fks,shat,srec)

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
        do i = 1, nexternal-1
          p_born_coll(0:3,i) = p_born(0:3,i)
        enddo
      endif
c
c Preserve Born-indexed m for the next endpoint's Born generation.
c Only the checker needs the emitted leg inserted in real indexing.
      real_mass(i_fks)=0d0
      do i=1,nexternal-1
         ireal=i
         if(i.ge.i_fks)ireal=i+1
         real_mass(ireal)=m_born(i)
      enddo

c All done, so check four-momentum conservation

      if(xjac.gt.0.d0)then
         call phspncheck_nocms(nexternal,sqrtshat,real_mass,xp,pass)
         if (.not.pass) then
            xjac=-199
            goto 112
         endif
      endif

      call compute_flux(shat,sqrtshat,real_mass(1),real_mass(2),
     $     xpswgt,xjac)
c      
 112  continue

      call fill_fks_point_data(icountevts,tau,ycm,ycm_born,shat,sqrtshat,xbjrk,
     $      xiimax,xinorm,xi_i_fks,xi_i_hat,p_i_fks,y_ij_fks,xp,p,xjac,jac)
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
      if( (icountevts.le.2.and.m_j_fks.eq.0.d0.and.(.not.nbody_only)).or.
     &    (icountevts.eq.0.and.m_j_fks.eq.0.d0.and.nbody_only) .or.
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
      end subroutine generate_noevpr_kinematics


      subroutine boost_born_momenta_noevpr(pborn,xp,xi_i_fks,
     &     y_ij_fks,phi_i_fks,i_fks,j_fks,shat,srec)
c The lepton chart samples the real incoming fractions first. Apply
c the same ISR recoil to its reduced Born, then express the result in
c the real CM. Only the sampling coordinates/measure differ from the
c event-projection chart; the finite-angle recoil is common to both.
      implicit none
      double precision pborn(0:3,nexternal-1),xp(0:3,nexternal)
      double precision xi_i_fks,y_ij_fks,phi_i_fks,shat,srec
      double precision pred(0:3),sqrtz,pplus,pminus
      integer i_fks,j_fks,idir,i,ireal
      idir=3-2*j_fks
      sqrtz=sqrt(1d0-xi_i_fks)
      pborn(0,1:2)=sqrt(shat)/2d0
      pborn(1:2,1:2)=0d0
      pborn(3,1)=pborn(0,1)
      pborn(3,2)=-pborn(0,2)
      xp(:,1:2)=pborn(:,1:2)
      do i=3,nexternal-1
         ireal=i
         if(i.ge.i_fks)ireal=i+1
         call boost_isr_recoil(pborn(0,i),pred,xi_i_fks,
     &        y_ij_fks,phi_i_fks,idir,.false.)
         pplus=(pred(0)+idir*pred(3))*sqrtz
         pminus=(pred(0)-idir*pred(3))/sqrtz
         xp(0,ireal)=(pplus+pminus)/2d0
         xp(1:2,ireal)=pred(1:2)
         xp(3,ireal)=idir*(pplus-pminus)/2d0
      enddo
      end subroutine boost_born_momenta_noevpr


      subroutine compute_flux(shat,sqrtshat,m1,m2,xpswgt,xjac)
      implicit none
      double precision shat,sqrtshat,m1,m2,xpswgt,xjac
      double precision flux
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
      end subroutine compute_flux


      subroutine fill_fks_point_data(icountevts,tau,ycm,ycm_born,shat,sqrtshat,xbjrk,
     $      xiimax,xinorm,xi_i_fks,xi_i_hat,p_i_fks,y_ij_fks,xp,p,xjac,jac)
      use fks_phase_space_data,only: xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt,xi_i_fks_cnt,
     $     xi_i_hat_ev,xi_i_hat_cnt,xbjrk_ev,xbjrk_cnt,sqrtshat_ev,shat_ev,sqrtshat_cnt,shat_cnt,tau_ev,
     $     ycm_ev,tau_cnt,ycm_cnt,xiimax_ev,xiimax_cnt,xinorm_ev,xinorm_cnt,p_ev
      implicit none
      integer,intent(in) :: icountevts
      double precision,intent(in) :: tau,ycm,ycm_born,shat,sqrtshat,
     $     xbjrk(2),xiimax,xinorm,xi_i_fks,xi_i_hat,p_i_fks(0:3),
     $     y_ij_fks,xp(0:3,nexternal),xjac
c Only the real slot writes the returned momenta and real Jacobian.
      double precision,intent(inout) :: p(0:3,nexternal),jac
c
c Keep metadata even for rejected slots; mark only their stored momenta
c invalid. The caller's kinematics and rapidity remain input values.
      if (icountevts.eq.-100) then
         tau_ev=tau
         ycm_ev=ycm
c The second massive-FSR solution has no counterevents. Its lab boost
c still needs the current Born rapidity, not the previous event's value.
         ycm_cnt(0)=ycm_born
         shat_ev=shat
         sqrtshat_ev=sqrtshat
         xbjrk_ev=xbjrk
         xiimax_ev=xiimax
         xinorm_ev=xinorm
         xi_i_fks_ev=xi_i_fks
         xi_i_hat_ev=xi_i_hat
         p_i_fks_ev=p_i_fks
         y_ij_fks_ev=y_ij_fks
         p=xp
         if(xjac.le.0d0)p(0,1)=-99d0
         p_ev=p
         jac=xjac
      else
         counter_y(icountevts)=0d0
         if(xjac.gt.0d0)counter_y(icountevts)=y_ij_fks
         tau_cnt(icountevts)=tau
c Special fix in the case the soft counter-events are not generated but
c the Born and real are. (This can happen if ptj>0 in the
c run_card). This fix is needed for set_cms_stuff to work properly.
         ycm_cnt(icountevts)=ycm
         if(icountevts.eq.0)ycm_cnt(icountevts)=ycm_born
         shat_cnt(icountevts)=shat
         sqrtshat_cnt(icountevts)=sqrtshat
         xbjrk_cnt(:,icountevts)=xbjrk
         xiimax_cnt(icountevts)=xiimax
         xinorm_cnt(icountevts)=xinorm
         xi_i_fks_cnt(icountevts)=xi_i_fks
         xi_i_hat_cnt(icountevts)=xi_i_hat
         p_i_fks_cnt(:,icountevts)=p_i_fks
         p1_cnt(:,:,icountevts)=xp
         if(xjac.le.0d0)p1_cnt(0,1,icountevts)=-99d0
         jac_cnt(icountevts)=xjac
      endif
      return
      end subroutine fill_fks_point_data


      subroutine generate_native_momenta(p_input,p,p_lab,p_cms,
     $     jac,pass,point)
      use fks_phase_space_data, only: tau_Born_lower_bound,tau_lower_bound_resonance,tau_lower_bound
! Project a fixed real point onto its native Born and invert ONLY the
! three radiation coordinates. The Born sampling measure is common to
! the real and counterevents and cancels in repartition_MC_H. Generate
! their radiation/flux measures with a unit Born measure instead.
      use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
      use mc_native_context, only: native_mapping
      implicit none
      include 'run.inc'
      double precision p_input(0:3,nexternal),p(0:3,nexternal),
     $     p_lab(0:3,nexternal),p_cms(0:3,nexternal),jac
      logical pass
      double precision pmass(nexternal)
      common/to_mass/pmass
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      double precision omx_ee(2)
      common/to_ee_omx1/omx_ee
      logical use_evpr
      common/to_use_evpr/use_evpr
      integer this_config
      common/to_mconfigs/this_config
      double precision x(3),pb(0:3,-max_branch:nexternal-1),
     $     m_born(nexternal-1),stot,tau_born,ycm_born,
     $     xbjrk_born(2),xjac0,xpswgt0,
     $     bounds_save(3),omx_save(2)
      type(fks_born_point) born
      type(fks_phase_space_point) generated
      type(fks_phase_space_point),optional,intent(out) :: point
      integer i

      pass=.false.
      born=fks_born_point()
      x=0d0
      bounds_save=[tau_Born_lower_bound,tau_lower_bound_resonance,tau_lower_bound]
      omx_save=omx_ee
      if(present(point))point=fks_phase_space_point()
      jac=-1d0
      p=0d0
      p(0,1)=-1d0
      p_lab=p
      p_cms=p
! Dressed lepton beams require their separate endpoint parametrisation.
! This path has the same two-beam event-projection scope as the old one.
      if (.not.native_mapping.or.nincoming.ne.2.or.
     $     any(abs(lpp).gt.2)) then
         write(*,*) 'Unsupported beams/map in generate_native_momenta'
         stop 1
      endif
      use_evpr=.true.
      if (.not.all(ieee_is_finite(p_input)).or.
     $     any(p_input(0,1:2).le.0d0))goto 900
      stot=4d0*ebeam(1)*ebeam(2)
      do i=1,nexternal-1
         if(i.lt.i_fks)then
            m_born(i)=pmass(i)
         else
            m_born(i)=pmass(i+1)
         endif
      enddo
! These are inputs to the radiation maps, not persistent native outputs.
! In particular a stale lepton endpoint must not change an ISR history.
      tau_Born_lower_bound=sum(m_born(nincoming+1:nexternal-1))**2/stot
      tau_lower_bound_resonance=sum(m_born(nincoming+1:nexternal-1))**2/stot
      tau_lower_bound=sum(m_born(nincoming+1:nexternal-1))**2/stot
      omx_ee=0d0
      x=0d0
      pb=0d0
      xjac0=1d0
      xpswgt0=1d0
      call invert_fks_radiation(x(1:3),xjac0,xpswgt0,stot,
     $     tau_born,ycm_born,xbjrk_born,p_input,pb)
      if (.not.ieee_is_finite(xjac0).or.xjac0.le.0d0.or.
     $     .not.all(ieee_is_finite(x(1:3))))goto 900
      if (.not.all(ieee_is_finite(pb(:,1:nexternal-1))).or.
     $     any(pb(0,1:2).le.0d0).or.
     $     .not.ieee_is_finite(tau_born).or.tau_born.le.0d0)
     $     goto 900
      born=fks_born_point()
      born%valid=.true.
      born%native=.true.
      born%external_mass=pmass
      born%beam_energy=ebeam
      born%beams=lpp
      born%p=pb(:,1:nexternal-1)
      born%mass=m_born
      born%stot=stot
      born%tau=tau_born
      born%ycm=ycm_born
      born%xbjrk=xbjrk_born
      born%shat=tau_born*stot
      born%sqrtshat=sqrt(born%shat)
      if(tau_born.lt.1d0)
     $     born%ycmhat=ycm_born/(-0.5d0*log(tau_born))
      born%bounds=[tau_Born_lower_bound,tau_lower_bound_resonance,tau_lower_bound]
      born%omx=omx_ee
! The inverse radiation measure only checks the projection. Native
! forward densities use the same unit Born measure for every history.
      born%xjac=1d0
      born%xpswgt=1d0
! No Born topology is sampled. A valid native index is still needed by
! clustering; the outer replay restores its integration-channel index.
      this_config=1
      use_evpr=.true.
      jac=1d0
      call generate_fks_radiation(born,x,.false.,jac,
     $     generated,pass)
      p=generated%p
      p_lab=generated%p_lab
      p_cms=generated%p_cms
      pass=ieee_is_finite(jac).and.jac.gt.0d0.and.p(0,1).gt.0d0
      if(.not.pass)goto 900
      pass=all(ieee_is_finite(p_lab)).and.
     $     maxval(abs(p_lab-p_input)).le.
     $     1d-7*max(1d0,maxval(abs(p_input)))
 900  continue
      tau_Born_lower_bound=bounds_save(1)
      tau_lower_bound_resonance=bounds_save(2)
      tau_lower_bound=bounds_save(3)
      omx_ee=omx_save
      if(.not.pass)then
         jac=-1d0
         call reject_fks_phase_space(jac,p,p_lab,p_cms)
         jac=-1d0
         call record_fks_phase_space(generated,born,jac,
     $        p,p_lab,p_cms)
         generated%radiation_coordinates=x
         generated%has_radiation=.true.
      endif
c The returned snapshot describes the restored active controls. The
c native generation inputs remain separately in generated%born.
      generated%bounds=bounds_save
      generated%omx=omx_save
      if(present(point))point=generated
      end subroutine generate_native_momenta


      subroutine invert_fks_radiation(xx,xjac0,xpswgt0,
     $     stot,tau_born,ycm_born,xbjrk_born,p_lab,pb)
      use fks_phase_space_data,only: resonance_momentum,resonance_mass2,resonance_recoil,
     $     resonance_members,initial_recoil_leg,p_i_fks_cnt
! Input momenta are in the symmetric hadron frame used by generate_momenta.
! No Born integration-channel coordinates or Jacobians are recovered.
      use fks_phase_space_helpers, only: boost_n1_to_its_cms,
     $     boost_n1_to_lab,get_xi_from_p,get_yij_from_p,get_phi_from_p
      implicit none
      double precision xjac0,xpswgt0,xx(3),p_cms(0:3,nexternal),stot
     $     ,tau_born,ycm_born,xbjrk_born(2),pb(0:3,
     $     -max_branch:nexternal-1),p_lab(0:3,nexternal)
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      double precision pmass(nexternal)
      common /to_mass/pmass
      double precision m_j_fks,xi_i_fks,y_ij_fks,phi_i_fks,xbjrk(2),shat
     $     ,sqrtshat,tau,ycm,y_lab_to_cms,xbar,trial_jac,trial_ps
      logical pass
      m_j_fks=pmass(j_fks)

      xbjrk(1:2)=p_lab(0,1:2)/(sqrt(stot)/2d0)

      if(initial_recoil_leg.gt.0.and.j_fks.gt.nincoming)then
c First find the covariant Born projection and its incoming fractions.
c Recover azimuth only after returning the real point to that Born CM:
c a direct inversion in the lab would use a different transverse basis.
         trial_jac=1d0
         trial_ps=1d0
         call invert_momenta_initial_recoil(p_lab,initial_recoil_leg,
     $        xbjrk(initial_recoil_leg),i_fks,j_fks,m_j_fks,
     $        pb(:,1:nexternal-1),xx,trial_jac,trial_ps,
     $        resonance_momentum,resonance_mass2,xbar,pass)
         if(.not.pass)then
            xjac0=-148d0
            return
         endif
         xbjrk_born=xbjrk
         xbjrk_born(initial_recoil_leg)=xbar
         tau_born=xbjrk_born(1)*xbjrk_born(2)
         ycm_born=log(xbjrk_born(1)/xbjrk_born(2))/2d0
         call boost_n1_to_lab(p_lab,p_cms,ycm_born)
         call invert_momenta_initial_recoil(p_cms,initial_recoil_leg,
     $        xbjrk(initial_recoil_leg),i_fks,j_fks,m_j_fks,
     $        pb(:,1:nexternal-1),xx,xjac0,xpswgt0,
     $        resonance_momentum,resonance_mass2,xbar,pass)
         if(.not.pass)xjac0=-148d0
         return
      endif

      call boost_n1_to_its_cms(p_lab,p_cms,y_lab_to_cms)

      xi_i_fks=get_xi_from_p(i_fks,j_fks,p_cms)
      y_ij_fks=get_yij_from_p(i_fks,j_fks,p_cms,
     $     p_i_fks_cnt(:,0))
      phi_i_fks=get_phi_from_p(i_fks,j_fks,p_cms)

      ycm=log(xbjrk(1)/xbjrk(2))/2d0
      tau=xbjrk(1)*xbjrk(2)
      shat=tau*stot
      sqrtshat=sqrt(shat)
      if (j_fks.gt.nincoming) then
         if(resonance_recoil)then
            call invert_momenta_resonance_final(p_cms,
     $           resonance_members,i_fks,j_fks,m_j_fks,
     $           pb(:,1:nexternal-1),xx,xjac0,xpswgt0,
     $           resonance_momentum,resonance_mass2,pass)
            if(.not.pass)xjac0=-146d0
         elseif (m_j_fks.eq.0d0) then
            call generate_momenta_massless_final_inverse(p_cms,xi_i_fks
     $           ,y_ij_fks,phi_i_fks,pb,xx,xjac0,xpswgt0
     $           ,shat,sqrtshat,i_fks,j_fks)
         else
            call generate_momenta_massive_final_inverse(p_cms,xi_i_fks
     $           ,y_ij_fks,phi_i_fks,pb,xx,xjac0,xpswgt0
     $           ,shat,sqrtshat,i_fks,j_fks,m_j_fks)
         endif
         tau_born=tau
         ycm_born=ycm
         xbjrk_born(1:2)=xbjrk(1:2)
      else
         call generate_momenta_initial_inverse(p_lab,xi_i_fks,y_ij_fks
     $        ,phi_i_fks,pb,xx,xjac0,xpswgt0,shat,sqrtshat
     $        ,i_fks,j_fks,stot,tau,ycm,xbjrk,tau_born,ycm_born
     $        ,xbjrk_born,y_lab_to_cms)
      endif
      end subroutine invert_fks_radiation


      subroutine capture_fks_phase_space(point)
      use fks_phase_space_data,only: resonance_momentum,resonance_mass2,resonance_recoil,
     $     resonance_members,initial_recoil_leg,p_ev,p_born_coll,p_born_norad,xi_ev => xi_i_fks_ev,
     $     y_ev => y_ij_fks_ev,p_i_ev => p_i_fks_ev,p_i_cnt => p_i_fks_cnt,xi_cnt => xi_i_fks_cnt,
     $     xi_hat_ev => xi_i_hat_ev,xi_hat_cnt => xi_i_hat_cnt,xbjrk_ev,xbjrk_cnt,sqrtshat_ev,shat_ev,
     $     sqrtshat_cnt,shat_cnt,tau_ev,ycm_ev,tau_cnt,ycm_cnt,xi_max_ev => xiimax_ev,
     $     xi_max_cnt => xiimax_cnt,xi_norm_ev => xinorm_ev,xi_norm_cnt => xinorm_cnt,spin_phase => xij_aor,
     $     veckn_ev,veckbarn_ev,xp0jfks,solution_sign => isolsign,no_counterevents => nocntevents,
     $     tau_Born_lower_bound,tau_lower_bound_resonance,tau_lower_bound,ybst_til_tolab,ybst_til_tocm,
     $     sqrtshat,shat
c Attach the active generated data to the caller-owned Born input,
c returned momenta and weighted measure. Capture endpoint metadata even
c when an endpoint is invalid: in particular ycm_cnt(0) defines the
c generation frame also for massive solutions with no counterevents.
      implicit none
      integer slot,i_fks,j_fks,nFKSprocess,this_config
      common/fks_indices/i_fks,j_fks
      common/c_nFKSprocess/nFKSprocess
      common/to_mconfigs/this_config
      logical event_projection
      common/to_use_evpr/event_projection
      double precision omx(2)
      common/to_ee_omx1/omx
      type(fks_phase_space_point),intent(inout) :: point

      point%event%p=p_ev
      point%event%jacobian=point%weight
      if(p_ev(0,1).le.0d0.and.point%weight.gt.0d0)
     $     point%event%jacobian=-1d0
      point%event%tau=tau_ev
      point%event%ycm=ycm_ev
      point%event%shat=shat_ev
      point%event%sqrtshat=sqrtshat_ev
      point%event%xbjrk=xbjrk_ev
      point%event%xi=xi_ev
      point%event%y=y_ev
      point%event%xi_hat=xi_hat_ev
      point%event%xi_max=xi_max_ev
      point%event%xi_norm=xi_norm_ev
      point%event%direction=p_i_ev
      point%event%valid=point%event%jacobian.gt.0d0.and.
     $     point%event%p(0,1).gt.0d0
      do slot=-2,2
         point%counterevent(slot)%p=p1_cnt(:,:,slot)
         point%counterevent(slot)%jacobian=jac_cnt(slot)
         point%counterevent(slot)%tau=tau_cnt(slot)
         point%counterevent(slot)%ycm=ycm_cnt(slot)
         point%counterevent(slot)%shat=shat_cnt(slot)
         point%counterevent(slot)%sqrtshat=sqrtshat_cnt(slot)
         point%counterevent(slot)%xbjrk=xbjrk_cnt(:,slot)
         point%counterevent(slot)%xi=xi_cnt(slot)
         point%counterevent(slot)%y=counter_y(slot)
         point%counterevent(slot)%xi_hat=xi_hat_cnt(slot)
         point%counterevent(slot)%xi_max=xi_max_cnt(slot)
         point%counterevent(slot)%xi_norm=xi_norm_cnt(slot)
         point%counterevent(slot)%direction=p_i_cnt(:,slot)
         point%counterevent(slot)%valid=jac_cnt(slot).gt.0d0.and.
     $        p1_cnt(0,1,slot).gt.0d0
      enddo
      point%valid=point%event%valid.or.
     $     any(point%counterevent%valid)
      point%p_born=p_born
      point%p_born_l=p_born_l
      point%p_born_ev=p_born_ev
      point%spin_phase=spin_phase
      point%radiation_energies=[veckn_ev,veckbarn_ev,xp0jfks]
      point%solution_sign=solution_sign
      point%no_counterevents=no_counterevents
      point%event_projection=event_projection
      point%bounds=[tau_Born_lower_bound,tau_lower_bound_resonance,tau_lower_bound]
      point%omx=omx
      point%active_cms=[ybst_til_tolab,ybst_til_tocm,sqrtshat,shat]
      point%resonance_momentum=resonance_momentum
      point%resonance_mass2=resonance_mass2
      point%resonance_recoil=resonance_recoil
      point%resonance_members=resonance_members
      point%initial_recoil_leg=initial_recoil_leg
      point%p_born_coll=0d0
      point%p_born_coll(0,1)=-99d0
      point%p_born_norad=0d0
      point%p_born_norad(0,1)=-99d0
      point%has_collinear_born=.false.
      point%has_reduced_born=.false.
      if(.not.event_projection)then
         point%has_collinear_born=p_born_coll(0,1).gt.0d0
         point%has_reduced_born=p_born_norad(0,1).gt.0d0
         if(point%has_collinear_born)point%p_born_coll=p_born_coll
         if(point%has_reduced_born)point%p_born_norad=p_born_norad
      endif
      point%sector=nFKSprocess
      point%i_fks=i_fks
      point%j_fks=j_fks
      point%config=this_config
      end subroutine capture_fks_phase_space


      subroutine restore_fks_phase_space(point)
      use fks_phase_space_data,only: resonance_momentum,resonance_mass2,resonance_recoil,
     $     resonance_members,initial_recoil_leg,p_ev,p_born_coll,p_born_norad,xi_ev => xi_i_fks_ev,
     $     y_ev => y_ij_fks_ev,p_i_ev => p_i_fks_ev,p_i_cnt => p_i_fks_cnt,xi_cnt => xi_i_fks_cnt,
     $     xi_hat_ev => xi_i_hat_ev,xi_hat_cnt => xi_i_hat_cnt,xbjrk_ev,xbjrk_cnt,sqrtshat_ev,shat_ev,
     $     sqrtshat_cnt,shat_cnt,tau_ev,ycm_ev,tau_cnt,ycm_cnt,xi_max_ev => xiimax_ev,
     $     xi_max_cnt => xiimax_cnt,xi_norm_ev => xinorm_ev,xi_norm_cnt => xinorm_cnt,spin_phase => xij_aor,
     $     veckn_ev,veckbarn_ev,xp0jfks,solution_sign => isolsign,no_counterevents => nocntevents,
     $     tau_Born_lower_bound,tau_lower_bound_resonance,tau_lower_bound,ybst_til_tolab,ybst_til_tocm,
     $     sqrtshat,shat
c Restore generated kinematics after another history used the module.
c The caller must first reactivate the point's sector and channel; this
c routine does not change process, shower, colour, PDF or coupling state.
      implicit none
      integer slot,i_fks,j_fks,nFKSprocess,this_config
      common/fks_indices/i_fks,j_fks
      common/c_nFKSprocess/nFKSprocess
      common/to_mconfigs/this_config
      logical event_projection
      common/to_use_evpr/event_projection
      double precision omx(2)
      common/to_ee_omx1/omx
      type(fks_phase_space_point),intent(in) :: point

      if(point%sector.ne.nFKSprocess.or.point%i_fks.ne.i_fks.or.
     $     point%j_fks.ne.j_fks.or.point%config.ne.this_config)then
         write(*,*) 'FKS phase-space restore has a different context',
     $        point%sector,nFKSprocess,point%i_fks,i_fks,
     $        point%j_fks,j_fks,point%config,this_config
         stop 1
      endif
      p_ev=point%event%p
      tau_ev=point%event%tau
      ycm_ev=point%event%ycm
      shat_ev=point%event%shat
      sqrtshat_ev=point%event%sqrtshat
      xbjrk_ev=point%event%xbjrk
      xi_ev=point%event%xi
      y_ev=point%event%y
      xi_hat_ev=point%event%xi_hat
      xi_max_ev=point%event%xi_max
      xi_norm_ev=point%event%xi_norm
      p_i_ev=point%event%direction
      do slot=-2,2
         p1_cnt(:,:,slot)=point%counterevent(slot)%p
         jac_cnt(slot)=point%counterevent(slot)%jacobian
         tau_cnt(slot)=point%counterevent(slot)%tau
         ycm_cnt(slot)=point%counterevent(slot)%ycm
         shat_cnt(slot)=point%counterevent(slot)%shat
         sqrtshat_cnt(slot)=point%counterevent(slot)%sqrtshat
         xbjrk_cnt(:,slot)=point%counterevent(slot)%xbjrk
         xi_cnt(slot)=point%counterevent(slot)%xi
         counter_y(slot)=point%counterevent(slot)%y
         xi_hat_cnt(slot)=point%counterevent(slot)%xi_hat
         xi_max_cnt(slot)=point%counterevent(slot)%xi_max
         xi_norm_cnt(slot)=point%counterevent(slot)%xi_norm
         p_i_cnt(:,slot)=point%counterevent(slot)%direction
      enddo
      p_born=point%p_born
      p_born_l=point%p_born_l
      p_born_ev=point%p_born_ev
      spin_phase=point%spin_phase
      veckn_ev=point%radiation_energies(1)
      veckbarn_ev=point%radiation_energies(2)
      xp0jfks=point%radiation_energies(3)
      solution_sign=point%solution_sign
      no_counterevents=point%no_counterevents
      event_projection=point%event_projection
      tau_Born_lower_bound=point%bounds(1)
      tau_lower_bound_resonance=point%bounds(2)
      tau_lower_bound=point%bounds(3)
      omx=point%omx
      ybst_til_tolab=point%active_cms(1)
      ybst_til_tocm=point%active_cms(2)
      sqrtshat=point%active_cms(3)
      shat=point%active_cms(4)
      resonance_momentum=point%resonance_momentum
      resonance_mass2=point%resonance_mass2
      resonance_recoil=point%resonance_recoil
      resonance_members=point%resonance_members
      initial_recoil_leg=point%initial_recoil_leg
      p_born_coll=point%p_born_coll
      p_born_norad=point%p_born_norad
      end subroutine restore_fks_phase_space


      subroutine record_fks_phase_space(point,born,wgt,p,p_lab,p_cms)
c Keep the reusable Born input separate from the complete generated
c result. All caller weights have already been applied to Jacobians.
      implicit none
      type(fks_phase_space_point),intent(out) :: point
      type(fks_born_point),intent(in) :: born
      double precision,intent(in) :: wgt,p(0:3,nexternal),
     $     p_lab(0:3,nexternal),p_cms(0:3,nexternal)
      point=fks_phase_space_point()
      point%born=born
      point%p=p
      point%p_lab=p_lab
      point%p_cms=p_cms
      point%weight=wgt
      call capture_fks_phase_space(point)
      end subroutine record_fks_phase_space
      end module fks_phase_space
