c Native Born projection and regeneration at a fixed real point.
c External routine interfaces and COMMON layouts are shared with
c genps_fks.f; keep coordinate and counterevent conventions consistent.

      subroutine generate_native_momenta(p_input,p,p_lab,p_cms,
     $     jac,pass)
! Project a fixed real point onto its native Born and invert ONLY the
! three radiation coordinates. The Born sampling measure is common to
! the real and counterevents and cancels in repartition_MC_H. Generate
! their radiation/flux measures with a unit Born measure instead.
      use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
      use kinematics_module, only: boost_n1_to_its_cms,boost_n1_to_lab
      use mc_native_context, only: native_mapping
      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
      include 'run.inc'
      double precision p_input(0:3,nexternal),p(0:3,nexternal),
     $     p_lab(0:3,nexternal),p_cms(0:3,nexternal),jac
      logical pass
      double precision pmass(nexternal)
      common/to_mass/pmass
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      double precision p_born(0:3,nexternal-1),
     $     p_born_l(0:3,nexternal-1),p_born_ev(0:3,nexternal-1)
      common/pborn/p_born
      common/pborn_l/p_born_l
      common/pborn_ev/p_born_ev
      double precision bounds(3),omx_ee(2)
      common/ctau_lower_bound/bounds
      common/to_ee_omx1/omx_ee
      logical nbody,use_evpr
      common/cnbody/nbody
      common/to_use_evpr/use_evpr
      double precision kn,knbar,kn0
      common/cgenps_fks/kn,knbar,kn0
      integer this_config
      common/to_mconfigs/this_config
      double precision p1_cnt(0:3,nexternal,-2:2),wgt_cnt(-2:2),
     $     pswgt_cnt(-2:2),jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
      double precision x(99),pb(0:3,-max_branch:nexternal-1),
     $     m(-max_branch:max_particles),m_born(nexternal-1),
     $     stot,tau_born,ycm_born,xbjrk_born(2),shat_born,
     $     sqrtshat_born,ycmhat,xjac0,xpswgt0,dummy,
     $     bounds_save(3),omx_save(2)
      logical nbody_save
      integer i

      pass=.false.
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
      if (.not.all(ieee_is_finite(p_input)).or.
     $     any(p_input(0,1:2).le.0d0))return
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
      bounds_save=bounds
      omx_save=omx_ee
      nbody_save=nbody
      bounds=sum(m_born(nincoming+1:nexternal-1))**2/stot
      omx_ee=0d0
      nbody=.false.
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
      p_born=pb(:,1:nexternal-1)
      p_born_l=p_born
      p_born_ev=p_born
      shat_born=tau_born*stot
      sqrtshat_born=sqrt(shat_born)
      ycmhat=0d0
      if(tau_born.lt.1d0)ycmhat=ycm_born/(-0.5d0*log(tau_born))
! No Born topology is sampled. A valid native index is still needed by
! clustering; the outer replay restores its integration-channel index.
      this_config=1
      use_evpr=.true.
! Inactive counterevents must not retain another history's momenta.
      kn=0d0
      knbar=0d0
      kn0=0d0
      p1_cnt=0d0
      p1_cnt(0,1,:)=-1d0
      wgt_cnt=-1d99
      pswgt_cnt=-1d99
      jac_cnt=-1d0
      xjac0=1d0
      xpswgt0=1d0
      m=0d0
      call generate_FKS_kinematics(x,3,xjac0,xpswgt0,stot,
     $     shat_born,sqrtshat_born,tau_born,ycm_born,ycmhat,
     $     xbjrk_born,m,m_born,jac,p,pass)
      pass=ieee_is_finite(jac).and.jac.gt.0d0.and.p(0,1).gt.0d0
      if(.not.pass)goto 900
      call boost_n1_to_its_cms(p,p_cms,dummy)
      call boost_n1_to_lab(p,p_lab,-ycm_born)
      pass=all(ieee_is_finite(p_lab)).and.
     $     maxval(abs(p_lab-p_input)).le.
     $     1d-7*max(1d0,maxval(abs(p_input)))
 900  continue
      bounds=bounds_save
      omx_ee=omx_save
      nbody=nbody_save
      if(.not.pass)jac=-1d0
      end


      subroutine invert_fks_radiation(xx,xjac0,xpswgt0,
     $     stot,tau_born,ycm_born,xbjrk_born,p_lab,pb)
! Input momenta are in the symmetric hadron frame used by generate_momenta.
! No Born integration-channel coordinates or Jacobians are recovered.
      use kinematics_module, only: boost_n1_to_its_cms,
     $     boost_n1_to_lab,get_xi_from_p,get_yij_from_p,get_phi_from_p
      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
      include 'resonance_recoil.inc'
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
      y_ij_fks=get_yij_from_p(i_fks,j_fks,p_cms)
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
      end
