      module fks_phase_space_data
c Active phase-space data shared by generation, subtraction and matching.
c This leaf module depends only on the generated particle dimensions.
c Ordinary explicit-shape arrays preserve the sequence association used
c by the legacy numerical kernels; there are no pointer storage aliases.
c Caller-owned fks_phase_space_point objects capture/restore this state.
c Run/process controls and the PDF-library endpoint bridge remain with
c their existing owners. This module still describes one active point.
      implicit none
      include 'nexternal.inc'
      private
      public p_born
      public p_born_l
      public p_born_ev
      public p_born_coll
      public p_born_norad
      public p_ev
      public p1_cnt,jac_cnt
      public xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt
      public xi_i_fks_cnt
      public xi_i_hat_ev,xi_i_hat_cnt
      public xbjrk_ev,xbjrk_cnt
      public sqrtshat_ev,shat_ev
      public sqrtshat_cnt,shat_cnt
      public tau_ev,ycm_ev
      public tau_cnt,ycm_cnt
      public xiimax_ev
      public xiimax_cnt
      public xinorm_ev
      public xinorm_cnt
      public xij_aor
      public veckn_ev,veckbarn_ev,xp0jfks
      public isolsign
      public nocntevents
      public ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      public tau_Born_lower_bound,tau_lower_bound_resonance,tau_lower_bound
      public resonance_momentum,resonance_mass2,resonance_recoil,resonance_members
      public initial_recoil_leg

c Former /pborn/ fields.
      double precision :: p_born(0:3,nexternal-1)=0d0

c Former /pborn_l/ fields.
      double precision :: p_born_l(0:3,nexternal-1)=0d0

c Former /pborn_ev/ fields.
      double precision :: p_born_ev(0:3,nexternal-1)=0d0

c Former /pborn_coll/ fields.
      double precision :: p_born_coll(0:3,nexternal-1)=0d0

c Former /pborn_norad/ fields.
      double precision :: p_born_norad(0:3,nexternal-1)=0d0

c Former /pev/ fields.
      double precision :: p_ev(0:3,nexternal)=0d0

c Former /counterevnts/ fields.
      double precision :: p1_cnt(0:3,nexternal,-2:2)=0d0
      double precision :: jac_cnt(-2:2)=-1d0

c Former /fksvariables/ fields.
      double precision :: xi_i_fks_ev=0d0
      double precision :: y_ij_fks_ev=0d0
      double precision :: p_i_fks_ev(0:3)=0d0
      double precision :: p_i_fks_cnt(0:3,-2:2)=0d0

c Former /cxiifkscnt/ fields.
      double precision :: xi_i_fks_cnt(-2:2)=0d0

c Former /cxi_i_hat/ fields.
      double precision :: xi_i_hat_ev=0d0
      double precision :: xi_i_hat_cnt(-2:2)=0d0

c Former /cbjorkenx/ fields.
      double precision :: xbjrk_ev(2)=0d0
      double precision :: xbjrk_cnt(2,-2:2)=0d0

c Former /parton_cms_ev/ fields.
      double precision :: sqrtshat_ev=0d0
      double precision :: shat_ev=0d0

c Former /parton_cms_cnt/ fields.
      double precision :: sqrtshat_cnt(-2:2)=0d0
      double precision :: shat_cnt(-2:2)=0d0

c Former /cbjrk12_ev/ fields.
      double precision :: tau_ev=0d0
      double precision :: ycm_ev=0d0

c Former /cbjrk12_cnt/ fields.
      double precision :: tau_cnt(-2:2)=0d0
      double precision :: ycm_cnt(-2:2)=0d0

c Former /cxiimaxev/ fields.
      double precision :: xiimax_ev=-1d0

c Former /cxiimaxcnt/ fields.
      double precision :: xiimax_cnt(-2:2)=-1d0

c Former /cxinormev/ fields.
      double precision :: xinorm_ev=0d0

c Former /cxinormcnt/ fields.
      double precision :: xinorm_cnt(-2:2)=0d0

c Former /cxij_aor/ fields.
      double complex :: xij_aor=(0d0,0d0)

c Former /cgenps_fks/ fields.
      double precision :: veckn_ev=0d0
      double precision :: veckbarn_ev=0d0
      double precision :: xp0jfks=0d0

c Former /c_isolsign/ fields.
      integer :: isolsign=0

c Former /cnocntevents/ fields.
      logical :: nocntevents=.true.

c Former /parton_cms_stuff/ fields.
      double precision :: ybst_til_tolab=0d0
      double precision :: ybst_til_tocm=0d0
      double precision :: sqrtshat=0d0
      double precision :: shat=0d0

c Former /ctau_lower_bound/ fields.
      double precision :: tau_Born_lower_bound=0d0
      double precision :: tau_lower_bound_resonance=0d0
      double precision :: tau_lower_bound=0d0

c Former /c_resonance_recoil/ fields.
      double precision :: resonance_momentum(0:3)=0d0
      double precision :: resonance_mass2=0d0
      logical :: resonance_recoil=.false.
      logical :: resonance_members(nexternal)=.false.

c Former /c_initial_recoil/ fields.
      integer :: initial_recoil_leg=0
      end module fks_phase_space_data
