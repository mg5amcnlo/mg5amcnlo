!ZW module to handle dynamic allocation of all the vectorisation
module vectorize
   implicit none
   include 'nexternal.inc'
   include 'orders.inc'
   include 'nFKSconfigs.inc'
   include 'genps.inc'

   ! Declare allocatable arrays
   !ZW: amp arrays
   double precision, allocatable :: AMP_SPLIT_STORE_R(:,:)
   double precision, allocatable :: AMP_SPLIT_STORE_B(:,:)
   double complex, allocatable :: AMP_SPLIT_STORE_CNT(:,:,:,:)
   double precision, allocatable :: AMP_SPLIT_STORE_BSF(:,:,:,:)

   !ZW: parton_cms_stuff
   double precision, allocatable :: ybst_til_tolab(:),ybst_til_tocm(:),sqrtshat(:),shat(:)

   !ZW: p_born
   double precision, allocatable :: p_born(:,:,:)

   !ZW: FKSvariables
   double precision, allocatable :: xi_i_fks_ev(:),y_ij_fks_ev(:)
   double precision, allocatable :: p_i_fks_ev(:,:),p_i_fks_cnt(:,:,:)

   !ZW: calculatedBorn
   logical, allocatable :: calculatedBorn(:)

   !ZW: c_FxFx_scales
   integer, allocatable :: nFxFx_ren_scales(:)
   double precision, allocatable :: FxFx_ren_scales(:,:), FxFx_fac_scale(:,:)

   !ZW: counterevents
   double precision, allocatable :: p1_cnt(:,:,:,:), wgt_cnt(:,:), pswgt_cnt(:,:), jac_cnt(:,:)

   !ZW: cxiifkscnt
   double precision, allocatable :: xi_i_fks_cnt(:,:)

   !ZW: cgenps_fks
   double precision, allocatable :: veckn_ev(:),veckbarn_ev(:),xp0jfks(:)

   !ZW: cbjorkenx
   double precision, allocatable :: xbjrk_ev(:,:),xbjrk_cnt(:,:,:)

   !ZW: to_amps
   double precision, allocatable :: amp2(:,:),jamp2(:,:)

   !ZW: cxij_aor
   double complex, allocatable :: xij_aor(:)

   !ZW: cnbody
   logical, allocatable :: nbody(:)

   !ZW: cxiimaxcnt
   double precision, allocatable :: xi_i_max_cnt(:,:)

   !ZW: cxi_i_hat
   double precision, allocatable :: xi_i_hat_ev(:),xi_i_hat_cnt(:,:)

   !ZW: cxinormev
   double precision, allocatable :: xinorm_ev(:)

   !ZW: cximaxev
   double precision, allocatable :: xiimax_ev(:)

   !ZW: pborn_ev
   double precision, allocatable :: p_born_ev(:,:,:)

   !ZW: pborn_coll
   double precision, allocatable :: p_born_coll(:,:,:)

   !ZW: pborn_norad
   double precision, allocatable :: p_born_norad(:,:,:)

   !ZW: pev
   double precision, allocatable :: p_ev(:,:,:)

   !ZW: c_granny_res
   integer, allocatable :: igranny(:),iaunt(:)
   logical, allocatable :: granny_chain(:,:),granny_is_res(:),granny_chain_real_final(:,:)

   !ZW: cnocntevents
   logical, allocatable :: nocntevents(:)

   !ZW: parton_cms_ev
   double precision, allocatable :: sqrtshat_ev(:),shat_ev(:)

   !ZW: parton_cms_cnt
   double precision, allocatable :: sqrtshat_cnt(:,:),shat_cnt(:,:)

   !ZW: cbjrk12_ev
   double precision, allocatable :: tau_ev(:),ycm_ev(:)

   !ZW: cbjrk12_cnt
   double precision, allocatable :: tau_cnt(:,:),ycm_cnt(:,:)

   !ZW: to_amp_split_soft
   double precision, allocatable :: amp_split_soft(:,:)

   !ZW: c_vegas_x
   double precision, allocatable :: xvar(:,:)

   !ZW: c_rat_xi
   double precision, allocatable :: rat_xi_orig(:)

   !ZW: to_virtgranny
   double precision, allocatable :: granny_m2_red(:,:)

   !ZW: pborn_l
   double precision, allocatable :: p_born_l(:,:)

   !ZW: virtgranny_boost
   double precision, allocatable :: shybst(:),chybst(:),chybstmo(:)

   !ZW: cgrannyrange
   real*8, allocatable :: xmbemin2(:),xmbemax2(:)

   !ZW: coffset
   real*8, allocatable :: offset(:)

   !ZW: c_skip_only_event_phsp
   logical, allocatable :: only_event_phsp(:),skip_event_phsp(:)

   !ZW: write_granny_resonance
   integer, allocatable :: which_is_granny(:,:)
   logical, allocatable :: write_granny(:,:)

   !ZW: c_isolsign
   integer, allocatable :: isolsign(:)

   !ZW: conflictung BW stuff
   double precision, allocatable :: cBW_mass(:,:,:),cBW_width(:,:,:)
   integer, allocatable :: cBW_level_max(:),cBW(:,:),cBW_level(:,:)

!   !ZW: to_ee_omx1
!   double precision, allocatable :: omx_ee(:,:)

   !ZW: to_phase_space_s_channel
   double precision, allocatable :: s_mass(:,:)

   !ZW: Ellis-Sexton scale
   double precision, allocatable :: QES2(:)

contains
   ! Procedure to allocate arrays dynamically based on vec_size
   subroutine allocate_storage(vector_size)
       integer, intent(in) :: vector_size
       ! Allocate arrays with runtime size
       !ZW: amp arrays
       allocate(AMP_SPLIT_STORE_R(AMP_SPLIT_SIZE, vector_size))
       allocate(AMP_SPLIT_STORE_B(AMP_SPLIT_SIZE, vector_size))
       allocate(AMP_SPLIT_STORE_CNT(AMP_SPLIT_SIZE, 2, NSPLITORDERS, vector_size))
       allocate(AMP_SPLIT_STORE_BSF(AMP_SPLIT_SIZE, 5, 5, vector_size))
       !ZW: parton_cms_stuff
       allocate(ybst_til_tolab(vector_size))
       allocate(ybst_til_tocm(vector_size))
       allocate(sqrtshat(vector_size))
       allocate(shat(vector_size))
       !ZW: p_born
       allocate(p_born(0:3,nexternal-1,vector_size))
       !ZW: FKSvariables
       allocate(xi_i_fks_ev(vector_size))
       allocate(y_ij_fks_ev(vector_size))
       allocate(p_i_fks_ev(0:3,vector_size))
       allocate(p_i_fks_cnt(0:3,-2:2,vector_size))
       !ZW: calculatedBorn
       allocate(calculatedBorn(vector_size))
       !ZW: c_FxFx_scales
       allocate(nFxFx_ren_scales(vector_size))
       allocate(FxFx_ren_scales(0:nexternal,vector_size))
       allocate(FxFx_fac_scale(2,vector_size))
       !ZW: counterevents
       allocate(p1_cnt(0:3,nexternal,-2:2,vector_size))
       allocate(wgt_cnt(-2:2,vector_size))
       allocate(pswgt_cnt(-2:2,vector_size))
       allocate(jac_cnt(-2:2,vector_size))
       !ZW: cxiifkscnt
       allocate(xi_i_fks_cnt(-2:2,vector_size))
       !ZW: cgenps_fks
       allocate(veckn_ev(vector_size))
       allocate(veckbarn_ev(vector_size))
       allocate(xp0jfks(vector_size))
       !ZW: cbjorkenx
       allocate(xbjrk_ev(2,vector_size))
       allocate(xbjrk_cnt(2,-2:2,vector_size))
       !ZW: to_amps
       allocate(amp2(ngraphs,vector_size)) !ZW: NGRAPHS MUST BE INCLUDED SOMEWHERE
       allocate(jamp2(0:ncolor,vector_size)) !ZW: NCOLOR MUST BE INCLUDED SOMEWHERE
       !ZW: cxij_aor
       allocate(xij_aor(vector_size))
       !ZW: cnbody
       allocate(nbody(vector_size))
       !ZW: cxiimaxcnt
       allocate(xi_i_max_cnt(-2:2,vector_size))
      !ZW: cxi_i_hat
       allocate(xi_i_hat_ev(vector_size))
       allocate(xi_i_hat_cnt(-2:2,vector_size))
      !ZW: cxinormev
       allocate(xi_norm_ev(vector_size))
   !ZW: cximaxev
       allocate(xi_i_max_ev(vector_size))
   !ZW: pborn_ev
       allocate(p_born_ev(0:3,nexternal-1,vector_size))
   !ZW: pborn_coll   
       allocate(p_born_coll(0:3,nexternal-1,vector_size))
   !ZW: pborn_norad
       allocate(p_born_norad(0:3,nexternal-1,vector_size))
   !ZW: pev
       allocate(p_ev(0:3,nexternal,vector_size))
   !ZW: c_granny_res
         allocate(igranny(vector_size))
         allocate(iaunt(vector_size))
         allocate(granny_chain(-nexternal:nexternal,vector_size))
         allocate(granny_is_res(vector_size))
         allocate(granny_chain_real_final(-nexternal:nexternal,vector_size))
   !ZW: cnocntevents
         allocate(nocntevents(vector_size))
   !ZW: parton_cms_ev
         allocate(sqrtshat_ev(vector_size))
         allocate(shat_ev(vector_size))
         !ZW: parton_cms_cnt
         allocate(sqrtshat_cnt(-2:2,vector_size))
         allocate(shat_cnt(-2:2,vector_size))
         !ZW: cbjrk12_ev
         allocate(tau_ev(vector_size))
         allocate(ycm_ev(vector_size))
         !ZW: cbrjk12_cnt
         allocate(tau_cnt(-2:2,vector_size))
         allocate(ycm_cnt(-2:2,vector_size))
         !ZW: to_amp_split_soft
         allocate(amp_split_soft(AMP_SPLIT_SIZE,vector_size))
         !ZW: c_vegas_x
         allocate(xvar(99,vector_size))
         !ZW: c_rat_xi
         allocate(rat_xi_orig(vector_size))
         !ZW: to_virtgranny
         allocate(granny_m2_red(-1:1,vector_size))
         !ZW: pborn_l
         allocate(p_born_l(0:3,nexternal-1,vector_size))
         !ZW: virtgranny_boost
         allocate(shybst(vector_size))
         allocate(chybst(vector_size))
         allocate(chybstmo(vector_size))
         !ZW: cgrannyrange
         allocate(xmbemin2(vector_size))
         allocate(xmbemax2(vector_size))
         !ZW: coffset
         allocate(offset(vector_size))
         !ZW: c_skip_only_event_phsp
         allocate(only_event_phsp(vector_size))
         allocate(skip_event_phsp(vector_size))
         !ZW: write_granny_resonance
         allocate(which_is_granny(fks_configs,vector_size))
         allocate(write_granny(fks_configs,vector_size))
         !ZW: c_isolsign
         allocate(isolsign(vector_size))
         !ZW: conflicting BW stuff
         allocate(cBW_mass(-1:1,-nexternal:-1,vector_size))
         allocate(cBW_width(-1:1,-nexternal:-1,vector_size))
         allocate(cBW_level_max(vector_size))
         allocate(cBW(-nexternal:-1,vector_size))
         allocate(cBW_level(-nexternal:-1,vector_size))
!         !ZW: to_ee_omx1
!         allocate(omx_ee(vector_size))
         !ZW: to_phase_space_s_channel
         allocate(s_mass(-nexternal:nexternal,vector_size))
         !ZW: Ellis-Sexton scale
         allocate(QES2(vector_size))
   end subroutine allocate_storage
   ! Add other module procedures here if necessary
end module vectorize
!ZW