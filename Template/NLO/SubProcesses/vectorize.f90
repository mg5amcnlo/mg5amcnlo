!ZW module to handle dynamic allocation of all the vectorisation
module pborn
    use mod_nexternal
    implicit none
!   include 'nexternal_mod.inc'
   double precision, allocatable :: p_born(:,:,:)
 contains
   subroutine allocate_pborn(vector_size)
       integer, intent(in) :: vector_size
       allocate(p_born(0:3,nexternal_mod-1,vector_size))
   end subroutine allocate_pborn
   subroutine reset_pborn
       p_born(:,:,:) = 0.d0
   end subroutine reset_pborn
   subroutine deallocate_pborn
       if (allocated(p_born)) deallocate(p_born)
   end subroutine deallocate_pborn
end module pborn
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module pborn_ev
    use mod_nexternal
    implicit none
   !include 'nexternal_mod.inc'
   double precision, allocatable :: p_born_ev(:,:,:)
 contains
   subroutine allocate_pborn_ev(vector_size)
       integer, intent(in) :: vector_size
       allocate(p_born_ev(0:3,nexternal_mod-1,vector_size))
   end subroutine allocate_pborn_ev
   subroutine reset_pborn_ev
       p_born_ev(:,:,:) = 0.d0
   end subroutine reset_pborn_ev
   subroutine deallocate_pborn_ev
       if (allocated(p_born_ev)) deallocate(p_born_ev)
   end subroutine deallocate_pborn_ev
end module pborn_ev
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module pborn_l
    use mod_nexternal
    implicit none
   !include 'nexternal_mod.inc'
   double precision, allocatable :: p_born_l(:,:,:)
 contains
   subroutine allocate_pborn_l(vector_size)
       integer, intent(in) :: vector_size
       allocate(p_born_l(0:3,nexternal_mod-1,vector_size))
   end subroutine allocate_pborn_l
   subroutine reset_pborn_l
       p_born_l(:,:,:) = 0.d0
   end subroutine reset_pborn_l
   subroutine deallocate_pborn_l
       if (allocated(p_born_l)) deallocate(p_born_l)
   end subroutine deallocate_pborn_l
end module pborn_l
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module pborn_coll
    use mod_nexternal
    implicit none
   !include 'nexternal_mod.inc'
   double precision, allocatable :: p_born_coll(:,:,:)
 contains
   subroutine allocate_pborn_coll(vector_size)
       integer, intent(in) :: vector_size
       allocate(p_born_coll(0:3,nexternal_mod-1,vector_size))
   end subroutine allocate_pborn_coll
    subroutine reset_pborn_coll
        p_born_coll(:,:,:) = 0.d0
    end subroutine reset_pborn_coll
   subroutine deallocate_pborn_coll
       if (allocated(p_born_coll)) deallocate(p_born_coll)
   end subroutine deallocate_pborn_coll
end module pborn_coll
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module pborn_norad
    use mod_nexternal
    implicit none
   !include 'nexternal_mod.inc'
   double precision, allocatable :: p_born_norad(:,:,:)
 contains
   subroutine allocate_pborn_norad(vector_size)
       integer, intent(in) :: vector_size
       allocate(p_born_norad(0:3,nexternal_mod-1,vector_size))
   end subroutine allocate_pborn_norad
    subroutine reset_pborn_norad
        p_born_norad(:,:,:) = 0.d0
    end subroutine reset_pborn_norad
   subroutine deallocate_pborn_norad
       if (allocated(p_born_norad)) deallocate(p_born_norad)
   end subroutine deallocate_pborn_norad
end module pborn_norad
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module factor_nbody
    implicit none
   double precision, allocatable :: f_b(:), f_nb(:)
 contains
   subroutine allocate_factor_nbody(vector_size)
       integer, intent(in) :: vector_size
       allocate(f_b(vector_size))
       allocate(f_nb(vector_size))
   end subroutine allocate_factor_nbody
   subroutine deallocate_factor_nbody
       if (allocated(f_b)) deallocate(f_b)
       if (allocated(f_nb)) deallocate(f_nb)
   end subroutine deallocate_factor_nbody
end module factor_nbody
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module factor_n1body
    implicit none
   double precision, allocatable :: f_r(:),f_s(:),f_c(:),f_dc(:),f_sc(:),f_dsc(:,:)
 contains
   subroutine allocate_factor_n1body(vector_size)
       integer, intent(in) :: vector_size
       allocate(f_r(vector_size))
       allocate(f_s(vector_size))
       allocate(f_c(vector_size))
       allocate(f_dc(vector_size))
       allocate(f_sc(vector_size))
       allocate(f_dsc(4,vector_size))
   end subroutine allocate_factor_n1body
   subroutine deallocate_factor_n1body
       if (allocated(f_r)) deallocate(f_r)
       if (allocated(f_s)) deallocate(f_s)
       if (allocated(f_c)) deallocate(f_c)
       if (allocated(f_dc)) deallocate(f_dc)
       if (allocated(f_sc)) deallocate(f_sc)
       if (allocated(f_dsc)) deallocate(f_dsc)
   end subroutine deallocate_factor_n1body
end module factor_n1body
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module factor_n1body_nlops
    implicit none
   double precision, allocatable :: f_s_MC_S(:),f_s_MC_H(:),f_c_MC_S(:),f_c_MC_H(:),f_sc_MC_S(:),f_sc_MC_H(:),f_MC_S(:),f_MC_H(:)
 contains
   subroutine allocate_factor_n1body_nlops(vector_size)
       integer, intent(in) :: vector_size
       allocate(f_s_MC_S(vector_size))
       allocate(f_s_MC_H(vector_size))
       allocate(f_c_MC_S(vector_size))
       allocate(f_c_MC_H(vector_size))
       allocate(f_sc_MC_S(vector_size))
       allocate(f_sc_MC_H(vector_size))
       allocate(f_MC_S(vector_size))
       allocate(f_MC_H(vector_size))
   end subroutine allocate_factor_n1body_nlops
   subroutine deallocate_factor_n1body_nlops
       if (allocated(f_s_MC_S)) deallocate(f_s_MC_S)
       if (allocated(f_s_MC_H)) deallocate(f_s_MC_H)
       if (allocated(f_c_MC_S)) deallocate(f_c_MC_S)
       if (allocated(f_c_MC_H)) deallocate(f_c_MC_H)
       if (allocated(f_sc_MC_S)) deallocate(f_sc_MC_S)
       if (allocated(f_sc_MC_H)) deallocate(f_sc_MC_H)
       if (allocated(f_MC_S)) deallocate(f_MC_S)
       if (allocated(f_MC_H)) deallocate(f_MC_H)
   end subroutine deallocate_factor_n1body_nlops
end module factor_n1body_nlops
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module to_amps_born
  use mod_genps
  use mod_orders
  implicit none
! include 'genps.inc'
 double precision, allocatable :: amp2b(:,:),jamp2b(:,:,:)
contains
 subroutine allocate_to_amps_born(vector_size)
     integer, intent(in) :: vector_size
     integer index
     allocate(amp2b(ngraphs_mod,vector_size)) !ZW: NGRAPHS MUST BE INCLUDED SOMEWHERE
     allocate(jamp2b(0:ncolor_mod,0:nampso_mod,vector_size)) !ZW: NCOLOR MUST BE INCLUDED SOMEWHERE
     do index=1,vector_size
        jamp2b(0,0,index) = ncolor_mod
     end do
 end subroutine allocate_to_amps_born
 subroutine reset_to_amps_born(vector_size)
     integer, intent(in) :: vector_size
     integer index
     amp2b(:,:) = 0.d0
     jamp2b(:,:,:) = 0.d0
     do index=1,vector_size
        jamp2b(0,0,index) = ncolor_mod
     end do
 end subroutine reset_to_amps_born
 subroutine deallocate_to_amps_born
      if (allocated(amp2b)) deallocate(amp2b)
      if (allocated(jamp2b)) deallocate(jamp2b)
 end subroutine deallocate_to_amps_born
end module to_amps_born
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module to_amps
  use mod_genps 
    implicit none
!   include 'genps.inc'
   double precision, allocatable :: amp2(:,:),jamp2(:,:)
 contains
   subroutine allocate_to_amps(vector_size)
       integer, intent(in) :: vector_size
       allocate(amp2(ngraphs_mod,vector_size)) !ZW: NGRAPHS MUST BE INCLUDED SOMEWHERE
       allocate(jamp2(0:ncolor_mod,vector_size)) !ZW: NCOLOR MUST BE INCLUDED SOMEWHERE
   end subroutine allocate_to_amps
    subroutine reset_to_amps
        amp2(:,:) = 0.d0
        jamp2(:,:) = 0.d0
    end subroutine reset_to_amps
   subroutine deallocate_to_amps
        if (allocated(amp2)) deallocate(amp2)
        if (allocated(jamp2)) deallocate(jamp2)
   end subroutine deallocate_to_amps
end module to_amps
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module counterevnts
    use mod_nexternal
    implicit none
   !include 'nexternal_mod.inc'
   double precision, allocatable :: p1_cnt(:,:,:,:), wgt_cnt(:,:), pswgt_cnt(:,:), jac_cnt(:,:)
 contains
   subroutine allocate_counterevnts(vector_size)
       integer, intent(in) :: vector_size
       allocate(p1_cnt(0:3,nexternal_mod,-2:2,vector_size))
       allocate(wgt_cnt(-2:2,vector_size))
       allocate(pswgt_cnt(-2:2,vector_size))
       allocate(jac_cnt(-2:2,vector_size))
   end subroutine allocate_counterevnts
   subroutine reset_counterevnts
       p1_cnt(:,:,:,:) = 0.d0
       wgt_cnt(:,:) = 0.d0
       pswgt_cnt(:,:) = 0.d0
       jac_cnt(:,:) = 0.d0
   end subroutine reset_counterevnts
   subroutine deallocate_counterevnts
        if (allocated(p1_cnt)) deallocate(p1_cnt)
        if (allocated(wgt_cnt)) deallocate(wgt_cnt)
        if (allocated(pswgt_cnt)) deallocate(pswgt_cnt)
        if (allocated(jac_cnt)) deallocate(jac_cnt)
   end subroutine deallocate_counterevnts
end module counterevnts
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module write_granny_resonance
    use mod_nfksconfigs 
    implicit none
   integer, allocatable :: which_is_granny(:,:)
   logical, allocatable :: write_granny(:,:)
 contains
   subroutine allocate_write_granny_resonance(vector_size)
       integer, intent(in) :: vector_size
         allocate(which_is_granny(fks_configs_mod,vector_size))
         allocate(write_granny(fks_configs_mod,vector_size))
   end subroutine allocate_write_granny_resonance
   subroutine reset_write_granny_resonance
       write_granny(:,:) = .false.
   end subroutine reset_write_granny_resonance
   subroutine deallocate_write_granny_resonance
        if (allocated(which_is_granny)) deallocate(which_is_granny)
        if (allocated(write_granny)) deallocate(write_granny)
   end subroutine deallocate_write_granny_resonance
end module write_granny_resonance
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module parton_cms_stuff
    implicit none
   double precision, allocatable :: ybst_til_tolab(:),ybst_til_tocm(:),sqrtshat(:),shat(:)
 contains
   subroutine allocate_parton_cms_stuff(vector_size)
       integer, intent(in) :: vector_size
       allocate(ybst_til_tolab(vector_size))
       allocate(ybst_til_tocm(vector_size))
       allocate(sqrtshat(vector_size))
       allocate(shat(vector_size))
   end subroutine allocate_parton_cms_stuff
   subroutine reset_parton_cms_stuff
       ybst_til_tolab(:) = 0.d0 
       ybst_til_tocm(:) = 0.d0 
       sqrtshat(:) = 0.d0 
       shat(:) = 0.d0 
   end subroutine reset_parton_cms_stuff
   subroutine deallocate_parton_cms_stuff
        if (allocated(ybst_til_tolab)) deallocate(ybst_til_tolab)
        if (allocated(ybst_til_tocm)) deallocate(ybst_til_tocm)
        if (allocated(sqrtshat)) deallocate(sqrtshat)
        if (allocated(shat)) deallocate(shat)
   end subroutine deallocate_parton_cms_stuff
end module parton_cms_stuff
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module fksvariables
    implicit none
   double precision, allocatable :: xi_i_fks_ev(:),y_ij_fks_ev(:)
   double precision, allocatable :: p_i_fks_ev(:,:),p_i_fks_cnt(:,:,:)
 contains
   subroutine allocate_fksvariables(vector_size)
       integer, intent(in) :: vector_size
       allocate(xi_i_fks_ev(vector_size))
       allocate(y_ij_fks_ev(vector_size))
       allocate(p_i_fks_ev(0:3,vector_size))
       allocate(p_i_fks_cnt(0:3,-2:2,vector_size))
   end subroutine allocate_fksvariables
   subroutine reset_fksvariables
       xi_i_fks_ev(:) = 0.d0
       y_ij_fks_ev(:) = 0.d0
       p_i_fks_ev(:,:) = 0.d0
       p_i_fks_cnt(:,:,:) = 0.d0
   end subroutine reset_fksvariables
   subroutine deallocate_fksvariables
        if (allocated(xi_i_fks_ev)) deallocate(xi_i_fks_ev)
        if (allocated(y_ij_fks_ev)) deallocate(y_ij_fks_ev)
        if (allocated(p_i_fks_ev)) deallocate(p_i_fks_ev)
        if (allocated(p_i_fks_cnt)) deallocate(p_i_fks_cnt)
   end subroutine deallocate_fksvariables
end module fksvariables
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module ccalculatedborn
    implicit none
   logical, allocatable :: calculatedborn(:)
 contains
   subroutine allocate_ccalculatedborn(vector_size)
       integer, intent(in) :: vector_size
       allocate(calculatedborn(vector_size))
   end subroutine allocate_ccalculatedborn
   subroutine reset_ccalculatedborn
       calculatedborn(:) = .false.
   end subroutine reset_ccalculatedborn
   subroutine deallocate_ccalculatedborn
        if (allocated(calculatedborn)) deallocate(calculatedborn)
   end subroutine deallocate_ccalculatedborn
end module ccalculatedborn
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module pev
    use mod_nexternal
    implicit none
   !include 'nexternal_mod.inc'
   double precision, allocatable :: p_ev(:,:,:)
 contains
   subroutine allocate_pev(vector_size)
       integer, intent(in) :: vector_size
       allocate(p_ev(0:3,nexternal_mod,vector_size))
   end subroutine allocate_pev
   subroutine reset_pev
       p_ev(:,:,:) = 0.d0
   end subroutine reset_pev
   subroutine deallocate_pev
        if (allocated(p_ev)) deallocate(p_ev)
   end subroutine deallocate_pev
end module pev
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module c_fxfx_scales
    use mod_nexternal
    implicit none
   !include 'nexternal_mod.inc'
   integer, allocatable :: nFxFx_ren_scales(:)
   double precision, allocatable :: FxFx_ren_scales(:,:), FxFx_fac_scale(:,:)
 contains
   subroutine allocate_c_fxfx_scales(vector_size)
       integer, intent(in) :: vector_size
       allocate(nFxFx_ren_scales(vector_size))
       allocate(FxFx_ren_scales(0:nexternal_mod,vector_size))
       allocate(FxFx_fac_scale(2,vector_size))
   end subroutine allocate_c_fxfx_scales
   subroutine reset_c_fxfx_scales
       nFxFx_ren_scales(:) = 0
       FxFx_ren_scales(:,:) = 0.d0
       FxFx_fac_scale(:,:) = 0.d0
   end subroutine reset_c_fxfx_scales
   subroutine deallocate_c_fxfx_scales
        if (allocated(nFxFx_ren_scales)) deallocate(nFxFx_ren_scales)
        if (allocated(FxFx_ren_scales)) deallocate(FxFx_ren_scales)
        if (allocated(FxFx_fac_scale)) deallocate(FxFx_fac_scale)
   end subroutine deallocate_c_fxfx_scales
end module c_fxfx_scales
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module to_amp_split_soft
   use mod_orders
   implicit none
   double precision, allocatable :: amp_split_soft(:,:)
 contains
   subroutine allocate_to_amp_split_soft(vector_size)
       integer, intent(in) :: vector_size
         allocate(amp_split_soft(AMP_SPLIT_SIZE,vector_size))
   end subroutine allocate_to_amp_split_soft
   subroutine reset_to_amp_split_soft
       amp_split_soft(:,:) = 0.d0
   end subroutine reset_to_amp_split_soft
   subroutine deallocate_to_amp_split_soft
        if (allocated(amp_split_soft)) deallocate(amp_split_soft)
   end subroutine deallocate_to_amp_split_soft
end module to_amp_split_soft
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module cnbody
    implicit none
   logical, allocatable :: nbody(:)
 contains
   subroutine allocate_cnbody(vector_size)
       integer, intent(in) :: vector_size
         allocate(nbody(vector_size))
   end subroutine allocate_cnbody
   subroutine reset_cnbody
       nbody(:) = .false.
   end subroutine reset_cnbody
   subroutine deallocate_cnbody
        if (allocated(nbody)) deallocate(nbody)
   end subroutine deallocate_cnbody
end module cnbody
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module cxiimaxcnt
    implicit none
   double precision, allocatable :: xiimax_cnt(:,:)
 contains
   subroutine allocate_cxiimaxcnt(vector_size)
       integer, intent(in) :: vector_size
       allocate(xiimax_cnt(-2:2,vector_size))
   end subroutine allocate_cxiimaxcnt
   subroutine reset_cxiimaxcnt
       xiimax_cnt(:,:) = 0.d0
   end subroutine reset_cxiimaxcnt
   subroutine deallocate_cxiimaxcnt
        if (allocated(xiimax_cnt)) deallocate(xiimax_cnt)
   end subroutine deallocate_cxiimaxcnt
end module cxiimaxcnt
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module cxi_i_hat
    implicit none
   double precision, allocatable :: xi_i_hat_ev(:),xi_i_hat_cnt(:,:)
 contains
   subroutine allocate_cxi_i_hat(vector_size)
       integer, intent(in) :: vector_size
       allocate(xi_i_hat_ev(vector_size))
       allocate(xi_i_hat_cnt(-2:2,vector_size))
   end subroutine allocate_cxi_i_hat
   subroutine reset_cxi_i_hat
       xi_i_hat_ev(:) = 0.d0
       xi_i_hat_cnt(:,:) = 0.d0
   end subroutine reset_cxi_i_hat
   subroutine deallocate_cxi_i_hat
        if (allocated(xi_i_hat_ev)) deallocate(xi_i_hat_ev)
        if (allocated(xi_i_hat_cnt)) deallocate(xi_i_hat_cnt)
   end subroutine deallocate_cxi_i_hat
end module cxi_i_hat
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module cxiifkscnt
    implicit none
   double precision, allocatable :: xi_i_fks_cnt(:,:)
 contains
   subroutine allocate_cxiifkscnt(vector_size)
       integer, intent(in) :: vector_size
       allocate(xi_i_fks_cnt(-2:2,vector_size))
   end subroutine allocate_cxiifkscnt
    subroutine reset_cxiifkscnt
        xi_i_fks_cnt(:,:) = 0.d0
    end subroutine reset_cxiifkscnt
   subroutine deallocate_cxiifkscnt
        if (allocated(xi_i_fks_cnt)) deallocate(xi_i_fks_cnt)
   end subroutine deallocate_cxiifkscnt
end module cxiifkscnt
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module cxinormev
    implicit none
   double precision, allocatable :: xinorm_ev(:)
 contains
   subroutine allocate_cxinormev(vector_size)
       integer, intent(in) :: vector_size
       allocate(xinorm_ev(vector_size))
   end subroutine allocate_cxinormev
   subroutine reset_cxinormev
       xinorm_ev(:) = 0.d0
   end subroutine reset_cxinormev
   subroutine deallocate_cxinormev
        if (allocated(xinorm_ev)) deallocate(xinorm_ev)
   end subroutine deallocate_cxinormev
end module cxinormev
module cxinormcnt
    implicit none
   double precision, allocatable :: xinorm_cnt(:,:)
 contains
   subroutine allocate_cxinormcnt(vector_size)
       integer, intent(in) :: vector_size
       allocate(xinorm_cnt(-2:2,vector_size))
   end subroutine allocate_cxinormcnt
    subroutine reset_cxinormcnt
        xinorm_cnt(:,:) = 0.d0
    end subroutine reset_cxinormcnt
   subroutine deallocate_cxinormcnt
        if (allocated(xinorm_cnt)) deallocate(xinorm_cnt)
   end subroutine deallocate_cxinormcnt
end module cxinormcnt
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module cxiimaxev
    implicit none
   double precision, allocatable :: xiimax_ev(:)
 contains
   subroutine allocate_cxiimaxev(vector_size)
       integer, intent(in) :: vector_size
       allocate(xiimax_ev(vector_size))
   end subroutine allocate_cxiimaxev
    subroutine reset_cxiimaxev
        xiimax_ev(:) = 0.d0
    end subroutine reset_cxiimaxev
   subroutine deallocate_cxiimaxev
        if (allocated(xiimax_ev)) deallocate(xiimax_ev)
   end subroutine deallocate_cxiimaxev
end module cxiimaxev
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module c_granny_res
    use mod_nexternal
    implicit none
   !include 'nexternal_mod.inc'
   integer, allocatable :: igranny(:),iaunt(:)
   logical, allocatable :: granny_chain(:,:),granny_is_res(:),granny_chain_real_final(:,:)
 contains
   subroutine allocate_c_granny_res(vector_size)
       integer, intent(in) :: vector_size
         allocate(igranny(vector_size))
         allocate(iaunt(vector_size))
         allocate(granny_chain(-nexternal_mod:nexternal_mod,vector_size))
         allocate(granny_is_res(vector_size))
         allocate(granny_chain_real_final(-nexternal_mod:nexternal_mod,vector_size))
   end subroutine allocate_c_granny_res
   subroutine reset_c_granny_res
       igranny(:) = 0
       iaunt(:) = 0
       granny_chain(:,:) = .false.
       granny_is_res(:) = .false.
       granny_chain_real_final(:,:) = .false.
   end subroutine reset_c_granny_res
   subroutine deallocate_c_granny_res
        if (allocated(igranny)) deallocate(igranny)
        if (allocated(iaunt)) deallocate(iaunt)
        if (allocated(granny_chain)) deallocate(granny_chain)
        if (allocated(granny_is_res)) deallocate(granny_is_res)
        if (allocated(granny_chain_real_final)) deallocate(granny_chain_real_final)
   end subroutine deallocate_c_granny_res
end module c_granny_res
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module parton_cms_ev
    implicit none
   double precision, allocatable :: sqrtshat_ev(:),shat_ev(:)
 contains
   subroutine allocate_parton_cms_ev(vector_size)
       integer, intent(in) :: vector_size
         allocate(sqrtshat_ev(vector_size))
         allocate(shat_ev(vector_size))
   end subroutine allocate_parton_cms_ev
   subroutine reset_parton_cms_ev
    sqrtshat_ev(:) = 0.d0 
    shat_ev(:) = 0.d0 
   end subroutine reset_parton_cms_ev
   subroutine deallocate_parton_cms_ev
        if (allocated(sqrtshat_ev)) deallocate(sqrtshat_ev)
        if (allocated(shat_ev)) deallocate(shat_ev)
   end subroutine deallocate_parton_cms_ev
end module parton_cms_ev
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module parton_cms_cnt
    implicit none
   double precision, allocatable :: sqrtshat_cnt(:,:),shat_cnt(:,:)
 contains
   subroutine allocate_parton_cms_cnt(vector_size)
       integer, intent(in) :: vector_size
         allocate(sqrtshat_cnt(-2:2,vector_size))
         allocate(shat_cnt(-2:2,vector_size))
   end subroutine allocate_parton_cms_cnt
   subroutine reset_parton_cms_cnt
    sqrtshat_cnt(:,:) = 0.d0
    shat_cnt(:,:) = 0.d0
   end subroutine reset_parton_cms_cnt
   subroutine deallocate_parton_cms_cnt
        if (allocated(sqrtshat_cnt)) deallocate(sqrtshat_cnt)
        if (allocated(shat_cnt)) deallocate(shat_cnt)
   end subroutine deallocate_parton_cms_cnt
end module parton_cms_cnt
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module camp_split_store
  use mod_orders
   implicit none
   double precision, allocatable :: AMP_SPLIT_STORE_R(:,:)
   double precision, allocatable :: AMP_SPLIT_STORE_B(:,:)
   double complex, allocatable :: AMP_SPLIT_STORE_CNT(:,:,:,:)
   double precision, allocatable :: AMP_SPLIT_STORE_BSF(:,:,:,:)
 contains
   subroutine allocate_camp_split_store(vector_size)
       integer, intent(in) :: vector_size
       allocate(AMP_SPLIT_STORE_R(AMP_SPLIT_SIZE, vector_size))
       allocate(AMP_SPLIT_STORE_B(AMP_SPLIT_SIZE, vector_size))
       allocate(AMP_SPLIT_STORE_CNT(AMP_SPLIT_SIZE, 2, NSPLITORDERS, vector_size))
       allocate(AMP_SPLIT_STORE_BSF(AMP_SPLIT_SIZE, 5, 5, vector_size))
   end subroutine allocate_camp_split_store
   subroutine reset_camp_split_store
    AMP_SPLIT_STORE_R(:,:) = 0.d0
    AMP_SPLIT_STORE_B(:,:) = 0.d0
    AMP_SPLIT_STORE_BSF(:,:,:,:) = 0.d0
    AMP_SPLIT_STORE_CNT(:,:,:,:) = 0.d0
   end subroutine reset_camp_split_store
   subroutine deallocate_camp_split_store
        if (allocated(AMP_SPLIT_STORE_R)) deallocate(AMP_SPLIT_STORE_R)
        if (allocated(AMP_SPLIT_STORE_B)) deallocate(AMP_SPLIT_STORE_B)
        if (allocated(AMP_SPLIT_STORE_CNT)) deallocate(AMP_SPLIT_STORE_CNT)
        if (allocated(AMP_SPLIT_STORE_BSF)) deallocate(AMP_SPLIT_STORE_BSF)
   end subroutine deallocate_camp_split_store
end module camp_split_store
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module cgenps_fks
    implicit none
   double precision, allocatable :: veckn_ev(:),veckbarn_ev(:),xp0jfks(:)
 contains
   subroutine allocate_cgenps_fks(vector_size)
       integer, intent(in) :: vector_size
       allocate(veckn_ev(vector_size))
       allocate(veckbarn_ev(vector_size))
       allocate(xp0jfks(vector_size))
   end subroutine allocate_cgenps_fks
   subroutine reset_cgenps_fks
       veckn_ev(:) = 0.d0
       veckbarn_ev(:) = 0.d0
       xp0jfks(:) = 0.d0
   end subroutine reset_cgenps_fks
   subroutine deallocate_cgenps_fks
        if (allocated(veckn_ev)) deallocate(veckn_ev)
        if (allocated(veckbarn_ev)) deallocate(veckbarn_ev)
        if (allocated(xp0jfks)) deallocate(xp0jfks)
   end subroutine deallocate_cgenps_fks
end module cgenps_fks
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module cbjorkenx
    implicit none
   double precision, allocatable :: xbjrk_ev(:,:),xbjrk_cnt(:,:,:)
 contains
   subroutine allocate_cbjorkenx(vector_size)
       integer, intent(in) :: vector_size
       allocate(xbjrk_ev(2,vector_size))
       allocate(xbjrk_cnt(2,-2:2,vector_size))
   end subroutine allocate_cbjorkenx
   subroutine reset_cbjorkenx
       xbjrk_ev(:,:) = 0.d0
       xbjrk_cnt(:,:,:) = 0.d0
   end subroutine reset_cbjorkenx
   subroutine deallocate_cbjorkenx
        if (allocated(xbjrk_ev)) deallocate(xbjrk_ev)
        if (allocated(xbjrk_cnt)) deallocate(xbjrk_cnt)
   end subroutine deallocate_cbjorkenx
end module cbjorkenx
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module cbjrk12_ev
    implicit none
   double precision, allocatable :: tau_ev(:),ycm_ev(:)
 contains
   subroutine allocate_cbjrk12_ev(vector_size)
       integer, intent(in) :: vector_size
       allocate(tau_ev(vector_size))
       allocate(ycm_ev(vector_size))
   end subroutine allocate_cbjrk12_ev
   subroutine reset_cbjrk12_ev
       tau_ev(:) = 0.d0
       ycm_ev(:) = 0.d0
   end subroutine reset_cbjrk12_ev
   subroutine deallocate_cbjrk12_ev
        if (allocated(tau_ev)) deallocate(tau_ev)
        if (allocated(ycm_ev)) deallocate(ycm_ev)
   end subroutine deallocate_cbjrk12_ev
end module cbjrk12_ev
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module cbjrk12_cnt
    implicit none
   double precision, allocatable :: tau_cnt(:,:),ycm_cnt(:,:)
 contains
   subroutine allocate_cbjrk12_cnt(vector_size)
       integer, intent(in) :: vector_size
       allocate(tau_cnt(-2:2,vector_size))
       allocate(ycm_cnt(-2:2,vector_size))
   end subroutine allocate_cbjrk12_cnt
   subroutine reset_cbjrk12_cnt
       tau_cnt(:,:) = 0.d0
       ycm_cnt(:,:) = 0.d0
   end subroutine reset_cbjrk12_cnt
   subroutine deallocate_cbjrk12_cnt
        if (allocated(tau_cnt)) deallocate(tau_cnt)
        if (allocated(ycm_cnt)) deallocate(ycm_cnt)
   end subroutine deallocate_cbjrk12_cnt
end module cbjrk12_cnt
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module virtgranny_boost
    implicit none
   double precision, allocatable :: shybst(:),chybst(:),chybstmo(:)
 contains
   subroutine allocate_virtgranny_boost(vector_size)
       integer, intent(in) :: vector_size
         allocate(shybst(vector_size))
         allocate(chybst(vector_size))
         allocate(chybstmo(vector_size))
   end subroutine allocate_virtgranny_boost
   subroutine reset_virtgranny_boost
       shybst(:) = 0.d0
       chybst(:) = 0.d0
       chybstmo(:) = 0.d0
   end subroutine reset_virtgranny_boost
   subroutine deallocate_virtgranny_boost
        if (allocated(shybst)) deallocate(shybst)
        if (allocated(chybst)) deallocate(chybst)
        if (allocated(chybstmo)) deallocate(chybstmo)
   end subroutine deallocate_virtgranny_boost
end module virtgranny_boost
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module c_conflictingbw
    use mod_nexternal
    implicit none
   !include 'nexternal_mod.inc'
   double precision, allocatable :: cBW_mass(:,:,:),cBW_width(:,:,:)
   integer, allocatable :: cBW_level_max(:),cBW(:,:),cBW_level(:,:)
 contains
   subroutine allocate_c_conflictingbw(vector_size)
       integer, intent(in) :: vector_size
         allocate(cBW_mass(-1:1,-nexternal_mod:-1,vector_size))
         allocate(cBW_width(-1:1,-nexternal_mod:-1,vector_size))
         allocate(cBW_level_max(vector_size))
         allocate(cBW(-nexternal_mod:-1,vector_size))
         allocate(cBW_level(-nexternal_mod:-1,vector_size))
   end subroutine allocate_c_conflictingbw
   subroutine reset_c_conflictingbw
       cBW_mass(:,:,:) = 0.d0
       cBW_width(:,:,:) = 0.d0
       cBW_level_max(:) = 0
       cBW(:,:) = 0
       cBW_level(:,:) = 0
   end subroutine reset_c_conflictingbw
   subroutine deallocate_c_conflictingbw
        if (allocated(cBW_mass)) deallocate(cBW_mass)
        if (allocated(cBW_width)) deallocate(cBW_width)
        if (allocated(cBW_level_max)) deallocate(cBW_level_max)
        if (allocated(cBW)) deallocate(cBW)
        if (allocated(cBW_level)) deallocate(cBW_level)
   end subroutine deallocate_c_conflictingbw
end module c_conflictingbw
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module cxij_aor
    implicit none
   double complex, allocatable :: xij_aor(:)
 contains
   subroutine allocate_cxij_aor(vector_size)
       integer, intent(in) :: vector_size
       allocate(xij_aor(vector_size))
   end subroutine allocate_cxij_aor
   subroutine reset_cxij_aor
       xij_aor(:) = (0.d0,0.d0)
   end subroutine reset_cxij_aor
   subroutine deallocate_cxij_aor
        if (allocated(xij_aor)) deallocate(xij_aor)
   end subroutine deallocate_cxij_aor
end module cxij_aor
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module cnocntevents
    implicit none
   logical, allocatable :: nocntevents(:)
 contains
   subroutine allocate_cnocntevents(vector_size)
       integer, intent(in) :: vector_size
         allocate(nocntevents(vector_size))
   end subroutine allocate_cnocntevents
    subroutine reset_cnocntevents
        nocntevents(:) = .false.
    end subroutine reset_cnocntevents
   subroutine deallocate_cnocntevents
        if (allocated(nocntevents)) deallocate(nocntevents)
   end subroutine deallocate_cnocntevents
end module cnocntevents
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module c_vegas_x
    implicit none
    double precision, allocatable :: xvar(:,:)
 contains
   subroutine allocate_c_vegas_x(vector_size)
       integer, intent(in) :: vector_size
         allocate(xvar(99,vector_size))
   end subroutine allocate_c_vegas_x
   subroutine reset_c_vegas_x
       xvar(:,:) = 0.d0
   end subroutine reset_c_vegas_x
   subroutine deallocate_c_vegas_x
        if (allocated(xvar)) deallocate(xvar)
   end subroutine deallocate_c_vegas_x
end module c_vegas_x
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module c_rat_xi
    implicit none
    double precision, allocatable :: rat_xi_orig(:)
 contains
   subroutine allocate_c_rat_xi(vector_size)
       integer, intent(in) :: vector_size
         allocate(rat_xi_orig(vector_size))
   end subroutine allocate_c_rat_xi
   subroutine reset_c_rat_xi
       rat_xi_orig(:) = 0.d0
   end subroutine reset_c_rat_xi
   subroutine deallocate_c_rat_xi
        if (allocated(rat_xi_orig)) deallocate(rat_xi_orig)
   end subroutine deallocate_c_rat_xi
end module c_rat_xi
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module to_virtgranny
    implicit none
    double precision, allocatable :: granny_m2_red(:,:)
 contains
   subroutine allocate_to_virtgranny(vector_size)
       integer, intent(in) :: vector_size
         allocate(granny_m2_red(-1:1,vector_size))
   end subroutine allocate_to_virtgranny
   subroutine reset_to_virtgranny
       granny_m2_red(:,:) = 0.d0
   end subroutine reset_to_virtgranny
   subroutine deallocate_to_virtgranny
        if (allocated(granny_m2_red)) deallocate(granny_m2_red)
   end subroutine deallocate_to_virtgranny
end module to_virtgranny
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module cgrannyrange
    implicit none
    double precision, allocatable :: xmbemin2(:),xmbemax2(:)
 contains
   subroutine allocate_cgrannyrange(vector_size)
       integer, intent(in) :: vector_size
         allocate(xmbemin2(vector_size))
         allocate(xmbemax2(vector_size))
   end subroutine allocate_cgrannyrange
   subroutine reset_cgrannyrange
       xmbemin2(:) = 0.d0
       xmbemax2(:) = 0.d0
   end subroutine reset_cgrannyrange
   subroutine deallocate_cgrannyrange
        if (allocated(xmbemin2)) deallocate(xmbemin2)
        if (allocated(xmbemax2)) deallocate(xmbemax2)
   end subroutine deallocate_cgrannyrange
end module cgrannyrange
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module coffset
    implicit none
    double precision, allocatable :: offset(:)
 contains
   subroutine allocate_coffset(vector_size)
       integer, intent(in) :: vector_size
         allocate(offset(vector_size))
   end subroutine allocate_coffset
   subroutine reset_coffset
       offset(:) = 0.d0
   end subroutine reset_coffset
   subroutine deallocate_coffset
        if (allocated(offset)) deallocate(offset)
   end subroutine deallocate_coffset
end module coffset
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module c_skip_only_event_phsp
    implicit none
    logical, allocatable :: only_event_phsp(:),skip_event_phsp(:)
 contains
   subroutine allocate_c_skip_only_event_phsp(vector_size)
       integer, intent(in) :: vector_size
         allocate(only_event_phsp(vector_size))
         allocate(skip_event_phsp(vector_size))
   end subroutine allocate_c_skip_only_event_phsp
   subroutine reset_c_skip_only_event_phsp
       only_event_phsp(:) = .false.
       skip_event_phsp(:) = .false.
   end subroutine reset_c_skip_only_event_phsp
   subroutine deallocate_c_skip_only_event_phsp
        if (allocated(only_event_phsp)) deallocate(only_event_phsp)
        if (allocated(skip_event_phsp)) deallocate(skip_event_phsp)
   end subroutine deallocate_c_skip_only_event_phsp
end module c_skip_only_event_phsp
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module c_isolsign
    implicit none
    integer, allocatable :: isolsign(:)
 contains
   subroutine allocate_c_isolsign(vector_size)
       integer, intent(in) :: vector_size
         allocate(isolsign(vector_size))
   end subroutine allocate_c_isolsign
   subroutine reset_c_isolsign
       isolsign(:) = 0
   end subroutine reset_c_isolsign
   subroutine deallocate_c_isolsign
        if (allocated(isolsign)) deallocate(isolsign)
   end subroutine deallocate_c_isolsign
end module c_isolsign
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module to_phase_space_s_channel
    use mod_nexternal
    implicit none
   !include 'nexternal_mod.inc'
    double precision, allocatable :: s_mass(:,:)
 contains
   subroutine allocate_to_phase_space_s_channel(vector_size)
       integer, intent(in) :: vector_size
         allocate(s_mass(-nexternal_mod:nexternal_mod,vector_size))
   end subroutine allocate_to_phase_space_s_channel
   subroutine reset_to_phase_space_s_channel
       s_mass(:,:) = 0.d0
   end subroutine reset_to_phase_space_s_channel
   subroutine deallocate_to_phase_space_s_channel
        if (allocated(s_mass)) deallocate(s_mass)
   end subroutine deallocate_to_phase_space_s_channel
end module to_phase_space_s_channel
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module coupl_es
    implicit none
    double precision, allocatable :: qes2(:)
 contains
   subroutine allocate_coupl_es(vector_size)
       integer, intent(in) :: vector_size
         allocate(qes2(vector_size))
   end subroutine allocate_coupl_es
   subroutine reset_coupl_es
       qes2(:) = 0.d0
   end subroutine reset_coupl_es
   subroutine deallocate_coupl_es
        if (allocated(qes2)) deallocate(qes2)
   end subroutine deallocate_coupl_es
end module coupl_es
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module c_goodhel
  use mod_nfksconfigs
  use mod_born_nhel
    implicit none
    logical, allocatable :: goodhel(:,:,:)
 contains
   subroutine allocate_c_goodhel(vector_size)
       integer, intent(in) :: vector_size
         allocate(goodhel(max_bhel_mod,FKS_CONFIGS_mod,vector_size))
   end subroutine allocate_c_goodhel
    subroutine reset_c_goodhel
        goodhel(:,:,:) = .false.
    end subroutine reset_c_goodhel
   subroutine deallocate_c_goodhel
        if (allocated(goodhel)) deallocate(goodhel)
   end subroutine deallocate_c_goodhel
end module c_goodhel
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module to_saveamp
  use mod_genps
  use mod_born_nhel
  implicit none
  double complex, allocatable :: saveamp(:,:,:)
  contains
  subroutine allocate_to_saveamp(vector_size)
    integer, intent(in) :: vector_size
    allocate(saveamp(ngraphs_mod,max_bhel_mod,vector_size))
  end subroutine allocate_to_saveamp
  subroutine reset_to_saveamp
    saveamp(:,:,:) = (0.d0,0.d0)
  end subroutine reset_to_saveamp
  subroutine deallocate_to_saveamp
    if (allocated(saveamp)) deallocate(saveamp)
  end subroutine deallocate_to_saveamp
end module to_saveamp
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module to_savemom
  use MOD_NEXTERNAL
  implicit none
  double precision, allocatable :: savemom(:,:,:)
  contains
  subroutine allocate_to_savemom(vector_size)
    integer, intent(in) :: vector_size
    allocate(savemom(nexternal_mod-1,2,vector_size))
  end subroutine allocate_to_savemom
  subroutine reset_to_savemom
    savemom(:,:,:) = 0.d0
  end subroutine reset_to_savemom
  subroutine deallocate_to_savemom
    if (allocated(savemom)) deallocate(savemom)
  end subroutine deallocate_to_savemom
end module to_savemom
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module ntry_real
  use mod_nfksconfigs
  integer, allocatable :: ntry(:,:)
  contains
  subroutine allocate_ntry_real(vector_size)
    integer, intent(in) :: vector_size
    allocate(ntry(FKS_CONFIGS_mod, vector_size))
  end subroutine allocate_ntry_real
  subroutine deallocate_ntry_real
   if (allocated(ntry)) deallocate(ntry)
  end subroutine deallocate_ntry_real
end module ntry_real
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module cBorn
  use mod_nfksconfigs
  double precision, allocatable :: hel_fac(:)
  integer, allocatable :: get_hel(:), skip(:,:), ntry(:,:)
  contains
  subroutine allocate_cBorn(vector_size)
    integer, intent(in) :: vector_size
    allocate(hel_fac(vector_size))
    allocate(get_hel(vector_size))
    allocate(skip(FKS_CONFIGS_mod, vector_size))
    allocate(ntry(FKS_CONFIGS_mod, vector_size))
    skip(:,:) = -1
    ntry(:,:) = 0
  end subroutine allocate_cBorn
  subroutine reset_cBorn
!    hel_fac(:) = 0.d0
!    get_hel(:) = 0
    skip(:,:) = -1
    ntry(:,:) = 0
  end subroutine reset_cBorn
  subroutine deallocate_cBorn
    if (allocated(hel_fac)) deallocate(hel_fac)
    if (allocated(get_hel)) deallocate(get_hel)
    if (allocated(skip)) deallocate(skip)
    if (allocated(ntry)) deallocate(ntry)
  end subroutine deallocate_cBorn
end module cBorn
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module c_wgt_ME_tree
  implicit none
  double precision, allocatable :: wgt_ME_born(:)
  double precision, allocatable :: wgt_ME_real(:)
contains
  subroutine allocate_c_wgt_ME_tree(vector_size)
    integer, intent(in) :: vector_size
    allocate(wgt_ME_born(vector_size))
    allocate(wgt_ME_real(vector_size))
  end subroutine allocate_c_wgt_ME_tree
  subroutine reset_c_wgt_ME_tree
    wgt_ME_born(:) = 0.d0
    wgt_ME_real(:) = 0.d0
  end subroutine reset_c_wgt_ME_tree
  subroutine deallocate_c_wgt_ME_tree
    if (allocated(wgt_ME_born)) deallocate(wgt_ME_born)
    if (allocated(wgt_ME_real)) deallocate(wgt_ME_real)
  end subroutine deallocate_c_wgt_ME_tree
end module c_wgt_ME_tree
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module c_born_cnt
  use mod_orders
  implicit none
  complex*16, allocatable :: ans_cnt(:,:,:)
  contains
  subroutine allocate_c_born_cnt(vector_size)
    integer, intent(in) :: vector_size
    allocate(ans_cnt(2,NSPLITORDERS,vector_size))
  end subroutine allocate_c_born_cnt
  subroutine reset_c_born_cnt
    ans_cnt(:,:,:) = 0.d0
  end subroutine reset_c_born_cnt
  subroutine deallocate_c_born_cnt
    if (allocated(ans_cnt)) deallocate(ans_cnt)
  end subroutine deallocate_c_born_cnt
end module c_born_cnt
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module vectorize
  use factor_nbody
  use factor_n1body
  use factor_n1body_nlops
  use pborn
  use pborn_ev
  use pborn_l
  use pborn_coll
  use pborn_norad
  use to_amps
  use to_amps_born
  use counterevnts
  use write_granny_resonance
  use parton_cms_stuff
  use fksvariables
  use ccalculatedborn
  use pev
  use c_fxfx_scales
  use to_amp_split_soft
  use cnbody
  use cxiimaxcnt
  use cxi_i_hat
  use cxiifkscnt
  use cxinormev
  use cxinormcnt
  use cxiimaxev
  use c_granny_res
  use parton_cms_ev
  use parton_cms_cnt
  use camp_split_store
  use cgenps_fks
  use cbjorkenx
  use cbjrk12_ev
  use cbjrk12_cnt
  use virtgranny_boost
  use c_conflictingbw
  use cxij_aor
  use cnocntevents
  use c_vegas_x
  use c_rat_xi
  use to_virtgranny
  use cgrannyrange
  use coffset
  use c_skip_only_event_phsp
  use c_isolsign
  use to_phase_space_s_channel
  use coupl_es
  use c_goodhel
  use to_saveamp
  use to_savemom
  use cBorn
  use c_wgt_ME_tree
  use c_born_cnt
  use ntry_real
!  use strong
!  use rscale
!  use couplings
  implicit none
  integer vec_size_store
    contains
    subroutine allocate_storage(vector_size)
       implicit none
       integer, intent(in) :: vector_size
       vec_size_store = vector_size
       call allocate_ntry_real(vector_size)
       call allocate_factor_nbody(vector_size)
       call allocate_factor_n1body(vector_size)
       call allocate_factor_n1body_nlops(vector_size)
       call allocate_pborn(vector_size)
       call allocate_pborn_ev(vector_size)
       call allocate_pborn_l(vector_size)
       call allocate_pborn_coll(vector_size)
       call allocate_pborn_norad(vector_size)
       call allocate_to_amps(vector_size)
       call allocate_to_amps_born(vector_size)
       call allocate_counterevnts(vector_size)
       call allocate_write_granny_resonance(vector_size)
       call allocate_parton_cms_stuff(vector_size)
       call allocate_fksvariables(vector_size)
       call allocate_ccalculatedborn(vector_size)
       call allocate_pev(vector_size)
       call allocate_c_fxfx_scales(vector_size)
       call allocate_to_amp_split_soft(vector_size)
       call allocate_cnbody(vector_size)
       call allocate_cxiimaxcnt(vector_size)
       call allocate_cxi_i_hat(vector_size)
       call allocate_cxiifkscnt(vector_size)
       call allocate_cxinormev(vector_size)
       call allocate_cxinormcnt(vector_size)
       call allocate_cxiimaxev(vector_size)
       call allocate_c_granny_res(vector_size)
       call allocate_parton_cms_ev(vector_size)
       call allocate_parton_cms_cnt(vector_size)
       call allocate_camp_split_store(vector_size)
       call allocate_cgenps_fks(vector_size)
       call allocate_cbjorkenx(vector_size)
       call allocate_cbjrk12_ev(vector_size)
       call allocate_cbjrk12_cnt(vector_size)
       call allocate_virtgranny_boost(vector_size)
       call allocate_c_conflictingbw(vector_size)
       call allocate_cxij_aor(vector_size)
       call allocate_cnocntevents(vector_size)
       call allocate_c_vegas_x(vector_size)
       call allocate_c_rat_xi(vector_size)
       call allocate_to_virtgranny(vector_size)
       call allocate_cgrannyrange(vector_size)
       call allocate_coffset(vector_size)
       call allocate_c_skip_only_event_phsp(vector_size)
       call allocate_c_isolsign(vector_size)
       call allocate_to_phase_space_s_channel(vector_size)
       call allocate_coupl_es(vector_size)
       call allocate_c_goodhel(vector_size)
       call allocate_to_saveamp(vector_size)
       call allocate_to_savemom(vector_size)
       call allocate_cBorn(vector_size)
       call allocate_c_wgt_ME_tree(vector_size)
       call allocate_c_born_cnt(vector_size)
!       call allocate_strong(vector_size)
!       call allocate_rscale(vector_size)
!       call allocate_couplings(vector_size)
    end subroutine allocate_storage

    subroutine event_reset(vector_size)
      implicit none 
      integer, intent(in) :: vector_size
!      call reset_pborn
!      call reset_pborn_ev
!      call reset_pborn_l
!      call reset_pborn_coll
!      call reset_pborn_norad
!      call reset_to_amps
!      call reset_to_amps_born(vector_size)
!      call reset_counterevnts
!      call reset_write_granny_resonance
!      call reset_parton_cms_stuff
!      call reset_fksvariables
!      call reset_ccalculatedborn
!      call reset_pev
!      call reset_c_fxfx_scales
!      call reset_to_amp_split_soft
!      call reset_cnbody
!      call reset_cxiimaxcnt
!      call reset_cxi_i_hat
!      call reset_cxiifkscnt
!      call reset_cxinormev
!      call reset_cxinormcnt
!      call reset_cxiimaxev
!      call reset_c_granny_res
!      call reset_parton_cms_ev
!      call reset_parton_cms_cnt
!      call reset_camp_split_store
!      call reset_cgenps_fks
!      call reset_cbjorkenx
!      call reset_cbjrk12_ev
!      call reset_cbjrk12_cnt
!      call reset_virtgranny_boost
!      call reset_c_conflictingbw
!      call reset_cxij_aor
!      call reset_cnocntevents
!      call reset_c_vegas_x
!      call reset_c_rat_xi
!      call reset_to_virtgranny
!      call reset_cgrannyrange
!      call reset_coffset
!      call reset_c_skip_only_event_phsp
!      call reset_c_isolsign
!      call reset_to_phase_space_s_channel
!      call reset_coupl_es
!      call reset_c_goodhel
!      call reset_to_saveamp
!      call reset_to_savemom
!      call reset_cBorn
!      call reset_c_wgt_ME_tree
!      call reset_c_born_cnt
!      call reset_strong
!      call reset_rscale
!      call reset_couplings
    end subroutine event_reset

    subroutine deallocate_storage
       call deallocate_ntry_real
       call deallocate_factor_nbody
       call deallocate_factor_n1body
       call deallocate_factor_n1body_nlops
       call deallocate_pborn
       call deallocate_pborn_ev
       call deallocate_pborn_l
       call deallocate_pborn_coll
       call deallocate_pborn_norad
       call deallocate_to_amps
       call deallocate_to_amps_born
       call deallocate_counterevnts
       call deallocate_write_granny_resonance
       call deallocate_parton_cms_stuff
       call deallocate_fksvariables
       call deallocate_ccalculatedborn
       call deallocate_pev
       call deallocate_c_fxfx_scales
       call deallocate_to_amp_split_soft
       call deallocate_cnbody
       call deallocate_cxiimaxcnt
       call deallocate_cxi_i_hat
       call deallocate_cxiifkscnt
       call deallocate_cxinormev
       call deallocate_cxinormcnt
       call deallocate_cxiimaxev
       call deallocate_c_granny_res
       call deallocate_parton_cms_ev
       call deallocate_parton_cms_cnt
       call deallocate_camp_split_store
       call deallocate_cgenps_fks
       call deallocate_cbjorkenx
       call deallocate_cbjrk12_ev
       call deallocate_cbjrk12_cnt
       call deallocate_virtgranny_boost
       call deallocate_c_conflictingbw
       call deallocate_cxij_aor
       call deallocate_cnocntevents
       call deallocate_c_vegas_x
       call deallocate_c_rat_xi
       call deallocate_to_virtgranny
       call deallocate_cgrannyrange
       call deallocate_coffset
       call deallocate_c_skip_only_event_phsp
       call deallocate_c_isolsign
       call deallocate_to_phase_space_s_channel
       call deallocate_coupl_es
       call deallocate_c_goodhel
       call deallocate_to_saveamp
       call deallocate_to_savemom
       call deallocate_cBorn
       call deallocate_c_wgt_ME_tree
       call deallocate_c_born_cnt
!       call deallocate_strong
!       call deallocate_rscale
!       call deallocate_couplings
    end subroutine deallocate_storage
end module vectorize
         
