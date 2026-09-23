module history_test_state
  implicit none
  include 'nexternal.inc'
  include 'orders.inc'
  include 'coupl.inc'
  include 'run.inc'
  include 'q_es.inc'
  double precision :: p1_cnt(0:3,nexternal,-2:2),wgt_cnt(-2:2),pswgt_cnt(-2:2),jac_cnt(-2:2)
  common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
  double precision :: p_born(0:3,nexternal-1),p_ev(0:3,nexternal)
  common/pborn/p_born
  common/pev/p_ev
  double precision :: ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
  common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
  double precision :: xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
  common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt
  double precision :: xi_i_fks_cnt(-2:2),xiimax_cnt(-2:2),xi_i_hat_ev,xi_i_hat_cnt(-2:2)
  common/cxiifkscnt/xi_i_fks_cnt
  common/cxiimaxcnt/xiimax_cnt
  common/cxi_i_hat/xi_i_hat_ev,xi_i_hat_cnt
  double precision :: xiScut_used,xiBSVcut_used
  common/cxiScut_used/xiScut_used,xiBSVcut_used
  integer :: i_fks,j_fks,nFKSprocess,fold,ifold_counter
  common/fks_indices/i_fks,j_fks
  common/c_nFKSprocess/nFKSprocess
  common/cfl/fold,ifold_counter
  character*4 :: abrv
  common/to_abrv/abrv
  double precision :: f_r,f_s,f_c,f_dc,f_sc,f_dsc(4)
  common/factor_n1body/f_r,f_s,f_c,f_dc,f_sc,f_dsc
  double precision :: f_s_MC_S,f_s_MC_H,f_c_MC_S,f_c_MC_H,f_sc_MC_S,f_sc_MC_H,f_MC_S,f_MC_H
  common/factor_n1body_NLOPS/f_s_MC_S,f_s_MC_H,f_c_MC_S,f_c_MC_H,f_sc_MC_S,f_sc_MC_H,f_MC_S,f_MC_H
  double precision :: f_pdfsch_d,f_pdfsch_p,f_pdfsch_l
  common/factor_pdfsch/f_pdfsch_d,f_pdfsch_p,f_pdfsch_l
  double precision :: deg_xi(3),deg_lxi(3),deg_muF(3),dis_p(3),dis_l(3),dis_d(3)
  common/to_amp_split_deg/deg_xi,deg_lxi,deg_muF
  common/to_amp_split_dis/dis_p,dis_l,dis_d
  double precision :: test_p,test_gsoft,test_gcoll
  integer :: real_calls,degenerate_calls
end module

program check_mc_history_weights
  use history_test_state
  use weight_lines
  implicit none
  double precision :: baseline(3,15,3),specialized(3,15,3),probne
  double precision :: probabilities(3),gvalues(3)
  integer :: imass,ip,igs,igc,icuts,mode,i,cases,baseline_reals
  logical :: cuts_born,cuts_real
  call allocate_weight_lines(nexternal)
  probabilities=[0d0,0.37d0,1d0]
  gvalues=[0d0,0.6d0,1d0]
  p_ev=0d0
  p_ev(0,:)=[100d0,100d0,80d0,60d0,60d0]
  p_born=p_ev(:,1:nexternal-1)
  do i=-2,2
    p1_cnt(:,:,i)=p_ev
  enddo
  xi_i_fks_ev=0.2d0
  y_ij_fks_ev=0.8d0
  xi_i_fks_cnt=0.3d0
  xiimax_cnt=1d0
  xi_i_hat_ev=0.2d0
  xiScut_used=0.5d0
  i_fks=5
  j_fks=3
  nFKSprocess=1
  fold=0
  ifold_counter=1
  abrv='all '
  ickkw=0
  lpp=1
  xbk=0.1d0
  scale=100d0
  q2fact=scale**2
  QES2=scale**2
  g=1.3d0
  ybst_til_tolab=0d0
  ybst_til_tocm=0d0
  shat=40000d0
  sqrtshat=200d0
  f_r=1.1d0
  f_s=1.2d0
  f_c=1.3d0
  f_dc=1.4d0
  f_sc=1.5d0
  f_dsc=[1.6d0,1.7d0,1.8d0,1.9d0]
  f_s_MC_S=2.1d0
  f_s_MC_H=2.2d0
  f_c_MC_S=2.3d0
  f_c_MC_H=2.4d0
  f_sc_MC_S=2.5d0
  f_sc_MC_H=2.6d0
  f_MC_S=2.7d0
  f_MC_H=2.8d0
  f_pdfsch_d=3.1d0
  f_pdfsch_p=3.2d0
  f_pdfsch_l=3.3d0
  cases=0
  do imass=0,1
    test_mass=173d0*imass
    do ip=1,3
      test_p=probabilities(ip)
      do igs=1,3
        test_gsoft=gvalues(igs)
        do igc=1,3
          test_gcoll=gvalues(igc)
          do icuts=0,3
            cuts_born=btest(icuts,0)
            cuts_real=btest(icuts,1)
            do mode=0,2
              mc_H_only=mode.eq.1
              mc_S_only=mode.eq.2
              icontr=0
              real_calls=0
              degenerate_calls=0
              ! Poison remnants: the H-only path must not read stale S data.
              deg_xi=1d30
              deg_lxi=-1d30
              deg_muF=2d30
              dis_p=3d30
              dis_l=-3d30
              dis_d=4d30
              call compute_native_NLOPS_weights(p_ev,p_ev,p_ev,1d0,cuts_born,cuts_real,probne)
              specialized=0d0
              do i=1,icontr
                specialized(:,itype(i),amppos(i))=specialized(:,itype(i),amppos(i))+wgt(:,i)
                if (mode.eq.1.and..not.H_event(i)) error stop 'S record in H-only evaluation'
                if (mode.eq.2.and.H_event(i)) error stop 'H record in S-only evaluation'
              enddo
              if (mode.eq.0) then
                baseline=specialized
                baseline_reals=real_calls
              else
                do i=1,15
                  if ((i.eq.1.or.i.eq.13.or.(i.ge.8.and.i.le.10)).eqv.(mode.eq.1)) then
                    if (maxval(abs(specialized(:,i,:)-baseline(:,i,:))).gt.1d-12) &
                      error stop 'specialized native weights differ'
                  endif
                enddo
                if (real_calls.gt.baseline_reals) error stop 'extra real evaluations'
              endif
              if (mode.eq.1) then
                if (degenerate_calls.ne.0) error stop 'H-only degenerate remnant evaluation'
                if (cuts_born.and.test_p.eq.0d0.and.real_calls.ne.0) &
                  error stop 'zero P evaluated H matrix elements'
                if (test_gsoft.eq.1d0.and.real_calls.gt.1) &
                  error stop 'zero G evaluated H counterterms'
              endif
            enddo
            cases=cases+1
          enddo
        enddo
      enddo
    enddo
  enddo
  write(*,*) 'PASS native H and S weights',cases,' cases'
end program

subroutine compute_MC_subt_term(p,p_lab,p_cms,jacPS,passcuts,probne)
  use history_test_state
  use extra_weights
  use kinematics_module
  implicit none
  double precision :: p(0:3,5),p_lab(0:3,5),p_cms(0:3,5),jacPS,probne
  logical :: passcuts
  integer :: i,orders(2)
  probne=test_p
  gfactsf=test_gsoft
  gfactcl=test_gcoll
  do i=1,3
    call amp_split_pos_to_orders(i,orders)
    QCD_power=orders(1)
    wgtcpower=0d0
    orders_tag=i
    amp_pos=i
    call add_wgt(12,orders,11d0*i*probne,0d0,0d0)
    call add_wgt(13,orders,-13d0*i*probne,0d0,0d0)
  enddo
end subroutine

subroutine sreal(p,xi,y,ans)
  use history_test_state
  implicit none
  double precision :: p(0:3,5),xi,y,ans
  real_calls=real_calls+1
  amp_split=[2d0+xi+3d0*y,4d0+2d0*xi+y,0d0]
  ans=sum(amp_split)
end subroutine

subroutine sreal_deg(p,xi,y,ans1,ans2)
  use history_test_state
  implicit none
  double precision :: p(0:3,5),xi,y,ans1,ans2
  degenerate_calls=degenerate_calls+1
  deg_xi=[3d0,5d0,7d0]
  deg_lxi=[4d0,6d0,8d0]
  deg_muF=[5d0,7d0,9d0]
  dis_p=[6d0,8d0,10d0]
  dis_l=[7d0,9d0,11d0]
  dis_d=[8d0,10d0,12d0]
  amp_split=-1d30
  ans1=1d0
  ans2=2d0
end subroutine

double precision function fks_Sij(p,i,j,xi,y)
  implicit none
  double precision :: p(0:3,5),xi,y
  integer :: i,j
  fks_Sij=0.7d0+0.1d0*xi+0.05d0*y
end function

subroutine amp_split_pos_to_orders(i,orders)
  implicit none
  integer :: i,orders(2)
  if (i.eq.1) then
    orders=[4,0]
  elseif (i.eq.2) then
    orders=[0,4]
  else
    orders=[2,2]
  endif
end subroutine

integer function get_orders_tag(orders)
  implicit none
  integer :: orders(2)
  get_orders_tag=10*orders(1)+orders(2)
end function

integer function get_n_tagged_photons()
  get_n_tagged_photons=0
end function

double precision function get_rescale_alpha_factor(n,order)
  integer :: n,order
  get_rescale_alpha_factor=1d0
end function

subroutine set_pdg(ict,sector)
  use weight_lines
  implicit none
  integer :: ict,sector
  pdg(:,ict)=[21,21,6,-6,21]
  pdg_uborn(:,ict)=pdg(:,ict)
end subroutine

subroutine set_cms_stuff(i)
  integer :: i
end subroutine

subroutine set_alphaS(p)
  double precision :: p(0:3,5)
end subroutine

subroutine set_FxFx_scale(i,p)
  integer :: i
  double precision :: p(0:3,5)
end subroutine

subroutine include_multichannel_enhance(i)
  integer :: i
end subroutine
