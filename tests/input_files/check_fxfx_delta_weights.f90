! Use the production FxFx cache and all real/MC/G weight assembly routines.
! Only matrix elements, clustering and Delta's probability are controlled.
module fks_phase_space_data
  implicit none
  double precision :: p1_cnt(0:3,5,-2:2)=1d0,p_born(0:3,4)=1d0
  double precision :: xi_i_fks_ev=0.2d0,y_ij_fks_ev=0.8d0
  double precision :: xi_i_fks_cnt(-2:2)=0.3d0,xiimax_cnt(-2:2)=1d0,xi_i_hat_ev=0.2d0
  double precision :: xbjrk_cnt(2,-2:2)=0.1d0
end module

module extra_weights
  implicit none
  integer :: QCD_power,orders_tag,amp_pos
  double precision :: wgtcpower
end module

module process_module
  logical :: mcatnlo_delta_mod=.true.
end module

module mint_module
  logical :: pass_cuts_check=.false.
end module

module weight_lines
  logical :: mc_H_only=.false.,mc_S_only=.false.
end module

module fxfx_delta_fixture
  implicit none
  include 'run.inc'
  double precision :: probability=1d0,born_sudakov=0.4d0,real_sudakov=0.7d0
  double precision :: records(15)=0d0,record_scale(15)=0d0,active_scale=99d0
  integer :: delta_calls=0,cluster_calls=0
  double precision :: f_r,f_s,f_c,f_dc,f_sc,f_dsc(4),f_b,f_nb
  common/factor_n1body/f_r,f_s,f_c,f_dc,f_sc,f_dsc
  common/factor_nbody/f_b,f_nb
  double precision :: fsS,fsH,fcS,fcH,fscS,fscH,fmcS,fmcH
  common/factor_n1body_NLOPS/fsS,fsH,fcS,fcH,fscS,fscH,fmcS,fmcH
  double precision :: fpd, fpp, fpl
  common/factor_pdfsch/fpd,fpp,fpl
contains
  subroutine reset_factors()
    f_r=1d0
    f_s=1d0
    f_c=1d0
    f_dc=1d0
    f_sc=1d0
    f_dsc=1d0
    f_b=1d0
    f_nb=1d0
    fsS=1d0
    fsH=1d0
    fcS=1d0
    fcH=1d0
    fscS=1d0
    fscH=1d0
    fmcS=1d0
    fmcH=1d0
    fpd=0d0
    fpp=0d0
    fpl=0d0
    records=0d0
    record_scale=0d0
    delta_calls=0
  end subroutine
end module

module mc_counterterms
  use fxfx_delta_fixture, only: probability,delta_calls
  implicit none
  double precision :: gfactsf,gfactcl,gfactazi
contains
  subroutine compute_MCsubtraction_kl(i,j,xi,y,p,p_cms,p_born,gfun,z,ncon,kernel)
    integer :: i,j,ncon
    double precision :: xi,y,p(0:3,5),p_cms(0:3,5),p_born(0:3,4),z(2),kernel(1,2)
    logical :: gfun
    ncon=2
    z=0.5d0
    kernel(1,:)=[2d0,3d0]/(xi**2*(1d0-y))
    gfactsf=0.4d0
    gfactcl=0.3d0
    gfactazi=0d0
  end subroutine
  subroutine compute_delta(p,probne)
    double precision :: p(0:3,5),probne
    probne=probability
    delta_calls=delta_calls+1
  end subroutine
  double precision function bogus_probne_fun(q)
    double precision :: q
    error stop 'physical Delta called artificial probability'
  end function
  double precision function get_qMC(xi,y)
    double precision :: xi,y
    get_qMC=1d0
  end function
end module

program check_fxfx_delta_weights
  use fxfx_delta_fixture
  use weight_lines
  use fks_phase_space_data
  implicit none
  include 'coupl.inc'
  integer :: i_fks,j_fks,nFKSprocess
  common/fks_indices/i_fks,j_fks
  common/c_nFKSprocess/nFKSprocess
  character*4 :: abrv
  common/to_abrv/abrv
  double precision :: xiScut_used,xiBSVcut_used
  common/cxiScut_used/xiScut_used,xiBSVcut_used
  double precision :: p(0:3,5),expected(15),probne,wb,wr,dp,mu,reference_total
  double precision, parameter :: probabilities(4)=[0d0,0.13d0,0.61d0,1d0]
  logical :: born_cuts,real_cuts,is_h(15)
  integer :: merge_mode,ip,mode,cuts,i,cases,clusters_before
  p=2d0
  g=1d0
  i_fks=5
  j_fks=1
  nFKSprocess=1
  abrv='all '
  xiScut_used=0.5d0
  xiBSVcut_used=1d0
  is_h=.false.
  is_h([1,8,9,10,13])=.true.
  cases=0
  do merge_mode=0,2
    ickkw=0
    if(merge_mode.gt.0)ickkw=3
    wb=1d0
    wr=1d0
    if(merge_mode.eq.1)then
      wb=0.4d0
      wr=0.7d0
    endif
    born_sudakov=wb
    real_sudakov=wr
    do cuts=0,3
      born_cuts=btest(cuts,0)
      real_cuts=btest(cuts,1)
      do ip=1,size(probabilities)
        probability=probabilities(ip)
        dp=1d0
        if(born_cuts)dp=probability
        do mode=0,2
          mc_H_only=mode.eq.1
          mc_S_only=mode.eq.2
          call reset_factors()
          if(ickkw.eq.3)then
            call set_FxFx_scale(0,p)
            call set_FxFx_scale(2,p1_cnt(:,:,0))
            call set_FxFx_scale(3,p)
          endif
          call compute_native_NLOPS_weights(p,p,born_cuts,real_cuts,probne)
          expected=0d0
          if(born_cuts)then
            ! R=10+xi+2y and G=(1-gsoft)*(1-gcoll) in collinear limits.
            expected(12)=dp*wb*5d0
            expected(13)=-dp*wr*5d0
            expected(8)=-dp*wr*0.6d0*11.6d0
            expected(9)=-dp*wr*0.42d0*12.3d0
            expected(10)=dp*wr*0.42d0*12d0
            expected(4)=wb*(dp*0.6d0-1d0)*11.6d0
            expected(5)=wb*(dp*0.42d0-1d0)*12.3d0
            expected(6)=wb*(1d0-dp*0.42d0)*12d0
          endif
          if(real_cuts)then
            expected(1)=dp*wr*11.8d0
            expected(11)=(1d0-dp)*wr*11.8d0
          endif
          if(mc_H_only)where(.not.is_h)expected=0d0
          if(mc_S_only)where(is_h)expected=0d0
          do i=1,15
            call close(records(i),expected(i),'combined contribution weight')
            if(abs(records(i)).lt.1d-12)cycle
            mu=99d0
            if(ickkw.eq.3)then
              mu=40d0
              if(i.ge.4.and.i.le.6)mu=10d0
            endif
            call close(record_scale(i),mu,'contribution scale')
          enddo
          call close(probne,dp,'native probability')
          if(delta_calls.ne.merge(1,0,born_cuts))error stop 'Delta evaluated more than once'
          if(mode.eq.0.and.cuts.eq.3)then
            call close(records(1)+records(11),wr*11.8d0,'real transfer preserves FxFx integrand')
            if(wb.eq.wr)then
              ! Identical merging factors must restore the undamped total
              ! for every Delta, including its zero/one endpoints.
              reference_total=wr*(11.8d0-11.6d0-12.3d0+12d0)
              call close(sum(records),reference_total,'Delta redistribution sum')
            endif
          endif
          cases=cases+1
        enddo
      enddo
    enddo
  enddo

  ! Reusing the outer real cache for another native H history must apply
  ! that history's Delta without reclustering or retaining its predecessor.
  ickkw=3
  born_sudakov=0.4d0
  real_sudakov=0.7d0
  call reset_factors()
  call set_FxFx_scale(0,p)
  call set_FxFx_scale(2,p1_cnt(:,:,0))
  call set_FxFx_scale(3,p)
  clusters_before=cluster_calls
  mc_H_only=.true.
  mc_S_only=.false.
  do ip=1,size(probabilities)
    probability=probabilities(ip)
    call reset_factors()
    call set_FxFx_scale(4,p)
    call compute_native_NLOPS_weights(p,p,.true.,.true.,probne)
    call close(records(1),probability*0.7d0*11.8d0,'cached H real weight')
    call close(records(13),-probability*0.7d0*5d0,'cached H MC weight')
    call close(sum(records([8,9,10])), &
         -probability*0.7d0*(0.6d0*11.6d0+0.42d0*(12.3d0-12d0)),'cached H G weight')
    if(delta_calls.ne.1)error stop 'native Delta was reused'
    if(cluster_calls.ne.clusters_before)error stop 'native H reclustered FxFx state'
  enddo
  print *, 'PASS combined FxFx Delta weights',cases,' cases'
contains
  subroutine close(actual,wanted,label)
    double precision :: actual,wanted
    character(*) :: label
    if(abs(actual-wanted).le.1d-11*max(1d0,abs(wanted)))return
    print *, 'FAIL ',label,actual,wanted,' mode/cuts/P=',mode,cuts,probability
    error stop 1
  end subroutine
end program

subroutine add_wgt(kind,orders,w1,w2,w3)
  use weight_lines
  use fxfx_delta_fixture
  implicit none
  integer :: kind,orders(1)
  double precision :: w1,w2,w3
  logical :: h
  h=kind.eq.1.or.kind.eq.13.or.(kind.ge.8.and.kind.le.10)
  if(mc_H_only.and..not.h)return
  if(mc_S_only.and.h)return
  records(kind)=records(kind)+w1
  record_scale(kind)=active_scale
end subroutine

subroutine cluster_and_reweight(iproc,sud,expanded,nren,ren,fac,matching,for_mc)
  use fxfx_delta_fixture
  implicit none
  integer :: iproc,nren,matching(5)
  double precision :: sud,expanded,ren(0:5),fac
  logical :: for_mc
  cluster_calls=cluster_calls+1
  nren=1
  sud=born_sudakov
  ren=10d0
  if(iproc.gt.0)then
    sud=real_sudakov
    ren=40d0
  endif
  expanded=-0.1d0
  fac=ren(1)
  matching=1
end subroutine

logical function momenta_equal(a,b)
  double precision :: a(0:3,5),b(0:3,5)
  momenta_equal=all(a.eq.b)
end function

subroutine set_alphaS(p)
  use fxfx_delta_fixture
  implicit none
  double precision :: p(0:3,5),fac(2),ren(0:5)
  integer :: nren
  common/c_FxFx_scales/fac,ren,nren
  active_scale=99d0
  if(ickkw.eq.3)active_scale=fac(1)
end subroutine

subroutine set_cms_stuff(i)
  integer :: i
end subroutine

subroutine include_multichannel_enhance(i)
  integer :: i
end subroutine

subroutine get_mc_lum(j,z,xi,lum,x)
  integer :: j
  double precision :: z,xi,lum,x(2)
  lum=1d0
end subroutine

double precision function fks_Sij(p,i,j,xi,y)
  integer :: i,j
  double precision :: p(0:3,5),xi,y
  fks_Sij=1d0
end function

double precision function fks_Hij(p,i,j)
  integer :: i,j
  double precision :: p(0:3,5)
  fks_Hij=1d0
end function

subroutine sreal(p,xi,y,ans)
  implicit none
  include 'orders.inc'
  double precision :: p(0:3,5),xi,y,ans
  amp_split=10d0+xi+2d0*y
  ans=amp_split(1)
end subroutine

subroutine sreal_deg(p,xi,ans1,ans2)
  implicit none
  double precision :: p(0:3,5),xi,ans1,ans2,deg_xi(1),deg_lxi(1),deg_muF(1),dis_p(1),dis_l(1),dis_d(1)
  common/to_amp_split_deg/deg_xi,deg_lxi,deg_muF
  common/to_amp_split_dis/dis_p,dis_l,dis_d
  deg_xi=0d0
  deg_lxi=0d0
  deg_muF=0d0
  dis_p=0d0
  dis_l=0d0
  dis_d=0d0
  ans1=0d0
  ans2=0d0
end subroutine

subroutine amp_split_pos_to_orders(i,orders)
  integer :: i,orders(1)
  orders=2
end subroutine

integer function get_orders_tag(orders)
  integer :: orders(1)
  get_orders_tag=orders(1)
end function
