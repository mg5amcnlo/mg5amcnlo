! Controlled process inputs for the production fixed-order driver and FxFx cache.
! Born and real clusterings have distinct renormalisation and factorisation scales.
module fks_phase_space_data
  implicit none
  double precision :: p1_cnt(0:3,7,0:2)=1d0,jac_cnt(0:2)=1d0,p_born(0:3,6)=1d0
end module

module fo_fixture
  implicit none
  integer :: point=1,active_cms=0,nrecords=0
  double precision :: mu(2)=0d0,record_mu(2,6)=0d0
  integer :: record_kind(6)=0
  logical :: born_cuts=.true.,real_cuts=.true.,coll_cuts=.true.
contains
  subroutine record(kind)
    integer :: kind
    nrecords=nrecords+1
    record_kind(nrecords)=kind
    record_mu(:,nrecords)=mu
  end subroutine
end module

module fks_phase_space
  use fks_phase_space_data
  implicit none
  type born_fixture
    logical :: valid=.false.
  end type
  type fks_phase_space_point
    double precision :: p(0:3,7)=1d0
    type(born_fixture) :: born
  end type
contains
  subroutine generate_born_contribution(ndim,config,jac,x,phase_space)
    use fo_fixture, only: point
    integer :: ndim,config
    double precision :: jac,x(99)
    type(fks_phase_space_point) :: phase_space
    phase_space%p=dble(point)
    p1_cnt=dble(point)
    p_born=dble(point)
  end subroutine
  subroutine generate_real_phase_space(ndim,config,jac,x,phase_space)
    integer :: ndim,config
    double precision :: jac,x(99)
    type(fks_phase_space_point) :: phase_space
    call generate_born_contribution(ndim,config,jac,x,phase_space)
  end subroutine
end module

module weight_lines
  integer :: icontr=0
end module

module extra_weights
  logical :: doreweight=.false.
  double precision :: H1_factor_virt=0d0
end module

module mint_module
  implicit none
  integer,parameter :: ndimmax=5,nintegrals=1
  integer :: iconfig=1,ichan=1
  logical :: new_point=.true.,pass_cuts_check=.false.
  double precision :: virt_wgt_mint(0:1)=0d0,born_wgt_mint(0:1)=0d0
end module

program check_fo_scales
  use fo_fixture
  use mint_module
  implicit none
  include 'run.inc'
  integer :: ini_fin_fks(1),nndim,i_fks,j_fks,nfksprocess,i,mode,ip,ie,merge_mode,n_expected
  common/fks_channels/ini_fin_fks
  common/tosigint/nndim
  common/fks_indices/i_fks,j_fks
  common/c_nfksprocess/nfksprocess
  character*4 :: abrv
  common/to_abrv/abrv
  logical :: pineappl,use_evpr
  common/for_pineappl/pineappl
  common/to_use_evpr/use_evpr
  double precision :: sigint,answer,f(1),xx(ndimmax),expected
  external sigint
  ini_fin_fks=1
  nndim=5
  i_fks=7
  j_fks=1
  nfksprocess=1
  abrv='all '
  pineappl=.false.
  use_evpr=.true.
  do_rwgt_scale=.false.
  do_rwgt_pdf=.false.
  xx=0.5d0
  do ie=0,1
    use_evpr=ie.eq.1
    do mode=0,7
      born_cuts=btest(mode,0)
      real_cuts=btest(mode,1)
      coll_cuts=btest(mode,2)
      do ip=1,2
        point=ip
        do merge_mode=0,1
          ickkw=3*merge_mode
          nrecords=0
          answer=sigint(xx,1d0,0,f)
          do i=1,nrecords
            expected=99d0
            if (ickkw.eq.3) then
              expected=10d0*point
              if (record_kind(i).eq.6) expected=40d0*point
            endif
            if (maxval(abs(record_mu(:,i)-expected)).gt.1d-12) then
              print *, 'Wrong contribution scales:',record_kind(i),record_mu(:,i),'expected',expected
              error stop 'fixed-order FxFx contribution scales'
            endif
          enddo
          n_expected=4*merge(1,0,born_cuts)+merge(1,0,real_cuts) &
                     +merge(1,0,coll_cuts.or.(use_evpr.and.born_cuts))
          if (nrecords.ne.n_expected) error stop 'wrong records'
        enddo
      enddo
    enddo
  enddo
  print *, 'PASS fixed-order FxFx counterevent scales'
end program

subroutine cluster_and_reweight(iproc,sud,expanded,nren,ren,fac,matching,for_mc)
  use fo_fixture
  implicit none
  integer :: iproc,nren,matching(7)
  double precision :: sud,expanded,ren(0:7),fac
  logical :: for_mc
  nren=1
  sud=0.8d0
  expanded=-0.2d0
  ren=0d0
  ren(0)=80d0
  ren(1)=10d0*point
  if (iproc.gt.0) ren(1)=40d0*point
  fac=ren(1)
  matching=1
end subroutine

logical function momenta_equal(a,b)
  implicit none
  double precision :: a(0:3,7),b(0:3,7)
  momenta_equal=all(a.eq.b)
end function

subroutine set_alphaS(p)
  use fo_fixture
  implicit none
  double precision :: p(0:3,7),fac(2),ren(0:7)
  integer :: nren
  common/c_FxFx_scales/fac,ren,nren
  include 'run.inc'
  mu=99d0
  if (ickkw.eq.3) mu=[ren(1),fac(1)]
end subroutine

subroutine set_cms_stuff(which)
  use fo_fixture
  integer :: which
  active_cms=which
end subroutine

logical function passcuts(p,rwgt)
  use fo_fixture
  double precision :: p(0:3,7),rwgt
  rwgt=1d0
  passcuts=born_cuts
  if (active_cms.eq.-100) passcuts=real_cuts
  if (active_cms.eq.1) passcuts=coll_cuts
end function

subroutine compute_born
  use fo_fixture
  call record(1)
end subroutine
subroutine compute_nbody_noborn
  use fo_fixture
  call record(2)
end subroutine
subroutine compute_soft_counter_term(replace)
  use fo_fixture
  double precision :: replace
  call record(3)
end subroutine
subroutine compute_soft_collinear_counter_term(replace)
  use fo_fixture
  double precision :: replace
  call record(4)
end subroutine
subroutine compute_collinear_counter_term(replace)
  use fo_fixture
  double precision :: replace
  call record(5)
end subroutine
subroutine compute_real_emission(p,prob)
  use fo_fixture
  double precision :: p(0:3,7),prob
  call record(6)
end subroutine

subroutine setup_ini_fin_fks_map(map)
  integer :: map(0:2,0:1)
  map=1
end subroutine
subroutine get_MC_integer(index,nmax,picked,vol)
  integer :: index,nmax,picked
  double precision :: vol
  picked=1
  vol=1d0
end subroutine
subroutine get_born_nfksprocess(a,b)
  integer :: a,b
  b=a
end subroutine
subroutine update_vegas_x(xx,x)
  double precision :: xx(5),x(99)
  x=0.5d0
end subroutine
subroutine get_wgt_nbody(sig)
  double precision :: sig
  sig=1d0
end subroutine
subroutine get_wgt_no_nbody(sig)
  double precision :: sig
  sig=1d0
end subroutine
subroutine fill_mint_function(f)
  double precision :: f(1)
  f=1d0
end subroutine
