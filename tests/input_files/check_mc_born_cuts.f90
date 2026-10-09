module cut_fixture
  implicit none
  integer, parameter :: nexternal=7
  integer :: ickkw,i_fks,j_fks,nFxFx_ren_scales
  integer :: need_matching_S(7),need_matching_H(7),need_matching_cuts(7)
  double precision :: FxFx_fac_scale(2),FxFx_ren_scales(0:7)
  common /test_run/ickkw
  common /fks_indices/i_fks,j_fks
  common /c_need_matching/need_matching_S,need_matching_H,need_matching_cuts
  common /c_FxFx_scales/FxFx_fac_scale,FxFx_ren_scales,nFxFx_ren_scales
  double precision :: ptj,ptgmin
  logical :: gamma_is_j
  common /test_cuts/ptj,ptgmin,gamma_is_j
  integer :: native_matching(7),pdg(7),cluster_calls=0
  double precision :: native_scales(0:2)=[100d0,30d0,40d0]
contains
  subroutine check_restored()
    if(any(need_matching_S.ne.77))error stop 'changed outer Born labels'
    if(any(need_matching_cuts.ne.need_matching_H))error stop 'changed real cut labels'
    if(any(FxFx_fac_scale.ne.[120d0,130d0]))error stop 'changed real factorisation scales'
    if(nFxFx_ren_scales.ne.1)error stop 'changed real scale count'
    if(any(FxFx_ren_scales(1:).ne.100d0))error stop 'changed real renormalisation scales'
  end subroutine
end module

program check_mc_born_cuts
  use cut_fixture
  implicit none
  double precision :: p(0:3,7),rwgt
  logical :: passcuts,passcuts_native_born
  external passcuts,passcuts_native_born
  ickkw=3
  ptj=8d0
  ptgmin=0d0
  gamma_is_j=.false.
  i_fks=6
  j_fks=2
  pdg=[2,-2,12,1,-11,-2,21]
  native_matching=[-99,-99,0,1,0,1,-99]
  need_matching_S=77
  need_matching_H=[-99,-99,0,-1,0,-1,1]
  need_matching_cuts=need_matching_H
  FxFx_fac_scale=[120d0,130d0]
  FxFx_ren_scales=100d0
  nFxFx_ren_scales=1
  p=0d0
  p(:,1)=[500d0,0d0,0d0,500d0]
  p(:,2)=[500d0,0d0,0d0,-500d0]
  p(:,4)=[40d0,40d0,0d0,0d0]
  p(:,7)=[0.0012d0,-0.0012d0,0d0,0d0]

  ! Regression: real EW labels omit both a quark and the zero FKS slot,
  ! so the old cut accepts an unresolved gluon in the native Born state.
  if(.not.passcuts(p,rwgt))error stop 'fixture does not reproduce missing Born cut'
  if(passcuts_native_born(p,rwgt))error stop 'accepted unresolved native Born gluon'
  call check_restored()

  ! Restoring only the zero slot would still miss a quark incorrectly
  ! classified as an EW decay product by the real-event clustering.
  need_matching_H(6)=1
  need_matching_cuts=need_matching_H
  p(:,4)=[0.0012d0,0.0012d0,0d0,0d0]
  p(:,7)=[40d0,-40d0,0d0,0d0]
  if(.not.passcuts(p,rwgt))error stop 'fixture does not reproduce stale EW label'
  if(passcuts_native_born(p,rwgt))error stop 'accepted unresolved native Born quark'
  call check_restored()

  p(:,4)=[30d0,30d0,0d0,0d0]
  if(.not.passcuts_native_born(p,rwgt))error stop 'rejected resolved native Born'
  call check_restored()
  native_scales(1)=7d0
  if(passcuts_native_born(p,rwgt))error stop 'ignored native Born clustering scale'
  call check_restored()
  native_scales(1)=30d0
  FxFx_ren_scales(0)=1d0
  if(passcuts(p,rwgt))error stop 'fixture does not reproduce failing real cut'
  if(.not.passcuts_native_born(p,rwgt))error stop 'applied real cut to native Born'
  if(FxFx_ren_scales(0).ne.1d0)error stop 'failed to restore real central scale'
  call check_restored()
  FxFx_ren_scales(0)=100d0

  ! A different FKS ordering must insert the zero slot in its own place.
  i_fks=7
  pdg(6:7)=[21,-2]
  p(:,6)=[0.0012d0,-0.0012d0,0d0,0d0]
  p(:,7)=0d0
  need_matching_H(6:7)=[1,-1]
  need_matching_cuts=need_matching_H
  if(passcuts_native_born(p,rwgt))error stop 'wrong zero-slot insertion'
  call check_restored()
  if(cluster_calls.ne.6)error stop 'incorrect number of native clusterings'

  ickkw=0
  if(.not.passcuts_native_born(p,rwgt))error stop 'changed non-FxFx cut path'
  if(cluster_calls.ne.6)error stop 'clustered non-FxFx point'
  call check_restored()
  if(rwgt.ne.0.375d0)error stop 'lost user cut weight'
  write(*,*)'PASS native Born cuts'
end program

subroutine cluster_and_reweight(iproc,sudakov,expanded,nscales,ren,fac,matching,scale_only)
  use cut_fixture
  implicit none
  integer :: iproc,nscales,matching(7)
  double precision :: sudakov,expanded,ren(0:7),fac
  logical :: scale_only
  if(iproc.ne.0)error stop 'cuts requested real clustering'
  if(.not.scale_only)error stop 'cuts computed an unnecessary Sudakov'
  cluster_calls=cluster_calls+1
  matching=native_matching
  nscales=2
  ren=0d0
  ren(0:2)=native_scales
  fac=30d0
  sudakov=0.456d0
  expanded=-1d0
end subroutine

logical function passcuts(p,rwgt)
  use cut_fixture
  implicit none
  double precision :: p(0:3,7),rwgt,pp(0:4,7),pqcd(0:3,7)
  integer :: status(7),nqcd
  logical :: is_iso(7),is_a_j(7),passcuts_fxfx
  external passcuts_fxfx
  rwgt=0.375d0
  if(ickkw.ne.3)then
    passcuts=p(0,1).gt.0d0
    return
  endif
  pp(0:3,:)=p
  pp(4,:)=0d0
  status=[-1,-1,1,1,1,1,1]
  is_iso=.true.
  call identify_QCD_partons(is_iso,pp,status,pdg,is_a_j,pqcd,nqcd)
  passcuts=passcuts_fxfx(pp,pqcd,nqcd)
end function
