module controlled_starting_scales
  implicit none
  double precision :: random_value=0.5d0,hard_reference=1000d0
  integer :: random_calls=0
end module

program check_pythia8_s_scales
  use mc_counterterms, only: compute_damping_weight,mc_shower_scale_mass, &
       ileg,fksfather,xtk,shat_n1
  use controlled_starting_scales
  use process_module
  use scale_module
  use weight_lines
  implicit none
  character(len=32) :: mode
  double precision :: p(0:3,4),saved(4,4),saved_hard,q,expected,observed,ff_cap,fi_cap
  double precision :: thresholds(5)=[19d0,20d0,21.5d0,23d0,24d0]
  double precision :: damping_weights(5)=[1d0,1d0,0.5d0,0d0,0d0]
  integer :: i,j,nabove,selected_fold,calls_before
  logical :: delta,herwig

  call get_command_argument(1,mode)
  delta=trim(mode).eq.'delta'
  herwig=trim(mode).eq.'herwig'
  call init_process_module_global('PYTHIA8   ','all ',5,2,delta,13000d0,2,1,0)
  if (herwig) shower_mc_mod='HERWIG7'
  ff_cap=500d0
  fi_cap=sqrt(50000d0)
  if (herwig) then
    ff_cap=250d0
    fi_cap=sqrt(200000d0)/4d0
  endif
  call init_scale_module(5,1d0,2,2)
  next_n=4
  mass_n=0d0
  fksfather=3
  valid_dipole_n=.false.
  valid_dipole_n(1,2,1)=.true.
  valid_dipole_n(2,1,1)=.true.
  valid_dipole_n(3,4,1)=.true.
  valid_dipole_n(4,3,1)=.true.
  valid_dipole_n(1,3,2)=.true.
  valid_dipole_n(3,1,2)=.true.
  valid_dipole_n(2,4,2)=.true.
  valid_dipole_n(4,2,2)=.true.
  p(:,1)=[500d0,0d0,0d0,500d0]
  p(:,2)=[500d0,0d0,0d0,-500d0]
  p(:,3)=[500d0,400d0,0d0,300d0]
  p(:,4)=[500d0,-400d0,0d0,-300d0]
  call compute_shower_scale_nbody(p,1,mc_shower_scale_mass())
  call close(shower_scale_hard,550d0,'damped hard scale')
  call close(shower_scale_nbody(1,2),550d0,'ISR uses SCALUP')
  call close(shower_scale_nbody(3,4),ff_cap,'FSR cap applied after hard-scale damping')
  call require(shower_scale_nbody(1,3).eq.-1d0,'selected flow mask')
  call require(random_calls.eq.1,'one damping draw per event')
  call save_shower_scale_nbody(1,1,fksfather)
  if (delta) then
    call require(all(emsca_S(1,1,:,:).eq.shower_scale_nbody),'Delta Born fallback retains matrix')
  else
    call close(emsca_S(1,1,1,1),550d0,'ordinary Born fallback uses scalar hard scale')
  endif

  random_value=0.2d0
  call compute_shower_scale_nbody(p,-3,mc_shower_scale_mass())
  call close(shower_scale_hard,400d0,'second damping draw')
  call close(shower_scale_nbody(1,3),400d0,'IF hard scale')
  call close(shower_scale_nbody(3,1),fi_cap,'FI kinematic limit')
  call close(shower_scale_nbody(3,4),min(400d0,ff_cap),'FF respects hard scale')
  call save_shower_scale_nbody(2,2,fksfather,4)
  saved=shower_scale_nbody
  saved_hard=shower_scale_hard

  ! The damping weight is the survival probability of the SAME hard-scale
  ! draw. A physical dipole cap truncates this distribution, creating an
  ! endpoint probability; it must not rescale the damping interval.
  q=400d0
  if (herwig) q=200d0
  ileg=1
  shat_n1=1000d0**2
  xtk=-500d0*q
  expected=compute_damping_weight(4,q/500d0,0d0)
  if (herwig) then
    call close(expected,64d0/65d0,'Herwig hard-scale damping in subtraction')
  else
    call close(expected,0.8d0,'hard-scale damping in subtraction')
  endif
  nabove=0
  do i=1,1000
    random_value=(dble(i)-0.5d0)/1000d0
    call compute_shower_scale_nbody(p,-3,mc_shower_scale_mass())
    if (shower_scale_nbody(3,4).gt.q) nabove=nabove+1
    call require(shower_scale_nbody(3,4).le.ff_cap,'FSR endpoint')
    call require(shower_scale_nbody(3,4).le.shower_scale_hard,'global shower ordering')
  enddo
  observed=dble(nabove)/1000d0
  call require(abs(observed-expected).le.0.5d0/1000d0, &
       'sampled shower and subtraction damping agree within quantile spacing')

  ! Later native/history calculations must not replace the chosen saved
  ! scale. Exercise the production fold/sector selection with one owner.
  call weight_lines_allocated(5,1,1,1)
  icontr=1
  H_event=.false.
  group_size=0
  call add_group_member(1,1)
  call pack_contribution_groups
  ifold_cnt=2
  nFKS=2
  itype=5
  wgts=1d0
  random_value=0.5d0
  call update_shower_scale_Sevents_v2(2,selected_fold)
  call require(selected_fold.eq.2,'selected S-event fold')
  call close(showerscaleS_hard,saved_hard,'selected scalar belongs to same sector and fold')
  if (delta) then
    call require(all(showerscaleS.eq.saved),'selected Delta matrix belongs to same fold')
  else
    call close(showerscaleS(1,1),saved_hard,'ordinary scalar independent of partner')
  endif
  itype=2
  nFKS=1
  ifold_cnt=1
  call update_shower_scale_Sevents_v2(2,selected_fold)
  call require(selected_fold.eq.1,'Born-only fallback fold')
  call close(showerscaleS_hard,550d0,'Born-only fallback hard scale')
  if (delta) then
    call close(showerscaleS(3,4),500d0,'Born-only fallback retains the FF cap')
    call require(showerscaleS(1,3).eq.-1d0,'Born-only fallback retains absent entries')
  endif

  ! Shower-scale variations, including upward variations, retain physical
  ! dipole bounds. Very small hard scales retain the existing infrared floor.
  call init_scale_module(5,2d0,2,2)
  call compute_shower_scale_nbody(p,-3,mc_shower_scale_mass())
  call close(shower_scale_hard,1100d0,'hard-scale variation')
  call close(shower_scale_nbody(3,4),ff_cap,'variation cannot exceed FF phase space')
  if (herwig) then
    call close(shower_scale_nbody(1,2),1000d0,'Herwig II angular ceiling')
    call close(shower_scale_nbody(1,3),sqrt(200000d0),'Herwig IF angular ceiling')
    ! A standalone Born event uses the undamped hard scale and its caps.
    call Bornonly_shower_scale(p,1)
    call close(shower_scale_hard,2000d0,'Herwig Born hard scale')
    call close(shower_scale_nbody(3,4),ff_cap,'Herwig Born FF ceiling')
    call require(all(shower_scale_nbody_max.eq.-1d0),'Born has no damping interval')
    mcatnlo_delta_mod=.true.
    call require(.not.herwig7_S_scales(),'Herwig prescription excludes Delta')
    mcatnlo_delta_mod=.false.
    nincoming_mod=1
    call require(.not.herwig7_S_scales(),'Herwig prescription excludes decays')
    nincoming_mod=2
    shower_mc_mod='HERWIG6'
    call require(.not.scalar_S_scales(),'HERWIG6 retains its prescription')
    shower_mc_mod='HERWIG7'
  endif
  hard_reference=0.01d0
  call compute_shower_scale_nbody(p,-3,mc_shower_scale_mass())
  call close(shower_scale_hard,4.5d0,'existing infrared floor and width')
  call require(all(pack(shower_scale_nbody,shower_scale_nbody.ge.0d0).le.shower_scale_hard), &
       'infrared scales respect SCALUP')

  ! FxFx continues to use its own undamped clustering prescription.
  hard_reference=1000d0
  ickkw_mod=3
  calls_before=random_calls
  call compute_shower_scale_nbody(p,-3,mc_shower_scale_mass())
  if (delta) then
    call close(shower_scale_nbody(1,2),2000d0,'FxFx Delta ISR hard scale')
    call close(shower_scale_nbody(1,3),2000d0,'FxFx Delta IF hard scale')
    call close(shower_scale_nbody(3,1),fi_cap,'FxFx Delta directed FI cap')
    call close(shower_scale_nbody(3,4),ff_cap,'FxFx Delta FF cap')
    call require(shower_scale_nbody(1,1).eq.-1d0,'FxFx Delta diagonal is absent')
    call require(shower_scale_nbody(1,4).eq.-1d0,'FxFx Delta absent colour connection')
  else
    call require(all(shower_scale_nbody.eq.2000d0),'FxFx scale prescription')
  endif
  call close(shower_scale_hard,2000d0,'FxFx undamped hard scale')
  call require(all(pack(shower_scale_nbody_min,shower_scale_nbody.ge.0d0).eq.2000d0), &
       'FxFx narrow lower bounds')
  call require(all(pack(shower_scale_nbody_max,shower_scale_nbody.ge.0d0).eq.2003d0), &
       'FxFx narrow upper bounds')
  call require(random_calls.eq.calls_before,'FxFx scale setting needs no damping draw')
  if (delta) then
    call Bornonly_shower_scale(p,1)
    call close(shower_scale_hard,2000d0,'FxFx Delta Born hard scale')
    call close(shower_scale_nbody(3,4),ff_cap,'FxFx Delta Born FF cap')
    call require(shower_scale_nbody(1,3).eq.-1d0,'FxFx Delta Born selected flow mask')
    call require(all(shower_scale_nbody_min.eq.-1d0).and.all(shower_scale_nbody_max.eq.-1d0), &
         'FxFx Delta Born has no damping interval')
    call require(random_calls.eq.calls_before,'FxFx Delta Born needs no random draw')
  endif

  ! At the merging boundary, the scale stays at Q for every RNG value.
  ! The subtraction retains its old three-GeV transition above Q.
  call init_scale_module(5,1d0,2,2)
  hard_reference=20d0
  do i=1,3
    random_value=dble(i)/4d0
    calls_before=random_calls
    call compute_shower_scale_nbody(p,-3,mc_shower_scale_mass())
    call close(shower_scale_hard,20d0,'FxFx scale at the merging boundary')
    call require(all(pack(shower_scale_nbody,shower_scale_nbody.ge.0d0).eq.20d0), &
         'FxFx scalar matrix at boundary')
    call require(random_calls.eq.calls_before,'FxFx scale is independent of RNG')
  enddo
  do j=1,size(thresholds)
    q=thresholds(j)
    ileg=1
    shat_n1=4000d0**2
    xtk=-2000d0*q
    observed=compute_damping_weight(4,q/2000d0,0d0)
    call close(observed,damping_weights(j),'FxFx narrow subtraction transition')
  enddo

  call save_shower_scale_nbody(1,1,fksfather)
  call save_shower_scale_nbody(2,2,fksfather,4)
  saved=shower_scale_nbody
  saved_hard=shower_scale_hard
  shower_scale_n1body=222d0
  emsca_H=333d0
  calls_before=random_calls
  call set_FxFx_shower_scale_nbody(100d0,p,-3)
  call close(shower_scale_hard,100d0,'native FxFx undamped clustering reference')
  call require(all(pack(shower_scale_nbody_min,shower_scale_nbody.ge.0d0).eq.100d0), &
       'native FxFx lower bounds')
  call require(all(pack(shower_scale_nbody_max,shower_scale_nbody.ge.0d0).eq.103d0), &
       'native FxFx upper bounds')
  call require(random_calls.eq.calls_before,'native FxFx scale needs no damping draw')
  call require(all(shower_scale_n1body.eq.222d0).and.all(emsca_H.eq.333d0), &
       'native FxFx scale leaves H-event matrices intact')
  itype=5
  nFKS=2
  ifold_cnt=2
  call update_shower_scale_Sevents_v2(2,selected_fold)
  call require(selected_fold.eq.2,'FxFx selected S-event fold')
  call close(showerscaleS_hard,saved_hard,'saved FxFx hard scale survives native histories')
  if (delta) then
    call require(all(showerscaleS.eq.saved),'saved FxFx Delta matrix survives native histories')
  else
    call close(showerscaleS(1,1),saved_hard,'saved FxFx scalar survives native histories')
  endif
  print *, 'PASS '//trim(mode)
contains
  subroutine require(condition,label)
    logical, intent(in) :: condition
    character(*), intent(in) :: label
    if (condition) return
    print *, 'FAIL: '//label
    stop 1
  end subroutine
  subroutine close(actual,wanted,label)
    double precision, intent(in) :: actual,wanted
    character(*), intent(in) :: label
    call require(abs(actual-wanted).lt.1d-10*max(1d0,abs(wanted)),label)
  end subroutine
end program

double precision function ran2()
  use controlled_starting_scales
  random_calls=random_calls+1
  ran2=random_value
end function

subroutine cluster_and_reweight(iproc,sudakov,reweight,nscales,scales,fac,matching,for_shower)
  use controlled_starting_scales
  implicit none
  integer :: iproc,nscales,matching(*)
  double precision :: sudakov,reweight,scales(0:*),fac(*)
  logical :: for_shower
  nscales=0
  scales(0)=hard_reference
end subroutine
