module controlled_starting_scales
  implicit none
  double precision :: random_value=0.5d0,hard_reference=1000d0
  integer :: random_calls=0
end module

program check_pythia8_s_scales
  use controlled_starting_scales
  use process_module
  use kinematics_module
  use scale_module
  use weight_lines
  implicit none
  character(len=32) :: mode
  double precision :: p(0:3,4),saved(4,4),saved_hard,q,expected,observed
  double precision, external :: compute_damping_weight
  integer :: i,nabove,selected_fold
  logical :: delta

  call get_command_argument(1,mode)
  delta=trim(mode).eq.'delta'
  call init_process_module_global('PYTHIA8   ','all ',5,2,delta,13000d0,2,1,0)
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
  call compute_shower_scale_nbody(p,1)
  call close(shower_scale_hard,550d0,'damped hard scale')
  call close(shower_scale_nbody(1,2),550d0,'ISR uses SCALUP')
  call close(shower_scale_nbody(3,4),500d0,'FSR cap applied after hard-scale damping')
  call require(shower_scale_nbody(1,3).eq.-1d0,'selected flow mask')
  call require(random_calls.eq.1,'one damping draw per event')
  call save_shower_scale_nbody(1,1)
  if (delta) then
    call require(all(emsca_S(1,1,:,:).eq.shower_scale_nbody),'Delta Born fallback retains matrix')
  else
    call close(emsca_S(1,1,1,1),550d0,'ordinary Born fallback uses scalar hard scale')
  endif

  random_value=0.2d0
  call compute_shower_scale_nbody(p,-3)
  call close(shower_scale_hard,400d0,'second damping draw')
  call close(shower_scale_nbody(1,3),400d0,'IF hard scale')
  call close(shower_scale_nbody(3,1),sqrt(50000d0),'FI kinematic limit')
  call close(shower_scale_nbody(3,4),400d0,'FF respects hard scale')
  call save_shower_scale_nbody(2,2,4)
  saved=shower_scale_nbody
  saved_hard=shower_scale_hard

  ! The damping weight is the survival probability of the SAME hard-scale
  ! draw. A physical dipole cap truncates this distribution, creating an
  ! endpoint probability; it must not rescale the damping interval.
  q=400d0
  ileg=1
  xtk=-q*q
  expected=compute_damping_weight(4,1d0,0d0)
  call close(expected,0.8d0,'hard-scale damping in subtraction')
  nabove=0
  do i=1,1000
    random_value=(dble(i)-0.5d0)/1000d0
    call compute_shower_scale_nbody(p,-3)
    if (shower_scale_nbody(3,4).gt.q) nabove=nabove+1
    call require(shower_scale_nbody(3,4).le.500d0,'FSR endpoint')
    call require(shower_scale_nbody(3,4).le.shower_scale_hard,'global shower ordering')
  enddo
  observed=dble(nabove)/1000d0
  call close(observed,expected,'sampled shower and subtraction damping agree')

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
  call compute_shower_scale_nbody(p,-3)
  call close(shower_scale_hard,1100d0,'hard-scale variation')
  call close(shower_scale_nbody(3,4),500d0,'variation cannot exceed FF phase space')
  hard_reference=0.01d0
  call compute_shower_scale_nbody(p,-3)
  call close(shower_scale_hard,4.5d0,'existing infrared floor and width')
  call require(all(pack(shower_scale_nbody,shower_scale_nbody.ge.0d0).le.shower_scale_hard), &
       'infrared scales respect SCALUP')

  ! FxFx continues to use its own undamped clustering prescription.
  hard_reference=1000d0
  ickkw_mod=3
  call compute_shower_scale_nbody(p,-3)
  call require(all(shower_scale_nbody.eq.2000d0),'FxFx scale prescription')
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
