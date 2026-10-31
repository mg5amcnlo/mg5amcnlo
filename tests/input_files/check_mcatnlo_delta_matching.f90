! Exercise the production compute_delta routine with controlled Born/real
! records and a simple Sudakov table. There is no shower runtime interface.
program check_mcatnlo_delta_matching
  use mc_counterterms, only: compute_delta,delta_native_ir_probability,delta_common_probability,isspecial
  use FKSParams, only: delta_common_scale
  use fks_phase_space_data, only: p1_cnt
  use process_module
  use scale_module
  implicit none
  integer :: idup(4,1),mothup(2,4,1),icolup(2,4,1)
  common /test_born/ idup,mothup,icolup
  integer :: i_fks,j_fks,nfksprocess,mccntcalled,fold,ifold_counter,lpp(2)
  common /fks_indices/ i_fks,j_fks
  common /c_nfksprocess/ nfksprocess
  common /c_MCcntcalled/ mccntcalled
  common /cfl/ fold,ifold_counter
  common /test_lpp/ lpp
  double precision :: seen_scales(4,4)
  logical*1 :: seen_dead(4,4)
  common /test_stopping/ seen_scales,seen_dead
  integer :: table_calls
  common /test_table_calls/ table_calls
  integer :: mass_calls(4:5)
  common /test_mass_calls/ mass_calls
  double precision :: p(0:3,5),prob,expected,t34,t43
  double precision :: history_t(16),survival(2)
  double precision :: start(4,4),source_prob,native_scale
  common /test_native_scale/ native_scale
  double precision :: fallback(5,5),fallback_original(5,5)
  double precision :: epsilon,previous
  integer :: history_count,rng_calls,point
  common /test_rng/ rng_calls
  logical :: ir_test,gluon_test,globalfsr_test
  common /test_globalfsr/ globalfsr_test
  common /test_ir/ ir_test
  common /test_gluon/ gluon_test
  character(len=32) :: mode

  call get_command_argument(1,mode)
  globalfsr_test=trim(mode)=='native_global_fsr'
  gluon_test=trim(mode)=='common_gluon'
  ir_test=trim(mode)=='common_ir'.or.gluon_test
  call init_process_module_global('PYTHIA8   ','born',5,2,.true.,1000d0,1,1,0)
  call init_scale_module(5,1d0,1,1)
  born_flow_picked=1
  i_fks=5
  j_fks=3
  call RealToBornMapping(i_fks)
  nfksprocess=1
  ifold_counter=1
  fold=0
  lpp=0
  isspecial=.false.
  table_calls=0
  mass_calls=0
  mccntcalled=0
  valid_dipole_n=.false.
  valid_dipole_n(3,4,1)=.true.
  valid_dipole_n(4,3,1)=.true.
  valid_dipole_n1=.false.
  idup(:,1)=(/11,-11,1,-1/)
  mothup=0
  icolup=0
  icolup(1,3,1)=501
  icolup(2,4,1)=501
  if (trim(mode) == 'invalid') icolup(2,4,1)=999
  shower_scale_nbody=50d0
  p(:,1)=(/500d0,0d0,0d0,500d0/)
  p(:,2)=(/500d0,0d0,0d0,-500d0/)
  p(:,3)=(/40d0,40d0,0d0,0d0/)
  p(:,4)=(/30d0,-30d0,0d0,0d0/)
  p(:,5)=(/30d0,0d0,30d0,0d0/)
  p1_cnt(:,1:4,0)=p(:,1:4)
  rng_calls=0
  native_scale=30d0

  if(trim(mode)=='common_hard')then
    ! The rare ttbar real-only packet had a common global maximum but
    ! unequal active dipole starts. No passing Born means no Delta call.
    fallback=-1d0
    fallback(1,2)=656.341705d0
    fallback(2,1)=624.385822d0
    fallback(3,5)=217.5d0
    fallback(5,3)=0d0
    fallback_original=fallback
    emsca_H(1,1,:,:)=fallback
    call common_H_hard_scale(emsca_H(1,1,:,:))
    fallback=emsca_H(1,1,:,:)
    call require(all(pack(fallback,fallback_original>0d0)==656.341705d0), &
                 'real-only hard fallback uses its original global maximum')
    call require(all(pack(fallback,fallback_original<=0d0)== &
                     pack(fallback_original,fallback_original<=0d0)), &
                 'hard fallback preserves zero stops and absent dipoles')
    call require(mccntcalled==0.and.rng_calls==0.and.table_calls==0, &
                 'hard fallback needs no Born, Sudakov, random draw or weight evaluation')
    call common_H_hard_scale(fallback)
    call require(all(fallback==emsca_H(1,1,:,:)),'common hard fallback is idempotent')
    fallback=-1d0
    fallback(1,2)=0d0
    fallback_original=fallback
    call common_H_hard_scale(fallback)
    call require(all(fallback==fallback_original),'an inactive packet remains inactive')
    print *, 'PASS real-only common H hard fallback'
    stop
  endif

  if(globalfsr_test)then
    ! Stock-Pythia accepted top-radiation endpoint from the ttbar1M
    ! validation, source-reservoir event1297. The native global-recoil
    ! evolution scale is valid; its local top/beam inverse is a dead zone.
    call compute_delta(p,prob) ! initialize the controlled Sudakov table
    idup(:,1)=[21,21,6,-6]
    p(:,1)=[1465.8327828863087d0,0d0,0d0,1465.8327828863087d0]
    p(:,2)=[501.09881476158785d0,0d0,0d0,-501.09881476158785d0]
    p(:,3)=[1444.4170454268033d0,41.69736059220277d0,-.7625903539854892d0,1433.4128329602372d0]
    p(:,4)=[505.7054951741191d0,-32.05592594680296d0,-11.953191972783538d0,-473.9605328111784d0]
    p(:,5)=[16.809057046973855d0,-9.641434645399803d0,12.715782326769023d0,5.281667975662063d0]
    p1_cnt(:,1,0)=[1465.832782886309d0,0d0,0d0,1465.832782886309d0]
    p1_cnt(:,2,0)=p(:,2)
    p1_cnt(:,3,0)=[1455.5010496837672d0,32.45026001439164d0,12.100233453323458d0,1444.768102709193d0]
    p1_cnt(:,4,0)=[511.43054796412946d0,-32.45026001439164d0,-12.100233453323458d0,-480.03413458447216d0]
    native_scale=34.131440508948806d0
    valid_dipole_n=.false.
    valid_dipole_n(3,1,1)=.true.
    shower_scale_nbody=-1d0
    shower_scale_nbody(3,1)=88.69246172113849d0
    call delta_native_ir_probability(p,p1_cnt(:,1:4,0),[21,21,6,-6,21], &
                             shower_scale_nbody,native_scale,source_prob,history_count,history_t)
    call require(history_count==0,'actual native top endpoint has no supported local inverse')
    call require(abs(source_prob-native_scale/shower_scale_nbody(3,1))<1d-14, &
                 'native global-FSR all-leg factor survives the local dead-zone sentinel')
    print *, 'PASS native global-FSR independent of local inverse'
    stop
  endif

  if(ir_test)then
    delta_common_scale=.true.
    idup(:,1)=[1,-1,11,-11]
    icolup=0
    icolup(1,1,1)=501
    icolup(2,2,1)=501
    valid_dipole_n=.false.
    valid_dipole_n(1,2,1)=.true.
    valid_dipole_n(2,1,1)=.true.
    j_fks=1
    shower_scale_nbody=300d0
    if(gluon_test)then
      idup(1:2,1)=21
      icolup(:,1,1)=[501,502]
      icolup(:,2,1)=[502,501]
      ! Deliberately leave the legacy global flag false. The common-scale
      ! source must count the two actual colour ends from this Born flow.
      isspecial=.false.
      p(:,5)=[100d0,80d0,0d0,60d0]
      native_scale=200d0
      call compute_delta(p,prob)
      call delta_native_ir_probability(p,p1_cnt(:,1:4,0),[21,21,11,-11,21], &
                               shower_scale_nbody,native_scale,source_prob,history_count,history_t)
      call require(history_count==2,'two gluon radiators, each sharing both colour ends with its partner')
      call require(maxval(abs(history_t(1:2)**2-[8000d0,32000d0]))<1d-8,'gluon Lund scales')
      survival=([8000d0,32000d0]/300d0**2)**2
      expected=minval(survival)
      call require(abs(prob-expected)<1d-12.and.abs(source_prob-expected)<1d-12, &
                   'both gluons contribute both colour ends: all-leg survival is (t/Q)^4')
      call require(rng_calls==0,'native gluon continuation requires no history draw')
      call require(.not.any(isspecial),'common source_prob does not mutate legacy special-flow flags')
      source_prob=delta_common_probability(p1_cnt(:,1:4,0),shower_scale_nbody,nearest(300d0,-1d0))
      call require(source_prob>.999d0.and.source_prob<=1d0,'rounded unit two-line ratios remain finite')
      print *, 'PASS common-scale Delta gluon colour multiplicities'
      stop
    endif
    j_fks=2 ! The OTHER valid history becomes collinear while native t_c stays finite.
    previous=1d0
    do point=1,4
      select case(point)
      case(1)
        epsilon=1d-2
      case(2)
        epsilon=1d-6
      case(3)
        epsilon=1d-12
      case(4)
        epsilon=0d0
      end select
      p(:,5)=[100d0,100d0*sqrt(max(0d0,2d0*epsilon-epsilon**2)),0d0,100d0*(1d0-epsilon)]
      mccntcalled=0
      call compute_delta(p,prob)
      call delta_native_ir_probability(p,p1_cnt(:,1:4,0),[1,-1,11,-11,21], &
                               shower_scale_nbody,native_scale,source_prob,history_count,history_t)
      call require(history_count==2,'both ISR histories retained into the deep-collinear limit')
      call require(abs(source_prob-prob)<1d-14,'source and explicit source_prob agree in the collinear limit')
      call require(prob<=previous,'common-scale Delta decreases toward collinearity')
      if(point>=3)then
        call require(history_t(1)==1d-6,'native tiny-virtuality sentinel retained')
        call require(prob==0d0,'a zero alternative Sudakov forces exactly zero common Delta')
      endif
      previous=prob
    enddo
    ! z becomes indistinguishable from one at the soft boundary. This
    ! limiting case must have zero suppression factor, rather than the
    ! unit factor used for an unsupported hard remainder.
    do point=1,2
      p(:,5)=0d0
      if(point==2)p(:,5)=[1d-14,1d-14,0d0,0d0]
      call delta_native_ir_probability(p,p1_cnt(:,1:4,0),[1,-1,11,-11,21], &
                               shower_scale_nbody,native_scale,source_prob,history_count,history_t)
      call require(history_count==0.and.source_prob==0d0,'exact and rounded soft boundaries retain zero Delta')
      mccntcalled=0
      call compute_delta(p,prob)
      call require(prob==0d0.and.rng_calls==0,'soft boundary creates no weighted H event or history draw')
    enddo
    previous=delta_common_probability(p1_cnt(:,1:4,0),shower_scale_nbody,.51d0)
    source_prob=delta_common_probability(p1_cnt(:,1:4,0),shower_scale_nbody,.5001d0)
    call require(previous>source_prob.and.source_prob>0d0,'common source approaches its zero continuously')
    call require(delta_common_probability(p1_cnt(:,1:4,0),shower_scale_nbody,.5d0)==0d0, &
                 'the artificial source infrared endpoint is exactly zero')
    print *, 'PASS common-scale Delta deep-collinear damping'
    stop
  endif

  if(trim(mode)=='common')then
    delta_common_scale=.true.
    valid_dipole_n1(3,5)=.true.
    valid_dipole_n1(5,3)=.true.
    valid_dipole_n1(4,5)=.true.
    valid_dipole_n1(5,4)=.true.
    call compute_delta(p,prob)
    t34=sqrt(201600d0)/19d0
    t43=sqrt(3850d0)/3d0
    expected=min(t34*t34,t43*t43)/50d0**2
    call require(abs(prob-expected)<1d-12,'source uses the lower envelope of all-leg Sudakovs')
    call require(all(abs(pack(emsca_H(1,1,:,:),valid_dipole_n1)-native_scale)<1d-12), &
                 'all active H dipoles inherit the native matching evolution scale')
    call require(rng_calls==0,'native continuation needs no conditional history draw')
    call delta_native_ir_probability(p,p1_cnt(:,1:4,0),[11,-11,1,-1,21], &
                             shower_scale_nbody,native_scale,source_prob,history_count,history_t)
    call require(history_count==2,'two valid fixed-Born quark singular reconstructions')
    call require(maxval(abs(history_t(1:2)-[t34,t43]))<1d-12,'inverse reconstruction scales')
    ! A smaller actual evolution scale must itself enter the envelope;
    ! it also sets H continuation, irrespective of any random draw.
    native_scale=10d0
    mccntcalled=0
    call compute_delta(p,prob)
    call require(abs(prob-.04d0)<1d-14,'native all-leg Sudakov participates in global envelope')
    call require(all(abs(pack(emsca_H(1,1,:,:),valid_dipole_n1)-10d0)<1d-12), &
                 'native scale fixes every active H dipole')
    native_scale=0d0
    call delta_native_ir_probability(p,p1_cnt(:,1:4,0),[11,-11,1,-1,21], &
                             shower_scale_nbody,native_scale,source_prob,history_count,history_t)
    call require(source_prob==0d0,'native zero scale forces the infrared source limit')
    native_scale=30d0
    ! A prepared global-recoil history can be valid even when every
    ! local inverse-dipole reconstruction is outside its source support.
    p(:,5)=10d0*p(:,5)
    mccntcalled=0
    call compute_delta(p,prob)
    call delta_native_ir_probability(p,p1_cnt(:,1:4,0),[11,-11,1,-1,21], &
                             shower_scale_nbody,native_scale,source_prob,history_count,history_t)
    call require(history_count==0.and.abs(prob-.36d0)<1d-14, &
                 'native damping does not require a supported local inverse')
    call require(all(abs(pack(emsca_H(1,1,:,:),valid_dipole_n1)-native_scale)<1d-12), &
                 'no local inverse cannot replace the native H continuation cap')
    p(:,5)=p(:,5)/10d0
    ! Leg 3's own reconstructed emission lies above its start (dead for
    ! that inverse history), but it must still compete above leg 4's
    ! lower reconstruction scale. Reusing the old dead mask would omit it.
    start=shower_scale_nbody
    start(3,4)=22d0
    call delta_native_ir_probability(p,p1_cnt(:,1:4,0),[11,-11,1,-1,21], &
                             start,native_scale,source_prob,history_count,history_t)
    call require(history_count==1.and.abs(history_t(1)-t43)<1d-12,'unsupported reconstruction excluded')
    call require(abs(source_prob-t43*t43/(22d0*50d0))<1d-12, &
                 'all competing legs use the common cutoff, regardless of old dead masks')
    start=1d0
    call delta_native_ir_probability(p,p1_cnt(:,1:4,0),[11,-11,1,-1,21], &
                             start,native_scale,source_prob,history_count,history_t)
    call require(history_count==0.and.source_prob==1d0,'no supported histories retain hard remainder')
    shower_scale_nbody=1d0
    mccntcalled=0
    call compute_delta(p,prob)
    call require(prob==1d0.and.rng_calls==0,'hard remainder needs no history draw')
    call require(all(abs(pack(emsca_H(1,1,:,:),valid_dipole_n1)-native_scale)<1d-12), &
                 'hard remainder retains the common native scale')
    print *, 'PASS native-history global-IR Delta weight and handoff'
    stop
  endif

  call compute_delta(p,prob)
  if (trim(mode) == 'invalid') stop 2
  t34=sqrt(201600d0)/19d0
  t43=sqrt(3850d0)/3d0
  expected=t34*t43/50d0**2
  call require(abs(prob-expected) < 1d-12,'Delta product of scale ratios')
  call require(abs(seen_scales(3,4)-t34) < 1d-12,'native stopping scale reaches H assignment')
  call require(abs(emsca_H(1,1,3,4)-t34) < 1d-12,'H scale stored for selected fold')
  call require(logical(.not.seen_dead(3,4) .and. .not.seen_dead(4,3)),'live FF dipoles')
  call require(mccntcalled == 8 .and. table_calls == 5,'Delta table calls and bookkeeping')
  call require(all(mass_calls == 1),'charm and bottom model masses both requested')

  ! Lowering the starting scale vetoes both dipoles before the Sudakov calls.
  shower_scale_nbody=1d0
  mccntcalled=0
  call compute_delta(p,prob)
  call require(logical(seen_dead(3,4) .and. seen_dead(4,3)),'starting-scale veto preserved')
  call require(prob == 1d0 .and. table_calls == 5,'dead dipoles do not enter Sudakov product')
  call require(all(mass_calls == 2),'model masses requested for each Delta calculation')
  print *, 'PASS native Delta matching and starting-scale veto'
contains
  subroutine require(condition,label)
    logical, intent(in) :: condition
    character(len=*), intent(in) :: label
    if (condition) return
    print *, 'FAIL: ',label
    stop 1
  end subroutine require
end program check_mcatnlo_delta_matching

subroutine read_leshouche_info(ids,mothers,colours,nprocs)
  implicit none
  integer :: ids(1,5,1),mothers(1,2,5,1),colours(1,2,5,1),nprocs(1)
  logical :: ir_test,gluon_test
  common /test_ir/ ir_test
  common /test_gluon/ gluon_test
  ids(1,:,1)=(/11,-11,1,-1,21/)
  if(ir_test)ids(1,:,1)=[1,-1,11,-11,21]
  if(gluon_test)ids(1,:,1)=[21,21,11,-11,21]
  mothers=0
  colours=0
  nprocs=1
end subroutine read_leshouche_info

subroutine fill_icolor_H(flow,jpart,overwrite)
  implicit none
  integer :: flow,jpart(7,-2:7)
  logical :: overwrite
  jpart=0
  jpart(4,3)=501
  jpart(5,4)=502
  jpart(4,5)=502
  jpart(5,5)=501
end subroutine fill_icolor_H

double precision function get_mass_from_id(id)
  implicit none
  integer :: id,mass_calls(4:5)
  common /test_mass_calls/ mass_calls
  if (id < 4 .or. id > 5) stop 3
  mass_calls(id)=mass_calls(id)+1
  if (id == 4) then
    get_mass_from_id=0d0
  else
    get_mass_from_id=4.7d0
  endif
end function get_mass_from_id

double precision function pysudakov_safe(scale,mass,id,kind,mcmass)
  implicit none
  double precision :: scale,mass,mcmass(21),cstlow,cstupp,cxmlow,cxmupp
  common /cstxmbds/ cstlow,cstupp,cxmlow,cxmupp
  integer :: id,kind,table_calls
  logical :: ir_test,gluon_test,globalfsr_test
  common /test_globalfsr/ globalfsr_test
  common /test_ir/ ir_test
  common /test_gluon/ gluon_test
  common /test_table_calls/ table_calls
  if (table_calls == 0) then
    cstlow=1d0
    cstupp=1000d0
    cxmlow=1d0
    cxmupp=50d0
  else
    ! The physical dipole mass sqrt(9000) is capped at the table upper limit.
    if (.not.globalfsr_test.and.mass /= 50d0) stop 4
    if (.not.globalfsr_test.and.(abs(id) /= merge(21,1,gluon_test).or.kind /= merge(1,2,ir_test))) stop 4
    if (mcmass(4) /= 1.5d0 .or. mcmass(5) /= 4.8d0) stop 5
  endif
  table_calls=table_calls+1
  pysudakov_safe=scale/1000d0
  if(ir_test.and.scale<1.01d0)pysudakov_safe=pysudakov_safe*max(scale-.5d0,0d0)/.51d0
end function pysudakov_safe

double precision function pdg2pdf(beam,id,sign,x,scale)
  implicit none
  integer :: beam,id,sign
  double precision :: x,scale
  logical :: ir_test
  common /test_ir/ ir_test
  if(ir_test)then
    pdg2pdf=1d0
    return
  endif
  ! This all-final-state test must never request a PDF.
  stop 6
  pdg2pdf=1d0
end function pdg2pdf

subroutine get_Hevent_starting_scales(stopping,dead,p,hscales)
  implicit none
  double precision :: stopping(4,4),p(0:3,5),hscales(5,5)
  logical*1 :: dead(4,4)
  double precision :: seen_scales(4,4)
  logical*1 :: seen_dead(4,4)
  common /test_stopping/ seen_scales,seen_dead
  seen_scales=stopping
  seen_dead=dead
  hscales=-1d0
  hscales(1:4,1:4)=stopping
end subroutine get_Hevent_starting_scales

double precision function ran2()
  implicit none
  integer :: rng_calls
  common /test_rng/ rng_calls
  rng_calls=rng_calls+1
  ran2=0.25d0
end function ran2

double precision function compute_pTparton(p)
  implicit none
  double precision :: p(0:3,5)
  compute_pTparton=11d0
end function compute_pTparton

double precision function get_qMC(xi,y)
  implicit none
  double precision :: xi,y,native_scale
  common /test_native_scale/ native_scale
  get_qMC=native_scale
end function get_qMC
