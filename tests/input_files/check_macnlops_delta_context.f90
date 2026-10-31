! Exercise actual weight-line storage and generic LHE packet routines only.
! No generated amplitudes, PDFs, Sudakov tables or shower are linked.
program check_macnlops_delta_context
  use, intrinsic :: ieee_arithmetic
  use weight_lines
  implicit none
  integer :: mode,i,key(3),ic(7,4),meta(15),first
  real(8) :: p(0:4,4),probability,expected(3),h,s,r,d,scales(12,12)
  logical :: replayed
  character(1024) :: arg,line
  character(64),parameter :: registry=repeat('a',64)
  call get_command_argument(1,arg)
  read(arg,*)mode
  call clear_macnlops_hard_event()
  call weight_lines_allocated(5,1,1,1)
  call begin_delta_context(3,.37d0)
  first=delta_context_active
  delta_history=2
  delta_recoil=5
  expected=[0d0,.37d0,1d0]
  do i=1,3
    call append_delta_probability(i,expected(i))
  enddo
  if(mode==1)then
    call append_delta_probability(1,.2d0)
    error stop 'duplicate not rejected'
  elseif(mode==2)then
    delta_history=0
    call append_delta_probability(1,.2d0)
    error stop 'invalid key not rejected'
  elseif(mode==3)then
    call append_delta_probability(4,ieee_value(0d0,ieee_quiet_nan))
    error stop 'nonfinite probability not rejected'
  endif
  delta_line_context(1)=first
  call weight_lines_allocated(5,2,3,4)
  if(delta_line_context(1)/=first.or.any(delta_line_context(2:)/=0))stop 1
  call copy_macnlops_line(1,2)
  call reset_macnlops_line(1)
  if(delta_line_context(1)/=0.or.delta_line_context(2)/=first)stop 2
  do i=1,10
    call begin_delta_context(3,.12d0)
  enddo
  if(delta_contexts(first)%count/=3)stop 3
  if(any(delta_contexts(first)%probability(1:3)/=expected))stop 4
  delta_context_active=first
  delta_replay=.true.
  delta_history=2
  delta_recoil=5
  if(complete_delta_replay())stop 5
  if(mode==4)then
    call replay_delta_probability(4,probability,replayed)
    error stop 'missing key not rejected'
  endif
  r=23d0;d=31d0
  do i=3,1,-1
    call replay_delta_probability(i,probability,replayed)
    if(.not.replayed.or.probability/=expected(i))stop 6
    ! The same source multiplier multiplies the COMPLETE H remainder;
    ! the complementary S transfer preserves H+S=R for every draw.
    h=probability*(r-d)
    s=(1d0-probability)*r+probability*d
    if(abs(h+s-r)>1d-13)stop 7
    if(abs(h-expected(i)*(r-d))>1d-13)stop 8
  enddo
  if(.not.complete_delta_replay())stop 9
  if(mode==5)then
    call replay_delta_probability(1,probability,replayed)
    error stop 'repeated consumption not rejected'
  endif
  if(mode==6)then
    read(*,'(a)')line
    call read_macnlops_hard_event(5,line)
    read(*,'(a)')line
    call read_macnlops_delta_context(5,line)
    call write_macnlops_hard_event(6)
  elseif(mode==7.or.mode==8.or.mode==10.or.mode==11)then
    ic=0;p=0d0;meta=0
    ic(1,:)=[2,-2,11,-11];ic(6,:)=[-1,-1,1,1];ic(7,:)=9
    p(0,:)=10d0;p(3,1)=10d0;p(3,2)=-10d0
    p(1,3)=10d0;p(1,4)=-10d0
    meta(1:3)=[2,3,3];meta(7:13)=[1,1,0,0,1,1,4]
    meta(14)=2;meta(15)=1
    if(mode==10)meta(14)=3
    if(mode==11)meta(14)=4
    call store_macnlops_hard_event(4,ic,p,meta)
    associate(packet=>delta_contexts(first))
      i=packet%count
      if(mode==8)i=0
      call store_macnlops_delta_context(packet%owner,packet%insertion, &
           i,packet%keys,packet%probability,registry)
    end associate
    call write_macnlops_hard_event(6)
    call write_macnlops_hard_event(6) ! Single-use: emits no stale packet.
  elseif(mode==9)then
    call read_macnlops_delta_scales(5,12,12,scales)
    do i=1,12
      do first=1,12
        if(scales(i,first)<0d0)cycle
        write(*,'(2(i0,1x),es26.17e3)')i,first,scales(i,first)
      enddo
    enddo
  else
    call deallocate_weight_lines()
    if(allocated(delta_contexts).or.allocated(delta_line_context))stop 10
    if(delta_context_active/=0.or.delta_replay)stop 11
    print *, 'PASS conditional Delta context, replay and paired transfer'
  endif
end program
