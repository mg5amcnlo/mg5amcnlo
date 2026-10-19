program check_virtual_pooled
  use virtual_pooled
  implicit none
  integer,parameter :: dp=kind(1d0)
  real(dp) :: p(0:3,4),mass(4),ratio(2),b(2),v(2),residual(2),before(2),prob,expected
  integer :: pdg(4),unit,cat,n,k,flag,orders,points,j
  logical :: incoming(4),valid
  character(len=40) :: test_case
  real(dp),allocatable :: cp(:,:),cm(:),x(:),cr(:)
  integer,allocatable :: ci(:)
  logical,allocatable :: cin(:)
  real(dp) :: mur2,qes2
  real(dp) :: dipoles(4,4),weights(1,1),physics(4)
  call get_command_argument(1,test_case)
  p=0d0
  p(:,1)=[250d0,0d0,0d0,250d0]
  p(:,2)=[250d0,0d0,0d0,-250d0]
  p(:,3)=[250d0,sqrt(250d0**2-172.5d0**2),0d0,0d0]
  p(:,4)=[250d0,-p(1,3),0d0,0d0]
  mass=[0d0,0d0,172.5d0,172.5d0]
  pdg=[21,21,6,-6]
  incoming=[.true.,.true.,.false.,.false.]
  ratio=[10d0,-3d0]
  select case(trim(test_case))
  case('antenna')
     open(newunit=unit,file='antenna_input.dat',status='old')
     read(unit,*) n,points
     allocate(cp(0:3,n),ci(n),cin(n))
     read(unit,*) ci
     cin=ci==1
     do j=1,points
        read(unit,*) cat,k,flag
        do orders=1,n
           read(unit,*) cp(:,orders)
        enddo
        call pooled_virtual_antenna_remainder(cp,cin,cat,k,flag,expected,valid)
        write(*,'(I0,1X,ES25.16E3)') merge(1,0,valid),expected
     enddo
     close(unit)
     stop
  case('collect_physics','apply_physics')
     call pooled_virtual_init(2,2)
     call pooled_virtual_prepare(p,mass,incoming,pdg,10000d0,ratio,20000d0,.true.,.true., &
          [.02d0,.03d0],[8,8,3,-3])
     dipoles=0d0
     dipoles(1,2)=.75d0;dipoles(1,3)=-.25d0;dipoles(3,4)=.5d0
     weights=1d0
     call pooled_virtual_set_helicity_weights(weights,dipoles)
     if(trim(test_case)=='collect_physics') then
        b=[2d0,-3d0];v=[0d0,7d0];prob=.25d0
        residual=(v-ratio*b)/prob
        call pooled_virtual_mark_sampled(prob)
        call pooled_virtual_observe(b,residual,7,1,[.2d0,.7d0],4)
        call pooled_virtual_iteration(1,.true.,23)
        call pooled_virtual_dipole_inputs(p,mass,incoming,20000d0,dipoles,physics,valid)
        if(.not.valid) stop 30
        write(*,'(*(ES25.16E3,1X))') physics
     else
        call check(pooled_virtual_ratio,[.1d0,-.21d0],'known extra coupling carried exactly once')
     endif
     stop
  case('batch_predict','batch_probability','batch_predict6','batch_probability6')
     open(newunit=unit,file='batch_input.dat',status='old')
     read(unit,*) n,orders,points
     allocate(cp(0:3,n),cm(n),ci(n),cin(n),cr(orders))
     do k=1,n
        read(unit,*) ci(k),cm(k),flag
        cin(k)=flag==1
     enddo
     call pooled_virtual_init(orders,2)
     do j=1,points
        if(trim(test_case)=='batch_predict6'.or.trim(test_case)=='batch_probability6') then
           read(unit,*) mur2,qes2,cr
        else
           read(unit,*) mur2,cr
        endif
        do k=1,n
           read(unit,*) cp(:,k)
        enddo
        call pooled_virtual_begin_point()
        if(trim(test_case)=='batch_predict6'.or.trim(test_case)=='batch_probability6') then
           call pooled_virtual_prepare(cp,cm,cin,ci,mur2,cr,qes2)
        else
           call pooled_virtual_prepare(cp,cm,cin,ci,mur2,cr)
        endif
        if(trim(test_case)=='batch_probability'.or.trim(test_case)=='batch_probability6') then
           write(*,'(I0,1X,*(ES25.16E3,1X))') j,pooled_virtual_ratio,pooled_virtual_call_probability(.125d0)
        else
           write(*,'(I0,1X,*(ES25.16E3,1X))') j,pooled_virtual_ratio
        endif
     enddo
     close(unit)
     stop
  case('invariants','invariants6')
     open(newunit=unit,file='coordinates_input.dat',status='old')
     if(trim(test_case)=='invariants6') then
        read(unit,*) n,mur2,qes2
     else
        read(unit,*) n,mur2
     endif
     allocate(cp(0:3,n),cm(n),ci(n),cin(n))
     do k=1,n
        read(unit,*) ci(k),cm(k),flag,cp(:,k)
        cin(k)=flag==1
     enddo
     close(unit)
     if(trim(test_case)=='invariants6') then
        call pooled_virtual_invariants(cp,cm,cin,ci,mur2,cat,x,valid,qes2)
     else
        call pooled_virtual_invariants(cp,cm,cin,ci,mur2,cat,x,valid)
     endif
     if(.not.valid) stop 2
     write(*,'(2(I0,1X),*(ES25.16E3,1X))') cat,size(x),x
     stop
  case('inactive')
     call pooled_virtual_init(2,2)
     if(pooled_virtual_enabled.or.pooled_virtual_collecting.or.pooled_virtual_applying) stop 3
     call pooled_virtual_prepare(p,mass,incoming,pdg,10000d0,ratio)
     call check(pooled_virtual_ratio,ratio,'disabled fallback')
     ratio=ratio*2d0
     call pooled_virtual_prepare(p,mass,incoming,pdg,10000d0,ratio)
     call check(pooled_virtual_ratio,ratio,'disabled fallback follows MINT')
     if(pooled_virtual_call_probability(.125d0)/=.125d0) stop 12
     call pooled_virtual_mark_sampled(0.2d0)
     if(pooled_virtual_sampled) stop 4
  case('collect','collect_reset_zero')
     call pooled_virtual_init(2,2)
     if(.not.pooled_virtual_collecting) stop 5
     call pooled_virtual_begin_point()
     call pooled_virtual_prepare(p,mass,incoming,pdg,10000d0,ratio)
     call check(pooled_virtual_ratio,ratio,'collection keeps MINT')
     b=[2d0,-3d0]
     v=[0d0,7d0] ! Exactly zero virtual is still observed.
     prob=0.25d0
     residual=(v-ratio*b)/prob
     call pooled_virtual_mark_sampled(prob)
     call pooled_virtual_observe(b,residual,7,1,[.2d0,.7d0],4)
     if(trim(test_case)=='collect_reset_zero') then
        call pooled_virtual_iteration(0,.false.,19)
     else
        call pooled_virtual_iteration(1,.false.,19)
     endif
     call pooled_virtual_begin_point()
     if(pooled_virtual_sampled) stop 6
     call pooled_virtual_prepare(p,mass,incoming,pdg,10000d0,ratio)
     call pooled_virtual_observe(b,residual,7,1,[.2d0,.7d0],4) ! Unsampled points omitted.
     call pooled_virtual_begin_point()
     call pooled_virtual_prepare(p,mass,incoming,pdg,10000d0,ratio)
     b=[0d0,-1d0]
     v=[5d0,-2d0]
     residual=(v-ratio*b)/prob
     call pooled_virtual_mark_sampled(prob)
     call pooled_virtual_observe(b,residual,8,1,[.3d0,.6d0],8) ! MINT iteration number reset.
     call pooled_virtual_iteration(1,.true.,23)
  case('apply','point_mismatch','mass_mismatch','zero_blend','reinitialize')
     call pooled_virtual_init(2,2)
     if(trim(test_case)=='reinitialize') call pooled_virtual_init(2,2)
     if(.not.pooled_virtual_enabled.or.pooled_virtual_collecting) stop 7
     call pooled_virtual_begin_point()
     if(trim(test_case)=='point_mismatch') pdg(1)=1
     if(trim(test_case)=='mass_mismatch') mass(3:4)=mass(3:4)+1d0
     call pooled_virtual_prepare(p,mass,incoming,pdg,10000d0,ratio)
     if(trim(test_case)=='point_mismatch'.or.trim(test_case)=='mass_mismatch') stop 8
     if(trim(test_case)=='zero_blend') then
        call check(pooled_virtual_ratio,ratio,'zero blend exact fallback')
     else
        expected=0.75d0*ratio(1)+0.25d0*(12d0+0.6d0*log(25d0))
        call check(pooled_virtual_ratio,[expected,ratio(2)],'scaled polynomial and blend')
        before=pooled_virtual_ratio
        ratio=999d0 ! MINT re-evaluates its grid at subsequent folds.
        call check(pooled_virtual_ratio,before,'frozen correction across folds')
        ! Far outside, retain the bounded polynomial and request an exact loop.
        p(0:3,1:2)=p(0:3,1:2)*100d0
        p(0,3:4)=p(0,3:4)*100d0
        p(1,3)=sqrt(p(0,3)**2-mass(3)**2)
        p(1,4)=-p(1,3)
        call pooled_virtual_begin_point()
        ratio=[10d0,-3d0]
        call pooled_virtual_prepare(p,mass,incoming,pdg,10000d0,ratio)
        call check(pooled_virtual_ratio,[12d0,ratio(2)],'tail bounded continuation')
        if(pooled_virtual_call_probability(.125d0)/=1d0) stop 11
     endif
  case('bad_model')
     call pooled_virtual_init(2,2)
     stop 9
  case default
     stop 10
  end select
  print *, 'PASS pooled virtual '//trim(test_case)
contains
  subroutine check(actual,wanted,label)
    real(dp),intent(in) :: actual(:),wanted(:)
    character(len=*),intent(in) :: label
    if(any(abs(actual-wanted)>2d-13*max(1d0,abs(wanted)))) then
       print *,label,actual,wanted
       stop 1
    endif
  end subroutine
end program
