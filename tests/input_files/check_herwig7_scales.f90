program check_herwig7_scales
  use herwig7_scales
  use, intrinsic :: ieee_arithmetic
  implicit none
  double precision :: p(0:3,4),mass(4),scales(4,4),angular(4,4),saved(4,4),saved_a(4,4)
  double precision :: e3,e4,m,z,pt2,peak,q2,beta,gamma,energy,pz
  logical :: connected(4,4)
  integer :: i,j,k,status
  character(32) :: mode

  call get_command_argument(1,mode)
  mass=0d0
  p(:,1)=[500d0,0d0,0d0,500d0]
  p(:,2)=[500d0,0d0,0d0,-500d0]
  p(:,3)=[500d0,400d0,0d0,300d0]
  p(:,4)=[500d0,-400d0,0d0,-300d0]
  connected=.true.
  do i=1,4
    connected(i,i)=.false.
  enddo

  select case(trim(mode))
  case('massless')
    call herwig7_starting_scales(4,p,mass,connected,10000d0,scales,status,angular)
    call require(status.eq.hw7_ok,'massless status')
    call close(angular(1,2),1000d0,'II angular scale')
    call close(angular(1,3),sqrt(200000d0),'IF angular scale')
    call close(angular(3,1),angular(1,3),'massless FI angular scale')
    call close(angular(3,4),1000d0,'FF angular scale')
    call close(scales(1,3),sqrt(200000d0),'ISR pT envelope')
    call close(scales(3,1),sqrt(200000d0)/4d0,'FI pT envelope')
    call close(scales(3,4),250d0,'FF pT envelope')
    saved=scales
    saved_a=angular
    call herwig7_starting_scales(4,p,mass,connected,150d0,scales,status,angular)
    call require(status.eq.hw7_ok,'capped status')
    call require(all(scales.eq.min(150d0,saved)),'scalar pT cap')
    call require(all(angular.eq.saved_a),'pT veto does not restrict angular evolution')
    connected(1,3)=.false.
    call herwig7_starting_scales(4,p,mass,connected,150d0,scales,status,angular)
    call require(scales(1,3).eq.-1d0.and.angular(1,3).eq.-1d0,'directed mask')
    call require(scales(3,1).gt.0d0,'reverse connection retained')

  case('massive')
    mass(3)=173d0
    mass(4)=80d0
    e3=sqrt(400d0**2+mass(3)**2)
    e4=sqrt(400d0**2+mass(4)**2)
    m=e3+e4
    p(:,1)=[m/2d0,0d0,0d0,m/2d0]
    p(:,2)=[m/2d0,0d0,0d0,-m/2d0]
    p(:,3)=[e3,240d0,0d0,320d0]
    p(:,4)=[e4,-240d0,0d0,-320d0]
    call herwig7_starting_scales(4,p,mass,connected,10000d0,scales,status,angular)
    call require(status.eq.hw7_ok,'massive status')
    call close(angular(3,4)**2,m*(e3+400d0),'massive FF angular scale')
    call close(angular(4,3)**2,m*(e4+400d0),'unequal-mass reverse scale')
    call close(angular(1,3)**2,m*(e3-320d0),'massive IF angular scale')
    call close(angular(3,1)**2,angular(1,3)**2+mass(3)**2,'massive FI angular scale')
    ! Independently scan the actual branching pT rather than duplicating
    ! the analytic maximizer used by the scale module.
    do i=3,4
      do j=1,4
        if (i.eq.j) cycle
        q2=angular(i,j)**2
        peak=0d0
        do k=0,20000
          z=dble(k)/20000d0
          pt2=(1d0-z)**2*(z*z*q2-mass(i)**2)
          peak=max(peak,pt2)
          call require(pt2.le.scales(i,j)**2*(1d0+1d-12),'pT bounded for all z')
        enddo
        call require(abs(sqrt(peak)-scales(i,j)).lt.1d-6*max(1d0,scales(i,j)), &
                     'pT endpoint is attained')
      enddo
    enddo
    saved=scales
    saved_a=angular
    beta=0.6d0
    gamma=1d0/sqrt(1d0-beta**2)
    do i=1,4
      energy=p(0,i)
      pz=p(3,i)
      p(0,i)=gamma*(energy+beta*pz)
      p(3,i)=gamma*(pz+beta*energy)
    enddo
    call herwig7_starting_scales(4,p,mass,connected,10000d0,scales,status,angular)
    call require(status.eq.hw7_ok,'boosted status')
    do i=1,4
      do j=1,4
        call close(scales(i,j),saved(i,j),'boost-invariant pT scales')
        call close(angular(i,j),saved_a(i,j),'boost-invariant angular scales')
      enddo
    enddo

  case('threshold')
    ! At a massive pair threshold the angular envelope is finite;
    ! subsequent shower energy reconstruction can restrict it further.
    mass(3:4)=[173d0,80d0]
    p(:,3)=[173d0,0d0,0d0,0d0]
    p(:,4)=[80d0,0d0,0d0,0d0]
    call herwig7_starting_scales(4,p,mass,connected,10000d0,scales,status,angular)
    call require(status.eq.hw7_ok,'threshold status')
    call close(angular(3,4)**2,253d0*173d0,'threshold angular bound')
    ! Degenerate massless collinear connection: no final-state radiation.
    mass=0d0
    p(:,3)=[500d0,0d0,0d0,500d0]
    p(:,4)=[500d0,0d0,0d0,500d0]
    call herwig7_starting_scales(4,p,mass,connected,10000d0,scales,status,angular)
    call require(status.eq.hw7_ok,'degenerate status')
    call close(scales(3,4),0d0,'zero FF phase space')
    call close(scales(3,1),0d0,'zero FI phase space')

  case('invalid')
    connected(3,3)=.true.
    call herwig7_starting_scales(4,p,mass,connected,100d0,scales,status)
    call require(status.eq.hw7_bad_input,'reject self connection')
    connected(3,3)=.false.
    call herwig7_starting_scales(4,p,mass,connected,-1d0,scales,status)
    call require(status.eq.hw7_bad_input,'reject negative hard scale')
    call herwig7_starting_scales(4,p,mass,connected, &
         ieee_value(0d0,ieee_quiet_nan),scales,status)
    call require(status.eq.hw7_bad_input,'reject NaN')
    mass(1)=1d0
    call herwig7_starting_scales(4,p,mass,connected,100d0,scales,status)
    call require(status.eq.hw7_bad_input,'reject massive beam')
    mass(1)=0d0
    mass(3)=2000d0
    call herwig7_starting_scales(4,p,mass,connected,100d0,scales,status)
    call require(status.eq.hw7_numerical,'reject unphysical FF threshold')
  case default
    stop 2
  end select
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
