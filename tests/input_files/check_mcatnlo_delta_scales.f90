! Adapted from mcatnlo_delta_scale/tests/test_invariants.f90.
! SPDX-License-Identifier: GPL-2.0-or-later
program check_mcatnlo_delta_scales
  use mcatnlo_delta_scales
  implicit none
  real(dp) :: p(0:3,5), moved(0:3,5), t(0:99,0:99), m(0:99,0:99)
  real(dp) :: tm(0:99,0:99), mm(0:99,0:99), raw, ch, sh
  integer :: ids(5), moved_ids(5), col(2,4), status, i, j, map(5)
  logical :: d(0:99,0:99), dm(0:99,0:99)
  character(len=32) :: mode

  call get_command_argument(1,mode)
  ! Deliberately arbitrary triples isolate the scale and mass conventions.
  p(:,1) = (/500._dp,0._dp,0._dp,500._dp/)
  p(:,2) = (/500._dp,0._dp,0._dp,-500._dp/)
  p(:,3) = (/400._dp,240._dp,0._dp,320._dp/)
  p(:,4) = (/350._dp,-210._dp,0._dp,-280._dp/)
  p(:,5) = (/50._dp,-30._dp,0._dp,-40._dp/)
  ids = (/1,-1,2,-2,21/)
  col(1,:) = (/501,0,501,0/)
  col(2,:) = (/0,502,0,502/)

  select case(trim(mode))
  case('mixed')
    call matrices()
    call require(count(t /= -1._dp) == 4,'IF and FI dipole ends')
    call require(t(1,3) > 0._dp .and. t(2,4) > 0._dp,'IF populated')
    call require(t(1,2) == -1._dp .and. all(t(5,:) == -1._dp),'absent entries')
    call require(all(t(0,:) == -1._dp) .and. all(d(0,:)),'zero padding')
    call require(all(t(6:,:) == -1._dp) .and. all(d(6:,:)),'high padding')
    call close(t(1,3),sqrt(9000._dp),'ISR stopping scale')
    call close(m(1,3),sqrt(900000._dp),'signed ISR mass')
    call close(m(3,1),300._dp,'signed FI mass')
    call pythia_pt_lund(p(:,1),p(:,5),p(:,2),1,21,.false.,.false., &
                        0._dp,4.7_dp,raw,status)
    call require(status == delta_ok .and. raw == t(1,3),'opposite beam for IF')

    ch = cosh(0.7_dp)
    sh = sinh(0.7_dp)
    moved = p
    do i=1,5
      moved(0,i) = ch*p(0,i)+sh*p(3,i)
      moved(3,i) = sh*p(0,i)+ch*p(3,i)
    end do
    call delta_scale_matrices(5,5,moved,ids,col,0._dp,4.7_dp,tm,mm,dm,status)
    call require(status == delta_ok,'boost status')
    call require(maxval(abs(tm-t)) < 1.e-8_dp,'Lorentz invariant scales')
    call require(maxval(abs(mm-m)) < 1.e-8_dp,'Lorentz invariant masses')
    call require(all(d .eqv. dm),'Lorentz invariant flags')

  case('final')
    ids = (/11,-11,1,-1,21/)
    col(1,:) = (/0,0,501,0/)
    col(2,:) = (/0,0,0,501/)
    p(:,3) = (/40._dp,40._dp,0._dp,0._dp/)
    p(:,4) = (/30._dp,-30._dp,0._dp,0._dp/)
    p(:,5) = (/30._dp,0._dp,30._dp,0._dp/)
    call matrices()
    call require(count(t /= -1._dp) == 2,'FF dipole ends')
    call close(t(3,4),sqrt(201600._dp)/19._dp,'FF analytic scale')
    call close(t(4,3),sqrt(3850._dp)/3._dp,'FF reverse analytic scale')
    call close(m(3,4),sqrt(9000._dp),'FF mass')
    call require(.not.d(3,4) .and. .not.d(4,3),'FF flags')

    ! Move the emission from slot 5 to slot 3. Born order stays unchanged.
    map = (/1,2,4,5,3/)
    do i=1,5
      moved(:,map(i)) = p(:,i)
      moved_ids(map(i)) = ids(i)
    end do
    call delta_scale_matrices(5,3,moved,moved_ids,col,0._dp,4.7_dp,tm,mm,dm,status)
    call require(status == delta_ok,'remapped status')
    do i=1,5
      do j=1,5
        call close(tm(map(i),map(j)),t(i,j),'remapped scale')
        call close(mm(map(i),map(j)),m(i,j),'remapped mass')
        call require(dm(map(i),map(j)) .eqv. d(i,j),'remapped flags')
      end do
    end do
    call require(all(tm(3,:) == -1._dp) .and. all(tm(:,3) == -1._dp),'emitted slots absent')

  case('double')
    ! Two incoming gluons connected twice occupy one cell per direction.
    ids = (/21,21,25,25,21/)
    col(1,:) = (/501,502,0,0/)
    col(2,:) = (/502,501,0,0/)
    call matrices()
    call require(count(t /= -1._dp) == 2,'double gluon connection')
    call close(t(1,2),sqrt(9000._dp),'II scale')
    call close(t(2,1),sqrt(1000._dp),'II reverse scale')
    call close(m(1,2),sqrt(900000._dp),'II mass')
    call require(.not.d(1,2) .and. .not.d(2,1),'II flags')

  case('sentinels')
    p(:,5) = (/1._dp,0._dp,0._dp,1._dp/)
    call matrices()
    call require(t(1,3) == 1.e-6_dp .and. .not.d(1,3),'tiny virtuality remains live')
    p(:,5) = (/600._dp,600._dp,0._dp,0._dp/)
    call matrices()
    call require(t(1,3) == 1.e-5_dp .and. d(1,3),'negative ISR invariant is dead')
    p(:,1) = (/1._dp,0._dp,0._dp,1._dp/)
    p(:,3) = (/40._dp,40._dp,0._dp,0._dp/)
    p(:,5) = (/30._dp,0._dp,30._dp,0._dp/)
    call matrices()
    call require(t(3,1) == 0._dp .and. d(3,1),'impossible FI reconstruction is zero and dead')

  case('thresholds')
    p(:,5) = (/1._dp,0.6_dp,0._dp,0.8_dp/)
    call pythia_pt_lund(p(:,1),p(:,5),p(:,2),21,5,.false.,.false., &
                        0._dp,4.7_dp,raw,status)
    call require(status == delta_ok,'bottom threshold status')
    call close(raw,sqrt(0.002_dp*(200._dp+4.7_dp**2)),'bottom particle-data mass correction')
    call pythia_pt_lund(p(:,1),p(:,5),p(:,2),4,5,.false.,.false., &
                        1.5_dp,4.7_dp,raw,status)
    call require(status == delta_ok,'charm threshold status')
    call close(raw,sqrt(0.002_dp*(200._dp+1.5_dp**2)),'charm branch precedence')
    call pythia_pt_lund(p(:,1),p(:,5),p(:,2),4,5,.false.,.false., &
                        0._dp,4.7_dp,raw,status)
    call require(status == delta_ok,'massless charm status')
    call close(raw,sqrt(0.4_dp),'MG5 zero charm mass')

  case('invalid')
    col(1,3) = 999
    call delta_scale_matrices(5,5,p,ids,col,0._dp,4.7_dp,t,m,d,status)
    call require(status == delta_bad_input .and. all(t == -1._dp),'unpaired colour rejected')
    col(1,3) = 501
    call delta_scale_matrices(5,2,p,ids,col,0._dp,4.7_dp,t,m,d,status)
    call require(status == delta_bad_input,'incoming emission rejected')
    call delta_scale_matrices(5,5,p,ids,col,-1._dp,4.7_dp,t,m,d,status)
    call require(status == delta_bad_input,'negative shower mass rejected')
    ids(3) = 11
    call delta_scale_matrices(5,5,p,ids,col,0._dp,4.7_dp,t,m,d,status)
    call require(status == delta_bad_input,'coloured non-QCD leg rejected')
    ids(3) = 2
    call delta_scale_matrices(5,5,p,ids,col,0._dp,4.7_dp,t(0:4,0:4),m,d,status)
    call require(status == delta_bad_input,'undersized matrix rejected')
    call pythia_pt_lund((/5._dp,0._dp,0._dp,0._dp/), &
      (/5._dp,0._dp,0._dp,0._dp/),(/10._dp,10._dp,0._dp,0._dp/), &
      5,-5,.true.,.true.,0._dp,4.7_dp,raw,status)
    call require(status == delta_numerical,'exact pair threshold is undefined')

  case default
    call require(.false.,'unknown mode')
  end select
  print *, 'PASS '//trim(mode)

contains

  subroutine matrices()
    call delta_scale_matrices(5,5,p,ids,col,0._dp,4.7_dp,t,m,d,status)
    call require(status == delta_ok,'matrix status')
  end subroutine matrices

  subroutine require(condition,label)
    logical, intent(in) :: condition
    character(len=*), intent(in) :: label
    if (condition) return
    print *, 'FAIL: ',label
    stop 1
  end subroutine require

  subroutine close(actual,expected,label)
    real(dp), intent(in) :: actual,expected
    character(len=*), intent(in) :: label
    call require(abs(actual-expected) <= 1.e-10_dp*max(1._dp,abs(expected)),label)
  end subroutine close

end program check_mcatnlo_delta_scales
