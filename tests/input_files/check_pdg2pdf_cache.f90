module pdg2pdf_test_state
  implicit none
  integer :: calls=0,active_set=1,active_member=0
contains
  double precision function density(iset,member,part,x,mu)
    integer,intent(in) :: iset,member,part
    double precision,intent(in) :: x,mu
    density=100000d0*iset+1000d0*member+10d0*part+x+mu
  end function
end module

program check_pdg2pdf_cache
  use pdg2pdf_test_state
  implicit none
  integer i

  call check(1,2,1,0.25d0,100d0,2,1)
  do i=1,100
     call check(1,2,1,0.25d0,100d0,2,0)
  enddo
  ! Each entry holds one flavour. A,B,A must preserve both values.
  call check(1,1,1,0.25d0,100d0,1,1)
  call check(1,2,1,0.25d0,100d0,2,0)
  call check(1,1,1,0.25d0,100d0,1,0)
  call check(1,2,1,0.5d0,100d0,2,1)
  call check(1,2,1,0.25d0,200d0,2,1)
  call check(-1,2,1,0.25d0,100d0,-2,1)
  call check(-1,2,-1,0.25d0,100d0,2,1)
  call check(-1,2,-2,0.25d0,100d0,2,0)
  active_set=2
  call check(1,2,1,0.25d0,100d0,2,1)
  active_member=1
  call check(1,2,1,0.25d0,100d0,2,1)
  active_set=1
  active_member=0
  call check(1,2,1,0.25d0,100d0,2,0)

  ! Cache the normalized gluon/photon IDs, including their aliases.
  call check(1,21,1,0.25d0,100d0,0,1)
  call check(1,0,1,0.25d0,100d0,0,0)
  call check(1,22,1,0.25d0,100d0,7,1)
  call check(1,7,1,0.25d0,100d0,7,0)

  ! Twenty-one distinct misses evict the first key. Hits must not
  ! consume additional slots or lose another recently stored key.
  do i=1,21
     call check(1,2,1,dble(i)/64d0,300d0,2,1)
     call check(1,2,1,dble(i)/64d0,300d0,2,0)
  enddo
  call check(1,2,1,1d0/64d0,300d0,2,1)
  call check(1,2,1,21d0/64d0,300d0,2,0)
  write(*,*)'PASS pdg2pdf cache'
contains
  subroutine check(ih,pdg,beam,x,mu,part,misses)
    integer,intent(in) :: ih,pdg,beam,part,misses
    double precision,intent(in) :: x,mu
    double precision actual,expected
    double precision,external :: pdg2pdf
    integer before
    before=calls
    expected=density(active_set,active_member,part,x,mu)
    actual=pdg2pdf(ih,pdg,beam,x,mu)
    if(abs(actual-expected).gt.1d-9)error stop 'incorrect PDF value'
    if(calls-before.ne.misses)then
       write(*,*)'PDF call mismatch',ih,pdg,beam,x,mu,calls-before,misses
       error stop 'incorrect PDF cache reuse'
    endif
  end subroutine
end program

subroutine getnset(iset)
  use pdg2pdf_test_state
  implicit none
  integer,intent(out) :: iset
  iset=active_set
end subroutine

subroutine getnmem(iset,member)
  use pdg2pdf_test_state
  implicit none
  integer,intent(in) :: iset
  integer,intent(out) :: member
  if(iset.ne.active_set)error stop 'incorrect PDF set'
  member=active_member
end subroutine

subroutine evolvepartm(iset,part,x,mu,xf)
  use pdg2pdf_test_state
  implicit none
  integer,intent(in) :: iset,part
  double precision,intent(in) :: x,mu
  double precision,intent(out) :: xf
  calls=calls+1
  xf=x*density(iset,active_member,part,x,mu)
end subroutine
