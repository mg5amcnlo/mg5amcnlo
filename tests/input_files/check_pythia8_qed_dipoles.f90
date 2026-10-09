program check_pythia8_qed_dipoles
  use qed_shower_support, only: build_pythia8_qed_dipoles
  implicit none
  double precision, allocatable :: p(:,:),mass(:),charge(:)
  integer, allocatable :: pdg(:),recoilers(:)
  logical, allocatable :: dipoles(:,:)
  integer :: n,nincoming,beam_recoil,i,j,ierr,ios

  do
    read(*,*,iostat=ios) n,nincoming,beam_recoil
    if (ios.ne.0) exit
    allocate(p(0:3,n),mass(n),charge(n),pdg(n),recoilers(n), &
         dipoles(n,n))
    do i=1,n
      read(*,*) pdg(i),charge(i),mass(i),p(:,i)
    enddo
    call build_pythia8_qed_dipoles(p,pdg,charge,mass,nincoming, &
         dipoles,ierr,beam_recoil.ne.0)
    recoilers=0
    do i=1,n
      do j=1,n
        if (dipoles(i,j)) then
          if (recoilers(i).ne.0.or.i.eq.j) stop 2
          recoilers(i)=j
        endif
      enddo
    enddo
    write(*,*) ierr,recoilers
    deallocate(p,mass,charge,pdg,recoilers,dipoles)
  enddo
end program check_pythia8_qed_dipoles
