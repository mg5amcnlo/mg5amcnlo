program check_virtual_envelope_carrier
  use virtual_envelope_carrier, only: virtual_envelope_carrier_factor
  implicit none
  integer :: n,i,j,flag,points
  real(kind(1d0)),allocatable :: p(:,:),mass(:)
  logical,allocatable :: incoming(:)
  real(kind(1d0)) :: factor
  logical :: valid
  read(*,*) points
  do j=1,points
     read(*,*) n
     allocate(p(0:3,n),mass(n),incoming(n))
     do i=1,n
        read(*,*) mass(i),flag,p(:,i)
        incoming(i)=flag==1
     enddo
     call virtual_envelope_carrier_factor(p,mass,incoming,factor,valid)
     write(*,'(L1,1X,ES25.16E3)') valid,factor
     deallocate(p,mass,incoming)
  enddo
end program check_virtual_envelope_carrier
