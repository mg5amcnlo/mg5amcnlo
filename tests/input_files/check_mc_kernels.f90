program check_mc_kernels
  implicit none
  include 'coupl.inc'
  character(len=16) :: mode
  integer :: mask,leg,octets(0:2),triplets(0:2),octet_type,triplet_type,i
  double precision :: mass,z,t,ap(2),reference(2),charge
  call get_command_argument(1,mode)
  if (mode.eq.'clustering') then
     ! All eight assignments of massless/massive daughters and mother.
     ! Octets and triplets have the same kinematic clustering rules,
     ! while retaining distinct colour labels for subsequent operations.
     do mask=0,7
        octets=0
        triplets=0
        do leg=0,2
           mass=0d0
           if (btest(mask,leg)) mass=607.7137d0
           call set_particle_type(octets(leg),8,mass,.true.)
           call set_particle_type(triplets(leg),3,mass,.true.)
           call set_particle_type(octets(leg),8,mass,.true.)
           if (popcnt(octets(leg)).ne.1) error stop 'duplicate particle type'
           if (octets(leg).eq.triplets(leg)) error stop 'lost colour label'
        enddo
        call get_clustering_type(octets,octet_type)
        call get_clustering_type(triplets,triplet_type)
        if (octet_type.ne.triplet_type) error stop 'massive octet clustering'
     enddo
  elseif (mode.eq.'susy') then
     do i=1,9
        g=1.1d0*i
        gal=cmplx(0.3d0/i,0d0,kind=8)
        z=i/10d0
        t=13d0*i
        charge=2d0/3d0
        call AP_reduced_SUSY(8,8,0d0,0d0,t,z,ap)
        call AP_reduced(3,8,0d0,0d0,t,z,reference)
        ! Gluino emission has the fermion kernel with CA instead of CF.
        reference=reference*9d0/4d0
        if (maxval(abs(ap-reference)).gt.1d-12) error stop 'gluino kernel'
        call AP_reduced_SUSY(3,8,charge,0d0,t,z,ap)
        reference=[(4d0/3d0)*g**2,charge**2*dble(gal(1))**2]*2d0*z/t
        if (maxval(abs(ap-reference)).gt.1d-12) error stop 'squark kernel'
        call AP_reduced_SUSY(-3,8,-charge,0d0,t,z,ap)
        if (maxval(abs(ap-reference)).gt.1d-12) error stop 'antisquark kernel'
     enddo
  else
     error stop 'unknown kernel test'
  endif
  write(*,*) 'PASS '//trim(mode)
end program
