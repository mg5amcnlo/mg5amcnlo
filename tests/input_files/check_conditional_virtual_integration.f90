! Exercise the actual frozen worker reader and selected/full predictions.
program check_conditional_virtual_integration
  use virtual_pooled
  implicit none
  type(pooled_virtual_context) :: frozen
  integer :: n,nlegs,norders,nhel,nborn,i,h,k,cat,u,ios
  integer,allocatable :: pdg(:),incoming_int(:),mult(:),active(:)
  logical,allocatable :: incoming(:)
  logical :: valid,context_valid,physics
  real(8),allocatable :: mass(:),p(:,:),weights(:,:),q(:),mint(:),born(:),target(:),x(:),mean(:)
  real(8),allocatable :: coordinate(:)
  real(8),allocatable :: coupling(:),dipoles(:,:)
  integer,allocatable :: colours(:)
  real(8) :: mur2,qes2,probability,velocity
  character(64) :: mode
  call get_command_argument(1,mode)
  open(newunit=u,file='points.dat',status='old')
  read(u,*) n,nlegs,norders,nhel,nborn
  allocate(pdg(nlegs),incoming_int(nlegs),incoming(nlegs),mass(nlegs),p(0:3,nlegs))
  allocate(weights(nborn,nhel),q(nhel),mult(nhel),mint(norders),born(norders),target(norders),x(2),mean(norders))
  read(u,*) pdg
  read(u,*) mass
  read(u,*) incoming_int
  read(u,*) mult
  incoming=incoming_int==1
  active=pack([(h,h=1,nhel)],mult>0)
  inquire(file='physics.dat',exist=physics)
  if(physics)then
     allocate(coupling(norders),colours(nlegs),dipoles(nlegs,nlegs))
     open(newunit=ios,file='physics.dat',status='old')
     read(ios,*) colours
     read(ios,*) coupling
     read(ios,*) dipoles
     close(ios)
  endif
  call pooled_virtual_init(norders,2)
  do i=1,n
     read(u,*) mur2,qes2,p,weights,mint,born,target
     call pooled_virtual_begin_point()
     context_valid=trim(mode)/='foreign'
     if(physics)then
        call pooled_virtual_prepare(p,mass,incoming,pdg,mur2,mint,qes2,.true.,context_valid,coupling,colours)
     else
        call pooled_virtual_prepare(p,mass,incoming,pdg,mur2,mint,qes2,.true.,context_valid)
     endif
     call pooled_virtual_probabilities(weights,mult,q,valid)
     if(.not.valid.and.trim(mode)/='zero_weights'.and.trim(mode)/='invalid_weights') error stop 'invalid probabilities'
     if(physics)then
        call pooled_virtual_set_helicity_weights(weights,dipoles)
     else
        call pooled_virtual_set_helicity_weights(weights)
     endif
     if(trim(mode)=='ood') then
        write(*,'(ES26.17E3)') pooled_virtual_call_probability(.2d0)
        cycle
     endif
     if(trim(mode)=='invalid_weights') then
        if(valid) error stop 'negative native weights accepted'
        call pooled_virtual_set_selection(.true.,-1,1d0)
        write(*,'(*(ES26.17E3,1X))') pooled_virtual_ratio,pooled_virtual_actual_ratio
        cycle
     endif
     if(context_valid)call pooled_virtual_validate_native_map(active,pack(mult,mult>0))
     mean=pooled_virtual_ratio
     if(trim(mode)=='capture') then
        if(mod(i,2)==0)then
           call pooled_virtual_set_selection(.true.,-1,1d0)
        else
           h=active(1)
           call pooled_virtual_set_selection(.false.,h,q(h))
        endif
        probability=.2d0
        call pooled_virtual_mark_sampled(probability)
        x=[.3d0,.7d0]
        call pooled_virtual_observe(born,(target-pooled_virtual_actual_ratio*born)/probability,1,1,x,4)
        cycle
     endif
     ! The q-weighted mean must be independent of whichever head was selected.
     call pooled_virtual_set_selection(.true.,-1,1d0)
     write(*,'(*(ES26.17E3,1X))') mean,pooled_virtual_actual_ratio
     call pooled_virtual_save_context(frozen)
     velocity=pooled_virtual_velocity_factor
     ! Mimic a later fold/context overwriting all predictions and q. The
     ! frozen first-fold table must still determine the late selected head.
     call pooled_virtual_prepare(p,mass,incoming,pdg,2d0*mur2,10d0*mint,qes2,.true.)
     call pooled_virtual_set_helicity_weights(2d0*weights(:,nhel:1:-1))
     call pooled_virtual_restore_context(frozen)
     if(any(pooled_virtual_ratio/=mean))error stop 'mean lost across deferred context'
     if(pooled_virtual_velocity_factor/=velocity)error stop 'carrier factor lost across deferred context'
     do k=1,size(active)
        h=active(k)
        if(q(h)>0d0)then
           call pooled_virtual_set_selection(.false.,h,q(h))
        else
           call pooled_virtual_set_selection(.true.,-1,1d0)
        endif
        write(*,'(*(ES26.17E3,1X))') pooled_virtual_ratio,pooled_virtual_actual_ratio
     enddo
     call pooled_virtual_set_selection(.true.,-1,1d0)
     if(any(pooled_virtual_actual_ratio/=mean))error stop 'full branch differs from mean'
     if(context_valid)then
        call pooled_virtual_invariants(p,mass,incoming,pdg,mur2,cat,coordinate,valid,qes2,.true.)
        if(.not.valid)error stop 'invalid coordinates'
     endif
  enddo
  if(trim(mode)=='capture')call pooled_virtual_iteration(1,.true.,n)
  close(u)
end program
