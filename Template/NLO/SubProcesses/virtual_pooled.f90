! Frozen, process-shared virtual approximation.  Collection is local to a job;
! only the controller fits a model, at the end-of-first-step synchronization.
! No matrix elements or random numbers are evaluated here.
module virtual_pooled
  use virtual_envelope_carrier, only: virtual_envelope_carrier_factor
  use virtual_features, only: dp,maxlegs,maxfeatures,pooled_virtual_invariants, &
       pooled_virtual_dipole_inputs,pooled_virtual_antenna_remainder, &
       canonical_coordinates,distinct_masses,dot4,epsilon4,logratio
  implicit none
  private
  integer, parameter :: maxterms=512
  integer, parameter :: maxheads=64, maxinputs=128, maxwidth=64, maxhelicities=65536
  type :: order_model
     real(dp) :: blend=0d0, born_scale=1d0, target_scale=1d0
     integer :: terms=0
     real(dp), allocatable :: lower(:), upper(:)
     real(dp), allocatable :: coefficient(:)
     integer, allocatable :: a(:), b(:)
     real(dp), allocatable :: surrogate_shared(:),surrogate_offset(:,:),channel_blend(:)
     integer, allocatable :: channel(:)
  end type
  type :: neural_layer
     real(dp), allocatable :: weight(:,:), bias(:)
  end type
  type :: neural_model
     integer :: inputs=0, heads=0, depth=0
     real(dp) :: born_scale=1d0, target_scale=1d0
     integer, allocatable :: descriptor(:,:), helicity(:), trained(:), width(:)
     real(dp), allocatable :: lower(:), upper(:), output(:,:), bias(:), linear(:,:)
     type(neural_layer), allocatable :: hidden(:)
  end type
  type, public :: pooled_virtual_context
     real(dp), allocatable :: mean(:), actual(:), base(:), heads(:,:), probability(:)
     real(dp), allocatable :: coupling(:)
     logical :: conditional=.false., full_branch=.true.
     logical :: prepared=.false., exact_tail=.false.
     integer :: selected=-1
     real(dp) :: selected_probability=1d0
     real(dp) :: velocity_factor=1d0
     logical :: velocity_valid=.true.
  end type
  logical, public :: pooled_virtual_enabled=.false., pooled_virtual_collecting=.false.
  logical, public :: pooled_virtual_sampled=.false.
  logical, public :: pooled_virtual_applying=.false.
  real(dp), public :: pooled_virtual_probability=0d0
  real(dp), public :: pooled_virtual_velocity_factor=1d0
  logical, public :: pooled_virtual_velocity_valid=.true.
  real(dp), allocatable, public :: pooled_virtual_ratio(:)
  real(dp), allocatable, public :: pooled_virtual_actual_ratio(:)
  integer :: mode=0, norders=0, catalog=0, nf=0, nlegs=0
  integer :: nx=0
  logical :: exact_tail=.false.
  integer :: sample_unit=0, iteration_unit=0, serial=0, iteration_serial=1
  integer, allocatable :: saved_pdg(:), saved_colour(:)
  real(dp), allocatable :: saved_mass(:), raw_ratio(:), common_ratio(:), features(:)
  logical, allocatable :: saved_incoming(:)
  type(order_model), allocatable :: models(:)
  type(neural_model), allocatable :: neural(:)
  integer, allocatable :: native_multiplicity(:), native_labels(:,:)
  real(dp), allocatable :: native_q(:), point_p(:,:), point_weights(:,:), head_ratio(:,:)
  real(dp), allocatable :: point_coupling(:)
  real(dp) :: point_qes2=1d0
  real(dp), allocatable :: point_surrogate(:)
  integer :: point_channel=0
  logical :: surrogate_valid=.false.
  real(dp) :: point_surrogate_gs2=1d0
  real(dp),allocatable :: point_soft(:),point_flavour(:)
  integer :: point_groups=0
  logical :: header_written=.false., have_neural=.false.
  logical :: conditional_point=.false., full_branch=.true., map_locked=.false.
  logical :: physics_point_valid=.false.
  integer :: selected_helicity=-1
  real(dp) :: selected_probability=1d0
  logical :: have_metadata=.false., prepared=.false., recorded=.false.
  public :: pooled_virtual_init, pooled_virtual_begin_point, pooled_virtual_prepare
  public :: pooled_virtual_mark_sampled, pooled_virtual_observe, pooled_virtual_iteration
  public :: pooled_virtual_invariants, pooled_virtual_call_probability
  public :: pooled_virtual_needs_helicities, pooled_virtual_set_helicity_weights
  public :: pooled_virtual_set_selection, pooled_virtual_validate_native_map
  public :: pooled_virtual_probabilities, pooled_virtual_save_context, pooled_virtual_restore_context
  public :: pooled_virtual_uses_physics, pooled_virtual_dipole_inputs
  public :: pooled_virtual_antenna_remainder
  public :: pooled_virtual_set_event_weights
  public :: pooled_virtual_reconstruct
contains
  subroutine fail(message)
    character(len=*), intent(in) :: message
    write(*,'(A)') 'Pooled virtual: '//trim(message)
    stop 1
  end subroutine

  subroutine pooled_virtual_init(orders,coordinates)
    integer, intent(in) :: orders
    integer, intent(in) :: coordinates
    integer :: unit, ios
    logical :: exists
    character(len=65536) :: line
    if(sample_unit/=0) close(sample_unit)
    if(iteration_unit/=0) close(iteration_unit)
    sample_unit=0
    iteration_unit=0
    if(allocated(saved_pdg)) deallocate(saved_pdg,saved_mass,saved_incoming)
    if(allocated(saved_colour)) deallocate(saved_colour)
    if(allocated(point_coupling)) deallocate(point_coupling)
    if(allocated(raw_ratio)) deallocate(raw_ratio)
    if(allocated(pooled_virtual_ratio)) deallocate(pooled_virtual_ratio)
    if(allocated(pooled_virtual_actual_ratio)) deallocate(pooled_virtual_actual_ratio,common_ratio)
    if(allocated(features)) deallocate(features)
    if(allocated(models)) deallocate(models)
    if(allocated(neural)) deallocate(neural)
    if(allocated(native_multiplicity)) deallocate(native_multiplicity,native_labels,native_q)
    if(allocated(point_p)) deallocate(point_p)
    if(allocated(point_weights)) deallocate(point_weights)
    if(allocated(point_surrogate)) deallocate(point_surrogate)
    if(allocated(point_soft)) deallocate(point_soft,point_flavour)
    allocate(point_soft(0),point_flavour(0))
    if(allocated(head_ratio)) deallocate(head_ratio)
    header_written=.false.
    have_neural=.false.
    map_locked=.false.
    mode=0
    if(coordinates<1.or.coordinates>256) call fail('invalid MINT coordinate count')
    nx=coordinates
    norders=orders
    catalog=0
    nf=0
    nlegs=0
    serial=0
    iteration_serial=1
    have_metadata=.false.
    pooled_virtual_enabled=.false.
    pooled_virtual_collecting=.false.
    pooled_virtual_applying=.false.
    call pooled_virtual_begin_point()
    if(orders<1.or.orders>1024) call fail('invalid number of coupling orders')
    allocate(pooled_virtual_ratio(orders),pooled_virtual_actual_ratio(orders),common_ratio(orders),raw_ratio(orders))
    allocate(point_coupling(orders));point_coupling=1d0
    pooled_virtual_ratio=0d0
    pooled_virtual_actual_ratio=0d0
    common_ratio=0d0
    raw_ratio=0d0
    inquire(file='virtual_pool_mode.dat',exist=exists)
    if(.not.exists) return
    open(newunit=unit,file='virtual_pool_mode.dat',status='old',action='read',iostat=ios)
    if(ios/=0) call fail('cannot open mode marker')
    call record_line(unit,line,1)
    read(line,*,iostat=ios) mode
    if(ios/=0) call fail('invalid mode marker')
    call require_end(unit)
    close(unit)
    if(mode<0.or.mode>2) call fail('mode must be 0, 1 or 2')
    if(mode==0) return
    pooled_virtual_enabled=.true.
    pooled_virtual_collecting=mode==1
    pooled_virtual_applying=mode==2
    if(mode==1) then
       open(newunit=sample_unit,file='virtual_pool_samples.dat',status='replace',action='write',iostat=ios)
       if(ios/=0) call fail('cannot create training records')
       open(newunit=iteration_unit,file='virtual_pool_iterations.dat',status='replace',action='write',iostat=ios)
       if(ios/=0) call fail('cannot create iteration records')
    else
       call read_model()
       inquire(file='virtual_pool_neural.dat',exist=exists)
       if(exists) call read_neural_model()
    endif
  end subroutine

  subroutine pooled_virtual_begin_point()
    prepared=.false.
    pooled_virtual_sampled=.false.
    recorded=.false.
    pooled_virtual_probability=0d0
    exact_tail=.false.
    conditional_point=.false.
    physics_point_valid=.false.
    surrogate_valid=.false.
    point_channel=0
    point_groups=0
    full_branch=.true.
    selected_helicity=-1
    selected_probability=1d0
    pooled_virtual_velocity_factor=1d0
    pooled_virtual_velocity_valid=.true.
    if(allocated(native_q)) native_q=0d0
    if(allocated(point_coupling)) point_coupling=1d0
  end subroutine

  subroutine pooled_virtual_prepare(p,mass,incoming,pdg,mur2,ratio,qes2,context_valid,coupling,colours, &
       surrogate,supported,channel,surrogate_gs2)
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    real(dp), intent(in) :: p(0:,:),mass(:),mur2
    logical, intent(in) :: incoming(:)
    integer, intent(in) :: pdg(:)
    real(dp), intent(in) :: ratio(:)
    real(dp), intent(in), optional :: qes2
    logical, intent(in), optional :: context_valid
    real(dp), intent(in), optional :: coupling(:)
    integer, intent(in), optional :: colours(:)
    real(dp), intent(in), optional :: surrogate(:)
    logical, intent(in), optional :: supported
    integer, intent(in), optional :: channel
    real(dp), intent(in), optional :: surrogate_gs2
    real(dp), allocatable :: coordinate(:)
    integer :: cat
    logical :: valid
    prepared=.false.
    conditional_point=.false.
    if(size(ratio)/=norders) call fail('coupling-order mismatch at point')
    pooled_virtual_ratio=ratio
    pooled_virtual_actual_ratio=ratio
    common_ratio=ratio
    surrogate_valid=.false.
    point_channel=0
    point_surrogate_gs2=1d0
    if(present(surrogate)) then
       if(.not.present(supported).or..not.present(channel)) call fail('incomplete surrogate point context')
       if(size(surrogate)<4.or.size(surrogate)>94.or.channel<1) call fail('invalid surrogate point context')
       if(allocated(point_surrogate)) then
          if(size(point_surrogate)/=size(surrogate)) call fail('surrogate basis changed during integration')
       endif
       point_surrogate=surrogate
       surrogate_valid=supported.and.all(ieee_is_finite(surrogate))
       point_channel=channel
       if(present(surrogate_gs2)) point_surrogate_gs2=surrogate_gs2
       if(.not.ieee_is_finite(point_surrogate_gs2).or.point_surrogate_gs2<0d0) &
            call fail('invalid surrogate colour coupling')
    endif
    call virtual_envelope_carrier_factor(p,mass,incoming,pooled_virtual_velocity_factor,pooled_virtual_velocity_valid)
    if(.not.pooled_virtual_enabled) return
    if(.not.all(ieee_is_finite(ratio))) call fail('nonfinite native virtual approximation')
    if(.not.present(qes2)) call fail('physics inputs require the loop scale')
    if(.not.present(coupling).or..not.present(colours)) then
       exact_tail=mode==2
       return
    endif
    if(size(coupling)/=norders.or.size(colours)/=size(pdg)) call fail('invalid physics input dimensions')
    if(.not.all(ieee_is_finite(coupling)).or.any(coupling<0d0).or.any(colours==0)) &
         call fail('invalid coupling/colour inputs')
    point_coupling=coupling
    if(present(context_valid)) then
       if(.not.context_valid) then
          ! Native HEL IDs belong to the local MadLoop process. A foreign
          ! counterevent/history must retain its native approximation.
          exact_tail=mode==2
          pooled_virtual_velocity_valid=.false.
          return
       endif
    endif
    if(.not.present(surrogate)) call fail('pooled refinement requires the native surrogate basis')
    call pooled_virtual_invariants(p,mass,incoming,pdg,mur2,cat,coordinate,valid,qes2)
    if(.not.valid) then
       exact_tail=mode==2
       return
    endif
    if(.not.have_metadata) then
       catalog=cat
       nf=size(coordinate)
       nlegs=size(pdg)
       allocate(saved_pdg(nlegs),saved_mass(nlegs),saved_incoming(nlegs),features(nf))
       saved_pdg=pdg
       saved_mass=mass
       saved_incoming=incoming
       saved_colour=colours
       have_metadata=.true.
    else
       if(cat/=catalog.or.size(coordinate)/=nf.or.size(pdg)/=nlegs) call fail('point/model catalog mismatch')
       if(any(saved_pdg/=pdg).or.any(saved_incoming.neqv.incoming)) call fail('point/model particle mismatch')
       if(any(saved_colour/=colours)) call fail('point/model colour mismatch')
       if(any(abs(saved_mass-mass)>1d-10*max(1d0,abs(saved_mass),abs(mass)))) &
            call fail('point/model mass mismatch')
    endif
    raw_ratio=ratio
    features=coordinate
    point_p=p
    point_qes2=qes2
    if(.not.allocated(native_multiplicity)) call load_native_map()
    prepared=.true.
  end subroutine

  subroutine predict_common()
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    real(dp) :: value,xa,xb,blend
    integer :: k,r,h
    pooled_virtual_ratio=raw_ratio
    do k=1,norders
       if(models(k)%blend==0d0) cycle
       blend=models(k)%blend
       if(allocated(models(k)%surrogate_shared)) then
          if(.not.surrogate_valid.or..not.allocated(point_surrogate)) then
             exact_tail=.true.
             cycle
          endif
          if(size(point_surrogate)/=size(models(k)%surrogate_shared)) call fail('model/surrogate basis mismatch')
          h=0
          do r=1,size(models(k)%channel)
             if(models(k)%channel(r)==point_channel) h=r
          enddo
          if(h==0) cycle
          blend=blend*models(k)%channel_blend(h)
          if(blend==0d0) cycle
       endif
       value=0d0
       if(any([(outside_bounds(features(r),models(k)%lower(r),models(k)%upper(r)),r=1,nf)])) then
          exact_tail=.true.
       endif
       ! Continue the physical logarithms and 1/v at training boundaries.
       ! Unsupported points still request an exact residual with p=1.
       do r=1,models(k)%terms
          xa=1d0
          xb=1d0
          if(models(k)%a(r)>0) xa=2d0*scaled_coordinate(k,models(k)%a(r))-1d0
          if(models(k)%b(r)>0) xb=2d0*scaled_coordinate(k,models(k)%b(r))-1d0
          value=value+models(k)%coefficient(r)*xa*xb
       enddo
       value=value*models(k)%target_scale/models(k)%born_scale
       if(allocated(models(k)%surrogate_shared)) value=value+ &
            dot_product(point_surrogate,models(k)%surrogate_shared+models(k)%surrogate_offset(:,h))
       value=value*point_coupling(k)
       if(.not.ieee_is_finite(value)) call fail('nonfinite model prediction')
       pooled_virtual_ratio(k)=(1d0-blend)*raw_ratio(k)+blend*value
       if(.not.ieee_is_finite(pooled_virtual_ratio(k))) call fail('nonfinite blended prediction')
    enddo
    common_ratio=pooled_virtual_ratio
  end subroutine

  pure recursive logical function outside_bounds(value,lower,upper) result(outside)
    real(dp),intent(in) :: value,lower,upper
    real(dp) :: width,allowance
    width=upper-lower
    allowance=1d-10*max(1d0,abs(lower))
    if(width>1d-12) allowance=0.1d0*width
    outside=value<lower-allowance.or.value>upper+allowance
  end function

  real(dp) function pooled_virtual_call_probability(base) result(probability)
    real(dp),intent(in) :: base
    probability=base
    if(pooled_virtual_applying.and.exact_tail) probability=1d0
    pooled_virtual_probability=probability
  end function

  subroutine pooled_virtual_mark_sampled(probability)
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    real(dp), intent(in) :: probability
    if(.not.pooled_virtual_enabled) return
    if(.not.ieee_is_finite(probability)) call fail('nonfinite virtual probability')
    if(probability<=0d0.or.probability>1d0) call fail('invalid virtual probability')
    pooled_virtual_probability=probability
    pooled_virtual_sampled=.true.
  end subroutine

  function pooled_virtual_reconstruct(born,residual) result(exact)
! Undo the selected-helicity subtraction, before grid/envelope bookkeeping
! can clear a Born carrier belonging to an exactly zero residual.
    real(dp),intent(in) :: born(:),residual(:)
    real(dp) :: exact(size(born))
    if(size(born)/=norders.or.size(residual)/=norders) call fail('reconstruction order mismatch')
    exact=pooled_virtual_probability*residual+pooled_virtual_actual_ratio*born
  end function

  subroutine pooled_virtual_observe(born,residual,channel,iteration,x,bins)
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    real(dp), intent(in) :: born(:),residual(:)
    integer, intent(in) :: channel,iteration
    real(dp), intent(in) :: x(:)
    integer, intent(in) :: bins
    real(dp) :: exact(norders),approximation(norders)
    real(dp),allocatable :: captured_q(:),observation(:)
    integer :: k,h,nh,nfields,offset
    if(.not.pooled_virtual_collecting.or..not.prepared.or..not.pooled_virtual_sampled) return
    if(recorded) call fail('duplicate observation for phase-space point')
    if(size(born)/=norders.or.size(residual)/=norders) call fail('observation order mismatch')
    if(channel<1.or.iteration<1) call fail('invalid observation channel/iteration')
    if(.not.all(ieee_is_finite(born)).or..not.all(ieee_is_finite(residual))) &
         call fail('nonfinite virtual observation')
    if(size(x)/=nx) call fail('MINT coordinate mismatch')
    if(.not.all(ieee_is_finite(x))) call fail('nonfinite MINT coordinates')
    if(any(x<0d0).or.any(x>1d0).or.(bins/=4.and.bins/=8)) call fail('invalid MINT coordinates/bins')
    approximation=pooled_virtual_actual_ratio*born
    exact=pooled_virtual_reconstruct(born,residual)
    if(.not.all(ieee_is_finite(exact)).or..not.all(ieee_is_finite(approximation))) &
         call fail('nonfinite reconstructed virtual')
    nh=0
    if(allocated(native_multiplicity)) nh=size(native_multiplicity)
    allocate(captured_q(nh))
    if(nh>0) captured_q=native_q
    if(.not.header_written) then
       call write_header(sample_unit,'VIRTUAL_POOL_SAMPLES 5')
       write(sample_unit,'(I0)') size(point_surrogate)
       write(sample_unit,'(I0)') nh
       do h=1,nh
          write(sample_unit,'(*(I0,1X))') h,native_multiplicity(h),native_labels(:,h)
       enddo
       header_written=.true.
       map_locked=.true.
    endif
    nfields=11+nf+nx+nh+4*nlegs+size(point_surrogate)+2*point_groups
    allocate(observation(nfields))
    serial=serial+1
    do k=1,norders
       ! Write the absolute Born/virtual prefix, surrogate inputs, native
       ! helicity context and folded S-event groups in their wire order.
       offset=0
       call append([pooled_virtual_probability,born(k),exact(k),approximation(k)])
       call append([point_coupling(k)])
       call append([point_surrogate_gs2,real(merge(1,0,surrogate_valid),dp)])
       call append(point_surrogate)
       call append(features)
       call append(x)
       call append([real(merge(1,0,full_branch),dp),real(selected_helicity,dp),selected_probability])
       call append(captured_q)
       call append(reshape(point_p,[4*nlegs]))
       call append([real(point_groups,dp)])
       call append(point_soft(:point_groups))
       call append(point_flavour(:point_groups))
       if(offset/=nfields) call fail('internal observation layout mismatch')
       write(sample_unit,'(5(I0,1X),*(ES25.16E3,1X))') serial,channel,iteration_serial,k,bins,observation
    enddo
    recorded=.true.
  contains
    subroutine append(values)
      real(dp),intent(in) :: values(:)
      observation(offset+1:offset+size(values))=values
      offset=offset+size(values)
    end subroutine
  end subroutine pooled_virtual_observe

  subroutine pooled_virtual_set_event_weights(soft,flavour)
! Fully folded S-event groups after PDF/bias weights. Fractions are checked
! against both the approximation and residual records by the native caller.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    real(dp),intent(in) :: soft(:),flavour(:)
    if(.not.pooled_virtual_collecting.or..not.prepared) return
    if(size(soft)/=size(flavour).or.size(soft)>10000) call fail('invalid S-event group dimensions')
    if(.not.all(ieee_is_finite(soft)).or..not.all(ieee_is_finite(flavour))) call fail('nonfinite S-event groups')
    if(any(flavour<0d0).or.abs(sum(flavour)-1d0)>1d-9) call fail('invalid S-event flavour fractions')
    point_soft=soft
    point_flavour=flavour
    point_groups=size(soft)
  end subroutine

  logical function pooled_virtual_needs_helicities() result(needed)
    needed=prepared
  end function

  logical function pooled_virtual_uses_physics() result(needed)
    needed=prepared
  end function

  pure recursive subroutine pooled_virtual_probabilities(weights,multiplicity,q,valid)
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    real(dp),intent(in) :: weights(:,:)
    integer,intent(in) :: multiplicity(:)
    real(dp),intent(out) :: q(size(multiplicity))
    logical,intent(out) :: valid
    real(dp) :: total
    integer :: k,h,nonzero
    q=0d0;valid=.false.;nonzero=0
    if(size(weights,2)/=size(multiplicity).or.any(multiplicity<0)) return
    if(.not.all(ieee_is_finite(weights))) return
    do k=1,size(weights,1)
       total=0d0
       do h=1,size(multiplicity)
          if(multiplicity(h)==0) cycle
          if(weights(k,h)<0d0) return
          total=total+weights(k,h)*real(multiplicity(h),dp)
       enddo
       if(.not.ieee_is_finite(total)) return
       if(total<=0d0) cycle
       nonzero=nonzero+1
       do h=1,size(multiplicity)
          if(multiplicity(h)>0) q(h)=q(h)+weights(k,h)/total*real(multiplicity(h),dp)
       enddo
    enddo
    if(nonzero==0) return
    q=q/real(nonzero,dp)
    valid=all(ieee_is_finite(q)).and.abs(sum(q)-1d0)<1d-10
  end subroutine

  subroutine pooled_virtual_set_helicity_weights(weights,dipoles)
    real(dp),intent(in) :: weights(:,:)
    real(dp),intent(in),optional :: dipoles(:,:)
    logical :: valid
    real(dp) :: inputs(4)
    integer :: ns
    if(.not.pooled_virtual_needs_helicities()) return
    if(catalog==9) then
       valid=physics_point_valid
       if(present(dipoles)) then
          call pooled_virtual_dipole_inputs(point_p,saved_mass,saved_incoming,point_qes2,dipoles,inputs,valid)
          if(valid) then
             ns=nf-(nlegs-2)*(nlegs-3)/2
             features(ns-4:ns-1)=inputs
          endif
       endif
       if(.not.valid) then
          exact_tail=mode==2
          prepared=.false.
          return
       endif
       physics_point_valid=.true.
       if(mode==2) then
          call predict_common()
          pooled_virtual_actual_ratio=pooled_virtual_ratio
       endif
    endif
    point_weights=weights
    if(.not.allocated(native_multiplicity)) return
    call pooled_virtual_probabilities(weights,native_multiplicity,native_q,valid)
    conditional_point=valid
    if(.not.valid) then
       ! A vanishing Born distribution has no usable conditional proposal.
       ! Keep the shared prediction and the exact residual support.
       native_q=0d0
       return
    endif
    if(have_neural) call predict_neural()
    pooled_virtual_actual_ratio=pooled_virtual_ratio
  end subroutine

  subroutine pooled_virtual_set_selection(is_full,actual_id,probability)
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    logical,intent(in) :: is_full
    integer,intent(in) :: actual_id
    real(dp),intent(in) :: probability
    integer :: k,h
    if(.not.pooled_virtual_enabled.or..not.prepared) return
    full_branch=is_full
    selected_helicity=-1;selected_probability=1d0
    pooled_virtual_actual_ratio=pooled_virtual_ratio
    if(is_full) return
    if(actual_id<1.or..not.ieee_is_finite(probability).or.probability<=0d0.or.probability>1d0) &
         call fail('invalid native selected helicity')
    selected_helicity=actual_id;selected_probability=probability
    if(.not.conditional_point) return
    if(actual_id>size(native_q)) call fail('native selected helicity outside map')
    if(abs(native_q(actual_id)-probability)>1d-10*max(native_q(actual_id),probability)+1d-14) &
         call fail('native helicity probability disagrees with frozen prediction')
    if(.not.have_neural) return
    pooled_virtual_actual_ratio=common_ratio
    do k=1,norders
       do h=1,neural(k)%heads
          if(neural(k)%helicity(h)==actual_id) pooled_virtual_actual_ratio(k)=head_ratio(h,k)
       enddo
    enddo
  end subroutine

  recursive subroutine pooled_virtual_validate_native_map(helicity,multiplicity)
    integer,intent(in) :: helicity(:),multiplicity(:)
    integer,allocatable :: expanded(:)
    integer :: h
    if(.not.pooled_virtual_enabled) return
    if(.not.allocated(native_multiplicity)) call load_native_map()
    if(.not.allocated(native_multiplicity)) return
    allocate(expanded(size(native_multiplicity)));expanded=0
    if(size(helicity)/=size(multiplicity)) call fail('native compressed map size mismatch')
    do h=1,size(helicity)
       if(helicity(h)<1.or.helicity(h)>size(expanded).or.multiplicity(h)<=0) &
            call fail('invalid native compressed helicity map')
       if(expanded(helicity(h))/=0) call fail('duplicate native helicity representative')
       expanded(helicity(h))=multiplicity(h)
    enddo
    if(any(expanded/=native_multiplicity)) call fail('native HEL/GOODHEL map changed')
    if(allocated(point_weights).and..not.conditional_point.and..not.map_locked) &
         call pooled_virtual_set_helicity_weights(point_weights)
  end subroutine

  recursive subroutine load_native_map()
    integer :: u,v,ios,n,h,value
    integer,allocatable :: labels(:,:),mult(:)
    character(len=65536) :: line
    character(len=1),allocatable :: flags(:)
    if(nlegs<1.or.allocated(native_multiplicity).or.map_locked) return
    open(newunit=u,file='../MadLoop5_resources/HelConfigs.dat',status='old',action='read',iostat=ios)
    if(ios/=0) return
    n=0
    do
       read(u,'(A)',iostat=ios) line
       if(ios<0) exit
       if(ios/=0) call fail('cannot read native HelConfigs.dat')
       if(len_trim(line)>0) n=n+1
    enddo
    if(n<1.or.n>maxhelicities) call fail('native helicity map outside limits')
    rewind(u);allocate(labels(nlegs,n),mult(n),flags(n))
    do h=1,n
       read(u,*,iostat=ios) labels(:,h)
       if(ios/=0) call fail('invalid native helicity labels')
    enddo
    close(u)
    open(newunit=v,file='../MadLoop5_resources/HelFilter.dat',status='old',action='read',iostat=ios)
    if(ios/=0) return
    do h=1,n
       read(v,*,iostat=ios) value
       if(ios/=0) exit
       mult(h)=0
       if(value>-10000.and.value/=0) mult(h)=value
    enddo
    if(ios/=0) then
       rewind(v);read(v,*,iostat=ios) flags
       if(ios/=0) call fail('invalid native HelFilter.dat')
       do h=1,n
          if(flags(h)/='T'.and.flags(h)/='F') call fail('invalid native helicity filter flag')
          mult(h)=merge(1,0,flags(h)=='T')
       enddo
    endif
    close(v)
    if(any(mult<0)) call fail('negative native helicity multiplicity')
    if(.not.any(mult>0)) return
    call move_alloc(mult,native_multiplicity);call move_alloc(labels,native_labels)
    allocate(native_q(n));native_q=0d0
  end subroutine

  subroutine pooled_virtual_save_context(context)
    type(pooled_virtual_context),intent(out) :: context
    context%mean=pooled_virtual_ratio
    context%actual=pooled_virtual_actual_ratio
    context%base=common_ratio
    if(allocated(head_ratio)) context%heads=head_ratio
    if(allocated(native_q)) context%probability=native_q
    context%conditional=conditional_point
    context%prepared=prepared
    context%exact_tail=exact_tail
    context%full_branch=full_branch
    context%selected=selected_helicity
    context%selected_probability=selected_probability
    context%velocity_factor=pooled_virtual_velocity_factor
    context%velocity_valid=pooled_virtual_velocity_valid
    context%coupling=point_coupling
  end subroutine

  subroutine pooled_virtual_restore_context(context)
    type(pooled_virtual_context),intent(in) :: context
    pooled_virtual_ratio=context%mean
    pooled_virtual_actual_ratio=context%actual
    common_ratio=context%base
    if(allocated(context%heads)) head_ratio=context%heads
    if(allocated(context%probability)) native_q=context%probability
    conditional_point=context%conditional
    prepared=context%prepared
    exact_tail=context%exact_tail
    full_branch=context%full_branch
    selected_helicity=context%selected
    selected_probability=context%selected_probability
    pooled_virtual_velocity_factor=context%velocity_factor
    pooled_virtual_velocity_valid=context%velocity_valid
    if(allocated(context%coupling)) point_coupling=context%coupling
  end subroutine

  subroutine pooled_virtual_iteration(iteration,accepted,attempts,channels,channel_attempts)
    integer, intent(in) :: iteration,attempts
    logical, intent(in) :: accepted
    integer, intent(in), optional :: channels(:),channel_attempts(:)
    integer :: k
    if(.not.pooled_virtual_collecting) return
    if(iteration<0.or.(iteration==0.and.accepted).or.attempts<0) call fail('invalid iteration accounting')
    if(present(channels).neqv.present(channel_attempts)) call fail('incomplete stratified accounting')
    if(present(channels)) then
       if(size(channels)/=size(channel_attempts)) call fail('invalid stratified dimensions')
       if(any(channels<1).or.any(channel_attempts<0).or.sum(channel_attempts)/=attempts) &
            call fail('invalid stratified attempt counts')
       do k=1,size(channels)
          if(any(channels(:k-1)==channels(k))) call fail('duplicate stratified channel')
       enddo
    endif
    ! MINT can restart its iteration counter after a rejected grid iteration.
    ! The reset happens before this callback, so that rejection arrives as 0.
    ! The file uses a monotonic ordinal to keep those sample sets distinct.
    write(iteration_unit,'(3(I0,1X))') iteration_serial,merge(1,0,accepted),attempts
    if(present(channels)) then
       do k=1,size(channels)
          write(iteration_unit,'(A,3(I0,1X))') 'CHANNEL ',iteration_serial,channels(k),channel_attempts(k)
       enddo
    endif
    flush(iteration_unit)
    flush(sample_unit)
    iteration_serial=iteration_serial+1
  end subroutine

  real(dp) function scaled_coordinate(k,a) result(z)
    integer,intent(in) :: k,a
    real(dp) :: width
    width=models(k)%upper(a)-models(k)%lower(a)
    z=0.5d0
    if(width>1d-12) then
       z=(features(a)-models(k)%lower(a))/width
    endif
  end function

  subroutine write_header(unit,title)
    integer,intent(in) :: unit
    character(len=*),intent(in) :: title
    write(unit,'(A)') title
    write(unit,'(5(I0,1X))') catalog,nf,norders,nlegs,nx
    write(unit,'(*(I0,1X))') saved_pdg
    write(unit,'(*(ES25.16E3,1X))') saved_mass
    write(unit,'(*(I0,1X))') merge(1,0,saved_incoming)
    write(unit,'(*(I0,1X))') saved_colour
  end subroutine

  subroutine read_model()
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    integer :: unit,ios,version_orders,k,order,r,nt,degree,model_nx,nm,nscalar,nmassive,nb,nc,h,reserved
    real(dp) :: unique(maxlegs)
    integer, allocatable :: incoming_int(:)
    character(len=65536) :: line
    character(len=9) :: basis_tag
    open(newunit=unit,file='virtual_pool_model.dat',status='old',action='read',iostat=ios)
    if(ios/=0) call fail('apply mode requires virtual_pool_model.dat')
    read(unit,'(A)',iostat=ios) line
    if(ios/=0) call fail('missing model header')
    if(trim(line)/='VIRTUAL_POOL_MODEL 4') call fail('unsupported model header; restart integration')
    call record_line(unit,line,5)
    read(line,*,iostat=ios) catalog,nf,version_orders,nlegs,model_nx
    if(ios/=0) call fail('invalid model dimensions')
    if(model_nx/=nx) call fail('model MINT coordinate mismatch')
    if(catalog/=9.or.nf<1.or.nf>maxfeatures.or.nlegs<3.or.nlegs>maxlegs) &
         call fail('model dimensions outside limits')
    if(version_orders/=norders) call fail('model coupling-order mismatch')
    allocate(saved_pdg(nlegs),saved_mass(nlegs),saved_incoming(nlegs),incoming_int(nlegs),features(nf))
    call record_line(unit,line,nlegs)
    read(line,*,iostat=ios) saved_pdg
    if(ios/=0) call fail('invalid model PDGs')
    call record_line(unit,line,nlegs)
    read(line,*,iostat=ios) saved_mass
    if(ios/=0) call fail('invalid model masses')
    if(.not.all(ieee_is_finite(saved_mass))) call fail('nonfinite model masses')
    if(any(saved_mass<0d0)) call fail('negative model masses')
    call distinct_masses(saved_mass,unique,nm)
    call record_line(unit,line,nlegs)
    read(line,*,iostat=ios) incoming_int
    if(ios/=0) call fail('invalid incoming flags')
    if(any(incoming_int<0).or.any(incoming_int>1).or.sum(incoming_int)<1.or.sum(incoming_int)>2) &
         call fail('model requires one or two incoming legs')
    saved_incoming=incoming_int==1
    allocate(saved_colour(nlegs))
    call record_line(unit,line,nlegs);read(line,*,iostat=ios) saved_colour
    if(ios/=0.or.any(saved_colour==0)) call fail('invalid model colour representations')
    nmassive=count(saved_mass>0d0.and..not.saved_incoming)
    nscalar=7+nlegs*(nlegs-1)/2+nm+nmassive*(nmassive-1)/2
    if(nf/=nscalar+(nlegs-2)*(nlegs-3)/2) &
         call fail('invalid invariant catalog feature count')
    allocate(models(norders))
    do k=1,norders
       call record_line(unit,line,6)
       read(line,*,iostat=ios) order,models(k)%blend,models(k)%born_scale,models(k)%target_scale,nt,degree
       if(ios/=0) call fail('invalid model order header')
       if(order/=k.or.nt<0.or.nt>maxterms) call fail('invalid model order dimensions')
       if(degree<0.or.degree>2) call fail('invalid polynomial degree')
       if(.not.ieee_is_finite(models(k)%blend).or..not.ieee_is_finite(models(k)%born_scale).or. &
            .not.ieee_is_finite(models(k)%target_scale)) call fail('nonfinite model scale/blend')
       if(models(k)%blend<0d0.or.models(k)%blend>1d0.or.models(k)%born_scale<=0d0.or. &
            models(k)%target_scale<=0d0) call fail('invalid model scale/blend')
       models(k)%terms=nt
       allocate(models(k)%lower(nf),models(k)%upper(nf),models(k)%a(nt),models(k)%b(nt))
       allocate(models(k)%coefficient(nt))
       call record_line(unit,line,nf)
       read(line,*,iostat=ios) models(k)%lower
       if(ios/=0) call fail('invalid lower bounds')
       call record_line(unit,line,nf)
       read(line,*,iostat=ios) models(k)%upper
       if(ios/=0) call fail('invalid upper bounds')
       if(.not.all(ieee_is_finite(models(k)%lower)).or..not.all(ieee_is_finite(models(k)%upper))) &
            call fail('nonfinite model bounds')
       if(any(models(k)%upper<models(k)%lower)) call fail('reversed model bounds')
       if(models(k)%blend>0d0) then
          if(any(models(k)%lower(nscalar+1:)/=-1d0).or.any(models(k)%upper(nscalar+1:)/=1d0)) &
               call fail('signed orientation coordinates require fixed bounds')
       endif
       do r=1,nt
          call record_line(unit,line,4)
          read(line,*,iostat=ios) models(k)%a(r),models(k)%b(r),reserved,models(k)%coefficient(r)
          if(ios/=0) call fail('invalid polynomial term')
          if(reserved/=0) call fail('residual polynomials have at most two factors')
          if(min(models(k)%a(r),models(k)%b(r))<0.or. &
               max(models(k)%a(r),models(k)%b(r))>nf) &
               call fail('invalid polynomial indices')
          if(models(k)%a(r)==0.and.models(k)%b(r)/=0) call fail('invalid polynomial constant')
          if(degree<2.and.models(k)%b(r)>0) call fail('term exceeds polynomial degree')
          if(degree==0.and.models(k)%a(r)>0) call fail('term exceeds polynomial degree')
          if(count([models(k)%a(r),models(k)%b(r)]>nscalar)>1) &
               call fail('polynomial term contains multiple orientation factors')
          if(.not.ieee_is_finite(models(k)%coefficient(r))) call fail('nonfinite polynomial coefficient')
       enddo
       call record_line(unit,line,3)
       read(line,*,iostat=ios) basis_tag,nb,nc
       if(ios/=0.or.basis_tag/='SURROGATE') call fail('missing pooled surrogate basis')
       if((nb/=0.and.(nb<4.or.nb>94)).or.nc<0.or.nc>10000.or.(nb==0.and.nc/=0)) &
            call fail('invalid pooled surrogate dimensions')
       if(models(k)%blend>0d0.and.nb==0.and.nt==0) call fail('active refinement has no coefficients')
       if(nb>0) then
          allocate(models(k)%surrogate_shared(nb),models(k)%surrogate_offset(nb,nc), &
               models(k)%channel(nc),models(k)%channel_blend(nc))
          call record_line(unit,line,nb)
          read(line,*,iostat=ios) models(k)%surrogate_shared
          if(ios/=0) call fail('invalid shared surrogate coefficients')
          if(.not.all(ieee_is_finite(models(k)%surrogate_shared))) call fail('nonfinite surrogate coefficients')
          do h=1,nc
             call record_line(unit,line,nb+2)
             read(line,*,iostat=ios) models(k)%channel(h),models(k)%channel_blend(h),models(k)%surrogate_offset(:,h)
             if(ios/=0) call fail('invalid channel surrogate coefficients')
             if(models(k)%channel(h)<1.or.any(models(k)%channel(:h-1)==models(k)%channel(h))) &
                  call fail('invalid/duplicate surrogate channel')
             if(.not.ieee_is_finite(models(k)%channel_blend(h)).or. &
                  .not.all(ieee_is_finite(models(k)%surrogate_offset(:,h)))) call fail('nonfinite channel surrogate')
             if(models(k)%channel_blend(h)<0d0.or.models(k)%channel_blend(h)>1d0) &
                  call fail('invalid channel surrogate blend')
          enddo
       endif
    enddo
    call require_end(unit)
    close(unit)
    deallocate(incoming_int)
    have_metadata=.true.
  end subroutine

  recursive subroutine read_neural_model()
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    integer :: unit,ios,cat,nfeat,nord,nleg,ncoord,nh,k,r,h,i,j,enabled,order,w1,w2,ident,mult,nsym
    integer,allocatable :: pdg(:),flags(:),labels(:),colours(:)
    real(dp),allocatable :: mass(:)
    real(dp) :: replacement
    character(len=65536) :: line
    open(newunit=unit,file='virtual_pool_neural.dat',status='old',action='read',iostat=ios)
    if(ios/=0) call fail('cannot open neural companion')
    read(unit,'(A)',iostat=ios) line
    if(ios/=0.or.trim(line)/='VIRTUAL_POOL_NEURAL 4') call fail('unsupported neural companion header')
    call record_line(unit,line,6)
    read(line,*,iostat=ios) cat,nfeat,nord,nleg,ncoord,nh
    if(ios/=0) call fail('invalid neural dimensions')
    if(cat/=catalog.or.nfeat/=nf.or.nord/=norders.or.nleg/=nlegs.or.ncoord/=nx) &
         call fail('neural/base metadata mismatch')
    if(nh<0.or.nh>maxhelicities) call fail('neural helicity map outside limits')
    allocate(pdg(nlegs),flags(nlegs),mass(nlegs),labels(nlegs))
    call record_line(unit,line,nlegs);read(line,*,iostat=ios) pdg
    if(ios/=0) call fail('invalid neural PDGs')
    call record_line(unit,line,nlegs);read(line,*,iostat=ios) mass
    if(ios/=0) call fail('invalid neural masses')
    call record_line(unit,line,nlegs);read(line,*,iostat=ios) flags
    if(ios/=0) call fail('invalid neural incoming flags')
    if(any(pdg/=saved_pdg).or.any(flags/=merge(1,0,saved_incoming))) call fail('neural particle metadata mismatch')
    if(.not.all(ieee_is_finite(mass))) call fail('nonfinite neural masses')
    if(any(abs(mass-saved_mass)>1d-10*max(1d0,abs(mass),abs(saved_mass)))) call fail('neural mass mismatch')
    if(catalog==9) then
       allocate(colours(nlegs))
       call record_line(unit,line,nlegs);read(line,*,iostat=ios) colours
       if(ios/=0.or.any(colours/=saved_colour)) call fail('neural colour mismatch')
    endif
    call record_line(unit,line,1)
    if(len_trim(line)/=64) call fail('invalid neural base SHA256')
    do i=1,64
       if(index('0123456789abcdef',line(i:i))==0) call fail('invalid neural base SHA256')
    enddo
    ! The controller verifies this digest against the frozen polynomial file
    ! before launch. The worker separately checks physical metadata and map.
    if(nh>0) then
       call load_native_map()
       if(.not.allocated(native_multiplicity)) call fail('neural companion requires native helicity resources')
       if(size(native_multiplicity)/=nh) call fail('neural/native helicity map size mismatch')
    endif
    do h=1,nh
       call record_line(unit,line,nlegs+2)
       read(line,*,iostat=ios) ident,mult,labels
       if(ios/=0) call fail('invalid neural helicity row')
       if(ident/=h.or.mult/=native_multiplicity(h).or.any(labels/=native_labels(:,h))) &
            call fail('neural/native helicity map mismatch')
    enddo
    call record_line(unit,line,1);read(line,*,iostat=ios) nsym
    if(ios/=0.or.nsym/=1) call fail('conditional refinement requires one native context')
    call record_line(unit,line,nlegs);read(line,*,iostat=ios) labels
    if(ios/=0.or.any(labels/=[(i,i=1,nlegs)])) call fail('conditional refinement requires native leg order')
    allocate(neural(norders))
    have_neural=.false.
    do k=1,norders
       call record_line(unit,line,9)
       read(line,*,iostat=ios) order,enabled,neural(k)%inputs,neural(k)%heads,w1,w2, &
            neural(k)%born_scale,neural(k)%target_scale,replacement
       if(ios/=0) call fail('invalid neural order header')
       if(order/=k.or.enabled<0.or.enabled>1) call fail('invalid neural order index')
       if(.not.ieee_is_finite(replacement)) call fail('nonfinite conditional head interpretation')
       if(replacement/=-1d0) call fail('conditional heads must refine the common prediction')
       if(.not.ieee_is_finite(neural(k)%born_scale).or..not.ieee_is_finite(neural(k)%target_scale)) &
            call fail('nonfinite neural scales')
       if(neural(k)%born_scale<=0d0.or.neural(k)%target_scale<=0d0) call fail('nonpositive neural scales')
       if(enabled==0) then
          if(neural(k)%inputs/=0.or.neural(k)%heads/=0.or.w1/=0.or.w2/=0) &
               call fail('disabled neural order has nonzero dimensions')
          cycle
       endif
       if(nh<1.or.neural(k)%inputs<1.or.neural(k)%inputs>maxinputs.or.neural(k)%heads<1.or. &
            neural(k)%heads>maxheads.or.w1<1.or.w1>maxwidth.or.w2<0.or.w2>maxwidth) &
            call fail('neural order dimensions outside limits')
       have_neural=.true.
       neural(k)%depth=merge(2,1,w2>0)
       allocate(neural(k)%width(neural(k)%depth),neural(k)%hidden(neural(k)%depth))
       neural(k)%width(1)=w1
       if(w2>0) neural(k)%width(2)=w2
       allocate(neural(k)%descriptor(4,neural(k)%inputs),neural(k)%lower(neural(k)%inputs), &
            neural(k)%upper(neural(k)%inputs),neural(k)%helicity(neural(k)%heads),neural(k)%trained(neural(k)%heads))
       do r=1,neural(k)%inputs
          call record_line(unit,line,6)
          read(line,*,iostat=ios) neural(k)%descriptor(:,r),neural(k)%lower(r),neural(k)%upper(r)
          if(ios/=0) call fail('invalid neural input descriptor')
          if(.not.ieee_is_finite(neural(k)%lower(r)).or..not.ieee_is_finite(neural(k)%upper(r))) &
               call fail('nonfinite neural input bounds')
          if(neural(k)%lower(r)>neural(k)%upper(r)) call fail('reversed neural input bounds')
          i=neural(k)%descriptor(2,r);j=neural(k)%descriptor(3,r);h=neural(k)%descriptor(4,r)
          select case(neural(k)%descriptor(1,r))
          case(1,2,7,9,10)
             if(i<1.or.i>nf.or.j/=0.or.h/=0) call fail('invalid neural catalog index')
          case(3)
             if(i<1.or.i>nlegs.or.j<0.or.j>3.or.h/=0) call fail('invalid canonical Born input')
             if(saved_incoming(i)) call fail('canonical input must reference outgoing Born leg')
          case(4,5,8)
             if(min(i,j,h)<1.or.max(i,j,h)>nlegs.or.i==j.or.i==h.or.j==h) &
                  call fail('invalid neural triple-invariant legs')
          case(6)
             if(i<1.or.i>nh.or.j/=0.or.h/=0) call fail('invalid neural probability input')
          case default
             call fail('unsupported neural input kind')
          end select
       enddo
       do h=1,neural(k)%heads
          call record_line(unit,line,2)
          read(line,*,iostat=ios) neural(k)%helicity(h),neural(k)%trained(h)
          if(ios/=0) call fail('invalid neural output head')
          if(neural(k)%helicity(h)<1.or.neural(k)%helicity(h)>nh.or. &
               neural(k)%trained(h)<0.or.neural(k)%trained(h)>1) call fail('invalid neural head ID/mask')
          if(any(neural(k)%helicity(:h-1)==neural(k)%helicity(h))) call fail('duplicate neural output head')
       enddo
       i=neural(k)%inputs
       do r=1,neural(k)%depth
          j=neural(k)%width(r)
          allocate(neural(k)%hidden(r)%weight(i,j),neural(k)%hidden(r)%bias(j))
          do h=1,i
             call record_line(unit,line,j);read(line,*,iostat=ios) neural(k)%hidden(r)%weight(h,:)
             if(ios/=0) call fail('invalid neural hidden weights')
          enddo
          call record_line(unit,line,j);read(line,*,iostat=ios) neural(k)%hidden(r)%bias
          if(ios/=0) call fail('invalid neural hidden bias')
          if(.not.all(ieee_is_finite(neural(k)%hidden(r)%weight)).or. &
               .not.all(ieee_is_finite(neural(k)%hidden(r)%bias))) call fail('nonfinite neural hidden layer')
          i=j
       enddo
       j=neural(k)%heads
       allocate(neural(k)%output(i,j),neural(k)%bias(j),neural(k)%linear(neural(k)%inputs+1,j))
       do h=1,i
          call record_line(unit,line,j);read(line,*,iostat=ios) neural(k)%output(h,:)
          if(ios/=0) call fail('invalid neural output weights')
       enddo
       call record_line(unit,line,j);read(line,*,iostat=ios) neural(k)%bias
       if(ios/=0) call fail('invalid neural output bias')
       do h=1,neural(k)%inputs+1
          call record_line(unit,line,j);read(line,*,iostat=ios) neural(k)%linear(h,:)
          if(ios/=0) call fail('invalid neural affine weights')
       enddo
       if(.not.all(ieee_is_finite(neural(k)%output)).or..not.all(ieee_is_finite(neural(k)%bias)).or. &
            .not.all(ieee_is_finite(neural(k)%linear))) call fail('nonfinite neural output layer')
    enddo
    call require_end(unit);close(unit)
    if(have_neural) allocate(head_ratio(maxheads,norders))
  end subroutine

  recursive subroutine predict_neural()
    real(dp) :: z(maxinputs),delta(maxheads),canonical(0:3,nlegs)
    real(dp) :: total(0:3),shat
    integer :: k,h,i,ni,nh
    logical :: canonical_ready
    canonical_ready=.false.
    total=0d0
    do i=1,nlegs
       if(saved_incoming(i)) total=total+point_p(:,i)
    enddo
    shat=dot4(total,total)
    pooled_virtual_ratio=common_ratio
    do k=1,norders
       head_ratio(:,k)=common_ratio(k)
       ni=neural(k)%inputs
       nh=neural(k)%heads
       if(nh==0) cycle
       call neural_inputs(neural(k),shat,canonical,canonical_ready,z(:ni))
       call neural_correction(neural(k),z(:ni),delta(:nh))
       delta(:nh)=delta(:nh)*point_coupling(k)
       do h=1,nh
          head_ratio(h,k)=common_ratio(k)+delta(h)
          pooled_virtual_ratio(k)=pooled_virtual_ratio(k)+native_q(neural(k)%helicity(h))*delta(h)
       enddo
    enddo
  end subroutine

  recursive subroutine neural_inputs(model,shat,canonical,canonical_ready,z)
    type(neural_model),intent(in) :: model
    real(dp),intent(in) :: shat
    real(dp),intent(inout) :: canonical(0:3,nlegs)
    logical,intent(inout) :: canonical_ready
    real(dp),intent(out) :: z(:)
    real(dp) :: triple(0:3),value,width
    integer :: r,i,j,c,kind
    logical :: valid
    do r=1,model%inputs
       kind=model%descriptor(1,r)
       i=model%descriptor(2,r)
       j=model%descriptor(3,r)
       c=model%descriptor(4,r)
       select case(kind)
       case(1,2,7,9,10)
          value=features(i)
       case(3)
          if(.not.canonical_ready) then
             call canonical_coordinates(point_p,saved_incoming,canonical,valid)
             canonical_ready=.true.
             if(.not.valid) then
                canonical=0d0
                exact_tail=.true.
             endif
          endif
          value=canonical(j,i)
       case(4,5)
          triple=merge(-1d0,1d0,saved_incoming(i))*point_p(:,i)+ &
               merge(-1d0,1d0,saved_incoming(j))*point_p(:,j)+merge(-1d0,1d0,saved_incoming(c))*point_p(:,c)
          value=dot4(triple,triple)
          if(kind==4) then
             value=logratio(abs(value),shat)
          else
             value=merge(sign(1d0,value),0d0,value/=0d0)
          endif
       case(6)
          value=2d0*native_q(i)-1d0
       case(8)
          call pooled_virtual_antenna_remainder(point_p,saved_incoming,i,j,c,value,valid)
          if(.not.valid) exact_tail=.true.
       end select
       z(r)=value
       if(kind==1.or.kind==4.or.kind>=7) then
          width=model%upper(r)-model%lower(r)
          z(r)=0d0
          if(width>1d-12) z(r)=max(-1d0,min(1d0,2d0*(value-model%lower(r))/width-1d0))
          if(kind>=8.and.width>1d-12) z(r)=2d0*(value-model%lower(r))/width-1d0
          if(outside_bounds(value,model%lower(r),model%upper(r))) exact_tail=.true.
          if(kind==7.or.kind==10) z(r)=z(r)*z(r)
       endif
    enddo
  end subroutine

  recursive subroutine neural_correction(model,z,delta)
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    type(neural_model),intent(in) :: model
    real(dp),intent(in) :: z(:)
    real(dp),intent(out) :: delta(:)
    real(dp) :: hidden(maxwidth),previous(maxwidth)
    integer :: r,ni,nw,np
    ni=model%inputs
    nw=model%width(1)
    hidden(:nw)=tanh(matmul(transpose(model%hidden(1)%weight),z)+model%hidden(1)%bias)
    do r=2,model%depth
       np=nw
       nw=model%width(r)
       previous(:np)=hidden(:np)
       hidden(:nw)=tanh(matmul(transpose(model%hidden(r)%weight),previous(:np))+model%hidden(r)%bias)
    enddo
    delta=matmul(transpose(model%output),hidden(:nw))+model%bias+ &
         matmul(transpose(model%linear(:ni,:)),z)+model%linear(ni+1,:)
    delta=delta*model%target_scale/model%born_scale
    if(.not.all(ieee_is_finite(delta))) call fail('nonfinite neural correction')
    where(model%trained==0) delta=0d0
  end subroutine

  subroutine record_line(unit,line,expected)
    integer,intent(in) :: unit,expected
    character(len=*),intent(out) :: line
    integer :: ios,i,n
    logical :: inword
    read(unit,'(A)',iostat=ios) line
    if(ios/=0) call fail('truncated formatted file')
    n=0
    inword=.false.
    do i=1,len_trim(line)
       if(line(i:i)==' '.or.iachar(line(i:i))==9) then
          inword=.false.
       else
          if(.not.inword) n=n+1
          inword=.true.
       endif
    enddo
    if(n/=expected) call fail('wrong formatted field count')
    if(index(line,',')/=0.or.index(line,'/')/=0.or.index(line,'*')/=0) call fail('invalid formatted token')
  end subroutine

  subroutine require_end(unit)
    integer,intent(in) :: unit
    integer :: ios
    character(len=65536) :: line
    do
       read(unit,'(A)',iostat=ios) line
       if(ios<0) exit
       if(ios/=0) call fail('formatted file read failure')
       if(len_trim(line)/=0) call fail('unexpected trailing data')
    enddo
  end subroutine

end module virtual_pooled
