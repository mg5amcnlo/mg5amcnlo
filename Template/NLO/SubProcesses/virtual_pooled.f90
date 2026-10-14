! Frozen, process-shared virtual approximation.  Collection is local to a job;
! only the controller fits a model, at the end-of-first-step synchronization.
! No matrix elements or random numbers are evaluated here.
module virtual_pooled
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use virtual_envelope_carrier, only: virtual_envelope_carrier_factor
  implicit none
  private
  integer, parameter :: dp=kind(1d0), maxlegs=64, maxfeatures=2048
  integer, parameter :: maxterms=512
  integer, parameter :: maxheads=64, maxinputs=128, maxwidth=64, maxhelicities=65536
  type :: order_model
     real(dp) :: blend=0d0, born_scale=1d0, target_scale=1d0
     integer :: terms=0
     real(dp), allocatable :: lower(:), upper(:)
     real(dp), allocatable :: coefficient(:)
     integer, allocatable :: a(:), b(:), c(:)
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
     logical :: conditional=.false., full_branch=.true.
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
  integer, allocatable :: saved_pdg(:)
  real(dp), allocatable :: saved_mass(:), raw_ratio(:), common_ratio(:), features(:)
  logical, allocatable :: saved_incoming(:)
  type(order_model), allocatable :: models(:)
  type(neural_model), allocatable :: neural(:)
  integer, allocatable :: native_multiplicity(:), native_labels(:,:)
  integer, allocatable :: symmetry_legs(:,:), symmetry_helicity(:,:)
  real(dp), allocatable :: native_q(:), point_p(:,:), point_weights(:,:), head_ratio(:,:)
  real(dp) :: point_mur2=1d0,point_qes2=1d0
  logical :: conditional_capture=.false., header_written=.false., have_neural=.false.
  logical :: conditional_point=.false., full_branch=.true., map_locked=.false.
  integer :: selected_helicity=-1
  real(dp) :: selected_probability=1d0
  logical :: have_metadata=.false., prepared=.false., recorded=.false.
  real(dp) :: loop_probability=0d0
  public :: pooled_virtual_init, pooled_virtual_begin_point, pooled_virtual_prepare
  public :: pooled_virtual_mark_sampled, pooled_virtual_observe, pooled_virtual_iteration
  public :: pooled_virtual_invariants, pooled_virtual_call_probability
  public :: pooled_virtual_needs_helicities, pooled_virtual_set_helicity_weights
  public :: pooled_virtual_set_selection, pooled_virtual_validate_native_map
  public :: pooled_virtual_probabilities, pooled_virtual_save_context, pooled_virtual_restore_context
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
    if(allocated(raw_ratio)) deallocate(raw_ratio)
    if(allocated(pooled_virtual_ratio)) deallocate(pooled_virtual_ratio)
    if(allocated(pooled_virtual_actual_ratio)) deallocate(pooled_virtual_actual_ratio,common_ratio)
    if(allocated(features)) deallocate(features)
    if(allocated(models)) deallocate(models)
    if(allocated(neural)) deallocate(neural)
    if(allocated(native_multiplicity)) deallocate(native_multiplicity,native_labels,native_q)
    if(allocated(point_p)) deallocate(point_p)
    if(allocated(point_weights)) deallocate(point_weights)
    if(allocated(head_ratio)) deallocate(head_ratio)
    if(allocated(symmetry_legs)) deallocate(symmetry_legs,symmetry_helicity)
    conditional_capture=.false.
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
    loop_probability=0d0
    pooled_virtual_probability=0d0
    exact_tail=.false.
    conditional_point=.false.
    full_branch=.true.
    selected_helicity=-1
    selected_probability=1d0
    pooled_virtual_velocity_factor=1d0
    pooled_virtual_velocity_valid=.true.
    if(allocated(native_q)) native_q=0d0
  end subroutine

  subroutine pooled_virtual_prepare(p,mass,incoming,pdg,mur2,ratio,qes2,conditional,context_valid)
    real(dp), intent(in) :: p(0:,:),mass(:),mur2
    logical, intent(in) :: incoming(:)
    integer, intent(in) :: pdg(:)
    real(dp), intent(in) :: ratio(:)
    real(dp), intent(in), optional :: qes2
    logical, intent(in), optional :: conditional
    logical, intent(in), optional :: context_valid
    real(dp), allocatable :: coordinate(:)
    integer :: cat
    logical :: valid, oriented
    prepared=.false.
    conditional_point=.false.
    if(size(ratio)/=norders) call fail('coupling-order mismatch at point')
    pooled_virtual_ratio=ratio
    pooled_virtual_actual_ratio=ratio
    common_ratio=ratio
    call virtual_envelope_carrier_factor(p,mass,incoming,pooled_virtual_velocity_factor,pooled_virtual_velocity_valid)
    if(.not.pooled_virtual_enabled) return
    if(.not.all(ieee_is_finite(ratio))) call fail('nonfinite MINT ratio')
    if(present(context_valid)) then
       if(.not.context_valid) then
          ! Native HEL IDs belong to the local MadLoop process. A foreign
          ! counterevent/history must retain its ordinary MINT approximation.
          exact_tail=mode==2
          pooled_virtual_velocity_valid=.false.
          return
       endif
    endif
    ! Existing catalog-5 captures/models keep their original scale meaning.
    ! New captures include the actual loop scale and an absolute energy scale.
    oriented=have_metadata.and.catalog==7
    if(present(conditional)) then
       if(mode==1) conditional_capture=conditional
    endif
    if(.not.have_metadata.and.conditional_capture) oriented=.true.
    if(present(qes2).and..not.(have_metadata.and.catalog==5)) then
       call pooled_virtual_invariants(p,mass,incoming,pdg,mur2,cat,coordinate,valid,qes2,oriented)
    else
       call pooled_virtual_invariants(p,mass,incoming,pdg,mur2,cat,coordinate,valid)
    endif
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
       have_metadata=.true.
       if(.not.conditional_capture) then
          call write_header(sample_unit,'VIRTUAL_POOL_SAMPLES 2')
          header_written=.true.
       endif
    else
       if(cat/=catalog.or.size(coordinate)/=nf.or.size(pdg)/=nlegs) call fail('point/model catalog mismatch')
       if(any(saved_pdg/=pdg).or.any(saved_incoming.neqv.incoming)) call fail('point/model particle mismatch')
       if(any(abs(saved_mass-mass)>1d-10*max(1d0,abs(saved_mass),abs(mass)))) &
            call fail('point/model mass mismatch')
    endif
    raw_ratio=ratio
    features=coordinate
    point_p=p
    point_mur2=mur2
    if(present(qes2)) point_qes2=qes2
    if(conditional_capture.and..not.allocated(native_multiplicity)) call load_native_map()
    prepared=.true.
    if(mode/=2) return
    call predict_common()
    pooled_virtual_actual_ratio=pooled_virtual_ratio
  end subroutine

  subroutine predict_common()
    real(dp) :: value,xa,xb,xc
    integer :: k,r
    pooled_virtual_ratio=raw_ratio
    do k=1,norders
       if(models(k)%blend==0d0) cycle
       value=0d0
       if(outside_training(k)) then
          exact_tail=.true.
       endif
       ! A bounded continuation avoids replacing a useful prediction with
       ! MINT precisely on the rare points that set the rejection envelope.
       ! The exact residual is still sampled with probability one there.
       do r=1,models(k)%terms
          xa=1d0
          xb=1d0
          xc=1d0
          if(models(k)%a(r)>0) xa=2d0*scaled_coordinate(k,models(k)%a(r))-1d0
          if(models(k)%b(r)>0) xb=2d0*scaled_coordinate(k,models(k)%b(r))-1d0
          if(models(k)%c(r)>0) xc=2d0*scaled_coordinate(k,models(k)%c(r))-1d0
          value=value+models(k)%coefficient(r)*xa*xb*xc
       enddo
       value=value*models(k)%target_scale/models(k)%born_scale
       if(.not.ieee_is_finite(value)) call fail('nonfinite model prediction')
       pooled_virtual_ratio(k)=(1d0-models(k)%blend)*raw_ratio(k)+models(k)%blend*value
       if(.not.ieee_is_finite(pooled_virtual_ratio(k))) call fail('nonfinite blended prediction')
    enddo
    common_ratio=pooled_virtual_ratio
  end subroutine

  logical function outside_training(k) result(outside)
    integer,intent(in) :: k
    real(dp) :: width,allowance
    integer :: i
    outside=.false.
    do i=1,nf
       width=models(k)%upper(i)-models(k)%lower(i)
       allowance=1d-10*max(1d0,abs(models(k)%lower(i)))
       if(width>1d-12) allowance=0.1d0*width
       if(features(i)<models(k)%lower(i)-allowance.or.features(i)>models(k)%upper(i)+allowance) &
            outside=.true.
    enddo
  end function

  real(dp) function pooled_virtual_call_probability(base) result(probability)
    real(dp),intent(in) :: base
    probability=base
    if(pooled_virtual_applying.and.exact_tail) probability=1d0
    pooled_virtual_probability=probability
  end function

  subroutine pooled_virtual_mark_sampled(probability)
    real(dp), intent(in) :: probability
    if(.not.pooled_virtual_enabled) return
    if(.not.ieee_is_finite(probability)) call fail('nonfinite virtual probability')
    if(probability<=0d0.or.probability>1d0) call fail('invalid virtual probability')
    loop_probability=probability
    pooled_virtual_probability=probability
    pooled_virtual_sampled=.true.
  end subroutine

  subroutine pooled_virtual_observe(born,residual,channel,iteration,x,bins)
    real(dp), intent(in) :: born(:),residual(:)
    integer, intent(in) :: channel,iteration
    real(dp), intent(in) :: x(:)
    integer, intent(in) :: bins
    real(dp) :: exact, approximation
    integer :: k,h,nh
    if(.not.pooled_virtual_collecting.or..not.prepared.or..not.pooled_virtual_sampled) return
    if(recorded) call fail('duplicate observation for phase-space point')
    if(size(born)/=norders.or.size(residual)/=norders) call fail('observation order mismatch')
    if(channel<1.or.iteration<1) call fail('invalid observation channel/iteration')
    if(.not.all(ieee_is_finite(born)).or..not.all(ieee_is_finite(residual))) &
         call fail('nonfinite virtual observation')
    if(size(x)/=nx) call fail('MINT coordinate mismatch')
    if(.not.all(ieee_is_finite(x))) call fail('nonfinite MINT coordinates')
    if(any(x<0d0).or.any(x>1d0).or.(bins/=4.and.bins/=8)) call fail('invalid MINT coordinates/bins')
    nh=0
    if(allocated(native_multiplicity)) nh=size(native_multiplicity)
    if(.not.header_written) then
       call write_header(sample_unit,'VIRTUAL_POOL_SAMPLES 3')
       write(sample_unit,'(I0)') nh
       do h=1,nh
          write(sample_unit,'(*(I0,1X))') h,native_multiplicity(h),native_labels(:,h)
       enddo
       header_written=.true.
       map_locked=.true.
    endif
    serial=serial+1
    do k=1,norders
       approximation=pooled_virtual_actual_ratio(k)*born(k)
       exact=loop_probability*residual(k)+approximation
       if(.not.ieee_is_finite(exact).or..not.ieee_is_finite(approximation)) &
            call fail('nonfinite reconstructed virtual')
       ! Retain exact zeros and signed/zero Born values.  Probability is kept so
       ! the controller can undo selective expensive-loop sampling in the fit.
       if(conditional_capture) then
          if(nh>0) then
             write(sample_unit,'(5(I0,1X),*(ES25.16E3,1X))') serial,channel,iteration_serial,k,bins, &
                  loop_probability,born(k),exact,approximation,features,x,real(merge(1,0,full_branch),dp), &
                  real(selected_helicity,dp),selected_probability,native_q,point_p
          else
             write(sample_unit,'(5(I0,1X),*(ES25.16E3,1X))') serial,channel,iteration_serial,k,bins, &
                  loop_probability,born(k),exact,approximation,features,x,real(merge(1,0,full_branch),dp), &
                  real(selected_helicity,dp),selected_probability,point_p
          endif
       else
          write(sample_unit,'(5(I0,1X),*(ES25.16E3,1X))') serial,channel,iteration_serial,k,bins, &
               loop_probability,born(k),exact,approximation,features,x
       endif
    enddo
    recorded=.true.
  end subroutine

  logical function pooled_virtual_needs_helicities() result(needed)
    needed=prepared.and.(conditional_capture.or.have_neural)
  end function

  pure recursive subroutine pooled_virtual_probabilities(weights,multiplicity,q,valid)
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

  subroutine pooled_virtual_set_helicity_weights(weights)
    real(dp),intent(in) :: weights(:,:)
    logical :: valid
    if(.not.pooled_virtual_needs_helicities()) return
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
    if(.not.pooled_virtual_enabled.or..not.(conditional_capture.or.have_neural)) return
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
    context%mean=pooled_virtual_ratio;context%actual=pooled_virtual_actual_ratio
    context%base=common_ratio
    if(allocated(head_ratio)) context%heads=head_ratio
    if(allocated(native_q)) context%probability=native_q
    context%conditional=conditional_point;context%full_branch=full_branch
    context%selected=selected_helicity;context%selected_probability=selected_probability
    context%velocity_factor=pooled_virtual_velocity_factor;context%velocity_valid=pooled_virtual_velocity_valid
  end subroutine

  subroutine pooled_virtual_restore_context(context)
    type(pooled_virtual_context),intent(in) :: context
    pooled_virtual_ratio=context%mean;pooled_virtual_actual_ratio=context%actual
    common_ratio=context%base
    if(allocated(context%heads)) head_ratio=context%heads
    if(allocated(context%probability)) native_q=context%probability
    conditional_point=context%conditional;full_branch=context%full_branch
    selected_helicity=context%selected;selected_probability=context%selected_probability
    pooled_virtual_velocity_factor=context%velocity_factor;pooled_virtual_velocity_valid=context%velocity_valid
  end subroutine

  subroutine pooled_virtual_iteration(iteration,accepted,attempts)
    integer, intent(in) :: iteration,attempts
    logical, intent(in) :: accepted
    if(.not.pooled_virtual_collecting) return
    if(iteration<1.or.attempts<0) call fail('invalid iteration accounting')
    ! MINT can restart its iteration counter after a rejected grid iteration.
    ! The file uses a monotonic ordinal to keep those sample sets distinct.
    write(iteration_unit,'(3(I0,1X))') iteration_serial,merge(1,0,accepted),attempts
    flush(iteration_unit)
    flush(sample_unit)
    iteration_serial=iteration_serial+1
  end subroutine

  real(dp) function scaled_coordinate(k,a) result(z)
    integer,intent(in) :: k,a
    real(dp) :: width
    width=models(k)%upper(a)-models(k)%lower(a)
    z=0.5d0
    if(width>1d-12) z=max(0d0,min(1d0,(features(a)-models(k)%lower(a))/width))
  end function

  subroutine write_header(unit,title)
    integer,intent(in) :: unit
    character(len=*),intent(in) :: title
    write(unit,'(A)') title
    write(unit,'(5(I0,1X))') catalog,nf,norders,nlegs,nx
    write(unit,'(*(I0,1X))') saved_pdg
    write(unit,'(*(ES25.16E3,1X))') saved_mass
    write(unit,'(*(I0,1X))') merge(1,0,saved_incoming)
  end subroutine

  subroutine read_model()
    integer :: unit,ios,version_orders,k,order,r,nt,degree,model_nx,nm,version,nscalar
    real(dp) :: unique(maxlegs)
    integer, allocatable :: incoming_int(:)
    character(len=65536) :: line
    open(newunit=unit,file='virtual_pool_model.dat',status='old',action='read',iostat=ios)
    if(ios/=0) call fail('apply mode requires virtual_pool_model.dat')
    read(unit,'(A)',iostat=ios) line
    if(ios/=0) call fail('missing model header')
    select case(trim(line))
    case('VIRTUAL_POOL_MODEL 2')
       version=2
    case('VIRTUAL_POOL_MODEL 3')
       version=3
    case default
       call fail('unsupported model header; restart integration')
    end select
    call record_line(unit,line,5)
    read(line,*,iostat=ios) catalog,nf,version_orders,nlegs,model_nx
    if(ios/=0) call fail('invalid model dimensions')
    if(model_nx/=nx) call fail('model MINT coordinate mismatch')
    if((catalog<5.or.catalog>7).or.nf<1.or.nf>maxfeatures.or.nlegs<3.or.nlegs>maxlegs) &
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
    nscalar=1+nlegs*(nlegs-1)/2+nm+merge(2,0,catalog>=6)
    if(nf/=1+nlegs*(nlegs-1)/2+nm+merge(2,0,catalog>=6)+ &
         merge((nlegs-2)*(nlegs-3)/2,0,catalog==7)) &
         call fail('invalid invariant catalog feature count')
    call record_line(unit,line,nlegs)
    read(line,*,iostat=ios) incoming_int
    if(ios/=0) call fail('invalid incoming flags')
    if(any(incoming_int<0).or.any(incoming_int>1).or.sum(incoming_int)<1.or.sum(incoming_int)>2) &
         call fail('model requires one or two incoming legs')
    saved_incoming=incoming_int==1
    allocate(models(norders))
    do k=1,norders
       call record_line(unit,line,6)
       read(line,*,iostat=ios) order,models(k)%blend,models(k)%born_scale,models(k)%target_scale,nt,degree
       if(ios/=0) call fail('invalid model order header')
       if(order/=k.or.nt<0.or.nt>maxterms) call fail('invalid model order dimensions')
       if(degree<0.or.degree>version) call fail('invalid polynomial degree')
       if(.not.ieee_is_finite(models(k)%blend).or..not.ieee_is_finite(models(k)%born_scale).or. &
            .not.ieee_is_finite(models(k)%target_scale)) call fail('nonfinite model scale/blend')
       if(models(k)%blend<0d0.or.models(k)%blend>1d0.or.models(k)%born_scale<=0d0.or. &
            models(k)%target_scale<=0d0) call fail('invalid model scale/blend')
       models(k)%terms=nt
       allocate(models(k)%lower(nf),models(k)%upper(nf),models(k)%a(nt),models(k)%b(nt),models(k)%c(nt))
       models(k)%c=0
       allocate(models(k)%coefficient(nt))
       if(models(k)%blend>0d0.and.nt==0) call fail('active polynomial has no coefficients')
       call record_line(unit,line,nf)
       read(line,*,iostat=ios) models(k)%lower
       if(ios/=0) call fail('invalid lower bounds')
       call record_line(unit,line,nf)
       read(line,*,iostat=ios) models(k)%upper
       if(ios/=0) call fail('invalid upper bounds')
       if(.not.all(ieee_is_finite(models(k)%lower)).or..not.all(ieee_is_finite(models(k)%upper))) &
            call fail('nonfinite model bounds')
       if(any(models(k)%upper<models(k)%lower)) call fail('reversed model bounds')
       if(catalog==7.and.models(k)%blend>0d0) then
          if(any(models(k)%lower(nscalar+1:)/=-1d0).or.any(models(k)%upper(nscalar+1:)/=1d0)) &
               call fail('signed orientation coordinates require fixed bounds')
       endif
       do r=1,nt
          call record_line(unit,line,version+1)
          if(version==2) then
             read(line,*,iostat=ios) models(k)%a(r),models(k)%b(r),models(k)%coefficient(r)
          else
             read(line,*,iostat=ios) models(k)%a(r),models(k)%b(r),models(k)%c(r),models(k)%coefficient(r)
          endif
          if(ios/=0) call fail('invalid polynomial term')
          if(min(models(k)%a(r),models(k)%b(r),models(k)%c(r))<0.or. &
               max(models(k)%a(r),models(k)%b(r),models(k)%c(r))>nf) &
               call fail('invalid polynomial indices')
          if(models(k)%a(r)==0.and.models(k)%b(r)/=0) call fail('invalid polynomial constant')
          if(models(k)%b(r)==0.and.models(k)%c(r)/=0) call fail('invalid polynomial constant')
          if(degree<3.and.models(k)%c(r)>0) call fail('term exceeds polynomial degree')
          if(degree<2.and.models(k)%b(r)>0) call fail('term exceeds polynomial degree')
          if(degree==0.and.models(k)%a(r)>0) call fail('term exceeds polynomial degree')
          if(catalog==7.and.count([models(k)%a(r),models(k)%b(r),models(k)%c(r)]>nscalar)>1) &
               call fail('polynomial term contains multiple orientation factors')
          if(.not.ieee_is_finite(models(k)%coefficient(r))) call fail('nonfinite polynomial coefficient')
       enddo
    enddo
    call require_end(unit)
    close(unit)
    deallocate(incoming_int)
    have_metadata=.true.
  end subroutine

  recursive subroutine read_neural_model()
    integer :: unit,ios,cat,nfeat,nord,nleg,ncoord,nh,k,r,h,i,j,enabled,order,w1,w2,ident,mult,version,nsym
    integer,allocatable :: pdg(:),flags(:),labels(:)
    real(dp),allocatable :: mass(:)
    character(len=65536) :: line
    open(newunit=unit,file='virtual_pool_neural.dat',status='old',action='read',iostat=ios)
    if(ios/=0) call fail('cannot open neural companion')
    read(unit,'(A)',iostat=ios) line
    version=0
    if(trim(line)=='VIRTUAL_POOL_NEURAL 1') version=1
    if(trim(line)=='VIRTUAL_POOL_NEURAL 2') version=2
    if(ios/=0.or.version==0) call fail('unsupported neural companion header')
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
    nsym=1
    if(version==2) then
       call record_line(unit,line,1);read(line,*,iostat=ios) nsym
       if(ios/=0.or.nsym<1.or.nsym>2) call fail('invalid neural permutation count')
    endif
    allocate(symmetry_legs(nlegs,nsym),symmetry_helicity(nh,nsym))
    symmetry_legs(:,1)=[(i,i=1,nlegs)]
    symmetry_helicity(:,1)=[(h,h=1,nh)]
    if(version==2) then
       do r=1,nsym
          call record_line(unit,line,nlegs);read(line,*,iostat=ios) symmetry_legs(:,r)
          if(ios/=0) call fail('invalid neural permutation row')
       enddo
    endif
    allocate(neural(norders))
    have_neural=.false.
    do k=1,norders
       call record_line(unit,line,8)
       read(line,*,iostat=ios) order,enabled,neural(k)%inputs,neural(k)%heads,w1,w2, &
            neural(k)%born_scale,neural(k)%target_scale
       if(ios/=0) call fail('invalid neural order header')
       if(order/=k.or.enabled<0.or.enabled>1) call fail('invalid neural order index')
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
          case(1,2)
             if(i<1.or.i>nf.or.j/=0.or.h/=0) call fail('invalid neural catalog index')
          case(3)
             if(i<1.or.i>nlegs.or.j<0.or.j>3.or.h/=0) call fail('invalid canonical Born input')
             if(saved_incoming(i)) call fail('canonical input must reference outgoing Born leg')
          case(4,5)
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
    call validate_symmetry()
    ! A symmetry-only companion may also average the complete scalar base.
    have_neural=have_neural.or.size(symmetry_legs,2)>1
    if(have_neural) allocate(head_ratio(maxheads,norders))
  end subroutine

  recursive subroutine validate_symmetry()
    integer,allocatable :: slots(:),changed(:)
    integer :: nh,ns,s,i,h,j,k,r,slot,capacity,matching
    integer(kind=8) :: key
    if(any(symmetry_legs(:,1)/=[(i,i=1,nlegs)])) call fail('first neural permutation must be identity')
    ns=size(symmetry_legs,2)
    if(ns==1) return
    nh=size(symmetry_helicity,1)
    if(nh<1) call fail('neural permutation requires complete helicity map')
    capacity=2*nh+1
    allocate(slots(capacity));slots=0
    ! Full native maps need not be lexicographically ordered. Hash complete
    ! tuples once rather than search every pair of helicities at startup.
    do h=1,nh
       key=0_8
       do i=1,nlegs
          key=ieor(ishftc(key,7),int(native_labels(i,h),8)+101_8)
       enddo
       slot=int(modulo(iand(key,huge(key)),int(capacity,8)))+1
       do while(slots(slot)/=0)
          if(all(native_labels(:,slots(slot))==native_labels(:,h))) call fail('duplicate native helicity tuple')
          slot=modulo(slot,capacity)+1
       enddo
       slots(slot)=h
    enddo
    do s=2,ns
       if(any(symmetry_legs(:,s)<1).or.any(symmetry_legs(:,s)>nlegs)) call fail('neural permutation leg outside bounds')
       changed=pack([(i,i=1,nlegs)],symmetry_legs(:,s)/=[(i,i=1,nlegs)])
       if(size(changed)/=2) call fail('neural permutation must exchange two outgoing legs')
       i=changed(1);j=changed(2)
       if(symmetry_legs(i,s)/=j.or.symmetry_legs(j,s)/=i) call fail('invalid neural transposition')
       if(saved_incoming(i).or.saved_incoming(j)) call fail('neural permutation changes incoming legs')
       if(saved_pdg(i)/=saved_pdg(j).or. &
            abs(saved_mass(i)-saved_mass(j))>1d-10*max(1d0,abs(saved_mass(i)),abs(saved_mass(j)))) &
            call fail('neural permutation exchanges different particles')
       do h=1,nh
          key=0_8
          do i=1,nlegs
             key=ieor(ishftc(key,7),int(native_labels(symmetry_legs(i,s),h),8)+101_8)
          enddo
          slot=int(modulo(iand(key,huge(key)),int(capacity,8)))+1
          matching=0
          do while(slots(slot)/=0)
             j=slots(slot)
             if(all(native_labels(:,j)==native_labels(symmetry_legs(:,s),h))) then
                matching=j
                exit
             endif
             slot=modulo(slot,capacity)+1
          enddo
          if(matching==0) call fail('native helicity map is not closed under permutation')
          if(native_multiplicity(h)/=native_multiplicity(matching)) &
               call fail('neural permutation changes native helicity multiplicity')
          symmetry_helicity(h,s)=matching
       enddo
       do k=1,norders
          do h=1,neural(k)%heads
             j=symmetry_helicity(neural(k)%helicity(h),s)
             matching=0
             do r=1,neural(k)%heads
                if(neural(k)%helicity(r)==j) matching=r
             enddo
             if(matching==0) call fail('neural output heads are not closed under permutation')
             if(neural(k)%trained(h)/=neural(k)%trained(matching)) &
                  call fail('neural trained masks are not closed under permutation')
          enddo
       enddo
    enddo
  end subroutine

  recursive subroutine predict_neural()
    real(dp),allocatable :: saved_p(:,:),saved_features(:),saved_q(:),base(:),heads(:,:),coordinate(:)
    real(dp) :: first_base(norders)
    integer :: ns,s,k,h,j,r,cat
    logical :: valid
    ns=size(symmetry_legs,2)
    if(ns==1) then
       call predict_neural_single()
       return
    endif
    saved_p=point_p;saved_features=features;saved_q=native_q;first_base=common_ratio
    allocate(base(norders),heads(maxheads,norders));base=0d0;heads=0d0
    do s=1,ns
       if(s>1) then
          point_p=saved_p(:,symmetry_legs(:,s))
          native_q(symmetry_helicity(:,s))=saved_q
          if(catalog==5) then
             call pooled_virtual_invariants(point_p,saved_mass,saved_incoming,saved_pdg,point_mur2,cat,coordinate,valid)
          else
             call pooled_virtual_invariants(point_p,saved_mass,saved_incoming,saved_pdg,point_mur2,cat,coordinate,valid, &
                  point_qes2,catalog==7)
          endif
          if(.not.valid) then
             ! All heads must share the same supported fallback. Force the
             ! ordinary exact residual path if any transformed input is bad.
             exact_tail=.true.;conditional_point=.false.
             point_p=saved_p;features=saved_features;native_q=saved_q
             common_ratio=first_base;pooled_virtual_ratio=first_base
             return
          endif
          features=coordinate
          call predict_common()
       endif
       call predict_neural_single()
       base=base+common_ratio
       do k=1,norders
          do h=1,neural(k)%heads
             j=symmetry_helicity(neural(k)%helicity(h),s)
             do r=1,neural(k)%heads
                if(neural(k)%helicity(r)==j) heads(h,k)=heads(h,k)+head_ratio(r,k)
             enddo
          enddo
       enddo
    enddo
    point_p=saved_p;features=saved_features;native_q=saved_q
    common_ratio=base/real(ns,dp);pooled_virtual_ratio=common_ratio
    do k=1,norders
       head_ratio(:,k)=common_ratio(k)
       do h=1,neural(k)%heads
          head_ratio(h,k)=heads(h,k)/real(ns,dp)
          pooled_virtual_ratio(k)=pooled_virtual_ratio(k)+ &
               native_q(neural(k)%helicity(h))*(head_ratio(h,k)-common_ratio(k))
       enddo
    enddo
  end subroutine

  recursive subroutine predict_neural_single()
    real(dp) :: z(maxinputs),hidden(maxwidth),previous(maxwidth),delta(maxheads),canonical(0:3,nlegs)
    real(dp) :: total(0:3),triple(0:3),shat,value,width,allowance
    integer :: k,r,h,i,j,c,ni,nh,nw,np
    logical :: valid,canonical_ready
    canonical_ready=.false.;valid=.true.
    total=0d0
    do i=1,nlegs
       if(saved_incoming(i)) total=total+point_p(:,i)
    enddo
    shat=dot4(total,total)
    pooled_virtual_ratio=common_ratio
    do k=1,norders
       head_ratio(:,k)=common_ratio(k)
       ni=neural(k)%inputs;nh=neural(k)%heads
       if(nh==0) cycle
       do r=1,ni
          i=neural(k)%descriptor(2,r);j=neural(k)%descriptor(3,r);c=neural(k)%descriptor(4,r)
          select case(neural(k)%descriptor(1,r))
          case(1,2)
             value=features(i)
          case(3)
             if(.not.canonical_ready) then
                call canonical_coordinates(point_p,saved_incoming,canonical,valid)
                canonical_ready=.true.
             endif
             if(.not.valid) then
                canonical=0d0
                exact_tail=.true.
             endif
             value=canonical(j,i)
          case(4,5)
             triple=merge(-1d0,1d0,saved_incoming(i))*point_p(:,i)+ &
                  merge(-1d0,1d0,saved_incoming(j))*point_p(:,j)+merge(-1d0,1d0,saved_incoming(c))*point_p(:,c)
             value=dot4(triple,triple)
             if(neural(k)%descriptor(1,r)==4) then
                value=logratio(abs(value),shat)
             else
                value=merge(sign(1d0,value),0d0,value/=0d0)
             endif
          case(6)
             value=2d0*native_q(i)-1d0
          end select
          z(r)=value
          if(neural(k)%descriptor(1,r)==1.or.neural(k)%descriptor(1,r)==4) then
             width=neural(k)%upper(r)-neural(k)%lower(r)
             z(r)=0d0
             allowance=1d-10*max(1d0,abs(neural(k)%lower(r)))
             if(width>1d-12) then
                z(r)=max(-1d0,min(1d0,2d0*(value-neural(k)%lower(r))/width-1d0))
                allowance=0.1d0*width
             endif
             if(value<neural(k)%lower(r)-allowance.or.value>neural(k)%upper(r)+allowance) exact_tail=.true.
          endif
       enddo
       nw=neural(k)%width(1)
       hidden(:nw)=tanh(matmul(transpose(neural(k)%hidden(1)%weight),z(:ni))+neural(k)%hidden(1)%bias)
       do r=2,neural(k)%depth
          np=nw;nw=neural(k)%width(r);previous(:np)=hidden(:np)
          hidden(:nw)=tanh(matmul(transpose(neural(k)%hidden(r)%weight),previous(:np))+neural(k)%hidden(r)%bias)
       enddo
       delta(:nh)=matmul(transpose(neural(k)%output),hidden(:nw))+neural(k)%bias+ &
            matmul(transpose(neural(k)%linear(:ni,:)),z(:ni))+neural(k)%linear(ni+1,:)
       delta(:nh)=delta(:nh)*neural(k)%target_scale/neural(k)%born_scale
       if(.not.all(ieee_is_finite(delta(:nh)))) call fail('nonfinite neural correction')
       do h=1,nh
          if(neural(k)%trained(h)==0) delta(h)=0d0
          head_ratio(h,k)=common_ratio(k)+delta(h)
          pooled_virtual_ratio(k)=pooled_virtual_ratio(k)+native_q(neural(k)%helicity(h))*delta(h)
       enddo
    enddo
  end subroutine

  recursive subroutine canonical_coordinates(p,incoming,coordinates,valid)
    real(dp),intent(in) :: p(0:,:)
    logical,intent(in) :: incoming(:)
    real(dp),intent(out) :: coordinates(0:3,size(incoming))
    logical,intent(out) :: valid
    real(dp) :: cm(0:3,size(incoming)),total(0:3),beta(3),z(3),x(3),y(3),t(3),s,q,b2,g,dot,coef,norm
    integer :: leg,anchor
    valid=.false.;coordinates=0d0;total=0d0
    do leg=1,size(incoming)
       if(incoming(leg)) total=total+p(:,leg)
    enddo
    s=dot4(total,total)
    if(s<=0d0.or.total(0)<=0d0) return
    q=sqrt(s);beta=total(1:3)/total(0);b2=sum(beta**2)
    if(b2>=1d0) return
    g=total(0)/q
    do leg=1,size(incoming)
       dot=sum(p(1:3,leg)*beta);coef=g*g/(g+1d0)*dot-g*p(0,leg)
       cm(0,leg)=g*(p(0,leg)-dot);cm(1:3,leg)=p(1:3,leg)+coef*beta
    enddo
    do anchor=1,size(incoming)
       if(incoming(anchor).eqv.(count(incoming)==2)) exit
    enddo
    z=cm(1:3,anchor);norm=sqrt(sum(z*z))
    if(norm<=q*1d-14) return
    z=z/norm
    do leg=1,size(incoming)
       if(incoming(leg)) cycle
       t=cm(1:3,leg)-sum(cm(1:3,leg)*z)*z;norm=sqrt(sum(t*t))
       if(norm>q*1d-12) exit
    enddo
    if(leg>size(incoming)) return
    x=t/norm;y=cross3(z,x)
    do leg=1,size(incoming)
       if(incoming(leg)) cycle
       coordinates(:,leg)=[4d0*cm(0,leg)/q-1d0,2d0*sum(cm(1:3,leg)*x)/q, &
            2d0*sum(cm(1:3,leg)*y)/q,2d0*sum(cm(1:3,leg)*z)/q]
    enddo
    valid=all(ieee_is_finite(coordinates))
  end subroutine

  subroutine distinct_masses(mass,unique,nm)
    real(dp),intent(in) :: mass(:)
    real(dp),intent(out) :: unique(:)
    integer,intent(out) :: nm
    integer :: i,j
    real(dp) :: swap
    nm=0
    do i=1,size(mass)
       if(mass(i)<=0d0) cycle
       if(any(unique(1:nm)==mass(i))) cycle
       nm=nm+1
       unique(nm)=mass(i)
    enddo
    do i=1,nm-1
       do j=i+1,nm
          if(unique(i)<=unique(j)) cycle
          swap=unique(i);unique(i)=unique(j);unique(j)=swap
       enddo
    enddo
  end subroutine

  subroutine pooled_virtual_invariants(p,mass,incoming,pdg,mur2,cat,x,valid,qes2,orientation)
    real(dp),intent(in) :: p(0:,:),mass(:),mur2
    logical,intent(in) :: incoming(:)
    integer,intent(in) :: pdg(:)
    integer,intent(out) :: cat
    real(dp),allocatable,intent(out) :: x(:)
    logical,intent(out) :: valid
    real(dp),intent(in),optional :: qes2
    logical,intent(in),optional :: orientation
    real(dp) :: total(0:3),shat,unique(maxlegs),a(0:3),b(0:3),c(0:3),d(0:3),den
    integer :: i,j,k,n,nm,ia,ib
    valid=.false.
    cat=5
    if(present(qes2)) then
       cat=6
       if(present(orientation)) then
          if(orientation) cat=7
       endif
       if(.not.ieee_is_finite(qes2)) return
       if(qes2<=0d0) return
    endif
    n=size(pdg)
    if(n<3.or.n>maxlegs.or.size(p,1)/=4.or.size(p,2)/=n.or.size(mass)/=n.or.size(incoming)/=n) return
    if(.not.all(ieee_is_finite(p)).or..not.all(ieee_is_finite(mass)).or..not.ieee_is_finite(mur2)) return
    if(any(mass<0d0).or.mur2<=0d0.or.count(incoming)<1.or.count(incoming)>2) return
    total=0d0
    do i=1,n
       if(incoming(i)) total=total+p(:,i)
    enddo
    shat=dot4(total,total)
    if(shat<=0d0) return
    call distinct_masses(mass,unique,nm)
    k=1+n*(n-1)/2+nm+merge(2,0,cat>=6)+merge((n-2)*(n-3)/2,0,cat==7)
    if(k>maxfeatures) return
    allocate(x(k))
    x(1)=logratio(shat,mur2)
    k=1
    do i=1,n-1
       do j=i+1,n
          k=k+1
          x(k)=logratio(abs(2d0*dot4(p(:,i),p(:,j))),shat)
       enddo
    enddo
    do i=1,nm
       k=k+1
       x(k)=logratio(shat,unique(i)**2)
    enddo
    if(present(qes2)) then
       x(k+1)=logratio(shat,qes2)
       x(k+2)=log(shat)
       k=k+2
    endif
    if(cat==7) then
       ia=1;ib=2
       if(count(incoming)==2) then
          do ia=1,n
             if(incoming(ia)) exit
          enddo
          do ib=ia+1,n
             if(incoming(ib)) exit
          enddo
       else
          do ia=1,n
             if(incoming(ia)) exit
          enddo
          do ib=1,n
             if(.not.incoming(ib)) exit
          enddo
       endif
       ! Dimensionless vectors avoid fourth powers of the physical scale.
       a=p(:,ia)/sqrt(shat);b=p(:,ib)/sqrt(shat)
       do i=1,n-1
          if(i==ia.or.i==ib) cycle
          c=p(:,i)/sqrt(shat)
          do j=i+1,n
             if(j==ia.or.j==ib) cycle
             d=p(:,j)/sqrt(shat)
             den=sqrt(max(0d0,16d0*dot4(a,c)*dot4(b,c)*dot4(a,d)*dot4(b,d)))
             k=k+1;x(k)=0d0
             if(den>1d-15) x(k)=max(-1d0,min(1d0,2d0*epsilon4(a,b,c,d)/den))
          enddo
       enddo
    endif
    valid=all(ieee_is_finite(x))
  end subroutine

  pure recursive function cross3(a,b) result(c)
    real(dp),intent(in) :: a(3),b(3)
    real(dp) :: c(3)
    c=[a(2)*b(3)-a(3)*b(2),a(3)*b(1)-a(1)*b(3),a(1)*b(2)-a(2)*b(1)]
  end function

  pure recursive real(dp) function epsilon4(a,b,c,d) result(value)
    real(dp),intent(in) :: a(0:3),b(0:3),c(0:3),d(0:3)
    real(dp) :: cd(3),bd(3),bc(3)
    cd=cross3(c(1:3),d(1:3))
    bd=cross3(b(1:3),d(1:3))
    bc=cross3(b(1:3),c(1:3))
    value=a(0)*sum(b(1:3)*cd)-b(0)*sum(a(1:3)*cd) &
         +c(0)*sum(a(1:3)*bd)-d(0)*sum(a(1:3)*bc)
  end function

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

  real(dp) function dot4(a,b) result(value)
    real(dp),intent(in) :: a(0:3),b(0:3)
    value=a(0)*b(0)-dot_product(a(1:3),b(1:3))
  end function

  real(dp) function logratio(a,b) result(value)
    real(dp),intent(in) :: a,b
    value=log(max(a/b,1d-15))
  end function

end module virtual_pooled
