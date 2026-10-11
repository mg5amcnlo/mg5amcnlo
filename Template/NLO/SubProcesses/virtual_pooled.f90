! Frozen, process-shared virtual approximation.  Collection is local to a job;
! only the controller fits a model, at the end-of-first-step synchronization.
! No matrix elements or random numbers are evaluated here.
module virtual_pooled
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  integer, parameter :: dp=kind(1d0), maxlegs=64, maxfeatures=2048
  integer, parameter :: maxterms=512
  type :: order_model
     real(dp) :: blend=0d0, born_scale=1d0, target_scale=1d0
     integer :: terms=0
     real(dp), allocatable :: lower(:), upper(:)
     real(dp), allocatable :: coefficient(:)
     integer, allocatable :: a(:), b(:)
  end type
  logical, public :: pooled_virtual_enabled=.false., pooled_virtual_collecting=.false.
  logical, public :: pooled_virtual_sampled=.false.
  logical, public :: pooled_virtual_applying=.false.
  real(dp), public :: pooled_virtual_probability=0d0
  real(dp), allocatable, public :: pooled_virtual_ratio(:)
  integer :: mode=0, norders=0, catalog=0, nf=0, nlegs=0
  integer :: nx=0
  logical :: exact_tail=.false.
  integer :: sample_unit=0, iteration_unit=0, serial=0, iteration_serial=1
  integer, allocatable :: saved_pdg(:)
  real(dp), allocatable :: saved_mass(:), raw_ratio(:), features(:)
  logical, allocatable :: saved_incoming(:)
  type(order_model), allocatable :: models(:)
  logical :: have_metadata=.false., prepared=.false., recorded=.false.
  real(dp) :: loop_probability=0d0
  public :: pooled_virtual_init, pooled_virtual_begin_point, pooled_virtual_prepare
  public :: pooled_virtual_mark_sampled, pooled_virtual_observe, pooled_virtual_iteration
  public :: pooled_virtual_invariants, pooled_virtual_call_probability
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
    if(allocated(features)) deallocate(features)
    if(allocated(models)) deallocate(models)
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
    allocate(pooled_virtual_ratio(orders),raw_ratio(orders))
    pooled_virtual_ratio=0d0
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
    endif
  end subroutine

  subroutine pooled_virtual_begin_point()
    prepared=.false.
    pooled_virtual_sampled=.false.
    recorded=.false.
    loop_probability=0d0
    pooled_virtual_probability=0d0
    exact_tail=.false.
  end subroutine

  subroutine pooled_virtual_prepare(p,mass,incoming,pdg,mur2,ratio)
    real(dp), intent(in) :: p(0:,:),mass(:),mur2
    logical, intent(in) :: incoming(:)
    integer, intent(in) :: pdg(:)
    real(dp), intent(in) :: ratio(:)
    real(dp), allocatable :: coordinate(:)
    real(dp) :: value, xa, xb
    integer :: cat, k, r
    logical :: valid
    prepared=.false.
    if(size(ratio)/=norders) call fail('coupling-order mismatch at point')
    pooled_virtual_ratio=ratio
    if(.not.pooled_virtual_enabled) return
    if(.not.all(ieee_is_finite(ratio))) call fail('nonfinite MINT ratio')
    call pooled_virtual_invariants(p,mass,incoming,pdg,mur2,cat,coordinate,valid)
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
       call write_header(sample_unit,'VIRTUAL_POOL_SAMPLES 2')
    else
       if(cat/=catalog.or.size(coordinate)/=nf.or.size(pdg)/=nlegs) call fail('point/model catalog mismatch')
       if(any(saved_pdg/=pdg).or.any(saved_incoming.neqv.incoming)) call fail('point/model particle mismatch')
       if(any(abs(saved_mass-mass)>1d-10*max(1d0,abs(saved_mass),abs(mass)))) &
            call fail('point/model mass mismatch')
    endif
    raw_ratio=ratio
    features=coordinate
    prepared=.true.
    if(mode/=2) return
    do k=1,norders
       if(models(k)%blend==0d0) cycle
       value=0d0
       if(outside_training(k)) then
          exact_tail=.true.
          cycle
       endif
       do r=1,models(k)%terms
          xa=1d0
          xb=1d0
          if(models(k)%a(r)>0) xa=2d0*scaled_coordinate(k,models(k)%a(r))-1d0
          if(models(k)%b(r)>0) xb=2d0*scaled_coordinate(k,models(k)%b(r))-1d0
          value=value+models(k)%coefficient(r)*xa*xb
       enddo
       value=value*models(k)%target_scale/models(k)%born_scale
       if(.not.ieee_is_finite(value)) call fail('nonfinite model prediction')
       pooled_virtual_ratio(k)=(1d0-models(k)%blend)*raw_ratio(k)+models(k)%blend*value
       if(.not.ieee_is_finite(pooled_virtual_ratio(k))) call fail('nonfinite blended prediction')
    enddo
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
    integer :: k
    if(.not.pooled_virtual_collecting.or..not.prepared.or..not.pooled_virtual_sampled) return
    if(recorded) call fail('duplicate observation for phase-space point')
    if(size(born)/=norders.or.size(residual)/=norders) call fail('observation order mismatch')
    if(channel<1.or.iteration<1) call fail('invalid observation channel/iteration')
    if(.not.all(ieee_is_finite(born)).or..not.all(ieee_is_finite(residual))) &
         call fail('nonfinite virtual observation')
    if(size(x)/=nx) call fail('MINT coordinate mismatch')
    if(.not.all(ieee_is_finite(x))) call fail('nonfinite MINT coordinates')
    if(any(x<0d0).or.any(x>1d0).or.(bins/=4.and.bins/=8)) call fail('invalid MINT coordinates/bins')
    serial=serial+1
    do k=1,norders
       approximation=raw_ratio(k)*born(k)
       exact=loop_probability*residual(k)+approximation
       if(.not.ieee_is_finite(exact).or..not.ieee_is_finite(approximation)) &
            call fail('nonfinite reconstructed virtual')
       ! Retain exact zeros and signed/zero Born values.  Probability is kept so
       ! the controller can undo selective expensive-loop sampling in the fit.
       write(sample_unit,'(5(I0,1X),*(ES25.16E3,1X))') serial,channel,iteration_serial,k,bins, &
            loop_probability,born(k),exact,approximation,features,x
    enddo
    recorded=.true.
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
    integer :: unit,ios,version_orders,k,order,r,nt,degree,model_nx,nm
    real(dp) :: unique(maxlegs)
    integer, allocatable :: incoming_int(:)
    character(len=65536) :: line
    open(newunit=unit,file='virtual_pool_model.dat',status='old',action='read',iostat=ios)
    if(ios/=0) call fail('apply mode requires virtual_pool_model.dat')
    read(unit,'(A)',iostat=ios) line
    if(ios/=0) call fail('missing model header')
    if(trim(line)/='VIRTUAL_POOL_MODEL 2') call fail('unsupported model header; restart integration')
    call record_line(unit,line,5)
    read(line,*,iostat=ios) catalog,nf,version_orders,nlegs,model_nx
    if(ios/=0) call fail('invalid model dimensions')
    if(model_nx/=nx) call fail('model MINT coordinate mismatch')
    if(catalog/=5.or.nf<1.or.nf>maxfeatures.or.nlegs<3.or.nlegs>maxlegs) &
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
    if(nf/=1+nlegs*(nlegs-1)/2+nm) call fail('invalid invariant catalog feature count')
    call record_line(unit,line,nlegs)
    read(line,*,iostat=ios) incoming_int
    if(ios/=0) call fail('invalid incoming flags')
    if(any(incoming_int<0).or.any(incoming_int>1).or.sum(incoming_int)/=2) &
         call fail('model requires two incoming legs')
    saved_incoming=incoming_int==1
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
       do r=1,nt
          call record_line(unit,line,3)
          read(line,*,iostat=ios) models(k)%a(r),models(k)%b(r),models(k)%coefficient(r)
          if(ios/=0) call fail('invalid polynomial term')
          if(min(models(k)%a(r),models(k)%b(r))<0.or.max(models(k)%a(r),models(k)%b(r))>nf) &
               call fail('invalid polynomial indices')
          if(models(k)%a(r)==0.and.models(k)%b(r)/=0) call fail('invalid polynomial constant')
          if(degree<2.and.models(k)%b(r)>0) call fail('term exceeds polynomial degree')
          if(degree==0.and.models(k)%a(r)>0) call fail('term exceeds polynomial degree')
          if(.not.ieee_is_finite(models(k)%coefficient(r))) call fail('nonfinite polynomial coefficient')
       enddo
    enddo
    call require_end(unit)
    close(unit)
    deallocate(incoming_int)
    have_metadata=.true.
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

  subroutine pooled_virtual_invariants(p,mass,incoming,pdg,mur2,cat,x,valid)
    real(dp),intent(in) :: p(0:,:),mass(:),mur2
    logical,intent(in) :: incoming(:)
    integer,intent(in) :: pdg(:)
    integer,intent(out) :: cat
    real(dp),allocatable,intent(out) :: x(:)
    logical,intent(out) :: valid
    real(dp) :: total(0:3),shat,unique(maxlegs)
    integer :: i,j,k,n,nm
    valid=.false.
    cat=5
    n=size(pdg)
    if(n<3.or.n>maxlegs.or.size(p,1)/=4.or.size(p,2)/=n.or.size(mass)/=n.or.size(incoming)/=n) return
    if(.not.all(ieee_is_finite(p)).or..not.all(ieee_is_finite(mass)).or..not.ieee_is_finite(mur2)) return
    if(any(mass<0d0).or.mur2<=0d0.or.count(incoming)/=2) return
    total=0d0
    do i=1,n
       if(incoming(i)) total=total+p(:,i)
    enddo
    shat=dot4(total,total)
    if(shat<=0d0) return
    call distinct_masses(mass,unique,nm)
    k=1+n*(n-1)/2+nm
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
    valid=all(ieee_is_finite(x))
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

  real(dp) function dot4(a,b) result(value)
    real(dp),intent(in) :: a(0:3),b(0:3)
    value=a(0)*b(0)-dot_product(a(1:3),b(1:3))
  end function

  real(dp) function logratio(a,b) result(value)
    real(dp),intent(in) :: a,b
    value=log(max(a/b,1d-15))
  end function

end module virtual_pooled
