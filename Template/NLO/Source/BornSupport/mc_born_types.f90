! Runtime-sized results never alias a subprocess's fixed Born COMMON blocks.
module mc_born_types
  implicit none
  integer, parameter :: BORN_OK=0, BORN_UNKNOWN_CONTEXT=1, BORN_BAD_SECTOR=2
  integer, parameter :: BORN_INVALID_REQUEST=3, BORN_MISSING_CORRELATION=4
  integer, parameter :: BORN_INCOMPLETE_HISTORY=5
  type BornModelState
    real(8), allocatable :: real_values(:)
    complex(8), allocatable :: complex_values(:)
  end type
  type BornHistory
    integer :: owner_sector=0, provider=0, context=0, sector=0, i=0, j=0, extra=0
    integer, allocatable :: permutation(:), flavours(:), orders(:), amplitudes(:)
  end type
  type BornMetadata
    integer :: provider=0, context=0, nexternal=0, nincoming=0, nprocesses=0
    integer :: ngraphs=0, ncolor=0, nhelicity=0, namps=0, nsqamps=0
    integer :: nsectors=0, nsplitorders=0, namplitudes=0, extra=0
    logical :: supports_correlations=.false.
    integer, allocatable :: fks(:,:)
    integer, allocatable :: born_ids(:,:), born_mothers(:,:,:), born_colours(:,:,:)
    integer, allocatable :: configurations(:), tree(:,:,:), sprop(:,:), tprid(:,:)
    logical, allocatable :: force_bw(:,:)
    logical, allocatable :: colour_amplitudes(:,:,:)
    integer, allocatable :: amplitude_orders(:,:)
    character(64), allocatable :: order_names(:)
    real(8) :: identical_factor=1d0
    logical, allocatable :: complete_histories(:)
    type(BornHistory), allocatable :: histories(:)
  end type
  type BornRequest
    integer :: sector=1, m=0, n=0, extra=0
    integer :: single_helicity=0
    integer, allocatable :: helicity(:)
    logical :: helicities=.false., colour=.false., charge=.false.
    real(8), allocatable :: charges(:)
  end type
  type BornResult
    real(8) :: born=0d0, correlation=0d0
    real(8) :: single_helicity=0d0
    ! Storage can survive a request; these flags describe the current result.
    logical :: has_helicities=.false., has_soft=.false.
    logical :: has_extra=.false., has_single_helicity=.false.
    complex(8), allocatable :: ewsudakov(:), ewsudakov_lo2(:)
    real(8), allocatable :: amplitudes(:), diagrams(:), flows(:), flow_orders(:,:)
    real(8), allocatable :: helicities(:), helicity_orders(:,:), soft(:)
    complex(8), allocatable :: counterterms(:,:), split_counterterms(:,:,:), extra(:,:)
  end type
contains
  logical function born_model_state_equal(a,b)
    type(BornModelState),intent(in) :: a,b
    born_model_state_equal=.false.
    if (.not.allocated(a%real_values).or..not.allocated(b%real_values)) return
    if (.not.allocated(a%complex_values).or..not.allocated(b%complex_values)) return
    if (size(a%real_values).ne.size(b%real_values)) return
    if (size(a%complex_values).ne.size(b%complex_values)) return
    born_model_state_equal=all(a%real_values.eq.b%real_values).and. &
                          all(a%complex_values.eq.b%complex_values)
  end function

  logical function born_helicity_state_equal(a,b)
    type(BornModelState),intent(in) :: a,b
    ! Finalization replaces this conservative default only after proving
    ! homogeneous G scaling in every used tree squared-order component.
    born_helicity_state_equal=born_model_state_equal(a,b)
  end function

  subroutine born_resize_model_state(state,nreal,ncomplex)
    type(BornModelState),intent(inout) :: state
    integer,intent(in) :: nreal,ncomplex
    if (allocated(state%real_values)) then
      if (size(state%real_values).ne.nreal) deallocate(state%real_values)
    endif
    if (allocated(state%complex_values)) then
      if (size(state%complex_values).ne.ncomplex) deallocate(state%complex_values)
    endif
    if (.not.allocated(state%real_values)) allocate(state%real_values(nreal))
    if (.not.allocated(state%complex_values)) allocate(state%complex_values(ncomplex))
  end subroutine

  subroutine born_reset_result(result)
    type(BornResult),intent(inout) :: result
    result%born=0d0
    result%correlation=0d0
    result%single_helicity=0d0
    result%has_helicities=.false.
    result%has_soft=.false.
    result%has_extra=.false.
    result%has_single_helicity=.false.
    ! Leave buffers allocated, but never expose stale optional values.
    if (allocated(result%helicities)) result%helicities=0d0
    if (allocated(result%helicity_orders)) result%helicity_orders=0d0
    if (allocated(result%soft)) result%soft=0d0
    if (allocated(result%extra)) result%extra=(0d0,0d0)
    if (allocated(result%ewsudakov)) result%ewsudakov=(0d0,0d0)
    if (allocated(result%ewsudakov_lo2)) result%ewsudakov_lo2=(0d0,0d0)
  end subroutine
end module
