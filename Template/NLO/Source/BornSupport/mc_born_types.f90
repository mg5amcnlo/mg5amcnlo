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
    logical :: helicities=.false., colour=.false., charge=.false.
    real(8), allocatable :: charges(:)
  end type
  type BornResult
    real(8) :: born=0d0, correlation=0d0
    real(8), allocatable :: amplitudes(:), diagrams(:), flows(:), flow_orders(:,:)
    real(8), allocatable :: helicities(:), helicity_orders(:,:), soft(:)
    complex(8), allocatable :: counterterms(:,:), split_counterterms(:,:,:), extra(:,:)
  end type
end module
