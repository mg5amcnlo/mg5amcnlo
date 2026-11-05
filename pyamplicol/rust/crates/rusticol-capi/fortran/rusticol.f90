! SPDX-License-Identifier: 0BSD

module rusticol
  use, intrinsic :: iso_c_binding
  use, intrinsic :: iso_fortran_env, only: error_unit
  implicit none
  private

  integer(c_int), parameter, public :: RUSTICOL_STATUS_OK = 0_c_int
  integer(c_int), parameter, public :: RUSTICOL_STATUS_INVALID_ARGUMENT = 1_c_int
  integer(c_int), parameter, public :: RUSTICOL_STATUS_BUFFER_TOO_SMALL = 2_c_int
  integer(c_int), parameter, public :: RUSTICOL_STATUS_RUNTIME_ERROR = 3_c_int
  integer(c_int), parameter, public :: RUSTICOL_STATUS_PANIC = 4_c_int

  integer(c_int32_t), parameter, public :: RUSTICOL_WARM_UP_EVENT_START = 0_c_int32_t
  integer(c_int32_t), parameter, public :: RUSTICOL_WARM_UP_EVENT_UPDATE = 1_c_int32_t
  integer(c_int32_t), parameter, public :: RUSTICOL_WARM_UP_EVENT_END = 2_c_int32_t
  integer(c_int32_t), parameter, public :: RUSTICOL_WARM_UP_STAGE_PROCESS_PREPARATION = &
      0_c_int32_t
  integer(c_int32_t), parameter, public :: RUSTICOL_WARM_UP_STAGE_QUERY_FAMILY = 1_c_int32_t
  integer(c_int32_t), parameter, public :: RUSTICOL_WARM_UP_STAGE_FAMILY_FINALIZATION = &
      2_c_int32_t
  integer(c_int32_t), parameter, public :: RUSTICOL_WARM_UP_STAGE_FIRST_EVALUATION = &
      3_c_int32_t

  type, bind(C), public :: rusticol_warm_up_progress_event
    integer(c_int32_t) :: schema_version
    integer(c_int32_t) :: kind
    integer(c_int32_t) :: stage
    integer(c_int32_t) :: reserved
    integer(c_int64_t) :: completed
    integer(c_int64_t) :: total
    real(c_double) :: elapsed_seconds
    integer(c_int64_t) :: current_rss_bytes
    integer(c_int64_t) :: peak_rss_bytes
    integer(c_int64_t) :: workers
    integer(c_int32_t) :: current_rss_available
    integer(c_int32_t) :: peak_rss_available
  end type rusticol_warm_up_progress_event

  type, bind(C), public :: rusticol_warm_up_result
    integer(c_int32_t) :: schema_version
    integer(c_int32_t) :: reserved
    real(c_double) :: elapsed_seconds
    integer(c_int64_t) :: query_count
    integer(c_int64_t) :: warmed_query_count
    integer(c_int64_t) :: current_rss_bytes
    integer(c_int64_t) :: peak_rss_bytes
    integer(c_int32_t) :: current_rss_available
    integer(c_int32_t) :: peak_rss_available
    integer(c_int32_t) :: already_warm
    integer(c_int32_t) :: first_evaluation_completed
  end type rusticol_warm_up_result

  type, public :: rusticol_string
    character(len=:), allocatable :: value
  end type rusticol_string

  type, public :: rusticol_external_particle
    integer(c_size_t) :: index = 0_c_size_t
    integer(c_int32_t) :: pdg = 0_c_int32_t
  end type rusticol_external_particle

  type, public :: rusticol_helicity_configuration
    character(len=:), allocatable :: id
    integer(c_int32_t), allocatable :: helicities(:)
  end type rusticol_helicity_configuration

  type, public :: rusticol_color_component
    character(len=:), allocatable :: id
    character(len=:), allocatable :: kind
    integer(c_size_t), allocatable :: word(:)
  end type rusticol_color_component

  type, public :: rusticol_model_parameter
    character(len=:), allocatable :: name
  end type rusticol_model_parameter

  type, public :: rusticol_spin_correlation_vector
    integer(c_size_t) :: leg = 0_c_size_t ! One-based public external leg.
    ! (E,px,py,pz) by point; a single point broadcasts over the batch.
    complex(c_double_complex), allocatable :: components(:, :)
  end type rusticol_spin_correlation_vector

  type, public :: rusticol_correlated_request
    character(len=:), allocatable :: color_correlation ! Unallocated means "born".
    ! Unallocated inherits the setter; allocated size zero uses physical helicities.
    type(rusticol_spin_correlation_vector), allocatable :: spin_vectors(:)
  end type rusticol_correlated_request

  type, bind(C) :: c_spin_correlation_vector
    integer(c_size_t) :: leg, point_count
    type(c_ptr) :: components
    integer(c_size_t) :: component_count
  end type c_spin_correlation_vector

  type, bind(C) :: c_correlated_request
    type(c_ptr) :: color_correlation, spin_vectors
    integer(c_size_t) :: spin_vector_count
    integer(c_int32_t) :: use_default_spin_vectors
  end type c_correlated_request

  ! Explicit real/imaginary transport avoids assumptions about COMPLEX layout.
  type :: spin_correlation_storage
    real(c_double), allocatable :: components(:)
    type(c_spin_correlation_vector), allocatable :: vectors(:)
    character(kind=c_char), allocatable :: id(:)
  end type spin_correlation_storage

  type, public :: rusticol_runtime
    private
    type(c_ptr) :: handle = c_null_ptr
  contains
    final :: rusticol_finalize
    procedure, public :: load => rusticol_load
    procedure, public :: save => rusticol_save
    procedure, public :: load_cache => rusticol_load_cache
    procedure, public :: close => rusticol_close
    procedure, public :: is_loaded => rusticol_is_loaded
    procedure, public :: process => rusticol_process
    procedure, public :: process_key => rusticol_process_key
    procedure, public :: representative_process_key => rusticol_representative_process_key
    procedure, public :: color_accuracy => rusticol_color_accuracy
    procedure, public :: execution_mode => rusticol_execution_mode
    procedure, public :: metadata_json => rusticol_metadata_json
    procedure, public :: physics_json => rusticol_physics_json
    procedure, public :: external_particles => rusticol_external_particles
    procedure, public :: external_permutation => rusticol_external_permutation
    procedure, public :: load_kinematics_json => rusticol_load_kinematics_json
    procedure, public :: helicities => rusticol_helicities
    procedure, public :: colors => rusticol_colors
    procedure, public :: model_parameters => rusticol_model_parameters
    procedure, public :: evaluate => rusticol_evaluate
    procedure, public :: color_correlation_ids => rusticol_color_correlation_ids
    procedure, public :: color_correlation_catalogue_json => rusticol_color_correlation_catalogue_json
    procedure, public :: set_spin_correlation_vectors => rusticol_set_spin_correlation_vectors
    procedure, public :: evaluate_correlated_many => rusticol_evaluate_correlated_many
    procedure, public :: evaluate_correlated => rusticol_evaluate_correlated
    procedure, public :: warm_up => rusticol_warm_up
    procedure, public :: evaluate_selected => rusticol_evaluate_selected
    procedure, public :: evaluate_resolved => rusticol_evaluate_resolved
    procedure, public :: set_model_parameter => rusticol_set_model_parameter
    procedure, public :: set_model_parameters => rusticol_set_model_parameters
    procedure, public :: set_model_parameters_json => rusticol_set_model_parameters_json
    procedure, public :: mute_warnings => rusticol_mute_warnings
    procedure, public :: unmute_warnings => rusticol_unmute_warnings
    procedure, public :: take_warnings_json => rusticol_take_warnings_json
  end type rusticol_runtime

  interface
    function c_rusticol_abi_version() bind(C, name="rusticol_abi_version") result(version)
      import :: c_int32_t
      integer(c_int32_t) :: version
    end function c_rusticol_abi_version

    function c_rusticol_last_error_message(buffer, capacity, required) &
        bind(C, name="rusticol_last_error_message") result(status)
      import :: c_ptr, c_size_t, c_int
      type(c_ptr), value :: buffer
      integer(c_size_t), value :: capacity
      integer(c_size_t) :: required
      integer(c_int) :: status
    end function c_rusticol_last_error_message

    function c_rusticol_runtime_load(process_dir, process_key, model_parameters, output) &
        bind(C, name="rusticol_runtime_load") result(status)
      import :: c_ptr, c_int
      type(c_ptr), value :: process_dir
      type(c_ptr), value :: process_key
      type(c_ptr), value :: model_parameters
      type(c_ptr) :: output
      integer(c_int) :: status
    end function c_rusticol_runtime_load

    function c_rusticol_runtime_free(handle) bind(C, name="rusticol_runtime_free") result(status)
      import :: c_ptr, c_int
      type(c_ptr), value :: handle
      integer(c_int) :: status
    end function c_rusticol_runtime_free

    function c_rusticol_runtime_save(handle, path) bind(C, name="rusticol_runtime_save") result(status)
      import :: c_ptr, c_int
      type(c_ptr), value :: handle, path
      integer(c_int) :: status
    end function c_rusticol_runtime_save

    function c_rusticol_runtime_load_cache(handle, path) &
        bind(C, name="rusticol_runtime_load_cache") result(status)
      import :: c_ptr, c_int
      type(c_ptr), value :: handle, path
      integer(c_int) :: status
    end function c_rusticol_runtime_load_cache

    function c_rusticol_runtime_metadata_json(handle, buffer, capacity, required) &
        bind(C, name="rusticol_runtime_metadata_json") result(status)
      import :: c_ptr, c_size_t, c_int
      type(c_ptr), value :: handle, buffer
      integer(c_size_t), value :: capacity
      integer(c_size_t) :: required
      integer(c_int) :: status
    end function c_rusticol_runtime_metadata_json

    function c_rusticol_runtime_execution_mode(handle, buffer, capacity, required) &
        bind(C, name="rusticol_runtime_execution_mode") result(status)
      import :: c_ptr, c_size_t, c_int
      type(c_ptr), value :: handle, buffer
      integer(c_size_t), value :: capacity
      integer(c_size_t) :: required
      integer(c_int) :: status
    end function c_rusticol_runtime_execution_mode

    function c_rusticol_runtime_physics_json(handle, buffer, capacity, required) &
        bind(C, name="rusticol_runtime_physics_json") result(status)
      import :: c_ptr, c_size_t, c_int
      type(c_ptr), value :: handle, buffer
      integer(c_size_t), value :: capacity
      integer(c_size_t) :: required
      integer(c_int) :: status
    end function c_rusticol_runtime_physics_json

    function c_rusticol_runtime_process(handle, buffer, capacity, required) &
        bind(C, name="rusticol_runtime_process") result(status)
      import :: c_ptr, c_size_t, c_int
      type(c_ptr), value :: handle, buffer
      integer(c_size_t), value :: capacity
      integer(c_size_t) :: required
      integer(c_int) :: status
    end function c_rusticol_runtime_process

    function c_rusticol_runtime_process_key(handle, buffer, capacity, required) &
        bind(C, name="rusticol_runtime_process_key") result(status)
      import :: c_ptr, c_size_t, c_int
      type(c_ptr), value :: handle, buffer
      integer(c_size_t), value :: capacity
      integer(c_size_t) :: required
      integer(c_int) :: status
    end function c_rusticol_runtime_process_key

    function c_rusticol_runtime_representative_process_key(handle, buffer, capacity, required) &
        bind(C, name="rusticol_runtime_representative_process_key") result(status)
      import :: c_ptr, c_size_t, c_int
      type(c_ptr), value :: handle, buffer
      integer(c_size_t), value :: capacity
      integer(c_size_t) :: required
      integer(c_int) :: status
    end function c_rusticol_runtime_representative_process_key

    function c_rusticol_runtime_color_accuracy(handle, buffer, capacity, required) &
        bind(C, name="rusticol_runtime_color_accuracy") result(status)
      import :: c_ptr, c_size_t, c_int
      type(c_ptr), value :: handle, buffer
      integer(c_size_t), value :: capacity
      integer(c_size_t) :: required
      integer(c_int) :: status
    end function c_rusticol_runtime_color_accuracy

    function c_rusticol_runtime_external_count(handle, output) &
        bind(C, name="rusticol_runtime_external_count") result(status)
      import :: c_ptr, c_size_t, c_int
      type(c_ptr), value :: handle
      integer(c_size_t) :: output
      integer(c_int) :: status
    end function c_rusticol_runtime_external_count

    function c_rusticol_runtime_external_pdg(handle, index, output) &
        bind(C, name="rusticol_runtime_external_pdg") result(status)
      import :: c_ptr, c_size_t, c_int32_t, c_int
      type(c_ptr), value :: handle
      integer(c_size_t), value :: index
      integer(c_int32_t) :: output
      integer(c_int) :: status
    end function c_rusticol_runtime_external_pdg

    function c_rusticol_runtime_external_permutation(handle, output, capacity, required) &
        bind(C, name="rusticol_runtime_external_permutation") result(status)
      import :: c_ptr, c_size_t, c_int
      type(c_ptr), value :: handle, output
      integer(c_size_t), value :: capacity
      integer(c_size_t) :: required
      integer(c_int) :: status
    end function c_rusticol_runtime_external_permutation

    function c_rusticol_runtime_load_kinematics_json(handle, path, output, capacity, required) &
        bind(C, name="rusticol_runtime_load_kinematics_json") result(status)
      import :: c_ptr, c_size_t, c_int
      type(c_ptr), value :: handle, path, output
      integer(c_size_t), value :: capacity
      integer(c_size_t) :: required
      integer(c_int) :: status
    end function c_rusticol_runtime_load_kinematics_json

    function c_rusticol_runtime_helicity_count(handle, output) &
        bind(C, name="rusticol_runtime_helicity_count") result(status)
      import :: c_ptr, c_size_t, c_int
      type(c_ptr), value :: handle
      integer(c_size_t) :: output
      integer(c_int) :: status
    end function c_rusticol_runtime_helicity_count

    function c_rusticol_runtime_helicity_id(handle, index, buffer, capacity, required) &
        bind(C, name="rusticol_runtime_helicity_id") result(status)
      import :: c_ptr, c_size_t, c_int
      type(c_ptr), value :: handle, buffer
      integer(c_size_t), value :: index, capacity
      integer(c_size_t) :: required
      integer(c_int) :: status
    end function c_rusticol_runtime_helicity_id

    function c_rusticol_runtime_helicity_vector(handle, index, output, capacity, required) &
        bind(C, name="rusticol_runtime_helicity_vector") result(status)
      import :: c_ptr, c_size_t, c_int
      type(c_ptr), value :: handle, output
      integer(c_size_t), value :: index, capacity
      integer(c_size_t) :: required
      integer(c_int) :: status
    end function c_rusticol_runtime_helicity_vector

    function c_rusticol_runtime_color_count(handle, output) &
        bind(C, name="rusticol_runtime_color_count") result(status)
      import :: c_ptr, c_size_t, c_int
      type(c_ptr), value :: handle
      integer(c_size_t) :: output
      integer(c_int) :: status
    end function c_rusticol_runtime_color_count

    function c_rusticol_runtime_color_id(handle, index, buffer, capacity, required) &
        bind(C, name="rusticol_runtime_color_id") result(status)
      import :: c_ptr, c_size_t, c_int
      type(c_ptr), value :: handle, buffer
      integer(c_size_t), value :: index, capacity
      integer(c_size_t) :: required
      integer(c_int) :: status
    end function c_rusticol_runtime_color_id

    function c_rusticol_runtime_color_kind(handle, index, buffer, capacity, required) &
        bind(C, name="rusticol_runtime_color_kind") result(status)
      import :: c_ptr, c_size_t, c_int
      type(c_ptr), value :: handle, buffer
      integer(c_size_t), value :: index, capacity
      integer(c_size_t) :: required
      integer(c_int) :: status
    end function c_rusticol_runtime_color_kind

    function c_rusticol_runtime_color_word(handle, index, output, capacity, required) &
        bind(C, name="rusticol_runtime_color_word") result(status)
      import :: c_ptr, c_size_t, c_int
      type(c_ptr), value :: handle, output
      integer(c_size_t), value :: index, capacity
      integer(c_size_t) :: required
      integer(c_int) :: status
    end function c_rusticol_runtime_color_word

    function c_rusticol_runtime_model_parameter_count(handle, output) &
        bind(C, name="rusticol_runtime_model_parameter_count") result(status)
      import :: c_ptr, c_size_t, c_int
      type(c_ptr), value :: handle
      integer(c_size_t) :: output
      integer(c_int) :: status
    end function c_rusticol_runtime_model_parameter_count

    function c_rusticol_runtime_model_parameter_name(handle, index, buffer, capacity, required) &
        bind(C, name="rusticol_runtime_model_parameter_name") result(status)
      import :: c_ptr, c_size_t, c_int
      type(c_ptr), value :: handle, buffer
      integer(c_size_t), value :: index, capacity
      integer(c_size_t) :: required
      integer(c_int) :: status
    end function c_rusticol_runtime_model_parameter_name

    function c_rusticol_runtime_resolved_shape(handle, helicity_ids, helicity_count, &
        color_ids, color_count, output_helicity_count, output_color_count) &
        bind(C, name="rusticol_runtime_resolved_shape") result(status)
      import :: c_ptr, c_size_t, c_int
      type(c_ptr), value :: handle, helicity_ids, color_ids
      integer(c_size_t), value :: helicity_count, color_count
      integer(c_size_t) :: output_helicity_count, output_color_count
      integer(c_int) :: status
    end function c_rusticol_runtime_resolved_shape

    function c_rusticol_runtime_warm_up_f64_with_cores(handle, momenta, momentum_count, &
        helicity_ids, helicity_count, color_ids, color_count, n_cores, &
        progress_callback, progress_user_data, output) &
        bind(C, name="rusticol_runtime_warm_up_f64_with_cores") result(status)
      import :: c_ptr, c_funptr, c_size_t, c_int, rusticol_warm_up_result
      type(c_ptr), value :: handle, momenta, helicity_ids, color_ids
      integer(c_size_t), value :: momentum_count, helicity_count, color_count, n_cores
      type(c_funptr), value :: progress_callback
      type(c_ptr), value :: progress_user_data
      type(rusticol_warm_up_result) :: output
      integer(c_int) :: status
    end function c_rusticol_runtime_warm_up_f64_with_cores

    function c_rusticol_color_correlation_count(handle, output) &
        bind(C, name="rusticol_runtime_color_correlation_count") result(status)
      import :: c_ptr, c_size_t, c_int
      type(c_ptr), value :: handle
      integer(c_size_t) :: output
      integer(c_int) :: status
    end function c_rusticol_color_correlation_count

    function c_rusticol_color_correlation_id(handle, index, buffer, capacity, required) &
        bind(C, name="rusticol_runtime_color_correlation_id") result(status)
      import :: c_ptr, c_size_t, c_int
      type(c_ptr), value :: handle, buffer
      integer(c_size_t), value :: index, capacity
      integer(c_size_t) :: required
      integer(c_int) :: status
    end function c_rusticol_color_correlation_id

    function c_rusticol_color_correlation_catalogue_json(handle, buffer, capacity, required) &
        bind(C, name="rusticol_runtime_color_correlation_catalogue_json") result(status)
      import :: c_ptr, c_size_t, c_int
      type(c_ptr), value :: handle, buffer
      integer(c_size_t), value :: capacity
      integer(c_size_t) :: required
      integer(c_int) :: status
    end function c_rusticol_color_correlation_catalogue_json

    function c_rusticol_set_spin_correlation_vectors(handle, vectors, vector_count) &
        bind(C, name="rusticol_runtime_set_spin_correlation_vectors_f64") result(status)
      import :: c_ptr, c_size_t, c_int
      type(c_ptr), value :: handle, vectors
      integer(c_size_t), value :: vector_count
      integer(c_int) :: status
    end function c_rusticol_set_spin_correlation_vectors

    function c_rusticol_evaluate_correlated_many(handle, momenta, momentum_count, point_count, &
        requests, request_count, helicity_ids, helicity_count, output, output_capacity) &
        bind(C, name="rusticol_runtime_evaluate_correlated_many_f64") result(status)
      import :: c_ptr, c_size_t, c_int
      type(c_ptr), value :: handle, momenta, requests, helicity_ids, output
      integer(c_size_t), value :: momentum_count, point_count, request_count
      integer(c_size_t), value :: helicity_count, output_capacity
      integer(c_int) :: status
    end function c_rusticol_evaluate_correlated_many

    function c_rusticol_runtime_evaluate_f64(handle, momenta, momentum_count, point_count, &
        output, output_capacity) bind(C, name="rusticol_runtime_evaluate_f64") result(status)
      import :: c_ptr, c_size_t, c_int
      type(c_ptr), value :: handle, momenta, output
      integer(c_size_t), value :: momentum_count, point_count, output_capacity
      integer(c_int) :: status
    end function c_rusticol_runtime_evaluate_f64

    function c_rusticol_runtime_evaluate_selected_f64(handle, momenta, momentum_count, &
        point_count, helicity_ids, helicity_count, color_ids, color_count, &
        helicity_by_point, helicity_by_point_count, color_flow_by_point, &
        color_flow_by_point_count, &
        output, output_capacity) bind(C, name="rusticol_runtime_evaluate_selected_f64") &
        result(status)
      import :: c_ptr, c_size_t, c_int
      type(c_ptr), value :: handle, momenta, helicity_ids, color_ids
      type(c_ptr), value :: helicity_by_point, color_flow_by_point, output
      integer(c_size_t), value :: momentum_count, point_count, helicity_count, color_count
      integer(c_size_t), value :: helicity_by_point_count, color_flow_by_point_count
      integer(c_size_t), value :: output_capacity
      integer(c_int) :: status
    end function c_rusticol_runtime_evaluate_selected_f64

    function c_rusticol_runtime_evaluate_resolved_f64(handle, momenta, momentum_count, &
        point_count, helicity_ids, helicity_count, color_ids, color_count, output, &
        output_capacity, output_helicity_count, output_color_count) &
        bind(C, name="rusticol_runtime_evaluate_resolved_f64") result(status)
      import :: c_ptr, c_size_t, c_int
      type(c_ptr), value :: handle, momenta, helicity_ids, color_ids, output
      integer(c_size_t), value :: momentum_count, point_count, helicity_count, color_count
      integer(c_size_t), value :: output_capacity
      integer(c_size_t) :: output_helicity_count, output_color_count
      integer(c_int) :: status
    end function c_rusticol_runtime_evaluate_resolved_f64

    function c_rusticol_runtime_set_model_parameter(handle, name, real, imaginary) &
        bind(C, name="rusticol_runtime_set_model_parameter") result(status)
      import :: c_ptr, c_double, c_int
      type(c_ptr), value :: handle, name
      real(c_double), value :: real, imaginary
      integer(c_int) :: status
    end function c_rusticol_runtime_set_model_parameter

    function c_rusticol_runtime_set_model_parameters(handle, names, real, imaginary, count) &
        bind(C, name="rusticol_runtime_set_model_parameters") result(status)
      import :: c_ptr, c_size_t, c_int
      type(c_ptr), value :: handle, names, real, imaginary
      integer(c_size_t), value :: count
      integer(c_int) :: status
    end function c_rusticol_runtime_set_model_parameters

    function c_rusticol_runtime_set_model_parameters_json(handle, path) &
        bind(C, name="rusticol_runtime_set_model_parameters_json") result(status)
      import :: c_ptr, c_int
      type(c_ptr), value :: handle, path
      integer(c_int) :: status
    end function c_rusticol_runtime_set_model_parameters_json

    function c_rusticol_runtime_mute_warnings(handle, muted) &
        bind(C, name="rusticol_runtime_mute_warnings") result(status)
      import :: c_ptr, c_int
      type(c_ptr), value :: handle
      integer(c_int), value :: muted
      integer(c_int) :: status
    end function c_rusticol_runtime_mute_warnings

    function c_rusticol_runtime_take_warnings_json(handle, buffer, capacity, required) &
        bind(C, name="rusticol_runtime_take_warnings_json") result(status)
      import :: c_ptr, c_size_t, c_int
      type(c_ptr), value :: handle, buffer
      integer(c_size_t), value :: capacity
      integer(c_size_t) :: required
      integer(c_int) :: status
    end function c_rusticol_runtime_take_warnings_json
  end interface

  abstract interface
    function rusticol_warm_up_progress_callback(event, user_data) bind(C) result(keep_going)
      import :: rusticol_warm_up_progress_event, c_ptr, c_int
      type(rusticol_warm_up_progress_event), intent(in) :: event
      type(c_ptr), value :: user_data
      integer(c_int) :: keep_going
    end function rusticol_warm_up_progress_callback

    function c_runtime_string_function(handle, buffer, capacity, required) result(status)
      import :: c_ptr, c_size_t, c_int
      type(c_ptr), value :: handle, buffer
      integer(c_size_t), value :: capacity
      integer(c_size_t) :: required
      integer(c_int) :: status
    end function c_runtime_string_function

    function c_runtime_indexed_string_function(handle, index, buffer, capacity, required) &
        result(status)
      import :: c_ptr, c_size_t, c_int
      type(c_ptr), value :: handle, buffer
      integer(c_size_t), value :: index, capacity
      integer(c_size_t) :: required
      integer(c_int) :: status
    end function c_runtime_indexed_string_function
  end interface

  public :: rusticol_abi_version, rusticol_last_error, rusticol_warm_up_progress_callback

contains

  function rusticol_abi_version() result(version)
    integer(c_int32_t) :: version
    version = c_rusticol_abi_version()
  end function rusticol_abi_version

  subroutine build_c_string(value, buffer)
    character(len=*), intent(in) :: value
    character(kind=c_char), allocatable, target, intent(out) :: buffer(:)
    integer :: index, length

    length = len_trim(value)
    allocate(buffer(length + 1))
    do index = 1, length
      buffer(index) = value(index:index)
    end do
    buffer(length + 1) = c_null_char
  end subroutine build_c_string

  function from_c_string(buffer) result(value)
    character(kind=c_char), intent(in) :: buffer(:)
    character(len=:), allocatable :: value
    integer :: index, length

    length = 0
    do index = 1, size(buffer)
      if (buffer(index) == c_null_char) exit
      length = length + 1
    end do
    allocate(character(len=length) :: value)
    do index = 1, length
      value(index:index) = buffer(index)
    end do
  end function from_c_string

  function rusticol_last_error() result(message)
    character(len=:), allocatable :: message
    character(kind=c_char), allocatable, target :: buffer(:)
    integer(c_size_t) :: required
    integer(c_int) :: status

    required = 0_c_size_t
    status = c_rusticol_last_error_message(c_null_ptr, 0_c_size_t, required)
    if (status /= RUSTICOL_STATUS_OK .or. required == 0_c_size_t) then
      message = "unknown Rusticol error"
      return
    end if
    allocate(buffer(required))
    status = c_rusticol_last_error_message(c_loc(buffer(1)), required, required)
    if (status /= RUSTICOL_STATUS_OK) then
      message = "could not retrieve Rusticol error"
    else
      message = from_c_string(buffer)
    end if
  end function rusticol_last_error

  logical function status_ok(status, ierr)
    integer(c_int), intent(in) :: status
    integer(c_int), intent(out), optional :: ierr

    if (present(ierr)) ierr = status
    if (status /= RUSTICOL_STATUS_OK .and. .not. present(ierr)) then
      write(error_unit, '(A)') rusticol_last_error()
      error stop 1
    end if
    status_ok = status == RUSTICOL_STATUS_OK
  end function status_ok

  logical function argument_ok(condition, message, ierr)
    logical, intent(in) :: condition
    character(len=*), intent(in) :: message
    integer(c_int), intent(out), optional :: ierr

    argument_ok = condition
    if (condition) return
    if (present(ierr)) then
      ierr = RUSTICOL_STATUS_INVALID_ARGUMENT
    else
      write(error_unit, '(A)') message
      error stop 1
    end if
  end function argument_ok

  subroutine rusticol_load(self, process_dir, process_key, model_parameters, ierr)
    class(rusticol_runtime), intent(inout) :: self
    character(len=*), intent(in) :: process_dir
    character(len=*), intent(in), optional :: process_key, model_parameters
    integer(c_int), intent(out), optional :: ierr
    character(kind=c_char), allocatable, target :: dir_c(:), key_c(:), model_c(:)
    type(c_ptr) :: key_ptr, model_ptr
    integer(c_int) :: status

    call self%close()
    call build_c_string(process_dir, dir_c)
    key_ptr = c_null_ptr
    if (present(process_key)) then
      if (len_trim(process_key) > 0) then
        call build_c_string(process_key, key_c)
        key_ptr = c_loc(key_c(1))
      end if
    end if
    model_ptr = c_null_ptr
    if (present(model_parameters)) then
      if (len_trim(model_parameters) > 0) then
        call build_c_string(model_parameters, model_c)
        model_ptr = c_loc(model_c(1))
      end if
    end if
    status = c_rusticol_runtime_load(c_loc(dir_c(1)), key_ptr, model_ptr, self%handle)
    if (.not. status_ok(status, ierr)) self%handle = c_null_ptr
  end subroutine rusticol_load

  ! Save completed OTF cache entries; the generated process output is still required.
  subroutine rusticol_save(self, path, ierr)
    class(rusticol_runtime), intent(in) :: self
    character(len=*), intent(in) :: path
    integer(c_int), intent(out), optional :: ierr
    character(kind=c_char), allocatable, target :: path_c(:)
    integer(c_int) :: status

    if (.not. argument_ok(index(path, achar(0)) == 0, "cache path contains a NUL byte", ierr)) return
    call build_c_string(path, path_c)
    status = c_rusticol_runtime_save(self%handle, c_loc(path_c(1)))
    if (.not. status_ok(status, ierr)) return
  end subroutine rusticol_save

  ! Load a saved OTF cache into an already-loaded matching process output.
  subroutine rusticol_load_cache(self, path, ierr)
    class(rusticol_runtime), intent(inout) :: self
    character(len=*), intent(in) :: path
    integer(c_int), intent(out), optional :: ierr
    character(kind=c_char), allocatable, target :: path_c(:)
    integer(c_int) :: status

    if (.not. argument_ok(index(path, achar(0)) == 0, "cache path contains a NUL byte", ierr)) return
    call build_c_string(path, path_c)
    status = c_rusticol_runtime_load_cache(self%handle, c_loc(path_c(1)))
    if (.not. status_ok(status, ierr)) return
  end subroutine rusticol_load_cache

  subroutine rusticol_close(self)
    class(rusticol_runtime), intent(inout) :: self
    integer(c_int) :: status
    if (c_associated(self%handle)) then
      status = c_rusticol_runtime_free(self%handle)
      self%handle = c_null_ptr
    end if
  end subroutine rusticol_close

  subroutine rusticol_finalize(self)
    type(rusticol_runtime), intent(inout) :: self
    call self%close()
  end subroutine rusticol_finalize

  logical function rusticol_is_loaded(self)
    class(rusticol_runtime), intent(in) :: self
    rusticol_is_loaded = c_associated(self%handle)
  end function rusticol_is_loaded

  function runtime_string(self, function, ierr) result(value)
    class(rusticol_runtime), intent(in) :: self
    procedure(c_runtime_string_function) :: function
    integer(c_int), intent(out), optional :: ierr
    character(len=:), allocatable :: value
    character(kind=c_char), allocatable, target :: buffer(:)
    integer(c_size_t) :: required
    integer(c_int) :: status

    required = 0_c_size_t
    status = function(self%handle, c_null_ptr, 0_c_size_t, required)
    if (.not. status_ok(status, ierr)) then
      value = ""
      return
    end if
    allocate(buffer(required))
    status = function(self%handle, c_loc(buffer(1)), required, required)
    if (.not. status_ok(status, ierr)) then
      value = ""
      return
    end if
    value = from_c_string(buffer)
  end function runtime_string

  function indexed_runtime_string(self, index, function, ierr) result(value)
    class(rusticol_runtime), intent(in) :: self
    integer(c_size_t), intent(in) :: index
    procedure(c_runtime_indexed_string_function) :: function
    integer(c_int), intent(out), optional :: ierr
    character(len=:), allocatable :: value
    character(kind=c_char), allocatable, target :: buffer(:)
    integer(c_size_t) :: required
    integer(c_int) :: status

    required = 0_c_size_t
    status = function(self%handle, index, c_null_ptr, 0_c_size_t, required)
    if (.not. status_ok(status, ierr)) then
      value = ""
      return
    end if
    allocate(buffer(required))
    status = function(self%handle, index, c_loc(buffer(1)), required, required)
    if (.not. status_ok(status, ierr)) then
      value = ""
      return
    end if
    value = from_c_string(buffer)
  end function indexed_runtime_string

  function rusticol_process(self, ierr) result(value)
    class(rusticol_runtime), intent(in) :: self
    integer(c_int), intent(out), optional :: ierr
    character(len=:), allocatable :: value
    value = runtime_string(self, c_rusticol_runtime_process, ierr)
  end function rusticol_process

  function rusticol_process_key(self, ierr) result(value)
    class(rusticol_runtime), intent(in) :: self
    integer(c_int), intent(out), optional :: ierr
    character(len=:), allocatable :: value
    value = runtime_string(self, c_rusticol_runtime_process_key, ierr)
  end function rusticol_process_key

  function rusticol_representative_process_key(self, ierr) result(value)
    class(rusticol_runtime), intent(in) :: self
    integer(c_int), intent(out), optional :: ierr
    character(len=:), allocatable :: value
    value = runtime_string(self, c_rusticol_runtime_representative_process_key, ierr)
  end function rusticol_representative_process_key

  function rusticol_color_accuracy(self, ierr) result(value)
    class(rusticol_runtime), intent(in) :: self
    integer(c_int), intent(out), optional :: ierr
    character(len=:), allocatable :: value
    value = runtime_string(self, c_rusticol_runtime_color_accuracy, ierr)
  end function rusticol_color_accuracy

  function rusticol_execution_mode(self, ierr) result(value)
    class(rusticol_runtime), intent(in) :: self
    integer(c_int), intent(out), optional :: ierr
    character(len=:), allocatable :: value
    value = runtime_string(self, c_rusticol_runtime_execution_mode, ierr)
  end function rusticol_execution_mode

  function rusticol_metadata_json(self, ierr) result(value)
    class(rusticol_runtime), intent(in) :: self
    integer(c_int), intent(out), optional :: ierr
    character(len=:), allocatable :: value
    value = runtime_string(self, c_rusticol_runtime_metadata_json, ierr)
  end function rusticol_metadata_json

  function rusticol_physics_json(self, ierr) result(value)
    class(rusticol_runtime), intent(in) :: self
    integer(c_int), intent(out), optional :: ierr
    character(len=:), allocatable :: value
    value = runtime_string(self, c_rusticol_runtime_physics_json, ierr)
  end function rusticol_physics_json

  function rusticol_external_particles(self, ierr) result(particles)
    class(rusticol_runtime), intent(in) :: self
    integer(c_int), intent(out), optional :: ierr
    type(rusticol_external_particle), allocatable :: particles(:)
    integer(c_size_t) :: count, index
    integer(c_int) :: status

    status = c_rusticol_runtime_external_count(self%handle, count)
    if (.not. status_ok(status, ierr)) then
      allocate(particles(0))
      return
    end if
    allocate(particles(count))
    do index = 1, count
      particles(index)%index = index - 1
      status = c_rusticol_runtime_external_pdg(self%handle, index - 1, particles(index)%pdg)
      if (.not. status_ok(status, ierr)) return
    end do
  end function rusticol_external_particles

  function rusticol_external_permutation(self, ierr) result(permutation)
    class(rusticol_runtime), intent(in) :: self
    integer(c_int), intent(out), optional :: ierr
    integer(c_size_t), allocatable, target :: permutation(:)
    integer(c_size_t) :: required
    integer(c_int) :: status

    required = 0_c_size_t
    status = c_rusticol_runtime_external_permutation( &
        self%handle, c_null_ptr, 0_c_size_t, required)
    if (.not. status_ok(status, ierr)) then
      allocate(permutation(0))
      return
    end if
    allocate(permutation(required))
    if (required > 0) then
      status = c_rusticol_runtime_external_permutation( &
          self%handle, c_loc(permutation(1)), required, required)
      if (.not. status_ok(status, ierr)) permutation = 0_c_size_t
    end if
  end function rusticol_external_permutation

  subroutine rusticol_load_kinematics_json(self, path, momenta, ierr)
    class(rusticol_runtime), intent(in) :: self
    character(len=*), intent(in) :: path
    real(c_double), allocatable, target, intent(out) :: momenta(:)
    integer(c_int), intent(out), optional :: ierr
    character(kind=c_char), allocatable, target :: path_c(:)
    integer(c_size_t) :: required
    integer(c_int) :: status

    call build_c_string(path, path_c)
    required = 0_c_size_t
    status = c_rusticol_runtime_load_kinematics_json( &
        self%handle, c_loc(path_c(1)), c_null_ptr, 0_c_size_t, required)
    if (.not. status_ok(status, ierr)) then
      allocate(momenta(0))
      return
    end if
    allocate(momenta(required))
    if (required > 0) then
      status = c_rusticol_runtime_load_kinematics_json( &
          self%handle, c_loc(path_c(1)), c_loc(momenta(1)), required, required)
      if (.not. status_ok(status, ierr)) momenta = 0.0_c_double
    end if
  end subroutine rusticol_load_kinematics_json

  function rusticol_helicities(self, ierr) result(items)
    class(rusticol_runtime), intent(in) :: self
    integer(c_int), intent(out), optional :: ierr
    type(rusticol_helicity_configuration), allocatable :: items(:)
    integer(c_int32_t), allocatable, target :: helicity_values(:)
    integer(c_size_t) :: count, index, required
    integer(c_int) :: status

    status = c_rusticol_runtime_helicity_count(self%handle, count)
    if (.not. status_ok(status, ierr)) then
      allocate(items(0))
      return
    end if
    allocate(items(count))
    do index = 1, count
      items(index)%id = indexed_runtime_string( &
          self, index - 1, c_rusticol_runtime_helicity_id, ierr)
      required = 0_c_size_t
      status = c_rusticol_runtime_helicity_vector( &
          self%handle, index - 1, c_null_ptr, 0_c_size_t, required)
      if (.not. status_ok(status, ierr)) return
      allocate(helicity_values(required))
      status = c_rusticol_runtime_helicity_vector( &
          self%handle, index - 1, c_loc(helicity_values(1)), required, required)
      if (.not. status_ok(status, ierr)) return
      call move_alloc(helicity_values, items(index)%helicities)
    end do
  end function rusticol_helicities

  function rusticol_colors(self, ierr) result(items)
    class(rusticol_runtime), intent(in) :: self
    integer(c_int), intent(out), optional :: ierr
    type(rusticol_color_component), allocatable :: items(:)
    integer(c_size_t), allocatable, target :: word_values(:)
    integer(c_size_t) :: count, index, required
    integer(c_int) :: status

    status = c_rusticol_runtime_color_count(self%handle, count)
    if (.not. status_ok(status, ierr)) then
      allocate(items(0))
      return
    end if
    allocate(items(count))
    do index = 1, count
      items(index)%id = indexed_runtime_string(self, index - 1, c_rusticol_runtime_color_id, ierr)
      items(index)%kind = indexed_runtime_string( &
          self, index - 1, c_rusticol_runtime_color_kind, ierr)
      required = 0_c_size_t
      status = c_rusticol_runtime_color_word( &
          self%handle, index - 1, c_null_ptr, 0_c_size_t, required)
      if (.not. status_ok(status, ierr)) return
      allocate(word_values(required))
      if (required > 0) then
        status = c_rusticol_runtime_color_word( &
            self%handle, index - 1, c_loc(word_values(1)), required, required)
        if (.not. status_ok(status, ierr)) return
      end if
      call move_alloc(word_values, items(index)%word)
    end do
  end function rusticol_colors

  function rusticol_model_parameters(self, ierr) result(items)
    class(rusticol_runtime), intent(in) :: self
    integer(c_int), intent(out), optional :: ierr
    type(rusticol_model_parameter), allocatable :: items(:)
    integer(c_size_t) :: count, index
    integer(c_int) :: status

    status = c_rusticol_runtime_model_parameter_count(self%handle, count)
    if (.not. status_ok(status, ierr)) then
      allocate(items(0))
      return
    end if
    allocate(items(count))
    do index = 1, count
      items(index)%name = indexed_runtime_string( &
          self, index - 1, c_rusticol_runtime_model_parameter_name, ierr)
    end do
  end function rusticol_model_parameters

  subroutine build_c_string_array(values, storage, pointers)
    character(len=*), intent(in), optional :: values(:)
    character(kind=c_char), allocatable, target, intent(out) :: storage(:, :)
    type(c_ptr), allocatable, target, intent(out) :: pointers(:)
    integer :: count, index, character_index, max_length

    if (.not. present(values)) then
      allocate(storage(1, 0), pointers(0))
      return
    end if
    count = size(values)
    if (count == 0) then
      allocate(storage(1, 0), pointers(0))
      return
    end if
    max_length = max(1, maxval(len_trim(values)))
    allocate(storage(max_length + 1, count), pointers(count))
    storage = c_null_char
    do index = 1, count
      do character_index = 1, len_trim(values(index))
        storage(character_index, index) = values(index)(character_index:character_index)
      end do
      pointers(index) = c_loc(storage(1, index))
    end do
  end subroutine build_c_string_array

  function rusticol_color_correlation_catalogue_json(self, ierr) result(value)
    class(rusticol_runtime), intent(inout) :: self
    integer(c_int), intent(out), optional :: ierr
    character(len=:), allocatable :: value
    value = runtime_string(self, c_rusticol_color_correlation_catalogue_json, ierr)
  end function rusticol_color_correlation_catalogue_json

  function rusticol_color_correlation_ids(self, ierr) result(items)
    class(rusticol_runtime), intent(inout) :: self
    integer(c_int), intent(out), optional :: ierr
    type(rusticol_string), allocatable :: items(:)
    integer(c_size_t) :: count, index
    integer(c_int) :: status

    status = c_rusticol_color_correlation_count(self%handle, count)
    if (.not. status_ok(status, ierr)) then
      allocate(items(0))
      return
    end if
    allocate(items(count))
    do index = 1, count
      items(index)%value = indexed_runtime_string( &
          self, index - 1, c_rusticol_color_correlation_id, status)
      if (.not. status_ok(status, ierr)) return
    end do
  end function rusticol_color_correlation_ids

  logical function build_spin_storage(input, storage, ierr) result(ok)
    type(rusticol_spin_correlation_vector), intent(in) :: input(:)
    type(spin_correlation_storage), intent(out), target :: storage
    integer(c_int), intent(out), optional :: ierr
    integer(c_size_t) :: total, count, offset, first, item, point, component

    ok = .false.
    total = 0_c_size_t
    do item = 1, size(input, kind=c_size_t)
      if (.not. argument_ok(allocated(input(item)%components), &
          "Each spin vector requires allocated components(4,points)", ierr)) return
      if (.not. argument_ok(size(input(item)%components, 1) == 4 .and. &
          size(input(item)%components, 2) > 0, &
          "Each spin vector requires components(4,points) with at least one point", ierr)) return
      count = size(input(item)%components, 2, kind=c_size_t)
      if (.not. argument_ok(count <= (huge(total) - total) / 8_c_size_t, &
          "Spin-vector component count overflows", ierr)) return
      total = total + 8_c_size_t * count
    end do
    allocate(storage%components(total), storage%vectors(size(input)))
    offset = 1_c_size_t
    do item = 1, size(input, kind=c_size_t)
      first = offset
      count = size(input(item)%components, 2, kind=c_size_t)
      do point = 1, count
        do component = 1, 4
          storage%components(offset) = real(input(item)%components(component, point), c_double)
          storage%components(offset + 1) = aimag(input(item)%components(component, point))
          offset = offset + 2
        end do
      end do
      storage%vectors(item) = c_spin_correlation_vector(input(item)%leg, count, &
          c_loc(storage%components(first)), 8_c_size_t * count)
    end do
    if (present(ierr)) ierr = RUSTICOL_STATUS_OK
    ok = .true.
  end function build_spin_storage

  subroutine rusticol_set_spin_correlation_vectors(self, vectors, ierr)
    class(rusticol_runtime), intent(inout) :: self
    type(rusticol_spin_correlation_vector), intent(in), optional :: vectors(:)
    integer(c_int), intent(out), optional :: ierr
    type(spin_correlation_storage), target :: storage
    type(c_ptr) :: vector_pointer
    integer(c_size_t) :: count
    integer(c_int) :: status

    vector_pointer = c_null_ptr
    count = 0_c_size_t
    if (present(vectors)) then
      if (.not. build_spin_storage(vectors, storage, ierr)) return
      count = size(storage%vectors, kind=c_size_t)
      if (count > 0) vector_pointer = c_loc(storage%vectors(1))
    end if
    status = c_rusticol_set_spin_correlation_vectors(self%handle, vector_pointer, count)
    if (.not. status_ok(status, ierr)) return
  end subroutine rusticol_set_spin_correlation_vectors

  subroutine rusticol_evaluate_correlated_many(self, momenta, point_count, requests, values, &
      helicity_ids, ierr)
    class(rusticol_runtime), intent(inout) :: self
    real(c_double), intent(in), contiguous, target :: momenta(:)
    integer(c_size_t), intent(in) :: point_count
    type(rusticol_correlated_request), intent(in) :: requests(:)
    ! values(point,request): Fortran ordering matches request-major C output.
    complex(c_double_complex), allocatable, intent(out) :: values(:, :)
    character(len=*), intent(in), optional :: helicity_ids(:)
    integer(c_int), intent(out), optional :: ierr
    type(spin_correlation_storage), allocatable, target :: storage(:)
    type(c_correlated_request), allocatable, target :: raw_requests(:)
    character(kind=c_char), allocatable, target :: helicity_storage(:, :)
    type(c_ptr), allocatable, target :: helicity_pointers(:)
    type(c_ptr) :: helicity_pointer
    real(c_double), allocatable, target :: raw(:)
    integer(c_size_t) :: request_count, index, point, offset, count
    integer(c_int) :: status

    allocate(values(0, 0))
    request_count = size(requests, kind=c_size_t)
    if (.not. argument_ok(point_count > 0 .and. request_count > 0 .and. size(momenta) > 0, &
        "Correlated evaluation requires positive point, request and momentum counts", ierr)) return
    if (.not. argument_ok(point_count <= huge(point_count) / 2_c_size_t / request_count, &
        "Correlated output dimensions overflow", ierr)) return
    allocate(storage(request_count), raw_requests(request_count))
    do index = 1, request_count
      if (allocated(requests(index)%color_correlation)) then
        if (.not. argument_ok(scan(requests(index)%color_correlation, c_null_char) == 0, &
            "Colour correlation ID contains a NUL byte", ierr)) return
      end if
      ! build_spin_storage initializes all storage fields, so assign the ID last.
      raw_requests(index)%spin_vectors = c_null_ptr
      raw_requests(index)%spin_vector_count = 0_c_size_t
      raw_requests(index)%use_default_spin_vectors = 1_c_int32_t
      if (allocated(requests(index)%spin_vectors)) then
        if (.not. build_spin_storage(requests(index)%spin_vectors, storage(index), ierr)) return
        count = size(storage(index)%vectors, kind=c_size_t)
        raw_requests(index)%spin_vector_count = count
        raw_requests(index)%use_default_spin_vectors = 0_c_int32_t
        if (count > 0) raw_requests(index)%spin_vectors = c_loc(storage(index)%vectors(1))
      end if
      if (allocated(requests(index)%color_correlation)) then
        call build_c_string(requests(index)%color_correlation, storage(index)%id)
      else
        call build_c_string("born", storage(index)%id)
      end if
      raw_requests(index)%color_correlation = c_loc(storage(index)%id(1))
    end do
    if (present(helicity_ids)) then
      do index = 1, size(helicity_ids, kind=c_size_t)
        if (.not. argument_ok(scan(helicity_ids(index), c_null_char) == 0, &
            "Helicity ID contains a NUL byte", ierr)) return
      end do
    end if
    call build_c_string_array(helicity_ids, helicity_storage, helicity_pointers)
    helicity_pointer = c_null_ptr
    if (size(helicity_pointers) > 0) helicity_pointer = c_loc(helicity_pointers(1))
    allocate(raw(2_c_size_t * point_count * request_count))
    status = c_rusticol_evaluate_correlated_many(self%handle, c_loc(momenta(1)), &
        size(momenta, kind=c_size_t), point_count, c_loc(raw_requests(1)), request_count, &
        helicity_pointer, size(helicity_pointers, kind=c_size_t), c_loc(raw(1)), &
        size(raw, kind=c_size_t))
    if (.not. status_ok(status, ierr)) return
    deallocate(values)
    allocate(values(point_count, request_count))
    offset = 1_c_size_t
    do index = 1, request_count
      do point = 1, point_count
        values(point, index) = cmplx(raw(offset), raw(offset + 1), kind=c_double)
        offset = offset + 2
      end do
    end do
  end subroutine rusticol_evaluate_correlated_many

  subroutine rusticol_evaluate_correlated(self, momenta, point_count, request, values, helicity_ids, ierr)
    class(rusticol_runtime), intent(inout) :: self
    real(c_double), intent(in), contiguous, target :: momenta(:)
    integer(c_size_t), intent(in) :: point_count
    type(rusticol_correlated_request), intent(in) :: request
    complex(c_double_complex), allocatable, intent(out) :: values(:)
    character(len=*), intent(in), optional :: helicity_ids(:)
    integer(c_int), intent(out), optional :: ierr
    complex(c_double_complex), allocatable :: batch(:, :)

    call self%evaluate_correlated_many(momenta, point_count, [request], batch, helicity_ids, ierr)
    if (size(batch, 2) == 0) then
      allocate(values(0))
    else
      values = batch(:, 1)
    end if
  end subroutine rusticol_evaluate_correlated

  subroutine rusticol_evaluate(self, momenta, point_count, values, ierr)
    class(rusticol_runtime), intent(inout) :: self
    real(c_double), intent(in), target :: momenta(:)
    integer(c_size_t), intent(in) :: point_count
    real(c_double), allocatable, intent(out), target :: values(:)
    integer(c_int), intent(out), optional :: ierr
    integer(c_int) :: status

    if (.not. argument_ok(point_count > 0_c_size_t .and. size(momenta) > 0, &
        "Rusticol evaluation requires positive point and momentum counts", ierr)) then
      allocate(values(0))
      return
    end if
    allocate(values(point_count))
    status = c_rusticol_runtime_evaluate_f64( &
        self%handle, c_loc(momenta(1)), size(momenta, kind=c_size_t), point_count, &
        c_loc(values(1)), size(values, kind=c_size_t))
    if (.not. status_ok(status, ierr)) values = 0.0_c_double
  end subroutine rusticol_evaluate

  subroutine rusticol_warm_up(self, point, result, helicity_ids, color_ids, &
      progress_callback, progress_user_data, ierr, n_cores)
    class(rusticol_runtime), intent(inout) :: self
    real(c_double), intent(in), target :: point(:)
    type(rusticol_warm_up_result), intent(out) :: result
    character(len=*), intent(in), optional :: helicity_ids(:), color_ids(:)
    procedure(rusticol_warm_up_progress_callback), optional :: progress_callback
    type(c_ptr), intent(in), optional :: progress_user_data
    integer(c_int), intent(out), optional :: ierr
    integer, intent(in), optional :: n_cores
    character(kind=c_char), allocatable, target :: helicity_storage(:, :), color_storage(:, :)
    type(c_ptr), allocatable, target :: helicity_pointers(:), color_pointers(:)
    type(c_ptr) :: helicity_pointer, color_pointer, user_data_pointer
    type(c_funptr) :: callback_pointer
    integer(c_int) :: status
    integer(c_size_t) :: construction_cores

    result = rusticol_warm_up_result( &
        0_c_int32_t, 0_c_int32_t, 0.0_c_double, 0_c_int64_t, 0_c_int64_t, &
        0_c_int64_t, 0_c_int64_t, 0_c_int32_t, 0_c_int32_t, 0_c_int32_t, 0_c_int32_t)
    if (.not. argument_ok(size(point) > 0, &
        "Rusticol warm_up requires exactly one non-empty binary64 point", ierr)) return
    construction_cores = 0_c_size_t
    if (present(n_cores)) then
      if (.not. argument_ok(n_cores > 0, "Rusticol warm_up n_cores must be positive", ierr)) return
      construction_cores = int(n_cores, c_size_t)
    end if

    call build_c_string_array(helicity_ids, helicity_storage, helicity_pointers)
    call build_c_string_array(color_ids, color_storage, color_pointers)
    helicity_pointer = c_null_ptr
    if (size(helicity_pointers) > 0) helicity_pointer = c_loc(helicity_pointers(1))
    color_pointer = c_null_ptr
    if (size(color_pointers) > 0) color_pointer = c_loc(color_pointers(1))

    callback_pointer = c_null_funptr
    if (present(progress_callback)) callback_pointer = c_funloc(progress_callback)
    user_data_pointer = c_null_ptr
    if (present(progress_user_data)) user_data_pointer = progress_user_data

    status = c_rusticol_runtime_warm_up_f64_with_cores( &
        self%handle, c_loc(point(1)), size(point, kind=c_size_t), &
        helicity_pointer, size(helicity_pointers, kind=c_size_t), &
        color_pointer, size(color_pointers, kind=c_size_t), &
        construction_cores, callback_pointer, user_data_pointer, result)
    if (.not. status_ok(status, ierr)) return
  end subroutine rusticol_warm_up

  subroutine rusticol_evaluate_selected(self, momenta, point_count, values, helicity_ids, &
      color_ids, helicity_by_point, color_flow_by_point, ierr)
    class(rusticol_runtime), intent(inout) :: self
    real(c_double), intent(in), target :: momenta(:)
    integer(c_size_t), intent(in) :: point_count
    real(c_double), allocatable, intent(out), target :: values(:)
    character(len=*), intent(in), optional :: helicity_ids(:), color_ids(:)
    integer(c_int32_t), intent(in), optional, target :: helicity_by_point(:)
    integer(c_int32_t), intent(in), optional, target :: color_flow_by_point(:)
    integer(c_int), intent(out), optional :: ierr
    character(kind=c_char), allocatable, target :: helicity_storage(:, :), color_storage(:, :)
    type(c_ptr), allocatable, target :: helicity_pointers(:), color_pointers(:)
    type(c_ptr) :: helicity_pointer, color_pointer
    type(c_ptr) :: helicity_by_point_pointer, color_flow_by_point_pointer
    integer(c_size_t) :: helicity_by_point_count, color_flow_by_point_count
    logical :: has_helicity_ids, has_color_ids
    integer(c_int) :: status

    if (.not. argument_ok(point_count > 0_c_size_t .and. size(momenta) > 0, &
        "Rusticol selected evaluation requires positive point and momentum counts", ierr)) then
      allocate(values(0))
      return
    end if
    has_helicity_ids = .false.
    if (present(helicity_ids)) has_helicity_ids = size(helicity_ids) > 0
    has_color_ids = .false.
    if (present(color_ids)) has_color_ids = size(color_ids) > 0

    helicity_by_point_count = 0_c_size_t
    helicity_by_point_pointer = c_null_ptr
    if (present(helicity_by_point)) then
      helicity_by_point_count = size(helicity_by_point, kind=c_size_t)
      if (.not. argument_ok(helicity_by_point_count == 0_c_size_t .or. &
          helicity_by_point_count == point_count, &
          "helicity_by_point must contain one zero-based selector per point", ierr)) then
        allocate(values(0))
        return
      end if
      if (.not. argument_ok(helicity_by_point_count == 0_c_size_t .or. &
          all(helicity_by_point >= 0_c_int32_t), &
          "helicity_by_point selectors must be nonnegative", ierr)) then
        allocate(values(0))
        return
      end if
      if (helicity_by_point_count > 0) helicity_by_point_pointer = c_loc(helicity_by_point(1))
    end if

    color_flow_by_point_count = 0_c_size_t
    color_flow_by_point_pointer = c_null_ptr
    if (present(color_flow_by_point)) then
      color_flow_by_point_count = size(color_flow_by_point, kind=c_size_t)
      if (.not. argument_ok(color_flow_by_point_count == 0_c_size_t .or. &
          color_flow_by_point_count == point_count, &
          "color_flow_by_point must contain one zero-based selector per point", ierr)) then
        allocate(values(0))
        return
      end if
      if (.not. argument_ok(color_flow_by_point_count == 0_c_size_t .or. &
          all(color_flow_by_point >= 0_c_int32_t), &
          "color_flow_by_point selectors must be nonnegative", ierr)) then
        allocate(values(0))
        return
      end if
      if (color_flow_by_point_count > 0) then
        color_flow_by_point_pointer = c_loc(color_flow_by_point(1))
      end if
    end if

    if (.not. argument_ok(.not. (has_helicity_ids .and. helicity_by_point_count > 0), &
        "helicity_ids and helicity_by_point are mutually exclusive", ierr)) then
      allocate(values(0))
      return
    end if
    if (.not. argument_ok(.not. (has_color_ids .and. color_flow_by_point_count > 0), &
        "color_ids and color_flow_by_point are mutually exclusive", ierr)) then
      allocate(values(0))
      return
    end if

    call build_c_string_array(helicity_ids, helicity_storage, helicity_pointers)
    call build_c_string_array(color_ids, color_storage, color_pointers)
    helicity_pointer = c_null_ptr
    if (size(helicity_pointers) > 0) helicity_pointer = c_loc(helicity_pointers(1))
    color_pointer = c_null_ptr
    if (size(color_pointers) > 0) color_pointer = c_loc(color_pointers(1))

    allocate(values(point_count))
    status = c_rusticol_runtime_evaluate_selected_f64( &
        self%handle, c_loc(momenta(1)), size(momenta, kind=c_size_t), point_count, &
        helicity_pointer, size(helicity_pointers, kind=c_size_t), color_pointer, &
        size(color_pointers, kind=c_size_t), helicity_by_point_pointer, &
        helicity_by_point_count, color_flow_by_point_pointer, &
        color_flow_by_point_count, &
        c_loc(values(1)), size(values, kind=c_size_t))
    if (.not. status_ok(status, ierr)) values = 0.0_c_double
  end subroutine rusticol_evaluate_selected

  subroutine rusticol_evaluate_resolved(self, momenta, point_count, values, helicity_ids, &
      color_ids, ierr)
    class(rusticol_runtime), intent(inout) :: self
    real(c_double), intent(in), target :: momenta(:)
    integer(c_size_t), intent(in) :: point_count
    real(c_double), allocatable, intent(out), target :: values(:, :, :)
    character(len=*), intent(in), optional :: helicity_ids(:), color_ids(:)
    integer(c_int), intent(out), optional :: ierr
    character(kind=c_char), allocatable, target :: helicity_storage(:, :), color_storage(:, :)
    type(c_ptr), allocatable, target :: helicity_pointers(:), color_pointers(:)
    type(c_ptr) :: helicity_pointer, color_pointer
    integer(c_size_t) :: helicity_count, color_count
    integer(c_int) :: status

    if (.not. argument_ok(point_count > 0_c_size_t .and. size(momenta) > 0, &
        "Rusticol resolved evaluation requires positive point and momentum counts", ierr)) then
      allocate(values(0, 0, 0))
      return
    end if
    call build_c_string_array(helicity_ids, helicity_storage, helicity_pointers)
    call build_c_string_array(color_ids, color_storage, color_pointers)
    helicity_pointer = c_null_ptr
    if (size(helicity_pointers) > 0) helicity_pointer = c_loc(helicity_pointers(1))
    color_pointer = c_null_ptr
    if (size(color_pointers) > 0) color_pointer = c_loc(color_pointers(1))
    status = c_rusticol_runtime_resolved_shape( &
        self%handle, helicity_pointer, size(helicity_pointers, kind=c_size_t), &
        color_pointer, size(color_pointers, kind=c_size_t), helicity_count, color_count)
    if (.not. status_ok(status, ierr)) then
      allocate(values(0, 0, 0))
      return
    end if
    allocate(values(color_count, helicity_count, point_count))
    status = c_rusticol_runtime_evaluate_resolved_f64( &
        self%handle, c_loc(momenta(1)), size(momenta, kind=c_size_t), point_count, &
        helicity_pointer, size(helicity_pointers, kind=c_size_t), color_pointer, &
        size(color_pointers, kind=c_size_t), c_loc(values(1, 1, 1)), &
        size(values, kind=c_size_t), helicity_count, color_count)
    if (.not. status_ok(status, ierr)) values = 0.0_c_double
  end subroutine rusticol_evaluate_resolved

  subroutine rusticol_set_model_parameter(self, name, real, imaginary, ierr)
    class(rusticol_runtime), intent(inout) :: self
    character(len=*), intent(in) :: name
    real(c_double), intent(in) :: real
    real(c_double), intent(in), optional :: imaginary
    integer(c_int), intent(out), optional :: ierr
    character(kind=c_char), allocatable, target :: name_c(:)
    real(c_double) :: imaginary_value
    integer(c_int) :: status

    imaginary_value = 0.0_c_double
    if (present(imaginary)) imaginary_value = imaginary
    call build_c_string(name, name_c)
    status = c_rusticol_runtime_set_model_parameter( &
        self%handle, c_loc(name_c(1)), real, imaginary_value)
    if (.not. status_ok(status, ierr)) return
  end subroutine rusticol_set_model_parameter

  subroutine rusticol_set_model_parameters(self, names, real, imaginary, ierr)
    class(rusticol_runtime), intent(inout) :: self
    character(len=*), intent(in) :: names(:)
    real(c_double), intent(in), target :: real(:), imaginary(:)
    integer(c_int), intent(out), optional :: ierr
    character(kind=c_char), allocatable, target :: storage(:, :)
    type(c_ptr), allocatable, target :: pointers(:)
    integer(c_int) :: status

    if (.not. argument_ok(size(names) > 0 .and. size(names) == size(real) .and. &
        size(names) == size(imaginary), &
        "Rusticol model parameter arrays must be non-empty and have equal lengths", ierr)) then
      return
    end if
    call build_c_string_array(names, storage, pointers)
    status = c_rusticol_runtime_set_model_parameters( &
        self%handle, c_loc(pointers(1)), c_loc(real(1)), c_loc(imaginary(1)), &
        size(names, kind=c_size_t))
    if (.not. status_ok(status, ierr)) return
  end subroutine rusticol_set_model_parameters

  subroutine rusticol_set_model_parameters_json(self, path, ierr)
    class(rusticol_runtime), intent(inout) :: self
    character(len=*), intent(in) :: path
    integer(c_int), intent(out), optional :: ierr
    character(kind=c_char), allocatable, target :: path_c(:)
    integer(c_int) :: status

    call build_c_string(path, path_c)
    status = c_rusticol_runtime_set_model_parameters_json(self%handle, c_loc(path_c(1)))
    if (.not. status_ok(status, ierr)) return
  end subroutine rusticol_set_model_parameters_json

  subroutine rusticol_mute_warnings(self, ierr)
    class(rusticol_runtime), intent(inout) :: self
    integer(c_int), intent(out), optional :: ierr
    integer(c_int) :: status
    status = c_rusticol_runtime_mute_warnings(self%handle, 1_c_int)
    if (.not. status_ok(status, ierr)) return
  end subroutine rusticol_mute_warnings

  subroutine rusticol_unmute_warnings(self, ierr)
    class(rusticol_runtime), intent(inout) :: self
    integer(c_int), intent(out), optional :: ierr
    integer(c_int) :: status
    status = c_rusticol_runtime_mute_warnings(self%handle, 0_c_int)
    if (.not. status_ok(status, ierr)) return
  end subroutine rusticol_unmute_warnings

  function rusticol_take_warnings_json(self, ierr) result(value)
    class(rusticol_runtime), intent(inout) :: self
    integer(c_int), intent(out), optional :: ierr
    character(len=:), allocatable :: value
    value = runtime_string(self, c_rusticol_runtime_take_warnings_json, ierr)
  end function rusticol_take_warnings_json

end module rusticol
