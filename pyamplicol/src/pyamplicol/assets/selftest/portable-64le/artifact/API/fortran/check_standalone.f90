! SPDX-License-Identifier: 0BSD

program check_standalone
  use, intrinsic :: iso_c_binding
  use, intrinsic :: iso_fortran_env, only: error_unit
  use rusticol
  implicit none

  integer, parameter :: MAX_OVERRIDES = 128
  type(rusticol_runtime) :: runtime
  type(rusticol_external_particle), allocatable :: particles(:)
  type(rusticol_helicity_configuration), allocatable :: helicities(:)
  type(rusticol_color_component), allocatable :: colors(:)
  character(len=4096) :: process_dir, process_key, representative_process_key
  character(len=4096) :: model_parameters, kinematics
  character(len=4096) :: executable
  character(len=256) :: override_names_buffer(MAX_OVERRIDES)
  character(len=256), allocatable :: override_names(:)
  real(c_double) :: override_real_buffer(MAX_OVERRIDES)
  real(c_double) :: override_imaginary_buffer(MAX_OVERRIDES)
  real(c_double), allocatable :: override_real(:), override_imaginary(:)
  real(c_double), allocatable, target :: momenta(:), totals(:), resolved(:, :, :)
  integer(c_size_t), allocatable :: external_permutation(:)
  logical :: json_output, point_available
  integer :: precision, override_count
  integer(c_int) :: status
  integer :: color_index, helicity_index
  real(c_double) :: explicit_total, scale

  call parse_options(process_key, model_parameters, kinematics, precision, json_output, &
      override_names_buffer, override_real_buffer, override_imaginary_buffer, &
      override_count)
  if (precision /= 16) call fail( &
      "the Fortran Rusticol API supports only double precision (--precision 16)")
  call resolve_process_dir(executable, process_dir)

  if (len_trim(process_key) > 0 .and. len_trim(model_parameters) > 0) then
    call runtime%load(trim(process_dir), trim(process_key), &
        trim(model_parameters), status)
  else if (len_trim(process_key) > 0) then
    call runtime%load(trim(process_dir), trim(process_key), ierr=status)
  else if (len_trim(model_parameters) > 0) then
    call runtime%load(trim(process_dir), model_parameters=trim(model_parameters), &
        ierr=status)
  else
    call runtime%load(trim(process_dir), ierr=status)
  end if
  if (status /= RUSTICOL_STATUS_OK) call fail(rusticol_last_error())

  if (override_count > 0) then
    allocate(override_names(override_count), override_real(override_count), &
        override_imaginary(override_count))
    override_names = override_names_buffer(1:override_count)
    override_real = override_real_buffer(1:override_count)
    override_imaginary = override_imaginary_buffer(1:override_count)
    call runtime%set_model_parameters( &
        override_names, override_real, override_imaginary, ierr=status)
    if (status /= RUSTICOL_STATUS_OK) call fail(rusticol_last_error())
  end if

  particles = runtime%external_particles(ierr=status)
  if (status /= RUSTICOL_STATUS_OK) call fail(rusticol_last_error())
  helicities = runtime%helicities(ierr=status)
  if (status /= RUSTICOL_STATUS_OK) call fail(rusticol_last_error())
  colors = runtime%colors(ierr=status)
  if (status /= RUSTICOL_STATUS_OK) call fail(rusticol_last_error())
  process_key = runtime%process_key(ierr=status)
  if (status /= RUSTICOL_STATUS_OK) call fail(rusticol_last_error())
  if (len_trim(kinematics) > 0) then
    call runtime%load_kinematics_json(trim(kinematics), momenta, ierr=status)
    if (status /= RUSTICOL_STATUS_OK) call fail(rusticol_last_error())
    point_available = .true.
  else
    representative_process_key = runtime%representative_process_key(ierr=status)
    if (status /= RUSTICOL_STATUS_OK) call fail(rusticol_last_error())
    external_permutation = runtime%external_permutation(ierr=status)
    if (status /= RUSTICOL_STATUS_OK) call fail(rusticol_last_error())
    call load_validation_point( &
        trim(process_dir) // "/API/validation_points.dat", &
        trim(representative_process_key), size(particles), momenta, point_available)
    if (point_available) call reorder_validation_point(momenta, external_permutation)
  end if

  if (.not. point_available) then
    if (json_output) then
      call emit_json_prefix(runtime, particles, helicities, colors, .false.)
      write(*, '(A)') &
          ',"diagnostic":"no bundled validation point is available"}'
    else
      write(*, '(A)') "process: " // runtime%process()
      write(*, '(A)') &
          "no bundled validation point is available; metadata load succeeded"
    end if
    stop 0
  end if

  call runtime%evaluate(momenta, 1_c_size_t, totals, ierr=status)
  if (status /= RUSTICOL_STATUS_OK) call fail(rusticol_last_error())
  call runtime%evaluate_resolved(momenta, 1_c_size_t, resolved, ierr=status)
  if (status /= RUSTICOL_STATUS_OK) call fail(rusticol_last_error())
  explicit_total = sum(resolved(:, :, 1))

  if (json_output) then
    call emit_json_prefix(runtime, particles, helicities, colors, .true.)
    write(*, '(A,I0,A,I0,A)', advance='no') &
        ',"shape":[1,', size(resolved, 2), ',', size(resolved, 1), &
        '],"values":['
    do helicity_index = 1, size(resolved, 2)
      do color_index = 1, size(resolved, 1)
        if (helicity_index > 1 .or. color_index > 1) &
            write(*, '(A)', advance='no') ','
        write(*, '(G0.17)', advance='no') &
            resolved(color_index, helicity_index, 1)
      end do
    end do
    write(*, '(A,G0.17,A,G0.17,A)') '],"resolved_sum":[', &
        explicit_total, '],"compatibility_total":[', totals(1), ']}'
  else
    write(*, '(A)') "process: " // runtime%process() // &
        " [" // trim(process_key) // "]"
    write(*, '(A,I0,A,I0,A)') "resolved shape: (1, ", &
        size(resolved, 2), ", ", size(resolved, 1), ")"
    do helicity_index = 1, size(helicities)
      do color_index = 1, size(colors)
        write(*, '(2X,A,2X,A,2X,G0.17)') &
            trim(helicities(helicity_index)%id), &
            trim(colors(color_index)%id), &
            resolved(color_index, helicity_index, 1)
      end do
    end do
    write(*, '(A,G0.17)') "explicit resolved sum: ", explicit_total
    write(*, '(A,G0.17)') "compatibility total:   ", totals(1)
  end if
  scale = max(abs(totals(1)), 1.0_c_double)
  if (abs(explicit_total - totals(1)) > 1.0e-12_c_double * scale) then
    call fail("resolved components do not reproduce the compatibility total")
  end if

contains

  subroutine fail(message)
    character(len=*), intent(in) :: message
    write(error_unit, '(A)') "check_standalone: " // trim(message)
    stop 1
  end subroutine fail

  subroutine parse_options(selected_process, model_card, kinematics_path, selected_precision, &
      json, names, real_values, imaginary_values, count)
    character(len=*), intent(out) :: selected_process, model_card, kinematics_path
    integer, intent(out) :: selected_precision, count
    logical, intent(out) :: json
    character(len=*), intent(out) :: names(:)
    real(c_double), intent(out) :: real_values(:), imaginary_values(:)
    character(len=4096) :: argument, value
    integer :: argc, index, parse_status

    selected_process = ""
    model_card = ""
    kinematics_path = ""
    selected_precision = 16
    count = 0
    json = .false.
    argc = command_argument_count()
    index = 1
    do while (index <= argc)
      call get_command_argument(index, argument)
      select case (trim(argument))
      case ("--process")
        if (index + 1 > argc) call fail("missing value after --process")
        index = index + 1
        call get_command_argument(index, selected_process)
      case ("--model-parameters")
        if (index + 1 > argc) call fail( &
            "missing value after --model-parameters")
        index = index + 1
        call get_command_argument(index, model_card)
      case ("--kinematics")
        if (index + 1 > argc) call fail("missing value after --kinematics")
        index = index + 1
        call get_command_argument(index, kinematics_path)
      case ("--set-parameter")
        if (index + 3 > argc) call fail( &
            "--set-parameter requires NAME REAL IMAG")
        if (count >= size(names)) call fail( &
            "too many --set-parameter options")
        count = count + 1
        index = index + 1
        call get_command_argument(index, names(count))
        index = index + 1
        call get_command_argument(index, value)
        read(value, *, iostat=parse_status) real_values(count)
        if (parse_status /= 0) call fail( &
            "invalid real model-parameter component")
        index = index + 1
        call get_command_argument(index, value)
        read(value, *, iostat=parse_status) imaginary_values(count)
        if (parse_status /= 0) call fail( &
            "invalid imaginary model-parameter component")
      case ("--precision")
        if (index + 1 > argc) call fail("missing value after --precision")
        index = index + 1
        call get_command_argument(index, value)
        read(value, *, iostat=parse_status) selected_precision
        if (parse_status /= 0) call fail("invalid --precision value")
      case ("--json")
        json = .true.
      case ("--help", "-h")
        write(*, '(A)') "usage: check_standalone [--process ID|EXPRESSION] " // &
            "[--model-parameters PATH] [--kinematics PATH] " // &
            "[--set-parameter NAME REAL IMAG] " // &
            "[--precision 16] [--json]"
        stop 0
      case default
        call fail("unknown option: " // trim(argument))
      end select
      index = index + 1
    end do
  end subroutine parse_options

  subroutine resolve_process_dir(program_path, root)
    character(len=*), intent(out) :: program_path, root
    integer :: slash
    logical :: artifact_in_working_directory
    call get_command_argument(0, program_path)
    inquire(file="artifact.json", exist=artifact_in_working_directory)
    if (artifact_in_working_directory) then
      root = "."
      return
    end if
    slash = scan(trim(program_path), "/", back=.true.)
    if (slash == 0) then
      root = "../.."
    else
      root = program_path(:slash - 1) // "/../.."
    end if
  end subroutine resolve_process_dir

  subroutine load_validation_point(path, key, external_count, values, available)
    character(len=*), intent(in) :: path, key
    integer, intent(in) :: external_count
    real(c_double), allocatable, target, intent(out) :: values(:)
    logical, intent(out) :: available
    character(len=65536) :: line
    character(len=4096) :: row_key
    integer :: unit, io_status, row_count

    available = .false.
    allocate(values(4 * external_count))
    open(newunit=unit, file=path, status="old", action="read", &
        iostat=io_status)
    if (io_status /= 0) return
    read(unit, '(A)', iostat=io_status) line
    if (io_status /= 0 .or. &
        trim(line) /= "RUSTICOL_VALIDATION_POINTS_V1") then
      close(unit)
      call fail("unsupported validation_points.dat format")
    end if
    do
      read(unit, '(A)', iostat=io_status) line
      if (io_status /= 0) exit
      if (len_trim(line) == 0 .or. line(1:1) == "#") cycle
      read(line, *, iostat=io_status) row_key, row_count
      if (io_status /= 0 .or. trim(row_key) /= trim(key)) cycle
      if (row_count /= external_count) then
        close(unit)
        call fail( &
            "validation point has an incompatible external-particle count")
      end if
      read(line, *, iostat=io_status) row_key, row_count, values
      if (io_status /= 0) then
        close(unit)
        call fail("invalid validation point row")
      end if
      available = .true.
      exit
    end do
    close(unit)
  end subroutine load_validation_point

  subroutine reorder_validation_point(values, permutation)
    real(c_double), allocatable, intent(inout) :: values(:)
    integer(c_size_t), intent(in) :: permutation(:)
    real(c_double), allocatable :: reordered(:)
    logical, allocatable :: seen(:)
    integer :: representative_index, public_index

    if (size(values) /= 4 * size(permutation)) call fail( &
        "validation point does not match external permutation")
    allocate(reordered(size(values)), seen(size(permutation)))
    seen = .false.
    do representative_index = 1, size(permutation)
      public_index = int(permutation(representative_index)) + 1
      if (public_index < 1 .or. public_index > size(permutation) .or. &
          seen(public_index)) call fail( &
          "runtime returned an invalid external permutation")
      seen(public_index) = .true.
      reordered(4 * public_index - 3:4 * public_index) = &
          values(4 * representative_index - 3:4 * representative_index)
    end do
    call move_alloc(reordered, values)
  end subroutine reorder_validation_point

  function json_escape(value) result(escaped)
    character(len=*), intent(in) :: value
    character(len=:), allocatable :: escaped
    integer :: index
    escaped = ""
    do index = 1, len_trim(value)
      select case (value(index:index))
      case ('"')
        escaped = escaped // '\"'
      case ('\')
        escaped = escaped // '\\'
      case default
        escaped = escaped // value(index:index)
      end select
    end do
  end function json_escape

  subroutine emit_json_prefix(active_runtime, external, helicity_items, &
      color_items, available)
    type(rusticol_runtime), intent(inout) :: active_runtime
    type(rusticol_external_particle), intent(in) :: external(:)
    type(rusticol_helicity_configuration), intent(in) :: helicity_items(:)
    type(rusticol_color_component), intent(in) :: color_items(:)
    logical, intent(in) :: available
    integer :: item, component

    write(*, '(A)', advance='no') '{"language":"fortran","available":'
    if (available) then
      write(*, '(A)', advance='no') 'true,"precision":16'
    else
      write(*, '(A)', advance='no') 'false'
    end if
    write(*, '(A)', advance='no') ',"process":"' // &
        json_escape(active_runtime%process()) // '","process_key":"' // &
        json_escape(active_runtime%process_key()) // &
        '","color_accuracy":"' // &
        json_escape(active_runtime%color_accuracy()) // &
        '","external_particles":['
    do item = 1, size(external)
      if (item > 1) write(*, '(A)', advance='no') ','
      write(*, '(A,I0,A,I0,A)', advance='no') &
          '{"index":', external(item)%index, ',"pdg":', &
          external(item)%pdg, '}'
    end do
    write(*, '(A)', advance='no') '],"helicities":['
    do item = 1, size(helicity_items)
      if (item > 1) write(*, '(A)', advance='no') ','
      write(*, '(A)', advance='no') '{"id":"' // &
          json_escape(helicity_items(item)%id) // '","helicities":['
      do component = 1, size(helicity_items(item)%helicities)
        if (component > 1) write(*, '(A)', advance='no') ','
        write(*, '(I0)', advance='no') &
            helicity_items(item)%helicities(component)
      end do
      write(*, '(A)', advance='no') ']}'
    end do
    write(*, '(A)', advance='no') '],"colors":['
    do item = 1, size(color_items)
      if (item > 1) write(*, '(A)', advance='no') ','
      write(*, '(A)', advance='no') '{"id":"' // &
          json_escape(color_items(item)%id) // '","kind":"' // &
          json_escape(color_items(item)%kind) // '","word":['
      do component = 1, size(color_items(item)%word)
        if (component > 1) write(*, '(A)', advance='no') ','
        write(*, '(I0)', advance='no') color_items(item)%word(component)
      end do
      write(*, '(A)', advance='no') ']}'
    end do
    write(*, '(A)', advance='no') ']'
  end subroutine emit_json_prefix

end program check_standalone
