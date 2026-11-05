! SPDX-License-Identifier: 0BSD
! Generate d d~ > d d~ g with the E.3 "cascade-interference" insertion first.
program correlated_fortran
  use, intrinsic :: iso_c_binding
  use rusticol
  implicit none
  type(rusticol_runtime) :: runtime
  type(rusticol_correlated_request) :: requests(3), inherited_request
  type(rusticol_spin_correlation_vector) :: spins(1)
  type(rusticol_string), allocatable :: ids(:)
  character(len=4096) :: artifact, process_key
  character(len=:), allocatable :: catalogue
  real(c_double), target :: momenta(40)
  real(c_double), allocatable :: ordinary(:), after(:)
  complex(c_double_complex), allocatable :: values(:, :), inherited(:), cleared(:)
  integer :: point, request, index
  logical :: has_born, has_cascade

  if (command_argument_count() /= 2) error stop "usage: correlated_fortran ARTIFACT PROCESS"
  call get_command_argument(1, artifact)
  call get_command_argument(2, process_key)
  call runtime%load(trim(artifact), process_key=trim(process_key))
  ids = runtime%color_correlation_ids()
  has_born = .false.
  has_cascade = .false.
  do index = 1, size(ids)
    has_born = has_born .or. ids(index)%value == "born"
    has_cascade = has_cascade .or. ids(index)%value == "cascade-interference"
  end do
  if (.not. has_born .or. .not. has_cascade) error stop "required colour correlation is not registered"
  catalogue = runtime%color_correlation_catalogue_json()
  if (len(catalogue) == 0) error stop "missing colour histories"
  momenta = real([ &
      400,0,0,400, 400,0,0,-400, 300,300,0,0, 250,-150,200,0, 250,-150,-200,0, &
      400,0,0,400, 400,0,0,-400, 300,0,300,0, 250,200,-150,0, 250,-200,-150,0 ], c_double)
  spins(1)%leg = 5_c_size_t
  allocate(spins(1)%components(4, 1))
  spins(1)%components(:, 1) = cmplx([0, 0, 0, 1], kind=c_double)
  requests(1)%color_correlation = "born"
  requests(2)%color_correlation = "cascade-interference"
  requests(3)%color_correlation = "cascade-interference"
  allocate(requests(1)%spin_vectors(0), requests(2)%spin_vectors(0))
  requests(3)%spin_vectors = spins
  call runtime%evaluate(momenta, 2_c_size_t, ordinary)
  call runtime%evaluate_correlated_many(momenta, 2_c_size_t, requests, values)
  call runtime%set_spin_correlation_vectors(spins)
  inherited_request%color_correlation = "cascade-interference"
  call runtime%evaluate_correlated(momenta, 2_c_size_t, inherited_request, inherited)
  call runtime%evaluate(momenta, 2_c_size_t, after)
  if (any(ordinary /= after)) error stop "spin setter changed ordinary evaluation"
  call runtime%set_spin_correlation_vectors()
  call runtime%evaluate_correlated(momenta, 2_c_size_t, inherited_request, cleared)
  do point = 1, 2
    if (.not. close(inherited(point), values(point, 3))) error stop "default spin source mismatch"
    if (.not. close(cleared(point), values(point, 2))) error stop "cleared spin source mismatch"
  end do
  do request = 1, size(requests)
    do point = 1, 2
      write(*, '(A,1X,I0,1X,I0,2(1X,ES25.17E3))') "VALUE", request - 1, point - 1, &
          real(values(point, request), c_double), aimag(values(point, request))
    end do
  end do
  call runtime%close()
contains
  logical function close(a, b)
    complex(c_double_complex), intent(in) :: a, b
    close = abs(a - b) <= 1e-11_c_double * max(1e-100_c_double, abs(a), abs(b))
  end function close
end program correlated_fortran
