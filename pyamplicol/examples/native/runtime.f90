! SPDX-License-Identifier: 0BSD

program runtime_fortran
  use, intrinsic :: iso_c_binding, only: c_double, c_size_t
  use rusticol, only: rusticol_runtime
  implicit none

  type(rusticol_runtime) :: runtime
  character(len=4096) :: artifact, process_key, parameters
  real(c_double), target :: momenta(20)
  real(c_double), allocatable :: totals(:), resolved(:, :, :)
  real(c_double) :: checked_total, scale

  if (command_argument_count() < 2 .or. command_argument_count() > 3) then
    error stop "usage: runtime_fortran ARTIFACT PROCESS [PARAMETERS.json]"
  end if
  call get_command_argument(1, artifact)
  call get_command_argument(2, process_key)
  parameters = ""
  if (command_argument_count() == 3) call get_command_argument(3, parameters)

  if (len_trim(parameters) > 0) then
    call runtime%load(trim(artifact), process_key=trim(process_key), &
                      model_parameters=trim(parameters))
  else
    call runtime%load(trim(artifact), process_key=trim(process_key))
  end if
  call runtime%set_model_parameter( &
      "aS", 0.117_c_double)

  ! The external-SM d d~ > Z g g subprocess p_p_to_z_j_j_3,
  ! flattened as
  ! [point][particle][E,px,py,pz].
  momenta = [ &
      500.0_c_double, 0.0_c_double, 0.0_c_double, 500.0_c_double, &
      500.0_c_double, 0.0_c_double, 0.0_c_double, -500.0_c_double, &
      462.6501613061637_c_double, 14.340107538562991_c_double, &
      155.76435943335707_c_double, -425.7484539710246_c_double, &
      369.7738416261408_c_double, -17.479290785282917_c_double, &
      2.0064955613504103_c_double, 369.3550355960509_c_double, &
      167.57599706769557_c_double, 3.1391832467199254_c_double, &
      -157.77085499470743_c_double, 56.3934183749737_c_double ]

  call runtime%evaluate(momenta, 1_c_size_t, totals)
  call runtime%evaluate_resolved(momenta, 1_c_size_t, resolved)
  checked_total = sum(resolved(:, :, 1))
  scale = max(1.0_c_double, abs(totals(1)))
  if (abs(totals(1) - checked_total) > 1.0e-12_c_double * scale) then
    error stop "resolved components do not reproduce the total"
  end if

  write(*, '(A)') "process=" // runtime%process_key()
  write(*, '(A)') "color_accuracy=" // runtime%color_accuracy()
  write(*, '(A,ES24.16)') "total=", totals(1)
  call runtime%close()
end program runtime_fortran
