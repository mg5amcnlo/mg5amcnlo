program check_born_finiteness
  use mc_born_types, only: born_is_finite
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic
  implicit none
  ! Both signs of zero, normals, extrema and subnormals, followed by both
  ! infinities and quiet/signaling NaNs with different signs and payloads.
  integer(int64), parameter :: patterns(18)=[ &
       int(z'0000000000000000',int64), int(z'8000000000000000',int64), &
       int(z'3ff0000000000000',int64), int(z'bff0000000000000',int64), &
       int(z'7fefffffffffffff',int64), int(z'ffefffffffffffff',int64), &
       int(z'0010000000000000',int64), int(z'8010000000000000',int64), &
       int(z'0000000000000001',int64), int(z'800fffffffffffff',int64), &
       int(z'7ff0000000000000',int64), int(z'fff0000000000000',int64), &
       int(z'7ff8000000000000',int64), int(z'fff8000000000001',int64), &
       int(z'7ff0000000000001',int64), int(z'fff0000000000001',int64), &
       int(z'7fffffffffffffff',int64), int(z'ffffffffffffffff',int64)]
  real(8) :: values(size(patterns))
  logical :: finite(size(patterns)),before(size(ieee_all)),after(size(ieee_all))
  type(ieee_round_type) :: rounding
  character(len=16) :: mode

  call get_command_argument(1,mode)
  values=transfer(patterns,values)
  if (mode.ne.'standard') then
    call ieee_set_flag(ieee_all,.false.)
    call ieee_set_rounding_mode(ieee_up)
    call ieee_set_flag(ieee_underflow,.true.)
    call ieee_set_flag(ieee_inexact,.true.)
    call ieee_get_flag(ieee_all,before)
  endif
  finite=born_is_finite(values)
  if (mode.ne.'standard') then
    call ieee_get_flag(ieee_all,after)
    if (any(before.neqv.after)) stop 1
    call ieee_get_rounding_mode(rounding)
    if (rounding.ne.ieee_up) stop 2
    call ieee_set_rounding_mode(ieee_nearest)
  endif
  if (.not.all(finite(1:10))) stop 3
  if (any(finite(11:18))) stop 4
  ! Scalar evaluation must also reject a signaling NaN, even with invalid
  ! operations trapped in the calling process.
  if (born_is_finite(values(15))) stop 5
  call ieee_set_flag(ieee_all,.false.)
  write(*,*) 'PASS Born finiteness ',trim(mode)
end program
