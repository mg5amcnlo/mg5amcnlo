! Vendored from AmpliCol SimpleIntegrator/helper_modules.f03, commit
! 61b9cd52f44b4cdfc4773daa197e94eeb482b560 (2026-10-04).
! Module names and numerical algorithms are retained unchanged.
module pchip_uniform_strict
  use iso_fortran_env, only: real64
  implicit none
contains

  ! Compute PCHIP (Fritsch–Carlson) slopes for a UNIFORM grid.
  ! Input y(:) MUST be strictly increasing (y(i+1) > y(i) for all i).
  subroutine pchip_slopes_uniform_strict(y, m)
    real(real64), intent(in)  :: y(:)   ! can have any lower/upper bounds
    real(real64), intent(out) :: m(lbound(y,1):ubound(y,1))

    integer :: lb, ub, n, i
    real(real64), allocatable :: d(:)  ! secant slopes (uniform spacing => delta)
    real(real64) :: di, dim1, m0

    lb = lbound(y,1)
    ub = ubound(y,1)
    n  = ub - lb + 1
    if (n < 2) error stop "pchip: need at least two samples"

    allocate(d(lb:ub-1))
    do i = lb, ub-1
       d(i) = y(i+1) - y(i)
       if (d(i) <= 0.0_real64) then
          write (*,*) d(i),i,y(i+1),y(i)
          error stop "pchip: input must be strictly increasing"
       endif
    end do

    if (n == 2) then
       ! Only one interval: derivative equals the secant on both ends
       m(lb) = d(lb)
       m(ub) = d(lb)
       deallocate(d)
       return
    end if

    ! Interior slopes (Fritsch–Carlson; uniform grid => simple harmonic mean)
    do i = lb+1, ub-1
       dim1 = d(i-1); di = d(i)
       ! Both positive by strict-increasing check → harmonic mean is positive
       m(i) = 2.0_real64 * dim1 * di / (dim1 + di)
    end do

    ! Endpoint slopes (PCHIP one-sided formulas, uniform spacing)
    ! Left
    m0 = 0.5_real64 * (3.0_real64*d(lb) - d(lb+1))
    if (m0 < 0.0_real64) then
       m(lb) = 0.0_real64  ! would create a tiny flat start, but only if data near-flat
    elseif (m0 > 3.0_real64*d(lb)) then
       m(lb) = 3.0_real64*d(lb)
    else
       m(lb) = m0
    end if
    ! Right
    m0 = 0.5_real64 * (3.0_real64*d(ub-1) - d(ub-2))
    if (m0 < 0.0_real64) then
       m(ub) = 0.0_real64
    elseif (m0 > 3.0_real64*d(ub-1)) then
       m(ub) = 3.0_real64*d(ub-1)
    else
       m(ub) = m0
    end if

    deallocate(d)
  end subroutine pchip_slopes_uniform_strict


  ! Evaluate the cubic Hermite on a larger UNIFORM grid 0..size_new
  ! Original y(:) is on integer grid lb..ub (spacing 1). Endpoints are matched exactly.
  subroutine upsample_pchip_uniform(y, m, arr_new)
    real(real64), intent(in)  :: y(:)
    real(real64), intent(in)  :: m(lbound(y,1):ubound(y,1))
    real(real64), intent(out) :: arr_new(:)  ! expected bounds 0..size_new

    integer :: lb, ub, n_orig, j, size_new, k, lbn, ubn
    real(real64) :: pos, t
    real(real64) :: y0, y1, m0, m1
    real(real64) :: t2, t3, h00, h10, h01, h11

    lb = lbound(y,1)
    ub = ubound(y,1)
    n_orig = ub - lb

    lbn = lbound(arr_new,1)
    ubn = ubound(arr_new,1)
    size_new = ubn - lbn
    if (size_new < 1) then
       if (size_new == 0) then
          arr_new(lbn) = y(lb)
          return
       else
          error stop "upsample_pchip_uniform: size_new must be >= 0"
       end if
    end if

    ! Exact endpoints
    arr_new(lbn) = y(lb)
    arr_new(ubn) = y(ub)

    ! Map linearly from new integer grid to original continuous index
    do j = lbn+1, ubn-1
       pos = real(j - lbn, real64) / real(size_new, real64) * real(n_orig, real64) + real(lb, real64)

       k = int(floor(pos))
       if (k < lb) k = lb
       if (k > ub-1) k = ub-1
       t = pos - real(k, real64)

       y0 = y(k);     y1 = y(k+1)
       m0 = m(k);     m1 = m(k+1)

       t2 = t*t; t3 = t2*t
       h00 =  2.0_real64*t3 - 3.0_real64*t2 + 1.0_real64
       h10 =        t3 - 2.0_real64*t2 + t
       h01 = -2.0_real64*t3 + 3.0_real64*t2
       h11 =        t3 -       t2

       arr_new(j) = h00*y0 + h10*m0 + h01*y1 + h11*m1
    end do
  end subroutine upsample_pchip_uniform


  ! Convenience wrapper:
  ! Given arr(0:isize) strictly increasing with arr(0)=0 and arr(isize)=1,
  ! produce arr_new(0:size_new) (allocated) with strictly monotone cubic interpolation.
  subroutine resize_arr_pchip_strict(arr, size_new, arr_new)
    real(real64), intent(in)  :: arr(:)     ! e.g., bounds 0:isize
    integer,      intent(in)  :: size_new
    real(real64), intent(out) :: arr_new(0:size_new)

    real(real64), allocatable :: m(:)
    integer :: lb, ub

    lb = lbound(arr,1)
    ub = ubound(arr,1)

    if (arr(lb) /= 0.0_real64 .or. arr(ub) /= 1.0_real64) then
       error stop "resize_arr_pchip_strict: endpoints must be 0 and 1"
    end if

    allocate(m(lb:ub))
    call pchip_slopes_uniform_strict(arr, m)

    call upsample_pchip_uniform(arr, m, arr_new)

    ! No clamping needed; PCHIP is shape-preserving and stays within [0,1]
    ! If you want to guard against rare floating noise at 1 ulp, you could:
    ! arr_new(0)        = 0.0_real64
    ! arr_new(size_new) = 1.0_real64
  end subroutine resize_arr_pchip_strict

end module pchip_uniform_strict










module topk_heap_mod
  use iso_fortran_env, only: real64, int32
  implicit none
contains

  ! Sift-down in a 1-based max-heap.
  pure subroutine sift_down_max(hv, hi, n, i)
    use iso_fortran_env, only: real64, int32
    implicit none
    real(real64),   intent(inout) :: hv(:)
    integer(int32), intent(inout) :: hi(:)
    integer,        intent(in)    :: n, i
    integer :: p, l, r, s

    p = i
    do
       l = 2*p
       if (l > n) exit
       r = l + 1
       s = l
       if (r <= n) then
          if (greater_pair(hv(r), hi(r), hv(l), hi(l))) s = r
       end if
       if (.not. greater_pair(hv(s), hi(s), hv(p), hi(p))) exit
       call swap(hv(s), hv(p)); call swap_i(hi(s), hi(p))
       p = s
    end do
  end subroutine sift_down_max

  ! Build max-heap in-place from first n elements.
  pure subroutine heapify_max(hv, hi, n)
    use iso_fortran_env, only: real64, int32
    implicit none
    real(real64),   intent(inout) :: hv(:)
    integer(int32), intent(inout) :: hi(:)
    integer,        intent(in)    :: n
    integer :: i

    do i = n/2, 1, -1
       call sift_down_max(hv, hi, n, i)
    end do
  end subroutine heapify_max

  ! Extract all items from a max-heap into descending order (largest -> smallest).
  pure subroutine heap_extract_all_descending(hv, hi, n, outv, outi)
    use iso_fortran_env, only: real64, int32
    implicit none
    real(real64),   intent(inout) :: hv(:)
    integer(int32), intent(inout) :: hi(:)
    integer,        intent(in)    :: n
    real(real64),   intent(out)   :: outv(n)
    integer(int32), intent(out)   :: outi(n)
    integer :: m

    do m = 1, n
       outv(m) = hv(1)
       outi(m) = hi(1)
       ! Move last to root, shrink heap, sift down
       hv(1) = hv(n-m+1); hi(1) = hi(n-m+1)
       call sift_down_max(hv, hi, n-m, 1)
    end do
  end subroutine heap_extract_all_descending

  ! Compare pairs (value, index) with index as deterministic tiebreaker.
  pure logical function less_pair(v1, i1, v2, i2) result(is_less)
    real(real64), intent(in) :: v1, v2
    integer(int32), intent(in) :: i1, i2
    if (v1 < v2) then
       is_less = .true.
    elseif (v1 > v2) then
       is_less = .false.
    else
       is_less = (i1 < i2)
    end if
  end function less_pair

  pure logical function greater_pair(v1, i1, v2, i2) result(is_greater)
    real(real64), intent(in) :: v1, v2
    integer(int32), intent(in) :: i1, i2
    if (v1 > v2) then
       is_greater = .true.
    elseif (v1 < v2) then
       is_greater = .false.
    else
       is_greater = (i1 > i2)
    end if
  end function greater_pair

  pure subroutine swap(a, b)
    real(real64), intent(inout) :: a
    real(real64), intent(inout) :: b
    real(real64) :: t
    t = a; a = b; b = t
  end subroutine swap

  pure subroutine swap_i(a, b)
    integer(int32), intent(inout) :: a, b
    integer(int32) :: t
    t = a; a = b; b = t
  end subroutine swap_i

  ! Sift-down in a 1-based min-heap.
  pure subroutine sift_down_min(hv, hi, n, i)
    real(real64),   intent(inout) :: hv(:)
    integer(int32), intent(inout) :: hi(:)
    integer,        intent(in)    :: n, i
    integer :: p, l, r, s
    p = i
    do
       l = 2*p
       if (l > n) exit
       r = l + 1
       s = l
       if (r <= n) then
          if (less_pair(hv(r), hi(r), hv(l), hi(l))) s = r
       end if
       if (.not. less_pair(hv(s), hi(s), hv(p), hi(p))) exit
       call swap(hv(s), hv(p)); call swap_i(hi(s), hi(p))
       p = s
    end do
  end subroutine sift_down_min

  ! Build min-heap in-place from first n elements.
  pure subroutine heapify_min(hv, hi, n)
    real(real64),   intent(inout) :: hv(:)
    integer(int32), intent(inout) :: hi(:)
    integer,        intent(in)    :: n
    integer :: i
    do i = n/2, 1, -1
       call sift_down_min(hv, hi, n, i)
    end do
  end subroutine heapify_min

  ! Extract all items from a min-heap into ascending order.
  pure subroutine heap_extract_all_ascending(hv, hi, n, outv, outi)
    real(real64),   intent(inout) :: hv(:)
    integer(int32), intent(inout) :: hi(:)
    integer,        intent(in)    :: n
    real(real64),   intent(out)   :: outv(n)
    integer(int32), intent(out)   :: outi(n)
    integer :: m
    do m = 1, n
       outv(m) = hv(1)
       outi(m) = hi(1)
       ! Move last to root, shrink heap, sift down
       hv(1) = hv(n-m+1); hi(1) = hi(n-m+1)
       call sift_down_min(hv, hi, n-m, 1)
    end do
  end subroutine heap_extract_all_ascending

  ! Public driver: find top-k largest values and their indices in descending order.
  subroutine topk_largest(values, k, top_vals, top_idx)
    real(real64), intent(in)           :: values(:)
    integer,      intent(in)           :: k
    real(real64), intent(out)          :: top_vals(k)
    integer(int32), intent(out)        :: top_idx(k)

    integer :: n, i
    real(real64),   allocatable :: hv(:)
    integer(int32), allocatable :: hi(:)

    n = size(values)
    if (k < 1 .or. k > n) error stop "topk_largest: 1 <= k <= size(values)"

    allocate(hv(k), hi(k))

    ! 1) Seed heap with first k elements (as (value, index)).
    do i = 1, k
       hv(i) = values(i)
       hi(i) = int(i, int32)
    end do
    call heapify_min(hv, hi, k)

    ! 2) Scan the rest; keep only the top-k in a min-heap.
    do i = k+1, n
       if (greater_pair(values(i), int(i,int32), hv(1), hi(1))) then
          hv(1) = values(i)
          hi(1) = int(i, int32)
          call sift_down_min(hv, hi, k, 1)
       end if
    end do

    ! 3) Extract ascending, then reverse to descending order.
    call heap_extract_all_ascending(hv, hi, k, top_vals, top_idx)
    call reverse_inplace(top_vals)
    call reverse_inplace_i(top_idx)

    deallocate(hv, hi)
  contains
    pure subroutine reverse_inplace(a)
      real(real64), intent(inout) :: a(:)
      integer :: l, r
      real(real64) :: t
      l = 1; r = size(a)
      do while (l < r)
         t = a(l); a(l) = a(r); a(r) = t
         l = l + 1; r = r - 1
      end do
    end subroutine

    pure subroutine reverse_inplace_i(a)
      integer(int32), intent(inout) :: a(:)
      integer :: l, r
      integer(int32) :: t
      l = 1; r = size(a)
      do while (l < r)
         t = a(l); a(l) = a(r); a(r) = t
         l = l + 1; r = r - 1
      end do
    end subroutine
  end subroutine topk_largest


  ! Public driver: find bottom-k smallest values and their indices in ascending order.
  subroutine bottomk_smallest(values, k, bot_vals, bot_idx)
    use iso_fortran_env, only: real64, int32
    implicit none
    real(real64),   intent(in)  :: values(:)
    integer,        intent(in)  :: k
    real(real64),   intent(out) :: bot_vals(k)
    integer(int32), intent(out) :: bot_idx(k)

    integer :: n, i
    real(real64),   allocatable :: hv(:)
    integer(int32), allocatable :: hi(:)

    n = size(values)
    if (k < 1 .or. k > n) error stop "bottomk_smallest: 1 <= k <= size(values)"

    allocate(hv(k), hi(k))

    ! 1) Seed heap with first k elements (as (value, index)).
    do i = 1, k
       hv(i) = values(i)
       hi(i) = int(i, int32)
    end do
    call heapify_max(hv, hi, k)

    ! 2) Scan the rest; keep only the bottom-k in a max-heap.
    do i = k+1, n
       if (less_pair(values(i), int(i,int32), hv(1), hi(1))) then
          hv(1) = values(i)
          hi(1) = int(i, int32)
          call sift_down_max(hv, hi, k, 1)
       end if
    end do

    ! 3) Extract descending, then reverse to get ascending order (smallest -> largest).
    call heap_extract_all_descending(hv, hi, k, bot_vals, bot_idx)
    call reverse_inplace(bot_vals)
    call reverse_inplace_i(bot_idx)

    deallocate(hv, hi)

  contains
    pure subroutine reverse_inplace(a)
      real(real64), intent(inout) :: a(:)
      integer :: l, r
      real(real64) :: t
      l = 1; r = size(a)
      do while (l < r)
         t = a(l); a(l) = a(r); a(r) = t
         l = l + 1; r = r - 1
      end do
    end subroutine reverse_inplace

    pure subroutine reverse_inplace_i(a)
      integer(int32), intent(inout) :: a(:)
      integer :: l, r
      integer(int32) :: t
      l = 1; r = size(a)
      do while (l < r)
         t = a(l); a(l) = a(r); a(r) = t
         l = l + 1; r = r - 1
      end do
    end subroutine reverse_inplace_i
  end subroutine bottomk_smallest

end module topk_heap_mod



module sort_array_mod
  use iso_fortran_env, only: real64
  implicit none
contains
  subroutine sort_indices_by_values(a, idx)
    real(real64), intent(in) :: a(:)
    integer, intent(out) :: idx(size(a))
    integer :: i, j, n, key_idx
    real(real64) :: key_val

    n = size(a)
    idx = [(i, i=1,n)]   ! initialize indices

    ! Insertion sort on indices based on a(idx)
    do i = 2, n
       key_idx = idx(i)
       key_val = a(key_idx)
       j = i - 1
       do while (a(idx(j)) > key_val)
          idx(j+1) = idx(j)
          j = j - 1
          if (j.eq.0) exit
       end do
       idx(j+1) = key_idx
    end do
  end subroutine sort_indices_by_values
end module sort_array_mod
