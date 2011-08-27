!*****************************************************************************

!   DDFUN: A Double-Double Floating Point Computation Package

!   IEEE Fortran-90 version
!   Version Date:  2005-01-26

!   Author:

!      David H. Bailey
!      NERSC, Lawrence Berkeley Lab
!      Mail Stop 50B-2239
!      Berkeley, CA 94720
!      Email: dhbailey@lbl.gov

!   Some system-dependent items are noted with !> comments.

!*****************************************************************************

module ddfunmod

contains

subroutine ddabrt
implicit none

!   This permits one to insert a call to a vendor-specific traceback routine.

stop
end subroutine

subroutine ddadd (dda, ddb, ddc)

!   This subroutine computes ddc = dda + ddb.

implicit none
real*8 dda(2), ddb(2), ddc(2)
real*8 e, t1, t2

!   Compute dda + ddb using Knuth's trick.

t1 = dda(1) + ddb(1)
e = t1 - dda(1)
t2 = ((ddb(1) - e) + (dda(1) - (t1 - e))) + dda(2) + ddb(2)

!   The result is t1 + t2, after normalization.

ddc(1) = t1 + t2
ddc(2) = t2 - (ddc(1) - t1)
return
end subroutine

subroutine ddang (x, y, a)

!   This computes the DD angle A subtended by the DD pair (X, Y) considered as
!   a point in the x-y plane.  This is more useful than an arctan or arcsin
!   routine, since it places the result correctly in the full circle, i.e.
!   -Pi < A <= Pi.

!   The Taylor series for Sin converges much more slowly than that of Arcsin.
!   Thus this routine does not employ Taylor series, but instead computes
!   Arccos or Arcsin by solving Cos (a) = x or Sin (a) = y using one of the
!   following Newton iterations, both of which converge to a:

!           z_{k+1} = z_k - [x - Cos (z_k)] / Sin (z_k)
!           z_{k+1} = z_k + [y - Sin (z_k)] / Cos (z_k)

!   The first is selected if Abs (x) <= Abs (y); otherwise the second is used.

implicit none
integer i, ix, iy, k, kk, nx, ny
real*8 t1, t2, t3
real*8 a(2), pi(2), x(2), y(2), s0(2), s1(2), s2(2), s3(2), s4(2)
save pi
!>
!   Uncomment one of the following two lines, preferably the first if it is
!   accepted by the compiler.

!data pi/ z'400921FB54442D18', z'3CA1A62633145C07'/
 data pi/ 3.1415926535897931D+00,  1.2246467991473532D-16/

!   Check if both X and Y are zero.

if (x(1) .eq. 0.d0 .and. y(1) .eq. 0.d0) then
  write (6, 1)
1 format ('*** DDANG: Both arguments are zero.')
  call ddabrt
  return
endif

!   Check if one of X or Y is zero.

if (x(1) .eq. 0.d0) then
  if (y(1) .gt. 0.d0) then
    call ddmuld (pi, 0.5d0, a)
  else
    call ddmuld (pi, -0.5d0, a)
  endif
  goto 120
elseif (y(1) .eq. 0.d0) then
  if (x(1) .gt. 0.d0) then
      a(1) = 0.d0
      a(2) = 0.d0
  else
    a(1) = pi(1)
    a(2) = pi(2)
  endif
  goto 120
endif

!   Normalize x and y so that x^2 + y^2 = 1.

call ddmul (x, x, s0)
call ddmul (y, y, s1)
call ddadd (s0, s1, s2)
call ddsqrt (s2, s3)
call dddiv (x, s3, s1)
call dddiv (y, s3, s2)

!   Compute initial approximation of the angle.

call ddqdc (s1, t1)
call ddqdc (s2, t2)
t3 = atan2 (t2, t1)
a(1) = t3
a(2) = 0.d0

!   The smaller of x or y will be used from now on to measure convergence.
!   This selects the Newton iteration (of the two listed above) that has the
!   largest denominator.

if (abs (t1) .le. abs (t2)) then
  kk = 1
  s0(1) = s1(1)
  s0(2) = s1(2)
else
  kk = 2
  s0(1) = s2(1)
  s0(2) = s2(2)
endif

!   Perform the Newton-Raphson iteration described.

do k = 1, 3
  call ddcssn (a, s1, s2)
  if (kk .eq. 1) then
    call ddsub (s0, s1, s3)
    call dddiv (s3, s2, s4)
    call ddsub (a, s4, s1)
  else
    call ddsub (s0, s2, s3)
    call dddiv (s3, s1, s4)
    call ddadd (a, s4, s1)
  endif
  a(1) = s1(1)
  a(2) = s1(2)
enddo

 120  continue

return
end subroutine

subroutine ddcadd (a, b, c)

!   This computes the sum of the DDC numbers A and B and returns the DDC
!   result in C.

implicit none
real*8 a(4), b(4), c(4)

call ddadd (a, b, c)
call ddadd (a(3), b(3), c(3))

return
end subroutine

subroutine ddcdiv (a, b, c)

!   This routine divides the DD complex numbers A and B to yield the DDC
!   quotient C.

!   This routine employs the formula described in DDCMUL to save multiprecision
!   multiplications.

implicit none
real*8 a(4), b(4), c(4), f(2), s0(2), s1(2), s2(2), s3(2), s4(2)

if (b(1) .eq. 0.d0 .and. b(3) .eq. 0.d0) then
  write (6, 1)
1 format ('*** DDCDIV: Divisor is zero.')
  call ddabrt
  return
endif

f(1) = 1.d0
f(2) = 0.d0
call ddmul (a, b, s0)
call ddmul (a(3), b(3), s1)
call ddadd (s0, s1, s2)
call ddsub (s0, s1, s3)
call ddadd (a, a(3), s0)
call ddsub (b, b(3), s1)
call ddmul (s0, s1, s4)
call ddsub (s4, s3, s1)
call ddmul (b, b, s0)
call ddmul (b(3), b(3), s3)
call ddadd (s0, s3, s4)
call dddiv (f, s4, s0)
call ddmul (s2, s0, c)
call ddmul (s1, s0, c(3))

return
end subroutine

subroutine ddceq (a, b)

!   This sets the DDC number B equal to the DDC number A.

implicit none
real*8 a(4), b(4)

b(1) = a(1)
b(2) = a(2)
b(3) = a(3)
b(4) = a(4)

return
end subroutine

subroutine ddcmul (a, b, c)

!   This routine multiplies the DD complex numbers A and B to yield the DDC
!   product C.

implicit none
real*8 a(4), b(4), c(4), s0(2), s1(2), s2(2), s3(2)

call ddmul (a, b, s0)
call ddmul (a(3), b(3), s1)
call ddmul (a, b(3), s2)
call ddmul (a(3), b, s3)
call ddsub (s0, s1, c)
call ddadd (s2, s3, c(3))

return
end subroutine

subroutine ddcpr (a, b, ic)

!   This routine compares the DD numbers A and B and returns in IC the value
!   -1, 0, or 1 depending on whether A < B, A = B, or A > B.  It is faster
!   than merely subtracting A and B and looking at the sign of the result.

implicit none
integer ic
real*8 a(2), b(2)

if (a(1) .lt. b(1)) then
  ic = -1
elseif (a(1) .eq. b(1)) then
  if (a(2) .lt. b(2)) then
    ic = -1
  elseif (a(2) .eq. b(2)) then
    ic = 0
  else
    ic = 1
  endif
else
  ic = 1
endif

return
end subroutine

subroutine ddcpwr_old (a, n, b)

!   This computes the N-th power of the DDC number A and returns the DDC
!   result C in B.  When N is zero, 1 is returned.  When N is negative, the
!   reciprocal of A ^ |N| is returned.

!   This routine employs the binary method for exponentiation.

implicit none
integer j, kk, kn, l1, mn, n, na1, na2, nn
real*8 cl2, t1
parameter (cl2 = 1.4426950408889633d0)
real*8 a(4), b(4), s0(4), s1(4), s2(4), s3(4)

if (a(1) .eq. 0.d0 .and. a(3) .eq. 0.d0) then
  if (n .ge. 0) then
    b(1) = 0.d0
    b(2) = 0.d0
    b(3) = 0.d0
    b(4) = 0.d0
    goto 120
  else
    write (6, 1)
1   format ('*** DDCPWR: Argument is zero and N is negative or zero.')
    call ddabrt
    return
  endif
endif

nn = abs (n)
if (nn .eq. 0) then
  s2(1) = 1.d0
  s2(2) = 0.d0
  s2(3) = 0.d0
  s2(4) = 0.d0
  goto 120
elseif (nn .eq. 1) then
  s2(1) = a(1)
  s2(2) = a(2)
  s2(3) = a(3)
  s2(4) = a(4)
  goto 110
elseif (nn .eq. 2) then
  call ddcmul (a, a, s2)
  goto 110
endif

!   Determine the least integer MN such that 2 ^ MN .GT. NN.

t1 = nn
mn = cl2 * log (t1) + 1.d0 + 1.d-14

s0(1) = a(1)
s0(2) = a(2)
s0(3) = a(3)
s0(4) = a(4)
s2(1) = 1.d0
s2(2) = 0.d0
s2(3) = 0.d0
s2(4) = 0.d0
kn = nn

!   Compute B ^ N using the binary rule for exponentiation.

do j = 1, mn
  kk = kn / 2
  if (kn .ne. 2 * kk) then
    call ddcmul (s2, s0, s1)
    s2(1) = s1(1)
    s2(2) = s1(2)
    s2(3) = s1(3)
    s2(4) = s2(4)
  endif
  kn = kk
  if (j .lt. mn) then
    call ddcmul (s0, s0, s1)
    s0(1) = s1(1)
    s0(2) = s1(2)
    s0(3) = s1(3)
    s0(4) = s1(4)
  endif
enddo

!   Compute reciprocal if N is negative.

110  continue

if (n .lt. 0) then
  s1(1) = 1.d0
  s1(2) = 0.d0
  s1(3) = 0.d0
  s1(4) = 0.d0
  call ddcdiv (s1, s2, s0)
  s2(1) = s0(1)
  s2(2) = s0(2)
  s2(3) = s0(3)
  s2(4) = s0(4)
endif

b(1) = s2(1)
b(2) = s2(2)
b(3) = s2(3)
b(4) = s2(4)

120  continue
return
end subroutine

subroutine ddcpwr (a, n, b)

!   This computes the N-th power of the DDC number A and returns the DDC
!   result C in B.  When N is zero, 1 is returned.  When N is negative, the
!   reciprocal of A ^ |N| is returned.

!   This routine employs the binary method for exponentiation.
!   Modified by Roberto Pittau

implicit none
integer j, kk, kn, l1, mn, n, na1, na2, nn
real*8 cl2, t1
parameter (cl2 = 1.4426950408889633d0)
real*8 a(4), b(4), s0(4), s1(4), s2(4), s3(4), aus(4)

nn = abs (n)
if (nn .eq. 0) then
  b(1) = 1.d0
  b(2) = 0.d0
  b(3) = 0.d0
  b(4) = 0.d0
  goto 120
elseif (nn .eq. 1) then
  s2(1) = a(1)
  s2(2) = a(2)
  s2(3) = a(3)
  s2(4) = a(4)
else
  aus(1) = 1.d0
  aus(2) = 0.d0
  aus(3) = 0.d0
  aus(4) = 0.d0
  do j= 1,nn
   call ddcmul (a, aus, aus)
  enddo
  s2(1) = aus(1)
  s2(2) = aus(2)
  s2(3) = aus(3)
  s2(4) = aus(4)
endif

if (n .lt. 0) then
  s1(1) = 1.d0
  s1(2) = 0.d0
  s1(3) = 0.d0
  s1(4) = 0.d0
  call ddcdiv (s1, s2, s0)
  s2(1) = s0(1)
  s2(2) = s0(2)
  s2(3) = s0(3)
  s2(4) = s0(4)
endif

b(1) = s2(1)
b(2) = s2(2)
b(3) = s2(3)
b(4) = s2(4)

120  continue
return
end subroutine

subroutine ddcsqrt (a, b)

!   This routine computes the complex square root of the DDC number C.
!   This routine uses the following formula, where A1 and A2 are the real and
!   imaginary parts of A, and where R = Sqrt [A1 ^ 2 + A2 ^2]:

!      B = Sqrt [(R + A1) / 2] + I Sqrt [(R - A1) / 2]

!   If the imaginary part of A is < 0, then the imaginary part of B is also
!   set to be < 0.

implicit none
real*8 a(4), b(4), s0(2), s1(2), s2(2)

if (a(1) .eq. 0.d0 .and. a(3) .eq. 0.d0) then
  b(1) = 0.d0
  b(2) = 0.d0
  b(3) = 0.d0
  b(4) = 0.d0
  goto 100
endif

call ddmul (a, a, s0)
call ddmul (a(3), a(3), s1)
call ddadd (s0, s1, s2)
call ddsqrt (s2, s0)
s1(1) = a(1)
s1(2) = a(2)
if (s1(1) .lt. 0.d0) then
  s1(1) = - s1(1)
  s1(2) = - s1(2)
endif
call ddadd (s0, s1, s2)
call ddmuld (s2, 0.5d0, s1)
call ddsqrt (s1, s0)
call ddmuld (s0, 2.d0, s1)
if (a(1) .ge. 0.d0) then
  b(1) = s0(1)
  b(2) = s0(2)
  call dddiv (a(3), s1, b(3))
else
  call dddiv (a(3), s1, b)
  if (b(1) .lt. 0.d0) then
    b(1) = - b(1)
    b(2) = - b(2)
  endif
  b(3) = s0(1)
  b(4) = s0(2)
  if (a(3) .lt. 0.d0) then
    b(3) = - b(3)
    b(4) = - b(4)
  endif
endif

 100  continue
return
end subroutine

subroutine ddcssh (a, x, y)

!   This computes the hyperbolic cosine and sine of the DD number A and
!   returns the two DD results in X and Y, respectively. 

implicit none
real*8 a(2), f(2), x(2), y(2), s0(2), s1(2), s2(2)

f(1) = 1.d0
f(2) = 0.d0
call ddexp (a, s0)
call dddiv (f, s0, s1)
call ddadd (s0, s1, s2)
call ddmuld (s2, 0.5d0, x)
call ddsub (s0, s1, s2)
call ddmuld (s2, 0.5d0, y)

return
end subroutine

subroutine ddcssn (a, x, y)

!   This computes the cosine and sine of the DD number A and returns the two DD
!   results in X and Y, respectively.

!   This routine uses the conventional Taylor's series for Sin (s):

!   Sin (s) =  s - s^3 / 3! + s^5 / 5! - s^7 / 7! ...

!   where s = t - a * pi / 2 - b * pi / 16 and the integers a and b are chosen
!   to minimize the absolute value of s.  We can then compute

!   Sin (t) = Sin (s + a * pi / 2 + b * pi / 16)
!   Cos (t) = Cos (s + a * pi / 2 + b * pi / 16)

!   by applying elementary trig identities for sums.  The sine and cosine of
!   b * pi / 16 are of the form 1/2 * Sqrt {2 +- Sqrt [2 +- Sqrt(2)]}.
!   Reducing t in this manner insures that -Pi / 32 < s <= Pi / 32, which
!   accelerates convergence in the above series.

implicit none
integer ia, ka, kb, kc, l1
real*8 t1, t2
real*8 a(2), f(2), pi(2), x(2), y(2), s0(2), s1(2), s2(2), s3(2), s4(2), &
  s5(2), s6(2)
real*8 cs(2,2,4)
save cs, pi
!>
!   Uncomment one of the following two sets of lines, preferably the first set
!   if it is accepted by the compiler.

!data cs/ &
!  z'3FEF6297CFF75CB0',  z'3C7562172A361FD3', &
!  z'3FC8F8B83C69A60B',  z'BC626D19B9FF8D82', &
!  z'3FED906BCF328D46',  z'3C7457E610231AC2', &
!  z'3FD87DE2A6AEA963',  z'BC672CEDD3D5A610', &
!  z'3FEA9B66290EA1A3',  z'3C39F630E8B6DAC8', &
!  z'3FE1C73B39AE68C8',  z'3C8B25DD267F6600', &
!  z'3FE6A09E667F3BCD',  z'BC8BDD3413B26456', &
!  z'3FE6A09E667F3BCD',  z'BC8BDD3413B26456' /
!data pi/ z'400921FB54442D18',  z'3CA1A62633145C07'/

 data cs/ &
    9.8078528040323043D-01,   1.8546939997825006D-17, &
    1.9509032201612828D-01,  -7.9910790684617313D-18, &
    9.2387953251128674D-01,   1.7645047084336677D-17, &
    3.8268343236508978D-01,  -1.0050772696461588D-17, &
    8.3146961230254524D-01,   1.4073856984728024D-18, &
    5.5557023301960218D-01,   4.7094109405616768D-17, &
    7.0710678118654757D-01,  -4.8336466567264567D-17, &
    7.0710678118654757D-01,  -4.8336466567264567D-17/
 data pi / 3.1415926535897931D+00,  1.2246467991473532D-16/

if (a(1) .eq. 0.d0) then
  x(1) = 1.d0
  x(2) = 0.d0
  y(1) = 0.d0
  y(2) = 0.d0
  goto 120
endif

f(1) = 1.d0
f(2) = 0.d0

!   Reduce to between - Pi and Pi.

call ddmuld (pi, 2.d0, s0)
call dddiv (a, s0, s1)
call ddnint (s1, s2)
call ddsub (s1, s2, s3)

!   Determine nearest multiple of Pi / 2, and within a quadrant, the nearest
!   multiple of Pi / 16.  Through most of the rest of this subroutine, KA and
!   KB are the integers a and b of the algorithm above.

t1 = s3(1)
t2 = 4.d0 * t1
ka = nint (t2)
kb = nint (8.d0 * (t2 - ka))
t1 = (8 * ka + kb) / 32.d0
s1(1) = t1
s1(2) = 0.d0
call ddsub (s3, s1, s2)
call ddmul (s0, s2, s1)

!   Compute cosine and sine of the reduced argument s using Taylor's series.

if (s1(1) .eq. 0.d0) then
  s0(1) = 0.d0
  s0(2) = 0.d0
  goto 110
endif
s0(1) = s1(1)
s0(2) = s1(2)
call ddmul (s0, s0, s2)
l1 = 0

100  l1 = l1 + 1
if (l1 .eq. 100) then
  write (6, 1)
1 format ('*** DDCSSN: Iteration limit exceeded.')
  call ddabrt
  return
endif

t2 = - (2.d0 * l1) * (2.d0 * l1 + 1.d0)
call ddmul (s2, s1, s3)
call dddivd (s3, t2, s1)
call ddadd (s1, s0, s3)
s0(1) = s3(1)
s0(2) = s3(2)

!   Check for convergence of the series.

if (abs (s1(1)) .gt. 1d-33 * abs (s3(1))) goto 100

!   Compute Cos (s) = Sqrt [1 - Sin^2 (s)].

110  continue
s1(1) = s0(1)
s1(2) = s0(2)
call ddmul (s0, s0, s2)
call ddsub (f, s2, s3)
call ddsqrt (s3, s0)

!   Compute cosine and sine of b * Pi / 16.

kc = abs (kb)
f(1) = 2.
if (kc .eq. 0) then
  s2(1) = 1.d0
  s2(2) = 0.d0
  s3(1) = 0.d0
  s3(2) = 0.d0
else
  s2(1) = cs(1,1,kc)
  s2(2) = cs(2,1,kc)
  s3(1) = cs(1,2,kc)
  s3(2) = cs(2,2,kc)
endif
if (kb .lt. 0) then
  s3(1) = - s3(1)
  s3(2) = - s3(2)
endif

!   Apply the trigonometric summation identities to compute cosine and sine of
!   s + b * Pi / 16.

call ddmul (s0, s2, s4)
call ddmul (s1, s3, s5)
call ddsub (s4, s5, s6)
call ddmul (s1, s2, s4)
call ddmul (s0, s3, s5)
call ddadd (s4, s5, s1)
s0(1) = s6(1)
s0(2) = s6(2)

!   This code in effect applies the trigonometric summation identities for
!   (s + b * Pi / 16) + a * Pi / 2.

if (ka .eq. 0) then
  x(1) = s0(1)
  x(2) = s0(2)
  y(1) = s1(1)
  y(2) = s1(2)
elseif (ka .eq. 1) then
  x(1) = - s1(1)
  x(2) = - s1(2)
  y(1) = s0(1)
  y(2) = s0(2)
elseif (ka .eq. -1) then
  x(1) = s1(1)
  x(2) = s1(2)
  y(1) = - s0(1)
  y(2) = - s0(2)
elseif (ka .eq. 2 .or. ka .eq. -2) then
  x(1) = - s0(1)
  x(2) = - s0(2)
  y(1) = - s1(1)
  y(2) = - s1(2)
endif

120  continue
return
end subroutine

subroutine ddcsub (a, b, c)

!   This subracts the DDC numbers A and B and returns the DDC difference in
!   C.

implicit none
real*8 a(4), b(4), c(4)

call ddsub (a, b, c)
call ddsub (a(3), b(3), c(3))

return
end subroutine

subroutine dddiv (dda, ddb, ddc)

!   This divides the DD number DDA by the DD number DDB to yield the DD
!   quotient DDC.

implicit none
real*8 dda(2), ddb(2), ddc(2)
real*8 a1, a2, b1, b2, cona, conb, c11, c2, c21, e, split, s1, s2, &
  t1, t2, t11, t12, t21, t22
parameter (split = 134217729.d0)

!   Compute a DP approximation to the quotient.

s1 = dda(1) / ddb(1)
!>
!   On systems with a fused multiply add, such as IBM systems, it is faster to
!   uncomment the next two lines and comment out the following lines until !>.
!   On other systems, do the opposite.

! c11 = s1 * ddb(1)
! c21 = s1 * ddb(1) - c11

!   This splits s1 and ddb(1) into high-order and low-order words.

cona = s1 * split
conb = ddb(1) * split
a1 = cona - (cona - s1)
b1 = conb - (conb - ddb(1))
a2 = s1 - a1
b2 = ddb(1) - b1

!   Multiply s1 * ddb(1) using Dekker's method.

c11 = s1 * ddb(1)
c21 = (((a1 * b1 - c11) + a1 * b2) + a2 * b1) + a2 * b2
!>
!   Compute s1 * ddb(2) (only high-order word is needed).

c2 = s1 * ddb(2)

!   Compute (c11, c21) + c2 using Knuth's trick.

t1 = c11 + c2
e = t1 - c11
t2 = ((c2 - e) + (c11 - (t1 - e))) + c21

!   The result is t1 + t2, after normalization.

t12 = t1 + t2
t22 = t2 - (t12 - t1)

!   Compute dda - (t12, t22) using Knuth's trick.

t11 = dda(1) - t12
e = t11 - dda(1)
t21 = ((-t12 - e) + (dda(1) - (t11 - e))) + dda(2) - t22

!   Compute high-order word of (t11, t21) and divide by ddb(1).

s2 = (t11 + t21) / ddb(1)

!   The result is s1 + s2, after normalization.

ddc(1) = s1 + s2
ddc(2) = s2 - (ddc(1) - s1)

return
end subroutine

subroutine dddivd (dda, db, ddc)

!   This routine divides the DD number A by the DP number B to yield
!   the DD quotient C.  

implicit none
real*8 dda(2), db, ddc(2)
real*8 a1, a2, b1, b2, cona, conb, e, split, t1, t2, t11, t12, t21, t22
parameter (split = 134217729.d0)

!   Compute a DP approximation to the quotient.

t1 = dda(1) / db
!>
!   On systems with a fused multiply add, such as IBM systems, it is faster to
!   uncomment the next two lines and comment out the following lines until !>.
!   On other systems, do the opposite.

! t12 = t1 * db
! t22 = t1 * db - t12

!   This splits t1 and db into high-order and low-order words.

cona = t1 * split
conb = db * split
a1 = cona - (cona - t1)
b1 = conb - (conb - db)
a2 = t1 - a1
b2 = db - b1

!   Multiply t1 * db using Dekker's method.

t12 = t1 * db
t22 = (((a1 * b1 - t12) + a1 * b2) + a2 * b1) + a2 * b2
!>
!   Compute dda - (t12, t22) using Knuth's trick.

t11 = dda(1) - t12
e = t11 - dda(1)
t21 = ((-t12 - e) + (dda(1) - (t11 - e))) + dda(2) - t22

!   Compute high-order word of (t11, t21) and divide by db.

t2 = (t11 + t21) / db

!   The result is t1 + t2, after normalization.

ddc(1) = t1 + t2
ddc(2) = t2 - (ddc(1) - t1)
return
end subroutine

subroutine dddqc (a, b)

!   This routine converts the DP number A to DD form in B.  All bits of
!   A are recovered in B.  However, note for example that if A = 0.1D0 and N
!   is 0, then B will NOT be the DD equivalent of 1/10.

implicit none
real*8 a, b(2)

b(1) = a
b(2) = 0.d0
return
end subroutine

subroutine ddeq (a, b)

!   This routine sets the DD number B equal to the DD number A. 

implicit none
real*8 a(2), b(2)

b(1) = a(1)
b(2) = a(2)
end subroutine

subroutine ddeform (a, n1, n2, b)

!   This routine converts the DD number A to E format, i.e. 1P,En1.n2.
!   B is the output array (type character*1 of size n1).  N1 must exceed
!   N2 by at least 8, and N1 must not exceed 80.  N2 must not exceed 30,
!   i.e., not more than 31 significant digits.

implicit none
integer i, ln, m1, n1, n2
parameter (ln = 40)
character*1 b(n1)
character*40 cs
real*8 a(2)

if (n1 .lt. 0 .or. n2 .lt. 0 .or. n1 .gt. 80 .or. n2 .gt. 30 &
  .or. n2 .gt. n1 - 8) then
  write (6, 1) n1, n2
1 format ('*** DDEFORM: Improper n1, n2 =',2i6)
  call ddabrt
endif

call ddoutc (a, cs)
m1 = n1 - n2 - 8

do i = 1, m1
  b(i) = ' '
enddo

do i = 1, n2 + 3
  b(i+m1) = cs(i+2:i+2)
enddo

do i = 1, 5
  b(i+m1+n2+3) = cs(i+35:i+35)
enddo

return
end subroutine

subroutine ddexp (a, b)

!   This computes the exponential function of the DD number A and returns the
!   DD result in B.

!   This routine uses a modification of the Taylor's series for Exp (t):

!   Exp (t) =  (1 + r + r^2 / 2! + r^3 / 3! + r^4 / 4! ...) ^ q * 2 ^ n

!   where q = 64, r = t' / q, t' = t - n Log(2) and where n is chosen so
!   that -0.5 Log(2) < t' <= 0.5 Log(2).  Reducing t mod Log(2) and
!   dividing by 64 insures that -0.004 < r <= 0.004, which accelerates
!   convergence in the above series.

implicit none
integer i, ia, l1, na, nq, nz, n1
real*8 t1, t2
parameter (nq = 6)
real*8 a(2), b(2), al2(2), f(2), s0(2), s1(2), s2(2), s3(2), tl
save al2
!>
!   Uncomment one of the following two lines, preferably the first if it is
!   accepted by the compiler.

!data al2 / z'3FE62E42FEFA39EF',  z'3C7ABC9E3B39803F'/
 data al2/ 6.9314718055994529D-01,  2.3190468138462996D-17/

!   Check for overflows and underflows.

if (abs (a(1)) .ge. 709.d0) then
  if (a(1) .gt. 0.d0) then
    write (6, 1) a(1)
1   format ('*** DDEXP: Argument is too large',f12.6)
    call ddabrt
    return
  else
    call dddqc (0.d0, b)
    goto 130
  endif
endif

f(1) = 1.d0
f(2) = 0.d0

!   Compute the reduced argument A' = A - Log(2) * Nint [A / Log(2)].  Save
!   NZ = Nint [A / Log(2)] for correcting the exponent of the final result.

call dddiv (a, al2, s0)
call ddnint (s0, s1)
t1 = s1(1)
nz = t1 + sign (1.d-14, t1)
call ddmul (al2, s1, s2)
call ddsub (a, s2, s0)

!   Check if the reduced argument is zero.

if (s0(1) .eq. 0.d0) then
  s0(1) = 1.d0
  s0(2) = 0.d0
  l1 = 0
  goto 120
endif

!   Divide the reduced argument by 2 ^ NQ.

call dddivd (s0, 2.d0 ** nq, s1)

!   Compute Exp using the usual Taylor series.

s2(1) = 1.d0
s2(2) = 0.d0
s3(1) = 1.d0
s3(2) = 0.d0
l1 = 0

100  l1 = l1 + 1
if (l1 .eq. 100) then
  write (6, 2)
2 format ('*** DDEXP: Iteration limit exceeded.')
  call ddabrt
  return
endif

t2 = l1
call ddmul (s2, s1, s0)
call dddivd (s0, t2, s2)
call ddadd (s3, s2, s0)
call ddeq (s0, s3)

!   Check for convergence of the series.

if (abs (s2(1)) .gt. 1d-33 * abs (s3(1))) goto 100

!   Raise to the (2 ^ NQ)-th power.

do i = 1, nq
  call ddmul (s0, s0, s1)
  s0(1) = s1(1)
  s0(2) = s1(2)
enddo

!  Multiply by 2 ^ NZ.

120  call ddmuld (s0, 2.d0 ** nz, b)

!   Restore original precision level.

 130  continue
return
end subroutine

subroutine ddfform (a, n1, n2, b)

!   This routine converts the DD number A to F format, i.e. Fn1.n2.
!   B is the output array (type character*1 of size n1).  N1 must exceed
!   N2 by at least 3, and N1 must not exceed 80.  N2 must not exceed 30.

implicit none
integer i, ix, kx, ln, ls, lz, mx, nx, n1, n2
parameter (ln = 40)
real*8 a(2)
character*1 b(n1)
character*40 c
character*40 chr40

if (n1 .lt. 0 .or. n2 .lt. 0 .or. n1 .gt. 80 .or. n2 .gt. 30 &
  .or. n1 - n2 .lt. 3) then
  write (6, 1) n1, n2
1 format ('*** DDFFORM: Improper n1, n2 =',2i6)
  call ddabrt
endif

!   Call ddoutc and extract exponent.

call ddoutc (a, c)
ix = dddigin (c(ln-3:ln), 4)

do i = 1, n1
  b(i) = ' '
enddo

if (a(1) .ge. 0.d0) then
  ls = 0
else
  ls = 1
endif
if (ix .ge. 0 .and. a(1) .ne. 0.d0) then
  lz = 0
else
  lz = 1
endif
mx = max (ix, 0)

!   Check for overflow of field length.

if (ls + lz + mx + n2 + 2 .gt. n1) then
  do i = 1, n1
    b(i) = '*'
  enddo

  goto 200
endif

!   Check if a zero should be output.

if (a(1) .eq. 0. .or. -ix .gt. n2) then
  do i = 1, n1 - n2 - 2
    b(i) = ' '
  enddo

  b(n1-n2-1) = '0'
  b(n1-n2) = '.'

  do i = 1, n2
    b(i+n1-n2) = '0'
  enddo

  goto 200
endif

!   Process other cases.

if (a(1) .lt. 0.) b(n1-n2-mx-2) = '-'
if (ix .ge. 0) then
  b(n1-n2-ix-1) = c(4:4)
  kx = min (ln - 9, ix)

  do i = 1, kx
    b(i+n1-n2-ix-1) = c(i+5:i+5)
  enddo

  do i = kx + 1, ix
    b(i+n1-n2-ix-1) = '0'
  enddo

  b(n1-n2) = '.'
  kx = max (min (ln - 9 - ix, n2), 0)

  do i = 1, kx
    b(i+n1-n2) = c(i+ix+5:i+ix+5)
  enddo

  do i = kx + 1, n2
    b(i+n1-n2) = '0'
  enddo
else
  nx = - ix
  b(n1-n2-1) = '0'
  b(n1-n2) = '.'

  do i = 1, nx - 1
    b(i+n1-n2) = '0'
  enddo

  b(n1-n2+nx) = c(4:4)
  kx = min (ln - 8, n2 - nx)

  do i = 1, kx
    b(i+n1-n2+nx) = c(i+5:i+5)
  enddo

  do i = kx + 1, n2 - nx
    b(i+n1-n2+nx) = '0'
  enddo
endif

200   continue

return
end subroutine
      
subroutine ddinfr (a, b, c)

!   Sets B to the integer part of the DD number A and sets C equal to the
!   fractional part of A.  Note that if A = -3.3, then B = -3 and C = -0.3.

implicit none
integer ic
real*8 a(2), b(2), c(2), con(2), f(2), s0(2), s1(2), t105, t52
parameter (t105 = 2.d0 ** 105, t52 = 2.d0 ** 52)
save con
data con / t105, t52/

!   Check if  A  is zero.

if (a(1) .eq. 0.d0)  then
  b(1) = 0.d0
  b(2) = 0.d0
  c(1) = 0.d0
  c(2) = 0.d0
  goto 120
endif

if (a(1) .ge. t105) then
  write (6, 1)
1 format ('*** DDINFR: Argument is too large.')
  call ddabrt
  return
endif

f(1) = 1.d0
f(2) = 0.d0
if (a(1) .gt. 0.d0) then
  call ddadd (a, con, s0)
  call ddsub (s0, con, b)
  call ddcpr (a, b, ic)
  if (ic .ge. 0) then
    call ddsub (a, b, c)
  else
    call ddsub (b, f, s1)
    b(1) = s1(1)
    b(2) = s1(2)
    call ddsub (a, b, c)
  endif
else
  call ddsub (a, con, s0)
  call ddadd (s0, con, b)
  call ddcpr (a, b, ic)
  if (ic .le. 0) then
    call ddsub (a, b, c)
  else
    call ddadd (b, f, s1)
    b(1) = s1(1)
    b(2) = s1(2)
    call ddsub (a, b, c)
  endif
endif

120  continue
return
end subroutine

subroutine ddinp (iu, a)

!   This routine reads the DD number A from logical unit IU.  The input
!   value must be placed on a single line of not more than 80 characters.

implicit none
integer iu, ln
parameter (ln = 80)
character*80 cs
real*8 a(2)

read (iu, '(a)', end = 100) cs
call ddinpc (cs, a)
goto 110

100 write (6, 1)
1  format ('*** DDINP: End-of-file encountered.')
call ddabrt

110 return
end subroutine

subroutine ddinpc (a, b)

!   Converts the CHARACTER*80 array A into the DD number B.

implicit none
integer i, ib, id, ie, inz, ip, is, ix, k, ln, lnn
parameter (ln = 80)
real*8 bi
character*80 a
character*1 ai
character*10 ca, dig
parameter (dig = '0123456789')
real*8 b(2), f(2), s0(2), s1(2), s2(2)

id = 0
ip = -1
is = 0
inz = 0
s1(1) = 0.d0
s1(2) = 0.d0

do i = 80, 1, -1
  if (a(i:i) /= ' ') goto 90
enddo

90 continue

lnn = i

!   Scan for digits, looking for the period also.

do i = 1, lnn
  ai = a(i:i)
  if (ai .eq. ' ' .and. id == 0) then
  elseif (ai .eq. '.') then
    if (ip >= 0) goto 210
    ip = id
    inz = 1
  elseif (ai .eq. '+') then
    if (id .ne. 0 .or. ip >= 0 .or. is .ne. 0) goto 210
    is = 1
  elseif (ai .eq. '-') then
    if (id .ne. 0 .or. ip >= 0 .or. is .ne. 0) goto 210
    is = -1
  elseif (ai .eq. 'e' .or. ai .eq. 'E' .or. ai .eq. 'd' .or. ai .eq. 'D') then
    goto 100
  elseif (index (dig, ai) .eq. 0) then
    goto 210
  else
!    read (ai, '(f1.0)') bi
    bi = index (dig, ai) - 1
    if (inz > 0 .or. bi > 0.d0) then
      inz = 1
      id = id + 1
      call ddmuld (s1, 10.d0, s0)
      f(1) = bi
      f(2) = 0.d0
      call dddqc (bi, f)
      call ddadd (s0, f, s1)
    endif
  endif
enddo

100   continue
if (is .eq. -1) then
  s1(1) = - s1(1)
  s1(2) = - s1(2)
endif
k = i
if (ip == -1) ip = id
ie = 0
is = 0
ca = ' '

do i = k + 1, lnn
  ai = a(i:i)
  if (ai .eq. ' ') then
  elseif (ai .eq. '+') then
    if (ie .ne. 0 .or. is .ne. 0) goto 210
    is = 1
  elseif (ai .eq. '-') then
    if (ie .ne. 0 .or. is .ne. 0) goto 210
    is = -1
  elseif (index (dig, ai) .eq. 0) then
    goto 210
  else
    ie = ie + 1
    if (ie .gt. 3) goto 210
    ca(ie:ie) = ai
  endif
enddo

! read (ca, '(i4)') ie
ie = dddigin (ca, 4)
if (is .eq. -1) ie = - ie
ie = ie + ip - id
s0(1) = 10.d0
s0(2) = 0.d0
call ddnpwr (s0, ie, s2)
call ddmul (s1, s2, b)
goto 220

210  write (6, 1)
1 format ('*** DDINPC: Syntax error in literal string.')
call ddabrt

220  continue

return
end subroutine

subroutine ddlog (a, b)

!   This computes the natural logarithm of the DD number A and returns the DD
!   result in B.

!   The Taylor series for Log converges much more slowly than that of Exp.
!   Thus this routine does not employ Taylor series, but instead computes
!   logarithms by solving Exp (b) = a using the following Newton iteration,
!   which converges to b:

!           x_{k+1} = x_k + [a - Exp (x_k)] / Exp (x_k)

!   These iterations are performed with a maximum precision level NW that
!   is dynamically changed, approximately doubling with each iteration.

implicit none
integer k
real*8 t1, t2
real*8 a(2), al2(2), b(2), s0(2), s1(2), s2(2)
save al2
!>
!   Uncomment one of the following two lines, preferably the first if it is
!   accepted by the compiler.

!data al2 / z'3FE62E42FEFA39EF',  z'3C7ABC9E3B39803F'/
 data al2/ 6.9314718055994529D-01,  2.3190468138462996D-17/

if (a(1) .le. 0.d0) then
  write (6, 1)
1 format ('*** DDLOG: Argument is less than or equal to zero.')
  call ddabrt
  return
endif

!   Compute initial approximation of Log (A).

t1 = a(1)
t2 = log (t1)
b(1) = t2
b(2) = 0.d0

!   Perform the Newton-Raphson iteration described above.

do k = 1, 3
  call ddexp (b, s0)
  call ddsub (a, s0, s1)
  call dddiv (s1, s0, s2)
  call ddadd (b, s2, s1)
  b(1) = s1(1)
  b(2) = s1(2)
enddo

120  continue

return
end subroutine

subroutine ddqdc (a, b)

!   This converts the DD number A to DP.

implicit none
real*8 a(2), b

b = a(1)
return
end subroutine

subroutine ddqqc (a, b, c)

!   This converts DD numbers A and B to DDC form in C, i.e. C = A + B i.

implicit none
real*8 a(2), b(2), c(4)

c(1) = a(1)
c(2) = a(2)
c(3) = b(1)
c(4) = b(2)
return
end subroutine

subroutine ddmul (dda, ddb, ddc)

!   This routine multiplies DD numbers DDA and DDB to yield the DD product DDC.

implicit none
real*8 dda(2), ddb(2), ddc(2)
real*8 a1, a2, b1, b2, cona, conb, c11, c21, c2, e, split, t1, t2
parameter (split = 134217729.d0)

!>
!   On systems with a fused multiply add, such as IBM systems, it is faster to
!   uncomment the next two lines and comment out the following lines until !>.
!   On other systems, do the opposite.

! c11 = dda(1) * ddb(1)
! c21 = dda(1) * ddb(1) - c11

!   This splits dda(1) and ddb(1) into high-order and low-order words.

cona = dda(1) * split
conb = ddb(1) * split
a1 = cona - (cona - dda(1))
b1 = conb - (conb - ddb(1))
a2 = dda(1) - a1
b2 = ddb(1) - b1

!   Multilply dda(1) * ddb(1) using Dekker's method.

c11 = dda(1) * ddb(1)
c21 = (((a1 * b1 - c11) + a1 * b2) + a2 * b1) + a2 * b2
!>
!   Compute dda(1) * ddb(2) + dda(2) * ddb(1) (only high-order word is needed).

c2 = dda(1) * ddb(2) + dda(2) * ddb(1)

!   Compute (c11, c21) + c2 using Knuth's trick, also adding low-order product.

t1 = c11 + c2
e = t1 - c11
t2 = ((c2 - e) + (c11 - (t1 - e))) + c21 + dda(2) * ddb(2)

!   The result is t1 + t2, after normalization.

ddc(1) = t1 + t2
ddc(2) = t2 - (ddc(1) - t1)

return
end subroutine

subroutine ddmuld (dda, db, ddc)

!   This routine multiplies the DD number DDA by the DP number DB to yield
!   the DD product DDC.

implicit none
real*8 dda(2), db, ddc(2)
real*8 a1, a2, b1, b2, cona, conb, c11, c21, c2, e, split, t1, t2
parameter (split = 134217729.d0)

!>
!   On systems with a fused multiply add, such as IBM systems, it is faster to
!   uncomment the next two lines and comment out the following lines until !>.
!   On other systems, do the opposite.

! c11 = dda(1) * db
! c21 = dda(1) * db - c11

!   This splits dda(1) and db into high-order and low-order words.

cona = dda(1) * split
conb = db * split
a1 = cona - (cona - dda(1))
b1 = conb - (conb - db)
a2 = dda(1) - a1
b2 = db - b1

!   Multilply dda(1) * db using Dekker's method.

c11 = dda(1) * db
c21 = (((a1 * b1 - c11) + a1 * b2) + a2 * b1) + a2 * b2
!>
!   Compute dda(2) * db (only high-order word is needed).

c2 = dda(2) * db

!   Compute (c11, c21) + c2 using Knuth's trick.

t1 = c11 + c2
e = t1 - c11
t2 = ((c2 - e) + (c11 - (t1 - e))) + c21

!   The result is t1 + t2, after normalization.

ddc(1) = t1 + t2
ddc(2) = t2 - (ddc(1) - t1)
return
end subroutine

subroutine ddmuldd (da, db, ddc)

!   This subroutine computes ddc = da x db.

implicit none
real*8 a1, a2, b1, b2, cona, conb, da, db, ddc(2), split, s1, s2
parameter (split = 134217729.d0)

!>
!   On systems with a fused multiply add, such as IBM systems, it is faster to
!   uncomment the next two lines and comment out the following lines until !>.
!   On other systems, do the opposite.

! s1 = da * db
! s2 = da * db - s1

!   This splits da and db into high-order and low-order words.

cona = da * split
conb = db * split
a1 = cona - (cona - da)
b1 = conb - (conb - db)
a2 = da - a1
b2 = db - b1

!   Multiply da * db using Dekker's method.

s1 = da * db
s2 = (((a1 * b1 - s1) + a1 * b2) + a2 * b1) + a2 * b2
!>
ddc(1) = s1
ddc(2) = s2

return
end subroutine

subroutine ddnint (a, b)

!   This sets B equal to the integer nearest to the DD number A.

implicit none
real*8 a(2), b(2), con(2), s0(2), t105, t52
parameter (t105 = 2.d0 ** 105, t52 = 2.d0 ** 52)
save con
data con / t105, t52/

!   Check if  A  is zero.

if (a(1) .eq. 0.d0)  then
  b(1) = 0.d0
  b(2) = 0.d0
  goto 120
endif

if (a(1) .ge. t105) then
  write (6, 1)
1 format ('*** DDINFR: Argument is too large.')
  call ddabrt
  return
endif

if (a(1) .gt. 0.d0) then
  call ddadd (a, con, s0)
  call ddsub (s0, con, b)
else
  call ddsub (a, con, s0)
  call ddadd (s0, con, b)
endif

120  continue
return
end subroutine

subroutine ddnpwr (a, n, b)

!   This computes the N-th power of the DD number A and returns the DD result
!   in B.  When N is zero, 1 is returned.  When N is negative, the reciprocal
!   of A ^ |N| is returned. 

!   This routine employs the binary method for exponentiation.

implicit none
integer j, kk, kn, l1, mn, n, na1, na2, nn
real*8 cl2, t1
parameter (cl2 = 1.4426950408889633d0)
real*8 a(2), b(2), s0(2), s1(2), s2(2), s3(2)

if (a(1) .eq. 0.d0) then
  if (n .ge. 0) then
    s2(1) = 0.d0
    s2(2) = 0.d0
    goto 120
  else
    write (6, 1)
1   format ('*** DDCPWR: Argument is zero and N is negative or zero.')
    call ddabrt
    return
  endif
endif

nn = abs (n)
if (nn .eq. 0) then
  s2(1) = 1.d0
  s2(2) = 0.d0
  goto 120
elseif (nn .eq. 1) then
  s2(1) = a(1)
  s2(2) = a(2)
  goto 110
elseif (nn .eq. 2) then
  call ddmul (a, a, s2)
  goto 110
endif

!   Determine the least integer MN such that 2 ^ MN .GT. NN.

t1 = nn
mn = cl2 * log (t1) + 1.d0 + 1.d-14
s0(1) = a(1)
s0(2) = a(2)
s2(1) = 1.d0
s2(2) = 0.d0
kn = nn

!   Compute B ^ N using the binary rule for exponentiation.

do j = 1, mn
  kk = kn / 2
  if (kn .ne. 2 * kk) then
    call ddmul (s2, s0, s1)
    s2(1) = s1(1)
    s2(2) = s1(2)
  endif
  kn = kk
  if (j .lt. mn) then
    call ddmul (s0, s0, s1)
    s0(1) = s1(1)
    s0(2) = s1(2)
  endif
enddo

!   Compute reciprocal if N is negative.

110  continue

if (n .lt. 0) then
  s1(1) = 1.d0
  s1(2) = 0.d0
  call dddiv (s1, s2, s0)
  s2(1) = s0(1)
  s2(2) = s0(2)
endif

120  continue

b(1) = s2(1)
b(2) = s2(2)
  
return
end subroutine

subroutine ddnrt (a, n, b)

!   This computes the N-th root of the DD number A and returns the DD result
!   in B.  N must be at least one.

!   This subroutine employs the following Newton-Raphson iteration, which
!   converges to A ^ (-1/N):

!    X_{k+1} = X_k + (X_k / N) * (1 - A * X_k^N)

!   The reciprocal of the final approximation to A ^ (-1/N) is the N-th root.

implicit none
integer i, k, n
real*8 t1, t2, tn
real*8 a(2), b(2), f1(2), s0(2), s1(2)

if (a(1) .eq. 0.d0) then
  b(1) = 0.d0
  b(2) = 0.d0
  goto 140
elseif (a(1) .lt. 0.d0) then
  write (6, 1)
1 format ('*** DDNRT: Argument is negative.')
  call ddabrt
  return
endif
if (n .le. 0) then
  write (6, 2) n
2 format ('*** DDNRT: Improper value of N',i10)
  call ddabrt
  return
endif

!   Handle cases N = 1 and 2.

if (n .eq. 1) then
  b(1) = a(1)
  b(2) = a(1)
  goto 140
elseif (n .eq. 2) then
  call ddsqrt (a, b)
  goto 140
endif

f1(1) = 1.d0
f1(2) = 0.d0

!   Compute the initial approximation of A ^ (-1/N).

tn = n
t1 = a(1)
t2 = exp (- log (t1) / tn)
b(1) = t2
b(2) = 0.d0

!   Perform the Newton-Raphson iteration described above.

do k = 1, 3
  call ddnpwr (b, n, s0)
  call ddmul (a, s0, s1)
  call ddsub (f1, s1, s0)
  call ddmul (b, s0, s1)
  call dddivd (s1, tn, s0)
  call ddadd (b, s0, s1)
  b(1) = s1(1)
  b(2) = s1(2)
enddo

!   Take the reciprocal to give final result.

call dddiv (f1, b, s1)
b(1) = s1(1)
b(2) = s1(2)

140  continue
return
end subroutine

subroutine ddout (iu, a)

!   This routine writes the DD number A on logical unit iu using a standard
!   E format, with lines 40 characters long.

implicit none
integer i, iu, ln
parameter (ln = 40)
character*40 cs
real*8 a(2)

call ddoutc (a, cs)
write (iu, '(a)') cs

return
end subroutine

subroutine ddoutc (a, b)

!   Converts the DD number A into character form in the CHARACTER*40 array B.
!   The format is analogous to the Fortran E format.

!   This routine is called by DDOUT, but it may be directly called by the user
!   if desired for custom output.

implicit none
integer i, ii, ix, ln, nx
parameter (ln = 40)
integer ib(ln)
real*8 t1
character*40 b
character*10 ca, digits
parameter (digits = '0123456789')
real*8 a(2), f(2), s0(2), s1(2)

f(1) = 10.d0
f(2) = 0.d0

do i = 1, ln
  ib(i) = 0
enddo

!   Determine exact power of ten for exponent.

if (a(1) .ne. 0.d0) then
  t1 = log10 (abs (a(1)))
  if (t1 .ge. 0.d0) then
    nx = t1
  else
    nx = t1 - 1.d0
  endif
  call ddnpwr (f, nx, s0)
  call dddiv (a, s0, s1)
  if (s1(1) .lt. 0.d0) then
    s1(1) = - s1(1)
    s1(2) = - s1(2)
  endif

!   If we didn't quite get it exactly right, multiply or divide by 10 to fix.

  i = 0

100 continue

  i = i + 1
  if (s1(1) .lt. 1.d0) then
    nx = nx - 1
    call ddmuld (s1, 10.d0, s0)
    s1(1) = s0(1)
    s1(2) = s0(2)
    if (i <= 3) goto 100
  elseif (s1(1) .ge. 10.d0) then
    nx = nx + 1
    call dddivd (s1, 10.d0, s0)
    s1(1) = s0(1)
    s1(2) = s0(2)
    goto 100
  endif
else
  nx = 0
  s1(1) = 0.d0
  s1(2) = 0.d0
endif

!   Compute digits.

do i = 1, ln - 8
  ii = s1(1)
  ib(i) = ii
  f(1) = ii
  call ddsub (s1, f, s0)
  call ddmuld (s0, 10.d0, s1)
enddo

!   Fix negative digits.

do i = ln - 8, 2, -1
  if (ib(i) .lt. 0) then
    ib(i) = ib(i) + 10
    ib(i-1) = ib(i-1) - 1
  endif
enddo

if (ib(1) .lt. 0) then
  write (6, 1) 
1 format ('ddoutc: negative leading digit')
  call ddabrt
endif

!   Round.

if (ib(ln-8) .ge. 5) then
  ib(ln-9) = ib(ln-9) + 1

  do i = ln - 9, 2, -1
    if (ib(i) .eq. 10) then
      ib(i) = 0
      ib(i-1) = ib(i-1) + 1
    endif
  enddo

  if (ib(1) .eq. 10) then
    ib(1) = 1
    nx = nx + 1
  endif
endif

!   Insert digit characters in ib.

b(1:1) = ' '
b(2:2) = ' '
if (a(1) .ge. 0.d0) then
  b(3:3) = ' '
else
  b(3:3) = '-'
endif
ii = ib(1)
b(4:4) = digits(ii+1:ii+1)
b(5:5) = '.'
b(ln:ln) = ' '

do i = 2, ln - 9
  ii = ib(i)  
  b(i+4:i+4) = digits(ii+1:ii+1)
enddo

!   Insert exponent.

190  continue
! write (ca, '(i4)') nx
ca = dddigout (dble (nx), 4)
b(ln-4:ln-4) = 'E'
ii = 0

do i = 1, 4
  if (ca(i:i) /= ' ') then
    ii = ii + 1
    b(ln-4+ii:ln-4+ii) = ca(i:i)
  endif
enddo

do i = ii + 1, 4
  b(ln-4+i:ln-4+i) = ' '
enddo

return
end subroutine

subroutine ddpic (pi)

!   This returns Pi to quad precision.

implicit none
real*8 pi(2), pic(2)
save pic
!>
!   Uncomment one of the following two lines, preferably the first if it is
!   accepted by the compiler.

!data pic/ z'400921FB54442D18', z'3CA1A62633145C07'/
 data pic/ 3.1415926535897931D+00,  1.2246467991473532D-16/

pi(1) = pic(1)
pi(2) = pic(2)

return
end subroutine

subroutine ddpoly (n, a, x0, x)

!   This finds the root x near x0 (input) for the nth-degree polynomial whose
!   coefficients are given in the n+1-long vector a.  It may be necessary to
!   adjust eps -- default value is 1d-30.

implicit none
integer i, it, n
real*8  a(2,0:n), ad(2,0:n), t1(2), t2(2), t3(2), t4(2), t5(2), &
  x(2), x0(2), dt1, eps
parameter (eps = 1.d-30)

do i = 0, n - 1
  dt1 = i + 1
  call ddmuld (a(1,i+1), dt1, ad(1,i))
enddo

ad(1,n) = 0.d0
ad(2,n) = 0.d0
x(1) = x0(1)
x(2) = x0(2)

do it = 1, 20
  t1(1) = 0.d0
  t1(2) = 0.d0
  t2(1) = 0.d0
  t2(2) = 0.d0
  t3(1) = 1.d0
  t3(2) = 0.d0

  do i = 0, n
    call ddmul (a(1,i), t3, t4)
    call ddadd (t1, t4, t5)
    t1(1) = t5(1)
    t1(2) = t5(2)
    call ddmul (ad(1,i), t3, t4)
    call ddadd (t2, t4, t5)
    t2(1) = t5(1)
    t2(2) = t5(2)
    call ddmul (t3, x, t4)
    t3(1) = t4(1)
    t3(2) = t4(2)
  enddo

  call dddiv (t1, t2, t3)
  call ddsub (x, t3, t4)
  x(1) = t4(1)
  x(2) = t4(2)
  if (abs (t3(1)) .le. eps) goto 110
enddo

write (6, 1)
1 format ('ddroot: failed to converge.')
call ddabrt

110 continue

return
end subroutine

subroutine ddrand (a)

!   This returns a pseudo-random DD number A between 0 and 1.

implicit none
real*8 f7, r28, r30, r53, r58, s0, s1, s2, sd, t1, t2, t30
parameter (f7 = 78125.d0, s0 = 314159265.d0, r30 = 0.5d0 ** 30, &
  r53 = 0.5d0 ** 53, r58 = 0.5d0 ** 58, t30 = 2.d0 ** 30)
real*8 a(2)
save sd
data sd /s0/

t1 = f7 * sd
t2 = aint (r30 * t1)
s1 = t1 - t30 * t2
t1 = f7 * s1
t2 = aint (r30 * t1)
s2 = t1 - t30 * t2
a(1) = r30 * s1 + r58 * s2
t1 = f7 * s2
t2 = aint (r30 * t1)
s1 = t1 - t30 * t2
t1 = f7 * s1
t2 = aint (r30 * t1)
s2 = t1 - t30 * t2
a(2) = r53 * a(1) * (r30 * s1 + r58 * s2)
sd = s2

return
end subroutine

subroutine ddsqrt (a, b)

!   This computes the square root of the DD number A and returns the DD result
!   in B.

!   This subroutine employs the following formula (due to Alan Karp):

!          Sqrt(A) = (A * X) + 0.5 * [A - (A * X)^2] * X  (approx.)

!   where X is a double precision approximation to the reciprocal square root,
!   and where the multiplications A * X and [] * X are performed with only
!   double precision.

implicit none
real*8 t1, t2, t3
real*8 a(2), b(2), f(2), s0(2), s1(2)

if (a(1) .eq. 0.d0) then
  b(1) = 0.d0
  b(2) = 0.d0
  goto 100
endif
t1 = 1.d0 / sqrt (a(1))
t2 = a(1) * t1
call ddmuldd (t2, t2, s0)
call ddsub (a, s0, s1)
t3 = 0.5d0 * s1(1) * t1
s0(1) = t2
s0(2) = 0.d0
s1(1) = t3
s1(2) = 0.d0
call ddadd (s0, s1, b)

100 continue

return
end subroutine

subroutine ddsub (dda, ddb, ddc)

!   This subroutine computes ddc = dda - ddb.

implicit none
real*8 dda(2), ddb(2), ddc(2)
real*8 e, t1, t2

!   Compute dda + ddb using Knuth's trick.

t1 = dda(1) - ddb(1)
e = t1 - dda(1)
t2 = ((-ddb(1) - e) + (dda(1) - (t1 - e))) + dda(2) - ddb(2)

!   The result is t1 + t2, after normalization.

ddc(1) = t1 + t2
ddc(2) = t2 - (ddc(1) - t1)
return
end subroutine

  real*8 function dddigin (ca, n)
    implicit none
    real*8 d1
    character*(*), ca
    character*16 digits
    integer i, k, n
    parameter (digits = '0123456789')

    d1 = 0.d0

    do i = 1, n
      k = index (digits, ca(i:i)) - 1
      if (k < 0) then
        write (6, *) 'dddigin: non-digit in character string'
      elseif (k <= 9) then
        d1 = 10.d0 * d1 + k
      endif
    enddo

    dddigin = d1
  end function

  character*16 function dddigout (a, n)
    implicit none
    real*8 a, d1, d2
    character*16 ca, digits
    parameter (digits = '0123456789')
    integer i, is, k, n

    ca = ' '
    is = sign (1.d0, a)
    d1 = abs (a)

    do i = n, 1, -1
      d2 = aint (d1 / 10.d0)
      k = 1.d0 + (d1 - 10.d0 * d2)
      d1 = d2
      ca(i:i) = digits(k:k)
      if (d1 == 0.d0) goto 100
    enddo

    i = 0

100 continue

    if (is < 0 .and. i > 1) then
      ca(i-1:i-1) = '-'
    elseif (i == 0 .or. is < 0 .and. i == 1) then
      ca = '****************'
    endif

    dddigout = ca
    return
  end function

end module
