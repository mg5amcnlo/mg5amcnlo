module mpdefmod

!   MPFUN-90 translation modules.

!   IEEE Fortran-90 version
!   David H Bailey    2010-07-16

!   Author:
!      David H. Bailey
!      NERSC, Lawrence Berkeley Lab
!      Mail Stop 50B-2239
!      Berkeley, CA 94720
!      Email: dhbailey@lbl.gov

!   This work was supported by the Director, Office of Science, Division
!   of Mathematical, Information, and Computational Sciences of the
!   U.S. Department of Energy under contract number DE-AC03-76SF00098.
!   See README file accompanying this software for other legal details.

!   A detailed description of this package, and instructions for compiling 
!   and testing this program on various specific systems are included in the
!   README file that accompanies this file.

!  The following notational scheme is used to designate datatypes below:

!  A   Alphabetic [i.e. ASCII]
!  D   Double precision or real*8 [i.e. REAL (KIND (0.D0))]
!  I   Integer
!  J   MP integer
!  Q   MP real
!  X   Double complex or real*16 [i.e. COMPLEX (KIND (0.D0))]
!  Z   MP complex

!  Note that ordinary real*4 and complex*8 types are not included -- if
!  you have code with these datatypes, convert them to real*8 and complex*16.

!  The following parameters are all that need to be changed in normal usage:

!  MPIPL   Initial precision level, in digits.
!  MPIOU   Initial output precision level, in digits.
!  MPIEP   Log_10 of initial MP epsilon level.

use mpfunmod
implicit none
integer mpipl, mpiou, mpiep, mpwds
parameter (mpipl = 2005, mpiou = 56, mpiep = 10 - mpipl)

!----------------------------------------------------------------------------

!  *** No code below this point needs to be altered in normal usage.

parameter (mpwds = mpipl / 7.224719896d0 + 2.d0)
integer, private:: kdb, mp4, mp24, mp41
parameter (kdb = kind (0.d0), mp4 = mpwds + 4, mp24 = 2 * mp4, mp41 = mp4 + 1)
type mp_integer
  sequence
  real mpi(mp4)
end type
type mp_real
  sequence
  real mpr(mp4)
end type
type mp_complex
  sequence
  real mpc(mp24)
end type
type (mp_real), public:: mpl02, mpl10, mppic, mpeps, mplrg, mpsml
integer, public:: mpnwx, mpoud, new_mpipl, new_mpwds

contains

  subroutine mpinit (n_mpipl)

!  MPINIT must be called at the start of execution in the user's main program.
!  It sets the numeric precision level, the MP epsilon level, the output
!  precision level, and computes the constants Pi, Log(2) and Log(10) and
!  trigonometric factors for the cos/sin routine.  Note that the numeric 
!  precision level MPNW is temporarily set to MPNWX + 1, in order to permit
!   extra-accurate calculations of the constants, and then is reset to MPNWX.

    integer, optional:: n_mpipl
    integer i, mpnw
    real*4 t0(mp4+1), t1(mp4+1), t2(mp4+1), t3(mp4+1), t4(mp4+1)

    if (present (n_mpipl)) then
      if (n_mpipl > mpipl) then
        write (mpldb, *) 'mpinit: argument too large'
        stop
      endif
      new_mpipl = n_mpipl
      new_mpwds = n_mpipl / 7.224719896d0 + 2.d0
      mpnwx = new_mpwds
    else
      new_mpipl = mpipl
      new_mpwds = mpwds
      mpnwx = new_mpwds
    endif

!   Compute log2, log10 and pi.

    mpier = 0
    mpnw = mpnwx + 1
    call mppi (t1, mpnw) 
    call mpdmc (2.d0, 0, t0)
    call mplog (t0, t2, t2, mpnw) 
    call mpdmc (10.d0, 0, t0)
    call mplog (t0, t2, t3, mpnw) 
    call mpnpwr (t0, mpiep, t4, mpnw) 
    mpnw = mpnwx
    call mpeq (t1, mppic%mpr, mpnw) 
    call mpeq (t2, mpl02%mpr, mpnw) 
    call mpeq (t3, mpl10%mpr, mpnw) 
    call mpeq (t4, mpeps%mpr, mpnw) 

!   Allocate and compute the mpcosq and mpsinq arrays, needed for cos and sin.

    call mpiniq (mpwds, mpnw)

!   Set mpoud, mplrg and mpsml.

    mpoud = mpiou
    mplrg%mpr(1) = 1.
    mplrg%mpr(2) = 2.e6
    mplrg%mpr(3) = 1.
    mplrg%mpr(4) = 0.
    mpsml%mpr(1) = 1.
    mpsml%mpr(2) = -2.e6
    mpsml%mpr(3) = 1.
    mpsml%mpr(4) = 0.
    return
  end subroutine

  subroutine mpsetprec (num_digits)
    integer num_digits
    if (num_digits > new_mpipl) then
      write (mpldb, *) 'mpsetprec: invalid argument; precision set to ', &
        new_mpipl, ' digits'
      mpnwx = new_mpwds
    else
      mpnwx = num_digits / 7.224719896d0 + 2.d0
    endif
  end subroutine

  subroutine mpgetprec (num_digits)
    integer num_digits
    num_digits = (mpnwx - 2) * 7.224719896d0
  end subroutine

  subroutine mpsetprecwords (num_words)
    integer num_words
    if (num_words > new_mpwds) then
      write (mpldb, *) 'mpsetprecwords: invalid argument; precision set to ', &
        new_mpwds, ' words'
      mpnwx = new_mpwds
    else
      mpnwx = num_words
    endif
  end subroutine

  subroutine mpgetprecwords (num_words)
    integer num_words
    num_words = mpnwx
  end subroutine

  subroutine mpsetoutputprec (num_digits)
    integer num_digits
    if (num_digits > new_mpipl) then
      write (mpldb, *) &
        'mpsetoutputprec: invalid argument; output precision set to ', &
          new_mpipl, ' digits'
      mpoud = new_mpipl
    else
      mpoud = num_digits
    endif
  end subroutine

  subroutine mpgetoutputprec (num_digits)
    integer num_digits
    num_digits = mpoud
  end subroutine

  subroutine mpgetpar (s, n, k)
!  MPGETPAR retrieves some ARPREC C++ integer parameters.
      character*(*), intent(in) :: s
      integer, intent(out) :: n
      integer, intent(in), optional :: k
      
      if (s == 'mpnw') then 
        n = mpnwx
      elseif (s == 'mpidb') then
        n = mpidb
      elseif (s == 'mpndb') then
        n = mpndb
      elseif (s == 'mpmcr') then
        n = mpmcr
      elseif (s == 'mpird') then
        n = mpird
      elseif (s == 'mpier') then
        n = mpier
      elseif (s == 'mpker') then
        n = mpker(k)
      else
        write (mpldb, 1) s
1       format ('mpgetpar: invalid parameter name: ',a)
        n = 0
      endif
  end subroutine

  subroutine mpsetpar (s, n, k)
!  MPSETPAR sets some ARPREC C++ integer parameters.
      character*(*), intent(in) :: s
      integer, intent(in) :: n
      integer, intent(in), optional :: k
      
      if (s == 'mpnw') then 
        mpnwx = n
      elseif (s == 'mpidb') then
        mpidb = n
      elseif (s == 'mpndb') then
        mpndb = n
      elseif (s == 'mpmcr') then
        mpmcr = n
      elseif (s == 'mpird') then
        mpird = n
      elseif (s == 'mpier') then
        mpier = n
      elseif (s == 'mpker') then
        mpker(k) = n
      else
        write (mpldb, 1) s
1       format ('mpsetpar: invalid parameter name: ',a)
      endif
  end subroutine

  subroutine mpeform (a, n1, n2, b)
    type (mp_real) a
    integer n1, n2
    character*1 b(n1)
    integer mpnw
    mpnw = mpnwx
    call mpeformx (a%mpr, n1, n2, b, mpnw)
    return
  end subroutine
    
  subroutine mpfform (a, n1, n2, b)
    type (mp_real) a
    integer n1, n2
    character*1 b(n1)
    integer mpnw
    mpnw = mpnwx
    call mpfformx (a%mpr, n1, n2, b, mpnw)
    return
  end subroutine
    
  subroutine mpdexc (a, l, b, mpnw)

!   This routine converts the character*1 string A, which
!   represents a multiprecision number in Fortran style, i.e.
!   '1234567890' or '1.23456789D-21', into standard MP binary format.
!   This routine is not intended to be called directly by the user.

    integer i, i1, l, l1, l2, mpnw
    character*1 a(l), c(mpipl+100)
    real b(mpnw+4)

    do i = 1, l
      if (a(i) .eq. 'D' .or. a(i) .eq. 'E' .or. a(i) .eq. 'd' &
        .or. a(i) .eq. 'e') goto 100
    enddo

    call mpinpc (a, l, b, mpnw) 
    goto 110

100 i1 = i
    l1 = i - 1
    l2 = l - i
    c(1) = '1'
    c(2) = '0'
    c(3) = '^'

    do i = 1, l2
      c(i+3) = a(i+i1)
    enddo

    c(l2+4) = 'x'

    do i = 1, l1
      c(i+l2+4) = a(i)
    enddo

    call mpinpc (c, l1 + l2 + 4, b, mpnw) 
110 return
  end subroutine

subroutine mpinp (iu, a, cs, mpnw)

!   This routine reads the MP number A from logical unit IU.  CS is a scratch
!   array of type CHARACTER*1.  CS must be dimensioned at least 7.225*MPNW
!   + 100.   The digits of A may span more than one line.  A comma at the end
!   of the last line denotes the end of the MP number.  The input lines may not
!   exceed 200 characters in length.  Embedded blanks are allowed anywhere.
!   However, if the input number contains more than 80 embedded blanks, then
!   the dimension of CS must be increased by a corresponding amount.  The
!   exponent is optional in the input number, but if present it must appear
!   first.  Two examples:

!   1073741824.,
!   10 ^  -4 x  3.14159 26535 89793 23846 26433 83279
!     50288 41971 69399 37510,

!   Max SP space for A: MPNW + 4 cells.

integer i, iu, l, l1, mpnw, nn
character*200 line
character*1 cs(*)
real a(mpnw+2)

if (mpier .ne. 0) then
  if (mpier .eq. 99) call mpabrt
  a(1) = 0.
  a(2) = 0.
  return
endif
l = 0
nn = mpipl + 100

100 continue
read (iu, '(A)', end = 200) line

do i = 200, 1, -1
  if (line(i:i) .ne. ' ') goto 110
enddo

i = 0
goto 100

110 continue
l1 = i

do i = 1, l1
  if (line(i:i) == ',') goto 150
  if (l + 1 <= nn) then
    l = l + 1
    cs(l) = line(i:i)
  endif
enddo

goto 100

150  continue

call mpinpc (cs, l, a, mpnw)
goto 300

200  continue

if (mpker(72) .ne. 0) then
  write (mpldb, 1)
1 format ('*** MPINP: End-of-file encountered.')
  mpier = 72
  if (mpker(mpier) .eq. 2) call mpabrt
endif

300 return
end subroutine

subroutine mpinpc (a, n, b, mpnw)

!   Converts the CHARACTER*1 array A of length N into the MP number B.  The
!   string A must be in the format '10^s a x tb.c' where a, b and c are digit
!   strings; s and t are '-', '+' or blank; x is either 'x' or '*'.  Blanks may
!   be embedded anywhere.  The digit string a is limited to nine digits and
!   80 total characters, including blanks.  The exponent portion (i.e. the
!   portion up to and including x) and the period may optionally be omitted.
!   Debug output starts with MPIDB = 7.

!   Max SP space for B: MPNW + 4 cells.

!   The following example shows how this routine may be used to input a MP
!   number:

!   CHARACTER*1 CX(800)
!   READ (1, '(80A1)') (CX(I), I = 1, ND)
!   CALL MPINPC (CX, ND, B)

integer i, ib, id, ier, ip, is, it, i1, i2, k0, k1, k2, l1, mpnw, n, nb, &
  nn, no, nws, n5
double precision bi
character*1 a(n), ai
character*10 dig
character*80 ca
parameter (dig = '0123456789')
real b(mpnw+4), f(8), s(3*mpnw+15)
! real*8 mpdigin

if (mpier .ne. 0) then
  if (mpier .eq. 99) call mpabrt
  b(1) = 0.
  b(2) = 0.
  return
endif
if (mpidb .ge. 7) then
  no = min (n, int (7.225 * mpndb) + 20)
  write (mpldb, 1) (a(i), i = 1, no)
1 format ('MPINPC I'/(78a1))
endif

n5 = mpnw + 5
k0 = 1
k1 = k0 + n5
k2 = k1 + n5
nws = mpnw
mpnw = mpnw + 1
i1 = 1
nn = 0

!   Find the carat, period, plus or minus sign, whichever comes first.

do i = 1, n
  ai = a(i)
  if (ai .eq. '^') goto 110
  if (ai .eq. '.' .or. ai .eq. '+' .or. ai .eq. '-') goto 160
enddo

goto 160

!   Make sure number preceding the carat is 10.

110 continue

i2 = i - 1
if (i2 .gt. 80) then
  ier = 1
  goto 210
endif
ca = ' '

do i = 1, i2
  ai = a(i)
  if (ai .eq. ' ') then
    goto 120
  elseif (index (dig, ai) .eq. 0) then
    ier = 2
    goto 210
  endif
  ca(i:i) = ai
120  continue
enddo

nn = mpdigin (ca, 80)
if (nn .ne. 10) then
  ier = 3
  goto 210
endif
i1 = i2 + 2

!   Find the x or *.

do i = i1, n
  ai = a(i)
  if (ai .eq. 'x' .or. ai .eq. '*') goto 140
enddo

ier = 4
goto 210

!   Convert the exponent.

140  i2 = i - 1
l1 = i2 - i1 + 1
if (l1 .gt. 80) then
  ier = 5
  goto 210
endif
ca = ' '
id = 0
is = 1

do i = 1, l1
  ai = a(i+i1-1)
  if (ai .eq. ' ' .or. ai .eq. '+') then
    goto 150
  elseif (ai .eq. '-' .and. id .eq. 0) then
    id = 1
    is = -1
    ca(i:i) = ' '
  else
    if (index (dig, ai) .eq. 0) then
      ier = 6
      goto 210
    endif
    id = 1
    ca(i:i) = ai
  endif
150  continue
enddo

nn = is * mpdigin (ca, 80)
i1 = i2 + 2

!   Find the next nonblank character.

160  do i = i1, n
  if (a(i) .ne. ' ') goto 180
enddo

ier = 7
goto 210

!   Check if the nonblank character is a plus or minus sign.

180 continue

i1 = i
if (a(i1) .eq. '+') then
  i1 = i1 + 1
  is = 1
elseif (a(i1) .eq. '-') then
  i1 = i1 + 1
  is = -1
else
  is = 1
endif
nb = 0
ib = 0
id = 0
ip = 0
s(k2) = 0.
s(k2+1) = 0.
f(1) = 1.
f(2) = 0.
it = 0

190 continue

ip = 0
ca(1:6) = '000000'

!   Scan for digits, looking for the period also.  On the first pass we just
!   count, so that on the second pass it will come out right.

do i = i1, n
  ai = a(i)
  if (ai .eq. ' ') then
  elseif (ai .eq. '.') then
    if (ip .ne. 0) then
      ier = 8
      goto 210
    endif
    ip = id
  elseif (index (dig, ai) .eq. 0) then
    ier = 9
    goto 210
  else
    ib = ib + 1
    id = id + 1
    ca(ib:ib) = ai
  endif
  if (ib .eq. 6 .or. i .eq. n .and. ib .ne. 0) then
    if (it .ne. 0) then
      nb = nb + 1
      bi = mpdigin (ca(1:6), 6)
      call mpmuld (s(k2), 1.d6, 0, s(k0), mpnw) 
      if (bi .ne. 0) then
        f(1) = 1.
        f(3) = bi
      else
        f(1) = 0.
      endif
      call mpadd (s(k0), f, s(k2), mpnw) 
      ca(1:6) = '000000'
    endif
    if (i .ne. n) ib = 0
  endif
enddo

if (it .eq. 0) then
  ib = 6 - ib
  if (ib .eq. 6) ib = 0
  it = 1
  goto 190
endif
if (is .eq. -1) s(k2) = - s(k2)
if (ip .eq. 0) ip = id
nn = nn + ip - id
f(1) = 1.
f(3) = 10.
call mpnpwr (f, nn, s(k0), mpnw) 
call mpmul (s(k2), s(k0), s(k1), mpnw) 
call mpeq (s(k1), b, mpnw) 
mpnw = nws
call mproun (b, mpnw) 

if (mpidb .ge. 7) then
  no = min (int (abs (b(1))), mpndb) + 2
  write (mpldb, 2) (b(i), i = 1, no)
2 format ('MPINPC O'/(6f12.0))
endif
goto 220

210 continue

if (mpker(41) .ne. 0) then
  write (mpldb, 3)
3 format ('*** MPINPC: Syntax error in literal string.')
  mpier = 41
  if (mpker(mpier) .eq. 2) call mpabrt
  mpnw = nws
endif

220  return
end subroutine

subroutine mpout (iu, a, la, cs, mpnw)

!   This routine writes the exponent plus LA mantissa digits of the MP number
!   A to logical unit IU.  CS is a scratch array of type CHARACTER*1.  CS must
!   be dimensioned at least LA + 25.  The digits of A may span more than one
!   line.  A comma is placed at the end of the last line to denote the end of
!   the MP number.  Here is an example of the output:

!   10 ^  -4 x  3.14159265358979323846264338327950288419716939937510,

integer i, iu, l, la, ll, mpnw, nws
character*1 cs(la+25)
real a(mpnw+2)

if (mpier .ne. 0) return

nws = mpnw
ll = la / log10 (mpbdx) + 2.d0
mpnw = min (mpnw, ll)
call mpoutc (a, cs, l, mpnw) 
mpnw = nws
l = min (l, la + 20) + 1
cs(l) = ','
write (iu, '(78A1)') (cs(i), i = 1, l)

return
end subroutine

subroutine mpoutc (a, b, n, mpnw)

!   Converts the MP number A into character form in the CHARACTER*1 array B.
!   N (an output parameter) is the length of the output.  In other words, B is
!   contained in B(1), ..., B(N).  The format is analogous to the Fortran
!   exponential format (E format), except that the exponent is placed first.
!   Debug output starts with MPIDB = 7.

!   Max CHARACTER*1 space for B: 7.225 * MPNW + 30 cells.

!   This routine is called by MPOUT, but it may be directly called by the user
!   if desired for custom output.  Example:

!   CHARACTER*1 CX(800)
!   CALL MPOUTC (A, CX, ND)
!   WRITE (1, '(20A1/(72A1))') (CX(I), I = 1, ND)

integer i, ia, ix, j, k0, k1, l, mpnw, na, nl, n5, nws, n, no, nx
character*1 b(n)
character*16 ca
real*8 aa, al2, con, t1
parameter (al2 = 0.301029995663981195d0, con = 0.8304820235d0)
real a(mpnw+2), f(8), s(2*mpnw+10)
real*8 an
! character*16 mpdigout

if (mpier .ne. 0) then
  if (mpier .eq. 99) call mpabrt
  b(1) = ' '
  n = 0
  return
endif
if (mpidb .ge. 7) then
  no = min (int (abs (a(1))), mpndb) + 2
  write (mpldb, 1) (a(i), i = 1, no)
1 format ('MPOUTC I'/(6f12.0))
endif

ia = sign (1., a(1))
na = min (int (abs (a(1))), mpnw)
n5 = mpnw + 5
k0 = 1
k1 = k0 + n5
nws = mpnw
mpnw = mpnw + 1
f(1) = 1.
f(2) = 0.
f(3) = 10.

!   Determine exact power of ten for exponent.

if (na .ne. 0) then
  aa = a(3)
  if (na .ge. 2) aa = aa + mprdx * a(4)
  if (na .ge. 3) aa = aa + mprx2 * a(5)
  if (na .ge. 4) aa = aa + mprdx * mprx2 * a(6)
  t1 = al2 * mpnbt * a(2) + log10 (aa)
  if (t1 .ge. 0.d0) then
    nx = t1
  else
    nx = t1 - 1.d0
  endif
  call mpnpwr (f, nx, s(k0), mpnw) 
  call mpdiv (a, s(k0), s(k1), mpnw) 

!   If we didn't quite get it exactly right, multiply or divide by 10 to fix.

100 continue

    if (s(k1+1) .lt. 0.) then
    nx = nx - 1
    call mpmuld (s(k1), 10.d0, 0, s(k0), mpnw) 
    call mpeq (s(k0), s(k1), mpnw) 
    goto 100
  elseif (s(k1+2) .ge. 10.) then
    nx = nx + 1
    call mpdivd (s(k1), 10.d0, 0, s(k0), mpnw) 
    call mpeq (s(k0), s(k1), mpnw) 
    goto 100
  endif
  s(k1) = abs (s(k1))
else
  nx = 0
endif

!   Place exponent first instead of at the very end as in Fortran.

b(1) = '1'
b(2) = '0'
b(3) = ' '
b(4) = '^'
ca = mpdigout (dble (nx), 10)

do i = 1, 10
  b(i+4) = ca(i:i)
enddo

b(15) = ' '
b(16) = 'x'
b(17) = ' '

!   Insert sign and first digit.

if (ia .eq. -1) then
  b(18) = '-'
else
  b(18) = ' '
endif
if (na .ne. 0) then
  an = s(k1+2)
else
  an = 0
endif
ca = mpdigout (an, 1)
b(19) = ca(1:1)
b(20) = '.'
ix = 20
if (na .eq. 0) goto 190
f(3) = an
call mpsub (s(k1), f, s(k0), mpnw) 
if (s(k0) .eq. 0) goto 190
call mpmuld (s(k0), 1.d6, 0, s(k1), mpnw) 
nl = max (mpnw * log10 (mpbdx) / 6.d0 - 1.d0, 1.d0)
nl = min (nl, mpoud/6 + 1)

!   Insert the digits of the remaining words.

do j = 1, nl
  if (s(k1+1) .eq. 0.) then
    an = s(k1+2)
    f(1) = 1.
    f(3) = an
  else
    f(1) = 0.
    an = 0.
  endif
  ca = mpdigout (an, 6)

  do i = 1, 6
    if (ca(i:i) == ' ') ca(i:i) = '0'
    b(i+ix) = ca(i:i)
  enddo

  ix = ix + 6
  call mpsub (s(k1), f, s(k0), mpnw) 
  call mpmuld (s(k0), 1.d6, 0, s(k1), mpnw) 
  if (s(k1) .eq. 0.) goto 140
enddo

!   Check if trailing zeroes should be trimmed.

j = nl + 1

140  l = ix
if (b(l) .eq. '0' .and. b(l-1) .eq. '0' .or. (j .gt. nl .and. b(l-2) .eq. '0' &
   .and. b(l-3) .eq. '0')) then
  b(l) = ' '
  b(l-1) = ' '

  do i = l - 2, 21, -1
    if (b(i) .ne. '0') then
      ix = i
      goto 190
    endif
    b(i) = ' '
  enddo

  ix = 20

!   Check if trailing nines should be rounded up.

elseif (j .gt. nl .and. b(l-2) .eq. '9' .and. b(l-3) .eq. '9') then
  b(l) = ' '
  b(l-1) = ' '

  do i = l - 2, 21, -1
    if (b(i) .ne. '9') goto 180
    b(i) = ' '
  enddo

!   We have rounded away all digits to the right of the decimal point, and the
!   digit to the left of the digit is a 9.  Set the digit to 1 and increase
!   the exponent by one.

  ix = 20
  if (b(19) .eq. '9') then
    b(19) = '1'
    ca = mpdigout (dble (nx+1), 10)

    do i = 1, 10
      b(i+4) = ca(i:i)
    enddo
  else
    ca = b(19)
    an = mpdigin (ca, 1)
    ca = mpdigout (an + 1.d0, 1)
    b(19) = ca(1:1)
  endif
  goto 190

180 continue

  ca = b(i)
  an = mpdigin (ca, 1)
  ca = mpdigout (an + 1.d0, 1)
  b(i) = ca(1:1)
  ix = i
endif

190 continue

n = min (ix, mpoud + 20)
mpnw = nws
if (mpidb .ge. 7) then
  no = min (n, 6 * mpndb + 20)
  write (mpldb, 2) (b(i), i = 1, no)
2 format ('MPOUTC O'/(78a1))
endif
return
end subroutine

recursive subroutine mpoutcx (a, b, n, mpnw)

!   Converts the MP number A into character form in the CHARACTER*1 array B.
!   N (an output parameter) is the length of the output.  In other words, B is
!   contained in B(1), ..., B(N).  The format is analogous to the Fortran
!   exponential format (E format), except that the exponent is placed first.
!   Before calling MPOUTX, the arrays UU1 and UU2 must be initialized by
!   calling MPINIX.  For modest levels of precision, use MPOUTC.  Debug output
!   starts with MPIDB = 7.

!   Max CHARACTER*1 space for B: 7.225 * MPNW + 30 cells.

implicit none
integer i, ia, ie, ie1, ie2, i1, i2, k0, k1, k2, k3, k4, m1, m2, &
  mpnw, mpnws, n, na, nb1, nb2, ncr, no, n4
double precision al2, t1, t2, t3
character*1 b(*), b1(8*mpnw+30), b2(8*mpnw+30)
character*10 dig
character*16 c1, c2
parameter (al2 = 0.301029995663981195d0, dig = '0123456789')
real a(mpnw+2), s(5*mpnw+20)

if (mpier .ne. 0) then
  if (mpier .eq. 99) call mpabrt
  b(1) = ' '
  n = 0
  return
endif
if (mpidb .ge. 7) then
  no = min (int (abs (a(1))), mpndb) + 2
  write (mpldb, 1) (a(i), i = 1, no)
1 format ('MPOUTCX I'/(6f12.0))
endif

ia = sign (1., a(1))
na = min (int (abs (a(1))), mpnw)
n4 = mpnw + 4
k0 = 1
k1 = k0 + n4
k2 = k1 + n4
k3 = k2 + n4
k4 = k3 + n4
ncr = 2 ** mpmcr

!   Check if actual precision level of argument is too low to justify the
!   advanced routine.

if (na .le. ncr) then
  call mpoutc (a, b, n, mpnw) 
  goto 110
endif

!   Normalize input to an integer by multiplying by a suitable power of 10.

t1 = a(3) + mprdx * a(4) + mprx2 * a(5)
t2 = log10 (t1)
m1 = max (al2 * mpnbt * (abs (a(1)) - a(2)) - t2, 0.d0)
call mpdmc (10.d0, 0, s(k0))
call mpnpwx (s(k0), m1, s(k2), mpnw) 
call mpmulx (a, s(k2), s(k1), mpnw) 
s(k1) = abs (s(k1))

!   Split large integer into two approximately equal decimal sections.

call mpmdc (s(k1), t1, i1)
call dpdec (t1, i1, t2, i2) 
m2 = i2 / 2
call mpnpwx (s(k0), m2, s(k3), mpnw) 
call mpdivx (s(k1), s(k3), s(k0), mpnw) 
call mpinfr (s(k0), s(k2), s(k4), mpnw) 
call mpmulx (s(k2), s(k3), s(k0), mpnw) 
call mpsub (s(k1), s(k0), s(k3), mpnw) 

!   Recursively convert each section.

mpnws = mpnw
mpnw = s(k2) + 1
call mpoutcx (s(k2), b1, nb1, mpnw) 
mpnw = s(k3) + 1
call mpoutcx (s(k3), b2, nb2, mpnw) 
mpnw = mpnws

!   Obtain decimal exponents from each section.

c1 = ' '
c2 = ' '

do i = 1, 10
  c1(i:i) = b1(i+4)
  c2(i:i) = b2(i+4)
enddo

read (c1, '(I10)') ie1
read (c2, '(I10)') ie2

!   Set exponent of result.

ie = ie1 + m2 - m1
write (c1, '(I14)') ie

do i = 1, 4
  b(i) = b1(i)
enddo

do i = 5, 14
  b(i) = c1(i:i)
enddo

!   Copy mantissa of first section.

do i = 15, nb1
  b(i) = b1(i)
enddo

i1 = ie1 + m2 - ie2 + 19

!   If first section is too long, then round trailing digits (probably 9s).

if (nb1 .gt. i1) then
  i2 = index (dig, b(i1+1)) - 1
  if (i2 .ge. 5) then
    do i = i1, 21, -1
      if (b(i) .ne. '9') goto 100
      b(i) = '0'
    enddo

    write (mpldb, 2)
2   format ('*** MPOUTCX: Exceptional case -- contact DHB.')
    stop

100 i2 = index (dig, b(i)) - 1
    write (c1, '(I1)') i2 + 1
    b(i) = c1(1:1)
  endif
elseif (nb1 .lt. i1) then

!   If first section is too short, then insert zeroes in gap.

  do i = nb1 + 1, i1
    b(i) = '0'
  enddo
endif

!   Copy mantissa of second section.

b(i1+1) = b2(19)
n = min (i1 + nb2 - 19, int (7.225 * mpnw + 30))

do i = i1 + 1, n
  b(i) = b2(i-i1+19)
enddo

!   Fix sign.

if (ia .eq. -1) b(18) = '-'

110 continue

if (mpidb .ge. 7) then
  no = min (n, 6 * mpndb + 20)
  write (mpldb, 3) (b(i), i = 1, no)
3 format ('MPOUTCX O'/(78a1))
endif

return
end subroutine

  real*8 function mpdigin (ca, n)
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
        write (mpldb, *) 'mpdigin: non-digit in character string'
      elseif (k <= 9) then
        d1 = 10.d0 * d1 + k
      endif
    enddo

    mpdigin = d1
  end function

  character*16 function mpdigout (a, n)
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

100  continue

    if (is < 0 .and. i > 1) then
      ca(i-1:i-1) = '-'
    elseif (i == 0 .or. is < 0 .and. i == 1) then
      ca = '****************'
    endif

    mpdigout = ca
    return
  end function

subroutine mpeformx (a, n1, n2, b, mpnw)

!   This routine converts the MP number A to E format, i.e. E N1.N2.
!   B is the output array (type CHARACTER*1) of size N1.

      integer i, j, k, lex, n, n1, n2, mpnw
      real*4 a(mpnw+4)
      character*1 b(n1), c(8*mpnw+100)

      if (n1 > mpoud) then
        write (mpldb, '("*** mpeformx: mpoud must exceed n1")')
        goto 110
      endif
      call mpoutc (a, c, n, mpnw)

!   Find length of exponent field.

      do i = 5, 14
        if (c(i) /= ' ') goto 100
      enddo

100   continue

      lex = 15 - i
      k = n1 - lex - n2 - 4

!   Check for overflow of field length.

      if (k < 0) then
         do j = 1, n1
            b(j) = '*'
         enddo

         goto 110
      endif

!   Copy characters to appropriate positions.

      do j = 1, k
        b(j) = ' '
      enddo

      do j = 1, min (n2 + 3, n - 17)
        b(j+k) = c(j+17)
      enddo

      do j = n - 16, n2 + 3
        b(j+k) = '0'
      enddo

      b(k+n2+4) = 'e'

      do j = 1, lex
        b(j+k+n2+4) = c(i+j-1)
      enddo

110   continue

  return
  end subroutine

subroutine mpfformx (a, n1, n2, b, mpnw)

!   This routine converts the MP number A to F format, i.e. F N1.N2.
!   B is the output array (type CHARACTER*1) of size N1.

integer i, ix, kx, ls, lz, mpnw, mx, n, n1, n2, nx
real a(mpnw+2)
character*1 b(n1), c(8*mpnw+100)
character*16 chr16
! real*8 mpdigin

if (n1 > mpoud) then
  write (mpldb, '("*** mpfformx: mpoud must exceed n1")')
  goto 200
endif

call mpoutc (a, c, n, mpnw) 
chr16 = ' '

do i = 1, 10
  chr16(i:i) = c(i+4)
enddo

ix = mpdigin (chr16, 16)
if (a(1) .ge. 0.) then
  ls = 0
else
  ls = 1
endif
if (ix .ge. 0 .and. a(1) .ne. 0.) then
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

if (a(1) .eq. 0 .or. -ix .gt. n2) then
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

do i = 1, n1 - n2 - mx - 2
  b(i) = ' '
enddo

if (a(1) .lt. 0.) b(n1-n2-mx-2) = '-'
if (ix .ge. 0) then
  b(n1-n2-ix-1) = c(19)
  kx = min (n - 20, ix)

  do i = 1, kx
    b(i+n1-n2-ix-1) = c(i+20)
  enddo

  do i = kx + 1, ix
    b(i+n1-n2-ix-1) = '0'
  enddo

  b(n1-n2) = '.'
  kx = max (min (n - ix - 20, n2), 0)

  do i = 1, kx
    b(i+n1-n2) = c(i+ix+20)
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

  b(n1-n2+nx) = c(19)
  kx = min (n - 20, n2 - nx)

  do i = 1, kx
    b(i+n1-n2+nx) = c(i+20)
  enddo

  do i = kx + 1, n2 - nx
    b(i+n1-n2+nx) = '0'
  enddo
endif

200 continue

return
end subroutine

  subroutine mpdotd (n, isa, a, isb, db, c)
!   This routine computes the dot product of the MP vector A with the DP
!   vector DB, returning the MP result in C.  This routine is used in the
!   author's customized PSLQ routine, resulting in substantial speedup.
!   The length of both the A and DB vectors is N, and ISA and ISB are the 
!   skip distances between successive elements of A and DB, measured in 
!   MP words and DP words, respectively.  The DP values in DB must be
!   whole numbers, so for example they cannot be larger than 2^53.

      integer n, isa, isb
      double precision db(isb*n)
      type (mp_real) a(isa*n), c
      integer mpnw
      mpnw = mpnwx
      call mpdotdx (n, isa * (mpwds + 4), a(1)%mpr, isb, db, c%mpr, mpnw)
  end subroutine

  subroutine mpxzc (a, b)

!  This converts the DC variable A to the MPC variable B.
!  This routine is not intended to be called directly by the user.

    complex (kdb) a
    double precision da
    real b(mp24)
    da = a
    call mpdmc (da, 0, b)
    da = aimag (a)
    call mpdmc (da, 0, b(mp41))
    return
  end subroutine

  subroutine mpmzc (a, b)

!  This converts the MP real or MP integer variable A to the MPC variable B.
!  This routine is not intended to be called directly by the user.

    real a(mp4), b(mp24)
    integer mpnw
    mpnw = mpnwx
    call mpeq (a, b, mpnw) 
    b(mp41) = 0.
    b(mp4+2) = 0.
    return
  end subroutine

end module


module mpintmod

!  This Fortran-90 module defines operator extensions involving the
!  MP_INTEGER datatype.  For operations involving two MP data types,
!  those whose first argument is MP_INTEGER are included here.
!  Others are handled in other modules.

!  The subroutines and functions defined in this module are private
!  and not intended to be called directly by the user.

use mpfunmod
use mpdefmod
private kdb, mp4, mp24, mp41
parameter (kdb = kind (0.d0), mp4 = mpwds + 4, mp24 = 2 * mp4, mp41 = mp4 + 1)
private &
  mp_eqjj, mp_eqjq, mp_eqjz, mp_eqij, mp_eqji, &
  mp_eqdj, mp_eqjd, mp_eqxj, mp_eqjx, mp_eqja, &
  mp_addjj, mp_addjq, mp_addjz, mp_addij, mp_addji, &
  mp_adddj, mp_addjd, mp_addxj, mp_addjx, &
  mp_subjj, mp_subjq, mp_subjz, mp_subij, mp_subji, &
  mp_subdj, mp_subjd, mp_subxj, mp_subjx, mp_negj, &
  mp_muljj, mp_muljq, mp_muljz, mp_mulij, mp_mulji, &
  mp_muldj, mp_muljd, mp_mulxj, mp_muljx, &
  mp_divjj, mp_divjq, mp_divjz, mp_divij, mp_divji, &
  mp_divdj, mp_divjd, mp_divxj, mp_divjx, &
  mp_expjj, mp_expjq, mp_expij, mp_expji, mp_expdj, mp_expjd, &
  mp_eqtjj, mp_eqtjq, mp_eqtjz, mp_eqtij, mp_eqtji, &
  mp_eqtdj, mp_eqtjd, mp_eqtxj, mp_eqtjx, &
  mp_netjj, mp_netjq, mp_netjz, mp_netij, mp_netji, &
  mp_netdj, mp_netjd, mp_netxj, mp_netjx, &
  mp_letjj, mp_letjq, mp_letij, mp_letji, mp_letdj, mp_letjd, &
  mp_getjj, mp_getjq, mp_getij, mp_getji, mp_getdj, mp_getjd, &
  mp_lttjj, mp_lttjq, mp_lttij, mp_lttji, mp_lttdj, mp_lttjd, &
  mp_gttjj, mp_gttjq, mp_gttij, mp_gttji, mp_gttdj, mp_gttjd

!  MPI operator extension interface blocks.

interface assignment (=)
  module procedure mp_eqjj
  module procedure mp_eqjq
  module procedure mp_eqjz
  module procedure mp_eqij
  module procedure mp_eqji
  module procedure mp_eqdj
  module procedure mp_eqjd
  module procedure mp_eqxj
  module procedure mp_eqjx

  module procedure mp_eqja
end interface

interface operator (+)
  module procedure mp_addjj
  module procedure mp_addjq
  module procedure mp_addjz
  module procedure mp_addij
  module procedure mp_addji
  module procedure mp_adddj
  module procedure mp_addjd
  module procedure mp_addxj
  module procedure mp_addjx
end interface

interface operator (-)
  module procedure mp_subjj
  module procedure mp_subjq
  module procedure mp_subjz
  module procedure mp_subij
  module procedure mp_subji
  module procedure mp_subdj
  module procedure mp_subjd
  module procedure mp_subxj
  module procedure mp_subjx

  module procedure mp_negj
end interface

interface operator (*)
  module procedure mp_muljj
  module procedure mp_muljq
  module procedure mp_muljz
  module procedure mp_mulij
  module procedure mp_mulji
  module procedure mp_muldj
  module procedure mp_muljd
  module procedure mp_mulxj
  module procedure mp_muljx
end interface

interface operator (/)
  module procedure mp_divjj
  module procedure mp_divjq
  module procedure mp_divjz
  module procedure mp_divij
  module procedure mp_divji
  module procedure mp_divdj
  module procedure mp_divjd
  module procedure mp_divxj
  module procedure mp_divjx
end interface

interface operator (**)
  module procedure mp_expjj
  module procedure mp_expjq
  module procedure mp_expij
  module procedure mp_expji
  module procedure mp_expdj
  module procedure mp_expjd
end interface

interface operator (.eq.)
  module procedure mp_eqtjj
  module procedure mp_eqtjq
  module procedure mp_eqtjz
  module procedure mp_eqtij
  module procedure mp_eqtji
  module procedure mp_eqtdj
  module procedure mp_eqtjd
  module procedure mp_eqtxj
  module procedure mp_eqtjx
end interface

interface operator (.ne.)
  module procedure mp_netjj
  module procedure mp_netjq
  module procedure mp_netjz
  module procedure mp_netij
  module procedure mp_netji
  module procedure mp_netdj
  module procedure mp_netjd
  module procedure mp_netxj
  module procedure mp_netjx
end interface

interface operator (.le.)
  module procedure mp_letjj
  module procedure mp_letjq
  module procedure mp_letij
  module procedure mp_letji
  module procedure mp_letdj
  module procedure mp_letjd
end interface

interface operator (.ge.)
  module procedure mp_getjj
  module procedure mp_getjq
  module procedure mp_getij
  module procedure mp_getji
  module procedure mp_getdj
  module procedure mp_getjd
end interface

interface operator (.lt.)
  module procedure mp_lttjj
  module procedure mp_lttjq
  module procedure mp_lttij
  module procedure mp_lttji
  module procedure mp_lttdj
  module procedure mp_lttjd
end interface

interface operator (.gt.)
  module procedure mp_gttjj
  module procedure mp_gttjq
  module procedure mp_gttij
  module procedure mp_gttji
  module procedure mp_gttdj
  module procedure mp_gttjd
end interface

contains

!  MPI assignment routines.

  subroutine mp_eqjj (ja, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    intent (out):: ja
    intent (in):: jb
    integer mpnw
    mpnw = mpnwx
    call mpeq (jb%mpi, ja%mpi, mpnw) 
    return
  end subroutine

  subroutine mp_eqjq (ja, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    intent (out):: ja
    intent (in):: qb
    type (mp_real) q1, q2
    integer mpnw
    mpnw = mpnwx
    call mpeq (qb%mpr, q1%mpr, mpnw) 
    call mpinfr (q1%mpr, ja%mpi, q2%mpr, mpnw) 
    return
  end subroutine

  subroutine mp_eqjz (ja, zb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    intent (out):: ja
    intent (in):: zb
    type (mp_real) q1, q2
    integer mpnw
    mpnw = mpnwx
    call mpeq (zb%mpc, q1%mpr, mpnw) 
    call mpinfr (q1%mpr, ja%mpi, q2%mpr, mpnw) 
    return
  end subroutine

  subroutine mp_eqij (ia, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    intent (out):: ia
    intent (in):: jb
    integer mpnw
    mpnw = mpnwx
    call mpmdc (jb%mpi, db, ib)
    ia = db * 2.d0 ** ib
    return
  end subroutine

  subroutine mp_eqji (ja, ib)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    intent (out):: ja
    intent (in):: ib
    integer mpnw
    mpnw = mpnwx
    db = ib
    call mpdmc (db, 0, ja%mpi)
    return
  end subroutine

  subroutine mp_eqdj (da, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    intent (out):: da
    intent (in):: jb
    integer mpnw
    mpnw = mpnwx
    call mpmdc (jb%mpi, db, ib)
    da = db * 2.d0 ** ib
    return
  end subroutine

  subroutine mp_eqjd (ja, db)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    intent (out):: ja
    intent (in):: db
    type (mp_real) q1, q2
    integer mpnw
    mpnw = mpnwx
    call mpdmc (db, 0, q1%mpr)
    call mpinfr (q1%mpr, ja%mpi, q2%mpr, mpnw) 
    return
  end subroutine

  subroutine mp_eqxj (xa, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    intent (out):: xa
    intent (in):: jb
    integer mpnw
    mpnw = mpnwx
    call mpmdc (jb%mpi, db, ib)
    xa = db * 2.d0 ** ib
    return
  end subroutine

  subroutine mp_eqjx (ja, xb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    intent (out):: ja
    intent (in):: xb
    type (mp_real) q1, q2
    integer mpnw
    mpnw = mpnwx
    db = xb
    call mpdmc (db, 0, q1%mpr)
    call mpinfr (q1%mpr, ja%mpi, q2%mpr, mpnw) 
    return
  end subroutine

  subroutine mp_eqja (ja, ab)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    character*(*), intent (in):: ab
    intent (out):: ja
    character*1 az(mpipl+100)
    type (mp_real) q1, q2
    integer mpnw
    mpnw = mpnwx
    l = len (ab)
    do i = 1, l
      az(i) = ab(i:i)
    enddo
    call mpdexc (az, l, q1%mpr, mpnw) 
    call mpinfr (q1%mpr, ja%mpi, q2%mpr, mpnw) 
    return
  end subroutine

!  MPI add routines.

  function mp_addjj (ja, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_integer):: mp_addjj
    intent (in):: ja, jb
    type (mp_real) q1, q2
    integer mpnw
    mpnw = mpnwx
    call mpadd (ja%mpi, jb%mpi, q1%mpr, mpnw) 
    call mpinfr (q1%mpr, mp_addjj%mpi, q2%mpr, mpnw) 
    return
  end function

  function mp_addjq (ja, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_addjq
    intent (in):: ja, qb
    integer mpnw
    mpnw = mpnwx
    call mpadd (ja%mpi, qb%mpr, mp_addjq%mpr, mpnw) 
    return
  end function

  function mp_addjz (ja, zb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_addjz
    intent (in):: ja, zb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    call mpmzc (ja%mpi, z1%mpc)
    call mpcadd (mp4, z1%mpc, zb%mpc, mp_addjz%mpc, mpnw) 
    return
  end function

  function mp_addij (ia, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_integer):: mp_addij
    intent (in):: ia, jb
    type (mp_real) q1, q2, q3
    integer mpnw
    mpnw = mpnwx
    da = ia
    call mpdmc (da, 0, q1%mpr)
    call mpadd (q1%mpr, jb%mpi, q2%mpr, mpnw) 
    call mpinfr (q2%mpr, mp_addij%mpi, q3%mpr, mpnw) 
    return
  end function

  function mp_addji (ja, ib)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_integer):: mp_addji
    intent (in):: ja, ib
    type (mp_real) q1, q2, q3
    integer mpnw
    mpnw = mpnwx
    db = ib
    call mpdmc (db, 0, q1%mpr)
    call mpadd (ja%mpi, q1%mpr, q2%mpr, mpnw) 
    call mpinfr (q2%mpr, mp_addji%mpi, q3%mpr, mpnw) 
    return
  end function

  function mp_adddj (da, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_adddj
    intent (in):: da, jb
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpdmc (da, 0, q1%mpr)
    call mpadd (q1%mpr, jb%mpi, mp_adddj%mpr, mpnw) 
    return
  end function

  function mp_addjd (ja, db)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_addjd
    intent (in):: ja, db
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpdmc (db, 0, q1%mpr)
    call mpadd (ja%mpi, q1%mpr, mp_addjd%mpr, mpnw) 
    return
  end function

  function mp_addxj (xa, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_addxj
    intent (in):: xa, jb
    type (mp_complex) z1, z2
    integer mpnw
    mpnw = mpnwx
    call mpxzc (xa, z1%mpc)
    call mpmzc (jb%mpi, z2%mpc)
    call mpcadd (mp4, z1%mpc, z2%mpc, mp_addxj%mpc, mpnw) 
    return
  end function

  function mp_addjx (ja, xb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_addjx
    intent (in):: ja, xb
    type (mp_complex) z1, z2
    integer mpnw
    mpnw = mpnwx
    call mpmzc (ja%mpi, z1%mpc)
    call mpxzc (xb, z2%mpc)
    call mpcadd (mp4, z1%mpc, z2%mpc, mp_addjx%mpc, mpnw) 
    return
  end function

!  MPI subtract routines.

  function mp_subjj (ja, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_integer):: mp_subjj
    intent (in):: ja, jb
    type (mp_real) q1, q2
    integer mpnw
    mpnw = mpnwx
    call mpsub (ja%mpi, jb%mpi, q1%mpr, mpnw) 
    call mpinfr (q1%mpr, mp_subjj%mpi, q2%mpr, mpnw) 
    return
  end function

  function mp_subjq (ja, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_subjq
    intent (in):: ja, qb
    integer mpnw
    mpnw = mpnwx
    call mpsub (ja%mpi, qb%mpr, mp_subjq%mpr, mpnw) 
    return
  end function

  function mp_subjz (ja, zb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_subjz
    intent (in):: ja, zb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    call mpmzc (ja%mpi, z1%mpc)
    call mpcsub (mp4, z1%mpc, zb%mpc, mp_subjz%mpc, mpnw) 
    return
  end function

  function mp_subij (ia, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_integer):: mp_subij
    intent (in):: ia, jb
    type (mp_real) q1, q2, q3
    integer mpnw
    mpnw = mpnwx
    da = ia
    call mpdmc (da, 0, q1%mpr)
    call mpsub (q1%mpr, jb%mpi, q2%mpr, mpnw) 
    call mpinfr (q2%mpr, mp_subij%mpi, q3%mpr, mpnw) 
    return
  end function

  function mp_subji (ja, ib)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_integer):: mp_subji
    intent (in):: ja, ib
    type (mp_real) q1, q2, q3
    integer mpnw
    mpnw = mpnwx
    db = ib
    call mpdmc (db, 0, q1%mpr)
    call mpsub (ja%mpi, q1%mpr, q2%mpr, mpnw) 
    call mpinfr (q2%mpr, mp_subji%mpi, q3%mpr, mpnw) 
    return
  end function

  function mp_subdj (da, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_subdj
    intent (in):: da, jb
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpdmc (da, 0, q1%mpr)
    call mpsub (q1%mpr, jb%mpi, mp_subdj%mpr, mpnw) 
    return
  end function

  function mp_subjd (ja, db)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_subjd
    intent (in):: ja, db
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpdmc (db, 0, q1%mpr)
    call mpsub (ja%mpi, q1%mpr, mp_subjd%mpr, mpnw) 
    return
  end function

  function mp_subxj (xa, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_subxj
    intent (in):: xa, jb
    type (mp_complex) z1, z2
    integer mpnw
    mpnw = mpnwx
    call mpxzc (xa, z1%mpc)
    call mpmzc (jb%mpi, z2%mpc)
    call mpcsub (mp4, z1%mpc, z2%mpc, mp_subxj%mpc, mpnw) 
    return
  end function

  function mp_subjx (ja, xb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_subjx
    intent (in):: ja, xb
    type (mp_complex) z1, z2
    integer mpnw
    mpnw = mpnwx
    call mpmzc (ja%mpi, z1%mpc)
    call mpxzc (xb, z2%mpc)
    call mpcsub (mp4, z1%mpc, z2%mpc, mp_subjx%mpc, mpnw) 
    return
  end function

!  MPI negation routine.

  function mp_negj (ja)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_integer):: mp_negj
    intent (in):: ja
    integer mpnw
    mpnw = mpnwx
    call mpeq (ja%mpi, mp_negj%mpi, mpnw) 
    mp_negj%mpi(1) = - ja%mpi(1)
    return
  end function

!  MPI multiply routines.

  function mp_muljj (ja, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_integer):: mp_muljj
    intent (in):: ja, jb
    type (mp_real) q1, q2
    integer mpnw
    mpnw = mpnwx
    call mpmul (ja%mpi, jb%mpi, q1%mpr, mpnw) 
    call mpinfr (q1%mpr, mp_muljj%mpi, q2%mpr, mpnw) 
    return
  end function

  function mp_muljq (ja, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_muljq
    intent (in):: ja, qb
    integer mpnw
    mpnw = mpnwx
    call mpmul (ja%mpi, qb%mpr, mp_muljq%mpr, mpnw) 
    return
  end function

  function mp_muljz (ja, zb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_muljz
    intent (in):: ja, zb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    call mpmzc (ja%mpi, z1%mpc)
    call mpcmul (mp4, z1%mpc, zb%mpc, mp_muljz%mpc, mpnw) 
    return
  end function

  function mp_mulij (ia, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_integer):: mp_mulij
    intent (in):: ia, jb
    type (mp_real) q1, q2
    integer mpnw
    mpnw = mpnwx
    da = ia
    call mpmuld (jb%mpi, da, 0, q1%mpr, mpnw) 
    call mpinfr (q1%mpr, mp_mulij%mpi, q2%mpr, mpnw) 
    return
  end function

  function mp_mulji (ja, ib)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_integer):: mp_mulji
    intent (in):: ja, ib
    type (mp_real) q1, q2
    integer mpnw
    mpnw = mpnwx
    db = ib
    call mpmuld (ja%mpi, db, 0, q1%mpr, mpnw) 
    call mpinfr (q1%mpr, mp_mulji%mpi, q2%mpr, mpnw) 
    return
  end function

  function mp_muldj (da, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_muldj
    intent (in):: da, jb
    integer mpnw
    mpnw = mpnwx
    call mpmuld (jb%mpi, da, 0, mp_muldj%mpr, mpnw) 
    return
  end function

  function mp_muljd (ja, db)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_muljd
    intent (in):: ja, db
    integer mpnw
    mpnw = mpnwx
    call mpmuld (ja%mpi, db, 0, mp_muljd%mpr, mpnw) 
    return
  end function

  function mp_mulxj (xa, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_mulxj
    intent (in):: xa, jb
    type (mp_complex) z1, z2
    integer mpnw
    mpnw = mpnwx
    call mpxzc (xa, z1%mpc)
    call mpmzc (jb%mpi, z2%mpc)
    call mpcmul (mp4, z1%mpc, z2%mpc, mp_mulxj%mpc, mpnw) 
    return
  end function

  function mp_muljx (ja, xb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_muljx
    intent (in):: ja, xb
    type (mp_complex) z1, z2
    integer mpnw
    mpnw = mpnwx
    call mpmzc (ja%mpi, z1%mpc)
    call mpxzc (xb, z2%mpc)
    call mpcmul (mp4, z1%mpc, z2%mpc, mp_muljx%mpc, mpnw) 
    return
  end function

!  MPI divide routines.

  function mp_divjj (ja, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_integer):: mp_divjj
    intent (in):: ja, jb
    type (mp_real) q1, q2
    integer mpnw
    mpnw = mpnwx
    call mpdiv (ja%mpi, jb%mpi, q1%mpr, mpnw) 
    call mpinfr (q1%mpr, mp_divjj%mpi, q2%mpr, mpnw) 
    return
  end function

  function mp_divjq (ja, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_divjq
    intent (in):: ja, qb
    integer mpnw
    mpnw = mpnwx
    call mpdiv (ja%mpi, qb%mpr, mp_divjq%mpr, mpnw) 
    return
  end function

  function mp_divjz (ja, zb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_divjz
    intent (in):: ja, zb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    call mpmzc (ja%mpi, z1%mpc)
    call mpcdiv (mp4, z1%mpc, zb%mpc, mp_divjz%mpc, mpnw) 
    return
  end function

  function mp_divij (ia, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_integer):: mp_divij
    intent (in):: ia, jb
    type (mp_real) q1, q2, q3
    integer mpnw
    mpnw = mpnwx
    da = ia
    call mpdmc (da, 0, q1%mpr)
    call mpdiv (q1%mpr, jb%mpi, q2%mpr, mpnw) 
    call mpinfr (q2%mpr, mp_divij%mpi, q3%mpr, mpnw) 
    return
  end function

  function mp_divji (ja, ib)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_integer):: mp_divji
    intent (in):: ja, ib
    type (mp_real) q1, q2
    integer mpnw
    mpnw = mpnwx
    db = ib
    call mpdivd (ja%mpi, db, 0, q1%mpr, mpnw) 
    call mpinfr (q1%mpr, mp_divji%mpi, q2%mpr, mpnw) 
    return
  end function

  function mp_divdj (da, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_divdj
    intent (in):: da, jb
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpdmc (da, 0, q1%mpr)
    call mpdiv (q1%mpr, jb%mpi, mp_divdj%mpr, mpnw) 
    return
  end function

  function mp_divjd (ja, db)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_divjd
    intent (in):: ja, db
    integer mpnw
    mpnw = mpnwx
    call mpdivd (ja%mpi, db, 0, mp_divjd%mpr, mpnw) 
    return
  end function

  function mp_divxj (xa, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_divxj
    intent (in):: xa, jb
    type (mp_complex) z1, z2
    integer mpnw
    mpnw = mpnwx
    call mpxzc (xa, z1%mpc)
    call mpmzc (jb%mpi, z2%mpc)
    call mpcdiv (mp4, z1%mpc, z2%mpc, mp_divxj%mpc, mpnw) 
    return
  end function

  function mp_divjx (ja, xb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_divjx
    intent (in):: ja, xb
    type (mp_complex) z1, z2
    integer mpnw
    mpnw = mpnwx
    call mpmzc (ja%mpi, z1%mpc)
    call mpxzc (xb, z2%mpc)
    call mpcdiv (mp4, z1%mpc, z2%mpc, mp_divjx%mpc, mpnw) 
    return
  end function

!  MPI exponentiation routines.

  function mp_expjj (ja, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_integer):: mp_expjj
    intent (in):: ja, jb
    type (mp_real) q1, q2
    integer mpnw
    mpnw = mpnwx
    call mplog (ja%mpi, mpl02%mpr, q1%mpr, mpnw) 
    call mpmul (q1%mpr, jb%mpi, q2%mpr, mpnw) 
    call mpexp (q2%mpr, mpl02%mpr, q1%mpr, mpnw) 
    call mpnint (q1%mpr, mp_expjj%mpi, mpnw) 
    return
  end function

  function mp_expjq (ja, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_expjq
    intent (in):: ja, qb
    type (mp_real) q1, q2
    integer mpnw
    mpnw = mpnwx
    call mplog (ja%mpi, mpl02%mpr, q1%mpr, mpnw) 
    call mpmul (q1%mpr, qb%mpr, q2%mpr, mpnw) 
    call mpexp (q2%mpr, mpl02%mpr, mp_expjq%mpr, mpnw) 
    return
  end function

  function mp_expij (ia, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_integer):: mp_expij
    intent (in):: ia, jb
    type (mp_real) q1, q2, q3
    integer mpnw
    mpnw = mpnwx
    da = ia
    call mpdmc (da, 0, q1%mpr)
    call mplog (q1%mpr, mpl02%mpr, q2%mpr, mpnw) 
    call mpmul (q2%mpr, jb%mpi, q3%mpr, mpnw) 
    call mpexp (q3%mpr, mpl02%mpr, q1%mpr, mpnw) 
    call mpnint (q1%mpr, mp_expij%mpi, mpnw) 
    return
  end function

  function mp_expji (ja, ib)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_integer):: mp_expji
    intent (in):: ja, ib
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpnpwr (ja%mpi, ib, q1%mpr, mpnw) 
    call mpnint (q1%mpr, mp_expji%mpi, mpnw) 
    return
  end function

  function mp_expdj (da, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_expdj
    intent (in):: da, jb
    type (mp_real) q1, q2, q3
    integer mpnw
    mpnw = mpnwx
    call mpdmc (da, 0, q1%mpr)
    call mplog (q1%mpr, mpl02%mpr, q2%mpr, mpnw) 
    call mpmul (q2%mpr, jb%mpi, q3%mpr, mpnw) 
    call mpexp (q3%mpr, mpl02%mpr, mp_expdj%mpr, mpnw) 
    return
    end function

  function mp_expjd (ja, db)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_expjd
    intent (in):: ja, db
    type (mp_real) q1, q2
    integer mpnw
    mpnw = mpnwx
    call mplog (ja%mpi, mpl02%mpr, q1%mpr, mpnw) 
    call mpmuld (q1%mpr, db, 0, q2%mpr, mpnw) 
    call mpexp (q2%mpr, mpl02%mpr, mp_expjd%mpr, mpnw) 
    return
  end function

!  MPI .EQ. routines.

  function mp_eqtjj (ja, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_eqtjj
    intent (in):: ja, jb
    integer mpnw
    mpnw = mpnwx
    call mpcpr (ja%mpi, jb%mpi, ic, mpnw) 
    if (ic .eq. 0) then
      mp_eqtjj = .true.
    else
      mp_eqtjj = .false.
    endif
    return
  end function

  function mp_eqtjq (ja, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_eqtjq
    intent (in):: ja, qb
    integer mpnw
    mpnw = mpnwx
    call mpcpr (ja%mpi, qb%mpr, ic, mpnw) 
    if (ic .eq. 0) then
      mp_eqtjq = .true.
    else
      mp_eqtjq = .false.
    endif
    return
  end function

  function mp_eqtjz (ja, zb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_eqtjz
    intent (in):: ja, zb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    call mpmzc (ja%mpi, z1%mpc)
    call mpcpr (z1%mpc, zb%mpc, ic1, mpnw) 
    call mpcpr (z1%mpc(mp41), zb%mpc(mp41), ic2, mpnw) 
    if (ic1 .eq. 0 .and. ic2 .eq. 0) then
      mp_eqtjz = .true.
    else
      mp_eqtjz = .false.
    endif
    return
  end function

  function mp_eqtij (ia, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_eqtij
    intent (in):: ia, jb
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    da = ia
    call mpdmc (da, 0, q1%mpr)
    call mpcpr (q1%mpr, jb%mpi, ic, mpnw) 
    if (ic .eq. 0) then
      mp_eqtij = .true.
    else
      mp_eqtij = .false.
    endif
    return
  end function

  function mp_eqtji (ja, ib)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_eqtji
    intent (in):: ja, ib
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    db = ib
    call mpdmc (db, 0, q1%mpr)
    call mpcpr (ja%mpi, q1%mpr, ic, mpnw) 
    if (ic .eq. 0) then
      mp_eqtji = .true.
    else
      mp_eqtji = .false.
    endif
    return
  end function

  function mp_eqtdj (da, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_eqtdj
    intent (in):: da, jb
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpdmc (da, 0, q1%mpr)
    call mpcpr (q1%mpr, jb%mpi, ic, mpnw) 
    if (ic .eq. 0) then
      mp_eqtdj = .true.
    else
      mp_eqtdj = .false.
    endif
    return
  end function

  function mp_eqtjd (ja, db)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_eqtjd
    intent (in):: ja, db
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpdmc (db, 0, q1%mpr)
    call mpcpr (ja%mpi, q1%mpr, ic, mpnw) 
    if (ic .eq. 0) then
      mp_eqtjd = .true.
    else
      mp_eqtjd = .false.
    endif
    return
  end function

  function mp_eqtxj (xa, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_eqtxj
    intent (in):: xa, jb
    type (mp_complex) z1, z2
    integer mpnw
    mpnw = mpnwx
    call mpxzc (xa, z1%mpc)
    call mpmzc (jb%mpi, z2%mpc)
    call mpcpr (z1%mpc, z2%mpc, ic1, mpnw) 
    call mpcpr (z1%mpc(mp41), z2%mpc(mp41), ic2, mpnw) 
    if (ic1 .eq. 0 .and. ic2 .eq. 0) then
      mp_eqtxj = .true.
    else
      mp_eqtxj = .false.
    endif
    return
  end function

  function mp_eqtjx (ja, xb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_eqtjx
    intent (in):: ja, xb
    type (mp_complex) z1, z2
    integer mpnw
    mpnw = mpnwx
    call mpmzc (ja%mpi, z1%mpc)
    call mpxzc (xb, z2%mpc)
    call mpcpr (z1%mpc, z2%mpc, ic1, mpnw) 
    call mpcpr (z1%mpc(mp41), z2%mpc(mp41), ic2, mpnw) 
    if (ic1 .eq. 0 .and. ic2 .eq. 0) then
      mp_eqtjx = .true.
    else
      mp_eqtjx = .false.
    endif
    return
  end function

!  MPI .NE. routines.

  function mp_netjj (ja, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_netjj
    intent (in):: ja, jb
    integer mpnw
    mpnw = mpnwx
    call mpcpr (ja%mpi, jb%mpi, ic, mpnw) 
    if (ic .ne. 0) then
      mp_netjj = .true.
    else
      mp_netjj = .false.
    endif
    return
  end function

  function mp_netjq (ja, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_netjq
    intent (in):: ja, qb
    integer mpnw
    mpnw = mpnwx
    call mpcpr (ja%mpi, qb%mpr, ic, mpnw) 
    if (ic .ne. 0) then
      mp_netjq = .true.
    else
      mp_netjq = .false.
    endif
    return
  end function

  function mp_netjz (ja, zb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_netjz
    intent (in):: ja, zb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    call mpmzc (ja%mpi, z1%mpc)
    call mpcpr (z1%mpc, zb%mpc, ic1, mpnw) 
    call mpcpr (z1%mpc(mp41), zb%mpc(mp41), ic2, mpnw) 
    if (ic1 .ne. 0 .or. ic2 .ne. 0) then
      mp_netjz = .true.
    else
      mp_netjz = .false.
    endif
    return
  end function

  function mp_netij (ia, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_netij
    intent (in):: ia, jb
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    da = ia
    call mpdmc (da, 0, q1%mpr)
    call mpcpr (q1%mpr, jb%mpi, ic, mpnw) 
    if (ic .ne. 0) then
      mp_netij = .true.
    else
      mp_netij = .false.
    endif
    return
  end function

  function mp_netji (ja, ib)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_netji
    intent (in):: ja, ib
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    db = ib
    call mpdmc (db, 0, q1%mpr)
    call mpcpr (ja%mpi, q1%mpr, ic, mpnw) 
    if (ic .ne. 0) then
      mp_netji = .true.
    else
      mp_netji = .false.
    endif
    return
  end function

  function mp_netdj (da, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_netdj
    intent (in):: da, jb
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpdmc (da, 0, q1%mpr)
    call mpcpr (q1%mpr, jb%mpi, ic, mpnw) 
    if (ic .ne. 0) then
      mp_netdj = .true.
    else
      mp_netdj = .false.
    endif
    return
  end function

  function mp_netjd (ja, db)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_netjd
    intent (in):: ja, db
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpdmc (db, 0, q1%mpr)
    call mpcpr (ja%mpi, q1%mpr, ic, mpnw) 
    if (ic .ne. 0) then
      mp_netjd = .true.
    else
      mp_netjd = .false.
    endif
    return
  end function

  function mp_netxj (xa, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_netxj
    intent (in):: xa, jb
    type (mp_complex) z1, z2
    integer mpnw
    mpnw = mpnwx
    call mpxzc (xa, z1%mpc)
    call mpmzc (jb%mpi, z2%mpc)
    call mpcpr (z1%mpc, z2%mpc, ic1, mpnw) 
    call mpcpr (z1%mpc(mp41), z2%mpc(mp41), ic2, mpnw) 
    if (ic1 .ne. 0 .or. ic2 .ne. 0) then
      mp_netxj = .true.
    else
      mp_netxj = .false.
    endif
    return
  end function

  function mp_netjx (ja, xb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_netjx
    intent (in):: ja, xb
    type (mp_complex) z1, z2
    integer mpnw
    mpnw = mpnwx
    call mpmzc (ja%mpi, z1%mpc)
    call mpxzc (xb, z2%mpc)
    call mpcpr (z1%mpc, z2%mpc, ic1, mpnw) 
    call mpcpr (z1%mpc(mp41), z2%mpc(mp41), ic2, mpnw) 
    if (ic1 .ne. 0 .or. ic2 .ne. 0) then
      mp_netjx = .true.
    else
      mp_netjx = .false.
    endif
    return
  end function

!  MPI .LE. routines.

  function mp_letjj (ja, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_letjj
    intent (in):: ja, jb
    integer mpnw
    mpnw = mpnwx
    call mpcpr (ja%mpi, jb%mpi, ic, mpnw) 
    if (ic .le. 0) then
      mp_letjj = .true.
    else
      mp_letjj = .false.
    endif
    return
  end function

  function mp_letjq (ja, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_letjq
    intent (in):: ja, qb
    integer mpnw
    mpnw = mpnwx
    call mpcpr (ja%mpi, qb%mpr, ic, mpnw) 
    if (ic .le. 0) then
      mp_letjq = .true.
    else
      mp_letjq = .false.
    endif
    return
  end function

  function mp_letij (ia, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_letij
    intent (in):: ia, jb
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    da = ia
    call mpdmc (da, 0, q1%mpr)
    call mpcpr (q1%mpr, jb%mpi, ic, mpnw) 
    if (ic .le. 0) then
      mp_letij = .true.
    else
      mp_letij = .false.
    endif
    return
  end function

  function mp_letji (ja, ib)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_letji
    intent (in):: ja, ib
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    db = ib
    call mpdmc (db, 0, q1%mpr)
    call mpcpr (ja%mpi, q1%mpr, ic, mpnw) 
    if (ic .le. 0) then
      mp_letji = .true.
    else
      mp_letji = .false.
    endif
    return
  end function

  function mp_letdj (da, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_letdj
    intent (in):: da, jb
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpdmc (da, 0, q1%mpr)
    call mpcpr (q1%mpr, jb%mpi, ic, mpnw) 
    if (ic .le. 0) then
      mp_letdj = .true.
    else
      mp_letdj = .false.
    endif
    return
  end function

  function mp_letjd (ja, db)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_letjd
    intent (in):: ja, db
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpdmc (db, 0, q1%mpr)
    call mpcpr (ja%mpi, q1%mpr, ic, mpnw) 
    if (ic .le. 0) then
      mp_letjd = .true.
    else
      mp_letjd = .false.
    endif
    return
  end function

!  MPI .GE. routines.

  function mp_getjj (ja, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_getjj
    intent (in):: ja, jb
    integer mpnw
    mpnw = mpnwx
    call mpcpr (ja%mpi, jb%mpi, ic, mpnw) 
    if (ic .ge. 0) then
      mp_getjj = .true.
    else
      mp_getjj = .false.
    endif
    return
  end function

  function mp_getjq (ja, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_getjq
    intent (in):: ja, qb
    integer mpnw
    mpnw = mpnwx
    call mpcpr (ja%mpi, qb%mpr, ic, mpnw) 
    if (ic .ge. 0) then
      mp_getjq = .true.
    else
      mp_getjq = .false.
    endif
    return
  end function

  function mp_getij (ia, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_getij
    intent (in):: ia, jb
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    da = ia
    call mpdmc (da, 0, q1%mpr)
    call mpcpr (q1%mpr, jb%mpi, ic, mpnw) 
    if (ic .ge. 0) then
      mp_getij = .true.
    else
      mp_getij = .false.
    endif
    return
  end function

  function mp_getji (ja, ib)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_getji
    intent (in):: ja, ib
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    db = ib
    call mpdmc (db, 0, q1%mpr)
    call mpcpr (ja%mpi, q1%mpr, ic, mpnw) 
    if (ic .ge. 0) then
      mp_getji = .true.
    else
      mp_getji = .false.
    endif
    return
  end function

  function mp_getdj (da, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_getdj
    intent (in):: da, jb
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpdmc (da, 0, q1%mpr)
    call mpcpr (q1%mpr, jb%mpi, ic, mpnw) 
    if (ic .ge. 0) then
      mp_getdj = .true.
    else
      mp_getdj = .false.
    endif
    return
  end function

  function mp_getjd (ja, db)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_getjd
    intent (in):: ja, db
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpdmc (db, 0, q1%mpr)
    call mpcpr (ja%mpi, q1%mpr, ic, mpnw) 
    if (ic .ge. 0) then
      mp_getjd = .true.
    else
      mp_getjd = .false.
    endif
    return
  end function

!  MPI .LT. routines.

  function mp_lttjj (ja, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_lttjj
    intent (in):: ja, jb
    integer mpnw
    mpnw = mpnwx
    call mpcpr (ja%mpi, jb%mpi, ic, mpnw) 
    if (ic .lt. 0) then
      mp_lttjj = .true.
    else
      mp_lttjj = .false.
    endif
    return
  end function

  function mp_lttjq (ja, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_lttjq
    intent (in):: ja, qb
    integer mpnw
    mpnw = mpnwx
    call mpcpr (ja%mpi, qb%mpr, ic, mpnw) 
    if (ic .lt. 0) then
      mp_lttjq = .true.
    else
      mp_lttjq = .false.
    endif
    return
  end function

  function mp_lttij (ia, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_lttij
    intent (in):: ia, jb
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    da = ia
    call mpdmc (da, 0, q1%mpr)
    call mpcpr (q1%mpr, jb%mpi, ic, mpnw) 
    if (ic .lt. 0) then
      mp_lttij = .true.
    else
      mp_lttij = .false.
    endif
    return
  end function

  function mp_lttji (ja, ib)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_lttji
    intent (in):: ja, ib
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    db = ib
    call mpdmc (db, 0, q1%mpr)
    call mpcpr (ja%mpi, q1%mpr, ic, mpnw) 
    if (ic .lt. 0) then
      mp_lttji = .true.
    else
      mp_lttji = .false.
    endif
    return
  end function

  function mp_lttdj (da, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_lttdj
    intent (in):: da, jb
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpdmc (da, 0, q1%mpr)
    call mpcpr (q1%mpr, jb%mpi, ic, mpnw) 
    if (ic .lt. 0) then
      mp_lttdj = .true.
    else
      mp_lttdj = .false.
    endif
    return
  end function

  function mp_lttjd (ja, db)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_lttjd
    intent (in):: ja, db
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpdmc (db, 0, q1%mpr)
    call mpcpr (ja%mpi, q1%mpr, ic, mpnw) 
    if (ic .lt. 0) then
      mp_lttjd = .true.
    else
      mp_lttjd = .false.
    endif
    return
  end function

!  MPI .GT. routines.

  function mp_gttjj (ja, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_gttjj
    intent (in):: ja, jb
    integer mpnw
    mpnw = mpnwx
    call mpcpr (ja%mpi, jb%mpi, ic, mpnw) 
    if (ic .gt. 0) then
      mp_gttjj = .true.
    else
      mp_gttjj = .false.
    endif
    return
  end function

  function mp_gttjq (ja, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_gttjq
    intent (in):: ja, qb
    integer mpnw
    mpnw = mpnwx
    call mpcpr (ja%mpi, qb%mpr, ic, mpnw) 
    if (ic .gt. 0) then
      mp_gttjq = .true.
    else
      mp_gttjq = .false.
    endif
    return
  end function

  function mp_gttij (ia, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_gttij
    intent (in):: ia, jb
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    da = ia
    call mpdmc (da, 0, q1%mpr)
    call mpcpr (q1%mpr, jb%mpi, ic, mpnw) 
    if (ic .gt. 0) then
      mp_gttij = .true.
    else
      mp_gttij = .false.
    endif
    return
  end function

  function mp_gttji (ja, ib)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_gttji
    intent (in):: ja, ib
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    db = ib
    call mpdmc (db, 0, q1%mpr)
    call mpcpr (ja%mpi, q1%mpr, ic, mpnw) 
    if (ic .gt. 0) then
      mp_gttji = .true.
    else
      mp_gttji = .false.
    endif
    return
  end function

  function mp_gttdj (da, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_gttdj
    intent (in):: da, jb
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpdmc (da, 0, q1%mpr)
    call mpcpr (q1%mpr, jb%mpi, ic, mpnw) 
    if (ic .gt. 0) then
      mp_gttdj = .true.
    else
      mp_gttdj = .false.
    endif
    return
  end function

  function mp_gttjd (ja, db)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_gttjd
    intent (in):: ja, db
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpdmc (db, 0, q1%mpr)
    call mpcpr (ja%mpi, q1%mpr, ic, mpnw) 
    if (ic .gt. 0) then
      mp_gttjd = .true.
    else
      mp_gttjd = .false.
    endif
    return
  end function

end module


module mprealmod

!  This Fortran-90 module defines operator extensions involving the
!  MP_REAL datatype.  For operations involving two MP data types,
!  those whose first argument is MP_REAL are included here.
!  Others are handled in other modules.

!  The subroutines and functions defined in this module are private
!  and not intended to be called directly by the user.

use mpfunmod
use mpdefmod
private kdb, mp4, mp24, mp41
parameter (kdb = kind (0.d0), mp4 = mpwds + 4, mp24 = 2 * mp4, mp41 = mp4 + 1)
private &
  mp_eqqj, mp_eqqq, mp_eqqz, mp_eqiq, mp_eqqi, &
  mp_eqdq, mp_eqqd, mp_eqxq, mp_eqqx, mp_eqqa, &
  mp_addqj, mp_addqq, mp_addqz, mp_addiq, mp_addqi, &
  mp_adddq, mp_addqd, mp_addxq, mp_addqx, &
  mp_subqj, mp_subqq, mp_subqz, mp_subiq, mp_subqi, &
  mp_subdq, mp_subqd, mp_subxq, mp_subqx, mp_negq, &
  mp_mulqj, mp_mulqq, mp_mulqz, mp_muliq, mp_mulqi, &
  mp_muldq, mp_mulqd, mp_mulxq, mp_mulqx, &
  mp_divqj, mp_divqq, mp_divqz, mp_diviq, mp_divqi, &
  mp_divdq, mp_divqd, mp_divxq, mp_divqx, &
  mp_expqj, mp_expqq, mp_expiq, mp_expqi, mp_expdq, mp_expqd, &
  mp_eqtqj, mp_eqtqq, mp_eqtqz, mp_eqtiq, mp_eqtqi, &
  mp_eqtdq, mp_eqtqd, mp_eqtxq, mp_eqtqx, &
  mp_netqj, mp_netqq, mp_netqz, mp_netiq, mp_netqi, &
  mp_netdq, mp_netqd, mp_netxq, mp_netqx, &
  mp_letqj, mp_letqq, mp_letiq, mp_letqi, mp_letdq, mp_letqd, &
  mp_getqj, mp_getqq, mp_getiq, mp_getqi, mp_getdq, mp_getqd, &
  mp_lttqj, mp_lttqq, mp_lttiq, mp_lttqi, mp_lttdq, mp_lttqd, &
  mp_gttqj, mp_gttqq, mp_gttiq, mp_gttqi, mp_gttdq, mp_gttqd

!  MPR operator extension interface blocks.

interface assignment (=)
  module procedure mp_eqqj
  module procedure mp_eqqq
  module procedure mp_eqqz
  module procedure mp_eqiq
  module procedure mp_eqqi
  module procedure mp_eqdq
  module procedure mp_eqqd
  module procedure mp_eqxq
  module procedure mp_eqqx

  module procedure mp_eqqa
end interface

interface operator (+)
  module procedure mp_addqj
  module procedure mp_addqq
  module procedure mp_addqz
  module procedure mp_addiq
  module procedure mp_addqi
  module procedure mp_adddq
  module procedure mp_addqd
  module procedure mp_addxq
  module procedure mp_addqx
end interface

interface operator (-)
  module procedure mp_subqj
  module procedure mp_subqq
  module procedure mp_subqz
  module procedure mp_subiq
  module procedure mp_subqi
  module procedure mp_subdq
  module procedure mp_subqd
  module procedure mp_subxq
  module procedure mp_subqx

  module procedure mp_negq
end interface

interface operator (*)
  module procedure mp_mulqj
  module procedure mp_mulqq
  module procedure mp_mulqz
  module procedure mp_muliq
  module procedure mp_mulqi
  module procedure mp_muldq
  module procedure mp_mulqd
  module procedure mp_mulxq
  module procedure mp_mulqx
end interface

interface operator (/)
  module procedure mp_divqj
  module procedure mp_divqq
  module procedure mp_divqz
  module procedure mp_diviq
  module procedure mp_divqi
  module procedure mp_divdq
  module procedure mp_divqd
  module procedure mp_divxq
  module procedure mp_divqx
end interface

interface operator (**)
  module procedure mp_expqj
  module procedure mp_expqq
  module procedure mp_expiq
  module procedure mp_expqi
  module procedure mp_expdq
  module procedure mp_expqd
end interface

interface operator (.eq.)
  module procedure mp_eqtqj
  module procedure mp_eqtqq
  module procedure mp_eqtqz
  module procedure mp_eqtiq
  module procedure mp_eqtqi
  module procedure mp_eqtdq
  module procedure mp_eqtqd
  module procedure mp_eqtxq
  module procedure mp_eqtqx
end interface

interface operator (.ne.)
  module procedure mp_netqj
  module procedure mp_netqq
  module procedure mp_netqz
  module procedure mp_netiq
  module procedure mp_netqi
  module procedure mp_netdq
  module procedure mp_netqd
  module procedure mp_netxq
  module procedure mp_netqx
end interface

interface operator (.le.)
  module procedure mp_letqj
  module procedure mp_letqq
  module procedure mp_letiq
  module procedure mp_letqi
  module procedure mp_letdq
  module procedure mp_letqd
end interface

interface operator (.ge.)
  module procedure mp_getqj
  module procedure mp_getqq
  module procedure mp_getiq
  module procedure mp_getqi
  module procedure mp_getdq
  module procedure mp_getqd
end interface

interface operator (.lt.)
  module procedure mp_lttqj
  module procedure mp_lttqq
  module procedure mp_lttiq
  module procedure mp_lttqi
  module procedure mp_lttdq
  module procedure mp_lttqd
end interface

interface operator (.gt.)
  module procedure mp_gttqj
  module procedure mp_gttqq
  module procedure mp_gttiq
  module procedure mp_gttqi
  module procedure mp_gttdq
  module procedure mp_gttqd
end interface

contains

!  MPR assignment routines.

  subroutine mp_eqqj (qa, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    intent (out):: qa
    intent (in):: jb
    integer mpnw
    mpnw = mpnwx
    call mpeq (jb%mpi, qa%mpr, mpnw) 
    return
  end subroutine

  subroutine mp_eqqq (qa, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    intent (out):: qa
    intent (in):: qb
    integer mpnw
    mpnw = mpnwx
    call mpeq (qb%mpr, qa%mpr, mpnw) 
    return
  end subroutine

  subroutine mp_eqqz (qa, zb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    intent (out):: qa
    intent (in):: zb
    integer mpnw
    mpnw = mpnwx
    call mpeq (zb%mpc, qa%mpr, mpnw) 
    return
  end subroutine

  subroutine mp_eqiq (ia, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    intent (out):: ia
    intent (in):: qb
    integer mpnw
    mpnw = mpnwx
    call mpmdc (qb%mpr, db, ib)
    ia = db * 2.d0 ** ib
    return
  end subroutine

  subroutine mp_eqqi (qa, ib)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    intent (out):: qa
    intent (in):: ib
    integer mpnw
    mpnw = mpnwx
    db = ib
    call mpdmc (db, 0, qa%mpr)
    return
  end subroutine

  subroutine mp_eqdq (da, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    intent (out):: da
    intent (in):: qb
    integer mpnw
    mpnw = mpnwx
    call mpmdc (qb%mpr, db, ib)
    da = db * 2.d0 ** ib
    return
  end subroutine

  subroutine mp_eqqd (qa, db)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    intent (out):: qa
    intent (in):: db
    integer mpnw
    mpnw = mpnwx
    call mpdmc (db, 0, qa%mpr)
    return
  end subroutine

  subroutine mp_eqxq (xa, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    intent (out):: xa
    intent (in):: qb
    integer mpnw
    mpnw = mpnwx
    call mpmdc (qb%mpr, db, ib)
    xa = db * 2.d0 ** ib
    return
  end subroutine

  subroutine mp_eqqx (qa, xb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    intent (out):: qa
    intent (in):: xb
    integer mpnw
    mpnw = mpnwx
    db = xb
    call mpdmc (db, 0, qa%mpr)
    return
  end subroutine

  subroutine mp_eqqa (qa, ab)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    character*(*), intent (in):: ab
    intent (out):: qa
    character*1 az(mpipl+100)
    integer mpnw
    mpnw = mpnwx
    l = len (ab)
    do i = 1, l
      az(i) = ab(i:i)
    enddo
    call mpdexc (az, l, qa%mpr, mpnw) 
    return
  end subroutine

!  MPR add routines.

  function mp_addqj (qa, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_addqj
    intent (in):: qa, jb
    integer mpnw
    mpnw = mpnwx
    call mpadd (qa%mpr, jb%mpi, mp_addqj%mpr, mpnw) 
    return
  end function

  function mp_addqq (qa, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_addqq
    intent (in):: qa, qb
    integer mpnw
    mpnw = mpnwx
    call mpadd (qa%mpr, qb%mpr, mp_addqq%mpr, mpnw) 
    return
  end function

  function mp_addqz (qa, zb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_addqz
    intent (in):: qa, zb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    call mpmzc (qa%mpr, z1%mpc)
    call mpcadd (mp4, z1%mpc, zb%mpc, mp_addqz%mpc, mpnw) 
    return
  end function

  function mp_addiq (ia, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_addiq
    intent (in):: ia, qb
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    da = ia
    call mpdmc (da, 0, q1%mpr)
    call mpadd (q1%mpr, qb%mpr, mp_addiq%mpr, mpnw) 
    return
  end function

  function mp_addqi (qa, ib)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_addqi
    intent (in):: qa, ib
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    db = ib
    call mpdmc (db, 0, q1%mpr)
    call mpadd (qa%mpr, q1%mpr, mp_addqi%mpr, mpnw) 
    return
  end function

  function mp_adddq (da, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_adddq
    intent (in):: da, qb
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpdmc (da, 0, q1%mpr)
    call mpadd (q1%mpr, qb%mpr, mp_adddq%mpr, mpnw) 
    return
  end function

  function mp_addqd (qa, db)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_addqd
    intent (in):: qa, db
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpdmc (db, 0, q1%mpr)
    call mpadd (qa%mpr, q1%mpr, mp_addqd%mpr, mpnw) 
    return
  end function

  function mp_addxq (xa, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_addxq
    intent (in):: xa, qb
    type (mp_complex) z1, z2
    integer mpnw
    mpnw = mpnwx
    call mpxzc (xa, z1%mpc)
    call mpmzc (qb%mpr, z2%mpc)
    call mpcadd (mp4, z1%mpc, z2%mpc, mp_addxq%mpc, mpnw) 
    return
  end function

  function mp_addqx (qa, xb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_addqx
    intent (in):: qa, xb
    type (mp_complex) z1, z2
    integer mpnw
    mpnw = mpnwx
    call mpmzc (qa%mpr, z1%mpc)
    call mpxzc (xb, z2%mpc)
    call mpcadd (mp4, z1%mpc, z2%mpc, mp_addqx%mpc, mpnw) 
    return
  end function

!  MPR subtract routines.

  function mp_subqj (qa, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_subqj
    intent (in):: qa, jb
    integer mpnw
    mpnw = mpnwx
    call mpsub (qa%mpr, jb%mpi, mp_subqj%mpr, mpnw) 
    return
  end function

  function mp_subqq (qa, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_subqq
    intent (in):: qa, qb
    integer mpnw
    mpnw = mpnwx
    call mpsub (qa%mpr, qb%mpr, mp_subqq%mpr, mpnw) 
    return
  end function

  function mp_subqz (qa, zb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_subqz
    intent (in):: qa, zb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    call mpmzc (qa%mpr, z1%mpc)
    call mpcsub (mp4, z1%mpc, zb%mpc, mp_subqz%mpc, mpnw) 
    return
  end function

  function mp_subiq (ia, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_subiq
    intent (in):: ia, qb
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    da = ia
    call mpdmc (da, 0, q1%mpr)
    call mpsub (q1%mpr, qb%mpr, mp_subiq%mpr, mpnw) 
    return
  end function

  function mp_subqi (qa, ib)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_subqi
    intent (in):: qa, ib
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    db = ib
    call mpdmc (db, 0, q1%mpr)
    call mpsub (qa%mpr, q1%mpr, mp_subqi%mpr, mpnw) 
    return
  end function

  function mp_subdq (da, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_subdq
    intent (in):: da, qb
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpdmc (da, 0, q1%mpr)
    call mpsub (q1%mpr, qb%mpr, mp_subdq%mpr, mpnw) 
    return
  end function

  function mp_subqd (qa, db)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_subqd
    intent (in):: qa, db
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpdmc (db, 0, q1%mpr)
    call mpsub (qa%mpr, q1%mpr, mp_subqd%mpr, mpnw) 
    return
  end function

  function mp_subxq (xa, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_subxq
    intent (in):: xa, qb
    type (mp_complex) z1, z2
    integer mpnw
    mpnw = mpnwx
    call mpxzc (xa, z1%mpc)
    call mpmzc (qb%mpr, z2%mpc)
    call mpcsub (mp4, z1%mpc, z2%mpc, mp_subxq%mpc, mpnw) 
    return
  end function

  function mp_subqx (qa, xb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_subqx
    intent (in):: qa, xb
    type (mp_complex) z1, z2
    integer mpnw
    mpnw = mpnwx
    call mpmzc (qa%mpr, z1%mpc)
    call mpxzc (xb, z2%mpc)
    call mpcsub (mp4, z1%mpc, z2%mpc, mp_subqx%mpc, mpnw) 
    return
  end function

!  MPR negation routine.

  function mp_negq (qa)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_negq
    intent (in):: qa
    integer mpnw
    mpnw = mpnwx
    call mpeq (qa%mpr, mp_negq%mpr, mpnw) 
    mp_negq%mpr(1) = - qa%mpr(1)
    return
  end function

!  MPR multiply routines.

  function mp_mulqj (qa, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_mulqj
    intent (in):: qa, jb
    integer mpnw
    mpnw = mpnwx
    call mpmul (qa%mpr, jb%mpi, mp_mulqj%mpr, mpnw) 
    return
  end function

  function mp_mulqq (qa, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_mulqq
    intent (in):: qa, qb
    integer mpnw
    mpnw = mpnwx
    call mpmul (qa%mpr, qb%mpr, mp_mulqq%mpr, mpnw) 
    return
  end function

  function mp_mulqz (qa, zb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_mulqz
    intent (in):: qa, zb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    call mpmzc (qa%mpr, z1%mpc)
    call mpcmul (mp4, z1%mpc, zb%mpc, mp_mulqz%mpc, mpnw) 
    return
  end function

  function mp_muliq (ia, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_muliq
    intent (in):: ia, qb
    integer mpnw
    mpnw = mpnwx
    da = ia
    call mpmuld (qb%mpr, da, 0, mp_muliq%mpr, mpnw) 
    return
  end function

  function mp_mulqi (qa, ib)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_mulqi
    intent (in):: qa, ib
    integer mpnw
    mpnw = mpnwx
    db = ib
    call mpmuld (qa%mpr, db, 0, mp_mulqi%mpr, mpnw) 
    return
  end function

  function mp_muldq (da, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_muldq
    intent (in):: da, qb
    integer mpnw
    mpnw = mpnwx
    call mpmuld (qb%mpr, da, 0, mp_muldq%mpr, mpnw) 
    return
  end function

  function mp_mulqd (qa, db)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_mulqd
    intent (in):: qa, db
    integer mpnw
    mpnw = mpnwx
    call mpmuld (qa%mpr, db, 0, mp_mulqd%mpr, mpnw) 
    return
  end function

  function mp_mulxq (xa, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_mulxq
    intent (in):: xa, qb
    type (mp_complex) z1, z2
    integer mpnw
    mpnw = mpnwx
    call mpxzc (xa, z1%mpc)
    call mpmzc (qb%mpr, z2%mpc)
    call mpcmul (mp4, z1%mpc, z2%mpc, mp_mulxq%mpc, mpnw) 
    return
  end function

  function mp_mulqx (qa, xb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_mulqx
    intent (in):: qa, xb
    type (mp_complex) z1, z2
    integer mpnw
    mpnw = mpnwx
    call mpmzc (qa%mpr, z1%mpc)
    call mpxzc (xb, z2%mpc)
    call mpcmul (mp4, z1%mpc, z2%mpc, mp_mulqx%mpc, mpnw) 
    return
  end function

!  MPR divide routines.

  function mp_divqj (qa, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_divqj
    intent (in):: qa, jb
    integer mpnw
    mpnw = mpnwx
    call mpdiv (qa%mpr, jb%mpi, mp_divqj%mpr, mpnw) 
    return
  end function

  function mp_divqq (qa, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_divqq
    intent (in):: qa, qb
    integer mpnw
    mpnw = mpnwx
    call mpdiv (qa%mpr, qb%mpr, mp_divqq%mpr, mpnw) 
    return
  end function

  function mp_divqz (qa, zb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_divqz
    intent (in):: qa, zb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    call mpmzc (qa%mpr, z1%mpc)
    call mpcdiv (mp4, z1%mpc, zb%mpc, mp_divqz%mpc, mpnw) 
    return
  end function

  function mp_diviq (ia, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_diviq
    intent (in):: ia, qb
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    da = ia
    call mpdmc (da, 0, q1%mpr)
    call mpdiv (q1%mpr, qb%mpr, mp_diviq%mpr, mpnw) 
    return
  end function

  function mp_divqi (qa, ib)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_divqi
    intent (in):: qa, ib
    integer mpnw
    mpnw = mpnwx
    db = ib
    call mpdivd (qa%mpr, db, 0, mp_divqi%mpr, mpnw) 
    return
  end function

  function mp_divdq (da, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_divdq
    intent (in):: da, qb
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpdmc (da, 0, q1%mpr)
    call mpdiv (q1%mpr, qb%mpr, mp_divdq%mpr, mpnw) 
    return
  end function

  function mp_divqd (qa, db)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_divqd
    intent (in):: qa, db
    integer mpnw
    mpnw = mpnwx
    call mpdivd (qa%mpr, db, 0, mp_divqd%mpr, mpnw) 
    return
  end function

  function mp_divxq (xa, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_divxq
    intent (in):: xa, qb
    type (mp_complex) z1, z2
    integer mpnw
    mpnw = mpnwx
    call mpxzc (xa, z1%mpc)
    call mpmzc (qb%mpr, z2%mpc)
    call mpcdiv (mp4, z1%mpc, z2%mpc, mp_divxq%mpc, mpnw) 
    return
  end function

  function mp_divqx (qa, xb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_divqx
    intent (in):: qa, xb
    type (mp_complex) z1, z2
    integer mpnw
    mpnw = mpnwx
    call mpmzc (qa%mpr, z1%mpc)
    call mpxzc (xb, z2%mpc)
    call mpcdiv (mp4, z1%mpc, z2%mpc, mp_divqx%mpc, mpnw) 
    return
  end function

!  MPR exponentiation routines.

  function mp_expqj (qa, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_expqj
    intent (in):: qa, jb
    type (mp_real) q1, q2
    integer mpnw
    mpnw = mpnwx
    call mplog (qa%mpr, mpl02%mpr, q1%mpr, mpnw) 
    call mpmul (q1%mpr, jb%mpi, q2%mpr, mpnw) 
    call mpexp (q2%mpr, mpl02%mpr, mp_expqj%mpr, mpnw) 
    return
  end function

  function mp_expqq (qa, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_expqq
    intent (in):: qa, qb
    type (mp_real) q1, q2
    integer mpnw
    mpnw = mpnwx
    call mplog (qa%mpr, mpl02%mpr, q1%mpr, mpnw) 
    call mpmul (q1%mpr, qb%mpr, q2%mpr, mpnw) 
    call mpexp (q2%mpr, mpl02%mpr, mp_expqq%mpr, mpnw) 
    return
  end function

  function mp_expiq (ia, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_expiq
    intent (in):: ia, qb
    type (mp_real) q1, q2, q3
    integer mpnw
    mpnw = mpnwx
    da = ia
    call mpdmc (da, 0, q1%mpr)
    call mplog (q1%mpr, mpl02%mpr, q2%mpr, mpnw) 
    call mpmul (q2%mpr, qb%mpr, q3%mpr, mpnw) 
    call mpexp (q3%mpr, mpl02%mpr, mp_expiq%mpr, mpnw) 
    return
  end function

  function mp_expqi (qa, ib)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_expqi
    intent (in):: qa, ib
    integer mpnw
    mpnw = mpnwx
    call mpnpwr (qa%mpr, ib, mp_expqi%mpr, mpnw) 
    return
  end function

  function mp_expdq (da, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_expdq
    intent (in):: da, qb
    type (mp_real) q1, q2, q3
    integer mpnw
    mpnw = mpnwx
    call mpdmc (da, 0, q1%mpr)
    call mplog (q1%mpr, mpl02%mpr, q2%mpr, mpnw) 
    call mpmul (q2%mpr, qb%mpr, q3%mpr, mpnw) 
    call mpexp (q3%mpr, mpl02%mpr, mp_expdq%mpr, mpnw) 
    return
    end function

  function mp_expqd (qa, db)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_expqd
    intent (in):: qa, db
    type (mp_real) q1, q2
    integer mpnw
    mpnw = mpnwx
    call mplog (qa%mpr, mpl02%mpr, q1%mpr, mpnw) 
    call mpmuld (q1%mpr, db, 0, q2%mpr, mpnw) 
    call mpexp (q2%mpr, mpl02%mpr, mp_expqd%mpr, mpnw) 
    return
  end function

!  MPR .EQ. routines.

  function mp_eqtqj (qa, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_eqtqj
    intent (in):: qa, jb
    integer mpnw
    mpnw = mpnwx
    call mpcpr (qa%mpr, jb%mpi, ic, mpnw) 
    if (ic .eq. 0) then
      mp_eqtqj = .true.
    else
      mp_eqtqj = .false.
    endif
    return
  end function

  function mp_eqtqq (qa, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_eqtqq
    intent (in):: qa, qb
    integer mpnw
    mpnw = mpnwx
    call mpcpr (qa%mpr, qb%mpr, ic, mpnw) 
    if (ic .eq. 0) then
      mp_eqtqq = .true.
    else
      mp_eqtqq = .false.
    endif
    return
  end function

  function mp_eqtqz (qa, zb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_eqtqz
    intent (in):: qa, zb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    call mpmzc (qa%mpr, z1%mpc)
    call mpcpr (z1%mpc, zb%mpc, ic1, mpnw) 
    call mpcpr (z1%mpc(mp41), zb%mpc(mp41), ic2, mpnw) 
    if (ic1 .eq. 0 .and. ic2 .eq. 0) then
      mp_eqtqz = .true.
    else
      mp_eqtqz = .false.
    endif
    return
  end function

  function mp_eqtiq (ia, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_eqtiq
    intent (in):: ia, qb
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    da = ia
    call mpdmc (da, 0, q1%mpr)
    call mpcpr (q1%mpr, qb%mpr, ic, mpnw) 
    if (ic .eq. 0) then
      mp_eqtiq = .true.
    else
      mp_eqtiq = .false.
    endif
    return
  end function

  function mp_eqtqi (qa, ib)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_eqtqi
    intent (in):: qa, ib
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    db = ib
    call mpdmc (db, 0, q1%mpr)
    call mpcpr (qa%mpr, q1%mpr, ic, mpnw) 
    if (ic .eq. 0) then
      mp_eqtqi = .true.
    else
      mp_eqtqi = .false.
    endif
    return
  end function

  function mp_eqtdq (da, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_eqtdq
    intent (in):: da, qb
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpdmc (da, 0, q1%mpr)
    call mpcpr (q1%mpr, qb%mpr, ic, mpnw) 
    if (ic .eq. 0) then
      mp_eqtdq = .true.
    else
      mp_eqtdq = .false.
    endif
    return
  end function

  function mp_eqtqd (qa, db)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_eqtqd
    intent (in):: qa, db
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpdmc (db, 0, q1%mpr)
    call mpcpr (qa%mpr, q1%mpr, ic, mpnw) 
    if (ic .eq. 0) then
      mp_eqtqd = .true.
    else
      mp_eqtqd = .false.
    endif
    return
  end function

  function mp_eqtxq (xa, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_eqtxq
    intent (in):: xa, qb
    type (mp_complex) z1, z2
    integer mpnw
    mpnw = mpnwx
    call mpxzc (xa, z1%mpc)
    call mpmzc (qb%mpr, z2%mpc)
    call mpcpr (z1%mpc, z2%mpc, ic1, mpnw) 
    call mpcpr (z1%mpc(mp41), z2%mpc(mp41), ic2, mpnw) 
    if (ic1 .eq. 0 .and. ic2 .eq. 0) then
      mp_eqtxq = .true.
    else
      mp_eqtxq = .false.
    endif
    return
  end function

  function mp_eqtqx (qa, xb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_eqtqx
    intent (in):: qa, xb
    type (mp_complex) z1, z2
    integer mpnw
    mpnw = mpnwx
    call mpmzc (qa%mpr, z1%mpc)
    call mpxzc (xb, z2%mpc)
    call mpcpr (z1%mpc, z2%mpc, ic1, mpnw) 
    call mpcpr (z1%mpc(mp41), z2%mpc(mp41), ic2, mpnw) 
    if (ic1 .eq. 0 .and. ic2 .eq. 0) then
      mp_eqtqx = .true.
    else
      mp_eqtqx = .false.
    endif
    return
  end function

!  MPR .NE. routines.

  function mp_netqj (qa, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_netqj
    intent (in):: qa, jb
    integer mpnw
    mpnw = mpnwx
    call mpcpr (qa%mpr, jb%mpi, ic, mpnw) 
    if (ic .ne. 0) then
      mp_netqj = .true.
    else
      mp_netqj = .false.
    endif
    return
  end function

  function mp_netqq (qa, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_netqq
    intent (in):: qa, qb
    integer mpnw
    mpnw = mpnwx
    call mpcpr (qa%mpr, qb%mpr, ic, mpnw) 
    if (ic .ne. 0) then
      mp_netqq = .true.
    else
      mp_netqq = .false.
    endif
    return
  end function

  function mp_netqz (qa, zb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_netqz
    intent (in):: qa, zb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    call mpmzc (qa%mpr, z1%mpc)
    call mpcpr (z1%mpc, zb%mpc, ic1, mpnw) 
    call mpcpr (z1%mpc(mp41), zb%mpc(mp41), ic2, mpnw) 
    if (ic1 .ne. 0 .or. ic2 .ne. 0) then
      mp_netqz = .true.
    else
      mp_netqz = .false.
    endif
    return
  end function

  function mp_netiq (ia, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_netiq
    intent (in):: ia, qb
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    da = ia
    call mpdmc (da, 0, q1%mpr)
    call mpcpr (q1%mpr, qb%mpr, ic, mpnw) 
    if (ic .ne. 0) then
      mp_netiq = .true.
    else
      mp_netiq = .false.
    endif
    return
  end function

  function mp_netqi (qa, ib)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_netqi
    intent (in):: qa, ib
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    db = ib
    call mpdmc (db, 0, q1%mpr)
    call mpcpr (qa%mpr, q1%mpr, ic, mpnw) 
    if (ic .ne. 0) then
      mp_netqi = .true.
    else
      mp_netqi = .false.
    endif
    return
  end function

  function mp_netdq (da, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_netdq
    intent (in):: da, qb
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpdmc (da, 0, q1%mpr)
    call mpcpr (q1%mpr, qb%mpr, ic, mpnw) 
    if (ic .ne. 0) then
      mp_netdq = .true.
    else
      mp_netdq = .false.
    endif
    return
  end function

  function mp_netqd (qa, db)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_netqd
    intent (in):: qa, db
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpdmc (db, 0, q1%mpr)
    call mpcpr (qa%mpr, q1%mpr, ic, mpnw) 
    if (ic .ne. 0) then
      mp_netqd = .true.
    else
      mp_netqd = .false.
    endif
    return
  end function

  function mp_netxq (xa, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_netxq
    intent (in):: xa, qb
    type (mp_complex) z1, z2
    integer mpnw
    mpnw = mpnwx
    call mpxzc (xa, z1%mpc)
    call mpmzc (qb%mpr, z2%mpc)
    call mpcpr (z1%mpc, z2%mpc, ic1, mpnw) 
    call mpcpr (z1%mpc(mp41), z2%mpc(mp41), ic2, mpnw) 
    if (ic1 .ne. 0 .or. ic2 .ne. 0) then
      mp_netxq = .true.
    else
      mp_netxq = .false.
    endif
    return
  end function

  function mp_netqx (qa, xb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_netqx
    intent (in):: qa, xb
    type (mp_complex) z1, z2
    integer mpnw
    mpnw = mpnwx
    call mpmzc (qa%mpr, z1%mpc)
    call mpxzc (xb, z2%mpc)
    call mpcpr (z1%mpc, z2%mpc, ic1, mpnw) 
    call mpcpr (z1%mpc(mp41), z2%mpc(mp41), ic2, mpnw) 
    if (ic1 .ne. 0 .or. ic2 .ne. 0) then
      mp_netqx = .true.
    else
      mp_netqx = .false.
    endif
    return
  end function

!  MPR .LE. routines.

  function mp_letqj (qa, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_letqj
    intent (in):: qa, jb
    integer mpnw
    mpnw = mpnwx
    call mpcpr (qa%mpr, jb%mpi, ic, mpnw) 
    if (ic .le. 0) then
      mp_letqj = .true.
    else
      mp_letqj = .false.
    endif
    return
  end function

  function mp_letqq (qa, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_letqq
    intent (in):: qa, qb
    integer mpnw
    mpnw = mpnwx
    call mpcpr (qa%mpr, qb%mpr, ic, mpnw) 
    if (ic .le. 0) then
      mp_letqq = .true.
    else
      mp_letqq = .false.
    endif
    return
  end function

  function mp_letiq (ia, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_letiq
    intent (in):: ia, qb
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    da = ia
    call mpdmc (da, 0, q1%mpr)
    call mpcpr (q1%mpr, qb%mpr, ic, mpnw) 
    if (ic .le. 0) then
      mp_letiq = .true.
    else
      mp_letiq = .false.
    endif
    return
  end function

  function mp_letqi (qa, ib)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_letqi
    intent (in):: qa, ib
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    db = ib
    call mpdmc (db, 0, q1%mpr)
    call mpcpr (qa%mpr, q1%mpr, ic, mpnw) 
    if (ic .le. 0) then
      mp_letqi = .true.
    else
      mp_letqi = .false.
    endif
    return
  end function

  function mp_letdq (da, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_letdq
    intent (in):: da, qb
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpdmc (da, 0, q1%mpr)
    call mpcpr (q1%mpr, qb%mpr, ic, mpnw) 
    if (ic .le. 0) then
      mp_letdq = .true.
    else
      mp_letdq = .false.
    endif
    return
  end function

  function mp_letqd (qa, db)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_letqd
    intent (in):: qa, db
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpdmc (db, 0, q1%mpr)
    call mpcpr (qa%mpr, q1%mpr, ic, mpnw) 
    if (ic .le. 0) then
      mp_letqd = .true.
    else
      mp_letqd = .false.
    endif
    return
  end function

!  MPR .GE. routines.

  function mp_getqj (qa, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_getqj
    intent (in):: qa, jb
    integer mpnw
    mpnw = mpnwx
    call mpcpr (qa%mpr, jb%mpi, ic, mpnw) 
    if (ic .ge. 0) then
      mp_getqj = .true.
    else
      mp_getqj = .false.
    endif
    return
  end function

  function mp_getqq (qa, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_getqq
    intent (in):: qa, qb
    integer mpnw
    mpnw = mpnwx
    call mpcpr (qa%mpr, qb%mpr, ic, mpnw) 
    if (ic .ge. 0) then
      mp_getqq = .true.
    else
      mp_getqq = .false.
    endif
    return
  end function

  function mp_getiq (ia, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_getiq
    intent (in):: ia, qb
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    da = ia
    call mpdmc (da, 0, q1%mpr)
    call mpcpr (q1%mpr, qb%mpr, ic, mpnw) 
    if (ic .ge. 0) then
      mp_getiq = .true.
    else
      mp_getiq = .false.
    endif
    return
  end function

  function mp_getqi (qa, ib)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_getqi
    intent (in):: qa, ib
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    db = ib
    call mpdmc (db, 0, q1%mpr)
    call mpcpr (qa%mpr, q1%mpr, ic, mpnw) 
    if (ic .ge. 0) then
      mp_getqi = .true.
    else
      mp_getqi = .false.
    endif
    return
  end function

  function mp_getdq (da, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_getdq
    intent (in):: da, qb
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpdmc (da, 0, q1%mpr)
    call mpcpr (q1%mpr, qb%mpr, ic, mpnw) 
    if (ic .ge. 0) then
      mp_getdq = .true.
    else
      mp_getdq = .false.
    endif
    return
  end function

  function mp_getqd (qa, db)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_getqd
    intent (in):: qa, db
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpdmc (db, 0, q1%mpr)
    call mpcpr (qa%mpr, q1%mpr, ic, mpnw) 
    if (ic .ge. 0) then
      mp_getqd = .true.
    else
      mp_getqd = .false.
    endif
    return
  end function

!  MPR .LT. routines.

  function mp_lttqj (qa, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_lttqj
    intent (in):: qa, jb
    integer mpnw
    mpnw = mpnwx
    call mpcpr (qa%mpr, jb%mpi, ic, mpnw) 
    if (ic .lt. 0) then
      mp_lttqj = .true.
    else
      mp_lttqj = .false.
    endif
    return
  end function

  function mp_lttqq (qa, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_lttqq
    intent (in):: qa, qb
    integer mpnw
    mpnw = mpnwx
    call mpcpr (qa%mpr, qb%mpr, ic, mpnw) 
    if (ic .lt. 0) then
      mp_lttqq = .true.
    else
      mp_lttqq = .false.
    endif
    return
  end function

  function mp_lttiq (ia, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_lttiq
    intent (in):: ia, qb
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    da = ia
    call mpdmc (da, 0, q1%mpr)
    call mpcpr (q1%mpr, qb%mpr, ic, mpnw) 
    if (ic .lt. 0) then
      mp_lttiq = .true.
    else
      mp_lttiq = .false.
    endif
    return
  end function

  function mp_lttqi (qa, ib)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_lttqi
    intent (in):: qa, ib
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    db = ib
    call mpdmc (db, 0, q1%mpr)
    call mpcpr (qa%mpr, q1%mpr, ic, mpnw) 
    if (ic .lt. 0) then
      mp_lttqi = .true.
    else
      mp_lttqi = .false.
    endif
    return
  end function

  function mp_lttdq (da, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_lttdq
    intent (in):: da, qb
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpdmc (da, 0, q1%mpr)
    call mpcpr (q1%mpr, qb%mpr, ic, mpnw) 
    if (ic .lt. 0) then
      mp_lttdq = .true.
    else
      mp_lttdq = .false.
    endif
    return
  end function

  function mp_lttqd (qa, db)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_lttqd
    intent (in):: qa, db
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpdmc (db, 0, q1%mpr)
    call mpcpr (qa%mpr, q1%mpr, ic, mpnw) 
    if (ic .lt. 0) then
      mp_lttqd = .true.
    else
      mp_lttqd = .false.
    endif
    return
  end function

!  MPR .GT. routines.

  function mp_gttqj (qa, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_gttqj
    intent (in):: qa, jb
    integer mpnw
    mpnw = mpnwx
    call mpcpr (qa%mpr, jb%mpi, ic, mpnw) 
    if (ic .gt. 0) then
      mp_gttqj = .true.
    else
      mp_gttqj = .false.
    endif
    return
  end function

  function mp_gttqq (qa, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_gttqq
    intent (in):: qa, qb
    integer mpnw
    mpnw = mpnwx
    call mpcpr (qa%mpr, qb%mpr, ic, mpnw) 
    if (ic .gt. 0) then
      mp_gttqq = .true.
    else
      mp_gttqq = .false.
    endif
    return
  end function

  function mp_gttiq (ia, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_gttiq
    intent (in):: ia, qb
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    da = ia
    call mpdmc (da, 0, q1%mpr)
    call mpcpr (q1%mpr, qb%mpr, ic, mpnw) 
    if (ic .gt. 0) then
      mp_gttiq = .true.
    else
      mp_gttiq = .false.
    endif
    return
  end function

  function mp_gttqi (qa, ib)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_gttqi
    intent (in):: qa, ib
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    db = ib
    call mpdmc (db, 0, q1%mpr)
    call mpcpr (qa%mpr, q1%mpr, ic, mpnw) 
    if (ic .gt. 0) then
      mp_gttqi = .true.
    else
      mp_gttqi = .false.
    endif
    return
  end function

  function mp_gttdq (da, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_gttdq
    intent (in):: da, qb
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpdmc (da, 0, q1%mpr)
    call mpcpr (q1%mpr, qb%mpr, ic, mpnw) 
    if (ic .gt. 0) then
      mp_gttdq = .true.
    else
      mp_gttdq = .false.
    endif
    return
  end function

  function mp_gttqd (qa, db)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_gttqd
    intent (in):: qa, db
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpdmc (db, 0, q1%mpr)
    call mpcpr (qa%mpr, q1%mpr, ic, mpnw) 
    if (ic .gt. 0) then
      mp_gttqd = .true.
    else
      mp_gttqd = .false.
    endif
    return
  end function

end module


module mpcmpmod

!  This Fortran-90 module defines operator extensions involving the
!  MP_COMPLEX datatype.  For operations involving two MP data types,
!  those whose first argument is MP_COMPLEX are included here.
!  Others are handled in other modules.

!  The subroutines and functions defined in this module are private
!  and not intended to be called directly by the user.

use mpfunmod
use mpdefmod
private kdb, mp4, mp24, mp41
parameter (kdb = kind (0.d0), mp4 = mpwds + 4, mp24 = 2 * mp4, mp41 = mp4 + 1)
private &
  mp_eqzj, mp_eqzq, mp_eqzz, mp_eqiz, mp_eqzi, &
  mp_eqdz, mp_eqzd, mp_eqxz, mp_eqzx, &
  mp_addzj, mp_addzq, mp_addzz, mp_addiz, mp_addzi, &
  mp_adddz, mp_addzd, mp_addxz, mp_addzx, &
  mp_subzj, mp_subzq, mp_subzz, mp_subiz, mp_subzi, &
  mp_subdz, mp_subzd, mp_subxz, mp_subzx, mp_negz, &
  mp_mulzj, mp_mulzq, mp_mulzz, mp_muliz, mp_mulzi, &
  mp_muldz, mp_mulzd, mp_mulxz, mp_mulzx, &
  mp_divzj, mp_divzq, mp_divzz, mp_diviz, mp_divzi, &
  mp_divdz, mp_divzd, mp_divxz, mp_divzx, mp_expzi, &
  mp_eqtzj, mp_eqtzq, mp_eqtzz, mp_eqtiz, mp_eqtzi, &
  mp_eqtdz, mp_eqtzd, mp_eqtxz, mp_eqtzx, &
  mp_netzj, mp_netzq, mp_netzz, mp_netiz, mp_netzi, &
  mp_netdz, mp_netzd, mp_netxz, mp_netzx

!  MPR operator extension interface blocks.

interface assignment (=)
  module procedure mp_eqzj
  module procedure mp_eqzq
  module procedure mp_eqzz
  module procedure mp_eqiz
  module procedure mp_eqzi
  module procedure mp_eqdz
  module procedure mp_eqzd
  module procedure mp_eqxz
  module procedure mp_eqzx
end interface

interface operator (+)
  module procedure mp_addzj
  module procedure mp_addzq
  module procedure mp_addzz
  module procedure mp_addiz
  module procedure mp_addzi
  module procedure mp_adddz
  module procedure mp_addzd
  module procedure mp_addxz
  module procedure mp_addzx
end interface

interface operator (-)
  module procedure mp_subzj
  module procedure mp_subzq
  module procedure mp_subzz
  module procedure mp_subiz
  module procedure mp_subzi
  module procedure mp_subdz
  module procedure mp_subzd
  module procedure mp_subxz
  module procedure mp_subzx

  module procedure mp_negz
end interface

interface operator (*)
  module procedure mp_mulzj
  module procedure mp_mulzq
  module procedure mp_mulzz
  module procedure mp_muliz
  module procedure mp_mulzi
  module procedure mp_muldz
  module procedure mp_mulzd
  module procedure mp_mulxz
  module procedure mp_mulzx
end interface

interface operator (/)
  module procedure mp_divzj
  module procedure mp_divzq
  module procedure mp_divzz
  module procedure mp_diviz
  module procedure mp_divzi
  module procedure mp_divdz
  module procedure mp_divzd
  module procedure mp_divxz
  module procedure mp_divzx
end interface

interface operator (**)
  module procedure mp_expzi
end interface

interface operator (.eq.)
  module procedure mp_eqtzj
  module procedure mp_eqtzq
  module procedure mp_eqtzz
  module procedure mp_eqtiz
  module procedure mp_eqtzi
  module procedure mp_eqtdz
  module procedure mp_eqtzd
  module procedure mp_eqtxz
  module procedure mp_eqtzx
end interface

interface operator (.ne.)
  module procedure mp_netzj
  module procedure mp_netzq
  module procedure mp_netzz
  module procedure mp_netiz
  module procedure mp_netzi
  module procedure mp_netdz
  module procedure mp_netzd
  module procedure mp_netxz
  module procedure mp_netzx
end interface

contains

!  MPC assignment routines.

  subroutine mp_eqzj (za, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    intent (out):: za
    intent (in):: jb
    call mpmzc (jb%mpi, za%mpc)
    return
  end subroutine

  subroutine mp_eqzq (za, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    intent (out):: za
    intent (in):: qb
    integer mpnw
    mpnw = mpnwx
    call mpmzc (qb%mpr, za%mpc)
    return
  end subroutine

  subroutine mp_eqzz (za, zb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    intent (out):: za
    intent (in):: zb
    integer mpnw
    mpnw = mpnwx
    call mpceq (mp4, zb%mpc, za%mpc, mpnw) 
    return
  end subroutine

  subroutine mp_eqiz (ia, zb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    intent (out):: ia
    intent (in):: zb
    integer mpnw
    mpnw = mpnwx
    call mpmdc (zb%mpc, db, ib)
    ia = db * 2.d0 ** ib
    return
  end subroutine

  subroutine mp_eqzi (za, ib)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    intent (out):: za
    intent (in):: ib
    integer mpnw
    mpnw = mpnwx
    xb = ib
    call mpxzc (xb, za%mpc)
    return
  end subroutine

  subroutine mp_eqdz (da, zb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    intent (out):: da
    intent (in):: zb
    integer mpnw
    mpnw = mpnwx
    call mpmdc (zb%mpc, db, ib)
    da = db * 2.d0 ** ib
    return
  end subroutine

  subroutine mp_eqzd (za, db)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    intent (out):: za
    intent (in):: db
    integer mpnw
    mpnw = mpnwx
    xb = db
    call mpxzc (xb, za%mpc)
    return
  end subroutine

  subroutine mp_eqxz (xa, zb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    intent (out):: xa
    intent (in):: zb
    integer mpnw
    mpnw = mpnwx
    call mpmdc (zb%mpc, db, ib)
    call mpmdc (zb%mpc(mp41), dc, ic)
    xa = cmplx (db * 2.d0 ** ib, dc * 2.d0 ** ic, kdb)
    return
  end subroutine

  subroutine mp_eqzx (za, xb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    intent (out):: za
    intent (in):: xb
    integer mpnw
    mpnw = mpnwx
    call mpxzc (xb, za%mpc)
    return
  end subroutine

!  MPC add routines.

  function mp_addzj (za, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_addzj
    intent (in):: za, jb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    call mpmzc (jb%mpi, z1%mpc)
    call mpcadd (mp4, za%mpc, z1%mpc, mp_addzj%mpc, mpnw) 
    return
  end function

  function mp_addzq (za, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_addzq
    intent (in):: za, qb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    call mpmzc (qb%mpr, z1%mpc)
    call mpcadd (mp4, za%mpc, z1%mpc, mp_addzq%mpc, mpnw) 
    return
  end function

  function mp_addzz (za, zb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_addzz
    intent (in):: za, zb
    integer mpnw
    mpnw = mpnwx
    call mpcadd (mp4, za%mpc, zb%mpc, mp_addzz%mpc, mpnw) 
    return
  end function

  function mp_addiz (ia, zb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_addiz
    intent (in):: ia, zb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    xa = ia
    call mpxzc (xa, z1%mpc)
    call mpcadd (mp4, z1%mpc, zb%mpc, mp_addiz%mpc, mpnw) 
    return
  end function

  function mp_addzi (za, ib)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_addzi
    intent (in):: za, ib
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    xb = ib
    call mpxzc (xb, z1%mpc)
    call mpcadd (mp4, za%mpc, z1%mpc, mp_addzi%mpc, mpnw) 
    return
  end function

  function mp_adddz (da, zb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_adddz
    intent (in):: da, zb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    xa = da
    call mpxzc (xa, z1%mpc)
    call mpcadd (mp4, z1%mpc, zb%mpc, mp_adddz%mpc, mpnw) 
    return
  end function

  function mp_addzd (za, db)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_addzd
    intent (in):: za, db
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    xb = db
    call mpxzc (xb, z1%mpc)
    call mpcadd (mp4, za%mpc, z1%mpc, mp_addzd%mpc, mpnw) 
    return
  end function

  function mp_addxz (xa, zb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_addxz
    intent (in):: xa, zb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    call mpxzc (xa, z1%mpc)
    call mpcadd (mp4, z1%mpc, zb%mpc, mp_addxz%mpc, mpnw) 
    return
  end function

  function mp_addzx (za, xb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_addzx
    intent (in):: za, xb
    type (mp_complex) q1
    integer mpnw
    mpnw = mpnwx
    call mpxzc (xb, q1%mpc)
    call mpcadd (mp4, za%mpc, q1%mpc, mp_addzx%mpc, mpnw) 
    return
  end function

!  MPC subtract routines.

  function mp_subzj (za, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_subzj
    intent (in):: za, jb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    call mpmzc (jb%mpi, z1%mpc)
    call mpcsub (mp4, za%mpc, z1%mpc, mp_subzj%mpc, mpnw) 
    return
  end function

  function mp_subzq (za, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_subzq
    type (mp_complex) z1
    intent (in):: za, qb
    integer mpnw
    mpnw = mpnwx
    call mpmzc (qb%mpr, z1%mpc)
    call mpcsub (mp4, za%mpc, z1%mpc, mp_subzq%mpc, mpnw) 
    return
  end function

  function mp_subzz (za, zb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_subzz
    intent (in):: za, zb
    integer mpnw
    mpnw = mpnwx
    call mpcsub (mp4, za%mpc, zb%mpc, mp_subzz%mpc, mpnw) 
    return
  end function

  function mp_subiz (ia, zb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_subiz
    intent (in):: ia, zb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    xa = ia
    call mpxzc (xa, z1%mpc)
    call mpcsub (mp4, z1%mpc, zb%mpc, mp_subiz%mpc, mpnw) 
    return
  end function

  function mp_subzi (za, ib)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_subzi
    intent (in):: za, ib
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    xb = ib
    call mpxzc (xb, z1%mpc)
    call mpcsub (mp4, za%mpc, z1%mpc, mp_subzi%mpc, mpnw) 
    return
  end function

  function mp_subdz (da, zb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_subdz
    intent (in):: da, zb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    xa = da
    call mpxzc (xa, z1%mpc)
    call mpcsub (mp4, z1%mpc, zb%mpc, mp_subdz%mpc, mpnw) 
    return
  end function

  function mp_subzd (za, db)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_subzd
    intent (in):: za, db
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    xb = db
    call mpxzc (xb, z1%mpc)
    call mpcsub (mp4, za%mpc, z1%mpc, mp_subzd%mpc, mpnw) 
    return
  end function

  function mp_subxz (xa, zb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_subxz
    intent (in):: xa, zb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    call mpxzc (xa, z1%mpc)
    call mpcsub (mp4, z1%mpc, zb%mpc, mp_subxz%mpc, mpnw) 
    return
  end function

  function mp_subzx (za, xb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_subzx
    intent (in):: za, xb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    call mpxzc (xb, z1%mpc)
    call mpcsub (mp4, za%mpc, z1%mpc, mp_subzx%mpc, mpnw) 
    return
  end function

!  MPC negation routine.

  function mp_negz (za)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_negz
    intent (in):: za
    integer mpnw
    mpnw = mpnwx
    call mpceq (mp4, za%mpc, mp_negz%mpc, mpnw) 
    mp_negz%mpc(1) = - za%mpc(1)
    mp_negz%mpc(mp41) = - za%mpc(mp41)
    return
  end function

!  MPC multiply routines.

  function mp_mulzj (za, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_mulzj
    intent (in):: za, jb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    call mpmzc (jb%mpi, z1%mpc)
    call mpcmul (mp4, za%mpc, z1%mpc, mp_mulzj%mpc, mpnw) 
    return
  end function

  function mp_mulzq (za, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_mulzq
    intent (in):: za, qb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    call mpmzc (qb%mpr, z1%mpc)
    call mpcmul (mp4, za%mpc, z1%mpc, mp_mulzq%mpc, mpnw) 
    return
  end function

  function mp_mulzz (za, zb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_mulzz
    intent (in):: za, zb
    integer mpnw
    mpnw = mpnwx
    call mpcmul (mp4, za%mpc, zb%mpc, mp_mulzz%mpc, mpnw) 
    return
  end function

  function mp_muliz (ia, zb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_muliz
    intent (in):: ia, zb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    xa = ia
    call mpxzc (xa, z1%mpc)
    call mpcmul (mp4, z1%mpc, zb%mpc, mp_muliz%mpc, mpnw) 
    return
  end function

  function mp_mulzi (za, ib)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_mulzi
    intent (in):: za, ib
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    xb = ib
    call mpxzc (xb, z1%mpc)
    call mpcmul (mp4, za%mpc, z1%mpc, mp_mulzi%mpc, mpnw) 
    return
  end function

  function mp_muldz (da, zb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_muldz
    intent (in):: da, zb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    xa = da
    call mpxzc (xa, z1%mpc)
    call mpcmul (mp4, z1%mpc, zb%mpc, mp_muldz%mpc, mpnw) 
    return
  end function

  function mp_mulzd (za, db)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_mulzd
    intent (in):: za, db
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    xb = db
    call mpxzc (xb, z1%mpc)
    call mpcmul (mp4, za%mpc, z1%mpc, mp_mulzd%mpc, mpnw) 
    return
  end function

  function mp_mulxz (xa, zb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_mulxz
    intent (in):: xa, zb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    call mpxzc (xa, z1%mpc)
    call mpcmul (mp4, z1%mpc, zb%mpc, mp_mulxz%mpc, mpnw) 
    return
  end function

  function mp_mulzx (za, xb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_mulzx
    intent (in):: za, xb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    call mpxzc (xb, z1%mpc)
    call mpcmul (mp4, za%mpc, z1%mpc, mp_mulzx%mpc, mpnw) 
    return
  end function

!  MPC divide routines.

  function mp_divzj (za, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_divzj
    intent (in):: za, jb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    call mpmzc (jb%mpi, z1%mpc)
    call mpcdiv (mp4, za%mpc, z1%mpc, mp_divzj%mpc, mpnw) 
    return
  end function

  function mp_divzq (za, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_divzq
    intent (in):: za, qb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    call mpmzc (qb%mpr, z1%mpc)
    call mpcdiv (mp4, za%mpc, z1%mpc, mp_divzq%mpc, mpnw) 
    return
  end function

  function mp_divzz (za, zb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_divzz
    intent (in):: za, zb
    integer mpnw
    mpnw = mpnwx
    call mpcdiv (mp4, za%mpc, zb%mpc, mp_divzz%mpc, mpnw) 
    return
  end function

  function mp_diviz (ia, zb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_diviz
    intent (in):: ia, zb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    xa = ia
    call mpxzc (xa, z1%mpc)
    call mpcdiv (mp4, z1%mpc, zb%mpc, mp_diviz%mpc, mpnw) 
    return
  end function

  function mp_divzi (za, ib)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_divzi
    intent (in):: za, ib
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    xb = ib
    call mpxzc (xb, z1%mpc)
    call mpcdiv (mp4, za%mpc, z1%mpc, mp_divzi%mpc, mpnw) 
    return
  end function

  function mp_divdz (da, zb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_divdz
    intent (in):: da, zb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    xa = da
    call mpxzc (xa, z1%mpc)
    call mpcdiv (mp4, z1%mpc, zb%mpc, mp_divdz%mpc, mpnw) 
    return
  end function

  function mp_divzd (za, db)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_divzd
    intent (in):: za, db
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    xb = db
    call mpxzc (xb, z1%mpc)
    call mpcdiv (mp4, za%mpc, z1%mpc, mp_divzd%mpc, mpnw) 
    return
  end function

  function mp_divxz (xa, zb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_divxz
    intent (in):: xa, zb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    call mpxzc (xa, z1%mpc)
    call mpcdiv (mp4, z1%mpc, zb%mpc, mp_divxz%mpc, mpnw) 
    return
  end function

  function mp_divzx (za, xb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_divzx
    intent (in):: za, xb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    call mpxzc (xb, z1%mpc)
    call mpcdiv (mp4, za%mpc, z1%mpc, mp_divzx%mpc, mpnw) 
    return
  end function

!  MPC exponentiation routines.

  function mp_expzi (za, ib)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_expzi
    intent (in):: za, ib
    integer mpnw
    mpnw = mpnwx
    call mpcpwr (mp4, za%mpc, ib, mp_expzi%mpc, mpnw) 
    return
  end function

!  MPC .EQ. routines.

  function mp_eqtzj (za, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_eqtzj
    intent (in):: za, jb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    call mpmzc (jb%mpi, z1%mpc)
    call mpcpr (za%mpc, z1%mpc, ic1, mpnw) 
    call mpcpr (za%mpc(mp41), z1%mpc(mp41), ic2, mpnw) 
    if (ic1 .eq. 0 .and. ic2 .eq. 0) then
      mp_eqtzj = .true.
    else
      mp_eqtzj = .false.
    endif
    return
  end function

  function mp_eqtzq (za, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_eqtzq
    intent (in):: za, qb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    call mpmzc (qb%mpr, z1%mpc)
    call mpcpr (za%mpc, z1%mpc, ic1, mpnw) 
    call mpcpr (za%mpc(mp41), z1%mpc(mp41), ic2, mpnw) 
    if (ic1 .eq. 0 .and. ic2 .eq. 0) then
      mp_eqtzq = .true.
    else
      mp_eqtzq = .false.
    endif
    return
  end function

  function mp_eqtzz (za, zb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_eqtzz
    intent (in):: za, zb
    integer mpnw
    mpnw = mpnwx
    call mpcpr (za%mpc, zb%mpc, ic1, mpnw) 
    call mpcpr (za%mpc(mp41), zb%mpc(mp41), ic2, mpnw) 
    if (ic1 .eq. 0 .and. ic2 .eq. 0) then
      mp_eqtzz = .true.
    else
      mp_eqtzz = .false.
    endif
    return
  end function

  function mp_eqtiz (ia, zb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_eqtiz
    intent (in):: ia, zb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    da = ia
    call mpdmc (da, 0, z1%mpc)
    call mpcpr (z1%mpc, zb%mpc, ic1, mpnw) 
    call mpcpr (z1%mpc(mp41), zb%mpc(mp41), ic2, mpnw) 
    if (ic1 .eq. 0 .and. ic2 .eq. 0) then
      mp_eqtiz = .true.
    else
      mp_eqtiz = .false.
    endif
    return
  end function

  function mp_eqtzi (za, ib)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_eqtzi
    intent (in):: za, ib
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    db = ib
    call mpdmc (db, 0, z1%mpc)
    call mpcpr (za%mpc, z1%mpc, ic1, mpnw) 
    call mpcpr (za%mpc(mp41), z1%mpc(mp41), ic2, mpnw) 
    if (ic1 .eq. 0 .and. ic2 .eq. 0) then
      mp_eqtzi = .true.
    else
      mp_eqtzi = .false.
    endif
    return
  end function

  function mp_eqtdz (da, zb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_eqtdz
    intent (in):: da, zb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    call mpdmc (da, 0, z1%mpc)
    call mpcpr (z1%mpc, zb%mpc, ic1, mpnw) 
    call mpcpr (z1%mpc(mp41), zb%mpc(mp41), ic2, mpnw) 
    if (ic1 .eq. 0 .and. ic2 .eq. 0) then
      mp_eqtdz = .true.
    else
      mp_eqtdz = .false.
    endif
    return
  end function

  function mp_eqtzd (za, db)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_eqtzd
    intent (in):: za, db
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    call mpdmc (db, 0, z1%mpc)
    call mpcpr (za%mpc, z1%mpc, ic1, mpnw) 
    call mpcpr (za%mpc(mp41), z1%mpc(mp41), ic2, mpnw) 
    if (ic1 .eq. 0 .and. ic2 .eq. 0) then
      mp_eqtzd = .true.
    else
      mp_eqtzd = .false.
    endif
    return
  end function

  function mp_eqtxz (xa, zb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_eqtxz
    intent (in):: xa, zb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    call mpxzc (xa, z1%mpc)
    call mpcpr (z1%mpc, zb%mpc, ic1, mpnw) 
    call mpcpr (z1%mpc(mp41), zb%mpc(mp41), ic2, mpnw) 
    if (ic1 .eq. 0 .and. ic2 .eq. 0) then
      mp_eqtxz = .true.
    else
      mp_eqtxz = .false.
    endif
    return
  end function

  function mp_eqtzx (za, xb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_eqtzx
    intent (in):: za, xb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    call mpxzc (xb, z1%mpc)
    call mpcpr (za%mpc, z1%mpc, ic1, mpnw) 
    call mpcpr (za%mpc(mp41), z1%mpc(mp41), ic2, mpnw) 
    if (ic1 .eq. 0 .and. ic2 .eq. 0) then
      mp_eqtzx = .true.
    else
      mp_eqtzx = .false.
    endif
    return
  end function

!  MPC .NE. routines.

  function mp_netzj (za, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_netzj
    intent (in):: za, jb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    call mpmzc (jb%mpi, z1%mpc)
    call mpcpr (za%mpc, z1%mpc, ic1, mpnw) 
    call mpcpr (za%mpc(mp41), z1%mpc(mp41), ic2, mpnw) 
    if (ic1 .ne. 0 .or. ic2 .ne. 0) then
      mp_netzj = .true.
    else
      mp_netzj = .false.
    endif
    return
  end function

  function mp_netzq (za, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_netzq
    intent (in):: za, qb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    call mpmzc (qb%mpr, z1%mpc)
    call mpcpr (za%mpc, z1%mpc, ic1, mpnw) 
    call mpcpr (za%mpc(mp41), z1%mpc(mp41), ic2, mpnw) 
    if (ic1 .ne. 0 .or. ic2 .ne. 0) then
      mp_netzq = .true.
    else
      mp_netzq = .false.
    endif
    return
  end function

  function mp_netzz (za, zb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_netzz
    intent (in):: za, zb
    integer mpnw
    mpnw = mpnwx
    call mpcpr (za%mpc, zb%mpc, ic1, mpnw) 
    call mpcpr (za%mpc(mp41), zb%mpc(mp41), ic2, mpnw) 
    if (ic1 .ne. 0 .or. ic2 .ne. 0) then
      mp_netzz = .true.
    else
      mp_netzz = .false.
    endif
    return
  end function

  function mp_netiz (ia, zb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_netiz
    intent (in):: ia, zb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    da = ia
    call mpdmc (da, 0, z1%mpc)
    call mpcpr (z1%mpc, zb%mpc, ic1, mpnw) 
    call mpcpr (z1%mpc(mp41), zb%mpc(mp41), ic2, mpnw) 
    if (ic1 .ne. 0 .or. ic2 .ne. 0) then
      mp_netiz = .true.
    else
      mp_netiz = .false.
    endif
    return
  end function

  function mp_netzi (za, ib)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_netzi
    intent (in):: za, ib
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    db = ib
    call mpdmc (db, 0, z1%mpc)
    call mpcpr (za%mpc, z1%mpc, ic1, mpnw) 
    call mpcpr (za%mpc(mp41), z1%mpc(mp41), ic2, mpnw) 
    if (ic1 .ne. 0 .or. ic2 .ne. 0) then
      mp_netzi = .true.
    else
      mp_netzi = .false.
    endif
    return
  end function

  function mp_netdz (da, zb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_netdz
    intent (in):: da, zb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    call mpdmc (da, 0, z1%mpc)
    call mpcpr (z1%mpc, zb%mpc, ic1, mpnw) 
    call mpcpr (z1%mpc(mp41), zb%mpc(mp41), ic2, mpnw) 
    if (ic1 .ne. 0 .or. ic2 .ne. 0) then
      mp_netdz = .true.
    else
      mp_netdz = .false.
    endif
    return
  end function

  function mp_netzd (za, db)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_netzd
    intent (in):: za, db
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    call mpdmc (db, 0, z1%mpc)
    call mpcpr (za%mpc, z1%mpc, ic1, mpnw) 
    call mpcpr (za%mpc(mp41), z1%mpc(mp41), ic2, mpnw) 
    if (ic1 .ne. 0 .or. ic2 .ne. 0) then
      mp_netzd = .true.
    else
      mp_netzd = .false.
    endif
    return
  end function

  function mp_netxz (xa, zb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_netxz
    intent (in):: xa, zb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    call mpxzc (xa, z1%mpc)
    call mpcpr (z1%mpc, zb%mpc, ic1, mpnw) 
    call mpcpr (z1%mpc(mp41), zb%mpc(mp41), ic2, mpnw) 
    if (ic1 .ne. 0 .or. ic2 .ne. 0) then
      mp_netxz = .true.
    else
      mp_netxz = .false.
    endif
    return
  end function

  function mp_netzx (za, xb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    logical mp_netzx
    intent (in):: za, xb
    type (mp_complex) z1
    integer mpnw
    mpnw = mpnwx
    call mpxzc (xb, z1%mpc)
    call mpcpr (za%mpc, z1%mpc, ic1, mpnw) 
    call mpcpr (za%mpc(mp41), z2%mpc(mp41), ic2, mpnw) 
    if (ic1 .ne. 0 .or. ic2 .ne. 0) then
      mp_netzx = .true.
    else
      mp_netzx = .false.
    endif
    return
  end function

end module


module mpgenmod

!  This Fortran-90 module defines generic functions involving all
!  MP datatypes.

!  The subroutines and functions defined in this module are private
!  and not intended to be called directly by the user.  The generic
!  names (i.e. interface block names) are publicly accessible, though.

use mpfunmod
use mpdefmod
private kdb, mp4, mp24, mp41
parameter (kdb = kind (0.d0), mp4 = mpwds + 4, mp24 = 2 * mp4, mp41 = mp4 + 1)
private &
  mp_absj, mp_absq, mp_absz, mp_acos, mp_imag, mp_aint, mp_anint, &
  mp_asin, mp_atan, mp_atan2, mp_jtoc, mp_qtoc, mp_ztoc, mp_conjg, &
  mp_cos, mp_cosz, mp_cosh, mp_jtod, mp_qtod, mp_ztod, mp_jtox, mp_qtox, &
  mp_ztox, mp_exp, mp_expz, mp_jtoi, mp_qtoi, mp_ztoi, mp_log, mp_logz, &
  mp_log10, mp_maxj, mp_maxq, mp_maxq3, mp_minj, mp_minq, mp_minq3, mp_modj, &
  mp_modq, mp_jtoz, mp_qtoz, mp_itoz, mp_dtoz, mp_xtoz, &
  mp_atoz, mp_jjtoz, mp_qqtoz, mp_iitoz, mp_ddtoz, mp_aatoz, &
  mp_cssh, mp_cssn, mp_qtoj, mp_ztoj, mp_itoj, mp_rtoj, mp_ctoj, mp_dtoj, &
  mp_xtoj, mp_atoj, mp_nrt, mp_rand, mp_inpj, mp_inpq, mp_inpz, &
  mp_jtoq, mp_ztoq, mp_itoq, mp_dtoq, mp_xtoq, &
  mp_atoq, mp_outj, mp_outq, mp_outz, mp_nint, &
  mp_signj, mp_signq, mp_sin, mp_sinz, mp_sinh, mp_sqrtq, &
  mp_sqrtz, mp_tan, mp_tanh

!  MP generic interface blocks.

interface abs
  module procedure mp_absj
  module procedure mp_absq
  module procedure mp_absz
end interface

interface acos
  module procedure mp_acos
end interface

interface aimag
  module procedure mp_imag
end interface

interface aint
  module procedure mp_aint
end interface

interface anint
  module procedure mp_anint
end interface

interface asin
  module procedure mp_asin
end interface

interface atan
  module procedure mp_atan
end interface

interface atan2
  module procedure mp_atan2
end interface

interface cmplx
  module procedure mp_jtoc
  module procedure mp_qtoc
  module procedure mp_ztoc
end interface

interface conjg
  module procedure mp_conjg
end interface

interface cos
  module procedure mp_cos
  module procedure mp_cosz
end interface

interface cosh
  module procedure mp_cosh
end interface

interface dble
  module procedure mp_jtod
  module procedure mp_qtod
  module procedure mp_ztod
end interface

interface dcmplx
  module procedure mp_jtox
  module procedure mp_qtox
  module procedure mp_ztox
end interface

interface exp
  module procedure mp_exp
  module procedure mp_expz
end interface

interface int
  module procedure mp_jtoi
  module procedure mp_qtoi
  module procedure mp_ztoi
end interface

interface log
  module procedure mp_log
  module procedure mp_logz
end interface

interface log10
  module procedure mp_log10
end interface

interface max
  module procedure mp_maxj
  module procedure mp_maxq
  module procedure mp_maxq3
end interface

interface min
  module procedure mp_minj
  module procedure mp_minq
  module procedure mp_minq3
end interface

interface mod
  module procedure mp_modj
  module procedure mp_modq
end interface

interface mpcmpl
  module procedure mp_jtoz
  module procedure mp_qtoz
  module procedure mp_itoz
  module procedure mp_dtoz
  module procedure mp_xtoz

  module procedure mp_atoz

  module procedure mp_jjtoz
  module procedure mp_qqtoz
  module procedure mp_iitoz
  module procedure mp_ddtoz

  module procedure mp_aatoz
end interface

interface mpcsshf
  module procedure mp_cssh
end interface

interface mpcssnf
  module procedure mp_cssn
end interface

interface mpint
  module procedure mp_qtoj
  module procedure mp_ztoj
  module procedure mp_itoj
  module procedure mp_dtoj
  module procedure mp_xtoj

  module procedure mp_atoj
end interface

interface mpnrtf
  module procedure mp_nrt
end interface

interface mpranf
  module procedure mp_rand
end interface

interface mpread
  module procedure mp_inpj
  module procedure mp_inpq
  module procedure mp_inpz
end interface

interface mpreal
  module procedure mp_jtoq
  module procedure mp_ztoq
  module procedure mp_itoq
  module procedure mp_dtoq
  module procedure mp_xtoq

  module procedure mp_atoq
end interface

interface mpwrite
  module procedure mp_outj
  module procedure mp_outq
  module procedure mp_outz
end interface

interface nint
  module procedure mp_nint
end interface

interface sign
  module procedure mp_signj
  module procedure mp_signq
end interface

interface sin
  module procedure mp_sin
  module procedure mp_sinz
end interface

interface sinh
  module procedure mp_sinh
end interface

interface sqrt
  module procedure mp_sqrtq
  module procedure mp_sqrtz
end interface

interface tan
  module procedure mp_tan
end interface

interface tanh
  module procedure mp_tanh
end interface

contains

  function mp_absj (ja)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_integer):: mp_absj
    intent (in):: ja
    integer mpnw
    mpnw = mpnwx
    call mpeq (ja%mpi, mp_absj%mpi, mpnw) 
    mp_absj%mpi(1) = abs (ja%mpi(1))
    return
  end function

  function mp_absq (qa)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_absq
    intent (in):: qa
    integer mpnw
    mpnw = mpnwx
    call mpeq (qa%mpr, mp_absq%mpr, mpnw) 
    mp_absq%mpr(1) = abs (qa%mpr(1))
    return
  end function

  function mp_absz (za)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_absz
    intent (in):: za
    type (mp_real) q1, q2, q3
    integer mpnw
    mpnw = mpnwx
    call mpmul (za%mpc, za%mpc, q1%mpr, mpnw) 
    call mpmul (za%mpc(mp41), za%mpc(mp41), q2%mpr, mpnw) 
    call mpadd (q1%mpr, q2%mpr, q3%mpr, mpnw) 
    call mpsqrt (q3%mpr, mp_absz%mpr, mpnw) 
    return
  end function

  function mp_acos (qa)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_acos
    intent (in):: qa
    type (mp_real) q1, q2, q3
    integer mpnw
    mpnw = mpnwx
    call mpdmc (1.d0, 0, q1%mpr)
    call mpmul (qa%mpr, qa%mpr, q2%mpr, mpnw) 
    call mpsub (q1%mpr, q2%mpr, q3%mpr, mpnw) 
    call mpsqrt (q3%mpr, q1%mpr, mpnw) 
    call mpang (qa%mpr, q1%mpr, mppic%mpr, mp_acos%mpr, mpnw) 
    return
  end function

  function mp_aint (qa)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_aint
    intent (in):: qa
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpinfr (qa%mpr, mp_aint%mpr, q1%mpr, mpnw) 
    return
  end function

  function mp_anint (qa)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_anint
    intent (in):: qa
    integer mpnw
    mpnw = mpnwx
    call mpnint (qa%mpr, mp_anint%mpr, mpnw) 
    return
  end function

  function mp_asin (qa)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_asin
    intent (in):: qa
    type (mp_real) q1, q2, q3
    integer mpnw
    mpnw = mpnwx
    call mpdmc (1.d0, 0, q1%mpr)
    call mpmul (qa%mpr, qa%mpr, q2%mpr, mpnw) 
    call mpsub (q1%mpr, q2%mpr, q3%mpr, mpnw) 
    call mpsqrt (q3%mpr, q1%mpr, mpnw) 
    call mpang (q1%mpr, qa%mpr, mppic%mpr, mp_asin%mpr, mpnw) 
    return
  end function

  function mp_atan (qa)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_atan
    intent (in):: qa
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpdmc (1.d0, 0, q1%mpr)
    call mpang (q1%mpr, qa%mpr, mppic%mpr, mp_atan%mpr, mpnw) 
    return
  end function

  function mp_atan2 (qa, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_atan2
    intent (in):: qa, qb
    integer mpnw
    mpnw = mpnwx
    call mpang (qb%mpr, qa%mpr, mppic%mpr, mp_atan2%mpr, mpnw) 
    return
  end function

  function mp_jtoc (ja, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    complex:: mp_jtoc
    intent (in):: ja, jb
    integer mpnw
    mpnw = mpnwx
    call mpmdc (ja%mpi, da, ia)
    call mpmdc (jb%mpi, db, ib)
    mp_jtoc = cmplx (da * 2.d0 ** ia, db * 2.d0 ** ib)
    return
  end function

  function mp_qtoc (qa, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    complex:: mp_qtoc
    intent (in):: qa, qb
    integer mpnw
    mpnw = mpnwx
    call mpmdc (qa%mpr, da, ia)
    call mpmdc (qb%mpr, db, ib)
    mp_qtoc = cmplx (da * 2.d0 ** ia, db * 2.d0 ** ib)
    return
  end function

  function mp_ztoc (za)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    complex:: mp_ztoc
    intent (in):: za
    integer mpnw
    mpnw = mpnwx
    call mpmdc (za%mpc, da, ia)
    call mpmdc (za%mpc(mp41), db, ib)
    mp_ztoc = cmplx (da * 2.d0 ** ia, db * 2.d0 ** ib)
    return
  end function

  function mp_conjg (za)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_conjg
    intent (in):: za
    integer mpnw
    mpnw = mpnwx
    call mpceq (mp4, za%mpc, mp_conjg%mpc, mpnw) 
    mp_conjg%mpc(mp41) = - za%mpc(mp41)
    return
  end function

  function mp_cos (qa)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_cos
    intent (in):: qa
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpcssn (qa%mpr, mppic%mpr, mp_cos%mpr, q1%mpr, mpnw) 
    return
  end function

  function mp_cosz (za)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_cosz
    intent (in):: za
    type (mp_real) q1, q2, q3, q4, q5, q6
    integer mpnw
    mpnw = mpnwx
    call mpeq (za%mpc(mp41), q2%mpr, mpnw) 
    q2%mpr(1) = - q2%mpr(1)
    call mpexp (q2%mpr, mpl02%mpr, q1%mpr, mpnw) 
    call mpdmc (1.d0, 0, q3%mpr)
    call mpdiv (q3%mpr, q1%mpr, q2%mpr, mpnw) 
    call mpcssn (za%mpc, mppic%mpr, q3%mpr, q4%mpr, mpnw) 
    call mpadd (q1%mpr, q2%mpr, q5%mpr, mpnw) 
    call mpmuld (q5%mpr, 0.5d0, 0, q6%mpr, mpnw) 
    call mpmul (q6%mpr, q3%mpr, mp_cosz%mpc, mpnw) 
    call mpsub (q1%mpr, q2%mpr, q5%mpr, mpnw) 
    call mpmuld (q5%mpr, 0.5d0, 0, q6%mpr, mpnw) 
    call mpmul (q6%mpr, q4%mpr, mp_cosz%mpc(mp41), mpnw) 
    return
  end function

  function mp_cosh (qa)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_cosh
    intent (in):: qa
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpcssh (qa%mpr, mpl02%mpr, mp_cosh%mpr, q1%mpr, mpnw) 
    return
  end function

  function mp_jtod (ja)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    intent (in):: ja
    double precision mp_jtod
    integer mpnw
    mpnw = mpnwx
    call mpmdc (ja%mpi, da, ia)
    mp_jtod = da * 2.d0 ** ia
    return
  end function

  function mp_qtod (qa)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    intent (in):: qa
    double precision:: mp_qtod, da
    integer mpnw
    mpnw = mpnwx
    call mpmdc (qa%mpr, da, ia)
    mp_qtod = da * 2.d0 ** ia
    return
  end function

  function mp_ztod (za)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    intent (in):: za
    double precision:: mp_ztod, da
    integer mpnw
    mpnw = mpnwx
    call mpmdc (za%mpc, da, ia)
    mp_ztod = da * 2.d0 ** ia
    return
  end function

  function mp_jtox (ja, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    complex (kdb):: mp_jtox
    intent (in):: ja, jb
    integer mpnw
    mpnw = mpnwx
    call mpmdc (ja%mpi, da, ia)
    call mpmdc (jb%mpi, db, ib)
    mp_jtox = cmplx (da * 2.d0 ** ia, db * 2.d0 ** ib, kdb)
    return
  end function

  function mp_qtox (qa, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    complex (kdb):: mp_qtox
    intent (in):: qa, qb
    integer mpnw
    mpnw = mpnwx
    call mpmdc (qa%mpr, da, ia)
    call mpmdc (qb%mpr, db, ib)
    mp_qtox = cmplx (da * 2.d0 ** ia, db * 2.d0 ** ib, kdb)
    return
  end function

  function mp_ztox (za)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    complex (kdb):: mp_ztox
    intent (in):: za
    integer mpnw
    mpnw = mpnwx
    call mpmdc (za%mpc, da, ia)
    call mpmdc (za%mpc(mp41), db, ib)
    mp_ztox = cmplx (da * 2.d0 ** ia, db * 2.d0 ** ib, kdb)
    return
  end function

  function mp_exp (qa)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_exp
    intent (in):: qa
    integer mpnw
    mpnw = mpnwx
    call mpexp (qa%mpr, mpl02%mpr, mp_exp%mpr, mpnw) 
    return
  end function

  function mp_expz (za)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_expz
    intent (in):: za
    type (mp_real) q1, q2, q3
    integer mpnw
    mpnw = mpnwx
    call mpexp (za%mpc, mpl02%mpr, q1%mpr, mpnw) 
    call mpcssn (za%mpc(mp41), mppic%mpr, q2%mpr, q3%mpr, mpnw) 
    call mpmul (q1%mpr, q2%mpr, mp_expz%mpc, mpnw) 
    call mpmul (q1%mpr, q3%mpr, mp_expz%mpc(mp41), mpnw) 
    return
  end function

  function mp_imag (za)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_imag
    intent (in):: za
    integer mpnw
    mpnw = mpnwx
    call mpeq (za%mpc(mp41), mp_imag%mpr, mpnw) 
    return
  end function

  function mp_jtoi (ja)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    integer:: mp_jtoi
    intent (in):: ja
    integer mpnw
    mpnw = mpnwx
    call mpmdc (ja%mpi, da, ia)
    mp_jtoi = da * 2.d0 ** ia
    return
  end function

  function mp_qtoi (qa)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    integer:: mp_qtoi
    intent (in):: qa
    integer mpnw
    mpnw = mpnwx
    call mpmdc (qa%mpr, da, ia)
    mp_qtoi = da * 2.d0 ** ia
    return
  end function

  function mp_ztoi (za)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    integer:: mp_ztoi
    intent (in):: za
    integer mpnw
    mpnw = mpnwx
    call mpmdc (za%mpc, da, ia)
    mp_ztoi = da * 2.d0 ** ia
    return
  end function

  function mp_log (qa)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_log
    intent (in):: qa
    integer mpnw
    mpnw = mpnwx
    call mplog (qa%mpr, mpl02%mpr, mp_log%mpr, mpnw) 
    return
  end function

  function mp_logz (za)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_logz
    intent (in):: za
    type (mp_real) q1, q2, q3, q4
    integer mpnw
    mpnw = mpnwx
    call mpmul (za%mpc, za%mpc, q1%mpr, mpnw) 
    call mpmul (za%mpc(mp41), za%mpc(mp41), q2%mpr, mpnw) 
    call mpadd (q1%mpr, q2%mpr, q3%mpr, mpnw) 
    call mplog (q3%mpr, mpl02%mpr, q4%mpr, mpnw) 
    call mpmuld (q4%mpr, 0.5d0, 0, mp_logz%mpc, mpnw) 
    call mpang (za%mpc, za%mpc(mp41), mppic%mpr, mp_logz%mpc(mp41), mpnw) 
    return
  end function

  function mp_log10 (qa)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_log10
    intent (in):: qa
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mplog (qa%mpr, mpl02%mpr, q1%mpr, mpnw) 
    call mpdiv (q1%mpr, mpl10%mpr, mp_log10%mpr, mpnw) 
    return
  end function

  function mp_maxj (ja, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_integer):: mp_maxj
    intent (in):: ja, jb
    integer mpnw
    mpnw = mpnwx
    call mpcpr (ja%mpi, jb%mpi, ic, mpnw) 
    if (ic .ge. 0) then
      call mpeq (ja%mpi, mp_maxj%mpi, mpnw) 
    else
      call mpeq (jb%mpi, mp_maxj%mpi, mpnw) 
    endif
    return
  end function

  function mp_maxq (qa, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_maxq
    intent (in):: qa, qb
    integer mpnw
    mpnw = mpnwx
    call mpcpr (qa%mpr, qb%mpr, ic, mpnw) 
    if (ic .ge. 0) then
      call mpeq (qa%mpr, mp_maxq%mpr, mpnw) 
    else
      call mpeq (qb%mpr, mp_maxq%mpr, mpnw) 
    endif
    return
  end function

  function mp_maxq3 (qa, qb, qc)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_maxq3
    intent (in):: qa, qb, qc
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpcpr (qa%mpr, qb%mpr, ic, mpnw) 
    if (ic .ge. 0) then
      call mpeq (qa%mpr, q1%mpr, mpnw) 
    else
      call mpeq (qb%mpr, q1%mpr, mpnw) 
    endif
    call mpcpr (q1%mpr, qc%mpr, ic, mpnw) 
    if (ic .ge. 0) then
      call mpeq (q1%mpr, mp_maxq3%mpr, mpnw) 
    else
      call mpeq (qc%mpr, mp_maxq3%mpr, mpnw) 
    endif
    return
  end function

  function mp_minj (ja, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_integer):: mp_minj
    intent (in):: ja, jb
    integer mpnw
    mpnw = mpnwx
    call mpcpr (ja%mpi, jb%mpi, ic, mpnw) 
    if (ic .lt. 0) then
      call mpeq (ja%mpi, mp_minj%mpi, mpnw) 
    else
      call mpeq (jb%mpi, mp_minj%mpi, mpnw) 
    endif
    return
  end function

  function mp_minq (qa, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_minq
    intent (in):: qa, qb
    integer mpnw
    mpnw = mpnwx
    call mpcpr (qa%mpr, qb%mpr, ic, mpnw) 
    if (ic .lt. 0) then
      call mpeq (qa%mpr, mp_minq%mpr, mpnw) 
    else
      call mpeq (qb%mpr, mp_minq%mpr, mpnw) 
    endif
    return
  end function

  function mp_minq3 (qa, qb, qc)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_minq3
    intent (in):: qa, qb, qc
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpcpr (qa%mpr, qb%mpr, ic, mpnw) 
    if (ic .lt. 0) then
      call mpeq (qa%mpr, q1%mpr, mpnw) 
    else
      call mpeq (qb%mpr, q1%mpr, mpnw) 
    endif
    call mpcpr (q1%mpr, qc%mpr, ic, mpnw) 
    if (ic .lt. 0) then
      call mpeq (q1%mpr, mp_minq3%mpr, mpnw) 
    else
      call mpeq (qc%mpr, mp_minq3%mpr, mpnw) 
    endif
    return
  end function

  function mp_modj (ja, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_integer):: mp_modj
    intent (in):: ja, jb
    type (mp_real) q1, q2, q3
    integer mpnw
    mpnw = mpnwx
    call mpdiv (ja%mpi, jb%mpi, q1%mpr, mpnw) 
    call mpinfr (q1%mpr, q2%mpr, q3%mpr, mpnw) 
    call mpmul (jb%mpi, q2%mpr, q1%mpr, mpnw) 
    call mpsub (ja%mpi, q1%mpr, mp_modj%mpi, mpnw) 
    return
  end function

  function mp_modq (qa, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_modq
    intent (in):: qa, qb
    type (mp_real) q1, q2, q3
    integer mpnw
    mpnw = mpnwx
    call mpdiv (qa%mpr, qb%mpr, q1%mpr, mpnw) 
    call mpinfr (q1%mpr, q2%mpr, q3%mpr, mpnw) 
    call mpmul (qb%mpr, q2%mpr, q1%mpr, mpnw) 
    call mpsub (qa%mpr, q1%mpr, mp_modq%mpr, mpnw) 
    return
  end function

  function mp_jtoz (ja)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_jtoz
    intent (in):: ja
    integer mpnw
    mpnw = mpnwx
    call mpmzc (ja%mpi, mp_jtoz%mpc)
    return
  end function

  function mp_qtoz (qa)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_qtoz
    intent (in):: qa
    integer mpnw
    mpnw = mpnwx
    call mpmzc (qa%mpr, mp_qtoz%mpc)
    return
  end function

  function mp_itoz (ia)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_itoz
    intent (in):: ia
    integer mpnw
    mpnw = mpnwx
    xa = ia
    call mpxzc (xa, mp_itoz%mpc)
    return
  end function

  function mp_rtoz (ra)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_rtoz
    intent (in):: ra
    integer mpnw
    mpnw = mpnwx
    xa = ra
    call mpxzc (xa, mp_rtoz%mpc)
    return
  end function

  function mp_ctoz (ca)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_ctoz
    intent (in):: ca
    integer mpnw
    mpnw = mpnwx
    xa = ca
    call mpxzc (xa, mp_ctoz%mpc)
    return
  end function

  function mp_dtoz (da)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_dtoz
    intent (in):: da
    integer mpnw
    mpnw = mpnwx
    xa = da
    call mpxzc (xa, mp_dtoz%mpc)
    return
  end function

  function mp_xtoz (xa)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_xtoz
    intent (in):: xa
    integer mpnw
    mpnw = mpnwx
    call mpxzc (xa, mp_xtoz%mpc)
    return
  end function

  function mp_atoz (aa)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    character*(*), intent (in):: aa
    type (mp_complex):: mp_atoz
    character*1 az(mpipl+100)
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    l = len (aa)
    do i = 1, l
      az(i) = aa(i:i)
    enddo
    call mpinpc (az, l, q1%mpr, mpnw) 
    call mpmzc (q1%mpr, mp_atoz%mpc)
    return
  end function

  function mp_jjtoz (ja, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_jjtoz
    intent (in):: ja, jb
    integer mpnw
    mpnw = mpnwx
    call mpmmpc (ja%mpi, jb%mpi, mp4, mp_jjtoz%mpc, mpnw) 
    return
  end function

  function mp_qqtoz (qa, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_qqtoz
    intent (in):: qa, qb
    integer mpnw
    mpnw = mpnwx
    call mpmmpc (qa%mpr, qb%mpr, mp4, mp_qqtoz%mpc, mpnw) 
    return
  end function

  function mp_iitoz (ia, ib)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_iitoz
    intent (in):: ia, ib
    integer mpnw
    mpnw = mpnwx
    xa = cmplx (ia, ib, kdb)
    call mpxzc (xa, mp_iitoz%mpc)
    return
  end function

  function mp_ddtoz (da, db)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_ddtoz
    intent (in):: da, db
    integer mpnw
    mpnw = mpnwx
    xa = cmplx (da, db, kdb)
    call mpxzc (xa, mp_ddtoz%mpc)
    return
  end function

  function mp_aatoz (aa, ab)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    character*(*), intent (in):: aa, ab
    type (mp_complex):: mp_aatoz
    character*1 az(mpipl+100)
    integer mpnw
    mpnw = mpnwx
    l = len (aa)
    do i = 1, l
      az(i) = aa(i:i)
    enddo
    call mpinpc (az, l, mp_aatoz%mpc, mpnw) 
    l = len (ab)
    do i = 1, l
      az(i) = ab(i:i)
    enddo
    call mpinpc (az, l, mp_aatoz%mpc(mp41), mpnw) 
    return
  end function

  subroutine mp_cssh (qa, qb, qc)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    intent (in):: qa
    intent (out):: qb, qc
    integer mpnw
    mpnw = mpnwx
    call mpcssh (qa%mpr, mpl02%mpr, qb%mpr, qc%mpr, mpnw) 
    return
  end subroutine

  subroutine mp_cssn (qa, qb, qc)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    intent (in):: qa
    intent (out):: qb, qc
    integer mpnw
    mpnw = mpnwx
    call mpcssn (qa%mpr, mppic%mpr, qb%mpr, qc%mpr, mpnw) 
    return
  end subroutine

  function mp_qtoj (qa)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_integer):: mp_qtoj
    intent (in):: qa
    type (mp_real) q1, q2
    integer mpnw
    mpnw = mpnwx
    call mpeq (qa%mpr, q1%mpr, mpnw) 
    call mpinfr (q1%mpr, mp_qtoj%mpi, q2%mpr, mpnw) 
    return
  end function

  function mp_ztoj (za)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_integer):: mp_ztoj
    intent (in):: za
    type (mp_real) q1, q2
    integer mpnw
    mpnw = mpnwx
    call mpeq (za%mpc, q1%mpr, mpnw) 
    call mpinfr (q1%mpr, mp_ztoj%mpi, q2%mpr, mpnw) 
    return
  end function

  function mp_itoj (ia)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_integer):: mp_itoj
    intent (in):: ia
    integer mpnw
    mpnw = mpnwx
    da = ia
    call mpdmc (da, 0, mp_itoj%mpi)
    return
  end function

  function mp_dtoj (da)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_integer):: mp_dtoj
    intent (in):: da
    type (mp_real) q1, q2
    integer mpnw
    mpnw = mpnwx
    call mpdmc (da, 0, q1%mpr)
    call mpinfr (q1%mpr, mp_dtoj%mpi, q2%mpr, mpnw) 
    return
  end function

  function mp_xtoj (xa)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_integer):: mp_xtoj
    intent (in):: xa
    type (mp_real) q1, q2
    integer mpnw
    mpnw = mpnwx
    da = xa
    call mpdmc (da, 0, q1%mpr)
    call mpinfr (q1%mpr, mp_xtoj%mpi, q2%mpr, mpnw) 
    return
  end function

  function mp_atoj (aa)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    character*(*), intent (in):: aa
    type (mp_integer):: mp_atoj
    character*1 az(mpipl+100)
    type (mp_real) q1, q2
    integer mpnw
    mpnw = mpnwx
    l = len (aa)
    do i = 1, l
      az(i) = aa(i:i)
    enddo
    call mpinpc (az, l, q1%mpr, mpnw) 
    call mpinfr (q1%mpr, mp_atoj%mpi, q2%mpr, mpnw) 
    return
  end function

  function mp_nrt (qa, ib)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_nrt
    intent (in):: qa, ib
    integer mpnw
    mpnw = mpnwx
    call mpnrt (qa%mpr, ib, mp_nrt%mpr, mpnw) 
    return
  end function

  function mp_rand ()
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_rand
    integer mpnw
    mpnw = mpnwx
    call mprand (mp_rand%mpr, mpnw) 
    return
  end function

  subroutine mp_inpj (iu, j1, j2, j3, j4, j5, j6, j7, j8, j9)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    intent (out):: j1, j2, j3, j4, j5, j6, j7, j8, j9
    optional:: j2, j3, j4, j5, j6, j7, j8, j9
    character*1 az(mpipl+100)
    integer mpnw
    mpnw = mpnwx
    call mpinp (iu, j1%mpi, az, mpnw) 
    if (present (j2)) call mpinp (iu, j2%mpi, az, mpnw) 
    if (present (j3)) call mpinp (iu, j3%mpi, az, mpnw) 
    if (present (j4)) call mpinp (iu, j4%mpi, az, mpnw) 
    if (present (j5)) call mpinp (iu, j5%mpi, az, mpnw) 
    if (present (j6)) call mpinp (iu, j6%mpi, az, mpnw) 
    if (present (j7)) call mpinp (iu, j7%mpi, az, mpnw) 
    if (present (j8)) call mpinp (iu, j8%mpi, az, mpnw) 
    if (present (j9)) call mpinp (iu, j9%mpi, az, mpnw) 
    return
  end subroutine

  subroutine mp_inpq (iu, q1, q2, q3, q4, q5, q6, q7, q8, q9)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    intent (out):: q1, q2, q3, q4, q5, q6, q7, q8, q9
    optional:: q2, q3, q4, q5, q6, q7, q8, q9
    character*1 az(mpipl+100)
    integer mpnw
    mpnw = mpnwx
    call mpinp (iu, q1%mpr, az, mpnw) 
    if (present (q2)) call mpinp (iu, q2%mpr, az, mpnw) 
    if (present (q3)) call mpinp (iu, q3%mpr, az, mpnw) 
    if (present (q4)) call mpinp (iu, q4%mpr, az, mpnw) 
    if (present (q5)) call mpinp (iu, q5%mpr, az, mpnw) 
    if (present (q6)) call mpinp (iu, q6%mpr, az, mpnw) 
    if (present (q7)) call mpinp (iu, q7%mpr, az, mpnw) 
    if (present (q8)) call mpinp (iu, q8%mpr, az, mpnw) 
    if (present (q9)) call mpinp (iu, q9%mpr, az, mpnw) 
    return
  end subroutine

  subroutine mp_inpz (iu, z1, z2, z3, z4, z5, z6, z7, z8, z9)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    intent (out):: z1, z2, z3, z4, z5, z6, z7, z8, z9
    optional:: z2, z3, z4, z5, z6, z7, z8, z9
    character*1 az(mpipl+100)
    integer mpnw
    mpnw = mpnwx
    call mpinp (iu, z1%mpc, az, mpnw) 
    call mpinp (iu, z1%mpc(mp41), az, mpnw) 
    if (present (z2)) call mpinp (iu, z2%mpc, az, mpnw) 
    if (present (z2)) call mpinp (iu, z2%mpc(mp41), az, mpnw) 
    if (present (z3)) call mpinp (iu, z3%mpc, az, mpnw) 
    if (present (z3)) call mpinp (iu, z3%mpc(mp41), az, mpnw) 
    if (present (z4)) call mpinp (iu, z4%mpc, az, mpnw) 
    if (present (z4)) call mpinp (iu, z4%mpc(mp41), az, mpnw) 
    if (present (z5)) call mpinp (iu, z5%mpc, az, mpnw) 
    if (present (z5)) call mpinp (iu, z5%mpc(mp41), az, mpnw) 
    if (present (z6)) call mpinp (iu, z6%mpc, az, mpnw) 
    if (present (z6)) call mpinp (iu, z6%mpc(mp41), az, mpnw) 
    if (present (z7)) call mpinp (iu, z7%mpc, az, mpnw) 
    if (present (z7)) call mpinp (iu, z7%mpc(mp41), az, mpnw) 
    if (present (z8)) call mpinp (iu, z8%mpc, az, mpnw) 
    if (present (z8)) call mpinp (iu, z8%mpc(mp41), az, mpnw) 
    if (present (z9)) call mpinp (iu, z9%mpc, az, mpnw) 
    if (present (z9)) call mpinp (iu, z9%mpc(mp41), az, mpnw) 
    return
  end subroutine

  function mp_jtoq (ja)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_jtoq
    intent (in):: ja
    integer mpnw
    mpnw = mpnwx
    call mpeq (ja%mpi, mp_jtoq%mpr, mpnw) 
    return
  end function

  function mp_ztoq (za)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_ztoq
    intent (in):: za
    integer mpnw
    mpnw = mpnwx
    call mpeq (za%mpc, mp_ztoq%mpr, mpnw) 
    return
  end function

  function mp_itoq (ia)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_itoq
    intent (in):: ia
    integer mpnw
    mpnw = mpnwx
    da = ia
    call mpdmc (da, 0, mp_itoq%mpr)
    return
  end function

  function mp_dtoq (da)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_dtoq
    intent (in):: da
    integer mpnw
    mpnw = mpnwx
    call mpdmc (da, 0, mp_dtoq%mpr)
    return
  end function

  function mp_xtoq (xa)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_xtoq
    intent (in):: xa
    integer mpnw
    mpnw = mpnwx
    da = xa
    call mpdmc (da, 0, mp_xtoq%mpr)
    return
  end function

  function mp_atoq (aa)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    character*(*), intent (in):: aa
    type (mp_real):: mp_atoq
    character*1 az(mpipl+100)
    integer mpnw
    mpnw = mpnwx
    l = len (aa)
    do i = 1, l
      az(i) = aa(i:i)
    enddo
    call mpdexc (az, l, mp_atoq%mpr, mpnw) 
    return
  end function

  subroutine mp_outj (iu, j1, j2, j3, j4, j5, j6, j7, j8, j9)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    intent (in):: j1, j2, j3, j4, j5, j6, j7, j8, j9
    optional:: j2, j3, j4, j5, j6, j7, j8, j9
    character*1 az(mpipl+100)
    integer mpnw
    mpnw = mpnwx
    call mpout (iu, j1%mpi, mpoud, az, mpnw) 
    if (present (j2)) call mpout (iu, j2%mpi, mpoud, az, mpnw) 
    if (present (j3)) call mpout (iu, j3%mpi, mpoud, az, mpnw) 
    if (present (j4)) call mpout (iu, j4%mpi, mpoud, az, mpnw) 
    if (present (j5)) call mpout (iu, j5%mpi, mpoud, az, mpnw) 
    if (present (j6)) call mpout (iu, j6%mpi, mpoud, az, mpnw) 
    if (present (j7)) call mpout (iu, j7%mpi, mpoud, az, mpnw) 
    if (present (j8)) call mpout (iu, j8%mpi, mpoud, az, mpnw) 
    if (present (j9)) call mpout (iu, j9%mpi, mpoud, az, mpnw) 
     return
  end subroutine

  subroutine mp_outq (iu, q1, q2, q3, q4, q5, q6, q7, q8, q9)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    intent (in):: q1, q2, q3, q4, q5, q6, q7, q8, q9
    optional:: q2, q3, q4, q5, q6, q7, q8, q9
    character*1 az(mpipl+100)
    integer mpnw
    mpnw = mpnwx
    call mpout (iu, q1%mpr, mpoud, az, mpnw) 
    if (present (q2)) call mpout (iu, q2%mpr, mpoud, az, mpnw) 
    if (present (q3)) call mpout (iu, q3%mpr, mpoud, az, mpnw) 
    if (present (q4)) call mpout (iu, q4%mpr, mpoud, az, mpnw) 
    if (present (q5)) call mpout (iu, q5%mpr, mpoud, az, mpnw) 
    if (present (q6)) call mpout (iu, q6%mpr, mpoud, az, mpnw) 
    if (present (q7)) call mpout (iu, q7%mpr, mpoud, az, mpnw) 
    if (present (q8)) call mpout (iu, q8%mpr, mpoud, az, mpnw) 
    if (present (q9)) call mpout (iu, q9%mpr, mpoud, az, mpnw) 
     return
  end subroutine

  subroutine mp_outz (iu, z1, z2, z3, z4, z5, z6, z7, z8, z9)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    intent (in):: z1, z2, z3, z4, z5, z6, z7, z8, z9
    optional:: z2, z3, z4, z5, z6, z7, z8, z9
    character*1 az(mpipl+100)
    integer mpnw
    mpnw = mpnwx
    call mpout (iu, z1%mpc, mpoud, az, mpnw) 
    call mpout (iu, z1%mpc(mp41), mpoud, az, mpnw) 
    if (present (z2)) call mpout (iu, z2%mpc, mpoud, az, mpnw) 
    if (present (z2)) call mpout (iu, z2%mpc(mp41), mpoud, az, mpnw) 
    if (present (z3)) call mpout (iu, z3%mpc, mpoud, az, mpnw) 
    if (present (z3)) call mpout (iu, z3%mpc(mp41), mpoud, az, mpnw) 
    if (present (z4)) call mpout (iu, z4%mpc, mpoud, az, mpnw) 
    if (present (z4)) call mpout (iu, z4%mpc(mp41), mpoud, az, mpnw) 
    if (present (z5)) call mpout (iu, z5%mpc, mpoud, az, mpnw) 
    if (present (z5)) call mpout (iu, z5%mpc(mp41), mpoud, az, mpnw) 
    if (present (z6)) call mpout (iu, z6%mpc, mpoud, az, mpnw) 
    if (present (z6)) call mpout (iu, z6%mpc(mp41), mpoud, az, mpnw) 
    if (present (z7)) call mpout (iu, z7%mpc, mpoud, az, mpnw) 
    if (present (z7)) call mpout (iu, z7%mpc(mp41), mpoud, az, mpnw) 
    if (present (z8)) call mpout (iu, z8%mpc, mpoud, az, mpnw) 
    if (present (z8)) call mpout (iu, z8%mpc(mp41), mpoud, az, mpnw) 
    if (present (z9)) call mpout (iu, z9%mpc, mpoud, az, mpnw) 
    if (present (z9)) call mpout (iu, z9%mpc(mp41), mpoud, az, mpnw) 
     return
  end subroutine

  function mp_nint (qa)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_integer):: mp_nint
    intent (in):: qa
    integer mpnw
    mpnw = mpnwx
    call mpnint (qa%mpr, mp_nint%mpi, mpnw) 
    return
  end function

  function mp_ztor (za)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    intent (in):: za
    real:: mp_ztor
    integer mpnw
    mpnw = mpnwx
    call mpmdc (za%mpc, da, ia)
    mp_ztor = da * 2.d0 ** ia
    return
  end function

  function mp_signj (ja, jb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_integer):: mp_signj
    intent (in):: ja, jb
    integer mpnw
    mpnw = mpnwx
    call mpeq (ja%mpi, mp_signj%mpi, mpnw) 
    mp_signj%mpi(1) = sign (mp_signj%mpi(1), jb%mpi(1))
    return
  end function

  function mp_signq (qa, qb)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_signq
    intent (in):: qa, qb
    integer mpnw
    mpnw = mpnwx
    call mpeq (qa%mpr, mp_signq%mpr, mpnw) 
    mp_signq%mpr(1) = sign (mp_signq%mpr(1), qb%mpr(1))
    return
  end function

  function mp_sin (qa)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_sin
    intent (in):: qa
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpcssn (qa%mpr, mppic%mpr, q1%mpr, mp_sin%mpr, mpnw) 
    return
  end function

  function mp_sinz (za)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_sinz
    intent (in):: za
    type (mp_real) q1, q2, q3, q4, q5, q6
    integer mpnw
    mpnw = mpnwx
    call mpeq (za%mpc(mp41), q2%mpr, mpnw) 
    q2%mpr(1) = - q2%mpr(1)
    call mpexp (q2%mpr, mpl02%mpr, q1%mpr, mpnw) 
    call mpdmc (1.d0, 0, q3%mpr)
    call mpdiv (q3%mpr, q1%mpr, q2%mpr, mpnw) 
    call mpcssn (za%mpc, mppic%mpr, q3%mpr, q4%mpr, mpnw) 
    call mpadd (q1%mpr, q2%mpr, q5%mpr, mpnw) 
    call mpmuld (q5%mpr, 0.5d0, 0, q6%mpr, mpnw) 
    call mpmul (q6%mpr, q4%mpr, mp_sinz%mpc, mpnw) 
    call mpsub (q1%mpr, q2%mpr, q5%mpr, mpnw) 
    call mpmuld (q5%mpr, -0.5d0, 0, q6%mpr, mpnw) 
    call mpmul (q6%mpr, q3%mpr, mp_sinz%mpc(mp41), mpnw) 
    return
  end function

  function mp_sinh (qa)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_sinh
    intent (in):: qa
    type (mp_real) q1
    integer mpnw
    mpnw = mpnwx
    call mpcssh (qa%mpr, mpl02%mpr, q1%mpr, mp_sinh%mpr, mpnw) 
    return
  end function

  function mp_sqrtq (qa)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_sqrtq
    intent (in):: qa
    integer mpnw
    mpnw = mpnwx
    call mpsqrt (qa%mpr, mp_sqrtq%mpr, mpnw) 
    return
  end function

  function mp_sqrtz (za)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_complex):: mp_sqrtz
    intent (in):: za
    integer mpnw
    mpnw = mpnwx
    call mpcsqt (mp4, za%mpc, mp_sqrtz%mpc, mpnw) 
    return
  end function

  function mp_tan (qa)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_tan
    intent (in):: qa
    type (mp_real) q1, q2
    integer mpnw
    mpnw = mpnwx
    call mpcssn (qa%mpr, mppic%mpr, q1%mpr, q2%mpr, mpnw) 
    call mpdiv (q2%mpr, q1%mpr, mp_tan%mpr, mpnw) 
    return
  end function

  function mp_tanh (qa)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_tanh
    intent (in):: qa
    type (mp_real) q1, q2
    integer mpnw
    mpnw = mpnwx
    call mpcssh (qa%mpr, mpl02%mpr, q1%mpr, q2%mpr, mpnw) 
    call mpdiv (q2%mpr, q1%mpr, mp_tanh%mpr, mpnw) 
    return
  end function

end module

!   This contains defines bessel, besselexp, erf, erfc and gamma functions.

module mpfunsubmod
use mpfunmod
use mpdefmod
use mprealmod
use mpgenmod
private mp_bessel, mp_besselexp, mp_erf, mp_erfc, mp_gamma
integer, private:: kdb
parameter (kdb = kind (0.d0))

interface bessel
  module procedure mp_bessel
end interface

interface besselexp
  module procedure mp_besselexp
end interface

interface erf
  module procedure mp_erf
end interface

interface erfc
  module procedure mp_erfc
end interface

interface gamma
  module procedure mp_gamma
end interface

contains

  function mp_bessel (qa)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_bessel
    intent (in):: qa
    call mpbessel (qa, mp_bessel)
    return
  end function

  function mp_besselexp (qa)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_besselexp
    intent (in):: qa
    call mpbesselexp (qa, mp_besselexp)
    return
  end function

  function mp_erf (qa)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_erf
    intent (in):: qa
    call mperf (qa, mp_erf)
    return
  end function

  function mp_erfc (qa)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_erfc
    intent (in):: qa
    call mperfc (qa, mp_erfc)
    return
  end function

  function mp_gamma (qa)
    implicit real*8 (d), type (mp_integer) (j), &
      type (mp_real) (q), complex (kdb) (x), type (mp_complex) (z)
    type (mp_real):: mp_gamma
    intent (in):: qa
    call mpgamma (qa, mp_gamma)
    return
  end function

subroutine mpbessel (t, z)

!   This evaluates the function BesselI (0, t).

implicit none
integer i, ndp, nwords
type (mp_real) eps, tsum, t, t1, t2, z

call mpgetprecwords (nwords)
eps = mpreal (2.d0) ** (-48 * nwords - 1)
ndp = (nwords - 2) * 7.224719896d0

!   Select either the direct or the asymptotic series.

if (0.85d0 * t < ndp) then
  tsum = 1.d0
  t1 = 1.d0
  t2 = t ** 2

  do i = 1, 1000000000
    t1 = t1 * t2 / (4.d0 * dble (i) ** 2)
    if (t1 < eps) goto 100
    tsum = tsum + t1
  enddo

  write (6, *) 'bessel: loop overflow 1'
  tsum = 0.d0

100 continue

  t1 = tsum
else
  tsum = 1.d0
  t1 = 1.d0

  do i = 1, 1000000000
    t2 = t1
    t1 = t1 * (2.d0 * i - 1.d0) ** 2 / (8.d0 * i * t)
    tsum = tsum + t1
    if (t1 < eps) goto 110
    if (t1 > t2) then
      write (6, *) 'bessel: t1 > t2; t ='
      call mpwrite (6, t)
      tsum = 0.d0
      goto 110
    endif
  enddo

  write (6, *) 'bessel: loop overflow 2'
  tsum = 0.d0

110 continue

  t1 = tsum * exp (t) / sqrt (2.d0 * mppic * t)
endif

z = t1
return
end subroutine

subroutine mpbesselexp (t, z)

!   This evaluates the function BesselI (0, t) / exp (t).

implicit none
integer i, ndp, nwords
type (mp_real) eps, tsum, t, t1, t2, z

call mpgetprecwords (nwords)
eps = mpreal (2.d0) ** (-48 * nwords - 1)
ndp = (nwords - 2) * 7.224719896d0

!   Select either the direct or the asymptotic series.

if (0.85d0 * t < ndp) then
  tsum = 1.d0
  t1 = 1.d0
  t2 = t ** 2

  do i = 1, 1000000000
    t1 = t1 * t2 / (4.d0 * dble (i) ** 2)
    if (t1 < eps) goto 100
    tsum = tsum + t1
  enddo

  write (6, *) 'besselexp: loop overflow 1'
  tsum = 0.d0

100 continue

  t1 = tsum / exp (t)
else
  tsum = 1.d0
  t1 = 1.d0

  do i = 1, 1000000000
    t2 = t1
    t1 = t1 * (2.d0 * i - 1.d0) ** 2 / (8.d0 * i * t)
    tsum = tsum + t1
    if (t1 < eps) goto 110
    if (t1 > t2) then
      write (6, *) 'besselexp: t1 > t2; t ='
      call mpwrite (6, t)
      tsum = 0.d0
      goto 110
    endif
  enddo

  write (6, *) 'besselexp: loop overflow 2'
  tsum = 0.d0

110 continue

  t1 = tsum / sqrt (2.d0 * mppic * t)
endif

z = t1
return
end subroutine

subroutine mperf (t, z)

!   Computes erf = Int_0^a 2/Sqrt(pi) * e^(-t^2)

implicit none
integer i, nw
real*8 ds
type (mp_real) eps, t, t0, t1, t2, t3, t4, z

if (abs (t) > 1d-4) then
  z = 1.d0 - erfc (t)
else
  call mpgetprecwords (nw)
  eps = mpreal (0.5d0) ** (mpnbt * nw + mpnbt)
  t0 = 1.d0
  t1 = 1.d0
  t2 = t ** 2
  t3 = 1.d0
  ds = 1.d0

  do i = 1, 1000000000
    ds = - ds
    t3 = dble (i) * t3
    t1 = t1 * t2
    t4 = ds * t1 / (dble (2 * i + 1) * t3)
    t0 = t0 + t4
    if (abs (t4) < eps) goto 100
  enddo

  write (6, *) 'erf: loop end error'
  t0 = 0.d0

100 continue

  z = 2.d0 / sqrt (mppic) * t * t0
endif

return
end subroutine

subroutine mperfc (t, z)

!   Computes erfc(a) = 1 - Int_0^a 2/sqrt(pi) * e^(-t^2) dt.

!   This algorithm is presented in Richard Crandall's book "Topics in
!   Advanced Scientific Computation", pg 82.  Crandall in turn references
!   a 1968 paper by Chiarella and Reichel.

  implicit none
  integer i, j, k, n, ndp1, ndps, ntab, nwks, nwords
  type (mp_real) eps, f, t, t1, t2, t3, t4, t5, z
  real*8 alpha, d1, d2, dpi, dlog10, dlog2
  type (mp_real) etab (:)
  allocatable etab
  save ndps, ntab, nwks, alpha, etab
  data ntab/0/

  call mpgetprecwords (nwords)
  eps = mpreal(2.d0) ** (-48 * nwords - 1)
  ndp1 = (nwords - 2) * 7.224719896d0
  dpi = acos (-1.d0)
  dlog10 = log (10.d0)
  dlog2 = log (2.d0)
  d1 = t
  if (d1 > 10000.d0) then
    z = 0.d0
    goto 200
  endif
  d2 = dpi / d1

  if (ntab == 0 .or. ndp1 > ndps .or. nwords > nwks .or. d2 < alpha) then

!   On the first call, or if working precision has been increased, or if
!   the argument exceeds a certain value, recalculate alpha and the etab table.

    ndps = ndp1
    nwks = nwords
    if (ntab > 0) deallocate (etab)

!   Multiply d1 (new alpha) by 0.95 (so we won't need to recalculate so often),
!   then round to some nice 6-bit rational.

    d1 = 0.95d0 * min (dpi / sqrt (ndp1 * dlog10), d2)
    n = abs (int (log (d1) / dlog2)) + 1
    alpha = 0.5d0 ** (n + 6) * anint (d1 * 2.d0 ** (n + 6))
    ntab = sqrt (ndp1 * dlog10) / alpha + 1.d0

!   Make sure that (alpha * ntab)^2 can be represented exactly in DP.
!   I don't think this will ever be a problem, but check just in case.

    d2 = 2.d0 * (6.d0 + log (dble (ntab)) / dlog2)
    if (d2 > 53.d0) then
      write (6, *) 'mperfcx: error; contact author'
      stop
    endif

!    write (6, *) 'alpha, ntab, bits =', alpha, ntab, d2

    allocate (etab(ntab))

!   Calculate table of exp(-k^2*alpha^2).

    t1 = - alpha ** 2
    t2 = exp (t1)
    t3 = t2 ** 2
    t4 = 1.d0

    do i = 1, ntab
      t4 = t2 * t4
      etab(i) = t4
      t2 = t2 * t3
    enddo
  endif

  if (t == 0.d0) then
    z = 1.d0
    goto 200
  endif

  t1 = 0.d0
  t2 = t ** 2
  t3 = exp (-t2)

  do k = 1, ntab

    t5 = etab(k) / (k ** 2 * alpha ** 2 + t2)
    t1 = t1 + t5
    if (abs (t5) < eps) goto 110
  enddo

110 continue

z = t3 * alpha * t / mppic * (1.d0 / t2 + 2.d0 * t1) &
       + 2.d0 / (1.d0 - exp (2.d0 * mppic * t / alpha))

200 continue

  return
end subroutine

subroutine mpgamma (t, z)

!   This evaluates the gamma function, using an algorithm of R. W. Potter.

implicit none
integer i, j, k, ndp, neps, nt, nwords
double precision alpha, con1, con2, d1, d2
parameter (con1 = 1.151292547d0, con2 = 1.974476770d0)
type (mp_real) eps, sum1, sum2, t, t1, t2, t3, t4, tn, z

call mpgetprecwords (nwords)
neps = (-nwords - 1) * 7.224719896d0
ndp = (nwords - 1) * 7.224719896d0
eps = mpreal(2.d0) ** (-24*nwords - 24)

!   Handle special arguments.

if (abs (t) > 1.d8) then
  write (6, *) 'gamma: argument too large'
  goto 120
elseif (t == anint (t) .and. t <= 0.d0) then
    write (6, *) 'gamma: invalid argument'
    z = 0.d0
    goto 120
endif

if (t > 0.d0) then
  nt = dble (t)
  if (t == dble (nt)) nt = nt - 1
  t1 = 1.d0

  do i = 1, nt
    t1 = t1 * (t - dble (i))
  enddo

  tn = t - dble (nt)
  if (t == aint (t)) then
    z = t1
    goto 120
  endif
else
  nt = 1 - t
  t1 = 1.d0

  do i = 0, nt - 1
    t1 = t1 / (t + dble (i))
  enddo

  tn = t + dble (nt)
endif

!   Calculate alpha, then take the next highest integer value, so that
!   d2 = 0.25 * alpha^2 can be calculated exactly in double precision.

alpha = aint (con1 * ndp + 1.d0)
t2 = tn
d2 = 0.25d0 * alpha**2
t3 = 1.d0 / t2
sum1 = t3

!   Evaluate the series with t, terminating when t3 < sum1 * epsilon.

do j = 1, 1000000000
  t3 = t3 * d2 / (j * (t2 + j))
  sum1 = sum1 + t3
  if (abs (t3) < abs (sum1) * eps) goto 100
enddo

write (6, *) 'gamma: loop overflow 1'
sum1 = 0.d0

100 continue

sum1 = t2 * (0.5d0 * alpha) ** t2 * sum1
t2 = - tn
t3 = 1.d0 / t2
sum2 = t3

!   Evaluate the same series with -t, terminating when t3 < sum1 * epsilon.

do j = 1, 1000000000
  t3 = t3 * d2 / (j * (t2 + j))
  sum2 = sum2 + t3
  if (abs (t3) < abs (sum2) * eps) goto 110
enddo

write (6, *) 'gamma: loop overflow 2'
sum2 = 0.d0

110 continue

sum2 = t2 * (0.5d0 * alpha) ** t2 * sum2

!   Conclude with this square root expression.

z = t1 * sqrt (mppic * sum1 / (tn * sin (mppic * tn) * sum2))

120 continue

return
end subroutine

end module

module mpmodule
use mpfunmod
use mpintmod
use mprealmod
use mpcmpmod
use mpgenmod
use mpfunsubmod
end module
