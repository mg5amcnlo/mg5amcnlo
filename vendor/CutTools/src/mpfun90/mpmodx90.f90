!  mpmodx90.f

!  This Fortran-90 code provides a basic translation facility for an "mp_realx"
!  datatype, which is an extra-high precision (typically double-long) variant of
!  the mp_real datatype.  The four arithmetic operations are defined between
!  two mp_realx variables, between a mp_realx variable and a double precision
!  variable, and between a mp_realx variable and a mp_real variable.
!  Comparison operations between two mp_realx variables, and a few basic
!  initrinsics are also defined here.  This satisfies the needs of the F-90
!  tquaderf and tquadts codes only -- it is not a complete package.

!  Note that these routines do NOT automatically change the working precision
!  level to extra-high precision, because of the overhead of saving, setting and
!  restoring precision with each individual call.  Instead, this is done in the
!  user program at the beginning and end of a section of extra-high precision
!  computation, by using calls to mpsetprec or mpsetprecwords.  See the programs
!  tquaderf.f and tquadts.f for some examples of this usage.

!  If order to use these routines, subroutine mpinitx (see below) must be
!  called immediately after the call to mpinit in the user's main program.

!   David H Bailey    2004-08-16

!  This software is provided for research purposes only.
!  Commercial usage requires license agreement.

!  This work was supported by the Director, Office of Science, Division
!  of Mathematical, Information, and Computational Sciences of the
!  U.S. Department of Energy under contract number DE-AC03-76SF00098.

module mpmodulex
use mpmodule
implicit none

!   mpiplx is the maximum precision level, in digits.

integer mpiplx
parameter (mpiplx = 4000)

!   NOTE:  This code should not be changed below this point.

integer mpwdsx, mpw4
parameter (mpwdsx = (mpiplx-1) / 7.224719896d0 + 2.d0, mpw4 = mpwdsx + 4)

type mp_realx
  sequence
  real*4 mpr(mpw4)
end type

type (mp_realx), public:: mpl02x, mpl10x, mppicx, mpepsx
private mpadd_xx, mpadd_xq, mpadd_qx, mpadd_xd, mpadd_dx, mpsub_xx, &
  mpsub_xq, mpsub_qx, mpsub_xd, mpsub_dx, mpneg_x, mpmul_xx, mpmul_xq, &
  mpmul_qx, mpmul_xd, mpmul_dx, mpdiv_xx, mpdiv_xq, mpdiv_qx, mpdiv_xd, &
  mpdiv_dx, mpexp_xi,  mpeq_xx, mpeq_xq, mpeq_qx, mpeq_xd, mpeq_dx, &
  mpeqt_xx, mpnet_xx, mplet_xx, mpget_xx, mpltt_xx, mpgtt_xx, mp_absx, &
  mp_xtod, mp_maxx, mp_minx, mp_inpx, mp_outx, mp_xtoq, mp_dtox, mp_qtox, &
  mp_sqrtx

! mp_realx operator extension interface blocks.

  interface operator (+)
      module procedure mpadd_xx
      module procedure mpadd_xq
      module procedure mpadd_qx
      module procedure mpadd_xd
      module procedure mpadd_dx
  end interface

  interface operator (-)
      module procedure mpsub_xx
      module procedure mpsub_xq
      module procedure mpsub_qx
      module procedure mpsub_xd
      module procedure mpsub_dx
! negation
      module procedure mpneg_x
  end interface

  interface operator (*)
      module procedure mpmul_xx
      module procedure mpmul_xq
      module procedure mpmul_qx
      module procedure mpmul_xd
      module procedure mpmul_dx
  end interface

  interface operator (/)
      module procedure mpdiv_xx
      module procedure mpdiv_xq
      module procedure mpdiv_qx
      module procedure mpdiv_xd
      module procedure mpdiv_dx
  end interface

  interface operator (**)
      module procedure mpexp_xi
  end interface

  interface assignment (=)
      module procedure mpeq_xx
      module procedure mpeq_xq
      module procedure mpeq_qx
      module procedure mpeq_xd
      module procedure mpeq_dx
  end interface

  interface operator (.eq.)
      module procedure mpeqt_xx
  end interface

  interface operator (.ne.)
      module procedure mpnet_xx
  end interface

  interface operator (.le.)
      module procedure mplet_xx
  end interface

  interface operator (.ge.)
      module procedure mpget_xx
  end interface

  interface operator (.lt.)
      module procedure mpltt_xx
  end interface

  interface operator (.gt.)
      module procedure mpgtt_xx
  end interface

  interface abs
    module procedure mp_absx
  end interface

  interface dble
    module procedure mp_xtod
  end interface

  interface max
    module procedure mp_maxx
  end interface

  interface min
    module procedure mp_minx
  end interface

  interface mpread
    module procedure mp_inpx
  end interface

  interface mpwrite
    module procedure mp_outx
  end interface

  interface mpreal
    module procedure mp_xtoq
  end interface

  interface mprealx
    module procedure mp_dtox
    module procedure mp_qtox
  end interface

  interface sqrt
    module procedure mp_sqrtx
  end interface
contains

  subroutine mpinitx (n_mpiplx)
!  MPINIT must be called at the start of execution in the user's main program,
!  immediately after the call to mpinit.  It sets the extra-high precision level
!  and epsilon level, and computes extra-high precision versions of the
!  constants Pi, Log(2) and Log(10).

!  The arguments are as follows:
!  n_mpiplx: integer, optional.  Extra-high working precision level, in digits.
!     Must not exceed mpiplx, which is set near the start of this file.
!     If n_mpiplx is not present, then precision level is set to mpiplx.

      implicit none
      integer, intent(in), optional :: n_mpiplx
      integer iconst_temp
      real t0(mpw4+1), t1(mpw4+1), t2(mpw4+1), t3(mpw4+1), t4(mpw4+1)
      integer mpnw

      if (present (n_mpiplx)) then
         if (n_mpiplx .gt. mpiplx) then
            write(mpldb, *) '*** MPINITX: new precision level is too high'
            stop
         endif
         new_mpipl = n_mpiplx
         new_mpwds = n_mpiplx / 7.224719896d0 + 2.d0
         mpnwx = new_mpwds
      else
        new_mpipl = mpiplx
        new_mpwds = mpwdsx
        mpnwx = new_mpwds
      endif
      mpier = 0
      mpnw = mpnwx + 1
      call mppi (t1, mpnw) 
      call mpdmc (2.d0, 0, t0)
      call mplog (t0, t2, t2, mpnw) 
      call mpdmc (10.d0, 0, t0)
      call mplog (t0, t2, t3, mpnw) 
      call mpnpwr (t0, mpiep, t4, mpnw) 
      mpnw = mpnwx
      call mpeq (t1, mppicx%mpr, mpnw) 
      call mpeq (t2, mpl02x%mpr, mpnw) 
      call mpeq (t3, mpl10x%mpr, mpnw) 
      call mpeq (t4, mpepsx%mpr, mpnw) 
  end subroutine

! Additions
  type (mp_realx) function mpadd_xx (a, b)
      type (mp_realx), intent(in) :: a, b
      integer mpnw
      mpnw = mpnwx
      call mpadd (a%mpr, b%mpr, mpadd_xx%mpr, mpnw) 
  end function

  type (mp_realx) function mpadd_xq (a, b)
      type (mp_realx), intent(in) :: a
      type (mp_real), intent(in) :: b
      integer mpnw
      mpnw = mpnwx
      call mpadd (a%mpr, b%mpr, mpadd_xq%mpr, mpnw) 
  end function

  type (mp_realx) function mpadd_qx (a, b)
      type (mp_real), intent(in) :: a
      type (mp_realx), intent(in) :: b
      integer mpnw
      mpnw = mpnwx
      call mpadd (a%mpr, b%mpr, mpadd_qx%mpr, mpnw) 
  end function

  type (mp_realx) function mpadd_xd (a, b)
      type (mp_realx), intent(in) :: a
      real*8, intent(in) :: b
      type (mp_realx) t1
      integer mpnw
      mpnw = mpnwx
      call mpdmc (b, 0, t1%mpr)
      call mpadd (a%mpr, t1%mpr, mpadd_xd%mpr, mpnw) 
  end function

  type (mp_realx) function mpadd_dx (a, b)
      real*8, intent(in) :: a
      type (mp_realx), intent(in) :: b
      type (mp_realx) t1
      integer mpnw
      mpnw = mpnwx
      call mpdmc (a, 0, t1%mpr)
      call mpadd (b%mpr, t1%mpr, mpadd_dx%mpr, mpnw) 
  end function

! Subtractions
  type (mp_realx) function mpsub_xx (a, b)
      type (mp_realx), intent(in) :: a, b
      integer mpnw
      mpnw = mpnwx
      call mpsub (a%mpr, b%mpr, mpsub_xx%mpr, mpnw) 
  end function

  type (mp_realx) function mpsub_xq (a, b)
      type (mp_realx), intent(in) :: a
      type (mp_real), intent(in) :: b
      integer mpnw
      mpnw = mpnwx
      call mpsub (a%mpr, b%mpr, mpsub_xq%mpr, mpnw) 
  end function

  type (mp_realx) function mpsub_qx (a, b)
      type (mp_real), intent(in) :: a
      type (mp_realx), intent(in) :: b
      integer mpnw
      mpnw = mpnwx
      call mpsub (a%mpr, b%mpr, mpsub_qx%mpr, mpnw) 
  end function

  type (mp_realx) function mpsub_xd (a, b)
      type (mp_realx), intent(in) :: a
      real*8, intent(in) :: b
      type (mp_real) t1
      integer mpnw
      mpnw = mpnwx
      call mpdmc (b, 0, t1%mpr)
      call mpsub (a%mpr, t1%mpr, mpsub_xd%mpr, mpnw) 
  end function

  type (mp_realx) function mpsub_dx (a, b)
      real*8, intent(in) :: a
      type (mp_realx), intent(in) :: b
      type (mp_real) t1
      integer mpnw
      mpnw = mpnwx
      call mpdmc (a, 0, t1%mpr)
      call mpsub (t1%mpr, b%mpr, mpsub_dx%mpr, mpnw) 
  end function

! Unary Minus
  type (mp_realx) function mpneg_x (a)
    type (mp_realx), intent(in) :: a
    integer mpnw
    mpnw = mpnwx
    call mpeq (a%mpr, mpneg_x%mpr, mpnw)  
    mpneg_x%mpr(1) = - mpneg_x%mpr(1)
  end function

! Multiplications
  type (mp_realx) function mpmul_xx (a, b)
      type (mp_realx), intent(in) :: a, b
      integer mpnw
      mpnw = mpnwx
      call mpmul (a%mpr, b%mpr, mpmul_xx%mpr, mpnw) 
  end function

  type (mp_realx) function mpmul_xq (a, b)
      type (mp_realx), intent(in) :: a
      type (mp_real), intent(in) :: b
      integer mpnw
      mpnw = mpnwx
      call mpmul (a%mpr, b%mpr, mpmul_xq%mpr, mpnw) 
  end function

  type (mp_realx) function mpmul_qx (a, b)
      type (mp_real), intent(in) :: a
      type (mp_realx), intent(in) :: b
      integer mpnw
      mpnw = mpnwx
      call mpmul (a%mpr, b%mpr, mpmul_qx%mpr, mpnw) 
  end function

  type (mp_realx) function mpmul_xd (a, b)
      type (mp_realx), intent(in) :: a
      real*8, intent(in) :: b
      integer mpnw
      mpnw = mpnwx
      call mpmuld (a%mpr, b, 0, mpmul_xd%mpr, mpnw) 
  end function

  type (mp_realx) function mpmul_dx (a, b)
      real*8, intent(in) :: a
      type (mp_realx), intent(in) :: b
      integer mpnw
      mpnw = mpnwx
      call mpmuld (b%mpr, a, 0, mpmul_dx%mpr, mpnw) 
  end function

! Divisions
  type (mp_realx) function mpdiv_xx (a, b)
      type (mp_realx), intent(in) :: a, b
      integer mpnw
      mpnw = mpnwx
      call mpdiv (a%mpr, b%mpr, mpdiv_xx%mpr, mpnw) 
  end function

  type (mp_realx) function mpdiv_xq (a, b)
      type (mp_realx), intent(in) :: a
      type (mp_real), intent(in) :: b
      integer mpnw
      mpnw = mpnwx
      call mpdiv (a%mpr, b%mpr, mpdiv_xq%mpr, mpnw) 
  end function

  type (mp_realx) function mpdiv_qx (a, b)
      type (mp_real), intent(in) :: a
      type (mp_realx), intent(in) :: b
      integer mpnw
      mpnw = mpnwx
      call mpdiv (a%mpr, b%mpr, mpdiv_qx%mpr, mpnw) 
  end function

  type (mp_realx) function mpdiv_xd (a, b)
      type (mp_realx), intent(in) :: a
      real*8, intent(in) :: b
      integer mpnw
      mpnw = mpnwx
      call mpdivd (a%mpr, b, 0, mpdiv_xd%mpr, mpnw) 
  end function

  type (mp_realx) function mpdiv_dx (a, b)
      real*8, intent(in) :: a
      type (mp_realx), intent(in) :: b
      type (mp_realx) t1
      integer mpnw
      mpnw = mpnwx
      call mpdmc (a, 0, t1%mpr)
      call mpdiv (t1%mpr, b%mpr, mpdiv_dx%mpr, mpnw) 
  end function

! Powers
  type (mp_realx) function mpexp_xi (a, ib)
      type (mp_realx), intent(in) :: a
      integer, intent(in) :: ib
      integer mpnw
      mpnw = mpnwx
      call mpnpwr (a%mpr, ib, mpexp_xi%mpr, mpnw) 
  end function

! Assignments
  subroutine mpeq_xx (a, b)
      type (mp_realx), intent(inout) :: a
      type (mp_realx), intent(in) :: b
      integer mpnw
      mpnw = mpnwx
      call mpeq (b%mpr, a%mpr, mpnw) 
  end subroutine

  subroutine mpeq_xq (a, b)
      type (mp_realx), intent(inout) :: a
      type (mp_real), intent(in) :: b
      integer mpnw
      mpnw = mpnwx
      call mpeq (b%mpr, a%mpr, mpnw) 
  end subroutine

  subroutine mpeq_qx (a, b)
      type (mp_real), intent(inout) :: a
      type (mp_realx), intent(in) :: b
      integer mpnw
      mpnw = mpnwx
      call mpeq (b%mpr, a%mpr, mpnw) 
  end subroutine

  subroutine mpeq_xd (a, b)
      type (mp_realx), intent(inout) :: a
      real*8, intent(in) :: b
      call mpdmc (b, 0, a%mpr)
  end subroutine

  subroutine mpeq_dx (a, b)
      real*8, intent(out) :: a
      type (mp_realx), intent(in) :: b
      double precision db
      integer ib
      call mpmdc (b%mpr, db, ib)
      a = db * 2.d0 ** ib
  end subroutine

! Equality
  logical function mpeqt_xx (a, b)
      type (mp_realx), intent(in) :: a, b
      integer ic
      integer mpnw
      mpnw = mpnwx
      call mpcpr (a%mpr, b%mpr, ic, mpnw) 
      if (ic .eq. 0) then
         mpeqt_xx = .true.
      else
         mpeqt_xx = .false.
      endif
      return
  end function

! Inequality
  logical function mpnet_xx (a, b)
      type (mp_realx), intent(in) :: a, b
      integer ic
      integer mpnw
      mpnw = mpnwx
      call mpcpr (a%mpr, b%mpr, ic, mpnw) 
      if (ic .ne. 0) then
         mpnet_xx = .true.
      else
         mpnet_xx = .false.
      endif
      return
  end function

! Less-Than-Or-Equal-To
  logical function mplet_xx (a, b)
      type (mp_realx), intent(in) :: a, b
      integer ic
      integer mpnw
      mpnw = mpnwx
      call mpcpr (a%mpr, b%mpr, ic, mpnw) 
      if (ic .le. 0) then
         mplet_xx = .true.
      else
         mplet_xx = .false.
      endif
      return
  end function

! Greater-Than-Or-Equal-To
  logical function mpget_xx (a, b)
      type (mp_realx), intent(in) :: a, b
      integer ic
      integer mpnw
      mpnw = mpnwx
      call mpcpr (a%mpr, b%mpr, ic, mpnw) 
      if (ic .ge. 0) then
         mpget_xx = .true.
      else
         mpget_xx = .false.
      endif
      return
  end function

! Less-Than
  logical function mpltt_xx (a, b)
      type (mp_realx), intent(in) :: a, b
      integer ic
      integer mpnw
      mpnw = mpnwx
      call mpcpr (a%mpr, b%mpr, ic, mpnw) 
      if (ic .lt. 0) then
         mpltt_xx = .true.
      else
         mpltt_xx = .false.
      endif
      return
  end function

! Greater-Than
  logical function mpgtt_xx (a, b)
      type (mp_realx), intent(in) :: a, b
      integer ic
      integer mpnw
      mpnw = mpnwx
      call mpcpr (a%mpr, b%mpr, ic, mpnw) 
      if (ic .gt. 0) then
         mpgtt_xx = .true.
      else
         mpgtt_xx = .false.
      endif
      return
  end function

! Absolute value
  type (mp_realx) function mp_absx (a)
    type (mp_realx), intent(in) :: a
    integer mpnw
    mpnw = mpnwx
    call mpeq (a%mpr, mp_absx%mpr, mpnw) 
    mp_absx%mpr(1) = abs (mp_absx%mpr(1))
  end function

  double precision function mp_xtod (a)
      type (mp_realx), intent (in):: a
      real*8 db
      integer ib
      call mpmdc (a%mpr, db, ib)
      mp_xtod = db * 2.d0 ** ib
  end function

  type (mp_realx) function mp_maxx (a, b)
      type (mp_realx), intent(in) :: a, b
      integer ic
      integer mpnw
      mpnw = mpnwx
      call mpcpr (a%mpr, b%mpr, ic, mpnw) 
      if (ic .ge. 0) then
         call mpeq (a%mpr, mp_maxx%mpr, mpnw) 
      else
         call mpeq (b%mpr, mp_maxx%mpr, mpnw) 
      endif
  end function

  type (mp_realx) function mp_minx (a, b)
      type (mp_realx), intent(in) :: a, b
      integer ic
      integer mpnw
      mpnw = mpnwx
      call mpcpr (a%mpr, b%mpr, ic, mpnw) 
      if (ic .le. 0) then
         call mpeq (a%mpr, mp_minx%mpr, mpnw) 
      else
         call mpeq (b%mpr, mp_minx%mpr, mpnw) 
      endif
  end function

! SQRT, etc.
  type (mp_realx) function mp_sqrtx (a)
    type (mp_realx), intent(in) :: a
    integer mpnw
    mpnw = mpnwx
    call mpsqrt (a%mpr, mp_sqrtx%mpr, mpnw) 
  end function

!  Conversion
  type (mp_real) function mp_xtoq (a)
      type (mp_realx), intent(in) :: a
      integer mpnw
      mpnw = mpnwx
      call mpeq (a%mpr, mp_xtoq%mpr, mpnw) 
  end function

  type (mp_realx) function mp_qtox (a)
      type (mp_real), intent(in) :: a
      integer mpnw
      mpnw = mpnwx
      call mpeq (a%mpr, mp_qtox%mpr, mpnw) 
  end function

  type (mp_realx) function mp_dtox (da)
      real*8, intent(in) :: da
      integer mpnw
      mpnw = mpnwx
      call mpdmc (da, 0, mp_dtox%mpr)
  end function

! Input

  subroutine mp_inpx (iu, q1, q2, q3, q4, q5, q6, q7, q8, q9)
    integer, intent(in) :: iu
    type (mp_realx), intent (out) :: q1, q2, q3, q4, q5, q6, q7, q8, q9
    optional :: q2, q3, q4, q5, q6, q7, q8, q9

    call mpinpx (iu, q1) 
    if (present (q2)) call mpinpx (iu, q2) 
    if (present (q3)) call mpinpx (iu, q3) 
    if (present (q4)) call mpinpx (iu, q4) 
    if (present (q5)) call mpinpx (iu, q5) 
    if (present (q6)) call mpinpx (iu, q6) 
    if (present (q7)) call mpinpx (iu, q7) 
    if (present (q8)) call mpinpx (iu, q8) 
    if (present (q9)) call mpinpx (iu, q9) 
    return
  end subroutine

  subroutine mpinpx (iu, a)
    integer, intent(in) :: iu
    type (mp_realx), intent(out) :: a
    integer i, l, l1
    character*200 line
    character*1 az(mpiplx+100)
    integer mpnw

    mpnw = mpnwx
    l = 0

100 continue
    read(iu, '(200a1)', end = 200) line

    do i = 200, 1, -1
      if (line(i:i) /= ' ') goto 110
    enddo

    i = 0

110 continue
    l1 = i

    do i = 1, l1
      az(l+i) = line(i:i)
    enddo

    if (line(l1:l1) /= ',') then
      l = l + l1
      goto 100
    endif
    call mpinpc (az, l, a%mpr, mpnw) 
    goto 300

200 continue
    write (6, 1)
1   format ('mpinpx: no comma terminating multiprecion input.')

300 continue
    return
  end subroutine

! Output
  subroutine mp_outx (iu, q1, q2, q3, q4, q5, q6, q7, q8, q9)
      integer, intent(in) :: iu
      type (mp_realx), intent(in) :: q1, q2, q3, q4, q5, q6, q7, q8, q9
      optional :: q2, q3, q4, q5, q6, q7, q8, q9
      call mpoutx (iu, q1) 
      if (present (q2)) call mpoutx (iu, q2) 
      if (present (q3)) call mpoutx (iu, q3) 
      if (present (q4)) call mpoutx (iu, q4) 
      if (present (q5)) call mpoutx (iu, q5) 
      if (present (q6)) call mpoutx (iu, q6) 
      if (present (q7)) call mpoutx (iu, q7) 
      if (present (q8)) call mpoutx (iu, q8) 
      if (present (q9)) call mpoutx (iu, q9) 
      return
  end subroutine

  subroutine mpoutx (iu, q)
      integer, intent(in) :: iu
      type (mp_realx), intent(in) :: q
      integer i, l
      character*1 az(mpiplx+100)
      integer mpnw
      mpnw = mpnwx
      call mpoutc (q%mpr, az, l, mpnw) 
      az(l+1) = ','
      write(iu, '(78A1)') (az(i), i = 1, l+1)
  end subroutine

end module mpmodulex
