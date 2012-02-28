!  mpmodm90.f

!  This Fortran-90 code provides a basic translation facility for an "mp_realm"
!  datatype, which is a moderate precision (typically 125 digit) variant of
!  the mp_real datatype.  The four arithmetic operations are defined between
!  two mp_realm variables, between a mp_realm variable and a double precision
!  variable, and between a mp_realm variable and a mp_real variable.
!  Comparison operations between two mp_realm variables, and a few basic
!  initrinsics are also defined here.  This satisfies the needs of the F-90
!  PSLQ3 and PSLQM3 codes only -- it is not a complete package.  The purpose
!  of these routines is to save both computation time and memory (since this
!  datatype uses only a few words).

!  Note that these routines do NOT automatically change the working precision
!  level to moderate precision, because of the overhead of saving, setting and
!  restoring precision with each individual call.  Instead, this is done in the
!  user program at the beginning and end of a section of moderate precision
!  computation, by using calls to mpsetprec or mpsetprecwords.  See the programs
!  tpslq3.f and tpslqm3.f for some examples of this usage.  Indeed, the working
!  precision should NOT be lowered for many of the routines below (example:
!  mpmul_mq, which performs moderate x full precision multiplication), since
!  such routines are normally used to return full precision results.

!   David H Bailey    2004-08-16

!  This software is provided for research purposes only.
!  Commercial usage requires license agreement.

!  This work was supported by the Director, Office of Science, Division
!  of Mathematical, Information, and Computational Sciences of the
!  U.S. Department of Energy under contract number DE-AC03-76SF00098.

module mpmodulem
use mpmodule
implicit none

!   mpiplm is the maximum precision level, in digits.

integer mpiplm
parameter (mpiplm = 125)

!   NOTE:  This code should not be changed below this point.

integer mpwdsm, mpm4
parameter (mpwdsm = (mpiplm-1) / 7.224719896d0 + 2.d0, mpm4 = mpwdsm + 4)

type mp_realm
  sequence
  real*4 mpr(mpm4)
end type

private mpadd_mm, mpadd_mq, mpadd_qm, mpadd_md, mpadd_dm, mpsub_mm, &
  mpsub_mq, mpsub_qm, mpsub_md, mpsub_dm, mpneg_m, mpmul_mm, mpmul_mq, &
  mpmul_qm, mpmul_md, mpmul_dm, mpdiv_mm, mpdiv_mq, mpdiv_qm, mpdiv_md, &
  mpdiv_dm, mpexp_mi,  mpeq_mm, mpeq_mq, mpeq_qm, mpeq_md, mpeq_dm, &
  mpeqt_mm, mpnet_mm, mplet_mm, mpget_mm, mpltt_mm, mpgtt_mm, mp_absm, &
  mp_aintm, mp_anintm, mp_mtod, mp_maxm, mp_maxm3, mp_minm, mp_minm3, &
  mp_inpm, mp_outm, mp_mtoq, mp_dtom, mp_qtom, mp_signm, mp_sqrtm

! mp_realm operator extension interface blocks.

  interface operator (+)
      module procedure mpadd_mm
      module procedure mpadd_mq
      module procedure mpadd_qm
      module procedure mpadd_md
      module procedure mpadd_dm
  end interface

  interface operator (-)
      module procedure mpsub_mm
      module procedure mpsub_mq
      module procedure mpsub_qm
      module procedure mpsub_md
      module procedure mpsub_dm
! negation
      module procedure mpneg_m
  end interface

  interface operator (*)
      module procedure mpmul_mm
      module procedure mpmul_mq
      module procedure mpmul_qm
      module procedure mpmul_md
      module procedure mpmul_dm
  end interface

  interface operator (/)
      module procedure mpdiv_mm
      module procedure mpdiv_mq
      module procedure mpdiv_qm
      module procedure mpdiv_md
      module procedure mpdiv_dm
  end interface

  interface operator (**)
      module procedure mpexp_mi
  end interface

  interface assignment (=)
      module procedure mpeq_mm
      module procedure mpeq_mq
      module procedure mpeq_qm
      module procedure mpeq_md
      module procedure mpeq_dm
  end interface

  interface operator (.eq.)
      module procedure mpeqt_mm
  end interface

  interface operator (.ne.)
      module procedure mpnet_mm
  end interface

  interface operator (.le.)
      module procedure mplet_mm
  end interface

  interface operator (.ge.)
      module procedure mpget_mm
  end interface

  interface operator (.lt.)
      module procedure mpltt_mm
  end interface

  interface operator (.gt.)
      module procedure mpgtt_mm
  end interface

  interface abs
    module procedure mp_absm
  end interface

  interface aint
     module procedure mp_aintm
  end interface

  interface anint
    module procedure mp_anintm
  end interface

  interface dble
    module procedure mp_mtod
  end interface

  interface max
    module procedure mp_maxm
    module procedure mp_maxm3
  end interface

  interface min
    module procedure mp_minm
    module procedure mp_minm3
  end interface

  interface mpread
    module procedure mp_inpm
  end interface

  interface mpwrite
    module procedure mp_outm
  end interface

  interface mpreal
    module procedure mp_mtoq
  end interface

  interface mprealm
    module procedure mp_dtom
    module procedure mp_qtom
  end interface

  interface sign
    module procedure mp_signm
  end interface

  interface sqrt
    module procedure mp_sqrtm
  end interface
contains

  subroutine mpdotdm (n, isa, a, isb, db, c)
!   This routine computes the dot product of the MPM vector A with the DP
!   vector DB, returning the MPM result in C.  This routine is used in the
!   author's customized PSLQ routine, resulting in substantial speedup.
!   The length of both the A and DB vectors is N, and ISA and ISB are the
!   skip distances between successive elements of A and DB, measured in
!   MPM words and DP words, respectively.  The DP values in DB must be
!   whole numbers, so for example they cannot be larger than 2^53.

      integer n, isa, isb
      double precision db(isb*n)
      type(mp_realm) a(isa*n), c
      integer mpnw
      mpnw = mpnwx
      call mpdotdx (n, isa * (mpwdsm + 4), a(1)%mpr, isb, db, c%mpr, mpnw) 
  end subroutine

! Additions
  type (mp_realm) function mpadd_mm (a, b)
      type (mp_realm), intent(in) :: a, b
      integer mpnw
      mpnw = mpnwx
      call mpadd (a%mpr, b%mpr, mpadd_mm%mpr, mpnw) 
  end function

  type (mp_real) function mpadd_mq (a, b)
      type (mp_realm), intent(in) :: a
      type (mp_real), intent(in) :: b
      integer mpnw
      mpnw = mpnwx
      call mpadd (a%mpr, b%mpr, mpadd_mq%mpr, mpnw) 
  end function

  type (mp_real) function mpadd_qm (a, b)
      type (mp_real), intent(in) :: a
      type (mp_realm), intent(in) :: b
      integer mpnw
      mpnw = mpnwx
      call mpadd (a%mpr, b%mpr, mpadd_qm%mpr, mpnw) 
  end function

  type (mp_realm) function mpadd_md (a, b)
      type (mp_realm), intent(in) :: a
      real*8, intent(in) :: b
      type (mp_realm) t1
      integer mpnw
      mpnw = mpnwx
      call mpdmc (b, 0, t1%mpr)
      call mpadd (a%mpr, t1%mpr, mpadd_md%mpr, mpnw) 
  end function

  type (mp_realm) function mpadd_dm (a, b)
      real*8, intent(in) :: a
      type (mp_realm), intent(in) :: b
      type (mp_realm) t1
      integer mpnw
      mpnw = mpnwx
      call mpdmc (a, 0, t1%mpr)
      call mpadd (t1%mpr, b%mpr, mpadd_dm%mpr, mpnw) 
  end function

! Subtractions
  type (mp_realm) function mpsub_mm (a, b)
      type (mp_realm), intent(in) :: a, b
      integer mpnw
      mpnw = mpnwx
      call mpsub (a%mpr, b%mpr, mpsub_mm%mpr, mpnw) 
  end function

  type (mp_real) function mpsub_mq (a, b)
      type (mp_realm), intent(in) :: a
      type (mp_real), intent(in) :: b
      integer mpnw
      mpnw = mpnwx
      call mpsub (a%mpr, b%mpr, mpsub_mq%mpr, mpnw) 
  end function

  type (mp_real) function mpsub_qm (a, b)
      type (mp_real), intent(in) :: a
      type (mp_realm), intent(in) :: b
      integer mpnw
      mpnw = mpnwx
      call mpsub (a%mpr, b%mpr, mpsub_qm%mpr, mpnw) 
  end function

  type (mp_realm) function mpsub_md (a, b)
      type (mp_realm), intent(in) :: a
      real*8, intent(in) :: b
      type (mp_realm) t1
      integer mpnw
      mpnw = mpnwx
      call mpdmc (b, 0, t1%mpr)
      call mpsub (a%mpr, t1%mpr, mpsub_md%mpr, mpnw) 
  end function

  type (mp_realm) function mpsub_dm (a, b)
      real*8, intent(in) :: a
      type (mp_realm), intent(in) :: b
      type (mp_realm) t1
      integer mpnw
      mpnw = mpnwx
      call mpdmc (a, 0, t1%mpr)
      call mpsub (t1%mpr, b%mpr, mpsub_dm%mpr, mpnw) 
  end function

! Unary Minus
  type (mp_realm) function mpneg_m (a)
    type (mp_realm), intent(in) :: a
    integer mpnw
    mpnw = mpnwx
    call mpeq (a%mpr, mpneg_m%mpr, mpnw)
    mpneg_m%mpr(1) = - mpneg_m%mpr(1)
  end function

! Multiplications
  type (mp_realm) function mpmul_mm (a, b)
      type (mp_realm), intent(in) :: a, b
      integer mpnw
      mpnw = mpnwx
      call mpmul (a%mpr, b%mpr, mpmul_mm%mpr, mpnw) 
  end function

  type (mp_real) function mpmul_mq (a, b)
      type (mp_realm), intent(in) :: a
      type (mp_real), intent(in) :: b
      integer mpnw
      mpnw = mpnwx
      call mpmul (a%mpr, b%mpr, mpmul_mq%mpr, mpnw) 
  end function

  type (mp_real) function mpmul_qm (a, b)
      type (mp_real), intent(in) :: a
      type (mp_realm), intent(in) :: b
      integer mpnw
      mpnw = mpnwx
      call mpmul (a%mpr, b%mpr, mpmul_qm%mpr, mpnw) 
  end function

  type (mp_realm) function mpmul_md (a, b)
      type (mp_realm), intent(in) :: a
      real*8, intent(in) :: b
      integer mpnw
      mpnw = mpnwx
      call mpmuld (a%mpr, b, 0, mpmul_md%mpr, mpnw) 
  end function

  type (mp_realm) function mpmul_dm (a, b)
      real*8, intent(in) :: a
      type (mp_realm), intent(in) :: b
      integer mpnw
      mpnw = mpnwx
      call mpmuld (b%mpr, a, 0, mpmul_dm%mpr, mpnw) 
  end function

! Divisions
  type (mp_realm) function mpdiv_mm (a, b)
      type (mp_realm), intent(in) :: a, b
      integer mpnw
      mpnw = mpnwx
      call mpdiv (a%mpr, b%mpr, mpdiv_mm%mpr, mpnw) 
  end function

  type (mp_real) function mpdiv_mq (a, b)
      type (mp_realm), intent(in) :: a
      type (mp_real), intent(in) :: b
      integer mpnw
      mpnw = mpnwx
      call mpdiv (a%mpr, b%mpr, mpdiv_mq%mpr, mpnw) 
  end function

  type (mp_real) function mpdiv_qm (a, b)
      type (mp_real), intent(in) :: a
      type (mp_realm), intent(in) :: b
      integer mpnw
      mpnw = mpnwx
      call mpdiv (a%mpr, b%mpr, mpdiv_qm%mpr, mpnw) 
  end function

  type (mp_realm) function mpdiv_md (a, b)
      type (mp_realm), intent(in) :: a
      real*8, intent(in) :: b
      integer mpnw
      mpnw = mpnwx
      call mpdivd (a%mpr, b, 0, mpdiv_md%mpr, mpnw) 
  end function

  type (mp_realm) function mpdiv_dm (a, b)
      real*8, intent(in) :: a
      type (mp_realm), intent(in) :: b
      type (mp_realm) t1
      integer mpnw
      mpnw = mpnwx
      call mpdmc (a, 0, t1%mpr)
      call mpdiv (t1%mpr, b%mpr, mpdiv_dm%mpr, mpnw) 
  end function

! Powers
  type (mp_realm) function mpexp_mi (a, ib)
      type (mp_realm), intent(in) :: a
      integer, intent(in) :: ib
      integer mpnw
      mpnw = mpnwx
      call mpnpwr (a%mpr, ib, mpexp_mi%mpr, mpnw) 
  end function

! Assignments
  subroutine mpeq_mm (a, b)
      type (mp_realm), intent(inout) :: a
      type (mp_realm), intent(in) :: b
      integer mpnw
      mpnw = mpnwx
      call mpeq (b%mpr, a%mpr, mpnw) 
  end subroutine

  subroutine mpeq_mq (a, b)
      type (mp_realm), intent(inout) :: a
      type (mp_real), intent(in) :: b
      integer mpnw
      mpnw = mpnwx
      call mpeq (b%mpr, a%mpr, mpnw) 
  end subroutine

  subroutine mpeq_qm (a, b)
      type (mp_real), intent(inout) :: a
      type (mp_realm), intent(in) :: b
      integer mpnw
      mpnw = mpnwx
      call mpeq (b%mpr, a%mpr, mpnw) 
  end subroutine

  subroutine mpeq_md (a, b)
      type (mp_realm), intent(inout) :: a
      real*8, intent(in) :: b
      call mpdmc (b, 0, a%mpr) 
  end subroutine

  subroutine mpeq_dm (a, b)
      real*8, intent(out) :: a
      type (mp_realm), intent(in) :: b
      double precision db
      integer ib
      call mpmdc (b%mpr, db, ib)
      a = db * 2.d0 ** ib
  end subroutine

! Equality
  logical function mpeqt_mm (a, b)
      type (mp_realm), intent(in) :: a, b
      integer ic
      integer mpnw
      mpnw = mpnwx
      call mpcpr (a%mpr, b%mpr, ic, mpnw) 
      if (ic .eq. 0) then
         mpeqt_mm = .true.
      else
         mpeqt_mm = .false.
      endif
      return
  end function

! Inequality
  logical function mpnet_mm (a, b)
      type (mp_realm), intent(in) :: a, b
      integer ic
      integer mpnw
      mpnw = mpnwx
      call mpcpr (a%mpr, b%mpr, ic, mpnw) 
      if (ic .ne. 0) then
         mpnet_mm = .true.
      else
         mpnet_mm = .false.
      endif
      return
  end function

! Less-Than-Or-Equal-To
  logical function mplet_mm (a, b)
      type (mp_realm), intent(in) :: a, b
      integer ic
      integer mpnw
      mpnw = mpnwx
      call mpcpr (a%mpr, b%mpr, ic, mpnw) 
      if (ic .le. 0) then
         mplet_mm = .true.
      else
         mplet_mm = .false.
      endif
      return
  end function

! Greater-Than-Or-Equal-To
  logical function mpget_mm (a, b)
      type (mp_realm), intent(in) :: a, b
      integer ic
      integer mpnw
      mpnw = mpnwx
      call mpcpr (a%mpr, b%mpr, ic, mpnw) 
      if (ic .ge. 0) then
         mpget_mm = .true.
      else
         mpget_mm = .false.
      endif
      return
  end function

! Less-Than
  logical function mpltt_mm (a, b)
      type (mp_realm), intent(in) :: a, b
      integer ic
      integer mpnw
      mpnw = mpnwx
      call mpcpr (a%mpr, b%mpr, ic, mpnw) 
      if (ic .lt. 0) then
         mpltt_mm = .true.
      else
         mpltt_mm = .false.
      endif
      return
  end function

! Greater-Than
  logical function mpgtt_mm (a, b)
      type (mp_realm), intent(in) :: a, b
      integer ic
      integer mpnw
      mpnw = mpnwx
      call mpcpr (a%mpr, b%mpr, ic, mpnw) 
      if (ic .gt. 0) then
         mpgtt_mm = .true.
      else
         mpgtt_mm = .false.
      endif
      return
  end function

! Absolute value
  type (mp_realm) function mp_absm (a)
    type (mp_realm), intent(in) :: a
    integer mpnw
    mpnw = mpnwx
    call mpeq (a%mpr, mp_absm%mpr, mpnw) 
    mp_absm%mpr(1) = abs (mp_absm%mpr(1))
  end function

  type (mp_realm) function mp_aintm (a)
      type (mp_realm), intent(in) :: a
      type (mp_realm) t1
      integer mpnw
      mpnw = mpnwx
      call mpinfr (a%mpr, mp_aintm%mpr, t1%mpr, mpnw) 
  end function

  type (mp_realm) function mp_anintm (a)
      type (mp_realm), intent(in) :: a
      integer mpnw
      mpnw = mpnwx
      call mpnint (a%mpr, mp_anintm%mpr, mpnw) 
  end function

  double precision function mp_mtod (a)
      type (mp_realm), intent (in):: a
      real*8 d1
      integer i1
      call mpmdc (a%mpr, d1, i1)
      mp_mtod = d1 * 2.d0 ** i1
  end function

  type (mp_realm) function mp_maxm (a, b)
      type (mp_realm), intent(in) :: a, b
      integer ic
      integer mpnw
      mpnw = mpnwx
      call mpcpr (a%mpr, b%mpr, ic, mpnw) 
      if (ic .ge. 0) then
         call mpeq (a%mpr, mp_maxm%mpr, mpnw) 
      else
         call mpeq (b%mpr, mp_maxm%mpr, mpnw) 
      endif
  end function

  type (mp_realm) function mp_maxm3 (a, b, c)
      type (mp_realm), intent(in) :: a, b, c
      integer ic
      integer mpnw
      mpnw = mpnwx
      call mpcpr (a%mpr, b%mpr, ic, mpnw) 
      if (ic .ge. 0) then
         call mpeq (a%mpr, mp_maxm3%mpr, mpnw) 
      else
         call mpeq (b%mpr, mp_maxm3%mpr, mpnw) 
      endif
      call mpcpr (c%mpr, mp_maxm3%mpr, ic, mpnw) 
      if (ic .ge. 0) call mpeq (c%mpr, mp_maxm3%mpr, mpnw) 
  end function

  type (mp_realm) function mp_minm (a, b)
      type (mp_realm), intent(in) :: a, b
      integer ic
      integer mpnw
      mpnw = mpnwx
      call mpcpr (a%mpr, b%mpr, ic, mpnw) 
      if (ic .le. 0) then
         call mpeq (a%mpr, mp_minm%mpr, mpnw) 
      else
         call mpeq (b%mpr, mp_minm%mpr, mpnw) 
      endif
  end function

  type (mp_realm) function mp_minm3 (a, b, c)
      type (mp_realm), intent(in) :: a, b, c
      integer ic
      integer mpnw
      mpnw = mpnwx
      call mpcpr (a%mpr, b%mpr, ic, mpnw) 
      if (ic .le. 0) then
         call mpeq (a%mpr, mp_minm3%mpr, mpnw) 
      else
         call mpeq (b%mpr, mp_minm3%mpr, mpnw) 
      endif
      call mpcpr (c%mpr, mp_minm3%mpr, ic, mpnw) 
      if (ic .le. 0) call mpeq (c%mpr, mp_minm3%mpr, mpnw) 
  end function

  type (mp_realm) function mp_signm (a, b)
      type (mp_realm), intent (in) :: a, b
      integer mpnw
      mpnw = mpnwx
      call mpeq (a%mpr, mp_signm%mpr, mpnw) 
      mp_signm%mpr(1) = sign (mp_signm%mpr(1), b%mpr(1))
  end function

! SQRT, etc.
  type (mp_realm) function mp_sqrtm (a)
    type (mp_realm), intent(in) :: a
    integer mpnw
    mpnw = mpnwx
    call mpsqrt (a%mpr, mp_sqrtm%mpr, mpnw) 
  end function

!  Conversion
  type (mp_real) function mp_mtoq (a)
      type (mp_realm), intent(in) :: a
      integer mpnw
      mpnw = mpnwx
      call mpeq (a%mpr, mp_mtoq%mpr, mpnw) 
  end function

  type (mp_realm) function mp_qtom (a)
      type (mp_real), intent(in) :: a
      integer mpnw
      mpnw = mpnwx
      call mpeq (a%mpr, mp_qtom%mpr, mpnw) 
  end function

  type (mp_realm) function mp_dtom (da)
      real*8, intent(in) :: da
      call mpdmc (da, 0, mp_dtom%mpr) 
  end function

! Input

  subroutine mp_inpm (iu, q1, q2, q3, q4, q5, q6, q7, q8, q9)
    integer, intent(in) :: iu
    type (mp_realm), intent (out) :: q1, q2, q3, q4, q5, q6, q7, q8, q9
    optional :: q2, q3, q4, q5, q6, q7, q8, q9
    call mpinpm (iu, q1) 
    if (present (q2)) call mpinpm (iu, q2) 
    if (present (q3)) call mpinpm (iu, q3) 
    if (present (q4)) call mpinpm (iu, q4) 
    if (present (q5)) call mpinpm (iu, q5) 
    if (present (q6)) call mpinpm (iu, q6) 
    if (present (q7)) call mpinpm (iu, q7) 
    if (present (q8)) call mpinpm (iu, q8) 
    if (present (q9)) call mpinpm (iu, q9) 
    return
  end subroutine

  subroutine mpinpm (iu, a)
    integer, intent(in) :: iu
    type (mp_realm), intent(out) :: a
    integer i, l, l1
    character*200 line
    character*1 az(mpiplm+100)
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
1   format ('mpinpm: no comma terminating multiprecion input.')

300 continue
    return
  end subroutine

! Output
  subroutine mp_outm (iu, q1, q2, q3, q4, q5, q6, q7, q8, q9)
      integer, intent(in) :: iu
      type (mp_realm), intent(in) :: q1, q2, q3, q4, q5, q6, q7, q8, q9
      optional :: q2, q3, q4, q5, q6, q7, q8, q9
      call mpoutm (iu, q1) 
      if (present (q2)) call mpoutm (iu, q2) 
      if (present (q3)) call mpoutm (iu, q3) 
      if (present (q4)) call mpoutm (iu, q4) 
      if (present (q5)) call mpoutm (iu, q5) 
      if (present (q6)) call mpoutm (iu, q6) 
      if (present (q7)) call mpoutm (iu, q7) 
      if (present (q8)) call mpoutm (iu, q8) 
      if (present (q9)) call mpoutm (iu, q9) 
      return
  end subroutine

  subroutine mpoutm (iu, q)
      integer, intent(in) :: iu
      type (mp_realm), intent(in) :: q
      integer i, l
      character*1 az(mpiplm+100)
      integer mpnw
      mpnw = mpnwx
      call mpoutc (q%mpr, az, l, mpnw) 
      az(l+1) = ','
      write(iu, '(78A1)') (az(i), i = 1, l+1)
  end subroutine

end module mpmodulem
