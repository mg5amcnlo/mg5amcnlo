module dddefmod

!  These Fortran-90 modules perform translation of Fortran for the double-double
!  package  Full details for usage are available in a separate document.
!  IEEE version.

!  David H. Bailey       2005-03-11

!  Interfaces 
!             dd_muliq, dd_mulqi, dd_diviq, dd_divqi
!             dd_muliz, dd_mulzi, dd_diviz, dd_divzi
!             dd_eqqi, dd_eqzi 
!
!  added by Roberto Pittau 
!
!  The following notational scheme is used to designate datatypes below:

!  A   Alphabetic [i.e. ASCII]
!  D   Double precision [i.e. REAL (KIND (0.D0))]
!  Q   Double-double real
!  X   Double complex  [i.e. COMPLEX (KIND (0.D0))]
!  Z   Double-double complex

!  The following parameters are all that need to be changed in normal usage:

use ddfunmod
private kdb
parameter (kdb = kind (0.d0))
type dd_real
  sequence
  real*8 ddr(2)
end type
type dd_complex
  sequence
  real*8 ddc(4)
end type
type (dd_real), public:: ddeps
real*8, private:: ddt0(4), ddt1(4), ddt2(4), ddt3(4), ddt4(4)

contains

  subroutine ddxzc (a, b)

!  This converts the DC variable A to the DDC variable B.
!  This routine is not intended to be called directly by the user.

    complex (kdb) a
    real*8 b(4)

    b(1) = a
    b(2) = 0.d0
    b(3) = aimag (a)
    b(4) = 0.d0
    return
  end subroutine

  subroutine ddmzc (a, b)

!  This converts the DD real variable A to the DDC variable B.
!  This routine is not intended to be called directly by the user.

    real*8 a(2), b(4)

    b(1) = a(1)
    b(2) = a(2)
    b(3) = 0.d0
    b(4) = 0.d0
    return
  end subroutine

end module

module ddrealmod

!  This Fortran-90 module defines operator extensions involving the
!  DD_REAL datatype.  For operations involving two DD data types,
!  those whose first argument is DD_REAL are included here.
!  Others are handled in other modules.

!  The subroutines and functions defined in this module are private
!  and not intended to be called directly by the user.

use ddfunmod
use dddefmod
private kdb
parameter (kdb = kind (0.d0))
real*8, private:: ddt0(4), ddt1(4), ddt2(4), ddt3(4), ddt4(4)
private &
  dd_eqqq, dd_eqqz, dd_eqrq, dd_eqqr, dd_eqdq, dd_eqqd, dd_eqxq, dd_eqqx, &
  dd_addqq, dd_addqz, dd_adddq, dd_addqd, dd_addxq, dd_addqx, &
  dd_subqq, dd_subqz, dd_subdq, dd_subqd, dd_subxq, dd_subqx, dd_negq, &
  dd_mulqq, dd_mulqz, dd_muldq, dd_mulqd, dd_mulxq, dd_mulqx, &
  dd_divqq, dd_divqz, dd_divdq, dd_divqd, dd_divxq, dd_divqx, &
  dd_expqq, dd_expqi, dd_expdq, dd_expqd, &
  dd_eqtqq, dd_eqtqz, dd_eqtdq, dd_eqtqd, dd_eqtxq, dd_eqtqx, &
  dd_netqq, dd_netqz, dd_netdq, dd_netqd, dd_netxq, dd_netqx, &
  dd_letqq, dd_letdq, dd_letqd, dd_getqq, dd_getdq, dd_getqd, &
  dd_lttqq, dd_lttdq, dd_lttqd, dd_gttqq, dd_gttdq, dd_gttqd, &  
  dd_muliq, dd_mulqi, dd_diviq, dd_divqi, &
  dd_eqqi

!  DDR operator extension interface blocks.

interface assignment (=)
  module procedure dd_eqqq
  module procedure dd_eqqz
  module procedure dd_eqdq
  module procedure dd_eqqd
  module procedure dd_eqxq
  module procedure dd_eqqx
  module procedure dd_eqqa
  module procedure dd_eqqi
end interface

interface operator (+)
  module procedure dd_addqq
  module procedure dd_addqz
  module procedure dd_adddq
  module procedure dd_addqd
  module procedure dd_addxq
  module procedure dd_addqx
end interface

interface operator (-)
  module procedure dd_subqq
  module procedure dd_subqz
  module procedure dd_subdq
  module procedure dd_subqd
  module procedure dd_subxq
  module procedure dd_subqx

  module procedure dd_negq
end interface

interface operator (*)
  module procedure dd_mulqq
  module procedure dd_mulqz
  module procedure dd_muldq
  module procedure dd_mulqd
  module procedure dd_mulxq
  module procedure dd_mulqx
  module procedure dd_muliq
  module procedure dd_mulqi
end interface

interface operator (/)
  module procedure dd_divqq
  module procedure dd_divqz
  module procedure dd_divdq
  module procedure dd_divqd
  module procedure dd_divxq
  module procedure dd_divqx
  module procedure dd_diviq
  module procedure dd_divqi
end interface

interface operator (**)
  module procedure dd_expqq
  module procedure dd_expqi
  module procedure dd_expdq
  module procedure dd_expqd
end interface

interface operator (.eq.)
  module procedure dd_eqtqq
  module procedure dd_eqtqz
  module procedure dd_eqtdq
  module procedure dd_eqtqd
  module procedure dd_eqtxq
  module procedure dd_eqtqx
end interface

interface operator (.ne.)
  module procedure dd_netqq
  module procedure dd_netqz
  module procedure dd_netdq
  module procedure dd_netqd
  module procedure dd_netxq
  module procedure dd_netqx
end interface

interface operator (.le.)
  module procedure dd_letqq
  module procedure dd_letdq
  module procedure dd_letqd
end interface

interface operator (.ge.)
  module procedure dd_getqq
  module procedure dd_getdq
  module procedure dd_getqd
end interface

interface operator (.lt.)
  module procedure dd_lttqq
  module procedure dd_lttdq
  module procedure dd_lttqd
end interface

interface operator (.gt.)
  module procedure dd_gttqq
  module procedure dd_gttdq
  module procedure dd_gttqd
end interface

contains

!  DDR assignment routines.

  subroutine dd_eqqq (qa, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    intent (out):: qa
    intent (in):: qb
    qa%ddr(1) = qb%ddr(1)
    qa%ddr(2) = qb%ddr(2)
    return
  end subroutine

  subroutine dd_eqqz (qa, zb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    intent (out):: qa
    intent (in):: zb
    qa%ddr(1) = zb%ddc(1)
    qa%ddr(2) = zb%ddc(2)
    return
  end subroutine

  subroutine dd_eqdq (da, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    intent (out):: da
    intent (in):: qb
    da = qb%ddr(1)
    return
  end subroutine

  subroutine dd_eqqd (qa, db)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    intent (out):: qa
    intent (in):: db
    qa%ddr(1) = db
    qa%ddr(2) = 0.d0
    return
  end subroutine

  subroutine dd_eqqi (qa, i)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    intent (out):: qa
    intent (in):: i
    qa%ddr(1) = i
    qa%ddr(2) = 0.d0
    return
  end subroutine

  subroutine dd_eqxq (xa, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    intent (out):: xa
    intent (in):: qb
    xa = qb%ddr(1)
    return
  end subroutine

  subroutine dd_eqqx (qa, xb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    intent (out):: qa
    intent (in):: xb
    qa%ddr(1) = xb
    qa%ddr(2) = 0.d0
    return
  end subroutine

  subroutine dd_eqqa (qa, ab)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    character*(*), intent (in):: ab
    intent (out):: qa
    character*80 t
    t = ab
    call ddinpc (t, qa%ddr)
    return
  end subroutine

!  DDR add routines.

  function dd_addqq (qa, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_addqq
    intent (in):: qa, qb
    call ddadd (qa%ddr, qb%ddr, dd_addqq%ddr)
    return
  end function

  function dd_addqz (qa, zb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_addqz
    intent (in):: qa, zb
    call ddmzc (qa%ddr, ddt1)
    call ddcadd (ddt1, zb%ddc, dd_addqz%ddc)
    return
  end function

  function dd_adddq (da, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_adddq
    intent (in):: da, qb
    ddt1(1) = da
    ddt1(2) = 0.d0
    call ddadd (ddt1, qb%ddr, dd_adddq%ddr)
    return
  end function

  function dd_addqd (qa, db)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_addqd
    intent (in):: qa, db
    ddt1(1) = db
    ddt1(2) = 0.d0
    call ddadd (qa%ddr, ddt1, dd_addqd%ddr)
    return
  end function

  function dd_addxq (xa, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_addxq
    intent (in):: xa, qb
    call ddxzc (xa, ddt1)
    call ddmzc (qb%ddr, ddt2)
    call ddcadd (ddt1, ddt2, dd_addxq%ddc)
    return
  end function

  function dd_addqx (qa, xb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_addqx
    intent (in):: qa, xb
    call ddmzc (qa%ddr, ddt1)
    call ddxzc (xb, ddt2)
    call ddcadd (ddt1, ddt2, dd_addqx%ddc)
    return
  end function

!  DDR subtract routines.

  function dd_subqq (qa, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_subqq
    intent (in):: qa, qb
    call ddsub (qa%ddr, qb%ddr, dd_subqq%ddr)
    return
  end function

  function dd_subqz (qa, zb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_subqz
    intent (in):: qa, zb
    call ddmzc (qa%ddr, ddt1)
    call ddcsub (ddt1, zb%ddc, dd_subqz%ddc)
    return
  end function

  function dd_subdq (da, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_subdq
    intent (in):: da, qb
    ddt1(1) = da
    ddt1(2) = 0.d0
    call ddsub (ddt1, qb%ddr, dd_subdq%ddr)
    return
  end function

  function dd_subqd (qa, db)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_subqd
    intent (in):: qa, db
    ddt1(1) = db
    ddt1(2) = 0.d0
    call ddsub (qa%ddr, ddt1, dd_subqd%ddr)
    return
  end function

  function dd_subxq (xa, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_subxq
    intent (in):: xa, qb
    call ddxzc (xa, ddt1)
    call ddmzc (qb%ddr, ddt2)
    call ddcsub (ddt1, ddt2, dd_subxq%ddc)
    return
  end function

  function dd_subqx (qa, xb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_subqx
    intent (in):: qa, xb
    call ddmzc (qa%ddr, ddt1)
    call ddxzc (xb, ddt2)
    call ddcsub (ddt1, ddt2, dd_subqx%ddc)
    return
  end function

!  DDR negation routine.

  function dd_negq (qa)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_negq
    intent (in):: qa
    call ddeq (qa%ddr, dd_negq%ddr)
    dd_negq%ddr(1) = - qa%ddr(1)
    dd_negq%ddr(2) = - qa%ddr(2)
    return
  end function

!  DDR multiply routines.

  function dd_mulqq (qa, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_mulqq
    intent (in):: qa, qb
    call ddmul (qa%ddr, qb%ddr, dd_mulqq%ddr)
    return
  end function

  function dd_mulqz (qa, zb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_mulqz
    intent (in):: qa, zb
    call ddmzc (qa%ddr, ddt1)
    call ddcmul (ddt1, zb%ddc, dd_mulqz%ddc)
    return
  end function

  function dd_muldq (da, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_muldq
    intent (in):: da, qb
    call ddmuld (qb%ddr, da, dd_muldq%ddr)
    return
  end function

  function dd_mulqd (qa, db)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_mulqd
    intent (in):: qa, db
    call ddmuld (qa%ddr, db, dd_mulqd%ddr)
    return
  end function

  function dd_muliq (i, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
     type (dd_real):: dd_muliq
    intent (in):: i, qb
    da= i*1.d0
    call ddmuld (qb%ddr, da, dd_muliq%ddr)
    return
  end function

  function dd_mulqi (qa, i)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_mulqi
    intent (in):: qa, i
    db= i*1.d0
    call ddmuld (qa%ddr, db, dd_mulqi%ddr)
    return
  end function

  function dd_mulxq (xa, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_mulxq
    intent (in):: xa, qb
    call ddxzc (xa, ddt1)
    call ddmzc (qb%ddr, ddt2)
    call ddcmul (ddt1, ddt2, dd_mulxq%ddc)
    return
  end function

  function dd_mulqx (qa, xb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_mulqx
    intent (in):: qa, xb
    call ddmzc (qa%ddr, ddt1)
    call ddxzc (xb, ddt2)
    call ddcmul (ddt1, ddt2, dd_mulqx%ddc)
    return
  end function

!  DDR divide routines.

  function dd_divqq (qa, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_divqq
    intent (in):: qa, qb
    call dddiv (qa%ddr, qb%ddr, dd_divqq%ddr)
    return
  end function

  function dd_divqz (qa, zb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_divqz
    intent (in):: qa, zb
    call ddmzc (qa%ddr, ddt1)
    call ddcdiv (ddt1, zb%ddc, dd_divqz%ddc)
    return
  end function

  function dd_divdq (da, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_divdq
    intent (in):: da, qb
    ddt1(1) = da
    ddt1(2) = 0.d0
    call dddiv (ddt1, qb%ddr, dd_divdq%ddr)
    return
  end function

  function dd_divqd (qa, db)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_divqd
    intent (in):: qa, db
    call dddivd (qa%ddr, db, dd_divqd%ddr)
    return
  end function

  function dd_diviq (i, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_diviq
    intent (in):: i, qb
    ddt1(1) = i*1.d0
    ddt1(2) = 0.d0
    call dddiv (ddt1, qb%ddr, dd_diviq%ddr)
    return
  end function

  function dd_divqi (qa, i)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_divqi
    intent (in):: qa, i
    db= i*1.d0
    call dddivd (qa%ddr, db, dd_divqi%ddr)
    return
  end function

  function dd_divxq (xa, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_divxq
    intent (in):: xa, qb
    call ddxzc (xa, ddt1)
    call ddmzc (qb%ddr, ddt2)
    call ddcdiv (ddt1, ddt2, dd_divxq%ddc)
    return
  end function

  function dd_divqx (qa, xb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_divqx
    intent (in):: qa, xb
    call ddmzc (qa%ddr, ddt1)
    call ddxzc (xb, ddt2)
    call ddcdiv (ddt1, ddt2, dd_divqx%ddc)
    return
  end function

!  DDR exponentiation routines.

  function dd_expqq (qa, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_expqq
    intent (in):: qa, qb
    call ddlog (qa%ddr, ddt1)
    call ddmul (ddt1, qb%ddr, ddt2)
    call ddexp (ddt2, dd_expqq%ddr)
    return
  end function

  function dd_expqi (qa, ib)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_expqi
    intent (in):: qa, ib
    call ddnpwr (qa%ddr, ib, dd_expqi%ddr)
    return
  end function

  function dd_expdq (da, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_expdq
    intent (in):: da, qb
    ddt1(1) = da
    ddt1(2) = 0.d0
    call ddlog (ddt1, ddt2)
    call ddmul (ddt2, qb%ddr, ddt3)
    call ddexp (ddt3, dd_expdq%ddr)
    return
    end function

  function dd_expqd (qa, db)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_expqd
    intent (in):: qa, db
    call ddlog (qa%ddr, ddt1)
    call ddmuld (ddt1, db, ddt2)
    call ddexp (ddt2, dd_expqd%ddr)
    return
  end function

!  DDR .EQ. routines.

  function dd_eqtqq (qa, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    logical dd_eqtqq
    intent (in):: qa, qb
    call ddcpr (qa%ddr, qb%ddr, ic)
    if (ic .eq. 0) then
      dd_eqtqq = .true.
    else
      dd_eqtqq = .false.
    endif
    return
  end function

  function dd_eqtqz (qa, zb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    logical dd_eqtqz
    intent (in):: qa, zb
    call ddmzc (qa%ddr, ddt1)
    call ddcpr (ddt1, zb%ddc, ic1)
    call ddcpr (ddt1(3), zb%ddc(3), ic2)
    if (ic1 .eq. 0 .and. ic2 .eq. 0) then
      dd_eqtqz = .true.
    else
      dd_eqtqz = .false.
    endif
    return
  end function

  function dd_eqtdq (da, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    logical dd_eqtdq
    intent (in):: da, qb
    ddt1(1) = da
    ddt1(2) = 0.d0
    call ddcpr (ddt1, qb%ddr, ic)
    if (ic .eq. 0) then
      dd_eqtdq = .true.
    else
      dd_eqtdq = .false.
    endif
    return
  end function

  function dd_eqtqd (qa, db)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    logical dd_eqtqd
    intent (in):: qa, db
    ddt1(1) = db
    ddt1(2) = 0.d0
    call ddcpr (qa%ddr, ddt1, ic)
    if (ic .eq. 0) then
      dd_eqtqd = .true.
    else
      dd_eqtqd = .false.
    endif
    return
  end function

  function dd_eqtxq (xa, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    logical dd_eqtxq
    intent (in):: xa, qb
    call ddxzc (xa, ddt1)
    call ddmzc (qb%ddr, ddt2)
    call ddcpr (ddt1, ddt2, ic1)
    call ddcpr (ddt1(3), ddt2(3), ic2)
    if (ic1 .eq. 0 .and. ic2 .eq. 0) then
      dd_eqtxq = .true.
    else
      dd_eqtxq = .false.
    endif
    return
  end function

  function dd_eqtqx (qa, xb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    logical dd_eqtqx
    intent (in):: qa, xb
    call ddmzc (qa%ddr, ddt1)
    call ddxzc (xb, ddt2)
    call ddcpr (ddt1, ddt2, ic1)
    call ddcpr (ddt1(3), ddt2(3), ic2)
    if (ic1 .eq. 0 .and. ic2 .eq. 0) then
      dd_eqtqx = .true.
    else
      dd_eqtqx = .false.
    endif
    return
  end function

!  DDR .NE. routines.

  function dd_netqq (qa, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    logical dd_netqq
    intent (in):: qa, qb
    call ddcpr (qa%ddr, qb%ddr, ic)
    if (ic .ne. 0) then
      dd_netqq = .true.
    else
      dd_netqq = .false.
    endif
    return
  end function

  function dd_netqz (qa, zb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    logical dd_netqz
    intent (in):: qa, zb
    call ddmzc (qa%ddr, ddt1)
    call ddcpr (ddt1, zb%ddc, ic1)
    call ddcpr (ddt1(3), zb%ddc(3), ic2)
    if (ic1 .ne. 0 .or. ic2 .ne. 0) then
      dd_netqz = .true.
    else
      dd_netqz = .false.
    endif
    return
  end function

  function dd_netdq (da, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    logical dd_netdq
    intent (in):: da, qb
    ddt1(1) = da
    ddt1(2) = 0.d0
    call ddcpr (ddt1, qb%ddr, ic)
    if (ic .ne. 0) then
      dd_netdq = .true.
    else
      dd_netdq = .false.
    endif
    return
  end function

  function dd_netqd (qa, db)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    logical dd_netqd
    intent (in):: qa, db
    ddt1(1) = db
    ddt1(2) = 0.d0
    call ddcpr (qa%ddr, ddt1, ic)
    if (ic .ne. 0) then
      dd_netqd = .true.
    else
      dd_netqd = .false.
    endif
    return
  end function

  function dd_netxq (xa, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    logical dd_netxq
    intent (in):: xa, qb
    call ddxzc (xa, ddt1)
    call ddmzc (qb%ddr, ddt2)
    call ddcpr (ddt1, ddt2, ic1)
    call ddcpr (ddt1(3), ddt2(3), ic2)
    if (ic1 .ne. 0 .or. ic2 .ne. 0) then
      dd_netxq = .true.
    else
      dd_netxq = .false.
    endif
    return
  end function

  function dd_netqx (qa, xb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    logical dd_netqx
    intent (in):: qa, xb
    call ddmzc (qa%ddr, ddt1)
    call ddxzc (xb, ddt2)
    call ddcpr (ddt1, ddt2, ic1)
    call ddcpr (ddt1(3), ddt2(3), ic2)
    if (ic1 .ne. 0 .or. ic2 .ne. 0) then
      dd_netqx = .true.
    else
      dd_netqx = .false.
    endif
    return
  end function

!  DDR .LE. routines.

  function dd_letqq (qa, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    logical dd_letqq
    intent (in):: qa, qb
    call ddcpr (qa%ddr, qb%ddr, ic)
    if (ic .le. 0) then
      dd_letqq = .true.
    else
      dd_letqq = .false.
    endif
    return
  end function

  function dd_letdq (da, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    logical dd_letdq
    intent (in):: da, qb
    ddt1(1) = da
    ddt1(2) = 0.d0
    call ddcpr (ddt1, qb%ddr, ic)
    if (ic .le. 0) then
      dd_letdq = .true.
    else
      dd_letdq = .false.
    endif
    return
  end function

  function dd_letqd (qa, db)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    logical dd_letqd
    intent (in):: qa, db
    ddt1(1) = db
    ddt1(2) = 0.d0
    call ddcpr (qa%ddr, ddt1, ic)
    if (ic .le. 0) then
      dd_letqd = .true.
    else
      dd_letqd = .false.
    endif
    return
  end function

!  DDR .GE. routines.

  function dd_getqq (qa, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    logical dd_getqq
    intent (in):: qa, qb
    call ddcpr (qa%ddr, qb%ddr, ic)
    if (ic .ge. 0) then
      dd_getqq = .true.
    else
      dd_getqq = .false.
    endif
    return
  end function

  function dd_getdq (da, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    logical dd_getdq
    intent (in):: da, qb
    ddt1(1) = da
    ddt1(2) = 0.d0
    call ddcpr (ddt1, qb%ddr, ic)
    if (ic .ge. 0) then
      dd_getdq = .true.
    else
      dd_getdq = .false.
    endif
    return
  end function

  function dd_getqd (qa, db)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    logical dd_getqd
    intent (in):: qa, db
    ddt1(1) = db
    ddt1(2) = 0.d0
    call ddcpr (qa%ddr, ddt1, ic)
    if (ic .ge. 0) then
      dd_getqd = .true.
    else
      dd_getqd = .false.
    endif
    return
  end function

!  DDR .LT. routines.

  function dd_lttqq (qa, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    logical dd_lttqq
    intent (in):: qa, qb
    call ddcpr (qa%ddr, qb%ddr, ic)
    if (ic .lt. 0) then
      dd_lttqq = .true.
    else
      dd_lttqq = .false.
    endif
    return
  end function

  function dd_lttdq (da, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    logical dd_lttdq
    intent (in):: da, qb
    ddt1(1) = da
    ddt1(2) = 0.d0
    call ddcpr (ddt1, qb%ddr, ic)
    if (ic .lt. 0) then
      dd_lttdq = .true.
    else
      dd_lttdq = .false.
    endif
    return
  end function

  function dd_lttqd (qa, db)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    logical dd_lttqd
    intent (in):: qa, db
    ddt1(1) = db
    ddt1(2) = 0.d0
    call ddcpr (qa%ddr, ddt1, ic)
    if (ic .lt. 0) then
      dd_lttqd = .true.
    else
      dd_lttqd = .false.
    endif
    return
  end function

!  DDR .GT. routines.

  function dd_gttqq (qa, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    logical dd_gttqq
    intent (in):: qa, qb
    call ddcpr (qa%ddr, qb%ddr, ic)
    if (ic .gt. 0) then
      dd_gttqq = .true.
    else
      dd_gttqq = .false.
    endif
    return
  end function

  function dd_gttdq (da, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    logical dd_gttdq
    intent (in):: da, qb
    ddt1(1) = da
    ddt1(2) = 0.d0
    call ddcpr (ddt1, qb%ddr, ic)
    if (ic .gt. 0) then
      dd_gttdq = .true.
    else
      dd_gttdq = .false.
    endif
    return
  end function

  function dd_gttqd (qa, db)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    logical dd_gttqd
    intent (in):: qa, db
    ddt1(1) = db
    ddt1(2) = 0.d0
    call ddcpr (qa%ddr, ddt1, ic)
    if (ic .gt. 0) then
      dd_gttqd = .true.
    else
      dd_gttqd = .false.
    endif
    return
  end function

end module


module ddcmpmod

!  This Fortran-90 module defines operator extensions involving the
!  DD_COMPLEX datatype.  For operations involving two DD data types,
!  those whose first argument is DD_COMPLEX are included here.
!  Others are handled in other modules.

!  The subroutines and functions defined in this module are private
!  and not intended to be called directly by the user.

use ddfunmod
use dddefmod
private kdb
parameter (kdb = kind (0.d0))
real*8, private:: ddt0(4), ddt1(4), ddt2(4), ddt3(4), ddt4(4)
private &
  dd_eqzq, dd_eqzz, dd_eqdz, dd_eqzd, dd_eqxz, dd_eqzx, &
  dd_addzq, dd_addzz, dd_adddz, dd_addzd, dd_addxz, dd_addzx, &
  dd_subzq, dd_subzz, dd_subdz, dd_subzd, dd_subxz, dd_subzx, dd_negz, &
  dd_mulzq, dd_mulzz, dd_muldz, dd_mulzd, dd_mulxz, dd_mulzx, &
  dd_divzq, dd_divzz, dd_divdz, dd_divzd, dd_divxz, dd_divzx, &
  dd_expzi, &
  dd_eqtzq, dd_eqtzz, dd_eqtdz, dd_eqtzd, dd_eqtxz, dd_eqtzx, &
  dd_netzq, dd_netzz, dd_netdz, dd_netzd, dd_netxz, dd_netzx, &
  dd_muliz, dd_mulzi, dd_diviz, dd_divzi, &
  dd_eqzi

!  DDR operator extension interface blocks.

interface assignment (=)
  module procedure dd_eqzq
  module procedure dd_eqzz
  module procedure dd_eqdz
  module procedure dd_eqzd
  module procedure dd_eqxz
  module procedure dd_eqzx
  module procedure dd_eqzi
end interface

interface operator (+)
  module procedure dd_addzq
  module procedure dd_addzz
  module procedure dd_adddz
  module procedure dd_addzd
  module procedure dd_addxz
  module procedure dd_addzx
end interface

interface operator (-)
  module procedure dd_subzq
  module procedure dd_subzz
  module procedure dd_subdz
  module procedure dd_subzd
  module procedure dd_subxz
  module procedure dd_subzx

  module procedure dd_negz
end interface

interface operator (*)
  module procedure dd_mulzq
  module procedure dd_mulzz
  module procedure dd_muldz
  module procedure dd_mulzd
  module procedure dd_mulxz
  module procedure dd_mulzx
  module procedure dd_muliz
  module procedure dd_mulzi
end interface

interface operator (/)
  module procedure dd_divzq
  module procedure dd_divzz
  module procedure dd_divdz
  module procedure dd_divzd
  module procedure dd_divxz
  module procedure dd_divzx
  module procedure dd_diviz
  module procedure dd_divzi
end interface

interface operator (**)
  module procedure dd_expzi
end interface

interface operator (.eq.)
  module procedure dd_eqtzq
  module procedure dd_eqtzz
  module procedure dd_eqtdz
  module procedure dd_eqtzd
  module procedure dd_eqtxz
  module procedure dd_eqtzx
end interface

interface operator (.ne.)
  module procedure dd_netzq
  module procedure dd_netzz
  module procedure dd_netdz
  module procedure dd_netzd
  module procedure dd_netxz
  module procedure dd_netzx
end interface

contains

!  DDC assignment routines.

  subroutine dd_eqzq (za, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    intent (out):: za
    intent (in):: qb
    call ddmzc (qb%ddr, za%ddc)
    return
  end subroutine

  subroutine dd_eqzz (za, zb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    intent (out):: za
    intent (in):: zb
    call ddceq (zb%ddc, za%ddc)
    return
  end subroutine

  subroutine dd_eqdz (da, zb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    intent (out):: da
    intent (in):: zb
    da = zb%ddc(1)
    return
  end subroutine

  subroutine dd_eqzd (za, db)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    intent (out):: za
    intent (in):: db
    xb = db
    call ddxzc (xb, za%ddc)
    return
  end subroutine

  subroutine dd_eqzi (za, i)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    intent (out):: za
    intent (in):: i
    xb = i
    call ddxzc (xb, za%ddc)
    return
  end subroutine

  subroutine dd_eqxz (xa, zb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    intent (out):: xa
    intent (in):: zb
    db = zb%ddc(1)
    dc = zb%ddc(3)
    xa = cmplx (db, dc, kdb)
    return
  end subroutine

  subroutine dd_eqzx (za, xb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    intent (out):: za
    intent (in):: xb
    call ddxzc (xb, za%ddc)
    return
  end subroutine

!  DDC add routines.

  function dd_addzq (za, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_addzq
    intent (in):: za, qb
    call ddmzc (qb%ddr, ddt1)
    call ddcadd (za%ddc, ddt1, dd_addzq%ddc)
    return
  end function

  function dd_addzz (za, zb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_addzz
    intent (in):: za, zb
    call ddcadd (za%ddc, zb%ddc, dd_addzz%ddc)
    return
  end function

  function dd_adddz (da, zb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_adddz
    intent (in):: da, zb
    xa = da
    call ddxzc (xa, ddt1)
    call ddcadd (ddt1, zb%ddc, dd_adddz%ddc)
    return
  end function

  function dd_addzd (za, db)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_addzd
    intent (in):: za, db
    xb = db
    call ddxzc (xb, ddt1)
    call ddcadd (za%ddc, ddt1, dd_addzd%ddc)
    return
  end function

  function dd_addxz (xa, zb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_addxz
    intent (in):: xa, zb
    call ddxzc (xa, ddt1)
    call ddcadd (ddt1, zb%ddc, dd_addxz%ddc)
    return
  end function

  function dd_addzx (za, xb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_addzx
    intent (in):: za, xb
    call ddxzc (xb, ddt2)
    call ddcadd (za%ddc, ddt2, dd_addzx%ddc)
    return
  end function

!  DDC subtract routines.

  function dd_subzq (za, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_subzq
    intent (in):: za, qb
    call ddmzc (qb%ddr, ddt1)
    call ddcsub (za%ddc, ddt1, dd_subzq%ddc)
    return
  end function

  function dd_subzz (za, zb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_subzz
    intent (in):: za, zb
    call ddcsub (za%ddc, zb%ddc, dd_subzz%ddc)
    return
  end function

  function dd_subdz (da, zb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_subdz
    intent (in):: da, zb
    xa = da
    call ddxzc (xa, ddt1)
    call ddcsub (ddt1, zb%ddc, dd_subdz%ddc)
    return
  end function

  function dd_subzd (za, db)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_subzd
    intent (in):: za, db
    xb = db
    call ddxzc (xb, ddt1)
    call ddcsub (za%ddc, ddt1, dd_subzd%ddc)
    return
  end function

  function dd_subxz (xa, zb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_subxz
    intent (in):: xa, zb
    call ddxzc (xa, ddt1)
    call ddcsub (ddt1, zb%ddc, dd_subxz%ddc)
    return
  end function

  function dd_subzx (za, xb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_subzx
    intent (in):: za, xb
    call ddxzc (xb, ddt2)
    call ddcsub (za%ddc, ddt2, dd_subzx%ddc)
    return
  end function

!  DDC negation routine.

  function dd_negz (za)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_negz
    intent (in):: za
    call ddceq (za%ddc, dd_negz%ddc)
    dd_negz%ddc(1) = - za%ddc(1)
    dd_negz%ddc(2) = - za%ddc(2)
    dd_negz%ddc(3) = - za%ddc(3)
    dd_negz%ddc(4) = - za%ddc(4)
    return
  end function

!  DDC multiply routines.

  function dd_mulzq (za, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_mulzq
    intent (in):: za, qb
    call ddmzc (qb%ddr, ddt1)
    call ddcmul (za%ddc, ddt1, dd_mulzq%ddc)
    return
  end function

  function dd_mulzz (za, zb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_mulzz
    intent (in):: za, zb
    call ddcmul (za%ddc, zb%ddc, dd_mulzz%ddc)
    return
  end function

  function dd_muldz (da, zb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_muldz
    intent (in):: da, zb
    xa = da
    call ddxzc (xa, ddt1)
    call ddcmul (ddt1, zb%ddc, dd_muldz%ddc)
    return
  end function

  function dd_mulzd (za, db)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_mulzd
    intent (in):: za, db
    xb = db
    call ddxzc (xb, ddt1)
    call ddcmul (za%ddc, ddt1, dd_mulzd%ddc)
    return
  end function

  function dd_muliz (i, zb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_muliz
    intent (in):: i, zb
    xa = i
    call ddxzc (xa, ddt1)
    call ddcmul (ddt1, zb%ddc, dd_muliz%ddc)
    return
  end function

  function dd_mulzi (za, i)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_mulzi
    intent (in):: za, i
    xb = i
    call ddxzc (xb, ddt1)
    call ddcmul (za%ddc, ddt1, dd_mulzi%ddc)
    return
  end function

  function dd_mulxz (xa, zb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_mulxz
    intent (in):: xa, zb
    call ddxzc (xa, ddt1)
    call ddcmul (ddt1, zb%ddc, dd_mulxz%ddc)
    return
  end function

  function dd_mulzx (za, xb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_mulzx
    intent (in):: za, xb
    call ddxzc (xb, ddt2)
    call ddcmul (za%ddc, ddt2, dd_mulzx%ddc)
    return
  end function

!  DDC divide routines.

  function dd_divzq (za, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_divzq
    intent (in):: za, qb
    call ddmzc (qb%ddr, ddt1)
    call ddcdiv (za%ddc, ddt1, dd_divzq%ddc)
    return
  end function

  function dd_divzz (za, zb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_divzz
    intent (in):: za, zb
    call ddcdiv (za%ddc, zb%ddc, dd_divzz%ddc)
    return
  end function

  function dd_divdz (da, zb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_divdz
    intent (in):: da, zb
    xa = da
    call ddxzc (xa, ddt1)
    call ddcdiv (ddt1, zb%ddc, dd_divdz%ddc)
    return
  end function

  function dd_divzd (za, db)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_divzd
    intent (in):: za, db
    xb = db
    call ddxzc (xb, ddt1)
    call ddcdiv (za%ddc, ddt1, dd_divzd%ddc)
    return
  end function

  function dd_diviz (i, zb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_diviz
    intent (in):: i, zb
    xa = i
    call ddxzc (xa, ddt1)
    call ddcdiv (ddt1, zb%ddc, dd_diviz%ddc)
    return
  end function

  function dd_divzi (za, i)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_divzi
    intent (in):: za, i
    xb = i
    call ddxzc (xb, ddt1)
    call ddcdiv (za%ddc, ddt1, dd_divzi%ddc)
    return
  end function

  function dd_divxz (xa, zb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_divxz
    intent (in):: xa, zb
    call ddxzc (xa, ddt1)
    call ddcdiv (ddt1, zb%ddc, dd_divxz%ddc)
    return
  end function

  function dd_divzx (za, xb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_divzx
    intent (in):: za, xb
    call ddxzc (xb, ddt2)
    call ddcdiv (za%ddc, ddt2, dd_divzx%ddc)
    return
  end function

!  DDC exponentiation routines.

  function dd_expzi (za, ib)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_expzi
    intent (in):: za, ib
    call ddcpwr (za%ddc, ib, dd_expzi%ddc)
    return
  end function

!  DDC .EQ. routines.

  function dd_eqtzq (za, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    logical dd_eqtzq
    intent (in):: za, qb
    call ddmzc (qb%ddr, ddt1)
    call ddcpr (za%ddc, ddt1, ic1)
    call ddcpr (za%ddc(3), ddt1(3), ic2)
    if (ic1 .eq. 0 .and. ic2 .eq. 0) then
      dd_eqtzq = .true.
    else
      dd_eqtzq = .false.
    endif
    return
  end function

  function dd_eqtzz (za, zb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    logical dd_eqtzz
    intent (in):: za, zb
    call ddcpr (za%ddc, zb%ddc, ic1)
    call ddcpr (za%ddc(3), zb%ddc(3), ic2)
    if (ic1 .eq. 0 .and. ic2 .eq. 0) then
      dd_eqtzz = .true.
    else
      dd_eqtzz = .false.
    endif
    return
  end function

  function dd_eqtdz (da, zb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    logical dd_eqtdz
    intent (in):: da, zb
    ddt1(1) = da
    ddt1(2) = 0.d0
    call ddcpr (ddt1, zb%ddc, ic1)
    call ddcpr (ddt1(3), zb%ddc(3), ic2)
    if (ic1 .eq. 0 .and. ic2 .eq. 0) then
      dd_eqtdz = .true.
    else
      dd_eqtdz = .false.
    endif
    return
  end function

  function dd_eqtzd (za, db)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    logical dd_eqtzd
    intent (in):: za, db
    ddt1(1) = db
    ddt1(2) = 0.d0
    call ddcpr (za%ddc, ddt1, ic1)
    call ddcpr (za%ddc(3), ddt1(3), ic2)
    if (ic1 .eq. 0 .and. ic2 .eq. 0) then
      dd_eqtzd = .true.
    else
      dd_eqtzd = .false.
    endif
    return
  end function

  function dd_eqtxz (xa, zb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    logical dd_eqtxz
    intent (in):: xa, zb
    call ddxzc (xa, ddt1)
    call ddcpr (ddt1, zb%ddc, ic1)
    call ddcpr (ddt1(3), zb%ddc(3), ic2)
    if (ic1 .eq. 0 .and. ic2 .eq. 0) then
      dd_eqtxz = .true.
    else
      dd_eqtxz = .false.
    endif
    return
  end function

  function dd_eqtzx (za, xb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    logical dd_eqtzx
    intent (in):: za, xb
    call ddxzc (xb, ddt2)
    call ddcpr (za%ddc, ddt2, ic1)
    call ddcpr (za%ddc(3), ddt2(3), ic2)
    if (ic1 .eq. 0 .and. ic2 .eq. 0) then
      dd_eqtzx = .true.
    else
      dd_eqtzx = .false.
    endif
    return
  end function

!  DDC .NE. routines.

  function dd_netzq (za, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    logical dd_netzq
    intent (in):: za, qb
    call ddmzc (qb%ddr, ddt1)
    call ddcpr (za%ddc, ddt1, ic1)
    call ddcpr (za%ddc(3), ddt1(3), ic2)
    if (ic1 .ne. 0 .or. ic2 .ne. 0) then
      dd_netzq = .true.
    else
      dd_netzq = .false.
    endif
    return
  end function

  function dd_netzz (za, zb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    logical dd_netzz
    intent (in):: za, zb
    call ddcpr (za%ddc, zb%ddc, ic1)
    call ddcpr (za%ddc(3), zb%ddc(3), ic2)
    if (ic1 .ne. 0 .or. ic2 .ne. 0) then
      dd_netzz = .true.
    else
      dd_netzz = .false.
    endif
    return
  end function

  function dd_netdz (da, zb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    logical dd_netdz
    intent (in):: da, zb
    ddt1(1) = da
    ddt1(2) = 0.d0
    call ddcpr (ddt1, zb%ddc, ic1)
    call ddcpr (ddt1(3), zb%ddc(3), ic2)
    if (ic1 .ne. 0 .or. ic2 .ne. 0) then
      dd_netdz = .true.
    else
      dd_netdz = .false.
    endif
    return
  end function

  function dd_netzd (za, db)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    logical dd_netzd
    intent (in):: za, db
    ddt1(1) = db
    ddt1(2) = 0.d0
    call ddcpr (za%ddc, ddt1, ic1)
    call ddcpr (za%ddc(3), ddt1(3), ic2)
    if (ic1 .ne. 0 .or. ic2 .ne. 0) then
      dd_netzd = .true.
    else
      dd_netzd = .false.
    endif
    return
  end function

  function dd_netxz (xa, zb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    logical dd_netxz
    intent (in):: xa, zb
    call ddxzc (xa, ddt1)
    call ddcpr (ddt1, zb%ddc, ic1)
    call ddcpr (ddt1(3), zb%ddc(3), ic2)
    if (ic1 .ne. 0 .or. ic2 .ne. 0) then
      dd_netxz = .true.
    else
      dd_netxz = .false.
    endif
    return
  end function

  function dd_netzx (za, xb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    logical dd_netzx
    intent (in):: za, xb
    call ddxzc (xb, ddt2)
    call ddcpr (za%ddc, ddt2, ic1)
    call ddcpr (za%ddc(3), ddt2(3), ic2)
    if (ic1 .ne. 0 .or. ic2 .ne. 0) then
      dd_netzx = .true.
    else
      dd_netzx = .false.
    endif
    return
  end function

end module


module ddgenmod

!  This Fortran-90 module defines generic functions involving all
!  DD datatypes.

!  The subroutines and functions defined in this module are private
!  and not intended to be called directly by the user.  The generic
!  names (i.e. interface block names) are publicly accessible, though.

use ddfunmod
use dddefmod
private kdb
parameter (kdb = kind (0.d0))
real*8, private:: ddt0(4), ddt1(4), ddt2(4), ddt3(4), ddt4(4), &
  ddt5(4), ddt6(4)
private &
  dd_absq, dd_absz, dd_acos, dd_imag, dd_aint, dd_anint, dd_asin, dd_atan, &
  dd_atan2, dd_conjg, dd_cos, dd_cosz, dd_cosh, dd_qtod, dd_ztod, dd_qtox, &
  dd_ztox, dd_exp, dd_expz, dd_log, dd_logz, dd_log10, dd_maxq, dd_maxq3, &
  dd_minq, dd_minq3, dd_modq, dd_qtoz, dd_dtoz, dd_xtoz, dd_atoz, dd_qqtoz, &
  dd_ddtoz, dd_aatoz, dd_cssh, dd_cssn, dd_nrt, dd_poly, dd_rand, dd_inpq, &
  dd_inpz, dd_ztoq, dd_dtoq, dd_xtoq, dd_atoq, dd_itoq, dd_outq, dd_outz, &
  dd_pi, dd_signq, dd_sin, dd_sinz, dd_sinh, dd_sqrtq, dd_sqrtz, dd_tan, dd_tanh

!  DD generic interface blocks.

interface abs
  module procedure dd_absq
  module procedure dd_absz
end interface

interface acos
  module procedure dd_acos
end interface

interface aimag
  module procedure dd_imag
end interface

interface aint
  module procedure dd_aint
end interface

interface anint
  module procedure dd_anint
end interface

interface asin
  module procedure dd_asin
end interface

interface atan
  module procedure dd_atan
end interface

interface atan2
  module procedure dd_atan2
end interface

interface conjg
  module procedure dd_conjg
end interface

interface cos
  module procedure dd_cos
  module procedure dd_cosz
end interface

interface cosh
  module procedure dd_cosh
end interface

interface dble
  module procedure dd_qtod
  module procedure dd_ztod
end interface

interface dcmplx
  module procedure dd_qtox
  module procedure dd_ztox
end interface

interface exp
  module procedure dd_exp
  module procedure dd_expz
end interface

interface log
  module procedure dd_log
  module procedure dd_logz
end interface

interface log10
  module procedure dd_log10
end interface

interface max
  module procedure dd_maxq
  module procedure dd_maxq3
end interface

interface min
  module procedure dd_minq
  module procedure dd_minq3
end interface

interface mod
  module procedure dd_modq
end interface

interface ddcmpl
  module procedure dd_qtoz
  module procedure dd_dtoz
  module procedure dd_xtoz
  module procedure dd_qqtoz
  module procedure dd_ddtoz
end interface

interface ddcsshf
  module procedure dd_cssh
end interface

interface ddcssnf
  module procedure dd_cssn
end interface

interface ddnrtf
  module procedure dd_nrt
end interface

interface ddpi
  module procedure dd_pi
end interface

interface ddpolyr
  module procedure dd_poly
end interface

interface ddranf
  module procedure dd_rand
end interface

interface ddread
  module procedure dd_inpq
  module procedure dd_inpz
end interface

interface ddreal
  module procedure dd_ztoq
  module procedure dd_dtoq
  module procedure dd_xtoq
  module procedure dd_atoq
  module procedure dd_itoq
end interface

interface ddwrite
  module procedure dd_outq
  module procedure dd_outz
end interface

interface sign
  module procedure dd_signq
end interface

interface sin
  module procedure dd_sin
  module procedure dd_sinz
end interface

interface sinh
  module procedure dd_sinh
end interface

interface sqrt
  module procedure dd_sqrtq
  module procedure dd_sqrtz
end interface

interface tan
  module procedure dd_tan
end interface

interface tanh
  module procedure dd_tanh
end interface

contains

  function dd_absq (qa)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_absq
    intent (in):: qa
    call ddeq (qa%ddr, dd_absq%ddr)
    if (qa%ddr(1) .ge. 0.d0) then
      dd_absq%ddr(1) = qa%ddr(1)
      dd_absq%ddr(2) = qa%ddr(2)
    else
      dd_absq%ddr(1) = - qa%ddr(1)
      dd_absq%ddr(2) = - qa%ddr(2)
    endif
    return
  end function

  function dd_absz (za)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_absz
    intent (in):: za
    call ddmul (za%ddc, za%ddc, ddt1)
    call ddmul (za%ddc(3), za%ddc(3), ddt2)
    call ddadd (ddt1, ddt2, ddt3)
    call ddsqrt (ddt3, dd_absz%ddr)
    return
  end function

  function dd_acos (qa)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_acos
    intent (in):: qa
    ddt1(1) = 1.d0
    ddt1(2) = 0.d0
    call ddmul (qa%ddr, qa%ddr, ddt2)
    call ddsub (ddt1, ddt2, ddt3)
    call ddsqrt (ddt3, ddt1)
    call ddang (qa%ddr, ddt1, dd_acos%ddr)
    return
  end function

  function dd_aint (qa)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_aint
    intent (in):: qa
    call ddinfr (qa%ddr, dd_aint%ddr, ddt1)
    return
  end function

  function dd_anint (qa)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_anint
    intent (in):: qa
    call ddnint (qa%ddr, dd_anint%ddr)
    return
  end function

  function dd_asin (qa)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_asin
    intent (in):: qa
    ddt1(1) = 1.d0
    ddt1(2) = 0.d0
    call ddmul (qa%ddr, qa%ddr, ddt2)
    call ddsub (ddt1, ddt2, ddt3)
    call ddsqrt (ddt3, ddt1)
    call ddang (ddt1, qa%ddr, dd_asin%ddr)
    return
  end function

  function dd_atan (qa)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_atan
    intent (in):: qa
    ddt1(1) = 1.d0
    ddt1(2) = 0.d0
    call ddang (ddt1, qa%ddr, dd_atan%ddr)
    return
  end function

  function dd_atan2 (qa, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_atan2
    intent (in):: qa, qb
    call ddang (qb%ddr, qa%ddr, dd_atan2%ddr)
    return
  end function

  function dd_conjg (za)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_conjg
    intent (in):: za
    call ddceq (za%ddc, dd_conjg%ddc)
    dd_conjg%ddc(3) = - za%ddc(3)
    dd_conjg%ddc(4) = - za%ddc(4)
    return
  end function

  function dd_cos (qa)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_cos
    intent (in):: qa
    call ddcssn (qa%ddr, dd_cos%ddr, ddt1)
    return
  end function

  function dd_cosz (za)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_cosz
    intent (in):: za
    call ddeq (za%ddc(3), ddt2)
    ddt2(1) = - ddt2(1)
    ddt2(2) = - ddt2(2)
    call ddexp (ddt2, ddt1)
    ddt3(1) = 1.d0
    ddt3(2) = 0.d0
    call dddiv (ddt3, ddt1, ddt2)
    call ddcssn (za%ddc, ddt3, ddt4)
    call ddadd (ddt1, ddt2, ddt5)
    call ddmuld (ddt5, 0.5d0, ddt6)
    call ddmul (ddt6, ddt3, dd_cosz%ddc)
    call ddsub (ddt1, ddt2, ddt5)
    call ddmuld (ddt5, 0.5d0, ddt6)
    call ddmul (ddt6, ddt4, dd_cosz%ddc(3))
    return
  end function

  function dd_cosh (qa)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_cosh
    intent (in):: qa
    call ddcssh (qa%ddr, dd_cosh%ddr, ddt1)
    return
  end function

  function dd_qtod (qa)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    intent (in):: qa
    real*8:: dd_qtod, da
    dd_qtod = qa%ddr(1)
    return
  end function

  function dd_ztod (za)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    intent (in):: za
    real*8:: dd_ztod, da
    dd_ztod = za%ddc(1)
    return
  end function

  function dd_qtox (qa, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    complex (kdb):: dd_qtox
    intent (in):: qa, qb
    da = qa%ddr(1)
    db = qb%ddr(1)
    dd_qtox = cmplx (da, db, kdb)
    return
  end function

  function dd_ztox (za)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    complex (kdb):: dd_ztox
    intent (in):: za
    da = za%ddc(1)
    db = za%ddc(3)
    dd_ztox = cmplx (da, db, kdb)
    return
  end function

  function dd_exp (qa)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_exp
    intent (in):: qa
    call ddexp (qa%ddr, dd_exp%ddr)
    return
  end function

  function dd_expz (za)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_expz
    intent (in):: za
    call ddexp (za%ddc, ddt1)
    call ddcssn (za%ddc(3), ddt2, ddt3)
    call ddmul (ddt1, ddt2, dd_expz%ddc)
    call ddmul (ddt1, ddt3, dd_expz%ddc(3))
    return
  end function

  function dd_imag (za)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_imag
    intent (in):: za
    call ddeq (za%ddc(3), dd_imag%ddr)
    return
  end function

  function dd_log (qa)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_log
    intent (in):: qa
    call ddlog (qa%ddr, dd_log%ddr)
    return
  end function

  function dd_logz (za)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_logz
    intent (in):: za
    call ddmul (za%ddc, za%ddc, ddt1)
    call ddmul (za%ddc(3), za%ddc(3), ddt2)
    call ddadd (ddt1, ddt2, ddt3)
    call ddlog (ddt3, ddt4)
    call ddmuld (ddt4, 0.5d0, dd_logz%ddc)
    call ddang (za%ddc, za%ddc(3), dd_logz%ddc(3))
    return
  end function

  function dd_log10 (qa)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_log10
    intent (in):: qa
    call ddlog (qa%ddr, ddt1)
    ddt2(1) = 10.d0
    ddt2(2) = 0.d0
    call ddlog (ddt2, ddt3)
    call dddiv (ddt1, ddt3, dd_log10%ddr)
    return
  end function

  function dd_maxq (qa, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_maxq
    intent (in):: qa, qb
    call ddcpr (qa%ddr, qb%ddr, ic)
    if (ic .ge. 0) then
      call ddeq (qa%ddr, dd_maxq%ddr)
    else
      call ddeq (qb%ddr, dd_maxq%ddr)
    endif
    return
  end function

  function dd_maxq3 (qa, qb, qc)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_maxq3
    intent (in):: qa, qb, qc
    call ddcpr (qa%ddr, qb%ddr, ic)
    if (ic .ge. 0) then
      call ddeq (qa%ddr, ddt0)
    else
      call ddeq (qb%ddr, ddt0)
    endif
    call ddcpr (ddt0, qc%ddr, ic)
    if (ic .ge. 0) then
      call ddeq (ddt0, dd_maxq3%ddr)
    else
      call ddeq (qc%ddr, dd_maxq3%ddr)
    endif
    return
  end function

  function dd_minq (qa, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_minq
    intent (in):: qa, qb
    call ddcpr (qa%ddr, qb%ddr, ic)
    if (ic .lt. 0) then
      call ddeq (qa%ddr, dd_minq%ddr)
    else
      call ddeq (qb%ddr, dd_minq%ddr)
    endif
    return
  end function

  function dd_minq3 (qa, qb, qc)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_minq3
    intent (in):: qa, qb, qc
    call ddcpr (qa%ddr, qb%ddr, ic)
    if (ic .lt. 0) then
      call ddeq (qa%ddr, ddt0)
    else
      call ddeq (qb%ddr, ddt0)
    endif
    call ddcpr (ddt0, qc%ddr, ic)
    if (ic .lt. 0) then
      call ddeq (ddt0, dd_minq3%ddr)
    else
      call ddeq (qc%ddr, dd_minq3%ddr)
    endif
    return
  end function

  function dd_modq (qa, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_modq
    intent (in):: qa, qb
    call dddiv (qa%ddr, qb%ddr, ddt1)
    call ddinfr (ddt1, ddt2, ddt3)
    call ddmul (qb%ddr, ddt2, ddt1)
    call ddsub (qa%ddr, ddt1, dd_modq%ddr)
    return
  end function

  function dd_qtoz (qa)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_qtoz
    intent (in):: qa
    call ddmzc (qa%ddr, dd_qtoz%ddc)
    return
  end function

  function dd_dtoz (da)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_dtoz
    intent (in):: da
    xa = da
    call ddxzc (xa, dd_dtoz%ddc)
    return
  end function

  function dd_xtoz (xa)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_xtoz
    intent (in):: xa
    call ddxzc (xa, dd_xtoz%ddc)
    return
  end function

  function dd_qqtoz (qa, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_qqtoz
    intent (in):: qa, qb
    call ddqqc (qa%ddr, qb%ddr, dd_qqtoz%ddc)
    return
  end function

  function dd_ddtoz (da, db)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_ddtoz
    intent (in):: da, db
    xa = cmplx (da, db, kdb)
    call ddxzc (xa, dd_ddtoz%ddc)
    return
  end function

  subroutine dd_cssh (qa, qb, qc)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    intent (in):: qa
    intent (out):: qb, qc
    call ddcssh (qa%ddr, qb%ddr, qc%ddr)
    return
  end subroutine

  subroutine dd_cssn (qa, qb, qc)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    intent (in):: qa
    intent (out):: qb, qc
    call ddcssn (qa%ddr, qb%ddr, qc%ddr)
    return
  end subroutine

  function dd_nrt (qa, ib)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_nrt
    intent (in):: qa, ib
    call ddnrt (qa%ddr, ib, dd_nrt%ddr)
    return
  end function

  function dd_pi ()
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_pi
    call ddpic (dd_pi%ddr)
  end function    

  subroutine dd_poly (ia, qa, qb, qc)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: qa(0:ia)
    intent (in):: ia, qa, qb
    intent (out):: qc
    call ddpoly (ia, qa(0)%ddr, qb%ddr, qc%ddr)
    return
  end subroutine

  function dd_rand ()
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_rand
    call ddrand (dd_rand%ddr)
    return
  end function

  subroutine dd_inpq (iu, q1, q2, q3, q4, q5, q6, q7, q8, q9)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    intent (in):: q1, q2, q3, q4, q5, q6, q7, q8, q9
    optional:: q2, q3, q4, q5, q6, q7, q8, q9
    call ddinp (iu, q1%ddr)
    if (present (q2)) call ddinp (iu, q2%ddr)
    if (present (q3)) call ddinp (iu, q3%ddr)
    if (present (q4)) call ddinp (iu, q4%ddr)
    if (present (q5)) call ddinp (iu, q5%ddr)
    if (present (q6)) call ddinp (iu, q6%ddr)
    if (present (q7)) call ddinp (iu, q7%ddr)
    if (present (q8)) call ddinp (iu, q8%ddr)
    if (present (q9)) call ddinp (iu, q9%ddr)
    return
  end subroutine

  subroutine dd_inpz (iu, z1, z2, z3, z4, z5, z6, z7, z8, z9)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    intent (in):: z1, z2, z3, z4, z5, z6, z7, z8, z9
    optional:: z2, z3, z4, z5, z6, z7, z8, z9
    call ddinp (iu, z1%ddc)
    call ddinp (iu, z1%ddc(3))
    if (present (z2)) call ddinp (iu, z2%ddc)
    if (present (z2)) call ddinp (iu, z2%ddc(3))
    if (present (z3)) call ddinp (iu, z3%ddc)
    if (present (z3)) call ddinp (iu, z3%ddc(3))
    if (present (z4)) call ddinp (iu, z4%ddc)
    if (present (z4)) call ddinp (iu, z4%ddc(3))
    if (present (z5)) call ddinp (iu, z5%ddc)
    if (present (z5)) call ddinp (iu, z5%ddc(3))
    if (present (z6)) call ddinp (iu, z6%ddc)
    if (present (z6)) call ddinp (iu, z6%ddc(3))
    if (present (z7)) call ddinp (iu, z7%ddc)
    if (present (z7)) call ddinp (iu, z7%ddc(3))
    if (present (z8)) call ddinp (iu, z8%ddc)
    if (present (z8)) call ddinp (iu, z8%ddc(3))
    if (present (z9)) call ddinp (iu, z9%ddc)
    if (present (z9)) call ddinp (iu, z9%ddc(3))
    return
  end subroutine

  function dd_ztoq (za)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_ztoq
    intent (in):: za
    call ddeq (za%ddc, dd_ztoq%ddr)
    return
  end function

  function dd_dtoq (da)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_dtoq
    intent (in):: da
    dd_dtoq%ddr(1) = da
    dd_dtoq%ddr(2) = 0.d0
    return
  end function

  function dd_xtoq (xa)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_xtoq
    intent (in):: xa
    dd_xtoq%ddr(1) = xa
    dd_xtoq%ddr(2) = 0.d0
    return
  end function

  function dd_atoq (aa)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    character*(*), intent (in):: aa
    type (dd_real):: dd_atoq
    character*80 t
    t = aa
    call ddinpc (t, dd_atoq%ddr)
    return
  end function

  function dd_itoq (ia)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_itoq
    intent (in):: ia
    dd_itoq%ddr(1) = ia
    dd_itoq%ddr(2) = 0.d0
    return
  end function

  subroutine dd_outq (iu, q1, q2, q3, q4, q5, q6, q7, q8, q9)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    intent (in):: q1, q2, q3, q4, q5, q6, q7, q8, q9
    optional:: q2, q3, q4, q5, q6, q7, q8, q9
    call ddout (iu, q1%ddr)
    if (present (q2)) call ddout (iu, q2%ddr)
    if (present (q3)) call ddout (iu, q3%ddr)
    if (present (q4)) call ddout (iu, q4%ddr)
    if (present (q5)) call ddout (iu, q5%ddr)
    if (present (q6)) call ddout (iu, q6%ddr)
    if (present (q7)) call ddout (iu, q7%ddr)
    if (present (q8)) call ddout (iu, q8%ddr)
    if (present (q9)) call ddout (iu, q9%ddr)
     return
  end subroutine

  subroutine dd_outz (iu, z1, z2, z3, z4, z5, z6, z7, z8, z9)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    intent (in):: z1, z2, z3, z4, z5, z6, z7, z8, z9
    optional:: z2, z3, z4, z5, z6, z7, z8, z9
    call ddout (iu, z1%ddc)
    call ddout (iu, z1%ddc(3))
    if (present (z2)) call ddout (iu, z2%ddc)
    if (present (z2)) call ddout (iu, z2%ddc(3))
    if (present (z3)) call ddout (iu, z3%ddc)
    if (present (z3)) call ddout (iu, z3%ddc(3))
    if (present (z4)) call ddout (iu, z4%ddc)
    if (present (z4)) call ddout (iu, z4%ddc(3))
    if (present (z5)) call ddout (iu, z5%ddc)
    if (present (z5)) call ddout (iu, z5%ddc(3))
    if (present (z6)) call ddout (iu, z6%ddc)
    if (present (z6)) call ddout (iu, z6%ddc(3))
    if (present (z7)) call ddout (iu, z7%ddc)
    if (present (z7)) call ddout (iu, z7%ddc(3))
    if (present (z8)) call ddout (iu, z8%ddc)
    if (present (z8)) call ddout (iu, z8%ddc(3))
    if (present (z9)) call ddout (iu, z9%ddc)
    if (present (z9)) call ddout (iu, z9%ddc(3))
     return
  end subroutine

  function dd_signq_old (qa, qb)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_signq
    intent (in):: qa, qb
    call ddeq (qa%ddr, dd_signq%ddr)
    dd_signq%ddr(1) = sign (dd_signq%ddr(1), qb%ddr(1))
    return
  end function

  function dd_signq (qa, qb)
!
!   Modified by R. Pittau
!
    implicit real*8 (d), &
    type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_signq
    intent (in):: qa, qb
    if (qa%ddr(1)*qb%ddr(1) .gt. 0.d0) then
      dd_signq%ddr(1)= qa%ddr(1) 
      dd_signq%ddr(2)= qa%ddr(2) 
    else
      dd_signq%ddr(1)= -qa%ddr(1) 
      dd_signq%ddr(2)= -qa%ddr(2) 
    endif 
    return
  end function

  function dd_sin (qa)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_sin
    intent (in):: qa
    call ddcssn (qa%ddr, ddt1, dd_sin%ddr)
    return
  end function

  function dd_sinz (za)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_sinz
    intent (in):: za
    call ddeq (za%ddc(3), ddt2)
    ddt2(1) = - ddt2(1)
    ddt2(2) = - ddt2(2)
    call ddexp (ddt2, ddt1)
    ddt3(1) = 1.d0
    ddt3(2) = 0.d0
    call dddiv (ddt3, ddt1, ddt2)
    call ddcssn (za%ddc, ddt3, ddt4)
    call ddadd (ddt1, ddt2, ddt5)
    call ddmuld (ddt5, 0.5d0, ddt6)
    call ddmul (ddt6, ddt4, dd_sinz%ddc)
    call ddsub (ddt1, ddt2, ddt5)
    call ddmuld (ddt5, -0.5d0, ddt6)
    call ddmul (ddt6, ddt3, dd_sinz%ddc(3))
    return
  end function

  function dd_sinh (qa)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_sinh
    intent (in):: qa
    call ddcssh (qa%ddr, ddt1, dd_sinh%ddr)
    return
  end function

  function dd_sqrtq (qa)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_sqrtq
    intent (in):: qa
    call ddsqrt (qa%ddr, dd_sqrtq%ddr)
    return
  end function

  function dd_sqrtz (za)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_complex):: dd_sqrtz
    intent (in):: za
    call ddcsqrt (za%ddc, dd_sqrtz%ddc)
    return
  end function

  function dd_tan (qa)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_tan
    intent (in):: qa
    call ddcssn (qa%ddr, ddt1, ddt2)
    call dddiv (ddt2, ddt1, dd_tan%ddr)
    return
  end function

  function dd_tanh (qa)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_tanh
    intent (in):: qa
    call ddcssh (qa%ddr, ddt1, ddt2)
    call dddiv (ddt2, ddt1, dd_tanh%ddr)
    return
  end function

end module

!   This contains defines bessel, besselexp, erf, erfc and gamma functions.
!   David H Bailey    2004-07-08

module ddfunsubmod
use ddfunmod
use dddefmod
use ddrealmod
use ddgenmod
private dd_bessel, dd_besselexp, dd_erf, dd_erfc, dd_gamma
integer, private:: kdb
parameter (kdb = kind (0.d0))

interface bessel
  module procedure dd_bessel
end interface

interface besselexp
  module procedure dd_besselexp
end interface

interface erf
  module procedure dd_erf
end interface

interface erfc
  module procedure dd_erfc
end interface

interface gamma
  module procedure dd_gamma
end interface

contains

  function dd_bessel (qa)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_bessel
    intent (in):: qa
    call ddbessel (qa, dd_bessel)
    return
  end function

  function dd_besselexp (qa)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_besselexp
    intent (in):: qa
    call ddbesselexp (qa, dd_besselexp)
    return
  end function

  function dd_erf (qa)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_erf
    intent (in):: qa
    type (dd_real) q1, q2
    q1%ddr(1) = 1.d0
    q1%ddr(2) = 0.d0
    call dderfc (qa, q2)
    call ddsub (q1%ddr, q2%ddr, dd_erf%ddr)
    return
  end function

  function dd_erfc (qa)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_erfc
    intent (in):: qa
    call dderfc (qa, dd_erfc)
    return
  end function

  function dd_gamma (qa)
    implicit real*8 (d), &
      type (dd_real) (q), complex (kdb) (x), type (dd_complex) (z)
    type (dd_real):: dd_gamma
    intent (in):: qa
    call ddgamma (qa, dd_gamma)
    return
  end function

subroutine ddbessel (t, z)

!   This evaluates the function BesselI (0, t).

implicit none
integer i, ndp, neps
type (dd_real) eps, tsum, t, t1, t2, z, ddpicx
parameter (ndp = 32, neps = -36)

ddpicx = ddpi ()
eps = 2.d0 ** neps

!   Select either the direct or the asymptotic series.

if (0.85d0 * t < dble (ndp)) then
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
      call ddwrite (6, t)
      tsum = 0.d0
      goto 110
    endif
  enddo

  write (6, *) 'bessel: loop overflow 2'
  tsum = 0.d0

110 continue

  t1 = tsum * exp (t) / sqrt (2.d0 * ddpicx * t)
endif

z = t1
return
end subroutine

subroutine ddbesselexp (t, z)

!   This evaluates the function BesselI (0, t) / exp (t).

implicit none
integer i, ndp, neps
type (dd_real) eps, tsum, t, t1, t2, z, ddpicx
parameter (ndp = 32, neps = -36)

ddpicx = ddpi ()
eps = 2.d0 ** neps

!   Select either the direct or the asymptotic series.

if (0.85d0 * t < dble (ndp)) then
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
      call ddwrite (6, t)
      tsum = 0.d0
      goto 110
    endif
  enddo

  write (6, *) 'besselexp: loop overflow 2'
  tsum = 0.d0

110 continue

  t1 = tsum / sqrt (2.d0 * ddpicx * t)
endif

z = t1
return
end subroutine

subroutine dderfc (t, z)

!   Computes erfc(a) = 1 - Int_0^a 2/sqrt(pi) * e^(-4t^2) dt.

!   This algorithm is presented in Richard Crandall's book "Topics in
!   Advanced Scientific Computation", pg 82.  Crandall in turn references
!   a 1968 paper by Chiarella and Reichel.

  implicit none
  integer i, j, k, n, ndp1, ndps, ntab, nwks, neps
  type (dd_real) eps, f, t, t1, t2, t3, t4, t5, z, ddpicx
  real*8 alpha, d1, d2, dpi, dlog10, dlog2
  type (dd_real) etab (:)
  allocatable etab
  save ntab, nwks, alpha, etab
  data ntab/0/
  parameter (ndp1 = 32, neps = -36)

  ddpicx = ddpi ()
  eps = 10.d0 ** neps
  dpi = acos (-1.d0)
  dlog10 = log (10.d0)
  dlog2 = log (2.d0)
  d1 = t
  if (d1 > 10000.d0) then
    z = 0.d0
    goto 200
  endif
  d2 = dpi / d1

  if (ntab == 0 .or. d2 < alpha) then

!   On the first call, or if working precision has been increased, or if
!   the argument exceeds a certain value, recalculate alpha and the etab table.

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
      write (6, *) 'dderfcx: error; contact author'
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

z = t3 * alpha * t / ddpicx * (1.d0 / t2 + 2.d0 * t1) &
       + 2.d0 / (1.d0 - exp (2.d0 * ddpicx * t / alpha))

200 continue

  return
end subroutine

subroutine ddgamma (t, z)

!   This evaluates the gamma function, using an algorithm of R. W. Potter.

implicit none
integer i, j, k, ndp, neps, nt
double precision alpha, con1, con2, d1, d2
parameter (con1 = 1.151292547d0, con2 = 1.974476770d0)
type (dd_real) eps, sum1, sum2, t, t1, t2, t3, t4, z, ddpicx
parameter (ndp = 32, neps = -36)

eps = 10.d0 ** neps
ddpicx = ddpi ()

!   Handle special arguments.

if (abs (t) > 1.d8) then
  write (6, *) 'gamma: argument too large'
  goto 120
elseif (t == anint (t)) then
  if (t <= 0.d0) then
    write (6, *) 'gamma: invalid negative argument'
    z = 0.d0
    goto 120
  endif
  nt = dble (t)
  t1 = 1.d0

  do i = 2, nt - 1
    t1 = dble (i) * t1
  enddo

  z = t1
  goto 120
endif

!   Calculate alpha, then take the next highest integer value, so that
!   d2 = 0.25 * alpha^2 can be calculated exactly in double precision.

alpha = aint (con1 * ndp + 1.d0)
t1 = t
d2 = 0.25d0 * alpha**2
t3 = 1.d0 / t1
sum1 = t3

!   Evaluate the series with t, terminating when t3 < sum1 * epsilon.

do j = 1, 1000000000
  t3 = t3 * d2 / (dble (j) * (t1 + dble (j)))
  sum1 = sum1 + t3
  if (abs (t3) < abs (sum1) * eps) goto 100
enddo

write (6, *) 'gamma: loop overflow 1'
sum1 = 0.d0

100 continue

sum1 = t1 * (0.5d0 * alpha) ** t1 * sum1
t1 = -t
t3 = 1.d0 / t1
sum2 = t3

!   Evaluate the same series with -t, terminating when t3 < sum1 * epsilon.

do j = 1, 1000000000
  t3 = t3 * d2 / (dble (j) * (t1 + dble (j)))
  sum2 = sum2 + t3
  if (abs (t3) < abs (sum2) * eps) goto 110
enddo

write (6, *) 'gamma: loop overflow 2'
sum2 = 0.d0

110 continue

sum2 = t1 * (0.5d0 * alpha) ** t1 * sum2

!   Conclude with this square root expression.

z = sqrt (ddpicx * sum1 / (t * sin (ddpicx * t) * sum2))

120 continue

return
end subroutine

end module

module ddmodule
use ddfunmod
use ddrealmod
use ddcmpmod
use ddgenmod
use ddfunsubmod
end module
