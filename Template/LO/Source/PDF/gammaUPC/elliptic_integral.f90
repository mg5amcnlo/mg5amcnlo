! this file is from https://people.sc.fsu.edu/~jburkardt/f_src/elliptic_integral/elliptic_integral.html
! there are also a C version and a C++ version and a FORTRAN77 version and a MATLAB version and a Python version

function elliptic_ea ( a )

!*****************************************************************************80
!
!! ELLIPTIC_EA evaluates the complete elliptic integral E(A).
!
!  Discussion:
!
!    The value is computed using Carlson elliptic integrals:
!
!      E(a) = RF ( 0, 1-sin^2(a), 1 ) - 1/3 sin^2(a) RD ( 0, 1-sin^2(a), 1 ).
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    30 May 2018
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input, real ( kind = rk ) A, the argument.
!
!    Output, real ( kind = rk ) ELLIPTIC_EA, the function value.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  real ( kind = rk ) a
  real ( kind = rk ) elliptic_ea
  real ( kind = rk ) errtol
  integer ierr
  real ( kind = rk ) k
  real ( kind = rk ), parameter :: r8_pi = 3.141592653589793D+00
  real ( kind = rk ) rd
  real ( kind = rk ) rf
  real ( kind = rk ) value
  real ( kind = rk ) x
  real ( kind = rk ) y
  real ( kind = rk ) z

  k = sin ( a * r8_pi / 180.0D+00 )

  x = 0.0D+00
  y = ( 1.0D+00 - k ) * ( 1.0D+00 + k )
  z = 1.0D+00
  errtol = 1.0D-03

  value = rf ( x, y, z, errtol, ierr ) &
    - k * k * rd ( x, y, z, errtol, ierr ) / 3.0D+00

  elliptic_ea = value

  return
end
subroutine elliptic_ea_values ( n_data, x, fx )

!*****************************************************************************80
!
!! ELLIPTIC_EA_VALUES returns values of the complete elliptic integral E(A).
!
!  Discussion:
!
!    This is one form of what is sometimes called the complete elliptic
!    integral of the second kind.
!
!    The function is defined by the formula:
!
!      E(A) = integral ( 0 <= T <= PI/2 )
!        sqrt ( 1 - sin ( A )^2 * sin ( T )^2 ) dT
!
!    In Mathematica, the function can be evaluated by:
!
!      EllipticE[(Sin[Pi*a/180])^2]
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    19 August 2004
!
!  Author:
!
!    John Burkardt
!
!  Reference:
!
!    Milton Abramowitz, Irene Stegun,
!    Handbook of Mathematical Functions,
!    National Bureau of Standards, 1964,
!    ISBN: 0-486-61272-4,
!    LC: QA47.A34.
!
!    Stephen Wolfram,
!    The Mathematica Book,
!    Fourth Edition,
!    Cambridge University Press, 1999,
!    ISBN: 0-521-64314-7,
!    LC: QA76.95.W65.
!
!  Parameters:
!
!    Input/output, integer N_DATA.  The user sets N_DATA to 0
!    before the first call.  On each call, the routine increments N_DATA by 1,
!    and returns the corresponding data; when there is no more data, the
!    output value of N_DATA will be 0 again.
!
!    Output, real ( kind = rk ) X, the argument of the function, measured
!    in degrees.
!
!    Output, real ( kind = rk ) FX, the value of the function.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  integer, parameter :: n_max = 19

  real ( kind = rk ) fx
  real ( kind = rk ), save, dimension ( n_max ) :: fx_vec = (/ &
    1.570796326794897D+00, &
    1.567809073977622D+00, &
    1.558887196601596D+00, &
    1.544150496914673D+00, &
    1.523799205259774D+00, &
    1.498114928422116D+00, &
    1.467462209339427D+00, &
    1.432290969306756D+00, &
    1.393140248523812D+00, &
    1.350643881047676D+00, &
    1.305539094297794D+00, &
    1.258679624779997D+00, &
    1.211056027568459D+00, &
    1.163827964493139D+00, &
    1.118377737969864D+00, &
    1.076405113076403D+00, &
    1.040114395706010D+00, &
    1.012663506234396D+00, &
    1.000000000000000D+00 /)
  integer n_data
  real ( kind = rk ) x
  real ( kind = rk ), save, dimension ( n_max ) :: x_vec = (/ &
     0.0D+00, &
     5.0D+00, &
    10.0D+00, &
    15.0D+00, &
    20.0D+00, &
    25.0D+00, &
    30.0D+00, &
    35.0D+00, &
    40.0D+00, &
    45.0D+00, &
    50.0D+00, &
    55.0D+00, &
    60.0D+00, &
    65.0D+00, &
    70.0D+00, &
    75.0D+00, &
    80.0D+00, &
    85.0D+00, &
    90.0D+00 /)

  if ( n_data < 0 ) then
    n_data = 0
  end if

  n_data = n_data + 1

  if ( n_max < n_data ) then
    n_data = 0
    x = 0.0D+00
    fx = 0.0D+00
  else
    x = x_vec(n_data)
    fx = fx_vec(n_data)
  end if

  return
end
function elliptic_ek ( k )

!*****************************************************************************80
!
!! ELLIPTIC_EK evaluates the complete elliptic integral E(K).
!
!  Discussion:
!
!    The value is computed using Carlson elliptic integrals:
!
!      E(k) = RF ( 0, 1-k^2, 1 ) - 1/3 k^2 RD ( 0, 1-k^2, 1 ).
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    30 May 2018
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input, real ( kind = rk ) K, the argument.
!
!    Output, real ( kind = rk ) ELLIPTIC_EK, the function value.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  real ( kind = rk ) elliptic_ek
  real ( kind = rk ) errtol
  integer ierr
  real ( kind = rk ) k
  real ( kind = rk ) rd
  real ( kind = rk ) rf
  real ( kind = rk ) value
  real ( kind = rk ) x
  real ( kind = rk ) y
  real ( kind = rk ) z

  x = 0.0D+00
  y = ( 1.0D+00 - k ) * ( 1.0D+00 + k )
  z = 1.0D+00
  errtol = 1.0D-03

  value = rf ( x, y, z, errtol, ierr ) &
    - k * k * rd ( x, y, z, errtol, ierr ) / 3.0D+00

  elliptic_ek = value

  return
end
subroutine elliptic_ek_values ( n_data, x, fx )

!*****************************************************************************80
!
!! ELLIPTIC_EK_VALUES returns values of the complete elliptic integral E(K).
!
!  Discussion:
!
!    This is one form of what is sometimes called the complete elliptic
!    integral of the second kind.
!
!    The function is defined by the formula:
!
!      E(K) = integral ( 0 <= T <= PI/2 )
!        sqrt ( 1 - K^2 * sin ( T )^2 ) dT
!
!    In Mathematica, the function can be evaluated by:
!
!      EllipticE[m]
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    29 May 2018
!
!  Author:
!
!    John Burkardt
!
!  Reference:
!
!    Milton Abramowitz, Irene Stegun,
!    Handbook of Mathematical Functions,
!    National Bureau of Standards, 1964,
!    ISBN: 0-486-61272-4,
!    LC: QA47.A34.
!
!    Stephen Wolfram,
!    The Mathematica Book,
!    Fourth Edition,
!    Cambridge University Press, 1999,
!    ISBN: 0-521-64314-7,
!    LC: QA76.95.W65.
!
!  Parameters:
!
!    Input/output, integer N_DATA.  The user sets N_DATA to 0
!    before the first call.  On each call, the routine increments N_DATA by 1,
!    and returns the corresponding data; when there is no more data, the
!    output value of N_DATA will be 0 again.
!
!    Output, real ( kind = rk ) X, the argument of the function.
!
!    Output, real ( kind = rk ) FX, the value of the function.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  integer, parameter :: n_max = 21

  real ( kind = rk ) fx
  real ( kind = rk ), save, dimension ( n_max ) :: fx_vec = (/ &
    1.570796326794897D+00, &
    1.550973351780472D+00, &
    1.530757636897763D+00, &
    1.510121832092819D+00, &
    1.489035058095853D+00, &
    1.467462209339427D+00, &
    1.445363064412665D+00, &
    1.422691133490879D+00, &
    1.399392138897432D+00, &
    1.375401971871116D+00, &
    1.350643881047676D+00, &
    1.325024497958230D+00, &
    1.298428035046913D+00, &
    1.270707479650149D+00, &
    1.241670567945823D+00, &
    1.211056027568459D+00, &
    1.178489924327839D+00, &
    1.143395791883166D+00, &
    1.104774732704073D+00, &
    1.060473727766278D+00, &
    1.000000000000000D+00 /)
  integer n_data
  real ( kind = rk ) x
  real ( kind = rk ), save, dimension ( n_max ) :: x_vec = (/ &
    0.0000000000000000D+00, &
    0.2236067977499790D+00, &
    0.3162277660168379D+00, &
    0.3872983346207417D+00, &
    0.4472135954999579D+00, &
    0.5000000000000000D+00, &
    0.5477225575051661D+00, &
    0.5916079783099616D+00, &
    0.6324555320336759D+00, &
    0.6708203932499369D+00, &
    0.7071067811865476D+00, &
    0.7416198487095663D+00, &
    0.7745966692414834D+00, &
    0.8062257748298550D+00, &
    0.8366600265340756D+00, &
    0.8660254037844386D+00, &
    0.8944271909999159D+00, &
    0.9219544457292888D+00, &
    0.9486832980505138D+00, &
    0.9746794344808963D+00, &
    1.0000000000000000D+00 /)

  if ( n_data < 0 ) then
    n_data = 0
  end if

  n_data = n_data + 1

  if ( n_max < n_data ) then
    n_data = 0
    x = 0.0D+00
    fx = 0.0D+00
  else
    x = x_vec(n_data)
    fx = fx_vec(n_data)
  end if

  return
end
! the argument m must be smaller than 1 (can be negative)
! m<1
! This has been checked with EllipticE[m]
function elliptic_em ( m )

!*****************************************************************************80
!
!! ELLIPTIC_EM evaluates the complete elliptic integral E(M).
!
!  Discussion:
!
!    The value is computed using Carlson elliptic integrals:
!
!      E(m) = RF ( 0, 1-m, 1 ) - 1/3 m RD ( 0, 1-m, 1 ).
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    30 May 2018
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input, real ( kind = rk ) M, the argument.
!
!    Output, real ( kind = rk ) ELLIPTIC_EM, the function value.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  real ( kind = rk ) elliptic_em
  real ( kind = rk ) errtol
  integer ierr
  real ( kind = rk ) m
  real ( kind = rk ) rd
  real ( kind = rk ) rf
  real ( kind = rk ) value
  real ( kind = rk ) x
  real ( kind = rk ) y
  real ( kind = rk ) z

  x = 0.0D+00
  y = 1.0D+00 - m
  z = 1.0D+00
  errtol = 1.0D-03

  value = rf ( x, y, z, errtol, ierr ) &
    - m * rd ( x, y, z, errtol, ierr ) / 3.0D+00

  elliptic_em = value

  return
end
subroutine elliptic_em_values ( n_data, x, fx )

!*****************************************************************************80
!
!! ELLIPTIC_EM_VALUES returns values of the complete elliptic integral E(M).
!
!  Discussion:
!
!    This is one form of what is sometimes called the complete elliptic
!    integral of the second kind.
!
!    The function is defined by the formula:
!
!      E(M) = integral ( 0 <= T <= PI/2 )
!        sqrt ( 1 - M * sin ( T )^2 ) dT
!
!    In Mathematica, the function can be evaluated by:
!
!      EllipticE[m]
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    14 August 2004
!
!  Author:
!
!    John Burkardt
!
!  Reference:
!
!    Milton Abramowitz, Irene Stegun,
!    Handbook of Mathematical Functions,
!    National Bureau of Standards, 1964,
!    ISBN: 0-486-61272-4,
!    LC: QA47.A34.
!
!    Stephen Wolfram,
!    The Mathematica Book,
!    Fourth Edition,
!    Cambridge University Press, 1999,
!    ISBN: 0-521-64314-7,
!    LC: QA76.95.W65.
!
!  Parameters:
!
!    Input/output, integer N_DATA.  The user sets N_DATA to 0
!    before the first call.  On each call, the routine increments N_DATA by 1,
!    and returns the corresponding data; when there is no more data, the
!    output value of N_DATA will be 0 again.
!
!    Output, real ( kind = rk ) X, the argument of the function.
!
!    Output, real ( kind = rk ) FX, the value of the function.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  integer, parameter :: n_max = 21

  real ( kind = rk ) fx
  real ( kind = rk ), save, dimension ( n_max ) :: fx_vec = (/ &
    1.570796326794897D+00, &
    1.550973351780472D+00, &
    1.530757636897763D+00, &
    1.510121832092819D+00, &
    1.489035058095853D+00, &
    1.467462209339427D+00, &
    1.445363064412665D+00, &
    1.422691133490879D+00, &
    1.399392138897432D+00, &
    1.375401971871116D+00, &
    1.350643881047676D+00, &
    1.325024497958230D+00, &
    1.298428035046913D+00, &
    1.270707479650149D+00, &
    1.241670567945823D+00, &
    1.211056027568459D+00, &
    1.178489924327839D+00, &
    1.143395791883166D+00, &
    1.104774732704073D+00, &
    1.060473727766278D+00, &
    1.000000000000000D+00 /)
  integer n_data
  real ( kind = rk ) x
  real ( kind = rk ), save, dimension ( n_max ) :: x_vec = (/ &
    0.00D+00, &
    0.05D+00, &
    0.10D+00, &
    0.15D+00, &
    0.20D+00, &
    0.25D+00, &
    0.30D+00, &
    0.35D+00, &
    0.40D+00, &
    0.45D+00, &
    0.50D+00, &
    0.55D+00, &
    0.60D+00, &
    0.65D+00, &
    0.70D+00, &
    0.75D+00, &
    0.80D+00, &
    0.85D+00, &
    0.90D+00, &
    0.95D+00, &
    1.00D+00 /)

  if ( n_data < 0 ) then
    n_data = 0
  end if

  n_data = n_data + 1

  if ( n_max < n_data ) then
    n_data = 0
    x = 0.0D+00
    fx = 0.0D+00
  else
    x = x_vec(n_data)
    fx = fx_vec(n_data)
  end if

  return
end
function elliptic_fa ( a )

!*****************************************************************************80
!
!! ELLIPTIC_FA evaluates the complete elliptic integral F(A).
!
!  Discussion:
!
!    The value is computed using Carlson elliptic integrals:
!
!      F(a) = RF ( 0, 1-sin^2(a), 1 ).
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    29 May 2018
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input, real ( kind = rk ) A, the argument.
!
!    Output, real ( kind = rk ) ELLIPTIC_FA, the function value.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  real ( kind = rk ) a
  real ( kind = rk ) elliptic_fa
  real ( kind = rk ) errtol
  integer ierr
  real ( kind = rk ), parameter :: r8_pi = 3.141592653589793D+00
  real ( kind = rk ) rf
  real ( kind = rk ) value
  real ( kind = rk ) x
  real ( kind = rk ) y
  real ( kind = rk ) z

  x = 0.0D+00
  y = 1.0D+00 - ( sin ( a * r8_pi / 180.0 ) ) ** 2
  z = 1.0D+00
  errtol = 1.0D-03

  value = rf ( x, y, z, errtol, ierr )

  elliptic_fa = value

  return
end
subroutine elliptic_fa_values ( n_data, x, fx )

!*****************************************************************************80
!
!! ELLIPTIC_FA_VALUES returns values of the complete elliptic integral F(A).
!
!  Discussion:
!
!    This is one form of what is sometimes called the complete elliptic integral
!    of the first kind.
!
!    The function is defined by the formula:
!
!      F(A) = integral ( 0 <= T <= PI/2 )
!        dT / sqrt ( 1 - sin ( A )^2 * sin ( T )^2 )
!
!    In Mathematica, the function can be evaluated by:
!
!      EllipticK[(Sin[a*Pi/180])^2]
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    19 August 2004
!
!  Author:
!
!    John Burkardt
!
!  Reference:
!
!    Milton Abramowitz, Irene Stegun,
!    Handbook of Mathematical Functions,
!    National Bureau of Standards, 1964,
!    ISBN: 0-486-61272-4,
!    LC: QA47.A34.
!
!    Stephen Wolfram,
!    The Mathematica Book,
!    Fourth Edition,
!    Cambridge University Press, 1999,
!    ISBN: 0-521-64314-7,
!    LC: QA76.95.W65.
!
!  Parameters:
!
!    Input/output, integer N_DATA.  The user sets N_DATA to 0
!    before the first call.  On each call, the routine increments N_DATA by 1,
!    and returns the corresponding data; when there is no more data, the
!    output value of N_DATA will be 0 again.
!
!    Output, real ( kind = rk ) X, the argument of the function, measured
!    in degrees.
!
!    Output, real ( kind = rk ) FX, the value of the function.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  integer, parameter :: n_max = 18

  real ( kind = rk ) fx
  real ( kind = rk ), save, dimension ( n_max ) :: fx_vec = (/ &
    0.1570796326794897D+01, &
    0.1573792130924768D+01, &
    0.1582842804338351D+01, &
    0.1598142002112540D+01, &
    0.1620025899124204D+01, &
    0.1648995218478530D+01, &
    0.1685750354812596D+01, &
    0.1731245175657058D+01, &
    0.1786769134885021D+01, &
    0.1854074677301372D+01, &
    0.1935581096004722D+01, &
    0.2034715312185791D+01, &
    0.2156515647499643D+01, &
    0.2308786798167196D+01, &
    0.2504550079001634D+01, &
    0.2768063145368768D+01, &
    0.3153385251887839D+01, &
    0.3831741999784146D+01 /)
  integer n_data
  real ( kind = rk ) x
  real ( kind = rk ), save, dimension ( n_max ) :: x_vec = (/ &
     0.0D+00, &
     5.0D+00, &
    10.0D+00, &
    15.0D+00, &
    20.0D+00, &
    25.0D+00, &
    30.0D+00, &
    35.0D+00, &
    40.0D+00, &
    45.0D+00, &
    50.0D+00, &
    55.0D+00, &
    60.0D+00, &
    65.0D+00, &
    70.0D+00, &
    75.0D+00, &
    80.0D+00, &
    85.0D+00 /)

  if ( n_data < 0 ) then
    n_data = 0
  end if

  n_data = n_data + 1

  if ( n_max < n_data ) then
    n_data = 0
    x = 0.0D+00
    fx = 0.0D+00
  else
    x = x_vec(n_data)
    fx = fx_vec(n_data)
  end if

  return
end
function elliptic_fk ( k )

!*****************************************************************************80
!
!! ELLIPTIC_FK evaluates the complete elliptic integral F(K).
!
!  Discussion:
!
!    The value is computed using Carlson elliptic integrals:
!
!      F(k) = RF ( 0, 1-k^2, 1 ).
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    29 May 2018
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input, real ( kind = rk ) K, the argument.
!
!    Output, real ( kind = rk ) ELLIPTIC_FK, the function value.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  real ( kind = rk ) elliptic_fk
  real ( kind = rk ) errtol
  integer ierr
  real ( kind = rk ) k
  real ( kind = rk ) rf
  real ( kind = rk ) value
  real ( kind = rk ) x
  real ( kind = rk ) y
  real ( kind = rk ) z

  x = 0.0D+00
  y = ( 1.0D+00 - k ) * ( 1.0D+00 + k )
  z = 1.0D+00
  errtol = 1.0D-03

  value = rf ( x, y, z, errtol, ierr )

  elliptic_fk = value

  return
end
subroutine elliptic_fk_values ( n_data, x, fx )

!*****************************************************************************80
!
!! ELLIPTIC_FK_VALUES returns values of the complete elliptic integral F(K).
!
!  Discussion:
!
!    This is one form of what is sometimes called the complete elliptic
!    integral of the first kind.
!
!    The function is defined by the formula:
!
!      F(K) = integral ( 0 <= T <= PI/2 )
!        dT / sqrt ( 1 - K^2 * sin ( T )^2 )
!
!    In Mathematica, the function can be evaluated by:
!
!      EllipticK[k^2]
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    10 August 2004
!
!  Author:
!
!    John Burkardt
!
!  Reference:
!
!    Milton Abramowitz, Irene Stegun,
!    Handbook of Mathematical Functions,
!    National Bureau of Standards, 1964,
!    ISBN: 0-486-61272-4,
!    LC: QA47.A34.
!
!    Stephen Wolfram,
!    The Mathematica Book,
!    Fourth Edition,
!    Cambridge University Press, 1999,
!    ISBN: 0-521-64314-7,
!    LC: QA76.95.W65.
!
!  Parameters:
!
!    Input/output, integer N_DATA.  The user sets N_DATA to 0
!    before the first call.  On each call, the routine increments N_DATA by 1,
!    and returns the corresponding data; when there is no more data, the
!    output value of N_DATA will be 0 again.
!
!    Output, real ( kind = rk ) X, the argument of the function.
!
!    Output, real ( kind = rk ) FX, the value of the function.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  integer, parameter :: n_max = 20

  real ( kind = rk ) fx
  real ( kind = rk ), save, dimension ( n_max ) :: fx_vec = (/ &
    1.570796326794897D+00, &
    1.591003453790792D+00, &
    1.612441348720219D+00, &
    1.635256732264580D+00, &
    1.659623598610528D+00, &
    1.685750354812596D+00, &
    1.713889448178791D+00, &
    1.744350597225613D+00, &
    1.777519371491253D+00, &
    1.813883936816983D+00, &
    1.854074677301372D+00, &
    1.898924910271554D+00, &
    1.949567749806026D+00, &
    2.007598398424376D+00, &
    2.075363135292469D+00, &
    2.156515647499643D+00, &
    2.257205326820854D+00, &
    2.389016486325580D+00, &
    2.578092113348173D+00, &
    2.908337248444552D+00 /)
  integer n_data
  real ( kind = rk ) x
  real ( kind = rk ), save, dimension ( n_max ) :: x_vec = (/ &
     0.0000000000000000D+00, &
     0.2236067977499790D+00, &
     0.3162277660168379D+00, &
     0.3872983346207417D+00, &
     0.4472135954999579D+00, &
     0.5000000000000000D+00, &
     0.5477225575051661D+00, &
     0.5916079783099616D+00, &
     0.6324555320336759D+00, &
     0.6708203932499369D+00, &
     0.7071067811865476D+00, &
     0.7416198487095663D+00, &
     0.7745966692414834D+00, &
     0.8062257748298550D+00, &
     0.8366600265340756D+00, &
     0.8660254037844386D+00, &
     0.8944271909999159D+00, &
     0.9219544457292888D+00, &
     0.9486832980505138D+00, &
     0.9746794344808963D+00 /)

  if ( n_data < 0 ) then
    n_data = 0
  end if

  n_data = n_data + 1

  if ( n_max < n_data ) then
    n_data = 0
    x = 0.0D+00
    fx = 0.0D+00
  else
    x = x_vec(n_data)
    fx = fx_vec(n_data)
  end if

  return
end
function elliptic_fm ( m )

!*****************************************************************************80
!
!! ELLIPTIC_FM evaluates the complete elliptic integral F(M).
!
!  Discussion:
!
!    The value is computed using Carlson elliptic integrals:
!
!      F(m) = RF ( 0, 1-m, 1 ).
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    29 May 2018
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input, real ( kind = rk ) M, the argument.
!
!    Output, real ( kind = rk ) ELLIPTIC_FM, the function value.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  real ( kind = rk ) elliptic_fm
  real ( kind = rk ) errtol
  integer ierr
  real ( kind = rk ) m
  real ( kind = rk ) rf
  real ( kind = rk ) value
  real ( kind = rk ) x
  real ( kind = rk ) y
  real ( kind = rk ) z

  x = 0.0D+00
  y = 1.0D+00 - m
  z = 1.0D+00
  errtol = 1.0D-03

  value = rf ( x, y, z, errtol, ierr )

  elliptic_fm = value

  return
end
subroutine elliptic_fm_values ( n_data, x, fx )

!*****************************************************************************80
!
!! ELLIPTIC_FM_VALUES returns values of the complete elliptic integral F(M).
!
!  Discussion:
!
!    This is one form of what is sometimes called the complete elliptic
!    integral of the first kind.
!
!    The function is defined by the formula:
!
!      F(M) = integral ( 0 <= T <= PI/2 )
!        dT / sqrt ( 1 - M * sin ( T )^2 )
!
!    In Mathematica, the function can be evaluated by:
!
!      EllipticK[m]
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    10 August 2004
!
!  Author:
!
!    John Burkardt
!
!  Reference:
!
!    Milton Abramowitz, Irene Stegun,
!    Handbook of Mathematical Functions,
!    National Bureau of Standards, 1964,
!    ISBN: 0-486-61272-4,
!    LC: QA47.A34.
!
!    Stephen Wolfram,
!    The Mathematica Book,
!    Fourth Edition,
!    Cambridge University Press, 1999,
!    ISBN: 0-521-64314-7,
!    LC: QA76.95.W65.
!
!  Parameters:
!
!    Input/output, integer N_DATA.  The user sets N_DATA to 0
!    before the first call.  On each call, the routine increments N_DATA by 1,
!    and returns the corresponding data; when there is no more data, the
!    output value of N_DATA will be 0 again.
!
!    Output, real ( kind = rk ) X, the argument of the function.
!
!    Output, real ( kind = rk ) FX, the value of the function.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  integer, parameter :: n_max = 20

  real ( kind = rk ) fx
  real ( kind = rk ), save, dimension ( n_max ) :: fx_vec = (/ &
    1.570796326794897D+00, &
    1.591003453790792D+00, &
    1.612441348720219D+00, &
    1.635256732264580D+00, &
    1.659623598610528D+00, &
    1.685750354812596D+00, &
    1.713889448178791D+00, &
    1.744350597225613D+00, &
    1.777519371491253D+00, &
    1.813883936816983D+00, &
    1.854074677301372D+00, &
    1.898924910271554D+00, &
    1.949567749806026D+00, &
    2.007598398424376D+00, &
    2.075363135292469D+00, &
    2.156515647499643D+00, &
    2.257205326820854D+00, &
    2.389016486325580D+00, &
    2.578092113348173D+00, &
    2.908337248444552D+00 /)
  integer n_data
  real ( kind = rk ) x
  real ( kind = rk ), save, dimension ( n_max ) :: x_vec = (/ &
     0.00D+00, &
     0.05D+00, &
     0.10D+00, &
     0.15D+00, &
     0.20D+00, &
     0.25D+00, &
     0.30D+00, &
     0.35D+00, &
     0.40D+00, &
     0.45D+00, &
     0.50D+00, &
     0.55D+00, &
     0.60D+00, &
     0.65D+00, &
     0.70D+00, &
     0.75D+00, &
     0.80D+00, &
     0.85D+00, &
     0.90D+00, &
     0.95D+00 /)

  if ( n_data < 0 ) then
    n_data = 0
  end if

  n_data = n_data + 1

  if ( n_max < n_data ) then
    n_data = 0
    x = 0.0D+00
    fx = 0.0D+00
  else
    x = x_vec(n_data)
    fx = fx_vec(n_data)
  end if

  return
end
function elliptic_inc_ea ( phi, a )

!*****************************************************************************80
!
!! ELLIPTIC_INC_EA evaluates the incomplete elliptic integral E(PHI,A).
!
!  Discussion:
!
!    The value is computed using Carlson elliptic integrals:
!
!      k = sin ( a * pi / 180 )
!      E(phi,a) = 
!                  sin ( phi )   RF ( cos^2 ( phi ), 1-k^2 sin^2 ( phi ), 1 ) 
!        - 1/3 k^2 sin^3 ( phi ) RD ( cos^2 ( phi ), 1-k^2 sin^2 ( phi ), 1 ).
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    30 May 2018
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input, real ( kind = rk ) PHI, A, the argument.
!    0 <= PHI <= PI/2.
!    0 <= sin^2 ( A * pi / 180 ) * sin^2(PHI) <= 1.
!
!    Output, real ( kind = rk ) ELLIPTIC_INC_EA, the function value.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  real ( kind = rk ) a
  real ( kind = rk ) cp
  real ( kind = rk ) elliptic_inc_ea
  real ( kind = rk ) errtol
  integer ierr
  real ( kind = rk ) k
  real ( kind = rk ) phi
  real ( kind = rk ), parameter :: r8_pi = 3.141592653589793D+00
  real ( kind = rk ) rd
  real ( kind = rk ) rf
  real ( kind = rk ) sp
  real ( kind = rk ) value
  real ( kind = rk ) value1
  real ( kind = rk ) value2
  real ( kind = rk ) x
  real ( kind = rk ) y
  real ( kind = rk ) z

  k = sin ( a * r8_pi / 180.0D+00 )

  cp = cos ( phi )
  sp = sin ( phi )
  x = cp * cp
  y = ( 1.0D+00 - k * sp ) * ( 1.0D+00 + k * sp )
  z = 1.0D+00
  errtol = 1.0D-03

  value1 = rf ( x, y, z, errtol, ierr )

  if ( ierr /= 0 ) then
    write ( *, '(a)' ) ''
    write ( *, '(a)' ) 'ELLIPTIC_INC_EA - Fatal error!'
    write ( *, '(a,i2)' ) '  RF returned IERR = ', ierr
    stop 1
  end if

  value2 = rd ( x, y, z, errtol, ierr )

  if ( ierr /= 0 ) then
    write ( *, '(a)' ) ''
    write ( *, '(a)' ) 'ELLIPTIC_INC_EA - Fatal error!'
    write ( *, '(a,i2)' ) '  RD returned IERR = ', ierr
    stop 1
  end if

  value = sp * value1 - k ** 2 * sp ** 3 * value2 / 3.0D+00

  elliptic_inc_ea = value

  return
end
subroutine elliptic_inc_ea_values ( n_data, phi, a, ea )

!*****************************************************************************80
!
!! ELLIPTIC_INC_EA_VALUES: values of the incomplete elliptic integral E(PHI,A).
!
!  Discussion:
!
!    This is one form of the incomplete elliptic integral of the second kind.
!
!      E(PHI,A) = integral ( 0 <= T <= PHI ) 
!        sqrt ( 1 - sin^2 ( A ) * sin^2 ( T ) ) dT
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    24 June 2018
!
!  Author:
!
!    John Burkardt
!
!  Reference:
!
!    Milton Abramowitz, Irene Stegun,
!    Handbook of Mathematical Functions,
!    US Department of Commerce, 1964.
!
!    Stephen Wolfram,
!    The Mathematica Book,
!    Fourth Edition,
!    Wolfram Media / Cambridge University Press, 1999.
!
!  Parameters:
!
!    Input/output, integer N_DATA.  The user sets N_DATA to 0 
!    before the first call.  On each call, the routine increments N_DATA by 1, 
!    and returns the corresponding data when there is no more data, the
!    output value of N_DATA will be 0 again.
!
!    Output, real ( kind = rk ) PHI, A, the arguments of the function.
!
!    Output, real ( kind = rk ) EA, the value of the function.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  integer, parameter :: n_max = 20

  real ( kind = rk ) a
  real ( kind = rk ) ea
  integer n_data
  real ( kind = rk ) phi

  real ( kind = rk ), save, dimension ( n_max ) :: a_vec = (/ &
         123.0821233267548D+00, &
         11.26931745051486D+00, &
        -94.88806452075445D+00, &
        -99.71407853545323D+00, &
         57.05881039324191D+00, &
        -19.71363287074183D+00, &
         56.31230299738043D+00, &
        -91.55605346417718D+00, &
        -27.00654574696468D+00, &
        -169.2293728595904D+00, &
         61.96859564803047D+00, &
        -158.7324398933148D+00, &
         105.0883958999383D+00, &
        -48.95883872360177D+00, &
        -42.58568835110901D+00, &
         11.65603284687828D+00, &
        -8.398113719173338D+00, &
         17.69362213019626D+00, &
          73.8803420626852D+00, &
        -69.82492339645128D+00 /)

  real ( kind = rk ), save, dimension ( n_max ) :: ea_vec = (/ &
        0.3384181367348019D+00, &
         1.292924624509506D+00, &
        0.6074183768796306D+00, &
        0.3939726730783567D+00, &
       0.06880814097089803D+00, &
        0.0969436473376824D+00, &
        0.6025937791452033D+00, &
        0.9500549494837583D+00, &
         1.342783372140486D+00, &
        0.1484915631401388D+00, &
         1.085432887050926D+00, &
        0.1932136916085597D+00, &
        0.3983689593057807D+00, &
        0.1780054133336934D+00, &
         1.164525270273536D+00, &
         1.080167047541845D+00, &
         1.346684963830312D+00, &
         1.402100272685504D+00, &
        0.2928091845544553D+00, &
        0.5889342583405707D+00 /)

   real ( kind = rk ), save, dimension ( n_max ) :: phi_vec = (/ &
        0.3430906586047127D+00, &
         1.302990057703935D+00, &
        0.6523628380743488D+00, &
        0.4046022501376546D+00, &
       0.06884642871852312D+00, &
        0.0969609046794745D+00, &
         0.630370432896175D+00, &
         1.252375418911598D+00, &
         1.409796082144801D+00, &
        0.1485105463502483D+00, &
         1.349466184634646D+00, &
        0.1933711786970301D+00, &
        0.4088829927466769D+00, &
        0.1785430666405224D+00, &
         1.292588374416351D+00, &
         1.087095515757691D+00, &
         1.352794600489329D+00, &
         1.432530166308616D+00, &
        0.2968093345769761D+00, &
        0.6235880396594726D+00 /)

  if ( n_data < 0 ) then
    n_data = 0
  end if

  n_data = n_data + 1

  if ( n_max < n_data ) then
    n_data = 0
    a = 0.0D+00
    ea = 0.0D+00
    phi = 0.0D+00
  else
    a = a_vec(n_data)
    ea = ea_vec(n_data)
    phi = phi_vec(n_data)
  end if

  return
end
function elliptic_inc_ek ( phi, k )

!*****************************************************************************80
!
!! ELLIPTIC_INC_EK evaluates the incomplete elliptic integral E(PHI,K).
!
!  Discussion:
!
!    The value is computed using Carlson elliptic integrals:
!
!      E(phi,k) = 
!                  sin ( phi )   RF ( cos^2 ( phi ), 1-k^2 sin^2 ( phi ), 1 ) 
!        - 1/3 k^2 sin^3 ( phi ) RD ( cos^2 ( phi ), 1-k^2 sin^2 ( phi ), 1 ).
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    30 May 2018
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input, real ( kind = rk ) PHI, K, the argument.
!    0 <= PHI <= PI/2.
!    0 <= K^2 * sin^2(PHI) <= 1.
!
!    Output, real ( kind = rk ) ELLIPTIC_INC_EK, the function value.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  real ( kind = rk ) cp
  real ( kind = rk ) elliptic_inc_ek
  real ( kind = rk ) errtol
  integer ierr
  real ( kind = rk ) k
  real ( kind = rk ) phi
  real ( kind = rk ), parameter :: r8_pi = 3.141592653589793D+00
  real ( kind = rk ) rd
  real ( kind = rk ) rf
  real ( kind = rk ) sp
  real ( kind = rk ) value
  real ( kind = rk ) value1
  real ( kind = rk ) value2
  real ( kind = rk ) x
  real ( kind = rk ) y
  real ( kind = rk ) z

  cp = cos ( phi )
  sp = sin ( phi )
  x = cp * cp
  y = ( 1.0D+00 - k * sp ) * ( 1.0D+00 + k * sp )
  z = 1.0D+00
  errtol = 1.0D-03

  value1 = rf ( x, y, z, errtol, ierr )

  if ( ierr /= 0 ) then
    write ( *, '(a)' ) ''
    write ( *, '(a)' ) 'ELLIPTIC_INC_EK - Fatal error!'
    write ( *, '(a,i2)' ) '  RF returned IERR = ', ierr
    stop 1
  end if

  value2 = rd ( x, y, z, errtol, ierr )

  if ( ierr /= 0 ) then
    write ( *, '(a)' ) ''
    write ( *, '(a)' ) 'ELLIPTIC_INC_EK - Fatal error!'
    write ( *, '(a,i2)' ) '  RD returned IERR = ', ierr
    stop 1
  end if

  value = sp * value1 - k ** 2 * sp ** 3 * value2 / 3.0D+00

  elliptic_inc_ek = value

  return
end
subroutine elliptic_inc_ek_values ( n_data, phi, k, ek )

!*****************************************************************************80
!
!! ELLIPTIC_INC_EK_VALUES: values of the incomplete elliptic integral E(PHI,K).
!
!  Discussion:
!
!    This is the incomplete elliptic integral of the second kind.
!
!      E(PHI,K) = integral ( 0 <= T <= PHI ) 
!        sqrt ( 1 - K^2 * sin ( T )^2 ) dT
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    22 June 2018
!
!  Author:
!
!    John Burkardt
!
!  Reference:
!
!    Milton Abramowitz, Irene Stegun,
!    Handbook of Mathematical Functions,
!    US Department of Commerce, 1964.
!
!    Stephen Wolfram,
!    The Mathematica Book,
!    Fourth Edition,
!    Wolfram Media / Cambridge University Press, 1999.
!
!  Parameters:
!
!    Input/output, integer N_DATA.  The user sets N_DATA to 0 
!    before the first call.  On each call, the routine increments N_DATA by 1, 
!    and returns the corresponding data when there is no more data, the
!    output value of N_DATA will be 0 again.
!
!    Output, real ( kind = rk ) PHI, K, the arguments.
!
!    Output, real ( kind = rk ) EK, the function value.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  integer, parameter :: n_max = 20

  real ( kind = rk ) ek
  real ( kind = rk ) k
  integer n_data
  real ( kind = rk ) phi

  real ( kind = rk ), save, dimension ( n_max ) :: ek_vec = (/ &
        0.2852345328295404D+00, &
         1.298690225567921D+00, &
        0.5508100202571943D+00, &
        0.3575401358115371D+00, &
       0.06801307805507453D+00, &
       0.09679584980231837D+00, &
        0.6003112504412838D+00, &
        0.8996717721794724D+00, &
         1.380715261453875D+00, &
        0.1191644625202453D+00, &
         1.196994838171557D+00, &
        0.1536260979667945D+00, &
        0.3546768920544152D+00, &
        0.1758756066650882D+00, &
         1.229819109410569D+00, &
          1.08381066114337D+00, &
          1.35023378157378D+00, &
         1.419775884709218D+00, &
        0.2824895528020034D+00, &
        0.5770427720982867D+00 /)

  real ( kind = rk ), save, dimension ( n_max ) :: k_vec = (/ &
         2.712952582080266D+00, &
        0.1279518954120547D+00, &
        -1.429437513650137D+00, &
        -1.981659235625333D+00, &
         3.894801879555818D+00, &
        -1.042486024983672D+00, &
        0.8641142168759754D+00, &
        -1.049058412826877D+00, &
       -0.3024062128402472D+00, &
        -6.574288841527263D+00, &
        0.6987397421988888D+00, &
         -5.12558591600033D+00, &
         2.074947853793764D+00, &
        -1.670886158426681D+00, &
       -0.4843595000931672D+00, &
        0.1393061679635559D+00, &
       -0.0946527302537008D+00, &
        0.1977207111754007D+00, &
         1.788159919089993D+00, &
        -1.077780624681256D+00 /)

  real ( kind = rk ), save, dimension ( n_max ) :: phi_vec = (/ &
        0.3430906586047127D+00, &
         1.302990057703935D+00, &
        0.6523628380743488D+00, &
        0.4046022501376546D+00, &
       0.06884642871852312D+00, &
        0.0969609046794745D+00, &
         0.630370432896175D+00, &
         1.252375418911598D+00, &
         1.409796082144801D+00, &
        0.1485105463502483D+00, &
         1.349466184634646D+00, &
        0.1933711786970301D+00, &
        0.4088829927466769D+00, &
        0.1785430666405224D+00, &
         1.292588374416351D+00, &
         1.087095515757691D+00, &
         1.352794600489329D+00, &
         1.432530166308616D+00, &
        0.2968093345769761D+00, &
        0.6235880396594726D+00 /)

  if ( n_data < 0 ) then
    n_data = 0
  end if

  n_data = n_data + 1

  if ( n_max < n_data ) then
    n_data = 0
    ek = 0.0D+00
    k = 0.0D+00
    phi = 0.0D+00
  else
    ek = ek_vec(n_data)
    k = k_vec(n_data)
    phi = phi_vec(n_data)
  end if

  return
end
function elliptic_inc_em ( phi, m )

!*****************************************************************************80
!
!! ELLIPTIC_INC_EM evaluates the incomplete elliptic integral E(PHI,M).
!
!  Discussion:
!
!    The value is computed using Carlson elliptic integrals:
!
!      E(phi,m) = 
!                sin ( phi )   RF ( cos^2 ( phi ), 1-m sin^2 ( phi ), 1 ) 
!        - 1/3 m sin^3 ( phi ) RD ( cos^2 ( phi ), 1-m sin^2 ( phi ), 1 ).
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    30 May 2018
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input, real ( kind = rk ) PHI, K, the argument.
!    0 <= PHI <= PI/2.
!    0 <= M * sin^2(PHI) <= 1.
!
!    Output, real ( kind = rk ) ELLIPTIC_INC_EM, the function value.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  real ( kind = rk ) cp
  real ( kind = rk ) elliptic_inc_em
  real ( kind = rk ) errtol
  integer ierr
  real ( kind = rk ) m
  real ( kind = rk ) phi
  real ( kind = rk ), parameter :: r8_pi = 3.141592653589793D+00
  real ( kind = rk ) rd
  real ( kind = rk ) rf
  real ( kind = rk ) sp
  real ( kind = rk ) value
  real ( kind = rk ) value1
  real ( kind = rk ) value2
  real ( kind = rk ) x
  real ( kind = rk ) y
  real ( kind = rk ) z

  cp = cos ( phi )
  sp = sin ( phi )
  x = cp * cp
  y = 1.0D+00 - m * sp * sp
  z = 1.0D+00
  errtol = 1.0D-03

  value1 = rf ( x, y, z, errtol, ierr )

  if ( ierr /= 0 ) then
    write ( *, '(a)' ) ''
    write ( *, '(a)' ) 'ELLIPTIC_INC_EM - Fatal error!'
    write ( *, '(a,i2)' ) '  RF returned IERR = ', ierr
    stop 1
  end if

  value2 = rd ( x, y, z, errtol, ierr )

  if ( ierr /= 0 ) then
    write ( *, '(a)' ) ''
    write ( *, '(a)' ) 'ELLIPTIC_INC_EM - Fatal error!'
    write ( *, '(a,i2)' ) '  RD returned IERR = ', ierr
    stop 1
  end if

  value = sp * value1 - m * sp ** 3 * value2 / 3.0D+00

  elliptic_inc_em = value

  return
end
subroutine elliptic_inc_em_values ( n_data, phi, m, em )

!*****************************************************************************80
!
!! ELLIPTIC_INC_EM_VALUES: values of the incomplete elliptic integral E(PHI,M).
!
!  Discussion:
!
!    This is the incomplete elliptic integral of the second kind.
!
!      E(PHI,M) = integral ( 0 <= T <= PHI ) 
!        sqrt ( 1 - M * sin ( T )^2 ) dT
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    24 June 2018
!
!  Author:
!
!    John Burkardt
!
!  Reference:
!
!    Milton Abramowitz, Irene Stegun,
!    Handbook of Mathematical Functions,
!    US Department of Commerce, 1964.
!
!    Stephen Wolfram,
!    The Mathematica Book,
!    Fourth Edition,
!    Wolfram Media / Cambridge University Press, 1999.
!
!  Parameters:
!
!    Input/output, integer N_DATA.  The user sets N_DATA to 0 
!    before the first call.  On each call, the routine increments N_DATA by 1, 
!    and returns the corresponding data when there is no more data, the
!    output value of N_DATA will be 0 again.
!
!    Output, real ( kind = rk ) PHI, M, the arguments.
!
!    Output, real ( kind = rk ) EM, the function value.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  integer, parameter :: n_max = 20

  real ( kind = rk ) em
  real ( kind = rk ) m
  integer n_data
  real ( kind = rk ) phi

  real ( kind = rk ), save, dimension ( n_max ) :: em_vec = (/ &
        0.2732317284159052D+00, &
         1.124749725099781D+00, &
        0.6446601913679151D+00, &
        0.3968902354370061D+00, &
       0.06063960799944668D+00, &
       0.08909411577948728D+00, &
         0.532402014802015D+00, &
         1.251888640660265D+00, &
          1.28897116191626D+00, &
        0.1481718153599732D+00, &
         1.038090185639913D+00, &
        0.1931275771541276D+00, &
        0.3304419611986801D+00, &
         0.167394796063963D+00, &
         1.214501175324736D+00, &
        0.9516560179840655D+00, &
         1.203682959526176D+00, &
         1.206426326185419D+00, &
        0.2522791382096692D+00, &
        0.6026499038720986D+00 /)

  real ( kind = rk ), save, dimension ( n_max ) :: m_vec = (/ &
         8.450689756874594D+00, &
        0.6039878267930615D+00, &
        0.1794126658351454D+00, &
        0.7095689301026752D+00, &
         133.9643389059188D+00, &
         47.96621393936416D+00, &
         2.172070586163255D+00, &
      0.002038130569431913D+00, &
        0.3600036705339421D+00, &
        0.6219544540067304D+00, &
        0.8834215943508453D+00, &
        0.2034290670379481D+00, &
         5.772526076430922D+00, &
         11.14853902343298D+00, &
        0.2889238477277305D+00, &
        0.7166617182589116D+00, &
        0.4760623731559658D+00, &
        0.6094948502068943D+00, &
         8.902276887883076D+00, &
        0.5434439226321253D+00 /)

  real ( kind = rk ), save, dimension ( n_max ) :: phi_vec = (/ &
      0.3430906586047127D+00, &
       1.302990057703935D+00, &
      0.6523628380743488D+00, &
      0.4046022501376546D+00, &
     0.06884642871852312D+00, &
      0.0969609046794745D+00, &
       0.630370432896175D+00, &
       1.252375418911598D+00, &
       1.409796082144801D+00, &
      0.1485105463502483D+00, &
       1.349466184634646D+00, &
      0.1933711786970301D+00, &
      0.4088829927466769D+00, &
      0.1785430666405224D+00, &
       1.292588374416351D+00, &
       1.087095515757691D+00, &
       1.352794600489329D+00, &
       1.432530166308616D+00, &
      0.2968093345769761D+00, &
      0.6235880396594726D+00 /)

  if ( n_data < 0 ) then
    n_data = 0
  end if

  n_data = n_data + 1

  if ( n_max < n_data ) then
    n_data = 0
    em = 0.0D+00
    m = 0.0D+00
    phi = 0.0D+00
  else
    em = em_vec(n_data)
    m = m_vec(n_data)
    phi = phi_vec(n_data)
  end if

  return
end
function elliptic_inc_fa ( phi, a )

!*****************************************************************************80
!
!! ELLIPTIC_INC_FA evaluates the incomplete elliptic integral F(PHI,A).
!
!  Discussion:
!
!    The value is computed using Carlson elliptic integrals:
!
!      k = sin ( a * pi / 180 )
!      F(phi,k) = sin(phi) * RF ( cos^2 ( phi ), 1-k^2 sin^2 ( phi ), 1 )
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    24 June 2018
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input, real ( kind = rk ) PHI, A, the argument.
!    0 <= PHI <= PI/2.
!    0 <= sin^2 ( A * pi / 180 ) * sin^2(PHI) <= 1.
!
!    Output, real ( kind = rk ) ELLIPTIC_INC_FA, the function value.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  real ( kind = rk ) a
  real ( kind = rk ) cp
  real ( kind = rk ) elliptic_inc_fa
  real ( kind = rk ) errtol
  integer ierr
  real ( kind = rk ) k
  real ( kind = rk ) phi
  real ( kind = rk ), parameter :: r8_pi = 3.141592653589793D+00
  real ( kind = rk ) rf
  real ( kind = rk ) sp
  real ( kind = rk ) value
  real ( kind = rk ) x
  real ( kind = rk ) y
  real ( kind = rk ) z

  k = sin ( a * r8_pi / 180.0D+00 )

  cp = cos ( phi )
  sp = sin ( phi )
  x = cp * cp
  y = ( 1.0D+00 - k * sp ) * ( 1.0D+00 + k * sp )
  z = 1.0D+00
  errtol = 1.0D-03

  value = rf ( x, y, z, errtol, ierr )

  if ( ierr /= 0 ) then
    write ( *, '(a)' ) ''
    write ( *, '(a)' ) 'ELLIPTIC_INC_FA - Fatal error!'
    write ( *, '(a,i2)' ) '  RF returned IERR = ', ierr
    stop 1
  end if

  elliptic_inc_fa = sp * value

  return
end
subroutine elliptic_inc_fa_values ( n_data, phi, a, fa )

!*****************************************************************************80
!
!! ELLIPTIC_INC_FA_VALUES: values of the incomplete elliptic integral F(PHI,A).
!
!  Discussion:
!
!    This is the incomplete elliptic integral of the first kind.
!
!      F(PHI,A) = integral ( 0 <= T <= PHI ) 
!        dT / sqrt ( 1 - sin^2 ( A ) * sin^2 ( T ) )
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    22 June 2018
!
!  Author:
!
!    John Burkardt
!
!  Reference:
!
!    Milton Abramowitz, Irene Stegun,
!    Handbook of Mathematical Functions,
!    US Department of Commerce, 1964.
!
!    Stephen Wolfram,
!    The Mathematica Book,
!    Fourth Edition,
!    Wolfram Media / Cambridge University Press, 1999.
!
!  Parameters:
!
!    Input/output, integer N_DATA.  The user sets N_DATA to 0 
!    before the first call.  On each call, the routine increments N_DATA by 1, 
!    and returns the corresponding data when there is no more data, the
!    output value of N_DATA will be 0 again.
!
!    Output, real ( kind = rk ) PHI, A, the arguments.
!
!    Output, real ( kind = rk ) FA, the function value.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  integer, parameter :: n_max = 20

  real ( kind = rk ) a
  real ( kind = rk ) fa
  integer n_data
  real ( kind = rk ) phi

  real ( kind = rk ), save, dimension ( n_max ) :: a_vec = (/ &
         123.0821233267548D+00, &
         11.26931745051486D+00, &
        -94.88806452075445D+00, &
        -99.71407853545323D+00, &
         57.05881039324191D+00, &
        -19.71363287074183D+00, &
         56.31230299738043D+00, &
        -91.55605346417718D+00, &
        -27.00654574696468D+00, &
        -169.2293728595904D+00, &
         61.96859564803047D+00, &
        -158.7324398933148D+00, &
         105.0883958999383D+00, &
        -48.95883872360177D+00, &
        -42.58568835110901D+00, &
         11.65603284687828D+00, &
        -8.398113719173338D+00, &
         17.69362213019626D+00, &
          73.8803420626852D+00, &
        -69.82492339645128D+00 /)

  real ( kind = rk ), save, dimension ( n_max ) :: fa_vec = (/ &
        0.3478806460316299D+00, &
         1.313180577009584D+00, &
        0.7037956689264326D+00, &
        0.4157626844675118D+00, &
       0.06888475483285136D+00, &
       0.09697816754845832D+00, &
        0.6605394722518515D+00, &
          1.82758346036751D+00, &
         1.482258783392487D+00, &
        0.1485295339221232D+00, &
         1.753800062701494D+00, &
         0.193528896465351D+00, &
        0.4199100508706138D+00, &
        0.1790836490491233D+00, &
         1.446048832279763D+00, &
         1.094097652100984D+00, &
         1.358947908427035D+00, &
          1.46400078231538D+00, &
        0.3009092014525799D+00, &
        0.6621341112075102D+00 /)

  real ( kind = rk ), save, dimension ( n_max ) :: phi_vec = (/ &
        0.3430906586047127D+00, &
         1.302990057703935D+00, &
        0.6523628380743488D+00, &
        0.4046022501376546D+00, &
       0.06884642871852312D+00, &
        0.0969609046794745D+00, &
         0.630370432896175D+00, &
         1.252375418911598D+00, &
         1.409796082144801D+00, &
        0.1485105463502483D+00, &
         1.349466184634646D+00, &
        0.1933711786970301D+00, &
        0.4088829927466769D+00, &
        0.1785430666405224D+00, &
         1.292588374416351D+00, &
         1.087095515757691D+00, &
         1.352794600489329D+00, &
         1.432530166308616D+00, &
        0.2968093345769761D+00, &
        0.6235880396594726D+00 /)

  if ( n_data < 0 ) then
    n_data = 0
  end if

  n_data = n_data + 1

  if ( n_max < n_data ) then
    n_data = 0
    a = 0.0D+00
    fa = 0.0D+00
    phi = 0.0D+00
  else
    a = a_vec(n_data)
    fa = fa_vec(n_data)
    phi = phi_vec(n_data)
  end if

  return
end
function elliptic_inc_fk ( phi, k )

!*****************************************************************************80
!
!! ELLIPTIC_INC_FK evaluates the incomplete elliptic integral F(PHI,K).
!
!  Discussion:
!
!    The value is computed using Carlson elliptic integrals:
!
!      F(phi,k) = sin(phi) * RF ( cos^2 ( phi ), 1-k^2 sin^2 ( phi ), 1 )
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    24 June 2018
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input, real ( kind = rk ) PHI, K, the argument.
!    0 <= PHI <= PI/2.
!    0 <= K^2 * sin^2(PHI) <= 1.
!
!    Output, real ( kind = rk ) ELLIPTIC_INC_FK, the function value.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  real ( kind = rk ) cp
  real ( kind = rk ) elliptic_inc_fk
  real ( kind = rk ) errtol
  integer ierr
  real ( kind = rk ) k
  real ( kind = rk ) phi
  real ( kind = rk ), parameter :: r8_pi = 3.141592653589793D+00
  real ( kind = rk ) rf
  real ( kind = rk ) sp
  real ( kind = rk ) value
  real ( kind = rk ) x
  real ( kind = rk ) y
  real ( kind = rk ) z

  cp = cos ( phi )
  sp = sin ( phi )
  x = cp * cp
  y = ( 1.0D+00 - k * sp ) * ( 1.0D+00 + k * sp )
  z = 1.0D+00
  errtol = 1.0D-03

  value = rf ( x, y, z, errtol, ierr )

  if ( ierr /= 0 ) then
    write ( *, '(a)' ) ''
    write ( *, '(a)' ) 'ELLIPTIC_INC_FK - Fatal error!'
    write ( *, '(a,i2)' ) '  RF returned IERR = ', ierr
    stop 1
  end if

  elliptic_inc_fk = sp * value

  return
end
subroutine elliptic_inc_fk_values ( n_data, phi, k, fk )

!*****************************************************************************80
!
!! ELLIPTIC_INC_FK_VALUES: values of the incomplete elliptic integral F(PHI,K).
!
!  Discussion:
!
!    This is the incomplete elliptic integral of the first kind.
!
!      F(PHI,K) = integral ( 0 <= T <= PHI ) 
!        dT / sqrt ( 1 - K^2 * sin ( T )^2 )
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    22 June 2018
!
!  Author:
!
!    John Burkardt
!
!  Reference:
!
!    Milton Abramowitz, Irene Stegun,
!    Handbook of Mathematical Functions,
!    US Department of Commerce, 1964.
!
!    Stephen Wolfram,
!    The Mathematica Book,
!    Fourth Edition,
!    Wolfram Media / Cambridge University Press, 1999.
!
!  Parameters:
!
!    Input/output, integer N_DATA.  The user sets N_DATA to 0 
!    before the first call.  On each call, the routine increments N_DATA by 1, 
!    and returns the corresponding data when there is no more data, the
!    output value of N_DATA will be 0 again.
!
!    Output, real ( kind = rk ) PHI, K, the arguments.
!
!    Output, real ( kind = rk ) FK, the value of the function.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  integer, parameter :: n_max = 20

  real ( kind = rk ) fk
  real ( kind = rk ) k
  integer n_data
  real ( kind = rk ) phi

  real ( kind = rk ), save, dimension ( n_max ) :: fk_vec = (/ &
       0.4340870330108736D+00, &
        1.307312511398114D+00, &
       0.8005154258533936D+00, &
       0.4656721451084328D+00, &
      0.06969849613441773D+00, &
      0.09712646708750489D+00, &
       0.6632598061016007D+00, &
          2.2308677858579D+00, &
        1.439846282888019D+00, &
       0.2043389243773096D+00, &
        1.537183574881771D+00, &
       0.2749229901565622D+00, &
       0.4828388342828284D+00, &
       0.1812848567886627D+00, &
        1.360729522341841D+00, &
         1.09039680912027D+00, &
        1.355363051581808D+00, &
        1.445462819732441D+00, &
       0.3125355489354676D+00, &
       0.6775731623807174D+00 /)

  real ( kind = rk ), save, dimension ( n_max ) :: k_vec = (/ &
        2.712952582080266D+00, &
       0.1279518954120547D+00, &
       -1.429437513650137D+00, &
       -1.981659235625333D+00, &
        3.894801879555818D+00, &
       -1.042486024983672D+00, &
       0.8641142168759754D+00, &
       -1.049058412826877D+00, &
      -0.3024062128402472D+00, &
       -6.574288841527263D+00, &
       0.6987397421988888D+00, &
        -5.12558591600033D+00, &
        2.074947853793764D+00, &
       -1.670886158426681D+00, &
      -0.4843595000931672D+00, &
       0.1393061679635559D+00, &
      -0.0946527302537008D+00, &
       0.1977207111754007D+00, &
        1.788159919089993D+00, &
       -1.077780624681256D+00 /)

  real ( kind = rk ), save, dimension ( n_max ) :: phi_vec = (/ &
       0.3430906586047127D+00, &
        1.302990057703935D+00, &
       0.6523628380743488D+00, &
       0.4046022501376546D+00, &
      0.06884642871852312D+00, &
       0.0969609046794745D+00, &
        0.630370432896175D+00, &
        1.252375418911598D+00, &
        1.409796082144801D+00, &
       0.1485105463502483D+00, &
        1.349466184634646D+00, &
       0.1933711786970301D+00, &
       0.4088829927466769D+00, &
       0.1785430666405224D+00, &
        1.292588374416351D+00, &
        1.087095515757691D+00, &
        1.352794600489329D+00, &
        1.432530166308616D+00, &
       0.2968093345769761D+00, &
       0.6235880396594726D+00 /)

  if ( n_data < 0 ) then
    n_data = 0
  end if

  n_data = n_data + 1

  if ( n_max < n_data ) then
    n_data = 0
    fk = 0.0D+00
    k = 0.0D+00
    phi = 0.0D+00
  else
    fk = fk_vec(n_data)
    k = k_vec(n_data)
    phi = phi_vec(n_data)
  end if

  return
end
function elliptic_inc_fm ( phi, m )

!*****************************************************************************80
!
!! ELLIPTIC_INC_FM evaluates the incomplete elliptic integral F(PHI,M).
!
!  Discussion:
!
!    The value is computed using Carlson elliptic integrals:
!
!      F(phi,m) = sin(phi) * RF ( cos^2 ( phi ), 1-m sin^2 ( phi ), 1 )
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    24 June 2018
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input, real ( kind = rk ) PHI, M, the argument.
!    0 <= PHI <= PI/2.
!    0 <= M * sin^2(PHI) <= 1.
!
!    Output, real ( kind = rk ) ELLIPTIC_INC_FM, the function value.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  real ( kind = rk ) cp
  real ( kind = rk ) elliptic_inc_fm
  real ( kind = rk ) errtol
  integer ierr
  real ( kind = rk ) m
  real ( kind = rk ) phi
  real ( kind = rk ), parameter :: r8_pi = 3.141592653589793D+00
  real ( kind = rk ) rf
  real ( kind = rk ) sp
  real ( kind = rk ) value
  real ( kind = rk ) x
  real ( kind = rk ) y
  real ( kind = rk ) z

  cp = cos ( phi )
  sp = sin ( phi )
  x = cp * cp
  y = 1.0D+00 - m * sp ** 2
  z = 1.0D+00
  errtol = 1.0D-03

  value = rf ( x, y, z, errtol, ierr )

  if ( ierr /= 0 ) then
    write ( *, '(a)' ) ''
    write ( *, '(a)' ) 'ELLIPTIC_INC_FM - Fatal error!'
    write ( *, '(a,i2)' ) '  RF returned IERR = ', ierr
    stop 1
  end if

  elliptic_inc_fm = sp * value

  return
end
subroutine elliptic_inc_fm_values ( n_data, phi, m, fm )

!*****************************************************************************80
!
!! ELLIPTIC_INC_FM_VALUES: values of the incomplete elliptic integral F(PHI,M).
!
!  Discussion:
!
!    This is the incomplete elliptic integral of the first kind.
!
!      F(PHI,M) = integral ( 0 <= T <= PHI ) 
!        dT / sqrt ( 1 - M * sin ( T )^2 )
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    22 June 2018
!
!  Author:
!
!    John Burkardt
!
!  Reference:
!
!    Milton Abramowitz, Irene Stegun,
!    Handbook of Mathematical Functions,
!    US Department of Commerce, 1964.
!
!    Stephen Wolfram,
!    The Mathematica Book,
!    Fourth Edition,
!    Wolfram Media / Cambridge University Press, 1999.
!
!  Parameters:
!
!    Input/output, integer N_DATA.  The user sets N_DATA to 0 
!    before the first call.  On each call, the routine increments N_DATA by 1, 
!    and returns the corresponding data when there is no more data, the
!    output value of N_DATA will be 0 again.
!
!    Output, real ( kind = rk ) PHI, M, the arguments.
!
!    Output, real ( kind = rk ) FM, the value of the function.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  integer, parameter :: n_max = 20

  real ( kind = rk ) fm
  real ( kind = rk ) m
  integer n_data
  real ( kind = rk ) phi

  real ( kind = rk ), save, dimension ( n_max ) :: fm_vec = (/ &
        0.4804314075855023D+00, &
         1.535634981092025D+00, &
        0.6602285297476601D+00, &
        0.4125884303785135D+00, &
       0.07964566007155376D+00, &
        0.1062834070535258D+00, &
        0.7733990864393913D+00, &
         1.252862499892228D+00, &
         1.549988686611532D+00, &
        0.1488506735822822D+00, &
         1.892229900799662D+00, &
        0.1936153327753556D+00, &
        0.5481932935424454D+00, &
        0.1911795073571756D+00, &
         1.379225069349756D+00, &
         1.261282453331402D+00, &
         1.535239838525378D+00, &
         1.739782418156071D+00, &
        0.3616930047198503D+00, &
        0.6458627645916422D+00 /)

  real ( kind = rk ), save, dimension ( n_max ) :: m_vec = (/ &
         8.450689756874594D+00, &
        0.6039878267930615D+00, &
        0.1794126658351454D+00, &
        0.7095689301026752D+00, &
         133.9643389059188D+00, &
         47.96621393936416D+00, &
         2.172070586163255D+00, &
      0.002038130569431913D+00, &
        0.3600036705339421D+00, &
        0.6219544540067304D+00, &
        0.8834215943508453D+00, &
        0.2034290670379481D+00, &
         5.772526076430922D+00, &
         11.14853902343298D+00, &
        0.2889238477277305D+00, &
        0.7166617182589116D+00, &
        0.4760623731559658D+00, &
        0.6094948502068943D+00, &
         8.902276887883076D+00, &
        0.5434439226321253D+00 /)

  real ( kind = rk ), save, dimension ( n_max ) :: phi_vec = (/ &
        0.3430906586047127D+00, &
         1.302990057703935D+00, &
        0.6523628380743488D+00, &
        0.4046022501376546D+00, &
       0.06884642871852312D+00, &
        0.0969609046794745D+00, &
         0.630370432896175D+00, &
         1.252375418911598D+00, &
         1.409796082144801D+00, &
        0.1485105463502483D+00, &
         1.349466184634646D+00, &
        0.1933711786970301D+00, &
        0.4088829927466769D+00, &
        0.1785430666405224D+00, &
         1.292588374416351D+00, &
         1.087095515757691D+00, &
         1.352794600489329D+00, &
         1.432530166308616D+00, &
        0.2968093345769761D+00, &
        0.6235880396594726D+00  /)

  if ( n_data < 0 ) then
    n_data = 0
  end if

  n_data = n_data + 1

  if ( n_max < n_data ) then
    n_data = 0
    fm = 0.0D+00
    m = 0.0D+00
    phi = 0.0D+00
  else
    fm = fm_vec(n_data)
    m = m_vec(n_data)
    phi = phi_vec(n_data)
  end if

  return
end
function elliptic_inc_pia ( phi, n, a )

!*****************************************************************************80
!
!! ELLIPTIC_INC_PIA evaluates the incomplete elliptic integral Pi(PHI,N,A).
!
!  Discussion:
!
!    The value is computed using Carlson elliptic integrals:
!
!      Pi(PHI,N,A) = integral ( 0 <= T <= PHI )
!        dT / (1 - N sin^2(T) ) sqrt ( 1 - sin^2(A*pi/180) * sin ( T )^2 )
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    24 June 2018
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input, real ( kind = rk ) PHI, N, A, the arguments.
!
!    Output, real ( kind = rk ) ELLIPTIC_INC_PIA, the function value.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  real ( kind = rk ) a
  real ( kind = rk ) cp
  real ( kind = rk ) elliptic_inc_pia
  real ( kind = rk ) errtol
  integer ierr
  real ( kind = rk ) k
  real ( kind = rk ) n
  real ( kind = rk ) p
  real ( kind = rk ) phi
  real ( kind = rk ), parameter :: r8_pi = 3.141592653589793D+00
  real ( kind = rk ) rf
  real ( kind = rk ) rj
  real ( kind = rk ) sp
  real ( kind = rk ) value
  real ( kind = rk ) value1
  real ( kind = rk ) value2
  real ( kind = rk ) x
  real ( kind = rk ) y
  real ( kind = rk ) z

  k = sin ( a * r8_pi / 180.0D+00 )

  cp = cos ( phi )
  sp = sin ( phi )
  x = cp * cp
  y = ( 1.0D+00 - k * sp ) * ( 1.0D+00 + k * sp )
  z = 1.0D+00
  p = 1.0D+00 - n * sp ** 2
  errtol = 1.0D-03

  value1 = rf ( x, y, z, errtol, ierr )

  if ( ierr /= 0 ) then
    write ( *, '(a)' ) ''
    write ( *, '(a)' ) 'ELLIPTIC_INC_PIA - Fatal error!'
    write ( *, '(a,i2)' ) '  RF returned IERR = ', ierr
    stop 1
  end if

  value2 = rj ( x, y, z, p, errtol, ierr )

  if ( ierr /= 0 ) then
    write ( *, '(a)' ) ''
    write ( *, '(a)' ) 'ELLIPTIC_INC_PIA - Fatal error!'
    write ( *, '(a,i2)' ) '  RJ returned IERR = ', ierr
    stop 1
  end if

  value = sp * value1 + n * sp ** 3 * value2 / 3.0D+00

  elliptic_inc_pia = value

  return
end
subroutine elliptic_inc_pia_values ( n_data, phi, n, a, pia )

!*****************************************************************************80
!
!! ELLIPTIC_INC_PIA_VALUES: values of incomplete elliptic integral Pi(PHI,N,A).
!
!  Discussion:
!
!    This is the incomplete elliptic integral of the third kind.
!
!      Pi(PHI,N,A) = integral ( 0 <= T <= PHI ) 
!        dT / (1 - N sin^2(T) ) sqrt ( 1 - sin^2(A) * sin ( T )^2 )
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    22 June 2018
!
!  Author:
!
!    John Burkardt
!
!  Reference:
!
!    Milton Abramowitz, Irene Stegun,
!    Handbook of Mathematical Functions,
!    US Department of Commerce, 1964.
!
!    Stephen Wolfram,
!    The Mathematica Book,
!    Fourth Edition,
!    Wolfram Media / Cambridge University Press, 1999.
!
!  Parameters:
!
!    Input/output, integer N_DATA.  The user sets N_DATA to 0 
!    before the first call.  On each call, the routine increments N_DATA by 1, 
!    and returns the corresponding data when there is no more data, the
!    output value of N_DATA will be 0 again.
!
!    Output, real ( kind = rk ) PHI, N, A, the arguments of the function.
!
!    Output, real ( kind = rk ) PIA, the value of the function.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  integer, parameter :: n_max = 20

  real ( kind = rk ) a
  real ( kind = rk ) n
  integer n_data
  real ( kind = rk ) phi
  real ( kind = rk ) pia

  real ( kind = rk ), save, dimension ( n_max ) :: a_vec = (/ &
         88.87822485052908D+00, &
        -86.55208740039521D+00, &
        -116.6195703112117D+00, &
        -9.742878017582015D+00, &
         65.73480919446207D+00, &
        -115.0387719677141D+00, &
         124.9421177735846D+00, &
        -89.78704401263703D+00, &
        -98.42673771271734D+00, &
        -53.74936192418378D+00, &
         68.28047574440727D+00, &
         20.82174673810708D+00, &
         -29.1042364797769D+00, &
        -37.80176710944693D+00, &
        -55.81173355852393D+00, &
        -37.66594589748672D+00, &
        -80.09408170610219D+00, &
         52.23806528467412D+00, &
         74.30945212430545D+00, &
        -17.22920703094039D+00 /)

  real ( kind = rk ), save, dimension ( n_max ) :: n_vec = (/ &
         8.064681366127422D+00, &
       -0.2840588974558835D+00, &
        -5.034023488967104D+00, &
        -1.244606253942751D+00, &
         1.465981775919188D+00, &
         95338.12857321106D+00, &
        -44.43130633436311D+00, &
       -0.8029374966926196D+00, &
         5.218883222649502D+00, &
         2.345821782626782D+00, &
         0.157358332363011D+00, &
         1.926593468907062D+00, &
         6.113982855261652D+00, &
         1.805710621498681D+00, &
       -0.4072847419780592D+00, &
       -0.9416404038595624D+00, &
        0.7009655305226739D+00, &
        -1.019830985340273D+00, &
       -0.4510798219577842D+00, &
        0.6028821390092596D+00  /)

  real ( kind = rk ), save, dimension ( n_max ) :: phi_vec = (/ &
        0.3430906586047127D+00, &
        0.8823091382756705D+00, &
        0.4046022501376546D+00, &
        0.9958310121985398D+00, &
         0.630370432896175D+00, &
      0.002887706662908567D+00, &
        0.1485105463502483D+00, &
         1.320800086884777D+00, &
        0.4088829927466769D+00, &
         0.552337007372852D+00, &
         1.087095515757691D+00, &
        0.7128175949111615D+00, &
        0.2968093345769761D+00, &
        0.2910907344062498D+00, &
        0.9695030752034163D+00, &
         1.122288759723523D+00, &
         1.295911610809573D+00, &
         1.116491437736542D+00, &
         1.170719322533712D+00, &
         1.199360682338851D+00 /)

  real ( kind = rk ), save, dimension ( n_max ) :: pia_vec = (/ &
        0.7099335174334724D+00, &
        0.9601963779142505D+00, &
        0.3362852532098376D+00, &
        0.7785343427543768D+00, &
         0.857889755214478D+00, &
      0.004630772344931844D+00, &
        0.1173842687902911D+00, &
         1.505788070660267D+00, &
        0.7213264194624553D+00, &
        0.8073261799642218D+00, &
         1.402853811110838D+00, &
         1.259245331474513D+00, &
        0.3779079263971614D+00, &
        0.3088493910496766D+00, &
        0.9782829177005183D+00, &
        0.9430491574504173D+00, &
         3.320796277384155D+00, &
        0.9730988737054799D+00, &
         1.301988094953789D+00, &
          1.64558360445259D+00 /)

  if ( n_data < 0 ) then
    n_data = 0
  end if

  n_data = n_data + 1

  if ( n_max < n_data ) then
    n_data = 0
    a = 0.0D+00
    n = 0.0D+00
    phi = 0.0D+00
    pia = 0.0D+00
  else
    a = a_vec(n_data)
    n = n_vec(n_data)
    phi = phi_vec(n_data)
    pia = pia_vec(n_data)
  end if

  return
end
function elliptic_inc_pik ( phi, n, k )

!*****************************************************************************80
!
!! ELLIPTIC_INC_PIK evaluates the incomplete elliptic integral Pi(PHI,N,K).
!
!  Discussion:
!
!    The value is computed using Carlson elliptic integrals:
!
!      Pi(PHI,N,K) = integral ( 0 <= T <= PHI )
!        dT / (1 - N sin^2(T) ) sqrt ( 1 - k^2 * sin ( T )^2 )
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    24 June 2018
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input, real ( kind = rk ) PHI, N, K, the arguments.
!
!    Output, real ( kind = rk ) ELLIPTIC_INC_PIK, the function value.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  real ( kind = rk ) cp
  real ( kind = rk ) elliptic_inc_pik
  real ( kind = rk ) errtol
  integer ierr
  real ( kind = rk ) k
  real ( kind = rk ) n
  real ( kind = rk ) p
  real ( kind = rk ) phi
  real ( kind = rk ), parameter :: r8_pi = 3.141592653589793D+00
  real ( kind = rk ) rf
  real ( kind = rk ) rj
  real ( kind = rk ) sp
  real ( kind = rk ) value
  real ( kind = rk ) value1
  real ( kind = rk ) value2
  real ( kind = rk ) x
  real ( kind = rk ) y
  real ( kind = rk ) z

  cp = cos ( phi )
  sp = sin ( phi )
  x = cp * cp
  y = ( 1.0D+00 - k * sp ) * ( 1.0D+00 + k * sp )
  z = 1.0D+00
  p = 1.0D+00 - n * sp ** 2
  errtol = 1.0D-03

  value1 = rf ( x, y, z, errtol, ierr )

  if ( ierr /= 0 ) then
    write ( *, '(a)' ) ''
    write ( *, '(a)' ) 'ELLIPTIC_INC_PIK - Fatal error!'
    write ( *, '(a,i2)' ) '  RF returned IERR = ', ierr
    stop 1
  end if

  value2 = rj ( x, y, z, p, errtol, ierr )

  if ( ierr /= 0 ) then
    write ( *, '(a)' ) ''
    write ( *, '(a)' ) 'ELLIPTIC_INC_PIK - Fatal error!'
    write ( *, '(a,i2)' ) '  RJ returned IERR = ', ierr
    stop 1
  end if

  value = sp * value1 + n * sp ** 3 * value2 / 3.0D+00

  elliptic_inc_pik = value

  return
end
subroutine elliptic_inc_pik_values ( n_data, phi, n, k, pik )

!*****************************************************************************80
!
!! ELLIPTIC_INC_PIK_VALUES: values of incomplete elliptic integral Pi(PHI,N,K).
!
!  Discussion:
!
!    This is the incomplete elliptic integral of the third kind.
!
!      Pi(PHI,N,K) = integral ( 0 <= T <= PHI ) 
!        dT / (1 - N sin^2(T) ) sqrt ( 1 - K^2 * sin ( T )^2 )
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    23 June 2018
!
!  Author:
!
!    John Burkardt
!
!  Reference:
!
!    Milton Abramowitz, Irene Stegun,
!    Handbook of Mathematical Functions,
!    US Department of Commerce, 1964.
!
!    Stephen Wolfram,
!    The Mathematica Book,
!    Fourth Edition,
!    Wolfram Media / Cambridge University Press, 1999.
!
!  Parameters:
!
!    Input/output, integer N_DATA.  The user sets N_DATA to 0 
!    before the first call.  On each call, the routine increments N_DATA by 1, 
!    and returns the corresponding data when there is no more data, the
!    output value of N_DATA will be 0 again.
!
!    Output, real ( kind = rk ) PHI, N, K, the arguments of the function.
!
!    Output, real ( kind = rk ) PIK, the value of the function.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  integer, parameter :: n_max = 20

  real ( kind = rk ) k
  real ( kind = rk ) n
  integer n_data
  real ( kind = rk ) phi
  real ( kind = rk ) pik

  real ( kind = rk ), save, dimension ( n_max ) :: k_vec = (/ &
       1.959036804709882D+00, &
      -1.123741823223131D+00, &
      -2.317629084640271D+00, &
     -0.1202582658444815D+00, &
       1.008702896970963D+00, &
      -103.3677494756118D+00, &
       4.853800240677973D+00, &
      -1.016577251056124D+00, &
       -1.94341484065839D+00, &
     -0.8876593284500023D+00, &
      0.8160487832898813D+00, &
      0.2994546721661018D+00, &
     -0.7044232294525243D+00, &
     -0.9266523277404759D+00, &
     -0.6962608926846425D+00, &
     -0.4453932031991797D+00, &
     -0.9104582513322106D+00, &
      0.6187501419936026D+00, &
      0.8672305032589989D+00, &
     -0.1996772638241632D+00 /)

  real ( kind = rk ), save, dimension ( n_max ) :: n_vec = (/ &
       8.064681366127422D+00, &
     -0.2840588974558835D+00, &
      -5.034023488967104D+00, &
      -1.244606253942751D+00, &
       1.465981775919188D+00, &
       95338.12857321106D+00, &
      -44.43130633436311D+00, &
     -0.8029374966926196D+00, &
       5.218883222649502D+00, &
       2.345821782626782D+00, &
       0.157358332363011D+00, &
       1.926593468907062D+00, &
       6.113982855261652D+00, &
       1.805710621498681D+00, &
     -0.4072847419780592D+00, &
     -0.9416404038595624D+00, &
      0.7009655305226739D+00, &
      -1.019830985340273D+00, &
     -0.4510798219577842D+00, &
      0.6028821390092596D+00  /)

  real ( kind = rk ), save, dimension ( n_max ) :: phi_vec = (/ &
      0.3430906586047127D+00, &
      0.8823091382756705D+00, &
      0.4046022501376546D+00, &
      0.9958310121985398D+00, &
       0.630370432896175D+00, &
    0.002887706662908567D+00, &
      0.1485105463502483D+00, &
       1.320800086884777D+00, &
      0.4088829927466769D+00, &
       0.552337007372852D+00, &
       1.087095515757691D+00, &
      0.7128175949111615D+00, &
      0.2968093345769761D+00, &
      0.2910907344062498D+00, &
      0.9695030752034163D+00, &
       1.122288759723523D+00, &
       1.295911610809573D+00, &
       1.116491437736542D+00, &
       1.170719322533712D+00, &
       1.199360682338851D+00 /)

  real ( kind = rk ), save, dimension ( n_max ) :: pik_vec = (/ &
      0.7982975462595892D+00, &
       1.024022134726036D+00, &
        0.40158120852642D+00, &
      0.7772649487439858D+00, &
      0.8737159913132074D+00, &
    0.004733334297691273D+00, &
      0.1280656893638068D+00, &
       1.594376037512564D+00, &
      0.8521145133671923D+00, &
      0.8154325229803082D+00, &
        1.31594514075427D+00, &
        1.25394623148424D+00, &
      0.3796503567258643D+00, &
      0.3111034454739552D+00, &
      0.9442477901112342D+00, &
      0.9153111661980959D+00, &
       2.842080644328393D+00, &
      0.9263253777034376D+00, &
       1.212396018757624D+00, &
       1.628083572710471D+00 /)

  if ( n_data < 0 ) then
    n_data = 0
  end if

  n_data = n_data + 1

  if ( n_max < n_data ) then
    n_data = 0
    k = 0.0D+00
    n = 0.0D+00
    phi = 0.0D+00
    pik = 0.0D+00
  else
    k = k_vec(n_data)
    n = n_vec(n_data)
    phi = phi_vec(n_data)
    pik = pik_vec(n_data)
  end if

  return
end
function elliptic_inc_pim ( phi, n, m )

!*****************************************************************************80
!
!! ELLIPTIC_INC_PIM evaluates the incomplete elliptic integral Pi(PHI,N,M).
!
!  Discussion:
!
!    The value is computed using Carlson elliptic integrals:
!
!      Pi(PHI,N,M) = integral ( 0 <= T <= PHI )
!        dT / (1 - N sin^2(T) ) sqrt ( 1 - m * sin ( T )^2 )
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    24 June 2018
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input, real ( kind = rk ) PHI, N, M, the arguments.
!
!    Output, real ( kind = rk ) ELLIPTIC_INC_PIM, the function value.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  real ( kind = rk ) cp
  real ( kind = rk ) elliptic_inc_pim
  real ( kind = rk ) errtol
  integer ierr
  real ( kind = rk ) m
  real ( kind = rk ) n
  real ( kind = rk ) p
  real ( kind = rk ) phi
  real ( kind = rk ), parameter :: r8_pi = 3.141592653589793D+00
  real ( kind = rk ) rf
  real ( kind = rk ) rj
  real ( kind = rk ) sp
  real ( kind = rk ) value
  real ( kind = rk ) value1
  real ( kind = rk ) value2
  real ( kind = rk ) x
  real ( kind = rk ) y
  real ( kind = rk ) z

  cp = cos ( phi )
  sp = sin ( phi )
  x = cp * cp
  y = 1.0D+00 - m * sp ** 2
  z = 1.0D+00
  p = 1.0D+00 - n * sp ** 2
  errtol = 1.0D-03

  value1 = rf ( x, y, z, errtol, ierr )

  if ( ierr /= 0 ) then
    write ( *, '(a)' ) ''
    write ( *, '(a)' ) 'ELLIPTIC_INC_PIM - Fatal error!'
    write ( *, '(a,i2)' ) '  RF returned IERR = ', ierr
    stop 1
  end if

  value2 = rj ( x, y, z, p, errtol, ierr )

  if ( ierr /= 0 ) then
    write ( *, '(a)' ) ''
    write ( *, '(a)' ) 'ELLIPTIC_INC_PIM - Fatal error!'
    write ( *, '(a,i2)' ) '  RJ returned IERR = ', ierr
    stop 1
  end if

  value = sp * value1 + n * sp ** 3 * value2 / 3.0D+00

  elliptic_inc_pim = value

  return
end
subroutine elliptic_inc_pim_values ( n_data, phi, n, m, pim )

!*****************************************************************************80
!
!! ELLIPTIC_INC_PIM_VALUES: values of incomplete elliptic integral Pi(PHI,N,M).
!
!  Discussion:
!
!    This is the incomplete elliptic integral of the third kind.
!
!      Pi(PHI,N,M) = integral ( 0 <= T <= PHI ) 
!        dT / (1 - N sin^2(T) ) sqrt ( 1 - M * sin ( T )^2 )
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    24 June 2018
!
!  Author:
!
!    John Burkardt
!
!  Reference:
!
!    Milton Abramowitz, Irene Stegun,
!    Handbook of Mathematical Functions,
!    US Department of Commerce, 1964.
!
!    Stephen Wolfram,
!    The Mathematica Book,
!    Fourth Edition,
!    Wolfram Media / Cambridge University Press, 1999.
!
!  Parameters:
!
!    Input/output, integer N_DATA.  The user sets N_DATA to 0 
!    before the first call.  On each call, the routine increments N_DATA by 1, 
!    and returns the corresponding data when there is no more data, the
!    output value of N_DATA will be 0 again.
!
!    Output, real ( kind = rk ) PHI, N, M, the arguments of the function.
!
!    Output, real ( kind = rk ) PIM, the value of the function.
!
   implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  integer, parameter :: n_max = 20

  real ( kind = rk ) m
  real ( kind = rk ) n
  integer n_data
  real ( kind = rk ) phi
  real ( kind = rk ) pim

  real ( kind = rk ), save, dimension ( n_max ) :: m_vec = (/ &
       7.330122710928245D+00, &
      0.1108806690614566D+00, &
      0.2828355944410993D+00, &
      0.6382999794812498D+00, &
       2.294718938593894D+00, &
       42062.55329826538D+00, &
        39.2394337789563D+00, &
    0.008002151065098688D+00, &
      0.7190579590867517D+00, &
      0.9703767630929055D+00, &
       1.098881295982823D+00, &
       1.398066725917478D+00, &
       4.641021931654496D+00, &
       4.455969064311461D+00, &
      0.3131448239736511D+00, &
      0.3686443684703166D+00, &
     0.06678210908100803D+00, &
      0.9635538974026796D+00, &
       1.060208762696207D+00, &
      0.4687160847955397D+00 /)

  real ( kind = rk ), save, dimension ( n_max ) :: n_vec = (/ &
       8.064681366127422D+00, &
     -0.2840588974558835D+00, &
      -5.034023488967104D+00, &
      -1.244606253942751D+00, &
       1.465981775919188D+00, &
       95338.12857321106D+00, &
      -44.43130633436311D+00, &
     -0.8029374966926196D+00, &
       5.218883222649502D+00, &
       2.345821782626782D+00, &
       0.157358332363011D+00, &
       1.926593468907062D+00, &
       6.113982855261652D+00, &
       1.805710621498681D+00, &
     -0.4072847419780592D+00, &
     -0.9416404038595624D+00, &
      0.7009655305226739D+00, &
      -1.019830985340273D+00, &
     -0.4510798219577842D+00, &
      0.6028821390092596D+00  /)

  real ( kind = rk ), save, dimension ( n_max ) :: phi_vec = (/ &
      0.3430906586047127D+00, &
      0.8823091382756705D+00, &
      0.4046022501376546D+00, &
      0.9958310121985398D+00, &
       0.630370432896175D+00, &
    0.002887706662908567D+00, &
      0.1485105463502483D+00, &
       1.320800086884777D+00, &
      0.4088829927466769D+00, &
       0.552337007372852D+00, &
       1.087095515757691D+00, &
      0.7128175949111615D+00, &
      0.2968093345769761D+00, &
      0.2910907344062498D+00, &
      0.9695030752034163D+00, &
       1.122288759723523D+00, &
       1.295911610809573D+00, &
       1.116491437736542D+00, &
       1.170719322533712D+00, &
       1.199360682338851D+00 /)

  real ( kind = rk ), save, dimension ( n_max ) :: pim_vec = (/ &
         1.0469349800785D+00, &
       0.842114448140669D+00, &
      0.3321642201520043D+00, &
      0.8483033529960849D+00, &
       1.055753817656772D+00, &
    0.005108896144265593D+00, &
      0.1426848042785896D+00, &
       1.031350958206424D+00, &
      0.7131013701418496D+00, &
      0.8268044665355507D+00, &
        1.57632867896015D+00, &
       1.542817120857211D+00, &
      0.4144629799126912D+00, &
      0.3313231611366746D+00, &
      0.9195822851915201D+00, &
      0.9422320754002217D+00, &
       2.036599002815859D+00, &
       1.076799231499882D+00, &
       1.416084462957852D+00, &
       1.824124922310891D+00 /)

  if ( n_data < 0 ) then
    n_data = 0
  end if

  n_data = n_data + 1

  if ( n_max < n_data ) then
    n_data = 0
    m = 0.0D+00
    n = 0.0D+00
    phi = 0.0D+00
    pim = 0.0D+00
  else
    m = m_vec(n_data)
    n = n_vec(n_data)
    phi = phi_vec(n_data)
    pim = pim_vec(n_data)
  end if

  return
end
function elliptic_pia ( n, a )

!*****************************************************************************80
!
!! ELLIPTIC_PIA evaluates the complete elliptic integral Pi(N,A).
!
!  Discussion:
!
!    This is one form of what is sometimes called the complete elliptic
!    integral of the third kind.
!
!    The function is defined by the formula:
!
!      Pi(N,A) = integral ( 0 <= T <= PI/2 )
!        dT / (1 - N sin^2(T) ) sqrt ( 1 - sin^2(A) * sin ( T )^2 )
!
!    In MATLAB, the function can be evaluated by:
!
!      ellipticPi(n,(sin(a*pi/180)^2)
!
!    The value is computed using Carlson elliptic integrals:
!
!      k = sin ( a * pi / 180 )
!      Pi(n,k) = RF ( 0, 1 - k^2, 1 ) + 1/3 n RJ ( 0, 1 - k^2, 1, 1 - n )
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    30 May 2018
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input, real ( kind = rk ) N, A, the arguments.
!
!    Output, real ( kind = rk ) ELLIPTIC_PIA, the function value.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  real ( kind = rk ) a
  real ( kind = rk ) elliptic_pia
  real ( kind = rk ) errtol
  integer ierr
  real ( kind = rk ) k
  real ( kind = rk ) n
  real ( kind = rk ) p
  real ( kind = rk ), parameter :: r8_pi = 3.141592653589793D+00
  real ( kind = rk ) rf
  real ( kind = rk ) rj
  real ( kind = rk ) value
  real ( kind = rk ) x
  real ( kind = rk ) y
  real ( kind = rk ) z

  k = sin ( a * r8_pi / 180.0D+00 )
  x = 0.0D+00
  y = ( 1.0D+00 - k ) * ( 1.0D+00 + k )
  z = 1.0D+00
  p = 1.0D+00 - n
  errtol = 1.0D-03

  value = rf ( x, y, z, errtol, ierr ) &
    + n * rj ( x, y, z, p, errtol, ierr ) / 3.0D+00

  elliptic_pia = value

  return
end
subroutine elliptic_pia_values ( n_data, n, a, pia )

!*****************************************************************************80
!
!! ELLIPTIC_PIA_VALUES returns values of the complete elliptic integral Pi(N,A).
!
!  Discussion:
!
!    This is one form of what is sometimes called the complete elliptic
!    integral of the third kind.
!
!    The function is defined by the formula:
!
!      Pi(N,A) = integral ( 0 <= T <= PI/2 )
!        dT / (1 - N sin^2(T) ) sqrt ( 1 - sin^2(A) * sin ( T )^2 )
!
!    In MATLAB, the function can be evaluated by:
!
!      ellipticPi(n,(sin(A*pi/180))^2)
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    30 May 2018
!
!  Author:
!
!    John Burkardt
!
!  Reference:
!
!    Milton Abramowitz, Irene Stegun,
!    Handbook of Mathematical Functions,
!    National Bureau of Standards, 1964,
!    ISBN: 0-486-61272-4,
!    LC: QA47.A34.
!
!    Stephen Wolfram,
!    The Mathematica Book,
!    Fourth Edition,
!    Cambridge University Press, 1999,
!    ISBN: 0-521-64314-7,
!    LC: QA76.95.W65.
!
!  Parameters:
!
!    Input/output, integer N_DATA.  The user sets N_DATA to 0
!    before the first call.  On each call, the routine increments N_DATA by 1,
!    and returns the corresponding data; when there is no more data, the
!    output value of N_DATA will be 0 again.
!
!    Output, real ( kind = rk ) N, A, the arguments of the function.
!
!    Output, real ( kind = rk ) PIA, the value of the function.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  integer, parameter :: n_max = 20

  real ( kind = rk ) a
  real ( kind = rk ), save, dimension ( n_max ) :: a_vec = (/ &
    30.00000000000000D+00, &
    45.00000000000000D+00, &
    60.00000000000000D+00, &
    77.07903361841643D+00, &
    30.00000000000000D+00, &
    45.00000000000000D+00, &
    60.00000000000000D+00, &
    77.07903361841643D+00, &
    30.00000000000000D+00, &
    45.00000000000000D+00, &
    60.00000000000000D+00, &
    77.07903361841643D+00, &
    30.00000000000000D+00, &
    45.00000000000000D+00, &
    60.00000000000000D+00, &
    77.07903361841643D+00, &
    30.00000000000000D+00, &
    45.00000000000000D+00, &
    60.00000000000000D+00, &
    77.07903361841643D+00 /)
  real ( kind = rk ) n
  integer n_data
  real ( kind = rk ), save, dimension ( n_max ) :: n_vec = (/ &
    -10.0D+00, &
    -10.0D+00, &
    -10.0D+00, &
    -10.0D+00, &
     -3.0D+00, &
     -3.0D+00, &
     -3.0D+00, &
     -3.0D+00, &
     -1.0D+00, &
     -1.0D+00, &
     -1.0D+00, &
     -1.0D+00, &
      0.0D+00, &
      0.0D+00, &
      0.0D+00, &
      0.0D+00, &
      0.5D+00, &
      0.5D+00, &
      0.5D+00, &
      0.5D+00 /)
  real ( kind = rk ) pia
  real ( kind = rk ), save, dimension ( n_max ) :: pia_vec = (/ &
    0.4892245275965397D+00, &
    0.5106765677902629D+00, &
    0.5460409271920561D+00, &
    0.6237325893535237D+00, &
    0.823045542660675D+00,  &
    0.8760028274011437D+00, &
    0.9660073560143946D+00, &
    1.171952391481798D+00, &
    1.177446843000566D+00, &
    1.273127366749682D+00, &
    1.440034318657551D+00, &
    1.836472172302591D+00, &
    1.685750354812596D+00, &
    1.854074677301372D+00, &
    2.156515647499643D+00, &
    2.908337248444552D+00, &
    2.413671504201195D+00, &
    2.701287762095351D+00, &
    3.234773471249465D+00, &
    4.633308147279891D+00 /)

  if ( n_data < 0 ) then
    n_data = 0
  end if

  n_data = n_data + 1

  if ( n_max < n_data ) then
    n_data = 0
    a = 0.0D+00
    n = 0.0D+00
    pia = 0.0D+00
  else
    a = a_vec(n_data)
    n = n_vec(n_data)
    pia = pia_vec(n_data)
  end if

  return
end
function elliptic_pik ( n, k )

!*****************************************************************************80
!
!! ELLIPTIC_PIK evaluates the complete elliptic integral Pi(N,K).
!
!  Discussion:
!
!    This is one form of what is sometimes called the complete elliptic
!    integral of the third kind.
!
!    The function is defined by the formula:
!
!      Pi(N,K) = integral ( 0 <= T <= PI/2 )
!        dT / (1 - N sin^2(T) ) sqrt ( 1 - K^2 * sin ( T )^2 )
!
!    In MATLAB, the function can be evaluated by:
!
!      ellipticPi(n,k^2)
!
!    The value is computed using Carlson elliptic integrals:
!
!      Pi(n,k) = RF ( 0, 1 - k^2, 1 ) + 1/3 n RJ ( 0, 1 - k^2, 1, 1 - n )
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    30 May 2018
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input, real ( kind = rk ) N, K, the arguments.
!
!    Output, real ( kind = rk ) ELLIPTIC_PIK, the function value.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  real ( kind = rk ) elliptic_pik
  real ( kind = rk ) errtol
  integer ierr
  real ( kind = rk ) k
  real ( kind = rk ) n
  real ( kind = rk ) p
  real ( kind = rk ) rf
  real ( kind = rk ) rj
  real ( kind = rk ) value
  real ( kind = rk ) x
  real ( kind = rk ) y
  real ( kind = rk ) z

  x = 0.0D+00
  y = ( 1.0D+00 - k ) * ( 1.0D+00 + k )
  z = 1.0D+00
  p = 1.0D+00 - n
  errtol = 1.0D-03

  value = rf ( x, y, z, errtol, ierr ) &
    + n * rj ( x, y, z, p, errtol, ierr ) / 3.0D+00

  elliptic_pik = value

  return
end
subroutine elliptic_pik_values ( n_data, n, k, pik )

!*****************************************************************************80
!
!! ELLIPTIC_PIK_VALUES returns values of the complete elliptic integral Pi(N,K).
!
!  Discussion:
!
!    This is one form of what is sometimes called the complete elliptic
!    integral of the third kind.
!
!    The function is defined by the formula:
!
!      Pi(N,K) = integral ( 0 <= T <= PI/2 )
!        dT / (1 - N sin^2(T) ) sqrt ( 1 - K^2 * sin ( T )^2 )
!
!    In MATLAB, the function can be evaluated by:
!
!      ellipticPi(n,k^2)
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    30 May 2018
!
!  Author:
!
!    John Burkardt
!
!  Reference:
!
!    Milton Abramowitz, Irene Stegun,
!    Handbook of Mathematical Functions,
!    National Bureau of Standards, 1964,
!    ISBN: 0-486-61272-4,
!    LC: QA47.A34.
!
!    Stephen Wolfram,
!    The Mathematica Book,
!    Fourth Edition,
!    Cambridge University Press, 1999,
!    ISBN: 0-521-64314-7,
!    LC: QA76.95.W65.
!
!  Parameters:
!
!    Input/output, integer N_DATA.  The user sets N_DATA to 0
!    before the first call.  On each call, the routine increments N_DATA by 1,
!    and returns the corresponding data; when there is no more data, the
!    output value of N_DATA will be 0 again.
!
!    Output, real ( kind = rk ) N, K, the arguments of the function.
!
!    Output, real ( kind = rk ) PIK, the value of the function.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  integer, parameter :: n_max = 20

  real ( kind = rk ) k
  real ( kind = rk ), save, dimension ( n_max ) :: k_vec = (/ &
    0.5000000000000000D+00, &
    0.7071067811865476D+00, &
    0.8660254037844386D+00, &
    0.9746794344808963D+00, &
    0.5000000000000000D+00, &
    0.7071067811865476D+00, &
    0.8660254037844386D+00, &
    0.9746794344808963D+00, &
    0.5000000000000000D+00, &
    0.7071067811865476D+00, &
    0.8660254037844386D+00, &
    0.9746794344808963D+00, &
    0.5000000000000000D+00, &
    0.7071067811865476D+00, &
    0.8660254037844386D+00, &
    0.9746794344808963D+00, &
    0.5000000000000000D+00, &
    0.7071067811865476D+00, &
    0.8660254037844386D+00, &
    0.9746794344808963D+00  /)
  real ( kind = rk ) n
  integer n_data
  real ( kind = rk ), save, dimension ( n_max ) :: n_vec = (/ &
    -10.0D+00, &
    -10.0D+00, &
    -10.0D+00, &
    -10.0D+00, &
     -3.0D+00, &
     -3.0D+00, &
     -3.0D+00, &
     -3.0D+00, &
     -1.0D+00, &
     -1.0D+00, &
     -1.0D+00, &
     -1.0D+00, &
      0.0D+00, &
      0.0D+00, &
      0.0D+00, &
      0.0D+00, &
      0.5D+00, &
      0.5D+00, &
      0.5D+00, &
      0.5D+00 /)
  real ( kind = rk ) pik
  real ( kind = rk ), save, dimension ( n_max ) :: pik_vec = (/ &
    0.4892245275965397D+00, &
    0.5106765677902629D+00, &
    0.5460409271920561D+00, &
    0.6237325893535237D+00, &
    0.823045542660675D+00, &
    0.8760028274011437D+00, &
    0.9660073560143946D+00, &
    1.171952391481798D+00, &
    1.177446843000566D+00, &
    1.273127366749682D+00, &
    1.440034318657551D+00, &
    1.836472172302591D+00, &
    1.685750354812596D+00, &
    1.854074677301372D+00, &
    2.156515647499643D+00, &
    2.908337248444552D+00, &
    2.413671504201195D+00, &
    2.701287762095351D+00, &
    3.234773471249465D+00, &
    4.633308147279891D+00 /)

  if ( n_data < 0 ) then
    n_data = 0
  end if

  n_data = n_data + 1

  if ( n_max < n_data ) then
    n_data = 0
    k = 0.0D+00
    n = 0.0D+00
    pik = 0.0D+00
  else
    k = k_vec(n_data)
    n = n_vec(n_data)
    pik = pik_vec(n_data)
  end if

  return
end
function elliptic_pim ( n, m )

!*****************************************************************************80
!
!! ELLIPTIC_PIM evaluates the complete elliptic integral Pi(N,M).
!
!  Discussion:
!
!    This is one form of what is sometimes called the complete elliptic
!    integral of the third kind.
!
!    The function is defined by the formula:
!
!      Pi(N,M) = integral ( 0 <= T <= PI/2 )
!        dT / (1 - N sin^2(T) ) sqrt ( 1 - M * sin ( T )^2 )
!
!    In MATLAB, the function can be evaluated by:
!
!      ellipticPi(n,m)
!
!    The value is computed using Carlson elliptic integrals:
!
!      Pi(n,m) = RF ( 0, 1 - m, 1 ) + 1/3 n RJ ( 0, 1 - m, 1, 1 - n )
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    30 May 2018
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input, real ( kind = rk ) N, M, the arguments.
!
!    Output, real ( kind = rk ) ELLIPTIC_PIM, the function value.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  real ( kind = rk ) elliptic_pim
  real ( kind = rk ) errtol
  integer ierr
  real ( kind = rk ) m
  real ( kind = rk ) n
  real ( kind = rk ) p
  real ( kind = rk ) rf
  real ( kind = rk ) rj
  real ( kind = rk ) value
  real ( kind = rk ) x
  real ( kind = rk ) y
  real ( kind = rk ) z

  x = 0.0D+00
  y = 1.0D+00 - m
  z = 1.0D+00
  p = 1.0D+00 - n
  errtol = 1.0D-03

  value = rf ( x, y, z, errtol, ierr ) &
    + n * rj ( x, y, z, p, errtol, ierr ) / 3.0D+00

  elliptic_pim = value

  return
end
subroutine elliptic_pim_values ( n_data, n, m, pim )

!*****************************************************************************80
!
!! ELLIPTIC_PIM_VALUES returns values of the complete elliptic integral Pi(N,M).
!
!  Discussion:
!
!    This is one form of what is sometimes called the complete elliptic
!    integral of the third kind.
!
!    The function is defined by the formula:
!
!      Pi(N,M) = integral ( 0 <= T <= PI/2 )
!        dT / (1 - N sin^2(T) ) sqrt ( 1 - M * sin ( T )^2 )
!
!    In MATLAB, the function can be evaluated by:
!
!      ellipticPi(n,m)
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    30 May 2018
!
!  Author:
!
!    John Burkardt
!
!  Reference:
!
!    Milton Abramowitz, Irene Stegun,
!    Handbook of Mathematical Functions,
!    National Bureau of Standards, 1964,
!    ISBN: 0-486-61272-4,
!    LC: QA47.A34.
!
!    Stephen Wolfram,
!    The Mathematica Book,
!    Fourth Edition,
!    Cambridge University Press, 1999,
!    ISBN: 0-521-64314-7,
!    LC: QA76.95.W65.
!
!  Parameters:
!
!    Input/output, integer N_DATA.  The user sets N_DATA to 0
!    before the first call.  On each call, the routine increments N_DATA by 1,
!    and returns the corresponding data; when there is no more data, the
!    output value of N_DATA will be 0 again.
!
!    Output, real ( kind = rk ) N, M, the arguments of the function.
!
!    Output, real ( kind = rk ) PIM, the value of the function.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  integer, parameter :: n_max = 20

  real ( kind = rk ) m
  real ( kind = rk ), save, dimension ( n_max ) :: m_vec = (/ &
    0.25D+00, &
    0.50D+00, &
    0.75D+00, &
    0.95D+00, &
    0.25D+00, &
    0.50D+00, &
    0.75D+00, &
    0.95D+00, &
    0.25D+00, &
    0.50D+00, &
    0.75D+00, &
    0.95D+00, &
    0.25D+00, &
    0.50D+00, &
    0.75D+00, &
    0.95D+00, &
    0.25D+00, &
    0.50D+00, &
    0.75D+00, &
    0.95D+00  /)
  real ( kind = rk ) n
  integer n_data
  real ( kind = rk ), save, dimension ( n_max ) :: n_vec = (/ &
    -10.0D+00, &
    -10.0D+00, &
    -10.0D+00, &
    -10.0D+00, &
     -3.0D+00, &
     -3.0D+00, &
     -3.0D+00, &
     -3.0D+00, &
     -1.0D+00, &
     -1.0D+00, &
     -1.0D+00, &
     -1.0D+00, &
      0.0D+00, &
      0.0D+00, &
      0.0D+00, &
      0.0D+00, &
      0.5D+00, &
      0.5D+00, &
      0.5D+00, &
      0.5D+00 /)
  real ( kind = rk ) pim
  real ( kind = rk ), save, dimension ( n_max ) :: pim_vec = (/ &
    0.4892245275965397D+00, &
    0.5106765677902629D+00, &
    0.5460409271920561D+00, &
    0.6237325893535237D+00, &
    0.823045542660675D+00, &
    0.8760028274011437D+00, &
    0.9660073560143946D+00, &
    1.171952391481798D+00, &
    1.177446843000566D+00, &
    1.273127366749682D+00, &
    1.440034318657551D+00, &
    1.836472172302591D+00, &
    1.685750354812596D+00, &
    1.854074677301372D+00, &
    2.156515647499643D+00, &
    2.908337248444552D+00, &
    2.413671504201195D+00, &
    2.701287762095351D+00, &
    3.234773471249465D+00, &
    4.633308147279891D+00 /)

  if ( n_data < 0 ) then
    n_data = 0
  end if

  n_data = n_data + 1

  if ( n_max < n_data ) then
    n_data = 0
    m = 0.0D+00
    n = 0.0D+00
    pim = 0.0D+00
  else
    m = m_vec(n_data)
    n = n_vec(n_data)
    pim = pim_vec(n_data)
  end if

  return
end
function jacobi_cn ( u, m )

!*****************************************************************************80
!
!! JACOBI_CN evaluates the Jacobi elliptic function CN(U,M).
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    25 June 2018
!
!  Author:
!
!    Original ALGOL version by Roland Bulirsch.
!    FORTRAN90 version by John Burkardt
!
!  Reference:
!
!    Roland Bulirsch,
!    Numerical calculation of elliptic integrals and elliptic functions,
!    Numerische Mathematik,
!    Volume 7, Number 1, 1965, pages 78-90.
!
!  Parameters:
!
!    Input, real ( kind = rk ) U, M, the arguments.
!
!    Output, real ( kind = rk ) JACOBI_CN, the function value.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  real ( kind = rk ) cn
  real ( kind = rk ) dn
  real ( kind = rk ) jacobi_cn
  real ( kind = rk ) m
  real ( kind = rk ) sn
  real ( kind = rk ) u

  call sncndn ( u, m, sn, cn, dn )

  jacobi_cn = cn

  return
end
subroutine jacobi_cn_values ( n_data, u, a, k, m, cn )

!*****************************************************************************80
!
!! jacobi_cn_values returns some values of the Jacobi elliptic function CN(U,M).
!
!  Discussion:
!
!    In Mathematica, the function can be evaluated by:
!
!      JacobiCN[ u, m ]
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    19 November 2020
!
!  Author:
!
!    John Burkardt
!
!  Reference:
!
!    Milton Abramowitz, Irene Stegun,
!    Handbook of Mathematical Functions,
!    National Bureau of Standards, 1964,
!    ISBN: 0-486-61272-4,
!    LC: QA47.A34.
!
!    Stephen Wolfram,
!    The Mathematica Book,
!    Fourth Edition,
!    Cambridge University Press, 1999,
!    ISBN: 0-521-64314-7,
!    LC: QA76.95.W65.
!
!  Input:
!
!    integer N_DATA.  The user sets N_DATA to 0 before the first call.  
!
!  Output:
!
!    integer N_DATA.  On each call, the routine increments N_DATA by 1,
!    and returns the corresponding data; when there is no more data, the
!    output value of N_DATA will be 0 again.
!
!    real ( kind = rk ) U, the argument of the function.
!
!    real ( kind = rk ) A, K, M, the parameters of the function.
!
!    real ( kind = rk ) CN, the value of the function.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  integer, parameter :: n_max = 20

  real ( kind = rk ) a
  real ( kind = rk ) k
  real ( kind = rk ) m
  real ( kind = rk ), save, dimension ( n_max ) :: m_vec = (/ &
    0.0D+00, &
    0.0D+00, &
    0.0D+00, &
    0.0D+00, &
    0.0D+00, &
    0.5D+00, &
    0.5D+00, &
    0.5D+00, &
    0.5D+00, &
    0.5D+00, &
    1.0D+00, &
    1.0D+00, &
    1.0D+00, &
    1.0D+00, &
    1.0D+00, &
    1.0D+00, &
    1.0D+00, &
    1.0D+00, &
    1.0D+00, &
    1.0D+00 /)
  real ( kind = rk ) cn
  real ( kind = rk ), save, dimension ( n_max ) :: cn_vec = (/ &
     0.9950041652780258D+00, &
     0.9800665778412416D+00, &
     0.8775825618903727D+00, &
     0.5403023058681397D+00, &
    -0.4161468365471424D+00, &
     0.9950124626090582D+00, &
     0.9801976276784098D+00, &
     0.8822663948904403D+00, &
     0.5959765676721407D+00, &
    -0.1031836155277618D+00, &
     0.9950207489532265D+00, &
     0.9803279976447253D+00, &
     0.8868188839700739D+00, &
     0.6480542736638854D+00, &
     0.2658022288340797D+00, &
     0.3661899347368653D-01, &
     0.9803279976447253D+00, &
     0.8868188839700739D+00, &
     0.6480542736638854D+00, &
     0.2658022288340797D+00 /)
  integer n_data
  real ( kind = rk ) u
  real ( kind = rk ), save, dimension ( n_max ) :: u_vec = (/ &
     0.1D+00, &
     0.2D+00, &
     0.5D+00, &
     1.0D+00, &
     2.0D+00, &
     0.1D+00, &
     0.2D+00, &
     0.5D+00, &
     1.0D+00, &
     2.0D+00, &
     0.1D+00, &
     0.2D+00, &
     0.5D+00, &
     1.0D+00, &
     2.0D+00, &
     4.0D+00, &
    -0.2D+00, &
    -0.5D+00, &
    -1.0D+00, &
    -2.0D+00 /)

  if ( n_data < 0 ) then
    n_data = 0
  end if

  n_data = n_data + 1

  if ( n_max < n_data ) then
    n_data = 0
    a = 0.0D+00
    k = 0.0D+00
    m = 0.0D+00
    u = 0.0D+00
    cn = 0.0D+00
  else
    m = m_vec(n_data)
    k = sqrt ( m )
    a = asin ( k )
    u = u_vec(n_data)
    cn = cn_vec(n_data)
  end if

  return
end
function jacobi_dn ( u, m )

!*****************************************************************************80
!
!! JACOBI_DN evaluates the Jacobi elliptic function DN(U,M).
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    25 June 2018
!
!  Author:
!
!    Original ALGOL version by Roland Bulirsch.
!    FORTRAN90 version by John Burkardt
!
!  Reference:
!
!    Roland Bulirsch,
!    Numerical calculation of elliptic integrals and elliptic functions,
!    Numerische Mathematik,
!    Volume 7, Number 1, 1965, pages 78-90.
!
!  Parameters:
!
!    Input, real ( kind = rk ) U, M, the arguments.
!
!    Output, real ( kind = rk ) JACOBI_DN, the function value.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  real ( kind = rk ) cn
  real ( kind = rk ) dn
  real ( kind = rk ) jacobi_dn
  real ( kind = rk ) m
  real ( kind = rk ) sn
  real ( kind = rk ) u

  call sncndn ( u, m, sn, cn, dn )

  jacobi_dn = dn

  return
end
subroutine jacobi_dn_values ( n_data, u, a, k, m, dn )

!*****************************************************************************80
!
!! jacobi_dn_values returns some values of the Jacobi elliptic function DN(U,M).
!
!  Discussion:
!
!    In Mathematica, the function can be evaluated by:
!
!      JacobiDN[ u, m ]
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    19 November 2020
!
!  Author:
!
!    John Burkardt
!
!  Reference:
!
!    Milton Abramowitz, Irene Stegun,
!    Handbook of Mathematical Functions,
!    National Bureau of Standards, 1964,
!    ISBN: 0-486-61272-4,
!    LC: QA47.A34.
!
!    Stephen Wolfram,
!    The Mathematica Book,
!    Fourth Edition,
!    Cambridge University Press, 1999,
!    ISBN: 0-521-64314-7,
!    LC: QA76.95.W65.
!
!  Input:
!
!    integer N_DATA.  The user sets N_DATA to 0 before the first call.  
!
!  Output:
!
!    integer N_DATA.  On each call, the routine increments N_DATA by 1,
!    and returns the corresponding data; when there is no more data, the
!    output value of N_DATA will be 0 again.
!
!    real ( kind = rk ) U, the argument of the function.
!
!    real ( kind = rk ) A, K, M, the parameters of the function.
!
!    real ( kind = rk ) CN, the value of the function.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  integer, parameter :: n_max = 20

  real ( kind = rk ) a
  real ( kind = rk ) k
  real ( kind = rk ) m
  real ( kind = rk ), save, dimension ( n_max ) :: m_vec = (/ &
    0.0D+00, &
    0.0D+00, &
    0.0D+00, &
    0.0D+00, &
    0.0D+00, &
    0.5D+00, &
    0.5D+00, &
    0.5D+00, &
    0.5D+00, &
    0.5D+00, &
    1.0D+00, &
    1.0D+00, &
    1.0D+00, &
    1.0D+00, &
    1.0D+00, &
    1.0D+00, &
    1.0D+00, &
    1.0D+00, &
    1.0D+00, &
    1.0D+00 /)
  real ( kind = rk ) dn
  real ( kind = rk ), save, dimension ( n_max ) :: dn_vec = (/ &
    0.1000000000000000D+01, &
    0.1000000000000000D+01, &
    0.1000000000000000D+01, &
    0.1000000000000000D+01, &
    0.1000000000000000D+01, &
    0.9975093485144243D+00, &
    0.9901483195224800D+00, &
    0.9429724257773857D+00, &
    0.8231610016315963D+00, &
    0.7108610477840873D+00, &
    0.9950207489532265D+00, &
    0.9803279976447253D+00, &
    0.8868188839700739D+00, &
    0.6480542736638854D+00, &
    0.2658022288340797D+00, &
    0.3661899347368653D-01, &
    0.9803279976447253D+00, &
    0.8868188839700739D+00, &
    0.6480542736638854D+00, &
    0.2658022288340797D+00 /)
  integer n_data
  real ( kind = rk ) u
  real ( kind = rk ), save, dimension ( n_max ) :: u_vec = (/ &
     0.1D+00, &
     0.2D+00, &
     0.5D+00, &
     1.0D+00, &
     2.0D+00, &
     0.1D+00, &
     0.2D+00, &
     0.5D+00, &
     1.0D+00, &
     2.0D+00, &
     0.1D+00, &
     0.2D+00, &
     0.5D+00, &
     1.0D+00, &
     2.0D+00, &
     4.0D+00, &
    -0.2D+00, &
    -0.5D+00, &
    -1.0D+00, &
    -2.0D+00 /)

  if ( n_data < 0 ) then
    n_data = 0
  end if

  n_data = n_data + 1

  if ( n_max < n_data ) then
    n_data = 0
    a = 0.0D+00
    k = 0.0D+00
    m = 0.0D+00
    u = 0.0D+00
    dn = 0.0D+00
  else
    m = m_vec(n_data)
    k = sqrt ( m )
    a = asin ( k )
    u = u_vec(n_data)
    dn = dn_vec(n_data)
  end if

  return
end
function jacobi_sn ( u, m )

!*****************************************************************************80
!
!! JACOBI_SN evaluates the Jacobi elliptic function SN(U,M).
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    25 June 2018
!
!  Author:
!
!    Original ALGOL version by Roland Bulirsch.
!    FORTRAN90 version by John Burkardt
!
!  Reference:
!
!    Roland Bulirsch,
!    Numerical calculation of elliptic integrals and elliptic functions,
!    Numerische Mathematik,
!    Volume 7, Number 1, 1965, pages 78-90.
!
!  Parameters:
!
!    Input, real ( kind = rk ) U, M, the arguments.
!
!    Output, real ( kind = rk ) JACOBI_SN, the function value.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  real ( kind = rk ) cn
  real ( kind = rk ) dn
  real ( kind = rk ) jacobi_sn
  real ( kind = rk ) m
  real ( kind = rk ) sn
  real ( kind = rk ) u

  call sncndn ( u, m, sn, cn, dn )

  jacobi_sn = sn

  return
end
subroutine jacobi_sn_values ( n_data, u, a, k, m, sn )

!*****************************************************************************80
!
!! jacobi_sn_values returns some values of the Jacobi elliptic function SN(U,M).
!
!  Discussion:
!
!    In Mathematica, the function can be evaluated by:
!
!      JacobiSN[ u, m ]
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    19 November 2020
!
!  Author:
!
!    John Burkardt
!
!  Reference:
!
!    Milton Abramowitz, Irene Stegun,
!    Handbook of Mathematical Functions,
!    National Bureau of Standards, 1964,
!    ISBN: 0-486-61272-4,
!    LC: QA47.A34.
!
!    Stephen Wolfram,
!    The Mathematica Book,
!    Fourth Edition,
!    Cambridge University Press, 1999,
!    ISBN: 0-521-64314-7,
!    LC: QA76.95.W65.
!
!  Input:
!
!    integer N_DATA.  The user sets N_DATA to 0 before the first call.  
!
!  Output:
!
!    integer N_DATA.  On each call, the routine increments N_DATA by 1,
!    and returns the corresponding data; when there is no more data, the
!    output value of N_DATA will be 0 again.
!
!    real ( kind = rk ) U, the argument of the function.
!
!    real ( kind = rk ) A, K, M, the parameters of the function.
!
!    real ( kind = rk ) CN, the value of the function.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  integer, parameter :: n_max = 20

  real ( kind = rk ) a
  real ( kind = rk ) k
  real ( kind = rk ) m
  real ( kind = rk ), save, dimension ( n_max ) :: m_vec = (/ &
    0.0D+00, &
    0.0D+00, &
    0.0D+00, &
    0.0D+00, &
    0.0D+00, &
    0.5D+00, &
    0.5D+00, &
    0.5D+00, &
    0.5D+00, &
    0.5D+00, &
    1.0D+00, &
    1.0D+00, &
    1.0D+00, &
    1.0D+00, &
    1.0D+00, &
    1.0D+00, &
    1.0D+00, &
    1.0D+00, &
    1.0D+00, &
    1.0D+00 /)
  real ( kind = rk ) sn
  real ( kind = rk ), save, dimension ( n_max ) :: sn_vec = (/ &
     0.9983341664682815D-01, &
     0.1986693307950612D+00, &
     0.4794255386042030D+00, &
     0.8414709848078965D+00, &
     0.9092974268256817D+00, &
     0.9975068547462484D-01, &
     0.1980217429819704D+00, &
     0.4707504736556573D+00, &
     0.8030018248956439D+00, &
     0.9946623253580177D+00, &
     0.9966799462495582D-01, &
     0.1973753202249040D+00, &
     0.4621171572600098D+00, &
     0.7615941559557649D+00, &
     0.9640275800758169D+00, &
     0.9993292997390670D+00, &
    -0.1973753202249040D+00, &
    -0.4621171572600098D+00, &
    -0.7615941559557649D+00, &
    -0.9640275800758169D+00 /)
  integer n_data
  real ( kind = rk ) u
  real ( kind = rk ), save, dimension ( n_max ) :: u_vec = (/ &
     0.1D+00, &
     0.2D+00, &
     0.5D+00, &
     1.0D+00, &
     2.0D+00, &
     0.1D+00, &
     0.2D+00, &
     0.5D+00, &
     1.0D+00, &
     2.0D+00, &
     0.1D+00, &
     0.2D+00, &
     0.5D+00, &
     1.0D+00, &
     2.0D+00, &
     4.0D+00, &
    -0.2D+00, &
    -0.5D+00, &
    -1.0D+00, &
    -2.0D+00 /)

  if ( n_data < 0 ) then
    n_data = 0
  end if

  n_data = n_data + 1

  if ( n_max < n_data ) then
    n_data = 0
    a = 0.0D+00
    k = 0.0D+00
    m = 0.0D+00
    u = 0.0D+00
    sn = 0.0D+00
  else
    m = m_vec(n_data)
    k = sqrt ( m )
    a = asin ( k )
    u = u_vec(n_data)
    sn = sn_vec(n_data)
  end if

  return
end
function rc ( x, y, errtol, ierr )

!*****************************************************************************80
!
!! RC computes the elementary integral RC(X,Y).
!
!  Discussion:
!
!    This function computes the elementary integral
!
!      RC(X,Y) = Integral ( 0 <= T < oo )
!
!                  -1/2     -1
!        (1/2)(T+X)    (T+Y)  DT,
!
!    where X is nonnegative and Y is positive.  The duplication
!    theorem is iterated until the variables are nearly equal,
!    and the function is then expanded in Taylor series to fifth
!    order.
!
!    Logarithmic, inverse circular, and inverse hyperbolic
!    functions can be expressed in terms of RC.
!
!    Check by addition theorem:
!
!      RC(X,X+Z) + RC(Y,Y+Z) = RC(0,Z),
!      where X, Y, and Z are positive and X * Y = Z * Z.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    30 May 2018
!
!  Author:
!
!    Original FORTRAN77 version by Bille Carlson, Elaine Notis.
!    This FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Bille Carlson,
!    Computing Elliptic Integrals by Duplication,
!    Numerische Mathematik,
!    Volume 33, 1979, pages 1-16.
!
!    Bille Carlson, Elaine Notis,
!    Algorithm 577, Algorithms for Incomplete Elliptic Integrals,
!    ACM Transactions on Mathematical Software,
!    Volume 7, Number 3, pages 398-403, September 1981.
!
!  Parameters:
!
!    Input, real ( kind = rk ) X, Y, the arguments in the integral.
!
!    Input, real ( kind = rk ) ERRTOL, the error tolerance.
!    Relative error due to truncation is less than
!      16 * ERRTOL ^ 6 / (1 - 2 * ERRTOL).
!    Sample choices:
!      ERRTOL   Relative truncation error less than
!      1.D-3    2.D-17
!      3.D-3    2.D-14
!      1.D-2    2.D-11
!      3.D-2    2.D-8
!      1.D-1    2.D-5
!
!    Output, integer IERR, the error flag.
!    0, no error occurred.
!    1, abnormal termination.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  real ( kind = rk ) c1
  real ( kind = rk ) c2
  real ( kind = rk ) errtol
  integer ierr
  real ( kind = rk ) lamda
  real ( kind = rk ) lolim
  real ( kind = rk ) mu
  real ( kind = rk ) rc
  real ( kind = rk ) s
  real ( kind = rk ) sn
  real ( kind = rk ) uplim
  real ( kind = rk ) x
  real ( kind = rk ) xn
  real ( kind = rk ) y
  real ( kind = rk ) yn
!
!  LOLIM AND UPLIM DETERMINE THE RANGE OF VALID ARGUMENTS.
!  LOLIM IS NOT LESS THAN THE MACHINE MINIMUM MULTIPLIED BY 5.
!  UPLIM IS NOT GREATER THAN THE MACHINE MAXIMUM DIVIDED BY 5.
!
  save lolim
  save uplim

  data lolim /3.D-78/
  data uplim /1.D+75/

  if ( &
    x < 0.0d0 .or. &
    y <= 0.0d0 .or. &
    ( x + y ) < lolim .or. &
    uplim < x .or. &
    uplim < y ) then
    write ( *, '(a)' ) ''
    write ( *, '(a)' ) 'RC - Error!'
    write ( *, '(a)' ) '  Invalid input arguments.'
    write ( *, '(a,d23.16)' ) '  X = ', x
    write ( *, '(a,d23.16)' ) '  Y = ', y
    write ( *, '(a)' ) ''
    ierr = 1
    rc = 0.0D+00
    return
  end if

  ierr = 0
  xn = x
  yn = y

  do

    mu = ( xn + yn + yn ) / 3.0d0
    sn = ( yn + mu ) / mu - 2.0d0

    if ( abs ( sn ) < errtol ) then
      c1 = 1.0d0 / 7.0d0
      c2 = 9.0d0 / 22.0d0
      s = sn * sn * ( 0.3d0 &
    + sn * ( c1 + sn * ( 0.375d0 + sn * c2 ) ) )
      rc = ( 1.0d0 + s ) / sqrt ( mu )
      return
    end if

    lamda = 2.0d0 * sqrt ( xn ) * sqrt ( yn ) + yn
    xn = ( xn + lamda ) * 0.25d0
    yn = ( yn + lamda ) * 0.25d0

  end do

end
function rd ( x, y, z, errtol, ierr )

!*****************************************************************************80
!
!! RD computes an incomplete elliptic integral of the second kind, RD(X,Y,Z).
!
!  Discussion:
!
!    This function computes an incomplete elliptic integral of the second kind.
!
!    RD(X,Y,Z) = Integral ( 0 <= T < oo )
!
!                    -1/2     -1/2     -3/2
!          (3/2)(T+X)    (T+Y)    (T+Z)    DT,
!
!    where X and Y are nonnegative, X + Y is positive, and Z is positive.
!
!    If X or Y is zero, the integral is complete.
!
!    The duplication theorem is iterated until the variables are
!    nearly equal, and the function is then expanded in Taylor
!    series to fifth order.
!
!    Check:
!
!      RD(X,Y,Z) + RD(Y,Z,X) + RD(Z,X,Y) = 3 / sqrt ( X * Y * Z ),
!      where X, Y, and Z are positive.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    30 May 2018
!
!  Author:
!
!    Original FORTRAN77 version by Bille Carlson, Elaine Notis.
!    This FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Bille Carlson,
!    Computing Elliptic Integrals by Duplication,
!    Numerische Mathematik,
!    Volume 33, 1979, pages 1-16.
!
!    Bille Carlson, Elaine Notis,
!    Algorithm 577, Algorithms for Incomplete Elliptic Integrals,
!    ACM Transactions on Mathematical Software,
!    Volume 7, Number 3, pages 398-403, September 1981.
!
!  Parameters:
!
!    Input, real ( kind = rk ) X, Y, Z, the arguments in the integral.
!
!    Input, real ( kind = rk ) ERRTOL, the error tolerance.
!    The relative error due to truncation is less than
!      3 * ERRTOL ^ 6 / (1-ERRTOL) ^ 3/2.
!    Sample choices:
!      ERRTOL   Relative truncation error less than
!      1.D-3    4.D-18
!      3.D-3    3.D-15
!      1.D-2    4.D-12
!      3.D-2    3.D-9
!      1.D-1    4.D-6
!
!    Output, integer IERR, the error flag.
!    0, no error occurred.
!    1, abnormal termination.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  real ( kind = rk ) c1
  real ( kind = rk ) c2
  real ( kind = rk ) c3
  real ( kind = rk ) c4
  real ( kind = rk ) ea
  real ( kind = rk ) eb
  real ( kind = rk ) ec
  real ( kind = rk ) ed
  real ( kind = rk ) ef
  real ( kind = rk ) epslon
  real ( kind = rk ) errtol
  integer ierr
  real ( kind = rk ) lamda
  real ( kind = rk ) lolim
  real ( kind = rk ) mu
  real ( kind = rk ) power4
  real ( kind = rk ) rd
  real ( kind = rk ) sigma
  real ( kind = rk ) s1
  real ( kind = rk ) s2
  real ( kind = rk ) uplim
  real ( kind = rk ) x
  real ( kind = rk ) xn
  real ( kind = rk ) xndev
  real ( kind = rk ) xnroot
  real ( kind = rk ) y
  real ( kind = rk ) yn
  real ( kind = rk ) yndev
  real ( kind = rk ) ynroot
  real ( kind = rk ) z
  real ( kind = rk ) zn
  real ( kind = rk ) zndev
  real ( kind = rk ) znroot
!
!  LOLIM AND UPLIM DETERMINE THE RANGE OF VALID ARGUMENTS.
!  LOLIM IS NOT LESS THAN 2 / (MACHINE MAXIMUM) ^ (2/3).
!  UPLIM IS NOT GREATER THAN (0.1 * ERRTOL / MACHINE
!  MINIMUM) ^ (2/3), WHERE ERRTOL IS DESCRIBED BELOW.
!  IN THE FOLLOWING TABLE IT IS ASSUMED THAT ERRTOL WILL
!  NEVER BE CHOSEN SMALLER THAN 1.D-5.
!
  save lolim
  save uplim

  data lolim /6.D-51/
  data uplim /1.D+48/

  if ( &
    x < 0.0D+00 .or. &
    y < 0.0D+00 .or. &
    x + y < lolim .or. &
    z < lolim .or. &
    uplim < x .or. &
    uplim < y .or. &
    uplim < z ) then
    write ( *, '(a)' ) ''
    write ( *, '(a)' ) 'RD - Error!'
    write ( *, '(a)' ) '  Invalid input arguments.'
    write ( *, '(a,d23.16)' ) '  X = ', x
    write ( *, '(a,d23.16)' ) '  Y = ', y
    write ( *, '(a,d23.16)' ) '  Z = ', z
    write ( *, '(a)' ) ''
    ierr = 1
    rd = 0.0D+00
    return
  end if

  ierr = 0
  xn = x
  yn = y
  zn = z
  sigma = 0.0d0
  power4 = 1.0d0

  do

    mu = ( xn + yn + 3.0d0 * zn ) * 0.2d0
    xndev = ( mu - xn ) / mu
    yndev = ( mu - yn ) / mu
    zndev = ( mu - zn ) / mu
    epslon = max ( abs ( xndev ), abs ( yndev ), abs ( zndev ) )

    if ( epslon < errtol ) then
      c1 = 3.0d0 / 14.0d0
      c2 = 1.0d0 / 6.0d0
      c3 = 9.0d0 / 22.0d0
      c4 = 3.0d0 / 26.0d0
      ea = xndev * yndev
      eb = zndev * zndev
      ec = ea - eb
      ed = ea - 6.0d0 * eb
      ef = ed + ec + ec
      s1 = ed * ( - c1 + 0.25d0 * c3 * ed - 1.5d0 * c4 * zndev * ef )
      s2 = zndev * ( c2 * ef + zndev * ( - c3 * ec + zndev * c4 * ea ) )
      rd = 3.0d0 * sigma + power4 * ( 1.0d0 + s1 + s2 ) / ( mu * sqrt ( mu ) )

      return
    end if

    xnroot = sqrt ( xn )
    ynroot = sqrt ( yn )
    znroot = sqrt ( zn )
    lamda = xnroot * ( ynroot + znroot ) + ynroot * znroot
    sigma = sigma + power4 / ( znroot * ( zn + lamda ) )
    power4 = power4 * 0.25d0
    xn = ( xn + lamda ) * 0.25d0
    yn = ( yn + lamda ) * 0.25d0
    zn = ( zn + lamda ) * 0.25d0

  end do

end
function rf ( x, y, z, errtol, ierr )

!*****************************************************************************80
!
!! RF computes an incomplete elliptic integral of the first kind, RF(X,Y,Z).
!
!  Discussion:
!
!    This function computes the incomplete elliptic integral of the first kind.
!
!    RF(X,Y,Z) = Integral ( 0 <= T < oo )
!
!                    -1/2     -1/2     -1/2
!          (1/2)(T+X)    (T+Y)    (T+Z)    DT,
!
!    where X, Y, and Z are nonnegative and at most one of them is zero.
!
!    If X or Y or Z is zero, the integral is complete.
!
!    The duplication theorem is iterated until the variables are
!    nearly equal, and the function is then expanded in Taylor
!    series to fifth order.
!
!    Check by addition theorem:
!
!      RF(X,X+Z,X+W) + RF(Y,Y+Z,Y+W) = RF(0,Z,W),
!      where X, Y, Z, W are positive and X * Y = Z * W.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    30 May 2018
!
!  Author:
!
!    Original FORTRAN77 version by Bille Carlson, Elaine Notis.
!    This FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Bille Carlson,
!    Computing Elliptic Integrals by Duplication,
!    Numerische Mathematik,
!    Volume 33, 1979, pages 1-16.
!
!    Bille Carlson, Elaine Notis,
!    Algorithm 577, Algorithms for Incomplete Elliptic Integrals,
!    ACM Transactions on Mathematical Software,
!    Volume 7, Number 3, pages 398-403, September 1981.
!
!  Parameters:
!
!    Input, real ( kind = rk ) X, Y, Z, the arguments in the integral.
!
!    Input, real ( kind = rk ) ERRTOL, the error tolerance.
!    Relative error due to truncation is less than
!      ERRTOL ^ 6 / (4 * (1 - ERRTOL)).
!    Sample choices:
!      ERRTOL   Relative truncation error less than
!      1.D-3    3.D-19
!      3.D-3    2.D-16
!      1.D-2    3.D-13
!      3.D-2    2.D-10
!      1.D-1    3.D-7
!
!    Output, integer IERR, the error flag.
!    0, no error occurred.
!    1, abnormal termination.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  real ( kind = rk ) c1
  real ( kind = rk ) c2
  real ( kind = rk ) c3
  real ( kind = rk ) e2
  real ( kind = rk ) e3
  real ( kind = rk ) epslon
  real ( kind = rk ) errtol
  integer ierr
  real ( kind = rk ) lamda
  real ( kind = rk ) lolim
  real ( kind = rk ) mu
  real ( kind = rk ) rf
  real ( kind = rk ) s
  real ( kind = rk ) uplim
  real ( kind = rk ) x
  real ( kind = rk ) xn
  real ( kind = rk ) xndev
  real ( kind = rk ) xnroot
  real ( kind = rk ) y
  real ( kind = rk ) yn
  real ( kind = rk ) yndev
  real ( kind = rk ) ynroot
  real ( kind = rk ) z
  real ( kind = rk ) zn
  real ( kind = rk ) zndev
  real ( kind = rk ) znroot
!
!  LOLIM AND UPLIM DETERMINE THE RANGE OF VALID ARGUMENTS.
!  LOLIM IS NOT LESS THAN THE MACHINE MINIMUM MULTIPLIED BY 5.
!  UPLIM IS NOT GREATER THAN THE MACHINE MAXIMUM DIVIDED BY 5.
!
  save lolim
  save uplim

  data lolim /3.D-78/
  data uplim /1.D+75/

  if ( &
    x < 0.0D+00 .or. &
    y < 0.0D+00 .or. &
    z < 0.0D+00 .or. &
    x + y < lolim .or. &
    x + z < lolim .or. &
    y + z < lolim .or. &
    uplim <= x .or. &
    uplim <= y .or. &
    uplim <= z ) then
    write ( *, '(a)' ) ''
    write ( *, '(a)' ) 'RF - Error!'
    write ( *, '(a)' ) '  Invalid input arguments.'
    write ( *, '(a,d23.16)' ) '  X = ', x
    write ( *, '(a,d23.16)' ) '  Y = ', y
    write ( *, '(a,d23.16)' ) '  Z = ', z
    write ( *, '(a)' ) ''
    ierr = 1
    rf = 0.0D+00
    return
  end if

  ierr = 0
  xn = x
  yn = y
  zn = z

  do

    mu = ( xn + yn + zn ) / 3.0d0
    xndev = 2.0d0 - ( mu + xn ) / mu
    yndev = 2.0d0 - ( mu + yn ) / mu
    zndev = 2.0d0 - ( mu + zn ) / mu
    epslon = max ( abs ( xndev ), abs ( yndev ), abs ( zndev ) )

    if ( epslon < errtol ) then
      c1 = 1.0d0 / 24.0d0
      c2 = 3.0d0 / 44.0d0
      c3 = 1.0d0 / 14.0d0
      e2 = xndev * yndev - zndev * zndev
      e3 = xndev * yndev * zndev
      s = 1.0d0 + ( c1 * e2 - 0.1d0 - c2 * e3 ) * e2 + c3 * e3
      rf = s / sqrt ( mu )
      return
    end if

    xnroot = sqrt ( xn )
    ynroot = sqrt ( yn )
    znroot = sqrt ( zn )
    lamda = xnroot * ( ynroot + znroot ) + ynroot * znroot
    xn = ( xn + lamda ) * 0.25d0
    yn = ( yn + lamda ) * 0.25d0
    zn = ( zn + lamda ) * 0.25d0

  end do

end
function rj ( x, y, z, p, errtol, ierr )

!*****************************************************************************80
!
!! RJ computes an incomplete elliptic integral of the third kind, RJ(X,Y,Z,P).
!
!  Discussion:
!
!    This function computes an incomplete elliptic integral of the third kind.
!
!    RJ(X,Y,Z,P) = Integral ( 0 <= T < oo )
!
!                  -1/2     -1/2     -1/2     -1
!        (3/2)(T+X)    (T+Y)    (T+Z)    (T+P)  DT,
!
!    where X, Y, and Z are nonnegative, at most one of them is
!    zero, and P is positive.
!
!    If X or Y or Z is zero, then the integral is complete.
!
!    The duplication theorem is iterated until the variables are nearly equal,
!    and the function is then expanded in Taylor series to fifth order.
!
!    Check by addition theorem:
!
!      RJ(X,X+Z,X+W,X+P)
!      + RJ(Y,Y+Z,Y+W,Y+P) + (A-B) * RJ(A,B,B,A) + 3 / sqrt ( A)
!      = RJ(0,Z,W,P), where X,Y,Z,W,P are positive and X * Y
!      = Z * W,  A = P * P * (X+Y+Z+W),  B = P * (P+X) * (P+Y),
!      and B - A = P * (P-Z) * (P-W).
!
!    The sum of the third and fourth terms on the left side is 3 * RC(A,B).
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    30 May 2018
!
!  Author:
!
!    Original FORTRAN77 version by Bille Carlson, Elaine Notis.
!    This FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Bille Carlson,
!    Computing Elliptic Integrals by Duplication,
!    Numerische Mathematik,
!    Volume 33, 1979, pages 1-16.
!
!    Bille Carlson, Elaine Notis,
!    Algorithm 577, Algorithms for Incomplete Elliptic Integrals,
!    ACM Transactions on Mathematical Software,
!    Volume 7, Number 3, pages 398-403, September 1981.
!
!  Parameters:
!
!    Input, real ( kind = rk ) X, Y, Z, P, the arguments in the integral.
!
!    Input, real ( kind = rk ) ERRTOL, the error tolerance.
!    Relative error due to truncation of the series for rj
!    is less than 3 * ERRTOL ^ 6 / (1 - ERRTOL) ^ 3/2.
!    An error tolerance (ETOLRC) will be passed to the subroutine
!    for RC to make the truncation error for RC less than for RJ.
!    Sample choices:
!      ERRTOL   Relative truncation error less than
!      1.D-3    4.D-18
!      3.D-3    3.D-15
!      1.D-2    4.D-12
!      3.D-2    3.D-9
!      1.D-1    4.D-6
!
!    Output, integer IERR, the error flag.
!    0, no error occurred.
!    1, abnormal termination.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  real ( kind = rk ) alfa
  real ( kind = rk ) beta
  real ( kind = rk ) c1
  real ( kind = rk ) c2
  real ( kind = rk ) c3
  real ( kind = rk ) c4
  real ( kind = rk ) ea
  real ( kind = rk ) eb
  real ( kind = rk ) ec
  real ( kind = rk ) e2
  real ( kind = rk ) e3
  real ( kind = rk ) epslon
  real ( kind = rk ) errtol
  real ( kind = rk ) etolrc
  integer ierr
  real ( kind = rk ) lamda
  real ( kind = rk ) lolim
  real ( kind = rk ) mu
  real ( kind = rk ) p
  real ( kind = rk ) pn
  real ( kind = rk ) pndev
  real ( kind = rk ) power4
  real ( kind = rk ) rc
  real ( kind = rk ) rj
  real ( kind = rk ) sigma
  real ( kind = rk ) s1
  real ( kind = rk ) s2
  real ( kind = rk ) s3
  real ( kind = rk ) uplim
  real ( kind = rk ) x
  real ( kind = rk ) xn
  real ( kind = rk ) xndev
  real ( kind = rk ) xnroot
  real ( kind = rk ) y
  real ( kind = rk ) yn
  real ( kind = rk ) yndev
  real ( kind = rk ) ynroot
  real ( kind = rk ) z
  real ( kind = rk ) zn
  real ( kind = rk ) zndev
  real ( kind = rk ) znroot
!
!  LOLIM AND UPLIM DETERMINE THE RANGE OF VALID ARGUMENTS.
!  LOLIM IS NOT LESS THAN THE CUBE ROOT OF THE VALUE
!  OF LOLIM USED IN THE SUBROUTINE FOR RC.
!  UPLIM IS NOT GREATER THAN 0.3 TIMES THE CUBE ROOT OF
!  THE VALUE OF UPLIM USED IN THE SUBROUTINE FOR RC.
!
  save lolim
  save uplim

  data lolim /2.D-26/
  data uplim /3.D+24/

  if ( &
    x < 0.0D+00 .or. &
    y < 0.0D+00 .or. &
    z < 0.0D+00 .or. &
    x + y < lolim .or. &
    x + z < lolim .or. &
    y + z < lolim .or. &
    p < lolim .or. &
    uplim < x .or. &
    uplim < y .or. &
    uplim < z .or. &
    uplim < p ) then
    write ( *, '(a)' ) ''
    write ( *, '(a)' ) 'RJ - Error!'
    write ( *, '(a)' ) '  Invalid input arguments.'
    write ( *, '(a,d23.16)' ) '  X = ', x
    write ( *, '(a,d23.16)' ) '  Y = ', y
    write ( *, '(a,d23.16)' ) '  Z = ', z
    write ( *, '(a,d23.16)' ) '  P = ', p
    write ( *, '(a)' ) ''
    ierr = 1
    rj = 0.0D+00
    return
  end if

  ierr = 0
  xn = x
  yn = y
  zn = z
  pn = p
  sigma = 0.0d0
  power4 = 1.0d0
  etolrc = 0.5d0 * errtol

  do

    mu = ( xn + yn + zn + pn + pn ) * 0.2d0
    xndev = ( mu - xn ) / mu
    yndev = ( mu - yn ) / mu
    zndev = ( mu - zn ) / mu
    pndev = ( mu - pn ) / mu
    epslon = max ( abs ( xndev ), abs ( yndev ), abs ( zndev ), abs ( pndev ) )

    if ( epslon < errtol ) then
      c1 = 3.0d0 / 14.0d0
      c2 = 1.0d0 / 3.0d0
      c3 = 3.0d0 / 22.0d0
      c4 = 3.0d0 / 26.0d0
      ea = xndev * ( yndev + zndev ) + yndev * zndev
      eb = xndev * yndev * zndev
      ec = pndev * pndev
      e2 = ea - 3.0d0 * ec
      e3 = eb + 2.0d0 * pndev * ( ea - ec )
      s1 = 1.0d0 + e2 * ( - c1 + 0.75d0 * c3 * e2 - 1.5d0 * c4 * e3 )
      s2 = eb * ( 0.5d0 * c2 + pndev * ( - c3 - c3 + pndev * c4 ) )
      s3 = pndev * ea * ( c2 - pndev * c3 ) - c2 * pndev * ec
      rj = 3.0d0 * sigma + power4 * ( s1 + s2 + s3 ) / ( mu * sqrt ( mu ) )
      return
    end if

    xnroot = sqrt ( xn )
    ynroot = sqrt ( yn )
    znroot = sqrt ( zn )
    lamda = xnroot * ( ynroot + znroot ) + ynroot * znroot
    alfa = pn * ( xnroot + ynroot + znroot ) &
      + xnroot * ynroot * znroot
    alfa = alfa * alfa
    beta = pn * ( pn + lamda ) * ( pn + lamda )
    sigma = sigma + power4 * rc ( alfa, beta, etolrc, ierr )

    if ( ierr /= 0 ) then
      rj = 0.0D+00
      return
    end if

    power4 = power4 * 0.25d0
    xn = ( xn + lamda ) * 0.25d0
    yn = ( yn + lamda ) * 0.25d0
    zn = ( zn + lamda ) * 0.25d0
    pn = ( pn + lamda ) * 0.25d0

  end do

end
subroutine sncndn ( u, m, sn, cn, dn )

!*****************************************************************************80
!
!! SNCNDN evaluates Jacobi elliptic functions.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    24 June 2018
!
!  Author:
!
!    Original ALGOL version by Roland Bulirsch.
!    FORTRAN90 version by John Burkardt
!
!  Reference:
!
!    Roland Bulirsch,
!    Numerical calculation of elliptic integrals and elliptic functions,
!    Numerische Mathematik,
!    Volume 7, Number 1, 1965, pages 78-90.
!
!  Parameters:
!
!    Input, real ( kind = rk ) U, M, the arguments.
!
!    Output, real ( kind = rk ) SN, CN, DN, the value of the Jacobi
!    elliptic functions sn(u,m), cn(u,m), and dn(u,m).
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  real ( kind = rk ) a
  real ( kind = rk ) b
  real ( kind = rk ) c
  real ( kind = rk ) ca
  real ( kind = rk ) cn
  real ( kind = rk ) d
  real ( kind = rk ) dn
  real ( kind = rk ) m_array(25)
  real ( kind = rk ) n_array(25)
  integer i
  integer l
  real ( kind = rk ) m
  real ( kind = rk ) m_comp
  real ( kind = rk ) sn
  real ( kind = rk ) u
  real ( kind = rk ) u_copy

  m_comp = 1.0D+00 - m
  u_copy = u

  if ( m_comp == 0.0D+00 ) then
    cn = 1.0D+00 / cosh ( u_copy )
    dn = cn
    sn = tanh ( u_copy )
    return
  end if

  if ( 1.0D+00 < m ) then
    d = 1.0D+00 - m_comp
    m_comp = - m_comp / d
    d = sqrt ( d )
    u_copy = d * u_copy
  end if

  ca = sqrt ( epsilon ( ca ) )

  a = 1.0D+00
  dn = 1.0D+00
  l = 25

  do i = 1, 25

    m_array(i) = a
    m_comp = sqrt ( m_comp )
    n_array(i) = m_comp
    c = 0.5D+00 * ( a + m_comp )

    if ( abs ( a - m_comp ) <= ca * a ) then
      l = i
      exit
    end if

    m_comp = a * m_comp
    a = c

  end do

  u_copy = c * u_copy
  sn = sin ( u_copy )
  cn = cos ( u_copy )

  if ( sn /= 0.0D+00 ) then

    a = cn / sn
    c = a * c

    do i = l, 1, -1
      b = m_array(i)
      a = c * a
      c = dn * c
      dn = ( n_array(i) + a ) / ( b + a )
      a = c / b
    end do

    a = 1.0D+00 / sqrt ( c * c + 1.0D+00 )

    if ( sn < 0.0D+00 ) then
      sn = - a
    else
      sn = a
    end if

    cn = c * sn

  end if

  if ( 1.0D+00 < m ) then
    a = dn
    dn = cn
    cn = a
    sn = sn / d
  end if

  return
end
subroutine timestamp ( )

!*****************************************************************************80
!
!! TIMESTAMP prints the current YMDHMS date as a time stamp.
!
!  Example:
!
!    31 May 2001   9:45:54.872 AM
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    18 May 2013
!
!  Author:
!
!    John Burkardt
!
  implicit none

  character ( len = 8 ) ampm
  integer d
  integer h
  integer m
  integer mm
  character ( len = 9 ), parameter, dimension(12) :: month = (/ &
    'January  ', 'February ', 'March    ', 'April    ', &
    'May      ', 'June     ', 'July     ', 'August   ', &
    'September', 'October  ', 'November ', 'December ' /)
  integer n
  integer s
  integer values(8)
  integer y

  call date_and_time ( values = values )

  y = values(1)
  m = values(2)
  d = values(3)
  h = values(5)
  n = values(6)
  s = values(7)
  mm = values(8)

  if ( h < 12 ) then
    ampm = 'AM'
  else if ( h == 12 ) then
    if ( n == 0 .and. s == 0 ) then
      ampm = 'Noon'
    else
      ampm = 'PM'
    end if
  else
    h = h - 12
    if ( h < 12 ) then
      ampm = 'PM'
    else if ( h == 12 ) then
      if ( n == 0 .and. s == 0 ) then
        ampm = 'Midnight'
      else
        ampm = 'AM'
      end if
    end if
  end if

  write ( *, '(i2.2,1x,a,1x,i4,2x,i2,a1,i2.2,a1,i2.2,a1,i3.3,1x,a)' ) &
    d, trim ( month(m) ), y, h, ':', n, ':', s, '.', mm, trim ( ampm )

  return
end
