      subroutine agm_values ( n_data, a, b, fx )

c*********************************************************************72
c
cc AGM_VALUES returns some values of the arithmetic geometric mean.
c
c  Discussion:
c
c    The AGM is defined for nonnegative A and B.
c
c    The AGM of numbers A and B is defined by setting
c
c      A(0) = A,
c      B(0) = B
c
c      A(N+1) = ( A(N) + B(N) ) / 2
c      B(N+1) = sqrt ( A(N) * B(N) )
c
c    The two sequences both converge to AGM(A,B).
c
c    In Mathematica, the AGM can be evaluated by
c
c      ArithmeticGeometricMean [ a, b ]
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    09 February 2008
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Stephen Wolfram,
c    The Mathematica Book,
c    Fourth Edition,
c    Cambridge University Press, 1999,
c    ISBN: 0-521-64314-7,
c    LC: QA76.95.W65.
c
c  Parameters:
c
c    Input/output, integer N_DATA.  The user sets N_DATA to 0 before the
c    first call.  On each call, the routine increments N_DATA by 1, and
c    returns the corresponding data; when there is no more data, the
c    output value of N_DATA will be 0 again.
c
c    Output, double precision A, B, the numbers whose AGM is desired.
c
c    Output, double precision FX, the value of the function.
c
      implicit none

      integer n_max
      parameter ( n_max = 15 )

      double precision a
      double precision a_vec(n_max)
      double precision b
      double precision b_vec(n_max)
      double precision fx
      double precision fx_vec(n_max)
      integer n_data

      save a_vec
      save b_vec
      save fx_vec

      data a_vec /
     &   22.0D+00,
     &   83.0D+00,
     &   42.0D+00,
     &   26.0D+00,
     &    4.0D+00,
     &    6.0D+00,
     &   40.0D+00,
     &   80.0D+00,
     &   90.0D+00,
     &    9.0D+00,
     &   53.0D+00,
     &    1.0D+00,
     &    1.0D+00,
     &    1.0D+00,
     &    1.5D+00 /
      data b_vec /
     &   96.0D+00,
     &   56.0D+00,
     &    7.0D+00,
     &   11.0D+00,
     &   63.0D+00,
     &   45.0D+00,
     &   75.0D+00,
     &    0.0D+00,
     &   35.0D+00,
     &    1.0D+00,
     &   53.0D+00,
     &    2.0D+00,
     &    4.0D+00,
     &    8.0D+00,
     &    8.0D+00 /
      data fx_vec /
     &   52.274641198704240049D+00,
     &   68.836530059858524345D+00,
     &   20.659301196734009322D+00,
     &   17.696854873743648823D+00,
     &   23.867049721753300163D+00,
     &   20.717015982805991662D+00,
     &   56.127842255616681863D+00,
     &    0.000000000000000000D+00,
     &   59.269565081229636528D+00,
     &   3.9362355036495554780D+00,
     &   53.000000000000000000D+00,
     &   1.4567910310469068692D+00,
     &   2.2430285802876025701D+00,
     &   3.6157561775973627487D+00,
     &   4.0816924080221632670D+00 /

      if ( n_data .lt. 0 ) then
        n_data = 0
      end if

      n_data = n_data + 1

      if ( n_max .lt. n_data ) then
        n_data = 0
        a = 0.0D+00
        b = 0.0D+00
        fx = 0.0D+00
      else
        a = a_vec(n_data)
        b = b_vec(n_data)
        fx = fx_vec(n_data)
      end if

      return
      end
      subroutine gamma_values ( n_data, x, fx )

c*********************************************************************72
c
cc GAMMA_VALUES returns some values of the Gamma function.
c
c  Discussion:
c
c    The Gamma function is defined as:
c
c      Gamma(Z) = Integral ( 0 <= T < +oo ) T^(Z-1) exp(-T) dT
c
c    It satisfies the recursion:
c
c      Gamma(X+1) = X * Gamma(X)
c
c    Gamma is undefined for nonpositive integral X.
c    Gamma(0.5) = sqrt(PI)
c    For N a positive integer, Gamma(N+1) = N!, the standard factorial.
c
c    In Mathematica, the function can be evaluated by:
c
c      Gamma[x]
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    18 January 2008
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Milton Abramowitz, Irene Stegun,
c    Handbook of Mathematical Functions,
c    National Bureau of Standards, 1964,
c    ISBN: 0-486-61272-4,
c    LC: QA47.A34.
c
c    Stephen Wolfram,
c    The Mathematica Book,
c    Fourth Edition,
c    Cambridge University Press, 1999,
c    ISBN: 0-521-64314-7,
c    LC: QA76.95.W65.
c
c  Parameters:
c
c    Input/output, integer N_DATA.  The user sets N_DATA to 0 before the
c    first call.  On each call, the routine increments N_DATA by 1, and
c    returns the corresponding data; when there is no more data, the
c    output value of N_DATA will be 0 again.
c
c    Output, double precision X, the argument of the function.
c
c    Output, double precision FX, the value of the function.
c
      implicit none

      integer n_max
      parameter ( n_max = 25 )

      double precision fx
      double precision fx_vec(n_max)
      integer n_data
      double precision x
      double precision x_vec(n_max)

      save fx_vec
      save x_vec

      data fx_vec /
     &  -0.3544907701811032D+01,
     &  -0.1005871979644108D+03,
     &   0.9943258511915060D+02,
     &   0.9513507698668732D+01,
     &   0.4590843711998803D+01,
     &   0.2218159543757688D+01,
     &   0.1772453850905516D+01,
     &   0.1489192248812817D+01,
     &   0.1164229713725303D+01,
     &   0.1000000000000000D+01,
     &   0.9513507698668732D+00,
     &   0.9181687423997606D+00,
     &   0.8974706963062772D+00,
     &   0.8872638175030753D+00,
     &   0.8862269254527580D+00,
     &   0.8935153492876903D+00,
     &   0.9086387328532904D+00,
     &   0.9313837709802427D+00,
     &   0.9617658319073874D+00,
     &   0.1000000000000000D+01,
     &   0.2000000000000000D+01,
     &   0.6000000000000000D+01,
     &   0.3628800000000000D+06,
     &   0.1216451004088320D+18,
     &   0.8841761993739702D+31 /
      data x_vec /
     &  -0.50D+00,
     &  -0.01D+00,
     &   0.01D+00,
     &   0.10D+00,
     &   0.20D+00,
     &   0.40D+00,
     &   0.50D+00,
     &   0.60D+00,
     &   0.80D+00,
     &   1.00D+00,
     &   1.10D+00,
     &   1.20D+00,
     &   1.30D+00,
     &   1.40D+00,
     &   1.50D+00,
     &   1.60D+00,
     &   1.70D+00,
     &   1.80D+00,
     &   1.90D+00,
     &   2.00D+00,
     &   3.00D+00,
     &   4.00D+00,
     &  10.00D+00,
     &  20.00D+00,
     &  30.00D+00 /

      if ( n_data .lt. 0 ) then
        n_data = 0
      end if

      n_data = n_data + 1

      if ( n_max .lt. n_data ) then
        n_data = 0
        x = 0.0D+00
        fx = 0.0D+00
      else
        x = x_vec(n_data)
        fx = fx_vec(n_data)
      end if

      return
      end
      subroutine gamma_log_values ( n_data, x, fx )

c*********************************************************************72
c
cc GAMMA_LOG_VALUES returns some values of the Log Gamma function.
c
c  Discussion:
c
c    In Mathematica, the function can be evaluated by:
c
c      Log[Gamma[x]]
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    03 January 2006
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Milton Abramowitz, Irene Stegun,
c    Handbook of Mathematical Functions,
c    National Bureau of Standards, 1964,
c    ISBN: 0-486-61272-4,
c    LC: QA47.A34.
c
c    Stephen Wolfram,
c    The Mathematica Book,
c    Fourth Edition,
c    Cambridge University Press, 1999,
c    ISBN: 0-521-64314-7,
c    LC: QA76.95.W65.
c
c  Parameters:
c
c    Input/output, integer N_DATA.  The user sets N_DATA to 0 before the
c    first call.  On each call, the routine increments N_DATA by 1, and
c    returns the corresponding data; when there is no more data, the
c    output value of N_DATA will be 0 again.
c
c    Output, double precision X, the argument of the function.
c
c    Output, double precision FX, the value of the function.
c
      implicit none

      integer n_max
      parameter ( n_max = 20 )

      double precision fx
      double precision fx_vec(n_max)
      integer n_data
      double precision x
      double precision x_vec(n_max)

      save fx_vec
      save x_vec

      data fx_vec /
     &  0.1524063822430784D+01,
     &  0.7966778177017837D+00,
     &  0.3982338580692348D+00,
     &  0.1520596783998375D+00,
     &  0.0000000000000000D+00,
     & -0.4987244125983972D-01,
     & -0.8537409000331584D-01,
     & -0.1081748095078604D+00,
     & -0.1196129141723712D+00,
     & -0.1207822376352452D+00,
     & -0.1125917656967557D+00,
     & -0.9580769740706586D-01,
     & -0.7108387291437216D-01,
     & -0.3898427592308333D-01,
     &  0.00000000000000000D+00,
     &  0.69314718055994530D+00,
     &  0.17917594692280550D+01,
     &  0.12801827480081469D+02,
     &  0.39339884187199494D+02,
     &  0.71257038967168009D+02 /
      data x_vec /
     &  0.20D+00,
     &  0.40D+00,
     &  0.60D+00,
     &  0.80D+00,
     &  1.00D+00,
     &  1.10D+00,
     &  1.20D+00,
     &  1.30D+00,
     &  1.40D+00,
     &  1.50D+00,
     &  1.60D+00,
     &  1.70D+00,
     &  1.80D+00,
     &  1.90D+00,
     &  2.00D+00,
     &  3.00D+00,
     &  4.00D+00,
     & 10.00D+00,
     & 20.00D+00,
     & 30.00D+00 /

      if ( n_data .lt. 0 ) then
        n_data = 0
      end if

      n_data = n_data + 1

      if ( n_max .lt. n_data ) then
        n_data = 0
        x = 0.0D+00
        fx = 0.0D+00
      else
        x = x_vec(n_data)
        fx = fx_vec(n_data)
      end if

      return
      end
      subroutine get_unit ( iunit )

c*********************************************************************72
c
cc GET_UNIT returns a free FORTRAN unit number.
c
c  Discussion:
c
c    A "free" FORTRAN unit number is a value between 1 and 99 which
c    is not currently associated with an I/O device.  A free FORTRAN unit
c    number is needed in order to open a file with the OPEN command.
c
c    If IUNIT = 0, then no free FORTRAN unit could be found, although
c    all 99 units were checked (except for units 5, 6 and 9, which
c    are commonly reserved for console I/O).
c
c    Otherwise, IUNIT is a value between 1 and 99, representing a
c    free FORTRAN unit.  Note that GET_UNIT assumes that units 5 and 6
c    are special, and will never return those values.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    02 September 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Output, integer IUNIT, the free unit number.
c
      implicit none

      integer i
      integer iunit
      logical value

      iunit = 0

      do i = 1, 99

        if ( i .ne. 5 .and. i .ne. 6 .and. i .ne. 9 ) then

          inquire ( unit = i, opened = value, err = 10 )

          if ( .not. value ) then
            iunit = i
            return
          end if

        end if

10      continue

      end do

      return
      end
      function i4_log_10 ( i )

c*********************************************************************72
c
cc I4_LOG_10 returns the integer part of the logarithm base 10 of ABS(X).
c
c  Discussion:
c
c    I4_LOG_10 ( I ) + 1 is the number of decimal digits in I.
c
c  Example:
c
c        I  I4_LOG_10
c    -----  --------
c        0    0
c        1    0
c        2    0
c        9    0
c       10    1
c       11    1
c       99    1
c      100    2
c      101    2
c      999    2
c     1000    3
c     1001    3
c     9999    3
c    10000    4
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    03 January 2007
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer I, the number whose logarithm base 10 is desired.
c
c    Output, integer I4_LOG_10, the integer part of the logarithm base 10 of
c    the absolute value of X.
c
      implicit none

      integer i
      integer i_abs
      integer i4_log_10
      integer ten_pow

      if ( i .eq. 0 ) then

        i4_log_10 = 0

      else

        i4_log_10 = 0
        ten_pow = 10

        i_abs = abs ( i )

10      continue

        if ( ten_pow .le. i_abs ) then
          i4_log_10 = i4_log_10 + 1
          ten_pow = ten_pow * 10
          go to 10
        end if

      end if

      return
      end
      function i4_modp ( i, j )

c*********************************************************************72
c
cc I4_MODP returns the nonnegative remainder of integer division.
c
c  Discussion:
c
c    If
c      NREM = I4_MODP ( I, J )
c      NMULT = ( I - NREM ) / J
c    then
c      I = J * NMULT + NREM
c    where NREM is always nonnegative.
c
c    The MOD function computes a result with the same sign as the
c    quantity being divided.  Thus, suppose you had an angle A,
c    and you wanted to ensure that it was between 0 and 360.
c    Then mod(A,360) would do, if A was positive, but if A
c    was negative, your result would be between -360 and 0.
c
c    On the other hand, I4_MODP(A,360) is between 0 and 360, always.
c
c  Example:
c
c        I     J     MOD I4_MODP    Factorization
c
c      107    50       7       7    107 =  2 *  50 + 7
c      107   -50       7       7    107 = -2 * -50 + 7
c     -107    50      -7      43   -107 = -3 *  50 + 43
c     -107   -50      -7      43   -107 =  3 * -50 + 43
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    30 December 2006
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer I, the number to be divided.
c
c    Input, integer J, the number that divides I.
c
c    Output, integer I4_MODP, the nonnegative remainder when I is
c    divided by J.
c
      implicit none

      integer i
      integer i4_modp
      integer j
      integer value

      if ( j .eq. 0 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'I4_MODP - Fatal error!'
        write ( *, '(a,i8)' ) '  Illegal divisor J = ', j
        stop 1
      end if

      value = mod ( i, j )

      if ( value .lt. 0 ) then
        value = value + abs ( j )
      end if

      i4_modp = value

      return
      end
      function i4_uniform_ab ( a, b, seed )

c*********************************************************************72
c
cc I4_UNIFORM_AB returns a scaled pseudorandom I4.
c
c  Discussion:
c
c    An I4 is an integer value.
c
c    The pseudorandom number should be uniformly distributed
c    between A and B.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    12 November 2006
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Paul Bratley, Bennett Fox, Linus Schrage,
c    A Guide to Simulation,
c    Springer Verlag, pages 201-202, 1983.
c
c    Pierre L'Ecuyer,
c    Random Number Generation,
c    in Handbook of Simulation,
c    edited by Jerry Banks,
c    Wiley Interscience, page 95, 1998.
c
c    Bennett Fox,
c    Algorithm 647:
c    Implementation and Relative Efficiency of Quasirandom
c    Sequence Generators,
c    ACM Transactions on Mathematical Software,
c    Volume 12, Number 4, pages 362-376, 1986.
c
c    Peter Lewis, Allen Goodman, James Miller
c    A Pseudo-Random Number Generator for the System/360,
c    IBM Systems Journal,
c    Volume 8, pages 136-143, 1969.
c
c  Parameters:
c
c    Input, integer A, B, the limits of the interval.
c
c    Input/output, integer SEED, the "seed" value, which should NOT be 0.
c    On output, SEED has been updated.
c
c    Output, integer I4_UNIFORM_AB, a number between A and B.
c
      implicit none

      integer a
      integer b
      integer i4_huge
      parameter ( i4_huge = 2147483647 )
      integer i4_uniform_ab
      integer k
      real r
      integer seed
      integer value

      if ( seed .eq. 0 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'I4_UNIFORM_AB - Fatal error!'
        write ( *, '(a)' ) '  Input value of SEED = 0.'
        stop 1
      end if

      k = seed / 127773

      seed = 16807 * ( seed - k * 127773 ) - k * 2836

      if ( seed .lt. 0 ) then
        seed = seed + i4_huge
      end if

      r = real ( seed ) * 4.656612875E-10
c
c  Scale R to lie between A-0.5 and B+0.5.
c
      r = ( 1.0E+00 - r ) * ( real ( min ( a, b ) ) - 0.5E+00 )
     &  +             r   * ( real ( max ( a, b ) ) + 0.5E+00 )
c
c  Use rounding to convert R to an integer between A and B.
c
      value = nint ( r )

      value = max ( value, min ( a, b ) )
      value = min ( value, max ( a, b ) )

      i4_uniform_ab = value

      return
      end
      function i4_wrap ( ival, ilo, ihi )

c*********************************************************************72
c
cc I4_WRAP forces an I4 to lie between given limits by wrapping.
c
c  Example:
c
c    ILO = 4, IHI = 8
c
c    I  Value
c
c    -2     8
c    -1     4
c     0     5
c     1     6
c     2     7
c     3     8
c     4     4
c     5     5
c     6     6
c     7     7
c     8     8
c     9     4
c    10     5
c    11     6
c    12     7
c    13     8
c    14     4
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    31 December 2006
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer IVAL, an integer value.
c
c    Input, integer ILO, IHI, the desired bounds for the integer value.
c
c    Output, integer I4_WRAP, a "wrapped" version of IVAL.
c
      implicit none

      integer i4_modp
      integer i4_wrap
      integer ihi
      integer ilo
      integer ival
      integer jhi
      integer jlo
      integer value
      integer wide

      jlo = min ( ilo, ihi )
      jhi = max ( ilo, ihi )

      wide = jhi - jlo + 1

      if ( wide .eq. 1 ) then
        value = jlo
      else
        value = jlo + i4_modp ( ival - jlo, wide )
      end if

      i4_wrap = value

      return
      end
      subroutine i4int_to_r8int ( imin, imax, i, rmin, rmax, r )

c*********************************************************************72
c
cc I4INT_TO_R8INT maps an I4INT to an R8INT.
c
c  Discussion:
c
c    The formula used is:
c
c      R := RMIN + ( RMAX - RMIN ) * ( I - IMIN ) / ( IMAX - IMIN )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    04 September 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer IMIN, IMAX, the range.
c
c    Input, integer I, the integer to be converted.
c
c    Input, double precision RMIN, RMAX, the range.
c
c    Output, double precision R, the corresponding value in [RMIN,RMAX].
c
      implicit none

      integer i
      integer imax
      integer imin
      double precision r
      double precision rmax
      double precision rmin

      if ( imax .eq. imin ) then

        r = 0.5D+00 * ( rmin + rmax )

      else

        r = ( dble ( imax - i        ) * rmin   
     &      + dble (        i - imin ) * rmax ) 
     &      / dble ( imax     - imin )

      end if

      return
      end
      subroutine i4vec_indicator0 ( n, a )

c*********************************************************************72
c
cc I4VEC_INDICATOR0 sets an I4VEC to the indicator vector (0,1,2,...).
c
c  Discussion:
c
c    An I4VEC is a vector of I4's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    27 September 2014
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of elements of A.
c
c    Output, integer A(N), the array.
c
      implicit none

      integer n

      integer a(n)
      integer i

      do i = 1, n
        a(i) = i - 1
      end do

      return
      end
      subroutine i4vec_indicator1 ( n, a )

c*********************************************************************72
c
cc I4VEC_INDICATOR1 sets an I4VEC to the indicator vector (1,2,3,...).
c
c  Discussion:
c
c    An I4VEC is a vector of I4's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    27 September 2014
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of elements of A.
c
c    Output, integer A(N), the array.
c
      implicit none

      integer n

      integer a(n)
      integer i

      do i = 1, n
        a(i) = i
      end do

      return
      end
      subroutine i4vec_permute ( n, p, a )

c*********************************************************************72
c
cc I4VEC_PERMUTE permutes an I4VEC in place.
c
c  Discussion:
c
c    An I4VEC is a vector of I4's.
c
c    This routine permutes an array of integer "objects", but the same
c    logic can be used to permute an array of objects of any arithmetic
c    type, or an array of objects of any complexity.  The only temporary
c    storage required is enough to store a single object.  The number
c    of data movements made is N + the number of cycles of order 2 or more,
c    which is never more than N + N/2.
c
c  Example:
c
c    Input:
c
c      N = 5
c      P = (   2,   4,   5,   1,   3 )
c      A = (   1,   2,   3,   4,   5 )
c
c    Output:
c
c      A    = (   2,   4,   5,   1,   3 ).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    18 July 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of objects.
c
c    Input, integer P(N), the permutation.  P(I) = J means
c    that the I-th element of the output array should be the J-th
c    element of the input array.
c
c    Input/output, integer A(N), the array to be permuted.
c
      implicit none

      integer n

      integer a(n)
      integer a_temp
      integer i
      integer ierror
      integer iget
      integer iput
      integer istart
      integer p(n)

      call perm_check1 ( n, p )
c
c  Search for the next element of the permutation that has not been used.
c
      do istart = 1, n

        if ( p(istart) .lt. 0 ) then

          go to 20

        else if ( p(istart) .eq. istart ) then

          p(istart) = - p(istart)
          go to 20

        else

          a_temp = a(istart)
          iget = istart
c
c  Copy the new value into the vacated entry.
c
10        continue

            iput = iget
            iget = p(iget)

            p(iput) = - p(iput)

            if ( iget .lt. 1 .or. n .lt. iget ) then
              write ( *, '(a)' ) ' '
              write ( *, '(a)' ) 'I4VEC_PERMUTE - Fatal error!'
              write ( *, '(a)' ) '  An index is out of range.'
              write ( *, '(a,i8,a,i8)' ) '  P(', iput, ') = ', iget
              stop 1
            end if

            if ( iget .eq. istart ) then
              a(iput) = a_temp
              go to 20
            end if

            a(iput) = a(iget)

          go to 10

        end if

20      continue

      end do
c
c  Restore the signs of the entries.
c
      do i = 1, n
        p(1:n) = - p(1:n)
      end do

      return
      end
      subroutine i4vec_print ( n, a, title )

c*********************************************************************72
c
cc I4VEC_PRINT prints an I4VEC.
c
c  Discussion:
c
c    An I4VEC is a vector of integer values.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    27 November 2006
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of components of the vector.
c
c    Input, integer A(N), the vector to be printed.
c
c    Input, character*(*) TITLE, a title.
c
      implicit none

      integer n

      integer a(n)
      integer i
      character*(*) title

      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) trim ( title )
      write ( *, '(a)' ) ' '
      do i = 1, n
        write ( *, '(2x,i8,a,1x,i12)' ) i, ':', a(i)
      end do

      return
      end
      subroutine legendre_zeros ( n, x )

c*********************************************************************72
c
cc LEGENDRE_ZEROS computes the zeros of the Legendre polynomial of degree N.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    17 June 2011
c
c  Author:
c
c    Original FORTRAN77 version by Philip Davis, Philip Rabinowitz.
c    This FORTRAN77 version by John Burkardt.
c
c  Reference:
c
c    Philip Davis, Philip Rabinowitz,
c    Methods of Numerical Integration,
c    Second Edition,
c    Dover, 2007,
c    ISBN: 0486453391,
c    LC: QA299.3.D28.
c
c  Parameters:
c
c    Input, integer N, the order.
c    0 .lt. N.
c
c    Output, double precision X(N), the abscissas.
c
      implicit none

      integer n

      double precision d1
      double precision d2pn
      double precision d3pn
      double precision d4pn
      double precision dp
      double precision dpn
      double precision e1
      double precision fx
      double precision h
      integer i
      integer iback
      integer k
      integer m
      integer mp1mi
      integer ncopy
      integer nmove
      double precision p
      double precision pk
      double precision pkm1
      double precision pkp1
      double precision r8_pi
      parameter ( r8_pi = 3.141592653589793D+00 )
      double precision t
      double precision u
      double precision v
      double precision x(n)
      double precision x0
      double precision xtemp

      e1 = dble ( n * ( n + 1 ) )

      m = ( n + 1 ) / 2

      do i = 1, m

        mp1mi = m + 1 - i

        t = dble ( 4 * i - 1 ) * r8_pi 
     &    / dble ( 4 * n + 2 )

        x0 = cos ( t ) * ( 1.0D+00 - ( 1.0D+00 - 1.0D+00 
     &    / dble ( n ) ) 
     &    / dble ( 8 * n * n ) )

        pkm1 = 1.0D+00
        pk = x0

        do k = 2, n
          pkp1 = 2.0D+00 * x0 * pk - pkm1 - ( x0 * pk - pkm1 ) 
     &      / dble ( k )
          pkm1 = pk
          pk = pkp1
        end do

        d1 = dble ( n ) * ( pkm1 - x0 * pk )

        dpn = d1 / ( 1.0D+00 - x0 ) / ( 1.0D+00 + x0 )

        d2pn = ( 2.0D+00 * x0 * dpn - e1 * pk ) / ( 1.0D+00 - x0 ) 
     &    / ( 1.0D+00 + x0 )

        d3pn = ( 4.0D+00 * x0 * d2pn + ( 2.0D+00 - e1 ) * dpn ) 
     &    / ( 1.0D+00 - x0 ) / ( 1.0D+00 + x0 )

        d4pn = ( 6.0D+00 * x0 * d3pn + ( 6.0D+00 - e1 ) * d2pn ) 
     &    / ( 1.0D+00 - x0 ) / ( 1.0D+00 + x0 )

        u = pk / dpn
        v = d2pn / dpn
c
c  Initial approximation H:
c
        h = - u * ( 1.0D+00 + 0.5D+00 * u * ( v + u * ( v * v - d3pn / 
     &    ( 3.0D+00 * dpn ) ) ) )
c
c  Refine H using one step of Newton's method:
c
        p = pk + h * ( dpn + 0.5D+00 * h * ( d2pn + h / 3.0D+00 
     &    * ( d3pn + 0.25D+00 * h * d4pn ) ) )

        dp = dpn + h * ( d2pn + 0.5D+00 * h * 
     &    ( d3pn + h * d4pn / 3.0D+00 ) )

        h = h - p / dp

        xtemp = x0 + h

        x(mp1mi) = xtemp

        fx = d1 - h * e1 * ( pk + 0.5D+00 * h * ( dpn + h / 3.0D+00 
     &    * ( d2pn + 0.25D+00 * h 
     &    * ( d3pn + 0.2D+00 * h * d4pn ) ) ) )

      end do

      if ( mod ( n, 2 ) .eq. 1 ) then
        x(1) = 0.0D+00
      end if
c
c  Shift the data up.
c
      nmove = ( n + 1 ) / 2
      ncopy = n - nmove

      do i = 1, nmove
        iback = n + 1 - i
        x(iback) = x(iback-ncopy)
      end do
c
c  Reflect values for the negative abscissas.
c
      do i = 1, n - nmove
        x(i) = - x(n+1-i)
      end do

      return
      end
      subroutine perm_check0 ( n, p )

c*********************************************************************72
c
cc PERM_CHECK0 checks a 0-based permutation.
c
c  Discussion:
c
c    The routine verifies that each of the integers from 0 to
c    to N-1 occurs among the N entries of the permutation.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    24 October 2014
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries.
c
c    Input, integer P(N), the array to check.
c
      implicit none

      integer n

      integer ierror
      integer location
      integer p(n)
      integer value

      do value = 0, n - 1

        ierror = 1

        do location = 1, n
          if ( p(location) .eq. value ) then
            ierror = 0
            go to 10
          end if
        end do

10      continue

        if ( ierror .ne. 0 ) then
          write ( *, '(a)' ) ' '
          write ( *, '(a)' ) 'PERM_CHECK0 - Fatal error!'
          write ( *, '(a,i4)' ) '  Permutation is missing value ', value
          stop 1
        end if

      end do

      return
      end
      subroutine perm_check1 ( n, p )

c*********************************************************************72
c
cc PERM_CHECK1 checks a 1-based permutation.
c
c  Discussion:
c
c    The routine verifies that each of the integers from 1 to
c    to N occurs among the N entries of the permutation.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    24 October 2014
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries.
c
c    Input, integer P(N), the array to check.
c
      implicit none

      integer n

      integer ierror
      integer location
      integer p(n)
      integer value

      do value = 1, n

        ierror = 1

        do location = 1, n
          if ( p(location) .eq. value ) then
            ierror = 0
            go to 10
          end if
        end do

10      continue

        if ( ierror .ne. 0 ) then
          write ( *, '(a)' ) ' '
          write ( *, '(a)' ) 'PERM_CHECK1 - Fatal error!'
          write ( *, '(a,i4)' ) '  Permutation is missing value ', value
          stop 1
        end if

      end do

      return
      end
      subroutine perm_uniform ( n, seed, p )

c*********************************************************************72
c
cc PERM_UNIFORM selects a random permutation of N objects.
c
c  Discussion:
c
c    The routine assumes the objects are labeled 1, 2, ... N.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    12 November 2014
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Albert Nijenhuis, Herbert Wilf,
c    Combinatorial Algorithms for Computers and Calculators,
c    Academic Press, 1978,
c    ISBN: 0-12-519260-6,
c    LC: QA164.N54.
c
c  Parameters:
c
c    Input, integer N, the number of objects to be permuted.
c
c    Input/output, integer SEED, a seed for the random number generator.
c
c    Output, integer P(N), a permutation of ( 1, 2, ..., N ), in standard
c    index form.
c
      implicit none

      integer n

      integer i
      integer i4_uniform_ab
      integer j
      integer p(n)
      integer pk
      integer seed

      do i = 1, n
        p(i) = i
      end do

      do i = 1, n - 1
        j = i4_uniform_ab ( i, n, seed )
        pk = p(i)
        p(i) = p(j)
        p(j) = pk
      end do

      return
      end
      function r8_abs ( x )

c*********************************************************************72
c
cc R8_ABS returns the absolute value of an R8.
c
c  Discussion:
c
c    FORTRAN90 supplies the ABS function, which should be used instead
c    of this function!
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    06 September 2005
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision X, the number whose absolute value is desired.
c
c    Output, double precision R8_ABS, the absolute value of X.
c
      implicit none

      double precision r8_abs
      double precision x

      if ( 0.0D+00 .le. x ) then
        r8_abs = + x
      else
        r8_abs = - x
      end if

      return
      end
      function r8_acos ( c )

c*********************************************************************72
c
cc R8_ACOS computes the arc cosine function, with argument truncation.
c
c  Discussion:
c
c    If you call your system ACOS routine with an input argument that is
c    even slightly outside the range [-1.0, 1.0 ], you may get an unpleasant
c    surprise (I did).
c
c    This routine simply truncates arguments outside the range.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    19 October 2012
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision C, the argument.
c
c    Output, double precision R8_ACOS, an angle whose cosine is C.
c
      implicit none

      double precision c
      double precision c2
      double precision r8_acos

      c2 = c
      c2 = max ( c2, -1.0D+00 )
      c2 = min ( c2, +1.0D+00 )

      r8_acos = acos ( c2 )

      return
      end
      function r8_acosh ( x )

c*********************************************************************72
c
cc R8_ACOSH returns the inverse hyperbolic cosine of a number.
c
c  Discussion:
c
c    One formula is:
c
c      R8_ACOSH = LOG ( X + SQRT ( X * X - 1.0 ) )
c
c    but this formula suffers from roundoff and overflow problems.
c    The formula used here was recommended by W Kahan, as discussed
c    by Moler.
c
c    Applying the inverse function
c
c      Y = R8_ACOSH ( X )
c
c    implies that
c
c      X = COSH(Y) = 0.5 * ( EXP(Y) + EXP(-Y) ).
c
c    For every X greater than or equal to 1, there are two possible
c    choices Y such that X = COSH(Y), differing only in sign.  It
c    is usual to resolve this choice by taking the value of 
c    R8_ACOSH ( X ) to be nonnegative.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    29 November 2007
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Cleve Moler,
c    Trigonometry is a Complex Subject,
c    MATLAB News and Notes,
c    Summer 1998.
c
c  Parameters:
c
c    Input, double precision X, the number whose inverse hyperbolic 
c    cosine is desired.  X should be greater than or equal to 1.
c
c    Output, double precision R8_ACOSH, the inverse hyperbolic cosine of 
c    X.  The principal value (that is, the positive value of the two ) 
c    is returned.
c
      implicit none

      double precision r8_acosh
      double precision x

      if ( x .lt. 1.0D+00 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8_ACOSH - Fatal error!'
        write ( *, '(a)' ) '  Argument X must satisfy 1 <= X.'
        write ( *, '(a,g14.6)' ) '  The input X = ', x
        stop
      end if

      r8_acosh = 2.0D+00 * log ( 
     &    sqrt ( 0.5D+00 * ( x + 1.0D+00 ) ) 
     &  + sqrt ( 0.5D+00 * ( x - 1.0D+00 ) ) )

      return
      end
      function r8_add ( x, y )

c*********************************************************************72
c
cc R8_ADD adds two R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    11 August 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision X, Y, the numbers to be added.
c
c    Output, double precision R8_ADD, the sum of X and Y.
c
      implicit none

      double precision r8_add
      double precision x
      double precision y

      r8_add = x + y

      return
      end
      function r8_agm ( a, b )

c*********************************************************************72
c
cc R8_AGM computes the arithmetic-geometric mean of A and B.
c
c  Discussion:
c
c    The AGM is defined for nonnegative A and B.
c
c    The AGM of numbers A and B is defined by setting
c
c      A(0) = A,
c      B(0) = B
c
c      A(N+1) = ( A(N) + B(N) ) / 2
c      B(N+1) = sqrt ( A(N) * B(N) )
c
c    The two sequences both converge to AGM(A,B).
c
c    In Mathematica, the AGM can be evaluated by
c
c      ArithmeticGeometricMean [ a, b ]
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    27 July 2014
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Stephen Wolfram,
c    The Mathematica Book,
c    Fourth Edition,
c    Cambridge University Press, 1999,
c    ISBN: 0-521-64314-7,
c    LC: QA76.95.W65.
c
c  Parameters:
c
c    Input, double precision A, B, the arguments whose AGM is to be computed.
c    0 <= A, 0 <= B.
c
c    Output, double precision R8_AGM, the arithmetic-geometric mean of A and B.
c
      implicit none

      double precision a
      double precision a1
      double precision a2
      double precision b
      double precision b1
      double precision b2
      integer it
      integer it_max
      parameter ( it_max = 1000 )
      double precision r8_agm
      double precision r8_epsilon
      double precision tol

      if ( a .lt. 0.0D+00 ) then
        write ( *, '(a)' ) ''
        write ( *, '(a)' ) 'R8_AGM - Fatal error!'
        write ( *, '(a)' ) '  A < 0.'
        stop 1
      end if

      if ( b .lt. 0.0D+00 ) then
        write ( *, '(a)' ) ''
        write ( *, '(a)' ) 'R8_AGM - Fatal error!'
        write ( *, '(a)' ) '  B < 0.'
        stop 1
      end if

      if ( a .eq. 0.0D+00 .or. b .eq. 0.0D+00 ) then
        r8_agm = 0.0D+00
        return
      end if

      it = 0
      tol = 100.0D+00 * r8_epsilon ( )

      a1 = a
      b1 = b

10    continue

        it = it + 1

        a2 = ( a1 + b1 ) / 2.0D+00
        b2 = sqrt ( a1 * b1 )

        if ( abs ( a2 - b2 ) .le. tol * ( a2 + b2 ) ) then
          go to 20
        end if

        if ( it_max .lt. it ) then
          go to 20
        end if

        a1 = a2
        b1 = b2

      go to 10

20    continue

      r8_agm = a2

      return
      end
      function r8_aint ( x )

c********************************************************************72
c
cc R8_AINT truncates an R8 argument to an integer.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    18 October 2011
c
c  Author:
c
c    John Burkardt.
c
c  Parameters:
c
c    Input, double precision X, the argument.
c
c    Output, double precision VALUE, the truncated version of X.
c
      implicit none

      double precision r8_aint
      double precision value
      double precision x

      if ( x .lt. 0.0D+00 ) then
        value = - int ( abs ( x ) )
      else
        value =   int ( abs ( x ) )
      end if

      r8_aint = value

      return
      end
      function r8_asin ( s )

c*********************************************************************72
c
cc R8_ASIN computes the arc sine function, with argument truncation.
c
c  Discussion:
c
c    If you call your system ASIN routine with an input argument that is
c    even slightly outside the range [-1.0, 1.0 ], you may get an unpleasant
c    surprise (I did).
c
c    This routine simply truncates arguments outside the range.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    06 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision S, the argument.
c
c    Output, double precision R8_ASIN, an angle whose sine is S.
c
      implicit none

      double precision r8_asin
      double precision s
      double precision s2

      s2 = s
      s2 = max ( s2, -1.0D+00 )
      s2 = min ( s2, +1.0D+00 )

      r8_asin = asin ( s2 )

      return
      end
      function r8_asinh ( x )

c*********************************************************************72
c
cc R8_ASINH returns the inverse hyperbolic sine of a number.
c
c  Definition:
c
c    The assertion that:
c
c      Y = R8_ASINH ( X )
c
c    implies that
c
c      X = SINH(Y) = 0.5 * ( EXP(Y) - EXP(-Y) ).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    29 November 2007
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision X, the number whose inverse hyperbolic 
c    sine is desired.
c
c    Output, double precision R8_ASINH, the inverse hyperbolic sine of X.
c
      implicit none

      double precision r8_asinh
      double precision x

      r8_asinh = log ( x + sqrt ( x * x + 1.0D+00 ) )

      return
      end
      function r8_atan ( y, x )

c*********************************************************************72
c
cc R8_ATAN computes the inverse tangent of the ratio Y / X.
c
c  Discussion:
c
c    R8_ATAN returns an angle whose tangent is ( Y / X ), a job which
c    the built in functions ATAN and ATAN2 already do.
c
c    However:
c
c    * R8_ATAN always returns a positive angle, between 0 and 2 PI,
c      while ATAN and ATAN2 return angles in the interval [-PI/2,+PI/2]
c      and [-PI,+PI] respectively;
c
c    * R8_ATAN accounts for the signs of X and Y, (as does ATAN2).  The ATAN
c     function by contrast always returns an angle in the first or fourth
c     quadrants.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    21 April 2014
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision Y, X, two quantities which represent the
c    tangent of an angle.  If Y is not zero, then the tangent is (Y/X).
c
c    Output, double precision R8_ATAN, an angle between 0 and 2 * PI, whose
c    tangent is (Y/X), and which lies in the appropriate quadrant so that
c    the signs of its cosine and sine match those of X and Y.
c
      implicit none

      double precision abs_x
      double precision abs_y
      double precision r8_atan
      double precision r8_pi
      parameter ( r8_pi = 3.141592653589793D+00 )
      double precision theta
      double precision theta_0
      double precision x
      double precision y
c
c  Special cases:
c
      if ( x .eq. 0.0D+00 ) then

        if ( 0.0D+00 .lt. y ) then
          theta = r8_pi / 2.0D+00
        else if ( y .lt. 0.0D+00 ) then
          theta = 3.0D+00 * r8_pi / 2.0D+00
        else if ( y .eq. 0.0D+00 ) then
          theta = 0.0D+00
        end if

      else if ( y .eq. 0.0D+00 ) then

        if ( 0.0D+00 .lt. x ) then
          theta = 0.0D+00
        else if ( x .lt. 0.0D+00 ) then
          theta = r8_pi
        end if
c
c  We assume that ATAN2 is correct when both arguments are positive.
c
      else

        abs_y = dabs ( y )
        abs_x = dabs ( x )

        theta_0 = atan2 ( abs_y, abs_x )

        if ( 0.0D+00 .lt. x .and. 0.0D+00 .lt. y ) then
          theta = theta_0
        else if ( x .lt. 0.0D+00 .and. 0.0D+00 .lt. y ) then
          theta = r8_pi - theta_0
        else if ( x .lt. 0.0D+00 .and. y .lt. 0.0D+00 ) then
          theta = r8_pi + theta_0
        else if ( 0.0D+00 .lt. x .and. y .lt. 0.0D+00 ) then
          theta = 2.0D+00 * r8_pi - theta_0
        end if

      end if

      r8_atan = theta

      return
      end
      function r8_atanh ( x )

c*********************************************************************72
c
cc R8_ATANH returns the inverse hyperbolic tangent of a number.
c
c  Definition:
c
c    Y = R8_ATANH ( X )
c
c    implies that
c
c    X = TANH(Y) = ( EXP(Y) - EXP(-Y) ) / ( EXP(Y) + EXP(-Y) )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    29 November 2007
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision X, the number whose inverse hyperbolic 
c    tangent is desired.  The absolute value of X should be less than 
c    or equal to 1.
c
c    Output, double precision R8_ATANH, the inverse hyperbolic tangent of X.
c
      implicit none

      double precision r8_atanh
      double precision x

      if ( 1.0D+00 .le. abs ( x ) ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8_ATANH - Fatal error!'
        write ( *, '(a)' ) '  ABS(X) must be < 1.'
        write ( *, '(a,g14.6)' ) '  Your input is X = ', x
        stop
      end if

      r8_atanh = 0.5D+00 * log ( ( 1.0D+00 + x ) / ( 1.0D+00 - x ) )

      return
      end
      function r8_big ( )

c*********************************************************************72
c
cc R8_BIG returns a "big" R8.
c
c  Discussion:
c
c    The value returned by this function is NOT required to be the
c    maximum representable R8.
c    We simply want a "very large" but non-infinite number.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    27 September 2014
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Output, double precision R8_BIG, a big number.
c
      implicit none

      double precision r8_big

      r8_big = 1.0D+30

      return
      end
      function r8_cas ( x )

c*********************************************************************72
c
cc R8_CAS returns the "casine" of an R8 value.
c
c  Discussion:
c
c    The "casine", used in the discrete Hartley transform, is abbreviated
c    CAS(X), and defined by:
c
c      CAS(X) = cos ( X ) + sin( X )
c             = sqrt ( 2 ) * sin ( X + pi/4 )
c             = sqrt ( 2 ) * cos ( X - pi/4 )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    01 January 2007
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Ralph Hartley,
c    A More Symmetrical Fourier Analysis Applied to Transmission Problems,
c    Proceedings of the Institute of Radio Engineers,
c    Volume 30, pages 144-150, 1942.
c
c  Parameters:
c
c    Input, double precision X, the number whose casine is desired.
c
c    Output, double precision R8_CAS, the casine of X, which will be between
c    plus or minus the square root of 2.
c
      implicit none

      double precision r8_cas
      double precision x

      r8_cas = cos ( x ) + sin ( x )

      return
      end
      function r8_ceiling ( r )

c*********************************************************************72
c
cc R8_CEILING rounds an R8 "up" to the nearest integral R8.
c
c  Example:
c
c     R     Value
c
c    -1.1  -1.0
c    -1.0  -1.0
c    -0.9   0.0
c     0.0   0.0
c     5.0   5.0
c     5.1   6.0
c     5.9   6.0
c     6.0   6.0
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    10 November 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision R, the value to be rounded up.
c
c    Output, double precision R8_CEILING, the rounded value.
c
      implicit none

      double precision r
      double precision r8_ceiling
      double precision value

      value = dble ( int ( r ) )
      if ( value .lt. r ) then
        value = value + 1.0D+00
      end if

      r8_ceiling = value

      return
      end
      function r8_choose ( n, k )

c*********************************************************************72
c
cc R8_CHOOSE computes the binomial coefficient C(N,K) as an R8.
c
c  Discussion:
c
c    The value is calculated in such a way as to avoid overflow and
c    roundoff.  The calculation is done in R8 arithmetic.
c
c    The formula used is:
c
c      C(N,K) = N! / ( K! * (N-K)! )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    07 June 2008
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    ML Wolfson, HV Wright,
c    Algorithm 160:
c    Combinatorial of M Things Taken N at a Time,
c    Communications of the ACM,
c    Volume 6, Number 4, April 1963, page 161.
c
c  Parameters:
c
c    Input, integer N, K, are the values of N and K.
c
c    Output, double precision R8_CHOOSE, the number of combinations of N
c    things taken K at a time.
c
      implicit none

      integer i
      integer k
      integer mn
      integer mx
      integer n
      double precision r8_choose
      double precision value

      mn = min ( k, n - k )

      if ( mn .lt. 0 ) then

        value = 0.0D+00

      else if ( mn .eq. 0 ) then

        value = 1.0D+00

      else

        mx = max ( k, n - k )
        value = dble ( mx + 1 )

        do i = 2, mn
          value = ( value * dble ( mx + i ) ) / dble ( i )
        end do

      end if

      r8_choose = value

      return
      end
      function r8_chop ( place, x )

c*********************************************************************72
c
cc R8_CHOP chops an R8 to a given number of binary places.
c
c  Example:
c
c    3.875 = 2 + 1 + 1/2 + 1/4 + 1/8.
c
c    The following values would be returned for the 'chopped' value of
c    3.875:
c
c    PLACE  Value
c
c       1      2
c       2      3     = 2 + 1
c       3      3.5   = 2 + 1 + 1/2
c       4      3.75  = 2 + 1 + 1/2 + 1/4
c       5+     3.875 = 2 + 1 + 1/2 + 1/4 + 1/8
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    16 January 2007
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer PLACE, the number of binary places to preserve.
c    PLACE = 0 means return the integer part of X.
c    PLACE = 1 means return the value of X, correct to 1/2.
c    PLACE = 2 means return the value of X, correct to 1/4.
c    PLACE = -1 means return the value of X, correct to 2.
c
c    Input, double precision X, the number to be chopped.
c
c    Output, double precision R8_CHOP, the chopped number.
c
      implicit none

      double precision fac
      integer place
      double precision r8_chop
      double precision r8_log_2
      double precision r8_sign
      double precision s
      integer temp
      double precision x

      s = r8_sign ( x )
      temp = int ( r8_log_2 ( abs ( x ) ) )
      fac = 2.0D+00 ** ( temp - place + 1 )
      r8_chop = s * dble ( int ( abs ( x ) / fac ) ) * fac

      return
      end
      function r8_cosd ( degrees )

c*********************************************************************72
c
cc R8_COSD returns the cosine of an angle given in degrees.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    27 July 2014
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision DEGREES, the angle in degrees.
c
c    Output, double precision R8_COSD, the cosine of the angle.
c
      implicit none

      double precision degrees
      double precision r8_cosd
      double precision r8_pi
      parameter ( r8_pi = 3.141592653589793D+00 )
      double precision radians

      radians = r8_pi * ( degrees / 180.0D+00 )

      r8_cosd  = cos ( radians )

      return
      end
      function r8_cot ( angle )

c*********************************************************************72
c
cc R8_COT returns the cotangent of an angle.
c
c  Discussion:
c
c    R8_COT ( THETA ) = COS ( THETA ) / SIN ( THETA )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    04 July 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision ANGLE, the angle, in radians.
c
c    Output, double precision R8_COT, the cotangent of the angle.
c
      implicit none

      double precision angle
      double precision r8_cot

      r8_cot = cos ( angle ) / sin ( angle )

      return
      end
      function r8_cotd ( degrees )

c*********************************************************************72
c
cc R8_COTD returns the cotangent of an angle given in degrees.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    27 July 2014
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision DEGREES, the angle in degrees.
c
c    Output, double precision R8_COTD, the cotangent of the angle.
c
      implicit none

      double precision degrees
      double precision r8_cotd
      double precision r8_pi
      parameter ( r8_pi = 3.141592653589793D+00 )
      double precision radians

      radians = r8_pi * ( degrees / 180.0D+00 )

      r8_cotd  = cos ( radians ) / sin ( radians )

      return
      end
      function r8_csc ( theta )

c*********************************************************************72
c
cc R8_CSC returns the cosecant of X.
c
c  Discussion:
c
c    R8_CSC ( THETA ) = 1.0 / SIN ( THETA )
c
c    The cosecant is not a built-in function in FORTRAN, and occasionally it
c    is handier, or more concise, to be able to refer to it directly
c    rather than through its definition in terms of the sine function.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    05 March 2012
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision THETA, the angle, in radians, whose
c    cosecant is desired.  It must be the case that SIN ( THETA ) is not zero.
c
c    Output, double precision R8_CSC, the cosecant of THETA.
c
      implicit none

      double precision r8_csc
      double precision theta
      double precision value

      value = sin ( theta )

      if ( value .eq. 0.0D+00 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8_CSC - Fatal error!'
        write ( *, '(a,g14.6)' ) 
     &    '  Cosecant undefined for THETA = ', theta
        stop 1
      end if

      r8_csc = 1.0D+00 / value

      return
      end
      function r8_cscd ( degrees )

c*********************************************************************72
c
cc R8_CSCD returns the cosecant of an angle given in degrees.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    27 July 2014
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision DEGREES, the angle in degrees.
c
c    Output, double precision R8_CSCD, the cosecant of the angle.
c
      implicit none

      double precision degrees
      double precision r8_cscd
      double precision r8_pi 
      parameter ( r8_pi = 3.141592653589793D+00 )
      double precision radians

      radians = r8_pi * ( degrees / 180.0D+00 )

      r8_cscd  = 1.0D+00 / sin ( radians )

      return
      end
      function r8_csqrt ( x )

c*********************************************************************72
c
cc R8_CSQRT returns the complex square root of an R8.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    23 August 20008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision X, the number whose square root is desired.
c
c    Output, double complex R8_CSQRT, the square root of X:
c
      implicit none

      double precision argument
      double precision magnitude
      double complex r8_csqrt
      double precision r8_pi
      parameter ( r8_pi = 3.141592653589793D+00 )
      double precision x

      if ( 0.0D+00 .lt. x ) then
        magnitude = x
        argument = 0.0D+00
      else if ( 0.0D+00 .eq. x ) then
        magnitude = 0.0D+00
        argument = 0.0D+00
      else if ( x .lt. 0.0D+00 ) then
        magnitude = -x
        argument = r8_pi
      end if

      magnitude = sqrt ( magnitude )
      argument = argument / 2.0D+00

      r8_csqrt = magnitude
     &  * dcmplx ( cos ( argument ), sin ( argument ) )

      return
      end
      function r8_cube_root ( x )

c*********************************************************************72
c
cc R8_CUBE_ROOT returns the cube root of an R8.
c
c  Discussion:
c
c    This routine is designed to avoid the possible problems that can occur
c    when formulas like 0.0**(1/3) or (-1.0) ** (1/3) are to be evaluated.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    16 January 2007
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision X, the number whose cube root is desired.
c
c    Output, double precision R8_CUBE_ROOT, the cube root of X.
c
      implicit none

      double precision r8_cube_root
      double precision value
      double precision x

      if ( 0.0D+00 .lt. x ) then
        value = x**(1.0D+00/3.0D+00)
      else if ( x .eq. 0.0D+00 ) then
        value = 0.0D+00
      else
        value = - ( abs ( x ) )**(1.0D+00/3.0D+00)
      end if

      r8_cube_root = value

      return
      end
      function r8_degrees ( radians )

c*********************************************************************72
c
cc R8_DEGREES converts an angle from radian to degree measure.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    15 May 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision RADIANS, the angle measurement in radians.
c
c    Output, double precision R8_DEGREES, the angle measurement in degrees.
c
      implicit none

      real ( kind = 8 ) r8_degrees
      real ( kind = 8 ) r8_pi
      parameter ( r8_pi = 3.1415926535897932384626434D+00 )
      real ( kind = 8 ) radians

      r8_degrees = radians * 180.0D+00 / r8_pi

      return
      end
      function r8_diff ( x, y, n )

c*********************************************************************72
c
cc R8_DIFF computes the difference of two R8's to a specified accuracy.
c
c  Discussion:
c
c    The user controls how many binary digits of accuracy
c    are to be used.
c
c    N determines the accuracy of the value of the result.  If N = 10,
c    for example, only 11 binary places will be used in the arithmetic.
c    In general, only N+1 binary places will be used.
c
c    N may be zero.  However, a negative value of N should
c    not be used, since this will cause both X and Y to look like 0.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    04 April 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision X, Y, the two values whose difference is desired.
c
c    Input, integer N, the number of binary digits to use.
c
c    Output, double precision R8_DIFF, the value of X-Y.
c
      implicit none

      double precision cx
      double precision cy
      integer n
      double precision pow2
      double precision r8_diff
      double precision size
      double precision x
      double precision y

      if ( x .eq. y ) then
        r8_diff = 0.0D+00
        return
      end if

      pow2 = 2.0D+00**n
c
c  Compute the magnitude of X and Y, and take the larger of the
c  two.  At least one of the two values is not zero!
c
      size = max ( abs ( x ), abs ( y ) )
c
c  Make normalized copies of X and Y.  One of the two values will
c  actually be equal to 1.
c
      cx = x / size
      cy = y / size
c
c  Here's where rounding comes in.  We know that the larger of the
c  the two values equals 1.  We multiply both values by 2**N,
c  where N+1 is the number of binary digits of accuracy we want
c  to use, truncate the values, and divide back by 2**N.
c
      cx = dble ( int ( cx * pow2 + sign ( 0.5D+00, cx ) ) ) / pow2
      cy = dble ( int ( cy * pow2 + sign ( 0.5D+00, cy ) ) ) / pow2
c
c  Take the difference now.
c
      r8_diff = cx - cy
c
c  Undo the scaling.
c
      r8_diff = r8_diff * size

      return
      end
      subroutine r8_digit ( x, idigit, digit )

c*********************************************************************72
c
cc R8_DIGIT returns a particular decimal digit of an R8.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    04 April 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, real double precision X, the number whose NDIG-th decimal digit
c    is desired.  If X is zero, all digits will be returned as 0.
c
c    Input, integer IDIGIT, the position of the desired decimal
c    digit.  A value of 1 means the leading digit, a value of 2 the second digit
c    and so on.
c
c    Output, integer DIGIT, the value of the IDIGIT-th decimal
c    digit of X.
c
      implicit none

      integer digit
      integer i
      integer idigit
      integer ival
      double precision x
      double precision xcopy

      if ( x .eq. 0.0D+00 ) then
        digit = 0
        return
      end if

      if ( idigit .le. 0 ) then
        digit = 0
        return
      end if
c
c  Set XCOPY = X, and then force XCOPY to lie between 1 and 10.
c
      xcopy = abs ( x )

10    continue

      if ( xcopy .lt. 1.0D+00 ) then
        xcopy = xcopy * 10.0D+00
        go to 10
      end if

20    continue

      if ( 10.0D+00 .le. xcopy ) then
        xcopy = xcopy / 10.0D+00
        go to 20
      end if

      do i = 1, idigit
        ival = int ( xcopy )
        xcopy = ( xcopy - ival ) * 10.0D+00
      end do

      digit = ival

      return
      end
      function r8_divide_i4 ( i, j )

c*********************************************************************72
c
cc R8_DIVIDE_I4 returns an I4 fraction as an R8.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    05 June 2012
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer I, J, the numerator and denominator.
c
c    Output, double precision R8_DIVIDE_I4, the value of (I/J).
c
      implicit none

      integer i
      integer j
      double precision r8_divide_i4

      r8_divide_i4 = dble ( i ) / dble ( j )

      return
      end
      function r8_e ( )

c*********************************************************************72
c
cc R8_E returns the value of the base of the natural logarithm system.
c
c  Discussion:
c
c    E = Limit ( N -> +oo ) ( 1 + 1 / N )^N
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    03 July 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Output, double precision R8_E, the base of the natural 
c    logarithm system.
c
      implicit none

      double precision r8_e

      r8_e = 2.718281828459045235360287D+00
 
      return
      end
      function r8_epsilon ( )

c*********************************************************************72
c
cc R8_EPSILON returns the R8 roundoff unit.
c
c  Discussion:
c
c    The roundoff unit is a number R which is a power of 2 with the
c    property that, to the precision of the computer's arithmetic,
c      1 .lt. 1 + R
c    but
c      1 = ( 1 + R / 2 )
c
c    FORTRAN90 provides the superior library routine
c
c      EPSILON ( X )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    01 September 2012
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Output, double precision R8_EPSILON, the R8 roundoff unit.
c
      implicit none

      double precision r8_epsilon

      r8_epsilon = 2.220446049250313D-016

      return
      end
      function r8_epsilon_compute ( )

c*********************************************************************72
c
cc R8_EPSILON_COMPUTE computes the R8 roundoff unit.
c
c  Discussion:
c
c    The roundoff unit is a number R which is a power of 2 with the
c    property that, to the precision of the computer's arithmetic,
c      1 .lt. 1 + R
c    but
c      1 = ( 1 + R / 2 )
c
c    FORTRAN90 provides the superior library routine
c
c      EPSILON ( X )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    01 September 2012
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Output, double precision R8_EPSILON_COMPUTE, the R8 roundoff unit.
c
      implicit none

      double precision one
      double precision r8_add
      double precision r8_epsilon_compute
      double precision temp
      double precision test
      double precision value

      save value

      data value / 0.0D+00 /

      if ( value .ne. 0.0D+00 ) then
        r8_epsilon_compute = value
        return
      end if

      one = dble ( 1 )

      value = one
      temp = value / 2.0D+00
      test = r8_add ( one, temp )

10    continue

      if ( one .lt. test ) then
        value = temp
        temp = value / 2.0D+00
        test = r8_add ( one, temp )
        go to 10
      end if

      r8_epsilon_compute = value

      return
      end
      function r8_exp ( x )

c*********************************************************************72
c
cc R8_EXP computes the exponential function, avoiding overflow and underflow.
c
c  Discussion:
c
c    For arguments of very large magnitude, the evaluation of the
c    exponential function can cause computational problems.  Some languages
c    and compilers may return an infinite value or a "Not-a-Number".  
c    An alternative, when dealing with a wide range of inputs, is simply
c    to truncate the calculation for arguments whose magnitude is too large.
c    Whether this is the right or convenient approach depends on the problem
c    you are dealing with, and whether or not you really need accurate
c    results for large magnitude inputs, or you just want your code to
c    stop crashing.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    19 September 2014
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision X, the argument of the exponential function.
c
c    Output, double precision R8_EXP, the value of exp ( X ).
c
      implicit none

      double precision r8_big
      parameter ( r8_big = 1.0D+30 )
      double precision r8_log_max
      parameter ( r8_log_max = +69.0776D+00 )
      double precision r8_log_min
      parameter ( r8_log_min = -69.0776D+00 )
      double precision r8_exp
      double precision x

      if ( x .le. r8_log_min ) then
        r8_exp = 0.0D+00
      else if ( x .lt. r8_log_max ) then
        r8_exp = exp ( x )
      else
        r8_exp = r8_big
      end if

      return
      end
      function r8_factorial ( n )

c*********************************************************************72
c
cc R8_FACTORIAL computes the factorial of N.
c
c  Discussion:
c
c    factorial ( N ) = product ( 1 <= I <= N ) I
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    07 June 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the argument of the factorial function.
c    If N is less than 1, the function value is returned as 1.
c
c    Output, double precision R8_FACTORIAL, the factorial of N.
c
      implicit none

      integer i
      integer n
      double precision r8_factorial

      r8_factorial = 1.0D+00

      do i = 1, n
        r8_factorial = r8_factorial * dble ( i )
      end do

      return
      end
      subroutine r8_factorial_values ( n_data, n, fn )

c*********************************************************************72
c
cc R8_FACTORIAL_VALUES returns values of the real factorial function.
c
c  Discussion:
c
c    0! = 1
c    I! = Product ( 1 <= J <= I ) J
c
c    Although the factorial is an integer valued function, it quickly
c    becomes too large for an integer to hold.  This routine still accepts
c    an integer as the input argument, but returns the function value
c    as a real number.
c
c    In Mathematica, the function can be evaluated by:
c
c      n!
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    25 March 2007
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Milton Abramowitz, Irene Stegun,
c    Handbook of Mathematical Functions,
c    National Bureau of Standards, 1964,
c    ISBN: 0-486-61272-4,
c    LC: QA47.A34.
c
c    Stephen Wolfram,
c    The Mathematica Book,
c    Fourth Edition,
c    Cambridge University Press, 1999,
c    ISBN: 0-521-64314-7,
c    LC: QA76.95.W65.
c
c  Parameters:
c
c    Input/output, integer N_DATA.  The user sets N_DATA to 0 before the
c    first call.  On each call, the routine increments N_DATA by 1, and
c    returns the corresponding data; when there is no more data, the
c    output value of N_DATA will be 0 again.
c
c    Output, integer N, the argument of the function.
c
c    Output, double precision FN, the value of the function.
c
      implicit none

      integer n_max
      parameter ( n_max = 25 )

      double precision fn
      double precision fn_vec(n_max)
      integer n
      integer n_data
      integer n_vec(n_max)

      save fn_vec
      save n_vec

      data fn_vec /
     &  0.1000000000000000D+01,
     &  0.1000000000000000D+01,
     &  0.2000000000000000D+01,
     &  0.6000000000000000D+01,
     &  0.2400000000000000D+02,
     &  0.1200000000000000D+03,
     &  0.7200000000000000D+03,
     &  0.5040000000000000D+04,
     &  0.4032000000000000D+05,
     &  0.3628800000000000D+06,
     &  0.3628800000000000D+07,
     &  0.3991680000000000D+08,
     &  0.4790016000000000D+09,
     &  0.6227020800000000D+10,
     &  0.8717829120000000D+11,
     &  0.1307674368000000D+13,
     &  0.2092278988800000D+14,
     &  0.3556874280960000D+15,
     &  0.6402373705728000D+16,
     &  0.1216451004088320D+18,
     &  0.2432902008176640D+19,
     &  0.1551121004333099D+26,
     &  0.3041409320171338D+65,
     &  0.9332621544394415D+158,
     &  0.5713383956445855D+263 /
      data n_vec /
     &     0,
     &     1,
     &     2,
     &     3,
     &     4,
     &     5,
     &     6,
     &     7,
     &     8,
     &     9,
     &    10,
     &    11,
     &    12,
     &    13,
     &    14,
     &    15,
     &    16,
     &    17,
     &    18,
     &    19,
     &    20,
     &    25,
     &    50,
     &   100,
     &   150 /

      if ( n_data .lt. 0 ) then
        n_data = 0
      end if

      n_data = n_data + 1

      if ( n_max .lt. n_data ) then
        n_data = 0
        n = 0
        fn = 0.0D+00
      else
        n = n_vec(n_data)
        fn = fn_vec(n_data)
      end if

      return
      end
      function r8_factorial2 ( n )

c*********************************************************************72
c
cc R8_FACTORIAL2 computes the double factorial function.
c
c  Discussion:
c
c    FACTORIAL2( N ) = Product ( N * (N-2) * (N-4) * ... * 2 )  (N even)
c                    = Product ( N * (N-2) * (N-4) * ... * 1 )  (N odd)
c
c  Example:
c
c     N Value
c
c     0     1
c     1     1
c     2     2
c     3     3
c     4     8
c     5    15
c     6    48
c     7   105
c     8   384
c     9   945
c    10  3840
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    02 June 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the argument of the double factorial
c    function.  If N is less than 1, R8_FACTORIAL2 is returned as 1.0.
c
c    Output, double precision R8_FACTORIAL2, the value.
c
      implicit none

      integer n
      double precision r8_factorial2
      double precision r8_n

      if ( n .lt. 1 ) then
        r8_factorial2 = 1.0D+00
        return
      end if

      r8_n = dble ( n )
      r8_factorial2 = 1.0D+00

10    continue

      if ( 1.0D+00 .lt. r8_n ) then
        r8_factorial2 = r8_factorial2 * r8_n
        r8_n = r8_n - 2.0D+00
        go to 10
      end if

      return
      end
      subroutine r8_factorial2_values ( n_data, n, f )

c*********************************************************************72
c
cc R8_FACTORIAL2_VALUES returns values of the double factorial function.
c
c  Discussion:
c
c    FACTORIAL2( N ) = Product ( N * (N-2) * (N-4) * ... * 2 )  (N even)
c                    = Product ( N * (N-2) * (N-4) * ... * 1 )  (N odd)
c
c    In Mathematica, the function can be evaluated by:
c
c      n!!
c
c  Example:
c
c     N    N!!
c
c     0     1
c     1     1
c     2     2
c     3     3
c     4     8
c     5    15
c     6    48
c     7   105
c     8   384
c     9   945
c    10  3840
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    07 February 2015
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Milton Abramowitz, Irene Stegun,
c    Handbook of Mathematical Functions,
c    National Bureau of Standards, 1964,
c    ISBN: 0-486-61272-4,
c    LC: QA47.A34.
c
c    Stephen Wolfram,
c    The Mathematica Book,
c    Fourth Edition,
c    Cambridge University Press, 1999,
c    ISBN: 0-521-64314-7,
c    LC: QA76.95.W65.
c
c    Daniel Zwillinger, editor,
c    CRC Standard Mathematical Tables and Formulae,
c    30th Edition,
c    CRC Press, 1996,
c    ISBN: 0-8493-2479-3,
c    LC: QA47.M315.
c
c  Parameters:
c
c    Input/output, integer N_DATA.  The user sets N_DATA to 0 before the
c    first call.  On each call, the routine increments N_DATA by 1, and
c    returns the corresponding data; when there is no more data, the
c    output value of N_DATA will be 0 again.
c
c    Output, integer N, the argument of the function.
c
c    Output, double precision F, the value of the function.
c
      implicit none

      integer n_max
      parameter ( n_max = 16 )

      double precision f_vec(n_max)
      double precision f
      integer n_data
      integer n
      integer n_vec(n_max)

      save f_vec
      save n_vec

      data f_vec /
     &        1.0D+00,
     &        1.0D+00,
     &        2.0D+00,
     &        3.0D+00,
     &        8.0D+00,
     &       15.0D+00,
     &       48.0D+00,
     &      105.0D+00,
     &      384.0D+00,
     &      945.0D+00,
     &     3840.0D+00,
     &    10395.0D+00,
     &    46080.0D+00,
     &   135135.0D+00,
     &   645120.0D+00,
     &  2027025.0D+00 /
      data n_vec /
     &   0,
     &   1,  2,  3,  4,  5,
     &   6,  7,  8,  9, 10,
     &  11, 12, 13, 14, 15 /

      if ( n_data .lt. 0 ) then
        n_data = 0
      end if

      n_data = n_data + 1

      if ( n_max .lt. n_data ) then
        n_data = 0
        n = 0
        f = 0.0D+00
      else
        n = n_vec(n_data)
        f = f_vec(n_data)
      end if

      return
      end
      function r8_fall ( x, n )

c*********************************************************************72
c
cc R8_FALL computes the falling factorial function [X]_N.
c
c  Discussion:
c
c    Note that the number of "injections" or 1-to-1 mappings from
c    a set of N elements to a set of M elements is [M]_N.
c
c    The number of permutations of N objects out of M is [M]_N.
c
c    Moreover, the Stirling numbers of the first kind can be used
c    to convert a falling factorial into a polynomial, as follows:
c
c      [X]_N = S^0_N + S^1_N * X + S^2_N * X^2 + ... + S^N_N X^N.
c
c    The formula used is:
c
c      [X]_N = X * ( X - 1 ) * ( X - 2 ) * ... * ( X - N + 1 ).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    03 January 2007
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision X, the argument of the falling factorial function.
c
c    Input, integer N, the order of the falling factorial function.
c    If N = 0, FALL = 1, if N = 1, FALL = X.  Note that if N is
c    negative, a "rising" factorial will be computed.
c
c    Output, double precision R8_FALL, the value of the falling 
c    factorial function.
c
      implicit none

      double precision arg
      integer i
      integer n
      double precision r8_fall
      double precision value
      double precision x

      value = 1.0D+00

      arg = x

      if ( 0 .lt. n ) then

        do i = 1, n
          value = value * arg
          arg = arg - 1.0D+00
        end do

      else if ( n .lt. 0 ) then

        do i = -1, n, -1
          value = value * arg
          arg = arg + 1.0D+00
        end do

      end if

      r8_fall = value

      return
      end
      subroutine r8_fall_values ( n_data, x, n, f )

c*********************************************************************72
c
cc R8_FALL_VALUES returns some values of the falling factorial function.
c
c  Discussion:
c
c    In Mathematica, the function can be evaluated by:
c
c      FactorialPower[X,Y]
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    20 December 2014
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Milton Abramowitz, Irene Stegun,
c    Handbook of Mathematical Functions,
c    National Bureau of Standards, 1964,
c    ISBN: 0-486-61272-4,
c    LC: QA47.A34.
c
c    Stephen Wolfram,
c    The Mathematica Book,
c    Fourth Edition,
c    Cambridge University Press, 1999,
c    ISBN: 0-521-64314-7,
c    LC: QA76.95.W65.
c
c  Parameters:
c
c    Input/output, integer N_DATA.  The user sets N_DATA to 0
c    before the first call.  On each call, the routine increments N_DATA by 1,
c    and returns the corresponding data; when there is no more data, the
c    output value of N_DATA will be 0 again.
c
c    Output, double precision X, integer N, the arguments of the function.
c
c    Output, double precision F, the value of the function.
c
      implicit none

      integer n_max
      parameter ( n_max = 15 )

      double precision f_vec(n_max)
      double precision f
      integer n
      integer n_data
      integer n_vec(n_max)
      double precision x
      double precision x_vec(n_max)

      save f_vec
      save n_vec
      save x_vec

      data f_vec /
     &  120.0000000000000D+00,
     &   163.1601562500000D+00,
     &   216.5625000000000D+00,
     &   281.6601562500000D+00,
     &   360.0000000000000D+00,
     &   1.000000000000000D+00,
     &   7.500000000000000D+00,
     &   48.75000000000000D+00,
     &   268.1250000000000D+00,
     &   1206.562500000000D+00,
     &   4222.968750000000D+00,
     &   10557.42187500000D+00,
     &   15836.13281250000D+00,
     &   7918.066406250000D+00,
     &   -3959.03320312500D+00 /

      data n_vec /
     &   4,
     &   4,
     &   4,
     &   4,
     &   4,
     &   0,
     &   1,
     &   2,
     &   3,
     &   4,
     &   5,
     &   6,
     &   7,
     &   8,
     &   9 /

      data x_vec /
     &   5.00D+00,
     &   5.25D+00,
     &   5.50D+00,
     &   5.75D+00,
     &   6.00D+00,
     &   7.50D+00,
     &   7.50D+00,
     &   7.50D+00,
     &   7.50D+00,
     &   7.50D+00,
     &   7.50D+00,
     &   7.50D+00,
     &   7.50D+00,
     &   7.50D+00,
     &   7.50D+00 /

      if ( n_data .lt. 0 ) then
        n_data = 0
      end if

      n_data = n_data + 1

      if ( n_max .lt. n_data ) then
        n_data = 0
        x = 0.0D+00
        n = 0
        f = 0.0D+00
      else
        x = x_vec(n_data)
        n = n_vec(n_data)
        f = f_vec(n_data)
      end if

      return
      end
      function r8_floor ( r )

c*********************************************************************72
c
cc R8_FLOOR rounds an R8 "down" (towards -infinity) to the nearest integral R8.
c
c  Example:
c
c    R     Value
c
c   -1.1  -2.0
c   -1.0  -1.0
c   -0.9  -1.0
c    0.0   0.0
c    5.0   5.0
c    5.1   5.0
c    5.9   5.0
c    6.0   6.0
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    10 Noember 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision R, the value to be rounded down.
c
c    Output, double precision R8_FLOOR, the rounded value.
c
      implicit none

      double precision r
      double precision r8_floor
      double precision value

      value = dble ( int ( r ) )
      if ( r .lt. value ) then
        value = value - 1.0D+00
      end if

      r8_floor = value

      return
      end
      function r8_fraction ( i, j )

c*********************************************************************72
c
cc R8_FRACTION uses real arithmetic on an integer ratio.
c
c  Discussion:
c
c    Given integer variables I and J, both FORTRAN and C will evaluate
c    an expression such as "I/J" using what is called "integer division",
c    with the result being an integer.  It is often convenient to express
c    the parts of a fraction as integers but expect the result to be computed
c    using real arithmetic.  This function carries out that operation.
c
c  Example:
c
c       I     J   I/J  R8_FRACTION
c
c       1     2     0  0.5
c       7     4     1  1.75
c       8     4     2  2.00
c       9     4     2  2.25
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    05 October 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer I, J, the arguments.
c
c    Output, double precision R8_FRACTION, the value of the ratio.
c
      implicit none

      integer i
      integer j
      double precision r8_fraction

      r8_fraction = dble ( i ) / dble ( j )

      return
      end
      function r8_fractional ( x )

c*********************************************************************72
c
cc R8_FRACTIONAL returns the fractional part of an R8.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    16 January 2007
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision X, the argument.
c
c    Output, double precision R8_FRACTIONAL, the fractional part of X.
c
      implicit none

      double precision r8_fractional
      double precision x

      r8_fractional = abs ( x ) - dble ( int ( abs ( x ) ) )

      return
      end
      function r8_gamma ( x )

c*********************************************************************72
c
cc R8_GAMMA evaluates Gamma(X) for a real argument.
c
c  Discussion:
c
c    This routine calculates the gamma function for a real argument X.
c    Computation is based on an algorithm outlined in reference 1.
c    The program uses rational functions that approximate the gamma
c    function to at least 20 significant decimal digits.  Coefficients
c    for the approximation over the interval (1,2) are unpublished.
c    Those for the approximation for 12 <= X are from reference 2.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    18 January 2008
c
c  Author:
c
c    Original FORTRAN77 version by William Cody, Laura Stoltz.
c    This FORTRAN77 version by John Burkardt.
c
c  Reference:
c
c    William Cody,
c    An Overview of Software Development for Special Functions,
c    in Numerical Analysis Dundee, 1975,
c    edited by GA Watson,
c    Lecture Notes in Mathematics 506,
c    Springer, 1976.
c
c    John Hart, Ward Cheney, Charles Lawson, Hans Maehly, 
c    Charles Mesztenyi, John Rice, Henry Thatcher, 
c    Christoph Witzgall,
c    Computer Approximations,
c    Wiley, 1968,
c    LC: QA297.C64.
c
c  Parameters:
c
c    Input, double precision X, the argument of the function.
c
c    Output, double precision R8_GAMMA, the value of the function.
c
      implicit none

      double precision c(7)
      double precision eps
      double precision fact
      integer i
      integer n
      double precision p(8)
      logical parity
      double precision q(8)
      double precision r8_gamma
      double precision r8_pi
      parameter ( r8_pi = 3.141592653589793D+00 )
      double precision res
      double precision sqrtpi
      double precision sum
      double precision x
      double precision xbig
      double precision xden
      double precision xinf
      double precision xminin
      double precision xnum
      double precision y
      double precision y1
      double precision ysq
      double precision z
c
c  Mathematical constants
c
      data sqrtpi / 0.9189385332046727417803297D+00 /
c
c  Machine dependent parameters
c
      data xbig / 171.624D+00 /
      data xminin / 2.23D-308 /
      data eps / 2.22D-16 /
      data xinf /1.79D+308 /
c
c  Numerator and denominator coefficients for rational minimax
c  approximation over (1,2).
c
      data p /
     & -1.71618513886549492533811d+00,
     &  2.47656508055759199108314d+01,
     & -3.79804256470945635097577d+02,
     &  6.29331155312818442661052d+02,
     &  8.66966202790413211295064d+02,
     & -3.14512729688483675254357d+04,
     & -3.61444134186911729807069d+04,
     &  6.64561438202405440627855d+04 /

      data q /
     & -3.08402300119738975254353D+01,
     &  3.15350626979604161529144D+02,
     & -1.01515636749021914166146D+03,
     & -3.10777167157231109440444D+03,
     &  2.25381184209801510330112D+04,
     &  4.75584627752788110767815D+03,
     & -1.34659959864969306392456D+05,
     & -1.15132259675553483497211D+05 /
c
c  Coefficients for minimax approximation over (12, INF).
c
      data c /
     & -1.910444077728D-03,
     &  8.4171387781295D-04,
     & -5.952379913043012D-04,
     &  7.93650793500350248D-04,
     & -2.777777777777681622553D-03,
     &  8.333333333333333331554247D-02,
     &  5.7083835261D-03 /

      parity = .false.
      fact = 1.0D+00
      n = 0
      y = x
c
c  Argument is negative.
c
      if ( y .le. 0.0D+00 ) then

        y = - x
        y1 = aint ( y )
        res = y - y1

        if ( res .ne. 0.0D+00 ) then

          if ( y1 .ne. aint ( y1 * 0.5D+00 ) * 2.0D+00 ) then
            parity = .true.
          end if

          fact = - r8_pi / sin ( r8_pi * res )
          y = y + 1.0D+00

        else

          res = xinf
          r8_gamma = res
          return

        end if

      end if
c
c  Argument is positive.
c
      if ( y .lt. eps ) then
c
c  Argument < EPS.
c
        if ( xminin .le. y ) then
          res = 1.0D+00 / y
        else
          res = xinf
          r8_gamma = res
          return
        end if

      else if ( y .lt. 12.0D+00 ) then

        y1 = y
c
c  0.0 < argument < 1.0.
c
        if ( y .lt. 1.0D+00 ) then

          z = y
          y = y + 1.0D+00
c
c  1.0 < argument < 12.0.
c  Reduce argument if necessary.
c
        else

          n = int ( y ) - 1
          y = y - dble ( n )
          z = y - 1.0D+00

        end if
c
c  Evaluate approximation for 1.0 < argument < 2.0.
c
        xnum = 0.0D+00
        xden = 1.0D+00
        do i = 1, 8
          xnum = ( xnum + p(i) ) * z
          xden = xden * z + q(i)
        end do

        res = xnum / xden + 1.0D+00
c
c  Adjust result for case  0.0 < argument < 1.0.
c
        if ( y1 .lt. y ) then

          res = res / y1
c
c  Adjust result for case 2.0 < argument < 12.0.
c
        else if ( y .lt. y1 ) then

          do i = 1, n
            res = res * y
            y = y + 1.0D+00
          end do

        end if

      else
c
c  Evaluate for 12.0 <= argument.
c
        if ( y .le. xbig ) then

          ysq = y * y
          sum = c(7)
          do i = 1, 6
            sum = sum / ysq + c(i)
          end do
          sum = sum / y - y + sqrtpi
          sum = sum + ( y - 0.5D+00 ) * log ( y )
          res = exp ( sum )

        else

          res = xinf
          r8_gamma = res
          return

        end if

      end if
c
c  Final adjustments and return.
c
      if ( parity ) then
        res = - res
      end if

      if ( fact .ne. 1.0D+00 ) then
        res = fact / res
      end if

      r8_gamma = res

      return
      end
      function r8_gamma_log ( x )

c*********************************************************************72
c
cc R8_GAMMA_LOG evaluates the logarithm of the gamma function.
c
c  Discussion:
c
c    This routine calculates the LOG(GAMMA) function for a positive real
c    argument X.  Computation is based on an algorithm outlined in
c    references 1 and 2.  The program uses rational functions that
c    theoretically approximate LOG(GAMMA) to at least 18 significant
c    decimal digits.  The approximation for X > 12 is from reference
c    3, while approximations for X < 12.0 are similar to those in
c    reference 1, but are unpublished.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    18 April 2013
c
c  Author:
c
c    Original FORTRAN77 version by William Cody, Laura Stoltz.
c    This FORTRAN77 version by John Burkardt.
c
c  Reference:
c
c    William Cody, Kenneth Hillstrom,
c    Chebyshev Approximations for the Natural Logarithm of the
c    Gamma Function,
c    Mathematics of Computation,
c    Volume 21, Number 98, April 1967, pages 198-203.
c
c    Kenneth Hillstrom,
c    ANL/AMD Program ANLC366S, DGAMMA/DLGAMA,
c    May 1969.
c
c    John Hart, Ward Cheney, Charles Lawson, Hans Maehly,
c    Charles Mesztenyi, John Rice, Henry Thatcher,
c    Christoph Witzgall,
c    Computer Approximations,
c    Wiley, 1968,
c    LC: QA297.C64.
c
c  Parameters:
c
c    Input, double precision X, the argument of the function.
c
c    Output, double precision R8_GAMMA_LOG, the value of the function.
c
      implicit none

      double precision c(7)
      double precision corr
      double precision d1 
      parameter ( d1 = -5.772156649015328605195174D-01 )
      double precision d2
      parameter ( d2 = 4.227843350984671393993777D-01 )
      double precision d4
      parameter ( d4 = 1.791759469228055000094023D+00 )
      double precision frtbig
      parameter ( frtbig = 2.25D+76 )
      integer i
      double precision p1(8)
      double precision p2(8)
      double precision p4(8)
      double precision q1(8)
      double precision q2(8)
      double precision q4(8)
      double precision r8_epsilon
      double precision r8_gamma_log
      double precision res
      double precision sqrtpi 
      parameter ( sqrtpi = 0.9189385332046727417803297D+00 )
      double precision x
      double precision xbig
      parameter ( xbig = 2.55D+305 )
      double precision xden
      double precision xinf
      parameter ( xinf = 1.79D+308 )
      double precision xm1
      double precision xm2
      double precision xm4
      double precision xnum
      double precision y
      double precision ysq

      save c
      save p1
      save p2
      save p4
      save q1
      save q2
      save q4

      data c /
     &  -1.910444077728D-03, 
     &   8.4171387781295D-04, 
     &  -5.952379913043012D-04, 
     &   7.93650793500350248D-04, 
     &  -2.777777777777681622553D-03, 
     &   8.333333333333333331554247D-02, 
     &   5.7083835261D-03 /
      data p1 /
     &  4.945235359296727046734888D+00, 
     &  2.018112620856775083915565D+02, 
     &  2.290838373831346393026739D+03, 
     &  1.131967205903380828685045D+04, 
     &  2.855724635671635335736389D+04, 
     &  3.848496228443793359990269D+04, 
     &  2.637748787624195437963534D+04, 
     &  7.225813979700288197698961D+03 /
      data p2 /
     &  4.974607845568932035012064D+00, 
     &  5.424138599891070494101986D+02, 
     &  1.550693864978364947665077D+04, 
     &  1.847932904445632425417223D+05, 
     &  1.088204769468828767498470D+06, 
     &  3.338152967987029735917223D+06, 
     &  5.106661678927352456275255D+06, 
     &  3.074109054850539556250927D+06 /
      data p4 /
     &  1.474502166059939948905062D+04, 
     &  2.426813369486704502836312D+06, 
     &  1.214755574045093227939592D+08, 
     &  2.663432449630976949898078D+09, 
     &  2.940378956634553899906876D+10, 
     &  1.702665737765398868392998D+11, 
     &  4.926125793377430887588120D+11, 
     &  5.606251856223951465078242D+11 /
      data q1 /
     &  6.748212550303777196073036D+01, 
     &  1.113332393857199323513008D+03, 
     &  7.738757056935398733233834D+03, 
     &  2.763987074403340708898585D+04, 
     &  5.499310206226157329794414D+04, 
     &  6.161122180066002127833352D+04, 
     &  3.635127591501940507276287D+04, 
     &  8.785536302431013170870835D+03 /
      data q2 /
     &  1.830328399370592604055942D+02, 
     &  7.765049321445005871323047D+03, 
     &  1.331903827966074194402448D+05, 
     &  1.136705821321969608938755D+06, 
     &  5.267964117437946917577538D+06, 
     &  1.346701454311101692290052D+07, 
     &  1.782736530353274213975932D+07, 
     &  9.533095591844353613395747D+06 /
      data q4 /
     &  2.690530175870899333379843D+03, 
     &  6.393885654300092398984238D+05, 
     &  4.135599930241388052042842D+07, 
     &  1.120872109616147941376570D+09, 
     &  1.488613728678813811542398D+10, 
     &  1.016803586272438228077304D+11, 
     &  3.417476345507377132798597D+11, 
     &  4.463158187419713286462081D+11 /

      y = x

      if ( 0.0D+00 .lt. y .and. y .le. xbig ) then

        if ( y .le. r8_epsilon ( ) ) then

          res = - log ( y )
c
c  EPS < X <= 1.5.
c
        else if ( y .le. 1.5D+00 ) then

          if ( y .lt. 0.6796875D+00 ) then
            corr = -log ( y )
            xm1 = y
          else
            corr = 0.0D+00
            xm1 = ( y - 0.5D+00 ) - 0.5D+00
          end if

          if ( y .le. 0.5D+00 .or. 0.6796875D+00 .le. y ) then

            xden = 1.0D+00
            xnum = 0.0D+00
            do i = 1, 8
              xnum = xnum * xm1 + p1(i)
              xden = xden * xm1 + q1(i)
            end do

            res = corr + ( xm1 * ( d1 + xm1 * ( xnum / xden ) ) )

          else

            xm2 = ( y - 0.5D+00 ) - 0.5D+00
            xden = 1.0D+00
            xnum = 0.0D+00
            do i = 1, 8
              xnum = xnum * xm2 + p2(i)
              xden = xden * xm2 + q2(i)
            end do

            res = corr + xm2 * ( d2 + xm2 * ( xnum / xden ) )

          end if
c
c  1.5 < X <= 4.0.
c
        else if ( y .le. 4.0D+00 ) then

          xm2 = y - 2.0D+00
          xden = 1.0D+00
          xnum = 0.0D+00
          do i = 1, 8
            xnum = xnum * xm2 + p2(i)
            xden = xden * xm2 + q2(i)
          end do

          res = xm2 * ( d2 + xm2 * ( xnum / xden ) )
c
c  4.0 < X <= 12.0.
c
        else if ( y .le. 12.0D+00 ) then

          xm4 = y - 4.0D+00
          xden = -1.0D+00
          xnum = 0.0D+00
          do i = 1, 8
            xnum = xnum * xm4 + p4(i)
            xden = xden * xm4 + q4(i)
          end do

          res = d4 + xm4 * ( xnum / xden )
c
c  Evaluate for 12 <= argument.
c
        else

          res = 0.0D+00

          if ( y .le. frtbig ) then

            res = c(7)
            ysq = y * y

            do i = 1, 6
              res = res / ysq + c(i)
            end do

          end if

          res = res / y
          corr = log ( y )
          res = res + sqrtpi - 0.5D+00 * corr
          res = res + y * ( corr - 1.0D+00 )

        end if
c
c  Return for bad arguments.
c
      else

        res = xinf

      end if
c
c  Final adjustments and return.
c
      r8_gamma_log = res

      return
      end
      function r8_huge ( )

c*********************************************************************72
c
cc R8_HUGE returns a "huge" R8.
c
c  Discussion:
c
c    The value returned by this function is intended to be the largest
c    representable real value.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    27 September 2014
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Output, double precision R8_HUGE, a huge number.
c
      implicit none

      double precision r8_huge

      r8_huge = 1.79769313486231571D+308

      return
      end
      function r8_hypot ( x, y )

c*********************************************************************72
c
cc R8_HYPOT returns the value of sqrt ( X^2 + Y^2 ).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    26 March 2012
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision X, Y, the arguments.
c
c    Output, double precision R8_HYPOT, the value of sqrt ( X^2 + Y^2 ).
c
      implicit none

      double precision a
      double precision b
      double precision c
      double precision r8_hypot
      double precision x
      double precision y

      if ( abs ( x ) .lt. abs ( y ) ) then
        a = abs ( y )
        b = abs ( x )
      else
        a = abs ( x )
        b = abs ( y )
      end if
c
c  A contains the larger value.
c
      if ( a .eq. 0.0D+00 ) then
        c = 0.0D+00
      else
        c = a * sqrt ( 1.0D+00 + ( b / a ) ** 2 )
      end if

      r8_hypot = c

      return
      end
      function r8_in_01 ( a )

c*********************************************************************72
c
cc R8_IN_01 is TRUE if an R8 is in the range [0,1].
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    01 January 2007
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision A, the value.
c
c    Output, logical R8_IN_01, is TRUE if 0 <= A <= 1.
c
      implicit none

      double precision a
      logical r8_in_01
      logical value

      if ( a .lt. 0.0D+00 .or. 1.0D+00 .lt. a ) then
        value = .false.
      else
        value = .true.
      end if

      r8_in_01 = value

      return
      end
      function r8_insignificant ( r, s )

c*********************************************************************72
c
cc R8_INSIGNIFICANT determines if an R8 is insignificant.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    26 November 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision R, the number to be compared against.
c
c    Input, double precision S, the number to be compared.
c
c    Output, logical R8_INSIGNIFICANT, is TRUE if S is insignificant
c    compared to R.
c
      implicit none

      double precision r
      double precision r8_epsilon
      logical r8_insignificant
      double precision s
      double precision t
      double precision tol
      logical value

      value = .true. 

      t = r + s
      tol = r8_epsilon ( ) * abs ( r )

      if ( tol .lt. abs ( r - t ) ) then 
        value = .false.
      end if
  
      r8_insignificant = value

      return
      end
      function r8_is_int ( r )

c*********************************************************************72
c
cc R8_IS_INT determines if an R8 represents an integer value.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    21 March 2001
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision R, the number to be checked.
c
c    Output, logical R8_IS_INT, is TRUE if R is an integer value.
c
      implicit none

      integer i
      integer i4_huge
      parameter ( i4_huge = 2147483647 )
      double precision r
      logical r8_is_int

      if ( dble ( i4_huge ) .lt. r ) then
        r8_is_int = .false.
      else if ( r .lt. - dble ( i4_huge ) ) then
        r8_is_int = .false.
      else if ( r .eq. dble ( int ( r ) ) ) then
        r8_is_int = .true.
      else
        r8_is_int = .false.
      end if

      return
      end
      function r8_log_2 ( x )

c*********************************************************************72
c
cc R8_LOG_2 returns the logarithm base 2 of an R8.
c
c  Discussion:
c
c    value = Log ( |X| ) / Log ( 2.0 )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    01 January 2007
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision X, the number whose base 2 logarithm is desired.
c    X should not be 0.
c
c    Output, double precision R8_LOG_2, the logarithm base 2 of the absolute
c    value of X.  It should be true that |X| = 2**D_LOG_2.
c
      implicit none

      double precision r8_big
      double precision r8_log_2
      double precision x

      if ( x .eq. 0.0D+00 ) then
        r8_log_2 = - r8_big ( )
      else
        r8_log_2 = log ( abs ( x ) ) / log ( 2.0D+00 )
      end if

      return
      end
      function r8_log_10 ( x )

c*********************************************************************72
c
cc R8_LOG_10 returns the logarithm base 10 of an R8.
c
c  Discussion:
c
c    value = Log10 ( |X| )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    16 January 2007
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision X, the number whose base 2 logarithm is desired.
c    X should not be 0.
c
c    Output, double precision R8_LOG_10, the logarithm base 10 of the absolute
c    value of X.  It should be true that |X| = 10^R_LOG_10.
c
      implicit none

      double precision r8_big
      double precision r8_log_10
      double precision x

      if ( x .eq. 0.0D+00 ) then
        r8_log_10 = - r8_big ( )
      else
        r8_log_10 = log10 ( abs ( x ) )
      end if

      return
      end
      function r8_log_b ( x, b )

c*********************************************************************72
c
cc R8_LOG_B returns the logarithm base B of an R8.
c
c  Discussion:
c
c    value = log ( |X| ) / log ( |B| )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    02 June 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision X, the number whose base B logarithm is desired.
c    X should not be 0.
c
c    Input, double precision B, the base, which should not be 0, 1 or -1.
c
c    Output, double precision R8_LOG_B, the logarithm base B of the absolute
c    value of X.  It should be true that |X| = |B|^R8_LOG_B.
c
      implicit none

      double precision b
      double precision r8_big
      double precision r8_log_b
      double precision x

      if ( b .eq. 0.0D+00 .or.
     &     b .eq. 1.0D+00 .or.
     &     b .eq. - 1.0D+00 ) then
        r8_log_b = - r8_big ( )
      else if ( abs ( x ) .eq. 0.0D+00 ) then
        r8_log_b = - r8_big ( )
      else
        r8_log_b = log ( abs ( x ) ) / log ( abs ( b ) )
      end if

      return
      end
      subroutine r8_mant ( x, s, r, l )

c*********************************************************************72
c
cc R8_MANT computes the "mantissa" or "fraction part" of an R8.
c
c  Discussion:
c
c    X = S * R * 2^L
c
c    S is +1 or -1,
c    R is an R8 value between 1.0 and 2.0,
c    L is an integer.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    02 June 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision X, the number to be decomposed.
c
c    Output, integer S, the "sign" of the number.
c    S will be -1 if X is less than 0, and +1 if X is greater
c    than or equal to zero.
c
c    Output, double precision R, the mantissa of X.  R will be greater
c    than or equal to 1, and strictly less than 2.  The one
c    exception occurs if X is zero, in which case R will also
c    be zero.
c
c    Output, integer L, the integer part of the logarithm
c    (base 2) of X.
c
      implicit none

      integer l
      double precision r
      integer s
      double precision x
c
c  Determine the sign.
c
      if ( x .lt. 0.0D+00 ) then
        s = -1
      else
        s = 1
      end if
c
c  Set R to the absolute value of X, and L to zero.
c  Then force R to lie between 1 and 2.
c
      if ( x .lt. 0.0D+00 ) then
        r = -x
      else
        r = x
      end if

      l = 0
c
c  Time to bail out if X is zero.
c
      if ( x .eq. 0.0D+00 ) then
        return
      end if

10    continue

      if ( 2.0D+00 .le. r ) then
        r = r / 2.0D+00
        l = l + 1
        go to 10
      end if

20    continue

      if ( r .lt. 1.0D+00 ) then
        r = r * 2.0D+00
        l = l - 1
        go to 20
      end if

      return
      end
      function r8_max ( x, y )

c*********************************************************************72
c
cc R8_MAX returns the maximum of two R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    05 May 2014
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision X, Y, the numbers to compare.
c
c    Output, double precision R8_MAX, the maximum of X and Y.
c
      implicit none

      double precision r8_max
      double precision x
      double precision y

      if ( x < y ) then
        r8_max = y
      else
        r8_max = x
      end if

      return
      end
      function r8_min ( x, y )

c*********************************************************************72
c
cc R8_MIN returns the minimum of two R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    09 March 2015
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision X, Y, the numbers to compare.
c
c    Output, double precision R8_MIN, the minimum of X and Y.
c
      implicit none

      double precision r8_min
      double precision x
      double precision y

      if ( x < y ) then
        r8_min = x
      else
        r8_min = y
      end if

      return
      end
      function r8_mod ( x, y )

c*********************************************************************72
c
cc R8_MOD returns the remainder of R8 division.
c
c  Discussion:
c
c    If
c      REM = R8_MOD ( X, Y )
c      RMULT = ( X - REM ) / Y
c    then
c      X = Y * RMULT + REM
c    where REM has the same sign as X, and abs ( REM ) < Y.
c
c  Example:
c
c        X         Y     R8_MOD  R8_MOD Factorization
c
c      107        50       7      107 =  2 *  50 + 7
c      107       -50       7      107 = -2 * -50 + 7
c     -107        50      -7     -107 = -2 *  50 - 7
c     -107       -50      -7     -107 =  2 * -50 - 7
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    02 June 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision X, the number to be divided.
c
c    Input, double precision Y, the number that divides X.
c
c    Output, double precision R8_MOD, the remainder when X is divided by Y.
c
      implicit none

      double precision r8_mod
      double precision x
      double precision y

      if ( y .eq. 0.0D+00 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8_MOD - Fatal error!'
        write ( *, '(a,g14.6)' ) '  R8_MOD ( X, Y ) called with Y = ', y
        stop 1
      end if

      r8_mod = x - dble ( int ( x / y ) ) * y

      if ( x .lt. 0.0D+00 .and. 0.0D+00 .lt. r8_mod ) then
        r8_mod = r8_mod - abs ( y )
      else if ( 0.0D+00 .lt. x .and. r8_mod .lt. 0.0D+00 ) then
        r8_mod = r8_mod + abs ( y )
      end if

      return
      end
      function r8_modp ( x, y )

c*********************************************************************72
c
cc R8_MODP returns the nonnegative remainder of R8 division.
c
c  Formula:
c
c    If
c      REM = R8_MODP ( X, Y )
c      RMULT = ( X - REM ) / Y
c    then
c      X = Y * RMULT + REM
c    where REM is always nonnegative.
c
c  Discussion:
c
c    The MOD function computes a result with the same sign as the
c    quantity being divided.  Thus, suppose you had an angle A,
c    and you wanted to ensure that it was between 0 and 360.
c    Then mod(A,360.0) would do, if A was positive, but if A
c    was negative, your result would be between -360 and 0.
c
c    On the other hand, R8_MODP(A,360.0) is between 0 and 360, always.
c
c  Example:
c
c        X         Y     MOD R8_MODP  R8_MODP Factorization
c
c      107        50       7       7    107 =  2 *  50 + 7
c      107       -50       7       7    107 = -2 * -50 + 7
c     -107        50      -7      43   -107 = -3 *  50 + 43
c     -107       -50      -7      43   -107 =  3 * -50 + 43
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    19 October 2004
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision X, the number to be divided.
c
c    Input, double precision Y, the number that divides X.
c
c    Output, double precision R8_MODP, the nonnegative remainder
c    when X is divided by Y.
c
      implicit none

      double precision r8_modp
      double precision x
      double precision y

      if ( y .eq. 0.0D+00 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8_MODP - Fatal error!'
        write ( *, '(a,g14.6)' )
     &    '  R8_MODP ( X, Y ) called with Y = ', y
        stop 1
      end if

      r8_modp = mod ( x, y )

      if ( r8_modp .lt. 0.0D+00 ) then
        r8_modp = r8_modp + abs ( y )
      end if

      return
      end
      function r8_mop ( i )

c*********************************************************************72
c
cc R8_MOP returns the I-th power of -1 as an R8.
c
c  Discussion:
c
c    An R8 is a double precision real value.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    26 July 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer I, the power of -1.
c
c    Output, double precision R8_MOP, the I-th power of -1.
c
      implicit none

      integer i
      double precision r8_mop
      double precision value

      if ( mod ( i, 2 ) .eq. 0 ) then
        value = + 1.0D+00
      else
        value = - 1.0D+00
      end if

      r8_mop = value

      return
      end
      function r8_nint ( x )

c*********************************************************************72
c
cc R8_NINT returns the nearest integer to an R8.
c
c  Example:
c
c        X        R8_NINT
c
c      1.3         1
c      1.4         1
c      1.5         1 or 2
c      1.6         2
c      0.0         0
c     -0.7        -1
c     -1.1        -1
c     -1.6        -2
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    08 September 2005
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision X, the value.
c
c    Output, integer R8_NINT, the nearest integer to X.
c
      implicit none

      integer r8_nint
      integer s
      double precision x

      if ( x .lt. 0.0D+00 ) then
        s = -1
      else
        s = 1
      end if

      r8_nint = s * int ( abs ( x ) + 0.5D+00 )

      return
      end
      function r8_normal_01 ( seed )

c*********************************************************************72
c
cc R8_NORMAL_01 returns a unit pseudonormal R8.
c
c  Discussion:
c
c    This routine uses the Box Muller method.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    05 August 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input/output, integer SEED, a seed for the random number generator.
c
c    Output, double precision R8_NORMAL_01, a sample of the standard normal PDF.
c
      implicit none

      double precision r1
      double precision r2
      double precision r8_normal_01
      double precision r8_pi
      parameter ( r8_pi = 3.141592653589793D+00 )
      double precision r8_uniform_01
      integer seed
      double precision x

      r1 = r8_uniform_01 ( seed )
      r2 = r8_uniform_01 ( seed )
      x = sqrt ( -2.0D+00 * log ( r1 ) ) * cos ( 2.0D+00 * r8_pi * r2 )

      r8_normal_01 = x

      return
      end
      function r8_normal_ab ( a, b, seed )

c*********************************************************************72
c
cc R8_NORMAL_AB returns a scaled pseudonormal R8.
c
c  Discussion:
c
c    The normal probability distribution function (PDF) is sampled,
c    with mean A and standard deviation B.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    05 August 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision A, the mean of the PDF.
c
c    Input, double precision B, the standard deviation of the PDF.
c
c    Input/output, integer SEED, a seed for the random number generator.
c
c    Output, double precision R8_NORMAL_AB, a sample of the normal PDF.
c
      implicit none

      double precision a
      double precision b
      double precision r1
      double precision r2
      double precision r8_normal_ab
      double precision r8_pi
      parameter ( r8_pi = 3.141592653589793D+00 )
      double precision r8_uniform_01
      integer seed
      double precision x

      r1 = r8_uniform_01 ( seed )
      r2 = r8_uniform_01 ( seed )
      x = sqrt ( -2.0D+00 * log ( r1 ) ) * cos ( 2.0D+00 * r8_pi * r2 )

      r8_normal_ab = a + b * x

      return
      end
      function r8_pi ( )

c*********************************************************************72
c
cc R8_PI returns the value of PI as an R8.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    01 January 2007
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Output, double precision R8_PI, the value of pi.
c
      implicit none

      double precision r8_pi

      r8_pi = 3.141592653589793D+00

      return
      end
      function r8_pi_sqrt ( )

c*********************************************************************72
c
cc R8_PI_SQRT returns the square root of PI as an R8.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    11 September 2012
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Output, double precision R8_PI_SQRT, the square root of pi.
c
      implicit none

      double precision r8_pi_sqrt

      r8_pi_sqrt = 1.7724538509055160273D+00

      return
      end
      function r8_power ( r, p )

c*********************************************************************72
c
cc R8_POWER computes the P-th power of an R8.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    01 January 2007
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision R, the base.
c
c    Input, integer P, the power, which may be negative.
c
c    Output, double precision R8_POWER, the value of the P-th power of R.
c
      implicit none

      integer p
      double precision r
      double precision r8_power
      double precision value
c
c  Special case.  R^0 = 1.
c
      if ( p .eq. 0 ) then

        value = 1.0D+00
c
c  Special case.  Positive powers of 0 are 0.
c  For negative powers of 0, we go ahead and compute R^P,
c  relying on the software to complain.
c
      else if ( r .eq. 0.0D+00 ) then

        if ( 0 .lt. p ) then
          value = 0.0D+00
        else
          value = r ** p
        end if

      else if ( 1 .le. p ) then
        value = r ** p
      else
        value = 1.0D+00 / r ** (-p)
      end if

      r8_power = value

      return
      end
      subroutine r8_power_fast ( r, p, rp, mults )

c*********************************************************************72
c
cc R8_POWER_FAST computes an integer power of an R8.
c
c  Discussion:
c
c    Obviously, R^P can be computed using P-1 multiplications.
c
c    However, R^P can also be computed using at most 2*LOG2(P) multiplications.
c    To do the calculation this way, let N = LOG2(P).
c    Compute A, A^2, A^4, ..., A^N by N-1 successive squarings.
c    Start the value of R^P at A, and each time that there is a 1 in
c    the binary expansion of P, multiply by the current result of the squarings.
c
c    This algorithm is not optimal.  For small exponents, and for special
c    cases, the result can be computed even more quickly.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    02 June 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision R, the base.
c
c    Input, integer P, the power, which may be negative.
c
c    Output, double precision RP, the value of R^P.
c
c    Output, integer MULTS, the number of multiplications
c    and divisions.
c
      implicit none

      integer mults
      integer p
      integer p_mag
      integer p_sign
      double precision r
      double precision r2
      double precision rp

      mults = 0
c
c  Special bases.
c
      if ( r .eq. 1.0D+00 ) then
        rp = 1.0D+00
        return
      end if

      if ( r .eq. -1.0D+00 ) then

        if ( mod ( p, 2 ) .eq. 1 ) then
          rp = -1.0D+00
        else
          rp = 1.0D+00
        end if

        return

      end if

      if ( r .eq. 0.0D+00 ) then

        if ( p .le. 0 ) then
          write ( *, '(a)' ) ' '
          write ( *, '(a)' ) 'R8_POWER_FAST - Fatal error!'
          write ( *, '(a)' )
     &      '  Base R is zero, and exponent is negative.'
          write ( *, '(a,i8)' ) '  Exponent P = ', p
          stop 1
        end if

        rp = 0.0D+00
        return

      end if
c
c  Special powers.
c
      if ( p .eq. -1 ) then
        rp = 1.0D+00 / r
        mults = mults + 1
        return
      else if ( p .eq. 0 ) then
        rp = 1.0D+00
        return
      else if ( p .eq. 1 ) then
        rp = r
        return
      end if
c
c  Some work to do.
c
      p_mag = abs ( p )
      p_sign = sign ( 1, p )

      rp = 1.0D+00
      r2 = r

10    continue

      if ( 0 .lt. p_mag ) then

        if ( mod ( p_mag, 2 ) .eq. 1 ) then
          rp = rp * r2
          mults = mults + 1
        end if

        p_mag = p_mag / 2
        r2 = r2 * r2
        mults = mults + 1

        go to 10

      end if

      if ( p_sign .eq. -1 ) then
        rp = 1.0D+00 / rp
        mults = mults + 1
      end if

      return
      end
      subroutine r8_print ( r, title )

c*********************************************************************72
c
cc R8_PRINT prints an R8.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    14 August 2014
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision R, the value.
c
c    Input, character * ( * ) TITLE, a title.
c
      implicit none

      double precision r
      character * ( * ) title

      write ( *, '(a,2x,g14.6)' ) trim ( title ), r

      return
      end
      function r8_pythag ( a, b )

c*********************************************************************72
c
cc R8_PYTHAG computes sqrt ( A * A + B * B ) as an R8.
c
c  Discussion:
c
c    The computation avoids unnecessary overflow and underflow.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    02 June 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision A, B, the values for which sqrt ( A * A + B * B )
c    is desired.
c
c    Output, double precision R8_PYTHAG, the value of sqrt ( A * A + B * B ).
c
      implicit none

      double precision a
      double precision a_abs
      double precision b
      double precision b_abs
      double precision r8_pythag

      a_abs = abs ( a )
      b_abs = abs ( b )

      if ( b_abs .lt. a_abs ) then
        r8_pythag = a_abs *
     &    sqrt ( 1.0D+00 + ( b_abs / a_abs ) * ( b_abs / a_abs ) )
      else if ( b_abs .eq. 0.0D+00 ) then
        r8_pythag = 0.0D+00
      else if ( a_abs .le. b_abs ) then
        r8_pythag = b_abs *
     &    sqrt ( 1.0D+00 + ( a_abs / b_abs ) * ( a_abs / b_abs ) )
      end if

      return
      end
      function r8_radians ( degrees )

c*********************************************************************72
c
cc R8_RADIANS converts an angle from degree to radian measure.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    15 May 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision DEGREES, the angle measurement in degrees.
c
c    Output, double precision R8_RADIANS, the angle measurement in radians.
c
      implicit none

      double precision degrees
      real ( kind = 8 ) r8_pi
      parameter ( r8_pi = 3.1415926535897932384626434D+00 )
      double precision r8_radians

      r8_radians = degrees * r8_pi / 180.0D+00

      return
      end
      function r8_rise ( x, n )

c*********************************************************************72
c
cc R8_RISE computes the rising factorial function [X]^N.
c
c  Discussion:
c
c    [X]^N = X * ( X + 1 ) * ( X + 2 ) * ... * ( X + N - 1 ).
c
c    Note that the number of ways of arranging N objects in M ordered
c    boxes is [M]^N.  (Here, the ordering of the objects in each box matters).  
c    Thus, 2 objects in 2 boxes have the following 6 possible arrangements:
c
c      -|12, 1|2, 12|-, -|21, 2|1, 21|-.
c
c    Moreover, the number of non-decreasing maps from a set of
c    N to a set of M ordered elements is [M]^N / Nc.  Thus the set of
c    nondecreasing maps from (1,2,3) to (a,b,c,d) is the 20 elements:
c
c      aaa, abb, acc, add, aab, abc, acd, aac, abd, aad
c      bbb, bcc, bdd, bbc, bcd, bbd, ccc, cdd, ccd, ddd.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    24 January 2007
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision X, the argument of the rising factorial function.
c
c    Input, integer N, the order of the rising factorial function.
c    If N = 0, RISE = 1, if N = 1, RISE = X.  Note that if N is
c    negative, a "falling" factorial will be computed.
c
c    Output, double precision R8_RISE, the value of the rising factorial 
c    function.
c
      implicit none

      double precision arg
      integer i
      integer n
      double precision r8_rise
      double precision value
      double precision x

      value = 1.0D+00

      arg = x

      if ( 0 .lt. n ) then

        do i = 1, n
          value = value * arg
          arg = arg + 1.0D+00
        end do

      else if ( n .lt. 0 ) then

        do i = -1, n, -1
          value = value * arg
          arg = arg - 1.0D+00
        end do

      end if

      r8_rise = value

      return
      end
      subroutine r8_rise_values ( n_data, x, n, f )

c*********************************************************************72
c
cc R8_RISE_VALUES returns some values of the rising factorial function.
c
c  Discussion:
c
c    Pochhammer(X,Y) = Gamma(X+Y) / Gamma(X)
c
c    For integer arguments, Pochhammer(M,N) = ( M + N - 1 )! / ( N - 1 )!
c
c    In Mathematica, the function can be evaluated by:
c
c      Pochhammer[X,Y]
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    20 December 2014
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Milton Abramowitz, Irene Stegun,
c    Handbook of Mathematical Functions,
c    National Bureau of Standards, 1964,
c    ISBN: 0-486-61272-4,
c    LC: QA47.A34.
c
c    Stephen Wolfram,
c    The Mathematica Book,
c    Fourth Edition,
c    Cambridge University Press, 1999,
c    ISBN: 0-521-64314-7,
c    LC: QA76.95.W65.
c
c  Parameters:
c
c    Input/output, integer N_DATA.  The user sets N_DATA to 0
c    before the first call.  On each call, the routine increments N_DATA by 1,
c    and returns the corresponding data; when there is no more data, the
c    output value of N_DATA will be 0 again.
c
c    Output, double precision X, integer N, the arguments of the function.
c
c    Output, double precision F, the value of the function.
c
      implicit none

      integer n_max
      parameter ( n_max = 15 )

      double precision f_vec(n_max)
      double precision f
      integer n
      integer n_data
      integer n_vec(n_max)
      double precision x
      double precision x_vec(n_max)

      save f_vec
      save n_vec
      save x_vec

      data f_vec /
     &  1680.000000000000D+00,
     &  1962.597656250000D+00,
     &  2279.062500000000D+00,
     &  2631.972656250000D+00,
     &  3024.000000000000D+00,
     &  1.000000000000000D+00,
     &  7.500000000000000D+00,
     &  63.75000000000000D+00,
     &  605.6250000000000D+00,
     &  6359.062500000000D+00,
     &  73129.21875000000D+00,
     &  914115.2343750000D+00,
     &  1.234055566406250D+07,
     &  1.789380571289063D+08,
     &  2.773539885498047D+09 /

      data n_vec /
     &  4,
     &  4,
     &  4,
     &  4,
     &  4,
     &  0,
     &  1,
     &  2,
     &  3,
     &  4,
     &  5,
     &  6,
     &  7,
     &  8,
     &  9 /

      data x_vec /
     &  5.00D+00,
     &  5.25D+00,
     &  5.50D+00,
     &  5.75D+00,
     &  6.00D+00,
     &  7.50D+00,
     &  7.50D+00,
     &  7.50D+00,
     &  7.50D+00,
     &  7.50D+00,
     &  7.50D+00,
     &  7.50D+00,
     &  7.50D+00,
     &  7.50D+00,
     &  7.50D+00 /

      if ( n_data .lt. 0 ) then
        n_data = 0
      end if

      n_data = n_data + 1

      if ( n_max .lt. n_data ) then
        n_data = 0
        x = 0.0D+00
        n = 0
        f = 0.0D+00
      else
        x = x_vec(n_data)
        n = n_vec(n_data)
        f = f_vec(n_data)
      end if

      return
      end
      function r8_round ( x )

c*********************************************************************72
c
cc R8_ROUND sets an R8 to the nearest integral value.
c
c  Example:
c
c        X        R8_ROUND
c
c      1.3         1.0
c      1.4         1.0
c      1.5         1.0 or 2.0
c      1.6         2.0
c      0.0         0.0
c     -0.7        -1.0
c     -1.1        -1.0
c     -1.6        -2.0
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    15 October 2012
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision X, the value.
c
c    Output, double precision R8_ROUND, the rounded value.
c
      implicit none

      double precision r8_round
      double precision value
      double precision x

      if ( x .lt. 0.0D+00 ) then
        value = - dble ( int ( - x + 0.5D+00 ) )
      else
        value =   dble ( int ( + x + 0.5D+00 ) )
      end if

      r8_round = value

      return
      end
      function r8_round_i4 ( x )

c*********************************************************************72
c
cc R8_ROUND_I4 sets an R8 to the nearest integral value, returning an I4.
c
c  Example:
c
c        X        R8_ROUND_I4
c
c      1.3         1
c      1.4         1
c      1.5         1 or 2
c      1.6         2
c      0.0         0
c     -0.7        -1
c     -1.1        -1
c     -1.6        -2
c
c  Discussion:
c
c    In FORTRAN77, we rely on the fact that, for positive X, int ( X )
c    is the "floor" function, returning the largest integer less than
c    or equal to X.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    25 March 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision X, the value.
c
c    Output, integer R8_ROUND_I4, the rounded value.
c
      implicit none

      integer r8_round_i4
      integer value
      double precision x

      if ( x .lt. 0.0D+00 ) then
        value = - int ( - x + 0.5D+00 )
      else
        value =   int ( + x + 0.5D+00 )
      end if

      r8_round_i4 = value

      return
      end
      subroutine r8_round2 ( nplace, x, xround )

c*********************************************************************72
c
cc R8_ROUND2 rounds an R8 to a specified number of binary digits.
c
c  Discussion:
c
c    Assume that the input quantity X has the form
c
c      X = S * J * 2^L
c
c    where S is plus or minus 1, L is an integer, and J is a binary
c    mantissa which is either exactly zero, or greater than or equal
c    to 0.5 and strictly less than 1.0.
c
c    Then on return, XROUND will satisfy
c
c      XROUND = S * K * 2^L
c
c    where S and L are unchanged, and K is a binary mantissa which
c    agrees with J in the first NPLACE binary digits and is zero
c    thereafter.
c
c    If NPLACE is 0, XROUND will always be zero.
c
c    If NPLACE is 1, the mantissa of XROUND will be 0 or 0.5.
c
c    If NPLACE is 2, the mantissa of XROUND will be 0, 0.25, 0.50,
c    or 0.75.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    02 June 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer NPLACE, the number of binary digits to
c    preserve.  NPLACE should be 0 or positive.
c
c    Input, double precision X, the number to be decomposed.
c
c    Output, double precision XROUND, the rounded value of X.
c
      implicit none

      integer iplace
      integer l
      integer nplace
      integer s
      double precision x
      double precision xmant
      double precision xround
      double precision xtemp

      xround = 0.0D+00
c
c  1: Handle the special case of 0.
c
      if ( x .eq. 0.0D+00 ) then
        return
      end if

      if ( nplace .le. 0 ) then
        return
      end if
c
c  2: Determine the sign S.
c
      if ( 0.0D+00 .lt. x ) then
        s = 1
        xtemp = x
      else
        s = -1
        xtemp = -x
      end if
c
c  3: Force XTEMP to lie between 1 and 2, and compute the
c  logarithm L.
c
      l = 0

10    continue

      if ( 2.0D+00 .le. xtemp ) then
        xtemp = xtemp / 2.0D+00
        l = l + 1
        go to 10
      end if

20    continue

      if ( xtemp .lt. 1.0D+00 ) then
        xtemp = xtemp * 2.0D+00
        l = l - 1
        go to 20
      end if
c
c  4: Strip out the digits of the mantissa as XMANT, and decrease L.
c
      xmant = 0.0D+00
      iplace = 0

30    continue

        xmant = 2.0D+00 * xmant

        if ( 1.0D+00 .le. xtemp ) then
          xmant = xmant + 1.0D+00
          xtemp = xtemp - 1.0D+00
        end if

        iplace = iplace + 1

        if ( xtemp .eq. 0.0D+00 .or. nplace .le. iplace ) then
          xround = s * xmant * 2.0D+00 ** l
          go to 40
        end if

        l = l - 1
        xtemp = xtemp * 2.0D+00

      go to 30

40    continue

      return
      end
      subroutine r8_roundb ( base, nplace, x, xround )

c*********************************************************************72
c
cc R8_ROUNDB rounds an R8 to a given number of digits in a given base.
c
c  Discussion:
c
c    The code does not seem to do a good job of rounding when
c    the base is negative.
c
c    Assume that the input quantity X has the form
c
c      X = S * J * BASE^L
c
c    where S is plus or minus 1, L is an integer, and J is a
c    mantissa base BASE which is either exactly zero, or greater
c    than or equal to (1/BASE) and less than 1.0.
c
c    Then on return, XROUND will satisfy
c
c      XROUND = S * K * BASE^L
c
c    where S and L are unchanged, and K is a mantissa base BASE
c    which agrees with J in the first NPLACE digits and is zero
c    thereafter.
c
c    Note that because of rounding, for most bases, most numbers
c    with a fractional quantities cannot be stored exactly in the
c    computer, and hence will have trailing "bogus" digits.
c
c    If NPLACE is 0, XROUND will always be zero.
c
c    If NPLACE is 1, the mantissa of XROUND will be 0,
c    1/BASE, 2/BASE, ..., (BASE-1)/BASE.
c
c    If NPLACE is 2, the mantissa of XROUND will be 0,
c    BASE/BASE^2, (BASE+1)/BASE^2, ...,
c    BASE^2-2/BASE^2, BASE^2-1/BASE^2.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    02 June 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer BASE, the base of the arithmetic.
c    BASE must not be zero.  Theoretically, BASE may be negative.
c
c    Input, integer NPLACE, the number of digits base BASE to
c    preserve.  NPLACE should be 0 or positive.
c
c    Input, double precision X, the number to be decomposed.
c
c    Output, double precision XROUND, the rounded value of X.
c
      implicit none

      integer base
      integer iplace
      integer is
      integer js
      integer l
      integer nplace
      double precision x
      double precision xmant
      double precision xround
      double precision xtemp

      xround = 0.0D+00
c
c  0: Error checks.
c
      if ( base .eq. 0 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8_ROUNDB - Fatal error!'
        write ( *, '(a)' ) '  The base BASE cannot be zero.'
        stop 1
      end if
c
c  1: Handle the special case of 0.
c
      if ( x .eq. 0.0D+00 ) then
        return
      end if

      if ( nplace .le. 0 ) then
        return
      end if
c
c  2: Determine the sign IS.
c
      if ( 0.0D+00 .lt. x ) then
        is = 1
        xtemp = x
      else
        is = -1
        xtemp = -x
      end if
c
c  3: Force XTEMP to lie between 1 and ABS(BASE), and compute the
c  logarithm L.
c
      l = 0

10    continue

      if ( abs ( base ) .le. abs ( xtemp ) ) then

        xtemp = xtemp / dble ( base )

        if ( xtemp .lt. 0.0D+00 ) then
          is = -is
          xtemp = -xtemp
        end if

        l = l + 1

        go to 10

      end if

20    continue

      if ( abs ( xtemp ) .lt. 1.0D+00 ) then

        xtemp = xtemp * base

        if ( xtemp .lt. 0.0D+00 ) then
          is = -is
          xtemp = -xtemp
        end if

        l = l - 1

        go to 20

      end if
c
c  4: Now strip out the digits of the mantissa as XMANT, and
c  decrease L.
c
      xmant = 0.0D+00
      iplace = 0
      js = is

30    continue

        xmant = base * xmant

        if ( xmant .lt. 0.0D+00 ) then
          js = -js
          xmant = -xmant
        end if

        if ( 1.0D+00 .le. xtemp ) then
          xmant = xmant + int ( xtemp )
          xtemp = xtemp - int ( xtemp )
        end if

        iplace = iplace + 1

        if ( xtemp .eq. 0.0D+00 .or. nplace .le. iplace ) then
          xround = js * xmant * dble ( base ) ** l
          go to 40
        end if

        l = l - 1
        xtemp = xtemp * base

        if ( xtemp .lt. 0.0D+00 ) then
          is = -is
          xtemp = -xtemp
        end if

      go to 30

40    continue

      return
      end
      subroutine r8_roundx ( nplace, x, xround )

c*********************************************************************72
c
cc R8_ROUNDX rounds an R8.
c
c  Discussion:
c
c    Assume that the input quantity X has the form
c
c      X = S * J * 10^L
c
c    where S is plus or minus 1, L is an integer, and J is a decimal
c    mantissa which is either exactly zero, or greater than or equal
c    to 0.1 and less than 1.0.
c
c    Then on return, XROUND will satisfy
c
c      XROUND = S * K * 10^L
c
c    where S and L are unchanged, and K is a decimal mantissa which
c    agrees with J in the first NPLACE decimal digits and is zero
c    thereafter.
c
c    Note that because of rounding, most decimal fraction quantities
c    cannot be stored exactly in the computer, and hence will have
c    trailing "bogus" digits.
c
c    If NPLACE is 0, XROUND will always be zero.
c
c    If NPLACE is 1, the mantissa of XROUND will be 0, 0.1,
c    0.2, ..., or 0.9.
c
c    If NPLACE is 2, the mantissa of XROUND will be 0, 0.01, 0.02,
c    0.03, ..., 0.98, 0.99.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    02 June 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer NPLACE, the number of decimal digits to
c    preserve.  NPLACE should be 0 or positive.
c
c    Input, double precision X, the number to be decomposed.
c
c    Output, double precision XROUND, the rounded value of X.
c
      implicit none

      integer iplace
      integer is
      integer l
      integer nplace
      double precision x
      double precision xmant
      double precision xround
      double precision xtemp

      xround = 0.0D+00
c
c  1: Handle the special case of 0.
c
      if ( x .eq. 0.0D+00 ) then
        return
      end if

      if ( nplace .le. 0 ) then
        return
      end if
c
c  2: Determine the sign IS.
c
      if ( 0.0D+00 .lt. x ) then
        is = 1
        xtemp = x
      else
        is = -1
        xtemp = -x
      end if
c
c  3: Force XTEMP to lie between 1 and 10, and compute the
c  logarithm L.
c
      l = 0

10    continue

      if ( 10.0D+00 .le. x ) then
        xtemp = xtemp / 10.0D+00
        l = l + 1
        go to 10
      end if

20    continue

      if ( xtemp .lt. 1.0D+00 ) then
        xtemp = xtemp * 10.0D+00
        l = l - 1
        go to 20
      end if
c
c  4: Now strip out the digits of the mantissa as XMANT, and
c  decrease L.
c
      xmant = 0.0D+00
      iplace = 0

30    continue

        xmant = 10.0D+00 * xmant

        if ( 1.0D+00 .le. xtemp ) then
          xmant = xmant + int ( xtemp )
          xtemp = xtemp - int ( xtemp )
        end if

        iplace = iplace + 1

        if ( xtemp .eq. 0.0D+00 .or. nplace .le. iplace ) then
          xround = is * xmant * ( 10.0D+00 ** l )
          go to 40
        end if

        l = l - 1
        xtemp = xtemp * 10.0D+00

      go to 30

40    continue

      return
      end
      function r8_secd ( degrees )

c*********************************************************************72
c
cc R8_SECD returns the secant of an angle given in degrees.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    27 July 2014
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision DEGREES, the angle in degrees.
c
c    Output, double precision R8_SECD, the secant of the angle.
c
      implicit none

      double precision degrees
      double precision r8_secd
      double precision r8_pi 
      parameter ( r8_pi = 3.141592653589793D+00 )
      double precision radians

      radians = r8_pi * ( degrees / 180.0D+00 )
      r8_secd = 1.0D+00 / cos ( radians )

      return
      end
      function r8_sech ( x )

c*********************************************************************72
c
cc R8_SECH evaluates the hyperbolic secant, while avoiding COSH overflow.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    29 August 2000
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision X, the argument of the function.
c
c    Output, double precision R8_SECH, the value of the function.
c
      implicit none

      double precision log_huge 
      parameter ( log_huge = 80.0D+00 )
      double precision r8_sech
      double precision x

      if ( log_huge .lt. abs ( x ) ) then
        r8_sech = 0.0D+00
      else
        r8_sech = 1.0D+00 / cosh ( x )
      end if

      return
      end
      function r8_sign ( x )

c*********************************************************************72
c
cc R8_SIGN returns the sign of an R8.
c
c  Discussion:
c
c    value = -1 if X < 0;
c    value = +1 if X => 0.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    23 August 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision X, the number whose sign is desired.
c
c    Output, double precision R8_SIGN, the sign of X.
c
      implicit none

      double precision r8_sign
      double precision value
      double precision x

      if ( x .lt. 0.0D+00 ) then
        value = -1.0D+00
      else
        value = +1.0D+00
      end if

      r8_sign = value

      return
      end
      function r8_sign3 ( x )

c*********************************************************************72
c
cc R8_SIGN3 returns the sign of an R8.
c
c  Discussion:
c
c    value = -1 if X < 0;
c    value =  0 if X = 0
c    value = +1 if X > 0.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    28 September 2014
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision X, the number whose sign is desired.
c
c    Output, double precision R8_SIGN3, the sign of X.
c
      implicit none

      double precision r8_sign3
      double precision value
      double precision x

      if ( x .lt. 0.0D+00 ) then
        value = -1.0D+00
      else if ( x .eq. 0.0D+00 ) then
        value = 0.0D+00
      else
        value = +1.0D+00
      end if

      r8_sign3 = value

      return
      end
      function r8_sign_char ( x )

c*********************************************************************72
c
cc R8_SIGN_CHAR returns a character indicating the sign of an R8.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    28 April 2012
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision X, the number whose sign is desired.
c
c    Output, character R8_SIGN_CHAR, the sign of X, '-', '0' or '+'.
c
      implicit none

      character r8_sign_char
      character value
      double precision x

      if ( x .lt. 0.0D+00 ) then
        value = '-'
      else if ( x .eq. 0.0D+00 ) then
        value = '0'
      else
        value = '+'
      end if

      r8_sign_char = value

      return
      end
      function r8_sign_match ( r1, r2 )

c*********************************************************************72
c
cc R8_SIGN_MATCH is TRUE if two R8's are of the same sign.
c
c  Discussion:
c
c    This test could be coded numerically as
c
c      if ( 0 <= r1 * r2 ) then ...
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    26 April 2012
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision R1, R2, the values to check.
c
c    Output, logical R8_SIGN_MATCH, is TRUE if ( R1 <= 0 and R2 <= 0 )
c    or ( 0 <= R1 and 0 <= R2 ).
c
      implicit none

      double precision r1
      double precision r2
      logical r8_sign_match

      r8_sign_match = ( r1 .le. 0.0D+00 .and. r2 .le. 0.0D+00 ) .or. 
     &                ( 0.0D+00 .le. r1 .and. 0.0D+00 .le. r2 )

      return
      end
      function r8_sign_match_strict ( r1, r2 )

c*********************************************************************72
c
cc R8_SIGN_MATCH_STRICT is TRUE if two R8's are of the same strict sign.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    28 April 2012
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision R1, R2, the values to check.
c
c    Output, logical R8_SIGN_MATCH_STRICT, is TRUE if the signs match.
c
      implicit none

      double precision r1
      double precision r2
      logical r8_sign_match_strict

      r8_sign_match_strict = 
     &  ( r1 .lt. 0.0D+00 .and. r2 .lt. 0.0D+00 ) .or.
     &  ( r1 .eq. 0.0D+00 .and. r2 .eq. 0.0D+00 ) .or.
     &  ( 0.0D+00 .lt. r1 .and. 0.0D+00 .lt. r2 )

      return
      end
      function r8_sign_opposite ( r1, r2 )

c*********************************************************************72
c
cc R8_SIGN_OPPOSITE is TRUE if two R8's are not of the same sign.
c
c  Discussion:
c
c    This test could be coded numerically as
c
c      if ( r1 * r2 <= 0.0 ) then ...
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    23 June 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision R1, R2, the values to check.
c
c    Output, logical R8_SIGN_OPPOSITE, is TRUE if ( R1 <= 0 and 0 <= R2 )
c    or ( R2 <= 0 and 0 <= R1 ).
c
      implicit none

      double precision r1
      double precision r2
      logical r8_sign_opposite

      r8_sign_opposite = 
     &  ( r1 .le. 0.0D+00 .and. 0.0D+00 .le. r2 ) .or.
     &  ( r2 .le. 0.0D+00 .and. 0.0D+00 .le. r1 )

      return
      end
      function r8_sign_opposite_strict ( r1, r2 )

c*********************************************************************72
c
cc R8_SIGN_OPPOSITE_STRICT is TRUE if two R8's are strictly of opposite sign.
c
c  Discussion:
c
c    This test could be coded numerically as
c
c      if ( r1 * r2 < 0.0 ) then ...
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    23 June 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision R1, R2, the values to check.
c
c    Output, logical R8_SIGN_OPPOSITE_STRICT, is TRUE if ( R1 < 0 and 0 < R2 )
c    or ( R2 < 0 and 0 < R1 ).
c
      implicit none

      double precision r1
      double precision r2
      logical r8_sign_opposite_strict

      r8_sign_opposite_strict = ( r1 < 0.0D+00 .and. 0.0D+00 < r2 ) .or.
     &                          ( r2 < 0.0D+00 .and. 0.0D+00 < r1 )

      return
      end
      function r8_sind ( degrees )

c*********************************************************************72
c
cc R8_SIND returns the sine of an angle given in degrees.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    27 July 2014
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision DEGREES, the angle in degrees.
c
c    Output, double precision R8_SIND, the sine of the angle.
c
      implicit none

      double precision degrees
      double precision r8_pi
      parameter ( r8_pi = 3.141592653589793D+00 )
      double precision r8_sind
      double precision radians

      radians = r8_pi * ( degrees / 180.0D+00 )

      r8_sind  = sin ( radians )

      return
      end
      function r8_sqrt_i4 ( i )

!*********************************************************************72
!
!! R8_SQRT_I4 returns the square root of an I4 as an R8.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license. 
!
!  Modified:
!
!    05 June 2012
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input, integer I, the number whose square root is desired.
!
!    Output, double precision R8_SQRT_I4, the value of sqrt(I).
!
      implicit none

      integer i
      double precision r8_sqrt_i4

      r8_sqrt_i4 = sqrt ( dble ( i ) )

      return
      end
      subroutine r8_swap ( x, y )

c*********************************************************************72
c
cc R8_SWAP switches two R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    30 November 1998
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input/output, double precision X, Y.  On output, the values of X and
c    Y have been interchanged.
c
      implicit none

      double precision x
      double precision y
      double precision z

      z = x
      x = y
      y = z

      return
      end
      subroutine r8_swap3 ( x, y, z )

c*********************************************************************72
c
cc R8_SWAP3 swaps three R8's.
c
c  Example:
c
c    Input:
c
c      X = 1, Y = 2, Z = 3
c
c    Output:
c
c      X = 2, Y = 3, Z = 1
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    08 June 2000
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input/output, double precision X, Y, Z, three values to be swapped.
c
      implicit none

      double precision w
      double precision x
      double precision y
      double precision z

      w = x
      x = y
      y = z
      z = w

      return
      end
      function r8_tand ( degrees )

c*********************************************************************72
c
cc R8_TAND returns the tangent of an angle given in degrees.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    27 July 2014
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision DEGREES, the angle in degrees.
c
c    Output, double precision R8_TAND, the tangent of the angle.
c
      implicit none

      double precision degrees
      double precision r8_pi
      parameter ( r8_pi = 3.141592653589793D+00 )
      double precision r8_tand
      double precision radians

      radians = r8_pi * ( degrees / 180.0D+00 )

      r8_tand  = sin ( radians ) / cos ( radians )

      return
      end
      function r8_tiny ( )

c*********************************************************************72
c
cc R8_TINY returns a very small but positive R8.
c
c  Discussion:
c
c    FORTRAN90 provides a built-in routine TINY ( X ) that
c    is more suitable for this purpose, returning the smallest positive
c    but normalized real number.
c
c    This routine does NOT try to provide an accurate value for TINY.
c    Instead, it simply returns a "reasonable" value, that is, a rather
c    small, but representable, real number.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    31 December 2007
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Output, double precision R8_TINY, a "tiny" value.
c
      implicit none

      double precision r8_tiny

      r8_tiny = 1.0D-30

      return
      end
      subroutine r8_to_r8_discrete ( r, rmin, rmax, nr, rd )

c*********************************************************************72
c
cc R8_TO_R8_DISCRETE maps R to RD in [RMIN, RMAX] with NR possible values.
c
c  Formula:
c
c    if ( R < RMIN ) then
c      RD = RMIN
c    else if ( RMAX < R ) then
c      RD = RMAX
c    else
c      T = nint ( ( NR - 1 ) * ( R - RMIN ) / ( RMAX - RMIN ) )
c      RD = RMIN + T * ( RMAX - RMIN ) / real ( NR - 1 )
c
c    In the special case where NR = 1, when
c
c      XD = 0.5 * ( RMAX + RMIN )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    02 June 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision R, the number to be converted.
c
c    Input, double precision RMAX, RMIN, the maximum and minimum
c    values for RD.
c
c    Input, integer NR, the number of allowed values for XD.
c    NR should be at least 1.
c
c    Output, double precision RD, the corresponding discrete value.
c
      implicit none

      integer f
      integer nr
      double precision r
      double precision rd
      double precision rmax
      double precision rmin
c
c  Check for errors.
c
      if ( nr .lt. 1 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8_TO_R8_DISCRETE - Fatal error!'
        write ( *, '(a,i8)' ) '  NR = ', nr
        write ( *, '(a)' ) '  but NR must be at least 1.'
        stop 1
      end if

      if ( nr .eq. 1 ) then
        rd = 0.5D+00 * ( rmin + rmax )
        return
      end if

      if ( rmax .eq. rmin ) then
        rd = rmax
        return
      end if

      f = nint ( dble ( nr ) * ( rmax - r ) / ( rmax - rmin ) )
      f = max ( f, 0 )
      f = min ( f, nr )

      rd = ( dble (      f ) * rmin
     &     + dble ( nr - f ) * rmax )
     &     / dble ( nr     )

      return
      end
      subroutine r8_to_dhms ( r, d, h, m, s )

c*********************************************************************72
c
cc R8_TO_DHMS converts decimal days into days, hours, minutes, seconds.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    02 June 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision R, a decimal number representing a time
c    period measured in days.
c
c    Output, integer D, H, M, S, the equivalent number of days,
c    hours, minutes and seconds.
c
      implicit none

      integer d
      integer h
      integer m
      double precision r
      double precision r_copy
      integer s

      r_copy = abs ( r )

      d = int ( r_copy )

      r_copy = r_copy - d
      r_copy = 24.0D+00 * r_copy
      h = int ( r_copy )

      r_copy = r_copy - h
      r_copy = 60.0D+00 * r_copy
      m = int ( r_copy )

      r_copy = r_copy - m
      r_copy = 60.0D+00 * r_copy
      s = int ( r_copy )

      if ( r .lt. 0.0D+00 ) then
        d = -d
        h = -h
        m = -m
        s = -s
      end if

      return
      end
      subroutine r8_to_i4 ( xmin, xmax, x, ixmin, ixmax, ix )

c*********************************************************************72
c
cc R8_TO_I4 maps X in [XMIN, XMAX] to integer IX in [IXMIN, IXMAX].
c
c  Formula:
c
c    IX := IXMIN + ( IXMAX - IXMIN ) * ( X - XMIN ) / ( XMAX - XMIN )
c    IX := min ( IX, max ( IXMIN, IXMAX ) )
c    IX := max ( IX, min ( IXMIN, IXMAX ) )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    19 April 2014
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision XMIN, XMAX, the range.  XMAX and
c    XMIN must not be equal.  It is not necessary that XMIN be less than XMAX.
c
c    Input, double precision X, the number to be converted.
c
c    Input, integer IXMIN, IXMAX, the allowed range of the output
c    variable.  IXMAX corresponds to XMAX, and IXMIN to XMIN.
c    It is not necessary that IXMIN be less than IXMAX.
c
c    Output, integer IX, the value in the range [IXMIN,IXMAX] that
c    corresponds to X.
c
      implicit none

      integer ix
      integer ixmax
      integer ixmin
      double precision temp
      double precision x
      double precision xmax
      double precision xmin

      if ( xmax .eq. xmin ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8_TO_I4 - Fatal error!'
        write ( *, '(a)' ) '  XMAX = XMIN, making a zero divisor.'
        write ( *, '(a,g14.6)' ) '  XMAX = ', xmax
        write ( *, '(a,g14.6)' ) '  XMIN = ', xmin
        stop 1
      end if

      temp =
     &    ( ( xmax - x        ) * dble ( ixmin )
     &    + (        x - xmin ) * dble ( ixmax ) )
     &    / ( xmax     - xmin )

      if ( 0.0D+00 .le. temp ) then
        temp = temp + 0.5D+00
      else
        temp = temp - 0.5D+00
      end if

      ix = int ( temp )

      return
      end
      function r8_uniform_01 ( seed )

c*********************************************************************72
c
cc R8_UNIFORM_01 returns a pseudorandom R8 scaled to [0,1].
c
c  Discussion:
c
c    This routine implements the recursion
c
c      seed = 16807 * seed mod ( 2^31 - 1 )
c      r8_uniform_01 = seed / ( 2^31 - 1 )
c
c    The integer arithmetic never requires more than 32 bits,
c    including a sign bit.
c
c    If the initial seed is 12345, then the first three computations are
c
c      Input     Output      R8_UNIFORM_01
c      SEED      SEED
c
c         12345   207482415  0.096616
c     207482415  1790989824  0.833995
c    1790989824  2035175616  0.947702
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    11 August 2004
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Paul Bratley, Bennett Fox, Linus Schrage,
c    A Guide to Simulation,
c    Springer Verlag, pages 201-202, 1983.
c
c    Pierre L'Ecuyer,
c    Random Number Generation,
c    in Handbook of Simulation,
c    edited by Jerry Banks,
c    Wiley Interscience, page 95, 1998.
c
c    Bennett Fox,
c    Algorithm 647:
c    Implementation and Relative Efficiency of Quasirandom
c    Sequence Generators,
c    ACM Transactions on Mathematical Software,
c    Volume 12, Number 4, pages 362-376, 1986.
c
c    Peter Lewis, Allen Goodman, James Miller,
c    A Pseudo-Random Number Generator for the System/360,
c    IBM Systems Journal,
c    Volume 8, pages 136-143, 1969.
c
c  Parameters:
c
c    Input/output, integer SEED, the "seed" value, which should NOT be 0.
c    On output, SEED has been updated.
c
c    Output, double precision R8_UNIFORM_01, a new pseudorandom variate,
c    strictly between 0 and 1.
c
      implicit none

      integer i4_huge
      parameter ( i4_huge = 2147483647 )
      integer k
      double precision r8_uniform_01
      integer seed

      if ( seed .eq. 0 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8_UNIFORM_01 - Fatal error!'
        write ( *, '(a)' ) '  Input value of SEED = 0.'
        stop 1
      end if

      k = seed / 127773

      seed = 16807 * ( seed - k * 127773 ) - k * 2836

      if ( seed .lt. 0 ) then
        seed = seed + i4_huge
      end if

      r8_uniform_01 = dble ( seed ) * 4.656612875D-10

      return
      end
      function r8_uniform_ab ( a, b, seed )

c*********************************************************************72
c
cc R8_UNIFORM_AB returns a pseudorandom R8 scaled to [A,B].
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    06 January 2006
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision A, B, the limits of the interval.
c
c    Input/output, integer SEED, the "seed" value, which should NOT be 0.
c    On output, SEED has been updated.
c
c    Output, double precision R8_UNIFORM_AB, a number strictly between A and B.
c
      implicit none

      double precision a
      double precision b
      integer i4_huge
      parameter ( i4_huge = 2147483647 )
      integer k
      double precision r8_uniform_ab
      integer seed

      if ( seed .eq. 0 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8_UNIFORM_AB - Fatal error!'
        write ( *, '(a)' ) '  Input value of SEED = 0.'
        stop 1
      end if

      k = seed / 127773

      seed = 16807 * ( seed - k * 127773 ) - k * 2836

      if ( seed .lt. 0 ) then
        seed = seed + i4_huge
      end if

      r8_uniform_ab = a + ( b - a ) * dble ( seed ) * 4.656612875D-10

      return
      end
      subroutine r8_unswap3 ( x, y, z )

c*********************************************************************72
c
cc R8_UNSWAP3 unswaps three R8's.
c
c  Example:
c
c    Input:
c
c      X = 2, Y = 3, Z = 1
c
c    Output:
c
c      X = 1, Y = 2, Z = 3
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    02 June 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input/output, double precision X, Y, Z, three values to be swapped.
c
      implicit none

      double precision w
      double precision x
      double precision y
      double precision z

      w = z
      z = y
      y = x
      x = w

      return
      end
      function r8_walsh_1d ( x, digit )

c*********************************************************************72
c
cc R8_WALSH_1D evaluates the Walsh function.
c
c  Discussion:
c
c    Consider the binary representation of X, and number the digits
c    in descending order, from leading to lowest, with the units digit
c    being numbered 0.
c
c    The Walsh function W(J)(X) is equal to the J-th binary digit of X.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    02 June 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision X, the argument of the Walsh function.
c
c    Input, integer DIGIT, the index of the Walsh function.
c
c    Output, double precision R8_WALSH_1D, the value of the Walsh function.
c
      implicit none

      integer digit
      integer n
      double precision r8_walsh_1d
      double precision x
      double precision x_copy
c
c  Hide the effect of the sign of X.
c
      x_copy = abs ( x )
c
c  If DIGIT is positive, divide by 2 DIGIT times.
c  If DIGIT is negative, multiply by 2 (-DIGIT) times.
c
      x_copy = x_copy / 2.0D+00**digit
c
c  Make it an integer.
c  Because it's positive, and we're using INT, we don't change the
c  units digit.
c
      n = int ( x_copy )
c
c  Is the units digit odd or even?
c
      if ( mod ( n, 2 ) .eq. 0 ) then
        r8_walsh_1d = 0.0D+00
      else
        r8_walsh_1d = 1.0D+00
      end if

      return
      end
      function r8_wrap ( r, rlo, rhi )

c*********************************************************************72
c
cc R8_WRAP forces an R8 to lie between given limits by wrapping.
c
c  Discussion:
c
c    An R8 is a double precision value.
c
c  Example:
c
c    RLO = 4.0, RHI = 8.0
c
c     R  Value
c
c    -2     8
c    -1     4
c     0     5
c     1     6
c     2     7
c     3     8
c     4     4
c     5     5
c     6     6
c     7     7
c     8     8
c     9     4
c    10     5
c    11     6
c    12     7
c    13     8
c    14     4
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    04 July 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision R, a value.
c
c    Input, double precision RLO, RHI, the desired bounds.
c
c    Output, double precision R8_WRAP, a "wrapped" version of the value.
c
      implicit none

      integer n
      double precision r
      double precision r8_wrap
      double precision rhi
      double precision rhi2
      double precision rlo
      double precision rlo2
      double precision rwide
      double precision value
c
c  Guarantee RLO2 .lt. RHI2.
c
      rlo2 = min ( rlo, rhi )
      rhi2 = max ( rlo, rhi )
c
c  Find the width.
c
      rwide = rhi2 - rlo2
c
c  Add enough copies of (RHI2-RLO2) to R so that the
c  result ends up in the interval RLO2 - RHI2.
c
      if ( rwide .eq. 0.0D+00 ) then
        value = rlo
      else if ( r .lt. rlo2 ) then
        n = int ( ( rlo2 - r ) / rwide ) + 1
        value = r + n * rwide
        if ( value .eq. rhi ) then
          value = rlo
        end if
      else
        n = int ( ( r - rlo2 ) / rwide )
        value = r - n * rwide
        if ( value .eq. rlo ) then
          value = rhi
        end if
      end if

      r8_wrap = value

      return
      end
      function r82_dist_l2 ( a1, a2 )

c*********************************************************************72
c
cc R82_DIST_L2 returns the L2 distance between a pair of R82's.
c
c  Discussion:
c
c    An R82 is a vector of type R8, with two entries.
c
c    The vector L2 norm is defined as:
c
c      sqrt ( sum ( 1 <= I <= N ) A(I) * A(I) ).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    03 September 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision A1(2), A2(2), the vectors.
c
c    Output, double precision R82_DIST_L2, the L2 norm of the distance
c    between A1 and A2.
c
      implicit none

      integer dim_num
      parameter ( dim_num = 2 )

      double precision a1(dim_num)
      double precision a2(dim_num)
      double precision r82_dist_l2

      r82_dist_l2 = sqrt ( 
     &  ( a1(1) - a2(1) ) ** 2 + ( a1(2) - a2(2) ) ** 2 )

      return
      end
      function r82_eq ( a1, a2 )

c*********************************************************************72
c
cc R82_EQ .eq. ( A1 .eq. A2 ) for two R82's.
c
c  Discussion:
c
c    An R82 is a vector of type R8, with two entries.
c
c    The comparison is lexicographic.
c
c    A1 .eq. A2  <=>  A1(1) .eq. A2(1) and A1(2) .eq. A2(2).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    04 September 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision A1(2), A2(2), two R82 vectors to be compared.
c
c    Output, logical R82_EQ, is TRUE if and only if A1 .eq. A2.
c
      implicit none

      integer dim_num
      parameter ( dim_num = 2 )

      double precision a1(dim_num)
      double precision a2(dim_num)
      integer i
      logical r82_eq

      r82_eq = .true.

      do i = 1, dim_num
        if ( a1(i) .ne. a2(i) ) then
          r82_eq = .false.
          return
        end if
      end do

      return
      end
      function r82_ge ( a1, a2 )

c*********************************************************************72
c
cc R82_GE .eq. ( A1 >= A2 ) for two R82's.
c
c  Discussion:
c
c    An R82 is a vector of type R8, with two entries.
c
c    The comparison is lexicographic.
c
c    A1 >= A2  <=>  A1(1) > A2(1) or ( A1(1) .eq. A2(1) and A1(2) >= A2(2) ).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    03 September 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision A1(2), A2(2), R82 vectors to be compared.
c
c    Output, logical R92_GE, is TRUE if and only if A1 >= A2.
c
      implicit none

      integer dim_num
      parameter ( dim_num = 2 )

      double precision a1(dim_num)
      double precision a2(dim_num)
      integer i
      logical r82_ge

      r82_ge = .true.

      do i = 1, dim_num

        if ( a2(i) .lt. a1(i) ) then
          r82_ge = .true.
          return
        else if ( a1(i) .lt. a2(i) ) then
          r82_ge = .false.
          return
        end if

      end do

      return
      end
      function r82_gt ( a1, a2 )

c*********************************************************************72
c
cc R82_GT .eq. ( A1 > A2 ) for two R82's.
c
c  Discussion:
c
c    An R82 is a vector of type R2, with two entries.
c
c    The comparison is lexicographic.
c
c    A1 > A2  <=>  A1(1) > A2(1) or ( A1(1) .eq. A2(1) and A1(2) > A2(2) ).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    03 September 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision A1(2), A2(2), R82 vectors to be compared.
c
c    Output, logical R82_GT, is TRUE if and only if A1 > A2.
c
      implicit none

      integer dim_num
      parameter ( dim_num = 2 )

      double precision a1(dim_num)
      double precision a2(dim_num)
      integer i
      logical r82_gt

      r82_gt = .false.

      do i = 1, dim_num

        if ( a2(i) .lt. a1(i) ) then
          r82_gt = .true.
          return
        else if ( a1(i) .lt. a2(i) ) then
          r82_gt = .false.
          return
        end if

      end do

      return
      end
      function r82_le ( a1, a2 )

c*********************************************************************72
c
cc R82_LE .eq. ( A1 <= A2 ) for two R82's.
c
c  Discussion:
c
c    An R82 is a vector of type R8, with two entries.
c
c    The comparison is lexicographic.
c
c    A1 <= A2  <=>  A1(1) < A2(1) or ( A1(1) .eq. A2(1) and A1(2) <= A2(2) ).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    03 September 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision A1(2), A2(2), R82 vectors to be compared.
c
c    Output, logical R82_LE, is TRUE if and only if A1 <= A2.
c
      implicit none

      integer dim_num
      parameter ( dim_num = 2 )

      double precision a1(dim_num)
      double precision a2(dim_num)
      integer i
      logical r82_le

      r82_le = .true.

      do i = 1, dim_num

        if ( a1(i) .lt. a2(i) ) then
          r82_le = .true.
          return
        else if ( a2(i) .lt. a1(i) ) then
          r82_le = .false.
          return
        end if

      end do

      return
      end
      function r82_lt ( a1, a2 )

c*********************************************************************72
c
cc R82_LT .eq. ( A1 < A2 ) for two R82's.
c
c  Discussion:
c
c    An R82 is a vector of type R8, with two entries.
c
c    The comparison is lexicographic.
c
c    A1 < A2  <=>  A1(1) < A2(1) or ( A1(1) .eq. A2(1) and A1(2) < A2(2) ).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    03 September 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision A1(2), A2(2), R82 vectors to be compared.
c
c    Output, logical R82_LT, is TRUE if and only if A1 < A2.
c
      implicit none

      integer dim_num
      parameter ( dim_num = 2 )

      double precision a1(dim_num)
      double precision a2(dim_num)
      integer i
      logical r82_lt

      r82_lt = .false.

      do i = 1, dim_num

        if ( a1(i) .lt. a2(i) ) then
          r82_lt = .true.
          return
        else if ( a2(i) .lt. a1(i) ) then
          r82_lt = .false.
          return
        end if

      end do

      return
      end
      function r82_ne ( a1, a2 )

c*********************************************************************72
c
cc R82_NE .eq. ( A1 /= A2 ) for two R82's.
c
c  Discussion:
c
c    An R82 is a vector of type R8, with two entries.
c
c    The comparison is lexicographic.
c
c    A1 /= A2  <=>  A1(1) /= A2(1) or A1(2) /= A2(2).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    03 September 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision A1(2), A2(2), R82 vectors to be compared.
c
c    Output, logical R82_NE, is TRUE if and only if A1 /= A2.
c
      implicit none

      integer dim_num
      parameter ( dim_num = 2 )

      double precision a1(dim_num)
      double precision a2(dim_num)
      integer i
      logical r82_ne

      r82_ne = .false.
      do i = 1, dim_num
        if ( a1(i) .ne. a2(i) ) then
          r82_ne = .true.
          return
        end if
      end do

      return
      end
      function r82_norm ( a )

c*********************************************************************72
c
cc R82_NORM returns the Euclidean norm of an R82.
c
c  Discussion:
c
c    An R82 is a vector of type R8, with two entries.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    03 September 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision A(2), the vector.
c
c    Output, double precision R82_NORM, the norm.
c
      implicit none

      double precision a(2)
      double precision r82_norm

      r82_norm = sqrt ( a(1) * a(1) + a(2) * a(2) )

      return
      end
      subroutine r82_normalize ( a )

c*********************************************************************72
c
cc R82_NORMALIZE Euclidean normalizes an R82.
c
c  Discussion:
c
c    An R82 is a vector of type R8, with two entries.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    03 September 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input/output, double precision A(2), the components of the vector.
c
      implicit none

      double precision a(2)
      double precision norm

      norm = sqrt ( a(1) * a(1) + a(2) * a(2) )

      if ( norm .ne. 0.0D+00 ) then
        a(1) = a(1) / norm
        a(2) = a(2) / norm
      end if

      return
      end
      subroutine r82_print ( a, title )

c*********************************************************************72
c
cc R82_PRINT prints an R82.
c
c  Discussion:
c
c    An R82 is a vector of type R8, with two entries.
c
c    A format is used which suggests a coordinate pair:
c
c  Example:
c
c    Center : ( 1.23, 7.45 )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    03 September 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision A(2), the coordinates of the vector.
c
c    Input, character * ( * ) TITLE, a title.
c
      implicit none

      double precision a(2)
      character * ( * ) title

      if ( 0 .lt. len_trim ( title ) ) then
        write ( *, '( 2x, a, a4, g14.6, a1, g14.6, a1 )' ) 
     &    trim ( title ), ' : (', a(1), ',', a(2), ')'
      else
        write ( *, '( 2x, a1, g14.6, a1, g14.6, a1 )' ) 
     &    '(', a(1), ',', a(2), ')'

      end if

      return
      end
      subroutine r82_swap ( x, y )

c*********************************************************************72
c
cc R82_SWAP swaps two R82 values.
c
c  Discussion:
c
c    An R82 is a vector of type R8, with two entries.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    03 September 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input/output, double precision X(2), Y(2).  On output, the values of X and
c    Y have been interchanged.
c
      implicit none

      integer dim_num
      parameter ( dim_num = 2 )

      integer i
      double precision x(dim_num)
      double precision y(dim_num)
      double precision t

      do i = 1, dim_num
        t    = x(i)
        x(i) = y(i)
        y(i) = t
      end do

      return
      end
      subroutine r82_uniform_ab ( a, b, seed, r )

c*********************************************************************72
c
cc R82_UNIFORM returns a pseudorandom R82 scaled to [A,B].
c
c  Discussion:
c
c    An R82 is a vector of type R8, with two entries.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    03 September 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision A, B, the minimum and maximum values.
c
c    Input/output, integer SEED, a seed for the random number
c    generator.
c
c    Output, double precision R(2), the randomly chosen value.
c
      implicit none

      integer dim_num
      parameter ( dim_num = 2 )

      double precision a
      double precision b
      double precision r(dim_num)
      double precision r8_uniform_ab
      integer i
      integer seed

      do i = 1, dim_num
        r(i) = r8_uniform_ab ( a, b, seed )
      end do

      return
      end
      subroutine r82col_print_part ( n, a, max_print, title )

c*********************************************************************72
c
cc R82COL_PRINT_PART prints "part" of an R82COL.
c
c  Discussion:
c
c    An R82COL is an (N,2) array of R8's.
c
c    The user specifies MAX_PRINT, the maximum number of lines to print.
c
c    If N, the size of the vector, is no more than MAX_PRINT, then
c    the entire vector is printed, one entry per line.
c
c    Otherwise, if possible, the first MAX_PRINT-2 entries are printed,
c    followed by a line of periods suggesting an omission,
c    and the last entry.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    10 April 2015
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries of the vector.
c
c    Input, double precision A(N,2), the vector to be printed.
c
c    Input, integer MAX_PRINT, the maximum number of lines
c    to print.
c
c    Input, character * ( * ) TITLE, a title.
c
      implicit none

      integer n

      double precision a(n,2)
      integer i
      integer max_print
      character * ( * ) title

      if ( max_print .le. 0 ) then
        return
      end if

      if ( n .le. 0 ) then
        return
      end if

      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) trim ( title )
      write ( *, '(a)' ) ' '

      if ( n .le. max_print ) then

        do i = 1, n
          write ( *, '(2x,i8,a,1x,g14.6,2x,g14.6)' ) 
     &      i, ':', a(i,1), a(i,2)
        end do

      else if ( 3 .le. max_print ) then

        do i = 1, max_print - 2
          write ( *, '(2x,i8,a,1x,g14.6,2x,g14.6)' ) 
     &      i, ':', a(i,1), a(i,2)
        end do
        write ( *, '(a)' ) '  ........  ..............  ..............'
        i = n
        write ( *, '(2x,i8,a,1x,g14.6,2x,g14.6)' ) 
     &    i, ':', a(i,1), a(i,2)

      else

        do i = 1, max_print - 1
          write ( *, '(2x,i8,a,1x,g14.6,2x,g14.6)' ) 
     &      i, ':', a(i,1), a(i,2)
        end do
        i = max_print
        write ( *, '(2x,i8,a,1x,g14.6,2x,g14.6,2x,a)' ) 
     &    i, ':', a(i,1), a(i,2), '...more entries...'

      end if

      return
      end
      subroutine r82poly2_print ( a, b, c, d, e, f )

c*********************************************************************72
c
cc R82POLY2_PRINT prints a second order polynomial in two variables.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    04 September 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision A, B, C, D, E, F, the coefficients.
c
      implicit none

      double precision a
      double precision b
      double precision c
      double precision d
      double precision e
      double precision f

      write ( *, 
     &  '( 2x, f8.4, '' * x^2 + '', f8.4, '' * y^2 + '', 
     &  f8.4, '' * xy  + '' )' ) 
     &  a, b, c

      write ( *, 
     &  '( 2x, f8.4, '' * x + '', f8.4, '' * y + '', 
     &  f8.4, '' = 0 '' )' ) d, e, f

      return
      end
      subroutine r82poly2_type ( a, b, c, d, e, f, type )

c*********************************************************************72
c
cc R82POLY2_TYPE analyzes a second order polynomial in two variables.
c
c  Discussion:
c
c    The polynomial has the form
c
c      A x^2 + B y^2 + C xy + Dx + Ey + F = 0
c
c    The possible types of the solution set are:
c
c     1: a hyperbola;
c        9x^2 -  4y^2       -36x - 24y -  36 = 0
c     2: a parabola;
c        4x^2 +  1y^2 - 4xy + 3x -  4y +   1 = 0;
c     3: an ellipse;
c        9x^2 + 16y^2       +36x - 32y -  92 = 0;
c     4: an imaginary ellipse (no real solutions);
c         x^2 +   y^2       - 6x - 10y + 115 = 0;
c     5: a pair of intersecting lines;
c                        xy + 3x -   y -   3 = 0
c     6: one point;
c         x^2 +  2y^2       - 2x + 16y +  33 = 0;
c     7: a pair of distinct parallel lines;
c                 y^2            -  6y +   8 = 0
c     8: a pair of imaginary parallel lines (no real solutions);
c                 y^2            -  6y +  10 = 0
c     9: a pair of coincident lines.
c                 y^2            -  2y +   1 = 0
c    10: a single line;
c                             2x -   y +   1 = 0;
c    11; all space;
c                                          0 = 0;
c    12; no solutions;
c                                          1 = 0;
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    04 September 2011
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Daniel Zwillinger, editor,
c    CRC Standard Mathematical Tables and Formulae,
c    CRC Press, 30th Edition, 1996, pages 282-284.
c
c  Parameters:
c
c    Input, double precision A, B, C, D, E, F, the coefficients.
c
c    Output, integer TYPE, indicates the type of the solution set.
c
      implicit none

      double precision a
      double precision b
      double precision c
      double precision d
      double precision delta
      double precision e
      double precision f
      double precision j
      double precision k
      integer type
c
c  Handle the degenerate case.
c
      if ( a .eq. 0.0D+00 .and. 
     &     b .eq. 0.0D+00 .and. 
     &     c .eq. 0.0D+00 ) then
        if ( d .eq. 0.0D+00 .and. e .eq. 0.0D+00 ) then
          if ( f .eq. 0.0D+00 ) then
            type = 11
          else
            type = 12
          end if
        else
          type = 10
        end if
        return
      end if

      delta = 
     &    8.0D+00 * a * b * f 
     &  + 2.0D+00 * c * e * d 
     &  - 2.0D+00 * a * e * e 
     &  - 2.0D+00 * b * d * d 
     &  - 2.0D+00 * f * c * c

      j = 4.0D+00 * a * b - c * c

      if ( delta .ne. 0.0D+00 ) then
        if ( j .lt. 0.0D+00 ) then
          type = 1
        else if ( j .eq. 0.0D+00 ) then
          type = 2
        else if ( 0.0D+00 .lt. j ) then
          if ( sign ( 1.0D+00, delta ) .ne. 
     &         sign ( 1.0D+00, ( a + b ) ) ) then
            type = 3
          else if ( sign ( 1.0D+00, delta ) .eq. 
     &              sign ( 1.0D+00, ( a + b ) ) ) then
            type = 4
          end if
        end if
      else if ( delta .eq. 0.0D+00 ) then
        if ( j .lt. 0.0D+00 ) then
          type = 5
        else if ( 0.0D+00 .lt. j ) then
          type = 6
        else if ( j .eq. 0.0D+00 ) then

          k = 4.0D+00 * ( a + b ) * f - d * d - e * e

          if ( k .lt. 0.0D+00 ) then
            type = 7
          else if ( 0.0D+00 .lt. k ) then
            type = 8
          else if ( k .eq. 0.0D+00 ) then
            type = 9
          end if

        end if
      end if

      return
      end
      subroutine r82poly2_type_print ( type )

c*********************************************************************72
c
cc R82POLY2_TYPE_PRINT prints the meaning of the output from R82POLY2_TYPE.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    04 September 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer TYPE, the type index returned by R82POLY2_TYPE.
c
      implicit none

      integer type

      if ( type .eq. 1 ) then
        write ( *, '(a)' ) '  The set of solutions forms a hyperbola.'
      else if ( type .eq. 2 ) then
        write ( *, '(a)' ) '  The set of solutions forms a parabola.'
      else if ( type .eq. 3 ) then
        write ( *, '(a)' ) '  The set of solutions forms an ellipse.'
      else if ( type .eq. 4 ) then
        write ( *, '(a)' ) 
     &    '  The set of solutions forms an imaginary ellipse.'
        write ( *, '(a)' ) '  (There are no real solutions).'
      else if ( type .eq. 5 ) then
        write ( *, '(a)' ) 
     &    '  The set of solutions forms a pair of intersecting lines.'
      else if ( type .eq. 6 ) then
        write ( *, '(a)' ) '  The set of solutions is a single point.'
      else if ( type .eq. 7 ) then
        write ( *, '(a)' ) 
     &    '  The set of solutions form a pair of '
        write ( *, '(a)' ) '  distinct parallel lines.'
      else if ( type .eq. 8 ) then
        write ( *, '(a)' ) 
     &    '  The set of solutions forms a pair of '
        write ( *, '(a)' ) '  imaginary parallel lines.'
        write ( *, '(a)' ) '  (There are no real solutions).'
      else if ( type .eq. 9 ) then
        write ( *, '(a)' ) 
     &    '  The set of solutions forms a pair of coincident lines.'
      else if ( type .eq. 10 ) then
        write ( *, '(a)' ) '  The set of solutions forms a single line.'
      else if ( type .eq. 11 ) then
        write ( *, '(a)' ) '  The set of solutions is all space.'
      else if ( type .eq. 12 ) then
        write ( *, '(a)' ) '  The set of solutions is empty.'
      else
        write ( *, '(a)' ) '  This type index is unknown.'
      end if

      return
      end
      subroutine r82row_max ( n, a, amax )

c*********************************************************************72
c
cc R82ROW_MAX returns the maximum value in an R82ROW.
c
c  Discussion:
c
c    An R82ROW is a (2,N) array of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    04 September 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the array.
c
c    Input, double precision A(2,N), the array.
c
c    Output, double precision AMAX(2); the largest entries in each row.
c
      implicit none

      integer n

      double precision a(2,n)
      double precision amax(2)
      integer i
      integer j
      double precision r8_huge
      parameter ( r8_huge = 1.79769313486231571D+308 )

      do i = 1, 2
        amax(i) = - r8_huge
        do j = 1, n
          amax(i) = max ( amax(i), a(i,j) )
        end do
      end do

      return
      end
      subroutine r82row_min ( n, a, amin )

c*********************************************************************72
c
cc R82ROW_MIN returns the minimum value in an R82ROW.
c
c  Discussion:
c
c    An R82ROW is a (2,N) array of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    11 September 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the array.
c
c    Input, double precision A(2,N), the array.
c
c    Output, double precision AMIN(2); the smallest entries in each row.
c
      implicit none

      integer n

      double precision a(2,n)
      double precision amin(2)
      integer i
      integer j
      double precision r8_huge
      parameter ( r8_huge = 1.79769313486231571D+308 )

      do i = 1, 2
        amin(i) = r8_huge
        do j = 1, n
          amin(i) = min ( amin(i), a(i,j) )
        end do
      end do

      return
      end
      subroutine r82row_order_type ( n, a, order )

c*********************************************************************72
c
cc R82ROW_ORDER_TYPE finds the order type of an R82ROW.
c
c  Discussion:
c
c    An R82ROW is a (2,N) array of R8's.
c
c    The dictionary or lexicographic ordering is used.
c
c    (X1,Y1) .lt. (X2,Y2)  <=>  X1 .lt. X2 or ( X1 = X2 and Y1 .lt. Y2).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    11 September 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries of the array.
c
c    Input, double precision A(2,N), the array to be checked.
c
c    Output, integer ORDER, order indicator:
c    -1, no discernable order;
c    0, all entries are equal;
c    1, ascending order;
c    2, strictly ascending order;
c    3, descending order;
c    4, strictly descending order.
c
      implicit none

      integer n
      integer dim_num
      parameter ( dim_num = 2 )

      double precision a(dim_num,n)
      integer i
      integer order
c
c  Search for the first value not equal to A(1,1).
c
      i = 1

10    continue

        i = i + 1

        if ( n .lt. i ) then
          order = 0
          return
        end if

        if ( 
     &       a(1,1) .lt. a(1,i) .or. 
     &     ( a(1,1) .eq. a(1,i) .and. a(2,1) .lt. a(2,i) ) 
     &     ) then

          if ( i .eq. 2 ) then
            order = 2
          else
            order = 1
          end if

          go to 20

        else if ( 
     &      a(1,i) .lt. a(1,1)  .or. 
     &    ( a(1,i) .eq. a(1,1) .and. a(2,i) .lt. a(2,1) ) 
     &    ) then

          if ( i .eq. 2 ) then
            order = 4
          else
            order = 3
          end if

          go to 20

        end if

      go to 10

20    continue
c
c  Now we have a "direction".  Examine subsequent entries.
c
30    continue

        i = i + 1
        if ( n .lt. i ) then
          go to 40
        end if

        if ( order .eq. 1 ) then

          if ( 
     &        a(1,i) .lt. a(1,i-1) .or. 
     &      ( a(1,i) .eq. a(1,i-1) .and. a(2,i) .lt. a(2,i-1) ) 
     &      ) then
            order = -1
            go to 40
          end if

        else if ( order .eq. 2 ) then

          if ( 
     &        a(1,i) .lt. a(1,i-1) .or. 
     &      ( a(1,i) .eq. a(1,i-1) .and. a(2,i) .lt. a(2,i-1) ) 
     &      ) then
            order = -1
            go to 40
          else if ( 
     &       a(1,i) .eq. a(1,i-1) .and. a(2,i) .eq. a(2,i-1) ) then
            order = 1
          end if

        else if ( order .eq. 3 ) then

          if ( 
     &        a(1,i-1) .lt. a(1,i) .or. 
     &      ( a(1,i-1) .eq. a(1,i) .and. a(2,i-1) .lt. a(2,i) ) 
     &      ) then
            order = -1
            go to 40
          end if

        else if ( order .eq. 4 ) then

          if ( 
     &        a(1,i-1) .lt. a(1,i) .or. 
     &      ( a(1,i-1) .eq. a(1,i) .and. a(2,i-1) .lt. a(2,i) ) 
     &      ) then
            order = -1
            go to 40
          else if ( a(1,i) .eq. a(1,i-1) .and. 
     &              a(2,i) .eq. a(2,i-1) ) then
            order = 3
          end if

        end if

      go to 30

40    continue

      return
      end
      subroutine r82row_part_quick_a ( n, a, l, r )

c*********************************************************************72
c
cc R82ROW_PART_QUICK_A reorders an R82ROW as part of a quick sort.
c
c  Discussion:
c
c    An R82ROW is a (2,N) array of R8's.
c
c    The routine reorders the entries of A.  Using A(1:2,1) as a
c    key, all entries of A that are less than or equal to the key will
c    precede the key, which precedes all entries that are greater than the key.
c
c  Example:
c
c    Input:
c
c      N = 8
c
c      A = ( (2,4), (8,8), (6,2), (0,2), (10,6), (10,0), (0,6), (4,8) )
c
c    Output:
c
c      L = 2, R = 4
c
c      A = ( (0,2), (0,6), (2,4), (8,8), (6,2), (10,6), (10,0), (4,8) )
c             -----------          ----------------------------------
c             LEFT          KEY    RIGHT
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    16 September 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries of A.
c
c    Input/output, double precision A(2,N).  On input, the array to be checked.
c    On output, A has been reordered as described above.
c
c    Output, integer L, R, the indices of A that define the three
c    segments.  Let KEY = the input value of A(1:2,1).  Then
c    I <= L                 A(1:2,I) < KEY;
c         L < I < R         A(1:2,I) = KEY;
c                 R <= I    KEY < A(1:2,I).
c
      implicit none

      integer n
      integer dim_num
      parameter ( dim_num = 2 )

      double precision a(dim_num,n)
      integer i
      integer j
      double precision key(dim_num)
      integer l
      integer m
      integer r
      logical r8vec_eq
      logical r8vec_gt
      logical r8vec_lt

      if ( n .lt. 1 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R82ROW_PART_QUICK_A - Fatal error!'
        write ( *, '(a)' ) '  N < 1.'
        write ( *, '(a,i8)' ) '  N = ', n
        stop 1
      else if ( n .eq. 1 ) then
        l = 0
        r = 2
        return
      end if

      do i = 1, dim_num
        key(i) = a(i,1)
      end do

      m = 1
c
c  The elements of unknown size have indices between L+1 and R-1.
c
      l = 1
      r = n + 1

      do i = 2, n

        if ( r8vec_gt ( dim_num, a(1,l+1), key ) ) then
          r = r - 1
          call r8vec_swap ( dim_num, a(1,r), a(1,l+1) )
        else if ( r8vec_eq ( dim_num, a(1,l+1), key ) ) then
          m = m + 1
          call r8vec_swap ( dim_num, a(1,m), a(1,l+1) )
          l = l + 1
        else if ( r8vec_lt ( dim_num, a(1,l+1), key ) ) then
          l = l + 1
        end if

      end do
c
c  Now shift small elements to the left, and KEY elements to center.
c
      do j = 1, l - m
        do i = 1, dim_num
          a(i,j) = a(i,j+m)
        end do
      end do

      l = l - m

      do j = l + 1, l + m
        do i = 1, dim_num
          a(i,j) = key(i)
        end do
      end do

      return
      end
      subroutine r82row_permute ( n, p, a )

c*********************************************************************72
c
cc R82ROW_PERMUTE permutes an R82ROW in place.
c
c  Discussion:
c
c    An R82ROW is a (2,N) array of R8's.
c
c    The same logic can be used to permute an array of objects of any
c    arithmetic type, or an array of objects of any complexity.  The only
c    temporary storage required is enough to store a single object.  The number
c    of data movements made is N + the number of cycles of order 2 or more,
c    which is never more than N + N/2.
c
c  Example:
c
c    Input:
c
c      N = 5
c      P = (   2,    4,    5,    1,    3 )
c      A = ( 1.0,  2.0,  3.0,  4.0,  5.0 )
c          (11.0, 22.0, 33.0, 44.0, 55.0 )
c
c    Output:
c
c      A    = (  2.0,  4.0,  5.0,  1.0,  3.0 )
c             ( 22.0, 44.0, 55.0, 11.0, 33.0 ).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    03 June 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of objects.
c
c    Input, integer P(N), the permutation.  P(I) = J means
c    that the I-th element of the output array should be the J-th
c    element of the input array.
c
c    Input/output, double precision A(2,N), the array to be permuted.
c
      implicit none

      integer n
      integer dim_num
      parameter ( dim_num = 2 )

      double precision a(dim_num,n)
      double precision a_temp(dim_num)
      integer dim
      integer ierror
      integer iget
      integer iput
      integer istart
      integer p(n)

      call perm_check1 ( n, p )
c
c  Search for the next element of the permutation that has not been used.
c
      do istart = 1, n

        if ( p(istart) .lt. 0 ) then

        else if ( p(istart) .eq. istart ) then

          p(istart) = - p(istart)

        else

          do dim = 1, dim_num
            a_temp(dim) = a(dim,istart)
          end do
          iget = istart
c
c  Copy the new value into the vacated entry.
c
10        continue

            iput = iget
            iget = p(iget)

            p(iput) = - p(iput)

            if ( iget .lt. 1 .or. n .lt. iget ) then
              write ( *, '(a)' ) ' '
              write ( *, '(a)' ) 'R82ROW_PERMUTE - Fatal error!'
              write ( *, '(a)' )
     &          '  A permutation index is out of range.'
              write ( *, '(a,i8,a,i8)' ) '  P(', iput, ') = ', iget
              stop 1
            end if

            if ( iget .eq. istart ) then
              do dim = 1, dim_num
                a(dim,iput) = a_temp(dim)
              end do
              go to 20
            end if

            do dim = 1, dim_num
              a(dim,iput) = a(dim,iget)
            end do

          go to 10

        end if

20      continue

      end do
c
c  Restore the signs of the entries.
c
      do istart = 1, n
        p(istart) = - p(istart)
      end do

      return
      end
      subroutine r82row_print ( n, a, title )

c*********************************************************************72
c
cc R82ROW_PRINT prints an R82ROW.
c
c  Discussion:
c
c    An R82ROW is a (2,N) array of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    28 November 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of components of the vector.
c
c    Input, double precision A(2,N), the R82 vector to be printed.
c
c    Input, character * ( * ) TITLE, a title.
c
      implicit none

      integer n
      integer dim_num
      parameter ( dim_num = 2 )

      double precision a(dim_num,n)
      integer i
      integer j
      character * ( * ) title

      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) trim ( title )
      write ( *, '(a)' ) ' '
      do j = 1, n
        write ( *, '(2x,i8,(5g14.6))' ) j, ( a(i,j), i = 1, dim_num )
      end do

      return
      end
      subroutine r82row_print_part ( n, a, max_print, title )

c*********************************************************************72
c
cc R82ROW_PRINT_PART prints "part" of an R82ROW.
c
c  Discussion:
c
c    An R82ROW is a (2,N) array of R8's.
c
c    The user specifies MAX_PRINT, the maximum number of lines to print.
c
c    If N, the size of the vector, is no more than MAX_PRINT, then
c    the entire vector is printed, one entry per line.
c
c    Otherwise, if possible, the first MAX_PRINT-2 entries are printed,
c    followed by a line of periods suggesting an omission,
c    and the last entry.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    09 November 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries of the vector.
c
c    Input, double precision A(2,N), the vector to be printed.
c
c    Input, integer MAX_PRINT, the maximum number of lines
c    to print.
c
c    Input, character * ( * ) TITLE, a title.
c
      implicit none

      integer n

      double precision a(2,n)
      integer i
      integer max_print
      character * ( * ) title

      if ( max_print .le. 0 ) then
        return
      end if

      if ( n .le. 0 ) then
        return
      end if

      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) trim ( title )
      write ( *, '(a)' ) ' '

      if ( n .le. max_print ) then

        do i = 1, n
          write ( *, '(2x,i8,a,1x,g14.6,2x,g14.6)' ) 
     &      i, ':', a(1,i), a(2,i)
        end do

      else if ( 3 .le. max_print ) then

        do i = 1, max_print - 2
          write ( *, '(2x,i8,a,1x,g14.6,2x,g14.6)' ) 
     &      i, ':', a(1,i), a(2,i)
        end do
        write ( *, '(a)' ) '  ........  ..............  ..............'
        i = n
        write ( *, '(2x,i8,a,1x,g14.6,2x,g14.6)' ) 
     &    i, ':', a(1,i), a(2,i)

      else

        do i = 1, max_print - 1
          write ( *, '(2x,i8,a,1x,g14.6,2x,g14.6)' ) 
     &      i, ':', a(1,i), a(2,i)
        end do
        i = max_print
        write ( *, '(2x,i8,a,1x,g14.6,2x,g14.6,2x,a)' ) 
     &    i, ':', a(1,i), a(2,i), '...more entries...'

      end if

      return
      end
      subroutine r82row_sort_heap_index_a ( n, a, indx )

c*********************************************************************72
c
cc R82ROW_SORT_HEAP_INDEX_A ascending index heaps an R82ROW.
c
c  Discussion:
c
c    An R82ROW is a (2,N) array of R8's.
c
c    The sorting is not actually carried out.  Rather an index array is
c    created which defines the sorting.  This array may be used to sort
c    or index the array, or to sort or index related arrays keyed on the
c    original array.
c
c    Once the index array is computed, the sorting can be carried out
c    "implicitly:
c
c      A(1:2,INDX(1:N)) is sorted,
c
c    or explicitly, by the call
c
c      call r82row_permute ( n, indx, a )
c
c    after which A(1:2,I), I = 1 to N is sorted.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    03 June 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the array.
c
c    Input, double precision A(2,N), an array to be index-sorted.
c
c    Output, integer INDX(N), the sort index.  The
c    I-th element of the sorted array is A(1:2,INDX(I)).
c
      implicit none

      integer dim_num
      parameter ( dim_num = 2 )
      integer n

      double precision a(dim_num,n)
      double precision aval(dim_num)
      integer dim
      integer i
      integer indx(n)
      integer indxt
      integer ir
      integer j
      integer l

      if ( n .lt. 1 ) then
        return
      end if

      do i = 1, n
        indx(i) = i
      end do

      if ( n .eq. 1 ) then
        return
      end if

      l = n / 2 + 1
      ir = n

10    continue

        if ( 1 .lt. l ) then

          l = l - 1
          indxt = indx(l)
          do dim = 1, dim_num
            aval(dim) = a(dim,indxt)
          end do

        else

          indxt = indx(ir)
          do dim = 1, dim_num
            aval(dim) = a(dim,indxt)
          end do
          indx(ir) = indx(1)
          ir = ir - 1

          if ( ir .eq. 1 ) then
            indx(1) = indxt
            go to 30
          end if

        end if

        i = l
        j = l + l

20      continue

        if ( j .le. ir ) then

          if ( j .lt. ir ) then
            if (   a(1,indx(j)) .lt. a(1,indx(j+1)) .or.
     &           ( a(1,indx(j)) .eq. a(1,indx(j+1)) .and.
     &             a(2,indx(j)) .lt. a(2,indx(j+1)) ) ) then
              j = j + 1
            end if
          end if

          if (   aval(1) .lt. a(1,indx(j)) .or.
     &         ( aval(1) .eq. a(1,indx(j)) .and.
     &           aval(2) .lt. a(2,indx(j)) ) ) then
            indx(i) = indx(j)
            i = j
            j = j + j
          else
            j = ir + 1
          end if

          go to 20

        end if

        indx(i) = indxt

      go to 10

30    continue

      return
      end
      subroutine r82row_sort_quick_a ( n, a )

c*********************************************************************72
c
cc R82ROW_SORT_QUICK_A ascending sorts an R82ROW using quick sort.
c
c  Discussion:
c
c    An R82ROW is a (2,N) array of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    16 September 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the array.
c
c    Input/output, double precision A(2,N).
c    On input, the array to be sorted.
c    On output, the array has been sorted.
c
      implicit none

      integer level_max
      parameter ( level_max = 30 )
      integer n
      integer dim_num
      parameter ( dim_num = 2 )

      double precision a(dim_num,n)
      integer base
      integer l_segment
      integer level
      integer n_segment
      integer rsave(level_max)
      integer r_segment

      if ( n .lt. 1 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R82ROW_SORT_QUICK_A - Fatal error!'
        write ( *, '(a)' ) '  N < 1.'
        write ( *, '(a,i8)' ) '  N = ', n
        stop 1
      else if ( n .eq. 1 ) then
        return
      end if

      level = 1
      rsave(level) = n + 1
      base = 1
      n_segment = n

10    continue
c
c  Partition the segment.
c
        call r82row_part_quick_a ( n_segment, a(1,base), l_segment, 
     &    r_segment )
c
c  If the left segment has more than one element, we need to partition it.
c
        if ( 1 .lt. l_segment ) then

          if ( level_max .lt. level ) then
            write ( *, '(a)' ) ' '
            write ( *, '(a)' ) 
     &        'R82ROW_SORT_QUICK_A - Fatal error!'
            write ( *, '(a,i8)' ) 
     &        '  Exceeding recursion maximum of ', level_max
            stop 1
          end if

          level = level + 1
          n_segment = l_segment
          rsave(level) = r_segment + base - 1
c
c  The left segment and the middle segment are sorted.
c  Must the right segment be partitioned?
c
        else if ( r_segment .lt. n_segment ) then

          n_segment = n_segment + 1 - r_segment
          base = base + r_segment - 1
c
c  Otherwise, we back up a level if there is an earlier one.
c
        else

20        continue

            if ( level .le. 1 ) then
              return
            end if

            base = rsave(level)
            n_segment = rsave(level-1) - rsave(level)
            level = level - 1

            if ( 0 .lt. n_segment ) then
              go to 30
            end if

          go to 20

30        continue

        end if

      go to 10

      return
      end
      function r83_norm ( x, y, z )

c**********************************************************************72
c
cc R83_NORM returns the Euclidean norm of an R83.
c
c  Discussion:
c
c    An R83 is a vector of 3 R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    12 September 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision X, Y, Z, the vector.
c
c    Output, double precision R83_NORM, the norm of the vector.
c
      implicit none

      double precision r83_norm
      double precision x
      double precision y
      double precision z

      r83_norm = sqrt ( x * x + y * y + z * z )

      return
      end
      subroutine r83_normalize ( x, y, z )

c**********************************************************************72
c
cc R83_NORMALIZE normalizes an R83.
c
c  Discussion:
c
c    An R83 is a vector of 3 R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    12 September 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input/output, double precision X, Y, Z, the components of the vector.
c
      implicit none

      double precision norm
      double precision x
      double precision y
      double precision z

      norm = sqrt ( x * x + y * y + z * z )

      if ( norm .ne. 0.0D+00 ) then
        x = x / norm
        y = y / norm
        z = z / norm
      end if

      return
      end
      subroutine r83_print ( x, y, z, title )

c**********************************************************************72
c
cc R83_PRINT prints an R83.
c
c  Discussion:
c
c    An R83 is a vector of 3 R8's.
c
c    A format is used which suggests a coordinate triple:
c
c  Example:
c
c    Center : ( 1.23, 7.45, -1.45 )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    12 September 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision X, Y, Z, the coordinates of the vector.
c
c    Input, character * ( * ) TITLE, a title.
c
      implicit none

      character * ( * ) title
      double precision x
      double precision y
      double precision z

      if ( 0 .lt. len_trim ( title ) ) then
        write ( *, '( 2x, a, a4, g14.6, a1, g14.6, a1, g14.6, a1 )' ) 
     &    trim ( title ), ' : (', x, ',', y, ',', z, ')'
      else
        write ( *, '( 2x, a1, g14.6, a1, g14.6, a1, g14.6, a1 )' ) 
     &    '(', x, ',', y, ',', z, ')'
      end if

      return
      end
      subroutine r83_swap ( x, y )

c**********************************************************************72
c
cc R83_SWAP swaps two R83's.
c
c  Discussion:
c
c    An R83 is a vector of 3 R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    12 September 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input/output, double precision X(3), Y(3).  On output, the values
c    of X and Y have been interchanged.
c
      implicit none

      integer dim_num
      parameter ( dim_num = 3 )

      integer i
      double precision x(dim_num)
      double precision y(dim_num)
      double precision z

      do i = 1, dim_num
        z    = x(i)
        x(i) = y(i)
        y(i) = z
      end do

      return
      end
      subroutine r83col_print_part ( n, a, max_print, title )

c*********************************************************************72
c
cc R83COL_PRINT_PART prints "part" of an R83COL.
c
c  Discussion:
c
c    An R83COL is an (N,3) array of R8's.
c
c    The user specifies MAX_PRINT, the maximum number of lines to print.
c
c    If N, the size of the vector, is no more than MAX_PRINT, then
c    the entire vector is printed, one entry per line.
c
c    Otherwise, if possible, the first MAX_PRINT-2 entries are printed,
c    followed by a line of periods suggesting an omission,
c    and the last entry.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    12 April 2015
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries of the vector.
c
c    Input, double precision A(N,3), the vector to be printed.
c
c    Input, integer MAX_PRINT, the maximum number of lines
c    to print.
c
c    Input, character * ( * ) TITLE, a title.
c
      implicit none

      integer n

      double precision a(n,3)
      integer i
      integer max_print
      character * ( * ) title

      if ( max_print .le. 0 ) then
        return
      end if

      if ( n .le. 0 ) then
        return
      end if

      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) trim ( title )
      write ( *, '(a)' ) ' '

      if ( n .le. max_print ) then

        do i = 1, n
          write ( *, '(2x,i8,a,1x,g14.6,2x,g14.6,2x,g14.6)' ) 
     &      i, ':', a(i,1), a(i,2), a(i,3)
        end do

      else if ( 3 .le. max_print ) then

        do i = 1, max_print - 2
          write ( *, '(2x,i8,a,1x,g14.6,2x,g14.6,2x,g14.6)' ) 
     &      i, ':', a(i,1), a(i,2), a(i,3)
        end do
        write ( *, '(a)' )      
     &    '  ........  ..............  ..............  ..............'
        i = n
        write ( *, '(2x,i8,a,1x,g14.6,2x,g14.6,2x,g14.6)' ) 
     &    i, ':', a(i,1), a(i,2), a(i,3)

      else

        do i = 1, max_print - 1
          write ( *, '(2x,i8,a,1x,g14.6,2x,g14.6,2x,g14.6)' ) 
     &      i, ':', a(i,1), a(i,2), a(i,3)
        end do
        i = max_print
        write ( *, '(2x,i8,a,1x,g14.6,2x,g14.6,2x,g14.6,2x,a)' ) 
     &    i, ':', a(i,1), a(i,2), a(i,3), '...more entries...'

      end if

      return
      end
      subroutine r83row_max ( n, a, amax )

c*********************************************************************72
c
cc R83ROW_MAX returns the maximum value in an R83ROW.
c
c  Discussion:
c
c    An R83ROW is a (3,N) array of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    10 November 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the array.
c
c    Input, double precision A(3,N), the array.
c
c    Output, double precision AMAX(3); the largest entries in each row.
c
      implicit none

      integer n

      double precision a(3,n)
      double precision amax(3)
      integer i
      integer j

      do i = 1, 3
        amax(i) = a(i,1)
        do j = 2, n
          amax(i) = max ( amax(i), a(i,j) )
        end do
      end do

      return
      end
      subroutine r83row_min ( n, a, amin )

c*********************************************************************72
c
cc R83ROW_MIN returns the minimum value in an R83ROW.
c
c  Discussion:
c
c    An R83ROW is a (3,N) array of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    10 November 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the array.
c
c    Input, double precision A(3,N), the array.
c
c    Output, double precision AMIN(3); the smallest entries in each row.
c
      implicit none

      integer n

      double precision a(3,n)
      double precision amin(3)
      integer i
      integer j

      do i = 1, 3
        amin(i) = a(i,1)
        do j = 2, n
          amin(i) = min ( amin(i), a(i,j) )
        end do
      end do

      return
      end
      subroutine r83row_normalize ( n, x )

c**********************************************************************72
c
cc R83ROW_NORMALIZE normalizes each R83 in an R83ROW.
c
c  Discussion:
c
c    An R83ROW is a (3,N) array of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    12 September 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of R83 vectors.
c
c    Input/output, double precision X(3,N), the coordinates of N R83 vectors.
c    On output, the nonzero vectors have been scaled to have unit L2 norm.
c
      implicit none

      integer n
      integer dim_num
      parameter ( dim_num = 3 )

      integer i
      integer j
      double precision norm
      double precision x(dim_num,n)

      do j = 1, n

        norm = 0.0D+00
        do i = 1, dim_num
          norm = norm + x(i,j) * x(i,j)
        end do
        norm = sqrt ( norm )

        if ( norm .ne. 0.0D+00 ) then
          do i = 1, dim_num
            x(i,j) = x(i,j) / norm
          end do
        end if

      end do

      return
      end
      subroutine r83row_print_part ( n, a, max_print, title )

c*********************************************************************72
c
cc R83ROW_PRINT_PART prints "part" of an R83ROW.
c
c  Discussion:
c
c    An R83ROW is a (3,N) array of R8's.
c
c    The user specifies MAX_PRINT, the maximum number of lines to print.
c
c    If N, the size of the vector, is no more than MAX_PRINT, then
c    the entire vector is printed, one entry per line.
c
c    Otherwise, if possible, the first MAX_PRINT-2 entries are printed,
c    followed by a line of periods suggesting an omission,
c    and the last entry.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    09 November 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries of the vector.
c
c    Input, double precision A(3,N), the vector to be printed.
c
c    Input, integer MAX_PRINT, the maximum number of lines
c    to print.
c
c    Input, character * ( * ) TITLE, a title.
c
      implicit none

      integer n

      double precision a(3,n)
      integer i
      integer max_print
      character * ( * ) title

      if ( max_print .le. 0 ) then
        return
      end if

      if ( n .le. 0 ) then
        return
      end if

      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) trim ( title )
      write ( *, '(a)' ) ' '

      if ( n .le. max_print ) then

        do i = 1, n
          write ( *, '(2x,i8,a,1x,g14.6,2x,g14.6,2x,g14.6)' ) 
     &      i, ':', a(1,i), a(2,i), a(3,i)
        end do

      else if ( 3 .le. max_print ) then

        do i = 1, max_print - 2
          write ( *, '(2x,i8,a,1x,g14.6,2x,g14.6,2x,g14.6)' ) 
     &      i, ':', a(1,i), a(2,i), a(3,i)
        end do
        write ( *, '(a)' ) 
     &    '  ........  ..............  ..............  ..............'
        i = n
        write ( *, '(2x,i8,a,1x,g14.6,2x,g14.6,2x,g14.6)' ) 
     &    i, ':', a(1,i), a(2,i), a(3,i)

      else

        do i = 1, max_print - 1
          write ( *, '(2x,i8,a,1x,g14.6,2x,g14.6,2x,g14.6)' ) 
     &      i, ':', a(1,i), a(2,i), a(3,i)
        end do
        i = max_print
        write ( *, '(2x,i8,a,1x,g14.6,2x,g14.6,2x,g14.6,2x,a)' ) 
     &    i, ':', a(1,i), a(2,i), a(3,i), '...more entries...'

      end if

      return
      end
      subroutine r84_normalize ( v )

c**********************************************************************72
c
cc R84_NORMALIZE normalizes an R84.
c
c  Discussion:
c
c    An R84 is a vector of four R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    12 September 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input/output, double precision V(4), the components of the vector.
c
      implicit none

      integer dim_num
      parameter ( dim_num = 4 )

      integer i
      double precision norm
      double precision v(dim_num)

      norm = 0.0D+00
      do i = 1, dim_num
        norm = norm + v(i) * v(i)
      end do
      norm = sqrt ( norm )

      if ( norm .ne. 0.0D+00 ) then
        do i = 1, dim_num
          v(i) = v(i) / norm
        end do
      end if

      return
      end
      subroutine r8block_expand_linear ( l, m, n, x, lfat, mfat, nfat, 
     &  xfat )

c*********************************************************************72
c
cc R8BLOCK_EXPAND_LINEAR linearly interpolates new data into an R8BLOCK.
c
c  Discussion:
c
c    An R8BLOCK is a 3D array of R8 values.
c
c    In this routine, the expansion is specified by giving the number
c    of intermediate values to generate between each pair of original
c    data rows and columns.
c
c    The interpolation is not actually linear.  It uses the functions
c
c      1, x, y, z, xy, xz, yz, xyz.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    02 December 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer L, M, N, the dimensions of the input data.
c
c    Input, double precision X(L,M,N), the original data.
c
c    Input, integer LFAT, MFAT, NFAT, the number of data values
c    to interpolate original data values in the first, second and third
c    dimensions.
c
c    Output, double precision XFAT(L2,M2,N2), the fattened data, where
c    L2 = (L-1)*(LFAT+1)+1,
c    M2 = (M-1)*(MFAT+1)+1,
c    N2 = (N-1)*(NFAT+1)+1.
c
      implicit none

      integer l
      integer lfat
      integer m
      integer mfat
      integer n
      integer nfat

      integer i
      integer ihi
      integer ii
      integer iii
      integer ip1
      integer j
      integer jhi
      integer jj
      integer jjj
      integer jp1
      integer k
      integer khi
      integer kk
      integer kkk
      integer kp1
      double precision r
      double precision rn
      double precision s
      double precision sn
      double precision t
      double precision tn
      double precision x(l,m,n)
      double precision x000
      double precision x001
      double precision x010
      double precision x011
      double precision x100
      double precision x101
      double precision x110
      double precision x111
      double precision 
     &  xfat((l-1)*(lfat+1)+1,(m-1)*(mfat+1)+1,(n-1)*(nfat+1)+1)

      do i = 1, l

        if ( i .lt. l ) then
          ihi = lfat
        else
          ihi = 0
        end if

        do j = 1, m

          if ( j .lt. m ) then
            jhi = mfat
          else
            jhi = 0
          end if

          do k = 1, n

            if ( k .lt. n ) then
              khi = nfat
            else
              khi = 0
            end if

            if ( i .lt. l ) then
              ip1 = i + 1
            else
              ip1 = i
            end if

            if ( j .lt. m ) then
              jp1 = j + 1
            else
              jp1 = j
            end if

            if ( k .lt. n ) then
              kp1 = k + 1
            else
              kp1 = k
            end if

            x000 = x(i,j,k)
            x001 = x(i,j,kp1)
            x100 = x(ip1,j,k)
            x101 = x(ip1,j,kp1)
            x010 = x(i,jp1,k)
            x011 = x(i,jp1,kp1)
            x110 = x(ip1,jp1,k)
            x111 = x(ip1,jp1,kp1)

            do ii = 0, ihi

              r = dble ( ii ) / dble ( ihi + 1 )

              do jj = 0, jhi

                s = dble ( jj ) / dble ( jhi + 1 )

                do kk = 0, khi

                  t = dble ( kk ) / dble ( khi + 1 )

                  iii = 1 + ( i - 1 ) * ( lfat + 1 ) + ii
                  jjj = 1 + ( j - 1 ) * ( mfat + 1 ) + jj
                  kkk = 1 + ( k - 1 ) * ( nfat + 1 ) + kk

                  rn = 1.0D+00 - r
                  sn = 1.0D+00 - s
                  tn = 1.0D+00 - t

                  xfat(iii,jjj,kkk) = 
     &                x000 * rn * sn * tn 
     &              + x001 * rn * sn * t 
     &              + x010 * rn * s  * tn 
     &              + x011 * rn * s  * t  
     &              + x100 * r  * sn * tn
     &              + x101 * r  * sn * t
     &              + x110 * r  * s  * tn
     &              + x111 * r  * s  * t

                end do

              end do

            end do

          end do

        end do

      end do

      return
      end
      subroutine r8block_print ( l, m, n, a, title )

c*********************************************************************72
c
cc R8BLOCK_PRINT prints an R8BLOCK.
c
c  Discussion:
c
c    An R8BLOCK is a 3D array of R8 values.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    12 September 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer L, M, N, the dimensions of the block.
c
c    Input, double precision A(L,M,N), the matrix to be printed.
c
c    Input, character * ( * ) TITLE, a title.
c
      implicit none

      integer l
      integer m
      integer n

      double precision a(l,m,n)
      integer i
      integer j
      integer jhi
      integer jlo
      integer k
      character * ( * ) title

      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) trim ( title )

      do k = 1, n

        write ( *, '(a)' ) ' '
        write ( *, '(a,i8)' ) '  K = ', k
        write ( *, '(a)' ) ' '

        do jlo = 1, m, 5
          jhi = min ( jlo + 4, m )
          write ( *, '(a)' ) ' '
          write ( *, '(10x,5(i8,6x))' ) (j, j = jlo, jhi )
          write ( *, '(a)' ) ' '
          do i = 1, l
            write ( *, '(2x,i8,5g14.6)' ) i, ( a(i,j,k), j = jlo, jhi )
          end do
        end do

      end do

      return
      end
      subroutine r8block_zero ( l, m, n, a )

c*********************************************************************72
c
cc R8BLOCK_ZERO zeroes an R8BLOCK.
c
c  Discussion:
c
c    An R8BLOCK is a triple dimensioned array of R8 values.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    13 April 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer L, M, N, the dimensions.
c
c    Output, double precision A(L,M,N), the block of zeroes.
c
      implicit none

      integer l
      integer m
      integer n

      double precision a(l,m,n)
      integer i
      integer j
      integer k

      do k = 1, n
        do j = 1, m
          do i = 1, l
            a(i,j,k) = 0.0D+00
          end do
        end do
      end do

      return
      end
      subroutine r8cmat_print ( lda, m, n, a, title )

c*********************************************************************72
c
cc R8CMAT_PRINT prints an R8CMAT.
c
c  Discussion:
c
c    An R8CMAT is an M by N array of R8's stored with leading dimension LD.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    18 March 2014
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer LDA, the leading dimension of A.
c
c    Input, integer M, the number of rows in A.
c
c    Input, integer N, the number of columns in A.
c
c    Input, double precision A(LDA,N), the M by N matrix.
c
c    Input, character * ( * ) TITLE, a title.
c
      implicit none

      integer lda
      integer m
      integer n

      double precision a(lda,n)
      character * ( * ) title

      call r8cmat_print_some ( lda, m, n, a, 1, 1, m, n, title )

      return
      end
      subroutine r8cmat_print_some ( lda, m, n, a, ilo, jlo, ihi, jhi,
     &  title )

c*********************************************************************72
c
cc R8CMAT_PRINT_SOME prints some of an R8CMAT.
c
c  Discussion:
c
c    An R8CMAT is an M by N array of R8's stored with leading dimension LD.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    18 March 2014
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer LDA, the leading dimension of A.
c
c    Input, integer M, the number of rows in A.
c
c    Input, integer N, the number of columns in A.
c
c    Input, double precision A(LDA,N), the M by N matrix.
c
c    Input, integer ILO, JLO, the first row and column to print.
c
c    Input, integer IHI, JHI, the last row and column to print.
c
c    Input, character * ( * ) TITLE, a title.
c
      implicit none

      integer incx
      parameter ( incx = 5 )
      integer lda
      integer m
      integer n

      double precision a(lda,n)
      character * ( 14 ) ctemp(incx)
      integer i
      integer i2hi
      integer i2lo
      integer ihi
      integer ilo
      integer inc
      integer j
      integer j2
      integer j2hi
      integer j2lo
      integer jhi
      integer jlo
      character * ( * ) title

      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) trim ( title )

      if ( m .le. 0 .or. n .le. 0 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) '  (None)'
        return
      end if

      do j2lo = max ( jlo, 1 ), min ( jhi, n ), incx

        j2hi = j2lo + incx - 1
        j2hi = min ( j2hi, n )
        j2hi = min ( j2hi, jhi )

        inc = j2hi + 1 - j2lo

        write ( *, '(a)' ) ' '

        do j = j2lo, j2hi
          j2 = j + 1 - j2lo
          write ( ctemp(j2), '(i7,7x)') j
        end do

        write ( *, '(''  Col   '',5a14)' ) ( ctemp(j), j = 1, inc )
        write ( *, '(a)' ) '  Row'
        write ( *, '(a)' ) ' '

        i2lo = max ( ilo, 1 )
        i2hi = min ( ihi, m )

        do i = i2lo, i2hi

          do j2 = 1, inc

            j = j2lo - 1 + j2

            write ( ctemp(j2), '(g14.6)' ) a(i,j)

          end do

          write ( *, '(i5,a,5a14)' ) i, ':', ( ctemp(j), j = 1, inc )

        end do

      end do

      return
      end
      subroutine r8cmat_to_r8mat ( lda, m, n, a1, a2 )

c*********************************************************************72
c
cc R8CMAT_TO_R8MAT transfers data from an R8CMAT to an R8MAT.
c
c  Discussion:
c
c    An R8CMAT is an MxN array of R8's, stored with a leading dimension LD,
c    accessible as a vector:
c      (I,J) -> (I+J*LD).
c    or as a doubly-dimensioned array, if declared A(LD,N):
c      (I,J) -> A(I,J)
c
c    An R8MAT is an MxN array of R8's, 
c    accessible as a vector:
c      (I,J) -> (I+J*M).
c    or as a doubly-dimensioned array, if declared A(M,N):
c      (I,J) -> A(I,J)
c      
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    19 March 2014
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer LDA, the leading dimension of A1.
c
c    Input, integer M, the number of rows of data.
c    M <= LDA.
c
c    Input, integer N, the number of columns of data.
c
c    Input, double precision A1(LDA,N), the matrix to be copied.
c
c    Output, double precision A2(M,N), a copy of the
c    information in the MxN submatrix of A1.
c
      implicit none

      integer lda
      integer m
      integer n

      double precision a1(lda,n)
      double precision a2(m,n)
      integer i
      integer j

      do j = 1, n
        do i = 1, m
          a2(i,j) = a1(i,j)
        end do
      end do
 
      return
      end
      subroutine r8col_compare ( m, n, a, i, j, value )

c*********************************************************************72
c
cc R8COL_COMPARE compares columns in an R8COL.
c
c  Discussion:
c
c    An R8COL is an M by N array of R8 values.
c    It is regarded as an array of N columns of length M.
c
c  Example:
c
c    Input:
c
c      M = 3, N = 4, I = 2, J = 4
c
c      A = (
c        1.  2.  3.  4.
c        5.  6.  7.  8.
c        9. 10. 11. 12. )
c
c    Output:
c
c      VALUE = -1
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    24 August 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns.
c
c    Input, double precision A(M,N), the M by N array.
c
c    Input, integer I, J, the columns to be compared.
c    I and J must be between 1 and N.
c
c    Output, integer VALUE, the results of the comparison:
c    -1, column I < column J,
c     0, column I = column J,
c    +1, column J < column I.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer isgn
      integer j
      integer k
      integer value
c
c  Check.
c
      if ( i .lt. 1 .or. n .lt. i ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8COL_COMPARE - Fatal error!'
        write ( *, '(a)' ) '  Column index I is out of bounds.'
        write ( *, '(a,i8)' ) '  I = ', i
        stop 1
      end if

      if ( j .lt. 1 .or. n .lt. j ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8COL_COMPARE - Fatal error!'
        write ( *, '(a)' ) '  Column index J is out of bounds.'
        write ( *, '(a,i8)' ) '  J = ', j
        stop 1
      end if

      value = 0

      if ( i .eq. j ) then
        return
      end if

      k = 1

10    continue

      if ( k .le. m ) then

        if ( a(k,i) .lt. a(k,j) ) then
          value = -1
          return
        else if ( a(k,j) .lt. a(k,i) ) then
          value = +1
          return
        end if

        k = k + 1

        go to 10

      end if

      return
      end
      subroutine r8col_duplicates ( m, n, n_unique, seed, a )

c*********************************************************************72
c
cc R8COL_DUPLICATES generates an R8COL with some duplicate columns.
c
c  Discussion:
c
c    An R8COL is an M by N array of R8's, regarded as an array of N columns,
c    each of length M.
c
c    This routine generates a random R8COL with a specified number of
c    duplicate columns.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    21 July 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the number of rows in each column of A.
c
c    Input, integer N, the number of columns in A.
c
c    Input, integer N_UNIQUE, the number of unique columns in A.
c    1 <= N_UNIQUE <= N.
c
c    Input/output, integer SEED, a seed for the random
c    number generator.
c
c    Output, double precision A(M,N), the array.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer i4_uniform_ab
      integer j1
      integer j2
      integer n_unique
      integer seed
      double precision temp(m)

      if ( n_unique .lt. 1 .or. n .lt. n_unique ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8COL_DUPLICATES - Fatal error!'
        write ( *, '(a)' ) '  1 <= N_UNIQUE <= N is required.'
        stop 1
      end if

      call r8mat_uniform_01 ( m, n_unique, seed, a )
c
c  Randomly copy unique columns.
c
      do j1 = n_unique + 1, n
        j2 = i4_uniform_ab ( 1, n_unique, seed )
        do i = 1, m
          a(i,j1) = a(i,j2)
        end do
      end do
c
c  Permute the columns.
c
      do j1 = 1, n
        j2 = i4_uniform_ab ( j1, n, seed )
        do i = 1, m
          temp(i) = a(i,j1)
          a(i,j1) = a(i,j2)
          a(i,j2) = temp(i)
        end do
      end do

      return
      end
      subroutine r8col_find ( m, n, a, x, col )

c*********************************************************************72
c
cc R8COL_FIND seeks a column value in an R8COL.
c
c  Discussion:
c
c    An R8COL is an M by N array of R8 values, regarded
c    as an array of N columns of length M.
c
c  Example:
c
c    Input:
c
c      M = 3,
c      N = 4,
c
c      A = (
c        1.  2.  3.  4.
c        5.  6.  7.  8.
c        9. 10. 11. 12. )
c
c      x = ( 3.,
c            7.,
c           11. )
c
c    Output:
c
c      COL = 3
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    31 August 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns.
c
c    Input, double precision A(M,N), a table of numbers, regarded as
c    N columns of vectors of length M.
c
c    Input, double precision X(M), a vector to be matched with a column of A.
c
c    Output, integer COL, the index of the first column of A
c    which exactly matches every entry of X, or -1 if no match
c    could be found.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer col
      integer i
      integer j
      double precision x(m)

      col = -1

      do j = 1, n

        col = j

        do i = 1, m
          if ( x(i) .ne. a(i,j) ) then
            col = -1
            go to 10
          end if
        end do

10      continue

        if ( col .ne. -1 ) then
          return
        end if

      end do

      return
      end
      subroutine r8col_first_index ( m, n, a, tol, first_index )

c*********************************************************************72
c
cc R8COL_FIRST_INDEX indexes the first occurrence of values in an R8COL.
c
c  Discussion:
c
c    An R8COL is an M by N array of R8 values.
c    It is regarded as an array of N columns of length M.
c
c    For element A(1:M,J) of the matrix, FIRST(J) is the index in A of
c    the first column whose entries are equal to A(1:M,J).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    24 August 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns of A.
c    The length of an "element" of A, and the number of "elements".
c
c    Input, double precision A(M,N), the array.
c
c    Input, double precision TOL, a tolerance for equality.
c
c    Output, integer FIRST_INDEX(N), the first occurrence index.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      double precision diff
      integer first_index(n)
      integer i
      integer j1
      integer j2
      double precision tol

      do j1 = 1, n
        first_index(j1) = -1
      end do

      do j1 = 1, n

        if ( first_index(j1) .eq. -1 ) then

          first_index(j1) = j1

          do j2 = j1 + 1, n
            diff = 0.0D+00
            do i = 1, m
              diff = diff + abs ( a(i,j1) - a(i,j2) )
            end do
            if ( diff .le. tol ) then
              first_index(j2) = j1
            end if
          end do

        end if

      end do

      return
      end
      subroutine r8col_insert ( n_max, m, n, a, x, col )

c*********************************************************************72
c
cc R8COL_INSERT inserts a column into an R8COL.
c
c  Discussion:
c
c    An R8COL is an M by N array of R8's, regarded as an array of N columns,
c    each of length M.
c
c  Example:
c
c    Input:
c
c      N_MAX = 10,
c      M = 3,
c      N = 4,
c
c      A = (
c        1.  2.  3.  4.
c        5.  6.  7.  8.
c        9. 10. 11. 12. )
c
c      X = ( 3., 4., 18. )
c
c    Output:
c
c      N = 5,
c
c      A = (
c        1.  2.  3.  3.  4.
c        5.  6.  4.  7.  8.
c        9. 10. 18. 11. 12. )
c
c      COL = 3
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    08 May 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N_MAX, the maximum number of columns in A.
c
c    Input, integer M, the number of rows.
c
c    Input/output, integer N, the number of columns.
c    If the new column is inserted into the table, then the output
c    value of N will be increased by 1.
c
c    Input/output, double precision A(M,N_MAX), a table of numbers, regarded
c    as an array of columns.  The columns must have been sorted
c    lexicographically.
c
c    Input, double precision X(M), a vector of data which will be inserted
c    into the table if it does not already occur.
c
c    Output, integer COL.
c    I, X was inserted into column I.
c    -I, column I was already equal to X.
c    0, N = N_MAX.
c
      implicit none

      integer m
      integer n_max

      double precision a(m,n_max)
      integer col
      integer high
      integer i
      integer isgn
      integer j
      integer low
      integer mid
      integer n
      double precision x(m)
c
c  Refuse to work if N_MAX <= N.
c
      if ( n_max .le. n ) then
        col = 0
        return
      end if
c
c  Stick X temporarily in column N+1, just so it's easy to use R8COL_COMPARE.
c
      do i = 1, m
        a(i,n+1) = x(i)
      end do
c
c  Do a binary search.
c
      low = 1
      high = n

10    continue

        if ( high .lt. low ) then
          col = low
          go to 20
        end if

        mid = ( low + high ) / 2

        call r8col_compare ( m, n + 1, a, mid, n + 1, isgn )

        if ( isgn .eq. 0 ) then
          col = -mid
          return
        else if ( isgn .eq. -1 ) then
          low = mid + 1
        else if ( isgn .eq. +1 ) then
          high = mid - 1
        end if

      go to 10

20    continue
c
c  Shift part of the table up to make room.
c
      do j = n, col, -1
        do i = 1, m
          a(i,j+1) = a(i,j)
        end do
      end do
c
c  Insert the new column.
c
      do i = 1, m
        a(i,col) = x(i)
      end do

      n = n + 1

      return
      end
      subroutine r8col_max ( m, n, a, amax )

c*********************************************************************72
c
cc R8COL_MAX returns the maximums in an R8COL.
c
c  Discussion:
c
c    An R8COL is an M by N array of R8's, regarded as an array of N columns,
c    each of length M.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    08 May 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns.
c
c    Input, double precision A(M,N), the array to be examined.
c
c    Output, double precision AMAX(N), the maximums of the columns.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      double precision amax(n)
      integer i
      integer j

      do j = 1, n

        amax(j) = a(1,j)
        do i = 2, m
          amax(j) = max ( amax(j), a(i,j) )
        end do

      end do

      return
      end
      subroutine r8col_max_index ( m, n, a, imax )

c*********************************************************************72
c
cc R8COL_MAX_INDEX returns the indices of column maximums in an R8COL.
c
c  Discussion:
c
c    An R8COL is an M by N array of R8's, regarded as an array of N columns,
c    each of length M.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    08 May 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns.
c
c    Input, double precision A(M,N), the array to be examined.
c
c    Output, integer IMAX(N); IMAX(I) is the row of A in which
c    the maximum for column I occurs.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      double precision amax
      integer i
      integer imax(n)
      integer j

      do j = 1, n

        imax(j) = 1
        amax = a(1,j)
        do i = 2, m
          if ( amax .lt. a(i,j) ) then
            imax(j) = i
            amax = a(i,j)
          end if
        end do

      end do

      return
      end
      subroutine r8col_max_one ( m, n, a )

c*********************************************************************72
c
cc R8COL_MAX_ONE rescales an R8COL so each column maximum is 1.
c
c  Discussion:
c
c    An R8COL is an M by N array of R8's, regarded as an array of N columns,
c    each of length M.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    08 May 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns.
c
c    Input/output, double precision A(M,N), the array to be rescaled.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer i_big
      integer j
      double precision temp

      do j = 1, n

        i_big = 1
        do i = 2, m
          if ( abs ( a(i_big,j) ) .lt. abs ( a(i,j) ) ) then
            i_big = i
          end if
        end do

        if ( a(i_big,j) .ne. 0.0D+00 ) then
          temp = a(i_big,j)
          do i = 1, m
            a(i,j) = a(i,j) / temp
          end do
        end if

      end do

      return
      end
      subroutine r8col_mean ( m, n, a, mean )

c*********************************************************************72
c
cc R8COL_MEAN returns the column means of an R8COL.
c
c  Discussion:
c
c    An R8COL is an M by N array of R8 values, regarded
c    as an array of N columns of length M.
c
c  Example:
c
c    A =
c      1  2  3
c      2  6  7
c
c    MEAN =
c      1.5  4.0  5.0
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    13 January 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns.
c
c    Input, double precision A(M,N), the array to be examined.
c
c    Output, double precision MEAN(N), the means, or averages, of the columns.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      double precision a_sum
      integer i
      integer j
      double precision mean(n)

      do j = 1, n

        a_sum = 0.0D+00
        do i = 1, m
          a_sum = a_sum + a(i,j)
        end do

        mean(j) = a_sum / dble ( m )

      end do

      return
      end
      subroutine r8col_min ( m, n, a, amin )

c*********************************************************************72
c
cc R8COL_MIN returns the minimums in an R8COL.
c
c  Discussion:
c
c    An R8COL is an M by N array of R8's, regarded as an array of N columns,
c    each of length M.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    08 May 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns.
c
c    Input, double precision A(M,N), the array to be examined.
c
c    Output, double precision AMIN(N), the minimums of the columns.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      double precision amin(n)
      integer i
      integer j

      do j = 1, n

        amin(j) = a(1,j)
        do i = 2, m
          amin(j) = min ( amin(j), a(i,j) )
        end do

      end do

      return
      end
      subroutine r8col_min_index ( m, n, a, imin )

c*********************************************************************72
c
cc R8COL_MIN_INDEX returns the indices of column minimums in an R8COL.
c
c  Discussion:
c
c    An R8COL is an M by N array of R8's, regarded as an array of N columns,
c    each of length M.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    08 May 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns.
c
c    Input, double precision A(M,N), the array to be examined.
c
c    Output, integer IMIN(N); IMIN(I) is the row of A in which
c    the minimum for column I occurs.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      double precision amin
      integer i
      integer imin(n)
      integer j

      do j = 1, n

        imin(j) = 1
        amin = a(1,j)
        do i = 2, m
          if ( a(i,j) .lt. amin ) then
            imin(j) = i
            amin = a(i,j)
          end if
        end do

      end do

      return
      end
      subroutine r8col_normalize_li ( m, n, a )

c*********************************************************************72
c
cc R8COL_NORMALIZE_LI normalizes an R8COL with the column infinity norm.
c
c  Discussion:
c
c    Each column is scaled so that the entry of maximum norm has the value 1.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    08 February 2012
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns.
c
c    Input/output, double precision A(M,N), the array to be normalized.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      double precision c
      integer i
      integer j

      do j = 1, n

        c = a(1,j)

        do i = 2, m
          if ( abs ( c ) .lt. abs ( a(i,j) ) ) then
            c = a(i,j)
          end if
        end do

        if ( c .ne. 0.0D+00 ) then
          do i = 1, m
            a(i,j) = a(i,j) / c
          end do
        end if

      end do

      return
      end
      subroutine r8col_part_quick_a ( m, n, a, l, r )

c*********************************************************************72
c
cc R8COL_PART_QUICK_A reorders the columns of an array as part of a quick sort.
c
c  Discussion:
c
c    The routine reorders the columns of A.  Using A(1:M,1) as a
c    key, all entries of A that are less than or equal to the key will
c    precede the key, which precedes all entries that are greater than the key.
c
c  Example:
c
c    Input:
c
c      M = 2, N = 8
c      A = ( 2  8  6  0 10 10  0  5
c            4  8  2  2  6  0  6  8 )
c
c    Output:
c
c      L = 2, R = 4
c
c      A = (  0  0  2  8  6 10 10  4
c             2  6  4  8  2  6  0  8 )
c             ----     -------------
c             LEFT KEY     RIGHT
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    22 July 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the row dimension of A, and the length of a column.
c
c    Input, integer N, the column dimension of A.
c
c    Input/output, double precision A(M,N).  On input, the array to be checked.
c    On output, A has been reordered as described above.
c
c    Output, integer L, R, the indices of A that define the three segments.
c    Let KEY = the input value of A(1:M,1).  Then
c    I <= L                 A(1:M,I) < KEY;
c         L < I < R         A(1:M,I) = KEY;
c                 R <= I    A(1:M,I) > KEY.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer j
      integer k
      double precision key(m)
      integer l
      integer r
      logical r8vec_eq
      logical r8vec_gt
      logical r8vec_lt

      if ( n .lt. 1 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8COL_PART_QUICK_A - Fatal error!'
        write ( *, '(a)' ) '  N < 1.'
        return
      end if

      if ( n .eq. 1 ) then
        l = 0
        r = 2
        return
      end if

      do i = 1, m
        key(i) = a(i,1)
      end do

      k = 1
c
c  The elements of unknown size have indices between L+1 and R-1.
c
      l = 1
      r = n + 1

      do i = 2, n

        if ( r8vec_gt ( m, a(1,l+1), key ) ) then
          r = r - 1
          call r8vec_swap ( m, a(1,r), a(1,l+1) )
        else if ( r8vec_eq ( m, a(1,l+1), key ) ) then
          k = k + 1
          call r8vec_swap ( m, a(1,k), a(1,l+1) )
          l = l + 1
        else if ( r8vec_lt ( m, a(1,l+1), key ) ) then
          l = l + 1
        end if

      end do
c
c  Shift small elements to the left.
c
      do j = 1, l - k
        do i = 1, m
          a(i,j) = a(i,j+k)
        end do
      end do
c
c  Shift KEY elements to center.
c
      do j = l-k+1, l
        do i = 1, m
          a(i,j) = key(i)
        end do
      end do
c
c  Update L.
c
      l = l - k

      return
      end
      subroutine r8col_permute ( m, n, p, a )

c*********************************************************************72
c
cc R8COL_PERMUTE permutes an R8COL in place.
c
c  Discussion:
c
c    An R8COL is an M by N array of R8's, regarded as an array of N columns,
c    each of length M.
c
c    The same logic can be used to permute an array of objects of any
c    arithmetic type, or an array of objects of any complexity.  The only
c    temporary storage required is enough to store a single object.  The number
c    of data movements made is N + the number of cycles of order 2 or more,
c    which is never more than N + N/2.
c
c  Example:
c
c    Input:
c
c      M = 2
c      N = 5
c      P = (   2,    4,    5,    1,    3 )
c      A = ( 1.0,  2.0,  3.0,  4.0,  5.0 )
c          (11.0, 22.0, 33.0, 44.0, 55.0 )
c
c    Output:
c
c      A    = (  2.0,  4.0,  5.0,  1.0,  3.0 )
c             ( 22.0, 44.0, 55.0, 11.0, 33.0 ).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    16 July 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the dimension of objects.
c
c    Input, integer N, the number of objects.
c
c    Input, integer P(N), the permutation.  P(I) = J means
c    that the I-th element of the output array should be the J-th
c    element of the input array.
c
c    Input/output, double precision A(M,N), the array to be permuted.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      double precision a_temp(m)
      integer i
      integer ierror
      integer iget
      integer iput
      integer istart
      integer p(n)

      call perm_check1 ( n, p )
c
c  Search for the next element of the permutation that has not been used.
c
      do istart = 1, n

        if ( p(istart) .lt. 0 ) then

          go to 30

        else if ( p(istart) .eq. istart ) then

          p(istart) = - p(istart)
          go to 30

        else

          do i = 1, m
            a_temp(i) = a(i,istart)
          end do
          iget = istart
c
c  Copy the new value into the vacated entry.
c
10        continue

            iput = iget
            iget = p(iget)

            p(iput) = - p(iput)

            if ( iget .lt. 1 .or. n .lt. iget ) then
              write ( *, '(a)' ) ' '
              write ( *, '(a)' ) 'R8COL_PERMUTE - Fatal error!'
              write ( *, '(a)' ) '  Permutation index out of range.'
              write ( *, '(a,i8,a,i8)' ) '  P(', iput, ') = ', iget
              stop 1
            end if

            if ( iget .eq. istart ) then
              do i = 1, m
                a(i,iput) = a_temp(i)
              end do
              go to 20
            end if

            do i = 1, m
              a(i,iput) = a(i,iget)
            end do

          go to 10

20        continue

        end if

30      continue

      end do
c
c  Restore the signs of the entries.
c
      do i = 1, n
        p(i) = - p(i)
      end do

      return
      end
      subroutine r8col_reverse ( m, n, a )

c*********************************************************************72
c
cc R8COL_REVERSE reverses the order of columns in an R8COL.
c
c  Discussion:
c
c    To reverse the columns is to start with something like
c
c      11 12 13 14 15
c      21 22 23 24 25
c      31 32 33 34 35
c      41 42 43 44 45
c      51 52 53 54 55
c
c    and return
c
c      15 14 13 12 11
c      25 24 23 22 21
c      35 34 33 32 31
c      45 44 43 42 41
c      55 54 53 52 51
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    06 May 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns.
c
c    Input/output, double precision A(M,N), the matrix.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer j
      integer jhi
      double precision t

      jhi = n / 2

      do j = 1, jhi
        do i = 1, m
          t          = a(i,j)
          a(i,j)     = a(i,n+1-j)
          a(i,n+1-j) = t
        end do
      end do

      return
      end
      subroutine r8col_separation ( m, n, a, d_min, d_max )

c*********************************************************************72
c
cc R8COL_SEPARATION returns the "separation" of an R8COL.
c
c  Discussion:
c
c    D_MIN is the minimum distance between two columns,
c    D_MAX is the maximum distance between two columns.
c
c    The distances are measured using the Loo norm.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    24 February 2014
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns 
c    in the array.  If N < 2, it does not make sense to call this routine.
c
c    Input, double precision A(M,N), the array whose variances are desired.
c
c    Output, double precision D_MIN, D_MAX, the minimum and maximum distances.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      double precision d
      double precision d_max
      double precision d_min
      integer i
      integer j1
      integer j2
      double precision r8_huge
      parameter ( r8_huge = 1.79769313486231571D+308 )

      d_min = r8_huge
      d_max = 0.0D+00

      do j1 = 1, n
        do j2 = j1 + 1, n
          d = 0.0D+00
          do i = 1, m
            d = max ( d, abs ( a(i,j1) - a(i,j2) ) )
          end do
          d_min = min ( d_min, d )
          d_max = max ( d_max, d )
        end do
      end do

      return
      end
      subroutine r8col_sort_heap_a ( m, n, a )

c*********************************************************************72
c
cc R8COL_SORT_HEAP_A ascending heapsorts an R8COL.
c
c  Discussion:
c
c    An R8COL is an M by N array of R8's, regarded as an array of N columns,
c    each of length M.
c
c    In lexicographic order, the statement "X < Y", applied to two real
c    vectors X and Y of length M, means that there is some index I, with
c    1 <= I <= M, with the property that
c
c      X(J) = Y(J) for J < I,
c    and
c      X(I) < Y(I).
c
c    In other words, the first time they differ, X is smaller.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    07 June 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns.
c
c    Input/output, double precision A(M,N).
c    On input, the array of N columns of M-vectors.
c    On output, the columns of A have been sorted in lexicographic order.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer indx
      integer isgn
      integer j

      if ( m .le. 0 ) then
        return
      end if

      if ( n .le. 1 ) then
        return
      end if
c
c  Initialize.
c
      i = 0
      indx = 0
      isgn = 0
      j = 0
c
c  Call the external heap sorter.
c
10    continue

        call sort_heap_external ( n, indx, i, j, isgn )
c
c  Interchange the I and J objects.
c
        if ( 0 .lt. indx ) then

          call r8col_swap ( m, n, a, i, j )
c
c  Compare the I and J objects.
c
        else if ( indx .lt. 0 ) then

          call r8col_compare ( m, n, a, i, j, isgn )

        else if ( indx .eq. 0 ) then

          go to 20

        end if

      go to 10

20    continue

      return
      end
      subroutine r8col_sort_heap_index_a ( m, n, a, indx )

c*********************************************************************72
c
cc R8COL_SORT_HEAP_INDEX_A does an indexed heap ascending sort of an R8COL.
c
c  Discussion:
c
c    An R8COL is an M by N array of R8's, regarded as an array of N columns,
c    each of length M.
c
c    The sorting is not actually carried out.  Rather an index array is
c    created which defines the sorting.  This array may be used to sort
c    or index the array, or to sort or index related arrays keyed on the
c    original array.
c
c    A(*,J1) < A(*,J2) if the first nonzero entry of A(*,J1)-A(*,J2)
c    is negative.
c
c    Once the index array is computed, the sorting can be carried out
c    "implicitly:
c
c      A(*,INDX(*)) is sorted,
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    24 July 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the number of rows in each column of A.
c
c    Input, integer N, the number of columns in A.
c
c    Input, double precision A(M,N), the array.
c
c    Output, integer INDX(N), the sort index.  The I-th element
c    of the sorted array is column INDX(I).
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      double precision column(m)
      integer i
      integer indx(n)
      integer indxt
      integer ir
      integer isgn
      integer j
      integer l

      if ( n .lt. 1 ) then
        return
      end if

      do i = 1, n
        indx(i) = i
      end do

      if ( n .eq. 1 ) then
        return
      end if

      l = ( n / 2 ) + 1
      ir = n

10    continue

        if ( 1 .lt. l ) then

          l = l - 1
          indxt = indx(l)
          do i = 1, m
            column(i) = a(i,indxt)
          end do

        else

          indxt = indx(ir)
          do i = 1, m
            column(i) = a(i,indxt)
          end do
          indx(ir) = indx(1)
          ir = ir - 1

          if ( ir .eq. 1 ) then
            indx(1) = indxt
            go to 30
          end if

        end if

        i = l
        j = l + l

20      continue

        if ( j .le. ir ) then

          if ( j .lt. ir ) then

            call r8vec_compare ( m, a(1,indx(j)), a(1,indx(j+1)), isgn )

            if ( isgn .lt. 0 ) then
              j = j + 1
            end if

          end if

          call r8vec_compare ( m, column, a(1,indx(j)), isgn )

          if ( isgn .lt. 0 ) then
            indx(i) = indx(j)
            i = j
            j = j + j
          else
            j = ir + 1
          end if

          go to 20

        end if

        indx(i) = indxt

      go to 10

30    continue

      return
      end
      subroutine r8col_sort_quick_a ( m, n, a )

c*********************************************************************72
c
cc R8COL_SORT_QUICK_A ascending sorts the columns of a table using quick sort.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    22 July 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the row order of A, and the length of a column.
c
c    Input, integer N, the number of columns of A.
c
c    Input/output, double precision A(M,N).
c    On input, the array to be sorted.
c    On output, the array has been sorted.
c
      implicit none

      integer MAXLEVEL
      parameter ( MAXLEVEL = 25 )

      integer m
      integer n

      double precision a(m,n)
      integer base
      integer l_segment
      integer level
      integer n_segment
      integer rsave(MAXLEVEL)
      integer r_segment

      if ( n .lt. 1 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8COL_SORT_QUICK_A - Fatal error!'
        write ( *, '(a)' ) '  N < 1.'
        stop 1
      end if

      if ( n .eq. 1 ) then
        return
      end if

      level = 1
      rsave(level) = n + 1
      base = 1
      n_segment = n

10    continue
c
c  Partition the segment.
c
        call r8col_part_quick_a ( m, n_segment, a(1,base), l_segment,
     &    r_segment )
c
c  If the left segment has more than one element, we need to partition it.
c
        if ( 1 .lt. l_segment ) then

          if ( MAXLEVEL .lt. level ) then
            write ( *, '(a)' ) ' '
            write ( *, '(a)' ) 'R8COL_SORT_QUICK_A - Fatal error!'
            write ( *, '(a,i8)' )
     &        '  Exceeding recursion maximum of ', MAXLEVEL
            stop 1
          end if

          level = level + 1
          n_segment = l_segment
          rsave(level) = r_segment + base - 1
c
c  The left segment and the middle segment are sorted.
c  Must the right segment be partitioned?
c
        else if ( r_segment .lt. n_segment ) then

          n_segment = n_segment + 1 - r_segment
          base = base + r_segment - 1
c
c  Otherwise, we back up a level if there is an earlier one.
c
        else

20        continue

            if ( level .le. 1 ) then
              go to 40
            end if

            base = rsave(level)
            n_segment = rsave(level-1) - rsave(level)
            level = level - 1

            if ( 0 .lt. n_segment ) then
              go to 30
            end if

          go to 20

30        continue

        end if

      go to 10

40    continue

      return
      end
      subroutine r8col_sorted_tol_undex ( m, n, a, unique_num, tol,
     &  undx, xdnu )

c*********************************************************************72
c
cc R8COL_SORTED_TOL_UNDEX: tolerably unique indexes for a sorted R8COL.
c
c  Discussion:
c
c    An R8COL is an M by N array of R8's, regarded as an array of N columns,
c    each of length M.
c
c    The goal of this routine is to determine a vector UNDX,
c    which points, to the tolerably unique elements of A, in sorted order,
c    and a vector XDNU, which identifies, for each entry of A, the index of
c    the unique sorted element of A.
c
c    This is all done with index vectors, so that the elements of
c    A are never moved.
c
c    Assuming A is already sorted, we examine the entries of A in order,
c    noting the unique entries, creating the entries of XDNU and
c    UNDX as we go.
c
c    Once this process has been completed, the vector A could be
c    replaced by a compressed vector XU, containing the unique entries
c    of A in sorted order, using the formula
c
c      XU(*) = A(UNDX(*)).
c
c    We could then, if we wished, reconstruct the entire vector A, or
c    any element of it, by index, as follows:
c
c      A(I) = XU(XDNU(I)).
c
c    We could then replace A by the combination of XU and XDNU.
c
c    Later, when we need the I-th entry of A, we can locate it as
c    the XDNU(I)-th entry of XU.
c
c    Here is an example of a vector A, the unique sort and
c    inverse unique sort vectors and the compressed unique sorted vector.
c
c      I      A      XU  Undx  Xdnu
c    ----+------+------+-----+-----+
c      1 | 11.0 |  11.0    1     1
c      2 | 11.0 |  22.0    5     1
c      3 | 11.0 |  33.0    8     1
c      4 | 11.0 |  55.0    9     1
c      5 | 22.0 |                2
c      6 | 22.0 |                2
c      7 | 22.0 |                2
c      8 | 33.0 |                3
c      9 | 55.0 |                4
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    17 July 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the dimension of the data values.
c
c    Input, integer N, the number of data values.
c
c    Input, double precision A(M,N), the data values.
c
c    Input, integer UNIQUE_NUM, the number of unique values
c    in A.  This value is only required for languages in which the size of
c    UNDX must be known in advance.
c
c    Input, double precision TOL, a tolerance for equality.
c
c    Output, integer UNDX(UNIQUE_NUM), the UNDX vector.
c
c    Output, integer XDNU(N), the XDNU vector.
c
      implicit none

      integer m
      integer n
      integer unique_num

      double precision a(m,n)
      double precision diff
      integer i
      integer i2
      integer i3
      integer j
      integer k
      double precision tol
      integer undx(unique_num)
      logical unique
      integer xdnu(n)
c
c  Consider entry I = 1.
c  It is unique, so set the number of unique items to K.
c  Set the K-th unique item to I.
c  Set the representative of item I to the K-th unique item.
c
      i = 1
      k = 1
      undx(k) = i
      xdnu(i) = k
c
c  Consider entry I.
c
c  If it is unique, increase the unique count K, set the
c  K-th unique item to I, and set the representative of I to K.
c
c  If it is not unique, set the representative of item I to a
c  previously determined unique item that is close to it.
c
      do i = 2, n

        unique = .true.

        do j = 1, k
          i2 = undx(j)
          diff = 0.0D+00
          do i3 = 1, m
            diff = max ( diff, abs ( a(i3,i) - a(i3,i2) ) )
          end do
          if ( diff .le. tol ) then
            unique = .false.
            xdnu(i) = j
            exit
          end if
        end do

        if ( unique ) then
          k = k + 1
          undx(k) = i
          xdnu(i) = k
        end if

      end do

      return
      end
      subroutine r8col_sorted_tol_unique ( m, n, a, tol, unique_num )

c*********************************************************************72
c
cc R8COL_SORTED_TOL_UNIQUE keeps tolerably unique elements in a sorted R8COL.
c
c  Discussion:
c
c    An R8COL is an M by N array of R8's, regarded as an array of N columns,
c    each of length M.
c
c    The columns of the array may be ascending or descending sorted.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    16 July 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns.
c
c    Input/output, double precision A(M,N).
c    On input, the sorted array of N columns of M-vectors.
c    On output, a sorted array of columns of M-vectors.
c
c    Input, double precision TOL, a tolerance for equality.
c
c    Output, integer UNIQUE_NUM, the number of unique columns.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      double precision diff
      integer i
      integer j
      integer k
      double precision tol
      logical unique
      integer unique_num

      unique_num = 0

      if ( n .le. 0 ) then
        return
      end if

      unique_num = 1

      do i = 2, n

        unique = .true.

        do j = 1, unique_num
          diff = 0.0D+00
          do k = 1, m
            diff = max ( diff, abs ( a(k,j) - a(k,i) ) )
          end do
          if ( diff .le. tol ) then
            unique = .false.
            exit
          end if
        end do

        if ( unique ) then
          unique_num = unique_num + 1
          do k = 1, m
            a(k,unique_num) = a(k,i)
          end do
        end if

      end do

      return
      end
      subroutine r8col_sorted_tol_unique_count ( m, n, a, tol,
     &  unique_num )

c*********************************************************************72
c
cc R8COL_SORTED_TOL_UNIQUE_COUNT counts tolerably unique elements in a sorted R8COL.
c
c  Discussion:
c
c    An R8COL is an M by N array of R8's, regarded as an array of N columns,
c    each of length M.
c
c    The columns of the array may be ascending or descending sorted.
c
c    If the tolerance is large enough, then the concept of uniqueness
c    can become ambiguous.  If we have a tolerance of 1.5, then in the
c    list ( 1, 2, 3, 4, 5, 6, 7, 8, 9 ) is it fair to say we have only
c    one unique entry?  That would be because 1 may be regarded as unique,
c    and then 2 is too close to 1 to be unique, and 3 is too close to 2 to
c    be unique and so on.
c
c    This seems wrongheaded.  So I prefer the idea that an item is not
c    unique under a tolerance only if it is close to something that IS unique.
c    Thus, the unique items are guaranteed to cover the space if we include
c    a disk of radius TOL around each one.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    16 July 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns.
c
c    Input, double precision A(M,N), a sorted array, containing
c    N columns of data.
c
c    Input, double precision TOL, a tolerance for equality.
c
c    Output, integer UNIQUE_NUM, the number of unique columns.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      double precision diff
      integer i
      integer i2
      integer i3
      integer j
      integer k
      double precision tol
      integer undx(n)
      logical unique
      integer unique_num
c
c  Consider entry I = 1.
c  It is unique, so set the number of unique items to K.
c  Set the K-th unique item to I.
c  Set the representative of item I to the K-th unique item.
c
      i = 1
      k = 1
      undx(k) = i
c
c  Consider entry I.
c
c  If it is unique, increase the unique count K, set the
c  K-th unique item to I, and set the representative of I to K.
c
c  If it is not unique, set the representative of item I to a
c  previously determined unique item that is close to it.
c
      do i = 2, n

        unique = .true.

        do j = 1, k
          i2 = undx(j)
          diff = 0.0D+00
          do i3 = 1, m
            diff = max ( diff, abs ( a(i3,i) - a(i3,i2) ) )
          end do
          if ( diff .le. tol ) then
            unique = .false.
            exit
          end if
        end do

        if ( unique ) then
          k = k + 1
          undx(k) = i
        end if

      end do

      unique_num = k

      return
      end
      subroutine r8col_sorted_undex ( m, n, a, unique_num, undx, xdnu )

c*********************************************************************72
c
cc R8COL_SORTED_UNDEX returns unique sorted indexes for a sorted R8COL.
c
c  Discussion:
c
c    An R8COL is an M by N array of R8's, regarded as an array of N columns,
c    each of length M.
c
c    The goal of this routine is to determine a vector UNDX,
c    which points, to the unique elements of A, in sorted order,
c    and a vector XDNU, which identifies, for each entry of A, the index of
c    the unique sorted element of A.
c
c    This is all done with index vectors, so that the elements of
c    A are never moved.
c
c    Assuming A is already sorted, we examine the entries of A in order,
c    noting the unique entries, creating the entries of XDNU and
c    UNDX as we go.
c
c    Once this process has been completed, the vector A could be
c    replaced by a compressed vector XU, containing the unique entries
c    of A in sorted order, using the formula
c
c      XU(*) = A(UNDX(*)).
c
c    We could then, if we wished, reconstruct the entire vector A, or
c    any element of it, by index, as follows:
c
c      A(I) = XU(XDNU(I)).
c
c    We could then replace A by the combination of XU and XDNU.
c
c    Later, when we need the I-th entry of A, we can locate it as
c    the XDNU(I)-th entry of XU.
c
c    Here is an example of a vector A, the unique sort and
c    inverse unique sort vectors and the compressed unique sorted vector.
c
c      I      A      XU  Undx  Xdnu
c    ----+------+------+-----+-----+
c      1 | 11.0 |  11.0    1     1
c      2 | 11.0 |  22.0    5     1
c      3 | 11.0 |  33.0    8     1
c      4 | 11.0 |  55.0    9     1
c      5 | 22.0 |                2
c      6 | 22.0 |                2
c      7 | 22.0 |                2
c      8 | 33.0 |                3
c      9 | 55.0 |                4
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    17 July 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the dimension of the data values.
c
c    Input, integer N, the number of data values.
c
c    Input, double precision A(M,N), the data values.
c
c    Input, integer UNIQUE_NUM, the number of unique values
c    in A.  This value is only required for languages in which the size of
c    UNDX must be known in advance.
c
c    Output, integer UNDX(UNIQUE_NUM), the UNDX vector.
c
c    Output, integer XDNU(N), the XDNU vector.
c
      implicit none

      integer m
      integer n
      integer unique_num

      double precision a(m,n)
      double precision diff
      integer i
      integer j
      integer k
      integer undx(unique_num)
      integer xdnu(n)
c
c  Walk through the sorted array.
c
      i = 1

      j = 1
      undx(j) = i

      xdnu(i) = j

      do i = 2, n

        diff = 0.0D+00
        do k = 1, m
          diff = max ( diff, abs ( a(k,i) - a(k,undx(j)) ) )
        end do

        if ( 0.0D+00 .ne. diff ) then
          j = j + 1
          undx(j) = i
        end if

        xdnu(i) = j

      end do

      return
      end
      subroutine r8col_sorted_unique ( m, n, a, unique_num )

c*********************************************************************72
c
cc R8COL_SORTED_UNIQUE keeps unique elements in a sorted R8COL.
c
c  Discussion:
c
c    An R8COL is an M by N array of R8's, regarded as an array of N columns,
c    each of length M.
c
c    The columns of the array may be ascending or descending sorted.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    16 July 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns.
c
c    Input/output, double precision A(M,N).
c    On input, the sorted array of N columns of M-vectors.
c    On output, a sorted array of columns of M-vectors.
c
c    Output, integer UNIQUE_NUM, the number of unique columns.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      logical equal
      integer i
      integer j1
      integer j2
      integer unique_num

      if ( n .le. 0 ) then
        unique_num = 0
        return
      end if

      j1 = 1

      do j2 = 2, n

        equal = .true.
        do i = 1, m
          if ( a(i,j1) .ne. a(i,j2) ) then
            equal = .false.
          end if
        end do

        if ( .not. equal ) then
          j1 = j1 + 1
          do i = 1, m
            a(i,j1) = a(i,j2)
          end do
        end if

      end do

      unique_num = j1

      return
      end
      subroutine r8col_sorted_unique_count ( m, n, a, unique_num )

c*********************************************************************72
c
cc R8COL_SORTED_UNIQUE_COUNT counts unique elements in a sorted R8COL.
c
c  Discussion:
c
c    An R8COL is an M by N array of R8's, regarded as an array of N columns,
c    each of length M.
c
c    The columns of the array may be ascending or descending sorted.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    16 July 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns.
c
c    Input, double precision A(M,N), a sorted array, containing
c    N columns of data.
c
c    Output, integer UNIQUE_NUM, the number of unique columns.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      logical equal
      integer i
      integer j1
      integer j2
      integer unique_num

      unique_num = 0

      if ( n .le. 0 ) then
        return
      end if

      unique_num = 1
      j1 = 1

      do j2 = 2, n

        equal = .true.
        do i = 1, m
          if ( a(i,j1) .ne. a(i,j2) ) then
            equal = .false.
          end if
        end do

        if ( .not. equal ) then
          unique_num = unique_num + 1
          j1 = j2
        end if

      end do

      return
      end
      subroutine r8col_sortr_a ( m, n, a, key )

c*********************************************************************72
c
cc R8COL_SORTR_A ascending sorts one column of an R8COL, adjusting all columns.
c
c  Discussion:
c
c    An R8COL is an M by N array of R8's, regarded as an array of N columns,
c    each of length M.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    16 July 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns.
c
c    Input/output, double precision A(M,N).
c    On input, an unsorted M by N array.
c    On output, rows of the array have been shifted in such
c    a way that column KEY of the array is in nondecreasing order.
c
c    Input, integer KEY, the column in which the "key" value
c    is stored.  On output, column KEY of the array will be
c    in nondecreasing order.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer indx
      integer isgn
      integer j
      integer key

      if ( m .le. 0 ) then
        return
      end if

      if ( key .lt. 1 .or. n .lt. key ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8COL_SORTR_A - Fatal error!'
        write ( *, '(a)' ) '  KEY is not a legal column index.'
        write ( *, '(a,i8)' ) '  KEY = ', key
        write ( *, '(a,i8)' ) '  N = ', n
        stop 1
      end if
c
c  Initialize.
c
      i = 0
      indx = 0
      isgn = 0
      j = 0
c
c  Call the external heap sorter.
c
10    continue

        call sort_heap_external ( m, indx, i, j, isgn )
c
c  Interchange the I and J objects.
c
        if ( 0 .lt. indx ) then

          call r8row_swap ( m, n, a, i, j )
c
c  Compare the I and J objects.
c
        else if ( indx .lt. 0 ) then

          if ( a(i,key) .lt. a(j,key) ) then
            isgn = -1
          else
            isgn = +1
          end if

        else if ( indx .eq. 0 ) then

          go to 20

        end if

      go to 10

20    continue

      return
      end
      subroutine r8col_sum ( m, n, a, colsum )

c*********************************************************************72
c
cc R8COL_SUM sums the columns of an R8COL.
c
c  Discussion:
c
c    An R8COL is an M by N array of R8's, regarded as an array of N columns,
c    each of length M.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    08 May 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns.
c
c    Input, double precision A(M,N), the array to be examined.
c
c    Output, double precision COLSUM(N), the sums of the columns.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      double precision colsum(n)
      integer i
      integer j

      do j = 1, n
        colsum(j) = 0.0D+00
        do i = 1, m
          colsum(j) = colsum(j) + a(i,j)
        end do
      end do

      return
      end
      subroutine r8col_swap ( m, n, a, j1, j2 )

c*********************************************************************72
c
cc R8COL_SWAP swaps columns I and J of an R8COL.
c
c  Discussion:
c
c    An R8COL is an M by N array of R8's, regarded as an array of N columns,
c    each of length M.
c
c  Example:
c
c    Input:
c
c      M = 3, N = 4, J1 = 2, J2 = 4
c
c      A = (
c        1.  2.  3.  4.
c        5.  6.  7.  8.
c        9. 10. 11. 12. )
c
c    Output:
c
c      A = (
c        1.  4.  3.  2.
c        5.  8.  7.  6.
c        9. 12. 11. 10. )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    05 March 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns.
c
c    Input/output, double precision A(M,N), the M by N array.
c
c    Input, integer J1, J2, the columns to be swapped.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer j1
      integer j2
      double precision temp

      if ( j1 .lt. 1 .or. n .lt. j1 .or. j2 .lt. 1 .or. n .lt. j2 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8COL_SWAP - Fatal error!'
        write ( *, '(a)' ) '  J1 or J2 is out of bounds.'
        write ( *, '(a,i8)' ) '  J1 =    ', j1
        write ( *, '(a,i8)' ) '  J2 =    ', j2
        write ( *, '(a,i8)' ) '  NCOL = ', n
        stop 1
      end if

      if ( j1 .eq. j2 ) then
        return
      end if

      do i = 1, m
        temp = a(i,j1)
        a(i,j1) = a(i,j2)
        a(i,j2) = temp
      end do

      return
      end
      subroutine r8col_to_r8vec ( m, n, a, x )

c*********************************************************************72
c
cc R8COL_TO_R8VEC converts an R8COL to an R8VEC.
c
c  Discussion:
c
c    An R8COL is an M by N array of R8's, regarded as an array of N columns,
c    each of length M.
c
c    An R8VEC is a vector of R8's.
c
c  Example:
c
c    M = 3, N = 4
c
c    A =
c      11 12 13 14
c      21 22 23 24
c      31 32 33 34
c
c    X = ( 11, 21, 31, 12, 22, 32, 13, 23, 33, 14, 24, 34 )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    08 May 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns.
c
c    Input, double precision A(M,N), the array.
c
c    Output, double precision X(M*N), a vector containing the N columns of A.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer j
      integer k
      double precision x(m*n)

      k = 1
      do j = 1, n
        do i = 1, m
          x(k) = a(i,j)
          k = k + 1
        end do
      end do

      return
      end
      subroutine r8col_tol_undex ( m, n, a, unique_num, tol, undx,
     &  xdnu )

c*********************************************************************72
c
cc R8COL_TOL_UNDEX indexes tolerably unique entries of an R8COL.
c
c  Discussion:
c
c    An R8COL is an M by N array of R8's, regarded as an array of N columns,
c    each of length M.
c
c    The goal of this routine is to determine a vector UNDX,
c    which points to the unique elements of A, in sorted order,
c    and a vector XDNU, which identifies, for each entry of A, the index of
c    the unique sorted element of A.
c
c    This is all done with index vectors, so that the elements of
c    A are never moved.
c
c    The first step of the algorithm requires the indexed sorting
c    of A, which creates arrays INDX and XDNI.  (If all the entries
c    of A are unique, then these arrays are the same as UNDX and XDNU.)
c
c    We then use INDX to examine the entries of A in sorted order,
c    noting the unique entries, creating the entries of XDNU and
c    UNDX as we go.
c
c    Once this process has been completed, the object A could be
c    replaced by a compressed object XU, containing the unique entries
c    of X in sorted order, using the formula
c
c      XU(*) = A(UNDX(*)).
c
c    We could then, if we wished, reconstruct the entire vector A, or
c    any element of it, by index, as follows:
c
c      A(I) = XU(XDNU(I)).
c
c    We could then replace A by the combination of XU and XDNU.
c
c    Later, when we need the I-th entry of A, we can locate it as
c    the XDNU(I)-th entry of XU.
c
c    Here is an example of a vector A, the sort and inverse sort
c    index vectors, and the unique sort and inverse unique sort vectors
c    and the compressed unique sorted vector.
c
c      I    A   Indx  Xdni      XU   Undx  Xdnu
c    ----+-----+-----+-----+--------+-----+-----+
c      1 | 11.     1     1 |    11.     1     1
c      2 | 22.     3     5 |    22.     2     2
c      3 | 11.     6     2 |    33.     4     1
c      4 | 33.     9     8 |    55.     5     3
c      5 | 55.     2     9 |                  4
c      6 | 11.     7     3 |                  1
c      7 | 22.     8     6 |                  2
c      8 | 22.     4     7 |                  2
c      9 | 11.     5     4 |                  1
c
c    INDX(2) = 3 means that sorted item(2) is A(3).
c    XDNI(2) = 5 means that A(2) is sorted item(5).
c
c    UNDX(3) = 4 means that unique sorted item(3) is at A(4).
c    XDNU(8) = 2 means that A(8) is at unique sorted item(2).
c
c    XU(XDNU(I))) = A(I).
c    XU(I)        = A(UNDX(I)).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    17 July 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the dimension of the data values.
c
c    Input, integer N, the number of data values.
c
c    Input, double precision A(M,N), the data values.
c
c    Input, integer UNIQUE_NUM, the number of unique values
c    in A.  This value is only required for languages in which the size of
c    UNDX must be known in advance.
c
c    Input, double precision TOL, a tolerance for equality.
c
c    Output, integer UNDX(UNIQUE_NUM), the UNDX vector.
c
c    Output, integer XDNU(N), the XDNU vector.
c
      implicit none

      integer m
      integer n
      integer unique_num

      double precision a(m,n)
      double precision diff
      integer i
      integer i2
      integer indx(n)
      integer j
      integer k
      double precision tol
      integer undx(unique_num)
      logical unique
      integer xdnu(n)
c
c  Implicitly sort the array.
c
      call r8col_sort_heap_index_a ( m, n, a, indx )
c
c  Consider entry I = 1.
c  It is unique, so set the number of unique items to K.
c  Set the K-th unique item to I.
c  Set the representative of item I to the K-th unique item.
c
      i = 1
      k = 1
      undx(k) = indx(i)
      xdnu(indx(i)) = k
c
c  Consider entry I.
c
c  If it is unique, increase the unique count K, set the
c  K-th unique item to I, and set the representative of I to K.
c
c  If it is not unique, set the representative of item I to a
c  previously determined unique item that is close to it.
c
      do i = 2, n

        unique = .true.

        do j = 1, k

          diff = 0.0D+00
          do i2 = 1, m
            diff = max ( diff, abs ( a(i2,indx(i)) - a(i2,undx(j)) ) )
          end do

          if ( diff .le. tol ) then
            unique = .false.
            xdnu(indx(i)) = j
            go to 10
          end if

        end do

        if ( unique ) then
          k = k + 1
          undx(k) = indx(i)
          xdnu(indx(i)) = k
        end if

10      continue

      end do

      return
      end
      subroutine r8col_tol_unique_count ( m, n, a, tol, unique_num )

c*********************************************************************72
c
cc R8COL_TOL_UNIQUE_COUNT counts tolerably unique entries in an R8COL.
c
c  Discussion:
c
c    An R8COL is an M by N array of R8's, regarded as an array of N columns,
c    each of length M.
c
c    Because the array is unsorted, this algorithm is O(N^2).
c
c    If the tolerance is large enough, then the concept of uniqueness
c    can become ambiguous.  If we have a tolerance of 1.5, then in the
c    list ( 1, 2, 3, 4, 5, 6, 7, 8, 9 ) is it fair to say we have only
c    one unique entry?  That would be because 1 may be regarded as unique,
c    and then 2 is too close to 1 to be unique, and 3 is too close to 2 to
c    be unique and so on.
c
c    This seems wrongheaded.  So I prefer the idea that an item is not
c    unique under a tolerance only if it is close to something that IS unique.
c    Thus, the unique items are guaranteed to cover the space if we include
c    a disk of radius TOL around each one.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    17 July 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the number of rows.
c
c    Input, integer N, the number of columns.
c
c    Input, double precision A(M,N), the array of N columns of data.
c
c    Input, double precision TOL, a tolerance for equality.
c
c    Output, integer UNIQUE_NUM, the number of unique columns.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      double precision diff
      integer i
      integer i2
      integer indx(n)
      integer j
      integer k
      double precision tol
      integer undx(n)
      logical unique
      integer unique_num
c
c  Implicitly sort the array.
c
      call r8col_sort_heap_index_a ( m, n, a, indx )
c
c  Consider entry I = 1.
c  It is unique, so set the number of unique items to K.
c  Set the K-th unique item to I.
c  Set the representative of item I to the K-th unique item.
c
      i = 1
      k = 1
      undx(k) = indx(i)
c
c  Consider entry I.
c
c  If it is unique, increase the unique count K, set the
c  K-th unique item to I, and set the representative of I to K.
c
c  If it is not unique, set the representative of item I to a
c  previously determined unique item that is close to it.
c
      do i = 2, n

        unique = .true.

        do j = 1, k

          diff = 0.0D+00
          do i2 = 1, m
            diff = max ( diff, abs ( a(i2,indx(i)) - a(i2,undx(j)) ) )
          end do

          if ( diff .le. tol ) then
            unique = .false.
            go to 10
          end if

        end do

        if ( unique ) then
          k = k + 1
          undx(k) = indx(i)
        end if

10      continue

      end do

      unique_num = k

      return
      end
      subroutine r8col_tol_unique_index ( m, n, a, tol, unique_index )

c*********************************************************************72
c
cc R8COL_TOL_UNIQUE_INDEX indexes tolerably unique entries in an R8COL.
c
c  Discussion:
c
c    An R8COL is an M by N array of R8 values.
c    It is regarded as an array of N columns of length M.
c
c    For element A(1:M,J) of the matrix, UNIQUE_INDEX(J) is the uniqueness index
c    of A(1:M,J).  That is, if A_UNIQUE contains the unique elements of A,
c    gathered in order, then
c
c      A_UNIQUE ( 1:M, UNIQUE_INDEX(J) ) = A(1:M,J)
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    17 July 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns of A.
c    The length of an "element" of A, and the number of "elements".
c
c    Input, double precision A(M,N), the array.
c
c    Input, double precision TOL, a tolerance for equality.
c
c    Output, integer UNIQUE_INDEX(N), the first occurrence index.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      double precision diff
      integer i
      integer j1
      integer j2
      double precision tol
      integer unique_index(n)
      integer unique_num

      do i = 1, n
        unique_index(i) = -1
      end do
      unique_num = 0

      do j1 = 1, n

        if ( unique_index(j1) .eq. -1 ) then

          unique_num = unique_num + 1
          unique_index(j1) = unique_num

          do j2 = j1 + 1, n
            diff = 0.0D+00
            do i = 1, m
              diff = max ( diff, abs ( a(i,j1) - a(i,j2) ) )
            end do
            if ( diff .le. tol ) then
              unique_index(j2) = unique_num
            end if
          end do

        end if

      end do

      return
      end
      subroutine r8col_undex ( m, n, a, unique_num, undx, xdnu )

c*********************************************************************72
c
cc R8COL_UNDEX indexes unique entries of an R8COL.
c
c  Discussion:
c
c    An R8COL is an M by N array of R8's, regarded as an array of N columns,
c    each of length M.
c
c    The goal of this routine is to determine a vector UNDX,
c    which points to the unique elements of A, in sorted order,
c    and a vector XDNU, which identifies, for each entry of A, the index of
c    the unique sorted element of A.
c
c    This is all done with index vectors, so that the elements of
c    A are never moved.
c
c    The first step of the algorithm requires the indexed sorting
c    of A, which creates arrays INDX and XDNI.  (If all the entries
c    of A are unique, then these arrays are the same as UNDX and XDNU.)
c
c    We then use INDX to examine the entries of A in sorted order,
c    noting the unique entries, creating the entries of XDNU and
c    UNDX as we go.
c
c    Once this process has been completed, the object A could be
c    replaced by a compressed object XU, containing the unique entries
c    of X in sorted order, using the formula
c
c      XU(*) = A(UNDX(*)).
c
c    We could then, if we wished, reconstruct the entire vector A, or
c    any element of it, by index, as follows:
c
c      A(I) = XU(XDNU(I)).
c
c    We could then replace A by the combination of XU and XDNU.
c
c    Later, when we need the I-th entry of A, we can locate it as
c    the XDNU(I)-th entry of XU.
c
c    Here is an example of a vector A, the sort and inverse sort
c    index vectors, and the unique sort and inverse unique sort vectors
c    and the compressed unique sorted vector.
c
c      I    A   Indx  Xdni      XU   Undx  Xdnu
c    ----+-----+-----+-----+--------+-----+-----+
c      1 | 11.     1     1 |    11.     1     1
c      2 | 22.     3     5 |    22.     2     2
c      3 | 11.     6     2 |    33.     4     1
c      4 | 33.     9     8 |    55.     5     3
c      5 | 55.     2     9 |                  4
c      6 | 11.     7     3 |                  1
c      7 | 22.     8     6 |                  2
c      8 | 22.     4     7 |                  2
c      9 | 11.     5     4 |                  1
c
c    INDX(2) = 3 means that sorted item(2) is A(3).
c    XDNI(2) = 5 means that A(2) is sorted item(5).
c
c    UNDX(3) = 4 means that unique sorted item(3) is at A(4).
c    XDNU(8) = 2 means that A(8) is at unique sorted item(2).
c
c    XU(XDNU(I))) = A(I).
c    XU(I)        = A(UNDX(I)).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    17 July 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the dimension of the data values.
c
c    Input, integer N, the number of data values.
c
c    Input, double precision A(M,N), the data values.
c
c    Input, integer UNIQUE_NUM, the number of unique values
c    in A.  This value is only required for languages in which the size of
c    UNDX must be known in advance.
c
c    Output, integer UNDX(UNIQUE_NUM), the UNDX vector.
c
c    Output, integer XDNU(N), the XDNU vector.
c
      implicit none

      integer m
      integer n
      integer unique_num

      double precision a(m,n)
      double precision diff
      integer i
      integer indx(n)
      integer j
      integer k
      integer undx(unique_num)
      integer xdnu(n)
c
c  Implicitly sort the array.
c
      call r8col_sort_heap_index_a ( m, n, a, indx )
c
c  Walk through the implicitly sorted array.
c
      i = 1
      j = 1
      undx(j) = indx(i)
      xdnu(indx(i)) = j

      do i = 2, n

        diff = 0.0D+00
        do k = 1, m
          diff = max ( diff, abs ( a(k,indx(i)) - a(k,undx(j)) ) )
        end do

        if ( 0.0D+00 .lt. diff ) then
          j = j + 1
          undx(j) = indx(i)
        end if

        xdnu(indx(i)) = j

      end do

      return
      end
      subroutine r8col_uniform_abvec ( m, n, a, b, seed, r )

c*********************************************************************72
c
cc R8COL_UNIFORM_ABVEC fills an R8COL with pseudorandom values.
c
c  Discussion:
c
c    An R8COL is an array of R8 values, regarded as a set of column vectors.
c
c    The user specifies a minimum and maximum value for each row.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    19 December 2011
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Paul Bratley, Bennett Fox, Linus Schrage,
c    A Guide to Simulation,
c    Springer Verlag, pages 201-202, 1983.
c
c    Bennett Fox,
c    Algorithm 647:
c    Implementation and Relative Efficiency of Quasirandom
c    Sequence Generators,
c    ACM Transactions on Mathematical Software,
c    Volume 12, Number 4, pages 362-376, 1986.
c
c    Peter Lewis, Allen Goodman, James Miller,
c    A Pseudo-Random Number Generator for the System/360,
c    IBM Systems Journal,
c    Volume 8, pages 136-143, 1969.
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns in
c    the array.
c
c    Input, double precision A(M), B(M), the lower and upper limits.
c
c    Input/output, integer SEED, the "seed" value, which
c    should NOT be 0.  On output, SEED has been updated.
c
c    Output, double precision R(M,N), the array of pseudorandom values.
c
      implicit none

      integer m
      integer n

      double precision a(m)
      double precision b(m)
      integer i
      integer i4_huge
      parameter ( i4_huge = 2147483647 )
      integer j
      integer k
      integer seed
      double precision r(m,n)

      do j = 1, n

        do i = 1, m

          k = seed / 127773

          seed = 16807 * ( seed - k * 127773 ) - k * 2836

          if ( seed .lt. 0 ) then
            seed = seed + i4_huge
          end if

          r(i,j) = a(i) 
     &      + ( b(i) - a(i) ) * dble ( seed ) * 4.656612875D-10

        end do
      end do

      return
      end
      subroutine r8col_unique_count ( m, n, a, unique_num )

c*********************************************************************72
c
cc R8COL_UNIQUE_COUNT counts the unique columns in an unsorted R8COL.
c
c  Discussion:
c
c    An R8COL is an M by N array of R8's, regarded as an array of N columns,
c    each of length M.
c
c    Because the array is unsorted, this algorithm is O(N^2).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    17 July 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the number of rows.
c
c    Input, integer N, the number of columns.
c
c    Input, double precision A(M,N), the array of N columns of data.
c
c    Output, integer UNIQUE_NUM, the number of unique columns.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      double precision diff
      integer i
      integer j1
      integer j2
      logical unique(n)
      integer unique_num

      unique_num = 0

      do j1 = 1, n

        unique_num = unique_num + 1
        unique(j1) = .true.

        do j2 = 1, j1 - 1

          if ( unique(j2) ) then

            diff = 0.0D+00
            do i = 1, m
              diff = max ( diff, abs ( a(i,j1) - a(i,j2) ) )
            end do

            if ( diff .eq. 0.0D+00 ) then
              unique_num = unique_num - 1
              unique(j1) = .false.
              go to 10
            end if

          end if

        end do

10      continue

      end do

      return
      end
      subroutine r8col_unique_index ( m, n, a, unique_index )

c*********************************************************************72
c
cc R8COL_UNIQUE_INDEX indexes the first occurrence of values in an R8COL.
c
c  Discussion:
c
c    An R8COL is an M by N array of R8 values.
c    It is regarded as an array of N columns of length M.
c
c    For element A(1:M,J) of the matrix, UNIQUE_INDEX(J) is the uniqueness index
c    of A(1:M,J).  That is, if A_UNIQUE contains the unique elements of A,
c    gathered in order, then
c
c      A_UNIQUE ( 1:M, UNIQUE_INDEX(J) ) = A(1:M,J)
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    17 July 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns of A.
c    The length of an "element" of A, and the number of "elements".
c
c    Input, double precision A(M,N), the array.
c
c    Output, integer UNIQUE_INDEX(N), the first occurrence index.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      double precision diff
      integer i
      integer j1
      integer j2
      integer unique_index(n)
      integer unique_num

      do i = 1, n
        unique_index(i) = -1
      end do
      unique_num = 0

      do j1 = 1, n

        if ( unique_index(j1) .eq. -1 ) then

          unique_num = unique_num + 1
          unique_index(j1) = unique_num

          do j2 = j1 + 1, n
            diff = 0.0D+00
            do i = 1, m
              diff = max ( diff, abs ( a(i,j1) - a(i,j2) ) )
            end do
            if ( diff .eq. 0.0D+00 ) then
              unique_index(j2) = unique_num
            end if
          end do

        end if

      end do

      return
      end
      subroutine r8col_variance ( m, n, a, variance )

c*********************************************************************72
c
cc R8COL_VARIANCE returns the variances of an R8COL.
c
c  Discussion:
c
c    An R8COL is an M by N array of R8 values, regarded
c    as an array of N columns of length M.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    13 January 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns in
c    the array.
c
c    Input, double precision A(M,N), the array whose variances are desired.
c
c    Output, double precision VARIANCE(N), the variances of the rows.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      double precision a_sum
      integer i
      integer j
      double precision mean
      double precision variance(n)

      do j = 1, n

        a_sum = 0.0D+00
        do i = 1, m
          a_sum = a_sum + a(i,j)
        end do
        mean = a_sum / dble ( m )

        variance(j) = 0.0D+00
        do i = 1, m
          variance(j) = variance(j) + ( a(i,j) - mean ) ** 2
        end do

        if ( 1 .lt. m ) then
          variance(j) = variance(j) / dble ( m - 1 )
        else
          variance(j) = 0.0D+00
        end if

      end do

      return
      end
      function r8r8_compare ( x1, y1, x2, y2 )

c*********************************************************************72
c
cc R8R8_COMPARE compares two R8R8's.
c
c  Discussion:
c
c    An R8R8 is simply a pair of R8 values, stored separately.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    08 May 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision X1, Y1, the first vector.
c
c    Input, double precision X2, Y2, the second vector.
c
c    Output, integer R8R8_COMPARE:
c    -1, (X1,Y1) < (X2,Y2);
c     0, (X1,Y1) = (X2,Y2);
c    +1, (X1,Y1) > (X2,Y2).
c
      implicit none

      integer compare
      integer r8r8_compare
      double precision x1
      double precision x2
      double precision y1
      double precision y2

      if ( x1 .lt. x2 ) then
        compare = -1
      else if ( x2 .lt. x1 ) then
        compare = +1
      else if ( y1 .lt. y2 ) then
        compare = -1
      else if ( y2 .lt. y1 ) then
        compare = +1
      else
        compare = 0
      end if

      r8r8_compare = compare

      return
      end
      subroutine r8r8_print ( a1, a2, title )

c*********************************************************************72
c
cc R8R8_PRINT prints an R8R8.
c
c  Discussion:
c
c    An R8R8 is simply a pair of R8R8's, stored separately.
c
c    A format is used which suggests a coordinate pair:
c
c  Example:
c
c    Center : ( 1.23, 7.45 )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    12 September 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision A1, A2, the coordinates of the vector.
c
c    Input, character * ( * ) TITLE, a title.
c
      implicit none

      double precision a1
      double precision a2
      character * ( * ) title

      if ( 0 .lt. len_trim ( title ) ) then
        write ( *, '( 2x, a, a4, g14.6, a1, g14.6, a1 )' ) 
     &    trim ( title ), ' : (', a1, ',', a2, ')'
      else
        write ( *, '( 2x, a1, g14.6, a1, g14.6, a1 )' ) '(', a1, ',', a2, ')'
      end if

      return
      end
      function r8r8r8_compare ( x1, y1, z1, x2, y2, z2 )

c*********************************************************************72
c
cc R8R8R8_COMPARE compares two R8R8R8's.
c
c  Discussion:
c
c    An R8R8R8 is simply 3 R8 values, stored as scalars.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    03 December 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision X1, Y1, Z1, the first vector.
c
c    Input, double precision X2, Y2, Z2, the second vector.
c
c    Output, integer R8R8R8_COMPARE:
c    -1, (X1,Y1,Z1) .lt. (X2,Y2,Z2);
c     0, (X1,Y1,Z1) = (X2,Y2,Z2);
c    +1, (X1,Y1,Z1) > (X2,Y2,Z2).
c
      implicit none

      integer compare
      integer r8r8r8_compare
      double precision x1
      double precision x2
      double precision y1
      double precision y2
      double precision z1
      double precision z2

      if ( x1 .lt. x2 ) then
        compare = -1
      else if ( x2 .lt. x1 ) then
        compare = +1
      else if ( y1 .lt. y2 ) then
        compare = -1
      else if ( y2 .lt. y1 ) then
        compare = +1
      else if ( z1 .lt. z2 ) then
        compare = -1
      else if ( z2 .lt. z1 ) then
        compare = +1
      else
        compare = 0
      end if

      r8r8r8_compare = compare

      return
      end
      subroutine r8r8r8vec_index_insert_unique ( n_max, n, x, y, z, 
     &  indx, xval, yval, zval, ival, ierror )

c*********************************************************************72
c
cc R8R8R8VEC_INDEX_INSERT_UNIQUE inserts unique R8R8R in an indexed sorted list.
c
c  Discussion:
c
c    An R8R8R8VEC is set of N R8R8R8 items.
c
c    An R8R8R8 is simply 3 R8 values, stored as scalars.
c
c    If the input value does not occur in the current list, it is added,
c    and N, X, Y, Z and INDX are updated.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    03 December 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N_MAX, the maximum size of the list.
c
c    Input/output, integer N, the size of the list.
c
c    Input/output, double precision X(N), Y(N), Z(N), the R8R8R8 vector.
c
c    Input/output, integer INDX(N), the sort index of the list.
c
c    Input, double precision XVAL, YVAL, ZVAL, the value to be inserted
c    if it is not already in the list.
c
c    Output, integer IVAL, the index in X, Y, Z corresponding
c    to the value XVAL, YVAL, ZVAL.
c
c    Output, integer IERROR, 0 for no error, 1 if an error
c    occurred.
c
      implicit none

      integer n_max

      integer equal
      integer ierror
      integer indx(n_max)
      integer ival
      integer less
      integer more
      integer n
      double precision x(n_max)
      double precision xval
      double precision y(n_max)
      double precision yval
      double precision z(n_max)
      double precision zval

      ierror = 0

      if ( n .le. 0 ) then

        if ( n_max .le. 0 ) then
          ierror = 1
          write ( *, '(a)' ) ' '
          write ( *, '(a)' ) 
     &      'R8R8R8VEC_INDEX_INSERT_UNIQUE - Fatal error!'
          write ( *, '(a)' ) '  Not enough space to store new data.'
          return
        end if

        n = 1
        x(1) = xval
        y(1) = yval
        z(1) = zval
        indx(1) = 1
        ival = 1
        return

      end if
c
c  Does ( XVAL, YVAL, ZVAL ) already occur in ( X, Y, Z)?
c
      call r8r8r8vec_index_search ( n, x, y, z, indx, xval, yval, zval, 
     &  less, equal, more )

      if ( equal .eq. 0 ) then

        if ( n_max .le. n ) then
          ierror = 1
          write ( *, '(a)' ) ' '
          write ( *, '(a)' ) 
     &      'R8R8R8VEC_INDEX_INSERT_UNIQUE - Fatal error!'
          write ( *, '(a)' ) '  Not enough space to store new data.'
          return
        end if

        x(n+1) = xval
        y(n+1) = yval
        z(n+1) = zval
        ival = n + 1
        indx(n+1:more+1:-1) = indx(n:more:-1)
        indx(more) = n + 1
        n = n + 1

      else

        ival = indx(equal)

      end if

      return
      end
      subroutine r8r8r8vec_index_search ( n, x, y, z, indx, xval, yval, 
     &  zval, less, equal, more )

c*********************************************************************72
c
cc R8R8R8VEC_INDEX_SEARCH searches for R8R8R8 value in an indexed sorted list.
c
c  Discussion:
c
c    An R8R8R8VEC is set of N R8R8R8 items.
c
c    An R8R8R8 is simply 3 R8 values, stored as scalars.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    03 December 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the size of the list.
c
c    Input, double precision X(N), Y(N), Z(N), the list.
c
c    Input, integer INDX(N), the sort index of the list.
c
c    Input, double precision XVAL, YVAL, ZVAL, the value to be sought.
c
c    Output, integer LESS, EQUAL, MORE, the indexes in INDX of the
c    entries of X that are just less than, equal to, and just greater
c    than XVAL.  If XVAL does not occur in X, then EQUAL is zero.
c    If XVAL is the minimum entry of X, then LESS is 0.  If XVAL
c    is the greatest entry of X, then MORE is N+1.
c
      implicit none

      integer n

      integer compare
      integer r8r8r8_compare
      integer equal
      integer hi
      integer indx(n)
      integer less
      integer lo
      integer mid
      integer more
      double precision x(n)
      double precision xhi
      double precision xlo
      double precision xmid
      double precision xval
      double precision y(n)
      double precision yhi
      double precision ylo
      double precision ymid
      double precision yval
      double precision z(n)
      double precision zhi
      double precision zlo
      double precision zmid
      double precision zval

      if ( n .le. 0 ) then
        less = 0
        equal = 0
        more = 0
        return
      end if

      lo = 1
      hi = n

      xlo = x(indx(lo))
      ylo = y(indx(lo))
      zlo = z(indx(lo))

      xhi = x(indx(hi))
      yhi = y(indx(hi))
      zhi = z(indx(hi))

      compare = r8r8r8_compare ( xval, yval, zval, xlo, ylo, zlo )

      if ( compare .eq. -1 ) then
        less = 0
        equal = 0
        more = 1
        return
      else if ( compare .eq. 0 ) then
        less = 0
        equal = 1
        more = 2
        return
      end if

      compare = r8r8r8_compare ( xval, yval, zval, xhi, yhi, zhi )

      if ( compare .eq. 1 ) then
        less = n
        equal = 0
        more = n + 1
        return
      else if ( compare .eq. 0 ) then
        less = n - 1
        equal = n
        more = n + 1
        return
      end if

10    continue

        if ( lo + 1 .eq. hi ) then
          less = lo
          equal = 0
          more = hi
          go to 20
        end if

        mid = ( lo + hi ) / 2
        xmid = x(indx(mid))
        ymid = y(indx(mid))
        zmid = z(indx(mid))

        compare = r8r8r8_compare ( xval, yval, zval, xmid, ymid, zmid )

        if ( compare .eq. 0 ) then
          equal = mid
          less = mid - 1
          more = mid + 1
          return
        else if ( compare .eq. -1 ) then
          hi = mid
        else if ( compare .eq. +1 ) then
          lo = mid
        end if

      go to 10

20    continue

      return
      end
      subroutine r8r8vec_index_insert_unique ( n_max, n, x, y, indx, 
     &  xval, yval, ival, ierror )

c*********************************************************************72
c
cc R8R8VEC_INDEX_INSERT_UNIQUE inserts a unique R8R8 in an indexed sorted list.
c
c  Discussion:
c
c    An R8R8VEC is set of N R8R8 items.
c
c    An R8R8 is simply 2 R8 values, stored as scalars.
c
c    If the input value does not occur in the current list, it is added,
c    and N, X, Y and INDX are updated.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    03 December 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N_MAX, the maximum size of the list.
c
c    Input/output, integer N, the size of the list.
c
c    Input/output, double precision X(N), Y(N), the list of R8R8 vectors.
c
c    Input/output, integer INDX(N), the sort index of the list.
c
c    Input, double precision XVAL, YVAL, the value to be inserted if it is
c    not already in the list.
c
c    Output, integer IVAL, the index in X, Y corresponding to the
c    value XVAL, YVAL.
c
c    Output, integer IERROR, 0 for no error, 1 if an
c    error occurred.
c
      implicit none

      integer n_max

      integer equal
      integer ierror
      integer indx(n_max)
      integer ival
      integer less
      integer more
      integer n
      double precision x(n_max)
      double precision xval
      double precision y(n_max)
      double precision yval

      ierror = 0

      if ( n .le. 0 ) then

        if ( n_max .le. 0 ) then
          ierror = 1
          write ( *, '(a)' ) ' '
          write ( *, '(a)' ) 
     &      'R8R8VEC_INDEX_INSERT_UNIQUE - Fatal error!'
          write ( *, '(a)' ) '  Not enough space to store new data.'
          return
        end if

        n = 1
        x(1) = xval
        y(1) = yval
        indx(1) = 1
        ival = 1
        return

      end if
c
c  Does ( XVAL, YVAL ) already occur in ( X, Y )?
c
      call r8r8vec_index_search ( n, x, y, indx, xval, yval, less, 
     &  equal, more )

      if ( equal .eq. 0 ) then

        if ( n_max .le. n ) then
          ierror = 1
          write ( *, '(a)' ) ' '
          write ( *, '(a)' ) 
     &      'R8R8VEC_INDEX_INSERT_UNIQUE - Fatal error!'
          write ( *, '(a)' ) '  Not enough space to store new data.'
          return
        end if

        x(n+1) = xval
        y(n+1) = yval
        ival = n + 1
        indx(n+1:more+1:-1) = indx(n:more:-1)
        indx(more) = n + 1
        n = n + 1

      else

        ival = indx(equal)

      end if

      return
      end
      subroutine r8r8vec_index_search ( n, x, y, indx, xval, yval, 
     &  less, equal, more )

c*********************************************************************72
c
cc R8R8VEC_INDEX_SEARCH searches for an R8R8 in an indexed sorted list.
c
c  Discussion:
c
c    An R8R8VEC is set of N R8R8 items.
c
c    An R8R8 is simply 2 R8 values, stored as scalars.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    03 December 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the size of the current list.
c
c    Input, double precision X(N), Y(N), the list.
c
c    Input, integer INDX(N), the sort index of the list.
c
c    Input, double precision XVAL, YVAL, the value to be sought.
c
c    Output, integer LESS, EQUAL, MORE, the indexes in INDX of the
c    entries of X that are just less than, equal to, and just greater
c    than XVAL.  If XVAL does not occur in X, then EQUAL is zero.
c    If XVAL is the minimum entry of X, then LESS is 0.  If XVAL
c    is the greatest entry of X, then MORE is N+1.
c
      implicit none

      integer n

      integer compare
      integer r8r8_compare
      integer equal
      integer hi
      integer indx(n)
      integer less
      integer lo
      integer mid
      integer more
      double precision x(n)
      double precision xhi
      double precision xlo
      double precision xmid
      double precision xval
      double precision y(n)
      double precision yhi
      double precision ylo
      double precision ymid
      double precision yval

      if ( n .le. 0 ) then
        less = 0
        equal = 0
        more = 0
        return
      end if

      lo = 1
      hi = n

      xlo = x(indx(lo))
      ylo = y(indx(lo))

      xhi = x(indx(hi))
      yhi = y(indx(hi))

      compare = r8r8_compare ( xval, yval, xlo, ylo )

      if ( compare .eq. -1 ) then
        less = 0
        equal = 0
        more = 1
        return
      else if ( compare .eq. 0 ) then
        less = 0
        equal = 1
        more = 2
        return
      end if

      compare = r8r8_compare ( xval, yval, xhi, yhi )

      if ( compare .eq. 1 ) then
        less = n
        equal = 0
        more = n + 1
        return
      else if ( compare .eq. 0 ) then
        less = n - 1
        equal = n
        more = n + 1
        return
      end if

10    continue

        if ( lo + 1 .eq. hi ) then
          less = lo
          equal = 0
          more = hi
          go to 20
        end if

        mid = ( lo + hi ) / 2
        xmid = x(indx(mid))
        ymid = y(indx(mid))

        compare = r8r8_compare ( xval, yval, xmid, ymid )

        if ( compare .eq. 0 ) then
          equal = mid
          less = mid - 1
          more = mid + 1
          return
        else if ( compare .eq. -1 ) then
          hi = mid
        else if ( compare .eq. +1 ) then
          lo = mid
        end if

      go to 10

20    continue

      return
      end
      subroutine r8int_to_r8int ( rmin, rmax, r, r2min, r2max, r2 )

c*********************************************************************72
c
cc R8INT_TO_R8INT maps one R8INT to another.
c
c  Discussion:
c
c    The formula used is
c
c      R2 := R2MIN + ( R2MAX - R2MIN ) * ( R - RMIN ) / ( RMAX - RMIN )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    03 December 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision RMIN, RMAX, the first range.
c
c    Input, double precision R, the number to be converted.
c
c    Input, double precision R2MAX, R2MIN, the second range.
c
c    Output, double precision R2, the corresponding value in
c    the range [R2MIN,R2MAX].
c
      implicit none

      double precision r
      double precision rmax
      double precision rmin
      double precision r2
      double precision r2max
      double precision r2min

      if ( rmax .eq. rmin ) then

        r2 = ( r2max + r2min ) / 2.0D+00

      else

        r2 = ( ( ( rmax - r        ) * r2min   
     &         + (        r - rmin ) * r2max ) 
     &         / ( rmax     - rmin ) )

      end if

      return
      end
      subroutine r8int_to_i4int ( rmin, rmax, r, imin, imax, i )

c*********************************************************************72
c
cc R8INT_TO_I4INT maps an R8INT to an integer interval.
c
c  Discussion:
c
c    The formula used is
c
c      I := IMIN + ( IMAX - IMIN ) * ( R - RMIN ) / ( RMAX - RMIN )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    03 December 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision RMIN, RMAX, the range.
c
c    Input, double precision R, the number to be converted.
c
c    Input, integer IMAX, IMIN, the integer range.
c
c    Output, integer I, the corresponding value in the
c    range [IMIN,IMAX].
c
      implicit none

      integer i
      integer imax
      integer imin
      double precision r
      double precision rmax
      double precision rmin

      if ( rmax .eq. rmin ) then

        i = ( imax + imin ) / 2

      else

        i = nint ( 
     &    ( ( rmax - r        ) * dble ( imin )   
     &    + (        r - rmin ) * dble ( imax ) ) 
     &    / ( rmax     - rmin ) )

      end if

      return
      end
      subroutine r8mat_add ( m, n, alpha, a, beta, b, c )

c*********************************************************************72
c
cc R8MAT_ADD computes C = alpha * A + beta * B for R8MAT's.
c
c  Discussion:
c
c    An R8MAT is an MxN array of R8's, stored by (I,J) -> [I+J*M].
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    01 December 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns.
c
c    Input, double precision ALPHA, the multiplier for A.
c
c    Input, double precision A(M,N), the first matrix.
c
c    Input, double precision BETA, the multiplier for A.
c
c    Input, double precision B(M,N), the second matrix.
c
c    Output, double precision C(M,N), the sum of alpha*A+beta*B.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      double precision alpha
      double precision b(m,n)
      double precision beta
      double precision c(m,n)
      integer i
      integer j

      do j = 1, n
        do i = 1, m
          c(i,j) = alpha * a(i,j) + beta * b(i,j)
        end do
      end do

      return
      end
      subroutine r8mat_amax ( m, n, a, amax )

c*********************************************************************72
c
cc R8MAT_AMAX computes the largest absolute value in an R8MAT.
c
c  Discussion:
c
c    An R8MAT is an MxN array of R8's, stored by (I,J) -> [I+J*M].
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    07 September 2012
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns.
c
c    Input, double precision A(M,N), the matrix.
c
c    Output, double precision AMAX, the largest absolute value in A.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      double precision amax
      integer i
      integer j

      amax = abs ( a(1,1) )

      do j = 1, n
        do i = 1, m
          amax = max ( amax, abs ( a(i,j) ) )
        end do
      end do

      return
      end
      subroutine r8mat_border_add ( m, n, table, table2 )

c*********************************************************************72
c
cc R8MAT_BORDER_ADD adds a "border" to an R8MAT.
c
c  Discussion:
c
c    An R8MAT is an MxN array of R8's, stored by (I,J) -> [I+J*M].
c
c    We suppose the input data gives values of a quantity on nodes
c    in the interior of a 2D grid, and we wish to create a new table
c    with additional positions for the nodes that would be on the
c    border of the 2D grid.
c
c                  0 0 0 0 0 0
c      * * * *     0 * * * * 0
c      * * * * --> 0 * * * * 0
c      * * * *     0 * * * * 0
c                  0 0 0 0 0 0
c
c    The illustration suggests the situation in which a 3 by 4 array
c    is input, and a 5 by 6 array is to be output.
c
c    The old data is shifted to its correct positions in the new array.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    28 April 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the spatial dimension.
c
c    Input, integer N, the number of points.
c
c    Input, double precision TABLE(M,N), the table data.
c
c    Output, double precision TABLE2(M+2,N+2), the augmented table data.
c
      implicit none

      integer m
      integer n

      integer i
      integer j
      double precision table(m,n)
      double precision table2(m+2,n+2)

      do j = 1, n + 2
        table2(1,j) = 0.0D+00
      end do

      do j = 1, n + 2
        table2(m+2,j) = 0.0D+00
      end do

      do i = 2, m + 1
        table2(i,1) = 0.0D+00
      end do

      do i = 2, m + 1
        table2(i,n+2) = 0.0D+00
      end do

      do j = 2, n + 1
        do i = 2, m + 1
          table2(i,j) = table(i-1,j-1)
        end do
      end do

      return
      end
      subroutine r8mat_border_cut ( m, n, table, table2 )

c*********************************************************************72
c
cc R8MAT_BORDER_CUT cuts the "border" of an R8MAT.
c
c  Discussion:
c
c    An R8MAT is an MxN array of R8's, stored by (I,J) -> [I+J*M].
c
c    We suppose the input data gives values of a quantity on nodes
c    on a 2D grid, and we wish to create a new table corresponding only
c    to those nodes in the interior of the 2D grid.
c
c      0 0 0 0 0 0
c      0 * * * * 0    * * * *
c      0 * * * * 0 -> * * * *
c      0 * * * * 0    * * * *
c      0 0 0 0 0 0
c
c    The illustration suggests the situation in which a 5 by 6 array
c    is input, and a 3 by 4 array is to be output.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    28 April 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the spatial dimension.
c
c    Input, integer N, the number of points.
c
c    Input, double precision TABLE(M,N), the table data.
c
c    Output, double precision TABLE2(M-2,N-2), the new table data.
c
      implicit none

      integer m
      integer n

      integer i
      integer j
      double precision table(m,n)
      double precision table2(m-2,n-2)

      if ( m .le. 2 .or. n .le. 2 ) then
        return
      end if

      do j = 1, n - 2
        do i = 1, m - 2
          table2(i,j) = table(i+1,j+1)
        end do
      end do

      return
      end
      subroutine r8mat_cholesky_factor ( n, a, c, flag )

c*********************************************************************72
c
cc R8MAT_CHOLESKY_FACTOR computes the Cholesky factor of a symmetric matrix.
c
c  Discussion:
c
c    An R8MAT is an MxN array of R8's, stored by (I,J) -> [I+J*M].
c
c    The matrix must be symmetric and positive semidefinite.
c
c    For a positive semidefinite symmetric matrix A, the Cholesky factorization
c    is a lower triangular matrix L such that:
c
c      A = L * L'
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    08 April 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of rows and columns of
c    the matrix A.
c
c    Input, double precision A(N,N), the N by N matrix.
c
c    Output, double precision C(N,N), the N by N lower triangular
c    Cholesky factor.
c
c    Output, integer FLAG:
c    0, no error occurred.
c    1, the matrix is not positive definite.
c
      implicit none

      integer n

      double precision a(n,n)
      double precision c(n,n)
      integer flag
      integer i
      integer j
      integer k
      double precision sum2

      flag = 0

      do j = 1, n
        do i = 1, n
          c(i,j) = a(i,j)
        end do
      end do

      do j = 1, n

        do i = 1, j - 1
          c(i,j) = 0.0D+00
        end do

        do i = j, n

          sum2 = 0.0D+00
          do k = 1, j - 1
            sum2 = sum2 + c(j,k) * c(i,k)
          end do
          sum2 = c(j,i) - sum2

          if ( i .eq. j ) then
            if ( sum2 .le. 0.0D+00 ) then
              flag = 1
              return
            else
              c(i,j) = sqrt ( sum2 )
            end if
          else
            if ( c(j,j) .ne. 0.0D+00 ) then
              c(i,j) = sum2 / c(j,j)
            else
              c(i,j) = 0.0D+00
            end if
          end if

        end do

      end do

      return
      end
      subroutine r8mat_cholesky_factor_upper ( n, a, c, flag )

!*********************************************************************72
!
!! R8MAT_CHOLESKY_FACTOR_UPPER: upper Cholesky factor of a symmetric matrix.
!
!  Discussion:
!
!    The matrix must be symmetric and positive semidefinite.
!
!    For a positive semidefinite symmetric matrix A, the Cholesky factorization
!    is an upper triangular matrix R such that:
!
!      A = R * R'
!
!    The lower Cholesky factor is a lower triangular matrix L such that
!
!      A = L * L'
!
!    An R8MAT is an MxN array of R8's, stored by (I,J) -> [I+J*M].
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    03 August 2013
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input, integer N, the number of rows and columns of
!    the matrix A.
!
!    Input, double precision A(N,N), the N by N matrix.
!
!    Output, double precision C(N,N), the N by N upper triangular
!    Cholesky factor.
!
!    Output, integer FLAG:
!    0, no error occurred.
!    1, the matrix is not positive definite.
!    2, the matrix is not nonnegative definite.
!
      implicit none

      integer n

      double precision a(n,n)
      double precision c(n,n)
      integer flag
      integer i
      integer j
      integer k
      double precision sum2
      double precision tol

      flag = 0

      do j = 1, n
        do i = 1, n
          c(i,j) = a(i,j)
        end do
      end do

      do j = 1, n

        do i = 1, j - 1
          c(j,i) = 0.0D+00
       end do

        do i = j, n

          sum2 = c(i,j)
          do k = 1, j - 1
            sum2 = sum2 - c(k,j) * c(k,i)
          end do

          if ( i .eq. j ) then
            if ( sum2 .le. 0.0D+00 ) then
              flag = 1
              return
            else
              c(j,i) = sqrt ( sum2 )
            end if
          else
            if ( c(j,j) .ne. 0.0D+00 ) then
              c(j,i) = sum2 / c(j,j)
            else
              c(j,i) = 0.0D+00
            end if
          end if

        end do

      end do

      return
      end
      subroutine r8mat_cholesky_inverse ( n, a )

c*********************************************************************72
c
cc R8MAT_CHOLESKY_INVERSE computes the inverse of a symmetric matrix.
c
c  Discussion:
c
c    The matrix must be symmetric and positive semidefinite.
c
c    The upper triangular Cholesky factorization R is computed, so that:
c
c      A = R' * R
c
c    Then the inverse B is computed by
c
c      B = inv ( A ) = inv ( R ) * inv ( R' )
c
c    An R8MAT is an MxN array of R8's, stored by (I,J) -> [I+J*M].
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    22 October 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of rows and columns of
c    the matrix A.
c
c    Input/output, double precision A(N,N).  On input, the matrix.
c    On output, the inverse of the matrix.
c
      implicit none

      integer n

      double precision a(n,n)
      integer i
      integer j
      integer k
      double precision s
      double precision t

      do j = 1, n

        s = 0.0D+00

        do k = 1, j - 1
          t = a(k,j)
          do i = 1, k - 1
            t = t - a(i,k) * a(i,j)
          end do
          t = t / a(k,k)
          a(k,j) = t
          s = s + t * t
        end do

        s = a(j,j) - s

        if ( s .le. 0.0D+00 ) then
          write ( *, '(a)' ) ' '
          write ( *, '(a)' ) 'R8MAT_CHOLESKY_INVERSE - Fatal error!'
          write ( *, '(a)' ) '  The matrix is singular.'
          stop 1
        end if

        a(j,j) = sqrt ( s )

        do i = j + 1, n
          a(i,j) = 0.0D+00
        end do

      end do
c
c  Compute inverse(R).
c
      do k = 1, n

        a(k,k) = 1.0D+00 / a(k,k)
        do i = 1, k - 1
          a(i,k) = - a(i,k) * a(k,k)
        end do

        do j = k + 1, n
          t = a(k,j)
          a(k,j) = 0.0D+00
          do i = 1, k
            a(i,j) = a(i,j) + t * a(i,k)
          end do
        end do

      end do
c
c  Form inverse(R) * (inverse(R))'.
c
      do j = 1, n
        do k = 1, j - 1
          t = a(k,j)
          do i = 1, k
            a(i,k) = a(i,k) + t * a(i,j)
          end do
        end do
        t = a(j,j)
        do i = 1, j
          a(i,j) = a(i,j) * t
        end do
      end do
c
c  Use reflection.
c
      do i = 1, n
        do j = 1, i - 1
          a(i,j) = a(j,i)
        end do
      end do

      return
      end
      subroutine r8mat_cholesky_solve ( n, l, b, x )

c*********************************************************************72
c
cc R8MAT_CHOLESKY_SOLVE solves a Cholesky factored linear system A * x = b.
c
c  Discussion:
c
c    An R8MAT is an MxN array of R8's, stored by (I,J) -> [I+J*M].
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    09 April 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of rows and columns of
c    the matrix A.
c
c    Input, double precision L(N,N), the N by N Cholesky factor of the
c    system matrix .
c
c    Input, double precision B(N), the right hand side of the linear system.
c
c    Output, double precision X(N), the solution of the linear system.
c
      implicit none

      integer n

      double precision b(n)
      double precision l(n,n)
      double precision x(n)
c
c  Solve L * y = b.
c
      call r8mat_l_solve ( n, l, b, x )
c
c  Solve L' * x = y.
c
      call r8mat_lt_solve ( n, l, x, x )

      return
      end
      subroutine r8mat_cholesky_solve_upper ( n, r, b, x )

c*********************************************************************72
c
cc R8MAT_CHOLESKY_SOLVE_UPPER solves a Cholesky factored system A * x = b.
c
c  Discussion:
c
c    An R8MAT is an MxN array of R8's, stored by (I,J) -> [I+J*M].
c
c    This routine works with the upper triangular Cholesky factor A = R' * R.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    21 October 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of rows and columns of
c    the matrix A.
c
c    Input, double precision R(N,N), the N by N upper Cholesky factor of the
c    system matrix.
c
c    Input, double precision B(N), the right hand side of the linear system.
c
c    Output, double precision X(N), the solution of the linear system.
c
      implicit none

      integer n

      double precision b(n)
      double precision r(n,n)
      double precision x(n)
c
c  Solve R' * y = b.
c
      call r8mat_ut_solve ( n, r, b, x )
c
c  Solve R * x = y.
c
      call r8mat_u_solve ( n, r, x, x )

      return
      end
      subroutine r8mat_copy ( m, n, a1, a2 )

c*********************************************************************72
c
cc R8MAT_COPY copies an R8MAT.
c
c  Discussion:
c
c    An R8MAT is an MxN array of R8's, stored by (I,J) -> [I+J*M].
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    26 July 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the order of the matrix.
c
c    Input, double precision A1(M,N), the matrix to be copied.
c
c    Output, double precision A2(M,N), a copy of the matrix.
c
      implicit none

      integer m
      integer n

      double precision a1(m,n)
      double precision a2(m,n)
      integer i
      integer j

      do j = 1, n
        do i = 1, m
          a2(i,j) = a1(i,j)
        end do
      end do

      return
      end
      subroutine r8mat_covariance ( m, n, x, c )

c*********************************************************************72
c
cc R8MAT_COVARIANCE computes the sample covariance of a set of vector data.
c
c  Discussion:
c
c    An R8MAT is an MxN array of R8's, stored by (I,J) -> [I+J*M].
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    26 June 2013
c
c  Author:
c
c    John Burkardt.
c
c  Parameters:
c
c    Input, integer M, the size of a single data vectors.
c
c    Input, integer N, the number of data vectors.
c    N should be greater than 1.
c
c    Input, double precision X(M,N), an array of N data vectors, each
c    of length M.
c
c    Output, double precision C(M,M), the covariance matrix for the data.
c
      implicit none

      integer m
      integer n

      double precision c(m,m)
      integer i
      integer j
      integer k
      double precision x(m,n)
      double precision x_mean(m)

      do j = 1, m
        do i = 1, m
          c(i,j) = 0.0D+00
        end do
      end do
c
c  Special case of N = 1.
c
      if ( n .eq. 1 ) then
        do i = 1, m
          c(i,i) = 1.0D+00
        end do
        return
      end if
c
c  Determine the sample means.
c
      do i = 1, m
        x_mean(i) = 0.0D+00
        do j = 1, n
          x_mean(i) = x_mean(i) + x(i,j)
        end do
        x_mean(i) = x_mean(i) / dble ( n )
      end do
c
c  Determine the sample covariance.
c
      do j = 1, m
        do i = 1, m
          do k = 1, n
            c(i,j) = c(i,j) 
     &        + ( x(i,k) - x_mean(i) ) * ( x(j,k) - x_mean(j) )
          end do
        end do
      end do

      do j = 1, m
        do i = 1, m
          c(i,j) = c(i,j) / dble ( n - 1 )
        end do
      end do

      return
      end
      subroutine r8mat_det ( n, a, det )

c*********************************************************************72
c
cc R8MAT_DET computes the determinant of an R8MAT.
c
c  Discussion:
c
c    An R8MAT is an MxN array of R8's, stored by (I,J) -> [I+J*M].
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    19 September 2010
c
c  Author:
c
c    Original FORTRAN77 version by Helmut Spaeth.
c    This FORTRAN77 version by John Burkardt.
c
c  Reference:
c
c    Helmut Spaeth,
c    Cluster Analysis Algorithms
c    for Data Reduction and Classification of Objects,
c    Ellis Horwood, 1980, page 125-127.
c
c  Parameters:
c
c    Input, integer N, the order of the matrix.
c
c    Input, double precision A(N,N), the matrix whose determinant is desired.
c
c    Output, double precision DET, the determinant of the matrix.
c
      implicit none

      integer n

      double precision a(n,n)
      double precision b(n,n)
      double precision det
      integer i
      integer j
      integer k
      integer m
      integer piv
      double precision t

      do j = 1, n
        do i = 1, n
          b(i,j) = a(i,j)
        end do
      end do

      det = 1.0D+00

      do k = 1, n

        piv = k
        do i = k + 1, n
          if ( abs ( b(piv,k) ) .lt. abs ( b(i,k) ) ) then
            piv = i
          end if
        end do

        m = piv

        if ( m .ne. k ) then
          det = - det
          t      = b(m,k)
          b(m,k) = b(k,k)
          b(k,k) = t
        end if

        det = det * b(k,k)

        if ( b(k,k) .ne. 0.0D+00 ) then

          do i = k + 1, n
            b(i,k) = - b(i,k) / b(k,k)
          end do

          do j = k + 1, n

            if ( m .ne. k ) then
              t      = b(m,j)
              b(m,j) = b(k,j)
              b(k,j) = t
            end if

            do i = k + 1, n
              b(i,j) = b(i,j) + b(i,k) * b(k,j)
            end do

          end do

        end if

      end do

      return
      end
      function r8mat_det_2d ( a )

c*********************************************************************72
c
cc R8MAT_DET_2D computes the determinant of a 2 by 2 matrix.
c
c  Discussion:
c
c    An R8MAT is an array of R8's.
c
c    The determinant of a 2 by 2 matrix is
c
c      a11 * a22 - a12 * a21.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    22 July 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision A(2,2), the matrix whose determinant is desired.
c
c    Output, double precision RMAT_DET_2D, the determinant of the matrix.
c
      implicit none

      double precision a(2,2)
      double precision r8mat_det_2d

      r8mat_det_2d = a(1,1) * a(2,2) - a(1,2) * a(2,1)

      return
      end
      function r8mat_det_3d ( a )

c*********************************************************************72
c
cc R8MAT_DET_3D computes the determinant of a 3 by 3 matrix.
c
c  Discussion:
c
c    An R8MAT is an array of R8's.
c
c    The determinant of a 3 by 3 matrix is
c
c        a11 * a22 * a33 - a11 * a23 * a32
c      + a12 * a23 * a31 - a12 * a21 * a33
c      + a13 * a21 * a32 - a13 * a22 * a31
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    22 July 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision A(3,3), the matrix whose determinant is desired.
c
c    Output, double precision RMAT_DET_3D, the determinant of the matrix.
c
      implicit none

      double precision a(3,3)
      double precision r8mat_det_3d

      r8mat_det_3d =
     &       a(1,1) * ( a(2,2) * a(3,3) - a(2,3) * a(3,2) )
     &     + a(1,2) * ( a(2,3) * a(3,1) - a(2,1) * a(3,3) )
     &     + a(1,3) * ( a(2,1) * a(3,2) - a(2,2) * a(3,1) )

      return
      end
      function r8mat_det_4d ( a )

c*********************************************************************72
c
cc R8MAT_DET_4D computes the determinant of a 4 by 4 R8MAT.
c
c  Discussion:
c
c    An R8MAT is an array of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    31 August 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision A(4,4), the matrix whose determinant is desired.
c
c    Output, double precision R8MAT_DET_4D, the determinant of the matrix.
c
      implicit none

      double precision a(4,4)
      double precision r8mat_det_4d

      r8mat_det_4d =
     &       a(1,1) * (
     &           a(2,2) * ( a(3,3) * a(4,4) - a(3,4) * a(4,3) )
     &         - a(2,3) * ( a(3,2) * a(4,4) - a(3,4) * a(4,2) )
     &         + a(2,4) * ( a(3,2) * a(4,3) - a(3,3) * a(4,2) ) )
     &     - a(1,2) * (
     &           a(2,1) * ( a(3,3) * a(4,4) - a(3,4) * a(4,3) )
     &         - a(2,3) * ( a(3,1) * a(4,4) - a(3,4) * a(4,1) )
     &         + a(2,4) * ( a(3,1) * a(4,3) - a(3,3) * a(4,1) ) )
     &     + a(1,3) * (
     &           a(2,1) * ( a(3,2) * a(4,4) - a(3,4) * a(4,2) )
     &         - a(2,2) * ( a(3,1) * a(4,4) - a(3,4) * a(4,1) )
     &         + a(2,4) * ( a(3,1) * a(4,2) - a(3,2) * a(4,1) ) )
     &     - a(1,4) * (
     &           a(2,1) * ( a(3,2) * a(4,3) - a(3,3) * a(4,2) )
     &         - a(2,2) * ( a(3,1) * a(4,3) - a(3,3) * a(4,1) )
     &         + a(2,3) * ( a(3,1) * a(4,2) - a(3,2) * a(4,1) ) )

      return
      end
      function r8mat_det_5d ( a )

c*********************************************************************72
c
cc R8MAT_DET_5D computes the determinant of a 5 by 5 R8MAT.
c
c  Discussion:
c
c    An R8MAT is an array of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    02 March 1999
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision A(5,5), the matrix whose determinant is desired.
c
c    Output, double precision R8MAT_DET_5D, the determinant of the matrix.
c
      implicit none

      double precision a(5,5)
      double precision b(4,4)
      integer i
      integer inc
      integer j
      integer k
      double precision r8mat_det_4d
      double precision r8mat_det_5d
c
c  Expand the determinant into the sum of the determinants of the
c  five 4 by 4 matrices created by dropping row 1, and column k.
c
      r8mat_det_5d = 0.0D+00

      do k = 1, 5

        do i = 1, 4
          do j = 1, 4

            if ( j .lt. k ) then
              inc = 0
            else
              inc = 1
            end if

            b(i,j) = a(i+1,j+inc)

          end do
        end do

        r8mat_det_5d = r8mat_det_5d + (-1) ** ( k + 1 )
     &    * a(1,k) * r8mat_det_4d ( b )

      end do

      return
      end
      subroutine r8mat_diag_add_scalar ( n, a, s )

c*********************************************************************72
c
cc R8MAT_DIAG_ADD_SCALAR adds a scalar to the diagonal of an R8MAT.
c
c  Discussion:
c
c    An R8MAT is an array of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    31 August 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of rows and columns of the matrix.
c
c    Input/output, double precision A(N,N), the N by N matrix to be modified.
c
c    Input, double precision S, the value to be added to the diagonal
c    of the matrix.
c
      implicit none

      integer n

      double precision a(n,n)
      integer i
      double precision s

      do i = 1, n
        a(i,i) = a(i,i) + s
      end do

      return
      end
      subroutine r8mat_diag_add_vector ( n, a, v )

c*********************************************************************72
c
cc R8MAT_DIAG_ADD_VECTOR adds a vector to the diagonal of an R8MAT.
c
c  Discussion:
c
c    An R8MAT is an array of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    02 July 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of rows and columns of
c    the matrix.
c
c    Input/output, double precision A(N,N), the N by N matrix.
c
c    Input, real double precision V(N), the vector to be added
c    to the diagonal of A.
c
      implicit none

      integer n

      double precision a(n,n)
      integer i
      double precision v(n)

      do i = 1, n
        a(i,i) = a(i,i) + v(i)
      end do

      return
      end
      subroutine r8mat_diag_get_vector ( n, a, v )

c*********************************************************************72
c
cc R8MAT_DIAG_GET_VECTOR gets the value of the diagonal of an R8MAT.
c
c  Discussion:
c
c    An R8MAT is an array of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    02 July 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of rows and columns of
c    the matrix.
c
c    Input, double precision A(N,N), the N by N matrix.
c
c    Output, double precision V(N), the diagonal entries
c    of the matrix.
c
      implicit none

      integer n

      double precision a(n,n)
      integer i
      double precision v(n)

      do i = 1, n
        v(i) = a(i,i)
      end do

      return
      end
      subroutine r8mat_diag_set_scalar ( n, a, s )

c*********************************************************************72
c
cc R8MAT_DIAG_SET_SCALAR sets the diagonal of a matrix to a scalar value.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    17 March 2000
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of rows and columns of the matrix.
c
c    Input/output, double precision A(N,N), the N by N matrix to be modified.
c
c    Input, double precision S, the value to be assigned to the
c    diagonal of the matrix.
c
      implicit none

      integer n

      double precision a(n,n)
      integer i
      double precision s

      do i = 1, n
        a(i,i) = s
      end do

      return
      end
      subroutine r8mat_diag_set_vector ( n, a, v )

c*********************************************************************72
c
cc R8MAT_DIAG_SET_VECTOR sets the diagonal of an R8MAT to a vector.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    08 May 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of rows and columns.
c
c    Input/output, double precision A(N,N), the N by N matrix.
c
c    Input, double precision V(N), the vector to be assigned to the
c    diagonal of A.
c
      implicit none

      integer n

      double precision a(n,n)
      integer i
      double precision v(n)

      do i = 1, n
        a(i,i) = v(i)
      end do

      return
      end
      subroutine r8mat_diagonal ( n, diag, a )

c*********************************************************************72
c
cc R8MAT_DIAGONAL returns a diagonal matrix as an R8MAT.
c
c  Discussion:
c
c    An R8MAT is an array of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    31 July 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the order of A.
c
c    Input, double precision DIAG(N), the diagonal entries.
c
c    Output, double precision A(N,N), the N by N identity matrix.
c
      implicit none

      integer n

      double precision a(n,n)
      double precision diag(n)
      integer i
      integer j

      do j = 1, n
        do i = 1, n
          a(i,j) = 0.0D+00
        end do
      end do

      do i = 1, n
        a(i,i) = diag(i)
      end do

      return
      end
      function r8mat_diff_frobenius ( m, n, a, b )

c*********************************************************************72
c
cc R8MAT_DIFF_FROBENIUS: Frobenius norm of the difference of two R8MAT's.
c
c  Discussion:
c
c    An R8MAT is a matrix of double precision values.
c
c    The Frobenius norm is defined as
c
c      R8MAT_DIFF_FROBENIUS = sqrt (
c        sum ( 1 <= I <= M ) sum ( 1 <= j <= N ) ( A(I,J) - B(I,J) )^2 )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    19 June 2012
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the number of rows in A and B.
c
c    Input, integer N, the number of columns in A and B.
c
c    Input, double precision A(M,N), B(M,N), the matrices
c    for which we want the Frobenius norm of the difference.
c
c    Output, double precision R8MAT_DIFF_FROBENIUS, the Frobenius norm of
c    the difference of A and B.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      double precision b(m,n)
      integer i
      integer j
      double precision r8mat_diff_frobenius
      double precision value

      value = 0.0D+00
      do j = 1, n
        do i = 1, m
          value = value + ( a(i,j) - b(i,j) ) ** 2
        end do
      end do

      value = sqrt ( value )

      r8mat_diff_frobenius = value

      return
      end
      subroutine r8mat_expand_linear ( m, n, x, mfat, nfat, xfat )

c*********************************************************************72
c
cc R8MAT_EXPAND_LINEAR linearly interpolates new data into an R8MAT.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c    In this routine, the expansion is specified by giving the number
c    of intermediate values to generate between each pair of original
c    data rows and columns.
c
c    The interpolation is not actually linear.  It uses the functions
c
c      1, x, y, and xy.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    11 October 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns of
c    input data.
c
c    Input, double precision X(M,N), the original data.
c
c    Input, integer MFAT, NFAT, the number of data values
c    to interpolate between each row, and each column, of original data values.
c
c    Output, double precision XFAT(M2,N2), the fattened data, where
c    M2 = (M-1)*(MFAT+1)+1,
c    N2 = (N-1)*(NFAT+1)+1.
c
      implicit none

      integer m
      integer mfat
      integer n
      integer nfat

      integer i
      integer ihi
      integer ii
      integer iii
      integer ip1
      integer j
      integer jhi
      integer jj
      integer jjj
      integer jp1
      double precision s
      double precision t
      double precision x(m,n)
      double precision x00
      double precision x01
      double precision x10
      double precision x11
      double precision xfat((m-1)*(mfat+1)+1,(n-1)*(nfat+1)+1)

      do i = 1, m

        if ( i .lt. m ) then
          ihi = mfat
        else
          ihi = 0
        end if

        do j = 1, n

          if ( j .lt. n ) then
            jhi = nfat
          else
            jhi = 0
          end if

          if ( i .lt. m ) then
            ip1 = i + 1
          else
            ip1 = i
          end if

          if ( j .lt. n ) then
            jp1 = j + 1
          else
            jp1 = j
          end if

          x00 = x(i,j)
          x10 = x(ip1,j)
          x01 = x(i,jp1)
          x11 = x(ip1,jp1)

          do ii = 0, ihi

            s = dble ( ii ) / dble ( ihi + 1 )

            do jj = 0, jhi

              t = dble ( jj ) / dble ( jhi + 1 )

              iii = 1 + ( i - 1 ) * ( mfat + 1 ) + ii
              jjj = 1 + ( j - 1 ) * ( nfat + 1 ) + jj

              xfat(iii,jjj) =
     &                                          x00
     &            + s     * (       x10       - x00 )
     &            + t     * (             x01 - x00 )
     &            + s * t * ( x11 - x10 - x01 + x00 )

            end do

          end do

        end do

      end do

      return
      end
      subroutine r8mat_expand_linear2 ( m, n, a, m2, n2, a2 )

c*********************************************************************72
c
cc R8MAT_EXPAND_LINEAR2 expands an R8MAT by linear interpolation.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c    In this version of the routine, the expansion is indicated
c    by specifying the dimensions of the expanded array.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    02 December 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns in A.
c
c    Input, double precision A(M,N), a "small" M by N array.
c
c    Input, integer M2, N2, the number of rows and columns in A2.
c
c    Output, double precision A2(M2,N2), the expanded array, which
c    contains an interpolated version of the data in A.
c
      implicit none

      integer m
      integer m2
      integer n
      integer n2

      double precision a(m,n)
      double precision a2(m2,n2)
      integer i
      integer i1
      integer i2
      integer j
      integer j1
      integer j2
      double precision r
      double precision r1
      double precision r2
      double precision s
      double precision s1
      double precision s2

      do i = 1, m2

        if ( m2 .eq. 1 ) then
          r = 0.5D+00
        else
          r = dble ( i - 1 ) / dble ( m2 - 1 )
        end if

        i1 = 1 + int ( r * dble ( m - 1 ) )
        i2 = i1 + 1

        if ( m .lt. i2 ) then
          i1 = m - 1
          i2 = m
        end if

        r1 = dble ( i1 - 1 ) / dble ( m - 1 )

        r2 = dble ( i2 - 1 ) / dble ( m - 1 )

        do j = 1, n2

          if ( n2 .eq. 1 ) then
            s = 0.5D+00
          else
            s = dble ( j - 1 ) / dble ( n2 - 1 )
          end if

          j1 = 1 + int ( s * dble ( n - 1 ) )
          j2 = j1 + 1

          if ( n .lt. j2 ) then
            j1 = n - 1
            j2 = n
          end if

          s1 = dble ( j1 - 1 ) / dble ( n - 1 )

          s2 = dble ( j2 - 1 ) / dble ( n - 1 )

          a2(i,j) = 
     &      ( ( r2 - r ) * ( s2 - s ) * a(i1,j1) 
     &      + ( r - r1 ) * ( s2 - s ) * a(i2,j1) 
     &      + ( r2 - r ) * ( s - s1 ) * a(i1,j2) 
     &      + ( r - r1 ) * ( s - s1 ) * a(i2,j2) ) 
     &      / ( ( r2 - r1 ) * ( s2 - s1 ) )

        end do

      end do

      return
      end
      subroutine r8mat_flip_cols ( m, n, a, b )

c*********************************************************************72
c
cc R8MAT_FLIP_COLS reverses the column order of an R8MAT.
c
c  Discussion:
c
c    An R8MAT is an MxN array of R8's, stored by (I,J) -> [I+J*M].
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    01 November 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the order of the matrix.
c
c    Input, double precision A(M,N), the matrix to be flipped.
c
c    Output, double precision B(M,N), a copy of A with the
c    columns in reverse order.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      double precision b(m,n)
      integer i
      integer j

      do j = 1, n
        do i = 1, m
          b(i,n+1-j) = a(i,j)
        end do
      end do

      return
      end
      subroutine r8mat_flip_rows ( m, n, a, b )

c*********************************************************************72
c
cc R8MAT_FLIP_ROWS reverses the row order of an R8MAT.
c
c  Discussion:
c
c    An R8MAT is an MxN array of R8's, stored by (I,J) -> [I+J*M].
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    01 November 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the order of the matrix.
c
c    Input, double precision A(M,N), the matrix to be flipped.
c
c    Output, double precision B(M,N), a copy of A with the
c    rows in reverse order.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      double precision b(m,n)
      integer i
      integer j

      do j = 1, n
        do i = 1, m
          b(m+1-i,j) = a(i,j)
        end do
      end do

      return
      end
      subroutine r8mat_fs ( n, a, b, info )

c*********************************************************************72
c
cc R8MAT_FS factors and solves a system with one right hand side.
c
c  Discussion:
c
c    An R8MAT is an MxN array of R8's, stored by (I,J) -> [I+J*M].
c
c    This routine differs from R8MAT_FSS in two ways:
c    * only one right hand side is allowed;
c    * the input matrix A is not modified.
c
c    This routine uses partial pivoting, but no pivot vector is required.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    20 January 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the order of the matrix.
c    N must be positive.
c
c    Input, double precision A(N,N), the coefficient matrix.
c
c    Input/output, double precision B(N).
c    On input, B is the right hand side of the linear system.
c    On output, B is the solution of the linear system.
c
c    Output, integer INFO, singularity flag.
c    0, no singularity detected.
c    nonzero, the factorization failed on the INFO-th step.
c
      implicit none

      integer n
      integer nb

      double precision a(n,n)
      double precision a2(n,n)
      double precision b(n)
      integer i
      integer info
      integer ipiv
      integer j
      integer jcol
      integer k
      double precision piv
      double precision temp
c
c  Copy the matrix.
c
      do j = 1, n
        do i = 1, n
          a2(i,j) = a(i,j)
        end do
      end do

      info = 0

      do jcol = 1, n
c
c  Find the maximum element in column I.
c
        piv = abs ( a2(jcol,jcol) )
        ipiv = jcol
        do i = jcol + 1, n
          if ( piv .lt. abs ( a2(i,jcol) ) ) then
            piv = abs ( a2(i,jcol) )
            ipiv = i
          end if
        end do

        if ( piv .eq. 0.0D+00 ) then
          info = jcol
          write ( *, '(a)' ) ' '
          write ( *, '(a)' ) 'R8MAT_FS - Fatal error!'
          write ( *, '(a,i8)' ) '  Zero pivot on step ', info
          return
        end if
c
c  Switch rows JCOL and IPIV, and B.
c
        if ( jcol .ne. ipiv ) then

          do j = 1, n
            temp = a2(jcol,j)
            a2(jcol,j) = a2(ipiv,j)
            a2(ipiv,j) = temp
          end do

          temp = b(jcol)
          b(jcol) = b(ipiv)
          b(ipiv) = temp

        end if
c
c  Scale the pivot row.
c
        do j = jcol + 1, n
          a2(jcol,j) = a2(jcol,j) / a2(jcol,jcol)
        end do
        b(jcol) = b(jcol) / a2(jcol,jcol)
        a2(jcol,jcol) = 1.0D+00
c
c  Use the pivot row to eliminate lower entries in that column.
c
        do i = jcol + 1, n
          if ( a2(i,jcol) .ne. 0.0D+00 ) then
            temp = - a2(i,jcol)
            a2(i,jcol) = 0.0D+00
            do j = jcol + 1, n
              a2(i,j) = a2(i,j) + temp * a2(jcol,j)
            end do
            b(i) = b(i) + temp * b(jcol)
          end if
        end do

      end do
c
c  Back solve.
c
      do k = n, 2, -1
        do i = 1, k - 1
          b(i) = b(i) - a2(i,k) * b(k)
        end do
      end do

      return
      end
      subroutine r8mat_fss ( n, a, nb, b, info )

c*********************************************************************72
c
cc R8MAT_FSS factors and solves a system with multiple right hand sides.
c
c  Discussion:
c
c    An R8MAT is an MxN array of R8's, stored by (I,J) -> [I+J*M].
c
c    This routine does not save the LU factors of the matrix, and hence cannot
c    be used to efficiently solve multiple linear systems, or even to
c    factor A at one time, and solve a single linear system at a later time.
c
c    This routine uses partial pivoting, but no pivot vector is required.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    28 November 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the order of the matrix.
c    N must be positive.
c
c    Input/output, double precision A(N,N).
c    On input, A is the coefficient matrix of the linear system.
c    On output, A is in unit upper triangular form, and
c    represents the U factor of an LU factorization of the
c    original coefficient matrix.
c
c    Input, integer NB, the number of right hand sides.
c
c    Input/output, double precision B(N,NB).
c    On input, B is the right hand side of the linear system.
c    On output, B is the solution of the linear system.
c
c    Output, integer INFO, singularity flag.
c    0, no singularity detected.
c    nonzero, the factorization failed on the INFO-th step.
c
      implicit none

      integer n
      integer nb

      double precision a(n,n)
      double precision b(n,nb)
      integer i
      integer info
      integer ipiv
      integer j
      integer jcol
      integer k
      double precision piv
      double precision temp

      info = 0

      do jcol = 1, n
c
c  Find the maximum element in column I.
c
        piv = abs ( a(jcol,jcol) )
        ipiv = jcol
        do i = jcol + 1, n
          if ( piv .lt. abs ( a(i,jcol) ) ) then
            piv = abs ( a(i,jcol) )
            ipiv = i
          end if
        end do

        if ( piv .eq. 0.0D+00 ) then
          info = jcol
          write ( *, '(a)' ) ' '
          write ( *, '(a)' ) 'R8MAT_FSS - Fatal error!'
          write ( *, '(a,i8)' ) '  Zero pivot on step ', info
          return
        end if
c
c  Switch rows JCOL and IPIV, and B.
c
        if ( jcol .ne. ipiv ) then

          do j = 1, n
            temp = a(jcol,j)
            a(jcol,j) = a(ipiv,j)
            a(ipiv,j) = temp
          end do

          do j = 1, nb
            temp = b(jcol,j)
            b(jcol,j) = b(ipiv,j)
            b(ipiv,j) = temp
          end do

        end if
c
c  Scale the pivot row.
c
        do j = jcol + 1, n
          a(jcol,j) = a(jcol,j) / a(jcol,jcol)
        end do
        do j = 1, nb
          b(jcol,j) = b(jcol,j) / a(jcol,jcol)
        end do
        a(jcol,jcol) = 1.0D+00
c
c  Use the pivot row to eliminate lower entries in that column.
c
        do i = jcol + 1, n
          if ( a(i,jcol) .ne. 0.0D+00 ) then
            temp = - a(i,jcol)
            a(i,jcol) = 0.0D+00
            do j = jcol + 1, n
              a(i,j) = a(i,j) + temp * a(jcol,j)
            end do
            do j = 1, nb
              b(i,j) = b(i,j) + temp * b(jcol,j)
            end do
          end if
        end do

      end do
c
c  Back solve.
c
      do k = n, 2, -1
        do i = 1, k - 1
          do j = 1, nb
            b(i,j) = b(i,j) - a(i,k) * b(k,j)
          end do
        end do
      end do

      return
      end
      subroutine r8mat_givens_post ( n, a, row, col, g )

c*********************************************************************72
c
cc R8MAT_GIVENS_POST computes the Givens postmultiplier rotation matrix.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c    The Givens post-multiplier matrix G(ROW,COL) has the property that
c    the (ROW,COL)-th entry of A*G is zero.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    09 May 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the order of the matrices A and G.
c
c    Input, double precision A(N,N), the matrix to be operated upon.
c
c    Input, integer ROW, COL, the row and column of the
c    entry of A*G which is to be zeroed out.
c
c    Output, double precision G(N,N), the Givens rotation matrix.
c    G is an orthogonal matrix, that is, the inverse of
c    G is the transpose of G.
c
      implicit none

      integer n

      double precision a(n,n)
      integer col
      double precision g(n,n)
      integer row
      double precision theta

      call r8mat_identity ( n, g )

      theta = atan2 ( a(row,col), a(row,row) )

      g(row,row) =  cos ( theta )
      g(row,col) = -sin ( theta )
      g(col,row) =  sin ( theta )
      g(col,col) =  cos ( theta )

      return
      end
      subroutine r8mat_givens_pre ( n, a, row, col, g )

c*********************************************************************72
c
cc R8MAT_GIVENS_PRE computes the Givens premultiplier rotation matrix.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c    The Givens premultiplier rotation matrix G(ROW,COL) has the
c    property that the (ROW,COL)-th entry of G*A is zero.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    09 May 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the order of the matrices A and G.
c
c    Input, double precision A(N,N), the matrix to be operated upon.
c
c    Input, integer ROW, COL, the row and column of the
c    entry of the G*A which is to be zeroed out.
c
c    Output, double precision G(N,N), the Givens rotation matrix.
c    G is an orthogonal matrix, that is, the inverse of
c    G is the transpose of G.
c
      implicit none

      integer n

      double precision a(n,n)
      integer col
      double precision g(n,n)
      integer row
      double precision theta

      call r8mat_identity ( n, g )

      theta = atan2 ( a(row,col), a(col,col) )

      g(row,row) =  cos ( theta )
      g(row,col) = -sin ( theta )
      g(col,row) =  sin ( theta )
      g(col,col) =  cos ( theta )

      return
      end
      subroutine r8mat_hess ( fx, n, x, h )

c*********************************************************************72
c
cc R8MAT_HESS approximates a Hessian matrix via finite differences.
c
c  Discussion:
c
c    An R8MAT is an MxN array of R8's, stored by (I,J) -> [I+J*M].
c
c    H(I,J) = d2 F / d X(I) d X(J)
c
c    The values returned by this routine will be only approximate.
c    In some cases, they will be so poor that they are useless.
c    However, one of the best applications of this routine is for
c    checking your own Hessian calculations, since as Heraclitus
c    said, you'll never get the same result twice when you differentiate
c    a complicated expression by hand.
c
c    The user function routine, here called "FX", should have the form:
c
c      subroutine fx ( n, x, f )
c      integer n
c      double precision f
c      double precision x(n)
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    12 March 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, external FX, the name of the user function routine.
c
c    Input, integer N, the number of variables.
c
c    Input, double precision X(N), the values of the variables.
c
c    Output, double precision H(N,N), the approximated N by N Hessian matrix.
c
      implicit none

      integer n

      double precision eps
      double precision f00
      double precision fmm
      double precision fmp
      double precision fpm
      double precision fpp
      external fx
      double precision h(n,n)
      integer i
      integer j
      double precision r8_epsilon
      double precision s(n)
      double precision x(n)
      double precision xi
      double precision xj
c
c  Choose the stepsizes.
c
      eps = ( r8_epsilon ( ) ) ** 0.33D+00

      do i = 1, n
        s(i) = eps * max ( abs ( x(i) ), 1.0D+00 )
      end do
c
c  Calculate the diagonal elements.
c
      do i = 1, n

        xi = x(i)

        call fx ( n, x, f00 )

        x(i) = xi + s(i)
        call fx ( n, x, fpp )

        x(i) = xi - s(i)
        call fx ( n, x, fmm )

        h(i,i) = ( ( fpp - f00 ) + ( fmm - f00 ) ) / s(i)**2

        x(i) = xi

      end do
c
c  Calculate the off diagonal elements.
c
      do i = 1, n

        xi = x(i)

        do j = i + 1, n

          xj = x(j)

          x(i) = xi + s(i)
          x(j) = xj + s(j)
          call fx ( n, x, fpp )

          x(i) = xi + s(i)
          x(j) = xj - s(j)
          call fx ( n, x, fpm )

          x(i) = xi - s(i)
          x(j) = xj + s(j)
          call fx ( n, x, fmp )

          x(i) = xi - s(i)
          x(j) = xj - s(j)
          call fx ( n, x, fmm )

          h(j,i) = ( ( fpp - fpm ) + ( fmm - fmp ) ) 
     &      / ( 4.0D+00 * s(i) * s(j) )

          h(i,j) = h(j,i)

          x(j) = xj

        end do

        x(i) = xi

      end do

      return
      end
      subroutine r8mat_house_axh ( n, a, v, ah )

c*********************************************************************72
c
cc R8MAT_HOUSE_AXH computes A*H where H is a compact Householder matrix.
c
c  Discussion:
c
c    An R8MAT is an array of R8's.
c
c    The Householder matrix H(V) is defined by
c
c      H(V) = I - 2 * v * v' / ( v' * v )
c
c    This routine is not particularly efficient.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    05 February 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the order of A.
c
c    Input, double precision A(N,N), the matrix.
c
c    Input, double precision V(N), a vector defining a Householder matrix.
c
c    Output, double precision AH(N,N), the product A*H.
c
      implicit none

      integer n

      double precision a(n,n)
      double precision ah(n,n)
      double precision av(n)
      integer i
      integer j
      double precision v(n)
      double precision v_normsq

      v_normsq = 0.0D+00
      do i = 1, n
        v_normsq = v_normsq + v(i) ** 2
      end do
c
c  Compute A*H' = A*H.
c
      do i = 1, n
        av(i) = 0.0D+00
        do j = 1, n
          av(i) = av(i) + a(i,j) * v(j)
        end do
      end do

      do i = 1, n
        do j = 1, n
          ah(i,j) = a(i,j)
        end do
      end do

      do i = 1, n
        do j = 1, n
          ah(i,j) = ah(i,j) - 2.0D+00 * av(i) * v(j)
        end do
      end do

      do i = 1, n
        do j = 1, n
          ah(i,j) = ah(i,j) / v_normsq
        end do
      end do

      return
      end
      subroutine r8mat_house_form ( n, v, h )

c*********************************************************************72
c
cc R8MAT_HOUSE_FORM constructs a Householder matrix from its compact form.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c    H(v) = I - 2 * v * v' / ( v' * v )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    09 May 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the order of the matrix.
c
c    Input, double precision V(N), the vector defining the Householder matrix.
c
c    Output, double precision H(N,N), the Householder matrix.
c
      implicit none

      integer n

      double precision beta
      double precision h(n,n)
      integer i
      integer j
      double precision v(n)
c
c  Compute the L2 norm of V.
c
      beta = 0.0D+00
      do i = 1, n
        beta = beta + v(i) ** 2
      end do
c
c  Form the matrix H.
c
      call r8mat_identity ( n, h )

      do i = 1, n
        do j = 1, n
          h(i,j) = h(i,j) - 2.0D+00 * v(i) * v(j) / beta
        end do
      end do

      return
      end
      subroutine r8mat_house_hxa ( n, a, v, ha )

c*********************************************************************72
c
cc R8MAT_HOUSE_HXA computes H*A where H is a compact Householder matrix.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c    The Householder matrix H(V) is defined by
c
c      H(V) = I - 2 * v * v' / ( v' * v )
c
c    This routine is not particularly efficient.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    09 May 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the order of A.
c
c    Input, double precision A(N,N), the matrix to be premultiplied.
c
c    Input, double precision V(N), a vector defining a Householder matrix.
c
c    Output, double precision HA(N,N), the product H*A.
c
      implicit none

      integer n

      double precision a(n,n)
      double precision ha(n,n)
      double precision ha_temp(n,n)
      integer i
      integer j
      integer k
      double precision v(n)
      double precision v_normsq

      v_normsq = 0.0D+00
      do i = 1, n
        v_normsq = v_normsq + v(i) ** 2
      end do
c
c  Compute A*H' = A*H
c
      do i = 1, n
        do j = 1, n
          ha_temp(i,j) = a(i,j)
          do k = 1, n
            ha_temp(i,j) = ha_temp(i,j)
     &        - 2.0D+00 * v(i) * v(k) * a(k,j) / v_normsq
          end do
        end do
      end do
c
c  Copy the temporary result into HA.
c  Doing it this way means the user can identify the input arguments A and HA.
c
      do j = 1, n
        do i = 1, n
          ha(i,j) = ha_temp(i,j)
        end do
      end do

      return
      end
      subroutine r8mat_house_post ( n, a, row, col, h )

c*********************************************************************72
c
cc R8MAT_HOUSE_POST computes a Householder post-multiplier matrix.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c    H(ROW,COL) has the property that the ROW-th column of
c    A*H(ROW,COL) is zero from entry COL+1 to the end.
c
c    In the most common case, where a QR factorization is being computed,
c    ROW = COL.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    09 May 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the order of the matrices.
c
c    Input, double precision A(N,N), the matrix whose Householder matrix
c    is to be computed.
c
c    Input, integer ROW, COL, specify the location of the
c    entry of the matrix A which is to be preserved.  The entries in
c    the same row, but higher column, will be zeroed out if
c    A is postmultiplied by H.
c
c    Output, double precision H(N,N), the Householder matrix.
c
      implicit none

      integer n

      double precision a(n,n)
      double precision a_row(n)
      integer col
      double precision h(n,n)
      integer j
      integer row
      double precision v(n)
c
c  Set up the vector V.
c
      do j = 1, col - 1
        a_row(j) = 0.0D+00
      end do
      do j = col, n
        a_row(j) = a(row,j)
      end do

      call r8vec_house_column ( n, a_row, col, v )
c
c  Form the matrix H(V).
c
      call r8mat_house_form ( n, v, h )

      return
      end
      subroutine r8mat_house_pre ( n, a, row, col, h )

c*********************************************************************72
c
cc R8MAT_HOUSE_PRE computes a Householder pre-multiplier matrix.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c    H(ROW,COL) has the property that the COL-th column of
c    H(ROW,COL)*A is zero from entry ROW+1 to the end.
c
c    In the most common case, where a QR factorization is being computed,
c    ROW = COL.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    09 May 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the order of the matrices.
c
c    Input, double precision A(N,N), the matrix whose Householder matrix
c    is to be computed.
c
c    Input, integer ROW, COL, specify the location of the
c    entry of the matrix A which is to be preserved.  The entries in
c    the same column, but higher rows, will be zeroed out if A is
c    premultiplied by H.
c
c    Output, double precision H(N,N), the Householder matrix.
c
      implicit none

      integer n

      double precision a(n,n)
      double precision a_col(n)
      integer col
      double precision h(n,n)
      integer i
      integer row
      double precision v(n)
c
c  Set up the vector V.
c
      do i = 1, row - 1
        a_col(i) = 0.0D+00
      end do
      do i = row, n
        a_col(i) = a(i,col)
      end do

      call r8vec_house_column ( n, a_col, row, v )
c
c  Form the matrix H(V).
c
      call r8mat_house_form ( n, v, h )

      return
      end
      subroutine r8mat_identity ( n, a )

c*********************************************************************72
c
cc R8MAT_IDENTITY stores the identity matrix in an R8MAT.
c
c  Discussion:
c
c    An R8MAT is an array of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    24 March 2000
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the order of A.
c
c    Output, double precision A(N,N), the N by N identity matrix.
c
      implicit none

      integer n

      double precision a(n,n)
      integer i
      integer j

      do j = 1, n
        do i = 1, n
          a(i,j) = 0.0D+00
        end do
      end do

      do i = 1, n
        a(i,i) = 1.0D+00
      end do

      return
      end
      function r8mat_in_01 ( m, n, a )

c*********************************************************************72
c
cc R8MAT_IN_01 is TRUE if the entries of an R8MAT are in the range [0,1].
c
c  Discussion:
c
c    An R8MAT is an array of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    03 July 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the number of rows in A.
c
c    Input, integer N, the number of columns in A.
c
c    Input, double precision A(M,N), the matrix.
c
c    Output, logical R8MAT_IN_01, is TRUE if every entry of A is
c    between 0 and 1.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer j
      logical r8mat_in_01

      do j = 1, n
        do i = 1, m
          if ( a(i,j) .lt. 0.0D+00 .or. 1.0D+00 .lt. a(i,j) ) then
            r8mat_in_01 = .false.
            return
          end if
        end do
      end do

      r8mat_in_01 = .true.

      return
      end
      subroutine r8mat_indicator ( m, n, table )

c*********************************************************************72
c
cc R8MAT_INDICATOR sets up an "indicator" R8MAT.
c
c  Discussion:
c
c    An R8MAT is an array of R8's.
c
c    The value of each entry suggests its location, as in:
c
c      11  12  13  14
c      21  22  23  24
c      31  32  33  34
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    28 May 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the number of rows of the matrix.
c    M must be positive.
c
c    Input, integer N, the number of columns of the matrix.
c    N must be positive.
c
c    Output, double precision TABLE(M,N), the table.
c
      implicit none

      integer m
      integer n

      integer fac
      integer i
      integer i4_log_10
      integer j
      double precision table(m,n)

      fac = 10 ** ( i4_log_10 ( n ) + 1 )

      do i = 1, m
        do j = 1, n
          table(i,j) = dble ( fac * i + j )
        end do
      end do

      return
      end
      function r8mat_insignificant ( m, n, r, s )

c*********************************************************************72
c
cc R8MAT_INSIGNIFICANT determines if an R8MAT is insignificant.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    26 November 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the dimension of the matrices.
c
c    Input, double precision R(M,N), the vector to be compared against.
c
c    Input, double precision S(M,N), the vector to be compared.
c
c    Output, logical R8MAT_INSIGNIFICANT, is TRUE if S is insignificant
c    compared to R.
c
      implicit none

      integer m
      integer n

      integer i
      integer j
      double precision r(m,n)
      double precision r8_epsilon
      logical r8mat_insignificant
      double precision s(m,n)
      double precision t
      double precision tol
      logical value

      value = .true.

      do j = 1, n
        do i = 1, m

          t = r(i,j) + s(i,j)
          tol = r8_epsilon ( ) * abs ( r(i,j) )

          if ( tol .lt. abs ( r(i,j) - t ) ) then 
            value = .false.
            go to 10
          end if

        end do
      end do
  
10    continue

      r8mat_insignificant = value

      return
      end
      subroutine r8mat_inverse_2d ( a, b, det )

c*********************************************************************72
c
cc R8MAT_INVERSE_2D inverts a 2 by 2 R8MAT using Cramer's rule.
c
c  Discussion:
c
c    An R8MAT is an array of R8's.
c
c    If the determinant is zero, then A is singular, and does not have an
c    inverse.  In that case, B is simply set to zero, and a
c    message is printed.
c
c    If the determinant is nonzero, then its value is roughly an estimate
c    of how nonsingular the matrix A is.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    31 August 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision A(2,2), the matrix to be inverted.
c
c    Output, double precision B(2,2), the inverse of the matrix A.
c
c    Output, double precision DET, the determinant of the matrix A.
c
      implicit none

      double precision a(2,2)
      double precision b(2,2)
      double precision det
      double precision r8mat_det_2d
c
c  Compute the determinant of A.
c
      det = r8mat_det_2d ( a )

      if ( det .eq. 0.0D+00 ) then

        b(1,1) = 0.0D+00
        b(1,2) = 0.0D+00
        b(2,1) = 0.0D+00
        b(2,2) = 0.0D+00

      else

        b(1,1) =  a(2,2) / det
        b(1,2) = -a(1,2) / det
        b(2,1) = -a(2,1) / det
        b(2,2) =  a(1,1) / det

      end if

      return
      end
      subroutine r8mat_inverse_3d ( a, b, det )

c*********************************************************************72
c
cc R8MAT_INVERSE_3D inverts a 3 by 3 R8MAT using Cramer's rule.
c
c  Discussion:
c
c    An R8MAT is an array of R8's.
c
c    If the determinant is zero, then A is singular, and does not have an
c    inverse.  In that case, B is simply set to zero, and a
c    message is printed.
c
c    If the determinant is nonzero, then its value is roughly an estimate
c    of how nonsingular the matrix A is.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    03 July 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision A(3,3), the matrix to be inverted.
c
c    Output, double precision B(3,3), the inverse of the matrix A.
c
c    Output, double precision DET, the determinant of the matrix A.
c
      implicit none

      double precision a(3,3)
      double precision b(3,3)
      double precision det
      double precision r8mat_det_3d
c
c  Compute the determinant of A.
c
      det = r8mat_det_3d ( a )
c
c  If the determinant is zero, bail out.
c
      if ( det .eq. 0.0D+00 ) then
        call r8mat_zero ( 3, 3, b )
        return
      end if
c
c  Compute the entries of the inverse matrix using an explicit
c  formula.
c
      b(1,1) =  ( a(2,2) * a(3,3) - a(2,3) * a(3,2) ) / det
      b(1,2) = -( a(1,2) * a(3,3) - a(1,3) * a(3,2) ) / det
      b(1,3) =  ( a(1,2) * a(2,3) - a(1,3) * a(2,2) ) / det

      b(2,1) = -( a(2,1) * a(3,3) - a(2,3) * a(3,1) ) / det
      b(2,2) =  ( a(1,1) * a(3,3) - a(1,3) * a(3,1) ) / det
      b(2,3) = -( a(1,1) * a(2,3) - a(1,3) * a(2,1) ) / det

      b(3,1) =  ( a(2,1) * a(3,2) - a(2,2) * a(3,1) ) / det
      b(3,2) = -( a(1,1) * a(3,2) - a(1,2) * a(3,1) ) / det
      b(3,3) =  ( a(1,1) * a(2,2) - a(1,2) * a(2,1) ) / det

      return
      end
      subroutine r8mat_inverse_4d ( a, b, det )

c*********************************************************************72
c
cc R8MAT_INVERSE_4D inverts a 4 by 4 R8MAT using Cramer's rule.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c    If the determinant is zero, then A is singular, and does not have an
c    inverse.  In that case, B is simply set to zero, and a
c    message is printed.
c
c    If the determinant is nonzero, then its value is roughly an estimate
c    of how nonsingular the matrix A is.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    13 September 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision A(4,4), the matrix to be inverted.
c
c    Output, double precision B(4,4), the inverse of the matrix A.
c
c    Output, double precision DET, the determinant of the matrix A.
c
      implicit none

      double precision a(4,4)
      double precision b(4,4)
      double precision det
      integer i
      integer j
      double precision r8mat_det_4d
c
c  Compute the determinant of A.
c
      det = r8mat_det_4d ( a )
c
c  If the determinant is zero, bail out.
c
      if ( det .eq. 0.0D+00 ) then

        do j = 1, 4
          do i = 1, 4
            b(1:4,1:4) = 0.0D+00
          end do
        end do

        return
      end if
c
c  Compute the entries of the inverse matrix using an explicit formula.
c
      b(1,1) = +( 
     &      + a(2,2) * ( a(3,3) * a(4,4) - a(3,4) * a(4,3) ) 
     &      + a(2,3) * ( a(3,4) * a(4,2) - a(3,2) * a(4,4) ) 
     &      + a(2,4) * ( a(3,2) * a(4,3) - a(3,3) * a(4,2) ) 
     &      ) / det

      b(2,1) = -( 
     &      + a(2,1) * ( a(3,3) * a(4,4) - a(3,4) * a(4,3) ) 
     &      + a(2,3) * ( a(3,4) * a(4,1) - a(3,1) * a(4,4) ) 
     &      + a(2,4) * ( a(3,1) * a(4,3) - a(3,3) * a(4,1) ) 
     &      ) / det

      b(3,1) = +( 
     &      + a(2,1) * ( a(3,2) * a(4,4) - a(3,4) * a(4,2) ) 
     &      + a(2,2) * ( a(3,4) * a(4,1) - a(3,1) * a(4,4) ) 
     &      + a(2,4) * ( a(3,1) * a(4,2) - a(3,2) * a(4,1) ) 
     &      ) / det

      b(4,1) = -( 
     &      + a(2,1) * ( a(3,2) * a(4,3) - a(3,3) * a(4,2) ) 
     &      + a(2,2) * ( a(3,3) * a(4,1) - a(3,1) * a(4,3) ) 
     &      + a(2,3) * ( a(3,1) * a(4,2) - a(3,2) * a(4,1) ) 
     &      ) / det

      b(1,2) = -( 
     &      + a(1,2) * ( a(3,3) * a(4,4) - a(3,4) * a(4,3) ) 
     &      + a(1,3) * ( a(3,4) * a(4,2) - a(3,2) * a(4,4) ) 
     &      + a(1,4) * ( a(3,2) * a(4,3) - a(3,3) * a(4,2) ) 
     &      ) / det

      b(2,2) = +( 
     &      + a(1,1) * ( a(3,3) * a(4,4) - a(3,4) * a(4,3) ) 
     &      + a(1,3) * ( a(3,4) * a(4,1) - a(3,1) * a(4,4) ) 
     &      + a(1,4) * ( a(3,1) * a(4,3) - a(3,3) * a(4,1) ) 
     &      ) / det

      b(3,2) = -( 
     &      + a(1,1) * ( a(3,2) * a(4,4) - a(3,4) * a(4,2) ) 
     &      + a(1,2) * ( a(3,4) * a(4,1) - a(3,1) * a(4,4) ) 
     &      + a(1,4) * ( a(3,1) * a(4,2) - a(3,2) * a(4,1) ) 
     &      ) / det

      b(4,2) = +( 
     &      + a(1,1) * ( a(3,2) * a(4,3) - a(3,3) * a(4,2) ) 
     &      + a(1,2) * ( a(3,3) * a(4,1) - a(3,1) * a(4,3) ) 
     &      + a(1,3) * ( a(3,1) * a(4,2) - a(3,2) * a(4,1) ) 
     &      ) / det

      b(1,3) = +( 
     &      + a(1,2) * ( a(2,3) * a(4,4) - a(2,4) * a(4,3) ) 
     &      + a(1,3) * ( a(2,4) * a(4,2) - a(2,2) * a(4,4) ) 
     &      + a(1,4) * ( a(2,2) * a(4,3) - a(2,3) * a(4,2) ) 
     &      ) / det

      b(2,3) = -( 
     &      + a(1,1) * ( a(2,3) * a(4,4) - a(2,4) * a(4,3) ) 
     &      + a(1,3) * ( a(2,4) * a(4,1) - a(2,1) * a(4,4) ) 
     &      + a(1,4) * ( a(2,1) * a(4,3) - a(2,3) * a(4,1) ) 
     &      ) / det

      b(3,3) = +( 
     &      + a(1,1) * ( a(2,2) * a(4,4) - a(2,4) * a(4,2) ) 
     &      + a(1,2) * ( a(2,4) * a(4,1) - a(2,1) * a(4,4) ) 
     &      + a(1,4) * ( a(2,1) * a(4,2) - a(2,2) * a(4,1) ) 
     &      ) / det

      b(4,3) = -( 
     &      + a(1,1) * ( a(2,2) * a(4,3) - a(2,3) * a(4,2) ) 
     &      + a(1,2) * ( a(2,3) * a(4,1) - a(2,1) * a(4,3) ) 
     &      + a(1,3) * ( a(2,1) * a(4,2) - a(2,2) * a(4,1) ) 
     &      ) / det

      b(1,4) = -( 
     &      + a(1,2) * ( a(2,3) * a(3,4) - a(2,4) * a(3,3) ) 
     &      + a(1,3) * ( a(2,4) * a(3,2) - a(2,2) * a(3,4) ) 
     &      + a(1,4) * ( a(2,2) * a(3,3) - a(2,3) * a(3,2) ) 
     &      ) / det

      b(2,4) = +( 
     &      + a(1,1) * ( a(2,3) * a(3,4) - a(2,4) * a(3,3) ) 
     &      + a(1,3) * ( a(2,4) * a(3,1) - a(2,1) * a(3,4) ) 
     &      + a(1,4) * ( a(2,1) * a(3,3) - a(2,3) * a(3,1) ) 
     &      ) / det

      b(3,4) = -( 
     &      + a(1,1) * ( a(2,2) * a(3,4) - a(2,4) * a(3,2) ) 
     &      + a(1,2) * ( a(2,4) * a(3,1) - a(2,1) * a(3,4) ) 
     &      + a(1,4) * ( a(2,1) * a(3,2) - a(2,2) * a(3,1) ) 
     &      ) / det

      b(4,4) = +( 
     &      + a(1,1) * ( a(2,2) * a(3,3) - a(2,3) * a(3,2) ) 
     &      + a(1,2) * ( a(2,3) * a(3,1) - a(2,1) * a(3,3) ) 
     &      + a(1,3) * ( a(2,1) * a(3,2) - a(2,2) * a(3,1) ) 
     &      ) / det

      return
      end
      subroutine r8mat_is_identity ( n, a, error_frobenius )

c*********************************************************************72
c
cc R8MAT_IS_IDENTITY determines if an R8MAT is the identity.
c
c  Discussion:
c
c    An R8MAT is a matrix of double precision values.
c
c    The routine returns the Frobenius norm of A - I.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    10 February 2012
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the order of the matrix.
c
c    Input, double precision A(N,N), the matrix.
c
c    Output, double precision ERROR_FROBENIUS, the Frobenius norm
c    of the difference matrix A - I, which would be exactly zero
c    if A were the identity matrix.
c
      implicit none

      integer n

      double precision a(n,n)
      double precision error_frobenius
      integer i
      integer j
      double precision value

      error_frobenius = 0.0D+00

      do i = 1, n
        do j = 1, n
          if ( i .eq. j ) then
            error_frobenius = error_frobenius 
     &        + ( a(i,j) - 1.0D+00 ) ** 2
          else
            error_frobenius = error_frobenius + a(i,j) ** 2
          end if
        end do 
      end do

      error_frobenius = sqrt ( error_frobenius )

      return
      end
      subroutine r8mat_is_symmetric ( m, n, a, error_frobenius )

c*********************************************************************72
c
cc R8MAT_IS_SYMMETRIC checks an R8MAT for symmetry.
c
c  Discussion:
c
c    An R8MAT is a matrix of double precision values.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    19 June 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the order of the matrix.
c
c    Input, double precision A(M,N), the matrix.
c
c    Output, double precision ERROR_FROBENIUS, measures the 
c    Frobenius norm of ( A - A' ), which would be zero if the matrix
c    were exactly symmetric.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      double precision error_frobenius
      integer i
      integer j
      double precision r8_huge
      parameter ( r8_huge = 1.79769313486231571D+308 )

      if ( m .ne. n ) then
        error_frobenius = r8_huge
        return
      end if

      error_frobenius = 0.0D+00
      do j = 1, n
        do i = 1, m
          error_frobenius = error_frobenius + ( a(i,j) - a(j,i) ) ** 2
        end do
      end do

      error_frobenius = sqrt ( error_frobenius )

      return
      end
      subroutine r8mat_jac ( m, n, eps, fx, x, fprime )

c*********************************************************************72
c
cc R8MAT_JAC estimates a dense jacobian matrix of the function FX.
c
c  Discussion:
c
c    An R8MAT is an MxN array of R8's, stored by (I,J) -> [I+J*M].
c
c    FPRIME(I,J) = d F(I) / d X(J).
c
c    The jacobian is assumed to be dense, and the LINPACK/LAPACK
c    double precision general matrix storage mode ("DGE") is used.
c
c    Forward differences are used, requiring N+1 function evaluations.
c
c    Values of EPS have typically been chosen between
c    sqrt ( EPSMCH ) and sqrt ( sqrt ( EPSMCH ) ) where EPSMCH is the
c    machine tolerance.
c
c    If EPS is too small, then F(X+EPS) will be the same as
c    F(X), and the jacobian will be full of zero entries.
c
c    If EPS is too large, the finite difference estimate will
c    be inaccurate.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    11 February 2012
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the number of functions.
c
c    Input, integer N, the number of variables.
c
c    Input, double precision EPS, a tolerance to be used for shifting the
c    X values during the finite differencing.  No single value
c    of EPS will be reliable for all vectors X and functions FX.
c
c    Input, external FX, the name of the user written
c    routine which evaluates the function at a given point X, of the form:
c      subroutine fx ( m, n, x, f )
c      integer m
c      integer n
c      double precision f(m)
c      double precision x(n)
c      f(1:m) = ...
c      return
c      end
c
c    Input, double precision X(N), the point where the jacobian
c    is to be estimated.
c
c    Output, double precision FPRIME(M,N), the M by N estimated jacobian
c    matrix.
c
      implicit none

      integer m
      integer n

      double precision del
      double precision eps
      double precision fprime(m,n)
      external fx
      integer i
      integer j
      double precision x(n)
      double precision xsave
      double precision work1(m)
      double precision work2(m)
c
c  Evaluate the function at the base point, X.
c
      call fx ( m, n, x, work2 )
c
c  Now, one by one, vary each component J of the base point X, and
c  estimate DF(I)/DX(J) = ( F(X+) - F(X) )/ DEL.
c
      do j = 1, n

        xsave = x(j)
        del = eps * ( 1.0D+00 + abs ( x(j) ) )
        x(j) = x(j) + del
        call fx ( m, n, x, work1 )
        x(j) = xsave
        do i = 1, m
          fprime(i,j) = ( work1(i) - work2(i) ) / del
        end do
      end do

      return
      end
      subroutine r8mat_kronecker ( m1, n1, a, m2, n2, b, c )

c*********************************************************************72
c
cc R8MAT_KRONECKER computes the Kronecker product of two R8MAT's.
c
c  Discussion:
c
c    An R8MAT is an MxN array of R8's, stored by (I,J) -> [I+J*M].
c
c    If A is an M1 by N1 array, and B is an M2 by N2 array, then
c    the Kronecker product of A and B is an M1*M2 by N1*N2 array
c      C(I,J) = A(I1,J1) * B(I2,J2)
c    where
c      I1 =     ( I - 1 ) / M2   + 1
c      I2 = mod ( I - 1,    M2 ) + 1
c      J1 =     ( J - 1 ) / N2   + 1
c      J2 = mod ( J - 1,    N2 ) + 1
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    01 December 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M1, N1, the order of the first matrix.
c
c    Input, double precision A(M1,N1), the first matrix.
c
c    Input, integer M2, N2, the order of the second matrix.
c
c    Input, double precision B(M2,N2), the second matrix.
c
c    Output, double precision C(M1*M2,N1*N2), the Kronecker product.
c
      implicit none

      integer m1
      integer m2
      integer n1
      integer n2

      double precision a(m1,n1)
      double precision b(m2,n2)
      double precision c(m1*m2,n1*n2)
      integer i
      integer i0
      integer i1
      integer i2
      integer j
      integer j0
      integer j1
      integer j2

      do j1 = 1, n1
        do i1 = 1, m1
          i0 = ( i1 - 1 ) * m2
          j0 = ( j1 - 1 ) * n2
          j = j0
          do j2 = 1, n2
            j = j + 1
            i = i0
            do i2 = 1, m2
              i = i + 1
              c(i,j) = a(i1,j1) * b(i2,j2)
            end do
          end do
        end do
      end do

      return
      end
      subroutine r8mat_l_inverse ( n, a, b )

c*********************************************************************72
c
cc R8MAT_L_INVERSE inverts a lower triangular R8MAT.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c    A lower triangular matrix is a matrix whose only nonzero entries
c    occur on or below the diagonal.
c
c    The inverse of a lower triangular matrix is a lower triangular matrix.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    14 May 2010
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Albert Nijenhuis, Herbert Wilf,
c    Combinatorial Algorithms for Computers and Calculators,
c    Academic Press, 1978,
c    ISBN: 0-12-519260-6,
c    LC: QA164.N54.
c
c  Parameters:
c
c    Input, integer N, the number of rows and columns in the matrix.
c
c    Input, double precision A(N,N), the lower triangular matrix.
c
c    Output, double precision B(N,N), the inverse matrix.
c
      implicit none

      integer n

      double precision a(n,n)
      double precision b(n,n)
      integer i
      integer j
      integer k

      do j = 1, n

        do i = 1, n

          if ( i .lt. j ) then
            b(i,j) = 0.0D+00
          else if ( j .eq. i ) then
            b(i,j) = 1.0D+00 / a(i,j)
          else
            b(i,j) = 0.0D+00
            do k = 1, i - 1
              b(i,j) = b(i,j) - a(i,k) * b(k,j)
            end do
            b(i,j) = b(i,j) / a(i,i)
          end if

        end do
      end do

      return
      end
      subroutine r8mat_l_print ( m, n, a, title )

c*********************************************************************72
c
cc R8MAT_L_PRINT prints a lower triangular R8MAT.
c
c  Discussion:
c
c    An R8MAT is an MxN array of R8's, stored by (I,J) -> [I+J*M].
c
c  Example:
c
c    M = 5, N = 5
c    A = (/ 11, 21, 31, 41, 51, 22, 32, 42, 52, 33, 43, 53, 44, 54, 55 /)
c
c    11
c    21 22
c    31 32 33
c    41 42 43 44
c    51 52 53 54 55
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    18 March 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the number of rows in A.
c
c    Input, integer N, the number of columns in A.
c
c    Input, double precision A(*), the M by N matrix.  Only the lower
c    triangular elements are stored, in column major order.
c
c    Input, character * ( * ) TITLE, a title.
c
      implicit none

      integer m
      integer n

      double precision a(*)
      integer i
      integer indx(10)
      integer j
      integer jhi
      integer jlo
      integer jmax
      integer nn
      integer size
      character * ( * ) title

      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) trim ( title )

      jmax = min ( n, m )

      if ( m .le. n ) then
       size = ( m * ( m + 1 ) ) / 2
      else if ( n .lt. m ) then
        size = ( n * ( n + 1 ) ) / 2 + ( m - n ) * n
      end if

      nn = 5

      do jlo = 1, jmax, nn
        jhi = min ( jlo + nn - 1, m - 1, jmax )
        write ( *, '(a)' ) ' '
        write ( *, '(8x,5(i8,6x))' ) ( j, j = jlo, jhi )
        write ( *, '(a)' ) ' '
        do i = jlo, m
          jhi = min ( jlo + nn - 1, i, jmax )
          do j = jlo, jhi
            indx(j+1-jlo) = ( j - 1 ) * m + i - ( j * ( j - 1 ) ) / 2
          end do
         write ( *, '(i8,5g14.6)' ) i, a(indx(1:jhi+1-jlo))
        end do
      end do

      return
      end
      subroutine r8mat_l_solve ( n, a, b, x )

c*********************************************************************72
c
cc R8MAT_L_SOLVE solves a lower triangular linear system.
c
c  Discussion:
c
c    An R8MAT is an array of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    07 June 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of rows and columns of
c    the matrix A.
c
c    Input, double precision A(N,N), the N by N lower triangular matrix.
c
c    Input, double precision B(N), the right hand side of the linear system.
c
c    Output, double precision X(N), the solution of the linear system.
c
      implicit none

      integer n

      double precision a(n,n)
      double precision b(n)
      double precision dot
      integer i
      integer j
      double precision x(n)
c
c  Solve L * x = b.
c
      do i = 1, n
        dot = 0.0D+00
        do j = 1, i - 1
          dot = dot + a(i,j) * x(j)
        end do
        x(i) = ( b(i) - dot ) / a(i,i)
      end do

      return
      end
      subroutine r8mat_l1_inverse ( n, a, b )

c*********************************************************************72
c
cc R8MAT_L1_INVERSE inverts a unit lower triangular R8MAT.
c
c  Discussion:
c
c    An R8MAT is an MxN array of R8's, stored by (I,J) -> [I+J*M].
c
c    A unit lower triangular matrix is a matrix with only 1's on the main
c    diagonal, and only 0's above the main diagonal.
c
c    The inverse of a unit lower triangular matrix is also
c    a unit lower triangular matrix.
c
c    This routine can invert a matrix in place, that is, with no extra
c    storage.  If the matrix is stored in A, then the call
c
c      call r8mat_l1_inverse ( n, a, a )
c
c    will result in A being overwritten by its inverse.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    11 February 2012
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Albert Nijenhuis, Herbert Wilf,
c    Combinatorial Algorithms for Computers and Calculators,
c    Academic Press, 1978,
c    ISBN: 0-12-519260-6,
c    LC: QA164.N54.
c
c  Parameters:
c
c    Input, integer N, number of rows and columns in the matrix.
c
c    Input, double precision A(N,N), the unit lower triangular matrix.
c
c    Output, double precision B(N,N), the inverse matrix.
c
      implicit none

      integer n

      double precision a(n,n)
      double precision b(n,n)
      double precision dot
      integer i
      integer j
      integer k

      do i = 1, n

        do j = 1, n

          if ( i .lt. j ) then
            b(i,j) = 0.0D+00
          else if ( j .eq. i ) then
            b(i,j) = 1.0D+00
          else
            dot = 0.0D+00
            do k = 1, i - 1
              dot = dot + a(i,k) * b(k,j)
            end do
            b(i,j) = - dot
          end if

        end do
      end do

      return
      end
      subroutine r8mat_lt_solve ( n, a, b, x )

c*********************************************************************72
c
cc R8MAT_LT_SOLVE solves a transposed lower triangular linear system.
c
c  Discussion:
c
c    An R8MAT is an array of R8's.
c
c    Given the lower triangular matrix A, the linear system to be solved is:
c
c      A' * x = b
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    08 April 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of rows and columns of the matrix A.
c
c    Input, double precision A(N,N), the N by N lower triangular matrix.
c
c    Input, double precision B(N), the right hand side of the linear system.
c
c    Output, double precision X(N), the solution of the linear system.
c
      implicit none

      integer n

      double precision a(n,n)
      double precision b(n)
      integer i
      double precision r8vec_dot_product
      double precision x(n)
c
c  Solve L'*x = b.
c
      do i = n, 1, -1
        x(i) = ( b(i) - r8vec_dot_product ( n - i, x(i+1), a(i+1,i) ) )
     &  / a(i,i)
      end do

      return
      end
      subroutine r8mat_lu ( m, n, a, l, p, u )

c*********************************************************************72
c
cc R8MAT_LU computes the LU factorization of a rectangular R8MAT.
c
c  Discussion:
c
c    An R8MAT is an MxN array of R8's, stored by (I,J) -> [I+J*M].
c
c    The routine is given an M by N matrix A, and produces
c
c      L, an M by M unit lower triangular matrix,
c      U, an M by N upper triangular matrix, and
c      P, an M by M permutation matrix P,
c
c    so that
c
c      A = P' * L * U.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    19 March 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the number of rows in A.
c
c    Input, integer N, the number of columns in A.
c
c    Input, double precision A(M,N), the M by N matrix to be factored.
c
c    Output, double precision L(M,M), the M by M unit lower triangular factor.
c
c    Output, double precision P(M,M), the M by M permutation matrix.
c
c    Output, double precision U(M,N), the M by N upper triangular factor.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer ipiv
      integer j
      integer jj
      double precision l(m,m)
      double precision p(m,m)
      double precision pivot
      double precision u(m,n)
c
c  Initialize:
c
c    U:=A
c    L:=Identity
c    P:=Identity
c
      call r8mat_copy ( m, n, a, u )
      call r8mat_identity ( m, l )
      call r8mat_identity ( m, p )
c
c  On step J, find the pivot row, IPIV, and the pivot value PIVOT.
c
      do j = 1, min ( m - 1, n )

        pivot = 0.0D+00
        ipiv = 0

        do i = j, m

          if ( pivot .lt. abs ( u(i,j) ) ) then
            pivot = abs ( u(i,j) )
            ipiv = i
          end if

        end do
c
c  Unless IPIV is zero, swap rows J and IPIV.
c
        if ( ipiv .ne. 0 ) then

          call r8row_swap ( m, n, u, j, ipiv )

          call r8row_swap ( m, m, l, j, ipiv )

          call r8row_swap ( m, m, p, j, ipiv )
c
c  Zero out the entries in column J, from row J+1 to M.
c
          do i = j + 1, m

            if ( u(i,j) .ne. 0.0D+00 ) then

              l(i,j) = u(i,j) / u(j,j)

              u(i,j) = 0.0D+00

              do jj = j + 1,  n
                u(i,jj) = u(i,jj) - l(i,j) * u(j,jj)
              end do

            end if

          end do

        end if

      end do

      return
      end
      function r8mat_max ( m, n, a )

c*********************************************************************72
c
cc R8MAT_MAX returns the maximum entry of an R8MAT.
c
c  Discussion:
c
c    An R8MAT is an array of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    19 May 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the number of rows in A.
c
c    Input, integer N, the number of columns in A.
c
c    Input, double precision A(M,N), the matrix.
c
c    Output, double precision R8MAT_MAX, the maximum entry of A.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer j
      double precision r8mat_max
      double precision value

      value = a(1,1)
      do j = 1, n
        do i = 1, m
          value = max ( value, a(i,j) )
        end do
      end do

      r8mat_max = value

      return
      end
      subroutine r8mat_max_index ( m, n, a, i, j )

c*********************************************************************72
c
cc R8MAT_MAX_INDEX returns the location of the maximum entry of an R8MAT.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    28 November 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the number of rows in A.
c
c    Input, integer N, the number of columns in A.
c
c    Input, double precision A(M,N), the M by N matrix.
c
c    Output, integer I, J, the indices of the maximum entry of A.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer ii
      integer j
      integer jj

      i = -1
      j = -1

      do jj = 1, n
        do ii = 1, m
          if ( ii .eq. 1 .and. jj .eq. 1 ) then
            i = ii
            j = jj
          else if ( a(i,j) .lt. a(ii,jj) ) then
            i = ii
            j = jj
          end if
        end do
      end do

      return
      end
      function r8mat_maxcol_minrow ( m, n, a )

c*********************************************************************72
c
cc R8MAT_MAXCOL_MINROW gets the maximum column minimum row of an M by N R8MAT.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c    R8MAT_MAXCOL_MINROW = max ( 1 <= I <= N ) ( min ( 1 <= J <= M ) A(I,J) )
c
c    For a given matrix, R8MAT_MAXCOL_MINROW <= R8MAT_MINROW_MAXCOL.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    28 November 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the number of rows in A.
c
c    Input, integer N, the number of columns in A.
c
c    Input, double precision A(M,N), the matrix.
c
c    Output, double precision R8MAT_MAXCOL_MINROW, the maximum column
c    minimum row entry of A.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer j
      double precision r8mat_maxcol_minrow
      double precision r8mat_minrow

      r8mat_maxcol_minrow = 0.0D+00

      do i = 1, m

        r8mat_minrow = a(i,1)
        do j = 2, n
         r8mat_minrow = min ( r8mat_minrow, a(i,j) )
        end do

        if ( i .eq. 1 ) then
          r8mat_maxcol_minrow = r8mat_minrow
        else
          r8mat_maxcol_minrow = max ( r8mat_maxcol_minrow, 
     &      r8mat_minrow )
        end if

      end do

      return
      end
      function r8mat_maxrow_mincol ( m, n, a )

c*********************************************************************72
c
cc R8MAT_MAXROW_MINCOL gets the maximum row minimum column of an M by N R8MAT.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c    R8MAT_MAXROW_MINCOL = max ( 1 <= J <= N ) ( min ( 1 <= I <= M ) A(I,J) )
c
c    For a given matrix, R8MAT_MAXROW_MINCOL <= R8MAT_MINCOL_MAXROW.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    20 March 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the number of rows in A.
c
c    Input, integer N, the number of columns in A.
c
c    Input, double precision A(M,N), the matrix.
c
c    Output, double precision R8MAT_MAXROW_MINCOL, the maximum row
c    minimum column entry of A.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer j
      double precision r8mat_maxrow_mincol
      double precision r8mat_mincol

      r8mat_maxrow_mincol = 0.0D+00

      do j = 1, n

        r8mat_mincol = a(1,j)
        do i = 2, m
          r8mat_mincol = min ( r8mat_mincol, a(i,j) )
        end do

        if ( j .eq. 1 ) then
          r8mat_maxrow_mincol = r8mat_mincol
        else
          r8mat_maxrow_mincol = max ( r8mat_maxrow_mincol, 
     &      r8mat_mincol )
        end if

      end do

      return
      end
      function r8mat_mean ( m, n, a )

c*********************************************************************72
c
cc R8MAT_MEAN returns the mean of an R8MAT.
c
c  Discussion:
c
c    An R8MAT is an array of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    03 September 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the number of rows in A.
c
c    Input, integer N, the number of columns in A.
c
c    Input, double precision A(M,N), the matrix.
c
c    Output, double precision R8MAT_MEAN, the mean of A.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer j
      double precision r8mat_mean
      double precision value

      value = 0.0D+00
      do j = 1, n
        do i = 1, m
          value = value + a(i,j)
        end do
      end do

      value = value / dble ( m * n )

      r8mat_mean = value

      return
      end
      function r8mat_min ( m, n, a )

c*********************************************************************72
c
cc R8MAT_MIN returns the minimum entry of an R8MAT.
c
c  Discussion:
c
c    An R8MAT is an array of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    19 May 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the number of rows.
c
c    Input, integer N, the number of columns.
c
c    Input, double precision A(M,N), the matrix.
c
c    Output, double precision R8MAT_MIN, the minimum entry.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer j
      double precision r8mat_min
      double precision value

      value = a(1,1)
      do j = 1, n
        do i = 1, m
          value = min ( value, a(i,j) )
        end do
      end do

      r8mat_min = value

      return
      end
      subroutine r8mat_min_index ( m, n, a, i, j )

c*********************************************************************72
c
cc R8MAT_MIN_INDEX returns the location of the minimum entry of an R8MAT.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    28 November 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the number of rows.
c
c    Input, integer N, the number of columns.
c
c    Input, double precision A(M,N), the matrix.
c
c    Output, integer I, J, the indices of the minimum entry.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer ii
      integer j
      integer jj

      i = -1
      j = -1

      do jj = 1, n
        do ii = 1, m
          if ( ii .eq. 1 .and. jj .eq. 1 ) then
            i = ii
            j = jj
          else if ( a(ii,jj) .lt. a(i,j) ) then
            i = ii
            j = jj
          end if
        end do
      end do

      return
      end
      function r8mat_mincol_maxrow ( m, n, a )

c*********************************************************************72
c
cc R8MAT_MINCOL_MAXROW gets the minimum column maximum row of an M by N R8MAT.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c    R8MAT_MINCOL_MAXROW = min ( 1 <= I <= N ) ( max ( 1 <= J <= M ) A(I,J) )
c
c    For a given matrix, R8MAT_MAXROW_MINCOL <= R8MAT_MINCOL_MAXROW.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    28 November 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the number of rows.
c
c    Input, integer N, the number of columns.
c
c    Input, double precision A(M,N), the matrix.
c
c    Output, double precision R8MAT_MINCOL_MAXROW, the minimum column
c    maximum row entry.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer j
      double precision r8mat_mincol_maxrow
      double precision r8mat_maxrow

      r8mat_mincol_maxrow = 0.0D+00

      do i = 1, m

        r8mat_maxrow = a(i,1)
        do j = 2, n
          r8mat_maxrow = max ( r8mat_maxrow, a(i,j) )
        end do

        if ( i .eq. 1 ) then
          r8mat_mincol_maxrow = r8mat_maxrow
        else
          r8mat_mincol_maxrow = min ( r8mat_mincol_maxrow, 
     &      r8mat_maxrow )
        end if

      end do

      return
      end
      function r8mat_minrow_maxcol ( m, n, a )

c*********************************************************************72
c
cc R8MAT_MINROW_MAXCOL gets the minimum row maximum column of an M by N R8MAT.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c    R8MAT_MINROW_MAXCOL = min ( 1 <= J <= N ) ( max ( 1 <= I <= M ) A(I,J) )
c
c    For a given matrix, R8MAT_MAXCOL_MINROW <= R8MAT_MINROW_MAXCOL.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    28 November 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the number of rows.
c
c    Input, integer N, the number of columns.
c
c    Input, double precision A(M,N), the matrix.
c
c    Output, double precision R8MAT_MINROW_MAXCOL, the minimum row
c    maximum column entry.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer j
      double precision r8mat_minrow_maxcol
      double precision r8mat_maxcol

      r8mat_minrow_maxcol = 0.0D+00
   
      do j = 1, n

        r8mat_maxcol = a(1,j)
        do i = 2, m
          r8mat_maxcol = max ( r8mat_maxcol, a(i,j) )
        end do

        if ( j .eq. 1 ) then
          r8mat_minrow_maxcol = r8mat_maxcol
        else
          r8mat_minrow_maxcol = min ( r8mat_minrow_maxcol, 
     &      r8mat_maxcol )
        end if

      end do

      return
      end
      subroutine r8mat_minvm ( n1, n2, a, b, c )

c*********************************************************************72
c
cc R8MAT_MINVM computes inverse(A) * B for R8MAT's.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    28 November 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N1, N2, the order of the matrices.
c
c    Input, double precision A(N1,N1), B(N1,N2), the matrices.
c
c    Output, double precision C(N1,N2), the result, C = inverse(A) * B.
c
      implicit none

      integer n1
      integer n2

      double precision a(n1,n1)
      double precision alu(n1,n1)
      double precision b(n1,n2)
      double precision c(n1,n2)
      integer info

      call r8mat_copy ( n1, n1, a, alu )
      call r8mat_copy ( n1, n2, b, c )

      call r8mat_fss ( n1, alu, n2, c, info )
 
      if ( info .ne. 0 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8MAT_MINVM - Fatal error!'
        write ( *, '(a)' ) '  The matrix A was numerically singular.'
        stop 1
      end if

      return
      end
      subroutine r8mat_mm ( n1, n2, n3, a, b, c )

c*********************************************************************72
c
cc R8MAT_MM multiplies two R8MAT's.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c    In FORTRAN90, this operation is more efficiently done by the
c    command:
c
c      C(1:N1,1:N3) = MATMUL ( A(1:N1,1;N2), B(1:N2,1:N3) )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    15 July 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N1, N2, N3, the order of the matrices.
c
c    Input, double precision A(N1,N2), B(N2,N3), the matrices to multiply.
c
c    Output, double precision C(N1,N3), the product matrix C = A * B.
c
      implicit none

      integer n1
      integer n2
      integer n3

      double precision a(n1,n2)
      double precision b(n2,n3)
      double precision c(n1,n3)
      double precision c1(n1,n3)
      integer i
      integer j
      integer k

      do i = 1, n1
        do j = 1, n3
          c1(i,j) = 0.0D+00
          do k = 1, n2
            c1(i,j) = c1(i,j) + a(i,k) * b(k,j)
          end do
        end do
      end do

      do j = 1, n3
        do i = 1, n1
          c(i,j) = c1(i,j)
        end do
      end do

      return
      end
      subroutine r8mat_mmt ( n1, n2, n3, a, b, c )

c*********************************************************************72
c
cc R8MAT_MMT multiplies computes C = A * B' for two R8MAT's.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c    In FORTRAN90, this operation is more efficiently done by the
c    command:
c
c      C(1:N1,1:N3) = matmul ( A(1:N1,1;N2), transpose ( B(1:N3,1:N2) ) )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    13 November 2012
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N1, N2, N3, the order of the matrices.
c
c    Input, double precision A(N1,N2), B(N3,N2), the matrices to multiply.
c
c    Output, double precision C(N1,N3), the product matrix C = A * B'.
c
      implicit none

      integer n1
      integer n2
      integer n3

      double precision a(n1,n2)
      double precision b(n3,n2)
      double precision c(n1,n3)
      double precision c1(n1,n3)
      integer i
      integer j
      integer k

      do i = 1, n1
        do j = 1, n3
          c1(i,j) = 0.0D+00
          do k = 1, n2
            c1(i,j) = c1(i,j) + a(i,k) * b(j,k)
          end do
        end do
      end do

      do j = 1, n3
        do i = 1, n1
          c(i,j) = c1(i,j)
        end do
      end do

      return
      end
      subroutine r8mat_mtm ( n1, n2, n3, a, b, c )

c*********************************************************************72
c
cc R8MAT_MTM multiplies computes C = A' * B for two R8MAT's.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c    In FORTRAN90, this operation is more efficiently done by the
c    command:
c
c      C(1:N1,1:N3) = matmul ( transpose ( A(1:N2,1:N1) ), B(1:N2,1:N3) )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    07 September 2012
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N1, N2, N3, the order of the matrices.
c
c    Input, double precision A(N2,N1), B(N2,N3), the matrices to multiply.
c
c    Output, double precision C(N1,N3), the product matrix C = A' * B.
c
      implicit none

      integer n1
      integer n2
      integer n3

      double precision a(n2,n1)
      double precision b(n2,n3)
      double precision c(n1,n3)
      double precision c1(n1,n3)
      integer i
      integer j
      integer k

      do i = 1, n1
        do j = 1, n3
          c1(i,j) = 0.0D+00
          do k = 1, n2
            c1(i,j) = c1(i,j) + a(k,i) * b(k,j)
          end do
        end do
      end do

      do j = 1, n3
        do i = 1, n1
          c(i,j) = c1(i,j)
        end do
      end do

      return
      end
      subroutine r8mat_mtv ( m, n, a, x, y )

c*****************************************************************************80
c
cc R8MAT_MTV multiplies a transposed matrix times a vector
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    08 August 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns of
c    the matrix.
c
c    Input, double precision A(M,N), the M by N matrix.
c
c    Input, double precision X(M), the vector to be multiplied by A.
c
c    Output, double precision Y(N), the product A'*X.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer j
      double precision x(m)
      double precision y(n)
      double precision y1(n)

      do i = 1, n
        y1(i) = 0.0D+00
        do j = 1, m
          y1(i) = y1(i) + a(j,i) * x(j)
        end do
      end do

      do i = 1, n
        y(i) = y1(i)
      end do

      return
      end
      subroutine r8mat_mv ( m, n, a, x, y )

c*********************************************************************72
c
cc R8MAT_MV multiplies a matrix times a vector.
c
c  Discussion:
c
c    An R8MAT is an array of R8's.
c
c    In FORTRAN90, this operation can be more efficiently carried
c    out by the command
c
c      Y(1:M) = MATMUL ( A(1:M,1:N), X(1:N) )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    12 December 2004
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns of the matrix.
c
c    Input, double precision A(M,N), the M by N matrix.
c
c    Input, double precision X(N), the vector to be multiplied by A.
c
c    Output, double precision Y(M), the product A*X.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer j
      double precision x(n)
      double precision y(m)
      double precision y1(m)

      do i = 1, m
        y1(i) = 0.0D+00
        do j = 1, n
          y1(i) = y1(i) + a(i,j) * x(j)
        end do
      end do

      do i = 1, m
        y(i) = y1(i)
      end do

      return
      end
      subroutine r8mat_nint ( m, n, a )

c*********************************************************************72
c
cc R8MAT_NINT rounds the entries of an R8MAT.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    22 November 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns.
c
c    Input/output, double precision A(M,N), the matrix to be NINT'ed.
c
      implicit none
   
      integer m
      integer n

      double precision a(m,n)
      integer i
      integer j

      do j = 1, n
        do i = 1, m
          a(i,j) = dble ( nint ( a(i,j) ) )
        end do
      end do

      return
      end
      function r8mat_nonzeros ( m, n, a )

c*********************************************************************72
c
cc R8MAT_NONZEROS counts the nonzeros in an R8MAT.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    31 August 2014
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns.
c
c    Input, double precision A(M,N), the matrix.
c
c    Output, integer R8MAT_NONZEROS, the number of nonzero entries.
c
      implicit none
   
      integer m
      integer n

      double precision a(m,n)
      integer i
      integer j
      integer r8mat_nonzeros
      integer value

      value = 0
      do j = 1, n
        do i = 1, m
          if ( a(i,j) .ne. 0.0D+00 ) then
            value = value + 1
          end if
        end do
      end do

      r8mat_nonzeros = value

      return
      end
      function r8mat_norm_eis ( m, n, a )

c*********************************************************************72
c
cc R8MAT_NORM_EIS returns the EISPACK norm of an R8MAT.
c
c  Discussion:
c
c    An R8MAT is an array of R8's.
c
c    The EISPACK norm is defined as:
c
c      R8MAT_NORM_EIS =
c        sum ( 1 <= I <= M ) sum ( 1 <= J <= N ) abs ( A(I,J) )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    23 August 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the number of rows in A.
c
c    Input, integer N, the number of columns in A.
c
c    Input, double precision A(M,N), the matrix whose EISPACK norm is desired.
c
c    Output, double precision R8MAT_NORM_EIS, the EISPACK norm of A.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer j
      double precision r8mat_norm_eis

      r8mat_norm_eis = 0.0D+00
      do j = 1, n
        do i = 1, m
          r8mat_norm_eis = r8mat_norm_eis + abs ( a(i,j) )
        end do
      end do

      return
      end
      function r8mat_norm_fro ( m, n, a )

c*********************************************************************72
c
cc R8MAT_NORM_FRO returns the Frobenius norm of an R8MAT.
c
c  Discussion:
c
c    An R8MAT is an array of R8's.
c
c    The Frobenius norm is defined as
c
c      R8MAT_NORM_FRO = sqrt (
c        sum ( 1 .le. I .le. M ) sum ( 1 .le. j .le. N ) A(I,J)^2 )
c
c    The matrix Frobenius norm is not derived from a vector norm, but
c    is compatible with the vector L2 norm, so that:
c
c      r8vec_norm_l2 ( A * x ) <= r8mat_norm_fro ( A ) * r8vec_norm_l2 ( x ).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    26 July 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the number of rows in A.
c
c    Input, integer N, the number of columns in A.
c
c    Input, double precision A(M,N), the matrix whose Frobenius
c    norm is desired.
c
c    Output, double precision R8MAT_NORM_FRO, the Frobenius norm of A.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer j
      double precision r8mat_norm_fro
      double precision value

      value = 0.0D+00
      do j = 1, n
        do i = 1, m
          value = value + a(i,j) * a(i,j)
        end do
      end do
      value = sqrt ( value )

      r8mat_norm_fro = value

      return
      end
      function r8mat_norm_fro_affine ( m, n, a1, a2 )

c*********************************************************************72
c
cc R8MAT_NORM_FRO_AFFINE returns the Frobenius norm of an R8MAT difference.
c
c  Discussion:
c
c    An R8MAT is an array of R8's.
c
c    The Frobenius norm is defined as
c
c      R8MAT_NORM_FRO = sqrt (
c        sum ( 1 .le. I .le. M ) sum ( 1 .le. j .le. N ) A(I,J)^2 )
c
c    The matrix Frobenius norm is not derived from a vector norm, but
c    is compatible with the vector L2 norm, so that:
c
c      r8vec_norm_l2 ( A * x ) <= r8mat_norm_fro ( A ) * r8vec_norm_l2 ( x ).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    26 September 2012
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the number of rows.
c
c    Input, integer N, the number of columns.
c
c    Input, double precision A1(M,N), A2(M,N), the matrices for which the
c    Frobenius norm of the difference is desired.
c
c    Output, double precision R8MAT_NORM_FRO_AFFINE, the Frobenius norm 
C    of A1 - A2.
c
      implicit none

      integer m
      integer n

      double precision a1(m,n)
      double precision a2(m,n)
      integer i
      integer j
      double precision r8mat_norm_fro_affine
      double precision value

      value = 0.0D+00
      do j = 1, n
        do i = 1, m
          value = value + ( a1(i,j) - a2(i,j) ) ** 2 
        end do
      end do
      value = sqrt ( value )

      r8mat_norm_fro_affine = value

      return
      end
      function r8mat_norm_l1 ( m, n, a )

c*********************************************************************72
c
cc R8MAT_NORM_L1 returns the matrix L1 norm of an R8MAT.
c
c  Discussion:
c
c    An R8MAT is an array of R8's.
c
c    The matrix L1 norm is defined as:
c
c      R8MAT_NORM_L1 = max ( 1 <= J <= N )
c        sum ( 1 <= I <= M ) abs ( A(I,J) ).
c
c    The matrix L1 norm is derived from the vector L1 norm, and
c    satisifies:
c
c      r8vec_norm_l1 ( A * x ) <= r8mat_norm_l1 ( A ) * r8vec_norm_l1 ( x ).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    31 August 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the number of rows in A.
c
c    Input, integer N, the number of columns in A.
c
c    Input, double precision A(M,N), the matrix whose L1 norm is desired.
c
c    Output, double precision R8MAT_NORM_L1, the L1 norm of A.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer j
      double precision r8mat_norm_l1
      double precision sum2

      r8mat_norm_l1 = 0.0D+00

      do j = 1, n
        sum2 = 0.0D+00
        do i = 1, m
          sum2 = sum2 + abs ( a(i,j) )
        end do
        r8mat_norm_l1 = max ( r8mat_norm_l1, sum2 )
      end do

      return
      end
      function r8mat_norm_l2 ( m, n, a )

c*********************************************************************72
c
cc R8MAT_NORM_L2 returns the matrix L2 norm of an R8MAT.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c    The matrix L2 norm is defined as:
c
c      R8MAT_NORM_L2 = sqrt ( max ( 1 <= I <= M ) LAMBDA(I) )
c
c    where LAMBDA contains the eigenvalues of A * A'.
c
c    The matrix L2 norm is derived from the vector L2 norm, and
c    satisifies:
c
c      r8vec_norm_l2 ( A * x ) <= r8mat_norm_l2 ( A ) * r8vec_norm_l2 ( x ).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    23 November 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the number of rows in A.
c
c    Input, integer N, the number of columns in A.
c
c    Input, double precision A(M,N), the matrix whose L2 norm is desired.
c
c    Output, double precision R8MAT_NORM_L2, the L2 norm of A.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      double precision b(m,m)
      double precision d
      double precision diag(m)
      integer i
      integer j
      integer k
      double precision r8mat_norm_l2
      double precision r8vec_max
c
c  Compute B = A * A'.
c
      do j = 1, m
        do i = 1, m
          b(i,j) = 0.0D+00
          do k = 1, n
            b(i,j) = b(i,j) + a(i,k) * a(j,k)
          end do
        end do
      end do
c
c  Diagonalize B.
c
      call r8mat_symm_jacobi ( m, b )
c
c  Find the maximum eigenvalue, and take its square root.
c
      call r8mat_diag_get_vector ( m, b, diag )

      d = r8vec_max ( m, diag )

      r8mat_norm_l2 = sqrt ( d )

      return
      end
      function r8mat_norm_li ( m, n, a )

c*********************************************************************72
c
cc R8MAT_NORM_LI returns the matrix L-oo norm of an R8MAT.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c    The matrix L-oo norm is defined as:
c
c      R8MAT_NORM_LI =  max ( 1 <= I <= M ) sum ( 1 <= J <= N ) abs ( A(I,J) ).
c
c    The matrix L-oo norm is derived from the vector L-oo norm,
c    and satisifies:
c
c      r8vec_norm_li ( A * x ) <= r8mat_norm_li ( A ) * r8vec_norm_li ( x ).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    15 May 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the number of rows in A.
c
c    Input, integer N, the number of columns in A.
c
c    Input, double precision A(M,N), the matrix whose L-oo
c    norm is desired.
c
c    Output, double precision R8MAT_NORM_LI, the L-oo norm of A.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer j
      double precision r8mat_norm_li
      double precision row_sum

      r8mat_norm_li = 0.0D+00

      do i = 1, m
        row_sum = 0.0D+00
        do j = 1, n
          row_sum = row_sum + abs ( a(i,j) )
        end do
        r8mat_norm_li = max ( r8mat_norm_li, row_sum )
      end do

      return
      end
      subroutine r8mat_normal_01 ( m, n, seed, r )

c*********************************************************************72
c
cc R8MAT_NORMAL_01 returns a unit pseudonormal R8MAT.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    12 November 2010
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Paul Bratley, Bennett Fox, Linus Schrage,
c    A Guide to Simulation,
c    Springer Verlag, pages 201-202, 1983.
c
c    Bennett Fox,
c    Algorithm 647:
c    Implementation and Relative Efficiency of Quasirandom
c    Sequence Generators,
c    ACM Transactions on Mathematical Software,
c    Volume 12, Number 4, pages 362-376, 1986.
c
c    Peter Lewis, Allen Goodman, James Miller,
c    A Pseudo-Random Number Generator for the System/360,
c    IBM Systems Journal,
c    Volume 8, pages 136-143, 1969.
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns in the array.
c
c    Input/output, integer SEED, the "seed" value, which should NOT be 0.
c    On output, SEED has been updated.
c
c    Output, double precision R(M,N), the array of pseudonormal values.
c
      implicit none

      integer m
      integer n

      integer seed
      double precision r(m,n)

      call r8vec_normal_01 ( m * n, seed, r )

      return
      end
      subroutine r8mat_nullspace ( m, n, a, nullspace_size, nullspace )

c*********************************************************************72
c
cc R8MAT_NULLSPACE computes the nullspace of a matrix.
c
c  Discussion:
c
c    An R8MAT is an MxN array of R8's, stored by (I,J) -> [I+J*M].
c
c    Let A be an MxN matrix.
c
c    If X is an N-vector, and A*X = 0, then X is a null vector of A.
c
c    The set of all null vectors of A is called the nullspace of A.
c
c    The 0 vector is always in the null space.
c
c    If the 0 vector is the only vector in the nullspace of A, then A
c    is said to have maximum column rank.  (Because A*X=0 can be regarded
c    as a linear combination of the columns of A).  In particular, if A
c    is square, and has maximum column rank, it is nonsingular.
c
c    The dimension of the nullspace is the number of linearly independent
c    vectors that span the nullspace.  If A has maximum column rank,
c    its nullspace has dimension 0.
c
c    This routine uses the reduced row echelon form of A to determine
c    a set of NULLSPACE_SIZE independent null vectors.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    25 March 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns of
c    the matrix A.
c
c    Input, double precision A(M,N), the matrix to be analyzed.
c
c    Input, integer NULLSPACE_SIZE, the size of the nullspace.
c
c    Output, double precision NULLSPACE(N,NULLSPACE_SIZE), vectors that
c    span the nullspace.
c
      implicit none

      integer m
      integer n
      integer nullspace_size

      double precision a(m,n)
      integer col(n)
      integer i
      integer i2
      integer j
      integer j2
      double precision nullspace(n,nullspace_size)
      integer row(m)
      double precision rref(m,n)
c
c  Make a copy of A.
c
      do j = 1, n
        do i = 1, m
          rref(i,j) = a(i,j)
        end do
      end do
c
c  Get the reduced row echelon form of A.
c
      call r8mat_rref ( m, n, rref )
c
c  Note in ROW the columns of the leading nonzeros.
c  COL(J) = +J if there is a leading 1 in that column, and -J otherwise.
c
      do i = 1, m
        row(i) = 0
      end do

      do j = 1, n
        col(j) = - j
      end do

      do i = 1, m

        do j = 1, n
          if ( rref(i,j) .eq. 1.0D+00 ) then
            row(i) = j
            col(j) = j
            go to 10
          end if
        end do

10      continue

      end do

      do j = 1, nullspace_size
        do i = 1, n
          nullspace(i,j) = 0.0D+00
        end do
      end do

      j2 = 0
c
c  If column J does not contain a leading 1, then it contains
c  information about a null vector.
c
      do j = 1, n

        if ( col(j) .lt. 0 ) then

          j2 = j2 + 1

          do i = 1, m
            if ( rref(i,j) .ne. 0.0D+00 ) then
              i2 = row(i)
              nullspace(i2,j2) = - rref(i,j)
            end if
          end do

          nullspace(j,j2) = 1.0D+00

        end if

      end do

      return
      end
      subroutine r8mat_nullspace_size ( m, n, a, nullspace_size )

c*********************************************************************72
c
cc R8MAT_NULLSPACE_SIZE computes the size of the nullspace of a matrix.
c
c  Discussion:
c
c    An R8MAT is an MxN array of R8's, stored by (I,J) -> [I+J*M].
c
c    Let A be an MxN matrix.
c
c    If X is an N-vector, and A*X = 0, then X is a null vector of A.
c
c    The set of all null vectors of A is called the nullspace of A.
c
c    The 0 vector is always in the null space.
c
c    If the 0 vector is the only vector in the nullspace of A, then A
c    is said to have maximum column rank.  (Because A*X=0 can be regarded
c    as a linear combination of the columns of A).  In particular, if A
c    is square, and has maximum column rank, it is nonsingular.
c
c    The dimension of the nullspace is the number of linearly independent
c    vectors that span the nullspace.  If A has maximum column rank,
c    its nullspace has dimension 0.
c
c    This routine ESTIMATES the dimension of the nullspace.  Cases of
c    singularity that depend on exact arithmetic will probably be missed.
c
c    The nullspace will be estimated by counting the leading 1's in the
c    reduced row echelon form of A, and subtracting this from N.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    25 March 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns of
c    the matrix A.
c
c    Input, double precision A(M,N), the matrix to be analyzed.
c
c    Output, integer NULLSPACE_SIZE, the estimated size
c    of the nullspace.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer j
      integer leading
      integer nullspace_size
      double precision rref(m,n)
c
c  Get the reduced row echelon form of A.
c
      do j = 1, n
        do i = 1, m
          rref(i,j) = a(i,j)
        end do
      end do

      call r8mat_rref ( m, n, rref )
c
c  Count the leading 1's in A.
c
      leading = 0
      do i = 1, m

        do j = 1, n
          if ( rref(i,j) .eq. 1.0D+00 ) then
            leading = leading + 1
            go to 10
          end if
        end do

10      continue

      end do

      nullspace_size = n - leading

      return
      end  
      subroutine r8mat_orth_uniform ( n, seed, q )

c*********************************************************************72
c
cc R8MAT_ORTH_UNIFORM returns a random orthogonal R8MAT.
c
c  Discussion:
c
c    An R8MAT is an array of R8's.
c
c    Thanks to Eugene Petrov, B I Stepanov Institute of Physics,
c    National Academy of Sciences of Belarus, for convincingly
c    pointing out the severe deficiencies of an earlier version of
c    this routine.
c
c    Essentially, the computation involves saving the Q factor of the
c    QR factorization of a matrix whose entries are normally distributed.
c    However, it is only necessary to generate this matrix a column at
c    a time, since it can be shown that when it comes time to annihilate
c    the subdiagonal elements of column K, these (transformed) elements of
c    column K are still normally distributed random values.  Hence, there
c    is no need to generate them at the beginning of the process and
c    transform them K-1 times.
c
c    For computational efficiency, the individual Householder transformations
c    could be saved, as recommended in the reference, instead of being
c    accumulated into an explicit matrix format.
c
c  Properties:
c
c    The inverse of A is equal to A'.
c
c    A * A'  = A' * A = I.
c
c    Columns and rows of A have unit Euclidean norm.
c
c    Distinct pairs of columns of A are orthogonal.
c
c    Distinct pairs of rows of A are orthogonal.
c
c    The L2 vector norm of A*x = the L2 vector norm of x for any vector x.
c
c    The L2 matrix norm of A*B = the L2 matrix norm of B for any matrix B.
c
c    The determinant of A is +1 or -1.
c
c    All the eigenvalues of A have modulus 1.
c
c    All singular values of A are 1.
c
c    All entries of A are between -1 and 1.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    04 November 2004
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Pete Stewart,
c    Efficient Generation of Random Orthogonal Matrices With an Application
c    to Condition Estimators,
c    SIAM Journal on Numerical Analysis,
c    Volume 17, Number 3, June 1980, pages 403-409.
c
c  Parameters:
c
c    Input, integer N, the order of Q.
c
c    Input/output, integer SEED, a seed for the random number generator.
c
c    Output, double precision Q(N,N), the orthogonal matrix.
c
      implicit none

      integer n

      double precision a_col(n)
      integer i
      integer j
      double precision q(n,n)
      double precision r8_normal_01
      integer seed
      double precision v(n)
c
c  Start with Q = the identity matrix.
c
      do i = 1, n
        do j = 1, n
          if ( i .eq. j ) then
            q(i,j) = 1.0D+00
          else
            q(i,j) = 0.0D+00
          end if
        end do
      end do
c
c  Now behave as though we were computing the QR factorization of
c  some other random matrix A.  Generate the N elements of the first column,
c  compute the Householder matrix H1 that annihilates the subdiagonal elements,
c  and set Q := Q * H1' = Q * H.
c
c  On the second step, generate the lower N-1 elements of the second column,
c  compute the Householder matrix H2 that annihilates them,
c  and set Q := Q * H2' = Q * H2 = H1 * H2.
c
c  On the N-1 step, generate the lower 2 elements of column N-1,
c  compute the Householder matrix HN-1 that annihilates them, and
c  and set Q := Q * H(N-1)' = Q * H(N-1) = H1 * H2 * ... * H(N-1).
c  This is our random orthogonal matrix.
c
      do j = 1, n - 1
c
c  Set the vector that represents the J-th column to be annihilated.
c
        do i = 1, j - 1
          a_col(i) = 0.0D+00
        end do

        do i = j, n
          a_col(i) = r8_normal_01 ( seed )
        end do
c
c  Compute the vector V that defines a Householder transformation matrix
c  H(V) that annihilates the subdiagonal elements of A.
c
        call r8vec_house_column ( n, a_col, j, v )
c
c  Postmultiply the matrix Q by H'(V) = H(V).
c
        call r8mat_house_axh ( n, q, v, q )

      end do

      return
      end
      subroutine r8mat_plot ( m, n, a, title )

c*********************************************************************72
c
cc R8MAT_PLOT "plots" an R8MAT, with an optional title.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    02 December 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the number of rows in A.
c
c    Input, integer N, the number of columns in A.
c
c    Input, double precision A(M,N), the matrix.
c
c    Input, character * ( * ) TITLE, a title.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer j
      integer jhi
      integer jlo
      character r8mat_plot_symbol
      character * ( 70 ) string
      character * ( * ) title

      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) trim ( title )

      do jlo = 1, n, 70
        jhi = min ( jlo + 70-1, n )
        write ( *, '(a)' ) ' '
        write ( *, '(8x,2x,70i1)' ) ( mod ( j, 10 ), j = jlo, jhi )
        write ( *, '(a)' ) ' '

        do i = 1, m
          do j = jlo, jhi
            string(j+1-jlo:j+1-jlo) = r8mat_plot_symbol ( a(i,j) )
          end do
          write ( *, '(i8,2x,a)' ) i, string(1:jhi+1-jlo)
        end do
      end do
 
      return
      end
      function r8mat_plot_symbol ( r )

c*********************************************************************72
c
cc R8MAT_PLOT_SYMBOL returns a symbol for an element of an R8MAT.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    24 November 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision R, a value whose symbol is desired.
c
c    Output, character R8MAT_PLOT_SYMBOL, is
c    '-' if R is negative,
c    '0' if R is zero,
c    '+' if R is positive.
c
      implicit none

      character r8mat_plot_symbol
      double precision r

      if ( r .lt. 0.0D+00 ) then
        r8mat_plot_symbol = '-'
      else if ( r .eq. 0.0D+00 ) then
        r8mat_plot_symbol = '0'
      else if ( 0.0D+00 .lt. r ) then
        r8mat_plot_symbol = '+'
      end if

      return
      end
      subroutine r8mat_poly_char ( n, a, p )

c*********************************************************************72
c
cc R8MAT_POLY_CHAR computes the characteristic polynomial of an R8MAT.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    24 November 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the order of the matrix A.
c
c    Input, double precision A(N,N), the N by N matrix.
c
c    Output, double precision P(0:N), the coefficients of the characteristic
c    polynomial of A.  P(N) contains the coefficient of X^N
c    (which will be 1), P(I) contains the coefficient of X^I,
c    and P(0) contains the constant term.
c
      implicit none

      integer n

      double precision a(n,n)
      integer i
      integer j
      integer k
      integer order
      double precision p(0:n)
      double precision r8mat_trace
      double precision trace
      double precision work1(n,n)
      double precision work2(n,n)
c
c  Initialize WORK1 to the identity matrix.
c
      call r8mat_identity ( n, work1 )

      p(n) = 1.0D+00

      do order = n - 1, 0, -1
c
c  Work2 = A * WORK1.
c
        do j = 1, n
          do i = 1, n
            work2(i,j) = 0.0D+00
            do k = 1, n
              work2(i,j) = work2(i,j) + a(i,k) * work1(k,j)
            end do
          end do
        end do
c
c  Take the trace.
c
        trace = r8mat_trace ( n, work2 )
c
c  P(ORDER) = -Trace ( WORK2 ) / ( N - ORDER )
c
        p(order) = - trace / dble ( n - order )
c
c  WORK1 := WORK2 + P(ORDER) * Identity.
c
        do j = 1, n
          do i = 1, n
            work1(i,j) = work2(i,j)
          end do
        end do

        do i = 1, n
          work1(i,i) = work1(i,i) + p(order)
        end do

      end do

      return
      end
      subroutine r8mat_power ( n, a, npow, b )

c*********************************************************************72
c
cc R8MAT_POWER computes a nonnegative power of an R8MAT.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c    The algorithm is:
c
c      B = I
c      do NPOW times:
c        B = A * B
c      end
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    23 April 2005
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the order of A.
c
c    Input, double precision A(N,N), the matrix to be raised to a power.
c
c    Input, integer NPOW, the power to which A is to be raised.
c    NPOW must be nonnegative.
c
c    Output, double precision B(N,N), the value of A^NPOW.
c
      implicit none

      integer n

      double precision a(n,n)
      double precision b(n,n)
      double precision c(n,n)
      integer i
      integer ipow
      integer j
      integer k
      integer npow

      if ( npow .lt. 0 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8MAT_POWER - Fatal error!'
        write ( *, '(a)' ) '  Input value of NPOW < 0.'
        write ( *, '(a,i8)' ) '  NPOW = ', npow
        stop 1
      end if

      call r8mat_identity ( n, b )

      do ipow = 1, npow

        do j = 1, n
          do i = 1, n
            c(i,j) = b(i,j)
          end do
        end do

        do j = 1, n
          do i = 1, n
            do k = 1, n
              b(i,j) = b(i,j) + a(i,k) * c(k,j)
            end do
          end do
        end do

      end do

      return
      end
      subroutine r8mat_power_method ( n, a, r, v )

c*********************************************************************72
c
cc R8MAT_POWER_METHOD applies the power method to an R8MAT.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c    If the power method has not converged, then calling the routine
c    again immediately with the output from the previous call will
c    continue the iteration.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    02 December 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the order of A.
c
c    Input, double precision A(N,N), the matrix.
c
c    Output, double precision R, the estimated eigenvalue.
c
c    Input/output, double precision V(N), on input, an estimate
c    for the eigenvector.  On output, an improved estimate for the
c    eigenvector.
c
      implicit none

      integer n

      double precision a(n,n)
      double precision av(n)
      double precision eps
      integer i
      integer it
      double precision it_eps
      parameter ( it_eps = 0.0001D+00 )
      integer it_max 
      parameter ( it_max = 100 )
      integer it_min
      parameter ( it_min = 10 )
      integer j
      double precision r
      double precision r2
      double precision r8_epsilon
      double precision r_old
      double precision v(n)

      eps = sqrt ( r8_epsilon ( ) )

      r = 0.0D+00
      do i = 1, n
        r = r + v(i) ** 2
      end do
      r = sqrt ( r )

      if ( r .eq. 0.0D+00 ) then
        do i = 1, n
          v(i) = 1.0D+00
        end do
        r = sqrt ( dble ( n ) )
      end if

      do i = 1, n
        v(i) = v(i) / r
      end do
 
      do it = 1, it_max

        call r8mat_mv ( n, n, a, v, av )

        r_old = r
 
        r = 0.0D+00
        do i = 1, n
          r = r + av(i) ** 2
        end do
        r = sqrt ( r )

        if ( it_min .lt. it ) then
          if ( abs ( r - r_old ) .le. 
     &      it_eps * ( 1.0D+00 + abs ( r ) ) ) then
            go to 10
          end if
        end if

        do i = 1, n
          v(i) = av(i)
        end do

        if ( r .ne. 0.0D+00 ) then
          do i = 1, n
            v(i) = v(i) / r
          end do
        end if
c
c  Perturb V a bit, to avoid cases where the initial guess is exactly
c  the eigenvector of a smaller eigenvalue.
c
        if ( it .lt. it_max / 2 ) then
          j = 1 + mod ( it - 1, n )
          v(j) = v(j) + eps * ( 1.0D+00 + abs ( v(j) ) )
          r2 = 0.0D+00
          do i = 1, n
            r2 = r2 + v(i) ** 2
          end do
          r2 = sqrt ( r2 )
          do i = 1, n
            v(i) = v(i) / r2
          end do
        end if

      end do

10    continue

      return
      end
      subroutine r8mat_print ( m, n, a, title )

c*********************************************************************72
c
cc R8MAT_PRINT prints an R8MAT.
c
c  Discussion:
c
c    An R8MAT is an array of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    20 May 2004
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the number of rows in A.
c
c    Input, integer N, the number of columns in A.
c
c    Input, double precision A(M,N), the matrix.
c
c    Input, character * ( * ) TITLE, a title.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      character * ( * ) title

      call r8mat_print_some ( m, n, a, 1, 1, m, n, title )

      return
      end
      subroutine r8mat_print_some ( m, n, a, ilo, jlo, ihi, jhi,
     &  title )

c*********************************************************************72
c
cc R8MAT_PRINT_SOME prints some of an R8MAT.
c
c  Discussion:
c
c    An R8MAT is an array of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    25 January 2007
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns.
c
c    Input, double precision A(M,N), an M by N matrix to be printed.
c
c    Input, integer ILO, JLO, the first row and column to print.
c
c    Input, integer IHI, JHI, the last row and column to print.
c
c    Input, character * ( * ) TITLE, a title.
c
      implicit none

      integer incx
      parameter ( incx = 5 )
      integer m
      integer n

      double precision a(m,n)
      character * ( 14 ) ctemp(incx)
      integer i
      integer i2hi
      integer i2lo
      integer ihi
      integer ilo
      integer inc
      integer j
      integer j2
      integer j2hi
      integer j2lo
      integer jhi
      integer jlo
      character * ( * ) title

      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) trim ( title )

      if ( m .le. 0 .or. n .le. 0 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) '  (None)'
        return
      end if

      do j2lo = max ( jlo, 1 ), min ( jhi, n ), incx

        j2hi = j2lo + incx - 1
        j2hi = min ( j2hi, n )
        j2hi = min ( j2hi, jhi )

        inc = j2hi + 1 - j2lo

        write ( *, '(a)' ) ' '

        do j = j2lo, j2hi
          j2 = j + 1 - j2lo
          write ( ctemp(j2), '(i7,7x)') j
        end do

        write ( *, '(''  Col   '',5a14)' ) ( ctemp(j), j = 1, inc )
        write ( *, '(a)' ) '  Row'
        write ( *, '(a)' ) ' '

        i2lo = max ( ilo, 1 )
        i2hi = min ( ihi, m )

        do i = i2lo, i2hi

          do j2 = 1, inc

            j = j2lo - 1 + j2

            write ( ctemp(j2), '(g14.6)' ) a(i,j)

          end do

          write ( *, '(i5,a,5a14)' ) i, ':', ( ctemp(j), j = 1, inc )

        end do

      end do

      return
      end
      subroutine r8mat_ref ( m, n, a )

c*********************************************************************72
c
cc R8MAT_REF computes the row echelon form of a matrix.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c    A matrix is in row echelon form if:
c
c    * The first nonzero entry in each row is 1.
c
c    * The leading 1 in a given row occurs in a column to
c      the right of the leading 1 in the previous row.
c
c    * Rows which are entirely zero must occur last.
c
c  Example:
c
c    Input matrix:
c
c     1.0  3.0  0.0  2.0  6.0  3.0  1.0
c    -2.0 -6.0  0.0 -2.0 -8.0  3.0  1.0
c     3.0  9.0  0.0  0.0  6.0  6.0  2.0
c    -1.0 -3.0  0.0  1.0  0.0  9.0  3.0
c
c    Output matrix:
c
c     1.0  3.0  0.0  2.0  6.0  3.0  1.0
c     0.0  0.0  0.0  1.0  2.0  4.5  1.5
c     0.0  0.0  0.0  0.0  0.0  1.0  0.3
c     0.0  0.0  0.0  0.0  0.0  0.0  0.0
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    02 December 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns of
c    the matrix A.
c
c    Input/output, double precision A(M,N).  On input, the matrix to be
c    analyzed.  On output, the REF form of the matrix.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer j
      integer lead
      integer r
      double precision temp

      lead = 1

      do r = 1, m

        if ( n .lt. lead ) then
          go to 30
        end if

        i = r
 
10      continue

        if ( a(i,lead) .eq. 0.0D+00 ) then

          i = i + 1

          if ( m .lt. i ) then
            i = r
            lead = lead + 1
            if ( n .lt. lead ) then
              lead = -1
              go to 20
            end if
          end if

          go to 10

        end if

20      continue

        if ( lead .lt. 0 ) then
          go to 30
        end if

        do j = 1, n
          temp   = a(i,j)
          a(i,j) = a(r,j)
          a(r,j) = temp
        end do

        do j = 1, n
          a(r,j) = a(r,j) / a(r,lead)
        end do

        do i = r + 1, m
          do j = 1, n
            a(i,j) = a(i,j) - a(i,lead) * a(r,j)
          end do
        end do

        lead = lead + 1

      end do

30    continue

      return
      end
      function r8mat_rms ( m, n, a )

c*********************************************************************72
c
cc R8MAT_RMS returns the RMS norm of an R8MAT.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c    The matrix RMS norm is defined as:
c
c      R8MAT_RMS = sqrt ( 
c        sum ( 1 <= J <= N ) sum ( 1 <= I <= M ) A(I,J)^2 / M / N ).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    14 December 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the dimensions of the matrix.
c
c    Input, double precision A(M,N), the matrix.
c
c    Output, double precision R8MAT_RMS, the RMS norm of A.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer j
      double precision r8mat_rms
      double precision value

      value = 0.0D+00

      do j = 1, n
        do i = 1, m
          value = value + a(i,j) * a(i,j)
        end do
      end do

      value = sqrt ( value / dble ( m ) / dble ( n ) )

      r8mat_rms = value

      return
      end
      subroutine r8mat_row_copy ( m, n, i, v, a )

c*********************************************************************72
c
cc R8MAT_ROW_COPY copies a vector into a row of an R8MAT.
c
c  Discussion:
c
c    An R8MAT is an MxN array of R8's, stored by (I,J) -> [I+J*M].
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    30 June 2014
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the order of the matrix.
c
c    Input, integer I, the index of the row.
c    1 <= I <= M.
c
c    Input, double precision V(N), the row to be copied.
c
c    Input/output, double precision A(M,N), the matrix into which
c    the row is to be copied.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer j
      double precision v(n)

      do j = 1, n
        a(i,j) = v(j)
      end do

      return
      end
      subroutine r8mat_row_set ( i, r, m, n, a )

c*********************************************************************72
c
cc R8MAT_ROW_SET copies a vector into a row of an R8MAT.
c
c  Discussion:
c
c    Because I try to avoid using "leading dimensions", which allow
c    a user to set aside too much space for an array, but then
c    still put things in the right place, I need to use a routine
c    like this when I occasionally have to deal with arrays that
c    are not "tight".
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    15 February 2014
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer I, the row index.
c
c    Input, double precision R(N), the vector.
c
c    Input, integer M, N, the number of rows and columns of the matrix.
c
c    Input/output, double precision A(M,N), the matrix to be updated.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer j
      double precision r(n)

      do j = 1, n
        a(i,j) = r(j)
      end do

      return
      end
      subroutine r8mat_rref ( m, n, a )

c*********************************************************************72
c
cc R8MAT_RREF computes the reduced row echelon form of a matrix.
c
c  Discussion:
c
c    An R8MAT is an MxN array of R8's, stored by (I,J) -> [I+J*M].
c
c    A matrix is in row echelon form if:
c
c    * The first nonzero entry in each row is 1.
c
c    * The leading 1 in a given row occurs in a column to
c      the right of the leading 1 in the previous row.
c
c    * Rows which are entirely zero must occur last.
c
c    The matrix is in reduced row echelon form if, in addition to
c    the first three conditions, it also satisfies:
c
c    * Each column containing a leading 1 has no other nonzero entries.
c
c  Example:
c
c    Input matrix:
c
c     1.0  3.0  0.0  2.0  6.0  3.0  1.0
c    -2.0 -6.0  0.0 -2.0 -8.0  3.0  1.0
c     3.0  9.0  0.0  0.0  6.0  6.0  2.0
c    -1.0 -3.0  0.0  1.0  0.0  9.0  3.0
c
c    Output matrix:
c
c     1.0  3.0  0.0  0.0  2.0  0.0  0.0
c     0.0  0.0  0.0  1.0  2.0  0.0  0.0
c     0.0  0.0  0.0  0.0  0.0  1.0  0.3
c     0.0  0.0  0.0  0.0  0.0  0.0  0.0
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    25 March 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer  M, N, the number of rows and columns of
c    the matrix A.
c
c    Input/output, double precision A(M,N).  On input, the matrix to be
c    analyzed.  On output, the RREF form of the matrix.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer j
      integer lead
      integer r
      double precision temp

      lead = 1

      do r = 1, m

        if ( n .lt. lead ) then
          go to 30
        end if

        i = r

10      continue

        if ( a(i,lead) .eq. 0.0D+00 ) then

          i = i + 1

          if ( m .lt. i ) then
            i = r
            lead = lead + 1
            if ( n .lt. lead ) then
              lead = -1
              go to 20
            end if
          end if

          go to 10

        end if

20      continue

        if ( lead .lt. 0 ) then
          go to 30
        end if

        do j = 1, n
          temp   = a(i,j)
          a(i,j) = a(r,j)
          a(r,j) = temp
        end do

        temp = a(r,lead)
        do j = 1, n
          a(r,j) = a(r,j) / temp
        end do

        do i = 1, m
          if ( i .ne. r ) then
            do j = 1, n
              a(i,j) = a(i,j) - a(i,lead) * a(r,j)
            end do
          end if
        end do

        lead = lead + 1

      end do

30    continue

      return
      end
      subroutine r8mat_scale ( m, n, s, a )

c*********************************************************************72
c
cc R8MAT_SCALE multiplies an R8MAT by a scalar.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    01 December 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns.
c
c    Input, double precision S, the scale factor.
c
c    Input/output, double precision A(M,N), the matrix to be scaled.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer j
      double precision s

      do j = 1, n
        do i = 1, m
          a(i,j) = a(i,j) * s
        end do
      end do

      return
      end
      subroutine r8mat_solve ( n, rhs_num, a, info )

c*********************************************************************72
c
cc R8MAT_SOLVE uses Gauss-Jordan elimination to solve an N by N linear system.
c
c  Discussion:
c
c    An R8MAT is an array of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    08 July 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the order of the matrix.
c
c    Input, integer RHS_NUM, the number of right hand sides.
c    RHS_NUM must be at least 0.
c
c    Input/output, double precision A(N,N+rhs_num), contains in rows and
c    columns 1 to N the coefficient matrix, and in columns N+1 through
c    N+rhs_num, the right hand sides.  On output, the coefficient matrix
c    area has been destroyed, while the right hand sides have
c    been overwritten with the corresponding solutions.
c
c    Output, integer INFO, singularity flag.
c    0, the matrix was not singular, the solutions were computed;
c    J, factorization failed on step J, and the solutions could not
c    be computed.
c
      implicit none

      integer n
      integer rhs_num

      double precision a(n,n+rhs_num)
      double precision apivot
      double precision factor
      integer i
      integer info
      integer ipivot
      integer j
      integer k
      double precision t

      info = 0

      do j = 1, n
c
c  Choose a pivot row.
c
        ipivot = j
        apivot = a(j,j)

        do i = j+1, n
          if ( abs ( apivot ) .lt. abs ( a(i,j) ) ) then
            apivot = a(i,j)
            ipivot = i
          end if
        end do

        if ( apivot .eq. 0.0D+00 ) then
          info = j
          return
        end if
c
c  Interchange.
c
        do i = 1, n + rhs_num
          t = a(ipivot,i)
          a(ipivot,i) = a(j,i)
          a(j,i) = t
        end do
c
c  A(J,J) becomes 1.
c
        a(j,j) = 1.0D+00
        do k = j + 1, n + rhs_num
          a(j,k) = a(j,k) / apivot
        end do
c
c  A(I,J) becomes 0.
c
        do i = 1, n

          if ( i .ne. j ) then

            factor = a(i,j)
            a(i,j) = 0.0D+00
            do k = j + 1, n + rhs_num
              a(i,k) = a(i,k) - factor * a(j,k)
            end do

          end if

        end do

      end do

      return
      end
      subroutine r8mat_solve_2d ( a, b, det, x )

c*********************************************************************72
c
cc R8MAT_SOLVE_2D solves a 2 by 2 linear system using Cramer's rule.
c
c  Discussion:
c
c    An R8MAT is an array of R8's.
c
c    If DET is zero, then A is singular, and does not have an
c    inverse.  In that case, X is simply set to zero, and a
c    message is printed.
c
c    If DET is nonzero, then its value is roughly an estimate
c    of how nonsingular the matrix A is.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    08 July 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision A(2,2), the matrix.
c
c    Input, double precision B(2), the right hand side.
c
c    Output, double precision DET, the determinant of the matrix A.
c
c    Output, double precision X(2), the solution of the system,
c    if DET is nonzero.
c
      implicit none

      double precision a(2,2)
      double precision b(2)
      double precision det
      double precision x(2)
c
c  Compute the determinant.
c
      det = a(1,1) * a(2,2) - a(1,2) * a(2,1)
c
c  If the determinant is zero, bail out.
c
      if ( det .eq. 0.0D+00 ) then
        x(1) = 0.0D+00
        x(2) = 0.0D+00
        return
      end if
c
c  Compute the solution.
c
      x(1) = (  a(2,2) * b(1) - a(1,2) * b(2) ) / det
      x(2) = ( -a(2,1) * b(1) + a(1,1) * b(2) ) / det

      return
      end
      subroutine r8mat_solve_3d ( a, b, det, x )

c*********************************************************************72
c
cc R8MAT_SOLVE_3D solves a 3 by 3 linear system using Cramer's rule.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c    If the determinant DET is returned as zero, then the matrix A is
c    singular, and does not have an inverse.  In that case, X is
c    returned as the zero vector.
c
c    If DET is nonzero, then its value is roughly an estimate
c    of how nonsingular the matrix A is.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    15 May 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision A(3,3), the matrix.
c
c    Input, double precision B(3), the right hand side.
c
c    Output, double precision DET, the determinant of the matrix A.
c
c    Output, double precision X(3), the solution of the system,
c    if DET is nonzero.
c
      implicit none

      double precision a(3,3)
      double precision b(3)
      double precision det
      double precision x(3)
c
c  Compute the determinant.
c
      det =  a(1,1) * ( a(2,2) * a(3,3) - a(2,3) * a(3,2) )
     &     + a(1,2) * ( a(2,3) * a(3,1) - a(2,1) * a(3,3) )
     &     + a(1,3) * ( a(2,1) * a(3,2) - a(2,2) * a(3,1) )
c
c  If the determinant is zero, bail out.
c
      if ( det .eq. 0.0D+00 ) then
        x(1) = 0.0D+00
        x(2) = 0.0D+00
        x(3) = 0.0D+00
        return
      end if
c
c  Compute the solution.
c
      x(1) = (   ( a(2,2) * a(3,3) - a(2,3) * a(3,2) ) * b(1)
     &         - ( a(1,2) * a(3,3) - a(1,3) * a(3,2) ) * b(2)
     &         + ( a(1,2) * a(2,3) - a(1,3) * a(2,2) ) * b(3) ) / det

      x(2) = ( - ( a(2,1) * a(3,3) - a(2,3) * a(3,1) ) * b(1)
     &         + ( a(1,1) * a(3,3) - a(1,3) * a(3,1) ) * b(2)
     &         - ( a(1,1) * a(2,3) - a(1,3) * a(2,1) ) * b(3) ) / det

      x(3) = (   ( a(2,1) * a(3,2) - a(2,2) * a(3,1) ) * b(1)
     &         - ( a(1,1) * a(3,2) - a(1,2) * a(3,1) ) * b(2)
     &         + ( a(1,1) * a(2,2) - a(1,2) * a(2,1) ) * b(3) ) / det

      return
      end
      subroutine r8mat_solve2 ( n, a, b, x, ierror )

c*********************************************************************72
c
cc R8MAT_SOLVE2 computes the solution of an N by N linear system.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c    The linear system may be represented as
c
c      A*X = B
c
c    If the linear system is singular, but consistent, then the routine will
c    still produce a solution.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    19 August 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of equations.
c
c    Input/output, double precision A(N,N).
c    On input, A is the coefficient matrix to be inverted.
c    On output, A has been overwritten.
c
c    Input/output, double precision B(N).
c    On input, B is the right hand side of the system.
c    On output, B has been overwritten.
c
c    Output, double precision X(N), the solution of the linear system.
c
c    Output, integer IERROR.
c    0, no error detected.
c    1, consistent singularity.
c    2, inconsistent singularity.
c
      implicit none

      integer n

      double precision a(n,n)
      double precision amax
      double precision b(n)
      integer i
      integer ierror
      integer imax
      integer ipiv(n)
      integer j
      integer k
      double precision x(n)

      ierror = 0

      do i = 1, n
        ipiv(i) = 0
      end do

      do i = 1, n
        x(i) = 0.0D+00
      end do
c
c  Process the matrix.
c
      do k = 1, n
c
c  In column K:
c    Seek the row IMAX with the properties that:
c      IMAX has not already been used as a pivot;
c      A(IMAX,K) is larger in magnitude than any other candidate.
c
        amax = 0.0D+00
        imax = 0
        do i = 1, n
          if ( ipiv(i) .eq. 0 ) then
            if ( amax .lt. abs ( a(i,k) ) ) then
              imax = i
              amax = abs ( a(i,k) )
            end if
          end if
        end do
c
c  If you found a pivot row IMAX, then,
c    eliminate the K-th entry in all rows that have not been used for pivoting.
c
        if ( imax .ne. 0 ) then

          ipiv(imax) = k
          do j = k + 1, n
            a(imax,j) = a(imax,j) / a(imax,k)
          end do
          b(imax) = b(imax) / a(imax,k)
          a(imax,k) = 1.0D+00

          do i = 1, n

            if ( ipiv(i) .eq. 0 ) then
              do j = k + 1, n
                a(i,j) = a(i,j) - a(i,k) * a(imax,j)
              end do
              b(i) = b(i) - a(i,k) * b(imax)
              a(i,k) = 0.0D+00
            end if

          end do

        end if

      end do
c
c  Now, every row with nonzero IPIV begins with a 1, and
c  all other rows are all zero.  Begin solution.
c
      do j = n, 1, -1

        imax = 0
        do k = 1, n
          if ( ipiv(k) .eq. j ) then
            imax = k
          end if
        end do

        if ( imax .eq. 0 ) then

          x(j) = 0.0D+00

          if ( b(j) .eq. 0.0D+00 ) then
            ierror = 1
            write ( *, '(a)' ) ' '
            write ( *, '(a)' ) 'R8MAT_SOLVE2 - Warning:'
            write ( *, '(a,i8)' )
     &        '  Consistent singularity, equation = ', j
          else
            ierror = 2
            write ( *, '(a)' ) ' '
            write ( *, '(a)' ) 'R8MAT_SOLVE2 - Error:'
            write ( *, '(a,i8)' )
     &        '  Inconsistent singularity, equation = ', j
          end if

        else

          x(j) = b(imax)

          do i = 1, n
            if ( i .ne. imax ) then
              b(i) = b(i) - a(i,j) * x(j)
            end if
          end do

        end if

      end do

      return
      end
      subroutine r8mat_sub ( m, n, a, b, c )

c*********************************************************************72
c
cc R8MAT_SUB computes the difference of two matrices.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    30 October 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the order of the matrices.
c
c    Input, double precision A(M,N), B(M,N), the matrices.
c
c    Output, double precision C(M,N), the result A - B.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      double precision b(m,n)
      double precision c(m,n)
      integer i
      integer j

      do j = 1, n
        do i = 1, m
          c(i,j) = a(i,j) - b(i,j)
        end do
      end do

      return
      end
      function r8mat_sum ( m, n, a )

c*********************************************************************72
c
cc R8MAT_SUM sums the entries of an R8MAT.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    04 January 2012
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns.
c
c    Input, double precision A(M,N), the array.
c
c    Output, double precision R8MAT_SUM, the sum of the entries.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer j
      double precision r8mat_sum
      double precision value

      value = 0.0D+00
      do j = 1, n
        do i = 1, m
          value = value + a(i,j)
        end do
      end do

      r8mat_sum = value

      return
      end
      subroutine r8mat_symm_eigen ( n, x, q, a )

c*********************************************************************72
c
cc R8MAT_SYMM_EIGEN returns a symmetric matrix with given eigensystem.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c    The user must supply the desired eigenvalue vector, and the desired
c    eigenvector matrix.  The eigenvector matrix must be orthogonal.  A
c    suitable random orthogonal matrix can be generated by R8MAT_ORTH_UNIFORM.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    15 May 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the order of A.
c
c    Input, double precision X(N), the desired eigenvalues for the matrix.
c
c    Input, double precision Q(N,N), the eigenvector matrix of A.
c
c    Output, double precision A(N,N), a symmetric matrix with
c    eigenvalues X and eigenvectors the columns of Q.
c
      implicit none

      integer n

      double precision a(n,n)
      integer i
      integer j
      integer k
      double precision q(n,n)
      double precision x(n)
c
c  Set A = Q * Lambda * Q'.
c
      do i = 1, n
        do j = 1, n
          a(i,j) = 0.0D+00
          do k = 1, n
            a(i,j) = a(i,j) + q(i,k) * x(k) * q(j,k)
          end do
        end do
      end do

      return
      end
      subroutine r8mat_symm_jacobi ( n, a )

c*********************************************************************72
c
cc R8MAT_SYMM_JACOBI applies Jacobi eigenvalue iteration to a symmetric matrix.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c    This code was modified so that it treats as zero the off-diagonal
c    elements that are sufficiently close to, but not exactly, zero.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    22 November 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the order of A.
c
c    Input/output, double precision A(N,N), a symmetric N by N matrix.
c    On output, the matrix has been overwritten by an approximately
c    diagonal matrix, with the eigenvalues on the diagonal.
c
      implicit none

      integer n

      double precision a(n,n)
      double precision c
      double precision r8mat_norm_fro
      double precision eps
      parameter ( eps = 0.00001D+00 )
      integer i
      integer it
      integer it_max
      parameter ( it_max = 100 )
      integer j
      integer k
      double precision norm_fro
      double precision s
      double precision sum2
      double precision t
      double precision t1
      double precision t2
      double precision u

      norm_fro = r8mat_norm_fro ( n, n, a )

      it = 0

10    continue

        it = it + 1

        do i = 1, n
          do j = 1, i - 1

            if ( eps * norm_fro .lt. 
     &           abs ( a(i,j) ) + abs ( a(j,i) ) ) then

              u = ( a(j,j) - a(i,i) ) / ( a(i,j) + a(j,i) )

              t = sign ( 1.0D+00, u ) 
     &          / ( abs ( u ) + sqrt ( u * u + 1.0D+00 ) )
              c = 1.0D+00 / sqrt ( t * t + 1.0D+00 )
              s = t * c
c
c  A -> A * Q.
c
              do k = 1, n
                t1 = a(i,k)
                t2 = a(j,k)
                a(i,k) = t1 * c - t2 * s
                a(j,k) = t1 * s + t2 * c
              end do
c
c  A -> QT * A
c
              do k = 1, n
                t1 = a(k,i)
                t2 = a(k,j)
                a(k,i) = c * t1 - s * t2
                a(k,j) = s * t1 + c * t2
              end do

            end if
          end do
        end do
c
c  Test the size of the off-diagonal elements.
c
        sum2 = 0.0D+00
        do i = 1, n
          do j = 1, i - 1
            sum2 = sum2 + abs ( a(i,j) )
          end do
        end do

        if ( sum2 .le. eps * ( norm_fro + 1.0D+00 ) ) then
          go to 20
        end if

        if ( it_max .le. it ) then
          go to 20
        end if

      go to 10

20    continue

      return
      end
      subroutine r8mat_to_r8cmat ( lda, m, n, a1, a2 )

c*********************************************************************72
c
cc R8MAT_TO_R8CMAT transfers data from an R8MAT to an R8CMAT.
c
c  Discussion:
c
c    An R8MAT is an MxN array of R8's, 
c    accessible as a vector:
c      (I,J) -> (I+J*M).
c    or as a doubly-dimensioned array, if declared A(M,N):
c      (I,J) -> A(I,J)
c      
c    An R8CMAT is an MxN array of R8's, stored with a leading dimension LD,
c    accessible as a vector:
c      (I,J) -> (I+J*LD).
c    or as a doubly-dimensioned array, if declared A(LD,N):
c      (I,J) -> A(I,J)
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    19 March 2014
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer LDA, the leading dimension of A2.
c
c    Input, integer M, the number of rows of data.
c    M <= LDA.
c
c    Input, integer N, the number of columns of data.
c
c    Input, double precision A1(M,N), the matrix to be copied.
c
c    Output, double precision A2(LDA,N), contains a copy of the
c    information in A1, in the MxN submatrix.
c
      implicit none

      integer lda
      integer m
      integer n

      double precision a1(m,n)
      double precision a2(lda,n)
      integer i
      integer j

      do j = 1, n
        do i = 1, m
          a2(i,j) = a1(i,j)
        end do
      end do
 
      return
      end
      subroutine r8mat_to_r8plu ( n, a, pivot, lu, info )

c*********************************************************************72
c
cc R8MAT_TO_R8PLU factors a general R8MAT.
c
c  Discussion:
c
c    The factorization can be written
c    A = P * L * U, where P is a permutation matrix, L a unit lower
c    triangular matrix and U is an upper triangular matrix.
c
c    An R8MAT is an MxN array of R8's, stored by (I,J) -> [I+J*M].
c
c    This routine is a simplified version of the LINPACK routine DGEFA.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    30 March 2013
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Jack Dongarra, Jim Bunch, Cleve Moler, Pete Stewart,
c    LINPACK User's Guide,
c    SIAM, 1979,
c    ISBN13: 978-0-898711-72-1,
c    LC: QA214.L56.
c
c  Parameters:
c
c    Input, integer N, the order of the matrix.
c    N must be positive.
c
c    Input, double precision A(N,N), the matrix to be factored.
c
c    Output, integer PIVOT(N), a vector of pivot indices.
c
c    Output, double precision LU(N,N), an upper triangular matrix and the 
c    multipliers used to obtain it.  
c
c    Output, integer INFO, singularity flag.
c    0, no singularity detected.
c    nonzero, the factorization failed on the INFO-th step.
c 
      implicit none

      integer n

      double precision a(n,n)
      integer i
      integer info
      double precision lu(n,n)
      integer pivot(n)
      integer j
      integer k
      integer l
      double precision t

      do j = 1, n
        do i = 1, n 
          lu(i,j) = a(i,j)
        end do
      end do

      info = 0

      do k = 1, n - 1
c
c  Find L, the index of the pivot row.
c
        l = k
        do i = k + 1, n
          if ( abs ( lu(l,k) ) .lt. abs ( lu(i,k) ) ) then
            l = i
          end if
        end do

        pivot(k) = l
c
c  If the pivot index is zero, the algorithm has failed.
c
        if ( lu(l,k) .eq. 0.0D+00 ) then
          info = k
          write ( *, '(a)' ) ' '
          write ( *, '(a)' ) 'R8MAT_TO_R8PLU - Fatal error!'
          write ( *, '(a,i8)' ) '  Zero pivot on step ', info
          return
        end if
c
c  Interchange rows L and K if necessary.
c
        if ( l .ne. k ) then
          t       = lu(l,k)
          lu(l,k) = lu(k,k)
          lu(k,k) = t
        end if
c
c  Normalize the values that lie below the pivot entry A(K,K).
c
        do i = k + 1, n
          lu(i,k) = - lu(i,k) / lu(k,k)
        end do
c
c  Row elimination with column indexing.
c
        do j = k + 1, n

          if ( l .ne. k ) then
            t       = lu(l,j)
            lu(l,j) = lu(k,j)
            lu(k,j) = t
          end if

          do i = k + 1, n
            lu(i,j) = lu(i,j) + lu(i,k) * lu(k,j)
          end do

        end do

      end do

      pivot(n) = n

      if ( a(n,n) .eq. 0.0D+00 ) then
        info = n
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8MAT_TO_R8PLU - Fatal error!'
        write ( *, '(a,i8)' ) '  Zero pivot on step ', info
      end if

      return
      end
      function r8mat_trace ( n, a )

c*********************************************************************72
c
cc R8MAT_TRACE computes the trace of an R8MAT.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c    The trace of a square matrix is the sum of the diagonal elements.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    15 May 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the order of the matrix A.
c
c    Input, double precision A(N,N), the matrix whose trace is desired.
c
c    Output, double precision R8MAT_TRACE, the trace of the matrix.
c
      implicit none

      integer n

      double precision a(n,n)
      integer i
      double precision r8mat_trace

      r8mat_trace = 0.0D+00
      do i = 1, n
        r8mat_trace = r8mat_trace + a(i,i)
      end do

      return
      end
      subroutine r8mat_transpose ( m, n, a, at )

c*********************************************************************72
c
cc R8MAT_TRANSPOSE makes a transposed copy of a matrix.
c
c  Discussion:
c
c    An R8MAT is an array of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    13 June 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns of the matrix A.
c
c    Input, double precision A(M,N), the matrix to be transposed.
c
c    Output, double precision AT(N,M), the transposed matrix.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      double precision at(n,m)
      integer i
      integer j

      do j = 1, m
        do i = 1, n
          at(i,j) = a(j,i)
        end do
      end do

      return
      end
      subroutine r8mat_transpose_in_place ( n, a )

c*********************************************************************72
c
cc R8MAT_TRANSPOSE_IN_PLACE transposes a square matrix in place.
c
c  Discussion:
c
c    An R8MAT is an array of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    27 June 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of rows and columns of the matrix A.
c
c    Input/output, double precision A(N,N), the matrix to be transposed.
c
      implicit none

      integer n

      double precision a(n,n)
      integer i
      integer j
      double precision t

      do j = 1, n
        do i = 1, j - 1
          t      = a(i,j)
          a(i,j) = a(j,i)
          a(j,i) = t
        end do
      end do

      return
      end
      subroutine r8mat_transpose_print ( m, n, a, title )

c*********************************************************************72
c
cc R8MAT_TRANSPOSE_PRINT prints an R8MAT, transposed.
c
c  Discussion:
c
c    An R8MAT is an array of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    28 April 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns.
c
c    Input, double precision A(M,N), an M by N matrix to be printed.
c
c    Input, character*(*) TITLE, a title.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      character*(*) title

      call r8mat_transpose_print_some ( m, n, a, 1, 1, m, n, title )

      return
      end
      subroutine r8mat_transpose_print_some ( m, n, a, ilo, jlo, ihi,
     &  jhi, title )

c*********************************************************************72
c
cc R8MAT_TRANSPOSE_PRINT_SOME prints some of an R8MAT transposed.
c
c  Discussion:
c
c    An R8MAT is an array of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    28 April 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns.
c
c    Input, double precision A(M,N), an M by N matrix to be printed.
c
c    Input, integer ILO, JLO, the first row and column to print.
c
c    Input, integer IHI, JHI, the last row and column to print.
c
c    Input, character * ( * ) TITLE, a title.
c
      implicit none

      integer incx
      parameter ( incx = 5 )
      integer m
      integer n

      double precision a(m,n)
      character * ( 14 ) ctemp(incx)
      integer i
      integer i2
      integer i2hi
      integer i2lo
      integer ihi
      integer ilo
      integer inc
      integer j
      integer j2hi
      integer j2lo
      integer jhi
      integer jlo
      character * ( * ) title

      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) trim ( title )

      if ( m .le. 0 .or. n .le. 0 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) '  (None)'
        return
      end if

      do i2lo = max ( ilo, 1 ), min ( ihi, m ), incx

        i2hi = i2lo + incx - 1
        i2hi = min ( i2hi, m )
        i2hi = min ( i2hi, ihi )

        inc = i2hi + 1 - i2lo

        write ( *, '(a)' ) ' '

        do i = i2lo, i2hi
          i2 = i + 1 - i2lo
          write ( ctemp(i2), '(i8,6x)') i
        end do

        write ( *, '(''       Row'',5a14)' ) ctemp(1:inc)
        write ( *, '(a)' ) '       Col'

        j2lo = max ( jlo, 1 )
        j2hi = min ( jhi, n )

        do j = j2lo, j2hi

          do i2 = 1, inc
            i = i2lo - 1 + i2
            write ( ctemp(i2), '(g14.6)' ) a(i,j)
          end do

          write ( *, '(2x,i8,a,5a14)' ) j, ':', ( ctemp(i), i = 1, inc )

        end do

      end do

      return
      end
      subroutine r8mat_u_inverse ( n, a, b )

c*********************************************************************72
c
cc R8MAT_U_INVERSE inverts an upper triangular R8MAT.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c    An upper triangular matrix is a matrix whose only nonzero entries
c    occur on or above the diagonal.
c
c    The inverse of an upper triangular matrix is an upper triangular matrix.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    15 May 2010
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Albert Nijenhuis, Herbert Wilf,
c    Combinatorial Algorithms for Computers and Calculators,
c    Academic Press, 1978,
c    ISBN: 0-12-519260-6,
c    LC: QA164.N54.
c
c  Parameters:
c
c    Input, integer N, number of rows and columns in the matrix.
c
c    Input, double precision A(N,N), the upper triangular matrix.
c
c    Output, double precision B(N,N), the inverse matrix.
c
      implicit none

      integer n

      double precision a(n,n)
      double precision b(n,n)
      integer i
      integer j
      integer k

      do j = n, 1, -1

        do i = n, 1, -1

          if ( j .lt. i ) then
            b(i,j) = 0.0D+00
          else if ( i .eq. j ) then
            b(i,j) = 1.0D+00 / a(i,j)
          else
            b(i,j) = 0.0D+00
            do k = i + 1, j
              b(i,j) = b(i,j) - a(i,k) * b(k,j)
            end do
            b(i,j) = b(i,j) / a(i,i)
          end if

        end do
      end do

      return
      end
      subroutine r8mat_u_solve ( n, a, b, x )

c*********************************************************************72
c
cc R8MAT_U_SOLVE solves an upper triangular linear system.
c
c  Discussion:
c
c    An R8MAT is an MxN array of R8's, stored by (I,J) -> [I+J*M].
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    21 October 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of rows and columns of
c    the matrix A.
c
c    Input, double precision A(N,N), the N by N upper triangular matrix.
c
c    Input, double precision B(N), the right hand side of the linear system.
c
c    Output, double precision X(N), the solution of the linear system.
c
      implicit none

      integer n

      double precision a(n,n)
      double precision b(n)
      integer i
      integer j
      double precision x(n)
c
c  Solve U * x = b.
c
      do i = n, 1, -1
        x(i) = b(i)
        do j = i + 1, n
          x(i) = x(i) - a(i,j) * x(j)
        end do
        x(i) = x(i) / a(i,i)
      end do

      return
      end
      subroutine r8mat_u1_inverse ( n, a, b )

c*********************************************************************72
c
cc R8MAT_U1_INVERSE inverts a unit upper triangular R8MAT.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c    A unit upper triangular matrix is a matrix with only 1's on the main
c    diagonal, and only 0's below the main diagonal.
c
c    The inverse of a unit upper triangular matrix is also
c    a unit upper triangular matrix.
c
c    This routine can invert a matrix in place, that is, with no extra
c    storage.  If the matrix is stored in A, then the call
c
c      call r8mat_u1_inverse ( n, a, a )
c
c    will result in A being overwritten by its inverse.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    16 May 2010
c
c  Author:
c
c    Original FORTRAN77 version by Albert Nijenhuis, Herbert Wilf.
c    FORTRAN90 version by John Burkardt.
c
c  Reference:
c
c    Albert Nijenhuis, Herbert Wilf,
c    Combinatorial Algorithms for Computers and Calculators,
c    Academic Press, 1978,
c    ISBN: 0-12-519260-6,
c    LC: QA164.N54.
c
c  Parameters:
c
c    Input, integer N, number of rows and columns in the matrix.
c
c    Input, double precision A(N,N), the unit upper triangular matrix.
c
c    Output, double precision B(N,N), the inverse matrix.
c
      implicit none

      integer n

      double precision a(n,n)
      double precision b(n,n)
      integer i
      integer j
      integer k

      do j = n, 1, -1

        do i = n, 1, -1

          if ( j .lt. i ) then
            b(i,j) = 0.0D+00
          else if ( i .eq. j ) then
            b(i,j) = 1.0D+00
          else
            b(i,j) = 0.0D+00
            do k = i + 1, j
              b(i,j) = b(i,j) - a(i,k) * b(k,j)
            end do
          end if

        end do
      end do

      return
      end
      subroutine r8mat_uniform_01 ( m, n, seed, r )

c*********************************************************************72
c
cc R8MAT_UNIFORM_01 returns a unit pseudorandom R8MAT.
c
c  Discussion:
c
c    An R8MAT is an array of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    11 August 2004
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Paul Bratley, Bennett Fox, Linus Schrage,
c    A Guide to Simulation,
c    Springer Verlag, pages 201-202, 1983.
c
c    Bennett Fox,
c    Algorithm 647:
c    Implementation and Relative Efficiency of Quasirandom
c    Sequence Generators,
c    ACM Transactions on Mathematical Software,
c    Volume 12, Number 4, pages 362-376, 1986.
c
c    Peter Lewis, Allen Goodman, James Miller,
c    A Pseudo-Random Number Generator for the System/360,
c    IBM Systems Journal,
c    Volume 8, pages 136-143, 1969.
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns in the array.
c
c    Input/output, integer SEED, the "seed" value, which should NOT be 0.
c    On output, SEED has been updated.
c
c    Output, double precision R(M,N), the array of pseudorandom values.
c
      implicit none

      integer m
      integer n

      integer i
      integer i4_huge
      parameter ( i4_huge = 2147483647 )
      integer j
      integer k
      integer seed
      double precision r(m,n)

      if ( seed .eq. 0 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8MAT_UNIFORM_01 - Fatal error!'
        write ( *, '(a)' ) '  Input value of SEED = 0.'
        stop 1
      end if

      do j = 1, n

        do i = 1, m

          k = seed / 127773

          seed = 16807 * ( seed - k * 127773 ) - k * 2836

          if ( seed .lt. 0 ) then
            seed = seed + i4_huge
          end if

          r(i,j) = dble ( seed ) * 4.656612875D-10

        end do
      end do

      return
      end
      subroutine r8mat_uniform_ab ( m, n, a, b, seed, r )

c*********************************************************************72
c
cc R8MAT_UNIFORM_AB returns a scaled pseudorandom R8MAT.
c
c  Discussion:
c
c    A <= R(I,J) <= B.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    05 February 2005
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Paul Bratley, Bennett Fox, Linus Schrage,
c    A Guide to Simulation,
c    Second Edition,
c    Springer, 1987,
c    ISBN: 0387964673,
c    LC: QA76.9.C65.B73.
c
c    Bennett Fox,
c    Algorithm 647:
c    Implementation and Relative Efficiency of Quasirandom
c    Sequence Generators,
c    ACM Transactions on Mathematical Software,
c    Volume 12, Number 4, December 1986, pages 362-376.
c
c    Pierre L'Ecuyer,
c    Random Number Generation,
c    in Handbook of Simulation,
c    edited by Jerry Banks,
c    Wiley, 1998,
c    ISBN: 0471134031,
c    LC: T57.62.H37.
c
c    Peter Lewis, Allen Goodman, James Miller,
c    A Pseudo-Random Number Generator for the System/360,
c    IBM Systems Journal,
c    Volume 8, Number 2, 1969, pages 136-143.
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns in the array.
c
c    Input, double precision A, B, the lower and upper limits.
c
c    Input/output, integer SEED, the "seed" value, which should NOT be 0.
c    On output, SEED has been updated.
c
c    Output, double precision R(M,N), the array of pseudorandom values.
c
      implicit none

      integer m
      integer n

      double precision a
      double precision b
      integer i
      integer i4_huge
      parameter ( i4_huge = 2147483647 )
      integer j
      integer k
      integer seed
      double precision r(m,n)

      if ( seed .eq. 0 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8MAT_UNIFORM_AB - Fatal error!'
        write ( *, '(a)' ) '  Input value of SEED = 0.'
        stop 1
      end if

      do j = 1, n

        do i = 1, m

          k = seed / 127773

          seed = 16807 * ( seed - k * 127773 ) - k * 2836

          if ( seed .lt. 0 ) then
            seed = seed + i4_huge
          end if

          r(i,j) = a + ( b - a ) * dble ( seed ) * 4.656612875D-10

        end do
      end do

      return
      end
      subroutine r8mat_uniform_abvec ( m, n, a, b, seed, r )

c*********************************************************************72
c
cc R8MAT_UNIFORM_ABVEC returns a scaled pseudorandom R8MAT.
c
c  Discussion:
c
c    A(I) <= R(I,J) <= B(I).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    02 October 2012
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Paul Bratley, Bennett Fox, Linus Schrage,
c    A Guide to Simulation,
c    Second Edition,
c    Springer, 1987,
c    ISBN: 0387964673,
c    LC: QA76.9.C65.B73.
c
c    Bennett Fox,
c    Algorithm 647:
c    Implementation and Relative Efficiency of Quasirandom
c    Sequence Generators,
c    ACM Transactions on Mathematical Software,
c    Volume 12, Number 4, December 1986, pages 362-376.
c
c    Pierre L'Ecuyer,
c    Random Number Generation,
c    in Handbook of Simulation,
c    edited by Jerry Banks,
c    Wiley, 1998,
c    ISBN: 0471134031,
c    LC: T57.62.H37.
c
c    Peter Lewis, Allen Goodman, James Miller,
c    A Pseudo-Random Number Generator for the System/360,
c    IBM Systems Journal,
c    Volume 8, Number 2, 1969, pages 136-143.
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns in the array.
c
c    Input, double precision A(M), B(M), the lower and upper limits.
c
c    Input/output, integer SEED, the "seed" value, which should NOT be 0.
c    On output, SEED has been updated.
c
c    Output, double precision R(M,N), the array of pseudorandom values.
c
      implicit none

      integer m
      integer n

      double precision a(m)
      double precision b(m)
      integer i
      integer i4_huge
      parameter ( i4_huge = 2147483647 )
      integer j
      integer k
      integer seed
      double precision r(m,n)

      if ( seed .eq. 0 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8MAT_UNIFORM_ABVEC - Fatal error!'
        write ( *, '(a)' ) '  Input value of SEED = 0.'
        stop 1
      end if

      do j = 1, n

        do i = 1, m

          k = seed / 127773

          seed = 16807 * ( seed - k * 127773 ) - k * 2836

          if ( seed .lt. 0 ) then
            seed = seed + i4_huge
          end if

          r(i,j) = a(i) + ( b(i) - a(i) ) * dble ( seed ) 
     &      * 4.656612875D-10

        end do
      end do

      return
      end
      subroutine r8mat_ut_solve ( n, a, b, x )

c*********************************************************************72
c
cc R8MAT_UT_SOLVE solves a transposed upper triangular linear system.
c
c  Discussion:
c
c    An R8MAT is an MxN array of R8's, stored by (I,J) -> [I+J*M].
c
c    Given the upper triangular matrix A, the linear system to be solved is:
c
c      A' * x = b
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    22 October 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of rows and columns
c    of the matrix.
c
c    Input, double precision A(N,N), the N by N upper triangular matrix.
c
c    Input, double precision B(N), the right hand side of the linear system.
c
c    Output, double precision X(N), the solution of the linear system.
c
      implicit none

      integer n

      double precision a(n,n)
      double precision b(n)
      integer i
      integer j
      double precision x(n)
c
c  Solve U' * x = b.
c
      do i = 1, n
        x(i) = b(i)
        do j = 1, i - 1
          x(i) = x(i) - a(j,i) * x(j)
        end do
        x(i) = x(i) / a(i,i)
      end do

      return
      end
      subroutine r8mat_vand2 ( n, x, a )

c*********************************************************************72
c
cc R8MAT_VAND2 returns the N by N row Vandermonde matrix A.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c    The row Vandermonde matrix returned by this routine reads "across"
c    rather than down.  In particular, each row begins with a 1, followed by
c    some value X, followed by successive powers of X.
c
c  Formula:
c
c    A(I,J) = X(I)^(J-1)
c
c  Properties:
c
c    A is nonsingular if, and only if, the X values are distinct.
c
c    The determinant of A is
c
c      det(A) = product ( 2 <= I <= N ) (
c        product ( 1 <= J <= I-1 ) ( ( X(I) - X(J) ) ) ).
c
c    The matrix A is generally ill-conditioned.
c
c  Example:
c
c    N = 5, X = (2, 3, 4, 5, 6)
c
c    1 2  4   8   16
c    1 3  9  27   81
c    1 4 16  64  256
c    1 5 25 125  625
c    1 6 36 216 1296
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    16 May 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the order of the matrix desired.
c
c    Input, double precision X(N), the values that define A.
c
c    Output, double precision A(N,N), the N by N row Vandermonde matrix.
c
      implicit none

      integer n

      double precision a(n,n)
      integer i
      integer j
      double precision x(n)

      do i = 1, n
        do j = 1, n

          if ( j .eq. 1 .and. x(i) .eq. 0.0D+00 ) then
            a(i,j) = 1.0D+00
          else
            a(i,j) = x(i) ** ( j - 1 )
          end if

        end do
      end do

      return
      end
      function r8mat_vtmv ( m, n, x, a, y )

c*********************************************************************72
c
cc R8MAT_VTMV multiplies computes the scalar x' * A * y.
c
c  Discussion:
c
c    An R8MAT is an MxN array of R8's, stored by (I,J) -> [I+J*M].
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    10 June 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns of
c    the matrix.
c
c    Input, double precision X(N), the first vector factor.
c
c    Input, double precision A(M,N), the M by N matrix.
c
c    Input, double precision Y(M), the second vector factor.
c
c    Output, double precision R8MAT_VTMV, the value of X' * A * Y.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer j
      double precision r8mat_vtmv
      double precision vtmv
      double precision x(m)
      double precision y(n)

      vtmv = 0.0D+00
      do j = 1, n
        do i = 1, m
          vtmv = vtmv + x(i) * a(i,j) * y(j)
        end do
      end do

      r8mat_vtmv = vtmv

      return
      end
      subroutine r8mat_zero ( m, n, a )

c*********************************************************************72
c
cc R8MAT_ZERO zeroes an R8MAT.
c
c  Discussion:
c
c    An R8MAT is an array of R8 values.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    18 July 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns.
c
c    Output, double precision A(M,N), the matrix of zeroes.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer j

      do j = 1, n
        do i = 1, m
          a(i,j) = 0.0D+00
        end do
      end do

      return
      end
      subroutine r8plu_det ( n, pivot, lu, det )

c*********************************************************************72
c
cc R8PLU_DET computes the determinant of an R8PLU matrix.
c
c  Discussion:
c
c    The matrix should have been factored by R8MAT_TO_R8PLU.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    16 May 2010
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Jack Dongarra, Jim Bunch, Cleve Moler, Pete Stewart,
c    LINPACK User's Guide,
c    SIAM, 1979,
c    ISBN13: 978-0-898711-72-1.
c
c  Parameters:
c
c    Input, integer N, the order of the matrix.
c    N must be positive.
c
c    Input, integer PIVOT(N), the pivot vector computed
c    by R8MAT_TO_R8PLU.
c
c    Input, double precision LU(N,N), the LU factors computed
c    by R8MAT_TO_R8PLU.
c
c    Output, double precision DET, the determinant of the matrix.
c
      implicit none

      integer n

      double precision det
      integer i
      double precision lu(n,n)
      integer pivot(n)

      det = 1.0D+00

      do i = 1, n
        det = det * lu(i,i)
        if ( pivot(i) .ne. i ) then
          det = - det
        end if
      end do

      return
      end
      subroutine r8plu_inverse ( n, pivot, lu, a_inverse )

c*********************************************************************72
c
cc R8PLU_INVERSE computes the inverse of an R8PLU matrix.
c
c  Discussion:
c
c    The matrix should have been factored by R8MAT_TO_R8PLU.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    30 March 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the order of the matrix A.
c
c    Input, integer PIVOT(N), the pivot vector from
c    R8MAT_TO_R8PLU.
c
c    Input, double precision LU(N,N), the LU factors computed by
c    R8MAT_TO_R8PLU.
c
c    Output, double precision A_INVERSE(N,N), the inverse of the original
c    matrix A that was factored by R8MAT_TO_R8PLU.
c
      implicit none

      integer n

      double precision a_inverse(n,n)
      integer i
      double precision lu(n,n)
      integer pivot(n)
      integer j
      integer k
      double precision temp
      double precision work(n)

      do j = 1, n
        do i = 1, n
          a_inverse(i,j) = lu(i,j)
        end do
      end do
c
c  Compute Inverse(U).
c
      do k = 1, n

        a_inverse(k,k)     = 1.0D+00 / a_inverse(k,k)
        do i = 1, k - 1
          a_inverse(i,k) = -a_inverse(i,k) * a_inverse(k,k)
        end do

        do j = k + 1, n

          temp             = a_inverse(k,j)
          a_inverse(k,j)   = 0.0D+00
          do i = 1, k
            a_inverse(i,j) = a_inverse(i,j) + temp * a_inverse(i,k)
          end do

        end do

      end do
c
c  Form Inverse(U) * Inverse(L).
c
      do k = n - 1, 1, -1

        do i = k + 1, n
          work(i) = a_inverse(i,k)
          a_inverse(i,k) = 0.0D+00
        end do

        do j = k + 1, n
          do i = 1, n
            a_inverse(i,k) = a_inverse(i,k) + a_inverse(i,j) * work(j)
          end do
        end do

        if ( pivot(k) .ne. k ) then

          do i = 1, n
            temp                  = a_inverse(i,k)
            a_inverse(i,k)        = a_inverse(i,pivot(k))
            a_inverse(i,pivot(k)) = temp
          end do

        end if

      end do

      return
      end
      subroutine r8plu_mul ( n, pivot, lu, x, b )

c*********************************************************************72
c
cc R8PLU_MUL computes A * x using the PLU factors of A.
c
c  Discussion:
c
c    It is assumed that R8MAT_TO_R8PLU has computed the PLU factors of
c    the matrix A.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    30 March 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the order of the matrix.
c    N must be positive.
c
c    Input, integer PIVOT(N), the pivot vector computed
c    by R8MAT_TO_R8PLU.
c
c    Input, double precision LU(N,N), the matrix factors computed by
c    R8MAT_TO_R8PLU.
c
c    Input, double precision X(N), the vector to be multiplied.
c
c    Output, double precision B(N), the result of the multiplication.
c
      implicit none

      integer n

      double precision b(n)
      integer i
      integer j
      integer k
      double precision lu(n,n)
      integer pivot(n)
      double precision temp
      double precision x(n)

      do i = 1, n
        b(i) = x(i)
      end do
c
c  Y = U * X.
c
      do j = 1, n
        do i = 1, j - 1
          b(i) = b(i) + lu(i,j) * b(j)
        end do
        b(j) = lu(j,j) * b(j)
      end do
c
c  B = PL * Y = PL * U * X = A * x.
c
      do j = n - 1, 1, -1

        do i = j + 1, n
          b(i) = b(i) - lu(i,j) * b(j)
        end do

        k = pivot(j)

        if ( k .ne. j ) then
          temp = b(k)
          b(k) = b(j)
          b(j) = temp
        end if

      end do

      return
      end
      subroutine r8plu_sol ( n, pivot, lu, b, x )

c*********************************************************************72
c
cc R8PLU_SOL solves a linear system A*x=b from the PLU factors.
c
c  Discussion:
c
c    The PLU factors should have been computed by R8MAT_TO_R8PLU.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    30 March 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the order of the matrix.
c
c    Input, integer PIVOT(N), the pivot vector from R8MAT_TO_R8PLU.
c
c    Input, double precision LU(N,N), the LU factors from R8MAT_TO_R8PLU.
c
c    Input, double precision B(N), the right hand side vector.
c
c    Output, double precision X(N), the solution vector.
c
      implicit none

      integer n

      double precision b(n)
      integer pivot(n)
      integer i
      integer j
      integer k
      double precision lu(n,n)
      double precision temp
      double precision x(n)
c
c  Solve PL * Y = B.
c
      do i = 1, n
        x(i) = b(i)
      end do

      do k = 1, n - 1

        j = pivot(k)

        if ( j .ne. k ) then
          temp = x(j)
          x(j) = x(k)
          x(k) = temp
        end if

        do i = k + 1, n
          x(i) = x(i) + lu(i,k) * x(k)
        end do

      end do
c
c  Solve U * X = Y.
c
      do k = n, 1, -1
        x(k) = x(k) / lu(k,k)
        do i = 1, k - 1
          x(i) = x(i) - lu(i,k) * x(k)
        end do
      end do

      return
      end
      subroutine r8plu_to_r8mat ( n, pivot, lu, a )

c*********************************************************************72
c
cc R8PLU_TO_R8MAT recovers the matrix A that was factored by R8MAT_TO_R8PLU.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    30 March 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the order of the matrix.
c    N must be positive.
c
c    Input, integer PIVOT(N), the pivot vector computed
c    by R8MAT_TO_R8PLU.
c
c    Input, double precision LU(N,N), the matrix factors computed by
c    R8MAT_TO_R8PLU.
c
c    Output, double precision A(N,N), the matrix whose factors are
c    represented by LU and PIVOT.
c
      implicit none

      integer n

      double precision a(n,n)
      integer i
      double precision lu(n,n)
      integer pivot(n)
      integer j
      integer k
      double precision temp

      do j = 1, n
        do i = 1, n
          a(i,j) = 0.0D+00
        end do
      end do

      do i = 1, n
        a(i,i) = 1.0D+00
      end do

      do j = 1, n

        do i = 1, n
          do k = 1, i - 1
            a(k,j) = a(k,j) + lu(k,i) * a(i,j)
          end do
          a(i,j) = lu(i,i) * a(i,j)
        end do
c
c  B = PL * Y = PL * U * X = A * x.
c
        do i = n - 1, 1, -1

          do k = i + 1, n
            a(k,j) = a(k,j) - lu(k,i) * a(i,j)
          end do

          k = pivot(i)

          if ( k .ne. i ) then
            temp   = a(k,j)
            a(k,j) = a(i,j)
            a(i,j) = temp
          end if

        end do

      end do

      return
      end
      function r8poly_degree ( na, a )

c*********************************************************************72
c
cc R8POLY_DEGREE returns the degree of a polynomial.
c
c  Discussion:
c
c    The degree of a polynomial is the index of the highest power
c    of X with a nonzero coefficient.
c
c    The degree of a constant polynomial is 0.  The degree of the
c    zero polynomial is debatable, but this routine returns the
c    degree as 0.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    05 January 2015
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer NA, the dimension of A.
c
c    Input, double precision A(0:NA), the coefficients of the polynomials.
c
c    Output, integer R8POLY_DEGREE, the degree of A.
c
      implicit none

      integer na

      double precision a(0:na)
      integer r8poly_degree
      integer value

      value = na

10    continue

      if ( 0 .lt. value ) then

        if ( a(value) .ne. 0.0D+00 ) then
          go to 20
        end if

        value = value - 1

        go to 10

      end if

20    continue

      r8poly_degree = value

      return
      end
      subroutine r8poly_deriv ( n, c, p, cp )

c*********************************************************************72
c
cc R8POLY_DERIV returns the derivative of a polynomial.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    28 March 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the degree of the polynomial.
c
c    Input, double precision C(0:N), the polynomial coefficients.
c    C(I) is the coefficient of X^I.
c
c    Input, integer P, the order of the derivative.
c    0 means no derivative is taken.
c    1 means first derivative,
c    2 means second derivative and so on.
c    Values of P less than 0 are meaningless.  Values of P greater
c    than N are meaningful, but the code will behave as though the
c    value of P was N+1.
c
c    Output, double precision CP(0:N-P), the polynomial coefficients of
c    the derivative.
c
      implicit none

      integer n

      double precision c(0:n)
      double precision cp(0:*)
      double precision cp_temp(0:n)
      integer d
      integer i
      integer p

      if ( n .lt. p ) then
        return
      end if

      do i = 0, n
        cp_temp(i) = c(i)
      end do

      do d = 1, p
        do i = 0, n - d
          cp_temp(i) = dble ( i + 1 ) * cp_temp(i+1)
        end do
        cp_temp(n-d+1) = 0.0D+00
      end do

      do i = 0, n - p
        cp(i) = cp_temp(i)
      end do

      return
      end
      subroutine r8poly_lagrange_0 ( npol, xpol, xval, wval )

c*********************************************************************72
c
cc R8POLY_LAGRANGE_0 evaluates the Lagrange factor at a point.
c
c  Formula:
c
c    W(X) = Product ( 1 <= I <= NPOL ) ( X - XPOL(I) )
c
c  Discussion:
c
c    For a set of points XPOL(I), 1 <= I <= NPOL, the IPOL-th Lagrange basis
c    polynomial L(IPOL)(X), has the property:
c
c      L(IPOL)( XPOL(J) ) = delta ( IPOL, J )
c
c    and may be expressed as:
c
c      L(IPOL)(X) = W(X) / ( ( X - XPOL(IPOL) ) * W'(XPOL(IPOL)) )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    30 March 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer NPOL, the number of abscissas.
c    NPOL must be at least 1.
c
c    Input, double precision XPOL(NPOL), the abscissas, which
c    should be distinct.
c
c    Input, double precision XVAL, the point at which the Lagrange
c    factor is to be evaluated.
c
c    Output, double precision WVAL, the value of the Lagrange factor at XVAL.
c
      implicit none

      integer npol

      integer i
      double precision wval
      double precision xpol(npol)
      double precision xval

      wval = 1.0D+00
      do i = 1, npol
        wval = wval * ( xval - xpol(i) )
      end do

      return
      end
      subroutine r8poly_lagrange_1 ( npol, xpol, xval, dwdx )

c*********************************************************************72
c
cc R8POLY_LAGRANGE_1 evaluates the first derivative of the Lagrange factor.
c
c  Formula:
c
c    W(XPOL(1:NPOL))(X) = Product ( 1 <= I <= NPOL ) ( X - XPOL(I) )
c
c    W'(XPOL(1:NPOL))(X)
c      = Sum ( 1 <= J <= NPOL ) Product ( I /= J ) ( X - XPOL(I) )
c
c    We also have the recursion:
c
c      W'(XPOL(1:NPOL))(X) = d/dX ( ( X - XPOL(NPOL) ) * W(XPOL(1:NPOL-1))(X) )
c                    = W(XPOL(1:NPOL-1))(X)
c                    + ( X - XPOL(NPOL) ) * W'(XPOL(1:NPOL-1))(X)
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    31 March 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer NPOL, the number of abscissas.
c
c    Input, double precision XPOL(NPOL), the abscissas, which should
c    be distinct.
c
c    Input, double precision XVAL, the point at which the Lagrange
c    factor is to be evaluated.
c
c    Output, double precision DWDX, the derivative of W with respect to X.
c
      implicit none

      integer npol

      double precision dwdx
      integer i
      double precision w
      double precision xpol(npol)
      double precision xval

      dwdx = 0.0D+00
      w = 1.0D+00

      do i = 1, npol

        dwdx = w + ( xval - xpol(i) ) * dwdx
        w = w * ( xval - xpol(i) )

      end do

      return
      end
      subroutine r8poly_lagrange_2 ( npol, xpol, xval, dw2dx2 )

c*********************************************************************72
c
cc R8POLY_LAGRANGE_2 evaluates the second derivative of the Lagrange factor.
c
c  Formula:
c
c    W(X)  = Product ( 1 <= I <= NPOL ) ( X - XPOL(I) )
c
c    W'(X) = Sum ( 1 <= J <= NPOL )
c            Product ( I /= J ) ( X - XPOL(I) )
c
c    W"(X) = Sum ( 1 <= K <= NPOL )
c            Sum ( J =/ K )
c            Product ( I /= K, J ) ( X - XPOL(I) )
c
c    For a set of points XPOL(I), 1 <= I <= NPOL, the IPOL-th Lagrange basis
c    polynomial L(IPOL)(X), has the property:
c
c      L(IPOL)( XPOL(J) ) = delta ( IPOL, J )
c
c    and may be expressed as:
c
c      L(IPOL)(X) = W(X) / ( ( X - XPOL(IPOL) ) * W'(XPOL(IPOL)) )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    31 March 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer NPOL, the number of abscissas.
c    NPOL must be at least 1.
c
c    Input, double precision XPOL(NPOL), the abscissas, which should
c    be distinct.
c
c    Input, double precision XVAL, the point at which the Lagrange
c    factor is to be evaluated.
c
c    Output, double precision DW2DX2, the second derivative of W
c    with respect to XVAL.
c
      implicit none

      integer npol

      double precision dw2dx2
      integer i
      integer j
      integer k
      double precision term
      double precision xpol(npol)
      double precision xval

      dw2dx2 = 0.0D+00

      do k = 1, npol

        do j = 1, npol

          if ( j .ne. k ) then
            term = 1.0D+00

            do i = 1, npol
              if ( i .ne. j .and. i .ne. k ) then
                term = term * ( xval - xpol(i) )
              end if
            end do

            dw2dx2 = dw2dx2 + term

          end if

        end do

      end do

      return
      end
      subroutine r8poly_lagrange_coef ( npol, ipol, xpol, pcof )

c*********************************************************************72
c
cc R8POLY_LAGRANGE_COEF returns the coefficients of a Lagrange polynomial.
c
c  Discussion:
c
c    Given distinct abscissas XPOL(1:NPOL), the IPOL-th Lagrange
c    polynomial L(IPOL)(X) is defined as the polynomial of degree
c    NPOL - 1 which is 1 at XPOL(IPOL) and 0 at the NPOL - 1 other
c    abscissas.
c
c    A formal representation is:
c
c      L(IPOL)(X) = Product ( 1 <= I <= NPOL, I /= IPOL )
c       ( X - X(I) ) / ( X(IPOL) - X(I) )
c
c    However sometimes it is desirable to be able to write down
c    the standard polynomial coefficients of L(IPOL)(X).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    28 March 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer NPOL, the number of abscissas.
c    NPOL must be at least 1.
c
c    Input, integer IPOL, the index of the polynomial to evaluate.
c    IPOL must be between 1 and NPOL.
c
c    Input, double precision XPOL(NPOL), the abscissas of the
c    Lagrange polynomials.  The entries in XPOL must be distinct.
c
c    Output, double precision PCOF(0:NPOL-1), the standard polynomial
c    coefficients of the IPOL-th Lagrange polynomial:
c      L(IPOL)(X) = SUM ( 0 <= I <= NPOL-1 ) PCOF(I) * X^I
c
      implicit none

      integer npol

      integer i
      integer indx
      integer ipol
      integer j
      double precision pcof(0:npol-1)
      logical r8vec_distinct
      double precision xpol(npol)
c
c  Make sure IPOL is legal.
c
      if ( ipol .lt. 1 .or. npol .lt. ipol ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8POLY_LAGRANGE_COEF - Fatal error!'
        write ( *, '(a)' ) '  1 <= IPOL <= NPOL is required.'
        write ( *, '(a,i8)' ) '  IPOL = ', ipol
        write ( *, '(a,i8)' ) '  NPOL = ', npol
        stop 1
      end if
c
c  Check that the abscissas are distinct.
c
      if ( .not. r8vec_distinct ( npol, xpol ) ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8POLY_LAGRANGE_COEF - Fatal error!'
        write ( *, '(a)' ) '  Two or more entries of XPOL are equal:'
        stop 1
      end if

      pcof(0) = 1.0D+00
      do i = 1, npol - 1
        pcof(i) = 0.0D+00
      end do

      indx = 0

      do i = 1, npol

        if ( i .ne. ipol ) then

          indx = indx + 1

          do j = indx, 0, -1

            pcof(j) = -xpol(i) * pcof(j) / ( xpol(ipol) - xpol(i) )

            if ( 0 .lt. j ) then
              pcof(j) = pcof(j) + pcof(j-1) / ( xpol(ipol) - xpol(i) )
            end if

          end do

        end if

      end do

      return
      end
      subroutine r8poly_lagrange_factor ( npol, xpol, xval, wval, dwdx )

c*********************************************************************72
c
cc R8POLY_LAGRANGE_FACTOR evaluates the polynomial Lagrange factor at a point.
c
c  Formula:
c
c    W(X) = Product ( 1 <= I <= NPOL ) ( X - XPOL(I) )
c
c  Discussion:
c
c    Suppose F(X) is at least N times continuously differentiable in the
c    interval [A,B].  Pick NPOL distinct points XPOL(I) in [A,B] and compute
c    the interpolating polynomial P(X) of order NPOL ( and degree NPOL-1)
c    which passes through all the points ( XPOL(I), F(XPOL(I)) ).
c    Then in the interval [A,B], the maximum error
c
c      abs ( F(X) - P(X) )
c
c    is bounded by:
c
c      C * FNMAX * W(X)
c
c    where
c
c      C is a constant,
c      FNMAX is the maximum value of the NPOL-th derivative of F in [A,B],
c      W(X) is the Lagrange factor.
c
c    Thus, the value of W(X) is useful as part of an estimated bound
c    for the interpolation error.
c
c    Note that the Chebyshev abscissas have the property that they minimize
c    the value of W(X) over the interval [A,B].  Hence, if the abscissas may
c    be chosen arbitrarily, the Chebyshev abscissas have this advantage over
c    other choices.
c
c    For a set of points XPOL(I), 1 <= I <= NPOL, the IPOL-th Lagrange basis
c    polynomial L(IPOL)(X), has the property:
c
c      L(IPOL)( XPOL(J) ) = delta ( IPOL, J )
c
c    and may be expressed as:
c
c      L(IPOL)(X) = W(X) / ( ( X - XPOL(IPOL) ) * W'(XPOL(IPOL)) )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    30 March 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer NPOL, the number of abscissas.
c    NPOL must be at least 1.
c
c    Input, double precision XPOL(NPOL), the abscissas, which should
c    be distinct.
c
c    Input, double precision XVAL, the point at which the Lagrange
c    factor is to be evaluated.
c
c    Output, double precision WVAL, the value of the Lagrange factor at XVAL.
c
c    Output, double precision DWDX, the derivative of W with respect to XVAL.
c
      implicit none

      integer npol

      double precision dwdx
      integer i
      integer j
      double precision term
      double precision wval
      double precision xpol(npol)
      double precision xval

      wval = 1.0D+00
      do i = 1, npol
        wval = wval * ( xval - xpol(i) )
      end do

      dwdx = 0.0D+00

      do i = 1, npol

        term = 1.0D+00

        do j = 1, npol
          if ( i .ne. j ) then
            term = term * ( xval - xpol(j) )
          end if
        end do

        dwdx = dwdx + term

      end do

      return
      end
      subroutine r8poly_lagrange_val ( npol, ipol, xpol, xval, pval, 
     &  dpdx )

c*********************************************************************72
c
cc R8POLY_LAGRANGE_VAL evaluates the IPOL-th Lagrange polynomial.
c
c  Discussion:
c
c    Given NPOL distinct abscissas, XPOL(1:NPOL), the IPOL-th Lagrange
c    polynomial L(IPOL)(X) is defined as the polynomial of degree
c    NPOL - 1 which is 1 at XPOL(IPOL) and 0 at the NPOL - 1 other
c    abscissas.
c
c    A formal representation is:
c
c      L(IPOL)(X) = Product ( 1 <= I <= NPOL, I /= IPOL )
c       ( X - X(I) ) / ( X(IPOL) - X(I) )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    31 March 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer NPOL, the number of abscissas.
c    NPOL must be at least 1.
c
c    Input, integer IPOL, the index of the polynomial to evaluate.
c    IPOL must be between 1 and NPOL.
c
c    Input, double precision XPOL(NPOL), the abscissas of the Lagrange
c    polynomials.  The entries in XPOL must be distinct.
c
c    Input, double precision XVAL, the point at which the IPOL-th
c    Lagrange polynomial is to be evaluated.
c
c    Output, double precision PVAL, the value of the IPOL-th Lagrange
c    polynomial at XVAL.
c
c    Output, double precision DPDX, the derivative of the IPOL-th
c    Lagrange polynomial at XVAL.
c
      implicit none

      integer ( kind = 4 ) npol

      double precision dpdx
      integer i
      integer ipol
      integer j
      double precision p2
      double precision pval
      logical r8vec_distinct
      double precision xpol(npol)
      double precision xval
c
c  Make sure IPOL is legal.
c
      if ( ipol .lt. 1 .or. npol .lt. ipol ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8POLY_LAGRANGE_VAL - Fatal error!'
        write ( *, '(a)' ) '  1 <= IPOL <= NPOL is required.'
        write ( *, '(a,i8)' ) '  IPOL = ', ipol
        stop 1
      end if
c
c  Check that the abscissas are distinct.
c
      if ( .not. r8vec_distinct ( npol, xpol ) ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8POLY_LAGRANGE_VAL - Fatal error!'
        write ( *, '(a)' ) '  Two or more entries of XPOL are equal:'
        stop 1
      end if
c
c  Evaluate the polynomial.
c
      pval = 1.0D+00

      do i = 1, npol

        if ( i .ne. ipol ) then

          pval = pval * ( xval - xpol(i) ) / ( xpol(ipol) - xpol(i) )

        end if

      end do
c
c  Evaluate the derivative, which can be found by summing up the result
c  of differentiating one factor at a time, successively.
c
      dpdx = 0.0D+00

      do i = 1, npol

        if ( i .ne. ipol ) then

          p2 = 1.0D+00
          do j = 1, npol

            if ( j .eq. i ) then
              p2 = p2                      / ( xpol(ipol) - xpol(j) )
            else if ( j .ne. ipol ) then
              p2 = p2 * ( xval - xpol(j) ) / ( xpol(ipol) - xpol(j) )
            end if

          end do

          dpdx = dpdx + p2

        end if

      end do

      return
      end
      subroutine r8poly_order ( na, a, order )

c*********************************************************************72
c
cc R8POLY_ORDER returns the order of a polynomial.
c
c  Discussion:
c
c    The order of a polynomial is one more than the degree.
c
c    The order of a constant polynomial is 1.  The order of the
c    zero polynomial is debatable, but this routine returns the
c    order as 1.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    02 December 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer NA, the dimension of A.
c
c    Input, double precision A(0:NA), the coefficients of the polynomials.
c
c    Output, integer ORDER, the order of A.
c
      implicit none

      integer na

      double precision a(0:na)
      integer order

      order = na + 1

10    continue

      if ( 1 .lt. order ) then

        if ( a(order-1) .ne. 0.0D+00 ) then
          return
        end if

        order = order - 1

        go to 10

      end if

      return
      end
      subroutine r8poly_print ( n, a, title )

c*********************************************************************72
c
cc R8POLY_PRINT prints out a polynomial.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    26 February 2015
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the dimension of A.
c
c    Input, double precision A(0:N), the polynomial coefficients.
c    A(0) is the constant term and
c    A(N) is the coefficient of X**N.
c
c    Input, character * ( * ) TITLE, a title.
c
      implicit none

      integer n

      double precision a(0:n)
      integer i
      double precision mag
      character plus_minus
      character * ( * ) title

      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) trim ( title )
      write ( *, '(a)' ) ' '

      if ( n .le. 0 ) then
        write ( *, '( ''  p(x) = 0'' )' )
        return
      end if

      if ( a(n) .lt. 0.0D+00 ) then
        plus_minus = '-'
      else
        plus_minus = ' '
      end if

      mag = abs ( a(n) )

      if ( 2 .le. n ) then
        write ( *, '( ''  p(x) = '', a1, g14.6, '' * x ^ '', i3 )' )
     &    plus_minus, mag, n
      else if ( n .eq. 1 ) then
        write ( *, '( ''  p(x) = '', a1, g14.6, '' * x'' )' )
     &    plus_minus, mag
      else if ( n .eq. 0 ) then
        write ( *, '( ''  p(x) = '', a1, g14.6 )' ) plus_minus, mag
      end if

      do i = n - 1, 0, -1

        if ( a(i) .lt. 0.0D+00 ) then
          plus_minus = '-'
        else
          plus_minus = '+'
        end if

        mag = abs ( a(i) )

        if ( mag .ne. 0.0D+00 ) then

          if ( 2 .le. i ) then
            write ( *,
     &        ' ( ''         '', a1, g14.6, '' * x ^ '', i3 )' )
     &        plus_minus, mag, i
          else if ( i .eq. 1 ) then
            write ( *,
     &        ' ( ''         '', a1, g14.6, '' * x'' )' )
     &        plus_minus, mag
          else if ( i .eq. 0 ) then
            write ( *, ' ( ''         '', a1, g14.6 )' )
     &        plus_minus, mag
          end if
        end if

      end do

      return
      end
      subroutine r8poly_shift ( scale, shift, n, poly_cof )

c*********************************************************************72
c
cc R8POLY_SHIFT adjusts the coefficients of a polynomial for a new argument.
c
c  Discussion:
c
c    Assuming P(X) is a polynomial in the argument X, of the form:
c
c      P(X) =
c          C(N) * X^N
c        + ...
c        + C(1) * X
c        + C(0),
c
c    and that Z is related to X by the formula:
c
c      Z = SCALE * X + SHIFT
c
c    then this routine computes coefficients C for the polynomial Q(Z):
c
c      Q(Z) =
c          C(N) * Z^N
c        + ...
c        + C(1) * Z
c        + C(0)
c
c    so that:
c
c      Q(Z(X)) = P(X)
c
c  Example:
c
c    P(X) = 2 * X^2 - X + 6
c
c    Z = 2.0 * X + 3.0
c
c    Q(Z) = 0.5 *         Z^2 -  3.5 * Z + 12
c
c    Q(Z(X)) = 0.5 * ( 4.0 * X^2 + 12.0 * X +  9 )
c            - 3.5 * (              2.0 * X +  3 )
c                                           + 12
c
c            = 2.0         * X^2 -  1.0 * X +  6
c
c            = P(X)
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    02 December 2011
c
c  Reference:
c
c    William Press, Brian Flannery, Saul Teukolsky, William Vetterling,
c    Numerical Recipes: The Art of Scientific Computing,
c    Cambridge University Press.
c
c  Parameters:
c
c    Input, double precision SHIFT, SCALE, the shift and scale applied to X,
c    so that Z = SCALE * X + SHIFT.
c
c    Input, integer N, the number of coefficients.
c
c    Input/output, double precision POLY_COF(0:N).
c    On input, the coefficient array in terms of the X variable.
c    On output, the coefficient array in terms of the Z variable.
c
      implicit none

      integer n

      integer i
      integer j
      double precision poly_cof(0:n)
      double precision scale
      double precision shift

      do i = 1, n
        do j = i, n
          poly_cof(j) = poly_cof(j) / scale
        end do
      end do

      do i = 0, n - 1
        do j = n - 1, i, -1
          poly_cof(j) = poly_cof(j) - shift * poly_cof(j+1)
        end do
      end do

      return
      end
      function r8poly_value_horner ( m, c, x )

c*********************************************************************72
c
cc R8POLY_VALUE_HORNER evaluates a polynomial using Horner's method.
c
c  Discussion:
c
c    The polynomial 
c
c      p(x) = c0 + c1 * x + c2 * x^2 + ... + cm * x^m
c
c    is to be evaluated at X.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    02 January 2015
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the degree.
c
c    Input, double precision C(0:M), the polynomial coefficients.  
c    C(I) is the coefficient of X^I.
c
c    Input, double precision X, the evaluation point.
c
c    Output, double precision R8POLY_VALUE_HORNER, the polynomial value.
c
      implicit none

      integer m

      double precision c(0:m)
      integer i
      double precision r8poly_value_horner
      double precision value
      double precision x

      value = c(m)
      do i = m - 1, 0, -1
        value = value * x + c(i)
      end do

      r8poly_value_horner = value

      return
      end
      subroutine r8poly_values_horner ( m, c, n, x, p )

c*********************************************************************72
c
cc R8POLY_VALUES_HORNER evaluates a polynomial using Horner's method.
c
c  Discussion:
c
c    The polynomial 
c
c      p(x) = c0 + c1 * x + c2 * x^2 + ... + cm * x^m
c
c    is to be evaluated at the vector of values X.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    21 September 2012
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the degree.
c
c    Input, double precision C(0:M), the polynomial coefficients.  
c    C(I) is the coefficient of X^I.
c
c    Input, integer N, the number of evaluation points.
c
c    Input, double precision X(N), the evaluation points.
c
c    Output, double precision P(N), the polynomial values.
c
      implicit none

      integer m
      integer n

      double precision c(0:m)
      integer i
      integer j
      double precision p(n)
      double precision x(n)

      do j = 1, n
        p(j) = c(m)
      end do

      do i = m - 1, 0, -1
        do j = 1, n 
          p(j) = p(j) * x(j) + c(i)
        end do
      end do

      return
      end
      subroutine r8poly_value_2d ( m, c, n, x, y, p )

c*********************************************************************72
c
cc R8POLY_VALUE_2D evaluates a polynomial in 2 variables, X and Y.
c
c  Discussion:
c
c    We assume the polynomial is of total degree M, and has the form:
c
c      p(x,y) = c00 
c             + c10 * x                + c01 * y
c             + c20 * x^2   + c11 * xy + c02 * y^2
c             + ...
c             + cm0 * x^(m) + ...      + c0m * y^m.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    23 September 2012
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the degree of the polynomial.
c
c    Input, double precision C(T(M+1)), the polynomial coefficients.  
c    C(1) is the constant term.  T(M+1) is the M+1-th triangular number.
c    The coefficients are stored consistent with the following ordering
c    of monomials: 1, X, Y, X^2, XY, Y^2, X^3, X^2Y, XY^2, Y^3, X^4, ...
c
c    Input, integer N, the number of evaluation points.
c
c    Input, double precision X(N), Y(N), the evaluation points.
c
c    Output, double precision P(N), the value of the polynomial at the 
c    evaluation points.
c
      implicit none

      integer n

      double precision c(*)
      integer ex
      integer ey
      integer i
      integer j
      integer m
      double precision p(n)
      integer s
      double precision x(n)
      double precision y(n)

      do i = 1, n
        p(i) = 0.0D+00
      end do

      j = 0
      do s = 0, m
        do ex = s, 0, -1
          ey = s - ex
          j = j + 1
          do i = 1, n
            p(i) = p(i) + c(j) * x(i) ** ex * y(i) ** ey
          end do
        end do
      end do

      return
      end
      subroutine r8poly2_ex ( x1, y1, x2, y2, x3, y3, x, y, ierror )

c*********************************************************************72
c
cc R8POLY2_EX finds the extremal point of a parabola determined by three points.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    01 April 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision X1, Y1, X2, Y2, X3, Y3, the coordinates of
c    three points on the parabola.  X1, X2 and X3 must be distinct.
c
c    Output, double precision X, Y, the X coordinate of the extremal point
c    of the parabola, and the value of the parabola at that point.
c
c    Output, integer IERROR, error flag.
c    0, no error.
c    1, two of the X values are equal.
c    2, the data lies on a straight line; there is no finite extremal
c    point.
c
      implicit none

      double precision bot
      integer ierror
      double precision x
      double precision x1
      double precision x2
      double precision x3
      double precision y
      double precision y1
      double precision y2
      double precision y3

      ierror = 0

      if ( x1 .eq. x2 .or. x2 .eq. x3 .or. x3 .eq. x1 ) then
        ierror = 1
        return
      end if

      if ( y1 .eq. y2 .and. y2 .eq. y3 .and. y3 .eq. y1 ) then
        x = x1
        y = y1
        return
      end if

      bot = ( x2 - x3 ) * y1 - ( x1 - x3 ) * y2 + ( x1 - x2 ) * y3

      if ( bot .eq. 0.0D+00 ) then
        ierror = 2
        return
      end if

      x = 0.5D+00 * ( 
     &        x1 ** 2 * ( y3 - y2 ) 
     &      + x2 ** 2 * ( y1 - y3 ) 
     &      + x3 ** 2 * ( y2 - y1 ) ) / bot

      y = ( 
     &       ( x - x2 ) * ( x - x3 ) * ( x2 - x3 ) * y1 
     &     - ( x - x1 ) * ( x - x3 ) * ( x1 - x3 ) * y2 
     &     + ( x - x1 ) * ( x - x2 ) * ( x1 - x2 ) * y3 ) / 
     &     ( ( x1 - x2 ) * ( x2 - x3 ) * ( x1 - x3 ) )

      return
      end
      subroutine r8poly2_ex2 ( x1, y1, x2, y2, x3, y3, x, y, a, b, c, 
     &  ierror )

c*********************************************************************72
c
cc R8POLY2_EX2 finds extremal point of a parabola determined by three points.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    01 April 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision X1, Y1, X2, Y2, X3, Y3, the coordinates of
c    three points on the parabola.  X1, X2 and X3 must be distinct.
c
c    Output, double precision X, Y, the X coordinate of the extremal
c    point of the parabola, and the value of the parabola at that point.
c
c    Output, double precision A, B, C, the coefficients that define the
c    parabola: P(X) = A * X * X + B * X + C.
c
c    Output, integer IERROR, error flag.
c    0, no error.
c    1, two of the X values are equal.
c    2, the data lies on a straight line; there is no finite extremal
c    point.
c
      implicit none

      real ( kind = 8 ) a
      real ( kind = 8 ) b
      real ( kind = 8 ) c
      real ( kind = 8 ) det
      integer ierror
      real ( kind = 8 ) v(3,3)
      real ( kind = 8 ) w(3,3)
      real ( kind = 8 ) x
      real ( kind = 8 ) x1
      real ( kind = 8 ) x2
      real ( kind = 8 ) x3
      real ( kind = 8 ) y
      real ( kind = 8 ) y1
      real ( kind = 8 ) y2
      real ( kind = 8 ) y3

      ierror = 0

      if ( x1 .eq. x2 .or. x2 .eq. x3 .or. x3 .eq. x1 ) then
        ierror = 1
        return
      end if

      if ( y1 .eq. y2 .and. y2 .eq. y3 .and. y3 .eq. y1 ) then
        x = x1
        y = y1
        return
      end if
c
c  Set up the Vandermonde matrix.
c
      v(1,1) = 1.0D+00
      v(1,2) = x1
      v(1,3) = x1 * x1

      v(2,1) = 1.0D+00
      v(2,2) = x2
      v(2,3) = x2 * x2

      v(3 ,1) = 1.0D+00
      v(3,2) = x3
      v(3,3) = x3 * x3
c
c  Get the inverse.
c
      call r8mat_inverse_3d ( v, w, det )
c
c  Compute the parabolic coefficients.
c
      c = w(1,1) * y1 + w(1,2) * y2 + w(1,3) * y3
      b = w(2,1) * y1 + w(2,2) * y2 + w(2,3) * y3
      a = w(3,1) * y1 + w(3,2) * y2 + w(3,3) * y3
c
c  Determine the extremal point.
c
      if ( a .eq. 0.0D+00 ) then
        ierror = 2
        return
      end if

      x = -b / ( 2.0D+00 * a )
      y = a * x * x + b * x + c

      return
      end
      subroutine r8poly2_root ( a, b, c, r1, r2 )

c*********************************************************************72
c
cc R8POLY2_ROOT returns the two roots of a quadratic polynomial.
c
c  Discussion:
c
c    The polynomial has the form:
c
c      A * X * X + B * X + C = 0
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    01 March 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision A, B, C, the coefficients of the polynomial.
c    A must not be zero.
c
c    Output, double complex R1, R2, the roots of the polynomial, which
c    might be real and distinct, real and equal, or complex conjugates.
c
      implicit none

      double precision a
      double precision b
      double precision c
      double complex disc
      double complex q
      double complex r1
      double complex r2

      if ( a .eq. 0.0D+00 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8POLY2_ROOT - Fatal error!'
        write ( *, '(a)' ) '  The coefficient A is zero.'
        stop 1
      end if

      disc = b * b - 4.0D+00 * a * c
      q = -0.5D+00 * ( b + sign ( 1.0D+00, b ) * sqrt ( disc ) )
      r1 = q / a
      r2 = c / q

      return
      end
      subroutine r8poly2_rroot ( a, b, c, r1, r2 )

c*********************************************************************72
c
cc R8POLY2_RROOT returns the real parts of the roots of a quadratic polynomial.
c
c  Example:
c
c     A    B    C       roots              R1   R2
c    --   --   --     ------------------   --   --
c     1   -4    3     1          3          1    3
c     1    0    4     2*i      - 2*i        0    0
c     1   -6   10     3 +   i    3 -   i    3    3
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    01 December 2016
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision A, B, C, the coefficients of the quadratic
c    polynomial A * X * X + B * X + C = 0 whose roots are desired.
c    A must not be zero.
c
c    Output, double precision R1, R2, the real parts of the roots
c    of the polynomial.
c
      implicit none

      double precision a
      double precision b
      double precision c
      double precision disc
      double precision q
      double precision r1
      double precision r2

      if ( a .eq. 0.0D+00 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8POLY2_RROOT - Fatal error!'
        write ( *, '(a)' ) '  The coefficient A is zero.'
        stop 1
      end if

      disc = b * b - 4.0D+00 * a * c
      if ( 0.0D+00 <= disc ) then
        q = ( b + sign ( 1.0D+00, b ) * sqrt ( disc ) )
        r1 = -0.5D+00 * q / a
        r2 = -2.0D+00 * c / q
      else
        r1 = b / 2.0D+00 / a
        r2 = b / 2.0D+00 / a
      end if

      return
      end
      subroutine r8poly2_val ( x1, y1, x2, y2, x3, y3, x, y, yp, ypp )

c*********************************************************************72
c
cc R8POLY2_VAL evaluates a parabola defined by three data values.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    02 April 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision X1, Y1, X2, Y2, X3, Y3, three pairs of data.
c    If the X values are distinct, then all the Y values represent
c    actual values of the parabola.
c
c    Three special cases are allowed:
c
c      X1 == X2 /= X3: Y2 is the derivative at X1;
c      X1 /= X2 == X3: Y3 is the derivative at X3;
c      X1 == X2 == X3: Y2 is the derivative at X1, and
c                      Y3 is the second derivative at X1.
c
c    Input, double precision X, an abscissa at which the parabola is to be
c    evaluated.
c
c    Output, double precision Y, YP, YPP, the values of the parabola and
c    its first and second derivatives at X.
c
      implicit none

      integer distinct
      double precision dif1
      double precision dif2
      double precision x
      double precision x1
      double precision x2
      double precision x3
      double precision y
      double precision y1
      double precision y2
      double precision y3
      double precision yp
      double precision ypp
c
c  If any X's are equal, put them and the Y data first.
c
      if ( x1 .eq. x2 .and. x2 .eq. x3 ) then
        distinct = 1
      else if ( x1 .eq. x2 ) then
        distinct = 2
      else if ( x1 .eq. x3 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8POLY2_VAL - Fatal error!'
        write ( *, '(a)' ) '  X1 = X3 =/= X2.'
        write ( *, '(a,g14.6)' ) '  X1 = ', x1
        write ( *, '(a,g14.6)' ) '  X2 = ', x2
        write ( *, '(a,g14.6)' ) '  X3 = ', x3
        stop 1
      else if ( x2 .eq. x3 ) then
        distinct = 2
        call r8_swap ( x1, x2 )
        call r8_swap ( x2, x3 )
        call r8_swap ( y1, y2 )
        call r8_swap ( y2, y3 )
      else
        distinct = 3
      end if
c
c  Set up the coefficients.
c
      if ( distinct .eq. 1 ) then

        dif1 = y2
        dif2 = 0.5D+00 * y3

      else if ( distinct .eq. 2 ) then

        dif1 = y2
        dif2 = ( ( y3 - y1 ) / ( x3 - x1 ) - y2 ) / ( x3 - x2 )

      else if ( distinct .eq. 3 ) then

        dif1 = ( y2 - y1 ) / ( x2 - x1 )
        dif2 =  ( ( y3 - y1 ) / ( x3 - x1 ) 
     &          - ( y2 - y1 ) / ( x2 - x1 ) ) / ( x3 - x2 )

      end if
c
c  Evaluate.
c
      y = y1 + ( x - x1 ) * dif1 + ( x - x1 ) * ( x - x2 ) * dif2
      yp = dif1 + ( 2.0D+00 * x - x1 - x2 ) * dif2
      ypp = 2.0D+00 * dif2

      return
      end
      subroutine r8poly2_val2 ( dim_num, ndata, tdata, ydata, left, 
     &  tval, yval )

c*********************************************************************72
c
cc R8POLY2_VAL2 evaluates a parabolic interpolant through tabular data.
c
c  Discussion:
c
c    This routine is a utility routine used by OVERHAUSER_SPLINE_VAL.
c    It constructs the parabolic interpolant through the data in
c    3 consecutive entries of a table and evaluates this interpolant
c    at a given abscissa value.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    02 April 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer DIM_NUM, the dimension of a single data point.
c    DIM_NUM must be at least 1.
c
c    Input, integer NDATA, the number of data points.
c    NDATA must be at least 3.
c
c    Input, double precision TDATA(NDATA), the abscissas of the data points.
c    The values in TDATA must be in strictly ascending order.
c
c    Input, double precision YDATA(DIM_NUM,NDATA), the data points
c    corresponding to the abscissas.
c
c    Input, integer LEFT, the location of the first of the three
c    consecutive data points through which the parabolic interpolant
c    must pass.  1 <= LEFT <= NDATA - 2.
c
c    Input, double precision TVAL, the value of T at which the parabolic
c    interpolant is to be evaluated.  Normally, TDATA(1) <= TVAL <= T(NDATA),
c    and the data will be interpolated.  For TVAL outside this range,
c    extrapolation will be used.
c
c    Output, double precision YVAL(DIM_NUM), the value of the parabolic
c    interpolant at TVAL.
c
      implicit none

      integer ndata
      integer dim_num

      double precision dif1
      double precision dif2
      integer i
      integer left
      double precision t1
      double precision t2
      double precision t3
      double precision tval
      double precision tdata(ndata)
      double precision ydata(dim_num,ndata)
      double precision y1
      double precision y2
      double precision y3
      double precision yval(dim_num)
c
c  Check.
c
      if ( left .lt. 1 .or. ndata - 2 .lt. left ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8POLY2_VAL2 - Fatal error!'
        write ( *, '(a)' ) '  LEFT < 1 or NDATA-2 < LEFT.'
        write ( *, '(a,i8)' ) '  LEFT = ', left
        stop 1
      end if

      if ( dim_num .lt. 1 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8POLY2_VAL2 - Fatal error!'
        write ( *, '(a)' ) '  DIM_NUM < 1.'
        write ( *, '(a,i8)' ) '  DIM_NUM = ', dim_num
        stop 1
      end if
c
c  Copy out the three abscissas.
c
      t1 = tdata(left)
      t2 = tdata(left+1)
      t3 = tdata(left+2)

      if ( t2 .le. t1 .or. t3 .le. t2 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8POLY2_VAL2 - Fatal error!'
        write ( *, '(a)' ) '  T2 <= T1 or T3 <= T2.'
        write ( *, '(a,g14.6)' ) '  T1 = ', t1
        write ( *, '(a,g14.6)' ) '  T2 = ', t2
        write ( *, '(a,g14.6)' ) '  T3 = ', t3
        stop 1
      end if
c
c  Construct and evaluate a parabolic interpolant for the data
c  in each dimension.
c
      do i = 1, dim_num

        y1 = ydata(i,left)
        y2 = ydata(i,left+1)
        y3 = ydata(i,left+2)

        dif1 = ( y2 - y1 ) / ( t2 - t1 )
        dif2 = ( ( y3 - y1 ) / ( t3 - t1 ) 
     &         - ( y2 - y1 ) / ( t2 - t1 ) ) / ( t3 - t2 )

        yval(i) = y1 + ( tval - t1 ) * ( dif1 + ( tval - t2 ) * dif2 )

      end do

      return
      end
      subroutine r8poly3_root ( a, b, c, d, r1, r2, r3 )

c*********************************************************************72
c
cc R8POLY3_ROOT returns the three roots of a cubic polynomial.
c
c  Discussion:
c
c    The polynomial has the form
c
c      A * X^3 + B * X^2 + C * X + D = 0
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    01 March 2013
c
c  Parameters:
c
c    Input, double precision A, B, C, D, the coefficients of the polynomial.
c    A must not be zero.
c
c    Output, double complex R1, R2, R3, the roots of the polynomial, which
c    will include at least one real root.
c
      implicit none

      double precision a
      double precision b
      double precision c
      double precision d
      double complex i
      double complex one
      double precision q
      double precision r
      double complex r1
      double complex r2
      double complex r3
      double precision r8_pi
      parameter ( r8_pi = 3.141592653589793D+00 )
      double precision s1
      double precision s2
      double precision temp
      double precision theta

      if ( a .eq. 0.0D+00 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8POLY3_ROOT - Fatal error!'
        write ( *, '(a)' ) '  A must not be zero!'
        stop 1
      end if

      one = dcmplx ( 1.0d+00, 0.0D+00 )
      i = cdsqrt ( -one )

      q = ( ( b / a )**2 - 3.0D+00 * ( c / a ) ) / 9.0D+00

      r = ( 2.0D+00 * ( b / a )**3 - 9.0D+00 * ( b / a ) * ( c / a ) 
     &    + 27.0D+00 * ( d / a ) ) / 54.0D+00

      if ( r * r .lt. q * q * q ) then

        theta = acos ( r / dsqrt ( q**3 ) )
        r1 = -2.0D+00 * dsqrt ( q ) 
     &    * dcos (   theta                  / 3.0D+00 )
        r2 = -2.0D+00 * dsqrt ( q ) 
     &    * dcos ( ( theta + 2.0D+00 * r8_pi ) / 3.0D+00 )
        r3 = -2.0D+00 * dsqrt ( q ) 
     &    * dcos ( ( theta + 4.0D+00 * r8_pi ) / 3.0D+00 )

      else if ( q * q * q .le. r * r ) then

        temp = -r + dsqrt ( r**2 - q**3 )
        s1 = dsign ( 1.0D+00, temp ) 
     &    * ( dabs ( temp ) )**(1.0D+00/3.0D+00)

        temp = -r - dsqrt ( r**2 - q**3 )
        s2 = dsign ( 1.0D+00, temp ) 
     &    * ( dabs ( temp ) )**(1.0D+00/3.0D+00)

        r1 = s1 + s2
        r2 = -0.5D+00 * ( s1 + s2 ) 
     &    + i * 0.5D+00 * dsqrt ( 3.0D+00 ) * ( s1 - s2 )
        r3 = -0.5D+00 * ( s1 + s2 ) 
     &    - i * 0.5D+00 * dsqrt ( 3.0D+00 ) * ( s1 - s2 )

      end if

      r1 = r1 - b / ( 3.0D+00 * a )
      r2 = r2 - b / ( 3.0D+00 * a )
      r3 = r3 - b / ( 3.0D+00 * a )

      return
      end
      subroutine r8poly4_root ( a, b, c, d, e, r1, r2, r3, r4 )

c*********************************************************************72
c
cc R8POLY4_ROOT returns the four roots of a quartic polynomial.
c
c  Discussion:
c
c    The polynomial has the form:
c
c      A * X^4 + B * X^3 + C * X^2 + D * X + E = 0
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    01 March 2013
c
c  Parameters:
c
c    Input, double precision A, B, C, D, the coefficients of the polynomial.
c    A must not be zero.
c
c    Output, double complex R1, R2, R3, R4, the roots of the polynomial.
c
      implicit none

      double precision a
      double precision a3
      double precision a4
      double precision b
      double precision b3
      double precision b4
      double precision c
      double precision c3
      double precision c4
      double precision d
      double precision d3
      double precision d4
      double precision e
      double complex p
      double complex q
      double complex r
      double complex r1
      double complex r2
      double complex r3
      double complex r4
      double complex zero

      zero = dcmplx ( 0.0D+00, 0.0D+00 )

      if ( a .eq. 0.0D+00 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8POLY4_ROOT - Fatal error!'
        write ( *, '(a)') '  A must not be zero!'
        stop 1
      end if

      a4 = b / a
      b4 = c / a
      c4 = d / a
      d4 = e / a
c
c  Set the coefficients of the resolvent cubic equation.
c
      a3 = 1.0D+00
      b3 = -b4
      c3 = a4 * c4 - 4.0D+00 * d4
      d3 = -a4 * a4 * d4 + 4.0D+00 * b4 * d4 - c4 * c4
c
c  Find the roots of the resolvent cubic.
c
      call r8poly3_root ( a3, b3, c3, d3, r1, r2, r3 )
c
c  Choose one root of the cubic, here R1.
c
c  Set R = sqrt ( 0.25D+00 * A4**2 - B4 + R1 )
c
      r = cdsqrt ( 0.25D+00 * a4**2 - b4 + r1 )

      if ( r .ne. zero ) then

        p = cdsqrt ( 0.75D+00 * a4**2 - r**2 - 2.0D+00 * b4 
     &    + 0.25D+00 * ( 4.0D+00 * a4 * b4 - 8.0D+00 * c4 - a4**3 ) 
     & / r )

        q = cdsqrt ( 0.75D+00 * a4**2 - r**2 - 2.0D+00 * b4 
     &    - 0.25D+00 * ( 4.0D+00 * a4 * b4 - 8.0D+00 * c4 - a4**3 ) 
     &    / r )

      else

        p = cdsqrt ( 0.75D+00 * a4**2 - 2.0D+00 * b4 
     &    + 2.0D+00 * cdsqrt ( r1**2 - 4.0D+00 * d4 ) )

        q = cdsqrt ( 0.75D+00 * a4**2 - 2.0D+00 * b4 
     &    - 2.0D+00 * cdsqrt ( r1**2 - 4.0D+00 * d4 ) )

      end if
c
c  Set the roots.
c
      r1 = -0.25D+00 * a4 + 0.5D+00 * r + 0.5D+00 * p
      r2 = -0.25D+00 * a4 + 0.5D+00 * r - 0.5D+00 * p
      r3 = -0.25D+00 * a4 - 0.5D+00 * r + 0.5D+00 * q
      r4 = -0.25D+00 * a4 - 0.5D+00 * r - 0.5D+00 * q

      return
      end
      subroutine r8row_compare ( m, n, a, i, j, value )

c*********************************************************************72
c
cc R8ROW_COMPARE compares rows in an R8ROW.
c
c  Discussion:
c
c    An R8ROW is an M by N array of R8's, regarded as an array of M rows,
c    each of length N.
c
c  Example:
c
c    Input:
c
c      M = 4, N = 3, I = 2, J = 4
c
c      A = (
c        1  5  9
c        2  6 10
c        3  7 11
c        4  8 12 )
c
c    Output:
c
c      VALUE = -1
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    25 May 2012
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns.
c
c    Input, double precision A(M,N), the M by N array.
c
c    Input, integer I, J, the rows to be compared.
c    I and J must be between 1 and M.
c
c    Output, integer VALUE, the results of the comparison:
c    -1, row I < row J,
c     0, row I = row J,
c    +1, row J < row I.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer j
      integer k
      integer value
c
c  Check.
c
      if ( i .lt. 1 .or. m .lt. i ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8ROW_COMPARE - Fatal error!'
        write ( *, '(a)' ) '  Row index I is out of bounds.'
        write ( *, '(a,i8)' ) '  I = ', i
        stop 1
      end if

      if ( j .lt. 1 .or. m .lt. j ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8ROW_COMPARE - Fatal error!'
        write ( *, '(a)' ) '  Row index J is out of bounds.'
        write ( *, '(a,i8)' ) '  J = ', j
        stop 1
      end if

      value = 0

      if ( i .eq. j ) then
        return
      end if

      k = 1

10    continue

      if ( k .le. n ) then

        if ( a(i,k) .lt. a(j,k) ) then
          value = -1
          return
        else if ( a(j,k) .lt. a(i,k) ) then
          value = +1
          return
        end if

        k = k + 1

        go to 10

      end if

      return
      end
      subroutine r8row_max ( m, n, a, amax )

c*********************************************************************72
c
cc R8ROW_MAX returns the maximums of an R8ROW.
c
c  Discussion:
c
c    An R8ROW is an M by N array of R8 values, regarded
c    as an array of M rows of length N.
c
c  Example:
c
c    A =
c      1  2  3
c      2  6  7
c
c    MAX =
c      3
c      7
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    16 May 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns
c    in the array.
c
c    Input, double precision A(M,N), the array to be examined.
c
c    Output, double precision AMAX(M), the maximums of the rows.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      double precision amax(m)
      integer i
      integer j

      do i = 1, m

        amax(i) = a(i,1)
        do j = 2, n
          if ( amax(i) .lt. a(i,j) ) then
            amax(i) = a(i,j)
          end if
        end do

      end do

      return
      end
      subroutine r8row_mean ( m, n, a, mean )

c*********************************************************************72
c
cc R8ROW_MEAN returns the means of an R8ROW.
c
c  Discussion:
c
c    An R8ROW is an M by N array of R8 values, regarded
c    as an array of M rows of length N.
c
c  Example:
c
c    A =
c      1  2  3
c      2  6  7
c
c    MEAN =
c      2
c      5
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    02 December 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns.
c
c    Input, double precision A(M,N), the array to be examined.
c
c    Output, double precision MEAN(M), the means, or averages, of the rows.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer j
      double precision mean(m)

      do i = 1, m
        mean(i) = 0.0D+00
        do j = 1, n
          mean(i) = mean(i) + a(i,j)
        end do
        mean(i) = mean(i) / dble ( n )
      end do

      return
      end
      subroutine r8row_min ( m, n, a, amin )

c*********************************************************************72
c
cc R8ROW_MIN returns the minimums of an R8ROW.
c
c  Discussion:
c
c    An R8ROW is an M by N array of R8 values, regarded
c    as an array of M rows of length N.
c
c  Example:
c
c    A =
c      1  2  3
c      2  6  7
c
c    MIN =
c      1
c      2
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    02 December 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns
c    in the array.
c
c    Input, double precision A(M,N), the array to be examined.
c
c    Output, double precision AMIN(M), the minimums of the rows.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      double precision amin(m)
      integer i
      integer j

      do i = 1, m

        amin(i) = a(i,1)
        do j = 2, n
          if ( a(i,j) .lt. amin(i) ) then
            amin(i) = a(i,j)
          end if
        end do

      end do

      return
      end
      subroutine r8row_part_quick_a ( m, n, a, l, r )

c*********************************************************************72
c
cc R8ROW_PART_QUICK_A reorders the rows of an R8ROW.
c
c  Discussion:
c
c    An R8ROW is an M by N array of R8's, regarded as an array of M rows,
c    each of length N.
c
c    The routine reorders the rows of A.  Using A(1,1:N) as a
c    key, all entries of A that are less than or equal to the key will
c    precede the key, which precedes all entries that are greater than the key.
c
c  Example:
c
c    Input:
c
c      M = 8, N = 2
c      A = ( 2 4
c            8 8
c            6 2
c            0 2
c           10 6
c           10 0
c            0 6
c            5 8 )
c
c    Output:
c
c      L = 2, R = 4
c
c      A = ( 0 2    LEFT
c            0 6
c            ----
c            2 4    KEY
c            ----
c            8 8    RIGHT
c            6 2
c           10 6
c           10 0
c            5 8 )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    26 May 2012
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the row dimension of A.
c
c    Input, integer N, the column dimension of A, and the
c    length of a row.
c
c    Input/output, double precision A(M,N).  On input, the array to be checked.
c    On output, A has been reordered as described above.
c
c    Output, integer L, R, the indices of A that define the three
c    segments.  Let KEY = the input value of A(1,1:N).  Then
c    I <= L                 A(I,1:N) < KEY;
c         L < I < R         A(I,1:N) = KEY;
c                 R <= I    KEY < A(I,1:N).
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer j
      integer k
      double precision key(n)
      integer l
      integer r
      logical r8vec_eq
      logical r8vec_gt
      logical r8vec_lt

      if ( m .lt. 1 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8ROW_PART_QUICK_A - Fatal error!'
        write ( *, '(a)' ) '  M < 1.'
        return
      end if

      if ( m .eq. 1 ) then
        l = 0
        r = 2
        return
      end if

      do i = 1, n
        key(i) = a(1,i)
      end do
      k = 1
c
c  The elements of unknown size have indices between L+1 and R-1.
c
      l = 1
      r = m + 1

      do j = 2, m

        if ( r8vec_gt ( n, a(l+1,1:n), key ) ) then
          r = r - 1
          call r8vec_swap ( n, a(r,1:n), a(l+1,1:n) )
        else if ( r8vec_eq ( n, a(l+1,1:n), key ) ) then
          k = k + 1
          call r8vec_swap ( n, a(k,1:n), a(l+1,1:n) )
          l = l + 1
        else if ( r8vec_lt ( n, a(l+1,1:n), key ) ) then
          l = l + 1
        end if

      end do
c
c  Shift small elements to the left.
c
      do j = 1, l - k
        do i = 1, n
          a(j,i) = a(j+k,i)
        end do
      end do
c
c  Shift KEY elements to center.
c
      do j = l - k + 1, l
        do i = 1, n
          a(j,i) = key(i)
        end do
      end do
c
c  Update L.
c
      l = l - k

      return
      end
      subroutine r8row_reverse ( m, n, a )

c********************************************************************72
c
cc R8ROW_REVERSE reverses the order of the rows of an R8MAT.
c
c  Discussion:
c
c    To reverse the rows is to start with something like
c
c      11 12 13 14 15
c      21 22 23 24 25
c      31 32 33 34 35
c      41 42 43 44 45
c      51 52 53 54 55
c
c    and return
c
c      51 52 53 54 55
c      41 42 43 44 45
c      31 32 33 34 35
c      21 22 23 24 25
c      11 12 13 14 15
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    06 May 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns.
c
c    Input/output, double precision A(M,N), the matrix.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer ihi
      integer j
      double precision t

      ihi = m / 2

      do i = 1, ihi
        do j = 1, n
          t          = a(i,j)
          a(i,j)     = a(m+1-i,j)
          a(m+1-i,j) = t
        end do
      end do

      return
      end
      subroutine r8row_sort_heap_a ( m, n, a )

c*********************************************************************72
c
cc R8ROW_SORT_HEAP_A ascending heapsorts an R8ROWL.
c
c  Discussion:
c
c    An R8ROW is an M by N array of R8's, regarded as an array of M rows,
c    each of length N.
c
c    In lexicographic order, the statement "X < Y", applied to two real
c    vectors X and Y of length M, means that there is some index I, with
c    1 <= I <= M, with the property that
c
c      X(J) = Y(J) for J < I,
c    and
c      X(I) < Y(I).
c
c    In other words, the first time they differ, X is smaller.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    25 May 2012
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns.
c
c    Input/output, double precision A(M,N).
c    On input, the array of M rows of N-vectors.
c    On output, the rows of A have been sorted in lexicographic order.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer indx
      integer isgn
      integer j

      if ( m .le. 1 ) then
        return
      end if
c
c  Initialize.
c
      i = 0
      indx = 0
      isgn = 0
      j = 0
c
c  Call the external heap sorter.
c
10    continue

        call sort_heap_external ( m, indx, i, j, isgn )
c
c  Interchange the I and J objects.
c
        if ( 0 .lt. indx ) then

          call r8row_swap ( m, n, a, i, j )
c
c  Compare the I and J objects.
c
        else if ( indx .lt. 0 ) then

          call r8row_compare ( m, n, a, i, j, isgn )

        else if ( indx .eq. 0 ) then

          go to 20

        end if

      go to 10

20    continue

      return
      end
      subroutine r8row_sort_heap_index_a ( m, n, a, indx )

c*********************************************************************72
c
cc R8ROW_SORT_HEAP_INDEX_A does an indexed heap ascending sort of an R8ROW.
c
c  Discussion:
c
c    An R8ROW is an M by N array of R8's, regarded as an array of M rows,
c    each of length N.
c
c    The sorting is not actually carried out.  Rather an index array is
c    created which defines the sorting.  This array may be used to sort
c    or index the array, or to sort or index related arrays keyed on the
c    original array.
c
c    A(I1,*) < A(I1,*) if the first nonzero entry of A(I1,*)-A(I2,*)
c    is negative.
c
c    Once the index array is computed, the sorting can be carried out
c    "implicitly:
c
c      A(INDX(1:M),1:N) is sorted.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    26 May 2012
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the number of rows in each column of A.
c
c    Input, integer N, the number of columns in A.
c
c    Input, double precision A(M,N), the array.
c
c    Output, integer INDX(M), the sort index.  The I-th element
c    of the sorted array is row INDX(I).
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer indx(m)
      integer indxt
      integer ir
      integer isgn
      integer j
      integer k
      integer l
      double precision row(n)
      double precision row2(n)

      do i = 1, m
        indx(i) = i
      end do

      if ( m .eq. 1 ) then
        return
      end if

      l = ( m / 2 ) + 1
      ir = m

10    continue

        if ( 1 .lt. l ) then

          l = l - 1
          indxt = indx(l)
          do k = 1, n
            row(k) = a(indxt,k)
          end do

        else

          indxt = indx(ir)
          do k = 1, n
            row(k) = a(indxt,k)
          end do
          indx(ir) = indx(1)
          ir = ir - 1

          if ( ir .eq. 1 ) then
            indx(1) = indxt
            go to 30
          end if

        end if

        i = l
        j = l + l

20      continue

        if ( j .le. ir ) then

          if ( j .lt. ir ) then

            call r8row_compare ( m, n, a, indx(j), indx(j+1), isgn )

            if ( isgn .lt. 0 ) then
              j = j + 1
            end if

          end if

          do k = 1, n
            row2(k) = a(indx(j),k)
          end do

          call r8vec_compare ( n, row, row2, isgn )

          if ( isgn .lt. 0 ) then
            indx(i) = indx(j)
            i = j
            j = j + j
          else
            j = ir + 1
          end if

          go to 20

        end if

        indx(i) = indxt

      go to 10

30    continue

      return
      end
      subroutine r8row_sort_quick_a ( m, n, a )

c*********************************************************************72
c
cc R8ROW_SORT_QUICK_A ascending quick sorts an R8ROW.
c
c  Discussion:
c
c    An R8ROW is an M by N array of R8's, regarded as an array of M rows,
c    each of length N.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    26 May 2012
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the number of rows of A.
c
c    Input, integer N, the number of columns of A,
c    and the length of a row.
c
c    Input/output, double precision A(M,N).
c    On input, the array to be sorted.
c    On output, the array has been sorted.
c
      implicit none

      integer level_max
      parameter ( level_max = 30 )
      integer m
      integer n

      double precision a(m,n)
      integer base
      integer l_segment
      integer level
      integer m_segment
      integer rsave(level_max)
      integer r_segment

      if ( n .le. 0 ) then
        return
      end if

      if ( m .lt. 1 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8ROW_SORT_QUICK_A - Fatal error!'
        write ( *, '(a)' ) '  M < 1.'
        write ( *, '(a,i8)' ) '  M = ', m
        stop 1
      end if

      if ( m .eq. 1 ) then
        return
      end if

      level = 1
      rsave(level) = m + 1
      base = 1
      m_segment = m

10    continue
c
c  Partition the segment.
c
        call r8row_part_quick_a ( m_segment, n, 
     &    a(base:base+m_segment-1,1:n), l_segment, r_segment )
c
c  If the left segment has more than one element, we need to partition it.
c
        if ( 1 .lt. l_segment ) then

          if ( level_max .lt. level ) then
            write ( *, '(a)' ) ' '
            write ( *, '(a)' ) 'R8ROW_SORT_QUICK_A - Fatal error!'
            write ( *, '(a,i8)' ) 
     &        '  Exceeding recursion maximum of ', level_max
            stop 1
          end if

          level = level + 1
          m_segment = l_segment
          rsave(level) = r_segment + base - 1
c
c  The left segment and the middle segment are sorted.
c  Must the right segment be partitioned?
c
        else if ( r_segment .lt. m_segment ) then

          m_segment = m_segment + 1 - r_segment
          base = base + r_segment - 1
c
c  Otherwise, we back up a level if there is an earlier one.
c
        else

20        continue

            if ( level .le. 1 ) then
              return
            end if

            base = rsave(level)
            m_segment = rsave(level-1) - rsave(level)
            level = level - 1

            if ( 0 .lt. m_segment ) then
              go to 30
            end if

          go to 20

30        continue

        end if

      go to 10

      return
      end
      subroutine r8row_sorted_unique_count ( m, n, a, unique_num )

c*********************************************************************72
c
cc R8ROW_SORTED_UNIQUE_COUNT counts unique elements in an R8ROW.
c
c  Discussion:
c
c    An R8ROW is an M by N array of R8 values, regarded
c    as an array of M rows of length N.
c
c    The rows of the array may be ascending or descending sorted.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    02 December 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns.
c
c    Input, double precision A(M,N), a sorted array, containing
c    M rows of data.
c
c    Output, integer UNIQUE_NUM, the number of unique rows.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i1
      integer i2
      integer j
      integer unique_num

      if ( n .le. 0 ) then
        unique_num = 0
        return
      end if

      unique_num = 1
      i1 = 1

      do i2 = 2, m

        do j = 1, n

          if ( a(i1,j) .ne. a(i2,j) ) then
            unique_num = unique_num + 1
            i1 = i2
            go to 10
          end if

        end do

10      continue

      end do

      return
      end
      subroutine r8row_sum ( m, n, a, rowsum )

c*********************************************************************72
c
cc R8ROW_SUM returns the sums of the rows of an R8ROW.
c
c  Discussion:
c
c    An R8ROW is an M by N array of R8 values, regarded
c    as an array of M rows of length N.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    06 December 2004
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns.
c
c    Input, double precision A(M,N), the M by N array.
c
c    Output, double precision ROWSUM(M), the sum of the entries of
c    each row.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      double precision rowsum(m)

      do i = 1, m
        rowsum(i) = sum ( a(i,1:n) )
      end do

      return
      end
      subroutine r8row_swap ( m, n, a, i1, i2 )

c*********************************************************************72
c
cc R8ROW_SWAP swaps two rows of an R8ROW.
c
c  Discussion:
c
c    An R8ROW is an M by N array of R8 values, regarded
c    as an array of M rows of length N.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    05 December 2004
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns.
c
c    Input/output, double precision A(M,N), the M by N array.
c
c    Input, integer I1, I2, the two rows to swap.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i1
      integer i2
      double precision row(n)

      if ( i1 .lt. 1 .or. m .lt. i1 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8ROW_SWAP - Fatal error!'
        write ( *, '(a)' ) '  I1 is out of range.'
        write ( *, '(a,i8)' ) '  I1 = ', i1
        stop 1
      end if

      if ( i2 .lt. 1 .or. m .lt. i2 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8ROW_SWAP - Fatal error!'
        write ( *, '(a)' ) '  I2 is out of range.'
        write ( *, '(a,i8)' ) '  I2 = ', i2
        stop 1
      end if

      if ( i1 .eq. i2 ) then
        return
      end if

      row(1:n) = a(i1,1:n)
      a(i1,1:n) = a(i2,1:n)
      a(i2,1:n) = row(1:n)

      return
      end
      subroutine r8row_to_r8vec ( m, n, a, x )

c*********************************************************************72
c
cc R8ROW_TO_R8VEC converts an R8ROW into an R8VEC.
c
c  Discussion:
c
c    An R8ROW is an M by N array of R8 values, regarded
c    as an array of M rows of length N.
c
c  Example:
c
c    M = 3, N = 4
c
c    A =
c      11 12 13 14
c      21 22 23 24
c      31 32 33 34
c
c    X = ( 11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 34 )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    13 July 2000
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns.
c
c    Input, double precision A(M,N), the M by N array.
c
c    Output, double precision X(M*N), a vector containing the M rows of A.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer j
      double precision x(m*n)

      j = 1
      do i = 1, m
        x(j:j+n-1) = a(i,1:n)
        j = j + n
      end do

      return
      end
      subroutine r8row_variance ( m, n, a, variance )

c*********************************************************************72
c
cc R8ROW_VARIANCE returns the variances of an R8ROW.
c
c  Discussion:
c
c    An R8ROW is an M by N array of R8 values, regarded
c    as an array of M rows of length N.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    05 April 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns
c    in the array.
c
c    Input, double precision A(M,N), the array whose variances are desired.
c
c    Output, double precision VARIANCE(M), the variances of the rows.
c
      implicit none

      integer m
      integer n

      double precision a(m,n)
      integer i
      integer j
      double precision mean
      double precision variance(m)

      do i = 1, m

        mean = 0.0D+00
        do j = 1, n
          mean = mean + a(i,j)
        end do
        mean = mean / dble ( n )

        variance(i) = 0.0D+00
        do j = 1, n
          variance(i) = variance(i) + ( a(i,j) - mean )**2
        end do

        if ( 1 .lt. n ) then
          variance(i) = variance(i) / dble ( n - 1 )
        else
          variance(i) = 0.0D+00
        end if

      end do

      return
      end
      subroutine r8slmat_print ( m, n, a, title )

c*********************************************************************72
c
cc R8SLMAT_PRINT prints a strict lower triangular R8MAT.
c
c  Example:
c
c    M = 5, N = 5
c    A = (/ 21, 31, 41, 51, 32, 42, 52, 43, 53, 54 /)
c
c    21
c    31 32
c    41 42 43
c    51 52 53 54
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    05 April 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the number of rows in A.
c
c    Input, integer N, the number of columns in A.
c
c    Input, double precision A(*), the M by N matrix.  Only the strict
c    lower triangular elements are stored, in column major order.
c
c    Input, character * ( * ) TITLE, a title.
c
      implicit none

      integer m
      integer n

      double precision a(*)
      integer i
      integer indx(10)
      integer j
      integer jhi
      integer jlo
      integer jmax
      integer nn
      integer size
      character * ( * ) title

      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) trim ( title )

      jmax = min ( n, m - 1 )

      if ( m-1 .le. n ) then
        size = ( m * ( m - 1 ) ) / 2
      else if ( n .lt. m-1 ) then
        size = ( n * ( n - 1 ) ) / 2 + ( m - n - 1 ) * n
      end if

      nn = 5

      do jlo = 1, jmax, nn
        jhi = min ( jlo + nn - 1, m - 1, jmax )
        write ( *, '(a)' ) ' '
        write ( *, '(a10,5(i8,6x))' ) '       Col', ( j, j = jlo, jhi )
        write ( *, '(a10)' ) '       Row'
        do i = jlo + 1, m
          jhi = min ( jlo + nn - 1, i - 1, jmax )
          do j = jlo, jhi
            indx(j+1-jlo) = ( j - 1 ) * m + i - ( j * ( j + 1 ) ) / 2
          end do
          write ( *, '(2x,i8,5g14.6)' ) i, a(indx(1:jhi+1-jlo))
        end do
      end do

      return
      end
      subroutine r8vec_01_to_ab ( n, a, amax, amin )

c*********************************************************************72
c
cc R8VEC_01_TO_AB shifts and rescales an R8VEC to lie within given bounds.
c
c  Discussion:
c
c    An R8VEC is a vector of R8 values.
c
c    On input, A contains the original data, which is presumed to lie
c    between 0 and 1.  However, it is not necessary that this be so.
c
c    On output, A has been shifted and rescaled so that all entries which
c    on input lay in [0,1] now lie between AMIN and AMAX.  Other entries will
c    be mapped in a corresponding way.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    26 August 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of data values.
c
c    Input/output, double precision A(N), the vector to be rescaled.
c
c    Input, double precision AMAX, AMIN, the maximum and minimum values
c    allowed for A.
c
      implicit none

      integer n

      double precision a(n)
      double precision amax
      double precision amax2
      double precision amax3
      double precision amin
      double precision amin2
      double precision amin3
      integer i

      if ( amax .eq. amin ) then
        a(1:n) = amin
        return
      end if

      amax2 = max ( amax, amin )
      amin2 = min ( amax, amin )

      amin3 = a(1)
      do i = 2, n
        amin3 = min ( amin3, a(i) )
      end do

      amax3 = a(1)
      do i = 2, n
        amax3 = max ( amax3, a(i) )
      end do

      if ( amax3 .ne. amin3 ) then

        do i = 1, n

          a(i) = ( ( amax3 - a(i)         ) * amin2
     &           + (         a(i) - amin3 ) * amax2 )
     &           / ( amax3        - amin3 )
        end do

      else

        do i = 1, n
          a(i) = 0.5D+00 * ( amax2 + amin2 )
        end do
      end if

      return
      end
      subroutine r8vec_ab_to_01 ( n, a )

c*********************************************************************72
c
cc R8VEC_AB_TO_01 shifts and rescales an R8VEC to lie within [0,1].
c
c  Discussion:
c
c    An R8VEC is a vector of R8 values.
c
c    On input, A contains the original data.  On output, A has been shifted
c    and scaled so that all entries lie between 0 and 1.
c
c  Formula:
c
c    A(I) := ( A(I) - AMIN ) / ( AMAX - AMIN )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    27 August 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of data values.
c
c    Input/output, double precision A(N), the data to be rescaled.
c
      implicit none

      integer n

      double precision a(n)
      double precision amax
      double precision amin
      integer i

      amin = a(1)
      do i = 2, n
        amin = min ( amin, a(i) )
      end do

      amax = a(1)
      do i = 2, n
        amax = max ( amax, a(i) )
      end do

      if ( amin .eq. amax ) then
        do i = 1, n
          a(i) = 0.5D+00
        end do
      else
        do i = 1, n
          a(i) = ( a(i) - amin ) / ( amax - amin )
        end do
      end if

      return
      end
      subroutine r8vec_ab_to_cd ( n, a, bmin, bmax, b )

c*********************************************************************72
c
cc R8VEC_AB_TO_CD shifts and rescales an R8VEC from one interval to another.
c
c  Discussion:
c
c    An R8VEC is a vector of R8 values.
c
c    The mininum entry of A is mapped to BMIN, the maximum entry
c    to BMAX, and values in between are mapped linearly.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    30 August 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of data values.
c
c    Input, double precision A(N), the data to be remapped.
c
c    Input, double precision BMIN, BMAX, the values to which min(A) and max(A)
c    are to be assigned.
c
c    Output, double precision B(N), the remapped data.
c
      implicit none

      integer n

      double precision a(n)
      double precision amax
      double precision amin
      double precision b(n)
      double precision bmax
      double precision bmin
      integer i

      if ( bmax .eq. bmin ) then
        do i = 1, n
          b(i) = bmin
        end do
        return
      end if

      amin = a(1)
      do i = 2, n
        amin = min ( amin, a(i) )
      end do

      amax = a(1)
      do i = 2, n
        amax = max ( amax, a(i) )
      end do

      if ( amax .eq. amin ) then
        do i = 1, n
          b(i) = 0.5D+00 * ( bmax + bmin )
        end do
        return
      end if

      do i = 1, n
        b(i) = ( ( amax - a(i)        ) * bmin
     &       + (          a(i) - amin ) * bmax )
     &         / ( amax        - amin )
      end do

      return
      end
      function r8vec_all_nonpositive ( n, a )

c*********************************************************************72
c
cc R8VEC_ALL_NONPOSITIVE: ( all ( A <= 0 ) ) for R8VEC's.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    08 October 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries.
c
c    Input, double precision A(N), the vector.
c
c    Output, logical R8VEC_ALL_NONPOSITIVE is TRUE if all entries
c    of A are less than or equal to 0.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      logical r8vec_all_nonpositive

      do i = 1, n
        if ( 0.0D+00 .lt. a(i) ) then
          r8vec_all_nonpositive = .false.
          return
        end if
      end do

      r8vec_all_nonpositive = .true.

      return
      end
      function r8vec_amax ( n, a )

c*********************************************************************72
c
cc R8VEC_AMAX returns the maximum absolute value in an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    02 June 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the array.
c
c    Input, double precision A(N), the array.
c
c    Output, double precision R8VEC_AMAX, the value of the entry
c    of largest magnitude.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      double precision r8vec_amax
      double precision value

      value = 0.0D+00
      do i = 1, n
        value = max ( value, abs ( a(i) ) )
      end do

      r8vec_amax = value

      return
      end
      subroutine r8vec_amax_index ( n, a, amax_index )

c*********************************************************************72
c
cc R8VEC_AMAX_INDEX returns the index of the maximum absolute value in an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    02 June 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the array.
c
c    Input, double precision A(N), the array.
c
c    Output, integer AMAX_INDEX, the index of the entry of largest magnitude.
c
      implicit none

      integer n

      double precision a(n)
      double precision amax
      integer amax_index
      integer i

      if ( n .le. 0 ) then

        amax_index = -1

      else

        amax_index = 1
        amax = abs ( a(1) )

        do i = 2, n
          if ( amax .lt. abs ( a(i) ) ) then
            amax_index = i
            amax = abs ( a(i) )
          end if
        end do

      end if

      return
      end
      function r8vec_amin ( n, a )

c*********************************************************************72
c
cc R8VEC_AMIN returns the minimum absolute value in an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    02 June 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the array.
c
c    Input, double precisionA(N), the array.
c
c    Output, double precision R8VEC_AMIN, the value of the entry
c    of smallest magnitude.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      double precision r8_huge
      parameter ( r8_huge = 1.79769313486231571D+308 )
      double precision r8vec_amin
      double precision value

      value = r8_huge
      do i = 1, n
        value = min ( value, abs ( a(i) ) )
      end do

      r8vec_amin = value

      return
      end
      subroutine r8vec_amin_index ( n, a, amin_index )

c*********************************************************************72
c
cc R8VEC_AMIN_INDEX returns the index of the minimum absolute value in an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    02 June 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the array.
c
c    Input, double precision A(N), the array.
c
c    Output, integer AMIN_INDEX, the index of the entry of smallest magnitude.
c
      implicit none

      integer n

      double precision a(n)
      double precision amin
      integer amin_index
      integer i

      if ( n .le. 0 ) then

        amin_index = 0

      else

        amin_index = 1
        amin = abs ( a(1) )

        do i = 2, n
          if ( abs ( a(i) ) .lt. amin ) then
            amin_index = i
            amin = abs ( a(i) )
          end if
        end do

      end if

      return
      end
      function r8vec_any_negative ( n, a )

c*********************************************************************72
c
cc R8VEC_ANY_NEGATIVE: ( any ( A < 0 ) ) for R8VEC's.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    09 October 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries.
c
c    Input, double precision A(N), the vector.
c
c    Output, logical R8VEC_ANY_NEGATIVE is TRUE if any entry
c    of A is less than zero.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      logical r8vec_any_negative

      do i = 1, n
        if ( a(i) .lt. 0.0D+00 ) then
          r8vec_any_negative = .true.
          return
        end if
      end do

      r8vec_any_negative = .false.

      return
      end
      function r8vec_any_nonzero ( n, a )

c*********************************************************************72
c
cc R8VEC_ANY_NONZERO: ( any A nonzero ) for R8VEC's.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    25 December 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries.
c
c    Input, double precision A(N), the vector.
c
c    Output, logical R8VEC_ANY_NONZERO is TRUE if any entry is nonzero.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      logical r8vec_any_nonzero

      do i = 1, n
        if ( a(i) .ne. 0.0D+00 ) then
          r8vec_any_nonzero = .true.
          return
        end if
      end do

      r8vec_any_nonzero = .false.

      return
      end
      subroutine r8vec_any_normal ( dim_num, v1, v2 )

c*********************************************************************72
c
cc R8VEC_ANY_NORMAL returns some normal vector to V1.
c
c  Discussion:
c
c    If DIM_NUM < 2, then no normal vector can be returned.
c
c    If V1 is the zero vector, then any unit vector will do.
c
c    No doubt, there are better, more robust algorithms.  But I will take
c    just about ANY reasonable unit vector that is normal to V1.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    18 May 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer DIM_NUM, the spatial dimension.
c
c    Input, double precision V1(DIM_NUM), the vector.
c
c    Output, double precision V2(DIM_NUM), a vector that is
c    normal to V2, and has unit Euclidean length.
c
      implicit none

      integer dim_num

      integer i
      integer j
      integer k
      double precision r8vec_norm
      double precision v1(dim_num)
      double precision v2(dim_num)
      double precision vj
      double precision vk

      if ( dim_num .lt. 2 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8VEC_ANY_NORMAL - Fatal error!'
        write ( *, '(a)' ) '  Called with DIM_NUM < 2.'
        stop 1
      end if

      if ( r8vec_norm ( dim_num, v1 ) .eq. 0.0D+00 ) then
        v2(1) = 1.0D+00
        do i = 2, dim_num
          v2(i) = 0.0D+00
        end do
        return
      end if
c
c  Seek the largest entry in V1, VJ = V1(J), and the
c  second largest, VK = V1(K).
c
c  Since V1 does not have zero norm, we are guaranteed that
c  VJ, at least, is not zero.
c
      j = - 1
      vj = 0.0D+00

      k = - 1
      vk = 0.0D+00

      do i = 1, dim_num

        if ( abs ( vk ) .lt. abs ( v1(i) ) .or. k .lt. 1 ) then

          if ( abs ( vj ) .lt. abs ( v1(i) ) .or. j .lt. 1 ) then
            k = j
            vk = vj
            j = i
            vj = v1(i)
          else
            k = i
            vk = v1(i)
          end if

        end if

      end do
c
c  Setting V2 to zero, except that V2(J) = -VK, and V2(K) = VJ,
c  will just about do the trick.
c
      do i = 1, dim_num
        v2(i) = 0.0D+00
      end do

      v2(j) = - vk / sqrt ( vk * vk + vj * vj )
      v2(k) =   vj / sqrt ( vk * vk + vj * vj )

      return
      end
      function r8vec_ascends ( n, x )

c*********************************************************************72
c
cc R8VEC_ASCENDS determines if an R8VEC is (weakly) ascending.
c
c  Discussion:
c
c    An R8VEC is a vector of R8 values.
c
c    For example, if:
c
c      X = ( -8.1, 1.3, 2.2, 3.4, 7.5, 7.5, 9.8 )
c
c    then
c
c      R8VEC_ASCENDS = TRUE
c
c    The sequence is not required to be strictly ascending.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    26 August 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the size of the array.
c
c    Input, double precision X(N), the array to be examined.
c
c    Output, logical R8VEC_ASCENDS, is TRUE if the
c    entries of X ascend.
c
      implicit none

      integer n

      integer i
      logical r8vec_ascends
      double precision x(n)

      do i = 1, n - 1
        if ( x(i+1) .lt. x(i) ) then
          r8vec_ascends = .false.
          return
        end if
      end do

      r8vec_ascends = .true.

      return
      end
      function r8vec_ascends_strictly ( n, x )

c*********************************************************************72
c
cc R8VEC_ASCENDS_STRICTLY determines if an R8VEC is strictly ascending.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    Notice the effect of entry number 6 in the following results:
c
c      X = ( -8.1, 1.3, 2.2, 3.4, 7.5, 7.4, 9.8 )
c      Y = ( -8.1, 1.3, 2.2, 3.4, 7.5, 7.5, 9.8 )
c      Z = ( -8.1, 1.3, 2.2, 3.4, 7.5, 7.6, 9.8 )
c
c      R8VEC_ASCENDS_STRICTLY ( X ) = FALSE
c      R8VEC_ASCENDS_STRICTLY ( Y ) = FALSE
c      R8VEC_ASCENDS_STRICTLY ( Z ) = TRUE
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    16 May 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the size of the array.
c
c    Input, double precision X(N), the array to be examined.
c
c    Output, logical R8VEC_ASCENDS_STRICTLY, is TRUE if the
c    entries of X strictly ascend.
c
      implicit none

      integer n

      integer i
      logical r8vec_ascends_strictly
      double precision x(n)

      do i = 1, n - 1
        if ( x(i+1) .le. x(i) ) then
          r8vec_ascends_strictly = .false.
          return
        end if
      end do

      r8vec_ascends_strictly = .true.

      return
      end
      function r8vec_asum ( n, v1 )

c*********************************************************************72
c
cc R8VEC_ASUM sums the absolute values of the entries of an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    24 January 2015
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the dimension of the vectors.
c
c    Input, double precision V1(N), the vector.
c
c    Output, double precision R8VEC_ASUM, the sum of the absolut
c    values of the entries.
c
      implicit none

      integer n

      integer i
      double precision r8vec_asum
      double precision v1(n)
      double precision value

      value = 0.0D+00
      do i = 1, n
        value = value + abs ( v1(i) )
      end do

      r8vec_asum = value

      return
      end
      subroutine r8vec_bin ( n, x, bin_num, bin_min, bin_max, bin, 
     &  bin_limit )

c*********************************************************************72
c
cc R8VEC_BIN computes bins based on a given R8VEC.
c
c  Discussion:
c
c    The user specifies minimum and maximum bin values, BIN_MIN and
c    BIN_MAX, and the number of bins, BIN_NUM.  This determines a
c    "bin width":
c
c      H = ( BIN_MAX - BIN_MIN ) / BIN_NUM
c
c    so that bin I will count all entries X(J) such that
c
c      BIN_LIMIT(I-1) <= X(J) < BIN_LIMIT(I).
c
c    The array X does NOT have to be sorted.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    29 July 1999
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries of X.
c
c    Input, double precision X(N), an (unsorted) array to be binned.
c
c    Input, integer BIN_NUM, the number of bins.  Two extra bins, #0 and
c    #BIN_NUM+1, count extreme values.
c
c    Input, double precision BIN_MIN, BIN_MAX, define the range and size
c    of the bins.  BIN_MIN and BIN_MAX must be distinct.
c    Normally, BIN_MIN < BIN_MAX, and the documentation will assume
c    this, but proper results will be computed if BIN_MIN > BIN_MAX.
c
c    Output, integer BIN(0:BIN_NUM+1).
c    BIN(0) counts entries of X less than BIN_MIN.
c    BIN(BIN_NUM+1) counts entries greater than or equal to BIN_MAX.
c    For 1 <= I <= BIN_NUM, BIN(I) counts the entries X(J) such that
c      BIN_LIMIT(I-1) <= X(J) < BIN_LIMIT(I).
c    where H is the bin spacing.
c
c    Output, double precision BIN_LIMIT(0:BIN_NUM), the "limits" of the bins.
c    BIN(I) counts the number of entries X(J) such that
c      BIN_LIMIT(I-1) <= X(J) < BIN_LIMIT(I).
c
      implicit none

      integer n
      integer bin_num

      integer bin(0:bin_num+1)
      double precision bin_limit(0:bin_num)
      double precision bin_max
      double precision bin_min
      integer i
      integer j
      double precision t
      double precision x(n)

      if ( bin_max .eq. bin_min ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8VEC_BIN - Fatal error!'
        write ( *, '(a)' ) '  BIN_MIN = BIN_MAX.'
        stop 1
      end if

      do i = 0, bin_num + 1
        bin(i) = 0
      end do

      do i = 1, n

        t = ( x(i) - bin_min ) / ( bin_max - bin_min )

        if ( t .lt. 0.0D+00 ) then
          j = 0
        else if ( 1.0D+00 .le. t ) then
          j = bin_num + 1
        else
          j = 1 + int ( dble ( bin_num ) * t )
        end if

        bin(j) = bin(j) + 1

      end do
c
c  Compute the bin limits.
c
      do i = 0, bin_num
        bin_limit(i) = (   dble ( bin_num - i ) * bin_min   
     &                   + dble (           i ) * bin_max ) 
     &                   / dble ( bin_num     )
      end do

      return
      end
      subroutine r8vec_blend ( n, t1, x1, t2, x2, x )

c*********************************************************************72
c
cc R8VEC_BLEND performs weighted interpolation of two R8VEC's.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    The formula used is:
c
c      x(i) = t * x1(i) + (1-t) * x2(i)
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    16 May 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in each  vector.
c
c    Input, double precision T1, the weight factor for vector 1.
c
c    Input, double precision X1(N), the first vector.
c
c    Input, double precision T2, the weight factor for vector 2.
c
c    Input, double precision X2(N), the second vector.
c
c    Output, double precision X(N), the interpolated or extrapolated value.
c
      implicit none

      integer n

      integer i
      double precision t1
      double precision t2
      double precision x(n)
      double precision x1(n)
      double precision x2(n)

      do i = 1, n
        x(i) = t1 * x1(i) + t2 * x2(i)
      end do

      return
      end
      subroutine r8vec_bracket ( n, x, xval, left, right )

c*********************************************************************72
c
cc R8VEC_BRACKET searches a sorted array for successive brackets of a value.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    If the values in the vector are thought of as defining intervals
c    on the real line, then this routine searches for the interval
c    nearest to or containing the given value.
c
c  Modified:
c
c    24 July 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, length of input array.
c
c    Input, double precision X(N), an array that has been sorted into
c    ascending order.
c
c    Input, double precision XVAL, a value to be bracketed.
c
c    Output, integer LEFT, RIGHT, the results of the search.
c    Either:
c      XVAL < X(1), when LEFT = 1, RIGHT = 2;
c      X(N) < XVAL, when LEFT = N-1, RIGHT = N;
c    or
c      X(LEFT) <= XVAL <= X(RIGHT).
c
      implicit none

      integer n

      integer i
      integer left
      integer right
      double precision x(n)
      double precision xval

      do i = 2, n - 1

        if ( xval .lt. x(i) ) then
          left = i - 1
          right = i
          return
        end if

       end do

      left = n - 1
      right = n

      return
      end
      subroutine r8vec_bracket2 ( n, x, xval, start, left, right )

c*********************************************************************72
c
cc R8VEC_BRACKET2 searches a sorted R8VEC for successive brackets of a value.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    If the values in the vector are thought of as defining intervals
c    on the real line, then this routine searches for the interval
c    containing the given value.
c
c    R8VEC_BRACKET2 is a variation on R8VEC_BRACKET.  It seeks to reduce
c    the search time by allowing the user to suggest an interval that
c    probably contains the value.  The routine will look in that interval
c    and the intervals to the immediate left and right.  If this does
c    not locate the point, a binary search will be carried out on
c    appropriate subportion of the sorted array.
c
c    In the most common case, 1 .le. LEFT .lt. LEFT + 1 = RIGHT .le. N,
c    and X(LEFT) .le. XVAL .le. X(RIGHT).
c
c    Special cases:
c      Value is less than all data values:
c    LEFT = -1, RIGHT = 1, and XVAL .lt. X(RIGHT).
c      Value is greater than all data values:
c    LEFT = N, RIGHT = -1, and X(LEFT) .lt. XVAL.
c      Value is equal to a data value:
c    LEFT = RIGHT, and X(LEFT) = X(RIGHT) = XVAL.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    28 May 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, length of the input array.
c
c    Input, double precision X(N), an array that has been sorted into
c    ascending order.
c
c    Input, double precision XVAL, a value to be bracketed by entries of X.
c
c    Input, integer START, between 1 and N, specifies that XVAL
c    is likely to be in the interval:
c
c      [ X(START), X(START+1) ]
c
c    or, if not in that interval, then either
c
c      [ X(START+1), X(START+2) ]
c    or
c      [ X(START-1), X(START) ].
c
c    Output, integer LEFT, RIGHT, the results of the search.
c
      implicit none

      integer n

      integer high
      integer left
      integer low
      integer right
      integer start
      double precision x(n)
      double precision xval
c
c  Check.
c
      if ( n .lt. 1 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8VEC_BRACKET2 - Fatal error!'
        write ( *, '(a)' ) '  N .lt. 1.'
        stop 1
      end if

      if ( start .lt. 1 .or. n .lt. start ) then
        start = ( n + 1 ) / 2
      end if
c
c  XVAL = X(START)?
c
      if ( x(start) .eq. xval ) then

        left = start
        right = start
        return
c
c  X(START) .lt. XVAL?
c
      else if ( x(start) .lt. xval ) then
c
c  X(START) = X(N) .lt. XVAL .lt. Infinity?
c
        if ( n .lt. start + 1 ) then

          left = start
          right = -1
          return
c
c  XVAL = X(START+1)?
c
        else if ( xval .eq. x(start+1) ) then

          left = start + 1
          right = start + 1
          return
c
c  X(START) .lt. XVAL .lt. X(START+1)?
c
        else if ( xval .lt. x(start+1) ) then

          left = start
          right = start + 1
          return
c
c  X(START+1) = X(N) .lt. XVAL .lt. Infinity?
c
        else if ( n .lt. start + 2 ) then

          left = start + 1
          right = -1
          return
c
c  XVAL = X(START+2)?
c
        else if ( xval .eq. x(start+2) ) then

          left = start + 2
          right = start + 2
          return
c
c  X(START+1) .lt. XVAL .lt. X(START+2)?
c
        else if ( xval .lt. x(start+2) ) then

          left = start + 1
          right = start + 2
          return
c
c  Binary search for XVAL in [ X(START+2), X(N) ],
c  where XVAL is guaranteed to be greater than X(START+2).
c
        else

          low = start + 2
          high = n
          call r8vec_bracket ( high + 1 - low, x(low), xval, left,
     &      right )
          left = left + low - 1
          right = right + low - 1

        end if
c
c  -Infinity .lt. XVAL .lt. X(START) = X(1).
c
      else if ( start .eq. 1 ) then

        left = -1
        right = start
        return
c
c  XVAL = X(START-1)?
c
      else if ( xval .eq. x(start-1) ) then

        left = start - 1
        right = start - 1
        return
c
c  X(START-1) .lt. XVAL .lt. X(START)?
c
      else if ( x(start-1) .le. xval ) then

        left = start - 1
        right = start
        return
c
c  Binary search for XVAL in [ X(1), X(START-1) ],
c  where XVAL is guaranteed to be less than X(START-1).
c
      else

        low = 1
        high = start - 1
        call r8vec_bracket ( high + 1 - low, x(1), xval, left, right )

      end if

      return
      end
      subroutine r8vec_bracket3 ( n, t, tval, left )

c*********************************************************************72
c
cc R8VEC_BRACKET3 finds the interval containing or nearest a given value.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    The routine always returns the index LEFT of the sorted array
c    T with the property that either
c    *  T is contained in the interval [ T(LEFT), T(LEFT+1) ], or
c    *  T .lt. T(LEFT) = T(1), or
c    *  T > T(LEFT+1) = T(N).
c
c    The routine is useful for interpolation problems, where
c    the abscissa must be located within an interval of data
c    abscissas for interpolation, or the "nearest" interval
c    to the (extreme) abscissa must be found so that extrapolation
c    can be carried out.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    28 May 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, length of the input array.
c
c    Input, double precision T(N), an array that has been sorted
c    into ascending order.
c
c    Input, double precision TVAL, a value to be bracketed by entries of T.
c
c    Input/output, integer LEFT.
c    On input, if 1 .le. LEFT .le. N-1, LEFT is taken as a suggestion for the
c    interval [ T(LEFT), T(LEFT+1) ] in which TVAL lies.  This interval
c    is searched first, followed by the appropriate interval to the left
c    or right.  After that, a binary search is used.
c    On output, LEFT is set so that the interval [ T(LEFT), T(LEFT+1) ]
c    is the closest to TVAL; it either contains TVAL, or else TVAL
c    lies outside the interval [ T(1), T(N) ].
c
      implicit none

      integer n

      integer high
      integer left
      integer low
      integer mid
      double precision t(n)
      double precision tval
c
c  Check the input data.
c
      if ( n .lt. 2 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8VEC_BRACKET3 - Fatal error!'
        write ( *, '(a)' ) '  N must be at least 2.'
        stop 1
      end if
c
c  If LEFT is not between 1 and N-1, set it to the middle value.
c
      if ( left .lt. 1 .or. n - 1 .lt. left ) then
        left = ( n + 1 ) / 2
      end if
c
c  CASE 1: TVAL .lt. T(LEFT):
c  Search for TVAL in [T(I), T(I+1)] for intervals I = 1 to LEFT-1.
c
      if ( tval .lt. t(left) ) then

        if ( left .eq. 1 ) then
          return
        else if ( left .eq. 2 ) then
          left = 1
          return
        else if ( t(left-1) .le. tval ) then
          left = left - 1
          return
        else if ( tval .le. t(2) ) then
          left = 1
          return
        end if
c
c  ...Binary search for TVAL in [T(I), T(I+1)] for intervals I = 2 to LEFT-2.
c
        low = 2
        high = left - 2

10      continue

          if ( low .eq. high ) then
            left = low
            return
          end if

          mid = ( low + high + 1 ) / 2

          if ( t(mid) .le. tval ) then
            low = mid
          else
            high = mid - 1
          end if

        go to 10
c
c  CASE2: T(LEFT+1) .lt. TVAL:
c  Search for TVAL in [T(I),T(I+1)] for intervals I = LEFT+1 to N-1.
c
      else if ( t(left+1) .lt. tval ) then

        if ( left .eq. n - 1 ) then
          return
        else if ( left .eq. n - 2 ) then
          left = left + 1
          return
        else if ( tval .le. t(left+2) ) then
          left = left + 1
          return
        else if ( t(n-1) .le. tval ) then
          left = n - 1
          return
        end if
c
c  ...Binary search for TVAL in [T(I), T(I+1)] for intervals I = LEFT+2 to N-2.
c
        low = left + 2
        high = n - 2

20      continue

          if ( low .eq. high ) then
            left = low
            return
          end if

          mid = ( low + high + 1 ) / 2

          if ( t(mid) .le. tval ) then
            low = mid
          else
            high = mid - 1
          end if

        go to 20
c
c  CASE3: T(LEFT) .le. TVAL .le. T(LEFT+1):
c  T is in [T(LEFT), T(LEFT+1)], as the user said it might be.
c
      else

      end if

      return
      end
      subroutine r8vec_bracket4 ( nt, t, ns, s, left )

c*********************************************************************72
c
cc R8VEC_BRACKET4 finds the nearest interval to each of a vector of values.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    The routine always returns the index LEFT of the sorted array
c    T with the property that either
c    *  T is contained in the interval [ T(LEFT), T(LEFT+1) ], or
c    *  T .lt. T(LEFT) = T(1), or
c    *  T > T(LEFT+1) = T(NT).
c
c    The routine is useful for interpolation problems, where
c    the abscissa must be located within an interval of data
c    abscissas for interpolation, or the "nearest" interval
c    to the (extreme) abscissa must be found so that extrapolation
c    can be carried out.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    28 May 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer NT, length of the input array.
c
c    Input, double precision T(NT), an array that has been sorted
c    into ascending order.
c
c    Input, integer NS, the number of points to be bracketed.
c
c    Input, double precision S(NS), values to be bracketed by entries of T.
c
c    Output, integer LEFT(NS).
c    LEFT(I) is set so that the interval [ T(LEFT(I)), T(LEFT(I)+1) ]
c    is the closest to S(I); it either contains S(I), or else S(I)
c    lies outside the interval [ T(1), T(NT) ].
c
      implicit none

      integer ns
      integer nt

      integer high
      integer i
      integer left(ns)
      integer low
      integer mid
      double precision s(ns)
      double precision t(nt)
c
c  Check the input data.
c
      if ( nt .lt. 2 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8VEC_BRACKET4 - Fatal error!'
        write ( *, '(a)' ) '  NT must be at least 2.'
        stop 1
      end if

      do i = 1, ns

        left(i) = ( nt + 1 ) / 2
c
c  CASE 1: S .lt. T(LEFT):
c  Search for S in [T(I), T(I+1)] for intervals I = 1 to LEFT-1.
c
        if ( s(i) .lt. t(left(i)) ) then

          if ( left(i) .eq. 1 ) then
            go to 100
          else if ( left(i) .eq. 2 ) then
            left(i) = 1
            go to 100
          else if ( t(left(i)-1) .le. s(i) ) then
            left(i) = left(i) - 1
            go to 100
          else if ( s(i) .le. t(2) ) then
            left(i) = 1
            go to 100
          end if
c
c  ...Binary search for S in [T(I), T(I+1)] for intervals I = 2 to LEFT-2.
c
          low = 2
          high = left(i) - 2

10        continue

            if ( low .eq. high ) then
              left(i) = low
              go to 20
            end if

            mid = ( low + high + 1 ) / 2

            if ( t(mid) .le. s(i) ) then
              low = mid
            else
              high = mid - 1
            end if

          go to 10

20        continue
c
c  CASE2: T(LEFT+1) .lt. S:
c  Search for S in [T(I),T(I+1)] for intervals I = LEFT+1 to N-1.
c
        else if ( t(left(i)+1) .lt. s(i) ) then

          if ( left(i) .eq. nt - 1 ) then
            go to 100
          else if ( left(i) .eq. nt - 2 ) then
            left(i) = left(i) + 1
            go to 100
          else if ( s(i) .le. t(left(i)+2) ) then
            left(i) = left(i) + 1
            go to 100
          else if ( t(nt-1) .le. s(i) ) then
            left(i) = nt - 1
            go to 100
          end if
c
c  ...Binary search for S in [T(I), T(I+1)] for intervals I = LEFT+2 to NT-2.
c
          low = left(i) + 2
          high = nt - 2

30        continue

            if ( low .eq. high ) then
              left(i) = low
              go to 40
            end if

            mid = ( low + high + 1 ) / 2

            if ( t(mid) .le. s(i) ) then
              low = mid
            else
              high = mid - 1
            end if

          go to 30

40        continue
c
c  CASE3: T(LEFT) .le. S .le. T(LEFT+1):
c  S is in [T(LEFT), T(LEFT+1)].
c
        else

        end if

100     continue

      end do

      return
      end
      function r8vec_bracket5 ( nd, xd, xi )

c*********************************************************************72
c
cc R8VEC_BRACKET5 brackets data between successive entries of a sorted R8VEC.
c
c  Discussion:
c
c    We assume XD is sorted.
c
c    If XI is contained in the interval [XD(1),XD(N)], then the returned 
c    value B indicates that XI is contained in [ XD(B), XD(B+1) ].
c
c    If XI is not contained in the interval [XD(1),XD(N)], then B = -1.
c
c    This code implements a version of binary search which is perhaps more
c    understandable than the usual ones.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    14 October 2012
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer ND, the number of data values.
c
c    Input, double precision XD(N), the sorted data.
c
c    Input, double precision XD, the query value.
c
c    Output, integer R8VEC_BRACKET5, the bracket information.
c
      implicit none

      integer nd

      integer b
      integer l
      integer m
      integer r
      integer r8vec_bracket5
      double precision xd(nd)
      double precision xi

      if ( xi .lt. xd(1) .or. xd(nd) .lt. xi ) then

        b = -1

      else

        l = 1
        r = nd

10      continue

        if ( l + 1 .lt. r ) then
          m = ( l + r ) / 2
          if ( xi .lt. xd(m) ) then
            r = m
          else
            l = m
          end if
          go to 10
        end if

        b = l

      end if

      r8vec_bracket5 = b

      return
      end
      subroutine r8vec_bracket6 ( nd, xd, ni, xi, b )

c*********************************************************************72
c
cc R8VEC_BRACKET6 brackets data between successive entries of a sorted R8VEC.
c
c  Discussion:
c
c    We assume XD is sorted.
c
c    If XI(I) is contained in the interval [XD(1),XD(N)], then the value of
c    B(I) indicates that XI(I) is contained in [ XD(B(I)), XD(B(I)+1) ].
c
c    If XI(I) is not contained in the interval [XD(1),XD(N)], then B(I) = -1.
c
c    This code implements a version of binary search which is perhaps more
c    understandable than the usual ones.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    14 October 2012
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer ND, the number of data values.
c
c    Input, double precision XD(N), the sorted data.
c
c    Input, integer NI, the number of inquiry values.
c
c    Input, double precision XD(NI), the query values.
c
c    Output, integer B(NI), the bracket information.
c
      implicit none

      integer nd
      integer ni

      integer b(ni)
      integer i
      integer l
      integer m
      integer r
      double precision xd(nd)
      double precision xi(ni)

      do i = 1, ni

        if ( xi(i) .lt. xd(1) .or. xd(nd) .lt. xi(i) ) then

          b(i) = -1

        else

          l = 1
          r = nd

10        continue

          if ( l + 1 .lt. r ) then
            m = ( l + r ) / 2
            if ( xi(i) .lt. xd(m) ) then
              r = m
            else
              l = m
            end if
            go to 10
          end if

          b(i) = l

        end if

      end do

      return
      end
      subroutine r8vec_ceiling ( n, r8vec, ceilingvec )

c*********************************************************************72
c
cc R8VEC_CEILING rounds "up" (towards +infinity) entries of an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Example:
c
c    R8    Value
c
c   -1.1  -1.0
c   -1.0  -1.0
c   -0.9   0.0
c    0.0   0.0
c    5.0   5.0
c    5.1   6.0
c    5.9   6.0
c    6.0   6.0
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    28 May 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries.
c
c    Input, double precision R8VEC(N), the vector.
c
c    Output, double precision CEILINGVEC(N), the rounded values.
c
      implicit none

      integer n

      double precision ceilingvec(n)
      integer i
      double precision r8vec(n)
      double precision value

      do i = 1, n

        value = dble ( int ( r8vec(i) ) )

        if ( value .lt. r8vec(i) ) then
          value = value + 1.0D+00
        end if

        ceilingvec(i) = value

      end do

      return
      end
      subroutine r8vec_chebyspace ( n, a, b, x )

c*********************************************************************72
c
cc R8VEC_CHEBYSPACE creates a vector of Chebyshev spaced values in [A,B].
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    08 June 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the vector.
c
c    Input, double precision A, B, the interval.
c
c    Output, double precision X(N), a vector of Chebyshev spaced data.
c
      implicit none

      integer n

      double precision a
      double precision b
      double precision c
      integer i
      double precision r8_pi
      parameter ( r8_pi = 3.141592653589793D+00 )
      double precision theta
      double precision x(n)

      if ( n .eq. 1 ) then

        x(1) = ( a + b ) / 2.0D+00

      else

        do i = 1, n

          theta = dble ( n - i ) * r8_pi / dble ( n - 1 )

          c = cos ( theta )

          if ( mod ( n, 2 ) .eq. 1 ) then
            if ( 2 * i - 1 .eq. n ) then
              c = 0.0D+00
            end if
          end if

          x(i) = ( ( 1.0D+00 - c ) * a
     &           + ( 1.0D+00 + c ) * b ) 
     &           /   2.0D+00

        end do

      end if

      return
      end
      subroutine r8vec_cheby1space ( n, a, b, x )

c*********************************************************************72
c
cc R8VEC_CHEBY1SPACE creates Type 1 Chebyshev spaced values in [A,B].
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    17 September 2012
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the vector.
c
c    Input, double precision A, B, the first and last entries.
c
c    Output, double precision X(N), a vector of Chebyshev spaced data.
c
      implicit none
 
      integer n

      double precision a
      double precision b
      double precision c
      integer i
      double precision r8_pi
      parameter ( r8_pi = 3.141592653589793D+00 )
      double precision theta
      double precision x(n)

      if ( n .eq. 1 ) then

        x(1) = ( a + b ) / 2.0D+00

      else

        do i = 1, n

          theta = dble ( 2 * ( n - i ) + 1 ) * r8_pi / dble ( 2 * n )

          c = cos ( theta )

          if ( mod ( n, 2 ) .eq. 1 ) then
            if ( 2 * i - 1 .eq. n ) then
              c = 0.0D+00
            end if
          end if

          x(i) = ( ( 1.0D+00 - c ) * a   
     &           + ( 1.0D+00 + c ) * b ) 
     &           /   2.0D+00

        end do

      end if

      return
      end
      subroutine r8vec_cheby2space ( n, a, b, x )

c*********************************************************************72
c
cc R8VEC_CHEBY2SPACE creates Type 2 Chebyshev spaced values in [A,B].
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    17 September 2012
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the vector.
c
c    Input, double precision A, B, the first and last entries.
c
c    Output, double precision X(N), a vector of Chebyshev spaced data.
c
      implicit none

      integer n

      double precision a
      double precision b
      double precision c
      integer i
      double precision r8_pi
      parameter ( r8_pi = 3.141592653589793D+00 )
      double precision theta
      double precision x(n)

      if ( n .eq. 1 ) then

        x(1) = ( a + b ) / 2.0D+00

      else

        do i = 1, n

          theta = dble ( n - i ) * r8_pi / dble ( n - 1 )

          c = cos ( theta )

          if ( mod ( n, 2 ) .eq. 1 ) then
            if ( 2 * i - 1 .eq. n ) then
              c = 0.0D+00
            end if
          end if

          x(i) = ( ( 1.0D+00 - c ) * a   
     &           + ( 1.0D+00 + c ) * b ) 
     &           /   2.0D+00

        end do

      end if

      return
      end
      subroutine r8vec_circular_variance ( n, x, circular_variance )

c*********************************************************************72
c
cc R8VEC_CIRCULAR_VARIANCE returns the circular variance of an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    29 May 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the vector.
c
c    Input, double precision X(N), the vector whose variance is desired.
c
c    Output, double precision CIRCULAR VARIANCE, the circular variance
c    of the vector entries.
c
      implicit none

      integer n

      double precision circular_variance
      double precision csum
      integer i
      double precision mean
      double precision ssum
      double precision x(n)

      call r8vec_mean ( n, x, mean )

      csum = 0.0D+00
      do i = 1, n
        csum = csum + cos ( x(i) - mean )
      end do

      ssum = 0.0D+00
      do i = 1, n
        ssum = ssum + sin ( x(i) - mean )
      end do

      circular_variance = csum * csum + ssum * ssum

      circular_variance = sqrt ( circular_variance ) / dble ( n )

      circular_variance = 1.0D+00 - circular_variance

      return
      end
      subroutine r8vec_compare ( n, a1, a2, isgn )

c*********************************************************************72
c
cc R8VEC_COMPARE compares two R8VEC's.
c
c  Discussion:
c
c    An R8VEC is a vector of R8 values.
c
c    The lexicographic ordering is used.
c
c  Example:
c
c    Input:
c
c      A1 = ( 2.0, 6.0, 2.0 )
c      A2 = ( 2.0, 8.0, 12.0 )
c
c    Output:
c
c      ISGN = -1
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    23 February 1999
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the vectors.
c
c    Input, double precision A1(N), A2(N), the vectors to be compared.
c
c    Output, integer ISGN, the results of the comparison:
c    -1, A1 < A2,
c     0, A1 = A2,
c    +1, A1 > A2.
c
      implicit none

      integer n

      double precision a1(n)
      double precision a2(n)
      integer isgn
      integer k

      isgn = 0

      k = 1

10    continue

      if ( k .le. n ) then

        if ( a1(k) .lt. a2(k) ) then
          isgn = -1
          return
        else if ( a2(k) .lt. a1(k) ) then
          isgn = + 1
          return
        end if

        k = k + 1

        go to 10

      end if

      return
      end
      subroutine r8vec_concatenate ( n1, a, n2, b, c )

c*********************************************************************72
c
cc R8VEC_CONCATENATE concatenates two R8VEC's.
c
c  Discussion:
c
c    An R8VEC is a vector of R8 values.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    22 November 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N1, the number of entries in the first vector.
c
c    Input, double precision A(N1), the first vector.
c
c    Input, integer N2, the number of entries in the second vector.
c
c    Input, double precision B(N2), the second vector.
c
c    Output, double precision C(N1+N2), the concatenation of A and B.
c
      implicit none

      integer n1
      integer n2

      double precision a(n1)
      double precision b(n2)
      double precision c(n1+n2)
      integer i

      do i = 1, n1
        c(i) = a(i)
      end do

      do i = 1, n2
        c(n1+i) = b(i)
      end do

      return
      end
      subroutine r8vec_convolution ( m, x, n, y, z )

c*********************************************************************72
c
cc R8VEC_CONVOLUTION returns the convolution of two R8VEC's.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    The I-th entry of the convolution can be formed by summing the products 
c    that lie along the I-th diagonal of the following table:
c
c    Y3 | 3   4   5   6   7
c    Y2 | 2   3   4   5   6
c    Y1 | 1   2   3   4   5
c       +------------------
c        X1  X2  X3  X4  X5
c
c    which will result in:
c
c    Z = ( X1 * Y1,
c          X1 * Y2 + X2 * Y1,
c          X1 * Y3 + X2 * Y2 + X3 * Y1,
c                    X2 * Y3 + X3 * Y2 + X4 * Y1,
c                              X3 * Y3 + X4 * Y2 + X5 * Y1,
c                                        X4 * Y3 + X5 * Y2,
c                                                  X5 * Y3 )
c            
c  Example:
c
c    Input:
c
c      X = (/ 1, 2, 3, 4 /)
c      Y = (/ -1, 5, 3 /)
c
c    Output:
c
c      Z = (/ -1, 3, 10, 17, 29, 12 /)
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    05 May 2012
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the dimension of X.
c
c    Input, double precision X(M), the first vector to be convolved.
c
c    Input, integer N, the dimension of Y.
c
c    Input, double precision Y(N), the second vector to be convolved.
c
c    Output, double precision Z(M+N-1), the convolution of X and Y.
c
      implicit none

      integer m
      integer n

      integer i
      integer j
      double precision x(m)
      double precision y(n)
      double precision z(m+n-1)

      do i = 1, m + n - 1
        z(i) = 0.0D+00
      end do

      do j = 1, n
        do i = 0, m - 1
          z(j+i) = z(j+i) + x(i+1) * y(j)
        end do
      end do

      return
      end
      subroutine r8vec_convolution_circ ( n, x, y, z )

c*********************************************************************72
c
cc R8VEC_CONVOLUTION_CIRC returns the discrete circular convolution of two R8VEC's.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    The formula used is:
c
c      z(1+m) = xCCy(m) = sum ( 0 <= k <= n-1 ) x(1+k) * y(1+m-k)
c
c    Here, if the index of Y becomes nonpositive, it is "wrapped around"
c    by having N added to it.
c
c    The circular convolution is equivalent to multiplication of Y by a
c    circulant matrix formed from the vector X.
c
c  Example:
c
c    Input:
c
c      X = (/ 1, 2, 3, 4 /)
c      Y = (/ 1, 2, 4, 8 /)
c
c    Output:
c
c      Circulant form:
c
c      Z = ( 1 4 3 2 )   ( 1 )
c          ( 2 1 4 3 )   ( 2 )
c          ( 3 2 1 4 ) * ( 4 )
c          ( 4 3 2 1 )   ( 8 )
c
c      The formula:
c
c      Z = (/ 1*1 + 2*8 + 3*4 + 4*2,
c             1*2 + 2*1 + 3*8 + 4*4,
c             1*4 + 2*2 + 3*1 + 4*8,
c             1*8 + 2*4 + 3*2 + 4*1 /)
c
c      Result:
c
c      Z = (/ 37, 44, 43, 26 /)
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    29 May 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the dimension of the vectors.
c
c    Input, double precision X(N), Y(N), the vectors to be convolved.
c
c    Output, double precision Z(N), the circular convolution of X and Y.
c
      implicit none

      integer n

      integer i
      integer m
      double precision x(n)
      double precision y(n)
      double precision z(n)

      do m = 1, n
        z(m) = 0.0D+00
        do i = 1, m
          z(m) = z(m) + x(i) * y(m+1-i)
        end do
        do i = m + 1, n
          z(m) = z(m) + x(i) * y(n+m+1-i)
        end do
      end do

      return
      end
      subroutine r8vec_copy ( n, a1, a2 )

c*********************************************************************72
c
cc R8VEC_COPY copies an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8 values.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    13 August 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the length of the vectors.
c
c    Input, double precision A1(N), the vector to be copied.
c
c    Output, double precision A2(N), a copy of A1.
c
      implicit none

      integer n

      double precision a1(n)
      double precision a2(n)
      integer i

      do i = 1, n
        a2(i) = a1(i)
      end do

      return
      end
      subroutine r8vec_correlation ( n, x, y, correlation )

c**********************************************************************72
c
cc R8VEC_CORRELATION returns the correlation of two R8VEC's.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    If X and Y are two nonzero vectors of length N, then
c
c      correlation = (x/||x||)' (y/||y||)
c
c    It is the cosine of the angle between the two vectors.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    21 August 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the dimension of the vectors.
c
c    Input, double precision X(N), Y(N), the vectors to be convolved.
c
c    Output, double precision CORRELATION, the correlation of X and Y.
c
      implicit none

      integer n

      double precision correlation
      double precision r8vec_dot_product
      double precision r8vec_norm
      double precision x(n)
      double precision x_norm
      double precision xy_dot
      double precision y(n)
      double precision y_norm

      x_norm = r8vec_norm ( n, x )
      y_norm = r8vec_norm ( n, y )
      xy_dot = r8vec_dot_product ( n, x, y )

      if ( x_norm .eq. 0.0D+00 .or. y_norm .eq. 0.0D+00 ) then
        correlation = 0.0D+00
      else
        correlation = xy_dot / x_norm / y_norm
      end if

      return
      end
      function r8vec_covar ( n, x, y )

c*********************************************************************72
c
cc R8VEC_COVAR computes the covariance of two vectors.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    20 April 2013
c
c  Author:
c
c    John Burkardt.
c
c  Parameters:
c
c    Input, double precision X(N), Y(N), the two vectors.
c
c    Input, integer N, the dimension of the two vectors.
c
c    Output, double precision R8VEC_COVAR, the covariance of the two vectors.
c
      implicit none

      integer n

      integer i
      double precision r8vec_covar
      double precision value
      double precision x(n)
      double precision x_average
      double precision y(n)
      double precision y_average

      x_average = 0.0D+00
      do i = 1, n
        x_average = x_average + x(i)
      end do
      x_average = x_average / dble ( n )

      y_average = 0.0D+00
      do i = 1, n
        y_average = y_average + y(i)
      end do
      y_average = y_average / dble ( n )

      value = 0.0D+00
      do i = 1, n
        value = value + ( x(i) - x_average ) * ( y(i) - y_average )
      end do

      r8vec_covar = value / dble ( n - 1 )

      return
      end
      function r8vec_cross_product_2d ( v1, v2 )

c*********************************************************************72
c
cc R8VEC_CROSS_PRODUCT_2D finds the cross product of a pair of vectors in 2D.
c
c  Discussion:
c
c    Strictly speaking, the vectors V1 and V2 should be considered
c    to lie in a 3D space, both having Z coordinate zero.  The cross
c    product value V3 then represents the standard cross product vector
c    (0,0,V3).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    27 October 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision V1(2), V2(2), the vectors.
c
c    Output, double precision R8VEC_CROSS_PRODUCT_2D, the cross product.
c
      implicit none

      double precision r8vec_cross_product_2d
      double precision v1(2)
      double precision v2(2)

      r8vec_cross_product_2d = v1(1) * v2(2) - v1(2) * v2(1)

      return
      end
      function r8vec_cross_product_affine_2d ( v0, v1, v2 )

c*********************************************************************72
c
cc R8VEC_CROSS_PRODUCT_AFFINE_2D finds the affine cross product in 2D.
c
c  Discussion:
c
c    Strictly speaking, the vectors V1 and V2 should be considered
c    to lie in a 3D space, both having Z coordinate zero.  The cross
c    product value V3 then represents the standard cross product vector
c    (0,0,V3).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    27 October 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision V0(2), the base vector.
c
c    Input, double precision V1(2), V2(2), the vectors.
c
c    Output, double precision R8VEC_CROSS_PRODUCT_AFFINE_2D,
c    the cross product (V1-V0) x (V2-V0).
c
      implicit none

      double precision r8vec_cross_product_affine_2d
      double precision v0(2)
      double precision v1(2)
      double precision v2(2)

      r8vec_cross_product_affine_2d =
     &    ( v1(1) - v0(1) ) * ( v2(2) - v0(2) )
     &  - ( v2(1) - v0(1) ) * ( v1(2) - v0(2) )

      return
      end
      subroutine r8vec_cross_product_3d ( v1, v2, v3 )

c*********************************************************************72
c
cc R8VEC_CROSS_PRODUCT_3D computes the cross product of two R8VEC's in 3D.
c
c  Discussion:
c
c    An R8VEC is a vector of R8 values.
c
c    The cross product in 3D can be regarded as the determinant of the
c    symbolic matrix:
c
c          |  i  j  k |
c      det | x1 y1 z1 |
c          | x2 y2 z2 |
c
c      = ( y1 * z2 - z1 * y2 ) * i
c      + ( z1 * x2 - x1 * z2 ) * j
c      + ( x1 * y2 - y1 * x2 ) * k
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    30 August 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision V1(3), V2(3), the two vectors.
c
c    Output, double precision V3(3), the cross product vector.
c
      implicit none

      integer dim_num
      parameter ( dim_num = 3 )

      double precision v1(dim_num)
      double precision v2(dim_num)
      double precision v3(dim_num)

      v3(1) = v1(2) * v2(3) - v1(3) * v2(2)
      v3(2) = v1(3) * v2(1) - v1(1) * v2(3)
      v3(3) = v1(1) * v2(2) - v1(2) * v2(1)

      return
      end
      subroutine r8vec_cross_product_affine_3d ( v0, v1, v2, v3 )

c*********************************************************************72
c
cc R8VEC_CROSS_PRODUCT_AFFINE_3D computes the affine cross product in 3D.
c
c  Discussion:
c
c    The cross product in 3D can be regarded as the determinant of the
c    symbolic matrix:
c
c          |  i  j  k |
c      det | x1 y1 z1 |
c          | x2 y2 z2 |
c
c      = ( y1 * z2 - z1 * y2 ) * i
c      + ( z1 * x2 - x1 * z2 ) * j
c      + ( x1 * y2 - y1 * x2 ) * k
c
c    Here, we use V0 as the base of an affine system so we compute
c    the cross product of (V1-V0) and (V2-V0).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    27 October 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision V0(3), the base vector.
c
c    Input, double precision V1(3), V2(3), the two vectors.
c
c    Output, double precision V3(3), the cross product vector
c    ( V1-V0) x (V2-V0).
c
      implicit none

      double precision v0(3)
      double precision v1(3)
      double precision v2(3)
      double precision v3(3)

      v3(1) = ( v1(2) - v0(2) ) * ( v2(3) - v0(3) )
     &      - ( v2(2) - v0(2) ) * ( v1(3) - v0(3) )

      v3(2) = ( v1(3) - v0(3) ) * ( v2(1) - v0(1) )
     &      - ( v2(3) - v0(3) ) * ( v1(1) - v0(1) )

      v3(3) = ( v1(1) - v0(1) ) * ( v2(2) - v0(2) )
     &      - ( v2(1) - v0(1) ) * ( v1(2) - v0(2) )

      return
      end
      subroutine r8vec_cum ( n, a, a_cum )

c*********************************************************************72
c
cc R8VEC_CUM computes the cumulutive sums of an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8 values.
c
c    Input:
c
c      A = (/ 1.0, 2.0, 3.0, 4.0 /)
c
c    Output:
c
c      A_CUM = (/ 1.0, 3.0, 6.0, 10.0 /)
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    07 May 2012
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the vector.
c
c    Input, double precision A(N), the vector to be summed.
c
c    Output, double precision A_CUM(N), the cumulative sums.
c
      implicit none

      integer n

      double precision a(n)
      double precision a_cum(n)
      integer i

      a_cum(1) = a(1)

      do i = 2, n
        a_cum(i) = a_cum(i-1) + a(i)
      end do

      return
      end
      subroutine r8vec_cum0 ( n, a, a_cum )

c*********************************************************************72
c
cc R8VEC_CUM0 computes the cumulutive sums of an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8 values.
c
c    Input:
c
c      A = (/ 1.0, 2.0, 3.0, 4.0 /)
c
c    Output:
c
c      A_CUM = (/ 0.0, 1.0, 3.0, 6.0, 10.0 /)
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    07 May 2012
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the vector.
c
c    Input, double precision A(N), the vector to be summed.
c
c    Output, double precision A_CUM(0:N), the cumulative sums.
c
      implicit none

      integer n

      double precision a(n)
      double precision a_cum(0:n)
      integer i

      a_cum(0) = 0.0D+00

      do i = 1, n
        a_cum(i) = a_cum(i-1) + a(i)
      end do

      return
      end
      subroutine r8vec_dif ( n, h, cof )

c*********************************************************************72
c
cc R8VEC_DIF computes coefficients for estimating the N-th derivative.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    The routine computes the N+1 coefficients for a centered finite difference
c    estimate of the N-th derivative of a function.
c
c    The estimate has the form
c
c      FDIF(N,X) = Sum (I = 0 to N) COF(I) * F ( X(I) )
c
c    To understand the computation of the coefficients, it is enough
c    to realize that the first difference approximation is
c
c      FDIF(1,X) = F(X+DX) - F(X-DX) ) / (2*DX)
c
c    and that the second difference approximation can be regarded as
c    the first difference approximation repeated:
c
c      FDIF(2,X) = FDIF(1,X+DX) - FDIF(1,X-DX) / (2*DX)
c         = F(X+2*DX) - 2 F(X) + F(X-2*DX) / (4*DX)
c
c    and so on for higher order differences.
c
c    Thus, the next thing to consider is the integer coefficients of
c    the sampled values of F, which are clearly the Pascal coefficients,
c    but with an alternating negative sign.  In particular, if we
c    consider row I of Pascal's triangle to have entries j = 0 through I,
c    then P(I,J) = P(I-1,J-1) - P(I-1,J), where P(*,-1) is taken to be 0,
c    and P(0,0) = 1.
c
c       1
c      -1  1
c       1 -2   1
c      -1  3  -3   1
c       1 -4   6  -4   1
c      -1  5 -10  10  -5  1
c       1 -6  15 -20  15 -6 1
c
c    Next, note that the denominator of the approximation for the
c    N-th derivative will be ( 2 * DX ) ** N.
c
c    And finally, consider the location of the N+1 sampling
c    points for F:
c
c      X-N*DX, X-(N-2)*DX, X-(N-4)*DX, ..., X+(N-4)*DX, X+(N-2*DX), X+N*DX.
c
c    Thus, a formula for evaluating FDIF(N,X) is
c
c      fdif = 0.0
c      do i = 0, n
c        xi = x + (2*i-n) * h
c        fdif = fdif + cof(i) * f(xi)
c      end do
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    29 May 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the order of the derivative to be approximated.
c    N must be 0 or greater.
c
c    Input, double precision H, the half spacing between points.
c    H must be positive.
c
c    Output, double precision COF(0:N), the coefficients needed to approximate
c    the N-th derivative of a function F.
c
      implicit none

      integer n

      double precision cof(0:n)
      double precision h
      integer i
      integer j

      if ( n .lt. 0 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8VEC_DIF - Fatal error!'
        write ( *, '(a,i8)' ) '  Derivative order N = ', n
        write ( *, '(a)' ) '  but N must be at least 0.'
        stop 1
      end if

      if ( h .le. 0.0D+00 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8VEC_DIF - Fatal error!'
        write ( *, '(a,g14.6)' )
     &    '  The half sampling spacing is H = ', h
        write ( *, '(a)' ) '  but H must be positive.'
        stop 1
      end if

      do i = 0, n

        cof(i) = 1.0D+00

        do j = i - 1, 1, -1
          cof(j) = -cof(j) + cof(j-1)
        end do

        if ( 0 .lt. i ) then
          cof(0) = -cof(0)
        end if

      end do

      do i = 0, n
        cof(i) = cof(i) / ( 2.0D+00 * h ) ** n
      end do

      return
      end
      function r8vec_diff_dot_product ( n, u1, v1, u2, v2 )

c*********************************************************************72
c
cc R8VEC_DIFF_DOT_PRODUCT: dot product of a pair of R8VEC differences.
c
c  Discussion:
c
c    An R8VEC is a vector of R8 values.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    31 March 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the dimension of the vectors.
c
c    Input, double precision U1(N), V1(N), defines the vector U1-V1.
c
c    Input, double precision U2(N), V2(N), defines the vector U2-V2.
c
c    Output, double precision R8VEC_DIFF_DOT_PRODUCT, the dot product 
c    of (U1-V1)*(U2-V2).
c
      implicit none

      integer n

      integer i
      double precision r8vec_diff_dot_product
      double precision u1(n)
      double precision u2(n)
      double precision v1(n)
      double precision v2(n)
      double precision value

      value = 0.0D+00
      do i = 1, n
        value = value + ( u1(i) - v1(i) ) * ( u2(i) - v2(i) )
      end do

      r8vec_diff_dot_product = value

      return
      end
      function r8vec_diff_norm ( n, a, b )

c*********************************************************************72
c
cc R8VEC_DIFF_NORM returns the L2 norm of the difference of R8VEC's.
c
c  Discussion:
c
c    An R8VEC is a vector of R8 values.
c
c    The vector L2 norm is defined as:
c
c      R8VEC_NORM_L2 = sqrt ( sum ( 1 <= I <= N ) A(I)^2 ).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    24 June 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in A.
c
c    Input, double precision A(N), B(N), the vectors.
c
c    Output, double precision R8VEC_DIFF_NORM, the L2 norm of A - B.
c
      implicit none

      integer n

      double precision a(n)
      double precision b(n)
      integer i
      double precision r8vec_diff_norm
      double precision value

      value = 0.0D+00
      do i = 1, n
        value = value + ( a(i) - b(i) )**2
      end do
      value = sqrt ( value )

      r8vec_diff_norm = value

      return
      end
      function r8vec_diff_norm_l1 ( n, a, b )

c*********************************************************************72
c
cc R8VEC_DIFF_NORM_L1 returns the L1 norm of the difference of R8VEC's.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    The vector L1 norm is defined as:
c
c      R8VEC_NORM_L1 = sum ( 1 <= I <= N ) abs ( A(I) ).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    02 April 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in A
c
c    Input, double precision A(N), B(N), the vectors.
c
c    Output, double precision R8VEC_DIFF_NORM_L1, the L1 norm of A - B.
c
      implicit none

      integer n

      double precision a(n)
      double precision b(n)
      integer i
      double precision r8vec_diff_norm_l1

      r8vec_diff_norm_l1 = 0.0D+00
      do i = 1, n
        r8vec_diff_norm_l1 = r8vec_diff_norm_l1 + abs ( a(i) - b(i) )
      end do

      return
      end
      function r8vec_diff_norm_l2 ( n, a, b )

c*********************************************************************72
c
cc R8VEC_DIFF_NORM_L2 returns the L2 norm of the difference of R8VEC's.
c
c  Discussion:
c
c    An R8VEC is a vector of R8 values.
c
c    The vector L2 norm is defined as:
c
c      R8VEC_NORM_L2 = sqrt ( sum ( 1 <= I <= N ) A(I)^2 ).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    24 June 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in A.
c
c    Input, double precision A(N), B(N), the vectors.
c
c    Output, double precision R8VEC_DIFF_NORM_L2, the L2 norm of A - B.
c
      implicit none

      integer n

      double precision a(n)
      double precision b(n)
      integer i
      double precision r8vec_diff_norm_l2
      double precision value

      value = 0.0D+00
      do i = 1, n
        value = value + ( a(i) - b(i) )**2
      end do
      value = sqrt ( value )

      r8vec_diff_norm_l2 = value

      return
      end
      function r8vec_diff_norm_li ( n, a, b )

c*********************************************************************72
c
cc R8VEC_DIFF_NORM_LI returns the L-infinity norm of the difference of R8VEC's.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    The vector L-infinity norm is defined as:
c
c      R8VEC_NORM_LI = max ( 1 <= I <= N ) abs ( A(I) ).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    02 April 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in A.
c
c    Input, double precision A(N), B(N), the vectors.
c
c    Output, double precision R8VEC_DIFF_NORM_LI, the L-infinity norm of A - B.
c
      implicit none

      integer n

      double precision a(n)
      double precision b(n)
      integer i
      double precision r8vec_diff_norm_li

      r8vec_diff_norm_li = 0.0D+00
      do i = 1, n
        r8vec_diff_norm_li =
     &    max ( r8vec_diff_norm_li, abs ( a(i) - b(i) ) )
      end do

      return
      end
      function r8vec_diff_norm_squared ( n, a, b )

c*********************************************************************72
c
cc R8VEC_DIFF_NORM_SQUARED: square of the L2 norm of the difference of R8VEC's.
c
c  Discussion:
c
c    An R8VEC is a vector of R8 values.
c
c    R8VEC_DIFF_NORM_SQUARED = sum ( 1 <= I <= N ) ( A(I) - B(I) )^2
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    24 June 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in A.
c
c    Input, double precision A(N), B(N), the vectors.
c
c    Output, double precision R8VEC_DIFF_NORM_SQUARED, the square
c    of the L2 norm of A - B.
c
      implicit none

      integer n

      double precision a(n)
      double precision b(n)
      integer i
      double precision r8vec_diff_norm_squared
      double precision value

      value = 0.0D+00
      do i = 1, n
        value = value + ( a(i) - b(i) )**2
      end do

      r8vec_diff_norm_squared = value

      return
      end
      subroutine r8vec_direct_product ( factor_index, factor_order,
     &  factor_value, factor_num, point_num, x )

c*********************************************************************72
c
cc R8VEC_DIRECT_PRODUCT creates a direct product of R8VEC's.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    To explain what is going on here, suppose we had to construct
c    a multidimensional quadrature rule as the product of K rules
c    for 1D quadrature.
c
c    The product rule will be represented as a list of points and weights.
c
c    The J-th item in the product rule will be associated with
c      item J1 of 1D rule 1,
c      item J2 of 1D rule 2,
c      ...,
c      item JK of 1D rule K.
c
c    In particular,
c      X(J) = ( X(1,J1), X(2,J2), ..., X(K,JK))
c    and
c      W(J) = W(1,J1) * W(2,J2) * ... * W(K,JK)
c
c    So we can construct the quadrature rule if we can properly
c    distribute the information in the 1D quadrature rules.
c
c    This routine carries out that task for the abscissas X.
c
c    Another way to do this would be to compute, one by one, the
c    set of all possible indices (J1,J2,...,JK), and then index
c    the appropriate information.  An advantage of the method shown
c    here is that you can process the K-th set of information and
c    then discard it.
c
c  Example:
c
c    Rule 1:
c      Order = 4
c      X(1:4) = ( 1, 2, 3, 4 )
c
c    Rule 2:
c      Order = 3
c      X(1:3) = ( 10, 20, 30 )
c
c    Rule 3:
c      Order = 2
c      X(1:2) = ( 100, 200 )
c
c    Product Rule:
c      Order = 24
c      X(1:24) =
c        ( 1, 10, 100 )
c        ( 2, 10, 100 )
c        ( 3, 10, 100 )
c        ( 4, 10, 100 )
c        ( 1, 20, 100 )
c        ( 2, 20, 100 )
c        ( 3, 20, 100 )
c        ( 4, 20, 100 )
c        ( 1, 30, 100 )
c        ( 2, 30, 100 )
c        ( 3, 30, 100 )
c        ( 4, 30, 100 )
c        ( 1, 10, 200 )
c        ( 2, 10, 200 )
c        ( 3, 10, 200 )
c        ( 4, 10, 200 )
c        ( 1, 20, 200 )
c        ( 2, 20, 200 )
c        ( 3, 20, 200 )
c        ( 4, 20, 200 )
c        ( 1, 30, 200 )
c        ( 2, 30, 200 )
c        ( 3, 30, 200 )
c        ( 4, 30, 200 )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    27 August 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer FACTOR_INDEX, the index of the factor being processed.
c    The first factor processed must be factor 1!
c
c    Input, integer FACTOR_ORDER, the order of the factor.
c
c    Input, double precision FACTOR_VALUE(FACTOR_ORDER), the factor values
c    for factor FACTOR_INDEX.
c
c    Input, integer FACTOR_NUM, the number of factors.
c
c    Input, integer POINT_NUM, the number of elements in the direct product.
c
c    Input/output, double precision X(FACTOR_NUM,POINT_NUM), the elements of the
c    direct product, which are built up gradually.  Before the first call,
c    X might be set to 0.  After each factor has been input, X should
c    have the correct value.
c
c  Local Parameters:
c
c    Local, integer START, the first location of a block of values to set.
c
c    Local, integer CONTIG, the number of consecutive values to set.
c
c    Local, integer SKIP, the distance from the current value of START
c    to the next location of a block of values to set.
c
c    Local, integer REP, the number of blocks of values to set.
c
      implicit none

      integer factor_num
      integer factor_order
      integer point_num

      integer contig
      integer factor_index
      double precision factor_value(factor_order)
      integer i
      integer j
      integer k
      integer rep
      integer skip
      integer start
      double precision x(factor_num,point_num)

      save contig
      save rep
      save skip

      data contig / 0 /
      data rep / 0 /
      data skip / 0 /

      if ( factor_index .eq. 1 ) then
        contig = 1
        skip = 1
        rep = point_num
      end if

      rep = rep / factor_order
      skip = skip * factor_order

      do j = 1, factor_order

        start = 1 + ( j - 1 ) * contig

        do k = 1, rep
          do i = start, start+contig-1
            x(factor_index,i) = factor_value(j)
          end do
          start = start + skip
        end do

      end do

      contig = contig * factor_order

      return
      end
      subroutine r8vec_direct_product2 ( factor_index, factor_order,
     &  factor_value, factor_num, point_num, w )

c*********************************************************************72
c
cc R8VEC_DIRECT_PRODUCT2 creates a direct product of R8VEC's.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    To explain what is going on here, suppose we had to construct
c    a multidimensional quadrature rule as the product of K rules
c    for 1D quadrature.
c
c    The product rule will be represented as a list of points and weights.
c
c    The J-th item in the product rule will be associated with
c      item J1 of 1D rule 1,
c      item J2 of 1D rule 2,
c      ...,
c      item JK of 1D rule K.
c
c    In particular,
c      X(J) = ( X(1,J1), X(2,J2), ..., X(K,JK))
c    and
c      W(J) = W(1,J1) * W(2,J2) * ... * W(K,JK)
c
c    So we can construct the quadrature rule if we can properly
c    distribute the information in the 1D quadrature rules.
c
c    This routine carries out the task involving the weights W.
c
c    Another way to do this would be to compute, one by one, the
c    set of all possible indices (J1,J2,...,JK), and then index
c    the appropriate information.  An advantage of the method shown
c    here is that you can process the K-th set of information and
c    then discard it.
c
c  Example:
c
c    Rule 1:
c      Order = 4
c      W(1:4) = ( 2, 3, 5, 7 )
c
c    Rule 2:
c      Order = 3
c      W(1:3) = ( 11, 13, 17 )
c
c    Rule 3:
c      Order = 2
c      W(1:2) = ( 19, 23 )
c
c    Product Rule:
c      Order = 24
c      W(1:24) =
c        ( 2 * 11 * 19 )
c        ( 3 * 11 * 19 )
c        ( 4 * 11 * 19 )
c        ( 7 * 11 * 19 )
c        ( 2 * 13 * 19 )
c        ( 3 * 13 * 19 )
c        ( 5 * 13 * 19 )
c        ( 7 * 13 * 19 )
c        ( 2 * 17 * 19 )
c        ( 3 * 17 * 19 )
c        ( 5 * 17 * 19 )
c        ( 7 * 17 * 19 )
c        ( 2 * 11 * 23 )
c        ( 3 * 11 * 23 )
c        ( 5 * 11 * 23 )
c        ( 7 * 11 * 23 )
c        ( 2 * 13 * 23 )
c        ( 3 * 13 * 23 )
c        ( 5 * 13 * 23 )
c        ( 7 * 13 * 23 )
c        ( 2 * 17 * 23 )
c        ( 3 * 17 * 23 )
c        ( 5 * 17 * 23 )
c        ( 7 * 17 * 23 )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    27 August 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer FACTOR_INDEX, the index of the factor being processed.
c    The first factor processed must be factor 1!
c
c    Input, integer FACTOR_ORDER, the order of the factor.
c
c    Input, double precision FACTOR_VALUE(FACTOR_ORDER), the factor values
c    for factor FACTOR_INDEX.
c
c    Input, integer FACTOR_NUM, the number of factors.
c
c    Input, integer POINT_NUM, the number of elements in the direct product.
c
c    Input/output, double precision W(POINT_NUM), the elements of the
c    direct product, which are built up gradually.  Before the first call,
c    W should be set to 1.
c
c  Local Parameters:
c
c    Local, integer START, the first location of a block of values to set.
c
c    Local, integer CONTIG, the number of consecutive values to set.
c
c    Local, integer SKIP, the distance from the current value of START
c    to the next location of a block of values to set.
c
c    Local, integer REP, the number of blocks of values to set.
c
      implicit none

      integer factor_num
      integer factor_order
      integer point_num

      integer contig
      integer factor_index
      double precision factor_value(factor_order)
      integer i
      integer j
      integer k
      integer rep
      integer skip
      integer start
      double precision w(point_num)

      save contig
      save rep
      save skip

      data contig / 0 /
      data rep / 0 /
      data skip / 0 /

      if ( factor_index .eq. 1 ) then
        contig = 1
        skip = 1
        rep = point_num
      end if

      rep = rep / factor_order
      skip = skip * factor_order

      do j = 1, factor_order

        start = 1 + ( j - 1 ) * contig

        do k = 1, rep
          do i = start, start+contig-1
            w(i) = w(i) * factor_value(j)
          end do
          start = start + skip
        end do

      end do

      contig = contig * factor_order

      return
      end
      function r8vec_distance ( dim_num, v1, v2 )

c*********************************************************************72
c
cc R8VEC_DISTANCE returns the Euclidean distance between two R8VEC's.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    29 May 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer DIM_NUM, the spatial dimension.
c
c    Input, double precision V1(DIM_NUM), V2(DIM_NUM), the vectors.
c
c    Output, double precision R8VEC_DISTANCE, the Euclidean distance
c    between the vectors.
c
      implicit none

      integer dim_num

      integer i
      double precision r8vec_distance
      double precision v1(dim_num)
      double precision v2(dim_num)

      r8vec_distance = 0.0D+00

      do i = 1, dim_num
        r8vec_distance = r8vec_distance + ( v1(i) - v2(i) )**2
      end do

      r8vec_distance = sqrt ( r8vec_distance )

      return
      end
      function r8vec_distinct ( n, a )

c*********************************************************************72
c
cc R8VEC_DISTINCT is true if the entries in an R8VEC are distinct.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    29 May 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the vector.
c
c    Input, double precision A(N), the vector to be checked.
c
c    Output, logical R8VEC_DISTINCT is TRUE if the elements of A are distinct.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      integer j
      logical r8vec_distinct

      r8vec_distinct = .false.

      do i = 2, n
        do j = 1, i - 1
          if ( a(i) .eq. a(j) ) then
            return
          end if
        end do
      end do

      r8vec_distinct = .true.

      return
      end
      function r8vec_dot_product ( n, v1, v2 )

c*********************************************************************72
c
cc R8VEC_DOT_PRODUCT finds the dot product of a pair of R8VEC's.
c
c  Discussion:
c
c    An R8VEC is a vector of R8 values.
c
c    In FORTRAN90, the system routine DOT_PRODUCT should be called
c    directly.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    27 May 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the dimension of the vectors.
c
c    Input, double precision V1(N), V2(N), the vectors.
c
c    Output, double precision R8VEC_DOT_PRODUCT, the dot product.
c
      implicit none

      integer n

      integer i
      double precision r8vec_dot_product
      double precision v1(n)
      double precision v2(n)
      double precision value

      value = 0.0D+00
      do i = 1, n
        value = value + v1(i) * v2(i)
      end do

      r8vec_dot_product = value

      return
      end
      function r8vec_dot_product_affine ( n, v0, v1, v2 )

c*********************************************************************72
c
cc R8VEC_DOT_PRODUCT_AFFINE computes the affine dot product V1-V0 * V2-V0.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    27 October 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the spatial dimension.
c
c    Input, double precision V0(N), the base vector.
c
c    Input, double precision V1(N), V2(N), the vectors.
c
c    Output, double precision R8VEC_DOT_PRODUCT_AFFINE, the dot product.
c
      implicit none

      integer n

      integer i
      double precision r8vec_dot_product_affine
      double precision v0(n)
      double precision v1(n)
      double precision v2(n)

      r8vec_dot_product_affine = 0.0D+00

      do i = 1, n
        r8vec_dot_product_affine = r8vec_dot_product_affine
     &    + ( v1(i) - v0(i) ) * ( v2(i) - v0(i) )
      end do

      return
      end
      function r8vec_entropy ( n, x )

c*********************************************************************72
c
cc R8VEC_ENTROPY computes the entropy of an R8VEC.
c
c  Discussion:
c
c    Typically, the entries represent probabilities, and must sum to 1.
c    For this function, the only requirement is that the entries be nonnegative.
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    30 August 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer  N, the number of entries.
c
c    Input, double precision X(N), the vector.
c    Each entry must be nonnegative.
c
c    Output, double precision R8VEC_ENTROPY, the entropy of the
c    normalized vector.
c
      implicit none

      integer n

      integer i
      double precision r8_log_2
      double precision r8vec_entropy
      double precision value
      double precision x(n)
      double precision x_sum
      double precision xi

      do i = 1, n
        if ( x(i) .lt. 0.0D+00 ) then
          write ( *, '(a)' ) ' '
          write ( *, '(a)' ) 'R8VEC_ENTROPY - Fatal error!'
          write ( *, '(a)' ) '  Some entries are negative.'
          stop 1
        end if
      end do

      x_sum = 0.0D+00
      do i = 1, n
        x_sum = x_sum + x(i)
      end do

      if ( x_sum .eq. 0.0D+00 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8VEC_ENTROPY - Fatal error!'
        write ( *, '(a)' ) '  Entries sum to 0.'
        stop 1
      end if

      value = 0.0D+00
      do i = 1, n
        if ( 0.0D+00 .lt. x(i) ) then
          xi = x(i) / x_sum
          value = value - r8_log_2 ( xi ) * xi
        end if
      end do

      r8vec_entropy = value

      return
      end
      function r8vec_eq ( n, a1, a2 )

c*********************************************************************72
c
cc R8VEC_EQ is true if every pair of entries in two vectors is equal.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    22 July 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the vectors.
c
c    Input, double precision A1(N), A2(N), two vectors to compare.
c
c    Output, logical R8VEC_EQ.
c    R8VEC_EQ is .TRUE. if every pair of elements A1(I) and A2(I) are equal,
c    and .FALSE. otherwise.
c
      implicit none

      integer n

      double precision a1(n)
      double precision a2(n)
      integer i
      logical r8vec_eq

      r8vec_eq = .false.

      do i = 1, n
        if ( a1(i) .ne. a2(i) ) then
          return
        end if
      end do

      r8vec_eq = .true.

      return
      end
      subroutine r8vec_even ( n, alo, ahi, a )

c*********************************************************************72
c
cc R8VEC_EVEN returns an R8VEC of evenly spaced values.
c
c  Discussion:
c
c    An R8VEC is a vector of R8 values.
c
c    If N is 1, then the midpoint is returned.
c
c    Otherwise, the two endpoints are returned, and N-2 evenly
c    spaced points between them.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    09 December 2004
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of values.
c
c    Input, double precision ALO, AHI, the low and high values.
c
c    Output, double precision A(N), N evenly spaced values.
c    Normally, A(1) = ALO and A(N) = AHI.
c    However, if N = 1, then A(1) = 0.5*(ALO+AHI).
c
      implicit none

      integer n

      double precision a(n)
      double precision ahi
      double precision alo
      integer i

      if ( n .eq. 1 ) then

        a(1) = 0.5D+00 * ( alo + ahi )

      else

        do i = 1, n
          a(i) = ( dble ( n - i     ) * alo
     &           + dble (     i - 1 ) * ahi )
     &           / dble ( n     - 1 )
        end do

      end if

      return
      end
      subroutine r8vec_even_select ( n, xlo, xhi, ival, xval )

c*********************************************************************72
c
cc R8VEC_EVEN_SELECT returns the I-th of N evenly spaced values in [ XLO, XHI ].
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    XVAL = ( (N-IVAL) * XLO + (IVAL-1) * XHI ) / real ( N - 1 )
c
c    Unless N = 1, X(1) = XLO and X(N) = XHI.
c
c    If N = 1, then X(1) = 0.5*(XLO+XHI).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    29 May 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of values.
c
c    Input, double precision XLO, XHI, the low and high values.
c
c    Input, integer IVAL, the index of the desired point.
c    IVAL is normally between 1 and N, but may be any integer value.
c
c    Output, double precision XVAL, the IVAL-th of N evenly spaced values
c    between XLO and XHI.
c
      implicit none

      integer n

      integer ival
      double precision xhi
      double precision xlo
      double precision xval

      if ( n .eq. 1 ) then

        xval = 0.5D+00 * ( xlo + xhi )

      else

        xval = ( dble ( n - ival     ) * xlo
     &         + dble (     ival - 1 ) * xhi )
     &         / dble ( n        - 1 )

      end if

      return
      end
      subroutine r8vec_even2 ( maxval, nfill, nold, xold, nval, xval )

c*********************************************************************72
c
cc R8VEC_EVEN2 linearly interpolates new numbers into an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    The number of values created between two old values can vary from
c    one pair of values to the next.
c
c    The interpolated values are evenly spaced.
c
c    This routine is a generalization of R8VEC_EVEN.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    30 May 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer MAXVAL, the size of the XVAL array, as declared
c    by the user.  MAXVAL must be large enough to hold the NVAL values computed
c    by this routine.  In other words, MAXVAL must be at least equal to
c    NOLD + SUM (1 <= I <= NOLD-1) NFILL(I).
c
c    Input, integer NFILL(NOLD-1), the number of values
c    to be interpolated between XOLD(I) and XOLD(I+1).
c    NFILL(I) does not count the endpoints.  Thus, if
c    NFILL(I) is 1, there will be one new point generated
c    between XOLD(I) and XOLD(I+1).
c    NFILL(I) must be nonnegative.
c
c    Input, integer NOLD, the number of values XOLD,
c    between which extra values are to be interpolated.
c
c    Input, double precision XOLD(NOLD), the original vector of numbers
c    between which new values are to be interpolated.
c
c    Output, integer NVAL, the number of values computed
c    in the XVAL array.
c    NVAL = NOLD + SUM ( 1 <= I <= NOLD-1 ) NFILL(I)
c
c    Output, doble precision XVAL(MAXVAL).  On output, XVAL contains the
c    NOLD values of XOLD, as well as the interpolated
c    values, making a total of NVAL values.
c
      implicit none

      integer maxval
      integer nold

      integer i
      integer j
      integer nadd
      integer nfill(nold-1)
      integer nval
      double precision xold(nold)
      double precision xval(maxval)

      nval = 1

      do i = 1, nold - 1

        if ( nfill(i) .lt. 0 ) then
          write ( *, '(a)' ) ' '
          write ( *, '(a)' ) 'R8VEC_EVEN2 - Fatal error!'
          write ( *, '(a,i8)' ) '  NFILL(I) is negative for I = ', i
          write ( *, '(a,i8)' ) '  NFILL(I) = ', nfill(i)
          stop 1
        end if

        if ( maxval .lt. nval + nfill(i) + 1 ) then
          write ( *, '(a)' ) ' '
          write ( *, '(a)' ) 'R8VEC_EVEN2 - Fatal error!'
          write ( *, '(a)' ) '  MAXVAL is not large enough.  '
          write ( *, '(a,i8)' ) '  MAXVAL = ', maxval
          write ( *, '(a)' )
     &      '  which is exceeded by storage requirements'
          write ( *, '(a,i8)' ) '  for interpolating in interval ', i
          stop 1
        end if

        nadd = nfill(i) + 2

        do j = 1, nadd
          xval(nval+j-1) = ( dble ( nadd - j     ) * xold(i)
     &                     + dble (        j - 1 ) * xold(i+1) )
     &                     / dble ( nadd     - 1 )
        end do

        nval = nval + nfill(i) + 1

      end do

      return
      end
      subroutine r8vec_even2_select ( n, xlo, xhi, ival, xval )

c*********************************************************************72
c
cc R8VEC_EVEN2_SELECT returns the I-th of N evenly spaced midpoint values.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    This function returns the I-th of N evenly spaced midpoints of N
c    equal subintervals of [XLO,XHI].
c
c    XVAL = ( ( 2 * N - 2 * IVAL + 1 ) * XLO 
c           + (         2 * IVAL - 1 ) * XHI ) 
c           / ( 2 * N                )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    25 July 2012
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of values.
c
c    Input, double precision XLO, XHI, the low and high values.
c
c    Input, integer IVAL, the index of the desired point.
c    IVAL is normally between 1 and N, but may be any integer value.
c
c    Output, double precision XVAL, the IVAL-th of N evenly spaced midpoints
c    between XLO and XHI.
c
      implicit none

      integer n

      integer ival
      double precision xhi
      double precision xlo
      double precision xval

      xval = ( dble ( 2 * n - 2 * ival + 1 ) * xlo   
     &       + dble (         2 * ival - 1 ) * xhi ) 
     &       / dble ( 2 * n )

      return
      end
      subroutine r8vec_even3 ( nold, nval, xold, xval )

c*********************************************************************72
c
cc R8VEC_EVEN3 evenly interpolates new data into an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    This routine accepts a short vector of numbers, and returns a longer
c    vector of numbers, created by interpolating new values between
c    the given values.
c
c    Between any two original values, new values are evenly interpolated.
c
c    Over the whole vector, the new numbers are interpolated in
c    such a way as to try to minimize the largest distance interval size.
c
c    The algorithm employed is not "perfect".
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    03 June 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer NOLD, the number of values XOLD, between
c    which extra values are to be interpolated.
c
c    Input, integer NVAL, the number of values to be computed
c    in the XVAL array.  NVAL should be at least NOLD.
c
c    Input, double precision XOLD(NOLD), the original vector of numbers
c    between which new values are to be interpolated.
c
c    Output, double precision XVAL(NVAL).  On output, XVAL contains the
c    NOLD values of XOLD, as well as interpolated
c    values, making a total of NVAL values.
c
      implicit none

      integer nval
      integer nold

      double precision density
      integer i
      integer ival
      integer j
      integer nmaybe
      integer npts
      integer ntemp
      integer ntot
      double precision xlen
      double precision xleni
      double precision xlentot
      double precision xold(nold)
      double precision xval(nval)

      xlen = 0.0D+00
      do i = 1, nold - 1
        xlen = xlen + abs ( xold(i+1) - xold(i) )
      end do

      ntemp = nval - nold

      density = dble ( ntemp ) / xlen

      ival = 1
      ntot = 0
      xlentot = 0.0D+00

      do i = 1, nold - 1

        xleni = abs ( xold(i+1) - xold(i) )
        npts = int ( density * xleni )
        ntot = ntot + npts
c
c  Determine if we have enough left-over density that it should
c  be changed into a point.  A better algorithm would agonize
c  more over where that point should go.
c
        xlentot = xlentot + xleni
        nmaybe = nint ( xlentot * density )

        if ( ntot .lt. nmaybe ) then
          npts = npts + nmaybe - ntot
          ntot = nmaybe
        end if

        do j = 1, npts + 2
          xval(ival+j-1) = ( dble ( npts+2 - j     ) * xold(i)
     &                     + dble (          j - 1 ) * xold(i+1) )
     &                     / dble ( npts+2     - 1 )
        end do

        ival = ival + npts + 1

      end do

      return
      end
      subroutine r8vec_expand_linear ( n, x, fat, xfat )

c*********************************************************************72
c
cc R8VEC_EXPAND_LINEAR linearly interpolates new data into an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    This routine copies the old data, and inserts NFAT new values
c    between each pair of old data values.  This would be one way to
c    determine places to evenly sample a curve, given the (unevenly
c    spaced) points at which it was interpolated.
c
c  Example:
c
c    N = 3
c    NFAT = 2
c
c    X(1:N)        = (/ 0.0,           6.0,             7.0 /)
c    XFAT(1:2*3+1) = (/ 0.0, 2.0, 4.0, 6.0, 6.33, 6.66, 7.0 /)
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    03 June 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of input data values.
c
c    Input, double precision X(N), the original data.
c
c    Input, integer FAT, the number of data values to interpolate
c    between each pair of original data values.
c
c    Output, double precision XFAT((N-1)*(FAT+1)+1), the "fattened" data.
c
      implicit none

      integer fat
      integer n

      integer i
      integer j
      integer k
      double precision x(n)
      double precision xfat((n-1)*(fat+1)+1)

      k = 0

      do i = 1, n - 1

        k = k + 1
        xfat(k) = x(i)

        do j = 1, fat
          k = k + 1
          xfat(k) = ( dble ( fat - j + 1 ) * x(i)
     &              + dble (       j     ) * x(i+1) )
     &              / dble ( fat     + 1 )
        end do

      end do

      k = k + 1
      xfat(k) = x(n)

      return
      end
      subroutine r8vec_expand_linear2 ( n, x, before, fat, after, xfat )

c*********************************************************************72
c
cc R8VEC_EXPAND_LINEAR2 linearly interpolates new data into an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    This routine starts with a vector of data.
c
c    The intent is to "fatten" the data, that is, to insert more points
c    between successive values of the original data.
c
c    There will also be extra points placed BEFORE the first original
c    value and AFTER that last original value.
c
c    The "fattened" data is equally spaced between the original points.
c
c    The BEFORE data uses the spacing of the first original interval,
c    and the AFTER data uses the spacing of the last original interval.
c
c  Example:
c
c    N = 3
c    BEFORE = 3
c    FAT = 2
c    AFTER = 1
c
c    X    = (/                   0.0,           6.0,             7.0       /)
c    XFAT = (/ -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 6.33, 6.66, 7.0, 7.66 /)
c            3 "BEFORE's"        Old  2 "FATS"  Old    2 "FATS"  Old  1 "AFTER"
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    03 June 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of input data values.
c    N must be at least 2.
c
c    Input, double precision X(N), the original data.
c
c    Input, integer BEFORE, the number of "before" values.
c
c    Input, integer FAT, the number of data values to interpolate
c    between each pair of original data values.
c
c    Input, integer AFTER, the number of "after" values.
c
c    Output, double precision XFAT(BEFORE+(N-1)*(FAT+1)+1+AFTER), the
c    "fattened" data.
c
      implicit none

      integer after
      integer before
      integer fat
      integer n

      integer i
      integer j
      integer k
      double precision x(n)
      double precision xfat(before+(n-1)*(fat+1)+1+after)

      k = 0
c
c  Points BEFORE.
c
      do j = 1 - before + fat, fat
        k = k + 1
        xfat(k) = ( dble ( fat - j + 1 ) * ( x(1) - ( x(2) - x(1) ) )
     &            + dble (       j     ) *   x(1)          )
     &            / dble ( fat     + 1 )
      end do
c
c  Original points and FAT points.
c
      do i = 1, n - 1

        k = k + 1
        xfat(k) = x(i)

        do j = 1, fat
          k = k + 1
          xfat(k) = ( dble ( fat - j + 1 ) * x(i)
     &              + dble (       j     ) * x(i+1) )
     &              / dble ( fat     + 1 )
        end do

      end do

      k = k + 1
      xfat(k) = x(n)
c
c  Points AFTER.
c
      do j = 1, after
        k = k + 1
        xfat(k) =
     &    ( dble ( fat - j + 1 ) * x(n)
     &    + dble (       j     ) * ( x(n) + ( x(n) - x(n-1) ) ) )
     &    / dble ( fat     + 1 )
      end do

      return
      end
      subroutine r8vec_first_index ( n, a, tol, first_index )

c*********************************************************************72
c
cc R8VEC_FIRST_INDEX indexes the first occurrence of values in an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    For element A(I) of the vector, FIRST(I) is the index in A of
c    the first occurrence of the value A(I).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    24 August 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of elements of A.
c
c    Input, double precision A(N), the unsorted array to examine.
c
c    Input, double precision TOL, a tolerance for equality.
c
c    Output, integer FIRST_INDEX(N), the first occurrence index.
c
      implicit none

      integer n

      double precision a(n)
      integer first_index(n)
      integer i
      integer j
      double precision tol

      do i = 1, n
        first_index(i) = -1
      end do

      do i = 1, n

        if ( first_index(i) .eq. -1 ) then

          first_index(i) = i

          do j = i + 1, n
            if ( abs ( a(i) - a(j) ) .le. tol ) then
              first_index(j) = i
            end if
          end do

        end if

      end do

      return
      end
      subroutine r8vec_floor ( n, r8vec, floorvec )

c*********************************************************************72
c
cc R8VEC_FLOOR rounds "down" (towards -infinity) entries of an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Example:
c
c    R8    Value
c
c   -1.1  -2
c   -1.0  -1
c   -0.9  -1
c    0.0   0
c    5.0   5
c    5.1   5
c    5.9   5
c    6.0   6
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    30 May 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries.
c
c    Input, double precision R8VEC(N), the values to be rounded down.
c
c    Output, integer FLOORVEC(N), the rounded value.
c
      implicit none

      integer n

      integer floorvec(n)
      integer i
      double precision r8vec(n)
      integer value

      do i = 1, n

        value = int ( r8vec(i) )

        if ( r8vec(i) .lt. dble ( value ) ) then
          value = value - 1
        end if

        floorvec(i) = value

      end do

      return
      end
      subroutine r8vec_frac ( n, a, k, frac )

c*********************************************************************72
c
cc R8VEC_FRAC searches for the K-th smallest entry in an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8 values.
c
c    Hoare's algorithm is used.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    17 July 2000
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of elements of A.
c
c    Input/output, double precision A(N).
c    On input, A is the array to search.
c    On output, the elements of A have been somewhat rearranged.
c
c    Input, integer K, the fractile to be sought.  If K = 1, the minimum
c    entry is sought.  If K = N, the maximum is sought.  Other values
c    of K search for the entry which is K-th in size.  K must be at
c    least 1, and no greater than N.
c
c    Output, double precision FRAC, the value of the K-th fractile of A.
c
      implicit none

      integer n

      double precision a(n)
      double precision frac
      integer i
      integer iryt
      integer j
      integer k
      integer left
      double precision temp
      double precision x

      if ( n .le. 0 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8VEC_FRAC - Fatal error!'
        write ( *, '(a,i8)' ) '  Illegal nonpositive value of N = ', n
        stop 1
      end if

      if ( k .le. 0 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8VEC_FRAC - Fatal error!'
        write ( *, '(a,i8)' ) '  Illegal nonpositive value of K = ', k
        stop 1
      end if

      if ( n .lt. k ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8VEC_FRAC - Fatal error!'
        write ( *, '(a,i8)' ) '  Illegal N < K, K = ', k
        stop 1
      end if

      left = 1
      iryt = n

10    continue

        if ( iryt .le. left ) then
          frac = a(k)
          go to 60
        end if

        x = a(k)
        i = left
        j = iryt

20      continue

          if ( j .lt. i ) then
            if ( j .lt. k ) then
              left = i
            end if
            if ( k .lt. i ) then
              iryt = j
            end if
            go to 50
          end if
c
c  Find I so that X <= A(I).
c
30        continue

          if ( a(i) .lt. x ) then
            i = i + 1
            go to 30
          end if
c
c  Find J so that A(J) <= X.
c
40        continue

          if ( x .lt. a(j) ) then
            j = j - 1
            go to 40
          end if

          if ( i .le. j ) then

            temp = a(i)
            a(i) = a(j)
            a(j) = temp

            i = i + 1
            j = j - 1

          end if

        go to 20

50      continue

      go to 10

60    continue

      return
      end
      subroutine r8vec_fraction ( n, x, fraction )

c*********************************************************************72
c
cc R8VEC_FRACTION returns the fraction parts of an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    If we regard a real number as
c
c      R8 = SIGN * ( WHOLE + FRACTION )
c
c    where
c
c      SIGN is +1 or -1,
c      WHOLE is a nonnegative integer
c      FRACTION is a nonnegative real number strictly less than 1,
c
c    then this routine returns the value of FRACTION.
c
c  Example:
c
c     R8    R8_FRACTION
c
c    0.00      0.00
c    1.01      0.01
c    2.02      0.02
c   19.73      0.73
c   -4.34      0.34
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    16 May 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of arguments.
c
c    Input, double precision X(N), the arguments.
c
c    Output, double precision FRACTION(N), the fraction parts.
c
      implicit none

      integer n

      double precision fraction(n)
      integer i
      double precision x(n)

      do i = 1, n
        fraction(i) = abs ( x(i) ) - dble ( int ( abs ( x(i) ) ) )
      end do

      return
      end
      function r8vec_gt ( n, a1, a2 )

c*********************************************************************72
c
cc R8VEC_GT .eq. ( A1 greater than A2 ) for double precision vectors.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    The comparison is lexicographic.
c
c    A1 > A2  <=>                              A1(1) > A2(1) or
c                 ( A1(1)     .eq. A2(1)     and A1(2) > A2(2) ) or
c                 ...
c                 ( A1(1:N-1) .eq. A2(1:N-1) and A1(N) > A2(N)
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    22 July 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the dimension of the vectors.
c
c    Input, double precision A1(N), A2(N), the vectors to be compared.
c
c    Output, logical R8VEC_GT, is TRUE if and only if A1 > A2.
c
      implicit none

      integer n

      double precision a1(n)
      double precision a2(n)
      integer i
      logical r8vec_gt

      r8vec_gt = .false.

      do i = 1, n

        if ( a2(i) .lt. a1(i) ) then
          r8vec_gt = .true.
          return
        else if ( a1(i) .lt. a2(i) ) then
          return
        end if

      end do

      return
      end
      subroutine r8vec_heap_a ( n, a )

c*********************************************************************72
c
cc R8VEC_HEAP_A reorders an R8VEC into an ascending heap.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    An ascending heap is an array A with the property that, for every index J,
c    A(J) <= A(2*J) and A(J) <= A(2*J+1), (as long as the indices
c    2*J and 2*J+1 are legal).
c
c                  A(1)
c                /      \
c            A(2)         A(3)
c          /     \        /  \
c      A(4)       A(5)  A(6) A(7)
c      /  \       /   \
c    A(8) A(9) A(10) A(11)
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    03 June 2009
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Albert Nijenhuis, Herbert Wilf,
c    Combinatorial Algorithms for Computers and Calculators,
c    Academic Press, 1978,
c    ISBN: 0-12-519260-6,
c    LC: QA164.N54.
c
c  Parameters:
c
c    Input, integer N, the size of the input array.
c
c    Input/output, double precision A(N).
c    On input, an unsorted array.
c    On output, the array has been reordered into a heap.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      integer ifree
      double precision key
      integer m
c
c  Only nodes N/2 down to 1 can be "parent" nodes.
c
      do i = n / 2, 1, -1
c
c  Copy the value out of the parent node.
c  Position IFREE is now "open".
c
        key = a(i)
        ifree = i

10      continue
c
c  Positions 2*IFREE and 2*IFREE + 1 are the descendants of position
c  IFREE.  (One or both may not exist because they exceed N.)
c
          m = 2 * ifree
c
c  Does the first position exist?
c
          if ( n .lt. m ) then
            go to 20
          end if
c
c  Does the second position exist?
c
          if ( m + 1 .le. n ) then
c
c  If both positions exist, take the smaller of the two values,
c  and update M if necessary.
c
            if ( a(m+1) .lt. a(m) ) then
              m = m + 1
            end if

          end if
c
c  If the small descendant is smaller than KEY, move it up,
c  and update IFREE, the location of the free position, and
c  consider the descendants of THIS position.
c
          if ( key .le. a(m) ) then
            go to 20
          end if

          a(ifree) = a(m)
          ifree = m

        go to 10
c
c  Once there is no more shifting to do, KEY moves into the free spot.
c
20      continue

        a(ifree) = key

      end do

      return
      end
      subroutine r8vec_heap_d ( n, a )

c*********************************************************************72
c
cc R8VEC_HEAP_D reorders an R8VEC into an descending heap.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    A descending heap is an array A with the property that, for every index J,
c    A(J) >= A(2*J) and A(J) >= A(2*J+1), (as long as the indices
c    2*J and 2*J+1 are legal).
c
c                  A(1)
c                /      \
c            A(2)         A(3)
c          /     \        /  \
c      A(4)       A(5)  A(6) A(7)
c      /  \       /   \
c    A(8) A(9) A(10) A(11)
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    03 June 2010
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Albert Nijenhuis, Herbert Wilf,
c    Combinatorial Algorithms for Computers and Calculators,
c    Academic Press, 1978,
c    ISBN: 0-12-519260-6,
c    LC: QA164.N54.
c
c  Parameters:
c
c    Input, integer N, the size of the input array.
c
c    Input/output, double precision A(N).
c    On input, an unsorted array.
c    On output, the array has been reordered into a heap.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      integer ifree
      double precision key
      integer m
c
c  Only nodes N/2 down to 1 can be "parent" nodes.
c
      do i = n / 2, 1, -1
c
c  Copy the value out of the parent node.
c  Position IFREE is now "open".
c
        key = a(i)
        ifree = i

10      continue
c
c  Positions 2*IFREE and 2*IFREE + 1 are the descendants of position
c  IFREE.  (One or both may not exist because they exceed N.)
c
          m = 2 * ifree
c
c  Does the first position exist?
c
          if ( n .lt. m ) then
            go to 20
          end if
c
c  Does the second position exist?
c
          if ( m + 1 .le. n ) then
c
c  If both positions exist, take the larger of the two values,
c  and update M if necessary.
c
            if ( a(m) .lt. a(m+1) ) then
              m = m + 1
            end if

          end if
c
c  If the large descendant is larger than KEY, move it up,
c  and update IFREE, the location of the free position, and
c  consider the descendants of THIS position.
c
          if ( a(m) .le. key ) then
            go to 20
          end if

          a(ifree) = a(m)
          ifree = m

        go to 10

20      continue
c
c  Once there is no more shifting to do, KEY moves into the free spot IFREE.
c
        a(ifree) = key

      end do

      return
      end
      subroutine r8vec_heap_d_extract ( n, a, value )

c*********************************************************************72
c
cc R8VEC_HEAP_D_EXTRACT: extract maximum from a heap descending sorted R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    In other words, the routine finds the maximum value in the
c    heap, returns that value to the user, deletes that value from
c    the heap, and restores the heap to its proper form.
c
c    This is one of three functions needed to model a priority queue.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    18 August 2010
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Thomas Cormen, Charles Leiserson, Ronald Rivest,
c    Introduction to Algorithms,
c    MIT Press, 2001,
c    ISBN: 0262032937,
c    LC: QA76.C662.
c
c  Parameters:
c
c    Input/output, integer N, the number of items in the heap.
c
c    Input/output, double precision A(N), the heap.
c
c    Output, double precision VALUE, the item of maximum value, which has
c    been removed from the heap.
c
      implicit none

      double precision a(*)
      integer n
      double precision value

      if ( n .lt. 1 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8VEC_HEAP_D_EXTRACT - Fatal error!'
        write ( *, '(a)' ) '  The heap is empty.'
        stop 1
      end if
c
c  Get the maximum value.
c
      value = a(1)

      if ( n .eq. 1 ) then
        n = 0
        return
      end if
c
c  Shift the last value down.
c
      a(1) = a(n)
c
c  Restore the heap structure.
c
      n = n - 1
      call r8vec_sort_heap_d ( n, a )

      return
      end
      subroutine r8vec_heap_d_insert ( n, a, value )

c*********************************************************************72
c
cc R8VEC_HEAP_D_INSERT inserts a value into a heap descending sorted R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    This is one of three functions needed to model a priority queue.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    18 August 2010
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Thomas Cormen, Charles Leiserson, Ronald Rivest,
c    Introduction to Algorithms,
c    MIT Press, 2001,
c    ISBN: 0262032937,
c    LC: QA76.C662.
c
c  Parameters:
c
c    Input/output, integer N, the number of items in the heap.
c
c    Input/output, double precision A(N), the heap.
c
c    Input, double precision VALUE, the value to be inserted.
c
      implicit none

      double precision a(*)
      integer i
      integer n
      integer parent
      double precision value

      n = n + 1
      i = n

10    continue

      if ( 1 .lt. i ) then

        parent = i / 2

        if ( value .le. a(parent) ) then
          go to 20
        end if

        a(i) = a(parent)
        i = parent

        go to 10

      end if

20    continue

      a(i) = value

      return
      end
      subroutine r8vec_heap_d_max ( n, a, value )

c*********************************************************************72
c
cc R8VEC_HEAP_D_MAX returns the maximum value in a heap descending sorted R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    This is one of three functions needed to model a priority queue.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    18 August 2010
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Thomas Cormen, Charles Leiserson, Ronald Rivest,
c    Introduction to Algorithms,
c    MIT Press, 2001,
c    ISBN: 0262032937,
c    LC: QA76.C662.
c
c  Parameters:
c
c    Input, integer N, the number of items in the heap.
c
c    Input, double precision A(N), the heap.
c
c    Output, double precision VALUE, the maximum value in the heap.
c
      implicit none

      integer n

      double precision a(n)
      double precision value

      value = a(1)

      return
      end
      subroutine r8vec_histogram ( n, a, a_lo, a_hi, histo_num,
     &  histo_gram )

c*********************************************************************72
c
cc R8VEC_HISTOGRAM histograms an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    Values between A_LO and A_HI will be histogrammed into the bins
c    1 through HISTO_NUM.  Values below A_LO are counted in bin 0,
c    and values greater than A_HI are counted in bin HISTO_NUM+1.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    04 June 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of elements of A.
c
c    Input, double precision A(N), the array to examine.
c
c    Input, double precision A_LO, A_HI, the lowest and highest
c    values to be histogrammed.  These values will also define the bins.
c
c    Input, integer HISTO_NUM, the number of bins to use.
c
c    Output, integer HISTO_GRAM(0:HISTO_NUM+1), contains the
c    number of entries of A in each bin.
c
      implicit none

      integer histo_num
      integer n

      double precision a(n)
      double precision a_hi
      double precision a_lo
      double precision delta
      integer histo_gram(0:histo_num+1)
      integer i
      integer j

      do i = 0, histo_num + 1
        histo_gram(i) = 0
      end do

      delta = ( a_hi - a_lo ) / dble ( 2 * histo_num )

      do i = 1, n

        if ( a(i) .lt. a_lo ) then

          histo_gram(0) = histo_gram(0) + 1

        else if ( a(i) .le. a_hi ) then

          j = nint (
     &      ( ( a_hi -           delta - a(i)        )
     &      * dble ( 1 )
     &      + (      -           delta + a(i) - a_lo )
     &      * dble ( histo_num ) )
     &      / ( a_hi - 2.0D+00 * delta        - a_lo ) )

          histo_gram(j) = histo_gram(j) + 1

        else if ( a_hi .lt. a(i) ) then

          histo_gram(histo_num+1) = histo_gram(histo_num+1) + 1

        end if

      end do

      return
      end
      subroutine r8vec_house_column ( n, a_vec, k, v )

c*********************************************************************72
c
cc R8VEC_HOUSE_COLUMN defines a Householder premultiplier that "packs" a column.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    The routine returns a vector V that defines a Householder
c    premultiplier matrix H(V) that zeros out the subdiagonal entries of
c    column K of the matrix A.
c
c       H(V) = I - 2 * v * v'
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    05 February 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the order of the matrix A.
c
c    Input, double precision A_VEC(N), a row or column of a matrix A.
c
c    Input, integer K, the "special" entry in A_VEC.
c    The Householder matrix will zero out the entries after this.
c
c    Output, double precision V(N), a vector of unit L2 norm which defines an
c    orthogonal Householder premultiplier matrix H with the property
c    that the K-th column of H*A is zero below the diagonal.
c
      implicit none

      integer n

      double precision a_vec(n)
      integer i
      integer k
      double precision s
      double precision v(n)

      do i = 1, n
        v(i) = 0.0D+00
      end do

      if ( k .lt. 1 .or. n .le. k ) then
        return
      end if

      s = 0.0D+00
      do i = k, n
        s = s + a_vec(i) ** 2
      end do
      s = sqrt ( s )

      if ( s .eq. 0.0D+00 ) then
        return
      end if

      v(k) = a_vec(k) + sign ( s, a_vec(k) )
      do i = k + 1, n
        v(i) = a_vec(i)
      end do

      s = 0.0D+00
      do i = k, n
        s = s + v(i) * v(i)
      end do
      s = sqrt ( s )

      do i = k, n
        v(i) = v(i) / s
      end do

      return
      end
      function r8vec_i4vec_dot_product ( n, r8vec, i4vec )

c*********************************************************************72
c
cc R8VEC_I4VEC_DOT_PRODUCT finds the dot product of an R8VEC and an I4VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    An I4VEC is a vector of I4's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    30 June 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the dimension of the vectors.
c
c    Input, double precision R8VEC(N), the first vector.
c
c    Input, integer I4VEC(N), the second vector.
c
c    Output, double precision R8VEC_I4VEC_DOT_PRODUCT, the dot product.
c
      implicit none

      integer n

      integer i
      integer i4vec(n)
      double precision r8vec(n)
      double precision r8vec_i4vec_dot_product
      double precision value

      value = 0.0D+00
      do i = 1, n
        value = value + r8vec(i) * dble ( i4vec(i) )
      end do

      r8vec_i4vec_dot_product = value

      return
      end
      function r8vec_in_01 ( n, a )

c*********************************************************************72
c
cc R8VEC_IN_01 is TRUE if the entries of an R8VEC are in the range [0,1].
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    16 May 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries.
c
c    Input, double precision A(N), the vector.
c
c    Output, logical R8VEC_IN_01, is TRUE if every entry is
c    between 0 and 1.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      logical r8vec_in_01

      do i = 1, n
        if ( a(i) .lt. 0.0D+00 .or. 1.0D+00 .lt. a(i) ) then
          r8vec_in_01 = .false.
          return
        end if
      end do

      r8vec_in_01 = .true.

      return
      end
      function r8vec_in_ab ( n, x, a, b )

c*********************************************************************72
c
cc R8VEC_IN_AB is TRUE if the entries of an R8VEC are in the range [A,B].
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    15 April 2012
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries.
c
c    Input, double precision X(N), the vector.
c
c    Input, double precision A, B, the limits of the range.
c
c    Output, logical R8VEC_IN_01, is TRUE if every entry is
c    between A and B.
c
      implicit none

      integer n

      double precision a
      double precision b
      integer i
      logical r8vec_in_ab
      double precision x(n)

      do i = 1, n
        if ( x(i) .lt.a .or. b .lt. x(i) ) then
          r8vec_in_ab = .false.
          return
        end if
      end do

      r8vec_in_ab = .true.

      return
      end
      subroutine r8vec_index_delete_all ( n, x, indx, xval )

c*********************************************************************72
c
cc R8VEC_INDEX_DELETE_ALL deletes a value from an indexed sorted R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    Note that the value of N is adjusted because of the deletionsc
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    09 April 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input/output, integer N, the size of the current list.
c
c    Input/output, double precision X(N), the list.
c
c    Input/output, integer INDX(N), the sort index of the list.
c
c    Input, double precision XVAL, the value to be sought.
c
      implicit none

      integer n

      integer equal
      integer equal1
      integer equal2
      integer get
      integer i
      integer indx(*)
      integer less
      integer more
      integer put
      double precision x(*)
      double precision xval

      if ( n .lt. 1 ) then
        n = 0
        return
      end if

      call r8vec_index_search ( n, x, indx, xval, less, equal, more )

      if ( equal .eq. 0 ) then
        return
      end if

      equal1 = equal

10    continue

        if ( equal1 .le. 1 ) then
          go to 20
        end if

        if ( x(indx(equal1-1)) .ne. xval ) then
          go to 20
        end if

        equal1 = equal1 - 1

      go to 10

20    continue

      equal2 = equal

30    continue

        if ( n .le. equal2 ) then
          go to 40
        end if

        if ( x(indx(equal2+1)) .ne. xval ) then
          go to 40
        end if

        equal2 = equal2 + 1

      go to 30

40    continue
c
c  Discard certain X values.
c
      put = 0

      do get = 1, n

        if ( x(get) .ne. xval ) then
          put = put + 1
          x(put) = x(get)
        end if

      end do

      do i = put + 1, n
        x(i) = 0.0D+00
      end do
c
c  Adjust the INDX values.
c
      do equal = equal1, equal2
        do i = 1, n
          if ( indx(equal) < indx(i) ) then
            indx(i) = indx(i) - 1
          end if
        end do
      end do
c
c  Discard certain INDX values.
c
      do i = equal1, n + equal1 - equal2 - 1
        indx(i) = indx(i-equal1+equal2+1)
      end do
      do i = n + equal1 - equal2, n
        indx(i) = 0
      end do
c
c  Adjust N.
c
      n = put

      return
      end
      subroutine r8vec_index_delete_dupes ( n, x, indx, n2, x2, indx2 )

c*********************************************************************72
c
cc R8VEC_INDEX_DELETE_DUPES deletes duplicates from an indexed sorted R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    The output quantities N2, X2, and INDX2 are computed from the
c    input quantities by sorting, and eliminating duplicates.
c
c    The output arrays should be dimensioned of size N, unless the user
c    knows in advance what the value of N2 will be.
c
c    The output arrays may be identified with the input arrays.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    09 April 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the size of the input list.
c
c    Input, double precision X(N), the list.
c
c    Input, integer INDX(N), the sort index of the list.
c
c    Output, integer N2, the number of unique entries in X.
c
c    Output, double precision X2(N2), a copy of the list which has
c    been sorted, and made unique.
c
c    Output, integer INDX2(N2), the sort index of the new list.
c
      implicit none

      integer n

      integer i
      integer indx(n)
      integer indx2(n)
      integer n2
      integer n3
      double precision x(n)
      double precision x2(n)
      double precision x3(n)

      i = 0
      n3 = 0

10    continue

        i = i + 1

        if ( n .lt. i ) then
          go to 20
        end if

        if ( 1 .lt. i ) then
          if ( x(indx(i)) .eq. x3(n3) ) then
            go to 10
          end if
        end if

        n3 = n3 + 1
        x3(n3) = x(indx(i))

      go to 10

20    continue
c
c  Copy data into output arrays.
c
      n2 = n3
      do i = 1, n2
        x2(i) = x3(i)
      end do
      call i4vec_indicator1 ( n2, indx2 )

      return
      end
      subroutine r8vec_index_delete_one ( n, x, indx, xval, n2, x2, 
     &  indx2 )

c*********************************************************************72
c
cc R8VEC_INDEX_DELETE_ONE deletes one copy of a value from indexed sorted R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    If the value occurs in the list more than once, only one copy is deleted.
c
c    Note that the value of N is adjusted because of the deletions.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    09 April 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the size of the current list.
c
c    Input, double precision X(N), the list.
c
c    Input, integer INDX(N), the sort index of the list.
c
c    Input, double precision XVAL, the value to be sought.
c
c    Output, integer N2, the size of the current list.
c
c    Output, double precision X2(N2), the list.
c
c    Output, integer INDX2(N2), the sort index of the list.
c
      implicit none

      integer n

      integer equal
      integer i
      integer indx(n)
      integer indx2(n)
      integer j
      integer less
      integer more
      integer n2
      double precision x(n)
      double precision x2(n)
      double precision xval

      if ( n .lt. 1 ) then
        n2 = 0
        return
      end if

      n2 = n

      do i = 1, n2
        indx2(i) = indx(i)
      end do

      do i = 1, n2
        x2(i) = x(i)
      end do

      call r8vec_index_search ( n2, x2, indx2, xval, less, equal, more )

      if ( equal .ne. 0 ) then
        j = indx2(equal)
        do i = j, n2 - 1
          x2(i) = x2(i+1)
        end do
        do i = equal, n2 - 1
          indx2(i) = indx2(i+1)
        end do
        do i = 1, n2 - 1
          if ( j .lt. indx2(i) ) then
            indx2(i) = indx2(i) - 1
          end if
        end do
        n2 = n2 - 1
      end if

      return
      end
      subroutine r8vec_index_insert ( n, x, indx, xval )

c*********************************************************************72
c
cc R8VEC_INDEX_INSERT inserts a value in an indexed sorted R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    21 August 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input/output, integer N, the size of the current list.
c
c    Input/output, double precision X(N), the list.
c
c    Input/output, integer INDX(N), the sort index of the list.
c
c    Input, double precision XVAL, the value to be sought.
c
      implicit none

      integer n

      integer equal
      integer i
      integer indx(*)
      integer less
      integer more
      double precision x(*)
      double precision xval

      if ( n .le. 0 ) then
        n = 1
        x(1) = xval
        indx(1) = 1
        return
      end if

      call r8vec_index_search ( n, x, indx, xval, less, equal, more )

      x(n+1) = xval
      do i = n, more, -1
        indx(i+1) = indx(i)
      end do
      indx(more) = n + 1
      n = n + 1

      return
      end
      subroutine r8vec_index_insert_unique ( n, x, indx, xval )

c*********************************************************************72
c
cc R8VEC_INDEX_INSERT_UNIQUE inserts a unique value in an indexed sorted R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    If the value does not occur in the list, it is included in the list,
c    and N, X and INDX are updated.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    19 August 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input/output, integer N, the size of the current list.
c
c    Input/output, double precision X(N), the list.
c
c    Input/output, integer INDX(N), the sort index of the list.
c
c    Input, double precision XVAL, the value to be sought.
c
      implicit none

      integer n

      integer equal
      integer i
      integer indx(*)
      integer less
      integer more
      double precision x(*)
      double precision xval

      if ( n .le. 0 ) then
        n = 1
        x(1) = xval
        indx(1) = 1
        return
      end if
c
c  Does XVAL already occur in X?
c
      call r8vec_index_search ( n, x, indx, xval, less, equal, more )

      if ( equal .eq. 0 ) then
        x(n+1) = xval
        do i = n, more, - 1
          indx(i+1) = indx(i)
        end do
        indx(more) = n + 1
        n = n + 1
      end if

      return
      end
      subroutine r8vec_index_order ( n, x, indx )

c*********************************************************************72
c
cc R8VEC_INDEX_ORDER sorts an R8VEC using an index vector.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    The index vector itself is not modified.  Therefore, the pair
c    (X,INDX) no longer represents an index sorted vector.  If this
c    relationship is to be preserved, then simply set INDX(1:N)=(1:N).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    16 May 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the size of the current list.
c
c    Input/output, double precision X(N), the list.  On output, the list
c    has been sorted.
c
c    Input, integer INDX(N), the sort index of the list.
c
      implicit none

      integer n

      integer i
      integer indx(n)
      double precision x(n)
      double precision y(n)

      do i = 1, n
        y(i) = x(indx(i))
      end do

      do i = 1, n
        x(i) = y(i)
      end do

      return
      end
      subroutine r8vec_index_search ( n, x, indx, xval, less, equal,
     &  more )

c*********************************************************************72
c
cc R8VEC_INDEX_SEARCH searches for a value in an indexed sorted R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    04 June 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the size of the current list.
c
c    Input, double precision X(N), the list.
c
c    Input, integer INDX(N), the sort index of the list.
c
c    Input, double precision XVAL, the value to be sought.
c
c    Output, integer LESS, EQUAL, MORE, the indexes in INDX of the
c    entries of X that are just less than, equal to, and just greater
c    than XVAL.  If XVAL does not occur in X, then EQUAL is zero.
c    If XVAL is the minimum entry of X, then LESS is 0.  If XVAL
c    is the greatest entry of X, then MORE is N+1.
c
      implicit none

      integer n

      integer equal
      integer hi
      integer indx(n)
      integer less
      integer lo
      integer mid
      integer more
      double precision x(n)
      double precision xhi
      double precision xlo
      double precision xmid
      double precision xval

      if ( n .le. 0 ) then
        less = 0
        equal = 0
        more = 0
        return
      end if

      lo = 1
      hi = n
      xlo = x(indx(lo))
      xhi = x(indx(hi))

      if ( xval .lt. xlo ) then
        less = 0
        equal = 0
        more = 1
        return
      else if ( xval .eq. xlo ) then
        less = 0
        equal = 1
        more = 2
        return
      end if

      if ( xhi .lt. xval ) then
        less = n
        equal = 0
        more = n + 1
        return
      else if ( xval .eq. xhi ) then
        less = n - 1
        equal = n
        more = n + 1
        return
      end if

10    continue

        if ( lo + 1 .eq. hi ) then
          less = lo
          equal = 0
          more = hi
          go to 20
        end if

        mid = ( lo + hi ) / 2
        xmid = x(indx(mid))

        if ( xval .eq. xmid ) then
          equal = mid
          less = equal - 1
          more = equal + 1
          go to 20
        else if ( xval .lt. xmid ) then
          hi = mid
        else if ( xmid .lt. xval ) then
          lo = mid
        end if

      go to 10

20    continue

      return
      end
      subroutine r8vec_index_sorted_range ( n, r, indx, r_lo, r_hi,
     &  i_lo, i_hi )

c*********************************************************************72
c
cc R8VEC_INDEX_SORTED_RANGE: search index sorted vector for elements in a range.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    27 September 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of items in the vector.
c
c    Input, double precision R(N), the index sorted vector.
c
c    Input, integer INDX(N), the vector used to sort R.
c    The vector R(INDX(*)) is sorted.
c
c    Input, double precision R_LO, R_HI, the limits of the range.
c
c    Output, integer I_LO, I_HI, the range of indices
c    so that I_LO <= I <= I_HI => R_LO <= R(INDX(I)) <= R_HI.  If no
c    values in R lie in the range, then I_HI .lt. I_LO will be returned.
c
      implicit none

      integer n

      integer i_hi
      integer i_lo
      integer i1
      integer i2
      integer indx(n)
      integer j1
      integer j2
      double precision r(n)
      double precision r_hi
      double precision r_lo
c
c  Cases we can handle immediately.
c
      if ( r(indx(n)) .lt. r_lo ) then
        i_lo = n + 1
        i_hi = n
        return
      end if

      if ( r_hi .lt. r(indx(1)) ) then
        i_lo = 1
        i_hi = 0
        return
      end if
c
c  Are there are least two intervals?
c
      if ( n .eq. 1 ) then
        if ( r_lo .le. r(indx(1)) .and. r(indx(1)) .le. r_hi ) then
          i_lo = 1
          i_hi = 1
        else
          i_lo = 0
          i_hi = -1
        end if
        return
      end if
c
c  Bracket R_LO.
c
      if ( r_lo .le. r(indx(1)) ) then

        i_lo = 1

      else
c
c  R_LO is in one of the intervals spanned by R(INDX(J1)) to R(INDX(J2)).
c  Examine the intermediate interval [R(INDX(I1)), R(INDX(I1+1))].
c  Does R_LO lie here, or below or above?
c
        j1 = 1
        j2 = n
        i1 = ( j1 + j2 - 1 ) / 2
        i2 = i1 + 1

10      continue

          if ( r_lo .lt. r(indx(i1)) ) then
            j2 = i1
            i1 = ( j1 + j2 - 1 ) / 2
            i2 = i1 + 1
          else if ( r(indx(i2)) .lt. r_lo ) then
            j1 = i2
            i1 = ( j1 + j2 - 1 ) / 2
            i2 = i1 + 1
          else
            i_lo = i1
            go to 20
          end if

        go to 10

20      continue

      end if
c
c  Bracket R_HI.
c
      if ( r(indx(n)) .le. r_hi ) then

        i_hi = n

      else

        j1 = i_lo
        j2 = n
        i1 = ( j1 + j2 - 1 ) / 2
        i2 = i1 + 1

30      continue

          if ( r_hi .lt. r(indx(i1)) ) then
            j2 = i1
            i1 = ( j1 + j2 - 1 ) / 2
            i2 = i1 + 1
          else if ( r(indx(i2)) .lt. r_hi ) then
            j1 = i2
            i1 = ( j1 + j2 - 1 ) / 2
            i2 = i1 + 1
          else
            i_hi = i2
            go to 40
          end if

        go to 30

40      continue

      end if
c
c  We expect to have computed the largest I_LO and smallest I_HI such that
c    R(INDX(I_LO)) <= R_LO <= R_HI <= R(INDX(I_HI))
c  but what we want is actually
c    R_LO <= R(INDX(I_LO)) <= R(INDX(I_HI)) <= R_HI
c  which we can usually get simply by incrementing I_LO and decrementing I_HI.
c
      if ( r(indx(i_lo)) .lt. r_lo ) then
        i_lo = i_lo + 1
        if ( n .lt. i_lo ) then
          i_hi = i_lo - 1
        end if
      end if

      if ( r_hi .lt. r(indx(i_hi)) ) then
        i_hi = i_hi - 1
        if ( i_hi .lt. 1 ) then
          i_lo = i_hi + 1
        end if
      end if

      return
      end
      subroutine r8vec_indexed_heap_d ( n, a, indx )

c*********************************************************************72
c
cc R8VEC_INDEXED_HEAP_D creates a descending heap from an indexed R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    An indexed R8VEC is an R8VEC of data values, and an R8VEC of N indices,
c    each referencing an entry of the data vector.
c
c    The function adjusts the index vector INDX so that, for 1 .le. J .le. N/2,
c    we have:
c      A(INDX(2*J))   .le. A(INDX(J))
c    and
c      A(INDX(2*J+1)) .le. A(INDX(J))
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    16 August 2010
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Albert Nijenhuis, Herbert Wilf,
c    Combinatorial Algorithms for Computers and Calculators,
c    Academic Press, 1978,
c    ISBN: 0-12-519260-6,
c    LC: QA164.N54.
c
c  Parameters:
c
c    Input, integer N, the size of the index array.
c
c    Input, double precision A(*), the data vector.
c
c    Input/output, integer INDX(N), the index array.
c    Each entry of INDX must be a valid index for the array A.
c    On output, the indices have been reordered into a descending heap.
c
      implicit none

      integer n

      double precision a(*)
      integer i
      integer ifree
      integer indx(n)
      integer key
      integer m
c
c  Only nodes N/2 down to 1 can be "parent" nodes.
c
      do i = n / 2, 1, -1
c
c  Copy the value out of the parent node.
c  Position IFREE is now "open".
c
        key = indx(i)
        ifree = i

10      continue
c
c  Positions 2*IFREE and 2*IFREE + 1 are the descendants of position
c  IFREE.  (One or both may not exist because they exceed N.)
c
          m = 2 * ifree
c
c  Does the first position exist?
c
          if ( n .lt. m ) then
            go to 20
          end if
c
c  Does the second position exist?
c
          if ( m + 1 .le. n ) then
c
c  If both positions exist, take the larger of the two values,
c  and update M if necessary.
c
            if ( a(indx(m)) .lt. a(indx(m+1)) ) then
              m = m + 1
            end if

          end if
c
c  If the large descendant is larger than KEY, move it up,
c  and update IFREE, the location of the free position, and
c  consider the descendants of THIS position.
c
          if ( a(indx(m)) .le. a(key) ) then
            go to 20
          end if

          indx(ifree) = indx(m)
          ifree = m

        go to 10
c
c  Once there is no more shifting to do, KEY moves into the free spot IFREE.
c
20      continue

        indx(ifree) = key

      end do

      return
      end
      subroutine r8vec_indexed_heap_d_extract ( n, a, indx,
     &  indx_extract )

c*********************************************************************72
c
cc R8VEC_INDEXED_HEAP_D_EXTRACT: extract from heap descending indexed R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    An indexed R8VEC is an R8VEC of data values, and an R8VEC of N indices,
c    each referencing an entry of the data vector.
c
c    The routine finds the maximum value in the heap, returns that value to the
c    user, deletes that value from the heap, and restores the heap to its
c    proper form.
c
c    Note that the argument N must be a variable, which will be decremented
c    before return, and that INDX will hold one less value on output than it
c    held on input.
c
c    This is one of three functions needed to model a priority queue.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    16 August 2010
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Thomas Cormen, Charles Leiserson, Ronald Rivest,
c    Introduction to Algorithms,
c    MIT Press, 2001,
c    ISBN: 0262032937,
c    LC: QA76.C662.
c
c  Parameters:
c
c    Input/output, integer N, the number of items in the
c    index vector.
c
c    Input, double precision A(*), the data vector.
c
c    Input/output, integer INDX(N), the index vector.
c
c    Output, integer INDX_EXTRACT, the index in A of the item of
c    maximum value, which has now been removed from the heap.
c
      implicit none

      double precision a(*)
      integer indx(*)
      integer indx_extract
      integer n

      if ( n .lt. 1 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8VEC_INDEXED_HEAP_D_EXTRACT - Fatal error!'
        write ( *, '(a)' ) '  The heap is empty.'
        stop 1
      end if
c
c  Get the index of the maximum value.
c
      indx_extract = indx(1)

      if ( n .eq. 1 ) then
        n = 0
        return
      end if
c
c  Shift the last index down.
c
      indx(1) = indx(n)
c
c  Restore the heap structure.
c
      n = n - 1
      call r8vec_indexed_heap_d ( n, a, indx )

      return
      end
      subroutine r8vec_indexed_heap_d_insert ( n, a, indx, indx_insert )

c*********************************************************************72
c
cc R8VEC_INDEXED_HEAP_D_INSERT: insert value into heap descending indexed R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    An indexed R8VEC is an R8VEC of data values, and an R8VEC of N indices,
c    each referencing an entry of the data vector.
c
c    Note that the argument N must be a variable, and will be incremented before
c    return, and that INDX must be able to hold one more entry on output than
c    it held on input.
c
c    This is one of three functions needed to model a priority queue.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    17 August 2010
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Thomas Cormen, Charles Leiserson, Ronald Rivest,
c    Introduction to Algorithms,
c    MIT Press, 2001,
c    ISBN: 0262032937,
c    LC: QA76.C662.
c
c  Parameters:
c
c    Input/output, integer N, the number of items in the
c    index vector.
c
c    Input, double precision A(*), the data vector.
c
c    Input/output, integer INDX(N), the index vector.
c
c    Input, integer INDX_INSERT, the index in A of the value
c    to be inserted into the heap.
c
      implicit none

      double precision a(*)
      integer i
      integer indx(*)
      integer indx_insert
      integer n
      integer parent

      n = n + 1
      i = n

10    continue

      if ( 1 .lt. i ) then

        parent = i / 2

        if ( a(indx_insert) .le. a(indx(parent)) ) then
          go to 20
        end if

        indx(i) = indx(parent)
        i = parent

        go to 10

      end if

20    continue

      indx(i) = indx_insert

      return
      end
      subroutine r8vec_indexed_heap_d_max ( n, a, indx, indx_max )

c*********************************************************************72
c
cc R8VEC_INDEXED_HEAP_D_MAX: maximum value in heap descending indexed R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    An indexed R8VEC is an R8VEC of data values, and an R8VEC of N indices,
c    each referencing an entry of the data vector.
c
c    This is one of three functions needed to model a priority queue.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    16 August 2010
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Thomas Cormen, Charles Leiserson, Ronald Rivest,
c    Introduction to Algorithms,
c    MIT Press, 2001,
c    ISBN: 0262032937,
c    LC: QA76.C662.
c
c  Parameters:
c
c    Input, integer N, the number of items in the index vector.
c
c    Input, double precision A(*), the data vector.
c
c    Input, integer INDX(N), the index vector.
c
c    Output, integer INDX_MAX, the index in A of the maximum value
c    in the heap.
c
      implicit none

      integer n

      double precision a(*)
      integer indx(n)
      integer indx_max

      indx_max = indx(1)

      return
      end
      subroutine r8vec_indicator0 ( n, a )

c*********************************************************************72
c
cc R8VEC_INDICATOR0 sets an R8VEC to the indicator vector (0,1,2,...).
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    27 September 2014
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of elements of A.
c
c    Output, double precision A(N), the array.
c
      implicit none

      integer n

      double precision a(n)
      integer i

      do i = 1, n
        a(i) = dble ( i - 1 )
      end do

      return
      end
      subroutine r8vec_indicator1 ( n, a )

c*********************************************************************72
c
cc R8VEC_INDICATOR1 sets an R8VEC to the indicator vector (1,2,3,...).
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    27 September 2014
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of elements of A.
c
c    Output, double precision A(N), the array.
c
      implicit none

      integer n

      double precision a(n)
      integer i

      do i = 1, n
        a(i) = dble ( i )
      end do

      return
      end
      subroutine r8vec_insert ( n, a, pos, value )

c*********************************************************************72
c
cc R8VEC_INSERT inserts a value into an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    30 May 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the dimension of the array on input.
c
c    Input/output, double precision A(N+1), the array.  On input, A is
c    assumed to contain only N entries, while on output, A actually
c    contains N+1 entries.
c
c    Input, integer POS, the position to be assigned the new entry.
c    1 <= POS <= N+1.
c
c    Input, double precision VALUE, the value to be inserted.
c
      implicit none

      integer n

      double precision a(n+1)
      integer i
      integer pos
      double precision value

      if ( pos .lt. 1 .or. n + 1 .lt. pos ) then

        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8VEC_INSERT - Fatal error!'
        write ( *, '(a,i8)' ) '  Illegal insertion position = ', pos
        stop 1

      else

        do i = n + 1, pos + 1, -1
          a(i) = a(i-1)
        end do

        a(pos) = value

      end if

      return
      end
      function r8vec_insignificant ( n, r, s )

c*********************************************************************72
c
cc R8VEC_INSIGNIFICANT determines if an R8VEC is insignificant.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    26 November 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the dimension of the vectors.
c
c    Input, double precision R(N), the vector to be compared against.
c
c    Input, double precision S(N), the vector to be compared.
c
c    Output, logical R8VEC_INSIGNIFICANT, is TRUE if S is insignificant
c    compared to R.
c
      implicit none

      integer n

      integer i
      double precision r(n)
      double precision r8_epsilon
      logical r8vec_insignificant
      double precision s(n)
      double precision t
      double precision tol
      logical value

      value = .true.

      do i = 1, n

        t = r(i) + s(i)
        tol = r8_epsilon ( ) * abs ( r(i) )

        if ( tol .lt. abs ( r(i) - t ) ) then 
          value = .false.
          go to 10
        end if

      end do
  
10    continue

      r8vec_insignificant = value

      return
      end
      function r8vec_is_int ( n, a )

c*********************************************************************72
c
cc R8VEC_IS_INT is TRUE if the entries of an R8VEC are integers.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    04 June 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in A.
c
c    Input, double precision A(N), the vector.
c
c    Output, logical R8VEC_IS_INT, is TRUE if every entry of A is
c    integral.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      logical r8vec_is_int

      do i = 1, n
        if ( a(i) .ne. aint ( a(i) ) ) then
          r8vec_is_int = .false.
          return
        end if
      end do

      r8vec_is_int = .true.

      return
      end
      function r8vec_is_nonnegative ( n, a )

c*****************************************************************************80
c
cc R8VEC_IS_NONNEGATIVE is TRUE if all the entries of an R8VEC are nonnegative.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    17 June 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in A.
c
c    Input, double precision A(N), the vector.
c
c    Output, logical R8VEC_IS_NONNEGATIVE, the value of the condition.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      logical r8vec_is_nonnegative

      do i = 1, n
        if ( a(i) .lt. 0.0D+00 ) then
          r8vec_is_nonnegative = .false.
          return
        end if
      end do

      r8vec_is_nonnegative = .true.

      return
      end
      function r8vec_is_zero ( n, a )

c*****************************************************************************80
c
cc R8VEC_IS_ZERO is TRUE if all the entries of an R8VEC are zero.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    17 June 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in A.
c
c    Input, double precision A(N), the vector.
c
c    Output, logical R8VEC_IS_ZERO, the value of the condition.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      logical r8vec_is_zero

      do i = 1, n
        if ( a(i) .ne. 0.0D+00 ) then
          r8vec_is_zero = .false.
          return
        end if
      end do

      r8vec_is_zero = .true.

      return
      end
      subroutine r8vec_legendre ( n, x_first, x_last, x )

c*****************************************************************************80
c
cc R8VEC_LEGENDRE creates a vector of Legendre-spaced values.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    17 June 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the vector.
c
c    Input, double precision X_FIRST, X_LAST, the first and last entries.
c
c    Output, double precision X(N), a vector of Legendre-spaced data.
c
      implicit none

      integer n

      integer i
      double precision x(n)
      double precision x_first
      double precision x_last

      call legendre_zeros ( n, x )

      do i = 1, n
        x(i) = ( ( 1.0D+00 - x(i) ) * x_first  
     &         + ( 1.0D+00 + x(i) ) * x_last ) 
     &         /   2.0D+00
      end do

      return
      end
      subroutine r8vec_linspace ( n, a, b, x )

c*********************************************************************72
c
cc R8VEC_LINSPACE creates a vector of linearly spaced values.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    4 points evenly spaced between 0 and 12 will yield 0, 4, 8, 12.
c
c    In other words, the interval is divided into N-1 even subintervals,
c    and the endpoints of intervals are used as the points.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    14 March 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the vector.
c
c    Input, double precision A, B, the first and last entries.
c
c    Output, double precision X(N), a vector of linearly spaced data.
c
      implicit none

      integer n

      double precision a
      double precision b
      integer i
      double precision x(n)

      if ( n .eq. 1 ) then

        x(1) = ( a + b ) / 2.0D+00

      else

        do i = 1, n
          x(i) = ( dble ( n - i     ) * a
     &           + dble (     i - 1 ) * b )
     &           / dble ( n     - 1 )
        end do

      end if

      return
      end
      subroutine r8vec_linspace2 ( n, a, b, x )

c*********************************************************************72
c
cc R8VEC_LINSPACE2 creates a vector of linearly spaced values.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    5 points evenly spaced between 0 and 12 will yield 2, 4, 6, 8, 10.
c
c    In other words, the interval is divided into N+1 even subintervals,
c    and the endpoints of internal intervals are used as the points.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    18 September 2012
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the vector.
c
c    Input, double precision A, B, the first and last entries.
c
c    Output, double precision X(N), a vector of linearly spaced data.
c
      implicit none

      integer n

      double precision a
      double precision b
      integer i
      double precision x(n)

      do i = 1, n
        x(i) = ( dble ( n  - i + 1 ) * a 
     &         + dble (      i     ) * b ) 
     &         / dble ( n      + 1 )
      end do

      return
      end
      function r8vec_lt ( n, a1, a2 )

c*********************************************************************72
c
cc R8VEC_LT evaluates the expression ( A1 < A2 ) for R8VEC's.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    The comparison is lexicographic.
c
c    A1 < A2  <=>                              A1(1) < A2(1) or
c                 ( A1(1)     .eq. A2(1)     and A1(2) < A2(2) ) or
c                 ...
c                 ( A1(1:N-1) .eq. A2(1:N-1) and A1(N) < A2(N)
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    22 July 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the dimension of the vectors.
c
c    Input, double precision A1(N), A2(N), the vectors to be compared.
c
c    Output, logical R8VEC_LT, is TRUE if and only if A1 < A2.
c
      implicit none

      integer n

      double precision a1(n)
      double precision a2(n)
      integer i
      logical r8vec_lt

      r8vec_lt = .false.

      do i = 1, n

        if ( a1(i) .lt. a2(i) ) then
          r8vec_lt = .true.
          return
        else if ( a2(i) .lt. a1(i) ) then
          return
        end if

      end do

      return
      end
      subroutine r8vec_mask_print ( n, a, mask_num, mask, title )

c*********************************************************************72
c
cc R8VEC_MASK_PRINT prints a masked R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    06 July 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of components of the vector.
c
c    Input, double precision A(N), the vector to be printed.
c
c    Input, integer MASK_NUM, the number of masked elements.
c
c    Input, integer MASK(MASK_NUM), the indices of the vector
c    to be printed.
c
c    Input, character * ( * ) TITLE, a title.
c
      implicit none

      integer mask_num
      integer n

      double precision a(n)
      integer i
      integer mask(mask_num)
      character * ( * ) title

      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) '  Masked vector printout:'
      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) trim ( title )

      write ( *, '(a)' ) ' '
      do i = 1, mask_num
        write ( *, '(2x,i8,a,1x,i8,2x,g14.6)' )
     &    i, ':', mask(i), a(mask(i))
      end do

      return
      end
      function r8vec_max ( n, a )

c*********************************************************************72
c
cc R8VEC_MAX returns the maximum value in an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    12 May 2014
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the array.
c
c    Input, double precision A(N), the array.
c
c    Output, double precision R8VEC_MAX, the value of the largest entry.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      double precision r8_huge
      parameter ( r8_huge = 1.79769313486231571D+308 )
      double precision r8vec_max
      double precision value

      value = - r8_huge
      do i = 1, n
        value = max ( value, a(i) )
      end do

      r8vec_max = value

      return
      end
      subroutine r8vec_max_abs_index ( n, a, max_index )

c*********************************************************************72
c
cc R8VEC_MAX_ABS_INDEX: index of the maximum absolute value in an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    08 April 2012
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the array.
c
c    Input, double precision A(N), the array.
c
c    Output, integer MAX_INDEX, the index of the largest entry.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      integer max_index

      if ( n .le. 0 ) then

        max_index = -1

      else

        max_index = 1

        do i = 2, n
          if ( abs ( a(max_index) ) .lt. abs ( a(i) ) ) then
            max_index = i
          end if
        end do

      end if

      return
      end
      subroutine r8vec_max_index ( n, a, max_index )

c*********************************************************************72
c
cc R8VEC_MAX_INDEX returns the index of the maximum value in an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    06 July 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the array.
c
c    Input, double precision A(N), the array.
c
c    Output, integer MAX_INDEX, the index of the largest entry.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      integer max_index

      if ( n .le. 0 ) then

        max_index = -1

      else

        max_index = 1

        do i = 2, n
          if ( a(max_index) .lt. a(i) ) then
            max_index = i
          end if
        end do

      end if

      return
      end
      subroutine r8vec_mean ( n, a, mean )

c*********************************************************************72
c
cc R8VEC_MEAN returns the mean of an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    29 May 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the vector.
c
c    Input, double precision A(N), the vector whose mean is desired.
c
c    Output, double precision MEAN, the mean of the vector entries.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      double precision mean

      mean = 0.0D+00
      do i = 1, n
        mean = mean + a(i)
      end do
      mean = mean / dble ( n )

      return
      end
      subroutine r8vec_mean_geometric ( n, a, mean )

c*********************************************************************72
c
cc R8VEC_MEAN_GEOMETRIC returns the geometric mean of an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    27 April 2014
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the vector.
c
c    Input, double precision A(N), the vector whose mean is desired.
c
c    Output, double precision MEAN, the geometric mean of the vector entries.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      double precision mean

      mean = 0.0D+00
      do i = 1, n
        mean = mean + log ( a(i) )
      end do
      mean = mean / dble ( n )
      mean = exp ( mean )

      return
      end
      subroutine r8vec_median ( n, a, median )

c*********************************************************************72
c
cc R8VEC_MEDIAN returns the median of an unsorted R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    Hoare's algorithm is used.  The values of the vector are
c    rearranged by this routine.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    02 June 2009
c
c  Parameters:
c
c    Input, integer N, the number of elements of A.
c
c    Input/output, double precision A(N), the array to search.  On output,
c    the order of the elements of A has been somewhat changed.
c
c    Output, double precision MEDIAN, the value of the median of A.
c
      implicit none

      integer n

      double precision a(n)
      integer k
      double precision median

      k = ( n + 1 ) / 2

      call r8vec_frac ( n, a, k, median )

      return
      end
      subroutine r8vec_mesh_2d ( nx, ny, xvec, yvec, xmat, ymat )

c*********************************************************************72
c
cc R8VEC_MESH_2D creates a 2D mesh from X and Y vectors.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    NX = 2
c    XVEC = ( 1, 2, 3 )
c    NY = 3
c    YVEC = ( 4, 5 )
c
c    XMAT = (
c      1, 2, 3
c      1, 2, 3 )
c
c    YMAT = (
c      4, 4, 4
c      5, 5, 5 ) 
c    
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    26 July 2013
c
c  Parameters:
c
c    Input, integer NX, NY, the number of X and Y values.
c
c    Input, double precision XVEC(NX), YVEC(NY), the X and Y coordinate
c    values.
c
c    Output, double precision XMAT(NX,NY), YMAT(NX,NY), the coordinate
c    values of points on an NX by NY mesh.
c
      implicit none

      integer nx
      integer ny

      integer i
      integer j
      double precision xmat(nx,ny)
      double precision xvec(nx)
      double precision ymat(nx,ny)
      double precision yvec(ny)

      do j = 1, ny
        do i = 1, nx
          xmat(i,j) = xvec(i)
        end do
      end do

      do j = 1, ny
        do i = 1, nx
          ymat(i,j) = yvec(j)
        end do
      end do

      return
      end
      subroutine r8vec_midspace ( n, a, b, x )

c*********************************************************************72
c
cc R8VEC_MIDSPACE creates a vector of linearly spaced values.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    This function divides the interval [a,b] into n subintervals, and then
c    returns the midpoints of those subintervals.
c
c  Example:
c
c    N = 5, A = 10, B = 20
c    X = [ 11, 13, 15, 17, 19 ]
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    03 June 2012
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the vector.
c
c    Input, double precision A, B, the endpoints of the interval.
c
c    Output, double precision X(N), a vector of linearly spaced data.
c
      implicit none

      integer n

      double precision a
      double precision b
      integer i
      double precision x(n)

      do i = 1, n
        x(i) = ( dble ( 2 * n - 2 * i + 1 ) * a 
     &         + dble (         2 * i - 1 ) * b ) 
     &         / dble ( 2 * n )
      end do

      return
      end
      function r8vec_min ( n, a )

c*********************************************************************72
c
cc R8VEC_MIN returns the minimum value in an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    18 January 2015
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the array.
c
c    Input, double precision A(N), the array.
c
c    Output, double precision R8VEC_MIN, the value of the smallest entry.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      double precision r8_huge
      parameter ( r8_huge = 1.79769313486231571D+308 )
      double precision r8vec_min
      double precision value

      value = r8_huge
      do i = 1, n
        value = min ( value, a(i) )
      end do

      r8vec_min = value

      return
      end
      subroutine r8vec_min_index ( n, a, min_index )

c*********************************************************************72
c
cc R8VEC_MIN_INDEX returns the index of the minimum value in an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    07 July 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the array.
c
c    Input, double precision A(N), the array.
c
c    Output, integer MIN_INDEX, the index of the smallest entry.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      integer min_index

      if ( n .le. 0 ) then

        min_index = -1

      else

        min_index = 1

        do i = 2, n
          if ( a(i) .lt. a(min_index) ) then
            min_index = i
          end if
        end do

      end if

      return
      end
      function r8vec_min_pos ( n, a )

c*********************************************************************72
c
cc R8VEC_MIN_POS returns the minimum positive value of an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    08 November 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries.
c
c    Input, double precision A(N), the array.
c
c    Output, double precision R8VEC_MIN_POS, the smallest positive entry.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      double precision r8_huge
      parameter ( r8_huge = 1.79769313486231571D+308 )
      double precision r8vec_min_pos
      double precision value

      value = r8_huge

      do i = 1, n
        if ( 0.0D+00 .lt. a(i) ) then
          value = min ( value, a(i) )
        end if
      end do

      r8vec_min_pos = value

      return
      end
      subroutine r8vec_mirror_next ( n, a, done )

c*********************************************************************72
c
cc R8VEC_MIRROR_NEXT steps through all sign variations of an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    In normal use, the user would set every element of A to be positive.
c    The routine will take the input value of A, and output a copy in
c    which the signs of one or more entries have been changed.  Repeatedly
c    calling the routine with the output from the previous call will generate
c    every distinct "variation" of A; that is, all possible sign variations.
c
c    When the output variable DONE is TRUE (or equal to 1), then the
c    output value of A_NEW is the last in the series.
c
c    Note that A may have some zero values.  The routine will essentially
c    ignore such entries; more exactly, it will not stupidly assume that -0
c    is a proper "variation" of 0c
c
c    Also, it is possible to call this routine with the signs of A set
c    in any way you like.  The routine will operate properly, but it
c    will nonethess terminate when it reaches the value of A in which
c    every nonzero entry has negative sign.
c
c    More efficient algorithms using the Gray code seem to require internal
c    memory in the routine, which is not one of MATLAB's strong points,
c    or the passing back and forth of a "memory array", or the use of
c    global variables, or unnatural demands on the user.  This form of
c    the routine is about as clean as I can make it.
c
c  Example:
c
c      Input         Output
c    ---------    --------------
c    A            A_NEW     DONE
c    ---------    --------  ----
c     1  2  3     -1  2  3  false
c    -1  2  3      1 -2  3  false
c     1 -2  3     -1 -2  3  false
c    -1 -2  3      1  2 -3  false
c     1  2 -3     -1  2 -3  false
c    -1  2 -3      1 -2 -3  false
c     1 -2 -3     -1 -2 -3  false
c    -1 -2 -3      1  2  3  true
c
c     1  0  3     -1  0  3  false
c    -1  0  3      1  0 -3  false
c     1  0 -3     -1  0 -3  false
c    -1  0 -3      1  0  3  true
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    07 July 2009
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Albert Nijenhuis, Herbert Wilf,
c    Combinatorial Algorithms for Computers and Calculators,
c    Academic Press, 1978,
c    ISBN: 0-12-519260-6,
c    LC: QA164.N54.
c
c  Parameters:
c
c    Input, integer N, the number of entries in the vector.
c
c    Input/output, double precision A(N), a vector of real numbers.
c    On output, the signs of some entries have been changed.
c
c    Output, logical DONE, is TRUE if the input vector A was the last element
c    in the series (every entry was nonpositive); the output vector is reset
c    so that all entries are nonnegative, but presumably the ride is overc
c
      implicit none

      integer n

      double precision a(n)
      logical done
      integer i
      integer positive
c
c  Seek the first strictly positive entry of A.
c
      positive = 0
      do i = 1, n
        if ( 0.0D+00 .lt. a(i) ) then
          positive = i
          go to 10
        end if
      end do

10    continue
c
c  If there is no strictly positive entry of A, there is no successor.
c
      if ( positive .eq. 0 ) then
        do i = 1, n
          a(i) = - a(i)
        end do
        done = .true.
        return
      end if
c
c  Otherwise, negate A up to the positive entry.
c
      do i = 1, positive
        a(i) = - a(i)
      end do

      done = .false.

      return
      end
      function r8vec_negative_strict ( n, a )

c*********************************************************************72
c
cc R8VEC_NEGATIVE_STRICT: every element of an R8VEC is strictly negative.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    24 June 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the vector.
c
c    Input, double precision A(N).
c
c    Output, logical R8VEC_NEGATIVE_STRICT, is TRUE every entry of the
c    vector is strictly negative.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      logical r8vec_negative_strict

      r8vec_negative_strict = .true.

      do i = 1, n
        if ( 0.0D+00 .le. a(i) ) then
          r8vec_negative_strict = .false.
          return
        end if
      end do

      return
      end
      subroutine r8vec_nint ( n, a )

c*********************************************************************72
c
cc R8VEC_NINT rounds entries of an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    07 July 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the vector.
c
c    Input/output, double precision A(N), the vector to be NINT'ed.
c
      implicit none

      integer n

      double precision a(n)
      integer i

      do i = 1, n
        a(i) = nint ( dble ( a(i) ) )
      end do

      return
      end
      function r8vec_norm ( n, a )

c*********************************************************************72
c
cc R8VEC_NORM returns the L2 norm of an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8 values.
c
c    The vector L2 norm is defined as:
c
c      R8VEC_NORM = sqrt ( sum ( 1 <= I <= N ) A(I)^2 ).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    27 May 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in A.
c
c    Input, double precision A(N), the vector whose L2 norm is desired.
c
c    Output, double precision R8VEC_NORM, the L2 norm of A.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      double precision r8vec_norm
      double precision value

      value = 0.0D+00
      do i = 1, n
        value = value + a(i) * a(i)
      end do
      value = sqrt ( value )

      r8vec_norm = value

      return
      end
      function r8vec_norm_affine ( n, v0, v1 )

c*********************************************************************72
c
cc R8VEC_NORM_AFFINE returns the affine norm of an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    The affine vector L2 norm is defined as:
c
c      R8VEC_NORM_AFFINE(V0,V1)
c        = sqrt ( sum ( 1 <= I <= N ) ( V1(I) - V0(I) )^2 )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    27 October 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the order of the vectors.
c
c    Input, double precision V0(N), the base vector.
c
c    Input, double precision V1(N), the vector whose affine norm is desired.
c
c    Output, double precision R8VEC_NORM_AFFINE, the L2 norm of V1-V0.
c
      implicit none

      integer n

      integer i
      double precision r8vec_norm_affine
      double precision v0(n)
      double precision v1(n)

      r8vec_norm_affine = 0.0D+00
      do i = 1, n
        r8vec_norm_affine = r8vec_norm_affine
     &    + ( v0(i) - v1(i) ) ** 2
      end do
      r8vec_norm_affine = sqrt ( r8vec_norm_affine )

      return
      end
      function r8vec_norm_l0 ( n, a )

c*********************************************************************72
c
cc R8VEC_NORM_L0 returns the l0 "norm" of an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    The l0 "norm" simply counts the number of nonzero entries in the vector.
c    It is not a true norm, but has some similarities to one.  It is useful
c    in the study of compressive sensing.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    01 January 2015
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the vector.
c
c    Input, double precision A(N), the vector.
c
c    Output, double precision R8VEC_NORM_L0, the value of the norm.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      double precision r8vec_norm_l0
      double precision value

      value = 0.0
      do i = 1, n
        if ( a(i) .ne. 0.0D+00 ) then
          value = value + 1.0D+00
        end if
      end do

      r8vec_norm_l0 = value

      return
      end
      function r8vec_norm_l1 ( n, a )

c*********************************************************************72
c
cc R8VEC_NORM_L1 returns the L1 norm of an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    The vector L1 norm is defined as:
c
c      R8VEC_NORM_L1 = sum ( 1 <= I <= N ) abs ( A(I) ).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    07 July 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in A.
c
c    Input, double precision A(N), the vector whose L1 norm is desired.
c
c    Output, double precision R8VEC_NORM_L1, the L1 norm of A.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      double precision r8vec_norm_l1

      r8vec_norm_l1 = 0.0D+00
      do i = 1, n
        r8vec_norm_l1 = r8vec_norm_l1 + abs ( a(i) )
      end do

      return
      end
      function r8vec_norm_l2 ( n, a )

c*********************************************************************72
c
cc R8VEC_NORM_L2 returns the L2 norm of an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8 values.
c
c    The vector L2 norm is defined as:
c
c      R8VEC_NORM_L2 = sqrt ( sum ( 1 <= I <= N ) A(I)^2 ).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    27 May 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in A.
c
c    Input, double precision A(N), the vector whose L2 norm is desired.
c
c    Output, double precision R8VEC_NORM_L2, the L2 norm of A.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      double precision r8vec_norm_l2
      double precision value

      value = 0.0D+00
      do i = 1, n
        value = value + a(i) * a(i)
      end do
      value = sqrt ( value )

      r8vec_norm_l2 = value

      return
      end
      function r8vec_norm_li ( n, a )

c*********************************************************************72
c
cc R8VEC_NORM_LI returns the L-infinity norm of an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    The vector L-infinity norm is defined as:
c
c      R8VEC_NORM_LI = max ( 1 <= I <= N ) abs ( A(I) ).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    07 July 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in A.
c
c    Input, double precision A(N), the vector whose L-infinity norm is desired.
c
c    Output, double precision R8VEC_NORM_LI, the L-infinity norm of A.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      double precision r8vec_norm_li

      r8vec_norm_li = 0.0D+00
      do i = 1, n
        r8vec_norm_li = max ( r8vec_norm_li, abs ( a(i) ) )
      end do

      return
      end
      function r8vec_norm_lp ( n, a, p )

c*********************************************************************72
c
cc R8VEC_NORM_LP returns the LP norm of an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    The vector LP norm is defined as:
c
c      R8VEC_NORM_LP = ( sum ( 1 <= I <= N ) ( abs ( A(I) ) )^P )^(1/P).
c
c    Usually, the LP norms with
c      1 <= P <= oo
c    are of interest.  This routine allows
c      0 < P <= Huge ( P ).
c    If P = Huge ( P ), then the L-oo norm is returned, which is
c    simply the maximum of the absolute values of the vector components.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    04 June 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in A.
c
c    Input, double precision A(N), the vector whose LP norm is desired.
c
c    Input, double precision P, the index of the norm.
c
c    Output, double precision R8VEC_NORM_LP, the LP norm of A.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      double precision p
      double precision r8vec_norm_lp

      if ( p .le. 0.0D+00 ) then
        r8vec_norm_lp = -1.0D+00
      else if ( p .eq. 1.0D+00 ) then
        r8vec_norm_lp = 0.0D+00
        do i = 1, n
          r8vec_norm_lp = r8vec_norm_lp + abs ( a(i) )
        end do
      else if ( p .eq. 2.0D+00 ) then
        r8vec_norm_lp = 0.0D+00
        do i = 1, n
          r8vec_norm_lp = r8vec_norm_lp + a(i) * a(i)
        end do
        r8vec_norm_lp = sqrt ( r8vec_norm_lp )
      else
        r8vec_norm_lp = 0.0D+00
        do i = 1, n
          r8vec_norm_lp = r8vec_norm_lp + ( abs ( a(i) ) ) ** p
        end do
        r8vec_norm_lp = ( r8vec_norm_lp ) ** ( 1.0D+00 / p )
      end if

      return
      end
      function r8vec_norm_squared ( n, a )

c*********************************************************************72
c
cc R8VEC_NORM_SQUARED returns the square of the L2 norm of an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8 values.
c
c    R8VEC_NORM_SQUARED = sum ( 1 <= I <= N ) A(I)^2
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    30 March 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in A.
c
c    Input, double precision A(N), the vector.
c
c    Output, double precision R8VEC_NORM_SQUARED, the square of the L2 norm.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      double precision r8vec_norm_squared
      double precision value

      value = 0.0D+00
      do i = 1, n
        value = value + a(i) * a(i)
      end do

      r8vec_norm_squared = value

      return
      end
      subroutine r8vec_normal_01 ( n, seed, x )

c*********************************************************************72
c
cc R8VEC_NORMAL_01 returns a unit pseudonormal R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    The standard normal probability distribution function (PDF) has
c    mean 0 and standard deviation 1.
c
c    This routine can generate a vector of values on one call.  It
c    has the feature that it should provide the same results
c    in the same order no matter how we break up the task.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    07 January 2014
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of values desired.  If N is negative,
c    then the code will flush its internal memory; in particular,
c    if there is a saved value to be used on the next call, it is
c    instead discarded.  This is useful if the user has reset the
c    random number seed, for instance.
c
c    Input/output, integer SEED, a seed for the random number generator.
c
c    Output, double precision X(N), a sample of the standard normal PDF.
c
c  Local parameters:
c
c    Local, integer X_LO_INDEX, X_HI_INDEX, records the range of entries of
c    X that we need to compute.  This starts off as 1:N, but is adjusted
c    if we have a saved value that can be immediately stored in X(1),
c    and so on.
c
      implicit none

      integer n

      integer i
      integer m
      double precision r(n+1)
      double precision r8_pi
      parameter ( r8_pi = 3.141592653589793D+00 )
      double precision r8_uniform_01
      integer seed
      double precision x(n)
      integer x_hi_index
      integer x_lo_index
c
c  Record the range of X we need to fill in.
c
      x_lo_index = 1
      x_hi_index = n
c
c  Maybe we don't need any more values.
c
      if ( x_hi_index - x_lo_index + 1 .eq. 1 ) then

        r(1) = r8_uniform_01 ( seed )

        if ( r(1) .eq. 0.0D+00 ) then
          write ( *, '(a)' ) ' '
          write ( *, '(a)' ) 'R8VEC_NORMAL_01 - Fatal error!'
          write ( *, '(a)' ) '  R8_UNIFORM_01 returned a value of 0.'
          stop 1
        end if

        r(2) = r8_uniform_01 ( seed )

        x(x_hi_index) =
     &           sqrt ( -2.0D+00 * log ( r(1) ) )
     &           * cos ( 2.0D+00 * r8_pi * r(2) )
c
c  If we require an even number of values, that's easy.
c
      else if ( mod ( x_hi_index - x_lo_index + 1, 2 ) .eq. 0 ) then

        m = ( x_hi_index - x_lo_index + 1 ) / 2

        call r8vec_uniform_01 ( 2 * m, seed, r )

        do i = 1, 2 * m, 2

          x(x_lo_index+i-1) =
     &      sqrt ( -2.0D+00 * log ( r(i) ) )
     &      * cos ( 2.0D+00 * r8_pi * r(i+1) )

          x(x_lo_index+i) =
     &      sqrt ( -2.0D+00 * log ( r(i) ) )
     &      * sin ( 2.0D+00 * r8_pi * r(i+1) )

        end do
c
c  If we require an odd number of values, we generate an even number,
c  and handle the last pair specially, storing one in X(N), and
c  saving the other for later.
c
      else

        x_hi_index = x_hi_index - 1

        m = ( x_hi_index - x_lo_index + 1 ) / 2 + 1

        call r8vec_uniform_01 ( 2 * m, seed, r )

        do i = 1, 2 * m - 3, 2

          x(x_lo_index+i-1) =
     &      sqrt ( -2.0D+00 * log ( r(i) ) )
     &      * cos ( 2.0D+00 * r8_pi * r(i+1) )

          x(x_lo_index+i) =
     &      sqrt ( -2.0D+00 * log ( r(i) ) )
     &      * sin ( 2.0D+00 * r8_pi * r(i+1) )

        end do

        x(n) = sqrt ( -2.0D+00 * log ( r(2*m-1) ) )
     &    * cos ( 2.0D+00 * r8_pi * r(2*m) )

      end if

      return
      end
      subroutine r8vec_normalize ( n, a )

c*********************************************************************72
c
cc R8VEC_NORMALIZE normalizes an R8VEC in the Euclidean norm.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    The euclidean norm is also sometimes called the l2 or
c    least squares norm.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    17 August 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the dimension of the vector.
c
c    Input/output, double precision A(N), the vector to be normalized.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      double precision norm

      norm = 0.0D+00
      do i = 1, n
        norm = norm + a(i) * a(i)
      end do
      norm = sqrt ( norm )

      if ( norm .ne. 0.0D+00 ) then
        do i = 1, n
          a(i) = a(i) / norm
        end do
      end if

      return
      end
      subroutine r8vec_normalize_l1 ( n, a )

c*********************************************************************72
c
cc R8VEC_NORMALIZE_L1 normalizes an R8VEC to have unit sum.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    17 August 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the vector.
c
c    Input/output, double precision A(N), the vector to be normalized.
c    On output, the entries of A should have unit sum.  However, if
c    the input vector has zero sum, the routine halts.
c
      implicit none

      integer n

      double precision a(n)
      double precision a_sum
      integer i

      a_sum = 0.0D+00
      do i = 1, n
        a_sum = a_sum + a(i)
      end do

      if ( a_sum .eq. 0.0D+00 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8VEC_NORMALIZE_L1 - Fatal error!'
        write ( *, '(a)' ) '  The vector entries sum to 0.'
        stop 1
      end if

      do i = 1, n
        a(i) = a(i) / a_sum
      end do

      return
      end
      function r8vec_normsq ( n, v )

c*********************************************************************72
c
cc R8VEC_NORMSQ returns the square of the L2 norm of an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    The square of the vector L2 norm is defined as:
c
c      R8VEC_NORMSQ = sum ( 1 <= I <= N ) V(I)^2.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    28 October 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the vector dimension.
c
c    Input, double precision V(N), the vector.
c
c    Output, double precision R8VEC_NORMSQ, the squared L2 norm.
c
      implicit none

      integer n

      integer i
      double precision r8vec_normsq
      double precision value
      double precision v(n)

      value = 0.0D+00
      do i = 1, n
        value = value + v(i) * v(i)
      end do
      
      r8vec_normsq = value

      return
      end
      function r8vec_normsq_affine ( n, v0, v1 )

c*****************************************************************************80
c
cc R8VEC_NORMSQ_AFFINE returns the affine squared norm of an R8VEC.
c
c  Discussion:
c
c   An R8VEC is a vector of R8's.
c
c    The affine squared vector L2 norm is defined as:
c
c      R8VEC_NORMSQ_AFFINE(V0,V1)
c        = sum ( 1 <= I <= N ) ( V1(I) - V0(I) )^2
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    28 October 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the vector dimension.
c
c    Input, double precision V0(N), the base vector.
c
c    Input, double precision V1(N), the vector.
c
c    Output, double precision R8VEC_NORMSQ_AFFINE, the squared affine L2 norm.
c
      implicit none

      integer n

      integer i
      double precision r8vec_normsq_affine
      double precision v0(n)
      double precision v1(n)

      r8vec_normsq_affine = 0.0D+00
      do i = 1, n
        r8vec_normsq_affine = r8vec_normsq_affine + ( v0(i) - v1(i) )**2
      end do

      return
      end
      subroutine r8vec_ones ( n, a )

c*********************************************************************72
c
cc R8VEC_ONES returns a vector of 1's.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    14 March 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the length of the vector.
c
c    Output, double precision A(N), a vector of 1's.
c
      implicit none

      integer n

      double precision a(n)
      integer i

      do i = 1, n
        a(i) = 1.0D+00
      end do

      return
      end
      subroutine r8vec_order_type ( n, a, order )

c*********************************************************************72
c
cc R8VEC_ORDER_TYPE determines if R8VEC is (non)strictly ascending/descending.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    19 August 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries of the array.
c
c    Input, double precision A(N), the array to be checked.
c
c    Output, integer ORDER, order indicator:
c    -1, no discernable order;
c    0, all entries are equal;
c    1, ascending order;
c    2, strictly ascending order;
c    3, descending order;
c    4, strictly descending order.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      integer order
c
c  Search for the first value not equal to A(1).
c
      i = 1

10    continue

        i = i + 1

        if ( n .lt. i ) then
          order = 0
          return
        end if

        if ( a(1) .lt. a(i) ) then

          if ( i .eq. 2 ) then
            order = 2
          else
            order = 1
          end if

          go to 20

        else if ( a(i) .lt. a(1) ) then

          if ( i .eq. 2 ) then
            order = 4
          else
            order = 3
          end if

          go to 20

        end if

      go to 10

20    continue
c
c  Now we have a "direction".  Examine subsequent entries.
c
30    continue

      if ( i .lt. n ) then

        i = i + 1

        if ( order .eq. 1 ) then

          if ( a(i) .lt. a(i-1) ) then
            order = -1
            go to 40
          end if

        else if ( order .eq. 2 ) then

          if ( a(i) .lt. a(i-1) ) then
            order = -1
            go to 40
          else if ( a(i) .eq. a(i-1) ) then
            order = 1
          end if

        else if ( order .eq. 3 ) then

          if ( a(i-1) .lt. a(i) ) then
            order = -1
            go to 40
          end if

        else if ( order .eq. 4 ) then

          if ( a(i-1) .lt. a(i) ) then
            order = -1
            go to 40
          else if ( a(i) .eq. a(i-1) ) then
            order = 3
          end if

        end if

        go to 30

      end if

40    continue

      return
      end
      subroutine r8vec_part_quick_a ( n, a, l, r )

c*********************************************************************72
c
cc R8VEC_PART_QUICK_A reorders an R8VEC as part of a quick sort.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    The routine reorders the entries of A.  Using A(1) as the key,
c    all entries of A that are less than or equal to the key will
c    precede the key which precedes all entries that are greater than the key.
c
c  Example:
c
c    Input:
c
c      N = 8
c
c      A = ( 6, 7, 3, 1, 6, 8, 2, 9 )
c
c    Output:
c
c      L = 3, R = 6
c
c      A = ( 3, 1, 2, 6, 6, 8, 9, 7 )
c            -------        -------
c
c  Modified:
c
c    25 July 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries of A.
c
c    Input/output, double precision A(N).  On input, the array to be checked.
c    On output, A has been reordered as described above.
c
c    Output, integer L, R, the indices of A that define the three segments.
c    Let KEY = the input value of A(1).  Then
c    I <= L                 A(I) < KEY;
c         L < I < R         A(I) = KEY;
c                 R <= I    KEY < A(I).
c
      implicit none

      integer n

      double precision a(n)
      integer i
      double precision key
      integer l
      integer m
      integer r
      double precision temp

      if ( n .lt. 1 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8VEC_PART_QUICK_A - Fatal error!'
        write ( *, '(a)' ) '  N < 1.'
        stop 1
      else if ( n .eq. 1 ) then
        l = 0
        r = 2
        return
      end if

      key = a(1)
      m = 1
c
c  The elements of unknown size have indices between L+1 and R-1.
c
      l = 1
      r = n + 1

      do i = 2, n

        if ( key .lt. a(l+1) ) then
          r = r - 1
          temp = a(r)
          a(r) = a(l+1)
          a(l+1) = temp
        else if ( a(l+1) .eq. key ) then
          m = m + 1
          temp = a(m)
          a(m) = a(l+1)
          a(l+1) = temp
          l = l + 1
        else if ( a(l+1) .lt. key ) then
          l = l + 1
        end if

      end do
c
c  Now shift small elements to the left, and KEY elements to center.
c
      do i = 1, l - m
        a(i) = a(i+m)
      end do
c
c  Out of bounds here, occasionally
c
      l = l - m

      do i = l + 1, l + m
        a(i) = key
      end do

      return
      end
      subroutine r8vec_permute ( n, p, a )

c*********************************************************************72
c
cc R8VEC_PERMUTE permutes an R8VEC in place.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    This routine permutes an array of real "objects", but the same
c    logic can be used to permute an array of objects of any arithmetic
c    type, or an array of objects of any complexity.  The only temporary
c    storage required is enough to store a single object.  The number
c    of data movements made is N + the number of cycles of order 2 or more,
c    which is never more than N + N/2.
c
c    P(I) = J means that the I-th element of the output array should be
c    the J-th element of the input array.  P must be a legal permutation
c    of the integers from 1 to N, otherwise the algorithm will
c    fail catastrophically.
c
c  Example:
c
c    Input:
c
c      N = 5
c      P = (   2,   4,   5,   1,   3 )
c      A = ( 1.0, 2.0, 3.0, 4.0, 5.0 )
c
c    Output:
c
c      A    = ( 2.0, 4.0, 5.0, 1.0, 3.0 ).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    18 July 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of objects.
c
c    Input, integer P(N), the permutation.
c
c    Input/output, double precision A(N), the array to be permuted.
c
      implicit none

      integer n

      double precision a(n)
      double precision a_temp
      integer ierror
      integer iget
      integer iput
      integer istart
      integer p(n)

      call perm_check1 ( n, p )
c
c  Search for the next element of the permutation that has not been used.
c
      do istart = 1, n

        if ( p(istart) .lt. 0 ) then

          go to 20

        else if ( p(istart) .eq. istart ) then

          p(istart) = - p(istart)
          go to 20

        else

          a_temp = a(istart)
          iget = istart
c
c  Copy the new value into the vacated entry.
c
10        continue

            iput = iget
            iget = p(iget)

            p(iput) = - p(iput)

            if ( iget .lt. 1 .or. n .lt. iget ) then
             write ( *, '(a)' ) ' '
              write ( *, '(a)' ) 'R8VEC_PERMUTE - Fatal error!'
              write ( *, '(a)' ) '  An index is out of range.'
              write ( *, '(a,i8,a,i8)' ) '  P(', iput, ') = ', iget
              stop 1
            end if

            if ( iget .eq. istart ) then
              a(iput) = a_temp
              go to 20
            end if

            a(iput) = a(iget)

          go to 10

        end if

20      continue

      end do
c
c  Restore the signs of the entries.
c
      p(1:n) = - p(1:n)

      return
      end
      subroutine r8vec_permute_cyclic ( n, k, a )

c*********************************************************************72
c
cc R8VEC_PERMUTE_CYCLIC performs a cyclic permutation of an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    For 0 <= K < N, this function cyclically permutes the input vector
c    to have the form
c
c     ( A(K+1), A(K+2), ..., A(N), A(1), ..., A(K) )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    22 August 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of objects.
c
c    Input, integer K, the increment used.
c
c    Input/output, double precision A(N), the array to be permuted.
c
      implicit none

      integer n

      double precision a(n)
      double precision b(n)
      integer i
      integer i4_modp
      integer i4_wrap
      integer ipk
      integer k

      do i = 1, n
        ipk = i4_wrap ( i + k, 1, n )
        b(i) = a(ipk)
      end do

      do i = 1, n
        a(i) = b(i)
      end  do

      return
      end
      subroutine r8vec_permute_uniform ( n, a, seed )

c*********************************************************************72
c
cc R8VEC_PERMUTE_UNIFORM randomly permutes an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    07 July 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of objects.
c
c    Input/output, double precision A(N), the array to be permuted.
c
c    Input/output, integer SEED, a seed for the random
c    number generator.
c
      implicit none

      integer n

      double precision a(n)
      integer p(n)
      integer seed

      call perm_uniform ( n, seed, p )

      call r8vec_permute ( n, p, a )

      return
      end
      subroutine r8vec_polarize ( n, a, p, a_normal, a_parallel )

c*********************************************************************72
c
cc R8VEC_POLARIZE decomposes an R8VEC into normal and parallel components.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    The (nonzero) vector P defines a direction.
c
c    The vector A can be written as the sum
c
c      A = A_normal + A_parallel
c
c    where A_parallel is a linear multiple of P, and A_normal
c    is perpendicular to P.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    19 August 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the array.
c
c    Input, double precision A(N), the vector to be polarized.
c
c    Input, double precision P(N), the polarizing direction.
c
c    Output, double precision A_NORMAL(N), A_PARALLEL(N), the normal
c    and parallel components of A.
c
      implicit none

      integer n

      double precision a(n)
      double precision a_dot_p
      double precision a_normal(n)
      double precision a_parallel(n)
      integer i
      double precision p(n)
      double precision p_norm
      double precision r8vec_dot_product

      p_norm = 0.0D+00
      do i = 1, n
        p_norm = p_norm + p(i) * p(i)
      end do
      p_norm = sqrt ( p_norm )

      if ( p_norm .eq. 0.0D+00 ) then
        do i = 1, n
          a_normal(i) = a(i)
        end do
        do i = 1, n
          a_parallel(i) = 0.0D+00
        end do
        return
      end if

      a_dot_p = r8vec_dot_product ( n, a, p ) / p_norm

      do i = 1, n
        a_parallel(i) = a_dot_p * p(i) / p_norm
      end do

      do i = 1, n
        a_normal(i) = a(i) - a_parallel(i)
      end do

      return
      end
      function r8vec_positive_strict ( n, a )

c*********************************************************************72
c
cc R8VEC_POSITIVE_STRICT: every element of an R8VEC is strictly positive.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    24 June 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the vector.
c
c    Input, double precision A(N).
c
c    Output, logical R8VEC_POSITIVE_STRICT, is TRUE every entry of the
c    vector is strictly positive.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      logical r8vec_positive_strict

      r8vec_positive_strict = .true.

      do i = 1, n
        if ( a(i) .le. 0.0D+00 ) then
          r8vec_positive_strict = .false.
          return
        end if
      end do

      return
      end
      subroutine r8vec_print ( n, a, title )

c*********************************************************************72
c
cc R8VEC_PRINT prints an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    12 January 2007
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of components of the vector.
c
c    Input, double precision A(N), the vector to be printed.
c
c    Input, character * ( * ) TITLE, a title.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      character * ( * ) title

      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) trim ( title )
      write ( *, '(a)' ) ' '
      do i = 1, n
        write ( *, '(2x,i8,a,1x,g16.8)' ) i, ':', a(i)
      end do

      return
      end
      subroutine r8vec_print_16 ( n, a, title )

c*********************************************************************72
c
cc R8VEC_PRINT_16 prints an R8VEC to 16 decimal places.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    29 May 2014
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of components of the vector.
c
c    Input, double precision A(N), the vector to be printed.
c
c    Input, character * ( * ) TITLE, a title.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      character * ( * ) title

      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) trim ( title )
      write ( *, '(a)' ) ' '
      do i = 1, n
        write ( *, '(2x,i8,a,1x,g24.16)' ) i, ':', a(i)
      end do

      return
      end
      subroutine r8vec_print_part ( n, a, max_print, title )

c*********************************************************************72
c
cc R8VEC_PRINT_PART prints "part" of an R8VEC.
c
c  Discussion:
c
c    The user specifies MAX_PRINT, the maximum number of lines to print.
c
c    If N, the size of the vector, is no more than MAX_PRINT, then
c    the entire vector is printed, one entry per line.
c
c    Otherwise, if possible, the first MAX_PRINT-2 entries are printed,
c    followed by a line of periods suggesting an omission,
c    and the last entry.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    22 June 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries of the vector.
c
c    Input, double precision A(N), the vector to be printed.
c
c    Input, integer MAX_PRINT, the maximum number of lines to print.
c
c    Input, character*(*) TITLE, a title.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      integer max_print
      character*(*) title

      if ( max_print .le. 0 ) then
        return
      end if

      if ( n .le. 0 ) then
        return
      end if

      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) title
      write ( *, '(a)' ) ' '

      if ( n .le. max_print ) then

        do i = 1, n
          write ( *, '(2x,i8,a1,1x,g14.6)' ) i, ':', a(i)
        end do

      else if ( 3 .le. max_print ) then

        do i = 1, max_print - 2
          write ( *, '(2x,i8,a1,1x,g14.6)' ) i, ':', a(i)
        end do

        write ( *, '(a)' ) '  ........  ..............'
        i = n

        write ( *, '(2x,i8,a1,1x,g14.6)' ) i, ':', a(i)

      else

        do i = 1, max_print - 1
          write ( *, '(2x,i8,a1,1x,g14.6)' ) i, ':', a(i)
        end do

        i = max_print

        write ( *, '(2x,i8,a1,1x,g14.6,a)' )
     &    i, ':', a(i), '...more entries...'

      end if

      return
      end
      subroutine r8vec_print_some ( n, a, i_lo, i_hi, title )

c*********************************************************************72
c
cc R8VEC_PRINT_SOME prints "some" of an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    13 November 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries of the vector.
c
c    Input, double precision A(N), the vector to be printed.
c
c    Input, integer I_LO, I_HI, the first and last indices
c    to print.  The routine expects 1 <= I_LO <= I_HI <= N.
c
c    Input, character * ( * ) TITLE, a title.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      integer i_hi
      integer i_lo
      character * ( * ) title

      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) trim ( title )

      write ( *, '(a)' ) ' '
      do i = max ( i_lo, 1 ), min ( i_hi, n )
        write ( *, '(2x,i8,a,1x,g14.8)' ) i, ':', a(i)
      end do

      return
      end
      subroutine r8vec_print2 ( n, a )

c*********************************************************************72
c
cc R8VEC_PRINT2 prints out an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    20 August 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries of A.
c
c    Input, double precision A(N), the vector to be printed.
c
      implicit none

      integer n

      double precision a(n)
      double precision amax
      double precision amin
      integer i
      character*11 iform
      logical integ
      integer lmax
      double precision r8_log_10
      double precision r8vec_amax
      double precision r8vec_amin
c
c Check if all entries are integral.
c
      integ = .true.

      do i = 1, n

        if ( a(i) .ne. dble ( int ( a(i) ) ) ) then
          integ = .false.
          go to 10
        end if

      end do

10    continue
c
c  Find the range of the array.
c
      amax = r8vec_amax ( n, a )
      amin = r8vec_amin ( n, a )
c
c  Use the information about the maximum size of an entry to
c  compute an intelligent format for use with integer entries.
c
c  Later, we might also do this for real vectors.
c
      lmax = int ( r8_log_10 ( amax ) )

      if ( integ ) then
        write ( iform, '( ''(2x,i'', i2, '')'' )' ) lmax + 3
      else
        iform = ' '
      end if

      do i = 1, n

        if ( integ ) then
          write ( *, iform ) int ( a(i) )
        else
          write ( *, '(2x,g14.6)' ) a(i)
        end if

      end do

      return
      end
      function r8vec_product ( n, v1 )

c*********************************************************************72
c
cc R8VEC_PRODUCT multiplies the entries of an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8 values.
c
c    In FORTRAN90, the system routine PRODUCT should be called
c    directly.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    22 July 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the dimension of the vectors.
c
c    Input, double precision V1(N), the vector.
c
c    Output, double precision R8VEC_PRODUCT, the product of the entries.
c
      implicit none

      integer n

      integer i
      double precision r8vec_product
      double precision v1(n)
      double precision value

      value = 1.0D+00
      do i = 1, n
        value = value * v1(i)
      end do

      r8vec_product = value

      return
      end
      subroutine r8vec_range ( n, x, xmin, xmax, y, ymin, ymax )

c*********************************************************************72
c
cc R8VEC_RANGE finds the range of Y's within a restricted X range.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    The routine is given a set of pairs of points (X,Y), and a range
c    XMIN to XMAX of valid X values.  Over this range, it seeks
c    YMIN and YMAX, the minimum and maximum values of Y for
c    valid X's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    20 August 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the array.
c
c    Input, double precision X(N), the X array.
c
c    Input, double precision XMIN, XMAX, the range of X values to check.
c
c    Input, double precision Y(N), the Y array.
c
c    Output, double precision YMIN, YMAX, the range of Y values whose
c    X value is within the X range.
c
      implicit none

      integer n

      integer i
      double precision r8_huge
      parameter ( r8_huge = 1.79769313486231571D+308 )
      double precision x(n)
      double precision xmax
      double precision xmin
      double precision y(n)
      double precision ymax
      double precision ymin

      ymin =   r8_huge
      ymax = - r8_huge

      do i = 1, n

        if ( xmin .le. x(i) .and. x(i) .le. xmax ) then

          ymin = min ( ymin, y(i) )
          ymax = max ( ymax, y(i) )

        end if

      end do

      return
      end
      subroutine r8vec_range_2 ( n, a, amin, amax )

c*********************************************************************72
c
cc R8VEC_RANGE_2 updates a range to include a new array.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    Given a range AMIN to AMAX, and an array A, the routine will
c    decrease AMIN if necessary, or increase AMAX if necessary, so that
c    every entry of A is between AMIN and AMAX.
c
c    However, AMIN will not be increased, nor AMAX decreased.
c
c    This routine may be used to compute the maximum and minimum of a
c    collection of arrays one at a time.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    11 September 201
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the array.
c
c    Input, double precision A(N), the array.
c
c    Input/output, double precision AMIN, AMAX.  On input, the
c    current legal range of values for A.  On output, AMIN and AMAX
c    are either unchanged, or else "widened" so that all entries
c    of A are within the range.
c
      implicit none

      integer n

      double precision a(n)
      double precision amax
      double precision amax2
      double precision amin
      double precision amin2
      double precision r8vec_max
      double precision r8vec_min

      amax2 = r8vec_max ( n, a )
      amin2 = r8vec_min ( n, a )

      amax = max ( amax, amax2 )
      amin = min ( amin, amin2 )

      return
      end
      subroutine r8vec_reverse ( n, a )

c*********************************************************************72
c
cc R8VEC_REVERSE reverses the elements of an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Example:
c
c    Input:
c
c      N = 5,
c      A = ( 11.0, 12.0, 13.0, 14.0, 15.0 ).
c
c    Output:
c
c      A = ( 15.0, 14.0, 13.0, 12.0, 11.0 ).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    04 July 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the array.
c
c    Input/output, double precision A(N), the array to be reversed.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      integer i_hi
      double precision t

      i_hi = n / 2

      do i = 1, i_hi
        t        = a(i)
        a(i)     = a(n+1-i)
        a(n+1-i) = t
      end do

      return
      end
      function r8vec_rms ( n, a )

c*********************************************************************72
c
cc R8VEC_RMS returns the RMS norm of an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8 values.
c
c    The vector RMS norm is defined as:
c
c      R8VEC_RMS = sqrt ( sum ( 1 <= I <= N ) A(I)^2 / N ).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    26 October 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in A.
c
c    Input, double precision A(N), the vector.
c
c    Output, double precision R8VEC_RMS, the RMS norm of A.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      double precision r8vec_rms
      double precision value

      value = 0.0D+00

      if ( 0 .lt. n ) then
        do i = 1, n
          value = value + a(i) * a(i)
        end do
        value = sqrt ( value / dble ( n  ) )
      end if

      r8vec_rms = value

      return
      end
      subroutine r8vec_rotate ( n, a, m )

c*********************************************************************72
c
cc R8VEC_ROTATE "rotates" the entries of an R8VEC in place.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    This routine rotates an array of real "objects", but the same
c    logic can be used to permute an array of objects of any arithmetic
c    type, or an array of objects of any complexity.  The only temporary
c    storage required is enough to store a single object.  The number
c    of data movements made is N + the number of cycles of order 2 or more,
c    which is never more than N + N/2.
c
c  Example:
c
c    Input:
c
c      N = 5, M = 2
c      A    = ( 1.0, 2.0, 3.0, 4.0, 5.0 )
c
c    Output:
c
c      A    = ( 4.0, 5.0, 1.0, 2.0, 3.0 ).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    07 July 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of objects.
c
c    Input, integer M, the number of positions to the right that
c    each element should be moved.  Elements that shift pass position
c    N "wrap around" to the beginning of the array.
c
c    Input/output, double precision A(N), the array to be rotated.
c
      implicit none

      integer n

      double precision a(n)
      integer i4_modp
      integer iget
      integer iput
      integer istart
      integer m
      integer mcopy
      integer nset
      double precision temp
c
c  Force M to be positive, between 0 and N-1.
c
      mcopy = i4_modp ( m, n )

      if ( mcopy .eq. 0 ) then
        return
      end if

      istart = 0
      nset = 0

10    continue

        istart = istart + 1

        if ( n .lt. istart ) then
          go to 40
        end if

        temp = a(istart)
        iget = istart
c
c  Copy the new value into the vacated entry.
c
20    continue

        iput = iget

          iget = iget - mcopy
          if ( iget .lt. 1 ) then
            iget = iget + n
          end if

          if ( iget .eq. istart ) then
            go to 30
          end if

          a(iput) = a(iget)
          nset = nset + 1

        go to 20

30      continue

        a(iput) = temp
        nset = nset + 1

        if ( n .le. nset ) then
          go to 40
        end if

      go to 10

40    continue

      return
      end
      function r8vec_scalar_triple_product ( v1, v2, v3 )

c*********************************************************************72
c
cc R8VEC_SCALAR_TRIPLE_PRODUCT computes the scalar triple product.
c
c  Discussion:
c
c    STRIPLE = V1 dot ( V2 x V3 ).
c
c    STRIPLE is the volume of the parallelogram whose sides are
c    formed by V1, V2 and V3.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    27 October 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision V1(3), V2(3), V3(3), the three vectors.
c
c    Output, double precision R8VEC_SCALAR_TRIPLE_PRODUCT, the scalar
c    triple product.
c
      implicit none

      double precision r8vec_scalar_triple_product
      double precision v1(3)
      double precision v2(3)
      double precision v3(3)

      r8vec_scalar_triple_product =
     &    v1(1) * ( v2(2) * v3(3) - v2(3) * v3(2) )
     &  + v1(2) * ( v2(3) * v3(1) - v2(1) * v3(3) )
     &  + v1(3) * ( v2(1) * v3(2) - v2(2) * v3(1) )

      return
      end
      subroutine r8vec_scale ( s, n, x )

c*********************************************************************72
c
cc R8VEC_SCALE multiplies an R8VEC by a scale factor.
c
c  Discussion:
c
c    An R8VEC is a vector of R8 values.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    25 October 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision S, the scale factor.
c
c    Input, integer N, the length of the vector.
c
c    Input/output, double precision X(N), the vector to be scaled.
c
      implicit none

      integer n

      integer i
      double precision s
      double precision x(n)

      do i = 1, n
        x(i) = s * x(i)
      end do

      return
      end
      subroutine r8vec_search_binary_a ( n, a, aval, indx )

c*********************************************************************72
c
cc R8VEC_SEARCH_BINARY_A searches an ascending sorted R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    Binary search is used.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    04 July 2009
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Donald Kreher, Douglas Simpson,
c    Algorithm 1.9,
c    Combinatorial Algorithms,
c    CRC Press, 1998, page 26.
c
c  Parameters:
c
c    Input, integer N, the number of elements in the array.
c
c    Input, double precision A(N), the array to be searched.  The array must
c    be sorted in ascending order.
c
c    Input, double precision AVAL, the value to be searched for.
c
c    Output, integer INDX, the result of the search.
c    -1, AVAL does not occur in the array.
c    I, A(I) = AVAL.
c
      implicit none

      integer n

      double precision a(n)
      double precision aval
      integer high
      integer indx
      integer low
      integer mid

      indx = -1

      low = 1
      high = n

10    continue

      if ( low .le. high ) then

        mid = ( low + high ) / 2

        if ( a(mid) .eq. aval ) then
          indx = mid
          go to 20
        else if ( a(mid) .lt. aval ) then
          low = mid + 1
        else if ( aval .lt. a(mid) ) then
          high = mid - 1
        end if

        go to 10

      end if

20    continue

      return
      end
      subroutine r8vec_shift ( shift, n, x )

c*********************************************************************72
c
cc R8VEC_SHIFT performs a shift on an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8 values.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    12 March 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer SHIFT, the amount by which each entry is to
c    be shifted.
c
c    Input, integer N, the length of the vector.
c
c    Input/output, double precision X(N), the vector to be shifted.
c
      implicit none

      integer n

      integer i
      integer j
      integer shift
      double precision x(n)
      double precision y(n)

      do i = 1, n
        y(i) = x(i)
      end do

      do i = 1, n
        x(i) = 0.0D+00
      end do

      do i = 1, n
        j = i - shift
        if ( 1 .le. j .and. j .le. n ) then
          x(i) = y(j)
        end if
      end do

      return
      end
      subroutine r8vec_shift_circular ( shift, n, x )

c*********************************************************************72
c
cc R8VEC_SHIFT_CIRCULAR performs a circular shift on an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8 values.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    11 March 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer SHIFT, the amount by which each entry is to
c    be shifted.
c
c    Input, integer N, the length of the vector.
c
c    Input/output, double precision X(N), the vector to be shifted.
c
      implicit none

      integer n

      integer i
      integer i4_wrap
      integer j
      integer shift
      double precision x(n)
      double precision y(n)

      do i = 1, n
        y(i) = x(i)
      end do

      do i = 1, n
        j = i4_wrap ( i - shift, 1, n )
        x(i) = y(j)
      end do

      return
      end
      subroutine r8vec_sort_bubble_a ( n, a )

c*********************************************************************72
c
cc R8VEC_SORT_BUBBLE_A ascending sorts an R8VEC using bubble sort.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    Bubble sort is simple to program, but inefficient.  It should not
c    be used for large arrays.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    31 May 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the array.
c
c    Input/output, double precision A(N).
c    On input, an unsorted array.
c    On output, the array has been sorted.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      integer  j
      double precision t

      do i = 1, n - 1
        do j = i + 1, n
          if ( a(j) .lt. a(i) ) then
            t    = a(i)
            a(i) = a(j)
            a(j) = t
          end if
        end do
      end do

      return
      end
      subroutine r8vec_sort_bubble_d ( n, a )

c*********************************************************************72
c
cc R8VEC_SORT_BUBBLE_D descending sorts an R8VEC using bubble sort.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    Bubble sort is simple to program, but inefficient.  It should not
c    be used for large arrays.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    31 May 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the array.
c
c    Input/output, double precision A(N).
c    On input, an unsorted array.
c    On output, the array has been sorted.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      integer j
      double precision t

      do i = 1, n - 1
        do j = i + 1, n
          if ( a(i) .lt. a(j) ) then
            t    = a(i)
            a(i) = a(j)
           a(j) = t
          end if
        end do
      end do

      return
      end
      subroutine r8vec_sort_heap_a ( n, a )

c*********************************************************************72
c
cc R8VEC_SORT_HEAP_A ascending sorts an R8VEC using heap sort.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    01 June 2009
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Albert Nijenhuis, Herbert Wilf,
c    Combinatorial Algorithms for Computers and Calculators,
c    Academic Press, 1978,
c    ISBN: 0-12-519260-6,
c    LC: QA164.N54.
c
c  Parameters:
c
c    Input, integer N, the number of entries in the array.
c
c    Input/output, double precision A(N).
c    On input, the array to be sorted;
c    On output, the array has been sorted.
c
      implicit none

      integer n

      double precision a(n)
      integer n1
      double precision temp

      if ( n .le. 1 ) then
        return
      end if
c
c  1: Put A into descending heap form.
c
      call r8vec_heap_d ( n, a )
c
c  2: Sort A.
c
c  The largest object in the heap is in A(1).
c  Move it to position A(N).
c
      temp = a(1)
      a(1) = a(n)
      a(n) = temp
c
c  Consider the diminished heap of size N1.
c
      do n1 = n - 1, 2, -1
c
c  Restore the heap structure of A(1) through A(N1).
c
        call r8vec_heap_d ( n1, a )
c
c  Take the largest object from A(1) and move it to A(N1).
c
        temp = a(1)
        a(1) = a(n1)
        a(n1) = temp

      end do

      return
      end
      subroutine r8vec_sort_heap_d ( n, a )

c*********************************************************************72
c
cc R8VEC_SORT_HEAP_D descending sorts an R8VEC using heap sort.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    01 June 2009
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Albert Nijenhuis, Herbert Wilf,
c    Combinatorial Algorithms for Computers and Calculators,
c    Academic Press, 1978,
c    ISBN: 0-12-519260-6,
c    LC: QA164.N54.
c
c  Parameters:
c
c    Input, integer N, the number of entries in the array.
c
c    Input/output, double precision A(N).
c    On input, the array to be sorted;
c    On output, the array has been sorted.
c
      implicit none

      integer n

      double precision a(n)
      integer n1

      if ( n .le. 1 ) then
        return
      end if
c
c  1: Put A into ascending heap form.
c
      call r8vec_heap_a ( n, a )
c
c  2: Sort A.
c
c  The smallest object in the heap is in A(1).
c  Move it to position A(N).
c
      call r8_swap ( a(1), a(n) )
c
c  Consider the diminished heap of size N1.
c
      do n1 = n - 1, 2, -1
c
c  Restore the heap structure of A(1) through A(N1).
c
        call r8vec_heap_a ( n1, a )
c
c  Take the smallest object from A(1) and move it to A(N1).
c
        call r8_swap ( a(1), a(n1) )

      end do

      return
      end
      subroutine r8vec_sort_heap_index_a ( n, a, indx )

c*********************************************************************72
c
cc R8VEC_SORT_HEAP_INDEX_A does an indexed heap ascending sort of an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    The sorting is not actually carried out.  Rather an index array is
c    created which defines the sorting.  This array may be used to sort
c    or index the array, or to sort or index related arrays keyed on the
c    original array.
c
c    Once the index array is computed, the sorting can be carried out
c    "implicitly:
c
c      A(INDX(I:N)) is sorted,
c
c    or explicitly, by the call
c
c      call r8vec_permute ( n, indx, a )
c
c    after which A(1:N) is sorted.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    01 June 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the array.
c
c    Input, double precision A(N), an array to be index-sorted.
c
c    Output, integer INDX(N), the sort index.  The
c    I-th element of the sorted array is A(INDX(I)).
c
      implicit none

      integer n

      double precision a(n)
      double precision aval
      integer i
      integer indx(n)
      integer indxt
      integer ir
      integer j
      integer l

      if ( n .lt. 1 ) then
        return
      end if

      do i = 1, n
        indx(i) = i
      end do

      if ( n .eq. 1 ) then
        return
      end if

      l = n / 2 + 1
      ir = n

10    continue

        if ( 1 .lt. l ) then

          l = l - 1
          indxt = indx(l)
          aval = a(indxt)

        else

          indxt = indx(ir)
          aval = a(indxt)
          indx(ir) = indx(1)
          ir = ir - 1

          if ( ir .eq. 1 ) then
            indx(1) = indxt
            go to 30
          end if

        end if

        i = l
        j = l + l

20      continue

        if ( j .le. ir ) then

          if ( j .lt. ir ) then
            if ( a(indx(j)) .lt. a(indx(j+1)) ) then
              j = j + 1
            end if
          end if

          if ( aval .lt. a(indx(j)) ) then
            indx(i) = indx(j)
            i = j
            j = j + j
          else
            j = ir + 1
          end if

          go to 20

        end if

        indx(i) = indxt

      go to 10

30    continue

      return
      end
      subroutine r8vec_sort_heap_index_d ( n, a, indx )

c*********************************************************************72
c
cc R8VEC_SORT_HEAP_INDEX_D does an indexed heap descending sort of an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    The sorting is not actually carried out.  Rather an index array is
c    created which defines the sorting.  This array may be used to sort
c    or index the array, or to sort or index related arrays keyed on the
c    original array.
c
c    Once the index array is computed, the sorting can be carried out
c    "implicitly:
c
c      A(INDX(1:N)) is sorted,
c
c    or explicitly, by the call
c
c      call r8vec_permute ( n, indx, a )
c
c    after which A(1:N) is sorted.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    01 June 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the array.
c
c    Input, double precision A(N), an array to be index-sorted.
c
c    Output, integer INDX(N), the sort index.  The
c    I-th element of the sorted array is A(INDX(I)).
c
      implicit none

      integer n

      double precision a(n)
      double precision aval
      integer i
      integer indx(n)
      integer indxt
      integer ir
      integer j
      integer l

      if ( n .lt. 1 ) then
        return
      end if

      do i = 1, n
        indx(i) = i
      end do

      if ( n .eq. 1 ) then
        return
      end if

      l = n / 2 + 1
      ir = n

10    continue

        if ( 1 .lt. l ) then

          l = l - 1
          indxt = indx(l)
          aval = a(indxt)

        else

          indxt = indx(ir)
          aval = a(indxt)
          indx(ir) = indx(1)
          ir = ir - 1

          if ( ir .eq. 1 ) then
            indx(1) = indxt
            go to 30
          end if

        end if

        i = l
        j = l + l

20      continue

        if ( j .le. ir ) then

          if ( j .lt. ir ) then
            if ( a(indx(j+1)) .lt. a(indx(j)) ) then
              j = j + 1
            end if
          end if

          if ( a(indx(j)) .lt. aval ) then
            indx(i) = indx(j)
            i = j
            j = j + j
          else
            j = ir + 1
          end if

          go to 20

        end if

        indx(i) = indxt

      go to 10

30    continue

      return
      end
      subroutine r8vec_sort_heap_mask_a ( n, a, mask_num, mask, indx )

c*********************************************************************72
c
cc R8VEC_SORT_HEAP_MASK_A: indexed heap ascending sort of a masked R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    An array A is given.  An array MASK of indices into A is given.
c    The routine produces a vector INDX, which is a permutation of the
c    entries of MASK, so that:
c
c      A(MASK(INDX(I)) <= A(MASK(INDX(J))
c
c    whenever
c
c      I <= J
c
c    In other words, only the elements of A that are indexed by MASK
c    are to be considered, and the only thing that happens is that
c    a rearrangment of the indices in MASK is returned that orders the
c    masked elements.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    22 August 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the array.
c
c    Input, double precision A(N), an array to be index-sorted.
c
c    Input, integer MASK_NUM, the number of mask elements.
c
c    Input, integer MASK(MASK_NUM), the mask array.  This is
c    simply a list of indices of A.  The entries of MASK should
c    be unique, and each one should be between 1 and N.
c
c    Output, integer INDX(MASK_NUM), the sort index.  There are
c    MASK_NUM elements of A selected by MASK.  If we want to list those
c    elements in order, then the I-th element is A(MASK(INDX(I))).
c
      implicit none

      integer mask_num
      integer n

      double precision a(n)
      double precision aval
      integer i
      integer indx(mask_num)
      integer indxt
      integer ir
      integer j
      integer l
      integer mask(mask_num)

      if ( n .lt. 1 ) then
        return
      end if

      if ( mask_num .lt. 1 ) then
        return
      end if

      if ( mask_num .eq. 1 ) then
        indx(1) = 1
        return
      end if

      call i4vec_indicator1 ( mask_num, indx )

      l = mask_num / 2 + 1
      ir = mask_num

10    continue

        if ( 1 .lt. l ) then

          l = l - 1
          indxt = indx(l)
          aval = a(mask(indxt))

        else

          indxt = indx(ir)
          aval = a(mask(indxt))
          indx(ir) = indx(1)
          ir = ir - 1

          if ( ir .eq. 1 ) then
            indx(1) = indxt
            go to 30
          end if

        end if

        i = l
        j = l + l

20      continue

        if ( j .le. ir ) then

          if ( j .lt. ir ) then
            if ( a(mask(indx(j))) .lt. a(mask(indx(j+1))) ) then
              j = j + 1
            end if
          end if

          if ( aval .lt. a(mask(indx(j))) ) then
            indx(i) = indx(j)
            i = j
            j = j + j
          else
            j = ir + 1
          end if

          go to 20

        end if

        indx(i) = indxt

      go to 10

30    continue

      return
      end
      subroutine r8vec_sort_insert_a ( n, a )

c*********************************************************************72
c
cc R8VEC_SORT_INSERT_A ascending sorts an R8VEC using an insertion sort.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    21 August 2010
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Donald Kreher, Douglas Simpson,
c    Algorithm 1.1,
c    Combinatorial Algorithms,
c    CRC Press, 1998, page 11.
c
c  Parameters:
c
c    Input, integer N, the number of items in the vector.
c    N must be positive.
c
c    Input/output, double precision A(N).
c    On input, the array to be sorted;
c    On output, the array has been sorted.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      integer j
      double precision x

      do i = 2, n

        x = a(i)

        j = i - 1

10      continue

        if ( 1 .le. j ) then

          if ( a(j) .le. x ) then
            go to 20
          end if

          a(j+1) = a(j)
          j = j - 1

          go to 10

        end if

20      continue

        a(j+1) = x

      end do

      return
      end
      subroutine r8vec_sort_insert_index_a ( n, a, indx )

c*********************************************************************72
c
cc R8VEC_SORT_INSERT_INDEX_A ascending index sorts an R8VEC using insertion.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    22 August 2010
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Donald Kreher, Douglas Simpson,
c    Algorithm 1.1,
c    Combinatorial Algorithms,
c    CRC Press, 1998, page 11.
c
c  Parameters:
c
c    Input, integer N, the number of items in the vector.
c    N must be positive.
c
c    Input, double precision A(N), the array to be sorted.
c
c    Output, integer INDX(N), the sorted indices.  The array
c    is sorted when listed from A(INDX(1)) through A(INDX(N)).
c
      implicit none

      integer n

      double precision a(n)
      integer i
      integer indx(n)
      integer j
      double precision x

      if ( n .lt. 1 ) then
        return
      end if

      do i = 1, n
        indx(i) = i
      end do

      do i = 2, n

        x = a(i)

        j = i - 1

10      continue

        if ( 1 .le. j ) then

          if ( a(indx(j)) .le. x ) then
            go to 20
          end if

          indx(j+1) = indx(j)
          j = j - 1

          go to 10

        end if

20      continue

        indx(j+1) = i

      end do

      return
      end
      subroutine r8vec_sort_insert_index_d ( n, a, indx )

c*********************************************************************72
c
cc R8VEC_SORT_INSERT_INDEX_D descending index sorts an R8VEC using insertion.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    20 August 2010
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Donald Kreher, Douglas Simpson,
c    Algorithm 1.1,
c    Combinatorial Algorithms,
c    CRC Press, 1998, page 11.
c
c  Parameters:
c
c    Input, integer N, the number of items in the vector.
c    N must be positive.
c
c    Input, double precision A(N), the array to be sorted.
c
c    Output, integer INDX(N), the sorted indices.  The array
c    is sorted when listed from A(INDX(1)) through A(INDX(N)).
c
      implicit none

      integer n

      double precision a(n)
      integer i
      integer indx(n)
      integer j
      double precision x

      if ( n .lt. 1 ) then
        return
      end if

      do i = 1, n
        indx(i) = i
      end do

      do i = 2, n

        x = a(i)

        j = i - 1

10      continue

        if ( 1 .le. j ) then

          if ( x .le. a(indx(j)) ) then
            go to 20
          end if

          indx(j+1) = indx(j)
          j = j - 1

          go to 10

        end if

20      continue

        indx(j+1) = i

      end do

      return
      end
      subroutine r8vec_sort_quick_a ( n, a )

c*********************************************************************72
c
cc R8VEC_SORT_QUICK_A ascending sorts an R8VEC using quick sort.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Example:
c
c    Input:
c
c      N = 7
c      A = ( 6, 7, 3, 2, 9, 1, 8 )
c
c    Output:
c
c      A = ( 1, 2, 3, 6, 7, 8, 9 )
c
c  Modified:
c
c    25 July 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the array.
c
c    Input/output, double precision A(N).
c    On input, the array to be sorted.
c    On output, the array has been sorted.
c
      implicit none

      integer level_max
      parameter ( level_max = 25 )
      integer n

      double precision a(n)
      integer base
      integer l_segment
      integer level
      integer n_segment
      integer rsave(level_max)
      integer r_segment

      if ( n .lt. 1 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8VEC_SORT_QUICK_A - Fatal error!'
        write ( *, '(a)' ) '  N < 1.'
        stop 1
      else if ( n .eq. 1 ) then
        return
      end if

      level = 1
      rsave(level) = n + 1
      base = 1
      n_segment = n

10    continue
c
c  Partition the segment.
c
        call r8vec_part_quick_a ( n_segment, a(base), l_segment,
     &    r_segment )
c
c  If the left segment has more than one element, we need to partition it.
c
        if ( 1 .lt. l_segment ) then

          if ( level_max .lt. level ) then
            write ( *, '(a)' ) ' '
            write ( *, '(a)' ) 'R8VEC_SORT_QUICK_A - Fatal error!'
            write ( *, '(a,i6)' )
     &        '  Exceeding recursion maximum of ', level_max
            stop 1
          end if

          level = level + 1
          n_segment = l_segment
          rsave(level) = r_segment + base - 1
c
c  The left segment and the middle segment are sorted.
c  Must the right segment be partitioned?
c
        else if ( r_segment .lt. n_segment ) then

          n_segment = n_segment + 1 - r_segment
          base = base + r_segment - 1
c
c  Otherwise, we back up a level if there is an earlier one.
c
        else

20        continue

            if ( level .le. 1 ) then
              go to 40
            end if

            base = rsave(level)
            n_segment = rsave(level-1) - rsave(level)
            level = level - 1

            if ( 0 .lt. n_segment ) then
              go to 30
            end if

          go to 20

30        continue

        end if

      go to 10

40    continue

      return
      end
      subroutine r8vec_sorted_merge_a ( na, a, nb, b, nc, c )

c*********************************************************************72
c
cc R8VEC_SORTED_MERGE_A merges two ascending sorted R8VEC's.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    The elements of A and B should be sorted in ascending order.
c
c    The elements in the output array C will also be in ascending order,
c    and unique.
c
c    The output vector C may share storage with A or B.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    25 September 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer NA, the dimension of A.
c
c    Input, double precision A(NA), the first sorted array.
c
c    Input, integer NB, the dimension of B.
c
c    Input, double precision B(NB), the second sorted array.
c
c    Output, integer NC, the number of elements in the output
c    array.  Note that C should usually be dimensioned at least NA+NB in the
c    calling routine.
c
c    Output, double precision C(NC), the merged unique sorted array.
c
      implicit none

      integer na
      integer nb

      double precision a(na)
      double precision b(nb)
      double precision c(na+nb)
      double precision d(na+nb)
      integer i
      integer j
      integer ja
      integer jb
      integer na2
      integer nb2
      integer nc
      integer order

      na2 = na
      nb2 = nb

      ja = 0
      jb = 0
      nc = 0

      call r8vec_order_type ( na2, a, order )

      if ( order .lt. 0 .or. 2 .lt. order ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8VEC_SORTED_MERGE_A - Fatal error!'
        write ( *, '(a)' )
     &    '  The input array A is not ascending sorted!'
        stop 1
      end if

      call r8vec_order_type ( nb2, b, order )

      if ( order .lt. 0 .or. 2 .lt. order ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8VEC_SORTED_MERGE_A - Fatal error!'
        write ( *, '(a)' )
     &    '  The input array B is not ascending sorted!'
        stop 1
      end if

10    continue
c
c  If we've used up all the entries of A, stick the rest of B on the end.
c
        if ( na2 .le. ja ) then

          do j = 1, nb2 - jb
            jb = jb + 1
            if ( nc .eq. 0 ) then
              nc = nc + 1
              d(nc) = b(jb)
            else if ( d(nc) .lt. b(jb) ) then
              nc = nc + 1
              d(nc) = b(jb)
            end if
          end do

          do i = 1, nc
            c(i) = d(i)
          end do

          go to 20
c
c  If we've used up all the entries of B, stick the rest of A on the end.
c
        else if ( nb2 .le. jb ) then

          do j = 1, na2 - ja
            ja = ja + 1
            if ( nc .eq. 0 ) then
              nc = nc + 1
              d(nc) = a(ja)
            else if ( d(nc) .lt. a(ja) ) then
              nc = nc + 1
              d(nc) = a(ja)
            end if
          end do

          do i = 1, nc
            c(i) = d(i)
          end do

          go to 20
c
c  Otherwise, if the next entry of A is smaller, that's our candidate.
c
        else if ( a(ja+1) .le. b(jb+1) ) then

          ja = ja + 1
          if ( nc .eq. 0 ) then
            nc = nc + 1
            d(nc) = a(ja)
          else if ( d(nc) .lt. a(ja) ) then
            nc = nc + 1
            d(nc) = a(ja)
          end if
c
c  ...or if the next entry of B is the smaller, consider that.
c
        else

          jb = jb + 1
          if ( nc .eq. 0 ) then
            nc = nc + 1
            d(nc) = b(jb)
          else if ( d(nc) .lt. b(jb) ) then
            nc = nc + 1
            d(nc) = b(jb)
          end if
        end if

      go to 10

20    continue

      return
      end
      function r8vec_sorted_nearest ( n, a, value )

c*********************************************************************72
c
cc R8VEC_SORTED_NEAREST returns the nearest element in a sorted R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    25 September 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of elements of A.
c
c    Input, double precision A(N), a sorted vector.
c
c    Input, double precision VALUE, the value whose nearest vector
c    entry is sought.
c
c    Output, integer R8VEC_SORTED_NEAREST, the index of the nearest
c    entry in the vector.
c
      implicit none

      integer n

      double precision a(n)
      integer r8vec_sorted_nearest
      integer hi
      integer lo
      integer mid
      double precision value

      if ( n .lt. 1 ) then
        r8vec_sorted_nearest = -1
        return
      end if

      if ( n .eq. 1 ) then
        r8vec_sorted_nearest = 1
        return
      end if

      if ( a(1) .lt. a(n) ) then

        if ( value .lt. a(1) ) then
          r8vec_sorted_nearest = 1
          return
        else if ( a(n) .lt. value ) then
          r8vec_sorted_nearest = n
          return
        end if
c
c  Seek an interval containing the value.
c
        lo = 1
        hi = n

10      continue

        if ( lo .lt. hi - 1 ) then

          mid = ( lo + hi ) / 2

          if ( value .eq. a(mid) ) then
            r8vec_sorted_nearest = mid
            return
          else if ( value .lt. a(mid) ) then
            hi = mid
          else
            lo = mid
          end if

          go to 10

        end if
c
c  Take the nearest.
c
        if ( abs ( value - a(lo) ) .lt. abs ( value - a(hi) ) ) then
          r8vec_sorted_nearest = lo
        else
          r8vec_sorted_nearest = hi
        end if

        return
c
c  A descending sorted vector A.
c
      else

        if ( value .lt. a(n) ) then
          r8vec_sorted_nearest = n
          return
        else if ( a(1) .lt. value ) then
          r8vec_sorted_nearest = 1
          return
        end if
c
c  Seek an interval containing the value.
c
        lo = n
        hi = 1

20      continue

        if ( lo .lt. hi - 1 ) then

          mid = ( lo + hi ) / 2

          if ( value .eq. a(mid) ) then
            r8vec_sorted_nearest = mid
            return
          else if ( value .lt. a(mid) ) then
            hi = mid
          else
            lo = mid
          end if

          go to 20

        end if
c
c  Take the nearest.
c
        if ( abs ( value - a(lo) ) .lt. abs ( value - a(hi) ) ) then
          r8vec_sorted_nearest = lo
        else
          r8vec_sorted_nearest = hi
        end if

        return

      end if

      return
      end
      subroutine r8vec_sorted_range ( n, r, r_lo, r_hi, i_lo, i_hi )

c*********************************************************************72
c
cc R8VEC_SORTED_RANGE searches a sorted vector for elements in a range.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    25 September 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of items in the vector.
c
c    Input, double precision R(N), the sorted vector.
c
c    Input, double precision R_LO, R_HI, the limits of the range.
c
c    Output, integer I_LO, I_HI, the range of indices
c    so that I_LO .le. I .le. I_HI => R_LO .le. R(I) .le. R_HI.  If no
c    values in R lie in the range, then I_HI .lt. I_LO will be returned.
c
      implicit none

      integer n

      integer i_hi
      integer i_lo
      integer i1
      integer i2
      integer j1
      integer j2
      double precision r(n)
      double precision r_hi
      double precision r_lo
c
c  Cases we can handle immediately.
c
      if ( r(n) .lt. r_lo ) then
        i_lo = 0
        i_hi = - 1
        return
      end if

      if ( r_hi .lt. r(1) ) then
        i_lo = 0
        i_hi = - 1
        return
      end if
c
c  Are there are least two intervals?
c
      if ( n .eq. 1 ) then
        if ( r_lo .le. r(1) .and. r(1) .le. r_hi ) then
          i_lo = 1
          i_hi = 1
        else
          i_lo = 0
          i_hi = -1
        end if
        return
      end if
c
c  Bracket R_LO.
c
      if ( r_lo .le. r(1) ) then

        i_lo = 1

      else
c
c  R_LO is in one of the intervals spanned by R(J1) to R(J2).
c  Examine the intermediate interval [R(I1), R(I1+1)].
c  Does R_LO lie here, or below or above?
c
        j1 = 1
        j2 = n
        i1 = ( j1 + j2 - 1 ) / 2
        i2 = i1 + 1

10      continue

          if ( r_lo .lt. r(i1) ) then
            j2 = i1
            i1 = ( j1 + j2 - 1 ) / 2
            i2 = i1 + 1
          else if ( r(i2) .lt. r_lo ) then
            j1 = i2
            i1 = ( j1 + j2 - 1 ) / 2
            i2 = i1 + 1
          else
            i_lo = i1
            go to 20
          end if

        go to 10

20      continue

      end if
c
c  Bracket R_HI
c
      if ( r(n) .le. r_hi ) then

        i_hi = n

      else

        j1 = i_lo
        j2 = n
        i1 = ( j1 + j2 - 1 ) / 2
        i2 = i1 + 1

30      continue

          if ( r_hi .lt. r(i1) ) then
            j2 = i1
            i1 = ( j1 + j2 - 1 ) / 2
            i2 = i1 + 1
          else if ( r(i2) .lt. r_hi ) then
            j1 = i2
            i1 = ( j1 + j2 - 1 ) / 2
            i2 = i1 + 1
          else
            i_hi = i2
            go to 40
          end if

        go to 30

40      continue

      end if
c
c  We expect to have computed the largest I_LO and smallest I_HI such that
c    R(I_LO) .le. R_LO .le. R_HI .le. R(I_HI)
c  but what we want is actually
c    R_LO .le. R(I_LO) .le. R(I_HI) .le. R_HI
c  which we can usually get simply by incrementing I_LO and decrementing I_HI.
c
      if ( r(i_lo) .lt. r_lo ) then
        i_lo = i_lo + 1
        if ( n .lt. i_lo ) then
          i_hi = i_lo - 1
        end if
      end if

      if ( r_hi .lt. r(i_hi) ) then
        i_hi = i_hi - 1
        if ( i_hi .lt. 1 ) then
          i_lo = i_hi + 1
        end if
      end if

      return
      end
      subroutine r8vec_sorted_split ( n, a, split, i_lt, i_gt )

c*********************************************************************72
c
cc R8VEC_SORTED_SPLIT "splits" a sorted R8VEC, given a splitting value.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    Given a splitting value SPLIT, the routine seeks indices
c    I_LT and I_GT so that
c
c      A(I_LT) .lt. SPLIT .lt. A(I_GT),
c
c    and if there are intermediate index values between I_LT and
c    I_GT, then those entries of A are exactly equal to SPLIT.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    25 September 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters
c
c    Input, integer N, the number of entries in A.
c
c    Input, double precision A(N), a sorted array.
c
c    Input, double precision SPLIT, a value to which the entries in A are
c    to be compared.
c
c    Output, integer I_LT:
c    0 if no entries are less than SPLIT;
c    N if all entries are less than SPLIT;
c    otherwise, the index of the last entry in A less than SPLIT.
c
c    Output, integer I_GT:
c    1 if all entries are greater than SPLIT;
c    N+1 if no entries are greater than SPLIT;
c    otherwise the index of the first entry in A greater than SPLIT.
c
      implicit none

      integer n

      double precision a(n)
      integer hi
      integer i
      integer i_gt
      integer i_lt
      integer lo
      integer mid
      double precision split

      if ( n .lt. 1 ) then
        i_lt = -1
        i_gt = -1
        return
      end if

      if ( split .lt. a(1) ) then
        i_lt = 0
        i_gt = 1
        return
      end if

      if ( a(n) .lt. split ) then
        i_lt = n
        i_gt = n + 1
        return
      end if

      lo = 1
      hi = n

10    continue

        if ( lo + 1 .eq. hi ) then
          i_lt = lo
          go to 20
        end if

        mid = ( lo + hi ) / 2

        if ( split .le. a(mid) ) then
          hi = mid
        else
          lo = mid
        end if

      go to 10

20    continue

      do i = i_lt + 1, n
        if ( split .lt. a(i) ) then
          i_gt = i
          return
        end if
      end do

      i_gt = n + 1

      return
      end
      subroutine r8vec_sorted_undex ( x_num, x_val, x_unique_num, tol,
     &  undx, xdnu )

c*********************************************************************72
c
cc R8VEC_SORTED_UNDEX returns unique sorted indexes for a sorted R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    The goal of this routine is to determine a vector UNDX,
c    which points, to the unique elements of X, in sorted order,
c    and a vector XDNU, which identifies, for each entry of X, the index of
c    the unique sorted element of X.
c
c    This is all done with index vectors, so that the elements of
c    X are never moved.
c
c    Assuming X is already sorted, we examine the entries of X in order,
c    noting the unique entries, creating the entries of XDNU and
c    UNDX as we go.
c
c    Once this process has been completed, the vector X could be
c    replaced by a compressed vector XU, containing the unique entries
c    of X in sorted order, using the formula
c
c      XU(I) = X(UNDX(I)).
c
c    We could then, if we wished, reconstruct the entire vector X, or
c    any element of it, by index, as follows:
c
c      X(I) = XU(XDNU(I)).
c
c    We could then replace X by the combination of XU and XDNU.
c
c    Later, when we need the I-th entry of X, we can locate it as
c    the XDNU(I)-th entry of XU.
c
c    Here is an example of a vector X, the sort and inverse sort
c    index vectors, and the unique sort and inverse unique sort vectors
c    and the compressed unique sorted vector.
c
c    Here is an example of a vector X, the unique sort and
c    inverse unique sort vectors and the compressed unique sorted vector.
c
c      I      X      XU  Undx  Xdnu
c    ----+------+------+-----+-----+
c      1 | 11.0 |  11.0    1     1
c      2 | 11.0 |  22.0    5     1
c      3 | 11.0 |  33.0    8     1
c      4 | 11.0 |  55.0    9     1
c      5 | 22.0 |                2
c      6 | 22.0 |                2
c      7 | 22.0 |                2
c      8 | 33.0 |                3
c      9 | 55.0 |
c
c    INDX(2) = 3 means that sorted item(2) is X(3).
c    XDNI(2) = 5 means that X(2) is sorted item(5).
c
c    UNDX(3) = 4 means that unique sorted item(3) is at X(4).
c    XDNU(8) = 2 means that X(8) is at unique sorted item(2).
c
c    XU(XDNU(I))) = X(I).
c    XU(I)        = X(UNDX(I)).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    07 July 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer X_NUM, the number of data values.
c
c    Input, double precision X_VAL(X_NUM), the data values.
c
c    Input, integer X_UNIQUE_NUM, the number of unique values
c    in X_VAL.  This value is only required for languages in which the size of
c    UNDX must be known in advance.
c
c    Input, double precision TOL, a tolerance for equality.
c
c    Output, integer UNDX(X_UNIQUE_NUM), the UNDX vector.
c
c    Output, integer XDNU(X_NUM), the XDNU vector.
c
      implicit none

      integer x_num
      integer x_unique_num

      integer i
      integer j
      double precision tol
      integer undx(x_unique_num)
      double precision x_val(x_num)
      integer xdnu(x_num)
c
c  Walk through the sorted array X.
c
      i = 1

      j = 1
      undx(j) = i

      xdnu(i) = j

      do i = 2, x_num

        if ( tol .lt. abs ( x_val(i) - x_val(undx(j)) ) ) then
          j = j + 1
          undx(j) = i
        end if

        xdnu(i) = j

      end do

      return
      end
      subroutine r8vec_sorted_unique ( n, a, tol, unique_num )

c*********************************************************************72
c
cc R8VEC_SORTED_UNIQUE keeps the unique elements in a sorted R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    04 July 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of elements of A.
c
c    Input/output, double precision A(N).
c    On input, the sorted array of N elements;
c    On output, the sorted unique array of UNIQUE_NUM elements.
c
c    Input, double precision TOL, a nonnegative tolerance for equality.
c    Set it to 0.0 for the strictest test.
c
c    Output, integer UNIQUE_NUM, the number of unique elements
c    of A.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      integer unique_num
      double precision tol

      if ( n .le. 0 ) then
        unique_num = 0
        return
      end if

      unique_num = 1

      do i = 2, n

        if ( tol .lt. abs ( a(i) - a(unique_num) ) ) then
          unique_num = unique_num + 1
          a(unique_num) = a(i)
        end if

      end do

      return
      end
      subroutine r8vec_sorted_unique_count ( n, a, tol, unique_num )

c*********************************************************************72
c
cc R8VEC_SORTED_UNIQUE_COUNT counts the unique elements in a sorted R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    Because the array is sorted, this algorithm is O(N).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    04 July 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of elements of A.
c
c    Input, double precision A(N), the sorted array to examine.
c
c    Input, double precision TOL, a nonnegative tolerance for equality.
c    Set it to 0.0 for the strictest test.
c
c    Output, integer UNIQUE_NUM, the number of unique elements
c    of A.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      integer unique_num
      double precision tol

      if ( n .lt. 1 ) then
        unique_num = 0
        return
      end if

      unique_num = 1

      do i = 2, n

        if ( tol .lt. abs ( a(i-1) - a(i) ) ) then
          unique_num = unique_num + 1
        end if

      end do

      return
      end
      subroutine r8vec_sorted_unique_hist ( n, a, tol, maxuniq,
     &  unique_num, auniq, acount )

c*********************************************************************72
c
cc R8VEC_SORTED_UNIQUE_HIST histograms the unique elements of a sorted R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    21 August 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of elements of A.
c
c    Input, double precision A(N), the array to examine.  The elements of A
c    should have been sorted.
c
c    Input, double precision TOL, a nonnegative tolerance for equality.
c    Set it to 0.0 for the strictest test.
c
c    Input, integer MAXUNIQ, the maximum number of unique elements
c    that can be handled.  If there are more than MAXUNIQ unique
c    elements in A, the excess will be ignored.
c
c    Output, integer UNIQUE_NUM, the number of unique elements
c    of A.
c
c    Output, double precision AUNIQ(UNIQUE_NUM), the unique elements of A.
c
c    Output, integer ACOUNT(UNIQUE_NUM), the number of times
c    each element of AUNIQ occurs in A.
c
      implicit none

      integer maxuniq
      integer n

      double precision a(n)
      integer acount(maxuniq)
      double precision auniq(maxuniq)
      integer i
      integer unique_num
      double precision tol
c
c  Start taking statistics.
c
      unique_num = 0

      do i = 1, n

        if ( i .eq. 1 ) then

          unique_num = 1
          auniq(unique_num) = a(1)
          acount(unique_num) = 1

        else if ( abs ( a(i) - auniq(unique_num) ) .le. tol ) then

          acount(unique_num) = acount(unique_num) + 1

        else if ( unique_num .lt. maxuniq ) then

          unique_num = unique_num + 1
          auniq(unique_num) = a(i)
          acount(unique_num) = 1

        end if

      end do

      return
      end
      subroutine r8vec_split ( n, a, split, isplit )

c*********************************************************************72
c
cc R8VEC_SPLIT "splits" an unsorted R8VEC based on a splitting value.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    If the vector is already sorted, it is simpler to do a binary search
c    on the data than to call this routine.
c
c    The vector is not assumed to be sorted before input, and is not
c    sorted during processing.  If sorting is not needed, then it is
c    more efficient to use this routine.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    01 June 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of elements of A.
c
c    Input/output, double precision A(N), the array to split.  On output,
c    all the entries of A that are less than or equal to SPLIT
c    are in A(1:ISPLIT).
c
c    Input, double precision SPLIT, the value used to split the vector.
c    It is not necessary that any value of A actually equal SPLIT.
c
c    Output, integer ISPLIT, indicates the position of the last
c    entry of the split vector that is less than or equal to SPLIT.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      integer i1
      integer i2
      integer i3
      integer isplit
      integer j1
      integer j2
      integer j3
      double precision split
c
c  Partition the vector into A1, A2, A3, where
c    A1 = A(I1:J1) holds values <= SPLIT,
c    A2 = A(I2:J2) holds untested values,
c    A3 = A(I3:J3) holds values > SPLIT.
c
      i1 = 1
      j1 = 0

      i2 = 1
      j2 = n

      i3 = n + 1
      j3 = n
c
c  Pick the next item from A2, and move it into A1 or A3.
c  Adjust indices appropriately.
c
      do i = 1, n

        if ( a(i2) .le. split ) then
          i2 = i2 + 1
          j1 = j1 + 1
        else
          call r8_swap ( a(i2), a(i3-1) )
          i3 = i3 - 1
          j2 = j2 - 1
        end if

      end do

      isplit = j1

      return
      end
      subroutine r8vec_std ( n, a, std )

c*********************************************************************72
c
cc R8VEC_STD returns the standard deviation of an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    The standard deviation of a vector X of length N is defined as
c
c      mean ( X(1:n) ) = sum ( X(1:n) ) / n
c
c      std ( X(1:n) ) = sqrt ( sum ( ( X(1:n) - mean )^2 ) / ( n - 1 ) )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    01 June 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the vector.
c    N should be at least 2.
c
c    Input, double precision A(N), the vector.
c
c    Output, double precision STD, the standard deviation of the vector.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      double precision mean
      double precision std

      if ( n .lt. 2 ) then

        std = 0.0D+00

      else

        mean = 0.0D+00
        do i = 1, n
          mean = mean + a(i)
        end do
        mean = mean / dble ( n )

        std = 0.0D+00
        do i = 1, n
          std = std + ( a(i) - mean ) ** 2
        end do
        std = sqrt ( std / dble ( n - 1 ) )

      end if

      return
      end
      subroutine r8vec_step ( x0, n, x, fx )

c*********************************************************************72
c
cc R8VEC_STEP evaluates a unit step function.
c
c  Discussion:
c
c    F(X) = 0 if X < X0
c           1 if     X0 <= X
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    30 May 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision X0, the location of the jump.
c
c    Input, integer N, the number of argument values.
c
c    Output, double precision X(N), the arguments.
c
c    Output, double precision FX(N), the function values.
c
      implicit none

      integer n

      double precision fx(n)
      integer i
      double precision x(n)
      double precision x0

      do i = 1, n
        if ( x(i) < x0 ) then
          fx(i) = 0.0D+00
        else
          fx(i) = 1.0D+00
        end if
      end do

      return
      end
      subroutine r8vec_stutter ( n, a, m, am )

c*********************************************************************72
c
cc R8VEC_STUTTER makes a "stuttering" copy of an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    Applying a stuttering factor M of 3, the vector A = ( 1, 5, 8 ) becomes
c    AM = ( 1, 1, 1, 5, 5, 5, 8, 8, 8 ).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    23 March 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the size of the input vector.
c
c    Input, double precision A(N), the vector.
c
c    Input, integer M, the "stuttering factor".
c
c    Output, double precision AM(M*N), the stuttering vector.
c
      implicit none

      integer m
      integer n

      double precision a(n)
      double precision am(m*n)
      integer i
      integer j
      integer jhi
      integer jlo

      do i = 1, n
        jlo = m * ( i - 1 ) + 1
        jhi = m *   i
        do j = jlo, jhi
          am(j) = a(i)
        end do
      end do

      return
      end
      function r8vec_sum ( n, v1 )

c*********************************************************************72
c
cc R8VEC_SUM sums the entries of an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    In FORTRAN90, the system routine SUM should be called
c    directly.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    22 July 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the dimension of the vectors.
c
c    Input, double precision V1(N), the vector.
c
c    Output, double precision R8VEC_SUM, the sum of the entries.
c
      implicit none

      integer n

      integer i
      double precision r8vec_sum
      double precision v1(n)
      double precision value

      value = 0.0D+00
      do i = 1, n
        value = value + v1(i)
      end do

      r8vec_sum = value

      return
      end
      subroutine r8vec_swap ( n, a1, a2 )

c*********************************************************************72
c
cc R8VEC_SWAP swaps two R8VEC's.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    22 July 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the arrays.
c
c    Input/output, double precision A1(N), A2(N), the vectors to swap.
c
      implicit none

      integer n

      double precision a1(n)
      double precision a2(n)
      integer i
      double precision t

      do i = 1, n
        t     = a1(i)
        a1(i) = a2(i)
        a2(i) = t
      end do

      return
      end
      subroutine r8vec_transpose_print ( n, a, title )

c*********************************************************************72
c
cc R8VEC_TRANSPOSE_PRINT prints an R8VEC "transposed".
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Example:
c
c    A = (/ 1.0, 2.1, 3.2, 4.3, 5.4, 6.5, 7.6, 8.7, 9.8, 10.9, 11.0 /)
c    TITLE = 'My vector:  '
c
c    My vector:   1.0    2.1    3.2    4.3    5.4
c                 6.5    7.6    8.7    9.8   10.9
c                11.0
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    22 December 2014
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of components of the vector.
c
c    Input, double precision A(N), the vector to be printed.
c
c    Input, character * ( * ) TITLE, a title.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      integer ihi
      integer ilo
      character * ( * ) title
      integer title_length

      title_length = len_trim ( title )

      do ilo = 1, n, 5
        ihi = min ( ilo + 5 - 1, n )
        if ( ilo .eq. 1 ) then
          write ( *, '(a,2x,5g14.6)' ) title(1:title_length), a(ilo:ihi)
        else
          do i = 1, title_length
            write ( *, '(a)', advance = 'no' ) ' '
          end do
          write ( *, '(2x,5g14.6)' ) a(ilo:ihi)
        end if
      end do

      return
      end
      subroutine r8vec_undex ( x_num, x_val, x_unique_num, tol, undx,
     &  xdnu )

c*********************************************************************72
c
cc R8VEC_UNDEX returns unique sorted indexes for an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    The goal of this routine is to determine a vector UNDX,
c    which points, to the unique elements of X, in sorted order,
c    and a vector XDNU, which identifies, for each entry of X, the index of
c    the unique sorted element of X.
c
c    This is all done with index vectors, so that the elements of
c    X are never moved.
c
c    The first step of the algorithm requires the indexed sorting
c    of X, which creates arrays INDX and XDNI.  (If all the entries
c    of X are unique, then these arrays are the same as UNDX and XDNU.)
c
c    We then use INDX to examine the entries of X in sorted order,
c    noting the unique entries, creating the entries of XDNU and
c    UNDX as we go.
c
c    Once this process has been completed, the vector X could be
c    replaced by a compressed vector XU, containing the unique entries
c    of X in sorted order, using the formula
c
c      XU(1:X_UNIQUE_NUM) = X(UNDX(1:X_UNIQUE_NUM)).
c
c    We could then, if we wished, reconstruct the entire vector X, or
c    any element of it, by index, as follows:
c
c      X(I) = XU(XDNU(I)).
c
c    We could then replace X by the combination of XU and XDNU.
c
c    Later, when we need the I-th entry of X, we can locate it as
c    the XDNU(I)-th entry of XU.
c
c    Here is an example of a vector X, the sort and inverse sort
c    index vectors, and the unique sort and inverse unique sort vectors
c    and the compressed unique sorted vector.
c
c      I    X   Indx  Xdni      XU   Undx  Xdnu
c    ----+-----+-----+-----+--------+-----+-----+
c      1 | 11.     1     1 |    11,     1     1
c      2 | 22.     3     5 |    22,     2     2
c      3 | 11.     6     2 |    33,     4     1
c      4 | 33.     9     8 |    55,     5     3
c      5 | 55.     2     9 |                  4
c      6 | 11.     7     3 |                  1
c      7 | 22.     8     6 |                  2
c      8 | 22.     4     7 |                  2
c      9 | 11.     5     4 |                  1
c
c    INDX(2) = 3 means that sorted item(2) is X(3).
c    XDNI(2) = 5 means that X(2) is sorted item(5).
c
c    UNDX(3) = 4 means that unique sorted item(3) is at X(4).
c    XDNU(8) = 2 means that X(8) is at unique sorted item(2).
c
c    XU(XDNU(I))) = X(I).
c    XU(I)        = X(UNDX(I)).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    04 July 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer X_NUM, the number of data values.
c
c    Input, double precision X_VAL(X_NUM), the data values.
c
c    Input, integer X_UNIQUE_NUM, the number of unique values
c    in X_VAL.  This value is only required for languages in which the size of
c    UNDX must be known in advance.
c
c    Input, double precision TOL, a tolerance for equality.
c
c    Output, integer UNDX(X_UNIQUE_NUM), the UNDX vector.
c
c    Output, integer XDNU(X_NUM), the XDNU vector.
c
      implicit none

      integer x_num
      integer x_unique_num

      integer i
      integer indx(x_num)
      integer j
      double precision tol
      integer undx(x_unique_num)
      double precision x_val(x_num)
      integer xdnu(x_num)
c
c  Implicitly sort the array.
c
      call r8vec_sort_heap_index_a ( x_num, x_val, indx )
c
c  Walk through the implicitly sorted array X.
c
      i = 1

      j = 1
      undx(j) = indx(i)

      xdnu(indx(i)) = j

      do i = 2, x_num

        if ( tol .lt. abs ( x_val(indx(i)) - x_val(undx(j)) ) ) then
          j = j + 1
          undx(j) = indx(i)
        end if

        xdnu(indx(i)) = j

      end do

      return
      end
      subroutine r8vec_uniform_01 ( n, seed, r )

c*********************************************************************72
c
cc R8VEC_UNIFORM_01 returns a unit pseudorandom R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    17 July 2006
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Paul Bratley, Bennett Fox, Linus Schrage,
c    A Guide to Simulation,
c    Springer Verlag, pages 201-202, 1983.
c
c    Bennett Fox,
c    Algorithm 647:
c    Implementation and Relative Efficiency of Quasirandom
c    Sequence Generators,
c    ACM Transactions on Mathematical Software,
c    Volume 12, Number 4, pages 362-376, 1986.
c
c    Peter Lewis, Allen Goodman, James Miller,
c    A Pseudo-Random Number Generator for the System/360,
c    IBM Systems Journal,
c    Volume 8, pages 136-143, 1969.
c
c  Parameters:
c
c    Input, integer N, the number of entries in the vector.
c
c    Input/output, integer SEED, the "seed" value, which should NOT be 0.
c    On output, SEED has been updated.
c
c    Output, double precision R(N), the vector of pseudorandom values.
c
      implicit none

      integer n

      integer i
      integer i4_huge
      parameter ( i4_huge = 2147483647 )
      integer k
      integer seed
      double precision r(n)

      do i = 1, n

        k = seed / 127773

        seed = 16807 * ( seed - k * 127773 ) - k * 2836

        if ( seed .lt. 0 ) then
          seed = seed + i4_huge
        end if

        r(i) = dble ( seed ) * 4.656612875D-10

      end do

      return
      end
      subroutine r8vec_uniform_ab ( n, a, b, seed, r )

c*********************************************************************72
c
cc R8VEC_UNIFORM_AB returns a scaled pseudorandom R8VEC.
c
c  Discussion:
c
c    Each dimension ranges from A to B.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    29 January 2005
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Paul Bratley, Bennett Fox, Linus Schrage,
c    A Guide to Simulation,
c    Second Edition,
c    Springer, 1987,
c    ISBN: 0387964673,
c    LC: QA76.9.C65.B73.
c
c    Bennett Fox,
c    Algorithm 647:
c    Implementation and Relative Efficiency of Quasirandom
c    Sequence Generators,
c    ACM Transactions on Mathematical Software,
c    Volume 12, Number 4, December 1986, pages 362-376.
c
c    Pierre L'Ecuyer,
c    Random Number Generation,
c    in Handbook of Simulation,
c    edited by Jerry Banks,
c    Wiley, 1998,
c    ISBN: 0471134031,
c    LC: T57.62.H37.
c
c    Peter Lewis, Allen Goodman, James Miller,
c    A Pseudo-Random Number Generator for the System/360,
c    IBM Systems Journal,
c    Volume 8, Number 2, 1969, pages 136-143.
c
c  Parameters:
c
c    Input, integer N, the number of entries in the vector.
c
c    Input, double precision A, B, the lower and upper limits.
c
c    Input/output, integer SEED, the "seed" value, which should NOT be 0.
c    On output, SEED has been updated.
c
c    Output, double precision R(N), the vector of pseudorandom values.
c
      implicit none

      integer n

      double precision a
      double precision b
      integer i
      integer i4_huge
      parameter ( i4_huge = 2147483647 )
      integer k
      integer seed
      double precision r(n)

      if ( seed .eq. 0 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8VEC_UNIFORM_AB - Fatal error!'
        write ( *, '(a)' ) '  Input value of SEED = 0.'
        stop 1
      end if

      do i = 1, n

        k = seed / 127773

        seed = 16807 * ( seed - k * 127773 ) - k * 2836

        if ( seed .lt. 0 ) then
          seed = seed + i4_huge
        end if

        r(i) = a + ( b - a ) * dble ( seed ) * 4.656612875D-10

      end do

      return
      end
      subroutine r8vec_uniform_abvec ( n, a, b, seed, r )

c*********************************************************************72
c
cc R8VEC_UNIFORM_ABVEC returns a scaled pseudorandom R8VEC.
c
c  Discussion:
c
c    Dimension I ranges from A(I) to B(I).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    01 October 2012
c
c  Author:
c
c    John Burkardt
c
c  Reference:
c
c    Paul Bratley, Bennett Fox, Linus Schrage,
c    A Guide to Simulation,
c    Second Edition,
c    Springer, 1987,
c    ISBN: 0387964673,
c    LC: QA76.9.C65.B73.
c
c    Bennett Fox,
c    Algorithm 647:
c    Implementation and Relative Efficiency of Quasirandom
c    Sequence Generators,
c    ACM Transactions on Mathematical Software,
c    Volume 12, Number 4, December 1986, pages 362-376.
c
c    Pierre L'Ecuyer,
c    Random Number Generation,
c    in Handbook of Simulation,
c    edited by Jerry Banks,
c    Wiley, 1998,
c    ISBN: 0471134031,
c    LC: T57.62.H37.
c
c    Peter Lewis, Allen Goodman, James Miller,
c    A Pseudo-Random Number Generator for the System/360,
c    IBM Systems Journal,
c    Volume 8, Number 2, 1969, pages 136-143.
c
c  Parameters:
c
c    Input, integer N, the number of entries in the vector.
c
c    Input, double precision A(N), B(N), the lower and upper limits.
c
c    Input/output, integer SEED, the "seed" value, which should NOT be 0.
c    On output, SEED has been updated.
c
c    Output, double precision R(N), the vector of pseudorandom values.
c
      implicit none

      integer n

      double precision a(n)
      double precision b(n)
      integer i
      integer i4_huge
      parameter ( i4_huge = 2147483647 )
      integer k
      integer seed
      double precision r(n)

      if ( seed .eq. 0 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'R8VEC_UNIFORM_ABVEC - Fatal error!'
        write ( *, '(a)' ) '  Input value of SEED = 0.'
        stop 1
      end if

      do i = 1, n

        k = seed / 127773

        seed = 16807 * ( seed - k * 127773 ) - k * 2836

        if ( seed .lt. 0 ) then
          seed = seed + i4_huge
        end if

        r(i) = a(i) + ( b(i) - a(i) ) * dble ( seed ) * 4.656612875D-10

      end do

      return
      end
      subroutine r8vec_uniform_unit ( m, seed, w )

c*********************************************************************72
c
cc R8VEC_UNIFORM_UNIT generates a uniformly random unit vector.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    04 October 2012
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, the spatial dimension.
c
c    Input/output, integer SEED, a seed for the random number 
c    generator.
c
c    Output, double precision W(M), a random direction vector,
c    with unit norm.
c
      implicit none

      integer m

      integer i
      double precision norm
      double precision r8vec_norm_l2
      integer seed
      double precision w(m)
c
c  Get N values from a standard normal distribution.
c
      call r8vec_normal_01 ( m, seed, w )
c
c  Compute the length of the vector.
c
      norm = r8vec_norm_l2 ( m, w )
c
c  Normalize the vector.
c
      do i = 1, m
        w(i) = w(i) / norm
      end do

      return
      end
      subroutine r8vec_unique_count ( n, a, tol, unique_num )

c*********************************************************************72
c
cc R8VEC_UNIQUE_COUNT counts the unique elements in an unsorted R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8 values.
c
c    Because the array is unsorted, this algorithm is O(N^2).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    28 August 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of elements of A.
c
c    Input, double precision A(N), the unsorted array to examine.
c
c    Input, double precision TOL, a nonnegative tolerance for equality.
c    Set it to 0.0 for the strictest test.
c
c    Output, integer UNIQUE_NUM, the number of unique elements
c    of A.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      integer j
      integer unique_num
      double precision tol

      unique_num = 0

      do i = 1, n

        unique_num = unique_num + 1

        do j = 1, i - 1

          if ( abs ( a(i) - a(j) ) .le. tol ) then
            unique_num = unique_num - 1
            exit
          end if

        end do

      end do

      return
      end
      subroutine r8vec_unique_index ( n, a, tol, unique_index )

c*********************************************************************72
c
cc R8VEC_UNIQUE_INDEX indexes the first occurrence of values in an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    For element A(I) of the vector, FIRST_UNIQUE(I) is the uniqueness index
c    of A(I).  That is, if A_UNIQUE contains the unique elements of A,
c    gathered in order, then
c
c      A_UNIQUE ( UNIQUE_INDEX(I) ) = A(I)
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    28 August 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of elements of A.
c
c    Input, double precision A(N), the array.
c
c    Input, double precision TOL, a tolerance for equality.
c
c    Output, integer UNIQUE_INDEX(N), the unique index.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      integer j
      double precision tol
      integer unique_index(n)
      integer unique_num

      do i = 1, n
        unique_index(i) = -1
      end do
      unique_num = 0

      do i = 1, n

        if ( unique_index(i) .eq. -1 ) then

          unique_num = unique_num + 1
          unique_index(i) = unique_num

          do j = i + 1, n
            if ( abs ( a(i) - a(j) ) .le. tol ) then
              unique_index(j) = unique_num
            end if
          end do

        end if

      end do

      return
      end
      subroutine r8vec_variance ( n, a, variance )

c*********************************************************************72
c
cc R8VEC_VARIANCE returns the variance of an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c    The variance of a vector X of length N is defined as
c
c      mean ( X(1:n) ) = sum ( X(1:n) ) / n
c
c      var ( X(1:n) ) = sum ( ( X(1:n) - mean )^2 ) / ( n - 1 )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    04 July 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the vector.
c    N should be at least 2.
c
c    Input, double precision A(N), the vector.
c
c    Output, double precision VARIANCE, the variance of the vector.
c
      implicit none

      integer n

      double precision a(n)
      integer i
      double precision mean
      double precision variance

      if ( n .lt. 2 ) then

        variance = 0.0D+00

      else

        mean = 0.0D+00
        do i = 1, n
          mean = mean + a(i)
        end do
        mean = mean / dble ( n )

        variance = 0.0D+00
        do i = 1, n
          variance = variance + ( a(i) - mean ) ** 2
        end do
        variance = variance / dble ( n - 1 )

      end if

      return
      end
      subroutine r8vec_vector_triple_product ( v1, v2, v3, v )

c*********************************************************************72
c
cc R8VEC_VECTOR_TRIPLE_PRODUCT computes the vector triple product.
c
c  Discussion:
c
c    VTRIPLE = V1 x ( V2 x V3 )
c
c    VTRIPLE is a vector perpendicular to V1, lying in the plane
c    spanned by V2 and V3.  The norm of VTRIPLE is the product
c    of the norms of V1, V2 and V3.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    27 October 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision V1(3), V2(3), V3(3), three vectors.
c
c    Output, double precision V(3), the vector triple product.
c
      implicit none

      double precision v(3)
      double precision v1(3)
      double precision v2(3)
      double precision v3(3)
      double precision v4(3)

      call r8vec_cross_product_3d ( v2, v3, v4 )

      call r8vec_cross_product_3d ( v1, v4, v )

      return
      end
      subroutine r8vec_write ( n, r, output_file )

c*********************************************************************72
c
cc R8VEC_WRITE writes an R8VEC to a file.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    20 August 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the order of the matrix.
c
c    Input, double precision R(N), the vector to be written.
c
c    Input, character * ( * ) OUTPUT_FILE, the name of the file to which
c    the information is to be written.
c
      implicit none

      integer n

      integer i
      character * ( * ) output_file
      integer output_unit
      double precision r(n)

      call get_unit ( output_unit )

      open ( unit = output_unit, file = output_file,
     &  status = 'replace' )

      do i = 1, n
        write ( output_unit, '(2x,g16.8)' ) r(i)
      end do

      close ( unit = output_unit )

      return
      end
      subroutine r8vec_zero ( n, a )

c*********************************************************************72
c
cc R8VEC_ZERO zeroes out an R8VEC.
c
c  Discussion:
c
c    An R8VEC is a vector of R8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    04 July 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the vector.
c
c    Output, double precision A(N), the vector to be zeroed.
c
      implicit none

      integer n

      double precision a(n)
      integer i

      do i = 1, n
        a(i) = 0.0D+00
      end do

      return
      end
      subroutine r8vec2_compare ( n, a1, a2, i, j, isgn )

c*********************************************************************72
c
cc R8VEC2_COMPARE compares two entries in an R8VEC2.
c
c  Discussion:
c
c    An R8VEC2 is a dataset consisting of N pairs of R8's, stored
c    as two separate vectors A1 and A2.
c
c    The lexicographic ordering is used.
c
c  Example:
c
c    A1(I) A2(I)   A1(J) A2(J)  ISGN
c    -----------   -----------  ----
c    1.0   5.0  <  1.0   6.0     -1
c    1.0   5.0  <  2.0   8.0     -1
c    1.0   5.0  <  9.0   1.0     -1
c    1.0   5.0  =  1.0   5.0      0
c    1.0   5.0  >  0.0   2.0     +1
c    1.0   5.0  >  0.0   5.0     +1
c    1.0   5.0  >  1.0   3.0     +1
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    22 August 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of data items.
c
c    Input, double precision A1(N), A2(N), the two components of each item.
c
c    Input, integer I, J, the items to be compared.
c
c    Output, integer ISGN, the results of the comparison:
c    -1, item I < item J,
c     0, item I = item J,
c    +1, item I > item J.
c
      implicit none

      integer n

      double precision a1(n)
      double precision a2(n)
      integer i
      integer isgn
      integer j

      isgn = 0

           if ( a1(i) .lt. a1(j) ) then

        isgn = -1

      else if ( a1(i) .eq. a1(j) ) then

             if ( a2(i) .lt. a2(j) ) then
          isgn = -1
        else if ( a2(i) .lt. a2(j) ) then
          isgn = 0
        else if ( a2(j) .lt. a2(i) ) then
          isgn = +1
        end if

      else if ( a1(j) .lt. a1(i) ) then

        isgn = +1

      end if

      return
      end
      subroutine r8vec2_print ( n, a1, a2, title )

c*********************************************************************72
c
cc R8VEC2_PRINT prints an R8VEC2.
c
c  Discussion:
c
c    An R8VEC2 is a dataset consisting of N pairs of R8s, stored
c    as two separate vectors A1 and A2.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    06 February 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of components of the vector.
c
c    Input, double precision A1(N), A2(N), the vectors to be printed.
c
c    Input, character * ( * ) TITLE, a title.
c
      implicit none

      integer n

      double precision a1(n)
      double precision a2(n)
      integer i
      character * ( * ) title

      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) trim ( title )
      write ( *, '(a)' ) ' '
      do i = 1, n
        write ( *, '(2x,i8,a,1x,g14.6,2x,g14.6)' ) i, ':', a1(i), a2(i)
      end do

      return
      end
      subroutine r8vec2_print_some ( n, x1, x2, max_print, title )

c*********************************************************************72
c
cc R8VEC2_PRINT_SOME prints "some" of an R8VEC2.
c
c  Discussion:
c
c    An R8VEC2 is a dataset consisting of N pairs of R8's, stored
c    as two separate vectors A1 and A2.
c
c    The user specifies MAX_PRINT, the maximum number of lines to print.
c
c    If N, the size of the vectors, is no more than MAX_PRINT, then
c    the entire vectors are printed, one entry of each per line.
c
c    Otherwise, if possible, the first MAX_PRINT-2 entries are printed,
c    followed by a line of periods suggesting an omission,
c    and the last entry.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    08 June 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries of the vectors.
c
c    Input, double precision X1(N), X2(N), the vector to be printed.
c
c    Input, integer MAX_PRINT, the maximum number of lines
c    to print.
c
c    Input, character * ( * ) TITLE, a title.
c
      implicit none

      integer n

      integer i
      integer max_print
      character * ( * ) title
      double precision x1(n)
      double precision x2(n)

      if ( max_print .le. 0 ) then
        return
      end if

      if ( n .le. 0 ) then
        return
      end if

      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) trim ( title )
      write ( *, '(a)' ) ' '

      if ( n .le. max_print ) then

        do i = 1, n
          write ( *, '(2x,i8,2x,g14.6,2x,g14.6)' ) i, x1(i), x2(i)
        end do

      else if ( 3 .le. max_print ) then

        do i = 1, max_print - 2
          write ( *, '(2x,i8,2x,g14.6,2x,g14.6)' ) i, x1(i), x2(i)
        end do
        write ( *, '(a)' ) '  ......  ..............  ..............'
        i = n
        write ( *, '(2x,i8,2x,g14.6,2x,g14.6)' ) i, x1(i), x2(i)

      else

        do i = 1, max_print - 1
          write ( *, '(2x,i8,2x,g14.6,2x,g14.6)' ) i, x1(i), x2(i)
        end do
        i = max_print
        write ( *, '(2x,i8,2x,g14.6,2x,g14.6,2x,a)' ) i, x1(i), x2(i), 
     &    '...more entries...'

      end if

      return
      end
      subroutine r8vec2_sort_a ( n, a1, a2 )

c*********************************************************************72
c
cc R8VEC2_SORT_A ascending sorts an R8VEC2.
c
c  Discussion:
c
c    An R8VEC2 is a dataset consisting of N pairs of R8's, stored
c    as two separate vectors A1 and A2.
c
c    Each item to be sorted is a pair (I,J), with the I
c    and J values stored in separate vectors A1 and A2.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    05 September 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of items of data.
c
c    Input/output, double precision A1(N), A2(N), the data to be sorted.
c
      implicit none

      integer n

      double precision a1(n)
      double precision a2(n)
      integer i
      integer indx
      integer isgn
      integer j

      if ( n .le. 1 ) then
        return
      end if
c
c  Initialize.
c
      i = 0
      indx = 0
      isgn = 0
      j = 0
c
c  Call the external heap sorter.
c
10    continue

        call sort_heap_external ( n, indx, i, j, isgn )
c
c  Interchange the I and J objects.
c
        if ( 0 .lt. indx ) then

          call r8_swap ( a1(i), a1(j) )
          call r8_swap ( a2(i), a2(j) )
c
c  Compare the I and J objects.
c
        else if ( indx .lt. 0 ) then

          call r8vec2_compare ( n, a1, a2, i, j, isgn )

        else if ( indx .eq. 0 ) then

          go to 20

        end if

      go to 10

20    continue

      return
      end
      subroutine r8vec2_sort_d ( n, a1, a2 )

c*********************************************************************72
c
cc R8VEC2_SORT_D descending sorts an R8VEC2.
c
c  Discussion:
c
c    An R8VEC2 is a dataset consisting of N pairs of R8's, stored
c    as two separate vectors A1 and A2.
c
c    Each item to be sorted is a pair (I,J), with the I
c    and J values stored in separate vectors A1 and A2.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    05 September 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of items of data.
c
c    Input/output, double precision A1(N), A2(N), the data to be sorted.
c
      implicit none

      integer n

      double precision a1(n)
      double precision a2(n)
      integer i
      integer indx
      integer isgn
      integer j

      if ( n .le. 1 ) then
        return
      end if
c
c  Initialize.
c
      i = 0
      indx = 0
      isgn = 0
      j = 0
c
c  Call the external heap sorter.
c
10    continue

        call sort_heap_external ( n, indx, i, j, isgn )
c
c  Interchange the I and J objects.
c
        if ( 0 .lt. indx ) then

          call r8_swap ( a1(i), a1(j) )
          call r8_swap ( a2(i), a2(j) )
c
c  Compare the I and J objects.
c  Reverse the value of ISGN to effect a descending sort.
c
        else if ( indx .lt. 0 ) then

          call r8vec2_compare ( n, a1, a2, i, j, isgn )

          isgn = -isgn

        else if ( indx .eq. 0 ) then

          go to 20

        end if

      go to 10

20    continue

      return
      end
      subroutine r8vec2_sort_heap_index_a ( n, x, y, indx )

c*********************************************************************72
c
cc R8VEC2_SORT_HEAP_INDEX_A does an indexed heap ascending sort of an R8VEC2.
c
c  Discussion:
c
c    An R8VEC2 is a dataset consisting of N pairs of R8's, stored
c    as two separate vectors A1 and A2.
c
c    The sorting is not actually carried out.  Rather an index array is
c    created which defines the sorting.  This array may be used to sort
c    or index the array, or to sort or index related arrays keyed on the
c    original array.
c
c    ( X(I), Y(I) ) < ( X(J), Y(J) ) if:
c
c    * X(I) < X(J), or
c
c    * X(I) = X(J), and Y(I) < Y(J).
c
c    Once the index array is computed, the sorting can be carried out
c    "implicitly:
c
c      ( X(INDX(1:N)), Y(INDX(1:N) ), is sorted,
c
c    or explicitly, by the call
c
c      call r8vec_permute ( n, indx, x )
c      call r8vec_permute ( n, indx, y )
c
c    after which ( X(1:N), Y(1:N) ), is sorted.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    05 September 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the array.
c
c    Input, double precision X(N),Y(N), pairs of X, Y coordinates of points.
c
c    Output, integer INDX(N), the sort index.  The
c    I-th element of the sorted array has coordinates ( X(INDX(I)), Y(INDX(I) ).
c
      implicit none

      integer n

      integer i
      integer indx(n)
      integer indxt
      integer ir
      integer j
      integer l
      double precision x(n)
      double precision xval
      double precision y(n)
      double precision yval

      if ( n .lt. 1 ) then
        return
      end if

      call i4vec_indicator1 ( n, indx )

      if ( n .eq. 1 ) then
        return
      end if

      l = n / 2 + 1
      ir = n

10    continue

        if ( 1 .lt. l ) then

          l = l - 1
          indxt = indx(l)
          xval = x(indxt)
          yval = y(indxt)

        else

          indxt = indx(ir)
          xval = x(indxt)
          yval = y(indxt)
          indx(ir) = indx(1)
          ir = ir - 1

          if ( ir .eq. 1 ) then
            indx(1) = indxt
            go to 30
          end if

        end if

        i = l
        j = l + l

20      continue

        if ( j .le. ir ) then

          if ( j .lt. ir ) then

            if ( x(indx(j)) .lt. x(indx(j+1)) .or. 
     &        ( x(indx(j)) .eq. x(indx(j+1)) .and. 
     &          y(indx(j)) .lt. y(indx(j+1)) ) ) then
              j = j + 1
            end if

          end if

          if ( xval .lt. x(indx(j)) .or. 
     &        ( xval .eq. x(indx(j)) .and. yval .lt. y(indx(j)) ) ) then
            indx(i) = indx(j)
            i = j
            j = j + j
          else
            j = ir + 1
          end if

          go to 20

        end if

        indx(i) = indxt

      go to 10

30    continue

      return
      end
      subroutine r8vec2_sorted_unique ( n, a1, a2, unique_num )

c*********************************************************************72
c
cc R8VEC2_SORTED_UNIQUE keeps unique elements in a sorted R8VEC2.
c
c  Discussion:
c
c    An R8VEC2 is a dataset consisting of N pairs of R8's, stored
c    as two separate vectors A1 and A2.
c
c    Item I is stored as the pair A1(I), A2(I).
c
c    The items must have been sorted, or at least it must be the
c    case that equal items are stored in adjacent vector locations.
c
c    If the items were not sorted, then this routine will only
c    replace a string of equal values by a single representative.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    05 September 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of items.
c
c    Input/output, double precision A1(N), A2(N).
c    On input, the array of N items.
c    On output, an array of unique items.
c
c    Output, integer UNIQUE_NUM, the number of unique items.
c
      implicit none

      integer n

      double precision a1(n)
      double precision a2(n)
      integer itest
      integer unique_num

      if ( n .le. 0 ) then
        unique_num = 0
        return
      end if

      unique_num = 1

      do itest = 2, n

        if ( a1(itest) .ne. a1(unique_num) .or. 
     &       a2(itest) .ne. a2(unique_num) ) then

          unique_num = unique_num + 1

          a1(unique_num) = a1(itest)
          a2(unique_num) = a2(itest)

        end if

      end do

      return
      end
      subroutine r8vec2_sorted_unique_index ( n, a1, a2, unique_num, 
     &  indx )

c*********************************************************************72
c
cc R8VEC2_SORTED_UNIQUE_INDEX indexes unique elements in a sorted R8VEC2.
c
c  Discussion:
c
c    An R8VEC2 is a dataset consisting of N pairs of R8's, stored
c    as two separate vectors A1 and A2.
c
c    Item I is stored as the pair A1(I), A2(I).
c
c    The items must have been sorted, or at least it should be the
c    case that equal items are stored in adjacent vector locations.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    05 September 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of items.
c
c    Input, double precision A1(N), A2(N), the array of N items.
c
c    Output, integer UNIQUE_NUM, the number of unique items.
c
c    Output, integer INDX(N), contains in entries 1 through
c    UNIQUE_NUM an index array of the unique items.  To build new arrays
c    with no repeated elements:
c      B1(1:UNIQUE_NUM) = A1(INDX(1:UNIQUE_NUM))
c
      implicit none

      integer n

      double precision a1(n)
      double precision a2(n)
      integer i
      integer indx(n)
      integer itest
      integer unique_num

      if ( n .le. 0 ) then
        unique_num = 0
        return
      end if

      unique_num = 1
      indx(1) = 1

      do itest = 2, n

        if ( a1(itest-1) .ne. a1(itest) .or. 
     &       a2(itest-1) .ne. a2(itest) ) then

          unique_num = unique_num + 1

          indx(unique_num) = itest

        end if

      end do

      do i = unique_num + 1, n
        indx(i) = 0
      end do

      return
      end
      subroutine r8vec2_sum_max_index ( n, a, b, sum_max_index )

c*********************************************************************72
c
cc R8VEC2_SUM_MAX_INDEX returns the index of the maximum sum of two R8VEC's.
c
c  Discussion:
c
c    An R8VEC2 is a dataset consisting of N pairs of R8's, stored
c    as two separate vectors A1 and A2.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    05 September 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the array.
c
c    Input, double precision A(N), B(N), two arrays whose sum
c    is to be examined.
c
c    Output, integer SUM_MAX_INDEX, the index of the largest
c    entry in A+B.
c
      implicit none

      integer n

      double precision a(n)
      double precision b(n)
      integer i
      double precision sum_max
      integer sum_max_index

      if ( n .le. 0 ) then

        sum_max_index = -1

      else

        sum_max_index = 1
        sum_max = a(1) + b(1)

        do i = 2, n
          if ( sum_max .lt. a(i) + b(i) ) then
            sum_max = a(i) + b(i)
            sum_max_index = i
          end if
        end do

      end if

      return
      end
      subroutine r8vec3_print ( n, a1, a2, a3, title )

c*********************************************************************72
c
cc R8VEC3_PRINT prints an R8VEC3.
c
c  Discussion:
c
c    An R8VEC3 is a dataset consisting of N triples of R8's, stored
c    as three separate vectors A1, A2, A3.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    05 September 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of components of the vector.
c
c    Input, double precision A1(N), A2(N), A3(N), the vectors to be printed.
c
c    Input, character * ( * ) TITLE, a title.
c
      implicit none

      integer n

      double precision a1(n)
      double precision a2(n)
      double precision a3(n)
      integer i
      character * ( * ) title

      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) trim ( title )
      write ( *, '(a)' ) ' '

      do i = 1, n
        write ( *, '(i8,3g14.6)' ) i, a1(i), a2(i), a3(i)
      end do

      return
      end
      subroutine roots_to_r8poly ( n, x, c )

c*********************************************************************72
c
cc ROOTS_TO_R8POLY converts polynomial roots to polynomial coefficients.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    17 July 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of roots specified.
c
c    Input, double precision X(N), the roots.
c
c    Output, double precision C(0:N), the coefficients of the polynomial.
c
      implicit none

      integer n

      double precision c(0:n)
      integer i
      integer j
      double precision x(n)
c
c  Initialize C to (0, 0, ..., 0, 1).
c  Essentially, we are setting up a divided difference table.
c
      do i = 0, n - 1
        c(i) = 0.0D+00
      end do
      c(n) = 1.0D+00
c
c  Convert to standard polynomial form by shifting the abscissas
c  of the divided difference table to 0.
c
      do j = 1, n
        do i = 1, n + 1 - j
          c(n-i) = c(n-i) - x(n+1-i-j+1) * c(n-i+1)
        end do
      end do

      return
      end
      subroutine sort_heap_external ( n, indx, i, j, isgn )

c*********************************************************************72
c
cc SORT_HEAP_EXTERNAL externally sorts a list of items into ascending order.
c
c  Discussion:
c
c    The actual list of data is not passed to the routine.  Hence this
c    routine may be used to sort integers, reals, numbers, names,
c    dates, shoe sizes, and so on.  After each call, the routine asks
c    the user to compare or interchange two items, until a special
c    return value signals that the sorting is completed.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    25 January 2007
c
c  Author:
c
c    Original FORTRAN77 version by Albert Nijenhuis, Herbert Wilf.
c    This FORTRAN77 version by John Burkardt.
c
c  Reference:
c
c    Albert Nijenhuis, Herbert Wilf,
c    Combinatorial Algorithms for Computers and Calculators,
c    Academic Press, 1978,
c    ISBN: 0-12-519260-6,
c    LC: QA164.N54.
c
c  Parameters:
c
c    Input, integer N, the number of items to be sorted.
c
c    Input/output, integer INDX, the main communication signal.
c
c    The user must set INDX to 0 before the first call.
c    Thereafter, the user should not change the value of INDX until
c    the sorting is done.
c
c    On return, if INDX is
c
c      greater than 0,
c      * interchange items I and J;
c      * call again.
c
c      less than 0,
c      * compare items I and J;
c      * set ISGN = -1 if I .lt. J, ISGN = +1 if J .lt. I;
c      * call again.
c
c      equal to 0, the sorting is done.
c
c    Output, integer I, J, the indices of two items.
c    On return with INDX positive, elements I and J should be interchanged.
c    On return with INDX negative, elements I and J should be compared, and
c    the result reported in ISGN on the next call.
c
c    Input, integer ISGN, results of comparison of elements I and J.
c    (Used only when the previous call returned INDX less than 0).
c    ISGN .le. 0 means I is less than or equal to J;
c    0 .le. ISGN means I is greater than or equal to J.
c
      implicit none

      integer i
      integer i_save
      integer indx
      integer isgn
      integer j
      integer j_save
      integer k
      integer k1
      integer n
      integer n1

      save i_save
      save j_save
      save k
      save k1
      save n1

      data i_save / 0 /
      data j_save / 0 /
      data k / 0 /
      data k1 / 0 /
      data n1 / 0 /
c
c  INDX = 0: This is the first call.
c
      if ( indx .eq. 0 ) then

        i_save = 0
        j_save = 0
        k = n / 2
        k1 = k
        n1 = n
c
c  INDX .lt. 0: The user is returning the results of a comparison.
c
      else if ( indx .lt. 0 ) then

        if ( indx .eq. -2 ) then

          if ( isgn .lt. 0 ) then
            i_save = i_save + 1
          end if

          j_save = k1
          k1 = i_save
          indx = -1
          i = i_save
          j = j_save
          return

        end if

        if ( 0 .lt. isgn ) then
          indx = 2
          i = i_save
          j = j_save
          return
        end if

        if ( k .le. 1 ) then

          if ( n1 .eq. 1 ) then
            i_save = 0
            j_save = 0
            indx = 0
          else
            i_save = n1
            n1 = n1 - 1
            j_save = 1
            indx = 1
          end if

          i = i_save
          j = j_save
          return

        end if

        k = k - 1
        k1 = k
c
c  0 .lt. INDX, the user was asked to make an interchange.
c
      else if ( indx .eq. 1 ) then

        k1 = k

      end if

10    continue

        i_save = 2 * k1

        if ( i_save .eq. n1 ) then
          j_save = k1
          k1 = i_save
          indx = -1
          i = i_save
          j = j_save
          return
        else if ( i_save .le. n1 ) then
          j_save = i_save + 1
          indx = -2
          i = i_save
          j = j_save
          return
        end if

        if ( k .le. 1 ) then
          go to 20
        end if

        k = k - 1
        k1 = k

      go to 10

20    continue

      if ( n1 .eq. 1 ) then
        i_save = 0
        j_save = 0
        indx = 0
        i = i_save
        j = j_save
      else
        i_save = n1
        n1 = n1 - 1
        j_save = 1
        indx = 1
        i = i_save
        j = j_save
      end if

      return
      end
      subroutine timestamp ( )

c*********************************************************************72
c
cc TIMESTAMP prints out the current YMDHMS date as a timestamp.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    12 January 2007
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    None
c
      implicit none

      character * ( 8 ) ampm
      integer d
      character * ( 8 ) date
      integer h
      integer m
      integer mm
      character * ( 9 ) month(12)
      integer n
      integer s
      character * ( 10 ) time
      integer y

      save month

      data month /
     &  'January  ', 'February ', 'March    ', 'April    ',
     &  'May      ', 'June     ', 'July     ', 'August   ',
     &  'September', 'October  ', 'November ', 'December ' /

      call date_and_time ( date, time )

      read ( date, '(i4,i2,i2)' ) y, m, d
      read ( time, '(i2,i2,i2,1x,i3)' ) h, n, s, mm

      if ( h .lt. 12 ) then
        ampm = 'AM'
      else if ( h .eq. 12 ) then
        if ( n .eq. 0 .and. s .eq. 0 ) then
          ampm = 'Noon'
        else
          ampm = 'PM'
        end if
      else
        h = h - 12
        if ( h .lt. 12 ) then
          ampm = 'PM'
        else if ( h .eq. 12 ) then
          if ( n .eq. 0 .and. s .eq. 0 ) then
            ampm = 'Midnight'
          else
            ampm = 'AM'
          end if
        end if
      end if

      write ( *,
     &  '(i2,1x,a,1x,i4,2x,i2,a1,i2.2,a1,i2.2,a1,i3.3,1x,a)' )
     &  d, month(m), y, h, ':', n, ':', s, '.', mm, ampm

      return
      end
