      function c8_abs ( z )

c*********************************************************************72
c
cc C8_ABS evaluates the absolute value of a C8.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    09 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double complex Z, the argument.
c
c    Output, double precision C8_ABS, the function value.
c
      implicit none

      double precision c8_abs
      double complex z

      c8_abs = dsqrt ( ( dreal ( z ) )**2 + ( dimag ( z ) )**2 )

      return
      end
      function c8_acos ( z )

c*********************************************************************72
c
cc C8_ACOS evaluates the inverse cosine of a C8.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c    FORTRAN77 does not have an intrinsic inverse cosine function for C8 arguments.
c
c    Here we use the relationship:
c
c      C8_ACOS ( Z ) = pi/2 - C8_ASIN ( Z ).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    10 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double complex Z, the argument.
c
c    Output, double complex C8_ACOS, the function value.
c
      implicit none

      double complex c8_acos
      double complex c8_asin
      double precision r8_pi_half
      parameter ( r8_pi_half = 1.57079632679489661923D+00 )
      double complex z

      c8_acos = r8_pi_half - c8_asin ( z )

      return
      end
      function c8_acosh ( z )

c*********************************************************************72
c
cc C8_ACOSH evaluates the inverse hyperbolic cosine of a C8.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c    FORTRAN77 does not have an intrinsic inverse hyperbolic 
c    cosine function for C8 arguments.
c
c    Here we use the relationship:
c
c      C8_ACOSH ( Z ) = i * C8_ACOS ( Z ).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    07 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double complex Z, the argument.
c
c    Output, double complex C8_ACOSH, the function value.
c
      implicit none

      double complex c8_acos
      double complex c8_acosh
      double complex c8_i
      double complex z

      c8_i = dcmplx ( 0.0D+00, 1.0D+00 )

      c8_acosh = c8_i * c8_acos ( z )

      return
      end
      function c8_add ( z1, z2 )

c*********************************************************************72
c
cc C8_ADD adds two C8's.
c
c  Discussion:
c
c    A C8 is a double complex value.
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
c    Input, double complex Z1, Z2, the values to add.
c
c    Output, double complex C8_ADD, the function value.
c
      implicit none

      double complex c8_add
      double complex z1
      double complex z2

      c8_add = z1 + z2

      return
      end
      function c8_arg ( x )

c*********************************************************************72
c
cc C8_ARG returns the argument of a C8.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c    The value returned by this function is always between 0.0 and 2*PI.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    08 February 2015
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double complex X, the value whose argument is desired.
c
c    Output, double precision C8_ARG, the function value.
c
      implicit none

      double precision c8_arg
      double complex x
      double precision r8_atan

      if ( dimag ( x ) .eq. 0.0D+00 .and. 
     &     dreal ( x ) .eq. 0.0D+00 ) then
  
        c8_arg = 0.0D+00

      else

        c8_arg = r8_atan ( dimag ( x ), dreal ( x ) )

      end if

      return
      end
      function c8_asin ( z )

c*********************************************************************72
c
cc C8_ASIN evaluates the inverse sine of a C8.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c    FORTRAN77 does not have an intrinsic inverse sine function for C8 arguments.
c
c    Here we use the relationship:
c
c      C8_ASIN ( Z ) = - i * log ( i * z + sqrt ( 1 - z * z ) )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    10 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double complex Z, the argument.
c
c    Output, double complex C8_ASIN, the function value.
c
      implicit none

      double complex c8_asin
      double complex c8_i
      double complex c8_log
      double complex c8_sqrt
      double complex z

      c8_i = dcmplx ( 0.0D+00, 1.0D+00 )

      c8_asin = - c8_i 
     &  * c8_log ( c8_i * z + c8_sqrt ( 1.0D+00 - z * z ) )

      return
      end
      function c8_asinh ( z )

c*********************************************************************72
c
cc C8_ASINH evaluates the inverse hyperbolic sine of a C8.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c    FORTRAN77 does not have an intrinsic inverse hyperbolic 
c    sine function for C8 arguments.
c
c    Here we use the relationship:
c
c      C8_ASINH ( Z ) = - i * C8_ASIN ( i * Z ).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    07 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double complex Z, the argument.
c
c    Output, double complex C8_ASINH, the function value.
c
      implicit none

      double complex c8_asin
      double complex c8_asinh
      double complex c8_i
      double complex z

      c8_i = dcmplx ( 0.0D+00, 1.0D+00 )

      c8_asinh = - c8_i * c8_asin ( c8_i * z )

      return
      end
      function c8_atan ( z )

c*********************************************************************72
c
cc C8_ATAN evaluates the inverse tangent of a C8.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c    FORTRAN77 does not have an intrinsic inverse tangent function
c    for C8 arguments.
c
c    FORTRAN77 does not have a logarithm function for C8 argumentsc
c
c    Here we use the relationship:
c
c      C8_ATAN ( Z ) = ( i / 2 ) * log ( ( 1 - i * z ) / ( 1 + i * z ) )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    07 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double complex Z, the argument.
c
c    Output, double complex C8_ATAN, the function value.
c
      implicit none

      double complex arg
      double complex c8_atan
      double complex c8_log
      double complex c8_i
      double complex z

      c8_i = dcmplx ( 0.0D+00, 1.0D+00 )

      arg = ( 1.0D+00 - c8_i * z ) / ( 1.0D+00 + c8_i * z )

      c8_atan = 0.5D+00 * c8_i * c8_log ( arg ) 

      return
      end
      function c8_atanh ( z )

c*********************************************************************72
c
cc C8_ATANH evaluates the inverse hyperbolic tangent of a C8.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c    FORTRAN77 does not have an intrinsic inverse hyperbolic 
c    tangent function for C8 arguments.
c
c    Here we use the relationship:
c
c      C8_ATANH ( Z ) = - i * C8_ATAN ( i * Z ).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    07 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double complex Z, the argument.
c
c    Output, double complex C8_ATANH, the function value.
c
      implicit none

      double complex c8_atan
      double complex c8_atanh
      double complex c8_i
      double complex z

      c8_i = dcmplx ( 0.0D+00, 1.0D+00 )

      c8_atanh = - c8_i * c8_atan ( c8_i * z )

      return
      end
      function c8_conj ( z )

c*********************************************************************72
c
cc C8_CONJ evaluates the conjugate of a C8.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    10 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double complex Z, the argument.
c
c    Output, double complex C8_CONJ, the function value.
c
      implicit none

      double complex c8_conj
      double complex z

      c8_conj = dcmplx ( dreal ( z ), - dimag ( z ) )

      return
      end
      function c8_copy ( z )

c*********************************************************************72
c
cc C8_COPY copies a C8.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    12 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double complex Z, the argument.
c
c    Output, double complex C8_COPY, the function value.
c
      implicit none

      double complex c8_copy
      double complex z

      c8_copy = z

      return
      end
      function c8_cos ( z )

c*********************************************************************72
c
cc C8_COS evaluates the cosine of a C8.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c    We use the relationship:
c
c      C8_COS ( C ) = ( C8_EXP ( i * C ) + C8_EXP ( - i * C ) ) / 2
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    12 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double complex Z, the argument.
c
c    Output, double complex C8_COS, the function value.
c
      implicit none

      double complex c8_cos
      double complex c8_exp
      double complex c8_i
      double complex z

      c8_i = cmplx ( 0.0D+00, 1.0D+00 )

      c8_cos = ( c8_exp ( c8_i * z ) + c8_exp ( - c8_i * z ) ) 
     &  / 2.0D+00

      return
      end
      function c8_cosh ( z )

c*********************************************************************72
c
cc C8_COSH evaluates the hyperbolic cosine of a C8.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    30 November 2007
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double complex Z, the argument.
c
c    Output, double complex C8_COSH, the function value.
c
      implicit none

      double complex c8_cosh
      double complex c8_exp
      double complex z

      c8_cosh = ( c8_exp ( z ) + c8_exp ( - z ) ) / 2.0D+00

      return
      end
      function c8_cube_root ( x )

c*********************************************************************72
c
cc C8_CUBE_ROOT returns the principal cube root of a C8.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    07 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double complex X, the number whose cube root is desired.
c
c    Output, double complex C8_CUBE_ROOT, the function value.
c
      implicit none

      double precision arg
      double precision c8_arg
      double complex c8_cube_root
      double precision c8_mag
      double precision mag
      double complex x

      arg = c8_arg ( x )

      mag = c8_mag ( x )

      if ( mag .eq. 0.0D+00 ) then

        c8_cube_root = dcmplx ( 0.0D+00, 0.0D+00 )

      else

        c8_cube_root = mag**( 1.0D+00 / 3.0D+00 ) 
     &    * dcmplx ( dcos ( arg / 3.0D+00 ), 
     &               dsin ( arg / 3.0D+00 ) )

      end if

      return
      end
      function c8_div ( z1, z2 )

c*********************************************************************72
c
cc C8_DIV divides two C8's.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    09 February 2015
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double complex Z1, Z2, the arguments.
c
c    Output, double complex C8_DIV, the function value.
c
      implicit none

      double precision a
      double precision b
      double precision c
      double complex c8_div
      double precision c8_imag
      double precision c8_real
      double precision d
      double precision e
      double precision f
      double precision g
      double complex z1
      double complex z2

      a = c8_real ( z1 )
      b = c8_imag ( z1 )
      c = c8_real ( z2 )
      d = c8_imag ( z2 )

      e = c * c + d * d

      f = ( a * c + b * d ) / e
      g = ( b * c - a * d ) / e

      c8_div = dcmplx ( f, g )

      return
      end
      function c8_div_r8 ( z1, r )

c*********************************************************************72
c
cc C8_DIV_R8 divides a C8 by an R8.
c
c  Discussion:
c
c    A C8 is a double complex value.
c    An R8 is a double precision value.
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
c    Input, double complex Z1, the value to be divided.
c
c    Input, double precision R, the divisor.
c
c    Output, double complex C8_DIV_R8, the function value.
c
      implicit none

      double complex c8_div_r8
      double precision r
      double complex z1

      c8_div_r8 = z1 / r

      return
      end
      function c8_exp ( z )

c*********************************************************************72
c
cc C8_EXP evaluates the exponential of a C8.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    07 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double complex Z, the argument.
c
c    Output, double complex C8_EXP, the function value.
c
      implicit none

      double complex   c8_exp
      double complex   z
      double precision zi
      double precision zr

      zr = dreal ( z )
      zi = dimag ( z )

      c8_exp = dexp ( zr ) * dcmplx ( dcos ( zi ), dsin ( zi ) )

      return
      end
      function c8_i ( )

c*********************************************************************72
c
cc C8_I returns the the imaginary unit, i as a C8.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    07 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Output, double complex C8_I, the value of complex i.
c
      implicit none

      double complex c8_i

      c8_i = dcmplx ( 0.0D+00, 1.0D+00 )

      return
      end
      function c8_imag ( z )

c*********************************************************************72
c
cc C8_IMAG evaluates the imaginary part of a C8.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    12 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double complex Z, the argument.
c
c    Output, double precision C8_IMAG, the function value.
c
      implicit none

      double precision c8_imag
      double complex z

      c8_imag = dimag ( z )

      return
      end
      function c8_inv ( z )

c*********************************************************************72
c
cc C8_INV evaluates the inverse of a C8.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    10 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double complex Z, the argument.
c
c    Output, double complex C8_INV, the function value.
c
      implicit none

      double complex c8_inv
      double complex z
      double precision z_imag
      double precision z_norm
      double precision z_real

      z_real = dreal ( z )
      z_imag = dimag ( z )

      z_norm = dsqrt ( z_real * z_real + z_imag * z_imag )

      c8_inv = dcmplx ( z_real, - z_imag ) / z_norm / z_norm

      return
      end
      function c8_le_l1 ( x, y )

c*********************************************************************72
c
cc C8_LE_L1 := X <= Y for C8 values, and the L1 norm.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c    The L1 norm can be defined here as:
c
c      C8_NORM_L1(X) = abs ( real (X) ) + abs ( imag (X) )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    07 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double complex X, Y, the values to be compared.
c
c    Output, logical C8_LE_L1, is TRUE if X <= Y.
c
      implicit none

      logical c8_le_l1
      double complex x
      double complex y

      if ( dabs ( dreal ( x ) ) + dabs ( dimag ( x ) ) .le. 
     &     dabs ( dreal ( y ) ) + dabs ( dimag ( y ) ) ) then
        c8_le_l1 = .true.
      else
        c8_le_l1 = .false.
      end if

      return
      end
      function c8_le_l2 ( x, y )

c*********************************************************************72
c
cc C8_LE_L2 := X <= Y for complex values, and the L2 norm.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c    The L2 norm can be defined here as:
c
c      value = sqrt ( ( real (X) )**2 + ( imag (X) )**2 )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    07 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double complex X, Y, the values to be compared.
c
c    Output, logical C8_LE_L2, is TRUE if X <= Y.
c
      implicit none

      logical c8_le_l2
      logical value
      double complex x
      double complex y

      if ( ( dreal ( x ) )**2 + ( dimag ( x ) )**2 .le.
     &     ( dreal ( y ) )**2 + ( dimag ( y ) )**2 ) then
        value = .true.
      else
        value = .false.
      end if

      c8_le_l2 = value

      return
      end
      function c8_le_li ( x, y )

c*********************************************************************72
c
cc C8_LE_LI := X <= Y for C8 values, and the L Infinity norm.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c    The L Infinity norm can be defined here as:
c
c      C8_NORM_LI(X) = max ( abs ( real (X) ), abs ( imag (X) ) )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    07 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double complex X, Y, the values to be compared.
c
c    Output, logical C8_LE_LI, is TRUE if X <= Y.
c
      implicit none

      logical c8_le_li
      double complex x
      double complex y

      if ( dmax1 ( dabs ( dreal ( x ) ), dabs ( dimag ( x ) ) ) .le.
     &     dmax1 ( dabs ( dreal ( y ) ), dabs ( dimag ( y ) ) ) ) then
        c8_le_li = .true.
      else
        c8_le_li = .false.
      end if

      return
      end
      function c8_log ( z )

c*********************************************************************72
c
cc C8_LOG evaluates the logarithm of a C8.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c    FORTRAN77 does not have a logarithm function for C8 argumentsc
c
c    Here we use the relationship:
c
c      C8_LOG ( Z ) = LOG ( R ) + i * ARG ( R )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    07 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double complex Z, the argument.
c
c    Output, double complex C8_LOG, the function value.
c
      implicit none

      double precision arg
      double precision c8_arg
      double complex c8_i
      double complex c8_log
      double precision c8_mag
      double precision r
      double complex z

      c8_i = dcmplx ( 0.0D+00, 1.0D+00 )

      arg = c8_arg ( z )
      r = c8_mag ( z )

      c8_log = dlog ( r ) + c8_i * arg
 
      return
      end
      function c8_mag ( x )

c*********************************************************************72
c
cc C8_MAG returns the magnitude of a C8.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    07 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double complex X, the value whose magnitude is desired.
c
c    Output, double precision C8_MAG, the function value.
c
      implicit none

      double precision c8_mag
      double complex x

      c8_mag = dsqrt ( ( dreal ( x ) )**2 + ( dimag ( x ) )**2 )

      return
      end
      function c8_mul ( z1, z2 )

c*********************************************************************72
c
cc C8_MUL multiplies two C8's.
c
c  Discussion:
c
c    A C8 is a double complex value.
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
c    Input, double complex Z1, Z2, the values to multiply.
c
c    Output, double complex C8_MUL, the function value.
c
      implicit none

      double complex c8_mul
      double complex z1
      double complex z2

      c8_mul = z1 * z2

      return
      end
      function c8_neg ( c1 )

c*********************************************************************72
c
cc C8_NEG returns the negative of a C8.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    12 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double complex C1, the value to be negated.
c
c    Output, double complex C8_NEG, the function value.
c
      implicit none

      double complex c1
      double complex c8_neg

      c8_neg = - c1

      return
      end
      function c8_nint ( c1 )

c*********************************************************************72
c
cc C8_NINT returns the nearest complex integer of a C8.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    02 March 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double complex C1, the value to be NINT'ed.
c
c    Output, double complex C8_NINT, the NINT'ed value.
c
      implicit none

      double complex c1
      double complex c8_nint
      double precision r
      double precision r_min
      double precision r8_floor
      double precision x
      double precision x_min
      double precision xc
      double precision y
      double precision y_min
      double precision yc

      xc = dreal ( c1 )
      yc = dimag ( c1 )
c
c  Lower left.
c
      x = r8_floor ( dreal ( c1 ) )
      y = r8_floor ( dimag ( c1 ) )
      r = ( x - xc )**2 + ( y - yc )**2
      r_min = r
      x_min = x
      y_min = y
c
c  Lower right.
c
      x = r8_floor ( dreal ( c1 ) ) + 1.0D+00
      y = r8_floor ( dimag ( c1 ) )
      r = ( x - xc )**2 + ( y - yc )**2
      if ( r .lt. r_min ) then
        r_min = r
        x_min = x
        y_min = y
      end if
c
c  Upper right.
c
      x = r8_floor ( dreal ( c1 ) ) + 1.0D+00
      y = r8_floor ( dimag ( c1 ) ) + 1.0D+00
      r = ( x - xc )**2 + ( y - yc )**2
      if ( r .lt. r_min ) then
        r_min = r
        x_min = x
        y_min = y
      end if
c
c  Upper left.
c
      x = r8_floor ( dreal ( c1 ) )
      y = r8_floor ( dimag ( c1 ) ) + 1.0D+00
      r = ( x - xc )**2 + ( y - yc )**2
      if ( r .lt. r_min ) then
        r_min = r
        x_min = x
        y_min = y
      end if

      c8_nint = dcmplx ( x_min, y_min )

      return
      end
      function c8_norm_l1 ( x )

c*********************************************************************72
c
cc C8_NORM_L1 evaluates the L1 norm of a C8.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c    Numbers of equal norm lie along diamonds centered at (0,0).
c
c    The L1 norm can be defined here as:
c
c      C8_NORM_L1(X) = abs ( real (X) ) + abs ( imag (X) )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    07 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double complex X, the value whose norm is desired.
c
c    Output, double precision C8_NORM_L1, the norm of X.
c
      implicit none

      double precision c8_norm_l1
      double complex x

      c8_norm_l1 = dabs ( dreal ( x ) ) + dabs ( dimag ( x ) )

      return
      end
      function c8_norm_l2 ( x )

c*********************************************************************72
c
cc C8_NORM_L2 evaluates the L2 norm of a C8.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c    Numbers of equal norm lie on circles centered at (0,0).
c
c    The L2 norm can be defined here as:
c
c      C8_NORM_L2(X) = sqrt ( ( real (X) )**2 + ( imag ( X ) )**2 )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    07 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double complex X, the value whose norm is desired.
c
c    Output, double precision C8_NORM_L2, the 2-norm of X.
c
      implicit none

      double precision c8_norm_l2
      double complex x

      c8_norm_l2 = dsqrt ( ( dreal ( x ) )**2 
     &                   + ( dimag ( x ) )**2 )

      return
      end
      function c8_norm_li ( x )

c*********************************************************************72
c
cc C8_NORM_LI evaluates the L-infinity norm of a C8.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c    Numbers of equal norm lie along squares whose centers are at (0,0).
c
c    The L-infinity norm can be defined here as:
c
c      C8_NORM_LI(X) = max ( abs ( real (X) ), abs ( imag (X) ) )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    07 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double complex X, the value whose norm is desired.
c
c    Output, double precision C8_NORM_LI, the infinity norm of X.
c
      implicit none

      double precision c8_norm_li
      double complex x

      c8_norm_li = dmax1 ( dabs ( dreal ( x ) ), dabs ( dimag ( x ) ) )

      return
      end
      function c8_normal_01 ( seed )

c*********************************************************************72
c
cc C8_NORMAL_01 returns a unit pseudonormal C8.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    07 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input/output, integer SEED, a seed for the random number generator.
c
c    Output, double complex C8_NORMAL_01, a sample of the PDF.
c
      implicit none

      double complex c8_normal_01
      double precision r8_pi
      parameter ( r8_pi = 3.141592653589793D+00 )
      double precision r8_uniform_01
      integer seed
      double precision v1
      double precision v2
      double precision x_c
      double precision x_r

      v1 = r8_uniform_01 ( seed )
      v2 = r8_uniform_01 ( seed )

      x_r = dsqrt ( - 2.0D+00 * dlog ( v1 ) ) 
     &  * dcos ( 2.0D+00 * r8_pi * v2 )

      x_c = dsqrt ( - 2.0D+00 * dlog ( v1 ) ) 
     &  * dsin ( 2.0D+00 * r8_pi * v2 )

      c8_normal_01 = dcmplx ( x_r, x_c )

      return
      end
      function c8_one ( )

c*********************************************************************72
c
cc C8_ONE returns the value of 1 as a C8.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    07 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Output, double complex C8_ONE, the value of complex 1.
c
      implicit none

      double complex c8_one

      c8_one = dcmplx ( 1.0D+00, 0.0D+00 )

      return
      end
      subroutine c8_print ( a, title )

c*********************************************************************72
c
cc C8_PRINT prints a C8.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    14 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double complex A, the value to be printed.
c
c    Input, character*(*) TITLE, a title.
c
      implicit none

      double complex a
      character*(*) title

      if ( 0 .lt. len_trim ( title ) ) then
        write ( *, '(a,2x,a,g14.6,a,g14.6,a)' ) 
     &    trim ( title ), '(', dreal ( a ), ',', dimag ( a ), ')'
      else
        write ( *, '(a,g14.6,a,g14.6,a)' ) 
     &    '(', dreal ( a ), ',', dimag ( a ), ')'
      end if

      return
      end
      function c8_real ( z )

c*********************************************************************72
c
cc C8_REAL evaluates the real part of a C8.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    12 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double complex Z, the argument.
c
c    Output, double precision C8_REAL, the function value.
c
      implicit none

      double precision c8_real
      double complex z

      c8_real = dreal ( z )

      return
      end
      function c8_sin ( z )

c*********************************************************************72
c
cc C8_SIN evaluates the sine of a C8.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c    We use the relationship:
c
c      C8_SIN ( C ) = - i * ( C8_EXP ( i * C ) - C8_EXP ( - i * C ) ) / 2
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    12 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double complex Z, the argument.
c
c    Output, double complex C8_SIN, the function value.
c
      implicit none

      double complex c8_exp
      double complex c8_i
      double complex c8_sin
      double complex z

      c8_i = dcmplx ( 0.0D+00, 1.0D+00 )

      c8_sin = - c8_i 
     &  * ( c8_exp ( c8_i * z ) - c8_exp ( - c8_i * z ) ) 
     &  / 2.0D+00

      return
      end
      function c8_sinh ( z )

c*********************************************************************72
c
cc C8_SINH evaluates the hyperbolic sine of a C8.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    07 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double complex Z, the argument.
c
c    Output, double complex C8_SINH, the function value.
c
      implicit none

      double complex c8_exp
      double complex c8_sinh
      double complex z

      c8_sinh = ( c8_exp ( z ) - c8_exp ( - z ) ) / 2.0D+00

      return
      end
      function c8_sqrt ( x )

c*********************************************************************72
c
cc C8_SQRT returns the principal square root of a C8.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    07 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double complex X, the number whose square root is desired.
c
c    Output, double complex C8_SQRT, the function value.
c
      implicit none

      double precision arg
      double precision c8_arg
      double precision c8_mag
      double complex c8_sqrt
      double precision mag
      double complex x

      arg = c8_arg ( x )
      mag = c8_mag ( x )

      if ( mag .eq. 0.0D+00 ) then

        c8_sqrt = dcmplx ( 0.0D+00, 0.0D+00 )

      else

        c8_sqrt = dsqrt ( mag ) 
     &    * dcmplx ( dcos ( arg / 2.0D+00 ), 
     &               dsin ( arg / 2.0D+00 ) )

      end if

      return
      end
      function c8_sub ( z1, z2 )

c*********************************************************************72
c
cc C8_SUB subtracts two C8's.
c
c  Discussion:
c
c    A C8 is a double complex value.
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
c    Input, double complex Z1, Z2, the values to subtract.
c
c    Output, double complex C8_SUB, the function value.
c
      implicit none

      double complex c8_sub
      double complex z1
      double complex z2

      c8_sub = z1 - z2

      return
      end
      subroutine c8_swap ( x, y )

c*********************************************************************72
c
cc C8_SWAP swaps two C8's.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    07 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input/output, double complex X, Y.  On output, the values of X and
c    Y have been interchanged.
c
      implicit none

      double complex x
      double complex y
      double complex z

      z = x
      x = y
      y = z

      return
      end
      function c8_tan ( z )

c*********************************************************************72
c
cc C8_TAN evaluates the tangent of a C8.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c    We use the relationship:
c
c      C8_TAN ( C ) = - i * ( C8_EXP ( i * C ) - C8_EXP ( - i * C ) ) 
c                         / ( C8_EXP ( I * C ) + C8_EXP ( - i * C ) )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    12 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double complex Z, the argument.
c
c    Output, double complex C8_TAN, the function value.
c
      implicit none

      double complex c8_exp
      double complex c8_i
      double complex c8_tan
      double complex z

      c8_i = dcmplx ( 0.0D+00, 1.0D+00 )

      c8_tan = - c8_i 
     &  * ( c8_exp ( c8_i * z ) - c8_exp ( - c8_i * z ) ) 
     &  / ( c8_exp ( c8_i * z ) + c8_exp ( - c8_i * z ) )

      return
      end
      function c8_tanh ( z )

c*********************************************************************72
c
cc C8_TANH evaluates the hyperbolic tangent of a C8.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    07 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double complex Z, the argument.
c
c    Output, double complex C8_TANH, the function value.
c
      implicit none

      double complex c8_exp
      double complex c8_tanh
      double complex z

      c8_tanh = ( c8_exp ( z ) - c8_exp ( - z ) ) 
     &        / ( c8_exp ( z ) + c8_exp ( - z ) )

      return
      end
      subroutine c8_to_cartesian ( z, x, y )

c*********************************************************************72
c
cc C8_TO_CARTESIAN converts a C8 to Cartesian form.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    12 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double complex Z, the argument.
c
c    Output, double precision X, Y, the Cartesian form.
c
      implicit none

      double precision x
      double precision y
      double complex z

      x = dreal ( z )
      y = dimag ( z )

      return
      end
      subroutine c8_to_polar ( z, r, theta )

c*********************************************************************72
c
cc C8_TO_POLAR converts a C8 to polar form.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    12 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double complex Z, the argument.
c
c    Output, double precision R, THETA, the polar form.
c
      implicit none

      double precision c8_arg
      double precision c8_mag
      double precision r
      double precision theta
      double complex z

      r = c8_mag ( z )
      theta = c8_arg ( z )

      return
      end
      function c8_uniform_01 ( seed )

c*********************************************************************72
c
cc C8_UNIFORM_01 returns a unit pseudorandom C8.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c    The angle should be uniformly distributed between 0 and 2 * PI,
c    the square root of the radius uniformly distributed between 0 and 1.
c
c    This results in a uniform distribution of values in the unit circle.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    07 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input/output, integer SEED, the "seed" value, which should NOT be 0.
c    On output, SEED has been updated.
c
c    Output, double complex Z_UNIFORM_01, a pseudorandom complex value.
c
      implicit none

      double precision r
      integer k
      double precision r8_pi
      parameter ( r8_pi = 3.141592653589793D+00 )
      integer seed
      double precision theta
      double complex c8_uniform_01

      if ( seed .eq. 0 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'C8_UNIFORM_01 - Fatal errorc'
        write ( *, '(a)' ) '  Input value of SEED = 0.'
        stop
      end if

      k = seed / 127773

      seed = 16807 * ( seed - k * 127773 ) - k * 2836

      if ( seed .lt. 0 ) then
        seed = seed + 2147483647
      end if

      r = sqrt ( dble ( seed ) * 4.656612875D-10 )

      k = seed / 127773

      seed = 16807 * ( seed - k * 127773 ) - k * 2836

      if ( seed .lt. 0 ) then
        seed = seed + 2147483647
      end if

      theta = 2.0D+00 * r8_pi * ( dble ( seed ) * 4.656612875D-10 )

      c8_uniform_01 = r * dcmplx ( dcos ( theta ), dsin ( theta ) )

      return
      end
      function c8_zero ( )

c*********************************************************************72
c
cc C8_ZERO returns the value of 0 as a C8.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    07 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Output, double complex C8_ZERO, the value of complex 0.
c
      implicit none

      double complex c8_zero

      c8_zero = dcmplx ( 0.0D+00, 0.0D+00 )

      return
      end
      subroutine c8mat_add ( m, n, alpha, a, beta, b, c )

c*********************************************************************72
c
cc C8MAT_ADD combines two C8MAT's with complex scalar factors.
c
c  Discussion:
c
c    A C8MAT is a matrix of C8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    03 March 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the order of the matrix.
c
c    Input, double complex ALPHA, the first scale factor.
c
c    Input, double complex A(M,N), the first matrix.
c
c    Input, double complex BETA, the second scale factor.
c
c    Input, double complex B(M,N), the second matrix.
c
c    Output, double complex C(M,N), the result.
c
      implicit none

      integer m
      integer n

      double complex a(m,n)
      double complex alpha
      double complex b(m,n)
      double complex beta
      double complex c(m,n)
      integer i
      integer j

      do j = 1, n
        do i = 1, m
          c(i,j) = alpha * a(i,j) + beta * b(i,j)
        end do
      end do

      return
      end
      subroutine c8mat_add_r8 ( m, n, alpha, a, beta, b, c )

c*********************************************************************72
c
cc C8MAT_ADD_R8 combines two C8MAT's with real scalar factors.
c
c  Discussion:
c
c    A C8MAT is a matrix of C8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    03 March 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the order of the matrix.
c
c    Input, double precision ALPHA, the first scale factor.
c
c    Input, double complex A(M,N), the first matrix.
c
c    Input, double precision BETA, the second scale factor.
c
c    Input, double complex B(M,N), the second matrix.
c
c    Output, double complex C(M,N), the result.
c
      implicit none

      integer m
      integer n

      double complex a(m,n)
      double precision alpha
      double complex b(m,n)
      double precision beta
      double complex c(m,n)
      integer i
      integer j

      do j = 1, n
        do i = 1, m
          c(i,j) = alpha * a(i,j) + beta * b(i,j)
        end do
      end do

      return
      end
      subroutine c8mat_copy ( m, n, a, b )

c*********************************************************************72
c
cc C8MAT_COPY copies a C8MAT.
c
c  Discussion:
c
c    A C8MAT is a matrix of C8's.
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
c    Input, integer M, N, the order of the matrix.
c
c    Input, double complex A(M,N), the matrix.
c
c    Output, double complex B(M,N), the copied matrix.
c
      implicit none

      integer m
      integer n

      double complex a(m,n)
      double complex b(m,n)
      integer i
      integer j

      do j = 1, n
        do i = 1, m
          b(i,j) = a(i,j)
        end do
      end do

      return
      end
      subroutine c8mat_fss ( n, a, nb, b, info )

c*********************************************************************72
c
cc C8MAT_FSS factors and solves a system with multiple right hand sides.
c
c  Discussion:
c
c    A C8MAT is an MxN array of C8's, stored by (I,J) -> [I+J*M].
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
c    01 March 2013
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
c    Input/output, double complex A(N,N).
c    On input, A is the coefficient matrix of the linear system.
c    On output, A is in unit upper triangular form, and
c    represents the U factor of an LU factorization of the
c    original coefficient matrix.
c
c    Input, integer NB, the number of right hand sides.
c
c    Input/output, double complex B(N,NB).
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

      double complex a(n,n)
      double complex b(n,nb)
      integer i
      integer info
      integer ipiv
      integer j
      integer jcol
      integer k
      double precision piv
      double complex temp

      info = 0

      do jcol = 1, n
c
c  Find the maximum element in column I.
c
        piv = cdabs ( a(jcol,jcol) )
        ipiv = jcol
        do i = jcol + 1, n
          if ( piv .lt. cdabs ( a(i,jcol) ) ) then
            piv = cdabs ( a(i,jcol) )
            ipiv = i
          end if
        end do

        if ( piv .eq. 0.0D+00 ) then
          info = jcol
          write ( *, '(a)' ) ' '
          write ( *, '(a)' ) 'C8MAT_FSS - Fatal error!'
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
      subroutine c8mat_identity ( n, a )

c*********************************************************************72
c
cc C8MAT_IDENTITY sets a C8MAT to the identity.
c
c  Discussion:
c
c    A C8MAT is a matrix of C8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    07 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the order of the matrix.
c
c    Output, double complex A(N,N), the matrix.
c
      implicit none

      integer n

      double complex a(n,n)
      integer i
      integer j

      do j = 1, n
        do i = 1, n
          if ( i .eq. j ) then
            a(i,j) = dcmplx ( 1.0D+00, 0.0D+00 )
          else
            a(i,j) = dcmplx ( 0.0D+00, 0.0D+00 )
          end if
        end do
      end do

      return
      end
      subroutine c8mat_indicator ( m, n, a )

c*********************************************************************72
c
cc C8MAT_INDICATOR returns the C8MAT indicator matrix.
c
c  Discussion:
c
c    A C8MAT is a matrix of C8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    07 December 2008
c
c  Author
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns.
c
c    Output, double complex A(M,N), the matrix.
c
      implicit none

      integer m
      integer n

      double complex a(m,n)
      integer i
      integer j

      do j = 1, n
        do i = 1, m
          a(i,j) = dcmplx ( i, j )
        end do
      end do

      return
      end
      subroutine c8mat_minvm ( n1, n2, a, b, c )

c*********************************************************************72
c
cc C8MAT_MINVM computes inverse(A) * B for C8MAT's.
c
c  Discussion:
c
c    A C8MAT is an array of C8 values.
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
c    Input, integer N1, N2, the order of the matrices.
c
c    Input, double complex A(N1,N1), B(N1,N2), the matrices.
c
c    Output, double complex C(N1,N2), the result, C = inverse(A) * B.
c
      implicit none

      integer n1
      integer n2

      double complex a(n1,n1)
      double complex alu(n1,n1)
      double complex b(n1,n2)
      double complex c(n1,n2)
      integer info

      call c8mat_copy ( n1, n1, a, alu )
      call c8mat_copy ( n1, n2, b, c )

      call c8mat_fss ( n1, alu, n2, c, info )
 
      if ( info .ne. 0 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'C8MAT_MINVM - Fatal error!'
        write ( *, '(a)' ) '  The matrix A was numerically singular.'
        stop
      end if

      return
      end
      subroutine c8mat_mm ( n1, n2, n3, a, b, c )

c*********************************************************************72
c
cc C8MAT_MM multiplies two C8MAT's.
c
c  Discussion:
c
c    A C8MAT is an MxN array of C8's, stored by (I,J) -> [I+J*M].
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    03 March 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N1, N2, N3, define the orders of
c    the matrices.
c
c    Input, double complex A(N1,N2), B(N2,N3), the matrix factors.
c
c    Output, double complex C(N1,N3), the product matrix.
c
      implicit none

      integer n1
      integer n2
      integer n3

      double complex a(n1,n2)
      double complex b(n2,n3)
      double complex c(n1,n3)
      double complex c1(n1,n3)
      integer i
      integer j
      integer k

      do k = 1, n3
        do i = 1, n1
          c1(i,k) = 0.0D+00
          do j = 1, n2
            c1(i,k) = c1(i,k) + a(i,j) * b(j,k)
          end do
        end do
      end do

      call c8mat_copy ( n1, n3, c1, c )

      return
      end
      subroutine c8mat_nint ( m, n, a )

c*********************************************************************72
c
cc C8MAT_NINT rounds the entries of a C8MAT.
c
c  Discussion:
c
c    A C8MAT is a matrix of C8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    07 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns of A.
c
c    Input/output, double complex A(M,N), the matrix to be NINT'ed.
c
      implicit none

      integer m
      integer n

      double complex a(m,n)
      double complex c8_nint
      integer i
      integer j

      do j = 1, n
        do i = 1, m
          a(i,j) = c8_nint ( a(i,j) )
        end do
      end do

      return
      end
      function c8mat_norm_fro ( m, n, a )

c*********************************************************************72
c
cc C8MAT_NORM_FRO returns the Frobenius norm of a C8MAT.
c
c  Discussion:
c
c    A C8MAT is an MxN array of C8's, stored by (I,J) -> [I+J*M].
c
c    The Frobenius norm is defined as
c
c      C8MAT_NORM_FRO = sqrt (
c        sum ( 1 <= I <= M ) sum ( 1 <= j <= N ) A(I,J) * A(I,J) )
c
c    The matrix Frobenius norm is not derived from a vector norm, but
c    is compatible with the vector L2 norm, so that:
c
c      c8vec_norm_l2 ( A * x ) <= c8mat_norm_fro ( A ) * c8vec_norm_l2 ( x ).
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
c    Input, integer M, the number of rows in A.
c
c    Input, integer N, the number of columns in A.
c
c    Input, double complex A(M,N), the matrix whose Frobenius
c    norm is desired.
c
c    Output, double precision C8MAT_NORM_FRO, the Frobenius norm of A.
c
      implicit none

      integer m
      integer n

      double complex a(m,n)
      double precision c8mat_norm_fro
      integer i
      integer j
      double precision value

      value = 0.0D+00
      do j = 1, n
        do i = 1, m
          value = value + ( cdabs ( a(i,j) ) )**2
        end do
      end do

      c8mat_norm_fro = sqrt ( value )

      return
      end
      function c8mat_norm_l1 ( m, n, a )

c*********************************************************************72
c
cc C8MAT_NORM_L1 returns the matrix L1 norm of a C8MAT.
c
c  Discussion:
c
c    A C8MAT is an MxN array of C8's, stored by (I,J) -> [I+J*M].
c
c    The matrix L1 norm is defined as:
c
c      C8MAT_NORM_L1 = max ( 1 <= J <= N )
c        sum ( 1 <= I <= M ) abs ( A(I,J) ).
c
c    The matrix L1 norm is derived from the vector L1 norm, and
c    satisifies:
c
c      c8vec_norm_l1 ( A * x ) <= c8mat_norm_l1 ( A ) * c8vec_norm_l1 ( x ).
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
c    Input, integer M, the number of rows in A.
c
c    Input, integer N, the number of columns in A.
c
c    Input, double complex A(M,N), the matrix whose L1 norm is desired.
c
c    Output, double precision C8MAT_NORM_L1, the L1 norm of A.
c
      implicit none

      integer m
      integer n

      double complex a(m,n)
      double precision c8mat_norm_l1
      double precision col_sum
      integer i
      integer j

      c8mat_norm_l1 = 0.0D+00

      do j = 1, n
        col_sum = 0.0D+00
        do i = 1, m
          col_sum = col_sum + abs ( a(i,j) )
        end do
        c8mat_norm_l1 = max ( c8mat_norm_l1, col_sum )
      end do

      return
      end
      function c8mat_norm_li ( m, n, a )

c*********************************************************************72
c
cc C8MAT_NORM_LI returns the matrix L-oo norm of a C8MAT.
c
c  Discussion:
c
c    A C8MAT is an array of C8 values.
c
c    The matrix L-oo norm is defined as:
c
c      C8MAT_NORM_LI =  max ( 1 <= I <= M ) sum ( 1 <= J <= N ) abs ( A(I,J) ).
c
c    The matrix L-oo norm is derived from the vector L-oo norm,
c    and satisifies:
c
c      c8vec_norm_li ( A * x ) <= c8mat_norm_li ( A ) * c8vec_norm_li ( x ).
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
c    Input, integer M, the number of rows in A.
c
c    Input, integer N, the number of columns in A.
c
c    Input, double complex A(M,N), the matrix whose L-oo
c    norm is desired.
c
c    Output, double precision C8MAT_NORM_LI, the L-oo norm of A.
c
      implicit none

      integer m
      integer n

      double complex a(m,n)
      double precision c8mat_norm_li
      integer i
      integer j
      double precision row_sum

      c8mat_norm_li = 0.0D+00

      do i = 1, m
        row_sum = 0.0D+00
        do j = 1, n
          row_sum = row_sum + cdabs ( a(i,j) )
        end do
        c8mat_norm_li = max ( c8mat_norm_li, row_sum )
      end do

      return
      end
      subroutine c8mat_print ( m, n, a, title )

c*********************************************************************72
c
cc C8MAT_PRINT prints a C8MAT.
c
c  Discussion:
c
c    A C8MAT is a matrix of C8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    07 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns 
c    in the matrix.
c
c    Input, double complex A(M,N), the matrix.
c
c    Input, character * ( * ) TITLE, a title.
c
      implicit none

      integer m
      integer n

      double complex a(m,n)
      character * ( * ) title

      call c8mat_print_some ( m, n, a, 1, 1, m, n, title )

      return
      end
      subroutine c8mat_print_some ( m, n, a, ilo, jlo, ihi, jhi, 
     &  title )

c*********************************************************************72
c
cc C8MAT_PRINT_SOME prints some of a C8MAT.
c
c  Discussion:
c
c    A C8MAT is a matrix of C8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    21 June 2010
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns 
c    in the matrix.
c
c    Input, double complex A(M,N), the matrix.
c
c    Input, integer ILO, JLO, IHI, JHI, the first row and
c    column, and the last row and column to be printed.
c
c    Input, character * ( * ) TITLE, a title.
c
      implicit none

      integer incx
      parameter ( incx = 4 )
      integer m
      integer n

      double complex a(m,n)
      character * ( 20 ) ctemp(incx)
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
      double complex zero

      zero = dcmplx ( 0.0D+00, 0.0D+00 )

      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) trim ( title )

      if ( m .le. 0 .or. n .le. 0 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) '  (None)' 
        return
      end if
c
c  Print the columns of the matrix, in strips of INCX.
c
      do j2lo = jlo, min ( jhi, n ), incx

        j2hi = j2lo + incx - 1
        j2hi = min ( j2hi, n )
        j2hi = min ( j2hi, jhi )

        inc = j2hi + 1 - j2lo

        write ( *, '(a)' ) ' '

        do j = j2lo, j2hi
          j2 = j + 1 - j2lo
          write ( ctemp(j2), '(i10,10x)' ) j
        end do

        write ( *, '(a,4a20)' ) '  Col: ', ( ctemp(j2), j2 = 1, inc )
        write ( *, '(a)' ) '  Row'
        write ( *, '(a)' ) '  ---'
c
c  Determine the range of the rows in this strip.
c
        i2lo = max ( ilo, 1 )
        i2hi = min ( ihi, m )

        do i = i2lo, i2hi
c
c  Print out (up to) INCX entries in row I, that lie in the current strip.
c
          do j2 = 1, inc

            j = j2lo - 1 + j2

            if ( dimag ( a(i,j) ) .eq. 0.0D+00 ) then
              write ( ctemp(j2), '(g10.3,10x)' ) dreal ( a(i,j) )
            else
              write ( ctemp(j2), '(2g10.3)' ) a(i,j)
            end if

          end do

          write ( *, '(i5,a,4a20)' ) i, ':', ( ctemp(j2), j2 = 1, inc )

        end do

      end do

      return
      end
      subroutine c8mat_scale ( m, n, alpha, a )

c*********************************************************************72
c
cc C8MAT_SCALE scales a C8MAT by a complex scalar.
c
c  Discussion:
c
c    A C8MAT is a matrix of C8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    03 March 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the order of the matrix.
c
c    Input, double complex ALPHA, the scale factor.
c
c    Input/output, double complex A(M,N), the matrix to be scaled.
c
      implicit none

      integer m
      integer n

      double complex a(m,n)
      double complex alpha
      integer i
      integer j

      do j = 1, n
        do i = 1, m
          a(i,j) = alpha * a(i,j)
        end do
      end do

      return
      end
      subroutine c8mat_scale_r8 ( m, n, alpha, a )

c*********************************************************************72
c
cc C8MAT_SCALE_R8 scales a C8MAT by a real scalar.
c
c  Discussion:
c
c    A C8MAT is a matrix of C8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    03 March 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the order of the matrix.
c
c    Input, double precision ALPHA, the scale factor.
c
c    Input/output, double complex A(M,N), the matrix to be scaled.
c
      implicit none

      integer m
      integer n

      double complex a(m,n)
      double precision alpha
      integer i
      integer j

      do j = 1, n
        do i = 1, m
          a(i,j) = alpha * a(i,j)
        end do
      end do

      return
      end
      subroutine c8mat_uniform_01 ( m, n, seed, c )

c*********************************************************************72
c
cc C8MAT_UNIFORM_01 returns a unit pseudorandom C8MAT.
c
c  Discussion:
c
c    A C8MAT is a matrix of C8's.
c
c    The angles should be uniformly distributed between 0 and 2 * PI,
c    the square roots of the radius uniformly distributed between 0 and 1.
c
c    This results in a uniform distribution of values in the unit circle.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    07 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer M, N, the number of rows and columns in the matrix.
c
c    Input/output, integer SEED, the "seed" value, which should NOT be 0.
c    On output, SEED has been updated.
c
c    Output, double complex C(M,N), the pseudorandom complex matrix.
c
      implicit none

      integer m
      integer n

      double complex c(m,n)
      integer i
      integer j
      integer k
      double precision r
      double precision r8_pi
      parameter ( r8_pi = 3.141592653589793D+00 )
      integer seed
      double precision theta

      if ( seed .eq. 0 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'C8MAT_UNIFORM_01 - Fatal error!'
        write ( *, '(a)' ) '  Input value of SEED = 0.'
        stop
      end if

      do j = 1, n
        do i = 1, m

          k = seed / 127773

          seed = 16807 * ( seed - k * 127773 ) - k * 2836

          if ( seed .lt. 0 ) then
            seed = seed + 2147483647
          end if

          r = dsqrt ( dble ( seed ) * 4.656612875D-10 )

          k = seed / 127773

          seed = 16807 * ( seed - k * 127773 ) - k * 2836

          if ( seed .lt. 0 ) then
            seed = seed + 2147483647
          end if

          theta = 2.0D+00 * r8_pi * ( dble ( seed ) * 4.656612875D-10 )

          c(i,j) = r * dcmplx ( dcos ( theta ), dsin ( theta ) )

        end do

      end do

      return
      end
      subroutine c8mat_zero ( m, n, a )

c*********************************************************************72
c
cc C8MAT_ZERO zeroes a C8MAT.
c
c  Discussion:
c
c    A C8MAT is a matrix of C8's.
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
c    Input, integer M, N, the order of the matrix.
c
c    Output, double complex A(M,N), the zeroed matrix.
c
      implicit none

      integer m
      integer n

      double complex a(m,n)
      integer i
      integer j

      do j = 1, n
        do i = 1, m
          a(i,j) = 0.0D+00
        end do
      end do

      return
      end
      subroutine c8vec_copy ( n, a, b )

c*********************************************************************72
c
cc C8VEC_COPY copies a C8VEC.
c
c  Discussion:
c
c    A C8VEC is a vector of C8's.
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
c    Input, integer N, the order of the vector.
c
c    Input, double complex A(N), the vector.
c
c    Output, double complex B(N), the copied vector.
c
      implicit none

      integer n

      double complex a(n)
      double complex b(n)
      integer i

      do i = 1, n
        b(i) = a(i)
      end do

      return
      end
      subroutine c8vec_indicator ( n, a )

c*********************************************************************72
c
cc C8VEC_INDICATOR sets a C8VEC to the indicator vector.
c
c  Discussion:
c
c    A C8VEC is a vector of C8's
c
c    X(1:N) = ( 1-1i, 2-2i, 3-3i, 4-4i, ... )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    07 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of elements of A.
c
c    Output, double complex A(N), the array to be initialized.
c
      implicit none

      integer n

      double complex a(n)
      integer i

      do i = 1, n
        a(i) = dcmplx ( i, -i )
      end do

      return
      end
      subroutine c8vec_nint ( n, a )

c*********************************************************************72
c
cc C8VEC_NINT rounds the entries of a C8VEC.
c
c  Discussion:
c
c    A C8VEC is a vector of C8's
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    02 March 2013
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the vector.
c
c    Input/output, double complex A(N), the vector to be NINT'ed.
c
      implicit none

      integer n

      double complex a(n)
      double complex c8_nint
      integer i

      do i = 1, n
        a(i) = c8_nint ( a(i) )
      end do

      return
      end
      function c8vec_norm_l1 ( n, a )

c*********************************************************************72
c
cc C8VEC_NORM_L1 returns the L1 norm of a C8VEC.
c
c  Discussion:
c
c    A C8VEC is a vector of C8's
c
c    The vector L1 norm is defined as:
c
c      C8VEC_NORM_L1 = sum ( 1 <= I <= N ) abs ( A(I) )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    09 February 2015
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries.
c
c    Input, double complex A(N), the vector.
c
c    Output, double precision C8VEC_NORM_L1, the norm.
c
      implicit none

      integer n

      double complex a(n)
      double precision c8vec_norm_l1
      integer i
      double precision value

      value = 0.0D+00
      do i = 1, n
        value = value + cdabs ( a(i) )
      end do

      c8vec_norm_l1 = value

      return
      end
      function c8vec_norm_l2 ( n, a )

c*********************************************************************72
c
cc C8VEC_NORM_L2 returns the L2 norm of a C8VEC.
c
c  Discussion:
c
c    A C8VEC is a vector of C8's
c
c    The vector L2 norm is defined as:
c
c      C8VEC_NORM_L2 = sqrt ( sum ( 1 <= I <= N ) conjg ( A(I) ) * A(I) ).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    07 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries.
c
c    Input, double complex A(N), the vector.
c
c    Output, double precision C8VEC_NORM_L2, the norm.
c
      implicit none

      integer n

      double complex a(n)
      double precision c8vec_norm_l2
      integer i

      c8vec_norm_l2 = 0.0D+00
      do i = 1, n
        c8vec_norm_l2 = c8vec_norm_l2 + dconjg ( a(i) ) * a(i)
      end do
      c8vec_norm_l2 = dsqrt ( c8vec_norm_l2 )

      return
      end
      function c8vec_norm_li ( n, a )

c*********************************************************************72
c
cc C8VEC_NORM_LI returns the Loo norm of a C8VEC.
c
c  Discussion:
c
c    A C8VEC is a vector of C8's
c
c    The vector Loo norm is defined as:
c
c      C8VEC_NORM_LI = max ( 1 <= I <= N ) abs ( A(I) )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    09 February 2015
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries.
c
c    Input, double complex A(N), the vector.
c
c    Output, double precision C8VEC_NORM_LI, the norm.
c
      implicit none

      integer n

      double complex a(n)
      double precision c8vec_norm_li
      integer i
      double precision value

      value = 0.0D+00
      do i = 1, n
        value = max ( value, cdabs ( a(i) ) )
      end do

      c8vec_norm_li = value

      return
      end
      function c8vec_norm_squared ( n, a )

c*********************************************************************72
c
cc C8VEC_NORM_SQUARED returns the square of the L2 norm of a C8VEC.
c
c  Discussion:
c
c    A C8VEC is a vector of C8's.
c
c    The square of the vector L2 norm is defined as:
c
c      C8VEC_NORM_SQUARED = sum ( 1 <= I <= N ) conjg ( A(I) ) * A(I).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    22 June 2011
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in A.
c
c    Input, double complex A(N), the vector whose L2 norm is desired.
c
c    Output, double precision C8VEC_NORM_SQUARED, the L2 norm of A.
c
      implicit none

      integer n

      double complex a(n)
      integer i
      double precision c8vec_norm_squared

      c8vec_norm_squared = 0.0D+00
      do i = 1, n
        c8vec_norm_squared = c8vec_norm_squared + dconjg ( a(i) ) * a(i)
      end do

      return
      end
      subroutine c8vec_print ( n, a, title )

c*********************************************************************72
c
cc C8VEC_PRINT prints a C8VEC, with an optional title.
c
c  Discussion:
c
c    A C8VEC is a vector of C8's
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    07 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of components of the vector.
c
c    Input, double complex A(N), the vector to be printed.
c
c    Input, character*(*) TITLE, a title.
c
      implicit none

      integer n

      double complex a(n)
      integer i
      character*(*) title

      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) trim ( title )
      write ( *, '(a)' ) ' '
      do i = 1, n
        write ( *, '(2x,i8,a,1x,2g14.6)' ) i, ':', a(i)
      end do

      return
      end
      subroutine c8vec_print_part ( n, a, max_print, title )

c*********************************************************************72
c
cc C8VEC_PRINT_PART prints "part" of a C8VEC.
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
c    Input, double complex A(N), the vector to be printed.
c
c    Input, integer MAX_PRINT, the maximum number of lines
c    to print.
c
c    Input, character * ( * ) TITLE, a title.
c
      implicit none

      integer n

      double complex a(n)
      integer i
      integer max_print
      character * ( * )  title

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
          write ( *, '(2x,i8,a,1x,g14.6,2x,g14.6)' ) i, ':', a(i)
        end do

      else if ( 3 .le. max_print ) then

        do i = 1, max_print - 2
          write ( *, '(2x,i8,a,1x,g14.6,2x,g14.6)' ) i, ':', a(i)
        end do
        write ( *, '(a)' ) '  ........  ..............  ..............'
        i = n
        write ( *, '(2x,i8,a,1x,g14.6,2x,g14.6)' ) i, ':', a(i)

      else

        do i = 1, max_print - 1
          write ( *, '(2x,i8,a,1x,g14.6,2x,g14.6)' ) i, ':', a(i)
        end do
        i = max_print
        write ( *, '(2x,i8,a,1x,g14.6,2x,g14.6,2x,a)' ) i, ':', a(i), 
     &    '...more entries...'

      end if

      return
      end
      subroutine c8vec_print_some ( n, x, i_lo, i_hi, title )

c*********************************************************************72
c
cc C8VEC_PRINT_SOME prints some of a C8VEC.
c
c  Discussion:
c
c    A C8VEC is a vector of C8's.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    18 October 2006
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries of the vector.
c
c    Input, double complex X(N), the vector to be printed.
c
c    Input, integer I_LO, I_HI, the first and last entries
c    to print.
c
c    Input, character*(*) TITLE, a title.
c
      implicit none

      integer n

      integer i
      integer i_hi
      integer i_lo
      character*(*) title
      double complex x(n)

      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) trim ( title )
      write ( *, '(a)' ) ' '

      do i = max ( 1, i_lo ), min ( n, i_hi )
        write ( *, '(2x,i8,a,1x,2g14.6)' ) i, ':', x(i)
      end do

      return
      end
      subroutine c8vec_sort_a_l1 ( n, x )

c*********************************************************************72
c
cc C8VEC_SORT_A_L1 ascending sorts a C8VEC by L1 norm.
c
c  Discussion:
c
c    A C8VEC is a vector of C8's.
c
c    The L1 norm of A+Bi is abs(A) + abs(B).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    07 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the array.
c
c    Input/output, double complex X(N).
c    On input, an unsorted array.
c    On output, X has been sorted.
c
      implicit none

      integer n

      logical c8_le_l1
      integer i
      integer indx
      integer isgn
      integer j
      double complex x(n)

      if ( n .le. 1 ) then
        return
      end if

      i = 0
      indx = 0
      isgn = 0
      j = 0

10    continue

        call sort_heap_external ( n, indx, i, j, isgn )

        if ( 0 .lt. indx ) then

          call c8_swap ( x(i), x(j) )

        else if ( indx .lt. 0 ) then

          if ( c8_le_l1 ( x(i), x(j) ) ) then
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
      subroutine c8vec_sort_a_l2 ( n, x )

c*********************************************************************72
c
cc C8VEC_SORT_A_L2 ascending sorts a C8VEC by L2 norm.
c
c  Discussion:
c
c    A C8VEC is a vector of C8's.
c
c    The L2 norm of A+Bi is sqrt ( A**2 + B**2 ).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    07 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the array.
c
c    Input/output, double complex X(N).
c    On input, an unsorted array.
c    On output, X has been sorted.
c
      implicit none

      integer n

      logical c8_le_l2
      integer i
      integer indx
      integer isgn
      integer j
      double complex x(n)

      if ( n .le. 1 ) then
        return
      end if

      i = 0
      indx = 0
      isgn = 0
      j = 0

10    continue

        call sort_heap_external ( n, indx, i, j, isgn )

        if ( 0 .lt. indx ) then

          call c8_swap ( x(i), x(j) )

        else if ( indx .lt. 0 ) then

          if ( c8_le_l2 ( x(i), x(j) ) ) then
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
      subroutine c8vec_sort_a_li ( n, x )

c*********************************************************************72
c
cc C8VEC_SORT_A_LI ascending sorts a C8VEC by L-infinity norm.
c
c  Discussion:
c
c    A C8VEC is a vector of C8's.
c
c    The L infinity norm of A+Bi is max ( abs ( A ), abs ( B ) ).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    07 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of entries in the array.
c
c    Input/output, double complex X(N).
c    On input, an unsorted array.
c    On output, X has been sorted.
c
      implicit none

      integer n

      logical c8_le_li
      integer i
      integer indx
      integer isgn
      integer j
      double complex x(n)

      if ( n .le. 1 ) then
        return
      end if

      i = 0
      indx = 0
      isgn = 0
      j = 0

10    continue

        call sort_heap_external ( n, indx, i, j, isgn )

        if ( 0 .lt. indx ) then

          call c8_swap ( x(i), x(j) )

        else if ( indx .lt. 0 ) then

          if ( c8_le_li ( x(i), x(j) ) ) then
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
      subroutine c8vec_spiral ( n, m, c1, c2, c )

c*********************************************************************72
c
cc C8VEC_SPIRAL returns N points on a spiral between C1 and C2.
c
c  Discussion:
c
c    A C8VEC is a vector of C8's.
c
c    Let the polar form of C1 be ( R1, T1 ) and the polar form of C2 
c    be ( R2, T2 ) where, if necessary, we increase T2 by 2*PI so that T1 <= T2.
c    
c    Then the polar form of the I-th point C(I) is:
c
c      R(I) = ( ( N - I     ) * R1 
c             + (     I - 1 ) * R2 ) 
c              / ( N    - 1 )
c
c      T(I) = ( ( N - I     ) * T1 
c             + (     I - 1 ) * ( T2 + M * 2 * PI ) ) 
c             / ( N     - 1 )
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    12 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of points on the spiral.
c
c    Input, integer M, the number of full circuits the spiral makes.
c
c    Input, double complex C1, C2, the first and last points on the spiral.
c
c    Output, double complex C(N), the points.
c
      implicit none

      integer n

      double complex c(n)
      double complex c1
      double complex c2
      double precision c8_arg
      double precision c8_mag
      integer i
      integer m
      double precision r1
      double precision r2
      double precision ri
      double precision r8_pi
      parameter ( r8_pi = 3.141592653589793 )
      double precision t1
      double precision t2
      double precision ti

      r1 = c8_mag ( c1 )
      r2 = c8_mag ( c2 )

      t1 = c8_arg ( c1 )
      t2 = c8_arg ( c2 )

      if ( m .eq. 0 ) then

        if ( t2 .lt. t1 ) then
          t2 = t2 + 2.0D+00 * r8_pi
        end if

      else if ( 0 .lt. m ) then

        if ( t2 .lt. t1 ) then
          t2 = t2 + 2.0D+00 * r8_pi
        end if

        t2 = t2 + dble ( m ) * 2.0D+00 * r8_pi

      else if ( m .lt. 0 ) then

        if ( t1 .lt. t2 ) then
          t2 = t2 - 2.0D+00 * r8_pi
        end if

        t2 = t2 - dble ( m ) * 2.0D+00 * r8_pi

      end if

      do i = 1, n

        ri = ( dble ( n - i     ) * r1 
     &       + dble (     i - 1 ) * r2 ) 
     &       / dble ( n     - 1 )

        ti = ( dble ( n - i     ) * t1 
     &       + dble (     i - 1 ) * t2 ) 
     &       / dble ( n     - 1 )

        call polar_to_c8 ( ri, ti, c(i) )

      end do

      return
      end
      subroutine c8vec_uniform_01 ( n, seed, c )

c*********************************************************************72
c
cc C8VEC_UNIFORM_01 returns a unit pseudorandom C8VEC.
c
c  Discussion:
c
c    A C8VEC is a vector of C8's
c
c    The angles should be uniformly distributed between 0 and 2 * PI,
c    the square roots of the radius uniformly distributed between 0 and 1.
c
c    This results in a uniform distribution of values in the unit circle.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    07 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of values to compute.
c
c    Input/output, integer SEED, the "seed" value, which should NOT be 0.
c    On output, SEED has been updated.
c
c    Output, double complex C(N), the pseudorandom complex vector.
c
      implicit none

      integer n

      double complex c(n)
      integer i
      integer k
      double precision r
      double precision r8_pi
      parameter ( r8_pi = 3.141592653589793D+00 )
      integer seed
      double precision theta

      if ( seed .eq. 0 ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'C8VEC_UNIFORM_01 - Fatal errorc'
        write ( *, '(a)' ) '  Input value of SEED = 0.'
        stop
      end if

      do i = 1, n

        k = seed / 127773

        seed = 16807 * ( seed - k * 127773 ) - k * 2836

        if ( seed .lt. 0 ) then
          seed = seed + 2147483647
        end if

        r = dsqrt ( dble ( seed ) * 4.656612875D-10 )

        k = seed / 127773

        seed = 16807 * ( seed - k * 127773 ) - k * 2836

        if ( seed .lt. 0 ) then
          seed = seed + 2147483647
        end if

        theta = 2.0D+00 * r8_pi * ( dble ( seed ) * 4.656612875D-10 )

        c(i) = r * dcmplx ( dcos ( theta ), dsin ( theta ) )

      end do

      return
      end
      subroutine c8vec_unity ( n, a )

c*********************************************************************72
c
cc C8VEC_UNITY returns the N roots of unity.
c
c  Discussion:
c
c    A C8VEC is a vector of C8's
c
c    X(1:N) = exp ( 2 * PI * (0:N-1) / N )
c
c    X(1:N)^N = ( (1,0), (1,0), ..., (1,0) ).
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    07 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer N, the number of elements of A.
c
c    Output, double complex A(N), the N roots of unity.
c
      implicit none

      integer n

      double complex a(n)
      integer i
      double precision r8_pi
      parameter ( r8_pi = 3.141592653589793D+00 )
      double precision theta

      do i = 1, n
        theta = r8_pi * dble ( 2 * ( i - 1 ) ) / dble ( n )
        a(i) = dcmplx ( dcos ( theta ), dsin ( theta ) )
      end do

      return
      end
      subroutine cartesian_to_c8 ( x, y, z )

c*********************************************************************72
c
cc CARTESIAN_TO_C8 converts a Cartesian form to a C8.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    12 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision X, Y, the Cartesian form.
c
c    Output, double complex Z, the complex number.
c
      implicit none

      double precision x
      double precision y
      double complex z

      z = dcmplx ( x, y )

      return
      end
      subroutine polar_to_c8 ( r, theta, z )

c*********************************************************************72
c
cc POLAR_TO_C8 converts a polar form to a C8.
c
c  Discussion:
c
c    A C8 is a double complex value.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license. 
c
c  Modified:
c
c    12 December 2008
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, double precision R, THETA, the polar form.
c
c    Output, double complex Z, the complex number.
c
      implicit none

      double precision r
      double precision theta
      double complex z

      z = r * dcmplx ( dcos ( theta ), dsin ( theta ) )

      return
      end

C$$$      function r8_atan2 ( y, x )

C$$$c*********************************************************************72
C$$$c
C$$$cc R8_ATAN computes the inverse tangent of the ratio Y / X.
C$$$c
C$$$c  Discussion:
C$$$c
C$$$c    R8_ATAN returns an angle whose tangent is ( Y / X ), a job which
C$$$c    the built in functions ATAN and ATAN2 already do.
C$$$c
C$$$c    However:
C$$$c
C$$$c    * R8_ATAN always returns a positive angle, between 0 and 2 PI,
C$$$c      while ATAN and ATAN2 return angles in the interval [-PI/2,+PI/2]
C$$$c      and [-PI,+PI] respectively;
C$$$c
C$$$c    * R8_ATAN accounts for the signs of X and Y, (as does ATAN2).  The ATAN
C$$$c     function by contrast always returns an angle in the first or fourth
C$$$c     quadrants.
C$$$c
C$$$c  Licensing:
C$$$c
C$$$c    This code is distributed under the GNU LGPL license.
C$$$c
C$$$c  Modified:
C$$$c
C$$$c    21 April 2014
C$$$c
C$$$c  Author:
C$$$c
C$$$c    John Burkardt
C$$$c
C$$$c  Parameters:
C$$$c
C$$$c    Input, double precision Y, X, two quantities which represent the
C$$$c    tangent of an angle.  If Y is not zero, then the tangent is (Y/X).
C$$$c
C$$$c    Output, double precision R8_ATAN, an angle between 0 and 2 * PI, whose
C$$$c    tangent is (Y/X), and which lies in the appropriate quadrant so that
C$$$c    the signs of its cosine and sine match those of X and Y.
C$$$c
C$$$      implicit none

C$$$      double precision abs_x
C$$$      double precision abs_y
C$$$      double precision r8_atan2
C$$$      double precision r8_pi
C$$$      parameter ( r8_pi = 3.141592653589793D+00 )
C$$$      double precision theta
C$$$      double precision theta_0
C$$$      double precision x
C$$$      double precision y
C$$$c
C$$$c  Special cases:
C$$$c
C$$$      if ( x .eq. 0.0D+00 ) then

C$$$        if ( 0.0D+00 .lt. y ) then
C$$$          theta = r8_pi / 2.0D+00
C$$$        else if ( y .lt. 0.0D+00 ) then
C$$$          theta = 3.0D+00 * r8_pi / 2.0D+00
C$$$        else if ( y .eq. 0.0D+00 ) then
C$$$          theta = 0.0D+00
C$$$        end if

C$$$      else if ( y .eq. 0.0D+00 ) then

C$$$        if ( 0.0D+00 .lt. x ) then
C$$$          theta = 0.0D+00
C$$$        else if ( x .lt. 0.0D+00 ) then
C$$$          theta = r8_pi
C$$$        end if
C$$$c
C$$$c  We assume that ATAN2 is correct when both arguments are positive.
C$$$c
C$$$      else

C$$$        abs_y = dabs ( y )
C$$$        abs_x = dabs ( x )

C$$$        theta_0 = atan2 ( abs_y, abs_x )

C$$$        if ( 0.0D+00 .lt. x .and. 0.0D+00 .lt. y ) then
C$$$          theta = theta_0
C$$$        else if ( x .lt. 0.0D+00 .and. 0.0D+00 .lt. y ) then
C$$$          theta = r8_pi - theta_0
C$$$        else if ( x .lt. 0.0D+00 .and. y .lt. 0.0D+00 ) then
C$$$          theta = r8_pi + theta_0
C$$$        else if ( 0.0D+00 .lt. x .and. y .lt. 0.0D+00 ) then
C$$$          theta = 2.0D+00 * r8_pi - theta_0
C$$$        end if

C$$$      end if

C$$$      r8_atan = theta

C$$$      return
C$$$      end
C$$$      function r8_csqrt2 ( x )

C$$$c*********************************************************************72
C$$$c
C$$$cc R8_CSQRT returns the complex square root of an R8.
C$$$c
C$$$c  Licensing:
C$$$c
C$$$c    This code is distributed under the GNU LGPL license.
C$$$c
C$$$c  Modified:
C$$$c
C$$$c    23 August 20008
C$$$c
C$$$c  Author:
C$$$c
C$$$c    John Burkardt
C$$$c
C$$$c  Parameters:
C$$$c
C$$$c    Input, double precision X, the number whose square root is desired.
C$$$c
C$$$c    Output, double complex R8_CSQRT, the square root of X:
C$$$c
C$$$      implicit none

C$$$      double precision argument
C$$$      double precision magnitude
C$$$      double precision pi
C$$$      parameter ( pi = 3.141592653589793D+00 )
C$$$      double complex r8_csqrt
C$$$      double precision x

C$$$      if ( 0.0D+00 .lt. x ) then
C$$$        magnitude = x
C$$$        argument = 0.0D+00
C$$$      else if ( 0.0D+00 .eq. x ) then
C$$$        magnitude = 0.0D+00
C$$$        argument = 0.0D+00
C$$$      else if ( x .lt. 0.0D+00 ) then
C$$$        magnitude = -x
C$$$        argument = pi
C$$$      end if

C$$$      magnitude = sqrt ( magnitude )
C$$$      argument = argument / 2.0D+00

C$$$      r8_csqrt = magnitude
C$$$     &  * dcmplx ( cos ( argument ), sin ( argument ) )

C$$$      return
C$$$      end
C$$$      function r8_floor ( r )

C$$$c*********************************************************************72
C$$$c
C$$$cc R8_FLOOR rounds an R8 "down" (towards -infinity) to the nearest integral R8.
C$$$c
C$$$c  Example:
C$$$c
C$$$c    R     Value
C$$$c
C$$$c   -1.1  -2.0
C$$$c   -1.0  -1.0
C$$$c   -0.9  -1.0
C$$$c    0.0   0.0
C$$$c    5.0   5.0
C$$$c    5.1   5.0
C$$$c    5.9   5.0
C$$$c    6.0   6.0
C$$$c
C$$$c  Licensing:
C$$$c
C$$$c    This code is distributed under the GNU LGPL license.
C$$$c
C$$$c  Modified:
C$$$c
C$$$c    10 November 2011
C$$$c
C$$$c  Author:
C$$$c
C$$$c    John Burkardt
C$$$c
C$$$c  Parameters:
C$$$c
C$$$c    Input, double precision R, the value to be rounded down.
C$$$c
C$$$c    Output, double precision R8_FLOOR, the rounded value.
C$$$c
C$$$      implicit none

C$$$      double precision r
C$$$      double precision r8_floor
C$$$      double precision value

C$$$      value = dble ( int ( r ) )
C$$$      if ( r .lt. value ) then
C$$$        value = value - 1.0D+00
C$$$      end if

C$$$      r8_floor = value

C$$$      return
C$$$      end
C$$$      function r8_huge ( )

C$$$c*********************************************************************72
C$$$c
C$$$cc R8_HUGE returns a "huge" R8.
C$$$c
C$$$c  Discussion:
C$$$c
C$$$c    The value returned by this function is NOT required to be the
C$$$c    maximum representable R8.  This value varies from machine to machine,
C$$$c    from compiler to compiler, and may cause problems when being printed.
C$$$c    We simply want a "very large" but non-infinite number.
C$$$c
C$$$c    FORTRAN90 provides a built-in routine HUGE ( X ) that
C$$$c    can return the maximum representable number of the same datatype
C$$$c    as X, if that is what is really desired.
C$$$c
C$$$c  Licensing:
C$$$c
C$$$c    This code is distributed under the GNU LGPL license.
C$$$c
C$$$c  Modified:
C$$$c
C$$$c    13 April 2004
C$$$c
C$$$c  Author:
C$$$c
C$$$c    John Burkardt
C$$$c
C$$$c  Parameters:
C$$$c
C$$$c    Output, double precision R8_HUGE, a huge number.
C$$$c
C$$$      implicit none

C$$$      double precision r8_huge

C$$$      r8_huge = 1.0D+30

C$$$      return
C$$$      end
C$$$      function r8_log_2 ( x )

C$$$c*********************************************************************72
C$$$c
C$$$cc R8_LOG_2 returns the logarithm base 2 of an R8.
C$$$c
C$$$c  Discussion:
C$$$c
C$$$c    value = Log ( |X| ) / Log ( 2.0 )
C$$$c
C$$$c  Licensing:
C$$$c
C$$$c    This code is distributed under the GNU LGPL license.
C$$$c
C$$$c  Modified:
C$$$c
C$$$c    01 January 2007
C$$$c
C$$$c  Author:
C$$$c
C$$$c    John Burkardt
C$$$c
C$$$c  Parameters:
C$$$c
C$$$c    Input, double precision X, the number whose base 2 logarithm is desired.
C$$$c    X should not be 0.
C$$$c
C$$$c    Output, double precision R8_LOG_2, the logarithm base 2 of the absolute
C$$$c    value of X.  It should be true that |X| = 2^R8_LOG_2.
C$$$c
C$$$      implicit none

C$$$      double precision r8_huge
C$$$      double precision r8_log_2
C$$$      double precision x

C$$$      if ( x .eq. 0.0D+00 ) then
C$$$        r8_log_2 = -r8_huge ( )
C$$$      else
C$$$        r8_log_2 = log ( abs ( x ) ) / log ( 2.0D+00 )
C$$$      end if

C$$$      return
C$$$      end
C$$$      function r8_uniform_01 ( seed )

C$$$c*********************************************************************72
C$$$c
C$$$cc R8_UNIFORM_01 returns a unit pseudorandom R8.
C$$$c
C$$$c  Discussion:
C$$$c
C$$$c    This routine implements the recursion
C$$$c
C$$$c      seed = 16807 * seed mod ( 2^31 - 1 )
C$$$c      r8_uniform_01 = seed / ( 2^31 - 1 )
C$$$c
C$$$c    The integer arithmetic never requires more than 32 bits,
C$$$c    including a sign bit.
C$$$c
C$$$c    If the initial seed is 12345, then the first three computations are
C$$$c
C$$$c      Input     Output      R8_UNIFORM_01
C$$$c      SEED      SEED
C$$$c
C$$$c         12345   207482415  0.096616
C$$$c     207482415  1790989824  0.833995
C$$$c    1790989824  2035175616  0.947702
C$$$c
C$$$c  Licensing:
C$$$c
C$$$c    This code is distributed under the GNU LGPL license.
C$$$c
C$$$c  Modified:
C$$$c
C$$$c    11 August 2004
C$$$c
C$$$c  Author:
C$$$c
C$$$c    John Burkardt
C$$$c
C$$$c  Reference:
C$$$c
C$$$c    Paul Bratley, Bennett Fox, Linus Schrage,
C$$$c    A Guide to Simulation,
C$$$c    Springer Verlag, pages 201-202, 1983.
C$$$c
C$$$c    Pierre L'Ecuyer,
C$$$c    Random Number Generation,
C$$$c    in Handbook of Simulation,
C$$$c    edited by Jerry Banks,
C$$$c    Wiley Interscience, page 95, 1998.
C$$$c
C$$$c    Bennett Fox,
C$$$c    Algorithm 647:
C$$$c    Implementation and Relative Efficiency of Quasirandom
C$$$c    Sequence Generators,
C$$$c    ACM Transactions on Mathematical Software,
C$$$c    Volume 12, Number 4, pages 362-376, 1986.
C$$$c
C$$$c    Peter Lewis, Allen Goodman, James Miller,
C$$$c    A Pseudo-Random Number Generator for the System/360,
C$$$c    IBM Systems Journal,
C$$$c    Volume 8, pages 136-143, 1969.
C$$$c
C$$$c  Parameters:
C$$$c
C$$$c    Input/output, integer SEED, the "seed" value, which should NOT be 0.
C$$$c    On output, SEED has been updated.
C$$$c
C$$$c    Output, double precision R8_UNIFORM_01, a new pseudorandom variate,
C$$$c    strictly between 0 and 1.
C$$$c
C$$$      implicit none

C$$$      double precision r8_uniform_01
C$$$      integer k
C$$$      integer seed

C$$$      if ( seed .eq. 0 ) then
C$$$        write ( *, '(a)' ) ' '
C$$$        write ( *, '(a)' ) 'R8_UNIFORM_01 - Fatal errorc'
C$$$        write ( *, '(a)' ) '  Input value of SEED = 0.'
C$$$        stop
C$$$      end if

C$$$      k = seed / 127773

C$$$      seed = 16807 * ( seed - k * 127773 ) - k * 2836

C$$$      if ( seed .lt. 0 ) then
C$$$        seed = seed + 2147483647
C$$$      end if
C$$$c
C$$$c  Although SEED can be represented exactly as a 32 bit integer,
C$$$c  it generally cannot be represented exactly as a 32 bit real numberc
C$$$c
C$$$      r8_uniform_01 = dble ( seed ) * 4.656612875D-10

C$$$      return
C$$$      end
C$$$      subroutine r8poly2_root ( a, b, c, r1, r2 )

C$$$c*********************************************************************72
C$$$c
C$$$cc R8POLY2_ROOT returns the two roots of a quadratic polynomial.
C$$$c
C$$$c  Discussion:
C$$$c
C$$$c    The polynomial has the form:
C$$$c
C$$$c      A * X * X + B * X + C = 0
C$$$c
C$$$c  Licensing:
C$$$c
C$$$c    This code is distributed under the GNU LGPL license.
C$$$c
C$$$c  Modified:
C$$$c
C$$$c    01 March 2013
C$$$c
C$$$c  Author:
C$$$c
C$$$c    John Burkardt
C$$$c
C$$$c  Parameters:
C$$$c
C$$$c    Input, double precision A, B, C, the coefficients of the polynomial.
C$$$c    A must not be zero.
C$$$c
C$$$c    Output, double complex R1, R2, the roots of the polynomial, which
C$$$c    might be real and distinct, real and equal, or complex conjugates.
C$$$c
C$$$      implicit none

C$$$      double precision a
C$$$      double precision b
C$$$      double precision c
C$$$      double complex disc
C$$$      double complex q
C$$$      double complex r1
C$$$      double complex r2

C$$$      if ( a .eq. 0.0D+00 ) then
C$$$        write ( *, '(a)' ) ' '
C$$$        write ( *, '(a)' ) 'R8POLY2_ROOT - Fatal error!'
C$$$        write ( *, '(a)' ) '  The coefficient A is zero.'
C$$$        stop
C$$$      end if

C$$$      disc = b * b - 4.0D+00 * a * c
C$$$      q = -0.5D+00 * ( b + sign ( 1.0D+00, b ) * sqrt ( disc ) )
C$$$      r1 = q / a
C$$$      r2 = c / q

C$$$      return
C$$$      end
C$$$      subroutine r8poly3_root ( a, b, c, d, r1, r2, r3 )

C$$$c*********************************************************************72
C$$$c
C$$$cc R8POLY3_ROOT returns the three roots of a cubic polynomial.
C$$$c
C$$$c  Discussion:
C$$$c
C$$$c    The polynomial has the form
C$$$c
C$$$c      A * X^3 + B * X^2 + C * X + D = 0
C$$$c
C$$$c  Licensing:
C$$$c
C$$$c    This code is distributed under the GNU LGPL license.
C$$$c
C$$$c  Modified:
C$$$c
C$$$c    01 March 2013
C$$$c
C$$$c  Parameters:
C$$$c
C$$$c    Input, double precision A, B, C, D, the coefficients of the polynomial.
C$$$c    A must not be zero.
C$$$c
C$$$c    Output, double complex R1, R2, R3, the roots of the polynomial, which
C$$$c    will include at least one real root.
C$$$c
C$$$      implicit none

C$$$      double precision a
C$$$      double precision b
C$$$      double precision c
C$$$      double precision d
C$$$      double complex i
C$$$      double complex one
C$$$      double precision pi
C$$$      parameter ( pi = 3.141592653589793D+00 )
C$$$      double precision q
C$$$      double precision r
C$$$      double complex r1
C$$$      double complex r2
C$$$      double complex r3
C$$$      double precision s1
C$$$      double precision s2
C$$$      double precision temp
C$$$      double precision theta

C$$$      if ( a .eq. 0.0D+00 ) then
C$$$        write ( *, '(a)' ) ' '
C$$$        write ( *, '(a)' ) 'R8POLY3_ROOT - Fatal error!'
C$$$        write ( *, '(a)' ) '  A must not be zero!'
C$$$        stop
C$$$      end if

C$$$      one = dcmplx ( 1.0d+00, 0.0D+00 )
C$$$      i = cdsqrt ( - one )

C$$$      q = ( ( b / a )**2 - 3.0D+00 * ( c / a ) ) / 9.0D+00

C$$$      r = ( 2.0D+00 * ( b / a )**3 - 9.0D+00 * ( b / a ) * ( c / a ) 
C$$$     &    + 27.0D+00 * ( d / a ) ) / 54.0D+00

C$$$      if ( r * r .lt. q * q * q ) then

C$$$        theta = acos ( r / dsqrt ( q**3 ) )
C$$$        r1 = -2.0D+00 * dsqrt ( q ) 
C$$$     &    * dcos (   theta                  / 3.0D+00 )
C$$$        r2 = -2.0D+00 * dsqrt ( q ) 
C$$$     &    * dcos ( ( theta + 2.0D+00 * pi ) / 3.0D+00 )
C$$$        r3 = -2.0D+00 * dsqrt ( q ) 
C$$$     &    * dcos ( ( theta + 4.0D+00 * pi ) / 3.0D+00 )

C$$$      else if ( q * q * q .le. r * r ) then

C$$$        temp = -r + dsqrt ( r**2 - q**3 )
C$$$        s1 = dsign ( 1.0D+00, temp ) 
C$$$     &    * ( dabs ( temp ) )**(1.0D+00/3.0D+00)

C$$$        temp = -r - dsqrt ( r**2 - q**3 )
C$$$        s2 = dsign ( 1.0D+00, temp ) 
C$$$     &    * ( dabs ( temp ) )**(1.0D+00/3.0D+00)

C$$$        r1 = s1 + s2
C$$$        r2 = -0.5D+00 * ( s1 + s2 ) 
C$$$     &    + i * 0.5D+00 * dsqrt ( 3.0D+00 ) * ( s1 - s2 )
C$$$        r3 = -0.5D+00 * ( s1 + s2 ) 
C$$$     &    - i * 0.5D+00 * dsqrt ( 3.0D+00 ) * ( s1 - s2 )

C$$$      end if

C$$$      r1 = r1 - b / ( 3.0D+00 * a )
C$$$      r2 = r2 - b / ( 3.0D+00 * a )
C$$$      r3 = r3 - b / ( 3.0D+00 * a )

C$$$      return
C$$$      end
C$$$      subroutine r8poly4_root ( a, b, c, d, e, r1, r2, r3, r4 )

C$$$c*********************************************************************72
C$$$c
C$$$cc R8POLY4_ROOT returns the four roots of a quartic polynomial.
C$$$c
C$$$c  Discussion:
C$$$c
C$$$c    The polynomial has the form:
C$$$c
C$$$c      A * X^4 + B * X^3 + C * X^2 + D * X + E = 0
C$$$c
C$$$c  Licensing:
C$$$c
C$$$c    This code is distributed under the GNU LGPL license.
C$$$c
C$$$c  Modified:
C$$$c
C$$$c    01 March 2013
C$$$c
C$$$c  Parameters:
C$$$c
C$$$c    Input, double precision A, B, C, D, the coefficients of the polynomial.
C$$$c    A must not be zero.
C$$$c
C$$$c    Output, double complex R1, R2, R3, R4, the roots of the polynomial.
C$$$c
C$$$      implicit none

C$$$      double precision a
C$$$      double precision a3
C$$$      double precision a4
C$$$      double precision b
C$$$      double precision b3
C$$$      double precision b4
C$$$      double precision c
C$$$      double precision c3
C$$$      double precision c4
C$$$      double precision d
C$$$      double precision d3
C$$$      double precision d4
C$$$      double precision e
C$$$      double complex p
C$$$      double complex q
C$$$      double complex r
C$$$      double complex r1
C$$$      double complex r2
C$$$      double complex r3
C$$$      double complex r4
C$$$      double complex zero

C$$$      zero = dcmplx ( 0.0D+00, 0.0D+00 )

C$$$      if ( a .eq. 0.0D+00 ) then
C$$$        write ( *, '(a)' ) ' '
C$$$        write ( *, '(a)' ) 'R8POLY4_ROOT - Fatal error!'
C$$$        write ( *, '(a)') '  A must not be zero!'
C$$$        stop
C$$$      end if

C$$$      a4 = b / a
C$$$      b4 = c / a
C$$$      c4 = d / a
C$$$      d4 = e / a
C$$$c
C$$$c  Set the coefficients of the resolvent cubic equation.
C$$$c
C$$$      a3 = 1.0D+00
C$$$      b3 = -b4
C$$$      c3 = a4 * c4 - 4.0D+00 * d4
C$$$      d3 = -a4 * a4 * d4 + 4.0D+00 * b4 * d4 - c4 * c4
C$$$c
C$$$c  Find the roots of the resolvent cubic.
C$$$c
C$$$      call r8poly3_root ( a3, b3, c3, d3, r1, r2, r3 )
C$$$c
C$$$c  Choose one root of the cubic, here R1.
C$$$c
C$$$c  Set R = sqrt ( 0.25D+00 * A4**2 - B4 + R1 )
C$$$c
C$$$      r = cdsqrt ( 0.25D+00 * a4**2 - b4 + r1 )

C$$$      if ( r .ne. zero ) then

C$$$        p = cdsqrt ( 0.75D+00 * a4**2 - r**2 - 2.0D+00 * b4 
C$$$     &    + 0.25D+00 * ( 4.0D+00 * a4 * b4 - 8.0D+00 * c4 - a4**3 ) 
C$$$     & / r )

C$$$        q = cdsqrt ( 0.75D+00 * a4**2 - r**2 - 2.0D+00 * b4 
C$$$     &    - 0.25D+00 * ( 4.0D+00 * a4 * b4 - 8.0D+00 * c4 - a4**3 ) 
C$$$     &    / r )

C$$$      else

C$$$        p = cdsqrt ( 0.75D+00 * a4**2 - 2.0D+00 * b4 
C$$$     &    + 2.0D+00 * cdsqrt ( r1**2 - 4.0D+00 * d4 ) )

C$$$        q = cdsqrt ( 0.75D+00 * a4**2 - 2.0D+00 * b4 
C$$$     &    - 2.0D+00 * cdsqrt ( r1**2 - 4.0D+00 * d4 ) )

C$$$      end if
C$$$c
C$$$c  Set the roots.
C$$$c
C$$$      r1 = -0.25D+00 * a4 + 0.5D+00 * r + 0.5D+00 * p
C$$$      r2 = -0.25D+00 * a4 + 0.5D+00 * r - 0.5D+00 * p
C$$$      r3 = -0.25D+00 * a4 - 0.5D+00 * r + 0.5D+00 * q
C$$$      r4 = -0.25D+00 * a4 - 0.5D+00 * r - 0.5D+00 * q

C$$$      return
C$$$      end
C$$$      subroutine sort_heap_external ( n, indx, i, j, isgn )

C$$$c*********************************************************************72
C$$$c
C$$$cc SORT_HEAP_EXTERNAL externally sorts a list of items into ascending order.
C$$$c
C$$$c  Discussion:
C$$$c
C$$$c    The actual list of data is not passed to the routine.  Hence this
C$$$c    routine may be used to sort integers, reals, numbers, names,
C$$$c    dates, shoe sizes, and so on.  After each call, the routine asks
C$$$c    the user to compare or interchange two items, until a special
C$$$c    return value signals that the sorting is completed.
C$$$c
C$$$c  Licensing:
C$$$c
C$$$c    This code is distributed under the GNU LGPL license.
C$$$c
C$$$c  Modified:
C$$$c
C$$$c    25 January 2007
C$$$c
C$$$c  Author:
C$$$c
C$$$c    Original FORTRAN77 version by Albert Nijenhuis, Herbert Wilf
C$$$c    This FORTRAN77 version by John Burkardt
C$$$c
C$$$c  Reference:
C$$$c
C$$$c    Albert Nijenhuis, Herbert Wilf,
C$$$c    Combinatorial Algorithms for Computers and Calculators,
C$$$c    Second Edition,
C$$$c    Academic Press, 1978,
C$$$c    ISBN: 0-12-519260-6,
C$$$c    LC: QA164.N54.
C$$$c
C$$$c  Parameters:
C$$$c
C$$$c    Input, integer N, the number of items to be sorted.
C$$$c
C$$$c    Input/output, integer INDX, the main communication signal.
C$$$c
C$$$c    The user must set INDX to 0 before the first call.
C$$$c    Thereafter, the user should not change the value of INDX until
C$$$c    the sorting is done.
C$$$c
C$$$c    On return, if INDX is
C$$$c
C$$$c      greater than 0,
C$$$c      * interchange items I and J;
C$$$c      * call again.
C$$$c
C$$$c      less than 0,
C$$$c      * compare items I and J;
C$$$c      * set ISGN = -1 if I .lt. J, ISGN = +1 if J .lt. I;
C$$$c      * call again.
C$$$c
C$$$c      equal to 0, the sorting is done.
C$$$c
C$$$c    Output, integer I, J, the indices of two items.
C$$$c    On return with INDX positive, elements I and J should be interchanged.
C$$$c    On return with INDX negative, elements I and J should be compared, and
C$$$c    the result reported in ISGN on the next call.
C$$$c
C$$$c    Input, integer ISGN, results of comparison of elements I and J.
C$$$c    (Used only when the previous call returned INDX less than 0).
C$$$c    ISGN .le. 0 means I is less than or equal to J;
C$$$c    0 .le. ISGN means I is greater than or equal to J.
C$$$c
C$$$      implicit none

C$$$      integer i
C$$$      integer i_save
C$$$      integer indx
C$$$      integer isgn
C$$$      integer j
C$$$      integer j_save
C$$$      integer k
C$$$      integer k1
C$$$      integer n
C$$$      integer n1

C$$$      save i_save
C$$$      save j_save
C$$$      save k
C$$$      save k1
C$$$      save n1

C$$$      data i_save / 0 /
C$$$      data j_save / 0 /
C$$$      data k / 0 /
C$$$      data k1 / 0 /
C$$$      data n1 / 0 /
C$$$c
C$$$c  INDX = 0: This is the first call.
C$$$c
C$$$      if ( indx .eq. 0 ) then

C$$$        i_save = 0
C$$$        j_save = 0
C$$$        k = n / 2
C$$$        k1 = k
C$$$        n1 = n
C$$$c
C$$$c  INDX .lt. 0: The user is returning the results of a comparison.
C$$$c
C$$$      else if ( indx .lt. 0 ) then

C$$$        if ( indx .eq. -2 ) then

C$$$          if ( isgn .lt. 0 ) then
C$$$            i_save = i_save + 1
C$$$          end if

C$$$          j_save = k1
C$$$          k1 = i_save
C$$$          indx = -1
C$$$          i = i_save
C$$$          j = j_save
C$$$          return

C$$$        end if

C$$$        if ( 0 .lt. isgn ) then
C$$$          indx = 2
C$$$          i = i_save
C$$$          j = j_save
C$$$          return
C$$$        end if

C$$$        if ( k .le. 1 ) then

C$$$          if ( n1 .eq. 1 ) then
C$$$            i_save = 0
C$$$            j_save = 0
C$$$            indx = 0
C$$$          else
C$$$            i_save = n1
C$$$            n1 = n1 - 1
C$$$            j_save = 1
C$$$            indx = 1
C$$$          end if

C$$$          i = i_save
C$$$          j = j_save
C$$$          return

C$$$        end if

C$$$        k = k - 1
C$$$        k1 = k
C$$$c
C$$$c  0 .lt. INDX, the user was asked to make an interchange.
C$$$c
C$$$      else if ( indx .eq. 1 ) then

C$$$        k1 = k

C$$$      end if

C$$$10    continue

C$$$        i_save = 2 * k1

C$$$        if ( i_save .eq. n1 ) then
C$$$          j_save = k1
C$$$          k1 = i_save
C$$$          indx = -1
C$$$          i = i_save
C$$$          j = j_save
C$$$          return
C$$$        else if ( i_save .le. n1 ) then
C$$$          j_save = i_save + 1
C$$$          indx = -2
C$$$          i = i_save
C$$$          j = j_save
C$$$          return
C$$$        end if

C$$$        if ( k .le. 1 ) then
C$$$          go to 20
C$$$        end if

C$$$        k = k - 1
C$$$        k1 = k

C$$$      go to 10

C$$$20    continue

C$$$      if ( n1 .eq. 1 ) then
C$$$        i_save = 0
C$$$        j_save = 0
C$$$        indx = 0
C$$$        i = i_save
C$$$        j = j_save
C$$$      else
C$$$        i_save = n1
C$$$        n1 = n1 - 1
C$$$        j_save = 1
C$$$        indx = 1
C$$$        i = i_save
C$$$        j = j_save
C$$$      end if

C$$$      return
C$$$      end
C$$$      subroutine timestamp ( )

C$$$c*********************************************************************72
C$$$c
C$$$cc TIMESTAMP prints out the current YMDHMS date as a timestamp.
C$$$c
C$$$c  Discussion:
C$$$c
C$$$c    This FORTRAN77 version is made available for cases where the
C$$$c    FORTRAN90 version cannot be used.
C$$$c
C$$$c  Licensing:
C$$$c
C$$$c    This code is distributed under the GNU LGPL license.
C$$$c
C$$$c  Modified:
C$$$c
C$$$c    12 January 2007
C$$$c
C$$$c  Author:
C$$$c
C$$$c    John Burkardt
C$$$c
C$$$c  Parameters:
C$$$c
C$$$c    None
C$$$c
C$$$      implicit none

C$$$      character * ( 8 ) ampm
C$$$      integer d
C$$$      character * ( 8 ) date
C$$$      integer h
C$$$      integer m
C$$$      integer mm
C$$$      character * ( 9 ) month(12)
C$$$      integer n
C$$$      integer s
C$$$      character * ( 10 ) time
C$$$      integer y

C$$$      save month

C$$$      data month /
C$$$     &  'January  ', 'February ', 'March    ', 'April    ', 
C$$$     &  'May      ', 'June     ', 'July     ', 'August   ', 
C$$$     &  'September', 'October  ', 'November ', 'December ' /

C$$$      call date_and_time ( date, time )

C$$$      read ( date, '(i4,i2,i2)' ) y, m, d
C$$$      read ( time, '(i2,i2,i2,1x,i3)' ) h, n, s, mm

C$$$      if ( h .lt. 12 ) then
C$$$        ampm = 'AM'
C$$$      else if ( h .eq. 12 ) then
C$$$        if ( n .eq. 0 .and. s .eq. 0 ) then
C$$$          ampm = 'Noon'
C$$$        else
C$$$          ampm = 'PM'
C$$$        end if
C$$$      else
C$$$        h = h - 12
C$$$        if ( h .lt. 12 ) then
C$$$          ampm = 'PM'
C$$$        else if ( h .eq. 12 ) then
C$$$          if ( n .eq. 0 .and. s .eq. 0 ) then
C$$$            ampm = 'Midnight'
C$$$          else
C$$$            ampm = 'AM'
C$$$          end if
C$$$        end if
C$$$      end if

C$$$      write ( *, 
C$$$     &  '(i2,1x,a,1x,i4,2x,i2,a1,i2.2,a1,i2.2,a1,i3.3,1x,a)' ) 
C$$$     &  d, month(m), y, h, ':', n, ':', s, '.', mm, ampm

C$$$      return
C$$$      end
