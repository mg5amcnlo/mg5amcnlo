      subroutine c8mat_expm1 ( n, a, e )

c*********************************************************************72
c
cc C8MAT_EXPM1 is essentially MATLAB's built-in matrix exponential algorithm.
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
c    Cleve Moler, Charles Van Loan
c
c  Reference:
c
c    Cleve Moler, Charles VanLoan,
c    Nineteen Dubious Ways to Compute the Exponential of a Matrix,
c    Twenty-Five Years Later,
c    SIAM Review,
c    Volume 45, Number 1, March 2003, pages 3-49.
c
c  Parameters:
c
c    Input, integer N, the dimension of the matrix.
c
c    Input, double complex A(N,N), the matrix.
c
c    Output, double complex E(N,N), the estimate for exp(A).
c
      implicit none

      integer n

      double complex a(n,n)
      double complex a2(n,n)
      double precision a_norm
      double precision c
      double precision c8mat_norm_li
      double complex d(n,n)
      double complex e(n,n)
      integer ee
      integer k
      double precision one
      parameter ( one = 1.0D+00 )
      logical p
      integer q
      parameter ( q = 6 )
      double precision r8_log_2 
      integer s
      double precision t
      double complex x(n,n)

      call c8mat_copy ( n, n, a, a2 )

      a_norm = c8mat_norm_li ( n, n, a2 )

      ee = int ( r8_log_2 ( a_norm ) ) + 1
   
      s = max ( 0, ee + 1 )

      t = 1.0D+00 / 2.0D+00**s

      call c8mat_scale_r8 ( n, n, t, a2 )

      call c8mat_copy ( n, n, a2, x )

      c = 0.5D+00

      call c8mat_identity ( n, e )

      call c8mat_add_r8 ( n, n, one, e, c, a2, e )

      call c8mat_identity ( n, d )

      call c8mat_add_r8 ( n, n, one, d, -c, a2, d )

      p = .true.

      do k = 2, q

        c = c * dble ( q - k + 1 ) / dble ( k * ( 2 * q - k + 1 ) )

        call c8mat_mm ( n, n, n, a2, x, x )

        call c8mat_add_r8 ( n, n, c, x, one, e, e )

        if ( p ) then
          call c8mat_add_r8 ( n, n, c, x, one, d, d )
        else
          call c8mat_add_r8 ( n, n, -c, x, one, d, d )
        end if

        p = .not. p

      end do
c
c  E -> inverse(D) * E
c
      call c8mat_minvm ( n, n, d, e, e )
c
c  E -> E^(2*S)
c
      do k = 1, s
        call c8mat_mm ( n, n, n, e, e, e )
      end do

      return
      end
      subroutine r8mat_expm1 ( n, a, e )

c*********************************************************************72
c
cc R8MAT_EXPM1 is essentially MATLAB's built-in matrix exponential algorithm.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    30 November 2011
c
c  Author:
c
c    Cleve Moler, Charles Van Loan
c
c  Reference:
c
c    Cleve Moler, Charles VanLoan,
c    Nineteen Dubious Ways to Compute the Exponential of a Matrix,
c    Twenty-Five Years Later,
c    SIAM Review,
c    Volume 45, Number 1, March 2003, pages 3-49.
c
c  Parameters:
c
c    Input, integer N, the dimension of the matrix.
c
c    Input, double precision A(N,N), the matrix.
c
c    Output, double precision E(N,N), the estimate for exp(A).
c
      implicit none

      integer n

      double precision a(n,n)
      double precision a2(n,n)
      double precision a_norm
      double precision c
      double precision d(n,n)
      double precision e(n,n)
      integer ee
      integer k
      double precision one
      parameter ( one = 1.0D+00 )
      logical p
      integer q
      parameter ( q = 6 )
      double precision r8_log_2
      double precision r8mat_norm_li
      integer s
      double precision t
      double precision x(n,n)

      call r8mat_copy ( n, n, a, a2 )

      a_norm = r8mat_norm_li ( n, n, a2 )

      ee = int ( r8_log_2 ( a_norm ) ) + 1
   
      s = max ( 0, ee + 1 )

      t = 1.0D+00 / 2.0D+00**s

      call r8mat_scale ( n, n, t, a2 )

      call r8mat_copy ( n, n, a2, x )

      c = 0.5D+00

      call r8mat_identity ( n, e )

      call r8mat_add ( n, n, one, e, c, a2, e )

      call r8mat_identity ( n, d )

      call r8mat_add ( n, n, one, d, -c, a2, d )

      p = .true.

      do k = 2, q

        c = c * dble ( q - k + 1 ) / dble ( k * ( 2 * q - k + 1 ) )

        call r8mat_mm ( n, n, n, a2, x, x )

        call r8mat_add ( n, n, c, x, one, e, e )

        if ( p ) then
          call r8mat_add ( n, n, c, x, one, d, d )
        else
          call r8mat_add ( n, n, -c, x, one, d, d )
        end if

        p = .not. p

      end do
c
c  E -> inverse(D) * E
c
      call r8mat_minvm ( n, n, d, e, e )
c
c  E -> E^(2*S)
c
      do k = 1, s
        call r8mat_mm ( n, n, n, e, e, e )
      end do

      return
      end
      subroutine r8mat_expm2 ( n, a, e )

c*********************************************************************72
c
cc R8MAT_EXPM2 uses the Taylor series for the matrix exponential.
c
c  Discussion:
c
c    Formally,
c
c      exp ( A ) = I + A + 1/2 A^2 + 1/3! A^3 + ...
c
c    This function sums the series until a tolerance is satisfied.
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
c    Cleve Moler, Charles Van Loan
c
c  Reference:
c
c    Cleve Moler, Charles VanLoan,
c    Nineteen Dubious Ways to Compute the Exponential of a Matrix,
c    Twenty-Five Years Later,
c    SIAM Review,
c    Volume 45, Number 1, March 2003, pages 3-49.
c
c  Parameters:
c
c    Input, integer N, the dimension of the matrix.
c
c    Input, double precision A(N,N), the matrix.
c
c    Output, double precision E(N,N), the estimate for exp(A).
c
      implicit none

      integer n

      double precision a(n,n)
      double precision e(n,n)
      double precision f(n,n)
      integer k
      double precision one
      parameter ( one = 1.0D+00 )
      logical r8mat_insignificant
      double precision s

      call r8mat_zero ( n, n, e )

      call r8mat_identity ( n, f )

      k = 1

10    continue

        if ( r8mat_insignificant ( n, n, e, f ) ) then
          go to 20
        end if

        call r8mat_add ( n, n, one, e, one, f, e )

        call r8mat_mm ( n, n, n, a, f, f )

        s = 1.0D+00 / dble ( k )

        call r8mat_scale ( n, n, s, f )
 
        k = k + 1

      go to 10

20    continue

      return
      end
      subroutine r8mat_expm3 ( n, a, e )

c*********************************************************************72
c
cc R8MAT_EXPM3 approximates the matrix exponential using an eigenvalue approach.
c
c  Discussion:
c
c    exp(A) = V * D * V
c
c    where V is the matrix of eigenvectors of A, and D is the diagonal matrix
c    whose i-th diagonal entry is exp(lambda(i)), for lambda(i) an eigenvalue
c    of A.
c
c    This function is accurate for matrices which are symmetric, orthogonal,
c    or normal.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    30 November 2011
c
c  Author:
c
c    Cleve Moler, Charles Van Loan
c
c  Reference:
c
c    Cleve Moler, Charles VanLoan,
c    Nineteen Dubious Ways to Compute the Exponential of a Matrix,
c    Twenty-Five Years Later,
c    SIAM Review,
c    Volume 45, Number 1, March 2003, pages 3-49.
c
c  Parameters:
c
c    Input, integer N, the dimension of the matrix.
c
c    Input, double precision A(N,N), the matrix.
c
c    Output, double precision E(N,N), the estimate for exp(A).
c
c     [ V, D ] = eig ( A );
c     E = V * diag ( exp ( diag ( D ) ) ) / V;
      return
      end
