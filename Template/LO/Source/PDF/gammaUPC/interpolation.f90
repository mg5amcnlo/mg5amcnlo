MODULE interpolation
  IMPLICIT NONE
  PRIVATE
  ! 1D
  PUBLIC::POLYNOMINAL_INTERPOLATE,SPLINE_INTERPOLATE
  ! 2D
  PUBLIC::lagrange_interp_2d,pwl_interp_2d
  ! ND
  PUBLIC::lagrange_interp_nd_value,lagrange_interp_nd_value2
CONTAINS
  ! interpolation with polynominals
  SUBROUTINE POLYNOMINAL_INTERPOLATE(XA,YA,N,X,Y,DY)
    IMPLICIT NONE
    INTEGER,PARAMETER::NMAX=500
    REAL(KIND(1d0)),INTENT(IN)::X
    REAL(KIND(1d0)),INTENT(OUT)::Y,DY
    INTEGER,INTENT(IN)::N
    REAL(KIND(1d0)),DIMENSION(N),INTENT(IN)::XA,YA
    REAL(KIND(1d0)),DIMENSION(NMAX)::C,D
    INTEGER::NS,I,M
    REAL(KIND(1d0))::DIF,DIFT,HO,HP,W,DEN
    NS=1
    DIF=DABS(X-XA(1))
    DO I=1,N
       DIFT=DABS(X-XA(I))
       IF(DIFT.LT.DIF)THEN
          NS=I
          DIF=DIFT
       ENDIF
       C(I)=YA(I)
       D(I)=YA(I)
    ENDDO
    Y=YA(NS)
    NS=NS-1
    DO M=1,N-1
       DO I=1,N-M
          HO=XA(I)-X
          HP=XA(I+M)-X
          W=C(I+1)-D(I)
          DEN=HO-HP
          DEN=W/DEN
          D(I)=HP*DEN
          C(I)=HO*DEN
       ENDDO
       IF(2*NS.LT.N-M)THEN
          DY=C(NS+1)
       ELSE
          DY=D(NS)
          NS=NS-1
       ENDIF
       Y=Y+DY
    ENDDO
    RETURN
  END SUBROUTINE POLYNOMINAL_INTERPOLATE

  SUBROUTINE SPLINE_INTERPOLATE(XI,YI,N,X,Y)
    !====================================================================
    ! Spline interpolation
    ! Comments: values of function f(x) are calculated in n base points
    ! then: spline coefficients are computed
    !       spline interpolation is computed in 2n-1 points, 
    !       a difference sum|f(u)-ispline(u)| 
    !====================================================================
    IMPLICIT NONE
    INTEGER,INTENT(IN)::N ! base points for interpolation
    REAL(KIND(1d0)),DIMENSION(N),INTENT(IN)::XI,YI
    REAL(KIND(1d0)),DIMENSION(N)::b,c,d
    REAL(KIND(1d0)),INTENT(IN)::x
    REAL(KIND(1d0)),INTENT(OUT)::y
    REAL(KIND(1d0))::error,errav
    INTEGER::i
    ! call spline to calculate spline coefficients
    CALL SPLINE(XI,YI,b,c,d,N)
    ! interpolation at ninit points
    Y=ISPLINE(X,XI,YI,b,c,d,N)
    RETURN
  END SUBROUTINE SPLINE_INTERPOLATE

  subroutine spline (x, y, b, c, d, n)
    !======================================================================
    !  Calculate the coefficients b(i), c(i), and d(i), i=1,2,...,n
    !  for cubic spline interpolation
    !  s(x) = y(i) + b(i)*(x-x(i)) + c(i)*(x-x(i))**2 + d(i)*(x-x(i))**3
    !  for  x(i) <= x <= x(i+1)
    !  Alex G: January 2010
    !----------------------------------------------------------------------
    !  input..
    !  x = the arrays of data abscissas (in strictly increasing order)
    !  y = the arrays of data ordinates
    !  n = size of the arrays xi() and yi() (n>=2)
    !  output..
    !  b, c, d  = arrays of spline coefficients
    !  comments ...
    !  spline.f90 program is based on fortran version of program spline.f
    !  the accompanying function fspline can be used for interpolation
    !======================================================================
    implicit none
    integer n
    double precision x(n), y(n), b(n), c(n), d(n)
    integer i, j, gap
    double precision h
    
    gap = n-1
    ! check input
    if ( n < 2 ) return
    if ( n < 3 ) then
       b(1) = (y(2)-y(1))/(x(2)-x(1))   ! linear interpolation
       c(1) = 0.
       d(1) = 0.
       b(2) = b(1)
       c(2) = 0.
       d(2) = 0.
       return
    end if
    !
    ! step 1: preparation
    !
    d(1) = x(2) - x(1)
    c(2) = (y(2) - y(1))/d(1)
    do i = 2, gap
       d(i) = x(i+1) - x(i)
       b(i) = 2.0*(d(i-1) + d(i))
       c(i+1) = (y(i+1) - y(i))/d(i)
       c(i) = c(i+1) - c(i)
    end do
    !
    ! step 2: end conditions 
    !
    b(1) = -d(1)
    b(n) = -d(n-1)
    c(1) = 0.0
    c(n) = 0.0
    if(n /= 3) then
       c(1) = c(3)/(x(4)-x(2)) - c(2)/(x(3)-x(1))
       c(n) = c(n-1)/(x(n)-x(n-2)) - c(n-2)/(x(n-1)-x(n-3))
       c(1) = c(1)*d(1)**2/(x(4)-x(1))
       c(n) = -c(n)*d(n-1)**2/(x(n)-x(n-3))
    end if
    !
    ! step 3: forward elimination 
    !
    do i = 2, n
       h = d(i-1)/b(i-1)
       b(i) = b(i) - h*d(i-1)
       c(i) = c(i) - h*c(i-1)
    end do
    !
    ! step 4: back substitution
    !
    c(n) = c(n)/b(n)
    do j = 1, gap
       i = n-j
       c(i) = (c(i) - d(i)*c(i+1))/b(i)
    end do
    !
    ! step 5: compute spline coefficients
    !
    b(n) = (y(n) - y(gap))/d(gap) + d(gap)*(c(gap) + 2.0*c(n))
    do i = 1, gap
       b(i) = (y(i+1) - y(i))/d(i) - d(i)*(c(i+1) + 2.0*c(i))
       d(i) = (c(i+1) - c(i))/d(i)
       c(i) = 3.*c(i)
    end do
    c(n) = 3.0*c(n)
    d(n) = d(n-1)
  end subroutine spline

  function ispline(u, x, y, b, c, d, n)
    !======================================================================
    ! function ispline evaluates the cubic spline interpolation at point z
    ! ispline = y(i)+b(i)*(u-x(i))+c(i)*(u-x(i))**2+d(i)*(u-x(i))**3
    ! where  x(i) <= u <= x(i+1)
    !----------------------------------------------------------------------
    ! input..
    ! u       = the abscissa at which the spline is to be evaluated
    ! x, y    = the arrays of given data points
    ! b, c, d = arrays of spline coefficients computed by spline
    ! n       = the number of data points
    ! output:
    ! ispline = interpolated value at point u
    !=======================================================================
    implicit none
    double precision ispline
    integer n
    double precision  u, x(n), y(n), b(n), c(n), d(n)
    integer i, j, k
    double precision dx
    
    ! if u is ouside the x() interval take a boundary value (left or right)
    if(u <= x(1)) then
       ispline = y(1)
       return
    end if
    if(u >= x(n)) then
       ispline = y(n)
       return
    end if
    
    !*
    !  binary search for for i, such that x(i) <= u <= x(i+1)
    !*
    i = 1
    j = n+1
    do while (j > i+1)
       k = (i+j)/2
       if(u < x(k)) then
          j=k
       else
          i=k
       end if
    end do
    !*
    !  evaluate spline interpolation
    !*
    dx = u - x(i)
    ispline = y(i) + dx*(b(i) + dx*(c(i) + dx*d(i)))
  end function ispline

  ! the following interpolation subroutines are from 
  ! https://people.sc.fsu.edu/~jburkardt/f_src/lagrange_interp_2d/lagrange_interp_2d.html
  subroutine lagrange_basis_function_1d(mx,xd,i,xi,yi)
    !*****************************************************************************80
    !
    !! LAGRANGE_BASIS_FUNCTION_1D evaluates one 1D Lagrange basis function.
    !
    !  Licensing:
    !
    !    This code is distributed under the GNU LGPL license.
    !
    !  Modified:
    !
    !    13 September 2012
    !
    !  Author:
    !
    !    John Burkardt
    !
    !  Parameters:
    !
    !    Input, integer ( kind = 4 ) MX, the degree of the basis function.
    !
    !    Input, real ( kind = 8 ) XD(MX+1), the interpolation nodes.
    !
    !    Input, integer ( kind = 4 ) I, the index of the basis function.
    !    1 <= I <= MX+1.
    !
    !    Input, real ( kind = 8 ) XI, the evaluation point.
    !
    !    Output, real ( kind = 8 ) YI, the value of the I-th Lagrange 1D basis
    !    function for the nodes XD, evaluated at XI.
    !
    implicit none    
    integer ( kind = 4 ) mx
    integer ( kind = 4 ) i
    integer ( kind = 4 ) j
    real ( kind = 8 ) xd(mx+1)
    real ( kind = 8 ) xi
    real ( kind = 8 ) yi
    
    yi = 1.0D+00

    if ( xi /= xd(i) ) then
       do j = 1, mx + 1
          if ( j /= i ) then
             yi = yi * ( xi - xd(j) ) / ( xd(i) - xd(j) )
          end if
       end do
    end if

    return
  end subroutine lagrange_basis_function_1d

  ! the one dim lagrange interpolation can refer to my notes DGLAPSolver.pdf
  subroutine lagrange_interp_2d ( mx, my, xd_1d, yd_1d, zd, ni, xi, yi, zi )
    !*****************************************************************************80
    !
    !! LAGRANGE_INTERP_2D evaluates the Lagrange interpolant for a product grid.
    !
    !  Licensing:
    !
    !    This code is distributed under the GNU LGPL license.
    !
    !  Modified:
    !
    !    13 September 2012
    !
    !  Author:
    !
    !    John Burkardt
    !
    !  Parameters:
    !
    !    Input, integer ( kind = 4 ) MX, MY, the polynomial degree in X and Y.
    !
    !    Input, real ( kind = 8 ) XD_1D(MX+1), YD_1D(MY+1), the 1D data locations.
    !
    !    Input, real ( kind = 8 ) ZD((MX+1),(MY+1)), the 2D array of data values.
    !
    !    Input, integer ( kind = 4 ) NI, the number of 2D interpolation points.
    !
    !    Input, real ( kind = 8 ) XI(NI), YI(NI), the 2D interpolation points.
    !
    !    Output, real ( kind = 8 ) ZI(NI), the interpolated values.
    !
    implicit none

    integer ( kind = 4 ) mx
    integer ( kind = 4 ) my
    integer ( kind = 4 ) ni
    
    integer ( kind = 4 ) i
    integer ( kind = 4 ) j
    integer ( kind = 4 ) k
    integer ( kind = 4 ) l
    real ( kind = 8 ) lx
    real ( kind = 8 ) ly
    real ( kind = 8 ) xd_1d(mx+1)
    real ( kind = 8 ) xi(ni)
    real ( kind = 8 ) yd_1d(my+1)
    real ( kind = 8 ) yi(ni)
    real ( kind = 8 ) zd(mx+1,my+1)
    real ( kind = 8 ) zi(ni)
    
    do k = 1, ni
       l = 0
       zi(k) = 0.0D+00
       do i = 1, mx + 1
          do j = 1, my + 1
             l = l + 1
             call lagrange_basis_function_1d ( mx, xd_1d, i, xi(k), lx )
             call lagrange_basis_function_1d ( my, yd_1d, j, yi(k), ly )
             zi(k) = zi(k) + zd(i,j) * lx * ly
          end do
       end do
    end do
    
    return
  end subroutine lagrange_interp_2d

  ! the following code is obained from
  ! https://people.sc.fsu.edu/~jburkardt/f_src/pwl_interp_2d/pwl_interp_2d.html
  subroutine pwl_interp_2d ( nxd, nyd, xd, yd, zd, ni, xi, yi, zi )
    !*****************************************************************************80
    !
    !! PWL_INTERP_2D: piecewise linear interpolant to data defined on a 2D grid.
    !
    !  Discussion:
    !
    !    Thanks to Adam Hirst for pointing out an error in the formula that
    !    chooses the interpolation triangle, 04 February 2018.
    !
    !  Licensing:
    !
    !    This code is distributed under the GNU LGPL license.
    !
    !  Modified:
    !
    !    04 February 2018
    !
    !  Author:
    !
    !    John Burkardt
    !
    !  Parameters:
    !
    !    Input, integer ( kind = 4 ) NXD, NYD, the number of X and Y data values.
    !
    !    Input, real ( kind = 8 ) XD(NXD), YD(NYD), the sorted X and Y data.
    !
    !    Input, real ( kind = 8 ) ZD(NXD,NYD), the Z data.
    !
    !    Input, integer ( kind = 4 ) NI, the number of interpolation points.
    !
    !    Input, real ( kind = 8 ) XI(NI), YI(NI), the coordinates of the
    !    interpolation points.
    !
    !    Output, real ( kind = 8 ) ZI(NI), the value of the interpolant.
    !
    implicit none
    integer ( kind = 4 ) ni
    integer ( kind = 4 ) nxd
    integer ( kind = 4 ) nyd
    
    real ( kind = 8 ) alpha
    real ( kind = 8 ) beta
    real ( kind = 8 ) det
    real ( kind = 8 ) dxa
    real ( kind = 8 ) dxb
    real ( kind = 8 ) dxi
    real ( kind = 8 ) dya
    real ( kind = 8 ) dyb
    real ( kind = 8 ) dyi
    real ( kind = 8 ) gamma
    integer ( kind = 4 ) i
    integer ( kind = 4 ) j
    integer ( kind = 4 ) k
!    real ( kind = 8 ) r8_huge
!    integer ( kind = 4 ) r8vec_bracket5
    real ( kind = 8 ) xd(nxd)
    real ( kind = 8 ) xi(ni)
    real ( kind = 8 ) yd(nyd)
    real ( kind = 8 ) yi(ni)
    real ( kind = 8 ) zd(nxd,nyd)
    real ( kind = 8 ) zi(ni)
    
    do k = 1, ni
       !
       !  For interpolation point (xi(k),yi(k)), find data intervals I and J so that:
       !
       !    xd(i) <= xi(k) <= xd(i+1),
       !    yd(j) <= yi(k) <= yd(j+1).
       !
       !  But if the interpolation point is not within a data interval,
       !  assign the dummy interpolant value zi(k) = infinity.
       !
       i = r8vec_bracket5 ( nxd, xd, xi(k) )
       if ( i == -1 ) then
          zi(k) = r8_huge ( )
          cycle
       end if
       
       j = r8vec_bracket5 ( nyd, yd, yi(k) )
       if ( j == -1 ) then
          zi(k) = r8_huge ( )
          cycle
       end if
       !
       !  The rectangular cell is arbitrarily split into two triangles.
       !  The linear interpolation formula depends on which triangle 
       !  contains the data point.
       !
       !    (I,J+1)--(I+1,J+1)
       !      |\       |
       !      | \      |
       !      |  \     |
       !      |   \    |
       !      |    \   |
       !      |     \  |
       !    (I,J)---(I+1,J)
       !
       if ( yi(k) < yd(j+1) &
            + ( yd(j) - yd(j+1) ) * ( xi(k) - xd(i) ) / ( xd(i+1) - xd(i) ) ) then

          dxa = xd(i+1) - xd(i)
          dya = yd(j)   - yd(j)
          
          dxb = xd(i)   - xd(i)
          dyb = yd(j+1) - yd(j)
          
          dxi = xi(k)   - xd(i)
          dyi = yi(k)   - yd(j)
          
          det = dxa * dyb - dya * dxb
          
          alpha = ( dxi * dyb - dyi * dxb ) / det
          beta =  ( dxa * dyi - dya * dxi ) / det
          gamma = 1.0D+00 - alpha - beta
          
          zi(k) = alpha * zd(i+1,j) + beta * zd(i,j+1) + gamma * zd(i,j)
          
       else
          
          dxa = xd(i)   - xd(i+1)
          dya = yd(j+1) - yd(j+1)

          dxb = xd(i+1) - xd(i+1)
          dyb = yd(j)   - yd(j+1)
          
          dxi = xi(k)   - xd(i+1)
          dyi = yi(k)   - yd(j+1)
          
          det = dxa * dyb - dya * dxb
          
          alpha = ( dxi * dyb - dyi * dxb ) / det
          beta =  ( dxa * dyi - dya * dxi ) / det
          gamma = 1.0D+00 - alpha - beta
          
          zi(k) = alpha * zd(i,j+1) + beta * zd(i+1,j) + gamma * zd(i+1,j+1)
          
       end if
       
    end do
    
    return
  end subroutine pwl_interp_2d

  function r8_huge ( )
    !*****************************************************************************80
    !
    !! R8_HUGE returns a very large R8.
    !
    !  Discussion:
    !
    !    The value returned by this function is intended to be the largest
    !    representable real value.
    !
    !    FORTRAN90 provides a built-in routine HUGE ( X ) that
    !    can return the maximum representable number of the same datatype
    !    as X, if that is what is really desired.
    !
    !  Licensing:
    !
    !    This code is distributed under the GNU LGPL license.
    !
    !  Modified:
    !
    !    27 September 2014
    !
    !  Author:
    !
    !    John Burkardt
    !
    !  Parameters:
    !
    !    Output, real ( kind = 8 ) R8_HUGE, a "huge" value.
    !
    implicit none

    real ( kind = 8 ) r8_huge
    real ( kind = 8 ), parameter :: t = 1.0D+00
    
    r8_huge = huge ( t )
    return
  end function r8_huge

  function r8vec_bracket5 ( nd, xd, xi )    
    !*****************************************************************************80
    !
    !! R8VEC_BRACKET5 brackets data between successive entries of a sorted R8VEC.
    !
    !  Discussion:
    !
    !    We assume XD is sorted.
    !
    !    If XI is contained in the interval [XD(1),XD(N)], then the returned
    !    value B indicates that XI is contained in [ XD(B), XD(B+1) ].
    !
    !    If XI is not contained in the interval [XD(1),XD(N)], then B = -1.
    !
    !    This code implements a version of binary search which is perhaps more
    !    understandable than the usual ones.
    !
    !  Licensing:
    !
    !    This code is distributed under the GNU LGPL license.
    !
    !  Modified:
    !
    !    14 October 2012
    !
    !  Author:
    !
    !    John Burkardt
    !
    !  Parameters:
    !
    !    Input, integer ( kind = 4 ) ND, the number of data values.
    !
    !    Input, real ( kind = 8 ) XD(N), the sorted data.
    !
    !    Input, real ( kind = 8 ) XD, the query value.
    !
    !    Output, integer ( kind = 4 ) R8VEC_BRACKET5, the bracket information.
    !
    implicit none

    integer ( kind = 4 ) nd

    integer ( kind = 4 ) b
    integer ( kind = 4 ) l
    integer ( kind = 4 ) m
    integer ( kind = 4 ) r
    integer ( kind = 4 ) r8vec_bracket5
    real ( kind = 8 ) xd(nd)
    real ( kind = 8 ) xi
    
    if ( xi < xd(1) .or. xd(nd) < xi ) then
       
       b = -1
       
    else
       
       l = 1
       r = nd
       
       do while ( l + 1 < r )
          m = ( l + r ) / 2
          if ( xi < xd(m) ) then
             r = m
          else
             l = m
          end if
       end do
       
       b = l
       
    end if
    
    r8vec_bracket5 = b
    
    return
  end function r8vec_bracket5

  ! the following code is from
  ! https://people.sc.fsu.edu/~jburkardt/f_src/lagrange_interp_nd/lagrange_interp_nd.html
  subroutine cc_compute_points ( n, points )
    !*****************************************************************************80
    !
    !! CC_COMPUTE_POINTS: abscissas of a Clenshaw Curtis rule.
    !
    !  Discussion:
    !
    !    Our convention is that the abscissas are numbered from left to right.
    !
    !    The rule is defined on [-1,1].
    !
    !  Licensing:
    !
    !    This code is distributed under the GNU LGPL license.
    !
    !  Modified:
    !
    !    08 October 2008
    !
    !  Author:
    !
    !    John Burkardt
    !
    !  Parameters:
    !
    !    Input, integer ( kind = 4 ) N, the order.
    !    1 <= N.
    !
    !    Output, real ( kind = 8 ) POINTS(N), the abscissas.
    !
    implicit none
    integer ( kind = 4 ) n
    
    integer ( kind = 4 ) i
    real ( kind = 8 ), parameter :: pi = 3.141592653589793D+00
    real ( kind = 8 ) points(n)
    
    if ( n < 1 ) then
       
       write ( *, '(a)' ) ' '
       write ( *, '(a)' ) 'CC_COMPUTE_POINTS - Fatal error!'
       write ( *, '(a,i8)' ) '  Illegal value of N = ', n
       stop
       
    else if ( n == 1 ) then
       
       points(1) = 0.0D+00
       
    else
       
       do i = 1, n
          points(i) = cos ( real ( n - i, kind = 8 ) * pi &
               / real ( n - 1, kind = 8 ) )
       end do
       
       points(1) = -1.0D+00
       if ( mod ( n, 2 ) == 1 ) then
          points((n+1)/2) = 0.0D+00
       end if
       points(n) = +1.0D+00
       
    end if
    
    return
  end subroutine cc_compute_points

  subroutine lagrange_basis_1d ( nd, xd, ni, xi, lb )
    !*****************************************************************************80
    !
    !! LAGRANGE_BASIS_1D evaluates a 1D Lagrange basis.
    !
    !  Licensing:
    !
    !    This code is distributed under the GNU LGPL license.
    !
    !  Modified:
    !
    !    09 October 2012
    !
    !  Author:
    !
    !    John Burkardt
    !
    !  Parameters:
    !
    !    Input, integer ( kind = 4 ) ND, the number of data points.
    !
    !    Input, real ( kind = 8 ) XD(ND), the interpolation nodes.
    !
    !    Input, integer ( kind = 4 ) NI, the number of evaluation points.
    !
    !    Input, real ( kind = 8 ) XI(NI), the evaluation points.
    !
    !    Output, real ( kind = 8 ) LB(NI,ND), the value, at the I-th point XI,
    !    of the Jth basis function.
    !
    implicit none

    integer ( kind = 4 ) nd
    integer ( kind = 4 ) ni
    
    integer ( kind = 4 ) i
    integer ( kind = 4 ) j
    real ( kind = 8 ) lb(ni,nd)
    real ( kind = 8 ) xd(nd)
    real ( kind = 8 ) xi(ni)
    
    do i = 1, ni
       do j = 1, nd
          lb(i,j) = product ( ( xi(i) - xd(1:j-1)  ) / ( xd(j) - xd(1:j-1)  ) ) &
               * product ( ( xi(i) - xd(j+1:nd) ) / ( xd(j) - xd(j+1:nd) ) )
       end do
    end do
    
    return
  end subroutine lagrange_basis_1d

  subroutine lagrange_interp_nd_grid ( m, n_1d, a, b, nd, xd )
    !*****************************************************************************80
    !
    !! LAGRANGE_INTERP_ND_GRID sets an M-dimensional Lagrange interpolant grid.
    !
    !  Licensing:
    !
    !    This code is distributed under the GNU LGPL license.
    !
    !  Modified:
    !
    !    29 September 2012
    !
    !  Author:
    !
    !    John Burkardt
    !
    !  Parameters:
    !
    !    Input, integer ( kind = 4 ) M, the spatial dimension.
    !
    !    Input, integer ( kind = 4 ) N_1D(M), the order of the 1D rule to be used
    !    in each dimension.
    !
    !    Input, real ( kind = 8 ) A(M), B(M), the lower and upper limits.
    !
    !    Input, integer ( kind = 4 ) ND, the number of points in the product grid.
    !
    !    Output, real ( kind = 8 ) XD(M,ND), the points at which data was sampled.
    !
    implicit none

    integer ( kind = 4 ) m
    integer ( kind = 4 ) nd
    
    real ( kind = 8 ) a(m)
    real ( kind = 8 ) b(m)
    integer ( kind = 4 ) i
    integer ( kind = 4 ) n
    integer ( kind = 4 ) n_1d(m)
    real ( kind = 8 ), allocatable :: x_1d(:)
    real ( kind = 8 ) xd(m,nd)
    !
    !  Compute the data points.
    !
    xd(1:m,1:nd) = 0.0D+00
    do i = 1, m
       n = n_1d(i)
       allocate ( x_1d(1:n) )
       call cc_compute_points ( n, x_1d )
       x_1d(1:n) = 0.5D+00 * ( ( 1.0D+00 - x_1d(1:n) ) * a(i) &
            + ( 1.0D+00 + x_1d(1:n) ) * b(i) )
       call r8vec_direct_product ( i, n, x_1d, m, nd, xd )
       deallocate ( x_1d )
    end do
    
    return
  end subroutine lagrange_interp_nd_grid

  subroutine lagrange_interp_nd_grid2 ( m, ind, a, b, nd, xd )
    !*****************************************************************************80
    !
    !! LAGRANGE_INTERP_ND_GRID2 sets an M-dimensional Lagrange interpolant grid.
    !
    !  Licensing:
    !
    !    This code is distributed under the GNU LGPL license.
    !
    !  Modified:
    !
    !    29 September 2012
    !
    !  Author:
    !
    !    John Burkardt
    !
    !  Parameters:
    !
    !    Input, integer ( kind = 4 ) M, the spatial dimension.
    !
    !    Input, integer ( kind = 4 ) IND(M), the index or level of the 1D rule
    !    to be used in each dimension.
    !
    !    Input, real ( kind = 8 ) A(M), B(M), the lower and upper limits.
    !
    !    Input, integer ( kind = 4 ) ND, the number of points in the product grid.
    !
    !    Output, real ( kind = 8 ) XD(M,ND), the points at which data was sampled.
    !
    implicit none

    integer ( kind = 4 ) m
    integer ( kind = 4 ) nd
    
    real ( kind = 8 ) a(m)
    real ( kind = 8 ) b(m)
    integer ( kind = 4 ) i
    integer ( kind = 4 ) ind(m)
    integer ( kind = 4 ) n
    real ( kind = 8 ), allocatable :: x_1d(:)
    real ( kind = 8 ) xd(m,nd)
    !
    !  Compute the data points.
    !
    xd(1:m,1:nd) = 0.0D+00
    do i = 1, m
       call order_from_level_135 ( ind(i), n )
       allocate ( x_1d(1:n) )
       call cc_compute_points ( n, x_1d )
       x_1d(1:n) = 0.5D+00 * ( ( 1.0D+00 - x_1d(1:n) ) * a(i) &
            + ( 1.0D+00 + x_1d(1:n) ) * b(i) )
       call r8vec_direct_product ( i, n, x_1d, m, nd, xd )
       deallocate ( x_1d )
    end do
    
    return
  end subroutine lagrange_interp_nd_grid2

  subroutine lagrange_interp_nd_size ( m, n_1d, nd )
    !*****************************************************************************80
    !
    !! LAGRANGE_INTERP_ND_SIZE sizes an M-dimensional Lagrange interpolant.
    !
    !  Licensing:
    !
    !    This code is distributed under the GNU LGPL license.
    !
    !  Modified:
    !
    !    28 September 2012
    !
    !  Author:
    !
    !    John Burkardt
    !
    !  Parameters:
    !
    !    Input, integer ( kind = 4 ) M, the spatial dimension.
    !
    !    Input, integer ( kind = 4 ) N_1D(M), the order of the 1D rule to be used
    !    in each dimension.
    !
    !    Output, integer ( kind = 4 ) ND, the number of points in the product grid.
    !
    implicit none

    integer ( kind = 4 ) m
    
    integer ( kind = 4 ) n_1d(m)
    integer ( kind = 4 ) nd
    !
    !  Determine the number of data points.
    !
    nd = product ( n_1d(1:m) )

    return
  end subroutine lagrange_interp_nd_size

  subroutine lagrange_interp_nd_size2 ( m, ind, nd )
    !*****************************************************************************80
    !
    !! LAGRANGE_INTERP_ND_SIZE2 sizes an M-dimensional Lagrange interpolant.
    !
    !  Licensing:
    !
    !    This code is distributed under the GNU LGPL license.
    !
    !  Modified:
    !
    !    28 September 2012
    !
    !  Author:
    !
    !    John Burkardt
    !
    !  Parameters:
    !
    !    Input, integer ( kind = 4 ) M, the spatial dimension.
    !
    !    Input, integer ( kind = 4 ) IND(M), the index or level of the 1D rule 
    !    to be used in each dimension.
    !
    !    Output, integer ( kind = 4 ) ND, the number of points in the product grid.
    !
    implicit none

    integer ( kind = 4 ) m
    
    integer ( kind = 4 ) i
    integer ( kind = 4 ) ind(m)
    integer ( kind = 4 ) n
    integer ( kind = 4 ) nd
    !
    !  Determine the number of data points.
    !
    nd = 1
    do i = 1, m
       call order_from_level_135 ( ind(i), n )
       nd = nd * n
    end do
    
    return
  end subroutine lagrange_interp_nd_size2

  subroutine lagrange_interp_nd_value ( m, n_1d, a, b, nd, zd, ni, xi, zi )
    !*****************************************************************************80
    !
    !! LAGRANGE_INTERP_ND_VALUE evaluates an ND Lagrange interpolant.
    !
    !  Licensing:
    !
    !    This code is distributed under the GNU LGPL license.
    !
    !  Modified:
    !
    !    28 September 2012
    !
    !  Author:
    !
    !    John Burkardt
    !
    !  Parameters:
    !
    !    Input, integer ( kind = 4 ) M, the spatial dimension.
    !
    !    Input, integer ( kind = 4 ) N_1D(M), the order of the 1D rule to be used
    !    in each dimension.
    !
    !    Input, real ( kind = 8 ) A(M), B(M), the lower and upper limits.
    !
    !    Input, integer ( kind = 4 ) ND, the number of points in the product grid.
    !
    !    Input, real ( kind = 8 ) ZD(ND), the function evaluated at the points XD.
    !
    !    Input, integer ( kind = 4 ) NI, the number of points at which the
    !    interpolant is to be evaluated.
    !
    !    Input, real ( kind = 8 ) XI(M,NI), the points at which the interpolant is 
    !    to be evaluated.
    !
    !    Output, real ( kind = 8 ) ZI(NI), the interpolant evaluated at the 
    !    points XI.
    !
    implicit none

    integer ( kind = 4 ) m
    integer ( kind = 4 ) nd
    integer ( kind = 4 ) ni
    
    real ( kind = 8 ) a(m)
    real ( kind = 8 ) b(m)
    integer ( kind = 4 ) i
    integer ( kind = 4 ) j
    integer ( kind = 4 ) n
    integer ( kind = 4 ) n_1d(m)
    real ( kind = 8 ), allocatable :: value(:)
    real ( kind = 8 ) w(nd)
    real ( kind = 8 ), allocatable :: x_1d(:)
    real ( kind = 8 ) xi(m,ni)
    real ( kind = 8 ) zd(nd)
    real ( kind = 8 ) zi(ni)
    
    do j = 1, ni
       
       w(1:nd) = 1.0D+00
       
       do i = 1, m
          n = n_1d(i)
          allocate ( x_1d(1:n) )
          allocate ( value(1:n) )
          call cc_compute_points ( n, x_1d )
          x_1d(1:n) = 0.5D+00 * ( ( 1.0D+00 - x_1d(1:n) ) * a(i) &
               + ( 1.0D+00 + x_1d(1:n) ) * b(i) )
          call lagrange_basis_1d ( n, x_1d, 1, xi(i,j), value )
          call r8vec_direct_product2 ( i, n, value, m, nd, w )
          deallocate ( value )
          deallocate ( x_1d )
       end do
       
       zi(j) = dot_product ( w, zd )
       
    end do

    return
  end subroutine lagrange_interp_nd_value

  subroutine lagrange_interp_nd_value2 ( m, ind, a, b, nd, zd, ni, xi, zi )
    !*****************************************************************************80
    !
    !! LAGRANGE_INTERP_ND_VALUE2 evaluates an ND Lagrange interpolant.
    !
    !  Licensing:
    !
    !    This code is distributed under the GNU LGPL license.
    !
    !  Modified:
    !
    !    28 September 2012
    !
    !  Author:
    !
    !    John Burkardt
    !
    !  Parameters:
    !
    !    Input, integer ( kind = 4 ) M, the spatial dimension.
    !
    !    Input, integer ( kind = 4 ) IND(M), the index or level of the 1D rule 
    !    to be used in each dimension.
    !
    !    Input, real ( kind = 8 ) A(M), B(M), the lower and upper limits.
    !
    !    Input, integer ( kind = 4 ) ND, the number of points in the product grid.
    !
    !    Input, real ( kind = 8 ) ZD(ND), the function evaluated at the points XD.
    !
    !    Input, integer ( kind = 4 ) NI, the number of points at which the 
    !    interpolant is to be evaluated.
    !
    !    Input, real ( kind = 8 ) XI(M,NI), the points at which the interpolant
    !    is to be evaluated.
    !
    !    Output, real ( kind = 8 ) ZI(NI), the interpolant evaluated at the 
    !    points XI.
    !
    implicit none

    integer ( kind = 4 ) m
    integer ( kind = 4 ) nd
    integer ( kind = 4 ) ni
    
    real ( kind = 8 ) a(m)
    real ( kind = 8 ) b(m)
    integer ( kind = 4 ) i
    integer ( kind = 4 ) ind(m)
    integer ( kind = 4 ) j
    integer ( kind = 4 ) n
    real ( kind = 8 ), allocatable :: value(:)
    real ( kind = 8 ) w(nd)
    real ( kind = 8 ), allocatable :: x_1d(:)
    real ( kind = 8 ) xi(m,ni)
    real ( kind = 8 ) zd(nd)
    real ( kind = 8 ) zi(ni)
    
    do j = 1, ni
       
       w(1:nd) = 1.0D+00
       
       do i = 1, m
          call order_from_level_135 ( ind(i), n )
          allocate ( x_1d(1:n) )
          allocate ( value(1:n) )
          call cc_compute_points ( n, x_1d )
          x_1d(1:n) = 0.5D+00 * ( ( 1.0D+00 - x_1d(1:n) ) * a(i) &
               + ( 1.0D+00 + x_1d(1:n) ) * b(i) )
          call lagrange_basis_1d ( n, x_1d, 1, xi(i,j), value )
          call r8vec_direct_product2 ( i, n, value, m, nd, w )
          deallocate ( value )
          deallocate ( x_1d )
       end do
       
       zi(j) = dot_product ( w, zd )
       
    end do
    
    return
  end subroutine lagrange_interp_nd_value2

  subroutine order_from_level_135 ( l, n )
    !*****************************************************************************80
    !
    !! ORDER_FROM_LEVEL_135 evaluates the 135 level-to-order relationship.
    !
    !  Discussion:
    !
    !    Clenshaw Curtis rules, and some others, often use the following
    !    scheme:
    !
    !    L: 0  1  2  3   4   5
    !    N: 1  3  5  9  17  33 ... 2^L+1
    !
    !  Licensing:
    !
    !    This code is distributed under the GNU LGPL license.
    !
    !  Modified:
    !
    !    28 September 2012
    !
    !  Author:
    !
    !    John Burkardt
    !
    !  Parameters:
    !
    !    Input, integer ( kind = 4 ) L, the level, which should be 0 or greater.
    !
    !    Output, integer ( kind = 4 ) N, the order.
    !
    implicit none

    integer ( kind = 4 ) l
    integer ( kind = 4 ) n
    
    if ( l < 0 ) then
       write ( *, '(a)' ) ''
       write ( *, '(a)' ) 'ORDER_FROM_LEVEL_135 - Fatal error!'
       write ( *, '(a)' ) '  Illegal input value of L!'
       stop
    else if ( l == 0 ) then
       n = 1
    else
       n = ( 2 ** l ) + 1
    end if
    
    return
  end subroutine order_from_level_135

  subroutine r8vec_direct_product ( factor_index, factor_order, factor_value, &
       factor_num, point_num, x )
    !*****************************************************************************80
    !
    !! R8VEC_DIRECT_PRODUCT creates a direct product of R8VEC's.
    !
    !  Discussion:
    !
    !    An R8VEC is a vector of R8's.
    !
    !    To explain what is going on here, suppose we had to construct
    !    a multidimensional quadrature rule as the product of K rules
    !    for 1D quadrature.
    !
    !    The product rule will be represented as a list of points and weights.
    !
    !    The J-th item in the product rule will be associated with
    !      item J1 of 1D rule 1,
    !      item J2 of 1D rule 2,
    !      ...,
    !      item JK of 1D rule K.
    !
    !    In particular,
    !      X(J) = ( X(1,J1), X(2,J2), ..., X(K,JK))
    !    and
    !      W(J) = W(1,J1) * W(2,J2) * ... * W(K,JK)
    !
    !    So we can construct the quadrature rule if we can properly
    !    distribute the information in the 1D quadrature rules.
    !
    !    This routine carries out that task for the abscissas X.
    !
    !    Another way to do this would be to compute, one by one, the
    !    set of all possible indices (J1,J2,...,JK), and then index
    !    the appropriate information.  An advantage of the method shown
    !    here is that you can process the K-th set of information and
    !    then discard it.
    !
    !  Example:
    !
    !    Rule 1:
    !      Order = 4
    !      X(1:4) = ( 1, 2, 3, 4 )
    !
    !    Rule 2:
    !      Order = 3
    !      X(1:3) = ( 10, 20, 30 )
    !
    !    Rule 3:
    !      Order = 2
    !      X(1:2) = ( 100, 200 )
    !
    !    Product Rule:
    !      Order = 24
    !      X(1:24) =
    !        ( 1, 10, 100 )
    !        ( 2, 10, 100 )
    !        ( 3, 10, 100 )
    !        ( 4, 10, 100 )
    !        ( 1, 20, 100 )
    !        ( 2, 20, 100 )
    !        ( 3, 20, 100 )
    !        ( 4, 20, 100 )
    !        ( 1, 30, 100 )
    !        ( 2, 30, 100 )
    !        ( 3, 30, 100 )
    !        ( 4, 30, 100 )
    !        ( 1, 10, 200 )
    !        ( 2, 10, 200 )
    !        ( 3, 10, 200 )
    !        ( 4, 10, 200 )
    !        ( 1, 20, 200 )
    !        ( 2, 20, 200 )
    !        ( 3, 20, 200 )
    !        ( 4, 20, 200 )
    !        ( 1, 30, 200 )
    !        ( 2, 30, 200 )
    !        ( 3, 30, 200 )
    !        ( 4, 30, 200 )
    !
    !  Licensing:
    !
    !    This code is distributed under the GNU LGPL license.
    !
    !  Modified:
    !
    !    18 April 2009
    !
    !  Author:
    !
    !    John Burkardt
    !
    !  Parameters:
    !
    !    Input, integer ( kind = 4 ) FACTOR_INDEX, the index of the factor being
    !    processed.  The first factor processed must be factor 1!
    !
    !    Input, integer ( kind = 4 ) FACTOR_ORDER, the order of the factor.
    !
    !    Input, real ( kind = 8 ) FACTOR_VALUE(FACTOR_ORDER), the factor values
    !    for factor FACTOR_INDEX.
    !
    !    Input, integer ( kind = 4 ) FACTOR_NUM, the number of factors.
    !
    !    Input, integer ( kind = 4 ) POINT_NUM, the number of elements in the
    !    direct product.
    !
    !    Input/output, real ( kind = 8 ) X(FACTOR_NUM,POINT_NUM), the elements of
    !    the direct product, which are built up gradually.
    !
    !  Local Parameters:
    !
    !    Local, integer ( kind = 4 ) START, the first location of a block of 
    !    values to set.
    !
    !    Local, integer ( kind = 4 ) CONTIG, the number of consecutive values 
    !    to set.
    !
    !    Local, integer ( kind = 4 ) SKIP, the distance from the current value 
    !    of START to the next location of a block of values to set.
    !
    !    Local, integer ( kind = 4 ) REP, the number of blocks of values to set.
    !
    implicit none

    integer ( kind = 4 ) factor_num
    integer ( kind = 4 ) factor_order
    integer ( kind = 4 ) point_num
    
    integer ( kind = 4 ), save :: contig
    integer ( kind = 4 ) factor_index
    real ( kind = 8 ) factor_value(factor_order)
    integer ( kind = 4 ) j
    integer ( kind = 4 ) k
    integer ( kind = 4 ), save :: rep
    integer ( kind = 4 ), save :: skip
    integer ( kind = 4 ) start
    real ( kind = 8 ) x(factor_num,point_num)
    
    if ( factor_index == 1 ) then
       contig = 1
       skip = 1
       rep = point_num
       x(1:factor_num,1:point_num) = 0.0D+00
    end if
    
    rep = rep / factor_order
    skip = skip * factor_order
    
    do j = 1, factor_order
       
       start = 1 + ( j - 1 ) * contig
       
       do k = 1, rep
          x(factor_index,start:start+contig-1) = factor_value(j)
          start = start + skip
       end do
       
    end do
    
    contig = contig * factor_order
    
    return
  end subroutine r8vec_direct_product

  subroutine r8vec_direct_product2 ( factor_index, factor_order, factor_value, &
       factor_num, point_num, w )
    !*****************************************************************************80
    !
    !! R8VEC_DIRECT_PRODUCT2 creates a direct product of R8VEC's.
    !
    !  Discussion:
    !
    !    An R8VEC is a vector of R8's.
    !
    !    To explain what is going on here, suppose we had to construct
    !    a multidimensional quadrature rule as the product of K rules
    !    for 1D quadrature.
    !
    !    The product rule will be represented as a list of points and weights.
    !
    !    The J-th item in the product rule will be associated with
    !      item J1 of 1D rule 1,
    !      item J2 of 1D rule 2,
    !      ...,
    !      item JK of 1D rule K.
    !
    !    In particular,
    !      X(J) = ( X(1,J1), X(2,J2), ..., X(K,JK))
    !    and
    !      W(J) = W(1,J1) * W(2,J2) * ... * W(K,JK)
    !
    !    So we can construct the quadrature rule if we can properly
    !    distribute the information in the 1D quadrature rules.
    !
    !    This routine carries out the task involving the weights W.
    !
    !    Another way to do this would be to compute, one by one, the
    !    set of all possible indices (J1,J2,...,JK), and then index
    !    the appropriate information.  An advantage of the method shown
    !    here is that you can process the K-th set of information and
    !    then discard it.
    !
    !  Example:
    !
    !    Rule 1:
    !      Order = 4
    !      W(1:4) = ( 2, 3, 5, 7 )
    !
    !    Rule 2:
    !      Order = 3
    !      W(1:3) = ( 11, 13, 17 )
    !
    !    Rule 3:
    !      Order = 2
    !      W(1:2) = ( 19, 23 )
    !
    !    Product Rule:
    !      Order = 24
    !      W(1:24) =
    !        ( 2 * 11 * 19 )
    !        ( 3 * 11 * 19 )
    !        ( 4 * 11 * 19 )
    !        ( 7 * 11 * 19 )
    !        ( 2 * 13 * 19 )
    !        ( 3 * 13 * 19 )
    !        ( 5 * 13 * 19 )
    !        ( 7 * 13 * 19 )
    !        ( 2 * 17 * 19 )
    !        ( 3 * 17 * 19 )
    !        ( 5 * 17 * 19 )
    !        ( 7 * 17 * 19 )
    !        ( 2 * 11 * 23 )
    !        ( 3 * 11 * 23 )
    !        ( 5 * 11 * 23 )
    !        ( 7 * 11 * 23 )
    !        ( 2 * 13 * 23 )
    !        ( 3 * 13 * 23 )
    !        ( 5 * 13 * 23 )
    !        ( 7 * 13 * 23 )
    !        ( 2 * 17 * 23 )
    !        ( 3 * 17 * 23 )
    !        ( 5 * 17 * 23 )
    !        ( 7 * 17 * 23 )
    !
    !  Licensing:
    !
    !    This code is distributed under the GNU LGPL license.
    !
    !  Modified:
    !
    !    18 April 2009
    !
    !  Author:
    !
    !    John Burkardt
    !
    !  Parameters:
    !
    !    Input, integer ( kind = 4 ) FACTOR_INDEX, the index of the factor being
    !    processed.  The first factor processed must be factor 1!
    !
    !    Input, integer ( kind = 4 ) FACTOR_ORDER, the order of the factor.
    !
    !    Input, real ( kind = 8 ) FACTOR_VALUE(FACTOR_ORDER), the factor values
    !    for factor FACTOR_INDEX.
    !
    !    Input, integer ( kind = 4 ) FACTOR_NUM, the number of factors.
    !
    !    Input, integer ( kind = 4 ) POINT_NUM, the number of elements in the
    !    direct product.
    !
    !    Input/output, real ( kind = 8 ) W(POINT_NUM), the elements of the
    !    direct product, which are built up gradually.
    !
    !  Local Parameters:
    !
    !    Local, integer ( kind = 4 ) START, the first location of a block of values
    !    to set.
    !
    !    Local, integer ( kind = 4 ) CONTIG, the number of consecutive values
    !    to set.
    !
    !    Local, integer ( kind = 4 ) SKIP, the distance from the current value
    !    of START to the next location of a block of values to set.
    !
    !    Local, integer ( kind = 4 ) REP, the number of blocks of values to set.
    !
    implicit none

    integer ( kind = 4 ) factor_num
    integer ( kind = 4 ) factor_order
    integer ( kind = 4 ) point_num
    
    integer ( kind = 4 ), save :: contig
    integer ( kind = 4 ) factor_index
    real ( kind = 8 ) factor_value(factor_order)
    integer ( kind = 4 ) j
    integer ( kind = 4 ) k
    integer ( kind = 4 ), save :: rep
    integer ( kind = 4 ), save :: skip
    integer ( kind = 4 ) start
    real ( kind = 8 ) w(point_num)

    call i4_fake_use ( factor_num )
    
    if ( factor_index == 1 ) then
       contig = 1
       skip = 1
       rep = point_num
       w(1:point_num) = 1.0D+00
    end if
    
    rep = rep / factor_order
    skip = skip * factor_order
    
    do j = 1, factor_order

       start = 1 + ( j - 1 ) * contig
       
       do k = 1, rep
          w(start:start+contig-1) = w(start:start+contig-1) * factor_value(j)
          start = start + skip
       end do
       
    end do
    
    contig = contig * factor_order
    
    return
  end subroutine r8vec_direct_product2

  subroutine i4_fake_use ( n )
    !*****************************************************************************80
    !
    !! i4_fake_use pretends to use a variable.
    !
    !  Discussion:
    !
    !    Some compilers will issue a warning if a variable is unused.
    !    Sometimes there's a good reason to include a variable in a program,
    !    but not to use it.  Calling this function with that variable as
    !    the argument will shut the compiler up.
    !
    !  Licensing:
    !
    !    This code is distributed under the GNU LGPL license.
    !
    !  Modified:
    !
    !    21 April 2020
    !
    !  Author:
    !
    !    John Burkardt
    !
    !  Input:
    !
    !    integer ( kind = 4 ) N, the variable to be "used".
    !
    implicit none

    integer ( kind = 4 ) n

    if ( n /= n ) then
       write ( *, '(a)' ) '  i4_fake_use: variable is NAN.'
    end if

    return
  end subroutine i4_fake_use
END MODULE interpolation
