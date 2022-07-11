MODULE NINTLIB
  IMPLICIT NONE
CONTAINS
  SUBROUTINE box_nd(func,dim_num,order,xtab,weight,res,eval_num)
    !****************************************************************************
    !
    !! BOX_ND estimates a multidimensional integral using a product rule.
    !
    !  Discussion:
    !
    !    The routine creates a DIM_NUM-dimensional product rule from a 1D rule
    !    supplied by the user.  The routine is fairly inflexible.  If
    !    you supply a rule for integration from -1 to 1, then your product
    !    box must be a product of DIM_NUM copies of the interval [-1,1].
    !
    !  Licensing:
    !
    !    This code is distributed under the GNU LGPL license.
    !
    !  Modified:
    !
    !    11 September 2006
    !
    !  Author:
    !
    !    John Burkardt
    !
    !  Reference:
    !
    !    Philip Davis, Philip Rabinowitz,
    !    Methods of Numerical Integration,
    !    Second Edition,
    !    Dover, 2007,
    !    ISBN: 0486453391,
    !    LC: QA299.3.D28.
    !
    !  Parameters:
    !
    !    Input, real ( kind = 8 ), external FUNC, a routine which evaluates
    !    the function to be integrated, of the form:
    !      function func ( dim_num, x )
    !      integer ( kind = 4 ) dim_num
    !      real ( kind = 8 ) func
    !      real ( kind = 8 ) x(dim_num)
    !      func = ...
    !      return
    !      end
    !
    !    Input, integer ( kind = 4 ) DIM_NUM, the spatial dimension.
    !
    !    Input, integer ( kind = 4 ) ORDER, the number of points used 
    !    in the 1D rule.
    !
    !    Input, real ( kind = 8 ) XTAB(ORDER), the abscissas of the 1D rule.
    !
    !    Input, real ( kind = 8 ) WEIGHT(ORDER), the weights of the 1D rule.
    !
    !    Output, real ( kind = 8 ) RES, the approximate value of the integral.
    !
    !    Output, integer ( kind = 4 ) EVAL_NUM, the number of function evaluations.
    !
    IMPLICIT NONE
    INTEGER,INTENT(IN)::dim_num
    INTEGER,INTENT(IN)::order

    INTEGER,INTENT(OUT)::eval_num
    REAL(KIND(1d0)),EXTERNAL::func
    INTEGER,DIMENSION(dim_num)::indx
    INTEGER::k
    REAL(KIND(1d0)),INTENT(OUT)::res
    REAL(KIND(1d0))::w
    REAL(KIND(1d0)),DIMENSION(order),INTENT(IN)::weight
    REAL(KIND(1d0)),DIMENSION(dim_num)::x
    REAL(KIND(1d0)),DIMENSION(order),INTENT(IN)::xtab
    
    eval_num = 0
    
    IF(dim_num.LT.1)THEN
       WRITE( *, '(a)' ) ' '
       WRITE( *, '(a)' ) 'BOX_ND - Fatal error!'
       WRITE( *, '(a)' ) '  DIM_NUM < 1'
       WRITE( *, '(a,i8)' ) '  DIM_NUM = ', dim_num
       STOP
    ENDIF
    
    IF( order.LT.1)THEN
       WRITE( *, '(a)' ) ' '
       WRITE( *, '(a)' ) 'BOX_ND - Fatal error!'
       WRITE( *, '(a)' ) '  ORDER < 1'
       WRITE( *, '(a,i8)' ) '  ORDER = ', order
       STOP
    ENDIF

    k = 0
    res=0.0D+00

    DO

       CALL tuple_next (1,order,dim_num,k,indx)
       IF(k==0)THEN
          EXIT
       ENDIF

       w = PRODUCT(weight(indx(1:dim_num)))

       x(1:dim_num) = xtab(indx(1:dim_num))

       res = res + w*func(dim_num,x)
       eval_num = eval_num + 1

    ENDDO

    RETURN
  END SUBROUTINE box_nd

  SUBROUTINE monte_carlo_nd(func,dim_num,a,b,eval_num,seed,res)
    !****************************************************************************
    !
    !! MONTE_CARLO_ND estimates a multidimensional integral using Monte Carlo.
    !
    !  Discussion:
    !
    !    Unlike the other routines, this routine requires the user to specify
    !    the number of function evaluations as an INPUT quantity.
    !
    !    No attempt at error estimation is made.
    !
    !  Licensing:
    !
    !    This code is distributed under the GNU LGPL license.
    !
    !  Modified:
    !
    !    25 February 2007
    !
    !  Author:
    !
    !    John Burkardt
    !
    !  Reference:
    !
    !    Philip Davis, Philip Rabinowitz,
    !    Methods of Numerical Integration,
    !    Second Edition,
    !    Dover, 2007,
    !    ISBN: 0486453391,
    !    LC: QA299.3.D28.
    !
    !  Parameters:
    !
    !    Input, real ( kind = 8 ), external FUNC, a routine which evaluates
    !    the function to be integrated, of the form:
    !      function func ( dim_num, x )
    !      integer ( kind = 4 ) dim_num
    !      real ( kind = 8 ) func
    !      real ( kind = 8 ) x(dim_num)
    !      func = ...
    !      return
    !      end
    !
    !    Input, integer ( kind = 4 ) DIM_NUM, the spatial dimension.
    !
    !    Input, real ( kind = 8 ) A(DIM_NUM), B(DIM_NUM), the integration limits.
    !
    !    Input, integer ( kind = 4 ) EVAL_NUM, the number of function evaluations.
    !
    !    Input/output, integer ( kind = 4 ) SEED, a seed for the random 
    !    number generator.
    !
    !    Output, real ( kind = 8 ) RES, the approximate value of the integral.
    !
    IMPLICIT NONE

    INTEGER,INTENT(IN)::dim_num

    REAL(KIND(1d0)),DIMENSION(dim_num),INTENT(IN)::a,b
    INTEGER,INTENT(IN)::eval_num
    REAL(KIND(1d0)),EXTERNAL::func
    INTEGER::i
    REAL(KIND(1d0)),INTENT(OUT)::res
    INTEGER,INTENT(INOUT)::seed
    REAL(KIND(1d0))::volume
    REAL(KIND(1d0)),DIMENSION(dim_num)::x

    res = 0.0D+00

    DO i = 1, eval_num

       call r8vec_uniform_01( dim_num,seed,x)

       res = res + func(dim_num,x)

    ENDDO

    volume = PRODUCT(b(1:dim_num)-a(1:dim_num))

    res = res*volume/DBLE(eval_num)

    RETURN
  END SUBROUTINE monte_carlo_nd

  
  SUBROUTINE p5_nd(func,dim_num,a,b,res,eval_num)
    !*****************************************************************************
    !
    !! P5_ND estimates a multidimensional integral with a formula of exactness 5.
    !
    !  Discussion:
    !
    !    The routine uses a method which is exact for polynomials of total
    !    degree 5 or less.
    !
    !  Licensing:
    !
    !    This code is distributed under the GNU LGPL license.
    !
    !  Modified:
    !
    !    11 September 2006
    !
    !  Author:
    !
    !    Original FORTRAN77 version by Philip Davis, Philip Rabinowitz.
    !    FORTRAN90 version by John Burkardt
    !
    !  Reference:
    !
    !    Philip Davis, Philip Rabinowitz,
    !    Methods of Numerical Integration,
    !    Second Edition,
    !    Dover, 2007,
    !    ISBN: 0486453391,
    !    LC: QA299.3.D28.
    !
    !  Parameters:
    !
    !    Input, real ( kind = 8 ), external FUNC, a routine which evaluates
    !    the function to be integrated, of the form:
    !      function func ( dim_num, x )
    !      integer ( kind = 4 ) dim_num
    !      real ( kind = 8 ) func
    !      real ( kind = 8 ) x(dim_num)
    !      func = ...
    !      return
    !      end
    !
    !    Input, integer ( kind = 4 ) DIM_NUM, the spatial dimension.
    !
    !    Input, real ( kind = 8 ) A(DIM_NUM), B(DIM_NUM), the integration limits.
    !
    !    Output, real ( kind = 8 ) RESULT, the approximate value of the integral.
    !
    !    Output, integer ( kind = 4 ) EVAL_NUM, the number of function evaluations.
    !
    IMPLICIT NONE

    INTEGER,INTENT(IN)::dim_num

    REAL(KIND(1d0)),DIMENSION(dim_num),INTENT(IN)::a,b
    REAL(KIND(1d0))::a0
    REAL(KIND(1d0))::a1
    REAL(KIND(1d0))::a2
    REAL(KIND(1d0))::a3
    REAL(KIND(1d0))::a4
    REAL(KIND(1d0))::a5
    REAL(KIND(1d0))::en
    INTEGER,INTENT(OUT)::eval_num
    REAL(KIND(1d0)),EXTERNAL::func
    INTEGER::i
    INTEGER::j
    REAL(KIND(1d0)),INTENT(OUT)::res
    REAL(KIND(1d0))::sum1
    REAL(KIND(1d0))::sum2
    REAL(KIND(1d0))::sum3
    REAL(KIND(1d0))::volume
    REAL(KIND(1d0)),DIMENSION(dim_num)::work

    eval_num = 0

    IF( dim_num.LT.1)THEN
       WRITE( *, '(a)' ) ' '
       WRITE( *, '(a)' ) 'P5_ND - Fatal error!'
       WRITE( *, '(a,i8)' ) '  DIM_NUM < 1, DIM_NUM = ', dim_num
       STOP
    ENDIF

    a2 = 25.0D+00 / 324.0D+00
    a3 = DSQRT( 0.6D+00 )
    en = DBLE(dim_num)
    a0 = ( 25.0D+00 * en * en - 115.0D+00 * en + 162.0D+00 ) / 162.0D+00
    a1 = ( 70.0D+00 - 25.0D+00 * en ) / 162.0D+00

    volume = PRODUCT(b(1:dim_num)-a(1:dim_num))
    work(1:dim_num) = 0.5D+00 * (a(1:dim_num)+b(1:dim_num))

    res = 0.0D+00
    IF(volume.EQ.0.0D+00)THEN
       WRITE( *, '(a)' ) ' '
       WRITE( *, '(a)' ) 'P5_ND - Warning!'
       WRITE( *, '(a)' ) '  Volume = 0, integral = 0.'
       RETURN
    ENDIF

    sum1 = a0 * func ( dim_num, work )
    eval_num = eval_num + 1

    sum2 = 0.0D+00
    sum3 = 0.0D+00
    
    DO i=1,dim_num

       work(i) = 0.5D+00 * ( ( a(i) + b(i) ) + a3 * ( b(i) - a(i) ) )
       sum2 = sum2 + func ( dim_num, work )
       eval_num = eval_num + 1
       
       work(i) = 0.5D+00 * ( ( a(i) + b(i) ) - a3 * ( b(i) - a(i) ) )
       sum2 = sum2 + func ( dim_num, work )
       eval_num = eval_num + 1
       
       work(i) = 0.5D+00 * ( a(i) + b(i) )
       
    ENDDO

    IF(1.LT.dim_num)THEN

       a4 = a3

       DO

          DO i=1,dim_num-1

             work(i) = 0.5D+00 * ( ( a(i) + b(i) ) + a4 * ( b(i) - a(i) ) )
             a5 = a3

             DO

                DO j = i + 1, dim_num
                   work(j) = 0.5D+00 * ( ( a(j) + b(j) ) + a5 * ( b(j) - a(j) ) )
                   sum3 = sum3 + func ( dim_num, work )
                   eval_num = eval_num + 1
                   work(j) = 0.5D+00 * ( a(j) + b(j) )
                ENDDO

                a5 = -a5
                
                IF( 0.0D+00.LE.a5 )THEN
                   EXIT
                ENDIF

             ENDDO

             work(i) = 0.5D+00 * ( a(i) + b(i) )

          ENDDO

          a4 = -a4

          IF(0.0D+00.LE.a4)THEN
             EXIT
          ENDIF

       ENDDO

    ENDIF

    res = volume * ( sum1 + a1 * sum2 + a2 * sum3 )

    RETURN
  END SUBROUTINE p5_nd

  SUBROUTINE r8vec_uniform_01(n,seed,r)
    !*****************************************************************************80
    !
    !! R8VEC_UNIFORM_01 returns a unit pseudorandom R8VEC.
    !
    !  Discussion:
    !
    !    An R8VEC is a vector of real ( kind = 8 ) values.
    !
    !    For now, the input quantity SEED is an integer ( kind = 4 ) variable.
    !
    !  Licensing:
    !
    !    This code is distributed under the GNU LGPL license.
    !
    !  Modified:
    !
    !    05 July 2006
    !
    !  Author:
    !
    !    John Burkardt
    !
    !  Reference:
    !
    !    Paul Bratley, Bennett Fox, Linus Schrage,
    !    A Guide to Simulation,
    !    Springer Verlag, pages 201-202, 1983.
    !
    !    Bennett Fox,
    !    Algorithm 647:
    !    Implementation and Relative Efficiency of Quasirandom
    !    Sequence Generators,
    !    ACM Transactions on Mathematical Software,
    !    Volume 12, Number 4, pages 362-376, 1986.
    !
    !    Peter Lewis, Allen Goodman, James Miller
    !    A Pseudo-Random Number Generator for the System/360,
    !    IBM Systems Journal,
    !    Volume 8, pages 136-143, 1969.
    !
    !  Parameters:
    !
    !    Input, integer ( kind = 4 ) N, the number of entries in the vector.
    !
    !    Input/output, integer ( kind = 4 ) SEED, the "seed" value, which
    !    should NOT be 0.  On output, SEED has been updated.
    !
    !    Output, real ( kind = 8 ) R(N), the vector of pseudorandom values.
    !
    IMPLICIT NONE

    INTEGER,INTENT(IN)::n

    INTEGER::i
    INTEGER::k
    INTEGER,INTENT(INOUT)::seed
    REAL(KIND(1d0)),DIMENSION(n)::r

    IF(seed.EQ.0)THEN
       WRITE( *, '(a)' ) ' '
       WRITE( *, '(a)' ) 'R8VEC_UNIFORM_01 - Fatal error!'
       WRITE( *, '(a)' ) '  Input value of SEED = 0.'
       STOP
    ENDIF

    DO i = 1, n

       k = seed / 127773

       seed = 16807 * ( seed - k * 127773 ) - k * 2836

       IF(seed.LT.0)THEN
          seed=seed+HUGE(seed)
       ENDIF

       r(i)=DBLE(seed) * 4.656612875D-10

    ENDDO

    RETURN
  END SUBROUTINE r8vec_uniform_01

  SUBROUTINE romberg_nd(func,a,b,dim_num,sub_num,it_max,tol,res,&
       ind,eval_num)
    !*****************************************************************************
    !
    !! ROMBERG_ND estimates a multidimensional integral using Romberg integration.
    !
    !  Discussion:
    !
    !    The routine uses a Romberg method based on the midpoint rule.
    !
    !    In the reference, this routine is called "NDIMRI".
    !
    !    Thanks to Barak Bringoltz for pointing out problems in a previous
    !    FORTRAN90 implementation of this routine.
    !
    !  Licensing:
    !
    !    This code is distributed under the GNU LGPL license.
    !
    !  Modified:
    !
    !    11 September 2006
    !
    !  Author:
    !
    !    Original FORTRAN77 version by Philip Davis, Philip Rabinowitz.
    !    FORTRAN90 version by John Burkardt
    !
    !  Reference:
    !
    !    Philip Davis, Philip Rabinowitz,
    !    Methods of Numerical Integration,
    !    Second Edition,
    !    Dover, 2007,
    !    ISBN: 0486453391,
    !    LC: QA299.3.D28.
    !
    !  Parameters:
    !
    !    Input, real ( kind = 8 ), external FUNC, a routine which evaluates
    !    the function to be integrated, of the form:
    !      function func ( dim_num, x )
    !      integer ( kind = 4 ) dim_num
    !      real ( kind = 8 ) func
    !      real ( kind = 8 ) x(dim_num)
    !      func = ...
    !      return
    !      end
    !
    !    Input, real ( kind = 8 ) A(DIM_NUM), B(DIM_NUM), the integration limits.
    !
    !    Input, integer ( kind = 4 ) DIM_NUM, the spatial dimension.
    !
    !    Input, integer ( kind = 4 ) SUB_NUM(DIM_NUM), the number of subintervals
    !    into which the I-th integration interval (A(I), B(I)) is
    !    initially subdivided.  SUB_NUM(I) must be greater than 0.
    !
    !    Input, integer ( kind = 4 ) IT_MAX, the maximum number of iterations to
    !    be performed.  The number of function evaluations on
    !    iteration J is at least J**DIM_NUM, which grows very rapidly.
    !    IT_MAX should be small!
    !
    !    Input, real ( kind = 8 ) TOL, an error tolerance for the approximation
    !    of the integral.
    !
    !    Output, real ( kind = 8 ) RES, the approximate value of the integral.
    !
    !    Output, integer ( kind = 4 ) IND, error return flag.
    !    IND = -1 if the error tolerance could not be achieved.
    !    IND = 1 if the error tolerance was achieved.
    !
    !    Output, integer ( kind = 4 ) EVAL_NUM, the number of function evaluations.
    !
    !  Local Parameters:
    !
    !    Local, integer ( kind = 4 ) IWORK(DIM_NUM), a pointer used to generate 
    !    all the points X in the product region.
    !
    !    Local, integer ( kind = 4 ) IWORK2(IT_MAX), a counter of the number of 
    !    points used at each step of the Romberg iteration.
    !
    !    Local, integer ( kind = 4 ) SUB_NUM2(DIM_NUM), the number of subintervals 
    !    used in each direction, a refinement of the user's input SUB_NUM.
    !
    !    Local, real ( kind = 8 ) TABLE(IT_MAX), the difference table.
    !
    !    Local, real ( kind = 8 ) X(DIM_NUM), an evaluation point.
    !
    IMPLICIT NONE

    INTEGER,INTENT(IN)::it_max
    INTEGER,INTENT(IN)::dim_num

    REAL(KIND(1d0)),DIMENSION(dim_num),INTENT(IN)::a,b
    REAL(KIND(1d0))::en
    INTEGER,INTENT(OUT)::eval_num
    REAL(KIND(1d0))::factor
    REAL(KIND(1d0)),EXTERNAL::func
    INTEGER::i
    INTEGER,INTENT(OUT)::ind
    INTEGER::it
    INTEGER,DIMENSION(dim_num)::iwork
    INTEGER,DIMENSION(it_max)::iwork2
    INTEGER::kdim
    INTEGER::ll
    INTEGER,DIMENSION(dim_num),INTENT(IN)::sub_num
    INTEGER,DIMENSION(dim_num)::sub_num2
    REAL(KIND(1d0)),INTENT(OUT)::res
    REAL(KIND(1d0))::result_old
    REAL(KIND(1d0))::rnderr
    REAL(KIND(1d0))::submid
    REAL(KIND(1d0))::sum1
    REAL(KIND(1d0))::weight
    REAL(KIND(1d0)),DIMENSION(it_max)::table
    REAL(KIND(1d0))::tol
    REAL(KIND(1d0)),DIMENSION(dim_num)::x

    eval_num = 0

    IF(dim_num.LT.1)THEN
       WRITE( *, '(a)' ) ' '
       WRITE( *, '(a)' ) 'ROMBERG_ND - Fatal error!'
       WRITE( *, '(a,i8)' ) '  DIM_NUM is less than 1.  DIM_NUM = ', dim_num
       STOP
    ENDIF

    IF(it_max.LT.1)THEN
       WRITE( *, '(a)' ) ' '
       WRITE( *, '(a)' ) 'ROMBERG_ND - Fatal error!'
       WRITE( *, '(a,i8)' ) '  IT_MAX is less than 1.  IT_MAX = ', it_max
       STOP
    ENDIF

    DO i = 1, dim_num
       IF(sub_num(i).LE.0)THEN
          WRITE( *, '(a)' ) ' '
          WRITE( *, '(a)' ) 'ROMBERG_ND - Fatal error!'
          WRITE( *, '(a)' ) '  SUB_NUM(I) is less than 1.'
          WRITE( *, '(a,i8)' ) '  for I = ', i
          WRITE( *, '(a,i8)' ) '  SUB_NUM(I) = ', sub_num(i)
          STOP
       ENDIF
    ENDDO

    ind = 0
    rnderr = EPSILON( 1.0D+00 )
    iwork2(1) = 1
    sub_num2(1:dim_num) = sub_num(1:dim_num)

    IF(1.LT.it_max)THEN
       iwork2(2) = 2
    ENDIF

    it = 1

    DO

       sum1 = 0.0D+00

       weight = PRODUCT(( b(1:dim_num) - a(1:dim_num) ) &
            /DBLE(sub_num2(1:dim_num)))
       !
       !  Generate every point X in the product region, and evaluate F(X).
       !
       iwork(1:dim_num) = 1

       DO

          x(1:dim_num) = &
               (DBLE(2*sub_num2(1:dim_num)-2*iwork(1:dim_num)+1) &
               * a(1:dim_num) &
               + DBLE(2*iwork(1:dim_num)-1) &
               * b(1:dim_num)) &
               /DBLE(2*sub_num2(1:dim_num))

          sum1 = sum1 + func(dim_num,x)
          eval_num = eval_num + 1

          kdim = dim_num

          DO WHILE(0.LT.kdim)

             IF(iwork(kdim).LT.sub_num2(kdim))THEN
                iwork(kdim) = iwork(kdim) + 1
                EXIT
             ENDIF

             iwork(kdim) = 1

             kdim = kdim - 1

          ENDDO

          IF(kdim.EQ.0)THEN
             EXIT
          ENDIF

       ENDDO
       !
       !  Done with summing.
       !
       table(it) = weight * sum1

       IF(it.LE.1)THEN

          res=table(1)
          result_old=res

          IF(it_max.LE.it)THEN
             ind = 1
             EXIT
          ENDIF

          it = it + 1
          
          sub_num2(1:dim_num) = iwork2(it) * sub_num2(1:dim_num)
          
          CYCLE

       ENDIF
       !
       !  Compute the difference table for Richardson extrapolation.
       !
       DO ll = 2, it
          i = it + 1 - ll
          factor=DBLE( iwork2(i)**2) &
               /DBLE(iwork2(it)**2-iwork2(i)**2)
          table(i) = table(i+1)+(table(i+1)-table(i))*factor
       ENDDO

       res = table(1)
       !
       !  Terminate successfully if the estimated error is acceptable.
       !
       IF(DABS(res-result_old).LE.DABS(res*(tol+rnderr)))THEN
          ind = 1
          EXIT
       ENDIF
       !
       !  Terminate unsuccessfully if the iteration limit has been reached.
       !
       IF(it_max.LE.it)THEN
          ind = -1
          EXIT
       ENDIF
       !
       !  Prepare for another step.
       !
       result_old = res

       it = it + 1

       iwork2(it) = INT(1.5D+00*DBLE(iwork2(it-1)))

       sub_num2(1:dim_num) = &
            INT(1.5D+00*DBLE(sub_num2(1:dim_num)))

    ENDDO

    RETURN
  END SUBROUTINE romberg_nd
  
  SUBROUTINE sample_nd(func,k1,k2,dim_num,est1,err1,dev1,est2, &
       err2,dev2,eval_num)
    !*****************************************************************************
    !
    !! SAMPLE_ND estimates a multidimensional integral using sampling.
    !
    !  Discussion:
    !
    !    This routine computes two sequences of integral estimates, EST1
    !    and EST2, for indices K going from K1 to K2.  These estimates are
    !    produced by the generation of 'random' abscissas in the region.
    !    The process can become very expensive if high accuracy is needed.
    !
    !    The total number of function evaluations is
    !    4*(K1**DIM_NUM+(K1+1)**DIM_NUM+...+(K2-1)**DIM_NUM+K2**DIM_NUM), and K2
    !    should be chosen so as to make this quantity reasonable.
    !    In most situations, EST2(K) are much better estimates than EST1(K).
    !
    !  Licensing:
    !
    !    This code is distributed under the GNU LGPL license.
    !
    !  Modified:
    !
    !    01 March 2007
    !
    !  Author:
    !
    !    Original FORTRAN77 version by Philip Davis, Philip Rabinowitz.
    !    FORTRAN90 version by John Burkardt
    !
    !  Reference:
    !
    !    Philip Davis, Philip Rabinowitz,
    !    Methods of Numerical Integration,
    !    Second Edition,
    !    Dover, 2007,
    !    ISBN: 0486453391,
    !    LC: QA299.3.D28.
    !
    !  Parameters:
    !
    !    Input, real ( kind = 8 ), external FUNC, a routine which evaluates
    !    the function to be integrated, of the form:
    !      function func ( dim_num, x )
    !      integer ( kind = 4 ) dim_num
    !      real ( kind = 8 ) func
    !      real ( kind = 8 ) x(dim_num)
    !      func = ...
    !      return
    !      end
    !
    !    Input, integer ( kind = 4 ) K1, the beginning index for the iteration.
    !    1 <= K1 <= K2.
    !
    !    Input, integer ( kind = 4 ) K2, the final index for the iteration.  
    !    K1 <= K2.  Increasing K2 increases the accuracy of the calculation,
    !    but vastly increases the work and running time of the code.
    !
    !    Input, integer ( kind = 4 ) DIM_NUM, the spatial dimension.
    !    1 <= DIM_NUM <= 10.
    !
    !    Output, real ( kind = 8 ) EST1(K2).  Entries K1 through K2 contain
    !    successively better estimates of the integral.
    !
    !    Output, real ( kind = 8 ) ERR1(K2).  Entries K1 through K2 contain
    !    the corresponding estimates of the integration errors.
    !
    !    Output, real ( kind = 8 ) DEV1(K2).  Entries K1 through K2 contain
    !    estimates of the reliability of the the integration.
    !    If consecutive values DEV1(K) and DEV1(K+1) do not differ
    !    by more than 10 percent, then ERR1(K) can be taken as
    !    a reliable upper bound on the difference between EST1(K)
    !    and the true value of the integral.
    !
    !    Output, real ( kind = 8 ) EST2(K2).  Entries K2 through K2 contain
    !    successively better estimates of the integral.
    !
    !    Output, real ( kind = 8 ) ERR2(K2).  Entries K2 through K2 contain
    !    the corresponding estimates of the integration errors.
    !
    !    Output, real ( kind = 8 ) DEV2(K2).  Entries K2 through K2 contain
    !    estimates of the reliability of the the integration.
    !    If consecutive values DEV2(K) and DEV2(K+2) do not differ
    !    by more than 10 percent, then ERR2(K) can be taken as
    !    a reliable upper bound on the difference between EST2(K)
    !    and the true value of the integral.
    !
    !    Output, integer ( kind = 4 ) EVAL_NUM, the number of function evaluations.
    !
    IMPLICIT NONE

    INTEGER::k2
    INTEGER,PARAMETER::dim_max = 10
    INTEGER,INTENT(IN)::dim_num

    REAL(KIND(1d0))::ak
    REAL(KIND(1d0))::ak1
    REAL(KIND(1d0))::akn
    REAL(KIND(1d0)),DIMENSION(dim_max)::al=(/ &
         0.4142135623730950D+00, &
         0.7320508075688773D+00, &
         0.2360679774997897D+00, &
         0.6457513110645906D+00, &
         0.3166247903553998D+00, &
         0.6055512754639893D+00, &
         0.1231056256176605D+00, &
         0.3589989435406736D+00, &
         0.7958315233127195D+00, &
         0.3851648071345040D+00 /)
    SAVE al
    REAL(KIND(1d0))::b
    REAL(KIND(1d0)),DIMENSION(dim_num)::be
    REAL(KIND(1d0))::bk
    REAL(KIND(1d0))::d1
    REAL(KIND(1d0))::d2
    REAL(KIND(1d0)),DIMENSION(k2)::dev1
    REAL(KIND(1d0)),DIMENSION(k2)::dev2
    REAL(KIND(1d0)),DIMENSION(dim_num)::dex
    REAL(KIND(1d0)),DIMENSION(k2)::err1
    REAL(KIND(1d0)),DIMENSION(k2)::err2
    REAL(KIND(1d0)),DIMENSION(k2)::est1
    REAL(KIND(1d0)),DIMENSION(k2)::est2
    INTEGER,INTENT(OUT)::eval_num
    REAL(KIND(1d0)),EXTERNAL::func
    REAL(KIND(1d0))::g
    REAL(KIND(1d0)),DIMENSION(dim_num)::ga
    INTEGER::i
    INTEGER::j
    INTEGER::k
    INTEGER::k1
    INTEGER::key
    LOGICAL::more
    REAL(KIND(1d0)),DIMENSION(dim_num)::p1
    REAL(KIND(1d0)),DIMENSION(dim_num)::p2
    REAL(KIND(1d0)),DIMENSION(dim_num)::p3
    REAL(KIND(1d0)),DIMENSION(dim_num)::p4
    REAL(KIND(1d0))::s1
    REAL(KIND(1d0))::s2
    REAL(KIND(1d0))::t
    REAL(KIND(1d0))::y1
    REAL(KIND(1d0))::y2
    REAL(KIND(1d0))::y3
    REAL(KIND(1d0))::y4

    eval_num = 0
    !
    !  Check input
    !
    IF(dim_num.LT.1)THEN
       WRITE( *, '(a)' ) ' '
       WRITE( *, '(a)' ) 'SAMPLE_ND - Fatal error!'
       WRITE( *, '(a)' ) '  DIM_NUM must be at least 1,'
       WRITE( *, '(a,i8)' ) '  but DIM_NUM = ', dim_num
       STOP
    ENDIF

    IF(dim_max.LT.dim_num)THEN
       WRITE( *, '(a)' ) ' '
       WRITE( *, '(a)' ) 'SAMPLE_ND - Fatal error!'
       WRITE( *, '(a,i8)' ) '  DIM_NUM must be no more than DIM_MAX = ', dim_max
       WRITE( *, '(a,i8)' ) '  but DIM_NUM = ', dim_num
       STOP
    ENDIF

    IF(k1.LT.1)THEN
       WRITE( *, '(a)' ) ' '
       WRITE( *, '(a)' ) 'SAMPLE_ND - Fatal error!'
       WRITE( *, '(a,i8)' ) '  K1 must be at least 1, but K1 = ', k1
       STOP
    ENDIF

    IF(k2.LT.k1)THEN
       WRITE( *, '(a)' ) ' '
       WRITE( *, '(a)' ) 'SAMPLE_ND - Fatal error!'
       WRITE( *, '(a)' ) '  K1 may not be greater than K2, but '
       WRITE( *, '(a,i8)' ) '  K1 = ', k1
       WRITE( *, '(a,i8)' ) '  K2 = ', k2
       STOP
    ENDIF

    be(1:dim_num) = al(1:dim_num)
    ga(1:dim_num) = al(1:dim_num)
    dex(1:dim_num) = 0.0D+00
    
    DO k = k1, k2

       ak = DBLE(k)
       key = 0
       ak1 = ak - 1.1D+00
       s1 = 0.0D+00
       d1 = 0.0D+00
       s2 = 0.0D+00
       d2 = 0.0D+00
       akn = ak**dim_num
       t = DSQRT(ak**dim_num)*ak
       bk = 1.0D+00 / ak
       
       DO

          key = key + 1

          IF(key/= 1)THEN

             key = key - 1
             more = .false.

             DO j = 1, dim_num

                IF( dex(j).LE.ak1 )THEN
                   dex(j) = dex(j) + 1.0D+00
                   more = .TRUE.
                   EXIT
                ENDIF

                dex(j) = 0.0D+00
                
             ENDDO

             IF(.NOT.more )THEN
                EXIT
             ENDIF

          ENDIF

          DO i = 1, dim_num

             b = be(i) + al(i)
             IF(1.0D+00.LT.b)THEN
                b = b - 1.0D+00
             ENDIF

             g = ga(i) + b
             IF(1.0D+00.LT.g)THEN
                g = g - 1.0D+00
             ENDIF

             be(i) = b + al(i)
             IF(1.0D+00.LT.be(i))THEN
                be(i) = be(i) - 1.0D+00
             ENDIF

             ga(i) = be(i) + g
             IF(1.0D+00.LT.ga(i))THEN
                ga(i) = ga(i) - 1.0D+00
             ENDIF

             p1(i) = ( dex(i) + g ) * bk
             p2(i) = ( dex(i) + 1.0D+00 - g ) * bk
             p3(i) = ( dex(i) + ga(i) ) * bk
             p4(i) = ( dex(i) + 1.0D+00 - ga(i) ) * bk

          ENDDO

          y1=func(dim_num,p1)
          eval_num = eval_num + 1
          !
          !  There may be an error in the next two lines,
          !  but oddly enough, that is how the original reads
          !
          y3 = func ( dim_num, p2 )
          eval_num = eval_num + 1
          y2 = func ( dim_num, p3 )
          eval_num = eval_num + 1
          y4 = func ( dim_num, p4 )
          eval_num = eval_num + 1
          
          s1 = s1 + y1 + y2
          d1 = d1 + ( y1 - y2 )**2
          s2 = s2 + y3 + y4
          d2 = d2 + ( y1 + y3 - y2 - y4 )**2
          
       ENDDO

       est1(k) = 0.5D+00 * s1 / akn
       err1(k) = 1.5D+00 *DSQRT(d1)/akn
       dev1(k) = err1(k) * t
       est2(k) = 0.25D+00 * ( s1 + s2 )/akn
       err2(k) = 0.75D+00 * DSQRT( d2 )/akn
       dev2(k) = err2(k) * t * ak

    ENDDO

    RETURN
  ENDSUBROUTINE sample_nd

  SUBROUTINE sum2_nd(func,xtab,weight,order,dim_num,res,eval_num)

    !*****************************************************************************
    !
    !! SUM2_ND estimates a multidimensional integral using a product rule.
    !
    !  Discussion:
    !
    !    The routine uses a product rule supplied by the user.
    !
    !    The region may be a product of any combination of finite,
    !    semi-infinite, or infinite intervals.
    !
    !    For each factor in the region, it is assumed that an integration
    !    rule is given, and hence, the region is defined implicitly by
    !    the integration rule chosen.
    !
    !  Licensing:
    !
    !    This code is distributed under the GNU LGPL license.
    !
    !  Modified:
    !
    !    25 February 2007
    !
    !  Author:
    !
    !    Original FORTRAN77 version by Philip Davis, Philip Rabinowitz.
    !    FORTRAN90 version by John Burkardt
    !
    !  Reference:
    !
    !    Philip Davis, Philip Rabinowitz,
    !    Methods of Numerical Integration,
    !    Second Edition,
    !    Dover, 2007,
    !    ISBN: 0486453391,
    !    LC: QA299.3.D28.
    !
    !  Parameters:
    !
    !    Input, real ( kind = 8 ), external FUNC, a routine which evaluates
    !    the function to be integrated, of the form:
    !      function func ( dim_num, x )
    !      integer ( kind = 4 ) dim_num
    !      real ( kind = 8 ) func
    !      real ( kind = 8 ) x(dim_num)
    !      func = ...
    !      return
    !      end
    !
    !    Input, real ( kind = 8 ) XTAB(DIM_NUM,ORDER_MAX).  XTAB(I,J) is the
    !    I-th abscissa of the J-th rule.
    !
    !    Input, real ( kind = 8 ) WEIGHT(DIM_NUM,ORDER_MAX).  WEIGHT(I,J) is the
    !    I-th weight for the J-th rule.
    !
    !    Input, integer ( kind = 4 ) ORDER(DIM_NUM).  ORDER(I) is the number of
    !    abscissas to be used in the J-th rule.  ORDER(I) must be
    !    greater than 0 and less than or equal to ORDER_MAX.
    !
    !    Input, integer ( kind = 4 ) DIM_NUM, the spatial dimension.
    !
    !    Output, real ( kind = 8 ) RES, the approximate value of the integral.
    !
    !    Output, integer ( kind = 4 ) EVAL_NUM, the number of function evaluations.
    !
    IMPLICIT NONE

    INTEGER,INTENT(IN)::dim_num

    INTEGER,INTENT(OUT)::eval_num
    REAL(KIND(1d0)),EXTERNAL::func
    INTEGER::i
    INTEGER,DIMENSION(dim_num)::iwork
    INTEGER::k
    INTEGER::m1
    INTEGER,DIMENSION(dim_num)::order
    REAL(KIND(1d0)),INTENT(OUT)::res
    REAL(KIND(1d0))::w1
    REAL(KIND(1d0)),DIMENSION(dim_num,*)::weight
    REAL(KIND(1d0)),DIMENSION(dim_num)::work
    REAL(KIND(1d0)),DIMENSION(dim_num,*)::xtab
    !
    !  Default values.
    !
    res = 0.0D+00
    eval_num = 0

    IF(dim_num.LT.1)THEN
       WRITE( *, '(a)' ) ' '
       WRITE( *, '(a)' ) 'SUM2_ND - Fatal error!'
       WRITE( *, '(a)' ) '  DIM_NUM < 1'
       WRITE( *, '(a,i8)' ) '  DIM_NUM = ', dim_num
       STOP
    ENDIF

    DO i = 1, dim_num

       IF(order(i).LT.1)THEN
          WRITE( *, '(a)' ) ' '
          WRITE( *, '(a)' ) 'SUM2_ND - Fatal error!'
          WRITE( *, '(a)' ) '  ORDER(I) < 1.'
          WRITE( *, '(a,i8)' ) '  For I = ', i
          WRITE( *, '(a,i8)' ) '  ORDER(I) = ', order(i)
          STOP
       ENDIF

    ENDDO

    iwork(1:dim_num) = 1

    DO

       k = 1
       
       w1 = 1.0D+00
       DO i = 1, dim_num
          m1 = iwork(i)
          work(i) = xtab(i,m1)
          w1 = w1 * weight(i,m1)
       ENDDO

       res = res + w1 * func ( dim_num, work )
       eval_num = eval_num + 1

       DO WHILE(iwork(k).EQ.order(k))

          iwork(k) = 1
          k = k + 1
          
          IF(dim_num.LT.k)THEN
             RETURN
          ENDIF

       ENDDO

       iwork(k) = iwork(k) + 1

    ENDDO

    RETURN
  END SUBROUTINE sum2_nd

  SUBROUTINE timestamp()
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
    !  Parameters:
    !
    !    None
    !
    IMPLICIT NONE

    CHARACTER(len = 8)::ampm
    INTEGER::d
    INTEGER::h
    INTEGER::m
    INTEGER::mm
    CHARACTER(len=9),PARAMETER,DIMENSION(12)::month=(/ &
         'January  ', 'February ', 'March    ', 'April    ', &
         'May      ', 'June     ', 'July     ', 'August   ', &
         'September', 'October  ', 'November ', 'December ' /)
    INTEGER::n
    INTEGER::s
    INTEGER,DIMENSION(8)::values
    INTEGER::y

    CALL date_and_time(values=values)

    y = values(1)
    m = values(2)
    d = values(3)
    h = values(5)
    n = values(6)
    s = values(7)
    mm = values(8)
    
    IF(h.LT.12)THEN
       ampm = 'AM'
    ELSEIF(h.EQ.12)THEN
       IF(n.EQ.0.AND.s.EQ.0)THEN
          ampm = 'Noon'
       ELSE
          ampm = 'PM'
       ENDIF
    ELSE
       h = h - 12
       IF(h.LT.12)THEN
          ampm = 'PM'
       ELSEIF(h.EQ.12)THEN
          IF(n.EQ.0.AND.s.EQ.0)THEN
             ampm = 'Midnight'
          ELSE
             ampm = 'AM'
          ENDIF
       ENDIF
    ENDIF

    WRITE( *, '(i2,1x,a,1x,i4,2x,i2,a1,i2.2,a1,i2.2,a1,i3.3,1x,a)' ) &
         d, TRIM(month(m)), y, h, ':', n, ':', s, '.', mm, TRIM(ampm)

    RETURN
  END SUBROUTINE timestamp

  SUBROUTINE tuple_next(m1,m2,n,rank,x)
    !*****************************************************************************
    !
    !! TUPLE_NEXT computes the next element of a tuple space.
    !
    !  Discussion:
    !
    !    The elements are N vectors.  Each entry is constrained to lie
    !    between M1 and M2.  The elements are produced one at a time.
    !    The first element is
    !      (M1,M1,...,M1),
    !    the second element is
    !      (M1,M1,...,M1+1),
    !    and the last element is
    !      (M2,M2,...,M2)
    !    Intermediate elements are produced in lexicographic order.
    !
    !  Example:
    !
    !    N = 2, M1 = 1, M2 = 3
    !
    !    INPUT        OUTPUT
    !    -------      -------
    !    Rank  X      Rank   X
    !    ----  ---    -----  ---
    !    0     * *    1      1 1
    !    1     1 1    2      1 2
    !    2     1 2    3      1 3
    !    3     1 3    4      2 1
    !    4     2 1    5      2 2
    !    5     2 2    6      2 3
    !    6     2 3    7      3 1
    !    7     3 1    8      3 2
    !    8     3 2    9      3 3
    !    9     3 3    0      0 0
    !
    !  Licensing:
    !
    !    This code is distributed under the GNU LGPL license.
    !
    !  Modified:
    !
    !    18 April 2003
    !
    !  Author:
    !
    !    John Burkardt
    !
    !  Parameters:
    !
    !    Input, integer ( kind = 4 ) M1, M2, the minimum and maximum entries.
    !
    !    Input, integer ( kind = 4 ) N, the number of components.
    !
    !    Input/output, integer ( kind = 4 ) RANK, counts the elements.
    !    On first call, set RANK to 0.  Thereafter, the output value of RANK
    !    will indicate the order of the element returned.  When there are no
    !    more elements, RANK will be returned as 0.
    !
    !    Input/output, integer ( kind = 4 ) X(N), on input the previous tuple.
    !    On output, the next tuple.
    !
    IMPLICIT NONE

    INTEGER,INTENT(IN)::n

    INTEGER::i
    INTEGER::m1
    INTEGER::m2
    INTEGER::rank
    INTEGER,DIMENSION(n),INTENT(INOUT)::x

    IF(m2.LT.m1)THEN
       rank = 0
       RETURN
    ENDIF

    IF(rank.LE.0)THEN

       x(1:n) = m1
       rank = 1
       
    ELSE

       rank = rank + 1
       i = n

       DO

          IF(x(i).LT.m2)THEN
             x(i) = x(i) + 1
             EXIT
          ENDIF

          x(i) = m1

          IF(i.EQ.1)THEN
             rank = 0
             x(1:n) = m1
             EXIT
          ENDIF

          i = i - 1

       ENDDO

    ENDIF

    RETURN
  END SUBROUTINE tuple_next

  ! For the infinite integrals of the highly oscilatory functions
  ! let us use the modfied W transform proposed by Sidi (https://www.jstor.org/stable/2008589)
  ! Original reference: "A User-Friendly Extrapolation Method for Oscillatory Infinite Integrals" By Avram Sidi
  ! Also see eq.(8) in "Evaluating infinite integrals involving bessel functions of arbitrary order", By S. Lucas and H. Stone
  ! Journal of Computational and Applied Mathematics 64 (1995) 217-231
  ! To calculate Integrate[g(x),{x,a,Infinity}]
  ! the zeros (or close to zeros) > a are x_0, x_1, ..., x_{p+2}, while x_{-1}=a
  ! psi_s = Integrate[g(x),{x,x_s, x_{s+1}}] = F_{s+1} - F_s, s=-1,..., p+1
  ! F_s = Integrate[g(x),{x,a,x_s}], s=-1,..., p+1
  ! M_{-1}^{(s)}=F_s/psi_s, N_{-1}^{(s)}=1/psi_s
  ! M_p^{(s)}=(M_{p-1}^{(s)}-M_{p-1}^{(s+1)})/(x_s^(-1)-x_{s+p+1}^{-1})
  ! N_p^{(s)}=(N_{p-1}^{(s)}-N_{p-1}^{(s+1)})/(x_s^(-1)-x_{s+p+1}^{-1})
  ! The integral can be well approximated by M_p^{(0)}/N_p^{(0)}
  RECURSIVE FUNCTION mWT_Mfun(p,s,nx,xs,psis,Fs) RESULT(Mfun)
    ! M_p^{(s)}
    IMPLICIT NONE
    INTEGER,INTENT(IN)::p,s,nx
    REAL(KIND(1d0)),DIMENSION(-1:nx),INTENT(IN)::xs
    REAL(KIND(1d0)),DIMENSION(-1:nx-1),INTENT(IN)::psis,Fs
    REAL(KIND(1d0))::Mfun
    REAL(KIND(1d0))::xxs1,xxs2
    IF(s.LT.-1)THEN
       WRITE(*,*)"Error: s<-1 in mWT_Mfun"
       STOP
    ENDIF
    IF(p.LE.-1)THEN
       Mfun=Fs(s)/psis(s)
       RETURN
    ENDIF
    xxs1=xs(s)
    xxs2=xs(s+p+1)
    Mfun=(mWT_Mfun(p-1,s,nx,xs,psis,Fs)-mWT_Mfun(p-1,s+1,nx,xs,psis,Fs))/(xxs1**(-1)-xxs2**(-1))
    RETURN
  END FUNCTION mWT_Mfun

  RECURSIVE FUNCTION mWT_Nfun(p,s,nx,xs,psis,Fs) RESULT(Nfun)
    ! N_p^{(s)}
    IMPLICIT NONE
    INTEGER,INTENT(IN)::p,s,nx
    REAL(KIND(1d0)),DIMENSION(-1:nx),INTENT(IN)::xs
    REAL(KIND(1d0)),DIMENSION(-1:nx-1),INTENT(IN)::psis,Fs
    REAL(KIND(1d0))::Nfun
    REAL(KIND(1d0))::xxs1,xxs2
    IF(s.LT.-1)THEN
       WRITE(*,*)"Error: s<-1 in mWT_Nfun"
       STOP
    ENDIF
    IF(p.LE.-1)THEN
       Nfun=psis(s)**(-1)
       RETURN
    ENDIF
    xxs1=xs(s)
    xxs2=xs(s+p+1)
    Nfun=(mWT_Nfun(p-1,s,nx,xs,psis,Fs)-mWT_Nfun(p-1,s+1,nx,xs,psis,Fs))/(xxs1**(-1)-xxs2**(-1))
    RETURN
  END FUNCTION mWT_Nfun

END MODULE NINTLIB
