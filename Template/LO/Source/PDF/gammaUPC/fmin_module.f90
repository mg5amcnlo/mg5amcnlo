!*****************************************************************************************
!>
!  Module for [[fmin]] 1D derative-free function minimizer.
!
!### License
!  * [BSD-3](https://github.com/jacobwilliams/fmin/blob/master/LICENSE)
! From https://github.com/jacobwilliams/fmin
!
module fmin_module

  implicit none
  
  private
  
  public :: fmin

contains
  !*****************************************************************************************

  !*****************************************************************************************
  !>
  !  An approximation x to the point where `f` attains a minimum on
  !  the interval `(ax,bx)` is determined.
  !
  !  the method used is a combination of golden section search and
  !  successive parabolic interpolation. convergence is never much slower
  !  than that for a fibonacci search. if `f` has a continuous second
  !  derivative which is positive at the minimum (which is not at `ax` or
  !  `bx`), then convergence is superlinear, and usually of the order of
  !  about 1.324.
  !
  !  the function `f` is never evaluated at two points closer together
  !  than `eps*abs(fmin) + (tol/3)`, where `eps` is approximately the square
  !  root of the relative machine precision. if `f` is a unimodal
  !  function and the computed values of `f` are always unimodal when
  !  separated by at least `eps*abs(x) + (tol/3)`, then fmin approximates
  !  the abcissa of the global minimum of `f` on the interval `ax,bx` with
  !  an error less than `3*eps*abs(fmin) + tol`. if `f` is not unimodal,
  !  then `fmin` may approximate a local, but perhaps non-global, minimum to
  !  the same accuracy.
  !x
  !### Reference
  !  * Richard brent, "algorithms for minimization without derivatives",
  !    prentice - hall, inc. (1973).
  !
  !### See also
  !  * [fmin from Netlib](http://www.netlib.org/fmm/fmin.f)

  function fmin(f,ax,bx,tol) result(xmin)
    
    implicit none
    
    real(kind(1d0)),external:: f    !! the function to minimize
    real(kind(1d0)),intent(in) :: ax   !! left endpoint of initial interval
    real(kind(1d0)),intent(in) :: bx   !! right endpoint of initial interval
    real(kind(1d0)),intent(in) :: tol  !! desired length of the interval of
    !! uncertainty of the final result (>=0)
    real(kind(1d0)):: xmin !! abcissa approximating the point where
    !! f attains a minimum

    real(kind(1d0))::a,b,d,e,xm,p,q,r,tol1,tol2,u,v,w
    real(kind(1d0))::fu,fv,fw,fx,x

    real(kind(1d0)),parameter::c=0.381966011250105151795413165634d0
    ! c = (3.0_wp-sqrt(5.0_wp))/2.0_wp  !! squared inverse of golden ratio
    real(kind(1d0)),parameter:: half=0.5d0
    real(kind(1d0))::sqrteps
    sqrteps=dsqrt(epsilon(1.0d0))

    ! initialization

    a = ax
    b = bx
    v = a + c*(b - a)
    w = v
    x = v
    e = 0.0d0
    fx = f(x)
    fv = fx
    fw = fx

    do    !  main loop starts here

       xm = half*(a + b)
       tol1 = sqrteps*dabs(x) + tol/3.0d0
       tol2 = 2.0d0*tol1

       !  check stopping criterion

       if (dabs(x - xm).le.(tol2 - half*(b - a))) then
          ! write(*,*) 'x             = ', x
          ! write(*,*) 'xm            = ', xm
          ! write(*,*) 'abs(x - xm)   = ', abs(x - xm)
          ! write(*,*) 'tol2          = ', tol2
          ! write(*,*) 'half*(b - a)  = ', half*(b - a)
          exit
       end if

       ! is golden-section necessary

       if (dabs(e).le.tol1) then

          !  a golden-section step

          if (x.ge.xm) then
             e = a - x
          else
             e = b - x
          end if
          d = c*e

       else

          !  fit parabola
          
          r = (x - w)*(fx - fv)
          q = (x - v)*(fx - fw)
          p = (x - v)*q - (x - w)*r
          q = 2.0d0*(q - r)
          if (q.gt.0.0d0)p = -p
          q =  dabs(q)
          r = e
          e = d

          !  is parabola acceptable

          if ((dabs(p).ge.dabs(half*q*r)).or.(p.le.q*(a-x)).or.(p.ge.q*(b-x))) then

             !  a golden-section step

             if (x.ge.xm) then
                e = a - x
             else
                e = b - x
             end if
             d = c*e

          else

             !  a parabolic interpolation step
             
             d = p/q
             u = x + d

             !  f must not be evaluated too close to ax or bx

             if (((u - a).lt.tol2).or.((b - u).lt.tol2))d=sign(tol1,xm-x)

          end if

       end if

       !  f must not be evaluated too close to x

       if (dabs(d).ge.tol1) then
          u = x + d
       else
          u = x + sign(tol1, d)
       end if
       fu = f(u)

       !  update a, b, v, w, and x

       if (fu.le.fx) then
          if (u.ge.x) a = x
          if (u.lt.x) b = x
          v = w
          fv = fw
          w = x
          fw = fx
          x = u
          fx = fu
       else
          if (u.lt.x) a = u
          if (u.ge.x) b = u
          if (fu.le.fw.or.w.eq.x) then
             v = w
             fv = fw
             w = u
             fw = fu
          else if (fu.le.fv.or.v.eq.x.or.v.eq.w)then
             v = u
             fv = fu
          end if
       end if
       
    end do    !  end of main loop
    
    xmin = x
    
  end function fmin
  !*****************************************************************************************
  
  !*****************************************************************************************
end module fmin_module
!*****************************************************************************************
