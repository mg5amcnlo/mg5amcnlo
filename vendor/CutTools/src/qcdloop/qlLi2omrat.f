      double complex function qlLi2omrat(x,y)
      implicit none
      include 'qlconstants.f'
c     expression for dilog(1-(x-i*ep)/(y-i*ep)) for real x and y
c     Hence arguments are typically negative invariants
      double precision x,y,omarg,arg,ddilog
      double complex qllnrat,wlog
      omarg=x/y
      arg=1d0-omarg
      if (arg .gt. 1d0) then
      wlog=qllnrat(x,y)
      qlLi2omrat=dcmplx(pisqo6-ddilog(omarg))-log(arg)*wlog
      else 
      qlLi2omrat=dcmplx(ddilog(arg))
      endif

      return 
      end
