      double complex function qlLi2omx2(v,w,x,y)
      implicit none
      include 'qlconstants.f'
c     expression for dilog(1-(v-i*ep)*(w-i*ep)/(x-i*ep)/(y-i*ep)) 
C     for real v,w,x and y
      double precision v,w,x,y,omarg,arg,ddilog
      double complex qllnrat,lnarg,lnomarg,prod
      arg=(v*w)/(x*y)
      omarg=1d0-arg
      if (arg .le. 1d0) then
         if (arg .eq. 0d0 .or. arg .eq. 1d0) then 
            prod=0d0
         else
            lnarg=qllnrat(v,x)+qllnrat(w,y)
            lnomarg=dcmplx(log(omarg))
            prod=lnarg*lnomarg
         endif
         qlLi2omx2=dcmplx(pisqo6-ddilog(arg))-prod
      elseif (arg .gt. 1d0) then
         arg=(x*y)/(v*w)
         lnarg=-qllnrat(v,x)-qllnrat(w,y)
         lnomarg=dcmplx(log(1d0-arg))
         qlLi2omx2=-dcmplx(pisqo6-ddilog(arg))+lnarg*lnomarg
     .   -0.5d0*lnarg**2
      endif
      return 
      end
