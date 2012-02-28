      double complex function qlLi2omx(x1,x2,ieps1,ieps2)
C     Calculate Li[2](1-(x1+ieps1)*(x2+ieps2)) for real x1,x2
C     Using +Li2(1-x1*x2)                           for x1*x2<1
C     and   -Li2(1-1/(x1*x2))-1/2*(ln(x1)+ln(x2))^2 for x1*x2>1
      implicit none
      include 'qlconstants.f'
      double precision x1,x2,arg,ieps1,ieps2,ieps
      double complex ln,lnarg,lnomarg,prod,denspence
      arg=x1*x2
      ieps=sign(one,x2*ieps1+x1*ieps2)
      if (arg .le. 1d0) then
         if (arg.eq. 1d0 .or. arg .eq.0d0) then
            prod=0d0 
         else
            lnarg=ln(x1,ieps1)+ln(x2,ieps2)
            lnomarg=dcmplx(log(1d0-arg),0d0)
            prod=lnarg*lnomarg
         endif
         qlLi2omx=dcmplx(pisqo6)-denspence(dcmplx(arg),ieps)-prod
      elseif (arg .gt. 1d0) then
         arg=1d0/(x1*x2)
         lnarg=-ln(x1,ieps1)-ln(x2,ieps2)
         lnomarg=dcmplx(log(1d0-arg),0d0)
         qlLi2omx=-dcmplx(pisqo6)+denspence(dcmplx(arg),ieps)
     .   +lnarg*lnomarg-0.5d0*lnarg**2
      endif
      return
      end
