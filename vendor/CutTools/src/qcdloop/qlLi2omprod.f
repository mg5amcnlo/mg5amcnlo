      double complex function qlLi2omprod(v,w,z,iep)
      implicit none
      include 'qlconstants.f'
c     expression for dilog(1-(v-i*ep)/(w-i*ep)*(z+iep)) 
C     for real v,w and complex z
      double precision v,w,y,ddilog,rarg,omrarg,iep
      double complex qllnrat,lnarg,lnomarg,prod,arg,omarg,z,cln,
     . denspence
      if (abs(dimag(z)) .lt. 1d-15) then
C----case for real z
      y=dreal(z)
      rarg=v*y/w
      omrarg=1d0-rarg
      if (rarg .le. 1d0) then
         if (rarg .eq. 0d0 .or. rarg .eq. 1d0) then 
            prod=0d0
         else
            lnarg=qllnrat(v,w)+cln(z,iep)
            lnomarg=dcmplx(log(omrarg))
            prod=lnarg*lnomarg
         endif
         qlLi2omprod=dcmplx(pisqo6-ddilog(rarg))-prod
      elseif (rarg .gt. 1d0) then
         rarg=w/(y*v)
         lnarg=-qllnrat(v,w)-cln(z,iep)
         lnomarg=dcmplx(log(1d0-rarg))
         qlLi2omprod=-dcmplx(pisqo6-ddilog(rarg))+lnarg*lnomarg
     .   -0.5d0*lnarg**2
      endif
      else
C----case for complex z
      arg=dcmplx(v/w)*z
      omarg=cone-arg
      if (abs(arg) .le. 1d0) then
         if (abs(arg) .eq. 0d0 .or. abs(arg) .eq. 1d0) then 
            prod=0d0
         else
            lnarg=qllnrat(v,w)+cln(z,iep)
            lnomarg=log(omarg)
            prod=lnarg*lnomarg
         endif
         qlLi2omprod=dcmplx(pisqo6)-denspence(arg,1d0)-prod
      elseif (abs(arg) .gt. 1d0) then
         arg=w/(z*v)
         lnarg=-qllnrat(v,w)-cln(z,iep)
         lnomarg=log(cone-arg)
         qlLi2omprod=-dcmplx(pisqo6)+denspence(arg,1d0)+lnarg*lnomarg
     .   -0.5d0*lnarg**2
      endif

      endif
      return 
      end
