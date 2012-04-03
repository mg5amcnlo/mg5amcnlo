      subroutine qlspencer(zrat1,zrat2,ieps1,ieps2,res)
      implicit none
      include 'qlconstants.f'
      double precision ieps1,ieps2,x1,x2,y1,y2,ieps
      double complex zrat1,zrat2,res,
     . lnarg,lnomarg,prod,denspence,qlLi2omx,arg,cln
      logical qlzero

      x1=dreal(zrat1)
      x2=dreal(zrat2)
      y1=dimag(zrat1)
      y2=dimag(zrat2)
      if (qlzero(y1) .and. qlzero(y2)) then
      res=qlLi2omx(x1,x2,ieps1,ieps2)
      else
      arg=zrat1*zrat2
      ieps=0d0
      if (abs(arg) .le. 1d0) then
         if (arg .eq. 0d0 .or. arg .eq. 1d0) then
            prod=0d0
         else
            lnarg=cln(zrat1,ieps1)+cln(zrat2,ieps2)
            lnomarg=log(cone-arg)
            prod = lnarg*lnomarg
         endif
         res=dcmplx(pisqo6)-denspence(arg,ieps)-prod
      elseif (abs(arg) .gt. 1d0) then
         arg=cone/(zrat1*zrat2)
         lnarg=-cln(zrat1,ieps1)-cln(zrat2,ieps2)
         lnomarg=log(cone-arg)
         res=-dcmplx(pisqo6)+denspence(arg,ieps)
     .   +lnarg*lnomarg-0.5d0*lnarg**2
      endif
      endif
      return
      end       
