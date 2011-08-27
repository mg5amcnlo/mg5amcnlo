      double complex function qlcLi2omx2(z1,z2,ieps1,ieps2)
C     Calculates Li[2](1-(z1+ieps1)*(z2+ieps2)) for complex z1,z2
C     Using +Li2(1-z1*z2)                           for z1*z2<1
C     and   -Li2(1-1/(z1*z2))-1/2*(ln(z1)+ln(z2))^2 for z1*z2>1
      implicit none
      include 'qlconstants.f'
      double precision ieps1,ieps2,ieps
      double complex z1,z2,lnarg,lnomarg,prod,cln,denspence,arg
      arg=z1*z2
      ieps=sign(one,dreal(z2)*ieps1+dreal(z1)*ieps2)
      if (abs(arg) .le. 1d0) then
         if (arg .eq. 0d0 .or. arg .eq. 1d0) then
            prod=0d0
         else
            lnarg=cln(z1,ieps1)+cln(z2,ieps2)
            lnomarg=cln(cone-arg,-ieps)
            prod = lnarg*lnomarg
         endif
         qlcLi2omx2=dcmplx(pisqo6)-denspence(arg,ieps)-prod
      elseif (abs(arg) .gt. 1d0) then
         arg=1d0/(z1*z2)
         lnomarg=cln(cone-arg,-ieps)
         lnarg=-cln(z1,ieps1)-cln(z2,ieps2)
         qlcLi2omx2=-dcmplx(pisqo6)+denspence(arg,ieps)
     .   +lnarg*lnomarg-0.5d0*lnarg**2
      endif
      return
      end
