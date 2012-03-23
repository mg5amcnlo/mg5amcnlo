      double complex function qlcLi2omx3(z1,z2,z3,ieps1,ieps2,ieps3)
C     Calculate Li[2](1-(z1+ieps1)*(z2+ieps2)*(z3+ieps3)) 
C     Using +cLi2(1-z1*z2*z3)                           
C --- for |z1*z2*z3|<1
C     and   -cLi2(1-1/(z1*z2*z3))-1/2*(ln(z1)+ln(z2)+ln(z3))^2 
C---- for |z1*z2*z3|>1
      implicit none
      include 'qlconstants.f'
      double precision ieps1,ieps2,ieps3,ieps
      double complex arg,lnarg,lnomarg,prod,denspence,z1,z2,z3,cln
      logical qlzero
      arg=z1*z2*z3
      if (qlzero(dimag(arg))) ieps=
     .sign(one,dreal(z2*z3)*ieps1+dreal(z1*z3)*ieps2+dreal(z1*z2)*ieps3)

      if (abs(arg) .le. 1d0) then
         if (arg .eq. 0d0 .or. arg .eq. 1d0) then
            prod=0d0
         else
            lnarg=cln(z1,ieps1)+cln(z2,ieps2)+cln(z3,ieps3)
            lnomarg=cln(cone-arg,0d0)
            prod = lnarg*lnomarg
         endif
         qlcLi2omx3=dcmplx(pisqo6)-denspence(dcmplx(arg),ieps)-prod
      elseif (abs(arg) .gt. 1d0) then
         arg=1d0/(z1*z2*z3)
         lnarg=-cln(z1,ieps1)-cln(z2,ieps2)-cln(z3,ieps3)
         lnomarg=cln(cone-arg,0d0)
         qlcLi2omx3=-dcmplx(pisqo6)+denspence(dcmplx(arg),ieps)
     .   +lnarg*lnomarg-0.5d0*lnarg**2
      endif
      return
      end
