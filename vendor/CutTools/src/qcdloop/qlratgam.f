      subroutine qlratgam(ratp,ratm,ieps,p3sq,m3sq,m4sq)
*****comment:************************************************************
*									*
*	Calculate the function                                          *
*									*
*		  +p3sq+m4sq-m3sq+sqrt((p3sq+m4sq-m3sq)^2-4*p3sq*m4sq)  *
* rat(p^2,m,mp) = --------------------------------------------------    *
*                 -p3sq+m4sq-m3sq+sqrt((p3sq+m4sq-m3sq)^2-4*p3sq*m4sq)  *
*									*
*	the roots are allowed to be imaginary				*
*									*
***comment:*************************************************************
      implicit none
      double precision p3sq,m3sq,m4sq,ieps
      double complex root,ratp,ratm

      root=dcmplx((p3sq-m3sq+m4sq)**2-4d0*m4sq*p3sq)
      root=sqrt(root)
      ratp=(dcmplx(+p3sq+m4sq-m3sq)+root)/(dcmplx(-p3sq+m4sq-m3sq)+root)
      ratm=(dcmplx(+p3sq+m4sq-m3sq)-root)/(dcmplx(-p3sq+m4sq-m3sq)-root)
      ieps=0d0
      end
