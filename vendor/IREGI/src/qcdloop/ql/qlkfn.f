      subroutine qlkfn(cx,ieps,xpi,xm,xmp)
*************************************************************************
*									*
*       RKE & GZ: adapted from ffzkfn routine (19/02/2008)
*	Calculate the K-function given in Eq. 2.7 of                    *
*	%\cite{Beenakker:1988jr}                                        *
*       \bibitem{Beenakker:1988jr}                                      *
*        W.~Beenakker and A.~Denner,                                    *
*  %``INFRARED DIVERGENT SCALAR BOX INTEGRALS WITH APPLICATIONS IN THE  *
*  %ELECTROWEAK STANDARD MODEL,''                                       *
*       Nucl.\ Phys.\  B {\bf 338}, 349 (1990).                         *
*       %%CITATION = NUPHA,B338,349;%%	                        	*
*									*
*			      1-sqrt(1-4*m*mp/(z-(m-mp)^2))		*
*		K(p^2,m,mp) = -----------------------------		*
*			      1+sqrt(1-4*m*mp/(z-(m-mp)^2))		*
*									*
*	and fill x(1) = -K, x(2) = 1+K, x(3) = 1-K			*
*	the roots are allowed to be imaginary				*
*	ieps gives the sign of the imaginary part of -K: 1 -> +i*eps    *
*									*
*************************************************************************
      implicit none
      include 'qlconstants.f'
      LOGICAL qlzero
      DOUBLE PRECISION xpi,xm,xmp,xx1,ieps
      DOUBLE COMPLEX root,invopr,cx(3),rat

      if  ((xm .eq. 0d0) .or. (xmp .eq. 0d0)) then
      write(6,*) 'Error in qlkfn,xm,xmp=',xm,xmp
      stop 
      endif

      xx1 = xpi - (xm-xmp)**2
      rat=dcmplx(xx1/(4d0*xm*xmp))

      if (qlzero(dble(rat))) then
           cx(2) = -2d0*im*sqrt(rat)+2d0*rat
           cx(1) = cone-cx(2)
           cx(3) = ctwo-cx(2) 
      else

           root = sqrt((rat-cone)/rat)
           invopr = cone/(cone+root)
           cx(1) = -invopr**2/rat
           cx(2) = ctwo*invopr
           cx(3) = ctwo*root*invopr
       endif
       ieps = 1d0
       return
       end
