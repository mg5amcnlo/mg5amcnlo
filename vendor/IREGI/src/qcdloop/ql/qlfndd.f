      double complex function qlfndd(n,x,iep)
C----Implementation of DD Eq. 4.11
C----%\cite{Denner:2005nn}
C----\bibitem{Denner:2005nn}
C----  A.~Denner and S.~Dittmaier,
C----  %``Reduction schemes for one-loop tensor integrals,''
C----  Nucl.\ Phys.\  B {\bf 734}, 62 (2006)
C----  [arXiv:hep-ph/0509141].
C----  %%CITATION = NUPHA,B734,62;%%

      implicit none
      include 'qlconstants.f'
      integer j,n,infty
      double complex xm1,x,cln
      double precision iep
      logical qlzero
      parameter(infty=16) ! number of terms in sum
      
      xm1=x-cone
      if (abs(x) .lt. 10d0) then
        if (qlzero(abs(x-cone))) then
          qlfndd=czip
        else
          qlfndd=(cone-dcmplx(x**(n+1)))*(cln(xm1,iep)-cln(x,iep))
        endif
        do j=0,n
          qlfndd=qlfndd-dcmplx(x**(n-j))/dfloat(j+1)
        enddo
      elseif (abs(x) .ge. 10d0) then
        qlfndd=cln(cone-cone/x,iep)
        do j=n+1,n+infty
          qlfndd=qlfndd+dcmplx(x**(n-j))/dfloat(j+1)
        enddo
      endif
           
      return
      end
