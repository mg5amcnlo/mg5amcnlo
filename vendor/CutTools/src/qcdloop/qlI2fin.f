      double complex function qlI2fin(p1sq,m0s,m1s,musq) 
C---- Implementation of the formulae of Denner and Dittmaier
C----%\cite{Denner:2005nn}
C----\bibitem{Denner:2005nn}
C----  A.~Denner and S.~Dittmaier,
C----  %``Reduction schemes for one-loop tensor integrals,''
C----  Nucl.\ Phys.\  B {\bf 734}, 62 (2006)
C----  [arXiv:hep-ph/0509141].
C----  %%CITATION = NUPHA,B734,62;%%
      implicit none
      include 'qlconstants.f'
      double precision p1sq,m0s,m1s,m0sq,m1sq,musq
      double complex xp,xm,b,rt,arg,arg1,qlfndd,cln
      logical qlzero

      m0sq=min(m0s,m1s)
      m1sq=max(m0s,m1s)

      if ((qlzero(abs(p1sq/musq)))
     . .and. (qlzero(abs(m0sq/musq)))
     . .and. (qlzero(abs(m1sq/musq)))) then
      write(6,*) 'setting psq=m0sq=m1sq=0  self-energy to zero'
      write(6,*) 'p1sq,m0sq,m1sq=',p1sq,m0sq,m1sq
      qlI2fin=czip
      return

      elseif (qlzero(m0sq/musq)) then
      arg=dcmplx(1d0-m1sq/p1sq)
      arg1=dcmplx((m1sq-p1sq)/musq)

C---deal with special cases for m0sq=0
C----- (a,0,a)  p1sq=m1sq, DD(4.13)
      if (qlzero(abs(arg1))) then
      qlI2fin=dcmplx(log(musq/m1sq))+ctwo
C----- (0,0,a) 
      elseif (qlzero(abs(p1sq/musq))) then
      qlI2fin=dcmplx(log(musq/m1sq))+cone
C----- (a,0,0)
      elseif (qlzero(abs(m1sq/musq))) then
      qlI2fin=-cln(arg1,-1d0)+ctwo
      else
C----- (a,0,c)
      qlI2fin=-cln(arg1,-1d0)+cone-qlfndd(0,arg,1d0)
      endif
      return
      elseif (qlzero(abs(p1sq/musq))) then
C---deal with special case, p1sq=0
      if (qlzero(abs((m1sq-m0sq)/musq))) then    ! (m1sq = m0sq)
      qlI2fin=dcmplx(log(musq/m0sq))
      else
      xp=dcmplx(m0sq/(m0sq-m1sq))  ! other root is formally infinite
      qlI2fin=dcmplx(log(musq/m0sq))-qlfndd(0,xp,1d0)
      endif     
      else
C----general case, DD (4.8)
      b=dcmplx(m1sq-m0sq-p1sq)
      rt=sqrt(dcmplx((m1sq-m0sq-p1sq)**2-4d0*p1sq*m0sq))
      xp=0.5d0*(-b+rt)/p1sq
      xm=0.5d0*(-b-rt)/p1sq

      qlI2fin=dcmplx(log(musq/m0sq))-qlfndd(0,xp,1d0)-qlfndd(0,xm,-1d0)

      endif
      return 

      end

