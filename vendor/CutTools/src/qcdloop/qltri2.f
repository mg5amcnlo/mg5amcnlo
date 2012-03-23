      subroutine qltri2(p1sq,p2sq,musq,Ival)
      implicit none
      include 'qlconstants.f'
      double precision p1sq,p2sq,musq,r
      double complex qllnrat,Ival(-2:0),wlog1,wlog2

      wlog1=qllnrat(musq,-p1sq)
      wlog2=qllnrat(musq,-p2sq)
      r = (p2sq-p1sq)/p1sq 

      Ival(-2)=czip
      if (abs(r) .lt. 1d-6) then
      Ival(-1)=dcmplx(1d0/p1sq)*(cone-dcmplx(r/2d0*musq/p1sq))
      Ival(0)=dcmplx(1d0/p1sq)*(wlog1-dcmplx(r/2d0)*(cone+wlog1))
      else   
      Ival(-1)=(wlog1-wlog2)/dcmplx(p1sq-p2sq)
      Ival(0)=chalf*Ival(-1)*(wlog1+wlog2)
      endif
      return
      end
