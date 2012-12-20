      subroutine qltri4(p2sq,msq,musq,Ival)
      implicit none
      include 'qlconstants.f'
      double precision p2sq,musq,msq,arg2,omarg2,ddilog
      double complex fac,qllnrat,Ival(-2:0),wlog,wlogm,dilog2
      wlog=qllnrat(msq,msq-p2sq)
      wlogm=qllnrat(musq,msq)
      fac=dcmplx(0.5d0/(p2sq-msq))
      arg2=-p2sq/(msq-p2sq)
      omarg2=1d0-arg2

      if (omarg2 .lt. 0d0) then 
      dilog2=dcmplx(pisqo6-ddilog(omarg2))-log(arg2)*wlog
      else 
      dilog2=dcmplx(ddilog(arg2))
      endif

      Ival(-2)=fac
      Ival(-1)=Ival(-2)*wlogm+fac*2d0*wlog
      Ival(0) =-Ival(-2)*0.5d0*wlogm**2+Ival(-1)*wlogm
     . +fac*(wlog**2+dcmplx(pisqo6)-2d0*dilog2)

      return
      end
