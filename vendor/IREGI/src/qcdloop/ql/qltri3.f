      subroutine qltri3(p2sq,p3sq,msq,musq,Ival)
      implicit none
      include 'qlconstants.f'
      double precision p2sq,p3sq,msq,musq,m2sqb,m3sqb,r
      double complex fac,qllnrat,Ival(-2:0),wlog2,wlog3,wlogm,
     . dilog2,dilog3,qlLi2omrat
      

      m2sqb = msq-p2sq
      m3sqb = msq-p3sq
      dilog2= qlLi2omrat(m2sqb,msq)
      dilog3= qlLi2omrat(m3sqb,msq)

      wlog2=qllnrat(m2sqb,msq)
      wlog3=qllnrat(m3sqb,msq)
      wlogm=qllnrat(musq,msq)
      r =(m3sqb-m2sqb)/m2sqb
      
      Ival(-2)=czip
      if (abs(r) .lt. 1d-6) then
      Ival(-1)=dcmplx((1d0-r/2d0)/m2sqb)
      Ival(0)=wlogm-dcmplx((msq+p2sq)/p2sq*wlog2)
      Ival(0)=Ival(0)-chalf*r*((msq**2-2d0*p2sq*msq-p2sq**2)*wlog2
     .+p2sq*(dcmplx(msq+p2sq)+p2sq*wlogm))/p2sq**2
      Ival(0)=Ival(0)/dcmplx(m2sqb)
      else
      fac=dcmplx(1d0/(p2sq-p3sq))
      Ival(-1)=fac*(wlog3-wlog2)
      Ival(0)=Ival(-1)*wlogm
     . +fac*(dilog2-dilog3+(wlog2**2-wlog3**2))
      endif
      return
      end
