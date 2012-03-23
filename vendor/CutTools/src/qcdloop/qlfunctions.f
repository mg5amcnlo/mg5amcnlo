************************************************************************
*     Author: R.K. Ellis & GZ                                               *
************************************************************************

      double complex function qlL0(x,y)
      implicit none
      include 'qlconstants.f'
      double complex qllnrat
      double precision x,y,denom
      denom=one-x/y
      if (abs(denom) .lt. 1d-7) then
      qlL0=-cone-dcmplx(denom*(half+denom/3d0))
      else
      qlL0=qllnrat(x,y)/dcmplx(denom)
      endif
      return
      end

      double complex function qlL1(x,y)
      implicit none
      include 'qlconstants.f'
      double precision x,y,denom
      double complex qlL0
      denom=one-x/y
      if (abs(denom) .lt. 1d-7) then
      qlL1=-half*cone-dcmplx(denom/3d0*(one+0.75d0*denom))
      else
      qlL1=(qlL0(x,y)+cone)/dcmplx(denom)
      endif
      return
      end

      double complex function qlL2(x,y)
      implicit none
      include 'qlconstants.f'
      double complex qllnrat
      double precision x,y,r,denom
      r=x/y
      denom=one-r
      if (abs(denom) .lt. 1d-7) then
      qlL2=(dcmplx(10d0)+denom*(dcmplx(15d0)+dcmplx(18d0)*denom))
     . /dcmplx(60d0)
      else
      qlL2=(qllnrat(x,y)-dcmplx(0.5d0*(r-1d0/r)))/dcmplx(denom)**3
      endif  
      return
      end

      double complex function qlLsm1(x1,y1,x2,y2)
      implicit none
      include 'qlconstants.f'
      double precision x1,x2,y1,y2
      double complex qlLnrat,qlLi2omrat

      qlLsm1=qlLi2omrat(x1,y1)+qlLi2omrat(x2,y2)
     . +qllnrat(x1,y1)*qllnrat(x2,y2)-dcmplx(pisqo6)
      return
      end


      double complex function qlLsm1_2mht(s,t,p1sq,p2sq)
      implicit none
      include 'qlconstants.f'
      double precision s,t,p1sq,p2sq
      double complex qlLnrat,qlLi2omrat

      qlLsm1_2mht=
     & -qlLi2omrat(-p1sq,-t)
     & -qlLi2omrat(-p2sq,-t)
     & +half*(qllnrat(-s,-p1sq)*qllnrat(-s,-p2sq)-qllnrat(-s,-t)**2)
      return
      end


      double complex function qlLsm1_2me(s,t,p1sq,p3sq)
      implicit none
      include 'qlconstants.f'
      double precision s,t,p1sq,p3sq
      double complex qlLnrat,Li2(5),qlLi2omrat,qlLi2omx2

      Li2(1)=qlLi2omrat(-p1sq,-s)
      Li2(2)=qlLi2omrat(-p1sq,-t)
      Li2(3)=qlLi2omrat(-p3sq,-s)
      Li2(4)=qlLi2omrat(-p3sq,-t)
      Li2(5)=qlLi2omx2(-p1sq,-p3sq,-s,-t)
      qlLsm1_2me=Li2(5)-Li2(1)-Li2(2)-Li2(3)-Li2(4)
     . -half*qlLnrat(-s,-t)**2
      return
      end

