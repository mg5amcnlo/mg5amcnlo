      subroutine qltri5(msq,musq,Ival)
      implicit none
      include 'qlconstants.f'
      double precision msq,musq
      double complex fac,qllnrat,Ival(-2:0),wlogm
      fac=dcmplx(1d0/msq)
      wlogm=qllnrat(musq,msq)
      Ival(-2)=czip
      Ival(-1)=-0.5d0*fac
      Ival(0)=Ival(-1)*wlogm+fac
      return
      end
