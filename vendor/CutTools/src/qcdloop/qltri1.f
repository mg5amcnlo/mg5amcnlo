      subroutine qltri1(psq,musq,Ival)
      implicit none
      double precision psq,musq
      double complex qllnrat,Ival(-2:0),wlogm
      wlogm=qllnrat(musq,-psq)
      Ival(-2)=dcmplx(1d0/psq)
      Ival(-1)=Ival(-2)*wlogm
      Ival(0)=dcmplx(0.5d0)*Ival(-2)*wlogm**2
      return
      end
