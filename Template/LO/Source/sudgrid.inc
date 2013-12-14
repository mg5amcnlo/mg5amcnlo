      integer npt2,nx1,nx2
      parameter(npt2=40,nx1=80,nx2=20)
      double precision points(nx2+nx1+npt2,2),sudgrid(nx2,nx1,npt2,-2:5)
      common/sudgrid/points,sudgrid
