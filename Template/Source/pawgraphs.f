      subroutine graph_init
c*************************************************************************
c     Set up graphing
c*************************************************************************
      implicit none
c
c     Local
c
      real xmin,xmax
c
c     Global
c
      real        h(80000)
      common/pawc/h
c-----
c  Begin Code
c-----
c      call hlimit(80000)
c
c     Total
c
c      print*,'Setting up graphs'
c      call hbook1(1,'s hat',100,0.,500.,0.)
      end

      subroutine graph_point2(x,y)
      double precision x,y
      end


      subroutine graph_point(p,dwgt)
c***************************************************************************
c     fill historgrams
c***************************************************************************
      implicit none
c
c     Constants
c
      double precision  pi              , to_deg
      parameter        (pi = 3.1415927d0, to_deg=180d0/pi)
c
c     Arguments
c
      double precision dwgt
      REAL*8 P(0:3,7)
c
c     Local
c
      real*4 wgt
      real*8 ptot(0:3),maxamp, shat
      integer i,iconfig, imax
c
c     Global
c
      include 'maxparticles.inc'
      include 'run.inc'

c
c     External
c
      double precision dot,et,eta,r2
c-----
c  Begin Code
c-----
      wgt=dwgt
c      call hfill(1,real(et(p(0,4))),0.,wgt)
      end

      subroutine graph_store
c*************************************************************************
c     Stores graphs
c*************************************************************************
      implicit none

c-----
c  Begin Code
c-----
c      call hcurve(1,'shat.dat')
c      call hrput(0,'wg.paw','N')
      end




