      subroutine jvvtlx(ga,gb,sc,g1,g2,xm,xw,jggs)
c
c- by RF - Feb. 2006	
c
c This subroutine computes an off-shell vector current from the coupling
c of three gauge bosons and a scalar particle. The outgoing vector
c current is given in the Feynman gauge.
c
c input:
c       complex ga(6)       : first  incoming vector (gluon)
c       complex gb(6)       : second incoming vector (gluon)
c       complex sc(3)       : incoming scalar        (Higgs)
c       real    g1          : coupling constant      (QCD)
c       complex g2(2)       : coupling constant      (Higgs Effct. Thr.)
c
c output:
c       complex jggs(6)         : vector current
c not used:
c       real    xm, xw
c
 
      implicit none
c--   dimension of the current set to arbitrary length
      INTEGER DIM
      PARAMETER(DIM=18)
c      include "dimension.inc"
      double complex ga(DIM),gb(DIM),jggs(DIM),sc(DIM),g2(2)
      double complex jggs1(DIM),jggs2(DIM)
      double complex sva,svb,vab,j12(0:3)
      double complex v12,v13,v14,v23,v24,v34,dv
      double precision p1(0:3),p2(0:3),q(0:3),q2,p12q(4)
      double precision g1,xm,xw


      jggs(5) = ga(5) + gb(5) + sc(2)
      jggs(6) = ga(6) + gb(6) + sc(3)

      p1(0) = dble( ga(5))
      p1(1) = dble( ga(6))
      p1(2) = dimag(ga(6))
      p1(3) = dimag(ga(5))

      p2(0) = dble( gb(5))
      p2(1) = dble( gb(6))
      p2(2) = dimag(gb(6))
      p2(3) = dimag(gb(5))

      q(0) = -dble( jggs(5))
      q(1) = -dble( jggs(6))
      q(2) = -dimag(jggs(6))
      q(3) = -dimag(jggs(5))

      q2 = q(0)**2 - q(1)**2 - q(2)**2 - q(3)**2

      dv = g1 * sc(1) /q2

      jggs1(1) = (0D0,0D0)
      jggs1(2) = (0D0,0D0)
      jggs1(3) = (0D0,0D0)
      jggs1(4) = (0D0,0D0)
      jggs2(1) = (0D0,0D0)
      jggs2(2) = (0D0,0D0)
      jggs2(3) = (0D0,0D0)
      jggs2(4) = (0D0,0D0)


      if (g2(1).NE.(0D0,0D0)) then
      vab = ga(1)*gb(1) - ga(2)*gb(2) - ga(3)*gb(3) - ga(4)*gb(4)
      sva =   (p2(0)-q(0)) *ga(1) - (p2(1)-q(1)) *ga(2)
     &      - (p2(2)-q(2)) *ga(3) - (p2(3)-q(3)) *ga(4)
      svb = - (p1(0)-q(0)) *gb(1) + (p1(1)-q(1)) *gb(2)
     &      + (p1(2)-q(2)) *gb(3) + (p1(3)-q(3)) *gb(4)

      jggs1(1)= g2(1)*((p1(0)-p2(0))*vab+sva*gb(1)+svb*ga(1))
      jggs1(2)= g2(1)*((p1(1)-p2(1))*vab+sva*gb(2)+svb*ga(2))
      jggs1(3)= g2(1)*((p1(2)-p2(2))*vab+sva*gb(3)+svb*ga(3))
      jggs1(4)= g2(1)*((p1(3)-p2(3))*vab+sva*gb(4)+svb*ga(4))
      endif
      
      if (g2(2).NE.(0D0,0D0)) then

      p12q(1) = p1(0) + p2(0) + q(0)
      p12q(2) = p1(1) + p2(1) + q(1)
      p12q(3) = p1(2) + p2(2) + q(2)
      p12q(4) = p1(3) + p2(3) + q(3)

      v12 = ga(1)*gb(2) - ga(2)*gb(1)
      v13 = ga(1)*gb(3) - ga(3)*gb(1)
      v14 = ga(1)*gb(4) - ga(4)*gb(1)
      v23 = ga(2)*gb(3) - ga(3)*gb(2)
      v24 = ga(2)*gb(4) - ga(4)*gb(2)
      v34 = ga(3)*gb(4) - ga(4)*gb(3)

      jggs2(1) =   g2(2)*(   v23*p12q(4) - v24*p12q(3) + v34*p12q(2) )
      jggs2(2) = - g2(2)*( - v13*p12q(4) + v14*p12q(3) - v34*p12q(1) )
      jggs2(3) = - g2(2)*(   v12*p12q(4) - v14*p12q(2) + v24*p12q(1) )
      jggs2(4) = - g2(2)*( - v12*p12q(3) + v13*p12q(2) - v23*p12q(1) )
      endif


      jggs(1) = dv * (jggs1(1) + jggs2(1))
      jggs(2) = dv * (jggs1(2) + jggs2(2))
      jggs(3) = dv * (jggs1(3) + jggs2(3))
      jggs(4) = dv * (jggs1(4) + jggs2(4))


      return
      end
