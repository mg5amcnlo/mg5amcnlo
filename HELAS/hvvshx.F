      subroutine hvvshx(v1,v2,sc,g1,mass,width, hvvsh)
c
c- by RF - Mar. 2006
c
c
c This subroutine computes an amplitude of the vector-vector-Higgs-Higgs 
c effective coupling.
c
c input:
c       complex v1(6)          : first  vector                        
c       complex v2(6)          : second vector                        
c       complex sc1(3)         : first  scalar                        
c       complex g1(2)          : first coupling constant                 
c       real    mass           : mass of the outgoing scalar
c       real    width          : width of the outgoing scalar
c
c output:
c       complex hvvsh(3)       : scalar current
c     
      implicit none
c--   dimension of the current set to arbitrary length
      INTEGER DIM
      PARAMETER(DIM=18)
c      include "dimension.inc"
      double complex v1(DIM),v2(DIM),sc(DIM),hvvsh(DIM)
      double complex hvvsh1,hvvsh2,dg,g1(2)
      double complex v12,p2v1,p1v2,v13,v14,v23,v24,v34
      double precision p12,p13,p14,p23,p24,p34
      
      double precision p1(0:3),p2(0:3),mass,width,q2,q(0:3)


      p1(0) = dble( v1(5))
      p1(1) = dble( v1(6))
      p1(2) = dimag(v1(6))
      p1(3) = dimag(v1(5))

      p2(0) = dble( v2(5))
      p2(1) = dble( v2(6))
      p2(2) = dimag(v2(6))
      p2(3) = dimag(v2(5))

      hvvsh(2) = v1(5)+v2(5)+sc(2)
      hvvsh(3) = v1(6)+v2(6)+sc(3)

      q(0) = -dble( hvvsh(2))
      q(1) = -dble( hvvsh(3))
      q(2) = -dimag(hvvsh(3))
      q(3) = -dimag(hvvsh(2))

      q2 = q(0)**2 - q(1)**2 - q(2)**2 - q(3)**2

      dg = dcmplx( q2-mass**2, mass*width )

      hvvsh1 = (0D0,0D0)
      hvvsh2 = (0D0,0D0)

      if (g1(1).NE.(0D0,0D0)) then

         v12  = v1(1)*v2(1) - v1(2)*v2(2) - v1(3)*v2(3) - v1(4)*v2(4)
         p12  = p1(0)*p2(0) - p1(1)*p2(1) - p1(2)*p2(2) - p1(3)*p2(3)
         p2v1 = v1(1)*p2(0) - v1(2)*p2(1) - v1(3)*p2(2) - v1(4)*p2(3)
         p1v2 = p1(0)*v2(1) - p1(1)*v2(2) - p1(2)*v2(3) - p1(3)*v2(4)	

         hvvsh1 = - g1(1) *(v12*p12 - p2v1*p1v2)
      endif

      if (g1(2).NE.(0D0,0D0)) then
          p12 = p1(0)*p2(1) - p1(1)*p2(0)
          p13 = p1(0)*p2(2) - p1(2)*p2(0)
          p14 = p1(0)*p2(3) - p1(3)*p2(0)
          p23 = p1(1)*p2(2) - p1(2)*p2(1)
          p24 = p1(1)*p2(3) - p1(3)*p2(1)
          p34 = p1(2)*p2(3) - p1(3)*p2(2)

          v12 = v1(1)*v2(2) - v1(2)*v2(1)
          v13 = v1(1)*v2(3) - v1(3)*v2(1)
          v14 = v1(1)*v2(4) - v1(4)*v2(1)
          v23 = v1(2)*v2(3) - v1(3)*v2(2)
          v24 = v1(2)*v2(4) - v1(4)*v2(2)
          v34 = v1(3)*v2(4) - v1(4)*v2(3)

          hvvsh2 = g1(2)*( v12*p34 - v13*p24 + v14*p23
     &                    +v23*p14 - v24*p13 + v34*p12 )
      endif
       
      hvvsh(1) = sc(1)*(hvvsh1 + hvvsh2) /dg

      return
      end
