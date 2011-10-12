      subroutine jvshxx(vc,sc,gc,xm,xw , jvs)
c
c- by RF - Mar. 2006
c
c
c This subroutine computes an off-shell vector current from the vector-
c vector-(pseudo-)scalar effective coupling.  
c input:
c       complex vc(6)          : input vector
c       complex sc(3)          : input scalar
c       real xm                : mass outgoing vector
c       real xw                : width outgoing vector
c       complex gc(2)          : coupling constant: gc(1) scalar
c                                                   gc(2) pseudo-scalar
c
c output:
c       complex jvs(6)         : outgoing vector current
c
c
      implicit none
c--   dimension of the current set to arbitrary length
      INTEGER DIM
      PARAMETER(DIM=18)
c      include "dimension.inc"
      double complex vc(DIM),sc(DIM),jvs(DIM),jvs1(DIM),jvs2(DIM)
      double complex qvc,dg
      double precision qp1,p12,p13,p14,p23,p24,p34
      double precision p1(4),q(4),xm,xw,q2
      double complex gc(2)

      jvs(5) = vc(5)+sc(2)
      jvs(6) = vc(6)+sc(3)

      p1(1) = dble( vc(5))
      p1(2) = dble( vc(6))
      p1(3) = dimag(vc(6))
      p1(4) = dimag(vc(5))

      q(1) = -dble( jvs(5))
      q(2) = -dble( jvs(6))
      q(3) = -dimag(jvs(6))
      q(4) = -dimag(jvs(5))

      q2 = q(1)**2 - q(2)**2 - q(3)**2 - q(4)**2

      jvs1(1) = (0D0,0D0)
      jvs1(2) = (0D0,0D0)
      jvs1(3) = (0D0,0D0)
      jvs1(4) = (0D0,0D0)
      jvs2(1) = (0D0,0D0)
      jvs2(2) = (0D0,0D0)
      jvs2(3) = (0D0,0D0)
      jvs2(4) = (0D0,0D0)

      if(xm.eq.0d0)then
         dg = sc(1) /q2
      else
         dg = sc(1)/ dcmplx( q2-xm**2 , xm*xw )
      endif

      if (gc(1).NE.(0D0,0D0)) then
         qvc = vc(1)*q(1) - vc(2)*q(2) - vc(3)*q(3) - vc(4)*q(4)	
         qp1 = q(1)*p1(1) - q(2)*p1(2) - q(3)*p1(3) - q(4)*p1(4)	
         
         jvs1(1) = -gc(1)* (vc(1)*qp1 - p1(1)*qvc)
         jvs1(2) = -gc(1)* (vc(2)*qp1 - p1(2)*qvc)
         jvs1(3) = -gc(1)* (vc(3)*qp1 - p1(3)*qvc)
         jvs1(4) = -gc(1)* (vc(4)*qp1 - p1(4)*qvc)
      endif
 
      if (gc(2).NE.(0D0,0D0)) then
         p12 = p1(1)*q(2) - p1(2)*q(1)
         p13 = p1(1)*q(3) - p1(3)*q(1)
         p14 = p1(1)*q(4) - p1(4)*q(1)
         p23 = p1(2)*q(3) - p1(3)*q(2)
         p24 = p1(2)*q(4) - p1(4)*q(2)
         p34 = p1(3)*q(4) - p1(4)*q(3)


         jvs2(1)=  gc(2)* (           -vc(2)*p34 +vc(3)*p24 -vc(4)*p23)
         jvs2(2)= -gc(2)* ( vc(1)*p34            -vc(3)*p14 +vc(4)*p13)
         jvs2(3)= -gc(2)* (-vc(1)*p24 +vc(2)*p14            -vc(4)*p12)
         jvs2(4)= -gc(2)* ( vc(1)*p23 -vc(2)*p13 +vc(3)*p12           )
      endif


      jvs(1) = dg * (jvs1(1) + jvs2(1))
      jvs(2) = dg * (jvs1(2) + jvs2(2))
      jvs(3) = dg * (jvs1(3) + jvs2(3))
      jvs(4) = dg * (jvs1(4) + jvs2(4))

      return
      end
