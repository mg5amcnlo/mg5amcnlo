      double complex function qlI4(p1,p2,p3,p4,s12,s23,
     . m1,m2,m3,m4,mu2,ep)
      implicit none
      double precision p1,p2,p3,p4,s12,s23,m1,m2,m3,m4,mu2,newmu2
      integer ep,j
C     pi=p(i)^2, i=1,2,3,4 are momentum squared of the external lines
C     mi=m(i)^2  i=1,2,3,4 are masses squared of the internal lines
C     sij=(pi+pj)^2 are external invariants
C     mu2 is the square of the scale mu
C     ep=-2,-1,0 chooses the coefficient in the Laurent series.
      double precision xpi(13),scalefac
      double complex qlI4array

C     Uses the ordering for the routine xpi wanted by FF
C     psq(1) lies between msq(1) and msq(2) and so on
C     xpi(1-4) = msq(1),msq(2),msq(3),msq(4)
C     xpi(5-8) = psq(1),psq(2),psq(3),psq(4)
C     xpi(9-10) = s12,s23
C     xpi(11) = +xpi(5)+xpi(6)+xpi(7)+xpi(8)-xpi(9)-xpi(10)
C     xpi(12) = -xpi(5)+xpi(6)-xpi(7)+xpi(8)+xpi(9)+xpi(10)
C     xpi(13) = +xpi(5)-xpi(6)+xpi(7)-xpi(8)+xpi(9)+xpi(10)

      xpi(1)=m1
      xpi(2)=m2
      xpi(3)=m3
      xpi(4)=m4
      xpi(5)=p1
      xpi(6)=p2
      xpi(7)=p3
      xpi(8)=p4
      xpi(9)=s12
      xpi(10)=s23
      xpi(11)=+xpi(5)+xpi(6)+xpi(7)+xpi(8)-xpi(9)-xpi(10)
      xpi(12)=-xpi(5)+xpi(6)-xpi(7)+xpi(8)+xpi(9)+xpi(10)
      xpi(13)=+xpi(5)-xpi(6)+xpi(7)-xpi(8)+xpi(9)+xpi(10)
      scalefac=max(abs(s12),abs(s23),abs(p1),abs(p2),abs(p3),abs(p4))
      do j=1,13
      xpi(j)=xpi(j)/scalefac
      enddo
      newmu2=mu2/scalefac
      qlI4=qlI4array(xpi,newmu2,ep)/scalefac**2
      return
      end
