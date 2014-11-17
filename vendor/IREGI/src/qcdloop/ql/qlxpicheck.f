      subroutine qlxpicheck(xpi)
      implicit none
      double precision xpi(13),y13,y24
      logical qlzero
C     Uses the ordering for the routine xpi wanted by FF
C     psq(1) lies between msq(1) and msq(2) and so on
C     xpi(1-4) = msq(1),msq(2),msq(3),msq(4)
C     xpi(5-8) = psq(1),psq(2),psq(3),psq(4)
C     xpi(9-10) = s12,s23
C     xpi(11) = +xpi(5)+xpi(6)+xpi(7)+xpi(8)-xpi(9)-xpi(10)
C     xpi(12) = -xpi(5)+xpi(6)-xpi(7)+xpi(8)+xpi(9)+xpi(10)
C     xpi(13) = +xpi(5)-xpi(6)+xpi(7)-xpi(8)+xpi(9)+xpi(10)
c      y13=abs(m1+m3s-s12)       
c      y24=abs(m2+m4s-s23)       
      y13=xpi(1)+xpi(3)-xpi(9)
      y24=xpi(2)+xpi(4)-xpi(10)
      if (qlzero(y13) .or. qlzero(y24)) then
      write(6,*) 'Modified Cayley elements y13 or y24=0',y13,y24
      stop
      endif
      end
