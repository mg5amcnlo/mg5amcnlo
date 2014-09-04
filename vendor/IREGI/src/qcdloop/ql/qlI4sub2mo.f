      subroutine qlI4sub2mo(xpi,musq,Ival)
      implicit none
C---  Calculates the box with two opposite internal masses

C     Uses the ordering for the routine xpi wanted by FF
C     psq(1) lies between msq(1) and msq(2) and so on
C     xpi(1-4) = msq(1),msq(2),msq(3),msq(4)
C     xpi(5-8) = psq(1),psq(2),psq(3),psq(4)
C     xpi(9-10) = s12,s23
C     xpi(11) = +xpi(5)+xpi(6)+xpi(7)+xpi(8)-xpi(9)-xpi(10)
C     xpi(12) = -xpi(5)+xpi(6)-xpi(7)+xpi(8)+xpi(9)+xpi(10)
C     xpi(13) = +xpi(5)-xpi(6)+xpi(7)-xpi(8)+xpi(9)+xpi(10)

      include 'qlconstants.f'
      double precision xpi(13),musq,Y(4,4),Yalt(4,4)
      double complex Ival(-2:0)
      logical qlnonzero,qlzero
      integer ier
  
      If (qlnonzero(xpi(1)) .or. qlnonzero(xpi(3))) then
      write(6,*)'Error in qlI4sub2mo,qlnonzero m1sq/m3sq',xpi(1),xpi(3)
      stop
      endif

      call qlYcalc(xpi,Y,Yalt) 
C----case 14
      if    ((qlzero(Y(1,1))) 
     . .and. (qlzero(Y(3,3)))
     . .and. (qlzero(Y(1,2)))
     . .and. (qlzero(Y(1,4)))
     . .and. (qlzero(Y(2,3)))
     . .and. (qlzero(Y(3,4)))) then
      call qlbox14(Y,musq,Ival)

C----case 15
      elseif((qlzero(Y(1,1))) 
     . .and. (qlzero(Y(3,3)))
     . .and. (qlzero(Y(1,2)))
     . .and. (qlzero(Y(1,4))))then
      call qlbox15(Y,musq,Ival)

      elseif((qlzero(Y(1,1))) 
     . .and. (qlzero(Y(3,3)))
     . .and. (qlzero(Y(2,3)))
     . .and. (qlzero(Y(3,4))))then
      call qlbox15(Yalt,musq,Ival)

      else
      Ival(-2)=czip
      Ival(-1)=czip
      xpi(11) = +xpi(5)+xpi(6)+xpi(7)+xpi(8)-xpi(9)-xpi(10)
      xpi(12) = -xpi(5)+xpi(6)-xpi(7)+xpi(8)+xpi(9)+xpi(10)
      xpi(13) = +xpi(5)-xpi(6)+xpi(7)-xpi(8)+xpi(9)+xpi(10)
      call qlI4fin(Ival(0),xpi,ier)
      endif

      return
      end
