      subroutine qlI4sub1m(xpi,musq,Ival)
      implicit none
C     Uses the ordering for the routine xpi wanted by FF
C     psq(1) lies between msq(1) and msq(2) and so on
C     xpi(1-4) = msq(1),msq(2),msq(3),msq(4)
C     xpi(5-8) = psq(1),psq(2),psq(3),psq(4)
C     xpi(9-10) = s12,s23
C     xpi(11) = +xpi(5)+xpi(6)+xpi(7)+xpi(8)-xpi(9)-xpi(10)
C     xpi(12) = -xpi(5)+xpi(6)-xpi(7)+xpi(8)+xpi(9)+xpi(10)
C     xpi(13) = +xpi(5)-xpi(6)+xpi(7)-xpi(8)+xpi(9)+xpi(10)
C Calculates the general box with qlzero masses
      include 'qlconstants.f'
      logical qlnonzero,qlzero
      double precision xpi(13),musq,xpiout(13),Y(4,4),Yalt(4,4)
      double complex Ival(-2:0)
      integer j,jsort,massive,swap(13,5),ier,Npt
      parameter(Npt=13)
      data swap/
     .      4,1,2,3,8,5,6,7,10,9,11,13,12,
     .      3,4,1,2,7,8,5,6,9,10,11,12,13,
     .      2,3,4,1,6,7,8,5,10,9,11,13,12,
     .      1,2,3,4,5,6,7,8,9,10,11,12,13,
     .      3,2,1,4,6,5,8,7,9,10,11,12,13/

      call qlxpicheck(xpi)

      massive=0
      do j=1,4
      if (qlnonzero(xpi(j))) then
      massive=massive+1
      jsort=j
      endif
      enddo

      do j=1,Npt
      xpiout(swap(j,jsort))=xpi(j)
      enddo

      do j=1,Npt
      xpi(j)=xpiout(j) 
      enddo

      call qlYcalc(xpiout,Y,Yalt)
      if ((qlnonzero(Y(1,1))).or.(qlnonzero(Y(2,2)))
     . .or.(qlnonzero(Y(3,3)))) then
      write(6,*) 'qlI4sub1m fails: wrong ordering'
      stop
      endif

      if   ((qlzero(Y(1,2)))
     . .and.(qlzero(Y(2,3)))
     . .and.(qlzero(Y(3,4)))
     . .and.(qlzero(Y(4,1)))) then
C     box6  $I_4(0,0,m^2,m^2;s_{12},s_{23};0,0,0,m^2)$}
      call qlbox6(Y,musq,Ival)

      elseif((qlzero(Y(1,2)))
     .  .and.(qlzero(Y(2,3)))
     .  .and.(qlzero(Y(3,4)))) then
C     box7  $I_4(0,0,m^2,\pq^2;s_{12},s_{23};0,0,0,m^2)$}
      call qlbox7(Y,musq,Ival)

      elseif((qlzero(Y(1,2)))
     .  .and.(qlzero(Y(2,3)))
     .  .and.(qlzero(Y(1,4)))) then
C     box7a  $I_4(0,0,\pt^2,m^2;s_{12},s_{23};0,0,0,m^2)$}
      call qlbox7(Yalt,musq,Ival)

      elseif((qlzero(Y(1,2)))
     .  .and.(qlzero(Y(2,3)))) then
C     box8  $I_4(0,0,\pt^2,\pq^2; s_{12},s_{23};0,0,0,m^2)$}
      call qlbox8(Y,musq,Ival)

      elseif((qlzero(Y(1,2)))
     .  .and.(qlzero(Y(1,4)))) then
C     box9  $I_4(0,p_2^2,p_3^2,m^2;s_{12},s_{23};0,0,0,m^2)$}
         call qlbox9(Y,musq,Ival)

      elseif((qlzero(Y(2,3)))
     .  .and.(qlzero(Y(3,4)))) then
C     box9a  $I_4(0,p_2^2,p_3^2,m^2;s_{12},s_{23};0,0,0,m^2)$}
         call qlbox9(Yalt,musq,Ival)

      elseif (qlzero(Y(1,2))) then
C     box10 $I_4(0,p_2^2,p_3^2,p_4^2;s_{12},s_{23};0,0,0,m^2)$}
         call qlbox10(Y,musq,Ival)

C     box10 $I_4(p_1^2,0,p_3^2,p_4^2;s_{12},s_{23};0,0,0,m^2)$}
      elseif (qlzero(Y(2,3))) then
         call qlbox10(Yalt,musq,Ival)
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
