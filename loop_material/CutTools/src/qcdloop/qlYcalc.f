      subroutine qlYcalc(xpi,Y,Yalt)
C----Calculate Y from xpi
C----Assumes, if we have 1 internal mass it is position 4
C----Assumes, if we have 2 internal masses they are positions 3,4 or 2,4
      implicit none
      include 'qlconstants.f' 
      double precision xpi(13),Y(4,4),Yalt(4,4)
      logical opposite,qlnonzero,qlzero
      integer j,k,massive
      massive=0
      do j=1,4
      if (qlnonzero(xpi(j))) massive=massive+1
      do k=1,4
      Yalt(j,k)=0d0
      enddo
      enddo
      if (qlzero(xpi(3))) then
      opposite=.true.
      else
      opposite=.false.
      endif
      
      Y(1,1)=xpi(1)
      Y(2,2)=xpi(2)
      Y(3,3)=xpi(3)
      Y(4,4)=xpi(4)
      Y(1,2)=half*(xpi(1)+xpi(2)-xpi(5))
      Y(1,3)=half*(xpi(1)+xpi(3)-xpi(9))
      Y(1,4)=half*(xpi(1)+xpi(4)-xpi(8))
      Y(2,1)=Y(1,2)
      Y(2,3)=half*(xpi(2)+xpi(3)-xpi(6))
      Y(2,4)=half*(xpi(2)+xpi(4)-xpi(10))
      Y(3,1)=Y(1,3)
      Y(3,2)=Y(2,3)
      Y(3,4)=half*(xpi(3)+xpi(4)-xpi(7))
      Y(4,1)=Y(1,4)
      Y(4,2)=Y(2,4)
      Y(4,3)=Y(3,4)

      if (massive .eq. 0) then
      Yalt(1,1)=zip
      Yalt(2,2)=zip
      Yalt(3,3)=zip
      Yalt(4,4)=zip
      Yalt(1,2)=zip
      Yalt(1,3)=zip
      Yalt(1,4)=zip
      Yalt(2,3)=zip
      Yalt(2,4)=zip
      Yalt(3,4)=zip
      elseif (massive .eq. 1) then
C---exchange (1<-->3)
      Yalt(1,1)=Y(3,3)
      Yalt(2,2)=Y(2,2)
      Yalt(3,3)=Y(1,1)
      Yalt(4,4)=Y(4,4)
      Yalt(1,2)=Y(2,3)
      Yalt(1,3)=Y(1,3)
      Yalt(1,4)=Y(3,4)
      Yalt(2,3)=Y(1,2)
      Yalt(2,4)=Y(2,4)
      Yalt(3,4)=Y(1,4)
      elseif ((massive .eq. 2) .and. (opposite .eqv. .true.)) then
C---exchange (2<-->4) .and (1<-->3)
      Yalt(1,1)=Y(3,3)
      Yalt(2,2)=Y(4,4)
      Yalt(3,3)=Y(1,1)
      Yalt(4,4)=Y(2,2)
      Yalt(1,2)=Y(3,4)
      Yalt(1,3)=Y(1,3)
      Yalt(1,4)=Y(2,3)
      Yalt(2,3)=Y(1,4)
      Yalt(2,4)=Y(2,4)
      Yalt(3,4)=Y(1,2)

      elseif ((massive .eq. 2) .and. (opposite .eqv. .false.)) then
C---exchange (1<-->2)and(3<-->4)
      Yalt(1,1)=Y(2,2)
      Yalt(2,2)=Y(1,1)
      Yalt(3,3)=Y(4,4)
      Yalt(4,4)=Y(3,3)
      Yalt(1,2)=Y(1,2)
      Yalt(3,4)=Y(3,4)
      Yalt(1,3)=Y(2,4)
      Yalt(1,4)=Y(2,3)
      Yalt(2,3)=Y(1,4)
      Yalt(2,4)=Y(1,3)
      endif
        
C----symmetrize Y
      do j=1,4
      do k=j+1,4
      Y(k,j)=Y(j,k)
      Yalt(k,j)=Yalt(j,k)
      enddo
      enddo

      return
      end
