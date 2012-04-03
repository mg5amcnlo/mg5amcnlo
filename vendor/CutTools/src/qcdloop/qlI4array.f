      Double complex function qlI4array(xpi,musq,ep)
      implicit none
      include 'qlconstants.f'
      logical qlnonzero
      double precision xpi(13),xpio(13),musq,musqo
      integer ep,j,massive,ier,Npt
      double complex Ival(-2:0)
      parameter(Npt=10)
      data xpio/13*0d0/
      save Ival,xpio,musqo

C--If we have already calculated this qlI4, use the saved value
C--else setup the arrays
      if    ((xpi(1) .eq. xpio(1))
     . .and. (xpi(2) .eq. xpio(2))
     . .and. (xpi(3) .eq. xpio(3))
     . .and. (xpi(4) .eq. xpio(4))
     . .and. (xpi(5) .eq. xpio(5))
     . .and. (xpi(6) .eq. xpio(6))
     . .and. (xpi(7) .eq. xpio(7))
     . .and. (xpi(8) .eq. xpio(8))
     . .and. (xpi(9) .eq. xpio(9))
     . .and. (xpi(10) .eq. xpio(10))
     . .and. (musq .eq. musqo)) then
      qlI4array=Ival(ep)
      return
      else
C--   save new array as old
      do j=1,Npt
      xpio(j)=xpi(j)
      enddo
      musqo=musq

C---  count number of internal masses
      massive=0
      do j=1,4
      if (qlnonzero(xpi(j))) massive=massive+1
      enddo

      if (massive .eq. 0) then 
      call qlI4sub0m(xpi,musq,Ival)
      elseif (massive .eq. 1) then 
      call qlI4sub1m(xpi,musq,Ival)
      elseif (massive .eq. 2) then 
      call qlI4sub2m(xpi,musq,Ival)
      elseif (massive .eq. 3) then 
      call qlI4sub3m(xpi,musq,Ival)
      elseif (massive .eq. 4) then 
      xpi(11) = +xpi(5)+xpi(6)+xpi(7)+xpi(8)-xpi(9)-xpi(10)
      xpi(12) = -xpi(5)+xpi(6)-xpi(7)+xpi(8)+xpi(9)+xpi(10)
      xpi(13) = +xpi(5)-xpi(6)+xpi(7)-xpi(8)+xpi(9)+xpi(10)
      Ival(-2)=czip
      Ival(-1)=czip
      ier = 0
      call qlI4fin(Ival(0),xpi,ier)
      endif
      endif
      qlI4array=Ival(ep)
      return
      end
