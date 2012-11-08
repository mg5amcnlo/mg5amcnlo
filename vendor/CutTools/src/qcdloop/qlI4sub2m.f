      subroutine qlI4sub2m(xpi,musq,Ival)
      implicit none
C     Uses the ordering for the routine xpi wanted by FF
C     psq(1) lies between msq(1) and msq(2) and so on
C     xpi(1-4) = msq(1),msq(2),msq(3),msq(4)
C     xpi(5-8) = psq(1),psq(2),psq(3),psq(4)
C     xpi(9-10) = s12,s23
C     xpi(11) = +xpi(5)+xpi(6)+xpi(7)+xpi(8)-xpi(9)-xpi(10)
C     xpi(12) = -xpi(5)+xpi(6)-xpi(7)+xpi(8)+xpi(9)+xpi(10)
C     xpi(13) = +xpi(5)-xpi(6)+xpi(7)-xpi(8)+xpi(9)+xpi(10)
C     Calculates the general box with qlzero masses
      logical qlnonzero
      double precision xpi(13),musq,xpiout(13)
      double complex Ival(-2:0)
      integer j,jsort1,jsort2,massive,swap(13,5),jdiff,Npt
      parameter(Npt=13)
      data swap/
     .      4,1,2,3,8,5,6,7,10,9,11,12,13,
     .      3,4,1,2,7,8,5,6,9,10,11,12,13,
     .      2,3,4,1,6,7,8,5,10,9,11,12,13,
     .      1,2,3,4,5,6,7,8,9,10,11,12,13,
     .      3,2,1,4,6,5,8,7,9,10,11,12,13/

      call qlxpicheck(xpi)

      massive=0
      jsort1=0
      jsort2=0
      do j=1,4
      if (qlnonzero(xpi(j))) then
      massive=massive+1
      if (jsort1 .eq. 0) then
      jsort1=j
      else
      jsort2=j
      endif
      endif
      enddo
      if (massive .ne. 2) then
      write(6,*) 'Error in qlI4sum2m: not exactly two masses'
      write(6,*) 'xpi(1-4)',xpi(1),xpi(2),xpi(3),xpi(4)
      stop
      endif

      jdiff=jsort2-jsort1
       
      if ((jdiff .eq. 1) .or. (jdiff .eq. 2)) then
         do j=1,Npt
         xpiout(swap(j,jsort2))=xpi(j)
         enddo
      elseif ((jdiff .eq. 3)) then
         do j=1,Npt
         xpiout(swap(j,1))=xpi(j)
         enddo
      endif

      if (jdiff .eq. 2) then
      call qlI4sub2mo(xpiout,musq,Ival)
      else
      call qlI4sub2ma(xpiout,musq,Ival)
      endif
       
      return
      end
