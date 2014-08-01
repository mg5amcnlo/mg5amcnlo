      subroutine qlI4sub0m(xpi,musq,Ival)
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
      double precision xpi(13),xpiout(13),musq,Y(4,4),Yalt(4,4)
      double complex Ival(-2:0)
      integer j,offshell,swap(13,4),jsort1,jsort2,jsort0,jdiff,Npt,
     . jsort(4)
      logical qlnonzero,swapped
      parameter(Npt=13)
      data swap/
     .      4,1,2,3,8,5,6,7,10,9,11,13,12,
     .      3,4,1,2,7,8,5,6,9,10,11,12,13,
     .      2,3,4,1,6,7,8,5,10,9,11,13,12,
     .      1,2,3,4,5,6,7,8,9,10,11,12,13/
      data jsort/4,1,2,3/
      save swap,jsort

      call qlxpicheck(xpi)

      do j=1,4
      if (qlnonzero(xpi(j))) then
      write(6,*) 'qlI4sub0m called in error:j,xpi(j)',j,xpi(j)
      stop
      endif
      enddo

      offshell=0
      jsort1=0
      jsort2=0
      do j=1,4
      if (qlnonzero(xpi(j+4))) then
      offshell=offshell+1
      if (jsort1 .eq. 0) then 
      jsort1=j
      else
      jsort2=j
      endif
      else
      jsort0=j
      endif

      enddo

      jdiff=jsort2-jsort1
      swapped=.true.
      if ((offshell .eq. 1)) then
         do j=1,Npt
         xpiout(swap(j,jsort1))=xpi(j)
         enddo
      elseif ((offshell .eq. 2) .and. (jdiff .eq. 2)) then
         do j=1,Npt
         xpiout(swap(j,jsort2))=xpi(j)
         enddo
      elseif ((offshell .eq. 2) .and. (jdiff .eq. 1)) then
         do j=1,Npt
         xpiout(swap(j,jsort2))=xpi(j)
         enddo
      elseif ((offshell .eq. 2) .and. (jdiff .eq. 3)) then
         do j=1,Npt
         xpiout(swap(j,1))=xpi(j)
         enddo
      elseif (offshell .eq. 3) then
         do j=1,Npt
         xpiout(swap(j,jsort(jsort0)))=xpi(j)
         enddo
      else
         swapped=.false.
      endif

C--if we performed a swap rename the array
      if (swapped) then
      do j=1,13
      xpi(j)=xpiout(j) 
      enddo
      endif

      call qlYcalc(xpi,Y,Yalt)      
C--------------four offshell external lines
      if (offshell .eq. 4) then
           Ival(-2)=czip
           Ival(-1)=czip
           call qlI4DNS41(Y,musq,Ival(0))
C--------------three offshell external lines
      elseif (offshell .eq. 3) then
           call qlbox5(Y,musq,Ival)
C--------------two offshell external lines
      elseif (offshell .eq. 2) then
           if ((qlnonzero(xpi(6))).and.(qlnonzero(xpi(6)))) then
c          opposite-easy
           call qlbox3(Y,musq,Ival)
c          adjacent-hard
           elseif ((qlnonzero(xpi(7))).and.(qlnonzero(xpi(8)))) then
           call qlbox4(Y,musq,Ival)
           endif
      elseif (offshell .eq. 1) then
           call qlbox2(Y,musq,Ival)
      elseif (offshell .eq. 0) then
           call qlbox1(Y,musq,Ival)
      endif
      return
      end
