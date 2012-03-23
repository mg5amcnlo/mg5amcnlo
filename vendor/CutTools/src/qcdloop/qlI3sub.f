      subroutine qlI3sub(msq,psq,musq,Ival)
C Calculates the general scalar triangle
C assumes that the msq(i) are positive and ordered
C so that msq(1) .le. msq(2) .le. msq(3)
      implicit none
      include 'qlconstants.f'
      double precision msq(3),psq(3),musq,Y(3,3),xpi(6)
      double complex Ival(-2:0)
      integer j,massive,ier
      logical qlzero, qlnonzero 

      do j=1,3
      xpi(j)=msq(j)
      xpi(j+3)=psq(j)
      Y(j,j)=msq(j) 
      enddo
      Y(1,2)=0.5d0*(msq(1)+msq(2)-psq(1))
      Y(1,3)=0.5d0*(msq(1)+msq(3)-psq(3))
      Y(2,3)=0.5d0*(msq(2)+msq(3)-psq(2))
      Y(2,1)=Y(1,2)
      Y(3,1)=Y(1,3)
      Y(3,2)=Y(2,3)

      massive=0
      do j=1,3
      if (qlnonzero(abs(msq(j)))) massive=massive+1
      enddo

      Ival(-2)=czip
      Ival(-1)=czip

C--------------three internal masses
      if (massive .eq. 3) then
               call qlI3fin(Ival(0),xpi,ier)

C--------------two internal masses
      elseif (massive .eq. 2) then
           if   (qlzero(abs(Y(1,2))) 
     .     .and. qlzero(abs(Y(1,3)))) then  
                call qltri6(psq(2),msq(2),msq(3),musq,Ival)
            else
                call qlI3fin(Ival(0),xpi,ier)
            endif

C--------------one internal masses
      elseif (massive .eq. 1) then
           if       (qlnonzero(abs(Y(1,2)))) then
                call qlI3fin(Ival(0),xpi,ier)
           else
                if (qlzero(abs(Y(1,3))) .and.
     .         qlzero(abs(Y(2,3)))) then
                      call qltri5(msq(3),musq,Ival)
                elseif (qlzero(abs(Y(1,3)))) then
                      call qltri4(psq(2),msq(3),musq,Ival)
                elseif (qlzero(abs(Y(2,3)))) then
                      call qltri4(psq(3),msq(3),musq,Ival)
                else
                      call qltri3(psq(2),psq(3),msq(3),musq,Ival)
                endif
           endif
C--------------qlzero internal masses
      elseif (massive .eq. 0) then
           if     (qlzero(abs(Y(1,2))) .and.
     .    qlzero(abs(Y(2,3)))) then
                  call qltri1(psq(3),musq,Ival)
           elseif  (qlzero(abs(Y(1,2)))) then
                 call qltri2(psq(2),psq(3),musq,Ival)
           else 
                 call qlI3fin(Ival(0),xpi,ier)
           endif
      endif
      return
      end



      
