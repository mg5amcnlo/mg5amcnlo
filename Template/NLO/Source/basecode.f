      subroutine basecode_test
      implicit none
      integer imax
      parameter (imax = 8)
      integer icode,iarray(imax),ibase,i,j
      logical done

      ibase = 3
c      do i=0,ibase**3-1
c         call decode(i,iarray,ibase,imax)
c         call encode(icode,iarray,ibase,imax)
c         write(*,*) i,icode,"=",(iarray(j),j=1,imax)
c      enddo
      icode = 0
      call decode(icode,iarray,ibase,imax)
      iarray(2)=1
      iarray(4)=1
      iarray(5)=1
      iarray(7)=1
      done = .false.
      write(*,*) (iarray(j),j=1,imax)
      do while (.not. done)
         write(*,*) (iarray(j),j=1,imax)
         call increment_array(iarray,imax,ibase,done)
      enddo
      end
      

      subroutine EnCode(icode,iarray,ibase,imax)
c******************************************************************************
c     Turns array of integers (iarray) values range (0,ibase-1) into a single
c     integer icode. icode = Sum[ iarray(k) * ibase^k]
c******************************************************************************
      implicit none
c
c     Arguments
c
      integer imax           !Number of integers to encode
      integer icode          !Output encoded value of iarray
      integer iarray(imax)   !Input values to be encoded
      integer ibase          !Base for encoding

c
c     Local
c     
      integer i
c-----
c  Begin Code
c-----
      icode = 0
      do i = 1, imax
         if (iarray(i) .ge. 0 .and. iarray(i) .lt. ibase) then 
            icode = icode + iarray(i)*ibase**(i-1)
         else
            write(*,*) 'Error invalid number to be encoded',i,iarray(i)
         endif
      enddo
      end

      subroutine DeCode(icode,iarray,ibase,imax)
c******************************************************************************
c     Decodes icode, into base integers used to create it.
c     integer icode. icode = Sum[ iarray(k) * ibase^k]
c******************************************************************************
      implicit none
c
c     Arguments
c
      integer imax           !Number of integers to encode
      integer icode          !Input encoded value of iarray
      integer iarray(imax)   !Output decoded values icode
      integer ibase          !Base for encoding

c
c     Local
c     
      integer i, jcode
c-----
c  Begin Code
c-----
      jcode = icode          !create copy for use
      do i =  imax, 1, -1
         iarray(i) = 0
         do while (jcode .ge. ibase**(i-1) .and. iarray(i) .lt. ibase)
            jcode = jcode-ibase**(i-1)
            iarray(i)=iarray(i)+1
         enddo
      enddo
      end

      subroutine increment_array(iarray,imax,ibase,done)
c************************************************************************
c     Increments iarray     
c************************************************************************
      implicit none
c
c     Arguments
c
      integer imax          !Input, number of elements in iarray
      integer ibase         !Base for incrementing, 0 is skipped
      integer iarray(imax)  !Output:Array of values being incremented
      logical done          !Output:Set when no more incrementing
c
c     Local
c
      integer i,j
      logical found
c-----
c  Begin Code
c-----
      found = .false.
      i = 1
      do while (i .le. imax .and. .not. found)
         if (iarray(i) .eq. 0) then    !don't increment this
            i=i+1
         elseif (iarray(i) .lt. ibase-1) then
            found = .true.
            iarray(i)=iarray(i)+1
         else
            iarray(i)=1
            i=i+1
         endif
      enddo
      done = .not. found
      end


