      subroutine readmadgraph(version,lok)

C...Double precision and integer declarations.
      implicit none
      
      real version

#include "hepeup.inc"
#include "stdlun.inc"

      real*8 test
      integer i,j,indx,id,lok
      character*80 cline


C...Zero the common block
      call zerohepeup
      lok = 0
      cline = '                                                       '

c...read the ascii file
c...must parse the first line to get nup,id,xwgtup
      read(lnhpdf,'(A)',end=888) cline
      call parseline1(cline,id)

      read(lnhpdf,'(11i7)',end=888) (idup(i),i=1,nup)
      istup(1)=-1
      istup(2)=-1
      do i=3,nup
         istup(i)=1
      enddo
      read(lnhpdf,'(11i7)',end=888) (mothup(1,i),i=1,nup)
      read(lnhpdf,'(11i7)',end=888) (mothup(2,i),i=1,nup)
      read(lnhpdf,'(11i7)',end=888) (icolup(1,i),i=1,nup)
      read(lnhpdf,'(11i7)',end=888) (icolup(2,i),i=1,nup)

      do i=1,nup
	  read(lnhpdf,'(i3,4e19.12)',end=888) indx,
     1                                        pup(4,i),(pup(j,i),j=1,3)
          if( indx .ne. i ) write(lnhout,1001) indx,i
	  test=pup(1,i)**2+pup(2,i)**2+pup(3,i)**2-pup(4,i)**2
	  if(test.le.0d0) then
              pup(5,i)=dsqrt(-test)
	  elseif(dabs(test).lt.1d-4) then
              pup(5,i)=0d0
	  else
              pup(5,i)=-1d0
              write(lnhout,*) ' NEGATIVE MASS '
	  endif
      enddo

C...default value of scalup
      scalup=-1d0

      return

 888  continue
      nup=0
      lok=-1

      return
 1001 format(' readmadgraph WARNING: madgraph index ',i6,
     1       ' does not match loop index ',i6)
      end

      subroutine parseline1(cline,id)

      implicit none

#include "hepeup.inc"

      character*(*) cline
      integer id,i1,i2,is

      is=1
      call findword(is,cline,i1,i2)
      read(cline(i1:i2-1),'(i4)') nup

      is=i2
      call findword(is,cline,i1,i2)
      read(cline(i1:i2-1),'(i6)') id

      is=i2
      call findword(is,cline,i1,i2)
      read(cline(i1:i2-1),'(E16.8)') xwgtup

      return
      end

      subroutine findword(istart,cline,ibeg,iend)

c  use blanks to identify a word

      implicit none

      character*(*) cline
      integer istart,ibeg,iend
      character*1 blank
      data blank/' '/

      ibeg=istart
      do while( cline(ibeg:ibeg).eq.blank )
          ibeg = ibeg + 1
      enddo
      iend=ibeg
      do while( cline(iend:iend).ne.blank )
          iend=iend+1
      enddo

      return
      end
