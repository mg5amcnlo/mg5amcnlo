      subroutine zerohepeup

C...Purpose: to zero the HEPEUP common block.
C
      implicit none

#include "hepeup.inc"

      integer i,j

C...Zero the common block
      NUP = 0
      IDPRUP = 0
      XWGTUP = 0
      SCALUP = 0
      AQEDUP = 0
      AQCDUP = 0
      do I=1,MAXNUP
          idup(i) = 0
	  istup(i) = 0
	  do j=1,2
	      mothup(j,i) = 0
	      icolup(j,i) = 0
	  enddo
          SPINUP(I)=0D0
          VTIMUP(I)=0D0
	  do j=1,5
	      pup(j,i) = 0D0
	  enddo
      enddo

      return
      end
