      subroutine dpmhep(mconv)

C...Convert DPMJET event record to and from the HEPEVT common block
C...convert (mconv=1) from DPMJET numbering scheme to StdHep numbering scheme
C...     or (mconv=2) from StdHep numbering scheme to DPMJET numbering scheme

C at this time, the particle ID numbers are assumed to be correct.

      IMPLICIT NONE

#include "stdhep.inc"
#include "stdlun.inc"
#include "hkkevt.inc"

      integer dpmtran
      external dpmtran

      integer i,j,l,mconv

      logical lfirst
      data lfirst/.true./
      save lfirst

C...print version number if this is the first call
      if(lfirst)then
        call stdversn
        lfirst=.false.
      endif
      if(mconv.EQ.1) then
C...convert from DPMJET to HEPEVT
        nhep = nhkk
	nevhep = nevhkk
	if(nhep.le.nmxhep)then
	  do i=1,nhep
	     isthep(i) = isthkk(i)
	     idhep(i) = dpmtran(idhkk(i),mconv)
	     do j=1,2
	       jmohep(j,i) = jmohkk(j,i)
	       jdahep(j,i) = jdahkk(j,i)
	     enddo
	     do j=1,5
	       phep(j,i) = phkk(j,i)
	     enddo
	     do j=1,4
	       vhep(j,i) = vhkk(j,i)
	     enddo
	  enddo
	else
	  write(lnhout,1001) nhkk,nmxhep
	  i = 0
	  do l=1,nhkk
	    if(isthkk(l).eq.1 .or. isthkk(l).eq.-1
     1          .or. isthkk(l).eq.1001)then
               i = i+1
	       if(i.gt.nmxhep) then
	          write(lnhout,1004)
	          return
	       endif
	       isthep(i) = isthkk(i)
	       idhep(i) = dpmtran(idhkk(i),mconv)
	       do j=1,2
		 jmohep(j,i) = jmohkk(j,i)
		 jdahep(j,i) = jdahkk(j,i)
	       enddo
	       do j=1,5
		 phep(j,i) = phkk(j,i)
	       enddo
	       do j=1,4
		 vhep(j,i) = vhkk(j,i)
	       enddo
	    endif
	  enddo
	endif
      elseif(mconv.EQ.2) then
C...convert from HEPEVT to DPMJET
        write(lnhout,1002)
      else
C...unsupported option
        write(lnhout,1003)
      endif
      return
1001  format(' DPMHEP: too many particles (',I4,
     1       ') for hepevt common block'/
     2       ' will only save isthkk = 1, -1, and 1001')

1002  format(' DPMHEP: conversion back to DPM not yet enabled')
1003  format(' DPMHEP: unallowed conversion option')
1004  format(' DPMHEP: still too many particles - truncating event')
      end
