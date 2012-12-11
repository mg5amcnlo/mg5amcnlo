c
c	stdHepFortranUtil.F : 
c		Bunch of small utility functions to communicate with the 
c		Fortran StdHep library.
c		The assumption is that only 1 file is open at a time.
c
	subroutine closestdhep
#include "stdlun.inc" 
#include "stdheputil.inc"
        COMMON /QUEST/ IQUEST(100)
	if (file_type .eq. 1) then
	  call fzendi(LNHRD,'I')
	elseif (file_type .eq. 2) then
          write(lnhout,*) ' ERROR - should not be in calling stdxend '
	else
 	  close(lnhrd)
 	end if
 	return
 	end 
c
c	These are dummy routine to avoid linking the entire GEANT pacakge.
c
c
	subroutine gsvert(vert, ntbeam, ntt, ub, nu, nv)
	dimension vert(*)
	return
	end
	subroutine gskine(vert, ntbeam, ntt, ub, nu, nv)
	dimension vert(*)
	return
	end
	
	
