      subroutine stdvtxmatch(istd,nvtx)
************************************************************************
*
*    for use with HEP2GEANT
*    create a vertex list and find a match
*    call gsvert and keep a list of matching vertex numbers
*    return vertex number in list
*
*    Nov. 1999   Lynn Garren
*    Jan. 1999   Paul Lebrun - improvements to the logic
*
*  istd - index in HEPEVT common block
*         reset the list if istd=0
*  nvtx - index in vertex list
*
************************************************************************
    
      implicit none

#include "stdhep.inc"
#include "stdlun.inc"

      integer MAXVTX
      parameter (MAXVTX=150)
      real vtx(3)
      double precision vtx_list(4,MAXVTX)
      integer nvtx, nvtx_list(MAXVTX), lvtx, i, istd, mvtx

      save lvtx, nvtx_list, vtx_list

      nvtx = 0
c  reset the vertex list
      if( istd.eq.0 ) then
          lvtx = 0
	  return
      endif

c  set the initial list entry and call gsvert
      if( lvtx.eq.0 ) then
	 lvtx = 1
         do i=1,4
             vtx_list(i,lvtx) = vhep(i,istd)
	 enddo
	 do i=1,3
	    vtx(i) = vhep(i,istd)/10.
	 enddo
	 call gsvert(vtx,0,0,0,0,nvtx)
	 nvtx_list(lvtx) = nvtx
      endif

c
c search for matches 
c use differences rather than exact matches 
c
      mvtx = 0    
      do i=1,lvtx
          if(abs(vhep(1,istd) - vtx_list(1,i)) .lt. 0.0001 .and. 
     1       abs(vhep(2,istd) - vtx_list(2,i)) .lt. 0.0001 .and.
     2       abs(vhep(3,istd) - vtx_list(3,i)) .lt. 0.0005 .and.
     3       abs(vhep(4,istd) - vtx_list(4,i)) .lt. 0.0005 ) then
              mvtx = i
          endif
      enddo

c no match - add this vertex to the list and call gsvert
      if( mvtx.eq.0 ) then
          lvtx = lvtx + 1
	  if( lvtx .gt. MAXVTX ) then
	      write(lnhout,1001)
	      lvtx = 1
	  endif
	  mvtx = lvtx
	  do i=1,4
             vtx_list(i,lvtx) = vhep(i,istd)
	  enddo
	  do i=1,3
	     vtx(i) = vhep(i,istd)/10.
	  enddo
	  call gsvert(vtx,0,0,0,0,nvtx)
	  nvtx_list(lvtx) = nvtx
c found a match - just need the vertex number
      else
          nvtx = nvtx_list(mvtx)
      endif

      return
1001  format("STDVTXMATCH: too many vertices - overwriting the list")
      end
