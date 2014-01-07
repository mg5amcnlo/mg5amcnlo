
      subroutine HEP2GEANT
************************************************************************
*     
*     Put all undecayed STDHEP particles in GEANT KINE stack.
*
*     The first user word in the KINE structure is
*     the STDHEP record number for the particle.
*     This permits a GEANT/HEP cross reference.
*
*     Remember STDHEP vertices are in mm's
*     and GEANT vertices are in cm's.
*
*     Particles unknown to GEANT are entered as geantino's.
*     A warning message is also printed.
*
*     Doug Wright, LLNL   Nov 1993
*     wright20@llnl.gov
*
*     July 94  D.Wright, NWHINE added, GEANTINO fixed
*     Oct. 95  L.Garren, move particle id translation to gtran
*     Nov. 99  L.Garren, get more intelligent about vertices
*
************************************************************************

      implicit none


#include "stdhep.inc"
#include "stdlun.inc"

      real    ptemp(5)
      integer i,j,IDGEANT,Nvertex, Ntrack, IHEP
      integer gtran

c  reset the vertex list
      call stdvtxmatch(0,Nvertex)

      do i=1, NHEP
         if( ISTHEP(I) .eq. 1 ) then ! particle hasn't decayed
            IHEP = IDHEP(I)
            IDGEANT = gtran(IHEP,2)
            do j=1,5
               ptemp(j) = PHEP(j,i)
            enddo
	    call stdvtxmatch(i,Nvertex)
            call GSKINE(ptemp(1),IDGEANT,Nvertex,float(i),1,Ntrack)
         endif
      enddo
      return
      end
